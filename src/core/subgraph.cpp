/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/subgraph.hpp>

#include <fmt/format.h>

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <holoscan/core/arg.hpp>
#include <holoscan/core/config.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/io_spec.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/logger/logger.hpp>

namespace {

// Helper to format a port map's keys as a comma-separated list
template <typename PortMapT>
std::string format_port_list(const PortMapT& ports) {
  std::string result;
  bool first = true;
  for (const auto& [name, _] : ports) {
    if (!first) {
      result += ", ";
    }
    result += name;
    first = false;
  }
  return result;
}

const holoscan::InterfacePort& get_data_interface_port_or_throw(const holoscan::Subgraph& subgraph,
                                                                const std::string& interface_port,
                                                                bool expect_input) {
  const auto& data_ports = subgraph.interface_ports();
  auto data_it = data_ports.find(interface_port);
  if (data_it == data_ports.end()) {
    const auto& exec_ports = subgraph.exec_interface_ports();
    auto exec_it = exec_ports.find(interface_port);
    if (exec_it != exec_ports.end()) {
      throw std::runtime_error(
          fmt::format("Subgraph '{}': interface port '{}' is an execution port, not a data port",
                      subgraph.name(),
                      interface_port));
    }
    throw std::runtime_error(
        fmt::format("Subgraph '{}': unknown interface port '{}'", subgraph.name(), interface_port));
  }

  if (data_it->second.is_input != expect_input) {
    throw std::runtime_error(fmt::format("Subgraph '{}': interface port '{}' is an {} port",
                                         subgraph.name(),
                                         interface_port,
                                         data_it->second.is_input ? "input" : "output"));
  }

  if (data_it->second.empty()) {
    throw std::runtime_error(
        fmt::format("Subgraph '{}': interface port '{}' has no internal mappings",
                    subgraph.name(),
                    interface_port));
  }

  return data_it->second;
}

void bind_interface_port_topic(const holoscan::Subgraph& subgraph,
                               const std::string& interface_port, const std::string& topic,
                               const std::optional<nvidia::gxf::QoSProfile>& qos, bool expect_input,
                               bool replace_connector) {
  const auto& resolved_port =
      get_data_interface_port_or_throw(subgraph, interface_port, expect_input);
  for (const auto& mapping : resolved_port.mappings) {
    if (!mapping.internal_operator) {
      throw std::runtime_error(
          fmt::format("Subgraph '{}': interface port '{}' has a null internal operator mapping",
                      subgraph.name(),
                      interface_port));
    }
    if (expect_input) {
      mapping.internal_operator->bind_input_topic(
          mapping.internal_port_name, topic, qos, replace_connector);
    } else {
      mapping.internal_operator->bind_output_topic(
          mapping.internal_port_name, topic, qos, replace_connector);
    }
  }
}

}  // namespace

namespace holoscan {

Subgraph::Subgraph(Fragment* fragment, const std::string& name, const std::string& config_file)
    : fragment_(fragment), name_(name) {
  if (fragment == nullptr) {
    throw std::runtime_error("Subgraph: fragment cannot be nullptr");
  }
  if (name.empty()) {
    throw std::runtime_error("Subgraph: name cannot be empty");
  }
  // Set configuration before compose() is called (by make_subgraph or Python __init__)
  if (!config_file.empty()) {
    config_ = std::make_shared<Config>(config_file);
  }
}

// ========== Configuration Methods ==========

void Subgraph::config(const std::string& config_file, const std::string& prefix) {
  if (config_) {
    HOLOSCAN_LOG_WARN("Subgraph config was already set. Overwriting...");
  }
  if (is_composed_) {
    HOLOSCAN_LOG_WARN(
        "Subgraph has already been composed. Please make sure that composition is not dependent "
        "on this config() call.");
  }

  if (!config_file.empty()) {
    config_ = std::make_shared<Config>(config_file, prefix);
  }
}

Config& Subgraph::config() {
  return *config_shared();
}

std::shared_ptr<Config> Subgraph::config_shared() {
  if (!config_) {
    config_ = std::make_shared<Config>();
  }
  return config_;
}

ArgList Subgraph::from_config(const std::string& key) {
  return config().from_config(key);
}

std::unordered_set<std::string> Subgraph::config_keys() {
  return config().config_keys();
}

void Subgraph::add_operator(const std::shared_ptr<Operator>& op) {
  if (!op) {
    auto err_msg = std::string("Cannot add null operator to subgraph");
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }

  // Set qualified name and add directly to Fragment's main graph
  const std::string qualified_name = get_qualified_name(op->name(), "operator");

  // Check if an operator with this qualified name already exists
  auto existing_node = fragment_->graph().find_node(qualified_name);
  if (existing_node) {
    HOLOSCAN_LOG_DEBUG(
        "Subgraph: an operator with qualified name '{}' already exists in fragment, skipping "
        "add_operator.",
        qualified_name);
    return;
  } else {
    HOLOSCAN_LOG_DEBUG("Subgraph: adding operator with name '{}' under qualified name '{}'",
                       op->name(),
                       qualified_name);
  }

  op->name(qualified_name);
  fragment_->add_operator(op);
}

void Subgraph::add_subgraph(const std::shared_ptr<Subgraph>& subgraph) {
  if (!subgraph) {
    HOLOSCAN_LOG_ERROR("Cannot add null subgraph to subgraph '{}'", name_);
    return;
  }

  // If this exact subgraph is already owned (e.g. make_subgraph then add_subgraph), no-op.
  for (const auto& existing : nested_subgraphs_) {
    if (existing == subgraph) {
      return;
    }
  }

  const std::string& current_name = subgraph->name();

  if (subgraph->is_composed()) {
    // Already composed (e.g. Python subgraphs auto-compose during __init__).
    // The name should already be qualified. Verify the prefix and extract the child name.
    std::string expected_prefix = name_ + "_";
    std::string child_name;
    if (current_name.size() >= expected_prefix.size() &&
        current_name.substr(0, expected_prefix.size()) == expected_prefix) {
      child_name = current_name.substr(expected_prefix.size());
    }
    if (child_name.empty()) {
      throw std::runtime_error(fmt::format(
          "Subgraph::add_subgraph: subgraph '{}' is already composed but its name does not "
          "start with the expected prefix '{}' followed by a non-empty child name. "
          "When adding an already-composed subgraph, it must have been constructed with this "
          "subgraph as the parent so the name is properly qualified.",
          current_name,
          expected_prefix));
    }

    if (nested_subgraph_names_.find(child_name) != nested_subgraph_names_.end()) {
      throw std::runtime_error(
          fmt::format("Subgraph::add_subgraph: Duplicate nested subgraph name '{}' in subgraph "
                      "'{}'. Each nested subgraph must have a unique name within the same parent "
                      "subgraph.",
                      child_name,
                      name_));
    }

    nested_subgraphs_.push_back(subgraph);
    nested_subgraph_names_.insert(std::move(child_name));
  } else {
    // Not yet composed -- qualify the name and compose.
    // This is the C++ factory pattern path.
    if (nested_subgraph_names_.find(current_name) != nested_subgraph_names_.end()) {
      throw std::runtime_error(
          fmt::format("Subgraph::add_subgraph: Duplicate nested subgraph name '{}' in subgraph "
                      "'{}'. Each nested subgraph must have a unique name within the same parent "
                      "subgraph.",
                      current_name,
                      name_));
    }

    std::string qualified_name = get_qualified_name(current_name, "subgraph");
    subgraph->name_ = std::move(qualified_name);

    subgraph->compose();
    subgraph->set_composed(true);

    nested_subgraphs_.push_back(subgraph);
    nested_subgraph_names_.insert(current_name);
  }
}

void Subgraph::add_flow(const std::shared_ptr<Operator>& upstream,
                        const std::shared_ptr<Operator>& downstream,
                        std::set<std::pair<std::string, std::string>> port_pairs) {
  if (!upstream || !downstream) {
    auto err_msg = fmt::format(
        "Cannot add flow in subgraph '{}': upstream or downstream operator is null", name_);
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }
  // update operator names and add them to the graph
  add_operator(downstream);
  add_operator(upstream);
  fragment_->add_flow(upstream, downstream, std::move(port_pairs));
}

void Subgraph::add_flow(const std::shared_ptr<Operator>& upstream_op,
                        const std::shared_ptr<Subgraph>& downstream_subgraph,
                        std::set<std::pair<std::string, std::string>> port_pairs) {
  if (!upstream_op || !downstream_subgraph) {
    auto err_msg = fmt::format(
        "Cannot add flow in subgraph '{}': upstream operator or downstream subgraph is null",
        name_);
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }
  add_operator(upstream_op);
  fragment_->add_flow(upstream_op, downstream_subgraph, std::move(port_pairs));
}

void Subgraph::add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                        const std::shared_ptr<Operator>& downstream_op,
                        std::set<std::pair<std::string, std::string>> port_pairs) {
  if (!upstream_subgraph || !downstream_op) {
    auto err_msg = fmt::format(
        "Cannot add flow in subgraph '{}': upstream subgraph or downstream operator is null",
        name_);
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }
  add_operator(downstream_op);
  fragment_->add_flow(upstream_subgraph, downstream_op, std::move(port_pairs));
}

void Subgraph::add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                        const std::shared_ptr<Subgraph>& downstream_subgraph,
                        std::set<std::pair<std::string, std::string>> port_pairs) {
  if (!upstream_subgraph || !downstream_subgraph) {
    auto err_msg = fmt::format(
        "Cannot add flow in subgraph '{}': upstream or downstream subgraph is null", name_);
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }
  fragment_->add_flow(upstream_subgraph, downstream_subgraph, std::move(port_pairs));
}

void Subgraph::bind_input_topic(const std::string& interface_port, const std::string& topic,
                                const std::optional<nvidia::gxf::QoSProfile>& qos,
                                bool replace_connector) {
  if (!is_composed_) {
    throw std::runtime_error(fmt::format(
        "Subgraph '{}': bind_input_topic() requires compose() to have been called first", name_));
  }
  bind_interface_port_topic(*this, interface_port, topic, qos, true, replace_connector);
}

void Subgraph::bind_output_topic(const std::string& interface_port, const std::string& topic,
                                 const std::optional<nvidia::gxf::QoSProfile>& qos,
                                 bool replace_connector) {
  if (!is_composed_) {
    throw std::runtime_error(fmt::format(
        "Subgraph '{}': bind_output_topic() requires compose() to have been called first", name_));
  }
  bind_interface_port_topic(*this, interface_port, topic, qos, false, replace_connector);
}

void Subgraph::add_flow(const std::shared_ptr<Operator>& upstream_op,
                        const std::shared_ptr<Subgraph>& downstream_subgraph,
                        const IOSpec::ConnectorType connector_type) {
  if (!upstream_op || !downstream_subgraph) {
    auto err_msg = fmt::format(
        "Cannot add flow in subgraph '{}': upstream operator or downstream subgraph is null",
        name_);
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }
  add_operator(upstream_op);
  fragment_->add_flow(upstream_op, downstream_subgraph, connector_type);
}

void Subgraph::add_flow(const std::shared_ptr<Operator>& upstream_op,
                        const std::shared_ptr<Subgraph>& downstream_subgraph,
                        std::set<std::pair<std::string, std::string>> port_pairs,
                        const IOSpec::ConnectorType connector_type) {
  if (!upstream_op || !downstream_subgraph) {
    auto err_msg = fmt::format(
        "Cannot add flow in subgraph '{}': upstream operator or downstream subgraph is null",
        name_);
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }
  add_operator(upstream_op);
  fragment_->add_flow(upstream_op, downstream_subgraph, std::move(port_pairs), connector_type);
}

void Subgraph::add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                        const std::shared_ptr<Operator>& downstream_op,
                        const IOSpec::ConnectorType connector_type) {
  if (!upstream_subgraph || !downstream_op) {
    auto err_msg = fmt::format(
        "Cannot add flow in subgraph '{}': upstream subgraph or downstream operator is null",
        name_);
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }
  add_operator(downstream_op);
  fragment_->add_flow(upstream_subgraph, downstream_op, connector_type);
}

void Subgraph::add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                        const std::shared_ptr<Operator>& downstream_op,
                        std::set<std::pair<std::string, std::string>> port_pairs,
                        const IOSpec::ConnectorType connector_type) {
  if (!upstream_subgraph || !downstream_op) {
    auto err_msg = fmt::format(
        "Cannot add flow in subgraph '{}': upstream subgraph or downstream operator is null",
        name_);
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }
  add_operator(downstream_op);
  fragment_->add_flow(upstream_subgraph, downstream_op, std::move(port_pairs), connector_type);
}

void Subgraph::add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                        const std::shared_ptr<Subgraph>& downstream_subgraph,
                        const IOSpec::ConnectorType connector_type) {
  if (!upstream_subgraph || !downstream_subgraph) {
    auto err_msg = fmt::format(
        "Cannot add flow in subgraph '{}': upstream or downstream subgraph is null", name_);
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }
  fragment_->add_flow(upstream_subgraph, downstream_subgraph, connector_type);
}

void Subgraph::add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                        const std::shared_ptr<Subgraph>& downstream_subgraph,
                        std::set<std::pair<std::string, std::string>> port_pairs,
                        const IOSpec::ConnectorType connector_type) {
  if (!upstream_subgraph || !downstream_subgraph) {
    auto err_msg = fmt::format(
        "Cannot add flow in subgraph '{}': upstream or downstream subgraph is null", name_);
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }
  fragment_->add_flow(
      upstream_subgraph, downstream_subgraph, std::move(port_pairs), connector_type);
}

void Subgraph::set_dynamic_flows(
    const std::shared_ptr<Operator>& op,
    const std::function<void(const std::shared_ptr<Operator>&)>& dynamic_flow_func) {
  if (fragment_) {
    fragment_->set_dynamic_flows(op, dynamic_flow_func);
  }
}

void Subgraph::add_data_logger(const std::shared_ptr<DataLogger>& logger) {
  if (fragment_) {
    fragment_->add_data_logger(logger);
  }
}

void Subgraph::add_interface_port(const std::string& external_name,
                                  const std::shared_ptr<Operator>& internal_op,
                                  const std::optional<std::string>& internal_port,
                                  std::optional<bool> is_input) {
  if (!internal_op) {
    auto err_msg =
        fmt::format("Cannot add interface port '{}': internal operator is null", external_name);
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }

  // Use external_name as internal_port if not specified
  const std::string& port_name = internal_port.value_or(external_name);

  // Add the operator to the fragment graph with qualified name (if not already added)
  std::string qualified_name = get_qualified_name(internal_op->name(), "operator");
  if (!fragment_->graph().find_node(qualified_name)) {
    add_operator(internal_op);
  }

  // Determine port direction: use provided value or auto-detect from operator's port definitions
  bool port_is_input;
  if (is_input.has_value()) {
    port_is_input = is_input.value();
  } else {
    // Auto-detect port direction by checking both inputs and outputs
    if (!internal_op->spec()) {
      auto err_msg = fmt::format(
          "Cannot auto-detect port direction for '{}' on operator '{}': operator spec is null",
          port_name,
          internal_op->name());
      HOLOSCAN_LOG_ERROR(err_msg);
      throw std::runtime_error(err_msg);
    }

    auto* op_spec = internal_op->spec();
    const auto& inputs = op_spec->inputs();
    const auto& outputs = op_spec->outputs();

    bool found_in_inputs = inputs.find(port_name) != inputs.end();
    bool found_in_outputs = outputs.find(port_name) != outputs.end();

    if (found_in_inputs && found_in_outputs) {
      auto err_msg = fmt::format(
          "Port '{}' exists as both an input and output on operator '{}'. "
          "Please specify is_input explicitly using add_input_interface_port() or "
          "add_output_interface_port() to disambiguate.",
          port_name,
          internal_op->name());
      HOLOSCAN_LOG_ERROR(err_msg);
      throw std::runtime_error(err_msg);
    } else if (found_in_inputs) {
      port_is_input = true;
      HOLOSCAN_LOG_DEBUG(
          "Auto-detected port '{}' as input port on operator '{}'", port_name, internal_op->name());
    } else if (found_in_outputs) {
      port_is_input = false;
      HOLOSCAN_LOG_DEBUG("Auto-detected port '{}' as output port on operator '{}'",
                         port_name,
                         internal_op->name());
    } else {
      auto err_msg =
          fmt::format("Port '{}' not found on operator '{}'. Available inputs: [{}], outputs: [{}]",
                      port_name,
                      internal_op->name(),
                      format_port_list(inputs),
                      format_port_list(outputs));
      HOLOSCAN_LOG_ERROR(err_msg);
      throw std::runtime_error(err_msg);
    }
  }

  auto it = interface_ports_.find(external_name);
  if (it != interface_ports_.end()) {
    // Port name already exists - check if we can add another mapping
    if (!port_is_input) {
      // Output ports cannot have multiple mappings
      auto err_msg = fmt::format(
          "Output interface port '{}' already exists in Subgraph '{}'. "
          "Output ports can only have a single mapping.",
          external_name,
          name_);
      HOLOSCAN_LOG_ERROR(err_msg);
      throw std::runtime_error(err_msg);
    }

    // For input ports, verify the existing port is also an input port
    if (!it->second.is_input) {
      auto err_msg = fmt::format(
          "Cannot add input interface port '{}': an output port with the same name already exists "
          "in Subgraph '{}'",
          external_name,
          name_);
      HOLOSCAN_LOG_ERROR(err_msg);
      throw std::runtime_error(err_msg);
    }

    // Input port with broadcast support - append to existing mappings
    HOLOSCAN_LOG_DEBUG(
        "Adding additional mapping to input interface port '{}' -> '{}:{}' in Subgraph '{}' "
        "(total mappings: {})",
        external_name,
        internal_op->name(),
        port_name,
        name_,
        it->second.size() + 1);

    // Validate that the operator has the specified port with correct type
    if (!validate_operator_port(internal_op, port_name, port_is_input)) {
      throw std::runtime_error("validation of interface port failed");
    }

    // Append to existing interface port mappings
    it->second.add_mapping(internal_op, port_name);
  } else {
    // Validate that the operator has the specified port with correct type
    if (!validate_operator_port(internal_op, port_name, port_is_input)) {
      throw std::runtime_error("validation of interface port failed");
    }

    // Create new interface port with first mapping
    InterfacePort new_port;
    new_port.is_input = port_is_input;
    new_port.port_type = InterfacePort::PortType::kData;
    new_port.add_mapping(internal_op, port_name);
    interface_ports_[external_name] = std::move(new_port);
  }

  HOLOSCAN_LOG_DEBUG("Added interface port '{}' -> '{}:{}' (input: {}) to Subgraph '{}'",
                     external_name,
                     internal_op->name(),
                     port_name,
                     port_is_input,
                     name_);
}

void Subgraph::add_input_interface_port(const std::string& external_name,
                                        const std::shared_ptr<Operator>& internal_op,
                                        const std::optional<std::string>& internal_port) {
  add_interface_port(external_name, internal_op, internal_port, true);
}

void Subgraph::add_output_interface_port(const std::string& external_name,
                                         const std::shared_ptr<Operator>& internal_op,
                                         const std::optional<std::string>& internal_port) {
  add_interface_port(external_name, internal_op, internal_port, false);
}

void Subgraph::add_interface_port(const std::string& external_name,
                                  const std::shared_ptr<Subgraph>& internal_subgraph,
                                  const std::optional<std::string>& internal_interface_port,
                                  std::optional<bool> is_input) {
  if (!internal_subgraph) {
    auto err_msg =
        fmt::format("Cannot add interface port '{}': internal subgraph is null", external_name);
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }

  // Use external_name as internal_interface_port if not specified
  const std::string& port_name = internal_interface_port.value_or(external_name);

  // Find the interface port in the nested subgraph
  const auto& nested_ports = internal_subgraph->interface_ports();
  auto nested_it = nested_ports.find(port_name);

  if (nested_it == nested_ports.end()) {
    // Check if this is an execution interface port to provide a better error message
    const auto& nested_exec_ports = internal_subgraph->exec_interface_ports();
    auto exec_it = nested_exec_ports.find(port_name);
    if (exec_it != nested_exec_ports.end()) {
      auto err_msg = fmt::format(
          "Cannot add interface port '{}': nested subgraph '{}' has an execution interface port "
          "named '{}', but a data interface port is required. Use add_input_interface_port()/"
          "add_output_interface_port() for data ports or add_input_exec_interface_port()/"
          "add_output_exec_interface_port() for execution ports.",
          external_name,
          internal_subgraph->name(),
          port_name);
      HOLOSCAN_LOG_ERROR(err_msg);
      throw std::runtime_error(err_msg);
    } else {
      auto err_msg = fmt::format(
          "Cannot add interface port '{}': nested subgraph '{}' does not have interface port '{}'",
          external_name,
          internal_subgraph->name(),
          port_name);
      HOLOSCAN_LOG_ERROR(err_msg);
      throw std::runtime_error(err_msg);
    }
  }

  const InterfacePort& nested_port = nested_it->second;

  // Determine port direction: use provided value or auto-detect from nested subgraph's port
  bool port_is_input;
  if (is_input.has_value()) {
    port_is_input = is_input.value();
    // Validate that the specified direction matches the nested port
    if (port_is_input != nested_port.is_input) {
      auto err_msg = fmt::format(
          "Cannot add interface port '{}': is_input={} was specified but nested subgraph '{}' "
          "interface port '{}' is an {} port",
          external_name,
          port_is_input,
          internal_subgraph->name(),
          port_name,
          nested_port.is_input ? "input" : "output");
      HOLOSCAN_LOG_ERROR(err_msg);
      throw std::runtime_error(err_msg);
    }
  } else {
    // Auto-detect from nested subgraph's interface port
    port_is_input = nested_port.is_input;
    HOLOSCAN_LOG_DEBUG(
        "Auto-detected interface port '{}' as {} port from nested subgraph '{}' port '{}'",
        external_name,
        port_is_input ? "input" : "output",
        internal_subgraph->name(),
        port_name);
  }

  auto it = interface_ports_.find(external_name);
  if (it != interface_ports_.end()) {
    // Port name already exists - check if we can add another mapping
    if (!port_is_input) {
      auto err_msg = fmt::format(
          "Output interface port '{}' already exists in Subgraph '{}'. "
          "Output ports can only have a single mapping.",
          external_name,
          name_);
      HOLOSCAN_LOG_ERROR(err_msg);
      throw std::runtime_error(err_msg);
    }

    // For input ports, verify the existing port is also an input port
    if (!it->second.is_input) {
      auto err_msg = fmt::format(
          "Cannot add input interface port '{}': an output port with the same name already exists "
          "in Subgraph '{}'",
          external_name,
          name_);
      HOLOSCAN_LOG_ERROR(err_msg);
      throw std::runtime_error(err_msg);
    }
  }

  // Add all resolved port mappings from the nested subgraph
  for (const auto& mapping : nested_port.mappings) {
    // Validate that the resolved port has the correct type
    if (!validate_operator_port(
            mapping.internal_operator, mapping.internal_port_name, port_is_input)) {
      auto err_msg = fmt::format(
          "Cannot add interface port '{}': nested subgraph '{}' interface port '{}' resolves to "
          "operator '{}' port '{}' which has incorrect type (expected {})",
          external_name,
          internal_subgraph->name(),
          port_name,
          mapping.internal_operator->name(),
          mapping.internal_port_name,
          port_is_input ? "input" : "output");
      HOLOSCAN_LOG_ERROR(err_msg);
      throw std::runtime_error(err_msg);
    }

    // Add to existing or create new interface port
    if (it != interface_ports_.end()) {
      it->second.add_mapping(mapping.internal_operator, mapping.internal_port_name);
    } else {
      InterfacePort new_port;
      new_port.is_input = port_is_input;
      new_port.port_type = InterfacePort::PortType::kData;
      new_port.add_mapping(mapping.internal_operator, mapping.internal_port_name);
      interface_ports_[external_name] = std::move(new_port);
      it = interface_ports_.find(external_name);  // Update iterator for subsequent mappings
    }
  }

  HOLOSCAN_LOG_DEBUG(
      "Added interface port '{}' -> nested subgraph '{}' interface port '{}' ({} mappings, "
      "input: {}) to Subgraph '{}'",
      external_name,
      internal_subgraph->name(),
      port_name,
      nested_port.mappings.size(),
      port_is_input,
      name_);
}

void Subgraph::add_input_interface_port(const std::string& external_name,
                                        const std::shared_ptr<Subgraph>& internal_subgraph,
                                        const std::optional<std::string>& internal_interface_port) {
  add_interface_port(external_name, internal_subgraph, internal_interface_port, true);
}

void Subgraph::add_output_interface_port(
    const std::string& external_name, const std::shared_ptr<Subgraph>& internal_subgraph,
    const std::optional<std::string>& internal_interface_port) {
  add_interface_port(external_name, internal_subgraph, internal_interface_port, false);
}

std::pair<std::shared_ptr<Operator>, std::string> Subgraph::get_interface_operator_port(
    const std::string& port_name) const {
  // First check local interface ports
  auto it = interface_ports_.find(port_name);
  if (it != interface_ports_.end() && !it->second.empty()) {
    // Return the first mapping
    const auto& first_mapping = it->second.mappings[0];
    return {first_mapping.internal_operator, first_mapping.internal_port_name};
  }

  // If not found locally, check nested subgraphs recursively
  for (const auto& nested_subgraph : nested_subgraphs_) {
    auto result = nested_subgraph->get_interface_operator_port(port_name);
    if (result.first) {  // Found in nested subgraph
      return result;
    }
  }

  // Port not found in this subgraph or its nested subgraphs
  return {nullptr, ""};
}

std::pair<std::shared_ptr<Operator>, std::string> Subgraph::get_exec_interface_operator_port(
    const std::string& port_name) const {
  // First check local exec interface ports
  auto it = exec_interface_ports_.find(port_name);
  if (it != exec_interface_ports_.end() && !it->second.empty()) {
    const auto& first_mapping = it->second.mappings[0];
    return {first_mapping.internal_operator, first_mapping.internal_port_name};
  }

  // If not found locally, check nested subgraphs recursively
  for (const auto& nested_subgraph : nested_subgraphs_) {
    auto result = nested_subgraph->get_exec_interface_operator_port(port_name);
    if (result.first) {  // Found in nested subgraph
      return result;
    }
  }

  // Port not found in this subgraph or its nested subgraphs
  return {nullptr, ""};
}

std::vector<std::shared_ptr<Operator>> Subgraph::operators() const {
  std::vector<std::shared_ptr<Operator>> result;

  // Get all nodes from the fragment's graph
  const auto& nodes = fragment_->graph().get_nodes();

  // Filter operators whose names start with this subgraph's name followed by underscore
  const std::string prefix = name_ + "_";
  for (const auto& node : nodes) {
    const auto& name = node->name();
    // Check if name starts with prefix using compare (C++17 compatible)
    if (name.size() >= prefix.size() && name.compare(0, prefix.size(), prefix) == 0) {
      result.push_back(node);
    }
  }

  return result;
}

std::string Subgraph::format_port_list(
    const std::unordered_map<std::string, std::shared_ptr<IOSpec>>& ports) {
  fmt::memory_buffer buf;
  bool first = true;
  for (const auto& [name, spec] : ports) {
    if (!first) {
      fmt::format_to(std::back_inserter(buf), ", ");
    }
    fmt::format_to(std::back_inserter(buf), "'{}'", name);
    first = false;
  }
  return fmt::to_string(buf);
}

bool Subgraph::validate_operator_port(const std::shared_ptr<Operator>& op,
                                      const std::string& port_name, bool expect_input) {
  if (!op->spec()) {
    auto err_msg = fmt::format(
        "Cannot validate port '{}' on operator '{}': operator spec is null", port_name, op->name());
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }

  auto* op_spec = op->spec();

  // Check input ports if we expect an input port
  if (expect_input) {
    const auto& inputs = op_spec->inputs();
    auto input_it = inputs.find(port_name);
    if (input_it != inputs.end()) {
      // Found the port, verify it's actually an input port
      if (input_it->second->io_type() == IOSpec::IOType::kInput) {
        return true;  // Valid input port
      } else {
        HOLOSCAN_LOG_ERROR(
            "Port '{}' on operator '{}' exists but is an output port, expected input port",
            port_name,
            op->name());
        return false;
      }
    }
  } else {
    // Check output ports if we expect an output port
    const auto& outputs = op_spec->outputs();
    auto output_it = outputs.find(port_name);
    if (output_it != outputs.end()) {
      // Found the port, verify it's actually an output port
      if (output_it->second->io_type() == IOSpec::IOType::kOutput) {
        return true;  // Valid output port
      } else {
        HOLOSCAN_LOG_ERROR(
            "Port '{}' on operator '{}' exists but is an input port, expected output port",
            port_name,
            op->name());
        return false;
      }
    }
  }

  // Port not found - provide helpful error message
  HOLOSCAN_LOG_ERROR("Operator '{}' does not have {} port '{}'",
                     op->name(),
                     expect_input ? "input" : "output",
                     port_name);

  // List available ports for debugging
  if (expect_input) {
    const auto& inputs = op_spec->inputs();
    if (!inputs.empty()) {
      HOLOSCAN_LOG_ERROR("Available input ports: {}", format_port_list(inputs));
    } else {
      HOLOSCAN_LOG_ERROR("Operator '{}' has no input ports", op->name());
    }
  } else {
    const auto& outputs = op_spec->outputs();
    if (!outputs.empty()) {
      HOLOSCAN_LOG_ERROR("Available output ports: {}", format_port_list(outputs));
    } else {
      HOLOSCAN_LOG_ERROR("Operator '{}' has no output ports", op->name());
    }
  }

  return false;
}

bool Subgraph::validate_operator_exec_port(const std::shared_ptr<Operator>& op) {
  if (!op) {
    HOLOSCAN_LOG_ERROR("Operator pointer is null");
    return false;
  }

  // Check that the operator is a Native operator
  if (op->operator_type() != Operator::OperatorType::kNative) {
    HOLOSCAN_LOG_ERROR(
        "Operator '{}' is not a Native operator. Only Native operators can be exposed as "
        "execution interface ports.",
        op->name());
    return false;
  }

  // Note: We don't validate execution specs here because they are created automatically
  // by the GXF executor during initialization via add_control_flow(), not during compose().
  // The executor will create the appropriate input/output execution specs when needed.

  return true;
}

bool Subgraph::check_exec_port_name_available(const std::string& external_name) {
  // Check for duplicate interface port names across both data and exec ports
  if (interface_ports_.find(external_name) != interface_ports_.end()) {
    auto err_msg = fmt::format(
        "Interface port '{}' already exists as a data port in subgraph '{}'", external_name, name_);
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }
  if (exec_interface_ports_.find(external_name) != exec_interface_ports_.end()) {
    auto err_msg = fmt::format(
        "Execution interface port '{}' already exists in subgraph '{}'", external_name, name_);
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }
  return true;
}

void Subgraph::register_exec_interface_port(const std::string& external_name,
                                            const std::shared_ptr<Operator>& internal_op,
                                            const std::string& internal_port_name, bool is_input) {
  InterfacePort port;
  port.is_input = is_input;
  port.port_type = InterfacePort::PortType::kExecution;
  port.add_mapping(internal_op, internal_port_name);

  exec_interface_ports_[external_name] = std::move(port);
  HOLOSCAN_LOG_DEBUG("Added {} execution interface port '{}' -> operator '{}' in subgraph '{}'",
                     is_input ? "input" : "output",
                     external_name,
                     internal_op->name(),
                     name_);
}

std::pair<std::shared_ptr<Operator>, std::string> Subgraph::resolve_nested_exec_port(
    const std::string& external_name, const std::shared_ptr<Subgraph>& internal_subgraph,
    const std::string& internal_interface_port, bool expect_input) {
  // Look up the interface port in the nested subgraph
  const auto& nested_exec_ports = internal_subgraph->exec_interface_ports();
  auto it = nested_exec_ports.find(internal_interface_port);

  if (it == nested_exec_ports.end()) {
    // Check if this is a data interface port to provide a better error message
    const auto& nested_data_ports = internal_subgraph->interface_ports();
    auto data_it = nested_data_ports.find(internal_interface_port);
    std::string err_msg;
    if (data_it != nested_data_ports.end()) {
      err_msg = fmt::format(
          "Nested subgraph '{}' has a data interface port named '{}', but an execution interface "
          "port is required. Use add_{}_interface_port() for data ports or "
          "add_{}_exec_interface_port() for execution ports.",
          internal_subgraph->name(),
          internal_interface_port,
          expect_input ? "input" : "output",
          expect_input ? "input" : "output");
    } else {
      err_msg =
          fmt::format("Nested subgraph '{}' does not have an execution interface port named '{}'",
                      internal_subgraph->name(),
                      internal_interface_port);
    }
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }

  const auto& nested_port = it->second;
  if (nested_port.is_input != expect_input) {
    auto err_msg = fmt::format(
        "Execution interface port '{}' in nested subgraph '{}' is an {} port, expected {}",
        internal_interface_port,
        internal_subgraph->name(),
        nested_port.is_input ? "input" : "output",
        expect_input ? "input" : "output");
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }

  if (nested_port.empty()) {
    auto err_msg = fmt::format(
        "Execution interface port '{}' in nested subgraph '{}' has no "
        "mappings",
        internal_interface_port,
        internal_subgraph->name());
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }

  const auto& first_mapping = nested_port.mappings[0];
  HOLOSCAN_LOG_DEBUG(
      "Resolved {} execution interface port '{}' -> nested subgraph '{}' port '{}' -> operator "
      "'{}'",
      expect_input ? "input" : "output",
      external_name,
      internal_subgraph->name(),
      internal_interface_port,
      first_mapping.internal_operator->name());

  return {first_mapping.internal_operator, first_mapping.internal_port_name};
}

// ========== Execution Interface Port Methods ==========

void Subgraph::add_input_exec_interface_port(const std::string& external_name,
                                             const std::shared_ptr<Operator>& internal_op) {
  if (!validate_operator_exec_port(internal_op)) {
    return;
  }

  // Add the operator to the fragment graph with qualified name (if not already added)
  std::string qualified_name = get_qualified_name(internal_op->name(), "operator");
  if (!fragment_->graph().find_node(qualified_name)) {
    add_operator(internal_op);
  }

  if (!check_exec_port_name_available(external_name)) {
    return;
  }

  register_exec_interface_port(external_name, internal_op, Operator::kInputExecPortName, true);
}

void Subgraph::add_output_exec_interface_port(const std::string& external_name,
                                              const std::shared_ptr<Operator>& internal_op) {
  if (!validate_operator_exec_port(internal_op)) {
    return;
  }

  // Add the operator to the fragment graph with qualified name (if not already added)
  std::string qualified_name = get_qualified_name(internal_op->name(), "operator");
  if (!fragment_->graph().find_node(qualified_name)) {
    add_operator(internal_op);
  }

  if (!check_exec_port_name_available(external_name)) {
    return;
  }

  register_exec_interface_port(external_name, internal_op, Operator::kOutputExecPortName, false);
}

void Subgraph::add_input_exec_interface_port(
    const std::string& external_name, const std::shared_ptr<Subgraph>& internal_subgraph,
    const std::optional<std::string>& internal_interface_port) {
  if (!internal_subgraph) {
    HOLOSCAN_LOG_ERROR("Internal subgraph pointer is null");
    throw std::runtime_error("Internal subgraph pointer is null");
  }

  // Use external_name as internal_interface_port if not specified
  const std::string& port_name = internal_interface_port.value_or(external_name);

  if (!check_exec_port_name_available(external_name)) {
    return;
  }

  auto [resolved_op, resolved_port] =
      resolve_nested_exec_port(external_name, internal_subgraph, port_name, true);

  if (!resolved_op) {
    return;
  }

  register_exec_interface_port(external_name, resolved_op, resolved_port, true);
}

void Subgraph::add_output_exec_interface_port(
    const std::string& external_name, const std::shared_ptr<Subgraph>& internal_subgraph,
    const std::optional<std::string>& internal_interface_port) {
  if (!internal_subgraph) {
    HOLOSCAN_LOG_ERROR("Internal subgraph pointer is null");
    throw std::runtime_error("Internal subgraph pointer is null");
  }

  // Use external_name as internal_interface_port if not specified
  const std::string& port_name = internal_interface_port.value_or(external_name);

  if (!check_exec_port_name_available(external_name)) {
    return;
  }

  auto [resolved_op, resolved_port] =
      resolve_nested_exec_port(external_name, internal_subgraph, port_name, false);

  if (!resolved_op) {
    return;
  }

  register_exec_interface_port(external_name, resolved_op, resolved_port, false);
}

}  // namespace holoscan
