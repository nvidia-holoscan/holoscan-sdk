/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "holoscan/core/subgraph.hpp"

#include <fmt/format.h>

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

#include "holoscan/core/fragment.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan {

Subgraph::Subgraph(Fragment* fragment, const std::string& name) : fragment_(fragment), name_(name) {
  if (fragment == nullptr) {
    throw std::runtime_error("Subgraph: fragment cannot be nullptr");
  }
  if (name.empty()) {
    throw std::runtime_error("Subgraph: name cannot be empty");
  }
}

void Subgraph::add_operator(std::shared_ptr<Operator> op) {
  if (!op) {
    HOLOSCAN_LOG_ERROR("Cannot add null operator to subgraph");
    return;
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

void Subgraph::add_flow(const std::shared_ptr<Operator>& upstream,
                        const std::shared_ptr<Operator>& downstream,
                        std::set<std::pair<std::string, std::string>> port_pairs) {
  // update operator names and add them to the graph
  add_operator(downstream);
  add_operator(upstream);
  fragment_->add_flow(upstream, downstream, port_pairs);
}

void Subgraph::add_flow(const std::shared_ptr<Operator>& upstream_op,
                        const std::shared_ptr<Subgraph>& downstream_subgraph,
                        std::set<std::pair<std::string, std::string>> port_pairs) {
  add_operator(upstream_op);
  fragment_->add_flow(upstream_op, downstream_subgraph, port_pairs);
}

void Subgraph::add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                        const std::shared_ptr<Operator>& downstream_op,
                        std::set<std::pair<std::string, std::string>> port_pairs) {
  add_operator(downstream_op);
  fragment_->add_flow(upstream_subgraph, downstream_op, port_pairs);
}

void Subgraph::add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                        const std::shared_ptr<Subgraph>& downstream_subgraph,
                        std::set<std::pair<std::string, std::string>> port_pairs) {
  fragment_->add_flow(upstream_subgraph, downstream_subgraph, port_pairs);
}

void Subgraph::add_flow(const std::shared_ptr<Operator>& upstream_op,
                        const std::shared_ptr<Subgraph>& downstream_subgraph,
                        const IOSpec::ConnectorType connector_type) {
  add_operator(upstream_op);
  fragment_->add_flow(upstream_op, downstream_subgraph, connector_type);
}

void Subgraph::add_flow(const std::shared_ptr<Operator>& upstream_op,
                        const std::shared_ptr<Subgraph>& downstream_subgraph,
                        std::set<std::pair<std::string, std::string>> port_pairs,
                        const IOSpec::ConnectorType connector_type) {
  add_operator(upstream_op);
  fragment_->add_flow(upstream_op, downstream_subgraph, port_pairs, connector_type);
}

void Subgraph::add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                        const std::shared_ptr<Operator>& downstream_op,
                        const IOSpec::ConnectorType connector_type) {
  add_operator(downstream_op);
  fragment_->add_flow(upstream_subgraph, downstream_op, connector_type);
}

void Subgraph::add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                        const std::shared_ptr<Operator>& downstream_op,
                        std::set<std::pair<std::string, std::string>> port_pairs,
                        const IOSpec::ConnectorType connector_type) {
  add_operator(downstream_op);
  fragment_->add_flow(upstream_subgraph, downstream_op, port_pairs, connector_type);
}

void Subgraph::add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                        const std::shared_ptr<Subgraph>& downstream_subgraph,
                        const IOSpec::ConnectorType connector_type) {
  fragment_->add_flow(upstream_subgraph, downstream_subgraph, connector_type);
}

void Subgraph::add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                        const std::shared_ptr<Subgraph>& downstream_subgraph,
                        std::set<std::pair<std::string, std::string>> port_pairs,
                        const IOSpec::ConnectorType connector_type) {
  fragment_->add_flow(upstream_subgraph, downstream_subgraph, port_pairs, connector_type);
}

void Subgraph::set_dynamic_flows(
    const std::shared_ptr<Operator>& op,
    const std::function<void(const std::shared_ptr<Operator>&)>& dynamic_flow_func) {
  if (fragment_) {
    fragment_->set_dynamic_flows(op, dynamic_flow_func);
  }
}

void Subgraph::add_interface_port(const std::string& external_name,
                                  const std::shared_ptr<Operator>& internal_op,
                                  const std::string& internal_port, bool is_input) {
  if (!internal_op) {
    auto err_msg =
        fmt::format("Cannot add interface port '{}': internal operator is null", external_name);
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }

  // Add the operator to the fragment graph with qualified name (if not already added)
  std::string qualified_name = get_qualified_name(internal_op->name(), "operator");
  if (!fragment_->graph().find_node(qualified_name)) {
    add_operator(internal_op);
  }

  if (interface_ports_.find(external_name) != interface_ports_.end()) {
    auto err_msg =
        fmt::format("Interface port '{}' already exists in Subgraph '{}'", external_name, name_);
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }

  // Validate that the operator has the specified port with correct type
  if (!validate_operator_port(internal_op, internal_port, is_input)) {
    throw std::runtime_error("validation of interface port failed");
  }

  // Store the interface port mapping
  interface_ports_[external_name] = InterfacePort{internal_op, internal_port, is_input};

  HOLOSCAN_LOG_DEBUG("Added interface port '{}' -> '{}:{}' (input: {}) to Subgraph '{}'",
                     external_name,
                     internal_op->name(),
                     internal_port,
                     is_input,
                     name_);
}

void Subgraph::add_input_interface_port(const std::string& external_name,
                                        const std::shared_ptr<Operator>& internal_op,
                                        const std::string& internal_port) {
  add_interface_port(external_name, internal_op, internal_port, true);
}

void Subgraph::add_output_interface_port(const std::string& external_name,
                                         const std::shared_ptr<Operator>& internal_op,
                                         const std::string& internal_port) {
  add_interface_port(external_name, internal_op, internal_port, false);
}

void Subgraph::add_interface_port(const std::string& external_name,
                                  const std::shared_ptr<Subgraph>& internal_subgraph,
                                  const std::string& internal_interface_port, bool is_input) {
  if (!internal_subgraph) {
    auto err_msg =
        fmt::format("Cannot add interface port '{}': internal subgraph is null", external_name);
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }

  if (interface_ports_.find(external_name) != interface_ports_.end()) {
    auto err_msg =
        fmt::format("Interface port '{}' already exists in Subgraph '{}'", external_name, name_);
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }

  // Resolve the interface port from the nested subgraph to find the actual operator and port
  auto [resolved_op, resolved_port] =
      internal_subgraph->get_interface_operator_port(internal_interface_port);

  if (!resolved_op) {
    // Check if this is an execution interface port to provide a better error message
    const auto& nested_exec_ports = internal_subgraph->exec_interface_ports();
    auto exec_it = nested_exec_ports.find(internal_interface_port);
    if (exec_it != nested_exec_ports.end()) {
      auto err_msg = fmt::format(
          "Cannot add interface port '{}': nested subgraph '{}' has an execution interface port "
          "named '{}', but a data interface port is required. Use add_{}_interface_port() for "
          "data ports or add_{}_exec_interface_port() for execution ports.",
          external_name,
          internal_subgraph->name(),
          internal_interface_port,
          is_input ? "input" : "output",
          is_input ? "input" : "output");
      HOLOSCAN_LOG_ERROR(err_msg);
      throw std::runtime_error(err_msg);
    } else {
      auto err_msg = fmt::format(
          "Cannot add interface port '{}': nested subgraph '{}' does not have interface port '{}'",
          external_name,
          internal_subgraph->name(),
          internal_interface_port);
      HOLOSCAN_LOG_ERROR(err_msg);
      throw std::runtime_error(err_msg);
    }
  }

  // Validate that the resolved port has the correct type
  if (!validate_operator_port(resolved_op, resolved_port, is_input)) {
    auto err_msg = fmt::format(
        "Cannot add interface port '{}': nested subgraph '{}' interface port '{}' resolves to "
        "operator '{}' port '{}' which has incorrect type (expected {})",
        external_name,
        internal_subgraph->name(),
        internal_interface_port,
        resolved_op->name(),
        resolved_port,
        is_input ? "input" : "output");
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }

  // Store the interface port mapping with the resolved operator and port
  interface_ports_[external_name] = InterfacePort{resolved_op, resolved_port, is_input};

  HOLOSCAN_LOG_DEBUG(
      "Added interface port '{}' -> nested subgraph '{}' interface port '{}' (resolves to "
      "'{}:{}', input: {}) to Subgraph '{}'",
      external_name,
      internal_subgraph->name(),
      internal_interface_port,
      resolved_op->name(),
      resolved_port,
      is_input,
      name_);
}

void Subgraph::add_input_interface_port(const std::string& external_name,
                                        const std::shared_ptr<Subgraph>& internal_subgraph,
                                        const std::string& internal_interface_port) {
  add_interface_port(external_name, internal_subgraph, internal_interface_port, true);
}

void Subgraph::add_output_interface_port(const std::string& external_name,
                                         const std::shared_ptr<Subgraph>& internal_subgraph,
                                         const std::string& internal_interface_port) {
  add_interface_port(external_name, internal_subgraph, internal_interface_port, false);
}

std::pair<std::shared_ptr<Operator>, std::string> Subgraph::get_interface_operator_port(
    const std::string& port_name) const {
  // First check local interface ports
  auto it = interface_ports_.find(port_name);
  if (it != interface_ports_.end()) {
    return {it->second.internal_operator, it->second.internal_port_name};
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
  if (it != exec_interface_ports_.end()) {
    return {it->second.internal_operator, it->second.internal_port_name};
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

bool Subgraph::validate_operator_port(std::shared_ptr<Operator> op, const std::string& port_name,
                                      bool expect_input) {
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

bool Subgraph::validate_operator_exec_port(std::shared_ptr<Operator> op) {
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
  port.internal_operator = internal_op;
  port.internal_port_name = internal_port_name;
  port.is_input = is_input;
  port.port_type = InterfacePort::PortType::kExecution;

  exec_interface_ports_[external_name] = port;
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

  HOLOSCAN_LOG_DEBUG(
      "Resolved {} execution interface port '{}' -> nested subgraph '{}' port '{}' -> operator "
      "'{}'",
      expect_input ? "input" : "output",
      external_name,
      internal_subgraph->name(),
      internal_interface_port,
      nested_port.internal_operator->name());

  return {nested_port.internal_operator, nested_port.internal_port_name};
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

void Subgraph::add_input_exec_interface_port(const std::string& external_name,
                                             const std::shared_ptr<Subgraph>& internal_subgraph,
                                             const std::string& internal_interface_port) {
  if (!internal_subgraph) {
    HOLOSCAN_LOG_ERROR("Internal subgraph pointer is null");
    throw std::runtime_error("Internal subgraph pointer is null");
  }

  if (!check_exec_port_name_available(external_name)) {
    return;
  }

  auto [resolved_op, resolved_port] =
      resolve_nested_exec_port(external_name, internal_subgraph, internal_interface_port, true);

  if (!resolved_op) {
    return;
  }

  register_exec_interface_port(external_name, resolved_op, resolved_port, true);
}

void Subgraph::add_output_exec_interface_port(const std::string& external_name,
                                              const std::shared_ptr<Subgraph>& internal_subgraph,
                                              const std::string& internal_interface_port) {
  if (!internal_subgraph) {
    HOLOSCAN_LOG_ERROR("Internal subgraph pointer is null");
    throw std::runtime_error("Internal subgraph pointer is null");
  }

  if (!check_exec_port_name_available(external_name)) {
    return;
  }

  auto [resolved_op, resolved_port] =
      resolve_nested_exec_port(external_name, internal_subgraph, internal_interface_port, false);

  if (!resolved_op) {
    return;
  }

  register_exec_interface_port(external_name, resolved_op, resolved_port, false);
}

}  // namespace holoscan
