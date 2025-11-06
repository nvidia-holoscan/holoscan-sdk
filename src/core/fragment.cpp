/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "holoscan/core/fragment.hpp"

#include <yaml-cpp/yaml.h>

#include <fmt/format.h>

#include <functional>
#include <iterator>  // for std::back_inserter
#include <memory>
#include <mutex>  // for std::call_once
#include <set>
#include <string>
#include <typeinfo>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "holoscan/core/arg.hpp"
#include "holoscan/core/config.hpp"
#include "holoscan/core/dataflow_tracker.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/core/executors/gpu_resident/gpu_resident_executor.hpp"
#include "holoscan/core/executors/gxf/gxf_executor.hpp"
#include "holoscan/core/gpu_resident_operator.hpp"
#include "holoscan/core/graphs/flow_graph.hpp"
#include "holoscan/core/gxf/entity_group.hpp"
#include "holoscan/core/gxf/gxf_network_context.hpp"
#include "holoscan/core/gxf/gxf_scheduler.hpp"
#include "holoscan/core/metadata.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/resources/gxf/system_resources.hpp"
#include "holoscan/core/schedulers/gxf/greedy_scheduler.hpp"
#include "holoscan/core/subgraph.hpp"

using std::string_literals::operator""s;

namespace holoscan {

Fragment::~Fragment() {
  // Set `is_run_called_` to true in case the fragment is being destroyed before
  // run() or run_async() has completed execution, enabling proper cleanup in reset_state().
  is_run_called_ = true;

  // Explicitly clean up graph entities.
  reset_backend_objects();
}

Fragment& Fragment::name(const std::string& name) & {
  if (name == "all") {
    HOLOSCAN_LOG_ERROR("Fragment name 'all' is reserved. Please use another name.");
  }
  name_ = name;
  return *this;
}

Fragment&& Fragment::name(const std::string& name) && {
  if (name == "all") {
    HOLOSCAN_LOG_ERROR("Fragment name 'all' is reserved. Please use another name.");
  }
  name_ = name;
  return std::move(*this);
}

const std::string& Fragment::name() const {
  return name_;
}

Fragment& Fragment::application(Application* app) {
  app_ = app;
  return *this;
}

Application* Fragment::application() const {
  return app_;
}

bool Fragment::is_metadata_enabled() const {
  if (is_metadata_enabled_.has_value()) {
    return is_metadata_enabled_.value();
  } else if (app_ != nullptr) {
    return app_->is_metadata_enabled();
  } else {
    return kDefaultMetadataEnabled;
  }
}

void Fragment::enable_metadata(bool enabled) {
  is_metadata_enabled_ = enabled;
}

void Fragment::is_metadata_enabled(bool enabled) {
  static std::once_flag warn_flag;
  std::call_once(warn_flag, []() {
    HOLOSCAN_LOG_WARN(
        "The Fragment::is_metadata_enabled(bool) setter is deprecated. Please use "
        "Fragment::enable_metadata(bool) instead.");
  });
  is_metadata_enabled_ = enabled;
}

MetadataPolicy Fragment::metadata_policy() const {
  if (metadata_policy_.has_value()) {
    return metadata_policy_.value();
  } else if (app_ != nullptr) {
    return app_->metadata_policy();
  } else {
    return kDefaultMetadataPolicy;
  }
}

void Fragment::metadata_policy(MetadataPolicy policy) {
  metadata_policy_ = policy;
}

void Fragment::add_data_logger(const std::shared_ptr<DataLogger>& logger) {
  data_loggers_.push_back(logger);
}

void Fragment::config(const std::string& config_file, const std::string& prefix) {
  if (config_) {
    HOLOSCAN_LOG_WARN("Config object was already created. Overwriting...");
  }
  if (is_composed_) {
    HOLOSCAN_LOG_WARN(
        "Graph has already been composed. Please make sure that graph composition is not dependent "
        "on this config() call.");
  }

  // If the application is executed with `--config` option or HOLOSCAN_CONFIG_PATH environment,
  // we ignore the config_file argument.
  if (app_) {
    auto& config_path = app_->options().config_path;
    if (!config_path.empty() && config_path.size() > 0) {
      HOLOSCAN_LOG_DEBUG("Configuration path would be overridden by --config option to '{}'",
                         config_path);
      return;
    } else {
      const char* env_value = std::getenv("HOLOSCAN_CONFIG_PATH");
      if (env_value != nullptr && env_value[0] != '\0') {
        HOLOSCAN_LOG_DEBUG(
            "Configuration path would be overridden by HOLOSCAN_CONFIG_PATH "
            "environment variable to '{}'",
            env_value);
        return;
      }
    }
  }

  config_ = make_config<Config>(config_file, prefix);
}

void Fragment::config(std::shared_ptr<Config>& config) {
  if (config_) {
    HOLOSCAN_LOG_WARN("Config object was already created. Overwriting...");
  }
  if (is_composed_) {
    HOLOSCAN_LOG_WARN(
        "Graph has already been composed. Please make sure that graph composition is not dependent "
        "on this config() call.");
  }
  config_ = config;
}

Config& Fragment::config() {
  return *config_shared();
}

std::shared_ptr<Config> Fragment::config_shared() {
  if (!config_) {
    // If the application is executed with `--config` option or HOLOSCAN_CONFIG_PATH environment
    // variable, we take the config file from there.
    if (app_) {
      auto& config_path = app_->options().config_path;
      if (!config_path.empty() && config_path.size() > 0) {
        HOLOSCAN_LOG_DEBUG("Loading config from '{}' (through --config option)", config_path);
        config_ = make_config<Config>(config_path.c_str());
        return config_;
      } else {
        const char* env_value = std::getenv("HOLOSCAN_CONFIG_PATH");
        if (env_value != nullptr && env_value[0] != '\0') {
          HOLOSCAN_LOG_DEBUG(
              "Loading config from '{}' (through HOLOSCAN_CONFIG_PATH environment variable)",
              env_value);
          config_ = make_config<Config>(env_value);
          return config_;
        }
      }
    }
    config_ = make_config<Config>();
  }
  return config_;
}

OperatorGraph& Fragment::graph() {
  return *graph_shared();
}

std::shared_ptr<OperatorGraph> Fragment::graph_shared() {
  if (!graph_) {
    graph_ = make_graph<OperatorFlowGraph>();
  }
  return graph_;
}

void Fragment::executor(const std::shared_ptr<Executor>& executor) {
  executor_ = executor;
}

Executor& Fragment::executor() {
  return *executor_shared();
}

std::shared_ptr<Executor> Fragment::executor_shared() {
  if (!executor_ && !is_gpu_resident_) {
    executor_ = make_executor<gxf::GXFExecutor>();
  } else if (!executor_) {
    executor_ = make_executor<GPUResidentExecutor>();
  }
  return executor_;
}

void Fragment::scheduler(const std::shared_ptr<Scheduler>& scheduler) {
  if (is_gpu_resident_) {
    auto err_msg = fmt::format(
        "Fragment ({}) has a GPU-resident operator. No scheduler is supported for such a fragment.",
        name());
    throw RuntimeError(ErrorCode::kInvalidArgument, err_msg);
  }
  scheduler_ = scheduler;
}

std::shared_ptr<Scheduler> Fragment::scheduler() {
  if (!scheduler_ && !is_gpu_resident_) {
    scheduler_ = make_scheduler<GreedyScheduler>();
  }
  return scheduler_;
}

std::shared_ptr<Scheduler> Fragment::scheduler() const {
  if (!scheduler_ && !is_gpu_resident_) {
    scheduler_ = const_cast<Fragment*>(this)->make_scheduler<GreedyScheduler>();
  }
  return scheduler_;
}

void Fragment::network_context(const std::shared_ptr<NetworkContext>& network_context) {
  if (is_gpu_resident_) {
    auto err_msg = fmt::format(
        "Fragment ({}) has a GPU-resident operator. No network context is supported for such a "
        "fragment.",
        name());
    throw RuntimeError(ErrorCode::kInvalidArgument, err_msg);
  }
  network_context_ = network_context;
}

std::shared_ptr<NetworkContext> Fragment::network_context() {
  return network_context_;
}

namespace {  // anonymous details to avoid polluting holoscan namespace

std::unordered_set<std::string> nested_yaml_map_keys_(YAML::Node yaml_node) {
  std::unordered_set<std::string> keys;
  for (auto it = yaml_node.begin(); it != yaml_node.end(); ++it) {
    const auto& key = it->first.as<std::string>();
    const auto& value = it->second;
    keys.emplace(key);
    if (value.IsMap()) {
      std::unordered_set<std::string> inner_keys = nested_yaml_map_keys_(it->second);
      for (const auto& inner_key : inner_keys) {
        keys.emplace(key + "."s + inner_key);
      }
    }
  }
  return keys;
}

}  // namespace

std::unordered_set<std::string> Fragment::config_keys() {
  auto& yaml_nodes = config().yaml_nodes();

  std::unordered_set<std::string> all_keys;
  for (const auto& yaml_node : yaml_nodes) {
    if (yaml_node.IsMap()) {
      auto node_keys = nested_yaml_map_keys_(yaml_node);
      for (const auto& k : node_keys) {
        all_keys.insert(k);
      }
    }
  }
  return all_keys;
}

ArgList Fragment::from_config(const std::string& key) {
  auto& yaml_nodes = config().yaml_nodes();
  ArgList args;

  std::vector<std::string> key_parts;

  size_t pos = 0;
  while (pos != std::string::npos) {
    size_t next_pos = key.find_first_of('.', pos);
    if (next_pos == std::string::npos) {
      break;
    }
    key_parts.push_back(key.substr(pos, next_pos - pos));
    pos = next_pos + 1;
  }
  key_parts.push_back(key.substr(pos));

  size_t key_parts_size = key_parts.size();

  for (const auto& yaml_node : yaml_nodes) {
    if (yaml_node.IsMap()) {
      auto yaml_map = yaml_node.as<YAML::Node>();
      size_t key_index = 0;
      for (const auto& key_part : key_parts) {
        (void)key_part;
        if (yaml_map.IsMap()) {
          yaml_map.reset(yaml_map[key_part]);
          ++key_index;
        } else {
          break;
        }
      }
      if (!yaml_map || key_index < key_parts_size) {
        HOLOSCAN_LOG_ERROR("Unable to find the parameter item/map with key '{}'", key);
        continue;
      }

      const auto& parameters = yaml_map;

      if (parameters.IsScalar()) {
        const std::string& param_key = key_parts[key_parts_size - 1];
        auto& value = parameters;
        args.add(Arg(param_key) = value);
        continue;
      }

      for (const auto& p : parameters) {
        const std::string param_key = p.first.as<std::string>();
        auto& value = p.second;
        args.add(Arg(param_key) = value);
      }
    }
  }

  return args;
}

bool Fragment::register_service_from(Fragment* fragment, std::string_view id) {
  if (!fragment) {
    return false;
  }

  const std::string id_str{id};
  bool found_any = false;

  // Lock both registries (always lock lower-address first to avoid dead-lock)
  auto* first = this < fragment ? this : fragment;
  auto* second = this < fragment ? fragment : this;

  std::unique_lock<std::shared_mutex> l1(first->fragment_service_registry_mutex_, std::defer_lock);
  std::shared_lock<std::shared_mutex> l2(second->fragment_service_registry_mutex_, std::defer_lock);
  std::lock(l1, l2);

  for (const auto& [service_key, service] : fragment->fragment_services_by_key_) {
    if (service_key.id == id_str) {
      fragment_services_by_key_[service_key] = service;
      found_any = true;
    }
  }

  for (const auto& [resource, service_key] : fragment->fragment_resource_to_service_key_map_) {
    if (service_key.id == id_str) {
      HOLOSCAN_LOG_DEBUG("register_service_from: resource = {} (0x{:x})",
                         resource->name(),
                         reinterpret_cast<uint64_t>(resource.get()));
      fragment_resource_to_service_key_map_.insert_or_assign(resource, service_key);
      fragment_resource_services_by_name_[resource->name()] = resource;
      found_any = true;
    }
  }
  return found_any;
}

const std::shared_ptr<Operator>& Fragment::start_op() {
  if (!start_op_) {
    // According to `GxfEntityCreateInfo` definition, the entity name should not start with
    // a double underscore. So, we use this unique name to avoid any conflicts.
    start_op_ = make_operator<Operator>(kStartOperatorName, make_condition<CountCondition>(1));
    add_operator(start_op_);
  }
  return start_op_;
}

void Fragment::add_operator(const std::shared_ptr<Operator>& op) {
  // try dynamic casting to GPUresident operator
  auto gpu_resident_op = std::dynamic_pointer_cast<holoscan::GPUResidentOperator>(op);
  if (graph().get_nodes().size() > 0 && gpu_resident_op && !is_gpu_resident_) {
    // trying to add GPU-resident operator to a non-GPU-resident fragment
    auto err_msg = fmt::format(
        "Operator ({}) is a GPU-resident operator but the fragment already has non-GPU-resident "
        "operators.",
        op->name());
    throw RuntimeError(ErrorCode::kInvalidArgument, err_msg);
  } else if (gpu_resident_op) {
    is_gpu_resident_ = true;
  } else if (is_gpu_resident_ && gpu_resident_op == nullptr) {
    auto err_msg = fmt::format(
        "Operator ({}) is not a GPU resident operator but the fragment already has a GPU resident "
        "operator. A fragment can not have a mix of GPU-resident and non-GPU-resident operators.",
        op->name());
    throw RuntimeError(ErrorCode::kInvalidArgument, err_msg);
  }
  op->set_self_shared(op);
  graph().add_node(op);
}

void Fragment::add_flow(const std::shared_ptr<Operator>& upstream_op,
                        const std::shared_ptr<Operator>& downstream_op) {
  add_flow(upstream_op, downstream_op, IOSpec::ConnectorType::kDefault);
}

void Fragment::add_flow(const std::shared_ptr<Operator>& upstream_op,
                        const std::shared_ptr<Operator>& downstream_op,
                        std::set<std::pair<std::string, std::string>> port_pairs) {
  add_flow(upstream_op, downstream_op, std::move(port_pairs), IOSpec::ConnectorType::kDefault);
}

void Fragment::add_flow(const std::shared_ptr<Operator>& upstream_op,
                        const std::shared_ptr<Operator>& downstream_op,
                        const IOSpec::ConnectorType connector_type) {
  add_flow(upstream_op, downstream_op, {}, connector_type);
}

void Fragment::add_flow(const std::shared_ptr<Operator>& upstream_op,
                        const std::shared_ptr<Operator>& downstream_op,
                        std::set<std::pair<std::string, std::string>> port_pairs,
                        const IOSpec::ConnectorType connector_type) {
  // Verify that both operators are GPU resident operators or not
  auto upstream_gpu_resident_op =
      std::dynamic_pointer_cast<holoscan::GPUResidentOperator>(upstream_op);
  auto downstream_gpu_resident_op =
      std::dynamic_pointer_cast<holoscan::GPUResidentOperator>(downstream_op);
  if (is_gpu_resident_ &&
      (upstream_gpu_resident_op == nullptr || downstream_gpu_resident_op == nullptr)) {
    auto err_msg = fmt::format(
        "One of the operators ({}) or ({}) is not a GPU resident operator but the fragment already "
        "has a GPU resident operator. A fragment can not have a mix of GPU-resident and "
        "non-GPU-resident operators.",
        upstream_op->name(),
        downstream_op->name());
    throw RuntimeError(ErrorCode::kInvalidArgument, err_msg);
  }

  // if there are already more operators in the fragment but the the fragment is
  // not GPU-resident, then either upstream or downstream operator being
  // GPU-resident should throw an error
  if (graph().get_nodes().size() > 0 && !is_gpu_resident_) {
    if (upstream_gpu_resident_op || downstream_gpu_resident_op) {
      auto err_msg = fmt::format(
          "One of the operators ({}) or ({}) is a GPU resident operator but the fragment already "
          "has non-GPU-resident operators. A fragment can not have a mix of GPU-resident and "
          "non-GPU-resident operators.",
          upstream_op->name(),
          downstream_op->name());
      throw RuntimeError(ErrorCode::kInvalidArgument, err_msg);
    }
  }
  if (upstream_gpu_resident_op && !downstream_gpu_resident_op) {
    auto err_msg = fmt::format(
        "Upstream operator ({}) is a GPU resident operator but downstream operator ({}) is not",
        upstream_op->name(),
        downstream_op->name());
    throw RuntimeError(ErrorCode::kInvalidArgument, err_msg);
  } else if (!upstream_gpu_resident_op && downstream_gpu_resident_op) {
    auto err_msg = fmt::format(
        "Downstream operator ({}) is a GPU resident operator but upstream operator ({}) is not",
        downstream_op->name(),
        upstream_op->name());
    throw RuntimeError(ErrorCode::kInvalidArgument, err_msg);
  } else if (upstream_gpu_resident_op && downstream_gpu_resident_op) {
    is_gpu_resident_ = true;
  }

  if (is_gpu_resident_ && connector_type != IOSpec::ConnectorType::kDefault) {
    auto err_msg = fmt::format(
        "GPU resident connection is only allowed with default connector type."
        "Upstream operator: {}, Downstream operator: {}, Connector "
        "type is: {}. Please use the default connector type.",
        upstream_op->name(),
        downstream_op->name(),
        static_cast<int>(connector_type));
    throw RuntimeError(ErrorCode::kInvalidArgument, err_msg);
  }
  const std::string async_buffer_err_msg =
      "kAsyncBuffer is only allowed between a single input and output port in one-to-one mapping.";
  auto port_map = std::make_shared<OperatorEdgeDataElementType>();

  auto upstream_op_spec = upstream_op->spec();
  if (upstream_op_spec == nullptr) {
    HOLOSCAN_LOG_ERROR("upstream_op_spec is nullptr");
    return;
  }
  auto downstream_op_spec = downstream_op->spec();
  if (downstream_op_spec == nullptr) {
    HOLOSCAN_LOG_ERROR("downstream_op_spec is nullptr");
    return;
  }

  auto& op_outputs = upstream_op_spec->outputs();
  auto& op_inputs = downstream_op_spec->inputs();
  if (port_pairs.empty()) {
    // Check if this is a control flow addition.
    // We also allow control flow addition if the the upstream operator has output ports but the
    // downstream operator has no input ports.
    if (op_inputs.empty() || downstream_op->input_exec_spec()) {
      // Validate control flow prerequisites
      if (!validate_control_flow_prerequisites(upstream_op, downstream_op, connector_type)) {
        return;
      }

      // Create and register the control flow connection
      create_control_flow_connection(upstream_op, downstream_op);
      return;
    }
    if (op_outputs.size() > 1) {
      std::vector<std::string> output_labels;
      for (const auto& [key, _] : op_outputs) {
        output_labels.push_back(key);
      }

      HOLOSCAN_LOG_ERROR(
          "The upstream operator has more than one output port ({}) so mapping should be "
          "specified explicitly!",
          fmt::join(output_labels, ", "));
      return;
    }
    if (op_inputs.size() > 1) {
      std::vector<std::string> input_labels;
      for (const auto& [key, _] : op_inputs) {
        input_labels.push_back(key);
      }
      HOLOSCAN_LOG_ERROR(
          "The downstream operator has more than one input port ({}) so mapping should be "
          "specified "
          "explicitly!",
          fmt::join(input_labels, ", "));
      return;
    }
    port_pairs.emplace("", "");
  }

  std::vector<std::string> output_labels;
  output_labels.reserve(port_pairs.size());

  // Convert port pairs to port map (set<pair<string, string>> -> map<string, set<string>>)
  for (const auto& [key, value] : port_pairs) {
    if (port_map->find(key) == port_map->end()) {
      (*port_map)[key] = std::set<std::string, std::less<>>();
      output_labels.push_back(key);  // maintain the order of output labels
    } else {
      // Key is already there, so a single output port is connected to multiple
      // input ports. This is not allowed for kAsyncBuffer
      if (connector_type == IOSpec::ConnectorType::kAsyncBuffer) {
        // throw a runtime error
        auto err_msg = fmt::format(
            "add_flow failed. The upstream operator ({}) is connected to multiple input ports of "
            "the downstream operator ({}). This is not allowed for "
            "IOSpec::ConnectorType::kAsyncBuffer. {}",
            upstream_op->name(),
            downstream_op->name(),
            async_buffer_err_msg);
        throw RuntimeError(ErrorCode::kInvalidArgument, err_msg);
      }
    }
    (*port_map)[key].insert(value);
  }

  // Verify that the upstream & downstream operators have the input and output ports specified by
  // the port_map
  if (op_outputs.size() == 1 && output_labels.size() != 1) {
    HOLOSCAN_LOG_ERROR(
        "The upstream operator({}) has only one port with label '{}' but port_map "
        "specifies {} labels({}) to the upstream operator's output port",
        upstream_op->name(),
        (*op_outputs.begin()).first,
        output_labels.size(),
        fmt::join(output_labels, ", "));
    return;
  }

  for (const auto& output_label : output_labels) {
    if (op_outputs.find(output_label) == op_outputs.end()) {
      if (op_outputs.size() == 1 && output_labels.size() == 1 && output_label == "") {
        // Set the default output port label
        (*port_map)[op_outputs.begin()->first] = std::move((*port_map)[output_label]);
        port_map->erase(output_label);
        // Update the output label
        output_labels[0] = op_outputs.begin()->first;
        break;
      }
      if (op_outputs.empty()) {
        HOLOSCAN_LOG_ERROR(
            "The upstream operator({}) does not have any output port but '{}' was specified in "
            "port_map",
            upstream_op->name(),
            output_label);
        return;
      }

      auto msg_buf = fmt::memory_buffer();
      for (const auto& [label, _] : op_outputs) {
        if (&label == &(op_outputs.begin()->first)) {
          fmt::format_to(std::back_inserter(msg_buf), "{}", label);
        } else {
          fmt::format_to(std::back_inserter(msg_buf), ", {}", label);
        }
      }
      HOLOSCAN_LOG_ERROR(
          "The upstream operator({}) does not have an output port with label '{}'. It should be "
          "one of ({:.{}})",
          upstream_op->name(),
          output_label,
          msg_buf.data(),
          msg_buf.size());
      return;
    }
  }

  for (const auto& output_label : output_labels) {
    auto& input_labels = (*port_map)[output_label];
    if (op_inputs.size() == 1 && input_labels.size() != 1) {
      HOLOSCAN_LOG_ERROR(
          "The downstream operator({}) has only one port with label '{}' but port_map "
          "specifies {} labels({}) to the downstream operator's input port",
          downstream_op->name(),
          (*op_inputs.begin()).first,
          input_labels.size(),
          fmt::join(input_labels, ", "));
      return;
    }

    // Create a vector to maintain the final input labels
    std::vector<std::string> new_input_labels;
    new_input_labels.reserve(input_labels.size());

    for (auto& input_label : input_labels) {
      auto op_input_iter = op_inputs.find(input_label);
      bool is_receivers = false;
      if (op_input_iter != op_inputs.end()) {
        auto& op_input_iospec = op_input_iter->second;
        if (op_input_iospec->queue_size() == static_cast<int64_t>(IOSpec::kAnySize)) {
          is_receivers = true;
        }
      }
      if (is_receivers && is_gpu_resident_) {
        auto err_msg = fmt::format(
            "GPU resident connection is not allowed with receivers. Please check the downstream "
            "operator ({})'s input port ({}).",
            downstream_op->name(),
            input_label);
        throw RuntimeError(ErrorCode::kInvalidArgument, err_msg);
      }

      if (is_receivers || op_input_iter == op_inputs.end()) {
        auto input_receivers_label = input_label;
        if (!is_receivers && op_inputs.size() == 1 && input_labels.size() == 1 &&
            input_label == "") {
          // Set the default input port label if the downstream operator has only one input port,
          // the input label is empty, and the queue size is not 'kAnySize'.
          if (op_inputs.begin()->second->queue_size() != static_cast<int64_t>(IOSpec::kAnySize)) {
            new_input_labels.push_back(op_inputs.begin()->first);
            break;
          } else {
            // Set the input_receivers_label to the default input port label
            input_receivers_label = op_inputs.begin()->first;
          }
        }

        // Support for the case where the destination input port label points to the
        // parameter name of the downstream operator, and the parameter type is
        // 'std::vector<holoscan::IOSpec*>'.
        // If we cannot find the input port with the specified label (e.g., 'receivers'),
        // then we need to find a downstream operator's parameter
        // (with 'std::vector<holoscan::IOSpec*>' type) and create a new input port
        // with a specific label ('<parameter name>:<index>'. e.g, 'receivers:0').
        auto& downstream_op_params = downstream_op_spec->params();
        if (downstream_op_params.find(input_receivers_label) != downstream_op_params.end()) {
          auto& downstream_op_param = downstream_op_params.at(input_receivers_label);
          if (std::type_index(downstream_op_param.type()) ==
              std::type_index(typeid(std::vector<holoscan::IOSpec*>))) {
            const std::any& any_value = downstream_op_param.value();
            auto& param = *std::any_cast<Parameter<std::vector<holoscan::IOSpec*>>*>(any_value);
            param.set_default_value();

            std::vector<holoscan::IOSpec*>& iospec_vector = param.get();

            // Create a new input port for this receivers parameter
            bool succeed = executor().add_receivers(
                downstream_op, input_receivers_label, new_input_labels, iospec_vector);
            if (!succeed) {
              HOLOSCAN_LOG_ERROR(
                  "Failed to add receivers to the downstream operator({}) with label '{}'",
                  downstream_op->name(),
                  input_receivers_label);
              return;
            }
            continue;
          }
        }
        if (op_inputs.empty()) {
          HOLOSCAN_LOG_ERROR(
              "The downstream operator({}) does not have any input port but '{}' was "
              "specified in the port_map",
              downstream_op->name(),
              input_receivers_label);
          return;
        }

        auto msg_buf = fmt::memory_buffer();
        for (const auto& [label, _] : op_inputs) {
          if (&label == &(op_inputs.begin()->first)) {
            fmt::format_to(std::back_inserter(msg_buf), "{}", label);
          } else {
            fmt::format_to(std::back_inserter(msg_buf), ", {}", label);
          }
        }
        HOLOSCAN_LOG_ERROR(
            "The downstream operator({}) does not have an input port with label '{}'. It should "
            "be one of ({:.{}})",
            downstream_op->name(),
            input_receivers_label,
            msg_buf.data(),
            msg_buf.size());
        return;
      }

      // Insert the input label as it is to the new_input_labels
      new_input_labels.push_back(input_label);
    }

    // Update the input labels with new_input_labels
    input_labels.clear();
    // Move the new_input_labels to input_labels
    using str_iter = std::vector<std::string>::iterator;
    input_labels.insert(std::move_iterator<str_iter>(new_input_labels.begin()),
                        std::move_iterator<str_iter>(new_input_labels.end()));
    new_input_labels.clear();
    // Do not use 'new_input_labels' after this point

    // check the outdegree of the upstream operator's output port
    // and do not allow more than one connection if is_gpu_resident_ is true
    if (is_gpu_resident_ && graph().get_outdegree(upstream_op, output_label) > 0) {
      auto err_msg = fmt::format(
          "GPU resident connection is not allowed from one operator to multiple operators. Please "
          "check the upstream operator ({})'s output port ({}).",
          upstream_op->name(),
          output_label);
      throw RuntimeError(ErrorCode::kInvalidArgument, err_msg);
    }

    // Check if the output port already has a connector type of kAsyncBuffer
    if (op_outputs[output_label]->connector_type() == IOSpec::ConnectorType::kAsyncBuffer) {
      auto err_msg = fmt::format(
          "add_flow failed. The upstream operator ({})'s output port '{}' is already connected "
          "via IOSpec::ConnectorType::kAsyncBuffer. "
          "Connecting the output port to other input ports ({}) is not allowed. {}",
          upstream_op->name(),
          output_label,
          fmt::join(input_labels, ", "),
          async_buffer_err_msg);
      throw RuntimeError(ErrorCode::kInvalidArgument, err_msg);
    }

    // Check if the upstream operator's output port is
    // already connected to other input ports.
    if (connector_type == IOSpec::ConnectorType::kAsyncBuffer &&
        graph().get_outdegree(upstream_op, output_label) > 0) {
      auto err_msg = fmt::format(
          "add_flow failed. The upstream operator ({})'s output port '{}' is already connected to "
          "other input ports. It cannot be connected to the downstream operator ({})'s input port "
          "'{}' with the connector type IOSpec::ConnectorType::kAsyncBuffer. {}",
          upstream_op->name(),
          output_label,
          downstream_op->name(),
          *(input_labels.begin()),
          async_buffer_err_msg);
      throw RuntimeError(ErrorCode::kInvalidArgument, err_msg);
    }

    if (connector_type == IOSpec::ConnectorType::kAsyncBuffer) {
      // if the connector type is kAsyncBuffer, then both the upstream and
      // downstream operators must be updated with the
      // IOSpec::ConnectorType::kAsyncBuffer for the connector type.
      // update the upstream operator's output port with the kAsyncBuffer connector type
      if (op_outputs[output_label]->connector_type() != IOSpec::ConnectorType::kDefault &&
          op_outputs[output_label]->connector_type() != IOSpec::ConnectorType::kAsyncBuffer) {
        auto err_msg = fmt::format(
            "add_flow failed. The upstream operator ({})'s output port '{}' is already assigned a "
            "non-default and non-(async lock-free buffer) connector. Please check the connector "
            "type "
            "of output port '{}' of {} operator. Currently connected connector type is {}.",
            upstream_op->name(),
            output_label,
            output_label,
            upstream_op->name(),
            (unsigned int)op_outputs[output_label]->connector_type());
        throw RuntimeError(ErrorCode::kInvalidArgument, err_msg);
      }
      if (op_inputs[*(input_labels.begin())]->connector_type() != IOSpec::ConnectorType::kDefault &&
          op_inputs[*(input_labels.begin())]->connector_type() !=
              IOSpec::ConnectorType::kAsyncBuffer) {
        auto err_msg = fmt::format(
            "add_flow failed. The downstream operator ({})'s input port '{}' is already assigned a "
            "non-default and non-(async lock-free buffer) connector. Please check the connector "
            "type "
            "of input port '{}' of {} operator. Currently connected connector type is {}.",
            downstream_op->name(),
            *(input_labels.begin()),
            *(input_labels.begin()),
            downstream_op->name(),
            (unsigned int)op_inputs[*(input_labels.begin())]->connector_type());
        throw RuntimeError(ErrorCode::kInvalidArgument, err_msg);
      }
      op_outputs[output_label]->connector(IOSpec::ConnectorType::kAsyncBuffer);
      // update the downstream operator's input port with the kAsyncBuffer connector type
      op_inputs[*(input_labels.begin())]->connector(IOSpec::ConnectorType::kAsyncBuffer);
    }
  }

  upstream_op->set_self_shared(upstream_op);
  downstream_op->set_self_shared(downstream_op);
  if (is_gpu_resident_) {
    if (!verify_gpu_resident_connections(upstream_op, downstream_op, port_map)) {
      throw RuntimeError(ErrorCode::kInvalidArgument,
                         fmt::format("Fragment '{}': Input/output memory block size configuration "
                                     "error for GPU-resident execution.",
                                     name()));
    }
  }
  graph().add_flow(upstream_op, downstream_op, port_map);
}

bool Fragment::verify_gpu_resident_connections(
    const std::shared_ptr<Operator>& upstream_op, const std::shared_ptr<Operator>& downstream_op,
    const std::shared_ptr<OperatorEdgeDataElementType> port_map) {
  auto upstream_op_spec = upstream_op->spec();
  auto downstream_op_spec = downstream_op->spec();
  for (const auto& [source_port, target_ports] : *port_map) {
    // we know one to one connection
    auto target_port = *(target_ports.begin());
    auto upstream_memory_block_size = upstream_op_spec->outputs()[source_port]->memory_block_size();
    auto downstream_memory_block_size =
        downstream_op_spec->inputs()[target_port]->memory_block_size();
    if (upstream_memory_block_size == 0 || downstream_memory_block_size == 0 ||
        (upstream_memory_block_size != downstream_memory_block_size)) {
      HOLOSCAN_LOG_ERROR(
          "Memory block sizes between upstream ({}) and downstream ({}) operators are not "
          "configured properly. Upstream output port: '{}', memory block size: {}, "
          "downstream input port: '{}', memory block size: {}",
          upstream_op->name(),
          downstream_op->name(),
          source_port,
          upstream_memory_block_size,
          target_port,
          downstream_memory_block_size);
      return false;
    }
  }
  return true;
}

void Fragment::set_dynamic_flows(
    const std::shared_ptr<Operator>& op,
    const std::function<void(const std::shared_ptr<Operator>&)>& dynamic_flow_func) {
  if (op) {
    op->set_dynamic_flows(dynamic_flow_func);
  }
}

void Fragment::compose() {}

void Fragment::run() {
  // Initialize clean state to ensure proper execution and support multiple consecutive runs
  reset_state();

  executor().run(graph());
  is_run_called_ = true;
}

std::future<void> Fragment::run_async() {
  // Initialize clean state to ensure proper execution and support multiple consecutive runs
  reset_state();

  auto future = executor().run_async(graph());
  is_run_called_ = true;
  return future;
}

holoscan::DataFlowTracker& Fragment::track(uint64_t num_start_messages_to_skip,
                                           uint64_t num_last_messages_to_discard,
                                           int latency_threshold, bool is_limited_tracking) {
  if (is_gpu_resident_) {
    auto err_msg = fmt::format(
        "GPU resident fragment cannot have data flow tracking. Please check the fragment ({}).",
        name());
    throw RuntimeError(ErrorCode::kInvalidArgument, err_msg);
  }
  if (!data_flow_tracker_) {
    data_flow_tracker_ = std::make_shared<holoscan::DataFlowTracker>();
    data_flow_tracker_->set_skip_starting_messages(num_start_messages_to_skip);
    data_flow_tracker_->set_discard_last_messages(num_last_messages_to_discard);
    data_flow_tracker_->set_skip_latencies(latency_threshold);
    data_flow_tracker_->set_limited_tracking(is_limited_tracking);
  }
  return *data_flow_tracker_;
}

void Fragment::compose_graph() {
  if (is_composed_) {
    HOLOSCAN_LOG_DEBUG("The fragment({}) has already been composed. Skipping...", name());
    return;
  }

  // Load extensions from the config file before composing the graph.
  // (The GXFCodeletOp and GXFComponentResource classes are required to access the underlying GXF
  //  types in the setup() method when composing a graph.)
  load_extensions_from_config();
  compose();
  is_composed_ = true;

  // Protect against the case where no add_operator or add_flow calls were made
  if (!graph_) {
    HOLOSCAN_LOG_ERROR(fmt::format(
        "Fragment '{}' does not have any operators. Please check that there is at least one call to"
        "`add_operator` or `add_flow` during `Fragment::compose`.",
        name()));
    graph();
  }
}

FragmentPortMap Fragment::port_info() const {
  HOLOSCAN_LOG_TRACE("getting port info for fragment: {}", name_);
  FragmentPortMap fragment_port_info;
  if (!is_composed_ || !graph_) {
    HOLOSCAN_LOG_ERROR("The fragment and its graph must be composed before calling port_info");
    return fragment_port_info;
  }
  std::vector<OperatorNodeType> operators = graph_->get_nodes();
  for (auto& op : operators) {
    HOLOSCAN_LOG_TRACE("\toperator: {}", name_, op->name());
    OperatorSpec* op_spec = op->spec();

    // set of input port names
    std::unordered_set<std::string> input_names;
    input_names.reserve(op_spec->inputs().size());
    for (auto& in : op_spec->inputs()) {
      input_names.insert(in.first);
    }
    HOLOSCAN_LOG_TRACE("\t\tadded {} inputs", input_names.size());

    // set of output port names
    std::unordered_set<std::string> output_names;
    output_names.reserve(op_spec->outputs().size());
    for (auto& out : op_spec->outputs()) {
      output_names.insert(out.first);
    }
    HOLOSCAN_LOG_TRACE("\t\tadded {} outputs", output_names.size());

    // set of multi-receiver parameter names
    std::unordered_set<std::string> receiver_names;
    for (auto& [param_name, param] : op_spec->params()) {
      // receivers parameter type is 'std::vector<holoscan::IOSpec*>'.
      if (std::type_index(param.type()) ==
          std::type_index(typeid(std::vector<holoscan::IOSpec*>))) {
        receiver_names.insert(param_name);
      }
    }
    HOLOSCAN_LOG_TRACE("\t\tadded {} receivers", receiver_names.size());

    fragment_port_info.try_emplace(
        op->name(), std::move(input_names), std::move(output_names), std::move(receiver_names));
  }
  return fragment_port_info;
}

void Fragment::stop_execution(const std::string& op_name) {
  if (!op_name.empty()) {
    // Stop the execution of the operator with the given name
    auto op = graph().find_node(op_name);
    if (op) {
      op->stop_execution();
    } else {
      HOLOSCAN_LOG_WARN("Operator with name '{}' not found, no operator was stopped", op_name);
    }
  } else {
    // Stop the execution of all operators in the fragment in the order they were added to the
    // graph.
    // (`get_nodes()` returns the nodes in the order they were added to the graph.)
    // If needed, we can use more sophisticated termination logic to stop the operators
    // (e.g., monitoring the operator statuses and stopping the operators when they are finished).
    for (auto& op : graph().get_nodes()) {
      op->stop_execution();
    }
  }
}

void Fragment::shutdown_data_loggers() {
  HOLOSCAN_LOG_DEBUG("Fragment '{}': Starting data logger shutdown", name());
  // Explicitly stop background threads and shutdown resources before clearing the vector
  // This ensures proper cleanup even if external shared_ptr references exist
  for (auto& data_logger : data_loggers_) {
    if (!data_logger)
      continue;
    try {
      data_logger->shutdown();
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR(
          "Fragment '{}': Exception during data logger shutdown: {}", name(), e.what());
    }
  }
  data_loggers_.clear();
}

void Fragment::reset_backend_objects() {
  for (auto& op : graph().get_nodes()) {
    if (op) {
      op->reset_backend_objects();
    }
  }
  if (scheduler()) {
    scheduler()->reset_backend_objects();
  }
  if (network_context()) {
    network_context()->reset_backend_objects();
  }
}

void Fragment::reset_state() {
  if (!is_run_called_) {
    HOLOSCAN_LOG_DEBUG(
        "skipping fragment state reset since run() or run_async() was not called yet");
    return;
  }

  // First clean up any backend-specific objects (e.g. GXF GraphEntity objects)
  reset_backend_objects();

  // If this has a GXFExecutor, we need to reset its flags
  auto gxf_executor = std::dynamic_pointer_cast<gxf::GXFExecutor>(executor_);
  if (gxf_executor) {
    // Reset the execution state (flags for graph initialization and activation)
    gxf_executor->reset_execution_state();
  }

  // Skip resetting the executor here as it needs to remain accessible before
  // run()/run_async() calls. The executor object will be shared between run() method calls.
  // executor_.reset();  // DO NOT RESET THIS.

  // Reset the graph to recreate it on the next run
  graph_.reset();
  subgraph_instance_names_.clear();

  // Skip resetting the scheduler since it is shared between run() method calls.
  // scheduler_.reset();  // DO NOT RESET THIS.

  // Skip resetting the network context since it is shared between run() method calls.
  // network_context_.reset();  // DO NOT RESET THIS.

  // Skip resetting data_flow_tracker as its lifecycle is managed outside the run() method.
  // data_flow_tracker_.reset();  // DO NOT RESET THIS.

  // Clear thread pools to prevent memory leaks
  thread_pools_.clear();

  // Clear service registry to prevent memory leaks and stale references
  {
    std::unique_lock<std::shared_mutex> lock(fragment_service_registry_mutex_);
    fragment_services_by_key_.clear();
    fragment_resource_services_by_name_.clear();
    fragment_resource_to_service_key_map_.clear();
    HOLOSCAN_LOG_DEBUG("Cleared service registry/resources for fragment '{}'", name_);
  }

  // Reset the is_composed_ flag to ensure graphs are recomposed
  is_composed_ = false;
}

void Fragment::load_extensions_from_config() {
  HOLOSCAN_LOG_INFO("Loading extensions from configs...");
  // Load any extensions that may be present in the config file
  for (const auto& yaml_node : config().yaml_nodes()) {
    executor().extension_manager()->load_extensions_from_yaml(yaml_node);
  }
}

/**
 * @brief Create a new thread pool resource.
 *
 * @param name The name of the resource.
 * @param args The arguments for the resource.
 * @return The shared pointer to the resource.
 */
std::shared_ptr<ThreadPool> Fragment::make_thread_pool(const std::string& name,
                                                       int64_t initial_size) {
  // Create a dedicated GXF Entity for the ThreadPool
  // (unlike a typical Condition/Resource, it does not belong to the same entity as an operator)
  auto pool_entity = std::make_shared<nvidia::gxf::GraphEntity>();
  auto pool_entity_name = fmt::format("{}_{}_entity", this->name(), name);
  auto maybe_pool = pool_entity->setup(executor().context(), pool_entity_name.c_str());
  if (!maybe_pool) {
    throw std::runtime_error(
        fmt::format("Failed to create thread pool entity: '{}'", pool_entity_name));
  }

  // Create the ThreadPool resource
  auto pool_resource = make_resource<ThreadPool>(name, holoscan::Arg("initial_size", initial_size));

  // Assign the pool to the entity that was created above and initialize it via add_to_graph_entity
  pool_resource->gxf_eid(pool_entity->eid());
  pool_resource->add_to_graph_entity(this, pool_entity);

  auto pool_group = std::make_shared<gxf::EntityGroup>(executor().context(),
                                                       fmt::format("{}_group", pool_entity_name));
  pool_resource->entity_group(std::move(pool_group));

  // Add this ThreadPool into the entity group
  pool_resource->entity_group()->add(*pool_resource);

  // Store pointers to all thread pools so initialization of entity groups can be
  // performed later by GXFExecutor. We can only add operators to the entity group AFTER they have
  // been initialized in GXFExecutor.
  thread_pools_.push_back(pool_resource);

  return pool_resource;
}

std::shared_ptr<FragmentService> Fragment::get_service_erased(const std::type_info& service_type,
                                                              std::string_view id) const {
  std::shared_lock<std::shared_mutex> lock(fragment_service_registry_mutex_);
  ServiceKey key{service_type, std::string(id)};
  auto it = fragment_services_by_key_.find(key);
  if (it == fragment_services_by_key_.end()) {
    HOLOSCAN_LOG_DEBUG(
        "Service (erased) for type_info {} id '{}' not found", service_type.name(), id);
    return nullptr;
  }
  return it->second;
}

void Fragment::setup_component_internals(ComponentBase* component) {
  if (component) {
    // 'this' (Fragment instance) is both the Fragment and the FragmentServiceProvider
    component->fragment(this);
    component->service_provider(this);
    // Potentially other common setup tasks for components could go here.
  }
}

std::shared_ptr<CudaGreenContextPool> Fragment::add_default_green_context_pool(
    int32_t dev_id, std::vector<uint32_t> sms_per_partition, int32_t default_context_index,
    uint32_t min_sm_size) {
  if (green_context_pools_.size() > 0) {
    HOLOSCAN_LOG_WARN("Fragment '{}': a CudaGreenContextPool already exists. Skipping...");
    return green_context_pools_.back();
  }

  // Create the CudaGreenContextPool resource
  auto green_context_pool =
      make_resource<CudaGreenContextPool>("fragment_default_green_context_pool",
                                          dev_id,
                                          0,
                                          sms_per_partition.size(),
                                          sms_per_partition,
                                          default_context_index,
                                          min_sm_size);
  if (!green_context_pool) {
    HOLOSCAN_LOG_ERROR("Failed to create fragment default green context pool");
    return nullptr;
  }

  HOLOSCAN_LOG_DEBUG("green_context_pool: {}", static_cast<void*>(green_context_pool.get()));
  green_context_pools_.push_back(green_context_pool);
  return green_context_pool;
}

std::shared_ptr<CudaGreenContextPool> Fragment::get_default_green_context_pool() {
  if (green_context_pools_.empty()) {
    HOLOSCAN_LOG_DEBUG("No default green context pool found for fragment '{}'", name());
    return nullptr;
  }
  return green_context_pools_.back();
}

std::shared_ptr<GPUResidentExecutor> Fragment::get_gpu_resident_executor(const char* func_name) {
  auto executor = executor_shared();
  auto gpu_resident_executor = std::dynamic_pointer_cast<holoscan::GPUResidentExecutor>(executor);
  if (!gpu_resident_executor) {
    auto err_msg = fmt::format(
        "Fragment '{}': Failed to cast executor to GPUResidentExecutor in {}", name(), func_name);
    throw RuntimeError(ErrorCode::kInvalidArgument, err_msg);
  }
  return gpu_resident_executor;
}

Fragment::GPUResidentAccessor Fragment::gpu_resident() {
  if (!is_gpu_resident()) {
    auto err_msg = fmt::format(
        "Fragment '{}': Cannot access GPU-resident functions because the fragment does not have "
        "GPU-resident operators",
        name());
    throw RuntimeError(ErrorCode::kInvalidArgument, err_msg);
  }
  // C++ will optimize so that explicit copy is not needed unless required.
  return GPUResidentAccessor(this);
}

// Fragment::GPUResidentAccessor method implementations

void Fragment::GPUResidentAccessor::timeout_ms(unsigned long long timeout_ms) {
  auto gpu_resident_executor = fragment_->get_gpu_resident_executor(__func__);
  gpu_resident_executor->timeout_ms(timeout_ms);
}

void Fragment::GPUResidentAccessor::tear_down() {
  auto gpu_resident_executor = fragment_->get_gpu_resident_executor(__func__);
  gpu_resident_executor->tear_down();
}

bool Fragment::GPUResidentAccessor::result_ready() {
  auto gpu_resident_executor = fragment_->get_gpu_resident_executor(__func__);
  return gpu_resident_executor->result_ready();
}

void Fragment::GPUResidentAccessor::data_ready() {
  auto gpu_resident_executor = fragment_->get_gpu_resident_executor(__func__);
  gpu_resident_executor->data_ready();
}

bool Fragment::GPUResidentAccessor::is_launched() {
  auto gpu_resident_executor = fragment_->get_gpu_resident_executor(__func__);
  return gpu_resident_executor->is_launched();
}

// ========== Helper functions for port auto-resolution ==========

std::vector<std::string> Fragment::get_operator_output_ports(const std::shared_ptr<Operator>& op) {
  std::vector<std::string> output_ports;
  auto op_spec = op->spec();
  if (op_spec) {
    auto& op_outputs = op_spec->outputs();
    for (const auto& [key, _] : op_outputs) {
      output_ports.push_back(key);
    }
  }
  return output_ports;
}

std::vector<std::string> Fragment::get_operator_input_ports(const std::shared_ptr<Operator>& op) {
  std::vector<std::string> input_ports;
  auto op_spec = op->spec();
  if (op_spec) {
    auto& op_inputs = op_spec->inputs();
    for (const auto& [key, _] : op_inputs) {
      input_ports.push_back(key);
    }
  }
  return input_ports;
}

std::vector<std::string> Fragment::get_subgraph_output_ports(
    const std::shared_ptr<Subgraph>& subgraph) {
  std::vector<std::string> output_ports;
  for (const auto& [port_name, interface_port] : subgraph->interface_ports()) {
    if (!interface_port.is_input) {
      output_ports.push_back(port_name);
    }
  }
  return output_ports;
}

std::vector<std::string> Fragment::get_subgraph_input_ports(
    const std::shared_ptr<Subgraph>& subgraph) {
  std::vector<std::string> input_ports;
  for (const auto& [port_name, interface_port] : subgraph->interface_ports()) {
    if (interface_port.is_input) {
      input_ports.push_back(port_name);
    }
  }
  return input_ports;
}

void Fragment::try_auto_resolve_ports(const std::vector<std::string>& upstream_ports,
                                      const std::vector<std::string>& downstream_ports,
                                      const std::string& upstream_name,
                                      const std::string& downstream_name,
                                      std::set<std::pair<std::string, std::string>>& port_pairs) {
  // Check if auto-connection is possible
  if (upstream_ports.empty()) {
    const std::string err_msg =
        fmt::format("Cannot auto-connect '{}' to '{}': upstream has no output ports",
                    upstream_name,
                    downstream_name);
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }

  if (downstream_ports.empty()) {
    const std::string err_msg =
        fmt::format("Cannot auto-connect '{}' to '{}': downstream has no input ports",
                    upstream_name,
                    downstream_name);
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }

  if (upstream_ports.size() > 1) {
    const std::string err_msg = fmt::format(
        "Cannot auto-connect '{}' to '{}': upstream has more than one "
        "output port ({}), so port mapping must be specified explicitly",
        upstream_name,
        downstream_name,
        fmt::join(upstream_ports, ", "));
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }

  if (downstream_ports.size() > 1) {
    const std::string err_msg = fmt::format(
        "Cannot auto-connect '{}' to '{}': downstream has more than "
        "one input port ({}), so port mapping must be specified explicitly",
        upstream_name,
        downstream_name,
        fmt::join(downstream_ports, ", "));
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }

  // Auto-connection is possible: create port mapping
  port_pairs.emplace(upstream_ports[0], downstream_ports[0]);
}

// ========== Helper functions for control flow connections ==========

bool Fragment::validate_control_flow_prerequisites(const std::shared_ptr<Operator>& upstream_op,
                                                   const std::shared_ptr<Operator>& downstream_op,
                                                   const IOSpec::ConnectorType connector_type) {
  // Check if both operators are Native type
  if (upstream_op->operator_type() != Operator::OperatorType::kNative ||
      downstream_op->operator_type() != Operator::OperatorType::kNative) {
    HOLOSCAN_LOG_ERROR(
        "Both upstream ('{}', type: {}) and downstream ('{}', type: {}) operators must be "
        "Native operators for control flow connections",
        upstream_op->name(),
        upstream_op->operator_type() == Operator::OperatorType::kNative ? "Native" : "GXF",
        downstream_op->name(),
        downstream_op->operator_type() == Operator::OperatorType::kNative ? "Native" : "GXF");
    return false;
  }

  // Check for async buffer connector type
  if (connector_type == IOSpec::ConnectorType::kAsyncBuffer) {
    HOLOSCAN_LOG_ERROR(
        "Execution port connection cannot be added with IOSpec::ConnectorType::kAsyncBuffer");
    return false;
  }

  return true;
}

void Fragment::create_control_flow_connection(const std::shared_ptr<Operator>& upstream_op,
                                              const std::shared_ptr<Operator>& downstream_op) {
  // Create the control flow port mapping
  auto port_map = std::make_shared<OperatorEdgeDataElementType>();
  (*port_map)[Operator::kOutputExecPortName] = {Operator::kInputExecPortName};

  // Set self_shared on both operators
  upstream_op->set_self_shared(upstream_op);
  downstream_op->set_self_shared(downstream_op);

  // Add to graph and register with executor
  graph().add_flow(upstream_op, downstream_op, port_map);
  executor().add_control_flow(upstream_op, downstream_op);
}

void Fragment::resolve_and_create_op_to_subgraph_flows(
    const std::shared_ptr<Operator>& upstream_op,
    const std::shared_ptr<Subgraph>& downstream_subgraph,
    const std::set<std::pair<std::string, std::string>>& port_pairs,
    const IOSpec::ConnectorType connector_type) {
  // Resolve subgraph interface ports to actual operators
  for (const auto& [upstream_port, interface_port] : port_pairs) {
    // Try data ports first
    auto [downstream_op, downstream_port] =
        downstream_subgraph->get_interface_operator_port(interface_port);

    // If not found in data ports, try exec ports
    if (!downstream_op) {
      std::tie(downstream_op, downstream_port) =
          downstream_subgraph->get_exec_interface_operator_port(interface_port);
    }

    if (!downstream_op) {
      auto err_msg = fmt::format("Interface port '{}' not found in Subgraph '{}'",
                                 interface_port,
                                 downstream_subgraph->instance_name());
      HOLOSCAN_LOG_ERROR(err_msg);
      throw std::runtime_error(err_msg);
    }

    // Check if this is a control flow connection (exec port)
    if (upstream_port == Operator::kOutputExecPortName &&
        downstream_port == Operator::kInputExecPortName) {
      // This is a control flow connection
      if (!validate_control_flow_prerequisites(upstream_op, downstream_op, connector_type)) {
        continue;
      }
      create_control_flow_connection(upstream_op, downstream_op);
    } else {
      // Regular data flow connection
      add_flow(upstream_op, downstream_op, {{upstream_port, downstream_port}}, connector_type);
    }
  }
}

void Fragment::resolve_and_create_subgraph_to_op_flows(
    const std::shared_ptr<Subgraph>& upstream_subgraph,
    const std::shared_ptr<Operator>& downstream_op,
    const std::set<std::pair<std::string, std::string>>& port_pairs,
    const IOSpec::ConnectorType connector_type) {
  // Resolve subgraph interface ports to actual operators
  for (const auto& [interface_port, downstream_port] : port_pairs) {
    // Try data ports first
    auto [upstream_op, upstream_port] =
        upstream_subgraph->get_interface_operator_port(interface_port);

    // If not found in data ports, try exec ports
    if (!upstream_op) {
      std::tie(upstream_op, upstream_port) =
          upstream_subgraph->get_exec_interface_operator_port(interface_port);
    }

    if (!upstream_op) {
      auto err_msg = fmt::format("Interface port '{}' not found in Subgraph '{}'",
                                 interface_port,
                                 upstream_subgraph->instance_name());
      HOLOSCAN_LOG_ERROR(err_msg);
      throw std::runtime_error(err_msg);
    }

    // Check if this is a control flow connection (exec port)
    if (upstream_port == Operator::kOutputExecPortName &&
        downstream_port == Operator::kInputExecPortName) {
      // This is a control flow connection
      if (!validate_control_flow_prerequisites(upstream_op, downstream_op, connector_type)) {
        continue;
      }
      create_control_flow_connection(upstream_op, downstream_op);
    } else {
      // Regular data flow connection
      add_flow(upstream_op, downstream_op, {{upstream_port, downstream_port}}, connector_type);
    }
  }
}

void Fragment::resolve_and_create_subgraph_to_subgraph_flows(
    const std::shared_ptr<Subgraph>& upstream_subgraph,
    const std::shared_ptr<Subgraph>& downstream_subgraph,
    const std::set<std::pair<std::string, std::string>>& port_pairs,
    const IOSpec::ConnectorType connector_type) {
  // Resolve both subgraph interface ports to actual operators
  for (const auto& [upstream_interface_port, downstream_interface_port] : port_pairs) {
    // Try data ports first for upstream
    auto [upstream_op, upstream_port] =
        upstream_subgraph->get_interface_operator_port(upstream_interface_port);

    // If not found in data ports, try exec ports
    if (!upstream_op) {
      std::tie(upstream_op, upstream_port) =
          upstream_subgraph->get_exec_interface_operator_port(upstream_interface_port);
    }

    // Try data ports first for downstream
    auto [downstream_op, downstream_port] =
        downstream_subgraph->get_interface_operator_port(downstream_interface_port);

    // If not found in data ports, try exec ports
    if (!downstream_op) {
      std::tie(downstream_op, downstream_port) =
          downstream_subgraph->get_exec_interface_operator_port(downstream_interface_port);
    }

    if (!upstream_op) {
      auto err_msg = fmt::format("Interface port '{}' not found in upstream Subgraph '{}'",
                                 upstream_interface_port,
                                 upstream_subgraph->instance_name());
      HOLOSCAN_LOG_ERROR(err_msg);
      throw std::runtime_error(err_msg);
    }

    if (!downstream_op) {
      auto err_msg = fmt::format("Interface port '{}' not found in downstream Subgraph '{}'",
                                 downstream_interface_port,
                                 downstream_subgraph->instance_name());
      HOLOSCAN_LOG_ERROR(err_msg);
      throw std::runtime_error(err_msg);
    }

    // Check if this is a control flow connection (exec port)
    if (upstream_port == Operator::kOutputExecPortName &&
        downstream_port == Operator::kInputExecPortName) {
      // This is a control flow connection
      if (!validate_control_flow_prerequisites(upstream_op, downstream_op, connector_type)) {
        continue;
      }
      create_control_flow_connection(upstream_op, downstream_op);
    } else {
      // Regular data flow connection
      add_flow(upstream_op, downstream_op, {{upstream_port, downstream_port}}, connector_type);
    }
  }
}

// ========== add_flow methods for subgraph support ==========

void Fragment::add_flow(const std::shared_ptr<Operator>& upstream_op,
                        const std::shared_ptr<Subgraph>& downstream_subgraph,
                        std::set<std::pair<std::string, std::string>> port_pairs) {
  // If port_pairs is empty, attempt auto-resolution
  if (port_pairs.empty()) {
    auto upstream_ports = get_operator_output_ports(upstream_op);
    auto downstream_ports = get_subgraph_input_ports(downstream_subgraph);

    // Try data flow first
    if (!downstream_ports.empty() && !upstream_ports.empty()) {
      try_auto_resolve_ports(upstream_ports,
                             downstream_ports,
                             fmt::format("Operator '{}'", upstream_op->name()),
                             fmt::format("Subgraph '{}'", downstream_subgraph->instance_name()),
                             port_pairs);
    } else {
      // Check if this could be a control flow connection
      const auto& exec_ports = downstream_subgraph->exec_interface_ports();
      std::vector<std::string> exec_input_ports;
      for (const auto& [port_name, port_info] : exec_ports) {
        if (port_info.is_input) {
          exec_input_ports.push_back(port_name);
        }
      }

      if (!exec_input_ports.empty()) {
        // This is a control flow connection
        if (upstream_op->operator_type() != Operator::OperatorType::kNative) {
          HOLOSCAN_LOG_ERROR(
              "Upstream operator '{}' must be a Native operator for control flow connections",
              upstream_op->name());
          return;
        }

        if (exec_input_ports.size() > 1) {
          auto err_msg = fmt::format(
              "Cannot auto-connect '{}' to '{}': subgraph has more than one "
              "execution input port ({}), so port mapping must be specified explicitly",
              upstream_op->name(),
              downstream_subgraph->instance_name(),
              fmt::join(exec_input_ports, ", "));
          HOLOSCAN_LOG_ERROR(err_msg);
          throw std::runtime_error(err_msg);
        }

        // Auto-connect to the single exec port
        port_pairs.emplace(Operator::kOutputExecPortName, exec_input_ports[0]);
      } else {
        auto err_msg = fmt::format(
            "Cannot auto-connect '{}' to '{}': no data or execution interface ports found",
            upstream_op->name(),
            downstream_subgraph->instance_name());
        HOLOSCAN_LOG_ERROR(err_msg);
        throw std::runtime_error(err_msg);
      }
    }
  }

  // Resolve subgraph interface ports and create flows
  resolve_and_create_op_to_subgraph_flows(
      upstream_op, downstream_subgraph, port_pairs, IOSpec::ConnectorType::kDefault);
}

void Fragment::add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                        const std::shared_ptr<Operator>& downstream_op,
                        std::set<std::pair<std::string, std::string>> port_pairs) {
  // If port_pairs is empty, attempt auto-resolution
  if (port_pairs.empty()) {
    auto upstream_ports = get_subgraph_output_ports(upstream_subgraph);
    auto downstream_ports = get_operator_input_ports(downstream_op);

    // Try data flow first
    if (!upstream_ports.empty() && !downstream_ports.empty()) {
      try_auto_resolve_ports(upstream_ports,
                             downstream_ports,
                             fmt::format("Subgraph '{}'", upstream_subgraph->instance_name()),
                             fmt::format("Operator '{}'", downstream_op->name()),
                             port_pairs);
    } else {
      // Check if this could be a control flow connection
      const auto& exec_ports = upstream_subgraph->exec_interface_ports();
      std::vector<std::string> exec_output_ports;
      for (const auto& [port_name, port_info] : exec_ports) {
        if (!port_info.is_input) {
          exec_output_ports.push_back(port_name);
        }
      }

      if (!exec_output_ports.empty() &&
          (downstream_ports.empty() || downstream_op->input_exec_spec())) {
        // This is a control flow connection
        if (downstream_op->operator_type() != Operator::OperatorType::kNative) {
          HOLOSCAN_LOG_ERROR(
              "Downstream operator '{}' must be a Native operator for control flow connections",
              downstream_op->name());
          return;
        }

        if (exec_output_ports.size() > 1) {
          auto err_msg = fmt::format(
              "Cannot auto-connect '{}' to '{}': subgraph has more than one "
              "execution output port ({}), so port mapping must be specified explicitly",
              upstream_subgraph->instance_name(),
              downstream_op->name(),
              fmt::join(exec_output_ports, ", "));
          HOLOSCAN_LOG_ERROR(err_msg);
          throw std::runtime_error(err_msg);
        }

        // Auto-connect to the single exec port
        port_pairs.emplace(exec_output_ports[0], Operator::kInputExecPortName);
      } else {
        HOLOSCAN_LOG_ERROR("Cannot auto-connect '{}' to '{}': no compatible interface ports found",
                           upstream_subgraph->instance_name(),
                           downstream_op->name());
        return;
      }
    }
  }

  // Resolve subgraph interface ports and create flows
  resolve_and_create_subgraph_to_op_flows(
      upstream_subgraph, downstream_op, port_pairs, IOSpec::ConnectorType::kDefault);
}

void Fragment::add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                        const std::shared_ptr<Subgraph>& downstream_subgraph,
                        std::set<std::pair<std::string, std::string>> port_pairs) {
  // If port_pairs is empty, attempt auto-resolution
  if (port_pairs.empty()) {
    auto upstream_ports = get_subgraph_output_ports(upstream_subgraph);
    auto downstream_ports = get_subgraph_input_ports(downstream_subgraph);

    // Try data flow first
    if (!upstream_ports.empty() && !downstream_ports.empty()) {
      try_auto_resolve_ports(upstream_ports,
                             downstream_ports,
                             fmt::format("Subgraph '{}'", upstream_subgraph->instance_name()),
                             fmt::format("Subgraph '{}'", downstream_subgraph->instance_name()),
                             port_pairs);
    } else {
      // Check if this could be a control flow connection
      const auto& upstream_exec_ports = upstream_subgraph->exec_interface_ports();
      const auto& downstream_exec_ports = downstream_subgraph->exec_interface_ports();

      std::vector<std::string> exec_output_ports;
      std::vector<std::string> exec_input_ports;

      for (const auto& [port_name, port_info] : upstream_exec_ports) {
        if (!port_info.is_input) {
          exec_output_ports.push_back(port_name);
        }
      }
      for (const auto& [port_name, port_info] : downstream_exec_ports) {
        if (port_info.is_input) {
          exec_input_ports.push_back(port_name);
        }
      }

      if (!exec_output_ports.empty() && !exec_input_ports.empty()) {
        // This is a control flow connection
        if (exec_output_ports.size() > 1 || exec_input_ports.size() > 1) {
          HOLOSCAN_LOG_ERROR(
              "Cannot auto-connect '{}' to '{}': multiple execution ports found, "
              "so port mapping must be specified explicitly",
              upstream_subgraph->instance_name(),
              downstream_subgraph->instance_name());
          return;
        }

        // Auto-connect to the single exec ports
        port_pairs.emplace(exec_output_ports[0], exec_input_ports[0]);
      } else {
        HOLOSCAN_LOG_ERROR("Cannot auto-connect '{}' to '{}': no compatible interface ports found",
                           upstream_subgraph->instance_name(),
                           downstream_subgraph->instance_name());
        return;
      }
    }
  }

  // Resolve both subgraph interface ports and create flows
  resolve_and_create_subgraph_to_subgraph_flows(
      upstream_subgraph, downstream_subgraph, port_pairs, IOSpec::ConnectorType::kDefault);
}

void Fragment::add_flow(const std::shared_ptr<Operator>& upstream_op,
                        const std::shared_ptr<Subgraph>& downstream_subgraph,
                        const IOSpec::ConnectorType connector_type) {
  // Attempt auto-connection when port_pairs is not provided
  std::set<std::pair<std::string, std::string>> port_pairs;
  auto upstream_ports = get_operator_output_ports(upstream_op);
  auto downstream_ports = get_subgraph_input_ports(downstream_subgraph);

  try_auto_resolve_ports(upstream_ports,
                         downstream_ports,
                         fmt::format("Operator '{}'", upstream_op->name()),
                         fmt::format("Subgraph '{}'", downstream_subgraph->instance_name()),
                         port_pairs);

  add_flow(upstream_op, downstream_subgraph, port_pairs, connector_type);
}

void Fragment::add_flow(const std::shared_ptr<Operator>& upstream_op,
                        const std::shared_ptr<Subgraph>& downstream_subgraph,
                        std::set<std::pair<std::string, std::string>> port_pairs,
                        const IOSpec::ConnectorType connector_type) {
  // If port_pairs is empty, attempt auto-resolution
  if (port_pairs.empty()) {
    auto upstream_ports = get_operator_output_ports(upstream_op);
    auto downstream_ports = get_subgraph_input_ports(downstream_subgraph);

    try_auto_resolve_ports(upstream_ports,
                           downstream_ports,
                           fmt::format("Operator '{}'", upstream_op->name()),
                           fmt::format("Subgraph '{}'", downstream_subgraph->instance_name()),
                           port_pairs);
  }

  resolve_and_create_op_to_subgraph_flows(
      upstream_op, downstream_subgraph, port_pairs, connector_type);
}

void Fragment::add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                        const std::shared_ptr<Operator>& downstream_op,
                        const IOSpec::ConnectorType connector_type) {
  // Attempt auto-connection when port_pairs is not provided
  std::set<std::pair<std::string, std::string>> port_pairs;
  auto upstream_ports = get_subgraph_output_ports(upstream_subgraph);
  auto downstream_ports = get_operator_input_ports(downstream_op);

  try_auto_resolve_ports(upstream_ports,
                         downstream_ports,
                         fmt::format("Subgraph '{}'", upstream_subgraph->instance_name()),
                         fmt::format("Operator '{}'", downstream_op->name()),
                         port_pairs);

  add_flow(upstream_subgraph, downstream_op, port_pairs, connector_type);
}

void Fragment::add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                        const std::shared_ptr<Operator>& downstream_op,
                        std::set<std::pair<std::string, std::string>> port_pairs,
                        const IOSpec::ConnectorType connector_type) {
  // If port_pairs is empty, attempt auto-resolution
  if (port_pairs.empty()) {
    auto upstream_ports = get_subgraph_output_ports(upstream_subgraph);
    auto downstream_ports = get_operator_input_ports(downstream_op);

    try_auto_resolve_ports(upstream_ports,
                           downstream_ports,
                           fmt::format("Subgraph '{}'", upstream_subgraph->instance_name()),
                           fmt::format("Operator '{}'", downstream_op->name()),
                           port_pairs);
  }

  resolve_and_create_subgraph_to_op_flows(
      upstream_subgraph, downstream_op, port_pairs, connector_type);
}

void Fragment::add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                        const std::shared_ptr<Subgraph>& downstream_subgraph,
                        const IOSpec::ConnectorType connector_type) {
  // Attempt auto-connection when port_pairs is not provided
  std::set<std::pair<std::string, std::string>> port_pairs;
  auto upstream_ports = get_subgraph_output_ports(upstream_subgraph);
  auto downstream_ports = get_subgraph_input_ports(downstream_subgraph);

  try_auto_resolve_ports(upstream_ports,
                         downstream_ports,
                         fmt::format("Subgraph '{}'", upstream_subgraph->instance_name()),
                         fmt::format("Subgraph '{}'", downstream_subgraph->instance_name()),
                         port_pairs);

  add_flow(upstream_subgraph, downstream_subgraph, port_pairs, connector_type);
}

void Fragment::add_flow(const std::shared_ptr<Subgraph>& upstream_subgraph,
                        const std::shared_ptr<Subgraph>& downstream_subgraph,
                        std::set<std::pair<std::string, std::string>> port_pairs,
                        const IOSpec::ConnectorType connector_type) {
  // If port_pairs is empty, attempt auto-resolution
  if (port_pairs.empty()) {
    auto upstream_ports = get_subgraph_output_ports(upstream_subgraph);
    auto downstream_ports = get_subgraph_input_ports(downstream_subgraph);

    try_auto_resolve_ports(upstream_ports,
                           downstream_ports,
                           fmt::format("Subgraph '{}'", upstream_subgraph->instance_name()),
                           fmt::format("Subgraph '{}'", downstream_subgraph->instance_name()),
                           port_pairs);
  }

  resolve_and_create_subgraph_to_subgraph_flows(
      upstream_subgraph, downstream_subgraph, port_pairs, connector_type);
}

std::pair<std::shared_ptr<Operator>, std::string> Fragment::resolve_subgraph_port(
    const std::shared_ptr<Subgraph>& subgraph, const std::string& interface_port) {
  return subgraph->get_interface_operator_port(interface_port);
}

}  // namespace holoscan
