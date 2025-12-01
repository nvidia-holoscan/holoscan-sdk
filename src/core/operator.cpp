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

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gxf/std/clock.hpp"
#include "gxf/std/codelet.hpp"
#include "holoscan/core/conditions/gxf/asynchronous.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/executors/gxf/gxf_executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_execution_context.hpp"
#include "holoscan/core/gxf/gxf_operator.hpp"
#include "holoscan/core/gxf/gxf_scheduler.hpp"
#include "holoscan/core/gxf/gxf_scheduling_term_wrapper.hpp"
#include "holoscan/core/gxf/gxf_wrapper.hpp"
#include "holoscan/core/messagelabel.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/resources/gxf/condition_combiner.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"
#include "holoscan/logger/logger.hpp"
#include "holoscan/profiler/profiler.hpp"

namespace holoscan {

void Operator::initialize() {
  // Initialize the operator through the executor
  auto fragment_ptr = fragment();
  if (fragment_ptr) {
    auto& executor = fragment_ptr->executor();
    if (executor.initialize_operator(this)) {
      // Set the operator codelet (or other backend). It is utilized for Data Frame Flow Tracking
      // (DFFT)
      this->set_op_backend();
    }
    if (!is_metadata_enabled_.has_value()) {
      // Enable or disable metadata on a per-fragment basis if the user didn't explicitly specify
      // a setting for this operator.
      is_metadata_enabled_ = fragment_ptr->is_metadata_enabled();
    }

    // Set unique_id for all input and output specs
    if (spec_) {
      for (auto& [port_name, input_spec] : spec_->inputs()) {
        input_spec->set_unique_id(fmt::format("{}.{}", qualified_name(), port_name));
      }
      for (auto& [port_name, output_spec] : spec_->outputs()) {
        output_spec->set_unique_id(fmt::format("{}.{}", qualified_name(), port_name));
      }
    }

    // Now we can set the contexts for the operator
    ensure_contexts();
  } else {
    HOLOSCAN_LOG_WARN("Operator::initialize() - Fragment is not set");
  }
}

bool Operator::is_metadata_enabled() const {
  if (is_metadata_enabled_.has_value()) {
    return is_metadata_enabled_.value();
  } else {
    return fragment_->is_metadata_enabled();
  }
}

void Operator::add_cuda_stream_pool(int32_t dev_id, uint32_t stream_flags, int32_t stream_priority,
                                    uint32_t reserved_size, uint32_t max_size) {
  if (!is_gxf_compatible_operator_type()) {
    throw std::runtime_error(
        fmt::format("Operator '{}' is not a native, GXF, or virtual operator. Cannot add CUDA "
                    "stream pool.",
                    name_));
  }
  // If a "cuda_stream_pool" parameter exists, do nothing
  if (!spec_) {
    throw std::runtime_error(
        fmt::format("OperatorSpec has not been set for Operator '{}'.", name_));
  }
  auto params = spec()->params();
  auto param_iter = params.find("cuda_stream_pool");
  if (param_iter != params.end()) {
    HOLOSCAN_LOG_WARN(
        "add_cuda_stream_pool call has no effect on an operator with an explicit Parameter "
        "named 'cuda_stream_pool' defined");
    return;
  }

  // If a CudaStreamPool resource is already present, do nothing
  for (auto& resource : resources_) {
    // If the user already passed in a CudaStreamPool argument, do nothing
    auto stream_pool_resource = std::dynamic_pointer_cast<CudaStreamPool>(resource.second);
    if (stream_pool_resource) {
      return;
    }
  }

  // check if cuda_green_context is already in resources
  std::shared_ptr<CudaGreenContext> cuda_green_context_ptr = nullptr;
  if (resources_.find("cuda_green_context") != resources_.end()) {
    cuda_green_context_ptr =
        std::dynamic_pointer_cast<CudaGreenContext>(resources_["cuda_green_context"]);
    HOLOSCAN_LOG_DEBUG("CudaGreenContext resource found for Operator '{}'.", name_);
  }

  // If no CudaStreamPool was found, create a default one
  auto cuda_stream_pool =
      fragment()->make_resource<CudaStreamPool>(fmt::format("{}_stream_pool", name_),
                                                dev_id,
                                                stream_flags,
                                                stream_priority,
                                                reserved_size,
                                                max_size,
                                                cuda_green_context_ptr);

  // set it to belong to the operator's GXF entity
  if (graph_entity_) {
    cuda_stream_pool->gxf_eid(graph_entity_->eid());
  }

  // if the operator has already been initialized, initialize the stream pool
  if (is_initialized_) {
    cuda_stream_pool->initialize();
    // manually register in resources mapping
    resources_[cuda_stream_pool->name()] = cuda_stream_pool;
  } else {
    // add to arguments to be processed during operator initialization
    add_arg(cuda_stream_pool);
  }
}

bool Operator::is_root() {
  if (!fragment()) {
    throw std::runtime_error("Operator::is_root(): Fragment is not set");
  }
  return fragment()->graph().is_root(self_shared());
}

bool Operator::is_user_defined_root() {
  if (!fragment()) {
    throw std::runtime_error("Operator::is_user_defined_root(): Fragment is not set");
  }
  return fragment()->graph().is_user_defined_root(self_shared());
}

bool Operator::is_leaf() {
  if (!fragment()) {
    throw std::runtime_error("Operator::is_leaf(): Fragment is not set");
  }
  return fragment()->graph().is_leaf(self_shared());
}

bool Operator::is_all_operator_successor_virtual(OperatorNodeType op, OperatorGraph& graph) {
  auto next_nodes = graph.get_next_nodes(op);
  for (auto& next_node : next_nodes) {
    if (next_node->operator_type() != Operator::OperatorType::kVirtual) {
      return false;
    }
  }
  return true;
}

bool Operator::is_all_operator_predecessor_virtual(OperatorNodeType op, OperatorGraph& graph) {
  auto prev_nodes = graph.get_previous_nodes(op);
  for (auto& prev_node : prev_nodes) {
    if (prev_node->operator_type() != Operator::OperatorType::kVirtual) {
      return false;
    }
  }
  return true;
}

std::string Operator::qualified_name() const {
  if (fragment() && !this->fragment()->name().empty()) {
    return fmt::format("{}.{}", this->fragment()->name(), name());
  } else {
    return name();
  }
}

std::pair<std::string, std::string> Operator::parse_port_name(const std::string& op_port_name) {
  auto pos = op_port_name.find('.');
  if (pos == std::string::npos) {
    return std::make_pair(op_port_name, "");
  }

  auto op_name = op_port_name.substr(0, pos);
  auto port_name = op_port_name.substr(pos + 1);

  return std::make_pair(op_name, port_name);
}

std::pair<std::string, std::string> Operator::parse_operator_port_key(
    const std::string& operator_port_key) {
  auto dash_pos = operator_port_key.rfind('-');
  if (dash_pos == std::string::npos) {
    return std::make_pair(operator_port_key, "");
  }

  auto operator_name = operator_port_key.substr(0, dash_pos);
  auto port_name = operator_port_key.substr(dash_pos + 1);

  return std::make_pair(operator_name, port_name);
}

void Operator::update_published_messages(std::string output_name) {
  if (num_published_messages_map_.find(output_name) == num_published_messages_map_.end()) {
    num_published_messages_map_[output_name] = 0;
  }
  num_published_messages_map_[output_name] += 1;
}

holoscan::MessageLabel Operator::get_consolidated_input_label() {
  if (!is_gxf_compatible_operator_type()) {
    throw std::runtime_error("Operator backend is not GXF. Cannot get consolidated input label.");
  }
  MessageLabel m;

  if (this->input_message_labels.size()) {
    // Flatten the message_paths in input_message_labels into a single MessageLabel
    for (auto& it : this->input_message_labels) {
      MessageLabel everyinput = it.second;

      // Preserve frame numbers from each input message (only when profiling is enabled)
      // Frame numbers are now keyed by "operatorname-portname" so can be directly copied
      if (holoscan::profiler::trace_enabled()) {
        const auto& frame_numbers = everyinput.get_frame_numbers();
        for (const auto& [operator_port_key, frame_number] : frame_numbers) {
          // Extract operator name and port name from the key using helper function
          auto [operator_name, port_name] = parse_operator_port_key(operator_port_key);
          if (!operator_name.empty() && !port_name.empty()) {
            m.set_frame_number(operator_name, port_name, frame_number);
          }
        }
      }

      for (auto& p : everyinput.paths()) {
        m.add_new_path(p);
      }
    }
  } else {  // Root operator
    if (!this->is_root() && !this->is_user_defined_root()) {
      HOLOSCAN_LOG_DEBUG(
          "[get_consolidated_input_label] Not a root operator but still there is no message label "
          "stored in Op: {}",
          name());
    }
    // Just return the current operator timestamp label because
    // there is no input label
    if (op_backend_ptr) {
      auto scheduler = fragment()->scheduler();
      auto scheduler_clock = scheduler->clock();

      // Calculate the current execution according to the scheduler clock and
      // convert nanoseconds to microseconds as GXF scheduler uses nanoseconds
      // and DFFT uses microseconds
      // op_backend_ptr should be non-null
      if (!scheduler_clock) {
        throw std::runtime_error("scheduler_clock is null. Cannot calculate root execution time.");
      }
      int64_t cur_exec_time = (scheduler_clock->timestamp() -
                               ((nvidia::gxf::Codelet*)op_backend_ptr)->getExecutionTimestamp()) /
                              1000;

      // Set the receive timestamp for the root operator
      OperatorTimestampLabel new_op_label(
          this->qualified_name(), get_current_time_us() - cur_exec_time, -1);

      m.add_new_op_timestamp(new_op_label);
    } else {
      HOLOSCAN_LOG_WARN("Codelet pointer is not set. Data Flow Tracking will not work.");
    }
  }
  return m;
}

void Operator::set_op_backend() {
  if (!op_backend_ptr) {
    const char* codelet_typename = nullptr;
    if (operator_type_ == Operator::OperatorType::kNative) {
      codelet_typename = "holoscan::gxf::GXFWrapper";
    } else if (operator_type_ == Operator::OperatorType::kGXF ||
               operator_type_ == Operator::OperatorType::kVirtual) {
      ops::GXFOperator* gxf_op = static_cast<ops::GXFOperator*>(this);
      codelet_typename = gxf_op->gxf_typename();
    } else {
      HOLOSCAN_LOG_WARN("Unrecognized operator type: {}", static_cast<int>(operator_type_));
      return;
    }

    gxf_tid_t codelet_tid;
    auto fragment_ptr = fragment();
    if (fragment_ptr) {
      auto& executor = static_cast<holoscan::gxf::GXFExecutor&>(fragment_ptr->executor());
      if (executor.owns_context()) {
        HOLOSCAN_GXF_CALL(GxfComponentTypeId(executor.context(), codelet_typename, &codelet_tid));

        HOLOSCAN_GXF_CALL(GxfComponentPointer(
            executor.context(), id(), codelet_tid, reinterpret_cast<void**>(&op_backend_ptr)));
      } else {
        HOLOSCAN_LOG_DEBUG("GXF Context is not owned by the executor.");
      }
    } else {
      HOLOSCAN_LOG_WARN("Fragment is not set");
      return;
    }
  }
}

gxf_uid_t Operator::initialize_graph_entity(void* context, const std::string& entity_prefix) {
  const std::string op_entity_name = fmt::format("{}{}", entity_prefix, name_);
  HOLOSCAN_LOG_TRACE(
      "initialize_graph_entity called for Operator {}, entity_name: {}", name_, op_entity_name);
  graph_entity_ = std::make_shared<nvidia::gxf::GraphEntity>();
  auto maybe = graph_entity_->setup(context, op_entity_name.c_str());
  if (!maybe) {
    throw std::runtime_error(fmt::format("Failed to create operator entity: '{}'", op_entity_name));
  }
  return graph_entity_->eid();
}

void Operator::initialize_async_condition() {
  if (!internal_async_condition_) {
    internal_async_condition_ =
        fragment()->make_condition<holoscan::AsynchronousCondition>("_internal_async_condition");
    add_arg(internal_async_condition_);
  }
}

gxf_uid_t Operator::add_codelet_to_graph_entity() {
  HOLOSCAN_LOG_TRACE("calling graph_entity()->addCodelet for {}", name());
  if (!graph_entity_) {
    throw std::runtime_error(
        fmt::format("graph entity is not initialized for operator '{}'", name_));
  }
  auto codelet_handle = graph_entity_->addCodelet<holoscan::gxf::GXFWrapper>(name().c_str());
  if (!codelet_handle) {
    throw std::runtime_error(
        fmt::format("Failed to create GXFWrapper codelet corresponding to operator '{}'", name_));
  }
  codelet_handle->set_operator(this);
  HOLOSCAN_LOG_TRACE("\tadded codelet with cid {} to entity with eid {}",
                     codelet_handle->cid(),
                     graph_entity_->eid());
  return codelet_handle->cid();
}

YAML::Node Operator::to_yaml_node() const {
  std::unordered_map<OperatorType, std::string> operatortype_namemap{
      {OperatorType::kGXF, "kGXF"s},
      {OperatorType::kNative, "kNative"s},
      {OperatorType::kVirtual, "kVirtual"s},
  };

  YAML::Node node = ComponentBase::to_yaml_node();
  node["type"] = operatortype_namemap[operator_type_];
  node["conditions"] = YAML::Node(YAML::NodeType::Sequence);
  for (const auto& c : conditions_) {
    node["conditions"].push_back(c.second->to_yaml_node());
  }
  node["resources"] = YAML::Node(YAML::NodeType::Sequence);
  for (const auto& r : resources_) {
    node["resources"].push_back(r.second->to_yaml_node());
  }
  if (spec_) {
    node["spec"] = spec_->to_yaml_node();
  } else {
    node["spec"] = YAML::Null;
  }
  return node;
}

void Operator::initialize_conditions() {
  if (!is_gxf_compatible_operator_type()) {
    throw std::runtime_error("Operator backend is not GXF. Cannot initialize conditions.");
  }
  // Inspect resources for any ConditionCombiner and automatically add any conditions
  // associated with those so they don't also have to also be passed individually to
  // Fragment::make_operator when creating the operator.
  for (const auto& [name, resource] : resources_) {
    HOLOSCAN_LOG_TRACE("\top '{}': initializing resource: {}", name_, resource->name());
    auto condition_combiner = std::dynamic_pointer_cast<ConditionCombiner>(resource);
    if (condition_combiner) {
      HOLOSCAN_LOG_DEBUG(
          "Found ConditionCombiner resource '{}'. Adding the associated conditions to "
          "operator '{}'",
          name,
          this->name());

      // Extract the required "terms" argument to the combiner and add each as an argument to
      // the operator.
      auto terms_arg_it = std::find_if(resource->args().begin(),
                                       resource->args().end(),
                                       [](const auto& arg) { return (arg.name() == "terms"); });
      if (terms_arg_it == resource->args().end()) {
        HOLOSCAN_LOG_ERROR(
            "ConditionCombiner '{}' did not have a 'terms' argument. There are no conditions to "
            "add as arguments to operator '{}'.",
            name,
            this->name());
      } else {
        HOLOSCAN_LOG_DEBUG(
            "Found ConditionCombiner resource '{}' with terms argument (for operator '{}')",
            name,
            this->name());
        auto terms_arg = *terms_arg_it;
        auto terms = std::any_cast<std::vector<std::shared_ptr<Condition>>>(terms_arg.value());
        // In the context of OperatorSpec::or_combiner_port_names, these port conditions will have
        // already been initialized. It should not hurt to call add_arg again on them again. It
        // will just result in debug level messages about initialize having already been called
        // for each.
        for (const auto& term : terms) {
          add_arg(term);
        }
      }
    }
  }

  for (const auto& [name, condition] : conditions_) {
    // Set the operator for the condition (needed for Condition methods converting a port name to a
    // Receiver or Transmitter object)
    condition->set_operator(this);

    HOLOSCAN_LOG_TRACE("\top '{}': initializing condition: {}", name_, condition->name());
    auto gxf_condition = std::dynamic_pointer_cast<gxf::GXFCondition>(condition);
    if (gxf_condition) {
      // assign the condition to the same entity as the operator and initialize it
      gxf_condition->add_to_graph_entity(this);
    } else {
      // native Condition support via GXFSchedulingTermWrapper
      HOLOSCAN_LOG_TRACE("calling graph_entity()->addSchedulingTerm for native condition '{}'",
                         condition->name());
      if (!graph_entity_) {
        throw std::runtime_error(
            fmt::format("graph entity is not initialized for condition '{}'", condition->name()));
      }
      auto term_handle = graph_entity_->addSchedulingTerm<holoscan::gxf::GXFSchedulingTermWrapper>(
          condition->name().c_str());
      if (!term_handle) {
        throw std::runtime_error(fmt::format(
            "Failed to create GXFSchedulingTermWrapper term corresponding to condition '{}'",
            condition->name()));
      }
      term_handle->set_condition(condition);
      HOLOSCAN_LOG_TRACE("\tadded native condition with cid {} to entity with eid {}",
                         term_handle->cid(),
                         graph_entity_->eid());

      // need to store the GXF cid (of type gxf_uid_t) for potential use by GXFParameterAdapter
      // note: gxf_uid_t is an alias for int64_t so the static cast isn't strictly necessary
      condition->wrapper_cid(static_cast<int64_t>(term_handle->cid()));

      // initialize as a native (non-GXF) resource
      condition->initialize();
    }
  }
}

void Operator::initialize_resources() {
  if (!is_gxf_compatible_operator_type()) {
    throw std::runtime_error("Operator backend is not GXF. Cannot initialize resources.");
  }
  for (const auto& [name, resource] : resources_) {
    HOLOSCAN_LOG_TRACE("\top '{}': initializing resource: {}", name_, resource->name());
    auto gxf_resource = std::dynamic_pointer_cast<gxf::GXFResource>(resource);
    if (gxf_resource) {
      // assign the resource to the same entity as the operator and initialize it
      gxf_resource->add_to_graph_entity(this);
    } else {
      // initialize as a native (non-GXF) resource
      resource->initialize();
    }
  }
}

void Operator::find_ports_used_by_condition_args() {
  for (auto& condition : conditions_) {
    auto& cond_args = condition.second->args();

    // replace any string "receiver" with Receiver or "transmitter" with Transmitter object.
    // GXF Condition type wrappers use names "receiver" and "transmitter", but for native
    // Python operators we use "receiver_name" and "transmitter_name" instead to avoid
    // conflict with Python `Operator.receiver` and `Operator.transmitter` method names.
    const std::vector<std::string> connector_arg_names = {
        "receiver", "receiver_name", "transmitter", "transmitter_name"};
    for (const auto& arg_name : connector_arg_names) {
      auto connector_arg_iter =
          std::find_if(cond_args.begin(), cond_args.end(), [arg_name](const auto& arg) {
            return (arg.name() == arg_name &&
                    (arg.arg_type().element_type() == ArgElementType::kString ||
                     arg.arg_type().element_type() == ArgElementType::kYAMLNode) &&
                    arg.arg_type().container_type() == ArgContainerType::kNative);
          });
      if (connector_arg_iter != cond_args.end()) {
        std::string connector_name;
        if (connector_arg_iter->arg_type().element_type() == ArgElementType::kYAMLNode) {
          auto node = std::any_cast<YAML::Node>(connector_arg_iter->value());
          // skip if this was not a scalar node or does not contain a string
          if (!node.IsScalar()) {
            continue;
          }
          connector_name = node.as<std::string>();
          if (connector_name.empty()) {
            continue;
          }
        } else {
          connector_name = std::any_cast<std::string>(connector_arg_iter->value());
        }
        if (arg_name == "receiver" || arg_name == "receiver_name") {
          non_default_input_ports_.emplace_back(std::move(connector_name));
        } else {
          non_default_output_ports_.emplace_back(std::move(connector_name));
        }
      }
    }
  }
}

void Operator::update_connector_arguments() {
  if (!spec_) {
    throw std::runtime_error(fmt::format("No operator spec for Operator '{}'", name_));
  }
  for (auto& condition : conditions_) {
    auto& cond_args = condition.second->args();

    // replace any string "receiver" with Receiver or "transmitter" with Transmitter object.
    const std::vector<std::string> connector_arg_names = {"receiver", "transmitter"};
    for (const auto& arg_name : connector_arg_names) {
      auto connector_arg_iter =
          std::find_if(cond_args.begin(), cond_args.end(), [arg_name](const auto& arg) {
            return (arg.name() == arg_name &&
                    (arg.arg_type().element_type() == ArgElementType::kString ||
                     arg.arg_type().element_type() == ArgElementType::kYAMLNode) &&
                    arg.arg_type().container_type() == ArgContainerType::kNative);
          });
      if (connector_arg_iter != cond_args.end()) {
        std::string connector_name;
        if (connector_arg_iter->arg_type().element_type() == ArgElementType::kYAMLNode) {
          auto node = std::any_cast<YAML::Node>(connector_arg_iter->value());
          // skip if this was not a scalar node or does not contain a string
          if (!node.IsScalar()) {
            continue;
          }
          connector_name = node.as<std::string>();
          if (connector_name.empty()) {
            continue;
          }
        } else {
          connector_name = std::any_cast<std::string>(connector_arg_iter->value());
        }
        HOLOSCAN_LOG_DEBUG(fmt::format(
            "'{}' argument was specified via a string. Replacing that argument with one "
            "using the actual {} object with name '{}'.",
            arg_name,
            arg_name,
            connector_name));
        using IOSpecIterator =
            std::unordered_map<std::string, std::shared_ptr<holoscan::IOSpec>>::const_iterator;
        IOSpecIterator it;
        // find the corresponding input port
        if (arg_name == "receiver") {
          const auto& inputs = spec_->inputs();
          it = inputs.find(connector_name);
          if (it == inputs.end()) {
            throw std::runtime_error(fmt::format(
                "Operator '{}' does not have an input port '{}'", name_, connector_name));
          }
        } else if (arg_name == "transmitter") {
          const auto& outputs = spec_->outputs();
          it = outputs.find(connector_name);
          if (it == outputs.end()) {
            throw std::runtime_error(fmt::format(
                "Operator '{}' does not have an output port '{}'", name_, connector_name));
          }
        }
        HOLOSCAN_LOG_DEBUG(
            "Operator '{}': removing old arg '{}' and adding actual connector for port '{}'",
            name_,
            arg_name,
            connector_name);
        // remove the old argument and add the new one
        auto new_arg_end =
            std::remove_if(cond_args.begin(), cond_args.end(), [arg_name](const auto& arg) {
              return arg.name() == arg_name;
            });
        cond_args.erase(new_arg_end, cond_args.end());
        condition.second->add_arg(Arg(arg_name, it->second->connector()));
      }
    }
  }
}

void Operator::update_params_from_args() {
  if (!spec_) {
    throw std::runtime_error(fmt::format("No operator spec for Operator '{}'", name_));
  }
  update_params_from_args(spec_->params());
}

void Operator::set_parameters() {
  update_params_from_args();
  if (!spec_) {
    throw std::runtime_error(fmt::format("No operator spec for Operator '{}'", name_));
  }

  // Set only default parameter values
  std::vector<std::string> errors;
  for (auto& [key, param_wrap] : spec_->params()) {
    // If no value is specified, the default value will be used by setting an empty argument.
    Arg empty_arg("");
    try {
      ArgumentSetter::set_param(param_wrap, empty_arg);
    } catch (const std::exception& e) {
      std::string error_msg = fmt::format("Parameter '{}': {}", key, e.what());
      HOLOSCAN_LOG_ERROR("Operator '{}': failed to set default parameter - {}", name_, error_msg);
      errors.push_back(error_msg);
    }
  }

  if (!errors.empty()) {
    throw std::runtime_error(
        fmt::format("Operator '{}': failed to set {} default parameter(s):\n  - {}",
                    name_,
                    errors.size(),
                    fmt::join(errors, "\n  - ")));
  }
}

bool Operator::has_ucx_connector() {
  if (!spec_) {
    throw std::runtime_error(fmt::format("No operator spec for Operator '{}'", name_));
  }
  for (const auto& [_, io_spec] : spec_->inputs()) {
    if (io_spec->connector_type() == IOSpec::ConnectorType::kUCX) {
      return true;
    }
  }
  for (const auto& [_, io_spec] : spec_->outputs()) {
    if (io_spec->connector_type() == IOSpec::ConnectorType::kUCX) {
      return true;
    }
  }
  return false;
}

void Operator::reset_backend_objects() {
  if (!spec_) {
    throw std::runtime_error(fmt::format("No operator spec for Operator '{}'", name_));
  }

  HOLOSCAN_LOG_TRACE("Operator '{}'::reset_backend_objects", name_);
  auto reset_resource = [](std::shared_ptr<holoscan::Resource> resource) {
    if (resource) {
      resource->reset_backend_objects();
    }
  };
  auto reset_condition = [](std::shared_ptr<holoscan::Condition> condition) {
    if (condition) {
      condition->reset_backend_objects();
    }
  };
  auto reset_iospec =
      [reset_resource,
       reset_condition](const std::unordered_map<std::string, std::shared_ptr<IOSpec>>& io_specs) {
        for (auto& [_, io_spec] : io_specs) {
          if (io_spec) {
            reset_resource(io_spec->connector());
            for (auto& [_, condition] : io_spec->conditions()) {
              reset_condition(condition);
            }
          }
        }
      };
  for (auto& [_, resource] : resources_) {
    reset_resource(resource);
  }
  for (auto& [_, condition] : conditions_) {
    reset_condition(condition);
  }
  resources_.clear();
  conditions_.clear();
  reset_iospec(spec_->inputs());
  reset_iospec(spec_->outputs());
  ComponentBase::reset_backend_objects();
  graph_entity_.reset();
}

std::optional<std::shared_ptr<Receiver>> Operator::receiver(const std::string& port_name) {
  if (!is_gxf_compatible_operator_type()) {
    throw std::runtime_error("Operator backend is not GXF. Cannot get Receiver.");
  }
  if (!spec_) {
    throw std::runtime_error(fmt::format("No operator spec for Operator '{}'", name_));
  }
  auto inputs = spec_->inputs();
  auto input_iter = inputs.find(port_name);
  if (input_iter == inputs.end()) {
    return std::nullopt;
  }
  auto connector = input_iter->second->connector();
  auto receiver = std::dynamic_pointer_cast<Receiver>(connector);
  if (receiver == nullptr) {
    return std::nullopt;
  }
  return receiver;
}

void Operator::queue_policy(const std::string& port_name, IOSpec::IOType port_type,
                            IOSpec::QueuePolicy policy) {
  if (!spec_) {
    throw std::runtime_error(fmt::format("No operator spec for Operator '{}'", name_));
  }
  const auto& iospecs = port_type == IOSpec::IOType::kInput ? spec_->inputs() : spec_->outputs();
  auto iospec_iter = iospecs.find(port_name);
  if (iospec_iter == iospecs.end()) {
    throw std::runtime_error(fmt::format(
        "No {} port with name '{}' found",
        port_type == IOSpec::IOType::kInput ? std::string("input") : std::string("output"),
        port_name));
  }
  auto iospec = iospec_iter->second;
  iospec->queue_policy(policy);
  return;
}

std::optional<std::shared_ptr<Transmitter>> Operator::transmitter(const std::string& port_name) {
  if (!is_gxf_compatible_operator_type()) {
    throw std::runtime_error("Operator backend is not GXF. Cannot get Transmitter.");
  }
  if (!spec_) {
    throw std::runtime_error(fmt::format("No operator spec for Operator '{}'", name_));
  }
  auto outputs = spec_->outputs();
  auto output_iter = outputs.find(port_name);
  if (output_iter == outputs.end()) {
    return std::nullopt;
  }
  auto connector = output_iter->second->connector();
  auto transmitter = std::dynamic_pointer_cast<Transmitter>(connector);
  if (transmitter == nullptr) {
    return std::nullopt;
  }
  return transmitter;
}

const std::shared_ptr<IOSpec>& Operator::input_exec_spec() {
  return input_exec_spec_;
}

const std::shared_ptr<IOSpec>& Operator::output_exec_spec() {
  return output_exec_spec_;
}

const std::function<void(const std::shared_ptr<Operator>&)>& Operator::dynamic_flow_func() {
  return dynamic_flow_func_;
}

std::shared_ptr<Operator> Operator::self_shared() {
  if (self_shared_.expired()) {
    throw std::runtime_error(fmt::format("shared_ptr of operator ({}) is not available", name_));
  }
  return self_shared_.lock();
}

void Operator::set_input_exec_spec(const std::shared_ptr<IOSpec>& input_exec_spec) {
  input_exec_spec_ = input_exec_spec;
}

void Operator::set_output_exec_spec(const std::shared_ptr<IOSpec>& output_exec_spec) {
  output_exec_spec_ = output_exec_spec;
}

void Operator::set_self_shared(const std::shared_ptr<Operator>& this_op) {
  self_shared_ = this_op;
}

void Operator::initialize_next_flows() {
  if (!next_flows_) {
    next_flows_ = std::make_shared<std::vector<std::shared_ptr<FlowInfo>>>();

    auto& graph = fragment()->graph();
    auto curr_op = graph.find_node(name());
    if (!curr_op) {
      HOLOSCAN_LOG_ERROR("find_node for operator '{}' was nullptr, cannot get next nodes.", name());
      return;
    }
    auto next_ops = graph.get_next_nodes(curr_op);
    if (curr_op->spec() != nullptr) {
      // Reserve enough space for the next flows (assuming each output port has a flow to each next
      // operator)
      next_flows_->reserve(curr_op->spec()->outputs().size() * next_ops.size());
    }
    for (auto& next_op : next_ops) {
      auto port_map = graph.get_port_map(curr_op, next_op).value_or(nullptr);
      if (port_map) {
        for (const auto& [out_port_name, in_port_names] : *port_map) {
          for (const auto& in_port_name : in_port_names) {
            next_flows_->emplace_back(
                std::make_shared<FlowInfo>(curr_op, out_port_name, next_op, in_port_name));
          }
        }
      }
    }
  }
}

const std::vector<std::shared_ptr<Operator::FlowInfo>>& Operator::next_flows() {
  if (!next_flows_) {
    initialize_next_flows();
  }
  return *next_flows_;
}

void Operator::add_dynamic_flow(const std::shared_ptr<FlowInfo>& flow) {
  if (!is_gxf_compatible_operator_type()) {
    throw std::runtime_error("Operator backend is not GXF. Cannot add dynamic flow.");
  }
  if (!dynamic_flows_) {
    dynamic_flows_ = std::make_shared<std::vector<std::shared_ptr<FlowInfo>>>();
    dynamic_flows_->reserve(next_flows().size());
  }
  dynamic_flows_->push_back(flow);
}

void Operator::add_dynamic_flow(const std::vector<std::shared_ptr<FlowInfo>>& flows) {
  if (!is_gxf_compatible_operator_type()) {
    throw std::runtime_error("Operator backend is not GXF. Cannot add dynamic flow.");
  }
  if (!dynamic_flows_) {
    dynamic_flows_ = std::make_shared<std::vector<std::shared_ptr<FlowInfo>>>();
    dynamic_flows_->reserve(next_flows().size());
  }
  dynamic_flows_->insert(dynamic_flows_->end(), flows.begin(), flows.end());
}

void Operator::add_dynamic_flow(const std::string& curr_output_port_name,
                                const std::shared_ptr<Operator>& next_op,
                                const std::string& next_input_port_name) {
  if (!is_gxf_compatible_operator_type()) {
    throw std::runtime_error("Operator backend is not GXF. Cannot add dynamic flow.");
  }
  if (!spec_) {
    throw std::runtime_error(
        fmt::format("OperatorSpec has not been set for Operator '{}'.", name_));
  }
  if (!dynamic_flows_) {
    dynamic_flows_ = std::make_shared<std::vector<std::shared_ptr<FlowInfo>>>();
    dynamic_flows_->reserve(next_flows().size());
  }

  std::string output_port_name = curr_output_port_name;

  if (curr_output_port_name.empty() && spec()->outputs().size() == 1) {
    output_port_name = spec()->outputs().begin()->first;
  }

  std::string input_port_name = next_input_port_name;

  if (next_input_port_name.empty() && next_op->spec()->inputs().size() == 1) {
    input_port_name = next_op->spec()->inputs().begin()->first;
  }

  if (spec()->outputs().find(output_port_name) == spec()->outputs().end()) {
    throw std::runtime_error(
        fmt::format("The upstream operator({}) does not have an output port with label '{}'",
                    name(),
                    output_port_name));
  }

  if (next_op->spec()->inputs().find(input_port_name) == next_op->spec()->inputs().end()) {
    throw std::runtime_error(
        fmt::format("The downstream operator({}) does not have an input port with label '{}'",
                    next_op->name(),
                    input_port_name));
  }

  auto flow = std::make_shared<FlowInfo>(self_shared(), output_port_name, next_op, input_port_name);
  dynamic_flows_->push_back(flow);
}

void Operator::add_dynamic_flow(const std::shared_ptr<Operator>& next_op,
                                const std::string& next_input_port_name) {
  add_dynamic_flow("", next_op, next_input_port_name);
}

const std::shared_ptr<std::vector<std::shared_ptr<Operator::FlowInfo>>>& Operator::dynamic_flows() {
  return dynamic_flows_;
}

const std::shared_ptr<Operator::FlowInfo>& Operator::find_flow_info(
    const std::function<bool(const std::shared_ptr<Operator::FlowInfo>&)>& predicate) {
  const auto& flows = next_flows();
  auto it = std::find_if(flows.begin(), flows.end(), predicate);
  if (it == flows.end()) {
    return kEmptyFlowInfo;
  }
  return *it;
}

std::vector<std::shared_ptr<Operator::FlowInfo>> Operator::find_all_flow_info(
    const std::function<bool(const std::shared_ptr<Operator::FlowInfo>&)>& predicate) {
  std::vector<std::shared_ptr<FlowInfo>> result;
  if (!next_flows().empty()) {
    std::copy_if(next_flows().begin(), next_flows().end(), std::back_inserter(result), predicate);
  }
  return result;
}

void Operator::set_dynamic_flows(
    const std::function<void(const std::shared_ptr<Operator>&)>& dynamic_flow_func) {
  if (is_gxf_compatible_operator_type()) {
    dynamic_flow_func_ = dynamic_flow_func;
  } else {
    throw std::runtime_error("Operator backend is not GXF. Cannot set dynamic flows.");
  }
}

std::shared_ptr<holoscan::AsynchronousCondition> Operator::async_condition() {
  return internal_async_condition_;
}

void Operator::stop_execution() {
  if (internal_async_condition_) {
    internal_async_condition_->event_state(holoscan::AsynchronousEventState::EVENT_NEVER);
  }
}

std::shared_ptr<holoscan::ExecutionContext> Operator::execution_context() const {
  return execution_context_;
}

void Operator::ensure_contexts() {
  if (!is_initialized_) {
    auto message =
        fmt::format("Operator::ensure_contexts(): Operator '{}' is not initialized yet", name());
    HOLOSCAN_LOG_ERROR(message);
    throw std::runtime_error(message);
  }

  if (!execution_context_) {
    execution_context_ = initialize_execution_context();
  }
}

std::shared_ptr<ExecutionContext> Operator::initialize_execution_context() {
  // Initialize a GXF execution context by default
  auto gxf_context =
      std::make_shared<gxf::GXFExecutionContext>(fragment()->executor().context(), this);
  auto gxf_exec_context = static_cast<gxf::GXFExecutionContext*>(gxf_context.get());
  auto input_context = static_cast<gxf::GXFInputContext*>(gxf_exec_context->input().get());
  auto output_context = static_cast<gxf::GXFOutputContext*>(gxf_exec_context->output().get());

  gxf_exec_context->init_cuda_object_handler(this);
  HOLOSCAN_LOG_TRACE(
      "Operator::ensure_contexts(): gxf_exec_context->cuda_object_handler() for op '{}' is "
      "{}null",
      name(),
      gxf_exec_context->cuda_object_handler() == nullptr ? "" : "not ");
  input_context->cuda_object_handler(gxf_exec_context->cuda_object_handler());
  output_context->cuda_object_handler(gxf_exec_context->cuda_object_handler());
  return gxf_context;
}

bool Operator::is_gxf_compatible_operator_type() const {
  return operator_type_ == Operator::OperatorType::kNative ||
         operator_type_ == Operator::OperatorType::kGXF ||
         operator_type_ == Operator::OperatorType::kVirtual;
}

std::shared_ptr<Executor> Operator::executor() {
  if (!fragment()) {
    throw std::runtime_error("Operator::executor(): Fragment is not set");
  }
  return fragment()->executor_shared();
}

void Operator::release_internal_resources() {
  // Note that the similar logic is implemented in the Python operator with the GIL guard
  internal_async_condition_.reset();
  dynamic_metadata_.reset();
  input_exec_spec_.reset();
  output_exec_spec_.reset();
  dynamic_flow_func_ = nullptr;
  next_flows_.reset();
  dynamic_flows_.reset();
}

}  // namespace holoscan
