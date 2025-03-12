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
#include "holoscan/core/executor.hpp"
#include "holoscan/core/executors/gxf/gxf_executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_operator.hpp"
#include "holoscan/core/gxf/gxf_scheduler.hpp"
#include "holoscan/core/gxf/gxf_scheduling_term_wrapper.hpp"
#include "holoscan/core/gxf/gxf_wrapper.hpp"
#include "holoscan/core/messagelabel.hpp"
#include "holoscan/core/operator.hpp"

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
  // If a "cuda_stream_pool" parameter exists, do nothing
  auto params = spec()->params();
  auto param_iter = params.find("cuda_stream_pool");
  if (param_iter != params.end()) { return; }

  // If a CudaStreamPool resource is already present, do nothing
  for (auto& resource : resources_) {
    // If the user already passed in a CudaStreamPool argument, do nothing
    auto stream_pool_resource = std::dynamic_pointer_cast<CudaStreamPool>(resource.second);
    if (stream_pool_resource) { return; }
  }

  // If no CudaStreamPool was found, create a default one
  auto cuda_stream_pool = std::make_shared<CudaStreamPool>(
      dev_id, stream_flags, stream_priority, reserved_size, max_size);
  cuda_stream_pool->name("default_cuda_stream_pool");

  // set it to belong to the operator's GXF entity
  if (graph_entity_) { cuda_stream_pool->gxf_eid(graph_entity_->eid()); }
  add_arg(cuda_stream_pool);
}

bool Operator::is_root() {
  std::shared_ptr<holoscan::Operator> op_shared_ptr(this, [](Operator*) {});

  return fragment()->graph().is_root(op_shared_ptr);
}

bool Operator::is_user_defined_root() {
  std::shared_ptr<holoscan::Operator> op_shared_ptr(this, [](Operator*) {});

  return fragment()->graph().is_user_defined_root(op_shared_ptr);
}

bool Operator::is_leaf() {
  std::shared_ptr<holoscan::Operator> op_shared_ptr(this, [](Operator*) {});

  return fragment()->graph().is_leaf(op_shared_ptr);
}

bool Operator::is_all_operator_successor_virtual(OperatorNodeType op, OperatorGraph& graph) {
  auto next_nodes = graph.get_next_nodes(op);
  for (auto& next_node : next_nodes) {
    if (next_node->operator_type() != Operator::OperatorType::kVirtual) { return false; }
  }
  return true;
}

bool Operator::is_all_operator_predecessor_virtual(OperatorNodeType op, OperatorGraph& graph) {
  auto prev_nodes = graph.get_previous_nodes(op);
  for (auto& prev_node : prev_nodes) {
    if (prev_node->operator_type() != Operator::OperatorType::kVirtual) { return false; }
  }
  return true;
}

std::string Operator::qualified_name() {
  if (!this->fragment()->name().empty()) {
    return fmt::format("{}.{}", this->fragment()->name(), name());
  } else {
    return name();
  }
}

std::pair<std::string, std::string> Operator::parse_port_name(const std::string& op_port_name) {
  auto pos = op_port_name.find('.');
  if (pos == std::string::npos) { return std::make_pair(op_port_name, ""); }

  auto op_name = op_port_name.substr(0, pos);
  auto port_name = op_port_name.substr(pos + 1);

  return std::make_pair(op_name, port_name);
}

void Operator::update_published_messages(std::string output_name) {
  if (num_published_messages_map_.find(output_name) == num_published_messages_map_.end()) {
    num_published_messages_map_[output_name] = 0;
  }
  num_published_messages_map_[output_name] += 1;
}

holoscan::MessageLabel Operator::get_consolidated_input_label() {
  MessageLabel m;

  if (this->input_message_labels.size()) {
    // Flatten the message_paths in input_message_labels into a single MessageLabel
    for (auto& it : this->input_message_labels) {
      MessageLabel everyinput = it.second;
      for (auto& p : everyinput.paths()) { m.add_new_path(p); }
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
      auto scheduler = std::dynamic_pointer_cast<gxf::GXFScheduler>(fragment()->scheduler());
      nvidia::gxf::Clock* scheduler_clock = scheduler->gxf_clock();

      // Calculate the current execution according to the scheduler clock and
      // convert nanoseconds to microseconds as GXF scheduler uses nanoseconds
      // and DFFT uses microseconds
      if (!op_backend_ptr) {
        throw std::runtime_error("op_backend_ptr is null. Cannot calculate root execution time.");
      } else if (!scheduler_clock) {
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
    } else {
      ops::GXFOperator* gxf_op = static_cast<ops::GXFOperator*>(this);
      codelet_typename = gxf_op->gxf_typename();
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
  for (const auto& c : conditions_) { node["conditions"].push_back(c.second->to_yaml_node()); }
  node["resources"] = YAML::Node(YAML::NodeType::Sequence);
  for (const auto& r : resources_) { node["resources"].push_back(r.second->to_yaml_node()); }
  if (spec_) {
    node["spec"] = spec_->to_yaml_node();
  } else {
    node["spec"] = YAML::Null;
  }
  return node;
}

void Operator::initialize_conditions() {
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
      // initialize as a native (non-GXF) resource
      condition->initialize();
    }
  }
}

void Operator::initialize_resources() {
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
          if (!node.IsScalar()) { continue; }
          connector_name = node.as<std::string>();
          if (connector_name.empty()) { continue; }
        } else {
          connector_name = std::any_cast<std::string>(connector_arg_iter->value());
        }
        if (arg_name == "receiver") {
          non_default_input_ports_.emplace_back(std::move(connector_name));
        } else {
          non_default_output_ports_.emplace_back(std::move(connector_name));
        }
      }
    }
  }
}

void Operator::update_connector_arguments() {
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
          if (!node.IsScalar()) { continue; }
          connector_name = node.as<std::string>();
          if (connector_name.empty()) { continue; }
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
  update_params_from_args(spec_->params());
}

void Operator::set_parameters() {
  update_params_from_args();

  // Set only default parameter values
  for (auto& [key, param_wrap] : spec_->params()) {
    // If no value is specified, the default value will be used by setting an empty argument.
    Arg empty_arg("");
    ArgumentSetter::set_param(param_wrap, empty_arg);
  }
}

bool Operator::has_ucx_connector() {
  for (const auto& [_, io_spec] : spec_->inputs()) {
    if (io_spec->connector_type() == IOSpec::ConnectorType::kUCX) { return true; }
  }
  for (const auto& [_, io_spec] : spec_->outputs()) {
    if (io_spec->connector_type() == IOSpec::ConnectorType::kUCX) { return true; }
  }
  return false;
}

void Operator::reset_graph_entities() {
  HOLOSCAN_LOG_TRACE("Operator '{}'::reset_graph_entities", name_);
  auto reset_resource = [](std::shared_ptr<holoscan::Resource> resource) {
    if (resource) {
      auto gxf_resource = std::dynamic_pointer_cast<holoscan::gxf::GXFResource>(resource);
      if (gxf_resource) { gxf_resource->reset_gxf_graph_entity(); }
      resource->reset_graph_entities();
    }
  };
  auto reset_condition = [](std::shared_ptr<holoscan::Condition> condition) {
    if (condition) {
      auto gxf_condition = std::dynamic_pointer_cast<holoscan::gxf::GXFCondition>(condition);
      if (gxf_condition) { gxf_condition->reset_gxf_graph_entity(); }
      condition->reset_graph_entities();
    }
  };
  auto reset_iospec =
      [reset_resource,
       reset_condition](const std::unordered_map<std::string, std::shared_ptr<IOSpec>>& io_specs) {
        for (auto& [_, io_spec] : io_specs) {
          reset_resource(io_spec->connector());
          for (auto& [_, condition] : io_spec->conditions()) { reset_condition(condition); }
        }
      };
  for (auto& [_, resource] : resources_) { reset_resource(resource); }
  for (auto& [_, condition] : conditions_) { reset_condition(condition); }
  reset_iospec(spec_->inputs());
  reset_iospec(spec_->outputs());
  ComponentBase::reset_graph_entities();
  graph_entity_.reset();
}

std::optional<std::shared_ptr<Receiver>> Operator::receiver(const std::string& port_name) {
  auto inputs = spec_->inputs();
  auto input_iter = inputs.find(port_name);
  if (input_iter == inputs.end()) { return std::nullopt; }
  auto connector = input_iter->second->connector();
  auto receiver = std::dynamic_pointer_cast<Receiver>(connector);
  if (receiver == nullptr) { return std::nullopt; }
  return receiver;
}

void Operator::queue_policy(const std::string& port_name, IOSpec::IOType port_type,
                            IOSpec::QueuePolicy policy) {
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
  auto outputs = spec_->outputs();
  auto output_iter = outputs.find(port_name);
  if (output_iter == outputs.end()) { return std::nullopt; }
  auto connector = output_iter->second->connector();
  auto transmitter = std::dynamic_pointer_cast<Transmitter>(connector);
  if (transmitter == nullptr) { return std::nullopt; }
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
    auto next_ops = graph.get_next_nodes(curr_op);
    // Reserve enough space for the next flows (assuming each output port has a flow to each next
    // operator)
    next_flows_->reserve(curr_op->spec()->outputs().size() * next_ops.size());
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
  if (!next_flows_) { initialize_next_flows(); }
  return *next_flows_;
}

void Operator::add_dynamic_flow(const std::shared_ptr<FlowInfo>& flow) {
  if (!dynamic_flows_) {
    dynamic_flows_ = std::make_shared<std::vector<std::shared_ptr<FlowInfo>>>();
    dynamic_flows_->reserve(next_flows().size());
  }
  dynamic_flows_->push_back(flow);
}

void Operator::add_dynamic_flow(const std::vector<std::shared_ptr<FlowInfo>>& flows) {
  if (!dynamic_flows_) {
    dynamic_flows_ = std::make_shared<std::vector<std::shared_ptr<FlowInfo>>>();
    dynamic_flows_->reserve(next_flows().size());
  }
  dynamic_flows_->insert(dynamic_flows_->end(), flows.begin(), flows.end());
}

void Operator::add_dynamic_flow(const std::string& curr_output_port_name,
                                const std::shared_ptr<Operator>& next_op,
                                const std::string& next_input_port_name) {
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
  if (it == flows.end()) { return kEmptyFlowInfo; }
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
  dynamic_flow_func_ = dynamic_flow_func;
}

}  // namespace holoscan
