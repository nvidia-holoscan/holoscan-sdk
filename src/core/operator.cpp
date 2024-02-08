/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "gxf/std/clock.hpp"
#include "gxf/std/codelet.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/executors/gxf/gxf_executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_operator.hpp"
#include "holoscan/core/gxf/gxf_scheduler.hpp"
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
  } else {
    HOLOSCAN_LOG_WARN("Operator::initialize() - Fragment is not set");
  }
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
    for (auto it : this->input_message_labels) {
      MessageLabel everyinput = it.second;
      for (auto p : everyinput.paths()) { m.add_new_path(p); }
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
      OperatorTimestampLabel new_op_label(this, get_current_time_us() - cur_exec_time, -1);

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
      if (executor.own_gxf_context()) {
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

YAML::Node Operator::to_yaml_node() const {
  std::unordered_map<OperatorType, std::string> operatortype_namemap{
      {OperatorType::kGXF, "kGXF"s},
      {OperatorType::kNative, "kNative"s},
      {OperatorType::kVirtual, "kVirtual"s},
  };

  YAML::Node node = Component::to_yaml_node();
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

}  // namespace holoscan
