/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/resources/gxf/cpu_thread.hpp"

#include <gxf/core/gxf.h>

#include <string>
#include <vector>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"

namespace holoscan {

void CPUThread::setup(ComponentSpec& spec) {
  spec.param(pin_operator_,
             "pin_entity",
             "Pin Operator ('pin_operator' can be used as an alias for 'pin_entity').",
             "Set the operator(s) to be pinned to a worker thread or not.",
             false);

  spec.param(pin_cores_,
             "pin_cores",
             "Pin Cores",
             "CPU core IDs to pin the worker thread to (empty means no core pinning).",
             ParameterFlag::kOptional);

  spec.param(sched_policy_,
             "sched_policy",
             "Scheduling Policy",
             "Real-time scheduling policy (SCHED_FIFO, SCHED_RR, SCHED_DEADLINE).",
             ParameterFlag::kOptional);

  spec.param(sched_priority_,
             "sched_priority",
             "Scheduling Priority",
             "Thread priority for FirstInFirstOut and RoundRobin policies.",
             ParameterFlag::kOptional);

  spec.param(sched_runtime_,
             "sched_runtime",
             "Scheduling Runtime",
             "Expected worst case execution time in nanoseconds for Deadline policy.",
             ParameterFlag::kOptional);

  spec.param(sched_deadline_,
             "sched_deadline",
             "Scheduling Deadline",
             "Relative deadline in nanoseconds for Deadline policy.",
             ParameterFlag::kOptional);

  spec.param(sched_period_,
             "sched_period",
             "Scheduling Period",
             "Period in nanoseconds for Deadline policy.",
             ParameterFlag::kOptional);
}

void CPUThread::initialize() {
  // Register converter for custom enum before calling parent class initialize
  register_converter<SchedulingPolicy>();
  auto& current_args = args();

  // Convert the SchedulingPolicy argument to a YAML node with a string representation
  // (SCHED_FIFO, SCHED_RR, or SCHED_DEADLINE) since Holoscan cannot directly set GXF parameters
  // with custom enum types.
  auto find_it = std::find_if(current_args.begin(), current_args.end(), [](const auto& arg) {
    return (arg.name() == "sched_policy" &&
            (arg.arg_type().element_type() == ArgElementType::kString ||
             arg.arg_type().element_type() == ArgElementType::kCustom ||
             arg.arg_type().element_type() == ArgElementType::kYAMLNode) &&
            arg.arg_type().container_type() == ArgContainerType::kNative);
  });

  if (find_it != current_args.end()) {
    bool yaml_conversion_failed = false;
    YAML::Node policy_node;

    if (find_it->arg_type().element_type() == ArgElementType::kString) {
      SchedulingPolicy policy = SchedulingPolicy::kFirstInFirstOut;
      auto policy_string = std::any_cast<std::string>(find_it->value());
      if (YAML::convert<nvidia::gxf::SchedulingPolicy>::decode(YAML::Node(policy_string), policy)) {
        policy_node = YAML::convert<nvidia::gxf::SchedulingPolicy>::encode(policy);
      } else {
        HOLOSCAN_LOG_ERROR(
            "Unable to decode 'sched_policy' argument to a SchedulingPolicy enum: {}",
            policy_string);
        yaml_conversion_failed = true;
      }
    } else if (find_it->arg_type().element_type() == ArgElementType::kCustom) {
      try {
        auto policy_enum = std::any_cast<SchedulingPolicy>(find_it->value());
        policy_node = YAML::convert<nvidia::gxf::SchedulingPolicy>::encode(policy_enum);
      } catch (const std::bad_any_cast& e) {
        HOLOSCAN_LOG_ERROR("Unable to cast 'sched_policy' argument to a SchedulingPolicy enum: {}",
                           e.what());
        yaml_conversion_failed = true;
      }
    } else if (find_it->arg_type().element_type() == ArgElementType::kYAMLNode) {
      // GXF's CPUThread resource requires YAML nodes to use string values (SCHED_FIFO, SCHED_RR,
      // or SCHED_DEADLINE) rather than integer strings, so we convert numeric values to their
      // corresponding string representations.
      SchedulingPolicy policy = SchedulingPolicy::kFirstInFirstOut;
      policy_node = std::any_cast<YAML::Node>(find_it->value());
      if (YAML::convert<nvidia::gxf::SchedulingPolicy>::decode(policy_node, policy)) {
        // Re-encode to convert numeric YAML values (e.g., "1", "2", "6") to string format.
        policy_node = YAML::convert<nvidia::gxf::SchedulingPolicy>::encode(policy);
      } else {
        HOLOSCAN_LOG_ERROR(
            "Unable to decode 'sched_policy' argument to a SchedulingPolicy enum: {}",
            policy_node.Scalar());
        yaml_conversion_failed = true;
      }
    }

    if (!yaml_conversion_failed) {
      auto new_arg_end =
          std::remove_if(current_args.begin(), current_args.end(), [](const auto& arg) {
            return arg.name() == "sched_policy";
          });
      current_args.erase(new_arg_end, current_args.end());
      add_arg(Arg("sched_policy", policy_node));
    }
  }

  // The underlying GXF nvidia::gxf::CPUThread component uses "pin_entity" as its parameter name,
  // but Holoscan's API exposes this as "pin_operator" in both C++ and Python interfaces.
  // If a "pin_operator" argument is provided, we must rename it to "pin_entity" before passing
  // it to the GXF component.
  for (auto& arg : current_args) {
    if (arg.name() == "pin_operator") {
      arg.name("pin_entity");
    }
  }

  // Call parent initialize
  gxf::GXFResource::initialize();
}

bool CPUThread::pinned() const {
  return pin_operator_;
}

std::vector<uint32_t> CPUThread::pin_cores() const {
  if (pin_cores_.has_value()) {
    return pin_cores_.get();
  }
  return {};
}

holoscan::expected<SchedulingPolicy, holoscan::RuntimeError> CPUThread::sched_policy() const {
  if (sched_policy_.has_value()) {
    SchedulingPolicy policy = SchedulingPolicy::kFirstInFirstOut;
    if (YAML::convert<nvidia::gxf::SchedulingPolicy>::decode(sched_policy_.get(), policy)) {
      return policy;
    } else {
      std::string err_msg =
          fmt::format("Unable to decode 'sched_policy' argument '{}' to a SchedulingPolicy enum",
                      sched_policy_.get().Scalar());
      HOLOSCAN_LOG_ERROR(err_msg);
      return make_unexpected(RuntimeError(ErrorCode::kFailure, err_msg));
    }
  }
  return make_unexpected(RuntimeError(ErrorCode::kFailure, "Scheduling policy not set"));
}

holoscan::expected<uint32_t, holoscan::RuntimeError> CPUThread::sched_priority() const {
  if (sched_priority_.has_value()) {
    return sched_priority_.get();
  }
  return make_unexpected(RuntimeError(ErrorCode::kFailure, "Scheduling priority not set"));
}

holoscan::expected<uint64_t, holoscan::RuntimeError> CPUThread::sched_runtime() const {
  if (sched_runtime_.has_value()) {
    return sched_runtime_.get();
  }
  return make_unexpected(RuntimeError(ErrorCode::kFailure, "Scheduling runtime not set"));
}

holoscan::expected<uint64_t, holoscan::RuntimeError> CPUThread::sched_deadline() const {
  if (sched_deadline_.has_value()) {
    return sched_deadline_.get();
  }
  return make_unexpected(RuntimeError(ErrorCode::kFailure, "Scheduling deadline not set"));
}

holoscan::expected<uint64_t, holoscan::RuntimeError> CPUThread::sched_period() const {
  if (sched_period_.has_value()) {
    return sched_period_.get();
  }
  return make_unexpected(RuntimeError(ErrorCode::kFailure, "Scheduling period not set"));
}

bool CPUThread::is_realtime() const {
  if (sched_policy_.has_value()) {
    SchedulingPolicy policy = SchedulingPolicy::kFirstInFirstOut;
    if (YAML::convert<nvidia::gxf::SchedulingPolicy>::decode(sched_policy_.get(), policy)) {
      return policy == SchedulingPolicy::kFirstInFirstOut ||
             policy == SchedulingPolicy::kRoundRobin || policy == SchedulingPolicy::kDeadline;
    } else {
      HOLOSCAN_LOG_ERROR("Unable to decode 'sched_policy_' value '{}' to a SchedulingPolicy enum",
                         sched_policy_.get().Scalar());
      return false;
    }
  }
  return false;
}

}  // namespace holoscan
