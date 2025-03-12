/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
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

#include "holoscan/core/conditions/gxf/periodic.hpp"

#include <string>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"

template <>
struct YAML::convert<nvidia::gxf::PeriodicSchedulingPolicy> {
  static Node encode(const nvidia::gxf::PeriodicSchedulingPolicy& rhs) {
    YAML::Node node(YAML::NodeType::Scalar);
    switch (rhs) {
      case nvidia::gxf::PeriodicSchedulingPolicy::kCatchUpMissedTicks: {
        node = std::string("CatchUpMissedTicks");
        break;
      }
      case nvidia::gxf::PeriodicSchedulingPolicy::kMinTimeBetweenTicks: {
        node = std::string("MinTimeBetweenTicks");
        break;
      }
      case nvidia::gxf::PeriodicSchedulingPolicy::kNoCatchUpMissedTicks: {
        node = std::string("NoCatchUpMissedTicks");
        break;
      }
      default:
        throw std::runtime_error("Unexpected PeriodicConditionPolicy string");
    }
    return node;
  }

  static bool decode(const Node& node, nvidia::gxf::PeriodicSchedulingPolicy& rhs) {
    if (!node.IsScalar()) return false;

    const std::string value = node.as<std::string>();
    if (strcmp(value.c_str(), "CatchUpMissedTicks") == 0) {
      rhs = nvidia::gxf::PeriodicSchedulingPolicy::kCatchUpMissedTicks;
      return true;
    }
    if (strcmp(value.c_str(), "MinTimeBetweenTicks") == 0) {
      rhs = nvidia::gxf::PeriodicSchedulingPolicy::kMinTimeBetweenTicks;
      return true;
    }
    if (strcmp(value.c_str(), "NoCatchUpMissedTicks") == 0) {
      rhs = nvidia::gxf::PeriodicSchedulingPolicy::kNoCatchUpMissedTicks;
      return true;
    }
    return false;
  }
};

namespace holoscan {

nvidia::gxf::PeriodicSchedulingTerm* PeriodicCondition::get() const {
  return static_cast<nvidia::gxf::PeriodicSchedulingTerm*>(gxf_cptr_);
}

PeriodicCondition::PeriodicCondition(const std::string& name,
                                     nvidia::gxf::PeriodicSchedulingTerm* term)
    : GXFCondition(name, term) {
  if (term) {
    recess_period_ns_ = term->recess_period_ns();
    recess_period_ = std::to_string(recess_period_ns_);
  } else {
    HOLOSCAN_LOG_ERROR("PeriodicCondition: term is null");
  }
}

PeriodicCondition::PeriodicCondition(int64_t recess_period_ns, PeriodicConditionPolicy policy) {
  recess_period_ = std::to_string(recess_period_ns);
  recess_period_ns_ = recess_period_ns;
  switch (policy) {
    case PeriodicConditionPolicy::kCatchUpMissedTicks:
      policy_ = YAML::Node("CatchUpMissedTicks");
      break;
    case PeriodicConditionPolicy::kMinTimeBetweenTicks:
      policy_ = YAML::Node("MinTimeBetweenTicks");
      break;
    case PeriodicConditionPolicy::kNoCatchUpMissedTicks:
      policy_ = YAML::Node("NoCatchUpMissedTicks");
      break;
    default:
      HOLOSCAN_LOG_ERROR("Unrecognized policy enum value: {}", static_cast<int>(policy));
  }
}

void PeriodicCondition::initialize() {
  register_converter<nvidia::gxf::PeriodicSchedulingPolicy>();
  auto& current_args = args();

  auto find_it = std::find_if(current_args.begin(), current_args.end(), [](const auto& arg) {
    return (arg.name() == "policy" &&
            (arg.arg_type().element_type() == ArgElementType::kString ||
             arg.arg_type().element_type() == ArgElementType::kCustom) &&
            arg.arg_type().container_type() == ArgContainerType::kNative);
  });

  if (find_it != current_args.end()) {
    bool yaml_conversion_failed = false;
    YAML::Node policy_node;

    if (find_it->arg_type().element_type() == ArgElementType::kString) {
      auto policy_string = std::any_cast<std::string>(find_it->value());
      if (policy_string == "CatchUpMissedTicks" || policy_string == "MinTimeBetweenTicks" ||
          policy_string == "NoCatchUpMissedTicks") {
        policy_node = YAML::Node(policy_string);
      } else {
        HOLOSCAN_LOG_ERROR("Unrecognized policy string value: {}", policy_string);
        yaml_conversion_failed = true;
      }
    } else {
      try {
        auto policy_enum = std::any_cast<PeriodicConditionPolicy>(find_it->value());
        switch (policy_enum) {
          case PeriodicConditionPolicy::kCatchUpMissedTicks:
            policy_node = YAML::Node("CatchUpMissedTicks");
            break;
          case PeriodicConditionPolicy::kMinTimeBetweenTicks:
            policy_node = YAML::Node("MinTimeBetweenTicks");
            break;
          case PeriodicConditionPolicy::kNoCatchUpMissedTicks:
            policy_node = YAML::Node("NoCatchUpMissedTicks");
            break;
          default:
            HOLOSCAN_LOG_ERROR("Unrecognized policy enum value: {}", static_cast<int>(policy_enum));
            yaml_conversion_failed = true;
        }
      } catch (const std::bad_any_cast& e) {
        HOLOSCAN_LOG_ERROR("Unable to cast 'policy' argument to a PeriodicConditionPolicy enum: {}",
                           e.what());
        yaml_conversion_failed = true;
      }
    }

    if (!yaml_conversion_failed) {
      auto new_arg_end = std::remove_if(current_args.begin(),
                                        current_args.end(),
                                        [](const auto& arg) { return arg.name() == "policy"; });
      current_args.erase(new_arg_end, current_args.end());
      add_arg(Arg("policy", policy_node));
    }
  }

  GXFCondition::initialize();
}

void PeriodicCondition::setup(ComponentSpec& spec) {
  spec.param(
      recess_period_,
      "recess_period",
      "RecessPeriod",
      "The recess period indicates the minimum amount of time which has to pass before the "
      "operator is permitted to execute again. The period is specified as a string containing "
      "a number and an (optional) unit. If no unit is given the value is assumed to be "
      "in nanoseconds. Supported units are: Hz, s, ms. Example: 10ms, 10000000, 0.2s, 50Hz");
  spec.param(policy_,
             "policy",
             "Policy",
             "How the scheduler handles the recess period: CatchUpMissedTicks (default), "
             "MinTimeBetweenTicks",
             YAML::Node("CatchUpMissedTicks"));
}

void PeriodicCondition::recess_period(int64_t recess_period_ns) {
  std::string recess_period = std::to_string(recess_period_ns);
  auto periodic = get();
  if (periodic) { periodic->setParameter<std::string>("recess_period", recess_period); }
  recess_period_ = recess_period;
  recess_period_ns_ = recess_period_ns;
}

int64_t PeriodicCondition::recess_period_ns() {
  auto periodic = get();
  if (periodic) {
    auto recess_period_ns = periodic->recess_period_ns();
    if (recess_period_ns != recess_period_ns_) {
      recess_period_ns_ = recess_period_ns;
      recess_period_ = std::to_string(recess_period_ns_);
    }
  }
  return recess_period_ns_;
}

int64_t PeriodicCondition::last_run_timestamp() {
  int64_t last_run_timestamp = 0;
  auto periodic = get();
  if (periodic) {
    auto result = periodic->last_run_timestamp();
    if (result) {
      last_run_timestamp = result.value();
    } else {
      HOLOSCAN_LOG_ERROR("PeriodicCondition: Unable to get the result of 'last_run_timestamp()'");
    }
  } else {
    HOLOSCAN_LOG_ERROR("PeriodicCondition: GXF component pointer is null");
  }
  return last_run_timestamp;
}

}  // namespace holoscan
