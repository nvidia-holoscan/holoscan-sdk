/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "holoscan/core/conditions/gxf/multi_message_available_timeout.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gxf/std/scheduling_terms.hpp>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/resource.hpp"

namespace holoscan {

void MultiMessageAvailableTimeoutCondition::setup(ComponentSpec& spec) {
  spec.param(receivers_,
             "receivers",
             "Receivers",
             "The scheduling term permits execution if the given channels have at least a given "
             "number of messages available. Note that this parameter is not intended to be passed "
             "via an argument to `Fragment::make_condition`. Instead, from the Operator::setup "
             "method, the receiver port names specified in the `port_names` argument to "
             "`OperatorSpec::multi_port_condition` will be used. Holoscan will take "
             "care of setting this parameter with the actual receiver objects created during "
             "initialization of the application.");
  spec.param(execution_frequency_,
             "execution_frequency",
             "Execution frequency of the entity",
             "The 'execution frequency' indicates the amount of time after which the entity will "
             "be allowed to execute again, even if the specified number of messages have not yet "
             "been received. The period is specified as a string containing  of a number and an "
             "(optional) unit. If no unit is given the value is assumed to be in nanoseconds. "
             "Supported units are: Hz, s, ms. Example: 10ms, 10000000, 0.2s, 50Hz");
  spec.param(sampling_mode_,
             "sampling_mode",
             "Sampling Mode",
             "The sampling method to use when checking for messages in receiver queues. "
             "Options: YAML::Node(\"SumOfAll\"), YAML::Node(\"PerReceiver\")",
             YAML::Node("SumOfAll"));
  spec.param(min_sizes_,
             "min_sizes",
             "Minimum message counts",
             "The scheduling term permits execution if all given receivers have at least the "
             "given number of messages available in this list. This option is only intended for "
             "use with `sampling_mode = 1` (per-receiver mode). The size of `min_sizes` must match "
             "the number of receivers.",
             ParameterFlag::kOptional);
  spec.param(min_sum_,
             "min_sum",
             "Minimum sum of message counts",
             "The scheduling term permits execution if the sum of message counts of all "
             "receivers have at least the given number of messages available. This option is only "
             "intended for use with `sampling_mode = 0` (sum-of-all mode).",
             ParameterFlag::kOptional);
}

nvidia::gxf::MessageAvailableFrequencyThrottler* MultiMessageAvailableTimeoutCondition::get()
    const {
  return static_cast<nvidia::gxf::MessageAvailableFrequencyThrottler*>(gxf_cptr_);
}

void MultiMessageAvailableTimeoutCondition::initialize() {
  // Automatically convert string or enum to YAML::Node for 'sampling_mode' argument
  auto& current_args = args();

  auto find_it = std::find_if(current_args.begin(), current_args.end(), [](const auto& arg) {
    bool check = (arg.name() == "sampling_mode" &&
                  (arg.arg_type().element_type() == ArgElementType::kString ||
                   arg.arg_type().element_type() == ArgElementType::kCustom) &&
                  arg.arg_type().container_type() == ArgContainerType::kNative);
    return check;
  });

  if (find_it != current_args.end()) {
    bool yaml_conversion_failed = false;
    YAML::Node sampling_mode;
    if (find_it->arg_type().element_type() == ArgElementType::kString) {
      auto mode_string = std::any_cast<std::string>(find_it->value());
      if (mode_string == "SumOfAll") {
        sampling_mode = YAML::Node("SumOfAll");
      } else if (mode_string == "PerReceiver") {
        sampling_mode = YAML::Node("PerReceiver");
      } else {
        HOLOSCAN_LOG_ERROR("Unrecognized sampling mode string value: {}", mode_string);
        yaml_conversion_failed = true;
      }
    } else {
      try {
        auto mode_enum =
            std::any_cast<MultiMessageAvailableTimeoutCondition::SamplingMode>(find_it->value());
        if (mode_enum == MultiMessageAvailableTimeoutCondition::SamplingMode::kSumOfAll) {
          sampling_mode = YAML::Node("SumOfAll");
        } else if (mode_enum == MultiMessageAvailableTimeoutCondition::SamplingMode::kPerReceiver) {
          sampling_mode = YAML::Node("PerReceiver");
        } else {
          HOLOSCAN_LOG_ERROR("Unrecognized sampling mode enum value: {}",
                             static_cast<int>(mode_enum));
          yaml_conversion_failed = true;
        }
      } catch (const std::bad_any_cast& e) {
        HOLOSCAN_LOG_ERROR(
            "Unable to cast 'sampling_mode' argument to a "
            "MultiMessageAvailableTimeoutCondition::SamplingMode enum: {}",
            e.what());
        yaml_conversion_failed = true;
      }
    }
    if (!yaml_conversion_failed) {
      // remove the old, non-YAML version of the argument
      auto new_arg_end =
          std::remove_if(current_args.begin(), current_args.end(), [](const auto& arg) {
            return arg.name() == "sampling_mode";
          });
      current_args.erase(new_arg_end, current_args.end());
      // add the YAML::Node argument
      add_arg(Arg("sampling_mode", sampling_mode));
    }
  }

  // parent class initialize() call must be after the argument modification above
  GXFCondition::initialize();
}

}  // namespace holoscan
