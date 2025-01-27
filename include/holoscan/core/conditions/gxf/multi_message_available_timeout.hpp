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

#ifndef HOLOSCAN_CORE_CONDITIONS_GXF_MULTI_MESSAGE_AVAILABLE_TIMEOUT_HPP
#define HOLOSCAN_CORE_CONDITIONS_GXF_MULTI_MESSAGE_AVAILABLE_TIMEOUT_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gxf/std/scheduling_terms.hpp>

#include "../../gxf/gxf_condition.hpp"
#include "../../resource.hpp"
#include "../../resources/gxf/realtime_clock.hpp"

namespace holoscan {

/**
 * @brief Condition class that allows an operator to execute only when one or more messages are
 * available across the specified input ports or a timeout interval since the previous `compute`
 * call has been reached.
 *
 * This condition applies to a specific set of input ports of the operator as determined by setting
 * the "receivers" argument. It can operate in one of two modes:
 *
 *  - "SumOfAll" mode: The condition checks if the sum of messages available across all input ports
 *    is greater than or equal to a given threshold. For this mode, `min_sum` should be specified.
 *  - "PerReceiver" mode: The condition checks if the number of messages available at each input
 *    port is greater than or equal to a given threshold. For this mode, `min_sizes` should be
 *    specified. This mode is equivalent to assigning individiaul MessageAvailableConditions to
 *    each of the receivers.
 *
 * **Note:** This condition is typically set via the `Operator::multi_port_condition`
 * method using `ConditionType::kMultiMessageAvailableTimeout`. The "receivers" argument must be set
 * based on the input port names as described in the "Parameters" section below.
 *
 * **Note:** This condition can also be used on a single port as a way to have a message-available
 * condition that also that supports a timeout interval. For this single input port use case,
 * the condition can be added within `Operator::setup` using the `IOSpec::condition` method with
 * condition type `ConditionType::kMultiMessageAvailableTimeout`. In this case, the input port is
 * already known from the `IOSpec` object, so the "receivers" argument is unnecessary.
 *
 * ==Parameters==
 *
 * - **sampling_mode** (std::string or MultiMessageAvailableCondition::SamplingMode) : The mode of
 * operation of this condition (see above). Options are currently "SumOfAll" or "PerReceiver".
 * - **min_sizes** (std::vector<size_t>): The condition permits execution if all given receivers
 * have at least the given number of messages available in this list. This option is only intended
 * for use with "PerReceiver" `sampling_mode`. The length of `min_sizes` must match the number of
 * receivers associated with the condition.
 * - **min_sum** (size_t): The condition permits execution if the sum of message counts of all
 * receivers have at least the given number of messages available. This option is only intended for
 * use with the "SumOfAll" `sampling_mode`.
 * - **execution_frequency** (std::string): The 'execution frequency' indicates the amount of time
 * after which the entity will be allowed to execute again, even if the specified number of
 * messages have not yet been received. The period is specified as a string containing  of a number
 * and an (optional) unit. If no unit is given, the value is assumed to be in nanoseconds. Supported
 * units are: Hz, s, ms. Examples: "10ms", "10000000", "0.2s", "50Hz".
 * - **receivers** (std::vector<std::string>): The receivers whose message queues will be checked.
 * This should be specified by a vector containing the names of the Operator's input ports the
 * condition will apply to. The Holoscan SDK will then automatically replace the port names with
 * the actual receiver objects at application run time.
 */
class MultiMessageAvailableTimeoutCondition : public gxf::GXFCondition {
 public:
  HOLOSCAN_CONDITION_FORWARD_ARGS_SUPER(MultiMessageAvailableTimeoutCondition, GXFCondition)

  /**
   * @brief sampling mode to apply to the conditions across the input ports (receivers).
   *
   * SamplingMode::kSumOfAll    - min_sum specified is for the sum of all messages at all receivers
   * SamplingMode::kPerReceiver - min_sizes specified is a minimum size per receiver connected
   */
  using SamplingMode = nvidia::gxf::SamplingMode;

  MultiMessageAvailableTimeoutCondition() = default;

  const char* gxf_typename() const override {
    return "nvidia::gxf::MessageAvailableFrequencyThrottler";
  }

  void receivers(std::vector<std::shared_ptr<Receiver>> receivers) { receivers_ = receivers; }
  std::vector<std::shared_ptr<Receiver>>& receivers() { return receivers_.get(); }

  void initialize() override;

  void setup(ComponentSpec& spec) override;

  nvidia::gxf::MessageAvailableFrequencyThrottler* get() const;

 private:
  Parameter<std::vector<std::shared_ptr<Receiver>>> receivers_;
  Parameter<std::string> execution_frequency_;
  Parameter<size_t> min_sum_;
  Parameter<std::vector<size_t>> min_sizes_;
  // use YAML::Node because GXFParameterAdaptor doesn't have a type specific to SamplingMode
  Parameter<YAML::Node> sampling_mode_;  // corresponds to nvidia::gxf::SamplingMode
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_CONDITIONS_GXF_MULTI_MESSAGE_AVAILABLE_TIMEOUT_HPP */
