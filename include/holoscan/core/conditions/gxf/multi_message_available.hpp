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

#ifndef HOLOSCAN_CORE_CONDITIONS_GXF_MULTI_MESSAGE_AVAILABLE_HPP
#define HOLOSCAN_CORE_CONDITIONS_GXF_MULTI_MESSAGE_AVAILABLE_HPP

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
 * available across the specified input ports.
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
 * **Note:** This condition **must** currently be set via the `Operator::multi_port_condition`
 * method using `ConditionType::kMultiMessageAvailable`. The "receivers" argument must be set based
 * on the input port names as described in the "Parameters" section below.
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
 * - **receivers** (std::vector<std::string>): The receivers whose message queues will be checked.
 * This should be specified by a vector containing the names of the Operator's input ports the
 * condition will apply to. The Holoscan SDK will then automatically replace the port names with
 * the actual receiver objects at application run time.
 */
class MultiMessageAvailableCondition : public gxf::GXFCondition {
 public:
  HOLOSCAN_CONDITION_FORWARD_ARGS_SUPER(MultiMessageAvailableCondition, GXFCondition)

  /**
   * @brief sampling mode to apply to the conditions across the input ports (receivers).
   *
   * SamplingMode::kSumOfAll    - min_sum specified is for the sum of all messages at all receivers
   * SamplingMode::kPerReceiver - min_sizes specified is a minimum size per receiver connected
   */
  using SamplingMode = nvidia::gxf::SamplingMode;

  MultiMessageAvailableCondition() = default;

  const char* gxf_typename() const override {
    return "nvidia::gxf::MultiMessageAvailableSchedulingTerm";
  }

  void receivers(std::vector<std::shared_ptr<gxf::GXFResource>> receivers) {
    receivers_ = receivers;
  }
  std::vector<std::shared_ptr<gxf::GXFResource>>& receivers() { return receivers_.get(); }

  void initialize() override;

  void setup(ComponentSpec& spec) override;

  // wrap setters available on the underling nvidia::gxf::MultiMessageAvailableSchedulingTerm
  // void min_size(size_t value);  // min_size parameter is deprecated
  void min_sum(size_t value);
  size_t min_sum() { return min_sum_; }

  void sampling_mode(SamplingMode value);
  SamplingMode sampling_mode() {
    std::string mode = sampling_mode_.get().as<std::string>();
    if (mode == "SumOfAll") {
      return SamplingMode::kSumOfAll;
    } else if (mode == "PerReceiver") {
      return SamplingMode::kPerReceiver;
    } else {
      throw std::runtime_error(fmt::format("unknown mode: {}", mode));
    }
  }

  void add_min_size(size_t value);

  std::vector<size_t> min_sizes() { return min_sizes_; }

  nvidia::gxf::MultiMessageAvailableSchedulingTerm* get() const;

 private:
  Parameter<std::vector<std::shared_ptr<gxf::GXFResource>>> receivers_;
  Parameter<size_t> min_sum_;
  Parameter<std::vector<size_t>> min_sizes_;
  // use YAML::Node because GXFParameterAdaptor doesn't have a type specific to SamplingMode
  Parameter<YAML::Node> sampling_mode_;  // corresponds to nvidia::gxf::SamplingMode
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_CONDITIONS_GXF_MULTI_MESSAGE_AVAILABLE_HPP */
