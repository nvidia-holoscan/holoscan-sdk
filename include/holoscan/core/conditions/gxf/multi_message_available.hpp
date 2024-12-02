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
