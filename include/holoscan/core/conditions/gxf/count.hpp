/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_CONDITIONS_GXF_COUNT_HPP
#define HOLOSCAN_CORE_CONDITIONS_GXF_COUNT_HPP

#include "../../gxf/gxf_condition.hpp"

namespace holoscan {

class CountCondition : public gxf::GXFCondition {
 public:
  HOLOSCAN_CONDITION_FORWARD_ARGS_SUPER(CountCondition, GXFCondition)
  CountCondition() = default;
  explicit CountCondition(int64_t count) : count_(count) {}

  const char* gxf_typename() const override { return "nvidia::gxf::CountSchedulingTerm"; }

  void count(int64_t count) { count_ = count; }
  int64_t count() { return count_; }

  void setup(ComponentSpec& spec) override;

 private:
  Parameter<int64_t> count_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_CONDITIONS_GXF_COUNT_HPP */
