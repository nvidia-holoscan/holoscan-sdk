/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_RESOURCES_GXF_CONDITION_COMBINER_HPP
#define HOLOSCAN_CORE_RESOURCES_GXF_CONDITION_COMBINER_HPP

#include <memory>
#include <string>
#include <vector>

#include <gxf/std/scheduling_term_combiner.hpp>

#include "../../condition.hpp"
#include "../../gxf/gxf_resource.hpp"

namespace holoscan {

/**
 * @brief Base class representing a combiner for multiple conditions.
 *
 * Base class representing a combiner for multiple conditions.
 */
class ConditionCombiner : public gxf::GXFResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(ConditionCombiner, gxf::GXFResource)
  ConditionCombiner() = default;
  ConditionCombiner(const std::string& name, nvidia::gxf::SchedulingTermCombiner* component);

  const char* gxf_typename() const override { return "nvidia::gxf::SchedulingTermCombiner"; }

  nvidia::gxf::SchedulingTermCombiner* get() const;
};

/**
 * @brief Simulates the bitwise OR operation when combining scheduling conditions
 */
class OrConditionCombiner : public ConditionCombiner {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(OrConditionCombiner, ConditionCombiner)
  OrConditionCombiner() = default;
  OrConditionCombiner(const std::string& name, nvidia::gxf::OrSchedulingTermCombiner* component);

  const char* gxf_typename() const override { return "nvidia::gxf::OrSchedulingTermCombiner"; }

  void initialize() override;

  void setup(ComponentSpec& spec) override;

  nvidia::gxf::OrSchedulingTermCombiner* get() const;

 private:
  Parameter<std::vector<std::shared_ptr<holoscan::Condition>>> terms_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_GXF_CONDITION_COMBINER_HPP */
