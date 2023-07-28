/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_GXF_GXF_SCHEDULER_HPP
#define HOLOSCAN_CORE_GXF_GXF_SCHEDULER_HPP

#include <memory>
#include <string>
#include <utility>

#include "../scheduler.hpp"
#include "./gxf_component.hpp"
#include "gxf/std/clock.hpp"
#include "../resources/gxf/clock.hpp"

namespace holoscan::gxf {

// note: in GXF there is also a System class that inherits from Component
//       and is the parent of Scheduler
class GXFScheduler : public holoscan::Scheduler, public GXFComponent {
 public:
  HOLOSCAN_SCHEDULER_FORWARD_ARGS_SUPER(GXFScheduler, holoscan::Scheduler)
  GXFScheduler() = default;

  /**
   * @brief Get the Clock used by the scheduler.
   *
   * @return The Clock used by the scheduler.
   */
  virtual std::shared_ptr<Clock> clock() = 0;

  /**
   * @brief Get the type name of the GXF scheduler.
   *
   * The returned string is the type name of the GXF scheduler and is used to
   * create the GXF scheduler.
   *
   * Example: "nvidia::holoscan::GreedyScheduler"
   *
   * @return The type name of the GXF scheduler.
   */
  virtual const char* gxf_typename() const = 0;

  /**
   * @brief Get the GXF Clock pointer.
   *
   * @return The GXF clock pointer used by the scheduler.
   */
  virtual nvidia::gxf::Clock* gxf_clock();
};

}  // namespace holoscan::gxf

#endif /* HOLOSCAN_CORE_GXF_GXF_SCHEDULER_HPP */
