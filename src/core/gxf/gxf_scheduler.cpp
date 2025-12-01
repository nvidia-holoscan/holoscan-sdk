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

#include <gxf/core/gxf.h>

#include <string>
#include <vector>

#include <gxf/std/clock.hpp>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/gxf/gxf_scheduler.hpp"

namespace holoscan::gxf {
// scheduler initialization is delayed until runtime via `GXFExecutor::initialize_scheduler`

nvidia::gxf::Clock* GXFScheduler::gxf_clock() {
  if (this->clock()) {
    return static_cast<nvidia::gxf::Clock*>(this->clock_gxf_cptr());
  } else {
    HOLOSCAN_LOG_ERROR("GXFScheduler clock is not set");
    return nullptr;
  }
}

void GXFScheduler::set_parameters() {
  update_params_from_args();

  // Set Handler parameters
  std::vector<std::string> errors;
  for (auto& [key, param_wrap] : spec_->params()) {
    HOLOSCAN_LOG_TRACE("GXFScheduler '{}':: setting GXF parameter '{}'", name_, key);
    try {
      set_gxf_parameter(name_, key, param_wrap);
    } catch (const std::exception& e) {
      std::string error_msg = fmt::format("Parameter '{}': {}", key, e.what());
      HOLOSCAN_LOG_ERROR("GXFScheduler '{}': failed to set GXF parameter - {}", name_, error_msg);
      errors.push_back(error_msg);
    }
  }

  if (!errors.empty()) {
    throw std::runtime_error(
        fmt::format("GXFScheduler '{}' (type '{}'): failed to set {} GXF parameter(s):\n  - {}",
                    name_,
                    gxf_typename(),
                    errors.size(),
                    fmt::join(errors, "\n  - ")));
  }
}

void GXFScheduler::reset_backend_objects() {
  HOLOSCAN_LOG_TRACE(
      "GXFScheduler '{}' of type '{}'::reset_backend_objects", gxf_cname_, gxf_typename());

  // Reset GraphEntity of resources_ and spec_->args() of Scheduler
  Scheduler::reset_backend_objects();

  // Reset the GraphEntity of this GXFScheduler itself
  reset_gxf_graph_entity();
}

YAML::Node GXFScheduler::to_yaml_node() const {
  YAML::Node node = Scheduler::to_yaml_node();
  node["gxf_eid"] = YAML::Node(gxf_eid());
  node["gxf_cid"] = YAML::Node(gxf_cid());
  node["gxf_typename"] = YAML::Node(gxf_typename());
  return node;
}

}  // namespace holoscan::gxf
