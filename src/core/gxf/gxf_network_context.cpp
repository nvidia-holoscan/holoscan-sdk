/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/gxf/gxf_network_context.hpp"

#include "holoscan/core/component_spec.hpp"

// network context initialization is delayed until runtime via
// `GXFExecutor::initialize_network_context`
namespace holoscan::gxf {

void GXFNetworkContext::set_parameters() {
  update_params_from_args();

  // Set Handler parameters
  for (auto& [key, param_wrap] : spec_->params()) { set_gxf_parameter(name_, key, param_wrap); }
}

void GXFNetworkContext::reset_graph_entities() {
  HOLOSCAN_LOG_TRACE(
      "GXFNetworkContext '{}' of type '{}'::reset_graph_entities", gxf_cname_, gxf_typename());

  // Reset GraphEntity of resources_ and spec_->args() of Scheduler
  NetworkContext::reset_graph_entities();

  // Reset the GraphEntity of this GXFNetworkContext itself
  reset_gxf_graph_entity();
}

}  // namespace holoscan::gxf
