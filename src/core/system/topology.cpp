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
#include "holoscan/core/system/topology.hpp"

#include <hwloc.h>

#include "holoscan/logger/logger.hpp"

namespace holoscan {

Topology::Topology() {
  // Initialize the hwloc topology object
  hwloc_topology_init(reinterpret_cast<hwloc_topology_t*>(&context_));
}

int Topology::load() {
  return hwloc_topology_load(static_cast<hwloc_topology_t>(context_));
}

void* Topology::context() const {
  return context_;
}

Topology::~Topology() {
  // Destroy the hwloc topology object
  hwloc_topology_destroy(static_cast<hwloc_topology_t>(context_));
}

}  // namespace holoscan
