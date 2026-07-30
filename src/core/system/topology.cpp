/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <holoscan/core/system/topology.hpp>

#include <hwloc.h>

#include <holoscan/logger/logger.hpp>

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
