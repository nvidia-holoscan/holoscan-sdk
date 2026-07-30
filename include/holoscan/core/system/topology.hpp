/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_SYSTEM_TOPOLOGY_HPP
#define HOLOSCAN_CORE_SYSTEM_TOPOLOGY_HPP

#include <memory>

namespace holoscan {

/**
 * @brief Topology class
 *
 * This class is responsible for managing the topology of the system.
 * Internally, it uses hwloc library to get the topology information.
 */
class Topology {
 public:
  Topology();
  virtual ~Topology();

  /**
   * @brief Load the topology
   *
   * @return The error code
   */
  int load();

  /**
   * @brief Get the pointer to the topology object
   *
   * @return The pointer to the topology object
   */
  void* context() const;

 protected:
  void* context_ = nullptr;  ///< The pointer to the topology object
};
}  // namespace holoscan

#endif /* HOLOSCAN_CORE_SYSTEM_TOPOLOGY_HPP */
