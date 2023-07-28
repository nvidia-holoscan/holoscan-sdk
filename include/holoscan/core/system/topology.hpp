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
