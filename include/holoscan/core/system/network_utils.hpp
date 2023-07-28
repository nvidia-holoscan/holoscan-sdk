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

#ifndef HOLOSCAN_CORE_SYSTEM_NETWORK_UTILS_HPP
#define HOLOSCAN_CORE_SYSTEM_NETWORK_UTILS_HPP

#include <cstdint>
#include <vector>

namespace holoscan {

/**
 * @brief Generate a list of unused network ports within a specified range.
 *
 * This method generates a vector of unused network ports based on the specified
 * parameters. It uses the `rand_r()` method to generate random numbers.
 *
 * @param num_ports The number of unused ports to generate (default: 1).
 * @param min_port The minimum value of the port range (default: 10000).
 * @param max_port The maximum value of the port range (default: 32767).
 * @param used_ports The vector of ports to exclude from the generated list (default: empty).
 * @return The vector containing the generated unused network ports.
 */
std::vector<int> get_unused_network_ports(uint32_t num_ports = 1, uint32_t min_port = 10000,
                                          uint32_t max_port = 32767,
                                          const std::vector<int>& used_ports = {});

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_SYSTEM_NETWORK_UTILS_HPP */
