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
#include <string>
#include <vector>

namespace holoscan {

/**
 * @brief Generate a list of unused network ports within a specified range.
 *
 * This method generates a vector of unused network ports based on the specified
 * parameters. The generated ports are guaranteed to be unused at the time of
 * generation. The generated ports can be used to bind to a network socket.
 * The generated ports are not guaranteed to remain unused after generation.
 *
 * @param num_ports The number of unused ports to generate (default: 1).
 * @param min_port The minimum value of the port range (default: 10000).
 * @param max_port The maximum value of the port range (default: 32767).
 * @param used_ports The vector of ports to exclude from the generated list (default: empty).
 * @param prefer_ports The vector of ports to prefer in the generated list (default: empty).
 * @return The vector containing the generated unused network ports.
 */
std::vector<int> get_unused_network_ports(uint32_t num_ports = 1, uint32_t min_port = 10000,
                                          uint32_t max_port = 32767,
                                          const std::vector<int>& used_ports = {},
                                          const std::vector<int>& prefer_ports = {});

/**
 * @brief Get the preferred network ports from the specified environment variable.
 *
 * This method reads the specified environment variable and returns a vector of preferred network
 * ports.
 * The environment variable is expected to be a comma-separated list of integers.
 * If the environment variable is not set, is empty, or contains invalid or non-integer values,
 * an empty vector is returned.
 *
 * @param env_var_name The name of the environment variable to read.
 * @return The vector containing the preferred network ports.
 *         If the environment variable is not set, is empty, or no valid port numbers are found,
 *         an empty vector is returned.
 */
std::vector<int> get_preferred_network_ports(const char* env_var_name);

/**
 * @brief Get the local IP address associated with a given remote IP address.
 *
 * This function queries the local network interfaces and returns the IP address of the interface
 * that would be used to connect to the given remote IP address. It supports both IPv4 and IPv6
 * addresses.
 *
 * @param remote_ip The remote IP address as a string.
 * @return A string representing the local IP address associated with the remote IP. If no
 * associated local IP could be found, or if an error occurred during the process, an empty string
 * is returned.
 * @note This function uses the getifaddrs and getaddrinfo system calls to retrieve network
 * interface and routing information. If these calls fail, an error message will be logged and the
 * function will return an empty string.
 */
std::string get_associated_local_ip(const std::string& remote_ip);

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_SYSTEM_NETWORK_UTILS_HPP */
