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

#ifndef HOLOSCAN_CORE_CLI_OPTIONS_HPP
#define HOLOSCAN_CORE_CLI_OPTIONS_HPP

#include <string>
#include <utility>  // std::pair
#include <vector>

namespace holoscan {

/**
 * @brief CLI Options struct.
 *
 * This struct is used to store the parsed command line arguments.
 */
struct CLIOptions {
  bool run_driver = false;                  ///< The flag to run the App Driver.
  bool run_worker = false;                  ///< The flag to run the App Worker.
  std::string driver_address;               ///< The address of the App Driver.
  std::string worker_address;               ///< The address of the App Worker.
  std::vector<std::string> worker_targets;  ///< The list of fragments for the App Worker.
  std::string config_path;                  ///< The path to the configuration file.

  /**
   * @brief Return the port from the given address.
   *
   * @param address The address to parse.
   * @param default_port The default port to return if the address does not contain a port.
   * @return The port.
   */
  static std::string parse_port(const std::string& address, const std::string& default_port = "");

  /**
   * @brief Return the IP address and port from the given address.
   *
   * @param address The address to parse.
   * @return The IP address and port.
   */
  static std::pair<std::string, std::string> parse_address(const std::string& address);

  /**
   * @brief Print the CLI Options.
   */
  void print() const;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_CLI_OPTIONS_HPP */
