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

#include "holoscan/core/cli_options.hpp"

#include <string>
#include <utility>  // std::pair

#include "holoscan/logger/logger.hpp"

namespace holoscan {

std::string CLIOptions::parse_port(const std::string& address, const std::string& default_port) {
  auto colon_pos = address.find(':');
  if (colon_pos != std::string::npos) {
    return address.substr(colon_pos + 1);
  } else {
    return default_port;
  }
}

std::pair<std::string, std::string> CLIOptions::parse_address(const std::string& address) {
  std::string ip_address;
  std::string port;

  auto colon_pos = address.find(':');
  if (colon_pos != std::string::npos) {
    ip_address = address.substr(0, colon_pos);
    port = address.substr(colon_pos + 1);
  } else {
    ip_address = address;
    port = "";
  }

  return std::make_pair(ip_address, port);
}

void CLIOptions::print() const {
  HOLOSCAN_LOG_INFO("CLI Options:");
  HOLOSCAN_LOG_INFO("  run_driver: {}", run_driver);
  HOLOSCAN_LOG_INFO("  run_worker: {}", run_worker);
  HOLOSCAN_LOG_INFO("  driver_address: {}", driver_address);
  HOLOSCAN_LOG_INFO("  worker_address: {}", worker_address);
  HOLOSCAN_LOG_INFO("  worker_targets: {}", fmt::join(worker_targets, ", "));
  HOLOSCAN_LOG_INFO("  config_path: {}", config_path);
}

}  // namespace holoscan
