/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SYSTEM_DISTRIBUTED_DISTRIBUTED_APP_FIXTURE_HPP
#define SYSTEM_DISTRIBUTED_DISTRIBUTED_APP_FIXTURE_HPP

#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <holoscan/holoscan.hpp>

#include "holoscan/core/system/network_utils.hpp"

#include "../env_wrapper.hpp"

namespace holoscan {

///////////////////////////////////////////////////////////////////////////////
// Network Utility Functions
///////////////////////////////////////////////////////////////////////////////

static std::vector<int> generate_random_ports(int num_ports, int min_port, int max_port) {
  std::vector<int> ports;
  std::unordered_set<int> used_ports;

  // Initialize a random number generator with a seed based on the current time
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(min_port, max_port);

  for (int i = 0; i < num_ports; ++i) {
    int port = dis(gen);
    while (used_ports.find(port) != used_ports.end()) { port = dis(gen); }
    used_ports.insert(port);
    ports.push_back(port);
  }
  return ports;
}

static bool are_ports_in_vector(const std::vector<int>& ports, const std::vector<int>& vector,
                                int range = 0) {
  for (const auto& port : ports) {
    bool found = false;
    for (const auto& v : vector) {
      if (port >= v && port <= v + range) {
        found = true;
        break;
      }
    }
    if (!found) { return false; }
  }
  return true;
}

}  // namespace holoscan

class DistributedApp : public ::testing::Test {
 protected:
  void SetUp() override {
    using namespace holoscan;

    log_level_orig_ = log_level();
    candidates_ = generate_random_ports(1, 11000, 60000);
    HOLOSCAN_LOG_INFO("candidate port: {}", candidates_[0]);
    env_var_value_ = fmt::format("{}", candidates_[0]);

    unsetenv("HOLOSCAN_LOG_LEVEL");
    set_log_level(LogLevel::DEBUG);

    wrapper_ =
        std::make_unique<EnvVarWrapper>(std::initializer_list<std::pair<std::string, std::string>>{
            {"HOLOSCAN_EXECUTOR_LOG_LEVEL", "INFO"}, {"HOLOSCAN_UCX_PORTS", env_var_value_}});
  }

  void TearDown() override {
    using namespace holoscan;
    set_log_level(log_level_orig_);
  }

  holoscan::LogLevel log_level_orig_;
  std::vector<int> candidates_;
  std::string env_var_value_;
  std::unique_ptr<EnvVarWrapper> wrapper_;
};

#endif /* SYSTEM_DISTRIBUTED_DISTRIBUTED_APP_FIXTURE_HPP */
