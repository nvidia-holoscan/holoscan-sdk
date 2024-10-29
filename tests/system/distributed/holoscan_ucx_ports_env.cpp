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

#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

#include "holoscan/core/system/network_utils.hpp"

#include "../env_wrapper.hpp"
#include "utility_apps.hpp"

#include "distributed_app_fixture.hpp"

namespace holoscan {

TEST(HOLOSCAN_UCX_PORTS, TestGetPreferredNetworkPorts) {
  // Test default value
  {
    EnvVarWrapper wrapper("HOLOSCAN_UCX_PORTS", "");
    auto ports = get_preferred_network_ports("HOLOSCAN_UCX_PORTS");
    EXPECT_EQ(ports.size(), 0);
  }
  // Test single value
  {
    EnvVarWrapper wrapper("HOLOSCAN_UCX_PORTS", "10000");
    auto ports = get_preferred_network_ports("HOLOSCAN_UCX_PORTS");
    EXPECT_EQ(ports.size(), 1);
    EXPECT_EQ(ports[0], 10000);
  }
  // Test multiple values
  {
    EnvVarWrapper wrapper("HOLOSCAN_UCX_PORTS", "10000,20000");
    auto ports = get_preferred_network_ports("HOLOSCAN_UCX_PORTS");
    EXPECT_EQ(ports.size(), 2);
    EXPECT_EQ(ports[0], 10000);
    EXPECT_EQ(ports[1], 20000);
  }
  // Test value having space
  {
    EnvVarWrapper wrapper("HOLOSCAN_UCX_PORTS", "10000, 20000");  // having space
    auto ports = get_preferred_network_ports("HOLOSCAN_UCX_PORTS");
    EXPECT_EQ(ports.size(), 2);
    EXPECT_EQ(ports[0], 10000);
    EXPECT_EQ(ports[1], 20000);
  }
  // Test incorrect values 1
  {
    EnvVarWrapper wrapper("HOLOSCAN_UCX_PORTS", "0,10000,20000,65536");  // 65536 is wrong
    auto ports = get_preferred_network_ports("HOLOSCAN_UCX_PORTS");
    EXPECT_EQ(ports.size(), 0);
  }
}

TEST(HOLOSCAN_UCX_PORTS, TestUnusedNetworkPortStartingFromCandidatePort) {
  auto log_level_orig = log_level();

  {
    auto candidates = generate_random_ports(1, 11000, 60000);

    // Unset HOLOSCAN_LOG_LEVEL environment variable so that the log level is not overridden
    unsetenv("HOLOSCAN_LOG_LEVEL");
    // use DEBUG log level to be able to check detailed messages in the output
    set_log_level(LogLevel::DEBUG);

    // capture output so that we can check that the expected value is present
    testing::internal::CaptureStderr();

    // Collect three unused network ports starting from a candidate port for verification
    auto unused_ports = get_unused_network_ports(3, 11000, 60000, {}, candidates);
    auto verification_str = fmt::format("unused_ports={}", fmt::join(unused_ports, ","));

    std::string log_output = testing::internal::GetCapturedStderr();

    EXPECT_EQ(unused_ports.size(), 3) << "=== LOG ===\n" << log_output << "\n===========\n";
    // The unused port should be the same with the candidate port or within the range of the
    // candidate port.
    EXPECT_TRUE(are_ports_in_vector(unused_ports, candidates, 100))
        << "=== LOG ===\n"
        << log_output << "\n===========\n";
    EXPECT_TRUE(log_output.find(verification_str) != std::string::npos)
        << "=== LOG ===\n"
        << log_output << "\n===========\n";
  }

  // restore the log level
  set_log_level(log_level_orig);
}

TEST(HOLOSCAN_UCX_PORTS, TestUnusedNetworkPortFromTwoCandidatePorts) {
  auto log_level_orig = log_level();

  {
    auto candidates = generate_random_ports(2, 11000, 60000);

    // Unset HOLOSCAN_LOG_LEVEL environment variable so that the log level is not overridden
    unsetenv("HOLOSCAN_LOG_LEVEL");
    // use DEBUG log level to be able to check detailed messages in the output
    set_log_level(LogLevel::DEBUG);

    // capture output so that we can check that the expected value is present
    testing::internal::CaptureStderr();

    // Collect three unused network ports starting from candidate ports for verification
    auto unused_ports = get_unused_network_ports(3, 11000, 60000, {}, candidates);
    auto verification_str = fmt::format("unused_ports={}", fmt::join(unused_ports, ","));

    std::string log_output = testing::internal::GetCapturedStderr();

    EXPECT_EQ(unused_ports.size(), 3) << "=== LOG ===\n" << log_output << "\n===========\n";
    // The unused port should be the same with the candidate port or within the range of the
    // candidate port.
    EXPECT_TRUE(are_ports_in_vector(unused_ports, candidates, 100))
        << "=== LOG ===\n"
        << log_output << "\n===========\n";
    EXPECT_TRUE(log_output.find(verification_str) != std::string::npos)
        << "=== LOG ===\n"
        << log_output << "\n===========\n";
  }

  // restore the log level
  set_log_level(log_level_orig);
}

}  // namespace holoscan
