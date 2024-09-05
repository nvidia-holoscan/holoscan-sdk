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

TEST(HOLOSCAN_UCX_PORTS, TestUCXBroadCastMultiReceiverAppLocal) {
  auto log_level_orig = log_level();

  {
    // When 'HOLOSCAN_UCX_PORTS=50007' and the application requires three ports,
    // the application should use 50007, 50008, and 50009.
    EnvVarWrapper wrapper({std::make_pair("HOLOSCAN_LOG_LEVEL", "DEBUG"),
                           std::make_pair("HOLOSCAN_EXECUTOR_LOG_LEVEL", "INFO"),
                           std::make_pair("HOLOSCAN_UCX_PORTS", "50007")});

    // Unset HOLOSCAN_LOG_LEVEL environment variable so that the log level is not overridden
    unsetenv("HOLOSCAN_LOG_LEVEL");
    // use DEBUG log level to be able to check detailed messages in the output
    set_log_level(LogLevel::DEBUG);

    // Collect three unused network ports starting from 50007 for verification
    auto unused_ports = get_unused_network_ports(3, 50007, 65535, {}, {50007});
    EXPECT_EQ(unused_ports.size(), 3);
    EXPECT_GE(unused_ports[0], 50007);
    EXPECT_GT(unused_ports[1], unused_ports[0]);
    EXPECT_GT(unused_ports[2], unused_ports[1]);
    EXPECT_LE(unused_ports[2], 65535);

    auto verification_str = fmt::format("unused_ports={}", fmt::join(unused_ports, ","));

    // 'AppDriver::launch_fragments_async()' path will be tested.
    auto app = make_application<UCXBroadCastMultiReceiverApp>();

    // capture output so that we can check that the expected value is present
    testing::internal::CaptureStderr();

    app->run();

    std::string log_output = testing::internal::GetCapturedStderr();

    EXPECT_TRUE(log_output.find(verification_str) != std::string::npos)
        << "=== LOG ===\n"
        << log_output << "\n===========\n";
    EXPECT_TRUE(log_output.find("RxParam fragment2.rx message received (count: 10, size: 2)") !=
                std::string::npos)
        << "=== LOG ===\n"
        << log_output << "\n===========\n";
    EXPECT_TRUE(log_output.find("Rx fragment4.rx message received count: 10") != std::string::npos)
        << "=== LOG ===\n"
        << log_output << "\n===========\n";
  }

  // restore the log level
  set_log_level(log_level_orig);
}

TEST(HOLOSCAN_UCX_PORTS, TestUCXBroadCastMultiReceiverAppWorker) {
  auto log_level_orig = log_level();

  {
    // When 'HOLOSCAN_UCX_PORTS=50101,50105' and the application requires three ports,
    // the application should use 50101, 50105, and 50106.
    EnvVarWrapper wrapper({std::make_pair("HOLOSCAN_LOG_LEVEL", "DEBUG"),
                           std::make_pair("HOLOSCAN_EXECUTOR_LOG_LEVEL", "INFO"),
                           std::make_pair("HOLOSCAN_UCX_PORTS", "50101,50105")});

    // Unset HOLOSCAN_LOG_LEVEL environment variable so that the log level is not overridden
    unsetenv("HOLOSCAN_LOG_LEVEL");
    // use DEBUG log level to be able to check detailed messages in the output
    set_log_level(LogLevel::DEBUG);

    // Collect three unused network ports including port numbers 50101, 50105 for verification
    auto unused_ports = get_unused_network_ports(3, 50101, 65535, {}, {50101, 50105});
    EXPECT_EQ(unused_ports.size(), 3);
    EXPECT_GE(unused_ports[0], 50007);
    EXPECT_GT(unused_ports[1], unused_ports[0]);
    EXPECT_GT(unused_ports[2], unused_ports[1]);
    EXPECT_LE(unused_ports[2], 65535);

    auto verification_str = fmt::format("unused_ports={}", fmt::join(unused_ports, ","));

    // With this arguments, this will go through 'AppWorkerServiceImpl::GetAvailablePorts()' path
    std::vector<std::string> args{"app", "--driver", "--worker", "--fragments=all"};
    auto app = make_application<UCXBroadCastMultiReceiverApp>(args);

    // capture output so that we can check that the expected value is present
    testing::internal::CaptureStderr();

    app->run();

    std::string log_output = testing::internal::GetCapturedStderr();

    EXPECT_TRUE(log_output.find(verification_str) != std::string::npos)
        << "=== LOG ===\n"
        << log_output << "\n===========\n";
    EXPECT_TRUE(log_output.find("RxParam fragment2.rx message received (count: 10, size: 2)") !=
                std::string::npos)
        << "=== LOG ===\n"
        << log_output << "\n===========\n";
    EXPECT_TRUE(log_output.find("Rx fragment4.rx message received count: 10") != std::string::npos)
        << "=== LOG ===\n"
        << log_output << "\n===========\n";
  }

  // restore the log level
  set_log_level(log_level_orig);
}

}  // namespace holoscan
