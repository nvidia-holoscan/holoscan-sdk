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

#include <gtest/gtest.h>

#include <string>

#include "holoscan/core/cli_options.hpp"

namespace holoscan {

// Test cases for CLIOptions::parse_port
TEST(CLIOptions, ParsePortIPv4) {
  std::string address = "192.168.1.1:8080";
  std::string default_port = "80";
  EXPECT_EQ(CLIOptions::parse_port(address, default_port), "8080");
}

TEST(CLIOptions, ParsePortIPv6) {
  std::string address = "[2001:db8::1]:8080";
  std::string default_port = "80";
  EXPECT_EQ(CLIOptions::parse_port(address, default_port), "8080");
}

TEST(CLIOptions, ParsePortNoPortIPv4) {
  std::string address = "192.168.1.1";
  std::string default_port = "80";
  EXPECT_EQ(CLIOptions::parse_port(address, default_port), default_port);
}

TEST(CLIOptions, ParsePortNoPortIPv6) {
  std::string address = "2001:db8::1";
  std::string default_port = "80";
  EXPECT_EQ(CLIOptions::parse_port(address, default_port), default_port);
}

TEST(CLIOptions, ParsePortEmptyAddress) {
  std::string address = "";
  std::string default_port = "80";
  EXPECT_EQ(CLIOptions::parse_port(address, default_port), default_port);
}

TEST(CLIOptions, ParsePortEmptyIP) {
  std::string address = ":8080";
  std::string default_port = "80";
  EXPECT_EQ(CLIOptions::parse_port(address, default_port), "8080");
}

// Test cases for CLIOptions::parse_address
TEST(CLIOptions, ParseAddressIPv4) {
  std::string address = "192.168.1.1:8080";
  std::string default_ip = "127.0.0.1";
  std::string default_port = "80";
  auto result = CLIOptions::parse_address(address, default_ip, default_port);
  EXPECT_EQ(result.first, "192.168.1.1");
  EXPECT_EQ(result.second, "8080");
}

TEST(CLIOptions, ParseAddressIPv6) {
  std::string address = "[2001:db8::1]:8080";
  std::string default_ip = "127.0.0.1";
  std::string default_port = "80";
  auto result = CLIOptions::parse_address(address, default_ip, default_port, true);
  EXPECT_EQ(result.first, "[2001:db8::1]");
  EXPECT_EQ(result.second, "8080");
}

TEST(CLIOptions, ParseAddressIPv6NoEnclose) {
  std::string address = "[2001:db8::1]:8080";
  std::string default_ip = "127.0.0.1";
  std::string default_port = "80";
  auto result = CLIOptions::parse_address(address, default_ip, default_port, false);
  EXPECT_EQ(result.first, "2001:db8::1");
  EXPECT_EQ(result.second, "8080");
}

TEST(CLIOptions, ParseAddressNoPortIPv4) {
  std::string address = "192.168.1.1";
  std::string default_ip = "127.0.0.1";
  std::string default_port = "80";
  auto result = CLIOptions::parse_address(address, default_ip, default_port);
  EXPECT_EQ(result.first, "192.168.1.1");
  EXPECT_EQ(result.second, default_port);
}

TEST(CLIOptions, ParseAddressNoPortIPv6) {
  std::string address = "2001:db8::1";
  std::string default_ip = "127.0.0.1";
  std::string default_port = "80";
  auto result = CLIOptions::parse_address(address, default_ip, default_port);
  EXPECT_EQ(result.first, "2001:db8::1");
  EXPECT_EQ(result.second, default_port);
}

TEST(CLIOptions, ParseAddressEmptyAddress) {
  std::string address = "";
  std::string default_ip = "127.0.0.1";
  std::string default_port = "80";
  auto result = CLIOptions::parse_address(address, default_ip, default_port);
  EXPECT_EQ(result.first, default_ip);
  EXPECT_EQ(result.second, default_port);
}

TEST(CLIOptions, ParseAddressEmptyIP) {
  std::string address = ":8080";
  std::string default_ip = "127.0.0.1";
  std::string default_port = "80";
  auto result = CLIOptions::parse_address(address, default_ip, default_port);
  EXPECT_EQ(result.first, default_ip);
  EXPECT_EQ(result.second, "8080");
}

TEST(CLIOptions, ParseAddressEmptyDefaultIP) {
  std::string address = ":8080";
  std::string default_ip = "";
  std::string default_port = "80";
  auto result = CLIOptions::parse_address(address, default_ip, default_port);
  EXPECT_EQ(result.first, "");
  EXPECT_EQ(result.second, "8080");
}

TEST(CLIOptions, ParseAddressEmptyDefaultIPAndPort) {
  std::string address = ":";
  std::string default_ip = "";
  std::string default_port = "";
  auto result = CLIOptions::parse_address(address, default_ip, default_port);
  EXPECT_EQ(result.first, "");
  EXPECT_EQ(result.second, "");
}

TEST(CLIOptions, ParseAddressEncloseIPv6WithoutPort) {
  std::string address = "2001:db8::1";
  std::string default_ip = "127.0.0.1";
  std::string default_port = "80";
  auto result = CLIOptions::parse_address(address, default_ip, default_port, true);
  EXPECT_EQ(result.first, "[2001:db8::1]");
  EXPECT_EQ(result.second, default_port);
}

TEST(CLIOptions, ParseAddressHostnameWithPort) {
  std::string address = "www.nvidia.com:8080";
  std::string default_ip = "127.0.0.1";
  std::string default_port = "80";
  auto result = CLIOptions::parse_address(address, default_ip, default_port, true, false);
  EXPECT_EQ(result.first, "www.nvidia.com");
  EXPECT_EQ(result.second, "8080");
}

TEST(CLIOptions, ParseAddressHostnameWithoutPort1) {
  std::string address = "www.nvidia.com";
  std::string default_ip = "127.0.0.1";
  std::string default_port = "80";
  auto result = CLIOptions::parse_address(address, default_ip, default_port, true, false);
  EXPECT_EQ(result.first, "www.nvidia.com");
  EXPECT_EQ(result.second, default_port);
}

TEST(CLIOptions, ParseAddressHostnameWithoutPort2) {
  std::string address = "localhost";
  std::string default_ip = "127.0.0.1";
  std::string default_port = "80";
  auto result = CLIOptions::parse_address(address, default_ip, default_port, true, false);
  EXPECT_EQ(result.first, "localhost");
  EXPECT_EQ(result.second, default_port);
}

TEST(CLIOptions, ParseAddressHostnameWithoutPort3) {
  std::string address = "localhost";
  std::string default_ip = "0.0.0.0";
  std::string default_port = "80";
  auto result = CLIOptions::parse_address(address, default_ip, default_port, true, true);
  EXPECT_EQ(result.first, "127.0.0.1");  // resolved IP address
  EXPECT_EQ(result.second, default_port);
}

}  // namespace holoscan
