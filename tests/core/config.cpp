/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/config.hpp"

#include <string>

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

namespace holoscan {

TEST(Config, TestDefault) {
  Config C = Config();
  ASSERT_TRUE(C.config_file() == "");
  ASSERT_TRUE(C.prefix() == "");
  ASSERT_TRUE(C.yaml_nodes().size() == 0);
}

TEST(Config, TestNonexistantFile) {
  std::string fname1 = "nonexistant.yaml";

  // capture stderr
  testing::internal::CaptureStderr();

  // constructor called with nonexistant YAML file
  Config C = Config(fname1, "temp1");
  ASSERT_TRUE(C.config_file() == fname1);
  ASSERT_TRUE(C.prefix() == "temp1");
  ASSERT_TRUE(C.yaml_nodes().size() == 0);

  // verify expected warning was logged to stderr
  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("[warning]") != std::string::npos);
  EXPECT_TRUE(log_output.find("Config file 'nonexistant.yaml' doesn't exist") != std::string::npos);
}

}  // namespace holoscan
