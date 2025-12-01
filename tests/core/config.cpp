/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <string>

namespace holoscan {

TEST(Config, TestDefault) {
  Config C = Config();
  ASSERT_EQ(C.config_file(), "");
  ASSERT_EQ(C.prefix(), "");
  ASSERT_EQ(C.yaml_nodes().size(), 0);
}

TEST(Config, TestNonexistentFile) {
  std::string fname1 = "nonexistent.yaml";

  // constructor called with nonexistent YAML file should throw an exception
  EXPECT_THROW(
      {
        try {
          Config C = Config(fname1, "temp1");
        } catch (const RuntimeError& e) {
          // Verify the exception message contains the expected text
          EXPECT_TRUE(std::string(e.what()).find("Config file 'nonexistent.yaml' doesn't exist") !=
                      std::string::npos);
          throw;
        }
      },
      RuntimeError);
}

}  // namespace holoscan
