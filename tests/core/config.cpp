/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/config.hpp>

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
