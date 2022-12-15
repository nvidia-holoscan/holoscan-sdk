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

#include <gtest/gtest.h>
#include <gxf/core/gxf.h>

#include <string>

#include "./config.hpp"
#include "holoscan/core/config.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "common/assert.hpp"

static HoloscanTestConfig test_config;

namespace holoscan {

// Fixture that creates a Fragment and initializes a GXF context for it
class TestWithGXFContext : public ::testing::Test {
 protected:
  void SetUp() override {
    F.config(config_file);
    auto context = F.executor().context();

    // Load GXF extensions included in the config file
    // We should do this before we can initialized GXF-based Conditions, Resources or Operators
    const char* manifest_filename = F.config().config_file().c_str();
    const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &manifest_filename, 1,
                                              nullptr};
    GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &load_ext_info));
  }

  Fragment F;
  const std::string config_file = test_config.get_test_data_file("app_config.yaml");
};

#if HOLOSCAN_BUILD_EMERGENT == 1
  // Fixture that creates a Fragment and initializes a GXF context for it loading emergent extension
class TestWithGXFEmergentContext : public ::testing::Test {
 protected:
    void SetUp() override {
      F.config(config_file);
      auto context = F.executor().context();

      // Load GXF extensions included in the config file
      // We should do this before we can initialized GXF-based Conditions, Resources or Operators
      const char* manifest_filename = F.config().config_file().c_str();
      const GxfLoadExtensionsInfo load_ext_info{nullptr, 0, &manifest_filename, 1,
                                                nullptr};
      GXF_ASSERT_SUCCESS(GxfLoadExtensions(context, &load_ext_info));
    }

    Fragment F;
    const std::string config_file = test_config.get_test_data_file("emergent.yaml");
};
#endif

}  // namespace holoscan
