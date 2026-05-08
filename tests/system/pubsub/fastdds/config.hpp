/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_TESTS_CONFIG_H
#define HOLOSCAN_TESTS_CONFIG_H

#include <cstdlib>
#include <filesystem>
#include <string>

struct HoloscanTestConfig {
  std::string test_folder;
  std::string test_file;
  std::string temp_folder = std::filesystem::temp_directory_path().string();
  std::string get_test_data_file(const std::string& default_value = "app_config.yaml") const {
    // If `test_file` is an absolute path, return it directly
    if (!test_file.empty() && test_file[0] == '/') {
      return test_file;
    } else {
      std::string test_data_folder = test_folder;
      if (test_data_folder.empty()) {
        if (const char* env_p = std::getenv("HOLOSCAN_TESTS_DATA_PATH")) {
          test_data_folder = env_p;
        } else {
          test_data_folder = "tests/data";
        }
      }
      if (test_file.empty()) {
        return test_data_folder + "/" + default_value;
      } else {
        return test_data_folder + "/" + test_file;
      }
    }
  }
};

extern HoloscanTestConfig holoscan_test_config;

#endif  // HOLOSCAN_TESTS_CONFIG_H
