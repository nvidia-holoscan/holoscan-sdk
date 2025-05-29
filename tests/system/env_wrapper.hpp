/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SYSTEM_ENV_WRAPPER_HPP
#define SYSTEM_ENV_WRAPPER_HPP

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

/**
 * @brief Wrapper for setting environment variables temporarily.
 *
 * This class is useful for setting environment variables temporarily in a scope.
 *
 * Example usage:
 *
 * ```cpp
 * {
 *     EnvVarWrapper wrapper({
 *         {"HOLOSCAN_CHECK_RECESSION_PERIOD_MS", "10"},
 *         {"HOLOSCAN_MAX_DURATION_MS", "10000"},
 *         {"HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT", "5000"}
 *     });
 *     // You can use make_pair if argument deduction is not available:
 *     //   EnvVarWrapper wrapper({std::make_pair("HOLOSCAN_LOG_LEVEL", "DEBUG")});
 *     // Or you can set a single environment variable:
 *     //   EnvVarWrapper wrapper("HOLOSCAN_CHECK_RECESSION_PERIOD_MS", "10");
 *
 *     // Here you would put your application code that relies on the temporarily set environment
 *     // variables.
 *     // When the wrapper goes out of scope, the environment variables are restored to their
 *     // original values.
 * }
 */
class EnvVarWrapper {
 public:
  /// Constructor takes a vector of pairs (name, value) of environment variables to set
  explicit EnvVarWrapper(
      const std::vector<std::pair<std::string, std::string>>& env_var_settings = {});

  /// Constructor takes a single environment variable to set
  EnvVarWrapper(std::string key, std::string value);

  /// Destructor
  ~EnvVarWrapper();

 private:
  std::vector<std::pair<std::string, std::string>> env_var_settings_;
  std::unordered_map<std::string, std::string> orig_env_vars_;
  int orig_log_level_ = 2;           ///< holoscan::LogLevel::INFO
  int orig_executor_log_level_ = 3;  ///< nvidia::Severity::INFO
};

#endif /* SYSTEM_ENV_WRAPPER_HPP */
