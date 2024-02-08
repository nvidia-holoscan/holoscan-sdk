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

#include "env_wrapper.hpp"

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

EnvVarWrapper::EnvVarWrapper(
    const std::vector<std::pair<std::string, std::string>>& env_var_settings)
    : env_var_settings_(env_var_settings) {
  // Save existing environment variables and apply new ones
  for (const auto& [env_var, value] : env_var_settings_) {
    const char* orig_value = getenv(env_var.c_str());
    if (orig_value) { orig_env_vars_[env_var] = orig_value; }
    setenv(env_var.c_str(), value.c_str(), 1);
  }
}
EnvVarWrapper::EnvVarWrapper(std::string key, std::string value) : EnvVarWrapper({{key, value}}) {}

EnvVarWrapper::~EnvVarWrapper() {
  // Restore original environment variables
  for (const auto& [env_var, _] : env_var_settings_) {
    auto it = orig_env_vars_.find(env_var);
    if (it == orig_env_vars_.end()) {
      unsetenv(env_var.c_str());
    } else {
      setenv(env_var.c_str(), it->second.c_str(), 1);
    }
  }
}
