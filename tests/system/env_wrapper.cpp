/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "env_wrapper.hpp"

#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "gxf/logger/gxf_logger.hpp"
#include "holoscan/logger/logger.hpp"

EnvVarWrapper::EnvVarWrapper(
    const std::vector<std::pair<std::string, std::string>>& env_var_settings)
    : env_var_settings_(env_var_settings) {
  // Save existing environment variables and apply new ones
  for (const auto& [env_var, value] : env_var_settings_) {
    const char* orig_value = getenv(env_var.c_str());
    if (orig_value) {
      orig_env_vars_[env_var] = orig_value;
    }
    setenv(env_var.c_str(), value.c_str(), 1);
  }
  orig_log_level_ = static_cast<int>(holoscan::log_level());
  nvidia::logger::GxfLogger& gxf_logger = nvidia::logger::GlobalGxfLogger::instance();
  orig_executor_log_level_ = static_cast<int>(gxf_logger.level());
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
  holoscan::set_log_level(static_cast<holoscan::LogLevel>(orig_log_level_));
  nvidia::logger::GxfLogger& gxf_logger = nvidia::logger::GlobalGxfLogger::instance();
  gxf_logger.level(static_cast<int>(orig_executor_log_level_));
}
