/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/analytics/data_exporter.hpp"

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>

#include <holoscan/logger/logger.hpp>

namespace holoscan {
namespace {
constexpr const char* kAnalyticsDataDirectoryEnvVarName = "HOLOSCAN_ANALYTICS_DATA_DIRECTORY";
}  // namespace

DataExporter::DataExporter(const std::string& app_name) : app_name_(app_name) {
  create_data_directory_with_timestamp();
}

void DataExporter::create_data_directory_with_timestamp() {
  auto data_directory_env = DataExporter::get_analytics_data_directory_env();
  const std::string current_dir = std::filesystem::current_path().string();
  std::string data_directory = data_directory_env ? data_directory_env.value() : current_dir;

  auto now = std::chrono::system_clock::now();
  auto local_time = std::chrono::system_clock::to_time_t(now);
  auto local_time_str = std::put_time(std::localtime(&local_time), "%Y%m%d%H%M%S");

  std::ostringstream dir_name;
  dir_name << data_directory << "/" << app_name_ << "/" << local_time_str;
  directory_name_ = dir_name.str();

  if (!std::filesystem::exists(directory_name_)) {
    std::filesystem::create_directories(directory_name_);
  }
}

expected<std::string, ErrorCode> DataExporter::get_analytics_data_directory_env() {
  const char* value = std::getenv(kAnalyticsDataDirectoryEnvVarName);
  if (value && value[0]) {
    return value;
  } else {
    return make_unexpected(ErrorCode::kNotFound);
  }
}

void DataExporter::cleanup_data_directory() {
  try {
    std::filesystem::path dir_path(directory_name_);
    dir_path = std::filesystem::absolute(dir_path).parent_path();
    std::uintmax_t removed = std::filesystem::remove_all(dir_path);
    if (removed > 0) {
      HOLOSCAN_LOG_INFO("Cleaned up {} files", removed);
    } else {
      HOLOSCAN_LOG_ERROR("Error: the directory ({}) does not exist or remove operation failed",
                         directory_name_);
    }
  } catch (const std::filesystem::filesystem_error& e) {
    HOLOSCAN_LOG_ERROR("Error: {}", e.what());
  }
}

}  // namespace holoscan
