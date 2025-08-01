/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/analytics/csv_data_exporter.hpp"

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

#include <holoscan/logger/logger.hpp>

namespace holoscan {
namespace {
constexpr const char* kAnalyticsDataFileNameEnvVarName = "HOLOSCAN_ANALYTICS_DATA_FILE_NAME";
}  // namespace

CsvDataExporter::CsvDataExporter(const std::string& app_name,
                                 const std::vector<std::string>& columns)
    : DataExporter(app_name), columns_(columns) {
  auto data_file_env = CsvDataExporter::get_analytics_data_file_name_env();
  file_name_ = data_file_env ? data_file_env.value() : kAnalyticsOutputFileName;
  std::filesystem::path file_path = std::filesystem::path(directory_name_) / file_name_;
  auto absolute_path = std::filesystem::absolute(file_path);
  file_ = std::ofstream(file_path, std::ios::app);

  if (file_.is_open()) {
    write_row(columns_);
  } else {
    HOLOSCAN_LOG_ERROR("Error: unable to open file '{}'", absolute_path.string());
  }
}

CsvDataExporter::~CsvDataExporter() {
  file_.close();
}

void CsvDataExporter::export_data(const std::vector<std::string>& data) {
  if (data.size() != columns_.size()) {
    HOLOSCAN_LOG_ERROR("Error: the number of values ({}) does not match the number of columns ({})",
                       data.size(),
                       columns_.size());
  }
  write_row(data);
}

expected<std::string, ErrorCode> CsvDataExporter::get_analytics_data_file_name_env() {
  const char* value = std::getenv(kAnalyticsDataFileNameEnvVarName);
  if (value && value[0]) {
    return value;
  } else {
    return make_unexpected(ErrorCode::kNotFound);
  }
}

void CsvDataExporter::write_row(const std::vector<std::string>& data) {
  if (file_.is_open()) {
    if (!data.empty()) {
      auto it = begin(data);
      file_ << *it;
      ++it;
      for (; it != end(data); ++it) {
        file_ << "," << *it;
      }
    }
    file_ << "\n";
  } else {
    HOLOSCAN_LOG_ERROR("Error: unable to open file");
  }
}

}  // namespace holoscan
