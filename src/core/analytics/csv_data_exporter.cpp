/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/analytics/csv_data_exporter.hpp>

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
