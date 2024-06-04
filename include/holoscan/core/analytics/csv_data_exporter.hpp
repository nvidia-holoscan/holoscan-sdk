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

#ifndef HOLOSCAN_CORE_ANALYTICS_CSVDATA_EXPORTER_HPP
#define HOLOSCAN_CORE_ANALYTICS_CSVDATA_EXPORTER_HPP

#include <fstream>
#include <string>
#include <vector>

#include "./data_exporter.hpp"

namespace holoscan {

// The default output file name for analytics data.
constexpr const char* kAnalyticsOutputFileName = "data.csv";

/**
 * @brief A class to support exporting Holoscan application data in CSV format for Holoscan
 *        Federated Analytics.
 *
 * The directory will be created with the app name in the data root directory if it is not present
 * already. Inside the application directory, a directory with the current timestamp will be
 * created.
 *
 * The output file name can be specified using the environment variable
 * `HOLOSCAN_ANALYTICS_DATA_FILE_NAME`. If not specified, the output file named `data.csv` will
 * be created inside the timestamp directory. The column names are added to the output file as a
 * first row.
 *
 * Using this class mainly involves two steps:
 * - Create `CsvDataExporter` object specifying app name and columns.
 * - Call `export_data()` method to add a single row to the output file.
 *
 * Example:
 *
 * ```cpp
 * #include "holoscan/core/analytics/csv_data_exporter.hpp"
 *
 * void export_data() {
 * const std::string app_name = "sample_app";
 * const std::vector<std::string> columns = {"column1", "column2", "column3"};
 * CsvDataExporter data_exporter(app_name, columns);
 *
 * const std::vector<std::string> data = {"1", "2", "3"};
 * data_exporter.export_data(data);
 * ...
 * }
 * ```
 */
class CsvDataExporter : public DataExporter {
 public:
  /**
   * @brief The constructor creates required directories and CSV file with the specified names.
   *
   * @param app_name The application name.
   *
   * @param columns The column names list which will be added to the CSV file as a first row.
   *
   */
  CsvDataExporter(const std::string& app_name, const std::vector<std::string>& columns);

  ~CsvDataExporter();

  /**
   * @brief Exports given data to a CSV file.
   *
   * Each call to the function will add one more row to the csv file.
   *
   * @param data The data to be written to the CSV file.
   */
  void export_data(const std::vector<std::string>& data) override;

  /**
   * @brief Get the value of analytics output file name environment variable
   *        `HOLOSCAN_ANALYTICS_DATA_FILE_NAME`.
   *
   * @return A string if the environment variable is set else it returns
   *         error code.
   */
  static expected<std::string, ErrorCode> get_analytics_data_file_name_env();

  /**
   * @brief Returns output file name.
   *
   */
  const std::string& output_file_name() const { return file_name_; }

  /**
   * @brief Returns the column names.
   *
   */
  const std::vector<std::string>& columns() const { return columns_; }

 private:
  /**
   * @brief Write one row to a CSV file.
   *
   * Each call to the function will just add one more row to the csv file.
   *
   * @param data The data to be written to the CSV file.
   *             The number of strings passed should be same as the number of columns in CSV file.
   */
  void write_row(const std::vector<std::string>& data);

  std::string file_name_;
  std::vector<std::string> columns_;
  std::ofstream file_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_ANALYTICS_CSVDATA_EXPORTER_HPP */
