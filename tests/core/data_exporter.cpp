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

#include <gtest/gtest.h>

#include <stdlib.h>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "common/assert.hpp"
#include "holoscan/core/analytics/csv_data_exporter.hpp"
#include "holoscan/core/analytics/data_exporter.hpp"

namespace holoscan {

TEST(DataExporterAPI, TestDataExporterConstructor) {
  const std::string app_name = "test_app1";
  const std::vector<std::string> columns = {"column1", "column2", "column3"};
  CsvDataExporter csv_data_exporter(app_name, columns);

  ASSERT_EQ(csv_data_exporter.app_name(), app_name);
  ASSERT_EQ(csv_data_exporter.columns().size(), columns.size());
  ASSERT_EQ(csv_data_exporter.output_file_name(), kAnalyticsOutputFileName);

  csv_data_exporter.cleanup_data_directory();
}

TEST(DataExporterAPI, TestDataExporterDefaultDirectory) {
  const std::string app_name = "test_app2";
  const std::vector<std::string> columns = {"column1", "column2", "column3"};
  CsvDataExporter csv_data_exporter(app_name, columns);

  auto data_dir = csv_data_exporter.data_directory();
  ASSERT_TRUE(std::filesystem::exists(data_dir));

  std::filesystem::path path1(data_dir);
  std::filesystem::path path2(std::filesystem::current_path());
  path2 = path2 / app_name;
  std::filesystem::path abs_path1 = std::filesystem::absolute(path1);
  std::filesystem::path abs_path2 = std::filesystem::absolute(path2);
  ASSERT_EQ(abs_path2, abs_path1.parent_path());

  csv_data_exporter.cleanup_data_directory();
}

TEST(DataExporterAPI, TestDataExporterDirectory) {
  const std::string app_name = "test_app3";
  const std::vector<std::string> columns = {"column1", "column2", "column3"};
  auto path = std::filesystem::current_path();
  path = path / "test_dir";
  const std::string root_data_dir = path.string();
  setenv("HOLOSCAN_ANALYTICS_DATA_DIRECTORY", root_data_dir.c_str(), 1);
  CsvDataExporter csv_data_exporter(app_name, columns);

  auto data_dir = csv_data_exporter.data_directory();
  ASSERT_TRUE(std::filesystem::exists(data_dir));

  std::filesystem::path path1(data_dir);
  std::filesystem::path path2(root_data_dir);
  path2 = path2 / app_name;
  std::filesystem::path abs_path1 = std::filesystem::absolute(path1);
  std::filesystem::path abs_path2 = std::filesystem::absolute(path2);
  ASSERT_EQ(abs_path2, abs_path1.parent_path());

  unsetenv("HOLOSCAN_ANALYTICS_DATA_DIRECTORY");
  csv_data_exporter.cleanup_data_directory();
  // Explicitly remove root directory as we have created the new one.
  std::filesystem::remove_all(path);
}

TEST(DataExporterAPI, TestDataExporterDirectoryWithTimestamp) {
  const std::string app_name = "test_app4";
  const std::vector<std::string> columns = {"column1", "column2", "column3"};
  CsvDataExporter csv_data_exporter(app_name, columns);

  auto now = std::chrono::system_clock::now();
  auto local_time = std::chrono::system_clock::to_time_t(now);
  std::ostringstream local_time_ss;
  local_time_ss << std::put_time(std::localtime(&local_time), "%Y%m%d");
  auto data_dir = csv_data_exporter.data_directory();
  std::filesystem::path data_dir_path(data_dir);
  auto file_name_str = data_dir_path.filename().string();
  ASSERT_TRUE(file_name_str.find(local_time_ss.str()) != std::string::npos);

  csv_data_exporter.cleanup_data_directory();
}

TEST(DataExporterAPI, TestDataExporterDirectoryWithTimestampOnEachRun) {
  const std::string app_name = "test_app5";
  const std::vector<std::string> columns = {"column1", "column2", "column3"};
  CsvDataExporter csv_data_exporter1(app_name, columns);

  const std::string app_name2 = "test_app_2";
  CsvDataExporter csv_data_exporter2(app_name2, columns);

  auto data_dir_1 = csv_data_exporter1.data_directory();
  auto data_dir_2 = csv_data_exporter2.data_directory();

  ASSERT_TRUE(data_dir_1 != data_dir_2);

  csv_data_exporter1.cleanup_data_directory();
  csv_data_exporter2.cleanup_data_directory();
}

TEST(DataExporterAPI, TestDataExporterDataFile) {
  const std::string app_name = "test_app6";
  const std::vector<std::string> columns = {"column1", "column2", "column3"};
  CsvDataExporter csv_data_exporter(app_name, columns);

  std::string file_name = csv_data_exporter.data_directory() + "/" + kAnalyticsOutputFileName;
  ASSERT_TRUE(std::filesystem::exists(file_name));
  ASSERT_EQ(csv_data_exporter.output_file_name(), kAnalyticsOutputFileName);

  csv_data_exporter.cleanup_data_directory();
}

TEST(DataExporterAPI, TestDataExporterCustomDataFile) {
  const std::string app_name = "test_app7";
  const std::vector<std::string> columns = {"column1", "column2", "column3"};
  const std::string custom_data_file = "output.csv";
  setenv("HOLOSCAN_ANALYTICS_DATA_FILE_NAME", custom_data_file.c_str(), 1);
  CsvDataExporter csv_data_exporter(app_name, columns);

  std::string file_name = csv_data_exporter.data_directory() + "/" + custom_data_file;
  ASSERT_TRUE(std::filesystem::exists(file_name));
  ASSERT_EQ(csv_data_exporter.output_file_name(), custom_data_file);

  unsetenv("HOLOSCAN_ANALYTICS_DATA_FILE_NAME");
  csv_data_exporter.cleanup_data_directory();
}

TEST(DataExporterAPI, TestDataExporterCsvFileColumns) {
  const std::string app_name = "test_app8";
  const std::vector<std::string> columns = {"column1", "column2", "column3"};
  std::string file_name;
  std::string directory;
  {
    CsvDataExporter csv_data_exporter(app_name, columns);
    directory = csv_data_exporter.data_directory();
    file_name = directory + "/" + csv_data_exporter.output_file_name();
    ASSERT_TRUE(std::filesystem::exists(file_name));
  }

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open());
  std::string line;
  std::getline(file, line);
  file.close();
  ASSERT_EQ(line, "column1,column2,column3");

  // Explicitly remove directory as DatExporter's cleanup function cannot be called here.
  std::filesystem::path abs_path = std::filesystem::absolute(directory);
  std::filesystem::remove_all(abs_path.parent_path());
}

TEST(DataExporterAPI, TestDataExporterCsvData) {
  const std::string app_name = "test_app9";
  const std::vector<std::string> columns = {"column1", "column2", "column3"};
  std::string file_name;
  std::string directory;
  {
    CsvDataExporter csv_data_exporter(app_name, columns);
    directory = csv_data_exporter.data_directory();
    file_name = directory + "/" + csv_data_exporter.output_file_name();
    ASSERT_TRUE(std::filesystem::exists(file_name));
    const std::vector<std::string> csv_data = {"1", "2", "3"};
    csv_data_exporter.export_data(csv_data);
  }

  std::ifstream file(file_name);
  ASSERT_TRUE(file.is_open());
  std::string line;
  std::getline(file, line);
  ASSERT_EQ(line, "column1,column2,column3");
  std::getline(file, line);
  ASSERT_EQ(line, "1,2,3");
  file.close();

  // Explicitly remove directory as DatExporter's cleanup function cannot be called here.
  std::filesystem::path abs_path = std::filesystem::absolute(directory);
  std::filesystem::remove_all(abs_path.parent_path());
}

}  // namespace holoscan
