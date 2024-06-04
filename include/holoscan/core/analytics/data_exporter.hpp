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

#ifndef HOLOSCAN_CORE_ANALYTICS_DATA_EXPORTER_HPP
#define HOLOSCAN_CORE_ANALYTICS_DATA_EXPORTER_HPP

#include <string>
#include <vector>

#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"

namespace holoscan {

/**
 * @brief A base class to support exporting Holoscan application data for
 *        Federated Analytics.
 *
 * This class will create a directory with the application name passed to the constructor. It will
 * also create a subdirectory based on the current timestamp within the application directory.
 *
 * The root directory for the application data can be specified by using
 * environment variable `HOLOSCAN_ANALYTICS_DATA_DIRECTORY`. If not specified,
 * it will default to current application directory.
 *
 */
class DataExporter {
 public:
  explicit DataExporter(const std::string& app_name);
  virtual ~DataExporter() = default;

  /**
   * @brief Get the value of analytics data directory environment variable
   *        `HOLOSCAN_ANALYTICS_DATA_DIRECTORY`.
   *
   * @return A string if the environment variable is set else it returns
   *         error code.
   */
  static expected<std::string, ErrorCode> get_analytics_data_directory_env();

  /**
   * @brief A pure virtual function that needs to be implemented by subclasses
   *        to export the data in required format.
   *
   * @param Data The data to be written to the CSV file.
   */
  virtual void export_data(const std::vector<std::string>& data) = 0;

  /**
   * @brief Return the application name.
   *
   */
  const std::string& app_name() const { return app_name_; }

  /**
   * @brief Returns a data directory name.
   *
   */
  const std::string& data_directory() const { return directory_name_; }

  /**
   * @brief Remove the data directory and its contents.
   *
   */
  void cleanup_data_directory();

 protected:
  std::string app_name_;
  std::string directory_name_;

 private:
  /**
   * @brief Create a data directory with the current timestamp inside the
   *        application directory.
   */
  void create_data_directory_with_timestamp();
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_ANALYTICS_DATA_EXPORTER_HPP */
