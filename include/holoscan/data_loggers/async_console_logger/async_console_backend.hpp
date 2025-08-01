/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_DATA_LOGGERS_ASYNC_CONSOLE_LOGGER_ASYNC_CONSOLE_BACKEND_HPP
#define HOLOSCAN_DATA_LOGGERS_ASYNC_CONSOLE_LOGGER_ASYNC_CONSOLE_BACKEND_HPP

#include <atomic>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

#include "holoscan/core/resources/async_data_logger.hpp"
#include "holoscan/data_loggers/basic_console_logger/simple_text_serializer.hpp"

namespace holoscan {
namespace data_loggers {

// External declaration of shared mutex for thread-safe console output across backends
extern std::mutex console_output_mutex;

/**
 * @brief Specialized backend for AsyncConsoleLogger that uses SimpleTextSerializer
 *
 * This backend provides the same console logging functionality as BasicConsoleLogger
 * but works with the dual-queue async architecture. It uses SimpleTextSerializer
 * for data serialization and provides formatted console output.
 */
class AsyncConsoleBackend : public AsyncDataLoggerBackend {
 public:
  explicit AsyncConsoleBackend(std::shared_ptr<SimpleTextSerializer> serializer);
  ~AsyncConsoleBackend() override = default;

  bool initialize() override;
  void shutdown() override;
  bool process_data_entry(const DataEntry& entry) override;
  bool process_large_data_entry(const DataEntry& entry) override;
  std::string get_statistics() const override;

  // Configuration methods
  void set_log_metadata(bool enable) { log_metadata_ = enable; }
  void set_log_tensor_data_content(bool enable) { log_tensor_data_content_ = enable; }

 private:
  std::shared_ptr<SimpleTextSerializer> serializer_;
  std::atomic<size_t> entries_written_{0};
  std::atomic<size_t> large_entries_written_{0};
  std::atomic<bool> initialized_{false};
  bool log_metadata_{true};
  bool log_tensor_data_content_{false};

  /**
   * @brief Format and log a data entry (metadata)
   * @return true if the entry was successfully logged, false otherwise
   */
  bool log_entry(const DataEntry& entry);

  /**
   * @brief Format and log a large data entry (full content)
   * @return true if the entry was successfully logged, false otherwise
   */
  bool log_large_entry(const DataEntry& entry);

  /**
   * @brief Get formatted type name for data
   */
  std::string get_data_type_name(const DataEntry& entry) const;

  /**
   * @brief Get formatted type name for large data
   */
  std::string get_large_data_type_name(const DataEntry& entry) const;

  /**
   * @brief Serialize data entry content
   */
  std::string serialize_data_content(const DataEntry& entry) const;

  /**
   * @brief Serialize large data entry content
   */
  std::string serialize_large_data_content(const DataEntry& entry) const;
};

}  // namespace data_loggers
}  // namespace holoscan

#endif /* HOLOSCAN_DATA_LOGGERS_ASYNC_CONSOLE_LOGGER_ASYNC_CONSOLE_BACKEND_HPP */
