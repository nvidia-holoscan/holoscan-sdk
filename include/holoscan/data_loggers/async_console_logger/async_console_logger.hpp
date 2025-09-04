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

#ifndef HOLOSCAN_DATA_LOGGERS_ASYNC_CONSOLE_LOGGER_ASYNC_CONSOLE_LOGGER_HPP
#define HOLOSCAN_DATA_LOGGERS_ASYNC_CONSOLE_LOGGER_ASYNC_CONSOLE_LOGGER_HPP

#include <atomic>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "async_console_backend.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/resources/async_data_logger.hpp"
#include "holoscan/data_loggers/basic_console_logger/simple_text_serializer.hpp"

namespace holoscan {
namespace data_loggers {

/**
 * @brief Async version of BasicConsoleLogger using dual-queue architecture
 *
 * This class provides the same console logging functionality as BasicConsoleLogger
 * but uses asynchronous processing with separate queues for metadata and data content.
 * This reduces the impact on the main execution thread while maintaining the same
 * output format and functionality.
 *
 * ==Parameters==
 *
 * All parameters from AsyncDataLoggerResource plus:
 *
 * - **serializer** : std::shared_ptr<SimpleTextSerializer>
 *   - Text serialization resource (optional). A SimpleTextSerializer initialized with default
 *     parameters will be automatically added if none is provided.
 *
 * Inherited parameters from DataLoggerResource:
 * - **log_inputs**: bool (optional, default: true)
 * - **log_outputs**: bool (optional, default: true)
 * - **log_metadata**: bool (optional, default: true)
 * - **log_tensor_data_content**: bool (optional, default: true)
 * - **use_scheduler_clock**: bool (optional, default: false)
 * - **clock**: std::shared_ptr<Clock> (optional, default: nullptr)
 * - **allowlist_patterns**: std::vector<std::string> (optional, default: empty)
 * - **denylist_patterns**: std::vector<std::string> (optional, default: empty)
 *
 * See the DataLoggerResource documentation for details on these inherited parameters.
 *
 * AsyncDataLoggerResource specific parameters:
 * - **max_queue_size**: size_t (optional, default: 50000)
 *   - Maximum number of entries in the data queue. When `enable_large_data_queue` is `true,
 *     The data queue handles tensor headers without full tensor content. Otherwise
 *     tensor data content will also be in this queue. In both cases, whether tensor data content
 *     is logged at all is controlled by `log_tensor_data_content`.  Default is 50000.
 * - **worker_sleep_time**: nanoseconds (optional, default: 50000ns = 50μs)
 *   - Sleep duration in nanoseconds when the data queue is empty. Lower values
 *     reduce latency but increase CPU usage. Default is 50000 (50μs).
 * - **queue_policy**: AsyncQueuePolicy (optional, default: AsyncQueuePolicy::kReject)
 *   - The `queue_policy` parameter controls how queue overflow is handled. Can be
 *     `AsyncQueuePolicy::kReject` (default) to reject new items with a warning, or `kRaise` to
 *     throw an exception. In the YAML configuration for this parameter, you can use string
 *     values "reject" or "raise" (case-insensitive).
 * - **large_data_max_queue_size**: size_t (optional, default: 1000)
 *   - Maximum number of entries in the large data queue. Default is 1000.
 * - **large_data_worker_sleep_time**: nanoseconds (optional, default: 50000ns = 50μs)
 *   - Sleep duration in nanoseconds when the large data queue is empty. Lower values
 *     reduce latency but increase CPU usage. Default is 50000 (50μs).
 * - **large_data_queue_policy**: bool (optional, default: AsyncQueuePolicy::kReject)
 *   - The `large_data_queue_policy` parameter controls how queue overflow is handled for the
 *     large data queue. Can be `AsyncQueuePolicy::kReject` (default) to reject new items with a
 *     warning, or `kRaise` to throw an exception. In the YAML configuration for this parameter,
 *     you can use string values "reject" or "raise" (case-insensitive).
 * - **enable_large_data_queue**: bool (optional, default: true);
 *   - Whether to enable the large data queue and worker thread for processing full
 *     tensor content. Default is True.*
 */
class AsyncConsoleLogger : public AsyncDataLoggerResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(AsyncConsoleLogger, AsyncDataLoggerResource)
  AsyncConsoleLogger() = default;

  void setup(ComponentSpec& spec) override;
  void initialize() override;

  // handle logging of GXF::Entity types
  bool log_backend_specific(const std::any& data, const std::string& unique_id,
                            int64_t acquisition_timestamp = -1,
                            const std::shared_ptr<MetadataDictionary>& metadata = nullptr,
                            IOSpec::IOType io_type = IOSpec::IOType::kOutput) override;

 private:
  Parameter<std::shared_ptr<SimpleTextSerializer>> serializer_;
};

}  // namespace data_loggers
}  // namespace holoscan

#endif /* HOLOSCAN_DATA_LOGGERS_ASYNC_CONSOLE_LOGGER_ASYNC_CONSOLE_LOGGER_HPP */
