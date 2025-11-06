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

#include "holoscan/core/resources/async_data_logger.hpp"

#include <chrono>
#include <future>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "magic_enum.hpp"

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/domain/tensor_map.hpp"
#include "holoscan/core/metadata.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan {

// DataEntry constructors
DataEntry::DataEntry(std::any data_arg, const std::string& id, int64_t acq_time, int64_t emit_time,
                     IOSpec::IOType io_type_arg, std::shared_ptr<MetadataDictionary> meta,
                     std::optional<cudaStream_t> stream_arg)
    : type(Generic),
      unique_id(id),
      acquisition_timestamp(acq_time),
      emit_timestamp(emit_time),
      io_type(io_type_arg),
      metadata(std::move(meta)),
      stream(stream_arg),
      data(std::move(data_arg)) {}

DataEntry::DataEntry(std::shared_ptr<holoscan::Tensor> tensor, const std::string& id,
                     int64_t acq_time, int64_t emit_time, IOSpec::IOType io_type_arg,
                     std::shared_ptr<MetadataDictionary> meta,
                     std::optional<cudaStream_t> stream_arg)
    : type(TensorData),
      unique_id(id),
      acquisition_timestamp(acq_time),
      emit_timestamp(emit_time),
      io_type(io_type_arg),
      metadata(std::move(meta)),
      stream(stream_arg),
      data(std::move(tensor)) {}

DataEntry::DataEntry(holoscan::TensorMap tensor_map, const std::string& id, int64_t acq_time,
                     int64_t emit_time, IOSpec::IOType io_type_arg,
                     std::shared_ptr<MetadataDictionary> meta,
                     std::optional<cudaStream_t> stream_arg)
    : type(TensorMapData),
      unique_id(id),
      acquisition_timestamp(acq_time),
      emit_timestamp(emit_time),
      io_type(io_type_arg),
      metadata(std::move(meta)),
      stream(stream_arg),
      data(std::move(tensor_map)) {}

// AsyncDataLoggerResource implementation
AsyncDataLoggerResource::~AsyncDataLoggerResource() {
  try {
    stop_worker_threads();
  } catch (const std::exception& e) {
    try {
      HOLOSCAN_LOG_ERROR("Excedtion raised in stop_worker_threads(): {}", e.what());
    } catch (...) {
      // discard any exception from fmt::format
    }
  }
}

void AsyncDataLoggerResource::setup(ComponentSpec& spec) {
  // Call parent setup first
  DataLoggerResource::setup(spec);

  // Data queue parameters
  spec.param(max_queue_size_,
             "max_queue_size",
             "Data Maximum Queue Size",
             "Maximum number of entries in the data queue",
             static_cast<size_t>(50000));

  spec.param(worker_sleep_time_,
             "worker_sleep_time",
             "Data Worker Sleep Time",
             "Nanoseconds to sleep when data queue is empty",
             static_cast<int64_t>(50000));  // 50μs in nanoseconds

  spec.param(queue_policy_,
             "queue_policy",
             "Data Queue Policy",
             "Policy for handling queue overflow (kReject or kRaise)",
             AsyncQueuePolicy::kReject);

  // Large data queue parameters
  spec.param(large_data_max_queue_size_,
             "large_data_max_queue_size",
             "Large Data Maximum Queue Size",
             "Maximum number of entries in the large data queue",
             static_cast<size_t>(1000));

  spec.param(large_data_worker_sleep_time_,
             "large_data_worker_sleep_time",
             "Large Data Worker Sleep Time",
             "Nanoseconds to sleep when large data queue is empty",
             static_cast<int64_t>(50000));  // 50μs in nanoseconds

  spec.param(large_data_queue_policy_,
             "large_data_queue_policy",
             "Large Data Queue Policy",
             "Policy for handling large data queue overflow (kReject or kRaise)",
             AsyncQueuePolicy::kReject);

  spec.param(enable_large_data_queue_,
             "enable_large_data_queue",
             "Enable Large Data Queue",
             "Whether to enable the large data queue and worker thread. If this is disabled, "
             "tensor data contents will not be logged (only the Tensor shape, dtype, etc.)",
             true);  // Enable large data queue by default
}

void AsyncDataLoggerResource::initialize() {
  // register argument setter for custom enum before calling parent class initialize
  register_converter<AsyncQueuePolicy>();

  // calling parent initialize will set all parameters from the provided arguments
  DataLoggerResource::initialize();

  // Create the lock-free queues
  data_queue_ = std::make_unique<moodycamel::ConcurrentQueue<DataEntry>>(max_queue_size_.get());

  // Conditionally create large data queue
  if (enable_large_data_queue_.get()) {
    large_data_queue_ =
        std::make_unique<moodycamel::ConcurrentQueue<DataEntry>>(large_data_max_queue_size_.get());
  }

  // Start the worker threads
  if (!start_worker_threads()) {
    throw std::runtime_error("AsyncDataLoggerResource: Failed to start worker threads");
  }

  if (enable_large_data_queue_.get()) {
    HOLOSCAN_LOG_INFO("AsyncDataLoggerResource initialized - queue: {}, Large queue: {}",
                      max_queue_size_.get(),
                      large_data_max_queue_size_.get());
  } else {
    HOLOSCAN_LOG_INFO("AsyncDataLoggerResource initialized - queue: {}, Large queue: disabled",
                      max_queue_size_.get());
  }
}

void AsyncDataLoggerResource::shutdown() {
  stop_worker_threads();
}

bool AsyncDataLoggerResource::log_tensor_data(const std::shared_ptr<Tensor>& tensor,
                                              const std::string& unique_id,
                                              int64_t acquisition_timestamp,
                                              const std::shared_ptr<MetadataDictionary>& metadata,
                                              IOSpec::IOType io_type,
                                              std::optional<cudaStream_t> stream) {
  HOLOSCAN_LOG_TRACE("AsyncDataLoggerResource: log_tensor_data called for unique_id: {}",
                     unique_id);

  // Check filtering conditions
  if (io_type == IOSpec::IOType::kOutput && !should_log_output()) {
    return true;
  }
  if (io_type == IOSpec::IOType::kInput && !should_log_input()) {
    return true;
  }
  if (!should_log_message(unique_id)) {
    return true;
  }

  // metadata is often updated in-place, so we need to make a copy here at the time of logging if
  // it is to be logged.
  bool log_metadata = log_metadata_.get();
  std::shared_ptr<MetadataDictionary> metadata_copy;
  if (log_metadata && metadata) {
    metadata_copy = std::make_shared<MetadataDictionary>(*metadata);
  }

  int64_t emit_timestamp = get_timestamp();

  // Always log tensor (data - without content when serialized)
  bool should_log_content = should_log_tensor_data_content();
  bool large_data_enabled = enable_large_data_queue_.get();

  // Conditionally log full tensor content (large data)
  bool success = true;
  if (should_log_content && large_data_enabled) {
    try {
      DataEntry large_entry(
          tensor, unique_id, acquisition_timestamp, emit_timestamp, io_type, metadata_copy, stream);
      success = enqueue_large_data_entry(std::move(large_entry));
    } catch (const std::exception& e) {
      // Log the exception instead of crashing the application
      HOLOSCAN_LOG_ERROR(
          "AsyncDataLoggerResource: Exception during large data enqueueing for {}: {}",
          unique_id,
          e.what());
      success = false;
    }
  }

  // Log to data queue in these cases:
  // 1. Large data queue is disabled
  // 2. should_log_content is false
  // 3. Large data enqueue failed
  if (!large_data_enabled || !should_log_content || !success) {
    try {
      DataEntry data_entry(
          tensor, unique_id, acquisition_timestamp, emit_timestamp, io_type, metadata_copy, stream);

      success = enqueue_data_entry(std::move(data_entry));
    } catch (const std::exception& e) {
      // Log the exception instead of crashing the application
      HOLOSCAN_LOG_ERROR("AsyncDataLoggerResource: Exception during data enqueueing for {}: {}",
                         unique_id,
                         e.what());
      success = false;
    }
  }
  return success;  // Consider success if either large or data was logged
}

bool AsyncDataLoggerResource::log_data(const std::any& data, const std::string& unique_id,
                                       int64_t acquisition_timestamp,
                                       const std::shared_ptr<MetadataDictionary>& metadata,
                                       IOSpec::IOType io_type, std::optional<cudaStream_t> stream) {
  // Check filtering conditions
  if (io_type == IOSpec::IOType::kOutput && !should_log_output()) {
    return true;
  }
  if (io_type == IOSpec::IOType::kInput && !should_log_input()) {
    return true;
  }
  if (!should_log_message(unique_id)) {
    return true;
  }

  // metadata is often updated in-place, so we need to make a copy here at the time of logging if
  // it is to be logged.
  bool log_metadata = log_metadata_.get();
  std::shared_ptr<MetadataDictionary> metadata_copy;
  if (log_metadata && metadata) {
    metadata_copy = std::make_shared<MetadataDictionary>(*metadata);
  }

  int64_t emit_timestamp = get_timestamp();

  // Log generic data (always goes to data queue)
  try {
    DataEntry data_entry(
        data, unique_id, acquisition_timestamp, emit_timestamp, io_type, metadata_copy, stream);

    return enqueue_data_entry(std::move(data_entry));
  } catch (const std::exception& e) {
    // Log the exception instead of crashing the application
    HOLOSCAN_LOG_ERROR("AsyncDataLoggerResource: Exception during data enqueueing for {}: {}",
                       unique_id,
                       e.what());
    return false;
  }
}

bool AsyncDataLoggerResource::log_tensormap_data(
    const holoscan::TensorMap& tensor_map, const std::string& unique_id,
    int64_t acquisition_timestamp, const std::shared_ptr<MetadataDictionary>& metadata,
    IOSpec::IOType io_type, std::optional<cudaStream_t> stream) {
  HOLOSCAN_LOG_TRACE("AsyncDataLoggerResource: log_tensormap_data called for unique_id: {}",
                     unique_id);

  // Check filtering conditions
  if (io_type == IOSpec::IOType::kOutput && !should_log_output()) {
    return true;
  }
  if (io_type == IOSpec::IOType::kInput && !should_log_input()) {
    return true;
  }
  if (!should_log_message(unique_id)) {
    return true;
  }

  int64_t emit_timestamp = get_timestamp();

  // Always log tensormap (data - without content when serialized)
  bool should_log_content = should_log_tensor_data_content();
  bool large_data_enabled = enable_large_data_queue_.get();

  // metadata is often updated in-place, so we need to make a copy here at the time of logging if
  // it is to be logged.
  bool log_metadata = log_metadata_.get();
  std::shared_ptr<MetadataDictionary> metadata_copy;
  if (log_metadata && metadata) {
    metadata_copy = std::make_shared<MetadataDictionary>(*metadata);
  }

  // Conditionally log full tensormap content (large data)
  bool success = true;
  if (should_log_content && large_data_enabled) {
    try {
      DataEntry large_entry(tensor_map,
                            unique_id,
                            acquisition_timestamp,
                            emit_timestamp,
                            io_type,
                            metadata_copy,
                            stream);
      success = enqueue_large_data_entry(std::move(large_entry));
    } catch (const std::exception& e) {
      // Log the exception instead of crashing the application
      HOLOSCAN_LOG_ERROR(
          "AsyncDataLoggerResource: Exception during large data enqueueing for {}: {}",
          unique_id,
          e.what());
      success = false;
    }
  }

  // Log to data queue in these cases:
  // 1. Large data queue is disabled
  // 2. should_log_content is false
  // 3. Large data enqueue failed
  if (!large_data_enabled || !should_log_content || !success) {
    try {
      DataEntry data_entry(tensor_map,
                           unique_id,
                           acquisition_timestamp,
                           emit_timestamp,
                           io_type,
                           metadata_copy,
                           stream);

      success = enqueue_data_entry(std::move(data_entry));
    } catch (const std::exception& e) {
      // Log the exception instead of crashing the application
      HOLOSCAN_LOG_ERROR("AsyncDataLoggerResource: Exception during data enqueueing for {}: {}",
                         unique_id,
                         e.what());
      success = false;
    }
  }

  return success;  // Consider success if metadata was logged
}

std::string AsyncDataLoggerResource::get_statistics() const {
  std::ostringstream ss;
  ss << "AsyncDataLoggerResource Statistics:\n";
  ss << "  Data Queue:\n";
  ss << "    Enqueued: " << data_enqueued_.load() << "\n";
  ss << "    Processed: " << data_processed_.load() << "\n";
  ss << "    Dropped: " << data_dropped_.load() << "\n";
  ss << "    Current Size: " << get_data_queue_size() << "\n";
  ss << "    Max Size: " << max_queue_size_.get() << "\n";

  if (enable_large_data_queue_.get()) {
    ss << "  Large Data Queue:\n";
    ss << "    Enqueued: " << large_data_enqueued_.load() << "\n";
    ss << "    Processed: " << large_data_processed_.load() << "\n";
    ss << "    Dropped: " << large_data_dropped_.load() << "\n";
    ss << "    Current Size: " << get_large_data_queue_size() << "\n";
    ss << "    Max Size: " << large_data_max_queue_size_.get() << "\n";
  } else {
    ss << "  Large Data Queue: Disabled\n";
  }

  ss << "  Workers Running: " << (workers_running_.load() ? "Yes" : "No") << "\n";

  if (backend_initialized_.load(std::memory_order_acquire)) {
    std::string backend_stats = backend_->get_statistics();
    if (!backend_stats.empty()) {
      ss << "  Backend Statistics:\n";
      ss << "    " << backend_stats << "\n";
    }
  }

  return ss.str();
}

size_t AsyncDataLoggerResource::get_data_queue_size() const {
  if (!data_queue_) {
    return 0;
  }
  return data_queue_->size_approx();
}

size_t AsyncDataLoggerResource::get_large_data_queue_size() const {
  if (!large_data_queue_) {
    return 0;
  }
  return large_data_queue_->size_approx();
}

void AsyncDataLoggerResource::set_backend(std::shared_ptr<AsyncDataLoggerBackend> backend) {
  if (workers_running_.load()) {
    HOLOSCAN_LOG_ERROR(
        "Cannot set backend while worker threads are running. "
        "Stop worker threads first to prevent data races.");
    return;
  }
  if (!backend) {
    HOLOSCAN_LOG_ERROR("AsyncDataLoggerResource: Cannot set backend to nullptr");
    return;
  }

  // Set backend first, then clear initialization flag
  backend_ = std::move(backend);
  backend_initialized_.store(false, std::memory_order_release);  // Clear initialization flag

  // Reset shutdown flag when setting a new backend
  backend_shutdown_called_.store(false);
}

// Worker thread and queue management implementations
bool AsyncDataLoggerResource::start_worker_threads() {
  if (workers_running_.load()) {
    HOLOSCAN_LOG_WARN("AsyncDataLoggerResource: Worker threads are already running");
    return true;  // Already running successfully
  }

  // Initialize backend before starting any worker threads
  if (backend_) {
    HOLOSCAN_LOG_DEBUG("AsyncDataLoggerResource: Initializing backend");
    if (!backend_->initialize()) {
      HOLOSCAN_LOG_ERROR("AsyncDataLoggerResource: Failed to initialize backend");
      return false;
    }
    HOLOSCAN_LOG_DEBUG("AsyncDataLoggerResource: Backend initialized successfully");
    backend_initialized_.store(true, std::memory_order_release);  // Set only after successful init
  }

  shutdown_requested_.store(false);

  // Start main data worker thread (always)
  data_worker_ = std::thread(&AsyncDataLoggerResource::data_worker_function, this);

  // Conditionally start large data worker thread
  if (enable_large_data_queue_.get()) {
    large_data_worker_ = std::thread(&AsyncDataLoggerResource::large_data_worker_function, this);
  }

  workers_running_.store(true);

  if (enable_large_data_queue_.get()) {
    HOLOSCAN_LOG_DEBUG("AsyncDataLoggerResource: Worker threads started (data + large)");
  } else {
    HOLOSCAN_LOG_DEBUG("AsyncDataLoggerResource: Worker thread started (data only)");
  }

  return true;
}

void AsyncDataLoggerResource::stop_worker_threads() {
  if (!workers_running_.load()) {
    return;  // Threads not running
  }

  HOLOSCAN_LOG_DEBUG("AsyncDataLoggerResource: Requesting worker threads shutdown");
  shutdown_requested_.store(true);

  // Join worker threads (block until completion)
  if (data_worker_.joinable()) {
    data_worker_.join();
  }

  if (enable_large_data_queue_.get() && large_data_worker_.joinable()) {
    large_data_worker_.join();
  }

  workers_running_.store(false);
  HOLOSCAN_LOG_DEBUG("AsyncDataLoggerResource: Worker threads stopped");

  if (backend_initialized_.load(std::memory_order_acquire) &&
      !backend_shutdown_called_.exchange(true)) {
    try {
      HOLOSCAN_LOG_DEBUG("AsyncDataLoggerResource: Calling backend shutdown");
      backend_->shutdown();
      backend_initialized_.store(false, std::memory_order_release);  // Clear after shutdown
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("AsyncDataLoggerResource: Exception during backend shutdown: {}",
                         e.what());
      backend_initialized_.store(false, std::memory_order_release);  // Clear even on exception
    }
  }
}

void AsyncDataLoggerResource::data_worker_function() {
  HOLOSCAN_LOG_DEBUG("AsyncDataLoggerResource: Data worker thread started");

  DataEntry entry;

  while (!shutdown_requested_.load()) {
    bool processed_any = false;

    // Process all available entries in a batch
    while (data_queue_ && data_queue_->try_dequeue(entry)) {
      processed_any = true;

      if (backend_initialized_.load(std::memory_order_acquire)) {
        try {
          if (backend_->process_data_entry(entry)) {
            data_processed_.fetch_add(1, std::memory_order_relaxed);
          } else {
            HOLOSCAN_LOG_WARN("AsyncDataLoggerResource: Backend failed to process data entry: {}",
                              entry.unique_id);
          }
        } catch (const std::exception& e) {
          HOLOSCAN_LOG_ERROR("AsyncDataLoggerResource: Exception processing data entry {}: {}",
                             entry.unique_id,
                             e.what());
        }
      } else {
        // No backend configured, just count as processed
        data_processed_.fetch_add(1, std::memory_order_relaxed);
        HOLOSCAN_LOG_WARN(
            "AsyncDataLoggerResource: No backend configured, dropping data entry: "
            "{}",
            entry.unique_id);
      }
    }

    // Sleep if no entries were processed
    if (!processed_any) {
      auto sleep_time_ns = worker_sleep_time_.get();
      auto sleep_time = std::chrono::nanoseconds(sleep_time_ns);
      std::this_thread::sleep_for(sleep_time);
    }
  }

  // Process remaining entries during shutdown
  HOLOSCAN_LOG_DEBUG("AsyncDataLoggerResource: Processing remaining data entries during shutdown");
  size_t remaining_entries = 0;
  while (data_queue_ && data_queue_->try_dequeue(entry)) {
    remaining_entries++;
    if (backend_initialized_.load(std::memory_order_acquire)) {
      try {
        if (backend_->process_data_entry(entry)) {
          data_processed_.fetch_add(1, std::memory_order_relaxed);
        }
      } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR(
            "AsyncDataLoggerResource: Exception processing data entry during "
            "shutdown {}: {}",
            entry.unique_id,
            e.what());
      }
    }
  }

  if (remaining_entries > 0) {
    HOLOSCAN_LOG_INFO(
        "AsyncDataLoggerResource: Processed {} remaining data entries during "
        "shutdown",
        remaining_entries);
  }

  HOLOSCAN_LOG_DEBUG("AsyncDataLoggerResource: Data worker thread finished");
}

void AsyncDataLoggerResource::large_data_worker_function() {
  HOLOSCAN_LOG_DEBUG("AsyncDataLoggerResource: Large data worker thread started");

  DataEntry entry;

  while (!shutdown_requested_.load()) {
    bool processed_any = false;

    // Process all available entries in a batch
    while (large_data_queue_ && large_data_queue_->try_dequeue(entry)) {
      processed_any = true;

      if (backend_initialized_.load(std::memory_order_acquire)) {
        try {
          if (backend_->process_large_data_entry(entry)) {
            large_data_processed_.fetch_add(1, std::memory_order_relaxed);
          } else {
            HOLOSCAN_LOG_WARN(
                "AsyncDataLoggerResource: Backend failed to process large data entry: {}",
                entry.unique_id);
          }
        } catch (const std::exception& e) {
          HOLOSCAN_LOG_ERROR(
              "AsyncDataLoggerResource: Exception processing large data entry {}: {}",
              entry.unique_id,
              e.what());
        }
      } else {
        // No backend configured, just count as processed
        large_data_processed_.fetch_add(1, std::memory_order_relaxed);
        HOLOSCAN_LOG_WARN(
            "AsyncDataLoggerResource: No backend configured, dropping large data entry: "
            "{}",
            entry.unique_id);
      }
    }

    // Sleep if no entries were processed
    if (!processed_any) {
      auto sleep_time_ns = large_data_worker_sleep_time_.get();
      auto sleep_time = std::chrono::nanoseconds(sleep_time_ns);
      std::this_thread::sleep_for(sleep_time);
    }
  }

  // Process remaining entries during shutdown
  HOLOSCAN_LOG_DEBUG(
      "AsyncDataLoggerResource: Processing remaining large data entries during shutdown");
  size_t remaining_entries = 0;
  while (large_data_queue_ && large_data_queue_->try_dequeue(entry)) {
    remaining_entries++;
    if (backend_initialized_.load(std::memory_order_acquire)) {
      try {
        if (backend_->process_large_data_entry(entry)) {
          large_data_processed_.fetch_add(1, std::memory_order_relaxed);
        }
      } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR(
            "AsyncDataLoggerResource: Exception processing large data entry during "
            "shutdown {}: {}",
            entry.unique_id,
            e.what());
      }
    }
  }

  if (remaining_entries > 0) {
    HOLOSCAN_LOG_INFO(
        "AsyncDataLoggerResource: Processed {} remaining large data entries during "
        "shutdown",
        remaining_entries);
  }
  HOLOSCAN_LOG_DEBUG("AsyncDataLoggerResource: Large data worker thread finished");
}

bool AsyncDataLoggerResource::enqueue_data_entry(DataEntry&& entry) {
  if (!data_queue_) {
    HOLOSCAN_LOG_ERROR("AsyncDataLoggerResource: Data queue not initialized");
    return false;
  }

  if (!workers_running_.load()) {
    HOLOSCAN_LOG_WARN("AsyncDataLoggerResource: Worker threads not running, dropping data entry");
    return false;
  }

  // Capture unique_id before move for potential error logging
  const std::string unique_id = entry.unique_id;

  // Try to enqueue the entry
  if (data_queue_->try_enqueue(std::move(entry))) {
    data_enqueued_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  // Queue is full
  data_dropped_.fetch_add(1, std::memory_order_relaxed);

  // Handle queue overflow based on policy
  switch (queue_policy_.get()) {
    case AsyncQueuePolicy::kReject:
      HOLOSCAN_LOG_WARN("AsyncDataLoggerResource: Data queue overflow, entry rejected: {}",
                        unique_id);
      return false;
    case AsyncQueuePolicy::kRaise:
      throw std::runtime_error(
          fmt::format("AsyncDataLoggerResource: Data queue overflow for entry: {}", unique_id));
    default:
      HOLOSCAN_LOG_ERROR("AsyncDataLoggerResource: Unknown queue policy: {}",
                         magic_enum::enum_name(queue_policy_.get()));
      return false;
  }
}

bool AsyncDataLoggerResource::enqueue_large_data_entry(DataEntry&& entry) {
  if (!enable_large_data_queue_.get()) {
    throw std::runtime_error(
        "AsyncDataLoggerResource: Large data queue is not enabled. "
        "Set enable_large_data_queue to true to use large data functionality.");
  }

  if (!large_data_queue_) {
    HOLOSCAN_LOG_ERROR("AsyncDataLoggerResource: Large data queue not initialized");
    return false;
  }

  if (!workers_running_.load()) {
    HOLOSCAN_LOG_WARN(
        "AsyncDataLoggerResource: Worker threads not running, dropping large data entry");
    return false;
  }

  // Capture unique_id before move for potential error logging
  const std::string unique_id = entry.unique_id;

  // Try to enqueue the entry
  if (large_data_queue_->try_enqueue(std::move(entry))) {
    large_data_enqueued_.fetch_add(1, std::memory_order_relaxed);
    return true;
  }

  // Queue is full
  large_data_dropped_.fetch_add(1, std::memory_order_relaxed);

  // Handle large data queue overflow based on policy
  switch (large_data_queue_policy_.get()) {
    case AsyncQueuePolicy::kReject:
      HOLOSCAN_LOG_WARN("AsyncDataLoggerResource: Large data queue overflow, entry rejected: {}",
                        unique_id);
      return false;
    case AsyncQueuePolicy::kRaise:
      throw std::runtime_error(fmt::format(
          "AsyncDataLoggerResource: Large data queue overflow for entry: {}", unique_id));
    default:
      HOLOSCAN_LOG_ERROR("AsyncDataLoggerResource: Unknown large data queue policy: {}",
                         magic_enum::enum_name(large_data_queue_policy_.get()));
      return false;
  }
}

}  // namespace holoscan
