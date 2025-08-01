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

#ifndef HOLOSCAN_CORE_RESOURCES_ASYNC_DATA_LOGGER_HPP
#define HOLOSCAN_CORE_RESOURCES_ASYNC_DATA_LOGGER_HPP

#include <algorithm>
#include <any>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include "yaml-cpp/yaml.h"

#include "concurrentqueue.h"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/domain/tensor_map.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/resources/data_logger.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan {

class MetadataDictionary;  // forward declaration

/**
 * @brief Policy for handling queue overflow in async data loggers
 */
enum class AsyncQueuePolicy {
  kReject = 0,  ///< Reject new items when queue is full (drop and log warning)
  kRaise = 1    ///< Raise an exception when queue is full
};

}  // namespace holoscan

// YAML conversion support for AsyncQueuePolicy
template <>
struct YAML::convert<holoscan::AsyncQueuePolicy> {
  static Node encode(const holoscan::AsyncQueuePolicy& rhs) {
    Node node;
    switch (rhs) {
      case holoscan::AsyncQueuePolicy::kReject:
        node = "reject";
        break;
      case holoscan::AsyncQueuePolicy::kRaise:
        node = "raise";
        break;
      default:
        node = static_cast<int>(rhs);  // fallback to numeric value
        break;
    }
    return node;
  }

  static bool decode(const Node& node, holoscan::AsyncQueuePolicy& rhs) {
    if (!node.IsScalar())
      return false;

    const std::string value = node.Scalar();

    // Support string values (case-insensitive)
    std::string lower_value = value;
    std::transform(
        lower_value.begin(), lower_value.end(), lower_value.begin(), [](unsigned char c) {
          return std::tolower(c);
        });

    if (lower_value == "reject") {
      rhs = holoscan::AsyncQueuePolicy::kReject;
      return true;
    } else if (lower_value == "raise") {
      rhs = holoscan::AsyncQueuePolicy::kRaise;
      return true;
    }

    // Support using the numeric enum values as well
    try {
      int numeric_value = std::stoi(value);
      if (numeric_value == static_cast<int>(holoscan::AsyncQueuePolicy::kReject)) {
        rhs = holoscan::AsyncQueuePolicy::kReject;
        return true;
      } else if (numeric_value == static_cast<int>(holoscan::AsyncQueuePolicy::kRaise)) {
        rhs = holoscan::AsyncQueuePolicy::kRaise;
        return true;
      }
    } catch (...) {
      // Not a valid number, continue to return false
    }

    return false;  // Invalid value
  }
};

namespace holoscan {

/**
 * @brief Entry for data queue (may have metadata only, no tensor content)
 */
struct DataEntry {
  enum Type { Generic, TensorData, TensorMapData };

  Type type;
  std::string unique_id;
  int64_t acquisition_timestamp;  ///< externally specified acquisition timestamp
  int64_t emit_timestamp;         ///< time when emit (or receive) was called
  IOSpec::IOType io_type;
  std::shared_ptr<MetadataDictionary> metadata{};

  std::variant<std::any, std::shared_ptr<holoscan::Tensor>, holoscan::TensorMap> data;

  // Default constructor
  DataEntry()
      : type(Generic),
        acquisition_timestamp(-1),
        emit_timestamp(-1),
        io_type(IOSpec::IOType::kOutput),
        data(std::any{}) {}

  // Constructors for different types
  DataEntry(std::any data_arg, const std::string& id, int64_t acq_time, int64_t emit_time,
            IOSpec::IOType io_type, std::shared_ptr<MetadataDictionary> meta = nullptr);

  DataEntry(std::shared_ptr<holoscan::Tensor> tensor, const std::string& id, int64_t acq_time,
            int64_t emit_time, IOSpec::IOType io_type,
            std::shared_ptr<MetadataDictionary> meta = nullptr);

  DataEntry(holoscan::TensorMap tensor_map, const std::string& id, int64_t acq_time,
            int64_t emit_time, IOSpec::IOType io_type,
            std::shared_ptr<MetadataDictionary> meta = nullptr);
};

/**
 * @brief Backend interface for dual-queue async logger
 */
class AsyncDataLoggerBackend {
 public:
  AsyncDataLoggerBackend() = default;
  virtual ~AsyncDataLoggerBackend() = default;

  virtual bool initialize() = 0;
  virtual void shutdown() = 0;

  // Separate processing methods for different data types
  virtual bool process_data_entry(const DataEntry& entry) = 0;
  virtual bool process_large_data_entry(const DataEntry& entry) = 0;

  virtual std::string get_statistics() const { return ""; }
};

/**
 * @brief Asynchronous data logger
 *
 * Maintains a queue of items to be logged that are processed by a background thread.
 *
 * The `log_data` method is used to send data entries to the primary data queue and is
 * intended to be used to log most data types (e.g. strings, numeric types,
 * small structs, etc.).
 *
 * This logger can be operated in single queue or dual queue modes.
 *
 * When the `enable_large_data_queue` parameter is true, a separate queue will be available
 * for "large" data (e.g. Tensor and TensorMap data). This large data queue is processed by
 * a separate worker thread. If `log_tensor_data_contents` is true, it is expected that
 * `AsyncDataLoggerBackend::process_large_entry` would handle logging the actual tensor contents.
 * The `AsyncDataLoggerBackend::process_entry` method corresponding to the primary queue would
 * typically be designed to log only generic tensor attributes such as shape and dtype.
 *
 * The dual queue design allows for prioritized processing and selective dropping of large data
 * contents while preserving important metadata if the large data queue becomes full. It is the
 * responsibility of the backend (`AsyncDataLoggerBackend`) to determine which data types to log
 * to which queue.
 *
 * When `enable_large_data_queue` is false, "large" data is sent to the primary queue instead.
 *
 * Both queues should handle logging the `MetadataDictionary` when Holoscan's metadata feature is
 * enabled.
 *
 */
class AsyncDataLoggerResource : public DataLoggerResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(AsyncDataLoggerResource, DataLoggerResource)
  AsyncDataLoggerResource() = default;
  ~AsyncDataLoggerResource() override;

  // holds non-joinable std::threads, so prevent copying or moving the resource
  AsyncDataLoggerResource(const AsyncDataLoggerResource&) = delete;
  AsyncDataLoggerResource& operator=(const AsyncDataLoggerResource&) = delete;
  AsyncDataLoggerResource(AsyncDataLoggerResource&&) = delete;
  AsyncDataLoggerResource& operator=(AsyncDataLoggerResource&&) = delete;

  void setup(ComponentSpec& spec) override;
  void initialize() override;

  // DataLogger interface implementation
  bool log_data(const std::any& data, const std::string& unique_id,
                int64_t acquisition_timestamp = -1,
                const std::shared_ptr<MetadataDictionary>& metadata = nullptr,
                IOSpec::IOType io_type = IOSpec::IOType::kOutput) override;

  bool log_tensor_data(const std::shared_ptr<Tensor>& tensor, const std::string& unique_id,
                       int64_t acquisition_timestamp = -1,
                       const std::shared_ptr<MetadataDictionary>& metadata = nullptr,
                       IOSpec::IOType io_type = IOSpec::IOType::kOutput) override;

  bool log_tensormap_data(const TensorMap& tensor_map, const std::string& unique_id,
                          int64_t acquisition_timestamp = -1,
                          const std::shared_ptr<MetadataDictionary>& metadata = nullptr,
                          IOSpec::IOType io_type = IOSpec::IOType::kOutput) override;

  bool log_backend_specific(
      [[maybe_unused]] const std::any& data, [[maybe_unused]] const std::string& unique_id,
      [[maybe_unused]] int64_t acquisition_timestamp = -1,
      [[maybe_unused]] const std::shared_ptr<MetadataDictionary>& metadata = nullptr,
      [[maybe_unused]] IOSpec::IOType io_type = IOSpec::IOType::kOutput) override {
    // Default implementation: backend-specific logging is not supported
    return false;
  }

  void shutdown() override;

  void set_backend(std::shared_ptr<AsyncDataLoggerBackend> backend);

  // Statistics methods
  std::string get_statistics() const;
  size_t get_data_dropped_count() const { return data_dropped_.load(); }
  size_t get_large_data_dropped_count() const { return large_data_dropped_.load(); }
  size_t get_data_queue_size() const;
  size_t get_large_data_queue_size() const;

 protected:
  bool start_worker_threads();
  void stop_worker_threads();
  void data_worker_function();
  void large_data_worker_function();

  bool enqueue_data_entry(DataEntry&& entry);
  bool enqueue_large_data_entry(DataEntry&& entry);

  /**
   * @brief Helper function to extract typed values from component arguments
   *
   * This function handles both direct values and YAML node values, providing
   * robust parameter extraction with fallback to default values.
   *
   * @note For a concrete example, see how this function is used in `AsyncConsoleLogger`.
   *
   * @tparam ArgT The type of the argument to extract
   * @param arg_name The name of the argument to look for
   * @param default_value The default value to return if the argument is not found or cannot be
   * parsed
   * @return The extracted value or the default value
   */
  template <typename ArgT>
  [[nodiscard]] ArgT copy_value_from_args(const std::string& arg_name, ArgT default_value) {
    auto arg_it = std::find_if(args().begin(), args().end(), [&arg_name](const auto& arg) {
      return (arg.name() == arg_name);
    });

    if (arg_it == args().end()) {
      return default_value;  // Argument not found
    }

    if (!arg_it->has_value()) {
      return default_value;  // Argument has no value
    }

    std::any& any_arg = arg_it->value();
    ArgT result = default_value;

    if (arg_it->arg_type().element_type() == ArgElementType::kYAMLNode) {
      // Handle YAML node
      try {
        auto& arg_value = std::any_cast<YAML::Node&>(any_arg);
        bool parse_ok = YAML::convert<ArgT>::decode(arg_value, result);
        if (!parse_ok) {
          HOLOSCAN_LOG_ERROR("Could not parse YAML parameter '{}' as requested type", arg_name);
          return default_value;
        }
      } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("Exception parsing YAML parameter '{}': {}", arg_name, e.what());
        return default_value;
      }
    } else {
      // Handle direct value
      try {
        result = std::any_cast<ArgT>(any_arg);
      } catch (const std::bad_any_cast& e) {
        HOLOSCAN_LOG_ERROR(
            "Could not cast parameter '{}' to requested type: {}", arg_name, e.what());
        return default_value;
      }
    }

    return result;
  }

 private:
  // Queue configuration parameters
  Parameter<size_t> max_queue_size_;                     // Default: 50,000
  Parameter<int64_t> worker_sleep_time_;                 // Default: 50000ns (50μs)
  Parameter<AsyncQueuePolicy> queue_policy_;             // Default: kReject
  Parameter<size_t> large_data_max_queue_size_;          // Default: 1,000
  Parameter<int64_t> large_data_worker_sleep_time_;      // Default: 50000ns (50μs)
  Parameter<AsyncQueuePolicy> large_data_queue_policy_;  // Default: kReject
  Parameter<bool>
      enable_large_data_queue_;  // Default: true (enable separate queue for large data processing)

  // Lock-free queues
  std::unique_ptr<moodycamel::ConcurrentQueue<DataEntry>> data_queue_;
  std::unique_ptr<moodycamel::ConcurrentQueue<DataEntry>> large_data_queue_;

  // Worker threads
  std::thread data_worker_;
  std::thread large_data_worker_;
  std::atomic<bool> shutdown_requested_{false};
  std::atomic<bool> workers_running_{false};

  // Track backend shutdown to prevent multiple shutdown calls
  std::atomic<bool> backend_shutdown_called_{false};

  // Statistics (separate for each queue)
  std::atomic<size_t> data_dropped_{0};
  std::atomic<size_t> data_processed_{0};
  std::atomic<size_t> data_enqueued_{0};
  std::atomic<size_t> large_data_dropped_{0};
  std::atomic<size_t> large_data_processed_{0};
  std::atomic<size_t> large_data_enqueued_{0};

  // Backend
  std::shared_ptr<AsyncDataLoggerBackend> backend_;
  std::atomic<bool> backend_initialized_{false};  // Atomic flag for backend initialization status
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_ASYNC_DATA_LOGGER_HPP */
