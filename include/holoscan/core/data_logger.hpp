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

#ifndef HOLOSCAN_CORE_DATA_LOGGER_HPP
#define HOLOSCAN_CORE_DATA_LOGGER_HPP

#include <any>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "./io_spec.hpp"

// Forward declaration to avoid including <cuda_runtime.h>
extern "C" {
typedef struct CUstream_st* cudaStream_t;
}

namespace holoscan {

class MetadataDictionary;  // forward declaration
class Tensor;              // forward declaration
class TensorMap;           // forward declaration

/**
 * @brief Pure virtual interface for data loggers.
 *
 * This interface defines the contract that all data loggers must implement
 * for logging various types of data including generic std::any data, Tensors,
 * and TensorMaps.
 */
class DataLogger {
 public:
  DataLogger() = default;
  virtual ~DataLogger() = default;

  /**
   * @brief Logs a message.
   *
   * The unique_id for the message will have the form:
   * - operator_name.port_name
   * - operator_name.port_name:index   (for multi-receivers with N:1 connection)
   *
   * For distributed applications, the fragment name will also appear in the unique id:
   * - fragment_name.operator_name.port_name
   * - fragment_name.operator_name.port_name:index
   *
   * @param data The data to log, passed as std::any.
   * @param unique_id A unique identifier for the message.
   * @param acquisition_timestamp Timestamp when the data was acquired (-1 if unknown).
   * @param metadata Associated metadata dictionary for the message.
   * @param io_type The type of I/O port (kInput or kOutput).
   * @param stream Optional CUDA stream for GPU operations.
   * @return true if logging (including serialization and sending) was successful, false otherwise.
   */
  virtual bool log_data(const std::any& data, const std::string& unique_id,
                        int64_t acquisition_timestamp = -1,
                        const std::shared_ptr<MetadataDictionary>& metadata = nullptr,
                        IOSpec::IOType io_type = IOSpec::IOType::kOutput,
                        std::optional<cudaStream_t> stream = std::nullopt) = 0;

  /**
   * @brief Logs a Tensor with optional data content logging.
   *
   * This specialized method allows efficient logging of tensor metadata without the overhead
   * of logging large tensor data arrays when only header information is needed.
   *
   * The unique_id for the message will have the form:
   * - operator_name.port_name
   * - operator_name.port_name:index   (for multi-receivers with N:1 connection)
   *
   * For distributed applications, the fragment name will also appear in the unique id:
   * - fragment_name.operator_name.port_name
   * - fragment_name.operator_name.port_name:index
   *
   * @param tensor The Tensor to log.
   * @param unique_id A unique identifier for the message.
   * @param acquisition_timestamp Timestamp when the data was acquired (-1 if unknown).
   * @param metadata Associated metadata dictionary for the message.
   * @param io_type The type of I/O port (kInput or kOutput).
   * @param stream Optional CUDA stream for GPU operations.
   * @return true if logging was successful, false otherwise.
   */
  virtual bool log_tensor_data(const std::shared_ptr<Tensor>& tensor, const std::string& unique_id,
                               int64_t acquisition_timestamp = -1,
                               const std::shared_ptr<MetadataDictionary>& metadata = nullptr,
                               IOSpec::IOType io_type = IOSpec::IOType::kOutput,
                               std::optional<cudaStream_t> stream = std::nullopt) = 0;

  /**
   * @brief Logs a TensorMap with optional data content logging.
   *
   * This specialized method allows efficient logging of tensor map metadata without the overhead
   * of logging large tensor data arrays when only header information is needed.
   *
   * The unique_id for the message will have the form:
   * - operator_name.port_name
   * - operator_name.port_name:index   (for multi-receivers with N:1 connection)
   *
   * For distributed applications, the fragment name will also appear in the unique id:
   * - fragment_name.operator_name.port_name
   * - fragment_name.operator_name.port_name:index
   *
   * @param tensor_map The TensorMap to log.
   * @param unique_id A unique identifier for the message.
   * @param acquisition_timestamp Timestamp when the data was acquired (-1 if unknown).
   * @param metadata Associated metadata dictionary for the message.
   * @param io_type The type of I/O port (kInput or kOutput).
   * @param stream Optional CUDA stream for GPU operations.
   * @return true if logging was successful, false otherwise.
   */
  virtual bool log_tensormap_data(const TensorMap& tensor_map, const std::string& unique_id,
                                  int64_t acquisition_timestamp = -1,
                                  const std::shared_ptr<MetadataDictionary>& metadata = nullptr,
                                  IOSpec::IOType io_type = IOSpec::IOType::kOutput,
                                  std::optional<cudaStream_t> stream = std::nullopt) = 0;

  /**
   * @brief Logs backend-specific data types.
   *
   * This method is called for logging backend-specific data types (intended for use with backends
   * that have separate emit/receive codepaths for backend-specific types). The data parameter is
   * kept as std::any here to avoid making the base interface specific to a particular backend, but
   * a backend-specific concrete implementation should be provided as needed via run-time type
   * checking.
   *
   * A concrete example of a backend-specific type is the GXF Entity type which is a
   * heterogeneous collection of components. An implementation of this method for GXF entities is
   * provided in the concrete implementation of the GXFConsoleLogger.
   *
   * The unique_id for the message will have the form:
   * - operator_name.port_name
   * - operator_name.port_name:index   (for multi-receivers with N:1 connection)
   *
   * For distributed applications, the fragment name will also appear in the unique id:
   * - fragment_name.operator_name.port_name
   * - fragment_name.operator_name.port_name:index
   *
   * @param data The backend-specific data to log, passed as std::any.
   * @param unique_id A unique identifier for the message.
   * @param acquisition_timestamp Timestamp when the data was acquired (-1 if unknown).
   * @param metadata Associated metadata dictionary for the message.
   * @param io_type The type of I/O port (kInput or kOutput).
   * @param stream Optional CUDA stream for GPU operations.
   * @return true if logging was successful, false if backend-specific logging is not supported.
   */
  virtual bool log_backend_specific(
      [[maybe_unused]] const std::any& data, [[maybe_unused]] const std::string& unique_id,
      [[maybe_unused]] int64_t acquisition_timestamp = -1,
      [[maybe_unused]] const std::shared_ptr<MetadataDictionary>& metadata = nullptr,
      [[maybe_unused]] IOSpec::IOType io_type = IOSpec::IOType::kOutput,
      [[maybe_unused]] std::optional<cudaStream_t> stream = std::nullopt) {
    // Default implementation: backend-specific logging is not supported
    return false;
  }

  /**
   * @brief Checks if the logger should log output ports.
   *
   * If False, the data logger will not be applied during op_output.emit() calls from
   * Operator::compute.
   *
   * @return true if the logger should log output ports, false otherwise.
   */
  virtual bool should_log_output() const = 0;

  /**
   * @brief Checks if the logger should log input ports.
   *
   * If False, the data logger will not be applied during op_input.receive() calls from
   * Operator::compute.
   *
   * @return true if the logger should log input ports, false otherwise.
   */
  virtual bool should_log_input() const = 0;

  /**
   * @brief Shutdown the data logger.
   *
   * This method should be called to properly shutdown the data logger, including stopping
   * any background threads and releasing resources. The default implementation does nothing.
   * Data loggers that use background threads or other resources should override this method
   * to perform proper cleanup.
   */
  virtual void shutdown() {
    // Default implementation: no shutdown needed
  }
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_DATA_LOGGER_HPP */
