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

#ifndef HOLOSCAN_CORE_RESOURCES_DATA_LOGGER_HPP
#define HOLOSCAN_CORE_RESOURCES_DATA_LOGGER_HPP

#include <any>
#include <cstdint>
#include <memory>  // For std::shared_ptr in parameters
#include <mutex>
#include <optional>
#include <regex>
#include <string>
#include <vector>

#include "../component_spec.hpp"
#include "../data_logger.hpp"
#include "../resource.hpp"

namespace holoscan {

class MetadataDictionary;  // forward declaration
class Tensor;              // forward declaration
class TensorMap;           // forward declaration

// Shared mutex for thread-safe console output coordination across all console logger types
extern std::mutex console_output_mutex;

class DataLoggerResource : public DataLogger, public Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(DataLoggerResource,
                                       Resource)  // Forward arguments to Resource
  DataLoggerResource() = default;
  ~DataLoggerResource() override = default;

  /**
   * @brief Defines parameters for the logging resource, including the serializer.
   *
   * @param spec The component specification.
   */
  void setup(ComponentSpec& spec) override;

  // initialize() is inherited from Resource and should be overridden by concrete loggers.

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
   * @return true if logging (including serialization and sending) was successful, false otherwise.
   */
  bool log_data(const std::any& data, const std::string& unique_id,
                int64_t acquisition_timestamp = -1,
                const std::shared_ptr<MetadataDictionary>& metadata = nullptr,
                IOSpec::IOType io_type = IOSpec::IOType::kOutput) override = 0;

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
   * @return true if logging was successful, false otherwise.
   */
  bool log_tensor_data(const std::shared_ptr<Tensor>& tensor, const std::string& unique_id,
                       int64_t acquisition_timestamp = -1,
                       const std::shared_ptr<MetadataDictionary>& metadata = nullptr,
                       IOSpec::IOType io_type = IOSpec::IOType::kOutput) override = 0;

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
   * @return true if logging was successful, false otherwise.
   */
  bool log_tensormap_data(const TensorMap& tensor_map, const std::string& unique_id,
                          int64_t acquisition_timestamp = -1,
                          const std::shared_ptr<MetadataDictionary>& metadata = nullptr,
                          IOSpec::IOType io_type = IOSpec::IOType::kOutput) override = 0;

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
   * @return true if logging was successful, false if backend-specific logging is not supported.
   */
  bool log_backend_specific(const std::any& data, [[maybe_unused]] const std::string& unique_id,
                            int64_t acquisition_timestamp = -1,
                            const std::shared_ptr<MetadataDictionary>& metadata = nullptr,
                            IOSpec::IOType io_type = IOSpec::IOType::kOutput) override;

  /**
   * @brief Checks if a message with the given unique_id should be logged based on
   * allowlist/denylist patterns.
   *
   * This utility function implements the filtering logic:
   * - First check if `denylist patterns` are specified and if there is a match, do not log it.
   * - Next check if `allowlist_patterns` were specified:
   *   - If no, return true (allow everything)
   *   - If yes, return true only if there is a match to the specified patterns.
   *
   * @param unique_id The unique identifier to check against patterns.
   * @return true if the message should be logged, false otherwise.
   */
  bool should_log_message(const std::string& unique_id) const;

  /**
   * @brief Checks if the logger should log output ports.
   *
   * If False, the data logger will not be applied during op_input.emit() calls from
   * Operator::compute.
   *
   * @return true if the logger should log output ports, false otherwise.
   */
  bool should_log_output() const override { return log_outputs_.get(); }

  /**
   * @brief Checks if the logger should log input ports.
   *
   * If False, the data logger will not be applied during op_input.receive() calls from
   * Operator::compute.
   *
   * @return true if the logger should log input ports, false otherwise.
   */
  bool should_log_input() const override { return log_inputs_.get(); }

  /**
   * @brief Checks if the logger should log metadata.
   *
   * If False, the data logger will not log metadata for each operator.
   *
   * @return true if the logger should log metadata, false otherwise.
   */
  bool should_log_metadata() const { return log_metadata_.get(); }

  /**
   * @brief Checks if the logger should log tensor data content.
   *
   * If False, only tensor header information will be logged, not the actual data arrays.
   * When true, the full tensor data is also logged.
   *
   * @return true if the logger should log tensor data content, false otherwise.
   */
  bool should_log_tensor_data_content() const { return log_tensor_data_content_.get(); }

  /**
   * @brief Get the current timestamp for logging operations.
   *
   * This method is called internally by the logging functions to obtain timestamps for
   * emit_timestamp (when io_type==IOSpec::IOType::kOutput) or receive_timestamp (when
   * io_type==IOSpec::IOType::kInput). The default implementation provides high-resolution
   * timestamps in microseconds since epoch. Implementations can override this to provide custom
   * timing mechanisms as appropriate.
   *
   * @return Current timestamp in microseconds since epoch, or -1 if not available.
   */
  virtual int64_t get_timestamp() const;

  void initialize() override;

 protected:
  // TODO(grelee): should we add a standard serializer interface here?
  // Parameter<std::shared_ptr<DataLoggerSerializer>> serializer_;
  Parameter<bool> log_outputs_;
  Parameter<bool> log_inputs_;
  Parameter<bool> log_tensor_data_content_;
  Parameter<bool> log_metadata_;

  // Filtering parameters
  Parameter<std::vector<std::string>> allowlist_patterns_;
  Parameter<std::vector<std::string>> denylist_patterns_;

 private:
  // Compiled regex patterns for efficient matching
  std::optional<std::regex> compiled_allowlist_pattern_;
  std::optional<std::regex> compiled_denylist_pattern_;
  bool patterns_compiled_ = false;

  /**
   * @brief Compiles string patterns into regex objects for efficient matching.
   */
  void compile_patterns();
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_RESOURCES_DATA_LOGGER_HPP */
