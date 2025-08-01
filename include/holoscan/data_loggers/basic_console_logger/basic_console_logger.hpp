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

#ifndef HOLOSCAN_DATA_LOGGERS_BASIC_CONSOLE_LOGGER_BASIC_CONSOLE_LOGGER_HPP
#define HOLOSCAN_DATA_LOGGERS_BASIC_CONSOLE_LOGGER_BASIC_CONSOLE_LOGGER_HPP

#include <any>
#include <cstdint>
#include <memory>  // For std::shared_ptr in parameters
#include <string>
#include <vector>

#include "./simple_text_serializer.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/metadata.hpp"
#include "holoscan/core/resource.hpp"
#include "holoscan/core/resources/data_logger.hpp"

namespace holoscan {
namespace data_loggers {

/**
 * @brief Class for logging information to the console at runtime.
 *
 * Information on message contents are logging on emit and/or receive.
 *
 * ==Parameters ==
 *
 * - **serializer** : std::shared_ptr<SimpleTextSerializer>
 *   - Text serialization resource (optional). A SimpleTextSerializer initialized with default
 *     parameters will be automatically added if none is provided.
 * - **log_inputs**: bool (optional, default: true)
 *   - Globally enable or disable logging on input ports (`InputContext::receive` calls)
 * - **log_outputs**: bool (optional, default: true)
 *   - Globally enable or disable logging on output ports (`OutputContext::emit` calls)
 * - **log_metadata**: bool (optional, default: true)
 *   - Globally enable or disable logging of MetadataDictionary contents
 * - **log_tensor_data_contents**: bool (optional, default: false)
 *   - Enable logging of the contents of tensors, not just basic description information. The
 *     `max_elements` parameter of the `SimpleTextSerialzier` provided to `serializer` can be
 *     used to control how many elements are logged.
 * - **allowlist_patterns**: std::vector<std::string> (optional, default: empty vector)
 * - **denylist_patterns**: std::vector<std::string> (optional, default: empty vector)
 *
 * Note on allowlist/denylist pattern matching:
 *
 * If `allowlist_patterns` or `denylist_patterns` are specified, they are applied to the
 * `unique_id` assigned to messages by the underlying framework.
 *
 * In a non-distributed application (without a fragment name), the unique_id for a message will
 * have one of the following forms:
 *
 * - operator_name.port_name
 * - operator_name.port_name:index  (for multi-receivers with N:1 connection)
 *
 * For distributed applications, the fragment name will also appear in the unique id:
 *
 * - fragment_name.operator_name.port_name
 * - fragment_name.operator_name.port_name:index  (for multi-receivers with N:1 connection)
 *
 * The pattern matching logic is as follows:
 *
 *   - If `denylist patterns` is not empty and there is a match, do not log it.
 *   - Next check if `allowlist_patterns` is empty:
 *
 *       - If yes, return true (allow everything)
 *       - If no, return true only if there is a match to at least one of the specified patterns.
 */
class BasicConsoleLogger : public DataLoggerResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(BasicConsoleLogger, DataLoggerResource)
  BasicConsoleLogger() = default;

  void setup(ComponentSpec& spec) override;
  void initialize() override;
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
  bool log_backend_specific(const std::any& data, const std::string& unique_id,
                            int64_t acquisition_timestamp = -1,
                            const std::shared_ptr<MetadataDictionary>& metadata = nullptr,
                            IOSpec::IOType io_type = IOSpec::IOType::kOutput) override;

 private:
  Parameter<std::shared_ptr<SimpleTextSerializer>> serializer_;
};

}  // namespace data_loggers
}  // namespace holoscan

#endif /* HOLOSCAN_DATA_LOGGERS_BASIC_CONSOLE_LOGGER_BASIC_CONSOLE_LOGGER_HPP */
