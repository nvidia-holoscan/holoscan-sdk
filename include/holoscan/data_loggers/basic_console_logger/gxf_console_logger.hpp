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

#ifndef HOLOSCAN_DATA_LOGGERS_BASIC_CONSOLE_LOGGER_GXF_BASIC_CONSOLE_LOGGER_HPP
#define HOLOSCAN_DATA_LOGGERS_BASIC_CONSOLE_LOGGER_GXF_BASIC_CONSOLE_LOGGER_HPP

#include <any>
#include <cstdint>
#include <memory>
#include <string>

#include "./basic_console_logger.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/metadata.hpp"

namespace holoscan {
namespace data_loggers {

/**
 * @brief GXF-specific extension of BasicConsoleLogger with support for GXF Entity logging.
 *
 * This logger extends BasicConsoleLogger to provide support for logging GXF-specific data types,
 * particularly nvidia::gxf::Entity and holoscan::gxf::Entity objects. It implements the
 * log_backend_specific method to handle these GXF entity types with appropriate runtime
 * type checking. Currently only `Tensor` components present within the entity will be logged.
 *
 * All the same parameters from BasicConsoleLogger are supported. See BasicConsoleLogger
 * documentation for parameter details.
 */
class GXFConsoleLogger : public BasicConsoleLogger {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(GXFConsoleLogger, BasicConsoleLogger)
  GXFConsoleLogger() = default;

  bool log_backend_specific(const std::any& data, const std::string& unique_id,
                            int64_t acquisition_timestamp = -1,
                            const std::shared_ptr<MetadataDictionary>& metadata = nullptr,
                            IOSpec::IOType io_type = IOSpec::IOType::kOutput) override;
};

}  // namespace data_loggers
}  // namespace holoscan

#endif /* HOLOSCAN_DATA_LOGGERS_BASIC_CONSOLE_LOGGER_GXF_BASIC_CONSOLE_LOGGER_HPP */
