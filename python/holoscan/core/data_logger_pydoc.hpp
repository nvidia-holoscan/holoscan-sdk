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

#ifndef PYHOLOSCAN_CORE_DATA_LOGGER_PYDOC_HPP
#define PYHOLOSCAN_CORE_DATA_LOGGER_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace DataLogger {

PYDOC(DataLogger, R"doc(
Base class for data loggers.

A data logger captures and persists data flowing through the Holoscan pipeline.
It can log data from input ports, output ports, and metadata.
)doc")

}  // namespace DataLogger

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CORE_DATA_LOGGER_PYDOC_HPP
