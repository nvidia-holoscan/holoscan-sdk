/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_LOGGER_PYDOC_HPP
#define PYHOLOSCAN_LOGGER_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace Logger {

PYDOC(LogLevel, R"doc(
Enum class for the logging level.
)doc")

PYDOC(set_log_level, R"doc(
Set the global logging level.

Parameters
----------
level : holoscan.logger.LogLevel
    The logging level to set
)doc")

PYDOC(log_level, R"doc(
Get the global logging level.
)doc")

PYDOC(set_log_pattern, R"doc(
Set the format pattern for the logger.

Parameters
----------
pattern : str
    The pattern to use for logging messages. Uses the spdlog format specified at [1].
    The default pattern used by spdlog is "[%Y-%m-%d %H:%M:%S.%e] [%l] [%n] %v".

References
----------
.. [1] https://spdlog.docsforge.com/v1.x/3.custom-formatting/
)doc")

}  // namespace Logger

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_LOGGER_PYDOC_HPP
