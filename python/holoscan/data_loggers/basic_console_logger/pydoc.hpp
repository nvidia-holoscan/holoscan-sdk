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

#ifndef PYHOLOSCAN_DATA_LOGGERS_BASIC_CONSOLE_LOGGER_PYDOC_HPP
#define PYHOLOSCAN_DATA_LOGGERS_BASIC_CONSOLE_LOGGER_PYDOC_HPP

#include <string>

#include "../../macros.hpp"

namespace holoscan::doc {

namespace BasicConsoleLogger {

PYDOC(BasicConsoleLogger, R"doc(
Basic console logger for debugging and development purposes.

This data logger outputs structured log messages to the console (via the Holoscan logging system)
for any data received. It can handle tensors, tensor maps, metadata, and general data types by
converting them to human-readable text format.

The logger provides filtering capabilities to control which messages are logged.

Parameters
----------
fragment : holoscan.core.Fragment (constructor only)
    The fragment that the data logger belongs to.
serializer : holoscan.data_loggers.SimpleTextSerializer, optional
    Text serializer used to convert data to string format. If not provided, a default
    SimpleTextSerializer will be created automatically.
log_inputs : bool, optional
    Whether to log input messages. Default is True.
log_outputs : bool, optional
    Whether to log output messages. Default is True.
log_tensor_data_content : bool, optional
    Whether to log the actual content of tensor data. Default is True.
log_metadata : bool, optional
    Whether to log metadata associated with messages. Default is True.
use_scheduler_clock : bool, optional
    Whether to use the scheduler's clock for timestamps (if False, uses the C++
    `std::chrono::steady_clock` time relative to epoch). Ignored if `clock` is provided.
    Default is True.
clock : clock : holoscan.resources.GXFClock or holoscan.core.Clock, optional
    If a clock is explicitly specified, all emit and receive timestamps will be determined via
    the timestamp method of this clock. Can be a GXF-based clock like
    ``holoscan.resources.RealtimeClock``.
allowlist_patterns : list of str, optional
    List of regex patterns to apply to message unique IDs. If empty, all messages not matching a
    denylist pattern will be logged. Otherwise, there must be a match to one of the allowlist
    patterns. See notes below for more details.
denylist_patterns : list of str, optional
    List of regex patterns to apply to message unique IDs. If specified and there is a match at
    least one of these patterns, the message is not logged. See notes below for more details.
name : str, optional (constructor only)
    The name of the data logger. Default value is ``"basic_console_logger"``.

Notes
-----
If `allowlist_patterns` or `denylist_patterns` are specified, they are applied to the `unique_id`
assigned to messages by the underlying framework.

In a non-distributed application (without a fragment name), the unique_id for a message will have
one of the following forms:

  - operator_name.port_name
  - operator_name.port_name:index  (for multi-receivers with N:1 connection)

For distributed applications, the fragment name will also appear in the unique id:

  - fragment_name.operator_name.port_name
  - fragment_name.operator_name.port_name:index  (for multi-receivers with N:1 connection)

The pattern matching logic is as follows:

  - If `denylist patterns` is specified and there is a match, do not log it.
  - Next check if `allowlist_patterns` is empty:
    - If yes, return true (allow everything)
    - If no, return true only if there is a match to at least one of the specified patterns.
)doc")
}  // namespace BasicConsoleLogger

namespace GXFConsoleLogger {

PYDOC(GXFConsoleLogger, R"doc(
GXF-based console logger for debugging and development purposes.

This logger operates the same as BasicConsoleLogger but also logs on emit or receive of
`holoscan::gxf::Entity` and `nvidia::gxf::Entity` types. Currently only Tensors present within the
entity will be logged.

This data logger outputs structured log messages to the console (via the Holoscan logging system)
for any data received. It can handle tensors, tensor maps, metadata, and general data types by
converting them to human-readable text format.

The logger provides filtering capabilities to control which messages are logged.

Parameters
----------
fragment : holoscan.core.Fragment (constructor only)
    The fragment that the data logger belongs to.
serializer : holoscan.data_loggers.SimpleTextSerializer, optional
    Text serializer used to convert data to string format. If not provided, a default
    SimpleTextSerializer will be created automatically.
log_inputs : bool, optional
    Whether to log input messages. Default is True.
log_outputs : bool, optional
    Whether to log output messages. Default is True.
log_tensor_data_content : bool, optional
    Whether to log the actual content of tensor data. Default is True.
log_metadata : bool, optional
    Whether to log metadata associated with messages. Default is True.
use_scheduler_clock : bool, optional
    Whether to use the scheduler's clock for timestamps (if False, uses the C++
    `std::chrono::steady_clock` time relative to epoch). Ignored if `clock` is provided.
    Default is True.
clock : clock : holoscan.resources.GXFClock or holoscan.core.Clock, optional
    If a clock is explicitly specified, all emit and receive timestamps will be determined via
    the timestamp method of this clock. Can be a GXF-based clock like
    ``holoscan.resources.RealtimeClock``.
allowlist_patterns : list of str, optional
    List of regex patterns to apply to message unique IDs. If empty, all messages not matching a
    denylist pattern will be logged. Otherwise, there must be a match to one of the allowlist
    patterns. See notes below for more details.
denylist_patterns : list of str, optional
    List of regex patterns to apply to message unique IDs. If specified and there is a match at
    least one of these patterns, the message is not logged. See notes below for more details.
name : str, optional (constructor only)
    The name of the data logger. Default value is ``"basic_console_logger"``.
}  // namespace BasicConsoleLogger

Notes
-----
If `allowlist_patterns` or `denylist_patterns` are specified, they are applied to the `unique_id`
assigned to messages by the underlying framework.

In a non-distributed application (without a fragment name), the unique_id for a message will have
one of the following forms:

  - operator_name.port_name
  - operator_name.port_name:index  (for multi-receivers with N:1 connection)

For distributed applications, the fragment name will also appear in the unique id:

  - fragment_name.operator_name.port_name
  - fragment_name.operator_name.port_name:index  (for multi-receivers with N:1 connection)

The pattern matching logic is as follows:

  - If `denylist patterns` is specified and there is a match, do not log it.
  - Next check if `allowlist_patterns` is empty:
    - If yes, return true (allow everything)
    - If no, return true only if there is a match to at least one of the specified patterns.
)doc")
}  // namespace GXFConsoleLogger

namespace SimpleTextSerializer {

PYDOC(SimpleTextSerializer, R"doc(
Simple text serializer for converting various data types to human-readable strings.

This serializer can handle common data types including integers, floats, strings, vectors,
metadata dictionaries, tensors, and tensor maps. It provides configurable limits for
vector elements and metadata items to prevent excessively long output.


Parameters
----------
fragment : holoscan.core.Fragment (constructor only)
    The fragment that the serializer belongs to.
max_elements : int, optional
    Maximum number of vector elements to display before truncation. Default is 10.
max_metadata_items : int, optional
    Maximum number of metadata dictionary items to display before truncation. Default is 10.
log_python_object_contents : bool, optional
    Whether to log Python object contents. Default is True. Warning: logging the contents of Python
    objects via the ``repr`` method requires acquiring the GIL which can be slow.
name : str, optional (constructor only)
    The name of the serializer. Default value is ``"simple_text_serializer"``.
)doc")
}  // namespace SimpleTextSerializer

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_DATA_LOGGERS_BASIC_CONSOLE_LOGGER_PYDOC_HPP */
