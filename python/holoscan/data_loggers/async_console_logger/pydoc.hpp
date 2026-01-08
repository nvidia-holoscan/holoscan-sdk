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

#ifndef PYHOLOSCAN_DATA_LOGGERS_ASYNC_CONSOLE_LOGGER_PYDOC_HPP
#define PYHOLOSCAN_DATA_LOGGERS_ASYNC_CONSOLE_LOGGER_PYDOC_HPP

#include <string>

#include "../../macros.hpp"

namespace holoscan::doc {

namespace AsyncConsoleLogger {

PYDOC(AsyncConsoleLogger, R"doc(
Asynchronous console logger for debugging and development purposes.

This data logger outputs structured log messages to the console (via the Holoscan logging system)
for any data received. It can handle tensors, tensor maps, metadata, and general data types by
converting them to human-readable text format.

The logger provides filtering capabilities to control which messages are logged.

This logger operators in the same way as `BasicConsoleLogger` except that logging is handled by
a dedicated background thread managed by `AsyncDataLoggerResource`.

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
clock : holoscan.resources.GXFClock or holoscan.core.Clock, optional
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
max_queue_size : int, optional
    Maximum number of entries in the data queue. When `enable_large_data_queue` is ``True``,
    The data queue handles tensor headers without full tensor content. Otherwise
    tensor data content will also be in this queue. In both cases, whether tensor data content
    is logged at all is controlled by `log_tensor_data_content`.
worker_sleep_time : int, optional
    Sleep duration in nanoseconds when the data queue is empty. Lower values
    reduce latency but increase CPU usage. Default is 50000 (50μs).
queue_policy : AsyncQueuePolicy, optional
    Policy for handling queue overflow. Can be ``AsyncQueuePolicy.kReject`` (default)
    to reject new items with a warning, or ``AsyncQueuePolicy.kRaise`` to raise an
    exception when the queue is full.
large_data_max_queue_size : int, optional
    Maximum number of entries in the large data queue. The large data queue handles
    full tensor content for detailed logging. Default is 1000.
large_data_worker_sleep_time : int, optional
    Sleep duration in nanoseconds when the large data queue is empty. Default is
    50000 (50μs).
large_data_queue_policy : AsyncQueuePolicy, optional
    Policy for handling large data queue overflow. Can be ``AsyncQueuePolicy.kReject``
    (default) to reject new items with a warning, or ``AsyncQueuePolicy.kRaise`` to
    raise an exception when the queue is full.
enable_large_data_queue : bool, optional
    Whether to enable the large data queue and worker thread for processing full
    tensor content. Default is True.
shutdown_wait_period_ms : int, optional
    Time in milliseconds to wait for queues to drain during shutdown. Use -1 to wait
    indefinitely (default), 0 to not wait, or a positive value for a specific
    timeout in milliseconds. Default is -1.
name : str, optional (constructor only)
    The name of the data logger. Default value is ``"basic_console_logger"``.
}  // namespace AsyncConsoleLogger

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

}  // namespace AsyncConsoleLogger
}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_DATA_LOGGERS_ASYNC_CONSOLE_LOGGER_PYDOC_HPP */
