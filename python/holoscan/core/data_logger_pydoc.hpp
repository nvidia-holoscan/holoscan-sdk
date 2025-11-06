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

PYDOC(log_data, R"doc(
Log generic data.

Parameters
----------
data : object
    The data to log.
unique_id : str
    A unique identifier for the message.
acquisition_timestamp : int, optional
    Timestamp when the data was acquired (-1 if unknown).
metadata : MetadataDictionary, optional
    Associated metadata dictionary for the message.
io_type : holoscan.core.IOType, optional
    The type of I/O port (IOType.INPUT or IOType.OUTPUT).
stream_ptr : int, optional
    Memory address of the CUDA stream for GPU operations.

Returns
-------
bool
    True if logging was successful, False otherwise.
)doc")

PYDOC(log_tensor_data, R"doc(
Log tensor data.

Parameters
----------
tensor : holoscan.core.Tensor
    The Tensor to log.
unique_id : str
    A unique identifier for the message.
acquisition_timestamp : int, optional
    Timestamp when the data was acquired (-1 if unknown).
metadata : MetadataDictionary, optional
    Associated metadata dictionary for the message.
io_type : holoscan.core.IOType, optional
    The type of I/O port (IOType.INPUT or IOType.OUTPUT).
stream_ptr : int, optional
    Memory address of the CUDA stream for GPU operations.

Returns
-------
bool
    True if logging was successful, False otherwise.
)doc")

PYDOC(log_tensormap_data, R"doc(
Log tensor map data.

Parameters
----------
tensor_map : holoscan.core.TensorMap
    The TensorMap to log.
unique_id : str
    A unique identifier for the message.
acquisition_timestamp : int, optional
    Timestamp when the data was acquired (-1 if unknown).
metadata : MetadataDictionary, optional
    Associated metadata dictionary for the message.
io_type : holoscan.core.IOType, optional
    The type of I/O port (IOType.INPUT or IOType.OUTPUT).
stream_ptr : int, optional
    Memory address of the CUDA stream for GPU operations.

Returns
-------
bool
    True if logging was successful, False otherwise.
)doc")

PYDOC(log_backend_specific, R"doc(
Log backend-specific data types.

This method is called for logging backend-specific data types. The default
implementation returns False indicating backend-specific logging is not supported.

Parameters
----------
data : object
    The backend-specific data to log.
unique_id : str
    A unique identifier for the message.
acquisition_timestamp : int, optional
    Timestamp when the data was acquired (-1 if unknown).
metadata : MetadataDictionary, optional
    Associated metadata dictionary for the message.
io_type : holoscan.core.IOType, optional
    The type of I/O port (IOType.INPUT or IOType.OUTPUT).
stream_ptr : int, optional
    Memory address of the CUDA stream for GPU operations.

Returns
-------
bool
    True if logging was successful, False if backend-specific logging is not supported.
)doc")

PYDOC(should_log_output, R"doc(
Check if the logger should log output ports.

Returns
-------
bool
    True if the logger should log output ports, False otherwise.
)doc")

PYDOC(should_log_input, R"doc(
Check if the logger should log input ports.

Returns
-------
bool
    True if the logger should log input ports, False otherwise.
)doc")

}  // namespace DataLogger

namespace DataLoggerResource {

PYDOC(DataLoggerResource, R"doc(
Base data logger resource class.

Inherits from both Resource and DataLogger to provide Parameter handling and a default
set of parameters likely to be useful across many concrete data logger implementations.

This simple class is intended for logging cases where the time needed to perform logging
is minimal because the logging methods would run on the same thread that is calling
`operator.compute`. Thus, any time spent in logging adds to the overall execution time of
the operator. For situations where this is not acceptable, `AsyncDataLoggerResource` is
provided as an alternative.
)doc")

PYDOC(get_timestamp, R"doc(
Get current timestamp via the logger's configured clock.

Returns
----------
timestamp : int
    When `use_scheduler_clock` is ``True``, this is the timestamp returned by the scheduler's clock.
    Otherwise it is the system clock time in nanoseconds relative to epoch).
)doc")
}  // namespace DataLoggerResource

namespace AsyncDataLoggerResource {

PYDOC(AsyncDataLoggerResource, R"doc(
Base asynchronous data logger resource class.

This is a version of DataLoggerResource where it is intended that messages are pushed to the
provided queue. Logging of messages from the queue is handled by a dedicated background thread
that is managed by the AsyncDataLoggerResource.
)doc")

}  // namespace AsyncDataLoggerResource

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_CORE_DATA_LOGGER_PYDOC_HPP */
