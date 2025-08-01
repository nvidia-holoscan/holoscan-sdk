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
