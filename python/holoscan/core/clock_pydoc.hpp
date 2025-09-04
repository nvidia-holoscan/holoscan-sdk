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

#ifndef PYHOLOSCAN_CORE_CLOCK_PYDOC_HPP
#define PYHOLOSCAN_CORE_CLOCK_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace Clock {

PYDOC(Clock_args_kwargs, R"doc(
Create a Clock resource from a clock implementation.

Parameters
----------
clock_impl : ClockInterface
    The clock implementation to wrap (e.g., ``holoscan.resources.RealtimeClock``,
    ``holoscan.resources.ManualClock`` or ``holoscan.resources.SimulationClock``).
)doc")

PYDOC(time, R"doc(
The current time of the clock (in seconds).

Parameters
----------
time : double
    The current time of the clock (in seconds).
)doc")

PYDOC(timestamp, R"doc(
The current timestamp of the clock (in nanoseconds).

Parameters
----------
timestamp : int
    The current timestamp of the clock (in nanoseconds).
)doc")

PYDOC(sleep_for, R"doc(
Set the scheduler to sleep for a specified duration.

Parameters
----------
duration_ns : int
    The duration to sleep (in nanoseconds).
)doc")

PYDOC(sleep_until, R"doc(
Set the scheduler to sleep until a specified timestamp.

Parameters
----------
target_time_ns : int
    The target timestamp (in nanoseconds).
)doc")

PYDOC(set_clock_impl, R"doc(
Set the clock implementation.

Parameters
----------
clock_impl : ClockInterface
    The new clock implementation to use.
)doc")

PYDOC(clock_impl, R"doc(
Get the underlying clock implementation.

Returns
-------
ClockInterface
    The clock implementation (e.g., ``holoscan.resources.RealtimeClock``,
    ``holoscan.resources.ManualClock`` or ``holoscan.resources.SimulationClock``).
)doc")

PYDOC(cast_to, R"doc(
Cast the clock implementation to a specific type.

This method provides type-safe access to clock-specific functionality.
For example, only RealtimeClock supports set_time_scale().

Parameters
----------
type : type
    The clock implementation (e.g., ``holoscan.resources.RealtimeClock``,
    ``holoscan.resources.ManualClock`` or ``holoscan.resources.SimulationClock``).

Returns
-------
ClockInterface
    The clock cast to the requested type.

Raises
------
RuntimeError
    If the clock cannot be cast to the requested type, or if the type
    argument is not a valid clock class.

Examples
--------
>>> from holoscan.resources import RealtimeClock
>>> clock = scheduler.clock
>>> realtime_clock = clock.cast_to(RealtimeClock)  # Raises if not RealtimeClock
>>> realtime_clock.set_time_scale(2.0)
)doc")

}  // namespace Clock

}  // namespace holoscan::doc

#endif /* PYHOLOSCAN_CORE_CLOCK_PYDOC_HPP */
