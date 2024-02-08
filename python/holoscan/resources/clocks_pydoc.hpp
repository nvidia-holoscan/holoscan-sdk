/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_RESOURCES_CLOCKS_PYDOC_HPP
#define PYHOLOSCAN_RESOURCES_CLOCKS_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace Clock {

PYDOC(Clock, R"doc(
Base clock class.
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
Set the GXF scheduler to sleep for a specified duration.

Parameters
----------
duration_ns : int
    The duration to sleep (in nanoseconds).
)doc")

PYDOC(sleep_until, R"doc(
Set the GXF scheduler to sleep until a specified timestamp.

Parameters
----------
target_time_ns : int
    The target timestamp (in nanoseconds).
)doc")

}  // namespace Clock

namespace RealtimeClock {

PYDOC(RealtimeClock, R"doc(
Real-time clock class.
)doc")

// Constructor
PYDOC(RealtimeClock_python, R"doc(
Realtime clock.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
initial_timestamp : float, optional
    The initial time offset used until time scale is changed manually.
initial_time_scale : float, optional
    The initial time scale used until time scale is changed manually.
use_time_since_epoch : bool, optional
    If ``True``, clock time is time since epoch + `initial_time_offset` at ``initialize()``.
    Otherwise clock time is `initial_time_offset` at ``initialize()``.
name : str, optional
    The name of the clock.
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the resource.

Returns
-------
str
    The GXF type name of the resource
)doc")

PYDOC(setup, R"doc(
Define the component specification.

Parameters
----------
spec : holoscan.core.ComponentSpec
    Component specification associated with the resource.
)doc")

PYDOC(set_time_scale, R"doc(
Adjust the time scaling used by the clock.

Parameters
----------
time_scale : float, optional
    Durations (e.g. for periodic condition or sleep_for) are reduced by this scale value. A scale
    of 1.0 represents real-time while a scale of 2.0 would represent a clock where time elapses
    twice as fast.
)doc")

}  // namespace RealtimeClock

namespace ManualClock {

PYDOC(ManualClock, R"doc(
Manual clock class.
)doc")

// Constructor
PYDOC(ManualClock_python, R"doc(
Manual clock.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment to assign the resource to.
initial_timestamp : int, optional
    The initial timestamp on the clock (in nanoseconds).
name : str, optional
    The name of the clock.
)doc")
PYDOC(gxf_typename, R"doc(
The GXF type name of the resource.

Returns
-------
str
    The GXF type name of the resource
)doc")

PYDOC(setup, R"doc(
Define the component specification.

Parameters
----------
spec : holoscan.core.ComponentSpec
    Component specification associated with the resource.
)doc")

}  // namespace ManualClock

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_RESOURCES_CLOCKS_PYDOC_HPP
