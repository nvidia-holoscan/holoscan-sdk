/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_CONDITIONS_PERIODIC_PYDOC_HPP
#define PYHOLOSCAN_CONDITIONS_PERIODIC_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace PeriodicCondition {

PYDOC(PeriodicCondition, R"doc(
Condition class to support periodic execution of operators.
The recess (pause) period indicates the minimum amount of
time that must elapse before the `compute()` method can be executed again.
The recess period can be specified as an integer value in nanoseconds.

For example:  1000 for 1 microsecond 1000000 for 1 millisecond,
and 10000000000 for 1 second.

The recess (pause) period can also be specified as a `datetime.timedelta` object
representing a duration.
(see https://docs.python.org/3/library/datetime.html#timedelta-objects)

For example: datetime.timedelta(minutes=1), datetime.timedelta(seconds=1),
datetime.timedelta(milliseconds=1) and datetime.timedelta(microseconds=1).
Supported argument names are: weeks| days | hours | minutes | seconds | millisecons | microseconds
This requires `import datetime`.

Parameters
----------
fragment : holoscan.core.Fragment or holoscan.core.Subgraph
    The fragment (or subgraph) the condition will be associated with
recess_period : int or datetime.timedelta
    The recess (pause) period value used by the condition.
    If an integer is provided, the units are in nanoseconds.
policy : PeriodicConditionPolicy, optional
    The scheduling policy for handling periodic ticks. One of:

    - CATCH_UP_MISSED_TICKS (default): Attempts to catch up on missed ticks by executing
      multiple ticks in quick succession. With a 100ms period:
      * tick 0 at 0ms   -> next_target = 100ms
      * tick 1 at 250ms -> next_target = 200ms (executes as next_target < current_time)
      * tick 2 at 255ms -> next_target = 300ms (double tick before 300ms)
    - MIN_TIME_BETWEEN_TICKS: Guarantees the recess period will elapse between ticks.
      With a 100ms period:
      * tick 0 at 0ms   -> next_target = 100ms
      * tick 1 at 101ms -> next_target = 201ms
      * tick 2 at 350ms -> next_target = 450ms
    - NO_CATCH_UP_MISSED_TICKS: Skips missed ticks without catch-up. With a 100ms period:
      * tick 0 at 0ms   -> next_target = 100ms
      * tick 1 at 250ms -> next_target = 300ms (single tick, no catch-up)
      * tick 2 at 305ms -> next_target = 400ms

name : str, optional
    The name of the condition.
)doc")

PYDOC(recess_period, R"doc(
Sets the recess (pause) period associated with the condition.
The recess period can be specified as an integer value in nanoseconds or
a `datetime.timedelta` object representing a duration.
)doc")

PYDOC(recess_period_ns, R"doc(
Gets the recess (pause) period value in nanoseconds.
)doc")

PYDOC(last_run_timestamp, R"doc(
Gets the integer representing the last run time stamp.
)doc")

PYDOC(policy, R"doc(
Gets the PeriodicConditionPolicy being used by the PeriodicCondition.
)doc")

}  // namespace PeriodicCondition
}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CONDITIONS_PERIODIC_PYDOC_HPP
