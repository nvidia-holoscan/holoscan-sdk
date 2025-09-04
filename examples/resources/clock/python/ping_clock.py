"""
SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""  # noqa: E501

import datetime

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import PingTxOp
from holoscan.resources import RealtimeClock


class TimedPingRxOp(Operator):
    """Simple receiver operator illustrating a scheduler's clock resource.

    This operator has a single input port:
        input: "in"

    This is an example of a native operator with one input port.
    On each tick, it receives an integer from the "in" port and then
    uses the scheduler's clock to retrieve timestamps and wait for
    specified periods.
    """

    def __init__(self, fragment, *args, **kwargs):
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        value = op_input.receive("in")
        print(f"Rx message value: {value}")

        # we can retrieve the scheduler used for this application via it's fragment
        scheduler = self.fragment.scheduler()

        # The scheduler's clock is available as a parameter.
        # The clock object has methods to retrieve timestamps and to sleep
        # the thread for a specified duration as demonstrated below.
        clock = scheduler.clock
        print(f"\treceive time (s) = {clock.time()}")
        print(f"\treceive timestamp (ns) = {clock.timestamp()}")

        # cast the clock to a RealtimeClock to call the type-specific set_time_scale method
        realtime_clock = clock.cast_to(RealtimeClock)

        print("\tnow pausing for 0.1 s...")
        realtime_clock.sleep_for(100_000_000)
        ts = realtime_clock.timestamp()
        print(f"\ttimestamp after pause = {ts}")

        print("\tnow pausing until a target time 0.25 s in the future")
        target_ts = ts + 250_000_000
        realtime_clock.sleep_until(target_ts)
        print(f"\ttimestamp = {realtime_clock.timestamp()}")

        print("\tnow pausing for 0.125 s via a datetime.timedelta object")
        realtime_clock.sleep_for(datetime.timedelta(seconds=0.125))
        print(f"\ttimestamp = {realtime_clock.timestamp()}")

        print("\tnow adjusting time scale to 4.0 (time runs 4x faster)")
        realtime_clock.set_time_scale(4.0)

        print(
            "\tnow pausing 2.0 s via std::chrono::duration, but real pause will be 0.5 s "
            "due to the adjusted time scale."
        )
        realtime_clock.sleep_for(datetime.timedelta(seconds=2.0))
        print(
            f"\tfinal timestamp = {clock.timestamp()} (2.0 s increase will be shown despite "
            "scale of 4.0)"
        )

        print("\tnow resetting the time scale back to 1.0")
        realtime_clock.set_time_scale(1.0)


class MyPingApp(Application):
    def compose(self):
        # Define the tx and rx operators, allowing tx to execute 3 times
        tx = PingTxOp(self, CountCondition(self, 3), name="tx")
        rx = TimedPingRxOp(self, name="rx")

        # Define the workflow:  tx -> rx
        self.add_flow(tx, rx)


def main():
    app = MyPingApp()
    app.run()


if __name__ == "__main__":
    main()
