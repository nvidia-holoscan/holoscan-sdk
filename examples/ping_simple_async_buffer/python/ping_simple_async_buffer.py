"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import time

from holoscan.conditions import CountCondition
from holoscan.core import Application, IOSpec, Tracker
from holoscan.operators import PingRxOp, PingTxOp
from holoscan.schedulers import EventBasedScheduler


class PingTxAsyncOp(PingTxOp):
    """Async transmit operator that inherits from PingTxOp.

    This operator adds a small delay to simulate async behavior,
    similar to the C++ version.
    """

    def compute(self, op_input, op_output, context):
        # Add a 10ms delay to simulate async behavior
        time.sleep(0.01)
        # Call the parent compute method
        super().compute(op_input, op_output, context)
        print("tx")


class PingRxAsyncOp(PingRxOp):
    """Async receive operator that inherits from PingRxOp.

    This operator adds a small delay and custom receive logic,
    similar to the C++ version.
    """

    def compute(self, op_input, op_output, context):
        # Add a 5ms delay to simulate async behavior
        time.sleep(0.005)

        # Custom receive logic similar to the C++ version
        try:
            value = op_input.receive("in")
            if value is None:
                print(f"Operator '{self.name}' did not receive a valid value.")
                return
            print(f"Rx message value: {value}")
        except Exception as e:
            print(f"Operator '{self.name}' failed to receive message: {e}")


class MyPingApp(Application):
    def compose(self):
        # Define the tx and rx operators, allowing the tx operator to execute 20 times
        tx = PingTxAsyncOp(self, CountCondition(self, 20), name="tx")
        rx = PingRxAsyncOp(self, CountCondition(self, 50), name="rx")

        # Define the workflow: tx -> rx using async buffer connector
        self.add_flow(tx, rx, IOSpec.ConnectorType.ASYNC_BUFFER)


def main():
    app = MyPingApp()

    # Create and configure the EventBasedScheduler with 2 worker threads
    scheduler = EventBasedScheduler(app, name="event-based-scheduler", worker_thread_number=2)
    app.scheduler(scheduler)

    with Tracker(app, num_start_messages_to_skip=0, num_last_messages_to_discard=0) as tracker:
        app.run()
        tracker.print()


if __name__ == "__main__":
    main()
