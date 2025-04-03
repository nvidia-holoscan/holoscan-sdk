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

from holoscan.conditions import CountCondition, PeriodicCondition
from holoscan.core import Application, Operator, OperatorSpec, OperatorStatus
from holoscan.schedulers import EventBasedScheduler


class FiniteSourceOp(Operator):
    """A simple operator that processes a fixed number of times and then completes."""

    def __init__(self, fragment, *args, max_count=5, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.max_count = max_count
        self.count = 0

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        if self.count < self.max_count:
            print(f"[{self.name}] Emitting data: {self.count}")
            op_output.emit(self.count, "out")

            # Simulate some processing time
            time.sleep(0.2)
        else:
            print(f"[{self.name}] Additional iterations after completion: {self.count}")

        self.count += 1

    def stop(self):
        print(f"[{self.name}] Stopping operator")


class ProcessorOp(Operator):
    """An operator that processes data."""

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        data = op_input.receive("in")

        # Process the data
        result = data * 2
        print(f"[{self.name}] Processing data: {data} -> {result}")

        # Emit the processed data
        op_output.emit(result, "out")

        # Simulate some processing time
        time.sleep(0.1)

    def stop(self):
        print(f"[{self.name}] Stopping operator")


class ConsumerOp(Operator):
    """An operator that consumes data."""

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        data = op_input.receive("in")

        # Consume the data
        print(f"[{self.name}] Consuming data: {data}")

        # Simulate some processing time
        time.sleep(0.15)

    def stop(self):
        print(f"[{self.name}] Stopping operator")


class MonitorOp(Operator):
    """A dedicated monitoring operator that runs independently of the main pipeline."""

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.monitored_operators = kwargs.pop("monitored_operators", [])
        self.consecutive_idle_count = 0

    def compute(self, op_input, op_output, context):
        print(f"[{self.name}] Operator status summary:")

        is_pipeline_idle = True
        for op_name in self.monitored_operators:
            status = context.get_operator_status(op_name)
            print(f"  - {op_name}: {status.name}")
            if status != OperatorStatus.IDLE:
                is_pipeline_idle = False

        if is_pipeline_idle:
            self.consecutive_idle_count += 1
            print("!!IDLE!!")

            # Currently, there's no way to retrieve the computed SchedulingCondition status from the
            # operator (i.e., whether the computed scheduling condition status is NEVER).
            # Instead, we use `consecutive_idle_count` to determine if all operators, except the
            # monitor operator, have completed.
            # If `consecutive_idle_count` is equal to or greater than the hardcoded value 3,
            # we consider them completed, as there are cases where all operators are idle but not
            # yet completed.
            # A better approach for checking an operator's computed scheduling condition will be
            # available in a future release.
            if self.consecutive_idle_count >= 3:
                print(f"[{self.name}] All operators have completed.")
                # Stop the monitor operator which is the only operator keeping the application alive
                self.stop_execution()  # the application will terminate through a deadlock
        else:
            self.consecutive_idle_count = 0


class OperatorStatusTrackingApp(Application):
    def compose(self):
        # Define the operators
        source = FiniteSourceOp(self, CountCondition(self, 10), name="source", max_count=5)
        processor = ProcessorOp(self, name="processor")
        consumer = ConsumerOp(self, name="consumer")

        # Create the independent monitor operator with its own scheduling condition
        # This will continue running even if the main pipeline stops
        monitor = MonitorOp(
            self,
            PeriodicCondition(self, 0.05),  # 50ms in seconds
            name="monitor",
            monitored_operators=["source", "processor", "consumer"],
        )

        # Define the workflow for the main pipeline
        self.add_flow(source, processor, {("out", "in")})
        self.add_flow(processor, consumer, {("out", "in")})

        # The monitor is not connected to any other operators - it runs independently
        self.add_operator(monitor)

        # Print information about the execution context API
        print("This example demonstrates the Operator Status Tracking API.")
        print("The source operator will emit 5 values and then stop executing after 10 iterations.")
        print("The monitor operator runs independently and tracks the status of all operators.")
        print("When operators complete, the monitor will be terminated.")
        print("-------------------------------------------------------------------")


def main():
    app = OperatorStatusTrackingApp()
    scheduler = EventBasedScheduler(app, worker_thread_number=2)
    app.scheduler(scheduler)
    app.run()

    print("-------------------------------------------------------------------")
    print("Application completed. All operators have finished processing.")


if __name__ == "__main__":
    main()
