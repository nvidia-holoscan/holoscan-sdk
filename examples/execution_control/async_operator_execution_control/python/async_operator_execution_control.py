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

import threading
import time

from holoscan.conditions import AsynchronousEventState
from holoscan.core import Application, Operator
from holoscan.schedulers import EventBasedScheduler


class SimpleOp(Operator):
    """A simple operator that can be controlled by the ControllerOp."""

    def __init__(self, fragment, *args, notification_callback=None, **kwargs):
        self.notification_callback = notification_callback
        super().__init__(fragment, *args, **kwargs)

    def initialize(self):
        # Call parent's initialize method
        super().initialize()

        # Set the operator to WAIT state to wait for the controller to change the state to READY
        self.async_condition.event_state = AsynchronousEventState.WAIT

    def compute(self, op_input, op_output, context):
        print(f"[{self.name}] Executing compute method")

        # Set async condition to WAIT state for next event
        self.async_condition.event_state = AsynchronousEventState.WAIT

        # Call the notification callback to notify the controller
        if self.notification_callback:
            self.notification_callback()


class ControllerOp(Operator):
    """A controller operator that manages the execution of other operators."""

    def __init__(self, fragment, *args, **kwargs):
        self.op_map = {}  # Map of operator name to operator
        self.mutex = threading.Lock()
        self.operator_execution_cv = threading.Condition(self.mutex)
        self.start_cv = threading.Condition(self.mutex)
        self.is_operator_executed = False
        self.is_started = False
        super().__init__(fragment, *args, **kwargs)

    def initialize(self):
        # Call parent's initialize method
        super().initialize()

        # Get all operators in the application except this one
        all_ops = self.fragment.graph.get_nodes()
        for op in all_ops:
            if op.name != self.name:
                self.op_map[op.name] = op

    def wait_for_start(self):
        """Wait for the controller to be started.

        Blocks the calling thread until the controller's start method has been called
        and the is_started flag is set to true.
        """
        with self.start_cv:
            self.start_cv.wait_for(lambda: self.is_started)

    def start(self):
        # Set this operator's async condition to EVENT_WAITING to prevent its compute method
        # from being called until we set it to EVENT_DONE
        # Note that at least one operator's event state (in this case, the controller) needs to be
        # set to EVENT_WAITING instead of WAIT to prevent the application from being terminated at
        # start by the deadlock detector.
        self.async_condition.event_state = AsynchronousEventState.EVENT_WAITING

        # Notify that controller has started
        with self.start_cv:
            self.is_started = True
            self.start_cv.notify_all()

    def stop(self):
        print(f"[{self.name}] Stopping controller")

    def compute(self, op_input, op_output, context):
        # If the event state is set to EVENT_DONE, the compute method will be called.
        # Setting the event state to EVENT_DONE triggers the scheduler,
        # which updates the operator's scheduling condition type to READY, leading to the execution
        # of the compute method.
        print(f"[{self.name}] Stopping controller execution")
        # Stop this operator's execution
        self.stop_execution()

    def notify_callback(self):
        """Method to be called by SimpleOp operators to notify this controller."""
        with self.operator_execution_cv:
            self.is_operator_executed = True
            self.operator_execution_cv.notify_all()

    def execute_operator(self, op_name):
        """Executes a specific operator by name."""
        if op_name not in self.op_map:
            print(f"[{self.name}] Operator {op_name} not found")
            return

        op = self.op_map[op_name]
        # Set the operator to EVENT_DONE to signal the scheduler that the operator is ready to
        # execute
        op.async_condition.event_state = AsynchronousEventState.EVENT_DONE

        # Wait for notification from the operator
        with self.operator_execution_cv:
            self.operator_execution_cv.wait_for(lambda: self.is_operator_executed)
            self.is_operator_executed = False

    def shutdown(self):
        """Shuts down the application by setting all operators to EVENT_NEVER.

        Set all operators' event states to EVENT_NEVER to change their scheduling condition type
        to NEVER.
        Then sets the controller to EVENT_DONE to stop the application.
        """
        print(f"[{self.name}] Shutting down controller")

        # Set all operators' event states to EVENT_NEVER
        for op in self.op_map.values():
            op.async_condition.event_state = AsynchronousEventState.EVENT_NEVER

        # Set the controller to EVENT_DONE to stop the application (this will trigger the scheduler
        # to get notified and make the scheduling condition type of the operator to be READY,
        # executing the compute method)
        self.async_condition.event_state = AsynchronousEventState.EVENT_DONE


class AsyncOperatorExecutionControlApp(Application):
    """Application demonstrating async operator execution control."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.controller = None

    def get_controller(self):
        """Get the controller operator after it has started."""
        while self.controller is None:
            time.sleep(0.1)
        self.controller.wait_for_start()
        return self.controller

    def compose(self):
        # Create the controller operator
        self.controller = ControllerOp(self, name="controller")

        # Create a notification callback function
        def notify_callback():
            self.controller.notify_callback()

        # Create multiple SimpleOp operators, all using the controller as callback
        op1 = SimpleOp(self, name="op1", notification_callback=notify_callback)
        op2 = SimpleOp(self, name="op2", notification_callback=notify_callback)
        op3 = SimpleOp(self, name="op3", notification_callback=notify_callback)

        # Add all operators to the application (no flows between them since controller manages
        # execution)
        self.add_operator(self.controller)
        self.add_operator(op1)
        self.add_operator(op2)
        self.add_operator(op3)

        # Print information about the example
        print("This example demonstrates async operator execution control.")
        print("The controller operator runs in a separate thread and synchronizes")
        print("the execution of multiple SimpleOp operators using async_condition.")
        print("-------------------------------------------------------------------")
        print("Key concepts demonstrated:")
        print("1. Using asynchronous conditions to control operator execution states")
        print("2. External execution control from outside the Holoscan runtime")
        print("3. Notification callbacks for coordination between operators")
        print("4. Manual scheduling of operators in a specific order (op3 → op2 → op1)")
        print("5. Graceful shutdown of an application with async operators")
        print("-------------------------------------------------------------------")
        print("Execution flow:")
        print("- All operators start in WAIT state except the controller (EVENT_WAITING)")
        print("- Main thread gets controller and executes operators in sequence")
        print("- Each operator signals completion via the notification callback")
        print("- The controller then shuts down the application")
        print("-------------------------------------------------------------------")


def main():
    """Main function to run the application."""
    app = AsyncOperatorExecutionControlApp()
    # This example works with any scheduler (GreedyScheduler, MultiThreadScheduler, etc.)
    # with any number of worker threads.
    # Here we use EventBasedScheduler for demonstration purposes.
    scheduler = EventBasedScheduler(app, name="EBS", worker_thread_number=3)
    app.scheduler(scheduler)
    # Run the application asynchronously
    future = app.run_async()

    # Get the controller after it has started
    controller = app.get_controller()

    # Executing operators outside of the Holoscan runtime
    controller.execute_operator("op3")
    controller.execute_operator("op2")
    controller.execute_operator("op1")

    # Shutting down the application
    controller.shutdown()
    # Waiting for the application to complete
    future.result()

    print("-------------------------------------------------------------------")
    print("Application completed. All operators have finished execution.")


if __name__ == "__main__":
    main()
