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

from holoscan.conditions import (
    AsynchronousCondition,
    AsynchronousEventState,
    CountCondition,
    PeriodicCondition,
)
from holoscan.core import Application, Operator, OperatorSpec, OperatorStatus
from holoscan.schedulers import EventBasedScheduler, MultiThreadScheduler


class SourceOp(Operator):
    """Test source operator that emits a fixed number of values."""

    def __init__(self, fragment, *args, max_count=3, **kwargs):
        self.count = 0
        self.max_count = max_count
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        if self.count < self.max_count:
            op_output.emit(self.count, "out")
        self.count += 1


class ProcessorOp(Operator):
    """Test processor operator that processes data."""

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        data = op_input.receive("in")
        result = data * 2
        op_output.emit(result, "out")


class ConsumerOp(Operator):
    """Test consumer operator that stops after receiving a certain number of values."""

    def __init__(self, fragment, *args, stop_after=2, **kwargs):
        self.received_values = []
        self.stop_after = stop_after
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        data = op_input.receive("in")
        self.received_values.append(data)

        # Stop execution after receiving a certain number of values
        if len(self.received_values) >= self.stop_after:
            self.stop_execution()


class MonitorOp(Operator):
    """Test monitor operator that tracks the status of other operators."""

    def __init__(self, fragment, *args, **kwargs):
        self.monitored_operators = kwargs.pop("monitored_operators", [])
        self.operator_statuses = {}
        self.idle_count = 0
        super().__init__(fragment, *args, **kwargs)

    def compute(self, op_input, op_output, context):
        # Check the status of all monitored operators
        is_pipeline_idle = True
        for op_name in self.monitored_operators:
            try:
                status = context.get_operator_status(op_name)
                self.operator_statuses[op_name] = status
                # If any operator is not idle, the pipeline is not idle
                if status != OperatorStatus.IDLE:
                    is_pipeline_idle = False
            except Exception as e:  # noqa: PERF203
                print(f"Error getting status for {op_name}: {e}")
                is_pipeline_idle = False

        # If all operators are idle, increment the idle count
        if is_pipeline_idle:
            self.idle_count += 1
            # Stop the application if all operators have been idle for a while
            if self.idle_count >= 3:
                print(
                    f"[{self.name}] All operators have been idle for "
                    f"{self.idle_count} iterations. Stopping application."
                )
                # Stop the entire application, not just this operator
                self.fragment.stop_execution()
        else:
            # Reset the idle count if any operator is not idle
            self.idle_count = 0


class OperatorStatusApp(Application):
    """Test application for operator status tracking."""

    def compose(self):
        # Create operators
        self.source = SourceOp(self, CountCondition(self, 10), name="source", max_count=3)
        self.processor = ProcessorOp(self, name="processor")
        self.consumer = ConsumerOp(self, name="consumer", stop_after=2)

        # Create monitor with its own scheduling condition
        self.monitor = MonitorOp(
            self,
            PeriodicCondition(self, 0.01),  # 10ms
            name="monitor",
            monitored_operators=["source", "processor", "consumer"],
        )

        # Define the workflow
        self.add_flow(self.source, self.processor, {("out", "in")})
        self.add_flow(self.processor, self.consumer, {("out", "in")})

        # Add monitor (not connected to other operators)
        self.add_operator(self.monitor)


def test_operator_status_tracking(capfd):
    """Test that we can track the status of operators."""
    app = OperatorStatusApp()
    app.scheduler(EventBasedScheduler(app, name="eventbased", worker_thread_number=2))
    app.run()

    # Verify that the consumer received the expected values
    consumer = app.consumer
    received_values = consumer.received_values

    # Should have received 2 values (0*2 and 1*2)
    assert len(received_values) == 2
    assert received_values[0] == 0
    assert received_values[1] == 2

    # Verify that the monitor collected operator statuses
    monitor = app.monitor
    statuses = monitor.operator_statuses

    # Should have status entries for all operators
    assert "source" in statuses
    assert "processor" in statuses
    assert "consumer" in statuses

    captured = capfd.readouterr()
    assert "error" not in captured.err


def test_stop_execution(capfd):
    """Test that operators can stop their own execution and fragments can stop the entire
    application.

    This test verifies:
    1. The consumer operator stops itself after receiving 2 values using operator.stop_execution()
    2. The monitor operator stops the entire application when all operators have been idle for 3
       iterations using fragment.stop_execution()
    """
    app = OperatorStatusApp()
    app.scheduler(MultiThreadScheduler(app, name="multithread", worker_thread_number=2))
    app.run()

    # Verify that the consumer stopped after receiving 2 values
    # (using operator.stop_execution() to stop just itself)
    consumer = app.consumer
    received_values = consumer.received_values

    assert len(received_values) == 2

    # Verify that the monitor also collected statuses before stopping the entire application
    # (using fragment.stop_execution() to stop all operators)
    monitor = app.monitor
    statuses = monitor.operator_statuses

    # Monitor should have collected statuses before stopping
    assert statuses

    captured = capfd.readouterr()
    assert "error" not in captured.err


def test_execution_context_available(capfd):
    """Test that we can find operators by name using the execution context."""
    app = OperatorStatusApp()
    app.scheduler(MultiThreadScheduler(app, name="multithread", worker_thread_number=2))
    app.run()

    # Get the monitor operator
    monitor = app.monitor

    # After the application finishes, the execution context is no longer available
    execution_context = monitor.execution_context
    assert execution_context is None

    captured = capfd.readouterr()
    assert "error" not in captured.err


def test_async_condition(capfd):
    """Test that we can access and use the async_condition property."""

    class AsyncTestOp(Operator):
        def compute(self, op_input, op_output, context):
            print("async_op: compute called")
            # Access the async_condition property
            assert self.async_condition is not None
            assert isinstance(self.async_condition, AsynchronousCondition)

            # Test setting the event state to EVENT_NEVER
            # This is equivalent to calling stop_execution() which internally does:
            # self.async_condition.event_state = AsynchronousEventState.EVENT_NEVER
            self.async_condition.event_state = AsynchronousEventState.EVENT_NEVER

    class AsyncTestApp(Application):
        def compose(self):
            self.op = AsyncTestOp(self, name="async_op")
            self.add_operator(self.op)

    app = AsyncTestApp()
    app.scheduler(EventBasedScheduler(app, name="eventbased", worker_thread_number=2))
    app.run()

    # The test passes if the app runs without errors
    captured = capfd.readouterr()
    assert "error" not in captured.err.lower()
    assert captured.out.count("async_op: compute called") == 1


def test_execution_context_in_lifecycle_methods(capfd):
    """Test that execution_context is available in all lifecycle methods.

    This test verifies that:
    1. execution_context returns a non-None value in initialize(), start(), stop(), and compute()
    2. find_operator() works in all these methods
    """

    class ExecutionContextTestOp(Operator):
        def __init__(self, fragment, *args, **kwargs):
            self.initialize_context_valid = False
            self.start_context_valid = False
            self.compute_context_valid = False
            self.stop_context_valid = False
            super().__init__(fragment, *args, **kwargs)

        def initialize(self):
            # Test execution_context in initialize()
            context = self.execution_context
            assert context is not None
            self.initialize_context_valid = True

            # Test find_operator() in initialize()
            self_op = context.find_operator(self.name)
            assert self_op is not None

            print(f"[{self.name}] initialize: execution_context is valid")

        def start(self):
            # Test execution_context in start()
            context = self.execution_context
            assert context is not None
            self.start_context_valid = True

            # Test find_operator() in start()
            self_op = context.find_operator(self.name)
            assert self_op is not None

            print(f"[{self.name}] start: execution_context is valid")

        def compute(self, op_input, op_output, context):
            # Test execution_context in compute()
            exec_context = self.execution_context
            assert exec_context is not None
            self.compute_context_valid = True

            # Test find_operator() in compute()
            self_op = exec_context.find_operator(self.name)
            assert self_op is not None

            print(f"[{self.name}] compute: execution_context is valid")

            self.stop_execution()

        def stop(self):
            # Test execution_context in stop()
            context = self.execution_context
            assert context is not None
            self.stop_context_valid = True

            # Test find_operator() in stop()
            self_op = context.find_operator(self.name)
            assert self_op is not None

            print(f"[{self.name}] stop: execution_context is valid")

    class ExecutionContextTestApp(Application):
        def compose(self):
            self.op = ExecutionContextTestOp(self, name="context_test_op")
            self.add_operator(self.op)

    app = ExecutionContextTestApp()
    app.scheduler(EventBasedScheduler(app, name="eventbased", worker_thread_number=2))
    app.run()

    # Verify that execution_context was valid in all lifecycle methods
    op = app.op
    assert op.initialize_context_valid
    assert op.start_context_valid
    assert op.compute_context_valid
    assert op.stop_context_valid

    # Verify the output
    captured = capfd.readouterr()
    assert captured.out.count("execution_context is valid") == 4


def test_find_other_operators(capfd):
    """Test that execution_context.find_operator() can be used to find other operators.

    This test verifies that an operator can find other operators in the application
    using the execution_context.find_operator() method during different lifecycle phases.
    """

    class SourceOp(Operator):
        def setup(self, spec):
            spec.output("out")

        def compute(self, op_input, op_output, context):
            op_output.emit(42, "out")
            self.stop_execution()

    class SinkOp(Operator):
        def __init__(self, fragment, *args, **kwargs):
            self.found_source_in_initialize = False
            self.found_source_in_start = False
            self.found_source_in_compute = False
            self.found_source_in_stop = False
            super().__init__(fragment, *args, **kwargs)

        def setup(self, spec):
            spec.input("in")

        def initialize(self):
            # Try to find the source operator during initialization
            context = self.execution_context
            assert context is not None

            source_op = context.find_operator("source_op")
            # Store the result for later verification
            self.found_source_in_initialize = source_op is not None

            print(f"[{self.name}] initialize: found_source = {self.found_source_in_initialize}")

        def start(self):
            # Try to find the source operator during start
            context = self.execution_context
            assert context is not None

            source_op = context.find_operator("source_op")
            # Store the result for later verification
            self.found_source_in_start = source_op is not None

            print(f"[{self.name}] start: found_source = {self.found_source_in_start}")

        def compute(self, op_input, op_output, context):
            # Try to find the source operator during compute
            exec_context = self.execution_context
            assert exec_context is not None

            source_op = exec_context.find_operator("source_op")
            # Store the result for later verification
            self.found_source_in_compute = source_op is not None

            print(f"[{self.name}] compute: found_source = {self.found_source_in_compute}")

            # Receive the value from the source
            value = op_input.receive("in")
            assert value == 42

        def stop(self):
            # Try to find the source operator during stop
            context = self.execution_context
            assert context is not None

            source_op = context.find_operator("source_op")
            # Store the result for later verification
            self.found_source_in_stop = source_op is not None

            print(f"[{self.name}] stop: found_source = {self.found_source_in_stop}")

    class FindOperatorsApp(Application):
        def compose(self):
            self.source = SourceOp(self, name="source_op")
            self.sink = SinkOp(self, name="sink_op")

            # Connect the operators
            self.add_flow(self.source, self.sink, {("out", "in")})

    app = FindOperatorsApp()
    app.scheduler(EventBasedScheduler(app, name="eventbased", worker_thread_number=2))
    app.run()

    # Verify that the sink operator could find the source operator
    sink = app.sink
    assert sink.found_source_in_initialize
    assert sink.found_source_in_start
    assert sink.found_source_in_compute
    assert sink.found_source_in_stop

    # Verify the output
    captured = capfd.readouterr()
    assert captured.out.count("found_source = True") == 4
