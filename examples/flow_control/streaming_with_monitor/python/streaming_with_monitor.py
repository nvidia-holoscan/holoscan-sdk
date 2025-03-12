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
"""

import time

from holoscan.core import Application, ConditionType, IOSpec, Operator, OperatorSpec
from holoscan.schedulers import EventBasedScheduler


class GenSignalOp(Operator):
    """Generates sequential integer values."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = 0

    def setup(self, spec: OperatorSpec):
        spec.output("output")

    def compute(self, op_input, op_output, context):
        self.value += 1
        print(f"{self.name} - Sending value {self.value}")
        op_output.emit(self.value, "output")


## Alternative way to define the operator
# from holoscan.core import Application, ConditionType, IOSpec, Operator
# from holoscan.decorator import Input, Output, create_op
# from holoscan.schedulers import EventBasedScheduler


# @create_op(op_param="self", outputs="output")
# def gen_signal_op(self):
#     if not hasattr(self, "value"):
#         self.value = 0
#     self.value += 1
#     print(f"{self.name} - Sending value {self.value}")
#     return self.value


class ProcessSignalOp(Operator):
    """Processes the input value with a simulated delay."""

    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("output")

    def compute(self, op_input, op_output, context):
        value = op_input.receive("input")
        print(f"{self.name} - Received value {value}")
        # Simulate the processing time
        time.sleep(0.001)  # 1ms
        op_output.emit(value, "output")


## Alternative way to define the operator
# @create_op(op_param="self", inputs=Input("input", arg_map="value"), outputs="output")
# def process_signal_op(self, value):
#     print(f"{self.name} - Received value {value}")
#     # Simulate the processing time
#     time.sleep(0.001)  # 1ms
#     return value


class ExecutionThrottlerOp(Operator):
    """Controls the flow of messages between operators with different processing rates."""

    def setup(self, spec: OperatorSpec):
        spec.input("input")
        # Set the queue policy to POP to prevent push failures when emitting a message to the output
        # port. Additionally, configure the condition to NONE for the execution throttler operator
        # to ensure it executes immediately upon receiving a message, even if the downstream
        # operator is not ready to receive it due to the detect event operator's lengthy (3ms)
        # computation.
        spec.output("output", policy=IOSpec.QueuePolicy.POP).condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):
        value = op_input.receive("input")
        if value is not None:
            op_output.emit(value, "output")


## Alternative way to define the operator
# @create_op(
#     op_param="self",
#     inputs=Input("input", arg_map="value"),
#     outputs=Output("output", policy=IOSpec.QueuePolicy.POP, condition_type=ConditionType.NONE),
# )
# def execution_throttler_op(self, value):
#     if value is not None:
#         return value


class DetectEventOp(Operator):
    """Detects events (odd numbers) with a longer processing time."""

    def setup(self, spec: OperatorSpec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        value = op_input.receive("input")

        self.metadata.set("value", value)
        is_detected = value % 2 == 1
        self.metadata.set("is_detected", is_detected)

        print(f"{self.name} - Received value {value} (set metadata (is_detected: {is_detected}))")
        time.sleep(0.003)  # 3ms


## Alternative way to define the operator
# @create_op(op_param="self", inputs=Input("input", arg_map="value"))
# def detect_event_op(self, value):
#     self.metadata.set("value", value)
#     is_detected = value % 2 == 1
#     self.metadata.set("is_detected", is_detected)

#     print(f"{self.name} - Received value {value} (set metadata (is_detected: {is_detected}))")
#     time.sleep(0.003)  # 3ms


class ReportGenOp(Operator):
    """Generates reports for detected events."""

    def compute(self, op_input, op_output, context):
        value = self.metadata.get("value", -1)
        is_detected = self.metadata.get("is_detected", False)
        print(f"{self.name} - Input value {value}, Metadata value (is_detected: {is_detected})")


## Alternative way to define the operator
# @create_op(op_param="self")
# def report_gen_op(self):
#     value = self.metadata.get("value", -1)
#     is_detected = self.metadata.get("is_detected", False)
#     print(f"{self.name} - Input value {value}, Metadata value (is_detected: {is_detected})")


class VisualizeOp(Operator):
    """Visualizes the processed data."""

    def setup(self, spec: OperatorSpec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        value = op_input.receive("input")
        print(f"{self.name} - Received value {value}")


## Alternative way to define the operator
# @create_op(op_param="self", inputs=Input("input", arg_map="value"))
# def visualize_op(self, value):
#     print(f"{self.name} - Received value {value}")


class StreamExecutionWithMonitorApp(Application):
    def compose(self):
        # Define the operators
        gen_signal = GenSignalOp(self, name="gen_signal")
        process_signal = ProcessSignalOp(self, name="process_signal")
        execution_throttler = ExecutionThrottlerOp(self, name="execution_throttler")
        detect_event = DetectEventOp(self, name="detect_event")
        # Configure the queue policy to POP to ensure the input port avoids a push failure
        # when the execution throttler (lacking DownstreamMessageAffordableCondition on the output
        # port) sends a message while the input port queue is full.
        detect_event.queue_policy("input", IOSpec.IOType.INPUT, IOSpec.QueuePolicy.POP)
        report_generation = ReportGenOp(self, name="report_generation")
        visualize = VisualizeOp(self, name="visualize")

        ## If decorator is used, the operator is defined as follows:
        # gen_signal = gen_signal_op(self, name="gen_signal")
        # process_signal = process_signal_op(self, name="process_signal")
        # execution_throttler = execution_throttler_op(self, name="execution_throttler")
        # detect_event = detect_event_op(self, name="detect_event")
        # detect_event.queue_policy("input", IOSpec.IOType.INPUT, IOSpec.QueuePolicy.POP)
        # report_generation = report_gen_op(self, name="report_generation")
        # visualize = visualize_op(self, name="visualize")

        # Define the streaming with monitor workflow
        #
        # Node Graph:
        #                                                               -> report_generation
        #                                                              |
        #               (cycle)                                        | (is_detected:true)
        #               -------         => execution_throttler => detect_event
        #               \     /         |                              | (is_detected:false)
        # <|start|> -> gen_signal => process_signal => visualize       |
        #              (triggered 5 times)                              -> (ignored)
        self.add_flow(self.start_op(), gen_signal)
        self.add_flow(gen_signal, gen_signal)
        self.add_flow(gen_signal, process_signal, {("output", "input")})
        self.add_flow(process_signal, execution_throttler)
        self.add_flow(execution_throttler, detect_event)
        self.add_flow(detect_event, report_generation)
        self.add_flow(process_signal, visualize)

        def gen_signal_callback(op):
            if not hasattr(gen_signal_callback, "iteration"):
                gen_signal_callback.iteration = 0
            gen_signal_callback.iteration += 1
            print(f"#iteration: {gen_signal_callback.iteration}")

            if gen_signal_callback.iteration <= 10:
                op.add_dynamic_flow("output", process_signal)
                # Signal to trigger itself in the next iteration
                op.add_dynamic_flow(Operator.OUTPUT_EXEC_PORT_NAME, op)
            else:
                gen_signal_callback.iteration = 0

        def detect_event_callback(op):
            # If the detect event operator detects an event, add a dynamic flow to the report
            # generation operator.
            is_detected = op.metadata.get("is_detected", False)
            if is_detected:
                op.add_dynamic_flow(report_generation)

        self.set_dynamic_flows(gen_signal, gen_signal_callback)
        self.set_dynamic_flows(detect_event, detect_event_callback)


def main():
    app = StreamExecutionWithMonitorApp()
    # Use the EventBasedScheduler (with 2 worker threads) because we want to ensure that the
    # visualize operator is scheduled to run in parallel with the detect event operator. The
    # detect event operator is expected to take a long time to process the event so both the
    # detect_event and visualize operators are scheduled to run in parallel.
    scheduler = EventBasedScheduler(app, worker_thread_number=2, stop_on_deadlock=True)
    app.scheduler(scheduler)
    app.run()


if __name__ == "__main__":
    main()

# Expected output:
# (The output is not deterministic, so the order of the logs may vary.)
#
# gen_signal - Sending value 1
# #iteration: 1
# process_signal - Received value 1
# gen_signal - Sending value 2
# #iteration: 2
# visualize - Received value 1
# process_signal - Received value 2
# detect_event - Received value 1 (set metadata (is_detected: true))
# gen_signal - Sending value 3
# #iteration: 3
# visualize - Received value 2
# process_signal - Received value 3
# gen_signal - Sending value 4
# #iteration: 4
# visualize - Received value 3
# process_signal - Received value 4
# gen_signal - Sending value 5
# #iteration: 5
# report_generation - Input value 1, Metadata value (is_detected: true)
# detect_event - Received value 3 (set metadata (is_detected: true))
# visualize - Received value 4
# process_signal - Received value 5
# gen_signal - Sending value 6
# #iteration: 6
# visualize - Received value 5
# process_signal - Received value 6
# gen_signal - Sending value 7
# #iteration: 7
# visualize - Received value 6
# process_signal - Received value 7
# gen_signal - Sending value 8
# #iteration: 8
# report_generation - Input value 3, Metadata value (is_detected: true)
# detect_event - Received value 6 (set metadata (is_detected: false))
# visualize - Received value 7
# process_signal - Received value 8
# gen_signal - Sending value 9
# #iteration: 9
# visualize - Received value 8
# process_signal - Received value 9
# gen_signal - Sending value 10
# #iteration: 10
# detect_event - Received value 8 (set metadata (is_detected: false))
# visualize - Received value 9
# process_signal - Received value 10
# gen_signal - Sending value 11
# #iteration: 11
# visualize - Received value 10
# detect_event - Received value 10 (set metadata (is_detected: false))
