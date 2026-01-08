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

from holoscan.conditions import CountCondition, PeriodicCondition
from holoscan.core import Application, ConditionType, Operator, OperatorSpec


class QueueSizeWarningTxOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.index = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        op_output.emit(self.index, "out")
        self.index += 1


class QueueSizeWarningDefaultRxOp(Operator):
    def setup(self, spec: OperatorSpec):
        # size > 1 with no explicit condition triggers the warning and uses min_size=size.
        spec.input("in", size=2)

    def compute(self, op_input, op_output, context):
        values = op_input.receive("in")
        assert len(values) == 2


class QueueSizeWarningExplicitMinSizeRxOp(Operator):
    def setup(self, spec: OperatorSpec):
        # size=2 (buffering) but min_size=1 (no batching)
        spec.input("in", size=2).condition(
            ConditionType.MESSAGE_AVAILABLE,
            min_size=1,
        )

    def compute(self, op_input, op_output, context):
        # Drain any messages that are currently available.
        while True:
            value = op_input.receive("in", kind="single")
            if value is None:
                break


class QueueSizeWarningDefaultApp(Application):
    def compose(self):
        tx = QueueSizeWarningTxOp(
            self,
            CountCondition(self, 2),
            PeriodicCondition(self, 10_000_000),
            name="tx",
        )
        rx = QueueSizeWarningDefaultRxOp(self, name="rx")
        self.add_flow(tx, rx)


class QueueSizeWarningExplicitMinSizeApp(Application):
    def compose(self):
        tx = QueueSizeWarningTxOp(
            self,
            CountCondition(self, 2),
            PeriodicCondition(self, 10_000_000),
            name="tx",
        )
        rx = QueueSizeWarningExplicitMinSizeRxOp(self, name="rx")
        self.add_flow(tx, rx)


def test_queue_size_warn_default_condition(capfd):
    app = QueueSizeWarningDefaultApp()
    app.run()

    captured = capfd.readouterr()
    assert "Input port 'in' of operator 'rx' is configured with queue_size=2 (> 1)." in captured.err


def test_queue_size_no_warn_when_min_size_is_explicit(capfd):
    app = QueueSizeWarningExplicitMinSizeApp()
    app.run()

    captured = capfd.readouterr()
    assert (
        "Input port 'in' of operator 'rx' is configured with queue_size=2 (> 1)."
        not in captured.err
    )
