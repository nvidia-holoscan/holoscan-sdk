"""
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

from holoscan.conditions import CountCondition, PeriodicCondition
from holoscan.core import Application, ConditionType, Operator, OperatorSpec
from holoscan.operators import PingTxOp


class DefaultMinSizeRxOp(Operator):
    def setup(self, spec: OperatorSpec):
        # size > 1 with no explicit condition triggers the warning and uses min_size=size.
        spec.input("in", size=2)

    def compute(self, op_input, op_output, context):
        values = op_input.receive("in")
        assert len(values) == 2


class ExplicitMinSizeRxOp(Operator):
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
        tx = PingTxOp(
            self,
            CountCondition(self, 2),
            PeriodicCondition(self, 10_000_000),
            name="tx",
        )
        rx = DefaultMinSizeRxOp(self, name="rx")
        self.add_flow(tx, rx)


class QueueSizeWarningExplicitMinSizeApp(Application):
    def compose(self):
        tx = PingTxOp(
            self,
            CountCondition(self, 2),
            PeriodicCondition(self, 10_000_000),
            name="tx",
        )
        rx = ExplicitMinSizeRxOp(self, name="rx")
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
