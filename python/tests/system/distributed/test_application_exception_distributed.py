"""
SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

from holoscan.conditions import CountCondition
from holoscan.core import Application, Fragment, Operator, OperatorSpec
from holoscan.operators import PingTxOp

from ..env_wrapper import env_var_context

NUM_EXCEPTIONS = 10


class BrokenRxOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        value = op_input.receive("in")  # noqa: F841

        # intentionally cause a ZeroDivisionError to test exception_handling
        1 / 0  # noqa: B018


class TxFragment(Fragment):
    def compose(self):
        tx = PingTxOp(self, CountCondition(self, NUM_EXCEPTIONS), name="tx")
        self.add_operator(tx)


class BadRxFragment(Fragment):
    def compose(self):
        rx = BrokenRxOp(self, CountCondition(self, NUM_EXCEPTIONS), name="rx")
        self.add_operator(rx)


class BadDistributedApplication(Application):
    def __init__(self, *args, value=1, **kwargs):
        self.value = value
        super().__init__(*args, **kwargs)

    def compose(self):
        tx_fragment = TxFragment(self, name="tx_fragment")
        rx_fragment = BadRxFragment(self, name="rx_fragment")

        # Connect the two fragments (tx.out -> rx.in)
        self.add_flow(tx_fragment, rx_fragment, {("tx.out", "rx.in")})


def test_exception_handling_distributed():
    app = BadDistributedApplication()

    # Global timeouts set for CTest Python distributed test runs are currently 2500, but this case
    # sometimes fails. Try 5000 instead to see if this helps reduce stochastic CI test failures.
    env_var_settings = {
        ("HOLOSCAN_MAX_DURATION_MS", "5000"),
        # network_connection_timeout (default 5s) handles connection setup, so shorter timeout is
        # safe
        ("HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT", "1000"),
    }

    exception_occurred = False
    with env_var_context(env_var_settings):
        try:
            app.run()
        except ZeroDivisionError:
            exception_occurred = True

    assert exception_occurred
