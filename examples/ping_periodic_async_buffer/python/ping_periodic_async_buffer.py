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

import argparse
import datetime

from holoscan.conditions import (
    CountCondition,
    PeriodicCondition,
    PeriodicConditionPolicy,
)
from holoscan.core import Application, IOSpec, Operator, OperatorSpec
from holoscan.schedulers import EventBasedScheduler


class PingTxOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        self._count = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):  # noqa: D401
        self._count += 1
        op_output.emit(self._count, "out")
        print(f"Tx message sent: {self._count}")


class PingRxOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        value = op_input.receive("in")
        if value is None:
            print(f"Operator '{self.name}' did not receive a valid value.")
            return
        print(f"Rx message received: {value}")


class PingPeriodicAsyncApp(Application):
    """Ping application demonstrating PeriodicCondition + AsyncBuffer connector."""

    def __init__(self, tx_period_ms: int = 100, rx_period_ms: int = 200, **kwargs):
        self._tx_period_ms = tx_period_ms
        self._rx_period_ms = rx_period_ms
        super().__init__(**kwargs)

    def compose(self):
        # Transmitter operator: executes 10 times with at least `tx_period_ms`
        # milliseconds between ticks.
        tx = PingTxOp(
            self,
            CountCondition(self, 10),
            PeriodicCondition(
                fragment=self,
                recess_period=datetime.timedelta(milliseconds=self._tx_period_ms),
                policy=PeriodicConditionPolicy.MIN_TIME_BETWEEN_TICKS,
                name="periodic-condition-tx",
            ),
            name="tx",
        )

        # Receiver operator: executes 10 times with at least `rx_period_ms`
        # milliseconds between ticks.
        rx = PingRxOp(
            self,
            CountCondition(self, 10),
            PeriodicCondition(
                fragment=self,
                recess_period=datetime.timedelta(milliseconds=self._rx_period_ms),
                policy=PeriodicConditionPolicy.MIN_TIME_BETWEEN_TICKS,
                name="periodic-condition-rx",
            ),
            name="rx",
        )

        # Connect using an async buffer so the two operators can progress
        # independently.
        self.add_flow(tx, rx, IOSpec.ConnectorType.ASYNC_BUFFER)
        # or with port names
        # self.add_flow(tx, rx, {("out", "in")}, IOSpec.ConnectorType.ASYNC_BUFFER)


def main():
    parser = argparse.ArgumentParser(
        prog="ping_periodic_async_buffer",
        description="Ping application demonstrating PeriodicCondition + AsyncBuffer connector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "tx_period_ms",
        nargs="?",
        type=int,
        help="Period for the transmitter operator in milliseconds.",
        default=100,
    )
    parser.add_argument(
        "rx_period_ms",
        nargs="?",
        type=int,
        help="Period for the receiver operator in milliseconds.",
        default=200,
    )
    parser.add_argument(
        "use_event_based_scheduler",
        nargs="?",
        choices=["1"],
        help="Use EventBasedScheduler instead of the default greedy scheduler (use '1' to enable).",
    )
    args = parser.parse_args()

    # Handle period defaults
    tx_period_ms = args.tx_period_ms
    rx_period_ms = args.rx_period_ms

    if tx_period_ms <= 0 or rx_period_ms <= 0:
        parser.error(
            "ERROR: tx_period_ms and rx_period_ms must be greater than 0."
            f"Currently provided values are: tx_period_ms: {tx_period_ms},"
            f"rx_period_ms: {rx_period_ms}"
        )

    use_event_based_scheduler = args.use_event_based_scheduler == "1"

    app = PingPeriodicAsyncApp(tx_period_ms=tx_period_ms, rx_period_ms=rx_period_ms)

    # Optional event-based scheduler matching the behaviour of the C++ demo
    if use_event_based_scheduler:
        scheduler = EventBasedScheduler(
            app,
            name="event-based-scheduler",
            worker_thread_number=2,
        )
        app.scheduler(scheduler)

    app.run()


if __name__ == "__main__":
    main()
