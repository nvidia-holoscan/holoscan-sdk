"""
SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

import datetime

from holoscan.conditions import CountCondition, PeriodicCondition
from holoscan.core import Application
from holoscan.operators import PingRxOp, PingTxOp


class MyPingApp(Application):
    def compose(self):
        # Note: Arguments must be positional, not keyword arguments
        tx = PingTxOp(
            self,
            CountCondition(self, 10),
            PeriodicCondition(
                fragment=self,
                recess_period=datetime.timedelta(microseconds=200_000),
                policy="MinTimeBetweenTicks",
                name="noname_periodic_condition",
            ),
            name="tx",
        )
        rx = PingRxOp(self, name="rx")

        self.add_flow(tx, rx)


def main():
    app = MyPingApp()
    app.run()


if __name__ == "__main__":
    main()
