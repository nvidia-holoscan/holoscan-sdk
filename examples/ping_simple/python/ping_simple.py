"""
SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

from holoscan.conditions import CountCondition
from holoscan.core import Application
from holoscan.operators import PingRxOp, PingTxOp


class MyPingApp(Application):
    def compose(self):
        # Define the tx and rx operators, allowing tx to execute 10 times
        tx = PingTxOp(self, CountCondition(self, 10), name="tx")
        rx = PingRxOp(self, name="rx")

        # Define the workflow:  tx -> rx
        self.add_flow(tx, rx)


def main():
    app = MyPingApp()
    app.run()


if __name__ == "__main__":
    main()
