"""
SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

import pytest

from holoscan.conditions import CountCondition
from holoscan.core import Application, Fragment
from holoscan.operators import PingRxOp, PingTxOp

from .utils import remove_ignored_errors


class TxFragment(Fragment):
    def __init__(self, *args, count=10, **kwargs):
        self.count = count
        super().__init__(*args, **kwargs)

    def compose(self):
        tx = PingTxOp(self, CountCondition(self, self.count), name="tx")
        self.add_operator(tx)


class RxFragment(Fragment):
    def compose(self):
        rx = PingRxOp(self, name="rx")  # noqa: F841
        # intentionally do not call add_operator in this case to test error message
        # self.add_operator(rx)


class MyPingApp(Application):
    def __init__(self, *args, count=10, **kwargs):
        self.count = count
        super().__init__(*args, **kwargs)

    def compose(self):
        tx_frag = TxFragment(self, count=self.count, name="tx_fragment")
        rx_frag = RxFragment(self, name="rx_fragment")

        self.add_flow(tx_frag, rx_frag, {("tx", "rx")})


def test_distributed_app_invalid_fragment_compose(capfd):
    count = 5
    app = MyPingApp(count=count)
    with pytest.raises(RuntimeError, match="Fragment 'rx_fragment' does not have any operators"):
        app.run()

    # assert that no errors were logged
    captured = capfd.readouterr()

    assert "error" in remove_ignored_errors(captured.err)
    assert "Fragment 'rx_fragment' does not have any operators" in captured.err
