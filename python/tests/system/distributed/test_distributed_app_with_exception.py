"""
SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest import mock

from holoscan.conditions import CountCondition
from holoscan.core import Application, Fragment
from holoscan.operators import PingRxOp, PingTxOp


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
        self.add_operator(rx)
        # We intentionally generate error here by calling unsupported function
        # so that the exception part gets tested.
        self.GenerateError()


class MyPingApp(Application):
    def __init__(self, *args, count=10, **kwargs):
        self.count = count
        super().__init__(*args, **kwargs)

    def compose(self):
        tx_frag = TxFragment(self, count=self.count, name="tx_fragment")
        rx_frag = RxFragment(self, name="rx_fragment")

        self.add_flow(tx_frag, rx_frag, {("tx", "rx")})


def test_distributed_app_with_exception(capfd):
    count = 5
    with mock.patch("sys.argv", ["app.py", "--driver", "--worker", "--fragments=all"]):
        app = MyPingApp(count=count)
        app.run()

    # assert that the expected error was logged
    captured = capfd.readouterr()

    assert "error" in captured.err
    assert (
        "Failed to retrieve port info for all fragments scheduled on worker with id" in captured.err
    )
    assert (
        "GetFragmentInfo failed: AttributeError: 'RxFragment' object has no attribute "
        "'GenerateError'" in captured.err
    )
