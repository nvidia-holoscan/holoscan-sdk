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

import pytest

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
        self.add_operator(PingRxOp(self, name="rx"))


class MyPingApp(Application):
    def __init__(self, *args, count=10, ports_arg, **kwargs):
        self.count = count
        self.ports_arg = ports_arg
        super().__init__(*args, **kwargs)

    def compose(self):
        tx_frag = TxFragment(self, count=self.count, name="tx_fragment")
        rx_frag = RxFragment(self, name="rx_fragment")

        if self.ports_arg:
            self.add_flow(tx_frag, rx_frag, self.ports_arg)
        else:
            self.add_flow(tx_frag, rx_frag)


@pytest.mark.parametrize(
    "ports_arg, invalid_port, invalid_type",
    [
        # valid names
        ({("tx.out", "rx.in")}, None, None),
        ({("tx", "rx")}, None, None),
        # non-existent operator name(s)
        ({("transmitter.out", "rx.in")}, "upstream", "operator"),
        ({("tx.out", "receiver.in")}, "downstream", "operator"),
        ({("transmitter.out", "receiver.in")}, "both", "operator"),
        # non-existent port name(s)
        ({("tx.output", "rx.in")}, "upstream", "port"),
        ({("tx.out", "rx.input")}, "downstream", "port"),
        ({("tx.output", "rx.input")}, "both", "port"),
    ],
)
def test_distributed_app_invalid_add_flow(ports_arg, invalid_port, invalid_type, capfd):
    count = 5
    app = MyPingApp(count=count, ports_arg=ports_arg)
    app.run()

    # assert that no errors were logged
    captured = capfd.readouterr()

    if invalid_port is None:
        assert "error" not in captured.err
        assert "Exception occurred" not in captured.err
    else:
        assert "error" in captured.err
        if invalid_type == "operator":
            err_msg_upstream = "Cannot find source operator"
            err_msg_downstream = "Cannot find target operator"
        elif invalid_type == "port":
            err_msg_upstream = (
                "Source operator 'tx' in fragment 'tx_fragment' does not have a port named"
            )
            err_msg_downstream = (
                "Target operator 'rx' in fragment 'rx_fragment' does not have a port named"
            )

        if invalid_port in ["upstream"]:
            assert err_msg_upstream in captured.err
        elif invalid_port in ["downstream"]:
            assert err_msg_downstream in captured.err
        elif invalid_port in ["both"]:
            # In pracitce, the app terminates after the upstream error is printed, but I didn't
            # want to guarantee which error would be printed first.
            assert err_msg_upstream in captured.err or err_msg_downstream in captured.err
