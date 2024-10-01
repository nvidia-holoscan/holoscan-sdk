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
from holoscan.core import Application
from holoscan.operators import PingRxOp, PingTxOp


class MyPingApp(Application):
    def __init__(self, *args, count=10, ports_arg, **kwargs):
        self.count = count
        self.ports_arg = ports_arg
        super().__init__(*args, **kwargs)

    def compose(self):
        tx = PingTxOp(self, CountCondition(self, self.count), name="tx")
        rx = PingRxOp(self, name="rx")

        if self.ports_arg:
            self.add_flow(tx, rx, self.ports_arg)
        else:
            self.add_flow(tx, rx)


@pytest.mark.parametrize(
    "ports_arg, invalid_port",
    [
        # valid names
        ({("out", "in")}, None),
        # rely on auto-determination for single port input/output
        (False, None),
        # non-existent port name
        ({("output", "in")}, "upstream"),
        ({("out", "input")}, "downstream"),
        ({("output", "input")}, "both"),
    ],
)
def test_ping_app_invalid_add_flow(ports_arg, invalid_port, capfd):
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
        err_msg_upstream = "does not have an output port with label"
        err_msg_downstream = "does not have an input port with label"
        if invalid_port in ["upstream"]:
            assert err_msg_upstream in captured.err
        elif invalid_port in ["downstream"]:
            assert err_msg_downstream in captured.err
        elif invalid_port in ["both"]:
            # In pracitce, the app terminates after the upstream error is printed, but I didn't
            # want to guarantee which error would be printed first.
            assert err_msg_upstream in captured.err or err_msg_downstream in captured.err
