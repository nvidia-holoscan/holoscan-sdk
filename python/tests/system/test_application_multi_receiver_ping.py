"""
 SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import sys

import pytest

from holoscan.conditions import CountCondition
from holoscan.core import Application, IOSpec, Operator, OperatorSpec


class PingTxOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.index = 1
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        op_output.emit(self.index, "out")
        self.index += 1


class PingRxOp(Operator):
    def __init__(self, fragment, *args, params=None, **kwargs):
        self.params = params
        self.count = 1
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("receivers", size=self.params["input_size"])

    def compute(self, op_input, op_output, context):
        values = op_input.receive("receivers", **self.params["input_kwargs"])
        if isinstance(values, tuple):
            print(
                f"Rx message received (count: {self.count}, size: {len(values)})", file=sys.stderr
            )
            self.count += 1
            print(f"Rx message values: {values}", file=sys.stderr)
        else:
            while values is not None:
                print(f"Rx message received (count: {self.count})", file=sys.stderr)
                self.count += 1
                print(f"Rx message value: {values}", file=sys.stderr)
                values = op_input.receive("receivers", **self.params["input_kwargs"])


class MyPingApp(Application):
    def __init__(self, *args, count=10, params=None, **kwargs):
        self.count = count
        self.params = params
        super().__init__(*args, **kwargs)

    def compose(self):
        tx = PingTxOp(
            self,
            CountCondition(self, self.count),
            name="tx",
        )
        rx = PingRxOp(self, params=self.params, name="rx")
        # test with default input/output port names
        self.add_flow(tx, rx)


@pytest.mark.parametrize(
    "params",
    [
        dict(
            input_size=IOSpec.ANY_SIZE,
            input_kwargs={},
            expected_string="Rx message values: (10,)",
        ),
        dict(
            input_size=IOSpec.ANY_SIZE,
            input_kwargs=dict(kind="single"),
            expected_string="Invalid kind 'single' for receive() method, cannot be 'single' for the input port with 'IOSpec.ANY_SIZE'",  # noqa: E501
            expect_error=True,
        ),
        dict(
            input_size=IOSpec.PRECEDING_COUNT,
            input_kwargs={},
            expected_string="Rx message values: (10,)",
        ),
        dict(
            input_size=IOSpec.PRECEDING_COUNT,
            input_kwargs=dict(kind="single"),
            expected_string="Rx message value: 10",
        ),
        dict(
            input_size=2,
            input_kwargs={},
            expected_string="Rx message values: (9, 10)",
        ),
        dict(
            input_size=IOSpec.IOSize(4),
            input_kwargs=dict(kind="single"),
            expected_string="Rx message value: 8",
        ),
        dict(
            input_size=1,
            input_kwargs=dict(kind="multi"),
            expected_string="Rx message values: (10,)",
        ),
        dict(
            input_size=-3,
            input_kwargs={},
            expected_string="Invalid queue size: -3 (op: 'rx', input port: 'receivers')",
            expect_error=True,
        ),
    ],
)
def test_my_ping_app(ping_config_file, params, capfd):
    count = 10
    app = MyPingApp(count=count, params=params)
    app.config(ping_config_file)

    if params.get("expect_error"):
        with pytest.raises(RuntimeError):
            app.run()
    else:
        app.run()

    # assert that the expected number of messages were received
    captured = capfd.readouterr()
    assert f"{params['expected_string']}" in captured.err
