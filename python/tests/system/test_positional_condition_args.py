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
import time

import cupy as cp

from holoscan.conditions import CountCondition, PeriodicCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import FormatConverterOp, PingRxOp
from holoscan.resources import UnboundedAllocator


class TxRGBA(Operator):
    """Transmit an RGBA device tensor of user-specified shape.

    The tensor must be on device for use with FormatConverterOp.
    """

    def __init__(self, fragment, *args, shape=(32, 64), **kwargs):
        self.shape = shape
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("rgba_out")

    def compute(self, op_input, op_output, context):
        img = cp.zeros(self.shape + (4,), dtype=cp.uint8)
        op_output.emit(dict(rgba=img), "rgba_out")


class MyFormatConverterApp(Application):
    """Test passing conditions positionally to a wrapped C++ operator (FormatConverterOp)."""

    def __init__(self, *args, count=10, period=None, explicitly_set_connectors=False, **kwargs):
        self.count = count
        self.period = period
        self.explicitly_set_connectors = explicitly_set_connectors
        super().__init__(*args, **kwargs)

    def compose(self):
        tx = TxRGBA(self, shape=(32, 16), name="tx")
        converter = FormatConverterOp(
            self,
            CountCondition(self, self.count),
            PeriodicCondition(self, recess_period=self.period),
            pool=UnboundedAllocator(self),
            in_tensor_name="rgba",
            in_dtype="rgba8888",
            out_dtype="rgb888",
        )
        rx = PingRxOp(self, name="rx")
        self.add_flow(tx, converter)
        self.add_flow(converter, rx)


def test_format_converter_app(capfd):
    count = 10
    period = 50_000_000  # 50 ms per frame
    app = MyFormatConverterApp(count=count, period=period)

    tstart = time.time()
    app.run()
    duration = time.time() - tstart

    # assert that the expected number of messages were received
    captured = capfd.readouterr()

    # verify that only the expected number of messages were received
    assert captured.out.count("Rx message value") == count

    # verify that run time was not faster than the specified period
    min_duration_seconds = (count - 1) * (period / 1.0e9)
    assert duration > min_duration_seconds
