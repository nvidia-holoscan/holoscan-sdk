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
import cupy as cp
import numpy as np
import pytest

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import FormatConverterOp, PingRxOp
from holoscan.resources import UnboundedAllocator


class TxRGBA(Operator):
    """Transmit an RGBA device tensor of user-specified shape.

    The tensor must be on device for use with FormatConverterOp.
    """

    def __init__(
        self,
        fragment,
        *args,
        shape=(32, 64),
        channels=4,
        fortran_ordered=False,
        padded=False,
        memory_location=False,
        **kwargs,
    ):
        self.shape = shape
        self.channels = channels
        self.fortran_ordered = fortran_ordered
        self.memory_location = memory_location
        self.padded = padded
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        shape = self.shape if self.channels is None else self.shape + (self.channels,)
        order = "F" if self.fortran_ordered else "C"
        if self.memory_location == "device":
            img = cp.zeros(shape, dtype=cp.uint8, order=order)
        elif self.memory_location == "cpu":
            img = np.zeros(shape, dtype=cp.uint8, order=order)
        if self.padded:
            # slice so reduce shape[1] without changing stride[0]
            # This matches a common case where padding is added to make CUDA kernels more efficient
            img = img[:, :-2, ...]
        op_output.emit(dict(frame=img), "out")


class MyFormatConverterApp(Application):
    """Test passing conditions positionally to a wrapped C++ operator (FormatConverterOp)."""

    def __init__(
        self,
        *args,
        count=10,
        shape=(32, 16),
        channels=4,
        padded=False,
        input_memory_location="device",
        in_dtype="rgba8888",
        out_dtype="rgb888",
        fortran_ordered=False,
        **kwargs,
    ):
        self.count = count
        self.shape = shape
        self.channels = channels
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.input_memory_location = input_memory_location
        self.padded = padded
        self.fortran_ordered = fortran_ordered
        super().__init__(*args, **kwargs)

    def compose(self):
        tx = TxRGBA(
            self,
            shape=self.shape,
            channels=self.channels,
            fortran_ordered=self.fortran_ordered,
            padded=self.padded,
            memory_location=self.input_memory_location,
            name="tx",
        )
        converter = FormatConverterOp(
            self,
            CountCondition(self, self.count),
            pool=UnboundedAllocator(self),
            in_tensor_name="frame",
            in_dtype=self.in_dtype,
            out_dtype=self.out_dtype,
        )
        rx = PingRxOp(self, name="rx")
        self.add_flow(tx, converter)
        self.add_flow(converter, rx)


@pytest.mark.parametrize("padded", [False, True])
def test_format_converter_invalid_memory_layout(padded, capfd):
    count = 10
    app = MyFormatConverterApp(
        count=count, channels=4, padded=padded, fortran_ordered=True, input_memory_location="device"
    )

    with pytest.raises(RuntimeError):
        app.run()
    captured = capfd.readouterr()
    assert "error" in captured.err
    assert "Tensor is expected to be in a C-contiguous memory layout" in captured.err


@pytest.mark.parametrize("shape", [(4,), (4, 4, 4, 4)])
def test_format_converter_invalid_rank(shape, capfd):
    count = 10
    app = MyFormatConverterApp(
        count=count,
        shape=shape,
        channels=None,
        fortran_ordered=False,
        input_memory_location="device",
    )

    with pytest.raises(RuntimeError):
        app.run()
    captured = capfd.readouterr()
    assert "error" in captured.err
    assert "Expected a tensor with 2 or 3 dimensions" in captured.err


@pytest.mark.parametrize("channels", [1, 2, 3, 4, 5])
def test_format_converter_invalid_num_channels(channels, capfd):
    count = 10
    # For in_dtype rgba8888, must have 4 channels
    app = MyFormatConverterApp(
        count=count, channels=channels, fortran_ordered=False, input_memory_location="device"
    )

    if channels == 4:
        app.run()
        captured = capfd.readouterr()
        assert "error" not in captured.err
    else:
        with pytest.raises(RuntimeError):
            app.run()
        captured = capfd.readouterr()
        assert "error" in captured.err
        assert "Failed to verify the channels for the expected input dtype" in captured.err


def test_format_converter_host_tensor(capfd):
    count = 10
    # For in_dtype rgba8888, must have 4 channels
    app = MyFormatConverterApp(
        count=count, channels=4, fortran_ordered=False, input_memory_location="cpu"
    )

    app.run()
    captured = capfd.readouterr()
    assert "error" not in captured.err
