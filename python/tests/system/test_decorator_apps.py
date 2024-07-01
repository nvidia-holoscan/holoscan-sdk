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
import numpy as np

from holoscan.conditions import CountCondition
from holoscan.core import Application
from holoscan.core._core import Tensor as TensorBase
from holoscan.decorator import Input, Output, create_op


@create_op(outputs="out")
def source_generator(count):
    yield from range(count)


@create_op
class Forward:
    def __init__(self, init_index=0):
        self.index = init_index

    def __call__(self, x):
        print(f"forward called - index: {self.index}")
        self.index += 1
        if x is None:
            return None
        else:
            return x


# verify we can call create_op with no arguments for a simple function like this
@create_op
def sink(y):
    print("sink called", y)


class SimpleGeneratorApp(Application):
    def __init__(
        self,
        *args,
        iterations: int = 10,
        generator_count: int = 5,
        start_count: int = 3,
        **kwargs,
    ):
        self.iterations = iterations
        self.generator_count = generator_count
        self.start_count = start_count
        super().__init__(*args, **kwargs)

    def compose(self):
        src_op = source_generator(
            self, CountCondition(self, self.iterations), name="src", count=self.generator_count
        )
        forward_op = Forward(self.start_count)(self, name="forwarding")
        sink_op = sink(self, name="sink")

        self.add_flow(src_op, forward_op)
        self.add_flow(forward_op, sink_op)


def test_generator_and_class_decorator_app(capfd):
    iterations = 10
    generator_count = 5
    start_count = 3
    app = SimpleGeneratorApp(
        iterations=iterations, generator_count=generator_count, start_count=start_count
    )
    app.run()

    captured = capfd.readouterr()
    assert "error" not in captured.out
    assert f"forward called - index: {start_count - 1}" not in captured.out
    assert f"forward called - index: {start_count}" in captured.out
    assert f"forward called - index: {start_count + generator_count - 1}" in captured.out
    assert f"forward called - index: {start_count + generator_count}" not in captured.out


@create_op(
    outputs=(
        Output("x", tensor_names=("x",)),
        Output("waveform", tensor_names=("waveform",)),
    ),
)
def dual_tensor_generate(x_shape=(128, 64), waveform_shape=(512,)):
    x = np.zeros(x_shape, dtype=np.float32)
    waveform = np.zeros(waveform_shape, dtype=np.float32)
    return dict(x=x, waveform=waveform)


@create_op(inputs=Input("in", arg_map="tensor"), outputs="y")
def increment(tensor, value=1.5):
    return tensor + value


@create_op(inputs="tensor")
def print_shape(tensor, *, expected_value=None):
    print(f"{tensor.shape = }")

    if expected_value is not None:
        assert tensor.ravel()[0] == expected_value


class MultipleOutputTensorApp(Application):
    def compose(self):
        src_op = dual_tensor_generate(
            self, CountCondition(self, 3), name="dual_gen", x_shape=(32, 16)
        )
        increment_op = increment(self, name="increment_x", value=2.0)
        assert increment_op.fixed_kwargs["value"] == 2.0
        print_op = print_shape(self, name="print_x", expected_value=2.0)
        assert print_op.fixed_kwargs["expected_value"] == 2.0

        increment_op2 = increment(self, name="increment_waveform")
        assert increment_op2.fixed_kwargs["value"] == 1.5
        print_op2 = print_shape(self, name="print_waveform", expected_value=1.5)
        assert print_op2.fixed_kwargs["expected_value"] == 1.5

        self.add_flow(src_op, increment_op, {("x", "in")})
        self.add_flow(increment_op, print_op)

        self.add_flow(src_op, increment_op2, {("waveform", "in")})
        self.add_flow(increment_op2, print_op2)


def test_multiple_output_tensor_app(capfd):
    app = MultipleOutputTensorApp()
    app.run()

    captured = capfd.readouterr()
    assert "error" not in captured.out
    assert captured.out.count("tensor.shape = (32, 16)") == 3
    assert captured.out.count("tensor.shape = (512,)") == 3


@create_op(inputs="tensor", cast_tensors=False)
def check_tensor_no_cast(tensor):
    assert isinstance(tensor, TensorBase)
    print("holoscan.Tensor check succeeded")


@create_op(inputs="tensor", cast_tensors=True)
def check_tensor_cast(tensor):
    assert isinstance(tensor, np.ndarray)
    print("cupy.ndarray check succeeded")


class TensorCastCheckApp(Application):
    def compose(self):
        src_op = dual_tensor_generate(
            self, CountCondition(self, 2), name="dual_gen", x_shape=(32, 16)
        )
        no_cast_check_op = check_tensor_no_cast(self, name="no_cast_check")
        cast_check_op = check_tensor_cast(self, name="cast_check")
        self.add_flow(src_op, no_cast_check_op, {("x", "tensor")})
        self.add_flow(src_op, cast_check_op, {("waveform", "tensor")})


def test_tensor_input_cast_app(capfd):
    app = TensorCastCheckApp()
    app.run()

    captured = capfd.readouterr()
    assert "error" not in captured.out
    assert captured.out.count("holoscan.Tensor check succeeded") == 2
    assert captured.out.count("cupy.ndarray check succeeded") == 2
