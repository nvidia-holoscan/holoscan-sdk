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

import sys

import cupy as cp
import pytest

import holoscan
from holoscan.conditions import CountCondition
from holoscan.core import kwargs_to_arglist, py_object_to_arg
from holoscan.operators import FormatConverterOp
from holoscan.resources import UnboundedAllocator

NUM_MSGS = 10


class ImageGenerator(holoscan.core.Operator):
    def setup(self, spec):
        spec.output("image")

    def generate_random_image(self, width, height):
        image = cp.random.randint(0, 256, (height, width, 3), dtype=cp.uint8)
        return image

    def compute(self, op_input, op_output, context):
        width = 854
        height = 480
        img = self.generate_random_image(width, height)
        op_output.emit({"": img}, "image")


class WatchdogOperator(holoscan.core.Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.index = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        print("Watchdog started", file=sys.stderr)
        tensor = op_input.receive("input")[""]
        self.index += 1
        print(f"Watchdog received data (count: {self.index})", file=sys.stderr)
        if self.index == 1:
            print(f"{tensor.shape = }", file=sys.stderr)
        print("Watchdog finished", file=sys.stderr)


class TensorSenderAppAddArg(holoscan.core.Application):
    def __init__(self, *args, add_arg_mode="arg", **kwargs):
        self.add_arg_mode = add_arg_mode
        super().__init__(*args, **kwargs)

    def compose(self):
        image_generator = ImageGenerator(
            self,
            CountCondition(self, NUM_MSGS),
            name="image_generator",
        )

        host_allocator = UnboundedAllocator(self, name="host_allocator")

        if self.add_arg_mode == "init_kwarg":
            format_conv_kwargs = dict(resize_width=400, resize_height=200)
        else:
            format_conv_kwargs = {}

        preprocessor = FormatConverterOp(
            self, name="preprocessor", pool=host_allocator, out_dtype="rgb888", **format_conv_kwargs
        )
        # calling add_arg later results in values that override any constructor kwarg defaults
        if self.add_arg_mode == "arg":
            # add as individual Args
            preprocessor.add_arg(py_object_to_arg(400, name="resize_width"))
            preprocessor.add_arg(py_object_to_arg(200, name="resize_height"))
        elif self.add_arg_mode == "arglist":
            # add Arglist
            arglist = kwargs_to_arglist(resize_width=400, resize_height=200)
            preprocessor.add_arg(arglist)
        elif self.add_arg_mode == "kwargs":
            # add via Python kwargs
            preprocessor.add_arg(resize_width=400, resize_height=200)

        watchdog = WatchdogOperator(self, name="watchdog")

        # Define the workflow
        self.add_flow(image_generator, preprocessor, {("image", "source_video")})
        self.add_flow(preprocessor, watchdog, {("tensor", "input")})


@pytest.mark.parametrize("add_arg_mode", ["init_kwarg", "arg", "arglist", "kwargs"])
def test_operator_add_arg_app(add_arg_mode, capfd):
    global NUM_MSGS

    app = TensorSenderAppAddArg(add_arg_mode=add_arg_mode)

    app.run()

    # assert that no errors were logged
    captured = capfd.readouterr()

    assert "error" not in captured.err
    assert "Exception occurred" not in captured.err

    # assert that the expected number of messages were received
    assert f"Watchdog received data (count: {NUM_MSGS})" in captured.err
    assert "tensor.shape = (200, 400, 3)" in captured.err
