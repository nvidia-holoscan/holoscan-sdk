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
from holoscan.core import Application, Operator, OperatorSpec, Tracker
from holoscan.operators import PingRxOp

cp = pytest.importorskip("cupy")
ndi = pytest.importorskip("cupyx.scipy.ndimage")


class RandomTensorMapTxOp(Operator):
    """Simple transmitter operator.

    This operator has a single output port:
        output: "out"

    On each tick, it transmits random RGB images to the "out" port.

    A tensormap (dict) is used in order to test issue 4196152
    """

    def __init__(self, fragment, *args, **kwargs):
        self.index = 1
        self.rng = cp.random.default_rng()
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        # use dict here so we test the fix for issue 4196152
        tensormap = dict(frame=self.rng.integers(0, 256, (256, 256, 3), dtype=cp.uint8))
        op_output.emit(tensormap, "out")
        self.index += 1


class ImageProcessingOp(Operator):
    """Example of an operator processing input video (as a tensor).

    This operator has:
        inputs:  "input_tensor"
        outputs: "output_tensor"

    The data from each input is processed by a CuPy gaussian filter and
    the result is sent to the output.

    In this demo, the input and output image (2D RGB) is a 3D array of shape
    (height, width, channels).
    """

    def __init__(self, fragment, *args, **kwargs):
        self.count = 1

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input_tensor")
        spec.output("output_tensor")
        spec.param("sigma")

    def compute(self, op_input, op_output, context):
        # in_message is a dict of tensors
        in_message = op_input.receive("input_tensor")

        # smooth along first two axes, but not the color channels
        sigma = (self.sigma, self.sigma, 0)

        # out_message will be a dict of tensors
        out_message = dict()

        for key, value in in_message.items():
            print(f"message received (count: {self.count})")
            self.count += 1

            cp_array = cp.asarray(value)

            # process cp_array
            cp_array = ndi.gaussian_filter(cp_array, sigma)

            out_message[key] = cp_array

        op_output.emit(out_message, "output_tensor")


# Now define a simple application using the operators defined above
class MyVideoProcessingApp(Application):
    """Example of an application that uses the operators defined above.

    This application has the following operators:

    - RandomTensorMapTxOp
    - ImageProcessingOp
    - PingRxOp

    The RandomTensorMapTxOp generates a random Tensor and sends it to the ImageProcessingOp.
    The ImageProcessingOp processes the frames and sends the processed frames to the PingRxOp.
    """

    def compose(self):
        source = RandomTensorMapTxOp(self, CountCondition(self, count=10), name="tensor_generator")
        image_processing = ImageProcessingOp(self, name="image_processing", sigma=3.0)
        rx1 = PingRxOp(self, name="rx1")
        self.add_flow(source, image_processing)
        self.add_flow(image_processing, rx1)


@pytest.mark.parametrize("track", [False, True])
def test_tensormap_flow_tracking(track, capfd):
    # Application designed to test issue 4196152
    # (a simplification of tensor_interop.py to not use visualization or require video data)
    app = MyVideoProcessingApp()
    if track:
        with Tracker(app) as tracker:
            app.run()
            tracker.print()
    else:
        app.run()

    captured = capfd.readouterr()
    # count should stop at 10 as specified in CountCondition
    assert captured.out.count("message received (count: 10)") == 1
    assert captured.out.count("message received (count: 11)") == 0
    # flow tracking result should be shown when track is True
    num_tracking_results = 1 if track else 0
    assert captured.out.count("Data Flow Tracking Results:") == num_tracking_results


if __name__ == "__main__":
    app = MyVideoProcessingApp()
    with Tracker(app) as tracker:
        app.run()
        tracker.print()
