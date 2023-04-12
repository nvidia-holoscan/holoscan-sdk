"""
SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""  # no qa

import os
import sys

import holoscan as hs
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.gxf import Entity
from holoscan.logger import load_env_log_level
from holoscan.operators import HolovizOp, VideoStreamReplayerOp

try:
    import cupy as cp
    import cupyx.scipy.ndimage as ndi
except ImportError:
    raise ImportError(
        "CuPy must be installed to run this example. See "
        "https://docs.cupy.dev/en/stable/install.html"
    )

sample_data_path = os.environ.get("HOLOSCAN_SAMPLE_DATA_PATH", "../data")


# Define custom Operators for use in the demo
class ImageProcessingOp(Operator):
    """Example of an operator processing input video (as a tensor).

    This operator has:
        inputs:  "input_tensor"
        outputs: "output_tensor"

    The data from each input is processed by a CuPy gaussian filter and
    the result is sent to the output.

    In this demo, the input and output image (2D RGB) is an 3D array of shape
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
        message = op_input.receive("input_tensor")

        input_tensor = message.get()

        print(f"message received (count: {self.count})")
        self.count += 1

        cp_array = cp.asarray(input_tensor)

        # smooth along first two axes, but not the color channels
        sigma = (self.sigma, self.sigma, 0)

        # process cp_array
        cp_array = ndi.gaussian_filter(cp_array, sigma)

        out_message = Entity(context)
        output_tensor = hs.as_tensor(cp_array)

        out_message.add(output_tensor)
        op_output.emit(out_message, "output_tensor")


# Now define a simple application using the operators defined above
class MyVideoProcessingApp(Application):
    """Example of an application that uses the operators defined above.

    This application has the following operators:

    - VideoStreamReplayerOp
    - ImageProcessingOp
    - HolovizOp

    The VideoStreamReplayerOp reads a video file and sends the frames to the ImageProcessingOp.
    The ImageProcessingOp processes the frames and sends the processed frames to the HolovizOp.
    The HolovizOp displays the processed frames.
    """

    def compose(self):
        width = 854
        height = 480
        video_dir = os.path.join(sample_data_path, "endoscopy", "video")
        if not os.path.exists(video_dir):
            raise ValueError(f"Could not find video data: {video_dir=}")
        source = VideoStreamReplayerOp(
            self,
            name="replayer",
            directory=video_dir,
            **self.kwargs("replayer"),
        )

        image_processing = ImageProcessingOp(
            self, name="image_processing", **self.kwargs("image_processing")
        )

        visualizer = HolovizOp(
            self,
            name="holoviz",
            width=width,
            height=height,
            **self.kwargs("holoviz"),
        )

        self.add_flow(source, image_processing)
        self.add_flow(image_processing, visualizer, {("", "receivers")})


if __name__ == "__main__":
    load_env_log_level()

    config_file = os.path.join(os.path.dirname(__file__), "tensor_interop.yaml")

    if len(sys.argv) >= 2:
        config_file = sys.argv[1]

    app = MyVideoProcessingApp()
    app.config(config_file)
    app.run()
