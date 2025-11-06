"""
SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os  # noqa: I001

from holoscan.core import Application
from holoscan.operators import FormatConverterOp, HolovizOp, V4L2VideoCaptureOp
from holoscan.resources import RMMAllocator

from holoscan.operators.v4l2_camera_passthrough import V4L2CameraPassthroughOp


class App(Application):
    """Example of an application that uses conditional routing for video format conversion.

    This application has the following operators:

    - V4L2VideoCaptureOp
    - FormatConverterOp
    - HolovizOp

    The V4L2VideoCaptureOp captures video streams. Based on the pixel format metadata,
    the application conditionally routes frames:
    - If YUYV format: source -> format_converter -> visualizer
    - Otherwise: source -> visualizer (direct)
    """

    def compose(self):
        source_args = self.kwargs("source")
        source = V4L2VideoCaptureOp(
            self,
            name="source",
            pass_through=True,
            **source_args,
        )

        # Create format converter for YUYV to RGBA conversion
        format_converter = FormatConverterOp(
            self,
            name="format_converter",
            pool=RMMAllocator(self, name="rmm-allocator", **self.kwargs("rmm_allocator")),
            in_dtype="yuyv",
            out_dtype="rgb888",
        )

        viz_args = self.kwargs("visualizer")
        if "width" in source_args and "height" in source_args:
            # Set Holoviz width and height from source resolution
            viz_args["width"] = source_args["width"]
            viz_args["height"] = source_args["height"]

        visualizer = HolovizOp(
            self,
            name="visualizer",
            **viz_args,
        )

        # Use a C++ passthrough operator to work around two issues:
        # 1. Connecting multiple operators to HolovizOp "receivers" would create multiple
        #    input ports with MessageAvailableConditions, resulting in a deadlock
        # 2. Pure Python operators cannot forward GXF::VideoBuffer objects
        passthrough = V4L2CameraPassthroughOp(self, name="passthrough")

        self.add_flow(passthrough, visualizer, {("output", "receivers")})

        # 1. Flow for YUYV format, not supported directly by Holoviz in display drivers >=R550
        # source -> format_converter -> passthrough -> visualizer
        self.add_flow(source, format_converter, {("signal", "source_video")})
        self.add_flow(format_converter, passthrough, {("tensor", "input")})

        # 2. Flow for other VideoBuffer formats (NV12, RGB24, etc.) directly compatible with Holoviz
        # source -> passthrough -> visualizer
        self.add_flow(source, passthrough, {("signal", "input")})

        def dynamic_flow_callback(op):
            """Route based on V4L2 pixel format metadata.

            YUYV is not supported directly by Holoviz in display drivers >=R550.
            """
            pixel_format = op.metadata.get("V4L2_pixel_format", "")

            if "YUYV" in pixel_format.upper():
                op.add_dynamic_flow("signal", format_converter, "source_video")
            else:
                op.add_dynamic_flow("signal", passthrough, "input")

        # Set up conditional routing on source operator
        self.set_dynamic_flows(source, dynamic_flow_callback)


def main(config_file):
    app = App()
    # if the --config command line argument was provided, it will override this config_file
    app.config(config_file)
    app.run()
    print("Application has finished running.")


if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(__file__), "v4l2_camera.yaml")
    main(config_file=config_file)
