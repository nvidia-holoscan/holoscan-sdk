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

import os

from holoscan.core import Application, Operator, OperatorSpec
from holoscan.core._core import Tensor as TensorBase
from holoscan.operators import FormatConverterOp, HolovizOp, V4L2VideoCaptureOp
from holoscan.resources import UnboundedAllocator


# This operator uses the metadata provided by the V4L2VideoCaptureOp and translates it to a
# HolovizOp::InputSpec so HolovizOp can display the video data. It also sets the YCbCr encoding
# model and quantization range.
# For V4L2 pixel formats that have a equivalent nvidia::gxf::VideoFormat enum this is not required
# since that information is then part of the video buffer send by V4L2VideoCaptureOp.
class V4L2FormatTranslateOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("input")
        spec.output("output_specs")

    def compute(self, op_input, op_output, context):
        if not self.is_metadata_enabled:
            raise RuntimeError("Metadata needs to be enabled for this operator")

        # we don't need the image data, just the metadata
        entities = op_input.receive("input")

        if len(entities) != 0 and isinstance(entities[""], TensorBase):
            # use the metadata provided by the V4L2VideoCaptureOp to build the input spec for the
            # HolovizOp
            specs = []
            spec = HolovizOp.InputSpec("", "color")
            specs.append(spec)

            v4l2_pixel_format = self.metadata["V4L2_pixel_format"]
            if v4l2_pixel_format == "YUYV":
                spec.image_format = HolovizOp.ImageFormat.Y8U8Y8V8_422_UNORM

                # also set the encoding and quantization
                if self.metadata["V4L2_ycbcr_encoding"] == "V4L2_YCBCR_ENC_601":
                    spec.yuv_model_conversion = HolovizOp.YuvModelConversion.YUV_601
                elif self.metadata["V4L2_ycbcr_encoding"] == "V4L2_YCBCR_ENC_709":
                    spec.yuv_model_conversion = HolovizOp.YuvModelConversion.YUV_709
                elif self.metadata["V4L2_ycbcr_encoding"] == "V4L2_YCBCR_ENC_2020":
                    spec.yuv_model_conversion = HolovizOp.YuvModelConversion.YUV_2020

                if self.metadata["V4L2_quantization"] == "V4L2_QUANTIZATION_FULL_RANGE":
                    spec.yuv_range = HolovizOp.YuvRange.ITU_FULL
                elif self.metadata["V4L2_quantization"] == "V4L2_QUANTIZATION_LIM_RANGE":
                    spec.yuv_range = HolovizOp.YuvRange.ITU_NARROW
            else:
                raise RuntimeError(f"Unhandled V4L2 pixel format {v4l2_pixel_format}")

            # don't pass the meta data along to avoid errors when MetadataPolicy is `kRaise`
            self.metadata.clear()

            # emit the output specs
            op_output.emit(specs, "output_specs")


# Now define a simple application using the operators defined above
class App(Application):
    """Example of an application that uses the operators defined above.

    This application has the following operators:

    - V4L2VideoCaptureOp
    - HolovizOp

    The V4L2VideoCaptureOp captures a video streams and visualizes it using HolovizOp.
    """

    def compose(self):
        source_args = self.kwargs("source")
        source = V4L2VideoCaptureOp(
            self,
            name="source",
            pass_through=True,
            **source_args,
        )

        viz_args = self.kwargs("visualizer")
        if "width" in source_args and "height" in source_args:
            # Set Holoviz width and height from source resolution
            viz_args["width"] = source_args["width"]
            viz_args["height"] = source_args["height"]

        allocator = UnboundedAllocator(self, name="allocator")
        format_converter = FormatConverterOp(
            self, name="format_converter", pool=allocator, **self.kwargs("format_converter")
        )

        visualizer = HolovizOp(
            self,
            name="visualizer",
            **viz_args,
        )

        self.add_flow(source, format_converter, {("signal", "source_video")})
        self.add_flow(format_converter, visualizer, {("", "receivers")})

        # enable metadata so V4L2FormatTranslateOp can translate the format

        # As of Holoscan 3.0, metadata is enabled by default at the Fragment
        # level. If we wanted to override that default for some operator, we
        # would call ``self.enable_metadata(False)``.


def main(config_file):
    app = App()
    # if the --config command line argument was provided, it will override this config_file
    app.config(config_file)
    app.run()
    print("Application has finished running.")


if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(__file__), "v4l2_camera.yaml")
    main(config_file=config_file)
