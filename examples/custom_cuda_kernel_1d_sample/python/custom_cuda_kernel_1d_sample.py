"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from holoscan.core import Application
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    InferenceProcessorOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import UnboundedAllocator

sample_data_path = os.environ.get("HOLOSCAN_INPUT_PATH", "../data")


class CustomCUDAKernel1DApp(Application):
    """Example of an application that uses the operators defined above.

    This application has the following operators:

    - VideoStreamReplayerOp
    - FormatConverterOp
    - InferenceProcessorOp
    - HolovizOp

    The VideoStreamReplayerOp reads a video file and sends the frames to the FormatConverterOp.
    FormatConverterOp after modifying the input frame, sends it to the InferenceProcessorOp and
    the HolovizOp.
    InferenceProcessorOp applies custom CUDA kernels to the input frame and sends the results
    to HoloVizOp.
    The HolovizOp displays the frames.
    """

    def compose(self):
        video_dir = os.path.join(sample_data_path, "racerx")
        if not os.path.exists(video_dir):
            raise ValueError(f"Could not find video data: {video_dir=}")

        pool = UnboundedAllocator(self, name="pool")

        in_dtype = "rgb888"

        replayer = VideoStreamReplayerOp(
            self,
            name="replayer",
            directory=video_dir,
            **self.kwargs("replayer"),
            allocator=pool,
        )

        formatconverter = FormatConverterOp(
            self,
            name="format_converter",
            pool=pool,
            in_dtype=in_dtype,
            **self.kwargs("format_converter"),
        )

        processor = InferenceProcessorOp(
            self,
            name="processor",
            allocator=pool,
            **self.kwargs("processor"),
        )

        specs = []

        spec = HolovizOp.InputSpec("input_formatted", HolovizOp.InputType.COLOR)
        view = HolovizOp.InputSpec.View()
        view.width = 0.5
        view.height = 1.0
        spec.views = [view]
        specs.append(spec)

        spec = HolovizOp.InputSpec("input_processed", HolovizOp.InputType.COLOR)
        view = HolovizOp.InputSpec.View()
        view.offset_x = 0.5
        view.width = 0.5
        view.height = 1.0
        spec.views = [view]
        specs.append(spec)

        visualizer = HolovizOp(self, name="holoviz", tensors=specs, **self.kwargs("holoviz"))

        # Define the workflow
        self.add_flow(replayer, formatconverter, {("output", "")})
        self.add_flow(formatconverter, visualizer, {("", "receivers")})
        self.add_flow(formatconverter, processor, {("", "receivers")})
        self.add_flow(processor, visualizer, {("transmitter", "receivers")})


def main(config_file):
    app = CustomCUDAKernel1DApp()
    # if the --config command line argument was provided, it will override this config_file
    app.config(config_file)
    app.run()


if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(__file__), "custom_cuda_kernel_1d_sample.yaml")
    main(config_file=config_file)
