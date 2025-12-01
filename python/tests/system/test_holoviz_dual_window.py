"""
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math

import cupy as cp
import numpy as np

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import HolovizOp


def get_frame(xp, height, width, channels, dtype=np.uint8):
    shape = (height, width, channels) if channels else (height, width)
    size = math.prod(shape)
    frame = xp.arange(size, dtype=dtype).reshape(shape)
    return frame


class FrameGeneratorOp(Operator):
    def __init__(
        self,
        fragment,
        *args,
        width=800,
        height=640,
        channels=3,
        on_host=False,
        dtype=np.uint8,
        **kwargs,
    ):
        self.height = height
        self.width = width
        self.channels = channels
        self.on_host = on_host
        self.dtype = dtype
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("frame")

    def compute(self, op_input, op_output, context):
        xp = np if self.on_host else cp
        frame = get_frame(xp, self.height, self.width, self.channels, self.dtype)
        print(f"Emitting frame with shape: {frame.shape}")
        op_output.emit(dict(frame=frame), "frame")


class HolovizHeadlessApp(Application):
    def __init__(
        self,
        *args,
        count=10,
        width=800,
        height=640,
        on_host=False,
        **kwargs,
    ):
        self.count = count
        self.width = width
        self.height = height
        self.on_host = on_host
        super().__init__(*args, **kwargs)

    def on_window_closed(self):
        """Application-level callback for Holoviz window close events."""
        self.stop_execution()

    def compose(self):
        source = FrameGeneratorOp(
            self,
            CountCondition(self, count=self.count),
            width=self.width,
            height=self.height,
            on_host=self.on_host,
            dtype=np.uint8,
            name="video_source",
        )

        common_visualizer_kwargs = dict(
            headless=True,
            width=self.width,
            height=self.height,
            window_close_callback=self.on_window_closed,
            tensors=[
                # name="" here to match the output of FrameGenerationOp
                dict(name="frame", type="color", opacity=0.5, priority=0),
            ],
        )

        vizualizer = HolovizOp(self, **common_visualizer_kwargs, name="visualizer")
        vizualizer2 = HolovizOp(self, **common_visualizer_kwargs, name="visualizer2")
        visualizers = [vizualizer, vizualizer2]
        for viz in visualizers:
            self.add_flow(source, viz, {("frame", "receivers")})


def test_holovizop_dual_window(capfd):
    """Test HolovizOp with dual windows and application-level window_close_callback."""
    count = 3
    width = 800
    height = 640
    holoviz_app = HolovizHeadlessApp(
        count=count,
        width=width,
        height=height,
        on_host=False,
    )

    holoviz_app.run()

    captured = capfd.readouterr()

    # assert that replayer_app emitted all frames
    assert captured.out.count("Emitting frame") == count
    # no warning about the deprecated parameter name is shown
    assert "window_close_scheduling_term" not in captured.out
