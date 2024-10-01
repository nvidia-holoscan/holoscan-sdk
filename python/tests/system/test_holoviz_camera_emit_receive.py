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

import math

import cupy as cp
import numpy as np
import pytest

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import HolovizOp, PingRxOp
from holoscan.operators.holoviz import Pose3D


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
        spec.output("camera_eye")
        spec.output("camera_look_at")
        spec.output("camera_up")

    def compute(self, op_input, op_output, context):
        xp = np if self.on_host else cp
        frame = get_frame(xp, self.height, self.width, self.channels, self.dtype)
        print(f"Emitting frame with shape: {frame.shape}")
        op_output.emit(dict(frame=frame), "frame")
        # emit camera pose vectors
        op_output.emit([0.0, 0.0, 1.0], "camera_eye", "std::array<float, 3>")
        op_output.emit([0.0, 0.0, 0.0], "camera_look_at", "std::array<float, 3>")
        op_output.emit([0.0, 1.0, 0.0], "camera_up", "std::array<float, 3>")


class CameraPoseForwardingOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("camera_pose_input")
        spec.input("camera_up_input")
        spec.output("camera_pose_output")

    def compute(self, op_input, op_output, context):
        # verify that values match output of FrameGeneratorOp
        camera_up = op_input.receive("camera_up_input")
        assert isinstance(camera_up, list)
        assert camera_up == [0.0, 1.0, 0.0]

        camera_pose = op_input.receive("camera_pose_input")
        # verify that received type matches expected types (depends on camera_pose_output_type)
        assert isinstance(camera_pose, (list, Pose3D))
        op_output.emit(camera_pose, "camera_pose_output")


class HolovizHeadlessApp(Application):
    def __init__(
        self,
        *args,
        count=10,
        width=800,
        height=640,
        on_host=False,
        enable_camera_pose_output=False,
        camera_pose_output_type="projection_matrix",
        **kwargs,
    ):
        self.count = count
        self.width = width
        self.height = height
        self.on_host = on_host
        self.enable_camera_pose_output = enable_camera_pose_output
        self.camera_pose_output_type = camera_pose_output_type
        super().__init__(*args, **kwargs)

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
        vizualizer = HolovizOp(
            self,
            headless=True,
            name="visualizer",
            width=self.width,
            height=self.height,
            enable_camera_pose_output=self.enable_camera_pose_output,
            camera_pose_output_type=self.camera_pose_output_type,
            tensors=[
                # name="" here to match the output of FrameGenerationOp
                dict(name="frame", type="color", opacity=0.5, priority=0),
            ],
        )
        self.add_flow(source, vizualizer, {("frame", "receivers")})
        self.add_flow(source, vizualizer, {("camera_eye", "camera_eye_input")})
        self.add_flow(source, vizualizer, {("camera_look_at", "camera_look_at_input")})
        self.add_flow(source, vizualizer, {("camera_up", "camera_up_input")})
        if self.enable_camera_pose_output:
            pose_forwarder = CameraPoseForwardingOp(self, name="pose_forwarder")
            camera_pose_rx = PingRxOp(self, name="camera_pose_rx")
            self.add_flow(vizualizer, pose_forwarder, {("camera_pose_output", "camera_pose_input")})
            self.add_flow(source, pose_forwarder, {("camera_up", "camera_up_input")})
            self.add_flow(pose_forwarder, camera_pose_rx, {("camera_pose_output", "in")})


@pytest.mark.parametrize("camera_pose_output_type", ["projection_matrix", "extrinsics_model"])
def test_holovizop_camera_inputs(camera_pose_output_type, capfd):
    """Test HolovizOp with valid (row-major) and invalid (column-major) memory layouts."""
    count = 3
    width = 800
    height = 640
    holoviz_app = HolovizHeadlessApp(
        count=count,
        width=width,
        height=height,
        on_host=False,
        enable_camera_pose_output=True,
        camera_pose_output_type=camera_pose_output_type,
    )

    holoviz_app.run()

    captured = capfd.readouterr()

    # assert that replayer_app emitted all frames
    assert captured.out.count("Emitting frame") == count

    # check that receive printed the expected type of object
    if camera_pose_output_type == "extrinsics_model":
        pose_repr = "Pose3D(rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [-0, -0, -1])"
        assert captured.out.count(f"Rx message value: {pose_repr}") == count
    elif camera_pose_output_type == "projection_matrix":
        # just check start of output value to avoid issues with floating point precision
        assert captured.out.count("Rx message value: [1.73205") == count
