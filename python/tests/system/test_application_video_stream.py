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
import os
import shutil
import tempfile

import cupy as cp
import numpy as np
import pytest

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import VideoStreamRecorderOp, VideoStreamReplayerOp


def get_frame(xp, height, width, channels, fortran_ordered):
    shape = (height, width, channels)
    size = math.prod(shape)
    frame = xp.arange(size, dtype=np.uint8).reshape(shape)
    if fortran_ordered:
        frame = xp.asfortranarray(frame)
    return frame


class FrameGeneratorOp(Operator):
    def __init__(
        self, fragment, *args, width=800, height=640, on_host=True, fortran_ordered=False, **kwargs
    ):
        self.height = height
        self.width = width
        self.on_host = on_host
        self.fortran_ordered = fortran_ordered
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("frame")

    def compute(self, op_input, op_output, context):
        xp = np if self.on_host else cp
        frame = get_frame(xp, self.height, self.width, 3, self.fortran_ordered)
        op_output.emit(dict(frame=frame), "frame")


class FrameValidationRxOp(Operator):
    def __init__(
        self, fragment, *args, expected_width=800, expected_height=640, on_host=True, **kwargs
    ):
        self.expected_height = expected_height
        self.expected_width = expected_width
        self.on_host = on_host
        self.count = 0
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        tensormap = op_input.receive("in")
        assert "frame" in tensormap
        tensor = tensormap["frame"]
        self.count += 1
        print(f"received frame {self.count}")
        assert tensor.shape == (self.expected_height, self.expected_width, 3)
        if self.on_host:
            xp = np
            assert hasattr(tensor, "__array_interface__")

        else:
            xp = cp
            assert hasattr(tensor, "__cuda_array_interface__")
        assert xp.asarray(tensor).dtype == xp.uint8
        expected_frame = get_frame(xp, self.expected_height, self.expected_width, 3, False)
        xp.testing.assert_array_equal(expected_frame, xp.ascontiguousarray(xp.asarray(tensor)))


class VideoRecorderApp(Application):
    def __init__(
        self,
        *args,
        count=10,
        width=800,
        height=640,
        on_host=True,
        fortran_ordered=False,
        directory="/tmp",
        basename="test_video",
        **kwargs,
    ):
        self.count = count
        self.directory = directory
        self.basename = basename
        self.width = width
        self.height = height
        self.on_host = on_host
        self.fortran_ordered = fortran_ordered
        super().__init__(*args, **kwargs)

    def compose(self):
        source = FrameGeneratorOp(
            self,
            CountCondition(self, count=self.count),
            width=self.width,
            height=self.height,
            on_host=self.on_host,
            fortran_ordered=self.fortran_ordered,
            name="video_source",
        )
        recorder = VideoStreamRecorderOp(
            self, directory=self.directory, basename=self.basename, name="recorder"
        )
        self.add_flow(source, recorder)


class VideoReplayerApp(Application):
    def __init__(
        self,
        *args,
        count=10,
        expected_width=800,
        expected_height=640,
        on_host=True,
        directory="/tmp",
        basename="test_video",
        **kwargs,
    ):
        self.count = count
        self.directory = directory
        self.basename = basename
        self.expected_width = expected_width
        self.expected_height = expected_height
        self.on_host = on_host
        super().__init__(*args, **kwargs)

    def compose(self):
        replayer = VideoStreamReplayerOp(
            self,
            directory=self.directory,
            basename=self.basename,
            repeat=False,
            realtime=False,
            name="replayer",
        )
        frame_validator = FrameValidationRxOp(
            self,
            expected_width=self.expected_width,
            expected_height=self.expected_height,
            on_host=self.on_host,
            name="frame_validator",
        )
        self.add_flow(replayer, frame_validator)


@pytest.mark.parametrize("on_host", [True, False])
@pytest.mark.parametrize("fortran_ordered", [True, False])
def test_recorder_and_replayer_roundtrip(fortran_ordered, on_host, capfd):
    """Test the functionality of VideoStreamRecorderOp and VideoStreamReplayerOp.

    This test case runs two applications back to back

    1.) recorder_app : serializes synthetic data frames to disk
    2.) replayer_app : deserializes frames from disk and validates them
    """
    directory = tempfile.mkdtemp()
    try:
        count = 10
        width = 800
        height = 640
        basename = "test_video_cpu" if on_host else "test_video_gpu"
        recorder_app = VideoRecorderApp(
            count=count,
            directory=directory,
            basename=basename,
            width=width,
            height=height,
            on_host=on_host,
            fortran_ordered=fortran_ordered,
        )
        recorder_app.run()

        # verify that expected files were generated
        out_files = os.listdir(directory)
        assert len(out_files) == 2
        assert basename + ".gxf_entities" in out_files
        assert basename + ".gxf_index" in out_files

        # verify that deserialized frames match the expected shape
        replayer_app = VideoReplayerApp(
            directory=directory,
            basename=basename,
            expected_width=width,
            expected_height=height,
            on_host=on_host,
        )
        replayer_app.run()

        # assert that replayer_app received all frames
        captured = capfd.readouterr()
        assert f"received frame {count}" in captured.out
    finally:
        shutil.rmtree(directory)
