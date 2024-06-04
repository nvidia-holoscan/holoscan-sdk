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
from holoscan.operators import BayerDemosaicOp, HolovizOp, SegmentationPostprocessorOp
from holoscan.resources import UnboundedAllocator


def get_frame(xp, height, width, channels, fortran_ordered, dtype=np.uint8):
    shape = (height, width, channels) if channels else (height, width)
    size = math.prod(shape)
    frame = xp.arange(size, dtype=dtype).reshape(shape)
    if fortran_ordered:
        frame = xp.asfortranarray(frame)
    return frame


class FrameGeneratorOp(Operator):
    def __init__(
        self,
        fragment,
        *args,
        width=800,
        height=640,
        channels=3,
        on_host=True,
        fortran_ordered=False,
        dtype=np.uint8,
        **kwargs,
    ):
        self.height = height
        self.width = width
        self.channels = channels
        self.on_host = on_host
        self.fortran_ordered = fortran_ordered
        self.dtype = dtype
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("frame")

    def compute(self, op_input, op_output, context):
        xp = np if self.on_host else cp
        frame = get_frame(
            xp, self.height, self.width, self.channels, self.fortran_ordered, self.dtype
        )
        print(f"Emitting frame with shape: {frame.shape}")
        op_output.emit(dict(frame=frame), "frame")


class HolovizHeadlessApp(Application):
    def __init__(
        self,
        *args,
        count=10,
        width=800,
        height=640,
        on_host=True,
        fortran_ordered=False,
        **kwargs,
    ):
        self.count = count
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
            dtype=np.uint8,
            name="video_source",
        )
        vizualizer = HolovizOp(
            self,
            headless=True,
            name="visualizer",
            width=self.width,
            height=self.height,
            enable_camera_pose_output=False,
            tensors=[
                # name="" here to match the output of FrameGenerationOp
                dict(name="frame", type="color", opacity=0.5, priority=0),
            ],
        )
        self.add_flow(source, vizualizer, {("frame", "receivers")})


@pytest.mark.parametrize("fortran_ordered", [True, False])
def test_holovizop_memory_layout(fortran_ordered, capfd):
    """Test HolovizOp with valid (row-major) and invalid (column-major) memory layouts."""
    count = 3
    width = 800
    height = 640
    holoviz_app = HolovizHeadlessApp(
        count=count,
        width=width,
        height=height,
        on_host=True,
        fortran_ordered=fortran_ordered,
    )

    if fortran_ordered:
        with pytest.raises(RuntimeError):
            holoviz_app.run()
        captured = capfd.readouterr()

        # assert that app raised exception on the first frame
        assert captured.out.count("Emitting frame") == 1
        assert "Tensor must have a row-major memory layout" in captured.err

    else:
        holoviz_app.run()

        captured = capfd.readouterr()

        # assert that replayer_app received all frames
        assert captured.out.count("Emitting frame") == count


class BayerDemosaicApp(Application):
    def __init__(
        self,
        *args,
        count=10,
        width=800,
        height=640,
        channels=3,
        on_host=False,
        fortran_ordered=False,
        **kwargs,
    ):
        self.count = count
        self.width = width
        self.height = height
        self.channels = channels
        self.on_host = on_host
        self.fortran_ordered = fortran_ordered
        super().__init__(*args, **kwargs)

    def compose(self):
        source = FrameGeneratorOp(
            self,
            CountCondition(self, count=self.count),
            width=self.width,
            height=self.height,
            channels=self.channels,
            on_host=self.on_host,  # BayerDemosaicOp expects device tensor
            fortran_ordered=self.fortran_ordered,
            dtype=np.uint8,
            name="video_source",
        )
        demosaic = BayerDemosaicOp(
            self,
            in_tensor_name="frame",
            out_tensor_name="frame_rgb",
            generate_alpha=False,
            bayer_grid_pos=2,
            interpolation_mode=0,
            pool=UnboundedAllocator(self, "device_pool"),
            name="demosaic",
        )
        vizualizer = HolovizOp(
            self,
            headless=True,
            name="visualizer",
            width=self.width,
            height=self.height,
            enable_camera_pose_output=False,
            tensors=[
                # name="" here to match the output of FrameGenerationOp
                dict(name="frame_rgb", type="color", opacity=0.5, priority=0),
            ],
        )
        self.add_flow(source, demosaic, {("frame", "receiver")})
        self.add_flow(demosaic, vizualizer, {("transmitter", "receivers")})


@pytest.mark.parametrize("fortran_ordered, on_host", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("channels", [3, None])
def test_bayer_demosaic_memory_layout(fortran_ordered, on_host, channels, capfd):
    """Test HolovizOp with valid (row-major) and invalid (column-major) memory layouts."""
    count = 3
    width = 800
    height = 640
    demosaic_app = BayerDemosaicApp(
        count=count,
        width=width,
        height=height,
        channels=channels,
        on_host=on_host,
        fortran_ordered=fortran_ordered,
    )
    if channels is None:
        with pytest.raises(RuntimeError):
            demosaic_app.run()
        captured = capfd.readouterr()

        # assert that app raised exception on the first frame
        assert captured.out.count("Emitting frame") == 1
        assert "Input tensor has 2 dimensions. Expected a tensor with 3 dimensions" in captured.err

    else:
        if fortran_ordered:
            with pytest.raises(RuntimeError):
                demosaic_app.run()
            captured = capfd.readouterr()

            # assert that app raised exception on the first frame
            assert captured.out.count("Emitting frame") == 1
            assert "Tensor must have a row-major memory layout" in captured.err

        else:
            demosaic_app.run()

            captured = capfd.readouterr()

            # assert that replayer_app received all frames
            assert captured.out.count("Emitting frame") == count


class SegmentationPostprocessorApp(Application):
    def __init__(
        self,
        *args,
        count=10,
        width=800,
        height=640,
        channels=1,
        on_host=False,
        fortran_ordered=False,
        network_output_type="softmax",
        dtype=np.float32,
        **kwargs,
    ):
        self.count = count
        self.width = width
        self.height = height
        self.channels = channels
        self.on_host = on_host
        self.fortran_ordered = fortran_ordered
        self.network_output_type = network_output_type
        self.dtype = dtype
        super().__init__(*args, **kwargs)

    def compose(self):
        source = FrameGeneratorOp(
            self,
            CountCondition(self, count=self.count),
            width=self.width,
            height=self.height,
            channels=self.channels,
            on_host=self.on_host,  # SegmentationPostprocessorOp expects device tensor
            fortran_ordered=self.fortran_ordered,
            dtype=self.dtype,
            name="video_source",
        )
        postprocessor = SegmentationPostprocessorOp(
            self,
            allocator=UnboundedAllocator(self, "device_allocator"),
            in_tensor_name="frame",
            data_format="hwc",
            network_output_type=self.network_output_type,
            name="postprocessor",
        )
        vizualizer = HolovizOp(
            self,
            headless=True,
            name="visualizer",
            width=self.width,
            height=self.height,
            enable_camera_pose_output=False,
            tensors=[
                # name="" here to match the output of FrameGenerationOp
                dict(name="out_tensor", type="color", opacity=0.5, priority=0),
            ],
        )
        self.add_flow(source, postprocessor, {("frame", "in_tensor")})
        self.add_flow(postprocessor, vizualizer, {("out_tensor", "receivers")})


@pytest.mark.parametrize(
    "fortran_ordered, on_host, dtype",
    [
        (False, False, np.float32),  # valid values
        (True, False, np.float32),  # invalid memory order
        (False, True, np.float32),  # tensor not on device
        (False, False, np.uint8),  # input is not 32-bit floating point
    ],
)
@pytest.mark.parametrize("channels", [1, 5])
@pytest.mark.parametrize("network_output_type", ["softmax", "sigmoid"])
def test_segmentation_postproceesor_memory_layout(
    fortran_ordered, on_host, dtype, channels, network_output_type, capfd
):
    """Test HolovizOp with valid (row-major) and invalid (column-major) memory layouts."""
    count = 3
    width = 800
    height = 640
    postprocessor_app = SegmentationPostprocessorApp(
        count=count,
        width=width,
        height=height,
        channels=channels,
        on_host=on_host,
        fortran_ordered=fortran_ordered,
        dtype=dtype,
        network_output_type=network_output_type,
    )
    if on_host or fortran_ordered or dtype != np.float32:
        with pytest.raises(RuntimeError):
            postprocessor_app.run()
        captured = capfd.readouterr()

        # assert that app raised exception on the first frame
        assert captured.out.count("Emitting frame") == 1
        if on_host:
            assert "Input tensor must be in CUDA device or pinned host memory" in captured.err
        elif fortran_ordered:
            assert "Input tensor must have row-major memory layout" in captured.err
        elif dtype != np.float32:
            assert "Input tensor must be of type float32" in captured.err
    else:
        postprocessor_app.run()

        captured = capfd.readouterr()

        # assert that replayer_app received all frames
        assert captured.out.count("Emitting frame") == count

        # multi-channel input to sigmoid warns
        warning_count = int(network_output_type == "sigmoid" and channels > 1)
        assert captured.err.count("Only the first channel will be used") == warning_count
