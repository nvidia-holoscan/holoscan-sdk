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

# # Uncomment the following line to use the real HolovizOp and VideoStreamReplayerOp operators
# import os

import numpy as np
import pytest
from env_wrapper import env_var_context

from holoscan.conditions import CountCondition
from holoscan.core import Application, Fragment, IOSpec, Operator, OperatorSpec
from utils import remove_ignored_errors

# # Uncomment the following line to use the real HolovizOp and VideoStreamReplayerOp operators
# from holoscan.operators import HolovizOp, VideoStreamReplayerOp


class DummyVideoStreamReplayerOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.rng = np.random.default_rng()
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("output")

    def compute(self, op_input, op_output, context):
        frame_data = self.rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
        op_output.emit(frame_data, "output")


class TriangleOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.output("outputs")

    def compute(self, op_input, op_output, context):
        ########################################
        # Create a tensor defining two triangles
        ########################################
        # Each triangle is defined by a set of 3 (x, y) coordinate pairs.
        triangle_coords = np.asarray(
                [
                (0.1, 0.8), (0.18, 0.75), (0.14, 0.66),
                (0.3, 0.8), (0.38, 0.75), (0.34, 0.56),
                ],
            dtype=np.float32,
        )  # fmt: skip

        out_message = {"triangles": triangle_coords}

        # emit the tensors
        op_output.emit(out_message, "outputs")


class RectangleOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.output("outputs")

    def compute(self, op_input, op_output, context):
        ##########################################
        # Create a tensor defining four rectnagles
        ##########################################
        # For rectangles (bounding boxes), they are defined by a pair of
        # 2-tuples defining the upper-left and lower-right coordinates of a
        # box: (x1, y1), (x2, y2).
        box_coords = np.asarray(
             [
                (0.1, 0.2), (0.8, 0.5),
                (0.2, 0.4), (0.3, 0.6),
                (0.3, 0.5), (0.4, 0.7),
                (0.5, 0.7), (0.6, 0.9),
             ],
            dtype=np.float32,
        )  # fmt: skip

        out_message = {
            "boxes": box_coords,
        }

        # emit the tensors
        op_output.emit(out_message, "outputs")


class DummyHolovizOp(Operator):
    def __init__(self, fragment, *args, use_new_receivers=True, **kwargs):
        self._count = 0
        self.use_new_receivers = use_new_receivers
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        if self.use_new_receivers:
            spec.input("receivers", size=IOSpec.ANY_SIZE)
        else:
            spec.param("receivers", kind="receivers")

    def compute(self, op_input, op_output, context):
        value = op_input.receive("receivers")
        self._count += 1
        print(f"Received message {self._count} (size: {len(value)})")


class Fragment1(Fragment):
    def compose(self):
        global NUM_MSGS

        replayer = DummyVideoStreamReplayerOp(self, CountCondition(self, NUM_MSGS), name="replayer")
        # # Replace the above line with the following line to use the real VideoStreamReplayerOp
        # replayer = VideoStreamReplayerOp(self, name="replayer", **self.kwargs("replayer"))
        triangle = TriangleOp(self, CountCondition(self, NUM_MSGS), name="triangle")
        rectangle = RectangleOp(self, CountCondition(self, NUM_MSGS), name="rectangle")

        self.add_operator(replayer)
        self.add_operator(triangle)
        self.add_operator(rectangle)


class Fragment2(Fragment):
    def __init__(self, *args, use_new_receivers=True, **kwargs):
        self.use_new_receivers = use_new_receivers
        super().__init__(*args, **kwargs)

    def compose(self):
        visualizer = DummyHolovizOp(self, use_new_receivers=self.use_new_receivers, name="holoviz")
        # # Replace the above line with the following line to use the real HolovizOp
        # visualizer = HolovizOp(self, name="holoviz", **self.kwargs("holoviz"))

        self.add_operator(visualizer)


class DistributedVideoReplayerApp(Application):
    """Example of a distributed application that uses the fragments and operators defined above.

    This application has the following fragments:
    - Fragment1
      - holding VideoStreamReplayerOp (DummyVideoStreamReplayerOp), TriangleOp, RectangleOp
    - Fragment2
      - holding HolovizOp (DummyHolovizOp)

    The VideoStreamReplayerOp reads a video file and sends the frames to the HolovizOp.
    The TriangleOp and RectangleOp create tensors defining triangles and rectangles.
    The HolovizOp displays the frames.
    """

    def __init__(self, *args, use_new_receivers=True, **kwargs):
        self.use_new_receivers = use_new_receivers
        super().__init__(*args, **kwargs)

    def compose(self):
        # Define the fragments
        fragment1 = Fragment1(self, name="fragment1")
        fragment2 = Fragment2(self, use_new_receivers=self.use_new_receivers, name="fragment2")

        # Define the workflow
        self.add_flow(fragment1, fragment2, {("replayer.output", "holoviz.receivers")})
        self.add_flow(fragment1, fragment2, {("triangle.outputs", "holoviz.receivers")})
        self.add_flow(fragment1, fragment2, {("rectangle.outputs", "holoviz.receivers")})


# define the number of messages to send
NUM_MSGS = 100


def launch_app(use_new_receivers=True):
    env_var_settings = {
        # set the recession period to 5 ms to reduce debug messages
        ("HOLOSCAN_CHECK_RECESSION_PERIOD_MS", "5"),
        # set the max duration to 10s to have enough time to run the test
        # (connection time takes ~5 seconds)
        ("HOLOSCAN_MAX_DURATION_MS", "10000"),
        # set the stop on deadlock timeout to 10s to have enough time to run the test
        ("HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT", "10000"),
    }

    with env_var_context(env_var_settings):
        app = DistributedVideoReplayerApp(use_new_receivers=use_new_receivers)

        # # Uncomment the following line to use the real HolovizOp and VideoStreamReplayerOp
        # # operators
        # config_file_path = os.path.join(os.path.dirname(__file__),
        #                                 "video_replayer_distributed.yaml")
        # app.config(config_file_path)

        app.run()


@pytest.mark.parametrize("use_new_receivers", [True, False])
def test_distributed_app_three_ucx_receivers(use_new_receivers, capfd):
    global NUM_MSGS

    launch_app(use_new_receivers=use_new_receivers)

    # assert that no errors were logged
    captured = capfd.readouterr()

    print("Captured stdout:", captured.out)
    print("Captured stderr:", captured.err)

    # avoid catching the expected error message
    # : "error handling callback was invoked with status -25 (Connection reset by remote peer)"
    captured_error = captured.err.replace("error handling callback", "ucx handling callback")
    assert "error" not in remove_ignored_errors(captured_error)
    assert "Exception occurred" not in captured_error

    # assert that the expected number of messages were received
    assert f"Received message {NUM_MSGS} (size: 3)" in captured.out


if __name__ == "__main__":
    launch_app()

# When running this test with real HolovizOp and VideoStreamReplayerOp operators, the following
# configuration file (as "video_replayer_distributed.yaml" ) can be used:

# application:
#   title: Holoscan - Distributed Video Replayer
#   version: 1.0
#   inputFormats: ["file"]
#   outputFormats: ["screen"]

# resources:
#   cpu: 1
#   gpu: 1
#   memory: 1Gi
#   gpuMemory: 1Gi

# replayer:
#   directory: "../data/racerx"
#   basename: "racerx"
#   frame_rate: 0   # as specified in timestamps
#   repeat: true    # default: false
#   realtime: true  # default: true
#   count: 0        # default: 0 (no frame count restriction)

# holoviz:
#   width: 854
#   height: 480
#   tensors:
#     - name: ""
#       type: color
#       opacity: 1.0
#       priority: 0
#     - name: "triangles"
#       type: triangles
#       opacity: 1.0
#       color: [1.0, 0.0, 0.0, 0.5]
#     - name: "boxes"
#       type: rectangles
#       opacity: 1.0
#       color: [0.4, 1.0, 0.4, 0.7]
#       line_width: 3
