"""
SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import random
from argparse import ArgumentParser

import numpy as np

from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import HolovizOp, VideoStreamReplayerOp

sample_data_path = os.environ.get("HOLOSCAN_INPUT_PATH", "../data")


# Define custom Operators for use in the demo
class GeometryGenerationOp(Operator):
    """Example creating geometric primitives for overlay on a video.

    This operator has:
        outputs: outputs, output_specs

    The entity transmitted on the "outputs" port will contain several additional internally
    generated tensors ("boxes", "crosses", "ovals", etc.) that correspond to coordinates defining
    various types of geometric primitives that will be displayed by the downstream Holoviz
    operator.

    The entity transmitted on the "output_specs" port is a text label of a randomly varying color.
    """

    def __init__(self, fragment, *args, **kwargs):
        self.count = 1

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("outputs")
        spec.output("output_specs")

    def compute(self, op_input, op_output, context):
        # Draw various different types of geometric primitives.
        # In all cases, x and y are normalized coordinates in the range [0, 1].
        # x runs from left to right and y from bottom to top.

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

        )  # fmt: skip

        ########################################
        # Create a tensor defining two triangles
        ########################################
        # Each triangle is defined by a set of 3 (x, y) coordinate pairs.
        triangle_coords = np.asarray(
            [
                (0.1, 0.8), (0.18, 0.75), (0.14, 0.66),
                (0.3, 0.8), (0.38, 0.75), (0.34, 0.56),
            ],
        )  # fmt: skip

        ######################################
        # Create a tensor defining two crosses
        ######################################
        # Each cross is defined by an (x, y, size) 3-tuple
        cross_coords = np.asarray(
            [
                (0.25, 0.25, 0.05),
                (0.75, 0.25, 0.10),
            ],
        )  # fmt: skip

        ######################################
        # Create a tensor defining three ovals
        ######################################
        # Each oval is defined by an (x, y, size_x, size_y) 4-tuple
        oval_coords = np.asarray(
            [
                (0.25, 0.65, 0.10, 0.05),
                (0.50, 0.65, 0.02, 0.15),
                (0.75, 0.65, 0.05, 0.10),
            ],
        )  # fmt: skip

        #######################################
        # Create a time-varying "points" tensor
        #######################################
        # Set of (x, y) points with 50 points equally spaced along x whose y
        # coordinate varies sinusoidally over time.
        x = np.linspace(0, 1.0, 50)
        y = 0.8 + 0.1 * np.sin(8 * np.pi * x + self.count / 60 * 2 * np.pi)
        self.count += 1
        # Stack so the final shape is (n_points, 2)
        point_coords = np.stack((x, y), axis=-1)

        ####################################
        # Create a tensor for "label_coords"
        ####################################
        # Set of two (x, y) points marking the location of text labels
        label_coords = np.asarray(
            [
                (0.10, 0.1),
                (0.70, 0.1),
            ],
        )  # fmt: skip

        ####################################
        # Create a tensor for "dynamic_text"
        ####################################
        # Set of two (x, y) points marking the location of text
        dynamic_text = np.asarray(
            [
                (0.0, 0.0),
            ],
        )  # fmt: skip

        out_message = {
            "boxes": box_coords,
            "triangles": triangle_coords,
            "crosses": cross_coords,
            "ovals": oval_coords,
            "points": point_coords,
            "label_coords": label_coords,
            "dynamic_text": dynamic_text,
        }

        # emit the tensors
        op_output.emit(out_message, "outputs")

        ########################################
        # Create a input spec for "dynamic_text"
        ########################################
        # To dynamically change the input spec create a list of HolovizOp.InputSpec objects
        # and pass it to Holoviz.
        # All properties of the input spec (type, color, text, line width, ...) can be changed
        # dynamically.
        specs = []
        spec = HolovizOp.InputSpec("dynamic_text", "text")
        spec.text = ["Frame " + str(self.count)]
        spec.color = [
            random.uniform(0.0, 1.0),
            random.uniform(0.0, 1.0),
            random.uniform(0.0, 1.0),
            1.0,
        ]
        specs.append(spec)
        # emit the output specs
        op_output.emit(specs, "output_specs")


# Now define a simple application using the operators defined above
class MyVideoProcessingApp(Application):
    """Example of an application that uses the operators defined above.

    This application has the following operators:

    - VideoStreamReplayerOp
    - GeometryGenerationOp
    - HolovizOp

    The VideoStreamReplayerOp reads a video file and sends the frames to the GeometryGenerationOp.
    The GeometryGenerationOp processes the frames and sends the processed frames to the HolovizOp.
    The HolovizOp displays the processed frames.
    """

    def __init__(self, config_count=0):
        """Initialize MyVideoProcessingAPP

        Parameters
        ----------
        config_count : optional
            Limits the number of frames to show before the application ends.
            Set to 0 by default. The video stream will not automatically stop.
            Any positive integer will limit on the number of frames displayed.
        """
        super().__init__()

        self.count = int(config_count)

    def compose(self):
        width = 854
        height = 480
        video_dir = os.path.join(sample_data_path, "racerx")
        if not os.path.exists(video_dir):
            raise ValueError(f"Could not find video data: {video_dir=}")
        source = VideoStreamReplayerOp(
            self,
            name="replayer",
            directory=video_dir,
            basename="racerx",
            frame_rate=0,  # as specified in timestamps
            repeat=True,  # default: false
            realtime=True,  # default: true
            count=self.count,  # default: 0 (no frame count restriction)
        )

        image_processing = GeometryGenerationOp(
            self,
            name="geometry_layer_object_generator",
        )

        visualizer = HolovizOp(
            self,
            name="holoviz",
            width=width,
            height=height,
            tensors=[
                # name="" here to match the output of VideoStreamReplayerOp
                dict(name="", type="color", opacity=0.5, priority=0),
                # The remaining tensors below come from GeometryGenerationOp
                # Parameters defining the rectangle primitives
                dict(
                    name="boxes",
                    type="rectangles",
                    opacity=1.0,
                    color=[1.0, 0.0, 1.0, 0.5],
                    line_width=2,
                ),
                # line strip reuses the rectangle coordinates. This will make
                # a connected set of line segments through the diagonals of
                # each box.
                dict(
                    name="boxes",
                    type="line_strip",
                    opacity=1.0,
                    color=[0.4, 0.4, 1.0, 0.7],
                    line_width=3,
                ),
                # Lines also reuses the boxes coordinates so will plot a set of
                # disconnected line segments along the box diagonals.
                dict(
                    name="boxes",
                    type="lines",
                    opacity=1.0,
                    color=[0.4, 1.0, 0.4, 0.7],
                    line_width=3,
                ),
                # Parameters defining the triangle primitives
                dict(
                    name="triangles",
                    type="triangles",
                    opacity=1.0,
                    color=[1.0, 0.0, 0.0, 0.5],
                    line_width=1,
                ),
                # Parameters defining the crosses primitives
                dict(
                    name="crosses",
                    type="crosses",
                    opacity=1.0,
                    color=[0.0, 1.0, 0.0, 1.0],
                    line_width=3,
                ),
                # Parameters defining the ovals primitives
                dict(
                    name="ovals",
                    type="ovals",
                    opacity=0.5,
                    color=[1.0, 1.0, 1.0, 1.0],
                    line_width=2,
                ),
                # Parameters defining the points primitives
                dict(
                    name="points",
                    type="points",
                    opacity=1.0,
                    color=[1.0, 1.0, 1.0, 1.0],
                    point_size=4,
                ),
                # Parameters defining the label_coords primitives
                dict(
                    name="label_coords",
                    type="text",
                    opacity=1.0,
                    color=[1.0, 1.0, 1.0, 1.0],
                    point_size=4,
                    text=["label_1", "label_2"],
                ),
            ],
        )
        self.add_flow(source, visualizer, {("output", "receivers")})
        self.add_flow(image_processing, visualizer, {("outputs", "receivers")})
        self.add_flow(image_processing, visualizer, {("output_specs", "input_specs")})


def main(config_count):
    app = MyVideoProcessingApp(config_count=config_count)
    app.run()


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="Example video processing application")
    parser.add_argument(
        "-c",
        "--count",
        default=0,
        help="Set the number of frames to display the video",
    )
    args = parser.parse_args()
    main(config_count=args.count)
