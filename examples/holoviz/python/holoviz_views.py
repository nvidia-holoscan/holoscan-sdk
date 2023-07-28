"""
SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import math
import os
from argparse import ArgumentParser

import numpy as np

from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import HolovizOp, VideoStreamReplayerOp
from holoscan.resources import CudaStreamPool

sample_data_path = os.environ.get("HOLOSCAN_INPUT_PATH", "../data")


class Matrix4x4:
    """4x4 row major matrix

    Note : as_...() means that the whole matrix is being modified
    set_...() only changes the concerned fields of the matrix

    """

    def __init__(self):
        self.identity()

    def identity(self):
        self.matrix_ = np.eye(4)

    def data(self):
        return self.matrix_.ravel()

    def set_scale(self, x, y, z):
        np.fill_diagonal(self.matrix_, (x, y, z, 1.0))

    def set_translate(self, x, y, z):
        self.matrix_[:3, 3] = (x, y, z)

    def set_rot_x(self, theta):
        self.set_rot(theta, (1.0, 0.0, 0.0))

    def set_rot_y(self, theta):
        self.set_rot(theta, (0.0, 1.0, 0.0))

    def set_rot_z(self, theta):
        self.set_rot(theta, (0.0, 0.0, 1.0))

    def set_rot(self, theta, axis):
        ct = math.cos(theta)
        st = math.sin(theta)
        if len(axis) != 3:
            raise ValueError("axis about which to rotate must have 3 elements")

        # normalize to unit vector along specified direction
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z

        self.matrix_[0, 0] = xx + ct * (1 - xx)
        self.matrix_[0, 1] = xy + ct * (-xy) + st * -z
        self.matrix_[0, 2] = xz + ct * (-xz) + st * y

        self.matrix_[1, 0] = xy + ct * (-xy) + st * z
        self.matrix_[1, 1] = yy + ct * (1 - yy)
        self.matrix_[1, 2] = yz + ct * (-yz) + st * -x

        self.matrix_[2, 0] = xz + ct * (-xz) + st * -y
        self.matrix_[2, 1] = yz + ct * (-yz) + st * x
        self.matrix_[2, 2] = zz + ct * (1 - zz)

    def as_scale(self, x, y, z):
        self.identity()
        self.set_scale(x, y, z)
        return self

    def as_translate(self, x, y, z):
        self.identity()
        self.set_translate(x, y, z)
        return self

    def as_rot_x(self, theta):
        self.identity()
        self.set_rot_x(theta)
        return self

    def as_rot_y(self, theta):
        self.identity()
        self.set_rot_y(theta)
        return self

    def as_rot_z(self, theta):
        self.identity()
        self.set_rot_z(theta)
        return self

    def as_rot(self, theta, x, y, z):
        self.identity()
        self.set_rot(theta, x, y, z)
        return self

    def mul(self, mat):
        self.matrix_ = self.matrix_ @ mat.matrix_
        return self


# Define custom Operators for use in the demo
class ImageViewsOp(Operator):
    """Example of an operator showing an input video in multiple views.

    This operator has:
        inputs:  "input_tensor"
        outputs: "output_tensor"
        output_specs: "output_specs"

    The the input is passed through and multiple dynamic and static views of that data are defined.

    In this demo, the input and output image (2D RGB) is a 3D array of shape
    (height, width, channels).
    """

    def __init__(self, fragment, *args, aspect_ratio=1.0, **kwargs):
        self.count = 1
        self.aspect_ratio = aspect_ratio

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("input_tensor")
        spec.output("outputs")
        spec.output("output_specs")

    def compute(self, op_input, op_output, context):
        message = op_input.receive("input_tensor")
        input_tensor = next(iter(message.values()))

        ####################################
        # Create a tensor for "text"
        ####################################
        # Set of two (x, y) points marking the location of text
        dynamic_text = np.asarray(
            [
                (0.0, 0.0, 0.2),
            ],
            dtype=np.float32,
        )
        dynamic_text = dynamic_text[np.newaxis, :, :]
        out_message = {
            "video": input_tensor,
            "dynamic_text": dynamic_text,
        }

        # emit the tensors
        op_output.emit(out_message, "outputs")

        ##################################
        # Create a input specs for "video"
        ##################################
        specs = []

        # rotate video in the upper right corner
        spec = HolovizOp.InputSpec("video", HolovizOp.InputType.COLOR)
        view = HolovizOp.InputSpec.View()
        view.offset_x = 0.55
        view.offset_y = 0.05
        view.width = 0.4
        view.height = 0.4
        angle = (math.pi / 180.0) * (self.count % 360.0)
        scale = 0.8
        mat = Matrix4x4().as_rot_z(angle)
        mat.mul(Matrix4x4().as_scale(scale, scale, scale))
        view.matrix = mat.data()
        spec.views = [view]
        specs.append(spec)

        # scale and translate video to move it around the whole window
        spec = HolovizOp.InputSpec("video", HolovizOp.InputType.COLOR)
        view = HolovizOp.InputSpec.View()
        view.offset_x = 0.0
        view.offset_y = 0.0
        view.width = 1.0
        view.height = 1.0
        mat = Matrix4x4().as_scale(
            (abs(self.count % 50 - 25) + 10) / 200.0,
            ((abs(self.count % 24 - 12) + 10) / 100.0) / self.aspect_ratio,
            1.0,
        )
        mat.set_translate(
            abs(self.count % 80 - 40) / 20.0 - 1.0, abs(self.count % 100 - 50) / 25.0 - 1.0, 0.0
        )
        view.matrix = mat.data()
        spec.views = [view]
        specs.append(spec)

        # rotate text in bottom right corner
        spec = HolovizOp.InputSpec("dynamic_text", HolovizOp.InputType.TEXT)
        spec.text = ["Frame " + str(self.count)]
        view = HolovizOp.InputSpec.View()
        view.offset_x = 0.55
        view.offset_y = 0.55
        view.width = 0.4
        view.height = 0.4
        angle = (math.pi / 180.0) * (self.count % 360.0)
        scale = 1.5
        mat = Matrix4x4().as_rot_z(angle)
        mat.mul(Matrix4x4().as_scale(scale, scale, scale))
        view.matrix = mat.data()
        spec.views = [view]
        specs.append(spec)

        # emit the output specs
        op_output.emit(specs, "output_specs")

        self.count += 1


# Now define a simple application using the operators defined above
class MyVideoViewsApp(Application):
    """Example of an application that uses the operators defined above.

    This application has the following operators:

    - VideoStreamReplayerOp
    - ImageViewsOp
    - HolovizOp

    The VideoStreamReplayerOp reads a video file and sends the frames to the ImageViewsOp.
    The ImageViewsOp reads the frames, defines views and sends the frames and views to the
    HolovizOp.
    The HolovizOp displays the frames using the views.
    """

    def __init__(self, config_count=0):
        """Initialize MyVideoViewsApp

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
        width = 800
        height = 800
        aspect_ratio = float(width) / float(height)
        video_dir = os.path.join(sample_data_path, "endoscopy", "video")
        if not os.path.exists(video_dir):
            raise ValueError(f"Could not find video data: {video_dir=}")
        source = VideoStreamReplayerOp(
            self,
            name="replayer",
            directory=video_dir,
            basename="surgical_video",
            frame_rate=0,  # as specified in timestamps
            repeat=True,  # default: false
            realtime=True,  # default: true
            count=self.count,  # default: 0 (no frame count restriction)
        )

        image_views = ImageViewsOp(self, name="image_views", aspect_ratio=aspect_ratio)

        # build an input spec for a grid of videos in the top left corner
        grid_spec = HolovizOp.InputSpec("video", HolovizOp.InputType.COLOR)
        views = []
        for y in range(5):
            for x in range(5):
                view = HolovizOp.InputSpec.View()
                view.offset_x = x * 0.1
                view.offset_y = y * 0.1
                view.width = 0.09
                view.height = 0.09
                views.append(view)
        grid_spec.views = views

        # flipped view of the video in bottom left corner
        rotated_view = HolovizOp.InputSpec.View()
        rotated_view.offset_x = 0.05
        rotated_view.offset_y = 0.95
        rotated_view.width = 0.4
        rotated_view.height = -0.4

        visualizer = HolovizOp(
            self,
            name="holoviz",
            width=width,
            height=height,
            tensors=[
                # add a rotated view using a dictionary
                dict(name="video", type="color", opacity=1.0, priority=0, views=[rotated_view]),
                # add the grid spec using the HolovizOp.InputSpec type
                grid_spec,
            ],
            cuda_stream_pool=CudaStreamPool(
                self,
                name="cuda_stream",
                dev_id=0,
                stream_flags=0,
                stream_priority=0,
                reserved_size=1,
                max_size=5,
            ),
        )
        self.add_flow(source, image_views)
        self.add_flow(image_views, visualizer, {("outputs", "receivers")})
        self.add_flow(image_views, visualizer, {("output_specs", "input_specs")})


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

    app = MyVideoViewsApp(config_count=args.count)
    app.run()
