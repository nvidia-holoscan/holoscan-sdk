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

import random
import time
from argparse import ArgumentParser

import numpy as np

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import HolovizOp


# This holds the information on a 3d geometric primitive
class Item:
    def __init__(self, fixed_type=""):
        self.fixed_type = fixed_type
        self.init()

    def init(self):
        if self.fixed_type:
            self.type = self.fixed_type
        else:
            self.type = random.choice(["points_3d", "lines_3d", "triangles_3d"])
        if self.type == "points_3d":
            count = 1
        elif self.type == "lines_3d":
            count = 2
        elif self.type == "line_strip_3d":
            count = 2
        elif self.type == "triangles_3d":
            count = 3
        count *= random.randrange(1, 4)
        center = (
            random.uniform(-1.0, 1.0),
            1.0 + random.uniform(-0.2, 0.2),
            random.uniform(-1.0, 1.0),
        )
        self.coords = []
        for _ in range(count):
            self.coords.append(
                [
                    center[0] + random.uniform(-0.1, 0.1),
                    center[1] + random.uniform(-0.1, 0.1),
                    center[2] + random.uniform(-0.1, 0.1),
                ]
            )
        self.speed = random.uniform(0.01, 0.02)

    def update(self):
        for coord in self.coords:
            coord[1] -= self.speed
        if self.coords[0][1] < -1.0:
            self.init()


# Define custom Operators for use in the demo
class Geometry3dOp(Operator):
    """Example of an operator drawing 3d geometry.

    This operator has:
        outputs: "outputs"
    """

    def __init__(self, fragment, *args, **kwargs):
        # start with one item of each type
        self.items = [
            Item("points_3d"),
            Item("lines_3d"),
            Item("line_strip_3d"),
            Item("triangles_3d"),
        ]
        # add a random list of items
        for _ in range(50):
            self.items.append(Item())

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("outputs")

    def compute(self, op_input, op_output, context):
        # delay execution to run at 50 fps
        time.sleep(1.0 / 50.0)
        out_message = dict()

        # Now draw various different types of geometric primitives.
        # In all cases, x, y and z are coordinates in 3d space. All coordinates
        # should be defined using a single precision (np.float32) dtype.

        prim_coords = {}
        prim_coords["points_3d"] = []
        prim_coords["lines_3d"] = []
        prim_coords["line_strip_3d"] = []
        prim_coords["triangles_3d"] = []

        for item in self.items:
            for coord in item.coords:
                prim_coords[item.type].append(coord)
            item.update()

        for name, coords in prim_coords.items():
            coords = np.asarray(
                coords,
                dtype=np.float32,
            )
            coords = coords[np.newaxis, :, :]
            out_message[name] = coords
        op_output.emit(out_message, "outputs")


# Now define a simple application using the operators defined above
class MyGeometry3DApp(Application):
    """Example of an application that uses the operator defined above.

    This application has the following operators:

    - Geometry3dOp
    - HolovizOp

    The Geometry3dOp generates 3D geometric primitives and sends it to HolovizOp.
    The HolovizOp renders and displays geometry.
    """

    def __init__(self, config_count=0):
        """Initialize MyGeometry3DApp

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
        width = 1080
        height = 768

        if self.count == 0:
            geometry_3d = Geometry3dOp(self, name="geometry_3d")
        else:
            geometry_3d = Geometry3dOp(self, CountCondition(self, self.count), name="geometry_3d")

        visualizer = HolovizOp(
            self,
            name="holoviz",
            width=width,
            height=height,
            tensors=[
                # Parameters defining the 3d points primitives
                dict(
                    name="points_3d",
                    type="points_3d",
                    opacity=1.0,
                    color=[1.0, 1.0, 1.0, 1.0],
                    point_size=4,
                ),
                # Parameters defining the 3d lines primitives
                dict(
                    name="lines_3d",
                    type="lines_3d",
                    opacity=1.0,
                    color=[1.0, 0.0, 0.0, 1.0],
                    line_width=6,
                ),
                # Parameters defining the 3d line strip primitives
                dict(
                    name="line_strip_3d",
                    type="line_strip_3d",
                    opacity=1.0,
                    color=[0.0, 1.0, 0.0, 1.0],
                    line_width=3,
                ),
                # Parameters defining the 3d triangles primitives
                dict(
                    name="triangles_3d",
                    type="triangles_3d",
                    opacity=1.0,
                    color=[0.0, 0.0, 1.0, 1.0],
                ),
            ],
        )
        self.add_flow(geometry_3d, visualizer, {("outputs", "receivers")})


if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="Example Holoviz geometry 3D application")
    parser.add_argument(
        "-c",
        "--count",
        default=0,
        help="Set the number of frames to display",
    )
    args = parser.parse_args()

    app = MyGeometry3DApp(config_count=args.count)
    app.run()
