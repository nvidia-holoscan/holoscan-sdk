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

import math
import os

import numpy as np

from holoscan.conditions import CountCondition
from holoscan.core import (
    Application,
    Operator,
)

# ───────────── PoseTree fragment-service wrapper ────────────────────────────
from holoscan.pose_tree import SO3, Pose3, PoseTreeManager

TWO_PI = 2.0 * math.pi
DAY_SEC = 86_400.0  # one simulated day


# ─────────────────── 1. OrbitSetterOp ─────────────────────────────────────────
class OrbitSetterOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.sim_time = 0.0  # seconds since t0
        self.pose_tree = self.service(PoseTreeManager, "pose_tree_manager").tree

        # Create frames & edges once.
        self.pose_tree.create_frame("sun")
        self.pose_tree.create_frame("earth")
        self.pose_tree.create_frame("moon")
        self.pose_tree.create_edges("sun", "earth")
        self.pose_tree.create_edges("earth", "moon")

    def compute(self, _op_input, _op_output, _context):
        pose_tree = self.pose_tree

        # advance simulation
        self.sim_time += DAY_SEC
        now = self.sim_time

        # orbital parameters (scaled)
        t_earth = 365.25 * DAY_SEC
        t_moon = 27.32 * DAY_SEC
        r_earth = 1.0  # AU (scaled)
        r_moon = 3.844e5 / 1.5e8  # ~0.00256 AU

        theta_e = TWO_PI * (now / t_earth)
        theta_m = TWO_PI * (now / t_moon)

        # Sun → Earth pose (translation + rotation about Z)
        earth_pos = np.array(
            [r_earth * math.cos(theta_e), r_earth * math.sin(theta_e), 0.0], dtype=float
        )
        earth_rot = SO3.from_axis_angle(np.array([0.0, 0.0, 1.0]), theta_e + math.pi / 2)
        pose_tree.set("sun", "earth", now, Pose3(earth_rot, earth_pos))

        # Earth → Moon pose
        moon_pos = np.array(
            [r_moon * math.cos(theta_m), r_moon * math.sin(theta_m), 0.0], dtype=float
        )
        moon_rot = SO3.from_axis_angle(np.array([0.0, 0.0, 1.0]), theta_m)
        pose_tree.set("earth", "moon", now, Pose3(moon_rot, moon_pos))


# ─────────────────── 2. TransformPrinterOp ────────────────────────────────────
class TransformPrinterOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.sim_time = 0.0
        self.pose_tree = self.service(PoseTreeManager, "pose_tree_manager").tree

    def compute(self, _op_input, _op_output, _context):
        pose_tree = self.pose_tree
        self.sim_time += DAY_SEC
        now = self.sim_time

        sun_earth = pose_tree.get("sun", "earth", now)
        earth_moon = pose_tree.get("earth", "moon", now)
        sun_moon = pose_tree.get("sun", "moon", now)

        day = now / DAY_SEC
        print(f"[day {day:4.0f}] Sun→Earth : T={sun_earth.translation}, Q={sun_earth.rotation}")
        print(f"[day {day:4.0f}] Earth→Moon: T={earth_moon.translation}, Q={earth_moon.rotation}")
        print(f"[day {day:4.0f}] Sun→Moon  : T={sun_moon.translation}, Q={sun_moon.rotation}")

        # Highlight interpolation: query pose at an intermediate time (12 hours ago)
        if day == 365:
            interpolated_time = now - DAY_SEC / 2.0
            sun_moon_interp = pose_tree.get("sun", "moon", interpolated_time)
            print(
                f"[day {day - 0.5:4.1f}] Sun→Moon  : T={sun_moon_interp.translation}, "
                f"Q={sun_moon_interp.rotation} (interpolated)"
            )


class PoseTreeOrbitApp(Application):
    def compose(self):
        # In a Holoscan application, a PoseTree is managed by the PoseTreeManager resource, which
        # acts as a FragmentService. This allows multiple operators within the same fragment to
        # share access to the same pose data, ensuring consistency.

        # Parameters to initialize the underlying PoseTree can be configured here.
        # These values are for demonstration; production apps should tune them.
        pose_tree_config = self.kwargs("pose_tree_config")  # see pose_tree_basic.yaml
        pose_tree_config["number_frames"] = 64  # override the default value

        pose_tree_service = PoseTreeManager(
            self,
            name="pose_tree_manager",
            **pose_tree_config,
        )
        self.register_service(pose_tree_service)

        # Operators
        orbit_setter_op = OrbitSetterOp(self, CountCondition(self, 365), name="orbit_setter_op")
        transform_printer_op = TransformPrinterOp(self, name="transform_printer_op")

        # Connect – single edge is enough
        self.add_flow(orbit_setter_op, transform_printer_op)


def main(config_file):
    app = PoseTreeOrbitApp()
    # if the --config command line argument was provided, it will override this config_file
    app.config(config_file)
    app.run()


if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(__file__), "pose_tree_basic.yaml")
    main(config_file=config_file)
