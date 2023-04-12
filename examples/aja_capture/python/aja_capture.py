"""
SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""  # noqa

from holoscan.core import Application
from holoscan.logger import load_env_log_level
from holoscan.operators import AJASourceOp, HolovizOp


class AJACaptureApp(Application):
    """
    Example of an application that uses the following operators:

    - AJASourceOp
    - HolovizOp

    The AJASourceOp reads frames from an AJA input device and sends it to the HolovizOp.
    The HolovizOp displays the frames.
    """

    def compose(self):

        width = 1920
        height = 1080

        source = AJASourceOp(
            self,
            name="aja",
            width=width,
            height=height,
            rdma=True,
            enable_overlay=False,
            overlay_rdma=True,
        )

        visualizer = HolovizOp(
            self,
            name="holoviz",
            width=width,
            height=height,
            tensors=[{"name": "", "type": "color", "opacity": 1.0, "priority": 0}],
        )

        self.add_flow(source, visualizer, {("video_buffer_output", "receivers")})


if __name__ == "__main__":

    load_env_log_level()

    app = AJACaptureApp()
    app.run()
