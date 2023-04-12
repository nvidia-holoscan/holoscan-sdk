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

import os
import sys

from holoscan.core import Application
from holoscan.operators import HolovizOp, VideoStreamReplayerOp

sample_data_path = os.environ.get("HOLOSCAN_SAMPLE_DATA_PATH", "../data")


class VideoReplayerApp(Application):
    """Example of an application that uses the operators defined above.

    This application has the following operators:

    - VideoStreamReplayerOp
    - HolovizOp

    The VideoStreamReplayerOp reads a video file and sends the frames to the HolovizOp.
    The HolovizOp displays the frames.
    """

    def compose(self):
        video_dir = os.path.join(sample_data_path, "endoscopy", "video")
        if not os.path.exists(video_dir):
            raise ValueError(f"Could not find video data: {video_dir=}")

        # Define the replayer and holoviz operators
        replayer = VideoStreamReplayerOp(
            self, name="replayer", directory=video_dir, **self.kwargs("replayer")
        )
        visualizer = HolovizOp(self, name="holoviz", **self.kwargs("holoviz"))

        # Define the workflow
        self.add_flow(replayer, visualizer, {("output", "receivers")})


if __name__ == "__main__":

    config_file = os.path.join(os.path.dirname(__file__), "video_replayer.yaml")

    if len(sys.argv) >= 2:
        config_file = sys.argv[1]

    app = VideoReplayerApp()
    app.config(config_file)
    app.run()
