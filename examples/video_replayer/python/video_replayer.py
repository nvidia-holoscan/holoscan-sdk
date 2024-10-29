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

from holoscan.core import Application
from holoscan.operators import HolovizOp, VideoStreamReplayerOp
from holoscan.resources import RMMAllocator

sample_data_path = os.environ.get("HOLOSCAN_INPUT_PATH", "../data")


class VideoReplayerApp(Application):
    """Example of an application that uses the operators defined above.

    This application has the following operators:

    - VideoStreamReplayerOp
    - HolovizOp

    The VideoStreamReplayerOp reads a video file and sends the frames to the HolovizOp.
    The HolovizOp displays the frames.
    """

    def compose(self):
        video_dir = os.path.join(sample_data_path, "racerx")
        if not os.path.exists(video_dir):
            raise ValueError(f"Could not find video data: {video_dir=}")

        # create an allocator supporting both host and device memory pools
        # (The video stream is copied to an intermediate host buffer before being copied to the GPU)
        rmm_allocator = RMMAllocator(self, name="rmm-allocator", **self.kwargs("rmm_allocator"))

        # Define the replayer and holoviz operators
        replayer = VideoStreamReplayerOp(
            self,
            name="replayer",
            directory=video_dir,
            **self.kwargs("replayer"),
            allocator=rmm_allocator,
        )
        visualizer = HolovizOp(self, name="holoviz", **self.kwargs("holoviz"))

        # Define the workflow
        self.add_flow(replayer, visualizer, {("output", "receivers")})

        # Check if the YAML dual_window parameter is set and add a second visualizer in that case
        dual_window = self.kwargs("dual_window").get("dual_window", False)
        if dual_window:
            visualizer2 = HolovizOp(self, name="holoviz-2", **self.kwargs("holoviz"))
            self.add_flow(replayer, visualizer2, {("output", "receivers")})


def main(config_file):
    app = VideoReplayerApp()
    # if the --config command line argument was provided, it will override this config_file
    app.config(config_file)

    dual_window = app.kwargs("dual_window").get("dual_window", False)
    if dual_window:
        from holoscan.schedulers import EventBasedScheduler

        # Use an event-based scheduler to allow multiple operators to run in
        # parallel
        app.scheduler(
            EventBasedScheduler(
                app,
                name="event-based-scheduler",
                worker_thread_number=3,
                stop_on_deadlock=True,
                stop_on_deadlock_timeout=200,
            )
        )

    app.run()


if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(__file__), "video_replayer.yaml")
    main(config_file=config_file)
