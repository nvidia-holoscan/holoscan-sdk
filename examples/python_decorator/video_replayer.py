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

import os

from holoscan.core import Application
from holoscan.decorator import Input, Output, create_op
from holoscan.operators import HolovizOp, VideoStreamReplayerOp

sample_data_path = os.environ.get("HOLOSCAN_INPUT_PATH", "../data")


@create_op(
    inputs="tensor",
    outputs="out_tensor",
)
def invert(tensor):
    tensor = 255 - tensor
    return tensor


@create_op(inputs=Input("in", arg_map="tensor"), outputs=Output("out", tensor_names=("frame",)))
def tensor_info(tensor):
    print(f"tensor from 'in' port: shape = {tensor.shape}, " f"dtype = {tensor.dtype.name}")
    return tensor


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

        # Define the replayer and holoviz operators
        replayer = VideoStreamReplayerOp(
            self,
            name="replayer",
            directory=video_dir,
            basename="racerx",
            frame_rate=0,  # as specified in timestamps
            repeat=False,  # default: false
            realtime=True,  # default: true
            count=40,  # default: 0 (no frame count restriction)
        )
        invert_op = invert(self, name="image_invert")
        info_op = tensor_info(self, name="tensor_info")
        visualizer = HolovizOp(
            self,
            name="holoviz",
            width=854,
            height=480,
            # name="frame" to match Output argument to create_op for tensor_info
            tensors=[dict(name="frame", type="color", opacity=1.0, priority=0)],
        )
        # Define the workflow
        self.add_flow(replayer, invert_op, {("output", "tensor")})
        self.add_flow(invert_op, info_op, {("out_tensor", "in")})
        self.add_flow(info_op, visualizer, {("out", "receivers")})


def main():
    app = VideoReplayerApp()
    app.run()


if __name__ == "__main__":
    main()
