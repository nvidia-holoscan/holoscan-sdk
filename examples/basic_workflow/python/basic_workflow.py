"""
SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from holoscan.core import Application
from holoscan.logger import load_env_log_level
from holoscan.operators import (
    FormatConverterOp,
    HolovizOp,
    LSTMTensorRTInferenceOp,
    ToolTrackingPostprocessorOp,
    VideoStreamRecorderOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import (
    BlockMemoryPool,
    CudaStreamPool,
    MemoryStorageType,
    UnboundedAllocator,
)

sample_data_path = os.environ.get("HOLOSCAN_SAMPLE_DATA_PATH", "../../../data")


class EndoscopyApp(Application):

    def compose(self):
        width = 854
        height = 480
        video_dir = os.path.join(sample_data_path, "endoscopy", "video")
        if not os.path.exists(video_dir):
            raise ValueError(f"Could not find video data: {video_dir=}")
        replayer = VideoStreamReplayerOp(
            self,
            name="replayer",
            directory=video_dir,
            **self.kwargs("replayer"),
        )
        # 4 bytes/channel, 3 channels
        source_pool_kwargs = dict(
            storage_type=MemoryStorageType.DEVICE,
            block_size=width * height * 3 * 4,
            num_blocks=2,
        )
        recorder = VideoStreamRecorderOp(
            name="recorder", fragment=self, **self.kwargs("recorder")
        )

        format_converter = FormatConverterOp(
            self,
            name="format_converter_replayer",
            pool=BlockMemoryPool(self, name="pool", **source_pool_kwargs),
            **self.kwargs("format_converter_replayer"),
        )

        lstm_cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )
        model_file_path = os.path.join(
            sample_data_path, "endoscopy", "model", "tool_loc_convlstm.onnx"
        )
        engine_cache_dir = os.path.join(
            sample_data_path, "endoscopy", "model", "tool_loc_convlstm_engines"
        )
        lstm_inferer = LSTMTensorRTInferenceOp(
            self,
            name="lstm_inferer",
            pool=UnboundedAllocator(self, name="pool"),
            cuda_stream_pool=lstm_cuda_stream_pool,
            model_file_path=model_file_path,
            engine_cache_dir=engine_cache_dir,
            **self.kwargs("lstm_inference"),
        )

        tool_tracking_postprocessor_block_size = 107 * 60 * 7 * 4
        tool_tracking_postprocessor_num_blocks = 2
        tool_tracking_postprocessor = ToolTrackingPostprocessorOp(
            self,
            name="tool_tracking_postprocessor",
            device_allocator=BlockMemoryPool(
                self,
                name="device_allocator",
                storage_type=MemoryStorageType.DEVICE,
                block_size=tool_tracking_postprocessor_block_size,
                num_blocks=tool_tracking_postprocessor_num_blocks,
            ),
            host_allocator=UnboundedAllocator(self, name="host_allocator"),
        )

        visualizer = HolovizOp(
            self, name="holoviz", width=width, height=height, **self.kwargs("holoviz"),
        )

        # Flow definition
        self.add_flow(replayer, visualizer, {("output", "receivers")})

        self.add_flow(replayer, format_converter)
        self.add_flow(format_converter, lstm_inferer)
        self.add_flow(lstm_inferer, tool_tracking_postprocessor, {("tensor", "in")})
        self.add_flow(tool_tracking_postprocessor, visualizer, {("out", "receivers")})

        self.add_flow(replayer, recorder)


if __name__ == "__main__":

    load_env_log_level()

    config_file = os.path.join(os.path.dirname(__file__), "basic_workflow.yaml")

    app = EndoscopyApp()
    app.config(config_file)
    app.run()
