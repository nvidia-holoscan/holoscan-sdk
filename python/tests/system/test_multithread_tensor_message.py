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

import sys

import cupy as cp
import pytest
from env_wrapper import env_var_context

import holoscan
from holoscan.conditions import CountCondition
from holoscan.operators import FormatConverterOp
from holoscan.resources import UnboundedAllocator
from holoscan.schedulers import EventBasedScheduler, MultiThreadScheduler

cuda_device = cp.cuda.Device()
# disable CuPy memory pool
cp.cuda.set_allocator(None)

GPU_MEMORY_HISTORY = []
NUM_MSGS = 200


class ImageGenerator(holoscan.core.Operator):
    def setup(self, spec):
        spec.output("image")

    def generate_random_image(self, width, height):
        image = cp.random.randint(0, 256, (height, width, 3), dtype=cp.uint8)
        return image

    def compute(self, op_input, op_output, context):
        width = 854
        height = 480
        img = self.generate_random_image(width, height)
        op_output.emit({"": img}, "image")


class WatchdogOperator(holoscan.core.Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.index = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        print("Watchdog started", file=sys.stderr)
        op_input.receive("input")
        self.index += 1
        print(f"Watchdog received data (count: {self.index})", file=sys.stderr)
        print("Watchdog finished", file=sys.stderr)


class PostProcessorOp(holoscan.core.Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.index = 0
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec):
        spec.input("tensor")

    def compute(self, op_input, op_output, context):
        global cuda_device
        print("PostProcessor started", file=sys.stderr)
        op_input.receive("tensor")
        self.index += 1
        print(f"PostProcessor received data (count: {self.index})", file=sys.stderr)
        print("PostProcessor finished", file=sys.stderr)
        # https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Device.html#cupy.cuda.Device.mem_info  # noqa: E501
        GPU_MEMORY_HISTORY.append(cuda_device.mem_info[1])


class MultithreadTensorSenderApp(holoscan.core.Application):
    def compose(self):
        image_generator = ImageGenerator(
            self,
            CountCondition(self, NUM_MSGS),
            name="image_generator",
        )

        host_allocator = UnboundedAllocator(self, name="host_allocator")

        preprocessor = FormatConverterOp(
            self, name="preprocessor", pool=host_allocator, out_dtype="rgb888"
        )

        postprocessor = PostProcessorOp(self, name="postprocessor")

        watchdog = WatchdogOperator(self, name="watchdog")

        # Define the workflow
        self.add_flow(image_generator, preprocessor, {("image", "source_video")})
        self.add_flow(preprocessor, postprocessor, {("tensor", "tensor")})
        self.add_flow(preprocessor, watchdog, {("tensor", "input")})


# define the number of messages to send
NUM_MSGS = 200
GPU_USAGE_TOLERANCE = 30_000_000  # 30 MiB
# with 854x480x3 image, the GPU memory usage should be stable
# (the GPU memory usage should not increase with the number of messages)
# the GPU memory usage should be less than 30 MiB
# #msgs: before fix => after fix (issue 4293741)
#     5:    6291456 => 2097152
#    10:   14680064 => 4194304
#    50:   98566144 => 5570560
#   100:  190119936 => 6291456
#   200:  414515200 => 5570560
#   500: 1045757952 => 5636096
#  1000: 2094333952 => 5636096, 18874368 (with some background processes running)


def launch_app(scheduler="multi_thread"):
    env_var_settings = {
        # set the recession period to 100 ms to reduce debug messages
        ("HOLOSCAN_CHECK_RECESSION_PERIOD_MS", "100"),
        ("HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT", "500"),
    }

    with env_var_context(env_var_settings):
        app = MultithreadTensorSenderApp()

        if scheduler == "multi_thread":
            scheduler = MultiThreadScheduler(
                app,
                worker_thread_number=4,
                stop_on_deadlock=True,
                stop_on_deadlock_timeout=500,
                check_recession_period_ms=0.0,
                name="multithread_scheduler",
            )
        elif scheduler == "event_based":
            scheduler = EventBasedScheduler(
                app,
                worker_thread_number=4,
                stop_on_deadlock=True,
                stop_on_deadlock_timeout=500,
                name="ebs",
            )
        else:
            raise ValueError("scheduler must be one of {'multi_thread', 'event_based'}")

        app.scheduler(scheduler)
        app.run()


@pytest.mark.parametrize("scheduler", ["event_based", "multi_thread"])
def test_multithread_tensor_message(scheduler, capfd):
    # Issue 4293741: Python application having more than two operators, using MultiThreadScheduler
    # (including distributed app), and sending Tensor can deadlock at runtime.
    global NUM_MSGS, GPU_MEMORY_HISTORY

    launch_app(scheduler)

    # assert that no errors were logged
    captured = capfd.readouterr()

    assert "error" not in captured.err
    assert "Exception occurred" not in captured.err

    # assert that the expected number of messages were received
    assert f"PostProcessor received data (count: {NUM_MSGS})" in captured.err
    # remove zeros from the list (somehow it happens that some values are 0)
    GPU_MEMORY_HISTORY = [x for x in GPU_MEMORY_HISTORY if x != 0]
    # assert that the GPU memory usage is stable
    assert len(GPU_MEMORY_HISTORY) > 0
    # delta between min and max should be less than GPU_USAGE_TOLERANCE
    # (it can be around 400MB with 200 messages if the GPU memory is not released properly)
    assert max(GPU_MEMORY_HISTORY) - min(GPU_MEMORY_HISTORY) < GPU_USAGE_TOLERANCE


if __name__ == "__main__":
    launch_app()
