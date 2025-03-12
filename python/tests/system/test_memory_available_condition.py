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

import pytest

from holoscan.conditions import CountCondition, MemoryAvailableCondition
from holoscan.core import Application
from holoscan.operators import PingTensorRxOp, PingTensorTxOp
from holoscan.resources import (
    BlockMemoryPool,
    MemoryStorageType,
    RMMAllocator,
    StreamOrderedAllocator,
    UnboundedAllocator,
)

# Define native Python Conditions used in the test applications below


class MemoryAvailableTestApp(Application):
    def __init__(
        self,
        *args,
        allocator_class,
        allocator_kwargs,
        min_bytes=None,
        min_blocks=None,
        tensor_kwargs=None,
        **kwargs,
    ):
        self.allocator_class = allocator_class
        self.allocator_kwargs = allocator_kwargs
        self.min_bytes = min_bytes
        self.min_blocks = min_blocks
        if tensor_kwargs is None:
            tensor_kwargs = dict(
                rows=2048,
                columns=2048,
                channels=3,
                dtype="uint8_t",
            )
        self.tensor_kwargs = tensor_kwargs
        super().__init__(*args, **kwargs)

    def compose(self):
        # Configure the operators. Here we use CountCondition to terminate
        # execution after a specific number of messages have been sent.
        # PeriodicCondition is used so that each subsequent message is
        # sent only after a period of 200 milliseconds has elapsed.

        allocator = self.allocator_class(self, **self.allocator_kwargs)
        mem_avail = MemoryAvailableCondition(
            self,
            allocator=allocator,
            min_bytes=self.min_bytes,
            min_blocks=self.min_blocks,
            name="mem_avail_tx",
        )
        tx = PingTensorTxOp(
            self,
            CountCondition(self, 5),
            mem_avail,
            allocator=allocator,
            storage_type="device",
            **self.tensor_kwargs,
            name="tx",
        )
        rx = PingTensorRxOp(self, name="rx")

        # Connect the operators into the workflow:  tx -> rx
        self.add_flow(tx, rx)


def test_memory_available_unbounded(capfd):
    app = MemoryAvailableTestApp(
        allocator_class=UnboundedAllocator,
        allocator_kwargs=dict(name="unbounded_alloc"),
        min_bytes=1_000_000,
        min_blocks=None,
    )
    app.run()

    # assert that the expected number of messages were received
    captured = capfd.readouterr()
    assert "rx received message 5" in captured.err


@pytest.mark.parametrize(
    "num_blocks, min_blocks, min_bytes, expect_success",
    [
        # pass: request memory equal to block size with 1 block
        (1, None, 2048 * 2048 * 3, True),
        # pass: check for 1 block from allocator with 1 block
        (1, 1, None, True),
        # fail: request 1 more byte than block_size when there is only 1 block
        (1, None, 2048 * 2048 * 3 + 1, False),
        # pass: request larger than block size, but there is a 2nd block available
        (2, None, 2048 * 2048 * 3 + 1, True),
        # fail: check availability of 3 blocks from allocator with 2 blocks
        (2, 3, None, False),
    ],
)
def test_memory_available_block(num_blocks, min_blocks, min_bytes, expect_success, capfd):
    rows = 2048
    columns = 2048
    channels = 3
    bytes_per_element = 1
    nbytes_per_tensor = rows * columns * channels * bytes_per_element
    tensor_kwargs = dict(rows=2048, columns=2048, channels=3, dtype="uint8_t")
    app = MemoryAvailableTestApp(
        allocator_class=BlockMemoryPool,
        allocator_kwargs=dict(
            name="block_alloc",
            storage_type=MemoryStorageType.DEVICE,
            block_size=nbytes_per_tensor,
            num_blocks=num_blocks,
        ),
        min_bytes=min_bytes,
        min_blocks=min_blocks,
        tensor_kwargs=tensor_kwargs,
    )
    app.run()

    # assert that the expected number of messages were received
    captured = capfd.readouterr()
    if expect_success:
        assert "rx received message 5" in captured.err
        assert "error" not in captured.err
    else:
        assert "rx received message 1" not in captured.err
        assert "rx received message 5" not in captured.err
        assert "error" not in captured.err
        assert "deadlock" in captured.err


@pytest.mark.parametrize(
    "min_bytes, expect_success",
    [
        # request valid number of bytes or blocks (based on app below)
        (2048 * 2048 * 3, True),
        # request more bytes or blocks than available
        #
        # Note: regardless of user setting for `device_memory_max_size`, the
        # pool was observed to always have at least 32MB available, so to fail
        # I had to choose a min_bytes value larger than 32 MB. I couldn't find
        # it explicitly documented in the CUDA docs, but it seems that whatever
        # maximum memory is requested is rounded up to the next largest integer
        # multiple of 32 MB.
        (2048 * 2048 * 16, False),
    ],
)
def test_memory_available_cuda_stream_ordered(min_bytes, expect_success, capfd):
    rows = 2048
    columns = 2048
    channels = 3
    bytes_per_element = 1
    nbytes_per_tensor = rows * columns * channels * bytes_per_element
    tensor_kwargs = dict(rows=2048, columns=2048, channels=3, dtype="uint8_t")
    app = MemoryAvailableTestApp(
        allocator_class=StreamOrderedAllocator,
        allocator_kwargs=dict(
            name="stream_ordered_alloc",
            device_memory_initial_size=f"{nbytes_per_tensor}B",
            device_memory_max_size=f"{nbytes_per_tensor}B",
            dev_id=0,
        ),
        min_bytes=min_bytes,
        min_blocks=None,
        tensor_kwargs=tensor_kwargs,
    )
    app.run()

    # assert that the expected number of messages were received
    captured = capfd.readouterr()
    if expect_success:
        assert "rx received message 5" in captured.err
        assert "error" not in captured.err
    else:
        assert "rx received message 1" not in captured.err
        assert "rx received message 5" not in captured.err
        assert "error" not in captured.err
        assert "deadlock" in captured.err


def test_memory_available_rmm(capfd):
    rows = 2048
    columns = 2048
    channels = 3
    bytes_per_element = 1
    nbytes_per_tensor = rows * columns * channels * bytes_per_element
    tensor_kwargs = dict(rows=2048, columns=2048, channels=3, dtype="uint8_t")
    app = MemoryAvailableTestApp(
        allocator_class=RMMAllocator,
        allocator_kwargs=dict(
            name="stream_ordered_alloc",
            device_memory_initial_size=f"{nbytes_per_tensor}B",
            device_memory_max_size=f"{nbytes_per_tensor}B",
            dev_id=0,
        ),
        min_bytes=4 * nbytes_per_tensor,
        min_blocks=None,
        tensor_kwargs=tensor_kwargs,
    )
    app.run()

    # assert that the expected number of messages were received
    captured = capfd.readouterr()

    # RMMAllocator does not support he API needed by MemoryAvailableCondition
    assert "warning" in captured.err
    assert "RMM allocator does not support this API"

    # app will run despite min_bytes condition not being satisfied
    assert "rx received message 5" in captured.err
    assert "error" not in captured.err
