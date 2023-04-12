# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from holoscan.core import Resource
from holoscan.gxf import GXFResource
from holoscan.resources import (
    Allocator,
    BlockMemoryPool,
    CudaStreamPool,
    DoubleBufferReceiver,
    DoubleBufferTransmitter,
    MemoryStorageType,
    Receiver,
    StdComponentSerializer,
    Transmitter,
    UnboundedAllocator,
    VideoStreamSerializer,
)


class TestBlockMemoryPool:
    def test_kwarg_based_initialization(self, app, capfd):
        pool = BlockMemoryPool(
            fragment=app,
            name="pool",
            storage_type=MemoryStorageType.DEVICE,  # 1 = "device",
            block_size=16 * 1024**2,
            num_blocks=4,
        )
        assert isinstance(pool, Allocator)
        assert isinstance(pool, GXFResource)
        assert isinstance(pool, Resource)
        assert pool.id != -1
        assert pool.gxf_typename == "nvidia::gxf::BlockMemoryPool"

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_positional_initialization(self, app):
        BlockMemoryPool(app, MemoryStorageType.DEVICE, 16 * 1024**2, 4)


class TestCudaStreamPool:
    def test_kwarg_based_initialization(self, app, capfd):
        pool = CudaStreamPool(
            fragment=app,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )
        assert isinstance(pool, Allocator)
        assert isinstance(pool, GXFResource)
        assert isinstance(pool, Resource)
        assert pool.id != -1
        assert pool.gxf_typename == "nvidia::gxf::CudaStreamPool"

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_positional_initialization(self, app):
        CudaStreamPool(app, 0, 0, 0, 1, 5)


class TestUnboundedAllocator:
    def test_kwarg_based_initialization(self, app, capfd):
        alloc = UnboundedAllocator(
            fragment=app,
            name="host_allocator",
        )
        assert isinstance(alloc, Allocator)
        assert isinstance(alloc, GXFResource)
        assert isinstance(alloc, Resource)
        assert alloc.id != -1
        assert alloc.gxf_typename == "nvidia::gxf::UnboundedAllocator"

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_default_initialization(self, app):
        UnboundedAllocator(app)


class TestStdDoubleBufferReceiver:
    def test_kwarg_based_initialization(self, app, capfd):
        r = DoubleBufferReceiver(
            fragment=app,
            capacity=1,
            policy=2,
            name="host_allocator",
        )
        assert isinstance(r, Receiver)
        assert isinstance(r, GXFResource)
        assert isinstance(r, Resource)
        assert r.id != -1
        assert r.gxf_typename == "nvidia::gxf::DoubleBufferReceiver"

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_default_initialization(self, app):
        DoubleBufferReceiver(app)


class TestStdDoubleBufferTransmitter:
    def test_kwarg_based_initialization(self, app, capfd):
        r = DoubleBufferTransmitter(
            fragment=app,
            capacity=1,
            policy=2,
            name="host_allocator",
        )
        assert isinstance(r, Transmitter)
        assert isinstance(r, GXFResource)
        assert isinstance(r, Resource)
        assert r.id != -1
        assert r.gxf_typename == "nvidia::gxf::DoubleBufferTransmitter"

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_default_initialization(self, app):
        DoubleBufferTransmitter(app)


class TestStdComponentSerializer:
    def test_kwarg_based_initialization(self, app, capfd):
        r = StdComponentSerializer(
            fragment=app,
            name="host_allocator",
        )
        assert isinstance(r, GXFResource)
        assert isinstance(r, Resource)
        assert r.id != -1
        assert r.gxf_typename == "nvidia::gxf::StdComponentSerializer"

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_default_initialization(self, app):
        StdComponentSerializer(app)


class TestVideoStreamSerializer:
    def test_kwarg_based_initialization(self, app, capfd):
        r = VideoStreamSerializer(
            fragment=app,
            name="host_allocator",
        )
        assert isinstance(r, GXFResource)
        assert isinstance(r, Resource)
        assert r.id != -1
        assert r.gxf_typename == "nvidia::holoscan::stream_playback::VideoStreamSerializer"

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_default_initialization(self, app):
        VideoStreamSerializer(app)
