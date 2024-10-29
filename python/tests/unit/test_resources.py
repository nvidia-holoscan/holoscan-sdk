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

from holoscan.core import ComponentSpec, Resource
from holoscan.core import _Resource as ResourceBase
from holoscan.gxf import GXFResource
from holoscan.operators import PingTxOp
from holoscan.resources import (
    Allocator,
    BlockMemoryPool,
    Clock,
    CudaAllocator,
    CudaStreamPool,
    DoubleBufferReceiver,
    DoubleBufferTransmitter,
    ManualClock,
    MemoryStorageType,
    RealtimeClock,
    Receiver,
    RMMAllocator,
    SerializationBuffer,
    StdComponentSerializer,
    StdEntitySerializer,
    StreamOrderedAllocator,
    Transmitter,
    UcxComponentSerializer,
    UcxEntitySerializer,
    UcxHoloscanComponentSerializer,
    UcxReceiver,
    UcxSerializationBuffer,
    UcxTransmitter,
    UnboundedAllocator,
)


class TestBlockMemoryPool:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "block-pool"
        pool = BlockMemoryPool(
            fragment=app,
            name=name,
            storage_type=MemoryStorageType.DEVICE,  # 1 = "device",
            block_size=16 * 1024**2,
            num_blocks=4,
        )
        assert isinstance(pool, Allocator)
        assert isinstance(pool, GXFResource)
        assert isinstance(pool, ResourceBase)
        assert pool.id == -1
        assert pool.gxf_typename == "nvidia::gxf::BlockMemoryPool"

        assert f"name: {name}" in repr(pool)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err

    def test_positional_initialization(self, app):
        BlockMemoryPool(app, MemoryStorageType.DEVICE, 16 * 1024**2, 4)


class TestCudaStreamPool:
    def test_default_initialization(self, app, capfd):
        name = "cuda_stream"
        pool = CudaStreamPool(fragment=app, name=name)
        assert isinstance(pool, Allocator)
        assert isinstance(pool, GXFResource)
        assert isinstance(pool, ResourceBase)
        assert pool.id == -1
        assert pool.gxf_typename == "nvidia::gxf::CudaStreamPool"
        assert f"name: {name}" in repr(pool)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err

    def test_kwarg_based_initialization(self, app, capfd):
        name = "cuda_stream"
        pool = CudaStreamPool(
            fragment=app,
            name=name,
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )
        assert isinstance(pool, Allocator)
        assert isinstance(pool, GXFResource)
        assert isinstance(pool, ResourceBase)
        assert pool.id == -1
        assert pool.gxf_typename == "nvidia::gxf::CudaStreamPool"
        assert f"name: {name}" in repr(pool)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err

    def test_positional_initialization(self, app):
        CudaStreamPool(app, 0, 0, 0, 1, 5)


class TestUnboundedAllocator:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "host_allocator"
        alloc = UnboundedAllocator(
            fragment=app,
            name=name,
        )
        assert isinstance(alloc, Allocator)
        assert isinstance(alloc, GXFResource)
        assert isinstance(alloc, ResourceBase)
        assert alloc.id == -1
        assert alloc.gxf_typename == "nvidia::gxf::UnboundedAllocator"
        assert f"name: {name}" in repr(alloc)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_default_initialization(self, app):
        UnboundedAllocator(app)


class TestRMMAllocator:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "rmm-pool"
        pool = RMMAllocator(
            fragment=app,
            name=name,
            # can specify with or without space between number and unit
            # supported units are B, KB, MB, GB, and TB.
            device_memory_initial_size="16 MB",
            device_memory_max_size="32MB",
            host_memory_initial_size="16.0 MB",
            host_memory_max_size="32768 KB",
            dev_id=0,
        )
        assert isinstance(pool, CudaAllocator)
        assert isinstance(pool, Allocator)
        assert isinstance(pool, GXFResource)
        assert isinstance(pool, ResourceBase)
        assert pool.id == -1
        assert pool.gxf_typename == "nvidia::gxf::RMMAllocator"

        assert f"name: {name}" in repr(pool)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err

    def test_default_initialization(self, app):
        RMMAllocator(app)


class TestStreamOrderedAllocator:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "stream-orered-pool"
        pool = StreamOrderedAllocator(
            fragment=app,
            name=name,
            device_memory_initial_size="16MB",
            device_memory_max_size="32MB",
            release_threshold="0B",
            dev_id=0,
        )
        assert isinstance(pool, CudaAllocator)
        assert isinstance(pool, Allocator)
        assert isinstance(pool, GXFResource)
        assert isinstance(pool, ResourceBase)
        assert pool.id == -1
        assert pool.gxf_typename == "nvidia::gxf::StreamOrderedAllocator"

        assert f"name: {name}" in repr(pool)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err

    def test_default_initialization(self, app):
        StreamOrderedAllocator(app)


class TestStdDoubleBufferReceiver:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "db-receiver"
        r = DoubleBufferReceiver(
            fragment=app,
            capacity=1,
            policy=2,
            name=name,
        )
        assert isinstance(r, Receiver)
        assert isinstance(r, GXFResource)
        assert isinstance(r, ResourceBase)
        assert r.id == -1
        assert r.gxf_typename == "nvidia::gxf::DoubleBufferReceiver"
        r.initialize()  # manually initialize so we can check resource_type
        assert r.resource_type == Resource.ResourceType.GXF
        assert f"name: {name}" in repr(r)

        # assert no unexpected warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        # expect one warning due to manually calling initialize() above
        assert captured.err.count("warning") < 2

    def test_default_initialization(self, app):
        DoubleBufferReceiver(app)


class TestStdDoubleBufferTransmitter:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "db-transmitter"
        r = DoubleBufferTransmitter(
            fragment=app,
            capacity=1,
            policy=2,
            name=name,
        )
        assert isinstance(r, Transmitter)
        assert isinstance(r, GXFResource)
        assert isinstance(r, ResourceBase)
        assert r.id == -1
        assert r.gxf_typename == "nvidia::gxf::DoubleBufferTransmitter"
        r.initialize()  # manually initialize so we can check resource_type
        assert r.resource_type == Resource.ResourceType.GXF
        assert f"name: {name}" in repr(r)

        # assert no unexpected warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        # expect one warning due to manually calling initialize() above
        assert captured.err.count("warning") < 2

    def test_default_initialization(self, app):
        DoubleBufferTransmitter(app)


class TestStdComponentSerializer:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "std-serializer"
        r = StdComponentSerializer(
            fragment=app,
            name=name,
        )
        assert isinstance(r, GXFResource)
        assert isinstance(r, ResourceBase)
        assert r.id == -1
        assert r.gxf_typename == "nvidia::gxf::StdComponentSerializer"
        assert f"name: {name}" in repr(r)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_default_initialization(self, app):
        StdComponentSerializer(app)


class TestStdEntitySerializer:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "std-entity-serializer"
        r = StdEntitySerializer(
            fragment=app,
            name=name,
        )
        assert isinstance(r, GXFResource)
        assert isinstance(r, ResourceBase)
        assert r.id == -1
        assert r.gxf_typename == "nvidia::gxf::StdEntitySerializer"
        assert f"name: {name}" in repr(r)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_default_initialization(self, app):
        StdEntitySerializer(app)


class TestManualClock:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "manual-clock"
        clk = ManualClock(
            fragment=app,
            name=name,
            initial_timestamp=100,
        )
        assert isinstance(clk, Clock)
        assert isinstance(clk, GXFResource)
        assert isinstance(clk, ResourceBase)
        assert clk.id == -1
        assert clk.gxf_typename == "nvidia::gxf::ManualClock"
        assert f"name: {name}" in repr(clk)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_positional_initialization(self, app):
        ManualClock(app, 100, "manual")


class TestRealtimeClock:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "realtime-clock"
        clk = RealtimeClock(
            fragment=app,
            name=name,
            initial_time_offset=10.0,
            initial_time_scale=2.0,
            use_time_since_epoch=True,
        )
        assert isinstance(clk, Clock)
        assert isinstance(clk, GXFResource)
        assert isinstance(clk, ResourceBase)
        assert clk.id == -1
        assert clk.gxf_typename == "nvidia::gxf::RealtimeClock"
        assert f"name: {name}" in repr(clk)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_positional_initialization(self, app):
        RealtimeClock(app, 10.0, 2.0, True, "realtime")


class TestSerializationBuffer:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "serialization_buffer"
        res = SerializationBuffer(
            fragment=app,
            allocator=UnboundedAllocator(fragment=app, name="host_allocator"),
            buffer_size=16 * 1024**2,
            name=name,
        )
        assert isinstance(res, GXFResource)
        assert isinstance(res, ResourceBase)
        assert res.id == -1  # -1 because initialize() isn't called
        assert res.gxf_typename == "nvidia::gxf::SerializationBuffer"
        assert f"name: {name}" in repr(res)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err


class TestUcxSerializationBuffer:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "ucx_serialization_buffer"
        res = UcxSerializationBuffer(
            fragment=app,
            allocator=UnboundedAllocator(fragment=app, name="host_allocator"),
            buffer_size=16 * 1024**2,
            name=name,
        )
        assert isinstance(res, GXFResource)
        assert isinstance(res, ResourceBase)
        assert res.id == -1
        assert res.gxf_typename == "nvidia::gxf::UcxSerializationBuffer"
        assert f"name: {name}" in repr(res)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err


class TestUcxComponentSerializer:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "ucx_component_serializer"
        res = UcxComponentSerializer(
            fragment=app,
            allocator=UnboundedAllocator(fragment=app, name="host_allocator"),
            name=name,
        )
        assert isinstance(res, GXFResource)
        assert isinstance(res, ResourceBase)
        assert res.id == -1
        assert res.gxf_typename == "nvidia::gxf::UcxComponentSerializer"
        assert f"name: {name}" in repr(res)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err


class TestUcxHoloscanComponentSerializer:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "ucx_holoscan_component_serializer"
        res = UcxHoloscanComponentSerializer(
            fragment=app,
            allocator=UnboundedAllocator(fragment=app, name="host_allocator"),
            name=name,
        )
        assert isinstance(res, GXFResource)
        assert isinstance(res, ResourceBase)
        assert res.id == -1
        assert res.gxf_typename == "nvidia::gxf::UcxHoloscanComponentSerializer"
        assert f"name: {name}" in repr(res)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err


class TestUcxEntitySerializer:
    def test_intialization_default_serializers(self, app, capfd):
        name = "ucx_entity_serializer"
        res = UcxEntitySerializer(
            fragment=app,
            verbose_warning=False,
            name=name,
        )
        assert isinstance(res, GXFResource)
        assert isinstance(res, ResourceBase)
        assert res.id == -1
        assert res.gxf_typename == "nvidia::gxf::UcxEntitySerializer"
        assert f"name: {name}" in repr(res)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err


class TestUcxReceiver:
    def test_kwarg_based_initialization(self, app, capfd):
        buffer = UcxSerializationBuffer(
            fragment=app,
            allocator=UnboundedAllocator(fragment=app, name="host_allocator"),
            buffer_size=16 * 1024**2,
            name="ucx_serialization_buffer",
        )
        name = "ucx_receiver"
        res = UcxReceiver(
            fragment=app,
            buffer=buffer,
            capacity=1,
            policy=2,
            address="0.0.0.0",
            port=13337,
            name=name,
        )
        assert isinstance(res, GXFResource)
        assert isinstance(res, ResourceBase)
        assert isinstance(res, Receiver)
        assert res.id == -1
        assert res.gxf_typename == "holoscan::HoloscanUcxReceiver"
        assert f"name: {name}" in repr(res)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err


class TestUcxTransmitter:
    def test_kwarg_based_initialization(self, app, capfd):
        buffer = UcxSerializationBuffer(
            fragment=app,
            allocator=UnboundedAllocator(fragment=app, name="host_allocator"),
            buffer_size=16 * 1024**2,
            name="ucx_serialization_buffer",
        )
        name = "ucx_transmitter"
        res = UcxTransmitter(
            fragment=app,
            buffer=buffer,
            capacity=1,
            policy=2,
            receiver_address="10.0.0.20",
            local_address="0.0.0.0",
            port=13337,
            local_port=0,
            maximum_connection_retries=10,
            name=name,
        )
        assert isinstance(res, GXFResource)
        assert isinstance(res, ResourceBase)
        assert isinstance(res, Transmitter)
        assert res.id == -1
        assert res.gxf_typename == "holoscan::HoloscanUcxTransmitter"
        assert f"name: {name}" in repr(res)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err


class DummyNativeResource(Resource):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: ComponentSpec):
        pass


class TestNativeResource:
    def test_native_resource_to_operator(self, app):
        """Tests passing a native resource as a positional argument to an operator."""
        tx = PingTxOp(
            app,
            DummyNativeResource(fragment=app, name="native_resource"),
            name="tx",
        )
        tx.initialize()

        # verify that the resource is included in the operator's description
        resource_repr = """
resources:
  - id: -1
    name: native_resource
"""
        assert resource_repr in tx.__repr__()
        assert resource_repr in tx.description
