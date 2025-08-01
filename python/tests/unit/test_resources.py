"""
SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from holoscan.conditions import CountCondition
from holoscan.core import ComponentSpec, Resource
from holoscan.core import _Resource as ResourceBase
from holoscan.gxf import GXFResource, GXFSystemResourceBase
from holoscan.operators import PingRxOp, PingTxOp
from holoscan.resources import (
    Allocator,
    AsyncBufferReceiver,
    AsyncBufferTransmitter,
    BlockMemoryPool,
    Clock,
    CudaAllocator,
    CudaStreamPool,
    DoubleBufferReceiver,
    DoubleBufferTransmitter,
    ManualClock,
    MemoryStorageType,
    OrConditionCombiner,
    RealtimeClock,
    Receiver,
    RMMAllocator,
    SchedulingPolicy,
    SerializationBuffer,
    StdComponentSerializer,
    StdEntitySerializer,
    StreamOrderedAllocator,
    ThreadPool,
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
            release_threshold="1MB",
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
        assert "warning" not in captured.err
        assert "not found in spec_.params()" not in captured.err

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


class TestStdAsyncBufferReceiver:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "async-receiver"
        r = AsyncBufferReceiver(
            fragment=app,
            name=name,
        )
        assert isinstance(r, Receiver)
        assert isinstance(r, GXFResource)
        assert isinstance(r, ResourceBase)
        assert r.id == -1
        assert r.gxf_typename == "holoscan::HoloscanAsyncBufferReceiver"
        r.initialize()  # manually initialize so we can check resource_type
        assert r.resource_type == Resource.ResourceType.GXF
        assert f"name: {name}" in repr(r)

        # assert no unexpected warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        # expect one warning due to manually calling initialize() above
        assert captured.err.count("warning") < 2

    def test_default_initialization(self, app):
        AsyncBufferReceiver(app)


class TestStdAsyncBufferTransmitter:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "async-transmitter"
        r = AsyncBufferTransmitter(
            fragment=app,
            name=name,
        )
        assert isinstance(r, Transmitter)
        assert isinstance(r, GXFResource)
        assert isinstance(r, ResourceBase)
        assert r.id == -1
        assert r.gxf_typename == "holoscan::HoloscanAsyncBufferTransmitter"
        r.initialize()  # manually initialize so we can check resource_type
        assert r.resource_type == Resource.ResourceType.GXF
        assert f"name: {name}" in repr(r)

        # assert no unexpected warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        # expect one warning due to manually calling initialize() above
        assert captured.err.count("warning") < 2

    def test_default_initialization(self, app):
        AsyncBufferTransmitter(app)


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


class TestThreadPool:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "my_thread_pool"
        pool = ThreadPool(
            fragment=app,
            name=name,
            initial_size=1,
        )
        assert isinstance(pool, GXFSystemResourceBase)
        assert isinstance(pool, GXFResource)
        assert isinstance(pool, ResourceBase)
        assert pool.id == -1
        assert pool.gxf_typename == "nvidia::gxf::ThreadPool"

        assert f"name: {name}" in repr(pool)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err

    def test_default_initialization(self, app):
        ThreadPool(fragment=app)

    def test_add_realtime_method_fifo(self, app):
        tx_op = PingTxOp(app, CountCondition(app, 5), name="tx")
        rx_op = PingRxOp(app, name="rx")

        # Test add_realtime with SCHED_FIFO real-time scheduling policy
        pool = ThreadPool(fragment=app, initial_size=2, name="test_pool")
        pool.add_realtime(
            tx_op,
            SchedulingPolicy.SCHED_FIFO,
            pin_operator=True,
            pin_cores=[0, 2],
            sched_priority=1,
        )

        # Test regular add with pin_cores
        pool.add(rx_op, pin_operator=True, pin_cores=[1, 3])

        assert len(pool.operators) == 2
        assert tx_op in pool.operators
        assert rx_op in pool.operators

    def test_add_realtime_method_rr(self, app):
        tx_op = PingTxOp(app, CountCondition(app, 5), name="tx")
        rx_op = PingRxOp(app, name="rx")

        # Test add_realtime with SCHED_RR real-time scheduling policy
        pool = ThreadPool(fragment=app, initial_size=2, name="test_pool")
        pool.add_realtime(
            tx_op,
            SchedulingPolicy.SCHED_RR,
            pin_operator=True,
            pin_cores=[0, 2],
            sched_priority=2,
        )

        # Test regular add with pin_cores
        pool.add(rx_op, pin_operator=True, pin_cores=[1, 3])

        assert len(pool.operators) == 2
        assert tx_op in pool.operators
        assert rx_op in pool.operators

    def test_add_realtime_method_deadline(self, app):
        tx_op = PingTxOp(app, CountCondition(app, 5), name="tx")
        rx_op = PingRxOp(app, name="rx")

        # Test add_realtime with SCHED_DEADLINE real-time scheduling policy
        pool = ThreadPool(fragment=app, initial_size=2, name="test_pool")
        pool.add_realtime(
            tx_op,
            SchedulingPolicy.SCHED_DEADLINE,
            pin_operator=True,
            pin_cores=[0, 2],
            sched_runtime=1000000,
            sched_deadline=1000000000,
            sched_period=1000000000,
        )

        # Test regular add with pin_cores
        pool.add(rx_op, pin_operator=True, pin_cores=[1, 3])

        assert len(pool.operators) == 2
        assert tx_op in pool.operators
        assert rx_op in pool.operators

    def test_add_with_pin_cores_parameter(self, app):
        pool = ThreadPool(fragment=app, initial_size=2, name="test_pool")
        tx_op = PingTxOp(app, CountCondition(app, 5), name="tx")

        # Test backward compatibility - no pin_cores
        pool.add(tx_op, pin_operator=True)

        # Test with pin_cores parameter
        pool.add(tx_op, pin_operator=True, pin_cores=[0, 2])

        assert len(pool.operators) == 2  # Same operator added twice


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


class TestOrConditionCombiner:
    def test_kwarg_based_initialization(self, app, capfd):
        name = "or-condition-combiner"
        count1 = CountCondition(app, count=10, name="count1")
        count2 = CountCondition(app, count=20, name="count2")
        count3 = CountCondition(app, count=30, name="count3")
        combiner = OrConditionCombiner(
            fragment=app,
            name=name,
            terms=[count1, count2, count3],
        )
        assert isinstance(combiner, GXFResource)
        assert isinstance(combiner, ResourceBase)
        assert combiner.id == -1
        assert combiner.gxf_typename == "nvidia::gxf::OrSchedulingTermCombiner"

        assert f"name: {name}" in repr(combiner)

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err


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
