"""
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

try:
    import cupy as cp

    has_cupy = True
except ImportError:
    has_cupy = False

from holoscan.conditions import CountCondition
from holoscan.core import Application, ConditionType, Operator, OperatorSpec
from holoscan.operators import PingTensorTxOp
from holoscan.resources import CudaStreamPool


class StreamRxOp(Operator):
    def __init__(self, fragment, *args, has_stream_pool=False, **kwargs):
        self.index = 0
        self.has_stream_pool = has_stream_pool
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        # use unconnected output port just to have a port for set_cuda_stream
        spec.output("out").condition(ConditionType.NONE)

    def compute(self, op_input, op_output, context):
        tensors = op_input.receive("in")
        assert "tensor" in tensors

        stream = op_input.receive_cuda_stream("in")
        print(f"{stream=}")
        assert stream != 0

        # expect that the stream corresponds to a device
        device_id = context.device_from_stream(stream)
        assert device_id is not None
        assert device_id >= 0

        streams = op_input.receive_cuda_streams("in")
        assert len(streams) == 1
        if self.has_stream_pool:
            # stream will be the operator's internal stream not the received one
            assert stream != streams[0]
        else:
            # stream will be the same as the received stream
            assert stream == streams[0]

        # assert streams[0] is not None
        print(f"{streams=}")

        # This operator doesn't have a cuda_stream_pool_ parameter, so will
        # get back the default stream.
        new_stream = context.allocate_cuda_stream("my_stream")
        if self.has_stream_pool:
            assert new_stream != 0
        else:
            assert new_stream == 0

        # emit an additional stream on the "out" port
        if self.has_stream_pool:
            op_output.set_cuda_stream(new_stream, "out")


class MyStreamTestApp(Application):
    def __init__(self, *args, add_rx_stream_pool=False, count=10, **kwargs):
        self.count = count
        self.add_rx_stream_pool = add_rx_stream_pool
        super().__init__(*args, **kwargs)

    def compose(self):
        stream_pool = CudaStreamPool(
            self,
            name="stream_pool",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )
        # tx op will only add the stream if all three of these are true
        #    1.) storage_type == "device"
        #    2.) a cuda_stream_pool is set
        #    3.) async_device_allocation == True
        tx = PingTensorTxOp(
            self,
            CountCondition(self, self.count),
            storage_type="device",
            cuda_stream_pool=stream_pool,
            async_device_allocation=True,
            name="tx",
        )
        args = (stream_pool,) if self.add_rx_stream_pool else ()
        rx = StreamRxOp(self, *args, has_stream_pool=self.add_rx_stream_pool, name="stream_rx")
        self.add_flow(tx, rx)


class MyCuPyExternalStreamApp(Application):
    def __init__(self, *args, count=10, **kwargs):
        self.count = count
        super().__init__(*args, **kwargs)

    def compose(self):
        stream_pool = CudaStreamPool(
            self,
            name="stream_pool",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )
        tx = PingTensorTxOp(
            self,
            CountCondition(self, self.count),
            storage_type="device",
            dtype="int32_t",
            cuda_stream_pool=stream_pool,
            async_device_allocation=True,
            name="tx",
        )
        rx = CuPyExternalStreamOp(self, stream_pool, name="stream_rx")
        self.add_flow(tx, rx)


class CuPyExternalStreamOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.index = 0
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        tensors = op_input.receive("in")

        stream = op_input.receive_cuda_stream("in")
        print(f"Received message {self.index}: {stream=}")
        assert stream != 0

        for tensor_name, tensor in tensors.items():
            # For PyTorch, It should work similarly to use:
            #     with torch.cuda.StreamContext(torch.cuda.ExternalStream(stream)):
            with cp.cuda.ExternalStream(stream, device_id=context.device_from_stream(stream)):
                cp_tensor = cp.asarray(tensor)

                # Need to set some value before the sum as PingTensorTxOp does not initialize the
                # tensor's values.
                cp_tensor[:] = 2
                s = cp_tensor.sum()
            print(f"{tensor_name=}, sum={s}")
            assert s == 2 * cp_tensor.size
        self.index += 1


class PingTxResourceCheckOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        if "default_cuda_stream_pool" in self.resources:
            print("tx: default CudaStreamPool found")
            assert isinstance(self.resources["default_cuda_stream_pool"], CudaStreamPool)
        else:
            print("tx: CudaStreamPool not found")
        op_output.emit(0, "out")


class PingRxResourceCheckOp(Operator):
    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        value = op_input.receive("in")
        print(f"rx message value: {value}")

        if "default_cuda_stream_pool" in self.resources:
            print("rx: default CudaStreamPool found")
            assert isinstance(self.resources["default_cuda_stream_pool"], CudaStreamPool)
        else:
            print("rx: CudaStreamPool not found")


# Trivial ping app just to test that a default CUDA stream pool has been added
class DefaultCudaStreamPoolApp(Application):
    def __init__(self, *args, count=10, **kwargs):
        self.count = count
        super().__init__(*args, **kwargs)

    def compose(self):
        tx = PingTxResourceCheckOp(self, CountCondition(self, count=1), name="tx")
        rx = PingRxResourceCheckOp(self, name="rx")
        self.add_flow(tx, rx)


@pytest.mark.parametrize("add_rx_stream_pool", [False, True])
def test_stream_pool_methods(add_rx_stream_pool):
    try:
        cp.cuda.Device()
    except RuntimeError:
        pytest.skip("no available CUDA device: skipping stream test")
    app = MyStreamTestApp(add_rx_stream_pool=add_rx_stream_pool)
    app.run()


@pytest.mark.skipif(not has_cupy, reason="CuPy is not installed")
def test_cupy_external_stream():
    try:
        cp.cuda.Device()
    except RuntimeError:
        pytest.skip("no available CUDA device: skipping stream test")
    app = MyCuPyExternalStreamApp()
    app.run()


@pytest.mark.skipif(not has_cupy, reason="CuPy is not installed")
def test_default_cuda_stream_pool(capfd):
    """Test for automatically added CudaStreamPool

    We can test the no-device case by running the following from the build folder:

      CUDA_VISIBLE_DEVICES="" python -m pytest ./python/lib/tests/system/test_stream_handling.py

    Cannot use `env_var_context` from `env_wrapper.py` to change CUDA_VISIBLE_DEVICES dynamically
    at run time because it must be set before the CUDA driver has been initialized (e.g. before
    CuPy import).
    """
    try:
        cp.cuda.Device()
        has_device = True
    except RuntimeError:
        has_device = False

    app = DefaultCudaStreamPoolApp()
    app.run()

    captured = capfd.readouterr()

    if has_device:
        assert "rx: default CudaStreamPool found" in captured.out
        assert "rx message value: 0" in captured.out
        assert "tx: default CudaStreamPool found" in captured.out
    else:
        assert "rx: CudaStreamPool not found" in captured.out
        assert "rx message value: 0" in captured.out
        assert "tx: CudaStreamPool not found" in captured.out
