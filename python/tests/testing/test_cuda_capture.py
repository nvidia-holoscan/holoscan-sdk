# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real CUDA staging and snapshot copies without optional GPU Python packages."""

import ctypes

import pytest

import holoscan as hs
from holoscan.testing import OperatorTestHarness


@pytest.fixture
def cuda():
    try:
        runtime = ctypes.CDLL("libcudart.so.13")
    except OSError as error:
        pytest.skip(f"CUDA runtime library is unavailable: {error}")
    count = ctypes.c_int()
    if runtime.cudaGetDeviceCount(ctypes.byref(count)) != 0 or count.value == 0:
        pytest.skip("CUDA device is unavailable")
    runtime.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    runtime.cudaFree.argtypes = [ctypes.c_void_p]
    runtime.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    runtime.cudaMemsetAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    runtime.cudaMemcpyAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    runtime.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    return runtime


# REQ: PY-API-030
def test_cuda_tensor_snapshot_is_independent_and_preserves_source_device(cuda):
    class DeviceSource(hs.Operator):
        out = hs.Output("out", hs.payload.Tensor)

        def setup(self, spec):
            spec.output(self.out).produces_tensor(
                hs.tensor.OutputSpec(
                    storage="runtime_pool",
                    memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                    dtype=hs.tensor.DLDataType(1, 8, 1),
                    rank=1,
                ),
                hs.tensor.byte_span(4),
            )

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

        def compute(self, context):
            stream = context.stream().__cuda_stream__()[1]
            with self.out.allocate_tensor(shape=(4,), dtype=hs.tensor.DLDataType(1, 8, 1)) as loan:
                assert cuda.cudaMemsetAsync(loan.tensor.data, 42, 4, stream) == 0

    with OperatorTestHarness(DeviceSource) as harness:
        harness.set_output_device("out", hs.execution.DeviceId(0))
        result = harness.run()
    assert result.snapshot("out").bytes == b"****"
    assert result.snapshot("out").source_device.device_id == 0


class DeviceArray:
    def __init__(self, cuda, values):
        self.cuda = cuda
        self.pointer = ctypes.c_void_p()
        assert cuda.cudaMalloc(ctypes.byref(self.pointer), len(values)) == 0
        host = (ctypes.c_uint8 * len(values))(*values)
        assert cuda.cudaMemcpy(self.pointer, ctypes.addressof(host), len(values), 1) == 0
        self.__cuda_array_interface__ = {
            "shape": (len(values),),
            "typestr": "|u1",
            "version": 3,
            "data": (self.pointer.value, False),
            "stream": 1,
        }

    def __del__(self):
        self.cuda.cudaFree(self.pointer)


# REQ: PY-API-030
def test_staged_device_tensor_is_copied_to_the_selected_cuda_source(cuda):
    observed = []

    class DeviceSum(hs.Operator):
        inp = hs.Input("in", hs.payload.Tensor)
        out = hs.Output("out", hs.payload.uint32)

        def compute(self, context):
            tensor = self.inp.receive_data()
            observed.append(tensor.device)
            host = (ctypes.c_uint8 * 4)()
            stream = context.stream().__cuda_stream__()[1]
            assert cuda.cudaMemcpyAsync(ctypes.addressof(host), tensor.data, 4, 2, stream) == 0
            # Test-only readback establishes the bytes copied by the fixture's native source.
            assert cuda.cudaStreamSynchronize(stream) == 0
            self.out.emit(sum(host))

    with OperatorTestHarness(DeviceSum) as harness:
        harness.set_input_source_tensor_spec(
            "in",
            hs.tensor.OutputSpec(
                storage="runtime_pool",
                memory_kind=hs.tensor.MemoryKind.CUDA_DEVICE,
                dtype=hs.tensor.DLDataType(1, 8, 1),
                rank=1,
                bounds=hs.tensor.TensorBounds(max_byte_span=4),
            ),
        )
        harness.set_input_source_device("in", hs.execution.DeviceId(0))
        harness.push("in", hs.Tensor.as_tensor(DeviceArray(cuda, (1, 2, 3, 4))))
        assert harness.run().value("out") == 10
    assert observed[0].device_id == 0


# REQ: PY-API-026
def test_missing_cuda_runtime_reports_a_skip(monkeypatch):
    def missing_library(_name):
        raise OSError("loader sentinel")

    monkeypatch.setattr(ctypes, "CDLL", missing_library)
    with pytest.raises(pytest.skip.Exception, match="CUDA runtime library.*loader sentinel"):
        cuda.__wrapped__()
