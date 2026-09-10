# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 1 (metadata-only) tests for the nanobind Tensor bindings.

DLDevice and DLDataType are constructible and inspectable from Python; Tensor is exposed as a type
but is not yet constructible from Python (no factory until the DLPack import/export phases), so here
we only assert the type and its expected property surface. Metadata round-trip tests
(Tensor.as_tensor(np_array) -> shape/strides/dtype/...) arrive with the import phase.
"""

import re

import pytest

import holoscan.core
from holoscan.core import (
    DLDataType,
    DLDataTypeCode,
    DLDevice,
    DLDeviceType,
    Tensor,
    is_tensor_like,
)
from holoscan.core._core import _test_dlpack_c_exchange_api

# DLDataTypeCode raw value (DLDataType.code stays a plain int field, matching DLPack/4.x).
_KDLFLOAT = 2


def _torch_version_at_least(torch, major: int, minor: int) -> bool:
    match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
    if match is None:
        return False
    return (int(match.group(1)), int(match.group(2))) >= (major, minor)


def _skip_without_cuda_devices(cupy, count):
    if cupy.cuda.runtime.getDeviceCount() < count:
        pytest.skip(f"at least {count} CUDA device(s) required")


class TestDLEnums:
    def test_device_type_values(self):
        assert int(DLDeviceType.DLCPU) == 1
        assert int(DLDeviceType.DLCUDA) == 2
        assert int(DLDeviceType.DLCUDAMANAGED) == 13

    def test_data_type_code_values(self):
        expected = {
            "DLINT": 0,
            "DLUINT": 1,
            "DLFLOAT": 2,
            "DLOPAQUEHANDLE": 3,
            "DLBFLOAT": 4,
            "DLCOMPLEX": 5,
            "DLBOOL": 6,
            "DLFLOAT8_E3M4": 7,
            "DLFLOAT8_E4M3": 8,
            "DLFLOAT8_E4M3B11FNUZ": 9,
            "DLFLOAT8_E4M3FN": 10,
            "DLFLOAT8_E4M3FNUZ": 11,
            "DLFLOAT8_E5M2": 12,
            "DLFLOAT8_E5M2FNUZ": 13,
            "DLFLOAT8_E8M0FNU": 14,
            "DLFLOAT6_E2M3FN": 15,
            "DLFLOAT6_E3M2FN": 16,
            "DLFLOAT4_E2M1FN": 17,
        }
        for name, value in expected.items():
            assert int(getattr(DLDataTypeCode, name)) == value


class TestDLDevice:
    def test_construct_and_fields(self):
        device = DLDevice(DLDeviceType.DLCUDA, 3)
        assert device.device_type == DLDeviceType.DLCUDA
        assert device.device_id == 3

    def test_device_type_is_enum(self):
        assert isinstance(DLDevice(DLDeviceType.DLCPU, 0).device_type, DLDeviceType)

    def test_repr(self):
        cpu = DLDevice(DLDeviceType.DLCPU, 0)
        cuda = DLDevice(DLDeviceType.DLCUDA, 2)
        assert repr(cpu) == "DLDevice(device_type=kDLCPU, device_id=0)"
        assert repr(cuda) == "DLDevice(device_type=kDLCUDA, device_id=2)"


class TestDLDataType:
    def test_construct_and_fields(self):
        dtype = DLDataType(_KDLFLOAT, 32, 1)
        assert dtype.code == _KDLFLOAT
        assert dtype.bits == 32
        assert dtype.lanes == 1

    def test_fields_are_mutable(self):
        dtype = DLDataType(_KDLFLOAT, 32, 1)
        dtype.bits = 16
        assert dtype.bits == 16

    def test_repr(self):
        assert repr(DLDataType(_KDLFLOAT, 32, 1)) == "DLDataType(code=DLFLOAT, bits=32, lanes=1)"


class TestTensorType:
    def test_tensor_is_exported(self):
        assert isinstance(Tensor, type)
        assert holoscan.core.Tensor is Tensor

    def test_expected_metadata_attributes_present(self):
        for name in (
            "ndim",
            "shape",
            "strides",
            "size",
            "itemsize",
            "nbytes",
            "byte_offset",
            "device",
            "dtype",
            "data",
            "is_contiguous",
            "is_read_only",
        ):
            assert hasattr(Tensor, name), f"Tensor is missing expected attribute: {name}"

    def test_metadata_properties_are_read_only(self):
        np = pytest.importorskip("numpy")
        tensor = Tensor.as_tensor(np.zeros(3, dtype=np.float32))
        for name in ("ndim", "shape", "size", "itemsize", "nbytes", "byte_offset"):
            with pytest.raises(AttributeError):
                setattr(tensor, name, 1)


class TestTensorDLPack:
    """DLPack export + minimal import (numpy host round-trips)."""

    # REQ: TENSOR-PY-011
    def test_as_tensor_metadata_roundtrip(self):
        np = pytest.importorskip("numpy")
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        tensor = Tensor.as_tensor(arr)
        assert tensor.ndim == 2
        assert tensor.shape == (2, 3)
        assert tensor.size == 6
        assert tensor.itemsize == arr.dtype.itemsize
        assert tensor.nbytes == arr.nbytes
        assert tensor.byte_offset == 0
        assert tensor.strides == arr.strides  # both in bytes for a C-contiguous array
        assert tensor.is_read_only

    def test_dlpack_device_is_cpu(self):
        np = pytest.importorskip("numpy")
        tensor = Tensor.as_tensor(np.zeros((4,), dtype=np.float32))
        device_type, device_id = tensor.__dlpack_device__()
        assert device_type == 1  # kDLCPU
        assert device_id == 0

    # REQ: TENSOR-PY-011
    def test_dlpack_requires_versioned_protocol_for_read_only_import(self):
        np = pytest.importorskip("numpy")
        tensor = Tensor.as_tensor(np.ones((2, 2), dtype=np.float32))
        with pytest.raises(BufferError, match="cannot represent read-only"):
            tensor.__dlpack__()

        capsule = tensor.__dlpack__(max_version=(1, 0))
        assert type(capsule).__name__ == "PyCapsule"

    def test_export_roundtrip_values(self):
        np = pytest.importorskip("numpy")
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        tensor = Tensor.as_tensor(arr)
        out = np.from_dlpack(tensor)
        assert out.shape == (3, 4)
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, arr)

    def test_dlpack_accepts_cuda_stream_protocol_object(self):
        np = pytest.importorskip("numpy")

        class StreamObject:
            def __cuda_stream__(self):
                return (0, 1234)

        tensor = Tensor.as_tensor(np.arange(6, dtype=np.float32))
        capsule = tensor.__dlpack__(stream=StreamObject(), max_version=(1, 0))
        assert type(capsule).__name__ == "PyCapsule"

    def test_dlpack_rejects_invalid_cuda_stream_protocol_object(self):
        np = pytest.importorskip("numpy")
        tensor = Tensor.as_tensor(np.arange(6, dtype=np.float32))

        class BadVersionStream:
            def __cuda_stream__(self):
                return (1, 1234)

        class AttributeStream:
            __cuda_stream__ = (0, 1234)

        class NegativePointerStream:
            def __cuda_stream__(self):
                return (0, -1)

        with pytest.raises(RuntimeError, match="protocol version must be 0"):
            tensor.__dlpack__(stream=BadVersionStream())
        with pytest.raises(RuntimeError, match="must be callable"):
            tensor.__dlpack__(stream=AttributeStream())
        with pytest.raises(RuntimeError, match="stream pointer must be non-negative"):
            tensor.__dlpack__(stream=NegativePointerStream())

    # REQ: TENSOR-PY-012
    def test_dlpack_rejects_plain_integer_zero_stream(self):
        np = pytest.importorskip("numpy")
        tensor = Tensor.as_tensor(np.arange(6, dtype=np.float32))

        with pytest.raises(RuntimeError, match="positive integer"):
            tensor.__dlpack__(stream=0)

    def test_tensor_outlives_source(self):
        # The imported tensor owns the producer's memory via a GIL-acquiring reference, so it stays
        # valid (and frees safely) after the source array is dropped.
        np = pytest.importorskip("numpy")
        tensor = Tensor.as_tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
        out = np.from_dlpack(tensor)
        np.testing.assert_array_equal(out, np.arange(6, dtype=np.float32).reshape(2, 3))

    def test_torch_cpu_roundtrip(self):
        torch = pytest.importorskip("torch")
        src = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        tensor = Tensor.as_tensor(src)
        assert tensor.shape == (2, 3)
        out = torch.from_dlpack(tensor)
        assert torch.equal(out, src)

    def test_torch_cpu_import_prefers_c_exchange(self):
        torch = pytest.importorskip("torch")
        if not _torch_version_at_least(torch, 2, 10):
            pytest.skip("PyTorch 2.10.0 or newer is required for the DLPack C exchange API")
        if not hasattr(torch.Tensor, "__dlpack_c_exchange_api__"):
            pytest.skip("PyTorch Tensor does not expose the DLPack C exchange API")

        class NoPythonDLPackTensor(torch.Tensor):
            @staticmethod
            def __new__(cls, tensor):
                return torch.Tensor._make_subclass(cls, tensor, tensor.requires_grad)

            def __dlpack__(self, *args, **kwargs):
                raise AssertionError("Tensor.as_tensor used the Python __dlpack__ fallback")

        base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        src = NoPythonDLPackTensor(base)
        assert is_tensor_like(src)

        tensor = Tensor.as_tensor(src)
        assert tensor.shape == (2, 3)
        assert tensor.data == src.data_ptr()
        out = torch.from_dlpack(tensor)
        assert torch.equal(out, base)

    def test_cupy_device_import(self):
        # Importing a CUDA tensor works; device/shape metadata is preserved.
        cupy = pytest.importorskip("cupy")
        _skip_without_cuda_devices(cupy, 1)
        src = cupy.arange(6, dtype=cupy.float32).reshape(2, 3)
        tensor = Tensor.as_tensor(src)
        assert tensor.shape == (2, 3)
        device_type, device_id = tensor.__dlpack_device__()
        assert device_type == 2  # kDLCUDA
        assert device_id == cupy.cuda.runtime.getDevice()

    def test_cupy_nonzero_device_import(self):
        cupy = pytest.importorskip("cupy")
        _skip_without_cuda_devices(cupy, 2)
        with cupy.cuda.Device(1):
            src = cupy.arange(6, dtype=cupy.float32).reshape(2, 3)
            tensor = Tensor.as_tensor(src)
            assert tensor.shape == (2, 3)
            device_type, device_id = tensor.__dlpack_device__()
            assert device_type == 2  # kDLCUDA
            assert device_id == 1
            cupy.testing.assert_array_equal(cupy.from_dlpack(tensor), src)

    def test_cupy_device_export_roundtrip(self):
        cupy = pytest.importorskip("cupy")
        _skip_without_cuda_devices(cupy, 1)
        src = cupy.arange(6, dtype=cupy.float32).reshape(2, 3)
        tensor = Tensor.as_tensor(src)
        out = cupy.from_dlpack(tensor)
        cupy.testing.assert_array_equal(out, src)


class TestArrayInterface:
    """Array-interface export: host -> __array_interface__, device -> __cuda_array_interface__."""

    # REQ: TENSOR-PY-011
    def test_host_exposes_array_interface_only(self):
        np = pytest.importorskip("numpy")
        tensor = Tensor.as_tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
        assert hasattr(tensor, "__array_interface__")
        assert not hasattr(tensor, "__cuda_array_interface__")
        interface = tensor.__array_interface__
        assert interface["shape"] == (2, 3)
        assert interface["typestr"] == np.dtype(np.float32).str
        assert interface["version"] == 3
        assert interface["data"][1] is True

    def test_numpy_asarray_roundtrip(self):
        np = pytest.importorskip("numpy")
        src = np.arange(6, dtype=np.float32).reshape(2, 3)
        out = np.asarray(Tensor.as_tensor(src))
        np.testing.assert_array_equal(out, src)

    def test_device_exposes_cuda_array_interface_only(self):
        cupy = pytest.importorskip("cupy")
        _skip_without_cuda_devices(cupy, 1)
        tensor = Tensor.as_tensor(cupy.arange(6, dtype=cupy.float32).reshape(2, 3))
        assert hasattr(tensor, "__cuda_array_interface__")
        assert not hasattr(tensor, "__array_interface__")
        assert tensor.__cuda_array_interface__["shape"] == (2, 3)

    def test_cupy_cuda_array_interface_reports_default_stream(self):
        cupy = pytest.importorskip("cupy")
        if cupy.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("no CUDA device available")
        tensor = Tensor.as_tensor(cupy.arange(6, dtype=cupy.float32).reshape(2, 3))
        interface = tensor.__cuda_array_interface__
        assert interface["stream"] == 1

    def test_cupy_asarray_roundtrip(self):
        cupy = pytest.importorskip("cupy")
        _skip_without_cuda_devices(cupy, 1)
        src = cupy.arange(6, dtype=cupy.float32).reshape(2, 3)
        out = cupy.asarray(Tensor.as_tensor(src))
        cupy.testing.assert_array_equal(out, src)


class TestArrayInterfaceImport:
    """as_tensor falls back to the array interfaces when DLPack is absent."""

    def test_host_array_interface_only(self):
        np = pytest.importorskip("numpy")

        # Exposes only __array_interface__ (no __dlpack__) and keeps the array alive -> host path.
        class HostArrayInterfaceOnly:
            def __init__(self, array):
                self._array = array
                self.__array_interface__ = array.__array_interface__

        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        tensor = Tensor.as_tensor(HostArrayInterfaceOnly(arr))
        assert tensor.shape == (2, 3)
        np.testing.assert_array_equal(np.asarray(tensor), arr)

    @pytest.mark.parametrize("strides", [(12,), (24, 12, 4)])
    def test_host_array_interface_rejects_strides_rank_mismatch(self, strides):
        np = pytest.importorskip("numpy")

        class HostArrayInterfaceOnly:
            def __init__(self, array):
                self._array = array
                self.__array_interface__ = dict(array.__array_interface__)
                self.__array_interface__["strides"] = strides

        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        with pytest.raises(RuntimeError, match="strides length must match shape rank"):
            Tensor.as_tensor(HostArrayInterfaceOnly(arr))

    def test_cuda_array_interface_only(self):
        cupy = pytest.importorskip("cupy")
        _skip_without_cuda_devices(cupy, 1)

        class CudaArrayInterfaceOnly:
            def __init__(self, array):
                self._array = array
                self.__cuda_array_interface__ = array.__cuda_array_interface__

        arr = cupy.arange(6, dtype=cupy.float32).reshape(2, 3)
        tensor = Tensor.as_tensor(CudaArrayInterfaceOnly(arr))
        assert tensor.shape == (2, 3)
        device_type, device_id = tensor.__dlpack_device__()
        assert device_type == 2  # kDLCUDA
        assert device_id == cupy.cuda.runtime.getDevice()
        cupy.testing.assert_array_equal(cupy.asarray(tensor), arr)

    def test_cuda_array_interface_only_preserves_nonzero_device(self):
        cupy = pytest.importorskip("cupy")
        _skip_without_cuda_devices(cupy, 2)

        class CudaArrayInterfaceOnly:
            def __init__(self, array):
                self._array = array
                self.__cuda_array_interface__ = array.__cuda_array_interface__

        with cupy.cuda.Device(1):
            arr = cupy.arange(6, dtype=cupy.float32).reshape(2, 3)
            tensor = Tensor.as_tensor(CudaArrayInterfaceOnly(arr))
            assert tensor.shape == (2, 3)
            device_type, device_id = tensor.__dlpack_device__()
            assert device_type == 2  # kDLCUDA
            assert device_id == 1
            cupy.testing.assert_array_equal(cupy.asarray(tensor), arr)

    def test_unsupported_object_raises(self):
        with pytest.raises(RuntimeError, match="does not support"):
            Tensor.as_tensor([1, 2, 3])


class TestIsTensorLike:
    def test_numpy_array_is_tensor_like(self):
        np = pytest.importorskip("numpy")
        assert is_tensor_like(np.zeros(3))

    def test_holoscan_tensor_is_tensor_like(self):
        np = pytest.importorskip("numpy")
        assert is_tensor_like(Tensor.as_tensor(np.zeros(3, dtype=np.float32)))

    def test_non_tensor_objects(self):
        assert not is_tensor_like([1, 2, 3])
        assert not is_tensor_like(5)
        assert not is_tensor_like("not a tensor")


class TestDLPackCExchange:
    """DLPack 1.3 fast C exchange API (__dlpack_c_exchange_api__)."""

    def test_capsule_on_type(self):
        assert hasattr(Tensor, "__dlpack_c_exchange_api__")
        capsule = Tensor.__dlpack_c_exchange_api__
        assert type(capsule).__name__ == "PyCapsule"
        assert "dlpack_exchange_api" in str(capsule)

    def _check_export(self, tensor, arr, expected_device_type, expected_device_id):
        result = _test_dlpack_c_exchange_api(tensor)
        element_strides = tuple(stride // arr.dtype.itemsize for stride in arr.strides)
        assert result["api_version"] == (1, 3)
        assert result["managed_version"] == (1, 3)
        assert result["managed_flags"] & 1
        assert result["device"] == (int(expected_device_type), expected_device_id)
        assert result["ndim"] == arr.ndim
        assert result["shape"] == arr.shape
        assert result["strides"] == element_strides
        assert result["byte_offset"] == tensor.byte_offset
        assert result["data"] == tensor.data
        assert result["stack_export_supported"] is False
        assert result["stack_data"] is None
        assert result["stack_shape"] is None
        assert result["stack_strides"] is None
        assert result["imported_shape"] == arr.shape
        assert result["imported_strides"] == arr.strides  # imported Tensor reports byte strides
        assert result["stream"] == 0

    # REQ: TENSOR-PY-011
    def test_numpy_export(self):
        np = pytest.importorskip("numpy")
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        self._check_export(Tensor.as_tensor(arr), arr, DLDeviceType.DLCPU, 0)

    def test_cupy_export(self):
        cupy = pytest.importorskip("cupy")
        _skip_without_cuda_devices(cupy, 1)
        arr = cupy.arange(6, dtype=cupy.float32).reshape(2, 3)
        self._check_export(
            Tensor.as_tensor(arr), arr, DLDeviceType.DLCUDA, cupy.cuda.runtime.getDevice()
        )

    def test_cupy_nonzero_device_export(self):
        cupy = pytest.importorskip("cupy")
        _skip_without_cuda_devices(cupy, 2)
        with cupy.cuda.Device(1):
            arr = cupy.arange(6, dtype=cupy.float32).reshape(2, 3)
            self._check_export(Tensor.as_tensor(arr), arr, DLDeviceType.DLCUDA, 1)

    def test_torch_cuda_export_orders_non_default_readiness_to_default_stream(self):
        torch = pytest.importorskip("torch")
        if not _torch_version_at_least(torch, 2, 10):
            pytest.skip("PyTorch 2.10.0 or newer is required for the DLPack C exchange API")
        if not hasattr(torch.Tensor, "__dlpack_c_exchange_api__"):
            pytest.skip("PyTorch Tensor does not expose the DLPack C exchange API")
        if not torch.cuda.is_available():
            pytest.skip("CUDA is not available")
        try:
            torch.cuda.init()
        except (RuntimeError, torch.cuda.DeferredCudaCallError) as error:
            pytest.skip(f"PyTorch could not initialize CUDA: {error}")

        # torch.cuda.Stream() creates a new non-default stream for the current CUDA device.
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            src = torch.arange(6, device="cuda", dtype=torch.float32).reshape(2, 3)
            tensor = Tensor.as_tensor(src)

        result = _test_dlpack_c_exchange_api(tensor)
        assert result["device"][0] == DLDeviceType.DLCUDA
        assert result["shape"] == src.shape
        assert result["data"] == src.data_ptr()
        # Holoscan reports nullptr as its type-level C exchange current work stream.
        assert result["stream"] == 0
