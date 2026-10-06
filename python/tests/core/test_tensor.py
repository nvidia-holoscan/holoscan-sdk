# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor metadata, array-protocol, ownership, and explicit-copy binding tests."""

import re

import pytest

import holoscan as hs
from holoscan import Tensor
from holoscan._core import _test_dlpack_c_exchange_api, _test_tensor_metadata_projection
from holoscan.tensor import (
    DLDataType,
    DLDataTypeCode,
    DLDevice,
    DLDeviceType,
    is_tensor_like,
)

# DLDataTypeCode raw value (DLDataType.code stays a plain int field, matching DLPack/4.x).
_KDLFLOAT = 2
_KDLUINT = 1


def _torch_version_at_least(torch, major: int, minor: int) -> bool:
    match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
    if match is None:
        return False
    return (int(match.group(1)), int(match.group(2))) >= (major, minor)


def _skip_without_cuda_devices(cupy, count):
    if cupy.cuda.runtime.getDeviceCount() < count:
        pytest.skip(f"at least {count} CUDA device(s) required")


class TestDLEnums:
    """Verify DLPack enum parity; for example, test_data_type_code_values."""

    def test_device_type_values(self):
        # Key setup: Read the public DLPack device-type members.

        # Key assertion: CPU, CUDA, and managed CUDA retain their ABI values.
        assert int(DLDeviceType.DLCPU) == 1
        assert int(DLDeviceType.DLCUDA) == 2
        assert int(DLDeviceType.DLCUDAMANAGED) == 13

    def test_data_type_code_values(self):
        # Key setup: Record the complete public DLPack data-type code table.
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

        # Key assertion: Every Python enum member matches the DLPack ABI value.
        for name, value in expected.items():
            assert int(getattr(DLDataTypeCode, name)) == value


class TestDLDevice:
    """Verify DLDevice value semantics; for example, test_construct_and_fields."""

    def test_construct_and_fields(self):
        # Key setup: Construct a non-default CUDA device value.
        device = DLDevice(DLDeviceType.DLCUDA, 3)

        # Key assertion: Construction preserves both device fields.
        assert device.device_type == DLDeviceType.DLCUDA
        assert device.device_id == 3

    def test_device_type_is_enum(self):
        # Key setup: Construct a CPU device through the public enum.

        # Key assertion: The exposed field remains strongly typed as DLDeviceType.
        assert isinstance(DLDevice(DLDeviceType.DLCPU, 0).device_type, DLDeviceType)

    def test_repr(self):
        # Key setup: Construct representative CPU and CUDA devices.
        cpu = DLDevice(DLDeviceType.DLCPU, 0)
        cuda = DLDevice(DLDeviceType.DLCUDA, 2)

        # Key assertion: Their repr strings expose stable symbolic types and ordinals.
        assert repr(cpu) == "DLDevice(device_type=kDLCPU, device_id=0)"
        assert repr(cuda) == "DLDevice(device_type=kDLCUDA, device_id=2)"


class TestDLDataType:
    """Verify DLDataType value semantics; for example, test_fields_are_mutable."""

    def test_construct_and_fields(self):
        # Key setup: Construct a scalar float32 DLPack type.
        dtype = DLDataType(_KDLFLOAT, 32, 1)

        # Key assertion: Construction preserves code, bit width, and lane count.
        assert dtype.code == _KDLFLOAT
        assert dtype.bits == 32
        assert dtype.lanes == 1

    def test_fields_are_mutable(self):
        # Key setup: Replace the bit width of a constructed DLPack type.
        dtype = DLDataType(_KDLFLOAT, 32, 1)
        dtype.bits = 16

        # Key assertion: DLDataType retains its intended mutable value semantics.
        assert dtype.bits == 16

    def test_repr(self):
        # Key setup: Construct the canonical scalar float32 DLPack type.

        # Key assertion: repr uses the stable symbolic data-type code.
        assert repr(DLDataType(_KDLFLOAT, 32, 1)) == "DLDataType(code=DLFLOAT, bits=32, lanes=1)"


class TestTensorType:
    """Verify Tensor type metadata; for example, test_expected_metadata_attributes_present."""

    def test_tensor_is_exported(self):
        # Key setup: Resolve Tensor from the root and payload namespaces.

        # Key assertion: Both public access paths expose the same native type.
        assert isinstance(Tensor, type)
        assert hs.Tensor is Tensor
        assert hs.payload.Tensor is Tensor

    def test_expected_metadata_attributes_present(self):
        # Key setup: Enumerate the promised metadata-only Tensor property surface.

        # Key assertion: Every documented metadata property exists on the type.
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

    # REQ: TENSOR-PY-002
    def test_metadata_properties_are_read_only(self):
        # Key setup: Import a NumPy array and enumerate immutable metadata properties.
        np = pytest.importorskip("numpy")
        tensor = Tensor.as_tensor(np.zeros(3, dtype=np.float32))
        # Key assertion: Python cannot replace metadata owned by the Tensor view.
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
            "is_read_only",
        ):
            with pytest.raises(AttributeError):
                setattr(tensor, name, 1)


class TestTensorDLPack:
    """Verify DLPack exchange; for example, test_export_roundtrip_values."""

    # REQ: TENSOR-PY-002
    # REQ: TENSOR-PY-011
    def test_as_tensor_metadata_roundtrip(self):
        # Key setup: Import a shaped NumPy array through DLPack.
        np = pytest.importorskip("numpy")
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        tensor = Tensor.as_tensor(arr)

        # Key assertion: Tensor metadata exactly describes the imported read-only view.
        assert tensor.ndim == 2
        assert tensor.shape == (2, 3)
        assert tensor.size == 6
        assert tensor.itemsize == arr.dtype.itemsize
        assert tensor.nbytes == arr.nbytes
        assert tensor.byte_offset == 0
        assert tensor.strides == arr.strides  # both in bytes for a C-contiguous array
        assert tensor.device.device_type == DLDeviceType.DLCPU
        assert tensor.device.device_id == 0
        assert tensor.dtype.code == _KDLFLOAT
        assert tensor.dtype.bits == 32
        assert tensor.dtype.lanes == 1
        assert tensor.data == arr.ctypes.data
        assert tensor.is_contiguous()
        assert tensor.is_read_only

    def test_dlpack_device_is_cpu(self):
        # Key setup: Import a host NumPy array and query its DLPack device tuple.
        np = pytest.importorskip("numpy")
        tensor = Tensor.as_tensor(np.zeros((4,), dtype=np.float32))
        device_type, device_id = tensor.__dlpack_device__()

        # Key assertion: Host storage reports the canonical CPU device identity.
        assert device_type == 1  # kDLCPU
        assert device_id == 0

    # REQ: TENSOR-PY-011
    def test_dlpack_requires_versioned_protocol_for_read_only_import(self):
        # Key setup: Import a read-only Tensor and request legacy and versioned capsules.
        np = pytest.importorskip("numpy")
        tensor = Tensor.as_tensor(np.ones((2, 2), dtype=np.float32))
        with pytest.raises(BufferError, match="cannot represent read-only"):
            tensor.__dlpack__()

        capsule = tensor.__dlpack__(max_version=(1, 0))

        # Key assertion: Only versioned DLPack can represent the read-only import.
        assert type(capsule).__name__ == "PyCapsule"

    def test_export_roundtrip_values(self):
        # Key setup: Round-trip a shaped NumPy array through Tensor and DLPack export.
        np = pytest.importorskip("numpy")
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        tensor = Tensor.as_tensor(arr)
        out = np.from_dlpack(tensor)

        # Key assertion: Export preserves shape, dtype, and every element.
        assert out.shape == (3, 4)
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, arr)

    # REQ: TENSOR-PY-013
    def test_dlpack_copy_true_returns_independent_host_storage(self):
        # Key setup: Ask a DLPack consumer wrapper to export an explicit copy of an imported array.
        np = pytest.importorskip("numpy")
        source = np.arange(6, dtype=np.float32).reshape(2, 3)
        tensor = Tensor.as_tensor(source)

        class CopyingProducer:
            """Request the explicit-copy path used by this DLPack consumer test."""

            def __dlpack_device__(self):
                return tensor.__dlpack_device__()

            def __dlpack__(self, stream=None, **kwargs):
                del kwargs
                return tensor.__dlpack__(stream=stream, copy=True)

        copied = np.from_dlpack(CopyingProducer())
        source.fill(-1)

        # Key assertion: copy=True preserves the pre-mutation values in independent storage.
        np.testing.assert_array_equal(copied, np.arange(6, dtype=np.float32).reshape(2, 3))

    # REQ: TENSOR-PY-013
    def test_dlpack_copy_true_accepts_empty_null_data_storage(self):
        np = pytest.importorskip("numpy")

        # Key setup: Import a valid zero-byte array-interface value whose data pointer is null.
        class EmptyArray:
            __array_interface__ = {
                "shape": (0,),
                "typestr": "|u1",
                "data": (0, True),
                "strides": None,
                "version": 3,
            }

        tensor = Tensor.as_tensor(EmptyArray())

        class CopyingProducer:
            """Request an owning DLPack copy of the empty Tensor."""

            def __dlpack_device__(self):
                return tensor.__dlpack_device__()

            def __dlpack__(self, stream=None, **kwargs):
                del kwargs
                return tensor.__dlpack__(stream=stream, copy=True)

        copied = np.from_dlpack(CopyingProducer())

        # Key assertion: A zero-byte copy needs no source data pointer.
        assert copied.shape == (0,)
        assert copied.dtype == np.uint8

    # REQ: TENSOR-PY-013
    def test_dlpack_cuda_copy_rejects_no_synchronization_stream(self):
        # Key setup: Request an owning device copy while explicitly declining stream ordering.
        cupy = pytest.importorskip("cupy")
        _skip_without_cuda_devices(cupy, 1)
        tensor = Tensor.as_tensor(cupy.arange(6, dtype=cupy.float32))

        # Key assertion: The export cannot return asynchronously copied storage with no dependency.
        with pytest.raises(BufferError, match="stream=-1"):
            tensor.__dlpack__(stream=-1, copy=True, max_version=(1, 0))

    # REQ: TENSOR-PY-013
    def test_dlpack_cross_device_request_rejects_copy_false_before_allocation(self):
        # Key setup: Request CUDA placement from a CPU Tensor while explicitly forbidding a copy.
        np = pytest.importorskip("numpy")
        tensor = Tensor.as_tensor(np.arange(6, dtype=np.float32))

        # Key assertion: The protocol conflict is diagnosed without requiring a CUDA device.
        with pytest.raises(BufferError, match="copy=False"):
            tensor.__dlpack__(dl_device=(int(DLDeviceType.DLCUDA), 0), copy=False)

    def test_dlpack_accepts_cuda_stream_protocol_object(self):
        # Key setup: Supply a valid CUDA stream protocol object to a host-backed export.
        np = pytest.importorskip("numpy")

        class StreamObject:
            """Provide the valid protocol for the enclosing stream-object test."""

            def __cuda_stream__(self):
                return (0, 1234)

        tensor = Tensor.as_tensor(np.arange(6, dtype=np.float32))
        capsule = tensor.__dlpack__(stream=StreamObject(), max_version=(1, 0))

        # Key assertion: A valid protocol object produces a DLPack capsule.
        assert type(capsule).__name__ == "PyCapsule"

    def test_dlpack_rejects_invalid_cuda_stream_protocol_object(self):
        # Key setup: Define malformed version, callable, and pointer protocol variants.
        np = pytest.importorskip("numpy")
        tensor = Tensor.as_tensor(np.arange(6, dtype=np.float32))

        class BadVersionStream:
            """Provide the bad-version case for this stream-protocol test."""

            def __cuda_stream__(self):
                return (1, 1234)

        class AttributeStream:
            """Provide the non-callable case for this stream-protocol test."""

            __cuda_stream__ = (0, 1234)

        class NegativePointerStream:
            """Provide the negative-pointer case for this stream-protocol test."""

            def __cuda_stream__(self):
                return (0, -1)

        # Key assertion: Each malformed protocol fails with its specific validation message.
        with pytest.raises(RuntimeError, match="protocol version must be 0"):
            tensor.__dlpack__(stream=BadVersionStream())
        with pytest.raises(RuntimeError, match="must be callable"):
            tensor.__dlpack__(stream=AttributeStream())
        with pytest.raises(RuntimeError, match="stream pointer must be non-negative"):
            tensor.__dlpack__(stream=NegativePointerStream())

    # REQ: TENSOR-PY-012
    def test_dlpack_rejects_plain_integer_zero_stream(self):
        # Key setup: Request export with the reserved integer-zero stream spelling.
        np = pytest.importorskip("numpy")
        tensor = Tensor.as_tensor(np.arange(6, dtype=np.float32))

        # Key assertion: Integer stream values must be positive.
        with pytest.raises(RuntimeError, match="positive integer"):
            tensor.__dlpack__(stream=0)

    def test_tensor_outlives_source(self):
        # Key setup: Import a temporary producer whose lifetime ends immediately.
        np = pytest.importorskip("numpy")
        tensor = Tensor.as_tensor(np.arange(6, dtype=np.float32).reshape(2, 3))
        out = np.from_dlpack(tensor)

        # Key assertion: Tensor ownership keeps the producer memory valid for export.
        np.testing.assert_array_equal(out, np.arange(6, dtype=np.float32).reshape(2, 3))

    def test_torch_cpu_roundtrip(self):
        # Key setup: Round-trip a CPU PyTorch tensor through the Tensor wrapper.
        torch = pytest.importorskip("torch")
        src = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        tensor = Tensor.as_tensor(src)
        assert tensor.shape == (2, 3)
        out = torch.from_dlpack(tensor)

        # Key assertion: The exported PyTorch tensor retains every source value.
        assert torch.equal(out, src)

    def test_torch_cpu_import_prefers_c_exchange(self):
        # Key setup: Disable Python DLPack on a PyTorch subclass that supports C exchange.
        torch = pytest.importorskip("torch")
        if not _torch_version_at_least(torch, 2, 10):
            pytest.skip("PyTorch 2.10.0 or newer is required for the DLPack C exchange API")
        if not hasattr(torch.Tensor, "__dlpack_c_exchange_api__"):
            pytest.skip("PyTorch Tensor does not expose the DLPack C exchange API")

        class NoPythonDLPackTensor(torch.Tensor):
            """Prove test_torch_cpu_import_prefers_c_exchange avoids Python fallback."""

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

        # Key assertion: C exchange imports and re-exports the original storage without fallback.
        assert torch.equal(out, base)

    def test_cupy_device_import(self):
        # Key setup: Import a CuPy array from the current CUDA device.
        cupy = pytest.importorskip("cupy")
        _skip_without_cuda_devices(cupy, 1)
        src = cupy.arange(6, dtype=cupy.float32).reshape(2, 3)
        tensor = Tensor.as_tensor(src)
        assert tensor.shape == (2, 3)
        device_type, device_id = tensor.__dlpack_device__()

        # Key assertion: Shape and current CUDA device metadata are preserved.
        assert device_type == 2  # kDLCUDA
        assert device_id == cupy.cuda.runtime.getDevice()

    def test_cupy_nonzero_device_import(self):
        # Key setup: Import and re-export a CuPy array while CUDA device 1 is active.
        cupy = pytest.importorskip("cupy")
        _skip_without_cuda_devices(cupy, 2)
        with cupy.cuda.Device(1):
            src = cupy.arange(6, dtype=cupy.float32).reshape(2, 3)
            tensor = Tensor.as_tensor(src)
            assert tensor.shape == (2, 3)
            device_type, device_id = tensor.__dlpack_device__()
            assert device_type == 2  # kDLCUDA
            assert device_id == 1

            # Key assertion: The nonzero device and values survive the round trip.
            cupy.testing.assert_array_equal(cupy.from_dlpack(tensor), src)

    def test_cupy_device_export_roundtrip(self):
        # Key setup: Export an imported current-device CuPy array back through DLPack.
        cupy = pytest.importorskip("cupy")
        _skip_without_cuda_devices(cupy, 1)
        src = cupy.arange(6, dtype=cupy.float32).reshape(2, 3)
        tensor = Tensor.as_tensor(src)
        out = cupy.from_dlpack(tensor)

        # Key assertion: Device export preserves every source value.
        cupy.testing.assert_array_equal(out, src)


class TestArrayInterface:
    """Verify array-interface export; for example, test_host_exposes_array_interface_only."""

    # REQ: TENSOR-PY-011
    def test_host_exposes_array_interface_only(self):
        # Key setup: Import a host NumPy array and inspect its protocol attributes.
        np = pytest.importorskip("numpy")
        tensor = Tensor.as_tensor(np.arange(6, dtype=np.float32).reshape(2, 3))

        # Key assertion: Host storage exposes only a read-only v3 array interface.
        assert hasattr(tensor, "__array_interface__")
        assert not hasattr(tensor, "__cuda_array_interface__")
        interface = tensor.__array_interface__
        assert interface["shape"] == (2, 3)
        assert interface["typestr"] == np.dtype(np.float32).str
        assert interface["version"] == 3
        assert interface["data"][1] is True

    def test_numpy_asarray_roundtrip(self):
        # Key setup: Convert an imported Tensor back through NumPy's array interface.
        np = pytest.importorskip("numpy")
        src = np.arange(6, dtype=np.float32).reshape(2, 3)
        out = np.asarray(Tensor.as_tensor(src))

        # Key assertion: NumPy sees the original values without a protocol mismatch.
        np.testing.assert_array_equal(out, src)

    def test_device_exposes_cuda_array_interface_only(self):
        # Key setup: Import a CuPy array and inspect its protocol attributes.
        cupy = pytest.importorskip("cupy")
        _skip_without_cuda_devices(cupy, 1)
        tensor = Tensor.as_tensor(cupy.arange(6, dtype=cupy.float32).reshape(2, 3))

        # Key assertion: Device storage exposes only the CUDA array interface.
        assert hasattr(tensor, "__cuda_array_interface__")
        assert not hasattr(tensor, "__array_interface__")
        assert tensor.__cuda_array_interface__["shape"] == (2, 3)

    def test_cupy_cuda_array_interface_reports_default_stream(self):
        # Key setup: Import a CuPy array and read its CUDA array-interface stream.
        cupy = pytest.importorskip("cupy")
        if cupy.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("no CUDA device available")
        tensor = Tensor.as_tensor(cupy.arange(6, dtype=cupy.float32).reshape(2, 3))
        interface = tensor.__cuda_array_interface__

        # Key assertion: The exported interface uses CUDA's default-stream sentinel.
        assert interface["stream"] == 1

    def test_cupy_asarray_roundtrip(self):
        # Key setup: Convert an imported Tensor through CuPy's CUDA array interface.
        cupy = pytest.importorskip("cupy")
        _skip_without_cuda_devices(cupy, 1)
        src = cupy.arange(6, dtype=cupy.float32).reshape(2, 3)
        out = cupy.asarray(Tensor.as_tensor(src))

        # Key assertion: CuPy sees every original device value.
        cupy.testing.assert_array_equal(out, src)


class TestTensorMetadataProjection:
    """Metadata projection behavior not constructible through normal Python protocols."""

    # REQ: TENSOR-PY-002
    def test_nonzero_offset_and_explicit_strides(self):
        np = pytest.importorskip("numpy")

        # Key setup: Project a strided view four bytes into a live host backing allocation.
        backing = np.arange(32, dtype=np.uint8)
        result = _test_tensor_metadata_projection(
            Tensor.as_tensor(backing), (2, 3), (8, 2), DLDataType(_KDLUINT, 8, 1), 4
        )

        # Key assertion: Both Tensor and array-interface metadata expose the logical offset view.
        assert result["shape"] == (2, 3)
        assert result["strides"] == (8, 2)
        assert result["data"] == backing.ctypes.data + 4
        assert result["array_interface"]["data"][0] == backing.ctypes.data + 4
        assert result["array_interface"]["strides"] == (8, 2)

    # REQ: TENSOR-PY-002
    def test_multi_lane_implicit_strides(self):
        np = pytest.importorskip("numpy")

        # Key setup: Leave strides implicit for a two-byte DLPack element encoded as two lanes.
        backing = np.arange(32, dtype=np.uint8)
        result = _test_tensor_metadata_projection(
            Tensor.as_tensor(backing), (2, 3), None, DLDataType(_KDLUINT, 8, 2), 0
        )

        # Key assertion: Contiguous strides use the complete lane-aware element size.
        assert result["itemsize"] == 2
        assert result["strides"] == (6, 2)
        assert result["array_interface"]["strides"] is None

    # REQ: TENSOR-PY-002
    def test_rank_zero_projection(self):
        np = pytest.importorskip("numpy")

        # Key setup: Project scalar metadata with no shape or stride entries.
        backing = np.zeros(1, dtype=np.uint8)
        result = _test_tensor_metadata_projection(
            Tensor.as_tensor(backing), (), None, DLDataType(_KDLUINT, 8, 1), 0
        )

        # Key assertion: Rank-zero Tensor and array-interface projections stay empty and implicit.
        assert result["shape"] == ()
        assert result["strides"] == ()
        assert result["array_interface"]["shape"] == ()
        assert result["array_interface"]["strides"] is None


class TestArrayInterfaceImport:
    """Verify array-interface fallback; for example, test_host_array_interface_only."""

    def test_host_array_interface_only(self):
        # Key setup: Wrap a NumPy array in an object exposing only __array_interface__.
        np = pytest.importorskip("numpy")

        class HostArrayInterfaceOnly:
            """Provide the fallback used by test_host_array_interface_only."""

            def __init__(self, array):
                self._array = array
                self.__array_interface__ = array.__array_interface__

        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        tensor = Tensor.as_tensor(HostArrayInterfaceOnly(arr))

        # Key assertion: Fallback import preserves shape and host values.
        assert tensor.shape == (2, 3)
        np.testing.assert_array_equal(np.asarray(tensor), arr)

    @pytest.mark.parametrize("strides", [(12,), (24, 12, 4)])
    def test_host_array_interface_rejects_strides_rank_mismatch(self, strides):
        # Key setup: Publish host strides whose length disagrees with the shape rank.
        np = pytest.importorskip("numpy")

        class HostArrayInterfaceOnly:
            """Inject malformed strides for the enclosing rank-mismatch test."""

            def __init__(self, array):
                self._array = array
                self.__array_interface__ = dict(array.__array_interface__)
                self.__array_interface__["strides"] = strides

        arr = np.arange(6, dtype=np.float32).reshape(2, 3)

        # Key assertion: Fallback validation rejects the malformed stride vector.
        with pytest.raises(RuntimeError, match="strides length must match shape rank"):
            Tensor.as_tensor(HostArrayInterfaceOnly(arr))

    def test_cuda_array_interface_only(self):
        # Key setup: Wrap a CuPy array in an object exposing only __cuda_array_interface__.
        cupy = pytest.importorskip("cupy")
        _skip_without_cuda_devices(cupy, 1)

        class CudaArrayInterfaceOnly:
            """Provide the fallback used by test_cuda_array_interface_only."""

            def __init__(self, array):
                self._array = array
                self.__cuda_array_interface__ = array.__cuda_array_interface__

        arr = cupy.arange(6, dtype=cupy.float32).reshape(2, 3)
        tensor = Tensor.as_tensor(CudaArrayInterfaceOnly(arr))
        assert tensor.shape == (2, 3)
        device_type, device_id = tensor.__dlpack_device__()
        assert device_type == 2  # kDLCUDA
        assert device_id == cupy.cuda.runtime.getDevice()

        # Key assertion: Fallback import preserves current-device values.
        cupy.testing.assert_array_equal(cupy.asarray(tensor), arr)

    def test_cuda_array_interface_only_preserves_nonzero_device(self):
        # Key setup: Import a CUDA-interface-only wrapper while device 1 is active.
        cupy = pytest.importorskip("cupy")
        _skip_without_cuda_devices(cupy, 2)

        class CudaArrayInterfaceOnly:
            """Provide the nonzero-device fallback used by this test."""

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

            # Key assertion: Fallback import retains the nonzero device and values.
            cupy.testing.assert_array_equal(cupy.asarray(tensor), arr)

    def test_unsupported_object_raises(self):
        # Key setup: Pass an object with neither DLPack nor array-interface support.

        # Key assertion: Import rejects the unsupported object explicitly.
        with pytest.raises(RuntimeError, match="does not support"):
            Tensor.as_tensor([1, 2, 3])


class TestIsTensorLike:
    """Verify protocol detection; for example, test_numpy_array_is_tensor_like."""

    def test_numpy_array_is_tensor_like(self):
        # Key setup: Construct a NumPy object with supported tensor protocols.
        np = pytest.importorskip("numpy")

        # Key assertion: NumPy arrays are recognized as tensor-like.
        assert is_tensor_like(np.zeros(3))

    def test_holoscan_tensor_is_tensor_like(self):
        # Key setup: Wrap a NumPy array as a Holoscan Tensor.
        np = pytest.importorskip("numpy")

        # Key assertion: Native Tensor objects are recognized as tensor-like.
        assert is_tensor_like(Tensor.as_tensor(np.zeros(3, dtype=np.float32)))

    def test_non_tensor_objects(self):
        # Key setup: Enumerate representative containers and scalar values without tensor protocols.

        # Key assertion: Ordinary Python values are not classified as tensor-like.
        assert not is_tensor_like([1, 2, 3])
        assert not is_tensor_like(5)
        assert not is_tensor_like("not a tensor")


class TestDLPackCExchange:
    """Verify DLPack 1.3 C exchange; for example, test_numpy_export."""

    def test_capsule_on_type(self):
        # Key setup: Read the type-level DLPack C exchange capsule.

        # Key assertion: Tensor exposes the named C exchange API capsule.
        assert hasattr(Tensor, "__dlpack_c_exchange_api__")
        capsule = Tensor.__dlpack_c_exchange_api__
        assert type(capsule).__name__ == "PyCapsule"
        assert "dlpack_exchange_api" in str(capsule)

    def _check_export(self, tensor, arr, expected_device_type, expected_device_id):
        """Check export metadata for representative test_numpy_export and CUDA variants."""
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
        # Key setup: Export a host NumPy array through the DLPack C exchange helper.
        np = pytest.importorskip("numpy")
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)

        # Key assertion: Shared export checks accept the complete host metadata record.
        self._check_export(Tensor.as_tensor(arr), arr, DLDeviceType.DLCPU, 0)

    def test_cupy_export(self):
        # Key setup: Export a current-device CuPy array through C exchange.
        cupy = pytest.importorskip("cupy")
        _skip_without_cuda_devices(cupy, 1)
        arr = cupy.arange(6, dtype=cupy.float32).reshape(2, 3)

        # Key assertion: Shared export checks accept the current CUDA device record.
        self._check_export(
            Tensor.as_tensor(arr), arr, DLDeviceType.DLCUDA, cupy.cuda.runtime.getDevice()
        )

    def test_cupy_nonzero_device_export(self):
        # Key setup: Export a CuPy array through C exchange while device 1 is active.
        cupy = pytest.importorskip("cupy")
        _skip_without_cuda_devices(cupy, 2)
        with cupy.cuda.Device(1):
            arr = cupy.arange(6, dtype=cupy.float32).reshape(2, 3)

            # Key assertion: Shared export checks retain the nonzero CUDA ordinal.
            self._check_export(Tensor.as_tensor(arr), arr, DLDeviceType.DLCUDA, 1)

    def test_torch_cuda_export_orders_non_default_readiness_to_default_stream(self):
        # Key setup: Import work produced on a non-default PyTorch CUDA stream.
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

        # Key assertion: Export preserves storage while ordering readiness to the default stream.
        assert result["device"][0] == DLDeviceType.DLCUDA
        assert result["shape"] == src.shape
        assert result["data"] == src.data_ptr()
        # Holoscan reports nullptr as its type-level C exchange current work stream.
        assert result["stream"] == 0
