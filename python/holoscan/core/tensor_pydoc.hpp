/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Docstrings for the holoscan.core Tensor bindings. Plain constants (no codegen) covering the
// surface bound in tensor.cpp / dl_converter.cpp.

namespace holoscan::doc {

namespace DLDeviceType {
inline constexpr const char* doc = R"doc(
DLPack device type.

Members
-------
- ``DLCPU``: system (host) memory.
- ``DLCUDA``: CUDA device memory.
- ``DLCUDAHOST``: CUDA pinned host memory.
- ``DLCUDAMANAGED``: CUDA managed (unified) memory.
)doc";
}  // namespace DLDeviceType

namespace DLDataTypeCode {
inline constexpr const char* doc = R"doc(
DLPack scalar type code (``DLINT``, ``DLUINT``, ``DLFLOAT``, ``DLOPAQUEHANDLE``, ``DLBFLOAT``,
``DLCOMPLEX``, ``DLBOOL``, and DLPack's FP8/FP6/FP4 storage codes).
)doc";
}  // namespace DLDataTypeCode

namespace DLDevice {
inline constexpr const char* doc = R"doc(
A DLPack device: a ``device_type`` (``DLDeviceType``) and an integer ``device_id``.
)doc";
inline constexpr const char* device_type = R"doc(The device type (``DLDeviceType``).)doc";
inline constexpr const char* device_id = R"doc(The device ordinal (int).)doc";
}  // namespace DLDevice

namespace DLDataType {
inline constexpr const char* doc = R"doc(
A DLPack data type: a scalar type ``code``, a ``bits`` width, and a ``lanes`` count.

See the DLPack documentation:
https://dmlc.github.io/dlpack/latest/c_api.html#_CPPv410DLDataType
)doc";
inline constexpr const char* code = R"doc(Scalar type code (int; see ``DLDataTypeCode``).)doc";
inline constexpr const char* bits = R"doc(Bit width of a single scalar (int).)doc";
inline constexpr const char* lanes = R"doc(Number of lanes (int; > 1 for vector types).)doc";
}  // namespace DLDataType

namespace Tensor {

inline constexpr const char* doc = R"doc(
An n-dimensional array, interoperable with NumPy, CuPy, and PyTorch via DLPack and the array
interface protocols. Wraps a DLPack managed tensor; instances are cheap handles that share the
underlying memory and its access authority. Zero-copy imports are read-only; writable consumers
must use an explicit allocation-and-copy path.
)doc";

inline constexpr const char* ndim = R"doc(Number of dimensions (int).)doc";
inline constexpr const char* shape = R"doc(Tuple of the tensor's dimensions.)doc";
inline constexpr const char* strides = R"doc(Tuple of the tensor's strides, in bytes.)doc";
inline constexpr const char* size = R"doc(Total number of elements (int).)doc";
inline constexpr const char* itemsize = R"doc(Size of a single element, in bytes (int).)doc";
inline constexpr const char* nbytes = R"doc(Total size of the elements, in bytes (int).)doc";
inline constexpr const char* byte_offset = R"doc(
Offset, in bytes, from the start of the underlying allocation to the tensor data (int).
)doc";
inline constexpr const char* device = R"doc(The ``DLDevice`` the tensor data resides on.)doc";
inline constexpr const char* dtype = R"doc(The ``DLDataType`` of the tensor elements.)doc";
inline constexpr const char* data = R"doc(Integer address of the tensor's data pointer.)doc";
inline constexpr const char* is_contiguous = R"doc(
Whether the tensor has a contiguous, row-major (C-contiguous) memory layout.

Returns
-------
bool
)doc";
inline constexpr const char* is_read_only = R"doc(
Whether the tensor's effective backing authority is permanently read-only.
)doc";

inline constexpr const char* as_tensor = R"doc(
Create a Tensor from a tensor-like object (zero-copy).

Parameters
----------
obj : array-like
    A NumPy array, CuPy array, PyTorch tensor, or any object supporting the DLPack protocol or an
    array interface.

Returns
-------
holoscan.core.Tensor

Notes
-----
Zero-copy imported storage is permanently read-only. The returned Tensor keeps the producer or
array-interface owner alive but does not grant local write authority.

The interchange protocol is selected by precedence: the DLPack C exchange API [1]_ first, then the
Python DLPack protocol [1]_, [2]_, then ``__cuda_array_interface__`` [3]_, then
``__array_interface__`` [4]_. DLPack is preferred because it transfers ownership (so the imported
Tensor keeps its data alive independently of the source), has explicit stream-synchronization
semantics, and is the array-API-endorsed interchange path. The C exchange API avoids the per-call
Python capsule path when a producer type exposes ``__dlpack_c_exchange_api__``.

References
----------
.. [1] https://dmlc.github.io/dlpack/latest/python_spec.html
.. [2] https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html
.. [3] https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
.. [4] https://numpy.org/doc/stable/reference/arrays.interface.html
)doc";

inline constexpr const char* from_dlpack = R"doc(
Create a Tensor from an object supporting the DLPack protocol [1]_, [2]_ (zero-copy).

Parameters
----------
obj : array-like
    An object exposing ``__dlpack__`` and ``__dlpack_device__``.

Returns
-------
holoscan.core.Tensor

Notes
-----
Zero-copy imported storage is permanently read-only.

For CUDA producers the import passes the legacy default stream to the producer's
``__dlpack__(stream=...)`` and records it as the Tensor's readiness stream, so a later export can
order consumer streams against the producer's work.

References
----------
.. [1] https://dmlc.github.io/dlpack/latest/python_spec.html
.. [2] https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html
)doc";

inline constexpr const char* dlpack = R"doc(
Export the tensor as a DLPack capsule (the array-API ``__dlpack__`` protocol).

Parameters
----------
stream : int or None, optional
    Consumer stream for synchronization. ``None``/``1`` = legacy default stream, ``2`` = per-thread
    default, ``-1`` = no synchronization, a stream pointer as a positive integer, or an object with
    callable ``__cuda_stream__()`` returning ``(0, stream_pointer)``. Ignored for host tensors.
max_version : tuple[int, int] or None, optional
    Maximum supported DLPack ``(major, minor)`` version; a major >= 1 selects the versioned capsule.
dl_device : tuple[int, int] or None, optional
    Requested ``(device_type, device_id)``. Copying to a different device is not supported and
    raises ``BufferError``.
copy : bool or None, optional
    ``copy=True`` is not supported and raises ``BufferError``.

Returns
-------
PyCapsule

Raises
------
BufferError
    If the tensor is read-only and ``max_version`` does not select a versioned capsule. Legacy
    DLPack cannot carry read-only authority, so Holoscan rejects rather than widening access.
)doc";

inline constexpr const char* dlpack_device = R"doc(
Return the tensor's ``(device_type, device_id)`` tuple (the array-API ``__dlpack_device__``
protocol). Both values are ints. CUDA device tensors preserve the owning CUDA ordinal; CPU,
pinned-host, and managed tensors report ``device_id == 0``.

Returns
-------
tuple[int, int]
)doc";

inline constexpr const char* array_interface = R"doc(
The NumPy ``__array_interface__`` dict. Present only for host (CPU / pinned-host) tensors;
accessing it on a device tensor raises ``AttributeError``.
)doc";

inline constexpr const char* cuda_array_interface = R"doc(
The ``__cuda_array_interface__`` dict. Present only for device (CUDA / managed) tensors; accessing
it on a host tensor raises ``AttributeError``. Readiness-aware tensors are synchronized to the
legacy default stream before export because the returned dictionary reports ``stream=1``.
)doc";

}  // namespace Tensor

inline constexpr const char* is_tensor_like = R"doc(
Whether an object is convertible to a Tensor, i.e. it exposes the DLPack protocol
(``__dlpack__`` and ``__dlpack_device__``) or an array interface.

Returns
-------
bool
)doc";

}  // namespace holoscan::doc
