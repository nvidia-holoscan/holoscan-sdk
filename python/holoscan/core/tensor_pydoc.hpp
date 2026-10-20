/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Docstrings for the Holoscan Tensor bindings. Plain constants (no codegen) covering the
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
stream : int or object, optional
    CUDA import stream, as an integer or an object exposing ``__cuda_stream__()``.
    Defaults to the activation stream inside ``compute()`` and the legacy default stream outside
    a callback. DLPack producers negotiate directly on this stream. Array-interface and C exchange
    producers retain their advertised readiness and order an explicitly supplied stream after it.
    Integer ``0`` is invalid; use ``1`` or ``cupy.cuda.Stream.null`` for the default stream.

Returns
-------
holoscan.Tensor

Notes
-----
Use ``with ctx.stream():`` for CuPy work in ``compute()``. Prefer ``as_=cupy`` on inputs or
``cupy.from_dlpack(tensor)`` for conversion. Outside callbacks, keep external consumer streams
alive until their exported views are released.

Zero-copy imported storage is permanently read-only. The returned Tensor keeps its producer
alive through queued consumer work, including release in a later callback and exception cleanup.
Retained views may outlive the session. Imported CuPy storage used on SDK callback streams
is protected when its final view is released elsewhere, including another partition, a different
thread, or cyclic garbage collection. SDK stream handles are borrowed during compute(): do not
submit through a saved handle from start(), stop(), or a finalizer. Tracking covers streams
registered when a Tensor is received or exported. Before using an existing view on a different
stream, convert the retained Tensor again on that stream, or synchronize that stream before
releasing the view. Inside callbacks, keep a
stream you create alive until the callback returns, and synchronize it before releasing a view
you used on it. Keep that stream on the operator so it survives callback-exit completion recording.
CUDA array-interface, no-sync C exchange, and no-sync or thread-relative DLPack exports require
conservative device completion at final release.

Readiness is captured at import. Finish producing the array before importing it; after subsequent
writes on other streams, import again or emit the array directly. Importing a read-only Tensor
returns the same object; importing a writable Tensor creates a new read-only wrapper.

The interchange protocol is selected by precedence: the DLPack C exchange API [1]_ first, then the
Python DLPack protocol [1]_, [2]_, then ``__cuda_array_interface__`` [3]_, then
``__array_interface__`` [4]_. DLPack is preferred because it transfers ownership (so the imported
Tensor keeps its data alive independently of the source), has explicit stream-synchronization
semantics, and is the array-API-endorsed interchange path. The C exchange API avoids the per-call
Python capsule path when a producer type exposes ``__dlpack_c_exchange_api__``.

Examples
--------
With Tensor input/output ports declared, prefer the SDK stream in ``compute(ctx)``::

    import cupy as cp

    with ctx.stream():
        view = cp.from_dlpack(self.inp.receive_data())
        self.out.emit(view * 2)

If you need a separate stream, keep it on the operator across callbacks and synchronize before
dropping a view, including a view retained from an earlier callback::

    with ctx.stream():
        if not hasattr(self, "_cupy_stream"):
            self._cupy_stream = cp.cuda.Stream(non_blocking=True)
        with self._cupy_stream:
            view = cp.from_dlpack(self.inp.receive_data())
            try:
                result = view * 2
            finally:
                self._cupy_stream.synchronize()
        del view
        self.out.emit(result)

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
stream : int or object, optional
    CUDA import stream, as an integer or an object exposing ``__cuda_stream__()``.
    Defaults to the activation stream inside ``compute()`` and the legacy default stream outside
    a callback. ``stream=-1`` is rejected because it supplies no readiness guarantee.
    Integer ``0`` is invalid; use ``1`` or ``cupy.cuda.Stream.null`` for the default stream.

Returns
-------
holoscan.Tensor

Notes
-----
Zero-copy imported storage is permanently read-only.

CUDA imports negotiate ``__dlpack__(stream=...)`` on the selected stream and record readiness
with a CUDA event. The Tensor retains the producer's storage until its views and queued consumer
work have finished. SDK callbacks snapshot consumer completion before their streams are destroyed.
Outside callbacks, keep external consumer streams alive until their exported views are released.
For work on the SDK callback stream, release retained views in the operator that last used them,
or synchronize that operator's stream before release elsewhere. Inside callbacks, keep a stream
you create alive until the callback returns, and synchronize it before releasing a view you used
on it. Keep that stream on the operator so it survives callback-exit completion recording.

Examples
--------
With Tensor input/output ports declared, prefer the SDK stream in ``compute(ctx)``::

    import cupy as cp

    with ctx.stream():
        view = cp.from_dlpack(self.inp.receive_data())
        self.out.emit(view * 2)

If you need a separate stream, keep it on the operator across callbacks and synchronize before
dropping a view, including a view retained from an earlier callback::

    with ctx.stream():
        if not hasattr(self, "_cupy_stream"):
            self._cupy_stream = cp.cuda.Stream(non_blocking=True)
        with self._cupy_stream:
            view = cp.from_dlpack(self.inp.receive_data())
            try:
                result = view * 2
            finally:
                self._cupy_stream.synchronize()
        del view
        self.out.emit(result)

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
    Requested ``(device_type, device_id)``. A different device requests an explicit copy.
copy : bool or None, optional
    Request an owning contiguous copy before export. ``copy=False`` rejects a different
    ``dl_device`` request.

Returns
-------
PyCapsule

Raises
------
BufferError
    If the tensor is read-only and ``max_version`` does not select a versioned capsule. Legacy
    DLPack cannot carry read-only authority, so Holoscan rejects rather than widening access.

Notes
-----
For imported CUDA storage, ``stream=2`` and ``stream=-1`` use conservative device retirement.
``stream=-1`` still skips producer-readiness ordering; the caller must establish it separately.
Prefer explicit streams for precise retirement.
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
accessing it on a device tensor raises ``AttributeError``. The dictionary carries no stream, so a
readiness-aware tensor first waits on the host for its producer's work, such as a pinned-host
tensor written on a CUDA stream. A tensor written on the host returns at once.
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
