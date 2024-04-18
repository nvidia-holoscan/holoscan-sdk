/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef PYHOLOSCAN_CORE_TENSOR_PYDOC_HPP
#define PYHOLOSCAN_CORE_TENSOR_PYDOC_HPP

#include <string>

#include "../macros.hpp"

namespace holoscan::doc {

namespace DLDevice {

// Constructor
PYDOC(DLDevice, R"doc(
DLDevice class.
)doc")

PYDOC(device_type, R"doc(
The device type (`DLDeviceType`).

The following device types are supported:

- `DLDeviceType.DLCPU`: system memory (kDLCPU)
- `DLDeviceType.DLCUDA`: CUDA GPU memory (kDLCUDA)
- `DLDeviceType.DLCUDAHost`: CUDA pinned memory (kDLCUDAHost)
- `DLDeviceType.DLCUDAManaged`: CUDA managed memory (kDLCUDAManaged)
)doc")

PYDOC(device_id, R"doc(
The device id (int).
)doc")

}  // namespace DLDevice

namespace Tensor {

// Constructor
PYDOC(Tensor, R"doc(
Base class representing a Holoscan Tensor.
)doc")

PYDOC(as_tensor, R"doc(
Convert a Python object to a Tensor.

Parameters
==========
object : array-like
    An object such as a NumPy array, CuPy array, PyTorch tensor, etc. supporting one of the
    supported protocols.

Returns
=======
holocan.Tensor

Notes
=====
For device arrays, this method first attempts to convert via ``__cuda_array_interface__`` [1]_,
but falls back to the DLPack protocol [2]_, [3]_ if it is unavailable.

For host arrays, this method first attempts to convert via the DLPack protocol, but falls back to
the ``__array_interface__`` [3]_ if it is unavailable.

References
==========
.. [1] https://numpy.org/doc/stable/reference/arrays.interface.html
.. [2] https://dmlc.github.io/dlpack/latest/python_spec.html
.. [3] https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.array.__dlpack__.html
.. [4] https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
)doc")

PYDOC(from_dlpack, R"doc(
Convert a Python object to a Tensor via the DLPack protocol [1]_, [2]_.

Parameters
==========
object : array-like
    An object such as a NumPy array, CuPy array, PyTorch tensor, etc. supporting one of the
    supported protocols.

Returns
=======
holocan.Tensor

References
==========
.. [1] https://dmlc.github.io/dlpack/latest/python_spec.html
.. [2] https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.array.__dlpack__.html
)doc")

PYDOC(py_dlpack, R"doc(
Convert a Tensor to __dlpack__ function.
)doc")

PYDOC(py_dlpack_device, R"doc(
Convert a Tensor to __dlpack_device__ function.
)doc")

PYDOC(ndim, R"doc(
The number of dimensions.
)doc")

PYDOC(data, R"doc(
PyCapsule handle to the tensor's data.
)doc")

PYDOC(device, R"doc(
DLDevice struct with device type/id information.
)doc")

PYDOC(dtype, R"doc(
DLDataType struct with data type information.

For details of the DLDataType struct see the DLPack documentation:
https://dmlc.github.io/dlpack/latest/c_api.html#_CPPv410DLDataType
)doc")

PYDOC(shape, R"doc(
Tuple containing the tensor's shape.
)doc")

PYDOC(strides, R"doc(
Tuple containing the tensor's strides (in elements).
)doc")

PYDOC(size, R"doc(
The total number of elements in the tensor.
)doc")

PYDOC(itemsize, R"doc(
The size of a single element of the tensor's data.
)doc")

PYDOC(nbytes, R"doc(
The size of the tensor's data in bytes.
)doc")

PYDOC(dlpack, R"doc(
Exports the array for consumption by ``from_dlpack()`` as a DLPack capsule.

Please refer to the DLPack and Python array API standard documentation for more details:

- https://dmlc.github.io/dlpack/latest/python_spec.html
- https://data-apis.org/array-api/latest/design_topics/data_interchange.html
- https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html?highlight=__dlpack__

Parameters
----------
stream : Optional[Union[int, Any]]
    For CUDA, a Python integer representing a pointer to a stream, on devices that support streams.
    ``stream`` is provided by the consumer to the producer to instruct the producer to ensure that
    operations can safely be performed on the array
    (e.g., by inserting a dependency between streams via “wait for event”).
    The pointer must be a positive integer or `-1`. If stream is `-1`, the value may be used by the
    consumer to signal “producer must not perform any synchronization”.
    The ownership of the stream stays with the consumer.
    On CPU and other device types without streams, only None is accepted.

    - None: producer must assume the legacy default stream (default).
    - 1: the legacy default stream.
    - 2: the per-thread default stream.
    - > 2: stream number represented as a Python integer.
    - 0 is disallowed due to its ambiguity: 0 could mean either None, 1, or 2.

Returns
-------
dlpack : PyCapsule
    A PyCapsule object with the DLPack data.
)doc")

PYDOC(dlpack_device, R"doc(
Returns device type and device ID in DLPack format. Meant for use within ``from_dlpack()``.

Please refer to the DLPack and Python array API standard documentation for more details:

- https://dmlc.github.io/dlpack/latest/python_spec.html
- https://data-apis.org/array-api/latest/design_topics/data_interchange.html
- https://data-apis.org/array-api/2022.12/API_specification/generated/array_api.array.__dlpack_device__.html

Returns
-------
device : Tuple[enum.Enum, int]
    A tuple of (device type, device id) in DLPack format.
)doc")

}  // namespace Tensor

}  // namespace holoscan::doc

#endif  // PYHOLOSCAN_CORE_TENSOR_PYDOC_HPP
