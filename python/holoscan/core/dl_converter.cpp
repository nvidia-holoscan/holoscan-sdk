/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dl_converter.hpp"

#include <cuda_runtime.h>

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "gxf/std/dlpack_utils.hpp"  // nvidia::gxf::numpyTypestr
#include "holoscan/core/common.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/utils/cuda_macros.hpp"

using pybind11::literals::operator""_a;  // NOLINT(misc-unused-using-decls)

namespace holoscan {

void set_array_interface(const py::object& obj,
                         const std::shared_ptr<DLManagedTensorContext>& ctx) {
  DLTensor& dl_tensor = ctx->tensor.dl_tensor;

  if (dl_tensor.data != nullptr) {
    // Prepare the array interface items

    // Main items
    auto maybe_type_str = nvidia::gxf::numpyTypestr(dl_tensor.dtype);
    if (!maybe_type_str) {
      throw std::runtime_error("Unable to determine NumPy dtype from DLPack tensor");
    }
    const char* type_str = maybe_type_str.value();
    py::tuple shape = array2pytuple<pybind11::int_>(dl_tensor.shape, dl_tensor.ndim);
    py::str typestr = py::str(type_str);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    py::tuple data = pybind11::make_tuple(py::int_(reinterpret_cast<uint64_t>(dl_tensor.data)),
                                          py::bool_(false));
    // Optional items
    py::object strides = py::none();
    if (dl_tensor.strides != nullptr) {
      const int32_t strides_length = dl_tensor.ndim;
      py::tuple strides_tuple(strides_length);
      // The array interface's stride is using bytes, not element size, so we need to multiply it by
      // the element size.
      auto& strides_arr = dl_tensor.strides;
      int64_t elem_size = dl_tensor.dtype.bits / 8;
      for (int index = 0; index < strides_length; ++index) {
        const auto& value = strides_arr[index];
        strides_tuple[index] = py::int_(value * elem_size);
      }

      strides = strides_tuple;
    }
    py::list descr;
    descr.append(py::make_tuple("", typestr));

    // Depending on container's memory type, expose either array_interface or cuda_array_interface
    switch (dl_tensor.device.device_type) {
      case kDLCPU:
      case kDLCUDAHost: {
        // Reference: https://numpy.org/doc/stable/reference/arrays.interface.html
        obj.attr("__array_interface__") = py::dict{"shape"_a = shape,
                                                   "typestr"_a = typestr,
                                                   "data"_a = data,
                                                   "version"_a = py::int_(3),
                                                   "strides"_a = strides,
                                                   "descr"_a = descr};
      } break;
      case kDLCUDA:
      case kDLCUDAManaged: {
        // Reference: https://numba.readthedocs.io/en/stable/cuda/cuda_array_interface.html
        // TODO(gbae): Add support for stream instead of always using the default stream
        obj.attr("__cuda_array_interface__") = py::dict{
            "shape"_a = shape,
            "typestr"_a = typestr,
            "data"_a = data,
            "version"_a = py::int_(3),
            "strides"_a = strides,
            "descr"_a = descr,
            "mask"_a = py::none(),
            "stream"_a = 1  // 1: The legacy default stream
        };
      } break;
      default:
        break;
    }
  } else {
    switch (dl_tensor.device.device_type) {
      case kDLCPU:
      case kDLCUDAHost: {
        if (py::hasattr(obj, "__array_interface__")) {
          py::delattr(obj, "__array_interface__");
        }
      } break;
      case kDLCUDA:
      case kDLCUDAManaged: {
        if (py::hasattr(obj, "__cuda_array_interface__")) {
          py::delattr(obj, "__cuda_array_interface__");
        }
      } break;
      default:
        break;
    }
  }
}

void synchronize_streams(cudaStream_t stream1, cudaStream_t stream2) {
  cudaEvent_t stream1_event{};
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaEventCreateWithFlags(&stream1_event, cudaEventDisableTiming),
                                 "Failure during call to cudaEventCreateWithFlags");
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaEventRecord(stream1_event, stream1),
                                 "Failure during call to cudaEventRecord");
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaStreamWaitEvent(stream2, stream1_event, 0),
                                 "Failure during call to cudaStreamWaitEvent");
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaEventDestroy(stream1_event),
                                 "Failure during call to cudaEventDestroy");
}

void process_dlpack_stream(py::object stream_obj) {
  int64_t stream_id = 1;  // legacy default stream
  cudaStream_t stream_ptr = nullptr;

  if (stream_obj.is_none()) {
    stream_obj = py::int_(1);  // legacy default stream
  } else if (py::isinstance<py::int_>(stream_obj)) {
    stream_id = stream_obj.cast<int64_t>();
    if (stream_id < -1) {
      throw std::runtime_error(
          "Invalid stream, valid stream should be -1 (non-blocking), 1 (legacy default stream), 2 "
          "(per-thread default stream), or a positive integer (stream pointer)");
    }
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
    if (stream_id > 2) {
      stream_ptr = reinterpret_cast<cudaStream_t>(stream_id);
    }
  } else {
    throw std::runtime_error(fmt::format("Invalid stream type: should be int type but given '{}'",
                                         std::string(py::str(stream_obj))));
  }

  // Wait for the current stream to finish before the provided stream starts consuming the memory.
  cudaStream_t curr_stream_ptr = nullptr;  // legacy stream
  if (stream_id >= 0 && curr_stream_ptr != stream_ptr) {
    synchronize_streams(curr_stream_ptr, stream_ptr);
  }
}

py::capsule py_dlpack(Tensor* tensor, py::object stream,
                      std::optional<std::tuple<int, int>> max_version,
                      std::optional<std::tuple<DLDeviceType, int>> dl_device,
                      std::optional<bool> copy) {
  // determine stream and synchronize it with the default stream if necessary
  process_dlpack_stream(std::move(stream));

  // Check if we should use versioned DLPack based on max_version
  bool use_versioned = false;  // Default to unversioned for backward compatibility
  if (max_version.has_value()) {
    auto [major, minor] = max_version.value();
    if (major >= 1) {
      use_versioned = true;
    }
  }
  // Note: Follow the array API standard for exception type that must be returned (BufferError)
  // https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html#array_api.array.__dlpack__
  int device_type = -1;
  int device_id = -1;
  if (dl_device.has_value()) {
    auto [dt, did] = dl_device.value();
    device_type = dt;
    device_id = did;

    // Check if the requested device is the same as the tensor's device
    py::tuple tensor_dl_device = py_dlpack_device(tensor);
    int tensor_device_type = tensor_dl_device[0].cast<int>();
    int tensor_device_id = tensor_dl_device[1].cast<int>();
    if (!(device_type == tensor_device_type && device_id == tensor_device_id)) {
      // raise BufferError to follow the array API standard
      throw pybind11::buffer_error(
          "use of dl_device to copy to a different device is not currently supported");
    }
  }

  // Handle copy parameter (if provided)
  bool make_copy = false;
  uint64_t flags = 0;
  if (copy.has_value() && copy.value()) {
    // TODO (grelee): implement handling for the user-provided `copy` keyword argument.
    // flags |= DLPACK_FLAG_BITMASK_IS_COPIED;
    throw pybind11::buffer_error("copy=True is not implemented");
  }

  // There is also a read-only flag, but we don't use it for Holoscan tensors.
  // flags |= DLPACK_FLAG_BITMASK_READ_ONLY;

  // Get the DLPack tensor (either versioned or unversioned)
  void* dl_managed_tensor_ptr;
  const char* capsule_name;

  if (use_versioned) {
    auto* dl_managed_tensor_ver = tensor->to_dlpack_versioned();
    dl_managed_tensor_ptr = dl_managed_tensor_ver;
    capsule_name = dlpack_versioned_capsule_name;

    // Handle dl_device parameter for versioned DLPack
    if (device_type != -1) {
      // Update the device information in the DLTensor if a copy was made
      dl_managed_tensor_ver->dl_tensor.device.device_type = static_cast<DLDeviceType>(device_type);
      dl_managed_tensor_ver->dl_tensor.device.device_id = device_id;
    }

    // set any flags based on copy parameter, etc.
    dl_managed_tensor_ver->flags = flags;
  } else {
    auto* dl_managed_tensor = tensor->to_dlpack();
    dl_managed_tensor_ptr = dl_managed_tensor;
    capsule_name = dlpack_capsule_name;

    // Handle dl_device parameter for unversioned DLPack
    if (device_type != -1) {
      // Update the device information in the DLTensor if a copy was made
      dl_managed_tensor->dl_tensor.device.device_type = static_cast<DLDeviceType>(device_type);
      dl_managed_tensor->dl_tensor.device.device_id = device_id;
    }
  }

  // Create capsule with appropriate name and deleter
  py::capsule dlpack_capsule(dl_managed_tensor_ptr, capsule_name, [](PyObject* ptr) {
    const char* name = PyCapsule_GetName(ptr);
    if (name == nullptr)
      return;

    bool is_versioned = (strcmp(name, dlpack_versioned_capsule_name) == 0);
    const char* valid_name = is_versioned ? dlpack_versioned_capsule_name : dlpack_capsule_name;
    const char* used_name =
        is_versioned ? used_dlpack_versioned_capsule_name : used_dlpack_capsule_name;

    if (PyCapsule_IsValid(ptr, valid_name) != 0) {
      if (is_versioned) {
        auto* dl_managed_tensor_ver =
            static_cast<DLManagedTensorVersioned*>(PyCapsule_GetPointer(ptr, valid_name));
        if (dl_managed_tensor_ver != nullptr) {
          dl_managed_tensor_ver->deleter(dl_managed_tensor_ver);
        }
      } else {
        auto* dl_managed_tensor =
            static_cast<DLManagedTensor*>(PyCapsule_GetPointer(ptr, valid_name));
        if (dl_managed_tensor != nullptr) {
          dl_managed_tensor->deleter(dl_managed_tensor);
        }
      }
    }
  });

  return dlpack_capsule;
}

py::tuple py_dlpack_device(Tensor* tensor) {
  auto& dl_tensor = tensor->dl_ctx()->tensor.dl_tensor;
  auto& device = dl_tensor.device;
  auto& device_type = device.device_type;
  auto& device_id = device.device_id;
  return py::make_tuple(py::int_(static_cast<int>(device_type)), py::int_(device_id));
}

}  // namespace holoscan
