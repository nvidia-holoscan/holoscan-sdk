/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/domain/dlpack_utils.hpp>

#include <cuda_runtime_api.h>

#include <fmt/format.h>

#include <charconv>
#include <cstddef>
#include <limits>
#include <string_view>
#include <system_error>
#include <vector>

namespace holoscan {

expected<DLDevice, Error> dldevice_from_pointer(void* ptr) {
  DLDevice device{kDLCUDA, 0};

  cudaPointerAttributes attributes{};
  const cudaError_t status = cudaPointerGetAttributes(&attributes, ptr);
  if (status != cudaSuccess) {
    return make_unexpected(Error{ErrorCode::kFailure,
                                 fmt::format("Failed to determine DLDevice based on pointer: {}",
                                             cudaGetErrorString(status))});
  }

  switch (attributes.type) {
    case cudaMemoryTypeUnregistered:
      device.device_type = kDLCPU;
      break;
    case cudaMemoryTypeHost:
      device = {kDLCUDAHost, attributes.device};
      break;
    case cudaMemoryTypeDevice:
      device = {kDLCUDA, attributes.device};
      break;
    case cudaMemoryTypeManaged:
      device = {kDLCUDAManaged, attributes.device};
      break;
  }
  return device;
}

void calc_strides(const DLTensor& tensor, std::vector<int64_t>& strides, bool to_num_elements) {
  const int64_t ndim = tensor.ndim;
  strides.clear();
  if (ndim <= 0) {
    return;
  }

  // Byte strides use the full element size, accounting for vectorized (multi-lane) dtypes.
  const int64_t elem_size =
      to_num_elements ? 1 : static_cast<int64_t>(dlpack_dtype_item_size_bytes(tensor.dtype));
  if (tensor.strides == nullptr) {
    if (tensor.shape == nullptr) {
      return;
    }
    strides.resize(static_cast<std::size_t>(ndim));
    int64_t step = 1;
    for (int64_t index = ndim - 1; index >= 0; --index) {
      strides[index] = step * elem_size;
      step *= tensor.shape[index];
    }
  } else {
    strides.resize(static_cast<std::size_t>(ndim));
    for (int64_t index = 0; index < ndim; ++index) {
      strides[index] = tensor.strides[index] * elem_size;
    }
  }
}

void copy_dl_tensor_with_owned_metadata(const DLTensor& source, DLTensor& destination,
                                        std::vector<int64_t>& owned_shape,
                                        std::vector<int64_t>& owned_strides,
                                        DLPackMetadataCopyOptions options) {
  destination = source;

  if (source.ndim > 0 && source.shape != nullptr) {
    owned_shape.assign(source.shape, source.shape + source.ndim);
    destination.shape = owned_shape.data();
  } else {
    owned_shape.clear();
    destination.shape = nullptr;
  }

  if (source.ndim > 0 && source.strides != nullptr) {
    owned_strides.assign(source.strides, source.strides + source.ndim);
    destination.strides = owned_strides.data();
  } else if (options.materialize_implicit_strides && source.ndim > 0 && source.shape != nullptr) {
    // DLPack 1.2+ versioned tensors require explicit strides when ndim != 0.
    calc_strides(source, owned_strides, /*to_num_elements=*/true);
    destination.strides = owned_strides.data();
  } else {
    owned_strides.clear();
    destination.strides = nullptr;
  }
}

void copy_dl_tensor_with_owned_metadata(DLManagedTensorContext& context, const DLTensor& source,
                                        DLPackMetadataCopyOptions options) {
  copy_dl_tensor_with_owned_metadata(
      source, context.tensor.dl_tensor, context.dl_shape, context.dl_strides, options);
  context.tensor.version = DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
  context.tensor.flags = 0;
  context.tensor.manager_ctx = nullptr;
  context.tensor.deleter = nullptr;
}

void copy_dl_tensor_with_owned_metadata(DLManagedTensorContext& context,
                                        const DLManagedTensorVersioned& source,
                                        DLPackMetadataCopyOptions options) {
  copy_dl_tensor_with_owned_metadata(
      source.dl_tensor, context.tensor.dl_tensor, context.dl_shape, context.dl_strides, options);
  context.tensor.version = source.version;
  context.tensor.flags = source.flags;
  context.tensor.manager_ctx = nullptr;
  context.tensor.deleter = nullptr;
}

expected<DLDataType, Error> dldatatype_from_typestr(std::string_view typestr) {
  uint8_t code = 0;
  const uint16_t lanes = 1;
  if (!typestr.empty() && typestr.front() == '>') {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "Failed to determine DLDataType from type string: big endian types "
              "are not supported"});
  }
  const std::string_view kind = typestr.size() > 1 ? typestr.substr(1, 1) : std::string_view{};
  if (kind == "i") {
    code = kDLInt;
  } else if (kind == "u") {
    code = kDLUInt;
  } else if (kind == "f") {
    code = kDLFloat;
  } else if (kind == "c") {
    code = kDLComplex;
  } else {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              fmt::format("Failed to determine DLDataType from type string: unsupported "
                          "dtype.kind '{}'",
                          kind)});
  }
  const std::string_view byte_count_text =
      typestr.size() > 2 ? typestr.substr(2) : std::string_view{};
  if (byte_count_text.empty()) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "Failed to determine DLDataType from type string: missing itemsize"});
  }
  uint32_t byte_count = 0;
  const auto* const begin = byte_count_text.data();
  const auto* const end = begin + byte_count_text.size();
  const auto result = std::from_chars(begin, end, byte_count);
  if (result.ec != std::errc() || result.ptr != end ||
      byte_count > std::numeric_limits<uint8_t>::max() / 8U) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              fmt::format("Failed to determine DLDataType from type string: unsupported "
                          "itemsize '{}'",
                          byte_count_text)});
  }
  const auto bits = static_cast<uint8_t>(byte_count * 8U);
  DLDataType data_type{code, bits, lanes};
  return data_type;
}

expected<const char*, Error> numpy_dtype(DLDataType dtype) {
  // future work: consider bfloat16: https://github.com/dmlc/dlpack/issues/45
  // future work: consider other byte-order?
  const uint8_t code = dtype.code;
  const uint8_t bits = dtype.bits;
  switch (code) {
    case kDLInt:
      switch (bits) {
        case 8:
          return "|i1";
        case 16:
          return "<i2";
        case 32:
          return "<i4";
        case 64:
          return "<i8";
        default:
          break;
      }
      break;
    case kDLUInt:
      switch (bits) {
        case 8:
          return "|u1";
        case 16:
          return "<u2";
        case 32:
          return "<u4";
        case 64:
          return "<u8";
        default:
          break;
      }
      break;
    case kDLFloat:
      switch (bits) {
        case 16:
          return "<f2";
        case 32:
          return "<f4";
        case 64:
          return "<f8";
        default:
          break;
      }
      break;
    case kDLComplex:
      switch (bits) {
        case 64:
          return "<c8";
        case 128:
          return "<c16";
        default:
          break;
      }
      break;
    default:
      break;
  }
  return make_unexpected(
      Error{ErrorCode::kInvalidArgument,
            fmt::format("Failed to determine type string from DLDataType: unsupported (code={}, "
                        "bits={})",
                        static_cast<int>(code),
                        static_cast<int>(bits))});
}

expected<void, Error> synchronize_cuda_streams(cudaStream_t producer_stream,
                                               cudaStream_t consumer_stream) {
  if (producer_stream == consumer_stream) {
    return {};
  }

  cudaEvent_t event{};
  cudaError_t status = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
  if (status != cudaSuccess) {
    return make_unexpected(
        Error{ErrorCode::kFailure,
              fmt::format("cudaEventCreateWithFlags failed during Tensor stream sync: {}",
                          cudaGetErrorString(status))});
  }

  auto destroy_event = [&event]() {
    if (event != nullptr) {
      cudaEventDestroy(event);
      event = nullptr;
    }
  };

  status = cudaEventRecord(event, producer_stream);
  if (status != cudaSuccess) {
    destroy_event();
    return make_unexpected(Error{ErrorCode::kFailure,
                                 fmt::format("cudaEventRecord failed during Tensor stream sync: {}",
                                             cudaGetErrorString(status))});
  }

  status = cudaStreamWaitEvent(consumer_stream, event, 0);
  if (status != cudaSuccess) {
    destroy_event();
    return make_unexpected(
        Error{ErrorCode::kFailure,
              fmt::format("cudaStreamWaitEvent failed during Tensor stream sync: {}",
                          cudaGetErrorString(status))});
  }

  destroy_event();
  return {};
}

expected<void, Error> synchronize_cuda_stream_for_host(cudaStream_t producer_stream) {
  const cudaError_t status = cudaStreamSynchronize(producer_stream);
  if (status != cudaSuccess) {
    return make_unexpected(
        Error{ErrorCode::kFailure,
              fmt::format("cudaStreamSynchronize failed during Tensor host sync: {}",
                          cudaGetErrorString(status))});
  }
  return {};
}

}  // namespace holoscan
