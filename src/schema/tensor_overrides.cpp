/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime_api.h>

#include <fmt/format.h>

#include <array>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/domain/tensor_access.hpp"
#include "core/domain/tensor_layout_validation.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/holoscan_schema_export.h"
#include "holoscan/schema/tensor_materialization.hpp"

#include "tensor_generated.h"

namespace {

class SharedOwnerMemoryReference final : public holoscan::TensorMemoryReference {
 public:
  explicit SharedOwnerMemoryReference(std::shared_ptr<void> owner) : owner_(std::move(owner)) {}

 private:
  std::shared_ptr<void> owner_;
};

[[noreturn]] void throw_schema_error(const char* message) {
  throw std::runtime_error(message);
}

void throw_if_cuda_failed(cudaError_t result, const char* message) {
  if (result != cudaSuccess) {
    throw std::runtime_error(fmt::format("{}: {}", message, cudaGetErrorString(result)));
  }
}

bool should_materialize_tensor(const holoscan::Tensor& tensor) {
  if (holoscan::schema::WithTensorMaterialization::enabled()) {
    return true;
  }
  if (holoscan::schema::WithTensorGather::enabled()) {
    const auto threshold = holoscan::schema::WithTensorGather::size_threshold_bytes();
    return threshold > 0 && static_cast<std::size_t>(tensor.nbytes()) < threshold;
  }
  return false;
}

holoscan::expected<void, holoscan::Error> validate_tensor_metadata(const holoscan::Tensor& tensor) {
  using holoscan::Error;
  using holoscan::ErrorCode;
  using holoscan::make_unexpected;

  const holoscan::detail::TensorAccessSnapshot snapshot =
      holoscan::detail::TensorAccess::snapshot(tensor);
  if (!snapshot.initialized) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Tensor metadata source must be initialized"});
  }
  if (!snapshot.valid) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "Tensor metadata layout is invalid"});
  }

  const auto rank = tensor.ndim();
  if (rank < 0 || static_cast<std::size_t>(rank) > holoscan::TensorViewMetadata::kMaxRank) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Tensor metadata rank exceeds descriptor limit"});
  }

  const auto device = snapshot.device;
  switch (device.device_type) {
    case kDLCPU:
    case kDLCUDA:
    case kDLCUDAHost:
    case kDLCUDAManaged:
      break;
    default:
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "Tensor metadata device is unsupported"});
  }
  if (device.device_id < 0) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Tensor metadata device id is negative"});
  }

  return {};
}

const std::uint8_t* tensor_data_begin(const holoscan::Tensor& tensor) {
  return static_cast<const std::uint8_t*>(tensor.data());
}

std::shared_ptr<void> make_cuda_owner(void* ptr) {
  return std::shared_ptr<void>(ptr, [](void* p) {
    if (p != nullptr) {
      cudaFree(p);
    }
  });
}

std::shared_ptr<void> make_cuda_host_owner(void* ptr) {
  return std::shared_ptr<void>(ptr, [](void* p) {
    if (p != nullptr) {
      cudaFreeHost(p);
    }
  });
}

}  // namespace

namespace holoscan::schema {

expected<flatbuffers::Offset<Tensor>, Error> create_tensor_metadata(
    flatbuffers::FlatBufferBuilder& builder, const holoscan::Tensor& tensor) noexcept {
  auto valid = validate_tensor_metadata(tensor);
  if (!valid) {
    return make_unexpected(std::move(valid).error());
  }

  try {
    const auto shape = tensor.shape_span();
    const auto shape_offset = builder.CreateVector(shape.data(), shape.size());

    const auto strides = tensor.dlpack_strides_span();
    flatbuffers::Offset<flatbuffers::Vector<std::int64_t>> strides_offset = 0;
    if (!strides.empty()) {
      strides_offset = builder.CreateVector(strides.data(), strides.size());
    }

    const auto dtype_value = tensor.dtype();
    const DLDataType dtype{
        static_cast<DLDataTypeCode>(dtype_value.code), dtype_value.bits, dtype_value.lanes};
    const auto device_value = tensor.device();
    const DLDevice device{static_cast<DLDeviceType>(device_value.device_type),
                          device_value.device_id};

    return CreateTensor(builder,
                        /*data=*/0,
                        shape_offset,
                        &dtype,
                        &device,
                        static_cast<std::uint32_t>(tensor.ndim()),
                        strides_offset);
  } catch (const std::bad_alloc&) {
    return make_unexpected(
        Error{ErrorCode::kResourceExhausted, "Tensor metadata exceeds bounded storage"});
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kCodecError, "Tensor metadata encoding failed"});
  }
}

// Keep the complete wire-metadata and overflow validation in one linear decoder path.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<TensorViewMetadata, Error> tensor_view_metadata(const Tensor& tensor) noexcept {
  const auto* shape = tensor.shape();
  if (shape == nullptr) {
    return make_unexpected(Error{ErrorCode::kCodecError, "Tensor metadata shape is missing"});
  }
  const auto* dtype = tensor.dtype();
  if (dtype == nullptr) {
    return make_unexpected(Error{ErrorCode::kCodecError, "Tensor metadata dtype is missing"});
  }
  const auto* device = tensor.device();
  if (device == nullptr) {
    return make_unexpected(Error{ErrorCode::kCodecError, "Tensor metadata device is missing"});
  }
  const auto* data = tensor.data();
  if (data != nullptr && !data->empty()) {
    return make_unexpected(
        Error{ErrorCode::kCodecError, "descriptor-backed Tensor metadata contains embedded bytes"});
  }

  const std::uint32_t rank = tensor.ndim();
  if (rank != shape->size()) {
    return make_unexpected(
        Error{ErrorCode::kCodecError, "Tensor metadata ndim does not match shape size"});
  }
  if (rank > TensorViewMetadata::kMaxRank) {
    return make_unexpected(
        Error{ErrorCode::kCodecError, "Tensor metadata rank exceeds descriptor limit"});
  }

  TensorViewMetadata metadata{};
  metadata.rank = rank;
  metadata.dtype = ::DLDataType{.code = static_cast<std::uint8_t>(dtype->code()),
                                .bits = dtype->bits(),
                                .lanes = dtype->lanes()};
  metadata.device = ::DLDevice{.device_type = static_cast<::DLDeviceType>(device->device_type()),
                               .device_id = device->device_id()};
  if (!holoscan::detail::valid_tensor_dtype(metadata.dtype)) {
    return make_unexpected(Error{ErrorCode::kCodecError, "Tensor metadata dtype is invalid"});
  }
  switch (metadata.device.device_type) {
    case kDLCPU:
    case kDLCUDA:
    case kDLCUDAHost:
    case kDLCUDAManaged:
      break;
    default:
      return make_unexpected(
          Error{ErrorCode::kCodecError, "Tensor metadata device is unsupported"});
  }
  if (metadata.device.device_id < 0) {
    return make_unexpected(Error{ErrorCode::kCodecError, "Tensor metadata device id is negative"});
  }

  for (std::uint32_t index = 0U; index < rank; ++index) {
    const auto dimension_value = shape->Get(index);
    metadata.shape[index] = dimension_value;
  }

  const auto* strides = tensor.strides();
  std::array<std::int64_t, TensorViewMetadata::kMaxRank> element_strides{};
  std::optional<std::span<const std::int64_t>> layout_strides;
  if (strides != nullptr) {
    if (strides->size() != rank) {
      return make_unexpected(
          Error{ErrorCode::kCodecError, "Tensor metadata strides size does not match rank"});
    }
    metadata.has_explicit_byte_strides = true;
    for (std::uint32_t index = 0U; index < rank; ++index) {
      element_strides[index] = strides->Get(index);
    }
    layout_strides = std::span<const std::int64_t>{element_strides.data(), rank};
  }
  const auto layout = holoscan::detail::validate_tensor_layout(
      std::span<const std::int64_t>{metadata.shape.data(), rank},
      metadata.dtype,
      layout_strides,
      holoscan::detail::TensorStrideUnit::kElements);
  if (!layout) {
    return make_unexpected(Error{ErrorCode::kCodecError, "Tensor metadata layout is invalid"});
  }
  metadata.byte_size = layout->byte_size;
  if (layout_strides) {
    for (std::uint32_t index = 0U; index < rank; ++index) {
      const auto stride = static_cast<std::uint64_t>(element_strides[index]);
      if (stride > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) /
                       layout->item_bytes) {
        return make_unexpected(
            Error{ErrorCode::kCodecError, "Tensor metadata byte stride overflows int64"});
      }
      metadata.byte_strides[index] = static_cast<std::int64_t>(stride * layout->item_bytes);
    }
  }
  return metadata;
}

// The flatc-generated declaration of Tensor::Pack cannot carry HOLOSCAN_SCHEMA_EXPORT, so the
// out-of-line definition is annotated here to export this symbol from the hidden-visibility DSO.
HOLOSCAN_SCHEMA_EXPORT ::flatbuffers::Offset<Tensor> Tensor::Pack(
    ::flatbuffers::FlatBufferBuilder& builder, const holoscan::Tensor* value,
    const ::flatbuffers::rehasher_function_t*) {
  if (value == nullptr || !holoscan::detail::TensorAccess::context(*value)) {
    return 0;
  }

  const auto shape = value->shape_span();
  ::flatbuffers::Offset<::flatbuffers::Vector<std::int64_t>> shape_offset = 0;
  if (!shape.empty()) {
    shape_offset = builder.CreateVector(shape.data(), shape.size());
  }

  const auto strides = value->dlpack_strides_span();
  ::flatbuffers::Offset<::flatbuffers::Vector<std::int64_t>> strides_offset = 0;
  if (!strides.empty()) {
    strides_offset = builder.CreateVector(strides.data(), strides.size());
  }

  const auto dtype_value = value->dtype();
  const DLDataType dtype{
      static_cast<DLDataTypeCode>(dtype_value.code), dtype_value.bits, dtype_value.lanes};
  const auto device_value = value->device();
  const DLDevice device{static_cast<DLDeviceType>(device_value.device_type),
                        device_value.device_id};

  ::flatbuffers::Offset<::flatbuffers::Vector<std::uint8_t>> data_offset = 0;
  if (should_materialize_tensor(*value) && value->data() != nullptr && value->nbytes() > 0) {
    const auto total_bytes = static_cast<std::size_t>(value->nbytes());
    std::uint8_t* destination = nullptr;
    data_offset = builder.CreateUninitializedVector<std::uint8_t>(total_bytes, &destination);

    switch (device_value.device_type) {
      case kDLCPU:
      case kDLCUDAHost:
        std::memcpy(destination, tensor_data_begin(*value), total_bytes);
        break;
      case kDLCUDA:
        throw_if_cuda_failed(
            cudaMemcpy(destination, tensor_data_begin(*value), total_bytes, cudaMemcpyDeviceToHost),
            "failed to copy CUDA Tensor bytes into FlatBuffer");
        break;
      case kDLCUDAManaged:
        throw_if_cuda_failed(
            cudaMemcpy(destination, tensor_data_begin(*value), total_bytes, cudaMemcpyDefault),
            "failed to copy managed Tensor bytes into FlatBuffer");
        break;
      default:
        throw_schema_error("unsupported Tensor device type for FlatBuffers materialization");
    }
  }

  return CreateTensor(builder,
                      data_offset,
                      shape_offset,
                      &dtype,
                      &device,
                      static_cast<std::uint32_t>(value->ndim()),
                      strides_offset);
}

HOLOSCAN_SCHEMA_EXPORT void Tensor::UnPackTo(holoscan::Tensor* value,
                                             const ::flatbuffers::resolver_function_t*) const {
  if (value == nullptr) {
    throw_schema_error("cannot unpack FlatBuffer Tensor into nullptr");
  }
  if (shape() == nullptr) {
    throw_schema_error("cannot unpack FlatBuffer Tensor without shape metadata");
  }
  if (dtype() == nullptr) {
    throw_schema_error("cannot unpack FlatBuffer Tensor without dtype metadata");
  }
  if (device() == nullptr) {
    throw_schema_error("cannot unpack FlatBuffer Tensor without device metadata");
  }

  const std::uint32_t rank = ndim();
  if (rank > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
    throw_schema_error("cannot unpack FlatBuffer Tensor whose ndim exceeds int32");
  }
  if (rank != shape()->size()) {
    throw_schema_error("cannot unpack FlatBuffer Tensor whose ndim does not match shape size");
  }
  if (strides() != nullptr && strides()->size() != rank) {
    throw_schema_error("cannot unpack FlatBuffer Tensor whose strides size does not match ndim");
  }

  auto context = std::make_shared<holoscan::DLManagedTensorContext>();
  context->dl_shape.assign(shape()->begin(), shape()->end());
  context->tensor.dl_tensor.shape = context->dl_shape.data();
  context->tensor.dl_tensor.dtype = ::DLDataType{.code = static_cast<std::uint8_t>(dtype()->code()),
                                                 .bits = dtype()->bits(),
                                                 .lanes = dtype()->lanes()};
  context->tensor.dl_tensor.device =
      ::DLDevice{.device_type = static_cast<::DLDeviceType>(device()->device_type()),
                 .device_id = device()->device_id()};
  context->tensor.dl_tensor.ndim = static_cast<std::int32_t>(rank);
  context->tensor.dl_tensor.byte_offset = 0;
  context->tensor.flags = 0;

  if (strides() != nullptr) {
    context->dl_strides.assign(strides()->begin(), strides()->end());
    context->tensor.dl_tensor.strides = context->dl_strides.data();
  } else {
    context->tensor.dl_tensor.strides = nullptr;
  }

  if (data() != nullptr && data()->size() > 0) {
    const auto byte_count = data()->size();
    switch (context->tensor.dl_tensor.device.device_type) {
      case kDLCPU: {
        auto owner = std::make_shared<std::vector<std::uint8_t>>(data()->begin(), data()->end());
        context->tensor.dl_tensor.data = owner->data();
        context->memory_reference = std::make_shared<SharedOwnerMemoryReference>(std::move(owner));
        break;
      }
      case kDLCUDAHost: {
        void* host_ptr = nullptr;
        throw_if_cuda_failed(cudaMallocHost(&host_ptr, byte_count),
                             "failed to allocate CUDA host memory for FlatBuffer Tensor unpack");
        std::memcpy(host_ptr, data()->data(), byte_count);
        context->tensor.dl_tensor.data = host_ptr;
        context->memory_reference =
            std::make_shared<SharedOwnerMemoryReference>(make_cuda_host_owner(host_ptr));
        break;
      }
      case kDLCUDAManaged: {
        void* managed_ptr = nullptr;
        throw_if_cuda_failed(cudaMallocManaged(&managed_ptr, byte_count),
                             "failed to allocate managed memory for FlatBuffer Tensor unpack");
        std::memcpy(managed_ptr, data()->data(), byte_count);
        context->tensor.dl_tensor.data = managed_ptr;
        context->memory_reference =
            std::make_shared<SharedOwnerMemoryReference>(make_cuda_owner(managed_ptr));
        break;
      }
      case kDLCUDA: {
        void* device_ptr = nullptr;
        throw_if_cuda_failed(cudaMalloc(&device_ptr, byte_count),
                             "failed to allocate CUDA memory for FlatBuffer Tensor unpack");
        throw_if_cuda_failed(
            cudaMemcpy(device_ptr, data()->data(), byte_count, cudaMemcpyHostToDevice),
            "failed to copy FlatBuffer Tensor bytes to CUDA memory");
        context->tensor.dl_tensor.data = device_ptr;
        context->memory_reference =
            std::make_shared<SharedOwnerMemoryReference>(make_cuda_owner(device_ptr));
        break;
      }
      default:
        throw_schema_error("unsupported Tensor device type for FlatBuffers unpack");
    }
    context->tensor.flags |= DLPACK_FLAG_BITMASK_IS_COPIED;
  } else {
    context->tensor.dl_tensor.data = nullptr;
  }

  holoscan::detail::TensorAccess::replace_context(
      *value, std::move(context), holoscan::detail::TensorAuthority::kLocalWritable);
}

}  // namespace holoscan::schema
