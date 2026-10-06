/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/memory/tensor_storage.hpp>

#include "core/domain/tensor_access.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/logger/logger.hpp>
#include <holoscan/memory/allocator.hpp>
#include "core/diagnostic_reporting.hpp"

namespace holoscan {

namespace {

class MemoryBufferTensorMemoryReference final : public TensorMemoryReference {
 public:
  MemoryBufferTensorMemoryReference(std::shared_ptr<MemoryBuffer> memory_buffer,
                                    bool stream_aware) noexcept
      : memory_buffer_(std::move(memory_buffer)), stream_aware_(stream_aware) {}

  bool set_deallocation_stream(cudaStream_t stream) override {
    if (!stream_aware_ || memory_buffer_ == nullptr) {
      return false;
    }
    return memory_buffer_->set_stream(stream);
  }

  [[nodiscard]] bool tracks_deallocation_stream() const noexcept override {
    return stream_aware_ && memory_buffer_ != nullptr;
  }

  bool set_readiness_stream(cudaStream_t stream) override {
    if (!stream_aware_ || memory_buffer_ == nullptr) {
      return false;
    }
    readiness_stream_.store(stream, std::memory_order_release);
    return true;
  }

  bool has_readiness_stream_contract() const noexcept override {
    return stream_aware_ && memory_buffer_ != nullptr;
  }

  [[nodiscard]] cudaStream_t readiness_stream() const noexcept override {
    if (!stream_aware_ || memory_buffer_ == nullptr) {
      return nullptr;
    }
    auto stream = readiness_stream_.load(std::memory_order_acquire);
    return stream != nullptr ? stream : static_cast<cudaStream_t>(memory_buffer_->stream());
  }

  [[nodiscard]] expected<void, Error> synchronize_for_consumer(
      cudaStream_t consumer_stream) override {
    if (!stream_aware_ || memory_buffer_ == nullptr) {
      return {};
    }
    auto readiness_stream = readiness_stream_.load(std::memory_order_acquire);
    if (readiness_stream == nullptr) {
      readiness_stream = static_cast<cudaStream_t>(memory_buffer_->stream());
    }
    return synchronize_cuda_streams(readiness_stream, consumer_stream);
  }

  [[nodiscard]] expected<void, Error> synchronize_for_host() override {
    if (!stream_aware_ || memory_buffer_ == nullptr) {
      return {};
    }
    auto readiness_stream = readiness_stream_.load(std::memory_order_acquire);
    if (readiness_stream == nullptr) {
      readiness_stream = static_cast<cudaStream_t>(memory_buffer_->stream());
    }
    return synchronize_cuda_stream_for_host(readiness_stream);
  }

 private:
  std::shared_ptr<MemoryBuffer> memory_buffer_;
  bool stream_aware_{};
  std::atomic<cudaStream_t> readiness_stream_{nullptr};
};

expected<uint64_t, Error> item_size(DLDataType dtype) {
  if (dtype.bits == 0 || dtype.lanes == 0) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Tensor dtype has zero-sized elements"});
  }
  return dlpack_dtype_item_size_bytes(dtype);
}

expected<void, Error> validate_shape(TensorShapeView shape) {
  if (shape.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "Tensor rank exceeds int32_t range"});
  }
  for (const int64_t dimension : shape) {
    if (dimension < 0) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "Tensor shape dimensions must be non-negative"});
    }
  }
  return {};
}

expected<uint64_t, Error> contiguous_size_bytes(TensorShapeView shape, uint64_t bytes_per_element) {
  uint64_t element_count = 1;
  for (const int64_t dimension : shape) {
    if (dimension == 0) {
      return 0;
    }
    const auto dim = static_cast<uint64_t>(dimension);
    if (element_count > std::numeric_limits<uint64_t>::max() / dim) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "Tensor element count overflows uint64_t"});
    }
    element_count *= dim;
  }
  if (element_count > std::numeric_limits<uint64_t>::max() / bytes_per_element) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Tensor byte size overflows uint64_t"});
  }
  return element_count * bytes_per_element;
}

expected<uint64_t, Error> strided_size_bytes(TensorShapeView shape,
                                             TensorByteStridesView byte_strides,
                                             uint64_t bytes_per_element) {
  if (byte_strides.size() != shape.size()) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Tensor stride rank does not match shape rank"});
  }

  if (shape.empty()) {
    return bytes_per_element;
  }

  uint64_t max_offset = 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (byte_strides[i] < 0) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "Tensor byte strides must be non-negative"});
    }
    if (shape[i] == 0) {
      return 0;
    }
    const auto stride = static_cast<uint64_t>(byte_strides[i]);
    const auto extent = static_cast<uint64_t>(shape[i] - 1);
    if (stride != 0 && extent > std::numeric_limits<uint64_t>::max() / stride) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "Tensor stride extent overflows uint64_t"});
    }
    const uint64_t offset = extent * stride;
    if (max_offset > std::numeric_limits<uint64_t>::max() - offset) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "Tensor strided byte size overflows uint64_t"});
    }
    max_offset += offset;
  }
  if (max_offset > std::numeric_limits<uint64_t>::max() - bytes_per_element) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Tensor strided byte size overflows uint64_t"});
  }
  return max_offset + bytes_per_element;
}

expected<std::vector<int64_t>, Error> dlpack_strides_from_byte_strides(
    TensorByteStridesView byte_strides, uint64_t bytes_per_element) {
  std::vector<int64_t> dl_strides;
  dl_strides.reserve(byte_strides.size());
  for (const int64_t stride : byte_strides) {
    if (stride < 0) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "Tensor byte strides must be non-negative"});
    }
    if (static_cast<uint64_t>(stride) % bytes_per_element != 0) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "Tensor byte strides must be divisible by item size"});
    }
    dl_strides.push_back(stride / static_cast<int64_t>(bytes_per_element));
  }
  return dl_strides;
}

struct DLDeviceMapping {
  bool supported = false;
  DLDevice device{kDLCPU, 0};
};

constexpr DLDeviceMapping map_storage_type_to_dl_device(MemoryStorageType storage_type) noexcept {
  switch (storage_type) {
    case MemoryStorageType::kHost:
      return {true, DLDevice{kDLCUDAHost, 0}};
    case MemoryStorageType::kDevice:
      return {true, DLDevice{kDLCUDA, 0}};
    case MemoryStorageType::kSystem:
      return {true, DLDevice{kDLCPU, 0}};
    case MemoryStorageType::kCudaManaged:
      return {true, DLDevice{kDLCUDAManaged, 0}};
    default:
      return {};
  }
}

static_assert(map_storage_type_to_dl_device(MemoryStorageType::kHost).device.device_type ==
              kDLCUDAHost);
static_assert(map_storage_type_to_dl_device(MemoryStorageType::kDevice).device.device_type ==
              kDLCUDA);
static_assert(map_storage_type_to_dl_device(MemoryStorageType::kSystem).device.device_type ==
              kDLCPU);
static_assert(map_storage_type_to_dl_device(MemoryStorageType::kCudaManaged).device.device_type ==
              kDLCUDAManaged);
static_assert(!map_storage_type_to_dl_device(static_cast<MemoryStorageType>(999)).supported);

// Only device-resident (kDLCUDA) allocations carry a meaningful CUDA device ordinal in DLPack.
// DLPack sets device_id to 0 for CPU, pinned host, and managed memory.
bool storage_type_has_device_ordinal(MemoryStorageType storage_type) noexcept {
  return storage_type == MemoryStorageType::kDevice;
}

expected<DLDevice, Error> device_from_storage_type(
    MemoryStorageType storage_type, void* pointer,
    int32_t device_id_hint = Allocator::kUnknownDeviceId) {
  const DLDeviceMapping mapping = map_storage_type_to_dl_device(storage_type);
  if (!mapping.supported) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Unsupported tensor memory storage type"});
  }
  if (!storage_type_has_device_ordinal(storage_type)) {
    // kDLCPU, kDLCUDAHost, and kDLCUDAManaged keep device_id 0 by DLPack convention.
    return mapping.device;
  }

  DLDevice device = mapping.device;

  // Prefer the device ordinal the allocator or wrapper recorded on the MemoryBuffer. A
  // nonnegative hint is trusted caller-provided provenance; this avoids a
  // cudaPointerGetAttributes driver query on the tensor-construction path.
  if (device_id_hint >= 0) {
    device.device_id = device_id_hint;
    return device;
  }

  // Otherwise derive it from the backing pointer (e.g. externally wrapped memory or allocators
  // that do not bind allocations to a fixed device).
  if (pointer == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "CUDA tensor storage pointer is null"});
  }

  auto maybe_device = dldevice_from_pointer(pointer);
  if (!maybe_device) {
    return forward_error(maybe_device);
  }

  const DLDevice pointer_device = maybe_device.value();
  if (pointer_device.device_type != device.device_type) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              "Tensor storage type does not match the CUDA pointer memory type"});
  }

  device.device_id = pointer_device.device_id;
  return device;
}

expected<void, Error> attach_tensor_context(Tensor& tensor, TensorShapeView shape, DLDataType dtype,
                                            TensorByteStridesView byte_strides,
                                            MemoryStorageType storage_type,
                                            const std::shared_ptr<MemoryBuffer>& memory_buffer,
                                            bool stream_aware_memory_buffer) {
  auto shape_result = validate_shape(shape);
  if (!shape_result) {
    return forward_error(shape_result);
  }

  auto bytes_per_element = item_size(dtype);
  if (!bytes_per_element) {
    return forward_error(bytes_per_element);
  }

  void* pointer = memory_buffer != nullptr ? memory_buffer->pointer() : nullptr;
  const int32_t device_id_hint =
      memory_buffer != nullptr ? memory_buffer->device_id() : Allocator::kUnknownDeviceId;
  auto maybe_device = device_from_storage_type(storage_type, pointer, device_id_hint);
  if (!maybe_device) {
    return forward_error(maybe_device);
  }

  auto ctx = std::make_shared<DLManagedTensorContext>();
  if (memory_buffer != nullptr) {
    ctx->memory_reference = std::make_shared<MemoryBufferTensorMemoryReference>(
        memory_buffer, stream_aware_memory_buffer);
  }
  ctx->dl_shape.assign(shape.begin(), shape.end());
  if (byte_strides) {
    auto maybe_strides =
        dlpack_strides_from_byte_strides(byte_strides.value(), bytes_per_element.value());
    if (!maybe_strides) {
      return forward_error(maybe_strides);
    }
    ctx->dl_strides = std::move(maybe_strides.value());
  }

  auto& dl_tensor = ctx->tensor.dl_tensor;
  dl_tensor.data = pointer;
  dl_tensor.device = maybe_device.value();
  dl_tensor.ndim = static_cast<int32_t>(ctx->dl_shape.size());
  dl_tensor.dtype = dtype;
  dl_tensor.shape = ctx->dl_shape.data();
  dl_tensor.strides = ctx->dl_strides.empty() ? nullptr : ctx->dl_strides.data();
  dl_tensor.byte_offset = 0;
  ctx->tensor.manager_ctx = nullptr;
  ctx->tensor.deleter = nullptr;

  detail::TensorAccess::replace_context(
      tensor, std::move(ctx), detail::TensorAuthority::kLocalWritable);
  return {};
}

expected<void, Error> attach_allocation_impl(Tensor& tensor, TensorShapeView shape,
                                             DLDataType dtype, TensorByteStridesView byte_strides,
                                             BufferAllocation allocation) {
  if (!allocation) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "attach_allocation allocation is empty"});
  }

  const auto memory_buffer = allocation.share_memory_buffer();
  const auto storage_type = memory_buffer->storage_type();
  return attach_tensor_context(tensor,
                               shape,
                               dtype,
                               std::move(byte_strides),
                               storage_type,
                               memory_buffer,
                               /*stream_aware_memory_buffer=*/true);
}

}  // namespace

expected<void, Error> reshape_custom_impl(Tensor& tensor, TensorShapeView shape, DLDataType dtype,
                                          TensorByteStridesView byte_strides,
                                          MemoryStorageType storage_type,
                                          const std::shared_ptr<Allocator>& allocator,
                                          MemoryBuffer::Options memory_options, void* stream) {
  if (allocator == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "reshape_custom allocator is null"});
  }

  auto bytes_per_element = item_size(dtype);
  if (!bytes_per_element) {
    return forward_error(bytes_per_element);
  }

  auto shape_result = validate_shape(shape);
  if (!shape_result) {
    return forward_error(shape_result);
  }

  const expected<uint64_t, Error> maybe_size =
      byte_strides ? strided_size_bytes(shape, byte_strides.value(), bytes_per_element.value())
                   : contiguous_size_bytes(shape, bytes_per_element.value());
  if (!maybe_size) {
    return forward_error(maybe_size);
  }

  auto allocation = BufferAllocation::allocate_with_options(
      allocator, maybe_size.value(), storage_type, memory_options, stream);
  if (!allocation) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR("reshape_custom failed to allocate tensor storage: {}",
                                 runtime_error_message(allocation.error()));
    return forward_error(allocation);
  }

  return attach_allocation_impl(
      tensor, shape, dtype, std::move(byte_strides), std::move(allocation).value());
}

expected<void, Error> reshape_custom(Tensor& tensor, TensorShapeView shape, DLDataType dtype,
                                     TensorByteStridesView byte_strides,
                                     MemoryStorageType storage_type,
                                     const std::shared_ptr<Allocator>& allocator, void* stream) {
  return reshape_custom_impl(tensor,
                             shape,
                             dtype,
                             std::move(byte_strides),
                             storage_type,
                             allocator,
                             MemoryBuffer::Options{},
                             stream);
}

expected<void, Error> reshape_custom(Tensor& tensor, TensorShapeView shape, DLDataType dtype,
                                     TensorByteStridesView byte_strides,
                                     MemoryStorageType storage_type,
                                     const std::shared_ptr<Allocator>& allocator,
                                     MemoryBuffer::Options memory_options, void* stream) {
  return reshape_custom_impl(tensor,
                             shape,
                             dtype,
                             std::move(byte_strides),
                             storage_type,
                             allocator,
                             memory_options,
                             stream);
}

expected<Tensor, Error> make_tensor(TensorShapeView shape, DLDataType dtype,
                                    TensorByteStridesView byte_strides,
                                    MemoryStorageType storage_type,
                                    const std::shared_ptr<Allocator>& allocator, void* stream) {
  Tensor tensor;
  auto result = reshape_custom(
      tensor, shape, dtype, std::move(byte_strides), storage_type, allocator, stream);
  if (!result) {
    return forward_error(result);
  }
  return tensor;
}

expected<Tensor, Error> make_tensor(TensorShapeView shape, DLDataType dtype,
                                    TensorByteStridesView byte_strides,
                                    MemoryStorageType storage_type,
                                    const std::shared_ptr<Allocator>& allocator,
                                    MemoryBuffer::Options memory_options, void* stream) {
  Tensor tensor;
  auto result = reshape_custom(tensor,
                               shape,
                               dtype,
                               std::move(byte_strides),
                               storage_type,
                               allocator,
                               memory_options,
                               stream);
  if (!result) {
    return forward_error(result);
  }
  return tensor;
}

expected<Tensor, Error> make_tensor(TensorShapeView shape, DLDataType dtype,
                                    TensorByteStridesView byte_strides,
                                    MemoryStorageType storage_type,
                                    const std::shared_ptr<Allocator>& allocator,
                                    ::cuda::stream_ref stream) {
  return make_tensor(shape,
                     dtype,
                     std::move(byte_strides),
                     storage_type,
                     allocator,
                     static_cast<void*>(stream.get()));
}

expected<Tensor, Error> make_tensor(TensorShapeView shape, DLDataType dtype,
                                    TensorByteStridesView byte_strides,
                                    MemoryStorageType storage_type,
                                    const std::shared_ptr<Allocator>& allocator,
                                    MemoryBuffer::Options memory_options,
                                    ::cuda::stream_ref stream) {
  return make_tensor(shape,
                     dtype,
                     std::move(byte_strides),
                     storage_type,
                     allocator,
                     memory_options,
                     static_cast<void*>(stream.get()));
}

expected<void, Error> attach_allocation(Tensor& tensor, TensorShapeView shape, DLDataType dtype,
                                        TensorByteStridesView byte_strides,
                                        BufferAllocation allocation) {
  return attach_allocation_impl(
      tensor, shape, dtype, std::move(byte_strides), std::move(allocation));
}

expected<Tensor, Error> make_tensor(TensorShapeView shape, DLDataType dtype,
                                    TensorByteStridesView byte_strides,
                                    BufferAllocation allocation) {
  Tensor tensor;
  auto result =
      attach_allocation(tensor, shape, dtype, std::move(byte_strides), std::move(allocation));
  if (!result) {
    return forward_error(result);
  }
  return tensor;
}

expected<Tensor, Error> make_tensor(TensorShapeView shape, DLDataType dtype,
                                    TensorByteStridesView byte_strides,
                                    MemoryStorageType storage_type, void* pointer,
                                    std::shared_ptr<TensorMemoryReference> memory_reference) {
  return make_tensor(shape,
                     dtype,
                     std::move(byte_strides),
                     storage_type,
                     pointer,
                     std::move(memory_reference),
                     Allocator::kUnknownDeviceId);
}

expected<Tensor, Error> make_tensor(TensorShapeView shape, DLDataType dtype,
                                    TensorByteStridesView byte_strides,
                                    MemoryStorageType storage_type, void* pointer,
                                    std::shared_ptr<TensorMemoryReference> memory_reference,
                                    const int32_t device_id) {
  if (pointer == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "make_tensor pointer is null"});
  }
  if (memory_reference == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "make_tensor memory reference is null"});
  }

  auto bytes_per_element = item_size(dtype);
  if (!bytes_per_element) {
    return forward_error(bytes_per_element);
  }

  auto shape_result = validate_shape(shape);
  if (!shape_result) {
    return forward_error(shape_result);
  }

  auto maybe_device = device_from_storage_type(storage_type, pointer, device_id);
  if (!maybe_device) {
    return forward_error(maybe_device);
  }

  const expected<uint64_t, Error> maybe_size =
      byte_strides ? strided_size_bytes(shape, byte_strides.value(), bytes_per_element.value())
                   : contiguous_size_bytes(shape, bytes_per_element.value());
  if (!maybe_size) {
    return forward_error(maybe_size);
  }

  auto ctx = std::make_shared<DLManagedTensorContext>();
  ctx->memory_reference = std::move(memory_reference);
  ctx->dl_shape.assign(shape.begin(), shape.end());
  if (byte_strides) {
    auto maybe_strides =
        dlpack_strides_from_byte_strides(byte_strides.value(), bytes_per_element.value());
    if (!maybe_strides) {
      return forward_error(maybe_strides);
    }
    ctx->dl_strides = std::move(maybe_strides.value());
  }

  auto& dl_tensor = ctx->tensor.dl_tensor;
  dl_tensor.data = pointer;
  dl_tensor.device = maybe_device.value();
  dl_tensor.ndim = static_cast<int32_t>(ctx->dl_shape.size());
  dl_tensor.dtype = dtype;
  dl_tensor.shape = ctx->dl_shape.data();
  dl_tensor.strides = ctx->dl_strides.empty() ? nullptr : ctx->dl_strides.data();
  dl_tensor.byte_offset = 0;
  ctx->tensor.manager_ctx = nullptr;
  ctx->tensor.deleter = nullptr;

  return Tensor(std::move(ctx));
}

expected<void, Error> wrap_memory(Tensor& tensor, TensorShapeView shape, DLDataType dtype,
                                  TensorByteStridesView byte_strides,
                                  MemoryStorageType storage_type, void* pointer,
                                  MemoryBuffer::release_function_t release_func,
                                  int32_t device_id) {
  if (pointer == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "wrap_memory pointer is null"});
  }

  auto bytes_per_element = item_size(dtype);
  if (!bytes_per_element) {
    return forward_error(bytes_per_element);
  }

  auto shape_result = validate_shape(shape);
  if (!shape_result) {
    return forward_error(shape_result);
  }

  expected<uint64_t, Error> maybe_size =
      byte_strides ? strided_size_bytes(shape, byte_strides.value(), bytes_per_element.value())
                   : contiguous_size_bytes(shape, bytes_per_element.value());
  if (!maybe_size) {
    return forward_error(maybe_size);
  }

  auto memory_buffer = std::make_shared<MemoryBuffer>();
  const auto wrap_result = memory_buffer->wrap_memory(
      pointer, maybe_size.value(), storage_type, std::move(release_func), device_id);
  if (!wrap_result) {
    return forward_error(wrap_result);
  }

  return attach_tensor_context(tensor,
                               shape,
                               dtype,
                               std::move(byte_strides),
                               storage_type,
                               memory_buffer,
                               /*stream_aware_memory_buffer=*/false);
}

expected<Tensor, Error> make_tensor(TensorShapeView shape, DLDataType dtype,
                                    TensorByteStridesView byte_strides,
                                    MemoryStorageType storage_type, void* pointer,
                                    MemoryBuffer::release_function_t release_func) {
  Tensor tensor;
  auto result = wrap_memory(tensor,
                            shape,
                            dtype,
                            std::move(byte_strides),
                            storage_type,
                            pointer,
                            std::move(release_func));
  if (!result) {
    return forward_error(result);
  }
  return tensor;
}

}  // namespace holoscan
