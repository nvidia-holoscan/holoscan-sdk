/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <holoscan/core/domain/dlpack_utils.hpp>
#include <holoscan/memory/allocator.hpp>
#include <holoscan/memory/memory_storage_type.hpp>

#include "../holomq/buffer_pool/stream_lifetime_access.hpp"
#include "../holomq/buffer_pool/sync/stream_release.hpp"
#include "channel_common.hpp"
#include "core/domain/tensor_access.hpp"
#include "holomq_tensor.hpp"
#include "scoped_cuda_device.hpp"
#include "tensor_stream_readiness.hpp"

namespace holoscan::channel_internal {

namespace {

[[nodiscard]] Error make_holomq_tensor_error(ErrorCode code, const char* message) {
  return Error{code, message};
}

[[nodiscard]] expected<MemoryStorageType, Error> storage_type_from_holomq_memory_type(
    const holomq::BufferMemoryType memory_type) {
  switch (memory_type) {
    case holomq::BufferMemoryType::Device:
      return MemoryStorageType::kDevice;
    case holomq::BufferMemoryType::HostPageable:
      return MemoryStorageType::kSystem;
    case holomq::BufferMemoryType::HostPinned:
      return MemoryStorageType::kHost;
    case holomq::BufferMemoryType::None:
    default:
      return make_unexpected(make_holomq_tensor_error(
          ErrorCode::kInvalidArgument, "HoloMQ buffer has no supported memory type"));
  }
}

[[nodiscard]] expected<HoloMQTensorDescriptorMemoryKind, Error> memory_kind_from_holomq_handle(
    const holomq::BufferHandle& handle) {
  const holomq::BufferMemoryType memory_type = handle.memory_type();
  switch (memory_type) {
    case holomq::BufferMemoryType::Device:
      return HoloMQTensorDescriptorMemoryKind::kDevice;
    case holomq::BufferMemoryType::HostPageable:
      return HoloMQTensorDescriptorMemoryKind::kHostPageable;
    case holomq::BufferMemoryType::HostPinned:
      return HoloMQTensorDescriptorMemoryKind::kHostPinned;
    case holomq::BufferMemoryType::None:
    default:
      return make_unexpected(make_holomq_tensor_error(
          ErrorCode::kInvalidArgument, "HoloMQ Tensor descriptor buffer handle is empty"));
  }
}

[[nodiscard]] expected<void, Error> validate_holomq_buffer_view_matches_descriptor(
    const HoloMQTensorDescriptor& descriptor, const holomq::BufferMemoryType memory_type,
    const std::optional<std::int32_t> cuda_device_ordinal) {
  if (memory_type != descriptor.buffer_handle.memory_type()) {
    return make_unexpected(make_holomq_tensor_error(
        ErrorCode::kInvalidArgument, "HoloMQ buffer memory type does not match descriptor"));
  }
  if (descriptor.memory_kind != HoloMQTensorDescriptorMemoryKind::kDevice) {
    return {};
  }
  if (!cuda_device_ordinal.has_value()) {
    return make_unexpected(make_holomq_tensor_error(
        ErrorCode::kInvalidArgument, "HoloMQ device buffer view has no CUDA device ordinal"));
  }
  if (descriptor.cuda_device_ordinal >= 0 &&
      descriptor.cuda_device_ordinal != cuda_device_ordinal.value()) {
    return make_unexpected(
        make_holomq_tensor_error(ErrorCode::kInvalidArgument,
                                 "HoloMQ descriptor CUDA device ordinal does not match buffer"));
  }
  return {};
}

[[nodiscard]] expected<MemoryStorageType, Error> storage_type_from_descriptor_memory_kind(
    const HoloMQTensorDescriptorMemoryKind memory_kind) {
  switch (memory_kind) {
    case HoloMQTensorDescriptorMemoryKind::kDevice:
      return MemoryStorageType::kDevice;
    case HoloMQTensorDescriptorMemoryKind::kHostPageable:
      return MemoryStorageType::kSystem;
    case HoloMQTensorDescriptorMemoryKind::kHostPinned:
      return MemoryStorageType::kHost;
    case HoloMQTensorDescriptorMemoryKind::kNone:
    default:
      return make_unexpected(make_holomq_tensor_error(
          ErrorCode::kInvalidArgument, "HoloMQ Tensor descriptor has no supported memory kind"));
  }
}

[[nodiscard]] constexpr bool descriptor_has_flag(const HoloMQTensorDescriptor& descriptor,
                                                 const HoloMQTensorDescriptorFlags flag) noexcept {
  return (descriptor.flags & static_cast<std::uint32_t>(flag)) != 0U;
}

void descriptor_set_flag(HoloMQTensorDescriptor& descriptor,
                         const HoloMQTensorDescriptorFlags flag) noexcept {
  descriptor.flags |= static_cast<std::uint32_t>(flag);
}

[[nodiscard]] expected<void, Error> validate_holomq_tensor_descriptor(
    const HoloMQTensorDescriptor& descriptor) {
  if (!descriptor.has_supported_abi()) {
    return make_unexpected(make_holomq_tensor_error(
        ErrorCode::kInvalidArgument, "HoloMQ Tensor descriptor ABI is not supported"));
  }
  if (descriptor.rank > kHoloMQTensorDescriptorMaxRank) {
    return make_unexpected(make_holomq_tensor_error(
        ErrorCode::kInvalidArgument, "HoloMQ Tensor descriptor rank exceeds fixed capacity"));
  }
  if (!descriptor.buffer_handle) {
    return make_unexpected(make_holomq_tensor_error(
        ErrorCode::kInvalidArgument, "HoloMQ Tensor descriptor buffer handle is empty"));
  }
  auto handle_memory_kind = memory_kind_from_holomq_handle(descriptor.buffer_handle);
  if (!handle_memory_kind) {
    return make_unexpected(std::move(handle_memory_kind).error());
  }
  if (handle_memory_kind.value() != descriptor.memory_kind) {
    return make_unexpected(make_holomq_tensor_error(
        ErrorCode::kInvalidArgument,
        "HoloMQ Tensor descriptor memory kind does not match buffer handle"));
  }
  if (auto storage_type = storage_type_from_descriptor_memory_kind(descriptor.memory_kind);
      !storage_type) {
    return make_unexpected(std::move(storage_type).error());
  }
  if (descriptor.dtype.bits == 0U || descriptor.dtype.lanes == 0U) {
    return make_unexpected(make_holomq_tensor_error(
        ErrorCode::kInvalidArgument, "HoloMQ Tensor descriptor dtype has zero-sized elements"));
  }
  return {};
}

[[nodiscard]] TensorShapeView descriptor_shape_view(
    const HoloMQTensorDescriptor& descriptor) noexcept {
  return TensorShapeView{descriptor.shape.data(), descriptor.rank};
}

[[nodiscard]] TensorByteStridesView descriptor_byte_strides_view(
    const HoloMQTensorDescriptor& descriptor) noexcept {
  if (!descriptor_has_flag(descriptor, HoloMQTensorDescriptorFlags::kHasByteStrides)) {
    return std::nullopt;
  }
  return TensorByteStridesView{
      TensorByteStridesSpan{descriptor.byte_strides.data(), descriptor.rank}};
}

[[nodiscard]] expected<std::uint64_t, Error> tensor_storage_extent_bytes(
    const Tensor& tensor, const TensorByteStridesView byte_strides) {
  if (!byte_strides.has_value()) {
    return static_cast<std::uint64_t>(tensor.nbytes());
  }

  const TensorShapeView shape = tensor.shape_span();
  if (byte_strides.size() != shape.size()) {
    return make_unexpected(make_holomq_tensor_error(
        ErrorCode::kInvalidArgument, "Tensor stride rank does not match shape rank"));
  }
  if (shape.empty()) {
    return static_cast<std::uint64_t>(tensor.itemsize());
  }

  std::uint64_t max_offset = 0U;
  for (std::size_t index = 0U; index < shape.size(); ++index) {
    if (shape[index] == 0) {
      return 0U;
    }
    if (shape[index] < 0 || byte_strides[index] < 0) {
      return make_unexpected(make_holomq_tensor_error(
          ErrorCode::kInvalidArgument, "Tensor shape and byte strides must be non-negative"));
    }
    const auto extent = static_cast<std::uint64_t>(shape[index] - 1);
    const auto stride = static_cast<std::uint64_t>(byte_strides[index]);
    if (stride != 0U && extent > std::numeric_limits<std::uint64_t>::max() / stride) {
      return make_unexpected(make_holomq_tensor_error(ErrorCode::kInvalidArgument,
                                                      "Tensor stride extent overflows uint64_t"));
    }
    const std::uint64_t offset = extent * stride;
    if (max_offset > std::numeric_limits<std::uint64_t>::max() - offset) {
      return make_unexpected(make_holomq_tensor_error(ErrorCode::kInvalidArgument,
                                                      "Tensor storage extent overflows uint64_t"));
    }
    max_offset += offset;
  }

  const auto item_size = static_cast<std::uint64_t>(tensor.itemsize());
  if (max_offset > std::numeric_limits<std::uint64_t>::max() - item_size) {
    return make_unexpected(make_holomq_tensor_error(ErrorCode::kInvalidArgument,
                                                    "Tensor storage extent overflows uint64_t"));
  }
  return max_offset + item_size;
}

class HostReadinessStream {
 public:
  explicit HostReadinessStream(const std::optional<std::int32_t> cuda_device_ordinal) noexcept
      : cuda_device_ordinal_(cuda_device_ordinal) {}
  HostReadinessStream(const HostReadinessStream&) = delete;
  HostReadinessStream& operator=(const HostReadinessStream&) = delete;
  ~HostReadinessStream() = default;

  [[nodiscard]] expected<cudaStream_t, Error> get_or_create() noexcept {
    const ScopedCUDADevice selected{cuda_device_ordinal_};
    if (!selected.active()) {
      return make_unexpected_nothrow<cudaStream_t>(ErrorCode::kFailure,
                                                   "host-readiness device selection failed");
    }
    if (stream_ == nullptr) {
      // A host-only importer may synchronize on a fresh thread with no current
      // context. Initialize its runtime-selected device before using the driver
      // endpoint factory, while preserving an existing caller context.
      CUcontext context{};
      if (cuCtxGetCurrent(&context) != CUDA_SUCCESS) {
        return make_unexpected_nothrow<cudaStream_t>(ErrorCode::kFailure,
                                                     "host-readiness context query failed");
      }
      if (context == nullptr) {
        int device{};
        if (cudaGetDevice(&device) != cudaSuccess || cudaSetDevice(device) != cudaSuccess) {
          return make_unexpected_nothrow<cudaStream_t>(
              ErrorCode::kFailure, "host-readiness context initialization failed");
        }
      }
      stream_ = holomq::EndpointStream::create();
      if (stream_ == nullptr) {
        return make_unexpected_nothrow<cudaStream_t>(ErrorCode::kFailure,
                                                     "host-readiness stream creation failed");
      }
    }
    return stream_->get();
  }
  [[nodiscard]] const std::shared_ptr<const holomq::cuda::StreamLifetime>& lifetime()
      const noexcept {
    return stream_->lifetime();
  }

 private:
  std::optional<std::int32_t> cuda_device_ordinal_;
  std::unique_ptr<holomq::EndpointStream> stream_;
};

class HoloMQBufferTensorMemoryReference final : public HoloMQExportableTensorMemoryReference,
                                                public detail::TensorStreamLifetimeConsumer {
 public:
  explicit HoloMQBufferTensorMemoryReference(holomq::Buffer buffer) : buffer_(std::move(buffer)) {}

  [[nodiscard]] expected<holomq::BufferHandle, Error> export_handle() noexcept override {
    if (!buffer_) {
      return make_unexpected_nothrow<holomq::BufferHandle>(ErrorCode::kInvalidArgument,
                                                           "HoloMQ buffer lease is empty");
    }
    cudaStream_t readiness_stream = readiness_.stream();
    auto handle = readiness_stream == nullptr
                      ? buffer_.export_handle_unsynchronized()
                      : holomq::BufferStreamAccess::export_handle(
                            buffer_, readiness_stream, readiness_.lifetime());
    if (!handle) {
      return make_unexpected(from_holomq_error(std::move(handle).error(),
                                               "failed to export HoloMQ Tensor buffer handle"));
    }
    return std::move(handle).value();
  }

  bool set_deallocation_stream(cudaStream_t stream) override {
    return register_consumer_stream(stream, detail::TensorConsumerScope::lifetime(stream));
  }
  bool register_consumer_stream(
      cudaStream_t stream,
      const std::shared_ptr<const holomq::cuda::StreamLifetime>& lifetime) noexcept override {
    return static_cast<bool>(buffer_) &&
           (stream == nullptr ||
            holomq::BufferStreamAccess::register_consumer(buffer_, stream, lifetime));
  }

  [[nodiscard]] bool tracks_deallocation_stream() const noexcept override {
    return static_cast<bool>(buffer_);
  }

  bool set_readiness_stream(cudaStream_t stream) override {
    if (!buffer_) {
      return false;
    }
    readiness_.set(stream, detail::TensorConsumerScope::lifetime(stream));
    if (stream == nullptr) {
      return true;
    }
    return holomq::BufferStreamAccess::export_handle(
               buffer_, stream, detail::TensorConsumerScope::lifetime(stream))
        .has_value();
  }

  bool has_readiness_stream_contract() const noexcept override {
    return static_cast<bool>(buffer_);
  }

  [[nodiscard]] cudaStream_t readiness_stream() const noexcept override {
    return readiness_.stream();
  }

  [[nodiscard]] expected<void, Error> synchronize_for_consumer(
      cudaStream_t consumer_stream) override {
    return readiness_.consumer(consumer_stream);
  }

  [[nodiscard]] expected<void, Error> synchronize_for_host() override { return readiness_.host(); }

 private:
  holomq::Buffer buffer_;
  channel_internal::TensorStreamReadiness readiness_;
};

class HoloMQConstBufferTensorMemoryReference final : public HoloMQExportableTensorMemoryReference,
                                                     public detail::TensorStreamLifetimeConsumer {
 public:
  explicit HoloMQConstBufferTensorMemoryReference(holomq::ConstBuffer buffer)
      : buffer_(std::move(buffer)), host_wait_stream_(buffer_.cuda_device_ordinal()) {}

  [[nodiscard]] expected<holomq::BufferHandle, Error> export_handle() noexcept override {
    if (!buffer_) {
      return make_unexpected_nothrow<holomq::BufferHandle>(ErrorCode::kInvalidArgument,
                                                           "HoloMQ imported buffer lease is empty");
    }
    auto handle = buffer_.export_handle();
    if (!handle) {
      return make_unexpected(from_holomq_error(
          std::move(handle).error(), "failed to re-export HoloMQ imported Tensor buffer handle"));
    }
    return std::move(handle).value();
  }

  bool set_deallocation_stream(cudaStream_t stream) override {
    return register_consumer_stream(stream, detail::TensorConsumerScope::lifetime(stream));
  }
  bool register_consumer_stream(
      cudaStream_t stream,
      const std::shared_ptr<const holomq::cuda::StreamLifetime>& lifetime) noexcept override {
    return static_cast<bool>(buffer_) &&
           (stream == nullptr ||
            holomq::BufferStreamAccess::register_consumer(buffer_, stream, lifetime));
  }

  [[nodiscard]] bool tracks_deallocation_stream() const noexcept override {
    return static_cast<bool>(buffer_);
  }

  bool set_readiness_stream(cudaStream_t stream) override {
    if (!buffer_) {
      return false;
    }
    readiness_.set(stream, detail::TensorConsumerScope::lifetime(stream));
    if (stream == nullptr) {
      return true;
    }
    return holomq::BufferStreamAccess::wait(
        buffer_, stream, detail::TensorConsumerScope::lifetime(stream));
  }

  bool has_readiness_stream_contract() const noexcept override {
    return static_cast<bool>(buffer_);
  }

  [[nodiscard]] cudaStream_t readiness_stream() const noexcept override {
    return readiness_.stream();
  }

  [[nodiscard]] expected<void, Error> synchronize_for_consumer(
      cudaStream_t consumer_stream) override {
    if (consumer_stream == nullptr) {
      return synchronize_for_host();
    }
    if (!holomq::BufferStreamAccess::wait(
            buffer_, consumer_stream, detail::TensorConsumerScope::lifetime(consumer_stream))) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    readiness_.set(consumer_stream, detail::TensorConsumerScope::lifetime(consumer_stream));
    return {};
  }

  [[nodiscard]] expected<void, Error> synchronize_for_host() override {
    const std::lock_guard lock{host_mutex_};
    if (!static_cast<bool>(buffer_)) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (!(buffer_).has_producer_readiness_fence()) {
      return {};
    }
    const ScopedCUDADevice selected{(buffer_).cuda_device_ordinal()};
    if (!selected.active()) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto stream = host_wait_stream_.get_or_create();
    if (!stream) {
      return make_unexpected(std::move(stream).error());
    }
    if (!holomq::BufferStreamAccess::wait(buffer_, stream.value(), host_wait_stream_.lifetime())) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    return synchronize_cuda_stream_for_host(stream.value());
  }

 private:
  holomq::ConstBuffer buffer_;
  HostReadinessStream host_wait_stream_;
  TensorStreamReadiness readiness_;
  std::mutex host_mutex_;
};

[[nodiscard]] expected<Tensor, Error> wrap_holomq_tensor(
    TensorShapeView shape, DLDataType dtype, TensorByteStridesView byte_strides,
    MemoryStorageType storage_type, void* pointer, std::size_t buffer_size,
    std::shared_ptr<TensorMemoryReference> memory_reference, const bool read_only,
    const std::int32_t device_id) {
  auto tensor = make_tensor(
      shape, dtype, byte_strides, storage_type, pointer, std::move(memory_reference), device_id);
  if (!tensor) {
    return make_unexpected(std::move(tensor).error());
  }
  auto storage_extent = tensor_storage_extent_bytes(tensor.value(), byte_strides);
  if (!storage_extent) {
    return make_unexpected(std::move(storage_extent).error());
  }
  if (storage_extent.value() > static_cast<std::uint64_t>(buffer_size)) {
    return make_unexpected(make_holomq_tensor_error(
        ErrorCode::kInvalidArgument, "Tensor storage extent exceeds HoloMQ buffer lease size"));
  }
  if (read_only) {
    detail::TensorAccess::narrow_read_only(*tensor);
  }
  return std::move(tensor).value();
}

}  // namespace

expected<HoloMQTensorDescriptor, Error> make_holomq_tensor_descriptor(
    TensorShapeView shape, DLDataType dtype, TensorByteStridesView byte_strides,
    const std::uint64_t byte_size, const holomq::BufferHandle& buffer_handle, const bool read_only,
    const std::int32_t cuda_device_ordinal) {
  if (!buffer_handle) {
    return make_unexpected(make_holomq_tensor_error(
        ErrorCode::kInvalidArgument, "HoloMQ Tensor descriptor buffer handle is empty"));
  }
  if (shape.size() > kHoloMQTensorDescriptorMaxRank) {
    return make_unexpected(make_holomq_tensor_error(
        ErrorCode::kInvalidArgument, "Tensor rank exceeds HoloMQ Tensor descriptor capacity"));
  }
  if (byte_strides.has_value() && byte_strides.size() != shape.size()) {
    return make_unexpected(make_holomq_tensor_error(ErrorCode::kInvalidArgument,
                                                    "Tensor byte strides must match Tensor rank"));
  }

  auto memory_kind = memory_kind_from_holomq_handle(buffer_handle);
  if (!memory_kind) {
    return make_unexpected(std::move(memory_kind).error());
  }

  HoloMQTensorDescriptor descriptor{};
  descriptor.rank = static_cast<std::uint32_t>(shape.size());
  descriptor.dtype = dtype;
  descriptor.memory_kind = memory_kind.value();
  descriptor.byte_size = byte_size;
  descriptor.cuda_device_ordinal = cuda_device_ordinal;
  descriptor.buffer_handle = buffer_handle;
  if (read_only) {
    descriptor_set_flag(descriptor, HoloMQTensorDescriptorFlags::kReadOnly);
  }

  std::copy(shape.begin(), shape.end(), descriptor.shape.begin());

  if (byte_strides.has_value()) {
    descriptor_set_flag(descriptor, HoloMQTensorDescriptorFlags::kHasByteStrides);
    std::copy(byte_strides.begin(), byte_strides.end(), descriptor.byte_strides.begin());
  }

  return descriptor;
}

expected<HoloMQTensorDescriptor, Error> make_holomq_tensor_descriptor(
    const Tensor& tensor, const holomq::BufferHandle& buffer_handle) {
  if (!detail::TensorAccess::context(tensor)) {
    return make_unexpected(
        make_holomq_tensor_error(ErrorCode::kInvalidArgument, "Tensor is empty"));
  }
  if (tensor.byte_offset() != 0U) {
    return make_unexpected(make_holomq_tensor_error(
        ErrorCode::kInvalidArgument, "HoloMQ Tensor descriptor does not support byte_offset"));
  }

  std::array<std::int64_t, kHoloMQTensorDescriptorMaxRank> byte_strides{};
  TensorByteStridesView byte_strides_view{std::nullopt};
  const auto dlpack_strides = tensor.dlpack_strides_span();
  if (dlpack_strides.size() > kHoloMQTensorDescriptorMaxRank) {
    return make_unexpected(make_holomq_tensor_error(
        ErrorCode::kInvalidArgument, "Tensor rank exceeds HoloMQ Tensor descriptor capacity"));
  }
  if (!dlpack_strides.empty()) {
    const std::int64_t item_size = static_cast<std::int64_t>(tensor.itemsize());
    for (std::size_t i = 0; i < dlpack_strides.size(); ++i) {
      byte_strides[i] = dlpack_strides[i] * item_size;
    }
    byte_strides_view =
        TensorByteStridesSpan{byte_strides.data(), static_cast<std::size_t>(dlpack_strides.size())};
  }

  return make_holomq_tensor_descriptor(
      tensor.shape_span(),
      tensor.dtype(),
      byte_strides_view,
      static_cast<std::uint64_t>(tensor.nbytes()),
      buffer_handle,
      tensor.is_read_only(),
      tensor.device().device_type == kDLCUDA ? tensor.device().device_id : -1);
}

HoloMQExportableTensorMemoryReference* holomq_exportable_tensor_memory_reference(
    const Tensor& tensor) noexcept {
  const auto& context = detail::TensorAccess::context(tensor);
  if (context == nullptr || context->memory_reference == nullptr) {
    return nullptr;
  }
  return dynamic_cast<HoloMQExportableTensorMemoryReference*>(context->memory_reference.get());
}

expected<HoloMQTensorDescriptor, Error> export_holomq_tensor_descriptor(
    const Tensor& tensor, HoloMQExportableTensorMemoryReference& memory) {
  auto handle = memory.export_handle();
  if (!handle) {
    return make_unexpected(std::move(handle).error());
  }
  return make_holomq_tensor_descriptor(tensor, handle.value());
}

expected<HoloMQTensorDescriptor, Error> export_holomq_tensor_descriptor(const Tensor& tensor) {
  auto* const memory = holomq_exportable_tensor_memory_reference(tensor);
  if (memory != nullptr) {
    return export_holomq_tensor_descriptor(tensor, *memory);
  }

  const auto& context = detail::TensorAccess::context(tensor);
  if (context == nullptr || context->memory_reference == nullptr) {
    return make_unexpected(
        make_holomq_tensor_error(ErrorCode::kInvalidArgument, "Tensor is empty"));
  }

  return make_unexpected(make_holomq_tensor_error(
      ErrorCode::kInvalidArgument, "Tensor is not backed by HoloMQ-exportable storage"));
}

PayloadDescriptor payload_descriptor_from_holomq_tensor_descriptor(
    const HoloMQTensorDescriptor& descriptor) noexcept {
  return PayloadDescriptor{.payload_kind = ChannelPayloadKind::kBufferHandle,
                           .flags = descriptor.flags,
                           .byte_size = descriptor.byte_size,
                           .buffer = make_holomq_payload_reference(descriptor.buffer_handle)};
}

expected<Tensor, Error> make_tensor_from_holomq_buffer(TensorShapeView shape, DLDataType dtype,
                                                       TensorByteStridesView byte_strides,
                                                       holomq::Buffer buffer) {
  if (!buffer) {
    return make_unexpected(
        make_holomq_tensor_error(ErrorCode::kInvalidArgument, "HoloMQ buffer lease is empty"));
  }
  auto storage_type = storage_type_from_holomq_memory_type(buffer.memory_type());
  if (!storage_type) {
    return make_unexpected(std::move(storage_type).error());
  }

  void* pointer = buffer.ptr();
  const std::size_t buffer_size = buffer.size_bytes();
  const std::int32_t device_id = buffer.cuda_device_ordinal().value_or(Allocator::kUnknownDeviceId);
  auto memory_reference = std::make_shared<HoloMQBufferTensorMemoryReference>(std::move(buffer));
  return wrap_holomq_tensor(shape,
                            dtype,
                            std::move(byte_strides),
                            storage_type.value(),
                            pointer,
                            buffer_size,
                            std::move(memory_reference),
                            /*read_only=*/false,
                            device_id);
}

expected<Tensor, Error> make_tensor_from_holomq_buffer(const HoloMQTensorDescriptor& descriptor,
                                                       holomq::Buffer buffer) {
  if (auto valid = validate_holomq_tensor_descriptor(descriptor); !valid) {
    return make_unexpected(std::move(valid).error());
  }
  if (auto valid_buffer = validate_holomq_buffer_view_matches_descriptor(
          descriptor, buffer.memory_type(), buffer.cuda_device_ordinal());
      !valid_buffer) {
    return make_unexpected(std::move(valid_buffer).error());
  }

  auto tensor = make_tensor_from_holomq_buffer(descriptor_shape_view(descriptor),
                                               descriptor.dtype,
                                               descriptor_byte_strides_view(descriptor),
                                               std::move(buffer));
  if (!tensor) {
    return make_unexpected(std::move(tensor).error());
  }
  if (static_cast<std::uint64_t>(tensor->nbytes()) != descriptor.byte_size) {
    return make_unexpected(make_holomq_tensor_error(ErrorCode::kInvalidArgument,
                                                    "HoloMQ Tensor descriptor byte size mismatch"));
  }
  if (descriptor_has_flag(descriptor, HoloMQTensorDescriptorFlags::kReadOnly)) {
    detail::TensorAccess::narrow_read_only(*tensor);
  }
  return std::move(tensor).value();
}

expected<Tensor, Error> make_tensor_from_holomq_const_buffer(TensorShapeView shape,
                                                             DLDataType dtype,
                                                             TensorByteStridesView byte_strides,
                                                             holomq::ConstBuffer buffer) {
  if (!buffer) {
    return make_unexpected(make_holomq_tensor_error(ErrorCode::kInvalidArgument,
                                                    "HoloMQ imported buffer lease is empty"));
  }
  auto storage_type = storage_type_from_holomq_memory_type(buffer.memory_type());
  if (!storage_type) {
    return make_unexpected(std::move(storage_type).error());
  }

  auto* pointer = const_cast<void*>(buffer.ptr());
  const std::size_t buffer_size = buffer.size_bytes();
  const std::int32_t device_id = buffer.cuda_device_ordinal().value_or(Allocator::kUnknownDeviceId);
  auto memory_reference =
      std::make_shared<HoloMQConstBufferTensorMemoryReference>(std::move(buffer));
  return wrap_holomq_tensor(shape,
                            dtype,
                            std::move(byte_strides),
                            storage_type.value(),
                            pointer,
                            buffer_size,
                            std::move(memory_reference),
                            /*read_only=*/true,
                            device_id);
}

expected<Tensor, Error> make_tensor_from_holomq_const_buffer(
    const HoloMQTensorDescriptor& descriptor, holomq::ConstBuffer buffer) {
  if (auto valid = validate_holomq_tensor_descriptor(descriptor); !valid) {
    return make_unexpected(std::move(valid).error());
  }
  if (auto valid_buffer = validate_holomq_buffer_view_matches_descriptor(
          descriptor, buffer.memory_type(), buffer.cuda_device_ordinal());
      !valid_buffer) {
    return make_unexpected(std::move(valid_buffer).error());
  }

  auto tensor = make_tensor_from_holomq_const_buffer(descriptor_shape_view(descriptor),
                                                     descriptor.dtype,
                                                     descriptor_byte_strides_view(descriptor),
                                                     std::move(buffer));
  if (!tensor) {
    return make_unexpected(std::move(tensor).error());
  }
  if (static_cast<std::uint64_t>(tensor->nbytes()) != descriptor.byte_size) {
    return make_unexpected(make_holomq_tensor_error(ErrorCode::kInvalidArgument,
                                                    "HoloMQ Tensor descriptor byte size mismatch"));
  }
  return std::move(tensor).value();
}

}  // namespace holoscan::channel_internal
