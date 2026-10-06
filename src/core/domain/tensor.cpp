/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>

#include "core/domain/tensor_access.hpp"
#include "core/domain/tensor_layout_validation.hpp"

#include <holoscan/core/error_helpers.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>

namespace holoscan {

namespace detail {

class TensorAccessState {
 public:
  explicit TensorAccessState(TensorAuthority authority) noexcept
      : read_only(authority == TensorAuthority::kReadOnly) {}

  static constexpr std::uint32_t kWriterClaim = std::uint32_t{1} << 31U;
  static constexpr std::uint32_t kReaderMask = kWriterClaim - 1U;

  std::atomic<bool> read_only{false};
  // One atomic word makes reader acquisition and the dedicated mutable DLPack handoff mutually
  // exclusive. Separate reader/writer atomics would leave a check-then-publish race in which both
  // authorities could be granted over the same storage.
  std::atomic<std::uint32_t> access_claims{0};
};

}  // namespace detail

namespace {

bool is_cuda_tensor_device(DLDeviceType device_type) {
  return device_type == kDLCUDA || device_type == kDLCUDAManaged;
}

bool is_host_tensor_device(DLDeviceType device_type) {
  return device_type == kDLCPU || device_type == kDLCUDAHost;
}

using error_detail::make_cuda_error;

class ScopedCudaDevice final {
 public:
  explicit ScopedCudaDevice(std::optional<std::int32_t> requested) noexcept {
    if (!requested || cudaGetDevice(&previous_) != cudaSuccess) {
      return;
    }
    if (previous_ == *requested || cudaSetDevice(*requested) == cudaSuccess) {
      active_ = true;
    }
  }

  ScopedCudaDevice(const ScopedCudaDevice&) = delete;
  ScopedCudaDevice& operator=(const ScopedCudaDevice&) = delete;

  ~ScopedCudaDevice() {
    if (active_) {
      (void)cudaSetDevice(previous_);
    }
  }

  [[nodiscard]] bool active() const noexcept { return active_; }

 private:
  int previous_{};
  bool active_{};
};

[[nodiscard]] bool is_cuda_copy_device(DLDeviceType device_type) noexcept {
  return device_type == kDLCUDA || device_type == kDLCUDAManaged;
}

[[nodiscard]] expected<MemoryStorageType, Error> copy_storage_type(DLDevice target) {
  switch (target.device_type) {
    case kDLCPU:
      if (target.device_id != 0) {
        break;
      }
      return MemoryStorageType::kSystem;
    case kDLCUDA:
      if (target.device_id >= 0) {
        return MemoryStorageType::kDevice;
      }
      break;
    case kDLCUDAHost:
      if (target.device_id == 0) {
        return MemoryStorageType::kHost;
      }
      break;
    case kDLCUDAManaged:
      if (target.device_id == 0) {
        return MemoryStorageType::kCudaManaged;
      }
      break;
    default:
      break;
  }
  return make_unexpected(
      Error{ErrorCode::kInvalidArgument, "Tensor copy target device is unsupported"});
}

// Export context used by to_dlpack(): owns an unversioned DLManagedTensor plus the memory
// reference and the shape/stride arrays the tensor points at.
struct UnversionedDLPackExportContext {
  DLManagedTensor tensor{};
  std::shared_ptr<TensorMemoryReference> memory_reference;
  std::shared_ptr<detail::TensorAccessState> access_state;
  std::vector<int64_t> dl_shape;
  std::vector<int64_t> dl_strides;

  ~UnversionedDLPackExportContext() {
    if (access_state != nullptr) {
      access_state->access_claims.fetch_and(~detail::TensorAccessState::kWriterClaim,
                                            std::memory_order_release);
    }
    if (memory_reference != nullptr) {
      memory_reference->release_tensor_handle();
    }
  }
};

struct VersionedDLPackExportContext {
  DLManagedTensorVersioned tensor{};
  std::shared_ptr<TensorMemoryReference> memory_reference;
  std::shared_ptr<detail::TensorAccessState> access_state;
  bool writer_claim = false;
  std::vector<int64_t> dl_shape;
  std::vector<int64_t> dl_strides;

  ~VersionedDLPackExportContext() {
    if (access_state != nullptr) {
      if (writer_claim) {
        access_state->access_claims.fetch_and(~detail::TensorAccessState::kWriterClaim,
                                              std::memory_order_release);
      } else {
        access_state->access_claims.fetch_sub(1U, std::memory_order_acq_rel);
      }
    }
    if (memory_reference != nullptr) {
      memory_reference->release_tensor_handle();
    }
  }
};

}  // namespace

DLManagedTensorContext::DLManagedTensorContext()
    : access_state_(
          std::make_shared<detail::TensorAccessState>(detail::TensorAuthority::kLocalWritable)) {}

DLManagedMemoryBuffer::DLManagedMemoryBuffer(DLManagedTensor* self) : self_(self) {}

DLManagedMemoryBuffer::~DLManagedMemoryBuffer() {
  if (self_ != nullptr && self_->deleter != nullptr) {
    self_->deleter(self_);
  }
}

DLManagedMemoryBufferVersioned::DLManagedMemoryBufferVersioned(DLManagedTensorVersioned* self)
    : self_(self) {}

DLManagedMemoryBufferVersioned::~DLManagedMemoryBufferVersioned() {
  if (self_ != nullptr && self_->deleter != nullptr) {
    self_->deleter(self_);
  }
}

expected<Tensor, Error> make_tensor_from_dlpack(DLManagedTensor* dl_managed_tensor_ptr) {
  if (dl_managed_tensor_ptr == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "DLManagedTensor pointer is null"});
  }
  auto context = std::make_shared<DLManagedTensorContext>();
  context->memory_reference = std::make_shared<DLManagedMemoryBuffer>(dl_managed_tensor_ptr);
  // The memory reference owns invoking the source deleter; copy only the (owned) metadata. The
  // unversioned metadata starts at the current version; TensorAccess then records the independent
  // permanent read-only authority required for imported storage.
  copy_dl_tensor_with_owned_metadata(*context, dl_managed_tensor_ptr->dl_tensor);
  return detail::TensorAccess::from_context(std::move(context), detail::TensorAuthority::kReadOnly);
}

expected<Tensor, Error> make_tensor_from_dlpack_versioned(
    DLManagedTensorVersioned* dl_managed_tensor_ver_ptr) {
  if (dl_managed_tensor_ver_ptr == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "DLManagedTensorVersioned pointer is null"});
  }
  // A consumer must reject a newer major version without taking ownership: the versioned struct
  // layout may have changed, so dereferencing its fields would be undefined.
  if (dl_managed_tensor_ver_ptr->version.major > HOLOSCAN_DLPACK_IMPL_VERSION_MAJOR) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              fmt::format("Unsupported DLPack major version {} (this build supports up to {})",
                          dl_managed_tensor_ver_ptr->version.major,
                          HOLOSCAN_DLPACK_IMPL_VERSION_MAJOR)});
  }
  auto context = std::make_shared<DLManagedTensorContext>();
  context->memory_reference =
      std::make_shared<DLManagedMemoryBufferVersioned>(dl_managed_tensor_ver_ptr);
  // Preserve the source version and flags; the memory reference owns invoking the source deleter.
  copy_dl_tensor_with_owned_metadata(*context, *dl_managed_tensor_ver_ptr);
  return detail::TensorAccess::from_context(std::move(context), detail::TensorAuthority::kReadOnly);
}

Tensor::Tensor(std::shared_ptr<DLManagedTensorContext> ctx) : dl_ctx_(std::move(ctx)) {
  if (dl_ctx_ != nullptr) {
    if ((dl_ctx_->tensor.flags & DLPACK_FLAG_BITMASK_READ_ONLY) != 0U) {
      dl_ctx_->access_state_->read_only.store(true, std::memory_order_release);
    }
    retain_handle();
  }
}

Tensor::Tensor(const Tensor& other) noexcept : dl_ctx_(other.dl_ctx_) {
  retain_handle();
}

Tensor& Tensor::operator=(const Tensor& other) noexcept {
  if (this == &other) {
    return *this;
  }
  auto next_context = other.dl_ctx_;
  if (next_context != nullptr && next_context->memory_reference != nullptr) {
    next_context->memory_reference->retain_tensor_handle();
  }
  release_handle();
  dl_ctx_ = std::move(next_context);
  return *this;
}

Tensor::Tensor(Tensor&& other) noexcept : dl_ctx_(std::move(other.dl_ctx_)) {}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  release_handle();
  dl_ctx_ = std::move(other.dl_ctx_);
  return *this;
}

void Tensor::retain_handle() const noexcept {
  if (dl_ctx_ != nullptr && dl_ctx_->memory_reference != nullptr) {
    dl_ctx_->memory_reference->retain_tensor_handle();
  }
}

void Tensor::release_handle() const noexcept {
  if (dl_ctx_ != nullptr && dl_ctx_->memory_reference != nullptr) {
    dl_ctx_->memory_reference->release_tensor_handle();
  }
}

Tensor::Tensor(DLManagedTensor* dl_managed_tensor_ptr) {
  auto maybe_tensor = make_tensor_from_dlpack(dl_managed_tensor_ptr);
  if (!maybe_tensor) {
    throw RuntimeError(maybe_tensor.error());
  }
  *this = std::move(maybe_tensor.value());
}

Tensor::Tensor(DLManagedTensorVersioned* dl_managed_tensor_ver_ptr) {
  auto maybe_tensor = make_tensor_from_dlpack_versioned(dl_managed_tensor_ver_ptr);
  if (!maybe_tensor) {
    throw RuntimeError(maybe_tensor.error());
  }
  *this = std::move(maybe_tensor.value());
}

const void* Tensor::data() const noexcept {
  if (dl_ctx_ == nullptr || dl_ctx_->tensor.dl_tensor.data == nullptr) {
    return nullptr;
  }
  const auto base = reinterpret_cast<std::uintptr_t>(dl_ctx_->tensor.dl_tensor.data);
  const std::uint64_t offset = dl_ctx_->tensor.dl_tensor.byte_offset;
  if (offset > std::numeric_limits<std::uintptr_t>::max() - base) {
    return nullptr;
  }
  return reinterpret_cast<const void*>(base + static_cast<std::uintptr_t>(offset));
}

bool Tensor::is_read_only() const noexcept {
  return dl_ctx_ != nullptr && dl_ctx_->access_state_->read_only.load(std::memory_order_acquire);
}

expected<void*, Error> Tensor::mutable_data() & {
  if (dl_ctx_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "Tensor context is empty"});
  }
  const void* const logical_data = data();
  if (logical_data == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "Tensor data pointer is null"});
  }
  if (is_read_only()) {
    return make_unexpected(Error{ErrorCode::kPermissionDenied, "Tensor data is read-only"});
  }
  if (dl_ctx_->access_state_->access_claims.load(std::memory_order_acquire) != 0U) {
    return make_unexpected(
        Error{ErrorCode::kPermissionDenied, "Tensor data has an active access claim"});
  }
  return const_cast<void*>(logical_data);
}

bool Tensor::is_contiguous() const {
  if (dl_ctx_ == nullptr) {
    return true;
  }
  const DLTensor& tensor = dl_ctx_->tensor.dl_tensor;
  const int32_t rank = tensor.ndim;
  if (rank <= 0) {
    return true;
  }
  if (tensor.shape == nullptr) {
    return false;
  }
  if (tensor.strides == nullptr) {
    return true;
  }

  // DLPack strides are in elements, so compare directly against the element strides a contiguous
  // row-major tensor would have (independent of dtype itemsize).
  int64_t expected_stride_elements = 1;
  for (int32_t i = rank - 1; i >= 0; --i) {
    if (tensor.shape[i] < 0 || tensor.strides[i] < 0 ||
        tensor.strides[i] != expected_stride_elements) {
      return false;
    }
    if (tensor.shape[i] != 0 &&
        expected_stride_elements > std::numeric_limits<std::int64_t>::max() / tensor.shape[i]) {
      return false;
    }
    expected_stride_elements *= tensor.shape[i];
  }
  return true;
}

expected<DLManagedTensor*, Error> Tensor::to_dlpack() const {
  if (dl_ctx_ == nullptr || data() == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "Tensor has no data to export"});
  }
  if (is_read_only()) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported,
              "Legacy DLPack cannot represent read-only Tensor authority or an active read lease"});
  }
  try {
    auto export_context = std::make_unique<UnversionedDLPackExportContext>();
    auto& dl_managed_tensor = export_context->tensor;

    std::uint32_t expected_claims = 0U;
    if (!dl_ctx_->access_state_->access_claims.compare_exchange_strong(
            expected_claims,
            detail::TensorAccessState::kWriterClaim,
            std::memory_order_acq_rel,
            std::memory_order_acquire)) {
      return make_unexpected(Error{
          ErrorCode::kNotSupported,
          "Legacy DLPack cannot represent read-only Tensor authority or an active read lease"});
    }
    export_context->access_state = dl_ctx_->access_state_;
    if (is_read_only()) {
      return make_unexpected(Error{ErrorCode::kNotSupported,
                                   "Legacy DLPack cannot represent read-only Tensor authority"});
    }

    export_context->memory_reference = dl_ctx_->memory_reference;
    if (export_context->memory_reference != nullptr) {
      export_context->memory_reference->retain_tensor_handle();
    }
    copy_dl_tensor_with_owned_metadata(dl_ctx_->tensor.dl_tensor,
                                       dl_managed_tensor.dl_tensor,
                                       export_context->dl_shape,
                                       export_context->dl_strides);

    dl_managed_tensor.manager_ctx = export_context.release();
    dl_managed_tensor.deleter = [](DLManagedTensor* self) {
      delete static_cast<UnversionedDLPackExportContext*>(self->manager_ctx);
    };

    return &dl_managed_tensor;
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted, "DLPack export allocation failed"});
  }
}

expected<DLManagedTensorVersioned*, Error> Tensor::to_dlpack_versioned() const {
  if (dl_ctx_ == nullptr || data() == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "Tensor has no data to export"});
  }
  try {
    auto export_context = std::make_unique<VersionedDLPackExportContext>();
    bool export_read_only = is_read_only();
    if (!export_read_only) {
      std::uint32_t claims = dl_ctx_->access_state_->access_claims.load(std::memory_order_acquire);
      while (true) {
        if ((claims & detail::TensorAccessState::kWriterClaim) != 0U) {
          return make_unexpected(
              Error{ErrorCode::kPermissionDenied, "Tensor has an active mutable DLPack export"});
        }
        if (claims == 0U) {
          if (dl_ctx_->access_state_->access_claims.compare_exchange_weak(
                  claims,
                  detail::TensorAccessState::kWriterClaim,
                  std::memory_order_acq_rel,
                  std::memory_order_acquire)) {
            export_context->writer_claim = true;
            break;
          }
          continue;
        }
        if ((claims & detail::TensorAccessState::kReaderMask) ==
            detail::TensorAccessState::kReaderMask) {
          return make_unexpected(
              Error{ErrorCode::kResourceExhausted, "Tensor read-lease count is exhausted"});
        }
        if (dl_ctx_->access_state_->access_claims.compare_exchange_weak(
                claims, claims + 1U, std::memory_order_acq_rel, std::memory_order_acquire)) {
          export_read_only = true;
          break;
        }
      }
      export_context->access_state = dl_ctx_->access_state_;
      export_read_only = export_read_only || is_read_only();
    }
    export_context->memory_reference = dl_ctx_->memory_reference;
    if (export_context->memory_reference != nullptr) {
      export_context->memory_reference->retain_tensor_handle();
    }
    export_context->tensor.version = dl_ctx_->tensor.version;
    export_context->tensor.flags = dl_ctx_->tensor.flags;
    // A finite read lease is not a permanent narrowing of the local Tensor, but an exported owner
    // can outlive that lease. Snapshot the least authority valid at export time so DLPack never
    // widens an activation-scoped read view into a mutable external owner.
    if (export_read_only) {
      export_context->tensor.flags |= DLPACK_FLAG_BITMASK_READ_ONLY;
    }
    copy_dl_tensor_with_owned_metadata(
        dl_ctx_->tensor.dl_tensor,
        export_context->tensor.dl_tensor,
        export_context->dl_shape,
        export_context->dl_strides,
        DLPackMetadataCopyOptions{.materialize_implicit_strides = true});

    export_context->tensor.manager_ctx = export_context.get();
    export_context->tensor.deleter = [](DLManagedTensorVersioned* self) {
      delete static_cast<VersionedDLPackExportContext*>(self->manager_ctx);
    };

    return &export_context.release()->tensor;
  } catch (const std::bad_alloc&) {
    return make_unexpected(
        Error{ErrorCode::kResourceExhausted, "Versioned DLPack export allocation failed"});
  }
}

std::vector<int64_t> Tensor::shape() const {
  if (dl_ctx_ == nullptr) {
    return {};
  }
  const auto ndim = dl_ctx_->tensor.dl_tensor.ndim;
  auto* const shape_ptr = dl_ctx_->tensor.dl_tensor.shape;
  if (ndim <= 0 || shape_ptr == nullptr) {
    return {};
  }
  return {shape_ptr, shape_ptr + ndim};
}

std::span<const int64_t> Tensor::shape_span() const {
  if (dl_ctx_ == nullptr) {
    return {};
  }
  const auto ndim = dl_ctx_->tensor.dl_tensor.ndim;
  const auto* shape_ptr = dl_ctx_->tensor.dl_tensor.shape;
  if (ndim <= 0 || shape_ptr == nullptr) {
    return {};
  }
  return {shape_ptr, static_cast<std::size_t>(ndim)};
}

std::vector<int64_t> Tensor::strides() const {
  if (dl_ctx_ == nullptr) {
    return {};
  }
  const DLTensor& dl_tensor = dl_ctx_->tensor.dl_tensor;
  std::vector<int64_t> strides;
  calc_strides(dl_tensor, strides, false);
  return strides;
}

std::span<const int64_t> Tensor::dlpack_strides_span() const {
  if (dl_ctx_ == nullptr) {
    return {};
  }
  const auto ndim = dl_ctx_->tensor.dl_tensor.ndim;
  const auto* strides_ptr = dl_ctx_->tensor.dl_tensor.strides;
  if (ndim <= 0 || strides_ptr == nullptr) {
    return {};
  }
  return {strides_ptr, static_cast<std::size_t>(ndim)};
}

int64_t Tensor::size() const noexcept {
  if (dl_ctx_ == nullptr) {
    return 0;
  }
  const auto ndim = dl_ctx_->tensor.dl_tensor.ndim;
  if (ndim < 0) {
    return 0;
  }
  if (ndim == 0) {
    return 1;  // DLPack scalar convention: ndim == 0 is a single-element tensor.
  }
  auto* const shape_ptr = dl_ctx_->tensor.dl_tensor.shape;
  if (shape_ptr == nullptr) {
    return 0;
  }
  int64_t size = 1;
  for (int i = 0; i < ndim; ++i) {
    const int64_t extent = shape_ptr[i];
    if (extent <= 0) {
      return 0;
    }
    if (size > std::numeric_limits<int64_t>::max() / extent) {
      return 0;
    }
    size *= extent;
  }
  return size;
}

uint64_t Tensor::itemsize() const noexcept {
  if (dl_ctx_ == nullptr) {
    return 0U;
  }
  const DLDataType dtype = dl_ctx_->tensor.dl_tensor.dtype;
  if (dtype.bits == 0U || dtype.lanes == 0U) {
    return 0U;
  }
  return dlpack_dtype_item_size_bytes(dtype);
}

int64_t Tensor::nbytes() const noexcept {
  const int64_t element_count = size();
  const uint64_t item_bytes = itemsize();
  if (element_count <= 0 || item_bytes == 0U) {
    return 0;
  }
  constexpr uint64_t kMaxByteCount = static_cast<uint64_t>(std::numeric_limits<int64_t>::max());
  const uint64_t unsigned_element_count = static_cast<uint64_t>(element_count);
  if (item_bytes > kMaxByteCount / unsigned_element_count) {
    return 0;
  }
  return static_cast<int64_t>(unsigned_element_count * item_bytes);
}

bool Tensor::set_readiness_stream(cudaStream_t stream) {
  return dl_ctx_ != nullptr && dl_ctx_->memory_reference != nullptr &&
         dl_ctx_->memory_reference->set_readiness_stream(stream);
}

expected<void, Error> Tensor::synchronize_for_consumer_stream(cudaStream_t consumer_stream) const {
  if (dl_ctx_ == nullptr || dl_ctx_->memory_reference == nullptr) {
    return {};
  }
  if (!dl_ctx_->memory_reference->has_readiness_stream_contract()) {
    return make_unexpected(Error{
        ErrorCode::kNotImplemented,
        "Tensor memory reference does not provide a producer-readiness stream contract; "
        "imported DLPack producers must synchronize through __dlpack__(stream=...) or provide a "
        "Holoscan readiness-aware memory reference"});
  }
  return dl_ctx_->memory_reference->synchronize_for_consumer(consumer_stream);
}

bool Tensor::has_readiness_stream_contract() const noexcept {
  return dl_ctx_ != nullptr && dl_ctx_->memory_reference != nullptr &&
         dl_ctx_->memory_reference->has_readiness_stream_contract();
}

expected<void, Error> Tensor::synchronize_for_host() const {
  if (dl_ctx_ == nullptr || dl_ctx_->memory_reference == nullptr) {
    return {};
  }
  return dl_ctx_->memory_reference->synchronize_for_host();
}

expected<void, Error> Tensor::copy_to_host_async(void* destination, std::size_t destination_bytes,
                                                 cudaStream_t stream) const {
  if (destination == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "Tensor copy destination is null"});
  }
  if (dl_ctx_ == nullptr || dl_ctx_->tensor.dl_tensor.data == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "Tensor copy source is null"});
  }
  if (!is_contiguous()) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Tensor copy_to_host only supports contiguous tensors"});
  }

  const auto byte_count = nbytes();
  if (byte_count < 0) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "Tensor byte size is negative"});
  }
  const auto bytes = static_cast<std::size_t>(byte_count);
  if (destination_bytes < bytes) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Tensor copy destination is too small"});
  }
  if (bytes == 0) {
    return {};
  }

  const auto* source = static_cast<const std::byte*>(data());
  if (source == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Tensor logical data pointer overflows"});
  }
  const auto device_type = dl_ctx_->tensor.dl_tensor.device.device_type;
  if (is_host_tensor_device(device_type)) {
    auto sync_result = synchronize_for_host();
    if (!sync_result) {
      return sync_result;
    }
    std::memcpy(destination, source, bytes);
    return {};
  }
  if (is_cuda_tensor_device(device_type)) {
    auto sync_result = synchronize_for_consumer_stream(stream);
    if (!sync_result) {
      return sync_result;
    }
    auto copied =
        make_cuda_error(cudaMemcpyAsync(destination, source, bytes, cudaMemcpyDeviceToHost, stream),
                        "cudaMemcpyAsync failed during Tensor host copy");
    if (!copied) {
      return copied;
    }
    if (dl_ctx_->memory_reference != nullptr &&
        dl_ctx_->memory_reference->set_deallocation_stream(stream)) {
      return {};
    }
    // Imported owners without a stream-aware retirement contract can run their deleter as soon as
    // the final Tensor handle disappears. Complete their read before returning.
    return make_cuda_error(
        cudaStreamSynchronize(stream),
        "cudaStreamSynchronize failed while retaining Tensor source for host copy");
  }

  return make_unexpected(
      Error{ErrorCode::kInvalidArgument, "Tensor device type is unsupported for host copy"});
}

expected<void, Error> Tensor::copy_to_host(void* destination, std::size_t destination_bytes,
                                           cudaStream_t stream) const {
  auto copy_result = copy_to_host_async(destination, destination_bytes, stream);
  if (!copy_result) {
    return copy_result;
  }
  const auto device_type = dl_ctx_->tensor.dl_tensor.device.device_type;
  if (is_cuda_tensor_device(device_type)) {
    return make_cuda_error(cudaStreamSynchronize(stream),
                           "cudaStreamSynchronize failed during Tensor host copy");
  }
  return {};
}

expected<Tensor, Error> Tensor::copy(cudaStream_t stream) const {
  if (dl_ctx_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "Tensor copy source is empty"});
  }
  return copy_to(device(), stream);
}

// Copy dispatch deliberately keeps host, device, and peer-device synchronization in one place so
// source-lifetime completion cannot diverge between paths on this hot operation.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<Tensor, Error> Tensor::copy_to(DLDevice target, cudaStream_t stream) const {
  if (dl_ctx_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "Tensor copy source is empty"});
  }
  if (ndim() < 0) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Tensor copy source rank is negative"});
  }
  if (!is_contiguous()) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Tensor copy only supports contiguous tensors"});
  }
  const auto byte_count = nbytes();
  if (byte_count < 0) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "Tensor byte size is negative"});
  }
  const auto bytes = static_cast<std::size_t>(byte_count);
  if (bytes != 0U && data() == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "Tensor copy source is empty"});
  }
  const auto destination_storage = copy_storage_type(target);
  if (!destination_storage) {
    return make_unexpected(destination_storage.error());
  }

  const DLDevice source = device();
  if (!is_host_tensor_device(source.device_type) && !is_cuda_copy_device(source.device_type)) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Tensor copy source device is unsupported"});
  }
  const std::optional<std::int32_t> selected_device =
      target.device_type == kDLCUDA
          ? std::optional<std::int32_t>{target.device_id}
          : (source.device_type == kDLCUDA ? std::optional<std::int32_t>{source.device_id}
                                           : std::nullopt);
  const ScopedCudaDevice selected{selected_device};
  if (selected_device && !selected.active()) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Tensor copy CUDA device selection failed"});
  }

  // Explicit copy is an escape hatch from borrowed or provider-owned storage, so its independent
  // lifetime deliberately uses a fresh unbounded allocation rather than an unrelated runtime pool.
  auto allocator = std::make_shared<UnboundedAllocator>();
  auto destination = make_tensor(shape_span(),
                                 dtype(),
                                 std::nullopt,
                                 *destination_storage,
                                 allocator,
                                 static_cast<void*>(stream));
  if (!destination) {
    return make_unexpected(std::move(destination).error());
  }
  if (bytes != 0U) {
    auto destination_data = destination->mutable_data();
    if (!destination_data) {
      return make_unexpected(std::move(destination_data).error());
    }
    if (is_host_tensor_device(source.device_type) && is_host_tensor_device(target.device_type)) {
      auto synchronized = synchronize_for_host();
      if (!synchronized) {
        return make_unexpected(std::move(synchronized).error());
      }
      std::memcpy(*destination_data, data(), bytes);
    } else {
      auto synchronized = is_host_tensor_device(source.device_type)
                              ? synchronize_for_host()
                              : synchronize_for_consumer_stream(stream);
      if (!synchronized) {
        return make_unexpected(std::move(synchronized).error());
      }
      cudaError_t copied = cudaSuccess;
      if (source.device_type == kDLCUDA && target.device_type == kDLCUDA &&
          source.device_id != target.device_id) {
        copied = cudaMemcpyPeerAsync(
            *destination_data, target.device_id, data(), source.device_id, bytes, stream);
      } else {
        copied = cudaMemcpyAsync(*destination_data, data(), bytes, cudaMemcpyDefault, stream);
      }
      auto copy_result = make_cuda_error(copied, "cudaMemcpyAsync failed during Tensor copy");
      if (!copy_result) {
        return make_unexpected(std::move(copy_result).error());
      }
      // The returned Tensor owns its destination, but it cannot retain every possible source
      // backing contract. Complete the source read before the caller can retire its last source
      // handle. A future asynchronous implementation must carry an equivalent source-lifetime
      // claim until this copy completes.
      auto synchronized_copy = make_cuda_error(cudaStreamSynchronize(stream),
                                               "cudaStreamSynchronize failed during Tensor copy");
      if (!synchronized_copy) {
        return make_unexpected(std::move(synchronized_copy).error());
      }
      if (!is_host_tensor_device(target.device_type)) {
        (void)destination->set_readiness_stream(stream);
        (void)destination->set_deallocation_stream(stream);
      }
    }
  }
  destination->dl_ctx_->tensor.flags |= DLPACK_FLAG_BITMASK_IS_COPIED;
  return std::move(*destination);
}

bool Tensor::set_deallocation_stream(cudaStream_t stream) {
  return dl_ctx_ != nullptr && dl_ctx_->memory_reference != nullptr &&
         dl_ctx_->memory_reference->set_deallocation_stream(stream);
}

namespace detail {

TensorReadLease::TensorReadLease(TensorReadLease&& other) noexcept
    : state_(std::move(other.state_)), context_(std::move(other.context_)) {}

TensorReadLease& TensorReadLease::operator=(TensorReadLease&& other) noexcept {
  if (this != &other) {
    release();
    state_ = std::move(other.state_);
    context_ = std::move(other.context_);
  }
  return *this;
}

TensorReadLease::~TensorReadLease() {
  release();
}

void TensorReadLease::release() noexcept {
  if (state_ != nullptr) {
    state_->access_claims.fetch_sub(1U, std::memory_order_acq_rel);
    if (context_ != nullptr && context_->memory_reference != nullptr) {
      context_->memory_reference->release_tensor_handle();
    }
    context_.reset();
    state_.reset();
  }
}

const std::shared_ptr<DLManagedTensorContext>& TensorAccess::context(
    const Tensor& tensor) noexcept {
  return tensor.dl_ctx_;
}

TensorAccessSnapshot TensorAccess::snapshot(const Tensor& tensor) noexcept {
  TensorAccessSnapshot result;
  if (tensor.dl_ctx_ == nullptr) {
    return result;
  }
  result.initialized = true;
  const DLTensor& dl_tensor = tensor.dl_ctx_->tensor.dl_tensor;
  result.data = dl_tensor.data;
  result.byte_offset = dl_tensor.byte_offset;
  result.device = dl_tensor.device;
  if (dl_tensor.ndim < 0 || (dl_tensor.ndim > 0 && dl_tensor.shape == nullptr) ||
      dl_tensor.dtype.bits == 0U || dl_tensor.dtype.lanes == 0U) {
    return result;
  }

  const std::span<const std::int64_t> shape{dl_tensor.shape,
                                            static_cast<std::size_t>(dl_tensor.ndim)};
  const std::optional<std::span<const std::int64_t>> strides =
      dl_tensor.strides == nullptr
          ? std::nullopt
          : std::optional<std::span<const std::int64_t>>{std::span<const std::int64_t>{
                dl_tensor.strides, static_cast<std::size_t>(dl_tensor.ndim)}};
  const auto layout =
      validate_tensor_layout(shape, dl_tensor.dtype, strides, TensorStrideUnit::kElements);
  if (!layout) {
    return result;
  }
  result.byte_size = layout->byte_size;
  result.byte_span = layout->byte_span;
  if (result.byte_span > 0U && result.data == nullptr) {
    return result;
  }
  if (result.byte_offset > std::numeric_limits<std::uint64_t>::max() - result.byte_span) {
    return result;
  }
  result.valid = true;
  return result;
}

}  // namespace detail

bool tensor_matches(const Tensor& tensor, const TensorViewMetadata& metadata) noexcept {
  const detail::TensorAccessSnapshot snapshot = detail::TensorAccess::snapshot(tensor);
  if (!snapshot.initialized || !snapshot.valid || metadata.rank > TensorViewMetadata::kMaxRank ||
      tensor.ndim() < 0 || static_cast<std::uint32_t>(tensor.ndim()) != metadata.rank ||
      tensor.dtype().code != metadata.dtype.code || tensor.dtype().bits != metadata.dtype.bits ||
      tensor.dtype().lanes != metadata.dtype.lanes ||
      tensor.device().device_type != metadata.device.device_type ||
      tensor.device().device_id != metadata.device.device_id ||
      snapshot.byte_size != metadata.byte_size) {
    return false;
  }

  const auto shape = tensor.shape_span();
  if (shape.size() != metadata.rank ||
      !std::equal(shape.begin(), shape.end(), metadata.shape.begin())) {
    return false;
  }

  const auto strides = tensor.dlpack_strides_span();
  if (metadata.has_explicit_byte_strides != !strides.empty()) {
    return false;
  }
  if (!metadata.has_explicit_byte_strides) {
    return true;
  }

  const std::uint64_t item_size_bytes = tensor.itemsize();
  if (strides.size() != metadata.rank || item_size_bytes == 0U) {
    return false;
  }
  for (std::size_t index = 0; index < strides.size(); ++index) {
    if (strides[index] < 0 ||
        static_cast<std::uint64_t>(strides[index]) >
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) /
                item_size_bytes ||
        metadata.byte_strides[index] !=
            static_cast<std::int64_t>(static_cast<std::uint64_t>(strides[index]) *
                                      item_size_bytes)) {
      return false;
    }
  }
  return true;
}

namespace detail {

cudaStream_t TensorAccess::readiness_stream(const Tensor& tensor) noexcept {
  return tensor.dl_ctx_ != nullptr && tensor.dl_ctx_->memory_reference != nullptr
             ? tensor.dl_ctx_->memory_reference->readiness_stream()
             : nullptr;
}

bool TensorAccess::tracks_deallocation_stream(const Tensor& tensor) noexcept {
  return tensor.dl_ctx_ != nullptr && tensor.dl_ctx_->memory_reference != nullptr &&
         tensor.dl_ctx_->memory_reference->tracks_deallocation_stream();
}

Tensor TensorAccess::from_context(std::shared_ptr<DLManagedTensorContext> context,
                                  TensorAuthority authority) {
  Tensor tensor(std::move(context));
  if (authority == TensorAuthority::kReadOnly) {
    narrow_read_only(tensor);
  }
  return tensor;
}

void TensorAccess::replace_context(Tensor& tensor, std::shared_ptr<DLManagedTensorContext> context,
                                   TensorAuthority authority) {
  Tensor replacement = from_context(std::move(context), authority);
  tensor = std::move(replacement);
}

void TensorAccess::narrow_read_only(Tensor& tensor) noexcept {
  if (tensor.dl_ctx_ != nullptr) {
    tensor.dl_ctx_->access_state_->read_only.store(true, std::memory_order_release);
    tensor.dl_ctx_->tensor.flags |= DLPACK_FLAG_BITMASK_READ_ONLY;
  }
}

expected<TensorReadLease, Error> TensorAccess::acquire_read_lease(const Tensor& tensor) {
  if (tensor.dl_ctx_ == nullptr || tensor.data() == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "Tensor has no data to lease"});
  }
  auto claims = tensor.dl_ctx_->access_state_->access_claims.load(std::memory_order_acquire);
  while (true) {
    if ((claims & TensorAccessState::kWriterClaim) != 0U) {
      return make_unexpected(
          Error{ErrorCode::kPermissionDenied, "Tensor has an active mutable DLPack export"});
    }
    if ((claims & TensorAccessState::kReaderMask) == TensorAccessState::kReaderMask) {
      return make_unexpected(
          Error{ErrorCode::kResourceExhausted, "Tensor read-lease count is exhausted"});
    }
    if (tensor.dl_ctx_->access_state_->access_claims.compare_exchange_weak(
            claims, claims + 1U, std::memory_order_acq_rel, std::memory_order_acquire)) {
      break;
    }
  }
  return TensorReadLease(tensor.dl_ctx_->access_state_, tensor.dl_ctx_);
}

}  // namespace detail

}  // namespace holoscan
