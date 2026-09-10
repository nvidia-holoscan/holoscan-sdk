/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstddef>
#include <new>
#include <utility>

#include "runtime_channel_storage.hpp"

namespace holoscan::detail {

// Aligned backing storage and bounded read-claim ownership primitives.

AlignedBlock::AlignedBlock(AlignedBlock&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      alignment_(std::exchange(other.alignment_, alignof(std::max_align_t))) {}

AlignedBlock& AlignedBlock::operator=(AlignedBlock&& other) noexcept {
  if (this != &other) {
    reset();
    data_ = std::exchange(other.data_, nullptr);
    alignment_ = std::exchange(other.alignment_, alignof(std::max_align_t));
  }
  return *this;
}

AlignedBlock::~AlignedBlock() {
  reset();
}

expected<AlignedBlock, Error> AlignedBlock::create(const std::size_t size,
                                                   const std::size_t alignment) noexcept {
  if (size == 0U || alignment == 0U || (alignment & (alignment - 1U)) != 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  void* storage = ::operator new(size, std::align_val_t{alignment}, std::nothrow);
  if (storage == nullptr) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  AlignedBlock result;
  result.data_ = storage;
  result.alignment_ = alignment;
  return result;
}

void AlignedBlock::reset() noexcept {
  if (data_ != nullptr) {
    ::operator delete(data_, std::align_val_t{alignment_});
    data_ = nullptr;
  }
}

expected<ReadClaimPool*, Error> ReadClaimPool::create(const std::size_t capacity) noexcept {
  if (capacity == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    return new ReadClaimPool(capacity);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
#endif
}

ReadClaimPool::ReadClaimPool(const std::size_t capacity)
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
    : slots_(std::make_unique<Slot[]>(capacity)), capacity_(capacity) {
  for (std::size_t index = 0U; index < capacity_; ++index) {
    slots_[index].pool = this;
  }
}

void ReadClaimPool::release_owner() noexcept {
  if (references_.fetch_sub(1U, std::memory_order_acq_rel) == 1U) {
    delete this;
  }
}

expected<DeviceId, Error> uniform_tensor_field_device(
    const DeviceId fallback,
    const std::span<const TensorFieldServiceStorageContract> fields) noexcept {
  DeviceId result = fallback;
  for (const auto& field : fields) {
    if (!field.device.valid()) {
      continue;
    }
    if (result.valid() && result != field.device) {
      return make_unexpected(
          Error{ErrorCode::kProtocolError,
                "descriptor Tensor field storage spans multiple CUDA devices for one endpoint"});
    }
    result = field.device;
  }
  return result;
}

}  // namespace holoscan::detail
