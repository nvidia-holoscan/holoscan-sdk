/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>

#include <cuda_runtime_api.h>

#include <holoscan/core/domain/dlpack_utils.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "../compiler/execution_plan_state.hpp"
#include "../core/domain/tensor_access.hpp"

namespace holoscan::detail {

/// Restore the ambient CUDA device after a bounded Channel scan or import operation.
class CudaDeviceCursor final {
 public:
  CudaDeviceCursor() noexcept : active_(cudaGetDevice(&previous_) == cudaSuccess) {
    current_ = previous_;
  }

  CudaDeviceCursor(const CudaDeviceCursor&) = delete;
  CudaDeviceCursor& operator=(const CudaDeviceCursor&) = delete;

  ~CudaDeviceCursor() { static_cast<void>(restore()); }

  [[nodiscard]] bool active() const noexcept { return active_; }

  [[nodiscard]] bool select(const DeviceId requested) noexcept {
    if (!active_) {
      return false;
    }
    if (!requested.valid()) {
      return restore();
    }
    if (current_ != requested.value) {
      if (cudaSetDevice(requested.value) != cudaSuccess) {
        return false;
      }
      current_ = requested.value;
    }
    return true;
  }

  [[nodiscard]] bool initialize(const DeviceId requested = {}) noexcept {
    return select(requested) && cudaFree(nullptr) == cudaSuccess;
  }

  [[nodiscard]] bool restore() noexcept {
    if (!active_ || current_ == previous_) {
      return active_;
    }
    if (cudaSetDevice(previous_) != cudaSuccess) {
      return false;
    }
    current_ = previous_;
    return true;
  }

 private:
  int previous_{};
  int current_{};
  bool active_{};
};

/// Clear borrowed Tensor scratch before its backing vector leaves the current operation.
class TensorScratchReset final {
 public:
  explicit TensorScratchReset(std::span<Tensor> tensors) noexcept : tensors_(tensors) {}
  TensorScratchReset(const TensorScratchReset&) = delete;
  TensorScratchReset& operator=(const TensorScratchReset&) = delete;
  ~TensorScratchReset() {
    for (auto& tensor : tensors_) {
      tensor = Tensor{};
    }
  }

 private:
  std::span<Tensor> tensors_;
};

/// Own one cold-allocated block with an explicit alignment contract.
class AlignedBlock {
 public:
  AlignedBlock() = default;
  AlignedBlock(const AlignedBlock&) = delete;
  AlignedBlock& operator=(const AlignedBlock&) = delete;
  AlignedBlock(AlignedBlock&& other) noexcept;
  AlignedBlock& operator=(AlignedBlock&& other) noexcept;
  ~AlignedBlock();

  [[nodiscard]] static expected<AlignedBlock, Error> create(std::size_t size,
                                                            std::size_t alignment) noexcept;

  [[nodiscard]] void* data() noexcept { return data_; }
  [[nodiscard]] const void* data() const noexcept { return data_; }

 private:
  void reset() noexcept;

  void* data_{};
  std::size_t alignment_{alignof(std::max_align_t)};
};

/// Own the fixed-capacity, address-stable claims protecting retained Channel samples.
class ReadClaimPool {
 public:
  struct Slot {
    ReadClaimPool* pool{};
    std::atomic<std::uint32_t> references{};
    std::atomic<bool> in_use{};
  };

  [[nodiscard]] static expected<ReadClaimPool*, Error> create(std::size_t capacity) noexcept;

  [[nodiscard]] expected<PublishedReadClaimBinding, Error> acquire() noexcept {
    for (std::size_t index = 0U; index < capacity_; ++index) {
      Slot& slot = slots_[index];
      bool available = false;
      if (slot.in_use.compare_exchange_strong(
              available, true, std::memory_order_acquire, std::memory_order_relaxed)) {
        slot.references.store(1U, std::memory_order_relaxed);
        live_claims_.fetch_add(1U, std::memory_order_release);
        references_.fetch_add(1U, std::memory_order_relaxed);
        return PublishedReadClaimBinding{
            .state = &slot,
            .retain = &retain,
            .release = &release,
        };
      }
    }
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }

  void release_owner() noexcept;

  [[nodiscard]] bool has_live_claims() const noexcept {
    return live_claims_.load(std::memory_order_acquire) != 0U;
  }

 private:
  explicit ReadClaimPool(std::size_t capacity);

  static void retain(void* opaque) noexcept {
    auto& slot = *static_cast<Slot*>(opaque);
    slot.references.fetch_add(1U, std::memory_order_relaxed);
  }

  static void release(void* opaque) noexcept {
    auto& slot = *static_cast<Slot*>(opaque);
    if (slot.references.fetch_sub(1U, std::memory_order_acq_rel) != 1U) {
      return;
    }
    slot.in_use.store(false, std::memory_order_release);
    ReadClaimPool* const pool = slot.pool;
    pool->live_claims_.fetch_sub(1U, std::memory_order_acq_rel);
    if (pool->references_.fetch_sub(1U, std::memory_order_acq_rel) == 1U) {
      delete pool;
    }
  }

  // Runtime capacity is frozen before construction, while Slot contains atomics and is not
  // relocatable. A single fixed-capacity allocation therefore preserves stable slot addresses.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
  std::unique_ptr<Slot[]> slots_;
  std::size_t capacity_{};
  std::atomic<std::uint32_t> references_{1U};
  std::atomic<std::uint32_t> live_claims_{};
};

/// Return whether a trusted Tensor snapshot fits one compiler-frozen storage envelope.
[[nodiscard]] inline bool tensor_snapshot_matches_plan(const TensorAccessSnapshot& snapshot,
                                                       const std::uint64_t max_byte_span,
                                                       const MemoryKind memory_kind,
                                                       const DeviceId device) noexcept {
  if (!snapshot.initialized || !snapshot.valid) {
    return false;
  }
  if (memory_kind == MemoryKind::kUnknown) {
    return max_byte_span == 0U && !device.valid();
  }
  if (max_byte_span == 0U || snapshot.byte_span > max_byte_span) {
    return false;
  }
  switch (memory_kind) {
    case MemoryKind::kHost:
      return !device.valid() && snapshot.device.device_type == kDLCPU &&
             snapshot.device.device_id == 0;
    case MemoryKind::kPinnedHost:
      // DLPack fixes pinned-host device_id at zero. A valid private plan device may still select
      // the HoloMQ generation-sync context; it is not encoded in public Tensor metadata.
      return snapshot.device.device_type == kDLCUDAHost && snapshot.device.device_id == 0;
    case MemoryKind::kCudaDevice:
      return device.valid() && snapshot.device.device_type == kDLCUDA &&
             snapshot.device.device_id == device.value;
    case MemoryKind::kCudaManaged:
      // DLPack uses device id zero for managed allocations. Compile still requires an explicit
      // source-private DeviceId so Runtime never infers device zero from that convention.
      return device.valid() && snapshot.device.device_type == kDLCUDAManaged &&
             snapshot.device.device_id == 0;
    case MemoryKind::kUnknown:
    default:
      return false;
  }
}

[[nodiscard]] inline bool tensor_matches_field_contract(
    const Tensor& tensor, const TensorAccessSnapshot& snapshot,
    const TensorFieldServiceStorageContract& field) noexcept {
  if (!tensor_snapshot_matches_plan(
          snapshot, field.max_byte_span, field.memory_kind, field.device)) {
    return false;
  }
  if (field.rank && tensor.ndim() != static_cast<std::int32_t>(*field.rank)) {
    return false;
  }
  if (field.dtype) {
    const DLDataType actual = tensor.dtype();
    if (actual.code != field.dtype->code || actual.bits != field.dtype->bits ||
        actual.lanes != field.dtype->lanes) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] expected<DeviceId, Error> uniform_tensor_field_device(
    DeviceId fallback, std::span<const TensorFieldServiceStorageContract> fields) noexcept;

[[nodiscard]] inline bool tensor_matches_port_layout(const Tensor& tensor,
                                                     const TensorAccessSnapshot& snapshot,
                                                     const TensorRepresentation& layout) noexcept {
  if (!snapshot.initialized || !snapshot.valid) {
    return false;
  }
  if (layout.memory_kind) {
    const bool memory_matches = [&]() noexcept {
      switch (*layout.memory_kind) {
        case MemoryKind::kHost:
          return snapshot.device.device_type == kDLCPU && snapshot.device.device_id == 0;
        case MemoryKind::kPinnedHost:
          return snapshot.device.device_type == kDLCUDAHost && snapshot.device.device_id == 0;
        case MemoryKind::kCudaDevice:
          return snapshot.device.device_type == kDLCUDA && snapshot.device.device_id >= 0;
        case MemoryKind::kCudaManaged:
          return snapshot.device.device_type == kDLCUDAManaged && snapshot.device.device_id == 0;
        case MemoryKind::kUnknown:
        default:
          return false;
      }
    }();
    if (!memory_matches) {
      return false;
    }
  }
  if (layout.dtype) {
    const DLDataType actual = tensor.dtype();
    if (actual.code != layout.dtype->code || actual.bits != layout.dtype->bits ||
        actual.lanes != layout.dtype->lanes) {
      return false;
    }
  }
  if (layout.rank && static_cast<std::uint32_t>(tensor.ndim()) != *layout.rank) {
    return false;
  }
  return !layout.strides || *layout.strides == TensorStridePolicy::kBoundedStrided ||
         tensor.is_contiguous();
}

[[nodiscard]] inline std::optional<std::uint64_t> tensor_axis_byte_stride(
    const Tensor& tensor, const std::size_t axis, const std::uint64_t item_bytes) noexcept {
  const auto shape = tensor.shape_span();
  if (axis >= shape.size()) {
    return std::nullopt;
  }
  const auto strides = tensor.dlpack_strides_span();
  if (!strides.empty()) {
    if (strides.size() != shape.size() || strides[axis] < 0 ||
        static_cast<std::uint64_t>(strides[axis]) >
            std::numeric_limits<std::uint64_t>::max() / item_bytes) {
      return std::nullopt;
    }
    return static_cast<std::uint64_t>(strides[axis]) * item_bytes;
  }
  std::uint64_t stride = item_bytes;
  for (std::size_t reverse = shape.size(); reverse > axis + 1U; --reverse) {
    if (shape[reverse - 1U] < 0) {
      return std::nullopt;
    }
    const auto extent =
        std::max<std::uint64_t>(static_cast<std::uint64_t>(shape[reverse - 1U]), 1U);
    if (stride > std::numeric_limits<std::uint64_t>::max() / extent) {
      return std::nullopt;
    }
    stride *= extent;
  }
  return stride;
}

[[nodiscard]] inline bool tensor_matches_port_contract(
    const Tensor& tensor, const TensorAccessSnapshot& snapshot,
    const TensorContractRecord& contract) noexcept {
  if (!snapshot.initialized || !snapshot.valid) {
    return false;
  }
  if (!tensor_matches_port_layout(tensor, snapshot, contract.representation)) {
    return false;
  }
  const auto rank = tensor.ndim();
  if (rank < 0 || (contract.max_rank && static_cast<std::uint32_t>(rank) > *contract.max_rank) ||
      (contract.max_byte_span && snapshot.byte_span > *contract.max_byte_span)) {
    return false;
  }
  const auto shape = tensor.shape_span();
  if (contract.axes.size() > shape.size()) {
    return false;
  }
  const std::uint64_t item_bytes = dlpack_dtype_item_size_bytes(tensor.dtype());
  if (item_bytes == 0U) {
    return false;
  }
  for (std::size_t axis = 0U; axis < contract.axes.size(); ++axis) {
    if (shape[axis] < 0 ||
        static_cast<std::uint64_t>(shape[axis]) > contract.axes[axis].max_extent) {
      return false;
    }
    const auto stride = tensor_axis_byte_stride(tensor, axis, item_bytes);
    if (!stride || (contract.axes[axis].max_abs_stride_bytes != 0U &&
                    *stride > contract.axes[axis].max_abs_stride_bytes)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline bool tensor_matches_logical_requirements(
    const Tensor& tensor, const TensorBoundaryContractRecord& requirements) noexcept {
  const auto rank = tensor.ndim();
  if (rank < 0) {
    return false;
  }
  const auto logical_rank = static_cast<std::uint32_t>(rank);
  if ((requirements.rank && logical_rank != *requirements.rank) ||
      (requirements.max_rank && logical_rank > *requirements.max_rank)) {
    return false;
  }
  if (requirements.dtype) {
    const DLDataType actual = tensor.dtype();
    if (actual.code != requirements.dtype->code || actual.bits != requirements.dtype->bits ||
        actual.lanes != requirements.dtype->lanes) {
      return false;
    }
  }
  const auto shape = tensor.shape_span();
  if (shape.size() != static_cast<std::size_t>(rank) || requirements.axes.size() > shape.size()) {
    return false;
  }
  for (std::size_t axis = 0U; axis < requirements.axes.size(); ++axis) {
    if (shape[axis] < 0 ||
        static_cast<std::uint64_t>(shape[axis]) > requirements.axes[axis].max_extent) {
      return false;
    }
  }
  const std::uint64_t item_size = dlpack_dtype_item_size_bytes(tensor.dtype());
  if (item_size == 0U) {
    return false;
  }
  std::uint64_t contiguous_stride = item_size;
  for (std::size_t reverse = shape.size(); reverse > 0U; --reverse) {
    const std::size_t axis = reverse - 1U;
    if (axis < requirements.axes.size() && requirements.axes[axis].max_abs_stride_bytes != 0U &&
        contiguous_stride > requirements.axes[axis].max_abs_stride_bytes) {
      return false;
    }
    const std::uint64_t extent =
        std::max<std::uint64_t>(static_cast<std::uint64_t>(shape[axis]), 1U);
    if (contiguous_stride > std::numeric_limits<std::uint64_t>::max() / extent) {
      return false;
    }
    contiguous_stride *= extent;
  }
  // Baseline external Tensor execution admits only contiguous storage. Keep this check at the live
  // import boundary so an explicitly strided DLPack view cannot bypass catalog-time validation.
  return requirements.strides == TensorStridePolicy::kContiguousOnly && tensor.is_contiguous();
}

}  // namespace holoscan::detail
