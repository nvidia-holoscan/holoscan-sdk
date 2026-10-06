/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <utility>

#include <holoscan/core/domain/tensor.hpp>

namespace holoscan::detail {

/// Initial mutation authority assigned by trusted Tensor construction paths.
enum class TensorAuthority : std::uint8_t { kLocalWritable = 0, kReadOnly = 1 };

/// Allocation-free metadata snapshot used by trusted runtime validation paths.
struct TensorAccessSnapshot {
  bool initialized = false;
  bool valid = false;
  const void* data = nullptr;
  std::uint64_t byte_offset = 0;
  std::uint64_t byte_size = 0;
  std::uint64_t byte_span = 0;
  DLDevice device{kDLCPU, 0};
};

/**
 * @brief Move-only internal claim that temporarily protects Tensor backing from mutable access.
 *
 * The lease owns one explicit TensorMemoryReference claim in addition to its shared metadata
 * context. This prevents a pool-backed allocation from being recycled while the runtime still
 * holds an activation-scoped read view. This type is an implementation boundary for runtime
 * selection and transport code. It is not an installed application API.
 */
class TensorReadLease {
 public:
  TensorReadLease() = default;
  TensorReadLease(const TensorReadLease&) = delete;
  TensorReadLease& operator=(const TensorReadLease&) = delete;
  HOLOSCAN_CORE_EXPORT TensorReadLease(TensorReadLease&& other) noexcept;
  HOLOSCAN_CORE_EXPORT TensorReadLease& operator=(TensorReadLease&& other) noexcept;
  HOLOSCAN_CORE_EXPORT ~TensorReadLease();

  [[nodiscard]] explicit operator bool() const noexcept { return state_ != nullptr; }

 private:
  friend class TensorAccess;
  TensorReadLease(std::shared_ptr<TensorAccessState> state,
                  std::shared_ptr<DLManagedTensorContext> context) noexcept
      : state_(std::move(state)), context_(std::move(context)) {
    if (context_ != nullptr && context_->memory_reference != nullptr) {
      context_->memory_reference->retain_tensor_handle();
    }
  }
  void release() noexcept;

  std::shared_ptr<TensorAccessState> state_;
  std::shared_ptr<DLManagedTensorContext> context_;
};

/**
 * @brief Internal friend adapter for trusted Tensor materialization and read-lease operations.
 *
 * Application code must use Tensor's public metadata and access methods. This adapter replaces
 * the former public mutable `dl_ctx()` escape hatch without exposing an authority-upgrade path.
 */
class TensorAccess {
 public:
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static const std::shared_ptr<DLManagedTensorContext>& context(
      const Tensor& tensor) noexcept;
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static TensorAccessSnapshot snapshot(
      const Tensor& tensor) noexcept;
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static cudaStream_t readiness_stream(
      const Tensor& tensor) noexcept;
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static bool tracks_deallocation_stream(
      const Tensor& tensor) noexcept;
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static Tensor from_context(
      std::shared_ptr<DLManagedTensorContext> context, TensorAuthority authority);
  HOLOSCAN_CORE_EXPORT static void replace_context(Tensor& tensor,
                                                   std::shared_ptr<DLManagedTensorContext> context,
                                                   TensorAuthority authority);
  HOLOSCAN_CORE_EXPORT static void narrow_read_only(Tensor& tensor) noexcept;
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<TensorReadLease, Error> acquire_read_lease(
      const Tensor& tensor);
};

}  // namespace holoscan::detail
