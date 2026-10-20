/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/stream>

#include <cstddef>
#include <optional>
#include <type_traits>
#include <utility>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/memory/tensor_shape.hpp>

namespace holoscan {

namespace channel_internal {
struct TensorLoanWriteGuardOps;
struct TensorPublishLoanOps;
class TensorPublishLoanAccess;
}  // namespace channel_internal

/// @brief Hold temporary producer write authority for one external Tensor publication loan.
///
/// The guard is move-constructible but neither copyable nor assignable. It borrows its backing
/// publication loan and must not outlive it. Committing consumes the guard and records readiness;
/// destruction without commit abandons the write phase through the source-private control state.
/// The guard never owns a Tensor, DLPack object, CUDA stream, pool, slot, or transport handle.
class HOLOSCAN_CORE_EXPORT TensorLoanWriteGuard {
 public:
  /// @brief Transfer the sole write phase from another guard.
  /// @param other Guard whose authority becomes moved from.
  /// @post `other` exposes null data, zero bytes, and no stream.
  TensorLoanWriteGuard(TensorLoanWriteGuard&& other) noexcept
      : state_(std::exchange(other.state_, nullptr)), ops_(std::exchange(other.ops_, nullptr)) {}

  /// @brief Reject copying sole producer write authority.
  /// @param other Guard that would otherwise be copied.
  TensorLoanWriteGuard(const TensorLoanWriteGuard& other) = delete;

  /// @brief Reject copy assignment of sole producer write authority.
  /// @param other Guard that would otherwise replace this one.
  TensorLoanWriteGuard& operator=(const TensorLoanWriteGuard& other) = delete;

  /// @brief Reject move assignment so replacing a live writer cannot hide abandonment.
  /// @param other Guard that would otherwise replace this one.
  TensorLoanWriteGuard& operator=(TensorLoanWriteGuard&& other) = delete;

  /// @brief Abandon an uncommitted write phase without throwing.
  ~TensorLoanWriteGuard() noexcept;

  /// @brief Return the writable allocation base address.
  /// @return Writable storage while live, or `nullptr` after move or commit.
  [[nodiscard]] void* data() const noexcept;

  /// @brief Return the writable allocation base address cast to an element type.
  /// @tparam U Element type used only to interpret the pointer.
  /// @return The same address as \ref data, cast to `U*`.
  /// @pre `U` matches the admitted DLPack element representation, the allocation address satisfies
  /// `alignof(U)`, and \ref byte_size covers every access made through the returned pointer.
  /// @warning This unchecked convenience cast does not validate dtype, alignment, extent, or C++
  /// object lifetime.
  template <typename U>
  [[nodiscard]] U* data_as() const noexcept {
    return static_cast<U*>(data());
  }

  /// @brief Return the bounded writable byte span.
  /// @return Allocation byte span while live, or zero after move or commit.
  [[nodiscard]] std::size_t byte_size() const noexcept;

  /// @brief Return the borrowed producer CUDA stream for a device write.
  /// @return The stream value for a live device writer, or no value for a host, moved, or
  /// committed writer.
  [[nodiscard]] std::optional<::cuda::stream_ref> stream() const noexcept;

  /// @brief Commit producer readiness and consume this write guard.
  /// @return Success when readiness was recorded, or a checked state/capability error.
  /// @retval ErrorCode::kInvalidArgument This guard is moved from, committed, or otherwise invalid.
  /// @note Commit records bounded readiness after producer work; it does not publish, wait for the
  /// stream, retain the borrowed stream, or allocate a readiness resource.
  [[nodiscard]] expected<void, Error> commit() &&;

 private:
  TensorLoanWriteGuard(void* state, const channel_internal::TensorLoanWriteGuardOps* ops) noexcept
      : state_(state), ops_(ops) {}

  void* state_{};
  const channel_internal::TensorLoanWriteGuardOps* ops_{};

  friend class channel_internal::TensorPublishLoanAccess;
};

/// @brief Hold sole unpublished authority for one catalog-backed external Tensor slot.
///
/// The loan is move-constructible but neither copyable nor assignable. Every exposed layout view
/// borrows from the live loan. Moving, emitting, cancelling, or destroying the loan invalidates
/// those views. The public object contains no owning Tensor or DLPack value and exposes no
/// allocator, pool, slot, endpoint generation, CUDA event, HoloMQ object, or BufferHandle.
class HOLOSCAN_CORE_EXPORT TensorPublishLoan {
 public:
  /// @brief Transfer unpublished publication authority from another loan.
  /// @param other Loan whose authority becomes moved from.
  /// @post `other` exposes empty layout, unknown storage, and zero bytes.
  TensorPublishLoan(TensorPublishLoan&& other) noexcept
      : state_(std::exchange(other.state_, nullptr)), ops_(std::exchange(other.ops_, nullptr)) {}

  /// @brief Reject copying sole unpublished publication authority.
  /// @param other Loan that would otherwise be copied.
  TensorPublishLoan(const TensorPublishLoan& other) = delete;

  /// @brief Reject copy assignment of sole unpublished publication authority.
  /// @param other Loan that would otherwise replace this one.
  TensorPublishLoan& operator=(const TensorPublishLoan& other) = delete;

  /// @brief Reject move assignment so replacing a live loan cannot hide abandonment.
  /// @param other Loan that would otherwise replace this one.
  TensorPublishLoan& operator=(TensorPublishLoan&& other) = delete;

  /// @brief Abandon unpublished authority without throwing.
  ~TensorPublishLoan() noexcept;

  /// @brief Return the logical Tensor shape.
  /// @return Borrowed shape while live, or an empty view after move or cancellation.
  [[nodiscard]] TensorShapeView shape() const noexcept;

  /// @brief Return the optional explicit byte strides.
  /// @return Borrowed strides while live, or no explicit strides after move or cancellation.
  [[nodiscard]] TensorByteStridesView byte_strides() const noexcept;

  /// @brief Return the normalized DLPack element type.
  /// @return Element type while live, or a zero-initialized value after move or cancellation.
  [[nodiscard]] DLDataType dtype() const noexcept;

  /// @brief Return the endpoint-selected CUDA device identity.
  /// @return Resolved device, or an invalid identity for host-only, moved, or invalid authority.
  [[nodiscard]] DeviceId device_id() const noexcept;

  /// @brief Return the endpoint-selected physical memory classification.
  /// @return Memory kind while live, or \ref MemoryKind::kUnknown otherwise.
  [[nodiscard]] MemoryKind memory_kind() const noexcept;

  /// @brief Return the checked logical byte span.
  /// @return Accessible byte span while live, or zero after move or cancellation.
  [[nodiscard]] std::size_t byte_size() const noexcept;

  /// @brief Begin one CUDA-stream-ordered producer write phase.
  /// @param producer_stream Borrowed stream on which producer writes are ordered.
  /// @return Sole write guard, or a checked state, stream, or resource error.
  /// @note The returned guard borrows this loan and must be committed or destroyed first.
  [[nodiscard]] expected<TensorLoanWriteGuard, Error> write(::cuda::stream_ref producer_stream) &;

  /// @brief Run one CUDA fill callback and commit its write phase.
  /// @tparam Fill Callable accepting `const TensorLoanWriteGuard&` and returning either `void` or
  /// `expected<void, Error>`.
  /// @param producer_stream Borrowed stream on which producer writes are ordered.
  /// @param fill Callback invoked exactly once after acquiring write authority.
  /// @return Success after callback and commit, or the exact acquisition, callback, or commit
  /// error.
  /// @throws Any exception propagated by \p fill; the active write is abandoned during unwinding.
  /// @note A callback error abandons the active write and makes this loan nonpublishable. The
  /// callback must not retain the borrowed guard or its data pointer beyond the invocation.
  template <typename Fill>
    requires detail::TensorLoanFill<Fill&&, TensorLoanWriteGuard>
  // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
  [[nodiscard]] expected<void, Error> write_and_commit(::cuda::stream_ref producer_stream,
                                                       Fill&& fill) & {
    return detail::fill_and_commit_tensor_loan(write(producer_stream), std::forward<Fill>(fill));
  }

  /// @brief Begin one host producer write phase.
  /// @return Sole write guard, or \ref ErrorCode::kInvalidArgument when the endpoint-selected
  /// storage is not host accessible.
  [[nodiscard]] expected<TensorLoanWriteGuard, Error> write_host() &;

  /// @brief Run one host fill callback and commit its write phase.
  /// @tparam Fill Callable accepting `const TensorLoanWriteGuard&` and returning either `void` or
  /// `expected<void, Error>`.
  /// @param fill Callback invoked exactly once after acquiring host write authority.
  /// @return Success after callback and commit, or the exact acquisition, callback, or commit
  /// error.
  /// @throws Any exception propagated by \p fill; the active write is abandoned during unwinding.
  /// @note A callback error abandons the active write and makes this loan nonpublishable. The
  /// callback must not retain the borrowed guard or its data pointer beyond the invocation.
  template <typename Fill>
    requires detail::TensorLoanFill<Fill&&, TensorLoanWriteGuard>
  // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
  [[nodiscard]] expected<void, Error> write_host_and_commit(Fill&& fill) & {
    return detail::fill_and_commit_tensor_loan(write_host(), std::forward<Fill>(fill));
  }

  /// @brief Cancel unpublished authority and consume this loan.
  /// @return Success when bounded retirement begins, or a checked invalid-state error.
  /// @note Every call consumes the public loan on success or failure. Cancellation never publishes
  /// or returns an owning Tensor.
  [[nodiscard]] expected<void, Error> cancel() &&;

 private:
  TensorPublishLoan(void* state, const channel_internal::TensorPublishLoanOps* ops) noexcept
      : state_(state), ops_(ops) {}

  void* state_{};
  const channel_internal::TensorPublishLoanOps* ops_{};

  friend class channel_internal::TensorPublishLoanAccess;
};

static_assert(std::is_move_constructible_v<TensorLoanWriteGuard>);
static_assert(!std::is_copy_constructible_v<TensorLoanWriteGuard>);
static_assert(!std::is_move_assignable_v<TensorLoanWriteGuard>);
static_assert(std::is_nothrow_destructible_v<TensorLoanWriteGuard>);
static_assert(std::is_move_constructible_v<TensorPublishLoan>);
static_assert(!std::is_copy_constructible_v<TensorPublishLoan>);
static_assert(!std::is_move_assignable_v<TensorPublishLoan>);
static_assert(std::is_nothrow_destructible_v<TensorPublishLoan>);

}  // namespace holoscan
