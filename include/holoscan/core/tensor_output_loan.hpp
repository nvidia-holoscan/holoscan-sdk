/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <dlpack/dlpack.h>
#include <cuda/stream>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/memory/tensor_storage.hpp>

namespace holoscan {

/// @brief Identify one CUDA device without treating an omitted value as device zero.
///
/// Device identity is a value carried by tensor allocation and publication records. Negative
/// values are invalid and must be resolved before device-visible execution. The value is not a
/// process-global device authority and grants no access to a CUDA context.
struct DeviceId {
  std::int32_t value{-1};  ///< CUDA ordinal, or a negative value when unresolved.

  /// @brief Return whether this value contains a resolved CUDA ordinal.
  /// @return \c true when \ref value is nonnegative; otherwise \c false.
  [[nodiscard]] constexpr bool valid() const noexcept { return value >= 0; }

  /// @brief Compare resolved or unresolved device identities.
  /// @param lhs Left identity.
  /// @param rhs Right identity.
  /// @return \c true when both numeric representations match.
  friend constexpr bool operator==(DeviceId lhs, DeviceId rhs) noexcept = default;
};

/// @brief Classify the physical memory that backs a tensor allocation.
///
/// This concrete-storage classification does not select a pool, allocator, transfer policy, or
/// device. `kUnknown` is a diagnostic value and is not admitted for a published device-visible
/// tensor.
enum class MemoryKind : std::uint8_t {
  kUnknown,     ///< Storage has not yet been classified.
  kHost,        ///< Ordinary pageable host memory.
  kPinnedHost,  ///< Page-locked host memory visible to CUDA transfer operations.
  kCudaDevice,  ///< CUDA device memory.
  kCudaManaged  ///< CUDA managed memory.
};

/// @brief Describe one logical Tensor publication-loan request.
///
/// The shape and optional stride views are borrowed inputs. The caller must keep their backing
/// arrays alive only for the duration of `Output<Tensor>::allocate_tensor` or
/// `ChannelPublisher<Tensor>::loan_tensor`; an admitted implementation copies the bounded metadata
/// before the call returns. Physical placement remains frozen plan or endpoint state and is
/// deliberately absent from this request.
struct TensorLoanRequest {
  TensorShapeView shape;  ///< Logical extents; rank is checked against the output bound.
  DLDataType dtype{};     ///< DLPack element type for the requested storage.
  TensorByteStridesView byte_strides{std::nullopt};  ///< Optional explicit byte strides.
};

/// @brief Preserve accepted finite Tensor-pool bounds in compiler and Runtime records.
///
/// Application authors request runtime storage through TensorOutputSpec and TensorBounds.
/// This deep-owned record is the normalized physical-plan form after those contracts are admitted.
struct TensorPoolBounds {
  std::uint32_t max_rank{};     ///< Maximum admitted tensor rank.
  std::size_t max_byte_span{};  ///< Maximum byte span including explicit strides.
};

namespace detail {
struct TensorOutputWriteGuardOps;
struct TensorOutputLoanOps;
class TensorOutputLoanAccess;
}  // namespace detail

/// @brief Hold temporary write access to one unpublished tensor output allocation.
///
/// The guard is move-constructible but neither copyable nor assignable. Destruction without a
/// successful commit abandons the write phase without publishing the tensor. A guard borrows the
/// underlying loan and must not outlive it. It performs no implicit stream synchronization.
class HOLOSCAN_CORE_EXPORT TensorOutputWriteGuard {
 public:
  /// @brief Transfer the sole write phase from another guard.
  /// @param other Guard whose authority becomes moved-from.
  TensorOutputWriteGuard(TensorOutputWriteGuard&& other) noexcept
      : state_(std::exchange(other.state_, nullptr)),
        ops_(std::exchange(other.ops_, nullptr)),
        state_owner_(std::move(other.state_owner_)) {}

  /// @brief Reject copying sole tensor write authority.
  /// @param other Guard that would otherwise be copied.
  TensorOutputWriteGuard(const TensorOutputWriteGuard& other) = delete;

  /// @brief Reject copy assignment of sole tensor write authority.
  /// @param other Guard that would otherwise replace this one.
  TensorOutputWriteGuard& operator=(const TensorOutputWriteGuard& other) = delete;

  /// @brief Reject move assignment so a live guard cannot be replaced implicitly.
  /// @param other Guard that would otherwise replace this one.
  TensorOutputWriteGuard& operator=(TensorOutputWriteGuard&& other) = delete;

  /// @brief Abandon an uncommitted write phase without throwing.
  ~TensorOutputWriteGuard() noexcept;

  /// @brief Return the writable allocation base address.
  /// @return Writable pointer while this guard is live, or `nullptr` when moved from.
  [[nodiscard]] void* data() const noexcept;

  /// @brief Return the writable allocation base address cast to an element type.
  /// @tparam U Element type used only for pointer interpretation.
  /// @return The same address as \ref data, cast to `U*`.
  template <typename U>
  [[nodiscard]] U* data_as() const noexcept {
    return static_cast<U*>(data());
  }

  /// @brief Return the bounded writable byte span.
  /// @return Allocation byte span, or zero when moved from.
  [[nodiscard]] std::size_t byte_size() const noexcept;

  /// @brief Return the CUDA stream recorded for this write phase.
  /// @return Producer stream for a device write, or no value for a host write or moved-from guard.
  [[nodiscard]] std::optional<::cuda::stream_ref> stream() const noexcept;

  /// @brief Return a writable Tensor view over this active write phase.
  /// @return A Tensor sharing the provider-owned allocation, or an invalid Tensor for a moved-from
  /// guard or a backend that cannot expose a view.
  /// @note The view may keep the backing slot retained after commit and publication. Writes must
  /// still finish on the guard's declared stream before the guard is committed.
  [[nodiscard]] Tensor tensor() const;

  /// @brief Commit the write phase back to its unpublished tensor loan.
  /// @return Success when the written storage is ready for a later explicit publication attempt.
  /// @retval ErrorCode::kInvalidArgument This guard is moved from or invalid.
  /// @note Commit consumes the guard but does not publish the sample or assign a sequence.
  [[nodiscard]] expected<void, Error> commit() &&;

 private:
  TensorOutputWriteGuard(void* state, const detail::TensorOutputWriteGuardOps* ops,
                         std::shared_ptr<void> state_owner = {}) noexcept
      : state_(state), ops_(ops), state_owner_(std::move(state_owner)) {}

  void* state_{};
  const detail::TensorOutputWriteGuardOps* ops_{};
  // Keep private runtime authority storage alive without exposing its type or ownership model.
  std::shared_ptr<void> state_owner_;

  friend class detail::TensorOutputLoanAccess;
};

namespace detail {

template <typename Fill, typename Guard>
concept TensorLoanFill =
    std::invocable<Fill, const Guard&> &&
    (std::same_as<std::invoke_result_t<Fill, const Guard&>, void> ||
     std::same_as<std::invoke_result_t<Fill, const Guard&>, expected<void, Error>>);

template <typename Guard, typename Fill>
  requires TensorLoanFill<Fill&&, Guard>
[[nodiscard]] expected<void, Error> fill_and_commit_tensor_loan(expected<Guard, Error> writer,
                                                                Fill&& fill) {
  if (!writer) {
    return make_unexpected(std::move(writer).error());
  }
  if constexpr (std::same_as<std::invoke_result_t<Fill&&, const Guard&>, expected<void, Error>>) {
    auto filled = std::invoke(std::forward<Fill>(fill), std::as_const(*writer));
    if (!filled) {
      return make_unexpected(std::move(filled).error());
    }
  } else {
    std::invoke(std::forward<Fill>(fill), std::as_const(*writer));
  }
  return std::move(*writer).commit();
}

}  // namespace detail

/// @brief Hold sole unpublished authority for one plan-owned tensor output allocation.
///
/// The loan is move-constructible but neither copyable nor assignable. It exposes bounded metadata
/// and an explicit host or CUDA write phase; it never converts to an owning Tensor or DLPack
/// handle. Destruction abandons an unpublished live loan. Every terminal tensor publication or
/// cancellation attempt consumes the loan, including an error result.
class HOLOSCAN_CORE_EXPORT TensorOutputLoan {
 public:
  /// @brief Transfer unpublished authority from another loan.
  /// @param other Loan whose authority becomes moved-from.
  TensorOutputLoan(TensorOutputLoan&& other) noexcept
      : state_(std::exchange(other.state_, nullptr)),
        ops_(std::exchange(other.ops_, nullptr)),
        state_owner_(std::move(other.state_owner_)) {}

  /// @brief Reject copying sole unpublished tensor authority.
  /// @param other Loan that would otherwise be copied.
  TensorOutputLoan(const TensorOutputLoan& other) = delete;

  /// @brief Reject copy assignment of sole unpublished tensor authority.
  /// @param other Loan that would otherwise replace this one.
  TensorOutputLoan& operator=(const TensorOutputLoan& other) = delete;

  /// @brief Reject move assignment so a live loan cannot be replaced implicitly.
  /// @param other Loan that would otherwise replace this one.
  TensorOutputLoan& operator=(TensorOutputLoan&& other) = delete;

  /// @brief Abandon an unpublished live allocation without throwing.
  ~TensorOutputLoan() noexcept;

  /// @brief Return the allocation shape.
  /// @return Borrowed shape view valid while this loan remains live.
  [[nodiscard]] TensorShapeView shape() const noexcept;

  /// @brief Return the allocation byte strides.
  /// @return Borrowed explicit strides, or the empty/implicit-stride representation.
  [[nodiscard]] TensorByteStridesView byte_strides() const noexcept;

  /// @brief Return the allocation element type.
  /// @return DLPack element type frozen at allocation.
  [[nodiscard]] DLDataType dtype() const noexcept;

  /// @brief Return the resolved CUDA device identity.
  /// @return Device identity, or an invalid value for host-only/unresolved storage.
  [[nodiscard]] DeviceId device_id() const noexcept;

  /// @brief Return the concrete storage classification.
  /// @return Memory kind frozen by the compiled allocation plan.
  [[nodiscard]] MemoryKind memory_kind() const noexcept;

  /// @brief Return the bounded allocation byte span.
  /// @return Number of accessible bytes, including any explicit stride span.
  [[nodiscard]] std::size_t byte_size() const noexcept;

  /// @brief Begin a CUDA-stream-ordered write phase.
  /// @param producer_stream Stream on which producer writes will be ordered.
  /// @return A sole write guard, or a typed state/capability error.
  /// @note This operation borrows the loan; the returned guard must be committed or destroyed
  /// before the loan is otherwise used.
  [[nodiscard]] expected<TensorOutputWriteGuard, Error> write(::cuda::stream_ref producer_stream) &;

  /// @brief Run one CUDA fill callback and commit its write phase.
  /// @tparam Fill Callable accepting `const TensorOutputWriteGuard&` and returning either `void`
  /// or `expected<void, Error>`.
  /// @param producer_stream Stream on which producer writes will be ordered.
  /// @param fill Callback invoked exactly once after acquiring write authority.
  /// @return Success after callback and commit, or the exact acquisition, callback, or commit
  /// error.
  /// @throws Any exception propagated by \p fill; the active write is abandoned during unwinding.
  /// @note A callback error abandons the active write and makes this loan nonpublishable. The
  /// callback must not retain the borrowed guard or its data pointer beyond the invocation.
  template <typename Fill>
    requires detail::TensorLoanFill<Fill&&, TensorOutputWriteGuard>
  [[nodiscard]] expected<void, Error> write_and_commit(::cuda::stream_ref producer_stream,
                                                       Fill&& fill) & {
    return detail::fill_and_commit_tensor_loan(write(producer_stream), std::forward<Fill>(fill));
  }

  /// @brief Begin a host write phase.
  /// @return A sole write guard, or a typed state/capability error.
  /// @note Host write is admitted only for host-accessible placement frozen by the plan.
  [[nodiscard]] expected<TensorOutputWriteGuard, Error> write_host() &;

  /// @brief Run one host fill callback and commit its write phase.
  /// @tparam Fill Callable accepting `const TensorOutputWriteGuard&` and returning either `void`
  /// or `expected<void, Error>`.
  /// @param fill Callback invoked exactly once after acquiring host write authority.
  /// @return Success after callback and commit, or the exact acquisition, callback, or commit
  /// error.
  /// @throws Any exception propagated by \p fill; the active write is abandoned during unwinding.
  /// @note A callback error abandons the active write and makes this loan nonpublishable. The
  /// callback must not retain the borrowed guard or its data pointer beyond the invocation.
  template <typename Fill>
    requires detail::TensorLoanFill<Fill&&, TensorOutputWriteGuard>
  [[nodiscard]] expected<void, Error> write_host_and_commit(Fill&& fill) & {
    return detail::fill_and_commit_tensor_loan(write_host(), std::forward<Fill>(fill));
  }

  /// @brief Cancel unpublished tensor authority.
  /// @return Success when bounded retirement begins, or an error for invalid authority.
  /// @retval ErrorCode::kInvalidArgument The loan is invalid, or a write guard is still active.
  /// @note The terminal attempt consumes this loan for both success and failure. If a write guard
  /// is active, the allocation remains unavailable and tombstoned until that guard commits or is
  /// destroyed; the guard cannot publish and its commit returns `kInvalidArgument`.
  [[nodiscard]] expected<void, Error> cancel() &&;

 private:
  TensorOutputLoan(void* state, const detail::TensorOutputLoanOps* ops,
                   std::shared_ptr<void> state_owner = {}) noexcept
      : state_(state), ops_(ops), state_owner_(std::move(state_owner)) {}

  void* state_{};
  const detail::TensorOutputLoanOps* ops_{};
  // Keep private runtime authority storage alive through late destruction and checked failures.
  std::shared_ptr<void> state_owner_;

  friend class detail::TensorOutputLoanAccess;
};

}  // namespace holoscan
