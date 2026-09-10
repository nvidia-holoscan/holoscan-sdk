/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/stream>

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

namespace holoscan::detail {

// Non-installed runtime construction contract. Runtime-owned state remains opaque to the public
// loan, while this table makes every lifetime and terminal transition explicit and testable.
struct TensorOutputWriteGuardOps {
  void (*abandon)(void*) noexcept;
  void* (*data)(const void*) noexcept;
  std::size_t (*byte_size)(const void*) noexcept;
  std::optional<::cuda::stream_ref> (*stream)(const void*) noexcept;
  expected<void, Error> (*commit)(void*) noexcept;
};

struct TensorOutputLoanOps {
  void (*abandon)(void*) noexcept;
  TensorShapeView (*shape)(const void*) noexcept;
  TensorByteStridesView (*byte_strides)(const void*) noexcept;
  DLDataType (*dtype)(const void*) noexcept;
  DeviceId (*device_id)(const void*) noexcept;
  MemoryKind (*memory_kind)(const void*) noexcept;
  std::size_t (*byte_size)(const void*) noexcept;
  expected<TensorOutputWriteGuard, Error> (*write)(void*, ::cuda::stream_ref) noexcept;
  expected<TensorOutputWriteGuard, Error> (*write_host)(void*) noexcept;
  expected<void, Error> (*cancel)(void*) noexcept;
};

class TensorOutputLoanAccess {
 public:
  [[nodiscard]] static TensorOutputLoan loan(void* state, const TensorOutputLoanOps* ops) noexcept {
    return TensorOutputLoan{state, ops};
  }

  [[nodiscard]] static TensorOutputLoan loan(
      void* state, const TensorOutputLoanOps* ops,
      std::shared_ptr<void> state_owner) noexcept {  // NOLINT(performance-unnecessary-value-param)
    return TensorOutputLoan{state, ops, std::move(state_owner)};
  }

  [[nodiscard]] static TensorOutputWriteGuard write_guard(
      void* state, const TensorOutputWriteGuardOps* ops) noexcept {
    return TensorOutputWriteGuard{state, ops};
  }

  [[nodiscard]] static TensorOutputWriteGuard write_guard(
      void* state, const TensorOutputWriteGuardOps* ops,
      std::shared_ptr<void> state_owner) noexcept {  // NOLINT(performance-unnecessary-value-param)
    return TensorOutputWriteGuard{state, ops, std::move(state_owner)};
  }

  /// Return whether a loan belongs to an exact private Runtime allocation state.
  ///
  /// This non-installed check lets a backend reject publication through the wrong output without
  /// exposing the allocation identity or an owning Tensor through the public facade.
  [[nodiscard]] static bool bound_to(const TensorOutputLoan& loan, const void* state) noexcept {
    return loan.state_ == state;
  }

  /// Return the private state only when the loan uses the expected operation table.
  ///
  /// Runtime uses this to recover exact allocation identity in O(1) while rejecting loans from a
  /// different backend before interpreting their opaque state.
  [[nodiscard]] static void* state_for_ops(const TensorOutputLoan& loan,
                                           const TensorOutputLoanOps* ops) noexcept {
    return loan.ops_ == ops ? loan.state_ : nullptr;
  }
};

}  // namespace holoscan::detail
