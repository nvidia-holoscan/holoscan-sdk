/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

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

/// @brief Preserve accepted finite Tensor-pool bounds in compiler and Runtime records.
///
/// Application authors request runtime storage through TensorOutputSpec and TensorBounds.
/// This deep-owned record is the normalized physical-plan form after those contracts are admitted.
struct TensorPoolBounds {
  std::uint32_t max_rank{};     ///< Maximum admitted tensor rank.
  std::size_t max_byte_span{};  ///< Maximum byte span including explicit strides.
};

}  // namespace holoscan
