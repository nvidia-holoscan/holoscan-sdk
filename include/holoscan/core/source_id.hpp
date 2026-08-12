/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>

namespace holoscan {

namespace detail {
struct SourceIdAccess;
}  // namespace detail

/// @brief Identify one logical activation-candidate source within an ExecutionPlan.
///
/// Compilation assigns nonzero values after canonical source ordering. The identifier is strong,
/// fixed-width, and plan-local. It covers both declared physical readiness sources and
/// compiler-synthesized clock or predicate sources, but does not encode source kind, an
/// authoring-object address, or a backend handle. Zero is the invalid or unbound value. `SourceId`
/// supports equality and hashing only because its numeric representation carries no semantic order.
///
/// Numeric values may be reused by different plans. Equality therefore has meaning only when both
/// operands are known to belong to the same \ref ExecutionPlan and plan/catalog generation.
/// Runtime, trace, and transport records that retain a `SourceId` beyond an immediate plan-local
/// lookup must also retain the originating plan or catalog identity. A value from a replaced,
/// foreign, or stale plan must not be looked up in the current plan even when its numeric
/// representation collides.
///
/// Construction and access to the numeric representation are restricted to Core implementation
/// helpers so callers cannot manufacture a live source authority. Copies remain ordinary values;
/// lifecycle generation and adapter authority are represented separately. The value owns no
/// resources, performs no allocation, and is safe to copy, hash, or compare concurrently.
class SourceId {
 public:
  /// @brief Construct an invalid source identifier.
  constexpr SourceId() noexcept = default;

  /// @brief Return whether this identifier contains a compiler-assigned nonzero value.
  ///
  /// @return \c true for a nonzero identifier; otherwise \c false.
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value_ != 0; }

  /// @brief Compare source identifiers from the same plan/catalog generation for equality.
  ///
  /// @param lhs Left source identifier.
  /// @param rhs Right source identifier.
  /// @return \c true when both identifiers have the same numeric representation.
  /// @pre Both identifiers originate from the same \ref ExecutionPlan and plan/catalog generation,
  /// or both are invalid values being compared only for invalidity.
  friend constexpr bool operator==(SourceId lhs, SourceId rhs) noexcept = default;

 private:
  friend struct detail::SourceIdAccess;
  friend struct std::hash<SourceId>;

  explicit constexpr SourceId(std::uint32_t value) noexcept : value_(value) {}

  std::uint32_t value_{};
};

static_assert(sizeof(SourceId) == 4, "SourceId is part of the fixed EA1 record layout");
static_assert(std::is_standard_layout_v<SourceId>);
static_assert(std::is_trivially_copyable_v<SourceId>);

}  // namespace holoscan

namespace std {

/// @brief Hash the numeric portion of a plan-local \ref holoscan::SourceId.
///
/// Containers spanning plans must include the originating plan/catalog generation in their key;
/// this specialization alone cannot distinguish numerically colliding identifiers from two plans.
template <>
struct hash<holoscan::SourceId> {
  /// @brief Return a hash suitable for equality-based containers.
  ///
  /// @param id Source identifier to hash.
  /// @return Hash of the private fixed-width representation.
  /// @pre Every nonzero identifier in the hashed key domain belongs to the same plan/catalog
  /// generation, unless that generation is represented separately in the enclosing key.
  [[nodiscard]] std::size_t operator()(holoscan::SourceId id) const noexcept {
    return std::hash<std::uint32_t>{}(id.value_);
  }
};

}  // namespace std
