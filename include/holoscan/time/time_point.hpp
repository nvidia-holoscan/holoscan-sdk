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

/// @brief Identify one resolved clock domain within an admitted execution plan.
///
/// A nonzero value identifies a clock domain assigned during compilation or admission. The numeric
/// value has no temporal ordering: consumers may compare or hash clock identifiers, but must use an
/// admitted conversion to compare coordinates from different domains. Zero represents an invalid
/// or unresolved domain.
///
/// \ref ClockId is a fixed-width, standard-layout, trivially copyable value suitable for bounded
/// runtime records and serialization. Its numeric assignment is plan-local and is not a durable
/// process- or deployment-global identity. It carries no originating plan, clock-catalog, or
/// registry-generation identity. Numeric equality and hashing have domain meaning only after the
/// caller establishes that the values belong to the same immutable clock-catalog scope. A foreign
/// or stale numeric value must not be interpreted through another registry merely because its
/// number matches a local entry.
///
/// Persisted, trace, and transport records must carry the originating scope directly or inherit it
/// unambiguously from a validated immutable plan, trace header, or catalog-bound endpoint. That
/// scope must be validated before a numeric identifier is resolved or rebound locally. The value
/// owns no resources, performs no allocation, and is safe to copy, hash, or compare concurrently.
struct ClockId {
  std::uint64_t value{};  ///< Nonzero plan-local clock-domain identifier, or zero when invalid.

  /// @brief Return whether this value identifies a resolved clock domain.
  ///
  /// @return \c true when \ref value is nonzero; otherwise \c false.
  /// @note A successful result establishes only representation validity. It does not prove that
  /// the identifier belongs to a particular plan, catalog, or registry generation.
  [[nodiscard]] constexpr bool valid() const noexcept { return value != 0; }

  /// @brief Compare clock-domain identities for equality.
  ///
  /// @param lhs Left clock identifier.
  /// @param rhs Right clock identifier.
  /// @return `true` when both values identify the same numeric domain.
  /// @note Equality is numeric field equality only and does not establish common provenance.
  friend constexpr bool operator==(ClockId lhs, ClockId rhs) noexcept = default;
};

/// @brief Represent a nanosecond coordinate in one resolved clock domain.
///
/// A valid value combines any signed nanosecond coordinate, including zero or a negative value,
/// with a nonzero `ClockId`. The default value `{0, ClockId{0}}` is the canonical absent time
/// representation. Timestamp zero in a valid clock domain is a real coordinate and is not absent.
///
/// Equality compares both fields. `TimePoint` deliberately has no numeric ordering because clock
/// domains are not implicitly comparable; use same-domain checks or an admitted explicit
/// conversion before temporal comparison. A `TimePoint` carries no originating plan,
/// clock-catalog, or registry-generation identity. Field equality has temporal meaning only after
/// both values' common clock-catalog scope has been established. This type is fixed-size,
/// standard-layout, and trivially copyable. It owns no resources, performs no allocation, and is
/// safe to copy, hash, or compare concurrently.
struct TimePoint {
  std::int64_t timestamp_ns{};  ///< Signed nanosecond coordinate in `clock_id`.
  ClockId clock_id{};           ///< Resolved domain for `timestamp_ns`, or zero when absent.

  /// @brief Return whether this value contains a nonzero clock-domain identifier.
  ///
  /// @return \c true when \ref clock_id is nonzero; otherwise \c false.
  /// @note A successful result establishes only representation validity. It does not prove that
  /// the identifier belongs to a particular plan, catalog, or registry generation, nor that this
  /// value is comparable with another time point.
  [[nodiscard]] constexpr bool valid() const noexcept { return clock_id.valid(); }

  /// @brief Compare time-point records for exact field equality.
  ///
  /// @param lhs Left time point.
  /// @param rhs Right time point.
  /// @return \c true when both the timestamp and clock identifier match.
  /// @note Equality is field equality only. It does not establish common provenance or that values
  /// from different clock domains are comparable.
  friend constexpr bool operator==(TimePoint lhs, TimePoint rhs) noexcept = default;
};

static_assert(sizeof(ClockId) == 8, "ClockId is part of the fixed EA1 record layout");
static_assert(offsetof(ClockId, value) == 0);
static_assert(std::is_standard_layout_v<ClockId>);
static_assert(std::is_trivially_copyable_v<ClockId>);
static_assert(sizeof(TimePoint) == 16, "TimePoint is part of the fixed EA1 record layout");
static_assert(offsetof(TimePoint, timestamp_ns) == 0);
static_assert(offsetof(TimePoint, clock_id) == 8);
static_assert(std::is_standard_layout_v<TimePoint>);
static_assert(std::is_trivially_copyable_v<TimePoint>);

}  // namespace holoscan

namespace std {

/// @brief Hash a \ref holoscan::ClockId by its plan-local numeric identity.
template <>
struct hash<holoscan::ClockId> {
  /// @brief Return a hash suitable for equality-based containers.
  ///
  /// @param id Clock identifier to hash.
  /// @return Hash of the complete identifier value.
  [[nodiscard]] std::size_t operator()(holoscan::ClockId id) const noexcept {
    return std::hash<std::uint64_t>{}(id.value);
  }
};

/// @brief Hash a \ref holoscan::TimePoint by its complete record identity.
template <>
struct hash<holoscan::TimePoint> {
  /// @brief Return a hash suitable for equality-based containers.
  ///
  /// @param point Time point to hash.
  /// @return Combined hash of the timestamp and clock identifier.
  [[nodiscard]] std::size_t operator()(holoscan::TimePoint point) const noexcept {
    std::size_t seed = std::hash<std::int64_t>{}(point.timestamp_ns);
    const std::size_t domain = std::hash<holoscan::ClockId>{}(point.clock_id);
    seed ^= domain + static_cast<std::size_t>(0x9e3779b9U) + (seed << 6U) + (seed >> 2U);
    return seed;
  }
};

}  // namespace std
