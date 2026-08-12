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

/// @brief Identify one frozen activation candidate within an Operator execution epoch.
///
/// Runtime assigns monotonically increasing sequences beginning at one within a nonzero execution
/// generation. A successfully published lifecycle restart advances the generation and restarts the
/// sequence. The complete identity also requires its enclosing RunSession and Operator identity;
/// identical numeric fields from different Operators or runs are unrelated.
///
/// \ref Tag supports equality and hashing only. Sequence and generation values are identity fields,
/// not temporal coordinates, so sorting by either field is unsupported. Runtime never wraps these
/// fields. The fixed-size record is standard-layout and trivially copyable. It owns no resources,
/// performs no allocation, and is safe to copy, hash, or compare concurrently.
struct Tag {
  std::uint64_t sequence{};    ///< Nonzero candidate sequence within `generation`.
  std::uint32_t generation{};  ///< Nonzero successfully published Operator execution generation.
  std::uint32_t reserved{};    ///< Record-version reservation; must be zero in EA1.

  /// @brief Return whether all fields form a supported activation identity.
  ///
  /// @return \c true when sequence and generation are nonzero and reserved is zero.
  /// @note \c {0,0,0} is the canonical invalid value. Other invalid combinations are malformed
  /// records rather than alternate invalid sentinels.
  [[nodiscard]] constexpr bool valid() const noexcept {
    return sequence != 0 && generation != 0 && reserved == 0;
  }

  /// @brief Compare activation identities for exact field equality.
  ///
  /// @param lhs Left tag.
  /// @param rhs Right tag.
  /// @return \c true when sequence, generation, and reserved fields match.
  friend constexpr bool operator==(Tag lhs, Tag rhs) noexcept = default;
};

/// @brief Represent the deterministic logical order of local activation candidates.
///
/// Unlike \ref Tag, \ref TagTime is an ordering coordinate. Valid EA1 records order
/// lexicographically by \ref logical_ns and then \ref microstep. The zero record is a valid
/// coordinate; absence, when required
/// by a containing contract, must use a separate validity mechanism. Reserved bits never
/// participate in ordering and must be zero for an EA1-supported record. Ordering remains a strict
/// weak relation for every bit pattern so generic ordered algorithms remain well-defined even when
/// a malformed or future-version record is inspected before validation.
///
/// `TagTime` is a fixed-size, standard-layout, trivially copyable record. It is local ordering
/// evidence, not physical or capture time. It owns no resources, performs no allocation, and is
/// safe to copy, hash, or compare concurrently.
struct TagTime {
  std::int64_t logical_ns{};  ///< Signed logical nanosecond coordinate.
  std::uint32_t microstep{};  ///< Deterministic order among candidates at `logical_ns`.
  std::uint32_t reserved{};   ///< Record-version reservation; must be zero in EA1.

  /// @brief Return whether this record uses the supported EA1 representation.
  ///
  /// @return \c true when \ref reserved is zero; otherwise \c false.
  [[nodiscard]] constexpr bool valid() const noexcept { return reserved == 0; }

  /// @brief Compare logical-order records for exact field equality.
  ///
  /// @param lhs Left logical coordinate.
  /// @param rhs Right logical coordinate.
  /// @return \c true when all fields match.
  friend constexpr bool operator==(TagTime lhs, TagTime rhs) noexcept = default;

  /// @brief Order two logical coordinates lexicographically.
  ///
  /// @param lhs Left logical coordinate.
  /// @param rhs Right logical coordinate.
  /// @return \c true when \p lhs has an earlier logical nanosecond than \p rhs, or when their
  /// logical nanoseconds match and \p lhs has the earlier microstep.
  /// @note Reserved fields never affect ordering. Call \ref valid separately before accepting a
  /// record as the supported EA1 representation.
  friend constexpr bool operator<(TagTime lhs, TagTime rhs) noexcept {
    return lhs.logical_ns < rhs.logical_ns ||
           (lhs.logical_ns == rhs.logical_ns && lhs.microstep < rhs.microstep);
  }
};

static_assert(sizeof(Tag) == 16, "Tag is part of the fixed EA1 record layout");
static_assert(offsetof(Tag, sequence) == 0);
static_assert(offsetof(Tag, generation) == 8);
static_assert(offsetof(Tag, reserved) == 12);
static_assert(std::is_standard_layout_v<Tag>);
static_assert(std::is_trivially_copyable_v<Tag>);
static_assert(sizeof(TagTime) == 16, "TagTime is part of the fixed EA1 record layout");
static_assert(offsetof(TagTime, logical_ns) == 0);
static_assert(offsetof(TagTime, microstep) == 8);
static_assert(offsetof(TagTime, reserved) == 12);
static_assert(std::is_standard_layout_v<TagTime>);
static_assert(std::is_trivially_copyable_v<TagTime>);

}  // namespace holoscan

namespace std {

/// @brief Hash a \ref holoscan::Tag by its complete fixed record.
template <>
struct hash<holoscan::Tag> {
  /// @brief Return a hash suitable for equality-based containers.
  ///
  /// @param tag Tag to hash.
  /// @return Combined hash of sequence, generation, and reserved fields.
  [[nodiscard]] std::size_t operator()(holoscan::Tag tag) const noexcept {
    std::size_t seed = std::hash<std::uint64_t>{}(tag.sequence);
    const auto combine = [&seed](std::size_t value) {
      seed ^= value + static_cast<std::size_t>(0x9e3779b9U) + (seed << 6U) + (seed >> 2U);
    };
    combine(std::hash<std::uint32_t>{}(tag.generation));
    combine(std::hash<std::uint32_t>{}(tag.reserved));
    return seed;
  }
};

/// @brief Hash a \ref holoscan::TagTime by its complete fixed record.
template <>
struct hash<holoscan::TagTime> {
  /// @brief Return a hash suitable for equality-based containers.
  ///
  /// @param time Logical coordinate to hash.
  /// @return Combined hash of logical nanoseconds, microstep, and reserved fields.
  [[nodiscard]] std::size_t operator()(holoscan::TagTime time) const noexcept {
    std::size_t seed = std::hash<std::int64_t>{}(time.logical_ns);
    const auto combine = [&seed](std::size_t value) {
      seed ^= value + static_cast<std::size_t>(0x9e3779b9U) + (seed << 6U) + (seed >> 2U);
    };
    combine(std::hash<std::uint32_t>{}(time.microstep));
    combine(std::hash<std::uint32_t>{}(time.reserved));
    return seed;
  }
};

}  // namespace std
