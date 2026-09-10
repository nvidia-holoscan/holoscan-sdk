/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/time/time_point.hpp>

namespace holoscan {

namespace detail {
class ClockRegistry;
class ClockRegistryBuilder;
struct ClockRefState;
}  // namespace detail

/// @brief Identify a clock declaration during cold graph and plan construction.
///
/// A key owns either a clock name or the distinct omitted/default identity. It is
/// safe to retain after the source string is destroyed. A key does not identify a
/// live clock generation and cannot be constructed from a \ref ClockId or
/// \ref ClockRef. Resolution to a live clock is a session operation.
///
/// Thread safety: distinct values may be used concurrently. Concurrent mutation
/// and access to the same value requires external synchronization. Creating or
/// copying a named key may allocate to own the name; queries do not allocate.
class ClockKey {
 public:
  /// @brief Create the omitted key that resolves to the graph or runtime default clock.
  ClockKey() noexcept = default;

  /// @brief Create a named cold clock key.
  ///
  /// @param name Clock declaration name to copy. An empty name remains a named
  /// key but is rejected when a registry declaration is validated.
  /// @throws std::bad_alloc if storage for the owned name cannot be allocated.
  /// @note This operation may allocate to own \p name.
  explicit ClockKey(std::string_view name) : name_(name), is_default_(false) {}

  /// @brief Create the omitted key that resolves to the default clock.
  /// @return The distinct default-clock key.
  /// @note This operation does not allocate.
  [[nodiscard]] static ClockKey default_clock() noexcept { return {}; }

  /// @brief Create a named cold clock key.
  /// @param name Clock declaration name to copy.
  /// @return A key that owns \p name.
  /// @throws std::bad_alloc if storage for the owned name cannot be allocated.
  /// @note This operation may allocate to own \p name.
  [[nodiscard]] static ClockKey named(std::string_view name) { return ClockKey{name}; }

  /// @brief Return whether this key denotes the graph or runtime default clock.
  /// @return \c true for the omitted/default identity; otherwise \c false.
  /// @note This operation does not allocate.
  [[nodiscard]] bool is_default() const noexcept { return is_default_; }

  /// @brief Return the owned declaration name.
  /// @return The declaration name, or an empty view for the default key.
  /// @note The view remains valid until this key is modified or destroyed.
  /// This operation does not allocate.
  [[nodiscard]] std::string_view name() const noexcept { return name_; }

  /// @brief Compare two cold keys by default identity and owned name.
  /// @param lhs Left key.
  /// @param rhs Right key.
  /// @return \c true when both keys have the same default identity and owned name.
  /// @note This operation does not allocate.
  friend bool operator==(const ClockKey& lhs, const ClockKey& rhs) noexcept = default;

 private:
  std::string name_;
  bool is_default_{true};
};

/// @brief Describe the fixed operations admitted for one resolved clock.
///
/// Capabilities are frozen when a session clock registry is built. They are
/// query-only: possessing this value does not grant mutation, waiting, or clock
/// lookup authority. A default-constructed value grants no capability.
///
/// The value is immutable to ordinary callers, safe to read concurrently, and
/// performs no allocation.
class ClockCapabilities {
 public:
  /// @brief Create a capability set that grants no operation.
  constexpr ClockCapabilities() noexcept = default;

  /// @brief Return whether \ref ClockRef::now may read the clock.
  /// @return \c true when current-time reads were admitted; otherwise \c false.
  [[nodiscard]] bool can_read_now() const noexcept { return (bits_ & kCanReadNow) != 0; }

  /// @brief Return whether scheduler infrastructure may wait on the clock.
  ///
  /// This capability is not exposed as an Operator-facing wait operation.
  /// @return \c true when scheduler infrastructure may wait; otherwise \c false.
  [[nodiscard]] bool can_wait_until() const noexcept { return (bits_ & kCanWaitUntil) != 0; }

  /// @brief Return whether the clock is admitted as monotonic within its generation.
  /// @return \c true when monotonic behavior was admitted; otherwise \c false.
  [[nodiscard]] bool steady() const noexcept { return (bits_ & kSteady) != 0; }

  /// @brief Return whether at least one direct outgoing conversion is frozen.
  /// @return \c true when an outgoing direct conversion exists; otherwise \c false.
  [[nodiscard]] bool has_outgoing_conversions() const noexcept {
    return (bits_ & kHasOutgoingConversions) != 0;
  }

  /// @brief Return whether at least one direct incoming conversion is frozen.
  /// @return \c true when an incoming direct conversion exists; otherwise \c false.
  [[nodiscard]] bool has_incoming_conversions() const noexcept {
    return (bits_ & kHasIncomingConversions) != 0;
  }

 private:
  static constexpr std::uint8_t kCanReadNow = 1U << 0U;
  static constexpr std::uint8_t kCanWaitUntil = 1U << 1U;
  static constexpr std::uint8_t kSteady = 1U << 2U;
  static constexpr std::uint8_t kHasOutgoingConversions = 1U << 3U;
  static constexpr std::uint8_t kHasIncomingConversions = 1U << 4U;

  explicit constexpr ClockCapabilities(std::uint8_t bits) noexcept : bits_(bits) {}

  std::uint8_t bits_{0};

  friend class detail::ClockRegistryBuilder;
};

/// @brief Borrow one resolved clock generation without extending its lifetime.
///
/// A reference carries a plan-local \ref ClockId assigned by one immutable registry generation and
/// frozen capabilities. It does not carry an externally comparable plan or clock-catalog identity;
/// callers retaining its identifier or a labelled time point must also retain that enclosing
/// provenance.
/// Copying it does not retain the registry or underlying clock. After the owning
/// registry generation is destroyed, live-state operations fail with
/// \ref ErrorCode::kStaleEndpoint while \ref id and \ref at_ns remain usable for
/// diagnostics and domain labeling. The reference performs no string lookup.
///
/// Thread safety: copying and read-only calls on independent references are safe.
/// A successful \ref now holds the generation alive for that call. The underlying
/// clock's own contract governs concurrent timestamp reads.
class ClockRef {
 public:
  /// @brief Create an inert reference with an invalid clock identity.
  /// @note This operation does not allocate or acquire generation ownership.
  ClockRef() noexcept = default;

  /// @brief Return the resolved plan-local clock identity.
  /// @return The identity retained by this reference, including after staleness.
  /// @note The result is meaningful only within this reference's originating clock-catalog scope.
  /// This operation does not allocate.
  [[nodiscard]] ClockId id() const noexcept { return id_; }

  /// @brief Return the capabilities frozen when this reference was resolved.
  /// @return The immutable capability set retained by this reference.
  /// @note This operation does not allocate.
  [[nodiscard]] ClockCapabilities capabilities() const noexcept { return capabilities_; }

  /// @brief Read and label the current timestamp from this clock generation.
  /// @return The current timestamp in this reference's domain on success.
  /// @retval ErrorCode::kInvalidArgument This reference has no valid identity.
  /// @retval ErrorCode::kStaleEndpoint The owning generation no longer exists.
  /// @retval ErrorCode::kNotSupported Current-time reads were not admitted.
  /// @retval ErrorCode::kFailure The underlying clock threw while being read.
  /// @note Failure precedence is invalid identity, stale generation, unsupported
  /// read capability, then underlying clock failure. The successful path
  /// performs no allocation or string lookup. The returned time point remains
  /// scoped to this reference's originating clock catalog.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<TimePoint, Error> now() const noexcept;

  /// @brief Label an existing integer timestamp with this reference's domain.
  /// @param timestamp_ns Timestamp in nanoseconds already expressed in this
  /// reference's domain.
  /// @return A time point carrying \p timestamp_ns and \ref id. The result is
  /// invalid when this reference has an invalid identity.
  /// @note This operation neither reads nor retains the underlying clock and
  /// remains available after the reference becomes stale. A result produced by a
  /// stale reference remains useful for diagnostics and domain labelling, but it
  /// is not authority to resolve the numeric identifier in another registry
  /// generation. This operation does not allocate.
  [[nodiscard]] TimePoint at_ns(std::int64_t timestamp_ns) const noexcept {
    return TimePoint{timestamp_ns, id_};
  }

 private:
  ClockRef(ClockId id, ClockCapabilities capabilities,
           std::weak_ptr<const detail::ClockRefState> state) noexcept
      : id_(id), capabilities_(capabilities), state_(std::move(state)) {}

  ClockId id_{};
  ClockCapabilities capabilities_{};
  std::weak_ptr<const detail::ClockRefState> state_;

  friend class detail::ClockRegistry;
};

/// @brief Represent a positive exact clock-rate ratio.
///
/// Registry construction accepts only a positive numerator and nonzero
/// denominator, then reduces the ratio to lowest terms. A default value is the
/// identity ratio. The public value is a cold authoring record; validation occurs
/// when the conversion is registered.
///
/// The value owns no resources, performs no allocation, and is safe to copy or
/// compare concurrently when callers do not mutate the same object.
struct ClockRateRatio {
  std::int64_t numerator{1};     ///< Positive scale numerator.
  std::uint64_t denominator{1};  ///< Nonzero scale denominator.

  /// @brief Compare exact authoring records for field equality.
  /// @param lhs Left ratio.
  /// @param rhs Right ratio.
  /// @return \c true when numerator and denominator fields match.
  friend bool operator==(const ClockRateRatio& lhs, const ClockRateRatio& rhs) noexcept = default;
};

/// @brief Select deterministic integer materialization for an affine clock conversion.
///
/// baseline admits only nearest rounding with halfway values rounded to the even
/// integer. The closed enum is stored in immutable plan records so runtime and
/// artifact consumers never infer rounding policy from the conversion formula.
enum class ClockRoundingMode : std::uint8_t {
  kNearestTiesToEven = 0,  ///< Round to the nearest integer and break ties toward even.
};

/// @brief Select overflow behavior for an affine clock conversion.
///
/// baseline admits only checked failure. Conversion never wraps, saturates, or
/// silently substitutes another timestamp.
enum class ClockOverflowPolicy : std::uint8_t {
  kError = 0,  ///< Return typed failure when the materialized coordinate is unrepresentable.
};

/// @brief Declare one immutable directed affine conversion between clock domains.
///
/// The exact converted coordinate is \c source*scale+offset_ns. Registry
/// construction canonicalizes the ratio and rejects invalid or conflicting
/// declarations. Integer materialization rounds to nearest with ties to even and
/// reports overflow; exact comparisons do not first round the coordinate.
///
/// This cold value owns no runtime clock, performs no allocation, and is safe to
/// copy or compare concurrently when callers do not mutate the same object.
struct AffineClockConversion {
  ClockRateRatio scale{};     ///< Positive exact rate ratio.
  std::int64_t offset_ns{0};  ///< Signed target-domain nanosecond offset.

  /// @brief Compare affine authoring records for exact field equality.
  /// @param lhs Left conversion.
  /// @param rhs Right conversion.
  /// @return \c true when scale and offset fields match exactly.
  friend bool operator==(const AffineClockConversion& lhs,
                         const AffineClockConversion& rhs) noexcept = default;
};

}  // namespace holoscan

namespace std {

/// @brief Hash a \ref holoscan::ClockKey by its complete cold identity.
template <>
struct hash<holoscan::ClockKey> {
  /// @brief Hash a cold clock key by its distinct default bit and owned name.
  /// @param key Key to hash.
  /// @return Hash of the default identity and complete owned name.
  /// @note This operation does not allocate.
  [[nodiscard]] std::size_t operator()(const holoscan::ClockKey& key) const noexcept {
    const std::size_t name_hash = std::hash<std::string_view>{}(key.name());
    const std::size_t default_hash = std::hash<bool>{}(key.is_default());
    return name_hash ^ (default_hash + 0x9e3779b9U + (name_hash << 6U) + (name_hash >> 2U));
  }
};

}  // namespace std
