/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Private cold-construction and immutable-resolution services for the public
// ClockKey/ClockRef facade. This header is not installed.

#pragma once

#include <compare>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/time/clock.hpp>
#include <holoscan/time/clock_registry.hpp>

namespace holoscan::detail {

/// Cold declaration flags used to freeze public ClockCapabilities.
struct ClockCapabilityOptions {
  bool can_read_now{false};
  bool can_wait_until{false};
  bool steady{false};
};

struct ClockRegistryState;

/// Immutable, generation-owning clock registry used by runtime internals.
class ClockRegistry;

/// A clock reference that holds its generation's clock for its own lifetime.
///
/// Runtime owners that read a clock on every activation, and that are destroyed before the registry
/// generation, use it instead of ClockRef to avoid locking a weak reference on every read. It
/// reports the same values and failures as ClockRef::now(), except that it cannot become stale.
class RetainedClock {
 public:
  RetainedClock() noexcept = default;
  RetainedClock(const RetainedClock&) noexcept = default;
  RetainedClock& operator=(const RetainedClock&) noexcept = default;

  // Transfer the owner and leave the source inert, keeping identity and ownership in step.
  RetainedClock(RetainedClock&& other) noexcept
      : id_(std::exchange(other.id_, ClockId{})),
        capabilities_(std::exchange(other.capabilities_, ClockCapabilities{})),
        state_(std::move(other.state_)) {}

  RetainedClock& operator=(RetainedClock&& other) noexcept {
    if (this != &other) {
      id_ = std::exchange(other.id_, ClockId{});
      capabilities_ = std::exchange(other.capabilities_, ClockCapabilities{});
      state_ = std::move(other.state_);
    }
    return *this;
  }

  [[nodiscard]] ClockId id() const noexcept { return id_; }

  /// @retval ErrorCode::kInvalidArgument This reference has no valid identity.
  /// @retval ErrorCode::kNotSupported Current-time reads were not admitted.
  /// @retval ErrorCode::kFailure The underlying clock threw while being read.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<TimePoint, Error> now() const noexcept;

 private:
  RetainedClock(ClockId id, ClockCapabilities capabilities,
                std::shared_ptr<const ClockRefState> state) noexcept
      : id_(id), capabilities_(capabilities), state_(std::move(state)) {}

  ClockId id_{};
  ClockCapabilities capabilities_{};
  std::shared_ptr<const ClockRefState> state_;

  friend class ClockRegistry;
};

class ClockRegistry {
 public:
  ClockRegistry() noexcept = default;

  /// Resolve a named or default cold key to a non-owning generation reference.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<ClockRef, Error> resolve(
      const ClockKey& key) const noexcept;

  /// Resolve one plan-local identity within this exact registry generation.
  ///
  /// Numeric identity alone is not portable across registries. Callers may use
  /// this overload only after retaining the enclosing immutable plan/catalog
  /// provenance that constructed this registry.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<ClockRef, Error> resolve(ClockId id) const noexcept;

  /// Return whether \p id names a clock in this registry generation.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT bool contains(ClockId id) const noexcept;

  /// Read one clock in this registry generation without minting a reference to it.
  ///
  /// The result and failures match `resolve(id)` followed by `ClockRef::now()`, except that the
  /// registry keeps the clock alive, so the reference cannot become stale. Runtime paths that
  /// already hold the registry use this to avoid updating the clock's shared reference count on
  /// every read.
  /// @retval ErrorCode::kNotFound \p id names no clock in this registry.
  /// @retval ErrorCode::kNotSupported The clock does not admit current-time reads.
  /// @retval ErrorCode::kFailure The underlying clock threw while being read.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<TimePoint, Error> now(ClockId id) const noexcept;

  /// Retain the clock that \p clock refers to for the lifetime of the result.
  ///
  /// An invalid \p clock yields an invalid RetainedClock.
  /// @retval ErrorCode::kStaleEndpoint The owning generation no longer exists.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<RetainedClock, Error> retain(
      const ClockRef& clock) noexcept;

  /// Wait on one clock already resolved in this registry generation.
  ///
  /// This is a scheduler-only operation. It neither exposes wait authority to
  /// Operators nor infers a clock from a raw timestamp.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> wait_until(
      ClockId id, std::int64_t timestamp_ns) const noexcept;

  /// Materialize one same-domain or explicitly converted timestamp.
  ///
  /// @pre \p source and \p target were resolved by this registry generation, or a caller validated
  /// their exact enclosing clock-catalog identity and rebound them to this registry before this
  /// call. Raw numeric collisions from another registry cannot be detected here.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<TimePoint, Error> convert(
      TimePoint source, ClockId target) const noexcept;

  /// Compare two exact coordinates in the requested target domain.
  ///
  /// @pre \p lhs, \p rhs, and \p target were resolved by this registry generation, or a caller
  /// validated their exact enclosing clock-catalog identity and rebound them to this registry
  /// before this call. Raw numeric collisions from another registry cannot be detected here.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<std::strong_ordering, Error> compare(
      TimePoint lhs, TimePoint rhs, ClockId target) const noexcept;

  /// Return whether a clock of this registry advances at exactly the steady clock's rate.
  ///
  /// Only an unscaled RealtimeClock qualifies. For such a clock a duration in its domain is the
  /// same steady-clock duration, so the Runtime may wait out its deadlines directly. Unknown ids
  /// and every other clock return false.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT bool advances_at_steady_rate(ClockId id) const noexcept;

 private:
  explicit ClockRegistry(std::shared_ptr<const ClockRegistryState> state) noexcept
      : state_(std::move(state)) {}

  std::shared_ptr<const ClockRegistryState> state_;

  friend class ClockRegistryBuilder;
};

/// Cold builder that validates declarations before publishing an immutable registry.
class ClockRegistryBuilder {
 public:
  ClockRegistryBuilder() = default;

  /// Add one owned live clock declaration and its fixed capability options.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> add_clock(
      const ClockKey& key, std::shared_ptr<Clock> clock, ClockCapabilityOptions options);

  /// Select one already-declared named clock as the default.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> set_default(const ClockKey& key);

  /// Add one explicit directed affine conversion.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> add_conversion(
      const ClockKey& source, const ClockKey& target, AffineClockConversion conversion);

  /// Validate and freeze deterministic ids, capabilities, and direct conversions.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<ClockRegistry, Error> build() const;

 private:
  [[nodiscard]] static bool advances_at_steady_rate(const Clock& clock) noexcept;

  struct ClockDeclaration {
    ClockKey key;
    std::shared_ptr<Clock> clock;
    ClockCapabilityOptions options;
  };

  struct ConversionDeclaration {
    ClockKey source;
    ClockKey target;
    AffineClockConversion conversion;
  };

  std::vector<ClockDeclaration> clocks_;
  std::vector<ConversionDeclaration> conversions_;
  std::optional<ClockKey> default_key_;
};

}  // namespace holoscan::detail
