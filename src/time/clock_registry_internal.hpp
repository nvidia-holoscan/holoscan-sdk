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
