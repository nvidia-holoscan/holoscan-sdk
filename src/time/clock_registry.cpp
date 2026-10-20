/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/time/clock_registry.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <algorithm>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "clock_registry_internal.hpp"

namespace holoscan::detail {

namespace {

[[nodiscard]] unexpected<Error> clock_error(const ErrorCode code) {
  return make_unexpected(Error{.code = code, .message = {}});
}

[[nodiscard]] unexpected<Error> clock_error(const ErrorCode code, std::string message) {
  return make_unexpected(Error{.code = code, .message = std::move(message)});
}

struct UInt128 {
  std::uint64_t high{0};
  std::uint64_t low{0};
};

[[nodiscard]] constexpr bool is_zero(const UInt128 value) noexcept {
  return value.high == 0 && value.low == 0;
}

[[nodiscard]] constexpr int compare_u128(const UInt128 lhs, const UInt128 rhs) noexcept {
  if (lhs.high != rhs.high) {
    return lhs.high < rhs.high ? -1 : 1;
  }
  if (lhs.low != rhs.low) {
    return lhs.low < rhs.low ? -1 : 1;
  }
  return 0;
}

[[nodiscard]] constexpr UInt128 subtract_u128(const UInt128 lhs, const UInt128 rhs) noexcept {
  const std::uint64_t borrow = lhs.low < rhs.low ? 1U : 0U;
  return UInt128{.high = lhs.high - rhs.high - borrow, .low = lhs.low - rhs.low};
}

[[nodiscard]] constexpr bool add_u128(const UInt128 lhs, const UInt128 rhs,
                                      UInt128& result) noexcept {
  result.low = lhs.low + rhs.low;
  const std::uint64_t carry = result.low < lhs.low ? 1U : 0U;
  result.high = lhs.high + rhs.high + carry;
  return result.high < lhs.high || (result.high == lhs.high && carry != 0 && rhs.high != 0);
}

// Split both operands into 32-bit limbs. This is the portable C++20 equivalent
// of a 64x64 -> 128 multiply and avoids compiler-specific integer extensions.
[[nodiscard]] constexpr UInt128 multiply_u64(const std::uint64_t lhs,
                                             const std::uint64_t rhs) noexcept {
  constexpr std::uint64_t kLow32 = 0xFFFF'FFFFULL;
  const std::uint64_t lhs_low = lhs & kLow32;
  const std::uint64_t lhs_high = lhs >> 32U;
  const std::uint64_t rhs_low = rhs & kLow32;
  const std::uint64_t rhs_high = rhs >> 32U;

  std::uint64_t work = lhs_low * rhs_low;
  const std::uint64_t result_low_low = work & kLow32;
  std::uint64_t carry = work >> 32U;

  work = lhs_high * rhs_low + carry;
  const std::uint64_t middle_low = work & kLow32;
  const std::uint64_t middle_high = work >> 32U;

  work = lhs_low * rhs_high + middle_low;
  carry = work >> 32U;

  return UInt128{
      .high = lhs_high * rhs_high + middle_high + carry,
      .low = (work << 32U) + result_low_low,
  };
}

[[nodiscard]] constexpr bool bit_at(const UInt128 value, const unsigned index) noexcept {
  return index < 64U ? ((value.low >> index) & 1U) != 0 : ((value.high >> (index - 64U)) & 1U) != 0;
}

constexpr void set_bit(UInt128& value, const unsigned index) noexcept {
  if (index < 64U) {
    value.low |= std::uint64_t{1} << index;
  } else {
    value.high |= std::uint64_t{1} << (index - 64U);
  }
}

[[nodiscard]] constexpr UInt128 shift_left_one(const UInt128 value) noexcept {
  return UInt128{.high = (value.high << 1U) | (value.low >> 63U), .low = value.low << 1U};
}

struct DivisionResult {
  UInt128 quotient;
  std::uint64_t remainder{0};
};

[[nodiscard]] constexpr DivisionResult divide_u128_u64(const UInt128 dividend,
                                                       const std::uint64_t divisor) noexcept {
  UInt128 quotient;
  UInt128 remainder;
  const UInt128 wide_divisor{.high = 0, .low = divisor};

  // Restoring division keeps a 65-bit intermediate remainder in UInt128. It is
  // bounded, deterministic, and identical on GCC, Clang, clang-cl, and MSVC.
  for (unsigned bit = 128U; bit > 0U; --bit) {
    remainder = shift_left_one(remainder);
    if (bit_at(dividend, bit - 1U)) {
      remainder.low |= 1U;
    }
    if (compare_u128(remainder, wide_divisor) >= 0) {
      remainder = subtract_u128(remainder, wide_divisor);
      set_bit(quotient, bit - 1U);
    }
  }

  return DivisionResult{.quotient = quotient, .remainder = remainder.low};
}

[[nodiscard]] constexpr std::uint64_t signed_magnitude(const std::int64_t value) noexcept {
  if (value >= 0) {
    return static_cast<std::uint64_t>(value);
  }
  // Avoid negating INT64_MIN in the signed domain.
  return static_cast<std::uint64_t>(-(value + 1)) + 1U;
}

struct SignedUInt128 {
  bool negative{false};
  UInt128 magnitude;
};

[[nodiscard]] constexpr SignedUInt128 signed_product(const std::int64_t lhs,
                                                     const std::uint64_t rhs) noexcept {
  SignedUInt128 result{.negative = lhs < 0, .magnitude = multiply_u64(signed_magnitude(lhs), rhs)};
  if (is_zero(result.magnitude)) {
    result.negative = false;
  }
  return result;
}

[[nodiscard]] constexpr bool add_signed_magnitudes(const SignedUInt128 lhs, const SignedUInt128 rhs,
                                                   SignedUInt128& result) noexcept {
  if (lhs.negative == rhs.negative) {
    result.negative = lhs.negative;
    const bool overflow = add_u128(lhs.magnitude, rhs.magnitude, result.magnitude);
    if (is_zero(result.magnitude)) {
      result.negative = false;
    }
    return overflow;
  }

  const int ordering = compare_u128(lhs.magnitude, rhs.magnitude);
  if (ordering == 0) {
    result = {};
  } else if (ordering > 0) {
    result = SignedUInt128{.negative = lhs.negative,
                           .magnitude = subtract_u128(lhs.magnitude, rhs.magnitude)};
  } else {
    result = SignedUInt128{.negative = rhs.negative,
                           .magnitude = subtract_u128(rhs.magnitude, lhs.magnitude)};
  }
  return false;
}

struct ExactCoordinate {
  SignedUInt128 numerator;
  std::uint64_t denominator{1};
};

struct ClockConversionRecord {
  ClockId source;
  ClockId target;
  AffineClockConversion conversion;
};

[[nodiscard]] constexpr bool conversion_record_less(const ClockConversionRecord& lhs,
                                                    const ClockConversionRecord& rhs) noexcept {
  return lhs.source.value < rhs.source.value ||
         (lhs.source.value == rhs.source.value && lhs.target.value < rhs.target.value);
}

[[nodiscard]] expected<AffineClockConversion, Error> canonicalize_conversion(
    const AffineClockConversion conversion) {
  if (conversion.scale.numerator <= 0 || conversion.scale.denominator == 0) {
    return clock_error(ErrorCode::kInvalidArgument);
  }
  const auto numerator = static_cast<std::uint64_t>(conversion.scale.numerator);
  const std::uint64_t divisor = std::gcd(numerator, conversion.scale.denominator);
  return AffineClockConversion{
      .scale = {.numerator = static_cast<std::int64_t>(numerator / divisor),
                .denominator = conversion.scale.denominator / divisor},
      .offset_ns = conversion.offset_ns,
  };
}

[[nodiscard]] constexpr bool are_exact_reciprocals(const AffineClockConversion forward,
                                                   const AffineClockConversion reverse) noexcept {
  const auto forward_numerator = static_cast<std::uint64_t>(forward.scale.numerator);
  const auto reverse_numerator = static_cast<std::uint64_t>(reverse.scale.numerator);

  // Both positive ratios are already reduced. Equality to an exact inverse can
  // therefore be checked limb-free, and an inverse numerator that cannot fit
  // the public signed field is correctly unrepresentable.
  if (reverse_numerator != forward.scale.denominator ||
      reverse.scale.denominator != forward_numerator) {
    return false;
  }

  // If y = x*a/b + o and x = y*b/a + p, the reciprocal offsets are
  // consistent exactly when b*o + a*p == 0. Signed magnitudes avoid overflow
  // and preserve INT64_MIN without compiler-specific extended integers.
  const SignedUInt128 scaled_forward_offset =
      signed_product(forward.offset_ns, forward.scale.denominator);
  const SignedUInt128 scaled_reverse_offset = signed_product(reverse.offset_ns, forward_numerator);
  SignedUInt128 sum;
  return !add_signed_magnitudes(scaled_forward_offset, scaled_reverse_offset, sum) &&
         is_zero(sum.magnitude);
}

[[nodiscard]] int compare_unsigned_rationals(const UInt128 lhs_numerator,
                                             const std::uint64_t lhs_denominator,
                                             const UInt128 rhs_numerator,
                                             const std::uint64_t rhs_denominator) noexcept {
  const DivisionResult lhs = divide_u128_u64(lhs_numerator, lhs_denominator);
  const DivisionResult rhs = divide_u128_u64(rhs_numerator, rhs_denominator);
  const int integral_ordering = compare_u128(lhs.quotient, rhs.quotient);
  if (integral_ordering != 0) {
    return integral_ordering;
  }

  // Once integral quotients agree, each remainder is 64-bit. Cross products
  // therefore fit exactly in the portable UInt128 representation.
  return compare_u128(multiply_u64(lhs.remainder, rhs_denominator),
                      multiply_u64(rhs.remainder, lhs_denominator));
}

[[nodiscard]] std::strong_ordering compare_exact(const ExactCoordinate& lhs,
                                                 const ExactCoordinate& rhs) noexcept {
  const bool lhs_zero = is_zero(lhs.numerator.magnitude);
  const bool rhs_zero = is_zero(rhs.numerator.magnitude);
  const bool lhs_negative = lhs.numerator.negative && !lhs_zero;
  const bool rhs_negative = rhs.numerator.negative && !rhs_zero;
  if (lhs_negative != rhs_negative) {
    return lhs_negative ? std::strong_ordering::less : std::strong_ordering::greater;
  }

  const int magnitude_ordering = compare_unsigned_rationals(
      lhs.numerator.magnitude, lhs.denominator, rhs.numerator.magnitude, rhs.denominator);
  if (magnitude_ordering == 0) {
    return std::strong_ordering::equal;
  }
  if (lhs_negative) {
    return magnitude_ordering < 0 ? std::strong_ordering::greater : std::strong_ordering::less;
  }
  return magnitude_ordering < 0 ? std::strong_ordering::less : std::strong_ordering::greater;
}

[[nodiscard]] expected<std::int64_t, Error> materialize_exact(
    const ExactCoordinate& coordinate) noexcept {
  DivisionResult divided = divide_u128_u64(coordinate.numerator.magnitude, coordinate.denominator);

  // Compare remainder with exactly half the denominator without computing
  // 2*remainder, which could overflow uint64_t.
  const std::uint64_t complement = coordinate.denominator - divided.remainder;
  const bool above_half = divided.remainder > complement;
  const bool tie = divided.remainder == complement;
  if (above_half || (tie && (divided.quotient.low & 1U) != 0)) {
    UInt128 rounded;
    if (add_u128(divided.quotient, UInt128{.high = 0, .low = 1}, rounded)) {
      return clock_error(ErrorCode::kInvalidArgument);
    }
    divided.quotient = rounded;
  }

  if (divided.quotient.high != 0) {
    return clock_error(ErrorCode::kInvalidArgument);
  }
  constexpr std::uint64_t kPositiveLimit =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  constexpr std::uint64_t kNegativeLimit = kPositiveLimit + 1U;
  if (!coordinate.numerator.negative) {
    if (divided.quotient.low > kPositiveLimit) {
      return clock_error(ErrorCode::kInvalidArgument);
    }
    return static_cast<std::int64_t>(divided.quotient.low);
  }
  if (divided.quotient.low > kNegativeLimit) {
    return clock_error(ErrorCode::kInvalidArgument);
  }
  if (divided.quotient.low == kNegativeLimit) {
    return std::numeric_limits<std::int64_t>::min();
  }
  return -static_cast<std::int64_t>(divided.quotient.low);
}

}  // namespace

struct ClockRefState {
  std::shared_ptr<Clock> clock;
};

struct ClockRegistryEntry {
  ClockKey key;
  ClockId id;
  ClockCapabilities capabilities;
  std::shared_ptr<const ClockRefState> live_state;
  bool steady_rate{};
};

struct ClockRegistryState {
  std::vector<ClockRegistryEntry> entries;
  std::vector<ClockConversionRecord> conversions;
  ClockId default_id;
};

namespace {

[[nodiscard]] const ClockRegistryEntry* find_entry(const ClockRegistryState& state,
                                                   const ClockId id) noexcept {
  if (!id.valid() || id.value > state.entries.size()) {
    return nullptr;
  }
  const ClockRegistryEntry& entry = state.entries[static_cast<std::size_t>(id.value - 1U)];
  return entry.id == id ? &entry : nullptr;
}

[[nodiscard]] const ClockRegistryEntry* find_named_entry(const ClockRegistryState& state,
                                                         const std::string_view name) noexcept {
  const auto found =
      std::lower_bound(state.entries.begin(),
                       state.entries.end(),
                       name,
                       [](const ClockRegistryEntry& entry, const std::string_view candidate) {
                         return entry.key.name() < candidate;
                       });
  return found != state.entries.end() && found->key.name() == name ? &*found : nullptr;
}

[[nodiscard]] const ClockConversionRecord* find_conversion(const ClockRegistryState& state,
                                                           const ClockId source,
                                                           const ClockId target) noexcept {
  const ClockConversionRecord requested{.source = source, .target = target, .conversion = {}};
  const auto found = std::lower_bound(
      state.conversions.begin(), state.conversions.end(), requested, conversion_record_less);
  return found != state.conversions.end() && found->source == source && found->target == target
             ? &*found
             : nullptr;
}

[[nodiscard]] expected<ExactCoordinate, Error> project_exact(const ClockRegistryState& state,
                                                             const TimePoint point,
                                                             const ClockId target) noexcept {
  if (!point.valid() || !target.valid()) {
    return clock_error(ErrorCode::kInvalidArgument);
  }
  if (find_entry(state, point.clock_id) == nullptr || find_entry(state, target) == nullptr) {
    return clock_error(ErrorCode::kNotFound);
  }
  if (point.clock_id == target) {
    return ExactCoordinate{
        .numerator = signed_product(point.timestamp_ns, 1),
        .denominator = 1,
    };
  }

  const ClockConversionRecord* conversion = find_conversion(state, point.clock_id, target);
  if (conversion == nullptr) {
    return clock_error(ErrorCode::kNotFound);
  }

  const SignedUInt128 scaled = signed_product(
      point.timestamp_ns, static_cast<std::uint64_t>(conversion->conversion.scale.numerator));
  const SignedUInt128 offset =
      signed_product(conversion->conversion.offset_ns, conversion->conversion.scale.denominator);
  SignedUInt128 numerator;
  if (add_signed_magnitudes(scaled, offset, numerator)) {
    return clock_error(ErrorCode::kInvalidArgument);
  }
  return ExactCoordinate{.numerator = numerator,
                         .denominator = conversion->conversion.scale.denominator};
}

[[nodiscard]] bool has_alternate_return_path(const std::vector<std::vector<std::size_t>>& adjacency,
                                             const std::size_t source, const std::size_t target,
                                             std::vector<bool>& visited,
                                             std::vector<std::size_t>& queue) {
  std::fill(visited.begin(), visited.end(), false);
  queue.clear();
  visited[target] = true;
  queue.push_back(target);
  for (std::size_t cursor = 0; cursor < queue.size(); ++cursor) {
    const std::size_t current = queue[cursor];
    for (const std::size_t next : adjacency[current]) {
      if (current == target && next == source) {
        continue;
      }
      if (next == source) {
        return true;
      }
      if (!visited[next]) {
        visited[next] = true;
        queue.push_back(next);
      }
    }
  }
  return false;
}

[[nodiscard]] bool contains_larger_cycle(const std::vector<std::vector<std::size_t>>& adjacency) {
  std::vector<bool> visited(adjacency.size());
  std::vector<std::size_t> queue;
  queue.reserve(adjacency.size());

  // A cycle with at least three distinct domains exists exactly when an edge
  // u->v has an alternate simple path v..u after the direct reciprocal v->u is
  // ignored. This preserves explicitly authored two-domain reciprocal pairs.
  for (std::size_t source = 0; source < adjacency.size(); ++source) {
    for (const std::size_t target : adjacency[source]) {
      if (has_alternate_return_path(adjacency, source, target, visited, queue)) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace

expected<void, Error> ClockRegistryBuilder::add_clock(const ClockKey& key,
                                                      std::shared_ptr<Clock> clock,
                                                      const ClockCapabilityOptions options) {
  if (key.is_default() || key.name().empty() || clock == nullptr) {
    return clock_error(ErrorCode::kInvalidArgument,
                       "clock declaration requires a nonempty named key and a non-null clock");
  }
  const auto duplicate = std::find_if(clocks_.begin(), clocks_.end(), [&](const auto& declaration) {
    return declaration.key == key;
  });
  if (duplicate != clocks_.end()) {
    return clock_error(ErrorCode::kDuplicateName,
                       "clock declaration name is already present in this builder");
  }
  clocks_.push_back(ClockDeclaration{.key = key, .clock = std::move(clock), .options = options});
  return {};
}

bool ClockRegistryBuilder::advances_at_steady_rate(const Clock& clock) noexcept {
  const auto* realtime = dynamic_cast<const RealtimeClock*>(&clock);
  return realtime != nullptr && realtime->time_scale_ == 1.0;
}

expected<void, Error> ClockRegistryBuilder::set_default(const ClockKey& key) {
  if (key.is_default() || key.name().empty()) {
    return clock_error(ErrorCode::kInvalidArgument,
                       "default clock selection requires a nonempty named key");
  }
  const auto found = std::find_if(clocks_.begin(), clocks_.end(), [&](const auto& declaration) {
    return declaration.key == key;
  });
  if (found == clocks_.end()) {
    return clock_error(ErrorCode::kNotFound,
                       "default clock selection names a clock not declared in this builder");
  }
  default_key_ = key;
  return {};
}

expected<void, Error> ClockRegistryBuilder::add_conversion(const ClockKey& source,
                                                           const ClockKey& target,
                                                           AffineClockConversion conversion) {
  if (source.is_default() || target.is_default() || source.name().empty() ||
      target.name().empty()) {
    return clock_error(ErrorCode::kInvalidArgument,
                       "clock conversion endpoints must be nonempty named keys");
  }
  if (source == target) {
    return clock_error(ErrorCode::kIncompatible,
                       "a direct clock conversion cannot target its source domain");
  }
  const auto has_clock = [&](const ClockKey& key) {
    return std::any_of(clocks_.begin(), clocks_.end(), [&](const auto& declaration) {
      return declaration.key == key;
    });
  };
  if (!has_clock(source) || !has_clock(target)) {
    return clock_error(ErrorCode::kNotFound,
                       "clock conversion endpoint is not declared in this builder");
  }

  auto canonical = canonicalize_conversion(conversion);
  if (!canonical) {
    return make_unexpected(std::move(canonical).error());
  }
  conversion = *canonical;
  const auto existing =
      std::find_if(conversions_.begin(), conversions_.end(), [&](const auto& declaration) {
        return declaration.source == source && declaration.target == target;
      });
  if (existing != conversions_.end()) {
    if (existing->conversion == conversion) {
      return {};
    }
    return clock_error(ErrorCode::kIncompatible,
                       "duplicate clock conversion direction has conflicting coefficients");
  }
  const auto reciprocal =
      std::find_if(conversions_.begin(), conversions_.end(), [&](const auto& declaration) {
        return declaration.source == target && declaration.target == source;
      });
  if (reciprocal != conversions_.end() &&
      !are_exact_reciprocals(reciprocal->conversion, conversion)) {
    return clock_error(ErrorCode::kIncompatible,
                       "reciprocal clock conversions are not exact mathematical inverses");
  }
  conversions_.push_back(
      ConversionDeclaration{.source = source, .target = target, .conversion = conversion});
  return {};
}

expected<ClockRegistry, Error> ClockRegistryBuilder::build() const {
  if (!default_key_.has_value()) {
    return clock_error(ErrorCode::kNotFound, "no default clock was selected");
  }

  std::vector<const ClockDeclaration*> ordered_clocks;
  ordered_clocks.reserve(clocks_.size());
  for (const ClockDeclaration& clock : clocks_) {
    ordered_clocks.push_back(&clock);
  }
  std::sort(ordered_clocks.begin(), ordered_clocks.end(), [](const auto* lhs, const auto* rhs) {
    return lhs->key.name() < rhs->key.name();
  });

  auto state = std::make_shared<ClockRegistryState>();
  state->entries.reserve(ordered_clocks.size());
  std::vector<std::uint8_t> capability_bits(ordered_clocks.size());
  for (std::size_t index = 0; index < ordered_clocks.size(); ++index) {
    const ClockDeclaration& declaration = *ordered_clocks[index];
    std::uint8_t bits = 0;
    if (declaration.options.can_read_now) {
      bits |= ClockCapabilities::kCanReadNow;
    }
    if (declaration.options.can_wait_until) {
      bits |= ClockCapabilities::kCanWaitUntil;
    }
    if (declaration.options.steady) {
      bits |= ClockCapabilities::kSteady;
    }
    capability_bits[index] = bits;
    state->entries.push_back(ClockRegistryEntry{
        .key = declaration.key,
        .id = ClockId{static_cast<std::uint64_t>(index + 1U)},
        .capabilities = ClockCapabilities{},
        .live_state = std::make_shared<ClockRefState>(ClockRefState{.clock = declaration.clock}),
        .steady_rate = advances_at_steady_rate(*declaration.clock),
    });
  }

  const auto entry_index = [&](const ClockKey& key) -> std::optional<std::size_t> {
    const auto found =
        std::lower_bound(state->entries.begin(),
                         state->entries.end(),
                         key.name(),
                         [](const ClockRegistryEntry& entry, const std::string_view candidate) {
                           return entry.key.name() < candidate;
                         });
    if (found == state->entries.end() || found->key.name() != key.name()) {
      return std::nullopt;
    }
    return static_cast<std::size_t>(found - state->entries.begin());
  };

  const std::optional<std::size_t> default_index = entry_index(*default_key_);
  if (!default_index.has_value()) {
    return clock_error(ErrorCode::kNotFound,
                       "the selected default clock is absent from the frozen declaration set");
  }
  if ((capability_bits[*default_index] & ClockCapabilities::kCanReadNow) == 0) {
    return clock_error(ErrorCode::kNotSupported,
                       "the selected default clock does not support current-time reads");
  }
  state->default_id = state->entries[*default_index].id;

  std::vector<std::vector<std::size_t>> adjacency(state->entries.size());
  state->conversions.reserve(conversions_.size());
  for (const ConversionDeclaration& declaration : conversions_) {
    const std::optional<std::size_t> source_index = entry_index(declaration.source);
    const std::optional<std::size_t> target_index = entry_index(declaration.target);
    if (!source_index.has_value() || !target_index.has_value()) {
      return clock_error(ErrorCode::kNotFound);
    }
    adjacency[*source_index].push_back(*target_index);
    capability_bits[*source_index] |= ClockCapabilities::kHasOutgoingConversions;
    capability_bits[*target_index] |= ClockCapabilities::kHasIncomingConversions;
    state->conversions.push_back(ClockConversionRecord{
        .source = state->entries[*source_index].id,
        .target = state->entries[*target_index].id,
        .conversion = declaration.conversion,
    });
  }
  if (contains_larger_cycle(adjacency)) {
    return clock_error(
        ErrorCode::kIncompatible,
        "clock conversion declarations contain a cycle spanning three or more domains");
  }
  std::sort(state->conversions.begin(), state->conversions.end(), conversion_record_less);

  for (std::size_t index = 0; index < state->entries.size(); ++index) {
    state->entries[index].capabilities = ClockCapabilities{capability_bits[index]};
  }
  return ClockRegistry{std::move(state)};
}

expected<ClockRef, Error> ClockRegistry::resolve(const ClockKey& key) const noexcept {
  if (state_ == nullptr) {
    return clock_error(ErrorCode::kNotFound);
  }
  const ClockRegistryEntry* entry = nullptr;
  if (key.is_default()) {
    entry = find_entry(*state_, state_->default_id);
  } else if (key.name().empty()) {
    return clock_error(ErrorCode::kInvalidArgument);
  } else {
    entry = find_named_entry(*state_, key.name());
  }
  if (entry == nullptr) {
    return clock_error(ErrorCode::kNotFound);
  }
  return ClockRef{entry->id, entry->capabilities, entry->live_state};
}

expected<ClockRef, Error> ClockRegistry::resolve(const ClockId id) const noexcept {
  if (state_ == nullptr) {
    return clock_error(ErrorCode::kNotFound);
  }
  const ClockRegistryEntry* const entry = find_entry(*state_, id);
  if (entry == nullptr) {
    return clock_error(ErrorCode::kNotFound);
  }
  return ClockRef{entry->id, entry->capabilities, entry->live_state};
}

bool ClockRegistry::contains(const ClockId id) const noexcept {
  return state_ != nullptr && find_entry(*state_, id) != nullptr;
}

expected<TimePoint, Error> ClockRegistry::now(const ClockId id) const noexcept {
  if (state_ == nullptr) {
    return clock_error(ErrorCode::kNotFound);
  }
  const ClockRegistryEntry* const entry = find_entry(*state_, id);
  if (entry == nullptr) {
    return clock_error(ErrorCode::kNotFound);
  }
  if (!entry->capabilities.can_read_now()) {
    return clock_error(ErrorCode::kNotSupported);
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    return TimePoint{entry->live_state->clock->timestamp(), entry->id};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    return clock_error(ErrorCode::kFailure);
  }
#endif
}

expected<RetainedClock, Error> ClockRegistry::retain(const ClockRef& clock) noexcept {
  if (!clock.id_.valid()) {
    return RetainedClock{};
  }
  std::shared_ptr<const ClockRefState> state = clock.state_.lock();
  if (state == nullptr) {
    return clock_error(ErrorCode::kStaleEndpoint);
  }
  return RetainedClock{clock.id_, clock.capabilities_, std::move(state)};
}

expected<TimePoint, Error> RetainedClock::now() const noexcept {
  if (!id_.valid()) {
    return clock_error(ErrorCode::kInvalidArgument);
  }
  if (!capabilities_.can_read_now()) {
    return clock_error(ErrorCode::kNotSupported);
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    return TimePoint{state_->clock->timestamp(), id_};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    return clock_error(ErrorCode::kFailure);
  }
#endif
}

expected<void, Error> ClockRegistry::wait_until(const ClockId id,
                                                const std::int64_t timestamp_ns) const noexcept {
  if (state_ == nullptr) {
    return clock_error(ErrorCode::kNotFound);
  }
  const ClockRegistryEntry* const entry = find_entry(*state_, id);
  if (entry == nullptr) {
    return clock_error(ErrorCode::kNotFound);
  }
  if (!entry->capabilities.can_wait_until()) {
    return clock_error(ErrorCode::kNotSupported);
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    return entry->live_state->clock->sleep_until(timestamp_ns);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    return clock_error(ErrorCode::kFailure);
  }
#endif
}

expected<TimePoint, Error> ClockRegistry::convert(const TimePoint source,
                                                  const ClockId target) const noexcept {
  if (state_ == nullptr) {
    return clock_error(ErrorCode::kNotFound);
  }
  auto exact = project_exact(*state_, source, target);
  if (!exact) {
    return make_unexpected(std::move(exact).error());
  }
  auto materialized = materialize_exact(*exact);
  if (!materialized) {
    return make_unexpected(std::move(materialized).error());
  }
  return TimePoint{*materialized, target};
}

expected<std::strong_ordering, Error> ClockRegistry::compare(const TimePoint lhs,
                                                             const TimePoint rhs,
                                                             const ClockId target) const noexcept {
  if (state_ == nullptr) {
    return clock_error(ErrorCode::kNotFound);
  }
  auto projected_lhs = project_exact(*state_, lhs, target);
  if (!projected_lhs) {
    return make_unexpected(std::move(projected_lhs).error());
  }
  auto projected_rhs = project_exact(*state_, rhs, target);
  if (!projected_rhs) {
    return make_unexpected(std::move(projected_rhs).error());
  }
  return compare_exact(*projected_lhs, *projected_rhs);
}

bool ClockRegistry::advances_at_steady_rate(const ClockId id) const noexcept {
  if (state_ == nullptr) {
    return false;
  }
  const ClockRegistryEntry* const entry = find_entry(*state_, id);
  return entry != nullptr && entry->steady_rate;
}

}  // namespace holoscan::detail

namespace holoscan {

expected<TimePoint, Error> ClockRef::now() const noexcept {
  if (!id_.valid()) {
    return detail::clock_error(ErrorCode::kInvalidArgument);
  }
  const std::shared_ptr<const detail::ClockRefState> state = state_.lock();
  if (state == nullptr) {
    return detail::clock_error(ErrorCode::kStaleEndpoint);
  }
  if (!capabilities_.can_read_now()) {
    return detail::clock_error(ErrorCode::kNotSupported);
  }
  try {
    return TimePoint{state->clock->timestamp(), id_};
  } catch (...) {
    return detail::clock_error(ErrorCode::kFailure);
  }
}

}  // namespace holoscan
