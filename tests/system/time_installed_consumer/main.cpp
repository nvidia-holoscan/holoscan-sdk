/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <concepts>
#include <cstdint>
#include <type_traits>
#include <unordered_set>

#include <holoscan/core/source_id.hpp>
#include <holoscan/core/tag.hpp>
#include <holoscan/time/clock_registry.hpp>
#include <holoscan/time/time_point.hpp>

template <typename T>
concept LessComparable = requires(const T lhs, const T rhs) {
  { lhs < rhs } -> std::convertible_to<bool>;
};

template <typename T>
concept HasSleepUntil = requires(T value) { value.sleep_until(std::int64_t{}); };

// REQ: CLOCK-021
static_assert(sizeof(holoscan::ClockId) == 8);
static_assert(sizeof(holoscan::TimePoint) == 16);
static_assert(sizeof(holoscan::Tag) == 16);
static_assert(sizeof(holoscan::TagTime) == 16);
static_assert(sizeof(holoscan::SourceId) == 4);
static_assert(std::is_standard_layout_v<holoscan::ClockId>);
static_assert(std::is_trivially_copyable_v<holoscan::ClockId>);
static_assert(std::is_standard_layout_v<holoscan::TimePoint>);
static_assert(std::is_trivially_copyable_v<holoscan::TimePoint>);
static_assert(std::is_standard_layout_v<holoscan::Tag>);
static_assert(std::is_trivially_copyable_v<holoscan::Tag>);
static_assert(std::is_standard_layout_v<holoscan::TagTime>);
static_assert(std::is_trivially_copyable_v<holoscan::TagTime>);
static_assert(std::is_standard_layout_v<holoscan::SourceId>);
static_assert(std::is_trivially_copyable_v<holoscan::SourceId>);
static_assert(!LessComparable<holoscan::ClockId>);
static_assert(!LessComparable<holoscan::TimePoint>);
static_assert(!LessComparable<holoscan::Tag>);
static_assert(LessComparable<holoscan::TagTime>);
static_assert(!LessComparable<holoscan::SourceId>);
static_assert(!HasSleepUntil<holoscan::ClockRef>);

int main() {
  const auto key = holoscan::ClockKey::named("installed_clock");
  const holoscan::ClockId id{5};
  const holoscan::TimePoint point{0, id};
  const holoscan::Tag tag{.sequence = 1, .generation = 1, .reserved = 0};
  const holoscan::TagTime order{.logical_ns = 0, .microstep = 0, .reserved = 0};
  const holoscan::AffineClockConversion conversion{.scale = {.numerator = 1, .denominator = 2},
                                                   .offset_ns = 7};
  const holoscan::ClockCapabilities no_capabilities;
  const holoscan::ClockRef inert_reference;
  const auto inert_now = inert_reference.now();
  const auto inert_label = inert_reference.at_ns(11);
  const holoscan::SourceId invalid_source;

  std::unordered_set<holoscan::ClockId> ids{id};
  std::unordered_set<holoscan::Tag> tags{tag};
  std::unordered_set<holoscan::SourceId> sources{invalid_source};
  return !key.is_default() && key.name() == "installed_clock" && point.valid() && tag.valid() &&
                 order.valid() && conversion.scale.numerator == 1 &&
                 conversion.scale.denominator == 2 && conversion.offset_ns == 7 &&
                 !no_capabilities.can_read_now() && !static_cast<bool>(invalid_source) &&
                 !inert_now && inert_now.error().code == holoscan::ErrorCode::kInvalidArgument &&
                 inert_label.timestamp_ns == 11 && !inert_label.valid() && ids.contains(id) &&
                 tags.contains(tag) && sources.contains(invalid_source)
             ? 0
             : 1;
}
