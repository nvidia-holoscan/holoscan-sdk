/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

#include <holoscan/time/clock_registry.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/time_point.hpp>

#include <time/clock_registry_internal.hpp>

namespace {

constinit thread_local bool g_count_clock_allocations = false;
constinit thread_local std::size_t g_clock_allocation_count = 0;

void* counted_malloc(const std::size_t size) {
  if (g_count_clock_allocations) {
    ++g_clock_allocation_count;
  }
  if (void* pointer = std::malloc(size == 0 ? 1 : size)) {
    return pointer;
  }
  throw std::bad_alloc();
}

void* counted_aligned_malloc(const std::size_t size, const std::align_val_t alignment) {
  if (g_count_clock_allocations) {
    ++g_clock_allocation_count;
  }
#if defined(_MSC_VER)
  if (void* pointer = _aligned_malloc(
          size == 0 ? 1 : size, std::max(static_cast<std::size_t>(alignment), sizeof(void*)))) {
    return pointer;
  }
#else
  void* pointer = nullptr;
  const auto normalized_alignment = std::max(static_cast<std::size_t>(alignment), sizeof(void*));
  if (posix_memalign(&pointer, normalized_alignment, size == 0 ? 1 : size) == 0) {
    return pointer;
  }
#endif
  throw std::bad_alloc();
}

class ScopedClockAllocationCounter {
 public:
  ScopedClockAllocationCounter() noexcept
      : previous_enabled_(g_count_clock_allocations), previous_count_(g_clock_allocation_count) {
    g_clock_allocation_count = 0;
    g_count_clock_allocations = true;
  }

  ~ScopedClockAllocationCounter() noexcept {
    g_count_clock_allocations = previous_enabled_;
    if (previous_enabled_) {
      g_clock_allocation_count += previous_count_;
    }
  }

  [[nodiscard]] std::size_t count() const noexcept { return g_clock_allocation_count; }

 private:
  bool previous_enabled_ = false;
  std::size_t previous_count_ = 0;
};

}  // namespace

void* operator new(const std::size_t size) {
  return counted_malloc(size);
}

void* operator new[](const std::size_t size) {
  return counted_malloc(size);
}

void* operator new(const std::size_t size, const std::align_val_t alignment) {
  return counted_aligned_malloc(size, alignment);
}

void* operator new[](const std::size_t size, const std::align_val_t alignment) {
  return counted_aligned_malloc(size, alignment);
}

void operator delete(void* pointer) noexcept {
  std::free(pointer);
}

void operator delete[](void* pointer) noexcept {
  std::free(pointer);
}

void operator delete(void* pointer, std::size_t) noexcept {
  std::free(pointer);
}

void operator delete[](void* pointer, std::size_t) noexcept {
  std::free(pointer);
}

void operator delete(void* pointer, std::align_val_t) noexcept {
#if defined(_MSC_VER)
  _aligned_free(pointer);
#else
  std::free(pointer);
#endif
}

void operator delete[](void* pointer, std::align_val_t alignment) noexcept {
  ::operator delete(pointer, alignment);
}

void operator delete(void* pointer, std::size_t, std::align_val_t alignment) noexcept {
  ::operator delete(pointer, alignment);
}

void operator delete[](void* pointer, std::size_t, std::align_val_t alignment) noexcept {
  ::operator delete(pointer, alignment);
}

namespace holoscan {
namespace {

template <typename T>
concept LessComparable = requires(const T lhs, const T rhs) {
  { lhs < rhs } -> std::convertible_to<bool>;
};

template <typename T>
concept HasSleepUntil = requires(T value) { value.sleep_until(std::int64_t{}); };

template <typename T>
concept HasStringLookup = requires(T value) { value.require("camera"); };

// REQ: CLOCK-009
static_assert(sizeof(ClockId) == 8);
static_assert(sizeof(TimePoint) == 16);
static_assert(offsetof(ClockId, value) == 0);
static_assert(offsetof(TimePoint, timestamp_ns) == 0);
static_assert(offsetof(TimePoint, clock_id) == 8);
static_assert(std::is_standard_layout_v<ClockId>);
static_assert(std::is_trivially_copyable_v<ClockId>);
static_assert(std::is_standard_layout_v<TimePoint>);
static_assert(std::is_trivially_copyable_v<TimePoint>);
static_assert(!LessComparable<ClockId>);
static_assert(!LessComparable<TimePoint>);
// REQ: CLOCK-010
static_assert(!std::is_constructible_v<ClockKey, ClockId>);
static_assert(!std::is_constructible_v<ClockKey, ClockRef>);
// REQ: CLOCK-011
static_assert(!HasSleepUntil<ClockRef>);
static_assert(!HasStringLookup<ClockRef>);

detail::ClockCapabilityOptions readable_capabilities(const bool waitable = true,
                                                     const bool steady = true) {
  return detail::ClockCapabilityOptions{
      .can_read_now = true,
      .can_wait_until = waitable,
      .steady = steady,
  };
}

class ThrowingClock final : public Clock {
 public:
  [[nodiscard]] double time() const override { throw 1; }
  [[nodiscard]] std::int64_t timestamp() const override { throw 1; }
  [[nodiscard]] expected<void, Error> sleep_for(std::int64_t) override { return {}; }
  [[nodiscard]] expected<void, Error> sleep_until(std::int64_t) override { return {}; }
};

expected<detail::ClockRegistry, Error> make_two_clock_registry(const bool reverse_order) {
  detail::ClockRegistryBuilder builder;
  const auto camera_key = ClockKey::named("camera");
  const auto control_key = ClockKey::named("control");

  if (reverse_order) {
    HOLOSCAN_RETURN_IF_ERROR(builder.add_clock(
        control_key, std::make_shared<ManualClock>(2'000), readable_capabilities()));
    HOLOSCAN_RETURN_IF_ERROR(builder.add_clock(
        camera_key, std::make_shared<ManualClock>(1'000), readable_capabilities()));
  } else {
    HOLOSCAN_RETURN_IF_ERROR(builder.add_clock(
        camera_key, std::make_shared<ManualClock>(1'000), readable_capabilities()));
    HOLOSCAN_RETURN_IF_ERROR(builder.add_clock(
        control_key, std::make_shared<ManualClock>(2'000), readable_capabilities()));
  }
  HOLOSCAN_RETURN_IF_ERROR(builder.set_default(control_key));
  return builder.build();
}

expected<detail::ClockRegistry, Error> make_conversion_registry(
    const AffineClockConversion conversion) {
  detail::ClockRegistryBuilder builder;
  const auto source_key = ClockKey::named("source");
  const auto target_key = ClockKey::named("target");
  HOLOSCAN_RETURN_IF_ERROR(
      builder.add_clock(source_key, std::make_shared<ManualClock>(), readable_capabilities()));
  HOLOSCAN_RETURN_IF_ERROR(
      builder.add_clock(target_key, std::make_shared<ManualClock>(), readable_capabilities()));
  HOLOSCAN_RETURN_IF_ERROR(builder.set_default(source_key));
  HOLOSCAN_RETURN_IF_ERROR(builder.add_conversion(source_key, target_key, conversion));
  return builder.build();
}

expected<std::int64_t, Error> convert_single_coordinate(const AffineClockConversion conversion,
                                                        const std::int64_t source_coordinate) {
  auto registry_result = make_conversion_registry(conversion);
  if (!registry_result) {
    return make_unexpected(registry_result.error());
  }
  auto registry = std::move(registry_result).value();
  auto source = registry.resolve(ClockKey::named("source"));
  if (!source) {
    return make_unexpected(source.error());
  }
  auto target = registry.resolve(ClockKey::named("target"));
  if (!target) {
    return make_unexpected(target.error());
  }
  auto converted = registry.convert(source->at_ns(source_coordinate), target->id());
  if (!converted) {
    return make_unexpected(converted.error());
  }
  return converted->timestamp_ns;
}

// REQ: CLOCK-009
TEST(ClockValues, PreserveFixedLayoutValidityEqualityAndHashing) {
  constexpr ClockId invalid{};
  constexpr ClockId camera{17};
  constexpr ClockId another_camera{17};
  constexpr ClockId control{18};
  constexpr TimePoint absent{};
  constexpr TimePoint zero_in_camera{0, camera};
  constexpr TimePoint captured{42, camera};

  static_assert(!invalid.valid());
  static_assert(camera.valid());
  static_assert(camera == another_camera);
  static_assert(camera != control);
  static_assert(!absent.valid());
  static_assert(zero_in_camera.valid());
  static_assert(captured.valid());
  static_assert(captured == TimePoint{42, camera});
  static_assert(captured != TimePoint{42, control});

  std::unordered_set<ClockId> clock_ids;
  clock_ids.insert(camera);
  clock_ids.insert(another_camera);
  clock_ids.insert(control);
  EXPECT_EQ(clock_ids.size(), 2);

  std::unordered_set<TimePoint> time_points;
  time_points.insert(captured);
  time_points.insert(TimePoint{42, camera});
  time_points.insert(TimePoint{42, control});
  EXPECT_EQ(time_points.size(), 2);
}

// REQ: CLOCK-024
TEST(ClockValues, PlanLocalNumericIdsRequireExternalProvenanceAcrossRegistries) {
  detail::ClockRegistryBuilder camera_builder;
  ASSERT_TRUE(camera_builder.add_clock(ClockKey::named("camera"),
                                       std::make_shared<ManualClock>(11),
                                       detail::ClockCapabilityOptions{.can_read_now = true}));
  ASSERT_TRUE(camera_builder.set_default(ClockKey::named("camera")));
  auto camera_registry_result = camera_builder.build();
  ASSERT_TRUE(camera_registry_result);
  auto camera_registry = std::move(camera_registry_result).value();

  detail::ClockRegistryBuilder control_builder;
  ASSERT_TRUE(control_builder.add_clock(ClockKey::named("control"),
                                        std::make_shared<ManualClock>(22),
                                        detail::ClockCapabilityOptions{.can_read_now = true}));
  ASSERT_TRUE(control_builder.set_default(ClockKey::named("control")));
  auto control_registry_result = control_builder.build();
  ASSERT_TRUE(control_registry_result);
  auto control_registry = std::move(control_registry_result).value();

  auto camera = camera_registry.resolve(ClockKey::default_clock());
  auto control = control_registry.resolve(ClockKey::default_clock());
  ASSERT_TRUE(camera);
  ASSERT_TRUE(control);

  // Independent catalogs may assign the same compact numeric identifier. The raw values therefore
  // cannot prove provenance and must never be passed to the other registry for interpretation.
  EXPECT_EQ(camera->id(), control->id());
  EXPECT_EQ(camera->at_ns(7), control->at_ns(7));
  auto camera_now = camera->now();
  auto control_now = control->now();
  ASSERT_TRUE(camera_now);
  ASSERT_TRUE(control_now);
  EXPECT_NE(camera_now->timestamp_ns, control_now->timestamp_ns);
}

// REQ: CLOCK-010
TEST(ClockKey, DistinguishesColdDefaultAndOwnedNamedKeys) {
  const auto default_key = ClockKey::default_clock();
  EXPECT_TRUE(default_key.is_default());
  EXPECT_TRUE(default_key.name().empty());
  EXPECT_EQ(default_key, ClockKey{});

  std::string mutable_name = "camera";
  const auto named = ClockKey::named(mutable_name);
  mutable_name.assign("changed");

  EXPECT_FALSE(named.is_default());
  EXPECT_EQ(named.name(), "camera");
  EXPECT_EQ(named, ClockKey{"camera"});
  EXPECT_NE(named, ClockKey::named("control"));

  std::unordered_set<ClockKey> keys;
  keys.insert(ClockKey::named("camera"));
  keys.insert(named);
  keys.insert(default_key);
  EXPECT_EQ(keys.size(), 2);
}

// REQ: CLOCK-011
TEST(ClockCapabilities, DefaultValueGrantsNoCapability) {
  const ClockCapabilities capabilities;
  EXPECT_FALSE(capabilities.can_read_now());
  EXPECT_FALSE(capabilities.can_wait_until());
  EXPECT_FALSE(capabilities.steady());
  EXPECT_FALSE(capabilities.has_outgoing_conversions());
  EXPECT_FALSE(capabilities.has_incoming_conversions());
}

// REQ: CLOCK-011
// REQ: CLOCK-022
TEST(ClockRef, DefaultReferenceIsInertButCanLabelItsInvalidDomain) {
  const ClockRef reference;
  EXPECT_FALSE(reference.id().valid());
  auto now = reference.now();
  ASSERT_FALSE(now);
  EXPECT_EQ(now.error().code, ErrorCode::kInvalidArgument);

  const TimePoint labeled = reference.at_ns(77);
  EXPECT_EQ(labeled.timestamp_ns, 77);
  EXPECT_FALSE(labeled.valid());
}

// REQ: CLOCK-012
TEST(ClockRegistry, AssignsDeterministicIdsAndResolvesTheDefault) {
  auto forward_result = make_two_clock_registry(false);
  auto reverse_result = make_two_clock_registry(true);
  ASSERT_TRUE(forward_result) << runtime_error_message(forward_result.error());
  ASSERT_TRUE(reverse_result) << runtime_error_message(reverse_result.error());
  auto forward = std::move(forward_result).value();
  auto reverse = std::move(reverse_result).value();

  auto forward_camera = forward.resolve(ClockKey::named("camera"));
  auto reverse_camera = reverse.resolve(ClockKey::named("camera"));
  auto forward_control = forward.resolve(ClockKey::named("control"));
  auto default_clock = forward.resolve(ClockKey::default_clock());
  ASSERT_TRUE(forward_camera);
  ASSERT_TRUE(reverse_camera);
  ASSERT_TRUE(forward_control);
  ASSERT_TRUE(default_clock);

  EXPECT_TRUE(forward_camera->id().valid());
  EXPECT_NE(forward_camera->id(), forward_control->id());
  EXPECT_EQ(forward_camera->id(), reverse_camera->id());
  EXPECT_EQ(default_clock->id(), forward_control->id());
  EXPECT_TRUE(forward_camera->capabilities().can_read_now());
  EXPECT_TRUE(forward_camera->capabilities().can_wait_until());
  EXPECT_TRUE(forward_camera->capabilities().steady());

  ASSERT_TRUE(forward_camera->now());
  EXPECT_EQ(forward_camera->now()->timestamp_ns, 1'000);
  EXPECT_EQ(forward_camera->now()->clock_id, forward_camera->id());
  EXPECT_EQ(forward_control->at_ns(-5), (TimePoint{-5, forward_control->id()}));
}

// REQ: CLOCK-012
// REQ: CLOCK-015
TEST(ClockRegistry, ConvertsAndComparesSameDomainCoordinatesWithoutAConversionEdge) {
  auto registry_result = make_two_clock_registry(false);
  ASSERT_TRUE(registry_result);
  auto registry = std::move(registry_result).value();
  auto camera = registry.resolve(ClockKey::named("camera"));
  ASSERT_TRUE(camera);

  auto converted = registry.convert(camera->at_ns(-7), camera->id());
  ASSERT_TRUE(converted);
  EXPECT_EQ(*converted, camera->at_ns(-7));

  auto less = registry.compare(camera->at_ns(-7), camera->at_ns(2), camera->id());
  auto equal = registry.compare(camera->at_ns(2), camera->at_ns(2), camera->id());
  ASSERT_TRUE(less);
  ASSERT_TRUE(equal);
  EXPECT_EQ(*less, std::strong_ordering::less);
  EXPECT_EQ(*equal, std::strong_ordering::equal);
}

// REQ: CLOCK-012
// REQ: CLOCK-022
TEST(ClockRegistry, RejectsDuplicateUnknownAndMissingDefaultDeclarations) {
  const auto key = ClockKey::named("duplicate");
  detail::ClockRegistryBuilder empty_name_builder;
  auto empty_name = empty_name_builder.add_clock(
      ClockKey::named(""), std::make_shared<ManualClock>(), readable_capabilities());
  ASSERT_FALSE(empty_name);
  EXPECT_EQ(empty_name.error().code, ErrorCode::kInvalidArgument);
  EXPECT_FALSE(empty_name.error().message.empty());

  detail::ClockRegistryBuilder duplicate_builder;
  ASSERT_TRUE(
      duplicate_builder.add_clock(key, std::make_shared<ManualClock>(), readable_capabilities()));
  auto duplicate =
      duplicate_builder.add_clock(key, std::make_shared<ManualClock>(), readable_capabilities());
  ASSERT_FALSE(duplicate);
  EXPECT_EQ(duplicate.error().code, ErrorCode::kDuplicateName);
  EXPECT_FALSE(duplicate.error().message.empty());

  detail::ClockRegistryBuilder unknown_default_builder;
  ASSERT_TRUE(unknown_default_builder.add_clock(
      key, std::make_shared<ManualClock>(), readable_capabilities()));
  auto unknown_default = unknown_default_builder.set_default(ClockKey::named("missing"));
  ASSERT_FALSE(unknown_default);
  EXPECT_EQ(unknown_default.error().code, ErrorCode::kNotFound);

  detail::ClockRegistryBuilder missing_default_builder;
  ASSERT_TRUE(missing_default_builder.add_clock(
      key, std::make_shared<ManualClock>(), readable_capabilities()));
  auto missing_default = missing_default_builder.build();
  ASSERT_FALSE(missing_default);
  EXPECT_EQ(missing_default.error().code, ErrorCode::kNotFound);

  detail::ClockRegistryBuilder unreadable_default_builder;
  ASSERT_TRUE(unreadable_default_builder.add_clock(
      key, std::make_shared<ManualClock>(), detail::ClockCapabilityOptions{}));
  ASSERT_TRUE(unreadable_default_builder.set_default(key));
  auto unreadable_default = unreadable_default_builder.build();
  ASSERT_FALSE(unreadable_default);
  EXPECT_EQ(unreadable_default.error().code, ErrorCode::kNotSupported);
}

// REQ: CLOCK-011
// REQ: CLOCK-012
// REQ: CLOCK-022
TEST(ClockRegistry, EnforcesReadCapabilityAndGenerationLifetime) {
  ClockRef stale_reference;
  {
    detail::ClockRegistryBuilder builder;
    const auto key = ClockKey::named("label_only");
    const auto default_key = ClockKey::named("readable_default");
    ASSERT_TRUE(builder.add_clock(
        key, std::make_shared<ManualClock>(123), detail::ClockCapabilityOptions{}));
    ASSERT_TRUE(builder.add_clock(
        default_key, std::make_shared<ManualClock>(456), readable_capabilities()));
    ASSERT_TRUE(builder.set_default(default_key));
    auto registry_result = builder.build();
    ASSERT_TRUE(registry_result);
    auto registry = std::move(registry_result).value();

    auto reference = registry.resolve(key);
    auto default_reference = registry.resolve(ClockKey::default_clock());
    ASSERT_TRUE(reference);
    ASSERT_TRUE(default_reference);
    ASSERT_TRUE(default_reference->now());
    EXPECT_EQ(default_reference->now()->timestamp_ns, 456);
    stale_reference = *reference;
    EXPECT_FALSE(reference->capabilities().can_read_now());
    auto unreadable_now = reference->now();
    ASSERT_FALSE(unreadable_now);
    EXPECT_EQ(unreadable_now.error().code, ErrorCode::kNotSupported);
    EXPECT_EQ(reference->at_ns(9), (TimePoint{9, reference->id()}));
  }

  EXPECT_TRUE(stale_reference.id().valid());
  auto stale_now = stale_reference.now();
  ASSERT_FALSE(stale_now);
  EXPECT_EQ(stale_now.error().code, ErrorCode::kStaleEndpoint);
  EXPECT_EQ(stale_reference.at_ns(10), (TimePoint{10, stale_reference.id()}));
}

// REQ: CLOCK-011
// REQ: CLOCK-022
TEST(ClockRef, MapsAnUnderlyingClockExceptionToFailure) {
  detail::ClockRegistryBuilder builder;
  const auto key = ClockKey::named("throwing");
  ASSERT_TRUE(builder.add_clock(key, std::make_shared<ThrowingClock>(), readable_capabilities()));
  ASSERT_TRUE(builder.set_default(key));
  auto registry_result = builder.build();
  ASSERT_TRUE(registry_result);
  auto registry = std::move(registry_result).value();
  auto reference = registry.resolve(key);
  ASSERT_TRUE(reference);

  auto now = reference->now();
  ASSERT_FALSE(now);
  EXPECT_EQ(now.error().code, ErrorCode::kFailure);
}

// REQ: CLOCK-023
TEST(ClockRef, ResolvedOperationsAvoidAllocationAndStringLookup) {
  detail::ClockRegistryBuilder builder;
  const auto key = ClockKey::named("resolved");
  ASSERT_TRUE(builder.add_clock(key, std::make_shared<ManualClock>(123), readable_capabilities()));
  ASSERT_TRUE(builder.set_default(key));
  auto registry_result = builder.build();
  ASSERT_TRUE(registry_result) << runtime_error_message(registry_result.error());
  auto registry = std::move(registry_result).value();
  auto reference = registry.resolve(ClockKey::default_clock());
  ASSERT_TRUE(reference);
  ASSERT_TRUE(reference->now());  // Warm up any test-framework or lazy process state.

  std::size_t allocation_count = 0;
  std::int64_t observed_sum = 0;
  bool every_operation_succeeded = true;
  {
    const ScopedClockAllocationCounter allocations;
    for (std::int64_t index = 0; index < 256; ++index) {
      auto now = reference->now();
      if (!now) {
        every_operation_succeeded = false;
        continue;
      }
      const auto labeled = reference->at_ns(index);
      every_operation_succeeded = every_operation_succeeded && labeled.valid() &&
                                  labeled.clock_id == reference->id() &&
                                  labeled.timestamp_ns == index;
      observed_sum += now->timestamp_ns + labeled.timestamp_ns;
    }
    allocation_count = allocations.count();
  }

  EXPECT_TRUE(every_operation_succeeded);
  EXPECT_GT(observed_sum, 0);
  EXPECT_EQ(allocation_count, 0U);
}

// REQ: CLOCK-013
TEST(ClockConversion, CanonicalizesDuplicatesAndDerivesParticipationCapabilities) {
  detail::ClockRegistryBuilder builder;
  const auto source_key = ClockKey::named("source");
  const auto target_key = ClockKey::named("target");
  ASSERT_TRUE(
      builder.add_clock(source_key, std::make_shared<ManualClock>(), readable_capabilities()));
  ASSERT_TRUE(
      builder.add_clock(target_key, std::make_shared<ManualClock>(), readable_capabilities()));
  ASSERT_TRUE(builder.set_default(source_key));
  ASSERT_TRUE(builder.add_conversion(
      source_key,
      target_key,
      AffineClockConversion{.scale = {.numerator = 2, .denominator = 4}, .offset_ns = 7}));
  ASSERT_TRUE(builder.add_conversion(
      source_key,
      target_key,
      AffineClockConversion{.scale = {.numerator = 1, .denominator = 2}, .offset_ns = 7}));

  auto registry_result = builder.build();
  ASSERT_TRUE(registry_result) << runtime_error_message(registry_result.error());
  auto registry = std::move(registry_result).value();
  auto source = registry.resolve(source_key);
  auto target = registry.resolve(target_key);
  ASSERT_TRUE(source);
  ASSERT_TRUE(target);
  EXPECT_TRUE(source->capabilities().has_outgoing_conversions());
  EXPECT_FALSE(source->capabilities().has_incoming_conversions());
  EXPECT_FALSE(target->capabilities().has_outgoing_conversions());
  EXPECT_TRUE(target->capabilities().has_incoming_conversions());

  auto converted = registry.convert(source->at_ns(10), target->id());
  ASSERT_TRUE(converted);
  EXPECT_EQ(*converted, (TimePoint{12, target->id()}));
}

// REQ: CLOCK-015
TEST(ClockConversion, UsesNearestTiesToEvenAndExactRationalComparison) {
  auto registry_result = make_conversion_registry(
      AffineClockConversion{.scale = {.numerator = 3, .denominator = 2}, .offset_ns = 0});
  ASSERT_TRUE(registry_result) << runtime_error_message(registry_result.error());
  auto registry = std::move(registry_result).value();
  auto source = registry.resolve(ClockKey::named("source"));
  auto target = registry.resolve(ClockKey::named("target"));
  ASSERT_TRUE(source);
  ASSERT_TRUE(target);

  EXPECT_EQ(registry.convert(source->at_ns(1), target->id()).value(), (TimePoint{2, target->id()}));
  EXPECT_EQ(registry.convert(source->at_ns(3), target->id()).value(), (TimePoint{4, target->id()}));
  EXPECT_EQ(registry.convert(source->at_ns(-1), target->id()).value(),
            (TimePoint{-2, target->id()}));
  EXPECT_EQ(registry.convert(source->at_ns(-3), target->id()).value(),
            (TimePoint{-4, target->id()}));

  // The exact converted value of source=1 is 1.5. Boundary classification must not compare its
  // rounded materialization (2) with an exact target coordinate.
  EXPECT_EQ(registry.compare(source->at_ns(1), target->at_ns(1), target->id()).value(),
            std::strong_ordering::greater);
  EXPECT_EQ(registry.compare(source->at_ns(1), target->at_ns(2), target->id()).value(),
            std::strong_ordering::less);
}

// REQ: CLOCK-015
TEST(ClockConversion, CoversPortableRoundingAndExactComparisonVectors) {
  const auto expect_coordinate = [](const AffineClockConversion conversion,
                                    const std::int64_t source,
                                    const std::int64_t expected_coordinate) {
    auto converted = convert_single_coordinate(conversion, source);
    ASSERT_TRUE(converted) << runtime_error_message(converted.error());
    EXPECT_EQ(*converted, expected_coordinate);
  };

  const AffineClockConversion half{.scale = {.numerator = 1, .denominator = 2}, .offset_ns = 0};
  expect_coordinate(half, 1, 0);    // +0.5 ties to even zero.
  expect_coordinate(half, 3, 2);    // +1.5 ties to even two.
  expect_coordinate(half, 5, 2);    // +2.5 ties to even two.
  expect_coordinate(half, -1, 0);   // -0.5 ties to even zero.
  expect_coordinate(half, -3, -2);  // -1.5 ties to even negative two.
  expect_coordinate(half, -5, -2);  // -2.5 ties to even negative two.

  // Fold the integral offset before rounding: 1 * 1/2 + 1 is 1.5 and materializes as 2.
  expect_coordinate(
      AffineClockConversion{.scale = {.numerator = 1, .denominator = 2}, .offset_ns = 1}, 1, 2);

  // Exercise half classification with a denominator too large for signed 64-bit arithmetic.
  constexpr std::uint64_t kLargeDenominator = 18'000'000'000'000'000'000ULL;
  expect_coordinate(AffineClockConversion{.scale = {.numerator = 8'999'999'999'999'999'999LL,
                                                    .denominator = kLargeDenominator},
                                          .offset_ns = 0},
                    1,
                    0);
  expect_coordinate(AffineClockConversion{.scale = {.numerator = 9'000'000'000'000'000'000LL,
                                                    .denominator = kLargeDenominator},
                                          .offset_ns = 0},
                    1,
                    0);
  expect_coordinate(AffineClockConversion{.scale = {.numerator = 9'000'000'000'000'000'001LL,
                                                    .denominator = kLargeDenominator},
                                          .offset_ns = 0},
                    1,
                    1);

  detail::ClockRegistryBuilder builder;
  const auto thirds_key = ClockKey::named("thirds");
  const auto fifths_key = ClockKey::named("fifths");
  const auto target_key = ClockKey::named("target");
  for (const auto& key : {thirds_key, fifths_key, target_key}) {
    ASSERT_TRUE(builder.add_clock(key, std::make_shared<ManualClock>(), readable_capabilities()));
  }
  ASSERT_TRUE(builder.set_default(target_key));
  ASSERT_TRUE(builder.add_conversion(
      thirds_key,
      target_key,
      AffineClockConversion{.scale = {.numerator = 1, .denominator = 3}, .offset_ns = 0}));
  ASSERT_TRUE(builder.add_conversion(
      fifths_key,
      target_key,
      AffineClockConversion{.scale = {.numerator = 2, .denominator = 5}, .offset_ns = 0}));
  auto registry_result = builder.build();
  ASSERT_TRUE(registry_result) << runtime_error_message(registry_result.error());
  auto registry = std::move(registry_result).value();
  auto thirds = registry.resolve(thirds_key);
  auto fifths = registry.resolve(fifths_key);
  auto target = registry.resolve(target_key);
  ASSERT_TRUE(thirds);
  ASSERT_TRUE(fifths);
  ASSERT_TRUE(target);

  EXPECT_EQ(registry.compare(thirds->at_ns(1), fifths->at_ns(1), target->id()).value(),
            std::strong_ordering::less);  // 1/3 < 2/5 exactly.
  EXPECT_EQ(registry.compare(thirds->at_ns(-1), fifths->at_ns(-1), target->id()).value(),
            std::strong_ordering::greater);  // -1/3 > -2/5 exactly.
}

// REQ: CLOCK-015
TEST(ClockConversion, HandlesSignedExtremesWithCheckedPortableArithmetic) {
  auto identity_registry_result = make_conversion_registry(
      AffineClockConversion{.scale = {.numerator = 1, .denominator = 1}, .offset_ns = 0});
  ASSERT_TRUE(identity_registry_result);
  auto identity_registry = std::move(identity_registry_result).value();
  auto identity_source = identity_registry.resolve(ClockKey::named("source"));
  auto identity_target = identity_registry.resolve(ClockKey::named("target"));
  ASSERT_TRUE(identity_source);
  ASSERT_TRUE(identity_target);

  EXPECT_EQ(identity_registry
                .convert(identity_source->at_ns(std::numeric_limits<std::int64_t>::min()),
                         identity_target->id())
                .value(),
            (TimePoint{std::numeric_limits<std::int64_t>::min(), identity_target->id()}));
  EXPECT_EQ(identity_registry
                .convert(identity_source->at_ns(std::numeric_limits<std::int64_t>::max()),
                         identity_target->id())
                .value(),
            (TimePoint{std::numeric_limits<std::int64_t>::max(), identity_target->id()}));
  EXPECT_EQ(identity_registry
                .compare(identity_source->at_ns(std::numeric_limits<std::int64_t>::min()),
                         identity_target->at_ns(std::numeric_limits<std::int64_t>::max()),
                         identity_target->id())
                .value(),
            std::strong_ordering::less);

  auto wide_ratio_registry_result = make_conversion_registry(
      AffineClockConversion{.scale = {.numerator = std::numeric_limits<std::int64_t>::max(),
                                      .denominator = std::numeric_limits<std::uint64_t>::max()},
                            .offset_ns = 0});
  ASSERT_TRUE(wide_ratio_registry_result);
  auto wide_ratio_registry = std::move(wide_ratio_registry_result).value();
  auto wide_source = wide_ratio_registry.resolve(ClockKey::named("source"));
  auto wide_target = wide_ratio_registry.resolve(ClockKey::named("target"));
  ASSERT_TRUE(wide_source);
  ASSERT_TRUE(wide_target);
  auto converted = wide_ratio_registry.convert(
      wide_source->at_ns(std::numeric_limits<std::int64_t>::max()), wide_target->id());
  ASSERT_TRUE(converted);
  EXPECT_EQ(converted->timestamp_ns, 4'611'686'018'427'387'903LL);

  auto offset_registry_result = make_conversion_registry(
      AffineClockConversion{.scale = {.numerator = 1, .denominator = 1},
                            .offset_ns = std::numeric_limits<std::int64_t>::max()});
  ASSERT_TRUE(offset_registry_result);
  auto offset_registry = std::move(offset_registry_result).value();
  auto offset_source = offset_registry.resolve(ClockKey::named("source"));
  auto offset_target = offset_registry.resolve(ClockKey::named("target"));
  ASSERT_TRUE(offset_source);
  ASSERT_TRUE(offset_target);
  auto offset_converted = offset_registry.convert(
      offset_source->at_ns(std::numeric_limits<std::int64_t>::min()), offset_target->id());
  ASSERT_TRUE(offset_converted);
  EXPECT_EQ(offset_converted->timestamp_ns, -1);

  // The intermediate product exceeds int64_t, but exact offset folding cancels it back to max.
  auto cancellation_registry_result = make_conversion_registry(
      AffineClockConversion{.scale = {.numerator = 2, .denominator = 1},
                            .offset_ns = -std::numeric_limits<std::int64_t>::max()});
  ASSERT_TRUE(cancellation_registry_result);
  auto cancellation_registry = std::move(cancellation_registry_result).value();
  auto cancellation_source = cancellation_registry.resolve(ClockKey::named("source"));
  auto cancellation_target = cancellation_registry.resolve(ClockKey::named("target"));
  ASSERT_TRUE(cancellation_source);
  ASSERT_TRUE(cancellation_target);
  auto cancellation = cancellation_registry.convert(
      cancellation_source->at_ns(std::numeric_limits<std::int64_t>::max()),
      cancellation_target->id());
  ASSERT_TRUE(cancellation);
  EXPECT_EQ(cancellation->timestamp_ns, std::numeric_limits<std::int64_t>::max());
}

// REQ: CLOCK-014
// REQ: CLOCK-022
TEST(ClockConversion, IsDirectDirectionalAndDoesNotInferMultiHopPaths) {
  detail::ClockRegistryBuilder builder;
  const auto a_key = ClockKey::named("a");
  const auto b_key = ClockKey::named("b");
  const auto c_key = ClockKey::named("c");
  for (const auto& key : {a_key, b_key, c_key}) {
    ASSERT_TRUE(builder.add_clock(key, std::make_shared<ManualClock>(), readable_capabilities()));
  }
  ASSERT_TRUE(builder.set_default(a_key));
  const AffineClockConversion identity{.scale = {.numerator = 1, .denominator = 1}, .offset_ns = 0};
  ASSERT_TRUE(builder.add_conversion(a_key, b_key, identity));
  ASSERT_TRUE(builder.add_conversion(b_key, c_key, identity));
  auto registry_result = builder.build();
  ASSERT_TRUE(registry_result);
  auto registry = std::move(registry_result).value();
  auto a = registry.resolve(a_key);
  auto b = registry.resolve(b_key);
  auto c = registry.resolve(c_key);
  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  ASSERT_TRUE(c);

  EXPECT_TRUE(registry.convert(a->at_ns(5), b->id()));
  auto missing_inverse = registry.convert(b->at_ns(5), a->id());
  ASSERT_FALSE(missing_inverse);
  EXPECT_EQ(missing_inverse.error().code, ErrorCode::kNotFound);
  auto missing_multi_hop = registry.convert(a->at_ns(5), c->id());
  ASSERT_FALSE(missing_multi_hop);
  EXPECT_EQ(missing_multi_hop.error().code, ErrorCode::kNotFound);
}

// REQ: CLOCK-014
// REQ: CLOCK-022
TEST(ClockConversion, AcceptsOnlyConsistentReciprocalDirectEdgesAndRejectsLargerCycles) {
  detail::ClockRegistryBuilder reciprocal_builder;
  const auto a_key = ClockKey::named("a");
  const auto b_key = ClockKey::named("b");
  ASSERT_TRUE(reciprocal_builder.add_clock(
      a_key, std::make_shared<ManualClock>(), readable_capabilities()));
  ASSERT_TRUE(reciprocal_builder.add_clock(
      b_key, std::make_shared<ManualClock>(), readable_capabilities()));
  ASSERT_TRUE(reciprocal_builder.set_default(a_key));
  const AffineClockConversion forward{.scale = {.numerator = 2, .denominator = 3}, .offset_ns = 4};
  const AffineClockConversion reverse{.scale = {.numerator = 3, .denominator = 2}, .offset_ns = -6};
  ASSERT_TRUE(reciprocal_builder.add_conversion(a_key, b_key, forward));
  ASSERT_TRUE(reciprocal_builder.add_conversion(b_key, a_key, reverse));
  auto reciprocal_result = reciprocal_builder.build();
  ASSERT_TRUE(reciprocal_result);
  auto reciprocal = std::move(reciprocal_result).value();
  auto a = reciprocal.resolve(a_key);
  auto b = reciprocal.resolve(b_key);
  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  auto converted_forward = reciprocal.convert(a->at_ns(9), b->id());
  auto converted_reverse = reciprocal.convert(b->at_ns(10), a->id());
  ASSERT_TRUE(converted_forward);
  ASSERT_TRUE(converted_reverse);
  EXPECT_EQ(converted_forward->timestamp_ns, 10);
  EXPECT_EQ(converted_reverse->timestamp_ns, 9);

  detail::ClockRegistryBuilder inconsistent_scale_builder;
  ASSERT_TRUE(inconsistent_scale_builder.add_clock(
      a_key, std::make_shared<ManualClock>(), readable_capabilities()));
  ASSERT_TRUE(inconsistent_scale_builder.add_clock(
      b_key, std::make_shared<ManualClock>(), readable_capabilities()));
  ASSERT_TRUE(inconsistent_scale_builder.set_default(a_key));
  ASSERT_TRUE(inconsistent_scale_builder.add_conversion(a_key, b_key, forward));
  auto inconsistent_scale = inconsistent_scale_builder.add_conversion(
      b_key,
      a_key,
      AffineClockConversion{.scale = {.numerator = 2, .denominator = 1}, .offset_ns = -6});
  ASSERT_FALSE(inconsistent_scale);
  EXPECT_EQ(inconsistent_scale.error().code, ErrorCode::kIncompatible);

  detail::ClockRegistryBuilder inconsistent_offset_builder;
  ASSERT_TRUE(inconsistent_offset_builder.add_clock(
      a_key, std::make_shared<ManualClock>(), readable_capabilities()));
  ASSERT_TRUE(inconsistent_offset_builder.add_clock(
      b_key, std::make_shared<ManualClock>(), readable_capabilities()));
  ASSERT_TRUE(inconsistent_offset_builder.set_default(a_key));
  ASSERT_TRUE(inconsistent_offset_builder.add_conversion(a_key, b_key, forward));
  auto inconsistent_offset = inconsistent_offset_builder.add_conversion(
      b_key,
      a_key,
      AffineClockConversion{.scale = {.numerator = 3, .denominator = 2}, .offset_ns = -5});
  ASSERT_FALSE(inconsistent_offset);
  EXPECT_EQ(inconsistent_offset.error().code, ErrorCode::kIncompatible);

  detail::ClockRegistryBuilder cycle_builder;
  const auto c_key = ClockKey::named("c");
  const AffineClockConversion identity{.scale = {.numerator = 1, .denominator = 1}, .offset_ns = 0};
  for (const auto& key : {a_key, b_key, c_key}) {
    ASSERT_TRUE(
        cycle_builder.add_clock(key, std::make_shared<ManualClock>(), readable_capabilities()));
  }
  ASSERT_TRUE(cycle_builder.set_default(a_key));
  ASSERT_TRUE(cycle_builder.add_conversion(a_key, b_key, identity));
  ASSERT_TRUE(cycle_builder.add_conversion(b_key, c_key, identity));
  const auto closes_larger_cycle = cycle_builder.add_conversion(c_key, a_key, identity);
  if (closes_larger_cycle) {
    auto cyclic = cycle_builder.build();
    ASSERT_FALSE(cyclic);
    EXPECT_EQ(cyclic.error().code, ErrorCode::kIncompatible);
  } else {
    EXPECT_EQ(closes_larger_cycle.error().code, ErrorCode::kIncompatible);
  }
}

// REQ: CLOCK-013
// REQ: CLOCK-015
// REQ: CLOCK-022
TEST(ClockConversion, RejectsInvalidConflictingAndOverflowingMappings) {
  const auto source_key = ClockKey::named("source");
  const auto target_key = ClockKey::named("target");

  detail::ClockRegistryBuilder invalid_builder;
  ASSERT_TRUE(invalid_builder.add_clock(
      source_key, std::make_shared<ManualClock>(), readable_capabilities()));
  ASSERT_TRUE(invalid_builder.add_clock(
      target_key, std::make_shared<ManualClock>(), readable_capabilities()));
  ASSERT_TRUE(invalid_builder.set_default(source_key));
  auto zero_numerator = invalid_builder.add_conversion(
      source_key,
      target_key,
      AffineClockConversion{.scale = {.numerator = 0, .denominator = 1}, .offset_ns = 0});
  ASSERT_FALSE(zero_numerator);
  EXPECT_EQ(zero_numerator.error().code, ErrorCode::kInvalidArgument);
  auto zero_denominator = invalid_builder.add_conversion(
      source_key,
      target_key,
      AffineClockConversion{.scale = {.numerator = 1, .denominator = 0}, .offset_ns = 0});
  ASSERT_FALSE(zero_denominator);
  EXPECT_EQ(zero_denominator.error().code, ErrorCode::kInvalidArgument);
  auto negative_numerator = invalid_builder.add_conversion(
      source_key,
      target_key,
      AffineClockConversion{.scale = {.numerator = -1, .denominator = 1}, .offset_ns = 0});
  ASSERT_FALSE(negative_numerator);
  EXPECT_EQ(negative_numerator.error().code, ErrorCode::kInvalidArgument);
  auto self_edge = invalid_builder.add_conversion(
      source_key,
      source_key,
      AffineClockConversion{.scale = {.numerator = 1, .denominator = 1}, .offset_ns = 0});
  ASSERT_FALSE(self_edge);
  EXPECT_EQ(self_edge.error().code, ErrorCode::kIncompatible);

  detail::ClockRegistryBuilder conflict_builder;
  ASSERT_TRUE(conflict_builder.add_clock(
      source_key, std::make_shared<ManualClock>(), readable_capabilities()));
  ASSERT_TRUE(conflict_builder.add_clock(
      target_key, std::make_shared<ManualClock>(), readable_capabilities()));
  ASSERT_TRUE(conflict_builder.set_default(source_key));
  ASSERT_TRUE(conflict_builder.add_conversion(
      source_key,
      target_key,
      AffineClockConversion{.scale = {.numerator = 1, .denominator = 2}, .offset_ns = 0}));
  auto conflict = conflict_builder.add_conversion(
      source_key,
      target_key,
      AffineClockConversion{.scale = {.numerator = 2, .denominator = 3}, .offset_ns = 0});
  ASSERT_FALSE(conflict);
  EXPECT_EQ(conflict.error().code, ErrorCode::kIncompatible);

  auto overflow_registry_result = make_conversion_registry(
      AffineClockConversion{.scale = {.numerator = 2, .denominator = 1}, .offset_ns = 0});
  ASSERT_TRUE(overflow_registry_result);
  auto overflow_registry = std::move(overflow_registry_result).value();
  auto source = overflow_registry.resolve(source_key);
  auto target = overflow_registry.resolve(target_key);
  ASSERT_TRUE(source);
  ASSERT_TRUE(target);
  auto overflow = overflow_registry.convert(source->at_ns(std::numeric_limits<std::int64_t>::max()),
                                            target->id());
  ASSERT_FALSE(overflow);
  EXPECT_EQ(overflow.error().code, ErrorCode::kInvalidArgument);

  auto invalid_source = overflow_registry.convert(TimePoint{1, ClockId{}}, target->id());
  ASSERT_FALSE(invalid_source);
  EXPECT_EQ(invalid_source.error().code, ErrorCode::kInvalidArgument);

  auto unknown_source = overflow_registry.convert(TimePoint{1, ClockId{999}}, target->id());
  ASSERT_FALSE(unknown_source);
  EXPECT_EQ(unknown_source.error().code, ErrorCode::kNotFound);

  auto unknown_key = overflow_registry.resolve(ClockKey::named("missing"));
  ASSERT_FALSE(unknown_key);
  EXPECT_EQ(unknown_key.error().code, ErrorCode::kNotFound);

  auto invalid_target = overflow_registry.convert(source->at_ns(1), ClockId{});
  ASSERT_FALSE(invalid_target);
  EXPECT_EQ(invalid_target.error().code, ErrorCode::kInvalidArgument);

  auto unknown_target = overflow_registry.convert(source->at_ns(1), ClockId{999});
  ASSERT_FALSE(unknown_target);
  EXPECT_EQ(unknown_target.error().code, ErrorCode::kNotFound);

  auto invalid_compare =
      overflow_registry.compare(TimePoint{1, ClockId{}}, target->at_ns(1), target->id());
  ASSERT_FALSE(invalid_compare);
  EXPECT_EQ(invalid_compare.error().code, ErrorCode::kInvalidArgument);

  auto unknown_compare =
      overflow_registry.compare(source->at_ns(1), TimePoint{1, ClockId{999}}, target->id());
  ASSERT_FALSE(unknown_compare);
  EXPECT_EQ(unknown_compare.error().code, ErrorCode::kNotFound);
}

}  // namespace
}  // namespace holoscan
