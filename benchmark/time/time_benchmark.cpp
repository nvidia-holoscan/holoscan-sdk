/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmark/benchmark.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/time/clock_registry.hpp>
#include <holoscan/time/manual_clock.hpp>

#include <time/clock_registry_internal.hpp>

namespace {

using holoscan::AffineClockConversion;
using holoscan::ClockKey;
using holoscan::ManualClock;
using holoscan::detail::ClockCapabilityOptions;
using holoscan::detail::ClockRegistry;
using holoscan::detail::ClockRegistryBuilder;

struct BenchmarkRegistry {
  ClockRegistry registry;
  ClockKey source_key{ClockKey::named("source")};
  ClockKey target_key{ClockKey::named("target")};
};

[[nodiscard]] holoscan::expected<BenchmarkRegistry, holoscan::Error> make_registry() {
  ClockRegistryBuilder builder;
  const ClockKey source_key = ClockKey::named("source");
  const ClockKey target_key = ClockKey::named("target");
  const ClockCapabilityOptions readable{
      .can_read_now = true,
      .can_wait_until = true,
      .steady = true,
  };
  auto configured = builder.add_clock(source_key, std::make_shared<ManualClock>(123), readable);
  if (!configured) {
    return holoscan::forward_error(std::move(configured));
  }
  configured = builder.add_clock(target_key, std::make_shared<ManualClock>(456), readable);
  if (!configured) {
    return holoscan::forward_error(std::move(configured));
  }
  configured = builder.set_default(source_key);
  if (!configured) {
    return holoscan::forward_error(std::move(configured));
  }
  configured = builder.add_conversion(source_key,
                                      target_key,
                                      AffineClockConversion{
                                          .scale = {.numerator = 3, .denominator = 2},
                                          .offset_ns = 7,
                                      });
  if (!configured) {
    return holoscan::forward_error(std::move(configured));
  }
  auto registry = builder.build();
  if (!registry) {
    return holoscan::forward_error(std::move(registry));
  }
  return BenchmarkRegistry{
      .registry = std::move(registry).value(),
      .source_key = source_key,
      .target_key = target_key,
  };
}

// Cold declaration-name resolution intentionally remains separate from every
// resolved-handle and conversion measurement.
// REQ: CLOCK-012
void BM_ClockColdResolution(benchmark::State& state) {
  auto fixture = make_registry();
  if (!fixture) {
    const std::string message = holoscan::runtime_error_message(fixture.error());
    state.SkipWithError(message.c_str());
    return;
  }
  for (auto _ : state) {
    auto resolved = fixture->registry.resolve(fixture->source_key);
    benchmark::DoNotOptimize(resolved);
  }
}

// REQ: CLOCK-023
void BM_ClockResolvedNow(benchmark::State& state) {
  auto fixture = make_registry();
  if (!fixture) {
    const std::string message = holoscan::runtime_error_message(fixture.error());
    state.SkipWithError(message.c_str());
    return;
  }
  auto source = fixture->registry.resolve(fixture->source_key);
  if (!source) {
    state.SkipWithError("source clock did not resolve");
    return;
  }
  for (auto _ : state) {
    auto now = source->now();
    benchmark::DoNotOptimize(now);
  }
}

// REQ: CLOCK-015
void BM_ClockDirectConversion(benchmark::State& state) {
  auto fixture = make_registry();
  if (!fixture) {
    const std::string message = holoscan::runtime_error_message(fixture.error());
    state.SkipWithError(message.c_str());
    return;
  }
  auto source = fixture->registry.resolve(fixture->source_key);
  auto target = fixture->registry.resolve(fixture->target_key);
  if (!source || !target) {
    state.SkipWithError("benchmark clocks did not resolve");
    return;
  }

  std::int64_t coordinate = 0;
  for (auto _ : state) {
    auto converted = fixture->registry.convert(source->at_ns(coordinate), target->id());
    benchmark::DoNotOptimize(converted);
    ++coordinate;
  }
}

}  // namespace

BENCHMARK(BM_ClockColdResolution);
BENCHMARK(BM_ClockResolvedNow);
BENCHMARK(BM_ClockDirectConversion);
