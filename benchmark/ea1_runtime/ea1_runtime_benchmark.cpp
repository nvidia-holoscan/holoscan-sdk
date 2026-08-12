/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>

#include <benchmark/benchmark.h>

#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/time_point.hpp>

#include "runtime/temporal_input_cache.hpp"

namespace {

using namespace std::chrono_literals;

constexpr holoscan::ClockId kBenchmarkClock{1U};
constexpr std::uint64_t kProducerEpoch{1U};
constexpr std::size_t kFastBurstDepth{64U};

[[nodiscard]] constexpr holoscan::detail::TemporalSampleRecord sample(
    std::int64_t capture_time, std::uint64_t sequence) noexcept {
  return {
      .capture_time = holoscan::TimePoint{capture_time, kBenchmarkClock},
      .source_sequence = sequence,
      .producer_epoch = kProducerEpoch,
      .arrival_sequence = sequence,
  };
}

[[nodiscard]] bool populate(holoscan::detail::TemporalInputCache& cache) noexcept {
  for (std::size_t index = 1U; index <= kFastBurstDepth; ++index) {
    const auto sequence = static_cast<std::uint64_t>(index);
    if (!cache.push(sample(static_cast<std::int64_t>(sequence), sequence))) {
      return false;
    }
  }
  return true;
}

// REQ: SYSTEM-005
void BM_EA1_S2MultiRateFresh(benchmark::State& state) {
  holoscan::detail::TemporalInputCache cache{kFastBurstDepth};
  if (!populate(cache)) {
    state.SkipWithError("failed to prepare the S2 temporal cache");
    return;
  }
  const holoscan::TimePoint reference{static_cast<std::int64_t>(kFastBurstDepth), kBenchmarkClock};
  for (auto _ : state) {
    auto result = cache.select_latest_within(reference, 1ns);
    benchmark::DoNotOptimize(result);
  }
  state.counters["fast_burst_depth"] = static_cast<double>(kFastBurstDepth);
  state.SetItemsProcessed(state.iterations());
}

// REQ: SYSTEM-005
void BM_EA1_S2MultiRateStalledSkip(benchmark::State& state) {
  holoscan::detail::TemporalInputCache cache{kFastBurstDepth};
  if (!populate(cache)) {
    state.SkipWithError("failed to prepare the S2 temporal cache");
    return;
  }
  const holoscan::TimePoint stalled_reference{static_cast<std::int64_t>(kFastBurstDepth + 100U),
                                              kBenchmarkClock};
  for (auto _ : state) {
    auto selected = cache.select_latest_within(stalled_reference, 1ns);
    auto result = holoscan::detail::apply_scalar_missing_policy(
        selected,
        std::variant<holoscan::Skip, holoscan::SafeDefault, holoscan::ReuseLast>{holoscan::Skip{}},
        std::nullopt,
        kProducerEpoch,
        stalled_reference);
    benchmark::DoNotOptimize(result);
  }
  state.counters["fast_burst_depth"] = static_cast<double>(kFastBurstDepth);
  state.SetItemsProcessed(state.iterations());
}

// REQ: SYSTEM-005
void BM_EA1_S2MultiRateStalledReuseLast(benchmark::State& state) {
  holoscan::detail::TemporalInputCache cache{kFastBurstDepth};
  if (!populate(cache)) {
    state.SkipWithError("failed to prepare the S2 temporal cache");
    return;
  }
  const auto retained = sample(static_cast<std::int64_t>(kFastBurstDepth), kFastBurstDepth);
  const holoscan::TimePoint stalled_reference{static_cast<std::int64_t>(kFastBurstDepth + 100U),
                                              kBenchmarkClock};
  for (auto _ : state) {
    auto selected = cache.select_latest_within(stalled_reference, 1ns);
    auto result = holoscan::detail::apply_scalar_missing_policy(
        selected,
        std::variant<holoscan::Skip, holoscan::SafeDefault, holoscan::ReuseLast>{
            holoscan::ReuseLast{200ns}},
        std::optional<holoscan::detail::TemporalSampleRecord>{retained},
        kProducerEpoch,
        stalled_reference);
    benchmark::DoNotOptimize(result);
  }
  state.counters["fast_burst_depth"] = static_cast<double>(kFastBurstDepth);
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK(BM_EA1_S2MultiRateFresh);
BENCHMARK(BM_EA1_S2MultiRateStalledSkip);
BENCHMARK(BM_EA1_S2MultiRateStalledReuseLast);

}  // namespace
