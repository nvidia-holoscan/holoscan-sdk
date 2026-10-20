/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include "runtime/temporal_input_cache.hpp"

namespace {

using namespace std::chrono_literals;

constexpr holoscan::ClockId kClock{1};
constexpr std::uint64_t kProducerEpoch{1};
constexpr std::size_t kFixtureBatchRecordBudget{32U * 1024U};
constexpr std::size_t kMinFixtureBatchSize{32U};
constexpr std::size_t kMaxFixtureBatchSize{128U};

[[nodiscard]] constexpr holoscan::detail::TemporalSampleRecord sample(
    std::int64_t timestamp, std::uint64_t sequence) noexcept {
  return {
      .capture_time = holoscan::TimePoint{timestamp, kClock},
      .source_sequence = sequence,
      .producer_epoch = kProducerEpoch,
      .arrival_sequence = sequence,
  };
}

[[nodiscard]] std::size_t depth_for(const benchmark::State& state) noexcept {
  return static_cast<std::size_t>(state.range(0));
}

[[nodiscard]] std::size_t fixture_batch_size(std::size_t depth) noexcept {
  const auto bounded_depth = std::max<std::size_t>(depth, 1U);
  return std::clamp(
      kFixtureBatchRecordBudget / bounded_depth, kMinFixtureBatchSize, kMaxFixtureBatchSize);
}

template <typename Fixture, typename Factory>
[[nodiscard]] bool fill_fixture_batch(std::vector<std::unique_ptr<Fixture>>& fixtures,
                                      std::size_t batch_size, Factory& factory) {
  fixtures.clear();
  fixtures.reserve(batch_size);
  for (std::size_t index = 0; index < batch_size; ++index) {
    auto fixture = factory();
    if (!fixture) {
      fixtures.clear();
      return false;
    }
    fixtures.push_back(std::move(fixture));
  }
  return true;
}

template <typename Fixture, typename Validator>
[[nodiscard]] bool validate_fixture_batch(const std::vector<std::unique_ptr<Fixture>>& fixtures,
                                          std::size_t count, Validator& validator) {
  return std::all_of(fixtures.begin(),
                     fixtures.begin() + static_cast<std::ptrdiff_t>(count),
                     [&](const auto& fixture) { return validator(*fixture); });
}

template <typename Fixture, typename Factory, typename Operation, typename Validator>
[[nodiscard]] bool run_batched_fixtures(benchmark::State& state, std::size_t depth, Factory factory,
                                        Operation operation, Validator validator,
                                        const char* preparation_error, const char* operation_error,
                                        const char* validation_error) {
  const std::size_t batch_size = fixture_batch_size(depth);
  std::vector<std::unique_ptr<Fixture>> fixtures;
  if (!fill_fixture_batch(fixtures, batch_size, factory)) {
    state.SkipWithError(preparation_error);
    return false;
  }
  state.counters["fixture_batch_size"] = static_cast<double>(batch_size);

  std::size_t fixture_index = 0;
  for (auto _ : state) {
    if (fixture_index == fixtures.size()) {
      state.PauseTiming();
      const bool valid = validate_fixture_batch(fixtures, fixture_index, validator);
      const bool refilled = valid && fill_fixture_batch(fixtures, batch_size, factory);
      state.ResumeTiming();
      if (!valid) {
        state.SkipWithError(validation_error);
        return false;
      }
      if (!refilled) {
        state.SkipWithError(preparation_error);
        return false;
      }
      fixture_index = 0;
    }
    if (!operation(*fixtures[fixture_index])) {
      state.SkipWithError(operation_error);
      return false;
    }
    ++fixture_index;
  }

  if (!validate_fixture_batch(fixtures, fixture_index, validator)) {
    state.SkipWithError(validation_error);
    return false;
  }
  return true;
}

[[nodiscard]] bool populate_cache(holoscan::detail::TemporalInputCache& cache,
                                  std::size_t depth) noexcept {
  for (std::size_t index = 1; index <= depth; ++index) {
    const auto sequence = static_cast<std::uint64_t>(index);
    if (!cache.push(sample(static_cast<std::int64_t>(index), sequence))) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] std::unique_ptr<holoscan::detail::TemporalInputCache> make_cache(
    std::size_t depth, std::size_t additional_capacity = 0) {
  auto cache = std::make_unique<holoscan::detail::TemporalInputCache>(depth + additional_capacity);
  if (!populate_cache(*cache, depth)) {
    return nullptr;
  }
  return cache;
}

struct ScalarTransactionFixture {
  std::unique_ptr<holoscan::detail::TemporalInputCache> cache;
  std::optional<holoscan::detail::TemporalInputCache::ScalarTransaction> transaction;
};

[[nodiscard]] std::unique_ptr<ScalarTransactionFixture> make_scalar_transaction_fixture(
    std::size_t depth) {
  auto fixture = std::make_unique<ScalarTransactionFixture>();
  fixture->cache = make_cache(depth);
  if (!fixture->cache) {
    return nullptr;
  }
  auto candidate = fixture->cache->select_exact_transaction(holoscan::TimePoint{1, kClock});
  if (!candidate) {
    return nullptr;
  }
  fixture->transaction.emplace(std::move(*candidate));
  return fixture;
}

enum class DrainFixtureState : std::uint8_t {
  kCacheOnly,
  kActive,
  kActiveWithLaterArrival,
};

struct DrainTransactionFixture {
  std::unique_ptr<holoscan::detail::TemporalInputCache> cache;
  std::optional<holoscan::detail::TemporalInputCache::DrainTransaction> transaction;
};

[[nodiscard]] std::unique_ptr<DrainTransactionFixture> make_drain_transaction_fixture(
    std::size_t depth, DrainFixtureState state, std::size_t additional_capacity = 0U) {
  auto fixture = std::make_unique<DrainTransactionFixture>();
  fixture->cache = make_cache(depth, additional_capacity);
  if (!fixture->cache) {
    return nullptr;
  }
  if (state == DrainFixtureState::kCacheOnly) {
    return fixture;
  }
  auto candidate = fixture->cache->snapshot_drain_all(depth, depth);
  if (!candidate) {
    return nullptr;
  }
  fixture->transaction.emplace(std::move(*candidate));
  if (state == DrainFixtureState::kActiveWithLaterArrival &&
      !fixture->cache->push(sample(static_cast<std::int64_t>(depth + 1U), depth + 1U))) {
    return nullptr;
  }
  return fixture;
}

enum class AlignmentPattern : std::uint8_t {
  kMatch,
  kMiss,
};

[[nodiscard]] std::unique_ptr<holoscan::detail::AlignedJoinEngine> make_aligned(
    std::size_t depth, AlignmentPattern pattern) {
  auto engine = std::make_unique<holoscan::detail::AlignedJoinEngine>(2, depth);
  for (std::size_t index = 1; index <= depth; ++index) {
    const auto sequence = static_cast<std::uint64_t>(index);
    const auto first_time = static_cast<std::int64_t>(index * 2U);
    const auto second_time = first_time + (pattern == AlignmentPattern::kMatch ? 0 : 1);
    if (!engine->push(0, sample(first_time, sequence)) ||
        !engine->push(1, sample(second_time, sequence))) {
      return nullptr;
    }
  }
  return engine;
}

struct AlignedTransactionFixture {
  std::unique_ptr<holoscan::detail::AlignedJoinEngine> engine;
  std::optional<holoscan::detail::AlignedJoinEngine::Transaction> transaction;
};

[[nodiscard]] std::unique_ptr<AlignedTransactionFixture> make_aligned_transaction_fixture(
    std::size_t depth, AlignmentPattern pattern, bool select_transaction) {
  auto fixture = std::make_unique<AlignedTransactionFixture>();
  fixture->engine = make_aligned(depth, pattern);
  if (!fixture->engine) {
    return nullptr;
  }
  if (!select_transaction) {
    return fixture;
  }
  const std::array missing{holoscan::detail::AlignedMissingPolicy{},
                           holoscan::detail::AlignedMissingPolicy{}};
  const std::array closure{holoscan::detail::AlignmentClosureProof::kOpen,
                           holoscan::detail::AlignmentClosureProof::kOpen};
  auto candidate = fixture->engine->select(0ns, missing, closure);
  if (!candidate) {
    return nullptr;
  }
  fixture->transaction.emplace(std::move(*candidate));
  return fixture;
}

[[nodiscard]] std::unique_ptr<holoscan::detail::BracketedWindowEngine> make_window(
    std::size_t depth, bool close_window) {
  auto engine = std::make_unique<holoscan::detail::BracketedWindowEngine>(
      holoscan::BracketedWindow{
          .lookbehind = 1,
          .lookahead = 1,
          .max_samples = static_cast<std::uint32_t>(depth + 2U),
      },
      depth + 2U,
      1U);
  if (!engine->push_sample(sample(1, 1)) || !engine->push_anchor(sample(2, 1))) {
    return nullptr;
  }
  for (std::size_t index = 0; index < depth; ++index) {
    const auto sequence = static_cast<std::uint64_t>(index + 2U);
    if (!engine->push_sample(sample(static_cast<std::int64_t>(index + 2U), sequence))) {
      return nullptr;
    }
  }
  if (!engine->push_anchor(sample(static_cast<std::int64_t>(depth + 2U), 2))) {
    return nullptr;
  }
  if (close_window && !engine->push_sample(sample(static_cast<std::int64_t>(depth + 3U),
                                                  static_cast<std::uint64_t>(depth + 2U)))) {
    return nullptr;
  }
  return engine;
}

void finish_depth_benchmark(benchmark::State& state, std::size_t items_per_iteration = 1U) {
  const auto depth = static_cast<std::int64_t>(depth_for(state));
  state.SetComplexityN(depth);
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(items_per_iteration));
}

// REQ: CONTRACT-014
void BM_TemporalCachePush(benchmark::State& state) {
  const std::size_t depth = depth_for(state);
  const bool completed = run_batched_fixtures<holoscan::detail::TemporalInputCache>(
      state,
      depth,
      [depth]() { return make_cache(depth, 1U); },
      [depth](auto& cache) {
        const auto result = cache.push(sample(static_cast<std::int64_t>(depth + 1U), depth + 1U));
        benchmark::DoNotOptimize(result.has_value());
        return result.has_value();
      },
      [depth](const auto& cache) { return cache.size() == depth + 1U; },
      "failed to prepare temporal cache batch",
      "temporal cache push failed",
      "temporal cache push changed unexpected state");
  if (!completed) {
    return;
  }
  finish_depth_benchmark(state);
}

// REQ: CONTRACT-004
void BM_TemporalCacheSelectExact(benchmark::State& state) {
  const std::size_t depth = depth_for(state);
  auto cache = make_cache(depth);
  if (!cache) {
    state.SkipWithError("failed to prepare temporal cache");
    return;
  }
  const holoscan::TimePoint reference{static_cast<std::int64_t>(depth), kClock};
  for (auto _ : state) {
    auto result = cache->select_exact(reference);
    benchmark::DoNotOptimize(result);
  }
  finish_depth_benchmark(state);
}

// REQ: CONTRACT-004
void BM_TemporalCacheSelectLatestWithin(benchmark::State& state) {
  const std::size_t depth = depth_for(state);
  auto cache = make_cache(depth);
  if (!cache) {
    state.SkipWithError("failed to prepare temporal cache");
    return;
  }
  const holoscan::TimePoint reference{static_cast<std::int64_t>(depth), kClock};
  for (auto _ : state) {
    auto result = cache->select_latest_within(reference, 1ns);
    benchmark::DoNotOptimize(result);
  }
  finish_depth_benchmark(state);
}

// REQ: CONTRACT-004
void BM_TemporalCacheScalarCancel(benchmark::State& state) {
  const std::size_t depth = depth_for(state);
  const bool completed = run_batched_fixtures<ScalarTransactionFixture>(
      state,
      depth,
      [depth]() { return make_scalar_transaction_fixture(depth); },
      [](auto& fixture) {
        if (!fixture.transaction) {
          return false;
        }
        fixture.transaction.reset();
        benchmark::ClobberMemory();
        return true;
      },
      [depth](const auto& fixture) {
        return !fixture.transaction && fixture.cache->size() == depth;
      },
      "failed to prepare scalar transaction batch",
      "scalar transaction cancellation failed",
      "scalar transaction cancellation changed the cache");
  if (!completed) {
    return;
  }
  finish_depth_benchmark(state);
}

// REQ: CONTRACT-004
void BM_TemporalCacheScalarCommitFront(benchmark::State& state) {
  const std::size_t depth = depth_for(state);
  const bool completed = run_batched_fixtures<ScalarTransactionFixture>(
      state,
      depth,
      [depth]() { return make_scalar_transaction_fixture(depth); },
      [](auto& fixture) {
        if (!fixture.transaction) {
          return false;
        }
        const auto result = fixture.transaction->commit();
        benchmark::DoNotOptimize(result.has_value());
        return result.has_value();
      },
      [depth](const auto& fixture) {
        return fixture.transaction.has_value() && fixture.cache->size() == depth - 1U;
      },
      "failed to prepare scalar transaction batch",
      "scalar commit failed",
      "scalar commit changed unexpected state");
  if (!completed) {
    return;
  }
  finish_depth_benchmark(state);
}

// REQ: CONTRACT-005
void BM_DrainAllSnapshot(benchmark::State& state) {
  const std::size_t depth = depth_for(state);
  const bool completed = run_batched_fixtures<DrainTransactionFixture>(
      state,
      depth,
      [depth]() { return make_drain_transaction_fixture(depth, DrainFixtureState::kCacheOnly); },
      [depth](auto& fixture) {
        auto candidate = fixture.cache->snapshot_drain_all(depth, depth);
        benchmark::DoNotOptimize(candidate.has_value());
        if (!candidate) {
          return false;
        }
        fixture.transaction.emplace(std::move(*candidate));
        return fixture.transaction->samples().size() == depth;
      },
      [depth](const auto& fixture) {
        return fixture.transaction.has_value() && fixture.cache->size() == depth;
      },
      "failed to prepare DrainAll snapshot batch",
      "DrainAll snapshot failed",
      "DrainAll snapshot changed unexpected state");
  if (!completed) {
    return;
  }
  finish_depth_benchmark(state, depth);
}

// REQ: CONTRACT-005
void BM_DrainAllCancel(benchmark::State& state) {
  const std::size_t depth = depth_for(state);
  const bool completed = run_batched_fixtures<DrainTransactionFixture>(
      state,
      depth,
      [depth]() { return make_drain_transaction_fixture(depth, DrainFixtureState::kActive); },
      [](auto& fixture) {
        if (!fixture.transaction) {
          return false;
        }
        fixture.transaction.reset();
        benchmark::ClobberMemory();
        return true;
      },
      [depth](const auto& fixture) {
        return !fixture.transaction && fixture.cache->size() == depth;
      },
      "failed to prepare DrainAll transaction batch",
      "DrainAll cancellation failed",
      "DrainAll cancellation changed the cache");
  if (!completed) {
    return;
  }
  finish_depth_benchmark(state, depth);
}

// REQ: CONTRACT-005
void BM_DrainAllCommit(benchmark::State& state) {
  const std::size_t depth = depth_for(state);
  const bool completed = run_batched_fixtures<DrainTransactionFixture>(
      state,
      depth,
      [depth]() { return make_drain_transaction_fixture(depth, DrainFixtureState::kActive); },
      [](auto& fixture) {
        if (!fixture.transaction) {
          return false;
        }
        const auto result = fixture.transaction->commit();
        benchmark::DoNotOptimize(result.has_value());
        return result.has_value();
      },
      [](const auto& fixture) {
        return fixture.transaction.has_value() && fixture.cache->size() == 0U;
      },
      "failed to prepare DrainAll transaction batch",
      "DrainAll commit failed",
      "DrainAll commit changed unexpected state");
  if (!completed) {
    return;
  }
  finish_depth_benchmark(state, depth);
}

// REQ: CONTRACT-005
void BM_DrainAllProvisionalPush(benchmark::State& state) {
  const std::size_t depth = depth_for(state);
  const bool completed = run_batched_fixtures<DrainTransactionFixture>(
      state,
      depth,
      [depth]() { return make_drain_transaction_fixture(depth, DrainFixtureState::kActive, 1U); },
      [depth](auto& fixture) {
        const auto result =
            fixture.cache->push(sample(static_cast<std::int64_t>(depth + 1U), depth + 1U));
        benchmark::DoNotOptimize(result.has_value());
        return result.has_value();
      },
      [depth](const auto& fixture) {
        return fixture.transaction.has_value() && fixture.cache->size() == depth + 1U;
      },
      "failed to prepare provisional DrainAll batch",
      "provisional DrainAll arrival failed",
      "provisional DrainAll arrival changed unexpected state");
  if (!completed) {
    return;
  }
  finish_depth_benchmark(state);
}

// REQ: CONTRACT-005
void BM_DrainAllCommitWithLaterArrival(benchmark::State& state) {
  const std::size_t depth = depth_for(state);
  const bool completed = run_batched_fixtures<DrainTransactionFixture>(
      state,
      depth,
      [depth]() {
        return make_drain_transaction_fixture(
            depth, DrainFixtureState::kActiveWithLaterArrival, 1U);
      },
      [](auto& fixture) {
        if (!fixture.transaction) {
          return false;
        }
        const auto result = fixture.transaction->commit();
        benchmark::DoNotOptimize(result.has_value());
        return result.has_value();
      },
      [](const auto& fixture) {
        return fixture.transaction.has_value() && fixture.cache->size() == 1U;
      },
      "failed to prepare DrainAll batch with later arrival",
      "DrainAll commit with later arrival failed",
      "DrainAll commit with later arrival changed unexpected state");
  if (!completed) {
    return;
  }
  finish_depth_benchmark(state, depth);
}

void run_aligned_select(benchmark::State& state, AlignmentPattern pattern) {
  const std::size_t depth = depth_for(state);
  const std::array missing{holoscan::detail::AlignedMissingPolicy{},
                           holoscan::detail::AlignedMissingPolicy{}};
  const std::array closure{holoscan::detail::AlignmentClosureProof::kOpen,
                           holoscan::detail::AlignmentClosureProof::kOpen};
  const bool completed = run_batched_fixtures<AlignedTransactionFixture>(
      state,
      depth,
      [depth, pattern]() { return make_aligned_transaction_fixture(depth, pattern, false); },
      [&](auto& fixture) {
        auto candidate = fixture.engine->select(0ns, missing, closure);
        benchmark::DoNotOptimize(candidate.has_value());
        if (!candidate) {
          return false;
        }
        fixture.transaction.emplace(std::move(*candidate));
        return true;
      },
      [depth](const auto& fixture) {
        return fixture.transaction.has_value() && fixture.engine->queue_size(0) == depth &&
               fixture.engine->queue_size(1) == depth;
      },
      "failed to prepare aligned selection batch",
      "aligned selection failed",
      "aligned selection changed unexpected state");
  if (!completed) {
    return;
  }
  finish_depth_benchmark(state);
}

void run_aligned_commit(benchmark::State& state, AlignmentPattern pattern) {
  const std::size_t depth = depth_for(state);
  const bool completed = run_batched_fixtures<AlignedTransactionFixture>(
      state,
      depth,
      [depth, pattern]() { return make_aligned_transaction_fixture(depth, pattern, true); },
      [](auto& fixture) {
        if (!fixture.transaction) {
          return false;
        }
        const auto result = fixture.transaction->commit();
        benchmark::DoNotOptimize(result.has_value());
        return result.has_value();
      },
      [depth, pattern](const auto& fixture) {
        const std::size_t expected_first = depth - 1U;
        const std::size_t expected_second =
            pattern == AlignmentPattern::kMatch ? depth - 1U : depth;
        return fixture.transaction.has_value() && fixture.engine->queue_size(0) == expected_first &&
               fixture.engine->queue_size(1) == expected_second;
      },
      "failed to prepare aligned transaction batch",
      "aligned commit failed",
      "aligned commit changed unexpected state");
  if (!completed) {
    return;
  }
  finish_depth_benchmark(state, pattern == AlignmentPattern::kMatch ? 2U : 1U);
}

// REQ: CONTRACT-006
void BM_AlignedMatchSelect(benchmark::State& state) {
  run_aligned_select(state, AlignmentPattern::kMatch);
}

// REQ: CONTRACT-006
void BM_AlignedMissSelect(benchmark::State& state) {
  run_aligned_select(state, AlignmentPattern::kMiss);
}

// REQ: CONTRACT-006
void BM_AlignedMatchCommit(benchmark::State& state) {
  run_aligned_commit(state, AlignmentPattern::kMatch);
}

// REQ: CONTRACT-006
void BM_AlignedMissCommit(benchmark::State& state) {
  run_aligned_commit(state, AlignmentPattern::kMiss);
}

// REQ: CONTRACT-007
void BM_BracketedWindowClose(benchmark::State& state) {
  const std::size_t depth = depth_for(state);
  const bool completed = run_batched_fixtures<holoscan::detail::BracketedWindowEngine>(
      state,
      depth,
      [depth]() { return make_window(depth, false); },
      [depth](auto& engine) {
        const auto result =
            engine.push_sample(sample(static_cast<std::int64_t>(depth + 3U), depth + 2U));
        benchmark::DoNotOptimize(result.has_value());
        return result.has_value();
      },
      [](const auto& engine) { return engine.front_closed() != nullptr; },
      "failed to prepare bracketed-window close batch",
      "bracketed-window close failed",
      "bracketed-window close changed unexpected state");
  if (!completed) {
    return;
  }
  finish_depth_benchmark(state, depth + 2U);
}

// REQ: CONTRACT-007
void BM_BracketedWindowPopAndPrune(benchmark::State& state) {
  const std::size_t depth = depth_for(state);
  const bool completed = run_batched_fixtures<holoscan::detail::BracketedWindowEngine>(
      state,
      depth,
      [depth]() { return make_window(depth, true); },
      [](auto& engine) {
        if (engine.front_closed() == nullptr) {
          return false;
        }
        engine.pop_closed();
        benchmark::ClobberMemory();
        return true;
      },
      [](const auto& engine) { return engine.front_closed() == nullptr; },
      "failed to prepare bracketed-window pop batch",
      "bracketed-window pop failed",
      "bracketed-window pop changed unexpected state");
  if (!completed) {
    return;
  }
  finish_depth_benchmark(state, depth + 2U);
}

#define HOLOSCAN_TEMPORAL_DEPTHS(complexity) \
  ->ArgName("depth")                         \
      ->Arg(1)                               \
      ->Arg(2)                               \
      ->Arg(4)                               \
      ->Arg(8)                               \
      ->Arg(16)                              \
      ->Arg(64)                              \
      ->Arg(256)                             \
      ->Arg(1024)                            \
      ->Complexity(complexity)

BENCHMARK(BM_TemporalCachePush) HOLOSCAN_TEMPORAL_DEPTHS(benchmark::o1);
BENCHMARK(BM_TemporalCacheSelectExact) HOLOSCAN_TEMPORAL_DEPTHS(benchmark::oN);
BENCHMARK(BM_TemporalCacheSelectLatestWithin) HOLOSCAN_TEMPORAL_DEPTHS(benchmark::oN);
BENCHMARK(BM_TemporalCacheScalarCancel) HOLOSCAN_TEMPORAL_DEPTHS(benchmark::o1);
BENCHMARK(BM_TemporalCacheScalarCommitFront) HOLOSCAN_TEMPORAL_DEPTHS(benchmark::oN);
BENCHMARK(BM_DrainAllSnapshot) HOLOSCAN_TEMPORAL_DEPTHS(benchmark::oN);
BENCHMARK(BM_DrainAllCancel) HOLOSCAN_TEMPORAL_DEPTHS(benchmark::o1);
BENCHMARK(BM_DrainAllCommit) HOLOSCAN_TEMPORAL_DEPTHS(benchmark::oN);
BENCHMARK(BM_DrainAllProvisionalPush) HOLOSCAN_TEMPORAL_DEPTHS(benchmark::o1);
BENCHMARK(BM_DrainAllCommitWithLaterArrival) HOLOSCAN_TEMPORAL_DEPTHS(benchmark::oN);
BENCHMARK(BM_AlignedMatchSelect) HOLOSCAN_TEMPORAL_DEPTHS(benchmark::o1);
BENCHMARK(BM_AlignedMissSelect) HOLOSCAN_TEMPORAL_DEPTHS(benchmark::o1);
BENCHMARK(BM_AlignedMatchCommit) HOLOSCAN_TEMPORAL_DEPTHS(benchmark::oN);
BENCHMARK(BM_AlignedMissCommit) HOLOSCAN_TEMPORAL_DEPTHS(benchmark::oN);
BENCHMARK(BM_BracketedWindowClose) HOLOSCAN_TEMPORAL_DEPTHS(benchmark::oN);
BENCHMARK(BM_BracketedWindowPopAndPrune) HOLOSCAN_TEMPORAL_DEPTHS(benchmark::oN);

#undef HOLOSCAN_TEMPORAL_DEPTHS

}  // namespace
