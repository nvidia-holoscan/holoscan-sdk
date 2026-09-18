/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <linux/futex.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include <benchmark/benchmark.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/clock.hpp>

#include "core/source_id_internal.hpp"
#include "runtime/dispatcher.hpp"
#include "runtime/execution_lane.hpp"
#include "runtime/physical_readiness_binding.hpp"
#include "runtime/port_binding.hpp"
#include "runtime/source_wait.hpp"
#include "support/direct_port_binding.hpp"

namespace {

constexpr std::uint32_t kSteadyActivationCount = 65'536U;
constexpr std::uint32_t kCompletionDrainMaximumComputeCalls = 1024U;
constexpr std::uint32_t kMixedCompletionUnruledCallbackCount = 1024U;
constexpr std::uint32_t kTensorActivationCount = 8192U;
constexpr std::int64_t kParkedLatencyIterations = 1024;

class BenchmarkClock final : public holoscan::Clock {
 public:
  [[nodiscard]] double time() const override { return holoscan::timestamp_to_time(timestamp_); }
  [[nodiscard]] std::int64_t timestamp() const override { return timestamp_; }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> sleep_for(
      std::int64_t duration_ns) override {
    timestamp_ += duration_ns;
    return {};
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> sleep_until(
      std::int64_t target_time_ns) override {
    timestamp_ = target_time_ns;
    return {};
  }

 private:
  std::int64_t timestamp_{};
};

class BenchmarkOperator final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.spin_source(spin_, "activation").capacity(1U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnSpin{.event = spin_});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

 private:
  holoscan::SpinSource spin_;
};

template <std::size_t QueryCount>
class BenchmarkFixedRateQueryOperator final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.spin_source(spin_, "activation").capacity(1U);
    spec.output(output_, "output").nominal_rate(holoscan::Rate::hz(60U));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnSpin{.event = spin_});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    std::uint64_t checksum{};
    for (std::size_t index = 0U; index < QueryCount; ++index) {
      auto view = context.rate(output_);
      if (!view) {
        return holoscan::make_unexpected(std::move(view).error());
      }
      if (view->resolved_nominal_rate != holoscan::Rate::hz(60U)) {
        return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
      }
      checksum += view->rate_generation;
    }
    benchmark::DoNotOptimize(checksum);
    return {};
  }

 private:
  holoscan::SpinSource spin_;
  holoscan::Output<std::uint32_t> output_;
};

class BenchmarkComputeCallCompletionOperator final : public holoscan::Operator<> {
 public:
  explicit BenchmarkComputeCallCompletionOperator(std::optional<std::uint64_t> completion_limit)
      : completion_limit_(completion_limit) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.spin_source(spin_, "compute-call-completion").capacity(1U);
    spec.output(output_, "output");
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnSpin{.event = spin_});
    if (completion_limit_) {
      result.complete_after(holoscan::ComputeCalls{.limit = *completion_limit_});
    }
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Output<std::uint32_t> output_;

 private:
  std::optional<std::uint64_t> completion_limit_;
  holoscan::SpinSource spin_;
};

class BenchmarkCompletionChannelSource final : public holoscan::Operator<> {
 public:
  explicit BenchmarkCompletionChannelSource(std::uint64_t limit) : limit_(limit) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.spin_source(spin_, "compute-call-completion").capacity(1U);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnSpin{.event = spin_})
        .complete_after(holoscan::ComputeCalls{.limit = limit_});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return output.emit(7U);
  }

  holoscan::Output<std::uint32_t> output;

 private:
  std::uint64_t limit_{};
  holoscan::SpinSource spin_;
};

class BenchmarkCompletionChannelSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input").queue_depth(64U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto received = input.receive();
    if (!received) {
      return holoscan::make_unexpected(std::move(received).error());
    }
    benchmark::DoNotOptimize(received->data);
    calls.fetch_add(1U, std::memory_order_relaxed);
    return {};
  }

  static inline std::atomic<std::uint64_t> calls{};
  holoscan::Input<std::uint32_t> input;
};

class BenchmarkMixedCompletionBurstSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.spin_source(spin_, "mixed-completion-burst").capacity(1U);
    spec.output(output, "output").max_emits_per_compute(kMixedCompletionUnruledCallbackCount);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnSpin{.event = spin_})
        .complete_after(holoscan::ComputeCalls{.limit = 1U});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    for (std::uint32_t index = 0U; index < kMixedCompletionUnruledCallbackCount; ++index) {
      auto emitted = output.emit(index);
      if (!emitted) {
        return holoscan::make_unexpected(std::move(emitted).error());
      }
    }
    calls.fetch_add(1U, std::memory_order_relaxed);
    return {};
  }

  static inline std::atomic<std::uint64_t> calls{};
  holoscan::Output<std::uint32_t> output;

 private:
  holoscan::SpinSource spin_;
};

class BenchmarkMixedCompletionUnruledSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(kMixedCompletionUnruledCallbackCount);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto received = input.receive();
    if (!received) {
      return holoscan::make_unexpected(std::move(received).error());
    }
    benchmark::DoNotOptimize(received->data);
    calls.fetch_add(1U, std::memory_order_relaxed);
    return {};
  }

  static inline std::atomic<std::uint64_t> calls{};
  holoscan::Input<std::uint32_t> input;
};

struct BenchmarkParams {
  std::int32_t value{7};
};

HOLOSCAN_PARAM_SCHEMA(BenchmarkParams,
                      holoscan::param_schema("nvidia.holoscan.benchmark.RuntimeDispatcherParams",
                                             1),
                      holoscan::field<&BenchmarkParams::value>("value").runtime_mutable());

class BenchmarkParameterOperator final : public holoscan::Operator<BenchmarkParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.spin_source(spin_, "activation").capacity(1U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnSpin{.event = spin_});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto value = context.params<BenchmarkParams>().value;
    benchmark::DoNotOptimize(value);
    return {};
  }

 private:
  holoscan::SpinSource spin_;
};

template <std::uint32_t kBurstCapacity>
class BenchmarkTensorOperator final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.spin_source(spin_, "activation").capacity(1U);
    spec.output(output_, "output")
        .max_emits_per_compute(kBurstCapacity)
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = holoscan::MemoryKind::kHost,
                    .dtype = DLDataType{kDLUInt, 64U, 1U},
                },
            .bounds = holoscan::TensorBounds{.max_rank = 1U, .max_byte_span = 8U},
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnSpin{.event = spin_});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto loan = output_.allocate_tensor(holoscan::TensorLoanRequest{
        .shape = holoscan::TensorShapeView{},
        .dtype = DLDataType{kDLUInt, 64U, 1U},
    });
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan.error()));
    }
    auto writer = loan->write_host();
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer.error()));
    }
    benchmark::DoNotOptimize(writer->data());
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return committed;
    }
    return std::move(*loan).cancel();
  }

 private:
  holoscan::SpinSource spin_;
  holoscan::Output<holoscan::Tensor> output_;
};

[[nodiscard]] holoscan::SchedulerPlanRecord scheduler_plan() {
  return {
      .ready_policy = holoscan::AlgorithmKey{"holoscan.ready.deterministic_serial", 1U},
      .execution_lane_count = 1U,
      .max_concurrent_activations = 1U,
      .activation_overlap = false,
      .worker_count = 1U,
      .intra_path_parallelism = 1U,
  };
}

[[nodiscard]] std::vector<holoscan::detail::DispatcherCandidate> make_candidates(
    std::size_t count) {
  std::vector<holoscan::detail::DispatcherCandidate> result;
  result.reserve(count);
  for (std::size_t index = 0U; index < count; ++index) {
    const auto one_based = static_cast<std::uint32_t>(index + 1U);
    const holoscan::detail::scheduling::ReadyCandidateView ready{
        .partition = holoscan::ExecutionPartitionId{1U},
        .execution_lane = 0U,
        .vertex = holoscan::VertexId{one_based},
        .tag_time = holoscan::TagTime{static_cast<std::int64_t>(count - index), 0U, 0U},
        .plan_vertex_order = one_based,
        .stable_source_id = holoscan::detail::make_source_id(one_based),
        .stable_port_id = holoscan::PortId{one_based},
        .stable_candidate_tie = one_based,
        .active = true,
    };
    result.push_back({
        .ready = ready,
        .tag = holoscan::Tag{static_cast<std::uint64_t>(index + 1U), 1U, 0U},
        .receiver_index = index,
    });
  }
  return result;
}

[[nodiscard]] holoscan::detail::RuntimeSourceBindingRecord make_runtime_source(std::size_t index) {
  constexpr holoscan::ClockId kReferenceClock{23U};
  const auto one_based = static_cast<std::uint32_t>(index + 1U);
  return {
      .candidate_kind = holoscan::detail::RuntimeEventKind::kInputArrival,
      .source = holoscan::detail::make_source_id(one_based),
      .port = holoscan::PortId{one_based},
      .source_clock = kReferenceClock,
      .reference_clock = kReferenceClock,
      .source_order = one_based,
      .capacity = 1U,
      .source_generation = 1U,
      .tag_generation = 1U,
      .receiver_index = index,
      .partition = holoscan::ExecutionPartitionId{1U},
      .execution_lane = 0U,
      .vertex = holoscan::VertexId{one_based},
      .plan_vertex_order = one_based,
      .stable_candidate_tie = one_based,
  };
}

// REQ: RUNTIME-007
// REQ: SYSTEM-005
void BM_DispatcherReadySetCycle(benchmark::State& state) {
  const auto candidate_count = static_cast<std::size_t>(state.range(0));
  const holoscan::SchedulerPlanRecord plan = scheduler_plan();
  auto dispatcher = holoscan::detail::Dispatcher::create(plan, candidate_count);
  if (!dispatcher) {
    state.SkipWithError("failed to create bounded Dispatcher state");
    return;
  }
  const auto candidates = make_candidates(candidate_count);

  for (auto _ : state) {
    auto frozen = dispatcher->freeze(candidates);
    if (!frozen) {
      state.SkipWithError("failed to freeze READY candidates");
      return;
    }
    for (std::size_t index = 0U; index < candidate_count; ++index) {
      auto selected = dispatcher->select();
      if (!selected) {
        state.SkipWithError("failed to select a READY candidate");
        return;
      }
      benchmark::DoNotOptimize(selected->tag);
      auto completed = dispatcher->complete(*selected);
      if (!completed) {
        state.SkipWithError("failed to complete a READY candidate");
        return;
      }
    }
  }

  state.counters["ready_candidates"] = static_cast<double>(candidate_count);
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(candidate_count));
}

// Measure the deterministic candidate lifecycle separately from Dispatcher and lane work.
// REQ: RUNTIME-002
// REQ: RUNTIME-007
// The benchmark keeps setup and the measured candidate lifecycle visibly paired.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void BM_RuntimeCandidateBatchCycle(benchmark::State& state) {
  constexpr std::size_t kMaximumCandidateCount = 4U;
  constexpr holoscan::ClockId kReferenceClock{23U};
  const auto candidate_count = static_cast<std::size_t>(state.range(0));
  if (candidate_count == 0U || candidate_count > kMaximumCandidateCount) {
    state.SkipWithError("candidate count is outside the benchmark bound");
    return;
  }

  std::array<holoscan::detail::RuntimeSourceBindingRecord, kMaximumCandidateCount> sources;
  for (std::size_t index = 0U; index < sources.size(); ++index) {
    sources[index] = make_runtime_source(index);
  }
  auto coordinator = holoscan::detail::RuntimeCandidateCoordinator::create(
      std::span<const holoscan::detail::RuntimeSourceBindingRecord>{sources}.first(candidate_count),
      candidate_count,
      candidate_count);
  if (!coordinator) {
    state.SkipWithError("failed to create candidate coordinator benchmark state");
    return;
  }

  holoscan::detail::BoundedEventPortBinding binding_0{1U};
  holoscan::detail::BoundedEventPortBinding binding_1{1U};
  holoscan::detail::BoundedEventPortBinding binding_2{1U};
  holoscan::detail::BoundedEventPortBinding binding_3{1U};
  const std::array<holoscan::detail::BoundedEventPortBinding*, kMaximumCandidateCount> bindings{
      &binding_0, &binding_1, &binding_2, &binding_3};

  std::uint64_t source_sequence = 0U;
  std::int64_t completed_candidates = 0;
  for (auto _ : state) {
    ++source_sequence;
    for (std::size_t index = 0U; index < candidate_count; ++index) {
      const holoscan::detail::RuntimeEvent event{
          .kind = sources[index].candidate_kind,
          .source = sources[index].source,
          .port = sources[index].port,
          .source_decision_time =
              holoscan::TimePoint{static_cast<std::int64_t>(source_sequence), kReferenceClock},
          .reference_time =
              holoscan::TimePoint{static_cast<std::int64_t>(source_sequence), kReferenceClock},
          .source_generation = sources[index].source_generation,
          .source_sequence = source_sequence,
      };
      auto pushed = bindings[index]->push(event);
      auto ingested = pushed ? coordinator->ingest(*bindings[index])
                             : holoscan::expected<std::size_t, holoscan::Error>{
                                   holoscan::make_unexpected(std::move(pushed.error()))};
      if (!ingested || *ingested != 1U) {
        state.SkipWithError("failed to ingest candidate batch");
        return;
      }
    }

    for (std::size_t index = 0U; index < candidate_count; ++index) {
      auto frozen = coordinator->freeze_next();
      if (!frozen || frozen->source_slot >= candidate_count) {
        state.SkipWithError("failed to freeze candidate batch");
        return;
      }
      benchmark::DoNotOptimize(frozen->candidate.tag);
      auto retired_source = coordinator->retire(*frozen);
      auto retired_binding =
          retired_source ? bindings[frozen->source_slot]->retire(frozen->witness.source_sequence)
                         : holoscan::expected<void, holoscan::Error>{
                               holoscan::make_unexpected(std::move(retired_source.error()))};
      if (!retired_binding) {
        state.SkipWithError("failed to retire candidate batch");
        return;
      }
      ++completed_candidates;
    }
  }

  state.counters["batch_candidates"] = static_cast<double>(candidate_count);
  state.SetItemsProcessed(completed_candidates);
}

enum class CandidateFreezeMode : std::uint8_t {
  kOrdinary,
  kConstrained,
};

void apply_candidate_freeze_args(benchmark::Benchmark* benchmark) {
  benchmark->Args({8, 1, 1, 0})
      ->Args({64, 1, 1, 0})
      ->Args({64, 4, 1, 0})
      ->Args({64, 4, 4, 0})
      ->Args({64, 4, 4, 4})
      ->ArgNames({"pending", "fences", "prefix_depth", "unrelated_owners"});
}

// Measure one full candidate-freeze API call at an unchanged pending-set shape. Retirement and
// refill run while timing is paused, and both modes freeze the same source, sequence, and pending
// index.
// REQ: RUNTIME-007
// REQ: RUNTIME-043
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void BM_RuntimeCandidateFreezeSelection(benchmark::State& state, CandidateFreezeMode mode) {
  constexpr holoscan::ClockId kReferenceClock{23U};
  constexpr std::int64_t kProbeReferenceNs = 1;
  constexpr std::int64_t kStaticReferenceNs = 1'000'000'000;
  const auto pending_count = static_cast<std::size_t>(state.range(0));
  const auto fence_count = static_cast<std::size_t>(state.range(1));
  const auto prefix_depth = static_cast<std::size_t>(state.range(2));
  const auto unrelated_owner_count = static_cast<std::size_t>(state.range(3));
  const std::size_t fenced_pending = fence_count * prefix_depth;
  if (pending_count == 0U || fence_count == 0U || prefix_depth == 0U ||
      fenced_pending + unrelated_owner_count > pending_count) {
    state.SkipWithError("invalid candidate-freeze benchmark shape");
    return;
  }
  const std::size_t filler_count = pending_count - fenced_pending - unrelated_owner_count;

  std::vector<holoscan::detail::RuntimeSourceBindingRecord> sources;
  sources.reserve(fence_count + unrelated_owner_count +
                  static_cast<std::size_t>(filler_count > 0U));
  const auto add_source = [&](const holoscan::detail::RuntimeEventKind kind,
                              const holoscan::VertexId owner,
                              const holoscan::PortId port,
                              const std::size_t capacity,
                              const std::size_t receiver_index) {
    const auto one_based = static_cast<std::uint32_t>(sources.size() + 1U);
    sources.push_back({
        .candidate_kind = kind,
        .source = holoscan::detail::make_source_id(one_based),
        .port = port,
        .source_clock = kReferenceClock,
        .reference_clock = kReferenceClock,
        .source_order = one_based,
        .capacity = static_cast<std::uint32_t>(capacity),
        .source_generation = 1U,
        .tag_generation = 1U,
        .receiver_index = receiver_index,
        .partition = holoscan::ExecutionPartitionId{1U},
        .execution_lane = 0U,
        .vertex = owner,
        .plan_vertex_order = static_cast<std::uint32_t>(owner.value),
        .stable_candidate_tie = one_based,
    });
  };

  constexpr holoscan::VertexId kFencedOwner{1U};
  for (std::size_t index = 0U; index < fence_count; ++index) {
    add_source(holoscan::detail::RuntimeEventKind::kInputArrival,
               kFencedOwner,
               holoscan::PortId{static_cast<std::uint32_t>(index + 1U)},
               prefix_depth,
               0U);
  }
  const std::size_t filler_source_index = sources.size();
  if (filler_count > 0U) {
    add_source(holoscan::detail::RuntimeEventKind::kClockTick, kFencedOwner, {}, filler_count, 0U);
  }
  const std::size_t unrelated_source_begin = sources.size();
  for (std::size_t index = 0U; index < unrelated_owner_count; ++index) {
    add_source(holoscan::detail::RuntimeEventKind::kClockTick,
               holoscan::VertexId{static_cast<std::uint32_t>(index + 2U)},
               {},
               1U,
               index + 1U);
  }

  auto coordinator = holoscan::detail::RuntimeCandidateCoordinator::create(
      sources, unrelated_owner_count + 1U, pending_count);
  if (!coordinator) {
    state.SkipWithError("failed to create candidate-freeze coordinator");
    return;
  }
  std::vector<holoscan::detail::RuntimeCandidateInputFence> fences;
  fences.reserve(fence_count);
  for (std::size_t index = 0U; index < fence_count; ++index) {
    fences.push_back({kFencedOwner, sources[index].source, sources[index].port});
  }

  holoscan::detail::BoundedEventPortBinding binding{pending_count};
  const auto push_event = [&](const std::size_t source_index,
                              const std::uint64_t sequence,
                              const std::int64_t reference_ns) {
    const auto& source = sources[source_index];
    return binding.push({
        .kind = source.candidate_kind,
        .source = source.source,
        .port = source.port,
        .source_decision_time = holoscan::TimePoint{reference_ns, kReferenceClock},
        .reference_time = holoscan::TimePoint{reference_ns, kReferenceClock},
        .source_generation = source.source_generation,
        .source_sequence = sequence,
    });
  };
  if (filler_count > 0U) {
    for (std::uint64_t sequence = 1U; sequence <= filler_count; ++sequence) {
      if (!push_event(filler_source_index, sequence, kStaticReferenceNs)) {
        state.SkipWithError("failed to preload candidate-freeze filler");
        return;
      }
    }
  }
  for (std::size_t index = 0U; index < unrelated_owner_count; ++index) {
    if (!push_event(unrelated_source_begin + index, 1U, kStaticReferenceNs)) {
      state.SkipWithError("failed to preload unrelated candidate");
      return;
    }
  }
  for (std::size_t source_index = 1U; source_index < fence_count; ++source_index) {
    for (std::uint64_t sequence = 1U; sequence <= prefix_depth; ++sequence) {
      if (!push_event(source_index, sequence, kStaticReferenceNs)) {
        state.SkipWithError("failed to preload fenced candidate");
        return;
      }
    }
  }
  for (std::uint64_t sequence = 1U; sequence <= prefix_depth; ++sequence) {
    if (!push_event(0U, sequence, kProbeReferenceNs)) {
      state.SkipWithError("failed to preload probe prefix");
      return;
    }
  }
  auto ingested = coordinator->ingest(binding);
  if (!ingested || *ingested != pending_count) {
    state.SkipWithError("failed to ingest candidate-freeze pending set");
    return;
  }

  std::uint64_t expected_sequence = 1U;
  std::uint64_t next_sequence = prefix_depth + 1U;
  for (auto _ : state) {
    auto frozen = mode == CandidateFreezeMode::kConstrained
                      ? coordinator->freeze_next_constrained(fences)
                      : coordinator->freeze_next();
    state.PauseTiming();
    if (!frozen || frozen->witness.source != sources.front().source ||
        frozen->witness.source_sequence != expected_sequence) {
      state.ResumeTiming();
      state.SkipWithError("candidate-freeze selected an unexpected witness");
      return;
    }
    benchmark::DoNotOptimize(frozen->candidate.tag);
    auto retired = coordinator->retire(*frozen);
    auto pushed = retired ? push_event(0U, next_sequence, kProbeReferenceNs)
                          : holoscan::expected<void, holoscan::Error>{
                                holoscan::make_unexpected(std::move(retired.error()))};
    auto refilled = pushed ? coordinator->ingest(binding)
                           : holoscan::expected<std::size_t, holoscan::Error>{
                                 holoscan::make_unexpected(std::move(pushed.error()))};
    if (!refilled || *refilled != 1U) {
      state.ResumeTiming();
      state.SkipWithError("failed to restore candidate-freeze pending set");
      return;
    }
    ++expected_sequence;
    ++next_sequence;
    state.ResumeTiming();
  }

  state.counters["fences"] = static_cast<double>(fence_count);
  state.counters["pending"] = static_cast<double>(pending_count);
  state.counters["prefix_depth"] = static_cast<double>(prefix_depth);
  state.counters["unrelated_owners"] = static_cast<double>(unrelated_owner_count);
  state.SetItemsProcessed(state.iterations());
}

// Measure the bounded activation metadata copy independently of scheduling and Channel ownership.
// REQ: RUNTIME-002
// REQ: RUNTIME-007
void BM_ActivationSelectionCycle(benchmark::State& state) {
  constexpr holoscan::ClockId kReferenceClock{23U};
  const auto selection_count = static_cast<std::size_t>(state.range(0));
  auto activation =
      holoscan::detail::ActivationSelectionState::create(selection_count, selection_count);
  if (!activation) {
    state.SkipWithError("failed to create activation selection benchmark state");
    return;
  }

  const holoscan::detail::FrozenRuntimeCandidate candidate{
      .witness =
          holoscan::detail::RuntimeEvent{
              .kind = holoscan::detail::RuntimeEventKind::kInputArrival,
              .source = holoscan::detail::make_source_id(1U),
              .source_decision_time = holoscan::TimePoint{1, kReferenceClock},
              .reference_time = holoscan::TimePoint{1, kReferenceClock},
              .source_generation = 1U,
              .source_sequence = 1U,
          },
      .candidate =
          holoscan::detail::DispatcherCandidate{
              .ready =
                  holoscan::detail::scheduling::ReadyCandidateView{
                      .partition = holoscan::ExecutionPartitionId{1U},
                      .execution_lane = 0U,
                      .vertex = holoscan::VertexId{1U},
                      .tag_time = holoscan::TagTime{1, 0U, 0U},
                      .plan_vertex_order = 0U,
                      .stable_source_id = holoscan::detail::make_source_id(1U),
                      .stable_port_id = holoscan::PortId{1U},
                      .stable_candidate_tie = 1U,
                      .active = true,
                  },
              .tag = holoscan::Tag{1U, 1U, 0U},
              .receiver_index = 0U,
          },
  };
  const std::vector<holoscan::InputSelection> selections(
      selection_count,
      holoscan::InputSelection{holoscan::InputState::kSelected, holoscan::SelectionReason::kNone});
  std::vector<holoscan::detail::TemporalSampleRecord> records;
  records.reserve(selection_count);
  for (std::size_t index = 0U; index < selection_count; ++index) {
    const auto sequence = static_cast<std::uint64_t>(index + 1U);
    records.push_back({
        .capture_time = holoscan::TimePoint{static_cast<std::int64_t>(sequence), kReferenceClock},
        .source_sequence = sequence,
        .producer_epoch = 1U,
        .arrival_sequence = sequence,
    });
  }

  std::int64_t completed_activations = 0;
  for (auto _ : state) {
    auto selected = activation->begin(candidate, selections, records);
    if (!selected) {
      state.SkipWithError("failed to begin activation selection cycle");
      return;
    }
    benchmark::DoNotOptimize(selected->inputs.data());
    benchmark::DoNotOptimize(selected->selected.data());
    auto retired = activation->retire(selected->tag);
    if (!retired) {
      state.SkipWithError("failed to retire activation selection cycle");
      return;
    }
    ++completed_activations;
  }

  state.counters["selected_records"] = static_cast<double>(selection_count);
  state.SetItemsProcessed(completed_activations);
}

// REQ: RUNTIME-007
void BM_PhysicalReadinessNotificationCycle(benchmark::State& state) {
  constexpr holoscan::ClockId kReferenceClock{17U};
  const holoscan::detail::PhysicalReadinessBindingDescriptor descriptor{
      .request =
          holoscan::detail::RuntimePortBindingRequest{
              .plan_generation = 1U,
              .run_generation = 1U,
              .kind = holoscan::detail::RuntimeEventKind::kNotification,
              .source = holoscan::detail::make_source_id(1U),
              .source_clock = kReferenceClock,
              .reference_clock = kReferenceClock,
              .capacity = 1U,
              .source_generation = 1U,
          },
      .wake_requirement = holoscan::ReadinessWakeRequirement::kDeploymentSelected,
      .wake_mode = holoscan::WakeMode::kSleepingWaitset,
  };
  auto factory = holoscan::detail::PhysicalReadinessBindingFactory::create(
      std::span<const holoscan::detail::PhysicalReadinessBindingDescriptor>{&descriptor, 1U});
  if (!factory) {
    state.SkipWithError("failed to create physical readiness factory");
    return;
  }
  auto binding = (*factory)->bind(descriptor.request, holoscan::TimePoint{1, kReferenceClock});
  if (!binding) {
    state.SkipWithError("failed to create physical readiness binding");
    return;
  }
  auto control = (*factory)->control(descriptor.request.source,
                                     descriptor.request.plan_generation,
                                     descriptor.request.run_generation,
                                     descriptor.request.source_generation);
  if (!control) {
    state.SkipWithError("failed to create physical readiness control");
    return;
  }

  holoscan::detail::RuntimeEvent event;
  for (auto _ : state) {
    auto published = control->post_notification(holoscan::TimePoint{2, kReferenceClock});
    if (!published) {
      state.SkipWithError("failed to publish physical readiness notification");
      return;
    }
    auto drained =
        (*binding)->drain(std::span<holoscan::detail::RuntimeEvent>{&event, std::size_t{1U}});
    if (!drained || *drained != 1U) {
      state.SkipWithError("failed to drain physical readiness notification");
      return;
    }
    benchmark::DoNotOptimize(event.source_sequence);
    auto retired = control->retire(event.source_sequence);
    if (!retired) {
      state.SkipWithError("failed to retire physical readiness notification");
      return;
    }
  }
  state.SetItemsProcessed(state.iterations());
}

struct ControllerActivationInvocation {
  holoscan::detail::Dispatcher* dispatcher{};
  const holoscan::detail::DispatcherSelection* selection{};
  std::uint64_t callbacks{};
  bool selection_completed{};
};

[[nodiscard]] holoscan::expected<void, holoscan::Error> invoke_controller_activation(
    void* opaque) noexcept {
  auto& invocation = *static_cast<ControllerActivationInvocation*>(opaque);
  ++invocation.callbacks;
  benchmark::DoNotOptimize(invocation.callbacks);
  auto completed = invocation.dispatcher->complete(*invocation.selection);
  invocation.selection_completed = completed.has_value();
  return completed;
}

// Compose the transport-free controller transaction in the same order used by RunSession.
// REQ: RUNTIME-002
// REQ: RUNTIME-007
// REQ: RUNTIME-011
// The benchmark intentionally spells out the complete measured activation transaction.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void BM_ControllerActivationTransaction(benchmark::State& state) {
  constexpr holoscan::ClockId kReferenceClock{23U};
  const holoscan::detail::RuntimeSourceBindingRecord source{
      .candidate_kind = holoscan::detail::RuntimeEventKind::kSpin,
      .source = holoscan::detail::make_source_id(1U),
      .source_clock = kReferenceClock,
      .reference_clock = kReferenceClock,
      .source_order = 0U,
      .capacity = 1U,
      .source_generation = 1U,
      .tag_generation = 1U,
      .receiver_index = 0U,
      .partition = holoscan::ExecutionPartitionId{1U},
      .execution_lane = 0U,
      .vertex = holoscan::VertexId{1U},
      .plan_vertex_order = 0U,
      .stable_candidate_tie = 1U,
  };
  auto coordinator = holoscan::detail::RuntimeCandidateCoordinator::create(
      std::span<const holoscan::detail::RuntimeSourceBindingRecord>{&source, 1U}, 1U, 1U);
  auto dispatcher = holoscan::detail::Dispatcher::create(scheduler_plan(), 1U);
  auto activation = holoscan::detail::ActivationSelectionState::create(0U, 0U);
  const std::array lanes{holoscan::ExecutionLanePlanRecord{
      .execution_lane = 0U,
      .partition = holoscan::ExecutionPartitionId{1U},
      .accepted_placement = holoscan::ThreadPlacement{.wake_mode = holoscan::WakeMode::kBusyPoll},
  }};
  auto lane_set = holoscan::detail::ExecutionLaneSet::create(scheduler_plan(), lanes);
  if (!coordinator || !dispatcher || !activation || !lane_set) {
    state.SkipWithError("failed to create controller activation benchmark state");
    return;
  }
  auto started = (*lane_set)->start();
  if (!started) {
    state.SkipWithError("failed to start controller activation benchmark lane");
    return;
  }

  holoscan::detail::BoundedEventPortBinding binding{1U};
  std::array<holoscan::detail::DispatcherCandidate, 1U> candidates;
  ControllerActivationInvocation invocation{.dispatcher = &*dispatcher};
  std::uint64_t source_sequence = 0U;
  std::int64_t completed_activations = 0;
  for (auto _ : state) {
    ++source_sequence;
    const holoscan::detail::RuntimeEvent event{
        .kind = source.candidate_kind,
        .source = source.source,
        .source_decision_time =
            holoscan::TimePoint{static_cast<std::int64_t>(source_sequence), kReferenceClock},
        .reference_time =
            holoscan::TimePoint{static_cast<std::int64_t>(source_sequence), kReferenceClock},
        .source_generation = source.source_generation,
        .source_sequence = source_sequence,
    };
    auto pushed = binding.push(event);
    auto ingested = pushed ? coordinator->ingest(binding)
                           : holoscan::expected<std::size_t, holoscan::Error>{
                                 holoscan::make_unexpected(std::move(pushed.error()))};
    auto frozen =
        ingested && *ingested == 1U
            ? coordinator->freeze_next()
            : holoscan::expected<holoscan::detail::FrozenRuntimeCandidate, holoscan::Error>{
                  holoscan::make_unexpected(ingested
                                                ? holoscan::Error{holoscan::ErrorCode::kFailure}
                                                : std::move(ingested.error()))};
    if (!frozen) {
      state.SkipWithError("failed to ingest and freeze controller activation candidate");
      break;
    }
    candidates.front() = frozen->candidate;
    auto dispatcher_frozen = dispatcher->freeze(candidates);
    if (!dispatcher_frozen) {
      state.SkipWithError("failed to freeze controller activation Dispatcher state");
      break;
    }

    const holoscan::detail::RuntimeCallbackContextScope dispatcher_scope{
        holoscan::detail::RuntimeCallbackContext::kDispatcher};
    auto selected = dispatcher->select();
    auto selection =
        selected ? activation->begin(*frozen, {}, {})
                 : holoscan::expected<holoscan::detail::ActivationSelectionView, holoscan::Error>{
                       holoscan::make_unexpected(std::move(selected.error()))};
    if (!selection) {
      state.SkipWithError("failed to select and prepare controller activation");
      break;
    }

    invocation.selection = &*selected;
    invocation.selection_completed = false;
    auto dispatched = (*lane_set)->dispatch(
        0U,
        holoscan::detail::ExecutionLaneTask{
            .context = &invocation,
            .invoke = invoke_controller_activation,
            .callback_identity =
                holoscan::detail::RuntimeCallbackIdentity{
                    .context = holoscan::detail::RuntimeCallbackContext::kCompute,
                    .operator_id = holoscan::OperatorInstanceId{1U},
                    .source = frozen->witness.source,
                    .source_sequence = frozen->witness.source_sequence,
                    .tag = frozen->candidate.tag,
                    .tag_time = frozen->candidate.ready.tag_time,
                    .reference_time = frozen->witness.reference_time,
                    .state_generation = frozen->candidate.tag.generation,
                },
        });
    auto retired_source = invocation.selection_completed
                              ? coordinator->retire(*frozen)
                              : holoscan::expected<void, holoscan::Error>{holoscan::make_unexpected(
                                    holoscan::Error{holoscan::ErrorCode::kFailure})};
    auto retired_binding = retired_source
                               ? binding.retire(frozen->witness.source_sequence)
                               : holoscan::expected<void, holoscan::Error>{
                                     holoscan::make_unexpected(std::move(retired_source.error()))};
    auto retired_activation =
        retired_binding ? activation->retire(selection->tag)
                        : holoscan::expected<void, holoscan::Error>{
                              holoscan::make_unexpected(std::move(retired_binding.error()))};
    if (!dispatched || !retired_activation) {
      state.SkipWithError("failed to dispatch or retire controller activation");
      break;
    }
    ++completed_activations;
  }
  state.SetItemsProcessed(completed_activations);
  benchmark::DoNotOptimize(invocation.callbacks);

  auto joined = (*lane_set)->join();
  if (!joined) {
    state.SkipWithError("failed to stop controller activation benchmark lane");
  }
}

struct LaneDispatchTiming {
  std::int64_t callback_entry_ns{};
  std::int64_t callback_exit_ns{};
};

[[nodiscard]] std::int64_t wall_timestamp_ns() noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void report_latency_percentiles(benchmark::State& state, std::span<std::uint64_t> samples,
                                const char* p50_name, const char* p90_name, const char* p99_name) {
  if (samples.empty()) {
    return;
  }
  std::sort(samples.begin(), samples.end());
  const auto nearest_rank = [&](const std::size_t numerator) {
    const std::size_t rank = (samples.size() * numerator + 99U) / 100U;
    return samples[std::clamp(rank, std::size_t{1U}, samples.size()) - 1U];
  };
  state.counters[p50_name] = static_cast<double>(nearest_rank(50U));
  state.counters[p90_name] = static_cast<double>(nearest_rank(90U));
  state.counters[p99_name] = static_cast<double>(nearest_rank(99U));
}

template <typename WakeState>
struct RuntimeWakeProbe {
  WakeState wake;
  std::atomic<std::uint64_t> requested_sequence{};
  std::atomic<std::int64_t> signal_timestamp_ns{};
  std::atomic_bool stop{};
};

template <typename WakeState>
void run_runtime_wake_driver(RuntimeWakeProbe<WakeState>& probe) noexcept {
  std::uint64_t handled_sequence = 0U;
  for (;;) {
    const std::uint64_t requested = probe.requested_sequence.load(std::memory_order_acquire);
    if (requested == handled_sequence) {
      if (probe.stop.load(std::memory_order_acquire)) {
        return;
      }
      std::this_thread::yield();
      continue;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{5});
    probe.signal_timestamp_ns.store(wall_timestamp_ns(), std::memory_order_release);
    probe.wake.signal();
    handled_sequence = requested;
  }
}

template <typename WakeState>
void run_runtime_wake_signal_to_return(benchmark::State& state) {
  RuntimeWakeProbe<WakeState> probe;
  if (!probe.wake.healthy()) {
    state.SkipWithError("Runtime wake probe initialization failed");
    return;
  }
  std::thread driver{run_runtime_wake_driver<WakeState>, std::ref(probe)};
  std::array<std::uint64_t, static_cast<std::size_t>(kParkedLatencyIterations)> samples{};
  std::size_t sample_count = 0U;
  std::uint64_t sequence = 0U;
  for (auto _ : state) {
    probe.requested_sequence.store(++sequence, std::memory_order_release);
    if (!probe.wake.wait_for(std::chrono::seconds{1})) {
      state.SkipWithError(probe.wake.healthy() ? "Runtime wake probe timed out"
                                               : "Runtime wake probe failed");
      break;
    }
    const std::int64_t returned_at_ns = wall_timestamp_ns();
    const std::int64_t signaled_at_ns = probe.signal_timestamp_ns.load(std::memory_order_acquire);
    if (signaled_at_ns <= 0 || returned_at_ns < signaled_at_ns) {
      state.SkipWithError("Runtime wake probe timing order is invalid");
      break;
    }
    const auto latency_ns = static_cast<std::uint64_t>(returned_at_ns - signaled_at_ns);
    if (sample_count < samples.size()) {
      samples[sample_count++] = latency_ns;
    }
    state.SetIterationTime(static_cast<double>(latency_ns) * 1.0e-9);
  }
  probe.stop.store(true, std::memory_order_release);
  driver.join();
  report_latency_percentiles(state,
                             std::span{samples}.first(sample_count),
                             "signal_to_return_p50_ns",
                             "signal_to_return_p90_ns",
                             "signal_to_return_p99_ns");
  state.SetItemsProcessed(state.iterations());
}

class CountingSemaphoreWakeProbe final {
 public:
  void signal() noexcept { wake_.signal(); }

  [[nodiscard]] bool wait_for(std::chrono::nanoseconds timeout) noexcept {
    return wake_.wait_for(timeout);
  }

  [[nodiscard]] constexpr bool healthy() const noexcept { return true; }

 private:
  holoscan::detail::RuntimeWakeState wake_;
};

// REQ: RUNTIME-025
// REQ: SYSTEM-005
void BM_RuntimeWakeSignalToReturn(benchmark::State& state) {
  run_runtime_wake_signal_to_return<CountingSemaphoreWakeProbe>(state);
}

#if defined(__linux__)

[[nodiscard]] timespec relative_timeout(const std::chrono::nanoseconds timeout) noexcept {
  const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(timeout);
  return timespec{.tv_sec = seconds.count(), .tv_nsec = (timeout - seconds).count()};
}

class FutexWakeProbe final {
 public:
  void signal() noexcept {
    std::uint32_t current = tokens_.load(std::memory_order_relaxed);
    while (current != std::numeric_limits<std::uint32_t>::max() &&
           !tokens_.compare_exchange_weak(
               current, current + 1U, std::memory_order_release, std::memory_order_relaxed)) {
    }
    if (current == std::numeric_limits<std::uint32_t>::max()) {
      error_.store(EOVERFLOW, std::memory_order_release);
      return;
    }
    while (::syscall(SYS_futex, address(), FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0) < 0) {
      if (errno != EINTR) {
        error_.store(errno, std::memory_order_release);
        return;
      }
    }
  }

  [[nodiscard]] bool wait_for(const std::chrono::nanoseconds timeout) noexcept {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (;;) {
      std::uint32_t current = tokens_.load(std::memory_order_acquire);
      while (current != 0U) {
        if (tokens_.compare_exchange_weak(
                current, current - 1U, std::memory_order_acquire, std::memory_order_relaxed)) {
          return true;
        }
      }
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) {
        return false;
      }
      const timespec remaining =
          relative_timeout(std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now));
      if (::syscall(SYS_futex, address(), FUTEX_WAIT_PRIVATE, 0, &remaining, nullptr, 0) == 0) {
        continue;
      }
      if (errno == EAGAIN || errno == EINTR) {
        continue;
      }
      if (errno == ETIMEDOUT) {
        return false;
      }
      error_.store(errno, std::memory_order_release);
      return false;
    }
  }

  [[nodiscard]] bool healthy() const noexcept {
    return error_.load(std::memory_order_acquire) == 0;
  }

 private:
  static_assert(sizeof(std::atomic<std::uint32_t>) == sizeof(std::uint32_t));
  static_assert(alignof(std::atomic<std::uint32_t>) >= alignof(std::uint32_t));

  [[nodiscard]] std::uint32_t* address() noexcept {
    return reinterpret_cast<std::uint32_t*>(&tokens_);
  }

  alignas(std::uint32_t) std::atomic<std::uint32_t> tokens_{};
  std::atomic_int error_{};
};

class EventfdWakeProbe final {
 public:
  EventfdWakeProbe() noexcept : fd_(::eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)) {
    if (fd_ < 0) {
      error_.store(errno, std::memory_order_relaxed);
    }
  }

  EventfdWakeProbe(const EventfdWakeProbe&) = delete;
  EventfdWakeProbe& operator=(const EventfdWakeProbe&) = delete;

  ~EventfdWakeProbe() {
    if (fd_ >= 0) {
      static_cast<void>(::close(fd_));
    }
  }

  void signal() noexcept {
    const eventfd_t increment = 1U;
    while (::eventfd_write(fd_, increment) != 0) {
      if (errno != EINTR) {
        error_.store(errno, std::memory_order_release);
        return;
      }
    }
  }

  [[nodiscard]] bool wait_for(const std::chrono::nanoseconds timeout) noexcept {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (;;) {
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) {
        return false;
      }
      const timespec remaining =
          relative_timeout(std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now));
      pollfd descriptor{.fd = fd_, .events = POLLIN, .revents = 0};
      const int result = ::ppoll(&descriptor, 1U, &remaining, nullptr);
      if (result == 0) {
        return false;
      }
      if (result < 0) {
        if (errno == EINTR) {
          continue;
        }
        error_.store(errno, std::memory_order_release);
        return false;
      }
      if ((descriptor.revents & POLLIN) != 0) {
        eventfd_t value = 0U;
        while (::eventfd_read(fd_, &value) != 0) {
          if (errno == EINTR) {
            continue;
          }
          error_.store(errno, std::memory_order_release);
          return false;
        }
        return true;
      }
      error_.store(descriptor.revents, std::memory_order_release);
      return false;
    }
  }

  [[nodiscard]] bool healthy() const noexcept {
    return error_.load(std::memory_order_acquire) == 0;
  }

 private:
  int fd_{-1};
  std::atomic_int error_{};
};

void BM_RuntimeFutexWakeSignalToReturn(benchmark::State& state) {
  run_runtime_wake_signal_to_return<FutexWakeProbe>(state);
}

void BM_RuntimeEventFdWakeSignalToReturn(benchmark::State& state) {
  run_runtime_wake_signal_to_return<EventfdWakeProbe>(state);
}

#endif

[[nodiscard]] holoscan::expected<void, holoscan::Error> invoke_empty_lane_task(
    void* opaque) noexcept {
  auto& timing = *static_cast<LaneDispatchTiming*>(opaque);
  timing.callback_entry_ns = wall_timestamp_ns();
  timing.callback_exit_ns = wall_timestamp_ns();
  return {};
}

// REQ: RUNTIME-011
// REQ: SYSTEM-005
void run_execution_lane_dispatch(benchmark::State& state, const holoscan::WakeMode wake_mode,
                                 const std::chrono::nanoseconds idle_before_dispatch) {
  const std::array lanes{holoscan::ExecutionLanePlanRecord{
      .execution_lane = 0U,
      .partition = holoscan::ExecutionPartitionId{1U},
      .accepted_placement = holoscan::ThreadPlacement{.wake_mode = wake_mode},
  }};
  auto created = holoscan::detail::ExecutionLaneSet::create(scheduler_plan(), lanes);
  if (!created) {
    state.SkipWithError("failed to create Runtime lane set");
    return;
  }
  auto lane_set = std::move(*created);
  auto started = lane_set->start();
  if (!started) {
    state.SkipWithError("failed to start Runtime lane set");
    return;
  }

  LaneDispatchTiming timing;
  std::array<std::uint64_t, static_cast<std::size_t>(kParkedLatencyIterations)> samples{};
  std::size_t sample_count = 0U;
  long double enqueue_to_entry_total_ns = 0.0L;
  long double exit_to_return_total_ns = 0.0L;
  for (auto _ : state) {
    if (idle_before_dispatch > std::chrono::nanoseconds::zero()) {
      std::this_thread::sleep_for(idle_before_dispatch);
    }
    const std::int64_t dispatch_start_ns = wall_timestamp_ns();
    auto dispatched = lane_set->dispatch(
        0U,
        holoscan::detail::ExecutionLaneTask{
            .context = &timing,
            .invoke = invoke_empty_lane_task,
            .callback_context = holoscan::detail::RuntimeCallbackContext::kCompute,
        });
    const std::int64_t dispatch_end_ns = wall_timestamp_ns();
    if (!dispatched) {
      state.SkipWithError("failed to dispatch Runtime lane task");
      return;
    }
    if (timing.callback_entry_ns < dispatch_start_ns ||
        timing.callback_exit_ns < timing.callback_entry_ns ||
        dispatch_end_ns < timing.callback_exit_ns) {
      state.SkipWithError("Runtime lane timing order is invalid");
      return;
    }
    enqueue_to_entry_total_ns +=
        static_cast<long double>(timing.callback_entry_ns - dispatch_start_ns);
    exit_to_return_total_ns += static_cast<long double>(dispatch_end_ns - timing.callback_exit_ns);
    if (idle_before_dispatch > std::chrono::nanoseconds::zero()) {
      const auto latency_ns = static_cast<std::uint64_t>(dispatch_end_ns - dispatch_start_ns);
      if (sample_count < samples.size()) {
        samples[sample_count++] = latency_ns;
      }
      state.SetIterationTime(static_cast<double>(latency_ns) * 1.0e-9);
    }
  }
  state.SetItemsProcessed(state.iterations());
  state.counters["enqueue_to_entry_ns"] = benchmark::Counter{
      static_cast<double>(enqueue_to_entry_total_ns), benchmark::Counter::kAvgIterations};
  state.counters["exit_to_return_ns"] = benchmark::Counter{
      static_cast<double>(exit_to_return_total_ns), benchmark::Counter::kAvgIterations};
  if (idle_before_dispatch > std::chrono::nanoseconds::zero()) {
    report_latency_percentiles(state,
                               std::span{samples}.first(sample_count),
                               "dispatch_p50_ns",
                               "dispatch_p90_ns",
                               "dispatch_p99_ns");
  }

  auto joined = lane_set->join();
  if (!joined) {
    state.SkipWithError("failed to stop Runtime lane set");
  }
}

void BM_ExecutionLaneDispatch(benchmark::State& state, const holoscan::WakeMode wake_mode) {
  run_execution_lane_dispatch(state, wake_mode, std::chrono::nanoseconds::zero());
}

void BM_ExecutionLaneDispatchAfterIdle(benchmark::State& state,
                                       const holoscan::WakeMode wake_mode) {
  run_execution_lane_dispatch(state, wake_mode, std::chrono::milliseconds{5});
}

// REQ: RUNTIME-004
// REQ: RUNTIME-007
template <typename OperatorT>
[[nodiscard]] holoscan::ExecutionPlan make_direct_run_plan(const char* graph_name,
                                                           std::uint32_t activation_count = 1U) {
  holoscan::Graph graph{graph_name};
  const auto op = graph.op<OperatorT>("operator");
  graph.partition("runtime-benchmark-worker").add(op);
  const holoscan::ClockKey clock = graph.add_clock<BenchmarkClock>("benchmark-clock");
  graph.set_default_clock(clock);
  holoscan::CompileOptions options;
  options.deployment.bind_thread_placement(
      "runtime-benchmark-worker",
      holoscan::ThreadPlacement{.wake_mode = holoscan::WakeMode::kBusyPoll});
  return holoscan::test::enable_direct_runtime(holoscan::compile(graph, options), activation_count);
}

[[nodiscard]] holoscan::ExecutionPlan make_compute_call_completion_run_plan(
    const std::uint32_t activation_count, const std::optional<std::uint64_t> completion_limit) {
  holoscan::Graph graph{"runtime-compute-call-completion-benchmark"};
  const auto source = graph.op<BenchmarkComputeCallCompletionOperator>("source", completion_limit);
  graph.partition("runtime-benchmark-worker").add(source);
  const holoscan::ClockKey clock = graph.add_clock<BenchmarkClock>("benchmark-clock");
  graph.set_default_clock(clock);
  holoscan::CompileOptions options;
  options.deployment.bind_thread_placement(
      "runtime-benchmark-worker",
      holoscan::ThreadPlacement{.wake_mode = holoscan::WakeMode::kBusyPoll});
  return holoscan::test::enable_direct_runtime(holoscan::compile(graph, options), activation_count);
}

[[nodiscard]] holoscan::ExecutionPlan make_compute_call_completion_drain_plan(
    const std::uint32_t compute_call_limit) {
  holoscan::Graph graph{"runtime-compute-call-completion-drain-benchmark"};
  const auto source = graph.op<BenchmarkCompletionChannelSource>("source", compute_call_limit);
  const auto sink = graph.op<BenchmarkCompletionChannelSink>("sink");
  graph.partition("runtime-source-worker").add(source);
  graph.partition("runtime-sink-worker").add(sink);
  graph.add_flow(source->output,
                 sink->input,
                 holoscan::ConnectionOptions{.queue_depth = kCompletionDrainMaximumComputeCalls});
  const holoscan::ClockKey clock = graph.add_clock<BenchmarkClock>("benchmark-clock");
  graph.set_default_clock(clock);
  holoscan::CompileOptions options;
  options.deployment
      .bind_thread_placement("runtime-source-worker",
                             holoscan::ThreadPlacement{.wake_mode = holoscan::WakeMode::kBusyPoll})
      .bind_thread_placement("runtime-sink-worker",
                             holoscan::ThreadPlacement{.wake_mode = holoscan::WakeMode::kBusyPoll});
  return holoscan::test::enable_direct_runtime(holoscan::compile(graph, options),
                                               compute_call_limit);
}

[[nodiscard]] holoscan::ExecutionPlan make_mixed_completion_unruled_run_plan() {
  holoscan::Graph graph{"runtime-mixed-completion-unruled-benchmark"};
  const auto source = graph.op<BenchmarkMixedCompletionBurstSource>("ruled-source");
  const auto sink = graph.op<BenchmarkMixedCompletionUnruledSink>("unruled-sink");
  graph.partition("runtime-mixed-completion-worker").add(source).add(sink);
  graph.add_flow(source->output,
                 sink->input,
                 holoscan::ConnectionOptions{
                     .queue_policy = holoscan::QueuePolicy::kBlocking,
                     .queue_depth = kMixedCompletionUnruledCallbackCount,
                 });
  const holoscan::ClockKey clock = graph.add_clock<BenchmarkClock>("benchmark-clock");
  graph.set_default_clock(clock);
  holoscan::CompileOptions options;
  options.deployment.bind_thread_placement(
      "runtime-mixed-completion-worker",
      holoscan::ThreadPlacement{.wake_mode = holoscan::WakeMode::kBusyPoll});
  return holoscan::test::enable_direct_runtime(holoscan::compile(graph, options), 1U);
}

// REQ: RUNTIME-004
// REQ: RUNTIME-007
// REQ: RUNTIME-047
// REQ: RUNTIME-049
enum class DirectSessionSignalMode : std::uint8_t {
  kCallerManaged,
  kPersistentManagedLifecycle,
  kTransientManagedDefault,
};

void run_direct_session_benchmark(benchmark::State& state, const char* graph_name,
                                  const std::uint32_t activation_count,
                                  const DirectSessionSignalMode signal_mode) {
  const bool uses_persistent_domain =
      signal_mode == DirectSessionSignalMode::kPersistentManagedLifecycle;
  std::optional<holoscan::ManagedShutdownSignals> managed_signals;
  const auto establish_managed_domain = [&]() {
    auto created = holoscan::ManagedShutdownSignals::create();
    if (!created) {
      state.SkipWithError("failed to establish managed signal benchmark domain");
      return false;
    }
    managed_signals.emplace(std::move(*created));
    return true;
  };
  const auto close_managed_domain = [&]() {
    if (!managed_signals) {
      return true;
    }
    auto closed = managed_signals->close();
    managed_signals.reset();
    if (!closed) {
      state.SkipWithError("failed to close managed signal benchmark domain");
      return false;
    }
    return true;
  };

  // Managed admission must precede graph compilation and any participant threads it may start.
  if (uses_persistent_domain && !establish_managed_domain()) {
    return;
  }
  const holoscan::ExecutionPlan plan =
      make_direct_run_plan<BenchmarkOperator>(graph_name, activation_count);
  if (!plan.ok()) {
    close_managed_domain();
    state.SkipWithError("failed to compile direct Runtime benchmark plan");
    return;
  }
  // The bootstrap domain protects compilation from creating an incompatible signal-mask lineage,
  // but it is outside the measurement. Every managed iteration below measures the same complete
  // create/run/close lifecycle; caller-managed rows measure only run(plan, kCallerManaged).
  if (uses_persistent_domain && !close_managed_domain()) {
    return;
  }

  for (auto _ : state) {
    if (uses_persistent_domain && !establish_managed_domain()) {
      return;
    }
    int result = signal_mode == DirectSessionSignalMode::kCallerManaged
                     ? holoscan::run(plan, holoscan::RunSignalPolicy::kCallerManaged)
                     : holoscan::run(plan);
    benchmark::DoNotOptimize(result);
    if (uses_persistent_domain && !close_managed_domain()) {
      return;
    }
    if (result != 0) {
      state.SkipWithError("direct Runtime benchmark did not complete successfully");
      return;
    }
  }
  state.SetItemsProcessed(state.iterations() * activation_count);
}

void BM_RunSessionOneDirectActivation(benchmark::State& state) {
  run_direct_session_benchmark(
      state, "runtime-dispatcher-benchmark", 1U, DirectSessionSignalMode::kCallerManaged);
}

void BM_RunSessionOneDirectActivationPersistentManagedLifecycle(benchmark::State& state) {
  run_direct_session_benchmark(state,
                               "runtime-dispatcher-managed-lifecycle-benchmark",
                               1U,
                               DirectSessionSignalMode::kPersistentManagedLifecycle);
}

void BM_RunSessionOneDirectActivationTransientManagedDefault(benchmark::State& state) {
  run_direct_session_benchmark(state,
                               "runtime-dispatcher-transient-managed-benchmark",
                               1U,
                               DirectSessionSignalMode::kTransientManagedDefault);
}

// REQ: PARAM-019
// REQ: RUNTIME-004
void BM_RunSessionOneDirectActivationRuntimeMutableNoUpdate(benchmark::State& state) {
  const holoscan::ExecutionPlan plan =
      make_direct_run_plan<BenchmarkParameterOperator>("runtime-parameter-benchmark");
  if (!plan.ok()) {
    state.SkipWithError("failed to compile parameter Runtime benchmark plan");
    return;
  }

  for (auto _ : state) {
    // PARAM-019 measures parameter access, not the default managed-signal lifecycle.
    int result = holoscan::run(plan, holoscan::RunSignalPolicy::kCallerManaged);
    benchmark::DoNotOptimize(result);
    if (result != 0) {
      state.SkipWithError("parameter Runtime benchmark did not complete successfully");
      return;
    }
  }
  state.SetItemsProcessed(state.iterations());
}

// REQ: RUNTIME-004
// REQ: RUNTIME-007
void BM_RunSessionSteadyDirectActivations(benchmark::State& state) {
  run_direct_session_benchmark(state,
                               "runtime-steady-dispatcher-benchmark",
                               kSteadyActivationCount,
                               DirectSessionSignalMode::kCallerManaged);
}

void BM_RunSessionSteadyDirectActivationsPersistentManagedLifecycle(benchmark::State& state) {
  run_direct_session_benchmark(state,
                               "runtime-steady-managed-lifecycle-benchmark",
                               kSteadyActivationCount,
                               DirectSessionSignalMode::kPersistentManagedLifecycle);
}

void BM_RunSessionSteadyDirectActivationsTransientManagedDefault(benchmark::State& state) {
  run_direct_session_benchmark(state,
                               "runtime-steady-transient-managed-benchmark",
                               kSteadyActivationCount,
                               DirectSessionSignalMode::kTransientManagedDefault);
}

template <std::size_t QueryCount>
void run_fixed_rate_query_benchmark(benchmark::State& state, const char* graph_name) {
  const holoscan::ExecutionPlan plan =
      make_direct_run_plan<BenchmarkFixedRateQueryOperator<QueryCount>>(graph_name,
                                                                        kSteadyActivationCount);
  if (!plan.ok()) {
    state.SkipWithError("failed to compile fixed-rate query benchmark plan");
    return;
  }

  for (auto _ : state) {
    int result = holoscan::run(plan, holoscan::RunSignalPolicy::kCallerManaged);
    benchmark::DoNotOptimize(result);
    if (result != 0) {
      state.SkipWithError("fixed-rate query benchmark did not complete successfully");
      return;
    }
  }
  state.counters["rate_queries_per_activation"] = static_cast<double>(QueryCount);
  state.SetItemsProcessed(state.iterations() * kSteadyActivationCount);
}

// REQ: LIFECYCLE-012
void BM_RunSessionSteadyFixedRateMetadataNoQuery(benchmark::State& state) {
  run_fixed_rate_query_benchmark<0U>(state, "runtime-fixed-rate-no-query-benchmark");
}

// REQ: LIFECYCLE-012
void BM_RunSessionSteadyFixedRateOneQuery(benchmark::State& state) {
  run_fixed_rate_query_benchmark<1U>(state, "runtime-fixed-rate-one-query-benchmark");
}

// REQ: LIFECYCLE-012
void BM_RunSessionSteadyFixedRateEightQueries(benchmark::State& state) {
  run_fixed_rate_query_benchmark<8U>(state, "runtime-fixed-rate-eight-query-benchmark");
}

// Match BM_RunSessionSteadyComputeCallCompletion's Operator, output, session API, callback count,
// and terminal validation while omitting only the completion rule.
// REQ: RUNTIME-050
void BM_RunSessionSteadyOutputNoCompletion(benchmark::State& state) {
  const holoscan::ExecutionPlan plan =
      make_compute_call_completion_run_plan(kSteadyActivationCount, std::nullopt);
  if (!plan.ok() || !plan.operator_completion_rules().empty() || plan.ports().size() != 1U ||
      !plan.payload_edges().empty()) {
    state.SkipWithError("failed to compile matched no-completion benchmark plan");
    return;
  }

  for (auto _ : state) {
    holoscan::RunSession session = holoscan::run_async(plan);
    session.wait();
    state.PauseTiming();
    const auto progress = session.operator_completion_progress();
    benchmark::DoNotOptimize(progress.data());
    if (session.termination() != holoscan::RunTermination::kCompleted || !progress.empty()) {
      state.ResumeTiming();
      state.SkipWithError("matched no-completion benchmark did not complete naturally");
      return;
    }
    state.ResumeTiming();
  }
  state.counters["completion_rules"] = 0.0;
  state.counters["terminal_progress_records"] = 0.0;
  state.SetItemsProcessed(state.iterations() * kSteadyActivationCount);
}

// Compare only with BM_RunSessionSteadyOutputNoCompletion. The matched pair isolates the bounded
// recurring admission check plus the final output-close and terminal-progress decisions.
// REQ: RUNTIME-050
void BM_RunSessionSteadyComputeCallCompletion(benchmark::State& state) {
  const holoscan::ExecutionPlan plan = make_compute_call_completion_run_plan(
      kSteadyActivationCount, std::uint64_t{kSteadyActivationCount});
  if (!plan.ok() || plan.operator_completion_rules().size() != 1U || plan.ports().size() != 1U ||
      !plan.payload_edges().empty()) {
    state.SkipWithError("failed to compile compute-call completion benchmark plan");
    return;
  }

  for (auto _ : state) {
    holoscan::RunSession session = holoscan::run_async(plan);
    session.wait();
    state.PauseTiming();
    const auto progress = session.operator_completion_progress();
    benchmark::DoNotOptimize(progress.data());
    if (session.termination() != holoscan::RunTermination::kCompleted || progress.size() != 1U ||
        progress.front().successful_compute_calls != kSteadyActivationCount ||
        !progress.front().limit_reached) {
      state.ResumeTiming();
      state.SkipWithError("compute-call completion benchmark did not reach its exact limit");
      return;
    }
    state.ResumeTiming();
  }
  state.counters["completion_rules"] = 1.0;
  state.counters["terminal_progress_records"] = 1.0;
  state.SetItemsProcessed(state.iterations() * kSteadyActivationCount);
}

// Measure the limit-reaching callback, per-output EOS publication, downstream delivery, and
// ordinary input-owner drain as one natural-completion transaction.
// REQ: RUNTIME-050
// REQ: CHAN-136
void BM_RunSessionComputeCallCompletionAndDrain(benchmark::State& state) {
  const auto compute_call_limit = static_cast<std::uint32_t>(state.range(0));
  const holoscan::ExecutionPlan plan = make_compute_call_completion_drain_plan(compute_call_limit);
  if (!plan.ok() || plan.operator_completion_rules().size() != 1U ||
      plan.execution_partitions().size() != 2U || plan.schedule_edges().size() != 1U ||
      plan.payload_edges().size() != 1U ||
      plan.payload_edges().front().disposition != holoscan::PlanDisposition::kAccepted ||
      plan.payload_edges().front().locality != holoscan::ChannelLocalityKind::kProcessLocal ||
      plan.payload_edges().front().queue_policy != holoscan::QueuePolicy::kBlocking ||
      plan.payload_edges().front().queue_depth != kCompletionDrainMaximumComputeCalls ||
      plan.payload_edges().front().producer_max_emits_per_compute != 1U ||
      plan.payload_edges().front().fanout != 1U) {
    state.SkipWithError("failed to compile compute-call completion drain benchmark plan");
    return;
  }

  std::uint64_t verified_receipts{};
  for (auto _ : state) {
    BenchmarkCompletionChannelSink::calls.store(0U, std::memory_order_relaxed);
    holoscan::RunSession session = holoscan::run_async(plan);
    session.wait();
    state.PauseTiming();
    const auto progress = session.operator_completion_progress();
    const std::uint64_t received =
        BenchmarkCompletionChannelSink::calls.load(std::memory_order_relaxed);
    benchmark::DoNotOptimize(progress.data());
    if (session.termination() != holoscan::RunTermination::kCompleted || progress.size() != 1U ||
        progress.front().successful_compute_calls != compute_call_limit ||
        !progress.front().limit_reached || received != compute_call_limit) {
      state.ResumeTiming();
      state.SkipWithError(
          "compute-call completion drain benchmark did not deliver every accepted value");
      return;
    }
    verified_receipts += received;
    state.ResumeTiming();
  }
  state.counters["completion_rules"] = 1.0;
  state.counters["terminal_progress_records"] = 1.0;
  state.SetItemsProcessed(static_cast<std::int64_t>(verified_receipts));
}

// One finite ruled callback publishes a bounded burst, then the same partition executes only the
// unruled sink. Every sink callback traverses the partition-level completion wrapper's no-rule
// fallthrough, quantifying the deferred receiver-specialization opportunity while the source keeps
// its exact limit.
// REQ: RUNTIME-050
void BM_RunSessionMixedCompletionUnruledCallbacks(benchmark::State& state) {
  const holoscan::ExecutionPlan plan = make_mixed_completion_unruled_run_plan();
  if (!plan.ok() || plan.operator_completion_rules().size() != 1U ||
      plan.operator_completion_rules().front().limit != 1U ||
      plan.execution_partitions().size() != 1U || plan.readiness().size() != 1U ||
      plan.temporal_triggers().size() != 2U || !plan.schedule_edges().empty() ||
      plan.payload_edges().size() != 1U ||
      plan.payload_edges().front().disposition != holoscan::PlanDisposition::kAccepted ||
      plan.payload_edges().front().locality != holoscan::ChannelLocalityKind::kProcessLocal ||
      plan.payload_edges().front().queue_policy != holoscan::QueuePolicy::kBlocking ||
      plan.payload_edges().front().queue_depth != kMixedCompletionUnruledCallbackCount ||
      plan.payload_edges().front().producer_max_emits_per_compute !=
          kMixedCompletionUnruledCallbackCount ||
      plan.payload_edges().front().fanout != 1U) {
    state.SkipWithError("failed to compile mixed completion/unruled benchmark plan");
    return;
  }

  std::uint64_t verified_unruled_callbacks{};
  for (auto _ : state) {
    BenchmarkMixedCompletionBurstSource::calls.store(0U, std::memory_order_relaxed);
    BenchmarkMixedCompletionUnruledSink::calls.store(0U, std::memory_order_relaxed);
    holoscan::RunSession session = holoscan::run_async(plan);
    session.wait();
    state.PauseTiming();
    const auto progress = session.operator_completion_progress();
    const std::uint64_t ruled_calls =
        BenchmarkMixedCompletionBurstSource::calls.load(std::memory_order_relaxed);
    const std::uint64_t unruled_calls =
        BenchmarkMixedCompletionUnruledSink::calls.load(std::memory_order_relaxed);
    benchmark::DoNotOptimize(progress.data());
    if (session.termination() != holoscan::RunTermination::kCompleted ||
        !session.diagnostics().empty() || progress.size() != 1U ||
        progress.front().configured_limit != 1U ||
        progress.front().successful_compute_calls != 1U || !progress.front().limit_reached ||
        ruled_calls != 1U || unruled_calls != kMixedCompletionUnruledCallbackCount) {
      state.ResumeTiming();
      state.SkipWithError("mixed completion/unruled benchmark violated its frozen invariants");
      return;
    }
    verified_unruled_callbacks += unruled_calls;
    state.ResumeTiming();
  }
  state.counters["completion_rules"] = 1.0;
  state.counters["ruled_callbacks_per_iteration"] = 1.0;
  state.counters["terminal_progress_records"] = 1.0;
  state.counters["unruled_callbacks_per_iteration"] =
      static_cast<double>(kMixedCompletionUnruledCallbackCount);
  state.SetItemsProcessed(static_cast<std::int64_t>(verified_unruled_callbacks));
}

// REQ: PARAM-019
// REQ: RUNTIME-004
void BM_RunSessionSteadyDirectActivationsRuntimeMutableNoUpdate(benchmark::State& state) {
  const holoscan::ExecutionPlan plan = make_direct_run_plan<BenchmarkParameterOperator>(
      "runtime-steady-parameter-benchmark", kSteadyActivationCount);
  if (!plan.ok()) {
    state.SkipWithError("failed to compile steady Parameter benchmark plan");
    return;
  }

  for (auto _ : state) {
    // Keep the steady PARAM-019 row comparable with the one-activation row above.
    int result = holoscan::run(plan, holoscan::RunSignalPolicy::kCallerManaged);
    benchmark::DoNotOptimize(result);
    if (result != 0) {
      state.SkipWithError("steady Parameter benchmark did not complete successfully");
      return;
    }
  }
  state.SetItemsProcessed(state.iterations() * kSteadyActivationCount);
}

template <std::uint32_t kBurstCapacity>
void run_tensor_pool_capacity_benchmark(benchmark::State& state) {
  const holoscan::ExecutionPlan plan =
      make_direct_run_plan<BenchmarkTensorOperator<kBurstCapacity>>("runtime-tensor-pool-benchmark",
                                                                    kTensorActivationCount);
  constexpr std::uint32_t kExpectedSlots = 2U * kBurstCapacity;
  if (!plan.ok() || plan.tensor_allocations().size() != 1U ||
      plan.tensor_allocations().front().slot_count != kExpectedSlots) {
    state.SkipWithError("failed to compile the expected direct Tensor pool capacity");
    return;
  }

  for (auto _ : state) {
    int result = holoscan::run(plan, holoscan::RunSignalPolicy::kCallerManaged);
    benchmark::DoNotOptimize(result);
    if (result != 0) {
      state.SkipWithError("direct Tensor pool benchmark did not complete successfully");
      return;
    }
  }
  state.counters["pool_slots"] = kExpectedSlots;
  state.SetItemsProcessed(state.iterations() * kTensorActivationCount);
}

// REQ: TENSOR-STORAGE-013
void BM_DirectTensorPoolCycleBySlotCapacity(benchmark::State& state) {
  switch (state.range(0)) {
    case 2:
      run_tensor_pool_capacity_benchmark<1U>(state);
      break;
    case 8:
      run_tensor_pool_capacity_benchmark<4U>(state);
      break;
    case 64:
      run_tensor_pool_capacity_benchmark<32U>(state);
      break;
    case 1024:
      run_tensor_pool_capacity_benchmark<512U>(state);
      break;
    default:
      state.SkipWithError("unsupported direct Tensor pool capacity");
      break;
  }
}

BENCHMARK(BM_DispatcherReadySetCycle)->Arg(1)->Arg(8)->Arg(64);
BENCHMARK(BM_RuntimeCandidateBatchCycle)->Arg(1)->Arg(2)->Arg(4);
BENCHMARK_CAPTURE(BM_RuntimeCandidateFreezeSelection, ordinary, CandidateFreezeMode::kOrdinary)
    ->Apply(apply_candidate_freeze_args);
BENCHMARK_CAPTURE(BM_RuntimeCandidateFreezeSelection, constrained,
                  CandidateFreezeMode::kConstrained)
    ->Apply(apply_candidate_freeze_args);
BENCHMARK(BM_ActivationSelectionCycle)->Arg(0)->Arg(1)->Arg(8);
BENCHMARK(BM_PhysicalReadinessNotificationCycle);
BENCHMARK(BM_ControllerActivationTransaction)->UseRealTime();
BENCHMARK(BM_RuntimeWakeSignalToReturn)->Iterations(kParkedLatencyIterations)->UseManualTime();
#if defined(__linux__)
BENCHMARK(BM_RuntimeFutexWakeSignalToReturn)->Iterations(kParkedLatencyIterations)->UseManualTime();
BENCHMARK(BM_RuntimeEventFdWakeSignalToReturn)
    ->Iterations(kParkedLatencyIterations)
    ->UseManualTime();
#endif
BENCHMARK_CAPTURE(BM_ExecutionLaneDispatch, sleeping_waitset, holoscan::WakeMode::kSleepingWaitset)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_ExecutionLaneDispatch, busy_poll, holoscan::WakeMode::kBusyPoll)
    ->UseRealTime();
BENCHMARK_CAPTURE(BM_ExecutionLaneDispatchAfterIdle, sleeping_waitset,
                  holoscan::WakeMode::kSleepingWaitset)
    ->Iterations(kParkedLatencyIterations)
    ->UseManualTime();
BENCHMARK_CAPTURE(BM_ExecutionLaneDispatchAfterIdle, busy_poll, holoscan::WakeMode::kBusyPoll)
    ->Iterations(kParkedLatencyIterations)
    ->UseManualTime();
BENCHMARK(BM_RunSessionOneDirectActivation)->UseRealTime();
BENCHMARK(BM_RunSessionOneDirectActivationPersistentManagedLifecycle)->UseRealTime();
BENCHMARK(BM_RunSessionOneDirectActivationTransientManagedDefault)->UseRealTime();
BENCHMARK(BM_RunSessionOneDirectActivationRuntimeMutableNoUpdate)->UseRealTime();
BENCHMARK(BM_RunSessionSteadyDirectActivations)->UseRealTime();
BENCHMARK(BM_RunSessionSteadyDirectActivationsPersistentManagedLifecycle)->UseRealTime();
BENCHMARK(BM_RunSessionSteadyDirectActivationsTransientManagedDefault)->UseRealTime();
BENCHMARK(BM_RunSessionSteadyFixedRateMetadataNoQuery)->UseRealTime();
BENCHMARK(BM_RunSessionSteadyFixedRateOneQuery)->UseRealTime();
BENCHMARK(BM_RunSessionSteadyFixedRateEightQueries)->UseRealTime();
BENCHMARK(BM_RunSessionSteadyOutputNoCompletion)->UseRealTime();
BENCHMARK(BM_RunSessionSteadyComputeCallCompletion)->UseRealTime();
BENCHMARK(BM_RunSessionComputeCallCompletionAndDrain)
    ->Arg(1)
    ->Arg(kCompletionDrainMaximumComputeCalls)
    ->UseRealTime();
BENCHMARK(BM_RunSessionMixedCompletionUnruledCallbacks)->UseRealTime();
BENCHMARK(BM_RunSessionSteadyDirectActivationsRuntimeMutableNoUpdate)->UseRealTime();
BENCHMARK(BM_DirectTensorPoolCycleBySlotCapacity)
    ->Arg(2)
    ->Arg(8)
    ->Arg(64)
    ->Arg(1024)
    ->UseRealTime();

}  // namespace
