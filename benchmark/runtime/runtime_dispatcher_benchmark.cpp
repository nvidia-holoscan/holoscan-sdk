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
        .produces_tensor(holoscan::TensorPortLayout{
            .memory_kind = holoscan::MemoryKind::kHost,
            .dtype = DLDataType{kDLUInt, 64U, 1U},
        })
        .tensor_allocation(holoscan::TensorAllocationBounds{.max_rank = 1U, .max_byte_span = 8U});
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

// REQ: RUNTIME-004
// REQ: RUNTIME-007
void BM_RunSessionOneDirectActivation(benchmark::State& state) {
  const holoscan::ExecutionPlan plan =
      make_direct_run_plan<BenchmarkOperator>("runtime-dispatcher-benchmark");
  if (!plan.ok()) {
    state.SkipWithError("failed to compile direct Runtime benchmark plan");
    return;
  }

  for (auto _ : state) {
    int result = holoscan::run(plan);
    benchmark::DoNotOptimize(result);
    if (result != 0) {
      state.SkipWithError("direct Runtime benchmark did not complete successfully");
      return;
    }
  }
  state.SetItemsProcessed(state.iterations());
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
    int result = holoscan::run(plan);
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
  const holoscan::ExecutionPlan plan = make_direct_run_plan<BenchmarkOperator>(
      "runtime-steady-dispatcher-benchmark", kSteadyActivationCount);
  if (!plan.ok()) {
    state.SkipWithError("failed to compile steady Runtime benchmark plan");
    return;
  }

  for (auto _ : state) {
    int result = holoscan::run(plan);
    benchmark::DoNotOptimize(result);
    if (result != 0) {
      state.SkipWithError("steady Runtime benchmark did not complete successfully");
      return;
    }
  }
  state.SetItemsProcessed(state.iterations() * kSteadyActivationCount);
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
    int result = holoscan::run(plan);
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
    int result = holoscan::run(plan);
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
BENCHMARK(BM_RunSessionOneDirectActivationRuntimeMutableNoUpdate)->UseRealTime();
BENCHMARK(BM_RunSessionSteadyDirectActivations)->UseRealTime();
BENCHMARK(BM_RunSessionSteadyDirectActivationsRuntimeMutableNoUpdate)->UseRealTime();
BENCHMARK(BM_DirectTensorPoolCycleBySlotCapacity)
    ->Arg(2)
    ->Arg(8)
    ->Arg(64)
    ->Arg(1024)
    ->UseRealTime();

}  // namespace
