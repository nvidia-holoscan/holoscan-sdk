/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "../common/benchmark_pod.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime_graph_benchmark_common.hpp"

namespace holoscan {
namespace {

using benchmarking::BenchmarkMonotonicClock;
using benchmarking::BenchmarkSupportThreadControl;
using benchmarking::SupportThreadBinding;
using benchmarking::SupportThreadRole;

struct Options {
  std::uint32_t producers{2U};
  std::uint32_t fanout{2U};
  std::uint32_t samples{10'000U};
  std::uint32_t warmup{100U};
  std::uint32_t queue_depth{2U};
  std::uint32_t period_us{};  // Zero requests an always-due 1 ns clock source.
  std::uint32_t sink_work_us{};
  std::uint32_t stall_us{};
  std::uint32_t timeout_ms{30'000U};
  int first_cpu{-1};
  int controller_cpu{-1};
  int pump_cpu{-1};
  bool shared_destinations{};
  bool busy_poll{};
  bool publication_credits{true};
  bool counters{};
  std::string delivery{"baseline"};
  std::string output;
};

// Each record is written only by its own callback lane and read after session.wait().
// The barrier is cold: there is no benchmark-global atomic update per publication.
struct alignas(64) ProducerRecord {
  std::uint64_t publications{};
  std::uint64_t retries{};
  std::uint64_t tid{};
  std::vector<std::int64_t> starts;
  std::vector<std::uint64_t> emits;
};

struct alignas(64) SinkRecord {
  std::uint64_t deliveries{};
  std::uint64_t tid{};
  std::int64_t stall_start{};
  std::int64_t stall_end{};
  std::vector<std::int64_t> receives;
};

struct State {
  explicit State(Options value)
      : options(std::move(value)),
        producers(options.producers),
        sinks(options.producers * options.fanout) {
    for (auto& producer : producers) {
      producer.starts.resize(options.samples);
      producer.emits.resize(options.samples);
    }
    for (auto& sink : sinks)
      sink.receives.resize(options.samples);
  }
  Options options;
  // Whether delivery inside emit() serves a service; Runtime then admits concurrently.
  bool concurrent_admission{};
  std::vector<ProducerRecord> producers;
  std::vector<SinkRecord> sinks;
  std::atomic<std::uint32_t> measurement_ready{};
};

State* g_state{};

void callback_work(const std::uint32_t microseconds) {
  if (microseconds == 0U)
    return;
  const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds{microseconds};
  while (std::chrono::steady_clock::now() < until) {
    std::atomic_signal_fence(std::memory_order_seq_cst);
  }
}

class ScalingProducer final : public Operator<> {
 public:
  explicit ScalingProducer(const std::uint32_t index) : index_(index) {}

  void setup(OperatorSpec& spec) override {
    output_count_ = g_state->options.delivery == "direct" ? g_state->options.fanout : 1U;
    for (std::uint32_t index = 0U; index < output_count_; ++index) {
      spec.output(outputs_[index], "output-" + std::to_string(index)).max_emits_per_compute(1U);
    }
  }

  [[nodiscard]] Contract contract() const override {
    const auto& options = g_state->options;
    Contract result;
    result
        .trigger(OnClock{.period = std::chrono::nanoseconds{std::max<std::int64_t>(
                             1, 1'000LL * options.period_us)}})
        .complete_after(ComputeCalls{.limit = options.warmup + options.samples});
    return result;
  }

  [[nodiscard]] const Output<BenchmarkPOD64>& output(const std::uint32_t sink) const {
    return outputs_[output_count_ == 1U ? 0U : sink];
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    State& state = *g_state;
    auto& record = state.producers[index_];
    const std::uint64_t sequence = record.publications + 1U;
    if (record.tid == 0U)
      record.tid = benchmarking::benchmark_thread_identity().value;
    if (sequence == state.options.warmup + 1U) {
      state.measurement_ready.fetch_add(1U, std::memory_order_acq_rel);
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::milliseconds{state.options.timeout_ms};
      while (state.measurement_ready.load(std::memory_order_acquire) != state.options.producers) {
        if (context.cancellation().stop_requested())
          return make_unexpected(Error{ErrorCode::kClosed});
        if (std::chrono::steady_clock::now() >= deadline)
          return make_unexpected(Error{ErrorCode::kTimeout});
        std::this_thread::yield();
      }
    }
    const auto start = BenchmarkMonotonicClock::now_ns();
    BenchmarkPOD64 payload{
        .sequence = sequence, .source_start_time_ns = start, .auxiliary0 = index_};
    payload.checksum = benchmark_pod_checksum(payload);
    for (std::uint32_t output = 0U; output < output_count_; ++output) {
      // Retry only a rejected attempt on this output, never an already accepted prefix.
      // The timed window includes pressure; successful EOS proves the final tail drains.
      for (;;) {
        auto emitted = outputs_[output].emit(payload, EmitOptions{.frame_id = sequence});
        if (emitted)
          break;
        if (emitted.error().code != ErrorCode::kBackpressured)
          return make_unexpected(std::move(emitted.error()));
        ++record.retries;
        if (context.cancellation().stop_requested())
          return make_unexpected(Error{ErrorCode::kClosed});
        std::this_thread::yield();
      }
    }
    if (sequence > state.options.warmup) {
      const auto slot = static_cast<std::size_t>(sequence - state.options.warmup - 1U);
      record.starts[slot] = start;
      record.emits[slot] = static_cast<std::uint64_t>(BenchmarkMonotonicClock::now_ns() - start);
    }
    ++record.publications;
    return {};
  }

 private:
  std::uint32_t index_{};
  std::uint32_t output_count_{};
  std::array<Output<BenchmarkPOD64>, 4U> outputs_;
};

[[nodiscard]] expected<void, Error> consume_input(Input<BenchmarkPOD64>& input,
                                                  const std::uint32_t index) {
  State& state = *g_state;
  auto& record = state.sinks[index];
  if (record.tid == 0U)
    record.tid = benchmarking::benchmark_thread_identity().value;
  auto sample = input.receive();
  if (!sample)
    return make_unexpected(std::move(sample.error()));
  const auto received = BenchmarkMonotonicClock::now_ns();
  const auto sequence = record.deliveries + 1U;
  if (sample->metadata.frame_id != sequence || sample->data.sequence != sequence ||
      sample->data.auxiliary0 != index / state.options.fanout ||
      !benchmark_pod_valid(sample->data)) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  if (sequence > state.options.warmup) {
    const auto slot = static_cast<std::size_t>(sequence - state.options.warmup - 1U);
    if (slot >= record.receives.size())
      return make_unexpected(Error{ErrorCode::kProtocolError});
    record.receives[slot] = received;
  }
  ++record.deliveries;
  callback_work(state.options.sink_work_us);
  if (index == 0U && state.options.stall_us != 0U &&
      sequence == state.options.warmup + state.options.samples / 2U + 1U) {
    // Deliberately suspend a callback holding its input witness, not its cache mutex.
    record.stall_start = BenchmarkMonotonicClock::now_ns();
    std::this_thread::sleep_for(std::chrono::microseconds{state.options.stall_us});
    record.stall_end = BenchmarkMonotonicClock::now_ns();
  }
  return {};
}

class ScalingSink final : public Operator<> {
 public:
  explicit ScalingSink(const std::uint32_t index) : index_(index) {}
  Input<BenchmarkPOD64> input;

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(g_state->options.queue_depth);
  }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    return consume_input(input, index_);
  }

 private:
  std::uint32_t index_{};
};

// One OnAnyInput activation selects the arriving stream; no multi-input join is introduced.
class ScalingSharedSink final : public Operator<> {
 public:
  explicit ScalingSharedSink(const std::uint32_t sink) : sink_(sink) {}
  [[nodiscard]] const Input<BenchmarkPOD64>& input(const std::uint32_t producer) const {
    return inputs_[producer];
  }
  void setup(OperatorSpec& spec) override {
    for (std::uint32_t producer = 0U; producer < g_state->options.producers; ++producer)
      spec.input(inputs_[producer], "input-" + std::to_string(producer))
          .queue_depth(g_state->options.queue_depth);
  }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnAnyInput{});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    bool consumed = false;
    for (std::uint32_t producer = 0U; producer < g_state->options.producers; ++producer) {
      if (inputs_[producer].is_missing())
        continue;
      auto result = consume_input(inputs_[producer], producer * g_state->options.fanout + sink_);
      if (!result)
        return result;
      consumed = true;
    }
    if (!consumed)
      return make_unexpected(Error{ErrorCode::kProtocolError});
    return {};
  }

 private:
  std::uint32_t sink_{};
  std::array<Input<BenchmarkPOD64>, 4U> inputs_;
};

[[nodiscard]] std::uint32_t sink_lane_count(const Options& options) {
  return options.shared_destinations ? options.fanout : options.producers * options.fanout;
}

[[nodiscard]] ExecutionPlan make_plan(const Options& options) {
  Graph graph{"fanout-scaling-" + std::to_string(BenchmarkMonotonicClock::now_ns())};
  const auto clock = graph.add_clock<RealtimeClock>("clock");
  graph.set_default_clock(clock);
  CompileOptions compile_options;
  const auto bind = [&](const std::string& name, const std::uint32_t ordinal) {
    compile_options.deployment.bind_thread_placement(
        name,
        benchmarking::make_placement(
            options.first_cpu < 0 ? -1 : options.first_cpu + static_cast<int>(ordinal),
            options.busy_poll));
  };
  std::vector<PartitionHandle> destinations;
  std::vector<OperatorHandle<ScalingSharedSink>> shared;
  for (std::uint32_t index = 0U; index < sink_lane_count(options); ++index) {
    const auto name = "sink-partition-" + std::to_string(index);
    destinations.push_back(graph.partition(name));
    bind(name, options.producers + index);
    if (options.shared_destinations) {
      auto child = graph.op<ScalingSharedSink>("destination-" + std::to_string(index), index);
      destinations.back().add(child);
      shared.push_back(child);
    }
  }
  for (std::uint32_t producer = 0U; producer < options.producers; ++producer) {
    const auto name = "producer-" + std::to_string(producer);
    auto source = graph.op<ScalingProducer>(name, producer);
    graph.partition(name).add(source);
    bind(name, producer);
    for (std::uint32_t sink = 0U; sink < options.fanout; ++sink) {
      const ConnectionOptions connection{.queue_policy = QueuePolicy::kBlocking,
                                         .queue_depth = options.queue_depth};
      if (options.shared_destinations) {
        graph.add_flow(source->output(sink), shared[sink]->input(producer), connection);
      } else {
        const auto index = producer * options.fanout + sink;
        auto target = graph.op<ScalingSink>("sink-" + std::to_string(index), index);
        destinations[index].add(target);
        graph.add_flow(source->output(sink), target->input, connection);
      }
    }
  }
  if (options.controller_cpu >= 0)
    compile_options.deployment.bind_support_thread_affinity(
        holoscan::SupportThreadRole::kController,
        CPUAffinity{static_cast<std::uint32_t>(options.controller_cpu)});
  if (options.pump_cpu >= 0)
    compile_options.deployment.bind_support_thread_affinity(
        holoscan::SupportThreadRole::kChannelPump,
        CPUAffinity{static_cast<std::uint32_t>(options.pump_cpu)});
  if (options.delivery == "emit")
    return detail::compile_with_process_local_fanout_experiment(
        graph, std::move(compile_options), detail::ProcessLocalFanoutExperiment::kEmitRecipients);
  if (options.delivery == "pump")
    return detail::compile_with_process_local_pump_delivery(graph, std::move(compile_options));
  return compile(graph, std::move(compile_options));
}

void emit_distribution(std::ostream& output, std::vector<std::uint64_t> values) {
  const auto summary = benchmarking::summarize_percentiles(values);
  output << "{\"samples\":" << summary.count << ",\"p50\":" << summary.p50
         << ",\"p99\":" << summary.p99 << ",\"p999\":";
  if (summary.count >= 10'000U)
    output << summary.p999;
  else
    output << "null";
  output << ",\"p9999\":";
  if (summary.count >= 100'000U) {
    std::sort(values.begin(), values.end());
    output << values[values.size() - values.size() / 10'000U - 1U];
  } else {
    output << "null";
  }
  output << ",\"max\":" << summary.p_max << '}';
}

[[nodiscard]] bool valid_callbacks(const State& state) {
  const auto expected = state.options.warmup + state.options.samples;
  std::set<std::uint64_t> owners;
  for (const auto& record : state.producers) {
    if (record.publications != expected || record.tid == 0U || !owners.insert(record.tid).second)
      return false;
  }
  std::vector<std::uint64_t> sink_owners(sink_lane_count(state.options));
  for (std::size_t index = 0U; index < state.sinks.size(); ++index) {
    const auto& record = state.sinks[index];
    if (record.deliveries != expected || record.tid == 0U || owners.contains(record.tid))
      return false;
    auto& owner =
        sink_owners[state.options.shared_destinations ? index % state.options.fanout : index];
    if (owner != 0U && owner != record.tid)
      return false;
    owner = record.tid;
    const auto& source = state.producers[index / state.options.fanout];
    for (std::size_t sample = 0U; sample < record.receives.size(); ++sample) {
      if (source.starts[sample] <= 0 || record.receives[sample] < source.starts[sample])
        return false;
    }
  }
  return std::set<std::uint64_t>(sink_owners.begin(), sink_owners.end()).size() ==
         sink_owners.size();
}

[[nodiscard]] std::int64_t emit_producers(std::ostream& output, const State& state) {
  const auto& options = state.options;
  std::int64_t start = std::numeric_limits<std::int64_t>::max();
  std::int64_t finish{};
  for (std::size_t producer = 0U; producer < state.producers.size(); ++producer) {
    if (producer != 0U)
      output << ',';
    const auto& record = state.producers[producer];
    std::vector<std::uint64_t> latency;
    for (std::size_t index = 0U; index < record.starts.size(); ++index) {
      if (record.starts[index] <= 0)
        continue;
      std::int64_t received{};
      for (std::uint32_t sink = 0U; sink < options.fanout; ++sink)
        received =
            std::max(received, state.sinks[producer * options.fanout + sink].receives[index]);
      if (received < record.starts[index])
        continue;
      start = std::min(start, record.starts[index]);
      finish = std::max(finish, received);
      latency.push_back(static_cast<std::uint64_t>(received - record.starts[index]));
    }
    output << "{\"index\":" << producer << ",\"publications\":" << record.publications
           << ",\"backpressure_retries\":" << record.retries << ",\"thread\":" << record.tid
           << ",\"first_start_ns\":" << record.starts.front()
           << ",\"last_start_ns\":" << record.starts.back() << ",\"latency_ns\":";
    emit_distribution(output, std::move(latency));
    output << ",\"emit_ns\":";
    emit_distribution(output, record.emits);
    output << '}';
  }
  return finish > start ? finish - start : 0;
}

void emit_lanes(std::ostream& output, const BenchmarkSupportThreadControl& threads) {
  bool first_lane = true;
  for (const auto& lane : threads.lane_evidence()) {
    if (!first_lane)
      output << ',';
    first_lane = false;
    output << "{\"lane\":" << lane.execution_lane
           << ",\"disposition\":" << static_cast<unsigned>(lane.disposition) << ",\"affinity\":[";
    const auto applied = lane.applied.value_or(detail::ExecutionLanePlacementSnapshot{});
    for (std::size_t index = 0; index < applied.affinity.cores.size(); ++index) {
      if (index != 0U)
        output << ',';
      output << applied.affinity.cores[index];
    }
    output << "],\"policy\":" << static_cast<unsigned>(applied.policy) << '}';
  }
  output << ']';
}

// Emit only after every Runtime thread joins; no result aggregation touches callback-owned storage.
void emit_result(std::ostream& output, const State& state, const bool valid,
                 const RunSession& session, const BenchmarkSupportThreadControl& threads,
                 const detail::RuntimePhaseCounters* counters) {
  const auto& options = state.options;
  output << "{\"schema\":\"holoscan.fanout_scaling.v1\",\"valid\":" << (valid ? "true" : "false")
         << ",\"completed\":"
         << (session.termination() == RunTermination::kCompleted ? "true" : "false")
         << ",\"producer_count\":" << options.producers << ",\"fanout\":" << options.fanout
         << ",\"samples\":" << options.samples << ",\"warmup\":" << options.warmup
         << ",\"delivery\":\"" << options.delivery
         << "\",\"shared_destinations\":" << (options.shared_destinations ? "true" : "false")
         << ",\"period_us\":" << options.period_us << ",\"queue_depth\":" << options.queue_depth
         << ",\"sink_work_us\":" << options.sink_work_us << ",\"stall_us\":" << options.stall_us
         << ",\"concurrent_admission\":" << (state.concurrent_admission ? "true" : "false")
         << ",\"publication_credits\":" << (options.publication_credits ? "true" : "false")
         << ",\"busy_poll\":" << (options.busy_poll ? "true" : "false") << ",\"producers\":[";
  const auto elapsed = emit_producers(output, state);
  output << "],\"measured_window_ns\":" << elapsed << ",\"publications_per_second\":"
         << (elapsed > 0 ? 1e9 * options.samples * options.producers / static_cast<double>(elapsed)
                         : 0.0)
         << ",\"sinks\":[";
  const auto& stalled = state.sinks.front();
  std::uint64_t peer_progress{};
  for (std::size_t index = 0U; index < state.sinks.size(); ++index) {
    const auto& record = state.sinks[index];
    if (index != 0U)
      output << ',';
    output << "{\"index\":" << index << ",\"deliveries\":" << record.deliveries
           << ",\"thread\":" << record.tid << '}';
    if (index >= options.fanout && stalled.stall_start > 0)
      for (const auto time : record.receives)
        peer_progress += time > stalled.stall_start && time < stalled.stall_end ? 1U : 0U;
  }
  output << "],\"stall_start_ns\":" << stalled.stall_start
         << ",\"stall_end_ns\":" << stalled.stall_end
         << ",\"other_producer_deliveries_during_stall\":" << peer_progress
         << ",\"support_threads\":";
  benchmarking::emit_support_thread_evidence(output, threads);
  output << ",\"lane_startup\":[";
  emit_lanes(output, threads);
  output << ",\"phase_counters\":{\"enabled\":" << (counters != nullptr ? "true" : "false");
  if (counters != nullptr) {
    const auto snapshot = counters->snapshot();
    const auto field = [&](const char* name, const detail::RuntimePhaseCounter counter) {
      output << ",\"" << name << "\":" << snapshot[detail::RuntimePhaseCounters::index(counter)];
    };
    field("cuts", detail::RuntimePhaseCounter::kChannelActivationCuts);
    field("cut_drains", detail::RuntimePhaseCounter::kChannelActivationCutDrains);
    field("cut_skips", detail::RuntimePhaseCounter::kChannelActivationCutSkips);
    field("cut_skips_after_lock", detail::RuntimePhaseCounter::kChannelActivationCutSkipsAfterLock);
    field("cut_scoped_skips", detail::RuntimePhaseCounter::kChannelActivationCutScopedSkips);
    field("cut_scoped_retries", detail::RuntimePhaseCounter::kChannelActivationCutScopedRetries);
    field("known_blocked_skips", detail::RuntimePhaseCounter::kChannelKnownBlockedSkips);
    field("pump_waits", detail::RuntimePhaseCounter::kChannelPumpWaits);
    field("pump_waits_ready", detail::RuntimePhaseCounter::kChannelPumpWaitsReady);
    field("pump_wait_timeouts", detail::RuntimePhaseCounter::kChannelPumpWaitTimeouts);
    field("pump_wait_timeouts_blocked",
          detail::RuntimePhaseCounter::kChannelPumpWaitTimeoutsWithBlockedHead);
    field("cuts_post_compute", detail::RuntimePhaseCounter::kChannelActivationCutsPostCompute);
    field("cuts_retirement_retry",
          detail::RuntimePhaseCounter::kChannelActivationCutsRetirementRetry);
    field("cuts_terminal", detail::RuntimePhaseCounter::kChannelActivationCutsTerminal);
    field("cut_drains_post_compute",
          detail::RuntimePhaseCounter::kChannelActivationCutDrainsPostCompute);
    field("cut_drains_retirement_retry",
          detail::RuntimePhaseCounter::kChannelActivationCutDrainsRetirementRetry);
    field("cut_drains_terminal", detail::RuntimePhaseCounter::kChannelActivationCutDrainsTerminal);
    field("cut_miss_blocked_latch",
          detail::RuntimePhaseCounter::kChannelActivationCutMissBlockedLatch);
    field("cut_miss_tracking_invalid",
          detail::RuntimePhaseCounter::kChannelActivationCutMissTrackingInvalid);
    field("cut_miss_in_flight", detail::RuntimePhaseCounter::kChannelActivationCutMissInFlight);
    field("cut_miss_undelivered",
          detail::RuntimePhaseCounter::kChannelActivationCutMissUndelivered);
    field("publications", detail::RuntimePhaseCounter::kChannelEmitAccepted);
    field("scan_lock_acquisitions_emit",
          detail::RuntimePhaseCounter::kChannelScanLockAcquisitionsEmit);
    field("scan_lock_acquisitions_cut",
          detail::RuntimePhaseCounter::kChannelScanLockAcquisitionsCut);
    field("scan_lock_acquisitions_pump",
          detail::RuntimePhaseCounter::kChannelScanLockAcquisitionsPump);
    field("scan_lock_contended_emit", detail::RuntimePhaseCounter::kChannelScanLockContendedEmit);
    field("scan_lock_contended_cut", detail::RuntimePhaseCounter::kChannelScanLockContendedCut);
    field("scan_lock_contended_pump", detail::RuntimePhaseCounter::kChannelScanLockContendedPump);
    field("scan_lock_wait_ns_emit", detail::RuntimePhaseCounter::kChannelScanLockWaitNsEmit);
    field("scan_lock_wait_ns_cut", detail::RuntimePhaseCounter::kChannelScanLockWaitNsCut);
    field("scan_lock_wait_ns_pump", detail::RuntimePhaseCounter::kChannelScanLockWaitNsPump);
    field("inventory_scans", detail::RuntimePhaseCounter::kChannelActivationCutInventoryScans);
    field("subscriber_probes", detail::RuntimePhaseCounter::kChannelActivationCutSubscriberProbes);
    field("transport_probes", detail::RuntimePhaseCounter::kChannelActivationCutTransportProbes);
    field("received", detail::RuntimePhaseCounter::kChannelSamplesReceived);
    field("receive_backpressure", detail::RuntimePhaseCounter::kChannelReceiveBackpressured);
    field("waits", detail::RuntimePhaseCounter::kWaitCalls);
  }
  output << "},\n";
  benchmarking::emit_runtime_thread_usage(output, counters);
  output << "\"diagnostics\":";
  benchmarking::emit_runtime_diagnostics(output, session.diagnostics());
  output << "}\n";
}

[[nodiscard]] int run_scaling(Options options) {
  State state{std::move(options)};
  g_state = &state;
  auto plan = make_plan(state.options);
  if (!plan.ok()) {
    for (const auto& diagnostic : plan.diagnostics())
      std::cerr << diagnostic.token << '\n';
    return 2;
  }
  // compile() delivers these inline payloads inside emit(), so only the pump control runs a pump.
  const bool pump = state.options.delivery == "pump";
  auto threads = std::make_shared<BenchmarkSupportThreadControl>(
      std::vector<SupportThreadBinding>{{.role = SupportThreadRole::kController,
                                         .requested_cpu = state.options.controller_cpu,
                                         .required = true},
                                        {.role = SupportThreadRole::kChannelPump,
                                         .requested_cpu = state.options.pump_cpu,
                                         .required = pump,
                                         .forbidden = !pump}});
  plan = detail::RuntimeTestPlanAccess::with_support_thread_control(plan, threads);
  plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(plan, threads);
  if (!state.options.publication_credits) {
    auto control = std::make_shared<detail::RuntimeFanoutTestControl>();
    control->disable_publication_credits = true;
    plan = detail::RuntimeTestPlanAccess::with_fanout_test_control(plan, std::move(control));
  }
  if (const auto accepted = detail::ExecutionPlanAccess::retain(plan); accepted != nullptr) {
    state.concurrent_admission =
        std::ranges::any_of(accepted->process_local_channel_services,
                            [](const auto& service) { return service.emit_admitted; });
  }
  std::shared_ptr<detail::RuntimePhaseCounters> counters;
  if (state.options.counters) {
    // Counters only: a timestamped phase trace would distort the saturation control further.
    counters = std::make_shared<detail::RuntimePhaseCounters>();
    plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
  }
  auto session = run_async(plan);
  const bool terminal = session.wait_for(std::chrono::milliseconds{state.options.timeout_ms});
  if (!terminal)
    session.request_stop();
  session.wait();
  const auto evidence = threads->lane_evidence();
  const bool lanes_valid =
      evidence.size() == state.options.producers + sink_lane_count(state.options) &&
      std::all_of(evidence.begin(), evidence.end(), [](const auto& lane) {
        return lane.disposition == detail::ExecutionLaneStartupDisposition::kReady &&
               lane.applied.has_value();
      });
  const bool valid = terminal && session.termination() == RunTermination::kCompleted &&
                     session.diagnostics().empty() && threads->required_roles_entered() &&
                     lanes_valid && valid_callbacks(state);
  std::ofstream file;
  if (!state.options.output.empty())
    file.open(state.options.output);
  if (!state.options.output.empty() && !file)
    return 2;
  std::ostream& output = file.is_open() ? file : std::cout;
  emit_result(output, state, valid, session, *threads, counters.get());
  output.flush();
  g_state = nullptr;
  if (!output) {
    std::cerr << "failed to write the fanout scaling result\n";
    return 2;
  }
  return valid ? 0 : 2;
}

[[nodiscard]] bool set_option(Options& options, const std::string_view key,
                              const std::string_view value) {
  if (key == "--delivery") {
    options.delivery = value;
    return true;
  }
  if (key == "--output") {
    options.output = value;
    return true;
  }
  std::int64_t number{};
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), number);
  if (error != std::errc() || end != value.data() + value.size() || number < -1 ||
      number > 10'000'000)
    return false;
  if (key == "--first-cpu") {
    options.first_cpu = static_cast<int>(number);
    return true;
  }
  if (key == "--controller-cpu") {
    options.controller_cpu = static_cast<int>(number);
    return true;
  }
  if (key == "--pump-cpu") {
    options.pump_cpu = static_cast<int>(number);
    return true;
  }
  if (number < 0)
    return false;
  using Field = std::pair<std::string_view, std::uint32_t Options::*>;
  constexpr std::array fields{Field{"--producers", &Options::producers},
                              Field{"--fanout", &Options::fanout},
                              Field{"--samples", &Options::samples},
                              Field{"--warmup", &Options::warmup},
                              Field{"--queue-depth", &Options::queue_depth},
                              Field{"--period-us", &Options::period_us},
                              Field{"--sink-work-us", &Options::sink_work_us},
                              Field{"--stall-us", &Options::stall_us},
                              Field{"--timeout-ms", &Options::timeout_ms}};
  const auto found = std::find_if(
      fields.begin(), fields.end(), [&](const auto& field) { return field.first == key; });
  if (found == fields.end())
    return false;
  options.*(found->second) = static_cast<std::uint32_t>(number);
  return true;
}

[[nodiscard]] std::optional<Options> parse(const int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view key{argv[index]};
    if (key == "--shared-destinations") {
      options.shared_destinations = true;
    } else if (key == "--busy-poll") {
      options.busy_poll = true;
    } else if (key == "--no-publication-credits") {
      options.publication_credits = false;
    } else if (key == "--counters") {
      options.counters = true;
    } else {
      if (++index == argc || !set_option(options, key, argv[index]))
        return std::nullopt;
    }
  }
  if (options.producers < 1U || options.producers > 4U || options.fanout < 2U ||
      options.fanout > 4U || options.samples == 0U || options.samples > 1'000'000U ||
      options.warmup > 100'000U || options.queue_depth == 0U || options.queue_depth > 64U ||
      options.stall_us > 100'000U || options.sink_work_us > 1'000U || options.timeout_ms == 0U ||
      (options.delivery != "baseline" && options.delivery != "pump" && options.delivery != "emit" &&
       options.delivery != "direct") ||
      (options.pump_cpu >= 0 && options.delivery != "pump"))
    return std::nullopt;
  const auto lane_end =
      options.first_cpu + static_cast<int>(options.producers + sink_lane_count(options));
  for (const int cpu : {options.controller_cpu, options.pump_cpu})
    if (cpu >= 0 && options.first_cpu >= 0 && cpu >= options.first_cpu && cpu < lane_end)
      return std::nullopt;
  if (options.controller_cpu >= 0 && options.controller_cpu == options.pump_cpu)
    return std::nullopt;
  return options;
}

}  // namespace
}  // namespace holoscan

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    std::cout
        << "Synthetic concurrent fanout: --producers 1..4 --fanout 2..4\n"
           "--delivery baseline|pump|emit|direct --shared-destinations --samples N --warmup N\n"
           "--queue-depth N --period-us N (0: saturated clock) --sink-work-us N --stall-us N\n"
           "--first-cpu N --controller-cpu N --pump-cpu N --busy-poll\n"
           "--no-publication-credits (turn off the default pre-submit transport credit check)\n"
           "--counters --timeout-ms N --output PATH\n";
    return 0;
  }
  auto options = holoscan::parse(argc, argv);
  if (!options) {
    std::cerr << "invalid fanout scaling options; see --help\n";
    return 2;
  }
  return holoscan::run_scaling(std::move(*options));
}
