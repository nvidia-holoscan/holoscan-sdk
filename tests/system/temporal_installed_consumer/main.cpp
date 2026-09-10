/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/predicate.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/temporal_contract_records.hpp>
#include <holoscan/holoscan.hpp>

namespace {

using namespace std::chrono_literals;

class InstalledTemporalOperator final : public holoscan::Operator<> {
 public:
  holoscan::Input<std::uint32_t> trigger_{"trigger"};
  holoscan::Input<std::uint32_t> latest_{"latest"};

  holoscan::Contract contract() const override {
    holoscan::Contract value;
    value.trigger(holoscan::OnEach{trigger_})
        .input(latest_, holoscan::LatestWithin{.max_age = 5ms})
        .on_missing(latest_, holoscan::ReuseLast{.max_age = 20ms})
        .deadline(2ms)
        .on_violation(holoscan::OnViolation::kFailRun);
    return value;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return {};
  }
};

struct InstalledFiniteSourceConfig {
  std::uint64_t compute_call_limit{};
  std::uint64_t restart_after{};
};

HOLOSCAN_PARAM_SCHEMA(
    InstalledFiniteSourceConfig,
    holoscan::param_schema("nvidia.holoscan.test.InstalledFiniteSourceConfig", 1),
    holoscan::field<&InstalledFiniteSourceConfig::compute_call_limit>("compute_call_limit"),
    holoscan::field<&InstalledFiniteSourceConfig::restart_after>("restart_after"));

// This installed-consumer scenario intentionally proves terminal drain after a published restart.
// Task 036 discards transport that is still queued at the restart cut, so make the first accepted
// delivery observable by the independent sink partition before the source asks to restart.
struct InstalledFiniteDeliveryState {
  [[nodiscard]] static bool wait_for_sink_calls(const std::uint64_t expected) noexcept {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, 5s, [expected] { return sink_calls.load(std::memory_order_acquire) >= expected; });
  }

  static void record_sink_call() noexcept {
    {
      const std::lock_guard lock{mutex};
      sink_calls.fetch_add(1U, std::memory_order_release);
    }
    ready.notify_all();
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::uint64_t> sink_calls{};
  static inline std::atomic_bool restart_delivery_timed_out{};
};

class InstalledFiniteSource final : public holoscan::Operator<InstalledFiniteSourceConfig> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  holoscan::Contract contract() const override {
    holoscan::Contract value;
    value.trigger(holoscan::OnClock{.period = 1ms})
        .complete_after(holoscan::ComputeCalls{.limit = params().compute_call_limit});
    return value;
  }

  void start() override { starts.fetch_add(1U, std::memory_order_relaxed); }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    const InstalledFiniteSourceConfig& config = context.params<InstalledFiniteSourceConfig>();
    const std::uint64_t call = calls.fetch_add(1U, std::memory_order_relaxed) + 1U;
    const std::uint32_t generation = context.tag().generation;
    if (call == 1U) {
      first_generation.store(generation, std::memory_order_relaxed);
    }
    last_generation.store(generation, std::memory_order_relaxed);
    auto emitted = output.emit(static_cast<std::uint32_t>(call));
    if (!emitted) {
      return emitted;
    }
    if (call == config.restart_after) {
      if (!InstalledFiniteDeliveryState::wait_for_sink_calls(call)) {
        InstalledFiniteDeliveryState::restart_delivery_timed_out.store(true,
                                                                       std::memory_order_release);
        return holoscan::make_unexpected(holoscan::Error{
            holoscan::ErrorCode::kTimeout, "INSTALLED_RESTART_DELIVERY_BARRIER_TIMEOUT"});
      }
      auto receipt = context.request_restart();
      if (!receipt) {
        return holoscan::make_unexpected(std::move(receipt).error());
      }
      restart_accepted.store(true, std::memory_order_release);
    }
    return {};
  }

  static inline std::atomic<std::uint64_t> calls{};
  static inline std::atomic<std::uint64_t> starts{};
  static inline std::atomic<std::uint32_t> first_generation{};
  static inline std::atomic<std::uint32_t> last_generation{};
  static inline std::atomic_bool restart_accepted{};
  holoscan::Output<std::uint32_t> output;
};

class InstalledFiniteSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  holoscan::Contract contract() const override {
    holoscan::Contract value;
    value.trigger(holoscan::OnEach{input});
    return value;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto received = input.receive();
    if (!received) {
      return holoscan::make_unexpected(std::move(received).error());
    }
    InstalledFiniteDeliveryState::record_sink_call();
    return {};
  }

  holoscan::Input<std::uint32_t> input;
};

static_assert(std::is_copy_constructible_v<holoscan::Contract>);
static_assert(std::is_copy_constructible_v<holoscan::PredicateBinding>);
template <typename T>
concept HasMaxSamples = requires(T value) { value.max_samples; };

static_assert(!HasMaxSamples<holoscan::DrainAll>);

}  // namespace

// REQ: CONTRACT-013
// REQ: CONTRACT-020
// REQ: RUNTIME-050
// REQ: CHAN-136
int main() {
  InstalledTemporalOperator op;
  auto contract = op.contract();
  auto copied = contract;
  constexpr InstalledFiniteSourceConfig kConfig{
      .compute_call_limit = 3U,
      .restart_after = 1U,
  };
  InstalledFiniteSource finite_source;
  auto finite_contract = finite_source.contract();
  auto finite_copy = finite_contract;
  auto binding =
      holoscan::predicate(holoscan::PredicateKey{"nvidia.holoscan.installed.always", {1, 0, 0}},
                          [](const holoscan::PredicateEvaluationContext&) -> bool { return true; });
  if (!binding || binding.name() != "nvidia.holoscan.installed.always") {
    return 1;
  }
  const holoscan::ContractRecord record{
      .operator_name = "installed",
      .trigger = holoscan::TemporalRecordTriggerKind::kAnyInput,
      .trigger_pins = {{.input_name = "trigger",
                        .retained_pin_capacity = 1,
                        .capacity_origin = holoscan::TemporalRecordBoundOrigin::kResolvedRetention,
                        .provenance = {.source = "compiler.trigger-pin", .declaration_order = 1}}},
      .provenance = {.source = "operator.contract", .declaration_order = 0},
  };
  const std::vector<holoscan::EdgeSelectionRecord> edges;
  const auto serialized = holoscan::serialize_temporal_records(record, edges);
  if (!serialized || serialized->empty()) {
    return 2;
  }
  static_cast<void>(copied);
  static_cast<void>(finite_copy);

  // A real application can populate this typed value from its configuration file before compile.
  // The value belongs to this source Operator's Contract; it is not a workflow iteration count.
  holoscan::Graph graph{"installed-configured-completion"};
  auto source = graph.op<InstalledFiniteSource>("configured-source");
  source.set_params(kConfig);
  const auto sink = graph.op<InstalledFiniteSink>("ordinary-eos-sink");
  graph.partition("configured-source-worker").add(source);
  graph.partition("ordinary-sink-worker").add(sink);
  graph.add_flow(source->output, sink->input, holoscan::ConnectionOptions{.queue_depth = 1U});
  graph.lifecycle().allow_self_restart(source);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));
  holoscan::CompileOptions options;
  options.deployment.bind_thread_placement("configured-source-worker", holoscan::ThreadPlacement{})
      .bind_thread_placement("ordinary-sink-worker", holoscan::ThreadPlacement{});
  options.lifecycle_transactions = holoscan::LifecycleTransactionOptions{
      .max_pending_requests = 64U,
      .max_pending_transactions = 64U,
      .max_receipts = 128U,
      .max_fenced_operators = 64U,
      .max_candidate_generations = 64U,
      .max_events = 1024U,
      .max_audit_records = 128U,
      .max_quiesce_records = 256U,
      .max_rollback_records = 256U,
  };
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok() || plan.operator_completion_rules().size() != 1U ||
      plan.operator_completion_rules().front().limit != kConfig.compute_call_limit) {
    std::cerr << "installed completion plan rejected: ok=" << plan.ok()
              << " rules=" << plan.operator_completion_rules().size();
    if (!plan.operator_completion_rules().empty()) {
      std::cerr << " actual-owner=" << plan.operator_completion_rules().front().owner.id
                << " expected-limit=" << kConfig.compute_call_limit
                << " actual-limit=" << plan.operator_completion_rules().front().limit;
    }
    for (const auto& diagnostic : plan.diagnostics()) {
      std::cerr << ' ' << diagnostic.token << "(" << diagnostic.provided << ')';
    }
    std::cerr << '\n';
    return 3;
  }
  const holoscan::OperatorRef completion_owner = plan.operator_completion_rules().front().owner;

  holoscan::RunSession session = holoscan::run_async(plan);
  session.wait();
  if (InstalledFiniteDeliveryState::restart_delivery_timed_out.load(std::memory_order_acquire)) {
    std::cerr << "installed restart delivery barrier timed out before self-restart\n";
    return 4;
  }
  if (session.termination() != holoscan::RunTermination::kCompleted ||
      !session.diagnostics().empty() ||
      InstalledFiniteSource::calls.load(std::memory_order_acquire) != kConfig.compute_call_limit ||
      InstalledFiniteSource::starts.load(std::memory_order_acquire) != 2U ||
      !InstalledFiniteSource::restart_accepted.load(std::memory_order_acquire) ||
      InstalledFiniteSource::first_generation.load(std::memory_order_relaxed) >=
          InstalledFiniteSource::last_generation.load(std::memory_order_relaxed)) {
    return 4;
  }
  const auto progress = session.operator_completion_progress();
  const std::uint64_t sink_calls =
      InstalledFiniteDeliveryState::sink_calls.load(std::memory_order_acquire);
  if (progress.size() != 1U || progress.front().owner.id != completion_owner.id ||
      progress.front().configured_limit != kConfig.compute_call_limit ||
      progress.front().successful_compute_calls != kConfig.compute_call_limit ||
      !progress.front().limit_reached || sink_calls != kConfig.compute_call_limit) {
    std::cerr << "installed completion evidence mismatch: progress-size=" << progress.size()
              << " sink-calls=" << sink_calls << " expected=" << kConfig.compute_call_limit;
    if (!progress.empty()) {
      std::cerr << " owner=" << progress.front().owner.id
                << " configured=" << progress.front().configured_limit
                << " successful=" << progress.front().successful_compute_calls
                << " reached=" << progress.front().limit_reached;
    }
    std::cerr << '\n';
    return 5;
  }
  return 0;
}
