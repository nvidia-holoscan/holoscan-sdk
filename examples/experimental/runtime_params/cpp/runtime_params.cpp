/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string_view>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_catalog.hpp>
#include <holoscan/core/parameter_control.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/parameter_update.hpp>
#include <holoscan/core/profile.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

using namespace std::chrono_literals;

constexpr std::chrono::nanoseconds kTickPeriod = 5ms;
constexpr std::uint32_t kSignal = 10U;  // constant input so the gain effect is unmistakable

// Operators run on runtime threads; main() must not poll or sleep. The gain
// stage records what each callback actually pinned — value, generation, and
// scaled output — and the control code in main() blocks on the condition
// variable until the pipeline demonstrably runs on the expected generation.
struct GainObservation {
  static void reset() noexcept {
    count.store(0U, std::memory_order_relaxed);
    last_gain.store(0.0F, std::memory_order_relaxed);
    last_generation.store(0U, std::memory_order_relaxed);
    last_scaled.store(0.0F, std::memory_order_relaxed);
  }

  static void record(float gain, std::uint64_t generation, float scaled) {
    {
      const std::lock_guard lock(mutex);
      last_gain.store(gain, std::memory_order_relaxed);
      last_generation.store(generation, std::memory_order_relaxed);
      last_scaled.store(scaled, std::memory_order_relaxed);
      count.fetch_add(1U, std::memory_order_release);
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_count(std::size_t expected) {
    std::unique_lock lock(mutex);
    return ready.wait_for(
        lock, 10s, [expected] { return count.load(std::memory_order_acquire) >= expected; });
  }

  [[nodiscard]] static bool wait_for_generation(std::uint64_t generation) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 10s, [generation] {
      return last_generation.load(std::memory_order_acquire) >= generation;
    });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::size_t> count{};
  static inline std::atomic<float> last_gain{};
  static inline std::atomic<std::uint64_t> last_generation{};
  static inline std::atomic<float> last_scaled{};
};

// Link 1 of the runtime-update opt-in chain: the field marker. Without
// `.runtime_mutable()` the field is frozen for the lifetime of the plan and a
// submit() against it rejects with kRequiresRecompile — typed, never silent.
// The macro must sit at namespace scope right beside the struct: it expands to
// ADL free functions, and ADL only finds them in the struct's own namespace.
struct GainParams {
  float gain{1.0F};  // generation-0 value, bound once at session start
};

HOLOSCAN_PARAM_SCHEMA(GainParams, holoscan::param_schema("holoscan.examples.RuntimeGainParams", 1),
                      holoscan::field<&GainParams::gain>("gain")
                          .runtime_mutable()
                          .range(0.0F, 8.0F)
                          .description("Linear gain applied to every sample."));

// A clock-driven source emitting the same reading forever. Keeping the input
// constant means every output change below is attributable to the gain update
// alone — nothing else in the pipeline varies.
class SignalSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(out, "out");  // max_emits_per_compute defaults to 1
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kTickPeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return out.emit(kSignal);
  }

  holoscan::Output<std::uint32_t> out;
};

// The tunable stage. Each activation reads exactly one coherent, pinned
// parameter generation: ctx.params<GainParams>() cannot change underneath a
// running compute() — an update published mid-run becomes visible only to a
// *later* callback, together with the matching ctx.params_generation() value.
class AmplifyStage final : public holoscan::Operator<GainParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(in, "in").queue_depth(4U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{in});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    HOLOSCAN_TRY(const std::uint32_t raw, in.receive_data());
    const GainParams& params = context.params<GainParams>();  // pinned for this callback
    GainObservation::record(
        params.gain, context.params_generation(), params.gain * static_cast<float>(raw));
    return {};
  }

  holoscan::Input<std::uint32_t> in;
};

[[nodiscard]] std::string_view state_name(holoscan::ParamUpdateState state) noexcept {
  switch (state) {
    case holoscan::ParamUpdateState::kPublished:
      return "published";
    case holoscan::ParamUpdateState::kObserved:
      return "observed";
    default:
      return "other";
  }
}

[[nodiscard]] const holoscan::ParameterFieldPlanRecord* find_runtime_gain_field(
    const holoscan::ExecutionPlan& plan) {
  const holoscan::ParameterFieldPlanRecord* gain_field = nullptr;
  for (const holoscan::ParameterFieldPlanRecord& record : plan.parameter_fields()) {
    if (record.canonical_name == "gain") {
      gain_field = &record;
    }
  }
  return gain_field != nullptr && gain_field->mutability == holoscan::ParamMutability::kRuntime
             ? gain_field
             : nullptr;
}

[[nodiscard]] holoscan::expected<holoscan::ParamUpdateState, holoscan::Error> await_observed_update(
    holoscan::ParamControlHandle& handle, holoscan::ParamUpdateSequence update_sequence) {
  holoscan::ParamUpdateState awaited_state = holoscan::ParamUpdateState::kRejected;
  const auto await_deadline = std::chrono::steady_clock::now() + 5s;
  while (std::chrono::steady_clock::now() < await_deadline) {
    const auto waited = handle.await(update_sequence, 250ms);
    if (!waited) {
      return holoscan::make_unexpected(waited.error());
    }
    awaited_state = waited->state;
    if (awaited_state == holoscan::ParamUpdateState::kObserved) {
      break;
    }
    const std::size_t seen = GainObservation::count.load(std::memory_order_acquire);
    (void)GainObservation::wait_for_count(seen + 1U);  // one more activation, then re-query
  }
  return awaited_state;
}

struct EventCounts {
  std::size_t drained_total{};
  std::size_t matching_update{};
  std::size_t gaps{};
};

[[nodiscard]] holoscan::expected<EventCounts, holoscan::Error> drain_update_events(
    holoscan::ParamEventCursor& cursor, holoscan::ParamUpdateSequence update_sequence) {
  std::array<holoscan::ParamEvent, 16U> events{};
  EventCounts counts;
  while (true) {
    const auto drained = cursor.drain(events);
    if (!drained) {
      return holoscan::make_unexpected(drained.error());
    }
    if (*drained == 0U) {
      break;
    }
    counts.drained_total += *drained;
    for (std::size_t index = 0; index < *drained; ++index) {
      if (events[index].kind == holoscan::ParamEventKind::kGap) {
        ++counts.gaps;
      } else if (events[index].update_sequence == update_sequence) {
        ++counts.matching_update;
      }
    }
  }
  return counts;
}

}  // namespace

int main() {
  GainObservation::reset();

  holoscan::Graph graph{"runtime-params"};
  const auto source = graph.op<SignalSource>("source");
  const auto amplify = graph.op<AmplifyStage>("amplify");
  graph.add_flow(source->out, amplify->in, holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // Link 2 of the opt-in chain: the compile-time policy. kEnabledIfAdmitted
  // enables updates only when every required proof is admitted (the Standard
  // profile admits it). Opting in reserves *bounded* capacity — ingress slots,
  // candidate generation storage, audit records — in the compiled plan, so
  // nothing is allocated on the hot path later.
  holoscan::CompileOptions options;
  options.profile = holoscan::Profile::kStandard;
  options.parameters.runtime_updates = holoscan::ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;

  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  // The plan is the address book: every declared parameter field appears as a
  // ParameterFieldPlanRecord carrying the plan-bound owner reference and the
  // stable field identity that ParamAddress needs.
  const holoscan::ParameterFieldPlanRecord* gain_field = find_runtime_gain_field(plan);
  if (gain_field == nullptr) {
    std::cerr << "runtime_params: plan does not expose a runtime-mutable 'gain' field\n";
    return 1;
  }
  std::cout << "plan accepted: field 'gain' is runtime-mutable\n";

  holoscan::RunSession session = holoscan::run_async(plan);

  // Generation 0 binds from the authoring defaults; prove the pipeline runs on
  // it before touching anything.
  if (!GainObservation::wait_for_count(1U)) {
    std::cerr << "runtime_params: timed out waiting for the first activation\n";
    session.request_stop();
    session.wait();
    return 2;
  }
  std::cout << "generation 0 pinned: gain=" << GainObservation::last_gain.load()
            << " scaled=" << GainObservation::last_scaled.load() << '\n';

  // Re-stamp the plan-bound owner with this run's identity. A ParamAddress
  // built from a previous (or future) run rejects as kStaleOperatorRef.
  const auto owner = session.operator_ref(gain_field->owner);
  if (!owner) {
    std::cerr << "runtime_params: operator_ref failed: "
              << holoscan::error_code_name(owner.error().code) << '\n';
    session.request_stop();
    session.wait();
    return 2;
  }

  // The policy is restriction-only: it can narrow the authority the plan and
  // session already admitted, never widen it. This one asks for the full local
  // set; a monitoring component would get a handle.restrict()-ed child instead.
  const holoscan::ParamAccessPolicy policy{
      .origin = holoscan::ParamControlOrigin::kLocalApplication,
      .permissions = holoscan::ParamPermission::kRead | holoscan::ParamPermission::kUpdate |
                     holoscan::ParamPermission::kAwait | holoscan::ParamPermission::kCapture |
                     holoscan::ParamPermission::kObserve,
      .max_event_cursors = 2U,
  };
  auto minted = session.parameter_control(policy);
  if (!minted) {
    std::cerr << "runtime_params: parameter_control failed: "
              << holoscan::error_code_name(minted.error().code) << '\n';
    session.request_stop();
    session.wait();
    return 2;
  }
  holoscan::ParamControlHandle handle = std::move(minted).value();

  // Open the event cursor *before* submitting so the update's transition
  // records are in the stream when we drain it later. The cursor is an
  // independent position — draining it consumes nothing for other observers.
  auto cursor_result = handle.open_event_cursor();
  if (!cursor_result) {
    std::cerr << "runtime_params: open_event_cursor failed: "
              << holoscan::error_code_name(cursor_result.error().code) << '\n';
    session.request_stop();
    session.wait();
    return 2;
  }
  holoscan::ParamEventCursor cursor = std::move(*cursor_result);

  const holoscan::ParamAddress address{.operator_instance = *owner, .field = gain_field->field};

  // Submit: synchronous, typed, and a *reservation* — committed means one
  // candidate generation and its audit evidence are reserved, not that the
  // value is live. Publication happens at the next scheduler-safe compute
  // boundary of the target operator.
  const holoscan::ParamBatchUpdateResult accepted = handle.submit(holoscan::ParamPatch{
      .target = address,
      .value = holoscan::ParamValue::from(2.0F),
  });
  if (!accepted.committed || accepted.target_generation != 1U) {
    std::cerr << "runtime_params: submit was not committed\n";
    session.request_stop();
    session.wait();
    return 3;
  }
  std::cout << "update committed: target_generation=" << accepted.target_generation
            << " update_sequence=" << accepted.update_sequence << '\n';

  // In-process semantic proof: a later callback pins generation 1 and the
  // same constant input now scales by 2.
  if (!GainObservation::wait_for_generation(1U)) {
    std::cerr << "runtime_params: timed out waiting for generation 1 to be pinned\n";
    session.request_stop();
    session.wait();
    return 3;
  }
  std::cout << "generation 1 pinned: gain=" << GainObservation::last_gain.load()
            << " scaled=" << GainObservation::last_scaled.load() << '\n';

  // await() treats kPublished as terminal; a later activation monotonically
  // advances the retained state to kObserved once a compute() has pinned the
  // new generation. We already watched that activation happen above, so this
  // resolves to kObserved — the bounded retry absorbs the bookkeeping instant
  // between "callback entered" and "lineage advanced".
  const auto awaited = await_observed_update(handle, accepted.update_sequence);
  if (!awaited) {
    std::cerr << "runtime_params: await failed: " << holoscan::error_code_name(awaited.error().code)
              << '\n';
    session.request_stop();
    session.wait();
    return 3;
  }
  const holoscan::ParamUpdateState awaited_state = *awaited;
  std::cout << "await resolved: state=" << state_name(awaited_state) << '\n';

  // read() returns one coherent record without waiting. kPublished is the
  // value a future compute() would see right now.
  const auto published = handle.read(address, holoscan::ParamReadView::kPublished);
  if (!published) {
    std::cerr << "runtime_params: read failed: "
              << holoscan::error_code_name(published.error().code) << '\n';
    session.request_stop();
    session.wait();
    return 3;
  }
  const auto published_gain = published->value.as<float>();
  if (!published_gain || *published_gain != 2.0F || published->generation != 1U) {
    std::cerr << "runtime_params: published read-back mismatch\n";
    session.request_stop();
    session.wait();
    return 3;
  }
  std::cout << "read kPublished: gain=" << *published_gain
            << " generation=" << published->generation << '\n';

  // Optimistic concurrency: expected_generation makes the patch conditional on
  // the latest *accepted* generation. This controller pretends it still thinks
  // the run is on generation 0, so the runtime rejects the batch atomically —
  // typed status, no partial effect, no new sequence.
  const holoscan::ParamBatchUpdateResult stale = handle.submit(holoscan::ParamPatch{
      .target = address,
      .value = holoscan::ParamValue::from(3.0F),
      .expected_generation = holoscan::ParamGeneration{0U},
  });
  if (stale.committed || stale.status != holoscan::ParamUpdateStatus::kExpectedGenerationMismatch ||
      stale.update_sequence != 0U) {
    std::cerr << "runtime_params: stale patch was not rejected as expected\n";
    session.request_stop();
    session.wait();
    return 3;
  }
  std::cout << "stale patch rejected: status=expected-generation-mismatch (no sequence assigned)\n";

  // Drain the event stream: value-free transition records (accepted,
  // published, observed...) keyed by update_sequence. Retention overwrite
  // would surface as an explicit kGap record — resynchronize with read() or
  // capture(), never by assuming continuity.
  const auto event_counts = drain_update_events(cursor, accepted.update_sequence);
  if (!event_counts) {
    std::cerr << "runtime_params: cursor drain failed: "
              << holoscan::error_code_name(event_counts.error().code) << '\n';
    session.request_stop();
    session.wait();
    return 3;
  }
  const std::size_t ours = event_counts->matching_update;
  std::cout << "event cursor: drained=" << event_counts->drained_total << " for_our_update=" << ours
            << " gaps=" << event_counts->gaps << '\n';

  // Irreversibly fence the run against further updates — from every path,
  // owning handles and compute-originated submissions alike. Updates accepted
  // before the fence still complete; the result records the final accepted
  // sequence so control code can await everything that made it in.
  const auto fence = session.close_parameter_update_ingress();
  if (!fence || fence->already_closed ||
      fence->terminal_update_sequence != accepted.update_sequence) {
    std::cerr << "runtime_params: ingress fence result unexpected\n";
    session.request_stop();
    session.wait();
    return 3;
  }
  std::cout << "ingress fenced: terminal_update_sequence=" << fence->terminal_update_sequence
            << '\n';

  const holoscan::ParamBatchUpdateResult after_close = handle.submit(holoscan::ParamPatch{
      .target = address,
      .value = holoscan::ParamValue::from(4.0F),
  });
  if (after_close.committed || after_close.status != holoscan::ParamUpdateStatus::kClosed) {
    std::cerr << "runtime_params: post-fence submit was not rejected as closed\n";
    session.request_stop();
    session.wait();
    return 3;
  }
  std::cout << "post-fence submit rejected: status=closed\n";

  session.request_stop();
  session.wait();

  if (session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << "runtime_params: unexpected termination state\n";
    return 4;
  }

  std::cout << "runtime_params: gain 1->2 generation=1 state=" << state_name(awaited_state)
            << " mismatch=rejected events=" << (ours > 0U ? "seen" : "missing")
            << " fence=closed termination=stopped\n";
  return 0;
}
