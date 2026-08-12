/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Smoke: a single spin-driven operator with a runtime_mutable parameter receives
// a mid-run patch, observes the updated value in its next compute cycle, and the
// parameter control API correctly sequences generation numbers and update states
// (Pending → Published → Observed).

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_control.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/manual_clock.hpp>

#include "runtime/runtime_test_access.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

// Parameter bag for the operator under test; "value" is declared runtime_mutable
// so the parameter control API can patch it between compute cycles.
struct UpdateSmokeParams {
  std::int32_t value{7};
};

HOLOSCAN_PARAM_SCHEMA(UpdateSmokeParams, param_schema("nvidia.holoscan.test.UpdateSmokeParams", 1),
                      field<&UpdateSmokeParams::value>("value").runtime_mutable());

// Records what the operator saw for "value" and the param generation both before
// and after the mid-compute barrier, plus the pointer identity of the params struct
// to verify that a new snapshot is served (or the same one when no update occurred).
struct UpdateSmokeObservation {
  std::int32_t before_value{};
  std::int32_t after_value{};
  ParamGeneration before_generation{};
  ParamGeneration after_generation{};
  const UpdateSmokeParams* before_address{};
  const UpdateSmokeParams* after_address{};
};

// Shared state between the operator thread and the main-thread test driver.
// The barrier (release_first) holds the first compute call mid-flight so the
// test can submit a patch before the operator re-reads its params.
struct UpdateSmokeState {
  static void reset() {
    const std::lock_guard lock(mutex);
    observations.clear();
    release_first = false;
  }

  [[nodiscard]] static bool wait_for_first() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return !observations.empty(); });
  }

  static void release() {
    const std::lock_guard lock(mutex);
    release_first = true;
    cv.notify_all();
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::vector<UpdateSmokeObservation> observations;
  static inline bool release_first{};
};

// Spin-driven operator that records its param snapshot on each compute cycle.
// On the first invocation it waits at the release barrier so the test driver can
// inject a patch before the operator calls context.params() a second time.
class UpdateSmokeOperator final : public Operator<UpdateSmokeParams> {
 public:
  void setup(OperatorSpec& spec) override { spec.spin_source(spin_, "activation").capacity(1); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnSpin{.event = spin_});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const UpdateSmokeParams& before = context.params<UpdateSmokeParams>();
    const ParamGeneration before_generation = context.params_generation();
    std::size_t index{};
    {
      std::unique_lock lock(UpdateSmokeState::mutex);
      index = UpdateSmokeState::observations.size();
      UpdateSmokeState::observations.push_back(UpdateSmokeObservation{
          .before_value = before.value,
          .after_value = before.value,
          .before_generation = before_generation,
          .after_generation = before_generation,
          .before_address = &before,
          .after_address = &before,
      });
      UpdateSmokeState::cv.notify_all();
      if (index == 0U && !UpdateSmokeState::cv.wait_for(
                             lock, 5s, [] { return UpdateSmokeState::release_first; })) {
        return make_unexpected(
            Error{ErrorCode::kTimeout, "PARAMETER_UPDATE_SMOKE_BARRIER_TIMEOUT"});
      }
    }

    const UpdateSmokeParams& after = context.params<UpdateSmokeParams>();
    const std::lock_guard lock(UpdateSmokeState::mutex);
    auto& observation = UpdateSmokeState::observations[index];
    observation.after_value = after.value;
    observation.after_generation = context.params_generation();
    observation.after_address = &after;
    return {};
  }

 private:
  SpinSource spin_;
};

// Releases the barrier (so the blocked operator can unblock), requests a stop,
// waits for the session to drain, then returns the given failure code.
[[nodiscard]] int stop_with_failure(RunSession& session, int code) {
  UpdateSmokeState::release();
  session.request_stop();
  session.wait();
  return code;
}

}  // namespace
}  // namespace holoscan

int main() {
  using namespace holoscan;

  // [spin source] ──► parameterized(UpdateSmokeOperator)
  //                   params: UpdateSmokeParams{value=7}  →  patched to value=42
  Graph graph{"ea1-parameter-update-smoke"};
  const auto parameterized = graph.op<UpdateSmokeOperator>("parameterized");
  graph.partition("runtime-spin-worker").add(parameterized);
  const ClockKey clock = graph.add_clock<ManualClock>("clock", std::int64_t{0});
  graph.set_default_clock(clock);

  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  options.deployment.bind_thread_placement("runtime-spin-worker",
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  const ExecutionPlan raw_plan = compile(graph, options);
  if (!raw_plan.ok()) {
    return 1;  // exit 1: graph must compile successfully
  }
  const auto field = std::find_if(raw_plan.parameter_fields().begin(),
                                  raw_plan.parameter_fields().end(),
                                  [](const ParameterFieldPlanRecord& candidate) {
                                    return candidate.canonical_name == "value";
                                  });
  if (field == raw_plan.parameter_fields().end()) {
    return 2;  // exit 2: "value" field must appear in the compiled parameter plan
  }
  const ParameterFieldPlanRecord& value_field = *field;
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::enable_direct_count(raw_plan, 2U);

  UpdateSmokeState::reset();
  RunSession session = run_async(plan);
  if (!UpdateSmokeState::wait_for_first()) {
    return stop_with_failure(session, 3);  // exit 3: first compute must fire within 5s
  }
  const auto run_owner = session.operator_ref(value_field.owner);
  if (!run_owner) {
    return stop_with_failure(session, 4);  // exit 4: operator instance must be resolvable
  }
  auto handle_result = session.parameter_control(ParamAccessPolicy{
      .origin = ParamControlOrigin::kLocalApplication,
      .permissions = ParamPermission::kRead | ParamPermission::kUpdate | ParamPermission::kAwait,
  });
  if (!handle_result) {
    return stop_with_failure(session, 5);  // exit 5: parameter control handle must be obtainable
  }
  ParamControlHandle handle = std::move(handle_result).value();
  const ParamAddress address{.operator_instance = *run_owner, .field = value_field.field};
  const std::array patches{
      ParamPatch{.target = address, .value = ParamValue::from(std::int32_t{42})},
  };
  const ParamBatchUpdateResult accepted =
      handle.submit(patches, ParamUpdateOptions{.effective = NextCompute{}});
  if (!accepted.committed || accepted.update_sequence == 0U || accepted.target_generation != 1U) {
    // Exit 6: patch must be committed with a non-zero sequence and generation 1
    return stop_with_failure(session, 6);
  }
  const auto published = handle.read(address, ParamReadView::kPublished);
  if (!published || published->generation != 0U) {
    // Exit 7: published view before barrier release must still show generation 0
    return stop_with_failure(session, 7);
  }

  UpdateSmokeState::release();
  const auto completed = handle.await(accepted.update_sequence, 5s);
  if (!completed || (completed->state != ParamUpdateState::kPublished &&
                     completed->state != ParamUpdateState::kObserved)) {
    // Exit 8: after barrier release, await must confirm the patch reached kPublished/kObserved
    return stop_with_failure(session, 8);
  }
  session.wait();
  if (session.termination() != RunTermination::kCompleted) {
    return 9;  // exit 9: session must terminate via count exhaustion, not an error
  }
  const auto observed = handle.await(accepted.update_sequence, 0ns);
  if (!observed || observed->state != ParamUpdateState::kObserved) {
    return 13;  // exit 13: update must reach kObserved once the session completes
  }

  const std::lock_guard lock(UpdateSmokeState::mutex);
  if (UpdateSmokeState::observations.size() != 2U) {
    return 10;  // exit 10: exactly two compute cycles must have been recorded
  }
  const auto& first = UpdateSmokeState::observations[0];
  const auto& second = UpdateSmokeState::observations[1];
  if (first.before_value != 7 || first.after_value != 7 || first.before_generation != 0U ||
      first.after_generation != 0U || first.before_address != first.after_address) {
    return 11;  // exit 11: first compute must see value=7 at generation 0 (no update yet)
  }
  if (second.before_value != 42 || second.after_value != 42 || second.before_generation != 1U ||
      second.after_generation != 1U || second.before_address != second.after_address) {
    return 12;  // exit 12: second compute must see value=42 at generation 1 (patch applied)
  }
  return 0;
}
