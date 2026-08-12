/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "common/allocation_counter.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "partition_parity_test_support.hpp"
#include "runtime/execution_lane.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/predicate_evaluator.hpp"
#include "runtime/runtime_callback_context.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_tensor_resources.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime_test_graph.hpp"

#include "run_session_channel_execution_test_support.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;
using namespace test::run_session_channel_execution;

struct RuntimeClosedIslandAllocationProbe {
  static void begin(void* opaque) noexcept {
    auto& self = *static_cast<RuntimeClosedIslandAllocationProbe*>(opaque);
    const std::uint32_t transaction =
        self.transaction_count.fetch_add(1U, std::memory_order_acq_rel) + 1U;
    measuring = transaction == 2U;
    if (measuring) {
      self.observed.store(false, std::memory_order_release);
      active.emplace();
    }
  }

  static void end(void* opaque) noexcept {
    if (!measuring) {
      return;
    }
    auto& self = *static_cast<RuntimeClosedIslandAllocationProbe*>(opaque);
    const std::size_t count = active ? active->count() : std::numeric_limits<std::size_t>::max();
    active.reset();
    self.allocations.store(count, std::memory_order_release);
    self.observed.store(true, std::memory_order_release);
    measuring = false;
  }

  static inline thread_local std::optional<test::ScopedAllocationCounter> active;
  static inline thread_local bool measuring{};
  std::atomic<std::uint32_t> transaction_count{};
  std::atomic<std::size_t> allocations{std::numeric_limits<std::size_t>::max()};
  std::atomic<bool> observed{};
};

struct RuntimeClosedIslandCompletionProbe {
  struct Snapshot {
    std::array<detail::RuntimeCallbackIdentity, 2U> identities{};
    std::array<ErrorCode, 2U> errors{};
    std::array<std::uint64_t, detail::kRuntimeDirectCompositePhaseDeltaCount> phase_deltas{};
    ErrorCode first_error{ErrorCode::kSuccess};
    std::uint8_t callbacks_attempted{};
    std::uint8_t callbacks_completed{};
  };

  static void capture(void* opaque,
                      const detail::RuntimeDirectCompositeCompletionReport& report) noexcept {
    auto& self = *static_cast<RuntimeClosedIslandCompletionProbe*>(opaque);
    const std::lock_guard lock(self.mutex);
    if (!self.snapshot) {
      self.snapshot = Snapshot{
          .identities = report.callback_identities,
          .errors = report.callback_errors,
          .phase_deltas = report.phase_deltas,
          .first_error = report.first_error.code,
          .callbacks_attempted = report.callbacks_attempted,
          .callbacks_completed = report.callbacks_completed,
      };
    }
  }

  [[nodiscard]] std::optional<Snapshot> read() const {
    const std::lock_guard lock(mutex);
    return snapshot;
  }

  mutable std::mutex mutex;
  std::optional<Snapshot> snapshot;
};

void expect_sample_metadata_equal(const SampleMetadata& expected, const SampleMetadata& actual) {
  EXPECT_EQ(actual.version.major, expected.version.major);
  EXPECT_EQ(actual.version.minor, expected.version.minor);
  EXPECT_EQ(actual.sequence, expected.sequence);
  EXPECT_EQ(actual.capture_timestamp_ns, expected.capture_timestamp_ns);
  EXPECT_EQ(actual.source_clock_id, expected.source_clock_id);
  EXPECT_EQ(actual.publish_timestamp_ns, expected.publish_timestamp_ns);
  EXPECT_EQ(actual.publish_clock_id, expected.publish_clock_id);
  EXPECT_EQ(actual.trace_id, expected.trace_id);
  EXPECT_EQ(actual.frame_id, expected.frame_id);
  EXPECT_EQ(actual.byte_size, expected.byte_size);
  EXPECT_EQ(actual.payload_kind, expected.payload_kind);
  EXPECT_EQ(actual.flags, expected.flags);
}

void expect_caller_sample_metadata_preserved(const SampleMetadata& expected,
                                             const SampleMetadata& actual) {
  EXPECT_EQ(actual.version.major, expected.version.major);
  EXPECT_EQ(actual.version.minor, expected.version.minor);
  EXPECT_EQ(actual.capture_timestamp_ns, expected.capture_timestamp_ns);
  EXPECT_EQ(actual.source_clock_id, expected.source_clock_id);
  EXPECT_EQ(actual.trace_id, expected.trace_id);
  EXPECT_EQ(actual.frame_id, expected.frame_id);
  EXPECT_EQ(actual.flags, expected.flags);
}

void expect_input_selection_equal(InputSelection expected, InputSelection actual) {
  EXPECT_EQ(actual.state, expected.state);
  EXPECT_EQ(actual.reason, expected.reason);
}

// This is the production path rather than a manually assembled RuntimeChannelSession: compile()
// freezes the POD identity/layout, run_async() reconstructs both typed ports, and the public
// Output<T>::emit(Sample<T>) overload retains the caller-authored metadata while Runtime stamps
// publication-owned sequence and representation fields.
// REQ: CHAN-074
// REQ: CHAN-096
// REQ: RUNTIME-019
TEST(RunSessionTest, CompiledGraphRoundTripsCustomPodWithCallerSampleMetadata) {
  // Key setup: compile() registers the POD identity/layout; run_async() reconstructs
  // both typed ports and stamps sequence/byte_size while preserving caller metadata.
  reset_payload_round_trip_probe();
  const ExecutionPlan plan = make_runtime_pod_round_trip_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool observed = wait_for_pod();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const auto expected = expected_pod_snapshot();
  const auto received = observed_pod_snapshot();
  if (!expected.has_value() || !received.has_value()) {
    ADD_FAILURE() << "custom-POD round trip omitted expected or observed evidence";
    return;
  }
  const auto& expected_sample = *expected;
  const auto& received_sample = *received;
  // Key assertion: field values and caller-authored metadata fields are identical;
  // sequence is > 0 (runtime-stamped) and payload_kind is kInlineTyped.
  EXPECT_EQ(received_sample.pod.first, expected_sample.pod.first);
  EXPECT_EQ(received_sample.pod.second, expected_sample.pod.second);
  expect_caller_sample_metadata_preserved(expected_sample.metadata, received_sample.metadata);
  EXPECT_GT(received_sample.metadata.sequence, 0U);
  EXPECT_EQ(received_sample.metadata.byte_size, sizeof(RuntimeChannelRoundTripPod));
  EXPECT_EQ(received_sample.metadata.payload_kind, PayloadKind::kInlineTyped);
}

// Generated roots use the same concise Output<T>::emit(Sample<T>) call, but the compiled runtime
// routes the owning value through the package encoder/verifier/decoder rather than transporting its
// nontrivial C++ object representation.
// REQ: CHAN-075
// REQ: CHAN-120
// REQ: CHAN-097
// REQ: RUNTIME-019
TEST(RunSessionTest, CompiledGraphRoundTripsGeneratedSchemaWithCallerSampleMetadata) {
  reset_payload_round_trip_probe();
  const ExecutionPlan plan = make_runtime_schema_round_trip_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool observed = wait_for_schema();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const auto expected = expected_schema_snapshot();
  const auto received = observed_schema_snapshot();
  if (!expected.has_value() || !received.has_value()) {
    ADD_FAILURE() << "schema round trip omitted expected or observed evidence";
    return;
  }
  const auto& expected_sample = *expected;
  const auto& received_sample = *received;
  EXPECT_EQ(received_sample.schema_value, expected_sample.schema_value);
  expect_caller_sample_metadata_preserved(expected_sample.metadata, received_sample.metadata);
  EXPECT_GT(received_sample.metadata.sequence, 0U);
  EXPECT_GT(received_sample.metadata.byte_size, received_sample.schema_value.size());
  EXPECT_EQ(received_sample.metadata.payload_kind, PayloadKind::kSerializedBytes);
}

// A direct-retained publication may carry one allocation-free continuation receipt back through
// the completed source callback. The stable partition owner performs the existing destination
// readiness poll inline, then validates and ingests its receipt; disabling only that elision keeps
// delivery direct and provides an otherwise-identical control arm.
// REQ: RUNTIME-011
// REQ: RUNTIME-019
TEST(RunSessionTest, DirectContinuationElidesOnePollForPodAndGeneratedSchema) {
  struct Case {
    RuntimeDirectContinuationPayloadKind payload_kind;
    std::string_view label;
  };
  constexpr std::array cases{
      Case{RuntimeDirectContinuationPayloadKind::kPod, "custom POD"},
      Case{RuntimeDirectContinuationPayloadKind::kGeneratedSchema, "generated schema"},
  };
  const auto count = [](const RuntimeDirectContinuationRoundTripEvidence& evidence,
                        detail::RuntimePhaseCounter counter) {
    return evidence.counters[detail::RuntimePhaseCounters::index(counter)];
  };

  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.label);
    const RuntimeDirectContinuationRoundTripEvidence optimized =
        run_direct_continuation_round_trip(test_case.payload_kind, false);
    const RuntimeDirectContinuationRoundTripEvidence control =
        run_direct_continuation_round_trip(test_case.payload_kind, true);

    ASSERT_TRUE(optimized.direct_delivery);
    ASSERT_TRUE(control.direct_delivery);
    ASSERT_TRUE(optimized.observed);
    ASSERT_TRUE(control.observed);
    EXPECT_EQ(optimized.termination, RunTermination::kStopped);
    EXPECT_EQ(control.termination, RunTermination::kStopped);
    EXPECT_EQ(optimized.diagnostic_count, 0U);
    EXPECT_EQ(control.diagnostic_count, 0U);
    ASSERT_TRUE(optimized.expected.has_value());
    ASSERT_TRUE(optimized.received.has_value());
    ASSERT_TRUE(control.expected.has_value());
    ASSERT_TRUE(control.received.has_value());

    if (test_case.payload_kind == RuntimeDirectContinuationPayloadKind::kPod) {
      EXPECT_EQ(optimized.received->pod.first, optimized.expected->pod.first);
      EXPECT_EQ(optimized.received->pod.second, optimized.expected->pod.second);
      EXPECT_EQ(control.received->pod.first, control.expected->pod.first);
      EXPECT_EQ(control.received->pod.second, control.expected->pod.second);
      EXPECT_EQ(optimized.received->pod.first, control.received->pod.first);
      EXPECT_EQ(optimized.received->pod.second, control.received->pod.second);
      EXPECT_EQ(optimized.received->metadata.byte_size, sizeof(RuntimeChannelRoundTripPod));
      EXPECT_EQ(control.received->metadata.byte_size, sizeof(RuntimeChannelRoundTripPod));
      EXPECT_EQ(optimized.received->metadata.payload_kind, PayloadKind::kInlineTyped);
      EXPECT_EQ(control.received->metadata.payload_kind, PayloadKind::kInlineTyped);
    } else {
      EXPECT_EQ(optimized.received->schema_value, optimized.expected->schema_value);
      EXPECT_EQ(control.received->schema_value, control.expected->schema_value);
      EXPECT_EQ(optimized.received->schema_value, control.received->schema_value);
      EXPECT_GT(optimized.received->metadata.byte_size, optimized.received->schema_value.size());
      EXPECT_GT(control.received->metadata.byte_size, control.received->schema_value.size());
      EXPECT_EQ(optimized.received->metadata.payload_kind, PayloadKind::kSerializedBytes);
      EXPECT_EQ(control.received->metadata.payload_kind, PayloadKind::kSerializedBytes);
    }
    expect_caller_sample_metadata_preserved(optimized.expected->metadata,
                                            optimized.received->metadata);
    expect_caller_sample_metadata_preserved(control.expected->metadata, control.received->metadata);
    EXPECT_EQ(optimized.received->metadata.trace_id, control.received->metadata.trace_id);
    EXPECT_EQ(optimized.received->metadata.frame_id, control.received->metadata.frame_id);
    EXPECT_EQ(optimized.received->metadata.flags, control.received->metadata.flags);
    EXPECT_GT(optimized.received->metadata.sequence, 0U);
    EXPECT_GT(control.received->metadata.sequence, 0U);

    EXPECT_EQ(count(optimized, detail::RuntimePhaseCounter::kDirectContinuationReceiptsAccepted),
              1U);
    EXPECT_EQ(count(optimized, detail::RuntimePhaseCounter::kDirectContinuationPollElisions), 1U);
    EXPECT_EQ(count(control, detail::RuntimePhaseCounter::kDirectContinuationReceiptsAccepted), 0U);
    EXPECT_EQ(count(control, detail::RuntimePhaseCounter::kDirectContinuationPollElisions), 0U);
    EXPECT_EQ(count(optimized, detail::RuntimePhaseCounter::kReadinessPollDispatches), 0U);
    EXPECT_EQ(count(control, detail::RuntimePhaseCounter::kReadinessPollDispatches), 0U);
    EXPECT_EQ(count(optimized, detail::RuntimePhaseCounter::kLaneDispatcherDispatches), 0U);
    EXPECT_EQ(count(control, detail::RuntimePhaseCounter::kLaneDispatcherDispatches), 0U);
    EXPECT_EQ(count(optimized, detail::RuntimePhaseCounter::kLaneDispatches), 0U);
    EXPECT_EQ(count(control, detail::RuntimePhaseCounter::kLaneDispatches), 0U);
    EXPECT_EQ(count(optimized, detail::RuntimePhaseCounter::kLaneComputeDispatches), 0U);
    EXPECT_EQ(count(control, detail::RuntimePhaseCounter::kLaneComputeDispatches), 0U);
    EXPECT_EQ(count(optimized, detail::RuntimePhaseCounter::kComputeDispatches),
              count(control, detail::RuntimePhaseCounter::kComputeDispatches));
    EXPECT_EQ(count(optimized, detail::RuntimePhaseCounter::kActivationsCompleted),
              count(control, detail::RuntimePhaseCounter::kActivationsCompleted));
  }
}

// Runtime reconstructs the compiler-proven closed-island authority before creating activation
// backends or entering user callbacks. Any private-record drift fails closed.
// REQ: RUNTIME-004
// REQ: RUNTIME-011
// REQ: RUNTIME-039
TEST(RunSessionTest, RejectsClosedIslandReplayMismatchBeforeCallbacks) {
  const ExecutionPlan baseline = make_runtime_pod_round_trip_plan();
  const auto baseline_state = detail::ExecutionPlanAccess::retain(baseline);
  ASSERT_NE(baseline_state, nullptr);
  ASSERT_EQ(baseline_state->closed_island_activations.size(), 1U);

  constexpr std::array mutations{
      detail::RuntimeTestPlanAccess::ClosedIslandMutation::kExecutionLane,
      detail::RuntimeTestPlanAccess::ClosedIslandMutation::kReadinessRoot,
      detail::RuntimeTestPlanAccess::ClosedIslandMutation::kReadinessSource,
      detail::RuntimeTestPlanAccess::ClosedIslandMutation::kReadinessClock,
      detail::RuntimeTestPlanAccess::ClosedIslandMutation::kEdge,
  };
  for (const auto mutation : mutations) {
    SCOPED_TRACE(static_cast<unsigned>(mutation));
    reset_payload_round_trip_probe();
    const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
    ExecutionPlan malformed =
        detail::RuntimeTestPlanAccess::corrupt_closed_island_activation_for_test(baseline,
                                                                                 mutation);
    malformed = detail::RuntimeTestPlanAccess::with_phase_counters(malformed, counters);
    ASSERT_TRUE(malformed.ok()) << malformed.json();

    RunSession session = run_async(malformed);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    ASSERT_EQ(session.diagnostics().size(), 1U);
    EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_OPERATOR_REPLAY_MISMATCH");
    EXPECT_EQ(payload_callback_evidence().callback_count, 0U);
    EXPECT_FALSE(expected_pod_snapshot());
    EXPECT_FALSE(observed_pod_snapshot());
    const auto snapshot = counters->snapshot();
    EXPECT_EQ(
        snapshot[detail::RuntimePhaseCounters::index(detail::RuntimePhaseCounter::kLaneDispatches)],
        0U);
  }
}

// The compiler-proven closed island executes the same two public callbacks and exact payload
// transaction as the Checkpoint-1 control. Only the lane-task geometry changes.
// REQ: RUNTIME-011
// REQ: RUNTIME-019
// REQ: RUNTIME-039
TEST(RunSessionTest, ClosedIslandCompletionAcceptsFreshLogicalSuccessorCoordinate) {
  detail::RuntimeDirectCompositeCompletionReport report;
  report.callback_identities[0] = detail::RuntimeCallbackIdentity{
      .context = detail::RuntimeCallbackContext::kCompute,
      .operator_id = 1U,
      .tag = Tag{.sequence = 1U, .generation = 1U},
      .tag_time = TagTime{.logical_ns = 100, .microstep = 7U},
  };
  report.callback_identities[1] = detail::RuntimeCallbackIdentity{
      .context = detail::RuntimeCallbackContext::kCompute,
      .operator_id = 2U,
      .tag = Tag{.sequence = 1U, .generation = 1U},
      .tag_time = TagTime{.logical_ns = 101, .microstep = 0U},
  };
  report.phase_deltas = {2U, 2U, 2U, 2U, 1U};
  report.continuation = detail::RuntimeDirectCompositeContinuationDisposition::kAccepted;
  report.resume = detail::RuntimeDirectCompositeResumeDisposition::kCompositeCompleted;
  report.callbacks_attempted = 2U;
  report.callbacks_completed = 2U;
  report.source_retired = true;
  report.successor_retired = true;
  report.activation_cut_completed = true;

  EXPECT_TRUE(report.complete());
  report.callback_identities[1].tag_time.microstep = 1U;
  EXPECT_FALSE(report.complete());
}

// REQ: RUNTIME-039
TEST(RunSessionTest, ClosedIslandCompositeMatchesOrdinaryPodAndGeneratedSchema) {
  struct Case {
    RuntimeDirectContinuationPayloadKind payload_kind;
    std::string_view label;
  };
  constexpr std::array cases{
      Case{RuntimeDirectContinuationPayloadKind::kPod, "custom POD"},
      Case{RuntimeDirectContinuationPayloadKind::kGeneratedSchema, "generated schema"},
  };
  const auto count = [](const RuntimeDirectContinuationRoundTripEvidence& evidence,
                        detail::RuntimePhaseCounter counter) {
    return evidence.counters[detail::RuntimePhaseCounters::index(counter)];
  };

  for (const Case& test_case : cases) {
    SCOPED_TRACE(test_case.label);
    RuntimeClosedIslandCompletionProbe completion_probe;
    const auto completion_control = std::make_shared<detail::RuntimeClosedIslandTestControl>();
    completion_control->set_completion_hook(&completion_probe,
                                            &RuntimeClosedIslandCompletionProbe::capture);
    const RuntimeDirectContinuationRoundTripEvidence candidate = run_direct_continuation_round_trip(
        test_case.payload_kind, false, false, false, completion_control);
    const RuntimeDirectContinuationRoundTripEvidence control = run_direct_continuation_round_trip(
        test_case.payload_kind, false, false, /*disable_closed_island=*/true);

    ASSERT_TRUE(candidate.direct_delivery);
    ASSERT_TRUE(control.direct_delivery);
    ASSERT_TRUE(candidate.observed);
    ASSERT_TRUE(control.observed);
    EXPECT_EQ(candidate.termination, RunTermination::kStopped);
    EXPECT_EQ(control.termination, RunTermination::kStopped);
    EXPECT_EQ(candidate.diagnostic_count, 0U);
    EXPECT_EQ(control.diagnostic_count, 0U);
    ASSERT_TRUE(candidate.expected);
    ASSERT_TRUE(candidate.received);
    ASSERT_TRUE(control.expected);
    ASSERT_TRUE(control.received);
    EXPECT_EQ(candidate.received->metadata.trace_id, control.received->metadata.trace_id);
    EXPECT_EQ(candidate.received->metadata.frame_id, control.received->metadata.frame_id);
    EXPECT_EQ(candidate.received->metadata.flags, control.received->metadata.flags);
    if (test_case.payload_kind == RuntimeDirectContinuationPayloadKind::kPod) {
      EXPECT_EQ(candidate.received->pod.first, control.received->pod.first);
      EXPECT_EQ(candidate.received->pod.second, control.received->pod.second);
    } else {
      EXPECT_EQ(candidate.received->schema_value, control.received->schema_value);
    }

    EXPECT_EQ(candidate.callbacks.order, (std::array<std::uint8_t, 2U>{1U, 2U}));
    EXPECT_EQ(candidate.callbacks.callback_count, 2U);
    EXPECT_EQ(candidate.callbacks.active_callbacks, 0U);
    EXPECT_EQ(candidate.callbacks.max_active_callbacks, 1U);
    EXPECT_FALSE(candidate.callbacks.overlap);
    EXPECT_TRUE(candidate.callbacks.same_thread);
    EXPECT_EQ(candidate.callbacks.identities[0].context, detail::RuntimeCallbackContext::kCompute);
    EXPECT_EQ(candidate.callbacks.identities[1].context, detail::RuntimeCallbackContext::kCompute);
    EXPECT_NE(candidate.callbacks.identities[0].operator_id, 0U);
    EXPECT_NE(candidate.callbacks.identities[1].operator_id, 0U);
    EXPECT_NE(candidate.callbacks.identities[0].operator_id,
              candidate.callbacks.identities[1].operator_id);
    EXPECT_EQ(candidate.callbacks.identities[0].tag_time.logical_ns,
              candidate.callbacks.identities[1].tag_time.logical_ns);
    ASSERT_NE(candidate.callbacks.identities[0].tag_time.microstep,
              std::numeric_limits<std::uint32_t>::max());
    EXPECT_EQ(candidate.callbacks.identities[1].tag_time.microstep,
              candidate.callbacks.identities[0].tag_time.microstep + 1U);
    const auto completion = completion_probe.read();
    ASSERT_TRUE(completion.has_value());
    EXPECT_EQ(completion->identities[0].operator_id, candidate.callbacks.identities[0].operator_id);
    EXPECT_EQ(completion->identities[0].tag_time, candidate.callbacks.identities[0].tag_time);
    EXPECT_EQ(completion->identities[1].operator_id, candidate.callbacks.identities[1].operator_id);
    EXPECT_EQ(completion->identities[1].tag_time, candidate.callbacks.identities[1].tag_time);
    EXPECT_EQ(completion->errors, (std::array{ErrorCode::kSuccess, ErrorCode::kSuccess}));
    EXPECT_EQ(completion->first_error, ErrorCode::kSuccess);
    EXPECT_EQ(completion->callbacks_attempted, 2U);
    EXPECT_EQ(completion->callbacks_completed, 2U);
    EXPECT_EQ(completion->phase_deltas,
              (std::array<std::uint64_t, detail::kRuntimeDirectCompositePhaseDeltaCount>{
                  2U, 2U, 2U, 2U, 1U}));
    EXPECT_EQ(control.callbacks.order, candidate.callbacks.order);
    EXPECT_EQ(control.callbacks.callback_count, 2U);
    EXPECT_EQ(control.callbacks.max_active_callbacks, 1U);
    EXPECT_FALSE(control.callbacks.overlap);

    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kDirectCompositeAttempts), 1U);
    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kDirectCompositeCompletions), 1U);
    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kDirectCompositeFallbacks), 0U);
    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kDirectCompositeFailures), 0U);
    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kDirectCompositeCallbacksAttempted),
              2U);
    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kDirectCompositeCallbacksCompleted),
              2U);
    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kLaneDispatcherDispatches), 0U);
    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kLaneComputeDispatches), 0U);
    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kLaneDispatches), 0U);
    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kLaneInlineOwnerInvocations), 1U);
    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kLaneCallbacksEntered), 1U);
    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kLaneCallbacksExited), 1U);
    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kComputeDispatches), 2U);
    EXPECT_EQ(count(candidate, detail::RuntimePhaseCounter::kActivationsCompleted), 2U);

    EXPECT_EQ(count(control, detail::RuntimePhaseCounter::kDirectCompositeCompletions), 0U);
    EXPECT_EQ(count(control, detail::RuntimePhaseCounter::kLaneComputeDispatches), 0U);
    EXPECT_EQ(count(control, detail::RuntimePhaseCounter::kLaneDispatches), 0U);
    EXPECT_EQ(count(control, detail::RuntimePhaseCounter::kLaneInlineOwnerInvocations), 2U);
    EXPECT_EQ(count(control, detail::RuntimePhaseCounter::kComputeDispatches), 2U);
    EXPECT_EQ(count(control, detail::RuntimePhaseCounter::kActivationsCompleted), 2U);
  }
}

// Physical Spin and Readable roots both derive the same compiler-frozen island. The source-tree
// finite binding then supplies one Spin witness to prove the one-lane Runtime transaction.
// REQ: RUNTIME-011
// REQ: RUNTIME-039
TEST(RunSessionTest, ClosedIslandEligibilityAcceptsPhysicalSpinAndReadableRoots) {
  for (const auto root :
       {RuntimePayloadRoundTripRoot::kSpin, RuntimePayloadRoundTripRoot::kReadable}) {
    const ExecutionPlan compiled_plan = make_runtime_pod_round_trip_plan(false, root);
    ASSERT_TRUE(compiled_plan.ok()) << compiled_plan.json();
    const auto compiled = detail::ExecutionPlanAccess::retain(compiled_plan);
    ASSERT_NE(compiled, nullptr);
    ASSERT_EQ(compiled->closed_island_activations.size(), 1U) << compiled_plan.json();
    EXPECT_EQ(compiled->closed_island_activations.front().readiness_root,
              detail::ClosedIslandReadinessRootKind::kPhysical);
  }

  reset_payload_round_trip_probe();
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  ExecutionPlan plan = make_runtime_pod_round_trip_plan(false, RuntimePayloadRoundTripRoot::kSpin);
  plan = detail::RuntimeTestPlanAccess::enable_direct_once(plan);
  plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool observed = wait_for_pod();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  const auto snapshot = counters->snapshot();
  const auto count = [&](detail::RuntimePhaseCounter counter) {
    return snapshot[detail::RuntimePhaseCounters::index(counter)];
  };
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectCompositeAttempts), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectCompositeCompletions), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectCompositeCallbacksAttempted), 2U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kPartitionCausalFrontierAttempts), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kPartitionCausalFrontierAccepted), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kPartitionCausalFrontierFallbacks), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneComputeDispatches), 0U);
}

// A lifecycle-acquired NotificationSender exercises the distinct startup materialization and
// replay branch, then proves the notified root retains the same one-task/two-callback geometry.
// REQ: RUNTIME-011
// REQ: RUNTIME-039
TEST(RunSessionTest, ClosedIslandEligibilityExecutesNotificationRoot) {
  test::RuntimeNotificationProbeState::reset();
  reset_payload_round_trip_probe();
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  ExecutionPlan plan =
      make_runtime_pod_round_trip_plan(false, RuntimePayloadRoundTripRoot::kNotification);
  const auto compiled = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(compiled, nullptr);
  ASSERT_EQ(compiled->closed_island_activations.size(), 1U) << plan.json();
  EXPECT_EQ(compiled->closed_island_activations.front().readiness_root,
            detail::ClosedIslandReadinessRootKind::kNotification);
  plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  ASSERT_TRUE(test::RuntimeNotificationProbeState::wait_until_sender());
  const NotificationSender sender = test::RuntimeNotificationProbeState::sender_copy();
  auto posted = sender.post();
  const auto start_deadline = std::chrono::steady_clock::now() + 5s;
  while (!posted && posted.error().code == ErrorCode::kNotReady &&
         std::chrono::steady_clock::now() < start_deadline) {
    std::this_thread::yield();
    posted = sender.post();
  }
  ASSERT_TRUE(posted);
  const bool observed = wait_for_pod();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  const RuntimePayloadRoundTripCallbackEvidence callbacks = payload_callback_evidence();
  EXPECT_EQ(callbacks.callback_count, 2U);
  EXPECT_TRUE(callbacks.same_thread);
  const auto snapshot = counters->snapshot();
  const auto count = [&](detail::RuntimePhaseCounter counter) {
    return snapshot[detail::RuntimePhaseCounters::index(counter)];
  };
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectCompositeAttempts), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectCompositeCompletions), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectCompositeCallbacksAttempted), 2U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kPartitionCausalFrontierAttempts), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kPartitionCausalFrontierAccepted), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kPartitionCausalFrontierFallbacks), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneComputeDispatches), 0U);
}

// Tensor transport, ambiguous incident delivery, side-input/advanced temporal policy, mutable
// Parameters, lifecycle transactions, and multi-emission sources remain on the ordinary path.
// Each authored plan proves that the compiler omits rather than partially admits a composite when
// one bounded eligibility fact is absent.
// REQ: RUNTIME-011
// REQ: RUNTIME-039
TEST(RunSessionTest, ClosedIslandEligibilityRejectsUnsupportedStatefulShapes) {
  const auto expect_omitted = [](const ExecutionPlan& plan, std::string_view label) {
    SCOPED_TRACE(label);
    ASSERT_TRUE(plan.ok()) << plan.json();
    const auto state = detail::ExecutionPlanAccess::retain(plan);
    ASSERT_NE(state, nullptr);
    EXPECT_TRUE(state->closed_island_activations.empty());
  };

  expect_omitted(make_runtime_closed_island_tensor_ineligible_plan(), "Tensor payload");
  expect_omitted(make_runtime_pod_nonretained_sibling_plan(), "non-retained sibling incident edge");
  expect_omitted(make_runtime_pod_external_export_ambiguity_plan(),
                 "mixed local and external publication");
  expect_omitted(make_runtime_pod_side_input_plan(), "independently clocked side input");
  expect_omitted(make_runtime_pod_predicate_plan(), "predicate trigger");
  expect_omitted(make_runtime_pod_deadline_plan(), "deadline policy");
  expect_omitted(make_runtime_pod_mutable_param_plan(), "runtime-mutable Parameters");
  expect_omitted(make_runtime_pod_round_trip_plan(true), "lifecycle transaction");
  expect_omitted(make_runtime_pod_double_emit_plan(), "multi-emission source");
}

// A closed-island composite is owned by one partition runner. Distinct explicit
// partitions may share a physical lane, but their edge must remain on the
// ordinary cross-partition path instead of creating cross-runner authority.
// REQ: RUNTIME-039
// REQ: RUNTIME-041
TEST(RunSessionTest, ClosedIslandEligibilityDoesNotCrossExplicitPartitionBoundary) {
  reset_payload_round_trip_probe();
  const ExecutionPlan plan = make_runtime_pod_cross_partition_round_trip_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_TRUE(state->closed_island_activations.empty());

  RunSession session = run_async(plan);
  const bool observed = wait_for_pod();
  session.request_stop();
  session.wait();

  EXPECT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
}

// The first complete transaction warms every run-owned cache. A test-only allocation counter then
// spans the second source/successor transaction, including the gap between callbacks, without
// instrumenting production plans.
// REQ: RUNTIME-011
// REQ: RUNTIME-039
TEST(RunSessionTest, ClosedIslandCompositeWarmPodTransactionAllocatesNothing) {
  reset_payload_round_trip_probe(
      RuntimePayloadRoundTripSourceBehavior::kEmitTwiceAcrossActivations);
  RuntimeClosedIslandAllocationProbe allocation_probe;
  const auto control = std::make_shared<detail::RuntimeClosedIslandTestControl>();
  control->set_transaction_hooks(&allocation_probe,
                                 &RuntimeClosedIslandAllocationProbe::begin,
                                 &RuntimeClosedIslandAllocationProbe::end);

  ExecutionPlan plan = make_runtime_pod_round_trip_plan();
  plan = detail::RuntimeTestPlanAccess::with_closed_island_test_control(plan, control);
  ASSERT_TRUE(plan.ok()) << plan.json();
  RunSession session = run_async(plan);
  const bool observed = wait_for_pod_count(2U);
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  EXPECT_GE(allocation_probe.transaction_count.load(std::memory_order_acquire), 2U);
  EXPECT_TRUE(allocation_probe.observed.load(std::memory_order_acquire));
  EXPECT_EQ(allocation_probe.allocations.load(std::memory_order_acquire), 0U);
}

// No publication, a typed Error, and an exception must never synthesize a successor callback.
// Typed and exceptional failures retain the ordinary terminal diagnostic.
// REQ: RUNTIME-011
// REQ: RUNTIME-024
// REQ: RUNTIME-039
TEST(RunSessionTest, ClosedIslandCompositePreservesNoEmissionAndSourceFailureSemantics) {
  struct Case {
    RuntimePayloadRoundTripSourceBehavior behavior;
    RunTermination termination;
    bool expect_diagnostic;
    ErrorCode expected_error;
  };
  constexpr std::array cases{
      Case{RuntimePayloadRoundTripSourceBehavior::kNoEmission,
           RunTermination::kStopped,
           false,
           ErrorCode::kSuccess},
      Case{RuntimePayloadRoundTripSourceBehavior::kBackpressured,
           RunTermination::kFailed,
           true,
           ErrorCode::kBackpressured},
      Case{RuntimePayloadRoundTripSourceBehavior::kTypedFailure,
           RunTermination::kFailed,
           true,
           ErrorCode::kFailure},
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      Case{RuntimePayloadRoundTripSourceBehavior::kException,
           RunTermination::kFailed,
           true,
           ErrorCode::kFailure},
#endif
  };

  for (const Case& test_case : cases) {
    SCOPED_TRACE(static_cast<unsigned>(test_case.behavior));
    reset_payload_round_trip_probe(test_case.behavior);
    const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
    RuntimeClosedIslandCompletionProbe completion_probe;
    const auto completion_control = std::make_shared<detail::RuntimeClosedIslandTestControl>();
    completion_control->set_completion_hook(&completion_probe,
                                            &RuntimeClosedIslandCompletionProbe::capture);
    ExecutionPlan plan = make_runtime_pod_round_trip_plan();
    plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
    plan = detail::RuntimeTestPlanAccess::with_closed_island_test_control(plan, completion_control);
    ASSERT_TRUE(plan.ok()) << plan.json();

    RunSession session = run_async(plan);
    ASSERT_TRUE(wait_for_payload_source());
    if (test_case.termination == RunTermination::kStopped) {
      session.request_stop();
    }
    session.wait();

    EXPECT_EQ(session.termination(), test_case.termination);
    EXPECT_FALSE(observed_pod_snapshot());
    const auto callbacks = payload_callback_evidence();
    EXPECT_EQ(callbacks.callback_count, 1U);
    EXPECT_EQ(callbacks.max_active_callbacks, 1U);
    EXPECT_FALSE(callbacks.overlap);
    if (test_case.expect_diagnostic) {
      ASSERT_EQ(session.diagnostics().size(), 1U);
      EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
      EXPECT_EQ(session.diagnostics().front().service, "operator_compute");
    } else {
      EXPECT_TRUE(session.diagnostics().empty());
    }
    const auto snapshot = counters->snapshot();
    const auto count = [&](detail::RuntimePhaseCounter counter) {
      return snapshot[detail::RuntimePhaseCounters::index(counter)];
    };
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectCompositeCompletions), 0U);
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectCompositeCallbacksAttempted), 1U);
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectCompositeCallbacksCompleted),
              test_case.expect_diagnostic ? 0U : 1U);
    const auto completion = completion_probe.read();
    ASSERT_TRUE(completion.has_value());
    EXPECT_EQ(completion->callbacks_attempted, 1U);
    EXPECT_EQ(completion->callbacks_completed, test_case.expect_diagnostic ? 0U : 1U);
    EXPECT_EQ(completion->errors[0], test_case.expected_error);
    EXPECT_EQ(completion->first_error, test_case.expected_error);
    EXPECT_EQ(completion->phase_deltas[static_cast<std::size_t>(
                  detail::RuntimeDirectCompositePhaseDelta::kCallbacksAttempted)],
              1U);
    EXPECT_EQ(completion->phase_deltas[static_cast<std::size_t>(
                  detail::RuntimeDirectCompositePhaseDelta::kCallbacksCompleted)],
              test_case.expect_diagnostic ? 0U : 1U);
    EXPECT_EQ(completion->phase_deltas[static_cast<std::size_t>(
                  detail::RuntimeDirectCompositePhaseDelta::kActivationsRetired)],
              1U);
    EXPECT_EQ(completion->phase_deltas[static_cast<std::size_t>(
                  detail::RuntimeDirectCompositePhaseDelta::kActivationCuts)],
              test_case.expect_diagnostic ? 0U : 1U);
  }
}

// A successor typed Error or exception retires the same selected payload, activation, and source
// credit as ordinary dispatch while preserving the successor's callback identity and diagnostic.
// REQ: RUNTIME-011
// REQ: RUNTIME-024
// REQ: RUNTIME-039
TEST(RunSessionTest, ClosedIslandCompositePreservesSuccessorFailureSemantics) {
  constexpr std::array sink_behaviors{
      RuntimePayloadRoundTripSinkBehavior::kTypedFailure,
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      RuntimePayloadRoundTripSinkBehavior::kException,
#endif
  };

  for (const RuntimePayloadRoundTripSinkBehavior sink_behavior : sink_behaviors) {
    SCOPED_TRACE(static_cast<unsigned>(sink_behavior));
    const auto candidate = run_direct_continuation_round_trip(
        RuntimeDirectContinuationPayloadKind::kPod, false, false, false, {}, sink_behavior);
    const auto control = run_direct_continuation_round_trip(
        RuntimeDirectContinuationPayloadKind::kPod, false, false, true, {}, sink_behavior);
    for (const auto* evidence : {&candidate, &control}) {
      EXPECT_TRUE(evidence->observed);
      EXPECT_EQ(evidence->termination, RunTermination::kFailed);
      EXPECT_EQ(evidence->diagnostic_count, 1U);
      EXPECT_EQ(evidence->callbacks.callback_count, 2U);
      EXPECT_EQ(evidence->callbacks.max_active_callbacks, 1U);
      EXPECT_FALSE(evidence->callbacks.overlap);
    }
    const auto candidate_count = [&](detail::RuntimePhaseCounter counter) {
      return candidate.counters[detail::RuntimePhaseCounters::index(counter)];
    };
    EXPECT_EQ(candidate_count(detail::RuntimePhaseCounter::kDirectCompositeAttempts), 1U);
    EXPECT_EQ(candidate_count(detail::RuntimePhaseCounter::kDirectCompositeCompletions), 0U);
    EXPECT_EQ(candidate_count(detail::RuntimePhaseCounter::kDirectCompositeCallbacksAttempted), 2U);
    EXPECT_EQ(candidate_count(detail::RuntimePhaseCounter::kDirectCompositeCallbacksCompleted), 1U);
  }
}

// A source that may publish twice is conservatively omitted from the compiler-proven island.
// Ordinary dispatch still preserves finite backpressure: handling the second result preserves the
// first publication, while returning it remains a callback failure and suppresses the successor.
// REQ: CHAN-101
// REQ: RUNTIME-024
// REQ: RUNTIME-039
TEST(RunSessionTest, ConservativeClosedIslandEligibilityFallsBackForDoubleEmission) {
  for (const auto behavior : {RuntimePayloadRoundTripSourceBehavior::kSecondEmitHandled,
                              RuntimePayloadRoundTripSourceBehavior::kSecondEmitReturned}) {
    SCOPED_TRACE(static_cast<unsigned>(behavior));
    reset_payload_round_trip_probe(behavior);
    const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
    ExecutionPlan plan = make_runtime_pod_double_emit_plan();
    plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
    ASSERT_TRUE(plan.ok()) << plan.json();

    RunSession session = run_async(plan);
    if (behavior == RuntimePayloadRoundTripSourceBehavior::kSecondEmitHandled) {
      EXPECT_TRUE(wait_for_pod());
      session.request_stop();
    }
    session.wait();

    const auto second_emit = second_emit_snapshot();
    ASSERT_TRUE(second_emit);
    // The fatal assertion above proves the optional is engaged.
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_EQ(*second_emit, ErrorCode::kBackpressured);
    const auto snapshot = counters->snapshot();
    const auto count = [&](detail::RuntimePhaseCounter counter) {
      return snapshot[detail::RuntimePhaseCounters::index(counter)];
    };
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelEmitAttempts), 2U);
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelEmitAccepted), 1U);
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelEmitBackpressured), 1U);
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectCompositeAttempts), 0U);
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectCompositeCompletions), 0U);
    if (behavior == RuntimePayloadRoundTripSourceBehavior::kSecondEmitHandled) {
      EXPECT_EQ(session.termination(), RunTermination::kStopped);
      EXPECT_TRUE(observed_pod_snapshot());
      EXPECT_TRUE(session.diagnostics().empty());
    } else {
      EXPECT_EQ(session.termination(), RunTermination::kFailed);
      EXPECT_FALSE(observed_pod_snapshot());
      ASSERT_EQ(session.diagnostics().size(), 1U);
      EXPECT_EQ(session.diagnostics().front().token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
    }
  }
}

// Stop and generation checks occur at the callback boundary inside one lane task. Stop wins
// without entering the successor; stale evidence falls back to the ordinary readiness scan and
// still delivers exactly once.
// REQ: RUNTIME-011
// REQ: RUNTIME-024
// REQ: RUNTIME-039
TEST(RunSessionTest, ClosedIslandCompositeHonorsStopAndGenerationMismatchAtSuccessorBoundary) {
  {
    reset_payload_round_trip_probe();
    const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
    const auto control = std::make_shared<detail::RuntimeClosedIslandTestControl>();
    control->set_pause_before_successor(true);
    ExecutionPlan plan = make_runtime_pod_round_trip_plan();
    plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
    plan = detail::RuntimeTestPlanAccess::with_closed_island_test_control(plan, control);

    RunSession session = run_async(plan);
    const bool reached_successor_boundary = control->wait_until_before_successor();
    session.request_stop();
    control->release();
    session.wait();

    ASSERT_TRUE(reached_successor_boundary);
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(expected_pod_snapshot());
    EXPECT_FALSE(observed_pod_snapshot());
    EXPECT_EQ(payload_callback_evidence().callback_count, 1U);
    const auto snapshot = counters->snapshot();
    EXPECT_EQ(snapshot[detail::RuntimePhaseCounters::index(
                  detail::RuntimePhaseCounter::kDirectCompositeCompletions)],
              0U);
  }
  {
    reset_payload_round_trip_probe();
    const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
    RuntimeClosedIslandCompletionProbe completion_probe;
    const auto control = std::make_shared<detail::RuntimeClosedIslandTestControl>();
    control->set_pause_at_successor_boundary(true);
    control->set_completion_hook(&completion_probe, &RuntimeClosedIslandCompletionProbe::capture);
    ExecutionPlan plan = make_runtime_pod_round_trip_plan();
    plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
    plan = detail::RuntimeTestPlanAccess::with_closed_island_test_control(plan, control);

    RunSession session = run_async(plan);
    const bool reached_successor_boundary = control->wait_until_successor_boundary();
    session.request_stop();
    control->release();
    session.wait();

    ASSERT_TRUE(reached_successor_boundary);
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(expected_pod_snapshot());
    EXPECT_FALSE(observed_pod_snapshot());
    EXPECT_EQ(payload_callback_evidence().callback_count, 1U);
    const auto snapshot = counters->snapshot();
    EXPECT_EQ(snapshot[detail::RuntimePhaseCounters::index(
                  detail::RuntimePhaseCounter::kDirectCompositeCompletions)],
              0U);
    const auto completion = completion_probe.read();
    ASSERT_TRUE(completion.has_value());
    // The fatal assertion above proves the optional is engaged.
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    EXPECT_EQ(completion->phase_deltas[static_cast<std::size_t>(
                  detail::RuntimeDirectCompositePhaseDelta::kActivationsRetired)],
              2U);
  }
  {
    const auto control = std::make_shared<detail::RuntimeClosedIslandTestControl>();
    control->set_corrupt_generation(true);
    const RuntimeDirectContinuationRoundTripEvidence evidence = run_direct_continuation_round_trip(
        RuntimeDirectContinuationPayloadKind::kPod, false, false, false, control);
    const auto count = [&](detail::RuntimePhaseCounter counter) {
      return evidence.counters[detail::RuntimePhaseCounters::index(counter)];
    };

    ASSERT_TRUE(evidence.observed);
    EXPECT_EQ(evidence.termination, RunTermination::kStopped);
    EXPECT_EQ(evidence.diagnostic_count, 0U);
    EXPECT_EQ(evidence.callbacks.callback_count, 2U);
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectCompositeCompletions), 0U);
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectCompositeFallbacks), 1U);
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneDispatcherDispatches), 0U);
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneComputeDispatches), 0U);
    EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneDispatches), 0U);
    EXPECT_GE(count(detail::RuntimePhaseCounter::kLaneInlineOwnerInvocations), 1U);
  }
}

// A callback-produced receipt is only accepted when the next readiness scan consumes it and
// removes an ordinary poll. Stopping after the lane stages the receipt but before that scan must
// retire the session without reporting an elision that never happened.
// REQ: RUNTIME-011
// REQ: RUNTIME-019
TEST(RunSessionTest, StopAfterDirectContinuationStagingRecordsNoUnelidedAcceptance) {
  reset_payload_round_trip_probe();
  ExecutionPlan plan = make_runtime_pod_round_trip_plan();
  const auto plan_state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(plan_state, nullptr);
  ASSERT_EQ(plan_state->process_local_channel_services.size(), 1U);
  ASSERT_EQ(plan_state->process_local_channel_services.front().delivery_kind,
            detail::ProcessLocalChannelDeliveryKind::kDirectRetained);

  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const auto control = std::make_shared<detail::RuntimeDirectContinuationTestControl>();
  plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
  plan = detail::RuntimeTestPlanAccess::disable_closed_island_composite_for_test(plan);
  plan = detail::RuntimeTestPlanAccess::with_direct_continuation_test_control(plan, control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool receipt_staged = control->wait_until_receipt_staged();
  session.request_stop();
  control->release();
  session.wait();

  ASSERT_TRUE(receipt_staged);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_TRUE(expected_pod_snapshot().has_value());
  EXPECT_FALSE(observed_pod_snapshot().has_value());

  const auto snapshot = counters->snapshot();
  const auto count = [&](const detail::RuntimePhaseCounter counter) {
    return snapshot[detail::RuntimePhaseCounters::index(counter)];
  };
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelEmitAccepted), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelSamplesReceived), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelReadinessPublished), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kActivationsCompleted), 1U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectContinuationReceiptsAccepted), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kDirectContinuationPollElisions), 0U);
}

// Restart-capable plans remain on the ordinary controller-owned path until Checkpoint 4C proves
// the complete lifecycle transaction. This conservative arm also demonstrates that merely
// compiling the private optimization cannot alter successful direct-retained delivery.
// REQ: RUNTIME-011
// REQ: RUNTIME-019
TEST(RunSessionTest, DirectContinuationRemainsDisabledForLifecyclePlans) {
  constexpr std::array payload_kinds{
      RuntimeDirectContinuationPayloadKind::kPod,
      RuntimeDirectContinuationPayloadKind::kGeneratedSchema,
  };
  const auto count = [](const RuntimeDirectContinuationRoundTripEvidence& evidence,
                        detail::RuntimePhaseCounter counter) {
    return evidence.counters[detail::RuntimePhaseCounters::index(counter)];
  };

  for (const auto payload_kind : payload_kinds) {
    const RuntimeDirectContinuationRoundTripEvidence evidence =
        run_direct_continuation_round_trip(payload_kind, false, true);
    ASSERT_TRUE(evidence.direct_delivery);
    ASSERT_TRUE(evidence.observed);
    EXPECT_EQ(evidence.termination, RunTermination::kStopped);
    EXPECT_EQ(evidence.diagnostic_count, 0U);
    ASSERT_TRUE(evidence.expected.has_value());
    ASSERT_TRUE(evidence.received.has_value());
    // The fatal assertions above prove both optionals are engaged.
    // NOLINTBEGIN(bugprone-unchecked-optional-access)
    expect_caller_sample_metadata_preserved(evidence.expected->metadata,
                                            evidence.received->metadata);
    // NOLINTEND(bugprone-unchecked-optional-access)
    EXPECT_EQ(count(evidence, detail::RuntimePhaseCounter::kDirectContinuationReceiptsAccepted),
              0U);
    EXPECT_EQ(count(evidence, detail::RuntimePhaseCounter::kDirectContinuationPollElisions), 0U);
    EXPECT_EQ(count(evidence, detail::RuntimePhaseCounter::kDirectCompositeAttempts), 0U);
    EXPECT_EQ(count(evidence, detail::RuntimePhaseCounter::kDirectCompositeCompletions), 0U);
    EXPECT_EQ(count(evidence, detail::RuntimePhaseCounter::kReadinessPollDispatches), 0U);
    EXPECT_EQ(count(evidence, detail::RuntimePhaseCounter::kLaneDispatches), 0U);
  }
}

// The public compile/run path must not let a capacity-one blocking queue monopolize the serial
// execution lane. The callback observes finite pressure, remains live under the cooperative
// cancellation contract, and then lets request_stop() complete the ordinary RunSession teardown.
// REQ: CHAN-101
// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(RunSessionTest, StopsWhileGraphOwnedBlockingQueueIsSaturated) {
  reset_blocking_pressure_probe();
  const ExecutionPlan plan = make_runtime_blocking_pressure_plan("runtime-blocking-pressure-stop");
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_FALSE(state->process_local_channel_services.empty());
  EXPECT_TRUE(std::ranges::all_of(state->process_local_channel_services, [](const auto& service) {
    return service.delivery_kind == detail::ProcessLocalChannelDeliveryKind::kHoloMQ;
  }));

  RunSession session = run_async(plan);
  const bool pressure_observed = wait_for_blocking_pressure();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(pressure_observed);
  const auto [result, accepted_count] = blocking_pressure_snapshot();
  EXPECT_EQ(result, ErrorCode::kBackpressured);
  EXPECT_GT(accepted_count, 0U);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_FALSE(blocking_pressure_callback_active());
}

// Returning a fatal Error after finite graph-owned pressure must preserve ordinary structural
// failure reporting and teardown. In particular, pressure handling cannot strand the callback or
// convert a user callback failure into a blocked stop path.
// REQ: CHAN-101
// REQ: RUNTIME-020
// REQ: RUNTIME-024
TEST(RunSessionTest, FailsStructurallyAfterGraphOwnedBlockingQueuePressure) {
  reset_blocking_pressure_probe(/*fail_after_pressure=*/true);
  const ExecutionPlan plan =
      make_runtime_blocking_pressure_plan("runtime-blocking-pressure-failure");
  ASSERT_TRUE(plan.ok()) << plan.json();

  const auto wait_started = std::chrono::steady_clock::now();
  RunSession session = run_async(plan);
  session.wait();
  const auto wait_duration = std::chrono::steady_clock::now() - wait_started;

  const auto [result, accepted_count] = blocking_pressure_snapshot();
  EXPECT_EQ(result, ErrorCode::kBackpressured);
  EXPECT_GT(accepted_count, 0U);
  EXPECT_LT(wait_duration, 5s);
  ASSERT_EQ(session.termination(), RunTermination::kFailed);
  ASSERT_EQ(session.diagnostics().size(), 1U);
  const Diagnostic& diagnostic = session.diagnostics().front();
  EXPECT_EQ(diagnostic.token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
  EXPECT_EQ(diagnostic.service, "operator_compute");
  EXPECT_TRUE(static_cast<bool>(diagnostic.source.vertex));
  EXPECT_TRUE(diagnostic.endpoint.empty());
  EXPECT_TRUE(diagnostic.json_pointer.empty());
  EXPECT_FALSE(blocking_pressure_callback_active());
}

// REQ: RUNTIME-004
// REQ: RUNTIME-015
// REQ: RUNTIME-018
// REQ: RUNTIME-034
TEST(RunSessionTest, ChannelAlignedTriggerAcceptsTheInclusiveExactBoundary) {
  test::RuntimeAlignedChannelProbeState::reset(test::RuntimeAlignedChannelScript::kExactBoundary);
  const ExecutionPlan plan = test::make_runtime_aligned_channel_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool observed = test::RuntimeAlignedChannelProbeState::wait_for_sink(1U);
  session.request_stop();
  session.wait();

  EXPECT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token) << " vertex="
      << (session.diagnostics().empty() ? 0U : session.diagnostics().front().source.vertex.value);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::source_compute_count.load(), 1U);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::sink_compute_count.load(), 1U);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_first.load(), 1U);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_second.load(), 2U);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_second_capture.load() -
                test::RuntimeAlignedChannelProbeState::observed_first_capture.load(),
            10);
}

// REQ: RUNTIME-004
// REQ: RUNTIME-018
TEST(RunSessionTest, ChannelAlignedTriggerRetiresEveryMinimumAndMakesProgress) {
  test::RuntimeAlignedChannelProbeState::reset(
      test::RuntimeAlignedChannelScript::kMisalignmentProgress);
  const ExecutionPlan plan = test::make_runtime_aligned_channel_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool observed = test::RuntimeAlignedChannelProbeState::wait_for_sink(1U);
  session.request_stop();
  session.wait();

  EXPECT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token) << " vertex="
      << (session.diagnostics().empty() ? 0U : session.diagnostics().front().source.vertex.value);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::sink_compute_count.load(), 1U);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_first.load(), 3U);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_second.load(), 2U);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_second_capture.load() -
                test::RuntimeAlignedChannelProbeState::observed_first_capture.load(),
            10);
}

// REQ: RUNTIME-004
// REQ: RUNTIME-018
TEST(RunSessionTest, ChannelAlignedTriggerKeepsMissingHeadPendingAndFreshRunResetsIt) {
  test::RuntimeAlignedChannelProbeState::reset(test::RuntimeAlignedChannelScript::kMissingHead);
  const ExecutionPlan plan = test::make_runtime_aligned_channel_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession pending = run_async(plan);
  const bool source_progressed = test::RuntimeAlignedChannelProbeState::wait_for_source(1U);
  const std::size_t pending_compute_count =
      test::RuntimeAlignedChannelProbeState::sink_compute_count.load(std::memory_order_acquire);
  pending.request_stop();
  pending.wait();

  ASSERT_TRUE(source_progressed);
  EXPECT_EQ(pending_compute_count, 0U);
  EXPECT_EQ(pending.termination(), RunTermination::kStopped);
  EXPECT_TRUE(pending.diagnostics().empty());

  test::RuntimeAlignedChannelProbeState::reset(test::RuntimeAlignedChannelScript::kExactBoundary);
  RunSession fresh = run_async(plan);
  const bool observed = test::RuntimeAlignedChannelProbeState::wait_for_sink(1U);
  fresh.request_stop();
  fresh.wait();

  EXPECT_TRUE(observed);
  EXPECT_EQ(fresh.termination(), RunTermination::kStopped);
  EXPECT_TRUE(fresh.diagnostics().empty())
      << (fresh.diagnostics().empty() ? "" : fresh.diagnostics().front().token);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::sink_compute_count.load(), 1U);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_first.load(), 1U);
  EXPECT_EQ(test::RuntimeAlignedChannelProbeState::observed_second.load(), 2U);
}

// Replayed declarations alone must not opt an independently scheduled Operator into Channel
// activation preparation when reconstruction did not materialize endpoints for those ports.
// REQ: RUNTIME-004
// REQ: RUNTIME-015
TEST(RunSessionTest, ChannelActivationOwnershipRequiresMaterializedEndpoints) {
  test::RuntimeAlignedChannelProbeState::reset(test::RuntimeAlignedChannelScript::kExactBoundary);
  test::RuntimeUnconnectedPortProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_mixed_channel_unconnected_port_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool channel_observed = test::RuntimeAlignedChannelProbeState::wait_for_sink(1U);
  const bool independent_observed = test::RuntimeUnconnectedPortProbeState::wait_for_compute();
  session.request_stop();
  session.wait();

  EXPECT_TRUE(channel_observed);
  EXPECT_TRUE(independent_observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// The transport scan order is deliberately t+10 then t+5. One serial compute must publish an
// indivisible process-local activation cut before the next freeze so global TagTime ordering still
// invokes the t+5 sink first, independent of the background pump's interleaving.
// REQ: RUNTIME-004
// REQ: RUNTIME-015
// REQ: RUNTIME-016
// REQ: RUNTIME-018
TEST(RunSessionTest, PostComputeActivationCutOrdersOpposingSubscriberScanByTagTime) {
  test::RuntimeActivationCutProbeState::reset();
  const ExecutionPlan plan = test::make_runtime_activation_cut_order_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto state = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->process_local_channel_subscribers.size(), 2U);
  const auto source_name = [&](PortId source) -> std::string_view {
    const auto port = std::find_if(state->ports.begin(), state->ports.end(), [&](const auto& row) {
      return row.id == source;
    });
    return port == state->ports.end() ? std::string_view{} : std::string_view{port->local_name};
  };
  EXPECT_EQ(source_name(state->process_local_channel_subscribers[0].source), "a-later");
  EXPECT_EQ(source_name(state->process_local_channel_subscribers[1].source), "z-earlier");

  RunSession session = run_async(plan);
  const bool observed = test::RuntimeActivationCutProbeState::wait_for_sinks(2U);
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  const auto observations = test::RuntimeActivationCutProbeState::snapshot();
  ASSERT_EQ(observations.size(), 2U);
  EXPECT_EQ(observations[0].value, 5U);
  EXPECT_EQ(observations[1].value, 10U);
  EXPECT_LT(observations[0].capture_timestamp_ns, observations[1].capture_timestamp_ns);
  EXPECT_LT(observations[0].tag_time, observations[1].tag_time);
  EXPECT_EQ(observations[0].capture_timestamp_ns, observations[0].tag_time.logical_ns);
  EXPECT_EQ(observations[1].capture_timestamp_ns, observations[1].tag_time.logical_ns);
}

// REQ: RUNTIME-004
// REQ: RUNTIME-007
// REQ: RUNTIME-015
TEST(RunSessionTest, ChannelWindowSuccessorKeepsAnchorButUsesLaterActivationTime) {
  test::RuntimeWindowChannelProbeState::reset(test::RuntimeWindowChannelScript::kSuccessorClose);
  const ExecutionPlan plan = test::make_runtime_window_channel_plan();
  test::partition_parity::expect_accepted_cover(plan);

  const test::partition_parity::ScopedEnvironmentMutation changed_after_compile{
      "definitely-not-an-algorithm"};
  RunSession session = run_async(plan);
  const bool observed = test::RuntimeWindowChannelProbeState::wait_for_compute();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(test::RuntimeWindowChannelProbeState::selection_state.load(), InputState::kSelected);
  EXPECT_EQ(test::RuntimeWindowChannelProbeState::selection_reason.load(), SelectionReason::kNone);
  EXPECT_EQ(test::RuntimeWindowChannelProbeState::batch_size.load(), 5U);
  EXPECT_EQ(test::RuntimeWindowChannelProbeState::anchor_capture_ns.load() -
                test::RuntimeWindowChannelProbeState::batch_first_capture_ns.load(),
            15);
  EXPECT_EQ(test::RuntimeWindowChannelProbeState::batch_last_capture_ns.load() -
                test::RuntimeWindowChannelProbeState::anchor_capture_ns.load(),
            5);
  EXPECT_GT(test::RuntimeWindowChannelProbeState::activation_time_ns.load(),
            test::RuntimeWindowChannelProbeState::anchor_capture_ns.load());
}

// REQ: RUNTIME-004
// REQ: RUNTIME-007
// REQ: RUNTIME-015
TEST(RunSessionTest, ChannelWindowTimeoutInvokesWithMissingWindowSelection) {
  test::RuntimeWindowChannelProbeState::reset(test::RuntimeWindowChannelScript::kTimeoutClose);
  const ExecutionPlan plan = test::make_runtime_window_channel_plan();
  test::partition_parity::expect_accepted_cover(plan);

  const test::partition_parity::ScopedEnvironmentMutation changed_after_compile{
      "definitely-not-an-algorithm"};
  RunSession session = run_async(plan);
  const bool observed = test::RuntimeWindowChannelProbeState::wait_for_compute();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(test::RuntimeWindowChannelProbeState::selection_state.load(), InputState::kSafeDefault);
  EXPECT_EQ(test::RuntimeWindowChannelProbeState::selection_reason.load(),
            SelectionReason::kWindowTimeout);
  EXPECT_EQ(test::RuntimeWindowChannelProbeState::batch_size.load(), 0U);
  EXPECT_GT(test::RuntimeWindowChannelProbeState::activation_time_ns.load(),
            test::RuntimeWindowChannelProbeState::anchor_capture_ns.load());
}

// Each partition runner observes only the window-trigger state it owns. A
// same-lane source partition must not attempt to evaluate the sink partition's
// selector state while polling timeout deadlines.
// REQ: RUNTIME-004
// REQ: RUNTIME-007
// REQ: RUNTIME-041
TEST(RunSessionTest, CrossPartitionWindowTimeoutEvaluatesOnlyTheOwningRunner) {
  test::RuntimeWindowChannelProbeState::reset(test::RuntimeWindowChannelScript::kTimeoutClose);
  const ExecutionPlan baseline = test::make_runtime_window_channel_plan(true);
  test::partition_parity::expect_accepted_cover(baseline);
  const auto* const compiled = detail::ExecutionPlanAccess::state(baseline);
  ASSERT_NE(compiled, nullptr);
  ASSERT_EQ(compiled->window_triggers.size(), 1U);
  const auto sink_factory =
      std::find_if(compiled->executable_factories.operators.begin(),
                   compiled->executable_factories.operators.end(),
                   [&](const auto& factory) {
                     return factory.operator_id == compiled->window_triggers.front().owner.id;
                   });
  ASSERT_NE(sink_factory, compiled->executable_factories.operators.end());
  const auto sink_partition =
      std::find_if(compiled->execution_partitions.begin(),
                   compiled->execution_partitions.end(),
                   [&](const auto& partition) {
                     return std::find(partition.members.begin(),
                                      partition.members.end(),
                                      sink_factory->vertex) != partition.members.end();
                   });
  ASSERT_NE(sink_partition, compiled->execution_partitions.end());
  const auto timeout_control = std::make_shared<detail::RuntimeWindowTimeoutTestControl>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_window_timeout_test_control(baseline, timeout_control);

  RunSession session = run_async(plan);
  const bool observed = test::RuntimeWindowChannelProbeState::wait_for_compute();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(test::RuntimeWindowChannelProbeState::selection_state.load(), InputState::kSafeDefault);
  EXPECT_EQ(test::RuntimeWindowChannelProbeState::selection_reason.load(),
            SelectionReason::kWindowTimeout);
  const auto timeout_observations = timeout_control->snapshot();
  ASSERT_EQ(timeout_observations.size(), 1U);
  EXPECT_EQ(timeout_observations.front().owner, sink_factory->vertex);
  EXPECT_EQ(timeout_observations.front().partition, sink_partition->id);
}

// REQ: RUNTIME-004
// REQ: RUNTIME-015
TEST(RunSessionTest, RejectsInputPredicateSelectorFiringAndDeadlineReplayMismatches) {
  const ExecutionPlan plan = test::make_runtime_input_predicate_replay_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();

  constexpr std::array mismatches{
      test::RuntimeInputContractReplayMode::kSelectorMismatch,
      test::RuntimeInputContractReplayMode::kPredicateFiringMismatch,
      test::RuntimeInputContractReplayMode::kDeadlineMismatch,
  };
  for (const auto mismatch : mismatches) {
    SCOPED_TRACE(static_cast<unsigned>(mismatch));
    test::RuntimeInputContractPredicateOperator::mode.store(mismatch, std::memory_order_relaxed);
    auto reconstructed = detail::GraphExec::reconstruct(detail::ExecutionPlanAccess::retain(plan),
                                                        RunGenerationId{19U});
    ASSERT_FALSE(reconstructed);
    EXPECT_EQ(reconstructed.error().message, "RUNTIME_OPERATOR_REPLAY_MISMATCH");
  }
  test::RuntimeInputContractPredicateOperator::mode.store(
      test::RuntimeInputContractReplayMode::kMatch, std::memory_order_relaxed);
}

// REQ: RUNTIME-004
// REQ: RUNTIME-007
// REQ: RUNTIME-019
// REQ: CONTRACT-011
TEST(RunSessionTest, SuccessfulFalsePredicateRetiresInputsBeyondTheQueueCapacity) {
  reset_predicate_retirement_probe();
  const ExecutionPlan plan = make_predicate_retirement_plan();
  test::partition_parity::expect_accepted_cover(plan);

  RunSession session = run_async(plan);
  const bool computed = wait_for_predicate_retirement_compute();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(computed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  // Four successful false evaluations exceed the two-slot queue. Reaching the
  // fifth witness proves each no-fire transaction retired its predicate input.
  const PredicateRetirementSnapshot snapshot = predicate_retirement_snapshot();
  EXPECT_EQ(snapshot.source_count, 5U);
  EXPECT_EQ(snapshot.evaluation_count, 5U);
  EXPECT_EQ(snapshot.compute_count, 1U);
  EXPECT_EQ(snapshot.computed_driver, 5U);
}

// REQ: RUNTIME-004
// REQ: RUNTIME-007
// REQ: RUNTIME-019
// REQ: CONTRACT-011
TEST(RunSessionTest, ComputeOnlySkipDoesNotRollBackASuccessfulPredicateInput) {
  reset_predicate_retirement_probe();
  const ExecutionPlan plan = make_predicate_compute_skip_plan();
  test::partition_parity::expect_accepted_cover(plan);

  RunSession session = run_async(plan);
  const bool computed = wait_for_predicate_retirement_compute();
  session.request_stop();
  session.wait();

  ASSERT_TRUE(computed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  // Driver 1 evaluates successfully but its missing compute-only peer suppresses
  // the callback. Driver 2 must then advance and adopt the peer with the same
  // capture coordinate; replaying driver 1 would suppress forever.
  const PredicateRetirementSnapshot snapshot = predicate_retirement_snapshot();
  EXPECT_EQ(snapshot.source_count, 2U);
  EXPECT_EQ(snapshot.evaluation_count, 2U);
  EXPECT_EQ(snapshot.compute_count, 1U);
  EXPECT_EQ(snapshot.computed_driver, 2U);
  EXPECT_EQ(snapshot.computed_peer, 200U);
}

// REQ: RUNTIME-004
// REQ: RUNTIME-007
// REQ: RUNTIME-019
// Key setup: for each vector the direct oracle runs first to derive expected
// evaluator/compute counts; the Channel binding then runs and its observations
// are compared element-by-element against the oracle's per-candidate output.
TEST(RunSessionTest, ChannelInputPredicateMatchesDirectOracleAcrossTheBehaviorMatrix) {
  struct Vector {
    test::RuntimeInputPredicatePattern pattern;
    PredicateFiring firing;
    bool configured;
    test::RuntimeInputPredicateMissing missing;
  };
  constexpr std::array vectors{
      Vector{test::RuntimeInputPredicatePattern::kFalseTrueTrue,
             PredicateFiring::kOnFalseToTrue,
             false,
             test::RuntimeInputPredicateMissing::kFresh},
      Vector{test::RuntimeInputPredicatePattern::kFalseTrueTrue,
             PredicateFiring::kOnEveryTrueEvaluation,
             false,
             test::RuntimeInputPredicateMissing::kFresh},
      Vector{test::RuntimeInputPredicatePattern::kAlwaysTrue,
             PredicateFiring::kOnEveryTrueEvaluation,
             true,
             test::RuntimeInputPredicateMissing::kFresh},
      Vector{test::RuntimeInputPredicatePattern::kAlwaysTrue,
             PredicateFiring::kOnEveryTrueEvaluation,
             false,
             test::RuntimeInputPredicateMissing::kReuseLast},
      Vector{test::RuntimeInputPredicatePattern::kAlwaysTrue,
             PredicateFiring::kOnEveryTrueEvaluation,
             false,
             test::RuntimeInputPredicateMissing::kSkip},
      Vector{test::RuntimeInputPredicatePattern::kAlwaysTrue,
             PredicateFiring::kOnEveryTrueEvaluation,
             false,
             test::RuntimeInputPredicateMissing::kSafeDefault},
  };

  for (const Vector& vector : vectors) {
    SCOPED_TRACE(static_cast<unsigned>(vector.firing));
    SCOPED_TRACE(static_cast<unsigned>(vector.missing));
    auto direct = run_direct_predicate_oracle(
        vector.pattern, vector.firing, vector.configured, vector.missing);
    ASSERT_TRUE(direct) << direct.error().message;
    const auto expected_evaluator_count = static_cast<std::size_t>(std::count_if(
        direct->begin(), direct->end(), [](const auto& row) { return row.evaluator_called; }));
    const auto expected_compute_count =
        static_cast<std::size_t>(std::count_if(direct->begin(), direct->end(), [](const auto& row) {
          return row.disposition == detail::PredicateCandidateDisposition::kDegradedCompute ||
                 (row.disposition == detail::PredicateCandidateDisposition::kNominalCompute &&
                  row.firing_accepted);
        }));

    test::RuntimeInputPredicateChannelState::reset();
    const ExecutionPlan plan = test::make_runtime_input_predicate_channel_plan(
        vector.pattern, vector.firing, vector.configured, vector.missing);
    test::partition_parity::expect_accepted_cover(plan);
    ASSERT_EQ(plan.predicate_triggers().size(), 1U);
    ASSERT_EQ(plan.predicate_inputs().size(), 2U);
    EXPECT_TRUE(plan.predicate_inputs()[0].implicit_driver);
    EXPECT_EQ(plan.predicate_inputs()[0].argument_ordinal, 0U);
    EXPECT_FALSE(plan.predicate_inputs()[1].implicit_driver);
    EXPECT_EQ(plan.predicate_inputs()[1].argument_ordinal, 1U);
    EXPECT_EQ(plan.predicate_triggers().front().has_config, vector.configured);
    // Callable admission is compiler-owned and is exhaustively classified by
    // CompilePredicatePlan.AppliesTheClosedCallableProfileMatrix. The Channel
    // binding consumes this admitted record without transport-side reclassification.
    ASSERT_EQ(plan.predicate_callables().size(), 1U);
    EXPECT_EQ(plan.predicate_callables().front().effective_profile, Profile::kStandard);
    EXPECT_EQ(plan.predicate_callables().front().disposition,
              PredicateCallableDisposition::kAdmitted);
    EXPECT_FALSE(plan.predicate_callables().front().implementation_fingerprint.empty());
    if (vector.configured) {
      EXPECT_EQ(plan.predicate_triggers().front().config_schema, "test.runtime.input-predicate/v1");
      ASSERT_TRUE(plan.predicate_triggers().front().config_digest.has_value());
      EXPECT_EQ(plan.predicate_triggers().front().config_digest.value_or(std::string{}).size(),
                71U);
    } else {
      EXPECT_FALSE(plan.predicate_triggers().front().config_schema.has_value());
      EXPECT_FALSE(plan.predicate_triggers().front().config_digest.has_value());
    }

    const test::partition_parity::ScopedEnvironmentMutation changed_after_compile{
        "definitely-not-an-algorithm"};
    RunSession session = run_async(plan);
    const bool sourced = test::RuntimeInputPredicateChannelState::wait_for_source();
    const bool evaluated =
        expected_evaluator_count == 0U ||
        test::RuntimeInputPredicateChannelState::wait_for_evaluator(expected_evaluator_count);
    const bool computed =
        expected_compute_count == 0U ||
        test::RuntimeInputPredicateChannelState::wait_for_compute(expected_compute_count);
    if (expected_evaluator_count == 0U || expected_compute_count == 0U) {
      std::this_thread::sleep_for(std::chrono::milliseconds{25});
    }
    session.request_stop();
    session.wait();

    ASSERT_TRUE(sourced);
    ASSERT_TRUE(evaluated);
    ASSERT_TRUE(computed);
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(session.diagnostics().empty())
        << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
    // Key assertion: channel evaluator and compute counts exactly match the oracle.
    EXPECT_EQ(test::RuntimeInputPredicateChannelState::evaluator_count.load(),
              expected_evaluator_count);
    EXPECT_EQ(test::RuntimeInputPredicateChannelState::compute_count.load(),
              expected_compute_count);

    const auto evaluations = test::RuntimeInputPredicateChannelState::evaluation_snapshot();
    const auto computes = test::RuntimeInputPredicateChannelState::compute_snapshot();
    ASSERT_EQ(evaluations.size(), expected_evaluator_count);
    ASSERT_EQ(computes.size(), expected_compute_count);
    std::size_t evaluation_index = 0U;
    std::size_t compute_index = 0U;
    for (std::size_t candidate = 0U; candidate < direct->size(); ++candidate) {
      const auto& expected = (*direct)[candidate];
      if (expected.evaluator_called) {
        ASSERT_LT(evaluation_index, evaluations.size());
        const auto& actual = evaluations[evaluation_index++];
        EXPECT_EQ(actual.driver_value, candidate + 1U);
        EXPECT_EQ(actual.candidate_state_count, expected.candidate_state_count);
        EXPECT_EQ(actual.value, expected.value);
        EXPECT_EQ(actual.failed,
                  expected.disposition == detail::PredicateCandidateDisposition::kEvaluatorError);
        expect_input_selection_equal({InputState::kSelected, SelectionReason::kNone},
                                     actual.driver_selection);
        expect_input_selection_equal(expected.additional_selection, actual.additional_selection);
        EXPECT_EQ(actual.reference_time, actual.driver_metadata.capture_time());
        if (vector.missing == test::RuntimeInputPredicateMissing::kReuseLast && candidate > 0U) {
          EXPECT_EQ(actual.additional_value, 10U);
          EXPECT_EQ(actual.additional_metadata.sequence, 1U);
        }
      }
      const bool expects_compute =
          expected.disposition == detail::PredicateCandidateDisposition::kDegradedCompute ||
          (expected.disposition == detail::PredicateCandidateDisposition::kNominalCompute &&
           expected.firing_accepted);
      if (!expects_compute) {
        continue;
      }
      ASSERT_LT(compute_index, computes.size());
      const auto& actual_compute = computes[compute_index++];
      EXPECT_EQ(actual_compute.driver_value, candidate + 1U);
      expect_input_selection_equal({InputState::kSelected, SelectionReason::kNone},
                                   actual_compute.driver_selection);
      expect_input_selection_equal(expected.additional_selection,
                                   actual_compute.additional_selection);
      if (expected.disposition == detail::PredicateCandidateDisposition::kDegradedCompute) {
        EXPECT_FALSE(actual_compute.has_additional);
        expect_input_selection_equal(
            {InputState::kNotSelected, SelectionReason::kNotSelectedForActivation},
            actual_compute.compute_only_selection);
        continue;
      }
      const auto found = std::find_if(evaluations.begin(), evaluations.end(), [&](const auto& row) {
        return row.driver_value == actual_compute.driver_value;
      });
      ASSERT_NE(found, evaluations.end());
      EXPECT_EQ(actual_compute.tag, found->tag);
      EXPECT_EQ(actual_compute.tag_time, found->tag_time);
      EXPECT_EQ(actual_compute.activation_time, found->reference_time);
      expect_sample_metadata_equal(found->driver_metadata, actual_compute.driver_metadata);
      ASSERT_TRUE(actual_compute.has_additional);
      EXPECT_EQ(actual_compute.additional_value, found->additional_value);
      expect_sample_metadata_equal(found->additional_metadata, actual_compute.additional_metadata);
    }
    if (vector.missing == test::RuntimeInputPredicateMissing::kFresh) {
      ASSERT_EQ(evaluations.size(), 3U);
      // All three additional samples were queued before any driver. Exact
      // evaluation nevertheless pins the candidate-time sample rather than the
      // newest arrival, and compute adopts that identical snapshot.
      EXPECT_EQ(evaluations.front().additional_metadata.sequence, 1U);
      EXPECT_EQ(evaluations.back().additional_metadata.sequence, 3U);
    }
  }
}

// REQ: RUNTIME-004
// REQ: RUNTIME-007
// REQ: RUNTIME-019
TEST(RunSessionTest, ChannelInputPredicateErrorRollsBackWithoutCompute) {
  // Key setup: kTrueThenError fires on the first candidate (compute==1), then
  // returns an error on the second. The evaluator's private state count must stay
  // at 2 — proving the error rolled back without resetting prior truth.
  auto direct = run_direct_predicate_oracle(test::RuntimeInputPredicatePattern::kTrueThenError,
                                            PredicateFiring::kOnEveryTrueEvaluation,
                                            false,
                                            test::RuntimeInputPredicateMissing::kFresh);
  ASSERT_TRUE(direct) << direct.error().message;
  ASSERT_EQ(direct->size(), 3U);
  EXPECT_EQ((*direct)[0].disposition, detail::PredicateCandidateDisposition::kNominalCompute);
  EXPECT_EQ((*direct)[1].disposition, detail::PredicateCandidateDisposition::kEvaluatorError);
  EXPECT_EQ((*direct)[2].disposition, detail::PredicateCandidateDisposition::kEvaluatorError);
  EXPECT_TRUE((*direct)[1].prior_truth);
  EXPECT_TRUE((*direct)[2].prior_truth);
  EXPECT_EQ((*direct)[1].candidate_state_count, 2U);
  EXPECT_EQ((*direct)[2].candidate_state_count, 2U);

  auto scenario = run_channel_input_predicate_error_scenario();
  ASSERT_TRUE(scenario) << scenario.error().message;
  const PredicateErrorScenarioEvidence& evidence = *scenario;
  const PredicateTriggerPlanRecord& trigger = evidence.trigger;

  // Key assertion: the session fails after exactly 2 evaluations and 1 compute;
  // the second evaluation's candidate_state_count==2 confirms state was preserved.
  EXPECT_EQ(evidence.termination, RunTermination::kFailed);
  EXPECT_EQ(evidence.evaluator_count, 2U);
  EXPECT_EQ(evidence.compute_count, 1U);
  const auto evaluations = evidence.evaluations;
  ASSERT_EQ(evaluations.size(), 2U);
  EXPECT_EQ(evaluations[0].candidate_state_count, 1U);
  EXPECT_EQ(evaluations[1].candidate_state_count, 2U);
  EXPECT_FALSE(evaluations[0].failed);
  EXPECT_TRUE(evaluations[1].failed);
  ASSERT_EQ(evidence.diagnostics.size(), 1U);
  EXPECT_EQ(evidence.diagnostics.front().token, "RUNTIME_OPERATOR_CALLBACK_FAILED");
  EXPECT_EQ(evidence.diagnostics.front().service, "predicate_evaluator");
  const auto published = evidence.published_details;
  ASSERT_EQ(published.size(), 1U);
  ASSERT_EQ(published.front().size(), 1U);
  const detail::RuntimeDiagnosticDetail& detail = published.front().front();
  EXPECT_TRUE(detail.present);
  EXPECT_EQ(detail.error.code, ErrorCode::kFailure);
  EXPECT_EQ(detail.error.message, "input predicate failure");
  EXPECT_EQ(detail.predicate_id, trigger.id);
  EXPECT_EQ(detail.operator_id, trigger.owner.id);
  EXPECT_TRUE(static_cast<bool>(detail.source));
  // The source emits three predicate inputs and three compute-only samples
  // before the driver stream, so the second driver is publication witness 8
  // even though its per-port Sample sequence is 2.
  EXPECT_EQ(detail.source_sequence, 8U);
  EXPECT_EQ(detail.tag, evaluations[1].tag);
  EXPECT_EQ(detail.tag_time, evaluations[1].tag_time);
  EXPECT_EQ(detail.reference_time, evaluations[1].reference_time);
  EXPECT_EQ(detail.state_generation, detail.tag.generation);
}

}  // namespace
}  // namespace holoscan
