/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/channel/pod_type.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/run.hpp>

#include "compiler/execution_plan_state.hpp"
#include "runtime/predicate_evaluator.hpp"
#include "runtime/runtime_callback_context.hpp"
#include "runtime/runtime_diagnostic_sink.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime_test_graph.hpp"

namespace holoscan {

struct RuntimeChannelRoundTripPod {
  std::uint32_t first{};
  std::uint32_t second{};
};

template <>
struct ChannelTypeIdentity<RuntimeChannelRoundTripPod> {
  static constexpr std::string_view name = "nvidia.holoscan.test.RuntimeChannelRoundTripPod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<RuntimeChannelRoundTripPod> {
  static constexpr auto fields =
      pod_fields(pod_field("first", &RuntimeChannelRoundTripPod::first,
                           offsetof(RuntimeChannelRoundTripPod, first)),
                 pod_field("second", &RuntimeChannelRoundTripPod::second,
                           offsetof(RuntimeChannelRoundTripPod, second)));
};

namespace test::run_session_channel_execution {

struct RuntimePayloadRoundTripObservation {
  RuntimeChannelRoundTripPod pod{};
  std::string schema_value;
  SampleMetadata metadata{};
};

enum class RuntimePayloadRoundTripSourceBehavior : std::uint8_t {
  kEmitOnce,
  kEmitTwiceAcrossActivations,
  kNoEmission,
  kBackpressured,
  kTypedFailure,
  kException,
  kSecondEmitHandled,
  kSecondEmitReturned,
};

enum class RuntimePayloadRoundTripSinkBehavior : std::uint8_t {
  kObserve,
  kTypedFailure,
  kException,
};

struct RuntimePayloadRoundTripCallbackEvidence {
  std::array<std::uint8_t, 2U> order{};
  std::array<detail::RuntimeCallbackIdentity, 2U> identities{};
  std::uint32_t callback_count{};
  std::uint32_t active_callbacks{};
  std::uint32_t max_active_callbacks{};
  bool overlap{};
  bool same_thread{};
};

enum class RuntimePayloadRoundTripRoot : std::uint8_t {
  kClock,
  kNotification,
  kReadable,
  kSpin,
};

enum class RuntimeDirectContinuationPayloadKind : std::uint8_t {
  kPod,
  kGeneratedSchema,
};

struct RuntimeDirectContinuationRoundTripEvidence {
  bool direct_delivery{};
  bool observed{};
  RunTermination termination{RunTermination::kFailed};
  std::size_t diagnostic_count{};
  std::optional<RuntimePayloadRoundTripObservation> expected;
  std::optional<RuntimePayloadRoundTripObservation> received;
  RuntimePayloadRoundTripCallbackEvidence callbacks;
  detail::RuntimePhaseCounters::Snapshot counters{};
};

// Report phase and optional-input evidence to
// RunSessionTest.OptionalComputeSideRemainsUnselectedUntilPredicateFires.
struct PredicateRetirementSnapshot {
  std::size_t degraded_callback_count{};
  std::size_t optional_not_selected_count{0U};
  InputState optional_state{InputState::kNotSelected};
  SelectionReason optional_reason{SelectionReason::kNotSelectedForActivation};
  ErrorCode optional_scalar_error{ErrorCode::kSuccess};
  ErrorCode optional_batch_error{ErrorCode::kSuccess};

  std::size_t source_count{};
  std::size_t evaluation_count{};
  std::size_t compute_count{};
  std::uint32_t computed_driver{};
  std::uint32_t computed_peer{};
};

struct DirectPredicateOutcome {
  detail::PredicateCandidateDisposition disposition{};
  bool evaluator_called{};
  bool value{};
  bool firing_accepted{};
  bool prior_truth{};
  InputSelection additional_selection{};
  std::uint32_t candidate_state_count{};
  Error evaluator_error{ErrorCode::kSuccess};
};

struct PredicateErrorScenarioEvidence {
  RunTermination termination{RunTermination::kFailed};
  std::size_t evaluator_count{};
  std::size_t compute_count{};
  std::vector<test::RuntimeInputPredicateEvaluationObservation> evaluations;
  std::vector<Diagnostic> diagnostics;
  std::vector<std::vector<detail::RuntimeDiagnosticDetail>> published_details;
  PredicateTriggerPlanRecord trigger;
};

void reset_payload_round_trip_probe(RuntimePayloadRoundTripSourceBehavior source_behavior =
                                        RuntimePayloadRoundTripSourceBehavior::kEmitOnce,
                                    RuntimePayloadRoundTripSinkBehavior sink_behavior =
                                        RuntimePayloadRoundTripSinkBehavior::kObserve);
[[nodiscard]] bool wait_for_payload_source();
[[nodiscard]] bool wait_for_pod();
[[nodiscard]] bool wait_for_pod_count(std::uint32_t count);
[[nodiscard]] bool wait_for_schema();
[[nodiscard]] RuntimePayloadRoundTripCallbackEvidence payload_callback_evidence();
[[nodiscard]] std::optional<ErrorCode> second_emit_snapshot();
[[nodiscard]] std::optional<RuntimePayloadRoundTripObservation> expected_pod_snapshot();
[[nodiscard]] std::optional<RuntimePayloadRoundTripObservation> observed_pod_snapshot();
[[nodiscard]] std::optional<RuntimePayloadRoundTripObservation> expected_schema_snapshot();
[[nodiscard]] std::optional<RuntimePayloadRoundTripObservation> observed_schema_snapshot();

void reset_blocking_pressure_probe(bool fail_after_pressure = false);
[[nodiscard]] bool wait_for_blocking_pressure();
[[nodiscard]] std::pair<ErrorCode, std::uint32_t> blocking_pressure_snapshot();
[[nodiscard]] bool blocking_pressure_callback_active();

void reset_predicate_retirement_probe();
[[nodiscard]] bool wait_for_predicate_retirement_compute();
[[nodiscard]] PredicateRetirementSnapshot predicate_retirement_snapshot();

[[nodiscard]] ExecutionPlan make_runtime_closed_island_tensor_ineligible_plan();
[[nodiscard]] ExecutionPlan make_runtime_pod_round_trip_plan(
    bool enable_lifecycle = false,
    RuntimePayloadRoundTripRoot root = RuntimePayloadRoundTripRoot::kClock);
[[nodiscard]] ExecutionPlan make_runtime_pod_cross_partition_round_trip_plan();
[[nodiscard]] ExecutionPlan make_runtime_pod_side_input_plan();
[[nodiscard]] ExecutionPlan make_runtime_pod_predicate_plan();
[[nodiscard]] ExecutionPlan make_runtime_pod_deadline_plan();
[[nodiscard]] ExecutionPlan make_runtime_pod_mutable_param_plan();
[[nodiscard]] ExecutionPlan make_runtime_pod_nonretained_sibling_plan();
[[nodiscard]] ExecutionPlan make_runtime_pod_external_export_ambiguity_plan();
[[nodiscard]] ExecutionPlan make_runtime_pod_double_emit_plan();
[[nodiscard]] ExecutionPlan make_runtime_schema_round_trip_plan(bool enable_lifecycle = false);
[[nodiscard]] ExecutionPlan make_runtime_blocking_pressure_plan(std::string_view graph_name);
// Compile predicate drivers and optional compute/evaluation sides for
// RunSessionTest.OptionalComputeSidePreservesAnEarlierDegradedPredicatePhase.
[[nodiscard]] ExecutionPlan make_predicate_retirement_plan(bool optional_side = false,
                                                           bool degraded_evaluation = false);
[[nodiscard]] ExecutionPlan make_predicate_compute_skip_plan();

[[nodiscard]] RuntimeDirectContinuationRoundTripEvidence run_direct_continuation_round_trip(
    RuntimeDirectContinuationPayloadKind payload_kind, bool disable_elision,
    bool enable_lifecycle = false, bool disable_closed_island = false,
    std::shared_ptr<detail::RuntimeClosedIslandTestControl> closed_island_control = {},
    RuntimePayloadRoundTripSinkBehavior sink_behavior =
        RuntimePayloadRoundTripSinkBehavior::kObserve);

[[nodiscard]] expected<std::vector<DirectPredicateOutcome>, Error> run_direct_predicate_oracle(
    test::RuntimeInputPredicatePattern pattern, PredicateFiring firing, bool configured,
    test::RuntimeInputPredicateMissing missing);

[[nodiscard]] expected<PredicateErrorScenarioEvidence, Error>
run_channel_input_predicate_error_scenario();

}  // namespace test::run_session_channel_execution
}  // namespace holoscan
