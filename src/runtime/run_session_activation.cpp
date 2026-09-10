/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/profiler/nvtx3.hpp>

#include "run_session_internal.hpp"

#include "runtime_test_access.hpp"

namespace holoscan::detail {

// Input selection, callback admission, and exact activation retirement remain
// one protocol so terminal paths cannot skip commit or rollback obligations.

namespace {

class ScopedComputeCudaDevice final {
 public:
  explicit ScopedComputeCudaDevice(const DeviceId requested) noexcept {
    if (!requested.valid()) {
      active_ = true;
      restored_ = true;
      return;
    }
    if (cudaGetDevice(&previous_) != cudaSuccess) {
      return;
    }
    if (previous_ != requested.value) {
      if (cudaSetDevice(requested.value) != cudaSuccess) {
        return;
      }
      changed_ = true;
    }
    active_ = true;
  }

  ScopedComputeCudaDevice(const ScopedComputeCudaDevice&) = delete;
  ScopedComputeCudaDevice& operator=(const ScopedComputeCudaDevice&) = delete;

  ~ScopedComputeCudaDevice() {
    if (!restored_ && changed_) {
      static_cast<void>(cudaSetDevice(previous_));
    }
  }

  [[nodiscard]] bool active() const noexcept { return active_; }

  /// Restore the lane's prior ambient device after user code, including when the requested device
  /// already matched on entry but the callback changed it. Host-only callbacks take the invalid
  /// DeviceId fast path and make no CUDA calls.
  [[nodiscard]] bool restore_after_callback() noexcept {
    if (restored_) {
      return true;
    }
    int current{};
    if (cudaGetDevice(&current) != cudaSuccess) {
      return false;
    }
    if (current != previous_) {
      changed_ = true;
      if (cudaSetDevice(previous_) != cudaSuccess) {
        return false;
      }
    }
    restored_ = true;
    changed_ = false;
    return true;
  }

 private:
  int previous_{};
  bool changed_{};
  bool active_{};
  bool restored_{};
};

struct ComputeParameterIngress {
  ParameterUpdateCoordinator* coordinator{};
  OperatorRef self{};
};

[[nodiscard]] ParamSubmitResult submit_compute_parameter_update(void* state,
                                                                const ParamInlinePatch* patches,
                                                                std::size_t patch_count) noexcept {
  auto* const ingress = static_cast<ComputeParameterIngress*>(state);
  if (ingress == nullptr || ingress->coordinator == nullptr || patches == nullptr ||
      patch_count == 0U) {
    return ParamSubmitResult{
        .status = ParamUpdateStatus::kRejected,
        .state = ParamUpdateState::kRejected,
    };
  }
  for (std::size_t index = 0U; index < patch_count; ++index) {
    if (patches[index].target.operator_instance != ingress->self) {
      return ParamSubmitResult{
          .status = ParamUpdateStatus::kUnsupportedAtomicScope,
          .state = ParamUpdateState::kRejected,
      };
    }
  }
  return ingress->coordinator->try_submit_from_compute(patches, patch_count);
}

void publish_pending_parameter_updates(ParameterUpdateCoordinator& coordinator, OperatorRef owner) {
  const ParamUpdateSequence accepted_cutoff = coordinator.publication_frontier();
  static_cast<void>(coordinator.publish_next_at_callback_boundary(owner, accepted_cutoff));
}

}  // namespace

class ActivationParameterBinding final {
 public:
  ActivationParameterBinding(ParameterUpdateCoordinator& coordinator,
                             const PinnedParameterHeader& static_header) noexcept
      : coordinator_(&coordinator), header_(&static_header) {}
  ActivationParameterBinding(const ActivationParameterBinding&) = delete;
  ActivationParameterBinding& operator=(const ActivationParameterBinding&) = delete;
  ~ActivationParameterBinding() { release(); }

  void bind(ParameterGenerationPin pin) noexcept {
    pin_.emplace(std::move(pin));
    header_ = pin_->header();
  }

  void release() noexcept {
    if (pin_.has_value()) {
      coordinator_->release(*pin_);
      pin_.reset();
    }
  }

  [[nodiscard]] const PinnedParameterHeader* header() const noexcept { return header_; }
  [[nodiscard]] const ParameterGenerationPin* pin() const noexcept {
    return pin_.has_value() ? &*pin_ : nullptr;
  }

 private:
  ParameterUpdateCoordinator* coordinator_{};
  const PinnedParameterHeader* header_{};
  std::optional<ParameterGenerationPin> pin_;
};

template <bool TraceCallback>
// Compute admission, observation, callback ownership, and retirement form one lane transaction.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> RunSessionState::invoke_compute_lane(void* state) noexcept {
  auto& task = *static_cast<ComputeLaneInvocation*>(state);
  // Device selection is callback admission, not callback work. Complete it before the final
  // deadline sample and parameter-generation observation so a failed selection cannot publish
  // callback evidence or hide selection latency.
  ScopedComputeCudaDevice selected_device{task.cuda_device};
  if (!selected_device.active()) {
    task.admission = ComputeAdmissionDisposition::kCudaDeviceUnavailable;
    return {};
  }
  // The accepted lane and process-local dispatch token are both held here. Sample the final
  // policy coordinate before observation commit and user compute so lane handoff and every
  // earlier same- or cross-Operator delay remain measurable.
  if (task.deadline != nullptr) {
    const auto& final_entry_control =
        task.session->plan_->executable_factories.test_operator_completion_final_entry_control;
    if (final_entry_control != nullptr) {
      final_entry_control->deadline_evaluation_started();
    }
    auto dispatch_time = task.deadline_clock.now();
    if (!dispatch_time) {
      task.admission = ComputeAdmissionDisposition::kDeadlineReadFailed;
      return {};
    }
    auto gate = evaluate_runtime_deadline(*task.deadline,
                                          task.source_decision_time,
                                          task.reference_time,
                                          *dispatch_time,
                                          std::chrono::nanoseconds{task.deadline_budget_ns});
    if (!gate) {
      task.admission_error = std::move(gate.error());
      task.admission = ComputeAdmissionDisposition::kDeadlineEvaluationFailed;
      return {};
    }
    task.deadline_gate.emplace(*gate);
    task.context->activation = merge_deadline_activation_status(task.context->activation, *gate);
    if (gate->disposition == DeadlineGateDisposition::kFailRun) {
      task.admission = ComputeAdmissionDisposition::kDeadlineFailRun;
      return {};
    }
  }
  const auto& admission_control =
      task.session->plan_->executable_factories.test_termination_admission_control;
  if (admission_control != nullptr) {
    admission_control->before_claim();
  }
  if (!task.callback_claim.acquire(task.session->termination_state_)) {
    task.admission = ComputeAdmissionDisposition::kRunClosing;
    return {};
  }
  if (admission_control != nullptr) {
    admission_control->after_claim();
  }
  // Observation is callback evidence, so commit it only after this activation owns the final
  // session-wide callback claim. Once claimed, run closure cannot retire the activation without
  // letting its callback complete.
  if (task.parameter_generation != nullptr) {
    auto observed = task.parameter_updates->commit_activation_observation(
        task.self, *task.parameter_generation);
    const bool injected_observation_failure =
        admission_control != nullptr && admission_control->consume_parameter_observation_failure();
    if (!observed || injected_observation_failure) {
      task.callback_claim.release();
      task.admission = ComputeAdmissionDisposition::kParameterObservationFailed;
      return {};
    }
  }
  task.admission = ComputeAdmissionDisposition::kAdmitted;
  expected<void, Error> result;
  if constexpr (TraceCallback) {
    task.phase_counters->trace(RuntimePhaseTracePoint::kComputeCallbackEntered,
                               task.selection->receiver_index,
                               task.selection->ready.stable_source_id,
                               task.source_sequence);
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    {
      const ParamGeneralControlContextScope general_control_scope;
      const nvtx3::unique_range_in<holoscan::profiler::holoscan_operator_domain> compute_range{
          task.instance->compute_nvtx_message, nvtx3::color{nvtx3::rgb{118, 185, 0}}};
      result = OperatorRuntimeAccess::invoke(*task.instance->op, *task.context);
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    result = make_unexpected(Error{ErrorCode::kFailure});
  }
#endif
  if constexpr (TraceCallback) {
    task.phase_counters->trace(RuntimePhaseTracePoint::kComputeCallbackReturned,
                               task.selection->receiver_index,
                               task.selection->ready.stable_source_id,
                               task.source_sequence);
  }
  if (!selected_device.restore_after_callback()) {
    result = make_unexpected(
        Error{ErrorCode::kNotSupported, "RUNTIME_COMPUTE_CUDA_DEVICE_RESTORE_FAILED"});
  }
  task.completed_selection = task.dispatcher->complete(*task.selection).has_value();
  if (result && task.completed_selection && task.direct_continuation_enabled &&
      task.channels != nullptr && task.direct_continuation_clock.id().valid()) {
    auto observed_at = task.direct_continuation_clock.now();
    if (!observed_at) {
      task.direct_continuation_error = std::move(observed_at.error());
      task.direct_continuation_clock_failed = true;
    } else {
      auto receipt = task.channels->poll_direct_continuation(
          task.channel_owner, task.execution_lane, *observed_at);
      if (!receipt) {
        task.direct_continuation_error = std::move(receipt.error());
      } else {
        task.direct_continuation_receipt = std::move(*receipt);
      }
    }
  }
  return result;
}

[[nodiscard]] expected<void, Error> RunSessionState::retire_frozen_source(
    ReadyRunState& state, FrozenRuntimeCandidate& candidate) noexcept {
  return retire_frozen_source_impl<true>(state, candidate);
}

void RunSessionState::clear_input_selection_scratch(ReadyRunState& state,
                                                    const FrozenRuntimeCandidate& candidate,
                                                    TimePoint& semantic_reference) noexcept {
  state.selection.selector_transactions.clear();
  state.selection.aligned_transactions.clear();
  state.selection.input_selections.clear();
  state.selection.selected_samples.clear();
  state.selection.channel_selections.clear();
  state.selection.arrival_trigger_sample.reset();
  state.selection.predicate_inputs.clear();
  semantic_reference = candidate.witness.reference_time;
}

[[nodiscard]] bool RunSessionState::append_selected_samples(
    ReadyRunState& state, std::span<const TemporalSampleRecord> samples) noexcept {
  const std::size_t used = state.selection.selected_samples.size();
  if (used > state.selection.maximum_selected_samples ||
      used > state.selection.selected_samples.capacity() ||
      samples.size() > state.selection.maximum_selected_samples - used ||
      samples.size() > state.selection.selected_samples.capacity() - used) {
    return false;
  }
  state.selection.selected_samples.insert(
      state.selection.selected_samples.end(), samples.begin(), samples.end());
  return true;
}

// Preserve selection identity, payload claims, and semantic time as one frozen append operation.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool RunSessionState::append_frozen_input_selection(
    ReadyRunState& state, ReadyRunState::InputSelectorSlot& slot,
    RuntimeInputSelectionState::Transaction& transaction) noexcept {
  const InputSelection selection = transaction.selection();
  const auto selected = transaction.selected();
  // freeze_next orders by reference time first, so an unfrozen candidate older than the witness
  // that did freeze would contradict that ordering outright. Reading the coordinate here is what
  // separates "the successor was left pending" from "the pending event is a later witness and
  // the successor was never created at all".
  std::int64_t pending_oldest_reference_ns = 0;
  if (state.coordinator != nullptr) {
    const auto oldest = state.coordinator->oldest_pending_reference();
    pending_oldest_reference_ns = oldest ? oldest->timestamp_ns : 0;
  }
  // Whether the record this round's eviction was traded for is still in the slot it was promoted
  // into. Absent means the producer took it back before it could publish, which is the one
  // remaining account of a loss that survives every other measurement.
  std::uint8_t refill_outcome = 0U;
  if (state.channels != nullptr) {
    const auto outcome = state.channels->last_refill_outcome(slot.input);
    refill_outcome = outcome.has_value() ? *outcome : 0U;
  }
  // The one place an ordinary selection is formed, so it is the only place the question can be
  // answered per activation: a selection that found nothing is split by whether this same
  // dispatch iteration's reconcile had just emptied the cache it read. The two counters are
  // exhaustive over lost selections, so a residual that lands entirely on the second one says
  // the eviction ordering is not the remaining cause and a different guard is needed.
  if (selection.state != InputState::kSelected && selection.state != InputState::kReused) {
    count_phase(!slot.freshness_evicted_this_round
                    ? RuntimePhaseCounter::kSelectionMissedWithoutFreshnessEviction
                : !slot.freshness_promoted_this_round
                    ? RuntimePhaseCounter::kSelectionMissedAfterUnrefilledEviction
                : slot.freshness_gate_inert_this_round
                    ? RuntimePhaseCounter::kSelectionMissedAfterUngatedEviction
                : state.poll_incomplete_this_round
                    ? RuntimePhaseCounter::kSelectionMissedAfterUnpolledEviction
                    : RuntimePhaseCounter::kSelectionMissedAfterFreshnessEviction);
    if (slot.freshness_evicted_this_round && phase_counters_ != nullptr) {
      phase_counters_->record_retention_miss(TemporalRetentionMissRecord{
          .witness_reference_ns = slot.selection_reference_ns,
          .successor_capture_ns = slot.freshness_successor_capture_ns,
          .floor_ns = slot.freshness_floor_ns,
          .selection_state = static_cast<std::uint8_t>(selection.state),
          .selection_reason = static_cast<std::uint8_t>(selection.reason),
          .floor_present = !slot.freshness_gate_inert_this_round,
          .ingested_this_round = slot.freshness_ingested_this_round,
          .refill_outcome = refill_outcome,
          .promoted_capture_ns = slot.freshness_promoted_capture_ns,
          .round_ingested_count = state.round_ingested_count,
          .pending_at_selection =
              state.coordinator != nullptr
                  ? static_cast<std::uint32_t>(state.coordinator->pending_count())
                  : 0U,
          .pending_oldest_reference_ns = pending_oldest_reference_ns,
      });
    }
  }
  if (!append_selected_samples(state, selected)) {
    return false;
  }
  state.selection.input_selections.push_back(selection);
  const InputSelectionPlanRecord& plan = slot.selector.plan();
  state.selection.channel_selections.push_back(RuntimeChannelActivationSelection{
      .port = slot.input,
      .selection = selection,
      .samples = selected,
      .batch = plan.cardinality == InputSelectionCardinality::kSampleBatch,
      .destructive = plan.batch_selection_mode != BatchSelectionMode::kNonDestructiveSnapshot,
  });
  return true;
}

[[nodiscard]] expected<void, Error> RunSessionState::append_input_selection(
    ReadyRunState& state, ReadyRunState::InputSelectorSlot& slot, TimePoint reference_time,
    TimePoint& semantic_reference, std::optional<TimePoint> future_reference_lower_bound) noexcept {
  slot.selection_reference_ns = reference_time.timestamp_ns;
  auto transaction = slot.selector.begin(
      reference_time, slot.arrival_watermark, slot.producer_epoch, future_reference_lower_bound);
  if (!transaction) {
    return make_unexpected(std::move(transaction.error()));
  }
  const TimePoint selector_reference = transaction->semantic_reference_time();
  if (selector_reference.valid()) {
    semantic_reference = selector_reference;
  }
  state.selection.selector_transactions.push_back(std::move(*transaction));
  if (!append_frozen_input_selection(state, slot, state.selection.selector_transactions.back())) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  return {};
}

// Form aligned, bracketed-window, and ordinary selections in authoring order while retaining one
// semantic reference and one transaction set for the activation.
[[nodiscard]] RunSessionState::SelectionPreparation
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
RunSessionState::prepare_nonpredicate_input_selections(ReadyRunState& state,
                                                       const RuntimeOperatorInstance& instance,
                                                       const FrozenRuntimeCandidate& candidate,
                                                       TimePoint& semantic_reference) noexcept {
  if (arrival_trigger_pin_matches(instance, candidate) &&
      std::none_of(
          state.input_selectors.begin(), state.input_selectors.end(), [&](const auto& slot) {
            return slot.owner == instance.vertex && slot.input == candidate.witness.port;
          })) {
    if (state.channels == nullptr || candidate.witness.source_sequence == 0U) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
      return SelectionPreparation::kFailed;
    }
    auto sample =
        state.channels->temporal_sample(candidate.witness.port, candidate.witness.source_sequence);
    if (!sample) {
      append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
      return SelectionPreparation::kFailed;
    }
    state.selection.arrival_trigger_sample = *sample;
    const std::span<const TemporalSampleRecord> trigger_sample{
        &*state.selection.arrival_trigger_sample, 1U};
    if (!append_selected_samples(state, trigger_sample)) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
      return SelectionPreparation::kFailed;
    }
    state.selection.input_selections.push_back(
        InputSelection{InputState::kSelected, SelectionReason::kNone});
    state.selection.channel_selections.push_back(RuntimeChannelActivationSelection{
        .port = candidate.witness.port,
        .selection = state.selection.input_selections.back(),
        .samples = trigger_sample,
        .batch = false,
        .destructive = true,
    });
  }
  const auto aligned = std::find_if(
      state.aligned_selectors.begin(), state.aligned_selectors.end(), [&](const auto& slot) {
        return slot.owner == instance.vertex;
      });
  if (aligned != state.aligned_selectors.end()) {
    AlignedCandidateDisposition disposition{AlignedCandidateDisposition::kPending};
    while (true) {
      auto transaction = aligned->selector.begin();
      if (!transaction) {
        return transaction.error().code == ErrorCode::kNotReady ? SelectionPreparation::kNoFire
                                                                : SelectionPreparation::kFailed;
      }
      if (transaction->disposition() != AlignedCandidateDisposition::kRetireMisaligned) {
        disposition = transaction->disposition();
        state.selection.aligned_transactions.push_back(std::move(*transaction));
        break;
      }
      // A window miss deterministically retires every minimum head. Continue on the same
      // readiness candidate so a backlog of incompatible heads cannot stall until unrelated
      // future input arrives.
      if (!transaction->finish(false)) {
        return SelectionPreparation::kFailed;
      }
      if (!synchronize_temporal_payload_claims(state, instance.vertex)) {
        return SelectionPreparation::kFailed;
      }
      continue;
    }
    auto& frozen = state.selection.aligned_transactions.back();
    if (frozen.reference_time().valid()) {
      semantic_reference = frozen.reference_time();
    }
    if (disposition == AlignedCandidateDisposition::kNoFire ||
        disposition == AlignedCandidateDisposition::kPending) {
      return SelectionPreparation::kNoFire;
    }

    const auto inputs = aligned->selector.member_inputs();
    const auto selections = frozen.selections();
    const auto selected = frozen.selected();
    if (inputs.size() != selections.size()) {
      return SelectionPreparation::kFailed;
    }
    std::size_t selected_index = 0U;
    for (std::size_t index = 0U; index < inputs.size(); ++index) {
      const InputSelection selection = selections[index];
      const bool has_sample =
          selection.state == InputState::kSelected || selection.state == InputState::kReused;
      if (has_sample && selected_index >= selected.size()) {
        return SelectionPreparation::kFailed;
      }
      const std::span<const TemporalSampleRecord> member_samples =
          has_sample ? selected.subspan(selected_index++, 1U)
                     : std::span<const TemporalSampleRecord>{};
      if (!append_selected_samples(state, member_samples)) {
        return SelectionPreparation::kFailed;
      }
      state.selection.input_selections.push_back(selection);
      state.selection.channel_selections.push_back(RuntimeChannelActivationSelection{
          .port = inputs[index],
          .selection = selection,
          .samples = member_samples,
          .batch = false,
          .destructive = selection.state == InputState::kSelected,
      });
    }
    if (selected_index != selected.size()) {
      return SelectionPreparation::kFailed;
    }
  }
  // OnWindow closes at a successor or timeout witness that is later than its semantic anchor.
  // Freeze the bracket first to recover that anchor, then evaluate every peer selector at the
  // anchor while retaining the close witness as activation/TagTime. Materialize selections in
  // authored input order even when the bracketed input is not declared first.
  auto bracketed = state.input_selectors.end();
  for (auto slot = state.input_selectors.begin(); slot != state.input_selectors.end(); ++slot) {
    if (slot->owner != instance.vertex || slot->aligned_member ||
        slot->selector.plan().selector != InputSelectorKind::kBracketedWindow) {
      continue;
    }
    if (bracketed != state.input_selectors.end()) {
      append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
      return SelectionPreparation::kFailed;
    }
    bracketed = slot;
  }

  std::optional<std::size_t> bracketed_transaction_index;
  if (bracketed != state.input_selectors.end()) {
    auto transaction = bracketed->selector.begin(
        candidate.witness.reference_time, bracketed->arrival_watermark, bracketed->producer_epoch);
    if (!transaction) {
      if (transaction.error().code == ErrorCode::kNotReady) {
        return SelectionPreparation::kNoFire;
      }
      append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
      return SelectionPreparation::kFailed;
    }
    const TimePoint anchor = transaction->semantic_reference_time();
    if (!anchor.valid() || anchor.clock_id != candidate.witness.reference_time.clock_id ||
        anchor.timestamp_ns > candidate.witness.reference_time.timestamp_ns) {
      append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
      return SelectionPreparation::kFailed;
    }
    semantic_reference = anchor;
    state.selection.selector_transactions.push_back(std::move(*transaction));
    bracketed_transaction_index = state.selection.selector_transactions.size() - 1U;
  }

  // CoalescingClockSchedule advances one handled tick index monotonically, so a direct OnClock
  // witness is a lower bound for every later semantic reference from that trigger. Input-backed
  // triggers provide no such proof: an independently accepted older witness can become visible
  // after a newer one and must still be allowed to select retained evidence.
  const std::optional<TimePoint> future_reference_lower_bound =
      instance.temporal_trigger != nullptr &&
              instance.temporal_trigger->kind == TemporalRecordTriggerKind::kClock
          ? std::optional<TimePoint>{candidate.witness.reference_time}
          : std::nullopt;

  for (auto& slot : state.input_selectors) {
    if (slot.owner != instance.vertex || slot.aligned_member) {
      continue;
    }
    if (&slot == (bracketed == state.input_selectors.end() ? nullptr : &*bracketed)) {
      const std::size_t transaction_index =
          bracketed_transaction_index.value_or(state.selection.selector_transactions.size());
      if (transaction_index >= state.selection.selector_transactions.size()) {
        append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
        return SelectionPreparation::kFailed;
      }
      if (!append_frozen_input_selection(
              state, slot, state.selection.selector_transactions[transaction_index])) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
        return SelectionPreparation::kFailed;
      }
      continue;
    }
    auto appended = append_input_selection(state,
                                           slot,
                                           bracketed_transaction_index.has_value()
                                               ? semantic_reference
                                               : candidate.witness.reference_time,
                                           semantic_reference,
                                           future_reference_lower_bound);
    if (!appended) {
      if (appended.error().code == ErrorCode::kNotReady) {
        return SelectionPreparation::kNoFire;
      }
      append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
      return SelectionPreparation::kFailed;
    }
  }

  return aggregate_missing_inputs(state.selection.input_selections) ==
                 MissingAggregateDisposition::kNoFire
             ? SelectionPreparation::kNoFire
             : SelectionPreparation::kReady;
}

[[nodiscard]] expected<void, Error> RunSessionState::prepare_predicate_input_selections(
    ReadyRunState& state, const RuntimeOperatorInstance& instance,
    const PredicateTriggerPlanRecord& predicate, const FrozenRuntimeCandidate& candidate,
    TimePoint& semantic_reference) noexcept {
  std::uint32_t expected_ordinal = 0U;
  for (auto& slot : state.input_selectors) {
    if (slot.owner != instance.vertex || !slot.predicate_input ||
        slot.predicate_id != predicate.id) {
      continue;
    }
    if (slot.predicate_ordinal != expected_ordinal) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto appended =
        append_input_selection(state, slot, candidate.witness.reference_time, semantic_reference);
    if (!appended) {
      return appended;
    }
    ++expected_ordinal;
  }
  return {};
}

[[nodiscard]] expected<std::span<const InputSelection>, Error>
RunSessionState::prepare_predicate_compute_selections(PredicateComputeSelectionContext& context,
                                                      TimePoint reference_time) noexcept {
  if (context.session != this || context.ready == nullptr || context.instance == nullptr ||
      context.predicate == nullptr || context.candidate == nullptr ||
      context.semantic_reference == nullptr ||
      reference_time != context.candidate->witness.reference_time) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  ReadyRunState& state = *context.ready;
  const std::size_t selection_begin = state.selection.input_selections.size();
  for (auto& slot : state.input_selectors) {
    if (slot.owner != context.instance->vertex ||
        (slot.predicate_input && slot.predicate_id == context.predicate->id)) {
      continue;
    }
    auto appended = append_input_selection(
        state, slot, context.candidate->witness.reference_time, *context.semantic_reference);
    if (!appended) {
      return make_unexpected(std::move(appended.error()));
    }
  }
  // Compute-only selections are not visible to the predicate callable. Keep their Channel
  // payload claims unprepared until the evaluator admits the callback so a successful no-fire
  // can commit its predicate inputs while rolling this suffix back independently.
  return std::span<const InputSelection>{state.selection.input_selections}.subspan(selection_begin);
}

[[nodiscard]] expected<std::span<const InputSelection>, Error>
RunSessionState::select_predicate_compute_inputs(void* opaque, TimePoint reference_time) noexcept {
  if (opaque == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto& context = *static_cast<PredicateComputeSelectionContext*>(opaque);
  return context.session->prepare_predicate_compute_selections(context, reference_time);
}

[[nodiscard]] bool RunSessionState::finish_input_selection_prefix(
    ReadyRunState& state, std::size_t accepted_selector_count, bool aligned_accepted) noexcept {
  if (accepted_selector_count > state.selection.selector_transactions.size()) {
    return false;
  }
  bool finished = true;
  for (std::size_t index = 0U; index < state.selection.selector_transactions.size(); ++index) {
    if (!state.selection.selector_transactions[index].finish(index < accepted_selector_count)) {
      finished = false;
    }
  }
  state.selection.selector_transactions.clear();
  for (auto& transaction : state.selection.aligned_transactions) {
    if (!transaction.finish(aligned_accepted)) {
      finished = false;
    }
  }
  state.selection.aligned_transactions.clear();
  if (!finished) {
    // A failed selector commit is terminal for this dispatch. Retaining the previous mirrored
    // claim superset is fail-closed; synchronizing here could otherwise retire payloads after a
    // partially committed transaction set has already reported failure.
    return false;
  }
  return synchronize_temporal_payload_claims(state);
}

[[nodiscard]] bool RunSessionState::finish_input_selections(ReadyRunState& state,
                                                            bool accepted) noexcept {
  return finish_input_selection_prefix(
      state, accepted ? state.selection.selector_transactions.size() : 0U, accepted);
}

// Execute the exact callback transaction admitted by ClosedIslandActivationPlanRecord.
//
// This is deliberately narrower than dispatch_ready(): it accepts no predicate, deadline,
// mutable Parameter, lifecycle, or Tensor state. Both the ordinary one-dispatch-per-callback
// control and the two-callback composite call this function, so changing lane-task geometry
// cannot silently change selection, Channel preparation, callback context, or retirement.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void RunSessionState::run_closed_island_activation_transaction(
    GraphExec& exec, ReadyRunState& state, const RuntimeDirectCompositeBinding& binding,
    const DispatcherSelection& selection, FrozenRuntimeCandidate& frozen,
    bool arm_direct_continuation, bool successor_boundary,
    ClosedIslandActivationResult& result) noexcept {
  result = {};
  if (selection.receiver_index >= exec.operators().size() ||
      (selection.receiver_index != binding.producer_receiver &&
       selection.receiver_index != binding.successor_receiver) ||
      selection.receiver_index >= state.parameter_policies.size() ||
      selection.receiver_index >= state.predicates.size() ||
      selection.receiver_index >= state.deadline_bindings.size() || state.channels == nullptr) {
    append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
    static_cast<void>(cancel_ready_sources(state));
    result.error = Error{ErrorCode::kProtocolError, "RUNTIME_OPERATOR_REPLAY_MISMATCH"};
    return;
  }
  // Closed-island derivation rejects every plan with lifecycle transaction capacity. Reaching
  // an enabled controller here therefore means the frozen plan/runtime structure disagrees; it
  // is not a pending-restart observation.
  if (lifecycle_control_->enabled()) {
    append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
    static_cast<void>(cancel_ready_sources(state));
    result.error = Error{ErrorCode::kProtocolError, "RUNTIME_OPERATOR_REPLAY_MISMATCH"};
    return;
  }

  RuntimeOperatorInstance& instance = exec.operators()[selection.receiver_index];
  const bool source_member = selection.receiver_index == binding.producer_receiver;
  if (instance.execution_lane != binding.execution_lane ||
      instance.vertex != (source_member ? binding.producer : binding.successor) ||
      instance.runtime_parameter_updates_enabled || instance.lifecycle_parameter_updates_enabled ||
      state.predicates[selection.receiver_index].has_value() ||
      state.deadline_bindings[selection.receiver_index].has_value() ||
      state.parameter_policies[selection.receiver_index].deadline_term_index.has_value() ||
      std::any_of(state.parameter_policies[selection.receiver_index].selector_indices.begin(),
                  state.parameter_policies[selection.receiver_index].selector_indices.end(),
                  [&](std::uint32_t index) {
                    return index >= state.input_selectors.size() ||
                           state.input_selectors[index].derived_term_index.has_value();
                  })) {
    append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
    static_cast<void>(cancel_ready_sources(state));
    result.error = Error{ErrorCode::kProtocolError, "RUNTIME_OPERATOR_REPLAY_MISMATCH"};
    return;
  }

  ActivationParameterBinding parameters{exec.parameter_updates(), instance.static_parameter_header};
  TimePoint semantic_reference{};
  clear_input_selection_scratch(state, frozen, semantic_reference);
  const std::uint64_t activation_generation =
      state.activation_selections[selection.receiver_index].generation();
  const SelectionPreparation prepared =
      prepare_nonpredicate_input_selections(state, instance, frozen, semantic_reference);
  if (prepared == SelectionPreparation::kFailed) {
    static_cast<void>(finish_input_selections(state, false));
    append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
    static_cast<void>(cancel_ready_sources(state));
    result.error = Error{ErrorCode::kFailure, "RUNTIME_PORT_BINDING_UNAVAILABLE"};
    return;
  }

  const bool channel_activation_required =
      selection.receiver_index < state.channel_activation_owners.size() &&
      state.channel_activation_owners[selection.receiver_index] != 0U;
  bool channel_activation_prepared = false;
  if (prepared == SelectionPreparation::kReady && channel_activation_required) {
    auto channel_prepared = state.channels->prepare_activation(
        instance.vertex, activation_generation, state.selection.channel_selections);
    if (!channel_prepared) {
      static_cast<void>(finish_input_selections(state, false));
      append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
      static_cast<void>(cancel_ready_sources(state));
      result.error = std::move(channel_prepared.error());
      return;
    }
    auto tensor_access = state.channels->prepare_tensor_access(
        instance.vertex, activation_generation, instance.compute_stream);
    if (!tensor_access) {
      static_cast<void>(finish_input_selections(state, false));
      static_cast<void>(
          state.channels->retire_activation(instance.vertex, activation_generation, false));
      append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
      static_cast<void>(cancel_ready_sources(state));
      result.error = std::move(tensor_access.error());
      return;
    }
    channel_activation_prepared = true;
  }

  if (prepared == SelectionPreparation::kNoFire) {
    const bool selectors_aborted = finish_input_selections(state, false);
    const bool channel_retired =
        !channel_activation_prepared ||
        state.channels->retire_activation(instance.vertex, activation_generation, false)
            .has_value();
    const bool dispatcher_completed = state.dispatcher->complete(selection).has_value();
    const bool source_retired =
        dispatcher_completed && retire_frozen_source(state, frozen).has_value();
    parameters.release();
    if (!selectors_aborted || !channel_retired || !dispatcher_completed || !source_retired) {
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE", instance.vertex);
      static_cast<void>(cancel_ready_sources(state));
      result.error = Error{ErrorCode::kFailure, "RUNTIME_READINESS_BINDING_UNAVAILABLE"};
      return;
    }
    result.activation_retired = true;
    result.disposition = ClosedIslandActivationDisposition::kNoFire;
    return;
  }

  auto activation =
      state.activation_selections[selection.receiver_index].begin(frozen,
                                                                  state.selection.input_selections,
                                                                  state.selection.selected_samples,
                                                                  semantic_reference);
  if (!activation) {
    static_cast<void>(finish_input_selections(state, false));
    if (channel_activation_prepared) {
      static_cast<void>(
          state.channels->retire_activation(instance.vertex, activation_generation, false));
    }
    append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
    static_cast<void>(cancel_ready_sources(state));
    result.error = std::move(activation.error());
    return;
  }

  const auto retire_without_callback = [&]() noexcept {
    const bool dispatcher_completed = state.dispatcher->complete(selection).has_value();
    const bool source_retired =
        dispatcher_completed && retire_frozen_source(state, frozen).has_value();
    const bool selectors_aborted = finish_input_selections(state, false);
    const bool channel_retired =
        !channel_activation_prepared ||
        state.channels->retire_activation(instance.vertex, activation_generation, false)
            .has_value();
    const bool activation_retired =
        state.activation_selections[selection.receiver_index].retire(selection.tag).has_value();
    parameters.release();
    return dispatcher_completed && source_retired && selectors_aborted && channel_retired &&
           activation_retired;
  };

  if (successor_boundary) {
    const auto& test_control = plan_->executable_factories.test_closed_island_control;
    if (test_control != nullptr) {
      test_control->successor_boundary();
    }
  }
  if (stop_requested_.load(std::memory_order_acquire) || lifecycle_control_->enabled()) {
    const bool stop_observed = stop_requested_.load(std::memory_order_acquire);
    if (!retire_without_callback()) {
      append_diagnostic("RUNTIME_LIFECYCLE_CALLBACK_FAILED", instance.vertex);
      static_cast<void>(cancel_ready_sources(state));
      result.error = Error{ErrorCode::kFailure, "RUNTIME_LIFECYCLE_CALLBACK_FAILED"};
      return;
    }
    result.activation_retired = true;
    result.disposition = stop_observed ? ClosedIslandActivationDisposition::kStopped
                                       : ClosedIslandActivationDisposition::kRestarted;
    return;
  }

  const OperatorRef self{
      .id = instance.operator_id,
      .plan_generation = plan_->plan_generation,
      .run_generation = run_generation_,
  };
  RuntimeRateCallbackBinding rate_binding{rate_state_.get(), instance.operator_id};
  ExecutionContextState context;
  context.tag = selection.tag;
  context.tag_time = selection.ready.tag_time;
  context.activation_time = frozen.witness.reference_time;
  context.clock = exec.default_clock();
  context.cuda_stream = instance.compute_stream;
  context.activation = ActivationStatus{ActivationReason::kNominal, OnViolation::kEmitWitness};
  context.cancellation_source = &stop_requested_;
  context.pinned_params = parameters.header();
  context.termination_request_state = this;
  context.request_termination = &RunSessionState::request_termination;
  context.self = self;
  context.rate_state = &rate_binding;
  context.resolve_rate = &RuntimeRateCallbackBinding::resolve;
  ComputeParameterIngress compute_parameter_ingress{
      .coordinator = &exec.parameter_updates(),
      .self = self,
  };
  context.param_control_state = &compute_parameter_ingress;
  context.submit_param_update = &submit_compute_parameter_update;

  ComputeLaneInvocation invocation{
      .session = this,
      .instance = &instance,
      .context = &context,
      .dispatcher = state.dispatcher.get(),
      .selection = &selection,
      .parameter_updates = &exec.parameter_updates(),
      .self = self,
      .source_sequence = frozen.witness.source_sequence,
      .cuda_device = instance.compute_device,
      .phase_counters = phase_counters_,
  };
  if (arm_direct_continuation &&
      !plan_->executable_factories.test_disable_direct_continuation_elision) {
    const auto target =
        state.channels->arm_direct_continuation(instance.vertex, instance.execution_lane);
    if (target) {
      const auto source_credit =
          state.coordinator->source_credit_remaining(target->readiness_source);
      if (!source_credit) {
        static_cast<void>(retire_without_callback());
        append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE", instance.vertex);
        static_cast<void>(cancel_ready_sources(state));
        result.error = std::move(source_credit.error());
        return;
      }
      // The causal receipt may replace a readiness poll only while the successor owns its full
      // source-credit budget. Any outstanding credit could represent an earlier publication,
      // so relaxing this equality would destroy the single-pending-event authority required by
      // freeze_causal_next().
      if (*source_credit == target->readiness_capacity) {
        auto clock = exec.resolve_clock(target->reference_clock);
        if (!clock) {
          static_cast<void>(retire_without_callback());
          append_diagnostic("RUNTIME_CLOCK_BINDING_UNAVAILABLE", instance.vertex);
          static_cast<void>(cancel_ready_sources(state));
          result.error = std::move(clock.error());
          return;
        }
        invocation.channels = state.channels;
        invocation.channel_owner = instance.vertex;
        invocation.execution_lane = instance.execution_lane;
        invocation.direct_continuation_clock = std::move(*clock);
        invocation.direct_continuation_enabled = true;
      }
    }
  }

  result.callback_identity = RuntimeCallbackIdentity{
      .context = RuntimeCallbackContext::kCompute,
      .operator_id = instance.operator_id,
      .source = frozen.witness.source,
      .source_sequence = frozen.witness.source_sequence,
      .tag = frozen.candidate.tag,
      .tag_time = frozen.candidate.ready.tag_time,
      .reference_time = frozen.witness.reference_time,
      .state_generation = frozen.candidate.tag.generation,
  };
  count_phase(RuntimePhaseCounter::kComputeDispatches);
  trace_phase(RuntimePhaseTracePoint::kComputeDispatchStarted,
              selection.receiver_index,
              frozen.witness.source,
              frozen.witness.source_sequence);
  expected<void, Error> computed;
  {
    const RuntimeCallbackContextScope callback_scope{result.callback_identity};
    computed = state.compute_lane_invoke(&invocation);
  }
  trace_phase(RuntimePhaseTracePoint::kComputeDispatchReturned,
              selection.receiver_index,
              frozen.witness.source,
              frozen.witness.source_sequence);
  if (invocation.admission == ComputeAdmissionDisposition::kRunClosing) {
    if (!retire_without_callback()) {
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE",
                        instance.vertex,
                        "termination_admission_retirement");
      static_cast<void>(cancel_ready_sources(state));
      result.error = Error{ErrorCode::kFailure, "RUNTIME_READINESS_BINDING_UNAVAILABLE"};
      return;
    }
    result.activation_retired = true;
    result.disposition = ClosedIslandActivationDisposition::kStopped;
    return;
  }
  if (invocation.admission == ComputeAdmissionDisposition::kOperatorCompleted) {
    if (invocation.admission_error.code != ErrorCode::kSuccess || !retire_without_callback()) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
      static_cast<void>(cancel_ready_sources(state));
      result.error = Error{ErrorCode::kProtocolError, "RUNTIME_OPERATOR_REPLAY_MISMATCH"};
      return;
    }
    result.activation_retired = true;
    result.disposition = ClosedIslandActivationDisposition::kCompleted;
    return;
  }
  if (invocation.admission != ComputeAdmissionDisposition::kAdmitted) {
    static_cast<void>(retire_without_callback());
    const std::string_view token =
        invocation.admission == ComputeAdmissionDisposition::kCudaDeviceUnavailable
            ? std::string_view{"RUNTIME_COMPUTE_CUDA_DEVICE_UNAVAILABLE"}
            : std::string_view{"RUNTIME_OPERATOR_REPLAY_MISMATCH"};
    append_diagnostic(token, instance.vertex);
    static_cast<void>(cancel_ready_sources(state));
    result.error = Error{ErrorCode::kFailure, std::string{token}};
    return;
  }

  result.callback_attempted = true;
  result.callback_completed = computed.has_value();
  result.error = computed ? Error{ErrorCode::kSuccess} : computed.error();
  result.direct_continuation_receipt = std::move(invocation.direct_continuation_receipt);
  result.direct_continuation_clock_failed = invocation.direct_continuation_clock_failed;
  const bool activation_retired =
      invocation.completion_index == kNoCompletionIndex &&
              frozen.witness.kind != RuntimeEventKind::kInputEndOfStream &&
              frozen.witness.kind != RuntimeEventKind::kNotification
          ? retire_ordinary_activation_transaction(exec,
                                                   state,
                                                   instance,
                                                   selection,
                                                   frozen,
                                                   activation_generation,
                                                   channel_activation_prepared,
                                                   parameters,
                                                   computed.has_value(),
                                                   invocation.completed_selection)
          : retire_extended_activation_transaction(exec,
                                                   state,
                                                   instance,
                                                   selection,
                                                   frozen,
                                                   activation_generation,
                                                   channel_activation_prepared,
                                                   parameters,
                                                   computed.has_value(),
                                                   invocation.completed_selection,
                                                   invocation.completion_index);
  invocation.callback_claim.release();
  if (!activation_retired) {
    append_diagnostic("RUNTIME_LIFECYCLE_CALLBACK_FAILED", instance.vertex);
    static_cast<void>(cancel_ready_sources(state));
    result.error = Error{ErrorCode::kFailure, "RUNTIME_LIFECYCLE_CALLBACK_FAILED"};
    return;
  }
  result.activation_retired = true;
  result.activation_cut_completed = computed.has_value();
  if (!computed) {
    const RuntimeDiagnosticDetail detail{
        .present = true,
        .error = computed.error(),
        .operator_id = instance.operator_id,
        .source = frozen.witness.source,
        .source_sequence = frozen.witness.source_sequence,
        .tag = frozen.candidate.tag,
        .tag_time = frozen.candidate.ready.tag_time,
        .reference_time = frozen.witness.reference_time,
        .state_generation = frozen.candidate.tag.generation,
    };
    append_diagnostic(
        "RUNTIME_OPERATOR_CALLBACK_FAILED", instance.vertex, "operator_compute", &detail);
    static_cast<void>(cancel_ready_sources(state));
    return;
  }
  if (invocation.direct_continuation_error.code != ErrorCode::kSuccess) {
    const RuntimeDiagnosticDetail detail{
        .present = true,
        .error = invocation.direct_continuation_error,
        .operator_id = instance.operator_id,
        .source = frozen.witness.source,
        .source_sequence = frozen.witness.source_sequence,
        .tag = frozen.candidate.tag,
        .tag_time = frozen.candidate.ready.tag_time,
        .reference_time = frozen.witness.reference_time,
        .state_generation = frozen.candidate.tag.generation,
    };
    append_diagnostic(invocation.direct_continuation_clock_failed
                          ? std::string_view{"RUNTIME_CLOCK_READ_FAILED"}
                          : std::string_view{"RUNTIME_READINESS_BINDING_UNAVAILABLE"},
                      instance.vertex,
                      "direct_continuation",
                      &detail);
    static_cast<void>(cancel_ready_sources(state));
    result.error = std::move(invocation.direct_continuation_error);
    return;
  }
  result.disposition = ClosedIslandActivationDisposition::kCompleted;
}

// Direct-composite execution preserves activation, CUDA, and fallback state in one loop.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void RunSessionState::run_direct_composite_lane(
    GraphExec& exec, ReadyRunState& state, const RuntimeDirectCompositeBinding& binding,
    const DispatcherSelection& source_selection, FrozenRuntimeCandidate& source_candidate,
    RuntimeDirectCompositeCompletionReport& report) noexcept {
  count_phase(RuntimePhaseCounter::kDirectCompositeAttempts);
  trace_phase(RuntimePhaseTracePoint::kDirectCompositeStarted,
              source_selection.receiver_index,
              source_candidate.witness.source,
              source_candidate.witness.source_sequence);
  const auto& test_control = plan_->executable_factories.test_closed_island_control;
  if (test_control != nullptr) {
    test_control->transaction_started();
  }
  struct TransactionHookGuard {
    const std::shared_ptr<RuntimeClosedIslandTestControl>* control{};
    const RuntimeDirectCompositeCompletionReport* report{};
    ~TransactionHookGuard() {
      if (control != nullptr && *control != nullptr && report != nullptr) {
        (*control)->transaction_finished(*report);
      }
    }
  } const transaction_hook{.control = &test_control, .report = &report};

  const auto account = [&](std::size_t slot,
                           const ClosedIslandActivationResult& transaction) noexcept {
    if (transaction.callback_attempted) {
      report.callback_identities[slot] = transaction.callback_identity;
      report.callback_errors[slot] = transaction.error.code;
      ++report.callbacks_attempted;
      ++report.phase_deltas[static_cast<std::size_t>(
          RuntimeDirectCompositePhaseDelta::kCallbacksAttempted)];
      count_phase(RuntimePhaseCounter::kDirectCompositeCallbacksAttempted);
    }
    if (transaction.callback_completed) {
      ++report.callbacks_completed;
      ++report.phase_deltas[static_cast<std::size_t>(
          RuntimeDirectCompositePhaseDelta::kCallbacksCompleted)];
      count_phase(RuntimePhaseCounter::kDirectCompositeCallbacksCompleted);
    }
    if (transaction.activation_retired) {
      ++report.phase_deltas[static_cast<std::size_t>(
          RuntimeDirectCompositePhaseDelta::kActivationsRetired)];
      if (slot == 0U) {
        report.source_retired = true;
      } else {
        report.successor_retired = true;
      }
    }
    if (transaction.activation_cut_completed) {
      ++report.phase_deltas[static_cast<std::size_t>(
          RuntimeDirectCompositePhaseDelta::kActivationCuts)];
    }
    if (report.first_error.code == ErrorCode::kSuccess &&
        transaction.error.code != ErrorCode::kSuccess) {
      report.first_error = transaction.error;
    }
    report.stop_observed = report.stop_observed ||
                           transaction.disposition == ClosedIslandActivationDisposition::kStopped;
    report.restart_observed =
        report.restart_observed ||
        transaction.disposition == ClosedIslandActivationDisposition::kRestarted;
  };
  const auto fail = [&]() noexcept {
    report.continuation = RuntimeDirectCompositeContinuationDisposition::kFailed;
    report.resume = RuntimeDirectCompositeResumeDisposition::kTerminalFailure;
    count_phase(RuntimePhaseCounter::kDirectCompositeFailures);
  };
  const auto fallback = [&]() noexcept {
    report.continuation = RuntimeDirectCompositeContinuationDisposition::kRejected;
    report.resume = RuntimeDirectCompositeResumeDisposition::kControllerResumesScan;
    count_phase(RuntimePhaseCounter::kDirectCompositeFallbacks);
  };

  ClosedIslandActivationResult source_result;
  run_closed_island_activation_transaction(
      exec, state, binding, source_selection, source_candidate, true, false, source_result);
  account(0U, source_result);
  if (source_result.disposition == ClosedIslandActivationDisposition::kFailed) {
    fail();
    return;
  }
  if (source_result.disposition != ClosedIslandActivationDisposition::kCompleted ||
      !source_result.direct_continuation_receipt) {
    fallback();
    return;
  }

  RuntimeChannelDirectContinuationReceipt receipt = *source_result.direct_continuation_receipt;
  report.phase_deltas[static_cast<std::size_t>(
      RuntimeDirectCompositePhaseDelta::kContinuationReceipts)] = 1U;
  if (test_control != nullptr) {
    test_control->before_successor();
    if (test_control->corrupt_generation()) {
      receipt.source_generation =
          receipt.source_generation == std::numeric_limits<std::uint64_t>::max()
              ? 0U
              : receipt.source_generation + 1U;
    }
  }
  if (stop_requested_.load(std::memory_order_acquire)) {
    report.stop_observed = true;
    fallback();
    return;
  }
  if (state.channels == nullptr || !state.channels->validate_direct_continuation_receipt(receipt) ||
      receipt.destination != binding.successor ||
      receipt.destination_input != binding.successor_input ||
      receipt.source_output != binding.source_output ||
      receipt.readiness_source != binding.readiness_source ||
      state.dispatcher->active_count() != 0U) {
    fallback();
    return;
  }

  // Re-run the ordinary whole-round ingress and global ordering operation while this lane owns
  // the controller transaction. The compiler admits only a complete two-Operator process group,
  // so every poll target is this lane and no nested lane dispatch is possible. A source
  // re-notification that races the publication therefore competes with the successor under the
  // same coordinator ordering as the ordinary path instead of being bypassed by the receipt.
  state.direct_chain.continuation_receipt = receipt;
  if (!freeze_ready_candidates(exec, state, binding.execution_lane)) {
    report.first_error = Error{ErrorCode::kFailure, "RUNTIME_READINESS_BINDING_UNAVAILABLE"};
    fail();
    return;
  }
  if (state.selection.frozen_candidates.empty()) {
    append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE", binding.successor);
    static_cast<void>(cancel_ready_sources(state));
    report.first_error = Error{ErrorCode::kProtocolError};
    fail();
    return;
  }
  FrozenRuntimeCandidate& successor_frozen = state.selection.frozen_candidates.front();
  if (successor_frozen.candidate.receiver_index != binding.successor_receiver ||
      successor_frozen.witness.source != binding.readiness_source ||
      successor_frozen.witness.port != binding.successor_input) {
    fallback();
    return;
  }
  auto successor_selection = state.dispatcher->select();
  if (!successor_selection || successor_selection->receiver_index != binding.successor_receiver) {
    append_diagnostic("READY_POLICY_PLAN_INVALID", binding.successor);
    static_cast<void>(cancel_ready_sources(state));
    report.first_error =
        successor_selection ? Error{ErrorCode::kProtocolError} : successor_selection.error();
    fail();
    return;
  }
  count_phase(RuntimePhaseCounter::kCandidatesSelected);
  trace_phase(RuntimePhaseTracePoint::kDirectCompositeSuccessorEntered,
              binding.successor_receiver,
              successor_frozen.witness.source,
              successor_frozen.witness.source_sequence);

  ClosedIslandActivationResult successor_result;
  run_closed_island_activation_transaction(
      exec, state, binding, *successor_selection, successor_frozen, false, true, successor_result);
  account(1U, successor_result);
  if (successor_result.disposition == ClosedIslandActivationDisposition::kFailed) {
    fail();
    return;
  }
  if (successor_result.disposition != ClosedIslandActivationDisposition::kCompleted) {
    fallback();
    return;
  }

  report.continuation = RuntimeDirectCompositeContinuationDisposition::kAccepted;
  report.resume = RuntimeDirectCompositeResumeDisposition::kCompositeCompleted;
  report.activation_cut_completed = true;
  count_phase(RuntimePhaseCounter::kDirectCompositeCompletions);
  trace_phase(RuntimePhaseTracePoint::kDirectCompositeReturned,
              binding.successor_receiver,
              successor_frozen.witness.source,
              successor_frozen.witness.source_sequence);
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] RunTermination RunSessionState::dispatch_ready(GraphExec& exec,
                                                             ReadyRunState& state) noexcept {
  while (state.dispatcher->active_count() != 0U) {
    if (stop_requested_.load(std::memory_order_acquire)) {
      state.dispatcher->cancel_all();
      static_cast<void>(cancel_ready_sources(state));
      return RunTermination::kStopped;
    }
    if (state.partition_owner_execution) {
      // The source-tree barrier holds one already-frozen candidate before Dispatcher::select()
      // marks an activation in progress. That lets lifecycle tests place an accepted restart or
      // stop on the exact admission boundary without changing production plan identity.
      if (plan_->executable_factories.test_pre_dispatch_control != nullptr &&
          state.runner != nullptr && state.runner->operators.size() == 1U) {
        plan_->executable_factories.test_pre_dispatch_control->candidate_frozen(
            state.runner->operators.front().operator_id);
        if (stop_requested_.load(std::memory_order_acquire)) {
          state.dispatcher->cancel_all();
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kStopped;
        }
      }
      if (lifecycle_control_->has_pending()) {
        // No activation is in progress at this boundary. Preserve the frozen prefix until the
        // complete owner round retires; the controller-side restart fence cancels affected
        // candidates, while unrelated candidates resume on the same stable owner afterward.
        return RunTermination::kCompleted;
      }
    }
    if (!state.partition_owner_execution && !process_pending_restarts(exec, state)) {
      append_diagnostic(exec.lifecycle_failure_token());
      static_cast<void>(cancel_ready_sources(state));
      return RunTermination::kFailed;
    }
    if (state.dispatcher->active_count() == 0U) {
      break;
    }
    const RuntimeCallbackContextScope dispatcher_scope{RuntimeCallbackContext::kDispatcher};
    auto selected = state.dispatcher->select();
    if (!selected || selected->receiver_index >= exec.operators().size()) {
      append_diagnostic("READY_POLICY_PLAN_INVALID");
      static_cast<void>(cancel_ready_sources(state));
      return RunTermination::kFailed;
    }
    count_phase(RuntimePhaseCounter::kCandidatesSelected);
    const scheduling::ReadyCandidateView candidate = selected->ready;
    const auto frozen =
        std::find_if(state.selection.frozen_candidates.begin(),
                     state.selection.frozen_candidates.end(),
                     [&](const FrozenRuntimeCandidate& value) {
                       // Tag sequence/generation is receiver-local. Match the
                       // receiver as well so two Operators' first candidates do
                       // not alias merely because both carry Tag{1, 1, 0}.
                       return value.candidate.receiver_index == selected->receiver_index &&
                              value.candidate.tag == selected->tag;
                     });
    if (frozen == state.selection.frozen_candidates.end()) {
      append_diagnostic("READY_POLICY_PLAN_INVALID");
      static_cast<void>(cancel_ready_sources(state));
      return RunTermination::kFailed;
    }
    RuntimeOperatorInstance& instance = exec.operators()[selected->receiver_index];
    if (selected->receiver_index >= state.parameter_policies.size()) {
      append_diagnostic("RUNTIME_PARAMETER_BINDING_UNAVAILABLE", instance.vertex);
      static_cast<void>(cancel_ready_sources(state));
      return RunTermination::kFailed;
    }
    RuntimeDirectCompositeBinding* closed_island = nullptr;
    if (selected->receiver_index < state.direct_chain.composite_owner_by_receiver.size()) {
      const std::size_t owner =
          state.direct_chain.composite_owner_by_receiver[selected->receiver_index];
      if (owner != std::numeric_limits<std::size_t>::max()) {
        if (owner >= state.direct_chain.composites_by_receiver.size() ||
            !state.direct_chain.composites_by_receiver[owner].valid()) {
          append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        }
        closed_island = &state.direct_chain.composites_by_receiver[owner];
      }
    }
    if (closed_island != nullptr) {
      const bool source_member = selected->receiver_index == closed_island->producer_receiver;
      const bool composite_enabled =
          source_member && !plan_->executable_factories.test_disable_closed_island_composite &&
          !plan_->executable_factories.test_disable_direct_continuation_elision;
      RuntimeDirectCompositeCompletionReport composite_report;
      ClosedIslandActivationResult activation_result;
      struct ClosedIslandLaneTask {
        RunSessionState* session{};
        GraphExec* exec{};
        ReadyRunState* ready{};
        RuntimeDirectCompositeBinding* binding{};
        const DispatcherSelection* selection{};
        FrozenRuntimeCandidate* candidate{};
        ClosedIslandActivationResult* activation{};
        RuntimeDirectCompositeCompletionReport* composite{};
        bool source_member{};
      } task{
          .session = this,
          .exec = &exec,
          .ready = &state,
          .binding = closed_island,
          .selection = &*selected,
          .candidate = &*frozen,
          .activation = &activation_result,
          .composite = composite_enabled ? &composite_report : nullptr,
          .source_member = source_member,
      };
      const auto invoke = [](void* opaque) noexcept -> expected<void, Error> {
        auto& lane_task = *static_cast<ClosedIslandLaneTask*>(opaque);
        if (lane_task.composite != nullptr) {
          lane_task.session->run_direct_composite_lane(*lane_task.exec,
                                                       *lane_task.ready,
                                                       *lane_task.binding,
                                                       *lane_task.selection,
                                                       *lane_task.candidate,
                                                       *lane_task.composite);
        } else {
          lane_task.session->run_closed_island_activation_transaction(*lane_task.exec,
                                                                      *lane_task.ready,
                                                                      *lane_task.binding,
                                                                      *lane_task.selection,
                                                                      *lane_task.candidate,
                                                                      lane_task.source_member,
                                                                      false,
                                                                      *lane_task.activation);
        }
        return {};
      };
      const RuntimeCallbackIdentity identity{
          .context = RuntimeCallbackContext::kCompute,
          .operator_id = instance.operator_id,
          .source = frozen->witness.source,
          .source_sequence = frozen->witness.source_sequence,
          .tag = frozen->candidate.tag,
          .tag_time = frozen->candidate.ready.tag_time,
          .reference_time = frozen->witness.reference_time,
          .state_generation = frozen->candidate.tag.generation,
      };
      const ExecutionLaneTask lane_task =
            composite_enabled
                ? ExecutionLaneTask{
                      .context = &task,
                      .invoke = invoke,
                      .callback_context = RuntimeCallbackContext::kDispatcher,
                  }
                : ExecutionLaneTask{
                      .context = &task,
                      .invoke = invoke,
                      .callback_identity = identity,
                  };
      const expected<void, Error> dispatched =
          state.uses_inline_invoke() ? invoke_inline_owner_task(lane_task)
                                     : state.lanes->dispatch(instance.execution_lane, lane_task);
      if (!dispatched) {
        append_diagnostic("RUNTIME_OPERATOR_CALLBACK_FAILED", instance.vertex);
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      }
      if (composite_enabled) {
        if (composite_report.resume == RuntimeDirectCompositeResumeDisposition::kTerminalFailure) {
          return RunTermination::kFailed;
        }
        if (composite_report.stop_observed) {
          return RunTermination::kStopped;
        }
        if (composite_report.resume ==
                RuntimeDirectCompositeResumeDisposition::kCompositeCompleted &&
            !composite_report.complete()) {
          // A claimed completion whose bounded report is internally inconsistent cannot be
          // resumed safely; fail the run instead of exposing partially retired callbacks.
          append_diagnostic("RUNTIME_LIFECYCLE_CALLBACK_FAILED", instance.vertex);
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        }
        continue;
      }
      if (activation_result.disposition == ClosedIslandActivationDisposition::kFailed) {
        return RunTermination::kFailed;
      }
      if (activation_result.disposition == ClosedIslandActivationDisposition::kStopped) {
        return RunTermination::kStopped;
      }
      if (activation_result.disposition == ClosedIslandActivationDisposition::kRestarted) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      }
      if (activation_result.direct_continuation_receipt &&
          state.channels->validate_direct_continuation_receipt(
              *activation_result.direct_continuation_receipt)) {
        if (state.direct_chain.continuation_receipt) {
          append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE",
                            instance.vertex,
                            "direct_continuation_receipt_overflow");
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        }
        state.direct_chain.continuation_receipt =
            std::move(activation_result.direct_continuation_receipt);
        if (plan_->executable_factories.test_direct_continuation_control != nullptr) {
          plan_->executable_factories.test_direct_continuation_control->receipt_staged();
        }
      }
      if (!state.partition_owner_execution && lifecycle_control_->enabled() &&
          !process_pending_restarts(exec, state)) {
        append_diagnostic(exec.lifecycle_failure_token(), instance.vertex);
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      }
      const CausalFrontierDisposition frontier =
          freeze_causal_successor(exec, state, selected->receiver_index, selected->ready.tag_time);
      if (frontier == CausalFrontierDisposition::kFailed) {
        return RunTermination::kFailed;
      }
      continue;
    }
    trace_phase(RuntimePhaseTracePoint::kCandidateSelected,
                selected->receiver_index,
                frozen->witness.source,
                frozen->witness.source_sequence);
    const OperatorRef self{
        .id = instance.operator_id,
        .plan_generation = plan_->plan_generation,
        .run_generation = run_generation_,
    };
    const bool active_parameter_updates =
        instance.runtime_parameter_updates_enabled || instance.lifecycle_parameter_updates_enabled;
    ActivationParameterBinding parameter_binding{exec.parameter_updates(),
                                                 instance.static_parameter_header};
    if (active_parameter_updates) {
      if (instance.runtime_parameter_updates_enabled) {
        publish_pending_parameter_updates(exec.parameter_updates(), self);
      }
      auto current = exec.parameter_updates().pin_published(self);
      if (!current) {
        append_diagnostic("RUNTIME_PARAMETER_BINDING_UNAVAILABLE", instance.vertex);
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      }
      parameter_binding.bind(std::move(current).value());
    }
    const auto& parameter_policy = state.parameter_policies[selected->receiver_index];
    for (const std::uint32_t selector_index : parameter_policy.selector_indices) {
      if (selector_index >= state.input_selectors.size()) {
        append_diagnostic("RUNTIME_PARAMETER_BINDING_UNAVAILABLE", instance.vertex);
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      }
      auto& selector = state.input_selectors[selector_index];
      if (!selector.derived_term_index.has_value()) {
        continue;
      }
      const ParameterGenerationPin* const generation = parameter_binding.pin();
      const auto max_age =
          generation == nullptr
              ? std::optional<std::int64_t>{}
              : generation->duration_term_at(*selector.derived_term_index,
                                             ParameterDerivedTermKind::kLatestWithinMaxAge,
                                             selector.input);
      if (!max_age ||
          !selector.selector.set_latest_within_max_age(std::chrono::nanoseconds{*max_age})) {
        append_diagnostic("RUNTIME_PARAMETER_BINDING_UNAVAILABLE", instance.vertex);
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      }
    }
    TimePoint semantic_reference{};
    clear_input_selection_scratch(state, *frozen, semantic_reference);
    const std::uint64_t activation_generation =
        state.activation_selections[selected->receiver_index].generation();
    const bool channel_activation_required =
        state.channels != nullptr &&
        selected->receiver_index < state.channel_activation_owners.size() &&
        state.channel_activation_owners[selected->receiver_index] != 0U;
    auto& predicate = state.predicates[selected->receiver_index];
    const PredicateTriggerPlanRecord* const predicate_trigger = instance.predicate_trigger;
    SelectionPreparation selection_preparation = SelectionPreparation::kReady;
    if (predicate_trigger == nullptr) {
      selection_preparation =
          prepare_nonpredicate_input_selections(state, instance, *frozen, semantic_reference);
    } else {
      auto predicate_inputs = prepare_predicate_input_selections(
          state, instance, *predicate_trigger, *frozen, semantic_reference);
      if (!predicate_inputs) {
        selection_preparation = predicate_inputs.error().code == ErrorCode::kNotReady
                                    ? SelectionPreparation::kNoFire
                                    : SelectionPreparation::kFailed;
      }
    }
    const std::size_t predicate_selector_transaction_count =
        state.selection.selector_transactions.size();
    const std::size_t predicate_channel_selection_count = state.selection.channel_selections.size();
    if (selection_preparation == SelectionPreparation::kFailed) {
      static_cast<void>(finish_input_selections(state, false));
      append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
      static_cast<void>(cancel_ready_sources(state));
      return RunTermination::kFailed;
    }
    if (selection_preparation == SelectionPreparation::kReady) {
      trace_phase(RuntimePhaseTracePoint::kInputSelectionPrepared,
                  selected->receiver_index,
                  frozen->witness.source,
                  frozen->witness.source_sequence);
    }
    bool channel_activation_prepared = false;
    if (selection_preparation == SelectionPreparation::kReady && channel_activation_required) {
      auto prepared = state.channels->prepare_activation(
          instance.vertex, activation_generation, state.selection.channel_selections);
      if (!prepared) {
        static_cast<void>(finish_input_selections(state, false));
        append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      }
      auto tensor_access = state.channels->prepare_tensor_access(
          instance.vertex, activation_generation, instance.compute_stream);
      if (!tensor_access) {
        static_cast<void>(finish_input_selections(state, false));
        static_cast<void>(
            state.channels->retire_activation(instance.vertex, activation_generation, false));
        append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      }
      channel_activation_prepared = true;
    }
    if (selection_preparation == SelectionPreparation::kNoFire) {
      const bool selectors_aborted = finish_input_selections(state, false);
      const bool channel_retired =
          !channel_activation_prepared ||
          state.channels->retire_activation(instance.vertex, activation_generation, false)
              .has_value();
      const bool dispatcher_completed = state.dispatcher->complete(*selected).has_value();
      // Bracketed finish(false) retires one immutable close, so another terminal result can make
      // bounded progress without compute. An aligned kNoFire deliberately consumes no head;
      // rearming it would repeat the same all-or-nothing miss forever, so terminal generation
      // retirement discards that non-computable suffix instead.
      const bool source_finalized =
          dispatcher_completed && selectors_aborted && channel_retired &&
          retire_or_rearm_terminal_source(state, *frozen, instance.vertex, false).has_value();
      if (!dispatcher_completed || !source_finalized || !selectors_aborted || !channel_retired) {
        append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE", instance.vertex);
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      }
      continue;
    }
    ActivationStatus predicate_status{ActivationReason::kNominal, OnViolation::kEmitWitness};
    if (predicate) {
      if (predicate_trigger == nullptr) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
        static_cast<void>(finish_input_selections(state, false));
        if (channel_activation_prepared) {
          static_cast<void>(
              state.channels->retire_activation(instance.vertex, activation_generation, false));
        }
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      }
      state.selection.predicate_inputs.clear();
      const std::size_t predicate_input_count = state.selection.input_selections.size();
      for (std::size_t index = 0U; index < predicate_input_count; ++index) {
        const auto& channel_selection = state.selection.channel_selections[index];
        const InputSelection input_selection = state.selection.input_selections[index];
        const bool has_value = input_selection.state == InputState::kSelected ||
                               input_selection.state == InputState::kReused;
        const void* typed_view = nullptr;
        if (has_value) {
          if (state.channels == nullptr) {
            append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
            static_cast<void>(finish_input_selections(state, false));
            static_cast<void>(cancel_ready_sources(state));
            return RunTermination::kFailed;
          }
          auto view = state.channels->predicate_view(channel_selection.port, activation_generation);
          if (!view) {
            append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
            static_cast<void>(finish_input_selections(state, false));
            static_cast<void>(
                state.channels->retire_activation(instance.vertex, activation_generation, false));
            static_cast<void>(cancel_ready_sources(state));
            return RunTermination::kFailed;
          }
          typed_view = *view;
        }
        state.selection.predicate_inputs.push_back(PredicateFrozenInput{
            .port = channel_selection.port,
            .selection = input_selection,
            .typed_view = typed_view,
        });
      }
      const std::size_t compute_input_count = static_cast<std::size_t>(std::count_if(
          state.input_selectors.begin(), state.input_selectors.end(), [&](const auto& slot) {
            return slot.owner == instance.vertex &&
                   !(slot.predicate_input && slot.predicate_id == predicate_trigger->id);
          }));
      PredicateComputeSelectionContext compute_context{
          .session = this,
          .ready = &state,
          .instance = &instance,
          .predicate = predicate_trigger,
          .candidate = &*frozen,
          .semantic_reference = &semantic_reference,
      };
      auto decision = predicate->evaluate(
          PredicateFrozenSource{
              .source = frozen->witness.source,
              .source_sequence = frozen->witness.source_sequence,
              .driver_port = predicate_trigger->driver,
              .reference_time = frozen->witness.reference_time,
              .tag = frozen->candidate.tag,
              .tag_time = frozen->candidate.ready.tag_time,
          },
          state.selection.predicate_inputs,
          PredicateComputeSelectionRequest{
              .planned_input_count = compute_input_count,
              .state = &compute_context,
              .select = &select_predicate_compute_inputs,
          },
          state.predicate_receivers[selected->receiver_index]);
      if (!decision) {
        const RuntimeDiagnosticDetail detail{
            .present = true,
            .error = decision.error(),
            .predicate_id = predicate_trigger != nullptr ? predicate_trigger->id : std::string{},
            .operator_id = instance.operator_id,
            .source = frozen->witness.source,
            .source_sequence = frozen->witness.source_sequence,
            .tag = frozen->candidate.tag,
            .tag_time = frozen->candidate.ready.tag_time,
            .reference_time = frozen->witness.reference_time,
            .state_generation =
                predicate_trigger != nullptr ? predicate_trigger->state_generation : 0U,
        };
        append_diagnostic(
            "RUNTIME_OPERATOR_CALLBACK_FAILED", instance.vertex, "predicate_runtime", &detail);
        static_cast<void>(finish_input_selections(state, false));
        if (channel_activation_prepared) {
          static_cast<void>(
              state.channels->retire_activation(instance.vertex, activation_generation, false));
        }
        static_cast<void>(state.dispatcher->complete(*selected));
        static_cast<void>(retire_frozen_source(state, *frozen));
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      }
      if (decision->disposition == PredicateCandidateDisposition::kEvaluatorError) {
        const RuntimeDiagnosticDetail detail{
            .present = true,
            .error = decision->evaluator_error,
            .predicate_id = std::string(decision->predicate_id),
            .operator_id = decision->operator_id,
            .source = decision->source.source,
            .source_sequence = decision->source.source_sequence,
            .tag = decision->source.tag,
            .tag_time = decision->source.tag_time,
            .reference_time = decision->source.reference_time,
            .state_generation = decision->state_generation,
        };
        append_diagnostic(
            "RUNTIME_OPERATOR_CALLBACK_FAILED", instance.vertex, "predicate_evaluator", &detail);
        static_cast<void>(finish_input_selections(state, false));
        if (channel_activation_prepared) {
          static_cast<void>(
              state.channels->retire_activation(instance.vertex, activation_generation, false));
        }
        static_cast<void>(state.dispatcher->complete(*selected));
        static_cast<void>(retire_frozen_source(state, *frozen));
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      }
      if (decision->disposition == PredicateCandidateDisposition::kNoFire) {
        // A successful evaluation commits remembered predicate truth, so its exact input
        // snapshot must retire with it. Any compute-only suffix remains uncommitted: a Skip or
        // other compute-input miss suppresses the callback without consuming data that the
        // callback never observed. Missing predicate inputs and evaluator failures keep the
        // existing all-or-nothing rollback path.
        const bool predicate_inputs_accepted = decision->evaluator_called;
        const bool dispatcher_completed = state.dispatcher->complete(*selected).has_value();
        const bool source_retired =
            dispatcher_completed && retire_frozen_source(state, *frozen).has_value();
        const bool commit_predicate_inputs = predicate_inputs_accepted && source_retired;
        const bool selectors_finished =
            commit_predicate_inputs
                ? finish_input_selection_prefix(state, predicate_selector_transaction_count, false)
                : finish_input_selections(state, false);
        const bool channel_retired =
            !channel_activation_prepared ||
            state.channels
                ->retire_activation(instance.vertex, activation_generation, commit_predicate_inputs)
                .has_value();
        if (!dispatcher_completed || !source_retired || !selectors_finished || !channel_retired) {
          append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE", instance.vertex);
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        }
        continue;
      }
      if (channel_activation_required &&
          state.selection.channel_selections.size() > predicate_channel_selection_count) {
        auto prepared = state.channels->prepare_activation(
            instance.vertex,
            activation_generation,
            std::span<const RuntimeChannelActivationSelection>{state.selection.channel_selections}
                .subspan(predicate_channel_selection_count));
        if (!prepared) {
          static_cast<void>(finish_input_selections(state, false));
          if (channel_activation_prepared) {
            static_cast<void>(
                state.channels->retire_activation(instance.vertex, activation_generation, false));
          }
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        }
        auto tensor_access = state.channels->prepare_tensor_access(
            instance.vertex, activation_generation, instance.compute_stream);
        if (!tensor_access) {
          static_cast<void>(finish_input_selections(state, false));
          static_cast<void>(
              state.channels->retire_activation(instance.vertex, activation_generation, false));
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        }
      }
      if (decision->disposition == PredicateCandidateDisposition::kDegradedCompute) {
        predicate_status =
            ActivationStatus{ActivationReason::kInvariantViolated, OnViolation::kSafeDefault};
      }
    }
    auto activation = state.activation_selections[selected->receiver_index].begin(
        *frozen,
        state.selection.input_selections,
        state.selection.selected_samples,
        semantic_reference);
    if (!activation) {
      static_cast<void>(finish_input_selections(state, false));
      if (channel_activation_prepared) {
        static_cast<void>(
            state.channels->retire_activation(instance.vertex, activation_generation, false));
      }
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
      static_cast<void>(cancel_ready_sources(state));
      return RunTermination::kFailed;
    }
    trace_phase(RuntimePhaseTracePoint::kActivationPrepared,
                selected->receiver_index,
                frozen->witness.source,
                frozen->witness.source_sequence);
    const auto abort_selection = [&]() noexcept {
      const bool selectors_aborted = finish_input_selections(state, false);
      const bool channel_retired =
          !channel_activation_prepared ||
          state.channels->retire_activation(instance.vertex, activation_generation, false)
              .has_value();
      const bool activation_retired =
          state.activation_selections[selected->receiver_index].retire(selected->tag).has_value();
      return selectors_aborted && channel_retired && activation_retired;
    };
    RuntimeRateCallbackBinding rate_binding{rate_state_.get(), instance.operator_id};
    ExecutionContextState context;
    context.tag = selected->tag;
    context.tag_time = candidate.tag_time;
    context.activation_time = frozen->witness.reference_time;
    context.clock = exec.default_clock();
    context.cuda_stream = instance.compute_stream;
    context.cancellation_source = &stop_requested_;
    context.pinned_params = parameter_binding.header();
    context.activation = predicate_status;
    context.termination_request_state = this;
    context.request_termination = &RunSessionState::request_termination;
    context.self = self;
    context.rate_state = &rate_binding;
    context.resolve_rate = &RuntimeRateCallbackBinding::resolve;
    ComputeParameterIngress compute_parameter_ingress{
        .coordinator = &exec.parameter_updates(),
        .self = self,
    };
    context.param_control_state = &compute_parameter_ingress;
    context.submit_param_update = &submit_compute_parameter_update;

    const auto& planned_deadline = state.deadline_bindings[selected->receiver_index];
    // EOS never carries a sample coordinate, even when another input resolves from ReuseLast.
    // It may satisfy only a compiled closure-origin deadline; sample- and predicate-driver
    // deadlines remain outside this ordered control transition.
    const bool end_of_stream_without_deadline_coordinate =
        frozen->witness.kind == RuntimeEventKind::kInputEndOfStream &&
        planned_deadline.has_value() &&
        (planned_deadline->deadline->source_coordinate ==
             DeadlineSourceCoordinateKind::kTriggerSample ||
         planned_deadline->deadline->source_coordinate ==
             DeadlineSourceCoordinateKind::kPredicateDriver);
    const RuntimeDeadlineBindingRecord* const deadline =
        planned_deadline.has_value() && !end_of_stream_without_deadline_coordinate
            ? &planned_deadline.value()
            : nullptr;
    std::int64_t deadline_budget_ns{};
    std::optional<ClockRef> deadline_clock;
    if (deadline != nullptr) {
      deadline_budget_ns = deadline->deadline->budget_ns;
      if (parameter_policy.deadline_term_index.has_value()) {
        const ParameterGenerationPin* const generation = parameter_binding.pin();
        const auto pinned_budget =
            generation == nullptr
                ? std::optional<std::int64_t>{}
                : generation->duration_term_at(*parameter_policy.deadline_term_index,
                                               ParameterDerivedTermKind::kContractDeadline);
        if (!pinned_budget) {
          static_cast<void>(abort_selection());
          append_diagnostic("RUNTIME_PARAMETER_BINDING_UNAVAILABLE", instance.vertex);
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        }
        deadline_budget_ns = *pinned_budget;
      }
      auto dispatch_clock = exec.resolve_clock(deadline->reference_clock);
      if (!dispatch_clock) {
        static_cast<void>(abort_selection());
        append_diagnostic("RUNTIME_CLOCK_BINDING_UNAVAILABLE", instance.vertex);
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      }
      deadline_clock.emplace(std::move(*dispatch_clock));
    }
    struct ComputeRestartBinding {
      LifecycleControlState* control{};
      OperatorRef requester{};
    } restart_binding{
        .control = lifecycle_control_.get(),
        .requester = context.self,
    };
    if (lifecycle_control_->enabled()) {
      context.restart_state = &restart_binding;
      context.request_restart =
          [](void* state, OperatorRef target, LifecycleRestartOptions options) noexcept {
            auto& binding = *static_cast<ComputeRestartBinding*>(state);
            return binding.control->submit(target.id == binding.requester.id
                                               ? LifecycleRuntimeRequestOrigin::kComputeSelf
                                               : LifecycleRuntimeRequestOrigin::kComputeCrossTarget,
                                           binding.requester,
                                           target,
                                           options,
                                           nullptr);
          };
    }

    ComputeLaneInvocation invocation{
        .session = this,
        .instance = &instance,
        .context = &context,
        .dispatcher = state.dispatcher.get(),
        .selection = &*selected,
        .deadline = deadline,
        .deadline_clock = deadline_clock.value_or(ClockRef{}),
        .source_decision_time = frozen->witness.source_decision_time,
        .reference_time = frozen->witness.reference_time,
        .deadline_budget_ns = deadline_budget_ns,
        .parameter_updates = &exec.parameter_updates(),
        .self = self,
        .parameter_generation = parameter_binding.pin(),
        .source_sequence = frozen->witness.source_sequence,
        .cuda_device = instance.compute_device,
        .phase_counters = phase_counters_,
    };
    if (state.channels != nullptr && !state.direct_chain.continuation_receipt &&
        !lifecycle_control_->enabled() &&
        !plan_->executable_factories.test_disable_direct_continuation_elision) {
      const auto target =
          state.channels->arm_direct_continuation(instance.vertex, instance.execution_lane);
      if (target) {
        const auto source_credit =
            state.coordinator->source_credit_remaining(target->readiness_source);
        if (!source_credit) {
          static_cast<void>(abort_selection());
          append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE", instance.vertex);
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        }
        // Full source credit proves that this receipt is the successor's sole pending
        // publication. A partially occupied binding must retain the ordinary globally ordered
        // readiness scan.
        if (*source_credit == target->readiness_capacity) {
          auto clock = exec.resolve_clock(target->reference_clock);
          if (!clock) {
            static_cast<void>(abort_selection());
            append_diagnostic("RUNTIME_CLOCK_BINDING_UNAVAILABLE", instance.vertex);
            static_cast<void>(cancel_ready_sources(state));
            return RunTermination::kFailed;
          }
          invocation.channels = state.channels;
          invocation.channel_owner = instance.vertex;
          invocation.execution_lane = instance.execution_lane;
          invocation.direct_continuation_clock = std::move(*clock);
          invocation.direct_continuation_enabled = true;
        }
      }
    }
    count_phase(RuntimePhaseCounter::kComputeDispatches);
    trace_phase(RuntimePhaseTracePoint::kComputeDispatchStarted,
                selected->receiver_index,
                frozen->witness.source,
                frozen->witness.source_sequence);
    const RuntimeCallbackIdentity source_identity{
        .context = RuntimeCallbackContext::kCompute,
        .operator_id = instance.operator_id,
        .source = frozen->witness.source,
        .source_sequence = frozen->witness.source_sequence,
        .tag = frozen->candidate.tag,
        .tag_time = frozen->candidate.ready.tag_time,
        .reference_time = frozen->witness.reference_time,
        .state_generation = frozen->candidate.tag.generation,
    };
    const ExecutionLaneTask lane_task{
        .context = &invocation,
        .invoke = state.compute_lane_invoke,
        .callback_identity = source_identity,
    };
    const expected<void, Error> computed =
        state.uses_inline_invoke() ? invoke_inline_owner_task(lane_task)
                                   : state.lanes->dispatch(instance.execution_lane, lane_task);
    trace_phase(RuntimePhaseTracePoint::kComputeDispatchReturned,
                selected->receiver_index,
                source_identity.source,
                source_identity.source_sequence);
    if (invocation.deadline_gate &&
        invocation.deadline_gate->disposition != DeadlineGateDisposition::kInvokeNominal) {
      const RuntimeDiagnosticDetail detail{
          .present = true,
          .operator_id = instance.operator_id,
          .source = frozen->witness.source,
          .source_sequence = frozen->witness.source_sequence,
          .tag = frozen->candidate.tag,
          .tag_time = frozen->candidate.ready.tag_time,
          .reference_time = frozen->witness.reference_time,
          .state_generation = frozen->candidate.tag.generation,
          .deadline_miss =
              RuntimeDeadlineMissDiagnostic{
                  .present = true,
                  .source_coordinate = deadline->deadline->source_coordinate,
                  .source_decision_time = frozen->witness.source_decision_time,
                  .reference_time = frozen->witness.reference_time,
                  .budget_ns = deadline_budget_ns,
                  .latest_start = invocation.deadline_gate->latest_start,
                  .dispatch_start = invocation.deadline_gate->dispatch_start,
                  .checked_lateness_ns = invocation.deadline_gate->checked_lateness_ns,
                  .action = deadline->deadline->action,
                  .callback_count = deadline->deadline->callback_count_if_late,
              },
      };
      append_diagnostic("DEADLINE_MISS",
                        instance.vertex,
                        "latest_start",
                        &detail,
                        {},
                        {},
                        ::holoscan::DiagnosticSeverity::kWarning);
    }
    switch (invocation.admission) {
      case ComputeAdmissionDisposition::kDeadlineReadFailed:
        static_cast<void>(abort_selection());
        append_diagnostic("RUNTIME_CLOCK_READ_FAILED", instance.vertex);
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      case ComputeAdmissionDisposition::kDeadlineEvaluationFailed: {
        static_cast<void>(abort_selection());
        const RuntimeDiagnosticDetail detail{
            .present = true,
            .error = std::move(invocation.admission_error),
            .operator_id = instance.operator_id,
            .source = frozen->witness.source,
            .source_sequence = frozen->witness.source_sequence,
            .tag = frozen->candidate.tag,
            .tag_time = frozen->candidate.ready.tag_time,
            .reference_time = frozen->witness.reference_time,
            .state_generation = frozen->candidate.tag.generation,
        };
        append_diagnostic(
            "RUNTIME_DEADLINE_EVALUATION_FAILED", instance.vertex, "latest_start", &detail);
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      }
      case ComputeAdmissionDisposition::kDeadlineFailRun:
        static_cast<void>(abort_selection());
        static_cast<void>(state.dispatcher->complete(*selected));
        static_cast<void>(retire_frozen_source(state, *frozen));
        append_diagnostic("RUNTIME_DEADLINE_FAIL_RUN", instance.vertex, "latest_start");
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      case ComputeAdmissionDisposition::kParameterObservationFailed:
        static_cast<void>(abort_selection());
        append_diagnostic("RUNTIME_PARAMETER_BINDING_UNAVAILABLE", instance.vertex);
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      case ComputeAdmissionDisposition::kCudaDeviceUnavailable:
        static_cast<void>(abort_selection());
        append_diagnostic("RUNTIME_COMPUTE_CUDA_DEVICE_UNAVAILABLE", instance.vertex);
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      case ComputeAdmissionDisposition::kRunClosing: {
        const bool activation_aborted = abort_selection();
        const bool dispatcher_completed = state.dispatcher->complete(*selected).has_value();
        const bool source_retired =
            dispatcher_completed && retire_frozen_source(state, *frozen).has_value();
        parameter_binding.release();
        if (!activation_aborted || !dispatcher_completed || !source_retired) {
          append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE",
                            instance.vertex,
                            "termination_admission_retirement");
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        }
        return RunTermination::kStopped;
      }
      case ComputeAdmissionDisposition::kOperatorCompleted: {
        const bool activation_aborted = abort_selection();
        const bool dispatcher_completed = state.dispatcher->complete(*selected).has_value();
        const bool source_retired =
            dispatcher_completed && retire_frozen_source(state, *frozen).has_value();
        parameter_binding.release();
        if (invocation.admission_error.code != ErrorCode::kSuccess || !activation_aborted ||
            !dispatcher_completed || !source_retired) {
          append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        }
        return RunTermination::kCompleted;
      }
      case ComputeAdmissionDisposition::kPending:
      case ComputeAdmissionDisposition::kAdmitted:
        break;
    }
    const bool activation_retired =
        invocation.completion_index == kNoCompletionIndex &&
                frozen->witness.kind != RuntimeEventKind::kInputEndOfStream &&
                frozen->witness.kind != RuntimeEventKind::kNotification
            ? retire_ordinary_activation_transaction(exec,
                                                     state,
                                                     instance,
                                                     *selected,
                                                     *frozen,
                                                     activation_generation,
                                                     channel_activation_prepared,
                                                     parameter_binding,
                                                     computed.has_value(),
                                                     invocation.completed_selection)
            : retire_extended_activation_transaction(exec,
                                                     state,
                                                     instance,
                                                     *selected,
                                                     *frozen,
                                                     activation_generation,
                                                     channel_activation_prepared,
                                                     parameter_binding,
                                                     computed.has_value(),
                                                     invocation.completed_selection,
                                                     invocation.completion_index);
    invocation.callback_claim.release();
    if (!computed) {
      const RuntimeDiagnosticDetail detail{
          .present = true,
          .error = computed.error(),
          .operator_id = instance.operator_id,
          .source = frozen->witness.source,
          .source_sequence = frozen->witness.source_sequence,
          .tag = frozen->candidate.tag,
          .tag_time = frozen->candidate.ready.tag_time,
          .reference_time = frozen->witness.reference_time,
          .state_generation = frozen->candidate.tag.generation,
      };
      append_diagnostic(
          "RUNTIME_OPERATOR_CALLBACK_FAILED", instance.vertex, "operator_compute", &detail);
      static_cast<void>(cancel_ready_sources(state));
      return RunTermination::kFailed;
    }
    if (!activation_retired) {
      append_diagnostic("RUNTIME_LIFECYCLE_CALLBACK_FAILED", instance.vertex);
      static_cast<void>(cancel_ready_sources(state));
      return RunTermination::kFailed;
    }
    if (invocation.direct_continuation_error.code != ErrorCode::kSuccess) {
      const RuntimeDiagnosticDetail detail{
          .present = true,
          .error = std::move(invocation.direct_continuation_error),
          .operator_id = instance.operator_id,
          .source = frozen->witness.source,
          .source_sequence = frozen->witness.source_sequence,
          .tag = frozen->candidate.tag,
          .tag_time = frozen->candidate.ready.tag_time,
          .reference_time = frozen->witness.reference_time,
          .state_generation = frozen->candidate.tag.generation,
      };
      append_diagnostic(invocation.direct_continuation_clock_failed
                            ? std::string_view{"RUNTIME_CLOCK_READ_FAILED"}
                            : std::string_view{"RUNTIME_READINESS_BINDING_UNAVAILABLE"},
                        instance.vertex,
                        "direct_continuation",
                        &detail);
      static_cast<void>(cancel_ready_sources(state));
      return RunTermination::kFailed;
    }
    if (!state.partition_owner_execution && lifecycle_control_->enabled() &&
        !process_pending_restarts(exec, state)) {
      append_diagnostic(exec.lifecycle_failure_token(), instance.vertex);
      static_cast<void>(cancel_ready_sources(state));
      return RunTermination::kFailed;
    }
    if (invocation.direct_continuation_receipt &&
        state.channels->validate_direct_continuation_receipt(
            *invocation.direct_continuation_receipt)) {
      if (state.direct_chain.continuation_receipt) {
        append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE",
                          instance.vertex,
                          "direct_continuation_receipt_overflow");
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      }
      state.direct_chain.continuation_receipt = std::move(invocation.direct_continuation_receipt);
      if (plan_->executable_factories.test_direct_continuation_control != nullptr) {
        plan_->executable_factories.test_direct_continuation_control->receipt_staged();
      }
    }
    const CausalFrontierDisposition frontier =
        freeze_causal_successor(exec, state, selected->receiver_index, selected->ready.tag_time);
    if (frontier == CausalFrontierDisposition::kFailed) {
      return RunTermination::kFailed;
    }
  }
  return RunTermination::kCompleted;
}

template <bool TraceCallback, bool ObserveFinalEntry>
[[nodiscard]] expected<void, Error> RunSessionState::invoke_compute_lane_with_completion(
    void* state) noexcept {
  auto& task = *static_cast<ComputeLaneInvocation*>(state);
  const auto completion =
      std::lower_bound(task.session->operator_completion_progress_.begin(),
                       task.session->operator_completion_progress_.end(),
                       task.instance->operator_id,
                       [](const OperatorCompletionProgressRecord& record,
                          const OperatorInstanceId owner) { return record.owner.id < owner; });
  if (completion == task.session->operator_completion_progress_.end() ||
      completion->owner.id != task.instance->operator_id) {
    return invoke_compute_lane<TraceCallback>(state);
  }
  if constexpr (ObserveFinalEntry) {
    const auto& final_entry_control =
        task.session->plan_->executable_factories.test_operator_completion_final_entry_control;
    final_entry_control->checked();
  }
  const std::size_t index = static_cast<std::size_t>(
      std::distance(task.session->operator_completion_progress_.begin(), completion));
  task.completion_index = index;
  if (index >= task.session->operator_completion_phases_.size()) {
    task.admission = ComputeAdmissionDisposition::kOperatorCompleted;
    task.admission_error = Error{ErrorCode::kProtocolError, "RUNTIME_OPERATOR_REPLAY_MISMATCH"};
    return {};
  }
  const OperatorCompletionProgressRecord& progress =
      task.session->operator_completion_progress_[index];
  if (task.session->operator_completion_phases_[index] != OperatorCompletionPhase::kOpen ||
      progress.successful_compute_calls >= progress.configured_limit) {
    if constexpr (ObserveFinalEntry) {
      const auto& final_entry_control =
          task.session->plan_->executable_factories.test_operator_completion_final_entry_control;
      final_entry_control->rejected();
    }
    task.admission = ComputeAdmissionDisposition::kOperatorCompleted;
    return {};
  }
  return invoke_compute_lane<TraceCallback>(state);
}

[[nodiscard]] RunSessionState::ComputeLaneInvoke
RunSessionState::select_compute_lane_with_completion_invoke(
    const bool trace_callback, const bool observe_final_entry) noexcept {
  if (trace_callback) {
    return observe_final_entry ? &invoke_compute_lane_with_completion<true, true>
                               : &invoke_compute_lane_with_completion<true, false>;
  }
  return observe_final_entry ? &invoke_compute_lane_with_completion<false, true>
                             : &invoke_compute_lane_with_completion<false, false>;
}

[[nodiscard]] bool RunSessionState::retire_ordinary_activation_transaction(
    GraphExec& exec, ReadyRunState& state, RuntimeOperatorInstance& instance,
    const DispatcherSelection& selection, FrozenRuntimeCandidate& frozen,
    std::uint64_t activation_generation, bool channel_activation_prepared,
    ActivationParameterBinding& parameter_binding, bool callback_succeeded,
    bool completed_selection) noexcept {
  const bool retired_source =
      completed_selection && retire_nonterminal_frozen_source(state, frozen).has_value();
  const bool finished_selectors = finish_input_selections(state, callback_succeeded);
  const bool retired_channel =
      !channel_activation_prepared ||
      state.channels->retire_activation(instance.vertex, activation_generation, callback_succeeded)
          .has_value();
  const bool retired_activation =
      state.activation_selections[selection.receiver_index].retire(selection.tag).has_value();
  parameter_binding.release();
  if (callback_succeeded && instance.runtime_parameter_updates_enabled) {
    static_cast<void>(exec.parameter_updates().drain_compute_ingress());
    if (!stop_requested_.load(std::memory_order_acquire)) {
      publish_pending_parameter_updates(exec.parameter_updates(),
                                        OperatorRef{
                                            .id = instance.operator_id,
                                            .plan_generation = plan_->plan_generation,
                                            .run_generation = run_generation_,
                                        });
    }
  }
  if (!completed_selection || !retired_source || !finished_selectors || !retired_channel ||
      !retired_activation) {
    return false;
  }
  if (!callback_succeeded) {
    return true;
  }
  trace_phase(RuntimePhaseTracePoint::kActivationRetirementCompleted,
              selection.receiver_index,
              frozen.witness.source,
              frozen.witness.source_sequence);
  if (state.channels != nullptr) {
    auto cut = state.channels->flush_process_local_activation_cut();
    if (!cut) {
      const RuntimeDiagnosticDetail detail{
          .present = true,
          .error = cut.error(),
          .operator_id = instance.operator_id,
          .source = frozen.witness.source,
          .source_sequence = frozen.witness.source_sequence,
          .tag = selection.tag,
          .tag_time = selection.ready.tag_time,
          .reference_time = frozen.witness.reference_time,
          .state_generation = selection.tag.generation,
      };
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE",
                        instance.vertex,
                        "post_compute_channel_cut",
                        &detail);
      return false;
    }
  }
  trace_phase(RuntimePhaseTracePoint::kActivationCutCompleted,
              selection.receiver_index,
              frozen.witness.source,
              frozen.witness.source_sequence);
  count_phase(RuntimePhaseCounter::kActivationsCompleted);
  return true;
}

// Extended retirement is one exact-once commit across Channel, temporal, parameter, and EOS state.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool RunSessionState::retire_extended_activation_transaction(
    GraphExec& exec, ReadyRunState& state, RuntimeOperatorInstance& instance,
    const DispatcherSelection& selection, FrozenRuntimeCandidate& frozen,
    std::uint64_t activation_generation, bool channel_activation_prepared,
    ActivationParameterBinding& parameter_binding, bool callback_succeeded,
    bool completed_selection, std::size_t completion_index) noexcept {
  const bool defer_terminal_source =
      callback_succeeded && frozen.witness.kind == RuntimeEventKind::kInputEndOfStream;
  const bool defer_notification_source = frozen.witness.kind == RuntimeEventKind::kNotification;
  const bool source_retired_before_selection =
      defer_terminal_source || defer_notification_source ||
      (completed_selection && retire_frozen_source(state, frozen).has_value());
  const bool finished_selectors = finish_input_selections(state, callback_succeeded);
  const bool terminal_source_finalized = [&]() noexcept {
    if (!defer_terminal_source) {
      return source_retired_before_selection;
    }
    if (!completed_selection) {
      return false;
    }
    return finished_selectors
               ? retire_or_rearm_terminal_source(state, frozen, instance.vertex, true).has_value()
               : retire_frozen_source(state, frozen).has_value();
  }();
  const bool retired_channel =
      !channel_activation_prepared ||
      state.channels->retire_activation(instance.vertex, activation_generation, callback_succeeded)
          .has_value();
  const bool retired_activation =
      state.activation_selections[selection.receiver_index].retire(selection.tag).has_value();
  parameter_binding.release();
  if (callback_succeeded && instance.runtime_parameter_updates_enabled) {
    static_cast<void>(exec.parameter_updates().drain_compute_ingress());
    if (!stop_requested_.load(std::memory_order_acquire)) {
      publish_pending_parameter_updates(exec.parameter_updates(),
                                        OperatorRef{
                                            .id = instance.operator_id,
                                            .plan_generation = plan_->plan_generation,
                                            .run_generation = run_generation_,
                                        });
    }
  }
  const bool source_finalized = defer_notification_source
                                    ? completed_selection && finished_selectors &&
                                          retired_channel && retired_activation &&
                                          retire_frozen_source(state, frozen).has_value()
                                    : terminal_source_finalized;
  if (!completed_selection || !source_finalized || !finished_selectors || !retired_channel ||
      !retired_activation) {
    return false;
  }
  if (!callback_succeeded) {
    return true;
  }
  trace_phase(RuntimePhaseTracePoint::kActivationRetirementCompleted,
              selection.receiver_index,
              frozen.witness.source,
              frozen.witness.source_sequence);
  if (state.channels != nullptr) {
    auto cut = state.channels->flush_process_local_activation_cut();
    if (!cut) {
      const RuntimeDiagnosticDetail detail{
          .present = true,
          .error = cut.error(),
          .operator_id = instance.operator_id,
          .source = frozen.witness.source,
          .source_sequence = frozen.witness.source_sequence,
          // A terminal successor can already own `frozen.candidate`. Preserve the completed
          // activation identity copied by Dispatcher::select() for this retirement diagnostic.
          .tag = selection.tag,
          .tag_time = selection.ready.tag_time,
          .reference_time = frozen.witness.reference_time,
          .state_generation = selection.tag.generation,
      };
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE",
                        instance.vertex,
                        "post_compute_channel_cut",
                        &detail);
      return false;
    }
  }
  trace_phase(RuntimePhaseTracePoint::kActivationCutCompleted,
              selection.receiver_index,
              frozen.witness.source,
              frozen.witness.source_sequence);
  if (completion_index != kNoCompletionIndex) {
    OperatorCompletionProgressRecord& progress = operator_completion_progress_[completion_index];
    OperatorCompletionPhase& phase = operator_completion_phases_[completion_index];
    if (phase != OperatorCompletionPhase::kOpen ||
        progress.successful_compute_calls >= progress.configured_limit) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
      return false;
    }
    ++progress.successful_compute_calls;
    if (progress.successful_compute_calls == progress.configured_limit) {
      progress.limit_reached = true;
      phase = OperatorCompletionPhase::kClosing;
      if (!cancel_completion_receiver(state, selection.receiver_index)) {
        return false;
      }
      state.active_completion_receivers.push_back(ReadyRunState::CompletionReceiverState{
          .receiver = selection.receiver_index,
          .completion_index = completion_index,
      });
    }
  }
  count_phase(RuntimePhaseCounter::kActivationsCompleted);
  return true;
}

[[nodiscard]] expected<void, Error> RunSessionState::release_terminal_selector_history(
    ReadyRunState& state, VertexId owner) noexcept {
  // A closed physical binding proves the sole EOS authority has retired. No bracketed successor
  // may remain at that point; aligned kNoFire is the deliberate exception because it consumes no
  // queue head and terminal retirement discards that non-computable suffix.
  if (terminal_successor_ready(state, owner, false)) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  for (auto& slot : state.input_selectors) {
    if (slot.owner != owner) {
      continue;
    }
    if (!slot.end_of_stream_received) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (slot.aligned_member ||
        slot.selector.plan().selector == InputSelectorKind::kBracketedWindow) {
      continue;
    }
    auto released = slot.selector.release_terminal_history();
    if (!released) {
      return make_unexpected(std::move(released.error()));
    }
  }
  for (auto& aligned : state.aligned_selectors) {
    if (aligned.owner != owner) {
      continue;
    }
    auto released = aligned.selector.release_terminal_history();
    if (!released) {
      return make_unexpected(std::move(released.error()));
    }
  }
  for (auto& slot : state.input_selectors) {
    if (slot.owner != owner) {
      continue;
    }
    if (state.channels == nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    auto released = state.channels->retire_terminal_selector_history_samples(
        owner, slot.input, slot.channel_payload_claims);
    if (!released) {
      return make_unexpected(std::move(released).error());
    }
    slot.channel_payload_claims.clear();
  }
  return {};
}

}  // namespace holoscan::detail
