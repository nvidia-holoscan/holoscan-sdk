/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "dispatcher.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <limits>
#include <memory>
#include <new>
#include <tuple>
#include <utility>
#include <vector>

#include "endpoint_clock_health.hpp"
#include "runtime_phase_counters.hpp"
#include "scheduling/ready_policy_catalog.hpp"

namespace holoscan::detail {

RuntimeInputSelectionState::ReuseReplacementTransaction::ReuseReplacementTransaction(
    ReuseReplacementTransaction&& other) noexcept
    : owner_(other.owner_), retained_(other.retained_) {
  other.owner_ = nullptr;
}

RuntimeInputSelectionState::ReuseReplacementTransaction::~ReuseReplacementTransaction() {
  if (owner_ != nullptr) {
    assert(owner_->transaction_active_);
    owner_->retained_ = retained_;
    owner_->transaction_active_ = false;
  }
}

void RuntimeInputSelectionState::ReuseReplacementTransaction::commit() noexcept {
  assert(owner_ != nullptr);
  assert(owner_->transaction_active_);
  owner_->transaction_active_ = false;
  owner_ = nullptr;
}

RuntimeInputSelectionState::CacheReplacementTransaction::CacheReplacementTransaction(
    CacheReplacementTransaction&& other) noexcept
    : owner_(other.owner_), retained_(other.retained_), index_(other.index_) {
  other.owner_ = nullptr;
}

RuntimeInputSelectionState::CacheReplacementTransaction::~CacheReplacementTransaction() {
  if (owner_ != nullptr) {
    assert(owner_->transaction_active_);
    owner_->restore_cache_replacement(retained_, index_);
    owner_->transaction_active_ = false;
  }
}

void RuntimeInputSelectionState::CacheReplacementTransaction::commit() noexcept {
  assert(owner_ != nullptr);
  assert(owner_->transaction_active_);
  owner_->transaction_active_ = false;
  owner_ = nullptr;
}

namespace {

[[nodiscard]] constexpr std::uint8_t candidate_kind_order(RuntimeEventKind kind) noexcept {
  // The class order is part of local deterministic candidate formation. It is
  // independent of adapter polling order; plan source order and source
  // sequence complete the equal-time ordering key.
  switch (kind) {
    case RuntimeEventKind::kInputArrival:
    case RuntimeEventKind::kInputEndOfStream:
      return 0U;
    case RuntimeEventKind::kWindowTimeout:
      return 1U;
    case RuntimeEventKind::kClockTick:
      return 2U;
    case RuntimeEventKind::kReadable:
      return 3U;
    case RuntimeEventKind::kNotification:
      return 4U;
    case RuntimeEventKind::kDoorbell:
      return 5U;
    case RuntimeEventKind::kSpin:
      return 6U;
    case RuntimeEventKind::kFiniteSourceComplete:
    case RuntimeEventKind::kSourceError:
      return std::numeric_limits<std::uint8_t>::max();
  }
  return std::numeric_limits<std::uint8_t>::max();
}

[[nodiscard]] constexpr bool candidate_kind(RuntimeEventKind kind) noexcept {
  return kind != RuntimeEventKind::kFiniteSourceComplete && kind != RuntimeEventKind::kSourceError;
}

[[nodiscard]] constexpr bool valid_clock_health_disposition(
    EndpointClockHealthDisposition disposition) noexcept {
  return disposition == EndpointClockHealthDisposition::kNotApplicable ||
         disposition == EndpointClockHealthDisposition::kSkip ||
         disposition == EndpointClockHealthDisposition::kDegraded ||
         disposition == EndpointClockHealthDisposition::kFailRun;
}

[[nodiscard]] constexpr bool valid_max_start_delay_source(
    MaxStartDelaySourceCoordinateKind source) noexcept {
  switch (source) {
    case MaxStartDelaySourceCoordinateKind::kTriggerSample:
    case MaxStartDelaySourceCoordinateKind::kPredicateDriver:
    case MaxStartDelaySourceCoordinateKind::kScheduledTick:
    case MaxStartDelaySourceCoordinateKind::kAlignedHighOrClosureProof:
    case MaxStartDelaySourceCoordinateKind::kWindowClose:
    case MaxStartDelaySourceCoordinateKind::kReadinessWitness:
      return true;
  }
  return false;
}

[[nodiscard]] constexpr bool valid_max_start_delay_action(OnViolation action) noexcept {
  switch (action) {
    case OnViolation::kEmitWitness:
    case OnViolation::kSafeDefault:
    case OnViolation::kFailRun:
      return true;
  }
  return false;
}

[[nodiscard]] std::optional<MaxStartDelaySourceCoordinateKind> expected_max_start_delay_source(
    const TemporalTriggerPlanRecord& trigger, const PredicateTriggerPlanRecord* predicate,
    PlanGenerationId plan_generation) noexcept {
  if (trigger.owner.plan_generation != plan_generation || trigger.owner.run_generation != 0U) {
    return std::nullopt;
  }
  switch (trigger.kind) {
    case TemporalRecordTriggerKind::kAnyInput:
    case TemporalRecordTriggerKind::kEach:
      return MaxStartDelaySourceCoordinateKind::kTriggerSample;
    case TemporalRecordTriggerKind::kClock:
      return MaxStartDelaySourceCoordinateKind::kScheduledTick;
    case TemporalRecordTriggerKind::kReadable:
    case TemporalRecordTriggerKind::kNotified:
    case TemporalRecordTriggerKind::kDoorbell:
    case TemporalRecordTriggerKind::kSpin:
      return MaxStartDelaySourceCoordinateKind::kReadinessWitness;
    case TemporalRecordTriggerKind::kAligned:
      return MaxStartDelaySourceCoordinateKind::kAlignedHighOrClosureProof;
    case TemporalRecordTriggerKind::kWindow:
      return MaxStartDelaySourceCoordinateKind::kWindowClose;
    case TemporalRecordTriggerKind::kPredicate:
      if (predicate == nullptr || predicate->owner != trigger.owner ||
          predicate->graph_generation != plan_generation) {
        return std::nullopt;
      }
      switch (predicate->source) {
        case PredicateEvaluationSourceKind::kInputUpdate:
          return MaxStartDelaySourceCoordinateKind::kPredicateDriver;
        case PredicateEvaluationSourceKind::kClock:
          return MaxStartDelaySourceCoordinateKind::kScheduledTick;
      }
      return std::nullopt;
  }
  return std::nullopt;
}

[[nodiscard]] constexpr bool valid_max_start_delay_plan(
    const MaxStartDelayPlanRecord& plan) noexcept {
  const std::uint32_t expected_callbacks = plan.action == OnViolation::kFailRun ? 0U : 1U;
  return plan.owner.plan_bound() && plan.budget_ns > 0 && plan.source_clock.valid() &&
         valid_max_start_delay_action(plan.action) &&
         valid_max_start_delay_source(plan.source_coordinate) &&
         plan.callback_count_if_late == expected_callbacks && plan.checked_addition &&
         plan.exact_boundary_is_on_time && plan.final_invocation_gate &&
         plan.consumes_candidate_tag && !plan.automatic_output && !plan.callback_state_rollback &&
         plan.duration_measurement_separate;
}

[[nodiscard]] bool valid_max_start_delay_binding_shape(
    const RuntimeMaxStartDelayBindingRecord& binding) noexcept {
  return binding.max_start_delay != nullptr &&
         valid_max_start_delay_plan(*binding.max_start_delay) &&
         binding.max_start_delay->source_clock == binding.source_clock &&
         binding.max_start_delay->conversion_id == binding.conversion_id &&
         binding.plan_generation == binding.max_start_delay->owner.plan_generation &&
         binding.source_clock.valid() && binding.reference_clock.valid() &&
         binding.rounding == ClockRoundingMode::kNearestTiesToEven &&
         binding.overflow == ClockOverflowPolicy::kError &&
         (binding.source_clock == binding.reference_clock) ==
             !binding.source_to_reference.has_value() &&
         (!binding.source_to_reference ||
          (binding.conversion_id != 0U &&
           binding.source_to_reference->source == binding.source_clock &&
           binding.source_to_reference->target == binding.reference_clock &&
           binding.source_to_reference->conversion.scale.numerator > 0 &&
           binding.source_to_reference->conversion.scale.denominator != 0U)) &&
         (binding.source_to_reference || binding.conversion_id == 0U);
}

[[nodiscard]] expected<void, Error> validate_event_coordinates(
    const RuntimeMaxStartDelayBindingRecord& binding, TimePoint source_decision_time,
    TimePoint reference_time) noexcept {
  if (!valid_max_start_delay_binding_shape(binding) || !source_decision_time.valid() ||
      source_decision_time.clock_id != binding.source_clock || !reference_time.valid() ||
      reference_time.clock_id != binding.reference_clock) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const DirectAffineClockEdge* const conversion =
      binding.source_to_reference ? &*binding.source_to_reference : nullptr;
  auto exact_source =
      convert_identical_or_direct_exact(source_decision_time, binding.reference_clock, conversion);
  if (!exact_source) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto materialized_source = materialize_exact_coordinate(*exact_source);
  if (!materialized_source) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  return *materialized_source == reference_time
             ? expected<void, Error>{}
             : expected<void, Error>{make_unexpected(Error{ErrorCode::kProtocolError})};
}

}  // namespace

RuntimeInputSelectionState::Transaction::Transaction(Transaction&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)),
      scalar_(std::move(other.scalar_)),
      drain_(std::move(other.drain_)),
      scalar_sample_(other.scalar_sample_),
      selection_(other.selection_),
      bracketed_(other.bracketed_) {}

RuntimeInputSelectionState::Transaction::~Transaction() {
  if (owner_ != nullptr) {
    owner_->release(*this);
  }
}

std::span<const TemporalSampleRecord> RuntimeInputSelectionState::Transaction::selected()
    const noexcept {
  if (owner_ == nullptr) {
    return {};
  }
  if (scalar_sample_) {
    return std::span<const TemporalSampleRecord>{&*scalar_sample_, 1U};
  }
  if (drain_) {
    return drain_->samples();
  }
  if (bracketed_ && owner_->bracketed_) {
    const auto* result = owner_->bracketed_->front_closed();
    if (result != nullptr) {
      return result->samples;
    }
  }
  return {};
}

TimePoint RuntimeInputSelectionState::Transaction::semantic_reference_time() const noexcept {
  if (owner_ == nullptr) {
    return {};
  }
  if (bracketed_ && owner_->bracketed_) {
    const auto* result = owner_->bracketed_->front_closed();
    return result == nullptr ? TimePoint{} : result->reference_time;
  }
  return {};
}

expected<void, Error> RuntimeInputSelectionState::Transaction::finish(
    bool activation_accepted) noexcept {
  if (owner_ == nullptr || !owner_->transaction_active_) {
    return make_unexpected(Error{ErrorCode::kStaleEndpoint});
  }

  if (activation_accepted) {
    if (scalar_) {
      auto committed = scalar_->commit();
      if (!committed) {
        owner_->release(*this);
        return committed;
      }
      // ReuseLast retention is part of the selection transaction. Publish the retained copy only
      // after the destructive cache commit succeeds so an error cannot leave retained state from
      // an activation that was not accepted.
      if (scalar_sample_) {
        owner_->retained_ = scalar_sample_;
      }
    } else if (drain_) {
      auto committed = drain_->commit();
      if (!committed) {
        owner_->release(*this);
        return committed;
      }
    }
  }
  if (bracketed_ && owner_->bracketed_) {
    // A window close is one terminal source decision. Its selected samples are
    // non-destructive, but the close itself must not be dispatched twice when
    // another input suppresses the candidate.
    owner_->bracketed_->pop_closed();
  }
  owner_->release(*this);
  return {};
}

RuntimeInputSelectionState::RuntimeInputSelectionState(
    InputSelectionPlanRecord plan, std::optional<BracketedWindowEngine> bracketed,
    RuntimeQueueCapacity rate_capacity)
    : plan_(std::move(plan)),
      cache_(rate_capacity.backing, rate_capacity.active),
      bracketed_(std::move(bracketed)),
      latest_within_max_age_(plan_.max_age_ns),
      rate_capacity_(rate_capacity) {}

bool RuntimeInputSelectionState::valid_plan(const InputSelectionPlanRecord& selection,
                                            const BracketedWindowPlanRecord* bracketed_window,
                                            std::uint32_t max_pending_windows) noexcept {
  if (!selection.input || !selection.reference_clock.valid() || selection.cache_capacity == 0U ||
      selection.max_selected_samples == 0U ||
      selection.max_selected_samples > selection.cache_capacity) {
    return false;
  }
  const bool scalar = selection.cardinality == InputSelectionCardinality::kSingleSample;
  const bool batch = selection.cardinality == InputSelectionCardinality::kSampleBatch;
  if (!scalar && !batch) {
    return false;
  }
  switch (selection.missing) {
    case TemporalRecordMissingKind::kUnspecified:
    case TemporalRecordMissingKind::kSkip:
    case TemporalRecordMissingKind::kSafeDefault:
      if (selection.reuse_max_age_ns != 0) {
        return false;
      }
      break;
    case TemporalRecordMissingKind::kReuseLast:
      if (!scalar || selection.reuse_max_age_ns <= 0) {
        return false;
      }
      break;
    default:
      return false;
  }

  switch (selection.selector) {
    case InputSelectorKind::kExact:
      return scalar && selection.max_selected_samples == 1U &&
             selection.max_selected_samples_origin == SelectedSampleBoundOrigin::kSingleSample &&
             selection.batch_selection_mode == BatchSelectionMode::kNotApplicable &&
             selection.max_age_ns == 0 && selection.lookbehind == 0U && selection.lookahead == 0U &&
             bracketed_window == nullptr && max_pending_windows == 0U;
    case InputSelectorKind::kLatestWithin:
      return scalar && selection.max_selected_samples == 1U &&
             selection.max_selected_samples_origin == SelectedSampleBoundOrigin::kSingleSample &&
             selection.batch_selection_mode == BatchSelectionMode::kNotApplicable &&
             selection.max_age_ns > 0 && selection.lookbehind == 0U && selection.lookahead == 0U &&
             bracketed_window == nullptr && max_pending_windows == 0U;
    case InputSelectorKind::kDrainAll:
      return batch &&
             selection.max_selected_samples_origin ==
                 SelectedSampleBoundOrigin::kResolvedRetainedCapacity &&
             selection.batch_selection_mode == BatchSelectionMode::kConsumeSelected &&
             selection.max_age_ns == 0 && selection.lookbehind == 0U && selection.lookahead == 0U &&
             bracketed_window == nullptr && max_pending_windows == 0U &&
             selection.missing == TemporalRecordMissingKind::kUnspecified;
    case InputSelectorKind::kBracketedWindow:
      return batch && bracketed_window != nullptr && max_pending_windows != 0U &&
             selection.max_selected_samples_origin ==
                 SelectedSampleBoundOrigin::kSelectorMaxSamples &&
             selection.batch_selection_mode == BatchSelectionMode::kNonDestructiveSnapshot &&
             selection.max_age_ns == 0 && selection.reuse_max_age_ns == 0 &&
             selection.lookbehind != 0U && selection.lookahead != 0U &&
             bracketed_window->input == selection.input && bracketed_window->anchor &&
             bracketed_window->lookbehind == selection.lookbehind &&
             bracketed_window->lookahead == selection.lookahead &&
             bracketed_window->max_samples == selection.max_selected_samples &&
             bracketed_window->event_clock == selection.reference_clock &&
             bracketed_window->cache_capacity == selection.cache_capacity &&
             bracketed_window->strict_outer_samples && bracketed_window->inclusive_interior &&
             bracketed_window->retain_overlap;
  }
  return false;
}

expected<RuntimeInputSelectionState, Error> RuntimeInputSelectionState::create(
    const InputSelectionPlanRecord& selection, const BracketedWindowPlanRecord* bracketed_window,
    std::uint32_t max_pending_windows, std::optional<RuntimeQueueCapacity> rate_capacity) {
  if (!valid_plan(selection, bracketed_window, max_pending_windows)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const RuntimeQueueCapacity effective_capacity = rate_capacity.value_or(RuntimeQueueCapacity{
      .backing = selection.cache_capacity,
      .active = selection.cache_capacity,
  });
  const std::uint32_t semantic_minimum =
      selection.selector == InputSelectorKind::kDrainAll ? selection.max_selected_samples
      : selection.selector == InputSelectorKind::kBracketedWindow && bracketed_window != nullptr
          ? bracketed_window->max_samples
          : 1U;
  if (effective_capacity.backing < selection.cache_capacity || effective_capacity.active == 0U ||
      effective_capacity.active > effective_capacity.backing ||
      effective_capacity.active < semantic_minimum) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    std::optional<BracketedWindowEngine> bracketed;
    if (bracketed_window != nullptr) {
      bracketed.emplace(BracketedWindow{.lookbehind = bracketed_window->lookbehind,
                                        .lookahead = bracketed_window->lookahead,
                                        .max_samples = bracketed_window->max_samples},
                        effective_capacity.backing,
                        max_pending_windows,
                        effective_capacity.active);
    }
    return RuntimeInputSelectionState{selection, std::move(bracketed), effective_capacity};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
#endif
}

expected<void, Error> RuntimeInputSelectionState::prepare_rate_candidate_capacity(
    std::uint32_t capacity) noexcept {
  const std::uint32_t semantic_minimum =
      plan_.selector == InputSelectorKind::kDrainAll ? plan_.max_selected_samples
      : plan_.selector == InputSelectorKind::kBracketedWindow && bracketed_
          ? plan_.max_selected_samples
          : 1U;
  if (transaction_active_) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  if (capacity == 0U || capacity > rate_capacity_.backing || capacity < semantic_minimum) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  candidate_rate_capacity_ = capacity;
  return {};
}

expected<void, Error> RuntimeInputSelectionState::validate_rate_candidate_capacity_commit()
    const noexcept {
  if (!candidate_rate_capacity_) {
    return {};
  }
  if (transaction_active_) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  if (bracketed_) {
    auto valid = bracketed_->validate_active_cache_capacity(*candidate_rate_capacity_);
    if (!valid) {
      return valid;
    }
  }
  return cache_.validate_active_capacity(*candidate_rate_capacity_);
}

expected<void, Error> RuntimeInputSelectionState::commit_rate_candidate_capacity() noexcept {
  if (!candidate_rate_capacity_) {
    return {};
  }
  auto valid = validate_rate_candidate_capacity_commit();
  if (!valid) {
    return valid;
  }
  commit_validated_rate_candidate_capacity();
  return {};
}

void RuntimeInputSelectionState::commit_validated_rate_candidate_capacity() noexcept {
  if (!candidate_rate_capacity_) {
    return;
  }
  const std::uint32_t capacity = *candidate_rate_capacity_;
  if (bracketed_) {
    bracketed_->commit_validated_active_cache_capacity(capacity);
  }
  cache_.commit_validated_active_capacity(capacity);
  rate_capacity_.active = capacity;
  candidate_rate_capacity_.reset();
}

expected<void, Error> RuntimeInputSelectionState::push(TemporalSampleRecord sample) noexcept {
  if (transaction_active_) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  if (bracketed_) {
    auto accepted = bracketed_->push_sample(sample);
    if (!accepted) {
      return make_unexpected(std::move(accepted.error()));
    }
    return {};
  }
  return cache_.push(sample);
}

expected<WindowAnchorResult, Error> RuntimeInputSelectionState::push_anchor(
    TemporalSampleRecord anchor) noexcept {
  if (transaction_active_) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  if (!bracketed_) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return bracketed_->push_anchor(anchor);
}

expected<void, Error> RuntimeInputSelectionState::close_oldest(SelectionReason reason) noexcept {
  if (transaction_active_) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  if (!bracketed_) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return bracketed_->close_oldest(reason);
}

expected<void, Error> RuntimeInputSelectionState::set_latest_within_max_age(
    std::chrono::nanoseconds max_age) noexcept {
  if (transaction_active_) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  if (plan_.selector != InputSelectorKind::kLatestWithin || max_age.count() <= 0) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  latest_within_max_age_ = max_age;
  return {};
}

expected<void, Error> RuntimeInputSelectionState::close_end_of_stream() noexcept {
  if (transaction_active_) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  if (bracketed_) {
    // End of stream is an unsolicited producer event. Whether the anchor has already closed the
    // last pending window when the producer's close() lands is a race, so a bracketed input can
    // legitimately have nothing pending. One EOS latches terminal closure for every bounded
    // pending window; if closed-result slots are occupied, the engine continues as those results
    // retire instead of waiting for a fabricated second EOS or a timeout.
    bracketed_->close_end_of_stream();
    return {};
  }
  closure_reason_ = SelectionReason::kEndOfStream;
  return {};
}

ScalarSelectionResult RuntimeInputSelectionState::apply_missing(
    ScalarSelectionResult result, std::uint64_t producer_epoch,
    TimePoint reference_time) const noexcept {
  switch (plan_.missing) {
    case TemporalRecordMissingKind::kUnspecified:
    case TemporalRecordMissingKind::kSkip:
      return result;
    case TemporalRecordMissingKind::kSafeDefault:
      return apply_scalar_missing_policy(
          result, SafeDefault{}, retained_, producer_epoch, reference_time);
    case TemporalRecordMissingKind::kReuseLast:
      return apply_scalar_missing_policy(
          result,
          ReuseLast{std::chrono::nanoseconds{plan_.reuse_max_age_ns}},
          retained_,
          producer_epoch,
          reference_time);
  }
  return result;
}

// Selector-specific transactions intentionally share admission and closure handling so every
// strategy exposes the same commit/rollback boundary to the activation dispatcher.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<RuntimeInputSelectionState::Transaction, Error> RuntimeInputSelectionState::begin(
    TimePoint reference_time, std::uint64_t arrival_watermark, std::uint64_t producer_epoch,
    std::optional<TimePoint> future_reference_lower_bound) noexcept {
  if (transaction_active_) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  if (!reference_time.valid() || reference_time.clock_id != plan_.reference_clock ||
      producer_epoch == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  Transaction transaction{this};
  switch (plan_.selector) {
    case InputSelectorKind::kExact: {
      auto scalar = cache_.select_exact_transaction(reference_time, arrival_watermark);
      if (!scalar) {
        return make_unexpected(std::move(scalar.error()));
      }
      auto selected = scalar->result();
      if (closure_reason_ && selected.selection.state != InputState::kSelected) {
        selected.selection.reason = *closure_reason_;
      }
      auto result = apply_missing(selected, producer_epoch, reference_time);
      transaction.selection_ = result.selection;
      transaction.scalar_sample_ = result.sample;
      transaction.scalar_.emplace(std::move(*scalar));
      break;
    }
    case InputSelectorKind::kLatestWithin: {
      auto scalar = cache_.select_latest_within_transaction(
          reference_time, latest_within_max_age_, arrival_watermark, future_reference_lower_bound);
      if (!scalar) {
        return make_unexpected(std::move(scalar.error()));
      }
      auto selected = scalar->result();
      if (closure_reason_ && selected.selection.state != InputState::kSelected) {
        selected.selection.reason = *closure_reason_;
      }
      auto result = apply_missing(selected, producer_epoch, reference_time);
      transaction.selection_ = result.selection;
      transaction.scalar_sample_ = result.sample;
      transaction.scalar_.emplace(std::move(*scalar));
      break;
    }
    case InputSelectorKind::kDrainAll: {
      // Automatic DrainAll consumes the complete logical prefix admitted by the published rate
      // generation. Explicit selector bounds remain frozen in the plan.
      const std::uint32_t service_bound =
          plan_.max_selected_samples_origin == SelectedSampleBoundOrigin::kResolvedRetainedCapacity
              ? rate_capacity_.active
              : plan_.max_selected_samples;
      auto drain = cache_.snapshot_drain_all(arrival_watermark, service_bound);
      if (!drain) {
        return make_unexpected(std::move(drain.error()));
      }
      transaction.selection_ = drain->samples().empty() && closure_reason_
                                   ? InputSelection{InputState::kAbsent, *closure_reason_}
                                   : InputSelection{InputState::kSelected, SelectionReason::kNone};
      transaction.drain_.emplace(std::move(*drain));
      break;
    }
    case InputSelectorKind::kBracketedWindow: {
      const auto* closed = bracketed_ ? bracketed_->front_closed() : nullptr;
      if (closed == nullptr) {
        return make_unexpected(Error{ErrorCode::kNotReady});
      }
      // The candidate reference is the coordinate that made the close observable (a successor
      // arrival, timeout, or EOS). The semantic reference remains the current anchor captured in
      // the pending window, so only the admitted domain and causal ordering must agree here.
      if (!closed->reference_time.valid() ||
          closed->reference_time.clock_id != reference_time.clock_id ||
          closed->reference_time.timestamp_ns > reference_time.timestamp_ns) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      transaction.selection_ = closed->selection;
      if (transaction.selection_.state == InputState::kAbsent &&
          plan_.missing == TemporalRecordMissingKind::kSafeDefault) {
        // A timeout/EOS close remains an all-or-nothing empty batch, but SafeDefault
        // deliberately turns that retained reason into one degraded callback.
        transaction.selection_.state = InputState::kSafeDefault;
      }
      transaction.bracketed_ = true;
      break;
    }
  }
  transaction_active_ = true;
  return transaction;
}

expected<void, Error> RuntimeInputSelectionState::reset() noexcept {
  if (transaction_active_) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  auto reset_cache = cache_.reset();
  if (!reset_cache) {
    return reset_cache;
  }
  if (bracketed_) {
    bracketed_->reset();
  }
  retained_.reset();
  closure_reason_.reset();
  return {};
}

expected<void, Error> RuntimeInputSelectionState::release_terminal_history() noexcept {
  if (transaction_active_) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  if (bracketed_ || closure_reason_ != SelectionReason::kEndOfStream) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto released = cache_.release_terminal_history();
  if (!released) {
    return released;
  }
  retained_.reset();
  return {};
}

std::size_t RuntimeInputSelectionState::retained_count() const noexcept {
  return cache_.size();
}

std::size_t RuntimeInputSelectionState::maximum_selected_capacity() const noexcept {
  return plan_.selector == InputSelectorKind::kDrainAll &&
                 plan_.max_selected_samples_origin ==
                     SelectedSampleBoundOrigin::kResolvedRetainedCapacity
             ? rate_capacity_.backing
             : plan_.max_selected_samples;
}

std::size_t RuntimeInputSelectionState::payload_claim_capacity() const noexcept {
  if (bracketed_) {
    return bracketed_->payload_claim_capacity();
  }
  const bool retains_reuse = plan_.missing == TemporalRecordMissingKind::kReuseLast;
  if (retains_reuse && cache_.capacity() == std::numeric_limits<std::size_t>::max()) {
    return 0U;
  }
  return cache_.capacity() + static_cast<std::size_t>(retains_reuse);
}

expected<void, Error> RuntimeInputSelectionState::evict_retained(
    const std::uint64_t arrival_sequence) noexcept {
  if (bracketed_) {
    // A bracketed window owns its retained set through closure ordering. Releasing one record from
    // underneath it would leave a closed window referencing a retired payload.
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }
  if (transaction_active_) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  return cache_.evict(arrival_sequence);
}

expected<RuntimeInputSelectionState::ReuseReplacementTransaction, Error>
RuntimeInputSelectionState::reserve_reuse_replacement(
    const TemporalSampleRecord& retained) noexcept {
  if (transaction_active_) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  if (bracketed_ || plan_.missing != TemporalRecordMissingKind::kReuseLast || !retained_ ||
      *retained_ != retained) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  retained_.reset();
  transaction_active_ = true;
  return ReuseReplacementTransaction{this, retained};
}

expected<RuntimeInputSelectionState::CacheReplacementTransaction, Error>
RuntimeInputSelectionState::reserve_cache_replacement(
    const TemporalSampleRecord& retained) noexcept {
  if (transaction_active_) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  if (bracketed_) {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }
  const auto records = cache_.records();
  const auto found = std::find(records.begin(), records.end(), retained);
  if (found == records.end()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const std::size_t index = static_cast<std::size_t>(std::distance(records.begin(), found));
  auto evicted = cache_.evict(retained.arrival_sequence);
  if (!evicted) {
    return make_unexpected(std::move(evicted).error());
  }
  transaction_active_ = true;
  return CacheReplacementTransaction{this, retained, index};
}

void RuntimeInputSelectionState::restore_cache_replacement(const TemporalSampleRecord& retained,
                                                           const std::size_t index) noexcept {
  assert(index <= cache_.samples_.size());
  assert(cache_.samples_.size() < cache_.capacity_);
  cache_.samples_.insert(cache_.samples_.begin() + static_cast<std::ptrdiff_t>(index), retained);
}

std::optional<TemporalSampleRecord> RuntimeInputSelectionState::reuse_retained_record()
    const noexcept {
  return plan_.missing == TemporalRecordMissingKind::kReuseLast ? retained_ : std::nullopt;
}

bool RuntimeInputSelectionState::staged_reuse_eligible(
    const TimePoint staged_reference, const TimePoint witness_reference) const noexcept {
  if (!staged_reference.valid() || !witness_reference.valid() ||
      staged_reference.clock_id != plan_.reference_clock ||
      witness_reference.clock_id != plan_.reference_clock) {
    return false;
  }
  switch (plan_.selector) {
    case InputSelectorKind::kExact:
      return staged_reference.timestamp_ns == witness_reference.timestamp_ns;
    case InputSelectorKind::kLatestWithin: {
      if (latest_within_max_age_.count() <= 0 ||
          staged_reference.timestamp_ns > witness_reference.timestamp_ns) {
        return false;
      }
      const auto age = static_cast<std::uint64_t>(witness_reference.timestamp_ns) -
                       static_cast<std::uint64_t>(staged_reference.timestamp_ns);
      return age <= static_cast<std::uint64_t>(latest_within_max_age_.count());
    }
    case InputSelectorKind::kDrainAll:
    case InputSelectorKind::kBracketedWindow:
      return false;
  }
  return false;
}

expected<std::size_t, Error> RuntimeInputSelectionState::copy_payload_claims(
    std::span<TemporalSampleRecord> output) const noexcept {
  if (bracketed_) {
    return bracketed_->copy_payload_claims(output);
  }
  std::size_t count = 0U;
  for (const TemporalSampleRecord& record : cache_.records()) {
    if (count == output.size()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    output[count++] = record;
  }
  if (plan_.missing == TemporalRecordMissingKind::kReuseLast && retained_) {
    if (count == output.size()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    output[count++] = *retained_;
  }
  return count;
}

std::size_t RuntimeInputSelectionState::pending_window_count() const noexcept {
  return bracketed_ ? bracketed_->pending_count() : 0U;
}

bool RuntimeInputSelectionState::terminal_decision_ready() const noexcept {
  return bracketed_ && bracketed_->front_closed() != nullptr;
}

std::optional<TimePoint> RuntimeInputSelectionState::oldest_pending_window_reference()
    const noexcept {
  return bracketed_ ? bracketed_->oldest_pending_reference() : std::nullopt;
}

RuntimeAlignedSelectionState::Transaction::Transaction(
    RuntimeAlignedSelectionState* owner, AlignedJoinEngine::Transaction transaction) noexcept
    : owner_(owner), transaction_(std::move(transaction)) {}

RuntimeAlignedSelectionState::Transaction::Transaction(Transaction&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), transaction_(std::move(other.transaction_)) {}

RuntimeAlignedSelectionState::Transaction::~Transaction() {
  if (owner_ != nullptr) {
    owner_->release(*this);
  }
}

AlignedCandidateDisposition RuntimeAlignedSelectionState::Transaction::disposition()
    const noexcept {
  return transaction_ ? transaction_->disposition() : AlignedCandidateDisposition::kPending;
}

TimePoint RuntimeAlignedSelectionState::Transaction::reference_time() const noexcept {
  return transaction_ ? transaction_->reference_time() : TimePoint{};
}

SelectionReason RuntimeAlignedSelectionState::Transaction::reason() const noexcept {
  return transaction_ ? transaction_->reason() : SelectionReason::kNone;
}

std::span<const InputSelection> RuntimeAlignedSelectionState::Transaction::selections()
    const noexcept {
  return transaction_ ? transaction_->selections() : std::span<const InputSelection>{};
}

std::span<const TemporalSampleRecord> RuntimeAlignedSelectionState::Transaction::selected()
    const noexcept {
  return transaction_ ? transaction_->members() : std::span<const TemporalSampleRecord>{};
}

expected<void, Error> RuntimeAlignedSelectionState::Transaction::finish(
    bool activation_accepted) noexcept {
  if (owner_ == nullptr || !owner_->transaction_active_ || !transaction_) {
    return make_unexpected(Error{ErrorCode::kStaleEndpoint});
  }
  const AlignedCandidateDisposition frozen_disposition = transaction_->disposition();
  if (activation_accepted && frozen_disposition != AlignedCandidateDisposition::kNominal &&
      frozen_disposition != AlignedCandidateDisposition::kDegraded) {
    owner_->release(*this);
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (activation_accepted || frozen_disposition == AlignedCandidateDisposition::kRetireMisaligned) {
    auto committed = transaction_->commit();
    if (!committed) {
      owner_->release(*this);
      return committed;
    }
  }
  owner_->release(*this);
  return {};
}

RuntimeAlignedSelectionState::RuntimeAlignedSelectionState(
    AlignedTriggerPlanRecord plan, std::vector<PortId> member_inputs,
    std::vector<std::uint32_t> member_backing_capacities,
    std::vector<std::uint32_t> member_active_capacities, std::vector<AlignedMissingPolicy> missing,
    std::size_t engine_capacity)
    : plan_(std::move(plan)),
      member_inputs_(std::move(member_inputs)),
      member_backing_capacities_(std::move(member_backing_capacities)),
      member_active_capacities_(std::move(member_active_capacities)),
      candidate_member_capacities_(member_inputs_.size()),
      missing_(std::move(missing)),
      open_closure_(member_inputs_.size(), AlignmentClosureProof::kOpen),
      engine_(member_inputs_.size(), engine_capacity) {}

expected<RuntimeAlignedSelectionState, Error> RuntimeAlignedSelectionState::create(
    const AlignedTriggerPlanRecord& aligned, std::span<const InputSelectionPlanRecord> selections,
    std::span<const RuntimeQueueCapacity> rate_capacities) {
  constexpr std::array kClosureProofs{
      AlignedClosureProofKind::kStreamAdvancement,
      AlignedClosureProofKind::kLossWitness,
      AlignedClosureProofKind::kPrivateWatermark,
      AlignedClosureProofKind::kEndOfStream,
      AlignedClosureProofKind::kProducerEpochClosed,
      AlignedClosureProofKind::kTemporalClosure,
  };
  if (!aligned.owner.plan_bound() || aligned.members.size() < 2U ||
      aligned.inclusive_window_ns < 0 || !aligned.comparison_clock.valid() ||
      aligned.algorithm != "holoscan.temporal.aligned-oldest-head/v1" || !aligned.fifo_heads_only ||
      !aligned.no_scan_ahead || !aligned.checked_direct_conversion ||
      !aligned.checked_spread_subtraction || !aligned.exact_window_matches ||
      !aligned.reference_is_maximum_head || !aligned.accepted_once_consumption ||
      !aligned.retire_all_minimum_heads ||
      aligned.retirement_order != "port_id_then_source_sequence" ||
      aligned.closure_proofs.size() != kClosureProofs.size() ||
      !std::equal(
          aligned.closure_proofs.begin(), aligned.closure_proofs.end(), kClosureProofs.begin()) ||
      !aligned.silence_is_not_closure || !aligned.missing_requires_finite_proof ||
      aligned.public_watermark_protocol) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (!rate_capacities.empty() && rate_capacities.size() != aligned.members.size()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    std::vector<PortId> member_inputs;
    std::vector<std::uint32_t> member_backing_capacities;
    std::vector<std::uint32_t> member_active_capacities;
    std::vector<AlignedMissingPolicy> missing;
    member_inputs.reserve(aligned.members.size());
    member_backing_capacities.reserve(aligned.members.size());
    member_active_capacities.reserve(aligned.members.size());
    missing.reserve(aligned.members.size());
    std::size_t engine_capacity = 0U;
    for (std::size_t index = 0U; index < aligned.members.size(); ++index) {
      const AlignedMemberPlanRecord& member = aligned.members[index];
      const auto selection =
          std::find_if(selections.begin(), selections.end(), [&](const auto& row) {
            return row.input == member.input;
          });
      const std::size_t selection_count = static_cast<std::size_t>(
          std::count_if(selections.begin(), selections.end(), [&](const auto& row) {
            return row.input == member.input;
          }));
      const bool conversion_valid = member.event_clock == aligned.comparison_clock
                                        ? member.conversion_id == 0U
                                        : member.conversion_id != 0U;
      if (!member.input || member.member_ordinal != index || !member.event_clock.valid() ||
          !conversion_valid || member.queue_capacity == 0U || member.cache_capacity == 0U ||
          !member.capture_time_required || !member.nondecreasing_normalized_time ||
          std::find(member_inputs.begin(), member_inputs.end(), member.input) !=
              member_inputs.end() ||
          selection == selections.end() || selection_count != 1U ||
          selection->selector != InputSelectorKind::kExact ||
          selection->cardinality != InputSelectionCardinality::kSingleSample ||
          selection->max_selected_samples != 1U ||
          selection->max_selected_samples_origin != SelectedSampleBoundOrigin::kSingleSample ||
          selection->batch_selection_mode != BatchSelectionMode::kNotApplicable ||
          selection->cache_capacity != member.cache_capacity ||
          selection->reference_clock != aligned.comparison_clock ||
          selection->conversion_id != member.conversion_id || selection->max_age_ns != 0 ||
          selection->lookbehind != 0U || selection->lookahead != 0U) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      const RuntimeQueueCapacity capacity =
          rate_capacities.empty() ? RuntimeQueueCapacity{.backing = member.cache_capacity,
                                                         .active = member.cache_capacity}
                                  : rate_capacities[index];
      if (capacity.backing < member.cache_capacity || capacity.active < member.cache_capacity ||
          capacity.active > capacity.backing) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      AlignedMissingPolicy policy;
      switch (selection->missing) {
        case TemporalRecordMissingKind::kUnspecified:
          policy.kind = AlignedMissingPolicyKind::kUnspecified;
          break;
        case TemporalRecordMissingKind::kSkip:
          policy.kind = AlignedMissingPolicyKind::kSkip;
          break;
        case TemporalRecordMissingKind::kSafeDefault:
          policy.kind = AlignedMissingPolicyKind::kSafeDefault;
          break;
        case TemporalRecordMissingKind::kReuseLast:
          if (selection->reuse_max_age_ns <= 0) {
            return make_unexpected(Error{ErrorCode::kInvalidArgument});
          }
          policy.kind = AlignedMissingPolicyKind::kReuseLast;
          policy.reuse_max_age = std::chrono::nanoseconds{selection->reuse_max_age_ns};
          break;
        default:
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      if (selection->missing != TemporalRecordMissingKind::kReuseLast &&
          selection->reuse_max_age_ns != 0) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      member_inputs.push_back(member.input);
      member_backing_capacities.push_back(capacity.backing);
      member_active_capacities.push_back(capacity.active);
      missing.push_back(policy);
      engine_capacity = std::max(engine_capacity, static_cast<std::size_t>(capacity.backing));
    }
    return RuntimeAlignedSelectionState{aligned,
                                        std::move(member_inputs),
                                        std::move(member_backing_capacities),
                                        std::move(member_active_capacities),
                                        std::move(missing),
                                        engine_capacity};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
#endif
}

expected<void, Error> RuntimeAlignedSelectionState::prepare_rate_candidate_capacity(
    PortId input, std::uint32_t capacity) noexcept {
  if (transaction_active_) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  const std::size_t index = member_index(input);
  if (index == std::numeric_limits<std::size_t>::max() || capacity == 0U ||
      capacity > member_backing_capacities_[index] ||
      capacity < plan_.members[index].cache_capacity) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  candidate_member_capacities_[index] = capacity;
  return {};
}

expected<void, Error> RuntimeAlignedSelectionState::validate_rate_candidate_capacities_commit()
    const noexcept {
  if (transaction_active_) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  for (std::size_t index = 0U; index < candidate_member_capacities_.size(); ++index) {
    const std::uint32_t candidate =
        candidate_member_capacities_[index].value_or(std::numeric_limits<std::uint32_t>::max());
    if (engine_.queue_size(index) > candidate) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
  }
  return {};
}

expected<void, Error> RuntimeAlignedSelectionState::commit_rate_candidate_capacities() noexcept {
  auto valid = validate_rate_candidate_capacities_commit();
  if (!valid) {
    return valid;
  }
  commit_validated_rate_candidate_capacities();
  return {};
}

void RuntimeAlignedSelectionState::commit_validated_rate_candidate_capacities() noexcept {
  for (std::size_t index = 0U; index < candidate_member_capacities_.size(); ++index) {
    auto& candidate = candidate_member_capacities_[index];
    if (!candidate) {
      continue;
    }
    member_active_capacities_[index] = candidate.value_or(member_active_capacities_[index]);
    candidate.reset();
  }
}

void RuntimeAlignedSelectionState::discard_rate_candidate_capacities() noexcept {
  for (auto& capacity : candidate_member_capacities_) {
    capacity.reset();
  }
}

std::size_t RuntimeAlignedSelectionState::member_index(PortId input) const noexcept {
  const auto found = std::find(member_inputs_.begin(), member_inputs_.end(), input);
  return found == member_inputs_.end()
             ? std::numeric_limits<std::size_t>::max()
             : static_cast<std::size_t>(std::distance(member_inputs_.begin(), found));
}

expected<void, Error> RuntimeAlignedSelectionState::push(PortId input,
                                                         TemporalSampleRecord sample) noexcept {
  if (transaction_active_) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  const std::size_t index = member_index(input);
  if (index == std::numeric_limits<std::size_t>::max() ||
      sample.capture_time.clock_id != plan_.comparison_clock ||
      engine_.queue_size(index) >= member_active_capacities_[index]) {
    return make_unexpected(Error{index == std::numeric_limits<std::size_t>::max() ||
                                         sample.capture_time.clock_id != plan_.comparison_clock
                                     ? ErrorCode::kInvalidArgument
                                     : ErrorCode::kResourceExhausted});
  }
  return engine_.push(index, sample);
}

expected<void, Error> RuntimeAlignedSelectionState::close_end_of_stream(
    PortId input, TimePoint closure_time) noexcept {
  if (transaction_active_) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  const std::size_t index = member_index(input);
  if (index == std::numeric_limits<std::size_t>::max() || !closure_time.valid() ||
      closure_time.clock_id != plan_.comparison_clock) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  open_closure_[index] = AlignmentClosureProof::kEndOfStream;
  if (!closure_time_.valid() || closure_time_.timestamp_ns < closure_time.timestamp_ns) {
    closure_time_ = closure_time;
  }
  return {};
}

expected<RuntimeAlignedSelectionState::Transaction, Error> RuntimeAlignedSelectionState::begin(
    std::span<const AlignmentClosureProof> closure, TimePoint closure_time) noexcept {
  if (transaction_active_) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  if (!closure.empty() && closure.size() != member_inputs_.size()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const std::span<const AlignmentClosureProof> effective_closure =
      closure.empty() ? std::span<const AlignmentClosureProof>{open_closure_} : closure;
  const TimePoint effective_closure_time = closure_time.valid() ? closure_time
                                           : closure.empty()    ? closure_time_
                                                                : TimePoint{};
  auto selected = engine_.select(std::chrono::nanoseconds{plan_.inclusive_window_ns},
                                 missing_,
                                 effective_closure,
                                 effective_closure_time);
  if (!selected) {
    return make_unexpected(std::move(selected.error()));
  }
  if (selected->disposition() == AlignedCandidateDisposition::kPending) {
    return make_unexpected(Error{ErrorCode::kNotReady});
  }
  transaction_active_ = true;
  return Transaction{this, std::move(*selected)};
}

expected<void, Error> RuntimeAlignedSelectionState::reset() noexcept {
  if (transaction_active_) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  auto reset = engine_.reset();
  if (!reset) {
    return reset;
  }
  std::fill(open_closure_.begin(), open_closure_.end(), AlignmentClosureProof::kOpen);
  closure_time_ = {};
  return {};
}

expected<void, Error> RuntimeAlignedSelectionState::release_terminal_history() noexcept {
  if (transaction_active_) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  const bool every_member_terminal =
      std::all_of(open_closure_.begin(), open_closure_.end(), [](AlignmentClosureProof proof) {
        return proof == AlignmentClosureProof::kEndOfStream ||
               proof == AlignmentClosureProof::kProducerEpochClosed;
      });
  if (!every_member_terminal) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return engine_.release_terminal_history();
}

std::size_t RuntimeAlignedSelectionState::queued(PortId input) const noexcept {
  const std::size_t index = member_index(input);
  return index == std::numeric_limits<std::size_t>::max() ? 0U : engine_.queue_size(index);
}

std::size_t RuntimeAlignedSelectionState::payload_claim_capacity(PortId input) const noexcept {
  const std::size_t index = member_index(input);
  if (index == std::numeric_limits<std::size_t>::max()) {
    return 0U;
  }
  const bool retains_reuse = missing_[index].kind == AlignedMissingPolicyKind::kReuseLast;
  const std::size_t member_capacity = member_backing_capacities_[index];
  if (retains_reuse && member_capacity == std::numeric_limits<std::size_t>::max()) {
    return 0U;
  }
  return member_capacity + static_cast<std::size_t>(retains_reuse);
}

bool RuntimeAlignedSelectionState::terminal_decision_ready() const noexcept {
  const bool every_member_closed =
      std::all_of(open_closure_.begin(), open_closure_.end(), [](AlignmentClosureProof proof) {
        return proof == AlignmentClosureProof::kEndOfStream ||
               proof == AlignmentClosureProof::kProducerEpochClosed;
      });
  return every_member_closed && std::any_of(member_inputs_.begin(),
                                            member_inputs_.end(),
                                            [&](PortId input) { return queued(input) != 0U; });
}

expected<std::size_t, Error> RuntimeAlignedSelectionState::copy_payload_claims(
    PortId input, std::span<TemporalSampleRecord> output) const noexcept {
  const std::size_t index = member_index(input);
  if (index == std::numeric_limits<std::size_t>::max()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  std::size_t count = 0U;
  for (const TemporalSampleRecord& record : engine_.queue_records(index)) {
    if (count == output.size()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    output[count++] = record;
  }
  if (missing_[index].kind == AlignedMissingPolicyKind::kReuseLast) {
    const auto retained = engine_.retained_record(index);
    if (retained) {
      if (count == output.size()) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      output[count++] = *retained;
    }
  }
  return count;
}

void RuntimeAlignedSelectionState::release(Transaction& transaction) noexcept {
  transaction.transaction_.reset();
  transaction.owner_ = nullptr;
  transaction_active_ = false;
}

void RuntimeInputSelectionState::release(Transaction& transaction) noexcept {
  transaction.scalar_.reset();
  transaction.drain_.reset();
  transaction.scalar_sample_.reset();
  transaction.owner_ = nullptr;
  transaction_active_ = false;
}

expected<MaxStartDelayGateResult, Error> evaluate_plan_max_start_delay(
    const MaxStartDelayPlanRecord& plan, TimePoint source_coordinate,
    TimePoint dispatch_coordinate) noexcept {
  if (!valid_max_start_delay_plan(plan) || source_coordinate.clock_id != plan.source_clock) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return evaluate_max_start_delay_gate(source_coordinate,
                                       std::chrono::nanoseconds{plan.budget_ns},
                                       dispatch_coordinate,
                                       plan.action);
}

// Keep the exact trigger, max_start_delay, conversion, and predicate cross-checks together so
// startup rejects any partially matching frozen plan rather than normalizing it.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<std::optional<RuntimeMaxStartDelayBindingRecord>, Error> bind_runtime_max_start_delay(
    std::span<const MaxStartDelayPlanRecord> max_start_delays,
    const TemporalTriggerPlanRecord& trigger, std::span<const ClockConversionRecord> conversions,
    PlanGenerationId plan_generation, const PredicateTriggerPlanRecord* predicate) noexcept {
  const MaxStartDelayPlanRecord* max_start_delay = nullptr;
  for (const auto& candidate : max_start_delays) {
    if (candidate.owner != trigger.owner) {
      continue;
    }
    if (max_start_delay != nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    max_start_delay = &candidate;
  }

  const bool trigger_has_max_start_delay =
      trigger.max_start_delay_ns.has_value() && trigger.violation.has_value();
  if (trigger.max_start_delay_ns.has_value() != trigger.violation.has_value() ||
      trigger.owner.plan_generation != plan_generation || trigger.owner.run_generation != 0U ||
      !trigger.reference_clock.valid()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (!trigger_has_max_start_delay) {
    return max_start_delay == nullptr && trigger.conversion_id == 0U
               ? expected<std::optional<RuntimeMaxStartDelayBindingRecord>, Error>{std::nullopt}
               : expected<std::optional<RuntimeMaxStartDelayBindingRecord>, Error>{
                     make_unexpected(Error{ErrorCode::kInvalidArgument})};
  }
  const std::int64_t max_start_delay_ns = trigger.max_start_delay_ns.value();
  const OnViolation violation = trigger.violation.value();
  const auto expected_source = expected_max_start_delay_source(trigger, predicate, plan_generation);
  if (max_start_delay == nullptr || !valid_max_start_delay_plan(*max_start_delay) ||
      max_start_delay->owner != trigger.owner || max_start_delay->budget_ns != max_start_delay_ns ||
      max_start_delay->action != violation ||
      max_start_delay->conversion_id != trigger.conversion_id || !expected_source ||
      max_start_delay->source_coordinate != expected_source.value()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const MaxStartDelaySourceCoordinateKind source_coordinate = expected_source.value();
  const bool input_derived_source =
      source_coordinate == MaxStartDelaySourceCoordinateKind::kTriggerSample ||
      source_coordinate == MaxStartDelaySourceCoordinateKind::kPredicateDriver;
  if (!input_derived_source && (max_start_delay->source_clock != trigger.reference_clock ||
                                max_start_delay->conversion_id != 0U)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  RuntimeMaxStartDelayBindingRecord result{
      .max_start_delay = max_start_delay,
      .plan_generation = plan_generation,
      .source_clock = max_start_delay->source_clock,
      .reference_clock = trigger.reference_clock,
      .conversion_id = max_start_delay->conversion_id,
  };
  if (max_start_delay->source_clock == trigger.reference_clock) {
    if (max_start_delay->conversion_id != 0U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    return std::optional<RuntimeMaxStartDelayBindingRecord>{result};
  }
  if (max_start_delay->conversion_id == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const ClockConversionRecord* conversion = nullptr;
  for (const auto& candidate : conversions) {
    if (candidate.id != max_start_delay->conversion_id) {
      continue;
    }
    if (conversion != nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    conversion = &candidate;
  }
  if (conversion == nullptr || conversion->plan_generation != plan_generation ||
      conversion->from != max_start_delay->source_clock ||
      conversion->to != trigger.reference_clock || conversion->scale.numerator <= 0 ||
      conversion->scale.denominator == 0U ||
      conversion->rounding != ClockRoundingMode::kNearestTiesToEven ||
      conversion->overflow != ClockOverflowPolicy::kError) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  result.rounding = conversion->rounding;
  result.overflow = conversion->overflow;
  result.source_to_reference = DirectAffineClockEdge{
      .source = conversion->from,
      .target = conversion->to,
      .conversion = {.scale = conversion->scale, .offset_ns = conversion->offset_ns},
  };
  return std::optional<RuntimeMaxStartDelayBindingRecord>{result};
}

expected<void, Error> validate_runtime_max_start_delay_inventory(
    std::span<const MaxStartDelayPlanRecord> max_start_delays,
    std::span<const TemporalTriggerPlanRecord> triggers,
    std::span<const PredicateTriggerPlanRecord> predicates,
    std::span<const ClockConversionRecord> conversions, PlanGenerationId plan_generation) noexcept {
  std::size_t bound_max_start_delays = 0U;
  for (const auto& trigger : triggers) {
    const PredicateTriggerPlanRecord* predicate = nullptr;
    if (trigger.kind == TemporalRecordTriggerKind::kPredicate) {
      for (const auto& candidate : predicates) {
        if (candidate.owner != trigger.owner) {
          continue;
        }
        if (predicate != nullptr) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        predicate = &candidate;
      }
    }
    auto binding = bind_runtime_max_start_delay(
        max_start_delays, trigger, conversions, plan_generation, predicate);
    if (!binding) {
      return make_unexpected(std::move(binding.error()));
    }
    bound_max_start_delays += binding->has_value() ? 1U : 0U;
  }
  if (bound_max_start_delays != max_start_delays.size()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return {};
}

expected<MaxStartDelayGateResult, Error> evaluate_runtime_max_start_delay(
    const RuntimeMaxStartDelayBindingRecord& binding, TimePoint source_decision_time,
    TimePoint reference_time, TimePoint dispatch_reference_time) noexcept {
  return evaluate_runtime_max_start_delay(
      binding,
      source_decision_time,
      reference_time,
      dispatch_reference_time,
      std::chrono::nanoseconds{
          binding.max_start_delay == nullptr ? 0 : binding.max_start_delay->budget_ns});
}

expected<MaxStartDelayGateResult, Error> evaluate_runtime_max_start_delay(
    const RuntimeMaxStartDelayBindingRecord& binding, TimePoint source_decision_time,
    TimePoint reference_time, TimePoint dispatch_reference_time,
    std::chrono::nanoseconds budget) noexcept {
  if (!valid_max_start_delay_binding_shape(binding) || !dispatch_reference_time.valid() ||
      dispatch_reference_time.clock_id != binding.reference_clock || budget.count() <= 0) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (auto coordinates = validate_event_coordinates(binding, source_decision_time, reference_time);
      !coordinates) {
    return make_unexpected(std::move(coordinates.error()));
  }

  const DirectAffineClockEdge* const conversion =
      binding.source_to_reference ? &*binding.source_to_reference : nullptr;
  if (source_decision_time.timestamp_ns >
      std::numeric_limits<std::int64_t>::max() - budget.count()) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  const TimePoint latest_source{source_decision_time.timestamp_ns + budget.count(),
                                binding.source_clock};
  auto exact_latest =
      convert_identical_or_direct_exact(latest_source, binding.reference_clock, conversion);
  auto exact_dispatch =
      convert_identical_or_direct_exact(dispatch_reference_time, binding.reference_clock, nullptr);
  if (!exact_latest || !exact_dispatch) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto order = compare_exact_coordinates(*exact_dispatch, *exact_latest);
  if (!order) {
    return make_unexpected(std::move(order.error()));
  }
  auto materialized_latest = materialize_exact_coordinate(*exact_latest);
  if (!materialized_latest) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  const bool on_time = *order != ExactCoordinateOrder::kGreater;
  std::int64_t checked_lateness_ns = 0;
  if (!on_time && dispatch_reference_time.timestamp_ns > materialized_latest->timestamp_ns) {
    if (materialized_latest->timestamp_ns < 0 &&
        dispatch_reference_time.timestamp_ns >
            std::numeric_limits<std::int64_t>::max() + materialized_latest->timestamp_ns) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    checked_lateness_ns = dispatch_reference_time.timestamp_ns - materialized_latest->timestamp_ns;
  }
  const OnViolation action = binding.max_start_delay->action;
  return MaxStartDelayGateResult{
      .disposition = on_time ? MaxStartDelayGateDisposition::kInvokeNominal
                     : action == OnViolation::kFailRun
                         ? MaxStartDelayGateDisposition::kFailRun
                         : MaxStartDelayGateDisposition::kInvokeViolation,
      .status = {on_time ? ActivationReason::kNominal : ActivationReason::kMaxStartDelayExceeded,
                 action},
      // Evidence is materialized only after the exact ordering decision, so
      // ties-to-even rounding cannot change nominal/late classification.
      .latest_start = *materialized_latest,
      .dispatch_start = dispatch_reference_time,
      .checked_lateness_ns = checked_lateness_ns,
  };
}

ActivationStatus merge_max_start_delay_activation_status(
    ActivationStatus current, const MaxStartDelayGateResult& max_start_delay) noexcept {
  if (max_start_delay.disposition == MaxStartDelayGateDisposition::kInvokeNominal &&
      current.reason != ActivationReason::kNominal) {
    return current;
  }
  return max_start_delay.status;
}

BoundedEventPortBinding::BoundedEventPortBinding(std::size_t capacity, SourceId source)
    : source_(source), storage_(capacity) {}

expected<void, Error> BoundedEventPortBinding::push(RuntimeEvent event) noexcept {
  if (closed_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  if (storage_.empty() || size_ == storage_.size()) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  storage_[tail_] = std::move(event);
  tail_ = (tail_ + 1U) % storage_.size();
  ++size_;
  return {};
}

expected<std::size_t, Error> BoundedEventPortBinding::drain(
    std::span<RuntimeEvent> destination) noexcept {
  const std::size_t count = std::min(size_, destination.size());
  for (std::size_t index = 0U; index < count; ++index) {
    destination[index] = std::move(storage_[head_]);
    head_ = (head_ + 1U) % storage_.size();
  }
  size_ -= count;
  return count;
}

RuntimeCandidateCoordinator::RuntimeCandidateCoordinator(
    std::vector<SourceState> sources, std::vector<std::unique_ptr<TagAllocator>> owned_tags,
    std::size_t max_outstanding, RuntimePhaseCounters* phase_counters)
    : sources_(std::move(sources)),
      owned_tags_(std::move(owned_tags)),
      tags_(owned_tags_.size()),
      max_outstanding_(max_outstanding),
      phase_counters_(phase_counters) {
  for (std::size_t index = 0U; index < owned_tags_.size(); ++index) {
    tags_[index] = owned_tags_[index].get();
  }
  ingress_scratch_.resize(max_outstanding);
  pending_.reserve(max_outstanding);
  input_fence_head_scratch_.reserve(max_outstanding);
}

expected<RuntimeCandidateCoordinator, Error> RuntimeCandidateCoordinator::create_with_tags(
    std::span<const RuntimeSourceBindingRecord> sources,
    std::span<TagAllocator* const> receiver_tags, std::size_t max_outstanding,
    RuntimePhaseCounters* phase_counters) {
  auto result = create(sources, receiver_tags.size(), max_outstanding, phase_counters);
  if (!result) {
    return make_unexpected(std::move(result.error()));
  }
  for (std::size_t index = 0U; index < receiver_tags.size(); ++index) {
    if (result->tags_[index] != nullptr &&
        (receiver_tags[index] == nullptr ||
         receiver_tags[index]->generation() != result->tags_[index]->generation())) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
  }
  result->owned_tags_.clear();
  result->tags_.assign(receiver_tags.begin(), receiver_tags.end());
  return result;
}

// Validate the complete closed source inventory in one pass before publishing any coordinator
// state; the adjacent checks are the fail-closed reconstruction contract.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<RuntimeCandidateCoordinator, Error> RuntimeCandidateCoordinator::create(
    std::span<const RuntimeSourceBindingRecord> sources, std::size_t receiver_count,
    std::size_t max_outstanding, RuntimePhaseCounters* phase_counters) {
  if (sources.empty() || receiver_count == 0U || max_outstanding == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    std::vector<SourceState> states;
    states.reserve(sources.size());
    std::vector<std::unique_ptr<TagAllocator>> tags(receiver_count);
    ClockId reference_clock{};
    for (const auto& source : sources) {
      if (!candidate_kind(source.candidate_kind) || !source.source ||
          !source.source_clock.valid() || !source.reference_clock.valid() ||
          source.capacity == 0U || source.capacity > max_outstanding ||
          source.source_generation == 0U || source.tag_generation == 0U ||
          source.receiver_index >= receiver_count || !source.partition || !source.vertex) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      if (source.max_start_delay &&
          (!valid_max_start_delay_binding_shape(*source.max_start_delay) ||
           source.max_start_delay->source_clock != source.source_clock ||
           source.max_start_delay->reference_clock != source.reference_clock ||
           source.max_start_delay->conversion_id != source.source_to_reference_conversion_id)) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      if (!source.max_start_delay && (source.source_clock != source.reference_clock ||
                                      source.source_to_reference_conversion_id != 0U)) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      if (source.clock_health != nullptr &&
          (!source.port || source.clock_health->endpoint() != source.port)) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      if (reference_clock.valid() && reference_clock != source.reference_clock) {
        return make_unexpected(
            Error{ErrorCode::kInvalidArgument, "RUNTIME_SCHEDULING_DOMAIN_MISMATCH"});
      }
      reference_clock = source.reference_clock;
      if (std::any_of(states.begin(), states.end(), [&](const SourceState& existing) {
            return existing.binding.source == source.source;
          })) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      if (std::any_of(states.begin(), states.end(), [&](const SourceState& existing) {
            return existing.binding.source_order == source.source_order;
          })) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      if (tags[source.receiver_index] == nullptr) {
        tags[source.receiver_index] = std::make_unique<TagAllocator>(source.tag_generation);
      } else if (tags[source.receiver_index]->generation() != source.tag_generation) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      SourceState state{.binding = source};
      state.frozen_sequences.reserve(source.capacity);
      states.push_back(std::move(state));
    }
    return RuntimeCandidateCoordinator{
        std::move(states), std::move(tags), max_outstanding, phase_counters};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
#endif
}

std::size_t RuntimeCandidateCoordinator::find_source(SourceId source) const noexcept {
  const auto found = std::find_if(sources_.begin(), sources_.end(), [&](const SourceState& state) {
    if (phase_counters_ != nullptr) {
      phase_counters_->increment(RuntimePhaseCounter::kCandidateSourceLookupProbes);
    }
    return state.binding.source == source;
  });
  return found == sources_.end() ? sources_.size()
                                 : static_cast<std::size_t>(found - sources_.begin());
}

// Validate the ordered ingress protocol in one state-machine step so a terminal event, a malformed
// coordinate, or exhausted credit cannot partially mutate the source sequence and queue state.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> RuntimeCandidateCoordinator::accept_event(RuntimeEvent event) noexcept {
  const std::size_t source_slot = find_source(event.source);
  if (source_slot == sources_.size()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  SourceState& source = sources_[source_slot];
  if (source.closed) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  if (event.source_generation != source.binding.source_generation) {
    return make_unexpected(Error{ErrorCode::kStaleEndpoint});
  }
  if (source.last_source_sequence == std::numeric_limits<std::uint64_t>::max() ||
      event.source_sequence != source.last_source_sequence + 1U) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  if (event.kind == RuntimeEventKind::kSourceError) {
    if (event.error == ErrorCode::kSuccess) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    source.last_source_sequence = event.source_sequence;
    source.closed = true;
    return make_unexpected(Error{event.error});
  }
  if (event.kind == RuntimeEventKind::kFiniteSourceComplete) {
    if (event.error != ErrorCode::kSuccess) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    source.last_source_sequence = event.source_sequence;
    source.closed = true;
    return {};
  }
  const bool port_matches = source.binding.candidate_kind != RuntimeEventKind::kInputArrival
                                ? event.port == source.binding.port
                            : source.binding.port ? event.port == source.binding.port
                                                  : static_cast<bool>(event.port);
  const bool kind_matches = event.kind == source.binding.candidate_kind ||
                            (source.binding.candidate_kind == RuntimeEventKind::kInputArrival &&
                             (event.kind == RuntimeEventKind::kWindowTimeout ||
                              event.kind == RuntimeEventKind::kInputEndOfStream));
  if (!kind_matches || !port_matches || event.error != ErrorCode::kSuccess ||
      !event.reference_time.valid() ||
      event.reference_time.clock_id != source.binding.reference_clock) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  const bool event_health_snapshot = event.clock_health_generation != 0U;
  if (!valid_clock_health_disposition(event.clock_health_disposition) ||
      (!event_health_snapshot &&
       event.clock_health_disposition != EndpointClockHealthDisposition::kNotApplicable) ||
      (event.kind != RuntimeEventKind::kInputArrival &&
       (event_health_snapshot ||
        event.clock_health_disposition != EndpointClockHealthDisposition::kNotApplicable))) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  if (event.kind == RuntimeEventKind::kInputEndOfStream) {
    // EOS is control evidence, not a sample-capture witness. It retains the reference-clock
    // observation in both fields for deterministic ordering but never establishes a max_start_delay
    // source coordinate.
    if (!event.source_decision_time.valid() || event.source_decision_time != event.reference_time) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
  } else if (source.binding.max_start_delay) {
    auto coordinates = validate_event_coordinates(
        *source.binding.max_start_delay, event.source_decision_time, event.reference_time);
    if (!coordinates) {
      return make_unexpected(std::move(coordinates.error()));
    }
  } else if (!event.source_decision_time.valid() ||
             event.source_decision_time != event.reference_time) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  if (outstanding_count_ == max_outstanding_ || source.outstanding == source.binding.capacity) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }

  pending_.push_back(PendingEvent{.event = event, .source_slot = source_slot});
  source.last_source_sequence = event.source_sequence;
  ++source.outstanding;
  ++outstanding_count_;
  return {};
}

expected<std::size_t, Error> RuntimeCandidateCoordinator::ingest(PortBinding& binding) noexcept {
  const SourceId source = binding.source();
  if (source) {
    auto credit = source_credit_remaining(source);
    if (!credit) {
      return make_unexpected(std::move(credit).error());
    }
    return ingest(binding, source, *credit);
  }
  return ingest(binding, {}, max_outstanding_ - outstanding_count_);
}

expected<std::size_t, Error> RuntimeCandidateCoordinator::ingest(
    PortBinding& binding, SourceId source, std::size_t source_credit) noexcept {
  if (outstanding_count_ == max_outstanding_) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  const std::size_t available = std::min(max_outstanding_ - outstanding_count_, source_credit);
  std::size_t drained_count = 0U;
  while (drained_count < available) {
    // Drain one event at a time so a malformed or terminal event cannot cause
    // later events from the same removed prefix to disappear unprocessed.
    auto drained = binding.drain(std::span<RuntimeEvent>{ingress_scratch_}.first(1U));
    if (!drained) {
      return make_unexpected(std::move(drained.error()));
    }
    if (*drained > 1U) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (*drained == 0U) {
      break;
    }
    ++drained_count;
    if (source && ingress_scratch_.front().source != source) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    auto accepted = accept_event(ingress_scratch_.front());
    if (!accepted) {
      return make_unexpected(std::move(accepted.error()));
    }
  }
  return drained_count;
}

std::optional<TimePoint> RuntimeCandidateCoordinator::oldest_pending_reference() const noexcept {
  std::optional<TimePoint> oldest;
  for (const PendingEvent& pending : pending_) {
    // Compare on the same key freeze_next orders by, so this reports the coordinate that ordering
    // would actually pick rather than a differently-derived approximation of it.
    if (!oldest || pending.event.reference_time.timestamp_ns < oldest->timestamp_ns) {
      oldest = pending.event.reference_time;
    }
  }
  return oldest;
}

std::optional<TimePoint> RuntimeCandidateCoordinator::oldest_pending_nonterminal_trigger_reference(
    const VertexId owner, const SourceId witness_source, const PortId witness_port) const noexcept {
  if (!owner || !witness_source) {
    return std::nullopt;
  }
  std::optional<TimePoint> oldest;
  for (const PendingEvent& pending : pending_) {
    if (pending.source_slot >= sources_.size() ||
        sources_[pending.source_slot].binding.vertex != owner ||
        sources_[pending.source_slot].binding.source != witness_source ||
        (witness_port && pending.event.port != witness_port) ||
        pending.event.kind == RuntimeEventKind::kInputEndOfStream) {
      continue;
    }
    if (!oldest || pending.event.reference_time.timestamp_ns < oldest->timestamp_ns) {
      oldest = pending.event.reference_time;
    }
  }
  return oldest;
}

bool RuntimeCandidateCoordinator::no_pending_trigger_witness(
    const VertexId owner, const PortId witness_port) const noexcept {
  if (!owner || !witness_port) {
    return false;
  }
  for (const PendingEvent& pending : pending_) {
    if (pending.source_slot >= sources_.size() ||
        sources_[pending.source_slot].binding.vertex != owner ||
        pending.event.port != witness_port) {
      continue;
    }
    if (pending.event.kind != RuntimeEventKind::kInputEndOfStream) {
      return false;
    }
  }
  return true;
}

bool RuntimeCandidateCoordinator::end_of_stream_prefix_retired(
    const PendingEvent& terminal) const noexcept {
  if (terminal.event.kind != RuntimeEventKind::kInputEndOfStream) {
    return true;
  }
  // Source slots cannot change generation with outstanding work. Preserve the exact accepted
  // prefix, including retirement holes; selector payload retention is a separate EOS consumer.
  for (const auto& pending : pending_) {
    if (pending.source_slot == terminal.source_slot &&
        pending.event.source_sequence < terminal.event.source_sequence) {
      return false;
    }
  }
  for (const auto sequence : sources_[terminal.source_slot].frozen_sequences) {
    if (sequence < terminal.event.source_sequence) {
      return false;
    }
  }
  return true;
}

expected<FrozenRuntimeCandidate, Error> RuntimeCandidateCoordinator::freeze_next() noexcept {
  if (pending_.empty()) {
    return make_unexpected(Error{ErrorCode::kNotReady});
  }
  const auto precedes = [&](const PendingEvent& lhs, const PendingEvent& rhs) {
    if (phase_counters_ != nullptr) {
      phase_counters_->increment(RuntimePhaseCounter::kCandidateOrderComparisons);
    }
    const SourceState& lhs_source = sources_[lhs.source_slot];
    const SourceState& rhs_source = sources_[rhs.source_slot];
    return std::tuple{lhs.event.reference_time.timestamp_ns,
                      candidate_kind_order(lhs.event.kind),
                      lhs_source.binding.source_order,
                      lhs.event.source_sequence} < std::tuple{rhs.event.reference_time.timestamp_ns,
                                                              candidate_kind_order(rhs.event.kind),
                                                              rhs_source.binding.source_order,
                                                              rhs.event.source_sequence};
  };
  // The ordinary minimum is also the eligible minimum unless it is an EOS held behind its
  // accepted source prefix. Keep prefix scans out of ordinary candidate comparisons; only that
  // terminal case needs the bounded eligible-minimum fallback.
  auto selected = std::min_element(pending_.begin(), pending_.end(), precedes);
  if (!end_of_stream_prefix_retired(*selected)) {
    selected = pending_.end();
    for (auto candidate = pending_.begin(); candidate != pending_.end(); ++candidate) {
      if ((selected == pending_.end() || precedes(*candidate, *selected)) &&
          end_of_stream_prefix_retired(*candidate)) {
        selected = candidate;
      }
    }
  }
  if (selected == pending_.end()) {
    return make_unexpected(Error{ErrorCode::kNotReady});
  }

  std::int64_t logical_ns = selected->event.reference_time.timestamp_ns;
  std::uint32_t microstep = 0U;
  if (tag_time_initialized_ && logical_ns <= last_logical_ns_) {
    if (last_microstep_ == std::numeric_limits<std::uint32_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    // Independently accepted sources can become visible after a later reference coordinate was
    // already frozen. Preserve the semantic reference time in the witness while advancing the
    // process-local logical cursor instead of backdating the new activation.
    logical_ns = last_logical_ns_;
    microstep = last_microstep_ + 1U;
  }
  return freeze_eligible_pending(
      static_cast<std::size_t>(std::distance(pending_.begin(), selected)),
      TagTime{logical_ns, microstep, 0U});
}

// Constraint filtering and deterministic tag ordering form one atomic candidate selection pass.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<FrozenRuntimeCandidate, Error> RuntimeCandidateCoordinator::freeze_next_constrained(
    const std::span<const RuntimeCandidateInputFence> input_fences) noexcept {
  if (input_fences.empty()) {
    return freeze_next();
  }
  if (pending_.empty()) {
    return make_unexpected(Error{ErrorCode::kNotReady});
  }
  constexpr std::size_t kNoCandidate = std::numeric_limits<std::size_t>::max();
  std::size_t selected_index = kNoCandidate;
  const auto precedes = [&](const PendingEvent& lhs, const PendingEvent& rhs) {
    if (phase_counters_ != nullptr) {
      phase_counters_->increment(RuntimePhaseCounter::kCandidateOrderComparisons);
    }
    const SourceState& lhs_source = sources_[lhs.source_slot];
    const SourceState& rhs_source = sources_[rhs.source_slot];
    return std::tuple{lhs.event.reference_time.timestamp_ns,
                      candidate_kind_order(lhs.event.kind),
                      lhs_source.binding.source_order,
                      lhs.event.source_sequence} < std::tuple{rhs.event.reference_time.timestamp_ns,
                                                              candidate_kind_order(rhs.event.kind),
                                                              rhs_source.binding.source_order,
                                                              rhs.event.source_sequence};
  };
  const auto consider = [&](const std::size_t index) {
    if (index < pending_.size() &&
        (selected_index == kNoCandidate || precedes(pending_[index], pending_[selected_index])) &&
        end_of_stream_prefix_retired(pending_[index])) {
      selected_index = index;
    }
  };

  input_fence_head_scratch_.assign(input_fences.size(), kNoCandidate);
  for (const RuntimeCandidateInputFence& fence : input_fences) {
    if (!fence.owner || !fence.source || !fence.input) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
  }

  // Unfenced owners retain the ordinary candidate set. For a fenced owner, remember only the
  // first admitted event on each named input. A preceding terminal therefore keeps later data
  // hidden just as a preceding arrival does. This is one bounded pass over pending events; the
  // fence count is fixed by the compiled graph, not by queue depth.
  for (std::size_t index = 0U; index < pending_.size(); ++index) {
    const PendingEvent& pending = pending_[index];
    if (pending.source_slot >= sources_.size()) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const VertexId owner = sources_[pending.source_slot].binding.vertex;
    bool owner_fenced = false;
    for (std::size_t fence_index = 0U; fence_index < input_fences.size(); ++fence_index) {
      const RuntimeCandidateInputFence& fence = input_fences[fence_index];
      if (fence.owner != owner) {
        continue;
      }
      owner_fenced = true;
      if (input_fence_head_scratch_[fence_index] == kNoCandidate &&
          sources_[pending.source_slot].binding.source == fence.source &&
          pending.event.port == fence.input &&
          (pending.event.kind == RuntimeEventKind::kInputArrival ||
           pending.event.kind == RuntimeEventKind::kInputEndOfStream)) {
        input_fence_head_scratch_[fence_index] = index;
      }
    }
    if (!owner_fenced) {
      consider(index);
    }
  }

  // Each fenced input contributes exactly its oldest admitted arrival.
  for (const std::size_t head_index : input_fence_head_scratch_) {
    if (head_index != kNoCandidate &&
        pending_[head_index].event.kind == RuntimeEventKind::kInputArrival) {
      consider(head_index);
    }
  }
  if (selected_index == kNoCandidate) {
    return make_unexpected(Error{ErrorCode::kNotReady});
  }

  const PendingEvent& selected = pending_[selected_index];
  std::int64_t logical_ns = selected.event.reference_time.timestamp_ns;
  std::uint32_t microstep = 0U;
  if (tag_time_initialized_ && logical_ns <= last_logical_ns_) {
    if (last_microstep_ == std::numeric_limits<std::uint32_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    logical_ns = last_logical_ns_;
    microstep = last_microstep_ + 1U;
  }
  return freeze_eligible_pending(selected_index, TagTime{logical_ns, microstep, 0U});
}

expected<FrozenRuntimeCandidate, Error> RuntimeCandidateCoordinator::freeze_causal_next(
    const SourceId expected_source, const TagTime parent) noexcept {
  if (!expected_source || !parent.valid() || pending_.size() != 1U || outstanding_count_ != 1U ||
      !tag_time_initialized_ || last_logical_ns_ != parent.logical_ns ||
      last_microstep_ != parent.microstep ||
      parent.microstep == std::numeric_limits<std::uint32_t>::max()) {
    return make_unexpected(Error{parent.microstep == std::numeric_limits<std::uint32_t>::max()
                                     ? ErrorCode::kResourceExhausted
                                     : ErrorCode::kProtocolError});
  }
  const PendingEvent& pending = pending_.front();
  if (pending.source_slot >= sources_.size() || pending.event.source != expected_source ||
      sources_[pending.source_slot].outstanding != 1U) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  // The sole pending witness also owns the sole outstanding credit. No pending or frozen
  // predecessor can therefore exist, including if that witness is EOS.
  return freeze_eligible_pending(
      0U, TagTime{parent.logical_ns, static_cast<std::uint32_t>(parent.microstep + 1U), 0U});
}

expected<FrozenRuntimeCandidate, Error> RuntimeCandidateCoordinator::freeze_eligible_pending(
    const std::size_t pending_index, const TagTime tag_time) noexcept {
  if (pending_index >= pending_.size() || !tag_time.valid()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const PendingEvent& selected = pending_[pending_index];
  if (selected.source_slot >= sources_.size()) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  SourceState& source = sources_[selected.source_slot];
  TagAllocator* allocator = tags_[source.binding.receiver_index];
  if (allocator == nullptr) {
    return make_unexpected(Error{ErrorCode::kFailure});
  }
  auto tag = allocator->reserve_candidate();
  if (!tag) {
    return make_unexpected(std::move(tag.error()));
  }
  if (phase_counters_ != nullptr) {
    phase_counters_->increment(RuntimePhaseCounter::kCandidateTagTimeReservations);
  }

  const bool event_health_snapshot = selected.event.clock_health_generation != 0U;
  const EndpointClockHealthDisposition clock_health_disposition =
      event_health_snapshot                    ? selected.event.clock_health_disposition
      : source.binding.clock_health == nullptr ? EndpointClockHealthDisposition::kNotApplicable
                                               : source.binding.clock_health->current_disposition();
  const std::uint64_t clock_health_generation =
      event_health_snapshot ? selected.event.clock_health_generation
      : source.binding.clock_health == nullptr
          ? 0U
          : source.binding.clock_health->state().health_generation;
  FrozenRuntimeCandidate result{
      .witness = selected.event,
      .candidate =
          DispatcherCandidate{
              .ready =
                  scheduling::ReadyCandidateView{
                      .partition = source.binding.partition,
                      .execution_lane = source.binding.execution_lane,
                      .vertex = source.binding.vertex,
                      .tag_time = tag_time,
                      .plan_vertex_order = source.binding.plan_vertex_order,
                      .stable_source_id = source.binding.source,
                      .stable_port_id =
                          source.binding.port ? source.binding.port : selected.event.port,
                      .stable_candidate_tie = source.binding.stable_candidate_tie,
                      .active = true,
                  },
              .tag = *tag,
              .receiver_index = source.binding.receiver_index,
              .clock_health_disposition = clock_health_disposition,
          },
      .source_slot = selected.source_slot,
      .clock_health_generation = clock_health_generation,
  };
  source.frozen_sequences.push_back(selected.event.source_sequence);
  last_logical_ns_ = tag_time.logical_ns;
  last_microstep_ = tag_time.microstep;
  tag_time_initialized_ = true;
  if (phase_counters_ != nullptr) {
    phase_counters_->add(RuntimePhaseCounter::kCandidateOrderMoves,
                         pending_.size() - pending_index - 1U);
  }
  pending_.erase(pending_.begin() + static_cast<std::ptrdiff_t>(pending_index));
  return result;
}

expected<void, Error> RuntimeCandidateCoordinator::rearm_terminal_successor(
    FrozenRuntimeCandidate& candidate) noexcept {
  if (candidate.source_retired || candidate.witness.kind != RuntimeEventKind::kInputEndOfStream ||
      candidate.source_slot >= sources_.size() || !candidate.candidate.tag.valid() ||
      !candidate.candidate.ready.tag_time.valid() || !tag_time_initialized_ ||
      candidate.candidate.ready.tag_time.logical_ns != last_logical_ns_ ||
      candidate.candidate.ready.tag_time.microstep != last_microstep_ ||
      last_microstep_ == std::numeric_limits<std::uint32_t>::max()) {
    return make_unexpected(Error{last_microstep_ == std::numeric_limits<std::uint32_t>::max()
                                     ? ErrorCode::kResourceExhausted
                                     : ErrorCode::kInvalidArgument});
  }
  SourceState& source = sources_[candidate.source_slot];
  if (candidate.witness.source != source.binding.source ||
      candidate.witness.source_generation != source.binding.source_generation ||
      candidate.candidate.receiver_index != source.binding.receiver_index ||
      source.outstanding == 0U || outstanding_count_ == 0U ||
      std::count(source.frozen_sequences.begin(),
                 source.frozen_sequences.end(),
                 candidate.witness.source_sequence) != 1) {
    return make_unexpected(Error{ErrorCode::kStaleEndpoint});
  }
  TagAllocator* const allocator = tags_[source.binding.receiver_index];
  if (allocator == nullptr) {
    return make_unexpected(Error{ErrorCode::kFailure});
  }
  auto tag = allocator->reserve_candidate();
  if (!tag) {
    return make_unexpected(std::move(tag.error()));
  }
  ++last_microstep_;
  candidate.candidate.tag = *tag;
  candidate.candidate.ready.tag_time = TagTime{last_logical_ns_, last_microstep_, 0U};
  if (phase_counters_ != nullptr) {
    phase_counters_->increment(RuntimePhaseCounter::kCandidateTagTimeReservations);
  }
  return {};
}

expected<void, Error> RuntimeCandidateCoordinator::retire(
    const FrozenRuntimeCandidate& candidate) noexcept {
  if (candidate.source_slot >= sources_.size()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  SourceState& source = sources_[candidate.source_slot];
  if (candidate.witness.source != source.binding.source ||
      candidate.witness.source_generation != source.binding.source_generation ||
      candidate.candidate.receiver_index != source.binding.receiver_index ||
      !candidate.candidate.tag.valid()) {
    return make_unexpected(Error{ErrorCode::kStaleEndpoint});
  }
  const auto frozen = std::find(source.frozen_sequences.begin(),
                                source.frozen_sequences.end(),
                                candidate.witness.source_sequence);
  if (frozen == source.frozen_sequences.end() || source.outstanding == 0U ||
      outstanding_count_ == 0U) {
    return make_unexpected(Error{ErrorCode::kStaleEndpoint});
  }
  source.frozen_sequences.erase(frozen);
  --source.outstanding;
  --outstanding_count_;
  return {};
}

expected<void, Error> RuntimeCandidateCoordinator::publish_restart(
    std::size_t receiver_index) noexcept {
  if (receiver_index >= tags_.size() || tags_[receiver_index] == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  for (const auto& source : sources_) {
    if (source.binding.receiver_index == receiver_index &&
        (source.outstanding != 0U ||
         source.binding.source_generation == std::numeric_limits<std::uint64_t>::max())) {
      return make_unexpected(Error{source.outstanding != 0U ? ErrorCode::kBackpressured
                                                            : ErrorCode::kResourceExhausted});
    }
  }
  auto published = tags_[receiver_index]->publish_restart();
  if (!published) {
    return make_unexpected(std::move(published.error()));
  }
  for (auto& source : sources_) {
    if (source.binding.receiver_index == receiver_index) {
      ++source.binding.source_generation;
      source.last_source_sequence = 0U;
      source.closed = false;
    }
  }
  return {};
}

expected<std::size_t, Error> RuntimeCandidateCoordinator::cancel_receiver(
    std::size_t receiver_index, std::span<RuntimeEvent> canceled) noexcept {
  if (receiver_index >= tags_.size() || tags_[receiver_index] == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  std::size_t pending_count = 0U;
  for (const PendingEvent& pending : pending_) {
    if (sources_[pending.source_slot].binding.receiver_index == receiver_index) {
      ++pending_count;
    }
  }
  if (pending_count > canceled.size()) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  if (std::any_of(sources_.begin(), sources_.end(), [&](const SourceState& source) {
        return source.binding.receiver_index == receiver_index && !source.frozen_sequences.empty();
      })) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }

  std::size_t output = 0U;
  const auto retained =
      std::remove_if(pending_.begin(), pending_.end(), [&](PendingEvent& pending) {
        SourceState& source = sources_[pending.source_slot];
        if (source.binding.receiver_index != receiver_index) {
          return false;
        }
        canceled[output++] = pending.event;
        assert(source.outstanding != 0U);
        assert(outstanding_count_ != 0U);
        --source.outstanding;
        --outstanding_count_;
        return true;
      });
  pending_.erase(retained, pending_.end());
  return output;
}

expected<std::uint64_t, Error> RuntimeCandidateCoordinator::source_generation(
    SourceId source) const noexcept {
  const std::size_t slot = find_source(source);
  if (slot == sources_.size()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  return sources_[slot].binding.source_generation;
}

expected<std::size_t, Error> RuntimeCandidateCoordinator::source_receiver(
    SourceId source) const noexcept {
  const std::size_t slot = find_source(source);
  if (slot == sources_.size()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  return sources_[slot].binding.receiver_index;
}

expected<std::size_t, Error> RuntimeCandidateCoordinator::source_credit_remaining(
    SourceId source) const noexcept {
  const std::size_t slot = find_source(source);
  if (slot == sources_.size()) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  const SourceState& state = sources_[slot];
  if (state.outstanding > state.binding.capacity) {
    return make_unexpected(Error{ErrorCode::kFailure});
  }
  return state.closed ? 0U : state.binding.capacity - state.outstanding;
}

void RuntimeCandidateCoordinator::cancel_all() noexcept {
  pending_.clear();
  for (auto& source : sources_) {
    source.frozen_sequences.clear();
    source.outstanding = 0U;
  }
  outstanding_count_ = 0U;
}

ActivationSelectionState::ActivationSelectionState(std::size_t max_inputs,
                                                   std::size_t max_selected) {
  max_inputs_ = max_inputs;
  max_selected_ = max_selected;
  inputs_.reserve(max_inputs);
  selected_.reserve(max_selected);
}

expected<ActivationSelectionState, Error> ActivationSelectionState::create(
    std::size_t max_inputs, std::size_t max_selected) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
    return ActivationSelectionState{max_inputs, max_selected};
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
#else
  return ActivationSelectionState{max_inputs, max_selected};
#endif
}

expected<ActivationSelectionView, Error> ActivationSelectionState::begin(
    const FrozenRuntimeCandidate& candidate, std::span<const InputSelection> inputs,
    std::span<const TemporalSampleRecord> selected, TimePoint semantic_reference_time) noexcept {
  const EndpointClockHealthDisposition clock_health = candidate.candidate.clock_health_disposition;
  const TimePoint close_time = candidate.witness.reference_time;
  const TimePoint reference_time =
      semantic_reference_time.valid() ? semantic_reference_time : close_time;
  // A tuple can close when an independently accepted, older witness becomes visible after a
  // newer member head. The coordinator's monotonic TagTime proves that late-visibility ordering;
  // retain both physical coordinates instead of restamping either one. All ordinary candidates
  // keep the stricter semantic-reference-at-or-before-close invariant and its single comparison.
  const bool invalid_late_semantic_reference =
      reference_time.timestamp_ns > close_time.timestamp_ns &&
      (candidate.candidate.ready.tag_time.microstep == 0U ||
       candidate.candidate.ready.tag_time.logical_ns < reference_time.timestamp_ns);
  if (active_ || !candidate.candidate.tag.valid() || !candidate.witness.reference_time.valid() ||
      !candidate.candidate.ready.tag_time.valid() || inputs.size() > max_inputs_ ||
      selected.size() > max_selected_ || !valid_clock_health_disposition(clock_health) ||
      !reference_time.valid() || reference_time.clock_id != close_time.clock_id ||
      invalid_late_semantic_reference) {
    return make_unexpected(Error{inputs.size() > max_inputs_ || selected.size() > max_selected_
                                     ? ErrorCode::kResourceExhausted
                                     : ErrorCode::kInvalidArgument});
  }
  if (clock_health == EndpointClockHealthDisposition::kSkip ||
      clock_health == EndpointClockHealthDisposition::kFailRun) {
    return make_unexpected(Error{clock_health == EndpointClockHealthDisposition::kSkip
                                     ? ErrorCode::kNotReady
                                     : ErrorCode::kFailure});
  }
  inputs_.assign(inputs.begin(), inputs.end());
  selected_.assign(selected.begin(), selected.end());
  tag_ = candidate.candidate.tag;
  tag_time_ = candidate.candidate.ready.tag_time;
  reference_time_ = reference_time;
  close_time_ = close_time;
  disposition_ = aggregate_missing_inputs(inputs_);
  clock_health_disposition_ = clock_health;
  if (clock_health_disposition_ == EndpointClockHealthDisposition::kDegraded &&
      disposition_ == MissingAggregateDisposition::kNominal) {
    disposition_ = MissingAggregateDisposition::kDegraded;
  }
  active_ = true;
  return view(generation_);
}

expected<ActivationSelectionView, Error> ActivationSelectionState::view(
    std::uint64_t generation) const noexcept {
  if (!active_ || generation == 0U || generation != generation_) {
    return make_unexpected(Error{ErrorCode::kStaleEndpoint});
  }
  return ActivationSelectionView{
      .tag = tag_,
      .tag_time = tag_time_,
      .reference_time = reference_time_,
      .close_time = close_time_,
      .disposition = disposition_,
      .clock_health_disposition = clock_health_disposition_,
      .inputs = inputs_,
      .selected = selected_,
      .generation = generation_,
  };
}

expected<void, Error> ActivationSelectionState::retire(Tag tag) noexcept {
  if (!active_ || tag != tag_) {
    return make_unexpected(Error{ErrorCode::kStaleEndpoint});
  }
  if (generation_ == std::numeric_limits<std::uint64_t>::max()) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  inputs_.clear();
  selected_.clear();
  tag_ = {};
  tag_time_ = {};
  reference_time_ = {};
  close_time_ = {};
  disposition_ = MissingAggregateDisposition::kNoFire;
  clock_health_disposition_ = EndpointClockHealthDisposition::kNotApplicable;
  active_ = false;
  ++generation_;
  return {};
}

Dispatcher::Dispatcher(const SchedulerPlanRecord& plan,
                       std::unique_ptr<scheduling::ReadyPolicy> policy,
                       scheduling::ReadyPolicyState policy_state, std::size_t max_candidates)
    : plan_(&plan),
      policy_(std::move(policy)),
      policy_state_(policy_state),
      max_candidates_(max_candidates) {
  ready_.reserve(max_candidates);
  tags_.reserve(max_candidates);
  receiver_indices_.reserve(max_candidates);
  clock_health_dispositions_.reserve(max_candidates);
}

expected<Dispatcher, Error> Dispatcher::create(const SchedulerPlanRecord& plan,
                                               std::size_t max_candidates) {
  auto catalog = scheduling::ReadyPolicyCatalog::create_default_builtins();
  if (!catalog) {
    return make_unexpected(std::move(catalog.error()));
  }
  auto policy = catalog->make(plan.ready_policy);
  if (!policy) {
    return make_unexpected(std::move(policy.error()));
  }
  scheduling::ReadyPolicyState policy_state;
  if (max_candidates != 0U) {
    auto initialized = policy.value()->initialize(
        scheduling::SchedulerPlanView{
            .scheduler = &plan,
            .max_ready_candidates = max_candidates,
            .available_capabilities = scheduling::kAllReadyPolicyCapabilities,
        },
        policy_state);
    if (!initialized) {
      return make_unexpected(std::move(initialized.error()));
    }
  }
  return Dispatcher{plan, std::move(policy).value(), policy_state, max_candidates};
}

expected<void, Error> Dispatcher::freeze(std::span<const DispatcherCandidate> candidates) {
  if (policy_state_.activation_in_progress || active_count_ != 0U ||
      candidates.size() > max_candidates_) {
    return make_unexpected(Error{candidates.size() > max_candidates_
                                     ? ErrorCode::kResourceExhausted
                                     : ErrorCode::kInvalidArgument});
  }
  // A completed freeze set leaves inactive slots behind for stable selection
  // indices. Reclaim those preallocated slots only between freeze cycles.
  ready_.clear();
  tags_.clear();
  receiver_indices_.clear();
  clock_health_dispositions_.clear();
  active_count_ = 0U;
  for (const auto& candidate : candidates) {
    if (!candidate.ready.active || !candidate.ready.vertex || !candidate.tag.valid() ||
        candidate.ready.execution_lane >= plan_->execution_lane_count ||
        !valid_clock_health_disposition(candidate.clock_health_disposition)) {
      ready_.clear();
      tags_.clear();
      receiver_indices_.clear();
      clock_health_dispositions_.clear();
      active_count_ = 0U;
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (candidate.clock_health_disposition == EndpointClockHealthDisposition::kFailRun) {
      ready_.clear();
      tags_.clear();
      receiver_indices_.clear();
      clock_health_dispositions_.clear();
      active_count_ = 0U;
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    scheduling::ReadyCandidateView ready = candidate.ready;
    if (candidate.clock_health_disposition == EndpointClockHealthDisposition::kSkip) {
      ready.active = false;
    } else {
      ++active_count_;
    }
    ready_.push_back(ready);
    tags_.push_back(candidate.tag);
    receiver_indices_.push_back(candidate.receiver_index);
    clock_health_dispositions_.push_back(candidate.clock_health_disposition);
  }
  return {};
}

expected<DispatcherSelection, Error> Dispatcher::select() noexcept {
  if (policy_state_.activation_in_progress) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if (active_count_ == 0U) {
    return make_unexpected(Error{ErrorCode::kNotReady});
  }
  const scheduling::ReadyCandidateIndex selected =
      policy_->select(scheduling::ReadySetView{ready_}, policy_state_);
  if (!selected || selected.value >= ready_.size() || !ready_[selected.value].active) {
    return make_unexpected(Error{ErrorCode::kFailure});
  }
  const scheduling::ReadyCandidateView candidate = ready_[selected.value];
  policy_->on_activation_started(candidate, policy_state_);
  return DispatcherSelection{
      .slot = selected.value,
      .receiver_index = receiver_indices_[selected.value],
      .ready = candidate,
      .tag = tags_[selected.value],
      .clock_health_disposition = clock_health_dispositions_[selected.value],
  };
}

expected<void, Error> Dispatcher::complete(const DispatcherSelection& selection) noexcept {
  if (!policy_state_.activation_in_progress || selection.slot >= ready_.size() ||
      !ready_[selection.slot].active ||
      receiver_indices_[selection.slot] != selection.receiver_index ||
      tags_[selection.slot] != selection.tag ||
      clock_health_dispositions_[selection.slot] != selection.clock_health_disposition) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  policy_->on_activation_completed(ready_[selection.slot], policy_state_);
  ready_[selection.slot].active = false;
  --active_count_;
  return {};
}

expected<std::size_t, Error> Dispatcher::cancel_receiver(std::size_t receiver_index) noexcept {
  if (policy_state_.activation_in_progress) {
    return make_unexpected(Error{ErrorCode::kBackpressured});
  }
  std::size_t canceled = 0U;
  for (std::size_t slot = 0U; slot < ready_.size(); ++slot) {
    if (receiver_indices_[slot] != receiver_index || !ready_[slot].active) {
      continue;
    }
    ready_[slot].active = false;
    --active_count_;
    ++canceled;
  }
  return canceled;
}

void Dispatcher::cancel_all() noexcept {
  if (policy_state_.activation_in_progress) {
    const auto active = std::find_if(
        ready_.begin(), ready_.end(), [](const auto& candidate) { return candidate.active; });
    if (active != ready_.end()) {
      policy_->on_activation_completed(*active, policy_state_);
    }
  }
  for (auto& candidate : ready_) {
    candidate.active = false;
  }
  active_count_ = 0U;
}

}  // namespace holoscan::detail
