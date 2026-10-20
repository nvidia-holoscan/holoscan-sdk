/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/predicate_input_view.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/temporal_contract_types.hpp>
#include <holoscan/core/temporal_plan.hpp>

#include "core/predicate_input_view_internal.hpp"

namespace holoscan::detail {

enum class PredicateCandidateDisposition : std::uint8_t {
  kNominalCompute,
  kDegradedCompute,
  kNoFire,
  kEvaluatorError,
};

enum class PredicateResetReason : std::uint8_t {
  kPlanGeneration,
  kColdRunStart,
  kOperatorRestart,
  kSessionRebind,
  kProducerEpoch,
  kClockDiscontinuity,
};

/// Record one accepted predicate-state reset boundary.
///
/// The generation is the newly active state generation. The reason is kept
/// separately from later evaluation outcomes so false, missing, evaluator
/// failure, and callback failure cannot be misreported as reset causes.
struct PredicateResetRecord {
  PredicateResetReason reason{PredicateResetReason::kColdRunStart};
  std::uint64_t generation{};

  friend constexpr bool operator==(PredicateResetRecord, PredicateResetRecord) noexcept = default;
};

enum class PredicateAggregateDisposition : std::uint8_t {
  kNominal,
  kDegraded,
  kNoFire,
};

struct PredicateFrozenInput {
  PortId port{};
  InputSelection selection{};
  const void* typed_view{};
};

struct PredicateFrozenSource {
  SourceId source{};
  std::uint64_t source_sequence{};
  PortId driver_port{};
  TimePoint reference_time{};
  Tag tag{};
  TagTime tag_time{};
};

struct PredicateComputeSelectionRequest {
  std::size_t planned_input_count{};
  void* state{};
  expected<std::span<const InputSelection>, Error> (*select)(void*, TimePoint) noexcept {};
};

template <std::size_t MaxPredicateInputs, std::size_t MaxComputeInputs>
struct PredicateCandidateDecision {
  PredicateCandidateDisposition disposition{PredicateCandidateDisposition::kNoFire};
  bool evaluator_called{};
  bool value{};
  bool firing_accepted{};
  std::array<PredicateInputSelection, MaxPredicateInputs> predicate_selections{};
  std::array<const void*, MaxPredicateInputs> predicate_views{};
  std::size_t predicate_input_count{};
  std::array<InputSelection, MaxComputeInputs> compute_selections{};
  std::size_t compute_input_count{};
  Error evaluator_error{ErrorCode::kSuccess};
  PredicateFrozenSource source{};
  std::string_view predicate_id;
  OperatorInstanceId operator_id{};
  std::uint64_t state_generation{};
  bool prior_truth{};
  PredicateResetReason last_reset_reason{PredicateResetReason::kColdRunStart};
  PredicateResetRecord reset_record{};
};

template <std::size_t MaxPredicateInputs, std::size_t MaxComputeInputs>
class PredicateEvaluatorEngine {
 public:
  PredicateEvaluatorEngine(const PredicateEvaluatorEngine&) = delete;
  PredicateEvaluatorEngine& operator=(const PredicateEvaluatorEngine&) = delete;
  PredicateEvaluatorEngine(PredicateEvaluatorEngine&&) noexcept = default;
  PredicateEvaluatorEngine& operator=(PredicateEvaluatorEngine&&) noexcept = default;

  PredicateEvaluatorEngine(std::unique_ptr<PredicateCallableStorage> callable, bool expects_driver,
                           std::size_t additional_input_count, PredicateFiring firing,
                           std::uint64_t generation)
      : callable_(std::move(callable)),
        expected_input_count_(additional_input_count + (expects_driver ? 1U : 0U)),
        firing_(firing),
        active_generation_(generation) {}

  /// Create a session evaluator from compiler-frozen predicate records.
  [[nodiscard]] static expected<PredicateEvaluatorEngine, Error> create(
      const PredicateTriggerPlanRecord& trigger, std::span<const PredicateInputPlanRecord> inputs,
      const PredicateCallablePlanRecord& callable_plan,
      std::unique_ptr<PredicateCallableStorage> callable) {
    if (!valid_plan(trigger, inputs, callable_plan) || !callable) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    PredicateEvaluatorEngine engine{
        std::move(callable),
        trigger.source == PredicateEvaluationSourceKind::kInputUpdate,
        inputs.size() - (trigger.source == PredicateEvaluationSourceKind::kInputUpdate ? 1U : 0U),
        trigger.firing,
        trigger.state_generation};
    engine.plan_bound_ = true;
    engine.source_kind_ = trigger.source;
    engine.operator_id_ = trigger.owner.id;
    engine.driver_ = trigger.driver;
    engine.clock_ = trigger.clock;
    engine.predicate_id_ = trigger.id;
    return engine;
  }

  // Evaluation validates and commits one bounded predicate transaction; separating the stages
  // would obscure which state is provisional and which state is committed.
  [[nodiscard]] expected<PredicateCandidateDecision<MaxPredicateInputs, MaxComputeInputs>, Error>
  evaluate(  // NOLINT(readability-function-cognitive-complexity)
      PredicateFrozenSource source, std::span<const PredicateFrozenInput> predicate_inputs = {},
      PredicateComputeSelectionRequest compute = {}, PredicateReceiver self_receiver = {}) {
    if (!callable_ || active_generation_ == 0U || !source.reference_time.valid() ||
        !source.tag.valid() || source.tag.generation != active_generation_ ||
        !source.tag_time.valid() || predicate_inputs.size() != expected_input_count_ ||
        predicate_inputs.size() > MaxPredicateInputs ||
        (firing_ != PredicateFiring::kOnFalseToTrue &&
         firing_ != PredicateFiring::kOnEveryTrueEvaluation) ||
        compute.planned_input_count > MaxComputeInputs ||
        (compute.planned_input_count != 0U && compute.select == nullptr) ||
        !valid_bound_source(source, predicate_inputs)) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }

    PredicateCandidateDecision<MaxPredicateInputs, MaxComputeInputs> decision;
    decision.source = source;
    decision.predicate_id = predicate_id_;
    decision.operator_id = operator_id_;
    decision.state_generation = active_generation_;
    decision.prior_truth = previous_truth_;
    decision.last_reset_reason = last_reset_reason_;
    decision.reset_record = last_reset_record_;
    decision.predicate_input_count = predicate_inputs.size();
    decision.compute_input_count = compute.planned_input_count;
    std::fill_n(
        decision.compute_selections.begin(),
        decision.compute_input_count,
        InputSelection{InputState::kNotSelected, SelectionReason::kNotSelectedForActivation});
    for (std::size_t index = 0; index < predicate_inputs.size(); ++index) {
      const auto& input = predicate_inputs[index];
      const bool has_view = input.selection.state == InputState::kSelected ||
                            input.selection.state == InputState::kReused;
      if (has_view != (input.typed_view != nullptr) || !valid_selection(input.selection) ||
          input.selection.state == InputState::kNotSelected) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      decision.predicate_selections[index] = {input.port, input.selection};
      decision.predicate_views[index] = input.typed_view;
    }

    const auto predicate_disposition = aggregate(predicate_inputs);
    if (predicate_disposition == PredicateAggregateDisposition::kDegraded) {
      decision.disposition = PredicateCandidateDisposition::kDegradedCompute;
      return decision;
    }
    if (predicate_disposition == PredicateAggregateDisposition::kNoFire) {
      decision.disposition = PredicateCandidateDisposition::kNoFire;
      return decision;
    }

    auto context = PredicateBindingAccess::context({
        .reference_time = source.reference_time,
        .tag = source.tag,
        .tag_time = source.tag_time,
        .selections = {decision.predicate_selections.data(), decision.predicate_input_count},
        .active_generation = &active_generation_,
        .expected_generation = active_generation_,
    });
    if (!context) {
      return make_unexpected(std::move(context).error());
    }
    decision.evaluator_called = true;
    auto evaluated = callable_->evaluate(
        *context, {decision.predicate_views.data(), decision.predicate_input_count}, self_receiver);
    if (!evaluated) {
      decision.disposition = PredicateCandidateDisposition::kEvaluatorError;
      decision.evaluator_error = std::move(evaluated).error();
      return decision;
    }

    decision.value = *evaluated;
    const bool firing_candidate =
        decision.value && (firing_ == PredicateFiring::kOnEveryTrueEvaluation || !previous_truth_);
    // Predicate state and remembered truth are committed by successful
    // evaluation, before the separately reached compute-input phase. A later
    // compute-only suppression cannot retroactively turn success into an
    // evaluator rollback.
    callable_->commit_candidate();
    previous_truth_ = decision.value;
    if (!firing_candidate) {
      decision.disposition = PredicateCandidateDisposition::kNoFire;
      return decision;
    }

    std::span<const InputSelection> compute_inputs;
    if (compute.planned_input_count != 0U) {
      auto selected = compute.select(compute.state, source.reference_time);
      if (!selected) {
        if (selected.error().code == ErrorCode::kNotReady) {
          decision.disposition = PredicateCandidateDisposition::kNoFire;
          return decision;
        }
        return make_unexpected(std::move(selected).error());
      }
      if (selected->size() != compute.planned_input_count) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      compute_inputs = *selected;
      if (std::any_of(compute_inputs.begin(), compute_inputs.end(), [](InputSelection selection) {
            return selection.state == InputState::kNotSelected || !valid_selection(selection);
          })) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      std::copy(compute_inputs.begin(), compute_inputs.end(), decision.compute_selections.begin());
    }
    const auto compute_disposition = aggregate(compute_inputs);
    if (compute_disposition == PredicateAggregateDisposition::kNoFire) {
      decision.disposition = PredicateCandidateDisposition::kNoFire;
      return decision;
    }
    decision.firing_accepted = true;
    decision.disposition = compute_disposition == PredicateAggregateDisposition::kDegraded
                               ? PredicateCandidateDisposition::kDegradedCompute
                               : PredicateCandidateDisposition::kNominalCompute;
    return decision;
  }

  void reset(PredicateResetReason reason, std::uint64_t generation) {
    if (callable_) {
      callable_->reset();
    }
    previous_truth_ = false;
    active_generation_ = generation;
    last_reset_reason_ = reason;
    last_reset_record_ = {.reason = reason, .generation = generation};
  }

  [[nodiscard]] bool previous_truth() const noexcept { return previous_truth_; }
  [[nodiscard]] std::uint64_t generation() const noexcept { return active_generation_; }
  [[nodiscard]] PredicateResetReason last_reset_reason() const noexcept {
    return last_reset_reason_;
  }
  [[nodiscard]] PredicateResetRecord last_reset_record() const noexcept {
    return last_reset_record_;
  }

 private:
  // Admission intentionally validates the complete cross-record predicate contract in one pass.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] static bool valid_plan(const PredicateTriggerPlanRecord& trigger,
                                       std::span<const PredicateInputPlanRecord> inputs,
                                       const PredicateCallablePlanRecord& callable) noexcept {
    if (trigger.id.empty() || !trigger.owner.plan_bound() || trigger.graph_generation == 0U ||
        trigger.graph_generation != trigger.owner.plan_generation ||
        trigger.state_generation != trigger.graph_generation || trigger.initial_truth ||
        !trigger.successful_evaluation_commits_truth || !trigger.missing_or_error_preserves_truth ||
        !trigger.candidate_state_is_transactional || !trigger.compute_adopts_evaluator_snapshot ||
        !trigger.consumes_reserved_candidate_tag ||
        trigger.reset_policy != TemporalPredicateResetPolicy::kResetEventsV1 ||
        (trigger.firing != PredicateFiring::kOnFalseToTrue &&
         trigger.firing != PredicateFiring::kOnEveryTrueEvaluation) ||
        trigger.missing_precedence != "safe_default_then_skip_or_exhausted_reuse" ||
        callable.predicate_id != trigger.id ||
        callable.disposition == PredicateCallableDisposition::kRejected ||
        (callable.result_kind != PredicateResultKind::kBool &&
         callable.result_kind != PredicateResultKind::kExpectedBoolError) ||
        inputs.size() > MaxPredicateInputs) {
      return false;
    }
    const bool input_driven = trigger.source == PredicateEvaluationSourceKind::kInputUpdate;
    const bool clock_driven = trigger.source == PredicateEvaluationSourceKind::kClock;
    if ((!input_driven && !clock_driven) ||
        (input_driven && (!trigger.driver || trigger.clock.valid() || trigger.period_ns != 0 ||
                          trigger.phase.has_value() || inputs.empty())) ||
        (clock_driven && (trigger.driver || !trigger.clock.valid() || trigger.period_ns <= 0))) {
      return false;
    }
    for (std::size_t index = 0U; index < inputs.size(); ++index) {
      const auto& input = inputs[index];
      if (input.predicate_id != trigger.id || !input.input || input.argument_ordinal != index ||
          !input.reference_clock.valid() || input.cache_capacity == 0U ||
          !input.capture_time_required ||
          (input.selector != InputSelectorKind::kExact &&
           input.selector != InputSelectorKind::kLatestWithin) ||
          (input.selector == InputSelectorKind::kExact && input.max_age_ns != 0) ||
          (input.selector == InputSelectorKind::kLatestWithin && input.max_age_ns <= 0) ||
          (input.missing == TemporalRecordMissingKind::kReuseLast && input.reuse_max_age_ns <= 0) ||
          (input.missing != TemporalRecordMissingKind::kReuseLast && input.reuse_max_age_ns != 0)) {
        return false;
      }
      const bool expected_driver = input_driven && index == 0U;
      if (input.implicit_driver != expected_driver ||
          (expected_driver &&
           (input.input != trigger.driver || input.selector != InputSelectorKind::kExact ||
            input.missing != TemporalRecordMissingKind::kUnspecified))) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool valid_bound_source(
      const PredicateFrozenSource& source,
      std::span<const PredicateFrozenInput> inputs) const noexcept {
    if (!plan_bound_) {
      return true;
    }
    if (!source.source || source.source_sequence == 0U) {
      return false;
    }
    if (source_kind_ == PredicateEvaluationSourceKind::kInputUpdate) {
      return source.driver_port == driver_ && !inputs.empty() && inputs.front().port == driver_;
    }
    return !source.driver_port && source.reference_time.clock_id == clock_;
  }

  [[nodiscard]] static bool valid_selection(InputSelection selection) noexcept {
    switch (selection.state) {
      case InputState::kSelected:
        return selection.reason == SelectionReason::kNone;
      case InputState::kReused:
      case InputState::kSafeDefault:
      case InputState::kAbsent:
        return selection.reason != SelectionReason::kNone &&
               selection.reason != SelectionReason::kNotSelectedForActivation;
      case InputState::kNotSelected:
        return selection.reason == SelectionReason::kNotSelectedForActivation;
    }
    return false;
  }

  template <typename Input>
  [[nodiscard]] static PredicateAggregateDisposition aggregate(
      std::span<const Input> inputs) noexcept {
    bool safe = false;
    bool absent = false;
    for (const auto& input : inputs) {
      const InputSelection selection = [&]() {
        if constexpr (std::same_as<Input, PredicateFrozenInput>) {
          return input.selection;
        } else {
          return input;
        }
      }();
      safe = safe || selection.state == InputState::kSafeDefault;
      absent = absent || selection.state == InputState::kAbsent;
    }
    if (safe) {
      return PredicateAggregateDisposition::kDegraded;
    }
    return absent ? PredicateAggregateDisposition::kNoFire
                  : PredicateAggregateDisposition::kNominal;
  }

  std::unique_ptr<PredicateCallableStorage> callable_;
  std::size_t expected_input_count_{};
  PredicateFiring firing_{PredicateFiring::kOnFalseToTrue};
  std::uint64_t active_generation_{};
  bool previous_truth_{};
  PredicateResetReason last_reset_reason_{PredicateResetReason::kColdRunStart};
  PredicateResetRecord last_reset_record_{PredicateResetReason::kColdRunStart, active_generation_};
  bool plan_bound_{};
  PredicateEvaluationSourceKind source_kind_{PredicateEvaluationSourceKind::kInputUpdate};
  OperatorInstanceId operator_id_{};
  PortId driver_{};
  ClockId clock_{};
  std::string predicate_id_;
};

/// Borrow one evaluation decision from a plan-sized Runtime predicate engine.
///
/// The spans remain valid until the owning engine is evaluated or reset again. Their storage is
/// allocated once during cold construction from compiler-frozen input counts.
struct RuntimePredicateCandidateDecision {
  PredicateCandidateDisposition disposition{PredicateCandidateDisposition::kNoFire};
  bool evaluator_called{};
  bool value{};
  bool firing_accepted{};
  std::span<const PredicateInputSelection> predicate_selections;
  std::span<const void* const> predicate_views;
  std::span<const InputSelection> compute_selections;
  Error evaluator_error{ErrorCode::kSuccess};
  PredicateFrozenSource source{};
  std::string_view predicate_id;
  OperatorInstanceId operator_id{};
  std::uint64_t state_generation{};
  bool prior_truth{};
  PredicateResetReason last_reset_reason{PredicateResetReason::kColdRunStart};
  PredicateResetRecord reset_record{};
};

/// Evaluate one predicate with exact compiler-frozen, plan-sized scratch storage.
class RuntimePredicateEvaluatorEngine {
 public:
  RuntimePredicateEvaluatorEngine(const RuntimePredicateEvaluatorEngine&) = delete;
  RuntimePredicateEvaluatorEngine& operator=(const RuntimePredicateEvaluatorEngine&) = delete;
  RuntimePredicateEvaluatorEngine(RuntimePredicateEvaluatorEngine&&) noexcept = default;
  RuntimePredicateEvaluatorEngine& operator=(RuntimePredicateEvaluatorEngine&&) noexcept = default;

  [[nodiscard]] static expected<RuntimePredicateEvaluatorEngine, Error> create(
      const PredicateTriggerPlanRecord& trigger, std::span<const PredicateInputPlanRecord> inputs,
      std::size_t compute_input_count, const PredicateCallablePlanRecord& callable_plan,
      std::unique_ptr<PredicateCallableStorage> callable) {
    if (!valid_plan(trigger, inputs, callable_plan) || !callable) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      RuntimePredicateEvaluatorEngine engine{
          std::move(callable), trigger, inputs.size(), compute_input_count};
      return engine;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (const std::bad_alloc&) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
#endif
  }

  // Evaluation validates and commits one bounded predicate transaction; separating the stages
  // would obscure which state is provisional and which state is committed.
  [[nodiscard]] expected<RuntimePredicateCandidateDecision, Error>
  evaluate(  // NOLINT(readability-function-cognitive-complexity)
      PredicateFrozenSource source, std::span<const PredicateFrozenInput> predicate_inputs,
      PredicateComputeSelectionRequest compute = {}, PredicateReceiver self_receiver = {}) {
    if (!callable_ || active_generation_ == 0U || !source.reference_time.valid() ||
        !source.tag.valid() || source.tag.generation != active_generation_ ||
        !source.tag_time.valid() || predicate_inputs.size() != expected_input_count_ ||
        compute.planned_input_count != compute_input_count_ ||
        (compute_input_count_ != 0U && compute.select == nullptr) ||
        !valid_bound_source(source, predicate_inputs)) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }

    predicate_selections_.clear();
    predicate_views_.clear();
    compute_selections_.assign(
        compute_input_count_,
        InputSelection{InputState::kNotSelected, SelectionReason::kNotSelectedForActivation});
    for (const auto& input : predicate_inputs) {
      const bool has_view = input.selection.state == InputState::kSelected ||
                            input.selection.state == InputState::kReused;
      if (has_view != (input.typed_view != nullptr) || !valid_selection(input.selection) ||
          input.selection.state == InputState::kNotSelected) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      predicate_selections_.push_back({input.port, input.selection});
      predicate_views_.push_back(input.typed_view);
    }

    RuntimePredicateCandidateDecision decision = base_decision(source);
    const auto predicate_disposition = aggregate(predicate_inputs);
    if (predicate_disposition == PredicateAggregateDisposition::kDegraded) {
      decision.disposition = PredicateCandidateDisposition::kDegradedCompute;
      return decision;
    }
    if (predicate_disposition == PredicateAggregateDisposition::kNoFire) {
      decision.disposition = PredicateCandidateDisposition::kNoFire;
      return decision;
    }

    auto context = PredicateBindingAccess::context({
        .reference_time = source.reference_time,
        .tag = source.tag,
        .tag_time = source.tag_time,
        .selections = predicate_selections_,
        .active_generation = &active_generation_,
        .expected_generation = active_generation_,
    });
    if (!context) {
      return make_unexpected(std::move(context).error());
    }
    decision.evaluator_called = true;
    auto evaluated = callable_->evaluate(*context, predicate_views_, self_receiver);
    if (!evaluated) {
      decision.disposition = PredicateCandidateDisposition::kEvaluatorError;
      decision.evaluator_error = std::move(evaluated).error();
      return decision;
    }

    decision.value = *evaluated;
    const bool firing_candidate =
        decision.value && (firing_ == PredicateFiring::kOnEveryTrueEvaluation || !previous_truth_);
    callable_->commit_candidate();
    previous_truth_ = decision.value;
    if (!firing_candidate) {
      decision.disposition = PredicateCandidateDisposition::kNoFire;
      return decision;
    }

    if (compute_input_count_ != 0U) {
      auto selected = compute.select(compute.state, source.reference_time);
      if (!selected) {
        if (selected.error().code == ErrorCode::kNotReady) {
          decision.disposition = PredicateCandidateDisposition::kNoFire;
          return decision;
        }
        return make_unexpected(std::move(selected).error());
      }
      if (selected->size() != compute_input_count_ ||
          std::any_of(selected->begin(), selected->end(), [](InputSelection selection) {
            return selection.state == InputState::kNotSelected || !valid_selection(selection);
          })) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      std::copy(selected->begin(), selected->end(), compute_selections_.begin());
    }
    const auto compute_disposition = aggregate<InputSelection>(compute_selections_);
    if (compute_disposition == PredicateAggregateDisposition::kNoFire) {
      decision.disposition = PredicateCandidateDisposition::kNoFire;
      return decision;
    }
    decision.firing_accepted = true;
    decision.disposition = compute_disposition == PredicateAggregateDisposition::kDegraded
                               ? PredicateCandidateDisposition::kDegradedCompute
                               : PredicateCandidateDisposition::kNominalCompute;
    decision.compute_selections = compute_selections_;
    return decision;
  }

  void reset(PredicateResetReason reason, std::uint64_t generation) {
    if (callable_) {
      callable_->reset();
    }
    previous_truth_ = false;
    active_generation_ = generation;
    last_reset_reason_ = reason;
    last_reset_record_ = {.reason = reason, .generation = generation};
  }

  [[nodiscard]] bool previous_truth() const noexcept { return previous_truth_; }
  [[nodiscard]] std::uint64_t generation() const noexcept { return active_generation_; }
  [[nodiscard]] PredicateResetReason last_reset_reason() const noexcept {
    return last_reset_reason_;
  }
  [[nodiscard]] PredicateResetRecord last_reset_record() const noexcept {
    return last_reset_record_;
  }

 private:
  RuntimePredicateEvaluatorEngine(std::unique_ptr<PredicateCallableStorage> callable,
                                  const PredicateTriggerPlanRecord& trigger,
                                  std::size_t predicate_input_count,
                                  std::size_t compute_input_count)
      : callable_(std::move(callable)),
        expected_input_count_(predicate_input_count),
        compute_input_count_(compute_input_count),
        firing_(trigger.firing),
        active_generation_(trigger.state_generation),
        source_kind_(trigger.source),
        operator_id_(trigger.owner.id),
        driver_(trigger.driver),
        clock_(trigger.clock),
        predicate_id_(trigger.id) {
    predicate_selections_.reserve(predicate_input_count);
    predicate_views_.reserve(predicate_input_count);
    compute_selections_.reserve(compute_input_count);
  }

  [[nodiscard]] RuntimePredicateCandidateDecision base_decision(
      PredicateFrozenSource source) const noexcept {
    return RuntimePredicateCandidateDecision{
        .predicate_selections = predicate_selections_,
        .predicate_views = predicate_views_,
        .compute_selections = compute_selections_,
        .source = source,
        .predicate_id = predicate_id_,
        .operator_id = operator_id_,
        .state_generation = active_generation_,
        .prior_truth = previous_truth_,
        .last_reset_reason = last_reset_reason_,
        .reset_record = last_reset_record_,
    };
  }

  // Admission intentionally validates the complete cross-record predicate contract in one pass.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] static bool valid_plan(const PredicateTriggerPlanRecord& trigger,
                                       std::span<const PredicateInputPlanRecord> inputs,
                                       const PredicateCallablePlanRecord& callable) noexcept {
    if (trigger.id.empty() || !trigger.owner.plan_bound() || trigger.graph_generation == 0U ||
        trigger.graph_generation != trigger.owner.plan_generation ||
        trigger.state_generation != trigger.graph_generation || trigger.initial_truth ||
        !trigger.successful_evaluation_commits_truth || !trigger.missing_or_error_preserves_truth ||
        !trigger.candidate_state_is_transactional || !trigger.compute_adopts_evaluator_snapshot ||
        !trigger.consumes_reserved_candidate_tag ||
        trigger.reset_policy != TemporalPredicateResetPolicy::kResetEventsV1 ||
        (trigger.firing != PredicateFiring::kOnFalseToTrue &&
         trigger.firing != PredicateFiring::kOnEveryTrueEvaluation) ||
        trigger.missing_precedence != "safe_default_then_skip_or_exhausted_reuse" ||
        callable.predicate_id != trigger.id ||
        callable.disposition == PredicateCallableDisposition::kRejected ||
        (callable.result_kind != PredicateResultKind::kBool &&
         callable.result_kind != PredicateResultKind::kExpectedBoolError)) {
      return false;
    }
    const bool input_driven = trigger.source == PredicateEvaluationSourceKind::kInputUpdate;
    const bool clock_driven = trigger.source == PredicateEvaluationSourceKind::kClock;
    if ((!input_driven && !clock_driven) ||
        (input_driven && (!trigger.driver || trigger.clock.valid() || trigger.period_ns != 0 ||
                          trigger.phase.has_value() || inputs.empty())) ||
        (clock_driven && (trigger.driver || !trigger.clock.valid() || trigger.period_ns <= 0))) {
      return false;
    }
    for (std::size_t index = 0U; index < inputs.size(); ++index) {
      const auto& input = inputs[index];
      const bool expected_driver = input_driven && index == 0U;
      if (input.predicate_id != trigger.id || !input.input || input.argument_ordinal != index ||
          !input.reference_clock.valid() || input.cache_capacity == 0U ||
          !input.capture_time_required ||
          (input.selector != InputSelectorKind::kExact &&
           input.selector != InputSelectorKind::kLatestWithin) ||
          (input.selector == InputSelectorKind::kExact && input.max_age_ns != 0) ||
          (input.selector == InputSelectorKind::kLatestWithin && input.max_age_ns <= 0) ||
          (input.missing == TemporalRecordMissingKind::kReuseLast && input.reuse_max_age_ns <= 0) ||
          (input.missing != TemporalRecordMissingKind::kReuseLast && input.reuse_max_age_ns != 0) ||
          input.implicit_driver != expected_driver ||
          (expected_driver &&
           (input.input != trigger.driver || input.selector != InputSelectorKind::kExact ||
            input.missing != TemporalRecordMissingKind::kUnspecified))) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool valid_bound_source(
      const PredicateFrozenSource& source,
      std::span<const PredicateFrozenInput> inputs) const noexcept {
    if (!source.source || source.source_sequence == 0U) {
      return false;
    }
    if (source_kind_ == PredicateEvaluationSourceKind::kInputUpdate) {
      return source.driver_port == driver_ && !inputs.empty() && inputs.front().port == driver_;
    }
    return !source.driver_port && source.reference_time.clock_id == clock_;
  }

  [[nodiscard]] static bool valid_selection(InputSelection selection) noexcept {
    switch (selection.state) {
      case InputState::kSelected:
        return selection.reason == SelectionReason::kNone;
      case InputState::kReused:
      case InputState::kSafeDefault:
      case InputState::kAbsent:
        return selection.reason != SelectionReason::kNone &&
               selection.reason != SelectionReason::kNotSelectedForActivation;
      case InputState::kNotSelected:
        return selection.reason == SelectionReason::kNotSelectedForActivation;
    }
    return false;
  }

  template <typename Input>
  [[nodiscard]] static PredicateAggregateDisposition aggregate(
      std::span<const Input> inputs) noexcept {
    bool safe = false;
    bool absent = false;
    for (const auto& input : inputs) {
      const InputSelection selection = [&]() {
        if constexpr (std::same_as<Input, PredicateFrozenInput>) {
          return input.selection;
        } else {
          return input;
        }
      }();
      safe = safe || selection.state == InputState::kSafeDefault;
      absent = absent || selection.state == InputState::kAbsent;
    }
    if (safe) {
      return PredicateAggregateDisposition::kDegraded;
    }
    return absent ? PredicateAggregateDisposition::kNoFire
                  : PredicateAggregateDisposition::kNominal;
  }

  std::unique_ptr<PredicateCallableStorage> callable_;
  std::size_t expected_input_count_{};
  std::size_t compute_input_count_{};
  PredicateFiring firing_{PredicateFiring::kOnFalseToTrue};
  std::uint64_t active_generation_{};
  bool previous_truth_{};
  PredicateResetReason last_reset_reason_{PredicateResetReason::kColdRunStart};
  PredicateResetRecord last_reset_record_{PredicateResetReason::kColdRunStart, active_generation_};
  PredicateEvaluationSourceKind source_kind_{PredicateEvaluationSourceKind::kInputUpdate};
  OperatorInstanceId operator_id_{};
  PortId driver_{};
  ClockId clock_{};
  std::string predicate_id_;
  std::vector<PredicateInputSelection> predicate_selections_;
  std::vector<const void*> predicate_views_;
  std::vector<InputSelection> compute_selections_;
};

}  // namespace holoscan::detail
