/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstdint>
#include <utility>

#include <gtest/gtest.h>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/predicate.hpp>

#include "../common/allocation_counter.hpp"
#include "core/predicate_input_view_internal.hpp"
#include "core/temporal_contract_internal.hpp"
#include "runtime/predicate_evaluator.hpp"

namespace holoscan::test {
namespace {

constexpr ClockId kClock{3};

// Minimal operator used as a typed receiver for bound predicate dispatch.
// The engine cross-checks the call-site local_type_token against the binding's
// stored token to prevent mismatched upcasts at evaluation time.
// Representative test: BoundMemberRequiresMatchingRuntimeReceiverType
class BoundPredicateOperator final : public Operator<> {
 public:
  [[nodiscard]] bool evaluate(const PredicateEvaluationContext&) noexcept {
    ++calls;
    return true;
  }

  [[nodiscard]] Contract contract() const override { return {}; }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  std::uint32_t calls{};
};

// Builds a PredicateFrozenSource stamped with a monotonic sequence number as both
// reference time and tag. The optional generation parameter distinguishes reset epochs
// so tests can verify the engine treats sources from different generations independently.
[[nodiscard]] detail::PredicateFrozenSource source(std::uint64_t sequence,
                                                   std::uint32_t generation = 1) {
  return {.reference_time = TimePoint{static_cast<std::int64_t>(sequence), kClock},
          .tag = Tag{.sequence = sequence, .generation = generation},
          .tag_time = TagTime{.logical_ns = static_cast<std::int64_t>(sequence)}};
}

// Stateful callback fixture for compute-only (no frozen inputs) predicate engines.
// Records how many times the selector is invoked; fail=true injects a selector error,
// which must leave the engine's previous_truth unchanged so the next good evaluation
// can re-arm the predicate cleanly.
template <std::size_t Size>
struct ComputeSelectionFixture {
  std::array<InputSelection, Size> selections{};
  std::uint32_t calls{};
  bool fail{};

  [[nodiscard]] static expected<std::span<const InputSelection>, Error> select(void* state,
                                                                               TimePoint) noexcept {
    auto& self = *static_cast<ComputeSelectionFixture*>(state);
    ++self.calls;
    if (self.fail) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    return std::span<const InputSelection>{self.selections};
  }

  [[nodiscard]] detail::PredicateComputeSelectionRequest request() noexcept {
    return {.planned_input_count = Size, .state = this, .select = &select};
  }
};

// REQ: CONTRACT-011
TEST(PredicateEvaluator, PreservesSnapshotAndCommitsStateTruthTransactionally) {
  struct State {
    std::uint32_t count{};
  };
  // Key setup: predicate is true on call counts 1 and 3 (false→true re-arm), and
  // returns an error when input data == 99.  The engine must not commit the error
  // result as a new truth state.
  auto binding = predicate<State>("transactional",
                                  [](const PredicateEvaluationContext&,
                                     const PredicateInputView<std::uint32_t>& input,
                                     State& state) -> expected<bool, Error> {
                                    ++state.count;
                                    if (input.sample().data == 99U) {
                                      return make_unexpected(Error{ErrorCode::kFailure});
                                    }
                                    return state.count == 1U || state.count == 3U;
                                  });
  auto storage = detail::ContractTestAccess::clone_predicate_storage(binding);
  ASSERT_TRUE(storage);
  detail::PredicateEvaluatorEngine<2, 2> engine{
      std::move(*storage), true, 0U, PredicateFiring::kOnFalseToTrue, 1U};

  const std::uint64_t generation = 1U;
  const Sample<std::uint32_t> first_sample{1U, SampleMetadata{.sequence = 1}};
  auto first_view = detail::PredicateBindingAccess::input<std::uint32_t>({
      .sample = &first_sample,
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .active_generation = &generation,
      .expected_generation = generation,
  });
  ASSERT_TRUE(first_view);
  const std::array first_inputs{detail::PredicateFrozenInput{
      .port = PortId{7},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .typed_view = &*first_view,
  }};
  // Key assertion: first call fires (count==1 → true); error on data==99 leaves
  // previous_truth==true; subsequent evaluations re-arm normally.
  auto first = engine.evaluate(source(1), first_inputs);
  ASSERT_TRUE(first);
  EXPECT_EQ(first->disposition, detail::PredicateCandidateDisposition::kNominalCompute);
  EXPECT_TRUE(first->evaluator_called);
  EXPECT_TRUE(first->firing_accepted);
  EXPECT_EQ(first->predicate_views[0], &*first_view);

  const Sample<std::uint32_t> error_sample{99U, SampleMetadata{.sequence = 2}};
  auto error_view = detail::PredicateBindingAccess::input<std::uint32_t>({
      .sample = &error_sample,
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .active_generation = &generation,
      .expected_generation = generation,
  });
  ASSERT_TRUE(error_view);
  const std::array error_inputs{detail::PredicateFrozenInput{
      .port = PortId{7},
      .selection = {InputState::kSelected, SelectionReason::kNone},
      .typed_view = &*error_view,
  }};
  auto failure = engine.evaluate(source(2), error_inputs);
  ASSERT_TRUE(failure);
  EXPECT_EQ(failure->disposition, detail::PredicateCandidateDisposition::kEvaluatorError);
  EXPECT_TRUE(engine.previous_truth());

  auto false_result = engine.evaluate(source(3), first_inputs);
  ASSERT_TRUE(false_result);
  EXPECT_FALSE(false_result->value);
  EXPECT_FALSE(false_result->firing_accepted);
  auto rearmed = engine.evaluate(source(4), first_inputs);
  ASSERT_TRUE(rearmed);
  EXPECT_TRUE(rearmed->value);
  EXPECT_TRUE(rearmed->firing_accepted);
}

// REQ: CONTRACT-011
TEST(PredicateEvaluator, ComputeOnlySuppressionPreservesCandidateStateAndTruth) {
  struct State {
    std::uint32_t count{};
  };
  auto binding = predicate<State>("compute-transaction",
                                  [](const PredicateEvaluationContext&, State& state) noexcept {
                                    ++state.count;
                                    return (state.count & 1U) != 0U;
                                  });

  // Key setup: two sub-scenarios share the same predicate.
  //   Scenario A — selector error: fail=true makes the selector return kFailure; the engine
  //     must propagate that error and preserve previous_truth.
  //   Scenario B — missing input: kAbsent selection suppresses firing without calling the
  //     evaluator, leaving the engine primed to fire on the next kSelected call.
  auto error_storage = detail::ContractTestAccess::clone_predicate_storage(binding);
  ASSERT_TRUE(error_storage);
  detail::PredicateEvaluatorEngine<1, 1> error_engine{
      std::move(*error_storage), false, 0U, PredicateFiring::kOnFalseToTrue, 1U};
  ComputeSelectionFixture<1> error_compute{{
      InputSelection{InputState::kSelected, SelectionReason::kNone},
  }};
  error_compute.fail = true;
  auto selector_error = error_engine.evaluate(source(1), {}, error_compute.request());
  ASSERT_FALSE(selector_error);
  EXPECT_EQ(selector_error.error().code, ErrorCode::kFailure);
  EXPECT_TRUE(error_engine.previous_truth());

  error_compute.fail = false;
  auto error_rearm = error_engine.evaluate(source(2), {}, error_compute.request());
  ASSERT_TRUE(error_rearm);
  EXPECT_FALSE(error_rearm->value);
  EXPECT_FALSE(error_rearm->firing_accepted);
  EXPECT_FALSE(error_engine.previous_truth());
  auto error_next_true = error_engine.evaluate(source(3), {}, error_compute.request());
  ASSERT_TRUE(error_next_true);
  EXPECT_TRUE(error_next_true->value);
  EXPECT_TRUE(error_next_true->firing_accepted);
  EXPECT_EQ(error_next_true->disposition, detail::PredicateCandidateDisposition::kNominalCompute);
  EXPECT_TRUE(error_engine.previous_truth());

  auto missing_storage = detail::ContractTestAccess::clone_predicate_storage(binding);
  ASSERT_TRUE(missing_storage);
  detail::PredicateEvaluatorEngine<1, 1> missing_engine{
      std::move(*missing_storage), false, 0U, PredicateFiring::kOnFalseToTrue, 1U};
  ComputeSelectionFixture<1> missing_compute{{
      InputSelection{InputState::kAbsent, SelectionReason::kNeverSeen},
  }};
  auto suppressed = missing_engine.evaluate(source(1), {}, missing_compute.request());
  ASSERT_TRUE(suppressed);
  EXPECT_TRUE(suppressed->value);
  EXPECT_FALSE(suppressed->firing_accepted);
  EXPECT_EQ(suppressed->disposition, detail::PredicateCandidateDisposition::kNoFire);
  EXPECT_TRUE(missing_engine.previous_truth());

  missing_compute.selections[0] = {InputState::kSelected, SelectionReason::kNone};
  auto missing_rearm = missing_engine.evaluate(source(2), {}, missing_compute.request());
  ASSERT_TRUE(missing_rearm);
  EXPECT_FALSE(missing_rearm->value);
  EXPECT_FALSE(missing_rearm->firing_accepted);
  EXPECT_FALSE(missing_engine.previous_truth());
  auto missing_next_true = missing_engine.evaluate(source(3), {}, missing_compute.request());
  ASSERT_TRUE(missing_next_true);
  EXPECT_TRUE(missing_next_true->value);
  EXPECT_TRUE(missing_next_true->firing_accepted);
  EXPECT_EQ(missing_next_true->disposition, detail::PredicateCandidateDisposition::kNominalCompute);
  EXPECT_TRUE(missing_engine.previous_truth());
}

// REQ: CONTRACT-003
// REQ: CONTRACT-011
TEST(PredicateEvaluator, AppliesMissingPrecedenceBeforeEvaluatorAndComputeOnlySelection) {
  auto binding = predicate("missing",
                           [](const PredicateEvaluationContext&,
                              const PredicateInputView<std::uint32_t>&) noexcept { return true; });
  auto storage = detail::ContractTestAccess::clone_predicate_storage(binding);
  ASSERT_TRUE(storage);
  detail::PredicateEvaluatorEngine<3, 3> engine{
      std::move(*storage), true, 0U, PredicateFiring::kOnFalseToTrue, 1U};

  // Key setup: a frozen input in the kSafeDefault/kTooStale state signals a degraded
  // compute.  The compute-selection callback must not be invoked; the evaluator must
  // not be called; the missing precedence overrides both.
  const std::array degraded_inputs{detail::PredicateFrozenInput{
      .port = PortId{1},
      .selection = {InputState::kSafeDefault, SelectionReason::kTooStale},
  }};
  ComputeSelectionFixture<1> deferred_compute{{
      InputSelection{InputState::kSelected, SelectionReason::kNone},
  }};
  auto degraded = engine.evaluate(source(1), degraded_inputs, deferred_compute.request());
  ASSERT_TRUE(degraded);
  EXPECT_EQ(degraded->disposition, detail::PredicateCandidateDisposition::kDegradedCompute);
  EXPECT_FALSE(degraded->evaluator_called);
  EXPECT_FALSE(engine.previous_truth());
  ASSERT_EQ(degraded->compute_input_count, 1U);
  EXPECT_EQ(degraded->compute_selections[0].state, InputState::kNotSelected);
  EXPECT_EQ(degraded->compute_selections[0].reason, SelectionReason::kNotSelectedForActivation);
  EXPECT_EQ(deferred_compute.calls, 0U);

  // Key assertion: a nominal input with kReused/kTooStale also flags degraded compute,
  // but this time the evaluator IS called and the compute-selection callback IS invoked.
  const std::uint64_t generation = 1U;
  const Sample<std::uint32_t> sample{5U, SampleMetadata{.sequence = 8}};
  auto view = detail::PredicateBindingAccess::input<std::uint32_t>({
      .sample = &sample,
      .selection = {InputState::kReused, SelectionReason::kTooStale},
      .active_generation = &generation,
      .expected_generation = generation,
  });
  ASSERT_TRUE(view);
  const std::array nominal_inputs{detail::PredicateFrozenInput{
      .port = PortId{1},
      .selection = {InputState::kReused, SelectionReason::kTooStale},
      .typed_view = &*view,
  }};
  ComputeSelectionFixture<2> compute_inputs{{
      InputSelection{InputState::kAbsent, SelectionReason::kNeverSeen},
      InputSelection{InputState::kSafeDefault, SelectionReason::kTooStale},
  }};
  auto compute_degraded = engine.evaluate(source(2), nominal_inputs, compute_inputs.request());
  ASSERT_TRUE(compute_degraded);
  EXPECT_TRUE(compute_degraded->evaluator_called);
  EXPECT_TRUE(compute_degraded->firing_accepted);
  EXPECT_EQ(compute_degraded->disposition, detail::PredicateCandidateDisposition::kDegradedCompute);
  EXPECT_EQ(compute_degraded->predicate_views[0], &*view);
  EXPECT_EQ(compute_inputs.calls, 1U);
}

// REQ: CONTRACT-011
TEST(PredicateEvaluator, ResetStartsFreshStateTruthAndGeneration) {
  auto binding =
      predicate("clock", [](const PredicateEvaluationContext&) noexcept { return true; });
  auto storage = detail::ContractTestAccess::clone_predicate_storage(binding);
  ASSERT_TRUE(storage);
  detail::PredicateEvaluatorEngine<1, 1> engine{
      std::move(*storage), false, 0U, PredicateFiring::kOnFalseToTrue, 1U};

  auto first = engine.evaluate(source(1));
  ASSERT_TRUE(first);
  EXPECT_TRUE(first->firing_accepted);
  auto repeated = engine.evaluate(source(2));
  ASSERT_TRUE(repeated);
  EXPECT_FALSE(repeated->firing_accepted);

  constexpr std::array reset_reasons{
      detail::PredicateResetReason::kPlanGeneration,
      detail::PredicateResetReason::kColdRunStart,
      detail::PredicateResetReason::kOperatorRestart,
      detail::PredicateResetReason::kSessionRebind,
      detail::PredicateResetReason::kProducerEpoch,
      detail::PredicateResetReason::kClockDiscontinuity,
  };
  std::uint32_t generation = 2U;
  for (const auto reason : reset_reasons) {
    engine.reset(reason, generation);
    auto after_reset = engine.evaluate(source(1, generation));
    ASSERT_TRUE(after_reset);
    EXPECT_TRUE(after_reset->firing_accepted);
    EXPECT_EQ(engine.last_reset_reason(), reason);
    ++generation;
  }
}

// REQ: CONTRACT-011
TEST(PredicateEvaluator, RejectsNotSelectedReturnedByComputeSelection) {
  auto binding =
      predicate("compute-pairing", [](const PredicateEvaluationContext&) noexcept { return true; });
  auto storage = detail::ContractTestAccess::clone_predicate_storage(binding);
  ASSERT_TRUE(storage);
  detail::PredicateEvaluatorEngine<1, 1> engine{
      std::move(*storage), false, 0U, PredicateFiring::kOnEveryTrueEvaluation, 1U};
  ComputeSelectionFixture<1> invalid{{
      InputSelection{InputState::kNotSelected, SelectionReason::kNotSelectedForActivation},
  }};
  auto decision = engine.evaluate(source(1), {}, invalid.request());
  ASSERT_FALSE(decision);
  EXPECT_EQ(decision.error().code, ErrorCode::kInvalidArgument);
}

// REQ: CONTRACT-014
TEST(PredicateEvaluator, WarmedEvaluationDoesNotAllocate) {
  auto binding =
      predicate("allocation", [](const PredicateEvaluationContext&) noexcept { return true; });
  auto storage = detail::ContractTestAccess::clone_predicate_storage(binding);
  ASSERT_TRUE(storage);
  detail::PredicateEvaluatorEngine<1, 1> engine{
      std::move(*storage), false, 0U, PredicateFiring::kOnEveryTrueEvaluation, 1U};
  ASSERT_TRUE(engine.evaluate(source(1)));

  std::size_t allocations = 0;
  bool accepted = false;
  {
    const ScopedAllocationCounter counter;
    auto decision = engine.evaluate(source(2));
    accepted = decision && decision->firing_accepted;
    allocations = counter.count();
  }
  EXPECT_TRUE(accepted);
  EXPECT_EQ(allocations, 0U);
}

// REQ: CONTRACT-014
TEST(PredicateEvaluator, WarmedCodeOnlyExpectedFailureDoesNotAllocate) {
  auto binding = predicate("code-only-error",
                           [](const PredicateEvaluationContext&) noexcept -> expected<bool, Error> {
                             return make_unexpected(Error{ErrorCode::kFailure});
                           });
  auto storage = detail::ContractTestAccess::clone_predicate_storage(binding);
  ASSERT_TRUE(storage);
  detail::PredicateEvaluatorEngine<1, 1> engine{
      std::move(*storage), false, 0U, PredicateFiring::kOnEveryTrueEvaluation, 1U};
  ASSERT_TRUE(engine.evaluate(source(1)));

  std::size_t allocations = 0;
  bool code_only = false;
  {
    const ScopedAllocationCounter counter;
    auto decision = engine.evaluate(source(2));
    code_only = decision &&
                decision->disposition == detail::PredicateCandidateDisposition::kEvaluatorError &&
                decision->evaluator_error.code == ErrorCode::kFailure &&
                decision->evaluator_error.message.empty();
    allocations = counter.count();
  }
  EXPECT_TRUE(code_only);
  EXPECT_EQ(allocations, 0U);
}

// REQ: CONTRACT-014
TEST(PredicateEvaluator, CoreAddsNoAllocationWhenCatchingAnEvaluatorException) {
  auto binding = predicate("throwing", [](const PredicateEvaluationContext&) -> bool { throw 7; });
  auto storage = detail::ContractTestAccess::clone_predicate_storage(binding);
  ASSERT_TRUE(storage);
  detail::PredicateEvaluatorEngine<1, 1> engine{
      std::move(*storage), false, 0U, PredicateFiring::kOnEveryTrueEvaluation, 1U};
  ASSERT_TRUE(engine.evaluate(source(1)));

  std::size_t allocations = 0;
  bool code_only = false;
  {
    const ScopedAllocationCounter counter;
    auto decision = engine.evaluate(source(2));
    code_only = decision &&
                decision->disposition == detail::PredicateCandidateDisposition::kEvaluatorError &&
                decision->evaluator_error.code == ErrorCode::kFailure &&
                decision->evaluator_error.message.empty();
    allocations = counter.count();
  }
  EXPECT_TRUE(code_only);
  EXPECT_EQ(allocations, 0U);
}

// REQ: CONTRACT-011
TEST(PredicateEvaluator, BoundMemberRequiresMatchingRuntimeReceiverType) {
  BoundPredicateOperator op;
  const auto& authoring_op = op;
  auto binding =
      predicate("bound", bind_predicate(authoring_op, &BoundPredicateOperator::evaluate));
  auto storage = detail::ContractTestAccess::clone_predicate_storage(binding);
  ASSERT_TRUE(storage);
  EXPECT_EQ((*storage)->receiver_type_token(),
            &detail::kPredicateReceiverLocalTypeToken<BoundPredicateOperator>);
  detail::PredicateEvaluatorEngine<1, 1> engine{
      std::move(*storage), false, 0U, PredicateFiring::kOnEveryTrueEvaluation, 1U};

  // Key setup: wrong receiver passes a null local_type_token; correct receiver passes
  // the token from kPredicateReceiverLocalTypeToken<BoundPredicateOperator>.
  // Key assertion: wrong token produces a code-only kInvalidArgument error with no
  // allocations; correct token produces a firing and increments op.calls.
  std::size_t allocations = 0U;
  bool code_only_receiver_error = false;
  {
    const ScopedAllocationCounter counter;
    auto wrong = engine.evaluate(
        source(1), {}, {}, detail::PredicateReceiver{.address = &op, .local_type_token = nullptr});
    code_only_receiver_error =
        wrong && wrong->disposition == detail::PredicateCandidateDisposition::kEvaluatorError &&
        wrong->evaluator_error.code == ErrorCode::kInvalidArgument &&
        wrong->evaluator_error.message.empty();
    allocations = counter.count();
  }
  EXPECT_TRUE(code_only_receiver_error);
  EXPECT_EQ(allocations, 0U);
  EXPECT_EQ(op.calls, 0U);

  auto correct = engine.evaluate(
      source(2),
      {},
      {},
      detail::PredicateReceiver{
          .address = &op,
          .local_type_token = &detail::kPredicateReceiverLocalTypeToken<BoundPredicateOperator>,
      });
  ASSERT_TRUE(correct);
  EXPECT_TRUE(correct->firing_accepted);
  EXPECT_EQ(op.calls, 1U);
}

TEST(PredicateEvaluator, NullCallableResetPreservesValidEngineState) {
  detail::PredicateEvaluatorEngine<1, 1> engine{
      nullptr, false, 0, PredicateFiring::kOnFalseToTrue, 1};

  engine.reset(detail::PredicateResetReason::kSessionRebind, 2);

  EXPECT_FALSE(engine.previous_truth());
  EXPECT_EQ(engine.generation(), 2U);
  EXPECT_EQ(engine.last_reset_reason(), detail::PredicateResetReason::kSessionRebind);
  const auto decision = engine.evaluate(source(2));
  ASSERT_FALSE(decision);
  EXPECT_EQ(decision.error().code, ErrorCode::kInvalidArgument);
}

}  // namespace
}  // namespace holoscan::test
