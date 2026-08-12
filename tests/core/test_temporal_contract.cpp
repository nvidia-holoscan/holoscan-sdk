/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstdint>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/predicate.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "../common/enum_test_utils.hpp"
#include "core/readiness_source_internal.hpp"
#include "core/source_id_internal.hpp"
#include "core/temporal_contract_internal.hpp"

namespace holoscan::test {
namespace {

using namespace std::chrono_literals;

// Minimal Operator<> with three typed inputs (anchor, values, follower). Provides concrete port
// references for Contract authoring and normalization tests throughout this file.
// Representative test: CopiesPortAndPredicateIdentityOutOfAuthoringObjects.
class ContractOperator final : public Operator<> {
 public:
  Input<std::uint32_t> anchor{"anchor"};
  Input<std::uint32_t> values{"values"};
  Input<std::uint32_t> follower{"follower"};

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  bool ready(const PredicateEvaluationContext&) noexcept { return true; }
};

// Free function predicate; verifies that the predicate() factory identifies kStaticFunction
// callable kind when the argument has no state or captures.
[[nodiscard]] bool free_predicate(const PredicateEvaluationContext&) noexcept {
  return true;
}

// Inline functor with a non-empty threshold field; tests that predicate() classifies non-empty
// callables as kInlineFunctor (distinct from kNonCapturingLambda for non-capturing lambdas).
struct InlinePredicate {
  std::uint32_t threshold{};

  [[nodiscard]] bool operator()(const PredicateEvaluationContext&) const noexcept {
    return threshold != 0U;
  }
};

// Empty callable struct with no data members; used alongside InlinePredicate to confirm that both
// empty and non-empty struct callables are classified as kInlineFunctor.
struct EmptyPredicate {
  [[nodiscard]] bool operator()(const PredicateEvaluationContext&) const noexcept { return true; }
};

// State type with a throwing copy constructor; the static_assert below verifies that
// PredicateState<T> requires nothrow-copyability, so this type must NOT satisfy the concept.
struct ThrowingPredicateState {
  ThrowingPredicateState() = default;
  ThrowingPredicateState(const ThrowingPredicateState&) noexcept(false) {}
  ThrowingPredicateState& operator=(const ThrowingPredicateState&) = default;
};

// REQ: CONTRACT-001
// REQ: CONTRACT-018
TEST(TemporalContract, EmptyInputfulContractMaterializesOwnedAnyInputDefault) {
  const Contract contract;
  auto normalized = detail::ContractTestAccess::normalize(contract, 2U);
  ASSERT_TRUE(normalized);
  EXPECT_EQ(normalized->trigger.kind, detail::TemporalTriggerKind::kAnyInput);
  EXPECT_TRUE(normalized->synthesized_default);

  auto zero_input = detail::ContractTestAccess::normalize(contract, 0U);
  ASSERT_FALSE(zero_input);
  EXPECT_EQ(zero_input.error().code, ErrorCode::kInvalidArgument);

  Contract explicit_any;
  explicit_any.trigger(OnAnyInput{});
  EXPECT_FALSE(detail::ContractTestAccess::normalize(explicit_any, 0U));

  const ContractOperator op;
  Contract customized_default;
  customized_default.input(op.values, Exact{}).on_missing(op.values, Skip{});
  auto customized = detail::ContractTestAccess::normalize(customized_default, 2U);
  ASSERT_FALSE(customized);
  EXPECT_EQ(customized.error().code, ErrorCode::kInvalidArgument);
  EXPECT_NE(customized.error().message.find("NO_TRIGGER"), std::string::npos);

  Contract deadline_default;
  deadline_default.deadline(3ms).on_violation(OnViolation::kFailRun);
  auto deadline_normalized = detail::ContractTestAccess::normalize(deadline_default, 2U);
  ASSERT_FALSE(deadline_normalized);
  EXPECT_EQ(deadline_normalized.error().code, ErrorCode::kInvalidArgument);
  EXPECT_NE(deadline_normalized.error().message.find("NO_TRIGGER"), std::string::npos);

  auto zero_input_deadline = detail::ContractTestAccess::normalize(deadline_default, 0U);
  ASSERT_FALSE(zero_input_deadline);
  EXPECT_EQ(zero_input_deadline.error().code, ErrorCode::kInvalidArgument);
}

// REQ: CONTRACT-001
TEST(TemporalContract, CopiesPortAndPredicateIdentityOutOfAuthoringObjects) {
  Contract contract;
  {
    const ContractOperator op;
    auto binding =
        predicate(PredicateKey{"nvidia.test.threshold", {1, 2, 3}},
                  [](const PredicateEvaluationContext&,
                     const PredicateInputView<std::uint32_t>&) noexcept { return true; });
    contract
        .trigger(
            OnPredicate{.evaluate_on = OnInputUpdate{op.anchor}, .predicate = std::move(binding)})
        .input(op.anchor, Exact{})
        .input(op.values, Exact{})
        .on_missing(op.values, Skip{});
  }
  auto copy = contract;
  auto normalized = detail::ContractTestAccess::normalize(copy, 2U);
  ASSERT_TRUE(normalized);
  EXPECT_EQ(normalized->trigger.port_name, "anchor");
  EXPECT_EQ(normalized->trigger.predicate.name(), "nvidia.test.threshold");
}

// REQ: CONTRACT-002
TEST(TemporalContract, RejectsInvalidSelectorAndMissingCombinations) {
  const ContractOperator op;
  Contract invalid_latest;
  invalid_latest.trigger(OnEach{op.anchor}).input(op.values, LatestWithin{});
  EXPECT_FALSE(detail::ContractTestAccess::normalize(invalid_latest, 2U));

  Contract invalid_reuse;
  invalid_reuse.trigger(OnEach{op.anchor})
      .input(op.values, DrainAll{})
      .on_missing(op.values, ReuseLast{.max_age = 1ms});
  EXPECT_FALSE(detail::ContractTestAccess::normalize(invalid_reuse, 2U));

  Contract valid;
  valid.trigger(OnEach{op.anchor})
      .input(op.values, LatestWithin{.max_age = 5ms})
      .on_missing(op.values, ReuseLast{.max_age = 20ms});
  EXPECT_TRUE(detail::ContractTestAccess::normalize(valid, 2U));
}

// REQ: CONTRACT-007
TEST(TemporalContract, ValidatesWindowAsExactAnchorAndBoundedBracket) {
  const ContractOperator op;
  Contract contract;
  contract.trigger(OnWindow{.anchor = op.anchor, .max_wait = 5ms, .max_pending_windows = 2})
      .input(op.anchor, Exact{})
      .input(op.values, BracketedWindow{.lookbehind = 2, .lookahead = 2, .max_samples = 16})
      .on_missing(op.values, Skip{});
  auto normalized = detail::ContractTestAccess::normalize(contract, 2U);
  ASSERT_TRUE(normalized);
  EXPECT_EQ(normalized->trigger.kind, detail::TemporalTriggerKind::kWindow);
  EXPECT_EQ(normalized->inputs[1].bracket.lookahead, 2U);

  Contract missing_policy;
  missing_policy.trigger(OnWindow{.anchor = op.anchor, .max_wait = 5ms})
      .input(op.anchor, Exact{})
      .input(op.values, BracketedWindow{.max_samples = 8});
  EXPECT_FALSE(detail::ContractTestAccess::normalize(missing_policy, 2U));
}

// REQ: CONTRACT-006
TEST(TemporalContract, ValidatesClosedAlignedMembersAndExactSelectors) {
  const ContractOperator op;
  Contract contract;
  contract.trigger(OnAligned{.ports = ports(op.anchor, op.values), .window = 2ms})
      .input(op.anchor, Exact{})
      .input(op.values, Exact{});
  EXPECT_TRUE(detail::ContractTestAccess::normalize(contract, 2U));

  Contract invalid;
  invalid.trigger(OnAligned{.ports = ports(op.anchor, op.values), .window = 2ms})
      .input(op.anchor, Exact{})
      .input(op.values, LatestWithin{.max_age = 2ms});
  EXPECT_FALSE(detail::ContractTestAccess::normalize(invalid, 2U));
}

// REQ: CONTRACT-009
TEST(TemporalContract, ReadinessTriggersCopyOnlyBoundLogicalIdentity) {
  ReadableSource source;
  const auto source_id = detail::make_source_id(9);
  detail::ReadinessSourceAccess::bind(
      source,
      detail::ReadinessSourceAccess::make_ref<ReadinessSourceKind::kReadable>(source_id, 3));

  Contract contract;
  contract.trigger(OnReadable{.event = source});
  auto normalized = detail::ContractTestAccess::normalize(contract, 0U);
  ASSERT_TRUE(normalized);
  EXPECT_EQ(normalized->trigger.kind, detail::TemporalTriggerKind::kReadable);
  EXPECT_EQ(normalized->trigger.source_id, source_id);

  Contract invalid;
  invalid.trigger(OnReadable{});
  EXPECT_FALSE(detail::ContractTestAccess::normalize(invalid, 0U));
}

// REQ: CONTRACT-010
TEST(TemporalContract, PredicateFactoriesOwnConfigStateInputsAndClassification) {
  // Key setup: exercises every predicate() factory form -- static function, opaque closure with
  //   State and Config, bound member function, non-capturing lambda, empty functor, inline functor.
  // Key assertion: callable_kind() reflects capture shape; has_config() and input_count() are
  //   correct; the owning copy of 'configured' preserves name/version after the authoring scope.
  struct State {
    std::uint32_t count{};
  };
  struct Config {
    std::uint32_t threshold{};
  };
  ContractOperator op;
  auto bare = predicate("bare", free_predicate);
  EXPECT_EQ(bare.callable_kind(), PredicateCallableKind::kStaticFunction);
  EXPECT_FALSE(bare.has_config());

  auto configured = predicate<State>(PredicateKey{"nvidia.test.configured", {2, 0, 0}},
                                     predicate_inputs(op.values),
                                     Config{.threshold = 4},
                                     [bias = 1U](const PredicateEvaluationContext&,
                                                 const PredicateInputView<std::uint32_t>&,
                                                 State& state,
                                                 const Config& config) -> expected<bool, Error> {
                                       ++state.count;
                                       return state.count + bias >= config.threshold;
                                     });
  EXPECT_EQ(configured.callable_kind(), PredicateCallableKind::kOpaqueClosure);
  EXPECT_TRUE(configured.has_config());
  EXPECT_EQ(configured.input_count(), 1U);
  // Exercise the binding's owning copy contract; a reference would not validate copied identity.
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  auto copied = configured;
  EXPECT_EQ(copied.name(), "nvidia.test.configured");

  auto member = predicate("bound-self", bind_predicate(op, &ContractOperator::ready));
  EXPECT_TRUE(member);
  EXPECT_EQ(member.callable_kind(), PredicateCallableKind::kBoundMember);

  auto noncapturing =
      predicate("noncapturing", [](const PredicateEvaluationContext&) noexcept { return true; });
  EXPECT_EQ(noncapturing.callable_kind(), PredicateCallableKind::kNonCapturingLambda);

  auto empty_functor = predicate("empty-functor", EmptyPredicate{});
  EXPECT_EQ(empty_functor.callable_kind(), PredicateCallableKind::kInlineFunctor);

  auto inline_functor = predicate("inline", InlinePredicate{.threshold = 1});
  EXPECT_EQ(inline_functor.callable_kind(), PredicateCallableKind::kInlineFunctor);
}

// REQ: CONTRACT-010
TEST(TemporalContract, PredicateAdmissionChecksSourceTypesDuplicatesAndMissingPolicies) {
  // Key setup: four contracts -- valid, missing on_missing for a predicate input, duplicate
  //   predicate_inputs referencing the driver port, and mismatched element type in the signature.
  // Key assertion: only the valid contract normalizes; each of the other three fails with a
  //   distinct structural error (missing policy, duplicate input, type mismatch).
  const ContractOperator op;
  auto valid_binding =
      predicate("valid",
                predicate_inputs(op.values),
                [](const PredicateEvaluationContext&,
                   const PredicateInputView<std::uint32_t>&,
                   const PredicateInputView<std::uint32_t>&) noexcept { return true; });
  Contract valid;
  valid
      .trigger(OnPredicate{.evaluate_on = OnInputUpdate{op.anchor},
                           .predicate = std::move(valid_binding)})
      .input(op.anchor, Exact{})
      .input(op.values, LatestWithin{.max_age = 1ms})
      .on_missing(op.values, Skip{});
  ASSERT_TRUE(detail::ContractTestAccess::normalize(valid, 2U));

  auto missing_binding =
      predicate("missing",
                predicate_inputs(op.values),
                [](const PredicateEvaluationContext&,
                   const PredicateInputView<std::uint32_t>&,
                   const PredicateInputView<std::uint32_t>&) noexcept { return true; });
  Contract missing;
  missing
      .trigger(OnPredicate{.evaluate_on = OnInputUpdate{op.anchor},
                           .predicate = std::move(missing_binding)})
      .input(op.anchor, Exact{})
      .input(op.values, Exact{});
  EXPECT_FALSE(detail::ContractTestAccess::normalize(missing, 2U));

  auto duplicate_binding =
      predicate("duplicate",
                predicate_inputs(op.anchor),
                [](const PredicateEvaluationContext&,
                   const PredicateInputView<std::uint32_t>&,
                   const PredicateInputView<std::uint32_t>&) noexcept { return true; });
  Contract duplicate;
  duplicate
      .trigger(OnPredicate{.evaluate_on = OnInputUpdate{op.anchor},
                           .predicate = std::move(duplicate_binding)})
      .input(op.anchor, Exact{});
  EXPECT_FALSE(detail::ContractTestAccess::normalize(duplicate, 1U));

  auto wrong_driver = predicate("wrong-driver",
                                [](const PredicateEvaluationContext&,
                                   const PredicateInputView<float>&) noexcept { return true; });
  Contract mismatch;
  mismatch
      .trigger(OnPredicate{.evaluate_on = OnInputUpdate{op.anchor},
                           .predicate = std::move(wrong_driver)})
      .input(op.anchor, Exact{});
  EXPECT_FALSE(detail::ContractTestAccess::normalize(mismatch, 1U));
}

// REQ: CONTRACT-012
TEST(TemporalContract, DeadlineRequiresPositiveBudgetAndExplicitDisposition) {
  const ContractOperator op;
  Contract valid;
  valid.trigger(OnEach{op.anchor}).deadline(3ms).on_violation(OnViolation::kFailRun);
  EXPECT_TRUE(detail::ContractTestAccess::normalize(valid, 1U));

  Contract missing_policy;
  missing_policy.trigger(OnEach{op.anchor}).deadline(3ms);
  EXPECT_FALSE(detail::ContractTestAccess::normalize(missing_policy, 1U));

  Contract zero;
  zero.trigger(OnEach{op.anchor}).deadline(0ns).on_violation(OnViolation::kEmitWitness);
  EXPECT_FALSE(detail::ContractTestAccess::normalize(zero, 1U));
}

// REQ: CONTRACT-002
TEST(TemporalContract, RejectsInvalidClosedPoliciesAndPresenceShapes) {
  // Key setup: seven contracts each with a distinct structural violation -- invalid OnClock phase,
  //   invalid ClockTickPolicy enum, duplicate aligned ports, OnEach re-selecting the driver port,
  //   input outside the presence set, redundant on_missing for the anchor, invalid OnViolation.
  // Key assertion: all seven normalize() calls return false.
  const ContractOperator op;
  Contract invalid_phase;
  invalid_phase.trigger(OnClock{.period = 1ms, .start_at = TimePoint{}});
  EXPECT_FALSE(detail::ContractTestAccess::normalize(invalid_phase, 0U));

  Contract invalid_tick;
  invalid_tick.trigger(OnClock{
      .period = 1ms,
      .tick_policy = enum_from_underlying<ClockTickPolicy>(std::uint8_t{255}),
  });
  EXPECT_FALSE(detail::ContractTestAccess::normalize(invalid_tick, 0U));

  Contract duplicate_aligned;
  duplicate_aligned.trigger(OnAligned{.ports = ports(op.anchor, op.anchor), .window = 1ms})
      .input(op.anchor, Exact{});
  EXPECT_FALSE(detail::ContractTestAccess::normalize(duplicate_aligned, 1U));

  Contract each_reselects_driver;
  each_reselects_driver.trigger(OnEach{op.anchor}).input(op.anchor, Exact{});
  EXPECT_FALSE(detail::ContractTestAccess::normalize(each_reselects_driver, 1U));

  Contract outside_presence_set;
  outside_presence_set.trigger(OnEach{op.anchor}).input(op.values, LatestWithin{.max_age = 1ms});
  EXPECT_FALSE(detail::ContractTestAccess::normalize(outside_presence_set, 2U));

  Contract redundant_missing;
  redundant_missing.trigger(OnWindow{.anchor = op.anchor, .max_wait = 1ms})
      .input(op.anchor, Exact{})
      .input(op.values, BracketedWindow{.max_samples = 8})
      .on_missing(op.anchor, Skip{})
      .on_missing(op.values, Skip{});
  EXPECT_FALSE(detail::ContractTestAccess::normalize(redundant_missing, 2U));

  Contract invalid_violation;
  invalid_violation.trigger(OnEach{op.anchor})
      .deadline(1ms)
      .on_violation(enum_from_underlying<OnViolation>(std::uint8_t{255}));
  EXPECT_FALSE(detail::ContractTestAccess::normalize(invalid_violation, 1U));
}

static_assert(std::is_copy_constructible_v<Contract>);
static_assert(std::is_nothrow_move_constructible_v<Contract>);
static_assert(std::is_copy_constructible_v<PredicateBinding>);
static_assert(std::same_as<decltype(SemanticVersion::major), std::uint16_t>);
static_assert(PredicateState<NoPredicateState>);
static_assert(!PredicateState<ThrowingPredicateState>);

}  // namespace
}  // namespace holoscan::test
