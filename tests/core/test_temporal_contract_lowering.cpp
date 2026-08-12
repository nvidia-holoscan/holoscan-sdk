/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/predicate.hpp>

#include "core/readiness_source_internal.hpp"
#include "core/source_id_internal.hpp"
#include "core/temporal_contract_internal.hpp"
#include "core/temporal_contract_lowering.hpp"

namespace holoscan::test {
namespace {

using namespace std::chrono_literals;

// Three-input operator used as the authoring context for building Contract objects.
// Its named ports (first, second, third) appear directly in trigger and input clauses,
// so the lowering under test can resolve port names from the operator's spec.
class LoweringOperator final : public Operator<> {
 public:
  Input<std::uint32_t> first{"first"};
  Input<std::uint32_t> second{"second"};
  Input<std::uint32_t> third{"third"};

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Factory helpers that produce resolved compiler-provided fact objects with sensible
// defaults.  Individual tests override only the fields relevant to their scenario so
// the boilerplate does not obscure the variation under test.

[[nodiscard]] detail::TemporalResolvedReferenceFacts reference_facts() {
  return {
      .resolved = true,
      .clock = {.is_default = true},
      .conversion = TemporalRecordConversionKind::kSameClock,
  };
}

[[nodiscard]] detail::TemporalResolvedEdgeFacts edge_facts(
    std::string name, std::uint32_t order, std::uint32_t cache_capacity = 16U,
    std::uint32_t drain_all_max_samples = 0U) {
  return {
      .resolved = true,
      .input_name = std::move(name),
      .reference = reference_facts(),
      .cache_capacity = cache_capacity,
      .cache_capacity_origin = TemporalRecordBoundOrigin::kResolvedRetention,
      .drain_all_max_samples = drain_all_max_samples,
      .provenance = {.source = "operator.contract.input", .declaration_order = order},
  };
}

[[nodiscard]] detail::TemporalResolvedPredicateFacts predicate_facts() {
  return {
      .resolved = true,
      .record_id = "predicate/1",
      .implementation_fingerprint = "predicate-implementation/v1",
      .build_evidence = "build/test",
      .no_dynamic_allocation = true,
      .bounded_call = true,
      .owned_lifetime = true,
  };
}

[[nodiscard]] detail::TemporalResolvedTriggerPinFacts pin_facts(std::string name,
                                                                std::uint32_t order) {
  return {
      .resolved = true,
      .input_name = std::move(name),
      .retained_pin_capacity = 2U,
      .provenance = {.source = "compiler.trigger-pin", .declaration_order = order},
  };
}

[[nodiscard]] detail::TemporalContractLoweringContext lowering_context(
    std::vector<detail::TemporalResolvedEdgeFacts> edges = {},
    std::uint32_t connected_input_count = 0U) {
  return {
      .operator_name = "lowering-test",
      .provenance = {.source = "operator.contract", .declaration_order = 0},
      .connected_input_count = connected_input_count,
      .trigger_reference = reference_facts(),
      .edges = std::move(edges),
  };
}

// REQ: CONTRACT-015
TEST(TemporalContractLowering, MapsAndSerializesEveryClosedTriggerKind) {
  const LoweringOperator op;
  auto check = [](const Contract& contract,
                  std::size_t connected_inputs,
                  const detail::TemporalContractLoweringContext& context,
                  TemporalRecordTriggerKind expected_kind) {
    auto snapshot = detail::ContractTestAccess::normalize(contract, connected_inputs);
    ASSERT_TRUE(snapshot);
    auto lowered = detail::lower_temporal_contract_records(*snapshot, context);
    ASSERT_TRUE(lowered);
    EXPECT_EQ(lowered->contract.trigger, expected_kind);
    EXPECT_TRUE(serialize_temporal_records(lowered->contract, lowered->edges));
  };

  const Contract any;
  auto any_context = lowering_context({}, 1U);
  any_context.trigger_pins = {pin_facts("first", 1U)};
  check(any, 1U, std::move(any_context), TemporalRecordTriggerKind::kAnyInput);

  Contract each;
  each.trigger(OnEach{op.first});
  auto each_context = lowering_context({}, 1U);
  each_context.trigger_pins = {pin_facts("first", 1U)};
  check(each, 1U, std::move(each_context), TemporalRecordTriggerKind::kEach);

  Contract clock;
  clock.trigger(OnClock{.period = 1ms, .clock = ClockKey::named("schedule")});
  check(clock, 0U, lowering_context(), TemporalRecordTriggerKind::kClock);

  ReadableSource readable;
  NotificationSource notified;
  DoorbellSource doorbell;
  SpinSource spin;
  detail::ReadinessSourceAccess::bind(
      readable,
      detail::ReadinessSourceAccess::make_ref<ReadinessSourceKind::kReadable>(
          detail::make_source_id(1), 1));
  detail::ReadinessSourceAccess::bind(
      notified,
      detail::ReadinessSourceAccess::make_ref<ReadinessSourceKind::kNotification>(
          detail::make_source_id(2), 2));
  detail::ReadinessSourceAccess::bind(
      doorbell,
      detail::ReadinessSourceAccess::make_ref<ReadinessSourceKind::kDoorbell>(
          detail::make_source_id(3), 2));
  detail::ReadinessSourceAccess::bind(
      spin,
      detail::ReadinessSourceAccess::make_ref<ReadinessSourceKind::kSpin>(detail::make_source_id(4),
                                                                          1));

  Contract readable_contract;
  readable_contract.trigger(OnReadable{readable});
  auto readable_context = lowering_context();
  readable_context.source_provenance = "source/readable";
  check(readable_contract, 0U, std::move(readable_context), TemporalRecordTriggerKind::kReadable);

  Contract notified_contract;
  notified_contract.trigger(OnNotified{notified});
  auto notified_context = lowering_context();
  notified_context.source_provenance = "source/notification";
  check(notified_contract, 0U, std::move(notified_context), TemporalRecordTriggerKind::kNotified);

  Contract doorbell_contract;
  doorbell_contract.trigger(OnDoorbell{doorbell});
  auto doorbell_context = lowering_context();
  doorbell_context.source_provenance = "source/doorbell";
  check(doorbell_contract, 0U, std::move(doorbell_context), TemporalRecordTriggerKind::kDoorbell);

  Contract spin_contract;
  spin_contract.trigger(OnSpin{spin});
  auto spin_context = lowering_context();
  spin_context.source_provenance = "source/spin";
  check(spin_contract, 0U, std::move(spin_context), TemporalRecordTriggerKind::kSpin);

  Contract aligned;
  aligned.trigger(OnAligned{.ports = ports(op.first, op.second), .window = 1ms})
      .input(op.first, Exact{})
      .input(op.second, Exact{});
  check(aligned,
        2U,
        lowering_context({edge_facts("first", 1U), edge_facts("second", 2U)}),
        TemporalRecordTriggerKind::kAligned);

  Contract window;
  window.trigger(OnWindow{.anchor = op.first, .max_wait = 2ms, .max_pending_windows = 2})
      .input(op.first, Exact{})
      .input(op.second, BracketedWindow{.lookbehind = 1, .lookahead = 1, .max_samples = 8})
      .on_missing(op.second, Skip{});
  check(window,
        2U,
        lowering_context({edge_facts("first", 1U), edge_facts("second", 2U)}),
        TemporalRecordTriggerKind::kWindow);

  // Key setup: the final trigger variant — OnPredicate — wraps a clock-driven predicate.
  // The predicate_context carries compiler-assigned facts (no_dynamic_allocation, etc.)
  // so the lowering can embed them into the TemporalPredicateRecord.
  auto binding =
      predicate("clock-predicate", [](const PredicateEvaluationContext&) noexcept { return true; });
  Contract predicate_contract;
  predicate_contract.trigger(OnPredicate{
      .evaluate_on = OnClock{.period = 1ms},
      .predicate = std::move(binding),
  });
  auto predicate_context = lowering_context();
  predicate_context.predicate = predicate_facts();
  check(
      predicate_contract, 0U, std::move(predicate_context), TemporalRecordTriggerKind::kPredicate);
}

// REQ: CONTRACT-015
TEST(TemporalContractLowering, RejectsAnyInputCatalogCountThatDiffersFromNormalization) {
  const Contract contract;
  auto snapshot = detail::ContractTestAccess::normalize(contract, 2U);
  ASSERT_TRUE(snapshot);

  auto context = lowering_context({}, 1U);
  context.trigger_pins = {pin_facts("first", 1U)};
  auto mismatch = detail::lower_temporal_contract_records(*snapshot, context);
  ASSERT_FALSE(mismatch);
  EXPECT_EQ(mismatch.error().code, ErrorCode::kInvalidArgument);
}

// REQ: CONTRACT-015
TEST(TemporalContractLowering, MapsEverySelectorCardinalityBatchAndMissingShape) {
  // Key setup: two sub-contracts exercise different selector / cardinality / missing shapes.
  //   scalar_and_fifo — Exact + LatestWithin(3ms) + DrainAll with scalar and batch modes.
  //   bracketed        — BracketedWindow with nondestructive snapshot mode and SafeDefault.
  const LoweringOperator op;
  Contract scalar_and_fifo;
  scalar_and_fifo.trigger(OnClock{.period = 1ms})
      .input(op.first, Exact{})
      .on_missing(op.first, Skip{})
      .input(op.second, LatestWithin{.max_age = 3ms})
      .on_missing(op.second, ReuseLast{.max_age = 9ms})
      .input(op.third, DrainAll{})
      .deadline(4ms)
      .on_violation(OnViolation::kEmitWitness);
  auto snapshot = detail::ContractTestAccess::normalize(scalar_and_fifo, 3U);
  ASSERT_TRUE(snapshot);
  auto lowered = detail::lower_temporal_contract_records(
      *snapshot,
      lowering_context(
          {edge_facts("first", 1U), edge_facts("second", 2U), edge_facts("third", 3U, 16U, 6U)}));
  // Key assertion: each edge's selector, cardinality, batch_mode, age limits, and
  // missing policy must match the contract clause exactly.  max_selected_samples
  // for DrainAll comes from the edge_facts drain_all_max_samples field.
  ASSERT_TRUE(lowered);
  ASSERT_EQ(lowered->edges.size(), 3U);
  EXPECT_EQ(lowered->edges[0].selector, TemporalRecordSelectorKind::kExact);
  EXPECT_EQ(lowered->edges[0].missing, TemporalRecordMissingKind::kSkip);
  EXPECT_EQ(lowered->edges[1].selector, TemporalRecordSelectorKind::kLatestWithin);
  EXPECT_EQ(lowered->edges[1].max_age_ns,
            std::chrono::duration_cast<std::chrono::nanoseconds>(3ms).count());
  EXPECT_EQ(lowered->edges[1].missing, TemporalRecordMissingKind::kReuseLast);
  EXPECT_EQ(lowered->edges[1].reuse_max_age_ns,
            std::chrono::duration_cast<std::chrono::nanoseconds>(9ms).count());
  EXPECT_EQ(lowered->edges[2].selector, TemporalRecordSelectorKind::kDrainAll);
  EXPECT_EQ(lowered->edges[2].cardinality, TemporalRecordCardinality::kBatch);
  EXPECT_EQ(lowered->edges[2].batch_mode, TemporalRecordBatchMode::kConsumeOnAcceptance);
  EXPECT_EQ(lowered->edges[2].max_selected_samples, 6U);
  EXPECT_EQ(lowered->contract.deadline_ns,
            std::chrono::duration_cast<std::chrono::nanoseconds>(4ms).count());
  EXPECT_EQ(lowered->contract.violation, OnViolation::kEmitWitness);

  Contract bracketed;
  bracketed.trigger(OnWindow{.anchor = op.first, .max_wait = 2ms})
      .input(op.first, Exact{})
      .input(op.second, BracketedWindow{.lookbehind = 2, .lookahead = 3, .max_samples = 9})
      .on_missing(op.second, SafeDefault{});
  auto bracketed_snapshot = detail::ContractTestAccess::normalize(bracketed, 2U);
  ASSERT_TRUE(bracketed_snapshot);
  auto bracketed_lowered = detail::lower_temporal_contract_records(
      *bracketed_snapshot, lowering_context({edge_facts("first", 1U), edge_facts("second", 2U)}));
  ASSERT_TRUE(bracketed_lowered);
  const auto& edge = bracketed_lowered->edges[1];
  EXPECT_EQ(edge.selector, TemporalRecordSelectorKind::kBracketedWindow);
  EXPECT_EQ(edge.cardinality, TemporalRecordCardinality::kBatch);
  EXPECT_EQ(edge.batch_mode, TemporalRecordBatchMode::kNondestructiveSnapshot);
  EXPECT_EQ(edge.lookbehind, 2U);
  EXPECT_EQ(edge.lookahead, 3U);
  EXPECT_EQ(edge.max_selected_samples, 9U);
  EXPECT_EQ(edge.missing, TemporalRecordMissingKind::kSafeDefault);
}

// REQ: CONTRACT-015
TEST(TemporalContractLowering, MapsCompletePredicateShapeAndCompilerEvidence) {
  struct State {
    std::uint32_t count{};
  };
  struct Config {
    std::uint32_t threshold{};
  };
  const LoweringOperator op;
  // Key setup: a qualified predicate (named key + semantic version) with typed State
  // and Config.  The lowering context provides state_max_bytes, config_schema, replay
  // kind, and other compiler-owned fields that must round-trip into the record.
  auto binding = predicate<State>(PredicateKey{"nvidia.test.lowering", {1, 2, 3}},
                                  predicate_inputs(op.second),
                                  Config{.threshold = 4},
                                  [](const PredicateEvaluationContext&,
                                     const PredicateInputView<std::uint32_t>&,
                                     const PredicateInputView<std::uint32_t>&,
                                     State&,
                                     const Config&) noexcept { return true; });
  Contract contract;
  contract
      .trigger(OnPredicate{
          .evaluate_on = OnInputUpdate{op.first},
          .predicate = std::move(binding),
          .firing = PredicateFiring::kOnEveryTrueEvaluation,
      })
      .input(op.first, Exact{})
      .input(op.second, LatestWithin{.max_age = 1ms})
      .on_missing(op.second, Skip{});
  auto snapshot = detail::ContractTestAccess::normalize(contract, 2U);
  ASSERT_TRUE(snapshot);

  auto context = lowering_context({edge_facts("first", 1U), edge_facts("second", 2U)});
  auto facts = predicate_facts();
  facts.record_id = "predicate/qualified/1";
  facts.state_max_bytes = 64U;
  facts.state_schema = "state/u32/v1";
  facts.config_max_bytes = 64U;
  facts.config_schema = "config/u32/v1";
  facts.config_digest = "sha256:test";
  facts.replay = TemporalPredicateReplayKind::kCrossBuildRecomputation;
  facts.replay_evidence = "qualified-build-match/v1";
  context.predicate = std::move(facts);
  // Key assertion: every compiler-supplied predicate field (identity, scope, version,
  // source kind, driver input, additional inputs, state/config bounds, replay kind)
  // must be faithfully preserved in the lowered TemporalPredicateRecord.
  auto lowered = detail::lower_temporal_contract_records(*snapshot, context);
  ASSERT_TRUE(lowered);
  ASSERT_TRUE(lowered->contract.predicate);
  const auto record = lowered->contract.predicate.value_or(TemporalPredicateRecord{});
  EXPECT_EQ(record.identity, "nvidia.test.lowering");
  EXPECT_EQ(record.scope, TemporalPredicateScope::kQualified);
  EXPECT_EQ(record.semantic_version, (SemanticVersion{1, 2, 3}));
  EXPECT_EQ(record.source, TemporalPredicateSourceKind::kInputUpdate);
  EXPECT_EQ(record.driver_input, "first");
  ASSERT_EQ(record.additional_inputs.size(), 1U);
  EXPECT_EQ(record.additional_inputs[0].input_name, "second");
  EXPECT_EQ(record.firing, PredicateFiring::kOnEveryTrueEvaluation);
  EXPECT_TRUE(record.callable.is_noexcept);
  EXPECT_TRUE(record.state.present);
  EXPECT_EQ(record.state.max_bytes, 64U);
  EXPECT_TRUE(record.config.present);
  EXPECT_EQ(record.config.digest, "sha256:test");
  EXPECT_EQ(record.replay, TemporalPredicateReplayKind::kCrossBuildRecomputation);
}

// REQ: CONTRACT-015
TEST(TemporalContractLowering, RejectsMissingCompilerOwnedFactsWithoutInventingDefaults) {
  // Key setup: each failure path removes or invalidates exactly one compiler-owned fact,
  // confirming that the lowering rejects missing context rather than substituting defaults.
  //   1. No edges at all for a DrainAll input.
  //   2. Edges present but drain_all_max_samples == 0 (missing fact).
  //   3. Edge present but reference.resolved == false.
  //   4. Aligned trigger present but trigger_reference.resolved == false.
  const LoweringOperator op;
  Contract drain;
  drain.trigger(OnClock{.period = 1ms}).input(op.first, DrainAll{});
  auto snapshot = detail::ContractTestAccess::normalize(drain, 1U);
  ASSERT_TRUE(snapshot);
  EXPECT_FALSE(detail::lower_temporal_contract_records(*snapshot, lowering_context()));
  EXPECT_FALSE(detail::lower_temporal_contract_records(
      *snapshot, lowering_context({edge_facts("first", 1U)})));

  auto unresolved = edge_facts("first", 1U, 8U, 4U);
  unresolved.reference.resolved = false;
  EXPECT_FALSE(detail::lower_temporal_contract_records(*snapshot,
                                                       lowering_context({std::move(unresolved)})));

  Contract aligned;
  aligned.trigger(OnAligned{.ports = ports(op.first, op.second), .window = 1ms})
      .input(op.first, Exact{})
      .input(op.second, Exact{});
  auto aligned_snapshot = detail::ContractTestAccess::normalize(aligned, 2U);
  ASSERT_TRUE(aligned_snapshot);
  auto missing_reference = lowering_context({edge_facts("first", 1U), edge_facts("second", 2U)});
  missing_reference.trigger_reference.resolved = false;
  EXPECT_FALSE(detail::lower_temporal_contract_records(*aligned_snapshot, missing_reference));
}

}  // namespace
}  // namespace holoscan::test
