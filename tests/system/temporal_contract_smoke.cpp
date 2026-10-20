/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Smoke test for the temporal-contract subsystem exercised entirely in one process.
// Steps through contract normalization, trigger-kind dispatch, readiness-source
// preselection, clock conversion, input caching, join/window engines, clock
// scheduling, and predicate evaluation — each failure mapped to a distinct exit code.

#include <array>
#include <chrono>
#include <cstdint>
#include <utility>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/predicate.hpp>
#include <holoscan/core/temporal_contract_records.hpp>

#include "core/readiness_source_internal.hpp"
#include "core/source_id_internal.hpp"
#include "core/temporal_contract_internal.hpp"
#include "runtime/predicate_evaluator.hpp"
#include "runtime/temporal_input_cache.hpp"
#include "runtime/temporal_preselection.hpp"

namespace {

using namespace std::chrono_literals;
constexpr holoscan::ClockId kClock{4};

// Constructs a minimal TemporalSampleRecord using kClock for identity; arrival_sequence
// mirrors source_sequence so callers only need to supply the two distinguishing fields.
[[nodiscard]] constexpr holoscan::detail::TemporalSampleRecord sample(
    std::int64_t timestamp, std::uint64_t sequence) noexcept {
  return {.capture_time = holoscan::TimePoint{timestamp, kClock},
          .source_sequence = sequence,
          .producer_epoch = 1,
          .arrival_sequence = sequence};
}

// SmokeOperator: a minimal two-input operator whose contract() is used as test
// input for normalization and trigger-kind checks. It is never scheduled or
// executed; only its Contract and port references are inspected.
class SmokeOperator final : public holoscan::Operator<> {
 public:
  holoscan::Input<std::uint32_t> anchor{"anchor"};
  holoscan::Input<std::uint32_t> values{"values"};

  holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnWindow{.anchor = anchor, .max_wait = 2ms})
        .input(anchor, holoscan::Exact{})
        .input(values, holoscan::BracketedWindow{.lookbehind = 1, .lookahead = 1, .max_samples = 8})
        .on_missing(values, holoscan::Skip{});
    return result;
  }

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    return {};
  }
};

}  // namespace

// REQ: CONTRACT-013
// This linear smoke intentionally keeps the public-to-private end-to-end sequence visible in one
// process; splitting it would obscure the ordering and installed-boundary evidence it exercises.
//
// Verification sequence (each step maps to one exit code on failure):
//
//   1  Contract normalization + OnWindow trigger
//   2  Trigger-kind dispatch: OnAnyInput / OnEach / OnClock / OnAligned / OnPredicate
//   3  Readiness-source trigger kinds: OnReadable / OnNotified / OnDoorbell / OnSpin
//   4  ReadinessPreselectionState creation for all four source kinds
//   5  Readiness witness fields (disposition, sequences, skipped tokens)
//   6  Direct-affine clock conversion + exact-interval and health checks
//   7  TemporalInputCache push
//   8  TemporalInputCache select_latest_within
//   9  AlignedJoinEngine push
//  10  AlignedJoinEngine select + commit
//  11  BracketedWindowEngine push (nominal)
//  12  BracketedWindowEngine front_closed state and sample count
//  13  BracketedWindowEngine push (terminal / absent path)
//  14  BracketedWindowEngine absent + kInsufficientHistory reason
//  15  CoalescingClockSchedule tick index
//  16  Predicate storage clone
//  17  PredicateEvaluatorEngine firing decision
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
int main() {
  const SmokeOperator op;
  auto normalized = holoscan::detail::ContractTestAccess::normalize(op.contract(), 2U);
  if (!normalized || normalized->inputs.size() != 2U) {
    return 1;  // exit 1: OnWindow contract normalization failed or input count wrong
  }

  auto accepts = [](const holoscan::Contract& contract,
                    std::size_t input_count,
                    holoscan::detail::TemporalTriggerKind kind) {
    auto value = holoscan::detail::ContractTestAccess::normalize(contract, input_count);
    return value && value->trigger.kind == kind;
  };
  const holoscan::Contract any;
  holoscan::Contract each;
  each.trigger(holoscan::OnEach{op.anchor});
  holoscan::Contract clock;
  clock.trigger(holoscan::OnClock{.period = 1ms});
  holoscan::Contract aligned;
  aligned
      .trigger(holoscan::OnAligned{.ports = holoscan::ports(op.anchor, op.values), .window = 1ms})
      .input(op.anchor, holoscan::Exact{})
      .input(op.values, holoscan::Exact{});
  auto predicate_binding = holoscan::predicate(
      "smoke-clock", [](const holoscan::PredicateEvaluationContext&) noexcept { return true; });
  holoscan::Contract predicate_contract;
  predicate_contract.trigger(holoscan::OnPredicate{
      .evaluate_on = holoscan::OnClock{.period = 1ms},
      .predicate = predicate_binding,
  });
  if (!accepts(any, 2U, holoscan::detail::TemporalTriggerKind::kAnyInput) ||
      !accepts(each, 2U, holoscan::detail::TemporalTriggerKind::kEach) ||
      !accepts(clock, 0U, holoscan::detail::TemporalTriggerKind::kClock) ||
      !accepts(aligned, 2U, holoscan::detail::TemporalTriggerKind::kAligned) ||
      !accepts(predicate_contract, 0U, holoscan::detail::TemporalTriggerKind::kPredicate)) {
    return 2;  // exit 2: at least one input-driven trigger kind did not normalize correctly
  }

  holoscan::ReadableSource readable;
  holoscan::NotificationSource notified;
  holoscan::DoorbellSource doorbell;
  holoscan::SpinSource spin;
  holoscan::detail::ReadinessSourceAccess::bind(
      readable,
      holoscan::detail::ReadinessSourceAccess::make_ref<holoscan::ReadinessSourceKind::kReadable>(
          holoscan::detail::make_source_id(1), 1));
  holoscan::detail::ReadinessSourceAccess::bind(
      notified,
      holoscan::detail::ReadinessSourceAccess::make_ref<
          holoscan::ReadinessSourceKind::kNotification>(holoscan::detail::make_source_id(2), 1));
  holoscan::detail::ReadinessSourceAccess::bind(
      doorbell,
      holoscan::detail::ReadinessSourceAccess::make_ref<holoscan::ReadinessSourceKind::kDoorbell>(
          holoscan::detail::make_source_id(3), 1));
  holoscan::detail::ReadinessSourceAccess::bind(
      spin,
      holoscan::detail::ReadinessSourceAccess::make_ref<holoscan::ReadinessSourceKind::kSpin>(
          holoscan::detail::make_source_id(4), 1));
  holoscan::Contract readable_contract;
  readable_contract.trigger(holoscan::OnReadable{.event = readable});
  holoscan::Contract notified_contract;
  notified_contract.trigger(holoscan::OnNotified{.event = notified});
  holoscan::Contract doorbell_contract;
  doorbell_contract.trigger(holoscan::OnDoorbell{.event = doorbell});
  holoscan::Contract spin_contract;
  spin_contract.trigger(holoscan::OnSpin{.event = spin});
  if (!accepts(readable_contract, 0U, holoscan::detail::TemporalTriggerKind::kReadable) ||
      !accepts(notified_contract, 0U, holoscan::detail::TemporalTriggerKind::kNotified) ||
      !accepts(doorbell_contract, 0U, holoscan::detail::TemporalTriggerKind::kDoorbell) ||
      !accepts(spin_contract, 0U, holoscan::detail::TemporalTriggerKind::kSpin)) {
    return 3;  // exit 3: a readiness-source trigger kind did not round-trip through normalization
  }

  auto readable_state = holoscan::detail::ReadinessPreselectionState::create(
      holoscan::ReadinessSourceKind::kReadable, 1U);
  auto notification_state = holoscan::detail::ReadinessPreselectionState::create(
      holoscan::ReadinessSourceKind::kNotification, 2U);
  auto doorbell_state = holoscan::detail::ReadinessPreselectionState::create(
      holoscan::ReadinessSourceKind::kDoorbell, 2U);
  auto spin_state = holoscan::detail::ReadinessPreselectionState::create(
      holoscan::ReadinessSourceKind::kSpin, 1U);
  if (!readable_state || !notification_state || !doorbell_state || !spin_state) {
    return 4;  // exit 4: ReadinessPreselectionState could not be created for one of the kinds
  }
  const auto readable_witness = readable_state->observe_readable(true);
  const auto notification_witness = notification_state->post_notification();
  const auto doorbell_witness = doorbell_state->observe_doorbell(3U);
  const auto spin_witness = spin_state->observe_spin(true);
  if (!readable_witness || !notification_witness || !doorbell_witness || !spin_witness ||
      readable_witness->disposition != holoscan::detail::ReadinessWitnessDisposition::kAccepted ||
      notification_witness->source_sequence != 1U ||
      doorbell_witness->skipped_doorbell_tokens != 2U || spin_witness->source_sequence != 1U) {
    // Exit 5: witness absent, disposition wrong, sequence not incremented, or
    //         skipped doorbell token count unexpected (observed 3 tokens, threshold 1 → 2 skipped)
    return 5;
  }

  constexpr holoscan::ClockId kConvertedClock{5};
  const holoscan::TemporalReferenceRecord admitted_reference{
      .kind = holoscan::TemporalRecordReferenceKind::kTriggerSample,
      .clock = {.is_default = false, .name = "converted"},
      .conversion = holoscan::TemporalRecordConversionKind::kDirectAffine,
      .conversion_record_id = "smoke-direct-affine/v1",
  };
  const holoscan::detail::DirectAffineClockEdge admitted_conversion{
      .source = kClock,
      .target = kConvertedClock,
      .conversion = {.scale = {.numerator = 1, .denominator = 1}, .offset_ns = 10},
  };
  const auto exact_coordinate = holoscan::detail::convert_identical_or_direct_exact(
      holoscan::TimePoint{10, kClock}, kConvertedClock, &admitted_conversion);
  const auto exact_boundary = holoscan::detail::convert_identical_or_direct_exact(
      holoscan::TimePoint{11, kClock}, kConvertedClock, &admitted_conversion);
  if (!exact_coordinate || !exact_boundary) {
    return 6;  // exit 6: direct-affine conversion produced no result (clock identity mismatch)
  }
  const auto within_boundary = holoscan::detail::exact_interval_within_inclusive_window(
      *exact_coordinate, *exact_boundary, 1ns);
  const auto health = holoscan::detail::apply_endpoint_health(
      within_boundary.value_or(false),
      {holoscan::detail::EndpointHealthState::kFreshDegraded,
       holoscan::detail::EndpointHealthAction::kDegrade});
  const auto materialized = holoscan::detail::materialize_exact_coordinate(*exact_coordinate);
  if (admitted_reference.conversion_record_id.empty() || !within_boundary || !*within_boundary ||
      !materialized || materialized->timestamp_ns != 20 || !health ||
      *health != holoscan::detail::HealthPreselectionResult::kDegraded) {
    // Exit 6: conversion record empty, coordinate outside window, materialized timestamp wrong
    //         (expected 10 + offset 10 = 20), or health action did not yield kDegraded
    return 6;
  }

  holoscan::detail::TemporalInputCache cache{4};
  if (!cache.push(sample(10, 1)) || !cache.push(sample(20, 2))) {
    return 7;  // exit 7: TemporalInputCache rejected a push (capacity or ordering violation)
  }
  auto selection = cache.select_latest_within(holoscan::TimePoint{20, kClock}, 10ns);
  if (!selection.sample || selection.sample->source_sequence != 2U) {
    return 8;  // exit 8: cache did not return the latest sample within the 10 ns window
  }

  holoscan::detail::AlignedJoinEngine join{2, 2};
  if (!join.push(0, sample(10, 1)) || !join.push(1, sample(11, 1))) {
    return 9;  // exit 9: AlignedJoinEngine rejected a push for one of the two lanes
  }
  const std::array policies{holoscan::detail::AlignedMissingPolicy{},
                            holoscan::detail::AlignedMissingPolicy{}};
  const std::array closure{holoscan::detail::AlignmentClosureProof::kOpen,
                           holoscan::detail::AlignmentClosureProof::kOpen};
  auto joined = join.select(1ns, policies, closure);
  if (!joined || joined->disposition() != holoscan::detail::AlignedCandidateDisposition::kNominal ||
      !joined->commit()) {
    return 10;  // exit 10: joined candidate absent, disposition not kNominal, or commit failed
  }

  holoscan::detail::BracketedWindowEngine window{
      holoscan::BracketedWindow{.lookbehind = 1, .lookahead = 1, .max_samples = 8}, 16, 2};
  if (!window.push_sample(sample(5, 1)) || !window.push_anchor(sample(10, 1)) ||
      !window.push_sample(sample(10, 2)) || !window.push_anchor(sample(20, 2)) ||
      !window.push_sample(sample(20, 3)) || !window.push_sample(sample(25, 4))) {
    return 11;  // exit 11: BracketedWindowEngine rejected a push in the nominal sequence
  }
  const auto* closed_window = window.front_closed();
  if (closed_window == nullptr ||
      closed_window->selection.state != holoscan::InputState::kSelected ||
      closed_window->samples.size() != 4U) {
    // Exit 12: nominal window not closed, state not kSelected, or sample count wrong
    //          (expected {5,10,10,20} — four samples bracketed by anchors at t=10 and t=20)
    return 12;
  }

  holoscan::detail::BracketedWindowEngine terminal_window{
      holoscan::BracketedWindow{.lookbehind = 1, .lookahead = 1, .max_samples = 8}, 8, 1};
  if (!terminal_window.push_anchor(sample(10, 1)) || !terminal_window.push_sample(sample(10, 1)) ||
      !terminal_window.push_anchor(sample(20, 2)) || !terminal_window.push_sample(sample(20, 2)) ||
      !terminal_window.push_sample(sample(21, 3))) {
    return 13;  // exit 13: terminal BracketedWindowEngine rejected a push
  }
  const auto* terminal = terminal_window.front_closed();
  if (terminal == nullptr || terminal->selection.state != holoscan::InputState::kAbsent ||
      terminal->selection.reason != holoscan::SelectionReason::kInsufficientHistory) {
    // Exit 14: terminal window absent state or kInsufficientHistory reason not reported
    //          (capacity=1 so the second anchor closes with no lookbehind sample present)
    return 14;
  }

  holoscan::detail::CoalescingClockSchedule schedule{holoscan::TimePoint{5, kClock}, 5ns};
  auto tick = schedule.due(holoscan::TimePoint{16, kClock});
  if (!tick || !tick->has_value()) {
    return 15;  // exit 15: schedule produced no tick or reported no pending tick at t=16
  }
  const auto tick_value = tick->value_or(holoscan::detail::ClockTickCandidate{});
  if (tick_value.index != 2U) {
    // Exit 15: wrong tick index — start=5, period=5ns, so t=16 falls on the 3rd tick (index 2)
    return 15;
  }

  auto storage = holoscan::detail::ContractTestAccess::clone_predicate_storage(predicate_binding);
  if (!storage) {
    return 16;  // exit 16: predicate storage could not be cloned from the binding
  }
  holoscan::detail::PredicateEvaluatorEngine<1, 1> evaluator{
      std::move(*storage), false, 0U, holoscan::PredicateFiring::kOnFalseToTrue, 1U};
  auto decision = evaluator.evaluate({
      .reference_time = holoscan::TimePoint{20, kClock},
      .tag = holoscan::Tag{.sequence = 1, .generation = 1},
      .tag_time = holoscan::TagTime{.logical_ns = 20},
  });
  if (!decision || !decision->firing_accepted) {
    return 17;  // exit 17: predicate evaluator rejected firing (always-true predicate must fire)
  }
  return 0;
}
