/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/temporal_contract_records.hpp>

namespace holoscan::test {
namespace {

// Produces an enum value whose underlying integer lies outside every named enumerator. Used to
// simulate a byte-stream payload that typed C++ authoring cannot construct, exercising defensive
// deserialization rejection paths.
template <typename Enum>
[[nodiscard]] Enum unknown_enum() noexcept {
  using Underlying = std::underlying_type_t<Enum>;
  // Deliberately construct a defensive-deserialization case that ordinary typed authoring cannot.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  return static_cast<Enum>(std::numeric_limits<Underlying>::max());
}

// Builds the minimal valid ContractRecord used as a baseline by most tests: kAnyInput trigger
// with one trigger pin named "in". Tests take a copy and mutate it rather than building anew.
[[nodiscard]] ContractRecord any_input_contract() {
  return ContractRecord{
      .operator_name = "op",
      .origin = TemporalRecordOrigin::kAuthored,
      .trigger = TemporalRecordTriggerKind::kAnyInput,
      .trigger_pins = {{.input_name = "in",
                        .retained_pin_capacity = 2,
                        .capacity_origin = TemporalRecordBoundOrigin::kResolvedRetention,
                        .provenance = {.source = "compiler.trigger-pin", .declaration_order = 1}}},
      .provenance = {.source = "operator.contract", .declaration_order = 0},
  };
}

// Builds a canonical scalar kExact edge with a kTriggerSample reference at the given declaration
// order. Tests copy-and-mutate this record to construct non-canonical edge configurations.
[[nodiscard]] EdgeSelectionRecord exact_edge(std::string name, std::uint32_t order) {
  return EdgeSelectionRecord{
      .input_name = std::move(name),
      .selector = TemporalRecordSelectorKind::kExact,
      .cardinality = TemporalRecordCardinality::kScalar,
      .reference = {.kind = TemporalRecordReferenceKind::kTriggerSample},
      .cache_capacity = 1,
      .cache_capacity_origin = TemporalRecordBoundOrigin::kSingleSample,
      .max_selected_samples = 1,
      .max_selected_origin = TemporalRecordBoundOrigin::kSingleSample,
      .batch_mode = TemporalRecordBatchMode::kScalar,
      .missing = TemporalRecordMissingKind::kUnspecified,
      .provenance = {.source = "contract.input", .declaration_order = order},
  };
}

// REQ: CONTRACT-015
// REQ: CONTRACT-020
// REQ: CONTRACT-023
// REQ: CONTRACT-024
TEST(TemporalContractRecords, SerializationMatchesVersionedCanonicalGolden) {
  // Key setup: minimal contract with zero edges; two independent calls must produce byte-identical
  //            output, confirming that serialization is deterministic.
  const auto contract = any_input_contract();
  const std::vector<EdgeSelectionRecord> edges;

  const auto first = serialize_temporal_records(contract, edges);
  const auto second = serialize_temporal_records(contract, edges);
  ASSERT_TRUE(first.has_value()) << first.error().message;
  ASSERT_TRUE(second.has_value()) << second.error().message;
  EXPECT_EQ(*first, *second);

  // Key assertion: serialized JSON matches the pinned versioned canonical string (CONTRACT-015).
  constexpr std::string_view kExpected =
      "{\"schema\":\"holoscan.temporal-records\",\"version\":2,\"contract\":{"
      "\"version\":2,\"operator_name\":\"op\",\"origin\":\"authored\","
      "\"trigger\":\"any_input\",\"trigger_input\":\"\",\"trigger_pins\":[{"
      "\"input_name\":\"in\",\"mode\":"
      "\"accepted_arrival_identity\",\"retained_pin_capacity\":2,"
      "\"capacity_origin\":\"resolved_retention\",\"provenance\":{"
      "\"source\":\"compiler.trigger-pin\",\"declaration_order\":1}}],"
      "\"trigger_source\":0,"
      "\"source_provenance\":\"\",\"trigger_clock\":{\"default\":true,\"name\":\"\"},"
      "\"period_ns\":0,\"phase\":null,\"aligned\":null,\"window\":null,"
      "\"predicate\":null,\"max_start_delay_ns\":null,\"violation\":null,\"provenance\":{"
      "\"source\":\"operator.contract\",\"declaration_order\":0}},\"edges\":[]}";
  EXPECT_EQ(*first, kExpected);
  // Operator completion is a separate physical-plan record family. A candidate count key must
  // not silently extend or version-skew the temporal-record artifact.
  EXPECT_EQ(first->find("\"count\""), std::string::npos);
}

// REQ: CONTRACT-015
// REQ: CONTRACT-023
TEST(TemporalContractRecords, RejectsLegacyVersionOneContractAndEdge) {
  auto contract = any_input_contract();
  contract.version = 1U;
  EXPECT_FALSE(validate_temporal_record(contract));
  EXPECT_FALSE(serialize_temporal_records(contract, {}));

  auto edge = exact_edge("in", 0U);
  edge.version = 1U;
  EXPECT_FALSE(validate_temporal_record(edge));
  const std::vector<EdgeSelectionRecord> edges{edge};
  EXPECT_FALSE(serialize_temporal_records(any_input_contract(), edges));
}

// REQ: CONTRACT-002
// REQ: CONTRACT-015
TEST(TemporalContractRecords, WindowRecordRetainsBoundsClosureAndOverlapSemantics) {
  auto contract = any_input_contract();
  contract.trigger = TemporalRecordTriggerKind::kWindow;
  contract.trigger_pins.clear();
  contract.trigger_input = "camera";
  contract.window = TemporalWindowRecord{
      .anchor_input = "camera",
      .wait_clock = {.is_default = false, .name = "steady"},
      .max_wait_ns = 1'000'000,
      .max_pending_windows = 4,
      .reference =
          TemporalReferenceRecord{
              .kind = TemporalRecordReferenceKind::kWindowClose,
              .clock = {.is_default = false, .name = "camera_hw"},
              .conversion = TemporalRecordConversionKind::kDirectAffine,
              .conversion_record_id = "clock-conversion/v1",
          },
  };

  auto camera = exact_edge("camera", 1);
  camera.reference.kind = TemporalRecordReferenceKind::kWindowClose;
  camera.reference.clock = {.is_default = false, .name = "camera_hw"};
  auto imu = EdgeSelectionRecord{
      .input_name = "imu",
      .selector = TemporalRecordSelectorKind::kBracketedWindow,
      .cardinality = TemporalRecordCardinality::kBatch,
      .lookbehind = 1,
      .lookahead = 2,
      .reference =
          TemporalReferenceRecord{
              .kind = TemporalRecordReferenceKind::kWindowClose,
              .clock = {.is_default = false, .name = "camera_hw"},
          },
      .cache_capacity = 16,
      .cache_capacity_origin = TemporalRecordBoundOrigin::kResolvedRetention,
      .max_selected_samples = 12,
      .max_selected_origin = TemporalRecordBoundOrigin::kSelectorMaximum,
      .batch_mode = TemporalRecordBatchMode::kNondestructiveSnapshot,
      .missing = TemporalRecordMissingKind::kSkip,
      .provenance = {.source = "contract.input", .declaration_order = 2},
  };
  // Key setup: kWindow contract with camera anchor (exact, kWindowClose reference) and a bracketed
  //            IMU edge (lookbehind=1, lookahead=2, batch snapshot, missing=kSkip).
  const std::vector edges{camera, imu};

  const auto serialized = serialize_temporal_records(contract, edges);
  ASSERT_TRUE(serialized.has_value()) << serialized.error().message;
  // Key assertion: all window closure-semantic and overlap flags appear in the output JSON.
  EXPECT_NE(serialized->find("\"previous_current_anchor_roles\":true"), std::string::npos);
  EXPECT_NE(serialized->find("\"strict_outer_samples\":true"), std::string::npos);
  EXPECT_NE(serialized->find("\"inclusive_interior\":true"), std::string::npos);
  EXPECT_NE(serialized->find("\"close_on_timeout\":true"), std::string::npos);
  EXPECT_NE(serialized->find("\"close_on_end_of_stream\":true"), std::string::npos);
  EXPECT_NE(serialized->find("\"retain_overlap\":true"), std::string::npos);
  EXPECT_NE(serialized->find("\"lookbehind\":1"), std::string::npos);
  EXPECT_NE(serialized->find("\"lookahead\":2"), std::string::npos);

  // Key assertion: setting missing=kSkip on the anchor input is a policy conflict and is rejected.
  camera.missing = TemporalRecordMissingKind::kSkip;
  const std::vector redundant_anchor_policy{camera, imu};
  EXPECT_FALSE(serialize_temporal_records(contract, redundant_anchor_policy).has_value());
}

// REQ: CONTRACT-015
TEST(TemporalContractRecords, RejectsReferenceClockThatConflictsWithTriggerCoordinate) {
  // Key setup: for each trigger kind, matching the reference clock to the trigger coordinate must
  //            succeed; any mismatch must produce an error (one positive + one negative per kind).

  // --- kClock sub-case ---
  auto clock = any_input_contract();
  clock.trigger = TemporalRecordTriggerKind::kClock;
  clock.trigger_pins.clear();
  clock.trigger_clock = {.is_default = false, .name = "schedule"};
  clock.period_ns = 1;
  auto scheduled = exact_edge("input", 1);
  scheduled.reference.kind = TemporalRecordReferenceKind::kScheduledTick;
  scheduled.reference.clock = clock.trigger_clock;
  scheduled.missing = TemporalRecordMissingKind::kSkip;
  EXPECT_TRUE(serialize_temporal_records(clock, std::vector{scheduled}).has_value());
  scheduled.reference.clock.name = "other";
  EXPECT_FALSE(serialize_temporal_records(clock, std::vector{scheduled}).has_value());

  // --- kAligned sub-case ---
  auto aligned = any_input_contract();
  aligned.trigger = TemporalRecordTriggerKind::kAligned;
  aligned.trigger_pins.clear();
  aligned.aligned = TemporalAlignedRecord{
      .member_inputs = {"first", "second"},
      .inclusive_window_ns = 1,
      .reference =
          TemporalReferenceRecord{
              .kind = TemporalRecordReferenceKind::kAlignedHigh,
              .clock = {.is_default = false, .name = "aligned"},
          },
  };
  auto first = exact_edge("first", 1);
  auto second = exact_edge("second", 2);
  first.reference = aligned.aligned->reference;
  second.reference = aligned.aligned->reference;
  EXPECT_TRUE(serialize_temporal_records(aligned, std::vector{first, second}).has_value());
  second.reference.clock.name = "other";
  EXPECT_FALSE(serialize_temporal_records(aligned, std::vector{first, second}).has_value());

  // --- kWindow sub-case ---
  auto window = any_input_contract();
  window.trigger = TemporalRecordTriggerKind::kWindow;
  window.trigger_pins.clear();
  window.trigger_input = "anchor";
  window.window = TemporalWindowRecord{
      .anchor_input = "anchor",
      .wait_clock = {.is_default = false, .name = "steady"},
      .max_wait_ns = 1,
      .max_pending_windows = 1,
      .reference =
          TemporalReferenceRecord{
              .kind = TemporalRecordReferenceKind::kWindowClose,
              .clock = {.is_default = false, .name = "camera"},
          },
  };
  auto anchor = exact_edge("anchor", 1);
  anchor.reference = window.window->reference;
  auto bracketed = EdgeSelectionRecord{
      .input_name = "samples",
      .selector = TemporalRecordSelectorKind::kBracketedWindow,
      .cardinality = TemporalRecordCardinality::kBatch,
      .lookbehind = 1,
      .lookahead = 1,
      .reference = window.window->reference,
      .cache_capacity = 4,
      .cache_capacity_origin = TemporalRecordBoundOrigin::kResolvedRetention,
      .max_selected_samples = 4,
      .max_selected_origin = TemporalRecordBoundOrigin::kSelectorMaximum,
      .batch_mode = TemporalRecordBatchMode::kNondestructiveSnapshot,
      .missing = TemporalRecordMissingKind::kSkip,
      .provenance = {.source = "contract.input", .declaration_order = 2},
  };
  EXPECT_TRUE(serialize_temporal_records(window, std::vector{anchor, bracketed}).has_value());
  bracketed.reference.clock.name = "other";
  EXPECT_FALSE(serialize_temporal_records(window, std::vector{anchor, bracketed}).has_value());

  // --- kPredicate sub-case ---
  auto predicate = any_input_contract();
  predicate.trigger = TemporalRecordTriggerKind::kPredicate;
  predicate.trigger_pins.clear();
  predicate.trigger_input = "driver";
  predicate.predicate = TemporalPredicateRecord{
      .record_id = "op/predicate/0",
      .identity = "ready",
      .source = TemporalPredicateSourceKind::kInputUpdate,
      .driver_input = "driver",
      .additional_inputs = {{.input_name = "context"}},
      .callable =
          TemporalPredicateCallableRecord{
              .kind = PredicateCallableKind::kStaticFunction,
              .implementation_fingerprint = "sha256:callable",
              .build_evidence = "package-evidence/v1",
              .size_bytes = 1,
              .alignment_bytes = 1,
          },
  };
  auto driver = exact_edge("driver", 1);
  auto context = exact_edge("context", 2);
  driver.reference.kind = TemporalRecordReferenceKind::kPredicateDriver;
  driver.reference.clock = {.is_default = false, .name = "driver"};
  context.reference = driver.reference;
  context.missing = TemporalRecordMissingKind::kSkip;
  EXPECT_TRUE(serialize_temporal_records(predicate, std::vector{driver, context}).has_value());
  context.reference.clock.name = "other";
  EXPECT_FALSE(serialize_temporal_records(predicate, std::vector{driver, context}).has_value());
}

// REQ: CONTRACT-015
TEST(TemporalContractRecords, RejectsPredicateTickAndDirectAffineClockConflicts) {
  auto predicate = any_input_contract();
  predicate.trigger = TemporalRecordTriggerKind::kPredicate;
  predicate.trigger_pins.clear();
  predicate.predicate = TemporalPredicateRecord{
      .record_id = "op/predicate/clock",
      .identity = "periodic_ready",
      .source = TemporalPredicateSourceKind::kClock,
      .clock = {.is_default = false, .name = "predicate-clock"},
      .period_ns = 1,
      .additional_inputs = {{.input_name = "context"}},
      .callable =
          TemporalPredicateCallableRecord{
              .kind = PredicateCallableKind::kStaticFunction,
              .implementation_fingerprint = "sha256:clock-callable",
              .build_evidence = "package-evidence/v1",
              .size_bytes = 1,
              .alignment_bytes = 1,
          },
  };
  auto context = exact_edge("context", 1);
  context.reference.kind = TemporalRecordReferenceKind::kPredicateTick;
  context.reference.clock = predicate.predicate->clock;
  context.missing = TemporalRecordMissingKind::kSkip;
  EXPECT_TRUE(serialize_temporal_records(predicate, std::vector{context}).has_value());
  context.reference.clock.name = "other";
  EXPECT_FALSE(serialize_temporal_records(predicate, std::vector{context}).has_value());

  auto clock = any_input_contract();
  clock.trigger = TemporalRecordTriggerKind::kClock;
  clock.trigger_pins.clear();
  clock.trigger_clock = {.is_default = false, .name = "schedule"};
  clock.period_ns = 1;
  auto converted = exact_edge("converted", 1);
  converted.reference = {
      .kind = TemporalRecordReferenceKind::kScheduledTick,
      .clock = clock.trigger_clock,
      .conversion = TemporalRecordConversionKind::kDirectAffine,
      .conversion_record_id = "conversion/input-to-schedule",
  };
  converted.missing = TemporalRecordMissingKind::kSkip;
  EXPECT_TRUE(serialize_temporal_records(clock, std::vector{converted}).has_value());
  converted.reference.clock.name = "other";
  EXPECT_FALSE(serialize_temporal_records(clock, std::vector{converted}).has_value());
}

// REQ: CONTRACT-010
// REQ: CONTRACT-015
TEST(TemporalContractRecords, PredicateRecordContainsMetadataButNoCallableOrTypeAddress) {
  auto contract = any_input_contract();
  contract.trigger = TemporalRecordTriggerKind::kPredicate;
  contract.trigger_pins.clear();
  contract.trigger_input = "temperature";
  contract.predicate = TemporalPredicateRecord{
      .record_id = "op/predicate/0",
      .identity = "thermal_trip",
      .scope = TemporalPredicateScope::kOperatorLocal,
      .source = TemporalPredicateSourceKind::kInputUpdate,
      .driver_input = "temperature",
      .additional_inputs = {{.input_name = "pressure"}},
      .firing = PredicateFiring::kOnFalseToTrue,
      .callable =
          TemporalPredicateCallableRecord{
              .kind = PredicateCallableKind::kInlineFunctor,
              .implementation_fingerprint = "sha256:callable",
              .build_evidence = "package-evidence/v1",
              .size_bytes = 16,
              .alignment_bytes = 8,
              .no_dynamic_allocation = true,
              .bounded_call = true,
              .is_noexcept = true,
              .owned_lifetime = true,
          },
      .state =
          TemporalPredicateStateRecord{
              .present = true,
              .size_bytes = 8,
              .alignment_bytes = 8,
              .max_bytes = 8,
              .schema = "thermal-state/v1",
          },
      .replay = TemporalPredicateReplayKind::kLocalGraphExecReevaluation,
  };

  auto temperature = exact_edge("temperature", 1);
  temperature.reference.kind = TemporalRecordReferenceKind::kPredicateDriver;
  auto pressure = exact_edge("pressure", 2);
  pressure.reference.kind = TemporalRecordReferenceKind::kPredicateDriver;
  pressure.missing = TemporalRecordMissingKind::kSkip;
  const std::vector edges{temperature, pressure};

  const auto serialized = serialize_temporal_records(contract, edges);
  ASSERT_TRUE(serialized.has_value()) << serialized.error().message;
  EXPECT_NE(serialized->find("\"local_graph_exec_reevaluation\""), std::string::npos);
  EXPECT_NE(serialized->find("\"reset_events_v1\""), std::string::npos);
  EXPECT_EQ(serialized->find("payload_type_key"), std::string::npos);
  EXPECT_EQ(serialized->find("0x"), std::string::npos);
}

// REQ: CONTRACT-010
// REQ: CONTRACT-015
TEST(TemporalContractRecords, RejectsPredicateEdgesWithNoncanonicalSelectionSemantics) {
  auto contract = any_input_contract();
  contract.trigger = TemporalRecordTriggerKind::kPredicate;
  contract.trigger_pins.clear();
  contract.trigger_input = "driver";
  contract.predicate = TemporalPredicateRecord{
      .record_id = "op/predicate/0",
      .identity = "ready",
      .source = TemporalPredicateSourceKind::kInputUpdate,
      .driver_input = "driver",
      .additional_inputs = {{.input_name = "context"}},
      .callable =
          TemporalPredicateCallableRecord{
              .kind = PredicateCallableKind::kStaticFunction,
              .implementation_fingerprint = "sha256:callable",
              .build_evidence = "package-evidence/v1",
              .size_bytes = 1,
              .alignment_bytes = 1,
          },
  };
  auto driver = exact_edge("driver", 1);
  auto context = exact_edge("context", 2);
  driver.reference.kind = TemporalRecordReferenceKind::kPredicateDriver;
  context.reference.kind = TemporalRecordReferenceKind::kPredicateDriver;
  context.missing = TemporalRecordMissingKind::kSkip;
  const std::vector valid_edges{driver, context};
  EXPECT_TRUE(serialize_temporal_records(contract, valid_edges).has_value());

  auto compute_only = exact_edge("compute", 3);
  compute_only.reference.kind = TemporalRecordReferenceKind::kPredicateDriver;
  EXPECT_FALSE(
      serialize_temporal_records(contract, std::vector{driver, context, compute_only}).has_value());
  compute_only.missing = TemporalRecordMissingKind::kSkip;
  EXPECT_TRUE(
      serialize_temporal_records(contract, std::vector{driver, context, compute_only}).has_value());

  context.missing = TemporalRecordMissingKind::kUnspecified;
  const std::vector missing_policy_absent{driver, context};
  EXPECT_FALSE(serialize_temporal_records(contract, missing_policy_absent).has_value());

  driver.selector = TemporalRecordSelectorKind::kLatestWithin;
  driver.max_age_ns = 1;
  context.missing = TemporalRecordMissingKind::kSkip;
  const std::vector nonexact_driver{driver, context};
  EXPECT_FALSE(serialize_temporal_records(contract, nonexact_driver).has_value());
}

// REQ: CONTRACT-002
// REQ: CONTRACT-015
TEST(TemporalContractRecords, RejectsUnknownEnumsInvalidBoundsAndNoncanonicalOrder) {
  // Key setup: each sub-case injects exactly one structural violation; all must independently fail.
  auto contract = any_input_contract();
  contract.trigger = unknown_enum<TemporalRecordTriggerKind>();
  EXPECT_FALSE(validate_temporal_record(contract).has_value());

  contract = any_input_contract();
  auto edge = exact_edge("in", 1);
  edge.selector = unknown_enum<TemporalRecordSelectorKind>();
  EXPECT_FALSE(validate_temporal_record(edge).has_value());

  edge = exact_edge("in", 1);
  edge.reference.conversion = unknown_enum<TemporalRecordConversionKind>();
  EXPECT_FALSE(validate_temporal_record(edge).has_value());

  edge = exact_edge("in", 1);
  edge.cache_capacity = 0;
  EXPECT_FALSE(validate_temporal_record(edge).has_value());

  edge = exact_edge("in", 1);
  edge.reference.kind = unknown_enum<TemporalRecordReferenceKind>();
  EXPECT_FALSE(validate_temporal_record(edge).has_value());

  edge = exact_edge("in", 1);
  edge.selector = TemporalRecordSelectorKind::kLatestWithin;
  edge.max_age_ns = 1;
  edge.max_selected_origin = TemporalRecordBoundOrigin::kSelectorMaximum;
  EXPECT_FALSE(validate_temporal_record(edge).has_value());

  contract = any_input_contract();
  contract.trigger_clock = {.is_default = false, .name = "irrelevant"};
  EXPECT_FALSE(validate_temporal_record(contract).has_value());

  contract = any_input_contract();
  contract.trigger = TemporalRecordTriggerKind::kPredicate;
  contract.trigger_pins.clear();
  contract.trigger_input = "temperature";
  contract.predicate = TemporalPredicateRecord{
      .record_id = "op/predicate/0",
      .identity = "thermal_trip",
      .source = TemporalPredicateSourceKind::kInputUpdate,
      .driver_input = "temperature",
      .callable =
          TemporalPredicateCallableRecord{
              .kind = unknown_enum<PredicateCallableKind>(),
              .implementation_fingerprint = "sha256:callable",
              .build_evidence = "package-evidence/v1",
              .size_bytes = 8,
              .alignment_bytes = 8,
          },
  };
  EXPECT_FALSE(validate_temporal_record(contract).has_value());

  edge = EdgeSelectionRecord{
      .input_name = "windowed",
      .selector = TemporalRecordSelectorKind::kBracketedWindow,
      .cardinality = TemporalRecordCardinality::kBatch,
      .lookbehind = std::numeric_limits<std::uint32_t>::max(),
      .lookahead = std::numeric_limits<std::uint32_t>::max(),
      .cache_capacity = std::numeric_limits<std::uint32_t>::max(),
      .cache_capacity_origin = TemporalRecordBoundOrigin::kResolvedRetention,
      .max_selected_samples = std::numeric_limits<std::uint32_t>::max(),
      .max_selected_origin = TemporalRecordBoundOrigin::kSelectorMaximum,
      .batch_mode = TemporalRecordBatchMode::kNondestructiveSnapshot,
      .missing = TemporalRecordMissingKind::kSkip,
      .provenance = {.source = "contract.input", .declaration_order = 1},
  };
  EXPECT_FALSE(validate_temporal_record(edge).has_value());

  // Key assertion: edges with non-ascending declaration_order are rejected (canonical order rule).
  contract = any_input_contract();
  const std::vector out_of_order{exact_edge("first", 2), exact_edge("second", 1)};
  EXPECT_FALSE(serialize_temporal_records(contract, out_of_order).has_value());
}

// REQ: CONTRACT-015
TEST(TemporalContractRecords, TriggerPinsAreCanonicalBoundedArrivalIdentityRecords) {
  auto contract = any_input_contract();
  EXPECT_TRUE(validate_temporal_record(contract).has_value());

  contract.trigger_pins.clear();
  EXPECT_FALSE(validate_temporal_record(contract).has_value());

  contract = any_input_contract();
  contract.trigger_pins.front().retained_pin_capacity = 0U;
  EXPECT_FALSE(validate_temporal_record(contract).has_value());

  contract = any_input_contract();
  contract.trigger_pins.push_back(contract.trigger_pins.front());
  contract.trigger_pins.back().provenance.declaration_order = 2U;
  EXPECT_FALSE(validate_temporal_record(contract).has_value());

  contract = any_input_contract();
  contract.trigger = TemporalRecordTriggerKind::kEach;
  contract.trigger_input = "in";
  EXPECT_TRUE(validate_temporal_record(contract).has_value());

  contract.trigger_input = "other";
  EXPECT_FALSE(validate_temporal_record(contract).has_value());

  contract = any_input_contract();
  contract.trigger = TemporalRecordTriggerKind::kClock;
  contract.trigger_pins.clear();
  contract.period_ns = 1U;
  EXPECT_TRUE(validate_temporal_record(contract).has_value());
  contract.trigger_pins = any_input_contract().trigger_pins;
  EXPECT_FALSE(validate_temporal_record(contract).has_value());
}

// REQ: CONTRACT-015
TEST(TemporalContractRecords, PresenceSetRequiresMissingOnlyForFailureCapableEdges) {
  auto clock = any_input_contract();
  clock.trigger = TemporalRecordTriggerKind::kClock;
  clock.trigger_pins.clear();
  clock.period_ns = 1;

  auto ordinary = exact_edge("ordinary", 1);
  ordinary.reference.kind = TemporalRecordReferenceKind::kScheduledTick;
  EXPECT_FALSE(serialize_temporal_records(clock, std::vector{ordinary}).has_value());
  ordinary.missing = TemporalRecordMissingKind::kSkip;
  EXPECT_TRUE(serialize_temporal_records(clock, std::vector{ordinary}).has_value());

  const EdgeSelectionRecord drain{
      .input_name = "batch",
      .selector = TemporalRecordSelectorKind::kDrainAll,
      .cardinality = TemporalRecordCardinality::kBatch,
      .reference = {.kind = TemporalRecordReferenceKind::kScheduledTick},
      .cache_capacity = 4,
      .cache_capacity_origin = TemporalRecordBoundOrigin::kResolvedRetention,
      .max_selected_samples = 4,
      .max_selected_origin = TemporalRecordBoundOrigin::kResolvedRetention,
      .batch_mode = TemporalRecordBatchMode::kConsumeOnAcceptance,
      .provenance = {.source = "contract.input", .declaration_order = 1},
  };
  EXPECT_TRUE(serialize_temporal_records(clock, std::vector{drain}).has_value());

  auto aligned = any_input_contract();
  aligned.trigger = TemporalRecordTriggerKind::kAligned;
  aligned.trigger_pins.clear();
  aligned.aligned = TemporalAlignedRecord{
      .member_inputs = {"first", "second"},
      .inclusive_window_ns = 1,
  };
  auto first = exact_edge("first", 1);
  auto second = exact_edge("second", 2);
  auto extra = exact_edge("extra", 3);
  first.reference.kind = TemporalRecordReferenceKind::kAlignedHigh;
  second.reference.kind = TemporalRecordReferenceKind::kAlignedHigh;
  extra.reference.kind = TemporalRecordReferenceKind::kAlignedHigh;
  EXPECT_FALSE(serialize_temporal_records(aligned, std::vector{first, second, extra}).has_value());
  extra.missing = TemporalRecordMissingKind::kSafeDefault;
  EXPECT_TRUE(serialize_temporal_records(aligned, std::vector{first, second, extra}).has_value());
}

}  // namespace
}  // namespace holoscan::test
