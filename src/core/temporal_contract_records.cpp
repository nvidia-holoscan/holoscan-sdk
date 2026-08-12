/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

#include <holoscan/core/temporal_contract_records.hpp>

#include "source_id_internal.hpp"

namespace holoscan {
namespace {

[[nodiscard]] unexpected<Error> invalid_record(std::string message) {
  return make_unexpected(Error{.code = ErrorCode::kInvalidArgument, .message = std::move(message)});
}

[[nodiscard]] bool valid_utf8(std::string_view text) noexcept {
  const auto* bytes = reinterpret_cast<const unsigned char*>(text.data());
  std::size_t index = 0;
  while (index < text.size()) {
    const unsigned char first = bytes[index];
    if (first <= 0x7fU) {
      ++index;
      continue;
    }

    std::size_t continuation_count = 0;
    std::uint32_t code_point = 0;
    if (first >= 0xc2U && first <= 0xdfU) {
      continuation_count = 1;
      code_point = first & 0x1fU;
    } else if (first >= 0xe0U && first <= 0xefU) {
      continuation_count = 2;
      code_point = first & 0x0fU;
    } else if (first >= 0xf0U && first <= 0xf4U) {
      continuation_count = 3;
      code_point = first & 0x07U;
    } else {
      return false;
    }
    if (index + continuation_count >= text.size()) {
      return false;
    }
    for (std::size_t offset = 1; offset <= continuation_count; ++offset) {
      const unsigned char next = bytes[index + offset];
      if ((next & 0xc0U) != 0x80U) {
        return false;
      }
      code_point = (code_point << 6U) | (next & 0x3fU);
    }
    if ((continuation_count == 2 && code_point < 0x800U) ||
        (continuation_count == 3 && code_point < 0x10000U) ||
        (code_point >= 0xd800U && code_point <= 0xdfffU) || code_point > 0x10ffffU) {
      return false;
    }
    index += continuation_count + 1;
  }
  return true;
}

[[nodiscard]] bool valid_text(std::string_view text, bool allow_empty = false) noexcept {
  if ((!allow_empty && text.empty()) || !valid_utf8(text)) {
    return false;
  }
  return std::none_of(text.begin(), text.end(), [](char value) {
    const auto byte = static_cast<unsigned char>(value);
    return byte < 0x20U || byte == 0x7fU;
  });
}

[[nodiscard]] bool power_of_two(std::uint64_t value) noexcept {
  return value != 0 && (value & (value - 1U)) == 0;
}

[[nodiscard]] bool default_clock(const TemporalClockKeyRecord& clock) noexcept {
  return clock.is_default && clock.name.empty();
}

template <typename Value>
[[nodiscard]] bool all_unique(const std::vector<Value>& values) {
  std::unordered_set<Value> seen;
  seen.reserve(values.size());
  return std::all_of(
      values.begin(), values.end(), [&](const Value& value) { return seen.insert(value).second; });
}

[[nodiscard]] std::optional<std::string_view> token(TemporalRecordTriggerKind value) noexcept {
  switch (value) {
    case TemporalRecordTriggerKind::kAnyInput:
      return "any_input";
    case TemporalRecordTriggerKind::kEach:
      return "each";
    case TemporalRecordTriggerKind::kClock:
      return "clock";
    case TemporalRecordTriggerKind::kReadable:
      return "readable";
    case TemporalRecordTriggerKind::kNotified:
      return "notified";
    case TemporalRecordTriggerKind::kDoorbell:
      return "doorbell";
    case TemporalRecordTriggerKind::kSpin:
      return "spin";
    case TemporalRecordTriggerKind::kAligned:
      return "aligned";
    case TemporalRecordTriggerKind::kWindow:
      return "window";
    case TemporalRecordTriggerKind::kPredicate:
      return "predicate";
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> token(TemporalRecordOrigin value) noexcept {
  switch (value) {
    case TemporalRecordOrigin::kAuthored:
      return "authored";
    case TemporalRecordOrigin::kCoreSynthesized:
      return "core_synthesized";
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> token(TemporalTriggerPinMode value) noexcept {
  switch (value) {
    case TemporalTriggerPinMode::kAcceptedArrivalIdentity:
      return "accepted_arrival_identity";
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> token(TemporalRecordSelectorKind value) noexcept {
  switch (value) {
    case TemporalRecordSelectorKind::kExact:
      return "exact";
    case TemporalRecordSelectorKind::kLatestWithin:
      return "latest_within";
    case TemporalRecordSelectorKind::kDrainAll:
      return "drain_all";
    case TemporalRecordSelectorKind::kBracketedWindow:
      return "bracketed_window";
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> token(TemporalRecordCardinality value) noexcept {
  switch (value) {
    case TemporalRecordCardinality::kScalar:
      return "scalar";
    case TemporalRecordCardinality::kBatch:
      return "batch";
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> token(TemporalRecordBatchMode value) noexcept {
  switch (value) {
    case TemporalRecordBatchMode::kScalar:
      return "scalar";
    case TemporalRecordBatchMode::kConsumeOnAcceptance:
      return "consume_on_acceptance";
    case TemporalRecordBatchMode::kNondestructiveSnapshot:
      return "nondestructive_snapshot";
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> token(TemporalRecordMissingKind value) noexcept {
  switch (value) {
    case TemporalRecordMissingKind::kUnspecified:
      return "unspecified";
    case TemporalRecordMissingKind::kSkip:
      return "skip";
    case TemporalRecordMissingKind::kSafeDefault:
      return "safe_default";
    case TemporalRecordMissingKind::kReuseLast:
      return "reuse_last";
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> token(TemporalRecordReferenceKind value) noexcept {
  switch (value) {
    case TemporalRecordReferenceKind::kTriggerSample:
      return "trigger_sample";
    case TemporalRecordReferenceKind::kScheduledTick:
      return "scheduled_tick";
    case TemporalRecordReferenceKind::kAlignedHigh:
      return "aligned_high";
    case TemporalRecordReferenceKind::kWindowClose:
      return "window_close";
    case TemporalRecordReferenceKind::kReadinessWitness:
      return "readiness_witness";
    case TemporalRecordReferenceKind::kPredicateDriver:
      return "predicate_driver";
    case TemporalRecordReferenceKind::kPredicateTick:
      return "predicate_tick";
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> token(TemporalRecordConversionKind value) noexcept {
  switch (value) {
    case TemporalRecordConversionKind::kSameClock:
      return "same_clock";
    case TemporalRecordConversionKind::kDirectAffine:
      return "direct_affine";
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> token(TemporalRecordBoundOrigin value) noexcept {
  switch (value) {
    case TemporalRecordBoundOrigin::kSingleSample:
      return "single_sample";
    case TemporalRecordBoundOrigin::kResolvedRetention:
      return "resolved_retention";
    case TemporalRecordBoundOrigin::kSelectorMaximum:
      return "selector_maximum";
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> token(TemporalPredicateSourceKind value) noexcept {
  switch (value) {
    case TemporalPredicateSourceKind::kInputUpdate:
      return "input_update";
    case TemporalPredicateSourceKind::kClock:
      return "clock";
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> token(TemporalPredicateScope value) noexcept {
  switch (value) {
    case TemporalPredicateScope::kOperatorLocal:
      return "operator_local";
    case TemporalPredicateScope::kQualified:
      return "qualified";
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> token(PredicateCallableKind value) noexcept {
  switch (value) {
    case PredicateCallableKind::kStaticFunction:
      return "static_function";
    case PredicateCallableKind::kNonCapturingLambda:
      return "noncapturing_lambda";
    case PredicateCallableKind::kInlineFunctor:
      return "inline_functor";
    case PredicateCallableKind::kOpaqueClosure:
      return "opaque_closure";
    case PredicateCallableKind::kBoundMember:
      return "bound_member";
    case PredicateCallableKind::kTypeErasedFunction:
      return "type_erased_function";
    case PredicateCallableKind::kPythonFunction:
      return "python_function";
    case PredicateCallableKind::kPythonLambda:
      return "python_lambda";
    case PredicateCallableKind::kPythonBoundMethod:
      return "python_bound_method";
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> token(TemporalPredicateResetPolicy value) noexcept {
  switch (value) {
    case TemporalPredicateResetPolicy::kEa1ResetEventsV1:
      return "ea1_reset_events_v1";
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> token(TemporalPredicateReplayKind value) noexcept {
  switch (value) {
    case TemporalPredicateReplayKind::kDecisionReplay:
      return "decision_replay";
    case TemporalPredicateReplayKind::kLocalGraphExecReevaluation:
      return "local_graph_exec_reevaluation";
    case TemporalPredicateReplayKind::kCrossBuildRecomputation:
      return "cross_build_recomputation";
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> token(PredicateFiring value) noexcept {
  switch (value) {
    case PredicateFiring::kOnFalseToTrue:
      return "on_false_to_true";
    case PredicateFiring::kOnEveryTrueEvaluation:
      return "on_every_true_evaluation";
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> token(OnViolation value) noexcept {
  switch (value) {
    case OnViolation::kSafeDefault:
      return "safe_default";
    case OnViolation::kEmitWitness:
      return "emit_witness";
    case OnViolation::kFailRun:
      return "fail_run";
  }
  return std::nullopt;
}

template <typename Enum>
[[nodiscard]] std::string_view serialized_token(Enum value) noexcept {
  return token(value).value_or("invalid");
}

[[nodiscard]] expected<void, Error> validate_clock(const TemporalClockKeyRecord& clock,
                                                   std::string_view subject) {
  if (clock.is_default && !clock.name.empty()) {
    return invalid_record(std::string(subject) + " default clock must not carry a name");
  }
  if (!clock.is_default && !valid_text(clock.name)) {
    return invalid_record(std::string(subject) + " named clock requires valid UTF-8 identity");
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_provenance(const TemporalRecordProvenance& provenance,
                                                        std::string_view subject) {
  if (!valid_text(provenance.source)) {
    return invalid_record(std::string(subject) + " requires stable UTF-8 provenance");
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_reference(const TemporalReferenceRecord& reference,
                                                       std::string_view subject) {
  if (!token(reference.kind) || !token(reference.conversion)) {
    return invalid_record(std::string(subject) + " has an unknown reference enum value");
  }
  auto clock = validate_clock(reference.clock, subject);
  if (!clock) {
    return clock;
  }
  if (reference.conversion == TemporalRecordConversionKind::kSameClock &&
      !reference.conversion_record_id.empty()) {
    return invalid_record(std::string(subject) +
                          " same-clock reference must omit conversion provenance");
  }
  if (reference.conversion == TemporalRecordConversionKind::kDirectAffine &&
      !valid_text(reference.conversion_record_id)) {
    return invalid_record(std::string(subject) +
                          " direct conversion requires a stable conversion-record identity");
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_state(const TemporalPredicateStateRecord& state) {
  if (!token(state.reset_policy)) {
    return invalid_record("predicate state has an unknown reset policy");
  }
  if (!state.present) {
    if (state.size_bytes != 0 || state.alignment_bytes != 0 || state.max_bytes != 0 ||
        !state.schema.empty()) {
      return invalid_record("absent predicate state must have zero bounds and no schema");
    }
    return {};
  }
  if (state.size_bytes == 0 || !power_of_two(state.alignment_bytes) ||
      state.max_bytes < state.size_bytes || !valid_text(state.schema)) {
    return invalid_record("present predicate state requires finite size/alignment/bound/schema");
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_config(const TemporalPredicateConfigRecord& config) {
  if (!config.present) {
    if (config.size_bytes != 0 || config.alignment_bytes != 0 || config.max_bytes != 0 ||
        !config.schema.empty() || !config.digest.empty()) {
      return invalid_record("absent predicate config must have zero bounds and no schema/digest");
    }
    return {};
  }
  if (config.size_bytes == 0 || !power_of_two(config.alignment_bytes) ||
      config.max_bytes < config.size_bytes || !valid_text(config.schema) ||
      !valid_text(config.digest)) {
    return invalid_record("present predicate config requires finite bounds, schema, and digest");
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_predicate_identity(
    const TemporalPredicateRecord& predicate) {
  if (!valid_text(predicate.record_id) || !valid_text(predicate.identity) ||
      !token(predicate.scope) || !token(predicate.source) || !token(predicate.firing) ||
      !token(predicate.callable.kind) || !token(predicate.replay)) {
    return invalid_record("predicate record has an invalid identity or unknown enum value");
  }
  if (predicate.scope == TemporalPredicateScope::kQualified &&
      !predicate.semantic_version.has_value()) {
    return invalid_record("qualified predicate identity requires a semantic version");
  }
  if (predicate.scope == TemporalPredicateScope::kOperatorLocal &&
      predicate.semantic_version.has_value()) {
    return invalid_record("operator-local predicate identity must omit semantic version");
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_predicate_source(
    const TemporalPredicateRecord& predicate) {
  if (predicate.source == TemporalPredicateSourceKind::kInputUpdate) {
    if (!valid_text(predicate.driver_input) || predicate.period_ns != 0 ||
        predicate.phase.has_value() || !default_clock(predicate.clock)) {
      return invalid_record("input predicate requires a driver and no clock schedule");
    }
    return {};
  }
  if (!predicate.driver_input.empty() || predicate.period_ns <= 0) {
    return invalid_record("clock predicate requires a positive period and no driver");
  }
  auto clock = validate_clock(predicate.clock, "predicate clock");
  if (!clock) {
    return clock;
  }
  if (predicate.phase.has_value() && !predicate.phase->valid()) {
    return invalid_record("predicate phase must carry a resolved clock domain");
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_predicate_inputs(
    const TemporalPredicateRecord& predicate) {
  std::vector<std::string> input_names;
  input_names.reserve(predicate.additional_inputs.size());
  for (const auto& input : predicate.additional_inputs) {
    if (!valid_text(input.input_name) || input.input_name == predicate.driver_input) {
      return invalid_record("predicate inputs must be valid, unique, and exclude the driver");
    }
    input_names.push_back(input.input_name);
  }
  if (!all_unique(input_names)) {
    return invalid_record("predicate additional input names must be unique");
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_predicate_callable(
    const TemporalPredicateCallableRecord& callable) {
  if (!valid_text(callable.implementation_fingerprint) || !valid_text(callable.build_evidence) ||
      callable.size_bytes == 0 || !power_of_two(callable.alignment_bytes)) {
    return invalid_record("predicate callable requires pointer-free identity and finite layout");
  }
  if (callable.kind == PredicateCallableKind::kBoundMember) {
    if (!callable.owned_lifetime || !callable.self_rebind) {
      return invalid_record("bound-member predicate callable requires owned self rebinding");
    }
  } else if (callable.self_rebind) {
    return invalid_record("only a bound-member predicate callable may request self rebinding");
  }
  const bool value_owned = callable.kind == PredicateCallableKind::kInlineFunctor ||
                           callable.kind == PredicateCallableKind::kOpaqueClosure ||
                           callable.kind == PredicateCallableKind::kTypeErasedFunction ||
                           callable.kind == PredicateCallableKind::kPythonFunction ||
                           callable.kind == PredicateCallableKind::kPythonLambda ||
                           callable.kind == PredicateCallableKind::kPythonBoundMethod;
  if (value_owned && !callable.owned_lifetime) {
    return invalid_record("value predicate callable requires session-owned lifetime evidence");
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_predicate_replay(
    const TemporalPredicateRecord& predicate) {
  if (predicate.replay != TemporalPredicateReplayKind::kCrossBuildRecomputation &&
      !predicate.replay_evidence.empty()) {
    return invalid_record("non-cross-build predicate must omit recomputation evidence");
  }
  if (predicate.replay == TemporalPredicateReplayKind::kCrossBuildRecomputation &&
      (predicate.scope != TemporalPredicateScope::kQualified ||
       !valid_text(predicate.replay_evidence))) {
    return invalid_record("cross-build predicate requires qualified replay evidence");
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_predicate(const TemporalPredicateRecord& predicate) {
  auto validation = validate_predicate_identity(predicate);
  if (!validation) {
    return validation;
  }
  validation = validate_predicate_source(predicate);
  if (!validation) {
    return validation;
  }
  validation = validate_predicate_inputs(predicate);
  if (!validation) {
    return validation;
  }
  validation = validate_predicate_callable(predicate.callable);
  if (!validation) {
    return validation;
  }
  validation = validate_state(predicate.state);
  if (!validation) {
    return validation;
  }
  validation = validate_config(predicate.config);
  if (!validation) {
    return validation;
  }
  return validate_predicate_replay(predicate);
}

[[nodiscard]] expected<void, Error> validate_aligned(const TemporalAlignedRecord& aligned) {
  if (aligned.member_inputs.size() < 2 || !all_unique(aligned.member_inputs) ||
      std::any_of(aligned.member_inputs.begin(),
                  aligned.member_inputs.end(),
                  [](const std::string& name) { return !valid_text(name); }) ||
      aligned.inclusive_window_ns < 0 ||
      aligned.algorithm != "holoscan.temporal.aligned-oldest-head/v1" || !aligned.no_scan_ahead ||
      !aligned.consume_accepted_once ||
      aligned.reference.kind != TemporalRecordReferenceKind::kAlignedHigh) {
    return invalid_record("aligned record violates the closed oldest-head contract");
  }
  return validate_reference(aligned.reference, "aligned reference");
}

[[nodiscard]] expected<void, Error> validate_window(const TemporalWindowRecord& window) {
  if (!valid_text(window.anchor_input) || window.max_wait_ns <= 0 ||
      window.max_pending_windows == 0 ||
      window.reference.kind != TemporalRecordReferenceKind::kWindowClose ||
      !window.previous_current_anchor_roles || !window.strict_outer_samples ||
      !window.inclusive_interior || !window.close_on_timeout || !window.close_on_end_of_stream ||
      !window.retain_overlap || !window.reset_on_epoch_change ||
      !window.reset_on_timestamp_regression) {
    return invalid_record("window record violates the closed bracketed-window contract");
  }
  auto wait_clock = validate_clock(window.wait_clock, "window wait clock");
  if (!wait_clock) {
    return wait_clock;
  }
  return validate_reference(window.reference, "window reference");
}

class CanonicalJson {
 public:
  void append(std::string_view value) { text_.append(value); }

  void string(std::string_view value) {
    text_.push_back('"');
    for (const unsigned char byte : value) {
      switch (byte) {
        case '"':
          text_.append("\\\"");
          break;
        case '\\':
          text_.append("\\\\");
          break;
        case '\b':
          text_.append("\\b");
          break;
        case '\f':
          text_.append("\\f");
          break;
        case '\n':
          text_.append("\\n");
          break;
        case '\r':
          text_.append("\\r");
          break;
        case '\t':
          text_.append("\\t");
          break;
        default:
          if (byte < 0x20U) {
            constexpr std::array kHex{
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
            text_.append("\\u00");
            text_.push_back(kHex[(byte >> 4U) & 0x0fU]);
            text_.push_back(kHex[byte & 0x0fU]);
          } else {
            text_.push_back(static_cast<char>(byte));
          }
      }
    }
    text_.push_back('"');
  }

  template <typename Integer>
  void integer(Integer value) {
    std::array<char, 32> buffer{};
    const auto [end, error] = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    if (error == std::errc{}) {
      text_.append(buffer.data(), end);
    }
  }

  void boolean(bool value) { text_.append(value ? "true" : "false"); }

  [[nodiscard]] std::string finish() && { return std::move(text_); }

 private:
  std::string text_;
};

void write_key(CanonicalJson& json, std::string_view key, bool& first) {
  if (!first) {
    json.append(",");
  }
  first = false;
  json.string(key);
  json.append(":");
}

void write_clock(CanonicalJson& json, const TemporalClockKeyRecord& clock) {
  json.append("{");
  bool first = true;
  write_key(json, "default", first);
  json.boolean(clock.is_default);
  write_key(json, "name", first);
  json.string(clock.name);
  json.append("}");
}

void write_time_point(CanonicalJson& json, const std::optional<TimePoint>& point) {
  if (!point.has_value()) {
    json.append("null");
    return;
  }
  json.append("{");
  bool first = true;
  write_key(json, "timestamp_ns", first);
  json.integer(point->timestamp_ns);
  write_key(json, "clock_id", first);
  json.integer(point->clock_id.value);
  json.append("}");
}

void write_provenance(CanonicalJson& json, const TemporalRecordProvenance& provenance) {
  json.append("{");
  bool first = true;
  write_key(json, "source", first);
  json.string(provenance.source);
  write_key(json, "declaration_order", first);
  json.integer(provenance.declaration_order);
  json.append("}");
}

void write_reference(CanonicalJson& json, const TemporalReferenceRecord& reference) {
  json.append("{");
  bool first = true;
  write_key(json, "kind", first);
  json.string(serialized_token(reference.kind));
  write_key(json, "clock", first);
  write_clock(json, reference.clock);
  write_key(json, "conversion", first);
  json.string(serialized_token(reference.conversion));
  write_key(json, "conversion_record_id", first);
  json.string(reference.conversion_record_id);
  json.append("}");
}

void write_aligned(CanonicalJson& json, const TemporalAlignedRecord& aligned) {
  json.append("{");
  bool first = true;
  write_key(json, "members", first);
  json.append("[");
  for (std::size_t index = 0; index < aligned.member_inputs.size(); ++index) {
    if (index != 0) {
      json.append(",");
    }
    json.string(aligned.member_inputs[index]);
  }
  json.append("]");
  write_key(json, "inclusive_window_ns", first);
  json.integer(aligned.inclusive_window_ns);
  write_key(json, "reference", first);
  write_reference(json, aligned.reference);
  write_key(json, "algorithm", first);
  json.string(aligned.algorithm);
  write_key(json, "no_scan_ahead", first);
  json.boolean(aligned.no_scan_ahead);
  write_key(json, "consume_accepted_once", first);
  json.boolean(aligned.consume_accepted_once);
  json.append("}");
}

void write_window(CanonicalJson& json, const TemporalWindowRecord& window) {
  json.append("{");
  bool first = true;
  write_key(json, "anchor_input", first);
  json.string(window.anchor_input);
  write_key(json, "wait_clock", first);
  write_clock(json, window.wait_clock);
  write_key(json, "max_wait_ns", first);
  json.integer(window.max_wait_ns);
  write_key(json, "max_pending_windows", first);
  json.integer(window.max_pending_windows);
  write_key(json, "reference", first);
  write_reference(json, window.reference);
  write_key(json, "previous_current_anchor_roles", first);
  json.boolean(window.previous_current_anchor_roles);
  write_key(json, "strict_outer_samples", first);
  json.boolean(window.strict_outer_samples);
  write_key(json, "inclusive_interior", first);
  json.boolean(window.inclusive_interior);
  write_key(json, "close_on_timeout", first);
  json.boolean(window.close_on_timeout);
  write_key(json, "close_on_end_of_stream", first);
  json.boolean(window.close_on_end_of_stream);
  write_key(json, "retain_overlap", first);
  json.boolean(window.retain_overlap);
  write_key(json, "reset_on_epoch_change", first);
  json.boolean(window.reset_on_epoch_change);
  write_key(json, "reset_on_timestamp_regression", first);
  json.boolean(window.reset_on_timestamp_regression);
  json.append("}");
}

void write_semantic_version(CanonicalJson& json, const std::optional<SemanticVersion>& version) {
  if (!version.has_value()) {
    json.append("null");
    return;
  }
  json.append("{");
  bool first = true;
  write_key(json, "major", first);
  json.integer(version->major);
  write_key(json, "minor", first);
  json.integer(version->minor);
  write_key(json, "patch", first);
  json.integer(version->patch);
  json.append("}");
}

void write_callable(CanonicalJson& json, const TemporalPredicateCallableRecord& callable) {
  json.append("{");
  bool first = true;
  write_key(json, "kind", first);
  json.string(serialized_token(callable.kind));
  write_key(json, "implementation_fingerprint", first);
  json.string(callable.implementation_fingerprint);
  write_key(json, "build_evidence", first);
  json.string(callable.build_evidence);
  write_key(json, "size_bytes", first);
  json.integer(callable.size_bytes);
  write_key(json, "alignment_bytes", first);
  json.integer(callable.alignment_bytes);
  write_key(json, "no_dynamic_allocation", first);
  json.boolean(callable.no_dynamic_allocation);
  write_key(json, "bounded_call", first);
  json.boolean(callable.bounded_call);
  write_key(json, "noexcept", first);
  json.boolean(callable.is_noexcept);
  write_key(json, "owned_lifetime", first);
  json.boolean(callable.owned_lifetime);
  write_key(json, "self_rebind", first);
  json.boolean(callable.self_rebind);
  json.append("}");
}

void write_state(CanonicalJson& json, const TemporalPredicateStateRecord& state) {
  json.append("{");
  bool first = true;
  write_key(json, "present", first);
  json.boolean(state.present);
  write_key(json, "size_bytes", first);
  json.integer(state.size_bytes);
  write_key(json, "alignment_bytes", first);
  json.integer(state.alignment_bytes);
  write_key(json, "max_bytes", first);
  json.integer(state.max_bytes);
  write_key(json, "schema", first);
  json.string(state.schema);
  write_key(json, "reset_policy", first);
  json.string(serialized_token(state.reset_policy));
  json.append("}");
}

void write_config(CanonicalJson& json, const TemporalPredicateConfigRecord& config) {
  json.append("{");
  bool first = true;
  write_key(json, "present", first);
  json.boolean(config.present);
  write_key(json, "size_bytes", first);
  json.integer(config.size_bytes);
  write_key(json, "alignment_bytes", first);
  json.integer(config.alignment_bytes);
  write_key(json, "max_bytes", first);
  json.integer(config.max_bytes);
  write_key(json, "schema", first);
  json.string(config.schema);
  write_key(json, "digest", first);
  json.string(config.digest);
  json.append("}");
}

void write_predicate(CanonicalJson& json, const TemporalPredicateRecord& predicate) {
  json.append("{");
  bool first = true;
  write_key(json, "record_id", first);
  json.string(predicate.record_id);
  write_key(json, "identity", first);
  json.string(predicate.identity);
  write_key(json, "scope", first);
  json.string(serialized_token(predicate.scope));
  write_key(json, "semantic_version", first);
  write_semantic_version(json, predicate.semantic_version);
  write_key(json, "source", first);
  json.string(serialized_token(predicate.source));
  write_key(json, "driver_input", first);
  json.string(predicate.driver_input);
  write_key(json, "clock", first);
  write_clock(json, predicate.clock);
  write_key(json, "period_ns", first);
  json.integer(predicate.period_ns);
  write_key(json, "phase", first);
  write_time_point(json, predicate.phase);
  write_key(json, "additional_inputs", first);
  json.append("[");
  for (std::size_t index = 0; index < predicate.additional_inputs.size(); ++index) {
    if (index != 0) {
      json.append(",");
    }
    json.append("{");
    bool input_first = true;
    write_key(json, "input_name", input_first);
    json.string(predicate.additional_inputs[index].input_name);
    json.append("}");
  }
  json.append("]");
  write_key(json, "firing", first);
  json.string(serialized_token(predicate.firing));
  write_key(json, "callable", first);
  write_callable(json, predicate.callable);
  write_key(json, "state", first);
  write_state(json, predicate.state);
  write_key(json, "config", first);
  write_config(json, predicate.config);
  write_key(json, "replay", first);
  json.string(serialized_token(predicate.replay));
  write_key(json, "replay_evidence", first);
  json.string(predicate.replay_evidence);
  json.append("}");
}

void write_trigger_pins(CanonicalJson& json, std::span<const TemporalTriggerPinRecord> pins) {
  json.append("[");
  for (std::size_t index = 0; index < pins.size(); ++index) {
    if (index != 0U) {
      json.append(",");
    }
    const auto& pin = pins[index];
    json.append("{");
    bool first = true;
    write_key(json, "input_name", first);
    json.string(pin.input_name);
    write_key(json, "mode", first);
    json.string(serialized_token(pin.mode));
    write_key(json, "retained_pin_capacity", first);
    json.integer(pin.retained_pin_capacity);
    write_key(json, "capacity_origin", first);
    json.string(serialized_token(pin.capacity_origin));
    write_key(json, "provenance", first);
    write_provenance(json, pin.provenance);
    json.append("}");
  }
  json.append("]");
}

void write_contract(CanonicalJson& json, const ContractRecord& contract) {
  json.append("{");
  bool first = true;
  write_key(json, "version", first);
  json.integer(contract.version);
  write_key(json, "operator_name", first);
  json.string(contract.operator_name);
  write_key(json, "origin", first);
  json.string(serialized_token(contract.origin));
  write_key(json, "trigger", first);
  json.string(serialized_token(contract.trigger));
  write_key(json, "trigger_input", first);
  json.string(contract.trigger_input);
  write_key(json, "trigger_pins", first);
  write_trigger_pins(json, contract.trigger_pins);
  write_key(json, "trigger_source", first);
  json.integer(detail::source_id_value(contract.trigger_source));
  write_key(json, "source_provenance", first);
  json.string(contract.source_provenance);
  write_key(json, "trigger_clock", first);
  write_clock(json, contract.trigger_clock);
  write_key(json, "period_ns", first);
  json.integer(contract.period_ns);
  write_key(json, "phase", first);
  write_time_point(json, contract.phase);
  write_key(json, "aligned", first);
  if (contract.aligned.has_value()) {
    write_aligned(json, *contract.aligned);
  } else {
    json.append("null");
  }
  write_key(json, "window", first);
  if (contract.window.has_value()) {
    write_window(json, *contract.window);
  } else {
    json.append("null");
  }
  write_key(json, "predicate", first);
  if (contract.predicate.has_value()) {
    write_predicate(json, *contract.predicate);
  } else {
    json.append("null");
  }
  write_key(json, "deadline_ns", first);
  if (contract.deadline_ns.has_value()) {
    json.integer(*contract.deadline_ns);
  } else {
    json.append("null");
  }
  write_key(json, "violation", first);
  if (contract.violation.has_value()) {
    json.string(serialized_token(*contract.violation));
  } else {
    json.append("null");
  }
  write_key(json, "provenance", first);
  write_provenance(json, contract.provenance);
  json.append("}");
}

void write_edge(CanonicalJson& json, const EdgeSelectionRecord& edge) {
  json.append("{");
  bool first = true;
  write_key(json, "version", first);
  json.integer(edge.version);
  write_key(json, "input_name", first);
  json.string(edge.input_name);
  write_key(json, "selector", first);
  json.string(serialized_token(edge.selector));
  write_key(json, "cardinality", first);
  json.string(serialized_token(edge.cardinality));
  write_key(json, "max_age_ns", first);
  json.integer(edge.max_age_ns);
  write_key(json, "lookbehind", first);
  json.integer(edge.lookbehind);
  write_key(json, "lookahead", first);
  json.integer(edge.lookahead);
  write_key(json, "reference", first);
  write_reference(json, edge.reference);
  write_key(json, "cache_capacity", first);
  json.integer(edge.cache_capacity);
  write_key(json, "cache_capacity_origin", first);
  json.string(serialized_token(edge.cache_capacity_origin));
  write_key(json, "max_selected_samples", first);
  json.integer(edge.max_selected_samples);
  write_key(json, "max_selected_origin", first);
  json.string(serialized_token(edge.max_selected_origin));
  write_key(json, "batch_mode", first);
  json.string(serialized_token(edge.batch_mode));
  write_key(json, "missing", first);
  json.string(serialized_token(edge.missing));
  write_key(json, "reuse_max_age_ns", first);
  json.integer(edge.reuse_max_age_ns);
  write_key(json, "provenance", first);
  write_provenance(json, edge.provenance);
  json.append("}");
}

[[nodiscard]] bool edge_names_unique(std::span<const EdgeSelectionRecord> edges) {
  std::unordered_set<std::string_view> names;
  names.reserve(edges.size());
  return std::all_of(edges.begin(), edges.end(), [&](const EdgeSelectionRecord& edge) {
    return names.insert(edge.input_name).second;
  });
}

[[nodiscard]] bool contains_edge(std::span<const EdgeSelectionRecord> edges,
                                 std::string_view input_name) {
  return std::any_of(edges.begin(), edges.end(), [&](const EdgeSelectionRecord& edge) {
    return edge.input_name == input_name;
  });
}

[[nodiscard]] const EdgeSelectionRecord* find_edge(std::span<const EdgeSelectionRecord> edges,
                                                   std::string_view input_name) {
  const auto match = std::find_if(
      edges.begin(), edges.end(), [&](const auto& edge) { return edge.input_name == input_name; });
  return match == edges.end() ? nullptr : &*match;
}

[[nodiscard]] bool scalar_predicate_edge(const EdgeSelectionRecord& edge) noexcept {
  return edge.cardinality == TemporalRecordCardinality::kScalar &&
         (edge.selector == TemporalRecordSelectorKind::kExact ||
          edge.selector == TemporalRecordSelectorKind::kLatestWithin);
}

[[nodiscard]] expected<void, Error> validate_edge_collection(
    std::span<const EdgeSelectionRecord> edges) {
  if (!edge_names_unique(edges)) {
    return invalid_record("temporal edge input names must be unique");
  }
  for (std::size_t index = 1; index < edges.size(); ++index) {
    if (edges[index - 1].provenance.declaration_order >=
        edges[index].provenance.declaration_order) {
      return invalid_record("temporal edges must use strictly increasing canonical order");
    }
  }
  return {};
}

[[nodiscard]] TemporalRecordReferenceKind contract_reference_kind(
    const ContractRecord& contract) noexcept {
  switch (contract.trigger) {
    case TemporalRecordTriggerKind::kAnyInput:
    case TemporalRecordTriggerKind::kEach:
      return TemporalRecordReferenceKind::kTriggerSample;
    case TemporalRecordTriggerKind::kClock:
      return TemporalRecordReferenceKind::kScheduledTick;
    case TemporalRecordTriggerKind::kReadable:
    case TemporalRecordTriggerKind::kNotified:
    case TemporalRecordTriggerKind::kDoorbell:
    case TemporalRecordTriggerKind::kSpin:
      return TemporalRecordReferenceKind::kReadinessWitness;
    case TemporalRecordTriggerKind::kAligned:
      return TemporalRecordReferenceKind::kAlignedHigh;
    case TemporalRecordTriggerKind::kWindow:
      return TemporalRecordReferenceKind::kWindowClose;
    case TemporalRecordTriggerKind::kPredicate:
      if (contract.predicate.has_value() &&
          contract.predicate->source == TemporalPredicateSourceKind::kClock) {
        return TemporalRecordReferenceKind::kPredicateTick;
      }
      return TemporalRecordReferenceKind::kPredicateDriver;
  }
  return TemporalRecordReferenceKind::kTriggerSample;
}

[[nodiscard]] const TemporalClockKeyRecord* contract_reference_clock(
    const ContractRecord& contract, std::span<const EdgeSelectionRecord> edges) noexcept {
  switch (contract.trigger) {
    case TemporalRecordTriggerKind::kClock:
      return &contract.trigger_clock;
    case TemporalRecordTriggerKind::kAligned:
      return contract.aligned ? &contract.aligned->reference.clock : nullptr;
    case TemporalRecordTriggerKind::kWindow:
      return contract.window ? &contract.window->reference.clock : nullptr;
    case TemporalRecordTriggerKind::kPredicate:
      if (!contract.predicate) {
        return nullptr;
      }
      if (contract.predicate->source == TemporalPredicateSourceKind::kClock) {
        return &contract.predicate->clock;
      }
      if (const auto* driver = find_edge(edges, contract.predicate->driver_input)) {
        return &driver->reference.clock;
      }
      return nullptr;
    case TemporalRecordTriggerKind::kAnyInput:
    case TemporalRecordTriggerKind::kEach:
    case TemporalRecordTriggerKind::kReadable:
    case TemporalRecordTriggerKind::kNotified:
    case TemporalRecordTriggerKind::kDoorbell:
    case TemporalRecordTriggerKind::kSpin:
      return nullptr;
  }
  return nullptr;
}

[[nodiscard]] expected<void, Error> validate_edge_references(
    const ContractRecord& contract, std::span<const EdgeSelectionRecord> edges) {
  const auto expected = contract_reference_kind(contract);
  if (std::any_of(edges.begin(), edges.end(), [expected](const auto& edge) {
        return edge.reference.kind != expected;
      })) {
    return invalid_record("edge reference does not match the normalized trigger coordinate");
  }
  const auto* expected_clock = contract_reference_clock(contract, edges);
  if (expected_clock != nullptr &&
      std::any_of(edges.begin(), edges.end(), [expected_clock](const auto& edge) {
        return edge.reference.clock != *expected_clock;
      })) {
    return invalid_record("edge reference clock does not match the trigger coordinate");
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_aligned_edges(
    const TemporalAlignedRecord& aligned, std::span<const EdgeSelectionRecord> edges) {
  for (const auto& member : aligned.member_inputs) {
    const auto* edge = find_edge(edges, member);
    if (edge == nullptr || edge->selector != TemporalRecordSelectorKind::kExact ||
        edge->cardinality != TemporalRecordCardinality::kScalar ||
        edge->reference.kind != TemporalRecordReferenceKind::kAlignedHigh) {
      return invalid_record("aligned member requires one normalized exact scalar edge");
    }
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_window_edges(
    const TemporalWindowRecord& window, std::span<const EdgeSelectionRecord> edges) {
  const auto* anchor = find_edge(edges, window.anchor_input);
  if (anchor == nullptr || anchor->selector != TemporalRecordSelectorKind::kExact ||
      anchor->cardinality != TemporalRecordCardinality::kScalar ||
      anchor->reference.kind != TemporalRecordReferenceKind::kWindowClose) {
    return invalid_record("window anchor requires one normalized exact scalar edge");
  }
  const auto bracketed = std::find_if(edges.begin(), edges.end(), [](const auto& edge) {
    return edge.selector == TemporalRecordSelectorKind::kBracketedWindow;
  });
  if (bracketed == edges.end()) {
    return invalid_record("window contract requires at least one bracketed edge");
  }
  if (std::any_of(edges.begin(), edges.end(), [](const auto& edge) {
        return edge.selector == TemporalRecordSelectorKind::kBracketedWindow &&
               edge.reference.kind != TemporalRecordReferenceKind::kWindowClose;
      })) {
    return invalid_record("bracketed edge requires the normalized window-close reference");
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_predicate_edges(
    const TemporalPredicateRecord& predicate, std::span<const EdgeSelectionRecord> edges) {
  if (!predicate.driver_input.empty()) {
    const auto* driver = find_edge(edges, predicate.driver_input);
    if (driver == nullptr || driver->selector != TemporalRecordSelectorKind::kExact ||
        driver->cardinality != TemporalRecordCardinality::kScalar ||
        driver->missing != TemporalRecordMissingKind::kUnspecified ||
        driver->reference.kind != TemporalRecordReferenceKind::kPredicateDriver) {
      return invalid_record(
          "predicate driver requires one exact scalar edge without missing policy");
    }
  }
  const auto reference = predicate.source == TemporalPredicateSourceKind::kInputUpdate
                             ? TemporalRecordReferenceKind::kPredicateDriver
                             : TemporalRecordReferenceKind::kPredicateTick;
  for (const auto& input : predicate.additional_inputs) {
    const auto* edge = find_edge(edges, input.input_name);
    if (edge == nullptr || !scalar_predicate_edge(*edge) ||
        edge->missing == TemporalRecordMissingKind::kUnspecified ||
        edge->reference.kind != reference) {
      return invalid_record(
          "predicate additional input requires scalar selection and missing policy");
    }
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_presence_set_semantics(
    const ContractRecord& contract, std::span<const EdgeSelectionRecord> edges) {
  for (const auto& edge : edges) {
    const bool aligned_member = contract.aligned.has_value() &&
                                std::find(contract.aligned->member_inputs.begin(),
                                          contract.aligned->member_inputs.end(),
                                          edge.input_name) != contract.aligned->member_inputs.end();
    const bool window_anchor =
        contract.window.has_value() && edge.input_name == contract.window->anchor_input;
    const bool predicate_driver = contract.predicate.has_value() &&
                                  !contract.predicate->driver_input.empty() &&
                                  edge.input_name == contract.predicate->driver_input;

    if ((window_anchor || predicate_driver) &&
        edge.missing != TemporalRecordMissingKind::kUnspecified) {
      return invalid_record("guaranteed trigger-presence edge must omit missing policy");
    }
    if (aligned_member || window_anchor || predicate_driver ||
        edge.selector == TemporalRecordSelectorKind::kDrainAll) {
      continue;
    }
    if (edge.missing == TemporalRecordMissingKind::kUnspecified) {
      return invalid_record(
          "failure-capable edge outside the guaranteed presence set requires missing policy");
    }
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_cross_record_shape(
    const ContractRecord& contract, std::span<const EdgeSelectionRecord> edges) {
  auto validation = validate_edge_collection(edges);
  if (!validation) {
    return validation;
  }
  validation = validate_edge_references(contract, edges);
  if (!validation) {
    return validation;
  }

  // AnyInput and OnEach pin the readiness-causing sample without manufacturing an explicit
  // selector edge. EA1-009 proves the AnyInput connected catalog; OnEach must remain absent from
  // selector edges because reselecting its pinned driver is structurally invalid.
  if (contract.trigger == TemporalRecordTriggerKind::kEach &&
      contains_edge(edges, contract.trigger_input)) {
    return invalid_record("each trigger input must not be reselected as a temporal edge");
  }
  if (!contract.trigger_input.empty() && contract.trigger != TemporalRecordTriggerKind::kEach &&
      !contains_edge(edges, contract.trigger_input)) {
    return invalid_record("contract trigger input is absent from normalized edges");
  }
  if (contract.aligned.has_value()) {
    validation = validate_aligned_edges(*contract.aligned, edges);
    if (!validation) {
      return validation;
    }
  }
  if (contract.window.has_value()) {
    validation = validate_window_edges(*contract.window, edges);
    if (!validation) {
      return validation;
    }
  }
  if (contract.predicate.has_value()) {
    validation = validate_predicate_edges(*contract.predicate, edges);
    if (!validation) {
      return validation;
    }
  }
  return validate_presence_set_semantics(contract, edges);
}

[[nodiscard]] expected<void, Error> validate_trigger_pins(const ContractRecord& record) {
  const bool input_readiness = record.trigger == TemporalRecordTriggerKind::kAnyInput ||
                               record.trigger == TemporalRecordTriggerKind::kEach;
  if (!input_readiness) {
    return record.trigger_pins.empty()
               ? expected<void, Error>{}
               : invalid_record("only any-input and each triggers may carry trigger pins");
  }
  if (record.trigger_pins.empty()) {
    return invalid_record("input-readiness trigger requires at least one resolved trigger pin");
  }
  if (record.trigger == TemporalRecordTriggerKind::kEach &&
      (record.trigger_pins.size() != 1U ||
       record.trigger_pins.front().input_name != record.trigger_input)) {
    return invalid_record("each trigger requires exactly one pin matching its trigger input");
  }

  std::vector<std::string> names;
  names.reserve(record.trigger_pins.size());
  for (std::size_t index = 0; index < record.trigger_pins.size(); ++index) {
    const auto& pin = record.trigger_pins[index];
    if (!valid_text(pin.input_name) || !token(pin.mode) || pin.retained_pin_capacity == 0U ||
        pin.capacity_origin != TemporalRecordBoundOrigin::kResolvedRetention) {
      return invalid_record("trigger pin requires accepted-arrival mode and finite retention");
    }
    auto provenance = validate_provenance(pin.provenance, "trigger pin");
    if (!provenance) {
      return provenance;
    }
    if (index != 0U && record.trigger_pins[index - 1U].provenance.declaration_order >=
                           pin.provenance.declaration_order) {
      return invalid_record("trigger pins must use strictly increasing canonical order");
    }
    names.push_back(pin.input_name);
  }
  if (!all_unique(names)) {
    return invalid_record("trigger pin input names must be unique");
  }
  return {};
}

}  // namespace

// Keep the closed trigger alternatives together so every irrelevant field is rejected in one
// auditable dispatch. Splitting the switch would obscure the canonical one-shape-per-trigger rule.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> validate_temporal_record(const ContractRecord& record) {
  if (record.version != kTemporalRecordVersion || !valid_text(record.operator_name) ||
      !token(record.origin) || !token(record.trigger)) {
    return invalid_record("contract record has unsupported version, identity, or enum value");
  }
  auto provenance = validate_provenance(record.provenance, "contract");
  if (!provenance) {
    return provenance;
  }
  if (record.origin == TemporalRecordOrigin::kCoreSynthesized &&
      record.trigger != TemporalRecordTriggerKind::kAnyInput) {
    return invalid_record("only the any-input default may be Core-synthesized");
  }
  auto pins = validate_trigger_pins(record);
  if (!pins) {
    return pins;
  }
  if (record.deadline_ns.has_value() != record.violation.has_value() ||
      (record.deadline_ns.has_value() && *record.deadline_ns <= 0) ||
      (record.violation.has_value() && !token(*record.violation))) {
    return invalid_record("deadline and violation must form one valid positive pair");
  }

  const bool readiness = record.trigger == TemporalRecordTriggerKind::kReadable ||
                         record.trigger == TemporalRecordTriggerKind::kNotified ||
                         record.trigger == TemporalRecordTriggerKind::kDoorbell ||
                         record.trigger == TemporalRecordTriggerKind::kSpin;
  if (readiness) {
    if (!record.trigger_source || !valid_text(record.source_provenance)) {
      return invalid_record("readiness trigger requires logical source identity and provenance");
    }
  } else if (record.trigger_source || !record.source_provenance.empty()) {
    return invalid_record("non-readiness trigger must omit logical source fields");
  }

  switch (record.trigger) {
    case TemporalRecordTriggerKind::kAnyInput:
      if (!record.trigger_input.empty() || record.period_ns != 0 || record.phase.has_value() ||
          !default_clock(record.trigger_clock)) {
        return invalid_record("any-input record must defer the actual triggering input to runtime");
      }
      break;
    case TemporalRecordTriggerKind::kEach:
      if (!valid_text(record.trigger_input) || record.period_ns != 0 || record.phase.has_value() ||
          !default_clock(record.trigger_clock)) {
        return invalid_record("each record requires one input and no clock schedule");
      }
      break;
    case TemporalRecordTriggerKind::kClock: {
      if (!record.trigger_input.empty() || record.period_ns <= 0 ||
          (record.phase.has_value() && !record.phase->valid())) {
        return invalid_record("clock record requires positive period and optional valid phase");
      }
      auto clock = validate_clock(record.trigger_clock, "trigger clock");
      if (!clock) {
        return clock;
      }
      break;
    }
    case TemporalRecordTriggerKind::kReadable:
    case TemporalRecordTriggerKind::kNotified:
    case TemporalRecordTriggerKind::kDoorbell:
    case TemporalRecordTriggerKind::kSpin:
      if (!record.trigger_input.empty() || record.period_ns != 0 || record.phase.has_value() ||
          !default_clock(record.trigger_clock)) {
        return invalid_record("readiness record must omit input and clock schedule fields");
      }
      break;
    case TemporalRecordTriggerKind::kAligned:
      if (!record.aligned.has_value() || !record.trigger_input.empty() || record.period_ns != 0 ||
          record.phase.has_value() || !default_clock(record.trigger_clock)) {
        return invalid_record("aligned trigger requires its normalized aligned record");
      }
      if (auto result = validate_aligned(*record.aligned); !result) {
        return result;
      }
      break;
    case TemporalRecordTriggerKind::kWindow:
      if (!record.window.has_value() || record.trigger_input != record.window->anchor_input ||
          record.period_ns != 0 || record.phase.has_value() ||
          !default_clock(record.trigger_clock)) {
        return invalid_record("window trigger requires its normalized window record");
      }
      if (auto result = validate_window(*record.window); !result) {
        return result;
      }
      break;
    case TemporalRecordTriggerKind::kPredicate:
      if (!record.predicate.has_value() || record.trigger_input != record.predicate->driver_input ||
          record.period_ns != 0 || record.phase.has_value() ||
          !default_clock(record.trigger_clock)) {
        return invalid_record("predicate trigger requires its normalized predicate record");
      }
      if (auto result = validate_predicate(*record.predicate); !result) {
        return result;
      }
      break;
  }

  if ((record.trigger == TemporalRecordTriggerKind::kAligned) != record.aligned.has_value() ||
      (record.trigger == TemporalRecordTriggerKind::kWindow) != record.window.has_value() ||
      (record.trigger == TemporalRecordTriggerKind::kPredicate) != record.predicate.has_value()) {
    return invalid_record("contract carries a record that does not match its trigger kind");
  }
  return {};
}

expected<void, Error> validate_temporal_record(const EdgeSelectionRecord& record) {
  if (record.version != kTemporalRecordVersion || !valid_text(record.input_name) ||
      !token(record.selector) || !token(record.cardinality) || !token(record.batch_mode) ||
      !token(record.missing) || !token(record.cache_capacity_origin) ||
      !token(record.max_selected_origin)) {
    return invalid_record("edge record has unsupported version, identity, or enum value");
  }
  if (record.cache_capacity == 0 || record.max_selected_samples == 0 ||
      record.cache_capacity < record.max_selected_samples) {
    return invalid_record("edge record requires finite cache and selected-sample bounds");
  }
  auto reference = validate_reference(record.reference, "edge reference");
  if (!reference) {
    return reference;
  }
  auto provenance = validate_provenance(record.provenance, "edge");
  if (!provenance) {
    return provenance;
  }

  switch (record.selector) {
    case TemporalRecordSelectorKind::kExact:
      if (record.cardinality != TemporalRecordCardinality::kScalar ||
          record.batch_mode != TemporalRecordBatchMode::kScalar || record.max_age_ns != 0 ||
          record.lookbehind != 0 || record.lookahead != 0 || record.max_selected_samples != 1 ||
          record.max_selected_origin != TemporalRecordBoundOrigin::kSingleSample) {
        return invalid_record("exact selector requires the canonical scalar shape");
      }
      break;
    case TemporalRecordSelectorKind::kLatestWithin:
      if (record.cardinality != TemporalRecordCardinality::kScalar ||
          record.batch_mode != TemporalRecordBatchMode::kScalar || record.max_age_ns <= 0 ||
          record.lookbehind != 0 || record.lookahead != 0 || record.max_selected_samples != 1 ||
          record.max_selected_origin != TemporalRecordBoundOrigin::kSingleSample) {
        return invalid_record("latest-within selector requires finite scalar shape");
      }
      break;
    case TemporalRecordSelectorKind::kDrainAll:
      if (record.cardinality != TemporalRecordCardinality::kBatch ||
          record.batch_mode != TemporalRecordBatchMode::kConsumeOnAcceptance ||
          record.max_age_ns != 0 || record.lookbehind != 0 || record.lookahead != 0 ||
          record.max_selected_origin != TemporalRecordBoundOrigin::kResolvedRetention ||
          record.missing != TemporalRecordMissingKind::kUnspecified) {
        return invalid_record("drain-all selector requires resolved consuming batch shape");
      }
      break;
    case TemporalRecordSelectorKind::kBracketedWindow: {
      const std::uint64_t outer_sample_count = static_cast<std::uint64_t>(record.lookbehind) +
                                               static_cast<std::uint64_t>(record.lookahead);
      if (record.cardinality != TemporalRecordCardinality::kBatch ||
          record.batch_mode != TemporalRecordBatchMode::kNondestructiveSnapshot ||
          record.max_age_ns != 0 || record.lookbehind == 0 || record.lookahead == 0 ||
          record.max_selected_origin != TemporalRecordBoundOrigin::kSelectorMaximum ||
          record.max_selected_samples < outer_sample_count ||
          record.missing == TemporalRecordMissingKind::kUnspecified ||
          record.missing == TemporalRecordMissingKind::kReuseLast) {
        return invalid_record("bracketed selector requires bounded nondestructive batch shape");
      }
      break;
    }
  }

  if (record.missing == TemporalRecordMissingKind::kReuseLast) {
    if (record.cardinality != TemporalRecordCardinality::kScalar || record.reuse_max_age_ns <= 0) {
      return invalid_record("reuse-last requires scalar shape and a positive finite age");
    }
  } else if (record.reuse_max_age_ns != 0) {
    return invalid_record("non-reuse missing policy must omit reuse age");
  }
  return {};
}

expected<std::string, Error> serialize_temporal_records(
    const ContractRecord& contract, std::span<const EdgeSelectionRecord> edges) {
  auto contract_validation = validate_temporal_record(contract);
  if (!contract_validation) {
    return make_unexpected(std::move(contract_validation.error()));
  }
  for (const auto& edge : edges) {
    auto edge_validation = validate_temporal_record(edge);
    if (!edge_validation) {
      return make_unexpected(std::move(edge_validation.error()));
    }
  }
  auto cross_validation = validate_cross_record_shape(contract, edges);
  if (!cross_validation) {
    return make_unexpected(std::move(cross_validation.error()));
  }

  CanonicalJson json;
  json.append("{");
  bool first = true;
  write_key(json, "schema", first);
  json.string(kTemporalRecordSchema);
  write_key(json, "version", first);
  json.integer(kTemporalRecordVersion);
  write_key(json, "contract", first);
  write_contract(json, contract);
  write_key(json, "edges", first);
  json.append("[");
  for (std::size_t index = 0; index < edges.size(); ++index) {
    if (index != 0) {
      json.append(",");
    }
    write_edge(json, edges[index]);
  }
  json.append("]}");
  return std::move(json).finish();
}

}  // namespace holoscan
