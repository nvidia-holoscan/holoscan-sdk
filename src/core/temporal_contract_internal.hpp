/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <holoscan/core/temporal_contract.hpp>

namespace holoscan::detail {

enum class TemporalTriggerKind : std::uint8_t {
  kNone,
  kAnyInput,
  kEach,
  kClock,
  kReadable,
  kNotified,
  kDoorbell,
  kSpin,
  kAligned,
  kWindow,
  kPredicate,
};

enum class TemporalSelectorKind : std::uint8_t {
  kExact,
  kLatestWithin,
  kDrainAll,
  kBracketedWindow,
};

enum class TemporalMissingKind : std::uint8_t {
  kUnspecified,
  kSkip,
  kSafeDefault,
  kReuseLast,
};

struct TemporalInputRecord {
  std::string port_name;
  const void* local_type_token{};
  TemporalSelectorKind selector{TemporalSelectorKind::kExact};
  std::chrono::nanoseconds max_age{};
  ParamDurationDependency max_age_dependency{};
  BracketedWindow bracket{};
  TemporalMissingKind missing{TemporalMissingKind::kUnspecified};
  std::chrono::nanoseconds reuse_max_age{};
};

struct TemporalTriggerRecord {
  TemporalTriggerKind kind{TemporalTriggerKind::kNone};
  std::string port_name;
  ClockKey clock{};
  std::optional<TimePoint> start_at{};
  std::chrono::nanoseconds duration{};
  ClockTickPolicy tick_policy{ClockTickPolicy::kCoalesceLateTicks};
  std::uint32_t capacity{};
  std::vector<std::string> ports;
  SourceId source_id{};
  PredicateBinding predicate{};
  std::vector<std::string> predicate_input_names;
  std::optional<SemanticVersion> predicate_semantic_version;
  PredicateFiring predicate_firing{PredicateFiring::kOnFalseToTrue};
  bool predicate_clock_source{};
  bool predicate_expects_driver{};
  const void* predicate_driver_type_token{};
  const void* predicate_receiver_type_token{};
  std::vector<const void*> predicate_input_type_tokens;
  bool predicate_callable_noexcept{};
  bool predicate_callable_returns_expected{};
  std::size_t predicate_callable_size{};
  std::size_t predicate_callable_alignment{};
  std::size_t predicate_state_size{};
  std::size_t predicate_state_alignment{};
  std::size_t predicate_config_size{};
  std::size_t predicate_config_alignment{};
  bool predicate_config_identity_available{};
  std::string predicate_config_schema;
  std::string predicate_config_digest;
};

struct TemporalContractSnapshot {
  TemporalTriggerRecord trigger;
  std::vector<TemporalInputRecord> inputs;
  std::size_t connected_input_count{};
  std::optional<std::chrono::nanoseconds> deadline;
  ParamDurationDependency deadline_dependency{};
  std::optional<OnViolation> violation;
  bool synthesized_default{};
};

class ContractTestAccess {
 public:
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<TemporalContractSnapshot, Error> normalize(
      const Contract& contract, std::size_t connected_input_count) noexcept;

  [[nodiscard]] HOLOSCAN_CORE_EXPORT static std::size_t trigger_count(
      const Contract& contract) noexcept;

  [[nodiscard]] HOLOSCAN_CORE_EXPORT static bool empty(const Contract& contract) noexcept;

  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<std::unique_ptr<PredicateCallableStorage>,
                                                     Error>
  clone_predicate_storage(const PredicateBinding& binding) noexcept;
};

}  // namespace holoscan::detail
