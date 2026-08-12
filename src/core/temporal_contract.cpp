/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <holoscan/core/temporal_contract.hpp>

#include "temporal_contract_internal.hpp"

namespace holoscan {

class PredicateBinding::Impl {
 public:
  Impl() = default;

  Impl(std::string name, std::optional<SemanticVersion> version, PredicateCallableKind kind,
       bool has_state, bool has_config, bool expects_driver, bool callable_noexcept,
       bool callable_returns_expected, std::size_t callable_size, std::size_t callable_alignment,
       std::size_t state_size, std::size_t state_alignment, std::size_t config_size,
       std::size_t config_alignment, bool config_identity_available, std::string config_schema,
       std::string config_digest, const void* driver_type_token,
       std::vector<std::string> input_names, std::vector<const void*> input_type_tokens,
       std::unique_ptr<detail::PredicateCallableStorage> storage)
      : name_(std::move(name)),
        version_(version),
        kind_(kind),
        has_state_(has_state),
        has_config_(has_config),
        expects_driver_(expects_driver),
        callable_noexcept_(callable_noexcept),
        callable_returns_expected_(callable_returns_expected),
        callable_size_(callable_size),
        callable_alignment_(callable_alignment),
        state_size_(state_size),
        state_alignment_(state_alignment),
        config_size_(config_size),
        config_alignment_(config_alignment),
        config_identity_available_(config_identity_available),
        config_schema_(std::move(config_schema)),
        config_digest_(std::move(config_digest)),
        driver_type_token_(driver_type_token),
        receiver_type_token_(storage ? storage->receiver_type_token() : nullptr),
        input_names_(std::move(input_names)),
        input_type_tokens_(std::move(input_type_tokens)),
        storage_(std::move(storage)) {}

  Impl(const Impl& other)
      : name_(other.name_),
        version_(other.version_),
        kind_(other.kind_),
        has_state_(other.has_state_),
        has_config_(other.has_config_),
        expects_driver_(other.expects_driver_),
        callable_noexcept_(other.callable_noexcept_),
        callable_returns_expected_(other.callable_returns_expected_),
        callable_size_(other.callable_size_),
        callable_alignment_(other.callable_alignment_),
        state_size_(other.state_size_),
        state_alignment_(other.state_alignment_),
        config_size_(other.config_size_),
        config_alignment_(other.config_alignment_),
        config_identity_available_(other.config_identity_available_),
        config_schema_(other.config_schema_),
        config_digest_(other.config_digest_),
        driver_type_token_(other.driver_type_token_),
        receiver_type_token_(other.receiver_type_token_),
        input_names_(other.input_names_),
        input_type_tokens_(other.input_type_tokens_),
        storage_(other.storage_ ? other.storage_->clone() : nullptr) {}

  std::string name_;
  std::optional<SemanticVersion> version_;
  PredicateCallableKind kind_{PredicateCallableKind::kStaticFunction};
  bool has_state_{};
  bool has_config_{};
  bool expects_driver_{};
  bool callable_noexcept_{};
  bool callable_returns_expected_{};
  std::size_t callable_size_{};
  std::size_t callable_alignment_{};
  std::size_t state_size_{};
  std::size_t state_alignment_{};
  std::size_t config_size_{};
  std::size_t config_alignment_{};
  bool config_identity_available_{};
  std::string config_schema_;
  std::string config_digest_;
  const void* driver_type_token_{};
  const void* receiver_type_token_{};
  std::vector<std::string> input_names_;
  std::vector<const void*> input_type_tokens_;
  std::unique_ptr<detail::PredicateCallableStorage> storage_;
};

PredicateBinding::PredicateBinding() : impl_(std::make_unique<Impl>()) {}

PredicateBinding::PredicateBinding(
    std::string name, std::optional<SemanticVersion> version, PredicateCallableKind kind,
    bool has_state, bool has_config, bool expects_driver, bool callable_noexcept,
    bool callable_returns_expected, std::size_t callable_size, std::size_t callable_alignment,
    std::size_t state_size, std::size_t state_alignment, std::size_t config_size,
    std::size_t config_alignment, bool config_identity_available, std::string config_schema,
    std::string config_digest, const void* driver_type_token, std::vector<std::string> input_names,
    std::vector<const void*> input_type_tokens,
    std::unique_ptr<detail::PredicateCallableStorage> storage)
    : impl_(std::make_unique<Impl>(
          std::move(name), version, kind, has_state, has_config, expects_driver, callable_noexcept,
          callable_returns_expected, callable_size, callable_alignment, state_size, state_alignment,
          config_size, config_alignment, config_identity_available, std::move(config_schema),
          std::move(config_digest), driver_type_token, std::move(input_names),
          std::move(input_type_tokens), std::move(storage))) {}

PredicateBinding::PredicateBinding(const PredicateBinding& other)
    : impl_(other.impl_ ? std::make_unique<Impl>(*other.impl_) : std::make_unique<Impl>()) {}

PredicateBinding& PredicateBinding::operator=(const PredicateBinding& other) {
  if (this != &other) {
    auto replacement =
        other.impl_ ? std::make_unique<Impl>(*other.impl_) : std::make_unique<Impl>();
    impl_.swap(replacement);
  }
  return *this;
}

PredicateBinding::PredicateBinding(PredicateBinding&& other) noexcept = default;
PredicateBinding& PredicateBinding::operator=(PredicateBinding&& other) noexcept = default;
PredicateBinding::~PredicateBinding() = default;

PredicateBinding::operator bool() const noexcept {
  return impl_ != nullptr && !impl_->name_.empty() && impl_->storage_ != nullptr;
}

std::string_view PredicateBinding::name() const noexcept {
  return impl_ ? std::string_view{impl_->name_} : std::string_view{};
}

PredicateCallableKind PredicateBinding::callable_kind() const noexcept {
  return impl_ ? impl_->kind_ : PredicateCallableKind::kStaticFunction;
}

bool PredicateBinding::has_config() const noexcept {
  return impl_ && impl_->has_config_;
}

std::size_t PredicateBinding::input_count() const noexcept {
  return impl_ ? impl_->input_names_.size() : 0U;
}

bool PredicateBinding::expects_driver() const noexcept {
  return impl_ && impl_->expects_driver_;
}

class Contract::Impl {
 public:
  detail::TemporalTriggerRecord trigger_;
  std::size_t trigger_count_{};
  std::vector<detail::TemporalInputRecord> inputs_;
  std::vector<std::pair<std::string, std::variant<Skip, SafeDefault, ReuseLast>>> missing_;
  std::optional<std::chrono::nanoseconds> deadline_;
  detail::ParamDurationDependency deadline_dependency_{};
  std::optional<OnViolation> violation_;
};

namespace {

[[nodiscard]] std::string copy_input_name(PortRef port) {
  if (!port || port.direction() != PortDirection::kInput) {
    return {};
  }
  return std::string{port.name()};
}

[[nodiscard]] bool positive(std::chrono::nanoseconds value) noexcept {
  return value.count() > 0;
}

[[nodiscard]] bool valid_tick_policy(ClockTickPolicy value) noexcept {
  return value == ClockTickPolicy::kCoalesceLateTicks;
}

[[nodiscard]] bool valid_predicate_firing(PredicateFiring value) noexcept {
  return value == PredicateFiring::kOnFalseToTrue ||
         value == PredicateFiring::kOnEveryTrueEvaluation;
}

[[nodiscard]] bool valid_violation(OnViolation value) noexcept {
  return value == OnViolation::kSafeDefault || value == OnViolation::kEmitWitness ||
         value == OnViolation::kFailRun;
}

[[nodiscard]] unexpected<Error> invalid(std::string message) noexcept {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, std::move(message)});
}

[[nodiscard]] detail::TemporalMissingKind missing_kind(
    const std::variant<Skip, SafeDefault, ReuseLast>& value) noexcept {
  if (std::holds_alternative<Skip>(value)) {
    return detail::TemporalMissingKind::kSkip;
  }
  if (std::holds_alternative<SafeDefault>(value)) {
    return detail::TemporalMissingKind::kSafeDefault;
  }
  return detail::TemporalMissingKind::kReuseLast;
}

// Keep every closed trigger alternative in one exhaustive validator so a new enum value cannot
// bypass its shape checks by being routed through a disconnected helper.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> validate_trigger(
    const detail::TemporalContractSnapshot& snapshot) noexcept {
  const auto& trigger = snapshot.trigger;
  switch (trigger.kind) {
    case detail::TemporalTriggerKind::kNone:
      return invalid("NO_TRIGGER: a zero-input or partially authored contract needs one trigger");
    case detail::TemporalTriggerKind::kEach:
      if (trigger.port_name.empty()) {
        return invalid("OnEach requires one named local input");
      }
      break;
    case detail::TemporalTriggerKind::kClock:
      if (!positive(trigger.duration) || !valid_tick_policy(trigger.tick_policy)) {
        return invalid("OnClock period must be positive");
      }
      if (trigger.start_at && !trigger.start_at->valid()) {
        return invalid("OnClock start_at must be a valid absolute TimePoint");
      }
      break;
    case detail::TemporalTriggerKind::kReadable:
    case detail::TemporalTriggerKind::kNotified:
    case detail::TemporalTriggerKind::kDoorbell:
    case detail::TemporalTriggerKind::kSpin:
      if (!trigger.source_id) {
        return invalid("readiness trigger requires a bound logical source identity");
      }
      break;
    case detail::TemporalTriggerKind::kAligned:
      if (trigger.ports.size() < 2U || trigger.duration.count() < 0) {
        return invalid("OnAligned requires at least two inputs and a nonnegative window");
      }
      if (std::any_of(trigger.ports.begin(), trigger.ports.end(), [](const std::string& name) {
            return name.empty();
          })) {
        return invalid("OnAligned members must be named local inputs");
      }
      for (auto member = trigger.ports.begin(); member != trigger.ports.end(); ++member) {
        if (std::find(trigger.ports.begin(), member, *member) != member) {
          return invalid("OnAligned members must be unique");
        }
      }
      break;
    case detail::TemporalTriggerKind::kWindow:
      if (trigger.port_name.empty() || !positive(trigger.duration) || trigger.capacity == 0U) {
        return invalid("OnWindow requires an anchor, positive max_wait, and finite capacity");
      }
      break;
    case detail::TemporalTriggerKind::kPredicate:
      if (!trigger.predicate) {
        return invalid("OnPredicate requires a nonempty owned predicate binding");
      }
      if (!trigger.predicate_clock_source && trigger.port_name.empty()) {
        return invalid("OnInputUpdate requires one named local driver input");
      }
      if (trigger.predicate_clock_source && !positive(trigger.duration)) {
        return invalid("clock-driven OnPredicate requires a positive period");
      }
      if (trigger.predicate_clock_source && trigger.start_at && !trigger.start_at->valid()) {
        return invalid("clock-driven OnPredicate start_at must be a valid absolute TimePoint");
      }
      if ((trigger.predicate_clock_source && !valid_tick_policy(trigger.tick_policy)) ||
          !valid_predicate_firing(trigger.predicate_firing)) {
        return invalid("OnPredicate contains an invalid closed policy value");
      }
      if (trigger.predicate_clock_source == trigger.predicate_expects_driver) {
        return invalid(
            "predicate evaluator driver shape must match OnInputUpdate or clock evaluation");
      }
      break;
    case detail::TemporalTriggerKind::kAnyInput:
      break;
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_input(
    const detail::TemporalContractSnapshot& snapshot,
    std::vector<detail::TemporalInputRecord>::const_iterator current) noexcept {
  const auto& input = *current;
  if (input.port_name.empty()) {
    return invalid("contract input must name a local input declaration");
  }
  if (std::any_of(snapshot.inputs.begin(), current, [&](const auto& earlier) {
        return earlier.port_name == input.port_name;
      })) {
    return invalid("contract input selector is duplicated");
  }
  if (input.selector == detail::TemporalSelectorKind::kLatestWithin && !input.max_age_dependency &&
      !positive(input.max_age)) {
    return invalid("LatestWithin max_age must be positive");
  }
  if (input.selector == detail::TemporalSelectorKind::kBracketedWindow &&
      (input.bracket.lookbehind == 0U || input.bracket.lookahead == 0U ||
       input.bracket.max_samples == 0U)) {
    return invalid("BracketedWindow counts and max_samples must be positive");
  }
  if (input.missing == detail::TemporalMissingKind::kReuseLast) {
    if (!positive(input.reuse_max_age)) {
      return invalid("ReuseLast max_age must be positive and finite");
    }
    if (input.selector == detail::TemporalSelectorKind::kDrainAll ||
        input.selector == detail::TemporalSelectorKind::kBracketedWindow) {
      return invalid("ReuseLast is valid only for scalar selectors");
    }
  }
  if (input.selector == detail::TemporalSelectorKind::kBracketedWindow &&
      input.missing == detail::TemporalMissingKind::kUnspecified) {
    return invalid("BracketedWindow requires an explicit terminal missing policy");
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_window_inputs(
    const detail::TemporalContractSnapshot& snapshot) noexcept {
  const auto anchor =
      std::find_if(snapshot.inputs.begin(), snapshot.inputs.end(), [&](const auto& input) {
        return input.port_name == snapshot.trigger.port_name;
      });
  if (anchor == snapshot.inputs.end() || anchor->selector != detail::TemporalSelectorKind::kExact) {
    return invalid("OnWindow anchor must have an Exact selector");
  }
  if (std::none_of(snapshot.inputs.begin(), snapshot.inputs.end(), [](const auto& input) {
        return input.selector == detail::TemporalSelectorKind::kBracketedWindow;
      })) {
    return invalid("OnWindow requires at least one BracketedWindow input");
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_aligned_inputs(
    const detail::TemporalContractSnapshot& snapshot) noexcept {
  for (const auto& member : snapshot.trigger.ports) {
    const auto input =
        std::find_if(snapshot.inputs.begin(), snapshot.inputs.end(), [&](const auto& candidate) {
          return candidate.port_name == member;
        });
    if (input == snapshot.inputs.end() || input->selector != detail::TemporalSelectorKind::kExact) {
      return invalid("every OnAligned member requires an Exact selector");
    }
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_predicate_inputs(
    const detail::TemporalContractSnapshot& snapshot) noexcept {
  if (snapshot.trigger.predicate_input_names.size() !=
      snapshot.trigger.predicate_input_type_tokens.size()) {
    return invalid("predicate input names and payload types must have identical arity");
  }
  if (!snapshot.trigger.predicate_clock_source) {
    const auto driver =
        std::find_if(snapshot.inputs.begin(), snapshot.inputs.end(), [&](const auto& input) {
          return input.port_name == snapshot.trigger.port_name;
        });
    if (driver == snapshot.inputs.end() ||
        driver->selector != detail::TemporalSelectorKind::kExact) {
      return invalid("OnInputUpdate driver must have an Exact selector");
    }
    if (driver->local_type_token == nullptr ||
        snapshot.trigger.predicate_driver_type_token == nullptr ||
        driver->local_type_token != snapshot.trigger.predicate_driver_type_token) {
      return invalid("OnInputUpdate driver payload type does not match the evaluator signature");
    }
  }
  for (std::size_t index = 0; index < snapshot.trigger.predicate_input_names.size(); ++index) {
    const auto& name = snapshot.trigger.predicate_input_names[index];
    if ((!snapshot.trigger.predicate_clock_source && name == snapshot.trigger.port_name) ||
        std::find(
            snapshot.trigger.predicate_input_names.begin(),
            snapshot.trigger.predicate_input_names.begin() + static_cast<std::ptrdiff_t>(index),
            name) !=
            snapshot.trigger.predicate_input_names.begin() + static_cast<std::ptrdiff_t>(index)) {
      return invalid(
          "predicate additional inputs must be unique and must not duplicate the driver");
    }
    const auto input =
        std::find_if(snapshot.inputs.begin(), snapshot.inputs.end(), [&](const auto& candidate) {
          return candidate.port_name == name;
        });
    if (input == snapshot.inputs.end() ||
        (input->selector != detail::TemporalSelectorKind::kExact &&
         input->selector != detail::TemporalSelectorKind::kLatestWithin)) {
      return invalid("predicate inputs must be declared scalar Exact or LatestWithin selectors");
    }
    if (input->local_type_token == nullptr ||
        snapshot.trigger.predicate_input_type_tokens[index] == nullptr ||
        input->local_type_token != snapshot.trigger.predicate_input_type_tokens[index]) {
      return invalid("predicate additional input payload type does not match evaluator signature");
    }
  }
  return {};
}

[[nodiscard]] bool trigger_guarantees_input(const detail::TemporalContractSnapshot& snapshot,
                                            std::string_view port_name) noexcept {
  switch (snapshot.trigger.kind) {
    case detail::TemporalTriggerKind::kEach:
    case detail::TemporalTriggerKind::kWindow:
      return snapshot.trigger.port_name == port_name;
    case detail::TemporalTriggerKind::kAligned:
      return std::find(snapshot.trigger.ports.begin(), snapshot.trigger.ports.end(), port_name) !=
             snapshot.trigger.ports.end();
    case detail::TemporalTriggerKind::kPredicate:
      return !snapshot.trigger.predicate_clock_source && snapshot.trigger.port_name == port_name;
    default:
      return false;
  }
}

[[nodiscard]] expected<void, Error> validate_presence_policy(
    const detail::TemporalContractSnapshot& snapshot) noexcept {
  for (const auto& input : snapshot.inputs) {
    const bool guaranteed = trigger_guarantees_input(snapshot, input.port_name);
    const bool failure_capable = input.selector == detail::TemporalSelectorKind::kExact ||
                                 input.selector == detail::TemporalSelectorKind::kLatestWithin ||
                                 input.selector == detail::TemporalSelectorKind::kBracketedWindow;
    if (failure_capable && !guaranteed &&
        input.missing == detail::TemporalMissingKind::kUnspecified) {
      return invalid("a failure-capable selector outside P(trigger) requires on_missing");
    }
    if (guaranteed && input.missing != detail::TemporalMissingKind::kUnspecified &&
        snapshot.trigger.kind != detail::TemporalTriggerKind::kAligned) {
      return invalid("on_missing is redundant for an input guaranteed present by its trigger");
    }
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_inputs(
    const detail::TemporalContractSnapshot& snapshot) noexcept {
  for (auto current = snapshot.inputs.begin(); current != snapshot.inputs.end(); ++current) {
    auto result = validate_input(snapshot, current);
    if (!result) {
      return result;
    }
  }

  if (snapshot.trigger.kind == detail::TemporalTriggerKind::kEach &&
      std::any_of(snapshot.inputs.begin(), snapshot.inputs.end(), [&](const auto& input) {
        return input.port_name == snapshot.trigger.port_name;
      })) {
    return invalid("OnEach pins its readiness sample and must not reselect its driver");
  }

  auto presence = validate_presence_policy(snapshot);
  if (!presence) {
    return presence;
  }

  switch (snapshot.trigger.kind) {
    case detail::TemporalTriggerKind::kWindow:
      return validate_window_inputs(snapshot);
    case detail::TemporalTriggerKind::kAligned:
      return validate_aligned_inputs(snapshot);
    case detail::TemporalTriggerKind::kPredicate:
      return validate_predicate_inputs(snapshot);
    default:
      return {};
  }
}

}  // namespace

namespace detail {

class ContractBuilderAccess {
 public:
  static void set_trigger(Contract& contract, TemporalTriggerRecord record) {
    ++contract.impl_->trigger_count_;
    contract.impl_->trigger_ = std::move(record);
  }

  template <typename Selector>
  static Contract& add_input(Contract& contract, PortRef port, Selector selector,
                             const void* local_type_token) {
    TemporalInputRecord record;
    record.port_name = copy_input_name(port);
    record.local_type_token = local_type_token;
    if constexpr (std::same_as<Selector, Exact>) {
      record.selector = TemporalSelectorKind::kExact;
    } else if constexpr (std::same_as<Selector, LatestWithin>) {
      record.selector = TemporalSelectorKind::kLatestWithin;
      record.max_age = selector.max_age.value;
      record.max_age_dependency = selector.max_age.dependency;
    } else if constexpr (std::same_as<Selector, DrainAll>) {
      record.selector = TemporalSelectorKind::kDrainAll;
    } else {
      record.selector = TemporalSelectorKind::kBracketedWindow;
      record.bracket = selector;
    }
    contract.impl_->inputs_.push_back(std::move(record));
    return contract;
  }

  template <typename Missing>
  static Contract& add_missing(Contract& contract, PortRef port, Missing strategy) {
    contract.impl_->missing_.emplace_back(copy_input_name(port), std::move(strategy));
    return contract;
  }

  [[nodiscard]] static std::vector<std::string> predicate_input_names(
      const PredicateBinding& binding) {
    return binding.impl_ ? binding.impl_->input_names_ : std::vector<std::string>{};
  }

  [[nodiscard]] static bool predicate_expects_driver(const PredicateBinding& binding) noexcept {
    return binding.expects_driver();
  }

  static void apply_predicate_metadata(TemporalTriggerRecord& record,
                                       const PredicateBinding& binding) {
    if (!binding.impl_) {
      return;
    }
    record.predicate_driver_type_token = binding.impl_->driver_type_token_;
    record.predicate_receiver_type_token = binding.impl_->receiver_type_token_;
    record.predicate_input_type_tokens = binding.impl_->input_type_tokens_;
    record.predicate_semantic_version = binding.impl_->version_;
    record.predicate_callable_noexcept = binding.impl_->callable_noexcept_;
    record.predicate_callable_returns_expected = binding.impl_->callable_returns_expected_;
    record.predicate_callable_size = binding.impl_->callable_size_;
    record.predicate_callable_alignment = binding.impl_->callable_alignment_;
    record.predicate_state_size = binding.impl_->state_size_;
    record.predicate_state_alignment = binding.impl_->state_alignment_;
    record.predicate_config_size = binding.impl_->config_size_;
    record.predicate_config_alignment = binding.impl_->config_alignment_;
    record.predicate_config_identity_available = binding.impl_->config_identity_available_;
    record.predicate_config_schema = binding.impl_->config_schema_;
    record.predicate_config_digest = binding.impl_->config_digest_;
  }
};

}  // namespace detail

Contract::Contract() : impl_(std::make_unique<Impl>()) {}
Contract::Contract(const Contract& other)
    : impl_(other.impl_ ? std::make_unique<Impl>(*other.impl_) : std::make_unique<Impl>()) {}

Contract& Contract::operator=(const Contract& other) {
  if (this != &other) {
    auto replacement =
        other.impl_ ? std::make_unique<Impl>(*other.impl_) : std::make_unique<Impl>();
    impl_.swap(replacement);
  }
  return *this;
}

Contract::Contract(Contract&& other) noexcept = default;
Contract& Contract::operator=(Contract&& other) noexcept = default;
Contract::~Contract() = default;

Contract& Contract::trigger(OnAnyInput) {
  detail::TemporalTriggerRecord record;
  record.kind = detail::TemporalTriggerKind::kAnyInput;
  detail::ContractBuilderAccess::set_trigger(*this, std::move(record));
  return *this;
}

Contract& Contract::trigger(OnEach trigger) {
  detail::TemporalTriggerRecord record;
  record.kind = detail::TemporalTriggerKind::kEach;
  record.port_name = copy_input_name(trigger.port);
  detail::ContractBuilderAccess::set_trigger(*this, std::move(record));
  return *this;
}

Contract& Contract::trigger(OnClock trigger) {
  detail::TemporalTriggerRecord record;
  record.kind = detail::TemporalTriggerKind::kClock;
  record.clock = std::move(trigger.clock);
  record.start_at = trigger.start_at;
  record.duration = trigger.period;
  record.tick_policy = trigger.tick_policy;
  detail::ContractBuilderAccess::set_trigger(*this, std::move(record));
  return *this;
}

Contract& Contract::trigger(OnReadable trigger) {
  detail::TemporalTriggerRecord record;
  record.kind = detail::TemporalTriggerKind::kReadable;
  record.source_id = trigger.event.id();
  detail::ContractBuilderAccess::set_trigger(*this, std::move(record));
  return *this;
}

Contract& Contract::trigger(OnNotified trigger) {
  detail::TemporalTriggerRecord record;
  record.kind = detail::TemporalTriggerKind::kNotified;
  record.source_id = trigger.event.id();
  detail::ContractBuilderAccess::set_trigger(*this, std::move(record));
  return *this;
}

Contract& Contract::trigger(OnDoorbell trigger) {
  detail::TemporalTriggerRecord record;
  record.kind = detail::TemporalTriggerKind::kDoorbell;
  record.source_id = trigger.event.id();
  detail::ContractBuilderAccess::set_trigger(*this, std::move(record));
  return *this;
}

Contract& Contract::trigger(OnSpin trigger) {
  detail::TemporalTriggerRecord record;
  record.kind = detail::TemporalTriggerKind::kSpin;
  record.source_id = trigger.event.id();
  detail::ContractBuilderAccess::set_trigger(*this, std::move(record));
  return *this;
}

Contract& Contract::trigger(const OnAligned& trigger) {
  detail::TemporalTriggerRecord record;
  record.kind = detail::TemporalTriggerKind::kAligned;
  record.duration = trigger.window;
  record.ports.reserve(trigger.ports.values().size());
  for (const auto port : trigger.ports.values()) {
    record.ports.push_back(copy_input_name(port));
  }
  detail::ContractBuilderAccess::set_trigger(*this, std::move(record));
  return *this;
}

Contract& Contract::trigger(OnWindow trigger) {
  detail::TemporalTriggerRecord record;
  record.kind = detail::TemporalTriggerKind::kWindow;
  record.port_name = copy_input_name(trigger.anchor);
  record.duration = trigger.max_wait;
  record.clock = std::move(trigger.wait_clock);
  record.capacity = trigger.max_pending_windows;
  detail::ContractBuilderAccess::set_trigger(*this, std::move(record));
  return *this;
}

Contract& Contract::trigger(OnPredicate trigger) {
  detail::TemporalTriggerRecord record;
  record.kind = detail::TemporalTriggerKind::kPredicate;
  record.predicate = std::move(trigger.predicate);
  record.predicate_input_names =
      detail::ContractBuilderAccess::predicate_input_names(record.predicate);
  record.predicate_expects_driver =
      detail::ContractBuilderAccess::predicate_expects_driver(record.predicate);
  detail::ContractBuilderAccess::apply_predicate_metadata(record, record.predicate);
  record.predicate_firing = trigger.firing;
  if (const auto* input = std::get_if<OnInputUpdate>(&trigger.evaluate_on)) {
    record.port_name = copy_input_name(input->port);
  } else {
    const auto& clock = std::get<OnClock>(trigger.evaluate_on);
    record.predicate_clock_source = true;
    record.clock = clock.clock;
    record.start_at = clock.start_at;
    record.duration = clock.period;
    record.tick_policy = clock.tick_policy;
  }
  detail::ContractBuilderAccess::set_trigger(*this, std::move(record));
  return *this;
}

Contract& Contract::input(PortRef port, Exact selector) {
  return detail::ContractBuilderAccess::add_input(*this, port, selector, nullptr);
}
Contract& Contract::input(PortRef port, LatestWithin selector) {
  return detail::ContractBuilderAccess::add_input(*this, port, selector, nullptr);
}
Contract& Contract::input(PortRef port, DrainAll selector) {
  return detail::ContractBuilderAccess::add_input(*this, port, selector, nullptr);
}
Contract& Contract::input(PortRef port, BracketedWindow selector) {
  return detail::ContractBuilderAccess::add_input(*this, port, selector, nullptr);
}

Contract& Contract::input_typed(PortRef port, Exact selector, const void* local_type_token) {
  return detail::ContractBuilderAccess::add_input(*this, port, selector, local_type_token);
}
Contract& Contract::input_typed(PortRef port, LatestWithin selector, const void* local_type_token) {
  return detail::ContractBuilderAccess::add_input(*this, port, selector, local_type_token);
}
Contract& Contract::input_typed(PortRef port, DrainAll selector, const void* local_type_token) {
  return detail::ContractBuilderAccess::add_input(*this, port, selector, local_type_token);
}
Contract& Contract::input_typed(PortRef port, BracketedWindow selector,
                                const void* local_type_token) {
  return detail::ContractBuilderAccess::add_input(*this, port, selector, local_type_token);
}

Contract& Contract::on_missing(PortRef port, Skip strategy) {
  return detail::ContractBuilderAccess::add_missing(*this, port, strategy);
}
Contract& Contract::on_missing(PortRef port, SafeDefault strategy) {
  return detail::ContractBuilderAccess::add_missing(*this, port, strategy);
}
Contract& Contract::on_missing(PortRef port, ReuseLast strategy) {
  return detail::ContractBuilderAccess::add_missing(*this, port, strategy);
}

Contract& Contract::deadline(std::chrono::nanoseconds budget) {
  impl_->deadline_ = budget;
  impl_->deadline_dependency_ = {};
  return *this;
}

Contract& Contract::deadline_dependency(detail::ParamDurationDependency dependency) {
  impl_->deadline_.reset();
  impl_->deadline_dependency_ = dependency;
  return *this;
}

Contract& Contract::on_violation(OnViolation policy) {
  impl_->violation_ = policy;
  return *this;
}

}  // namespace holoscan

namespace holoscan::detail {

// Normalization is one cold transaction: copy authoring state, apply defaults, merge missing
// policies, and validate the final snapshot before exposing any result.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<TemporalContractSnapshot, Error> ContractTestAccess::normalize(
    const Contract& contract, std::size_t connected_input_count) noexcept {
  try {
    TemporalContractSnapshot snapshot;
    snapshot.trigger = contract.impl_->trigger_;
    snapshot.inputs = contract.impl_->inputs_;
    snapshot.connected_input_count = connected_input_count;
    snapshot.deadline = contract.impl_->deadline_;
    snapshot.deadline_dependency = contract.impl_->deadline_dependency_;
    snapshot.violation = contract.impl_->violation_;

    const bool has_nontrigger_content =
        !snapshot.inputs.empty() || !contract.impl_->missing_.empty() ||
        snapshot.deadline.has_value() || snapshot.deadline_dependency ||
        snapshot.violation.has_value();
    if (contract.impl_->trigger_count_ == 0U && !has_nontrigger_content &&
        connected_input_count > 0U) {
      snapshot.trigger.kind = TemporalTriggerKind::kAnyInput;
      snapshot.synthesized_default = true;
    } else {
      if (contract.impl_->trigger_count_ != 1U) {
        return invalid(
            contract.impl_->trigger_count_ == 0U
                ? "NO_TRIGGER: a zero-input or partially authored contract needs one trigger"
                : "a contract must declare exactly one trigger");
      }
      if (snapshot.trigger.kind == TemporalTriggerKind::kAnyInput && connected_input_count == 0U) {
        return invalid("NO_TRIGGER: OnAnyInput requires at least one connected input");
      }
      auto result = validate_trigger(snapshot);
      if (!result) {
        return make_unexpected(std::move(result).error());
      }
    }

    for (const auto& missing_entry : contract.impl_->missing_) {
      const auto& port_name = missing_entry.first;
      const auto& missing = missing_entry.second;
      auto input = snapshot.inputs.begin();
      while (input != snapshot.inputs.end() && input->port_name != port_name) {
        ++input;
      }
      if (input == snapshot.inputs.end()) {
        return invalid("missing policy must reference an explicitly selected input");
      }
      if (input->missing != TemporalMissingKind::kUnspecified) {
        return invalid("missing policy is duplicated for one input");
      }
      input->missing = missing_kind(missing);
      if (const auto* reuse = std::get_if<ReuseLast>(&missing)) {
        input->reuse_max_age = reuse->max_age;
      }
    }

    auto input_result = validate_inputs(snapshot);
    if (!input_result) {
      return make_unexpected(std::move(input_result).error());
    }
    const bool has_deadline = snapshot.deadline.has_value() || snapshot.deadline_dependency;
    if (has_deadline != snapshot.violation.has_value()) {
      return invalid("deadline and on_violation must be declared together");
    }
    if (snapshot.violation && !valid_violation(*snapshot.violation)) {
      return invalid("deadline contains an invalid OnViolation value");
    }
    if (snapshot.deadline && !positive(*snapshot.deadline)) {
      return invalid("deadline budget must be positive");
    }
    return snapshot;
  } catch (const std::exception& error) {
    return make_unexpected(Error{ErrorCode::kFailure, error.what()});
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kFailure, "contract normalization failed"});
  }
}

bool ContractTestAccess::empty(const Contract& contract) noexcept {
  return contract.impl_->trigger_count_ == 0U && contract.impl_->inputs_.empty() &&
         contract.impl_->missing_.empty() && !contract.impl_->deadline_ &&
         !contract.impl_->deadline_dependency_ && !contract.impl_->violation_;
}

std::size_t ContractTestAccess::trigger_count(const Contract& contract) noexcept {
  return contract.impl_->trigger_count_;
}

expected<std::unique_ptr<PredicateCallableStorage>, Error>
ContractTestAccess::clone_predicate_storage(const PredicateBinding& binding) noexcept {
  try {
    if (!binding.impl_ || !binding.impl_->storage_) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    return binding.impl_->storage_->clone();
  } catch (const std::exception& error) {
    return make_unexpected(Error{ErrorCode::kFailure, error.what()});
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kFailure});
  }
}

}  // namespace holoscan::detail
