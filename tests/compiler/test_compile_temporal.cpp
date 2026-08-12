/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/predicate.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "compile_test_graph.hpp"
#include "compiler/plan_artifact_lowering.hpp"
#include "compiler/profile_admission.hpp"
#include "compiler/runtime_package_capabilities.hpp"
#include "compiler/temporal_plan_lowering.hpp"
#include "core/graph_compiler_access.hpp"
#include "core/source_id_internal.hpp"

// GTest fatal assertions establish optional preconditions that clang-tidy does not model.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
namespace holoscan {

// Typed config structs for ConfiguredPredicateOperator template tests.
// PredicateConfig and ReorderedPredicateConfig share "test.predicate.threshold" so their
// digests match for equal threshold values regardless of struct layout differences.
// UnqualifiedPredicateConfig and OversizedPredicateConfig both trigger
// PREDICATE_CONFIG_IDENTITY_UNAVAILABLE.
struct PredicateConfig {
  std::uint32_t threshold{};
};

struct ReorderedPredicateConfig {
  std::uint8_t reserved{};
  std::uint32_t threshold{};
};

struct UnqualifiedPredicateConfig {
  std::uint8_t marker{};
  std::uint32_t threshold{};
};

struct EmptyPredicateConfig {};

struct OversizedPredicateConfig {
  std::uint32_t value{};
};

// Schema-parameterized struct with a runtime-mutable max_age and a recompile-required deadline.
// Powers ParameterDerivedTemporalOperator in the parameter-derived temporal selection tests.
struct TemporalDependencyParams {
  std::chrono::milliseconds max_age{3};
  std::chrono::microseconds deadline{5000};
};

HOLOSCAN_PARAM_SCHEMA(TemporalDependencyParams,
                      param_schema("nvidia.holoscan.test.TemporalDependencyParams", 1),
                      field<&TemporalDependencyParams::max_age>("max_age").runtime_mutable().range(
                          std::chrono::milliseconds{1}, std::chrono::milliseconds{10}),
                      field<&TemporalDependencyParams::deadline>("deadline")
                          .recompile_required()
                          .range(std::chrono::milliseconds{2}, std::chrono::milliseconds{20}));

template <>
struct PredicateConfigCodec<PredicateConfig> {
  static constexpr std::string_view schema{"test.predicate.threshold"};
  static constexpr std::uint32_t version{1U};
  static constexpr std::size_t max_encoded_bytes{4U};

  [[nodiscard]] static expected<std::vector<std::byte>, Error> encode(
      const PredicateConfig& value) {
    return std::vector<std::byte>{std::byte(value.threshold & 0xFFU),
                                  std::byte((value.threshold >> 8U) & 0xFFU),
                                  std::byte((value.threshold >> 16U) & 0xFFU),
                                  std::byte((value.threshold >> 24U) & 0xFFU)};
  }
};

template <>
struct PredicateConfigCodec<ReorderedPredicateConfig> {
  static constexpr std::string_view schema{"test.predicate.threshold"};
  static constexpr std::uint32_t version{1U};
  static constexpr std::size_t max_encoded_bytes{4U};

  [[nodiscard]] static expected<std::vector<std::byte>, Error> encode(
      const ReorderedPredicateConfig& value) {
    return PredicateConfigCodec<PredicateConfig>::encode(
        PredicateConfig{.threshold = value.threshold});
  }
};

template <>
struct PredicateConfigCodec<EmptyPredicateConfig> {
  static constexpr std::string_view schema{"test.predicate.empty"};
  static constexpr std::uint32_t version{1U};
  static constexpr std::size_t max_encoded_bytes{1U};

  [[nodiscard]] static expected<std::vector<std::byte>, Error> encode(const EmptyPredicateConfig&) {
    return std::vector<std::byte>{};
  }
};

template <>
struct PredicateConfigCodec<OversizedPredicateConfig> {
  static constexpr std::string_view schema{"test.predicate.oversized"};
  static constexpr std::uint32_t version{1U};
  static constexpr std::size_t max_encoded_bytes{1U};

  [[nodiscard]] static expected<std::vector<std::byte>, Error> encode(
      const OversizedPredicateConfig& value) {
    return std::vector<std::byte>{std::byte(value.value & 0xFFU), std::byte{0U}};
  }
};

namespace {

using namespace std::chrono_literals;

// Emits two uint32 outputs ("first" and "second"); provides connected inputs for all trigger-mode
// and selection tests (e.g. ReplaysEveryClosedTriggerAndFailsClosedOnMissingWindowEvidence).
class TemporalSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(first, "first");
    spec.output(second, "second");
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<std::uint32_t> first;
  Output<std::uint32_t> second;
};

class ImportedClockRelay final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(2);
    spec.output(output, "output");
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
  Output<std::uint32_t> output;
};

class ImportedClockRelayGraph final : public Graph {
 public:
  [[nodiscard]] ClockKey clock_import() const noexcept { return clock_import_; }

  Input<std::uint32_t> input{"input"};
  Output<std::uint32_t> output{"output"};

 protected:
  void compose() override {
    clock_import_ = import_clock("timeline");
    set_default_clock(clock_import_);
    const ClockKey local_clock = add_clock<ManualClock>("local_timeline", std::int64_t{0});
    add_clock_conversion(clock_import_, local_clock, AffineClockConversion{.offset_ns = 5});
    auto relay = op<ImportedClockRelay>("relay");
    export_input(input, relay->input);
    export_output(output, relay->output);
  }

 private:
  ClockKey clock_import_;
};

enum class TriggerMode : std::uint8_t {
  kAny,
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

// Exercises every trigger variant (Any/Each/Clock/Readable/Notified/Doorbell/Spin/Aligned/
// Window/Predicate) via a runtime TriggerMode enum, with an optional deadline violation policy.
// Used in ReplaysEveryClosedTriggerAndFailsClosedOnMissingWindowEvidence and deadline tests.
class TriggerMatrixOperator final : public Operator<> {
 public:
  explicit TriggerMatrixOperator(TriggerMode mode,
                                 std::optional<OnViolation> violation = std::nullopt)
      : mode_(mode), violation_(violation) {}

  void setup(OperatorSpec& spec) override {
    spec.input(first, "first").queue_depth(8);
    spec.input(second, "second").queue_depth(8);
    spec.readable_source(readable, "readable");
    spec.notification_source(notified, "notified").capacity(2).sender_reference_capacity(1);
    spec.doorbell_source(doorbell, "doorbell").capacity(2);
    spec.spin_source(spin, "spin");
  }

  Contract contract() const override {
    Contract result;
    switch (mode_) {
      case TriggerMode::kAny:
        result.trigger(OnAnyInput{});
        break;
      case TriggerMode::kEach:
        result.trigger(OnEach{first});
        break;
      case TriggerMode::kClock:
        result.trigger(OnClock{.period = 1ms});
        break;
      case TriggerMode::kReadable:
        result.trigger(OnReadable{readable});
        break;
      case TriggerMode::kNotified:
        result.trigger(OnNotified{notified});
        break;
      case TriggerMode::kDoorbell:
        result.trigger(OnDoorbell{doorbell});
        break;
      case TriggerMode::kSpin:
        result.trigger(OnSpin{spin});
        break;
      case TriggerMode::kAligned:
        result.trigger(OnAligned{.ports = ports(first, second), .window = 1ms})
            .input(first, Exact{})
            .input(second, Exact{});
        break;
      case TriggerMode::kWindow:
        result.trigger(OnWindow{.anchor = first, .max_wait = 2ms, .max_pending_windows = 2})
            .input(first, Exact{})
            .input(second, BracketedWindow{.lookbehind = 1, .lookahead = 1, .max_samples = 3})
            .on_missing(second, Skip{});
        break;
      case TriggerMode::kPredicate: {
        auto evaluator =
            predicate("driver-ready",
                      [](const PredicateEvaluationContext&,
                         const PredicateInputView<std::uint32_t>&) noexcept { return true; });
        result
            .trigger(
                OnPredicate{.evaluate_on = OnInputUpdate{first}, .predicate = std::move(evaluator)})
            .input(first, Exact{});
        break;
      }
    }
    if (violation_) {
      result.deadline(2ms).on_violation(*violation_);
    }
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> first;
  Input<std::uint32_t> second;
  ReadableSource readable;
  NotificationSource notified;
  DoorbellSource doorbell;
  SpinSource spin;

 private:
  TriggerMode mode_;
  std::optional<OnViolation> violation_;
};

// Minimal arrival-backed trigger fixture with no physical-readiness declarations.
// It isolates the default-clock admission boundary exercised by
// RejectsEveryArrivalBackedTriggerWithoutAResolvedClock.
class ArrivalTriggerWithoutClockOperator final : public Operator<> {
 public:
  explicit ArrivalTriggerWithoutClockOperator(TriggerMode mode) : mode_(mode) {}

  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(2U); }

  Contract contract() const override {
    Contract result;
    if (mode_ == TriggerMode::kAny) {
      result.trigger(OnAnyInput{});
    } else if (mode_ == TriggerMode::kEach) {
      result.trigger(OnEach{input});
    } else {
      auto evaluator =
          predicate("driver-ready",
                    [](const PredicateEvaluationContext&,
                       const PredicateInputView<std::uint32_t>&) noexcept { return true; });
      result
          .trigger(
              OnPredicate{.evaluate_on = OnInputUpdate{input}, .predicate = std::move(evaluator)})
          .input(input, Exact{});
    }
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;

 private:
  TriggerMode mode_;
};

// Readiness-triggered input selector whose reference clock comes from the
// physical source rather than an owning-Graph default.
class ReadinessClockSelectionOperator final : public Operator<> {
 public:
  explicit ReadinessClockSelectionOperator(TriggerMode mode) : mode_(mode) {}

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(2U);
    switch (mode_) {
      case TriggerMode::kReadable:
        spec.readable_source(readable, "readable").clock(ClockKey::named("readiness"));
        break;
      case TriggerMode::kNotified:
        spec.notification_source(notified, "notified")
            .capacity(2U)
            .sender_reference_capacity(1U)
            .clock(ClockKey::named("readiness"));
        break;
      case TriggerMode::kDoorbell:
        spec.doorbell_source(doorbell, "doorbell").capacity(2U).clock(ClockKey::named("readiness"));
        break;
      case TriggerMode::kSpin:
        spec.spin_source(spin, "spin").clock(ClockKey::named("readiness"));
        break;
      default:
        break;
    }
  }

  Contract contract() const override {
    Contract result;
    switch (mode_) {
      case TriggerMode::kReadable:
        result.trigger(OnReadable{readable});
        break;
      case TriggerMode::kNotified:
        result.trigger(OnNotified{notified});
        break;
      case TriggerMode::kDoorbell:
        result.trigger(OnDoorbell{doorbell});
        break;
      case TriggerMode::kSpin:
        result.trigger(OnSpin{spin});
        break;
      default:
        break;
    }
    result.input(input, Exact{}).on_missing(input, Skip{});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
  ReadableSource readable;
  NotificationSource notified;
  DoorbellSource doorbell;
  SpinSource spin;

 private:
  TriggerMode mode_;
};

// Clock-triggered operator with an explicit absolute TimePoint phase; the compiler must
// resolve the phase through the clock-conversion graph.
// Tests PreservesExplicitClockPhaseAndDirectConversion.
class ExplicitClockOperator final : public Operator<> {
 public:
  explicit ExplicitClockOperator(TimePoint phase) : phase_(phase) {}

  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 2ms, .start_at = phase_});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

 private:
  TimePoint phase_{};
};

// Clock-triggered operator that references a named clock by key rather than the graph default.
// Two instances referencing distinct clocks trigger TEMPORAL_SCHEDULING_DOMAIN_UNSUPPORTED.
// Tests RejectsMultipleSchedulingDomainsBeforeRuntimeConstruction.
class NamedClockOperator final : public Operator<> {
 public:
  explicit NamedClockOperator(ClockKey clock) : clock_(std::move(clock)) {}

  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms, .clock = clock_});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

 private:
  ClockKey clock_;
};

// Predicate with three inputs across two missing-aggregate phases: "driver" and "additional"
// are visible during predicate evaluation (LatestWithin+SafeDefault); "compute_only" is
// admitted only during the compute phase (DrainAll).
// Tests SeparatesPredicateEvaluationAndComputeOnlyAggregates.
class PredicatePhaseOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(driver, "driver").queue_depth(4);
    spec.input(additional, "additional").queue_depth(4);
    spec.input(compute_only, "compute_only").queue_depth(6);
  }

  Contract contract() const override {
    auto evaluator =
        predicate("phase-separated",
                  predicate_inputs(additional),
                  [](const PredicateEvaluationContext&,
                     const PredicateInputView<std::uint32_t>&,
                     const PredicateInputView<std::uint32_t>&) noexcept { return true; });
    Contract result;
    result
        .trigger(
            OnPredicate{.evaluate_on = OnInputUpdate{driver}, .predicate = std::move(evaluator)})
        .input(driver, Exact{})
        .input(additional, LatestWithin{.max_age = 2ms})
        .on_missing(additional, SafeDefault{})
        .input(compute_only, DrainAll{});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> driver;
  Input<std::uint32_t> additional;
  Input<std::uint32_t> compute_only;
};

// Embedded subgraph that wraps a TriggerMatrixOperator (kAny) with its own ManualClock default.
// Used in JoinsEmbeddedBoundaryOwnersThroughOperatorIdentityMap to verify that the compiler
// maps the inner operator reference through the embedding context's identity map.
class TemporalIdentityLeaf final : public Graph {
 public:
  Input<std::uint32_t> input{"input"};

 protected:
  void compose() override {
    auto target = op<TriggerMatrixOperator>("target", TriggerMode::kAny);
    export_input(input, target->first);
    const ClockKey clock = add_clock<ManualClock>("clock", std::int64_t{0});
    set_default_clock(clock);
  }
};

// Clock-triggered operator exercising three input selector kinds on separate ports: Exact
// (with Skip), LatestWithin (with ReuseLast), and DrainAll.
// Used in DerivesScalarAndDrainAllBoundsFromRetention and determinism tests.
class SelectorOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(exact, "exact").queue_depth(4);
    spec.input(latest, "latest").queue_depth(4);
    spec.input(drain, "drain").queue_depth(7);
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms})
        .input(drain, DrainAll{})
        .input(latest, LatestWithin{.max_age = 3ms})
        .on_missing(latest, ReuseLast{.max_age = 4ms})
        .input(exact, Exact{})
        .on_missing(exact, Skip{});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> exact;
  Input<std::uint32_t> latest;
  Input<std::uint32_t> drain;
};

// Clock-triggered operator whose LatestWithin max_age and contract deadline are both driven
// by param_ref<> fields from TemporalDependencyParams. The compiler must emit pointer-free
// ParameterDerivedTermRecord entries and omit raw pointer tokens from the JSON plan.
// Tests ResolvesBoundedDurationParameterDependenciesIntoPointerFreeTerms.
class ParameterDerivedTemporalOperator final : public Operator<TemporalDependencyParams> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(latest, "latest").queue_depth(4); }

  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms})
        .input(latest, LatestWithin{param_ref<&TemporalDependencyParams::max_age>()})
        .on_missing(latest, Skip{})
        .deadline(param_ref<&TemporalDependencyParams::deadline>())
        .on_violation(OnViolation::kFailRun);
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> latest;
};

// Template operator that attaches a typed Config struct to a predicate via PredicateConfigCodec.
// Tests AdmitsConfiglessAndRejectsUnidentifiedConfiguredValues for each Config specialization.
template <typename Config>
class ConfiguredPredicateOperator final : public Operator<> {
 public:
  explicit ConfiguredPredicateOperator(Config config) : config_(config) {}

  void setup(OperatorSpec& spec) override { spec.input(driver, "driver").queue_depth(2); }

  Contract contract() const override {
    Contract result;
    auto evaluator = predicate("configured",
                               config_,
                               [](const PredicateEvaluationContext&,
                                  const PredicateInputView<std::uint32_t>&,
                                  const Config&) noexcept { return true; });
    result.trigger(
        OnPredicate{.evaluate_on = OnInputUpdate{driver}, .predicate = std::move(evaluator)});
    result.input(driver, Exact{});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> driver;

 private:
  Config config_{};
};

// Predicate operator with no config attachment; verifies that the compiler records zero-byte
// config evidence and omits config schema/digest fields from the plan.
// Tests AdmitsConfiglessAndRejectsUnidentifiedConfiguredValues.
class ConfiglessPredicateOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(driver, "driver").queue_depth(2); }

  Contract contract() const override {
    auto evaluator =
        predicate("configless",
                  [](const PredicateEvaluationContext&,
                     const PredicateInputView<std::uint32_t>&) noexcept { return true; });
    Contract result;
    result
        .trigger(
            OnPredicate{.evaluate_on = OnInputUpdate{driver}, .predicate = std::move(evaluator)})
        .input(driver, Exact{});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> driver;
};

// Predicate operator identified by a PredicateKey with a semantic version triplet. The compiler
// upgrades replay capability to kCrossBuildRecomputation only when the runtime package manifest
// records an offer with a matching key and the local implementation fingerprint.
// Tests RequiresExactPackageFingerprintForCrossBuildRecomputation.
class KeyedPredicateOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(driver, "driver").queue_depth(2); }

  Contract contract() const override {
    auto evaluator =
        predicate(PredicateKey{"nvidia.holoscan.test.keyed", {1U, 2U, 3U}},
                  [](const PredicateEvaluationContext&,
                     const PredicateInputView<std::uint32_t>&) noexcept { return true; });
    Contract result;
    result
        .trigger(
            OnPredicate{.evaluate_on = OnInputUpdate{driver}, .predicate = std::move(evaluator)})
        .input(driver, Exact{});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> driver;
};

// Free function and stateful functor used by CallableFormPredicateOperator via factory types
// to exercise the three C++ callable dispatch forms: static, inline-owned, and bound member.
[[nodiscard]] bool static_predicate(const PredicateEvaluationContext&,
                                    const PredicateInputView<std::uint32_t>&) noexcept {
  return true;
}

struct InlinePredicateFunctor {
  std::uint32_t threshold{1U};

  [[nodiscard]] bool operator()(const PredicateEvaluationContext&,
                                const PredicateInputView<std::uint32_t>&) const noexcept {
    return threshold != 0U;
  }
};

// Template operator that delegates predicate construction to an EvaluatorFactory type parameter,
// enabling static/inline/bound dispatch in RecordsEveryEa1CallableDispatchForm.
template <typename EvaluatorFactory>
class CallableFormPredicateOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(driver, "driver").queue_depth(2); }

  Contract contract() const override {
    auto evaluator = EvaluatorFactory::make(*this);
    Contract result;
    result
        .trigger(
            OnPredicate{.evaluate_on = OnInputUpdate{driver}, .predicate = std::move(evaluator)})
        .input(driver, Exact{});
    return result;
  }

  [[nodiscard]] bool bound_predicate(const PredicateEvaluationContext&,
                                     const PredicateInputView<std::uint32_t>&) const noexcept {
    return true;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> driver;
};

// Wraps a free function pointer — produces dispatch_model="direct_function".
struct StaticPredicateFactory {
  template <typename OperatorType>
  [[nodiscard]] static auto make(const OperatorType&) {
    return predicate("static", &static_predicate);
  }
};

// Wraps a stateful inline functor — produces dispatch_model="inline_owned".
struct InlinePredicateFactory {
  template <typename OperatorType>
  [[nodiscard]] static auto make(const OperatorType&) {
    return predicate("inline", InlinePredicateFunctor{});
  }
};

// Wraps a bound member function pointer — produces dispatch_model="self_rebound_member".
struct BoundPredicateFactory {
  template <typename OperatorType>
  [[nodiscard]] static auto make(const OperatorType& owner) {
    return predicate("bound", bind_predicate(owner, &OperatorType::bound_predicate));
  }
};

// Stub PredicateCallableStorage that mimics Python-originated callables: evaluate() always
// returns kNotSupported. Paired with ReservedPythonPredicateOperator.
class ReservedPythonCallableStorage final : public detail::PredicateCallableStorage {
 public:
  [[nodiscard]] std::unique_ptr<detail::PredicateCallableStorage> clone() const override {
    return std::make_unique<ReservedPythonCallableStorage>();
  }

  [[nodiscard]] expected<bool, Error> evaluate(const PredicateEvaluationContext&,
                                               std::span<const void* const>,
                                               detail::PredicateReceiver) override {
    return make_unexpected(Error{ErrorCode::kNotSupported, "reserved Python predicate"});
  }

  void commit_candidate() noexcept override {}
  void discard_candidate() noexcept override {}
  [[nodiscard]] const void* receiver_type_token() const noexcept override { return nullptr; }
  void reset() override {}
};

// Injects a Python-reserved callable kind via PredicateBindingFactoryAccess; the compiler must
// reject all three Python kinds while still recording them as inspectable callable records.
// Tests KeepsReservedPythonCallableKindsRejectedAndInspectable.
class ReservedPythonPredicateOperator final : public Operator<> {
 public:
  explicit ReservedPythonPredicateOperator(PredicateCallableKind kind) : kind_(kind) {}

  void setup(OperatorSpec& spec) override { spec.input(driver, "driver").queue_depth(2); }

  Contract contract() const override {
    auto evaluator = detail::PredicateBindingFactoryAccess::make(
        "python-reserved",
        std::nullopt,
        kind_,
        false,
        false,
        true,
        false,
        false,
        1U,
        1U,
        0U,
        0U,
        0U,
        0U,
        false,
        {},
        {},
        detail::predicate_driver_type_token<std::uint32_t>(),
        {},
        {},
        std::make_unique<ReservedPythonCallableStorage>());
    Contract result;
    result
        .trigger(
            OnPredicate{.evaluate_on = OnInputUpdate{driver}, .predicate = std::move(evaluator)})
        .input(driver, Exact{});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> driver;

 private:
  PredicateCallableKind kind_;
};

// Predicate whose evaluator returns expected<bool, Error> rather than plain bool; the compiler
// must record result_kind=kExpectedBoolError.
// Tests RecordsExactResultAndOwnedTypeErasureAdmissionFacts.
class ExpectedPredicateOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(driver, "driver").queue_depth(2); }

  Contract contract() const override {
    auto evaluator =
        predicate("expected-result",
                  [](const PredicateEvaluationContext&,
                     const PredicateInputView<std::uint32_t>&) noexcept -> expected<bool, Error> {
                    return true;
                  });
    Contract result;
    result
        .trigger(
            OnPredicate{.evaluate_on = OnInputUpdate{driver}, .predicate = std::move(evaluator)})
        .input(driver, Exact{});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> driver;
};

// Predicate wrapped in std::function; compile-time type erasure forces kAdmittedWithWarning
// under Standard/Research profiles and kRejected under Robotics/RealTime/Regulated.
// Tests RecordsExactResultAndOwnedTypeErasureAdmissionFacts and the profile matrix test.
class TypeErasedPredicateOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(driver, "driver").queue_depth(2); }

  Contract contract() const override {
    std::function<bool(const PredicateEvaluationContext&, const PredicateInputView<std::uint32_t>&)>
        callable = [](const PredicateEvaluationContext&, const PredicateInputView<std::uint32_t>&) {
          return true;
        };
    auto evaluator = predicate("type-erased", std::move(callable));
    Contract result;
    result
        .trigger(
            OnPredicate{.evaluate_on = OnInputUpdate{driver}, .predicate = std::move(evaluator)})
        .input(driver, Exact{});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> driver;
};

// Lambda predicate that captures a local variable, producing a kOpaqueClosure callable.
// Tests AppliesTheClosedCallableProfileMatrix (captured closures rejected under kRobotics+).
class CapturedPredicateOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(driver, "driver").queue_depth(2); }

  Contract contract() const override {
    const std::uint32_t threshold = 1U;
    auto evaluator = predicate(
        "captured",
        [threshold](const PredicateEvaluationContext&,
                    const PredicateInputView<std::uint32_t>&) noexcept { return threshold != 0U; });
    Contract result;
    result
        .trigger(
            OnPredicate{.evaluate_on = OnInputUpdate{driver}, .predicate = std::move(evaluator)})
        .input(driver, Exact{});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> driver;
};

// Predicate evaluated on every OnClock tick (no input driver) with kOnEveryTrueEvaluation
// firing; the compiler must record capture-time from the clock rather than an input port.
// Tests FreezesClockSourceAndExactSnapshotSemantics.
class ClockPredicateOperator final : public Operator<> {
 public:
  Contract contract() const override {
    auto evaluator = predicate("clock-predicate",
                               [](const PredicateEvaluationContext&) noexcept { return true; });
    Contract result;
    result.trigger(OnPredicate{
        .evaluate_on = OnClock{.period = 2ms},
        .predicate = std::move(evaluator),
        .firing = PredicateFiring::kOnEveryTrueEvaluation,
    });
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Minimal clock-triggered operator with no inputs or outputs.
// Used in AllocatesPhysicalAndSynthesizedSourcesInOneStableIdentitySpace as the clock-sourced
// member alongside a physical-readiness operator, verifying synthesized ID ordering.
class DirectClockOperator final : public Operator<> {
 public:
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Operator with a physical ReadableSource trigger; provides the only physical readiness record
// in AllocatesPhysicalAndSynthesizedSourcesInOneStableIdentitySpace. Synthesized source IDs
// (for clock and predicate) must be numerically greater than the physical source ID.
class MixedPhysicalReadinessOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.readable_source(readable, "readable"); }

  Contract contract() const override {
    Contract result;
    result.trigger(OnReadable{readable});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  ReadableSource readable;
};

// Multi-mode operator that exercises valid and invalid additional-input shapes for predicates:
// duplicate extra input (kDuplicate), DrainAll batch selector (kBatch), and missing selector
// (kMissingSelector). Tests RejectsInvalidDriverAndAdditionalInputShapes.
class PredicateAdditionalInputOperator final : public Operator<> {
 public:
  enum class Mode : std::uint8_t {
    kValid,
    kDuplicate,
    kBatch,
    kMissingSelector,
  };

  explicit PredicateAdditionalInputOperator(Mode mode) : mode_(mode) {}

  void setup(OperatorSpec& spec) override {
    spec.input(driver, "driver").queue_depth(4);
    spec.input(additional, "additional").queue_depth(4);
  }

  Contract contract() const override {
    Contract result;
    if (mode_ == Mode::kDuplicate) {
      auto evaluator =
          predicate("invalid-additional",
                    predicate_inputs(additional, additional),
                    [](const PredicateEvaluationContext&,
                       const PredicateInputView<std::uint32_t>&,
                       const PredicateInputView<std::uint32_t>&,
                       const PredicateInputView<std::uint32_t>&) noexcept { return true; });
      result.trigger(
          OnPredicate{.evaluate_on = OnInputUpdate{driver}, .predicate = std::move(evaluator)});
    } else {
      auto evaluator =
          predicate("additional",
                    predicate_inputs(additional),
                    [](const PredicateEvaluationContext&,
                       const PredicateInputView<std::uint32_t>&,
                       const PredicateInputView<std::uint32_t>&) noexcept { return true; });
      result.trigger(
          OnPredicate{.evaluate_on = OnInputUpdate{driver}, .predicate = std::move(evaluator)});
    }
    result.input(driver, Exact{});
    if (mode_ == Mode::kBatch) {
      result.input(additional, DrainAll{});
    } else if (mode_ != Mode::kMissingSelector) {
      result.input(additional, LatestWithin{.max_age = 1ms}).on_missing(additional, Skip{});
    }
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> driver;
  Input<std::uint32_t> additional;

 private:
  Mode mode_;
};

// Tests three driver-validation failure modes: undeclared port reference (kUnknown), a
// non-Exact selector on the driver input (kNonExact), and driver type mismatching the
// evaluator signature (kTypeMismatch). Tests RejectsInvalidDriverAndAdditionalInputShapes.
class InvalidPredicateDriverOperator final : public Operator<> {
 public:
  enum class Mode : std::uint8_t { kUnknown, kNonExact, kTypeMismatch };

  explicit InvalidPredicateDriverOperator(Mode mode) : mode_(mode) {}

  void setup(OperatorSpec& spec) override { spec.input(driver, "driver").queue_depth(2); }

  Contract contract() const override {
    Contract result;
    if (mode_ == Mode::kTypeMismatch) {
      auto evaluator = predicate("type-mismatch",
                                 [](const PredicateEvaluationContext&,
                                    const PredicateInputView<float>&) noexcept { return true; });
      result
          .trigger(
              OnPredicate{.evaluate_on = OnInputUpdate{driver}, .predicate = std::move(evaluator)})
          .input(driver, Exact{});
      return result;
    }

    auto evaluator =
        predicate("invalid-driver",
                  [](const PredicateEvaluationContext&,
                     const PredicateInputView<std::uint32_t>&) noexcept { return true; });
    const Input<std::uint32_t>& selected = mode_ == Mode::kUnknown ? unknown : driver;
    result.trigger(
        OnPredicate{.evaluate_on = OnInputUpdate{selected}, .predicate = std::move(evaluator)});
    if (mode_ == Mode::kNonExact) {
      result.input(driver, LatestWithin{.max_age = 1ms});
    } else {
      result.input(unknown, Exact{});
    }
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> driver;
  Input<std::uint32_t> unknown{"unknown"};

 private:
  Mode mode_;
};

// Two-input operator with OnAligned trigger; requires both inputs to carry capture-time
// evidence stamped by their respective clocks before the aligned trigger is admitted.
// Used in compile_external_aligned and RequiresGenuineCaptureTimeAndEveryDirectedConversion.
class AlignedExternalOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(first, "first").queue_depth(4);
    spec.input(second, "second").queue_depth(4);
  }

  Contract contract() const override {
    Contract result;
    result.trigger(OnAligned{.ports = ports(first, second), .window = 1ms})
        .input(first, Exact{})
        .input(second, Exact{});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> first;
  Input<std::uint32_t> second;
};

// Declares "declared" in setup() but references the undeclared "unknown" port in its contract.
// Tests RejectsAContractPortOutsideTheSetupCatalog.
class UnknownInputOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(declared, "declared"); }

  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms}).input(unknown, Exact{}).on_missing(unknown, Skip{});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> declared;
  Input<std::uint32_t> unknown{"unknown"};
};

void add_default_clock(Graph& graph) {
  const ClockKey clock = graph.add_clock<ManualClock>("clock", std::int64_t{0});
  graph.set_default_clock(clock);
}

void add_default_realtime_clock(Graph& graph) {
  const ClockKey clock = graph.add_clock<RealtimeClock>("clock");
  graph.set_default_clock(clock);
}

// Builds a two-import graph feeding AlignedExternalOperator with configurable capture-time
// stamps, clock conversions, and an optional extreme-overflow affine conversion.
//
//   [holomq://.../first]  --[depth=4]--> target->first
//   [holomq://.../second] --[depth=4]--> target->second
//
// Used in RequiresGenuineCaptureTimeAndEveryDirectedConversion and
// FreezesCheckedOverflowAndFormationFailurePolicy.
[[nodiscard]] ExecutionPlan compile_external_aligned(bool first_capture_time,
                                                     bool second_capture_time,
                                                     bool first_conversion, bool second_conversion,
                                                     bool extreme_conversion = false) {
  Graph graph{"external-aligned"};
  const ClockKey first_clock = graph.add_clock<ManualClock>("first-clock", std::int64_t{0});
  const ClockKey second_clock = graph.add_clock<ManualClock>("second-clock", std::int64_t{0});
  const ClockKey comparison_clock = graph.add_clock<RealtimeClock>("comparison-clock");
  graph.set_default_clock(comparison_clock);

  const AffineClockConversion first_affine =
      extreme_conversion
          ? AffineClockConversion{
                .scale = ClockRateRatio{
                    .numerator = std::numeric_limits<std::int64_t>::max(), .denominator = 1U},
                .offset_ns = std::numeric_limits<std::int64_t>::max(),
            }
          : AffineClockConversion{};
  if (first_conversion) {
    graph.add_clock_conversion(first_clock, comparison_clock, first_affine);
  }
  if (second_conversion) {
    graph.add_clock_conversion(second_clock, comparison_clock, AffineClockConversion{});
  }

  auto topic_options = [](std::string_view name, ClockKey clock, bool capture_time) {
    TopicOptions options;
    options.endpoint_name = name;
    if (capture_time) {
      options.capture_time = RequireCaptureTime{.clock = std::move(clock)};
      options.clock_health = ClockHealthOptions{
          .capability = ClockHealthCapability::kStatusReports,
          .requirement = ClockHealthRequirement::kFreshHealthyReport,
          .max_report_age = 50ms,
      };
    }
    return options;
  };
  auto first = graph.import_topic<std::uint32_t>(
      "holomq://external-aligned/first",
      topic_options("external_aligned_first", first_clock, first_capture_time));
  auto second = graph.import_topic<std::uint32_t>(
      "holomq://external-aligned/second",
      topic_options("external_aligned_second", second_clock, second_capture_time));
  auto target = graph.op<AlignedExternalOperator>("target");
  graph.add_flow(first.output(), target->first, ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(second.output(), target->second, ConnectionOptions{.queue_depth = 4U});
  graph.partition("aligned-target").add(target);
  return compile(graph);
}

// Iterates over all TriggerMode values; each iteration compiles the same data-flow graph and
// checks the mode-specific plan fields. A separate graph verifies that kWindow fails without
// a realtime clock.
//
//   source --[first,  depth=8]--> target(TriggerMode)
//          --[second, depth=8]--> target
//
// REQ: COMPILE-015
// REQ: COMPILE-060
TEST(CompileTemporalTriggers, ReplaysEveryClosedTriggerAndFailsClosedOnMissingWindowEvidence) {
  // Key setup: every TriggerMode enum value compiled in order; kSpin needs a busy-poll placement.
  for (std::uint8_t value = 0U; value <= static_cast<std::uint8_t>(TriggerMode::kPredicate);
       ++value) {
    Graph graph{"trigger-matrix"};
    auto source = graph.op<TemporalSource>("source");
    auto target = graph.op<TriggerMatrixOperator>("target", static_cast<TriggerMode>(value));
    graph.add_flow(source->first, target->first, ConnectionOptions{.queue_depth = 8U});
    graph.add_flow(source->second, target->second, ConnectionOptions{.queue_depth = 8U});
    if (static_cast<TriggerMode>(value) == TriggerMode::kWindow) {
      add_default_realtime_clock(graph);
    } else {
      add_default_clock(graph);
    }
    graph.partition("trigger-source").add(source);
    graph.partition("trigger-target").add(target);

    CompileOptions options;
    if (static_cast<TriggerMode>(value) == TriggerMode::kSpin) {
      options.deployment.bind_thread_placement("trigger-target",
                                               ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
    }
    const ExecutionPlan plan = compile(graph, options);
    const ExecutionPlan repeated = compile(graph, options);
    ASSERT_TRUE(plan.ok()) << static_cast<unsigned>(value) << '\n' << plan.json();
    ASSERT_TRUE(repeated.ok()) << static_cast<unsigned>(value) << '\n' << repeated.json();
    EXPECT_EQ(plan.json(), repeated.json());
    EXPECT_EQ(plan.digest(), repeated.digest());
    ASSERT_EQ(plan.temporal_triggers().size(), 1U);
    const auto& common = plan.temporal_triggers().front();
    // Key assertion: trigger kind matches the iterated TriggerMode enum ordinal.
    EXPECT_EQ(static_cast<std::uint8_t>(common.kind), value);
    EXPECT_EQ(common.origin, TemporalRecordOrigin::kAuthored);
    EXPECT_TRUE(common.owner.valid());
    EXPECT_TRUE(static_cast<bool>(common.readiness_source));
    EXPECT_GE(common.bounded_state_capacity, 1U);
    EXPECT_FALSE(common.witness_identity.empty());
    EXPECT_EQ(common.replay_identity, "holoscan.temporal.replay.v1/decision");
    EXPECT_NE(plan.json().find("\"temporal_triggers\":[{\"bounded_state_capacity\":"),
              std::string_view::npos);
    const TriggerMode mode = static_cast<TriggerMode>(value);
    const bool explicit_readiness = mode == TriggerMode::kReadable ||
                                    mode == TriggerMode::kNotified ||
                                    mode == TriggerMode::kDoorbell || mode == TriggerMode::kSpin;
    EXPECT_EQ(
        std::any_of(plan.readiness().begin(),
                    plan.readiness().end(),
                    [&](const auto& row) { return row.source_id == common.readiness_source; }),
        explicit_readiness);
    if (static_cast<TriggerMode>(value) == TriggerMode::kClock) {
      ASSERT_EQ(plan.clock_triggers().size(), 1U);
      const auto& schedule = plan.clock_triggers().front();
      EXPECT_EQ(schedule.owner, plan.temporal_triggers().front().owner);
      EXPECT_TRUE(schedule.clock.valid());
      EXPECT_EQ(schedule.period_ns, 1'000'000);
      EXPECT_EQ(schedule.phase_origin, ClockPhaseOrigin::kCapturedAtSuccessfulStart);
      EXPECT_FALSE(schedule.explicit_phase.has_value());
      EXPECT_EQ(schedule.phase_conversion_id, 0U);
      EXPECT_EQ(schedule.tick_policy, ClockTickPolicy::kCoalesceLateTicks);
      EXPECT_EQ(schedule.initial_tick_index, 0U);
      EXPECT_FALSE(schedule.initial_handled_index.has_value());
      EXPECT_EQ(schedule.coalesced_range_capacity, 1U);
      EXPECT_TRUE(schedule.global_start_barrier);
      EXPECT_TRUE(schedule.omitted_phase_index_zero_immediate);
      EXPECT_TRUE(schedule.greatest_due_constant_time);
      EXPECT_TRUE(schedule.checked_terminal_no_wrap);
      EXPECT_EQ(schedule.epoch_discontinuity,
                ClockEpochDiscontinuityDisposition::kTerminalRestartRequired);
      EXPECT_NE(plan.json().find("\"clock_triggers\":[{"), std::string_view::npos);
    } else {
      EXPECT_TRUE(plan.clock_triggers().empty());
    }
    if (static_cast<TriggerMode>(value) == TriggerMode::kReadable) {
      ASSERT_EQ(plan.readable_triggers().size(), 1U);
      const auto& readable = plan.readable_triggers().front();
      EXPECT_EQ(readable.owner, plan.temporal_triggers().front().owner);
      EXPECT_TRUE(static_cast<bool>(readable.source));
      EXPECT_EQ(readable.plan_generation, plan.clock_catalog().plan_generation);
      EXPECT_TRUE(readable.witness_clock.valid());
      EXPECT_EQ(readable.initial_source_generation, 1U);
      EXPECT_EQ(readable.initial_source_sequence, 0U);
      EXPECT_EQ(readable.effective_capacity, 1U);
      EXPECT_EQ(readable.max_outstanding_candidates, 1U);
      EXPECT_TRUE(readable.level_readable);
      EXPECT_TRUE(readable.restart_advances_generation);
      EXPECT_TRUE(readable.stale_generation_rejected);
      EXPECT_FALSE(readable.error_creates_candidate);
      EXPECT_FALSE(readable.automatic_retry);
      EXPECT_EQ(readable.fatal_error, ReadableSourceErrorDisposition::kFailRun);
      EXPECT_NE(plan.json().find("\"readable_triggers\":[{"), std::string_view::npos);
    } else {
      EXPECT_TRUE(plan.readable_triggers().empty());
    }
    if (static_cast<TriggerMode>(value) == TriggerMode::kAligned) {
      ASSERT_EQ(plan.aligned_triggers().size(), 1U);
      const auto& aligned = plan.aligned_triggers().front();
      ASSERT_EQ(aligned.members.size(), 2U);
      EXPECT_EQ(aligned.members[0].member_ordinal, 0U);
      EXPECT_EQ(aligned.members[1].member_ordinal, 1U);
      EXPECT_EQ(aligned.inclusive_window_ns, 1'000'000);
      EXPECT_TRUE(aligned.comparison_clock.valid());
      EXPECT_TRUE(aligned.fifo_heads_only);
      EXPECT_TRUE(aligned.no_scan_ahead);
      EXPECT_TRUE(aligned.checked_direct_conversion);
      EXPECT_TRUE(aligned.checked_spread_subtraction);
      EXPECT_TRUE(aligned.exact_window_matches);
      EXPECT_TRUE(aligned.reference_is_maximum_head);
      EXPECT_TRUE(aligned.accepted_once_consumption);
      EXPECT_TRUE(aligned.retire_all_minimum_heads);
      EXPECT_EQ(aligned.retirement_order, "port_id_then_source_sequence");
      EXPECT_EQ(aligned.closure_proofs.size(), 6U);
      EXPECT_TRUE(aligned.silence_is_not_closure);
      EXPECT_TRUE(aligned.missing_requires_finite_proof);
      EXPECT_FALSE(aligned.public_watermark_protocol);
      EXPECT_NE(plan.json().find("\"aligned_triggers\":[{"), std::string_view::npos);
    } else {
      EXPECT_TRUE(plan.aligned_triggers().empty());
    }
    if (static_cast<TriggerMode>(value) == TriggerMode::kPredicate) {
      ASSERT_EQ(plan.missing_aggregates().size(), 2U);
      EXPECT_EQ(plan.missing_aggregates()[0].phase, MissingSelectionPhase::kPredicateEvaluation);
      EXPECT_EQ(plan.missing_aggregates()[1].phase, MissingSelectionPhase::kPredicateCompute);
      EXPECT_NE(plan.json().find("\"missing_aggregates\":[{"), std::string_view::npos);
    } else {
      ASSERT_EQ(plan.missing_aggregates().size(), 1U);
      EXPECT_EQ(plan.missing_aggregates().front().phase, MissingSelectionPhase::kActivation);
    }
    EXPECT_NE(plan.json().find("\"temporal_triggers\""), std::string_view::npos);
    EXPECT_NE(plan.json().find("\"witness_identity\""), std::string_view::npos);
  }

  Graph graph{"window-missing-capability"};
  auto source = graph.op<TemporalSource>("source");
  auto target = graph.op<TriggerMatrixOperator>("target", TriggerMode::kWindow);
  graph.add_flow(source->first, target->first, ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(source->second, target->second, ConnectionOptions{.queue_depth = 8U});
  add_default_clock(graph);

  const ExecutionPlan missing_window_evidence = compile(graph);
  EXPECT_FALSE(missing_window_evidence.ok());
  EXPECT_TRUE(test::has_diagnostic(missing_window_evidence,
                                   "TEMPORAL_WINDOW_WAIT_CLOCK_CAPABILITY_UNAVAILABLE"))
      << missing_window_evidence.json();
}

// Compiles a kSpin trigger graph under three placement configurations and checks admission.
//
//   source --[first,  depth=8]--> target(kSpin)
//          --[second, depth=8]--> target
//
// REQ: COMPILE-015
// REQ: COMPILE-060
TEST(CompileTemporalTriggers, RejectsReferencedSpinWithoutBusyPollPlacementEvidence) {
  auto compile_spin = [](std::optional<ThreadPlacement> placement) {
    Graph graph{"spin-placement-admission"};
    auto source = graph.op<TemporalSource>("source");
    auto target = graph.op<TriggerMatrixOperator>("target", TriggerMode::kSpin);
    graph.add_flow(source->first, target->first, ConnectionOptions{.queue_depth = 8U});
    graph.add_flow(source->second, target->second, ConnectionOptions{.queue_depth = 8U});
    add_default_clock(graph);
    graph.partition("spin-source").add(source);
    graph.partition("spin-target").add(target);

    CompileOptions options;
    if (placement.has_value()) {
      options.deployment.bind_thread_placement("spin-target", *placement);
    }
    return compile(graph, std::move(options));
  };

  const ExecutionPlan unbound = compile_spin(std::nullopt);
  EXPECT_FALSE(unbound.ok());
  EXPECT_TRUE(test::has_diagnostic(unbound, "SPIN_UNPLACED")) << unbound.json();

  const ExecutionPlan sleeping =
      compile_spin(ThreadPlacement{.wake_mode = WakeMode::kSleepingWaitset});
  EXPECT_FALSE(sleeping.ok());
  EXPECT_TRUE(test::has_diagnostic(sleeping, "SPIN_UNPLACED")) << sleeping.json();

  const ExecutionPlan busy_poll = compile_spin(ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  EXPECT_TRUE(busy_poll.ok()) << busy_poll.json();
  EXPECT_FALSE(test::has_diagnostic(busy_poll, "SPIN_UNPLACED"));
}

// Graph with two named clocks and an affine conversion; ExplicitClockOperator references
// the source clock phase directly. The compiler must freeze the conversion ID.
//
//   a_phase --[affine conversion]--> b_schedule (default)
//   (ExplicitClockOperator uses phase_clock{ClockId{1}}, period=2ms)
//
// REQ: COMPILE-021
TEST(CompileTemporalTriggers, PreservesExplicitClockPhaseAndDirectConversion) {
  Graph graph{"clock-phase"};
  const ClockKey phase_clock = graph.add_clock<ManualClock>("a_phase", std::int64_t{0});
  const ClockKey schedule_clock = graph.add_clock<ManualClock>("b_schedule", std::int64_t{0});
  graph.set_default_clock(schedule_clock);
  graph.add_clock_conversion(phase_clock, schedule_clock, AffineClockConversion{.offset_ns = 11});
  (void)graph.op<ExplicitClockOperator>("target",
                                        TimePoint{.timestamp_ns = 17, .clock_id = ClockId{1U}});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.clock_triggers().size(), 1U);
  const auto& schedule = plan.clock_triggers().front();
  // Key setup: phase_clock (ClockId{1}) is converted to schedule_clock (ClockId{2}).
  EXPECT_EQ(schedule.clock, ClockId{2U});
  EXPECT_EQ(schedule.period_ns, 2'000'000);
  EXPECT_EQ(schedule.phase_origin, ClockPhaseOrigin::kExplicitAbsolute);
  ASSERT_TRUE(schedule.explicit_phase.has_value());
  EXPECT_EQ(schedule.explicit_phase->timestamp_ns, 17);
  EXPECT_EQ(schedule.explicit_phase->clock_id, ClockId{1U});
  EXPECT_EQ(schedule.phase_conversion_id, 1U);
  EXPECT_TRUE(schedule.explicit_phase_preserved_on_restart);
  EXPECT_TRUE(schedule.failed_rollback_restores_watermark);
  // Key assertion: phase_conversion_id is non-zero and explicit phase is preserved on restart.
  EXPECT_NE(plan.json().find("\"phase_conversion_id\":1"), std::string_view::npos);

  Graph missing_conversion{"clock-phase-missing-conversion"};
  (void)missing_conversion.add_clock<ManualClock>("a_phase", std::int64_t{0});
  const ClockKey only_schedule =
      missing_conversion.add_clock<ManualClock>("b_schedule", std::int64_t{0});
  missing_conversion.set_default_clock(only_schedule);
  (void)missing_conversion.op<ExplicitClockOperator>(
      "target", TimePoint{.timestamp_ns = 17, .clock_id = ClockId{1U}});
  const ExecutionPlan rejected = compile(missing_conversion);
  EXPECT_FALSE(rejected.ok());
  EXPECT_TRUE(test::has_diagnostic(rejected, "TEMPORAL_CLOCK_CONVERSION_UNAVAILABLE"))
      << rejected.json();

  Graph undeclared_phase{"clock-phase-undeclared"};
  const ClockKey only_clock = undeclared_phase.add_clock<ManualClock>("schedule", std::int64_t{0});
  undeclared_phase.set_default_clock(only_clock);
  (void)undeclared_phase.op<ExplicitClockOperator>(
      "target", TimePoint{.timestamp_ns = 17, .clock_id = ClockId{2U}});
  const ExecutionPlan undeclared = compile(undeclared_phase);
  ASSERT_FALSE(undeclared.ok()) << undeclared.json();
  const auto phase_diagnostic = std::find_if(
      undeclared.diagnostics().begin(), undeclared.diagnostics().end(), [](const auto& d) {
        return d.token == "TEMPORAL_CLOCK_UNDECLARED";
      });
  ASSERT_NE(phase_diagnostic, undeclared.diagnostics().end()) << undeclared.json();
  EXPECT_TRUE(phase_diagnostic->source.vertex) << undeclared.json();
  EXPECT_FALSE(phase_diagnostic->json_pointer.empty()) << undeclared.json();
  EXPECT_FALSE(phase_diagnostic->required.empty()) << undeclared.json();
  EXPECT_NE(phase_diagnostic->provided.find("ClockId 2"), std::string::npos) << undeclared.json();
}

// REQ: COMPILE-053
// Two instances of one reusable body bind their default clock import to one
// concrete root domain without creating duplicate runtime clocks.
TEST(CompileTemporalTriggers, ResolvesExplicitReusableGraphClockImports) {
  Graph graph{"clock-imports"};
  auto source = graph.op<TemporalSource>("source");
  auto left = graph.embed<ImportedClockRelayGraph>("left");
  auto right = graph.embed<ImportedClockRelayGraph>("right");
  graph.add_flow(source->first, left->input, ConnectionOptions{.queue_depth = 2U});
  graph.add_flow(source->second, right->input, ConnectionOptions{.queue_depth = 2U});
  const ClockKey root_clock = graph.add_clock<ManualClock>("root_timeline", std::int64_t{0});
  graph.set_default_clock(root_clock);
  graph.bind_clock(left, left->clock_import(), root_clock);
  graph.bind_clock(right, right->clock_import(), root_clock);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.clock_catalog().clocks.size(), 3U);
  ASSERT_EQ(plan.temporal_triggers().size(), 2U);
  EXPECT_EQ(plan.temporal_triggers()[0].reference_clock, ClockId{1U});
  EXPECT_EQ(plan.temporal_triggers()[1].reference_clock, ClockId{1U});
  ASSERT_EQ(plan.clock_conversions().size(), 2U);
  EXPECT_EQ(plan.clock_conversions()[0].from, ClockId{1U});
  EXPECT_EQ(plan.clock_conversions()[1].from, ClockId{1U});
  EXPECT_NE(plan.clock_conversions()[0].to, plan.clock_conversions()[0].from);
  EXPECT_NE(plan.clock_conversions()[1].to, plan.clock_conversions()[1].from);
  EXPECT_NE(plan.clock_conversions()[0].to, plan.clock_conversions()[1].to);
  EXPECT_NE(plan.digest(), PlanDigest{});
  EXPECT_NE(plan.json().find("\"clock_import_resolutions\""), std::string::npos);

  // Global scheduling admission requires one domain, while COMPILE-053 also permits distinct
  // parent bindings. Exercise that cold lowerer contract without asking an executable plan to
  // admit two scheduling domains.
  Graph distinct{"distinct-clock-imports"};
  auto distinct_source = distinct.op<TemporalSource>("source");
  auto distinct_left = distinct.embed<ImportedClockRelayGraph>("left");
  auto distinct_right = distinct.embed<ImportedClockRelayGraph>("right");
  distinct.add_flow(
      distinct_source->first, distinct_left->input, ConnectionOptions{.queue_depth = 2U});
  distinct.add_flow(
      distinct_source->second, distinct_right->input, ConnectionOptions{.queue_depth = 2U});
  const ClockKey left_clock = distinct.add_clock<ManualClock>("a_root_timeline", std::int64_t{0});
  const ClockKey right_clock = distinct.add_clock<ManualClock>("b_root_timeline", std::int64_t{0});
  distinct.set_default_clock(left_clock);
  distinct.bind_clock(distinct_left, distinct_left->clock_import(), left_clock);
  distinct.bind_clock(distinct_right, distinct_right->clock_import(), right_clock);

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(distinct);
  ASSERT_TRUE(snapshot);
  const std::vector<std::optional<Profile>> profiles(snapshot->vertices.size(), Profile::kStandard);
  const auto lowered =
      detail::lower_clocks_and_readiness(*snapshot, PlanGenerationId{1U}, profiles);
  ASSERT_TRUE(lowered);
  ASSERT_EQ(lowered->clock_import_resolutions.size(), 2U);
  EXPECT_NE(lowered->clock_import_resolutions[0].resolved_clock,
            lowered->clock_import_resolutions[1].resolved_clock);
  ASSERT_EQ(lowered->clock_conversions.size(), 2U);
  EXPECT_NE(lowered->clock_conversions[0].from, lowered->clock_conversions[1].from);

  const ExecutionPlan distinct_plan = compile(distinct);
  EXPECT_FALSE(distinct_plan.ok());
  EXPECT_TRUE(test::has_diagnostic(distinct_plan, "TEMPORAL_SCHEDULING_DOMAIN_UNSUPPORTED"))
      << distinct_plan.json();
}

// REQ: COMPILE-053
TEST(CompileTemporalTriggers, RejectsUnboundReusableGraphClockImport) {
  Graph graph{"unbound-clock-import"};
  auto source = graph.op<TemporalSource>("source");
  auto child = graph.embed<ImportedClockRelayGraph>("child");
  graph.add_flow(source->first, child->input, ConnectionOptions{.queue_depth = 2U});
  const ClockKey root_clock = graph.add_clock<ManualClock>("root_timeline", std::int64_t{0});
  graph.set_default_clock(root_clock);

  const ExecutionPlan plan = compile(graph);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "CLOCK_IMPORT_UNBOUND")) << plan.json();
}

// Arrival-backed triggers use the Graph default clock as their deterministic
// arrival/readiness coordinate. Reject every such trigger family at compile
// time when that coordinate is absent instead of deferring a generic replay
// mismatch to Runtime.
//
//   source->first --[depth=2]--> target->first
//
// REQ: COMPILE-054
TEST(CompileTemporalTriggers, RejectsEveryArrivalBackedTriggerWithoutAResolvedClock) {
  constexpr std::array modes{
      TriggerMode::kAny,
      TriggerMode::kEach,
      TriggerMode::kPredicate,
  };

  for (const TriggerMode mode : modes) {
    Graph graph{"arrival-trigger-without-clock"};
    auto source = graph.op<TemporalSource>("source");
    auto target = graph.op<ArrivalTriggerWithoutClockOperator>("target", mode);
    graph.add_flow(source->first, target->input, ConnectionOptions{.queue_depth = 2U});

    const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
    ASSERT_TRUE(snapshot);
    ASSERT_EQ(snapshot->graphs.size(), 1U);
    const ExecutionPlan plan = compile(graph);
    ASSERT_FALSE(plan.ok()) << plan.json();
    ASSERT_EQ(plan.diagnostics().size(), 1U) << plan.json();
    const auto& diagnostic = plan.diagnostics().front();
    EXPECT_EQ(diagnostic.token, "TEMPORAL_CLOCK_UNDECLARED") << plan.json();
    EXPECT_EQ(diagnostic.source.graph, snapshot->graphs.front().id) << plan.json();
    EXPECT_FALSE(diagnostic.source.vertex) << plan.json();
    EXPECT_EQ(diagnostic.json_pointer, "/graphs/0/default_clock") << plan.json();
    EXPECT_FALSE(diagnostic.required.empty()) << plan.json();
    EXPECT_NE(diagnostic.provided.find("set_default_clock"), std::string::npos) << plan.json();
  }
}

// A physical readiness source supplies the trigger reference clock even when
// the owning Graph has no default. Input-selector conversion admission must use
// that same source clock rather than comparing against an invalid default.
//
//   [external capture clock] --[depth=2]--> target(readiness source clock)
//
// REQ: COMPILE-054
TEST(CompileTemporalTriggers, UsesReadinessSourceClockForInputSelectionAdmission) {
  constexpr std::array modes{
      TriggerMode::kReadable,
      TriggerMode::kNotified,
      TriggerMode::kDoorbell,
      TriggerMode::kSpin,
  };

  const auto compile_with = [](TriggerMode mode, bool add_conversion) {
    Graph graph{"readiness-source-selection-clock"};
    const ClockKey readiness_clock = graph.add_clock<RealtimeClock>("readiness");
    const ClockKey capture_clock = graph.add_clock<RealtimeClock>("capture");
    if (add_conversion) {
      graph.add_clock_conversion(
          capture_clock, readiness_clock, AffineClockConversion{.offset_ns = 17});
    }

    TopicOptions topic_options;
    topic_options.endpoint_name = "readiness_source_selection_clock_input";
    topic_options.capture_time = RequireCaptureTime{.clock = capture_clock};
    topic_options.clock_health = ClockHealthOptions{
        .capability = ClockHealthCapability::kStatusReports,
        .requirement = ClockHealthRequirement::kFreshHealthyReport,
        .max_report_age = 50ms,
    };
    const auto imported = graph.import_topic<std::uint32_t>(
        "holomq://readiness-source-selection/input", std::move(topic_options));
    const auto target = graph.op<ReadinessClockSelectionOperator>("target", mode);
    graph.add_flow(imported.output(), target->input, ConnectionOptions{.queue_depth = 2U});
    graph.partition("readiness-source").add(target);

    CompileOptions options;
    if (mode == TriggerMode::kSpin) {
      options.deployment.bind_thread_placement("readiness-source",
                                               ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
    }
    return compile(graph, std::move(options));
  };

  for (const TriggerMode mode : modes) {
    const ExecutionPlan converted = compile_with(mode, true);
    ASSERT_TRUE(converted.ok()) << static_cast<unsigned>(mode) << '\n' << converted.json();
    ASSERT_EQ(converted.input_selections().size(), 1U);
    ASSERT_EQ(converted.clock_conversions().size(), 1U);
    EXPECT_EQ(converted.input_selections().front().reference_clock,
              converted.clock_conversions().front().to);
    EXPECT_EQ(converted.input_selections().front().conversion_id,
              converted.clock_conversions().front().id);
    const auto source = std::find_if(converted.readiness().begin(),
                                     converted.readiness().end(),
                                     [](const auto& record) { return record.referenced; });
    ASSERT_NE(source, converted.readiness().end());
    EXPECT_EQ(converted.input_selections().front().reference_clock, source->resolved_clock);

    const ExecutionPlan missing_conversion = compile_with(mode, false);
    EXPECT_FALSE(missing_conversion.ok()) << static_cast<unsigned>(mode);
    EXPECT_TRUE(test::has_diagnostic(missing_conversion, "TEMPORAL_CLOCK_CONVERSION_UNAVAILABLE"))
        << missing_conversion.json();
  }

  Graph missing_capture_clock{"readiness-source-missing-capture-clock"};
  const ClockKey readiness_clock = missing_capture_clock.add_clock<RealtimeClock>("readiness");
  TopicOptions topic_options;
  topic_options.endpoint_name = "readiness_source_missing_capture_clock_input";
  topic_options.capture_time =
      RequireCaptureTime{.clock = ClockKey::named("missing_capture_clock")};
  topic_options.clock_health = ClockHealthOptions{
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 50ms,
  };
  const auto imported = missing_capture_clock.import_topic<std::uint32_t>(
      "holomq://readiness-source-selection/missing-clock", std::move(topic_options));
  const auto target =
      missing_capture_clock.op<ReadinessClockSelectionOperator>("target", TriggerMode::kReadable);
  missing_capture_clock.add_flow(
      imported.output(), target->input, ConnectionOptions{.queue_depth = 2U});
  static_cast<void>(readiness_clock);

  const ExecutionPlan missing_capture = compile(missing_capture_clock);
  ASSERT_FALSE(missing_capture.ok()) << missing_capture.json();
  const auto capture_diagnostic =
      std::find_if(missing_capture.diagnostics().begin(),
                   missing_capture.diagnostics().end(),
                   [](const auto& d) { return d.token == "TEMPORAL_CLOCK_UNDECLARED"; });
  ASSERT_NE(capture_diagnostic, missing_capture.diagnostics().end()) << missing_capture.json();
  EXPECT_TRUE(capture_diagnostic->source.vertex) << missing_capture.json();
  EXPECT_TRUE(capture_diagnostic->source.port) << missing_capture.json();
  EXPECT_FALSE(capture_diagnostic->json_pointer.empty()) << missing_capture.json();
  EXPECT_FALSE(capture_diagnostic->required.empty()) << missing_capture.json();
  EXPECT_NE(capture_diagnostic->provided.find("declare or bind"), std::string::npos)
      << missing_capture.json();
}

// REQ: COMPILE-053
// Source-tree mutations model corrupted compiler handoff records that public
// Graph authoring prevents from entering a canonical snapshot.
TEST(CompileTemporalTriggers, RejectsMalformedDuplicateNonImmediateAndCyclicClockImports) {
  Graph graph{"invalid-clock-import-inventory"};
  auto child = graph.embed<ImportedClockRelayGraph>("child");
  const ClockKey root_clock = graph.add_clock<ManualClock>("root_timeline", std::int64_t{0});
  graph.set_default_clock(root_clock);
  graph.bind_clock(child, child->clock_import(), root_clock);
  const auto canonical = detail::GraphCompilerAccess::canonical_snapshot(graph);
  ASSERT_TRUE(canonical);
  ASSERT_EQ(canonical->graphs.size(), 2U);
  ASSERT_EQ(canonical->clock_imports.size(), 1U);
  ASSERT_EQ(canonical->clock_bindings.size(), 1U);
  const std::vector<std::optional<Profile>> profiles(canonical->vertices.size(),
                                                     Profile::kStandard);

  const auto expect_rejection = [&](const detail::GraphSnapshotData& snapshot,
                                    std::string_view token) {
    const auto lowered =
        detail::lower_clocks_and_readiness(snapshot, PlanGenerationId{1U}, profiles);
    ASSERT_FALSE(lowered);
    EXPECT_EQ(lowered.error().message, token);
  };

  {
    auto malformed = *canonical;
    malformed.clock_imports.front().key = {};
    expect_rejection(malformed, "CLOCK_IMPORT_DECLARATION_INVALID");
  }
  {
    auto duplicate = *canonical;
    duplicate.clock_imports.push_back(duplicate.clock_imports.front());
    expect_rejection(duplicate, "CLOCK_IMPORT_DUPLICATE");
  }
  {
    auto collision = *canonical;
    collision.clocks.push_back(ClockDeclarationRecord{
        .graph = collision.clock_imports.front().graph,
        .key = collision.clock_imports.front().key,
        .is_default = true,
    });
    expect_rejection(collision, "CLOCK_IMPORT_DUPLICATE");
  }
  {
    auto non_immediate = *canonical;
    non_immediate.clock_bindings.front().parent_graph =
        non_immediate.clock_bindings.front().child_graph;
    expect_rejection(non_immediate, "CLOCK_IMPORT_BINDING_INVALID");
  }
  {
    auto cyclic = *canonical;
    const GraphId root_graph = cyclic.graphs.front().id;
    const GraphId child_graph = cyclic.graphs.back().id;
    const ClockKey root_import = ClockKey::named("root_loop");
    cyclic.graphs.front().parent = child_graph;
    cyclic.clock_imports.push_back(ClockImportAuthoringRecord{
        .graph = root_graph,
        .key = root_import,
    });
    cyclic.clock_bindings.front().parent_clock = root_import;
    cyclic.clock_bindings.push_back(EmbeddedClockBindingAuthoringRecord{
        .child_graph = root_graph,
        .child_clock = root_import,
        .parent_graph = child_graph,
        .parent_clock = cyclic.clock_imports.front().key,
    });
    expect_rejection(cyclic, "CLOCK_IMPORT_BINDING_INVALID");
  }
  {
    auto duplicate_defaults = *canonical;
    duplicate_defaults.clock_imports.front().is_default = true;
    ClockImportAuthoringRecord second = duplicate_defaults.clock_imports.front();
    second.key = ClockKey::named("second_import");
    duplicate_defaults.clock_imports.push_back(second);
    expect_rejection(duplicate_defaults, "CLOCK_DEFAULT_DUPLICATE");
  }
  {
    auto concrete_import_defaults = *canonical;
    concrete_import_defaults.clock_imports.front().is_default = true;
    concrete_import_defaults.clocks.push_back(ClockDeclarationRecord{
        .graph = concrete_import_defaults.clock_imports.front().graph,
        .key = ClockKey::named("child_concrete"),
        .is_default = true,
    });
    expect_rejection(concrete_import_defaults, "CLOCK_DEFAULT_DUPLICATE");
  }
}

// Two NamedClockOperators each referencing a distinct ManualClock constitute two scheduling
// domains; the compiler must reject regardless of operator registration order.
//
//   a (OnClock{first_clock}) | b (OnClock{second_clock}) — no data edges
//
// REQ: COMPILE-046
TEST(CompileTemporalTriggers, RejectsMultipleSchedulingDomainsBeforeRuntimeConstruction) {
  const auto compile_with = [](bool reverse, bool distinct) {
    Graph graph{"scheduling-domain"};
    const ClockKey first = graph.add_clock<ManualClock>("first", std::int64_t{0});
    const ClockKey second = graph.add_clock<ManualClock>("second", std::int64_t{0});
    graph.set_default_clock(first);
    if (reverse) {
      auto b = graph.op<NamedClockOperator>("b", distinct ? second : first);
      auto a = graph.op<NamedClockOperator>("a", first);
      graph.partition("a").add(a);
      graph.partition("b").add(b);
    } else {
      auto a = graph.op<NamedClockOperator>("a", first);
      auto b = graph.op<NamedClockOperator>("b", distinct ? second : first);
      graph.partition("a").add(a);
      graph.partition("b").add(b);
    }
    return compile(graph);
  };

  const ExecutionPlan first = compile_with(false, true);
  const ExecutionPlan reversed = compile_with(true, true);
  EXPECT_FALSE(first.ok());
  EXPECT_FALSE(reversed.ok());
  EXPECT_TRUE(test::has_diagnostic(first, "TEMPORAL_SCHEDULING_DOMAIN_UNSUPPORTED"))
      << first.json();
  EXPECT_TRUE(test::has_diagnostic(reversed, "TEMPORAL_SCHEDULING_DOMAIN_UNSUPPORTED"))
      << reversed.json();
  // Key assertion: diagnostic output is identical regardless of operator registration order.
  EXPECT_EQ(first.json(), reversed.json());
  EXPECT_EQ(first.digest(), reversed.digest());

  const ExecutionPlan shared = compile_with(false, false);
  EXPECT_TRUE(shared.ok()) << shared.json();
}

// All four combinations of (capture_time × clock_conversion) are probed; only the fully
// equipped case is admitted. See compile_external_aligned for the graph topology.
//
// REQ: COMPILE-020
TEST(CompileTemporalAligned, RequiresGenuineCaptureTimeAndEveryDirectedConversion) {
  // Key setup: both inputs carry capture time and both have directed clock conversions.
  const ExecutionPlan admitted = compile_external_aligned(true, true, true, true);
  ASSERT_TRUE(admitted.ok()) << admitted.json();
  ASSERT_EQ(admitted.aligned_triggers().size(), 1U);
  const auto& aligned = admitted.aligned_triggers().front();
  ASSERT_EQ(aligned.members.size(), 2U);
  // Key assertion: each member has a nonzero conversion_id and capture_time_required set.
  EXPECT_NE(aligned.members[0].conversion_id, 0U);
  EXPECT_NE(aligned.members[1].conversion_id, 0U);
  EXPECT_TRUE(aligned.members[0].capture_time_required);
  EXPECT_TRUE(aligned.members[1].capture_time_required);

  const ExecutionPlan missing_conversion = compile_external_aligned(true, true, true, false);
  EXPECT_FALSE(missing_conversion.ok());
  EXPECT_TRUE(test::has_diagnostic(missing_conversion, "TEMPORAL_CLOCK_CONVERSION_UNAVAILABLE"))
      << missing_conversion.json();

  const ExecutionPlan absent_capture_time = compile_external_aligned(true, false, true, false);
  EXPECT_FALSE(absent_capture_time.ok());
  EXPECT_TRUE(test::has_diagnostic(absent_capture_time, "TEMPORAL_CAPTURE_TIME_REQUIRED"))
      << absent_capture_time.json();
}

// Passes extreme_conversion=true to compile_external_aligned; the affine parameters saturate
// int64 max so the compiler must record overflow=kError and rounding=kNearestTiesToEven.
//
// REQ: COMPILE-020
TEST(CompileTemporalAligned, FreezesCheckedOverflowAndFormationFailurePolicy) {
  const ExecutionPlan plan = compile_external_aligned(true, true, true, true, true);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.aligned_triggers().size(), 1U);
  const auto& aligned = plan.aligned_triggers().front();
  EXPECT_TRUE(aligned.checked_direct_conversion);
  EXPECT_TRUE(aligned.checked_spread_subtraction);
  EXPECT_EQ(aligned.formation_failure_identity, "holoscan.temporal.alignment-formation-failure.v1");
  ASSERT_EQ(aligned.members.size(), 2U);
  ASSERT_NE(aligned.members[0].conversion_id, 0U);
  const auto conversion = std::find_if(
      plan.clock_conversions().begin(), plan.clock_conversions().end(), [&](const auto& record) {
        return record.id == aligned.members[0].conversion_id;
      });
  ASSERT_NE(conversion, plan.clock_conversions().end());
  // Key assertion: extreme affine parameters freeze overflow=kError and ties-to-even rounding.
  EXPECT_EQ(conversion->scale.numerator, std::numeric_limits<std::int64_t>::max());
  EXPECT_EQ(conversion->scale.denominator, 1U);
  EXPECT_EQ(conversion->offset_ns, std::numeric_limits<std::int64_t>::max());
  EXPECT_EQ(conversion->rounding, ClockRoundingMode::kNearestTiesToEven);
  EXPECT_EQ(conversion->overflow, ClockOverflowPolicy::kError);
  EXPECT_NE(plan.json().find("\"overflow\":\"error\""), std::string_view::npos);
  EXPECT_NE(plan.json().find("\"rounding\":\"ties_to_even\""), std::string_view::npos);
}

// First loop: standalone TriggerMatrixOperator(kClock) with each OnViolation action; checks
// deadline record fields. Second loop: verifies DeadlineSourceCoordinateKind per trigger mode.
//
//   source --[first,  depth=8]--> target(TriggerMode, deadline=2ms)
//          --[second, depth=8]--> target
//
// REQ: COMPILE-021
TEST(CompileTemporalDeadlines, FreezesActionAndSourceCoordinateForEveryTriggerFamily) {
  constexpr std::array<OnViolation, 3U> kActions{
      OnViolation::kSafeDefault, OnViolation::kEmitWitness, OnViolation::kFailRun};
  for (const OnViolation action : kActions) {
    Graph graph{"deadline-action"};
    (void)graph.op<TriggerMatrixOperator>("target", TriggerMode::kClock, action);
    add_default_clock(graph);
    const ExecutionPlan plan = compile(graph);
    ASSERT_TRUE(plan.ok()) << plan.json();
    ASSERT_EQ(plan.deadlines().size(), 1U);
    const auto& deadline = plan.deadlines().front();
    EXPECT_EQ(deadline.budget_ns, 2'000'000);
    EXPECT_EQ(deadline.action, action);
    EXPECT_EQ(deadline.source_coordinate, DeadlineSourceCoordinateKind::kScheduledTick);
    EXPECT_TRUE(deadline.source_clock.valid());
    EXPECT_EQ(deadline.callback_count_if_late, action == OnViolation::kFailRun ? 0U : 1U);
    EXPECT_TRUE(deadline.checked_addition);
    EXPECT_TRUE(deadline.exact_boundary_is_on_time);
    EXPECT_TRUE(deadline.final_invocation_gate);
    EXPECT_TRUE(deadline.consumes_candidate_tag);
    EXPECT_FALSE(deadline.automatic_output);
    EXPECT_FALSE(deadline.callback_state_rollback);
    EXPECT_TRUE(deadline.duration_measurement_separate);
    EXPECT_NE(plan.json().find("\"deadlines\":[{"), std::string_view::npos);
  }

  // Key setup: kExpectedCoordinates maps each TriggerMode to the deadline source coordinate kind.
  constexpr std::array<DeadlineSourceCoordinateKind, 10U> kExpectedCoordinates{
      DeadlineSourceCoordinateKind::kTriggerSample,
      DeadlineSourceCoordinateKind::kTriggerSample,
      DeadlineSourceCoordinateKind::kScheduledTick,
      DeadlineSourceCoordinateKind::kReadinessWitness,
      DeadlineSourceCoordinateKind::kReadinessWitness,
      DeadlineSourceCoordinateKind::kReadinessWitness,
      DeadlineSourceCoordinateKind::kReadinessWitness,
      DeadlineSourceCoordinateKind::kAlignedHighOrClosureProof,
      DeadlineSourceCoordinateKind::kWindowClose,
      DeadlineSourceCoordinateKind::kPredicateDriver,
  };
  for (std::uint8_t value = 0U; value <= static_cast<std::uint8_t>(TriggerMode::kPredicate);
       ++value) {
    const TriggerMode mode = static_cast<TriggerMode>(value);
    Graph graph{"deadline-coordinate"};
    auto source = graph.op<TemporalSource>("source");
    auto target = graph.op<TriggerMatrixOperator>("target", mode, OnViolation::kEmitWitness);
    graph.add_flow(source->first, target->first, ConnectionOptions{.queue_depth = 8U});
    graph.add_flow(source->second, target->second, ConnectionOptions{.queue_depth = 8U});
    if (mode == TriggerMode::kWindow) {
      add_default_realtime_clock(graph);
    } else {
      add_default_clock(graph);
    }
    graph.partition("deadline-source").add(source);
    graph.partition("deadline-target").add(target);
    CompileOptions options;
    if (mode == TriggerMode::kSpin) {
      options.deployment.bind_thread_placement("deadline-target",
                                               ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
    }
    const ExecutionPlan plan = compile(graph, std::move(options));
    ASSERT_TRUE(plan.ok()) << static_cast<unsigned>(value) << '\n' << plan.json();
    ASSERT_EQ(plan.deadlines().size(), 1U);
    EXPECT_EQ(plan.deadlines().front().source_coordinate, kExpectedCoordinates[value]);
  }
}

// An imported topic stamped by a capture clock distinct from the reference clock requires a
// directed affine conversion. OnAny with mixed-clock inputs is ambiguous and must be rejected.
//
//   [holomq://.../input] --[depth=8]--> target->first  (capture_clock != reference_clock)
//   (optional) source->second          --> target->second
//
// REQ: COMPILE-021
TEST(CompileTemporalDeadlines, FreezesCrossClockInputConversionAndRejectsAmbiguousAnyInput) {
  const auto compile_cross_clock = [](TriggerMode mode,
                                      bool add_conversion,
                                      bool connect_reference_clock_input) {
    Graph graph{"cross-clock-deadline"};
    const ClockKey capture_clock = graph.add_clock<ManualClock>("capture", std::int64_t{0});
    const ClockKey reference_clock = graph.add_clock<RealtimeClock>("reference");
    graph.set_default_clock(reference_clock);
    if (add_conversion) {
      graph.add_clock_conversion(
          capture_clock, reference_clock, AffineClockConversion{.offset_ns = 17});
    }

    TopicOptions topic_options;
    topic_options.endpoint_name = "cross_clock_deadline_input";
    topic_options.capture_time = RequireCaptureTime{.clock = capture_clock};
    topic_options.clock_health = ClockHealthOptions{
        .capability = ClockHealthCapability::kStatusReports,
        .requirement = ClockHealthRequirement::kFreshHealthyReport,
        .max_report_age = 50ms,
    };
    const auto imported = graph.import_topic<std::uint32_t>("holomq://cross-clock-deadline/input",
                                                            std::move(topic_options));
    const auto target = graph.op<TriggerMatrixOperator>("target", mode, OnViolation::kEmitWitness);
    graph.add_flow(imported.output(), target->first, ConnectionOptions{.queue_depth = 8U});
    if (connect_reference_clock_input) {
      const auto source = graph.op<TemporalSource>("source");
      graph.add_flow(source->second, target->second, ConnectionOptions{.queue_depth = 8U});
    }
    return compile(graph);
  };

  // Key setup: kEach trigger with a single cross-clock input and a directed conversion.
  const ExecutionPlan admitted = compile_cross_clock(TriggerMode::kEach, true, false);
  ASSERT_TRUE(admitted.ok()) << admitted.json();
  ASSERT_EQ(admitted.deadlines().size(), 1U);
  ASSERT_EQ(admitted.clock_conversions().size(), 1U);
  const auto& conversion = admitted.clock_conversions().front();
  const auto& deadline = admitted.deadlines().front();
  // Key assertion: deadline source_clock is the capture clock; conversion_id links the record.
  EXPECT_EQ(deadline.source_coordinate, DeadlineSourceCoordinateKind::kTriggerSample);
  EXPECT_EQ(deadline.source_clock, conversion.from);
  EXPECT_EQ(deadline.conversion_id, conversion.id);
  const auto trigger =
      std::find_if(admitted.temporal_triggers().begin(),
                   admitted.temporal_triggers().end(),
                   [&](const auto& record) { return record.owner == deadline.owner; });
  ASSERT_NE(trigger, admitted.temporal_triggers().end());
  EXPECT_EQ(trigger->conversion_id, conversion.id);

  const ExecutionPlan missing_conversion = compile_cross_clock(TriggerMode::kEach, false, false);
  EXPECT_FALSE(missing_conversion.ok());
  EXPECT_TRUE(test::has_diagnostic(missing_conversion, "TEMPORAL_CLOCK_CONVERSION_UNAVAILABLE"))
      << missing_conversion.json();

  const ExecutionPlan ambiguous_any = compile_cross_clock(TriggerMode::kAny, true, true);
  EXPECT_FALSE(ambiguous_any.ok());
  EXPECT_TRUE(test::has_diagnostic(ambiguous_any, "TEMPORAL_DEADLINE_SOURCE_AMBIGUOUS"))
      << ambiguous_any.json();
}

// Three inputs with different missing policies split into two aggregate phases; the compiler
// must emit one record per phase with only the phase-appropriate inputs listed.
//
//   source --[first,  depth=4]--> target->driver       (Exact, predicate evaluation phase)
//          --[second, depth=4]--> target->additional   (LatestWithin+SafeDefault, eval phase)
//          --[first,  depth=6]--> target->compute_only (DrainAll, compute phase only)
//
// REQ: COMPILE-021
TEST(CompileTemporalMissing, SeparatesPredicateEvaluationAndComputeOnlyAggregates) {
  Graph graph{"predicate-missing-phases"};
  auto source = graph.op<TemporalSource>("source");
  auto target = graph.op<PredicatePhaseOperator>("target");
  graph.add_flow(source->first, target->driver, ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(source->second, target->additional, ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(source->first, target->compute_only, ConnectionOptions{.queue_depth = 6U});
  add_default_clock(graph);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.missing_aggregates().size(), 2U);
  const auto& evaluation = plan.missing_aggregates()[0];
  const auto& compute = plan.missing_aggregates()[1];
  EXPECT_EQ(evaluation.owner, compute.owner);
  // Key assertion: compute_only (DrainAll) belongs exclusively to the kPredicateCompute phase.
  EXPECT_EQ(evaluation.phase, MissingSelectionPhase::kPredicateEvaluation);
  ASSERT_EQ(evaluation.inputs.size(), 2U);
  EXPECT_EQ(evaluation.inputs[0].ordinal, 0U);
  EXPECT_EQ(evaluation.inputs[0].selector, InputSelectorKind::kExact);
  EXPECT_EQ(evaluation.inputs[0].missing, TemporalRecordMissingKind::kUnspecified);
  EXPECT_EQ(evaluation.inputs[1].ordinal, 1U);
  EXPECT_EQ(evaluation.inputs[1].selector, InputSelectorKind::kLatestWithin);
  EXPECT_EQ(evaluation.inputs[1].missing, TemporalRecordMissingKind::kSafeDefault);

  EXPECT_EQ(compute.phase, MissingSelectionPhase::kPredicateCompute);
  ASSERT_EQ(compute.inputs.size(), 1U);
  EXPECT_EQ(compute.inputs[0].ordinal, 0U);
  EXPECT_EQ(compute.inputs[0].selector, InputSelectorKind::kDrainAll);
  EXPECT_EQ(compute.inputs[0].missing, TemporalRecordMissingKind::kUnspecified);
  for (const auto* aggregate : {&evaluation, &compute}) {
    EXPECT_TRUE(aggregate->evaluate_complete_set);
    EXPECT_TRUE(aggregate->safe_default_dominates);
    EXPECT_TRUE(aggregate->absent_without_safe_default_no_fires);
    EXPECT_EQ(aggregate->degraded_callback_count, 1U);
    EXPECT_EQ(aggregate->absent_callback_count, 0U);
    EXPECT_EQ(aggregate->nominal_callback_count, 1U);
    EXPECT_TRUE(aggregate->degraded_commits_destructive_selection);
    EXPECT_TRUE(aggregate->no_fire_preserves_provisional_selection);
    EXPECT_TRUE(aggregate->bracketed_window_all_or_nothing);
  }
  EXPECT_NE(plan.json().find("\"missing_aggregates\":[{"), std::string_view::npos);
}

// TemporalIdentityLeaf embeds an internal "target" operator; the compiler must surface the
// trigger through the leaf's exported input while preserving the inner operator's identity.
//
//   source --[first, depth=2]--> leaf->input  (leaf wraps TriggerMatrixOperator + ManualClock)
//
// REQ: COMPILE-015
TEST(CompileTemporalTriggers, JoinsEmbeddedBoundaryOwnersThroughOperatorIdentityMap) {
  Graph graph{"temporal-identity-map"};
  auto source = graph.op<TemporalSource>("source");
  auto leaf = graph.embed<TemporalIdentityLeaf>("leaf");
  graph.add_flow(source->first, leaf->input, ConnectionOptions{.queue_depth = 2U});

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 1U);
  EXPECT_FALSE(plan.execution_partitions().front().authored_name.has_value());
  EXPECT_EQ(plan.execution_partitions().front().members.size(), 2U);
  ASSERT_EQ(plan.temporal_triggers().size(), 1U);
  const OperatorRef temporal_owner = plan.temporal_triggers().front().owner;

  const auto parameter =
      std::find_if(plan.operator_parameters().begin(),
                   plan.operator_parameters().end(),
                   [&](const auto& record) { return record.owner == temporal_owner; });
  ASSERT_NE(parameter, plan.operator_parameters().end());
  EXPECT_NE(parameter->operator_path.find("leaf"), std::string::npos);

  const auto lifecycle =
      std::find_if(plan.lifecycle_restart_closures().begin(),
                   plan.lifecycle_restart_closures().end(),
                   [&](const auto& record) { return record.target == temporal_owner; });
  ASSERT_NE(lifecycle, plan.lifecycle_restart_closures().end());

  const auto vertex = std::find_if(
      plan.graph().vertices().begin(), plan.graph().vertices().end(), [](const auto& record) {
        return record.kind == VertexKind::kOperator && record.name == "target";
      });
  ASSERT_NE(vertex, plan.graph().vertices().end());
  // The parameter path and lifecycle join above prove that the exported interface edge resolved
  // to the inner target rather than manufacturing an execution root at the embedded boundary.
}

// A contract-selected clock is an independent activation root even when the Operator also has an
// upstream payload predecessor. Declaration-only readiness sources on the same test Operator stay
// inert and must not affect which two roots are diagnosed.
//
//   source --[first, depth=2]--> clock-target
//
// REQ: COMPILE-058
TEST(CompileTemporalTriggers, RejectsClockDrivenOperatorWithUpstreamPayloadAsSecondRoot) {
  Graph graph{"clock-source-with-upstream"};
  auto source = graph.op<TemporalSource>("source");
  auto target = graph.op<TriggerMatrixOperator>("clock-target", TriggerMode::kClock);
  graph.add_flow(source->first, target->first, ConnectionOptions{.queue_depth = 2U});
  add_default_clock(graph);

  const ExecutionPlan plan = compile(graph);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PARTITION_MANUAL_AUTHORING_REQUIRED")) << plan.json();
  EXPECT_FALSE(plan.partitioning_plan().accepted_algorithm.has_value());
}

// A contract-selected physical source has the same root semantics as a clock source: an upstream
// payload edge does not convert that independently driven Operator into an input-driven successor.
//
//   source --[first, depth=2]--> readable-target
//
// REQ: COMPILE-058
TEST(CompileTemporalTriggers, RejectsReadableOperatorWithUpstreamPayloadAsSecondRoot) {
  Graph graph{"readable-source-with-upstream"};
  auto source = graph.op<TemporalSource>("source");
  auto target = graph.op<TriggerMatrixOperator>("readable-target", TriggerMode::kReadable);
  graph.add_flow(source->first, target->first, ConnectionOptions{.queue_depth = 2U});
  add_default_clock(graph);

  const ExecutionPlan plan = compile(graph);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PARTITION_MANUAL_AUTHORING_REQUIRED")) << plan.json();
  EXPECT_FALSE(plan.partitioning_plan().accepted_algorithm.has_value());
}

// Three inputs with distinct selectors; DrainAll max_selected_samples must equal the queue
// depth (7) while Exact and LatestWithin are always bounded to 1.
//
//   source --[first,  depth=4]--> target->exact   (Exact, Skip on missing)
//          --[second, depth=4]--> target->latest  (LatestWithin=3ms, ReuseLast on missing)
//          --[first,  depth=7]--> target->drain   (DrainAll)
//
// REQ: COMPILE-017
// REQ: COMPILE-062
TEST(CompileTemporalSelections, DerivesScalarAndDrainAllBoundsFromRetention) {
  Graph graph{"selectors"};
  auto source = graph.op<TemporalSource>("source");
  auto target = graph.op<SelectorOperator>("target");
  graph.add_flow(source->first, target->exact, ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(source->second, target->latest, ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(source->first, target->drain, ConnectionOptions{.queue_depth = 7U});
  add_default_clock(graph);
  graph.partition("selector-source").add(source);
  graph.partition("selector-target").add(target);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.input_selections().size(), 3U);
  const auto exact = std::find_if(
      plan.input_selections().begin(), plan.input_selections().end(), [](const auto& selection) {
        return selection.selector == InputSelectorKind::kExact;
      });
  const auto latest = std::find_if(
      plan.input_selections().begin(), plan.input_selections().end(), [](const auto& selection) {
        return selection.selector == InputSelectorKind::kLatestWithin;
      });
  const auto drain = std::find_if(
      plan.input_selections().begin(), plan.input_selections().end(), [](const auto& selection) {
        return selection.selector == InputSelectorKind::kDrainAll;
      });
  ASSERT_NE(exact, plan.input_selections().end());
  ASSERT_NE(latest, plan.input_selections().end());
  ASSERT_NE(drain, plan.input_selections().end());
  // Key assertion: DrainAll max_selected_samples matches queue_depth(7); others are bounded to 1.
  EXPECT_EQ(exact->max_selected_samples, 1U);
  EXPECT_EQ(latest->max_selected_samples, 1U);
  EXPECT_EQ(drain->max_selected_samples, 7U);
  EXPECT_EQ(drain->batch_selection_mode, BatchSelectionMode::kConsumeSelected);
  EXPECT_TRUE(std::is_sorted(plan.input_selections().begin(),
                             plan.input_selections().end(),
                             [](const auto& left, const auto& right) {
                               return left.input.value() < right.input.value();
                             }));
  ASSERT_EQ(plan.missing_aggregates().size(), 1U);
  const auto& aggregate = plan.missing_aggregates().front();
  ASSERT_EQ(aggregate.inputs.size(), plan.input_selections().size());
  for (std::size_t index = 0U; index < aggregate.inputs.size(); ++index) {
    EXPECT_EQ(aggregate.inputs[index].ordinal, index);
    EXPECT_EQ(aggregate.inputs[index].input, plan.input_selections()[index].input);
  }
}

// Parameter-derived durations (max_age=kRuntime, deadline=kRecompile) must be frozen into
// pointer-free ParameterDerivedTermRecord entries; no internal reference tokens in the JSON.
//
//   source --[first, depth=4]--> target->latest  (LatestWithin{param_ref<max_age>})
//
// REQ: COMPILE-017, PARAM-006
TEST(CompileTemporalSelections, ResolvesBoundedDurationParameterDependenciesIntoPointerFreeTerms) {
  Graph graph{"parameter-derived-temporal"};
  auto source = graph.op<TemporalSource>("source");
  auto target = graph.op<ParameterDerivedTemporalOperator>("target");
  graph.add_flow(source->first, target->latest, ConnectionOptions{.queue_depth = 4U});
  add_default_clock(graph);
  graph.partition("parameter-source").add(source);
  graph.partition("parameter-target").add(target);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.parameter_derived_terms().size(), 2U);

  const auto max_age = std::find_if(
      plan.parameter_derived_terms().begin(),
      plan.parameter_derived_terms().end(),
      [](const auto& term) { return term.kind == ParameterDerivedTermKind::kLatestWithinMaxAge; });
  const auto deadline = std::find_if(
      plan.parameter_derived_terms().begin(),
      plan.parameter_derived_terms().end(),
      [](const auto& term) { return term.kind == ParameterDerivedTermKind::kContractDeadline; });
  ASSERT_NE(max_age, plan.parameter_derived_terms().end());
  ASSERT_NE(deadline, plan.parameter_derived_terms().end());
  const auto& schema = param_schema<TemporalDependencyParams>();
  const auto max_age_field =
      std::find_if(schema.fields.begin(), schema.fields.end(), [](const auto& field) {
        return field.canonical_name == "max_age";
      });
  const auto deadline_field =
      std::find_if(schema.fields.begin(), schema.fields.end(), [](const auto& field) {
        return field.canonical_name == "deadline";
      });
  ASSERT_NE(max_age_field, schema.fields.end());
  ASSERT_NE(deadline_field, schema.fields.end());
  EXPECT_NE(max_age->owner.id, 0U);
  EXPECT_EQ(max_age->owner, deadline->owner);
  EXPECT_EQ(max_age->term_id,
            "operator:" + std::to_string(max_age->owner.id) +
                "/input:" + std::to_string(max_age->input.value()) + "/latest_within.max_age");
  EXPECT_EQ(deadline->term_id,
            "operator:" + std::to_string(deadline->owner.id) + "/contract.deadline");
  EXPECT_EQ(max_age->field, max_age_field->id);
  EXPECT_EQ(max_age->field_full_id, max_age_field->full_id);
  EXPECT_EQ(deadline->field, deadline_field->id);
  EXPECT_EQ(deadline->field_full_id, deadline_field->full_id);
  EXPECT_EQ(max_age->field_path, "max_age");
  EXPECT_EQ(max_age->schema_id, "nvidia.holoscan.test.TemporalDependencyParams");
  EXPECT_EQ(max_age->schema_version, 1U);
  EXPECT_EQ(max_age->initial_value_ns, 3'000'000);
  EXPECT_EQ(max_age->minimum_value_ns, 1'000'000);
  EXPECT_EQ(max_age->maximum_value_ns, 10'000'000);
  EXPECT_EQ(max_age->mutability, ParamMutability::kRuntime);
  EXPECT_EQ(max_age->update_disposition, ParameterDerivedTermUpdateDisposition::kFrozenCompileTime);
  EXPECT_TRUE(max_age->input);

  EXPECT_EQ(deadline->field_path, "deadline");
  EXPECT_EQ(deadline->schema_id, "nvidia.holoscan.test.TemporalDependencyParams");
  EXPECT_EQ(deadline->schema_version, 1U);
  EXPECT_EQ(deadline->initial_value_ns, 5'000'000);
  EXPECT_EQ(deadline->minimum_value_ns, 2'000'000);
  EXPECT_EQ(deadline->maximum_value_ns, 20'000'000);
  EXPECT_EQ(deadline->mutability, ParamMutability::kRecompile);
  EXPECT_EQ(deadline->update_disposition,
            ParameterDerivedTermUpdateDisposition::kRecompileRequired);
  EXPECT_FALSE(deadline->input);

  ASSERT_EQ(plan.input_selections().size(), 1U);
  EXPECT_EQ(plan.input_selections().front().max_age_ns, max_age->initial_value_ns);
  ASSERT_EQ(plan.temporal_triggers().size(), 1U);
  const auto target_trigger = std::find_if(
      plan.temporal_triggers().begin(), plan.temporal_triggers().end(), [&](const auto& trigger) {
        return trigger.owner == deadline->owner;
      });
  ASSERT_NE(target_trigger, plan.temporal_triggers().end());
  EXPECT_EQ(target_trigger->deadline_ns, deadline->initial_value_ns);
  ASSERT_EQ(plan.deadlines().size(), 1U);
  const auto& deadline_plan = plan.deadlines().front();
  EXPECT_EQ(deadline_plan.owner, deadline->owner);
  EXPECT_EQ(deadline_plan.budget_ns, deadline->initial_value_ns);
  EXPECT_EQ(deadline_plan.action, OnViolation::kFailRun);
  EXPECT_EQ(deadline_plan.source_coordinate, DeadlineSourceCoordinateKind::kScheduledTick);
  EXPECT_EQ(deadline_plan.callback_count_if_late, 0U);
  EXPECT_TRUE(deadline_plan.checked_addition);
  EXPECT_TRUE(deadline_plan.exact_boundary_is_on_time);
  EXPECT_TRUE(deadline_plan.final_invocation_gate);

  // Key assertion: plan JSON contains no internal pointer tokens from the compile pipeline.
  EXPECT_EQ(plan.json().find("ParamRef"), std::string_view::npos);
  EXPECT_EQ(plan.json().find("local_type_token"), std::string_view::npos);
  EXPECT_EQ(plan.json().find("initial_parameter_value"), std::string_view::npos);
  EXPECT_NE(plan.json().find("\"parameter_derived_terms\":["), std::string_view::npos);
  EXPECT_NE(plan.json().find(max_age->term_id), std::string_view::npos);
  EXPECT_NE(plan.json().find(deadline->term_id), std::string_view::npos);
}

// Same graph as ResolvesBoundedDurationParameterDependencies; enables runtime updates to
// confirm that max_age gets kRuntimeAtomic disposition while deadline stays kRecompileRequired.
//
//   source --[first, depth=4]--> target->latest  (LatestWithin{param_ref<max_age>})
//
// REQ: COMPILE-017, PARAM-006
TEST(CompileTemporalSelections, AdmitsRuntimeMutableDurationDependencyForAtomicUpdates) {
  Graph graph{"runtime-parameter-derived-temporal"};
  auto source = graph.op<TemporalSource>("source");
  auto target = graph.op<ParameterDerivedTemporalOperator>("target");
  graph.add_flow(source->first, target->latest, ConnectionOptions{.queue_depth = 4U});
  add_default_clock(graph);
  graph.partition("parameter-source").add(source);
  graph.partition("parameter-target").add(target);

  CompileOptions options;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  const ExecutionPlan plan = compile(graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.parameter_derived_terms().size(), 2U);

  const auto max_age = std::find_if(
      plan.parameter_derived_terms().begin(),
      plan.parameter_derived_terms().end(),
      [](const auto& term) { return term.kind == ParameterDerivedTermKind::kLatestWithinMaxAge; });
  const auto deadline = std::find_if(
      plan.parameter_derived_terms().begin(),
      plan.parameter_derived_terms().end(),
      [](const auto& term) { return term.kind == ParameterDerivedTermKind::kContractDeadline; });
  ASSERT_NE(max_age, plan.parameter_derived_terms().end());
  ASSERT_NE(deadline, plan.parameter_derived_terms().end());
  // Key assertion: runtime-mutable field → kRuntimeAtomic; recompile field → kRecompileRequired.
  EXPECT_EQ(max_age->mutability, ParamMutability::kRuntime);
  EXPECT_EQ(max_age->update_disposition, ParameterDerivedTermUpdateDisposition::kRuntimeAtomic);
  EXPECT_EQ(deadline->mutability, ParamMutability::kRecompile);
  EXPECT_EQ(deadline->update_disposition,
            ParameterDerivedTermUpdateDisposition::kRecompileRequired);
}

// Probes three rejection causes for the bracketed-window input before admitting the fully
// valid configuration (blocking policy, depth≥6, realtime clock).
//
//   source --[first,  depth=8]         --> target->first  (anchor)
//          --[second, policy, depth=N] --> target->second (bracketed window member)
//
// REQ: COMPILE-017
// REQ: CHAN-115
// REQ: COMPILE-062
TEST(CompileTemporalWindows, RejectsDropOldestAndInsufficientCompleteRetention) {
  auto compile_window = [](QueuePolicy policy, std::uint32_t depth, bool realtime = false) {
    Graph graph{"window-retention"};
    auto source = graph.op<TemporalSource>("source");
    auto target = graph.op<TriggerMatrixOperator>("target", TriggerMode::kWindow);
    graph.add_flow(source->first, target->first, ConnectionOptions{.queue_depth = 8U});
    graph.add_flow(source->second,
                   target->second,
                   ConnectionOptions{.queue_policy = policy, .queue_depth = depth});
    if (realtime) {
      add_default_realtime_clock(graph);
    } else {
      add_default_clock(graph);
    }
    graph.partition("window-source").add(source);
    graph.partition("window-target").add(target);
    return compile(graph);
  };

  const ExecutionPlan drop_oldest = compile_window(QueuePolicy::kDropOldest, 8U);
  EXPECT_FALSE(drop_oldest.ok());
  EXPECT_TRUE(test::has_diagnostic(drop_oldest, "TEMPORAL_WINDOW_QUEUE_POLICY_DROP_OLDEST"))
      << drop_oldest.json();

  const ExecutionPlan insufficient = compile_window(QueuePolicy::kBlocking, 2U);
  EXPECT_FALSE(insufficient.ok());
  EXPECT_TRUE(test::has_diagnostic(insufficient, "TEMPORAL_RETENTION_INSUFFICIENT"))
      << insufficient.json();

  const ExecutionPlan unproven_wait_clock = compile_window(QueuePolicy::kBlocking, 6U);
  EXPECT_FALSE(unproven_wait_clock.ok());
  EXPECT_TRUE(test::has_diagnostic(unproven_wait_clock,
                                   "TEMPORAL_WINDOW_WAIT_CLOCK_CAPABILITY_UNAVAILABLE"))
      << unproven_wait_clock.json();

  // Key setup: kBlocking policy, depth=6 (≥ lookbehind+lookahead+1), and a realtime clock.
  const ExecutionPlan admitted = compile_window(QueuePolicy::kBlocking, 6U, true);
  ASSERT_TRUE(admitted.ok()) << admitted.json();
  ASSERT_EQ(admitted.window_triggers().size(), 1U);
  const auto& window = admitted.window_triggers().front();
  EXPECT_EQ(window.capability_evidence_id, "holoscan.clock.realtime.v1");
  EXPECT_FALSE(window.external_driver_used);
  EXPECT_EQ(window.presence_guaranteed_inputs, std::vector<PortId>{window.anchor});
  EXPECT_EQ(window.primed_witness_identity, "holoscan.temporal.window-primed.v1");
  EXPECT_EQ(window.pending_limit_witness_identity, "holoscan.temporal.window-pending-limit.v1");
  EXPECT_EQ(window.end_of_stream_witness_identity, "holoscan.temporal.window-end-of-stream.v1");
  EXPECT_EQ(window.timestamp_regression_witness_identity,
            "holoscan.temporal.window-timestamp-regression.v1");
  EXPECT_EQ(window.cancelled_witness_identity, "holoscan.temporal.window-cancelled.v1");
  ASSERT_EQ(admitted.bracketed_windows().size(), 1U);
  const auto& bracketed = admitted.bracketed_windows().front();
  EXPECT_EQ(bracketed.payload_identity.kind, PayloadIdentityKind::kScalar);
  EXPECT_EQ(bracketed.producer_max_emits_per_compute, 1U);
  EXPECT_EQ(bracketed.retained_overlap_capacity, 6U);
  EXPECT_EQ(bracketed.producer_epoch_identity_schema, "holoscan.producer-epoch.v1");
  EXPECT_EQ(bracketed.source_sequence_identity_schema, "holoscan.source-sequence.v1");
  EXPECT_NE(admitted.json().find("\"holoscan.temporal.window-pending-limit.v1\""),
            std::string_view::npos);
  EXPECT_NE(admitted.json().find("\"payload_identity\""), std::string_view::npos);
}

// Covers the full config-codec admission matrix: configless (no codec), explicitly empty,
// matching/reordered/changed codec values, and two rejection causes (unqualified + oversized).
//
//   source --[first, depth=2]--> target->driver
//
// REQ: COMPILE-016
TEST(CompilePredicatePlan, AdmitsConfiglessAndRejectsUnidentifiedConfiguredValues) {
  auto compile_configless = [] {
    Graph graph{"predicate-config"};
    add_default_clock(graph);
    auto source = graph.op<TemporalSource>("source");
    auto target = graph.op<ConfiglessPredicateOperator>("target");
    graph.add_flow(source->first, target->driver, ConnectionOptions{.queue_depth = 2U});
    return compile(graph);
  };
  auto compile_qualified = []<typename Config>(Config config) {
    Graph graph{"predicate-config"};
    add_default_clock(graph);
    auto source = graph.op<TemporalSource>("source");
    auto target = graph.op<ConfiguredPredicateOperator<Config>>("target", config);
    graph.add_flow(source->first, target->driver, ConnectionOptions{.queue_depth = 2U});
    return compile(graph);
  };  // NOLINT(readability/braces): cpplint misparses a templated lambda terminator.

  const ExecutionPlan configless = compile_configless();
  ASSERT_TRUE(configless.ok()) << configless.json();
  ASSERT_EQ(configless.predicate_triggers().size(), 1U);
  ASSERT_EQ(configless.predicate_callables().size(), 1U);
  EXPECT_FALSE(configless.predicate_triggers().front().has_config);
  EXPECT_EQ(configless.predicate_triggers().front().config_size_bytes, 0U);
  EXPECT_EQ(configless.predicate_triggers().front().config_alignment_bytes, 0U);
  EXPECT_FALSE(configless.predicate_triggers().front().config_schema.has_value());
  EXPECT_FALSE(configless.predicate_triggers().front().config_digest.has_value());
  EXPECT_FALSE(configless.predicate_triggers().front().config_type_digest.has_value());
  EXPECT_EQ(configless.predicate_callables().front().predicate_id,
            configless.predicate_triggers().front().id);
  EXPECT_EQ(configless.predicate_callables().front().result_kind, PredicateResultKind::kBool);
  EXPECT_EQ(configless.predicate_callables().front().dispatch_model, "direct_function");
  EXPECT_EQ(configless.predicate_triggers().front().graph_generation,
            configless.clock_catalog().plan_generation);
  EXPECT_TRUE(configless.predicate_triggers().front().candidate_state_is_transactional);
  EXPECT_TRUE(configless.predicate_triggers().front().compute_adopts_evaluator_snapshot);
  EXPECT_EQ(configless.predicate_triggers().front().reset_reasons.size(), 5U);
  EXPECT_EQ(configless.predicate_inputs().front().payload_identity.kind,
            PayloadIdentityKind::kScalar);
  EXPECT_TRUE(configless.predicate_inputs().front().capture_time_required);
  EXPECT_NE(configless.json().find("\"predicate_callables\""), std::string_view::npos);
  EXPECT_NE(configless.json().find("\"reset_reasons\""), std::string_view::npos);
  EXPECT_EQ(configless.json().find("NoPredicateConfig"), std::string_view::npos);

  const ExecutionPlan explicitly_empty = compile_qualified(EmptyPredicateConfig{});
  ASSERT_TRUE(explicitly_empty.ok()) << explicitly_empty.json();
  ASSERT_EQ(explicitly_empty.predicate_triggers().size(), 1U);
  const auto& empty_record = explicitly_empty.predicate_triggers().front();
  EXPECT_TRUE(empty_record.has_config);
  EXPECT_EQ(empty_record.config_size_bytes, sizeof(EmptyPredicateConfig));
  EXPECT_EQ(empty_record.config_alignment_bytes, alignof(EmptyPredicateConfig));
  EXPECT_EQ(empty_record.config_schema, "test.predicate.empty/v1");
  ASSERT_TRUE(empty_record.config_digest);
  EXPECT_EQ(empty_record.config_digest->size(), 71U);
  EXPECT_TRUE(empty_record.config_type_digest.has_value());

  const ExecutionPlan configured = compile_qualified(PredicateConfig{.threshold = 5U});
  ASSERT_TRUE(configured.ok()) << configured.json();
  ASSERT_EQ(configured.predicate_triggers().size(), 1U);
  EXPECT_TRUE(configured.predicate_triggers().front().has_config);
  EXPECT_EQ(configured.predicate_triggers().front().config_schema, "test.predicate.threshold/v1");
  ASSERT_TRUE(configured.predicate_triggers().front().config_digest);
  EXPECT_EQ(configured.predicate_triggers().front().config_digest->size(), 71U);

  const ExecutionPlan reordered =
      compile_qualified(ReorderedPredicateConfig{.reserved = 77U, .threshold = 5U});
  ASSERT_TRUE(reordered.ok()) << reordered.json();
  EXPECT_EQ(configured.predicate_triggers().front().config_digest,
            reordered.predicate_triggers().front().config_digest);

  const ExecutionPlan changed = compile_qualified(PredicateConfig{.threshold = 6U});
  ASSERT_TRUE(changed.ok()) << changed.json();
  EXPECT_NE(configured.predicate_triggers().front().config_digest,
            changed.predicate_triggers().front().config_digest);

  // Key assertion: no PredicateConfigCodec specialization → PREDICATE_CONFIG_IDENTITY_UNAVAILABLE.
  const ExecutionPlan unqualified =
      compile_qualified(UnqualifiedPredicateConfig{.marker = 1U, .threshold = 5U});
  EXPECT_FALSE(unqualified.ok());
  EXPECT_TRUE(test::has_diagnostic(unqualified, "PREDICATE_CONFIG_IDENTITY_UNAVAILABLE"))
      << unqualified.json();

  const ExecutionPlan oversized = compile_qualified(OversizedPredicateConfig{.value = 1U});
  EXPECT_FALSE(oversized.ok());
  EXPECT_TRUE(test::has_diagnostic(oversized, "PREDICATE_CONFIG_IDENTITY_UNAVAILABLE"))
      << oversized.json();
}

// Verifies result_kind and type-erasure disposition for expected<bool,Error> and std::function
// predicates. source --> target topology is the same for each predicate variant.
//
//   source --[first, depth=2]--> target->driver
//
// REQ: COMPILE-016
TEST(CompilePredicatePlan, RecordsExactResultAndOwnedTypeErasureAdmissionFacts) {
  auto compile_predicate = []<typename OperatorType>() {
    Graph graph{"predicate-callable"};
    add_default_clock(graph);
    auto source = graph.op<TemporalSource>("source");
    auto target = graph.op<OperatorType>("target");
    graph.add_flow(source->first, target->driver, ConnectionOptions{.queue_depth = 2U});
    return compile(graph);
  };  // NOLINT(readability/braces): cpplint misparses a templated lambda terminator.

  const ExecutionPlan expected_result =
      compile_predicate.template operator()<ExpectedPredicateOperator>();
  ASSERT_TRUE(expected_result.ok()) << expected_result.json();
  ASSERT_EQ(expected_result.predicate_callables().size(), 1U);
  EXPECT_EQ(expected_result.predicate_callables().front().result_kind,
            PredicateResultKind::kExpectedBoolError);

  const ExecutionPlan type_erased =
      compile_predicate.template operator()<TypeErasedPredicateOperator>();
  ASSERT_TRUE(type_erased.ok()) << type_erased.json();
  ASSERT_EQ(type_erased.predicate_callables().size(), 1U);
  const auto& callable = type_erased.predicate_callables().front();
  EXPECT_EQ(callable.kind, PredicateCallableKind::kTypeErasedFunction);
  EXPECT_EQ(callable.disposition, PredicateCallableDisposition::kAdmittedWithWarning);
  EXPECT_TRUE(callable.type_erased);
  EXPECT_EQ(callable.dispatch_model, "type_erased_owned");
  EXPECT_EQ(callable.diagnostic_ids, std::vector<std::string>{"PREDICATE_CALLABLE_TYPE_ERASED"});
  EXPECT_TRUE(test::has_diagnostic(type_erased, "PREDICATE_CALLABLE_TYPE_ERASED"));
}

// Checks that each factory type (static/inline/bound) produces the correct dispatch_model,
// binding_provenance, and self_rebind fields in the compiled callable record.
//
//   source --[first, depth=2]--> target->driver  (per factory variant)
//
// REQ: COMPILE-016
TEST(CompilePredicatePlan, RecordsEveryEa1CallableDispatchForm) {
  struct Expectation {
    ExecutionPlan (*compile_plan)();
    PredicateCallableKind kind;
    std::string_view dispatch;
    std::string_view binding;
    bool self_rebind;
  };

  const auto compile_static = []() {
    Graph graph{"predicate-static"};
    add_default_clock(graph);
    auto source = graph.op<TemporalSource>("source");
    auto target = graph.op<CallableFormPredicateOperator<StaticPredicateFactory>>("target");
    graph.add_flow(source->first, target->driver, ConnectionOptions{.queue_depth = 2U});
    return compile(graph);
  };
  const auto compile_inline = []() {
    Graph graph{"predicate-inline"};
    add_default_clock(graph);
    auto source = graph.op<TemporalSource>("source");
    auto target = graph.op<CallableFormPredicateOperator<InlinePredicateFactory>>("target");
    graph.add_flow(source->first, target->driver, ConnectionOptions{.queue_depth = 2U});
    return compile(graph);
  };
  const auto compile_bound = []() {
    Graph graph{"predicate-bound"};
    add_default_clock(graph);
    auto source = graph.op<TemporalSource>("source");
    auto target = graph.op<CallableFormPredicateOperator<BoundPredicateFactory>>("target");
    graph.add_flow(source->first, target->driver, ConnectionOptions{.queue_depth = 2U});
    return compile(graph);
  };
  const std::array expectations{
      Expectation{compile_static,
                  PredicateCallableKind::kStaticFunction,
                  "direct_function",
                  "static_or_stateless",
                  false},
      Expectation{compile_inline,
                  PredicateCallableKind::kInlineFunctor,
                  "inline_owned",
                  "owned_inline_functor",
                  false},
      Expectation{compile_bound,
                  PredicateCallableKind::kBoundMember,
                  "self_rebound_member",
                  "operator_self_rebind",
                  true},
  };

  for (const auto& expectation : expectations) {
    const ExecutionPlan plan = expectation.compile_plan();
    ASSERT_TRUE(plan.ok()) << plan.json();
    ASSERT_EQ(plan.predicate_callables().size(), 1U);
    const auto& callable = plan.predicate_callables().front();
    EXPECT_EQ(callable.kind, expectation.kind);
    EXPECT_EQ(callable.dispatch_model, expectation.dispatch);
    EXPECT_EQ(callable.binding_provenance, expectation.binding);
    EXPECT_EQ(callable.self_rebind, expectation.self_rebind);
    EXPECT_TRUE(callable.owned_lifetime);
    EXPECT_FALSE(callable.requires_python_gil);
  }
}

// Iterates over all three Python-reserved callable kinds; each must be rejected with
// PREDICATE_CALLABLE_PROFILE_UNSUPPORTED yet still appear as an inspectable callable record.
//
//   source --[first, depth=2]--> target->driver  (per Python callable kind)
//
// REQ: COMPILE-016
TEST(CompilePredicatePlan, KeepsReservedPythonCallableKindsRejectedAndInspectable) {
  constexpr std::array kinds{
      PredicateCallableKind::kPythonFunction,
      PredicateCallableKind::kPythonLambda,
      PredicateCallableKind::kPythonBoundMethod,
  };
  for (const auto kind : kinds) {
    Graph graph{"predicate-python-reserved"};
    add_default_clock(graph);
    auto source = graph.op<TemporalSource>("source");
    auto target = graph.op<ReservedPythonPredicateOperator>("target", kind);
    graph.add_flow(source->first, target->driver, ConnectionOptions{.queue_depth = 2U});

    const ExecutionPlan plan = compile(graph);
    ASSERT_FALSE(plan.ok()) << plan.json();
    ASSERT_EQ(plan.predicate_callables().size(), 1U);
    const auto& callable = plan.predicate_callables().front();
    EXPECT_EQ(callable.kind, kind);
    EXPECT_EQ(callable.disposition, PredicateCallableDisposition::kRejected);
    EXPECT_EQ(callable.dispatch_model, "python_reserved");
    EXPECT_EQ(callable.binding_provenance, "python_adapter");
    EXPECT_EQ(callable.capture_provenance, "python_owned");
    EXPECT_TRUE(callable.requires_python_gil);
    EXPECT_FALSE(callable.owned_lifetime);
    EXPECT_TRUE(test::has_diagnostic(plan, "PREDICATE_CALLABLE_PROFILE_UNSUPPORTED"));
  }
}

// Verifies that OnPredicate{.evaluate_on=OnClock} records predicate.source=kClock and that
// period/phase/firing semantics are consistent across the trigger, clock-trigger, and predicate
// records. No data edges — operator has no inputs.
//
// REQ: COMPILE-016
TEST(CompilePredicatePlan, FreezesClockSourceAndExactSnapshotSemantics) {
  Graph graph{"clock-predicate"};
  (void)graph.op<ClockPredicateOperator>("target");
  add_default_clock(graph);

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.predicate_triggers().size(), 1U);
  const auto& predicate = plan.predicate_triggers().front();
  ASSERT_EQ(plan.temporal_triggers().size(), 1U);
  const auto& trigger = plan.temporal_triggers().front();
  ASSERT_EQ(plan.clock_triggers().size(), 1U);
  const auto& schedule = plan.clock_triggers().front();
  EXPECT_EQ(predicate.source, PredicateEvaluationSourceKind::kClock);
  EXPECT_FALSE(predicate.driver);
  EXPECT_TRUE(predicate.clock.valid());
  EXPECT_EQ(predicate.period_ns, 2'000'000);
  EXPECT_EQ(trigger.period_ns, predicate.period_ns);
  EXPECT_EQ(trigger.phase, predicate.phase);
  EXPECT_EQ(schedule.period_ns, predicate.period_ns);
  EXPECT_EQ(schedule.explicit_phase, predicate.phase);
  EXPECT_EQ(predicate.firing, PredicateFiring::kOnEveryTrueEvaluation);
  EXPECT_TRUE(predicate.candidate_state_is_transactional);
  EXPECT_TRUE(predicate.compute_adopts_evaluator_snapshot);
  EXPECT_TRUE(predicate.missing_or_error_preserves_truth);
  EXPECT_EQ(predicate.replay, PredicateReplayCapability::kDecisionReplay);
  EXPECT_TRUE(predicate.replay_evidence.empty());
  EXPECT_NE(plan.json().find("\"compute_adopts_evaluator_snapshot\":true"), std::string_view::npos);
}

// Three standalone operators produce one physical readiness source and two synthesized
// clock-derived sources; ID values must be globally unique and stable across compilations.
//
//   MixedPhysicalReadinessOperator  (OnReadable  — physical source)
//   DirectClockOperator             (OnClock     — synthesized source)
//   ClockPredicateOperator          (OnPredicate/OnClock — synthesized source)
//
// REQ: COMPILE-007, COMPILE-015, COMPILE-016, COMPILE-017
// REQ: COMPILE-044
TEST(CompileTemporalSources, AllocatesPhysicalAndSynthesizedSourcesInOneStableIdentitySpace) {
  auto make_plan = [] {
    Graph graph{"mixed-readiness-sources"};
    auto physical = graph.op<MixedPhysicalReadinessOperator>("physical");
    auto clock = graph.op<DirectClockOperator>("clock");
    auto predicate = graph.op<ClockPredicateOperator>("predicate-clock");
    add_default_clock(graph);
    graph.partition("physical").add(physical);
    graph.partition("clock").add(clock);
    graph.partition("predicate-clock").add(predicate);
    return compile(graph);
  };

  const ExecutionPlan first = make_plan();
  const ExecutionPlan second = make_plan();
  ASSERT_TRUE(first.ok()) << first.json();
  ASSERT_TRUE(second.ok()) << second.json();
  ASSERT_EQ(first.readiness().size(), 1U);
  ASSERT_EQ(first.temporal_triggers().size(), 3U);
  ASSERT_EQ(first.clock_triggers().size(), 2U);
  ASSERT_EQ(first.predicate_triggers().size(), 1U);

  const auto trigger_source = [](const ExecutionPlan& plan, TemporalRecordTriggerKind kind) {
    const auto found = std::find_if(plan.temporal_triggers().begin(),
                                    plan.temporal_triggers().end(),
                                    [kind](const auto& trigger) { return trigger.kind == kind; });
    return found == plan.temporal_triggers().end() ? SourceId{} : found->readiness_source;
  };
  const SourceId physical = first.readiness().front().source_id;
  const SourceId clock = trigger_source(first, TemporalRecordTriggerKind::kClock);
  const SourceId predicate = trigger_source(first, TemporalRecordTriggerKind::kPredicate);
  ASSERT_TRUE(static_cast<bool>(physical));
  ASSERT_TRUE(static_cast<bool>(clock));
  ASSERT_TRUE(static_cast<bool>(predicate));
  EXPECT_NE(physical, clock);
  EXPECT_NE(physical, predicate);
  EXPECT_NE(clock, predicate);
  // Key assertion: synthesized source IDs are numerically greater than the physical one.
  const std::uint32_t physical_max = detail::source_id_value(physical);
  EXPECT_GT(detail::source_id_value(clock), physical_max);
  EXPECT_GT(detail::source_id_value(predicate), physical_max);

  EXPECT_TRUE(first.readiness().front().referenced);
  EXPECT_TRUE(first.readiness().front().materialized);
  EXPECT_EQ(physical, second.readiness().front().source_id);
  EXPECT_EQ(clock, trigger_source(second, TemporalRecordTriggerKind::kClock));
  EXPECT_EQ(predicate, trigger_source(second, TemporalRecordTriggerKind::kPredicate));
  EXPECT_EQ(first.json(), second.json());
  EXPECT_EQ(first.digest(), second.digest());
}

// Forces the physical readiness source ID to UINT32_MAX to saturate the ID space, then
// verifies the temporal lowering pass emits READINESS_SOURCE_ID_EXHAUSTED.
//
//   MixedPhysicalReadinessOperator (source_id manually set to UINT32_MAX)
//   DirectClockOperator            (needs a synthesized ID — space is exhausted)
//
// REQ: COMPILE-007, COMPILE-015
// REQ: COMPILE-044
TEST(CompileTemporalSources, RejectsSynthesizedSourceAfterThePhysicalIdentitySpaceIsExhausted) {
  Graph graph{"exhausted-readiness-sources"};
  (void)graph.op<MixedPhysicalReadinessOperator>("physical");
  (void)graph.op<DirectClockOperator>("clock");
  add_default_clock(graph);

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  const auto profile = detail::resolve_profile_admission(*snapshot, Profile::kStandard);
  const PlanGenerationId generation{1U};
  const auto clocks =
      detail::lower_clocks_and_readiness(*snapshot, generation, profile.effective_profiles());
  ASSERT_TRUE(clocks);
  ASSERT_EQ(clocks->readiness.size(), 1U);
  auto readiness = clocks->readiness;
  // Key setup: physical source_id is set to UINT32_MAX, leaving no room for synthesized IDs.
  readiness.front().source_id = detail::make_source_id(std::numeric_limits<std::uint32_t>::max());
  const auto factories = detail::GraphCompilerAccess::executable_factories(graph, *snapshot);

  const auto lowered = detail::lower_temporal_plan(factories,
                                                   *snapshot,
                                                   generation,
                                                   clocks->clocks,
                                                   clocks->clock_conversions,
                                                   readiness,
                                                   profile.effective_profiles(),
                                                   ParameterCompileOptions{},
                                                   detail::runtime_package_capability_view());
  EXPECT_FALSE(lowered.accepted());
  EXPECT_TRUE(std::any_of(lowered.errors.begin(), lowered.errors.end(), [](const auto& error) {
    return error.token == "READINESS_SOURCE_ID_EXHAUSTED";
  }));
  ASSERT_EQ(readiness.size(), 1U);
  EXPECT_EQ(readiness.front().source_id,
            detail::make_source_id(std::numeric_limits<std::uint32_t>::max()));
  EXPECT_TRUE(readiness.front().referenced);
  EXPECT_TRUE(readiness.front().materialized);
  ASSERT_EQ(lowered.triggers.size(), 1U);
  EXPECT_EQ(lowered.triggers.front().readiness_source, readiness.front().source_id);
}

// Probes four package-manifest configurations: no offer → kDecisionReplay; matching offer →
// kCrossBuildRecomputation; wrong fingerprint or wrong key/version → kDecisionReplay again.
//
//   source --[first, depth=2]--> target->driver  (KeyedPredicateOperator)
//
// REQ: COMPILE-016, COMPILE-030
TEST(CompilePredicatePlan, RequiresExactPackageFingerprintForCrossBuildRecomputation) {
  Graph graph{"keyed-predicate-package-evidence"};
  add_default_clock(graph);
  auto source = graph.op<TemporalSource>("source");
  auto target = graph.op<KeyedPredicateOperator>("target");
  graph.add_flow(source->first, target->driver, ConnectionOptions{.queue_depth = 2U});

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  const auto profile = detail::resolve_profile_admission(*snapshot, Profile::kStandard);
  const PlanGenerationId generation{1U};
  const auto clocks =
      detail::lower_clocks_and_readiness(*snapshot, generation, profile.effective_profiles());
  ASSERT_TRUE(clocks);
  const auto factories = detail::GraphCompilerAccess::executable_factories(graph, *snapshot);
  const auto lower_with = [&](const detail::RuntimePackageCapabilityView& package) {
    auto readiness = clocks->readiness;
    return detail::lower_temporal_plan(factories,
                                       *snapshot,
                                       generation,
                                       clocks->clocks,
                                       clocks->clock_conversions,
                                       readiness,
                                       profile.effective_profiles(),
                                       ParameterCompileOptions{},
                                       package);
  };

  const auto local = lower_with(detail::runtime_package_capability_view());
  ASSERT_TRUE(local.accepted());
  ASSERT_EQ(local.predicate_triggers.size(), 1U);
  ASSERT_EQ(local.predicate_callables.size(), 1U);
  EXPECT_EQ(local.predicate_triggers.front().replay, PredicateReplayCapability::kDecisionReplay);
  EXPECT_TRUE(local.predicate_triggers.front().replay_evidence.empty());

  constexpr std::string_view kKey = "nvidia.holoscan.test.keyed";
  constexpr SemanticVersion kVersion{1U, 2U, 3U};
  auto qualified_package = detail::runtime_package_capability_manifest();
  qualified_package.offers.push_back(detail::RuntimePackageCapabilityOffer{
      .id = detail::runtime_predicate_capability_id(kKey, kVersion),
      .version = 1U,
      .evidence_id = local.predicate_callables.front().implementation_fingerprint,
  });
  ASSERT_TRUE(detail::seal_runtime_package_capability_manifest(qualified_package));
  const auto qualified_view =
      detail::validate_runtime_package_capability_manifest(qualified_package);
  ASSERT_TRUE(qualified_view);
  const auto qualified = lower_with(qualified_view.value());
  ASSERT_TRUE(qualified.accepted());
  ASSERT_EQ(qualified.predicate_triggers.size(), 1U);
  // Key assertion: replay upgrades to kCrossBuildRecomputation only when fingerprint matches.
  EXPECT_EQ(qualified.predicate_triggers.front().replay,
            PredicateReplayCapability::kCrossBuildRecomputation);
  EXPECT_EQ(qualified.predicate_triggers.front().replay_evidence,
            qualified.predicate_callables.front().implementation_fingerprint);

  auto wrong_evidence = qualified_package;
  const auto predicate_offer = std::find_if(
      wrong_evidence.offers.begin(), wrong_evidence.offers.end(), [](const auto& offer) {
        return offer.id.starts_with("holoscan.runtime.predicate_recompute/");
      });
  ASSERT_NE(predicate_offer, wrong_evidence.offers.end());
  predicate_offer->evidence_id = "sha256:wrong-implementation";
  ASSERT_TRUE(detail::seal_runtime_package_capability_manifest(wrong_evidence));
  const auto wrong_view = detail::validate_runtime_package_capability_manifest(wrong_evidence);
  ASSERT_TRUE(wrong_view);
  const auto rejected = lower_with(wrong_view.value());
  ASSERT_TRUE(rejected.accepted());
  EXPECT_EQ(rejected.predicate_triggers.front().replay, PredicateReplayCapability::kDecisionReplay);
  EXPECT_TRUE(rejected.predicate_triggers.front().replay_evidence.empty());

  for (const auto& [key, version] : std::array{
           std::pair<std::string_view, SemanticVersion>{"nvidia.holoscan.test.other", kVersion},
           std::pair<std::string_view, SemanticVersion>{kKey, SemanticVersion{1U, 2U, 4U}}}) {
    auto wrong_identity = detail::runtime_package_capability_manifest();
    wrong_identity.offers.push_back(detail::RuntimePackageCapabilityOffer{
        .id = detail::runtime_predicate_capability_id(key, version),
        .version = 1U,
        .evidence_id = local.predicate_callables.front().implementation_fingerprint,
    });
    ASSERT_TRUE(detail::seal_runtime_package_capability_manifest(wrong_identity));
    const auto wrong_identity_view =
        detail::validate_runtime_package_capability_manifest(wrong_identity);
    ASSERT_TRUE(wrong_identity_view);
    const auto wrong_identity_result = lower_with(wrong_identity_view.value());
    ASSERT_TRUE(wrong_identity_result.accepted());
    EXPECT_EQ(wrong_identity_result.predicate_triggers.front().replay,
              PredicateReplayCapability::kDecisionReplay);
  }
}

// Iterates the expectations matrix (profile × callable kind) and verifies that admission and
// disposition outcomes match exactly. Non-capturing lambdas, captured closures, and type-erased
// callables all have different thresholds.
//
//   source --[first, depth=2]--> target->driver  (per OperatorType per profile)
//
// REQ: COMPILE-016
TEST(CompilePredicatePlan, AppliesTheClosedCallableProfileMatrix) {
  struct Expectation {
    Profile profile;
    bool configless_admitted;
    bool captured_admitted;
    bool type_erased_admitted;
    PredicateCallableDisposition type_erased_disposition;
  };
  constexpr std::array expectations{
      Expectation{
          Profile::kStandard, true, true, true, PredicateCallableDisposition::kAdmittedWithWarning},
      Expectation{Profile::kRobotics, true, false, false, PredicateCallableDisposition::kRejected},
      Expectation{Profile::kRealTime, false, false, false, PredicateCallableDisposition::kRejected},
      Expectation{
          Profile::kRegulated, false, false, false, PredicateCallableDisposition::kRejected},
      Expectation{
          Profile::kResearch, true, true, true, PredicateCallableDisposition::kAdmittedWithWarning},
  };

  auto compile_predicate = []<typename OperatorType>(Profile profile) {
    Graph graph{"predicate-profile"};
    add_default_clock(graph);
    auto source = graph.op<TemporalSource>("source");
    auto target = graph.op<OperatorType>("target");
    graph.add_flow(source->first, target->driver, ConnectionOptions{.queue_depth = 2U});
    CompileOptions options;
    options.profile = profile;
    return compile(graph, std::move(options));
  };  // NOLINT(readability/braces): cpplint misparses a templated lambda terminator.

  for (const auto& expectation : expectations) {
    const ExecutionPlan configless =
        compile_predicate.template operator()<ConfiglessPredicateOperator>(expectation.profile);
    EXPECT_EQ(configless.ok(), expectation.configless_admitted) << configless.json();
    ASSERT_EQ(configless.predicate_callables().size(), 1U);
    EXPECT_EQ(configless.predicate_callables().front().effective_profile, expectation.profile);
    EXPECT_EQ(configless.predicate_callables().front().kind,
              PredicateCallableKind::kNonCapturingLambda);
    EXPECT_EQ(configless.predicate_callables().front().disposition,
              expectation.configless_admitted ? PredicateCallableDisposition::kAdmitted
                                              : PredicateCallableDisposition::kRejected);

    const ExecutionPlan captured =
        compile_predicate.template operator()<CapturedPredicateOperator>(expectation.profile);
    EXPECT_EQ(captured.ok(), expectation.captured_admitted) << captured.json();
    ASSERT_EQ(captured.predicate_callables().size(), 1U);
    EXPECT_EQ(captured.predicate_callables().front().kind, PredicateCallableKind::kOpaqueClosure);
    EXPECT_EQ(captured.predicate_callables().front().capture_provenance, "owned_opaque");
    EXPECT_EQ(captured.predicate_callables().front().disposition,
              expectation.captured_admitted ? PredicateCallableDisposition::kAdmitted
                                            : PredicateCallableDisposition::kRejected);

    const ExecutionPlan type_erased =
        compile_predicate.template operator()<TypeErasedPredicateOperator>(expectation.profile);
    EXPECT_EQ(type_erased.ok(), expectation.type_erased_admitted) << type_erased.json();
    ASSERT_EQ(type_erased.predicate_callables().size(), 1U);
    const auto& callable = type_erased.predicate_callables().front();
    EXPECT_EQ(callable.kind, PredicateCallableKind::kTypeErasedFunction);
    EXPECT_EQ(callable.effective_profile, expectation.profile);
    EXPECT_EQ(callable.disposition, expectation.type_erased_disposition);
    EXPECT_FALSE(callable.implementation_fingerprint.empty());
    EXPECT_GT(callable.size_bytes, 0U);
    EXPECT_GT(callable.alignment_bytes, 0U);
    EXPECT_TRUE(callable.type_erased);
    EXPECT_TRUE(callable.owned_lifetime);
    EXPECT_TRUE(callable.possible_dynamic_allocation);
    EXPECT_FALSE(callable.requires_python_gil);
  }
}

// Two sub-tests: driver validation (unknown port / non-Exact selector / type mismatch) and
// additional-input validation (duplicate / DrainAll batch / missing selector / unconnected).
//
//   source --[first, depth=2]--> target->driver
//   (optional) source->second  --> target->additional
//
// REQ: COMPILE-016
TEST(CompilePredicatePlan, RejectsInvalidDriverAndAdditionalInputShapes) {
  auto compile_driver = [](InvalidPredicateDriverOperator::Mode mode) {
    Graph graph{"invalid-predicate-driver"};
    add_default_clock(graph);
    auto source = graph.op<TemporalSource>("source");
    auto target = graph.op<InvalidPredicateDriverOperator>("target", mode);
    graph.add_flow(source->first, target->driver, ConnectionOptions{.queue_depth = 2U});
    return compile(graph);
  };
  constexpr std::array driver_failures{
      std::pair{InvalidPredicateDriverOperator::Mode::kUnknown, "TEMPORAL_INPUT_PORT_INVALID"},
      std::pair{InvalidPredicateDriverOperator::Mode::kNonExact, "TEMPORAL_CONTRACT_INVALID"},
      std::pair{InvalidPredicateDriverOperator::Mode::kTypeMismatch, "TEMPORAL_CONTRACT_INVALID"},
  };
  for (const auto& [mode, diagnostic] : driver_failures) {
    const ExecutionPlan plan = compile_driver(mode);
    EXPECT_FALSE(plan.ok()) << plan.json();
    EXPECT_TRUE(test::has_diagnostic(plan, diagnostic)) << plan.json();
  }

  auto compile_additional = [](PredicateAdditionalInputOperator::Mode mode,
                               bool connect_additional) {
    Graph graph{"invalid-predicate-additional"};
    add_default_clock(graph);
    auto source = graph.op<TemporalSource>("source");
    auto target = graph.op<PredicateAdditionalInputOperator>("target", mode);
    graph.add_flow(source->first, target->driver, ConnectionOptions{.queue_depth = 4U});
    if (connect_additional) {
      graph.add_flow(source->second, target->additional, ConnectionOptions{.queue_depth = 4U});
    }
    return compile(graph);
  };
  for (const auto mode : {PredicateAdditionalInputOperator::Mode::kDuplicate,
                          PredicateAdditionalInputOperator::Mode::kBatch,
                          PredicateAdditionalInputOperator::Mode::kMissingSelector}) {
    const ExecutionPlan plan = compile_additional(mode, true);
    EXPECT_FALSE(plan.ok()) << plan.json();
    EXPECT_TRUE(test::has_diagnostic(plan, "TEMPORAL_CONTRACT_INVALID")) << plan.json();
  }
  const ExecutionPlan unconnected =
      compile_additional(PredicateAdditionalInputOperator::Mode::kValid, false);
  EXPECT_FALSE(unconnected.ok());
  EXPECT_TRUE(test::has_diagnostic(unconnected, "TEMPORAL_INPUT_UNCONNECTED"))
      << unconnected.json();
}

// Standalone operator whose contract references an undeclared port; no data edges.
//
// REQ: COMPILE-015
// REQ: COMPILE-060
TEST(CompileTemporalValidation, RejectsAContractPortOutsideTheSetupCatalog) {
  Graph graph{"unknown-contract-port"};
  (void)graph.op<UnknownInputOperator>("invalid");
  add_default_clock(graph);

  const ExecutionPlan plan = compile(graph);
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "TEMPORAL_INPUT_PORT_INVALID"));
}

// Compiles the same three-input SelectorOperator graph twice; JSON and digest must be identical.
//
//   source --[first,  depth=4]--> target->exact
//          --[second, depth=4]--> target->latest
//          --[first,  depth=7]--> target->drain
//
// REQ: COMPILE-015, COMPILE-016, COMPILE-017
// REQ: COMPILE-060
TEST(CompileTemporalDeterminism, EquivalentGraphsProduceIdenticalTemporalProjections) {
  auto make_plan = [] {
    Graph graph{"stable-temporal"};
    auto source = graph.op<TemporalSource>("source");
    auto target = graph.op<SelectorOperator>("target");
    graph.add_flow(source->first, target->exact, ConnectionOptions{.queue_depth = 4U});
    graph.add_flow(source->second, target->latest, ConnectionOptions{.queue_depth = 4U});
    graph.add_flow(source->first, target->drain, ConnectionOptions{.queue_depth = 7U});
    add_default_clock(graph);
    graph.partition("selector-source").add(source);
    graph.partition("selector-target").add(target);
    return compile(graph);
  };

  const ExecutionPlan first = make_plan();
  const ExecutionPlan second = make_plan();
  ASSERT_TRUE(first.ok()) << first.json();
  ASSERT_TRUE(second.ok()) << second.json();
  EXPECT_EQ(first.json(), second.json());
  EXPECT_EQ(first.digest(), second.digest());
}

}  // namespace
}  // namespace holoscan
// NOLINTEND(bugprone-unchecked-optional-access)
