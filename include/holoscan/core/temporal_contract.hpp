/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/activation.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/predicate_input_view.hpp>
#include <holoscan/core/readiness_source.hpp>
#include <holoscan/time/clock_registry.hpp>
#include <holoscan/time/time_point.hpp>

namespace holoscan {

class OperatorBase;

/// @brief Customize canonical semantic encoding for one predicate configuration type.
/// @tparam Config Application-defined immutable predicate configuration type.
///
/// A specialization must provide nonempty `static constexpr std::string_view schema`, a nonzero
/// `static constexpr std::uint32_t version`, a positive finite
/// `static constexpr std::size_t max_encoded_bytes`, and
/// `static expected<std::vector<std::byte>, Error> encode(const Config&)`. The encoding must be
/// canonical for the logical value and independent of C++ padding, member order, addresses,
/// compiler, and process. Compilation rejects configured predicates without a conforming codec or
/// whose encoder fails/exceeds its declared bound. The canonical bytes are hashed and discarded;
/// they are never published in an ExecutionPlan.
template <typename Config>
struct PredicateConfigCodec;

template <typename Object, typename Method>
class BoundPredicate;

namespace detail {
class ContractBuilderAccess;
class ContractTestAccess;
struct PredicateBindingFactoryAccess;
struct PredicateInputFactoryAccess;

struct PredicateReceiver {
  void* address{};
  const void* local_type_token{};
};

struct PredicateConfigIdentity {
  std::string schema;
  std::string digest;
};

template <typename Config>
concept QualifiedPredicateConfig = requires(const Config& value) {
  { PredicateConfigCodec<Config>::schema } -> std::convertible_to<std::string_view>;
  { PredicateConfigCodec<Config>::version } -> std::convertible_to<std::uint32_t>;
  { PredicateConfigCodec<Config>::max_encoded_bytes } -> std::convertible_to<std::size_t>;
  {
    PredicateConfigCodec<Config>::encode(value)
  } -> std::same_as<expected<std::vector<std::byte>, Error>>;
};

[[nodiscard]] inline std::string predicate_sha256(const ParamSchemaHash& digest) {
  constexpr std::array<char, 16> kHex{
      '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
  std::string result{"sha256:"};
  result.reserve(71U);
  for (const std::byte value : digest.value) {
    const auto byte = std::to_integer<std::uint8_t>(value);
    result.push_back(kHex[byte >> 4U]);
    result.push_back(kHex[byte & 0x0FU]);
  }
  return result;
}

template <typename Config>
[[nodiscard]] std::optional<PredicateConfigIdentity> predicate_config_identity(
    const Config& config) noexcept {
  if constexpr (!QualifiedPredicateConfig<Config>) {
    return std::nullopt;
  } else {
    try {
      const std::string_view schema = PredicateConfigCodec<Config>::schema;
      const std::uint32_t version = PredicateConfigCodec<Config>::version;
      const std::size_t maximum = PredicateConfigCodec<Config>::max_encoded_bytes;
      if (schema.empty() || version == 0U || maximum == 0U ||
          maximum > std::numeric_limits<std::uint64_t>::max()) {
        return std::nullopt;
      }
      auto encoded = PredicateConfigCodec<Config>::encode(config);
      if (!encoded || encoded->size() > maximum) {
        return std::nullopt;
      }
      StableHashBuilder hash;
      hash.add("holoscan.temporal.predicate-config.v1");
      hash.add(schema);
      hash.add(version);
      hash.add_canonical_size(encoded->size());
      for (const std::byte value : *encoded) {
        hash.add_byte(std::to_integer<std::uint8_t>(value));
      }
      return PredicateConfigIdentity{
          .schema = std::string(schema) + "/v" + std::to_string(version),
          .digest = predicate_sha256(hash.finish()),
      };
    } catch (...) {
      return std::nullopt;
    }
  }
}

template <typename T>
inline constexpr char kPredicateReceiverLocalTypeToken{};

/// @internal
/// Internal template machinery used to clone cold predicate values without exposing their bytes.
///
/// This type is visible only because public factory templates must instantiate storage for an
/// application-defined callable, state, and config type. It is not an extension point, serialized
/// record, callable invocation interface, or supported application-facing ABI.
class PredicateCallableStorage {
 public:
  virtual ~PredicateCallableStorage() = default;
  [[nodiscard]] virtual std::unique_ptr<PredicateCallableStorage> clone() const = 0;
  /// @brief Evaluate into an owned candidate state without changing committed state.
  /// @return The candidate truth value, or an error after discarding the candidate.
  /// @note After success, the Runtime must call exactly one of `commit_candidate()` or
  /// `discard_candidate()` before the next evaluation.
  [[nodiscard]] virtual expected<bool, Error> evaluate(const PredicateEvaluationContext& context,
                                                       std::span<const void* const> input_views,
                                                       PredicateReceiver self_receiver) = 0;
  /// @brief Publish the prepared candidate state without throwing or allocating.
  virtual void commit_candidate() noexcept = 0;
  /// @brief Discard the prepared candidate state without changing committed state.
  virtual void discard_candidate() noexcept = 0;
  [[nodiscard]] virtual const void* receiver_type_token() const noexcept = 0;
  virtual void reset() = 0;
};

template <typename Signature, typename Evaluator, typename State, typename Config>
class PredicateCallableModel;
}  // namespace detail

namespace detail {

/// Hold either one concrete cold duration or one compile-resolved Parameter dependency.
///
/// This authoring carrier is intentionally operation-bearing and process-local. Compilation
/// resolves it to canonical signed nanoseconds plus pointer-free field identity before publishing
/// an ExecutionPlan.
struct TemporalDurationExpression {
  std::chrono::nanoseconds value{};
  ParamDurationDependency dependency{};

  constexpr TemporalDurationExpression() noexcept = default;

  template <typename Rep, typename Period>
    requires std::is_convertible_v<std::chrono::duration<Rep, Period>, std::chrono::nanoseconds>
  // Implicit conversion preserves concise designated-initializer authoring for temporal fields.
  // NOLINTNEXTLINE(runtime/explicit)
  constexpr TemporalDurationExpression(std::chrono::duration<Rep, Period> duration) noexcept
      : value(duration) {}

  template <auto... MemberPath>
  // Implicit conversion preserves the same authoring syntax for Parameter-backed durations.
  // NOLINTNEXTLINE(runtime/explicit)
  TemporalDurationExpression(ParamRef<MemberPath...> reference) noexcept
      : dependency(make_param_duration_dependency(reference)) {}

  [[nodiscard]] constexpr std::int64_t count() const noexcept { return value.count(); }
  [[nodiscard]] constexpr operator std::chrono::nanoseconds() const noexcept { return value; }
};

}  // namespace detail

/// @brief Select the sample at the exact frozen event-time reference.
struct Exact {};

/// @brief Select the newest causal sample no older than a finite inclusive age.
struct LatestWithin {
  detail::TemporalDurationExpression max_age{};  ///< Positive age or duration dependency.
};

/// @brief Select every causally available FIFO survivor as one batch.
///
/// The selector is intentionally fieldless. Compilation derives its finite maximum from retained
/// queue and cache geometry; adding an authored limit would change its consume-all contract.
struct DrainAll {};

/// @brief Select strict outer brackets and the inclusive middle between consecutive anchors.
struct BracketedWindow {
  std::uint32_t lookbehind{1};  ///< Positive count of nearest strict predecessors.
  std::uint32_t lookahead{1};   ///< Positive count of earliest strict successors.
  std::uint32_t max_samples{};  ///< Required positive all-or-nothing complete-set bound.
};

/// @brief Suppress one candidate when the corresponding selector cannot produce a value.
struct Skip {};

/// @brief Enter the operator-authored fieldless degraded path without fabricating input data.
struct SafeDefault {};

/// @brief Reuse one previously admitted scalar selection within a finite inclusive age.
struct ReuseLast {
  std::chrono::nanoseconds max_age{};  ///< Required positive inclusive retention age.
};

/// @brief Trigger when any connected input accepts a sample.
struct OnAnyInput {};

/// @brief Trigger once for every accepted sample on one local input.
struct OnEach {
  PortRef port;  ///< Borrowed authoring port copied into the owning \ref Contract.
};

/// @brief Coalesce late periodic ticks while preserving one fixed phase.
enum class ClockTickPolicy : std::uint8_t {
  kCoalesceLateTicks = 0,  ///< Freeze only the greatest due unhandled tick in O(1).
};

/// @brief Trigger from one positive-period clock schedule.
struct OnClock {
  std::chrono::nanoseconds period{};    ///< Required positive representable period.
  ClockKey clock{};                     ///< Cold key; omission selects the graph default clock.
  std::optional<TimePoint> start_at{};  ///< Optional absolute phase in a comparable domain.
  ClockTickPolicy tick_policy{ClockTickPolicy::kCoalesceLateTicks};  ///< Closed baseline policy.
};

/// @brief Own a cold declaration-order list of borrowed input port references.
///
/// This value may allocate during authoring. A \ref Contract immediately copies each referenced
/// name and direction, so neither this list nor an authoring address enters a plan or run session.
class PortList {
 public:
  /// @brief Construct an empty list.
  PortList() = default;

  /// @brief Construct a declaration-order list from borrowed references.
  /// @param ports Borrowed references whose declarations outlive this cold value.
  explicit PortList(std::vector<PortRef> ports) : ports_(std::move(ports)) {}

  /// @brief Return declaration-order borrowed references.
  /// @return Span valid until this list is modified or destroyed.
  [[nodiscard]] std::span<const PortRef> values() const noexcept { return ports_; }

 private:
  std::vector<PortRef> ports_;
};

/// @brief Create an ordered port list for a closed multi-input trigger.
/// @tparam Ports Input-port declaration types convertible to \ref PortRef.
/// @param ports Stable input declarations borrowed for the duration of contract construction.
/// @return Cold declaration-order list.
template <typename... Ports>
  requires(sizeof...(Ports) > 0 &&
           (std::derived_from<std::remove_cvref_t<Ports>, InputBase> && ...))
[[nodiscard]] PortList ports(const Ports&... ports) {
  std::vector<PortRef> refs;
  refs.reserve(sizeof...(Ports));
  (refs.emplace_back(ports), ...);
  return PortList{std::move(refs)};
}

/// @brief Trigger a deterministic consuming FIFO-head alignment over a closed port set.
struct OnAligned {
  PortList ports;                     ///< At least two declaration-order member inputs.
  std::chrono::nanoseconds window{};  ///< Nonnegative inclusive maximum head spread.
};

/// @brief Trigger one bounded pending interval for each accepted anchor after priming.
struct OnWindow {
  PortRef anchor;                        ///< Local exact anchor input.
  std::chrono::nanoseconds max_wait{};   ///< Required positive processing wait bound.
  ClockKey wait_clock{};                 ///< Omission selects the graph default clock.
  std::uint32_t max_pending_windows{1};  ///< Required positive pending-record capacity.
};

/// @brief Trigger predicate evaluation for every accepted exact driver sample.
struct OnInputUpdate {
  PortRef port;  ///< Local scalar driver pinned as the implicit first evaluator input.
};

/// @brief Identify a stable semantic version for independently keyed predicate logic.
struct SemanticVersion {
  std::uint16_t major{};  ///< Breaking semantic version component.
  std::uint16_t minor{};  ///< Backward-compatible semantic version component.
  std::uint16_t patch{};  ///< Corrective semantic version component.

  friend constexpr bool operator==(SemanticVersion, SemanticVersion) noexcept = default;
};

/// @brief Opt into a qualified predicate identity independent of one operator instance.
struct PredicateKey {
  std::string_view qualified_name;     ///< Nonempty borrowed identity copied by \ref predicate.
  SemanticVersion semantic_version{};  ///< Explicit semantic version for replay evidence.
};

/// @brief Identify when a successful predicate result creates a compute candidate.
enum class PredicateFiring : std::uint8_t {
  kOnFalseToTrue = 0,      ///< Fire only after remembered false changes to true.
  kOnEveryTrueEvaluation,  ///< Fire after every successful true evaluation.
};

/// @brief Classify the owned callable form for later profile admission.
enum class PredicateCallableKind : std::uint8_t {
  kStaticFunction = 0,      ///< Free or static function pointer.
  kNonCapturingLambda = 1,  ///< Stateless fixed-signature callable value.
  kInlineFunctor = 2,       ///< Fixed-size owned functor with inspectable inline state.
  kOpaqueClosure = 3,       ///< Owned closure whose captured semantics are opaque to Core.
  kBoundMember = 4,         ///< Pointer-free self-member descriptor rebound per session.
  kTypeErasedFunction = 5,  ///< Explicit typed \c std::function value.
  kPythonFunction = 6,      ///< Reserved later Standard/Research Python function adapter.
  kPythonLambda = 7,        ///< Reserved later Standard/Research Python lambda adapter.
  kPythonBoundMethod = 8,   ///< Reserved later Standard/Research Python method adapter.
};

/// @brief Mark the absence of mutable predicate state.
struct NoPredicateState {};

/// @brief Constrain transactional predicate state to nonthrowing bounded commit operations.
/// @tparam State Candidate state type copied and swapped for every successful evaluation.
template <typename State>
concept PredicateState = std::default_initializable<std::remove_cvref_t<State>> &&
                         std::is_nothrow_copy_constructible_v<std::remove_cvref_t<State>> &&
                         std::is_nothrow_swappable_v<std::remove_cvref_t<State>> &&
                         std::is_nothrow_destructible_v<std::remove_cvref_t<State>>;

/// @brief Own declaration-order additional scalar predicate-input identities and payload types.
/// @tparam Payloads Exact published payload types exposed to the evaluator in declaration order.
///
/// The value stores copied port names, while its template arguments preserve the canonical
/// evaluator signature. Selector and missing-policy semantics continue to be declared through
/// \ref Contract::input and \ref Contract::on_missing.
template <PublishedPayload... Payloads>
class PredicateInputs {
 public:
  /// @brief Name the exact declaration-order payload tuple.
  using PayloadTuple = std::tuple<Payloads...>;

  /// @brief Return copied declaration-order input names.
  /// @return Borrowed views remain valid until this value is destroyed or assigned.
  [[nodiscard]] std::span<const std::string> names() const noexcept { return names_; }

 private:
  explicit PredicateInputs(std::vector<std::string> names) : names_(std::move(names)) {}
  std::vector<std::string> names_;

  friend struct detail::PredicateInputFactoryAccess;
};

namespace detail {
template <typename T>
inline constexpr bool is_predicate_inputs_v = false;

template <PublishedPayload... Payloads>
inline constexpr bool is_predicate_inputs_v<PredicateInputs<Payloads...>> = true;

struct PredicateInputFactoryAccess {
  template <PublishedPayload... Payloads>
  [[nodiscard]] static PredicateInputs<Payloads...> make(std::vector<std::string> names) {
    return PredicateInputs<Payloads...>{std::move(names)};
  }
};
}  // namespace detail

/// @brief Create a declaration-order list of additional scalar predicate inputs.
/// @tparam Payloads Published payload types of the input declarations.
/// @param inputs Stable input declarations whose names are copied immediately.
/// @return Owning cold predicate-input descriptor.
template <PublishedPayload... Payloads>
  requires(sizeof...(Payloads) > 0)
[[nodiscard]] PredicateInputs<Payloads...> predicate_inputs(const Input<Payloads>&... inputs) {
  std::vector<std::string> names;
  names.reserve(sizeof...(Payloads));
  (names.emplace_back(inputs.name()), ...);
  return detail::PredicateInputFactoryAccess::make<Payloads...>(std::move(names));
}

/// @brief Own one cold predicate identity, callable, optional state, config, and input list.
///
/// Copying deep-clones the owned values. The descriptor has no runtime receiver address, plan
/// identity, evaluator invocation API, or profile-admission authority. Compilation classifies the
/// immutable descriptor and session construction copies or rebinds the accepted callable.
/// Cold construction and copying may allocate and may propagate exceptions from owned values.
/// Concurrent access is not safe when any thread mutates or replaces the binding; externally
/// synchronize mutation and any overlapping observation.
class HOLOSCAN_CORE_EXPORT PredicateBinding {
 public:
  /// @brief Construct an invalid empty binding.
  PredicateBinding();

  /// @brief Deep-copy the owned cold predicate descriptor.
  /// @param other Descriptor to copy.
  PredicateBinding(const PredicateBinding& other);

  /// @brief Replace this descriptor with a deep copy.
  /// @param other Descriptor to copy.
  /// @return This descriptor after replacement.
  PredicateBinding& operator=(const PredicateBinding& other);

  /// @brief Move the complete cold descriptor.
  /// @param other Descriptor whose storage is transferred.
  PredicateBinding(PredicateBinding&& other) noexcept;

  /// @brief Replace this descriptor by moving another descriptor.
  /// @param other Descriptor whose storage is transferred.
  /// @return This descriptor after replacement.
  PredicateBinding& operator=(PredicateBinding&& other) noexcept;

  /// @brief Destroy all owned cold callable, state, and config values.
  ~PredicateBinding();

  /// @brief Return whether this descriptor contains a callable and valid identity.
  /// @return \c true for a factory-created binding.
  [[nodiscard]] explicit operator bool() const noexcept;

  /// @brief Return the copied local or qualified predicate name.
  /// @return Borrowed view valid until this binding is modified or destroyed.
  [[nodiscard]] std::string_view name() const noexcept;

  /// @brief Return the cold callable classification.
  /// @return Function, owned-functor, or explicit type-erased category.
  [[nodiscard]] PredicateCallableKind callable_kind() const noexcept;

  /// @brief Return whether an explicit config value is owned.
  /// @return \c true only for a config-bearing factory overload.
  [[nodiscard]] bool has_config() const noexcept;

  /// @brief Return the number of declared additional predicate inputs.
  /// @return Finite cold descriptor count.
  [[nodiscard]] std::size_t input_count() const noexcept;

 private:
  class Impl;

  PredicateBinding(std::string name, std::optional<SemanticVersion> version,
                   PredicateCallableKind kind, bool has_state, bool has_config, bool expects_driver,
                   bool callable_noexcept, bool callable_returns_expected,
                   std::size_t callable_size, std::size_t callable_alignment,
                   std::size_t state_size, std::size_t state_alignment, std::size_t config_size,
                   std::size_t config_alignment, bool config_identity_available,
                   std::string config_schema, std::string config_digest,
                   const void* driver_type_token, std::vector<std::string> input_names,
                   std::vector<const void*> input_type_tokens,
                   std::unique_ptr<detail::PredicateCallableStorage> storage);

  [[nodiscard]] bool expects_driver() const noexcept;

  std::unique_ptr<Impl> impl_;

  friend class detail::ContractBuilderAccess;
  friend class detail::ContractTestAccess;
  friend struct detail::PredicateBindingFactoryAccess;
};

namespace detail {
struct PredicateBindingFactoryAccess {
  [[nodiscard]] static PredicateBinding make(
      std::string name, std::optional<SemanticVersion> version, PredicateCallableKind kind,
      bool has_state, bool has_config, bool expects_driver, bool callable_noexcept,
      bool callable_returns_expected, std::size_t callable_size, std::size_t callable_alignment,
      std::size_t state_size, std::size_t state_alignment, std::size_t config_size,
      std::size_t config_alignment, bool config_identity_available, std::string config_schema,
      std::string config_digest, const void* driver_type_token,
      std::vector<std::string> input_names, std::vector<const void*> input_type_tokens,
      std::unique_ptr<PredicateCallableStorage> storage) {
    return PredicateBinding{std::move(name),
                            version,
                            kind,
                            has_state,
                            has_config,
                            expects_driver,
                            callable_noexcept,
                            callable_returns_expected,
                            callable_size,
                            callable_alignment,
                            state_size,
                            state_alignment,
                            config_size,
                            config_alignment,
                            config_identity_available,
                            std::move(config_schema),
                            std::move(config_digest),
                            driver_type_token,
                            std::move(input_names),
                            std::move(input_type_tokens),
                            std::move(storage)};
  }
};

template <typename T>
inline constexpr bool is_std_function_v = false;
template <typename R, typename... Args>
inline constexpr bool is_std_function_v<std::function<R(Args...)>> = true;

struct NoPredicateConfig {};

template <typename T, typename = void>
struct PredicateCallableSignature {};

template <typename Result, typename... Args>
struct PredicateCallableSignature<Result(Args...), void> {
  using result_type = Result;
  using argument_types = std::tuple<Args...>;
  using function_pointer_type = Result (*)(Args...);
  static constexpr bool is_noexcept = false;
};

template <typename Result, typename... Args>
struct PredicateCallableSignature<Result (*)(Args...), void>
    : PredicateCallableSignature<Result(Args...)> {};

template <typename Result, typename... Args>
struct PredicateCallableSignature<Result (*)(Args...) noexcept, void>
    : PredicateCallableSignature<Result(Args...)> {
  static constexpr bool is_noexcept = true;
};

template <typename Result, typename... Args>
struct PredicateCallableSignature<std::function<Result(Args...)>, void>
    : PredicateCallableSignature<Result(Args...)> {};

template <typename Owner, typename Result, typename... Args>
struct PredicateCallableSignature<Result (Owner::*)(Args...), void>
    : PredicateCallableSignature<Result(Args...)> {};

template <typename Owner, typename Result, typename... Args>
struct PredicateCallableSignature<Result (Owner::*)(Args...) const, void>
    : PredicateCallableSignature<Result(Args...)> {};

template <typename Owner, typename Result, typename... Args>
struct PredicateCallableSignature<Result (Owner::*)(Args...) noexcept, void>
    : PredicateCallableSignature<Result(Args...)> {
  static constexpr bool is_noexcept = true;
};

template <typename Owner, typename Result, typename... Args>
struct PredicateCallableSignature<Result (Owner::*)(Args...) const noexcept, void>
    : PredicateCallableSignature<Result(Args...)> {
  static constexpr bool is_noexcept = true;
};

template <typename Callable>
struct PredicateCallableSignature<Callable,
                                  std::void_t<decltype(&std::remove_cvref_t<Callable>::operator())>>
    : PredicateCallableSignature<decltype(&std::remove_cvref_t<Callable>::operator())> {};

template <typename Callable>
concept FixedPredicateCallable = requires {
  typename PredicateCallableSignature<std::remove_cvref_t<Callable>>::result_type;
  typename PredicateCallableSignature<std::remove_cvref_t<Callable>>::argument_types;
};

template <typename Callable>
struct PredicateBoundObject {
  using type = void;
};

template <typename... Tuples>
using tuple_cat_type_t = decltype(std::tuple_cat(std::declval<Tuples>()...));

template <typename State>
using predicate_state_arguments_t =
    std::conditional_t<std::same_as<State, NoPredicateState>, std::tuple<>, std::tuple<State&>>;

template <typename Config>
using predicate_config_arguments_t = std::conditional_t<std::same_as<Config, NoPredicateConfig>,
                                                        std::tuple<>, std::tuple<const Config&>>;

template <typename State, typename Config, PublishedPayload... Payloads>
using canonical_clock_predicate_arguments_t = tuple_cat_type_t<
    std::tuple<const PredicateEvaluationContext&, const PredicateInputView<Payloads>&...>,
    predicate_state_arguments_t<State>, predicate_config_arguments_t<Config>>;

template <typename Tuple>
struct PredicateArgumentsWithoutDriver;

template <typename Context, typename Driver, typename... Tail>
struct PredicateArgumentsWithoutDriver<std::tuple<Context, Driver, Tail...>> {
  using type = std::tuple<Context, Tail...>;
  using driver_type = Driver;
};

template <typename T>
inline constexpr bool is_predicate_input_view_argument_v = false;

template <PublishedPayload Payload>
inline constexpr bool is_predicate_input_view_argument_v<const PredicateInputView<Payload>&> = true;

template <typename T>
struct PredicateInputViewArgumentPayload;

template <PublishedPayload Payload>
struct PredicateInputViewArgumentPayload<const PredicateInputView<Payload>&> {
  using type = Payload;
};

struct NoPredicateDriver {};

template <typename Arguments, bool ClockDriven>
struct PredicateDriverFromArguments;

template <typename Arguments>
struct PredicateDriverFromArguments<Arguments, true> {
  using type = NoPredicateDriver;
};

template <typename Arguments>
struct PredicateDriverFromArguments<Arguments, false> {
  using argument_type = std::tuple_element_t<1, Arguments>;
  using type = typename PredicateInputViewArgumentPayload<argument_type>::type;
};

template <typename Driver, PublishedPayload... Payloads>
struct CanonicalPredicateSignature {
  using driver_payload = Driver;
  using additional_payloads = std::tuple<Payloads...>;
  static constexpr std::size_t input_count =
      sizeof...(Payloads) + (!std::same_as<Driver, NoPredicateDriver> ? 1U : 0U);
};

template <typename State, typename Config, typename Evaluator, PublishedPayload... Payloads>
struct CanonicalPredicateSignatureFor {
  using evaluator_type = std::decay_t<Evaluator>;
  using arguments = typename PredicateCallableSignature<evaluator_type>::argument_types;
  using clock_arguments =
      canonical_clock_predicate_arguments_t<std::remove_cvref_t<State>, std::remove_cvref_t<Config>,
                                            Payloads...>;
  static constexpr bool clock_driven = std::same_as<arguments, clock_arguments>;
  using driver_payload = typename PredicateDriverFromArguments<arguments, clock_driven>::type;
  using type = CanonicalPredicateSignature<driver_payload, Payloads...>;
};

template <typename Evaluator>
struct PredicateEvaluatorInvoker {
  template <typename... Args>
  [[nodiscard]] static decltype(auto) invoke(Evaluator& evaluator, void*, Args&&... args) {
    return std::invoke(evaluator, std::forward<Args>(args)...);
  }
};

template <typename T>
inline constexpr bool is_bound_predicate_v = false;

template <typename Signature, typename Evaluator, typename State, typename Config>
class PredicateCallableModel;

template <typename Driver, PublishedPayload... Payloads, typename Evaluator, typename State,
          typename Config>
class PredicateCallableModel<CanonicalPredicateSignature<Driver, Payloads...>, Evaluator, State,
                             Config>
    final : public PredicateCallableStorage {
 public:
  PredicateCallableModel(Evaluator evaluator, State state, Config config)
      : evaluator_(std::move(evaluator)),
        initial_state_(state),
        state_(std::move(state)),
        config_(std::move(config)) {}

  [[nodiscard]] std::unique_ptr<PredicateCallableStorage> clone() const override {
    return std::make_unique<PredicateCallableModel>(*this);
  }

  [[nodiscard]] expected<bool, Error> evaluate(const PredicateEvaluationContext& context,
                                               std::span<const void* const> input_views,
                                               PredicateReceiver self_receiver) override {
    candidate_state_.reset();
    if (input_views.size() != CanonicalPredicateSignature<Driver, Payloads...>::input_count) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (std::any_of(input_views.begin(), input_views.end(), [](const void* view) {
          return view == nullptr;
        })) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if constexpr (is_bound_predicate_v<Evaluator>) {
      using Object = typename PredicateBoundObject<Evaluator>::type;
      if (self_receiver.address == nullptr ||
          self_receiver.local_type_token != &kPredicateReceiverLocalTypeToken<Object>) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
    }
    auto evaluate_candidate = [&]() -> expected<bool, Error> {
      candidate_state_.emplace(state_);
      auto result = invoke(context,
                           input_views,
                           self_receiver,
                           *candidate_state_,
                           std::index_sequence_for<Payloads...>{});
      if constexpr (std::same_as<decltype(result), bool>) {
        return result;
      } else {
        if (!result) {
          candidate_state_.reset();
          return make_unexpected(std::move(result).error());
        }
        return *result;
      }
    };
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
      return evaluate_candidate();
    } catch (const std::exception&) {
      candidate_state_.reset();
      return make_unexpected(Error{ErrorCode::kFailure});
    } catch (...) {
      candidate_state_.reset();
      return make_unexpected(Error{ErrorCode::kFailure});
    }
#else
    return evaluate_candidate();
#endif
  }

  void commit_candidate() noexcept override {
    if (!candidate_state_) {
      return;
    }
    using std::swap;
    swap(state_, *candidate_state_);
    candidate_state_.reset();
  }

  void discard_candidate() noexcept override { candidate_state_.reset(); }

  void reset() override {
    candidate_state_.reset();
    State candidate = initial_state_;
    using std::swap;
    swap(state_, candidate);
  }

  [[nodiscard]] const void* receiver_type_token() const noexcept override {
    if constexpr (is_bound_predicate_v<Evaluator>) {
      using Object = typename PredicateBoundObject<Evaluator>::type;
      return &kPredicateReceiverLocalTypeToken<Object>;
    }
    return nullptr;
  }

 private:
  template <typename... Inputs>
  [[nodiscard]] decltype(auto) invoke_tail(PredicateReceiver self_receiver,
                                           const PredicateEvaluationContext& context,
                                           State& candidate, Inputs&&... inputs) {
    if constexpr (std::same_as<State, NoPredicateState> &&
                  std::same_as<Config, NoPredicateConfig>) {
      return PredicateEvaluatorInvoker<Evaluator>::invoke(
          evaluator_, self_receiver.address, context, std::forward<Inputs>(inputs)...);
    } else if constexpr (std::same_as<State, NoPredicateState>) {
      return PredicateEvaluatorInvoker<Evaluator>::invoke(evaluator_,
                                                          self_receiver.address,
                                                          context,
                                                          std::forward<Inputs>(inputs)...,
                                                          static_cast<const Config&>(config_));
    } else if constexpr (std::same_as<Config, NoPredicateConfig>) {
      return PredicateEvaluatorInvoker<Evaluator>::invoke(
          evaluator_, self_receiver.address, context, std::forward<Inputs>(inputs)..., candidate);
    } else {
      return PredicateEvaluatorInvoker<Evaluator>::invoke(evaluator_,
                                                          self_receiver.address,
                                                          context,
                                                          std::forward<Inputs>(inputs)...,
                                                          candidate,
                                                          static_cast<const Config&>(config_));
    }
  }

  template <std::size_t... Index>
  [[nodiscard]] decltype(auto) invoke(const PredicateEvaluationContext& context,
                                      std::span<const void* const> input_views,
                                      PredicateReceiver self_receiver, State& candidate,
                                      std::index_sequence<Index...>) {
    constexpr std::size_t offset = std::same_as<Driver, NoPredicateDriver> ? 0U : 1U;
    if constexpr (std::same_as<Driver, NoPredicateDriver>) {
      return invoke_tail(
          self_receiver,
          context,
          candidate,
          *static_cast<const PredicateInputView<Payloads>*>(input_views[offset + Index])...);
    } else {
      return invoke_tail(
          self_receiver,
          context,
          candidate,
          *static_cast<const PredicateInputView<Driver>*>(input_views[0]),
          *static_cast<const PredicateInputView<Payloads>*>(input_views[offset + Index])...);
    }
  }

  Evaluator evaluator_;
  State initial_state_;
  State state_;
  std::optional<State> candidate_state_;
  Config config_;
};

template <typename State, typename Config, typename Evaluator, PublishedPayload... Payloads>
consteval bool validate_predicate_signature() {
  using E = std::decay_t<Evaluator>;
  using S = std::remove_cvref_t<State>;
  using C = std::remove_cvref_t<Config>;
  static_assert(std::copy_constructible<E>, "predicate evaluator must be copy constructible");
  static_assert(std::default_initializable<S>, "predicate state must be default constructible");
  static_assert(std::is_nothrow_copy_constructible_v<S>,
                "predicate state must be nothrow copy constructible");
  static_assert(std::is_nothrow_swappable_v<S>,
                "predicate state must be nothrow swappable for transactional commit");
  static_assert(std::is_nothrow_destructible_v<S>, "predicate state must be nothrow destructible");
  static_assert(std::copy_constructible<C>, "predicate config must be copy constructible");
  static_assert(!std::is_member_function_pointer_v<E>,
                "unbound predicate member pointers are not supported");
  static_assert(FixedPredicateCallable<E>,
                "predicate evaluator must have one fixed non-generic signature");
  if constexpr (!FixedPredicateCallable<E>) {
    return false;
  } else {
    using Signature = PredicateCallableSignature<E>;
    using Result = typename Signature::result_type;
    using Args = typename Signature::argument_types;
    using ClockArgs = canonical_clock_predicate_arguments_t<S, C, Payloads...>;
    static_assert(
        std::same_as<Result, bool> || std::same_as<Result, expected<bool, Error>>,
        "predicate evaluator must return exactly bool or holoscan::expected<bool, Error>");

    constexpr bool clock_driven = std::same_as<Args, ClockArgs>;
    if constexpr (std::tuple_size_v<Args> < 2U) {
      static_assert(
          clock_driven,
          "predicate evaluator must match one canonical context/input/state/config signature");
      return false;
    } else {
      using WithoutDriver = PredicateArgumentsWithoutDriver<Args>;
      constexpr bool input_driven =
          is_predicate_input_view_argument_v<typename WithoutDriver::driver_type> &&
          std::same_as<typename WithoutDriver::type, ClockArgs>;
      static_assert(
          clock_driven || input_driven,
          "predicate evaluator must match one canonical context/input/state/config signature");
      return input_driven;
    }
  }
}

template <typename Evaluator>
[[nodiscard]] constexpr PredicateCallableKind predicate_callable_kind() noexcept {
  using E = std::decay_t<Evaluator>;
  constexpr bool has_function_pointer_conversion =
      std::is_convertible_v<E, typename PredicateCallableSignature<E>::function_pointer_type>;
  constexpr bool is_portable_inline_functor =
      std::is_empty_v<E> ||
      (std::is_aggregate_v<E> && std::is_trivially_copyable_v<E> && std::is_standard_layout_v<E>);
  if constexpr (is_bound_predicate_v<E>) {
    return PredicateCallableKind::kBoundMember;
  } else if constexpr (is_std_function_v<E>) {
    return PredicateCallableKind::kTypeErasedFunction;
  } else if constexpr (std::is_pointer_v<E> && std::is_function_v<std::remove_pointer_t<E>>) {
    return PredicateCallableKind::kStaticFunction;
  } else if constexpr (has_function_pointer_conversion) {
    // Fixed-signature noncapturing lambdas have the standard function-pointer conversion. This
    // does not rely on compiler lambda traits; a named callable that intentionally provides the
    // same conversion has equivalent static-dispatch capability for admission purposes.
    return PredicateCallableKind::kNonCapturingLambda;
  } else if constexpr (is_portable_inline_functor) {
    // An empty or structural named functor without the standard function-pointer conversion
    // remains inline; empty storage alone does not prove lambda provenance.
    return PredicateCallableKind::kInlineFunctor;
  } else {
    // Standard C++ has no portable lambda-type trait. Conservatively classify every remaining
    // owned closure as opaque so a constrained profile never receives stronger evidence than the
    // compiler can prove. A named aggregate functor takes the explicit inline-functor route above.
    return PredicateCallableKind::kOpaqueClosure;
  }
}

template <typename Driver>
[[nodiscard]] consteval const void* predicate_driver_type_token() noexcept {
  if constexpr (std::same_as<Driver, NoPredicateDriver>) {
    return nullptr;
  } else {
    return &kPublishedPayloadLocalTypeToken<Driver>;
  }
}

template <PublishedPayload... Payloads>
[[nodiscard]] std::vector<const void*> predicate_input_type_tokens() {
  return {&kPublishedPayloadLocalTypeToken<Payloads>...};
}

template <typename T, typename Absent>
inline constexpr std::size_t predicate_storage_size_v = std::same_as<T, Absent> ? 0U : sizeof(T);

template <typename T, typename Absent>
inline constexpr std::size_t predicate_storage_alignment_v =
    std::same_as<T, Absent> ? 0U : alignof(T);

template <typename Identity>
[[nodiscard]] std::pair<std::string, std::optional<SemanticVersion>> predicate_identity(
    Identity identity) {
  using I = std::remove_cvref_t<Identity>;
  if constexpr (std::same_as<I, PredicateKey>) {
    return {std::string{identity.qualified_name}, identity.semantic_version};
  } else {
    return {std::string{std::string_view{identity}}, std::nullopt};
  }
}

}  // namespace detail

/// @brief Create a configless predicate binding with no additional inputs.
/// @tparam State Fixed copyable state type, or \ref NoPredicateState.
/// @tparam Identity A local string-like identity or \ref PredicateKey.
/// @tparam Evaluator Copy-owned evaluator value.
/// @param identity Stable local or qualified identity copied into the descriptor.
/// @param evaluator Evaluator copied into cold storage.
/// @return Deep-owning cold predicate descriptor.
/// @throws std::bad_alloc if cold descriptor storage cannot be allocated.
/// @note Construction may also propagate an exception from copying or moving \p identity or
/// \p evaluator. Runtime evaluation uses a transactional copy of `State` when state is present.
template <typename State = NoPredicateState, typename Identity, typename Evaluator>
  requires PredicateState<State>
[[nodiscard]] PredicateBinding predicate(Identity&& identity, Evaluator&& evaluator) {
  constexpr bool expects_driver =
      detail::validate_predicate_signature<State, detail::NoPredicateConfig, Evaluator>();
  auto [name, version] = detail::predicate_identity(std::forward<Identity>(identity));
  using E = std::decay_t<Evaluator>;
  using S = std::remove_cvref_t<State>;
  using C = detail::NoPredicateConfig;
  using Signature = typename detail::CanonicalPredicateSignatureFor<S, C, E>::type;
  using Driver = typename Signature::driver_payload;
  auto storage = std::make_unique<detail::PredicateCallableModel<Signature, E, S, C>>(
      std::forward<Evaluator>(evaluator), S{}, C{});
  return detail::PredicateBindingFactoryAccess::make(
      std::move(name),
      version,
      detail::predicate_callable_kind<E>(),
      !std::same_as<S, NoPredicateState>,
      false,
      expects_driver,
      detail::PredicateCallableSignature<E>::is_noexcept,
      std::same_as<typename detail::PredicateCallableSignature<E>::result_type,
                   expected<bool, Error>>,
      sizeof(E),
      alignof(E),
      detail::predicate_storage_size_v<S, NoPredicateState>,
      detail::predicate_storage_alignment_v<S, NoPredicateState>,
      0U,
      0U,
      false,
      {},
      {},
      detail::predicate_driver_type_token<Driver>(),
      {},
      {},
      std::move(storage));
}

/// @brief Create a configless predicate binding with additional scalar inputs.
/// @tparam State Fixed copyable transactional state type, or `NoPredicateState`.
/// @tparam Identity A local string-like identity or `PredicateKey`.
/// @tparam Payloads Published payload types viewed in the declaration order of \p inputs.
/// @tparam Evaluator Copy-owned evaluator with the exact canonical driver/input/state signature.
/// @param identity Stable local or qualified identity copied into the descriptor.
/// @param inputs Declaration-order input names and payload identities copied into cold storage.
/// @param evaluator Evaluator copied into cold storage.
/// @return Deep-owning cold predicate descriptor that freezes the input list and evaluator.
/// @throws std::bad_alloc if cold descriptor or input-identity storage cannot be allocated.
/// @note Construction may also propagate an exception from copying or moving \p identity or
/// \p evaluator. Runtime evaluation presents one frozen typed view for every declared input.
template <typename State = NoPredicateState, typename Identity, PublishedPayload... Payloads,
          typename Evaluator>
  requires PredicateState<State>
[[nodiscard]] PredicateBinding predicate(Identity&& identity,
                                         const PredicateInputs<Payloads...>& inputs,
                                         Evaluator&& evaluator) {
  constexpr bool expects_driver = detail::
      validate_predicate_signature<State, detail::NoPredicateConfig, Evaluator, Payloads...>();
  auto [name, version] = detail::predicate_identity(std::forward<Identity>(identity));
  using E = std::decay_t<Evaluator>;
  using S = std::remove_cvref_t<State>;
  using C = detail::NoPredicateConfig;
  using Signature = typename detail::CanonicalPredicateSignatureFor<S, C, E, Payloads...>::type;
  using Driver = typename Signature::driver_payload;
  auto storage = std::make_unique<detail::PredicateCallableModel<Signature, E, S, C>>(
      std::forward<Evaluator>(evaluator), S{}, C{});
  return detail::PredicateBindingFactoryAccess::make(
      std::move(name),
      version,
      detail::predicate_callable_kind<E>(),
      !std::same_as<S, NoPredicateState>,
      false,
      expects_driver,
      detail::PredicateCallableSignature<E>::is_noexcept,
      std::same_as<typename detail::PredicateCallableSignature<E>::result_type,
                   expected<bool, Error>>,
      sizeof(E),
      alignof(E),
      detail::predicate_storage_size_v<S, NoPredicateState>,
      detail::predicate_storage_alignment_v<S, NoPredicateState>,
      0U,
      0U,
      false,
      {},
      {},
      detail::predicate_driver_type_token<Driver>(),
      {inputs.names().begin(), inputs.names().end()},
      detail::predicate_input_type_tokens<Payloads...>(),
      std::move(storage));
}

/// @brief Create a configured predicate binding with no additional inputs.
/// @tparam State Fixed copyable transactional state type, or `NoPredicateState`.
/// @tparam Identity A local string-like identity or `PredicateKey`.
/// @tparam Config Copy-owned immutable evaluator configuration type.
/// @tparam Evaluator Copy-owned evaluator with the exact canonical driver/state/config signature.
/// @param identity Stable local or qualified identity copied into the descriptor.
/// @param config Immutable configuration copied or moved into cold storage.
/// @param evaluator Evaluator copied into cold storage.
/// @return Deep-owning cold predicate descriptor with immutable configuration.
/// @throws std::bad_alloc if cold descriptor storage cannot be allocated.
/// @note Construction may also propagate an exception from copying or moving \p identity,
/// \p config, or \p evaluator. Config is passed as a const reference during evaluation.
template <typename State = NoPredicateState, typename Identity, typename Config, typename Evaluator>
  requires PredicateState<State> && std::copy_constructible<std::remove_cvref_t<Config>> &&
           (!detail::is_predicate_inputs_v<std::remove_cvref_t<Config>>)
[[nodiscard]] PredicateBinding predicate(Identity&& identity, Config&& config,
                                         Evaluator&& evaluator) {
  constexpr bool expects_driver = detail::validate_predicate_signature<State, Config, Evaluator>();
  auto [name, version] = detail::predicate_identity(std::forward<Identity>(identity));
  using E = std::decay_t<Evaluator>;
  using S = std::remove_cvref_t<State>;
  using C = std::remove_cvref_t<Config>;
  using Signature = typename detail::CanonicalPredicateSignatureFor<S, C, E>::type;
  using Driver = typename Signature::driver_payload;
  const auto config_identity = detail::predicate_config_identity(static_cast<const C&>(config));
  auto storage = std::make_unique<detail::PredicateCallableModel<Signature, E, S, C>>(
      std::forward<Evaluator>(evaluator), S{}, std::forward<Config>(config));
  return detail::PredicateBindingFactoryAccess::make(
      std::move(name),
      version,
      detail::predicate_callable_kind<E>(),
      !std::same_as<S, NoPredicateState>,
      true,
      expects_driver,
      detail::PredicateCallableSignature<E>::is_noexcept,
      std::same_as<typename detail::PredicateCallableSignature<E>::result_type,
                   expected<bool, Error>>,
      sizeof(E),
      alignof(E),
      detail::predicate_storage_size_v<S, NoPredicateState>,
      detail::predicate_storage_alignment_v<S, NoPredicateState>,
      sizeof(C),
      alignof(C),
      config_identity.has_value(),
      config_identity ? config_identity->schema : std::string{},
      config_identity ? config_identity->digest : std::string{},
      detail::predicate_driver_type_token<Driver>(),
      {},
      {},
      std::move(storage));
}

/// @brief Create a configured predicate binding with additional scalar inputs.
/// @tparam State Fixed copyable transactional state type, or `NoPredicateState`.
/// @tparam Identity A local string-like identity or `PredicateKey`.
/// @tparam Payloads Published payload types viewed in the declaration order of \p inputs.
/// @tparam Config Copy-owned immutable evaluator configuration type.
/// @tparam Evaluator Copy-owned evaluator with the exact canonical input/state/config signature.
/// @param identity Stable local or qualified identity copied into the descriptor.
/// @param inputs Declaration-order input names and payload identities copied into cold storage.
/// @param config Immutable configuration copied or moved into cold storage.
/// @param evaluator Evaluator copied into cold storage.
/// @return Deep-owning cold predicate descriptor that freezes inputs, config, and evaluator.
/// @throws std::bad_alloc if cold descriptor or input-identity storage cannot be allocated.
/// @note Construction may also propagate an exception from copying or moving \p identity,
/// \p config, or \p evaluator. Runtime evaluation uses one frozen typed input snapshot.
template <typename State = NoPredicateState, typename Identity, PublishedPayload... Payloads,
          typename Config, typename Evaluator>
  requires PredicateState<State> && std::copy_constructible<std::remove_cvref_t<Config>>
[[nodiscard]] PredicateBinding predicate(Identity&& identity,
                                         const PredicateInputs<Payloads...>& inputs,
                                         Config&& config, Evaluator&& evaluator) {
  constexpr bool expects_driver =
      detail::validate_predicate_signature<State, Config, Evaluator, Payloads...>();
  auto [name, version] = detail::predicate_identity(std::forward<Identity>(identity));
  using E = std::decay_t<Evaluator>;
  using S = std::remove_cvref_t<State>;
  using C = std::remove_cvref_t<Config>;
  using Signature = typename detail::CanonicalPredicateSignatureFor<S, C, E, Payloads...>::type;
  using Driver = typename Signature::driver_payload;
  const auto config_identity = detail::predicate_config_identity(static_cast<const C&>(config));
  auto storage = std::make_unique<detail::PredicateCallableModel<Signature, E, S, C>>(
      std::forward<Evaluator>(evaluator), S{}, std::forward<Config>(config));
  return detail::PredicateBindingFactoryAccess::make(
      std::move(name),
      version,
      detail::predicate_callable_kind<E>(),
      !std::same_as<S, NoPredicateState>,
      true,
      expects_driver,
      detail::PredicateCallableSignature<E>::is_noexcept,
      std::same_as<typename detail::PredicateCallableSignature<E>::result_type,
                   expected<bool, Error>>,
      sizeof(E),
      alignof(E),
      detail::predicate_storage_size_v<S, NoPredicateState>,
      detail::predicate_storage_alignment_v<S, NoPredicateState>,
      sizeof(C),
      alignof(C),
      config_identity.has_value(),
      config_identity ? config_identity->schema : std::string{},
      config_identity ? config_identity->digest : std::string{},
      detail::predicate_driver_type_token<Driver>(),
      {inputs.names().begin(), inputs.names().end()},
      detail::predicate_input_type_tokens<Payloads...>(),
      std::move(storage));
}

/// @brief Name the sole baseline predicate evaluation source alternatives.
using PredicateEvaluationSource = std::variant<OnInputUpdate, OnClock>;

/// @brief Trigger bounded predicate evaluation from one exact input update or one clock tick.
struct OnPredicate {
  PredicateEvaluationSource evaluate_on;  ///< Exact input-update or clock source.
  PredicateBinding predicate;             ///< Deep-owning cold callable descriptor.
  PredicateFiring firing{PredicateFiring::kOnFalseToTrue};  ///< Explicit firing rule.
};

/// @brief Complete one finite source after a fixed number of successful compute calls.
/// @note This \c holoscan.core rule is cold authoring state. Compilation freezes its value for
/// one logical Operator; it is not a scheduler selector or mutable run option.
struct ComputeCalls {
  std::uint64_t limit{};  ///< Exact limit; zero and \c UINT64_MAX are ordinary values.
};

/// @brief Build one closed, deep-owning operator temporal contract.
///
/// Builder calls are cold and may allocate while copying port names, predicate values, and trigger
/// descriptors. A contract owns no queue, runtime cache, scheduler, clock object, backend handle,
/// or authoring address after a builder call returns. Copying deep-copies the complete descriptor.
/// Builder calls may throw `std::bad_alloc` or propagate an exception from an owned cold value.
/// Concurrent access is not safe when any thread mutates the contract; externally synchronize
/// mutation and any overlapping observation or copy.
class HOLOSCAN_CORE_EXPORT Contract {
 public:
  /// @brief Construct a completely empty authoring contract.
  /// @note Construction allocates only cold descriptor storage and creates no runtime state.
  Contract();
  /// @brief Deep-copy an authored contract.
  /// @param other Contract whose owned trigger, selectors, policies, and predicate values to copy.
  /// @throws std::bad_alloc if cold storage cannot be copied.
  Contract(const Contract& other);
  /// @brief Replace this contract with a deep copy.
  /// @param other Contract whose complete cold descriptor replaces this value.
  /// @return This contract after successful replacement.
  /// @throws std::bad_alloc if replacement storage cannot be copied; this value remains unchanged.
  Contract& operator=(const Contract& other);
  /// @brief Move the complete authored contract.
  /// @param other Contract whose owned descriptor is transferred.
  Contract(Contract&& other) noexcept;
  /// @brief Replace this contract by moving another authored contract.
  /// @param other Contract whose owned descriptor is transferred.
  /// @return This contract after replacement.
  Contract& operator=(Contract&& other) noexcept;
  /// @brief Destroy all cold descriptor storage.
  ~Contract();

  /// @brief Declare the sole activation trigger.
  /// @param trigger Closed any-connected-input trigger value.
  /// @return This contract for chained cold authoring.
  /// @throws std::bad_alloc if copied descriptor storage cannot grow.
  /// @note Builder calls copy identities immediately. Structural conflicts are reported during
  /// compilation/normalization rather than by this cold mutator.
  Contract& trigger(OnAnyInput trigger);
  /// @brief Declare one exact input-update trigger.
  /// @param trigger Local input whose accepted sample is pinned by readiness.
  /// @return This contract for chained cold authoring.
  Contract& trigger(OnEach trigger);
  /// @brief Declare one periodic clock trigger.
  /// @param trigger Positive-period schedule copied into this contract.
  /// @return This contract for chained cold authoring.
  Contract& trigger(OnClock trigger);
  /// @brief Declare one logical readable-source trigger.
  /// @param trigger Bound logical source reference; no backend handle is retained.
  /// @return This contract for chained cold authoring.
  Contract& trigger(OnReadable trigger);
  /// @brief Declare one counted-notification trigger.
  /// @param trigger Bound logical notification source reference.
  /// @return This contract for chained cold authoring.
  Contract& trigger(OnNotified trigger);
  /// @brief Declare one monotonic doorbell trigger.
  /// @param trigger Bound logical doorbell source reference.
  /// @return This contract for chained cold authoring.
  Contract& trigger(OnDoorbell trigger);
  /// @brief Declare one false-to-true polling trigger.
  /// @param trigger Bound logical spin source reference.
  /// @return This contract for chained cold authoring.
  Contract& trigger(OnSpin trigger);
  /// @brief Declare one deterministic oldest-head alignment trigger.
  /// @param trigger Ordered member set and inclusive comparison window copied immediately.
  /// @return This contract for chained cold authoring.
  /// @throws std::bad_alloc if the copied member-name set cannot be stored.
  Contract& trigger(const OnAligned& trigger);
  /// @brief Declare one bounded consecutive-anchor window trigger.
  /// @param trigger Anchor, finite wait, wait clock, and pending bound to copy.
  /// @return This contract for chained cold authoring.
  Contract& trigger(OnWindow trigger);
  /// @brief Declare one typed input- or clock-driven predicate trigger.
  /// @param trigger Deep-owning predicate descriptor, source, and firing rule to copy.
  /// @return This contract for chained cold authoring.
  /// @throws std::bad_alloc if predicate storage cannot be copied.
  Contract& trigger(OnPredicate trigger);

  /// @brief Declare an exact scalar selector for one input.
  /// @tparam T Published payload type and immutable evaluator-access type.
  /// @param port Local input declaration whose name and type identity are copied.
  /// @param selector Fieldless exact selector.
  /// @return This contract for chained cold authoring.
  template <PublishedPayload T>
  Contract& input(const Input<T>& port, Exact selector) {
    return input_typed(PortRef{port}, selector, &detail::kPublishedPayloadLocalTypeToken<T>);
  }
  /// @brief Declare a finite latest-causal scalar selector for one input.
  /// @tparam T Published payload type.
  /// @param port Local input declaration whose name and type identity are copied.
  /// @param selector Positive inclusive maximum-age selector.
  /// @return This contract for chained cold authoring.
  template <PublishedPayload T>
  Contract& input(const Input<T>& port, LatestWithin selector) {
    return input_typed(PortRef{port}, selector, &detail::kPublishedPayloadLocalTypeToken<T>);
  }
  /// @brief Declare a fieldless bounded FIFO batch selector for one input.
  /// @tparam T Published payload type.
  /// @param port Local input declaration whose name and type identity are copied.
  /// @param selector Fieldless consume-all selector; compilation derives its finite bound.
  /// @return This contract for chained cold authoring.
  template <PublishedPayload T>
  Contract& input(const Input<T>& port, DrainAll selector) {
    return input_typed(PortRef{port}, selector, &detail::kPublishedPayloadLocalTypeToken<T>);
  }
  /// @brief Declare an all-or-nothing bracketed batch selector for one input.
  /// @tparam T Published payload type.
  /// @param port Local input declaration whose name and type identity are copied.
  /// @param selector Positive predecessor/successor counts and complete-set bound.
  /// @return This contract for chained cold authoring.
  template <PublishedPayload T>
  Contract& input(const Input<T>& port, BracketedWindow selector) {
    return input_typed(PortRef{port}, selector, &detail::kPublishedPayloadLocalTypeToken<T>);
  }

  /// @brief Declare a skip disposition for one selected input.
  /// @tparam T Published payload type.
  /// @param port Previously declared selected input.
  /// @param strategy Fieldless current-candidate suppression policy.
  /// @return This contract for chained cold authoring.
  template <PublishedPayload T>
  Contract& on_missing(const Input<T>& port, Skip strategy) {
    return on_missing(PortRef{port}, strategy);
  }
  /// @brief Declare the fieldless operator-authored safe disposition for one selected input.
  /// @tparam T Published payload type.
  /// @param port Previously declared selected input.
  /// @param strategy Fieldless degraded-compute policy; no sample is fabricated.
  /// @return This contract for chained cold authoring.
  template <PublishedPayload T>
  Contract& on_missing(const Input<T>& port, SafeDefault strategy) {
    return on_missing(PortRef{port}, strategy);
  }
  /// @brief Declare finite scalar reuse for one selected input.
  /// @tparam T Published payload type.
  /// @param port Previously declared scalar selected input.
  /// @param strategy Positive inclusive maximum reuse age.
  /// @return This contract for chained cold authoring.
  template <PublishedPayload T>
  Contract& on_missing(const Input<T>& port, ReuseLast strategy) {
    return on_missing(PortRef{port}, strategy);
  }

  /// @brief Declare an exact scalar selector through a borrowed authoring reference.
  /// @param port Borrowed local input reference whose name and direction are copied immediately.
  /// @param selector Fieldless exact selector.
  /// @return This contract for chained cold authoring.
  /// @note This untyped overload cannot preserve a payload token for predicate-signature checks;
  /// prefer the typed `Input<T>` overload for ordinary authoring.
  Contract& input(PortRef port, Exact selector);
  /// @brief Declare a finite latest-causal scalar selector through a borrowed authoring reference.
  /// @param port Borrowed local input reference whose name and direction are copied immediately.
  /// @param selector Positive inclusive maximum-age selector.
  /// @return This contract for chained cold authoring.
  /// @note This untyped overload cannot preserve a payload token for predicate-signature checks.
  Contract& input(PortRef port, LatestWithin selector);
  /// @brief Declare a FIFO batch selector through a borrowed authoring reference.
  /// @param port Borrowed local input reference whose name and direction are copied immediately.
  /// @param selector Fieldless consume-all selector whose finite bound compilation resolves.
  /// @return This contract for chained cold authoring.
  /// @note This untyped overload cannot preserve a payload token for predicate-signature checks.
  Contract& input(PortRef port, DrainAll selector);
  /// @brief Declare a bracketed batch selector through a borrowed authoring reference.
  /// @param port Borrowed local input reference whose name and direction are copied immediately.
  /// @param selector Positive outer counts and all-or-nothing selected-sample bound.
  /// @return This contract for chained cold authoring.
  /// @note This untyped overload cannot preserve a payload token for predicate-signature checks.
  Contract& input(PortRef port, BracketedWindow selector);
  /// @brief Declare a skip disposition through a borrowed authoring reference.
  /// @param port Borrowed reference naming a previously declared selected input.
  /// @param strategy Fieldless current-candidate suppression policy.
  /// @return This contract for chained cold authoring.
  Contract& on_missing(PortRef port, Skip strategy);
  /// @brief Declare a safe disposition through a borrowed authoring reference.
  /// @param port Borrowed reference naming a previously declared selected input.
  /// @param strategy Fieldless degraded-compute policy; no sample is fabricated.
  /// @return This contract for chained cold authoring.
  Contract& on_missing(PortRef port, SafeDefault strategy);
  /// @brief Declare finite scalar reuse through a borrowed authoring reference.
  /// @param port Borrowed reference naming a previously declared scalar selected input.
  /// @param strategy Positive inclusive maximum reuse age.
  /// @return This contract for chained cold authoring.
  Contract& on_missing(PortRef port, ReuseLast strategy);

  /// @brief Declare a positive finite latest-permitted-compute-start budget.
  /// @param budget Positive duration added to the frozen eligibility time with checked arithmetic.
  /// @return This contract for chained cold authoring.
  Contract& deadline(std::chrono::nanoseconds budget);
  /// @brief Declare natural completion after a fixed number of successful compute calls.
  /// @param rule Exact Operator-local compute-call rule. Zero and \c UINT64_MAX remain ordinary
  /// values and are never interpreted as absent or unbounded sentinels.
  /// @return This contract for chained cold authoring.
  /// @note Repeated calls replace the previously authored rule; the last call wins.
  /// @note This rule supplies no activation trigger. Without an explicit trigger, contract
  /// normalization reports `NO_TRIGGER` before completion-placement validation.
  Contract& complete_after(ComputeCalls rule);
  /// @brief Bind the latest-permitted-compute-start budget to one duration Parameter field.
  /// @tparam MemberPath Valid declared duration field path.
  /// @param dependency Compile-resolved dependency token.
  /// @return This contract for chained cold authoring.
  template <auto... MemberPath>
  Contract& deadline(ParamRef<MemberPath...> dependency) {
    return deadline_dependency(detail::make_param_duration_dependency(dependency));
  }
  /// @brief Declare the sole action applied when the final invocation gate is late.
  /// @param policy Closed safe-default, witness, or fail-run disposition.
  /// @return This contract for chained cold authoring.
  /// @note A deadline and violation policy must be authored together.
  Contract& on_violation(OnViolation policy);

 private:
  Contract& input_typed(PortRef port, Exact selector, const void* local_type_token);
  Contract& input_typed(PortRef port, LatestWithin selector, const void* local_type_token);
  Contract& input_typed(PortRef port, DrainAll selector, const void* local_type_token);
  Contract& input_typed(PortRef port, BracketedWindow selector, const void* local_type_token);
  Contract& deadline_dependency(detail::ParamDurationDependency dependency);

  class Impl;
  std::unique_ptr<Impl> impl_;
  friend class detail::ContractBuilderAccess;
  friend class detail::ContractTestAccess;
};

}  // namespace holoscan
