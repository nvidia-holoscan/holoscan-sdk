/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/channel/clock_health.hpp>
#include <holoscan/channel/pod_type.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/external_topic.hpp>
#include <holoscan/core/graph_ids.hpp>
#include <holoscan/core/graph_records.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_ref.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/parameter_input.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/payload_identity.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/profile.hpp>
#include <holoscan/core/rate.hpp>
#include <holoscan/core/readiness_source.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/schema_identity.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_field.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/clock.hpp>
#include <holoscan/time/clock_registry.hpp>

namespace holoscan {

namespace authoring {
class HOLOSCAN_CORE_EXPORT OpaqueGraphRef;
class HOLOSCAN_CORE_EXPORT OpaqueOperatorRef;
}  // namespace authoring

class Graph;
class HOLOSCAN_CORE_EXPORT TensorOutputPortConfigurator;
class ExecutionPartitionsView;

namespace detail {
class GraphState;
class GraphCompilerAccess;
class GraphOperatorRegistrationAccess;
class GraphPayloadRegistrationAccess;
class OperatorTestingAccess;
class ParameterConfigBindingAccess;
class ParameterInputAccess;
struct GraphSnapshotData;
struct ParameterInputEvidenceState;
}  // namespace detail

template <typename G>
class GraphHandle;

namespace detail {

class GraphState;
class GraphCompilerAccess;
struct GraphSnapshotData;
struct DiagnosticCapture;

/// @brief Log cold user-factory exception detail without placing logger macros in public headers.
/// @param context Stable bounded construction context token.
/// @param detail Arbitrary exception text used only for cold diagnostics.
HOLOSCAN_CORE_EXPORT void graph_log_factory_exception(std::string_view context,
                                                      std::string_view detail) noexcept;

/// @brief Capture the currently handled factory exception with bounded explanatory context.
/// @param token Stable construction or declaration failure token, or empty for cause text only.
/// @return Original category and contextual message, or a code-only allocation-safe fallback.
HOLOSCAN_CORE_EXPORT Error graph_factory_error(std::string_view token) noexcept;

/// @brief Capture into borrowed fixed storage, or render the ordinary authoring error.
/// @param token Stable authoring failure token when no capture is supplied.
/// @param capture Storage borrowed for this call only; nonnull capture returns a code-only Error.
/// @return Original error category with independently retained evidence.
HOLOSCAN_CORE_EXPORT Error graph_factory_error(std::string_view token,
                                               DiagnosticCapture* capture) noexcept;

class GraphPortOwnershipAccess {
 public:
  [[nodiscard]] static const void* owner(const InputBase& port) noexcept {
    return port.graph_owner_token_;
  }
  [[nodiscard]] static const void* owner(const OutputBase& port) noexcept {
    return port.graph_owner_token_;
  }
  static void claim(const InputBase& port, const void* owner) noexcept {
    port.graph_owner_token_ = owner;
  }
  static void claim(const OutputBase& port, const void* owner) noexcept {
    port.graph_owner_token_ = owner;
  }
};

template <typename T>
inline constexpr const std::byte& kGraphOperatorTypeToken = kLifecycleDeclaringTypeToken<T>;

template <typename T>
inline constexpr std::byte kGraphEmbeddingTypeToken{};

class GraphOperatorAccess {
 public:
  template <typename Op>
  [[nodiscard]] static expected<void, Error> bind(Op& op, typename Op::Params params) {
    const ParamValidationResult validation = validate_params(params);
    if (!validation.ok())
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, parameter_validation_error_message(validation)});
    if constexpr (std::same_as<typename Op::Params, NoParams>) {
      return {};
    } else if constexpr (std::is_nothrow_swappable_v<typename Op::Params>) {
      op.bind_initial_params(params);
      return {};
    } else {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
  }

 private:
  static constexpr std::size_t kMaximumRenderedIdentityBytes = 96U;

  [[nodiscard]] static constexpr std::string_view validation_code_name(
      ParamValidationCode code) noexcept {
    switch (code) {
      case ParamValidationCode::kOk:
        return "ok";
      case ParamValidationCode::kOutOfRange:
        return "out_of_range";
      case ParamValidationCode::kNotAllowed:
        return "not_allowed";
      case ParamValidationCode::kCardinalityExceeded:
        return "cardinality_exceeded";
      case ParamValidationCode::kInvalidEncoding:
        return "invalid_encoding";
      case ParamValidationCode::kValidatorRejected:
        return "validator_rejected";
    }
    return "unknown";
  }

  static void append_bounded_identity(std::string& message, std::string_view identity) {
    if (identity.size() <= kMaximumRenderedIdentityBytes) {
      message.append(identity);
      return;
    }

    detail::StableHashBuilder hash;
    hash.add("holoscan.param.validation-diagnostic.identity/v1");
    hash.add(identity);
    const ParamSchemaHash digest = hash.finish();
    constexpr std::string_view kHexDigits{"0123456789abcdef"};

    std::size_t prefix_bytes = kMaximumRenderedIdentityBytes;
    while (prefix_bytes != 0U && !detail::valid_utf8(identity.substr(0U, prefix_bytes))) {
      --prefix_bytes;
    }
    message.append(identity.substr(0U, prefix_bytes));
    message.append("...;bytes=");
    message.append(std::to_string(identity.size()));
    message.append(";sha256=");
    for (const std::byte byte : digest.value) {
      const auto value = static_cast<std::uint8_t>(byte);
      message.push_back(kHexDigits[value >> 4U]);
      message.push_back(kHexDigits[value & 0x0FU]);
    }
  }

  [[nodiscard]] static std::string parameter_validation_error_message(
      const ParamValidationResult& validation) {
    std::string message{"PARAM_VALIDATION_FAILED category="};
    message.append(validation_code_name(validation.code));
    message.append(" field_id=");
    message.append(std::to_string(validation.field_id));
    if (!validation.field_name.empty()) {
      message.append(" field_name=");
      append_bounded_identity(message, validation.field_name);
    }
    if (!validation.validator_id.empty()) {
      message.append(" validator_id=");
      append_bounded_identity(message, validation.validator_id);
    }
    return message;
  }
};

class OperatorFactoryBase {
 public:
  virtual ~OperatorFactoryBase() = default;
  [[nodiscard]] virtual expected<std::unique_ptr<OperatorBase>, Error> create(
      DiagnosticCapture* capture) const = 0;
  [[nodiscard]] virtual std::shared_ptr<OperatorFactoryBase> clone() const = 0;
  [[nodiscard]] virtual expected<std::shared_ptr<OperatorFactoryBase>, Error> clone_with_params(
      const void* params, const void* type_token) const = 0;

  /// @brief Return process-local whole-aggregate Parameter operations.
  /// @return Static descriptor for the concrete Operator's exact parameter type.
  /// @note The compiler copies only pointer-free descriptor fields into a plan.
  [[nodiscard]] virtual const ParamTypeDescriptor& parameter_type() const noexcept = 0;

  /// @brief Return the concrete Operator's declared Parameter schema.
  /// @return Static local descriptor, or \c nullptr for opaque-local mode.
  /// @note Local operation pointers remain private executable-factory metadata.
  [[nodiscard]] virtual const ParamSchemaDescriptor* parameter_schema() const noexcept = 0;

  /// @brief Return the immutable generation-zero aggregate used by this factory.
  /// @return Process-local typed value matching \ref parameter_type, never \c nullptr.
  /// @note Compilation may read declared fields through schema-owned local thunks. The address and
  /// object representation never enter an ExecutionPlan or portable artifact.
  [[nodiscard]] virtual const void* initial_parameter_value() const noexcept = 0;

  /// @brief Return whether authoring supplied a whole typed generation-zero baseline.
  /// @return \c true only when `OperatorHandle::set_params()` replaced `ParamsT{}`.
  /// @note Compilation uses this fact to preserve default-versus-baseline provenance without
  /// comparing values or inferring intent from aggregate equality.
  [[nodiscard]] virtual bool has_explicit_parameter_baseline() const noexcept = 0;

  /// @brief Return whether this factory's Operator uses \ref NoParams.
  /// @return \c true only for the static no-storage generation-zero contract.
  [[nodiscard]] virtual bool has_no_params() const noexcept = 0;
  struct LifecycleHookSet {
    bool start{};
    bool stop{};
  };
  [[nodiscard]] virtual LifecycleHookSet lifecycle_hooks() const noexcept = 0;
};

template <typename Owner>
auto lifecycle_start_owner(void (Owner::*)()) -> std::type_identity<Owner>;

template <typename Owner>
auto lifecycle_start_owner(void (Owner::*)() noexcept) -> std::type_identity<Owner>;

template <typename Owner>
auto lifecycle_stop_owner(void (Owner::*)()) -> std::type_identity<Owner>;

template <typename Owner>
auto lifecycle_stop_owner(void (Owner::*)() noexcept) -> std::type_identity<Owner>;

template <typename Op>
concept SelectableOperatorLifecycleHooks = requires {
  typename decltype(lifecycle_start_owner(&Op::start))::type;
  typename decltype(lifecycle_stop_owner(&Op::stop))::type;
};

template <typename Op>
  requires SelectableOperatorLifecycleHooks<Op>
[[nodiscard]] constexpr OperatorFactoryBase::LifecycleHookSet operator_lifecycle_hooks() noexcept {
  using StartOwner = typename decltype(lifecycle_start_owner(&Op::start))::type;
  using StopOwner = typename decltype(lifecycle_stop_owner(&Op::stop))::type;
  return {
      .start = !std::same_as<StartOwner, OperatorBase>,
      .stop = !std::same_as<StopOwner, OperatorBase>,
  };
}

template <typename T>
struct IsSpan : std::false_type {};
template <typename T, std::size_t Extent>
struct IsSpan<std::span<T, Extent>> : std::true_type {};
template <typename T>
struct IsReferenceWrapper : std::false_type {};
template <typename T>
struct IsReferenceWrapper<std::reference_wrapper<T>> : std::true_type {};
template <typename T>
struct IsInitializerList : std::false_type {};
template <typename T>
struct IsInitializerList<std::initializer_list<T>> : std::true_type {};
template <typename T>
inline constexpr bool kStringCapture =
    std::same_as<std::remove_cvref_t<T>, std::string> ||
    std::same_as<std::remove_cvref_t<T>, std::string_view> ||
    (std::is_array_v<std::remove_reference_t<T>> &&
     std::same_as<std::remove_extent_t<std::remove_reference_t<T>>, const char>);

template <std::size_t N>
// A bounded C-array reference is the only portable way to distinguish a string literal from a
// borrowed character pointer before converting it into Graph-owned storage.
[[nodiscard]] std::string normalize_graph_capture(
    const char (&value)[N]) {  // NOLINT(cppcoreguidelines-avoid-c-arrays)
  return std::string(value, N == 0 ? 0 : N - 1);
}

[[nodiscard]] inline std::string normalize_graph_capture(std::string_view value) {
  return std::string(value);
}

template <typename T>
  requires(!std::is_array_v<std::remove_reference_t<T>> &&
           !std::same_as<std::remove_cvref_t<T>, std::string_view>)
[[nodiscard]] std::remove_cvref_t<T> normalize_graph_capture(T&& value) {
  return std::forward<T>(value);
}

template <typename T>
using NormalizedGraphCapture = decltype(normalize_graph_capture(std::declval<T>()));

template <typename T>
concept OperatorParameterCapture = requires(const T& capture) {
  typename T::HoloscanOperatorParameterCapture;
  { capture.parameter_type() } -> std::same_as<const ParamTypeDescriptor&>;
  { capture.parameter_schema() } -> std::same_as<const ParamSchemaDescriptor*>;
  { capture.initial_parameter_value() } -> std::same_as<const void*>;
  { capture.has_explicit_parameter_baseline() } -> std::same_as<bool>;
  { capture.has_no_params() } -> std::same_as<bool>;
};

template <typename Tuple>
struct OperatorParameterCaptureTuple : std::false_type {};

template <OperatorParameterCapture Capture>
struct OperatorParameterCaptureTuple<std::tuple<Capture>> : std::true_type {};
template <typename Op, typename... Captures>
class OperatorFactoryModel final : public OperatorFactoryBase {
 public:
  static_assert((!OperatorParameterCapture<Captures> && ...) ||
                    OperatorParameterCaptureTuple<std::tuple<Captures...>>::value,
                "an Operator parameter capture must be the factory's only constructor capture");

  explicit OperatorFactoryModel(Captures... captures) : captures_(std::move(captures)...) {}

  [[nodiscard]] expected<std::unique_ptr<OperatorBase>, Error> create(
      DiagnosticCapture* capture) const override {
    try {
      auto op = std::apply([](const auto&... values) { return std::make_unique<Op>(values...); },
                           captures_);
      if (params_) {
        auto params = *params_;
        auto bound = GraphOperatorAccess::bind(*op, std::move(params));
        if (!bound)
          return make_unexpected(std::move(bound.error()));
      }
      return std::unique_ptr<OperatorBase>(std::move(op));
    } catch (const RuntimeError& exception) {
      auto error = graph_factory_error("GRAPH_OPERATOR_CONSTRUCTION_FAILED", capture);
      graph_log_factory_exception("operator", exception.what());
      return make_unexpected(std::move(error));
    } catch (const std::exception& exception) {
      auto error = graph_factory_error("GRAPH_OPERATOR_CONSTRUCTION_FAILED", capture);
      graph_log_factory_exception("operator", exception.what());
      return make_unexpected(std::move(error));
    } catch (...) {
      auto error = graph_factory_error("GRAPH_OPERATOR_CONSTRUCTION_FAILED", capture);
      graph_log_factory_exception("operator", "non-standard exception");
      return make_unexpected(std::move(error));
    }
  }

  [[nodiscard]] std::shared_ptr<OperatorFactoryBase> clone() const override {
    return std::make_shared<OperatorFactoryModel>(*this);
  }

  [[nodiscard]] expected<std::shared_ptr<OperatorFactoryBase>, Error> clone_with_params(
      const void* params, const void* type_token) const override {
    if constexpr (OperatorParameterCaptureTuple<std::tuple<Captures...>>::value) {
      (void)params;
      (void)type_token;
      return make_unexpected(
          Error{ErrorCode::kNotSupported, "GRAPH_CAPTURED_PARAMETER_REPLACEMENT_UNSUPPORTED"});
    }
    if (params == nullptr ||
        type_token != param_type_descriptor<typename Op::Params>().local_type_token) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument, "GRAPH_PARAMETER_TYPE_MISMATCH"});
    }
    auto copy = std::make_shared<OperatorFactoryModel>(*this);
    copy->params_.reset();
    copy->params_.emplace(*static_cast<const typename Op::Params*>(params));
    return std::shared_ptr<OperatorFactoryBase>(std::move(copy));
  }

  [[nodiscard]] const ParamTypeDescriptor& parameter_type() const noexcept override {
    if constexpr (OperatorParameterCaptureTuple<std::tuple<Captures...>>::value) {
      return std::get<0>(captures_).parameter_type();
    }
    return param_type_descriptor<typename Op::Params>();
  }

  [[nodiscard]] const ParamSchemaDescriptor* parameter_schema() const noexcept override {
    if constexpr (OperatorParameterCaptureTuple<std::tuple<Captures...>>::value) {
      return std::get<0>(captures_).parameter_schema();
    }
    return param_schema_descriptor<typename Op::Params>();
  }

  [[nodiscard]] const void* initial_parameter_value() const noexcept override {
    if constexpr (OperatorParameterCaptureTuple<std::tuple<Captures...>>::value) {
      return std::get<0>(captures_).initial_parameter_value();
    }
    if (params_) {
      return &*params_;
    }
    if constexpr (std::same_as<typename Op::Params, NoParams>) {
      return &kNoParams;
    } else {
      static const typename Op::Params default_value{};
      return &default_value;
    }
  }

  [[nodiscard]] bool has_explicit_parameter_baseline() const noexcept override {
    if constexpr (OperatorParameterCaptureTuple<std::tuple<Captures...>>::value) {
      return std::get<0>(captures_).has_explicit_parameter_baseline();
    }
    return params_.has_value();
  }

  [[nodiscard]] bool has_no_params() const noexcept override {
    if constexpr (OperatorParameterCaptureTuple<std::tuple<Captures...>>::value) {
      return std::get<0>(captures_).has_no_params();
    }
    return std::same_as<typename Op::Params, NoParams>;
  }

  [[nodiscard]] LifecycleHookSet lifecycle_hooks() const noexcept override {
    return operator_lifecycle_hooks<Op>();
  }

 private:
  std::tuple<Captures...> captures_;
  std::optional<typename Op::Params> params_;
};

class ClockFactoryBase {
 public:
  virtual ~ClockFactoryBase() = default;
  [[nodiscard]] virtual expected<std::unique_ptr<Clock>, Error> create(
      DiagnosticCapture* capture) const = 0;
  [[nodiscard]] virtual std::shared_ptr<ClockFactoryBase> clone() const = 0;
};

template <typename ClockT, typename... Captures>
class ClockFactoryModel final : public ClockFactoryBase {
 public:
  explicit ClockFactoryModel(Captures... captures) : captures_(std::move(captures)...) {}
  [[nodiscard]] expected<std::unique_ptr<Clock>, Error> create(
      DiagnosticCapture* capture) const override {
    try {
      auto clock = std::apply(
          [](const auto&... values) { return std::make_unique<ClockT>(values...); }, captures_);
      return std::unique_ptr<Clock>(std::move(clock));
    } catch (const std::exception& exception) {
      auto error = graph_factory_error("GRAPH_CLOCK_CONSTRUCTION_FAILED", capture);
      graph_log_factory_exception("clock", exception.what());
      return make_unexpected(std::move(error));
    } catch (...) {
      auto error = graph_factory_error("GRAPH_CLOCK_CONSTRUCTION_FAILED", capture);
      graph_log_factory_exception("clock", "non-standard exception");
      return make_unexpected(std::move(error));
    }
  }
  [[nodiscard]] std::shared_ptr<ClockFactoryBase> clone() const override {
    return std::make_shared<ClockFactoryModel>(*this);
  }

 private:
  std::tuple<Captures...> captures_;
};

template <typename T>
inline constexpr bool kStableExternalScalar =
    std::same_as<T, bool> || std::same_as<T, std::uint8_t> || std::same_as<T, std::int8_t> ||
    std::same_as<T, std::uint16_t> || std::same_as<T, std::int16_t> ||
    std::same_as<T, std::uint32_t> || std::same_as<T, std::int32_t> ||
    std::same_as<T, std::uint64_t> || std::same_as<T, std::int64_t> ||
    (std::same_as<T, float> && std::numeric_limits<float>::is_iec559 && sizeof(float) == 4) ||
    (std::same_as<T, double> && std::numeric_limits<double>::is_iec559 && sizeof(double) == 8);

template <typename T>
concept ExternalTopicPayload =
    PublishedPayload<T> && (kStableExternalScalar<T> || std::same_as<T, Tensor> ||
                            (HasSchemaIdentity<T> != HasChannelPodContract<T>));

struct OperatorRegistration {
  std::size_t index{};
  std::shared_ptr<const void> handle_identity;
};

struct InitialOperatorParameters {
  const void* value{};
  const void* local_type_token{};
  const ParamSchemaDescriptor* schema{};
};

struct FlowRegistration {
  EdgeId edge{};
  GraphPortId source{};
  GraphPortId target{};
  std::shared_ptr<const void> handle_identity;
};

struct RegionRegistration {
  RegionId id{};
  std::shared_ptr<const void> handle_identity;
};

struct PartitionRegistration {
  PartitionId id{};
  std::shared_ptr<const void> handle_identity;
};

struct EmbeddedGraphRegistration {
  std::size_t index{};
  std::shared_ptr<const void> handle_identity;  ///< Non-reusable exact registration identity.
};

/// @brief Invoke one type-erased operation inside a Graph-owned mutation rollback scope.
using GraphAtomicMutationOperation = expected<void, Error> (*)(void* context);

/// @brief Carry one defining-module parameter target for optional authoring tooling.
struct GraphOperatorParameterTarget {
  VertexId vertex{};
  const ParamSchemaDescriptor* schema{};
};

// These exported functions are private ABI bridges for the inline public authoring types below.
// HOLOSCAN_CORE_EXPORT makes their out-of-line definitions linkable from the Core DSO; it does not
// make them supported user APIs. Applications should use Graph and its public handle types.
/// @cond INTERNAL_GRAPH_BRIDGES
HOLOSCAN_CORE_EXPORT expected<OperatorRegistration, Error> graph_register_operator(
    const std::shared_ptr<GraphState>& state, std::string_view name,
    std::shared_ptr<OperatorFactoryBase> factory, std::unique_ptr<OperatorBase> prototype,
    const void* operator_type_token, const InitialOperatorParameters* initial_params = nullptr);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_prepare_authoring(
    const std::shared_ptr<GraphState>& state);
/// @brief Run one compound authoring operation atomically against an already composed Graph.
/// @param state Owning Graph state.
/// @param context Call-scoped storage passed to \p operation.
/// @param operation Non-null callback that may perform nested public Graph mutations.
/// @return The callback result after commit, or its error after restoring the prior Graph state.
/// @throws Any callback exception after restoring the prior Graph state.
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_apply_atomic_mutation(
    const std::shared_ptr<GraphState>& state, void* context,
    GraphAtomicMutationOperation operation);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_prepare_embed(
    const std::shared_ptr<GraphState>& state, std::string_view instance_name);
HOLOSCAN_CORE_EXPORT const OperatorBase* graph_operator_prototype(
    const std::shared_ptr<GraphState>& state, std::size_t index, const void* operator_type_token,
    const void* handle_identity) noexcept;
HOLOSCAN_CORE_EXPORT const void* graph_operator_port_declaration(
    const std::shared_ptr<GraphState>& state, std::size_t index, const void* handle_identity,
    std::string_view name) noexcept;
HOLOSCAN_CORE_EXPORT expected<std::vector<PortRecord>, Error> graph_operator_ports(
    const std::shared_ptr<GraphState>& state, std::size_t index, const void* operator_type_token,
    const void* handle_identity);
HOLOSCAN_CORE_EXPORT expected<const void*, Error> graph_resolve_operator_port(
    const std::shared_ptr<GraphState>& state, std::size_t index, const void* operator_type_token,
    const void* handle_identity, std::string_view name, PortDirection direction,
    const PayloadIdentityRecord& payload_identity);
// Exact native-type variant used by optional testing adapters; the existing erased ABI is retained.
HOLOSCAN_CORE_EXPORT expected<const void*, Error> graph_resolve_operator_port(
    const std::shared_ptr<GraphState>& state, std::size_t index, const void* operator_type_token,
    const void* handle_identity, std::string_view name, PortDirection direction,
    const PayloadIdentityRecord& payload_identity, const void* payload_local_type_token);
// Cold RTTI supplements the local token only for schema/POD payloads. Both the native ABI and
// the complete semantic identity must agree. Providers must remain loaded while their native
// declarations or type witnesses are reachable; neither witness is serialized into Graph views.
HOLOSCAN_CORE_EXPORT expected<const void*, Error> graph_resolve_operator_port(
    const std::shared_ptr<GraphState>& state, std::size_t index, const void* operator_type_token,
    const void* handle_identity, std::string_view name, PortDirection direction,
    const PayloadIdentityRecord& payload_identity, const void* payload_local_type_token,
    const std::type_info* payload_local_type_info);
HOLOSCAN_CORE_EXPORT expected<PortRecord, Error> graph_operator_port_record(
    const std::shared_ptr<GraphState>& state, std::size_t index, const void* operator_type_token,
    const void* handle_identity, std::string_view name, PortDirection direction);
HOLOSCAN_CORE_EXPORT bool graph_operator_valid(const std::shared_ptr<GraphState>& state,
                                               std::size_t index, const void* operator_type_token,
                                               const void* handle_identity) noexcept;
HOLOSCAN_CORE_EXPORT std::string_view graph_operator_name(const std::shared_ptr<GraphState>& state,
                                                          std::size_t index,
                                                          const void* handle_identity) noexcept;
HOLOSCAN_CORE_EXPORT expected<std::string, Error> graph_operator_path(
    const std::shared_ptr<GraphState>& state, std::size_t index, const void* handle_identity);
HOLOSCAN_CORE_EXPORT expected<OperatorRegistration, Error> graph_find_operator(
    const std::shared_ptr<GraphState>& state, std::string_view name) noexcept;
HOLOSCAN_CORE_EXPORT OperatorRef graph_operator_ref(const std::shared_ptr<GraphState>& state,
                                                    std::size_t index,
                                                    const void* handle_identity) noexcept;
HOLOSCAN_CORE_EXPORT expected<VertexId, Error> graph_operator_vertex(
    const std::shared_ptr<GraphState>& state, std::size_t index, const void* operator_type_token,
    const void* handle_identity);
HOLOSCAN_CORE_EXPORT expected<GraphOperatorParameterTarget, Error> graph_operator_parameter_target(
    const std::shared_ptr<GraphState>& state, std::size_t index, const void* operator_type_token,
    const void* handle_identity);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_set_operator_params(
    const std::shared_ptr<GraphState>& state, std::size_t index, const void* params,
    const void* operator_type_token, const void* handle_identity,
    const void* caller_parameter_type_token, const ParamSchemaDescriptor* caller_schema,
    std::shared_ptr<const ParameterInputEvidenceState> evidence = nullptr);

HOLOSCAN_CORE_EXPORT expected<FlowRegistration, Error> graph_add_flow(
    const std::shared_ptr<GraphState>& state, const OutputBase* source, const InputBase* target,
    const PayloadIdentityRecord& source_identity, const PayloadIdentityRecord& target_identity,
    ConnectionOptions options,
    std::optional<TensorProjectionRecord> tensor_projection = std::nullopt);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_bind_input_interface(
    const std::shared_ptr<GraphState>& state, const InputBase* first, const InputBase* second,
    PayloadIdentityRecord payload_identity);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_bind_output_interface(
    const std::shared_ptr<GraphState>& state, const OutputBase* first, const OutputBase* second,
    PayloadIdentityRecord payload_identity);
HOLOSCAN_CORE_EXPORT expected<GraphPortId, Error> graph_find_port(
    const std::shared_ptr<GraphState>& state, const void* declaration, PortDirection direction,
    const PayloadIdentityRecord& payload_identity);
HOLOSCAN_CORE_EXPORT expected<std::shared_ptr<const void>, Error> graph_port_registration_identity(
    const std::shared_ptr<GraphState>& state, GraphPortId port);
HOLOSCAN_CORE_EXPORT const void* graph_port_declaration(const std::shared_ptr<GraphState>& state,
                                                        GraphPortId port,
                                                        PortDirection direction) noexcept;
HOLOSCAN_CORE_EXPORT const void* graph_port_declaration(const std::shared_ptr<GraphState>& state,
                                                        GraphPortId port, PortDirection direction,
                                                        const void* handle_identity) noexcept;
HOLOSCAN_CORE_EXPORT const void* graph_port_declaration(
    const std::shared_ptr<GraphState>& state, GraphPortId port, PortDirection direction,
    const PayloadIdentityRecord& payload_identity) noexcept;
// Cold testing lookup retains the same registration capability as Graph::add_flow.
HOLOSCAN_CORE_EXPORT expected<FlowRegistration, Error> graph_testing_flow_registration(
    const std::shared_ptr<GraphState>& state, GraphPortId source, GraphPortId target,
    const void* source_identity, const void* target_identity);
HOLOSCAN_CORE_EXPORT bool graph_flow_valid(const std::shared_ptr<GraphState>& state, EdgeId edge,
                                           GraphPortId source, GraphPortId target,
                                           const void* handle_identity) noexcept;

HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_input_queue(
    const std::shared_ptr<GraphState>& state, GraphPortId port, const void* handle_identity,
    std::uint32_t depth);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_input_tensor_spec(
    const std::shared_ptr<GraphState>& state, GraphPortId port, const void* handle_identity,
    const TensorInputSpec& spec);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_port_alias(
    const std::shared_ptr<GraphState>& state, GraphPortId port, const void* handle_identity,
    std::string_view alias);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_output_emits(
    const std::shared_ptr<GraphState>& state, GraphPortId port, const void* handle_identity,
    std::uint32_t count);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_output_history(
    const std::shared_ptr<GraphState>& state, GraphPortId port, const void* handle_identity,
    std::uint32_t depth);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_output_tensor_spec(
    const std::shared_ptr<GraphState>& state, GraphPortId port, const void* handle_identity,
    const TensorOutputSpec& spec);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_output_payload(
    const std::shared_ptr<GraphState>& state, GraphPortId port, const void* handle_identity,
    PayloadOptions options);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_flow_policy(
    const std::shared_ptr<GraphState>& state, EdgeId edge, const void* handle_identity,
    QueuePolicy policy);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_flow_depth(
    const std::shared_ptr<GraphState>& state, EdgeId edge, const void* handle_identity,
    std::uint32_t depth);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_flow_topic(
    const std::shared_ptr<GraphState>& state, EdgeId edge, const void* handle_identity,
    std::string_view topic_name);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_flow_memory(
    const std::shared_ptr<GraphState>& state, EdgeId edge, const void* handle_identity,
    MemoryConversionPolicy policy);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_flow_inflight(
    const std::shared_ptr<GraphState>& state, EdgeId edge, const void* handle_identity,
    std::uint32_t count);

HOLOSCAN_CORE_EXPORT expected<GraphPortId, Error> graph_register_import(
    const std::shared_ptr<GraphState>& state, std::string_view uri, TopicOptions options,
    std::unique_ptr<OutputBase> port, PayloadIdentityRecord payload_identity,
    std::uint64_t serialized_payload_capacity_bytes,
    std::vector<AuthoredDescriptorTensorTopicField> tensor_fields = {},
    std::uint16_t descriptor_tensor_field_count = 0U);
HOLOSCAN_CORE_EXPORT expected<GraphPortId, Error> graph_register_export(
    const std::shared_ptr<GraphState>& state, std::string_view uri, TopicOptions options,
    std::unique_ptr<InputBase> port, PayloadIdentityRecord payload_identity,
    std::uint64_t serialized_payload_capacity_bytes,
    std::vector<AuthoredDescriptorTensorTopicField> tensor_fields = {},
    std::uint16_t descriptor_tensor_field_count = 0U);

HOLOSCAN_CORE_EXPORT expected<RegionRegistration, Error> graph_add_region(
    const std::shared_ptr<GraphState>& state, std::string_view name);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_region_add_operator(
    const std::shared_ptr<GraphState>& state, RegionId region, const void* region_handle_identity,
    std::size_t operator_index);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_region_add_operators(
    const std::shared_ptr<GraphState>& state, RegionId region, const void* region_handle_identity,
    std::span<const std::size_t> operator_indexes);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_region_add_embedded_graph(
    const std::shared_ptr<GraphState>& state, RegionId region, const void* region_handle_identity,
    std::size_t embedded_graph_index, const void* graph_handle_identity);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_region_add_members(
    const std::shared_ptr<GraphState>& state, RegionId region, const void* region_handle_identity,
    std::span<const std::size_t> operator_indexes,
    std::span<const std::size_t> embedded_graph_indexes);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_region_add_constraint(
    const std::shared_ptr<GraphState>& state, RegionId region, const void* region_handle_identity,
    RegionConstraint constraint);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_region_set_profile(
    const std::shared_ptr<GraphState>& state, RegionId region, const void* region_handle_identity,
    Profile profile);

HOLOSCAN_CORE_EXPORT expected<PartitionRegistration, Error> graph_add_partition(
    const std::shared_ptr<GraphState>& state, std::string_view name);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_partition_add_operator(
    const std::shared_ptr<GraphState>& state, PartitionId partition,
    const void* partition_handle_identity, std::size_t operator_index);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_partition_add_operators(
    const std::shared_ptr<GraphState>& state, PartitionId partition,
    const void* partition_handle_identity, std::span<const std::size_t> operator_indexes);
/// @brief Assign one direct-child instance without partial partition membership updates.
/// @pre The non-null GraphState comes from matching live partition and Graph handles.
/// @return An error for invalid identities, conflicting assignments, an empty subtree or
/// internal partition declarations; otherwise publish the complete instance selection.
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_partition_add_embedded_graph(
    const std::shared_ptr<GraphState>& state, PartitionId partition,
    const void* partition_handle_identity, std::size_t embedded_graph_index,
    const void* graph_handle_identity);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_partition_include_all(
    const std::shared_ptr<GraphState>& state, PartitionId partition,
    const void* partition_handle_identity);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_auto_partition_remaining(
    const std::shared_ptr<GraphState>& state);

HOLOSCAN_CORE_EXPORT expected<void, Error> graph_lifecycle_before(
    const std::shared_ptr<GraphState>& state, std::size_t before_index, std::size_t after_index,
    LifecycleStage stage);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_lifecycle_permission(
    const std::shared_ptr<GraphState>& state, LifecycleRestartOrigin origin,
    std::optional<std::size_t> requester_index, std::size_t target_index);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_lifecycle_before_resolved(
    const std::shared_ptr<GraphState>& state, const std::shared_ptr<GraphState>& before_state,
    std::size_t before_index, const std::shared_ptr<GraphState>& after_state,
    std::size_t after_index, LifecycleStage stage);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_lifecycle_permission_resolved(
    const std::shared_ptr<GraphState>& state, LifecycleRestartOrigin origin,
    const std::shared_ptr<GraphState>& requester_state, std::optional<std::size_t> requester_index,
    const std::shared_ptr<GraphState>& target_state, std::size_t target_index);

HOLOSCAN_CORE_EXPORT expected<ClockKey, Error> graph_add_clock(
    const std::shared_ptr<GraphState>& state, std::string_view name,
    std::shared_ptr<ClockFactoryBase> factory);
HOLOSCAN_CORE_EXPORT expected<ClockKey, Error> graph_bind_shared_clock(
    const std::shared_ptr<GraphState>& state, std::string_view name, std::shared_ptr<Clock> clock);
HOLOSCAN_CORE_EXPORT expected<ClockKey, Error> graph_import_clock(
    const std::shared_ptr<GraphState>& state, std::string_view name);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_bind_clock_import(
    const std::shared_ptr<GraphState>& state, std::size_t embedded_graph_index,
    const void* handle_identity, ClockKey child_import, ClockKey parent_clock);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_set_default_clock(
    const std::shared_ptr<GraphState>& state, ClockKey clock);
HOLOSCAN_CORE_EXPORT expected<ClockKey, Error> graph_default_clock(
    const std::shared_ptr<GraphState>& state);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_add_clock_conversion(
    const std::shared_ptr<GraphState>& state, ClockKey from, ClockKey to,
    AffineClockConversion conversion);

HOLOSCAN_CORE_EXPORT std::shared_ptr<GraphState> make_graph_state(Graph* owner,
                                                                  std::string_view name);
HOLOSCAN_CORE_EXPORT expected<EmbeddedGraphRegistration, Error> graph_register_embedded(
    const std::shared_ptr<GraphState>& parent, std::string_view instance_name,
    std::unique_ptr<Graph> child, const std::shared_ptr<GraphState>& child_state,
    const std::shared_ptr<const GraphSnapshotData>& child_snapshot);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_prepare_embedded_type(
    const std::shared_ptr<GraphState>& parent, const std::shared_ptr<GraphState>& child,
    const void* child_type_token);
HOLOSCAN_CORE_EXPORT const Graph* graph_embedded_graph(const std::shared_ptr<GraphState>& state,
                                                       std::size_t embedded_graph_index,
                                                       const void* handle_identity) noexcept;
HOLOSCAN_CORE_EXPORT bool graph_embedded_valid(const std::shared_ptr<GraphState>& state,
                                               std::size_t embedded_graph_index,
                                               const void* handle_identity) noexcept;
HOLOSCAN_CORE_EXPORT std::optional<GraphInstancePath> graph_embedded_path(
    const std::shared_ptr<GraphState>& state, std::size_t embedded_graph_index,
    const void* handle_identity);
HOLOSCAN_CORE_EXPORT expected<std::vector<PortRecord>, Error> graph_embedded_interface_ports(
    const std::shared_ptr<GraphState>& state, std::size_t embedded_graph_index,
    const void* handle_identity);
HOLOSCAN_CORE_EXPORT expected<PortRecord, Error> graph_find_embedded_interface_port(
    const std::shared_ptr<GraphState>& state, std::size_t embedded_graph_index,
    const void* handle_identity, std::string_view name);
HOLOSCAN_CORE_EXPORT expected<ClockKey, Error> graph_find_embedded_clock_import(
    const std::shared_ptr<GraphState>& state, std::size_t embedded_graph_index,
    const void* handle_identity, std::string_view name);
HOLOSCAN_CORE_EXPORT std::shared_ptr<const GraphSnapshotData> graph_snapshot(
    const std::shared_ptr<GraphState>& state);
HOLOSCAN_CORE_EXPORT expected<TensorOutputPortConfigurator*, Error> graph_tensor_configurator(
    const std::shared_ptr<GraphState>& state, GraphPortId port);
/// @endcond

inline void throw_graph_error(Error&& error) {
  throw RuntimeError{std::move(error)};
}

}  // namespace detail

template <PublishedPayload T>
Output<T>& ImportedTopic<T>::output() const {
  auto state = state_.lock();
  const auto* declaration =
      state ? detail::graph_port_declaration(
                  state, GraphPortId{graph_port_}, PortDirection::kOutput, handle_identity_.get())
            : nullptr;
  if (declaration == nullptr) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_EXTERNAL_TOPIC_HANDLE_INVALID"}};
  }
  return *static_cast<Output<T>*>(const_cast<void*>(declaration));
}

template <PublishedPayload T>
bool ImportedTopic<T>::graph_valid() const noexcept {
  auto state = state_.lock();
  return state && graph_port_ != 0U &&
         detail::graph_port_declaration(
             state, GraphPortId{graph_port_}, PortDirection::kOutput, handle_identity_.get()) !=
             nullptr;
}

template <PublishedPayload T>
Input<T>& ExportedTopic<T>::input() const {
  auto state = state_.lock();
  const auto* declaration =
      state ? detail::graph_port_declaration(
                  state, GraphPortId{graph_port_}, PortDirection::kInput, handle_identity_.get())
            : nullptr;
  if (declaration == nullptr) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_EXTERNAL_TOPIC_HANDLE_INVALID"}};
  }
  return *static_cast<Input<T>*>(const_cast<void*>(declaration));
}

template <PublishedPayload T>
bool ExportedTopic<T>::graph_valid() const noexcept {
  auto state = state_.lock();
  return state && graph_port_ != 0U &&
         detail::graph_port_declaration(
             state, GraphPortId{graph_port_}, PortDirection::kInput, handle_identity_.get()) !=
             nullptr;
}

template <std::derived_from<OperatorBase> Op>
class OperatorPortProxy {
 public:
  /// @brief Opt this authoring handle out of replayable Graph captures.
  using graph_authoring_handle_tag = void;

  /// @brief Resolve const access to the current sealed authoring prototype.
  /// @return Non-null current prototype pointer.
  /// @throws RuntimeError with `GRAPH_HANDLE_INVALID` if the Graph or declaration is retired.
  /// @note Resolution occurs on every access so parameter reconstruction cannot
  /// leave a cached pointer to a retired prototype.
  [[nodiscard]] const Op* operator->() const {
    auto state = state_.lock();
    const auto* base =
        state ? detail::graph_operator_prototype(
                    state, index_, &detail::kGraphOperatorTypeToken<Op>, handle_identity_.get())
              : nullptr;
    if (base == nullptr) {
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"});
    }
    return static_cast<const Op*>(base);
  }

 private:
  OperatorPortProxy(std::weak_ptr<detail::GraphState> state, std::size_t index,
                    std::shared_ptr<const void> handle_identity) noexcept
      : state_(std::move(state)), index_(index), handle_identity_(std::move(handle_identity)) {}
  std::weak_ptr<detail::GraphState> state_;
  std::size_t index_{};
  std::shared_ptr<const void> handle_identity_;
  template <std::derived_from<OperatorBase>>
  friend class OperatorHandle;
};

/// @brief Hold copyable Graph declaration identity for one Operator type.
template <std::derived_from<OperatorBase> Op>
class OperatorHandle {
 public:
  /// @brief Opt this authoring handle out of replayable Graph captures.
  using graph_authoring_handle_tag = void;

  /// @brief Name the exact concrete Operator type represented by this handle.
  using Operator = Op;

  /// @brief Name the complete generation-zero parameter aggregate for \ref Operator.
  using Params = typename Op::Params;

  /// @brief Name the restricted dynamic prototype accessor used for typed port wiring.
  using PortAccess = OperatorPortProxy<Op>;

  /// @brief Return restricted const prototype access for typed port wiring.
  [[nodiscard]] PortAccess operator->() const noexcept {
    return PortAccess{state_, index_, handle_identity_};
  }

  /// @brief Replace the complete generation-zero parameter aggregate.
  /// @param params Valid candidate aggregate copied into replayable factory state.
  /// @throws RuntimeError if validation/reconstruction fails or this handle is stale.
  void set_params(Params params) {
    auto state = state_.lock();
    if (!state)
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"});
    auto result =
        detail::graph_set_operator_params(state,
                                          index_,
                                          &params,
                                          operator_type_token_,
                                          handle_identity_.get(),
                                          param_type_descriptor<Params>().local_type_token,
                                          param_schema_descriptor<Params>(),
                                          nullptr);
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
  }

  /// @brief Replace parameters and retain their immutable producer evidence.
  /// @param params Prepared value for this exact Operator parameter type.
  /// @throws RuntimeError if validation/reconstruction fails or this handle is stale.
  void set_params(PreparedParams<Params> params) {
    auto state = state_.lock();
    if (!state)
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"});
    auto evidence = detail::PreparedParamsAccess::evidence(params);
    Params value = std::move(params).take();
    auto result =
        detail::graph_set_operator_params(state,
                                          index_,
                                          &value,
                                          operator_type_token_,
                                          handle_identity_.get(),
                                          param_type_descriptor<Params>().local_type_token,
                                          param_schema_descriptor<Params>(),
                                          std::move(evidence));
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
  }

  /// @brief Return the owned local declaration name.
  [[nodiscard]] std::string_view name() const noexcept {
    auto state = state_.lock();
    return state ? detail::graph_operator_name(state, index_, handle_identity_.get())
                 : std::string_view{};
  }

  /// @brief Return whether the source Graph still owns the exact typed declaration.
  [[nodiscard]] bool graph_valid() const noexcept {
    auto state = state_.lock();
    return state && detail::graph_operator_valid(
                        state, index_, operator_type_token_, handle_identity_.get());
  }

  /// @brief Return whether this handle names a live Graph declaration.
  [[nodiscard]] explicit operator bool() const noexcept { return graph_valid(); }

  /// @brief Return the compiled operator reference when one exists.
  /// @return Invalid zero reference in HSDK5-006; HSDK5-009 binds plan identity.
  [[nodiscard]] OperatorRef ref() const noexcept {
    auto state = state_.lock();
    return state ? detail::graph_operator_ref(state, index_, handle_identity_.get())
                 : OperatorRef{};
  }

  /// @brief Return the compiled operator identity when one exists.
  /// @return Zero in HSDK5-006 because authoring handles never fabricate plan IDs.
  [[nodiscard]] OperatorInstanceId id() const noexcept { return ref().id; }

  /// @brief Return the immutable compiled-plan generation when one exists.
  /// @return Zero until HSDK5-009 freezes this declaration into a plan.
  [[nodiscard]] PlanGenerationId plan_generation() const noexcept { return ref().plan_generation; }

  /// @brief Return whether this handle is bound to an immutable compiled plan.
  /// @return Always false for the HSDK5-006 authoring-only handle.
  [[nodiscard]] bool plan_bound() const noexcept { return ref().plan_bound(); }

 private:
  OperatorHandle(const void* operator_type_token, std::weak_ptr<detail::GraphState> state,
                 std::size_t index, std::shared_ptr<const void> handle_identity) noexcept
      : operator_type_token_(operator_type_token),
        state_(std::move(state)),
        index_(index),
        handle_identity_(std::move(handle_identity)) {}
  const void* operator_type_token_{};
  std::weak_ptr<detail::GraphState> state_;
  std::size_t index_{};
  std::shared_ptr<const void> handle_identity_;
  friend class Graph;
  friend class RegionHandle;
  friend class PartitionHandle;
  friend class GraphLifecycleBuilder;
  friend class ExecutionPartitionsView;
  friend class detail::ParameterConfigBindingAccess;
  friend class authoring::OpaqueOperatorRef;
};

namespace detail {

/// @brief Expose call-scoped authoring ownership to optional Parameter tooling.
class ParameterConfigBindingAccess {
 public:
  [[nodiscard]] static const std::shared_ptr<GraphState>& state(Graph& graph) noexcept;

  template <std::derived_from<OperatorBase> Op>
  [[nodiscard]] static const std::weak_ptr<GraphState>& state(
      const OperatorHandle<Op>& handle) noexcept {
    return handle.state_;
  }

  template <std::derived_from<OperatorBase> Op>
  [[nodiscard]] static expected<GraphOperatorParameterTarget, Error> target(
      const OperatorHandle<Op>& handle) {
    auto state = handle.state_.lock();
    if (!state) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument, "PARAM_CONFIG_HANDLE_INVALID"});
    }
    return graph_operator_parameter_target(
        state, handle.index_, handle.operator_type_token_, handle.handle_identity_.get());
  }

  template <std::derived_from<OperatorBase> Op>
  [[nodiscard]] static expected<VertexId, Error> vertex(const OperatorHandle<Op>& handle) {
    auto resolved = target(handle);
    if (!resolved) {
      return make_unexpected(std::move(resolved.error()));
    }
    return resolved->vertex;
  }
};

}  // namespace detail

/// @brief Hold read-only typed identity for one independently owned embedded Graph.
///
/// The direct parent Graph owns the exact child object and its complete authoring
/// state. A handle never extends that lifetime, never permits child mutation, and
/// never aliases a separately embedded instance of the same reusable Graph type.
/// Its path is relative to that direct parent; canonical root-qualified instance
/// identity after any outer embedding is exposed by \ref GraphView.
///
/// @tparam G Concrete reusable Graph type.
template <typename G>
class GraphHandle {
 public:
  /// @brief Opt this authoring handle out of replayable Graph captures.
  using graph_authoring_handle_tag = void;

  static_assert(std::derived_from<G, Graph>, "GraphHandle requires a Graph-derived type");
  /// @brief Name the exact reusable Graph type represented by this handle.
  using GraphType = G;

  /// @brief Resolve read-only access to the embedded Graph's public interface members.
  /// @return Non-null exact child instance.
  /// @throws RuntimeError with `GRAPH_HANDLE_INVALID` if the parent or embedded declaration is
  /// retired.
  /// @note The returned pointer grants no Graph authoring mutation authority.
  [[nodiscard]] const G* operator->() const {
    auto state = state_.lock();
    const auto* child = state ? static_cast<const G*>(detail::graph_embedded_graph(
                                    state, index_, handle_identity_.get()))
                              : nullptr;
    if (child == nullptr) {
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"});
    }
    return child;
  }

  /// @brief Return an owning structured path relative to the direct parent.
  /// @return Parent-relative path while the parent owns this registration, or
  /// no value after parent retirement or if the registration is invalid.
  /// @throws std::bad_alloc if allocating the owning path copy fails.
  /// @note A containing GraphView supplies the root-qualified path after this
  /// Graph is itself embedded in an outer Graph.
  [[nodiscard]] std::optional<GraphInstancePath> path() const {
    auto state = state_.lock();
    return state ? detail::graph_embedded_path(state, index_, handle_identity_.get())
                 : std::nullopt;
  }

  /// @brief Return whether the parent still owns this exact typed child instance.
  /// @return True only while the parent state and typed registration remain live.
  [[nodiscard]] bool graph_valid() const noexcept {
    auto state = state_.lock();
    return state && detail::graph_embedded_valid(state, index_, handle_identity_.get());
  }

  /// @brief Return whether this handle names a live embedded Graph.
  [[nodiscard]] explicit operator bool() const noexcept { return graph_valid(); }

 private:
  GraphHandle(std::weak_ptr<detail::GraphState> state, std::size_t index,
              std::shared_ptr<const void> handle_identity)
      : state_(std::move(state)), index_(index), handle_identity_(std::move(handle_identity)) {}
  std::weak_ptr<detail::GraphState> state_;
  std::size_t index_{};
  std::shared_ptr<const void> handle_identity_;
  friend class Graph;
  friend class RegionHandle;
  friend class PartitionHandle;
  friend class authoring::OpaqueGraphRef;
};

/// @brief Hold one semantically compatible typed Graph edge declaration.
/// @tparam SourceT Exact C++ source payload type.
/// @tparam TargetT Exact C++ target payload type; defaults to \c SourceT.
///
/// Different endpoint types are admitted only when both declare explicit raw-POD
/// contracts and their complete canonical name, version, object layout, and field
/// records compare equal. Every other payload family remains exact-C++-type only.
template <PublishedPayload SourceT, PublishedPayload TargetT = SourceT>
class FlowHandle {
 public:
  /// @brief Opt this authoring handle out of replayable Graph captures.
  using graph_authoring_handle_tag = void;

  /// @brief Return the snapshot-local authored edge identity.
  /// @return Nonzero edge identity while the source Graph exists.
  [[nodiscard]] EdgeId id() const noexcept { return id_; }

  /// @brief Resolve the current source declaration.
  /// @return Current output member after any atomic parameter reconstruction.
  /// @throws RuntimeError if the Graph retired or the structural declaration changed.
  [[nodiscard]] const Output<SourceT>& source() const {
    auto state = state_.lock();
    if (!state || !detail::graph_flow_valid(state, id_, source_, target_, handle_identity_.get())) {
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_FLOW_HANDLE_INVALID"});
    }
    const void* declaration = detail::graph_port_declaration(
        state, source_, PortDirection::kOutput, detail::operator_spec_payload_identity<SourceT>());
    if (declaration == nullptr) {
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_FLOW_SOURCE_INVALID"});
    }
    return *static_cast<const Output<SourceT>*>(declaration);
  }

  /// @brief Resolve the current target declaration.
  /// @return Current input member after any atomic parameter reconstruction.
  /// @throws RuntimeError if the Graph retired or the structural declaration changed.
  [[nodiscard]] const Input<TargetT>& target() const {
    auto state = state_.lock();
    if (!state || !detail::graph_flow_valid(state, id_, source_, target_, handle_identity_.get())) {
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_FLOW_HANDLE_INVALID"});
    }
    const void* declaration = detail::graph_port_declaration(
        state, target_, PortDirection::kInput, detail::operator_spec_payload_identity<TargetT>());
    if (declaration == nullptr) {
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_FLOW_TARGET_INVALID"});
    }
    return *static_cast<const Input<TargetT>*>(declaration);
  }

  /// @brief Return whether this edge remains owned by its source Graph.
  /// @return True while the Graph and nonzero authoring identity remain live.
  [[nodiscard]] bool graph_valid() const noexcept {
    auto state = state_.lock();
    return state && detail::graph_flow_valid(state, id_, source_, target_, handle_identity_.get());
  }

 private:
  FlowHandle(std::weak_ptr<detail::GraphState> state, EdgeId id, GraphPortId source,
             GraphPortId target, std::shared_ptr<const void> handle_identity) noexcept
      : state_(std::move(state)),
        id_(id),
        source_(source),
        target_(target),
        handle_identity_(std::move(handle_identity)) {}
  std::weak_ptr<detail::GraphState> state_;
  EdgeId id_{};
  GraphPortId source_{};
  GraphPortId target_{};
  std::shared_ptr<const void> handle_identity_;
  friend class Graph;
  friend class detail::OperatorTestingAccess;
};

/// @brief Configure cold authoring policy for one Graph-owned input port.
///
/// A configurator is a non-owning handle. It remains usable only while the
/// originating Graph and exact port declaration remain live. Each successful
/// call atomically updates cold authoring state; a rejected call leaves the
/// port and its incident edges unchanged. Configurators do not make Graph
/// authoring thread-safe, so callers must serialize them with all other reads
/// and mutations of the same Graph.
///
/// @tparam T Exact published payload type of the input declaration.
template <PublishedPayload T>
class InputPortConfigurator {
 public:
  /// @brief Opt this authoring handle out of replayable Graph captures.
  using graph_authoring_handle_tag = void;

  /// @brief Override the finite input history depth.
  /// @param depth Positive number of samples retained for this logical input.
  /// @return This configurator for fluent authoring.
  /// @throws RuntimeError if \p depth is zero or the source Graph is unavailable.
  InputPortConfigurator& queue_depth(std::uint32_t depth) {
    apply(detail::graph_configure_input_queue(state(), port_, handle_identity_.get(), depth));
    return *this;
  }

  /// @brief Override this input with one common consumer Tensor spec.
  /// @param spec Representation, finite bounds, and conversion requirements.
  /// @return This configurator for fluent authoring.
  InputPortConfigurator& expects_tensor(const TensorInputSpec& spec) {
    apply(detail::graph_configure_input_tensor_spec(state(), port_, handle_identity_.get(), spec));
    return *this;
  }

  /// @brief Assign an owned diagnostic alias without changing port identity.
  /// @param alias Nonempty bounded UTF-8 alias copied before return.
  /// @return This configurator for fluent authoring.
  /// @throws RuntimeError if \p alias is invalid or the source Graph is unavailable.
  InputPortConfigurator& diagnostic_alias(std::string_view alias) {
    apply(detail::graph_configure_port_alias(state(), port_, handle_identity_.get(), alias));
    return *this;
  }

 private:
  InputPortConfigurator(std::weak_ptr<detail::GraphState> state, GraphPortId port,
                        std::shared_ptr<const void> handle_identity) noexcept
      : state_(std::move(state)), port_(port), handle_identity_(std::move(handle_identity)) {}
  [[nodiscard]] std::shared_ptr<detail::GraphState> state() const {
    auto value = state_.lock();
    if (!value)
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"});
    return value;
  }
  static void apply(expected<void, Error> result) {
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
  }
  std::weak_ptr<detail::GraphState> state_;
  GraphPortId port_{};
  std::shared_ptr<const void> handle_identity_;
  friend class Graph;
};

/// @brief Configure cold authoring policy for one Graph-owned output port.
///
/// A configurator is a non-owning handle. It remains usable only while the
/// originating Graph and exact port declaration remain live. Each successful
/// call atomically updates cold authoring state; a rejected call leaves the
/// port and its incident edges unchanged. Configurators do not make Graph
/// authoring thread-safe, so callers must serialize them with all other reads
/// and mutations of the same Graph.
///
/// @tparam T Exact published payload type of the output declaration.
template <PublishedPayload T>
class OutputPortConfigurator {
 public:
  /// @brief Opt this authoring handle out of replayable Graph captures.
  using graph_authoring_handle_tag = void;

  /// @brief Override the positive per-compute emission bound.
  /// @param count Positive maximum accepted emissions per compute call.
  /// @return This configurator for fluent authoring.
  /// @throws RuntimeError if \p count is zero or the source Graph is unavailable.
  OutputPortConfigurator& max_emits_per_compute(std::uint32_t count = 1) {
    apply(detail::graph_configure_output_emits(state(), port_, handle_identity_.get(), count));
    return *this;
  }

  /// @brief Override the finite late-joiner history depth.
  /// @param depth Positive retained-sample depth.
  /// @return This configurator for fluent authoring.
  /// @throws RuntimeError if \p depth is zero or the source Graph is unavailable.
  OutputPortConfigurator& late_joiner_history_depth(std::uint32_t depth) {
    apply(detail::graph_configure_output_history(state(), port_, handle_identity_.get(), depth));
    return *this;
  }

  /// @brief Override this output with one Tensor spec.
  /// @param spec Representation, finite bounds, and transfer promises.
  /// @return This configurator for fluent authoring.
  /// @note kPassThrough does not remove Runtime-pool storage declared during Operator setup;
  /// Graph overrides may tighten setup resource declarations but do not retract them.
  OutputPortConfigurator& produces_tensor(const TensorOutputSpec& spec) {
    apply(detail::graph_configure_output_tensor_spec(state(), port_, handle_identity_.get(), spec));
    return *this;
  }

  /// @brief Override schema identity and payload options for this output.
  /// @param options Options copied before return.
  /// @return This configurator for fluent authoring.
  OutputPortConfigurator& payload_options(PayloadOptions options) {
    apply(detail::graph_configure_output_payload(state(), port_, handle_identity_.get(), options));
    return *this;
  }

  /// @brief Assign an owned diagnostic alias without changing port identity.
  /// @param alias Nonempty bounded UTF-8 alias copied before return.
  /// @return This configurator for fluent authoring.
  /// @throws RuntimeError if \p alias is invalid or the source Graph is unavailable.
  OutputPortConfigurator& diagnostic_alias(std::string_view alias) {
    apply(detail::graph_configure_port_alias(state(), port_, handle_identity_.get(), alias));
    return *this;
  }

 private:
  OutputPortConfigurator(std::weak_ptr<detail::GraphState> state, GraphPortId port,
                         std::shared_ptr<const void> handle_identity) noexcept
      : state_(std::move(state)), port_(port), handle_identity_(std::move(handle_identity)) {}
  [[nodiscard]] std::shared_ptr<detail::GraphState> state() const {
    auto value = state_.lock();
    if (!value)
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"});
    return value;
  }
  static void apply(expected<void, Error> result) {
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
  }
  std::weak_ptr<detail::GraphState> state_;
  GraphPortId port_{};
  std::shared_ptr<const void> handle_identity_;
  friend class Graph;
};

/// @brief Configure a direct Tensor output while preserving specialized fluent type.
class HOLOSCAN_CORE_EXPORT TensorOutputPortConfigurator {
 public:
  /// @brief Opt this authoring handle out of replayable Graph captures.
  using graph_authoring_handle_tag = void;

  /// @brief Override the positive per-compute emission bound.
  /// @param count Positive maximum accepted emissions per compute call.
  /// @return This stable Graph-owned configurator.
  /// @throws RuntimeError if \p count is zero or the source Graph is unavailable.
  TensorOutputPortConfigurator& max_emits_per_compute(std::uint32_t count = 1);

  /// @brief Override the finite late-joiner history depth.
  /// @param depth Positive retained-sample depth.
  /// @return This stable Graph-owned configurator.
  /// @throws RuntimeError if \p depth is zero or the source Graph is unavailable.
  TensorOutputPortConfigurator& late_joiner_history_depth(std::uint32_t depth);

  /// @brief Override this output with one common producer Tensor spec.
  /// @param spec Representation, finite bounds, storage intent, and transfer policy.
  /// @return This stable Graph-owned configurator.
  TensorOutputPortConfigurator& produces_tensor(const TensorOutputSpec& spec);

  /// @brief Override schema identity and payload options for this output.
  /// @param options Options copied before return.
  /// @return This stable Graph-owned configurator.
  TensorOutputPortConfigurator& payload_options(PayloadOptions options);

  /// @brief Assign an owned diagnostic alias without changing port identity.
  /// @param alias Nonempty bounded UTF-8 alias copied before return.
  /// @return This stable Graph-owned configurator.
  /// @throws RuntimeError if \p alias is invalid or the source Graph is unavailable.
  TensorOutputPortConfigurator& diagnostic_alias(std::string_view alias);

 private:
  TensorOutputPortConfigurator(std::weak_ptr<detail::GraphState> state, GraphPortId port,
                               std::shared_ptr<const void> handle_identity) noexcept
      : state_(std::move(state)), port_(port), handle_identity_(std::move(handle_identity)) {}
  [[nodiscard]] std::shared_ptr<detail::GraphState> state() const;
  static void apply(expected<void, Error> result);
  std::weak_ptr<detail::GraphState> state_;
  GraphPortId port_{};
  std::shared_ptr<const void> handle_identity_;
  friend class Graph;
  friend class detail::GraphState;
  friend expected<TensorOutputPortConfigurator*, Error> detail::graph_tensor_configurator(
      const std::shared_ptr<detail::GraphState>& state, GraphPortId port);
};

/// @brief Configure cold queue and transport policy for one Graph-owned flow.
///
/// A configurator is a non-owning handle to an already-authored edge. It
/// remains usable only while the originating Graph and edge remain live. Each
/// successful call atomically updates the edge; rejected calls preserve the
/// previous declaration. Configurators do not make Graph authoring thread-safe,
/// so callers must serialize them with all other reads and mutations of the
/// same Graph.
///
/// @tparam SourceT Exact admitted source payload type.
/// @tparam TargetT Exact admitted target payload type. A distinct type is
/// admitted only through an equal explicit raw-POD contract.
template <PublishedPayload SourceT, PublishedPayload TargetT = SourceT>
class FlowConfigurator {
 public:
  /// @brief Opt this authoring handle out of replayable Graph captures.
  using graph_authoring_handle_tag = void;

  /// @brief Override the authored finite-queue behavior.
  /// @param policy Closed queue policy.
  /// @return This configurator for fluent authoring.
  /// @throws RuntimeError if \p policy is invalid or the flow is unavailable.
  FlowConfigurator& queue_policy(QueuePolicy policy) {
    apply(detail::graph_configure_flow_policy(state(), edge_, handle_identity_.get(), policy));
    return *this;
  }
  /// @brief Override the positive destination queue depth.
  /// @param depth Positive queue depth.
  /// @return This configurator for fluent authoring.
  /// @throws RuntimeError if \p depth is zero or the flow is unavailable.
  FlowConfigurator& queue_depth(std::uint32_t depth) {
    apply(detail::graph_configure_flow_depth(state(), edge_, handle_identity_.get(), depth));
    return *this;
  }
  /// @brief Override the owned logical flow alias.
  /// @param name Nonempty bounded UTF-8 alias copied before return.
  /// @return This configurator for fluent authoring.
  /// @throws RuntimeError if \p name is invalid or the flow is unavailable.
  FlowConfigurator& topic_name(std::string_view name) {
    apply(detail::graph_configure_flow_topic(state(), edge_, handle_identity_.get(), name));
    return *this;
  }
  /// @brief Override the Tensor memory-conversion request.
  /// @param policy Closed conversion policy.
  /// @return This configurator for fluent authoring.
  /// @throws RuntimeError if \p policy is invalid, the flow is unavailable, or
  /// neither endpoint has direct or setup-declared Tensor semantics.
  FlowConfigurator& memory_conversion(MemoryConversionPolicy policy) {
    apply(detail::graph_configure_flow_memory(state(), edge_, handle_identity_.get(), policy));
    return *this;
  }
  /// @brief Override the positive in-flight retention bound.
  /// @param count Positive maximum in-flight sample count.
  /// @return This configurator for fluent authoring.
  /// @throws RuntimeError if \p count is zero or the flow is unavailable.
  FlowConfigurator& max_inflight_samples(std::uint32_t count) {
    apply(detail::graph_configure_flow_inflight(state(), edge_, handle_identity_.get(), count));
    return *this;
  }

 private:
  FlowConfigurator(std::weak_ptr<detail::GraphState> state, EdgeId edge,
                   std::shared_ptr<const void> handle_identity) noexcept
      : state_(std::move(state)), edge_(edge), handle_identity_(std::move(handle_identity)) {}
  [[nodiscard]] std::shared_ptr<detail::GraphState> state() const {
    auto value = state_.lock();
    if (!value)
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"});
    return value;
  }
  static void apply(expected<void, Error> result) {
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
  }
  std::weak_ptr<detail::GraphState> state_;
  EdgeId edge_{};
  std::shared_ptr<const void> handle_identity_;
  friend class Graph;
};

/// @brief Author one named logical region and its cold placement constraints.
///
/// This non-owning handle refers to one region in the originating Graph. It
/// remains usable only while that Graph and declaration remain live. Region
/// mutation is atomic on failure but is not thread-safe with any other read or
/// mutation of the same Graph.
class HOLOSCAN_CORE_EXPORT RegionHandle {
 public:
  /// @brief Opt this authoring handle out of replayable Graph captures.
  using graph_authoring_handle_tag = void;

  /// @brief Add one logical Operator to this region.
  /// @tparam Op Exact Operator type.
  /// @param op Graph-owned Operator declaration.
  /// @return This handle for fluent authoring.
  /// @throws RuntimeError if the handle is foreign, stale, or already a member.
  template <std::derived_from<OperatorBase> Op>
  RegionHandle& add(const OperatorHandle<Op>& op) {
    auto state = state_.lock();
    auto op_state = op.state_.lock();
    if (!state || state != op_state || !op.graph_valid()) {
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_REGION_HANDLE_INVALID"});
    }
    auto result = detail::graph_region_add_operator(state, id_, handle_identity_.get(), op.index_);
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return *this;
  }

  /// @brief Add every flattened Operator from one embedded Graph to this region.
  /// @tparam G Exact reusable Graph type.
  /// @param graph Parent-owned embedded Graph instance.
  /// @return This handle for fluent authoring.
  /// @throws RuntimeError if the handle is foreign, stale, or already a member.
  /// @note A region containing an embedded Graph cannot use
  /// \ref RegionConstraint::cuda_graph(). CUDA Graph regions are currently limited to directly
  /// listed operators in the root Graph.
  template <typename G>
    requires std::derived_from<G, Graph>
  RegionHandle& add(const GraphHandle<G>& graph) {
    auto state = state_.lock();
    if (!state || state != graph.state_.lock() || !graph.graph_valid()) {
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_REGION_HANDLE_INVALID"});
    }
    auto result = detail::graph_region_add_embedded_graph(
        state, id_, handle_identity_.get(), graph.index_, graph.handle_identity_.get());
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return *this;
  }
  /// @brief Add one unique backend-neutral region constraint.
  /// @param constraint Valid constraint not already present by kind.
  /// @return This handle for fluent authoring.
  /// @throws RuntimeError if the constraint is invalid or duplicated.
  RegionHandle& constrain(RegionConstraint constraint);

  /// @brief Set this region's capability profile request.
  /// @param profile Closed profile value.
  /// @return This handle for fluent authoring.
  /// @throws RuntimeError if \p profile is invalid or already set.
  RegionHandle& profile(Profile profile);

  /// @brief Return this region's snapshot-local authoring identity.
  /// @return Nonzero region identity while the declaration is live.
  [[nodiscard]] RegionId id() const noexcept { return id_; }

 private:
  RegionHandle(std::weak_ptr<detail::GraphState> state, RegionId id,
               std::shared_ptr<const void> handle_identity) noexcept
      : state_(std::move(state)), id_(id), handle_identity_(std::move(handle_identity)) {}
  std::weak_ptr<detail::GraphState> state_;
  RegionId id_{};
  std::shared_ptr<const void> handle_identity_;
  friend class Graph;
  friend class authoring::OpaqueOperatorRef;
};

/// @brief Author one developer-preview manual execution partition.
///
/// This non-owning handle refers to one partition declaration in the
/// originating Graph. Explicit membership accepts direct Operator declarations
/// and complete embedded instances from that Graph. The compile-root \ref include_all directive
/// instead expands to all composed Operators. A compiler validates the final
/// cover according to the Graph's exact manual or automatic-remainder intent.
///
/// The handle remains usable only while the originating Graph and declaration
/// remain live. Mutation is atomic on failure but is not thread-safe with any
/// other read or mutation of the same Graph.
class HOLOSCAN_CORE_EXPORT PartitionHandle {
 public:
  /// @brief Opt this authoring handle out of replayable Graph captures.
  using graph_authoring_handle_tag = void;

  /// @brief Add one direct logical Operator to this partition.
  /// @tparam Op Exact Operator type.
  /// @param op Graph-owned direct Operator declaration.
  /// @return This handle for fluent authoring.
  /// @throws RuntimeError if either handle is foreign or stale, or if the
  /// Operator is already assigned to any manual partition in this Graph.
  template <std::derived_from<OperatorBase> Op>
  PartitionHandle& add(const OperatorHandle<Op>& op) {
    auto state = state_.lock();
    auto op_state = op.state_.lock();
    if (!state || !op_state || !op.graph_valid()) {
      detail::throw_graph_error(
          Error{ErrorCode::kInvalidArgument, "GRAPH_PARTITION_HANDLE_INVALID"});
    }
    if (state != op_state) {
      detail::throw_graph_error(
          Error{ErrorCode::kInvalidArgument, "GRAPH_PARTITION_MEMBER_FOREIGN"});
    }
    auto result =
        detail::graph_partition_add_operator(state, id_, handle_identity_.get(), op.index_);
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return *this;
  }

  /// @brief Add a complete batch of direct logical Operators to this partition.
  /// @tparam Ops Exact Operator types.
  /// @param operators Graph-owned direct Operator declarations.
  /// @return This handle for fluent authoring.
  /// @throws RuntimeError if a handle is foreign or stale, if any Operator is
  /// duplicated within the batch or already assigned, or if this declaration
  /// uses \ref include_all. A rejected batch leaves membership unchanged.
  template <std::derived_from<OperatorBase>... Ops>
    requires(sizeof...(Ops) > 1U)
  PartitionHandle& add(const OperatorHandle<Ops>&... operators) {
    auto state = state_.lock();
    if (!state) {
      detail::throw_graph_error(
          Error{ErrorCode::kInvalidArgument, "GRAPH_PARTITION_HANDLE_INVALID"});
    }

    std::array<std::size_t, sizeof...(Ops)> indexes{};
    std::size_t index{};
    bool invalid{};
    bool foreign{};
    const auto collect = [&](const auto& op) {
      auto op_state = op.state_.lock();
      if (!op_state || !op.graph_valid()) {
        invalid = true;
      } else if (state != op_state) {
        foreign = true;
      }
      indexes[index++] = op.index_;
    };
    (collect(operators), ...);
    if (invalid) {
      detail::throw_graph_error(
          Error{ErrorCode::kInvalidArgument, "GRAPH_PARTITION_HANDLE_INVALID"});
    }
    if (foreign) {
      detail::throw_graph_error(
          Error{ErrorCode::kInvalidArgument, "GRAPH_PARTITION_MEMBER_FOREIGN"});
    }

    auto result =
        detail::graph_partition_add_operators(state, id_, handle_identity_.get(), indexes);
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return *this;
  }

  /// @brief Assign one embedded Graph instance and all descendant Operators atomically.
  /// @tparam G Exact embedded Graph type.
  /// @param graph Live direct-child instance owned by this partition's Graph.
  /// @return This handle for fluent authoring with other instances or direct Operators.
  /// @throws RuntimeError if either handle is stale or foreign, the instance is already
  /// assigned, the subtree has no Operators or declares any internal partition (even empty),
  /// or this partition uses \ref include_all. Rejection leaves Graph identity and membership
  /// unchanged. Compilation preserves the exact Operator set and validates placement constraints.
  /// @note Independent sources in the selected subtree share one serial partition owner.
  template <std::derived_from<Graph> G>
  PartitionHandle& add(const GraphHandle<G>& graph) {
    auto state = state_.lock();
    auto graph_state = graph.state_.lock();
    if (!state || !graph_state) {
      detail::throw_graph_error(
          Error{ErrorCode::kInvalidArgument, "GRAPH_PARTITION_HANDLE_INVALID"});
    }
    if (state != graph_state) {
      detail::throw_graph_error(
          Error{ErrorCode::kInvalidArgument, "GRAPH_PARTITION_MEMBER_FOREIGN"});
    }
    auto result = detail::graph_partition_add_embedded_graph(
        state, id_, handle_identity_.get(), graph.index_, graph.handle_identity_.get());
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return *this;
  }

  /// @brief Expand this partition to the complete compile-root application.
  /// @return This handle for fluent authoring.
  /// @throws RuntimeError if the declaration has explicit members, another
  /// partition exists, automatic remainder authoring was requested, or the
  /// declaration is later embedded below the compile root.
  /// @note Expansion occurs from the final composed snapshot at compile time,
  /// so Operators and embedded Graphs authored after this call are included.
  PartitionHandle& include_all();

 private:
  PartitionHandle(std::weak_ptr<detail::GraphState> state, PartitionId id,
                  std::shared_ptr<const void> handle_identity) noexcept
      : state_(std::move(state)), id_(id), handle_identity_(std::move(handle_identity)) {}
  std::weak_ptr<detail::GraphState> state_;
  PartitionId id_{};
  std::shared_ptr<const void> handle_identity_;
  friend class Graph;
  friend class authoring::OpaqueOperatorRef;
  friend class detail::OperatorTestingAccess;
};

/// @brief Author lifecycle ordering and restart permissions for one Graph.
///
/// The builder is a non-owning view of the originating Graph's cold lifecycle
/// declaration. It remains usable only while that Graph remains live. Each
/// accepted call records one dependency or permission and each rejected call
/// leaves the declaration unchanged. The builder does not make Graph authoring
/// thread-safe; callers must serialize it with all other Graph access.
class HOLOSCAN_CORE_EXPORT GraphLifecycleBuilder {
 public:
  /// @brief Opt this authoring handle out of replayable Graph captures.
  using graph_authoring_handle_tag = void;

  /// @brief Order two logical checkpoints within one lifecycle stage.
  /// @tparam BeforeT Predecessor Operator type.
  /// @tparam AfterT Successor Operator type.
  /// @param before Operator whose checkpoint must complete first.
  /// @param after Operator released after \p before.
  /// @param stage Exact same-stage ordering scope.
  /// @return This builder for fluent authoring.
  /// @throws RuntimeError for foreign/stale handles, duplicates, cycles, or invalid stages.
  /// @note The declaring Graph scope may order Operators in that scope or any descendant scope.
  /// Both endpoints must be descendants of the declaring scope; sibling Graphs cannot author an
  /// edge over one another's Operators.
  template <std::derived_from<OperatorBase> BeforeT, std::derived_from<OperatorBase> AfterT>
  GraphLifecycleBuilder& before(const OperatorHandle<BeforeT>& before,
                                const OperatorHandle<AfterT>& after, LifecycleStage stage) {
    auto state = state_.lock();
    auto before_state = before.state_.lock();
    auto after_state = after.state_.lock();
    if (!state || !before_state || !after_state || !before.graph_valid() || !after.graph_valid()) {
      detail::throw_graph_error(
          Error{ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_HANDLE_INVALID"});
    }
    auto result = detail::graph_lifecycle_before_resolved(
        state, before_state, before.index_, after_state, after.index_, stage);
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return *this;
  }

  /// @brief Authorize one Operator to request its own lifecycle restart.
  /// @tparam TargetT Target Operator type.
  /// @param target Exact restart target.
  /// @return This builder for fluent authoring.
  /// @throws RuntimeError if \p target is foreign/stale or permission is duplicated.
  template <std::derived_from<OperatorBase> TargetT>
  GraphLifecycleBuilder& allow_self_restart(const OperatorHandle<TargetT>& target) {
    return permission(LifecycleRestartOrigin::kSelf, std::nullopt, target);
  }

  /// @brief Authorize external run control to restart one Operator.
  /// @tparam TargetT Target Operator type.
  /// @param target Exact restart target.
  /// @return This builder for fluent authoring.
  /// @throws RuntimeError if \p target is foreign/stale or permission is duplicated.
  template <std::derived_from<OperatorBase> TargetT>
  GraphLifecycleBuilder& allow_external_restart(const OperatorHandle<TargetT>& target) {
    return permission(LifecycleRestartOrigin::kExternal, std::nullopt, target);
  }

  /// @brief Authorize one exact Operator to request restart of another.
  /// @tparam RequesterT Requesting Operator type.
  /// @tparam TargetT Target Operator type.
  /// @param requester Exact authorized requester.
  /// @param target Exact restart target.
  /// @return This builder for fluent authoring.
  /// @throws RuntimeError for foreign/stale handles or duplicate permission.
  template <std::derived_from<OperatorBase> RequesterT, std::derived_from<OperatorBase> TargetT>
  GraphLifecycleBuilder& allow_restart_request(const OperatorHandle<RequesterT>& requester,
                                               const OperatorHandle<TargetT>& target) {
    auto state = state_.lock();
    auto requester_state = requester.state_.lock();
    auto target_state = target.state_.lock();
    if (!state || !requester_state || !target_state || !requester.graph_valid() ||
        !target.graph_valid()) {
      detail::throw_graph_error(
          Error{ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_HANDLE_INVALID"});
    }
    auto result =
        detail::graph_lifecycle_permission_resolved(state,
                                                    LifecycleRestartOrigin::kOperatorRequest,
                                                    requester_state,
                                                    requester.index_,
                                                    target_state,
                                                    target.index_);
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return *this;
  }

 private:
  explicit GraphLifecycleBuilder(std::weak_ptr<detail::GraphState> state) noexcept
      : state_(std::move(state)) {}
  template <std::derived_from<OperatorBase> TargetT>
  GraphLifecycleBuilder& permission(LifecycleRestartOrigin origin,
                                    std::optional<std::size_t> requester,
                                    const OperatorHandle<TargetT>& target) {
    auto state = state_.lock();
    auto target_state = target.state_.lock();
    if (!state || !target_state || !target.graph_valid()) {
      detail::throw_graph_error(
          Error{ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_HANDLE_INVALID"});
    }
    auto result = detail::graph_lifecycle_permission_resolved(
        state, origin, state, requester, target_state, target.index_);
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return *this;
  }
  std::weak_ptr<detail::GraphState> state_;
  friend class Graph;
  friend class authoring::OpaqueOperatorRef;
};

namespace detail {

template <typename T>
concept GraphAuthoringHandle =
    requires { typename std::remove_cvref_t<T>::graph_authoring_handle_tag; };

template <typename T>
concept ShallowCapture =
    (!std::is_pointer_v<std::remove_cvref_t<T>> ||
     std::is_function_v<std::remove_pointer_t<std::remove_cvref_t<T>>>) &&
    !IsSpan<std::remove_cvref_t<T>>::value && !IsReferenceWrapper<std::remove_cvref_t<T>>::value &&
    !IsInitializerList<std::remove_cvref_t<T>>::value && !GraphAuthoringHandle<T> &&
    (!std::ranges::view<std::remove_cvref_t<T>> || kStringCapture<T>) &&
    (!(std::ranges::range<std::remove_cvref_t<T>> &&
       std::ranges::borrowed_range<std::remove_cvref_t<T>>) ||
     kStringCapture<T>);

template <typename T>
concept ReplayableCapture = ShallowCapture<T> && std::copy_constructible<NormalizedGraphCapture<T>>;

}  // namespace detail

/// @brief Own an editable cold graph declaration and replayable construction factories.
///
/// Graph is the only HSDK 5 root authoring abstraction. It performs lazy
/// composition and cold validation but does not compile or execute a graph.
/// The lazy-composition gate is internally synchronized, so concurrent observers
/// share one composer and receive its terminal result. This synchronization does
/// not make general authoring thread-safe: after composition, callers must
/// serialize mutation with every other mutation or observation of the same
/// Graph. Handles do not extend Graph lifetime and become invalid when the Graph
/// is destroyed.
///
/// @warning The complete most-derived Graph object must remain alive until all
/// authoring, observation, and lazy-composition calls have returned. Destruction
/// must not race any Graph access; handle invalidation does not provide concurrent
/// destruction synchronization.
///
/// Every successful public compound mutation is atomic. If allocation or an
/// injected exceptional boundary interrupts an in-progress mutation, Graph
/// restores the prior snapshot identity, generation, IDs, ownership claims,
/// and declaration indexes before the exception escapes. A rejected external
/// topic declaration is the deliberate exception to counter rollback: ordinary
/// typed validation failures consume their mixed import/export declaration
/// ordinal, while exceptional failures restore the reserved ordinal.
class HOLOSCAN_CORE_EXPORT Graph {
 public:
  /// @brief Create an empty named Graph authoring root.
  /// @param name Optional bounded UTF-8 name copied before return.
  /// @throws RuntimeError if \p name is invalid.
  explicit Graph(std::string_view name = "");

  /// @brief Destroy Graph-owned prototypes, factories, and authoring state.
  /// @warning Destruction must not overlap any authoring, observation, composition,
  /// or handle operation associated with this Graph.
  virtual ~Graph();

  /// @brief Reject copying because authoring identities and ownership claims are instance-local.
  Graph(const Graph&) = delete;

  /// @brief Reject copy assignment because Graph-owned declarations cannot change roots.
  Graph& operator=(const Graph&) = delete;

  /// @brief Reject moving because handles retain the source Graph's stable instance identity.
  Graph(Graph&&) = delete;

  /// @brief Reject move assignment because existing handles cannot be rebound to another root.
  Graph& operator=(Graph&&) = delete;

  /// @brief Add one typed Operator declaration and replayable factory.
  /// @tparam Op Concrete Operator type.
  /// @tparam Args Constructor capture argument types.
  /// @param name Nonempty unique bounded UTF-8 declaration name.
  /// @param args Constructor values normalized and copied into replayable state.
  /// @return Copyable authoring handle that dynamically resolves current ports.
  /// @throws RuntimeError if construction, setup, naming, or registration fails.
  /// @note If this is the first external authoring operation on a derived Graph, Core
  /// completes or terminally fails `compose()` before constructing the requested
  /// Operator factory or prototype.
  /// @pre Each normalized capture must be copy-constructible. Copies follow the
  /// captured type's ordinary C++ semantics, including shared identity for
  /// `std::shared_ptr` and non-owning observation for `std::weak_ptr`.
  /// @note Obvious top-level borrowed forms, move-only values, and Graph authoring
  /// handles are rejected at compile time. Function pointers and copyable callables
  /// are accepted. String views and string literals normalize to owned strings.
  /// Checks are intentionally shallow: nested container elements, user-defined
  /// fields, and callable captures remain the application's lifetime responsibility.
  /// @warning Do not call Graph authoring verbs from a derived Graph constructor;
  /// author the reusable body only in the Core-managed \ref compose override.
  /// @pre Any legacy \c Operator::start or \c Operator::stop hook must be public, unambiguous,
  /// and have the exact zero-argument \c void signature. Any advanced lifecycle member callback
  /// registered by \c setup() must be declared by \p Op itself rather than by a base or unrelated
  /// class.
  template <std::derived_from<OperatorBase> Op, typename... Args>
    requires((detail::ReplayableCapture<Args> && ...) &&
             std::constructible_from<Op, const detail::NormalizedGraphCapture<Args>&...>)
  OperatorHandle<Op> op(std::string_view name, Args&&... args) {
    if constexpr (!detail::SelectableOperatorLifecycleHooks<Op>) {
      static_assert(detail::SelectableOperatorLifecycleHooks<Op>,
                    "Operator lifecycle hooks must be public, unambiguous, and have the exact "
                    "void() signature");
    } else {
      auto prepared = detail::graph_prepare_authoring(state_);
      if (!prepared)
        detail::throw_graph_error(std::move(prepared.error()));
      using Factory = detail::OperatorFactoryModel<Op, detail::NormalizedGraphCapture<Args>...>;
      auto factory =
          std::make_shared<Factory>(detail::normalize_graph_capture(std::forward<Args>(args))...);
      auto prototype = factory->create(nullptr);
      if (!prototype) {
        prototype.error().message += "; operator=" + std::string{name};
        detail::throw_graph_error(std::move(prototype.error()));
      }
      auto registered = detail::graph_register_operator(state_,
                                                        name,
                                                        factory,
                                                        std::move(prototype.value()),
                                                        &detail::kGraphOperatorTypeToken<Op>);
      if (!registered)
        detail::throw_graph_error(std::move(registered.error()));
      return OperatorHandle<Op>{&detail::kGraphOperatorTypeToken<Op>,
                                state_,
                                registered.value().index,
                                std::move(registered.value().handle_identity)};
    }
  }

  /// @brief Add one typed Operator declaration using the exact `op` alias.
  /// @tparam Op Concrete Operator type.
  /// @tparam Args Constructor capture argument types.
  /// @param name Nonempty unique bounded UTF-8 declaration name.
  /// @param args Constructor values copied under the same ownership precondition as `op`.
  /// @return Copyable typed authoring handle.
  /// @throws RuntimeError under the same conditions as `op`.
  template <std::derived_from<OperatorBase> Op, typename... Args>
    requires((detail::ReplayableCapture<Args> && ...) &&
             std::constructible_from<Op, const detail::NormalizedGraphCapture<Args>&...>)
  OperatorHandle<Op> add_operator(std::string_view name, Args&&... args) {
    return op<Op>(name, std::forward<Args>(args)...);
  }

  /// @brief Resolve one existing graph-local root Operator by name and exact concrete type.
  /// @tparam Op Expected concrete Operator type.
  /// @param name Exact local declaration name from this Graph's root authoring inventory.
  /// @return Copyable typed authoring handle interchangeable with the creation handle.
  /// @throws RuntimeError with `GRAPH_OPERATOR_NOT_FOUND` when no root declaration has this name.
  /// @throws RuntimeError with `GRAPH_OPERATOR_TYPE_MISMATCH` when the named root declaration has
  /// another concrete Operator type.
  /// @note This is a cold authoring-time lookup. It does not trigger lazy composition, inspect
  /// embedded Graph operators, or participate in per-activation execution.
  /// @note Whole-aggregate parameter reconstruction does not invalidate either the creation handle
  /// or a handle returned by this lookup.
  template <std::derived_from<OperatorBase> Op>
  [[nodiscard]] OperatorHandle<Op> handle(std::string_view name) {
    auto found = detail::graph_find_operator(state_, name);
    if (!found) {
      detail::throw_graph_error(std::move(found.error()));
    }
    if (!detail::graph_operator_valid(state_,
                                      found->index,
                                      &detail::kGraphOperatorTypeToken<Op>,
                                      found->handle_identity.get())) {
      detail::throw_graph_error(Error{ErrorCode::kIncompatible, "GRAPH_OPERATOR_TYPE_MISMATCH"});
    }
    return OperatorHandle<Op>{&detail::kGraphOperatorTypeToken<Op>,
                              state_,
                              found->index,
                              std::move(found->handle_identity)};
  }

  /// @brief Independently construct, compose, and own one reusable Graph instance.
  /// @tparam G Concrete reusable Graph type.
  /// @tparam Args Constructor capture argument types.
  /// @param instance_name Nonempty unique local embedded-instance name.
  /// @param args Constructor values normalized and forwarded once into the child object.
  /// @return Read-only typed handle exposing the child's public interface members.
  /// @throws RuntimeError if parent/child composition, construction, canonical
  /// remapping, name validation, or publication fails.
  /// @note Child composition reaches a terminal state before one atomic parent
  /// publication. The child Graph is never copied or moved and remains owned by
  /// the parent until parent retirement.
  /// @note Unlike replayed Operator and clock factories, embedding constructs the
  /// child once, so constructor arguments may transfer move-only ownership. The
  /// same shallow top-level borrow checks still apply, and Graph authoring handles
  /// remain rejected because they are composition-time identity.
  /// @note Embedded graphs do not inherit the parent's default clock. Declare clock dependencies
  /// in the child's \c compose() using \ref import_clock, and use \ref set_default_clock when an
  /// imported clock is the child's default. After embedding, the immediate parent must supply
  /// each import using \ref bind_clock before compilation. Nested reusable graphs require an
  /// explicit binding at every boundary; clock resolution never searches ancestor scopes by name.
  /// @see import_clock
  /// @see set_default_clock
  /// @see bind_clock
  /// @warning The handle permits interface inspection only. It does not grant
  /// mutable access to the child authoring Graph.
  template <typename G, typename... Args>
    requires std::derived_from<G, Graph> && (detail::ShallowCapture<Args> && ...) &&
             (std::constructible_from<detail::NormalizedGraphCapture<Args>, Args> && ...) &&
             std::constructible_from<G, detail::NormalizedGraphCapture<Args>...>
  GraphHandle<G> embed(std::string_view instance_name, Args&&... args) {
    auto prepared = detail::graph_prepare_authoring(state_);
    if (!prepared)
      detail::throw_graph_error(std::move(prepared.error()));
    auto valid_instance = detail::graph_prepare_embed(state_, instance_name);
    if (!valid_instance)
      detail::throw_graph_error(std::move(valid_instance.error()));

    std::unique_ptr<G> child;
    try {
      child = std::make_unique<G>(detail::normalize_graph_capture(std::forward<Args>(args))...);
    } catch (const std::exception& exception) {
      detail::graph_log_factory_exception("embedded_graph", exception.what());
      auto error = detail::graph_factory_error("GRAPH_EMBED_CONSTRUCTION_FAILED");
      error.message += "; graph instance=" + std::string(instance_name);
      detail::throw_graph_error(std::move(error));
    } catch (...) {
      detail::graph_log_factory_exception("embedded_graph", "non-standard exception");
      auto error = detail::graph_factory_error("GRAPH_EMBED_CONSTRUCTION_UNKNOWN_EXCEPTION");
      error.message += "; graph instance=" + std::string(instance_name);
      detail::throw_graph_error(std::move(error));
    }
    const std::shared_ptr<detail::GraphState> child_state = child->state_;
    const void* child_type_identity = static_cast<const Graph&>(*child).embedding_type_identity();
    if (child_type_identity == nullptr) {
      child_type_identity = &detail::kGraphEmbeddingTypeToken<G>;
    }
    auto typed = detail::graph_prepare_embedded_type(state_, child_state, child_type_identity);
    if (!typed)
      detail::throw_graph_error(std::move(typed.error()));
    std::shared_ptr<const detail::GraphSnapshotData> child_snapshot;
    try {
      child_snapshot = detail::graph_snapshot(child_state);
    } catch (const RuntimeError& exception) {
      auto error = make_error(exception);
      if (error.message == "GRAPH_EMBED_CYCLE" || error.message == "GRAPH_COMPOSE_REENTRANT") {
        throw;
      }
      error.message += "; graph instance=" + std::string(instance_name);
      detail::throw_graph_error(std::move(error));
    }
    auto registered = detail::graph_register_embedded(state_,
                                                      instance_name,
                                                      std::unique_ptr<Graph>(std::move(child)),
                                                      child_state,
                                                      child_snapshot);
    if (!registered)
      detail::throw_graph_error(std::move(registered.error()));
    return GraphHandle<G>{
        state_, registered.value().index, std::move(registered.value().handle_identity)};
  }

  /// @brief Add one exact-type or exact-contract raw-POD directed queue edge.
  /// @tparam SourceT Exact admitted source payload type.
  /// @tparam TargetT Exact admitted target payload type.
  /// @param source Graph-owned output declaration.
  /// @param target Graph-owned input declaration.
  /// @param options Cold edge options copied before return.
  /// @return Typed dynamic flow handle.
  /// @throws RuntimeError for foreign/stale ports, fan-in, cycles, duplicate
  /// topology, unequal complete POD contracts, or invalid options.
  /// @note Distinct C++ endpoint types participate only when both have explicit
  /// raw-POD contracts. Scalar, enum, Tensor, and schema-root edges require the
  /// same C++ payload type.
  template <PublishedPayload SourceT, PublishedPayload TargetT>
    requires(std::same_as<SourceT, TargetT> ||
             (detail::HasChannelPodContract<SourceT> && detail::HasChannelPodContract<TargetT>))
  FlowHandle<SourceT, TargetT> add_flow(const Output<SourceT>& source, const Input<TargetT>& target,
                                        ConnectionOptions options = {}) {
    auto result = detail::graph_add_flow(state_,
                                         &source,
                                         &target,
                                         detail::operator_spec_payload_identity<SourceT>(),
                                         detail::operator_spec_payload_identity<TargetT>(),
                                         options);
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return FlowHandle<SourceT, TargetT>{state_,
                                        result.value().edge,
                                        result.value().source,
                                        result.value().target,
                                        std::move(result.value().handle_identity)};
  }

  /// @brief Add one payload-narrowing edge from a named schema Tensor field to a Tensor input.
  /// @tparam SchemaT Descriptor-backed schema published by the source Operator.
  /// @param source Tensor field named on a Graph-owned schema output.
  /// @param target Graph-owned Tensor input that admits a projected Tensor.
  /// @param options Cold edge options copied before return.
  /// @return Typed dynamic flow handle.
  /// @throws RuntimeError for the same conditions as an exact-type edge, and when the consumer
  /// did not declare \ref InputSpecBuilder::accepts_projected_tensor.
  /// @note Multiple Tensor and schema consumers may share an output. Same-process projection
  /// shares the selected data buffer until every consumer releases it. Inter-process delivery
  /// uses schema transport and requires bounded, copy-capable Tensor transfer contracts; staging
  /// may copy data when the backing cannot be shared. Projection does not convert representation.
  /// Runtime setup still allocates bookkeeping state. An explicit converter Operator remains
  /// responsible for representation changes.
  template <PublishedPayload SchemaT>
  FlowHandle<SchemaT, Tensor> add_flow(const ProjectedTensorOutput<SchemaT>& source,
                                       const Input<Tensor>& target,
                                       ConnectionOptions options = {}) {
    return add_flow_projected(source.source(), target, source.selector(), options);
  }

  /// @brief Add one payload-narrowing edge from a single-Tensor schema output to a Tensor input.
  /// @tparam SchemaT Descriptor-backed schema carrying exactly one Tensor field.
  /// @param source Graph-owned schema output declaration.
  /// @param target Graph-owned Tensor input that admits a projected Tensor.
  /// @param options Cold edge options copied before return.
  /// @return Typed dynamic flow handle.
  /// @note The field is unambiguous only when the schema carries exactly one Tensor. A schema
  /// with several requires `.project()` so the Graph records which field was chosen.
  template <PublishedPayload SchemaT>
    requires detail::kRuntimeDescriptorTensorSchemaPayload<SchemaT>
  FlowHandle<SchemaT, Tensor> add_flow(const Output<SchemaT>& source, const Input<Tensor>& target,
                                       ConnectionOptions options = {}) {
    static_assert(
        detail::kRuntimeSingularDescriptorTensorSchemaPayload<SchemaT>,
        "add_flow(Output<SchemaT>, Input<Tensor>) is ambiguous because this schema declares more "
        "than one Tensor field. Name the field instead, for example "
        "add_flow(source.project(<Schema>TensorFields::<field>), target).");
    constexpr TensorFieldSelector<SchemaT> kOnlyField{
        detail::descriptor_tensor_schema_field_id_at<SchemaT>(0U).value_or(0U), 0U};
    return add_flow_projected(source, target, kOnlyField, options);
  }

  /// @brief Bind a reusable Graph public input as an alias of an internal input.
  /// @tparam T Exact admitted local payload type.
  /// @param public_input Stable public Graph input declaration.
  /// @param internal_input Stable internal Operator input declaration.
  /// @throws RuntimeError if either declaration belongs to another Graph or the
  /// payload/direction contract differs.
  /// @note This is an interface alias, not a runtime queue edge, and therefore
  /// accepts no \ref ConnectionOptions and returns no \ref FlowHandle.
  template <PublishedPayload T>
  void add_flow(const Input<T>& public_input, const Input<T>& internal_input) {
    auto result = detail::graph_bind_input_interface(
        state_, &public_input, &internal_input, detail::operator_spec_payload_identity<T>());
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
  }

  /// @brief Bind a reusable Graph public output as an alias of an internal output.
  /// @tparam T Exact admitted local payload type.
  /// @param internal_output Stable internal Operator output declaration.
  /// @param public_output Stable public Graph output declaration.
  /// @throws RuntimeError if either declaration belongs to another Graph or the
  /// payload/direction contract differs.
  /// @note This is an interface alias, not a runtime queue edge, and therefore
  /// accepts no \ref ConnectionOptions and returns no \ref FlowHandle.
  template <PublishedPayload T>
  void add_flow(const Output<T>& internal_output, const Output<T>& public_output) {
    auto result = detail::graph_bind_output_interface(
        state_, &internal_output, &public_output, detail::operator_spec_payload_identity<T>());
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
  }

  /// @brief Expose an internal input through a reusable Graph input alias.
  /// @tparam T Exact admitted local payload type.
  /// @param public_port Public Graph input declaration.
  /// @param internal_port Internal Operator input declaration.
  /// @throws RuntimeError if alias validation fails.
  template <PublishedPayload T>
  void export_input(const Input<T>& public_port, const Input<T>& internal_port) {
    (void)add_flow(public_port, internal_port);
  }

  /// @brief Expose an internal output through a reusable Graph output alias.
  /// @tparam T Exact admitted local payload type.
  /// @param public_port Public Graph output declaration.
  /// @param internal_port Internal Operator output declaration.
  /// @throws RuntimeError if alias validation fails.
  template <PublishedPayload T>
  void export_output(const Output<T>& public_port, const Output<T>& internal_port) {
    (void)add_flow(internal_port, public_port);
  }

  /// @brief Create a throwing configurator for one Graph-owned input.
  /// @tparam T Exact admitted local payload type.
  /// @param port Exact input declaration.
  /// @return Short-lived fluent configurator referencing stable authoring identity.
  /// @throws RuntimeError if \p port is foreign, stale, or structurally unknown.
  template <PublishedPayload T>
  InputPortConfigurator<T> configure(const Input<T>& port) {
    auto id = detail::graph_find_port(
        state_, &port, PortDirection::kInput, detail::operator_spec_payload_identity<T>());
    if (!id)
      detail::throw_graph_error(std::move(id.error()));
    auto identity = detail::graph_port_registration_identity(state_, id.value());
    if (!identity)
      detail::throw_graph_error(std::move(identity.error()));
    return InputPortConfigurator<T>{state_, id.value(), std::move(identity.value())};
  }

  /// @brief Create a throwing configurator for one non-Tensor output.
  /// @tparam T Exact admitted non-Tensor local payload type.
  /// @param port Exact output declaration.
  /// @return Short-lived fluent configurator referencing stable authoring identity.
  /// @throws RuntimeError if \p port is foreign, stale, or structurally unknown.
  template <PublishedPayload T>
    requires(!std::same_as<T, Tensor> && !detail::kRuntimeDescriptorTensorSchemaPayload<T>)
  OutputPortConfigurator<T> configure(const Output<T>& port) {
    auto id = detail::graph_find_port(
        state_, &port, PortDirection::kOutput, detail::operator_spec_payload_identity<T>());
    if (!id)
      detail::throw_graph_error(std::move(id.error()));
    auto identity = detail::graph_port_registration_identity(state_, id.value());
    if (!identity)
      detail::throw_graph_error(std::move(identity.error()));
    return OutputPortConfigurator<T>{state_, id.value(), std::move(identity.value())};
  }

  /// @brief Return the stable Graph-owned configurator for a descriptor-backed Tensor output.
  /// @tparam T Admitted descriptor-backed Tensor schema payload.
  /// @param port Exact output declaration.
  /// @return Stable configurator reference valid until Graph destruction.
  template <PublishedPayload T>
    requires detail::kRuntimeDescriptorTensorSchemaPayload<T>
  TensorOutputPortConfigurator& configure(const Output<T>& port) {
    auto id = detail::graph_find_port(
        state_, &port, PortDirection::kOutput, detail::operator_spec_payload_identity<T>());
    if (!id)
      detail::throw_graph_error(std::move(id.error()));
    auto configurator = detail::graph_tensor_configurator(state_, id.value());
    if (!configurator)
      detail::throw_graph_error(std::move(configurator.error()));
    return *configurator.value();
  }

  /// @brief Return the stable Graph-owned configurator for a Tensor output.
  /// @param port Exact mutable output declaration.
  /// @return Stable configurator reference valid until Graph destruction.
  /// @throws RuntimeError if \p port is foreign, stale, or structurally unknown.
  TensorOutputPortConfigurator& configure(Output<Tensor>& port) {
    return configure(static_cast<const Output<Tensor>&>(port));
  }

  /// @brief Return the stable Graph-owned configurator for a Tensor output.
  /// @param port Exact output declaration.
  /// @return Stable configurator reference valid until Graph destruction.
  /// @throws RuntimeError if \p port is foreign, stale, or structurally unknown.
  TensorOutputPortConfigurator& configure(const Output<Tensor>& port) {
    auto id = detail::graph_find_port(
        state_, &port, PortDirection::kOutput, detail::operator_spec_payload_identity<Tensor>());
    if (!id)
      detail::throw_graph_error(std::move(id.error()));
    auto configurator = detail::graph_tensor_configurator(state_, id.value());
    if (!configurator)
      detail::throw_graph_error(std::move(configurator.error()));
    return *configurator.value();
  }

  /// @brief Create a throwing configurator for one authored flow.
  /// @tparam SourceT Exact source payload type retained by the flow handle.
  /// @tparam TargetT Exact target payload type retained by the flow handle.
  /// @param flow Exact flow handle owned by this Graph.
  /// @return Short-lived fluent configurator referencing stable edge identity.
  /// @throws RuntimeError if \p flow is foreign or stale.
  template <PublishedPayload SourceT, PublishedPayload TargetT>
  FlowConfigurator<SourceT, TargetT> configure(FlowHandle<SourceT, TargetT> flow) {
    if (!flow.graph_valid() || flow.state_.lock() != state_) {
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_FLOW_HANDLE_INVALID"});
    }
    return FlowConfigurator<SourceT, TargetT>{state_, flow.id(), flow.handle_identity_};
  }

  /// @brief Try to create an input configurator without throwing.
  /// @tparam T Exact admitted local payload type.
  /// @param port Exact input declaration.
  /// @return Configurator or typed lookup/ownership error.
  template <PublishedPayload T>
  expected<InputPortConfigurator<T>, Error> try_configure(const Input<T>& port) {
    auto id = detail::graph_find_port(
        state_, &port, PortDirection::kInput, detail::operator_spec_payload_identity<T>());
    if (!id)
      return make_unexpected(std::move(id.error()));
    auto identity = detail::graph_port_registration_identity(state_, id.value());
    if (!identity)
      return make_unexpected(std::move(identity.error()));
    return InputPortConfigurator<T>{state_, id.value(), std::move(identity.value())};
  }

  /// @brief Try to create a non-Tensor output configurator without throwing.
  /// @tparam T Exact admitted non-Tensor local payload type.
  /// @param port Exact output declaration.
  /// @return Configurator or typed lookup/ownership error.
  template <PublishedPayload T>
    requires(!std::same_as<T, Tensor> && !detail::kRuntimeDescriptorTensorSchemaPayload<T>)
  expected<OutputPortConfigurator<T>, Error> try_configure(const Output<T>& port) {
    auto id = detail::graph_find_port(
        state_, &port, PortDirection::kOutput, detail::operator_spec_payload_identity<T>());
    if (!id)
      return make_unexpected(std::move(id.error()));
    auto identity = detail::graph_port_registration_identity(state_, id.value());
    if (!identity)
      return make_unexpected(std::move(identity.error()));
    return OutputPortConfigurator<T>{state_, id.value(), std::move(identity.value())};
  }

  /// @brief Try to create a descriptor-backed Tensor output configurator.
  /// @tparam T Admitted descriptor-backed Tensor schema payload.
  /// @param port Exact output declaration.
  /// @return Configurator value or typed lookup/ownership error.
  template <PublishedPayload T>
    requires detail::kRuntimeDescriptorTensorSchemaPayload<T>
  expected<TensorOutputPortConfigurator, Error> try_configure(const Output<T>& port) {
    auto id = detail::graph_find_port(
        state_, &port, PortDirection::kOutput, detail::operator_spec_payload_identity<T>());
    if (!id)
      return make_unexpected(std::move(id.error()));
    auto configurator = detail::graph_tensor_configurator(state_, id.value());
    if (!configurator)
      return make_unexpected(std::move(configurator.error()));
    return *configurator.value();
  }

  /// @brief Try to create a value Tensor output configurator without throwing.
  /// @param port Exact mutable Tensor output declaration.
  /// @return Configurator value or typed lookup/ownership error.
  /// @retval ErrorCode::kResourceExhausted if stable configurator storage cannot
  /// be allocated. This result carries no diagnostic string so the error path
  /// itself remains allocation-free.
  expected<TensorOutputPortConfigurator, Error> try_configure(Output<Tensor>& port) {
    return try_configure(static_cast<const Output<Tensor>&>(port));
  }

  /// @brief Try to create a value Tensor output configurator without throwing.
  /// @param port Exact Tensor output declaration.
  /// @return Configurator value or typed lookup/ownership error.
  /// @retval ErrorCode::kResourceExhausted if stable configurator storage cannot
  /// be allocated. This result carries no diagnostic string so the error path
  /// itself remains allocation-free.
  expected<TensorOutputPortConfigurator, Error> try_configure(const Output<Tensor>& port) {
    auto id = detail::graph_find_port(
        state_, &port, PortDirection::kOutput, detail::operator_spec_payload_identity<Tensor>());
    if (!id)
      return make_unexpected(std::move(id.error()));
    auto configurator = detail::graph_tensor_configurator(state_, id.value());
    if (!configurator)
      return make_unexpected(std::move(configurator.error()));
    return *configurator.value();
  }

  /// @brief Try to create a flow configurator without throwing.
  /// @tparam SourceT Exact source payload type retained by the flow handle.
  /// @tparam TargetT Exact target payload type retained by the flow handle.
  /// @param flow Exact flow handle owned by this Graph.
  /// @return Configurator or typed ownership/staleness error.
  template <PublishedPayload SourceT, PublishedPayload TargetT>
  expected<FlowConfigurator<SourceT, TargetT>, Error> try_configure(
      FlowHandle<SourceT, TargetT> flow) {
    if (!flow.graph_valid() || flow.state_.lock() != state_) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument, "GRAPH_FLOW_HANDLE_INVALID"});
    }
    return FlowConfigurator<SourceT, TargetT>{state_, flow.id(), flow.handle_identity_};
  }

  /// @brief Declare an intrinsic external-to-Graph typed topic boundary.
  /// @tparam T Exact portable scalar, Tensor, schema-root, or explicit POD payload type.
  /// @param uri Nonempty bounded `holomq://` topic URI copied before return.
  /// @param options Complete cold boundary contract copied before return.
  /// @return Typed handle exposing the stable Graph-side output declaration.
  /// @throws RuntimeError if identity, URI, options, bounds, or name validation fails.
  /// @note On first external use of a derived Graph, Core completes or terminally
  /// fails composition before constructing the requested boundary port.
  /// @note The value parameter is intentional: it defines one immutable authoring snapshot while
  /// Core deep-copies every view-bearing field before this call returns.
  /// @note Ordinary typed validation rejection consumes the next mixed external
  /// declaration ordinal. An exception before publication restores that ordinal.
  template <detail::ExternalTopicPayload T>
    requires(!detail::kRuntimeDescriptorTensorSchemaPayload<T>)
  ImportedTopic<T> import_topic(
      std::string_view uri,
      TopicOptions options = {}) {  // NOLINT(performance-unnecessary-value-param)
    auto prepared = detail::graph_prepare_authoring(state_);
    if (!prepared)
      detail::throw_graph_error(std::move(prepared.error()));
    auto port = std::make_unique<Output<T>>("external");
    auto result = detail::graph_register_import(state_,
                                                uri,
                                                options,
                                                std::unique_ptr<OutputBase>(std::move(port)),
                                                detail::operator_spec_payload_identity<T>(),
                                                detail::serialized_payload_capacity_bytes<T>());
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    auto handle_identity = detail::graph_port_registration_identity(state_, result.value());
    if (!handle_identity)
      detail::throw_graph_error(std::move(handle_identity.error()));
    return ImportedTopic<T>{state_, result.value().value, std::move(handle_identity.value())};
  }

  /// @brief Declare a descriptor-backed schema external-to-Graph topic boundary.
  /// @tparam T Owning singular or multi-Tensor descriptor-backed schema type.
  /// @param uri Nonempty bounded `holomq://` topic URI copied before return.
  /// @param options Common cold boundary contract copied before return.
  /// @param tensors Complete typed Tensor field inventory copied before return.
  /// @return Typed handle exposing the stable Graph-side output declaration.
  /// @throws RuntimeError if identity, URI, field inventory, bounds, or name validation fails.
  template <detail::ExternalTopicPayload T>
    requires detail::kRuntimeDescriptorTensorSchemaPayload<T>
  ImportedTopic<T> import_topic(
      std::string_view uri,
      TopicOptions options,  // NOLINT(performance-unnecessary-value-param)
      DescriptorTensorTopicContract<T> tensors) {
    auto prepared = detail::graph_prepare_authoring(state_);
    if (!prepared)
      detail::throw_graph_error(std::move(prepared.error()));
    auto port = std::make_unique<Output<T>>("external");
    auto result = detail::graph_register_import(
        state_,
        uri,
        std::move(options),
        std::unique_ptr<OutputBase>(std::move(port)),
        detail::operator_spec_payload_identity<T>(),
        detail::external_topic_payload_capacity_bytes<T>(),
        std::move(tensors.fields_),
        static_cast<std::uint16_t>(detail::descriptor_tensor_schema_tensor_count_v<T>));
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    auto handle_identity = detail::graph_port_registration_identity(state_, result.value());
    if (!handle_identity)
      detail::throw_graph_error(std::move(handle_identity.error()));
    return ImportedTopic<T>{state_, result.value().value, std::move(handle_identity.value())};
  }

  /// @brief Declare an intrinsic Graph-to-external typed topic boundary.
  /// @tparam T Exact portable scalar, Tensor, schema-root, or explicit POD payload type.
  /// @param uri Nonempty bounded `holomq://` topic URI copied before return.
  /// @param options Complete cold boundary contract copied before return.
  /// @return Typed handle exposing the stable Graph-side input declaration.
  /// @throws RuntimeError if identity, URI, options, bounds, or name validation fails.
  /// @note On first external use of a derived Graph, Core completes or terminally
  /// fails composition before constructing the requested boundary port.
  /// @note The value parameter is intentional: it defines one immutable authoring snapshot while
  /// Core deep-copies every view-bearing field before this call returns.
  /// @note Ordinary typed validation rejection consumes the next mixed external
  /// declaration ordinal. An exception before publication restores that ordinal.
  template <detail::ExternalTopicPayload T>
    requires(!detail::kRuntimeDescriptorTensorSchemaPayload<T>)
  ExportedTopic<T> export_topic(
      std::string_view uri,
      TopicOptions options = {}) {  // NOLINT(performance-unnecessary-value-param)
    auto prepared = detail::graph_prepare_authoring(state_);
    if (!prepared)
      detail::throw_graph_error(std::move(prepared.error()));
    auto port = std::make_unique<Input<T>>("external");
    auto result = detail::graph_register_export(state_,
                                                uri,
                                                options,
                                                std::unique_ptr<InputBase>(std::move(port)),
                                                detail::operator_spec_payload_identity<T>(),
                                                detail::serialized_payload_capacity_bytes<T>());
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    auto handle_identity = detail::graph_port_registration_identity(state_, result.value());
    if (!handle_identity)
      detail::throw_graph_error(std::move(handle_identity.error()));
    return ExportedTopic<T>{state_, result.value().value, std::move(handle_identity.value())};
  }

  /// @brief Declare a descriptor-backed schema Graph-to-external topic boundary.
  /// @tparam T Owning singular or multi-Tensor descriptor-backed schema type.
  /// @param uri Nonempty bounded `holomq://` topic URI copied before return.
  /// @param options Common cold boundary contract copied before return.
  /// @param tensors Complete typed Tensor field inventory copied before return.
  /// @return Typed handle exposing the stable Graph-side input declaration.
  /// @throws RuntimeError if identity, URI, field inventory, bounds, or name validation fails.
  template <detail::ExternalTopicPayload T>
    requires detail::kRuntimeDescriptorTensorSchemaPayload<T>
  ExportedTopic<T> export_topic(
      std::string_view uri,
      TopicOptions options,  // NOLINT(performance-unnecessary-value-param)
      DescriptorTensorTopicContract<T> tensors) {
    auto prepared = detail::graph_prepare_authoring(state_);
    if (!prepared)
      detail::throw_graph_error(std::move(prepared.error()));
    auto port = std::make_unique<Input<T>>("external");
    auto result = detail::graph_register_export(
        state_,
        uri,
        std::move(options),
        std::unique_ptr<InputBase>(std::move(port)),
        detail::operator_spec_payload_identity<T>(),
        detail::external_topic_payload_capacity_bytes<T>(),
        std::move(tensors.fields_),
        static_cast<std::uint16_t>(detail::descriptor_tensor_schema_tensor_count_v<T>));
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    auto handle_identity = detail::graph_port_registration_identity(state_, result.value());
    if (!handle_identity)
      detail::throw_graph_error(std::move(handle_identity.error()));
    return ExportedTopic<T>{state_, result.value().value, std::move(handle_identity.value())};
  }

  /// @brief Add one uniquely named backend-neutral policy region.
  /// @param name Nonempty unique bounded UTF-8 name copied before return.
  /// @return Fluent region handle.
  /// @throws RuntimeError if \p name is invalid or duplicated.
  RegionHandle region(std::string_view name);

  /// @brief Add one uniquely named developer-preview manual execution partition.
  /// @param name Nonempty unique bounded UTF-8 name copied before return.
  /// @return Fluent partition handle accepting direct Operators and complete embedded instances.
  /// @throws RuntimeError if \p name is invalid or duplicated.
  /// @note Without \ref auto_partition_remaining or
  /// \ref PartitionHandle::include_all, compile requires all partitions to be
  /// nonempty and to cover every executable Operator exactly once.
  /// @note Manual partitions and \ref region declarations are orthogonal;
  /// regions may overlap partition boundaries.
  /// @note Direct root Operators and selected child subtrees may anchor automatic remainder.
  /// Selected subtrees include every descendant Operator and must not declare partitions
  /// internally. A root \ref PartitionHandle::include_all covers the complete application.
  PartitionHandle partition(std::string_view name);

  /// @brief Automatically partition Operators omitted from exact authored partitions.
  /// @throws RuntimeError if a partition already uses \ref PartitionHandle::include_all.
  /// @note The request is idempotent and may precede or follow exact partition authoring.
  /// Compilation rejects a request with no nonempty exact partition anchor.
  void auto_partition_remaining();

  /// @brief Create a lifecycle ordering/permission authoring builder.
  /// @return Short-lived builder tied to this Graph.
  GraphLifecycleBuilder lifecycle();

  /// @brief Freeze and return one immutable deep-owning authoring snapshot.
  /// @return Snapshot that remains valid across Graph mutation and destruction.
  /// @throws RuntimeError if lazy composition failed or observation is reentrant.
  [[nodiscard]] GraphView view();

  /// @brief Add one Graph-owned replayable clock declaration.
  /// @tparam ClockT Concrete Clock type.
  /// @tparam Args Constructor capture argument types.
  /// @param name Nonempty unique bounded UTF-8 clock name.
  /// @param args Constructor values copied under the same ownership precondition as `op`.
  /// @return Owned named cold clock key.
  /// @throws RuntimeError if construction, naming, or registration fails.
  /// @note On first external use of a derived Graph, Core completes or terminally
  /// fails composition before constructing the requested clock factory.
  /// @note Registration constructs one Clock for validation and discards it. A
  /// later compiler or RunSession replays the retained factory to create its own
  /// ordinary Clock; only `bind_shared_clock()` retains a live instance.
  /// @pre Every normalized capture is copy-constructible and follows its ordinary
  /// C++ copy semantics. Obvious top-level borrowed forms and Graph authoring handles
  /// are rejected. Checks are shallow; nested borrows remain the application's lifetime
  /// responsibility. Construction must be repeatable, and constructor side effects are
  /// not authoritative runtime state.
  template <std::derived_from<Clock> ClockT, typename... Args>
    requires((detail::ReplayableCapture<Args> && ...) &&
             std::constructible_from<ClockT, const detail::NormalizedGraphCapture<Args>&...>)
  ClockKey add_clock(std::string_view name, Args&&... args) {
    auto prepared = detail::graph_prepare_authoring(state_);
    if (!prepared)
      detail::throw_graph_error(std::move(prepared.error()));
    using Factory = detail::ClockFactoryModel<ClockT, detail::NormalizedGraphCapture<Args>...>;
    auto factory =
        std::make_shared<Factory>(detail::normalize_graph_capture(std::forward<Args>(args))...);
    auto result = detail::graph_add_clock(state_, name, std::move(factory));
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return std::move(result.value());
  }

  /// @brief Bind one explicitly shared live clock into cold Graph ownership.
  /// @param name Nonempty unique bounded UTF-8 clock name.
  /// @param clock Non-null shared clock retained by Graph state.
  /// @return Owned named cold clock key.
  /// @throws RuntimeError if naming or pointer validation fails.
  ClockKey bind_shared_clock(std::string_view name, std::shared_ptr<Clock> clock);

  /// @brief Declare one explicit clock dependency for a reusable Graph.
  /// @param name Nonempty unique bounded UTF-8 clock import name.
  /// @return Owned named import key that contracts and conversions may reference.
  /// @throws RuntimeError if naming or registration fails.
  /// @note An import creates no runtime clock. Every embedded instance must be
  /// bound explicitly by its direct parent before compilation.
  ClockKey import_clock(std::string_view name);

  /// @brief Bind one embedded Graph clock import to one exact parent clock.
  /// @tparam G Concrete reusable Graph type.
  /// @param child Exact live child instance owned by this Graph.
  /// @param child_import Named clock import declared by that child root.
  /// @param parent_clock Named clock declaration or import in this Graph.
  /// @return This Graph.
  /// @throws RuntimeError for a foreign/stale child handle, a non-import child
  /// key, an unknown parent key, or a duplicate/conflicting binding.
  /// @note Resolution follows only these instance-qualified records. It never
  /// searches an ancestor scope by clock name.
  template <typename G>
    requires std::derived_from<G, Graph>
  // ClockKey is an owning cold-authoring value. Taking both keys by value preserves efficient
  // rvalue transfer into the type-erased implementation while giving lvalue callers one
  // explicit ownership copy at this public boundary.
  // NOLINTNEXTLINE(performance-unnecessary-value-param)
  Graph& bind_clock(const GraphHandle<G>& child, ClockKey child_import, ClockKey parent_clock) {
    auto child_owner = child.state_.lock();
    if (!child_owner || child_owner.get() != state_.get()) {
      detail::throw_graph_error(
          Error{ErrorCode::kInvalidArgument, "GRAPH_CLOCK_BINDING_HANDLE_INVALID"});
    }
    auto result = detail::graph_bind_clock_import(state_,
                                                  child.index_,
                                                  child.handle_identity_.get(),
                                                  std::move(child_import),
                                                  std::move(parent_clock));
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return *this;
  }

  /// @brief Select one previously declared named clock as Graph default.
  /// @param key Existing non-default named key.
  /// @return This Graph.
  /// @throws RuntimeError if \p key is default, empty, or not declared.
  Graph& set_default_clock(ClockKey key);

  /// @brief Return the selected default clock key.
  /// @return Named key, or the distinct default/omitted key when none was selected.
  /// @throws RuntimeError if lazy composition failed or observation is reentrant.
  [[nodiscard]] ClockKey default_clock();

  /// @brief Add one immutable directed affine clock conversion.
  /// @param from Existing named source clock.
  /// @param to Existing distinct named destination clock.
  /// @param conversion Positive exact rate and signed nanosecond offset.
  /// @return This Graph.
  /// @throws RuntimeError for missing/self clocks, an invalid scale, a
  /// conflicting conversion for the same directed pair, or a directed cycle.
  /// @note Core reduces the positive ratio to lowest terms. Repeating the same
  /// pair with an equivalent canonical ratio and equal offset is idempotent.
  Graph& add_clock_conversion(ClockKey from, ClockKey to, AffineClockConversion conversion);

 protected:
  /// @brief Author a derived Graph body during Core-managed lazy composition.
  ///
  /// Core invokes this override at most once after derived construction. Normal
  /// authoring verbs are valid inside the active transaction. Topology observers
  /// and other same-Graph paths that re-enter the composition gate fail with
  /// `GRAPH_COMPOSE_REENTRANT`; a terminal failure is retained and replayed.
  /// @warning Do not author from the derived constructor and do not call this
  /// virtual function directly. Constructor misuse and direct self-recursion
  /// have no promised runtime diagnosis.
  /// @warning The complete most-derived Graph must remain alive for the entire
  /// callback; concurrent destruction is outside the Graph lifetime contract.
  virtual void compose() {}

  /// @brief Supply a process-stable logical type identity for a language-hosted Graph adapter.
  /// @return A non-null identity shared by every instance of the same logical hosted type, or
  /// `nullptr` to use the concrete C++ Graph type.
  /// @note Ordinary C++ Graph subclasses should retain the default. Language bindings that host
  /// multiple logical graph types behind one C++ adapter may override this hook so recursion checks
  /// distinguish those hosted types without weakening cycle detection.
  /// @note This virtual is deliberately appended after the pre-existing `compose()` slot so adding
  /// the hosted-language hook does not change the slot used by existing Graph subclasses.
  [[nodiscard]] virtual const void* embedding_type_identity() const noexcept { return nullptr; }

 private:
  template <std::derived_from<OperatorBase> Op>
  OperatorHandle<Op> add_operator_with_initial_params(std::string_view name,
                                                      const typename Op::Params& params) {
    if constexpr (!detail::SelectableOperatorLifecycleHooks<Op>) {
      static_assert(detail::SelectableOperatorLifecycleHooks<Op>,
                    "Operator lifecycle hooks must be public, unambiguous, and have the exact "
                    "void() signature");
    } else {
      auto prepared = detail::graph_prepare_authoring(state_);
      if (!prepared)
        detail::throw_graph_error(std::move(prepared.error()));
      using Factory = detail::OperatorFactoryModel<Op>;
      auto factory = std::make_shared<Factory>();
      auto prototype = factory->create(nullptr);
      if (!prototype) {
        prototype.error().message += "; operator=" + std::string{name};
        detail::throw_graph_error(std::move(prototype.error()));
      }
      const detail::InitialOperatorParameters initial_params{
          .value = &params,
          .local_type_token = param_type_descriptor<typename Op::Params>().local_type_token,
          .schema = param_schema_descriptor<typename Op::Params>(),
      };
      auto registered = detail::graph_register_operator(state_,
                                                        name,
                                                        factory,
                                                        std::move(prototype.value()),
                                                        &detail::kGraphOperatorTypeToken<Op>,
                                                        &initial_params);
      if (!registered)
        detail::throw_graph_error(std::move(registered.error()));
      return OperatorHandle<Op>{&detail::kGraphOperatorTypeToken<Op>,
                                state_,
                                registered.value().index,
                                std::move(registered.value().handle_identity)};
    }
  }

  /// @brief Register one schema-to-Tensor edge after the field identity is known.
  template <PublishedPayload SchemaT>
  FlowHandle<SchemaT, Tensor> add_flow_projected(const Output<SchemaT>& source,
                                                 const Input<Tensor>& target,
                                                 TensorFieldSelector<SchemaT> selector,
                                                 ConnectionOptions options) {
    static_assert(detail::kRuntimeDescriptorTensorSchemaPayload<SchemaT>,
                  "Only a descriptor-backed schema output can be projected to a Tensor input.");
    if (!detail::descriptor_tensor_schema_field_selector_matches<SchemaT>(selector.field_id,
                                                                          selector.tensor_index)) {
      detail::throw_graph_error(
          Error{ErrorCode::kInvalidArgument, "GRAPH_PROJECTED_TENSOR_FIELD_UNKNOWN"});
    }
    auto result =
        detail::graph_add_flow(state_,
                               &source,
                               &target,
                               detail::operator_spec_payload_identity<SchemaT>(),
                               detail::operator_spec_payload_identity<Tensor>(),
                               options,
                               TensorProjectionRecord{.field_id = selector.field_id,
                                                      .tensor_index = selector.tensor_index});
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return FlowHandle<SchemaT, Tensor>{state_,
                                       result.value().edge,
                                       result.value().source,
                                       result.value().target,
                                       std::move(result.value().handle_identity)};
  }

  std::shared_ptr<detail::GraphState> state_;
  friend class detail::GraphCompilerAccess;
  friend class detail::GraphOperatorRegistrationAccess;
  friend class detail::GraphPayloadRegistrationAccess;
  friend class detail::OperatorTestingAccess;
  friend class detail::GraphState;
  friend class detail::ParameterConfigBindingAccess;
  friend class detail::ParameterInputAccess;
};

namespace detail {

class GraphOperatorRegistrationAccess {
 public:
  template <std::derived_from<OperatorBase> Op>
  static OperatorHandle<Op> add_with_params(Graph& graph, std::string_view name,
                                            const typename Op::Params& params) {
    return graph.add_operator_with_initial_params<Op>(name, params);
  }
};

/// @brief Route type-erased registry declarations through Graph-owned validation.
class GraphPayloadRegistrationAccess {
 public:
  static FlowRegistration add_flow(Graph& graph, const OutputBase& source, const InputBase& target,
                                   const PayloadIdentityRecord& source_identity,
                                   const PayloadIdentityRecord& target_identity,
                                   ConnectionOptions options) {
    auto result = graph_add_flow(
        graph.state_, &source, &target, source_identity, target_identity, std::move(options));
    if (!result) {
      throw_graph_error(std::move(result.error()));
    }
    return result.value();
  }

  static FlowRegistration add_tensor_projection(Graph& graph, const OutputBase& source,
                                                const InputBase& target,
                                                const PayloadIdentityRecord& source_identity,
                                                TensorProjectionRecord projection,
                                                ConnectionOptions options) {
    auto result = graph_add_flow(graph.state_,
                                 &source,
                                 &target,
                                 source_identity,
                                 operator_spec_payload_identity<Tensor>(),
                                 std::move(options),
                                 projection);
    if (!result) {
      throw_graph_error(std::move(result.error()));
    }
    return result.value();
  }

  static GraphPortId port(Graph& graph, const void* declaration, PortDirection direction,
                          const PayloadIdentityRecord& identity) {
    auto result = graph_find_port(graph.state_, declaration, direction, identity);
    if (!result) {
      throw_graph_error(std::move(result.error()));
    }
    return result.value();
  }

  [[nodiscard]] static std::weak_ptr<GraphState> state(Graph& graph) noexcept {
    return graph.state_;
  }

  static void export_input(Graph& graph, const InputBase& public_port,
                           const InputBase& internal_port, PayloadIdentityRecord identity) {
    auto result =
        graph_bind_input_interface(graph.state_, &public_port, &internal_port, std::move(identity));
    if (!result) {
      throw_graph_error(std::move(result.error()));
    }
  }

  static void export_output(Graph& graph, const OutputBase& public_port,
                            const OutputBase& internal_port, PayloadIdentityRecord identity) {
    auto result = graph_bind_output_interface(
        graph.state_, &internal_port, &public_port, std::move(identity));
    if (!result) {
      throw_graph_error(std::move(result.error()));
    }
  }
};

}  // namespace detail

inline const std::shared_ptr<detail::GraphState>& detail::ParameterConfigBindingAccess::state(
    Graph& graph) noexcept {
  return graph.state_;
}

}  // namespace holoscan
