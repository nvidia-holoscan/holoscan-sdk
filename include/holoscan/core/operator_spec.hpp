/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <typeinfo>
#include <unordered_set>
#include <utility>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/channel/pod_type.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/payload_identity.hpp>
#include <holoscan/core/payload_kind.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/predicate_input_view.hpp>
#include <holoscan/core/rate.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/schema_identity.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/tensor_field.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/core/tensor_schema_output_loan.hpp>
#include <holoscan/schema/schema_package.hpp>
#include <holoscan/time/clock_registry.hpp>

namespace holoscan {

class ReadableSource;
class NotificationSource;
class DoorbellSource;
class SpinSource;
enum class ReadinessSourceKind : std::uint8_t;
class OperatorSpec;
class TensorFieldInputSpecBuilder;
class TensorFieldOutputSpecBuilder;

/// @brief Derive Tensor slot capacity from one owning Operator input's finite compiled capacity.
/// @param input Input declaration from the same Operator setup pass as the derived output.
/// @return Compile-resolved input-capacity expression.
/// @note The connected producer must have a finite runtime-pool allocation or be an imported
/// external Tensor boundary. Embedded-Graph interface aliases are resolved to their concrete
/// producer before capacity evaluation. A generic unbounded pass-through producer is rejected at
/// compile.
[[nodiscard]] inline TensorByteSpanExpression input_max_byte_span(const InputBase& input) {
  return detail_make_input_max_byte_span(&input);
}

namespace detail {
class SyntheticOperatorBinding;
class OperatorSpecCompilerAccess;
class OperatorSpecRuntimeAccess;
class OperatorSpecTestAccess;

/// @brief Retain one setup-time rate declaration before Graph canonicalization.
///
/// The follows pointer is cold process-local evidence only. Graph replaces it with an owner-local
/// canonical port identity before any snapshot, plan, diagnostic, or artifact is published.
struct AuthoredPortRate {
  PortRateKind kind{PortRateKind::kUnspecified};
  std::optional<Rate> nominal_rate;
  std::optional<Rate> expected_rate;
  const InputBase* follows_input{};
  std::optional<RateRatio> follows_ratio;
  std::optional<LifecycleStage> lifecycle_binding_stage;
  std::optional<Rate> max_nominal_rate;
  Error error{ErrorCode::kSuccess};
  std::string_view error_token;
};

/// @brief Provide a bounded process-local discriminator for one admitted port payload type.
///
/// The address is useful only while compiling an in-process authoring graph. It is never semantic
/// schema identity and must never enter an ExecutionPlan, snapshot, catalog, trace, or wire record.
/// Each instantiated token occupies one byte and requires neither RTTI nor a compiler type name.
template <PublishedPayload T>
inline constexpr const std::byte& kOperatorSpecLocalTypeToken = kPublishedPayloadLocalTypeToken<T>;

/// Process-local cold discriminator used only to verify typed callback replay and rebinding.
/// Its address is never copied into a Graph snapshot, plan, trace, diagnostic, or wire record.
template <typename T>
inline constexpr std::byte kLifecycleDeclaringTypeToken{};

/// @brief Return the cold semantic payload family associated with an admitted port type.
/// @tparam T Admitted public payload type.
/// @return Default semantic family used to select the typed binding path during compilation.
/// @note A later admitted interface policy may refine the representation of an individual Tensor
/// sample without changing this declaration's exact local type token.
template <PublishedPayload T>
[[nodiscard]] consteval PayloadKind operator_spec_payload_kind() noexcept {
  if constexpr (std::same_as<T, Tensor> || HasDescriptorTensorSchemaNativeTraits<T>) {
    return PayloadKind::kDescriptorBacked;
  } else if constexpr (HasSchemaIdentity<T> && !kPyObjectPayload<T>) {
    return PayloadKind::kSerializedBytes;
  } else {
    return PayloadKind::kInlineTyped;
  }
}

/// @brief Return complete stable schema identity when the port type is a generated schema root.
/// @tparam T Admitted public payload type.
/// @return Complete schema identity for generated roots; otherwise `std::nullopt`.
template <PublishedPayload T>
[[nodiscard]] consteval std::optional<SchemaIdentity> operator_spec_schema_identity() noexcept {
  if constexpr (HasSchemaIdentity<T>) {
    return schema_identity<T>();
  } else {
    return std::nullopt;
  }
}

/// @brief Return the closed scalar identity for an admitted arithmetic or enum payload.
/// @tparam T Admitted arithmetic or enum payload type.
/// @return Stable scalar identity kind.
template <typename T>
  requires(std::is_arithmetic_v<T> || std::is_enum_v<T>)
[[nodiscard]] consteval ScalarIdentityKind operator_spec_scalar_identity_kind() {
  if constexpr (std::same_as<T, bool>) {
    return ScalarIdentityKind::kBoolean;
  } else if constexpr (std::same_as<T, std::uint8_t>) {
    return ScalarIdentityKind::kUInt8;
  } else if constexpr (std::same_as<T, std::int8_t>) {
    return ScalarIdentityKind::kInt8;
  } else if constexpr (std::same_as<T, std::uint16_t>) {
    return ScalarIdentityKind::kUInt16;
  } else if constexpr (std::same_as<T, std::int16_t>) {
    return ScalarIdentityKind::kInt16;
  } else if constexpr (std::same_as<T, std::uint32_t>) {
    return ScalarIdentityKind::kUInt32;
  } else if constexpr (std::same_as<T, std::int32_t>) {
    return ScalarIdentityKind::kInt32;
  } else if constexpr (std::same_as<T, std::uint64_t>) {
    return ScalarIdentityKind::kUInt64;
  } else if constexpr (std::same_as<T, std::int64_t>) {
    return ScalarIdentityKind::kInt64;
  } else if constexpr (std::same_as<T, float>) {
    static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4,
                  "Graph ports require the portable IEEE-754 binary32 representation");
    return ScalarIdentityKind::kFloat32;
  } else if constexpr (std::same_as<T, double>) {
    static_assert(std::numeric_limits<double>::is_iec559 && sizeof(double) == 8,
                  "Graph ports require the portable IEEE-754 binary64 representation");
    return ScalarIdentityKind::kFloat64;
  } else if constexpr (std::is_enum_v<T>) {
    return ScalarIdentityKind::kLocalEnum;
  } else {
    static_assert(!std::same_as<T, T>,
                  "Graph ports reject platform-dependent arithmetic representations");
  }
}

/// @brief Return complete pointer-free payload identity for a declared port type.
/// @tparam T Admitted public payload type.
/// @return Closed scalar, Tensor, complete schema, or complete raw-POD identity record.
template <PublishedPayload T>
[[nodiscard]] PayloadIdentityRecord operator_spec_payload_identity() {
  static_assert(assert_channel_pod_contract_if_declared<T>());
  constexpr bool builtin_family =
      std::is_arithmetic_v<T> || std::is_enum_v<T> || std::same_as<T, Tensor>;
  static_assert(!(builtin_family && (HasSchemaIdentity<T> || HasChannelPodContract<T>)),
                "Built-in scalar, enum, and Tensor identities cannot be replaced by schema/POD");
  PayloadIdentityRecord result;
  result.payload_kind = operator_spec_payload_kind<T>();
  if constexpr (std::is_arithmetic_v<T> || std::is_enum_v<T>) {
    result.kind = PayloadIdentityKind::kScalar;
    if constexpr (std::is_enum_v<T>) {
      static_assert(sizeof(T) == 1U || sizeof(T) == 2U || sizeof(T) == 4U || sizeof(T) == 8U,
                    "Local enum Graph ports require a 1-, 2-, 4-, or 8-byte representation");
      result.scalar =
          ScalarIdentityRecord{.kind = ScalarIdentityKind::kLocalEnum,
                               .native_object_size = static_cast<std::uint8_t>(sizeof(T))};
    } else {
      result.scalar = ScalarIdentityRecord{.kind = operator_spec_scalar_identity_kind<T>()};
    }
  } else if constexpr (std::same_as<T, Tensor>) {
    result.kind = PayloadIdentityKind::kTensor;
  } else {
    static_assert(HasSchemaIdentity<T> != HasChannelPodContract<T>,
                  "A payload must select exactly one schema or raw-POD identity");
    if constexpr (HasSchemaIdentity<T>) {
      result.kind = PayloadIdentityKind::kSchema;
      const SchemaIdentity identity = schema_identity<T>();
      result.schema = make_schema_identity_record(identity);
      if constexpr (HasDescriptorTensorSchemaNativeTraits<T>) {
        result.transport_profile_tag =
            schema_transport_profile_tag(descriptor_tensor_schema_transport_profile_v<T>);
      } else if constexpr (HasSchemaPackage<T>) {
        result.transport_profile_tag =
            schema_transport_profile_tag(schema_package<T>().transport_profile());
      }
    } else {
      static_assert(sizeof(T) <= std::numeric_limits<std::uint32_t>::max());
      static_assert(alignof(T) <= std::numeric_limits<std::uint32_t>::max());
      result.kind = PayloadIdentityKind::kPod;
      PodIdentityRecord pod{
          .canonical_name = std::string(ChannelTypeIdentity<T>::name),
          .version = ChannelTypeIdentity<T>::version,
          .object_size = static_cast<std::uint32_t>(sizeof(T)),
          .object_alignment = static_cast<std::uint32_t>(alignof(T)),
      };
      std::apply(
          [&pod](const auto&... field) {
            (pod.fields.push_back(PodFieldIdentityRecord{
                 .name = std::string(field.name),
                 .byte_offset = field.byte_offset,
                 .scalar_kind = field.scalar_kind,
                 .element_count = field.element_count,
             }),
             ...);
          },
          ChannelPodLayout<T>::fields);
      result.pod = std::move(pod);
    }
  }
  return result;
}

/// @brief Validate one generated package for a port payload type.
/// @tparam T Admitted public payload type.
/// @return Cached validation outcome for schema payloads.
template <PublishedPayload T>
[[nodiscard]] const Error& operator_spec_schema_package_validation() {
  if constexpr (HasSchemaPackage<T>) {
    static const Error kResult = [] {
      const auto& package = schema_package<T>();
      auto valid = validate_schema_package(package);
      if (!valid) {
        return std::move(valid.error());
      }
      return Error{ErrorCode::kSuccess};
    }();
    return kResult;
  } else if constexpr (HasDescriptorTensorSchemaNativeTraits<T>) {
    using Root = typename DescriptorTensorSchemaNativeTraits<T>::root_type;
    static_assert(HasSchemaPackage<Root>,
                  "a descriptor-backed Tensor schema must expose its generated root package");
    static const Error kResult = [] {
      const auto& package = schema_package<Root>();
      auto valid = validate_schema_package(package);
      if (!valid) {
        return std::move(valid.error());
      }
      return Error{ErrorCode::kSuccess};
    }();
    return kResult;
  } else {
    static const Error kResult{ErrorCode::kSuccess};
    return kResult;
  }
}

/// @brief Return the direct Tensor field count carried by one admitted port type.
template <PublishedPayload T>
[[nodiscard]] consteval std::uint16_t operator_spec_descriptor_tensor_field_count() noexcept {
  if constexpr (HasDescriptorTensorSchemaNativeTraits<T>) {
    constexpr std::size_t count = descriptor_tensor_schema_tensor_count_v<T>;
    static_assert(count > 0U && count <= kMaximumTensorSchemaLoanFields,
                  "descriptor Tensor field count must fit the multi-field Tensor loan");
    return static_cast<std::uint16_t>(count);
  } else if constexpr (std::same_as<T, Tensor>) {
    return 1U;
  } else {
    return 0U;
  }
}

/// @brief Copy view-bearing producer options into OperatorSpec-owned storage.
/// @param options Call-lifetime producer options.
/// @return Deep-owned options safe for later Graph capture.
[[nodiscard]] inline PayloadOptionsRecord copy_payload_options(const PayloadOptions& options) {
  PayloadOptionsRecord result{
      .max_inline_bytes = options.max_inline_bytes,
      .max_tensor_byte_span = options.max_tensor_byte_span,
      .interface_tensor_emit_policy = options.interface_tensor_emit_policy,
  };
  if (options.schema) {
    result.schema = make_schema_identity_record(*options.schema);
  }
  return result;
}

/// @brief Return whether producer options author Tensor transfer policy rather than schema alone.
[[nodiscard]] inline bool payload_options_carry_tensor_transfer(
    const PayloadOptionsRecord& options) noexcept {
  return options.interface_tensor_emit_policy != InterfaceTensorEmitPolicy::kRequireExportable ||
         options.max_inline_bytes.has_value() || options.max_tensor_byte_span.has_value();
}

/// @brief Project allocation bounds guaranteed by a normalized runtime-pool contract.
[[nodiscard]] inline TensorPoolBounds normalized_tensor_pool_bounds(
    const TensorContractRecord& contract) noexcept {
  // Runtime-pool normalization guarantees both values; keep that invariant explicit here instead
  // of scattering unchecked optional accesses across each authoring builder.
  // NOLINTBEGIN(bugprone-unchecked-optional-access)
  return TensorPoolBounds{
      .max_rank = contract.max_rank.value(),
      .max_byte_span = static_cast<std::size_t>(contract.max_byte_span.value()),
  };
  // NOLINTEND(bugprone-unchecked-optional-access)
}

/// @brief Return transfer policy already proven present by the caller's normalized-contract gate.
[[nodiscard]] inline const TensorTransferPolicy& normalized_tensor_transfer(
    const TensorContractRecord& contract) noexcept {
  // Callers first establish that normalized transfer policy is present.
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  return contract.transfer.value();
}

/// Cold-only field declaration plus its process-local schema-type discriminator.
struct AuthoredTensorFieldConfiguration {
  TensorFieldConfigurationRecord configuration;
  std::optional<TensorByteSpanExpression> tensor_byte_span_expression;
  std::optional<TensorMemoryKindExpression> tensor_memory_kind_expression;
  const void* schema_type_token{};
  bool selector_valid{};
};

/// @brief Add the selected field identity to a common Tensor-contract diagnostic.
[[nodiscard]] inline Error tensor_field_contract_error(
    Error error, const TensorFieldConfigurationRecord& field) {
  if (!error.message.empty()) {
    error.message.append(" (field_id=");
    error.message.append(std::to_string(field.field_id));
    error.message.append(", tensor_index=");
    error.message.append(std::to_string(field.tensor_index));
    error.message.push_back(')');
  }
  return error;
}
}  // namespace detail

/// @brief Register sparse lifecycle callbacks and ordering facts during Operator setup.
///
/// The builder is a non-owning view of one `OperatorSpec` declaration pass. Callback registration
/// stores bounded cold typed-erasure metadata; it does not invoke the callback or retain the
/// authoring Operator in a frozen Graph record. Lifecycle dependency names are copied immediately.
class HOLOSCAN_CORE_EXPORT LifecycleBuilder {
 public:
  /// @brief Register one required callback for a lifecycle stage.
  /// @tparam T Exact concrete Operator type registered with the owning Graph.
  /// @param stage Exact logical checkpoint whose callback is registered.
  /// @param method Non-throwing member callback invoked by the lifecycle executor.
  /// @return This builder for fluent setup declaration.
  /// @throws std::bad_alloc if bounded cold registration storage cannot be allocated.
  /// @pre \p method must be declared by the exact concrete Operator type. A base-class or foreign
  /// member callback is rejected when Graph authoring validates the declaration.
  template <typename T>
  LifecycleBuilder& stage(LifecycleStage stage,
                          LifecycleStatus (T::*method)(LifecycleContext&) noexcept);

  /// @brief Register one optional callback for a lifecycle stage.
  ///
  /// Optional means callback absence is permitted. A registered callback returning failure still
  /// fails the transition, and baseline attempts it exactly once.
  /// @tparam T Exact concrete Operator type registered with the owning Graph.
  /// @param stage Exact logical checkpoint whose callback is registered.
  /// @param method Non-throwing member callback invoked when present.
  /// @return This builder for fluent setup declaration.
  /// @throws std::bad_alloc if bounded cold registration storage cannot be allocated.
  /// @pre \p method must be declared by the exact concrete Operator type. A base-class or foreign
  /// member callback is rejected when Graph authoring validates the declaration.
  template <typename T>
  LifecycleBuilder& optional(LifecycleStage stage,
                             LifecycleStatus (T::*method)(LifecycleContext&) noexcept);

  /// @brief Declare this Operator as the unique producer of one same-stage ordering fact.
  /// @param stage Stage whose successful logical checkpoint satisfies the fact.
  /// @param key Stable qualified dependency name copied by the declaration pass.
  /// @return This builder for fluent setup declaration.
  /// @throws std::bad_alloc if the copied dependency record cannot be allocated.
  LifecycleBuilder& produces(LifecycleStage stage, LifecycleDependencyKey key);

  /// @brief Require one same-stage ordering fact before this Operator's checkpoint.
  /// @param stage Stage in which the dependency applies.
  /// @param key Stable qualified dependency name copied by the declaration pass.
  /// @return This builder for fluent setup declaration.
  /// @throws std::bad_alloc if the copied dependency record cannot be allocated.
  LifecycleBuilder& requires_dependency(LifecycleStage stage, LifecycleDependencyKey key);

 private:
  explicit LifecycleBuilder(OperatorSpec* spec) noexcept : spec_(spec) {}

  OperatorSpec* spec_{};

  friend class OperatorSpec;
};

/// @brief Configure one typed input declaration during cold Operator setup.
///
/// Builder references remain valid for the lifetime of their `OperatorSpec`. Configuration records
/// metadata only; it does not allocate runtime queues or bind backend endpoints.
class InputSpecBuilder {
 public:
  /// @brief Permit this input to be intentionally unwired.
  /// @return This builder for fluent declaration; repeated calls have no further effect.
  /// @note The contract must use the single-sample \c Exact selector and explicit \c SafeDefault.
  /// Selector cardinality does not restrict the admitted payload family.
  /// Trigger-guaranteed inputs, window selectors/anchors, and predicate evaluation inputs remain
  /// required. Ordinary compute-only side inputs preserve their existing selection phases.
  /// Connected inputs retain ordinary selection. An unwired input supplies no payload or fallback
  /// value; the Operator handles \c kSafeDefault with reason \c kNotConnected.
  InputSpecBuilder& optional() noexcept {
    optional_ = true;
    return *this;
  }

  /// @brief Set the finite input history capacity requested by this declaration.
  /// @param depth Positive number of retained sample slots.
  /// @return This builder for fluent cold-path configuration.
  InputSpecBuilder& queue_depth(std::uint32_t depth) noexcept {
    queue_depth_ = depth;
    return *this;
  }

  /// @brief Declare an advisory expected nominal input rate.
  /// @param rate Exact positive rational samples per second.
  /// @return This builder after retaining the normalized declaration or first error.
  HOLOSCAN_CORE_EXPORT InputSpecBuilder& expected_rate(Rate rate);

  /// @brief Record an alternate diagnostic display name.
  /// @param alias Display-only name copied into cold metadata.
  /// @return This builder for fluent cold-path configuration.
  /// @throws std::bad_alloc if storage for the copied alias cannot be allocated.
  /// @throws std::length_error if \p alias exceeds the implementation's maximum string size.
  /// @note An alias does not change canonical port identity or repair a declaration-name mismatch.
  InputSpecBuilder& diagnostic_alias(std::string_view alias) {
    diagnostic_alias_.assign(alias);
    return *this;
  }

  /// @brief Record one normalized consumer Tensor spec carried by this input.
  /// @param spec Representation, finite bounds, and optional conversion requirements.
  /// @return This builder for fluent cold-path configuration.
  InputSpecBuilder& expects_tensor(const TensorInputSpec& spec) {
    auto normalized = detail::normalize_tensor_input_spec(spec);
    if (!normalized) {
      if (tensor_contract_error_.code == ErrorCode::kSuccess) {
        tensor_contract_error_ = std::move(normalized.error());
      }
      return *this;
    }
    tensor_layout_ = detail::tensor_representation(*normalized);
    memory_conversion_ = normalized->memory_conversion;
    tensor_contract_ = std::move(*normalized);
    return *this;
  }

  /// @brief Admit a Tensor projected out of a producer's descriptor-backed schema.
  /// @return This builder for fluent cold-path configuration.
  /// @note Opting in is a property of the consumer, not the producer: an edge that narrows a
  /// richer payload is only well formed when the receiving Operator has said it wants the Tensor
  /// alone. Any declared Tensor contract still applies to the projected field.
  /// @note Projection supports bounded same-process fanout and inter-process schema transport.
  /// Each consumer selects its own field and retains its own sample lease. Same-process delivery
  /// shares storage; inter-process delivery requires the producer's bounded transfer policy and
  /// may stage a copy. Projection preserves sample identity without representation conversion.
  InputSpecBuilder& accepts_projected_tensor() noexcept {
    accepts_projected_tensor_ = true;
    return *this;
  }

  /// @brief Record schema identity and payload options carried by this input.
  /// @param options Complete payload options copied into this builder.
  /// @return This builder for fluent cold-path configuration.
  InputSpecBuilder& payload_options(PayloadOptions options) {
    payload_options_ = detail::copy_payload_options(options);
    return *this;
  }

  /// @brief Select one Tensor member for field-scoped input declarations.
  /// @tparam SchemaT Owning generated descriptor-backed schema type.
  /// @param selector Generated schema-typed field token.
  /// @return Field-scoped builder for logical and materialization requirements.
  /// @warning The returned non-owning builder must not outlive this parent builder.
  template <typename SchemaT>
  [[nodiscard]] inline TensorFieldInputSpecBuilder tensor_field(
      TensorFieldSelector<SchemaT> selector);

  /// @brief Declare one Tensor field from a direction-specific input spec.
  /// @tparam SchemaT Owning generated descriptor-backed schema type.
  /// @param selector Generated schema-typed field token.
  /// @param spec Logical and conversion requirements copied into this declaration.
  /// @return This input builder.
  /// @throws std::bad_alloc if field-record storage cannot be allocated.
  template <typename SchemaT>
  InputSpecBuilder& tensor_field(TensorFieldSelector<SchemaT> selector,
                                 const TensorInputSpec& spec);

  /// @brief Apply one input spec to a statically listed set of Tensor fields.
  /// @tparam SchemaT Owning generated descriptor-backed schema type.
  /// @tparam N Number of listed selectors.
  /// @param selectors Generated schema-typed field tokens.
  /// @param spec Shared logical and conversion requirements.
  /// @return This input builder.
  /// @throws std::bad_alloc if field-record storage cannot be allocated.
  template <typename SchemaT, std::size_t N>
  InputSpecBuilder& tensor_fields(const std::array<TensorFieldSelector<SchemaT>, N>& selectors,
                                  const TensorInputSpec& spec);

  /// @brief Declare heterogeneous common input specs for selected fields.
  template <typename SchemaT, std::size_t N>
  InputSpecBuilder& tensor_fields(
      const std::array<std::pair<TensorFieldSelector<SchemaT>, TensorInputSpec>, N>& fields);

 private:
  std::optional<std::uint32_t> queue_depth_;
  bool optional_{};
  detail::AuthoredPortRate rate_;
  std::string diagnostic_alias_;
  std::optional<TensorRepresentation> tensor_layout_;
  std::optional<MemoryConversionPolicy> memory_conversion_;
  std::optional<PayloadOptionsRecord> payload_options_;
  std::optional<TensorContractRecord> tensor_contract_;
  Error tensor_contract_error_{ErrorCode::kSuccess};
  std::uint16_t descriptor_tensor_field_count_{};
  bool accepts_projected_tensor_{};
  std::vector<detail::AuthoredTensorFieldConfiguration> tensor_fields_;

  friend class OperatorSpec;
  friend class TensorFieldInputSpecBuilder;
  friend class detail::OperatorSpecCompilerAccess;
  friend class detail::OperatorSpecRuntimeAccess;
  friend class detail::SyntheticOperatorBinding;
};

/// @brief Configure one ordinary typed output during cold Operator setup.
///
/// The common builder deliberately has no tensor-allocation method. Provider-backed tensor
/// allocation is available only from `TensorOutputSpecBuilder` returned for `Output<Tensor>` or a
/// descriptor-backed Tensor schema output.
class OutputSpecBuilder {
 public:
  /// @brief Declare one fixed nominal output rate.
  HOLOSCAN_CORE_EXPORT OutputSpecBuilder& nominal_rate(Rate rate);

  /// @brief Declare the forward lifecycle stage that selects this output's nominal rate.
  HOLOSCAN_CORE_EXPORT OutputSpecBuilder& lifecycle_nominal_rate(LifecycleStage bind_stage);

  /// @brief Declare the maximum selectable nominal average for a lifecycle-selected output.
  HOLOSCAN_CORE_EXPORT OutputSpecBuilder& max_nominal_rate(Rate rate);

  /// @brief Derive this output's nominal rate from one input owned by the same Operator.
  template <PublishedPayload U>
  OutputSpecBuilder& follows(const Input<U>& input, RateRatio ratio = {}) {
    return follows_input(input, ratio);
  }

  /// @brief Set the maximum publications admitted from one compute activation.
  /// @param count Positive finite burst bound; the declaration default is one.
  /// @return This builder for fluent cold-path configuration.
  OutputSpecBuilder& max_emits_per_compute(std::uint32_t count = 1) noexcept {
    max_emits_per_compute_ = count;
    return *this;
  }

  /// @brief Set optional retained history for late subscribers.
  /// @param depth Finite retained sample count; zero disables retained history.
  /// @return This builder for fluent cold-path configuration.
  OutputSpecBuilder& late_joiner_history_depth(std::uint32_t depth) noexcept {
    late_joiner_history_depth_ = depth;
    return *this;
  }

  /// @brief Record an alternate diagnostic display name.
  /// @param alias Display-only name copied into cold metadata.
  /// @return This builder for fluent cold-path configuration.
  /// @throws std::bad_alloc if storage for the copied alias cannot be allocated.
  /// @throws std::length_error if \p alias exceeds the implementation's maximum string size.
  OutputSpecBuilder& diagnostic_alias(std::string_view alias) {
    diagnostic_alias_.assign(alias);
    return *this;
  }

  /// @brief Record one normalized pass-through producer Tensor spec.
  /// @param spec Representation, finite bounds, and transfer promises.
  /// @return This builder for fluent cold-path configuration.
  /// @note Ordinary carrier outputs reject runtime-pool storage intent.
  OutputSpecBuilder& produces_tensor(const TensorOutputSpec& spec) {
    auto normalized = detail::normalize_tensor_output_spec(spec);
    if (!normalized) {
      if (tensor_contract_error_.code == ErrorCode::kSuccess) {
        tensor_contract_error_ = std::move(normalized.error());
      }
      return *this;
    }
    if (!tensor_contract_ && payload_options_ &&
        detail::payload_options_carry_tensor_transfer(*payload_options_)) {
      if (tensor_contract_error_.code == ErrorCode::kSuccess) {
        tensor_contract_error_ =
            Error{ErrorCode::kInvalidArgument, "GRAPH_TENSOR_TRANSFER_SOURCE_CONFLICT"};
      }
      return *this;
    }
    if (normalized->storage != TensorOutputStorage::kPassThrough) {
      if (tensor_contract_error_.code == ErrorCode::kSuccess) {
        tensor_contract_error_ =
            Error{ErrorCode::kInvalidArgument, "GRAPH_TENSOR_ALLOCATION_UNSUPPORTED"};
      }
      return *this;
    }
    tensor_layout_ = detail::tensor_representation(*normalized);
    std::optional<SchemaIdentityRecord> schema;
    if (payload_options_) {
      schema = std::move(payload_options_->schema);
    }
    payload_options_.reset();
    if (normalized->transfer || schema) {
      PayloadOptionsRecord options{.schema = std::move(schema)};
      if (normalized->transfer) {
        const auto& transfer = detail::normalized_tensor_transfer(*normalized);
        options.max_inline_bytes =
            transfer.max_inline_bytes
                ? std::optional{static_cast<std::uint32_t>(*transfer.max_inline_bytes)}
                : std::nullopt;
        options.max_tensor_byte_span = transfer.max_staging_byte_span;
        options.interface_tensor_emit_policy = transfer.foreign_emit_policy;
      }
      payload_options_ = std::move(options);
    }
    tensor_contract_ = std::move(*normalized);
    return *this;
  }

  /// @brief Record schema identity and payload options carried by this output.
  /// @param options Complete payload options copied into this builder.
  /// @return This builder for fluent cold-path configuration.
  OutputSpecBuilder& payload_options(PayloadOptions options) {
    PayloadOptionsRecord copied = detail::copy_payload_options(options);
    if (tensor_contract_ && detail::payload_options_carry_tensor_transfer(copied)) {
      if (tensor_contract_error_.code == ErrorCode::kSuccess) {
        tensor_contract_error_ =
            Error{ErrorCode::kInvalidArgument, "GRAPH_TENSOR_TRANSFER_SOURCE_CONFLICT"};
      }
      return *this;
    }
    if (tensor_contract_ && payload_options_) {
      copied.max_inline_bytes = payload_options_->max_inline_bytes;
      copied.max_tensor_byte_span = payload_options_->max_tensor_byte_span;
      copied.interface_tensor_emit_policy = payload_options_->interface_tensor_emit_policy;
    }
    payload_options_ = std::move(copied);
    return *this;
  }

 private:
  HOLOSCAN_CORE_EXPORT OutputSpecBuilder& follows_input(const InputBase& input, RateRatio ratio);

  std::uint32_t max_emits_per_compute_{1};
  std::uint32_t late_joiner_history_depth_{};
  std::string diagnostic_alias_;
  std::optional<TensorRepresentation> tensor_layout_;
  std::optional<PayloadOptionsRecord> payload_options_;
  std::optional<TensorContractRecord> tensor_contract_;
  Error tensor_contract_error_{ErrorCode::kSuccess};
  detail::AuthoredPortRate rate_;

  friend class OperatorSpec;
  friend class detail::OperatorSpecCompilerAccess;
  friend class detail::SyntheticOperatorBinding;
};

/// @brief Configure a Tensor or descriptor-backed Tensor schema output.
///
/// Every common output method preserves this specialized fluent type. Runtime-pool intent and its
/// finite bounds are declared together in TensorOutputSpec; compilation admits and
/// materializes physical placement in a later component.
class TensorOutputSpecBuilder {
 public:
  /// @brief Declare one fixed nominal output rate.
  HOLOSCAN_CORE_EXPORT TensorOutputSpecBuilder& nominal_rate(Rate rate);

  /// @brief Declare the forward lifecycle stage that selects this output's nominal rate.
  HOLOSCAN_CORE_EXPORT TensorOutputSpecBuilder& lifecycle_nominal_rate(LifecycleStage bind_stage);

  /// @brief Declare the maximum selectable nominal average for a lifecycle-selected output.
  HOLOSCAN_CORE_EXPORT TensorOutputSpecBuilder& max_nominal_rate(Rate rate);

  /// @brief Derive this output's nominal rate from one input owned by the same Operator.
  template <PublishedPayload U>
  TensorOutputSpecBuilder& follows(const Input<U>& input, RateRatio ratio = {}) {
    return follows_input(input, ratio);
  }

  /// @brief Set the maximum publications admitted from one compute activation.
  /// @param count Positive finite burst bound; the declaration default is one.
  /// @return This specialized builder.
  TensorOutputSpecBuilder& max_emits_per_compute(std::uint32_t count = 1) noexcept {
    max_emits_per_compute_ = count;
    return *this;
  }

  /// @brief Set optional retained history for late subscribers.
  /// @param depth Finite retained sample count; zero disables retained history.
  /// @return This specialized builder.
  TensorOutputSpecBuilder& late_joiner_history_depth(std::uint32_t depth) noexcept {
    late_joiner_history_depth_ = depth;
    return *this;
  }

  /// @brief Record an alternate diagnostic display name.
  /// @param alias Display-only name copied into cold metadata.
  /// @return This specialized builder.
  /// @throws std::bad_alloc if storage for the copied alias cannot be allocated.
  /// @throws std::length_error if \p alias exceeds the implementation's maximum string size.
  TensorOutputSpecBuilder& diagnostic_alias(std::string_view alias) {
    diagnostic_alias_.assign(alias);
    return *this;
  }

  /// @brief Record one normalized producer Tensor spec.
  /// @param spec Representation, finite bounds, storage intent, and transfer policy.
  /// @return This specialized builder.
  // Contract validation is deliberately centralized so every builder overload records the same
  // first error and leaves the partially authored specification unchanged.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  TensorOutputSpecBuilder& produces_tensor(const TensorOutputSpec& spec) {
    if (tensor_byte_span_expression_ || tensor_memory_kind_expression_) {
      if (tensor_contract_error_.code == ErrorCode::kSuccess) {
        tensor_contract_error_ =
            Error{ErrorCode::kInvalidArgument,
                  tensor_memory_kind_expression_ ? "GRAPH_TENSOR_MEMORY_KIND_SOURCE_CONFLICT"
                                                 : "GRAPH_TENSOR_BYTE_SPAN_SOURCE_CONFLICT"};
      }
      return *this;
    }
    auto normalized = detail::normalize_tensor_output_spec(spec);
    if (!normalized) {
      if (tensor_contract_error_.code == ErrorCode::kSuccess) {
        tensor_contract_error_ = std::move(normalized.error());
      }
      return *this;
    }
    if (!tensor_contract_ && payload_options_ &&
        detail::payload_options_carry_tensor_transfer(*payload_options_)) {
      if (tensor_contract_error_.code == ErrorCode::kSuccess) {
        tensor_contract_error_ =
            Error{ErrorCode::kInvalidArgument, "GRAPH_TENSOR_TRANSFER_SOURCE_CONFLICT"};
      }
      return *this;
    }
    tensor_layout_ = detail::tensor_representation(*normalized);
    if (normalized->storage == TensorOutputStorage::kRuntimePool) {
      tensor_allocation_ = detail::normalized_tensor_pool_bounds(*normalized);
      has_tensor_allocation_ = true;
    } else {
      tensor_allocation_ = {};
      has_tensor_allocation_ = false;
    }
    std::optional<SchemaIdentityRecord> schema;
    if (payload_options_) {
      schema = std::move(payload_options_->schema);
    }
    payload_options_.reset();
    if (normalized->transfer || schema) {
      PayloadOptionsRecord options{.schema = std::move(schema)};
      std::optional<std::uint64_t> schema_tensor_staging_byte_span;
      if (normalized->transfer) {
        const auto& transfer = detail::normalized_tensor_transfer(*normalized);
        options.max_inline_bytes =
            transfer.max_inline_bytes
                ? std::optional{static_cast<std::uint32_t>(*transfer.max_inline_bytes)}
                : std::nullopt;
        options.interface_tensor_emit_policy = transfer.foreign_emit_policy;
        if (descriptor_tensor_schema_) {
          schema_tensor_staging_byte_span = transfer.max_staging_byte_span;
        }
      }
      options.max_tensor_byte_span = schema_tensor_staging_byte_span;
      payload_options_ = std::move(options);
    }
    tensor_contract_ = std::move(*normalized);
    return *this;
  }

  /// @brief Record a runtime-pool Tensor spec with compile-derived slot byte capacity.
  /// @param spec Runtime-pool representation, rank capacity, and optional transfer policy.
  /// @param byte_span Immutable scalar expression resolved from generation-zero dependencies.
  /// @return This specialized builder.
  /// @note A literal maximum byte span and a derived expression are mutually exclusive. Changing
  /// a recompile-required Parameter requires compiling a new plan, not rebuilding the binary.
  TensorOutputSpecBuilder& produces_tensor(const TensorOutputSpec& spec,
                                           TensorByteSpanExpression byte_span) {
    return produces_tensor(spec, TensorOutputDerivations{.max_byte_span = std::move(byte_span)});
  }

  /// @brief Record a runtime-pool Tensor spec with compile-derived allocation facts.
  /// @param spec Runtime-pool representation, finite bounds, and optional transfer policy.
  /// @param derivations Named output facts resolved from generation-zero Parameters.
  /// @return This specialized builder.
  /// @note Each derivation is the sole authoring source for its fact. Changing a
  /// recompile-required Parameter requires compiling a new plan.
  // Derived-fact validation shares the base contract path to preserve its first-error semantics.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  TensorOutputSpecBuilder& produces_tensor(const TensorOutputSpec& spec,
                                           TensorOutputDerivations derivations) {
    const bool derive_byte_span = derivations.max_byte_span.has_value();
    const bool derive_memory_kind = derivations.memory_kind.has_value();
    if (!derive_byte_span && !derive_memory_kind) {
      record_contract_error("GRAPH_TENSOR_DERIVATION_INVALID");
      return *this;
    }
    if (descriptor_tensor_schema_) {
      record_contract_error(derive_memory_kind ? "GRAPH_TENSOR_MEMORY_KIND_DERIVATION_INVALID"
                                               : "GRAPH_TENSOR_BYTE_SPAN_DERIVATION_INVALID");
      return *this;
    }
    if (tensor_contract_ || tensor_byte_span_expression_ || tensor_memory_kind_expression_) {
      record_contract_error(derive_memory_kind ? "GRAPH_TENSOR_MEMORY_KIND_SOURCE_CONFLICT"
                                               : "GRAPH_TENSOR_BYTE_SPAN_SOURCE_CONFLICT");
      return *this;
    }
    if (derive_memory_kind && spec.storage != TensorOutputStorage::kRuntimePool) {
      record_contract_error("GRAPH_TENSOR_MEMORY_KIND_DERIVATION_INVALID");
      return *this;
    }
    if (derive_byte_span && (derivations.max_byte_span->invalid_reason_ !=
                                 TensorByteSpanExpressionInvalidReason::kNone ||
                             derivations.max_byte_span->nodes_.empty())) {
      record_contract_error(derivations.max_byte_span->invalid_reason_ ==
                                    TensorByteSpanExpressionInvalidReason::kProjectionInvalid
                                ? "GRAPH_TENSOR_BYTE_SPAN_PROJECTION_INVALID"
                                : "GRAPH_TENSOR_BYTE_SPAN_DERIVATION_INVALID");
      return *this;
    }
    if (derive_memory_kind && (derivations.memory_kind->invalid_reason_ !=
                                   TensorMemoryKindExpressionInvalidReason::kNone ||
                               !derivations.memory_kind->parameter_)) {
      record_contract_error("GRAPH_TENSOR_MEMORY_KIND_DERIVATION_INVALID");
      return *this;
    }
    if (payload_options_ && detail::payload_options_carry_tensor_transfer(*payload_options_)) {
      record_contract_error("GRAPH_TENSOR_TRANSFER_SOURCE_CONFLICT");
      return *this;
    }
    auto normalized = detail::normalize_tensor_output_spec_with_derivations(
        spec, derive_byte_span, derive_memory_kind);
    if (!normalized) {
      record_contract_error(std::move(normalized.error()));
      return *this;
    }
    tensor_layout_ = detail::tensor_representation(*normalized);
    if (derive_byte_span) {
      tensor_allocation_ = {};
      has_tensor_allocation_ = false;
    } else {
      tensor_allocation_ = detail::normalized_tensor_pool_bounds(*normalized);
      has_tensor_allocation_ = true;
    }
    std::optional<SchemaIdentityRecord> schema;
    if (payload_options_) {
      schema = std::move(payload_options_->schema);
    }
    payload_options_.reset();
    if (normalized->transfer || schema) {
      PayloadOptionsRecord options{.schema = std::move(schema)};
      if (normalized->transfer) {
        const auto& transfer = detail::normalized_tensor_transfer(*normalized);
        options.max_inline_bytes =
            transfer.max_inline_bytes
                ? std::optional{static_cast<std::uint32_t>(*transfer.max_inline_bytes)}
                : std::nullopt;
        options.interface_tensor_emit_policy = transfer.foreign_emit_policy;
      }
      payload_options_ = std::move(options);
    }
    tensor_contract_ = std::move(*normalized);
    tensor_byte_span_expression_ = std::move(derivations.max_byte_span);
    tensor_memory_kind_expression_ = std::move(derivations.memory_kind);
    return *this;
  }

  /// @brief Record schema identity and payload options carried by this output.
  /// @param options Complete payload options copied into this builder.
  /// @return This specialized builder.
  TensorOutputSpecBuilder& payload_options(PayloadOptions options) {
    PayloadOptionsRecord copied = detail::copy_payload_options(options);
    if (tensor_contract_ && detail::payload_options_carry_tensor_transfer(copied)) {
      if (tensor_contract_error_.code == ErrorCode::kSuccess) {
        tensor_contract_error_ =
            Error{ErrorCode::kInvalidArgument, "GRAPH_TENSOR_TRANSFER_SOURCE_CONFLICT"};
      }
      return *this;
    }
    if (tensor_contract_ && payload_options_) {
      copied.max_inline_bytes = payload_options_->max_inline_bytes;
      copied.max_tensor_byte_span = payload_options_->max_tensor_byte_span;
      copied.interface_tensor_emit_policy = payload_options_->interface_tensor_emit_policy;
    }
    payload_options_ = std::move(copied);
    return *this;
  }

  /// @brief Select one Tensor member for field-scoped output declarations.
  /// @tparam SchemaT Owning generated descriptor-backed schema type.
  /// @param selector Generated schema-typed field token.
  /// @return Field-scoped builder for logical, materialization, and allocation facts.
  /// @warning The returned non-owning builder must not outlive this parent builder.
  template <typename SchemaT>
  [[nodiscard]] inline TensorFieldOutputSpecBuilder tensor_field(
      TensorFieldSelector<SchemaT> selector);

  /// @brief Declare one Tensor field from a direction-specific output spec.
  /// @tparam SchemaT Owning generated descriptor-backed schema type.
  /// @param selector Generated schema-typed field token.
  /// @param spec Logical, allocation, and optional materialization facts.
  /// @return This output builder.
  /// @throws std::bad_alloc if field-record storage cannot be allocated.
  template <typename SchemaT>
  TensorOutputSpecBuilder& tensor_field(TensorFieldSelector<SchemaT> selector,
                                        const TensorOutputSpec& spec);

  /// @brief Declare one Tensor field with compile-derived slot byte capacity.
  /// @tparam SchemaT Owning generated descriptor-backed schema type.
  /// @param selector Generated schema-typed field token.
  /// @param spec Runtime-pool representation, rank capacity, and transfer policy.
  /// @param byte_span Immutable scalar expression resolved from generation-zero dependencies.
  /// @return This output builder.
  template <typename SchemaT>
  TensorOutputSpecBuilder& tensor_field(TensorFieldSelector<SchemaT> selector,
                                        const TensorOutputSpec& spec,
                                        TensorByteSpanExpression byte_span);

  /// @brief Declare one Tensor field with compile-derived allocation facts.
  /// @tparam SchemaT Owning generated descriptor-backed schema type.
  /// @param selector Generated schema-typed field token.
  /// @param spec Runtime-pool representation, rank/capacity bounds, and transfer policy.
  /// @param derivations Named field facts resolved from generation-zero Parameters.
  /// @return This output builder.
  template <typename SchemaT>
  TensorOutputSpecBuilder& tensor_field(TensorFieldSelector<SchemaT> selector,
                                        const TensorOutputSpec& spec,
                                        TensorOutputDerivations derivations);

  /// @brief Apply one output spec to a statically listed set of Tensor fields.
  /// @tparam SchemaT Owning generated descriptor-backed schema type.
  /// @tparam N Number of listed selectors.
  /// @param selectors Generated schema-typed field tokens.
  /// @param spec Shared logical, allocation, and optional materialization facts.
  /// @return This output builder.
  /// @throws std::bad_alloc if field-record storage cannot be allocated.
  template <typename SchemaT, std::size_t N>
  TensorOutputSpecBuilder& tensor_fields(
      const std::array<TensorFieldSelector<SchemaT>, N>& selectors, const TensorOutputSpec& spec);

  /// @brief Declare a complete heterogeneous common output field inventory.
  template <typename SchemaT, std::size_t N>
  TensorOutputSpecBuilder& tensor_fields(
      const std::array<std::pair<TensorFieldSelector<SchemaT>, TensorOutputSpec>, N>& fields);

 private:
  HOLOSCAN_CORE_EXPORT TensorOutputSpecBuilder& follows_input(const InputBase& input,
                                                              RateRatio ratio);

  void record_contract_error(std::string_view token) {
    record_contract_error(Error{ErrorCode::kInvalidArgument, std::string{token}});
  }

  void record_contract_error(Error error) {
    if (tensor_contract_error_.code == ErrorCode::kSuccess) {
      tensor_contract_error_ = std::move(error);
    }
  }

  std::uint32_t max_emits_per_compute_{1};
  std::uint32_t late_joiner_history_depth_{};
  std::string diagnostic_alias_;
  std::optional<TensorRepresentation> tensor_layout_;
  std::optional<PayloadOptionsRecord> payload_options_;
  TensorPoolBounds tensor_allocation_{};
  bool has_tensor_allocation_{};
  std::optional<TensorContractRecord> tensor_contract_;
  std::optional<TensorByteSpanExpression> tensor_byte_span_expression_;
  std::optional<TensorMemoryKindExpression> tensor_memory_kind_expression_;
  Error tensor_contract_error_{ErrorCode::kSuccess};
  std::uint16_t descriptor_tensor_field_count_{};
  bool descriptor_tensor_schema_{};
  std::vector<detail::AuthoredTensorFieldConfiguration> tensor_fields_;
  detail::AuthoredPortRate rate_;

  friend class OperatorSpec;
  friend class TensorFieldOutputSpecBuilder;
  friend class detail::OperatorSpecCompilerAccess;
  friend class detail::SyntheticOperatorBinding;
};

/// @brief Configure one selected Tensor field on an input schema port.
class TensorFieldInputSpecBuilder {
 public:
  TensorFieldInputSpecBuilder(const TensorFieldInputSpecBuilder&) = delete;
  TensorFieldInputSpecBuilder& operator=(const TensorFieldInputSpecBuilder&) = delete;
  TensorFieldInputSpecBuilder(TensorFieldInputSpecBuilder&&) = delete;
  TensorFieldInputSpecBuilder& operator=(TensorFieldInputSpecBuilder&&) = delete;

  /// @brief Record a normalized consumer spec for the selected field.
  /// @param spec Representation, finite bounds, and conversion requirements.
  /// @return This field-scoped builder.
  TensorFieldInputSpecBuilder& expects_tensor(const TensorInputSpec& spec) {
    auto normalized = detail::normalize_tensor_input_spec(spec);
    if (!normalized) {
      if (parent_->tensor_contract_error_.code == ErrorCode::kSuccess) {
        parent_->tensor_contract_error_ =
            detail::tensor_field_contract_error(std::move(normalized.error()), record());
      }
      return *this;
    }
    record().tensor_layout = detail::tensor_representation(*normalized);
    record().memory_conversion = normalized->memory_conversion;
    record().tensor_contract = std::move(*normalized);
    return *this;
  }

 private:
  TensorFieldInputSpecBuilder(InputSpecBuilder* parent, std::size_t index) noexcept
      : parent_(parent), index_(index) {}

  TensorFieldConfigurationRecord& record() noexcept {
    return parent_->tensor_fields_[index_].configuration;
  }

  InputSpecBuilder* parent_{};
  std::size_t index_{};

  friend class InputSpecBuilder;
};

/// @brief Configure one selected Tensor field on an output schema port.
class TensorFieldOutputSpecBuilder {
 public:
  TensorFieldOutputSpecBuilder(const TensorFieldOutputSpecBuilder&) = delete;
  TensorFieldOutputSpecBuilder& operator=(const TensorFieldOutputSpecBuilder&) = delete;
  TensorFieldOutputSpecBuilder(TensorFieldOutputSpecBuilder&&) = delete;
  TensorFieldOutputSpecBuilder& operator=(TensorFieldOutputSpecBuilder&&) = delete;

  /// @brief Record a normalized producer spec for the selected field.
  /// @param spec Representation, finite bounds, storage intent, and transfer policy.
  /// @return This field-scoped builder.
  TensorFieldOutputSpecBuilder& produces_tensor(const TensorOutputSpec& spec) {
    if (authored_record().tensor_byte_span_expression ||
        authored_record().tensor_memory_kind_expression) {
      record_error(authored_record().tensor_memory_kind_expression
                       ? "GRAPH_TENSOR_MEMORY_KIND_SOURCE_CONFLICT"
                       : "GRAPH_TENSOR_BYTE_SPAN_SOURCE_CONFLICT");
      return *this;
    }
    auto normalized = detail::normalize_tensor_output_spec(spec);
    if (!normalized) {
      if (parent_->tensor_contract_error_.code == ErrorCode::kSuccess) {
        parent_->tensor_contract_error_ =
            detail::tensor_field_contract_error(std::move(normalized.error()), record());
      }
      return *this;
    }
    record().tensor_layout = detail::tensor_representation(*normalized);
    if (normalized->storage == TensorOutputStorage::kRuntimePool) {
      record().tensor_allocation = detail::normalized_tensor_pool_bounds(*normalized);
    } else {
      record().tensor_allocation.reset();
    }
    record().payload_options.reset();
    if (normalized->transfer) {
      const auto& transfer = detail::normalized_tensor_transfer(*normalized);
      record().payload_options = TensorFieldTransferRecord{
          .max_inline_bytes = transfer.max_inline_bytes,
          .max_tensor_byte_span = transfer.max_staging_byte_span,
          .interface_tensor_emit_policy = transfer.foreign_emit_policy,
      };
    }
    record().tensor_contract = std::move(*normalized);
    return *this;
  }

  /// @brief Record a runtime-pool field spec with compile-derived slot byte capacity.
  /// @param spec Runtime-pool representation, rank capacity, and optional transfer policy.
  /// @param byte_span Immutable scalar expression resolved from generation-zero dependencies.
  /// @return This field-scoped builder.
  TensorFieldOutputSpecBuilder& produces_tensor(const TensorOutputSpec& spec,
                                                TensorByteSpanExpression byte_span) {
    return produces_tensor(spec, TensorOutputDerivations{.max_byte_span = std::move(byte_span)});
  }

  /// @brief Record a runtime-pool field spec with compile-derived allocation facts.
  /// @param spec Runtime-pool representation, finite bounds, and optional transfer policy.
  /// @param derivations Named field facts resolved from generation-zero Parameters.
  /// @return This field-scoped builder.
  TensorFieldOutputSpecBuilder& produces_tensor(const TensorOutputSpec& spec,
                                                TensorOutputDerivations derivations) {
    const bool derive_byte_span = derivations.max_byte_span.has_value();
    const bool derive_memory_kind = derivations.memory_kind.has_value();
    if (!derive_byte_span && !derive_memory_kind) {
      record_error("GRAPH_TENSOR_DERIVATION_INVALID");
      return *this;
    }
    if (record().tensor_contract || authored_record().tensor_byte_span_expression ||
        authored_record().tensor_memory_kind_expression) {
      record_error(derive_memory_kind ? "GRAPH_TENSOR_MEMORY_KIND_SOURCE_CONFLICT"
                                      : "GRAPH_TENSOR_BYTE_SPAN_SOURCE_CONFLICT");
      return *this;
    }
    if (derive_memory_kind && spec.storage != TensorOutputStorage::kRuntimePool) {
      record_error("GRAPH_TENSOR_MEMORY_KIND_DERIVATION_INVALID");
      return *this;
    }
    if (derive_byte_span && (derivations.max_byte_span->invalid_reason_ !=
                                 TensorByteSpanExpressionInvalidReason::kNone ||
                             derivations.max_byte_span->nodes_.empty())) {
      record_error(derivations.max_byte_span->invalid_reason_ ==
                           TensorByteSpanExpressionInvalidReason::kProjectionInvalid
                       ? "GRAPH_TENSOR_BYTE_SPAN_PROJECTION_INVALID"
                       : "GRAPH_TENSOR_BYTE_SPAN_DERIVATION_INVALID");
      return *this;
    }
    if (derive_memory_kind && (derivations.memory_kind->invalid_reason_ !=
                                   TensorMemoryKindExpressionInvalidReason::kNone ||
                               !derivations.memory_kind->parameter_)) {
      record_error("GRAPH_TENSOR_MEMORY_KIND_DERIVATION_INVALID");
      return *this;
    }
    auto normalized = detail::normalize_tensor_output_spec_with_derivations(
        spec, derive_byte_span, derive_memory_kind);
    if (!normalized) {
      if (parent_->tensor_contract_error_.code == ErrorCode::kSuccess) {
        parent_->tensor_contract_error_ =
            detail::tensor_field_contract_error(std::move(normalized.error()), record());
      }
      return *this;
    }
    record().tensor_layout = detail::tensor_representation(*normalized);
    if (derive_byte_span) {
      record().tensor_allocation.reset();
    } else {
      record().tensor_allocation = detail::normalized_tensor_pool_bounds(*normalized);
    }
    record().payload_options.reset();
    if (normalized->transfer) {
      const auto& transfer = detail::normalized_tensor_transfer(*normalized);
      record().payload_options = TensorFieldTransferRecord{
          .max_inline_bytes = transfer.max_inline_bytes,
          .max_tensor_byte_span = transfer.max_staging_byte_span,
          .interface_tensor_emit_policy = transfer.foreign_emit_policy,
      };
    }
    record().tensor_contract = std::move(*normalized);
    authored_record().tensor_byte_span_expression = std::move(derivations.max_byte_span);
    authored_record().tensor_memory_kind_expression = std::move(derivations.memory_kind);
    return *this;
  }

 private:
  TensorFieldOutputSpecBuilder(TensorOutputSpecBuilder* parent, std::size_t index) noexcept
      : parent_(parent), index_(index) {}

  TensorFieldConfigurationRecord& record() noexcept {
    return parent_->tensor_fields_[index_].configuration;
  }

  detail::AuthoredTensorFieldConfiguration& authored_record() noexcept {
    return parent_->tensor_fields_[index_];
  }

  void record_error(std::string_view token) {
    if (parent_->tensor_contract_error_.code == ErrorCode::kSuccess) {
      parent_->tensor_contract_error_ = detail::tensor_field_contract_error(
          Error{ErrorCode::kInvalidArgument, std::string{token}}, record());
    }
  }

  TensorOutputSpecBuilder* parent_{};
  std::size_t index_{};

  friend class TensorOutputSpecBuilder;
};

template <typename SchemaT>
TensorFieldInputSpecBuilder InputSpecBuilder::tensor_field(
    const TensorFieldSelector<SchemaT> selector) {
  constexpr bool descriptor_schema = detail::HasDescriptorTensorSchemaNativeTraits<SchemaT>;
  const bool selector_valid = [&]() {
    if constexpr (descriptor_schema) {
      return detail::descriptor_tensor_schema_field_selector_matches<SchemaT>(
          selector.field_id, selector.tensor_index);
    } else {
      return false;
    }
  }();
  tensor_fields_.push_back(detail::AuthoredTensorFieldConfiguration{
      .configuration =
          TensorFieldConfigurationRecord{
              .field_id = selector.field_id,
              .tensor_index = selector.tensor_index,
          },
      .schema_type_token = &detail::kOperatorSpecLocalTypeToken<SchemaT>,
      .selector_valid = selector_valid,
  });
  return TensorFieldInputSpecBuilder{this, tensor_fields_.size() - 1U};
}

template <typename SchemaT>
InputSpecBuilder& InputSpecBuilder::tensor_field(const TensorFieldSelector<SchemaT> selector,
                                                 const TensorInputSpec& spec) {
  auto field_builder = tensor_field(selector);
  field_builder.expects_tensor(spec);
  return *this;
}

template <typename SchemaT, std::size_t N>
InputSpecBuilder& InputSpecBuilder::tensor_fields(
    const std::array<TensorFieldSelector<SchemaT>, N>& selectors, const TensorInputSpec& spec) {
  for (const auto selector : selectors) {
    tensor_field(selector, spec);
  }
  return *this;
}

template <typename SchemaT, std::size_t N>
InputSpecBuilder& InputSpecBuilder::tensor_fields(
    const std::array<std::pair<TensorFieldSelector<SchemaT>, TensorInputSpec>, N>& fields) {
  static_assert(detail::HasDescriptorTensorSchemaNativeTraits<SchemaT>,
                "Tensor field specs require descriptor Tensor schema traits");
  static_assert(N <= detail::descriptor_tensor_schema_tensor_count_v<SchemaT>,
                "aggregate input specs cannot exceed the schema Tensor field count");
  for (const auto& [field, spec] : fields) {
    tensor_field(field, spec);
  }
  return *this;
}

template <typename SchemaT>
TensorFieldOutputSpecBuilder TensorOutputSpecBuilder::tensor_field(
    const TensorFieldSelector<SchemaT> selector) {
  constexpr bool descriptor_schema = detail::HasDescriptorTensorSchemaNativeTraits<SchemaT>;
  const bool selector_valid = [&]() {
    if constexpr (descriptor_schema) {
      return detail::descriptor_tensor_schema_field_selector_matches<SchemaT>(
          selector.field_id, selector.tensor_index);
    } else {
      return false;
    }
  }();
  tensor_fields_.push_back(detail::AuthoredTensorFieldConfiguration{
      .configuration =
          TensorFieldConfigurationRecord{
              .field_id = selector.field_id,
              .tensor_index = selector.tensor_index,
          },
      .schema_type_token = &detail::kOperatorSpecLocalTypeToken<SchemaT>,
      .selector_valid = selector_valid,
  });
  return TensorFieldOutputSpecBuilder{this, tensor_fields_.size() - 1U};
}

template <typename SchemaT>
TensorOutputSpecBuilder& TensorOutputSpecBuilder::tensor_field(
    const TensorFieldSelector<SchemaT> selector, const TensorOutputSpec& spec) {
  auto field_builder = tensor_field(selector);
  field_builder.produces_tensor(spec);
  return *this;
}

template <typename SchemaT>
TensorOutputSpecBuilder& TensorOutputSpecBuilder::tensor_field(
    const TensorFieldSelector<SchemaT> selector, const TensorOutputSpec& spec,
    TensorByteSpanExpression byte_span) {
  auto field_builder = tensor_field(selector);
  field_builder.produces_tensor(spec, std::move(byte_span));
  return *this;
}

template <typename SchemaT>
TensorOutputSpecBuilder& TensorOutputSpecBuilder::tensor_field(
    const TensorFieldSelector<SchemaT> selector, const TensorOutputSpec& spec,
    TensorOutputDerivations derivations) {
  auto field_builder = tensor_field(selector);
  field_builder.produces_tensor(spec, std::move(derivations));
  return *this;
}

template <typename SchemaT, std::size_t N>
TensorOutputSpecBuilder& TensorOutputSpecBuilder::tensor_fields(
    const std::array<TensorFieldSelector<SchemaT>, N>& selectors, const TensorOutputSpec& spec) {
  for (const auto selector : selectors) {
    tensor_field(selector, spec);
  }
  return *this;
}

template <typename SchemaT, std::size_t N>
TensorOutputSpecBuilder& TensorOutputSpecBuilder::tensor_fields(
    const std::array<std::pair<TensorFieldSelector<SchemaT>, TensorOutputSpec>, N>& fields) {
  static_assert(detail::HasDescriptorTensorSchemaNativeTraits<SchemaT>,
                "Tensor field specs require descriptor Tensor schema traits");
  static_assert(N == detail::descriptor_tensor_schema_tensor_count_v<SchemaT>,
                "aggregate output specs require one entry per schema Tensor field");
  for (const auto& [field, spec] : fields) {
    tensor_field(field, spec);
  }
  return *this;
}

/// @brief Configure a capacity-one readable or spin readiness declaration.
class SingleWitnessSourceSpecBuilder {
 public:
  /// @brief Set the required effective capacity.
  /// @param count Must equal one; admission rejects every other value.
  /// @return This builder.
  SingleWitnessSourceSpecBuilder& capacity(std::uint32_t count = 1) noexcept {
    capacity_ = count;
    return *this;
  }

  /// @brief Select the source reference clock.
  /// @param key Owned cold clock key; omission resolves to the graph default later.
  /// @return This builder.
  SingleWitnessSourceSpecBuilder& clock(ClockKey key) {
    clock_ = std::move(key);
    return *this;
  }

 private:
  std::uint32_t capacity_{1};
  ClockKey clock_{};

  friend class OperatorSpec;
  friend class detail::SyntheticOperatorBinding;
  friend class detail::OperatorSpecCompilerAccess;
};

/// @brief Configure a positive finite counted doorbell declaration.
class DoorbellSourceSpecBuilder {
 public:
  /// @brief Set accepted-through-retirement witness capacity.
  /// @param count Positive finite capacity.
  /// @return This builder.
  DoorbellSourceSpecBuilder& capacity(std::uint32_t count = 1) noexcept {
    capacity_ = count;
    return *this;
  }

  /// @brief Select the source reference clock.
  /// @param key Owned cold clock key; omission resolves to the graph default later.
  /// @return This builder.
  DoorbellSourceSpecBuilder& clock(ClockKey key) {
    clock_ = std::move(key);
    return *this;
  }

 private:
  std::uint32_t capacity_{1};
  ClockKey clock_{};

  friend class OperatorSpec;
  friend class detail::SyntheticOperatorBinding;
  friend class detail::OperatorSpecCompilerAccess;
};

/// @brief Configure a positive finite notification declaration and sender-reference budget.
class NotificationSourceSpecBuilder {
 public:
  /// @brief Set accepted-through-retirement witness capacity.
  /// @param count Positive finite witness capacity.
  /// @return This builder.
  NotificationSourceSpecBuilder& capacity(std::uint32_t count = 1) noexcept {
    capacity_ = count;
    return *this;
  }

  /// @brief Set the maximum admitted number of live sender references.
  /// @param count Positive finite reference bound independent of witness capacity.
  /// @return This builder.
  NotificationSourceSpecBuilder& sender_reference_capacity(std::uint32_t count) noexcept {
    sender_reference_capacity_ = count;
    return *this;
  }

  /// @brief Select the source reference clock.
  /// @param key Owned cold clock key; omission resolves to the graph default later.
  /// @return This builder.
  NotificationSourceSpecBuilder& clock(ClockKey key) {
    clock_ = std::move(key);
    return *this;
  }

 private:
  std::uint32_t capacity_{1};
  std::uint32_t sender_reference_capacity_{1};
  ClockKey clock_{};

  friend class OperatorSpec;
  friend class detail::SyntheticOperatorBinding;
  friend class detail::OperatorSpecCompilerAccess;
};

/// @brief Collect one side-effect-free Operator declaration pass.
///
/// `setup()` binds stable public member declarations by reference. A fresh spec is used for every
/// declaration pass, so ports retain no registration state. Builder references remain stable for
/// the spec lifetime. This cold object may allocate and is not available from `compute()`.
class HOLOSCAN_CORE_EXPORT OperatorSpec {
 public:
  /// @brief Construct an empty declaration pass.
  /// @throws std::bad_alloc if cold declaration storage cannot be allocated.
  OperatorSpec() : lifecycle_builder_(this) {}

  /// @brief Reject copying one declaration pass and its stable builder storage.
  /// @param other Declaration pass that would otherwise be copied.
  OperatorSpec(const OperatorSpec& other) = delete;

  /// @brief Reject copy assignment of one declaration pass.
  /// @param other Declaration pass that would otherwise replace this one.
  OperatorSpec& operator=(const OperatorSpec& other) = delete;

  /// @brief Reject moving one declaration pass while builder references may exist.
  /// @param other Declaration pass that would otherwise be moved.
  OperatorSpec(OperatorSpec&& other) = delete;

  /// @brief Reject move assignment of one declaration pass.
  /// @param other Declaration pass that would otherwise replace this one.
  OperatorSpec& operator=(OperatorSpec&& other) = delete;

  /// @brief Destroy cold declaration records after setup consumers release their references.
  /// @note Defined out of line so private port storage has one Core-owned destruction ABI.
  ~OperatorSpec() noexcept;

  /// @brief Bind one typed input using its owned member name.
  /// @tparam T Admitted published payload type.
  /// @param input Stable Operator member declaration.
  /// @return Stable builder reference for this declaration pass.
  template <PublishedPayload T>
  InputSpecBuilder& input(Input<T>& input) {
    admit_schema_package<T>();
    auto& builder = bind_input(input,
                               input.name(),
                               &detail::kOperatorSpecLocalTypeToken<T>,
                               detail::operator_spec_payload_identity<T>(),
                               detail::runtime_port_type_ops<T>(),
                               &typeid(T));
    builder.descriptor_tensor_field_count_ =
        detail::operator_spec_descriptor_tensor_field_count<T>();
    return builder;
  }

  /// @brief Bind one typed input with an explicit local name.
  /// @tparam T Admitted published payload type.
  /// @param input Stable Operator member declaration.
  /// @param name Local declaration name copied into spec storage.
  /// @return Stable builder reference for this declaration pass.
  template <PublishedPayload T>
  InputSpecBuilder& input(Input<T>& input, std::string_view name) {
    admit_schema_package<T>();
    auto& builder = bind_input(input,
                               name,
                               &detail::kOperatorSpecLocalTypeToken<T>,
                               detail::operator_spec_payload_identity<T>(),
                               detail::runtime_port_type_ops<T>(),
                               &typeid(T));
    builder.descriptor_tensor_field_count_ =
        detail::operator_spec_descriptor_tensor_field_count<T>();
    return builder;
  }

  /// @brief Bind one ordinary typed output using its owned member name.
  /// @tparam T Admitted non-Tensor payload type.
  /// @param output Stable Operator member declaration.
  /// @return Stable common output builder.
  template <PublishedPayload T>
    requires(!std::same_as<T, Tensor> && !detail::kRuntimeDescriptorTensorSchemaPayload<T>)
  OutputSpecBuilder& output(Output<T>& output) {
    admit_schema_package<T>();
    return bind_output(output,
                       output.name(),
                       &detail::kOperatorSpecLocalTypeToken<T>,
                       detail::operator_spec_payload_identity<T>(),
                       detail::runtime_port_type_ops<T>(),
                       &typeid(T));
  }

  /// @brief Bind one ordinary typed output with an explicit local name.
  /// @tparam T Admitted non-Tensor payload type.
  /// @param output Stable Operator member declaration.
  /// @param name Local declaration name copied into spec storage.
  /// @return Stable common output builder.
  template <PublishedPayload T>
    requires(!std::same_as<T, Tensor> && !detail::kRuntimeDescriptorTensorSchemaPayload<T>)
  OutputSpecBuilder& output(Output<T>& output, std::string_view name) {
    admit_schema_package<T>();
    return bind_output(output,
                       name,
                       &detail::kOperatorSpecLocalTypeToken<T>,
                       detail::operator_spec_payload_identity<T>(),
                       detail::runtime_port_type_ops<T>(),
                       &typeid(T));
  }

  /// @brief Bind one descriptor-backed Tensor schema output using its owned member name.
  /// @tparam T Admitted descriptor-backed Tensor schema payload.
  /// @param output Stable output declaration.
  /// @return Stable specialized Tensor output builder.
  template <PublishedPayload T>
    requires detail::kRuntimeDescriptorTensorSchemaPayload<T>
  TensorOutputSpecBuilder& output(Output<T>& output) {
    admit_schema_package<T>();
    auto& builder = bind_tensor_output(output,
                                       output.name(),
                                       &detail::kOperatorSpecLocalTypeToken<T>,
                                       detail::operator_spec_payload_identity<T>(),
                                       detail::runtime_port_type_ops<T>(),
                                       &typeid(T));
    builder.descriptor_tensor_field_count_ =
        detail::operator_spec_descriptor_tensor_field_count<T>();
    builder.descriptor_tensor_schema_ = true;
    return builder;
  }

  /// @brief Bind one descriptor-backed Tensor schema output with an explicit local name.
  /// @tparam T Admitted descriptor-backed Tensor schema payload.
  /// @param output Stable output declaration.
  /// @param name Local declaration name copied into spec storage.
  /// @return Stable specialized Tensor output builder.
  template <PublishedPayload T>
    requires detail::kRuntimeDescriptorTensorSchemaPayload<T>
  TensorOutputSpecBuilder& output(Output<T>& output, std::string_view name) {
    admit_schema_package<T>();
    auto& builder = bind_tensor_output(output,
                                       name,
                                       &detail::kOperatorSpecLocalTypeToken<T>,
                                       detail::operator_spec_payload_identity<T>(),
                                       detail::runtime_port_type_ops<T>(),
                                       &typeid(T));
    builder.descriptor_tensor_field_count_ =
        detail::operator_spec_descriptor_tensor_field_count<T>();
    builder.descriptor_tensor_schema_ = true;
    return builder;
  }

  /// @brief Bind a direct Tensor output using its owned member name.
  /// @param output Stable Tensor output declaration.
  /// @return Stable specialized Tensor output builder.
  TensorOutputSpecBuilder& output(Output<Tensor>& output) {
    admit_schema_package<Tensor>();
    auto& builder = bind_tensor_output(output,
                                       output.name(),
                                       &detail::kOperatorSpecLocalTypeToken<Tensor>,
                                       detail::operator_spec_payload_identity<Tensor>(),
                                       detail::runtime_port_type_ops<Tensor>(),
                                       &typeid(Tensor));
    builder.descriptor_tensor_field_count_ = 1U;
    return builder;
  }

  /// @brief Bind a direct Tensor output with an explicit local name.
  /// @param output Stable Tensor output declaration.
  /// @param name Local declaration name copied into spec storage.
  /// @return Stable specialized Tensor output builder.
  TensorOutputSpecBuilder& output(Output<Tensor>& output, std::string_view name) {
    admit_schema_package<Tensor>();
    auto& builder = bind_tensor_output(output,
                                       name,
                                       &detail::kOperatorSpecLocalTypeToken<Tensor>,
                                       detail::operator_spec_payload_identity<Tensor>(),
                                       detail::runtime_port_type_ops<Tensor>(),
                                       &typeid(Tensor));
    builder.descriptor_tensor_field_count_ = 1U;
    return builder;
  }

  /// @brief Declare whether `compute()` is eligible for CUDA Graph capture.
  /// @param enabled Whether this Operator is eligible; defaults to `true` for no-argument opt-in.
  /// @return This spec.
  /// @note An Operator is ineligible by default. Graph composition freezes this setup declaration
  /// into immutable identity. Eligibility alone has no execution effect unless the Operator is a
  /// member of a region constrained by `RegionConstraint::cuda_graph()`.
  /// @warning Capture invokes `compute()` once. CPU side effects from that invocation are not
  /// repeated during replay; required per-activation effects must be captured CUDA work.
  /// @note Every output feeding a CUDA Graph capture edge must declare a complete maximum Tensor
  /// shape. For that edge, CUDA Graph lowering treats the declared maximum shape as the exact
  /// capture and replay shape; a different Tensor loan shape is rejected. Ordinary Tensor edges
  /// continue to treat the same declaration as an upper bound. All captured CUDA work must use
  /// capture-compatible operations enqueued on `ExecutionContext::cuda_stream()`. During capture
  /// this is a private stream in the selected source CUDA context; replay uses the
  /// compiler-selected source stream.
  /// @note Captured Tensor outputs must use `allocate_tensor()` and CUDA-stream writes. Direct
  /// `emit(Tensor)` and host writes are unsupported during capture. This declaration is not proof
  /// of stable storage and does not itself begin capture.
  /// @see RegionConstraint::cuda_graph()
  OperatorSpec& cuda_graph_capture_eligible(bool enabled = true) noexcept {
    cuda_graph_capture_eligible_ = enabled;
    return *this;
  }

  /// @brief Return the lifecycle declaration builder for this setup pass.
  /// @return Stable non-owning builder associated with this `OperatorSpec`.
  /// @note Registration is cold metadata only; callbacks are never invoked from `setup()`.
  [[nodiscard]] LifecycleBuilder& lifecycle() noexcept { return lifecycle_builder_; }

  /// @brief Declare one capacity-one readable source.
  /// @param source Stable nonmovable source member.
  /// @param name Local source name copied into spec storage.
  /// @return Stable capacity-one builder.
  SingleWitnessSourceSpecBuilder& readable_source(ReadableSource& source, std::string_view name);

  /// @brief Declare one bounded counted notification source.
  /// @param source Stable nonmovable source member.
  /// @param name Local source name copied into spec storage.
  /// @return Stable notification builder.
  NotificationSourceSpecBuilder& notification_source(NotificationSource& source,
                                                     std::string_view name);

  /// @brief Declare one bounded monotonic doorbell source.
  /// @param source Stable nonmovable source member.
  /// @param name Local source name copied into spec storage.
  /// @return Stable doorbell builder.
  DoorbellSourceSpecBuilder& doorbell_source(DoorbellSource& source, std::string_view name);

  /// @brief Declare one capacity-one false-to-true polling source.
  /// @param source Stable nonmovable source member.
  /// @param name Local source name copied into spec storage.
  /// @return Stable capacity-one builder.
  SingleWitnessSourceSpecBuilder& spin_source(SpinSource& source, std::string_view name);

 private:
  enum class LifecycleFactKind : std::uint8_t {
    kProduces,
    kRequires,
  };

  class LifecycleCallbackBinding {
   public:
    LifecycleCallbackBinding() = default;
    virtual ~LifecycleCallbackBinding() = default;
    LifecycleCallbackBinding(const LifecycleCallbackBinding&) = delete;
    LifecycleCallbackBinding& operator=(const LifecycleCallbackBinding&) = delete;
    LifecycleCallbackBinding(LifecycleCallbackBinding&&) = delete;
    LifecycleCallbackBinding& operator=(LifecycleCallbackBinding&&) = delete;

    [[nodiscard]] virtual LifecycleStatus invoke(void* self,
                                                 LifecycleContext& context) const noexcept = 0;
    [[nodiscard]] virtual const void* declaring_type_token() const noexcept = 0;
  };

  template <typename T>
  class LifecycleCallbackBindingModel final : public LifecycleCallbackBinding {
   public:
    using Method = LifecycleStatus (T::*)(LifecycleContext&) noexcept;

    explicit LifecycleCallbackBindingModel(Method method) noexcept : method_(method) {}

    [[nodiscard]] LifecycleStatus invoke(void* self,
                                         LifecycleContext& context) const noexcept override {
      return (static_cast<T*>(self)->*method_)(context);
    }

    [[nodiscard]] const void* declaring_type_token() const noexcept override {
      return &detail::kLifecycleDeclaringTypeToken<T>;
    }

   private:
    Method method_;
  };

  struct LifecycleCallbackRecord {
    LifecycleStage stage{LifecycleStage::kConfigure};
    bool required{};
    LifecycleMethodId method{};
    std::unique_ptr<LifecycleCallbackBinding> binding;
  };

  struct LifecycleFactRecord {
    LifecycleStage stage{LifecycleStage::kConfigure};
    LifecycleFactKind kind{LifecycleFactKind::kProduces};
    std::string stable_name;
  };

  struct PortRecord {
    void* declaration{};
    PortDirection direction{PortDirection::kInput};
    std::string name;
    bool tensor{};
    std::size_t builder_index{};
    const void* local_type_token{};
    PayloadIdentityRecord payload_identity{};
    const detail::RuntimePortTypeOps* runtime_type_ops{};
    // Cold exact native-type witness; not part of semantic payload identity.
    const std::type_info* local_type_info{};
  };

  enum class SourceBuilderFamily : std::uint8_t {
    kSingleWitness,
    kNotification,
    kDoorbell,
  };

  struct SourceRecord {
    void* declaration{};
    std::string name;
    ReadinessSourceKind kind{static_cast<ReadinessSourceKind>(0)};
    SourceBuilderFamily builder_family{SourceBuilderFamily::kSingleWitness};
    std::size_t builder_index{};
  };

  InputSpecBuilder& bind_input(InputBase& input, std::string_view name,
                               const void* local_type_token, PayloadIdentityRecord payload_identity,
                               const detail::RuntimePortTypeOps* runtime_type_ops);
  InputSpecBuilder& bind_input(InputBase& input, std::string_view name,
                               const void* local_type_token, PayloadIdentityRecord payload_identity,
                               const detail::RuntimePortTypeOps* runtime_type_ops,
                               const std::type_info* local_type_info);
  OutputSpecBuilder& bind_output(OutputBase& output, std::string_view name,
                                 const void* local_type_token,
                                 PayloadIdentityRecord payload_identity,
                                 const detail::RuntimePortTypeOps* runtime_type_ops);
  OutputSpecBuilder& bind_output(OutputBase& output, std::string_view name,
                                 const void* local_type_token,
                                 PayloadIdentityRecord payload_identity,
                                 const detail::RuntimePortTypeOps* runtime_type_ops,
                                 const std::type_info* local_type_info);
  TensorOutputSpecBuilder& bind_tensor_output(OutputBase& output, std::string_view name,
                                              const void* local_type_token,
                                              PayloadIdentityRecord payload_identity,
                                              const detail::RuntimePortTypeOps* runtime_type_ops);
  TensorOutputSpecBuilder& bind_tensor_output(OutputBase& output, std::string_view name,
                                              const void* local_type_token,
                                              PayloadIdentityRecord payload_identity,
                                              const detail::RuntimePortTypeOps* runtime_type_ops,
                                              const std::type_info* local_type_info);
  void validate_declaration_name(std::string_view member_name, std::string_view setup_name);
  void record_name(std::string_view name);
  void record_declaration(const void* declaration);
  void validate_source_declaration(const void* declaration, std::string_view name);
  void register_lifecycle_fact(LifecycleStage stage, LifecycleDependencyKey key,
                               LifecycleFactKind kind);

  template <PublishedPayload T>
  void admit_schema_package() {
    const Error& validation_error = detail::operator_spec_schema_package_validation<T>();
    if (validation_error.code != ErrorCode::kSuccess &&
        declaration_error_.code == ErrorCode::kSuccess) {
      declaration_error_ = validation_error;
    }
  }

  template <typename T>
  void register_lifecycle_callback(LifecycleStage stage,
                                   LifecycleStatus (T::*method)(LifecycleContext&) noexcept,
                                   bool required) {
    static_assert(std::is_class_v<T>);
    const auto stage_value = static_cast<std::uint8_t>(stage);
    if (!is_valid_lifecycle_stage(stage) || method == nullptr) {
      if (declaration_error_.code == ErrorCode::kSuccess) {
        declaration_error_ =
            Error{ErrorCode::kInvalidArgument, "operator lifecycle callback is invalid"};
      }
      return;
    }
    if (std::any_of(lifecycle_callbacks_.begin(),
                    lifecycle_callbacks_.end(),
                    [stage](const auto& existing) { return existing.stage == stage; })) {
      if (declaration_error_.code == ErrorCode::kSuccess) {
        declaration_error_ =
            Error{ErrorCode::kDuplicateName, "operator lifecycle stage callback is duplicated"};
      }
      return;
    }
    lifecycle_callbacks_.push_back(LifecycleCallbackRecord{
        .stage = stage,
        .required = required,
        .method = static_cast<LifecycleMethodId>(stage_value + 1U),
        .binding = std::make_unique<LifecycleCallbackBindingModel<T>>(method),
    });
  }

  std::deque<InputSpecBuilder> input_builders_;
  std::deque<OutputSpecBuilder> output_builders_;
  std::deque<TensorOutputSpecBuilder> tensor_output_builders_;
  std::deque<SingleWitnessSourceSpecBuilder> single_source_builders_;
  std::deque<NotificationSourceSpecBuilder> notification_source_builders_;
  std::deque<DoorbellSourceSpecBuilder> doorbell_source_builders_;
  std::vector<PortRecord> ports_;
  std::vector<SourceRecord> sources_;
  std::vector<LifecycleCallbackRecord> lifecycle_callbacks_;
  std::vector<LifecycleFactRecord> lifecycle_facts_;
  std::unordered_set<std::string> names_;
  std::unordered_set<const void*> declarations_;
  Error declaration_error_{ErrorCode::kSuccess};
  bool cuda_graph_capture_eligible_{};
  LifecycleBuilder lifecycle_builder_;

  friend class detail::SyntheticOperatorBinding;
  friend class detail::OperatorSpecCompilerAccess;
  friend class detail::OperatorSpecRuntimeAccess;
  friend class detail::OperatorSpecTestAccess;
  friend class LifecycleBuilder;
};

template <typename T>
LifecycleBuilder& LifecycleBuilder::stage(
    LifecycleStage stage, LifecycleStatus (T::*method)(LifecycleContext&) noexcept) {
  spec_->register_lifecycle_callback<T>(stage, method, true);
  return *this;
}

template <typename T>
LifecycleBuilder& LifecycleBuilder::optional(
    LifecycleStage stage, LifecycleStatus (T::*method)(LifecycleContext&) noexcept) {
  spec_->register_lifecycle_callback<T>(stage, method, false);
  return *this;
}

}  // namespace holoscan
