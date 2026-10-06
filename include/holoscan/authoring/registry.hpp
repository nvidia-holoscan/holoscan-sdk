/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/authoring/parameter_materializer.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/core/tensor_schema_output_loan.hpp>

namespace holoscan {
class ExecutionPartitionView;
class ExecutionPartitionsView;
}  // namespace holoscan

namespace holoscan::config {
class Path;
}  // namespace holoscan::config

namespace holoscan::params {
class ParamConfigBinding;
enum class ParamBindingMode : std::uint8_t;
}  // namespace holoscan::params

namespace holoscan::detail {
class OperatorTestingAccess;
}  // namespace holoscan::detail

namespace holoscan::authoring {

/// @file
/// @brief Prototype language-neutral authoring type registry.
///
/// The declarations in this header have Prototype API stability. Registry rows contain only
/// metadata and cold-path function pointers. A future C facade may wrap the same rows, while C++
/// callers receive ordinary public Graph exceptions from throwing row thunks. The explicitly
/// `expected`-returning assignment thunk instead preserves failures as data.

template <std::derived_from<OperatorBase> Op>
struct ManualOperatorRegistrationThunks;

namespace detail {
struct AutomaticRegistrationFailureAccess;
template <std::derived_from<Graph> G>
struct GraphRegistrationThunks;
template <std::derived_from<OperatorBase> Op>
struct OperatorRegistrationThunks;
template <holoscan::detail::ExternalTopicPayload T>
struct PayloadRegistrationThunks;
}  // namespace detail

class MaterializedParameters;
class OpaquePortRef;
class OpaqueOperatorRef;

class OpaqueChannelPublisher;
class OpaqueChannelSubscriber;
struct RegisteredChannelThunks;

/// @brief Carry one generated-schema value across a language-binding boundary.
///
/// Objects retain canonical generated field names, sequences retain element order, and null
/// represents an absent optional field. This is an authoring-adapter representation only; Runtime
/// continues to carry the exact registered C++ payload and use its generated schema codec.
class ForeignSchemaValue {
 public:
  using Sequence = std::vector<ForeignSchemaValue>;  ///< Ordered field sequence.
  using Object =
      std::map<std::string, ForeignSchemaValue, std::less<>>;  ///< Canonical named fields.
  using Storage = std::variant<std::monostate, bool, std::int64_t, std::uint64_t, double,
                               std::string, Tensor, Sequence, Object>;

  /// @brief Create an absent optional field.
  ForeignSchemaValue() = default;

  /// @brief Create an exact Boolean field.
  [[nodiscard]] static ForeignSchemaValue boolean(bool value) { return ForeignSchemaValue{value}; }
  /// @brief Create an exact signed-integer field.
  [[nodiscard]] static ForeignSchemaValue signed_integer(std::int64_t value) {
    return ForeignSchemaValue{value};
  }
  /// @brief Create an exact unsigned-integer field.
  [[nodiscard]] static ForeignSchemaValue unsigned_integer(std::uint64_t value) {
    return ForeignSchemaValue{value};
  }
  /// @brief Create a floating-point field represented at binding precision.
  [[nodiscard]] static ForeignSchemaValue floating_point(double value) {
    return ForeignSchemaValue{value};
  }
  /// @brief Create an owning string field.
  [[nodiscard]] static ForeignSchemaValue string(std::string value) {
    return ForeignSchemaValue{std::move(value)};
  }
  /// @brief Create an owning Tensor field handle.
  [[nodiscard]] static ForeignSchemaValue tensor(Tensor value) {
    return ForeignSchemaValue{std::move(value)};
  }
  /// @brief Create an ordered sequence field.
  [[nodiscard]] static ForeignSchemaValue sequence(Sequence value) {
    return ForeignSchemaValue{std::move(value)};
  }
  /// @brief Create a canonical generated object.
  [[nodiscard]] static ForeignSchemaValue object(Object value) {
    return ForeignSchemaValue{std::move(value)};
  }

  /// @brief Return whether this value represents an absent optional field.
  [[nodiscard]] bool is_null() const noexcept {
    return std::holds_alternative<std::monostate>(storage_);
  }
  /// @brief Borrow the exact stored alternative, or null for a different kind.
  template <typename T>
  [[nodiscard]] const T* get() const noexcept {
    return std::get_if<T>(&storage_);
  }
  /// @brief Find one canonical object field, ignoring unknown fields at native conversion.
  [[nodiscard]] const ForeignSchemaValue* field(std::string_view name) const noexcept {
    const auto* fields = get<Object>();
    if (fields == nullptr) {
      return nullptr;
    }
    const auto found = fields->find(name);
    return found == fields->end() ? nullptr : &found->second;
  }

 private:
  template <typename T>
  explicit ForeignSchemaValue(T value) : storage_(std::move(value)) {}

  Storage storage_;
};

/// @brief Convert one generated native payload to and from its language-neutral field tree.
/// @tparam T Exact owning generated schema type.
/// @note `holoscan_add_flatbuffer_schema(PYTHON)` generates the specialization.
/// @note The specialization belongs only to the hidden-visibility Python codec translation unit.
/// The native schema library intentionally instantiates its registry row without this facade, and
/// the codec DSO installs its facade-enabled row after load. Moving the specialization into a
/// shared traits header would give one template specialization different definitions across DSOs.
template <typename T>
struct PythonSchemaNativeTraits;

namespace detail {

template <typename T>
concept HasPythonSchemaNativeTraits = requires(const T& value, const ForeignSchemaValue& foreign) {
  {
    PythonSchemaNativeTraits<T>::to_foreign(value)
  } -> std::same_as<expected<ForeignSchemaValue, Error>>;
  { PythonSchemaNativeTraits<T>::from_foreign(foreign) } -> std::same_as<expected<T, Error>>;
};

// The scalar alternatives are intentionally kept together so every generated-schema scalar
// conversion shares one bounded range- and type-checking path.
template <typename T>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<T, Error> foreign_schema_scalar(const ForeignSchemaValue& value) {
  if constexpr (std::same_as<T, bool>) {
    if (const auto* source = value.get<bool>()) {
      return *source;
    }
  } else if constexpr (std::is_enum_v<T>) {
    using Underlying = std::underlying_type_t<T>;
    auto converted = foreign_schema_scalar<Underlying>(value);
    if (converted) {
      return static_cast<T>(*converted);
    }
    return make_unexpected(std::move(converted).error());
  } else if constexpr (std::integral<T>) {
    if (const auto* source = value.get<std::int64_t>(); source != nullptr) {
      if (std::in_range<T>(*source)) {
        return static_cast<T>(*source);
      }
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "PYTHON_SCHEMA_FIELD_VALUE_OUT_OF_RANGE"});
    }
    if (const auto* source = value.get<std::uint64_t>(); source != nullptr) {
      if (std::in_range<T>(*source)) {
        return static_cast<T>(*source);
      }
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "PYTHON_SCHEMA_FIELD_VALUE_OUT_OF_RANGE"});
    }
  } else if constexpr (std::floating_point<T>) {
    double source{};
    if (const auto* real = value.get<double>()) {
      source = *real;
    } else if (const auto* integer = value.get<std::int64_t>()) {
      source = static_cast<double>(*integer);
    } else if (const auto* integer = value.get<std::uint64_t>()) {
      source = static_cast<double>(*integer);
    } else {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "PYTHON_SCHEMA_FIELD_TYPE_MISMATCH"});
    }
    if (!std::isfinite(source) || source < static_cast<double>(std::numeric_limits<T>::lowest()) ||
        source > static_cast<double>(std::numeric_limits<T>::max())) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "PYTHON_SCHEMA_FIELD_VALUE_OUT_OF_RANGE"});
    }
    return static_cast<T>(source);
  } else if constexpr (std::same_as<T, std::string>) {
    if (const auto* source = value.get<std::string>()) {
      return *source;
    }
  }
  return make_unexpected(Error{ErrorCode::kInvalidArgument, "PYTHON_SCHEMA_FIELD_TYPE_MISMATCH"});
}

}  // namespace detail

/// @brief Carry one language-neutral fixed-scalar sample across an authoring adapter boundary.
struct ForeignScalarSample {
  bool has_data{};            ///< Whether `data` contains a selected payload value.
  ParamScalarValue data{};    ///< Exact fixed-width scalar bits when data is present.
  SampleMetadata metadata{};  ///< Complete transport-neutral sample metadata.
  bool is_stale{};            ///< Frozen input-selection stale/reuse classification.
  bool is_missing{};          ///< Frozen input-selection absence classification.
};

/// @brief Carry one Tensor sample across an authoring adapter boundary.
struct ForeignTensorSample {
  bool has_data{};            ///< Whether `data` contains a selected Tensor.
  Tensor data;                ///< Owning Tensor handle when data is present.
  SampleMetadata metadata{};  ///< Complete transport-neutral sample metadata.
  bool is_stale{};            ///< Frozen input-selection stale/reuse classification.
  bool is_missing{};          ///< Frozen input-selection absence classification.
};

/// @brief Own one exact type-erased process-local payload value across an adapter boundary.
class ForeignPayloadValue {
 public:
  /// @brief Create an empty value.
  ForeignPayloadValue() = default;

  /// @brief Own one copyable published payload with its process-local type token.
  /// @tparam T Exact published payload type.
  /// @param value Value moved into shared adapter-owned storage.
  /// @return Type-erased owning value.
  template <PublishedPayload T>
  [[nodiscard]] static ForeignPayloadValue from(T value) {
    ForeignPayloadValue result;
    auto holder = std::make_shared<T>(std::move(value));
    result.value_ = holder.get();
    result.holder_ = std::move(holder);
    result.local_type_token_ = &holoscan::detail::kPublishedPayloadLocalTypeToken<T>;
    return result;
  }

  /// @brief Recover the exact borrowed value when the process-local token matches.
  /// @tparam T Expected published payload type.
  /// @return Borrowed value, or null for an empty or different exact type.
  template <PublishedPayload T>
  [[nodiscard]] const T* get() const noexcept {
    return local_type_token_ == &holoscan::detail::kPublishedPayloadLocalTypeToken<T>
               ? static_cast<const T*>(value_)
               : nullptr;
  }

  /// @brief Return whether this value owns a typed payload.
  [[nodiscard]] explicit operator bool() const noexcept { return holder_ != nullptr; }

 private:
  std::shared_ptr<const void> holder_;
  const void* value_{};
  const void* local_type_token_{};
};

/// @brief Carry one owning process-local object sample across an authoring adapter boundary.
struct ForeignObjectSample {
  bool has_data{};            ///< Whether `data` contains a selected payload value.
  ForeignPayloadValue data;   ///< Exact owning process-local payload when present.
  SampleMetadata metadata{};  ///< Complete transport-neutral sample metadata.
  bool is_stale{};            ///< Frozen input-selection stale/reuse classification.
  bool is_missing{};          ///< Frozen input-selection absence classification.
};

/// @brief Carry one owning generated-schema sample across an authoring adapter boundary.
struct ForeignSchemaSample {
  bool has_data{};            ///< Whether `data` contains a selected payload value.
  ForeignSchemaValue data;    ///< Generated field tree when data is present.
  SampleMetadata metadata{};  ///< Complete transport-neutral sample metadata.
  bool is_stale{};            ///< Frozen input-selection stale/reuse classification.
  bool is_missing{};          ///< Frozen input-selection absence classification.
};

/// @brief Describe one named descriptor-Tensor field allocation at a binding boundary.
struct ForeignTensorFieldLoanRequest {
  std::string_view field_name;  ///< Canonical generated Tensor field name.
  TensorLoanRequest tensor;     ///< Borrowed shape, dtype, and optional byte strides.
};

/// @brief Type-erase one sole descriptor-Tensor schema output transaction.
///
/// The field inventory remains ordered by the generated descriptor schema. Write guards borrow
/// this bundle and must be committed or destroyed before the bundle is published or destroyed.
class ForeignTensorSchemaLoan {
 public:
  /// @brief Transfer the complete unpublished field transaction.
  ForeignTensorSchemaLoan(ForeignTensorSchemaLoan&& other) noexcept
      : storage_(std::move(other.storage_)) {
    other.storage_.field_count = 0U;
  }

  /// @brief Reject copying a sole publication transaction.
  ForeignTensorSchemaLoan(const ForeignTensorSchemaLoan&) = delete;

  /// @brief Reject copy assignment of a sole publication transaction.
  ForeignTensorSchemaLoan& operator=(const ForeignTensorSchemaLoan&) = delete;

  /// @brief Reject move assignment so live field authority cannot be replaced implicitly.
  ForeignTensorSchemaLoan& operator=(ForeignTensorSchemaLoan&&) = delete;

  /// @brief Destroy the bundle and abandon any unpublished field allocations.
  ~ForeignTensorSchemaLoan() noexcept = default;

  /// @brief Return the number of Tensor fields retained by this transaction.
  [[nodiscard]] std::size_t size() const noexcept { return storage_.field_count; }

  /// @brief Return one field's concrete memory kind.
  /// @param index Canonical generated Tensor-field ordinal.
  /// @return Planned memory kind, or \c kUnknown for an invalid or consumed field.
  [[nodiscard]] MemoryKind memory_kind(std::size_t index) const noexcept {
    const TensorOutputLoan* loan = loan_at(index);
    return loan != nullptr ? loan->memory_kind() : MemoryKind::kUnknown;
  }

  /// @brief Begin one CUDA-stream-ordered field write.
  /// @param index Canonical generated Tensor-field ordinal.
  /// @param producer_stream Stream on which producer writes are ordered.
  /// @return Sole write guard, or a typed selector or state error.
  [[nodiscard]] expected<TensorOutputWriteGuard, Error> write(
      std::size_t index, ::cuda::stream_ref producer_stream) & {
    TensorOutputLoan* loan = loan_at(index);
    if (loan == nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    return loan->write(producer_stream);
  }

  /// @brief Begin one host-accessible field write.
  /// @param index Canonical generated Tensor-field ordinal.
  /// @return Sole write guard, or a typed selector or state error.
  [[nodiscard]] expected<TensorOutputWriteGuard, Error> write_host(std::size_t index) & {
    TensorOutputLoan* loan = loan_at(index);
    if (loan == nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    return loan->write_host();
  }

 private:
  explicit ForeignTensorSchemaLoan(holoscan::detail::TensorSchemaLoanBundleStorage storage) noexcept
      : storage_(std::move(storage)) {}

  [[nodiscard]] TensorOutputLoan* loan_at(std::size_t index) noexcept {
    if (index >= storage_.field_count || !storage_.fields[index]) {
      return nullptr;
    }
    // The guard above proves this field owns a loan; clang-tidy does not model that relation.
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return &storage_.fields[index].value();
  }

  [[nodiscard]] const TensorOutputLoan* loan_at(std::size_t index) const noexcept {
    if (index >= storage_.field_count || !storage_.fields[index]) {
      return nullptr;
    }
    // The guard above proves this field owns a loan; clang-tidy does not model that relation.
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return &storage_.fields[index].value();
  }

  [[nodiscard]] holoscan::detail::TensorSchemaLoanBundleStorage release() && noexcept {
    holoscan::detail::TensorSchemaLoanBundleStorage result = std::move(storage_);
    storage_.field_count = 0U;
    return result;
  }

  holoscan::detail::TensorSchemaLoanBundleStorage storage_;

  template <holoscan::detail::ExternalTopicPayload T>
  friend struct detail::PayloadRegistrationThunks;
};

/// @brief Type-erase one exact typed reusable-Graph handle.
///
/// Copies retain only the public `GraphHandle<G>` value, whose ownership is weak. Type-owned thunks
/// use the stored handle to call the public region and clock-binding APIs without exposing a
/// concrete reusable Graph type to language bindings.
class HOLOSCAN_CORE_EXPORT OpaqueGraphRef {
 public:
  /// @brief Create an empty invalid reference.
  OpaqueGraphRef() = default;

  /// @brief Return whether the exact embedded Graph declaration remains live.
  [[nodiscard]] bool graph_valid() const noexcept;

  /// @brief Return the parent-relative embedded-instance path.
  /// @return No value for an empty, stale, or retired reference.
  [[nodiscard]] std::optional<GraphInstancePath> path() const;

  /// @brief Find one exported interface port record by exact name.
  /// @return Current pointer-free record copied from the embedded interface table.
  /// @throws RuntimeError with `GRAPH_HANDLE_INVALID` or `GRAPH_PORT_NOT_FOUND` on failure.
  [[nodiscard]] PortRecord interface_port_record(std::string_view name) const;

  /// @brief Return current exported interface port records in declaration order.
  /// @return Pointer-free records copied from this exact embedded Graph instance.
  /// @throws RuntimeError with `GRAPH_HANDLE_INVALID` on stale or retired references.
  [[nodiscard]] std::vector<PortRecord> interface_port_records() const;

  /// @brief Return one exported interface port as a logical erased reference.
  /// @return Reference that revalidates the declaration and complete payload identity on use.
  /// @throws RuntimeError with `GRAPH_HANDLE_INVALID` or `GRAPH_PORT_NOT_FOUND` on failure.
  [[nodiscard]] OpaquePortRef interface_port(std::string_view name) const;

  /// @brief Find one named clock import declared by the embedded child root.
  /// @return Owning clock key suitable for the parent's `bind_clock` operation.
  /// @throws RuntimeError with `GRAPH_HANDLE_INVALID` or `GRAPH_CLOCK_IMPORT_NOT_FOUND` on failure.
  [[nodiscard]] ClockKey imported_clock(std::string_view name) const;

  /// @brief Add this exact embedded Graph to a region through its public typed handle.
  /// @param region Destination region in the direct parent Graph.
  /// @throws RuntimeError under the matching `RegionHandle::add` contract.
  void region_add(RegionHandle& region) const;

  /// @brief Bind one child clock import through the public typed Graph API.
  /// @param graph Direct parent Graph.
  /// @param child_import Named import obtained from \ref imported_clock.
  /// @param parent_clock Named parent clock or import.
  /// @throws RuntimeError under the matching `Graph::bind_clock` contract.
  void bind_clock(Graph& graph, ClockKey child_import, ClockKey parent_clock) const;

  /// @brief Return whether this reference contains a live embedded Graph.
  [[nodiscard]] explicit operator bool() const noexcept { return graph_valid(); }

  /// @brief Type-erase one public typed reusable-Graph handle.
  /// @tparam G Exact embedded Graph type.
  /// @param handle Handle returned by `Graph::embed<G>`.
  /// @return Weak-ownership reference preserving the handle's exact public operations.
  template <std::derived_from<Graph> G>
  [[nodiscard]] static OpaqueGraphRef from_handle(const GraphHandle<G>& handle) {
    return OpaqueGraphRef{handle};
  }

 private:
  template <std::derived_from<Graph> G>
  explicit OpaqueGraphRef(const GraphHandle<G>& handle)
      : holder_(std::make_shared<GraphHandle<G>>(handle)),
        state_(handle.state_),
        index_(handle.index_),
        handle_identity_(handle.handle_identity_),
        region_add_([](RegionHandle& region, const void* value) {
          (void)region.add(*static_cast<const GraphHandle<G>*>(value));
        }),
        bind_clock_(
            [](Graph& graph, const void* value, ClockKey child_import, ClockKey parent_clock) {
              (void)graph.bind_clock(*static_cast<const GraphHandle<G>*>(value),
                                     std::move(child_import),
                                     std::move(parent_clock));
            }) {}

  std::shared_ptr<void> holder_;
  std::weak_ptr<holoscan::detail::GraphState> state_;
  std::size_t index_{};
  std::shared_ptr<const void> handle_identity_;
  void (*region_add_)(RegionHandle&, const void*) = nullptr;
  void (*bind_clock_)(Graph&, const void*, ClockKey, ClockKey) = nullptr;

  template <std::derived_from<Graph> G>
  friend struct detail::GraphRegistrationThunks;
  friend class OpaqueOperatorRef;
  friend class OpaquePortRef;
};

/// @brief Type-erase one exact typed Operator authoring handle.
///
/// Only the thunk table emitted for the exact Operator type can reconstruct its typed handle from
/// stable Graph identity. The reference retains the Graph weakly so lifecycle thunks can connect
/// distinct registered Operator types without extending the Graph lifetime.
class HOLOSCAN_CORE_EXPORT OpaqueOperatorRef {
 public:
  /// @brief Create an empty invalid reference.
  OpaqueOperatorRef() = default;

  /// @brief Return whether the exact Graph declaration remains live.
  /// @return `true` only while the Graph owns the registered Operator.
  [[nodiscard]] bool graph_valid() const noexcept;

  /// @brief Return the Operator's owned local declaration name.
  /// @return Empty view after Graph retirement.
  [[nodiscard]] std::string_view name() const noexcept;

  /// @brief Return the canonical percent-encoded Graph/Operator lookup path.
  /// @return Owned path using the no-leading-slash OperatorPath grammar.
  /// @throws RuntimeError when the Graph declaration is no longer live.
  [[nodiscard]] std::string path() const;

  /// @brief Return the accepted partition containing this exact root-Graph Operator.
  /// @param partitions Partition authority retained by one accepted execution plan.
  /// @return Independently retained partition view.
  /// @throws RuntimeError with the native typed code for an unusable, foreign, stale, or unknown
  /// identity.
  [[nodiscard]] ExecutionPartitionView execution_partition(
      const ExecutionPartitionsView& partitions) const;

  /// @brief Resolve one current local port without knowing the exact Operator type.
  /// @param name Exact setup-declared local port name.
  /// @return Logical reference that preserves the owning Operator's exact type token.
  /// @throws RuntimeError when the reference is stale or the port is unknown.
  [[nodiscard]] OpaquePortRef port(std::string_view name) const;

  /// @brief Copy current setup-order port records without knowing the exact Operator type.
  /// @throws RuntimeError when the reference is stale.
  [[nodiscard]] std::vector<PortRecord> ports() const;

  /// @brief Return whether this reference contains a registered declaration.
  /// @return The same result as \ref graph_valid.
  [[nodiscard]] explicit operator bool() const noexcept { return graph_valid(); }

  /// @brief Add an erased Operator batch to one region atomically.
  /// @param region Destination region owned by the same live Graph.
  /// @param operators Non-empty registered Operator references.
  /// @throws RuntimeError under the matching `RegionHandle::add` contract.
  static void region_add(RegionHandle& region, std::span<const OpaqueOperatorRef> operators);

  /// @brief Add erased Operator and embedded-Graph members to one region atomically.
  /// @param region Destination region owned by the same live Graph.
  /// @param operators Registered Operator references, which may be empty.
  /// @param graphs Embedded-Graph references, which may be empty when operators are present.
  /// @throws RuntimeError under the matching `RegionHandle::add` contract.
  static void region_add_members(RegionHandle& region, std::span<const OpaqueOperatorRef> operators,
                                 std::span<const OpaqueGraphRef> graphs);

  /// @brief Add an erased Operator batch to one partition atomically.
  /// @param partition Destination partition owned by the same live Graph.
  /// @param operators Non-empty registered Operator references.
  /// @throws RuntimeError under the matching variadic `PartitionHandle::add` contract.
  static void partition_add(PartitionHandle& partition,
                            std::span<const OpaqueOperatorRef> operators);

  /// @brief Author one cross-type lifecycle dependency through resolved Graph identities.
  static void lifecycle_before(GraphLifecycleBuilder& builder, const OpaqueOperatorRef& before,
                               const OpaqueOperatorRef& after, LifecycleStage stage);

  /// @brief Authorize one erased Operator to request its own restart.
  static void lifecycle_allow_self_restart(GraphLifecycleBuilder& builder,
                                           const OpaqueOperatorRef& target);

  /// @brief Authorize external control to restart one erased Operator.
  static void lifecycle_allow_external_restart(GraphLifecycleBuilder& builder,
                                               const OpaqueOperatorRef& target);

  /// @brief Authorize one erased requester to restart another erased Operator.
  static void lifecycle_allow_restart_request(GraphLifecycleBuilder& builder,
                                              const OpaqueOperatorRef& requester,
                                              const OpaqueOperatorRef& target);

  /// @brief Type-erase one public typed Operator handle.
  template <std::derived_from<OperatorBase> Op>
  [[nodiscard]] static OpaqueOperatorRef from_handle(const OperatorHandle<Op>& handle) {
    return OpaqueOperatorRef{handle};
  }

 private:
  template <std::derived_from<OperatorBase> Op>
  explicit OpaqueOperatorRef(const OperatorHandle<Op>& handle) noexcept
      : state_(handle.state_),
        index_(handle.index_),
        handle_identity_(handle.handle_identity_),
        operator_type_token_(&holoscan::detail::kGraphOperatorTypeToken<Op>) {}

  template <std::derived_from<OperatorBase> Op>
  [[nodiscard]] bool holds_type() const noexcept {
    return operator_type_token_ == &holoscan::detail::kGraphOperatorTypeToken<Op>;
  }

  template <std::derived_from<OperatorBase> Op>
  [[nodiscard]] OperatorHandle<Op> typed_handle() const noexcept {
    return OperatorHandle<Op>{operator_type_token_, state_, index_, handle_identity_};
  }

  std::weak_ptr<holoscan::detail::GraphState> state_;
  std::size_t index_{};
  std::shared_ptr<const void> handle_identity_;
  const void* operator_type_token_{};

  template <std::derived_from<OperatorBase> Op>
  friend struct detail::OperatorRegistrationThunks;
  template <std::derived_from<OperatorBase> Op>
  friend struct ManualOperatorRegistrationThunks;
  friend class OpaquePortRef;
  friend class holoscan::detail::OperatorTestingAccess;
};

/// @brief Type-erase one weak Graph port configurator.
///
/// The configurator retains weak Graph identity and the stable port ID. Configurators derived from
/// an embedded interface additionally retain its exact reusable-Graph registration identity so
/// rollback and index reuse cannot retarget them. Direction-specific methods reject calls that do
/// not apply to the configured port rather than fabricating an alternate configuration path.
class HOLOSCAN_CORE_EXPORT OpaquePortConfigurator {
 public:
  /// @brief Create an empty invalid configurator.
  OpaquePortConfigurator() = default;

  /// @brief Configure a positive input queue depth.
  /// @return This configurator for fluent language-neutral authoring.
  OpaquePortConfigurator& queue_depth(std::uint32_t depth);

  /// @brief Configure a positive output emission bound.
  /// @return This configurator for fluent language-neutral authoring.
  OpaquePortConfigurator& max_emits_per_compute(std::uint32_t count);

  /// @brief Configure a positive output late-joiner history depth.
  /// @return This configurator for fluent language-neutral authoring.
  OpaquePortConfigurator& late_joiner_history_depth(std::uint32_t depth);

  /// @brief Configure a diagnostic alias for either port direction.
  /// @return This configurator for fluent language-neutral authoring.
  OpaquePortConfigurator& diagnostic_alias(std::string_view alias);

 private:
  OpaquePortConfigurator(std::weak_ptr<holoscan::detail::GraphState> state, GraphPortId port,
                         std::shared_ptr<const void> port_handle_identity, PortDirection direction,
                         std::size_t graph_index,
                         std::shared_ptr<const void> graph_handle_identity) noexcept
      : state_(std::move(state)),
        port_(port),
        port_handle_identity_(std::move(port_handle_identity)),
        direction_(direction),
        graph_index_(graph_index),
        graph_handle_identity_(std::move(graph_handle_identity)) {}

  [[nodiscard]] std::shared_ptr<holoscan::detail::GraphState> require_live_state() const;

  std::weak_ptr<holoscan::detail::GraphState> state_;
  GraphPortId port_{};
  std::shared_ptr<const void> port_handle_identity_;
  PortDirection direction_{PortDirection::kInput};
  std::size_t graph_index_{};
  std::shared_ptr<const void> graph_handle_identity_;

  template <holoscan::detail::ExternalTopicPayload T>
  friend struct detail::PayloadRegistrationThunks;
};

/// @brief Retain one type-erased logical port without exposing its declaration address.
///
/// Operator and embedded-interface ports retain weak Graph identity and re-resolve the current
/// declaration on every thunk use. Embedded-interface references additionally retain the exact
/// reusable-Graph registration identity so rollback and index reuse cannot retarget them. Topic and
/// directly hosted interface ports retain their typed handle owner. In all cases the payload thunk
/// verifies direction and complete payload identity before recovering a typed declaration.
class HOLOSCAN_CORE_EXPORT OpaquePortRef {
 public:
  /// @brief Create an empty invalid port reference.
  OpaquePortRef() = default;

  /// @brief Return whether this reference can currently resolve a declaration.
  /// @return `true` for a held declaration, live named Operator port, or live embedded interface.
  [[nodiscard]] explicit operator bool() const noexcept;

 private:
  OpaquePortRef(const OpaqueOperatorRef& op, std::string_view port)
      : state_(op.state_),
        index_(op.index_),
        handle_identity_(op.handle_identity_),
        operator_type_token_(op.operator_type_token_),
        port_name_(port) {}

  template <PublishedPayload T>
  [[nodiscard]] static OpaquePortRef held(std::shared_ptr<const void> holder,
                                          const void* (*declaration)(const void*),
                                          PortDirection direction,
                                          bool (*valid)(const void*) noexcept = nullptr,
                                          const void* authoring_builder = nullptr,
                                          bool tensor_output_builder = false) {
    OpaquePortRef result;
    result.holder_ = std::move(holder);
    result.declaration_ = declaration;
    result.valid_ = valid;
    result.payload_identity_ = holoscan::detail::operator_spec_payload_identity<T>();
    result.direction_ = direction;
    result.authoring_builder_ = authoring_builder;
    result.tensor_output_builder_ = tensor_output_builder;
    return result;
  }

  [[nodiscard]] const void* resolve(const PayloadIdentityRecord& expected,
                                    PortDirection direction) const;

  std::shared_ptr<const void> holder_;
  const void* (*declaration_)(const void*) = nullptr;
  bool (*valid_)(const void*) noexcept = nullptr;
  PayloadIdentityRecord payload_identity_;
  PortDirection direction_{PortDirection::kInput};
  const void* authoring_builder_{};
  bool tensor_output_builder_{};
  std::weak_ptr<holoscan::detail::GraphState> state_;
  std::optional<GraphPortId> graph_port_;
  std::size_t index_{};
  std::shared_ptr<const void> handle_identity_;
  std::shared_ptr<const void> graph_handle_identity_;
  const void* operator_type_token_{};
  std::string port_name_;

  template <std::derived_from<OperatorBase> Op>
  friend struct detail::OperatorRegistrationThunks;
  template <std::derived_from<OperatorBase> Op>
  friend struct ManualOperatorRegistrationThunks;
  template <holoscan::detail::ExternalTopicPayload T>
  friend struct detail::PayloadRegistrationThunks;
  friend class OpaqueOperatorRef;
  friend class OpaqueImportedEndpoint;
  friend class OpaqueExportedEndpoint;
  friend class OpaqueGraphRef;
  friend class OpaqueOperatorRef;
  friend class InterfacePortPair;
};

/// @brief Type-erase one typed Graph flow handle.
class HOLOSCAN_CORE_EXPORT OpaqueFlowRef {
 public:
  /// @brief Create an empty flow reference.
  OpaqueFlowRef() = default;

  /// @brief Return whether the exact flow remains live.
  /// @return `true` only while the source Graph owns the edge.
  [[nodiscard]] bool graph_valid() const noexcept;

  /// @brief Return whether this reference resolves to a live flow.
  [[nodiscard]] explicit operator bool() const noexcept { return graph_valid(); }

  /// @brief Validate that one Graph owns this exact live flow.
  /// @throws RuntimeError with `GRAPH_FLOW_HANDLE_INVALID` for a foreign or stale flow.
  void validate_owner(Graph& graph) const;

  /// @brief Configure the authored finite-queue behavior.
  /// @return This flow for fluent language-neutral authoring.
  OpaqueFlowRef& queue_policy(QueuePolicy policy);

  /// @brief Configure a positive destination queue depth.
  /// @return This flow for fluent language-neutral authoring.
  OpaqueFlowRef& queue_depth(std::uint32_t depth);

  /// @brief Configure the logical flow alias.
  /// @return This flow for fluent language-neutral authoring.
  OpaqueFlowRef& topic_name(std::string_view name);

  /// @brief Configure the Tensor memory-conversion request.
  /// @return This flow for fluent language-neutral authoring.
  OpaqueFlowRef& memory_conversion(MemoryConversionPolicy policy);

  /// @brief Configure a positive in-flight retention bound.
  /// @return This flow for fluent language-neutral authoring.
  OpaqueFlowRef& max_inflight_samples(std::uint32_t count);

 private:
  OpaqueFlowRef(std::weak_ptr<holoscan::detail::GraphState> state,
                holoscan::detail::FlowRegistration flow) noexcept
      : state_(std::move(state)),
        edge_(flow.edge),
        source_(flow.source),
        target_(flow.target),
        handle_identity_(std::move(flow.handle_identity)) {}

  std::weak_ptr<holoscan::detail::GraphState> state_;
  EdgeId edge_{};
  GraphPortId source_{};
  GraphPortId target_{};
  std::shared_ptr<const void> handle_identity_;

  template <holoscan::detail::ExternalTopicPayload T>
  friend struct detail::PayloadRegistrationThunks;
};

/// @brief Type-erase one Graph import-topic handle.
class HOLOSCAN_CORE_EXPORT OpaqueImportedEndpoint {
 public:
  /// @brief Create an empty imported endpoint.
  OpaqueImportedEndpoint() = default;

  /// @brief Return the lazy Graph-side output reference.
  /// @return Type-erased reference suitable for a registered payload thunk.
  [[nodiscard]] const OpaquePortRef& port() const noexcept { return port_; }

 private:
  template <PublishedPayload T>
  explicit OpaqueImportedEndpoint(const std::shared_ptr<std::optional<ImportedTopic<T>>>& holder) {
    port_ = OpaquePortRef::held<T>(
        holder,
        [](const void* value) -> const void* {
          const auto& endpoint = *static_cast<const std::optional<ImportedTopic<T>>*>(value);
          return endpoint ? static_cast<const OutputBase*>(&endpoint->output()) : nullptr;
        },
        PortDirection::kOutput,
        [](const void* value) noexcept {
          const auto& endpoint = *static_cast<const std::optional<ImportedTopic<T>>*>(value);
          return endpoint && endpoint->graph_valid();
        });
  }

  OpaquePortRef port_;

  template <holoscan::detail::ExternalTopicPayload T>
  friend struct detail::PayloadRegistrationThunks;
};

/// @brief Type-erase one Graph export-topic handle.
class HOLOSCAN_CORE_EXPORT OpaqueExportedEndpoint {
 public:
  /// @brief Create an empty exported endpoint.
  OpaqueExportedEndpoint() = default;

  /// @brief Return the lazy Graph-side input reference.
  /// @return Type-erased reference suitable for a registered payload thunk.
  [[nodiscard]] const OpaquePortRef& port() const noexcept { return port_; }

 private:
  template <PublishedPayload T>
  explicit OpaqueExportedEndpoint(const std::shared_ptr<std::optional<ExportedTopic<T>>>& holder) {
    port_ = OpaquePortRef::held<T>(
        holder,
        [](const void* value) -> const void* {
          const auto& endpoint = *static_cast<const std::optional<ExportedTopic<T>>*>(value);
          return endpoint ? static_cast<const InputBase*>(&endpoint->input()) : nullptr;
        },
        PortDirection::kInput,
        [](const void* value) noexcept {
          const auto& endpoint = *static_cast<const std::optional<ExportedTopic<T>>*>(value);
          return endpoint && endpoint->graph_valid();
        });
  }

  OpaquePortRef port_;

  template <holoscan::detail::ExternalTopicPayload T>
  friend struct detail::PayloadRegistrationThunks;
};

/// @brief Own one typed input/output pair for a reusable-Graph interface.
class HOLOSCAN_CORE_EXPORT InterfacePortPair {
 public:
  /// @brief Create an empty interface-port pair.
  InterfacePortPair() = default;

  /// @brief Return the owned input reference.
  /// @return Type-erased reference suitable for a registered payload thunk.
  [[nodiscard]] const OpaquePortRef& input() const noexcept { return input_; }

  /// @brief Return the owned output reference.
  /// @return Type-erased reference suitable for a registered payload thunk.
  [[nodiscard]] const OpaquePortRef& output() const noexcept { return output_; }

  /// @brief Construct an exact typed interface pair.
  /// @tparam T Registered payload type.
  /// @param name Owned declaration name copied into both ports.
  template <PublishedPayload T>
  [[nodiscard]] static InterfacePortPair make(std::string_view name) {
    struct Pair {
      explicit Pair(std::string_view port_name) : input(port_name), output(port_name) {}
      Input<T> input;
      Output<T> output;
    };
    auto holder = std::make_shared<Pair>(name);
    InterfacePortPair result;
    result.input_ = OpaquePortRef::held<T>(
        holder,
        [](const void* value) -> const void* {
          return static_cast<const InputBase*>(&static_cast<const Pair*>(value)->input);
        },
        PortDirection::kInput);
    result.output_ = OpaquePortRef::held<T>(
        std::move(holder),
        [](const void* value) -> const void* {
          return static_cast<const OutputBase*>(&static_cast<const Pair*>(value)->output);
        },
        PortDirection::kOutput);
    return result;
  }

 private:
  OpaquePortRef input_;
  OpaquePortRef output_;
};

/// @brief Describe one harvested port exposed by a registered Operator type.
struct RegisteredOperatorPort {
  std::string name;                                ///< Owned setup declaration name.
  PortDirection direction{PortDirection::kInput};  ///< Input or output direction.
  PayloadIdentityRecord payload_identity;          ///< Complete semantic payload identity.
};

/// @brief Classify the parameter metadata exposed by one registered Operator row.
enum class OperatorParameterIntrospection : std::uint8_t {
  kNone,      ///< The Operator has no configurable parameters.
  kDeclared,  ///< A complete schema describes graph-owned replaceable parameters.
  /// A complete schema describes immutable constructor-replay inputs.
  kConstructorReplay,
  kPassThrough,  ///< A hand-authored thunk accepts named values without discoverable metadata.
};

/// @brief Describe one registered parameters-only Operator type and its generated thunks.
///
/// Every function pointer is a C++-tier cold authoring operation. Throwing thunks preserve the
/// corresponding public template contract; `add_to_graph_from_assignments` returns ordinary
/// authoring failures as data. The registration name and descriptor pointers must remain valid for
/// the process lifetime.
struct RegisteredOperatorType {
  std::string_view name;                        ///< Stable registry name.
  const ParamTypeDescriptor* param_type{};      ///< Exact aggregate storage operations.
  const ParamSchemaDescriptor* param_schema{};  ///< Declared schema, when discoverable.
  /// Parameter metadata availability and post-construction replacement semantics.
  OperatorParameterIntrospection parameter_introspection{OperatorParameterIntrospection::kNone};
  /// Canonical constructor-parameter names that language bindings must require from callers.
  ///
  /// The referenced names must have process lifetime. Declared and constructor-replay rows must
  /// identify top-level fields in `param_schema`; pass-through rows may name fields enforced by
  /// their hand-authored assignment thunk. Empty assignment lists remain valid for registry
  /// metadata discovery.
  std::span<const std::string_view> required_parameters;
  /// Add an Operator, optionally applying a materialized aggregate of the row's exact type.
  OpaqueOperatorRef (*add_to_graph)(Graph&, std::string_view,
                                    const MaterializedParameters*) = nullptr;
  /// Materialize named values or map them through a hand-authored constructor thunk.
  ///
  /// Empty assignments must construct a representative instance for discovery. Implementations
  /// must translate ordinary authoring failures into the returned `Error` rather than allowing an
  /// exception to cross this erased boundary.
  expected<OpaqueOperatorRef, Error> (*add_to_graph_from_assignments)(
      Graph&, std::string_view, std::span<const ParameterAssignment>) = nullptr;
  /// Replace parameters after verifying the materialized aggregate's exact process-local type.
  void (*set_params)(const OpaqueOperatorRef&, const MaterializedParameters&) = nullptr;
  /// Resolve a logical port reference that revalidates the current declaration on each use.
  OpaquePortRef (*port)(const OpaqueOperatorRef&, std::string_view) = nullptr;
  /// Copy this Operator's setup-order port records without composing or snapshotting its Graph.
  std::vector<PortRecord> (*ports)(const OpaqueOperatorRef&) = nullptr;
  /// Add the exact Operator to one region through its typed public handle.
  void (*region_add)(RegionHandle&, const OpaqueOperatorRef&) = nullptr;
  /// Add the exact Operator to one partition through its typed public handle.
  void (*partition_add)(PartitionHandle&, const OpaqueOperatorRef&) = nullptr;
  /// Add one lifecycle ordering edge after resolving both stable Operator identities.
  void (*lifecycle_before)(GraphLifecycleBuilder&, const OpaqueOperatorRef&,
                           const OpaqueOperatorRef&, LifecycleStage) = nullptr;
  /// Authorize one exact typed Operator to request its own restart.
  void (*lifecycle_allow_self_restart)(GraphLifecycleBuilder&, const OpaqueOperatorRef&) = nullptr;
  /// Authorize external control to restart one exact typed Operator.
  void (*lifecycle_allow_external_restart)(GraphLifecycleBuilder&,
                                           const OpaqueOperatorRef&) = nullptr;
  /// Authorize one exact typed requester to restart another registered Operator.
  void (*lifecycle_allow_restart_request)(GraphLifecycleBuilder&, const OpaqueOperatorRef&,
                                          const OpaqueOperatorRef&) = nullptr;
  /// Create a typed graph-owned configuration binding when this row explicitly supports one.
  params::ParamConfigBinding (*bind_config)(Graph&, std::string_view, config::Path,
                                            params::ParamBindingMode) = nullptr;
};

/// @brief Map one generated descriptor Tensor field to its wire and descriptor coordinates.
struct RegisteredDescriptorTensorField {
  std::string_view name;     ///< Generated canonical field name.
  std::uint16_t field_id{};  ///< Stable FlatBuffers field ID.
  std::uint16_t ordinal{};   ///< Canonical descriptor-array ordinal.
};

/// @brief Describe one registered closed payload type and its typed Graph thunks.
///
/// Each thunk validates both port directions and complete identities before downcasting either
/// declaration. Interface export thunks consistently accept `(public_port, internal_port)`.
/// Every thunk may throw under the matching public Graph template contract. Local enums are
/// excluded: their pointer-free identities cannot establish exact C++ enum type equality.
/// Use typed local C++ Graph authoring for enum flows instead.
struct RegisteredPayloadType {
  std::string_view name;           ///< Stable language-facing payload name.
  PayloadIdentityRecord identity;  ///< Complete matching identity.
  /// Generated schema transport profile, including an exact descriptor-Tensor profile tag.
  SchemaTransportProfile schema_transport_profile{};
  /// Whether this row materializes a generated schema through a foreign-language value facade.
  bool generated_schema_facade{};
  /// Cold standalone Channel callback family, isolated from the general authoring include graph.
  const RegisteredChannelThunks* channel{};
  /// Allocate and bind one stable typed input owned by a foreign Operator instance.
  expected<OpaquePortRef, Error> (*bind_spec_input)(OperatorSpec&, std::string_view,
                                                    std::optional<std::uint32_t>,
                                                    bool) noexcept = nullptr;
  /// Allocate and bind one stable typed output owned by a foreign Operator instance.
  expected<OpaquePortRef, Error> (*bind_spec_output)(
      OperatorSpec&, std::string_view, std::optional<std::uint32_t>) noexcept = nullptr;
  /// Declare one exact expected input rate during hosted Operator setup.
  expected<void, Error> (*input_expected_rate)(const OpaquePortRef&, Rate) noexcept = nullptr;
  /// Attach one exact Tensor input contract during hosted Operator setup.
  expected<void, Error> (*tensor_input_expects)(const OpaquePortRef&,
                                                const TensorInputSpec&) noexcept = nullptr;
  /// Admit a descriptor-schema Tensor-field projection on one exact Tensor input.
  expected<void, Error> (*tensor_input_accepts_projection)(const OpaquePortRef&) noexcept = nullptr;
  /// Declare one exact fixed nominal output rate during hosted Operator setup.
  expected<void, Error> (*output_nominal_rate)(const OpaquePortRef&, Rate) noexcept = nullptr;
  /// Derive an arbitrary hosted output rate from this exact typed input.
  expected<void, Error> (*output_follows_input)(const OpaquePortRef&, const OpaquePortRef&,
                                                RateRatio) noexcept = nullptr;
  /// Borrow the exact typed input declaration for cold temporal-contract authoring.
  PortRef (*contract_input_ref)(const OpaquePortRef&) = nullptr;
  /// Add an exact selector while preserving the registered input's process-local type token.
  Contract& (*contract_input_exact)(Contract&, const OpaquePortRef&, Exact) = nullptr;
  /// Add a latest-within selector while preserving the registered input's type token.
  Contract& (*contract_input_latest_within)(Contract&, const OpaquePortRef&,
                                            LatestWithin) = nullptr;
  /// Add a drain-all selector while preserving the registered input's process-local type token.
  Contract& (*contract_input_drain_all)(Contract&, const OpaquePortRef&, DrainAll) = nullptr;
  /// Add a bracketed selector while preserving the registered input's process-local type token.
  Contract& (*contract_input_bracketed_window)(Contract&, const OpaquePortRef&,
                                               BracketedWindow) = nullptr;
  /// Inspect the frozen input-selection state for the current activation.
  expected<InputSelection, Error> (*input_selection)(const OpaquePortRef&) noexcept = nullptr;
  /// Receive one owning fixed-scalar sample; null for non-scalar payload rows.
  expected<ForeignScalarSample, Error> (*receive_scalar)(const OpaquePortRef&) noexcept = nullptr;
  /// Return the frozen selected-batch size without materializing payload values.
  expected<std::size_t, Error> (*selected_batch_size)(const OpaquePortRef&) noexcept = nullptr;
  /// Copy one selected fixed-scalar sample at an exact frozen selector index.
  expected<ForeignScalarSample, Error> (*selected_scalar_batch_item)(
      const OpaquePortRef&, std::size_t) noexcept = nullptr;
  /// Receive only one fixed-scalar payload; null for non-scalar payload rows.
  expected<ParamScalarValue, Error> (*receive_scalar_data)(const OpaquePortRef&) noexcept = nullptr;
  /// Emit one exact fixed-scalar payload; null for non-scalar payload rows.
  expected<void, Error> (*emit_scalar)(const OpaquePortRef&, ParamScalarValue,
                                       EmitOptions) noexcept = nullptr;
  /// Receive one owning process-local object sample; returns `kNotSupported` for other rows.
  expected<ForeignObjectSample, Error> (*receive_object)(const OpaquePortRef&) noexcept = nullptr;
  /// Copy one selected process-local object sample at an exact frozen selector index.
  expected<ForeignObjectSample, Error> (*selected_object_batch_item)(
      const OpaquePortRef&, std::size_t) noexcept = nullptr;
  /// Receive one process-local object value; returns `kNotSupported` for other rows.
  expected<ForeignPayloadValue, Error> (*receive_object_data)(const OpaquePortRef&) noexcept =
      nullptr;
  /// Emit one exact process-local object; returns `kNotSupported` for other rows.
  expected<void, Error> (*emit_object)(const OpaquePortRef&, ForeignPayloadValue,
                                       EmitOptions) noexcept = nullptr;
  /// Receive one owning generated-schema sample; returns `kNotSupported` for other rows.
  expected<ForeignSchemaSample, Error> (*receive_schema)(const OpaquePortRef&) noexcept = nullptr;
  /// Copy one selected generated-schema sample at an exact frozen selector index.
  expected<ForeignSchemaSample, Error> (*selected_schema_batch_item)(
      const OpaquePortRef&, std::size_t) noexcept = nullptr;
  /// Receive one generated-schema field tree; returns `kNotSupported` for other rows.
  expected<ForeignSchemaValue, Error> (*receive_schema_data)(const OpaquePortRef&) noexcept =
      nullptr;
  /// Materialize and emit one generated-schema field tree.
  expected<void, Error> (*emit_schema)(const OpaquePortRef&, ForeignSchemaValue,
                                       EmitOptions) noexcept = nullptr;
  /// Validate that one generated-schema field tree can be materialized by this payload row.
  expected<void, Error> (*validate_schema)(const ForeignSchemaValue&) noexcept = nullptr;
  /// Configure one generated descriptor Tensor output field during Operator setup.
  expected<void, Error> (*schema_tensor_output_produces)(
      const OpaquePortRef&, std::string_view, TensorOutputSpec,
      TensorByteSpanExpression) noexcept = nullptr;
  /// Allocate all fields in one generated descriptor-Tensor transaction.
  expected<ForeignTensorSchemaLoan, Error> (*allocate_schema_tensors)(
      const OpaquePortRef&, std::span<const ForeignTensorFieldLoanRequest>) noexcept = nullptr;
  /// Materialize descriptor metadata and publish one generated descriptor-Tensor transaction.
  expected<void, Error> (*emit_schema_tensors)(const OpaquePortRef&, ForeignTensorSchemaLoan,
                                               ForeignSchemaValue, EmitOptions) noexcept = nullptr;
  /// Receive one owning Tensor sample; returns `kNotSupported` for non-Tensor rows.
  expected<ForeignTensorSample, Error> (*receive_tensor)(const OpaquePortRef&) noexcept = nullptr;
  /// Copy one selected Tensor sample at an exact frozen selector index.
  expected<ForeignTensorSample, Error> (*selected_tensor_batch_item)(
      const OpaquePortRef&, std::size_t) noexcept = nullptr;
  /// Receive only one Tensor payload; returns `kNotSupported` for non-Tensor rows.
  expected<Tensor, Error> (*receive_tensor_data)(const OpaquePortRef&) noexcept = nullptr;
  /// Emit one Tensor value; returns `kNotSupported` for non-Tensor rows.
  expected<void, Error> (*emit_tensor)(const OpaquePortRef&, Tensor,
                                       EmitOptions) noexcept = nullptr;
  /// Allocate one provider-backed Tensor loan; returns `kNotSupported` for non-Tensor rows.
  expected<TensorOutputLoan, Error> (*allocate_tensor)(const OpaquePortRef&,
                                                       TensorLoanRequest) noexcept = nullptr;
  /// Emit one provider-backed Tensor loan; returns `kNotSupported` for non-Tensor rows.
  expected<void, Error> (*emit_tensor_loan)(const OpaquePortRef&, TensorOutputLoan,
                                            EmitOptions) noexcept = nullptr;
  /// Attach an exact Tensor contract and derived byte span to the setup-time output builder.
  expected<void, Error> (*tensor_output_produces)(const OpaquePortRef&, const TensorOutputSpec&,
                                                  TensorOutputDerivations) noexcept = nullptr;
  /// Derive a Tensor byte span from one exact setup-time input declaration.
  expected<TensorByteSpanExpression, Error> (*tensor_input_max_byte_span)(
      const OpaquePortRef&) noexcept = nullptr;
  /// Add one typed output-to-input flow after resolving both logical port references.
  OpaqueFlowRef (*add_flow)(Graph&, const OpaquePortRef&, const OpaquePortRef&,
                            ConnectionOptions) = nullptr;
  /// Add one descriptor-schema Tensor-field projection to an exact Tensor input.
  OpaqueFlowRef (*add_tensor_projection)(Graph&, const OpaquePortRef&, const OpaquePortRef&,
                                         std::string_view, ConnectionOptions) = nullptr;
  /// Configure one typed input after resolving its logical port reference.
  OpaquePortConfigurator (*configure_input)(Graph&, const OpaquePortRef&) = nullptr;
  /// Configure one typed output after resolving its logical port reference.
  OpaquePortConfigurator (*configure_output)(Graph&, const OpaquePortRef&) = nullptr;
  /// Export `(public_input, internal_input)` after resolving both logical port references.
  void (*export_input)(Graph&, const OpaquePortRef&, const OpaquePortRef&) = nullptr;
  /// Export `(public_output, internal_output)` after resolving both logical port references.
  void (*export_output)(Graph&, const OpaquePortRef&, const OpaquePortRef&) = nullptr;
  /// Create one typed external-to-Graph topic handle.
  OpaqueImportedEndpoint (*import_topic)(Graph&, std::string_view, TopicOptions) = nullptr;
  /// Create one typed Graph-to-external topic handle.
  OpaqueExportedEndpoint (*export_topic)(Graph&, std::string_view, TopicOptions) = nullptr;
  /// Create one typed descriptor-Tensor external-to-Graph topic handle.
  OpaqueImportedEndpoint (*import_descriptor_topic)(
      Graph&, std::string_view, TopicOptions,
      std::span<const std::pair<std::string_view, TensorBoundaryContract>>) = nullptr;
  /// Create one typed descriptor-Tensor Graph-to-external topic handle.
  OpaqueExportedEndpoint (*export_descriptor_topic)(
      Graph&, std::string_view, TopicOptions,
      std::span<const std::pair<std::string_view, TensorBoundaryContract>>) = nullptr;
  /// Generated name-to-field-ID/ordinal map; empty for non-descriptor payloads.
  std::span<const RegisteredDescriptorTensorField> descriptor_tensor_fields;
  /// Own a matching typed input/output pair for Graph interface authoring.
  InterfacePortPair (*make_interface_ports)(std::string_view) = nullptr;
};

/// @brief Describe one registered SDK Clock type and its typed Graph thunk.
struct RegisteredClockType {
  std::string_view name;                                         ///< Stable registry name.
  ClockKey (*add_to_graph)(Graph&, std::string_view) = nullptr;  ///< Default-construction thunk.
};

/// @brief Return harvested static authoring metadata for one registered Operator type.
struct RegisteredOperatorDescription {
  std::string name;                             ///< Owned registry name.
  const ParamSchemaDescriptor* param_schema{};  ///< Static schema, when discoverable.
  /// Parameter metadata availability and post-construction replacement semantics.
  OperatorParameterIntrospection parameter_introspection{OperatorParameterIntrospection::kNone};
  std::vector<std::string> required_parameters;  ///< Required language-facing constructor fields.
  std::string parameter_repr;                    ///< Stable human-readable parameter summary.
  std::vector<RegisteredOperatorPort> ports;     ///< Ports in setup declaration order.
};

/// @brief Classify constructor-parameter metadata exposed by one registered Graph row.
enum class GraphParameterIntrospection : std::uint8_t {
  kNone,         ///< The Graph is default-constructible and has no constructor parameters.
  kDeclared,     ///< A complete schema describes the Graph constructor aggregate.
  kPassThrough,  ///< A hand-authored thunk accepts names without discoverable metadata.
};

/// @brief Describe one registered reusable Graph type and its typed embed thunks.
///
/// Function pointers are C++-tier cold authoring operations. Direct embed and port thunks may throw
/// the same exceptions as their public typed Graph calls; assignment embedding returns failures as
/// data. The registration name and descriptor pointers must remain valid for the process lifetime.
struct RegisteredGraphType {
  std::string_view name;                        ///< Stable registry identity.
  const ParamTypeDescriptor* param_type{};      ///< Exact constructor aggregate operations.
  const ParamSchemaDescriptor* param_schema{};  ///< Declared schema, when discoverable.
  /// Constructor metadata availability and pass-through behavior.
  GraphParameterIntrospection parameter_introspection{GraphParameterIntrospection::kNone};
  /// Embed one instance from an optional complete constructor aggregate.
  OpaqueGraphRef (*embed)(Graph&, std::string_view, const MaterializedParameters*) = nullptr;
  /// Materialize named constructor assignments and embed one instance, returning failure as data.
  expected<OpaqueGraphRef, Error> (*embed_from_assignments)(
      Graph&, std::string_view, std::span<const ParameterAssignment>) = nullptr;
  /// Copy the current exported port records from one exact embedded instance.
  std::vector<PortRecord> (*ports)(const OpaqueGraphRef&) = nullptr;
};

/// @brief Own the complete result of one registered reusable-Graph embedding.
struct RegisteredGraphEmbedding {
  OpaqueGraphRef graph;           ///< Exact weak embedded-Graph identity.
  std::vector<PortRecord> ports;  ///< Exact instance interface copied before publication.
};

/// @brief Embed through one registry row and harvest its exact interface atomically.
/// @param graph Destination parent Graph.
/// @param row Stable registered reusable-Graph row.
/// @param name Parent-local child instance name.
/// @param assignments Named constructor assignments.
/// @return Complete erased identity and interface, or failure with no parent mutation.
/// @throws Any exceptional row failure after restoring the parent Graph.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<RegisteredGraphEmbedding, Error> embed_registered_graph(
    Graph& graph, const RegisteredGraphType& row, std::string_view name,
    std::span<const ParameterAssignment> assignments);

/// @brief Describe one exported interface port on a registered reusable Graph type.
struct RegisteredGraphPort {
  std::string name;                                ///< Owned interface declaration name.
  PortDirection direction{PortDirection::kInput};  ///< Input or output direction.
  PayloadIdentityRecord payload_identity;          ///< Complete semantic payload identity.
};

/// @brief Return harvested static authoring metadata for one registered reusable Graph type.
struct RegisteredGraphDescription {
  std::string name;                             ///< Owned registry identity.
  const ParamSchemaDescriptor* param_schema{};  ///< Constructor schema, when discoverable.
  /// Constructor metadata availability and pass-through behavior.
  GraphParameterIntrospection parameter_introspection{GraphParameterIntrospection::kNone};
  std::string parameter_repr;              ///< Stable constructor-parameter summary.
  std::vector<RegisteredGraphPort> ports;  ///< Root interface ports in declaration order.
};

/// @brief Own the process-local registry reached through \ref authoring_registry.
///
/// The registry has Prototype API stability. Registration and lookup are thread-safe. Returned row
/// references are immutable and remain valid until process shutdown; a later facade installation
/// is visible to later lookups without changing earlier references. Registrant modules must remain
/// loaded while their thunk pointers are reachable.
class HOLOSCAN_CORE_EXPORT AuthoringRegistry {
 public:
  /// @brief Construct an empty registry and install the SDK built-in payload and Clock rows.
  AuthoringRegistry();

  /// @brief Destroy registry-owned names, rows, and harvested descriptions.
  ~AuthoringRegistry();

  AuthoringRegistry(const AuthoringRegistry&) = delete;
  AuthoringRegistry& operator=(const AuthoringRegistry&) = delete;
  AuthoringRegistry(AuthoringRegistry&&) = delete;
  AuthoringRegistry& operator=(AuthoringRegistry&&) = delete;

  /// @brief Register one unique Operator row.
  /// @param row Complete row whose name and functions remain callable.
  /// @return Success, or a `kDuplicateName`/validation error as data.
  [[nodiscard]] expected<void, Error> register_operator(RegisteredOperatorType row);

  /// @brief Register one unique reusable Graph row.
  /// @param row Complete row whose name and functions remain callable.
  /// @return Success, or a `kDuplicateName`/validation error as data.
  [[nodiscard]] expected<void, Error> register_graph(RegisteredGraphType row);

  /// @brief Register one unique payload row.
  /// @param row Complete typed payload row.
  /// @return Success, or a duplicate/validation error as data.
  [[nodiscard]] expected<void, Error> register_payload(RegisteredPayloadType row);

  /// @brief Install or publish the canonical row for a generated-schema value facade.
  /// @param row Complete row whose identity and name must match any native row already present.
  /// @return Success after insertion or publication of an immutable facade generation.
  [[nodiscard]] expected<void, Error> install_payload_facade(RegisteredPayloadType row);

  /// @brief Register one unique Clock row.
  /// @param row Complete typed Clock row.
  /// @return Success, or a duplicate/validation error as data.
  [[nodiscard]] expected<void, Error> register_clock(RegisteredClockType row);

  /// @brief Register one Operator row during static initialization without throwing.
  /// @param row Complete generated row.
  /// @return `true` on success; otherwise the failure is retained by \ref registration_status.
  [[nodiscard]] bool register_operator_automatically(RegisteredOperatorType row) noexcept;

  /// @brief Register one reusable Graph row during static initialization without throwing.
  /// @param row Complete generated row.
  /// @return `true` on success; otherwise the failure is retained by \ref registration_status.
  [[nodiscard]] bool register_graph_automatically(RegisteredGraphType row) noexcept;

  /// @brief Register one payload row during static initialization without throwing.
  /// @param row Complete generated row.
  /// @return `true` on success; otherwise the failure is retained by \ref registration_status.
  [[nodiscard]] bool register_payload_automatically(RegisteredPayloadType row) noexcept;

  /// @brief Install a generated-schema facade during Python module initialization without throwing.
  /// @param row Complete facade-capable row.
  /// @return `true` on success; otherwise the failure is retained by registration_status().
  [[nodiscard]] bool install_payload_facade_automatically(RegisteredPayloadType row) noexcept;

  /// @brief Register one Clock row during static initialization without throwing.
  /// @param row Complete generated row.
  /// @return `true` on success; otherwise the failure is retained by \ref registration_status.
  [[nodiscard]] bool register_clock_automatically(RegisteredClockType row) noexcept;

  /// @brief Return the first deferred automatic-registration failure, if any.
  /// @return Success when every automatic registration succeeded, or the first failure as data.
  [[nodiscard]] expected<void, Error> registration_status() const;

  /// @brief Find an Operator row by exact stable name.
  /// @param name Exact registry name.
  /// @return Stable row reference, or a rich `kNotFound` error.
  [[nodiscard]] expected<std::reference_wrapper<const RegisteredOperatorType>, Error> operator_type(
      std::string_view name) const;

  /// @brief Find a reusable Graph row by exact stable name.
  /// @param name Exact registry identity.
  /// @return Stable row reference, or a rich `kNotFound` error.
  [[nodiscard]] expected<std::reference_wrapper<const RegisteredGraphType>, Error> graph_type(
      std::string_view name) const;

  /// @brief Find a payload row by exact stable name.
  /// @param name Exact registry name.
  /// @return Stable row reference, or a rich `kNotFound` error.
  [[nodiscard]] expected<std::reference_wrapper<const RegisteredPayloadType>, Error> payload_type(
      std::string_view name) const;

  /// @brief Find a payload row by complete semantic identity.
  /// @param identity Exact pointer-free identity.
  /// @return Stable row reference, or a rich `kNotFound` error.
  [[nodiscard]] expected<std::reference_wrapper<const RegisteredPayloadType>, Error> payload_type(
      const PayloadIdentityRecord& identity) const;

  /// @brief Find a Clock row by exact stable name.
  /// @param name Exact registry name.
  /// @return Stable row reference, or a rich `kNotFound` error.
  [[nodiscard]] expected<std::reference_wrapper<const RegisteredClockType>, Error> clock_type(
      std::string_view name) const;

  /// @brief Enumerate registered Operator rows in stable lexical-name order.
  /// @return Stable reference wrappers copied into an owning vector.
  [[nodiscard]] std::vector<std::reference_wrapper<const RegisteredOperatorType>> operators() const;

  /// @brief Enumerate registered reusable Graph rows in stable lexical-name order.
  /// @return Stable reference wrappers copied into an owning vector.
  [[nodiscard]] std::vector<std::reference_wrapper<const RegisteredGraphType>> graphs() const;

  /// @brief Enumerate registered payload rows in stable lexical-name order.
  /// @return Stable reference wrappers copied into an owning vector.
  [[nodiscard]] std::vector<std::reference_wrapper<const RegisteredPayloadType>> payloads() const;

  /// @brief Enumerate registered Clock rows in stable lexical-name order.
  /// @return Stable reference wrappers copied into an owning vector.
  [[nodiscard]] std::vector<std::reference_wrapper<const RegisteredClockType>> clocks() const;

  /// @brief Lazily harvest one Operator type through its own add thunk and a scratch Graph.
  ///
  /// The first call default-constructs the registered Operator and executes `setup()`; constructors
  /// and setup routines that allocate, log, or acquire external resources therefore incur those
  /// effects during discovery. The owned description is cached after a successful harvest.
  /// @param name Exact registered Operator name.
  /// @return Owned port/schema description or a registration/authoring error as data.
  [[nodiscard]] expected<RegisteredOperatorDescription, Error> describe_operator(
      std::string_view name);

  /// @brief Lazily harvest one reusable Graph type through its own embed thunk and a scratch
  /// parent.
  /// @param name Exact registered Graph name.
  /// @return Owned interface/schema description or a registration/authoring error as data.
  [[nodiscard]] expected<RegisteredGraphDescription, Error> describe_graph(std::string_view name);

 private:
  static void record_automatic_registration_failure(AuthoringRegistry* registry) noexcept;

  class Impl;
  std::unique_ptr<Impl> impl_;

  friend struct detail::AutomaticRegistrationFailureAccess;
};

namespace detail {

/// @brief Bridge automatic-registration wrappers to the allocation-free failure latch.
struct AutomaticRegistrationFailureAccess {
  static void record(AuthoringRegistry* registry) noexcept {
    AuthoringRegistry::record_automatic_registration_failure(registry);
  }
};

}  // namespace detail

/// @brief Return the Core-owned process-local Meyers singleton registry.
/// @return Process-lifetime registry storage.
[[nodiscard]] HOLOSCAN_CORE_EXPORT AuthoringRegistry& authoring_registry();

/// @brief Compare two complete payload identities without using local C++ type evidence.
/// @param left First pointer-free identity.
/// @param right Second pointer-free identity.
/// @return `true` only when every semantic identity field is equal.
[[nodiscard]] HOLOSCAN_CORE_EXPORT bool payload_identities_equal(
    const PayloadIdentityRecord& left, const PayloadIdentityRecord& right) noexcept;

}  // namespace holoscan::authoring
