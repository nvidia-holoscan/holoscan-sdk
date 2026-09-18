/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/parameter_schema.hpp>

namespace holoscan::config {
class Path;
}  // namespace holoscan::config

namespace holoscan::params {
class ParamConfigBinding;
enum class ParamBindingMode : std::uint8_t;
}  // namespace holoscan::params

namespace holoscan::authoring {

/// @file
/// @brief Prototype language-neutral authoring type registry.
///
/// The declarations in this header have Prototype API stability. Registry rows contain only
/// metadata and cold-path function pointers. A future C facade may wrap the same rows, while C++
/// callers receive exceptions from a row thunk under the ordinary public Graph API contract.

namespace detail {
struct AutomaticRegistrationFailureAccess;
template <std::derived_from<OperatorBase> Op>
struct OperatorRegistrationThunks;
template <holoscan::detail::ExternalTopicPayload T>
struct PayloadRegistrationThunks;
}  // namespace detail

class MaterializedParameters;
class OpaquePortRef;

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

  /// @brief Return whether this reference contains a registered declaration.
  /// @return The same result as \ref graph_valid.
  [[nodiscard]] explicit operator bool() const noexcept { return graph_valid(); }

 private:
  template <std::derived_from<OperatorBase> Op>
  explicit OpaqueOperatorRef(const OperatorHandle<Op>& handle) noexcept
      : state_(handle.state_),
        index_(handle.index_),
        operator_type_token_(&holoscan::detail::kGraphOperatorTypeToken<Op>) {}

  template <std::derived_from<OperatorBase> Op>
  [[nodiscard]] bool holds_type() const noexcept {
    return operator_type_token_ == &holoscan::detail::kGraphOperatorTypeToken<Op>;
  }

  template <std::derived_from<OperatorBase> Op>
  [[nodiscard]] OperatorHandle<Op> typed_handle() const noexcept {
    return OperatorHandle<Op>{operator_type_token_, state_, index_};
  }

  static void lifecycle_before(GraphLifecycleBuilder& builder, const OpaqueOperatorRef& before,
                               const OpaqueOperatorRef& after, LifecycleStage stage);

  std::weak_ptr<holoscan::detail::GraphState> state_;
  std::size_t index_{};
  const void* operator_type_token_{};

  template <std::derived_from<OperatorBase> Op>
  friend struct detail::OperatorRegistrationThunks;
  friend class OpaquePortRef;
};

/// @brief Retain one type-erased logical port without exposing its declaration address.
///
/// Operator ports retain weak Graph identity and re-resolve the current declaration on every thunk
/// use. Topic and interface ports retain their typed handle owner. In both cases the payload thunk
/// verifies direction and complete payload identity before recovering a typed declaration.
class HOLOSCAN_CORE_EXPORT OpaquePortRef {
 public:
  /// @brief Create an empty invalid port reference.
  OpaquePortRef() = default;

  /// @brief Return whether this reference can currently resolve a declaration.
  /// @return `true` for a held declaration or a live named Operator port.
  [[nodiscard]] explicit operator bool() const noexcept;

 private:
  OpaquePortRef(const OpaqueOperatorRef& op, std::string_view port)
      : state_(op.state_),
        index_(op.index_),
        operator_type_token_(op.operator_type_token_),
        port_name_(port) {}

  template <PublishedPayload T>
  [[nodiscard]] static OpaquePortRef held(std::shared_ptr<const void> holder,
                                          const void* (*declaration)(const void*),
                                          PortDirection direction,
                                          bool (*valid)(const void*) noexcept = nullptr) {
    OpaquePortRef result;
    result.holder_ = std::move(holder);
    result.declaration_ = declaration;
    result.valid_ = valid;
    result.payload_identity_ = holoscan::detail::operator_spec_payload_identity<T>();
    result.direction_ = direction;
    return result;
  }

  [[nodiscard]] const void* resolve(const PayloadIdentityRecord& expected,
                                    PortDirection direction) const;

  std::shared_ptr<const void> holder_;
  const void* (*declaration_)(const void*) = nullptr;
  bool (*valid_)(const void*) noexcept = nullptr;
  PayloadIdentityRecord payload_identity_;
  PortDirection direction_{PortDirection::kInput};
  std::weak_ptr<holoscan::detail::GraphState> state_;
  std::size_t index_{};
  const void* operator_type_token_{};
  std::string port_name_;

  template <std::derived_from<OperatorBase> Op>
  friend struct detail::OperatorRegistrationThunks;
  template <holoscan::detail::ExternalTopicPayload T>
  friend struct detail::PayloadRegistrationThunks;
  friend class OpaqueImportedEndpoint;
  friend class OpaqueExportedEndpoint;
  friend class InterfacePortPair;
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

/// @brief Describe one registered parameters-only Operator type and its generated thunks.
///
/// Every function pointer is a C++-tier cold authoring operation and may throw the same exception
/// as the public template call in its body. The registration name and descriptor pointers must
/// remain valid for the process lifetime.
struct RegisteredOperatorType {
  std::string_view name;                        ///< Stable registry name.
  const ParamTypeDescriptor* param_type{};      ///< Exact aggregate storage operations.
  const ParamSchemaDescriptor* param_schema{};  ///< Declared schema, or null for `NoParams`.
  /// Add an Operator, optionally applying a materialized aggregate of the row's exact type.
  OpaqueOperatorRef (*add_to_graph)(Graph&, std::string_view,
                                    const MaterializedParameters*) = nullptr;
  /// Replace parameters after verifying the materialized aggregate's exact process-local type.
  void (*set_params)(const OpaqueOperatorRef&, const MaterializedParameters&) = nullptr;
  /// Resolve a logical port reference that revalidates the current declaration on each use.
  OpaquePortRef (*port)(const OpaqueOperatorRef&, std::string_view) = nullptr;
  /// Add the exact Operator to one region through its typed public handle.
  void (*region_add)(RegionHandle&, const OpaqueOperatorRef&) = nullptr;
  /// Add the exact Operator to one partition through its typed public handle.
  void (*partition_add)(PartitionHandle&, const OpaqueOperatorRef&) = nullptr;
  /// Add one lifecycle ordering edge after resolving both stable Operator identities.
  void (*lifecycle_before)(GraphLifecycleBuilder&, const OpaqueOperatorRef&,
                           const OpaqueOperatorRef&, LifecycleStage) = nullptr;
  /// Create a typed graph-owned configuration binding when this row explicitly supports one.
  params::ParamConfigBinding (*bind_config)(Graph&, std::string_view, config::Path,
                                            params::ParamBindingMode) = nullptr;
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
  /// Add one typed output-to-input flow after resolving both logical port references.
  void (*add_flow)(Graph&, const OpaquePortRef&, const OpaquePortRef&, ConnectionOptions) = nullptr;
  /// Export `(public_input, internal_input)` after resolving both logical port references.
  void (*export_input)(Graph&, const OpaquePortRef&, const OpaquePortRef&) = nullptr;
  /// Export `(public_output, internal_output)` after resolving both logical port references.
  void (*export_output)(Graph&, const OpaquePortRef&, const OpaquePortRef&) = nullptr;
  /// Create one typed external-to-Graph topic handle.
  OpaqueImportedEndpoint (*import_topic)(Graph&, std::string_view, TopicOptions) = nullptr;
  /// Create one typed Graph-to-external topic handle.
  OpaqueExportedEndpoint (*export_topic)(Graph&, std::string_view, TopicOptions) = nullptr;
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
  const ParamSchemaDescriptor* param_schema{};  ///< Static schema, or null for `NoParams`.
  std::vector<RegisteredOperatorPort> ports;    ///< Ports in setup declaration order.
};

/// @brief Own the process-local registry reached through \ref authoring_registry.
///
/// The registry has Prototype API stability. Registration and lookup are thread-safe. Returned
/// row references remain valid until process shutdown; registrant modules must remain loaded while
/// their thunk pointers are reachable.
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

  /// @brief Register one unique payload row.
  /// @param row Complete typed payload row.
  /// @return Success, or a duplicate/validation error as data.
  [[nodiscard]] expected<void, Error> register_payload(RegisteredPayloadType row);

  /// @brief Register one unique Clock row.
  /// @param row Complete typed Clock row.
  /// @return Success, or a duplicate/validation error as data.
  [[nodiscard]] expected<void, Error> register_clock(RegisteredClockType row);

  /// @brief Register one Operator row during static initialization without throwing.
  /// @param row Complete generated row.
  /// @return `true` on success; otherwise the failure is retained by \ref registration_status.
  [[nodiscard]] bool register_operator_automatically(RegisteredOperatorType row) noexcept;

  /// @brief Register one payload row during static initialization without throwing.
  /// @param row Complete generated row.
  /// @return `true` on success; otherwise the failure is retained by \ref registration_status.
  [[nodiscard]] bool register_payload_automatically(RegisteredPayloadType row) noexcept;

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
