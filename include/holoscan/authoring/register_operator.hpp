/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/authoring/parameter_materializer.hpp>
#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/parameter_schema.hpp>

namespace holoscan::authoring {

/// @file
/// @brief Prototype automatic Operator registration macros and typed authoring thunks.

/// @brief Provide the common erased thunks needed by a hand-authored Operator row.
/// @tparam Op Exact registered Operator type.
///
/// Derive a library-local thunk type from this class when constructor arguments cannot be
/// represented by the automatic registration path. The derived type supplies `add_to_graph`,
/// `add_to_graph_from_assignments`, and `set_params`; this base supplies safe handle erasure,
/// exception translation, port lookup, region/partition membership, and lifecycle ordering.
template <std::derived_from<OperatorBase> Op>
struct ManualOperatorRegistrationThunks {
  /// @brief Translate ordinary typed authoring exceptions into failure data.
  /// @tparam Function Nullary callable returning an \ref OpaqueOperatorRef.
  /// @param function Typed authoring operation to invoke.
  /// @return Its erased Operator reference, or the caught failure as data.
  template <typename Function>
  [[nodiscard]] static expected<OpaqueOperatorRef, Error> invoke_add(Function&& function) {
    try {
      return std::forward<Function>(function)();
    } catch (const RuntimeError& error) {
      try {
        return make_unexpected(make_error(error));
      } catch (...) {
        return make_unexpected(Error{error.code()});
      }
    } catch (const std::exception&) {
      return make_unexpected(Error{ErrorCode::kFailure});
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
  }

  /// @brief Resolve one logical port reference without exposing a prototype pointer.
  static OpaquePortRef port(const OpaqueOperatorRef& op, std::string_view name) {
    if (!holds_type(op)) {
      throw RuntimeError{
          Error{ErrorCode::kInvalidArgument, "AUTHORING_OPERATOR_REFERENCE_TYPE_MISMATCH"}};
    }
    if (!op.graph_valid()) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"}};
    }
    OpaquePortRef result{op, name};
    if (!result) {
      throw RuntimeError{
          Error{ErrorCode::kNotFound, "AUTHORING_OPERATOR_PORT_UNKNOWN name=" + std::string(name)}};
    }
    return result;
  }

  /// @brief Copy setup-order port records without composing or snapshotting the Graph.
  static std::vector<PortRecord> ports(const OpaqueOperatorRef& op) {
    if (!holds_type(op)) {
      throw RuntimeError{
          Error{ErrorCode::kInvalidArgument, "AUTHORING_OPERATOR_REFERENCE_TYPE_MISMATCH"}};
    }
    auto result = holoscan::detail::graph_operator_ports(
        op.state_.lock(), op.index_, op.operator_type_token_, op.handle_identity_.get());
    if (!result) {
      throw RuntimeError{result.error()};
    }
    return std::move(*result);
  }

  /// @brief Add one exact typed Operator to a region.
  static void region_add(RegionHandle& region, const OpaqueOperatorRef& op) {
    if (!holds_type(op)) {
      throw RuntimeError{
          Error{ErrorCode::kInvalidArgument, "AUTHORING_OPERATOR_REFERENCE_TYPE_MISMATCH"}};
    }
    (void)region.add(typed_handle(op));
  }

  /// @brief Add one exact typed Operator to a partition.
  static void partition_add(PartitionHandle& partition, const OpaqueOperatorRef& op) {
    if (!holds_type(op)) {
      throw RuntimeError{
          Error{ErrorCode::kInvalidArgument, "AUTHORING_OPERATOR_REFERENCE_TYPE_MISMATCH"}};
    }
    (void)partition.add(typed_handle(op));
  }

  /// @brief Author one cross-type lifecycle dependency through resolved Graph identities.
  static void lifecycle_before(GraphLifecycleBuilder& builder, const OpaqueOperatorRef& before,
                               const OpaqueOperatorRef& after, LifecycleStage stage) {
    if (!holds_type(before)) {
      throw RuntimeError{
          Error{ErrorCode::kInvalidArgument, "AUTHORING_OPERATOR_REFERENCE_TYPE_MISMATCH"}};
    }
    OpaqueOperatorRef::lifecycle_before(builder, before, after, stage);
  }

  /// @brief Authorize one exact typed Operator to request its own restart.
  static void lifecycle_allow_self_restart(GraphLifecycleBuilder& builder,
                                           const OpaqueOperatorRef& target) {
    if (!holds_type(target)) {
      throw RuntimeError{
          Error{ErrorCode::kInvalidArgument, "AUTHORING_OPERATOR_REFERENCE_TYPE_MISMATCH"}};
    }
    OpaqueOperatorRef::lifecycle_allow_self_restart(builder, target);
  }

  /// @brief Authorize external control to restart one exact typed Operator.
  static void lifecycle_allow_external_restart(GraphLifecycleBuilder& builder,
                                               const OpaqueOperatorRef& target) {
    if (!holds_type(target)) {
      throw RuntimeError{
          Error{ErrorCode::kInvalidArgument, "AUTHORING_OPERATOR_REFERENCE_TYPE_MISMATCH"}};
    }
    OpaqueOperatorRef::lifecycle_allow_external_restart(builder, target);
  }

  /// @brief Authorize one exact typed requester to restart another registered Operator.
  static void lifecycle_allow_restart_request(GraphLifecycleBuilder& builder,
                                              const OpaqueOperatorRef& requester,
                                              const OpaqueOperatorRef& target) {
    if (!holds_type(requester)) {
      throw RuntimeError{
          Error{ErrorCode::kInvalidArgument, "AUTHORING_OPERATOR_REFERENCE_TYPE_MISMATCH"}};
    }
    OpaqueOperatorRef::lifecycle_allow_restart_request(builder, requester, target);
  }

  /// @brief Reject the value-only assignment route of a row that requires constructor handles.
  ///
  /// Use as `add_to_graph_from_assignments` for a row with
  /// \ref RegisteredOperatorType::constructor_inputs. Such rows are constructed only through
  /// \ref add_registered_operator and described only through their description thunk.
  /// @return Always `AUTHORING_CONSTRUCTOR_HANDLE_ROUTE_REQUIRED`.
  [[nodiscard]] static expected<OpaqueOperatorRef, Error> reject_without_constructor_handles(
      Graph&, std::string_view, std::span<const ParameterAssignment>) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "AUTHORING_CONSTRUCTOR_HANDLE_ROUTE_REQUIRED"});
  }

  /// @brief Reject the materialized-parameter route of a row that requires constructor handles.
  ///
  /// Use as `add_to_graph` for a row with \ref RegisteredOperatorType::constructor_inputs.
  /// @throws RuntimeError Always, with `AUTHORING_CONSTRUCTOR_HANDLE_ROUTE_REQUIRED`.
  [[noreturn]] static OpaqueOperatorRef add_without_constructor_handles(
      Graph&, std::string_view, const MaterializedParameters*) {
    throw RuntimeError{
        Error{ErrorCode::kInvalidArgument, "AUTHORING_CONSTRUCTOR_HANDLE_ROUTE_REQUIRED"}};
  }

 protected:
  /// @brief Test whether an erased reference names this exact Operator type.
  [[nodiscard]] static bool holds_type(const OpaqueOperatorRef& op) noexcept {
    return op.template holds_type<Op>();
  }

  /// @brief Erase one exact typed Operator handle for a hand-authored add thunk.
  /// @param handle Typed Graph handle to retain.
  /// @return Type-erased reference accepted by registry thunks.
  [[nodiscard]] static OpaqueOperatorRef make_operator_ref(const OperatorHandle<Op>& handle) {
    return OpaqueOperatorRef{handle};
  }

  /// @brief Recover the exact typed handle retained by one erased reference.
  /// @param op Type-erased reference to inspect.
  /// @return Reconstructed typed handle. Call `holds_type()` before use.
  [[nodiscard]] static OperatorHandle<Op> typed_handle(const OpaqueOperatorRef& op) noexcept {
    return op.template typed_handle<Op>();
  }
};

namespace detail {

/// @brief Return the schema fields that language-facing constructors must receive.
/// @tparam Params Exact registered Operator parameter aggregate.
/// @return Process-lifetime view of top-level fields marked required on complete load.
template <typename Params>
[[nodiscard]] std::span<const std::string_view> required_parameter_names() {
  static const std::vector<std::string_view> names = [] {
    std::vector<std::string_view> result;
    const auto* schema = param_schema_descriptor<Params>();
    if (schema != nullptr) {
      for (const ParamFieldDescriptor& field : schema->fields) {
        if (field.required_on_load) {
          result.push_back(field.canonical_name);
        }
      }
    }
    return result;
  }();
  return std::span<const std::string_view>{names};
}

/// @brief Instantiate public Graph templates for one parameters-only Operator type.
/// @tparam Op Exact registered Operator type.
template <std::derived_from<OperatorBase> Op>
struct OperatorRegistrationThunks : ManualOperatorRegistrationThunks<Op> {
  using Base = ManualOperatorRegistrationThunks<Op>;
  using Params = typename Op::Params;

  static_assert(std::default_initializable<Op>,
                "HOLOSCAN_REGISTER_OPERATOR requires a default-constructible Operator");
  static_assert(std::same_as<Params, NoParams> ||
                    param_schema_availability<Params>() == ParamSchemaAvailability::kDeclared,
                "HOLOSCAN_REGISTER_OPERATOR requires a declared schema for configurable Params");

  /// @brief Add one typed Operator and apply an optional complete parameter aggregate.
  static OpaqueOperatorRef add_to_graph(Graph& graph, std::string_view name,
                                        const MaterializedParameters* params) {
    if (params != nullptr &&
        (params->type() == nullptr ||
         params->type()->local_type_token != param_type_descriptor<Params>().local_type_token)) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "AUTHORING_PARAMETER_TYPE_MISMATCH"}};
    }
    auto handle = params == nullptr
                      ? graph.op<Op>(name)
                      : ::holoscan::detail::GraphOperatorRegistrationAccess::add_with_params<Op>(
                            graph, name, *static_cast<const Params*>(params->data()));
    return OpaqueOperatorRef{handle};
  }

  /// @brief Materialize named values and add one typed Operator.
  static expected<OpaqueOperatorRef, Error> add_to_graph_from_assignments(
      Graph& graph, std::string_view name, std::span<const ParameterAssignment> assignments) {
    if (assignments.empty()) {
      return Base::invoke_add([&] { return add_to_graph(graph, name, nullptr); });
    }
    auto materialized = materialize_parameters(
        param_type_descriptor<Params>(), param_schema_descriptor<Params>(), assignments);
    if (!materialized) {
      return make_unexpected(std::move(materialized).error());
    }
    if constexpr (std::same_as<Params, NoParams>) {
      return Base::invoke_add([&] { return add_to_graph(graph, name, nullptr); });
    } else {
      return Base::invoke_add([&] { return add_to_graph(graph, name, &*materialized); });
    }
  }

  /// @brief Replace one registered Operator's complete parameter aggregate.
  static void set_params(const OpaqueOperatorRef& op, const MaterializedParameters& params) {
    if (!op.template holds_type<Op>()) {
      throw RuntimeError{
          Error{ErrorCode::kInvalidArgument, "AUTHORING_OPERATOR_REFERENCE_TYPE_MISMATCH"}};
    }
    if (params.type() == nullptr ||
        params.type()->local_type_token != param_type_descriptor<Params>().local_type_token) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "AUTHORING_PARAMETER_TYPE_MISMATCH"}};
    }
    op.template typed_handle<Op>().set_params(*static_cast<const Params*>(params.data()));
  }
};

/// @brief Build the plain-data registry row for one parameters-only Operator.
/// @tparam Op Exact registered Operator type.
/// @param name Process-lifetime stable registry name.
/// @return Complete generated thunk row.
template <std::derived_from<OperatorBase> Op>
[[nodiscard]] RegisteredOperatorType make_registered_operator(
    std::string_view name,
    params::ParamConfigBinding (*bind_config)(Graph&, std::string_view, config::Path,
                                              params::ParamBindingMode) = nullptr) {
  using Thunks = OperatorRegistrationThunks<Op>;
  using Params = typename Op::Params;
  return RegisteredOperatorType{
      .name = name,
      .param_type = &param_type_descriptor<Params>(),
      .param_schema = param_schema_descriptor<Params>(),
      .parameter_introspection = std::same_as<Params, NoParams>
                                     ? OperatorParameterIntrospection::kNone
                                     : OperatorParameterIntrospection::kDeclared,
      .required_parameters = required_parameter_names<Params>(),
      .add_to_graph = &Thunks::add_to_graph,
      .add_to_graph_from_assignments = &Thunks::add_to_graph_from_assignments,
      .set_params = &Thunks::set_params,
      .port = &Thunks::port,
      .ports = &Thunks::ports,
      .region_add = &Thunks::region_add,
      .partition_add = &Thunks::partition_add,
      .lifecycle_before = &Thunks::lifecycle_before,
      .lifecycle_allow_self_restart = &Thunks::lifecycle_allow_self_restart,
      .lifecycle_allow_external_restart = &Thunks::lifecycle_allow_external_restart,
      .lifecycle_allow_restart_request = &Thunks::lifecycle_allow_restart_request,
      .bind_config = bind_config,
  };
}

/// @brief File one generated row during dynamic initialization without throwing.
/// @tparam Op Exact registered Operator type.
/// @param name Process-lifetime stable registry name.
/// @return `true` after successful registration.
template <std::derived_from<OperatorBase> Op>
[[nodiscard]] bool register_operator_automatically(std::string_view name) noexcept {
  AuthoringRegistry* registry = nullptr;
  try {
    registry = &authoring_registry();
    return registry->register_operator_automatically(make_registered_operator<Op>(name));
  } catch (...) {
    AutomaticRegistrationFailureAccess::record(registry);
    return false;
  }
}

}  // namespace detail
}  // namespace holoscan::authoring

#ifndef HOLOSCAN_DETAIL_AUTHORING_CONCAT
#define HOLOSCAN_DETAIL_AUTHORING_CONCAT_INNER(X, Y) X##Y
#define HOLOSCAN_DETAIL_AUTHORING_CONCAT(X, Y) HOLOSCAN_DETAIL_AUTHORING_CONCAT_INNER(X, Y)
#endif

/// @brief Register one parameters-only Operator type in the process-local authoring registry.
/// @param OP Exact default-constructible `Operator<Params>` type.
/// @param NAME Stable process-lifetime registry name string literal. SDK types use names such as
/// \c holoscan.examples.ExampleSource; downstream types use names such as
/// \c holoscan.hsb.LinuxReceiverOp.
/// @note Invoke once in a translation unit that has seen the complete Operator definition. A
/// duplicate or invalid registration is retained by `AuthoringRegistry::registration_status()` and
/// never replaces the existing row or throws across module initialization.
/// @note The generated variable name uses the non-standard \c __COUNTER__ compiler extension,
/// which is supported by GCC, Clang, and MSVC.
#define HOLOSCAN_REGISTER_OPERATOR(OP, NAME)                                                 \
  namespace {                                                                                \
  [[maybe_unused]] const bool HOLOSCAN_DETAIL_AUTHORING_CONCAT(kHoloscanRegisteredOperator_, \
                                                               __COUNTER__) =                \
      ::holoscan::authoring::detail::register_operator_automatically<OP>(NAME);              \
  }
