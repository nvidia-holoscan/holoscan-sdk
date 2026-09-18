/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <holoscan/authoring/parameter_materializer.hpp>
#include <holoscan/authoring/registry.hpp>

namespace holoscan::authoring::detail {

/// @file
/// @brief Prototype automatic Operator registration macros and typed authoring thunks.

/// @brief Instantiate public Graph templates for one parameters-only Operator type.
/// @tparam Op Exact registered Operator type.
template <std::derived_from<OperatorBase> Op>
struct OperatorRegistrationThunks {
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

  /// @brief Resolve one logical port reference without exposing a prototype pointer.
  static OpaquePortRef port(const OpaqueOperatorRef& op, std::string_view name) {
    if (!op.template holds_type<Op>()) {
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

  /// @brief Add one exact typed Operator to a region.
  static void region_add(RegionHandle& region, const OpaqueOperatorRef& op) {
    if (!op.template holds_type<Op>()) {
      throw RuntimeError{
          Error{ErrorCode::kInvalidArgument, "AUTHORING_OPERATOR_REFERENCE_TYPE_MISMATCH"}};
    }
    (void)region.add(op.template typed_handle<Op>());
  }

  /// @brief Add one exact typed Operator to a partition.
  static void partition_add(PartitionHandle& partition, const OpaqueOperatorRef& op) {
    if (!op.template holds_type<Op>()) {
      throw RuntimeError{
          Error{ErrorCode::kInvalidArgument, "AUTHORING_OPERATOR_REFERENCE_TYPE_MISMATCH"}};
    }
    (void)partition.add(op.template typed_handle<Op>());
  }

  /// @brief Author one cross-type lifecycle dependency through resolved Graph identities.
  static void lifecycle_before(GraphLifecycleBuilder& builder, const OpaqueOperatorRef& before,
                               const OpaqueOperatorRef& after, LifecycleStage stage) {
    if (!before.template holds_type<Op>()) {
      throw RuntimeError{
          Error{ErrorCode::kInvalidArgument, "AUTHORING_OPERATOR_REFERENCE_TYPE_MISMATCH"}};
    }
    OpaqueOperatorRef::lifecycle_before(builder, before, after, stage);
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
      .add_to_graph = &Thunks::add_to_graph,
      .set_params = &Thunks::set_params,
      .port = &Thunks::port,
      .region_add = &Thunks::region_add,
      .partition_add = &Thunks::partition_add,
      .lifecycle_before = &Thunks::lifecycle_before,
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

}  // namespace holoscan::authoring::detail

#ifndef HOLOSCAN_DETAIL_AUTHORING_CONCAT
#define HOLOSCAN_DETAIL_AUTHORING_CONCAT_INNER(X, Y) X##Y
#define HOLOSCAN_DETAIL_AUTHORING_CONCAT(X, Y) HOLOSCAN_DETAIL_AUTHORING_CONCAT_INNER(X, Y)
#endif

/// @brief Register one parameters-only Operator type in the process-local authoring registry.
/// @param OP Exact default-constructible `Operator<Params>` type.
/// @param NAME Stable process-lifetime registry name string literal. SDK types use names such as
/// \c holoscan.examples.ExampleSource; downstream types use names such as
/// \c holoscan.ext.hsb.LinuxReceiverOp.
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
