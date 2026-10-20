/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>
#include <string_view>
#include <utility>

#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/params/config.hpp>

namespace holoscan::authoring::detail {

/// @brief Instantiate the optional typed configuration-binding thunk for one Operator.
/// @tparam Op Exact registered Operator type.
template <std::derived_from<OperatorBase> Op>
struct OperatorConfigBindingThunk {
  /// @brief Bind configuration to an existing exact graph-local Operator.
  static params::ParamConfigBinding bind(Graph& graph, std::string_view name, config::Path path,
                                         params::ParamBindingMode mode) {
    return params::bind(graph.handle<Op>(name), std::move(path), mode);
  }
};

/// @brief File one config-capable Operator row during dynamic initialization without throwing.
/// @tparam Op Exact registered Operator type.
/// @param name Process-lifetime stable registry name.
/// @return `true` after successful registration; otherwise validation failures are retained by
/// `AuthoringRegistry::registration_status()`.
template <std::derived_from<OperatorBase> Op>
[[nodiscard]] bool register_configurable_operator_automatically(std::string_view name) noexcept {
  AuthoringRegistry* registry = nullptr;
  try {
    registry = &authoring_registry();
    return registry->register_operator_automatically(
        make_registered_operator<Op>(name, &OperatorConfigBindingThunk<Op>::bind));
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

/// @brief Register one Operator with the optional typed configuration-binding thunk.
/// @param OP Exact default-constructible `Operator<Params>` type.
/// @param NAME Stable process-lifetime registry name string literal.
/// @note Use this explicit params-component macro when registry callers require `bind_config`.
/// Automatic-registration failures are retained by `AuthoringRegistry::registration_status()`.
#define HOLOSCAN_REGISTER_CONFIGURABLE_OPERATOR(OP, NAME)                                    \
  namespace {                                                                                \
  [[maybe_unused]] const bool HOLOSCAN_DETAIL_AUTHORING_CONCAT(                              \
      kHoloscanRegisteredConfigurableOperator_, __COUNTER__) =                               \
      ::holoscan::authoring::detail::register_configurable_operator_automatically<OP>(NAME); \
  }
