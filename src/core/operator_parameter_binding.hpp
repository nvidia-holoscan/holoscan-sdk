/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <type_traits>
#include <utility>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/parameter_schema.hpp>

namespace holoscan::detail {

/// @brief Bind one validated generation-zero aggregate before compilation or callback execution.
///
/// This non-installed seam exists so the compiler/runtime factory and focused tests use the same
/// whole-aggregate replacement rule. It grants no active-run mutation or field-patching authority.
/// HSDK5-009 owns the pointer-free plan record and fresh-session factory binding.
class OperatorParameterBinding {
 public:
  /// @brief Return aggregate operations from the Operator-defining module.
  /// @param op Operator whose dynamic implementation owns the descriptor identity.
  /// @return Defining-module descriptor used for cold binding and callback headers.
  [[nodiscard]] static const ParamTypeDescriptor& defining_type(const OperatorBase& op) noexcept {
    return op.defining_param_type();
  }

  /// @brief Return field operations from the Operator-defining module.
  /// @param op Operator whose dynamic implementation owns the schema descriptor.
  /// @return Defining-module schema descriptor, or `nullptr` for opaque parameters.
  [[nodiscard]] static const ParamSchemaDescriptor* defining_schema(
      const OperatorBase& op) noexcept {
    return op.defining_param_schema();
  }

  template <typename Op>
  [[nodiscard]] static expected<void, Error> bind_initial(Op& op, typename Op::Params params) {
    const ParamValidationResult validation = validate_params(params);
    if (!validation.ok()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if constexpr (std::same_as<typename Op::Params, NoParams>) {
      return {};
    } else if constexpr (std::is_nothrow_swappable_v<typename Op::Params>) {
      // Swap publishes the complete validated candidate with a demonstrable strong guarantee.
      // HSDK5-009 owns additional admitted plan-selected clone/materialization strategies.
      op.bind_initial_params(params);
      return {};
    } else {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
  }
};

}  // namespace holoscan::detail
