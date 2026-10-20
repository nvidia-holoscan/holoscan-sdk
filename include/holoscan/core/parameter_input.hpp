/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <memory>
#include <type_traits>
#include <utility>

namespace holoscan {

class Graph;

namespace detail {
class PreparedParamsAccess;
struct ParameterInputEvidenceState;
struct ParameterTransactionState;
struct ParameterAuthoringInventory;
}  // namespace detail

template <typename ParamsT>
/// @brief Identify typed parameter aggregates with a declared static schema.
concept ParamSchemaDeclared = ParameterAggregate<ParamsT> && detail::HasParamSchema<ParamsT>;

/// @brief Own one graph-free typed parameter value with immutable producer evidence.
/// @tparam ParamsT Typed parameter aggregate.
template <ParameterAggregate ParamsT>
class PreparedParams {
 public:
  /// @brief Copy a prepared value while sharing its immutable evidence.
  PreparedParams(const PreparedParams&) = default;
  /// @brief Move a prepared value and its immutable evidence.
  PreparedParams(PreparedParams&&) noexcept(std::is_nothrow_move_constructible_v<ParamsT>) =
      default;
  /// @brief Replace this prepared value with a copy of another.
  PreparedParams& operator=(const PreparedParams&) = default;
  /// @brief Replace this prepared value by moving another.
  PreparedParams& operator=(PreparedParams&&) noexcept(std::is_nothrow_move_assignable_v<ParamsT>) =
      default;

  /// @brief Return a pointer to the immutable typed value.
  [[nodiscard]] const ParamsT* operator->() const noexcept { return &value_; }
  /// @brief Return the immutable typed value.
  [[nodiscard]] const ParamsT& operator*() const noexcept { return value_; }
  /// @brief Return the immutable typed value.
  [[nodiscard]] const ParamsT& value() const noexcept { return value_; }

  /// @brief Extract the plain value and discard its producer evidence.
  [[nodiscard]] ParamsT take() && {
    evidence_.reset();
    return std::move(value_);
  }

 private:
  PreparedParams(ParamsT value, std::shared_ptr<const detail::ParameterInputEvidenceState> evidence)
      : value_(std::move(value)), evidence_(std::move(evidence)) {}

  ParamsT value_;
  std::shared_ptr<const detail::ParameterInputEvidenceState> evidence_;

  friend class detail::PreparedParamsAccess;
};

namespace detail {

class PreparedParamsAccess {
 public:
  template <ParameterAggregate ParamsT>
  [[nodiscard]] static PreparedParams<ParamsT> make(
      ParamsT value, std::shared_ptr<const ParameterInputEvidenceState> evidence) {
    return PreparedParams<ParamsT>{std::move(value), std::move(evidence)};
  }

  template <ParameterAggregate ParamsT>
  [[nodiscard]] static const std::shared_ptr<const ParameterInputEvidenceState>& evidence(
      const PreparedParams<ParamsT>& value) noexcept {
    return value.evidence_;
  }
};

class HOLOSCAN_CORE_EXPORT ParameterInputAccess {
 public:
  /// @brief Inspect only the current authored identities and static parameter declarations.
  [[nodiscard]] static expected<ParameterAuthoringInventory, Error> authoring_targets(Graph& graph);

  [[nodiscard]] static expected<void, Error> attach_transaction(
      Graph& graph, std::shared_ptr<const ParameterTransactionState> transaction);
};

}  // namespace detail
}  // namespace holoscan
