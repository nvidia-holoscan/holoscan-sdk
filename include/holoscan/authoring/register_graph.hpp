/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>
#include <exception>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/authoring/registry.hpp>

namespace holoscan::authoring::detail {

/// @file
/// @brief Prototype automatic reusable Graph registration macro and typed embed thunks.

template <typename G, typename = void>
struct RegisteredGraphParams {
  using Type = NoParams;
  static constexpr bool kDeclared = false;
};

template <typename G>
struct RegisteredGraphParams<G, std::void_t<typename G::Params>> {
  using Type = typename G::Params;
  static constexpr bool kDeclared = !std::same_as<Type, NoParams>;
};

/// @brief Instantiate public Graph templates for one reusable Graph type.
/// @tparam G Exact registered Graph type.
template <std::derived_from<Graph> G>
struct GraphRegistrationThunks {
  using Params = typename RegisteredGraphParams<G>::Type;
  static constexpr bool kHasParams = RegisteredGraphParams<G>::kDeclared;

  static_assert(
      kHasParams || std::default_initializable<G>,
      "HOLOSCAN_REGISTER_GRAPH requires a default-constructible Graph or a Params aggregate");
  static_assert(!kHasParams ||
                    param_schema_availability<Params>() == ParamSchemaAvailability::kDeclared,
                "HOLOSCAN_REGISTER_GRAPH requires a declared schema for constructor Params");
  static_assert(!kHasParams || std::constructible_from<G, const Params&>,
                "HOLOSCAN_REGISTER_GRAPH requires construction from const Params&");

  /// @brief Translate ordinary typed embedding exceptions into failure data.
  template <typename Function>
  [[nodiscard]] static expected<OpaqueGraphRef, Error> invoke_embed(Function&& function) {
    try {
      return std::forward<Function>(function)();
    } catch (const RuntimeError& error) {
      return make_unexpected(make_error(error));
    } catch (const std::exception& error) {
      return make_unexpected(
          Error{ErrorCode::kFailure, "AUTHORING_GRAPH_EMBED_FAILED: " + std::string(error.what())});
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kFailure, "AUTHORING_GRAPH_EMBED_FAILED"});
    }
  }

  /// @brief Embed one typed reusable Graph using an optional complete constructor aggregate.
  static OpaqueGraphRef embed(Graph& graph, std::string_view name,
                              const MaterializedParameters* params) {
    if (params != nullptr &&
        (params->type() == nullptr ||
         params->type()->local_type_token != param_type_descriptor<Params>().local_type_token)) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "AUTHORING_PARAMETER_TYPE_MISMATCH"}};
    }
    if constexpr (kHasParams) {
      if (params != nullptr) {
        return OpaqueGraphRef{graph.embed<G>(name, *static_cast<const Params*>(params->data()))};
      }
      const Params defaults{};
      return OpaqueGraphRef{graph.embed<G>(name, defaults)};
    } else {
      (void)params;
      return OpaqueGraphRef{graph.embed<G>(name)};
    }
  }

  /// @brief Materialize named constructor values and embed one typed reusable Graph.
  static expected<OpaqueGraphRef, Error> embed_from_assignments(
      Graph& graph, std::string_view name, std::span<const ParameterAssignment> assignments) {
    if (assignments.empty()) {
      return invoke_embed([&] { return embed(graph, name, nullptr); });
    }
    auto materialized = materialize_parameters(
        param_type_descriptor<Params>(), param_schema_descriptor<Params>(), assignments);
    if (!materialized) {
      return make_unexpected(std::move(materialized).error());
    }
    if constexpr (kHasParams) {
      return invoke_embed([&] { return embed(graph, name, &*materialized); });
    } else {
      return invoke_embed([&] { return embed(graph, name, nullptr); });
    }
  }

  /// @brief Copy exported port records from one exact embedded Graph instance.
  static std::vector<PortRecord> ports(const OpaqueGraphRef& graph) {
    return graph.interface_port_records();
  }
};

/// @brief Build the plain-data registry row for one reusable Graph type.
/// @tparam G Exact registered Graph type.
/// @param name Process-lifetime stable registry identity.
/// @return Complete generated thunk row.
template <std::derived_from<Graph> G>
[[nodiscard]] RegisteredGraphType make_registered_graph(std::string_view name) {
  using Thunks = GraphRegistrationThunks<G>;
  using Params = typename Thunks::Params;
  return RegisteredGraphType{
      .name = name,
      .param_type = &param_type_descriptor<Params>(),
      .param_schema = param_schema_descriptor<Params>(),
      .parameter_introspection = Thunks::kHasParams ? GraphParameterIntrospection::kDeclared
                                                    : GraphParameterIntrospection::kNone,
      .embed = &Thunks::embed,
      .embed_from_assignments = &Thunks::embed_from_assignments,
      .ports = &Thunks::ports,
  };
}

/// @brief File one generated reusable Graph row during dynamic initialization without throwing.
/// @tparam G Exact registered Graph type.
/// @param name Process-lifetime stable registry identity.
/// @return `true` after successful registration.
template <std::derived_from<Graph> G>
[[nodiscard]] bool register_graph_automatically(std::string_view name) noexcept {
  AuthoringRegistry* registry = nullptr;
  try {
    registry = &authoring_registry();
    return registry->register_graph_automatically(make_registered_graph<G>(name));
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

/// @brief Register one reusable Graph type in the process-local authoring registry.
/// @param GRAPH Exact default-constructible Graph type or type declaring schema-backed `Params`.
/// @param NAME Stable process-lifetime registry name string literal. SDK types use names such as
/// \c holoscan.examples.ExamplePipeline; downstream types use names such as
/// \c holoscan.ext.hsb.ReceiverPipeline.
/// @note Invoke once in a translation unit that has seen the complete Graph definition. A duplicate
/// or invalid registration is retained by `AuthoringRegistry::registration_status()` and never
/// replaces the existing row or throws across module initialization.
/// @note The generated variable name uses the non-standard \c __COUNTER__ compiler extension,
/// which is supported by GCC, Clang, and MSVC.
#define HOLOSCAN_REGISTER_GRAPH(GRAPH, NAME)                                              \
  namespace {                                                                             \
  [[maybe_unused]] const bool HOLOSCAN_DETAIL_AUTHORING_CONCAT(kHoloscanRegisteredGraph_, \
                                                               __COUNTER__) =             \
      ::holoscan::authoring::detail::register_graph_automatically<GRAPH>(NAME);           \
  }
