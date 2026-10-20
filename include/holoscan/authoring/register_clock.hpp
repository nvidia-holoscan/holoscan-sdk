/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>
#include <string_view>

#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/time/clock.hpp>
#include <holoscan/time/clock_registry.hpp>

namespace holoscan::authoring::detail {

/// @file
/// @brief Prototype build-time Clock registration and typed Graph thunks.

/// @brief Instantiate the default public Graph Clock template for one registered type.
/// @tparam ClockT Exact default-constructible SDK Clock type.
template <std::derived_from<Clock> ClockT>
struct ClockRegistrationThunks {
  static_assert(std::default_initializable<ClockT>,
                "registered Clock types must be default-constructible");

  static ClockKey add_to_graph(Graph& graph, std::string_view name) {
    return graph.add_clock<ClockT>(name);
  }
};

/// @brief Build one complete default-constructed Clock registry row.
/// @tparam ClockT Exact registered Clock type.
/// @param name Stable registry name.
/// @return Complete Clock thunk row.
template <std::derived_from<Clock> ClockT>
[[nodiscard]] RegisteredClockType make_registered_clock(std::string_view name) {
  return RegisteredClockType{
      .name = name,
      .add_to_graph = &ClockRegistrationThunks<ClockT>::add_to_graph,
  };
}

/// @brief File one Clock row during module initialization without throwing.
/// @tparam ClockT Exact default-constructible Clock type.
/// @param name Process-lifetime stable registry name.
/// @return `true` after successful registration.
template <std::derived_from<Clock> ClockT>
[[nodiscard]] bool register_clock_automatically(std::string_view name) noexcept {
  AuthoringRegistry* registry = nullptr;
  try {
    registry = &authoring_registry();
    return registry->register_clock_automatically(make_registered_clock<ClockT>(name));
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

/// @brief Register one default-constructible Clock in the process-local authoring registry.
/// @param TYPE Exact `Clock`-derived type.
/// @param NAME Stable process-lifetime registry name string literal.
/// @note A duplicate is retained by `AuthoringRegistry::registration_status()` without throwing
/// across module initialization.
#define HOLOSCAN_REGISTER_CLOCK(TYPE, NAME)                                               \
  namespace {                                                                             \
  [[maybe_unused]] const bool HOLOSCAN_DETAIL_AUTHORING_CONCAT(kHoloscanRegisteredClock_, \
                                                               __COUNTER__) =             \
      ::holoscan::authoring::detail::register_clock_automatically<TYPE>(NAME);            \
  }
