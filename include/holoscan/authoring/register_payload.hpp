/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>
#include <memory>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>

#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/operator_spec.hpp>

namespace holoscan::authoring::detail {

/// @file
/// @brief Prototype build-time payload registration and typed Graph thunks.

/// @brief Instantiate the public Graph payload templates for one registered type.
/// @tparam T Exact payload type supported by Graph external topics.
template <holoscan::detail::ExternalTopicPayload T>
struct PayloadRegistrationThunks {
  static void add_flow(Graph& graph, const OpaquePortRef& output, const OpaquePortRef& input,
                       ConnectionOptions options) {
    const PayloadIdentityRecord expected = holoscan::detail::operator_spec_payload_identity<T>();
    const void* output_declaration = output.resolve(expected, PortDirection::kOutput);
    const void* input_declaration = input.resolve(expected, PortDirection::kInput);
    holoscan::detail::GraphPayloadRegistrationAccess::add_flow(
        graph,
        *static_cast<const OutputBase*>(output_declaration),
        *static_cast<const InputBase*>(input_declaration),
        expected,
        expected,
        std::move(options));
  }

  static void export_input(Graph& graph, const OpaquePortRef& public_port,
                           const OpaquePortRef& internal_port) {
    PayloadIdentityRecord expected = holoscan::detail::operator_spec_payload_identity<T>();
    const void* public_declaration = public_port.resolve(expected, PortDirection::kInput);
    const void* internal_declaration = internal_port.resolve(expected, PortDirection::kInput);
    holoscan::detail::GraphPayloadRegistrationAccess::export_input(
        graph,
        *static_cast<const InputBase*>(public_declaration),
        *static_cast<const InputBase*>(internal_declaration),
        std::move(expected));
  }

  static void export_output(Graph& graph, const OpaquePortRef& public_port,
                            const OpaquePortRef& internal_port) {
    PayloadIdentityRecord expected = holoscan::detail::operator_spec_payload_identity<T>();
    const void* public_declaration = public_port.resolve(expected, PortDirection::kOutput);
    const void* internal_declaration = internal_port.resolve(expected, PortDirection::kOutput);
    holoscan::detail::GraphPayloadRegistrationAccess::export_output(
        graph,
        *static_cast<const OutputBase*>(public_declaration),
        *static_cast<const OutputBase*>(internal_declaration),
        std::move(expected));
  }

  static OpaqueImportedEndpoint import_topic(Graph& graph, std::string_view uri,
                                             TopicOptions options) {
    static_assert(std::is_nothrow_move_constructible_v<ImportedTopic<T>>);
    auto holder = std::make_shared<std::optional<ImportedTopic<T>>>();
    OpaqueImportedEndpoint result{holder};
    holder->emplace(graph.import_topic<T>(uri, std::move(options)));
    return result;
  }

  static OpaqueExportedEndpoint export_topic(Graph& graph, std::string_view uri,
                                             TopicOptions options) {
    static_assert(std::is_nothrow_move_constructible_v<ExportedTopic<T>>);
    auto holder = std::make_shared<std::optional<ExportedTopic<T>>>();
    OpaqueExportedEndpoint result{holder};
    holder->emplace(graph.export_topic<T>(uri, std::move(options)));
    return result;
  }

  static InterfacePortPair make_interface_ports(std::string_view name) {
    return InterfacePortPair::make<T>(name);
  }
};

/// @brief Build one complete payload registry row.
/// @note Process-local enums require typed local C++ Graph authoring and cannot be registered.
/// @tparam T Exact payload type supported by Graph external topics.
/// @param name Stable language-facing registry name.
/// @return Complete identity and thunk row.
template <holoscan::detail::ExternalTopicPayload T>
[[nodiscard]] RegisteredPayloadType make_registered_payload(std::string_view name) {
  using Thunks = PayloadRegistrationThunks<T>;
  return RegisteredPayloadType{
      .name = name,
      .identity = holoscan::detail::operator_spec_payload_identity<T>(),
      .add_flow = &Thunks::add_flow,
      .export_input = &Thunks::export_input,
      .export_output = &Thunks::export_output,
      .import_topic = &Thunks::import_topic,
      .export_topic = &Thunks::export_topic,
      .make_interface_ports = &Thunks::make_interface_ports,
  };
}

/// @brief File one payload row during module initialization without throwing.
/// @tparam T Exact payload type supported by Graph external topics.
/// @param name Process-lifetime stable registry name.
/// @return `true` after successful registration.
template <holoscan::detail::ExternalTopicPayload T>
[[nodiscard]] bool register_payload_automatically(std::string_view name) noexcept {
  AuthoringRegistry* registry = nullptr;
  try {
    registry = &authoring_registry();
    return registry->register_payload_automatically(make_registered_payload<T>(name));
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

/// @brief Register one build-time payload type in the process-local authoring registry.
/// @param TYPE Exact payload type supported by Graph external topics.
/// @param NAME Stable process-lifetime registry name string literal.
/// @note Invoke once in a translation unit that has seen all payload traits. A duplicate identity
/// or name is retained by `AuthoringRegistry::registration_status()` without throwing across module
/// initialization. Process-local enums remain available only through typed local C++ Graph
/// authoring; this macro rejects them at its template constraint.
#define HOLOSCAN_REGISTER_PAYLOAD(TYPE, NAME)                                               \
  namespace {                                                                               \
  [[maybe_unused]] const bool HOLOSCAN_DETAIL_AUTHORING_CONCAT(kHoloscanRegisteredPayload_, \
                                                               __COUNTER__) =               \
      ::holoscan::authoring::detail::register_payload_automatically<TYPE>(NAME);            \
  }
