/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <memory>
#include <typeinfo>
#include <utility>
#include <vector>

#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/graph.hpp>

namespace holoscan::detail {

// A narrow Graph authoring adapter, never a Runtime binding or callback authority.
class OperatorTestingAccess {
 public:
  template <std::derived_from<OperatorBase> Op>
  [[nodiscard]] static authoring::OpaqueOperatorRef erase(const OperatorHandle<Op>& handle) {
    return authoring::OpaqueOperatorRef{handle};
  }
  template <std::derived_from<OperatorBase> Op>
  [[nodiscard]] static OperatorHandle<Op> typed_handle(const authoring::OpaqueOperatorRef& op) {
    if (!op.holds_type<Op>() || !op.graph_valid()) {
      throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"});
    }
    return op.typed_handle<Op>();
  }
  static void require_graph(Graph& graph, const authoring::OpaqueOperatorRef& op) {
    if (!op.graph_valid() || op.state_.lock() != graph.state_) {
      throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"});
    }
  }
  static void add_to(PartitionHandle& partition,
                     std::span<const authoring::OpaqueOperatorRef> operators) {
    const auto state = partition.state_.lock();
    if (!state) {
      throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_PARTITION_HANDLE_INVALID"});
    }
    std::vector<std::size_t> indexes;
    indexes.reserve(operators.size());
    for (const auto& op : operators) {
      if (!op.graph_valid() || op.state_.lock() != state) {
        throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_PARTITION_MEMBER_FOREIGN"});
      }
      indexes.push_back(op.index_);
    }
    auto result = graph_partition_add_operators(
        state, partition.id_, partition.handle_identity_.get(), indexes);
    if (!result)
      throw_graph_error(std::move(result.error()));
  }
  [[nodiscard]] static PortRecord port_record(const authoring::OpaqueOperatorRef& op,
                                              std::string_view name, PortDirection direction) {
    auto result = graph_operator_port_record(op.state_.lock(),
                                             op.index_,
                                             op.operator_type_token_,
                                             op.handle_identity_.get(),
                                             name,
                                             direction);
    if (!result)
      throw_graph_error(std::move(result.error()));
    return std::move(*result);
  }
  static void input_queue_depth(const authoring::OpaqueOperatorRef& op, std::string_view name,
                                std::uint32_t depth) {
    auto state = op.state_.lock();
    auto port = graph_operator_port_record(state,
                                           op.index_,
                                           op.operator_type_token_,
                                           op.handle_identity_.get(),
                                           name,
                                           PortDirection::kInput);
    if (!port)
      throw_graph_error(std::move(port.error()));
    const auto identity = port_identity(state, port->id);
    auto configured = graph_configure_input_queue(state, port->id, identity.get(), depth);
    if (!configured)
      throw_graph_error(std::move(configured.error()));
  }
  static void input_tensor_spec(const authoring::OpaqueOperatorRef& op, std::string_view name,
                                const TensorInputSpec& spec) {
    const auto port = port_record(op, name, PortDirection::kInput);
    const auto state = op.state_.lock();
    const auto identity = port_identity(state, port.id);
    auto result = graph_configure_input_tensor_spec(state, port.id, identity.get(), spec);
    if (!result)
      throw_graph_error(std::move(result.error()));
  }
  static void output_tensor_spec(const authoring::OpaqueOperatorRef& op, std::string_view name,
                                 const TensorOutputSpec& spec) {
    const auto port = port_record(op, name, PortDirection::kOutput);
    const auto state = op.state_.lock();
    const auto identity = port_identity(state, port.id);
    auto result = graph_configure_output_tensor_spec(state, port.id, identity.get(), spec);
    if (!result)
      throw_graph_error(std::move(result.error()));
  }
  [[nodiscard]] static const void* resolve(const authoring::OpaqueOperatorRef& op,
                                           std::string_view name, PortDirection direction,
                                           const PayloadIdentityRecord& identity,
                                           const void* local_type) {
    return resolve(op, name, direction, identity, local_type, nullptr);
  }
  [[nodiscard]] static const void* resolve(const authoring::OpaqueOperatorRef& op,
                                           std::string_view name, PortDirection direction,
                                           const PayloadIdentityRecord& identity,
                                           const void* local_type,
                                           const std::type_info* local_type_info) {
    auto resolved = graph_resolve_operator_port(op.state_.lock(),
                                                op.index_,
                                                op.operator_type_token_,
                                                op.handle_identity_.get(),
                                                name,
                                                direction,
                                                identity,
                                                local_type,
                                                local_type_info);
    if (!resolved) {
      throw_graph_error(std::move(resolved.error()));
    }
    return *resolved;
  }
  template <PublishedPayload T>
  [[nodiscard]] static FlowHandle<T> flow(Graph& graph, const authoring::OpaqueOperatorRef& source,
                                          std::string_view source_name,
                                          const authoring::OpaqueOperatorRef& target,
                                          std::string_view target_name) {
    require_graph(graph, source);
    require_graph(graph, target);
    const auto identity = operator_spec_payload_identity<T>();
    const auto* token = &kOperatorSpecLocalTypeToken<T>;
    static_cast<void>(
        resolve(source, source_name, PortDirection::kOutput, identity, token, &typeid(T)));
    static_cast<void>(
        resolve(target, target_name, PortDirection::kInput, identity, token, &typeid(T)));
    const auto from = port_record(source, source_name, PortDirection::kOutput);
    const auto to = port_record(target, target_name, PortDirection::kInput);
    const auto from_identity = port_identity(graph.state_, from.id);
    const auto to_identity = port_identity(graph.state_, to.id);
    auto registration = graph_testing_flow_registration(
        graph.state_, from.id, to.id, from_identity.get(), to_identity.get());
    if (!registration) {
      throw_graph_error(std::move(registration.error()));
    }
    return FlowHandle<T>{
        graph.state_, registration->edge, from.id, to.id, std::move(registration->handle_identity)};
  }

 private:
  [[nodiscard]] static std::shared_ptr<const void> port_identity(
      const std::shared_ptr<GraphState>& state, GraphPortId port) {
    auto identity = graph_port_registration_identity(state, port);
    if (!identity) {
      throw_graph_error(std::move(identity.error()));
    }
    return std::move(*identity);
  }
};

}  // namespace holoscan::detail
