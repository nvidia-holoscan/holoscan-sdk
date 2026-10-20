/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/authoring/registry.hpp>

#include <holoscan/authoring/channel_registry.hpp>
#include <holoscan/authoring/register_clock.hpp>

#include <algorithm>
#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "authoring/constructor_binding_factory.hpp"
#include "authoring/registry_builtin_payloads.hpp"
#include "core/graph_state.hpp"
#include "core/plan_record_contracts.hpp"

#include <holoscan/core/execution_plan.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/time/synthetic_clock.hpp>

namespace holoscan::authoring {
namespace {

[[nodiscard]] std::atomic_bool& process_automatic_registration_failure_latch() noexcept {
  static constinit std::atomic_bool failed{false};
  return failed;
}

[[nodiscard]] bool has_local_enum_identity(const PayloadIdentityRecord& identity) noexcept {
  return identity.scalar && identity.scalar->kind == ScalarIdentityKind::kLocalEnum;
}

[[nodiscard]] Error local_enum_error(ErrorCode code, std::string_view token) noexcept {
  Error error{code};
  try {
    error.message = token;
    error.message += "; process-local enums require typed C++ local Graph authoring";
  } catch (...) {
    // Supplemental guidance cannot replace the rejection code.
    error.message.clear();
  }
  return error;
}

[[nodiscard]] Error invalid_row(std::string_view kind, std::string_view field = {}) {
  std::string message = "AUTHORING_REGISTRY_ROW_INVALID kind=" + std::string(kind);
  if (!field.empty()) {
    message.append(" field=");
    message.append(field);
  }
  return Error{ErrorCode::kInvalidArgument, std::move(message)};
}

[[nodiscard]] Error duplicate_row(std::string_view kind, std::string_view name) {
  return Error{
      ErrorCode::kDuplicateName,
      "AUTHORING_REGISTRY_DUPLICATE kind=" + std::string(kind) + " name=" + std::string(name)};
}

[[nodiscard]] Error unknown_row(std::string_view kind, std::string_view name) {
  return Error{ErrorCode::kNotFound,
               "AUTHORING_REGISTRY_UNKNOWN kind=" + std::string(kind) +
                   " name=" + std::string(name) + "; did you load the module that provides it?"};
}

[[nodiscard]] bool valid_parameter_introspection(const RegisteredOperatorType& row) noexcept {
  switch (row.parameter_introspection) {
    case OperatorParameterIntrospection::kNone:
      return row.param_schema == nullptr;
    case OperatorParameterIntrospection::kDeclared:
      return row.param_schema != nullptr && row.param_schema->local_type == row.param_type;
    case OperatorParameterIntrospection::kConstructorReplay:
      return row.param_schema != nullptr && row.param_schema->local_type == row.param_type &&
             row.bind_config == nullptr;
    case OperatorParameterIntrospection::kPassThrough:
      return row.param_schema == nullptr && row.bind_config == nullptr;
  }
  return false;
}

[[nodiscard]] bool valid_required_parameters(const RegisteredOperatorType& row) noexcept {
  if (row.required_parameters.empty()) {
    return true;
  }
  const bool has_declared_schema =
      row.parameter_introspection == OperatorParameterIntrospection::kDeclared ||
      row.parameter_introspection == OperatorParameterIntrospection::kConstructorReplay;
  if (!has_declared_schema &&
      row.parameter_introspection != OperatorParameterIntrospection::kPassThrough) {
    return false;
  }
  for (auto current = row.required_parameters.begin(); current != row.required_parameters.end();
       ++current) {
    const std::string_view name = *current;
    if (name.empty()) {
      return false;
    }
    if (std::find(row.required_parameters.begin(), current, name) != current) {
      return false;
    }
    if (has_declared_schema) {
      const auto field = std::find_if(row.param_schema->fields.begin(),
                                      row.param_schema->fields.end(),
                                      [name](const ParamFieldDescriptor& candidate) {
                                        return candidate.canonical_name == name;
                                      });
      if (field == row.param_schema->fields.end() || !field->alias.empty()) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] Error constructor_handle_error(ErrorCode code, std::string_view token,
                                             std::string_view field) {
  return detail::constructor_handle_error(code, token, "field", field);
}

[[nodiscard]] bool names_value_field(const RegisteredOperatorType& row,
                                     std::string_view name) noexcept {
  if (std::ranges::find(row.required_parameters, name) != row.required_parameters.end()) {
    return true;
  }
  if (row.param_schema == nullptr) {
    return false;
  }
  return std::ranges::any_of(row.param_schema->fields, [name](const ParamFieldDescriptor& field) {
    return field.canonical_name == name || (!field.alias.empty() && field.alias == name);
  });
}

// Handle names must be distinct from each other and from every value name that the row declares, so
// that a language binding can route each keyword argument without consulting the thunk.
[[nodiscard]] bool valid_constructor_inputs(const RegisteredOperatorType& row) noexcept {
  const RegisteredConstructorInputs* table = row.constructor_inputs;
  if (table == nullptr) {
    return true;
  }
  if (table->add_to_graph_from_inputs == nullptr || table->add_for_description == nullptr) {
    return false;
  }
  for (auto current = table->handles.begin(); current != table->handles.end(); ++current) {
    if (current->name.empty() || current->type == nullptr || current->type->name.empty() ||
        names_value_field(row, current->name)) {
      return false;
    }
    const auto same_name = [&](const ConstructorHandleInput& other) {
      return other.name == current->name;
    };
    if (std::any_of(table->handles.begin(), current, same_name)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] const ConstructorHandleInput* find_handle_input(
    const RegisteredConstructorInputs& table, std::string_view name) noexcept {
  const auto found = std::ranges::find(table.handles, name, &ConstructorHandleInput::name);
  return found == table.handles.end() ? nullptr : &*found;
}

// Validate a complete call before any Graph mutation so that a failing handle never leaves a
// partial declaration or a retained temporary owner behind.
[[nodiscard]] expected<void, Error> validate_constructor_handles(
    const RegisteredConstructorInputs& table, const OperatorConstructionInputs& inputs) {
  for (auto current = inputs.handles.begin(); current != inputs.handles.end(); ++current) {
    const ConstructorHandleInput* declared = find_handle_input(table, current->name);
    if (declared == nullptr) {
      return make_unexpected(constructor_handle_error(
          ErrorCode::kInvalidArgument, "AUTHORING_CONSTRUCTOR_HANDLE_UNKNOWN", current->name));
    }
    const auto same_name = [&](const ConstructorHandleAssignment& other) {
      return other.name == current->name;
    };
    if (std::any_of(inputs.handles.begin(), current, same_name)) {
      return make_unexpected(constructor_handle_error(
          ErrorCode::kDuplicateName, "AUTHORING_CONSTRUCTOR_HANDLE_DUPLICATE", current->name));
    }
    if (!current->value) {
      return make_unexpected(constructor_handle_error(
          ErrorCode::kInvalidArgument, "AUTHORING_CONSTRUCTOR_HANDLE_NULL", current->name));
    }
    if (current->value.type() != declared->type) {
      return make_unexpected(constructor_handle_error(
          ErrorCode::kIncompatible, "AUTHORING_CONSTRUCTOR_HANDLE_TYPE_MISMATCH", current->name));
    }
    if (!current->value.created_in_current_process()) {
      return make_unexpected(
          constructor_handle_error(ErrorCode::kIncompatible,
                                   "AUTHORING_CONSTRUCTOR_HANDLE_PROCESS_MISMATCH",
                                   current->name));
    }
  }
  for (const ParameterAssignment& value : inputs.values) {
    if (find_handle_input(table, value.name) != nullptr) {
      return make_unexpected(constructor_handle_error(
          ErrorCode::kInvalidArgument, "AUTHORING_CONSTRUCTOR_HANDLE_VALUE_CONFLICT", value.name));
    }
  }
  for (const ConstructorHandleInput& declared : table.handles) {
    if (declared.required && inputs.find(declared.name) == nullptr) {
      return make_unexpected(constructor_handle_error(
          ErrorCode::kInvalidArgument, "AUTHORING_CONSTRUCTOR_HANDLE_REQUIRED", declared.name));
    }
  }
  return {};
}

// Record supplied handles in row declaration order so that the record is independent of the
// caller's keyword order.
[[nodiscard]] std::shared_ptr<const detail::ConstructorBindingRecord> make_binding_record(
    const RegisteredConstructorInputs& table, const OperatorConstructionInputs& inputs) {
  auto record = std::make_shared<detail::ConstructorBindingRecord>();
  record->origin_process = detail::constructor_handle_process_id();
  for (const ConstructorHandleInput& declared : table.handles) {
    if (const ConstructorHandle* supplied = inputs.find(declared.name); supplied != nullptr) {
      record->bindings.push_back(detail::ConstructorBinding{
          .name = std::string(declared.name), .type = declared.type, .owner = *supplied});
    }
  }
  return record;
}

[[nodiscard]] bool valid_parameter_introspection(const RegisteredGraphType& row) noexcept {
  switch (row.parameter_introspection) {
    case GraphParameterIntrospection::kNone:
      return row.param_schema == nullptr;
    case GraphParameterIntrospection::kDeclared:
      return row.param_schema != nullptr && row.param_schema->local_type == row.param_type;
    case GraphParameterIntrospection::kPassThrough:
      return row.param_schema == nullptr;
  }
  return false;
}

[[nodiscard]] std::shared_ptr<holoscan::detail::GraphState> require_graph_state(
    const std::weak_ptr<holoscan::detail::GraphState>& state, std::string_view token) {
  auto locked = state.lock();
  if (!locked) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, std::string{token}}};
  }
  return locked;
}

void apply_graph_configuration(expected<void, Error> result) {
  if (!result) {
    throw RuntimeError{std::move(result.error())};
  }
}

[[nodiscard]] std::string parameter_repr(const RegisteredOperatorType& row) {
  switch (row.parameter_introspection) {
    case OperatorParameterIntrospection::kNone:
      return "no parameters";
    case OperatorParameterIntrospection::kDeclared:
      return "declared parameters: " + std::string(row.param_schema->stable_id);
    case OperatorParameterIntrospection::kConstructorReplay:
      return "constructor parameters: " + std::string(row.param_schema->stable_id);
    case OperatorParameterIntrospection::kPassThrough:
      return "unknown parameters, passed through by hand-authored row";
  }
  return "invalid parameter introspection";
}

[[nodiscard]] std::string parameter_repr(const RegisteredGraphType& row) {
  switch (row.parameter_introspection) {
    case GraphParameterIntrospection::kNone:
      return "no constructor parameters";
    case GraphParameterIntrospection::kDeclared:
      return "declared constructor parameters: " + std::string(row.param_schema->stable_id);
    case GraphParameterIntrospection::kPassThrough:
      return "unknown constructor parameters, passed through by hand-authored row";
  }
  return "invalid constructor parameter introspection";
}

template <typename EntryT, typename RowT>
[[nodiscard]] std::vector<std::reference_wrapper<const RowT>> enumerate_rows(
    const std::map<std::string, std::unique_ptr<EntryT>, std::less<>>& entries) {
  std::vector<std::reference_wrapper<const RowT>> result;
  result.reserve(entries.size());
  for (const auto& [name, entry] : entries) {
    (void)name;
    result.emplace_back(entry->row);
  }
  return result;
}

}  // namespace

class AuthoringRegistry::Impl {
 public:
  struct OperatorEntry {
    explicit OperatorEntry(RegisteredOperatorType candidate)
        : name(candidate.name), row(std::move(candidate)) {
      row.name = name;
    }
    std::string name;
    RegisteredOperatorType row;
    std::optional<RegisteredOperatorDescription> description;
  };

  struct PayloadEntry {
    explicit PayloadEntry(RegisteredPayloadType candidate)
        : name(candidate.name), row(std::make_unique<RegisteredPayloadType>(std::move(candidate))) {
      row->name = name;
    }
    std::string name;
    std::unique_ptr<RegisteredPayloadType> row;
    std::vector<std::unique_ptr<RegisteredPayloadType>> retired_rows;
  };

  struct GraphEntry {
    explicit GraphEntry(RegisteredGraphType candidate)
        : name(candidate.name), row(std::move(candidate)) {
      row.name = name;
    }
    std::string name;
    RegisteredGraphType row;
    std::optional<RegisteredGraphDescription> description;
  };

  struct ClockEntry {
    explicit ClockEntry(RegisteredClockType candidate)
        : name(candidate.name), row(std::move(candidate)) {
      row.name = name;
    }
    std::string name;
    RegisteredClockType row;
  };

  mutable std::mutex mutex;
  std::map<std::string, std::unique_ptr<OperatorEntry>, std::less<>> operators;
  std::map<std::string, std::unique_ptr<GraphEntry>, std::less<>> graphs;
  std::map<std::string, std::unique_ptr<PayloadEntry>, std::less<>> payloads;
  std::map<std::string, std::unique_ptr<ClockEntry>, std::less<>> clocks;
  std::map<std::string, const ConstructorHandleDescriptor*, std::less<>> constructor_handle_types;
  std::vector<Error> registration_errors;
  std::atomic_bool automatic_registration_failed{};
};

bool OpaqueGraphRef::graph_valid() const noexcept {
  auto state = state_.lock();
  return state && holder_ &&
         holoscan::detail::graph_embedded_valid(state, index_, handle_identity_.get());
}

std::optional<GraphInstancePath> OpaqueGraphRef::path() const {
  auto state = state_.lock();
  return state && holder_
             ? holoscan::detail::graph_embedded_path(state, index_, handle_identity_.get())
             : std::nullopt;
}

PortRecord OpaqueGraphRef::interface_port_record(std::string_view name) const {
  auto state = state_.lock();
  if (!state || !holder_) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"}};
  }
  auto port = holoscan::detail::graph_find_embedded_interface_port(
      state, index_, handle_identity_.get(), name);
  if (!port) {
    throw RuntimeError{port.error()};
  }
  return std::move(*port);
}

std::vector<PortRecord> OpaqueGraphRef::interface_port_records() const {
  auto state = state_.lock();
  if (!state || !holder_) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"}};
  }
  auto ports =
      holoscan::detail::graph_embedded_interface_ports(state, index_, handle_identity_.get());
  if (!ports) {
    throw RuntimeError{std::move(ports).error()};
  }
  return std::move(*ports);
}

OpaquePortRef OpaqueGraphRef::interface_port(std::string_view name) const {
  const PortRecord port = interface_port_record(name);
  OpaquePortRef result;
  result.state_ = state_;
  result.graph_port_ = port.id;
  result.index_ = index_;
  result.graph_handle_identity_ = handle_identity_;
  result.payload_identity_ = port.payload_identity;
  result.direction_ = port.direction;
  return result;
}

ClockKey OpaqueGraphRef::imported_clock(std::string_view name) const {
  auto state = state_.lock();
  if (!state || !holder_) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"}};
  }
  auto key = holoscan::detail::graph_find_embedded_clock_import(
      state, index_, handle_identity_.get(), name);
  if (!key) {
    throw RuntimeError{key.error()};
  }
  return std::move(*key);
}

void OpaqueGraphRef::region_add(RegionHandle& region) const {
  if (!holder_ || region_add_ == nullptr) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"}};
  }
  region_add_(region, holder_.get());
}

void OpaqueGraphRef::bind_clock(Graph& graph, ClockKey child_import, ClockKey parent_clock) const {
  if (!holder_ || bind_clock_ == nullptr) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_CLOCK_BINDING_HANDLE_INVALID"}};
  }
  bind_clock_(graph, holder_.get(), std::move(child_import), std::move(parent_clock));
}

bool OpaqueOperatorRef::graph_valid() const noexcept {
  auto state = state_.lock();
  return state && operator_type_token_ != nullptr &&
         holoscan::detail::graph_operator_valid(
             state, index_, operator_type_token_, handle_identity_.get());
}

std::string_view OpaqueOperatorRef::name() const noexcept {
  auto state = state_.lock();
  return state ? holoscan::detail::graph_operator_name(state, index_, handle_identity_.get())
               : std::string_view{};
}

std::string OpaqueOperatorRef::path() const {
  auto result =
      holoscan::detail::graph_operator_path(state_.lock(), index_, handle_identity_.get());
  if (!result) {
    throw RuntimeError{result.error()};
  }
  return std::move(*result);
}

ExecutionPartitionView OpaqueOperatorRef::execution_partition(
    const ExecutionPartitionsView& partitions) const {
  const auto state = state_.lock();
  if (!state) {
    throw RuntimeError{ErrorCode::kInvalidArgument};
  }
  auto locator = holoscan::detail::graph_operator_locator(state, index_, handle_identity_.get());
  if (!locator) {
    throw RuntimeError{locator.error()};
  }
  return partitions.of_root_locator(*locator);
}

OpaquePortRef OpaqueOperatorRef::port(std::string_view port_name) const {
  if (!graph_valid()) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"}};
  }
  OpaquePortRef result{*this, port_name};
  if (!result) {
    throw RuntimeError{Error{ErrorCode::kNotFound,
                             "AUTHORING_OPERATOR_PORT_UNKNOWN name=" + std::string(port_name)}};
  }
  return result;
}

std::vector<PortRecord> OpaqueOperatorRef::ports() const {
  auto result = holoscan::detail::graph_operator_ports(
      state_.lock(), index_, operator_type_token_, handle_identity_.get());
  if (!result) {
    throw RuntimeError{result.error()};
  }
  return std::move(*result);
}

OpaquePortRef::operator bool() const noexcept {
  if (holder_) {
    return declaration_ != nullptr && (valid_ == nullptr || valid_(holder_.get()));
  }
  auto state = state_.lock();
  if (state && graph_port_.has_value()) {
    if (!graph_handle_identity_ ||
        !holoscan::detail::graph_embedded_valid(state, index_, graph_handle_identity_.get())) {
      return false;
    }
    return holoscan::detail::graph_port_declaration(
               state, *graph_port_, direction_, payload_identity_) != nullptr;
  }
  return state && operator_type_token_ != nullptr &&
         holoscan::detail::graph_operator_valid(
             state, index_, operator_type_token_, handle_identity_.get()) &&
         holoscan::detail::graph_operator_port_declaration(
             state, index_, handle_identity_.get(), port_name_) != nullptr;
}

const void* OpaquePortRef::resolve(const PayloadIdentityRecord& expected,
                                   PortDirection direction) const {
  if (holder_) {
    // Native enum width does not identify the exact C++ enum behind an erased declaration.
    if (has_local_enum_identity(payload_identity_) || has_local_enum_identity(expected)) {
      throw RuntimeError{local_enum_error(ErrorCode::kIncompatible, "GRAPH_PORT_TYPE_MISMATCH")};
    }
    if (declaration_ == nullptr || direction_ != direction ||
        !payload_identities_equal(payload_identity_, expected)) {
      throw RuntimeError{Error{ErrorCode::kIncompatible, "GRAPH_PORT_TYPE_MISMATCH"}};
    }
    if (valid_ != nullptr && !valid_(holder_.get())) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"}};
    }
    const void* declaration = declaration_(holder_.get());
    if (declaration == nullptr) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_PORT_DECLARATION_INVALID"}};
    }
    return declaration;
  }
  auto state = state_.lock();
  if (graph_port_.has_value()) {
    if (!state || !graph_handle_identity_ ||
        !holoscan::detail::graph_embedded_valid(state, index_, graph_handle_identity_.get())) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"}};
    }
    if (direction_ != direction || !payload_identities_equal(payload_identity_, expected)) {
      throw RuntimeError{Error{ErrorCode::kIncompatible, "GRAPH_PORT_TYPE_MISMATCH"}};
    }
    const void* declaration =
        holoscan::detail::graph_port_declaration(state, *graph_port_, direction, expected);
    if (declaration == nullptr) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_PORT_DECLARATION_INVALID"}};
    }
    return declaration;
  }
  auto resolved = holoscan::detail::graph_resolve_operator_port(
      state, index_, operator_type_token_, handle_identity_.get(), port_name_, direction, expected);
  if (!resolved) {
    throw RuntimeError{resolved.error()};
  }
  return *resolved;
}

void OpaqueOperatorRef::partition_add(PartitionHandle& partition,
                                      std::span<const OpaqueOperatorRef> operators) {
  auto owner_state = partition.state_.lock();
  if (!owner_state) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_PARTITION_HANDLE_INVALID"}};
  }
  if (operators.empty()) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_PARTITION_MEMBER_REQUIRED"}};
  }

  std::vector<std::size_t> indexes;
  indexes.reserve(operators.size());
  for (const OpaqueOperatorRef& op : operators) {
    auto op_state = op.state_.lock();
    if (!op_state || op.operator_type_token_ == nullptr || !op.graph_valid()) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_PARTITION_HANDLE_INVALID"}};
    }
    if (owner_state != op_state) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_PARTITION_MEMBER_FOREIGN"}};
    }
    indexes.push_back(op.index_);
  }
  auto result = holoscan::detail::graph_partition_add_operators(
      owner_state, partition.id_, partition.handle_identity_.get(), indexes);
  if (!result) {
    throw RuntimeError{result.error()};
  }
}

void OpaqueOperatorRef::region_add(RegionHandle& region,
                                   std::span<const OpaqueOperatorRef> operators) {
  region_add_members(region, operators, {});
}

void OpaqueOperatorRef::region_add_members(RegionHandle& region,
                                           std::span<const OpaqueOperatorRef> operators,
                                           std::span<const OpaqueGraphRef> graphs) {
  auto owner_state = region.state_.lock();
  if (!owner_state) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_REGION_HANDLE_INVALID"}};
  }
  if (operators.empty() && graphs.empty()) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_REGION_MEMBER_REQUIRED"}};
  }

  std::vector<std::size_t> operator_indexes;
  operator_indexes.reserve(operators.size());
  for (const OpaqueOperatorRef& op : operators) {
    auto op_state = op.state_.lock();
    if (!op_state || op.operator_type_token_ == nullptr || owner_state != op_state ||
        !op.graph_valid()) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_REGION_HANDLE_INVALID"}};
    }
    operator_indexes.push_back(op.index_);
  }
  std::vector<std::size_t> embedded_indexes;
  embedded_indexes.reserve(graphs.size());
  for (const OpaqueGraphRef& graph : graphs) {
    auto graph_state = graph.state_.lock();
    if (!graph_state || !graph.holder_ || owner_state != graph_state ||
        !holoscan::detail::graph_embedded_valid(
            graph_state, graph.index_, graph.handle_identity_.get())) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_REGION_HANDLE_INVALID"}};
    }
    embedded_indexes.push_back(graph.index_);
  }
  auto result = holoscan::detail::graph_region_add_members(
      owner_state, region.id_, region.handle_identity_.get(), operator_indexes, embedded_indexes);
  if (!result) {
    throw RuntimeError{result.error()};
  }
}

void OpaqueOperatorRef::lifecycle_before(GraphLifecycleBuilder& builder,
                                         const OpaqueOperatorRef& before,
                                         const OpaqueOperatorRef& after, LifecycleStage stage) {
  auto owner_state = builder.state_.lock();
  auto before_state = before.state_.lock();
  auto after_state = after.state_.lock();
  if (!owner_state || !before_state || !after_state || before.operator_type_token_ == nullptr ||
      after.operator_type_token_ == nullptr || !before.graph_valid() || !after.graph_valid()) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_HANDLE_INVALID"}};
  }
  auto result = holoscan::detail::graph_lifecycle_before_resolved(
      owner_state, before_state, before.index_, after_state, after.index_, stage);
  if (!result) {
    throw RuntimeError{result.error()};
  }
}

void OpaqueOperatorRef::lifecycle_allow_self_restart(GraphLifecycleBuilder& builder,
                                                     const OpaqueOperatorRef& target) {
  auto owner_state = builder.state_.lock();
  auto target_state = target.state_.lock();
  if (!owner_state || !target_state || target.operator_type_token_ == nullptr ||
      !target.graph_valid()) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_HANDLE_INVALID"}};
  }
  auto result = holoscan::detail::graph_lifecycle_permission_resolved(owner_state,
                                                                      LifecycleRestartOrigin::kSelf,
                                                                      owner_state,
                                                                      std::nullopt,
                                                                      target_state,
                                                                      target.index_);
  if (!result) {
    throw RuntimeError{result.error()};
  }
}

void OpaqueOperatorRef::lifecycle_allow_external_restart(GraphLifecycleBuilder& builder,
                                                         const OpaqueOperatorRef& target) {
  auto owner_state = builder.state_.lock();
  auto target_state = target.state_.lock();
  if (!owner_state || !target_state || target.operator_type_token_ == nullptr ||
      !target.graph_valid()) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_HANDLE_INVALID"}};
  }
  auto result =
      holoscan::detail::graph_lifecycle_permission_resolved(owner_state,
                                                            LifecycleRestartOrigin::kExternal,
                                                            owner_state,
                                                            std::nullopt,
                                                            target_state,
                                                            target.index_);
  if (!result) {
    throw RuntimeError{result.error()};
  }
}

void OpaqueOperatorRef::lifecycle_allow_restart_request(GraphLifecycleBuilder& builder,
                                                        const OpaqueOperatorRef& requester,
                                                        const OpaqueOperatorRef& target) {
  auto owner_state = builder.state_.lock();
  auto requester_state = requester.state_.lock();
  auto target_state = target.state_.lock();
  if (!owner_state || !requester_state || !target_state ||
      requester.operator_type_token_ == nullptr || target.operator_type_token_ == nullptr ||
      !requester.graph_valid() || !target.graph_valid()) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_HANDLE_INVALID"}};
  }
  auto result = holoscan::detail::graph_lifecycle_permission_resolved(
      owner_state,
      LifecycleRestartOrigin::kOperatorRequest,
      requester_state,
      requester.index_,
      target_state,
      target.index_);
  if (!result) {
    throw RuntimeError{result.error()};
  }
}

std::shared_ptr<holoscan::detail::GraphState> OpaquePortConfigurator::require_live_state() const {
  auto state = require_graph_state(state_, "GRAPH_HANDLE_INVALID");
  if (graph_handle_identity_ &&
      !holoscan::detail::graph_embedded_valid(state, graph_index_, graph_handle_identity_.get())) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"}};
  }
  return state;
}

OpaquePortConfigurator& OpaquePortConfigurator::queue_depth(std::uint32_t depth) {
  if (direction_ != PortDirection::kInput) {
    throw RuntimeError{
        Error{ErrorCode::kInvalidArgument, "GRAPH_PORT_CONFIGURATOR_DIRECTION_MISMATCH"}};
  }
  apply_graph_configuration(holoscan::detail::graph_configure_input_queue(
      require_live_state(), port_, port_handle_identity_.get(), depth));
  return *this;
}

OpaquePortConfigurator& OpaquePortConfigurator::max_emits_per_compute(std::uint32_t count) {
  if (direction_ != PortDirection::kOutput) {
    throw RuntimeError{
        Error{ErrorCode::kInvalidArgument, "GRAPH_PORT_CONFIGURATOR_DIRECTION_MISMATCH"}};
  }
  apply_graph_configuration(holoscan::detail::graph_configure_output_emits(
      require_live_state(), port_, port_handle_identity_.get(), count));
  return *this;
}

OpaquePortConfigurator& OpaquePortConfigurator::late_joiner_history_depth(std::uint32_t depth) {
  if (direction_ != PortDirection::kOutput) {
    throw RuntimeError{
        Error{ErrorCode::kInvalidArgument, "GRAPH_PORT_CONFIGURATOR_DIRECTION_MISMATCH"}};
  }
  apply_graph_configuration(holoscan::detail::graph_configure_output_history(
      require_live_state(), port_, port_handle_identity_.get(), depth));
  return *this;
}

OpaquePortConfigurator& OpaquePortConfigurator::diagnostic_alias(std::string_view alias) {
  apply_graph_configuration(holoscan::detail::graph_configure_port_alias(
      require_live_state(), port_, port_handle_identity_.get(), alias));
  return *this;
}

bool OpaqueFlowRef::graph_valid() const noexcept {
  auto state = state_.lock();
  return state &&
         holoscan::detail::graph_flow_valid(state, edge_, source_, target_, handle_identity_.get());
}

void OpaqueFlowRef::validate_owner(Graph& graph) const {
  auto state = state_.lock();
  auto owner = holoscan::detail::GraphPayloadRegistrationAccess::state(graph).lock();
  if (!state || state != owner ||
      !holoscan::detail::graph_flow_valid(state, edge_, source_, target_, handle_identity_.get())) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_FLOW_HANDLE_INVALID"}};
  }
}

OpaqueFlowRef& OpaqueFlowRef::queue_policy(QueuePolicy policy) {
  apply_graph_configuration(holoscan::detail::graph_configure_flow_policy(
      require_graph_state(state_, "GRAPH_HANDLE_INVALID"), edge_, handle_identity_.get(), policy));
  return *this;
}

OpaqueFlowRef& OpaqueFlowRef::queue_depth(std::uint32_t depth) {
  apply_graph_configuration(holoscan::detail::graph_configure_flow_depth(
      require_graph_state(state_, "GRAPH_HANDLE_INVALID"), edge_, handle_identity_.get(), depth));
  return *this;
}

OpaqueFlowRef& OpaqueFlowRef::topic_name(std::string_view name) {
  apply_graph_configuration(holoscan::detail::graph_configure_flow_topic(
      require_graph_state(state_, "GRAPH_HANDLE_INVALID"), edge_, handle_identity_.get(), name));
  return *this;
}

OpaqueFlowRef& OpaqueFlowRef::memory_conversion(MemoryConversionPolicy policy) {
  apply_graph_configuration(holoscan::detail::graph_configure_flow_memory(
      require_graph_state(state_, "GRAPH_HANDLE_INVALID"), edge_, handle_identity_.get(), policy));
  return *this;
}

OpaqueFlowRef& OpaqueFlowRef::max_inflight_samples(std::uint32_t count) {
  apply_graph_configuration(holoscan::detail::graph_configure_flow_inflight(
      require_graph_state(state_, "GRAPH_HANDLE_INVALID"), edge_, handle_identity_.get(), count));
  return *this;
}
AuthoringRegistry::AuthoringRegistry() : impl_(std::make_unique<Impl>()) {
  // Core owns the built-in callback type's stable name in every registry from the start, so no
  // provider row can claim it first and callback-row admission never depends on import order.
  const ConstructorHandleDescriptor& callback = callback_handle_type().descriptor();
  impl_->constructor_handle_types.emplace(std::string(callback.name), &callback);
  const auto add_payload = [this](RegisteredPayloadType row) {
    auto result = register_payload(std::move(row));
    if (!result) {
      impl_->registration_errors.push_back(std::move(result.error()));
    }
  };
  for (auto& row : detail::make_builtin_other_payloads()) {
    add_payload(std::move(row));
  }
  for (auto& row : detail::make_builtin_integer_payloads()) {
    add_payload(std::move(row));
  }

  const auto add_clock = [this](RegisteredClockType row) {
    auto result = register_clock(std::move(row));
    if (!result) {
      impl_->registration_errors.push_back(std::move(result.error()));
    }
  };
  add_clock(detail::make_registered_clock<RealtimeClock>("holoscan.RealtimeClock"));
  add_clock(detail::make_registered_clock<ManualClock>("holoscan.ManualClock"));
  add_clock(detail::make_registered_clock<SyntheticClock>("holoscan.SyntheticClock"));
}

AuthoringRegistry::~AuthoringRegistry() = default;

void AuthoringRegistry::record_automatic_registration_failure(
    AuthoringRegistry* registry) noexcept {
  if (registry == nullptr) {
    process_automatic_registration_failure_latch().store(true, std::memory_order_relaxed);
    return;
  }
  registry->impl_->automatic_registration_failed.store(true, std::memory_order_relaxed);
}

// Keep the callback inventory as one auditable linear validation sequence.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> AuthoringRegistry::register_operator(RegisteredOperatorType row) {
  const std::string_view invalid_field = [&]() -> std::string_view {
    if (row.name.empty())
      return "name";
    if (row.param_type == nullptr)
      return "param_type";
    if (row.add_to_graph == nullptr)
      return "add_to_graph";
    if (row.add_to_graph_from_assignments == nullptr)
      return "add_to_graph_from_assignments";
    if (row.set_params == nullptr)
      return "set_params";
    if (row.port == nullptr)
      return "port";
    if (row.ports == nullptr)
      return "ports";
    if (row.region_add == nullptr)
      return "region_add";
    if (row.partition_add == nullptr)
      return "partition_add";
    if (row.lifecycle_before == nullptr)
      return "lifecycle_before";
    if (row.lifecycle_allow_self_restart == nullptr)
      return "lifecycle_allow_self_restart";
    if (row.lifecycle_allow_external_restart == nullptr)
      return "lifecycle_allow_external_restart";
    if (row.lifecycle_allow_restart_request == nullptr)
      return "lifecycle_allow_restart_request";
    if (!valid_parameter_introspection(row))
      return "parameter_introspection";
    if (!valid_required_parameters(row))
      return "required_parameters";
    if (!valid_constructor_inputs(row))
      return "constructor_inputs";
    return {};
  }();
  if (!invalid_field.empty()) {
    return make_unexpected(invalid_row("operator", invalid_field));
  }
  const std::span<const ConstructorHandleInput> handles =
      row.constructor_inputs == nullptr ? std::span<const ConstructorHandleInput>{}
                                        : row.constructor_inputs->handles;
  auto entry = std::make_unique<Impl::OperatorEntry>(std::move(row));
  const std::lock_guard lock(impl_->mutex);
  if (impl_->operators.contains(entry->name)) {
    return make_unexpected(duplicate_row("operator", entry->name));
  }
  // One stable name identifies one descriptor; a second descriptor under the same name would let
  // name-based lookup and generated annotations disagree with exact descriptor identity. Check
  // earlier inputs of this row as well as the published catalog, because the row's own types are
  // published only after the whole row is accepted. Reusing one descriptor stays valid.
  for (auto current = handles.begin(); current != handles.end(); ++current) {
    const auto found = impl_->constructor_handle_types.find(current->type->name);
    const bool catalog_conflict =
        found != impl_->constructor_handle_types.end() && found->second != current->type;
    const bool row_conflict =
        std::any_of(handles.begin(), current, [&](const ConstructorHandleInput& earlier) {
          return earlier.type != current->type && earlier.type->name == current->type->name;
        });
    if (catalog_conflict || row_conflict) {
      return make_unexpected(Error{
          ErrorCode::kDuplicateName,
          "AUTHORING_CONSTRUCTOR_HANDLE_TYPE_CONFLICT type=" + std::string(current->type->name)});
    }
  }
  // Stage every allocation in local maps, then publish both catalogs by node transfer. merge()
  // allocates nothing, so a failed allocation leaves the Operator and type catalogs unchanged and
  // the same row can be registered again.
  decltype(impl_->operators) staged_operators;
  decltype(impl_->constructor_handle_types) staged_types;
  staged_operators.emplace(entry->name, std::move(entry));
  for (const ConstructorHandleInput& input : handles) {
    if (!impl_->constructor_handle_types.contains(input.type->name)) {
      staged_types.emplace(std::string(input.type->name), input.type);
    }
  }
  impl_->operators.merge(staged_operators);
  impl_->constructor_handle_types.merge(staged_types);
  return {};
}

expected<std::reference_wrapper<const ConstructorHandleDescriptor>, Error>
AuthoringRegistry::constructor_handle_type(std::string_view name) const {
  const std::lock_guard lock(impl_->mutex);
  const auto found = impl_->constructor_handle_types.find(name);
  if (found == impl_->constructor_handle_types.end()) {
    return make_unexpected(unknown_row("constructor handle type", name));
  }
  return std::cref(*found->second);
}

expected<void, Error> AuthoringRegistry::register_graph(RegisteredGraphType row) {
  if (row.name.empty() || row.param_type == nullptr || row.embed == nullptr ||
      row.embed_from_assignments == nullptr || row.ports == nullptr ||
      !valid_parameter_introspection(row)) {
    return make_unexpected(invalid_row("graph"));
  }
  auto entry = std::make_unique<Impl::GraphEntry>(std::move(row));
  const std::lock_guard lock(impl_->mutex);
  if (impl_->graphs.contains(entry->name)) {
    return make_unexpected(duplicate_row("graph", entry->name));
  }
  impl_->graphs.emplace(entry->name, std::move(entry));
  return {};
}

expected<RegisteredGraphEmbedding, Error> embed_registered_graph(
    Graph& graph, const RegisteredGraphType& row, std::string_view name,
    std::span<const ParameterAssignment> assignments) {
  struct Context {
    Graph& graph;
    const RegisteredGraphType& row;
    std::string_view name;
    std::span<const ParameterAssignment> assignments;
    std::optional<RegisteredGraphEmbedding> result;
  } context{graph, row, name, assignments, std::nullopt};

  const auto operation = [](void* opaque) -> expected<void, Error> {
    auto& call = *static_cast<Context*>(opaque);
    auto embedded = call.row.embed_from_assignments(call.graph, call.name, call.assignments);
    if (!embedded) {
      return make_unexpected(std::move(embedded).error());
    }
    auto ports = call.row.ports(*embedded);
    call.result.emplace(RegisteredGraphEmbedding{std::move(*embedded), std::move(ports)});
    return {};
  };
  auto applied = holoscan::detail::graph_apply_atomic_mutation(
      holoscan::detail::GraphPayloadRegistrationAccess::state(graph).lock(), &context, operation);
  if (!applied) {
    return make_unexpected(std::move(applied).error());
  }
  if (!context.result) {
    return make_unexpected(Error{ErrorCode::kFailure, "AUTHORING_GRAPH_EMBED_RESULT_MISSING"});
  }
  return std::move(*context.result);
}

namespace detail {

/// Compare an erased Operator reference with the declaration a Graph resolves by name.
struct RegisteredOperatorAccess {
  [[nodiscard]] static bool names_declaration(
      const OpaqueOperatorRef& op, const std::shared_ptr<holoscan::detail::GraphState>& state,
      const holoscan::detail::OperatorRegistration& declared) noexcept {
    return op.state_.lock() == state && op.index_ == declared.index &&
           op.handle_identity_ == declared.handle_identity && op.graph_valid();
  }
};

}  // namespace detail

expected<RegisteredOperatorAddition, Error> add_registered_operator(
    Graph& graph, const RegisteredOperatorType& row, std::string_view name,
    const OperatorConstructionInputs& inputs) {
  const RegisteredConstructorInputs* const table = row.constructor_inputs;
  if (row.ports == nullptr || (table == nullptr ? row.add_to_graph_from_assignments == nullptr
                                                : table->add_to_graph_from_inputs == nullptr)) {
    return make_unexpected(invalid_row("operator"));
  }
  if (table == nullptr && !inputs.handles.empty()) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "AUTHORING_CONSTRUCTOR_HANDLE_UNSUPPORTED"});
  }
  if (table != nullptr) {
    auto validated = validate_constructor_handles(*table, inputs);
    if (!validated) {
      return make_unexpected(std::move(validated).error());
    }
  }
  const auto state = holoscan::detail::GraphPayloadRegistrationAccess::state(graph).lock();
  struct Context {
    Graph& graph;
    const std::shared_ptr<holoscan::detail::GraphState>& state;
    const RegisteredOperatorType& row;
    std::string_view name;
    const OperatorConstructionInputs& inputs;
    std::optional<RegisteredOperatorAddition> result;
  } context{graph, state, row, name, inputs, std::nullopt};

  const auto operation = [](void* opaque) -> expected<void, Error> {
    auto& call = *static_cast<Context*>(opaque);
    const RegisteredConstructorInputs* const table = call.row.constructor_inputs;
    // A name declared before the thunk ran cannot identify the thunk's new declaration.
    const bool name_was_declared =
        holoscan::detail::graph_find_operator(call.state, call.name).has_value();
    auto added =
        table == nullptr
            ? call.row.add_to_graph_from_assignments(call.graph, call.name, call.inputs.values)
            : table->add_to_graph_from_inputs(call.graph, call.name, call.inputs);
    if (!added) {
      return make_unexpected(std::move(added).error());
    }
    const auto declared = holoscan::detail::graph_find_operator(call.state, call.name);
    if (name_was_declared || !declared ||
        !detail::RegisteredOperatorAccess::names_declaration(*added, call.state, *declared)) {
      return make_unexpected(Error{ErrorCode::kFailure, "AUTHORING_OPERATOR_ADD_RESULT_INVALID"});
    }
    if (table != nullptr) {
      // Wrap the declaration's factory once; every later copy and replay inherits the record.
      auto record = make_binding_record(*table, call.inputs);
      const std::lock_guard lock(call.state->mutex_);
      auto& factory = call.state->operators_[declared->index].factory;
      factory = std::make_shared<detail::ConstructorBindingFactory>(std::move(factory),
                                                                    std::move(record));
    }
    auto ports = call.row.ports(*added);
    call.result.emplace(RegisteredOperatorAddition{std::move(*added), std::move(ports)});
    return {};
  };
  auto applied = holoscan::detail::graph_apply_atomic_mutation(state, &context, operation);
  if (!applied) {
    return make_unexpected(std::move(applied).error());
  }
  if (!context.result) {
    return make_unexpected(Error{ErrorCode::kFailure, "AUTHORING_OPERATOR_ADD_RESULT_MISSING"});
  }
  return std::move(*context.result);
}

expected<void, Error> AuthoringRegistry::register_payload(RegisteredPayloadType row) {
  const RegisteredChannelThunks* const channel = row.channel;
  if (row.name.empty() || row.bind_spec_input == nullptr || row.bind_spec_output == nullptr ||
      channel == nullptr || channel->create_publisher == nullptr ||
      channel->create_name_publisher == nullptr || channel->create_subscriber == nullptr ||
      channel->create_name_subscriber == nullptr || channel->emit_scalar == nullptr ||
      channel->emit_captured_scalar == nullptr || channel->emit_object == nullptr ||
      channel->emit_captured_object == nullptr || channel->emit_schema == nullptr ||
      channel->emit_captured_schema == nullptr || channel->loan_tensor == nullptr ||
      channel->emit_tensor_loan == nullptr || channel->emit_tensor_copy == nullptr ||
      channel->publisher_close == nullptr || channel->update_connections == nullptr ||
      channel->has_subscribers == nullptr || channel->make_capture_time == nullptr ||
      channel->report_clock_health == nullptr || channel->try_receive_scalar == nullptr ||
      channel->wait_and_receive_scalar == nullptr || channel->try_receive_object == nullptr ||
      channel->wait_and_receive_object == nullptr || channel->try_receive_schema == nullptr ||
      channel->wait_and_receive_schema == nullptr || channel->try_receive_tensor == nullptr ||
      channel->wait_and_receive_tensor == nullptr || channel->wait == nullptr ||
      channel->subscriber_readiness == nullptr || channel->subscriber_close == nullptr ||
      channel->refresh_connections == nullptr || channel->has_samples == nullptr ||
      channel->has_publishers == nullptr || row.add_flow == nullptr ||
      row.export_input == nullptr || row.export_output == nullptr ||
      row.configure_input == nullptr || row.configure_output == nullptr ||
      row.import_topic == nullptr || row.export_topic == nullptr ||
      row.import_descriptor_topic == nullptr || row.export_descriptor_topic == nullptr ||
      row.make_interface_ports == nullptr) {
    return make_unexpected(invalid_row("payload"));
  }
  if (has_local_enum_identity(row.identity)) {
    return make_unexpected(local_enum_error(ErrorCode::kInvalidArgument,
                                            "AUTHORING_REGISTRY_ROW_INVALID kind=payload"));
  }
  auto entry = std::make_unique<Impl::PayloadEntry>(std::move(row));
  const std::lock_guard lock(impl_->mutex);
  const auto found = impl_->payloads.find(entry->name);
  if (found != impl_->payloads.end()) {
    // A generated facade may be installed before its owning native component registers. Preserve
    // the facade generation when the later native row proves it represents the same payload.
    if (found->second->row->generated_schema_facade && !entry->row->generated_schema_facade &&
        payload_identities_equal(found->second->row->identity, entry->row->identity)) {
      return {};
    }
    return make_unexpected(duplicate_row("payload", entry->name));
  }
  const auto duplicate_identity =
      std::find_if(impl_->payloads.begin(), impl_->payloads.end(), [&](const auto& existing) {
        return payload_identities_equal(existing.second->row->identity, entry->row->identity);
      });
  if (duplicate_identity != impl_->payloads.end()) {
    return make_unexpected(duplicate_row("payload_identity", entry->name));
  }
  impl_->payloads.emplace(entry->name, std::move(entry));
  return {};
}

expected<void, Error> AuthoringRegistry::install_payload_facade(RegisteredPayloadType row) {
  if (!row.generated_schema_facade || row.name.empty() || row.add_flow == nullptr ||
      row.export_input == nullptr || row.export_output == nullptr ||
      row.configure_input == nullptr || row.configure_output == nullptr ||
      row.import_topic == nullptr || row.export_topic == nullptr ||
      row.import_descriptor_topic == nullptr || row.export_descriptor_topic == nullptr ||
      row.make_interface_ports == nullptr || row.receive_schema == nullptr ||
      row.emit_schema == nullptr || row.validate_schema == nullptr) {
    return make_unexpected(invalid_row("payload_facade"));
  }
  if (has_local_enum_identity(row.identity)) {
    return make_unexpected(local_enum_error(ErrorCode::kInvalidArgument,
                                            "AUTHORING_REGISTRY_ROW_INVALID kind=payload_facade"));
  }
  auto entry = std::make_unique<Impl::PayloadEntry>(std::move(row));
  const std::lock_guard lock(impl_->mutex);
  const auto found = impl_->payloads.find(entry->name);
  if (found == impl_->payloads.end()) {
    const auto duplicate_identity =
        std::find_if(impl_->payloads.begin(), impl_->payloads.end(), [&](const auto& existing) {
          return payload_identities_equal(existing.second->row->identity, entry->row->identity);
        });
    if (duplicate_identity != impl_->payloads.end()) {
      return make_unexpected(duplicate_row("payload_identity", entry->name));
    }
    impl_->payloads.emplace(entry->name, std::move(entry));
    return {};
  }
  auto& existing = *found->second;
  if (!payload_identities_equal(existing.row->identity, entry->row->identity)) {
    return make_unexpected(duplicate_row("payload_facade_identity", entry->name));
  }
  if (existing.row->generated_schema_facade) {
    if (existing.row->receive_schema == entry->row->receive_schema &&
        existing.row->emit_schema == entry->row->emit_schema) {
      return {};
    }
    return make_unexpected(duplicate_row("payload_facade", entry->name));
  }
  // Every thunk produced by make_registered_payload<T> depends on whether the compiling DSO sees
  // PythonSchemaNativeTraits<T>. Publish one immutable generation instead of mixing its non-facade
  // thunks with the hidden-visibility codec DSO's facade-enabled thunks. Retaining prior
  // generations keeps every escaped registry reference valid and race-free until registry shutdown.
  entry->row->name = existing.name;
  existing.retired_rows.reserve(existing.retired_rows.size() + 1U);
  existing.retired_rows.push_back(std::move(existing.row));
  existing.row = std::move(entry->row);
  return {};
}

expected<void, Error> AuthoringRegistry::register_clock(RegisteredClockType row) {
  if (row.name.empty() || row.add_to_graph == nullptr) {
    return make_unexpected(invalid_row("clock"));
  }
  auto entry = std::make_unique<Impl::ClockEntry>(std::move(row));
  const std::lock_guard lock(impl_->mutex);
  if (impl_->clocks.contains(entry->name)) {
    return make_unexpected(duplicate_row("clock", entry->name));
  }
  impl_->clocks.emplace(entry->name, std::move(entry));
  return {};
}

bool AuthoringRegistry::register_operator_automatically(RegisteredOperatorType row) noexcept {
  try {
    auto result = register_operator(std::move(row));
    if (result) {
      return true;
    }
    record_automatic_registration_failure(this);
    const std::lock_guard lock(impl_->mutex);
    impl_->registration_errors.push_back(std::move(result.error()));
  } catch (...) {
    record_automatic_registration_failure(this);
    try {
      const std::lock_guard lock(impl_->mutex);
      impl_->registration_errors.push_back(
          Error{ErrorCode::kFailure, "AUTHORING_REGISTRY_AUTOMATIC_FAILED"});
    } catch (...) {
      return false;
    }
  }
  return false;
}

bool AuthoringRegistry::register_graph_automatically(RegisteredGraphType row) noexcept {
  try {
    auto result = register_graph(std::move(row));
    if (result) {
      return true;
    }
    record_automatic_registration_failure(this);
    const std::lock_guard lock(impl_->mutex);
    impl_->registration_errors.push_back(std::move(result.error()));
  } catch (...) {
    record_automatic_registration_failure(this);
    try {
      const std::lock_guard lock(impl_->mutex);
      impl_->registration_errors.push_back(
          Error{ErrorCode::kFailure, "AUTHORING_REGISTRY_AUTOMATIC_FAILED"});
    } catch (...) {  // NOLINT(bugprone-empty-catch)
      // Recording an allocation failure is best effort in this noexcept path.
    }
  }
  return false;
}

bool AuthoringRegistry::register_payload_automatically(RegisteredPayloadType row) noexcept {
  try {
    auto result = register_payload(std::move(row));
    if (result) {
      return true;
    }
    record_automatic_registration_failure(this);
    const std::lock_guard lock(impl_->mutex);
    impl_->registration_errors.push_back(std::move(result.error()));
  } catch (...) {
    record_automatic_registration_failure(this);
    try {
      const std::lock_guard lock(impl_->mutex);
      impl_->registration_errors.push_back(
          Error{ErrorCode::kFailure, "AUTHORING_REGISTRY_AUTOMATIC_FAILED"});
    } catch (...) {
      return false;
    }
  }
  return false;
}

bool AuthoringRegistry::install_payload_facade_automatically(RegisteredPayloadType row) noexcept {
  try {
    auto result = install_payload_facade(std::move(row));
    if (result) {
      return true;
    }
    record_automatic_registration_failure(this);
    const std::lock_guard lock(impl_->mutex);
    impl_->registration_errors.push_back(std::move(result.error()));
  } catch (...) {
    record_automatic_registration_failure(this);
    try {
      const std::lock_guard lock(impl_->mutex);
      impl_->registration_errors.push_back(
          Error{ErrorCode::kFailure, "AUTHORING_REGISTRY_AUTOMATIC_FAILED"});
    } catch (...) {
      return false;
    }
  }
  return false;
}

bool AuthoringRegistry::register_clock_automatically(RegisteredClockType row) noexcept {
  try {
    auto result = register_clock(std::move(row));
    if (result) {
      return true;
    }
    record_automatic_registration_failure(this);
    const std::lock_guard lock(impl_->mutex);
    impl_->registration_errors.push_back(std::move(result.error()));
  } catch (...) {
    record_automatic_registration_failure(this);
    try {
      const std::lock_guard lock(impl_->mutex);
      impl_->registration_errors.push_back(
          Error{ErrorCode::kFailure, "AUTHORING_REGISTRY_AUTOMATIC_FAILED"});
    } catch (...) {
      return false;
    }
  }
  return false;
}

expected<void, Error> AuthoringRegistry::registration_status() const {
  const std::lock_guard lock(impl_->mutex);
  if (!impl_->registration_errors.empty()) {
    return make_unexpected(impl_->registration_errors.front());
  }
  if (impl_->automatic_registration_failed.load(std::memory_order_relaxed) ||
      process_automatic_registration_failure_latch().load(std::memory_order_relaxed)) {
    return make_unexpected(Error{ErrorCode::kFailure});
  }
  return {};
}

expected<std::reference_wrapper<const RegisteredOperatorType>, Error>
AuthoringRegistry::operator_type(std::string_view name) const {
  const std::lock_guard lock(impl_->mutex);
  const auto found = impl_->operators.find(name);
  if (found == impl_->operators.end()) {
    return make_unexpected(unknown_row("operator", name));
  }
  return std::cref(found->second->row);
}

expected<std::reference_wrapper<const RegisteredGraphType>, Error> AuthoringRegistry::graph_type(
    std::string_view name) const {
  const std::lock_guard lock(impl_->mutex);
  const auto found = impl_->graphs.find(name);
  if (found == impl_->graphs.end()) {
    return make_unexpected(unknown_row("graph", name));
  }
  return std::cref(found->second->row);
}

expected<std::reference_wrapper<const RegisteredPayloadType>, Error>
AuthoringRegistry::payload_type(std::string_view name) const {
  const std::lock_guard lock(impl_->mutex);
  const auto found = impl_->payloads.find(name);
  if (found == impl_->payloads.end()) {
    return make_unexpected(unknown_row("payload", name));
  }
  return std::cref(*found->second->row);
}

expected<std::reference_wrapper<const RegisteredPayloadType>, Error>
AuthoringRegistry::payload_type(const PayloadIdentityRecord& identity) const {
  const std::lock_guard lock(impl_->mutex);
  const auto found =
      std::find_if(impl_->payloads.begin(), impl_->payloads.end(), [&](const auto& candidate) {
        return payload_identities_equal(candidate.second->row->identity, identity);
      });
  if (found == impl_->payloads.end()) {
    return make_unexpected(unknown_row("payload_identity", "<identity>"));
  }
  return std::cref(*found->second->row);
}

expected<std::reference_wrapper<const RegisteredClockType>, Error> AuthoringRegistry::clock_type(
    std::string_view name) const {
  const std::lock_guard lock(impl_->mutex);
  const auto found = impl_->clocks.find(name);
  if (found == impl_->clocks.end()) {
    return make_unexpected(unknown_row("clock", name));
  }
  return std::cref(found->second->row);
}

std::vector<std::reference_wrapper<const RegisteredOperatorType>> AuthoringRegistry::operators()
    const {
  const std::lock_guard lock(impl_->mutex);
  return enumerate_rows<Impl::OperatorEntry, RegisteredOperatorType>(impl_->operators);
}

std::vector<std::reference_wrapper<const RegisteredGraphType>> AuthoringRegistry::graphs() const {
  const std::lock_guard lock(impl_->mutex);
  return enumerate_rows<Impl::GraphEntry, RegisteredGraphType>(impl_->graphs);
}

std::vector<std::reference_wrapper<const RegisteredPayloadType>> AuthoringRegistry::payloads()
    const {
  const std::lock_guard lock(impl_->mutex);
  std::vector<std::reference_wrapper<const RegisteredPayloadType>> result;
  result.reserve(impl_->payloads.size());
  for (const auto& [name, entry] : impl_->payloads) {
    (void)name;
    result.emplace_back(*entry->row);
  }
  return result;
}

std::vector<std::reference_wrapper<const RegisteredClockType>> AuthoringRegistry::clocks() const {
  const std::lock_guard lock(impl_->mutex);
  return enumerate_rows<Impl::ClockEntry, RegisteredClockType>(impl_->clocks);
}

expected<RegisteredOperatorDescription, Error> AuthoringRegistry::describe_operator(
    std::string_view name) {
  RegisteredOperatorType row;
  {
    const std::lock_guard lock(impl_->mutex);
    const auto found = impl_->operators.find(name);
    if (found == impl_->operators.end()) {
      return make_unexpected(unknown_row("operator", name));
    }
    const auto& description = found->second->description;
    if (description.has_value()) {
      return description.value();
    }
    row = found->second->row;
  }

  RegisteredOperatorDescription harvested{
      .name = std::string(row.name),
      .param_schema = row.param_schema,
      .parameter_introspection = row.parameter_introspection,
      .required_parameters =
          [&] {
            std::vector<std::string> required;
            required.reserve(row.required_parameters.size());
            for (const std::string_view name : row.required_parameters) {
              required.emplace_back(name);
            }
            return required;
          }(),
      .parameter_repr = parameter_repr(row),
  };
  if (row.constructor_inputs != nullptr) {
    for (const ConstructorHandleInput& input : row.constructor_inputs->handles) {
      harvested.constructor_handles.push_back(RegisteredConstructorHandleDescription{
          .name = std::string(input.name),
          .description = std::string(input.description),
          .type_name = std::string(input.type->name),
          .required = input.required,
      });
    }
  }
  try {
    Graph scratch{"authoring_registry_probe"};
    // A row with constructor handles is described without any handle or external resource.
    auto op = row.constructor_inputs != nullptr
                  ? row.constructor_inputs->add_for_description(scratch, "operator")
                  : row.add_to_graph_from_assignments(scratch, "operator", {});
    if (!op) {
      return make_unexpected(std::move(op).error());
    }
    if (!*op) {
      return make_unexpected(
          Error{ErrorCode::kFailure, "AUTHORING_OPERATOR_HARVEST_REFERENCE_INVALID"});
    }
    for (const PortRecord& port : row.ports(*op)) {
      harvested.ports.push_back(RegisteredOperatorPort{
          .name = port.name,
          .direction = port.direction,
          .payload_identity = port.payload_identity,
      });
    }
  } catch (const RuntimeError& error) {
    return make_unexpected(make_error(error));
  } catch (const std::exception& error) {
    return make_unexpected(Error{
        ErrorCode::kFailure, "AUTHORING_OPERATOR_HARVEST_FAILED: " + std::string(error.what())});
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kFailure, "AUTHORING_OPERATOR_HARVEST_FAILED"});
  }

  const std::lock_guard lock(impl_->mutex);
  const auto found = impl_->operators.find(name);
  if (found == impl_->operators.end()) {
    return make_unexpected(unknown_row("operator", name));
  }
  auto& description = found->second->description;
  if (!description.has_value()) {
    description = harvested;
  }
  return description.value_or(harvested);
}

expected<RegisteredGraphDescription, Error> AuthoringRegistry::describe_graph(
    std::string_view name) {
  RegisteredGraphType row;
  {
    const std::lock_guard lock(impl_->mutex);
    const auto found = impl_->graphs.find(name);
    if (found == impl_->graphs.end()) {
      return make_unexpected(unknown_row("graph", name));
    }
    const auto& description = found->second->description;
    if (description.has_value()) {
      return description.value();
    }
    row = found->second->row;
  }

  RegisteredGraphDescription harvested{
      .name = std::string(row.name),
      .param_schema = row.param_schema,
      .parameter_introspection = row.parameter_introspection,
      .parameter_repr = parameter_repr(row),
  };
  try {
    Graph scratch{"authoring_registry_graph_probe"};
    const std::span<const ParameterAssignment> assignments{};
    auto embedded = row.embed_from_assignments(scratch, "graph", assignments);
    if (!embedded || !*embedded) {
      return make_unexpected(
          embedded ? Error{ErrorCode::kFailure, "AUTHORING_GRAPH_HARVEST_REFERENCE_INVALID"}
                   : std::move(embedded).error());
    }
    for (const PortRecord& port : row.ports(*embedded)) {
      harvested.ports.push_back(RegisteredGraphPort{
          .name = port.name,
          .direction = port.direction,
          .payload_identity = port.payload_identity,
      });
    }
  } catch (const RuntimeError& error) {
    return make_unexpected(make_error(error));
  } catch (const std::exception& error) {
    return make_unexpected(
        Error{ErrorCode::kFailure, "AUTHORING_GRAPH_HARVEST_FAILED: " + std::string(error.what())});
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kFailure, "AUTHORING_GRAPH_HARVEST_FAILED"});
  }

  const std::lock_guard lock(impl_->mutex);
  const auto found = impl_->graphs.find(name);
  if (found == impl_->graphs.end()) {
    return make_unexpected(unknown_row("graph", name));
  }
  if (!found->second->description) {
    found->second->description = harvested;
  }
  return found->second->description.value_or(harvested);
}

AuthoringRegistry& authoring_registry() {
  static AuthoringRegistry registry;
  return registry;
}

bool payload_identities_equal(const PayloadIdentityRecord& left,
                              const PayloadIdentityRecord& right) noexcept {
  return holoscan::detail::payload_identity_exactly_equal(left, right);
}

}  // namespace holoscan::authoring
