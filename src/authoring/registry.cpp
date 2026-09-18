/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/authoring/registry.hpp>

#include <holoscan/authoring/register_clock.hpp>
#include <holoscan/authoring/register_payload.hpp>

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

#include "core/plan_record_contracts.hpp"

#include <holoscan/core/domain/tensor.hpp>
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

[[nodiscard]] Error invalid_row(std::string_view kind) {
  return Error{ErrorCode::kInvalidArgument,
               "AUTHORING_REGISTRY_ROW_INVALID kind=" + std::string(kind)};
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
        : name(candidate.name), row(std::move(candidate)) {
      row.name = name;
    }
    std::string name;
    RegisteredPayloadType row;
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
  std::map<std::string, std::unique_ptr<PayloadEntry>, std::less<>> payloads;
  std::map<std::string, std::unique_ptr<ClockEntry>, std::less<>> clocks;
  std::vector<Error> registration_errors;
  std::atomic_bool automatic_registration_failed{};
};

bool OpaqueOperatorRef::graph_valid() const noexcept {
  auto state = state_.lock();
  return state && operator_type_token_ != nullptr &&
         holoscan::detail::graph_operator_valid(state, index_, operator_type_token_);
}

std::string_view OpaqueOperatorRef::name() const noexcept {
  auto state = state_.lock();
  return state ? holoscan::detail::graph_operator_name(state, index_) : std::string_view{};
}

OpaquePortRef::operator bool() const noexcept {
  if (holder_) {
    return declaration_ != nullptr && (valid_ == nullptr || valid_(holder_.get()));
  }
  auto state = state_.lock();
  return state && operator_type_token_ != nullptr &&
         holoscan::detail::graph_operator_valid(state, index_, operator_type_token_) &&
         holoscan::detail::graph_operator_port_declaration(state, index_, port_name_) != nullptr;
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
  auto resolved = holoscan::detail::graph_resolve_operator_port(
      state, index_, operator_type_token_, port_name_, direction, expected);
  if (!resolved) {
    throw RuntimeError{resolved.error()};
  }
  return *resolved;
}

void OpaqueOperatorRef::lifecycle_before(GraphLifecycleBuilder& builder,
                                         const OpaqueOperatorRef& before,
                                         const OpaqueOperatorRef& after, LifecycleStage stage) {
  auto owner_state = builder.state_.lock();
  auto before_state = before.state_.lock();
  auto after_state = after.state_.lock();
  if (!owner_state || !before_state || !after_state || before.operator_type_token_ == nullptr ||
      after.operator_type_token_ == nullptr) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_HANDLE_INVALID"}};
  }
  auto result = holoscan::detail::graph_lifecycle_before_resolved(
      owner_state, before_state, before.index_, after_state, after.index_, stage);
  if (!result) {
    throw RuntimeError{result.error()};
  }
}

AuthoringRegistry::AuthoringRegistry() : impl_(std::make_unique<Impl>()) {
  const auto add_payload = [this](RegisteredPayloadType row) {
    auto result = register_payload(std::move(row));
    if (!result) {
      impl_->registration_errors.push_back(std::move(result.error()));
    }
  };
  add_payload(detail::make_registered_payload<bool>("holoscan.boolean"));
  add_payload(detail::make_registered_payload<std::uint8_t>("holoscan.uint8"));
  add_payload(detail::make_registered_payload<std::int8_t>("holoscan.int8"));
  add_payload(detail::make_registered_payload<std::uint16_t>("holoscan.uint16"));
  add_payload(detail::make_registered_payload<std::int16_t>("holoscan.int16"));
  add_payload(detail::make_registered_payload<std::uint32_t>("holoscan.uint32"));
  add_payload(detail::make_registered_payload<std::int32_t>("holoscan.int32"));
  add_payload(detail::make_registered_payload<std::uint64_t>("holoscan.uint64"));
  add_payload(detail::make_registered_payload<std::int64_t>("holoscan.int64"));
  add_payload(detail::make_registered_payload<float>("holoscan.float32"));
  add_payload(detail::make_registered_payload<double>("holoscan.float64"));
  add_payload(detail::make_registered_payload<Tensor>("holoscan.Tensor"));

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

expected<void, Error> AuthoringRegistry::register_operator(RegisteredOperatorType row) {
  if (row.name.empty() || row.param_type == nullptr || row.add_to_graph == nullptr ||
      row.set_params == nullptr || row.port == nullptr || row.region_add == nullptr ||
      row.partition_add == nullptr || row.lifecycle_before == nullptr) {
    return make_unexpected(invalid_row("operator"));
  }
  auto entry = std::make_unique<Impl::OperatorEntry>(std::move(row));
  const std::lock_guard lock(impl_->mutex);
  if (impl_->operators.contains(entry->name)) {
    return make_unexpected(duplicate_row("operator", entry->name));
  }
  impl_->operators.emplace(entry->name, std::move(entry));
  return {};
}

expected<void, Error> AuthoringRegistry::register_payload(RegisteredPayloadType row) {
  if (row.name.empty() || row.add_flow == nullptr || row.export_input == nullptr ||
      row.export_output == nullptr || row.import_topic == nullptr || row.export_topic == nullptr ||
      row.make_interface_ports == nullptr) {
    return make_unexpected(invalid_row("payload"));
  }
  if (has_local_enum_identity(row.identity)) {
    return make_unexpected(local_enum_error(ErrorCode::kInvalidArgument,
                                            "AUTHORING_REGISTRY_ROW_INVALID kind=payload"));
  }
  auto entry = std::make_unique<Impl::PayloadEntry>(std::move(row));
  const std::lock_guard lock(impl_->mutex);
  if (impl_->payloads.contains(entry->name)) {
    return make_unexpected(duplicate_row("payload", entry->name));
  }
  const auto duplicate_identity =
      std::find_if(impl_->payloads.begin(), impl_->payloads.end(), [&](const auto& existing) {
        return payload_identities_equal(existing.second->row.identity, entry->row.identity);
      });
  if (duplicate_identity != impl_->payloads.end()) {
    return make_unexpected(duplicate_row("payload_identity", entry->name));
  }
  impl_->payloads.emplace(entry->name, std::move(entry));
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

expected<std::reference_wrapper<const RegisteredPayloadType>, Error>
AuthoringRegistry::payload_type(std::string_view name) const {
  const std::lock_guard lock(impl_->mutex);
  const auto found = impl_->payloads.find(name);
  if (found == impl_->payloads.end()) {
    return make_unexpected(unknown_row("payload", name));
  }
  return std::cref(found->second->row);
}

expected<std::reference_wrapper<const RegisteredPayloadType>, Error>
AuthoringRegistry::payload_type(const PayloadIdentityRecord& identity) const {
  const std::lock_guard lock(impl_->mutex);
  const auto found =
      std::find_if(impl_->payloads.begin(), impl_->payloads.end(), [&](const auto& candidate) {
        return payload_identities_equal(candidate.second->row.identity, identity);
      });
  if (found == impl_->payloads.end()) {
    return make_unexpected(unknown_row("payload_identity", "<identity>"));
  }
  return std::cref(found->second->row);
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

std::vector<std::reference_wrapper<const RegisteredPayloadType>> AuthoringRegistry::payloads()
    const {
  const std::lock_guard lock(impl_->mutex);
  return enumerate_rows<Impl::PayloadEntry, RegisteredPayloadType>(impl_->payloads);
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
  };
  try {
    Graph scratch{"authoring_registry_probe"};
    const OpaqueOperatorRef op = row.add_to_graph(scratch, "operator", nullptr);
    if (!op) {
      return make_unexpected(
          Error{ErrorCode::kFailure, "AUTHORING_OPERATOR_HARVEST_REFERENCE_INVALID"});
    }
    const GraphView view = scratch.view();
    const auto vertex =
        std::find_if(view.vertices().begin(), view.vertices().end(), [](const auto& v) {
          return v.kind == VertexKind::kOperator && v.name == "operator";
        });
    if (vertex == view.vertices().end()) {
      return make_unexpected(
          Error{ErrorCode::kFailure, "AUTHORING_OPERATOR_HARVEST_VERTEX_MISSING"});
    }
    for (const PortRecord& port : view.ports()) {
      if (port.owner == vertex->id) {
        harvested.ports.push_back(RegisteredOperatorPort{
            .name = port.name,
            .direction = port.direction,
            .payload_identity = port.payload_identity,
        });
      }
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
  return description.value();
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
