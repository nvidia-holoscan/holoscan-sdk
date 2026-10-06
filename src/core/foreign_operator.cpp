/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/foreign_operator.hpp>

#include <algorithm>
#include <memory>
#include <new>
#include <string>
#include <utility>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "operator_spec_compiler_access.hpp"

namespace holoscan {

expected<authoring::OpaquePortRef, Error> ForeignOperatorSpec::input(
    const authoring::RegisteredPayloadType& payload, std::string_view name,
    std::optional<std::uint32_t> queue_depth, bool optional) noexcept {
  if (spec_ == nullptr || ports_ == nullptr || payload.bind_spec_input == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    auto port = payload.bind_spec_input(*spec_, name, queue_depth, optional);
    if (!port) {
      return make_unexpected(std::move(port.error()));
    }
    const Error& declaration_error = detail::OperatorSpecCompilerAccess::declaration_error(*spec_);
    if (declaration_error.code != ErrorCode::kSuccess) {
      Error result = declaration_error;
      result.message.append(" port_name=");
      result.message.append(name);
      return make_unexpected(std::move(result));
    }
    ports_->push_back(std::move(port).value());
    return ports_->back();
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kFailure});
  }
#endif
}

expected<authoring::OpaquePortRef, Error> ForeignOperatorSpec::output(
    const authoring::RegisteredPayloadType& payload, std::string_view name,
    std::optional<std::uint32_t> max_emits_per_compute) noexcept {
  if (spec_ == nullptr || ports_ == nullptr || payload.bind_spec_output == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    auto port = payload.bind_spec_output(*spec_, name, max_emits_per_compute);
    if (!port) {
      return make_unexpected(std::move(port.error()));
    }
    const Error& declaration_error = detail::OperatorSpecCompilerAccess::declaration_error(*spec_);
    if (declaration_error.code != ErrorCode::kSuccess) {
      Error result = declaration_error;
      result.message.append(" port_name=");
      result.message.append(name);
      return make_unexpected(std::move(result));
    }
    ports_->push_back(std::move(port).value());
    return ports_->back();
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kFailure});
  }
#endif
}

struct ForeignOperatorCallbacks::State {
  explicit State(ForeignOperatorCallbackTable value) noexcept : table(value) {}

  ~State() noexcept {
    if (table.destroy != nullptr) {
      table.destroy(table.context);
    }
  }

  ForeignOperatorCallbackTable table;
};

ForeignOperatorCallbacks::ForeignOperatorCallbacks(ForeignOperatorCallbackTable table) {
  const bool has_lifecycle_table = table.has_lifecycle_cb != nullptr ||
                                   table.lifecycle_required_cb != nullptr ||
                                   table.lifecycle_cb != nullptr;
  if (table.compute_cb == nullptr ||
      (has_lifecycle_table &&
       (table.has_lifecycle_cb == nullptr || table.lifecycle_required_cb == nullptr ||
        table.lifecycle_cb == nullptr))) {
    if (table.destroy != nullptr) {
      table.destroy(table.context);
    }
    throw RuntimeError{
        Error{ErrorCode::kInvalidArgument, "FOREIGN_OPERATOR_CALLBACK_TABLE_INVALID"}};
  }
  try {
    state_ = std::make_shared<State>(table);
  } catch (...) {
    if (table.destroy != nullptr) {
      table.destroy(table.context);
    }
    throw;
  }
}

const ForeignOperatorCallbackTable& ForeignOperatorCallbacks::table() const noexcept {
  return state_->table;
}

ForeignOperatorCore::ForeignOperatorCore(ForeignOperatorCallbacks callbacks,
                                         ForeignOperatorOptions options)
    : callbacks_(std::move(callbacks)), options_(std::move(options)) {
  if (!callbacks_.state_) {
    throw RuntimeError{
        Error{ErrorCode::kInvalidArgument, "FOREIGN_OPERATOR_CALLBACK_TABLE_INVALID"}};
  }
  if (options_.serialization_domain &&
      !valid_operator_serialization_domain(*options_.serialization_domain)) {
    throw RuntimeError{
        Error{ErrorCode::kInvalidArgument, "FOREIGN_OPERATOR_SERIALIZATION_DOMAIN_INVALID"}};
  }
}

Error ForeignOperatorCore::callback_error(ForeignOperatorResult result) const {
  if (result.code == ErrorCode::kSuccess) {
    return Error{ErrorCode::kSuccess};
  }
  if (result.message == nullptr || result.message_size == 0U) {
    return Error{result.code};
  }
  const std::size_t count = std::min(result.message_size, kForeignOperatorMessageCapacity);
  return Error{result.code, std::string(result.message, count)};
}

void ForeignOperatorCore::setup(OperatorSpec& spec) {
  ports_.clear();
  const auto& table = callbacks_.table();
  if (table.setup_cb == nullptr) {
    return;
  }
  ForeignOperatorSpec foreign_spec{spec, ports_};
  const Error error = callback_error(table.setup_cb(table.context, &foreign_spec));
  if (error.code != ErrorCode::kSuccess) {
    throw RuntimeError{error};
  }
}

Contract ForeignOperatorCore::contract() const {
  const auto& table = callbacks_.table();
  if (table.contract_cb == nullptr) {
    return {};
  }
  Contract result;
  const Error error = callback_error(table.contract_cb(table.context, &result));
  if (error.code != ErrorCode::kSuccess) {
    throw RuntimeError{error};
  }
  return result;
}

void ForeignOperatorCore::start() {
  const auto& table = callbacks_.table();
  if (table.start_cb == nullptr) {
    return;
  }
  const Error error = callback_error(table.start_cb(table.context));
  if (error.code != ErrorCode::kSuccess) {
    throw RuntimeError{error};
  }
}

void ForeignOperatorCore::stop() {
  const auto& table = callbacks_.table();
  if (table.stop_cb == nullptr) {
    return;
  }
  const Error error = callback_error(table.stop_cb(table.context));
  if (error.code != ErrorCode::kSuccess) {
    throw RuntimeError{error};
  }
}

expected<void, Error> ForeignOperatorCore::compute(ExecutionContext& context) {
  const auto& table = callbacks_.table();
  Error error = callback_error(table.compute_cb(table.context, &context));
  if (error.code != ErrorCode::kSuccess) {
    return make_unexpected(std::move(error));
  }
  return {};
}

OperatorHostMetadata ForeignOperatorCore::host_metadata() const noexcept {
  return OperatorHostMetadata{
      .kind = OperatorHostKind::kForeign,
      .serialization_domain = options_.serialization_domain
                                  ? std::string_view{*options_.serialization_domain}
                                  : std::string_view{},
  };
}

bool ForeignOperatorCore::has_foreign_lifecycle_callback(LifecycleStage stage) const noexcept {
  const auto& table = callbacks_.table();
  return is_valid_lifecycle_stage(stage) && table.has_lifecycle_cb != nullptr &&
         table.lifecycle_cb != nullptr && table.has_lifecycle_cb(table.context, stage);
}

bool ForeignOperatorCore::foreign_lifecycle_callback_required(LifecycleStage stage) const noexcept {
  const auto& table = callbacks_.table();
  return has_foreign_lifecycle_callback(stage) && table.lifecycle_required_cb != nullptr &&
         table.lifecycle_required_cb(table.context, stage);
}

LifecycleStatus ForeignOperatorCore::invoke_foreign_lifecycle_callback(
    LifecycleStage stage, LifecycleContext& context) noexcept {
  const auto& table = callbacks_.table();
  if (table.lifecycle_cb == nullptr) {
    context.record_callback_failure(
        Error{ErrorCode::kInvalidArgument, "FOREIGN_OPERATOR_LIFECYCLE_CALLBACK_INVALID"});
    return LifecycleStatus::kFatalFailure;
  }
  LifecycleStatus status = LifecycleStatus::kFatalFailure;
  const ForeignOperatorResult result = table.lifecycle_cb(table.context, stage, &context, &status);
  if (result.code != ErrorCode::kSuccess) {
    try {
      context.record_callback_failure(callback_error(result));
    } catch (...) {
      context.record_callback_failure(Error{result.code});
    }
    return LifecycleStatus::kFatalFailure;
  }
  switch (status) {
    case LifecycleStatus::kOk:
    case LifecycleStatus::kSkipped:
    case LifecycleStatus::kRetryableFailure:
    case LifecycleStatus::kFatalFailure:
      return status;
  }
  context.record_callback_failure(
      Error{ErrorCode::kInvalidArgument, "FOREIGN_OPERATOR_LIFECYCLE_STATUS_INVALID"});
  return LifecycleStatus::kFatalFailure;
}

}  // namespace holoscan
