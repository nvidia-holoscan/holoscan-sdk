/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/authoring/constructor_handle.hpp>
#include <holoscan/core/graph.hpp>

namespace holoscan::authoring::detail {

/// Record one validated constructor-handle input retained by an Operator factory.
struct ConstructorBinding {
  std::string name;
  const ConstructorHandleDescriptor* type{};
  ConstructorHandle owner;
};

/// Record every handle bound to one registered Operator declaration and its origin process.
///
/// The record is immutable and shared by every clone of the declaration's factory. It retains the
/// owners independently of the factory's own constructor captures, including owners that a thunk
/// adapted into callbacks, so binding metadata and the origin check can never be dropped by a
/// thunk.
struct ConstructorBindingRecord {
  std::int64_t origin_process{};
  std::vector<ConstructorBinding> bindings;
};

/// Forward one Operator factory while enforcing its constructor-handle binding record.
///
/// Every Graph, embedding, compiler, parameter-update, and runtime path copies factories through
/// `clone()` or `clone_with_params()` and constructs through `create()`, so wrapping the
/// declaration's factory once propagates the record to every replay without changes to those paths.
class ConstructorBindingFactory final : public holoscan::detail::OperatorFactoryBase {
 public:
  ConstructorBindingFactory(std::shared_ptr<holoscan::detail::OperatorFactoryBase> inner,
                            std::shared_ptr<const ConstructorBindingRecord> record) noexcept
      : inner_(std::move(inner)), record_(std::move(record)) {}

  [[nodiscard]] expected<std::unique_ptr<OperatorBase>, Error> create(
      holoscan::detail::DiagnosticCapture* capture) const override {
    if (record_->origin_process != constructor_handle_process_id()) {
      return make_unexpected(
          Error{ErrorCode::kIncompatible, "CONSTRUCTOR_HANDLE_PROCESS_MISMATCH"});
    }
    return inner_->create(capture);
  }

  [[nodiscard]] std::shared_ptr<holoscan::detail::OperatorFactoryBase> clone() const override {
    return std::make_shared<ConstructorBindingFactory>(inner_->clone(), record_);
  }

  [[nodiscard]] expected<std::shared_ptr<holoscan::detail::OperatorFactoryBase>, Error>
  clone_with_params(const void* params, const void* type_token) const override {
    auto cloned = inner_->clone_with_params(params, type_token);
    if (!cloned) {
      return make_unexpected(std::move(cloned).error());
    }
    return std::shared_ptr<holoscan::detail::OperatorFactoryBase>(
        std::make_shared<ConstructorBindingFactory>(std::move(*cloned), record_));
  }

  [[nodiscard]] const ParamTypeDescriptor& parameter_type() const noexcept override {
    return inner_->parameter_type();
  }

  [[nodiscard]] const ParamSchemaDescriptor* parameter_schema() const noexcept override {
    return inner_->parameter_schema();
  }

  [[nodiscard]] const void* initial_parameter_value() const noexcept override {
    return inner_->initial_parameter_value();
  }

  [[nodiscard]] bool has_explicit_parameter_baseline() const noexcept override {
    return inner_->has_explicit_parameter_baseline();
  }

  [[nodiscard]] bool has_no_params() const noexcept override { return inner_->has_no_params(); }

  [[nodiscard]] LifecycleHookSet lifecycle_hooks() const noexcept override {
    return inner_->lifecycle_hooks();
  }

  /// Return the shared binding record for inspection by tests and diagnostics.
  [[nodiscard]] const ConstructorBindingRecord& record() const noexcept { return *record_; }

 private:
  std::shared_ptr<holoscan::detail::OperatorFactoryBase> inner_;
  std::shared_ptr<const ConstructorBindingRecord> record_;
};

}  // namespace holoscan::authoring::detail
