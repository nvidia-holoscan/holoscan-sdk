/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>

#include <holoscan/core/operator_spec.hpp>

#include "lifecycle_context_internal.hpp"

namespace holoscan::detail {

/// Construct callback-lifetime LifecycleContext facades for runtime invocation.
class LifecycleContextRuntimeAccess {
 public:
  [[nodiscard]] static LifecycleContext make(LifecycleContextState& state) noexcept {
    return LifecycleContext{&state};
  }
};

/// Runtime-only bridge from replayed setup metadata to retained member trampolines.
///
/// The bridge never exposes a callback pointer in a public plan or artifact.
/// Callers must first verify the replayed method/stage/type metadata against
/// the immutable plan and keep the source OperatorSpec alive for every invoke.
class OperatorSpecRuntimeAccess {
 public:
  [[nodiscard]] static std::size_t lifecycle_callback_count(const OperatorSpec& spec) noexcept {
    return spec.lifecycle_callbacks_.size();
  }

  [[nodiscard]] static LifecycleStage lifecycle_stage(const OperatorSpec& spec, std::size_t index) {
    return spec.lifecycle_callbacks_.at(index).stage;
  }

  [[nodiscard]] static LifecycleMethodId lifecycle_method(const OperatorSpec& spec,
                                                          std::size_t index) {
    return spec.lifecycle_callbacks_.at(index).method;
  }

  [[nodiscard]] static LifecycleStatus invoke(const OperatorSpec& spec, std::size_t index,
                                              void* self, LifecycleContextState& state) noexcept {
    LifecycleContext context = LifecycleContextRuntimeAccess::make(state);
    return spec.lifecycle_callbacks_.at(index).binding->invoke(self, context);
  }
};

}  // namespace holoscan::detail
