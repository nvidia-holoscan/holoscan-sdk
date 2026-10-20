/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <memory>

#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator_ref.hpp>

namespace holoscan {

namespace nb = nanobind;

// Generation-stamped restart target shared by callback and run-control Python surfaces.
class PyLifecycleOperatorRef {
 public:
  explicit PyLifecycleOperatorRef(OperatorRef value) noexcept : value_(value) {}

  [[nodiscard]] OperatorRef value() const noexcept { return value_; }
  [[nodiscard]] std::uint32_t id() const noexcept { return value_.id; }
  [[nodiscard]] std::uint64_t plan_generation() const noexcept { return value_.plan_generation; }
  [[nodiscard]] std::uint64_t run_generation() const noexcept { return value_.run_generation; }

 private:
  OperatorRef value_{};
};

// Bridges Python lifecycle-policy calls to the Graph-owned C++ lifecycle builder.
class PyGraphLifecycleBuilder {
 public:
  explicit PyGraphLifecycleBuilder(GraphLifecycleBuilder builder);

  PyGraphLifecycleBuilder& before(nb::handle before, nb::handle after, LifecycleStage stage);
  PyGraphLifecycleBuilder& allow_self_restart(nb::handle target);
  PyGraphLifecycleBuilder& allow_external_restart(nb::handle target);
  PyGraphLifecycleBuilder& allow_restart_request(nb::handle requester, nb::handle target);

 private:
  GraphLifecycleBuilder builder_;
};

void init_lifecycle_authoring(nb::module_& module);

}  // namespace holoscan
