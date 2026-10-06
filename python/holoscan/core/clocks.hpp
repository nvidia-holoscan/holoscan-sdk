/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

#include <string>
#include <utility>

#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/graph.hpp>

namespace holoscan {

namespace nb = nanobind;

// Describes one registered native clock type without exposing its registry row to Python.
class PyClockType {
 public:
  explicit PyClockType(const authoring::RegisteredClockType* row) : row_(row) {}

  [[nodiscard]] std::string name() const;
  [[nodiscard]] std::string repr() const;

 private:
  const authoring::RegisteredClockType* row_{};
};

// Carries a validated native ClockKey through Python graph-authoring APIs.
class PyClockKey {
 public:
  explicit PyClockKey(ClockKey key) : key_(std::move(key)) {}

  [[nodiscard]] std::string name() const;
  [[nodiscard]] bool is_default() const noexcept;
  [[nodiscard]] std::string repr() const;
  [[nodiscard]] const ClockKey& value() const noexcept { return key_; }

 private:
  ClockKey key_;
};

void init_clocks(nb::module_& module);

}  // namespace holoscan
