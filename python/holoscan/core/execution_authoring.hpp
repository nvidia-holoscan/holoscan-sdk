/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

#include <holoscan/core/graph.hpp>

namespace holoscan {

namespace nb = nanobind;

// Owns a C++ RegionHandle while exposing its fluent policy-authoring operations to Python.
class PyRegionHandle {
 public:
  explicit PyRegionHandle(RegionHandle handle);

  PyRegionHandle& add(const nb::args& operators);
  PyRegionHandle& constrain(RegionConstraint constraint);
  PyRegionHandle& profile(Profile profile);

 private:
  RegionHandle handle_;
};

// Owns a C++ PartitionHandle while exposing partition membership authoring to Python.
class PyPartitionHandle {
 public:
  explicit PyPartitionHandle(PartitionHandle handle);

  PyPartitionHandle& add(const nb::args& operators);
  PyPartitionHandle& include_all();

 private:
  PartitionHandle handle_;
};

void init_execution_authoring(nb::module_& module);

}  // namespace holoscan
