/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "execution_authoring.hpp"

#include <nanobind/nanobind.h>

#include <chrono>
#include <utility>

#include <holoscan/core/graph.hpp>

#include "graph.hpp"
#include "graph_pydoc.hpp"
#include "module_name.hpp"
#include "time.hpp"

namespace holoscan {

PyRegionHandle::PyRegionHandle(RegionHandle handle) : handle_(std::move(handle)) {}

PyRegionHandle& PyRegionHandle::add(const nb::args& operators) {
  python_region_add(handle_, operators);
  return *this;
}

PyRegionHandle& PyRegionHandle::constrain(RegionConstraint constraint) {
  handle_.constrain(std::move(constraint));
  return *this;
}

PyRegionHandle& PyRegionHandle::profile(Profile profile) {
  handle_.profile(profile);
  return *this;
}

PyPartitionHandle::PyPartitionHandle(PartitionHandle handle) : handle_(std::move(handle)) {}

PyPartitionHandle& PyPartitionHandle::add(const nb::args& operators) {
  python_partition_add(handle_, operators);
  return *this;
}

PyPartitionHandle& PyPartitionHandle::include_all() {
  handle_.include_all();
  return *this;
}

void init_execution_authoring(nb::module_& module) {
  auto region_constraint_kind_type = python::bind_enum_in_module<RegionConstraintKind>(
      module, "holoscan.execution", "RegionConstraintKind");
  region_constraint_kind_type.value("STATIC_TOPOLOGY", RegionConstraintKind::kStaticTopology)
      .value("BOUNDED_MEMORY", RegionConstraintKind::kBoundedMemory)
      .value("DEVICE_COMPATIBLE_OPERATORS", RegionConstraintKind::kDeviceCompatibleOperators)
      .value("LATENCY_BUDGET", RegionConstraintKind::kLatencyBudget)
      .value("CUDA_GRAPH", RegionConstraintKind::kCudaGraph);
  auto region_constraint_type = python::bind_class_in_module<RegionConstraint>(
      module, "holoscan.execution", "RegionConstraint", doc::graph::kRegionConstraint);
  region_constraint_type.def_static("static_topology", &RegionConstraint::static_topology)
      .def_static("bounded_memory", &RegionConstraint::bounded_memory)
      .def_static("device_compatible_ops", &RegionConstraint::device_compatible_ops)
      .def_static("cuda_graph", &RegionConstraint::cuda_graph)
      .def_static(
          "latency_budget",
          [](nb::handle budget) {
            return RegionConstraint::latency_budget(std::chrono::nanoseconds{
                python_duration_nanoseconds(budget,
                                            "budget",
                                            "GRAPH_REGION_LATENCY_INVALID",
                                            "GRAPH_REGION_LATENCY_INVALID")});
          },
          nb::arg("budget"))
      .def_prop_ro("kind", &RegionConstraint::kind)
      .def_prop_ro("latency_ns",
                   [](const RegionConstraint& constraint) { return constraint.latency().count(); });

  auto region_handle_type = python::bind_class_in_module<PyRegionHandle>(
      module, "holoscan.execution", "RegionHandle", doc::graph::kRegionHandle);
  region_handle_type
      .def("add",
           &PyRegionHandle::add,
           nb::rv_policy::reference_internal,
           nb::sig("def add(self, *members: holoscan.operators.OperatorHandle | "
                   "holoscan.GraphHandle) -> typing.Self"))
      .def("constrain",
           &PyRegionHandle::constrain,
           nb::arg("constraint"),
           nb::rv_policy::reference_internal)
      .def("profile",
           &PyRegionHandle::profile,
           nb::arg("profile"),
           nb::rv_policy::reference_internal);

  auto partition_handle_type = python::bind_class_in_module<PyPartitionHandle>(
      module, "holoscan.execution", "PartitionHandle", doc::graph::kPartitionHandle);
  partition_handle_type
      .def("add",
           &PyPartitionHandle::add,
           nb::rv_policy::reference_internal,
           nb::sig("def add(self, *members: holoscan.operators.OperatorHandle) -> typing.Self"))
      .def("include_all", &PyPartitionHandle::include_all, nb::rv_policy::reference_internal);
}

}  // namespace holoscan
