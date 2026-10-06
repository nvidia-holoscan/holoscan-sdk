/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "lifecycle_authoring.hpp"

#include <nanobind/nanobind.h>

#include <utility>

#include <holoscan/core/graph.hpp>

#include "graph.hpp"
#include "graph_pydoc.hpp"
#include "module_name.hpp"

namespace holoscan {

PyGraphLifecycleBuilder::PyGraphLifecycleBuilder(GraphLifecycleBuilder builder)
    : builder_(std::move(builder)) {}

PyGraphLifecycleBuilder& PyGraphLifecycleBuilder::before(nb::handle before, nb::handle after,
                                                         LifecycleStage stage) {
  python_lifecycle_before(builder_, before, after, stage);
  return *this;
}

PyGraphLifecycleBuilder& PyGraphLifecycleBuilder::allow_self_restart(nb::handle target) {
  python_lifecycle_allow_self_restart(builder_, target);
  return *this;
}

PyGraphLifecycleBuilder& PyGraphLifecycleBuilder::allow_external_restart(nb::handle target) {
  python_lifecycle_allow_external_restart(builder_, target);
  return *this;
}

PyGraphLifecycleBuilder& PyGraphLifecycleBuilder::allow_restart_request(nb::handle requester,
                                                                        nb::handle target) {
  python_lifecycle_allow_restart_request(builder_, requester, target);
  return *this;
}

void init_lifecycle_authoring(nb::module_& module) {
  auto lifecycle_stage_type =
      python::bind_enum_in_module<LifecycleStage>(module, "holoscan.lifecycle", "LifecycleStage");
  lifecycle_stage_type.value("CONFIGURE", LifecycleStage::kConfigure)
      .value("DISCOVER", LifecycleStage::kDiscover)
      .value("ALLOCATE", LifecycleStage::kAllocate)
      .value("WARM_UP", LifecycleStage::kWarmUp)
      .value("ARM", LifecycleStage::kArm)
      .value("START", LifecycleStage::kStart)
      .value("STOP", LifecycleStage::kStop)
      .value("RELEASE", LifecycleStage::kRelease);

  auto lifecycle_status_type =
      python::bind_enum_in_module<LifecycleStatus>(module, "holoscan.lifecycle", "LifecycleStatus");
  lifecycle_status_type.value("OK", LifecycleStatus::kOk)
      .value("SKIPPED", LifecycleStatus::kSkipped)
      .value("RETRYABLE_FAILURE", LifecycleStatus::kRetryableFailure)
      .value("FATAL_FAILURE", LifecycleStatus::kFatalFailure);

  auto transition_phase_type =
      python::bind_enum_in_module<LifecycleTransitionPhase>(module,
                                                            "holoscan.lifecycle",
                                                            "LifecycleTransitionPhase",
                                                            doc::graph::kLifecycleTransitionPhase);
  transition_phase_type.value("NONE", LifecycleTransitionPhase::kNone)
      .value("STARTUP_FORWARD", LifecycleTransitionPhase::kStartupForward)
      .value("PRE_RESTART_TEARDOWN", LifecycleTransitionPhase::kPreRestartTeardown)
      .value("RESTART_FORWARD", LifecycleTransitionPhase::kRestartForward)
      .value("FORWARD_ROLLBACK", LifecycleTransitionPhase::kForwardRollback)
      .value("RESTORE_OLD", LifecycleTransitionPhase::kRestoreOld)
      .value("SHUTDOWN_TEARDOWN", LifecycleTransitionPhase::kShutdownTeardown);

  auto lifecycle_builder_type = python::bind_class_in_module<PyGraphLifecycleBuilder>(
      module, "holoscan.lifecycle", "GraphLifecycleBuilder", doc::graph::kGraphLifecycleBuilder);
  lifecycle_builder_type
      .def("before",
           &PyGraphLifecycleBuilder::before,
           nb::arg("before"),
           nb::arg("after"),
           nb::arg("stage"),
           nb::rv_policy::reference_internal,
           nb::sig("def before(self, before: holoscan.operators.OperatorHandle, "
                   "after: holoscan.operators.OperatorHandle, "
                   "stage: holoscan.lifecycle.LifecycleStage) -> "
                   "holoscan.lifecycle.GraphLifecycleBuilder"))
      .def("allow_self_restart",
           &PyGraphLifecycleBuilder::allow_self_restart,
           nb::arg("target"),
           nb::rv_policy::reference_internal,
           nb::sig("def allow_self_restart(self, target: holoscan.operators.OperatorHandle) -> "
                   "holoscan.lifecycle.GraphLifecycleBuilder"))
      .def("allow_external_restart",
           &PyGraphLifecycleBuilder::allow_external_restart,
           nb::arg("target"),
           nb::rv_policy::reference_internal,
           nb::sig("def allow_external_restart(self, target: holoscan.operators.OperatorHandle) -> "
                   "holoscan.lifecycle.GraphLifecycleBuilder"))
      .def("allow_restart_request",
           &PyGraphLifecycleBuilder::allow_restart_request,
           nb::arg("requester"),
           nb::arg("target"),
           nb::rv_policy::reference_internal,
           nb::sig("def allow_restart_request(self, "
                   "requester: holoscan.operators.OperatorHandle, "
                   "target: holoscan.operators.OperatorHandle) -> "
                   "holoscan.lifecycle.GraphLifecycleBuilder"));
}

}  // namespace holoscan
