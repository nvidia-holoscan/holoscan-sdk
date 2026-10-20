// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <holoscan/core/lifecycle_control.hpp>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <nanobind/nanobind.h>

#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/graph.hpp>

namespace holoscan::python {

struct LifecyclePolicyData {
  std::vector<std::string> paths;
  std::vector<OperatorRef> targets;
  std::vector<LifecycleStage> stages;
};
LifecyclePolicyData read_lifecycle_policy(nanobind::handle);
nanobind::object wrap_operator_ref(OperatorRef);
OperatorRef read_operator_ref(nanobind::handle);
nanobind::object wrap_lifecycle_control(LifecycleControlHandle, const std::vector<OperatorRef>&);

// Opaque preallocated retirement for a hosted owner. The Core reaper owns the ticket after
// enqueue; destroy is called on an interpreter-owned thread after join has completed.
struct CleanupOperations {
  void* state;
  void (*request_stop)(void*) noexcept;
  void (*join)(void*);
  void (*destroy)(void*) noexcept;
};

// Private matched-build bridge. Optional bindings retain the exporting module and use its
// canonical construction/conversion services; Core knows nothing about their component types.
struct BindingServices {
  std::uint32_t abi_version;
  std::size_t size;
  const char* sdk_version;
  const char* compatibility_id;
  authoring::OpaqueOperatorRef (*add_operator)(Graph&, nanobind::handle, std::string_view,
                                               const nanobind::dict&);
  ParamScalarValue (*scalar_from_python)(nanobind::handle, ScalarIdentityKind);
  nanobind::object (*scalar_to_python)(ParamScalarValue);
  authoring::ForeignPayloadValue (*object_from_python)(nanobind::handle);
  nanobind::object (*object_to_python)(const authoring::ForeignPayloadValue&);
  Tensor (*tensor_from_python)(nanobind::handle);
  authoring::ForeignSchemaValue (*schema_from_python)(nanobind::handle);
  nanobind::object (*schema_to_python)(const authoring::ForeignSchemaValue&, nanobind::handle);
  nanobind::object (*scalar_sample)(authoring::ForeignScalarSample);
  nanobind::object (*object_sample)(const authoring::ForeignObjectSample&);
  nanobind::object (*tensor_sample)(authoring::ForeignTensorSample);
  nanobind::object (*schema_sample)(const authoring::ForeignSchemaSample&, nanobind::handle);
  ConnectionOptions (*connection_options)(nanobind::handle);
  TensorInputSpec (*tensor_input_spec)(nanobind::handle);
  TensorOutputSpec (*tensor_output_spec)(nanobind::handle);
  void* (*prepare_cleanup)(CleanupOperations, nanobind::handle);
  void (*enqueue_cleanup)(void*) noexcept;
  void (*destroy_cleanup)(void*) noexcept;
  bool (*callback_context_restricted)() noexcept;
  LifecyclePolicyData (*lifecycle_policy)(nanobind::handle);
  nanobind::object (*operator_ref)(OperatorRef);
  OperatorRef (*read_operator_ref)(nanobind::handle);
  nanobind::object (*lifecycle_control)(LifecycleControlHandle, const std::vector<OperatorRef>&);
};

// Keep the numeric prefix and capsule name on one version source.
#define HOLOSCAN_PYTHON_BINDING_SERVICES_ABI 3
#define HOLOSCAN_PYTHON_STRINGIFY_IMPL(value) #value
#define HOLOSCAN_PYTHON_STRINGIFY(value) HOLOSCAN_PYTHON_STRINGIFY_IMPL(value)
inline constexpr std::uint32_t kBindingServicesAbi = HOLOSCAN_PYTHON_BINDING_SERVICES_ABI;
inline constexpr const char* kBindingServicesCapsule =
    "holoscan.python.BindingServices.v" HOLOSCAN_PYTHON_STRINGIFY(
        HOLOSCAN_PYTHON_BINDING_SERVICES_ABI);
#undef HOLOSCAN_PYTHON_STRINGIFY
#undef HOLOSCAN_PYTHON_STRINGIFY_IMPL
#undef HOLOSCAN_PYTHON_BINDING_SERVICES_ABI

void init_binding_services(nanobind::module_& module);

}  // namespace holoscan::python
