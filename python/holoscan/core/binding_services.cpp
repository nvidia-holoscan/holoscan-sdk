// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "binding_services.hpp"
#include <memory>
#include "binding_services_compatibility.hpp"

#include <holoscan/core/errors.hpp>

#include "../../../src/runtime/runtime_callback_context.hpp"
#include "dl_converter.hpp"
#include "graph.hpp"
#include "native_io.hpp"
#include "payload.hpp"
#include "py_object_payload.hpp"
#include "schema_value.hpp"
#include "session_reaper.hpp"
#include "tensor_authoring.hpp"

namespace holoscan::python {
namespace {

authoring::ForeignPayloadValue object_from_python(nb::handle value) {
  return authoring::ForeignPayloadValue::from(PyObjectPayload{value});
}

nb::object object_to_python(const authoring::ForeignPayloadValue& value) {
  const auto* payload = value.get<PyObjectPayload>();
  if (payload == nullptr) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "AUTHORING_PAYLOAD_VALUE_TYPE_MISMATCH"}};
  }
  return payload->borrow();
}

Tensor tensor_from_python(nb::handle value) {
  return *as_tensor(nb::borrow<nb::object>(value));
}

nb::object schema_to_python(const authoring::ForeignSchemaValue& value, nb::handle type) {
  return materialize_python_schema(value, GilSafeObject{type});
}

class HostedRetirement final : public detail::RunSessionReaper::Retirement {
 public:
  HostedRetirement(CleanupOperations operations, nb::handle module)
      : module_(module), operations_(operations) {}
  ~HostedRetirement() noexcept override { operations_.destroy(operations_.state); }
  void request_stop() noexcept override { operations_.request_stop(operations_.state); }
  void join() override { operations_.join(operations_.state); }

 private:
  GilSafeObject module_;
  CleanupOperations operations_;
};

void* prepare_cleanup(CleanupOperations operations, nb::handle module) {
  static_cast<void>(detail::RunSessionReaper::instance());
  detail::RunSessionReaper::destroy_retired();
  return new HostedRetirement{operations, module};
}
void enqueue_cleanup(void* ticket) noexcept {
  try {
    detail::RunSessionReaper::instance().enqueue(
        std::unique_ptr<detail::RunSessionReaper::Retirement>{
            static_cast<HostedRetirement*>(ticket)});
  } catch (...) {  // NOLINT(bugprone-empty-catch)
    // Initialization was attempted before live work. Preserve the entire owner if an exceptional
    // process-local reinitialization cannot create a worker; destroying it here could self-join.
  }
}
void destroy_cleanup(void* ticket) noexcept {
  delete static_cast<HostedRetirement*>(ticket);
}

const BindingServices kServices{
    kBindingServicesAbi,
    sizeof(BindingServices),
    HOLOSCAN_PYTHON_SERVICES_VERSION,
    kBindingServicesCompatibility,
    &add_python_operator,
    &payload_scalar_from_python,
    &payload_scalar_to_python,
    &object_from_python,
    &object_to_python,
    &tensor_from_python,
    &python_schema_value,
    &schema_to_python,
    &python_sample,
    &python_sample,
    &python_sample,
    &python_sample,
    &python_connection_options,
    [](nb::handle value) { return nb::cast<const PyTensorInputSpec&>(value).value(); },
    [](nb::handle value) { return nb::cast<const PyTensorOutputSpec&>(value).value(); },
    &prepare_cleanup,
    &enqueue_cleanup,
    &destroy_cleanup,
    &holoscan::detail::runtime_callback_context_restricted,
    &read_lifecycle_policy,
    &wrap_operator_ref,
    &read_operator_ref,
    &wrap_lifecycle_control};

}  // namespace

void init_binding_services(nb::module_& module) {
  module.attr("_binding_services") = nb::capsule(&kServices, kBindingServicesCapsule);
}

}  // namespace holoscan::python
