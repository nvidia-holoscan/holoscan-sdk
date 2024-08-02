/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "io_context.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // needed for py::cast to work with std::vector types

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../gxf/entity.hpp"
#include "core.hpp"
#include "emitter_receiver_registry.hpp"
#include "emitter_receivers.hpp"
#include "gxf/std/tensor.hpp"
#include "holoscan/core/application.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/expected.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/operators/holoviz/holoviz.hpp"
#include "io_context_pydoc.hpp"
#include "operator.hpp"  // for PyOperator
#include "tensor.hpp"    // for PyTensor

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan {

class PyRegistryContext {
 public:
  PyRegistryContext() = default;

  EmitterReceiverRegistry& registry() { return registry_; }

 private:
  EmitterReceiverRegistry& registry_ = EmitterReceiverRegistry::get_instance();
};

template <>
struct codec<std::shared_ptr<GILGuardedPyObject>> {
  static expected<size_t, RuntimeError> serialize(std::shared_ptr<GILGuardedPyObject> value,
                                                  Endpoint* endpoint) {
    HOLOSCAN_LOG_TRACE("py_emit: cloudpickle serialization of Python object over a UCX connector");
    std::string serialized_string;
    {
      py::gil_scoped_acquire acquire;
      py::module_ cloudpickle;
      try {
        cloudpickle = py::module_::import("cloudpickle");
      } catch (const py::error_already_set& e) {
        std::string err_msg = e.what() +
                              "\nThe cloudpickle module is required for Python distributed apps."s
                              "\nPlease install it with `python -m pip install cloudpickle`"s;
        HOLOSCAN_LOG_ERROR(err_msg);
        return make_unexpected<RuntimeError>(RuntimeError(ErrorCode::kCodecError, err_msg));
      }
      py::bytes serialized = cloudpickle.attr("dumps")(value->obj());
      serialized_string = serialized.cast<std::string>();
    }
    // Move `serialized_string` because it will not be used again, and to prevent a copy.
    CloudPickleSerializedObject serialized_object{std::move(serialized_string)};
    return codec<CloudPickleSerializedObject>::serialize(serialized_object, endpoint);
  }

  static expected<CloudPickleSerializedObject, RuntimeError> deserialize(Endpoint* endpoint) {
    HOLOSCAN_LOG_TRACE(
        "\tdeserialize CloudPickleSerializedObject corresponding to "
        "std::shared_ptr<GILGuardedPyObject>");
    auto maybe_obj = codec<CloudPickleSerializedObject>::deserialize(endpoint);
    if (!maybe_obj) { return forward_error(maybe_obj); }
    return std::move(maybe_obj.value());
  }
};

static void register_py_object_codec() {
  auto& codec_registry = CodecRegistry::get_instance();
  codec_registry.add_codec<std::shared_ptr<GILGuardedPyObject>>(
      "std::shared_ptr<GILGuardedPyObject>"s);
}

py::object PyInputContext::py_receive(const std::string& name, const std::string& kind) {
  auto py_op = py_op_.cast<PyOperator*>();
  auto py_op_spec = py_op->py_shared_spec();

  bool should_return_tuple = false;
  bool is_receivers = false;
  for (const auto& receivers : py_op_spec->py_receivers()) {
    if (receivers.key() == name) {
      is_receivers = true;
      should_return_tuple = true;
    }
  }

  // If 'kind' is provided, override the default behavior.
  if (!kind.empty()) {
    if (kind == "single") {
      if (is_receivers) {
        HOLOSCAN_LOG_ERROR(
            "Invalid kind '{}' for receive() method, cannot be 'single' for the input port with "
            "'IOSpec.ANY_SIZE'",
            kind);
        throw std::runtime_error(fmt::format(
            "Invalid kind '{}' for receive() method, cannot be 'single' for the input port with "
            "'IOSpec.ANY_SIZE'",
            kind));
      }
      should_return_tuple = false;
    } else if (kind == "multi") {
      should_return_tuple = true;
    } else {
      HOLOSCAN_LOG_ERROR("Invalid kind '{}' for receive() method, must be 'single' or 'multi'",
                         kind);
      throw std::runtime_error(
          fmt::format("Invalid kind '{}' for receive() method, must be 'single' or 'multi'", kind));
    }
  } else {
    // If the 'queue_size' equals IOSpec.PRECEDING_COUNT (0) or 'queue_size > 1', returns a tuple.
    if (!should_return_tuple) {
      auto input_spec = py_op_spec->inputs().find(name);
      if (input_spec != py_op_spec->inputs().end()) {
        auto queue_size = input_spec->second->queue_size();
        if (queue_size == IOSpec::kPrecedingCount || queue_size > 1) { should_return_tuple = true; }
      }
    }
  }

  if (should_return_tuple) {
    auto maybe_any_result = receive<std::vector<std::any>>(name.c_str());
    if (!maybe_any_result.has_value()) {
      HOLOSCAN_LOG_ERROR("Unable to receive input (std::vector<std::any>) with name '{}'", name);
      return py::none();
    }
    auto any_result = maybe_any_result.value();
    if (any_result.empty()) { return py::make_tuple(); }

    // Check element type (querying the first element using the name '{name}:0')
    auto& element = any_result[0];
    auto& element_type = element.type();
    auto& registry = holoscan::EmitterReceiverRegistry::get_instance();
    const auto& receiver_func = registry.get_receiver(element_type);

    py::tuple result_tuple(any_result.size());
    int counter = 0;
    try {
      for (auto& any_item : any_result) {
        if (any_item.type() == typeid(nullptr_t)) {
          // add None to the tuple
          PyTuple_SET_ITEM(result_tuple.ptr(), counter++, py::none().release().ptr());
          continue;
        }
        // Get the Python object from the entity
        py::object in_obj = receiver_func(any_item, name, *this);

        // Move the Python object into the tuple
        PyTuple_SET_ITEM(result_tuple.ptr(), counter++, in_obj.release().ptr());
      }
    } catch (const std::bad_any_cast& e) {
      HOLOSCAN_LOG_ERROR(
          "Unable to receive input (std::vector<holoscan::gxf::Entity>) with name "
          "'{}' ({})",
          name,
          e.what());
    }
    return result_tuple;
  } else {
    auto maybe_result = receive<std::any>(name.c_str());
    if (!maybe_result.has_value()) {
      HOLOSCAN_LOG_DEBUG("Unable to receive input (std::any) with name '{}'", name);
      return py::none();
    }
    auto result = maybe_result.value();
    auto& result_type = result.type();
    auto& registry = holoscan::EmitterReceiverRegistry::get_instance();
    const auto& receiver_func = registry.get_receiver(result_type);
    return receiver_func(result, name, *this);
  }
}

void PyOutputContext::py_emit(py::object& data, const std::string& name,
                              const std::string& emitter_name, int64_t acq_timestamp) {
  // Note:: Issue 4206197
  // In the UcxTransmitter::sync_io_abi(), while popping an entity from the queue,
  // Runtime::GxfEntityRefCountDec() on the entity can be called (which locks 'ref_count_mutex_').
  // If the entity held a GXF tensor, the deletor of the GXF tensor's memory buffer is called
  // when the entity is destroyed. In Python, the memory buffer is a PyDLManagedMemoryBuffer and
  // the deleter (PyDLManagedMemoryBuffer::~PyDLManagedMemoryBuffer()) will try to acquire the GIL.
  // This can cause a deadlock if the GIL is already held by another thread and the thread calls
  // GXF entity's ref-count-related functions (which locks 'ref_count_mutex_').
  // For this reason, we need to release the GIL before entity ref-count-related functions are
  // called.

// avoid overhead of retrieving operator name for release builds
#ifdef NDEBUG

#else
  auto op_name = py_op_.attr("name").cast<std::string>();
  HOLOSCAN_LOG_DEBUG("py_emit (operator name={}, port name={}):", op_name, name);
#endif

  auto& registry = holoscan::EmitterReceiverRegistry::get_instance();
  if (!emitter_name.empty()) {
    HOLOSCAN_LOG_DEBUG("py_emit: emitting a {}", emitter_name);
    const auto& emit_func = registry.get_emitter(emitter_name);
    emit_func(data, name, *this, acq_timestamp);
    return;
  }

  // If this is a PyEntity emit a gxf::Entity so that it can be consumed by non-Python operator.
  if (py::isinstance<holoscan::PyEntity>(data)) {
    HOLOSCAN_LOG_DEBUG("py_emit: emitting a holoscan::PyEntity");
    const auto& emit_func = registry.get_emitter(typeid(holoscan::PyEntity));
    emit_func(data, name, *this, acq_timestamp);
    return;
  }

  /// @todo Workaround for HolovizOp which expects a list of input specs.
  /// If we don't do the cast here the operator receives a python list object. There should be a
  /// generic way for this, or the operator needs to register expected types.
  if (py::isinstance<py::list>(data) || py::isinstance<py::tuple>(data)) {
    if (py::len(data) > 0) {
      auto seq = data.cast<py::sequence>();
      if (py::isinstance<holoscan::ops::HolovizOp::InputSpec>(seq[0])) {
        HOLOSCAN_LOG_DEBUG(
            "py_emit: emitting a std::vector<holoscan::ops::HolovizOp::InputSpec> object");
        const auto& emit_func =
            registry.get_emitter(typeid(std::vector<holoscan::ops::HolovizOp::InputSpec>));
        emit_func(data, name, *this, acq_timestamp);
        return;
      }
    }
  }

  // handle pybind11::dict separately from other Python types for special TensorMap treatment
  if (py::isinstance<py::dict>(data)) {
    const auto& emit_func = registry.get_emitter(typeid(pybind11::dict));
    emit_func(data, name, *this, acq_timestamp);
    return;
  }

  bool is_ucx_connector = false;
  if (outputs_.find(name) != outputs_.end()) {
    auto connector_type = outputs_.at(name)->connector_type();
    is_ucx_connector = connector_type == IOSpec::ConnectorType::kUCX;
  }

  bool is_distributed_app = false;
  if (is_ucx_connector) {
    is_distributed_app = true;
  } else {
    // If this operator doesn't have a UCX connector, can still determine if the app is
    // a multi-fragment app via the application pointer assigned to the fragment.
    auto py_op = py_op_.cast<PyOperator*>();
    auto py_op_spec = py_op->py_shared_spec();
    auto app_ptr = py_op_spec->fragment()->application();
    if (app_ptr) {
      // a non-empty fragment graph means that the application is multi-fragment
      if (!(app_ptr->fragment_graph().is_empty())) { is_distributed_app = true; }
    }
  }
  HOLOSCAN_LOG_DEBUG("py_emit: detected {}distributed app", is_distributed_app ? "" : "non-");

  // Note: issue 4290043
  // For distributed applications, always convert tensor-like data to an entity containing a
  // holoscan::Tensor. Previously this was only done on operators where `is_ucx_connector` was
  // true, but that lead to a bug in cases where an implicit broadcast codelet was inserted at
  // run time by the GXFExecutor. To ensure the UCX transmitter downstream of the broadcast
  // will receive an entity containiner a holoscan::Tensor for any array-like object, we need to
  // always make the conversion here. This would have additional overhead of entity creation for
  // single fragment applications, where serialization of tensors is not necessary, so we guard
  // this loop in an `is_distributed_app` condition. This way single fragment applications will
  // still just directly pass the Python object.
  if (is_distributed_app && is_tensor_like(data)) {
    HOLOSCAN_LOG_DEBUG("py_emit: emitting a tensor-like object over a UCX connector");
    const auto& emit_func = registry.get_emitter(typeid(holoscan::Tensor));
    emit_func(data, name, *this, acq_timestamp);
    return;
  }

  // Emit everything else as a Python object.
  // Note: issue 4290043
  // Instead of calling cloudpickle directly here to serialize to a string, we instead register
  // a codec for type std::shared_ptr<GILGuardedPyObject> in this module, so that proper
  // serialization will occur for distributed applications even in the case where an implicit
  // broadcast codelet was inserted.
  HOLOSCAN_LOG_DEBUG("py_emit: emitting a std::shared_ptr<GILGuardedPyObject>");
  const auto& emit_func = registry.get_emitter(typeid(std::shared_ptr<GILGuardedPyObject>));
  emit_func(data, name, *this, acq_timestamp);
  return;
}

void init_io_context(py::module_& m) {
  py::class_<Message>(m, "Message", doc::Message::doc_Message);

  py::class_<InputContext, std::shared_ptr<InputContext>> input_context(
      m, "InputContext", doc::InputContext::doc_InputContext);

  input_context.def(
      "receive", [](const InputContext&, const std::string&) { return py::none(); }, "name"_a);

  py::class_<OutputContext, std::shared_ptr<OutputContext>> output_context(
      m, "OutputContext", doc::OutputContext::doc_OutputContext);

  output_context.def(
      "emit",
      [](const OutputContext&, py::object&, const std::string&) {},
      "data"_a,
      "name"_a = "");

  py::enum_<OutputContext::OutputType>(output_context, "OutputType")
      .value("SHARED_POINTER", OutputContext::OutputType::kSharedPointer)
      .value("GXF_ENTITY", OutputContext::OutputType::kGXFEntity);

  py::class_<PyInputContext, InputContext, std::shared_ptr<PyInputContext>>(
      m, "PyInputContext", R"doc(Input context class.)doc")
      .def("receive",
           &PyInputContext::py_receive,
           "name"_a,
           py::kw_only(),
           "kind"_a = "",
           doc::InputContext::doc_receive);

  py::class_<PyOutputContext, OutputContext, std::shared_ptr<PyOutputContext>>(
      m, "PyOutputContext", R"doc(Output context class.)doc")
      .def("emit",
           &PyOutputContext::py_emit,
           "data"_a,
           "name"_a,
           "emitter_name"_a = "",
           "acq_timestamp"_a = -1,
           doc::OutputContext::doc_emit);

  // register a cloudpickle-based serializer for Python objects
  register_py_object_codec();

  py::class_<EmitterReceiverRegistry>(
      m, "EmitterReceiverRegistry", doc::EmitterReceiverRegistry::doc_EmitterReceiverRegistry)
      .def("registered_types",
           &EmitterReceiverRegistry::registered_types,
           doc::EmitterReceiverRegistry::doc_registered_types);

  // See how this register_types method is called in __init__.py to register handling of these
  // types. For user-defined operators that need to add additional types, the registry can be
  // imported from holoscan.core. See the holoscan.operators.HolovizOp source for an example.
  m.def("register_types", [](EmitterReceiverRegistry& registry) {
    registry.add_emitter_receiver<nullptr_t>("nullptr_t"s, true);
    registry.add_emitter_receiver<CloudPickleSerializedObject>("CloudPickleSerializedObject"s,
                                                               true);
    registry.add_emitter_receiver<std::string>("std::string"s, true);
    registry.add_emitter_receiver<std::shared_ptr<holoscan::GILGuardedPyObject>>("PyObject"s, true);

    // receive-only case (emit occurs via holoscan::PyEntity instead)
    registry.add_emitter_receiver<holoscan::gxf::Entity>("holoscan::gxf::Entity"s, true);

    // emitter-only cases (each of these is received as holoscan::gxf::Entity)
    registry.add_emitter_receiver<holoscan::PyEntity>("holoscan::PyEntity"s, true);
    registry.add_emitter_receiver<pybind11::dict>("pybind11::dict"s, true);
    registry.add_emitter_receiver<holoscan::Tensor>("holoscan::Tensor"s, true);
  });

  m.def(
      "registry",
      []() { return EmitterReceiverRegistry::get_instance(); },
      py::return_value_policy::reference);

  auto registry = EmitterReceiverRegistry::get_instance();
  py::class_<PyRegistryContext, std::shared_ptr<PyRegistryContext>>(
      m, "PyRegistryContext", "PyRegistryContext class")
      .def(py::init<>())
      // return a reference to the C++ static registry object
      .def("registry",
           &PyRegistryContext::registry,
           "Return a reference to the static EmitterReceiverRegistry",
           py::return_value_policy::reference_internal);
}

PyInputContext::PyInputContext(ExecutionContext* execution_context, Operator* op,
                               std::unordered_map<std::string, std::shared_ptr<IOSpec>>& inputs,
                               py::object py_op)
    : gxf::GXFInputContext::GXFInputContext(execution_context, op, inputs),
      py_op_(std::move(py_op)) {}

PyOutputContext::PyOutputContext(ExecutionContext* execution_context, Operator* op,
                                 std::unordered_map<std::string, std::shared_ptr<IOSpec>>& outputs,
                                 py::object py_op)
    : gxf::GXFOutputContext::GXFOutputContext(execution_context, op, outputs),
      py_op_(std::move(py_op)) {}

}  // namespace holoscan
