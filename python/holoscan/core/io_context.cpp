/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "holoscan/core/gxf/codec_registry.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/operators/holoviz/holoviz.hpp"
#include "holoscan/operators/inference/inference.hpp"
#include "holoscan/profiler/profiler.hpp"
#include "io_context_pydoc.hpp"
#include "operator.hpp"  // for PyOperator
#include "tensor.hpp"    // for PyTensor

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

class PyRegistryContext {
 public:
  PyRegistryContext() = default;

  EmitterReceiverRegistry& registry() { return registry_; }

 private:
  EmitterReceiverRegistry& registry_ = EmitterReceiverRegistry::get_instance();
};

// NOLINTBEGIN(altera-struct-pack-align)
template <>
struct codec<std::shared_ptr<GILGuardedPyObject>> {
  static expected<size_t, RuntimeError> serialize(const std::shared_ptr<GILGuardedPyObject>& value,
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
    if (!maybe_obj) {
      return forward_error(maybe_obj);
    }
    return std::move(maybe_obj.value());
  }
};
// NOLINTEND(altera-struct-pack-align)

static void register_py_object_codec() {
  auto& codec_registry = gxf::CodecRegistry::get_instance();
  codec_registry.add_codec<std::shared_ptr<GILGuardedPyObject>>(
      "std::shared_ptr<GILGuardedPyObject>"s);
}

namespace {

/**
 * @brief Determine if tuple of objects will be received based on kind or the OperatorSpec
 *
 * @param name The name of the input
 * @param kind The kind of the input
 * @param py_op_spec The OperatorSpec object
 *
 * @return True if the input should be returned as a tuple, false otherwise
 */
bool should_return_as_tuple(const std::string& name, const std::string& kind,
                            const std::shared_ptr<PyOperatorSpec>& py_op_spec) {
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
        std::string err_msg = fmt::format(
            "Invalid kind '{}' for receive() method, cannot be 'single' for the input port with "
            "'IOSpec.ANY_SIZE'",
            kind);
        HOLOSCAN_LOG_ERROR(err_msg);
        throw std::runtime_error(err_msg);
      }
      should_return_tuple = false;
    } else if (kind == "multi") {
      should_return_tuple = true;
    } else {
      std::string err_msg =
          fmt::format("Invalid kind '{}' for receive() method, must be 'single' or 'multi'", kind);
      HOLOSCAN_LOG_ERROR(err_msg);
      throw std::runtime_error(err_msg);
    }
  } else {
    if (!should_return_tuple) {
      // If the 'queue_size' equals IOSpec.PRECEDING_COUNT (0) or 'queue_size > 1', returns a tuple.
      auto input_spec = py_op_spec->inputs().find(name);
      if (input_spec != py_op_spec->inputs().end()) {
        auto queue_size = input_spec->second->queue_size();
        if (queue_size == IOSpec::kPrecedingCount || queue_size > 1) {
          should_return_tuple = true;
        }
      }
    }
  }
  return should_return_tuple;
}
}  // namespace

py::object PyInputContext::receive_as_tuple(const std::string& name) {
  auto maybe_any_result = receive<std::vector<std::any>>(name.c_str());
  if (!maybe_any_result.has_value()) {
    HOLOSCAN_LOG_ERROR("Unable to receive input (std::vector<std::any>) with name '{}'", name);
    return py::none();
  }
  auto any_result = maybe_any_result.value();
  if (any_result.empty()) {
    return py::make_tuple();
  }

  // Get receiver from registry based only on the type of the first element
  auto& registry = holoscan::EmitterReceiverRegistry::get_instance();
  const auto& receiver_func = registry.get_receiver(any_result[0].type());

  py::tuple result_tuple(any_result.size());
  int counter = 0;
  try {
    for (const auto& any_item : any_result) {
      const auto& item_type = any_item.type();
      if (item_type == typeid(kNoReceivedMessage) || item_type == typeid(std::nullptr_t)) {
        PyTuple_SET_ITEM(result_tuple.ptr(), counter++, py::none().release().ptr());
        continue;
      }
      py::object in_obj = receiver_func(any_item, name, *this);
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
}

py::object PyInputContext::receive_as_single(const std::string& name) {
  auto maybe_result = receive<std::any>(name.c_str());
  if (!maybe_result.has_value()) {
    HOLOSCAN_LOG_DEBUG("Unable to receive input (std::any) with name '{}'", name);
    return py::none();
  }
  auto result = maybe_result.value();
  auto& registry = holoscan::EmitterReceiverRegistry::get_instance();
  const auto& receiver_func = registry.get_receiver(result.type());
  return receiver_func(result, name, *this);
}

py::object PyInputContext::py_receive(const std::string& name, const std::string& kind) {
  PROF_SCOPED_EVENT(py_op_->id(), event_py_receive);

  auto py_op_spec = py_op_->py_shared_spec();

  bool should_return_tuple = should_return_as_tuple(name, kind, py_op_spec);
  if (should_return_tuple) {
    return receive_as_tuple(name);
  }
  return receive_as_single(name);
}

bool PyOutputContext::handle_py_entity(py::object& data, const std::string& name,
                                       int64_t acq_timestamp, EmitterReceiverRegistry& registry) {
  if (py::isinstance<holoscan::PyEntity>(data)) {
    HOLOSCAN_LOG_DEBUG("py_emit: emitting a holoscan::PyEntity");
    const auto& emit_func = registry.get_emitter(typeid(holoscan::PyEntity));
    emit_func(data, name, *this, acq_timestamp);
    return true;
  }
  return false;
}

bool PyOutputContext::handle_py_dict(py::object& data, const std::string& name,
                                     int64_t acq_timestamp, EmitterReceiverRegistry& registry) {
  if (py::isinstance<py::dict>(data)) {
    const auto& emit_func = registry.get_emitter(typeid(pybind11::dict));
    emit_func(data, name, *this, acq_timestamp);
    return true;
  }
  return false;
}

bool PyOutputContext::handle_holoviz_op(py::object& data, const std::string& name,
                                        int64_t acq_timestamp, EmitterReceiverRegistry& registry) {
  // Emit a sequence of HolovizOp.InputSpec as a C++ object without having to explicitly set
  // emitter_name="std::vector<holoscan::ops::HolovizOp::InputSpec>" when calling emit.
  if ((py::isinstance<py::list>(data) || py::isinstance<py::tuple>(data)) && py::len(data) > 0) {
    auto seq = data.cast<py::sequence>();
    if (py::isinstance<holoscan::ops::HolovizOp::InputSpec>(seq[0])) {
      HOLOSCAN_LOG_DEBUG(
          "py_emit: emitting a std::vector<holoscan::ops::HolovizOp::InputSpec> object");
      const auto& emit_func =
          registry.get_emitter(typeid(std::vector<holoscan::ops::HolovizOp::InputSpec>));
      emit_func(data, name, *this, acq_timestamp);
      return true;
    }
  }
  return false;
}

bool PyOutputContext::handle_inference_op(py::object& data, const std::string& name,
                                          int64_t acq_timestamp,
                                          EmitterReceiverRegistry& registry) {
  // Emit a sequence of InferenceOp.ActivationSpec as a C++ object without having to explicitly set
  // emitter_name="std::vector<holoscan::ops::InferenceOp::ActivationSpec>" when calling emit.
  if ((py::isinstance<py::list>(data) || py::isinstance<py::tuple>(data)) && py::len(data) > 0) {
    auto seq = data.cast<py::sequence>();
    if (py::isinstance<holoscan::ops::InferenceOp::ActivationSpec>(seq[0])) {
      HOLOSCAN_LOG_DEBUG(
          "py_emit: emitting a std::vector<holoscan::ops::InferenceOp::ActivationSpec> object");
      const auto& emit_func =
          registry.get_emitter(typeid(std::vector<holoscan::ops::InferenceOp::ActivationSpec>));
      emit_func(data, name, *this, acq_timestamp);
      return true;
    }
  }
  return false;
}

bool PyOutputContext::check_distributed_app(const std::string& name) {
  bool is_ucx_connector = false;
  if (outputs_.find(name) != outputs_.end()) {
    auto connector_type = outputs_.at(name)->connector_type();
    is_ucx_connector = connector_type == IOSpec::ConnectorType::kUCX;
  }

  if (is_ucx_connector) {
    return true;
  }

  // If this operator doesn't have a UCX connector, can still determine if the app is
  // a multi-fragment app via the application pointer assigned to the fragment
  auto py_op_spec = py_op_->py_shared_spec();
  auto* app_ptr = py_op_spec->fragment()->application();
  if (app_ptr != nullptr) {
    // a non-empty fragment graph means that the application is multi-fragment
    if (!(app_ptr->fragment_graph().is_empty())) {
      return true;
    }
  }
  return false;
}

void PyOutputContext::emit_tensor_like_distributed(py::object& data, const std::string& name,
                                                   int64_t acq_timestamp,
                                                   EmitterReceiverRegistry& registry) {
  HOLOSCAN_LOG_DEBUG("py_emit: emitting a tensor-like object over a UCX connector");
  const auto& emit_func = registry.get_emitter(typeid(holoscan::Tensor));
  emit_func(data, name, *this, acq_timestamp);
}

void PyOutputContext::emit_python_object(py::object& data, const std::string& name,
                                         int64_t acq_timestamp, EmitterReceiverRegistry& registry) {
  // Note: issue 4290043
  // Instead of calling cloudpickle directly here to serialize to a string, we instead register
  // a codec for type std::shared_ptr<GILGuardedPyObject> in this module, so that proper
  // serialization will occur for distributed applications even in the case where an implicit
  // broadcast codelet was inserted.
  HOLOSCAN_LOG_DEBUG("py_emit: emitting a std::shared_ptr<GILGuardedPyObject>");
  const auto& emit_func = registry.get_emitter(typeid(std::shared_ptr<GILGuardedPyObject>));
  emit_func(data, name, *this, acq_timestamp);
}

void PyOutputContext::py_emit(py::object& data, const std::string& name,
                              const std::string& emitter_name, int64_t acq_timestamp) {
  PROF_SCOPED_EVENT(py_op_->id(), event_py_emit);

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
  if (py_op_) {
    auto op_name = py_op_->name();
    HOLOSCAN_LOG_DEBUG("py_emit (operator name={}, port name={}):", op_name, name);
  } else {
    HOLOSCAN_LOG_ERROR("PyOutputContext: py_op_ is not set");
    throw std::runtime_error("PyOutputContext: py_op_ is not set");
  }
#endif

  auto& registry = holoscan::EmitterReceiverRegistry::get_instance();

  // If the user specified emitter_name, emit using that
  if (!emitter_name.empty()) {
    HOLOSCAN_LOG_DEBUG("py_emit: emitting a {}", emitter_name);
    const auto& emit_func = registry.get_emitter(emitter_name);
    emit_func(data, name, *this, acq_timestamp);
    return;
  }

  // If this is a PyEntity emit a gxf::Entity so that it can be consumed by non-Python operator.
  if (handle_py_entity(data, name, acq_timestamp, registry)) {
    return;
  }

  /// @todo Workaround for HolovizOp which expects a list of input specs.
  /// If we don't do the cast here the operator receives a python list object. There should be a
  /// generic way for this, or the operator needs to register expected types.
  if (handle_holoviz_op(data, name, acq_timestamp, registry)) {
    return;
  }
  if (handle_inference_op(data, name, acq_timestamp, registry)) {
    return;
  }

  // handle pybind11::dict separately from other Python types for special TensorMap treatment
  if (handle_py_dict(data, name, acq_timestamp, registry)) {
    return;
  }

  bool is_distributed_app = check_distributed_app(name);
  HOLOSCAN_LOG_DEBUG("py_emit: detected {}distributed app", is_distributed_app ? "" : "non-");
  if (is_distributed_app && is_tensor_like(data)) {
    // Note: issue 4290043
    // For distributed applications, always convert tensor-like data to an entity containing a
    // holoscan::Tensor. Previously this was only done on operators where `is_ucx_connector` was
    // true, but that lead to a bug in cases where an implicit broadcast codelet was inserted at
    // run time by the GXFExecutor. To ensure the UCX transmitter downstream of the broadcast
    // will receive an entity containing a holoscan::Tensor for any array-like object, we need to
    // always make the conversion here. This would have additional overhead of entity creation for
    // single fragment applications, where serialization of tensors is not necessary, so we guard
    // this loop in an `is_distributed_app` condition. This way single fragment applications will
    // still just directly pass the Python object.
    emit_tensor_like_distributed(data, name, acq_timestamp, registry);
    return;
  }

  // Emit everything else as a Python object.
  emit_python_object(data, name, acq_timestamp, registry);
}

void init_io_context(py::module_& m) {
  // NOLINTNEXTLINE(bugprone-unused-raii)
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

  py::class_<PyInputContext, InputContext, std::shared_ptr<PyInputContext>>(
      m, "PyInputContext", R"doc(Input context class.)doc")
      .def("receive",
           &PyInputContext::py_receive,
           "name"_a,
           py::kw_only(),
           "kind"_a = "",
           doc::InputContext::doc_receive)
      .def(
          "receive_cuda_stream",
          [](PyInputContext& op_input, const char* input_port_name, bool allocate) -> intptr_t {
            auto cuda_stream = op_input.receive_cuda_stream(input_port_name, allocate);
            auto stream_ptr_address = reinterpret_cast<intptr_t>(static_cast<void*>(cuda_stream));
            return stream_ptr_address;
          },
          "input_port_name"_a = nullptr,
          "allocate"_a = true,
          doc::InputContext::doc_receive_cuda_stream)
      .def(
          "receive_cuda_streams",
          [](PyInputContext& op_input,
             const char* input_port_name) -> std::vector<std::optional<intptr_t>> {
            auto cuda_streams = op_input.receive_cuda_streams(input_port_name);
            std::vector<std::optional<intptr_t>> out_ptrs{};
            out_ptrs.reserve(cuda_streams.size());
            for (const auto& stream : cuda_streams) {
              if (stream) {
                out_ptrs.emplace_back(
                    reinterpret_cast<intptr_t>(static_cast<void*>(stream.value())));
              } else {
                out_ptrs.emplace_back(std::nullopt);
              }
            }
            return out_ptrs;
          },
          "input_port_name"_a = nullptr,
          doc::InputContext::doc_receive_cuda_streams);

  py::class_<PyOutputContext, OutputContext, std::shared_ptr<PyOutputContext>>(
      m, "PyOutputContext", R"doc(Output context class.)doc")
      .def("emit",
           &PyOutputContext::py_emit,
           "data"_a,
           "name"_a,
           "emitter_name"_a = "",
           "acq_timestamp"_a = -1,
           doc::OutputContext::doc_emit)
      .def(
          "set_cuda_stream",
          [](PyOutputContext& op_output,
             intptr_t stream_ptr,
             const char* output_port_name = nullptr) {
            auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
            op_output.set_cuda_stream(cuda_stream, output_port_name);
            return;
          },
          "stream_ptr"_a,
          "output_port_name"_a = nullptr,
          doc::OutputContext::doc_set_cuda_stream);

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
    registry.add_emitter_receiver<std::nullptr_t>("nullptr_t"s, true);  // (deprecated name)
    registry.add_emitter_receiver<std::nullptr_t>("std::nullptr_t"s, true);

    registry.add_emitter_receiver<CloudPickleSerializedObject>("CloudPickleSerializedObject"s,
                                                               true);
    // receive-only case (emit occurs via holoscan::PyEntity instead)
    registry.add_emitter_receiver<holoscan::gxf::Entity>("holoscan::gxf::Entity"s, true);

    // emitter-only cases (each of these is received as holoscan::gxf::Entity)
    registry.add_emitter_receiver<holoscan::PyEntity>("holoscan::PyEntity"s, true);
    registry.add_emitter_receiver<pybind11::dict>("pybind11::dict"s, true);
    // Note: The holoscan::Tensor emitter actually emits a TensorMap containing a single tensor
    registry.add_emitter_receiver<holoscan::Tensor>("holoscan::Tensor"s, true);

    // Python objects
    registry.add_emitter_receiver<std::shared_ptr<holoscan::GILGuardedPyObject>>("PyObject"s, true);

    // Note: The types below are relevant to transmit between wrapped C++ operators and native
    // Python operators for both within fragment and inter-fragment communication. In the case of
    // inter-fragment communication (distributed apps), corresponding types must also be registered
    // with the GXF UCX serialization components in `core/gxf/codec_registry.hpp`.

    // Unordered map types
    registry.add_emitter_receiver<std::unordered_map<std::string, std::string>>(
        "std::unordered_map<std::string, std::string>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::unordered_map<std::string, std::string>>>(
        "std::shared_ptr<std::unordered_map<std::string, std::string>>"s, true);

    // register various basic numeric and string types
    registry.add_emitter_receiver<std::string>("std::string"s, true);
    registry.add_emitter_receiver<float>("float"s, true);
    registry.add_emitter_receiver<double>("double"s, true);
    registry.add_emitter_receiver<std::complex<float>>("std::complex<float>"s, true);
    registry.add_emitter_receiver<std::complex<double>>("std::complex<double>"s, true);
    registry.add_emitter_receiver<bool>("bool"s, true);
    registry.add_emitter_receiver<int8_t>("int8_t"s, true);
    registry.add_emitter_receiver<uint8_t>("uint8_t"s, true);
    registry.add_emitter_receiver<int16_t>("int16_t"s, true);
    registry.add_emitter_receiver<uint16_t>("uint16_t"s, true);
    registry.add_emitter_receiver<int32_t>("int32_t"s, true);
    registry.add_emitter_receiver<uint32_t>("uint32_t"s, true);
    registry.add_emitter_receiver<int64_t>("int64_t"s, true);
    registry.add_emitter_receiver<uint64_t>("uint64_t"s, true);

    // register shared_ptr ofvarious basic numeric and string types
    registry.add_emitter_receiver<std::shared_ptr<std::string>>("std::shared_ptr<std::string>"s,
                                                                true);
    registry.add_emitter_receiver<std::shared_ptr<float>>("std::shared_ptr<float>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<double>>("std::shared_ptr<double>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::complex<float>>>(
        "std::shared_ptr<std::complex<float>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::complex<double>>>(
        "std::shared_ptr<std::complex<double>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<bool>>("std::shared_ptr<bool>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<int8_t>>("std::shared_ptr<int8_t>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<uint8_t>>("std::shared_ptr<uint8_t>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<int16_t>>("std::shared_ptr<int16_t>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<uint16_t>>("std::shared_ptr<uint16_t>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<int32_t>>("std::shared_ptr<int32_t>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<uint32_t>>("std::shared_ptr<uint32_t>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<int64_t>>("std::shared_ptr<int64_t>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<uint64_t>>("std::shared_ptr<uint64_t>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::complex<float>>>(
        "std::shared_ptr<std::complex<float>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::complex<double>>>(
        "std::shared_ptr<std::complex<double>>"s, true);

    // register vector of numeric and string types
    registry.add_emitter_receiver<std::vector<std::string>>("std::vector<std::string>"s, true);
    registry.add_emitter_receiver<std::vector<float>>("std::vector<float>"s, true);
    registry.add_emitter_receiver<std::vector<double>>("std::vector<double>"s, true);
    registry.add_emitter_receiver<std::vector<std::complex<float>>>(
        "std::vector<std::complex<float>>"s, true);
    registry.add_emitter_receiver<std::vector<std::complex<double>>>(
        "std::vector<std::complex<double>>"s, true);
    registry.add_emitter_receiver<std::vector<bool>>("std::vector<bool>"s, true);
    registry.add_emitter_receiver<std::vector<int8_t>>("std::vector<int8_t>"s, true);
    registry.add_emitter_receiver<std::vector<uint8_t>>("std::vector<uint8_t>"s, true);
    registry.add_emitter_receiver<std::vector<int16_t>>("std::vector<int16_t>"s, true);
    registry.add_emitter_receiver<std::vector<uint16_t>>("std::vector<uint16_t>"s, true);
    registry.add_emitter_receiver<std::vector<int32_t>>("std::vector<int32_t>"s, true);
    registry.add_emitter_receiver<std::vector<uint32_t>>("std::vector<uint32_t>"s, true);
    registry.add_emitter_receiver<std::vector<int64_t>>("std::vector<int64_t>"s, true);
    registry.add_emitter_receiver<std::vector<uint64_t>>("std::vector<uint64_t>"s, true);

    // shared_ptr<vector<T>> types
    registry.add_emitter_receiver<std::shared_ptr<std::vector<std::string>>>(
        "std::shared_ptr<std::vector<std::string>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::vector<float>>>(
        "std::shared_ptr<std::vector<float>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::vector<double>>>(
        "std::shared_ptr<std::vector<double>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::vector<std::complex<float>>>>(
        "std::shared_ptr<std::vector<std::complex<float>>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::vector<std::complex<double>>>>(
        "std::shared_ptr<std::vector<std::complex<double>>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::vector<bool>>>(
        "std::shared_ptr<std::vector<bool>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::vector<int8_t>>>(
        "std::shared_ptr<std::vector<int8_t>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::vector<uint8_t>>>(
        "std::shared_ptr<std::vector<uint8_t>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::vector<int16_t>>>(
        "std::shared_ptr<std::vector<int16_t>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::vector<uint16_t>>>(
        "std::shared_ptr<std::vector<uint16_t>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::vector<int32_t>>>(
        "std::shared_ptr<std::vector<int32_t>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::vector<uint32_t>>>(
        "std::shared_ptr<std::vector<uint32_t>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::vector<int64_t>>>(
        "std::shared_ptr<std::vector<int64_t>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::vector<uint64_t>>>(
        "std::shared_ptr<std::vector<uint64_t>>"s, true);

    // vector<vector<T>> types
    registry.add_emitter_receiver<std::vector<std::vector<std::string>>>(
        "std::vector<std::vector<std::string>>"s, true);
    registry.add_emitter_receiver<std::vector<std::vector<float>>>(
        "std::vector<std::vector<float>>"s, true);
    registry.add_emitter_receiver<std::vector<std::vector<double>>>(
        "std::vector<std::vector<double>>"s, true);
    registry.add_emitter_receiver<std::vector<std::vector<std::complex<float>>>>(
        "std::vector<std::vector<std::complex<float>>>"s, true);
    registry.add_emitter_receiver<std::vector<std::vector<std::complex<double>>>>(
        "std::vector<std::vector<std::complex<double>>>"s, true);
    registry.add_emitter_receiver<std::vector<std::vector<bool>>>("std::vector<std::vector<bool>>"s,
                                                                  true);
    registry.add_emitter_receiver<std::vector<std::vector<int8_t>>>(
        "std::vector<std::vector<int8_t>>"s, true);
    registry.add_emitter_receiver<std::vector<std::vector<uint8_t>>>(
        "std::vector<std::vector<uint8_t>>"s, true);
    registry.add_emitter_receiver<std::vector<std::vector<int16_t>>>(
        "std::vector<std::vector<int16_t>>"s, true);
    registry.add_emitter_receiver<std::vector<std::vector<uint16_t>>>(
        "std::vector<std::vector<uint16_t>>"s, true);
    registry.add_emitter_receiver<std::vector<std::vector<int32_t>>>(
        "std::vector<std::vector<int32_t>>"s, true);
    registry.add_emitter_receiver<std::vector<std::vector<uint32_t>>>(
        "std::vector<std::vector<uint32_t>>"s, true);
    registry.add_emitter_receiver<std::vector<std::vector<int64_t>>>(
        "std::vector<std::vector<int64_t>>"s, true);
    registry.add_emitter_receiver<std::vector<std::vector<uint64_t>>>(
        "std::vector<std::vector<uint64_t>>"s, true);

    // shared_ptr<vector<vector<T>>> types
    registry.add_emitter_receiver<std::shared_ptr<std::vector<std::vector<std::string>>>>(
        "std::shared_ptr<std::vector<std::vector<std::string>>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::vector<std::vector<float>>>>(
        "std::shared_ptr<std::vector<std::vector<float>>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::vector<std::vector<double>>>>(
        "std::shared_ptr<std::vector<std::vector<double>>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::vector<std::vector<std::complex<float>>>>>(
        "std::shared_ptr<std::vector<std::vector<std::complex<float>>>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::vector<std::vector<std::complex<double>>>>>(
        "std::shared_ptr<std::vector<std::vector<std::complex<double>>>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::vector<std::vector<bool>>>>(
        "std::shared_ptr<std::vector<std::vector<bool>>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::vector<std::vector<int8_t>>>>(
        "std::shared_ptr<std::vector<std::vector<int8_t>>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::vector<std::vector<uint8_t>>>>(
        "std::shared_ptr<std::vector<std::vector<uint8_t>>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::vector<std::vector<int16_t>>>>(
        "std::shared_ptr<std::vector<std::vector<int16_t>>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::vector<std::vector<uint16_t>>>>(
        "std::shared_ptr<std::vector<std::vector<uint16_t>>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::vector<std::vector<int32_t>>>>(
        "std::shared_ptr<std::vector<std::vector<int32_t>>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::vector<std::vector<uint32_t>>>>(
        "std::shared_ptr<std::vector<std::vector<uint32_t>>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::vector<std::vector<int64_t>>>>(
        "std::shared_ptr<std::vector<std::vector<int64_t>>>"s, true);
    registry.add_emitter_receiver<std::shared_ptr<std::vector<std::vector<uint64_t>>>>(
        "std::shared_ptr<std::vector<std::vector<uint64_t>>>"s, true);
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
                               const std::shared_ptr<PyOperator>& py_op)
    : gxf::GXFInputContext::GXFInputContext(execution_context, op, inputs), py_op_(py_op.get()) {}

PyOutputContext::PyOutputContext(ExecutionContext* execution_context, Operator* op,
                                 std::unordered_map<std::string, std::shared_ptr<IOSpec>>& outputs,
                                 const std::shared_ptr<PyOperator>& py_op)
    : gxf::GXFOutputContext::GXFOutputContext(execution_context, op, outputs),
      py_op_(py_op.get()) {}

}  // namespace holoscan
