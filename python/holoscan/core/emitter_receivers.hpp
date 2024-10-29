/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_CORE_EMITTER_RECEIVERS_HPP
#define PYHOLOSCAN_CORE_EMITTER_RECEIVERS_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // needed for py::cast to work with STL container types

#include <any>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../gxf/entity.hpp"

#include "emitter_receiver_registry.hpp"
#include "gil_guarded_pyobject.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/io_context.hpp"
#include "io_context.hpp"
#include "tensor.hpp"  // for PyTensor

namespace py = pybind11;

namespace holoscan {

py::tuple vector2pytuple(const std::vector<std::shared_ptr<GILGuardedPyObject>>& vec) {
  py::tuple result(vec.size());
  int counter = 0;
  for (auto& arg_value : vec) {
    // Increase the ref count of the Python object because we are going to store (copy) it in a
    // tuple.
    arg_value->obj().inc_ref();
    // We should not release arg_value->obj() because the Python object can be referenced by the
    // input context of other operators
    PyTuple_SET_ITEM(result.ptr(), counter++, arg_value->obj().ptr());
  }
  return result;
}

py::object gxf_entity_to_py_object(holoscan::gxf::Entity in_entity) {
  // Create a shared Entity (increase ref count)
  holoscan::PyEntity entity_wrapper(in_entity);

  try {
    auto components_expected = entity_wrapper.findAll();
    auto components = components_expected.value();
    auto n_components = components.size();

    HOLOSCAN_LOG_DEBUG("py_receive: Entity Case");
    if ((n_components == 1) && (components[0]->name()[0] == '#')) {
      // special case for single non-TensorMap tensor
      // (will have been serialized with a name starting with #numpy, #cupy or #holoscan)
      HOLOSCAN_LOG_DEBUG("py_receive: SINGLE COMPONENT WITH # NAME");
      auto component = components[0];
      std::string component_name = component->name();
      py::object holoscan_pytensor = entity_wrapper.py_get(component_name.c_str());

      if (component_name.find("#numpy") != std::string::npos) {
        HOLOSCAN_LOG_DEBUG("py_receive: name starting with #numpy");
        // cast the holoscan::Tensor to a NumPy array
        py::module_ numpy;
        try {
          numpy = py::module_::import("numpy");
        } catch (const pybind11::error_already_set& e) {
          if (e.matches(PyExc_ImportError)) {
            throw pybind11::import_error(
                fmt::format("Failed to import numpy to deserialize array with "
                            "__array_interface__ attribute: {}",
                            e.what()));
          } else {
            throw;
          }
        }
        // py::object holoscan_pytensor_obj = py::cast(holoscan_pytensor);
        py::object numpy_array = numpy.attr("asarray")(holoscan_pytensor);
        return numpy_array;
      } else if (component_name.find("#cupy") != std::string::npos) {
        HOLOSCAN_LOG_DEBUG("py_receive: name starting with #cupy");
        // cast the holoscan::Tensor to a CuPy array
        py::module_ cupy;
        try {
          cupy = py::module_::import("cupy");
        } catch (const pybind11::error_already_set& e) {
          if (e.matches(PyExc_ImportError)) {
            throw pybind11::import_error(
                fmt::format("Failed to import cupy to deserialize array with "
                            "__cuda_array_interface__ attribute: {}",
                            e.what()));
          } else {
            throw;
          }
        }
        py::object cupy_array = cupy.attr("asarray")(holoscan_pytensor);
        return cupy_array;
      } else if (component_name.find("#holoscan") != std::string::npos) {
        HOLOSCAN_LOG_DEBUG("py_receive: name starting with #holoscan");
        return holoscan_pytensor;
      } else {
        throw std::runtime_error(
            fmt::format("Invalid tensor name (if # is the first character in the name, the "
                        "name must start with #numpy, #cupy or #holoscan). Found: {}",
                        component_name));
      }
    } else {
      HOLOSCAN_LOG_DEBUG("py_receive: TensorMap case");
      py::dict dict_tensor;
      for (size_t i = 0; i < n_components; i++) {
        auto component = components[i];
        auto component_name = component->name();
        if (std::string(component_name).compare("message_label") == 0) {
          // Skip checking for Tensor as it's the message label for flow tracking
          continue;
        }
        if (std::string(component_name).compare("metadata_") == 0) {
          // Skip checking for Tensor as it's a  MetadataDictionary object
          continue;
        }
        if (std::string(component_name).compare("cuda_stream_id_") == 0) {
          // Skip checking for Tensor as it's a stream ID from CudaStreamHandler
          continue;
        }
        auto holoscan_pytensor = entity_wrapper.py_get(component_name);
        if (holoscan_pytensor) { dict_tensor[component_name] = holoscan_pytensor; }
      }
      return dict_tensor;
    }
  } catch (const std::bad_any_cast& e) {
    throw std::runtime_error(
        fmt::format("Unable to cast the received data to the specified type (holoscan::gxf::"
                    "Entity): {}",
                    e.what()));
  }
}

/* Emitter for case where user explicitly creates a GXF holoscan.gxf.Entity from Python
 *
 * Using holoscan.gxf.Entity was the primary way to transmit multiple tensors before TensorMap
 * support was added. After tensormap, it is more common to transmit a dict of tensor-like
 * objects instead.
 */
template <>
struct emitter_receiver<holoscan::PyEntity> {
  static void emit(py::object& data, const std::string& name, PyOutputContext& op_output,
                   const int64_t acq_timestamp = -1) {
    py::gil_scoped_release release;
    auto entity = gxf::Entity(static_cast<nvidia::gxf::Entity>(data.cast<holoscan::PyEntity>()));
    op_output.emit<holoscan::gxf::Entity>(entity, name.c_str(), acq_timestamp);
    return;
  }
  static py::object receive(std::any result, const std::string& name, PyInputContext& op_input) {
    // unused (receive type is always holoscan::gxf:Entity, not holoscan::PyEntity)
    return py::none();
  }
};

/* Receiver for holoscan::gxf::Entity.
 *
 * This receiver currently only extracts Holoscan, NumPy or CuPy tensors from the entity. It
 * explicitly ignores any "cuda_stream_pool" component that may have been added by
 * CudaStreamHandler. It also explicitly ignores any component named "message_label" that may
 * have been added by the data flow tracking feature.
 */
template <>
struct emitter_receiver<holoscan::gxf::Entity> {
  static void emit(py::object& data, const std::string& name, PyOutputContext& op_output,
                   const int64_t acq_timestamp = -1) {
    // unused (emit type is holoscan::PyEntity, not holoscan::gxf:Entity)
    return;
  }

  static py::object receive(std::any result, const std::string& name, PyInputContext& op_input) {
    auto in_entity = std::any_cast<holoscan::gxf::Entity>(result);
    return gxf_entity_to_py_object(std::move(in_entity));
  }
};

/* Emitter for holoscan.core.Tensor with special handling for 3rd party tensor interoperability.
 *
 * Names any holoscan::Tensor supporting ``__cuda_array_interface__`` as "#cupy: tensor".
 * This will cause ``emitter_receiver<holoscan::gxf::Entity>::receive`` to convert this tensor
 * to a CuPy tensor on receive by a downstream Python operator.
 *
 * Names any holoscan::Tensor supporting ``__array_interface__`` as "#numpy: tensor".
 * This will cause ``emitter_receiver<holoscan::gxf::Entity>::receive`` to convert this tensor
 * to a NumPy tensor on receive by a downstream Python operator.
 *
 * If the tensor-like object, ``data``, doesn't support either ``__array_interface__`` or
 * ``__cuda_array_interface__``, but only DLPack then it will be received by a downstream
 * Python operator as a ``holoscan::Tensor``.
 *
 * This special handling is done to allow equivalent behavior when sending NumPy and CuPy
 * tensors between operators in both single fragment and distributed applications.
 */
template <>
struct emitter_receiver<holoscan::Tensor> {
  static void emit(py::object& data, const std::string& name, PyOutputContext& op_output,
                   const int64_t acq_timestamp = -1) {
    HOLOSCAN_LOG_DEBUG("py_emit: tensor-like over UCX connector");
    // For tensor-like data, we should create an entity and transmit using the holoscan::Tensor
    // serializer. cloudpickle fails to serialize PyTensor and we want to avoid using it anyways
    // as it would be inefficient to serialize the tensor to a string.
    py::gil_scoped_release release;
    auto entity = nvidia::gxf::Entity::New(op_output.gxf_context());
    if (!entity) { throw std::runtime_error("Failed to create entity"); }
    auto py_entity = static_cast<PyEntity>(entity.value());

    py::gil_scoped_acquire acquire;
    auto py_tensor_obj = PyTensor::as_tensor(data);
    if (py::hasattr(data, "__cuda_array_interface__")) {
      // checking with __cuda_array_interface__ instead of
      // if (py::isinstance(value, cupy.attr("ndarray")))

      // This way we don't have to add try/except logic around importing the CuPy module.
      // One consequence of this is that Non-CuPy arrays having __cuda_array_interface__ will be
      // cast to CuPy arrays on deserialization.
      py_entity.py_add(py_tensor_obj, "#cupy: tensor");
    } else if (py::hasattr(data, "__array_interface__")) {
      // objects with __array_interface__ defined will be cast to NumPy array on
      // deserialization.
      py_entity.py_add(py_tensor_obj, "#numpy: tensor");
    } else {
      py_entity.py_add(py_tensor_obj, "#holoscan: tensor");
    }
    py::gil_scoped_release release2;
    op_output.emit<holoscan::gxf::Entity>(py_entity, name.c_str(), acq_timestamp);
    return;
  }
  static py::object receive(std::any result, const std::string& name, PyInputContext& op_input) {
    return py::none();
  }
};

/* Emit a Python dict as a TensorMap entity when possible, otherwise emit the Python object.
 *
 * If every entry in the dict is a tensor-like object then create a holoscan::gxf::Entity
 * containing a TensorMap corresponding to the tensors (no data copying is done). This allows
 * interfacing with any wrapped C++ operators taking tensors as input.
 *
 * Otherwise, emit the dictionary as-is via a std::shared_ptr<GILGuardedPyObject>.
 */
template <>
struct emitter_receiver<pybind11::dict> {
  static void emit(py::object& data, const std::string& name, PyOutputContext& op_output,
                   const int64_t acq_timestamp = -1) {
    bool is_tensormap = true;
    auto dict_obj = data.cast<py::dict>();

    // Check if all items in the dict are tensor-like
    for (auto& item : dict_obj) {
      auto& value = item.second;
      // Check if item is tensor-like
      if (!is_tensor_like(py::reinterpret_borrow<py::object>(value))) {
        is_tensormap = false;
        break;
      }
    }
    if (is_tensormap) {
      // Create an Entity containing the TensorMap
      auto dict_obj = data.cast<py::dict>();
      py::gil_scoped_release release;
      auto entity = nvidia::gxf::Entity::New(op_output.gxf_context());
      if (!entity) { throw std::runtime_error("Failed to create entity"); }
      auto py_entity = static_cast<PyEntity>(entity.value());

      py::gil_scoped_acquire acquire;
      for (auto& item : dict_obj) {
        std::string key = item.first.cast<std::string>();
        auto& value = item.second;
        auto value_obj = py::reinterpret_borrow<py::object>(value);
        auto py_tensor_obj = PyTensor::as_tensor(value_obj);
        py_entity.py_add(py_tensor_obj, key.c_str());
      }
      py::gil_scoped_release release2;
      op_output.emit<holoscan::gxf::Entity>(py_entity, name.c_str(), acq_timestamp);
      return;
    } else {
      // If the dict is not a TensorMap, pass it as a Python object
      HOLOSCAN_LOG_DEBUG("py_emit: dict, but not a tensormap");
      auto data_ptr = std::make_shared<GILGuardedPyObject>(data);
      py::gil_scoped_release release;
      op_output.emit<std::shared_ptr<GILGuardedPyObject>>(
          std::move(data_ptr), name.c_str(), acq_timestamp);
      return;
    }
  }
  // pybind11::dict is never received directly, but only as a std::shared_ptr<GILGuardedPyObject>
  static py::object receive(std::any result, const std::string& name, PyInputContext& op_input) {
    return py::none();
  }
};

/* Emit and receive generic Python objects
 *
 * This is the typical emitter/receiver used for within-fragment communication between native
 * Python operators.
 */
template <>
struct emitter_receiver<std::shared_ptr<GILGuardedPyObject>> {
  static void emit(py::object& data, const std::string& name, PyOutputContext& op_output,
                   const int64_t acq_timestamp = -1) {
    // Emit everything else as a Python object.
    auto data_ptr = std::make_shared<GILGuardedPyObject>(data);
    py::gil_scoped_release release;
    op_output.emit<std::shared_ptr<GILGuardedPyObject>>(
        std::move(data_ptr), name.c_str(), acq_timestamp);
    return;
  }

  static py::object receive(std::any result, const std::string& name, PyInputContext& op_input) {
    HOLOSCAN_LOG_DEBUG("py_receive: Python object case");
    auto in_message = std::any_cast<std::shared_ptr<GILGuardedPyObject>>(result);
    return in_message->obj();
  }
};

/* Emit and receive Python objects that have been serialized to a string via ``cloudpickle``.
 *
 * This is used to send native Python objects between fragments of a distributed application.
 */
template <>
struct emitter_receiver<CloudPickleSerializedObject> {
  static void emit(py::object& data, const std::string& name, PyOutputContext& op_output,
                   const int64_t acq_timestamp = -1) {
    // use cloudpickle to serialize as a string
    py::module_ cloudpickle = py::module_::import("cloudpickle");
    py::bytes serialized = cloudpickle.attr("dumps")(data);
    py::gil_scoped_release release;
    CloudPickleSerializedObject serialized_obj{serialized.cast<std::string>()};
    op_output.emit<CloudPickleSerializedObject>(
        std::move(serialized_obj), name.c_str(), acq_timestamp);
    return;
  }

  static py::object receive(std::any result, const std::string& name, PyInputContext& op_input) {
    HOLOSCAN_LOG_DEBUG("py_receive: cloudpickle string case");
    auto serialized_obj = std::any_cast<CloudPickleSerializedObject>(result);
    py::module_ cloudpickle = py::module_::import("cloudpickle");
    py::object deserialized = cloudpickle.attr("loads")(py::bytes(serialized_obj.serialized));
    return deserialized;
  }
};

/* Emit or receive a nullptr.
 *
 * A Python operator receiving a C++ nullptr will convert it to Python's None.
 */
template <>
struct emitter_receiver<std::nullptr_t> {
  static void emit(py::object& data, const std::string& name, PyOutputContext& op_output,
                   const int64_t acq_timestamp = -1) {
    op_output.emit<std::nullptr_t>(nullptr, name.c_str(), acq_timestamp);
    return;
  }
  static py::object receive(std::any result, const std::string& name, PyInputContext& op_input) {
    return py::none();
  }
};

}  // namespace holoscan

#endif /* PYHOLOSCAN_CORE_EMITTER_RECEIVERS_HPP */
