/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
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
#include "trampolines.hpp"
#include <pybind11/stl.h>  // needed for py::cast to work with std::vector types
#include <memory>
#include <string>
#include <vector>
#include "../gxf/gxf.hpp"
#include "core.hpp"
#include "gxf/std/tensor.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/gxf/gxf_tensor.hpp"
#include "holoscan/operators/holoviz/holoviz.hpp"

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

py::tuple vector2pytuple(const std::vector<holoscan::gxf::Entity>& vec) {
  py::tuple result(vec.size());
  int counter = 0;
  for (auto& arg_value : vec) {
    // Create a shared Entity
    holoscan::PyEntity entity_wrapper(arg_value);
    // Create a Python object from the shared Entity
    py::object item = py::cast(entity_wrapper);
    // Move the Python object into the tuple
    PyTuple_SET_ITEM(result.ptr(), counter++, item.release().ptr());
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
      HOLOSCAN_LOG_DEBUG("py_receive: SINGLE COMPONENT WITH # NAME");
      // special case for single non-TensorMap tensor
      // (will have been serialized with a name starting with #numpy, #cupy or #holoscan)
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

py::object PyInputContext::py_receive(const std::string& name) {
  auto py_op = py_op_.cast<PyOperator*>();
  auto py_op_spec = py_op->py_shared_spec();

  bool is_receivers = false;
  for (const auto& receivers : py_op_spec->py_receivers()) {
    if (receivers.key() == name) { is_receivers = true; }
  }
  if (is_receivers) {
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

    if (element_type == typeid(holoscan::gxf::Entity)) {
      py::tuple result_tuple(any_result.size());
      int counter = 0;
      try {
        for (auto& any_item : any_result) {
          if (element_type == typeid(nullptr_t)) {
            // add None to the tuple
            PyTuple_SET_ITEM(result_tuple.ptr(), counter++, py::none().release().ptr());
            continue;
          }
          auto in_entity = std::any_cast<holoscan::gxf::Entity>(any_item);

          // Get the Python object from the entity
          py::object in_obj = gxf_entity_to_py_object(in_entity);

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
    } else if (element_type == typeid(std::shared_ptr<GILGuardedPyObject>)) {
      std::vector<std::shared_ptr<GILGuardedPyObject>> result;
      try {
        for (auto& any_item : any_result) {
          result.push_back(std::any_cast<std::shared_ptr<GILGuardedPyObject>>(any_item));
        }
      } catch (const std::bad_any_cast& e) {
        HOLOSCAN_LOG_ERROR(
            "Unable to receive input (std::vector<std::shared_ptr<GILGuardedPyObject>>) with name "
            "'{}' ({})",
            name,
            e.what());
      }
      py::tuple result_tuple = vector2pytuple(result);
      return result_tuple;
    } else if (element_type == typeid(std::string)) {
      std::vector<std::shared_ptr<GILGuardedPyObject>> result;
      try {
        for (auto& any_item : any_result) {
          std::string obj_str = std::any_cast<std::string>(any_item);

          py::module_ cloudpickle = py::module_::import("cloudpickle");
          py::object deserialized = cloudpickle.attr("loads")(py::bytes(obj_str));

          result.push_back(std::make_shared<GILGuardedPyObject>(deserialized));
        }
      } catch (const std::bad_any_cast& e) {
        HOLOSCAN_LOG_ERROR(
            "Unable to receive input (std::vector<std::str>) with name "
            "'{}' ({})",
            name,
            e.what());
      }
      py::tuple result_tuple = vector2pytuple(result);
      return result_tuple;
    } else if (element_type == typeid(std::vector<holoscan::ops::HolovizOp::InputSpec>)) {
      py::tuple result_tuple(any_result.size());
      int counter = 0;
      try {
        for (auto& any_item : any_result) {
          HOLOSCAN_LOG_DEBUG("py_receive: HolovizOp::InputSpec case");

          auto specs = std::any_cast<std::vector<holoscan::ops::HolovizOp::InputSpec>>(any_item);

          // cast vector to Python list of HolovizOp.InputSpec
          py::object py_specs = py::cast(specs);

          // Move the Python object into the tuple
          PyTuple_SET_ITEM(result_tuple.ptr(), counter++, py_specs.release().ptr());
        }
      } catch (const std::bad_any_cast& e) {
        HOLOSCAN_LOG_ERROR(
            "Unable to receive input (std::vector<holoscan::ops::HolovizOp::InputSpec>) with name "
            "'{}' ({})",
            name,
            e.what());
      }
      return result_tuple;
    } else {
      return py::none();
    }
  } else {
    auto maybe_result = receive<std::any>(name.c_str());
    if (!maybe_result.has_value()) {
      HOLOSCAN_LOG_ERROR("Unable to receive input (std::any) with name '{}'", name);
      return py::none();
    }
    auto result = maybe_result.value();
    auto& result_type = result.type();
    if (result_type == typeid(holoscan::gxf::Entity)) {
      auto in_entity = std::any_cast<holoscan::gxf::Entity>(result);
      return gxf_entity_to_py_object(in_entity);
    } else if (result_type == typeid(std::shared_ptr<GILGuardedPyObject>)) {
      HOLOSCAN_LOG_DEBUG("py_receive: Python object case");
      auto in_message = std::any_cast<std::shared_ptr<GILGuardedPyObject>>(result);
      return in_message->obj();
    } else if (result_type == typeid(std::string)) {
      HOLOSCAN_LOG_DEBUG("py_receive: cloudpickle string case");
      std::string obj_str = std::any_cast<std::string>(result);

      py::module_ cloudpickle = py::module_::import("cloudpickle");
      py::object deserialized = cloudpickle.attr("loads")(py::bytes(obj_str));

      return deserialized;
    } else if (result_type == typeid(std::vector<holoscan::ops::HolovizOp::InputSpec>)) {
      HOLOSCAN_LOG_DEBUG("py_receive: HolovizOp::InputSpec case");
      // can directly return vector<InputSpec>
      auto specs = std::any_cast<std::vector<holoscan::ops::HolovizOp::InputSpec>>(result);
      py::object py_specs = py::cast(specs);
      return py_specs;
    } else if (result_type == typeid(nullptr_t)) {
      return py::none();
    } else {
      HOLOSCAN_LOG_ERROR("Invalid message type: {}", result_type.name());
      return py::none();
    }
  }
}

bool is_tensor_like(py::object value) {
  return ((py::hasattr(value, "__dlpack__") && py::hasattr(value, "__dlpack_device__")) ||
          py::isinstance<holoscan::PyTensor>(value) ||
          py::hasattr(value, "__cuda_array_interface__") ||
          py::hasattr(value, "__array_interface__"));
}

void PyOutputContext::py_emit(py::object& data, const std::string& name) {
  // If this is a PyEntity emit a gxf::Entity so that it can be consumed by non-Python operator.
  if (py::isinstance<holoscan::PyEntity>(data)) {
    auto entity = gxf::Entity(static_cast<nvidia::gxf::Entity>(data.cast<holoscan::PyEntity>()));
    emit<holoscan::gxf::Entity>(entity, name.c_str());
    return;
  }

  /// @todo Workaround for HoloviOp which expects a list of input specs.
  /// If we don't do the cast here the operator receives a python list object. There should be a
  /// generic way for this, or the operator needs to register expected types.
  if (py::isinstance<py::list>(data) || py::isinstance<py::tuple>(data)) {
    if (py::len(data) > 0) {
      auto seq = data.cast<py::sequence>();
      if (py::isinstance<holoscan::ops::HolovizOp::InputSpec>(seq[0])) {
        auto input_spec = data.cast<std::vector<holoscan::ops::HolovizOp::InputSpec>>();
        emit<std::vector<holoscan::ops::HolovizOp::InputSpec>>(input_spec, name.c_str());
        return;
      }
    }
  }
  bool is_tensormap_like = false;
  // check if data is dict and all items in the dict are array-like/holoscan Tensor type
  if (py::isinstance<py::dict>(data)) {
    is_tensormap_like = true;
    auto dict_obj = data.cast<py::dict>();

    // Check if all items in the dict are tensor-like
    for (auto& item : dict_obj) {
      auto& value = item.second;
      // Check if item is tensor-like
      if (!is_tensor_like(py::reinterpret_borrow<py::object>(value))) {
        is_tensormap_like = false;
        break;
      }
    }
    if (is_tensormap_like) {
      auto entity = nvidia::gxf::Entity::New(gxf_context());
      if (!entity) { throw std::runtime_error("Failed to create entity"); }
      auto py_entity = static_cast<PyEntity>(entity.value());

      for (auto& item : dict_obj) {
        std::string key = item.first.cast<std::string>();
        auto& value = item.second;
        auto value_obj = py::reinterpret_borrow<py::object>(value);
        auto py_tensor_obj = PyTensor::as_tensor(value_obj);
        py_entity.py_add(py_tensor_obj, key.c_str());
      }
      emit<holoscan::gxf::Entity>(py_entity, name.c_str());
      return;
    }
  }

  bool is_ucx_connector = false;
  if (outputs_.find(name) != outputs_.end()) {
    auto connector_type = outputs_.at(name)->connector_type();
    is_ucx_connector = connector_type == IOSpec::ConnectorType::kUCX;
  }
  if (is_ucx_connector) {
    // TensorMap case was already handled above
    if (is_tensor_like(data)) {
      // For tensor-like data, we should create an entity and transmit using the holoscan::Tensor
      // serializer. cloudpickle fails to serialize PyTensor and we want to avoid using it anyways
      // as it would be inefficient to serialize the tensor to a string.
      auto entity = nvidia::gxf::Entity::New(gxf_context());
      if (!entity) { throw std::runtime_error("Failed to create entity"); }
      auto py_entity = static_cast<PyEntity>(entity.value());

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
      emit<holoscan::gxf::Entity>(py_entity, name.c_str());
      return;
    }
    // use cloudpickle to serialize as a string
    py::module_ cloudpickle = py::module_::import("cloudpickle");
    py::bytes serialized = cloudpickle.attr("dumps")(data);
    emit<std::string>(serialized.cast<std::string>(), name.c_str());
    return;
  }
  // Emit everything else as a Python object.
  auto data_ptr = std::make_shared<GILGuardedPyObject>(data);
  emit<std::shared_ptr<GILGuardedPyObject>>(data_ptr, name.c_str());
}

}  // namespace holoscan
