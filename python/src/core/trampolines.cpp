/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gxf/std/tensor.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/gxf/gxf_tensor.hpp"

#include "../gxf/gxf.hpp"

namespace holoscan {

py::tuple vector2pytuple(const std::vector<std::shared_ptr<GILGuardedPyObject>>& vec) {
  py::tuple result(vec.size());
  int counter = 0;
  for (auto& arg_value : vec) {
    PyTuple_SET_ITEM(result.ptr(), counter++, arg_value->obj().release().ptr());
  }
  return result;
}

py::tuple vector2pytuple(const std::vector<holoscan::gxf::Entity>& vec) {
  py::tuple result(vec.size());
  int counter = 0;
  for (auto& arg_value : vec) {
    // Create a shared Entity (increase ref count)
    holoscan::PyEntity entity_wrapper(arg_value);
    py::object item = py::cast(entity_wrapper);
    PyTuple_SET_ITEM(result.ptr(), counter++, item.release().ptr());
  }
  return result;
}

py::object PyInputContext::py_receive(const std::string& name) {
  auto py_op = py_op_.cast<PyOperator*>();
  auto py_op_spec = py_op->py_shared_spec();

  bool is_receivers = false;
  for (const auto& receivers : py_op_spec->py_receivers()) {
    if (receivers.key() == name) { is_receivers = true; }
  }
  if (is_receivers) {
    auto any_result = receive<std::vector<std::any>>(name.c_str());
    if (any_result.empty()) { return py::make_tuple(); }

    // Check element type (querying the first element using the name '{name}:0')
    auto& element = any_result[0];
    auto& element_type = element.type();

    if (element_type == typeid(holoscan::gxf::Entity)) {
      std::vector<holoscan::gxf::Entity> result;
      try {
        for (auto& any_item : any_result) {
          if (element_type == typeid(nullptr_t)) {
            result.emplace_back();
            continue;
          }
          result.push_back(std::any_cast<holoscan::gxf::Entity>(any_item));
        }
      } catch (const std::bad_any_cast& e) {
        HOLOSCAN_LOG_ERROR(
            "Unable to receive input (std::vector<holoscan::gxf::Entity>) with name "
            "'{}' ({})",
            name,
            e.what());
      }
      py::tuple result_tuple = vector2pytuple(result);
      return result_tuple;
    } else if (element_type == typeid(std::shared_ptr<GILGuardedPyObject>)) {
      std::vector<std::shared_ptr<GILGuardedPyObject>> result;
      try {
        for (auto& any_item : any_result) {
          if (element_type == typeid(nullptr_t)) {
            result.emplace_back(nullptr);
            continue;
          }
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
    } else {
      return py::none();
    }
  } else {
    auto result = receive<std::any>(name.c_str());
    auto& result_type = result.type();

    if (result_type == typeid(holoscan::gxf::Entity)) {
      auto in_entity = std::any_cast<holoscan::gxf::Entity>(result);

      // Create a shared Entity (increase ref count)
      holoscan::PyEntity entity_wrapper(in_entity);
      return py::cast(entity_wrapper);
    } else if (result_type == typeid(std::shared_ptr<GILGuardedPyObject>)) {
      auto in_message = std::any_cast<std::shared_ptr<GILGuardedPyObject>>(result);
      return in_message->obj();
    } else if (result_type == typeid(nullptr_t)) {
      return py::none();
    } else {
      HOLOSCAN_LOG_ERROR("Invalid message type: {}", result_type.name());
      return py::none();
    }
  }
}

void PyOutputContext::py_emit(py::object& data, const std::string& name) {
  if (py::isinstance<holoscan::PyEntity>(data)) {
    auto entity = gxf::Entity(static_cast<nvidia::gxf::Entity>(data.cast<holoscan::PyEntity>()));
    emit<holoscan::gxf::Entity>(entity, name.c_str());
  } else {
    auto data_ptr = std::make_shared<GILGuardedPyObject>(data);
    emit<GILGuardedPyObject>(data_ptr, name.c_str());
  }
}

}  // namespace holoscan
