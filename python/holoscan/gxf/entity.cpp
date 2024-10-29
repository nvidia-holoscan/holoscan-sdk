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

#include "entity.hpp"

#include <pybind11/pybind11.h>

#include <memory>

#include "../core/dl_converter.hpp"
#include "../core/tensor.hpp"

#include "entity_pydoc.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/gxf/entity.hpp"

using pybind11::literals::operator""_a;  // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

void init_entity(py::module_& m) {
  py::class_<gxf::Entity, std::shared_ptr<gxf::Entity>>(m, "Entity", doc::Entity::doc_Entity)
      .def(py::init<>(), doc::Entity::doc_Entity);

  py::class_<PyEntity, gxf::Entity, std::shared_ptr<PyEntity>>(
      m, "PyEntity", doc::Entity::doc_Entity)
      .def(py::init(&PyEntity::py_create), doc::Entity::doc_Entity)
      .def("__bool__", &PyEntity::operator bool)
      .def("get", &PyEntity::py_get, "name"_a = "", doc::Entity::doc_get)
      .def("add", &PyEntity::py_add, "obj"_a, "name"_a = "");
}  // PYBIND11_MODULE

////////////////////////////////////////////////////////////////////////////////////////////////////
// PyEntity definition
////////////////////////////////////////////////////////////////////////////////////////////////////

PyEntity PyEntity::py_create(const PyExecutionContext& ctx) {
  auto result = nvidia::gxf::Entity::New(ctx.context());
  if (!result) { throw std::runtime_error("Failed to create entity"); }
  return static_cast<PyEntity>(result.value());
}

py::object PyEntity::py_get(const char* name) const {
  auto tensor = get<Tensor>(name);
  if (!tensor) { return py::none(); }

  auto py_tensor = py::cast(tensor);

  // Set array interface attributes
  set_array_interface(py_tensor, tensor->dl_ctx());
  return py_tensor;
}

py::object PyEntity::py_add(const py::object& value, const char* name) {
  if (py::isinstance<PyTensor>(value)) {
    std::shared_ptr<Tensor> tensor =
        std::static_pointer_cast<Tensor>(py::cast<std::shared_ptr<PyTensor>>(value));
    add<Tensor>(tensor, name);
    return value;
  }
  return py::none();
}

}  // namespace holoscan
