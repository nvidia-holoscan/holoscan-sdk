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

#ifndef PYBIND11_GXF_GXF_HPP
#define PYBIND11_GXF_GXF_HPP

#include "./gxf_pydoc.hpp"

#include <pybind11/pybind11.h>

#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/gxf/gxf_tensor.hpp"

#include "../core/trampolines.hpp"

namespace py = pybind11;

namespace holoscan {

class PyEntity : public gxf::Entity {
 public:
  /* Inherit the constructors */
  using gxf::Entity::Entity;

  static PyEntity py_create(const PyExecutionContext& ctx);

  using gxf::Entity::operator bool;  // inherit operator bool

  py::object py_get(const char* name = nullptr) const;
  py::object py_add(const py::object& value, const char* name = nullptr);
};

}  // namespace holoscan

#endif /* PYBIND11_GXF_GXF_HPP */
