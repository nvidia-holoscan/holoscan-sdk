/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_GXF_ENTITY_HPP
#define PYHOLOSCAN_GXF_ENTITY_HPP

#include <pybind11/pybind11.h>

#include "../core/execution_context.hpp"

#include <holoscan/core/gxf/entity.hpp>

namespace py = pybind11;

namespace holoscan {

void init_entity(py::module_&);

class PyEntity : public gxf::Entity {
 public:
  /* Inherit the constructors */
  using gxf::Entity::Entity;

  static PyEntity py_create(const PyExecutionContext& ctx);

  using gxf::Entity::operator bool;  // inherit operator bool

  py::object py_get(const char* name = nullptr, bool log_errors = true) const;
  py::object py_add(const py::object& value, const char* name = nullptr);
};

}  // namespace holoscan

#endif /* PYHOLOSCAN_GXF_ENTITY_HPP */
