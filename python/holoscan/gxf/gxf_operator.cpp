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

#include <pybind11/pybind11.h>

#include <memory>
#include <string>

#include "gxf_operator_pydoc.hpp"

#include "holoscan/core/gxf/gxf_operator.hpp"
#include "holoscan/core/operator.hpp"

#include "gxf/core/gxf.h"

namespace py = pybind11;

namespace holoscan::ops {

// define trampoline class for GXFOperator to allow overriding virtual gxf_typename

class PyGXFOperator : public GXFOperator {
 public:
  /* Inherit the constructors */
  using GXFOperator::GXFOperator;

  /* Trampolines (need one for each virtual function) */
  const char* gxf_typename() const override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE_PURE(const char*, GXFOperator, gxf_typename);
  }
};

}  // namespace holoscan::ops

namespace holoscan {

void init_gxf_operator(py::module_& m) {
  py::class_<ops::GXFOperator, ops::PyGXFOperator, Operator, std::shared_ptr<ops::GXFOperator>>(
      m, "GXFOperator", doc::GXFOperator::doc_GXFOperator)
      .def(py::init<>(), doc::GXFOperator::doc_GXFOperator)
      .def_property_readonly(
          "gxf_typename", &ops::GXFOperator::gxf_typename, doc::GXFOperator::doc_gxf_typename)
      .def_property_readonly(
          "gxf_context", &ops::GXFOperator::gxf_context, doc::GXFOperator::doc_gxf_context)
      .def_property("gxf_eid",
                    py::overload_cast<>(&ops::GXFOperator::gxf_eid, py::const_),
                    py::overload_cast<gxf_uid_t>(&ops::GXFOperator::gxf_eid),
                    doc::GXFOperator::doc_gxf_eid)
      .def_property("gxf_cid",
                    py::overload_cast<>(&ops::GXFOperator::gxf_cid, py::const_),
                    py::overload_cast<gxf_uid_t>(&ops::GXFOperator::gxf_cid),
                    doc::GXFOperator::doc_gxf_cid)
      .def_property_readonly("gxf_entity_group_name",
                             &ops::GXFOperator::gxf_entity_group_name,
                             doc::GXFOperator::doc_gxf_entity_group_name)
      .def_property_readonly(
          "description", &ops::GXFOperator::description, doc::GXFOperator::doc_description)
      .def(
          "__repr__",
          [](const py::object& obj) {
            // use py::object and obj.cast to avoid a segfault if object has not been
            // initialized
            auto op = obj.cast<std::shared_ptr<ops::GXFOperator>>();
            if (op) { return op->description(); }
            return std::string("<GXFOperator: None>");
          },
          R"doc(Return repr(self).)doc");
}  // PYBIND11_MODULE

}  // namespace holoscan
