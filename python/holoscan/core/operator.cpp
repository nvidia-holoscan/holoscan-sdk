/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "operator.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

#include <memory>
#include <string>
#include <vector>

#include "kwarg_handling.hpp"
#include "operator_pydoc.hpp"
#include "gil_guarded_pyobject.hpp"
#include "holoscan/core/component.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/condition.hpp"
#include "holoscan/core/expected.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resource.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan {

void init_operator(py::module_& m) {
  py::class_<OperatorSpec, ComponentSpec, std::shared_ptr<OperatorSpec>>(
      m, "OperatorSpec", R"doc(Operator specification class.)doc")
      .def(py::init<Fragment*>(), "fragment"_a, doc::OperatorSpec::doc_OperatorSpec)
      .def("input",
           py::overload_cast<>(&OperatorSpec::input<gxf::Entity>),
           doc::OperatorSpec::doc_input,
           py::return_value_policy::reference_internal)
      .def("input",
           py::overload_cast<std::string>(&OperatorSpec::input<gxf::Entity>),
           "name"_a,
           doc::OperatorSpec::doc_input_kwargs,
           py::return_value_policy::reference_internal)
      .def("output",
           py::overload_cast<>(&OperatorSpec::output<gxf::Entity>),
           doc::OperatorSpec::doc_output,
           py::return_value_policy::reference_internal)
      .def("output",
           py::overload_cast<std::string>(&OperatorSpec::output<gxf::Entity>),
           "name"_a,
           doc::OperatorSpec::doc_output_kwargs,
           py::return_value_policy::reference_internal)
      .def_property_readonly(
          "description", &OperatorSpec::description, doc::OperatorSpec::doc_description)
      .def(
          "__repr__",
          [](const OperatorSpec& spec) { return spec.description(); },
          R"doc(Return repr(self).)doc");

  // Note: In the case of OperatorSpec, InputContext, OutputContext, ExecutionContext,
  //       there are a separate, independent wrappers for PyOperatorSpec, PyInputContext,
  //       PyOutputContext, PyExecutionContext. These Py* variants are not exposed directly
  //       to end users of the API, but are used internally to enable native operators
  //       defined from python via inheritance from the `Operator` class as defined in
  //       core/__init__.py.

  py::enum_<ParameterFlag>(m, "ParameterFlag", doc::ParameterFlag::doc_ParameterFlag)
      .value("NONE", ParameterFlag::kNone)
      .value("OPTIONAL", ParameterFlag::kOptional)
      .value("DYNAMIC", ParameterFlag::kDynamic);

  py::class_<PyOperatorSpec, OperatorSpec, std::shared_ptr<PyOperatorSpec>>(
      m, "PyOperatorSpec", R"doc(Operator specification class.)doc")
      .def(py::init<Fragment*, py::object>(),
           "fragment"_a,
           "op"_a,
           doc::OperatorSpec::doc_OperatorSpec)
      .def("param",
           &PyOperatorSpec::py_param,
           "Register parameter",
           "name"_a,
           "default_value"_a = py::none(),
           "flag"_a = ParameterFlag::kNone,
           doc::OperatorSpec::doc_param);

  // note: added py::dynamic_attr() to allow dynamically adding attributes in a Python subclass
  py::class_<Operator, Component, PyOperator, std::shared_ptr<Operator>> operator_class(
      m, "Operator", py::dynamic_attr(), doc::Operator::doc_Operator_args_kwargs);

  operator_class
      .def(py::init<py::object, Fragment*, const py::args&, const py::kwargs&>(),
           doc::Operator::doc_Operator_args_kwargs)
      .def_property("name",
                    py::overload_cast<>(&Operator::name, py::const_),
                    (Operator & (Operator::*)(const std::string&)) & Operator::name,
                    doc::Operator::doc_name)
      .def_property_readonly(
          "fragment", py::overload_cast<>(&Operator::fragment), doc::Operator::doc_fragment)
      .def_property("spec",
                    &Operator::spec_shared,
                    py::overload_cast<const std::shared_ptr<OperatorSpec>&>(&Operator::spec))
      .def_property_readonly("conditions", &Operator::conditions, doc::Operator::doc_conditions)
      .def_property_readonly("resources", &Operator::resources, doc::Operator::doc_resources)
      .def_property_readonly(
          "operator_type", &Operator::operator_type, doc::Operator::doc_operator_type)
      .def("add_arg",
           py::overload_cast<const Arg&>(&Operator::add_arg),
           "arg"_a,
           doc::Operator::doc_add_arg_Arg)
      .def("add_arg",
           py::overload_cast<const ArgList&>(&Operator::add_arg),
           "arg"_a,
           doc::Operator::doc_add_arg_ArgList)
      .def(
          "add_arg",
          [](Operator& op, const py::kwargs& kwargs) {
            return op.add_arg(kwargs_to_arglist(kwargs));
          },
          doc::Operator::doc_add_arg_kwargs)
      .def("add_arg",
           py::overload_cast<const std::shared_ptr<Condition>&>(&Operator::add_arg),
           "arg"_a,
           doc::Operator::doc_add_arg_condition)
      // Note: to avoid a doc build error, only list Parameters in docstring of the last overload
      .def("add_arg",
           py::overload_cast<const std::shared_ptr<Resource>&>(&Operator::add_arg),
           "arg"_a,
           doc::Operator::doc_add_arg_resource)
      .def("initialize",
           &Operator::initialize,
           doc::Operator::doc_initialize)                        // note: virtual function
      .def("setup", &Operator::setup, doc::Operator::doc_setup)  // note: virtual function
      .def("start",
           &Operator::start,  // note: virtual function
           doc::Operator::doc_start,
           py::call_guard<py::gil_scoped_release>())  // note: should release GIL
      .def("stop",
           &Operator::stop,  // note: virtual function
           doc::Operator::doc_stop,
           py::call_guard<py::gil_scoped_release>())  // note: should release GIL
      .def("compute",
           &Operator::compute,  // note: virtual function
           doc::Operator::doc_compute,
           py::call_guard<py::gil_scoped_release>())  // note: should release GIL
      .def_property_readonly("description", &Operator::description, doc::Operator::doc_description)
      .def(
          "__repr__",
          [](const py::object& obj) {
            // use py::object and obj.cast to avoid a segfault if object has not been initialized
            auto op = obj.cast<std::shared_ptr<Operator>>();
            if (op) { return op->description(); }
            return std::string("<Component: None>");
          },
          R"doc(Return repr(self).)doc");

  py::enum_<Operator::OperatorType>(operator_class, "OperatorType")
      .value("NATIVE", Operator::OperatorType::kNative)
      .value("GXF", Operator::OperatorType::kGXF);
}
}  // namespace holoscan
