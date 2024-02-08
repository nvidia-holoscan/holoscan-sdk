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

#include "application.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "application_pydoc.hpp"
#include "fragment_pydoc.hpp"
#include "holoscan/core/application.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "tensor.hpp"

using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan {

void init_application(py::module_& m) {
  // note: added py::dynamic_attr() to allow dynamically adding attributes in a Python subclass
  //       added std::shared_ptr<Fragment> to allow the custom holder type to be used
  //         (see https://github.com/pybind/pybind11/issues/956)
  py::class_<Application, Fragment, PyApplication, std::shared_ptr<Application>>(
      m, "Application", py::dynamic_attr(), doc::Application::doc_Application)
      .def(py::init<const std::vector<std::string>&>(),
           "argv"_a = std::vector<std::string>(),
           doc::Application::doc_Application)
      .def_property("description",
                    py::overload_cast<>(&Application::description),
                    (Application & (Application::*)(const std::string&)&)&Application::description,
                    doc::Application::doc_description)
      .def_property("version",
                    py::overload_cast<>(&Application::version),
                    (Application & (Application::*)(const std::string&)&)&Application::version,
                    doc::Application::doc_version)
      .def_property_readonly(
          "argv", [](PyApplication& app) { return app.py_argv(); }, doc::Application::doc_argv)
      .def_property_readonly("options",
                             &Application::options,
                             doc::Application::doc_options,
                             py::return_value_policy::reference_internal)
      .def_property_readonly(
          "fragment_graph", &Application::fragment_graph, doc::Application::doc_fragment_graph)
      .def("add_operator",
           &Application::add_operator,
           "op"_a,
           doc::Application::doc_add_operator)  // note: virtual function
      .def("add_fragment",
           &Application::add_fragment,
           "frag"_a,
           doc::Application::doc_add_fragment)  // note: virtual function
      // TODO: sphinx API doc build complains if more than one overloaded add_flow method has a
      //       docstring specified. For now using the docstring defined for 3-argument
      //       Operator-based version and describing the other variants in the Notes section.
      .def(  // note: virtual function
          "add_flow",
          py::overload_cast<const std::shared_ptr<Operator>&, const std::shared_ptr<Operator>&>(
              &Application::add_flow),
          "upstream_op"_a,
          "downstream_op"_a)
      .def(  // note: virtual function
          "add_flow",
          py::overload_cast<const std::shared_ptr<Operator>&,
                            const std::shared_ptr<Operator>&,
                            std::set<std::pair<std::string, std::string>>>(&Application::add_flow),
          "upstream_op"_a,
          "downstream_op"_a,
          "port_pairs"_a,
          doc::Fragment::doc_add_flow_pair)
      .def(  // note: virtual function
          "add_flow",
          py::overload_cast<const std::shared_ptr<Fragment>&,
                            const std::shared_ptr<Fragment>&,
                            std::set<std::pair<std::string, std::string>>>(&Application::add_flow),
          "upstream_frag"_a,
          "downstream_frag"_a,
          "port_pairs"_a)
      .def("compose",
           &Application::compose,
           doc::Application::doc_compose)  // note: virtual function
      .def("run",
           &Application::run,
           doc::Application::doc_run,
           py::call_guard<py::gil_scoped_release>())  // note: virtual function/should release GIL
      .def(
          "__repr__",
          [](const py::object& obj) {
            // use py::object and obj.cast to avoid a segfault if object has not been initialized
            auto app = obj.cast<std::shared_ptr<Application>>();
            if (app) { return fmt::format("<holoscan.Application: name:{}>", app->name()); }
            return std::string("<Application: None>");
          },
          R"doc(Return repr(self).)doc");
}

}  // namespace holoscan
