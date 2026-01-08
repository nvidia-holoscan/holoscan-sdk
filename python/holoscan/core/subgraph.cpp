/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "subgraph.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fmt/format.h>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "gil_guarded_pyobject.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/subgraph.hpp"
#include "holoscan/logger/logger.hpp"
#include "kwarg_handling.hpp"
#include "subgraph_pydoc.hpp"

using pybind11::literals::operator""_a;  // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

PySubgraph::PySubgraph(py::object subgraph, Fragment* fragment, const std::string& name)
    : Subgraph(fragment, name), py_subgraph_(std::move(subgraph)) {
  using std::string_literals::operator""s;

  py::gil_scoped_acquire scope_guard;
  py_compose_ = py::getattr(py_subgraph_, "compose");
}

PySubgraph::~PySubgraph() {
  try {
    py::gil_scoped_acquire scope_guard;
    // Clear Python references
    py_subgraph_ = py::none();
    py_compose_ = py::none();
  } catch (const std::exception& e) {
    // Silently handle any exceptions during cleanup
    try {
      HOLOSCAN_LOG_ERROR("PySubgraph destructor failed with {}", e.what());
    } catch (...) {
    }
  }
}

void PySubgraph::compose() {
  /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
  // Call the Python compose method
  py::gil_scoped_acquire scope_guard;
  py_compose_.operator()();
}

// Note: add_flow methods are implemented in Python via delegation to self.fragment.add_flow

void init_subgraph(py::module_& m) {
  // note: added py::dynamic_attr() to allow dynamically adding attributes in a Python subclass
  //       added std::shared_ptr<Subgraph> to allow the custom holder type to be used
  py::class_<Subgraph, PySubgraph, std::shared_ptr<Subgraph>>(
      m, "Subgraph", py::dynamic_attr(), doc::Subgraph::doc_Subgraph)
      .def(py::init<py::object, Fragment*, const std::string&>(),
           "subgraph"_a,
           "fragment"_a,
           "name"_a,
           doc::Subgraph::doc_Subgraph)
      .def(py::init([](py::object subgraph,
                       std::shared_ptr<Subgraph>
                           parent_subgraph,
                       const std::string& name) {
             // Extract the fragment from the parent subgraph
             Fragment* fragment = parent_subgraph->fragment();

             // Apply qualified naming using the parent subgraph's instance name
             std::string qualified_name = parent_subgraph->get_qualified_name(name, "subgraph");

             // Create the PySubgraph in the parent Subgraph's fragment
             auto py_subgraph = std::make_shared<PySubgraph>(subgraph, fragment, qualified_name);
             return py_subgraph;
           }),
           "subgraph"_a,
           "parent_subgraph"_a,
           "name"_a)
      .def_property_readonly("name", &Subgraph::name, doc::Subgraph::doc_name)
      // Note: `set_dynamic_flows` is not added here, but as a Python method to dispatch to
      // `Fragment.set_dynamic_flows` which will handle the operator lifetime.
      .def("add_operator", &Subgraph::add_operator, "op"_a, doc::Subgraph::doc_add_operator)
      // Note: add_flow methods are implemented in Python via delegation to self.fragment.add_flow
      // Interface port methods - Operator overloads
      .def("add_input_interface_port",
           py::overload_cast<const std::string&,
                             const std::shared_ptr<Operator>&,
                             const std::string&>(&Subgraph::add_input_interface_port),
           "external_name"_a,
           "internal_op"_a,
           "internal_port"_a,
           doc::Subgraph::doc_add_input_interface_port)
      .def("add_output_interface_port",
           py::overload_cast<const std::string&,
                             const std::shared_ptr<Operator>&,
                             const std::string&>(&Subgraph::add_output_interface_port),
           "external_name"_a,
           "internal_op"_a,
           "internal_port"_a,
           doc::Subgraph::doc_add_output_interface_port)
      .def("add_interface_port",
           py::overload_cast<const std::string&,
                             const std::shared_ptr<Operator>&,
                             const std::string&,
                             bool>(&Subgraph::add_interface_port),
           "external_name"_a,
           "internal_op"_a,
           "internal_port"_a,
           "is_input"_a,
           doc::Subgraph::doc_add_interface_port)
      // Interface port methods - Subgraph overloads (must omit docstring here and use
      // use a common one for both overloads).
      .def("add_input_interface_port",
           py::overload_cast<const std::string&,
                             const std::shared_ptr<Subgraph>&,
                             const std::string&>(&Subgraph::add_input_interface_port),
           "external_name"_a,
           "internal_subgraph"_a,
           "internal_interface_port"_a)
      .def("add_output_interface_port",
           py::overload_cast<const std::string&,
                             const std::shared_ptr<Subgraph>&,
                             const std::string&>(&Subgraph::add_output_interface_port),
           "external_name"_a,
           "internal_subgraph"_a,
           "internal_interface_port"_a)
      .def("add_interface_port",
           py::overload_cast<const std::string&,
                             const std::shared_ptr<Subgraph>&,
                             const std::string&,
                             bool>(&Subgraph::add_interface_port),
           "external_name"_a,
           "internal_subgraph"_a,
           "internal_interface_port"_a,

           "is_input"_a)
      // Execution interface port methods - Operator overloads
      .def("add_input_exec_interface_port",
           py::overload_cast<const std::string&, const std::shared_ptr<Operator>&>(
               &Subgraph::add_input_exec_interface_port),
           "external_name"_a,
           "internal_op"_a,
           doc::Subgraph::doc_add_input_exec_interface_port)
      .def("add_output_exec_interface_port",
           py::overload_cast<const std::string&, const std::shared_ptr<Operator>&>(
               &Subgraph::add_output_exec_interface_port),
           "external_name"_a,
           "internal_op"_a,
           doc::Subgraph::doc_add_output_exec_interface_port)
      // Execution interface port methods - Subgraph overloads
      .def("add_input_exec_interface_port",
           py::overload_cast<const std::string&,
                             const std::shared_ptr<Subgraph>&,
                             const std::string&>(&Subgraph::add_input_exec_interface_port),
           "external_name"_a,
           "internal_subgraph"_a,
           "internal_interface_port"_a)
      .def("add_output_exec_interface_port",
           py::overload_cast<const std::string&,
                             const std::shared_ptr<Subgraph>&,
                             const std::string&>(&Subgraph::add_output_exec_interface_port),
           "external_name"_a,
           "internal_subgraph"_a,
           "internal_interface_port"_a)
      .def("compose", &Subgraph::compose, doc::Subgraph::doc_compose)  // note: virtual function
      .def("is_composed", &Subgraph::is_composed, doc::Subgraph::doc_is_composed)
      .def("set_composed", &Subgraph::set_composed, "composed"_a, doc::Subgraph::doc_set_composed)
      .def(
          "interface_port_names",
          [](const Subgraph& self) {
            std::vector<std::string> names;
            for (const auto& [name, _] : self.interface_ports()) {
              names.push_back(name);
            }
            return names;
          },
          "Get the list of interface port names")
      .def(
          "__repr__",
          [](const py::object& obj) {
            // use py::object and obj.cast to avoid a segfault if object has not been initialized
            try {
              // cast either succeeds and returns a valid non-null pointer or throws py::cast_error
              auto subgraph = obj.cast<std::shared_ptr<Subgraph>>();
              return fmt::format("<holoscan.Subgraph: name:{}>", subgraph->name());
            } catch (const py::cast_error&) {
              return std::string("<Subgraph: None>");
            }
          },
          R"doc(Return repr(self).)doc");
}

}  // namespace holoscan
