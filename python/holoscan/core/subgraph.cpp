/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "subgraph.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fmt/format.h>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/config.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/io_spec.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/subgraph.hpp>
#include <holoscan/logger/logger.hpp>
#include "gil_guarded_pyobject.hpp"
#include "kwarg_handling.hpp"
#include "subgraph_pydoc.hpp"

using pybind11::literals::operator""_a;  // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

PySubgraph::PySubgraph(py::object subgraph, Fragment* fragment, const std::string& name,
                       const std::string& config_file)
    : Subgraph(fragment, name, config_file), py_subgraph_(std::move(subgraph)) {
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
  // Bind InterfacePort::PortType enum
  py::enum_<InterfacePort::PortType>(m, "InterfacePortType", "Type of interface port")
      .value("DATA", InterfacePort::PortType::kData, "Regular data port (for data flow)")
      .value("EXECUTION",
             InterfacePort::PortType::kExecution,
             "Execution control port (for control flow)");

  // Bind InterfacePort::Mapping struct
  py::class_<InterfacePort::Mapping>(
      m,
      "InterfacePortMapping",
      "A single mapping from external interface port to internal operator port")
      .def_readonly("internal_operator",
                    &InterfacePort::Mapping::internal_operator,
                    "Internal operator that owns the port")
      .def_readonly("internal_port_name",
                    &InterfacePort::Mapping::internal_port_name,
                    "Port name on the internal operator")
      .def("__repr__", [](const InterfacePort::Mapping& mapping) {
        return fmt::format("<InterfacePortMapping: operator={}, port={}>",
                           mapping.internal_operator ? mapping.internal_operator->name() : "None",
                           mapping.internal_port_name);
      });

  // Bind InterfacePort struct
  py::class_<InterfacePort>(
      m,
      "InterfacePort",
      "Interface port that maps external subgraph port name to internal operator port(s)")
      .def_readonly("mappings", &InterfacePort::mappings, "List of internal operator/port mappings")
      .def_readonly(
          "is_input", &InterfacePort::is_input, "Whether this is an input port (vs output)")
      .def_readonly("port_type", &InterfacePort::port_type, "Port type (data or execution)")
      .def("__len__", &InterfacePort::size, "Get number of mappings")
      .def("__repr__", [](const InterfacePort& port) {
        std::string mappings_str;
        for (size_t i = 0; i < port.mappings.size(); ++i) {
          if (i > 0) {
            mappings_str += ", ";
          }
          mappings_str += fmt::format("{}:{}",
                                      port.mappings[i].internal_operator
                                          ? port.mappings[i].internal_operator->name()
                                          : "None",
                                      port.mappings[i].internal_port_name);
        }
        return fmt::format("<InterfacePort: mappings=[{}], is_input={}, type={}>",
                           mappings_str,
                           port.is_input,
                           port.port_type == InterfacePort::PortType::kData ? "DATA" : "EXECUTION");
      });

  // note: added py::dynamic_attr() to allow dynamically adding attributes in a Python subclass
  //       added std::shared_ptr<Subgraph> to allow the custom holder type to be used
  py::class_<Subgraph, PySubgraph, std::shared_ptr<Subgraph>>(
      m, "Subgraph", py::dynamic_attr(), doc::Subgraph::doc_Subgraph)
      .def(py::init<py::object, Fragment*, const std::string&, const std::string&>(),
           "subgraph"_a,
           "fragment"_a,
           "name"_a,
           "config"_a = "",
           doc::Subgraph::doc_Subgraph)
      .def(py::init([](py::object subgraph,
                       std::shared_ptr<Subgraph>
                           parent_subgraph,
                       const std::string& name,
                       const std::string& config_file) {
             // Extract the fragment from the parent subgraph
             Fragment* fragment = parent_subgraph->fragment();

             // Apply qualified naming using the parent subgraph's instance name
             std::string qualified_name = parent_subgraph->get_qualified_name(name, "subgraph");

             // Create the PySubgraph in the parent Subgraph's fragment
             auto py_subgraph =
                 std::make_shared<PySubgraph>(subgraph, fragment, qualified_name, config_file);
             return py_subgraph;
           }),
           "subgraph"_a,
           "parent_subgraph"_a,
           "name"_a,
           "config"_a = "")
      .def_property_readonly("name", &Subgraph::name, doc::Subgraph::doc_name)
      // Configuration methods (getters only - setters are protected, config must be passed to
      // constructor)
      .def("config", &Subgraph::config_shared, doc::Subgraph::doc_config_kwargs)
      .def("config_keys", &Subgraph::config_keys, doc::Subgraph::doc_config_keys)
      .def(
          "from_config",
          [](Subgraph& subgraph, const std::string& key) {
            ArgList arg_list = subgraph.from_config(key);
            if (arg_list.size() == 1) {
              return py::cast(arg_list.args()[0]);
            }
            return py::cast(arg_list);
          },
          "key"_a,
          doc::Subgraph::doc_from_config)
      .def(
          "kwargs",
          [](Subgraph& subgraph, const std::string& key) {
            ArgList arg_list = subgraph.from_config(key);
            return arglist_to_kwargs(arg_list);
          },
          "key"_a,
          doc::Subgraph::doc_kwargs)
      // Note: `set_dynamic_flows` is not added here, but as a Python method to dispatch to
      // `Fragment.set_dynamic_flows` which will handle the operator lifetime.
      .def("add_operator", &Subgraph::add_operator, "op"_a, doc::Subgraph::doc_add_operator)
      .def("add_subgraph", &Subgraph::add_subgraph, "subgraph"_a, doc::Subgraph::doc_add_subgraph)
      .def("add_data_logger",
           &Subgraph::add_data_logger,
           "logger"_a,
           doc::Subgraph::doc_add_data_logger)
      // Note: add_flow methods are implemented in Python via delegation to self.fragment.add_flow
      // Interface port methods - Operator overloads
      .def(
          "add_input_interface_port",
          py::overload_cast<const std::string&,
                            const std::shared_ptr<Operator>&,
                            const std::optional<std::string>&>(&Subgraph::add_input_interface_port),
          "external_name"_a,
          "internal_op"_a,
          "internal_port"_a = py::none(),
          doc::Subgraph::doc_add_input_interface_port)
      .def("add_output_interface_port",
           py::overload_cast<const std::string&,
                             const std::shared_ptr<Operator>&,
                             const std::optional<std::string>&>(
               &Subgraph::add_output_interface_port),
           "external_name"_a,
           "internal_op"_a,
           "internal_port"_a = py::none(),
           doc::Subgraph::doc_add_output_interface_port)
      .def("add_interface_port",
           py::overload_cast<const std::string&,
                             const std::shared_ptr<Operator>&,
                             const std::optional<std::string>&,
                             std::optional<bool>>(&Subgraph::add_interface_port),
           "external_name"_a,
           "internal_op"_a,
           "internal_port"_a = py::none(),
           "is_input"_a = py::none(),
           doc::Subgraph::doc_add_interface_port)
      // Interface port methods - Subgraph overloads (must omit docstring here and use
      // use a common one for both overloads).
      .def(
          "add_input_interface_port",
          py::overload_cast<const std::string&,
                            const std::shared_ptr<Subgraph>&,
                            const std::optional<std::string>&>(&Subgraph::add_input_interface_port),
          "external_name"_a,
          "internal_subgraph"_a,
          "internal_interface_port"_a = py::none())
      .def("add_output_interface_port",
           py::overload_cast<const std::string&,
                             const std::shared_ptr<Subgraph>&,
                             const std::optional<std::string>&>(
               &Subgraph::add_output_interface_port),
           "external_name"_a,
           "internal_subgraph"_a,
           "internal_interface_port"_a = py::none())
      .def("add_interface_port",
           py::overload_cast<const std::string&,
                             const std::shared_ptr<Subgraph>&,
                             const std::optional<std::string>&,
                             std::optional<bool>>(&Subgraph::add_interface_port),
           "external_name"_a,
           "internal_subgraph"_a,
           "internal_interface_port"_a = py::none(),
           "is_input"_a = py::none())
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
                             const std::optional<std::string>&>(
               &Subgraph::add_input_exec_interface_port),
           "external_name"_a,
           "internal_subgraph"_a,
           "internal_interface_port"_a = py::none())
      .def("add_output_exec_interface_port",
           py::overload_cast<const std::string&,
                             const std::shared_ptr<Subgraph>&,
                             const std::optional<std::string>&>(
               &Subgraph::add_output_exec_interface_port),
           "external_name"_a,
           "internal_subgraph"_a,
           "internal_interface_port"_a = py::none())
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
      .def("interface_ports",
           &Subgraph::interface_ports,
           py::return_value_policy::reference_internal,
           doc::Subgraph::doc_interface_ports)
      .def("exec_interface_ports",
           &Subgraph::exec_interface_ports,
           py::return_value_policy::reference_internal,
           doc::Subgraph::doc_exec_interface_ports)
      .def("get_interface_operator_port",
           &Subgraph::get_interface_operator_port,
           "port_name"_a,
           doc::Subgraph::doc_get_interface_operator_port)
      .def("get_exec_interface_operator_port",
           &Subgraph::get_exec_interface_operator_port,
           "port_name"_a,
           doc::Subgraph::doc_get_exec_interface_operator_port)
      .def("operators", &Subgraph::operators, doc::Subgraph::doc_operators)
      .def_property_readonly("nested_subgraphs",
                             &Subgraph::nested_subgraphs,
                             doc::Subgraph::doc_nested_subgraphs,
                             py::return_value_policy::reference_internal)
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
