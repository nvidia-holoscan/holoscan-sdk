/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "fragment.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <set>
#include <string>
#include <utility>

#include "application_pydoc.hpp"
#include "fragment_pydoc.hpp"
#include "holoscan/core/application.hpp"
#include "holoscan/core/config.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/network_context.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/resources/gxf/system_resources.hpp"
#include "holoscan/core/scheduler.hpp"
#include "kwarg_handling.hpp"

using pybind11::literals::operator""_a;  // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

void init_fragment(py::module_& m) {
  py::class_<Config, std::shared_ptr<Config>>(m, "Config", doc::Config::doc_Config)
      .def(py::init<const std::string&, const std::string&>(),
           "config_file"_a,
           "prefix"_a = "",
           doc::Config::doc_Config_kwargs)
      .def_property_readonly("config_file", &Config::config_file, doc::Config::doc_config_file)
      .def_property_readonly("prefix", &Config::prefix, doc::Config::doc_prefix);

  // note: added py::dynamic_attr() to allow dynamically adding attributes in a Python subclass
  //       added std::shared_ptr<Fragment> to allow the custom holder type to be used
  //         (see https://github.com/pybind/pybind11/issues/956)
  py::class_<Fragment, PyFragment, std::shared_ptr<Fragment>>(
      m, "Fragment", py::dynamic_attr(), doc::Fragment::doc_Fragment)
      .def(py::init<py::object>(), doc::Fragment::doc_Fragment)
      // notation for this name setter is a bit tricky (couldn't seem to do it with overload_cast)
      .def_property(
          "name",
          py::overload_cast<>(&Fragment::name, py::const_),
          [](Fragment& f, const std::string& name) -> Fragment& { return f.name(name); },
          doc::Fragment::doc_name)
      .def_property("application",
                    py::overload_cast<>(&Fragment::application, py::const_),
                    py::overload_cast<Application*>(&Fragment::application),
                    doc::Fragment::doc_application)
      // sphinx API doc build complains if more than one config
      // method has a docstring specified. For now just set the docstring for the
      // first overload only and document the variants in its docstring.
      .def("config",
           py::overload_cast<const std::string&, const std::string&>(&Fragment::config),
           "config_file"_a,
           "prefix"_a = "",
           doc::Fragment::doc_config_kwargs)
      .def("config", py::overload_cast<std::shared_ptr<Config>&>(&Fragment::config))
      .def("config", &Fragment::config_shared)
      .def("config_keys", &Fragment::config_keys, doc::Fragment::doc_config_keys)
      .def_property_readonly("graph", &Fragment::graph_shared, doc::Fragment::doc_graph)
      .def_property_readonly("executor", &Fragment::executor_shared, doc::Fragment::doc_executor)
      .def(
          "from_config",
          [](Fragment& fragment, const std::string& key) {
            ArgList arg_list = fragment.from_config(key);
            if (arg_list.size() == 1) { return py::cast(arg_list.args()[0]); }
            return py::cast(arg_list);
          },
          "key"_a,
          doc::Fragment::doc_from_config)
      .def(
          "kwargs",
          [](Fragment& fragment, const std::string& key) {
            ArgList arg_list = fragment.from_config(key);
            return arglist_to_kwargs(arg_list);
          },
          "key"_a,
          doc::Fragment::doc_kwargs)
      .def("add_operator",
           &Fragment::add_operator,
           "op"_a,
           doc::Fragment::doc_add_operator)  // note: virtual function
      // TODO(unknown): sphinx API doc build complains if more than one overloaded add_flow method
      // has a docstring specified. For now using the docstring defined for 3-argument
      // Operator-based version and describing the other variants in the Notes section.
      .def(  // note: virtual function
          "add_flow",
          py::overload_cast<const std::shared_ptr<Operator>&, const std::shared_ptr<Operator>&>(
              &Fragment::add_flow),
          "upstream_op"_a,
          "downstream_op"_a)
      .def(  // note: virtual function
          "add_flow",
          py::overload_cast<const std::shared_ptr<Operator>&,
                            const std::shared_ptr<Operator>&,
                            std::set<std::pair<std::string, std::string>>>(&Fragment::add_flow),
          "upstream_op"_a,
          "downstream_op"_a,
          "port_pairs"_a,
          doc::Fragment::doc_add_flow_pair)
      .def("compose", &Fragment::compose, doc::Fragment::doc_compose)  // note: virtual function
      .def("scheduler",
           py::overload_cast<const std::shared_ptr<Scheduler>&>(&Fragment::scheduler),
           "scheduler"_a,
           doc::Fragment::doc_scheduler_kwargs)
      .def("scheduler", py::overload_cast<>(&Fragment::scheduler), doc::Fragment::doc_scheduler)
      .def("network_context",
           py::overload_cast<const std::shared_ptr<NetworkContext>&>(&Fragment::network_context),
           "network_context"_a,
           doc::Fragment::doc_network_context_kwargs)
      .def("network_context",
           py::overload_cast<>(&Fragment::network_context),
           doc::Fragment::doc_network_context)
      .def("track",
           &Fragment::track,
           "num_start_messages_to_skip"_a = kDefaultNumStartMessagesToSkip,
           "num_last_messages_to_discard"_a = kDefaultNumLastMessagesToDiscard,
           "latency_threshold"_a = kDefaultLatencyThreshold,
           "is_limited_tracking"_a = false,
           doc::Fragment::doc_track,
           py::return_value_policy::reference_internal)
      .def_property("is_metadata_enabled",
                    py::overload_cast<>(&Fragment::is_metadata_enabled, py::const_),
                    py::overload_cast<bool>(&Fragment::is_metadata_enabled),
                    doc::Fragment::doc_is_metadata_enabled)
      .def("enable_metadata",
           &Fragment::enable_metadata,
           "enabled"_a,
           doc::Fragment::doc_enable_metadata)
      .def_property("metadata_policy",
                    py::overload_cast<>(&Fragment::metadata_policy, py::const_),
                    py::overload_cast<MetadataPolicy>(&Fragment::metadata_policy),
                    doc::Fragment::doc_metadata_policy)
      .def("make_thread_pool",
           &Fragment::make_thread_pool,
           "name"_a,
           "initialize_size"_a = 1,
           doc::Fragment::doc_make_thread_pool)
      .def("run",
           &Fragment::run,
           doc::Fragment::doc_run,
           py::call_guard<py::gil_scoped_release>())  // note: virtual function/should release GIL
      .def("start_op",
           &Fragment::start_op,
           doc::Fragment::doc_start_op,
           py::return_value_policy::reference_internal)
      .def(
          "set_dynamic_flows",
          [](Fragment& fragment, const std::shared_ptr<Operator>& op, py::function func) {
            fragment.set_dynamic_flows(
                op, [func = std::move(func)](const std::shared_ptr<Operator>& op_) {
                  // Acquire GIL before calling into Python code
                  py::gil_scoped_acquire gil;
                  func(op_);
                });
          },
          "op"_a,
          "dynamic_flow_func"_a,
          doc::Fragment::doc_set_dynamic_flows,
          py::keep_alive<1, 2>())  // keep op alive as long as the Fragment exists
      .def(
          "__repr__",
          [](const py::object& obj) {
            // use py::object and obj.cast to avoid a segfault if object has not been initialized
            auto fragment = obj.cast<std::shared_ptr<Fragment>>();
            if (fragment) { return fmt::format("<holoscan.Fragment: name:{}>", fragment->name()); }
            return std::string("<Fragment: None>");
          },
          R"doc(Return repr(self).)doc");
}

PyFragment::PyFragment(const py::object& op) {
  py::gil_scoped_acquire scope_guard;
  py_compose_ = py::getattr(op, "compose");
}

void PyFragment::add_operator(const std::shared_ptr<Operator>& op) {
  /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
  PYBIND11_OVERRIDE(void, Fragment, add_operator, op);
}

void PyFragment::add_flow(const std::shared_ptr<Operator>& upstream_op,
                          const std::shared_ptr<Operator>& downstream_op) {
  /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
  PYBIND11_OVERRIDE(void, Fragment, add_flow, upstream_op, downstream_op);
}

void PyFragment::add_flow(const std::shared_ptr<Operator>& upstream_op,
                          const std::shared_ptr<Operator>& downstream_op,
                          std::set<std::pair<std::string, std::string>> io_map) {
  /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
  PYBIND11_OVERRIDE(void, Fragment, add_flow, upstream_op, downstream_op, io_map);
}

void PyFragment::compose() {
  /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
  // PYBIND11_OVERRIDE(void, Fragment, compose);

  // PYBIND11_doesn't work when Fragment object is created during Application::compose().
  // So we take the py::object from the constructor and call it here.
  py::gil_scoped_acquire scope_guard;
  py_compose_.operator()();
}

void PyFragment::run() {
  /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
  PYBIND11_OVERRIDE(void, Fragment, run);
}

}  // namespace holoscan
