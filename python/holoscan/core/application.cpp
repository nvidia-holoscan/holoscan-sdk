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

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "application_pydoc.hpp"
#include "fragment_pydoc.hpp"
#include "holoscan/core/application.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "tensor.hpp"

using pybind11::literals::operator""_a;  // NOLINT(misc-unused-using-decls)

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
      .def_property(
          "description",
          py::overload_cast<>(&Application::description),
          [](Application& app, const std::string& name) -> Application& {
            return app.description(name);
          },
          doc::Application::doc_description)
      .def_property(
          "version",
          py::overload_cast<>(&Application::version),
          [](Application& app, const std::string& name) -> Application& {
            return app.version(name);
          },
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
      // TODO(unknown): sphinx API doc build complains if more than one overloaded add_flow method
      // has a docstring specified. For now using the docstring defined for 3-argument
      // Operator-based version and describing the other variants in the Notes section.
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
      .def("compose_graph", &Application::compose_graph, doc::Application::doc_compose_graph)
      .def("run",
           &Application::run,
           doc::Application::doc_run,
           py::call_guard<py::gil_scoped_release>())  // note: virtual function/should release GIL
      .def(
          "track_distributed",
          // This version of `track_distributed differs from the C++ API only in return type, using
          //   std::unordered_map<std::string, std::reference_wrapper<DataFlowTracker>>
          // instead of
          //   std::unordered_map<std::string, DataFlowTracker*>
          // to use the trackers from Python.
          [](Application& app,
             uint64_t num_start_messages_to_skip,
             uint64_t num_last_messages_to_discard,
             int latency_threshold,
             bool is_limited_tracking)
              -> std::unordered_map<std::string, std::reference_wrapper<DataFlowTracker>> {
            auto tracker_pointers = app.track_distributed(num_start_messages_to_skip,
                                                          num_last_messages_to_discard,
                                                          latency_threshold,
                                                          is_limited_tracking);
            std::unordered_map<std::string, std::reference_wrapper<DataFlowTracker>> trackers;
            for (const auto& [name, tracker_ptr] : tracker_pointers) {
              trackers.emplace(name, std::ref(*tracker_ptr));
            }
            return trackers;
          },
          "num_start_messages_to_skip"_a = kDefaultNumStartMessagesToSkip,
          "num_last_messages_to_discard"_a = kDefaultNumLastMessagesToDiscard,
          "latency_threshold"_a = kDefaultLatencyThreshold,
          "is_limited_tracking"_a = false,
          // doc::Fragment::doc_track_distributed,
          py::return_value_policy::reference_internal)
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

py::list PyApplication::py_argv() {
  py::list argv;
  // In Python, `sys.argv` returns `['']` if there are no arguments (i.e., when just `python` is
  // called). We'll do the same here.
  if (argv_.empty()) {
    argv.append(py::cast("", py::return_value_policy::reference));
    return argv;
  }

  for (auto iter = std::next(argv_.begin()); iter != argv_.end(); ++iter) {
    argv.append(py::cast(*iter, py::return_value_policy::reference));
  }

  if (argv.empty()) { argv.append(py::cast("", py::return_value_policy::reference)); }
  return argv;
}

void PyApplication::add_operator(const std::shared_ptr<Operator>& op) {
  /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
  PYBIND11_OVERRIDE(void, Application, add_operator, op);
}

void PyApplication::add_flow(const std::shared_ptr<Operator>& upstream_op,
                             const std::shared_ptr<Operator>& downstream_op) {
  /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
  PYBIND11_OVERRIDE(void, Application, add_flow, upstream_op, downstream_op);
}

void PyApplication::add_flow(const std::shared_ptr<Operator>& upstream_op,
                             const std::shared_ptr<Operator>& downstream_op,
                             std::set<std::pair<std::string, std::string>> io_map) {
  /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
  PYBIND11_OVERRIDE(void, Application, add_flow, upstream_op, downstream_op, io_map);
}

void PyApplication::add_flow(const std::shared_ptr<Fragment>& upstream_frag,
                             const std::shared_ptr<Fragment>& downstream_frag,
                             std::set<std::pair<std::string, std::string>> port_pairs) {
  /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
  PYBIND11_OVERRIDE(void, Application, add_flow, upstream_frag, downstream_frag, port_pairs);
}

void PyApplication::compose() {
  /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
  PYBIND11_OVERRIDE(void, Application, compose);
}

void PyApplication::run() {
  // Debug log to show that the run() function is executed
  // (with the logging function pointer info to check if the logging function pointer address is
  // the same as the one set in the Python side).
  // This message is checked by the test_app_log_function in test_application_minimal.py.

  // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
  HOLOSCAN_LOG_DEBUG("Executing PyApplication::run()... (log_func_ptr=0x{:x})",
                     reinterpret_cast<uint64_t>(&nvidia::LoggingFunction));
  // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)

  // Create a deleter for DLManagedTensor objects so that they can be deleted in a separate thread
  // to avoid blocking the GXF runtime mutex.
  LazyDLManagedTensorDeleter deleter;

  // Get the trace and profile functions from sys
  {
    pybind11::gil_scoped_acquire gil;

    auto sys_module = py::module::import("sys");

    // Note that when cProfile is used, the profile_func_ is a cProfile.Profile object, not a
    // function. If the return value of getprofile() is not a function, we need to use the
    // existing c_profilefunc_ and c_profileobj_ instead of calling sys.setprofile() with
    // profile_func_.
    py_profile_func_ = sys_module.attr("getprofile")();
    py_trace_func_ = sys_module.attr("gettrace")();

    auto* py_thread_state = _PyThreadState_UncheckedGet();
    c_profilefunc_ = py_thread_state->c_profilefunc;
    c_profileobj_ = py_thread_state->c_profileobj;
    c_tracefunc_ = py_thread_state->c_tracefunc;
    c_traceobj_ = py_thread_state->c_traceobj;

#if PY_VERSION_HEX >= 0x030b0000  // >= Python 3.11.0
    // _PyInterpreterFrame*
    py_last_frame_ = py_thread_state->cframe->current_frame;
#else
    // PyFrameObject*
    py_last_frame_ = py_thread_state->frame;  // = PyEval_GetFrame();
#endif
  }

  /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
  PYBIND11_OVERRIDE(void, Application, run);
}

}  // namespace holoscan
