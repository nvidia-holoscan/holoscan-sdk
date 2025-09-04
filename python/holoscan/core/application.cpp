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

#include "application.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "application_pydoc.hpp"
#include "fragment.hpp"
#include "fragment_pydoc.hpp"
#include "holoscan/core/application.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "operator.hpp"
#include "tensor.hpp"

using pybind11::literals::operator""_a;  // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

void init_application(py::module_& m) {
  // note: added py::dynamic_attr() to allow dynamically adding attributes in a Python subclass
  //       added std::shared_ptr<Fragment> to allow the custom holder type to be used
  //         (see https://github.com/pybind/pybind11/issues/956)
  py::class_<Application, PyApplication, Fragment, std::shared_ptr<Application>>(
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
          "port_pairs"_a)
      .def(  // note: virtual function
          "add_flow",
          py::overload_cast<const std::shared_ptr<Operator>&,
                            const std::shared_ptr<Operator>&,
                            IOSpec::ConnectorType>(&Application::add_flow),
          "upstream_op"_a,
          "downstream_op"_a,
          "connector_type"_a)
      .def(  // note: virtual function
          "add_flow",
          py::overload_cast<const std::shared_ptr<Operator>&,
                            const std::shared_ptr<Operator>&,
                            std::set<std::pair<std::string, std::string>>,
                            IOSpec::ConnectorType>(&Application::add_flow),
          "upstream_op"_a,
          "downstream_op"_a,
          "port_pairs"_a,
          "connector_type"_a,
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
      .def_property("is_metadata_enabled",
                    py::overload_cast<>(&Application::is_metadata_enabled, py::const_),
                    py::overload_cast<bool>(&Application::is_metadata_enabled),
                    doc::Application::doc_is_metadata_enabled)
      .def("enable_metadata",
           &Application::enable_metadata,
           "enabled"_a,
           doc::Application::doc_enable_metadata)
      .def_property("metadata_policy",
                    py::overload_cast<>(&Application::metadata_policy, py::const_),
                    py::overload_cast<MetadataPolicy>(&Application::metadata_policy),
                    doc::Application::doc_metadata_policy)
      .def("add_data_logger",
           &Application::add_data_logger,
           "logger"_a,
           doc::Application::doc_add_data_logger)
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
            if (app) {
              return fmt::format("<holoscan.Application: name:{}>", app->name());
            }
            return std::string("<Application: None>");
          },
          R"doc(Return repr(self).)doc");
}

PyApplication::~PyApplication() {
  try {
    py::gil_scoped_acquire scope_guard;
    // We must not call the parent class destructor here because Pybind11 already handles this
    // through its holder mechanism. Specifically, Pybind11 calls destructors explicitly via
    // `v_h.holder<holder_type>().~holder_type();` in the `class_::dealloc()` method in pybind11.h.
    // Application::~Application(); // DO NOT CALL THIS - would cause double destruction

    // Clear the operator registry
    python_operator_registry_.clear();
    // Clear the service registry
    python_service_registry_.clear();

    // #if PY_VERSION_HEX >= 0x030D0000  // >= Python 3.13.0
    //     // decrement strong reference obtained via PyThreadState_GetFrame
    //     Py_XDECREF(py_last_frame_);
    // #endif
  } catch (const std::exception& e) {
    // Silently handle any exceptions during cleanup
    try {
      HOLOSCAN_LOG_ERROR("PyApplication destructor failed with {}", e.what());
    } catch (...) {
    }
  }
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

  if (argv.empty()) {
    argv.append(py::cast("", py::return_value_policy::reference));
  }
  return argv;
}

void PyApplication::add_operator(const std::shared_ptr<Operator>& op) {
  {
    pybind11::gil_scoped_acquire gil;
    // Store a reference to the Python operator in PyFragment's internal registry
    // to maintain the reference to the Python operator in case it's used by the
    // data flow tracker after `run()` or `run_async()` is called.
    // See the explanation in the `PyOperator::release_internal_resources()` method for details.
    python_operator_registry_[op.get()] = py::cast(op);
  }

  /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
  PYBIND11_OVERRIDE(void, Application, add_operator, op);
}

void PyApplication::add_flow(const std::shared_ptr<Operator>& upstream_op,
                             const std::shared_ptr<Operator>& downstream_op) {
  {
    pybind11::gil_scoped_acquire gil;
    // Store a reference to the Python operator in PyFragment's internal registry
    // to maintain the reference to the Python operator in case it's used by the
    // data flow tracker after `run()` or `run_async()` is called.
    // See the explanation in the `PyOperator::release_internal_resources()` method for details.
    python_operator_registry_[upstream_op.get()] = py::cast(upstream_op);
    python_operator_registry_[downstream_op.get()] = py::cast(downstream_op);
  }

  PYBIND11_OVERRIDE(void, Application, add_flow, upstream_op, downstream_op);
}

void PyApplication::add_flow(const std::shared_ptr<Operator>& upstream_op,
                             const std::shared_ptr<Operator>& downstream_op,
                             std::set<std::pair<std::string, std::string>> io_map) {
  {
    pybind11::gil_scoped_acquire gil;
    // Store a reference to the Python operator in PyFragment's internal registry
    // to maintain the reference to the Python operator in case it's used by the
    // data flow tracker after `run()` or `run_async()` is called.
    // See the explanation in the `PyOperator::release_internal_resources()` method for details.
    python_operator_registry_[upstream_op.get()] = py::cast(upstream_op);
    python_operator_registry_[downstream_op.get()] = py::cast(downstream_op);
  }

  /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
  PYBIND11_OVERRIDE(void, Application, add_flow, upstream_op, downstream_op, io_map);
}

void PyApplication::add_flow(const std::shared_ptr<Operator>& upstream_op,
                             const std::shared_ptr<Operator>& downstream_op,
                             IOSpec::ConnectorType connector_type) {
  {
    pybind11::gil_scoped_acquire gil;
    // Store a reference to the Python operator in PyFragment's internal registry
    // to maintain the reference to the Python operator in case it's used by the
    // data flow tracker after `run()` or `run_async()` is called.
    // See the explanation in the `PyOperator::release_internal_resources()` method for details.
    python_operator_registry_[upstream_op.get()] = py::cast(upstream_op);
    python_operator_registry_[downstream_op.get()] = py::cast(downstream_op);
  }

  /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
  PYBIND11_OVERRIDE(void, Application, add_flow, upstream_op, downstream_op, connector_type);
}

void PyApplication::add_flow(const std::shared_ptr<Operator>& upstream_op,
                             const std::shared_ptr<Operator>& downstream_op,
                             std::set<std::pair<std::string, std::string>> io_map,
                             IOSpec::ConnectorType connector_type) {
  {
    pybind11::gil_scoped_acquire gil;
    // Store a reference to the Python operator in PyFragment's internal registry
    // to maintain the reference to the Python operator in case it's used by the
    // data flow tracker after `run()` or `run_async()` is called.
    // See the explanation in the `PyOperator::release_internal_resources()` method for details.
    python_operator_registry_[upstream_op.get()] = py::cast(upstream_op);
    python_operator_registry_[downstream_op.get()] = py::cast(downstream_op);
  }

  /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
  PYBIND11_OVERRIDE(
      void, Application, add_flow, upstream_op, downstream_op, io_map, connector_type);
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

  {
    pybind11::gil_scoped_acquire gil;

    // Get the trace and profile functions from sys
    auto sys_module = py::module::import("sys");

    // Note that when cProfile is used, the profile_func_ is a cProfile.Profile object, not a
    // function. If the return value of getprofile() is not a function, we need to use the
    // existing c_profilefunc_ and c_profileobj_ instead of calling sys.setprofile() with
    // profile_func_.
    py_profile_func_ = sys_module.attr("getprofile")();
    py_trace_func_ = sys_module.attr("gettrace")();

#if PY_VERSION_HEX >= 0x030D0000  // >= Python 3.13.0
    // public API available in Python 3.13
    auto* py_thread_state = PyThreadState_GetUnchecked();
#else
    auto* py_thread_state = _PyThreadState_UncheckedGet();
#endif
    // Warning: these PyThreadState fields are part of CPython's private C API
    c_profilefunc_ = py_thread_state->c_profilefunc;
    c_profileobj_ = py_thread_state->c_profileobj;
    c_tracefunc_ = py_thread_state->c_tracefunc;
    c_traceobj_ = py_thread_state->c_traceobj;

#if PY_VERSION_HEX >= 0x030D0000  // >= Python 3.13.0
    // Note:
    // Python 3.12 implemented PEP-669: Low Impact Monitoring for CPython
    //   https://peps.python.org/pep-0669/
    // as sys.monitoring:
    //   https://docs.python.org/3/library/sys.monitoring.html#module-sys.monitoring)
    // Python 3.13 introduced a corresponding C-API:
    //   https://docs.python.org/3/c-api/monitoring.html

    // PyThreadState_GetFrame returns a strong reference so a Py_DECREF will be needed
    // py_last_frame_ = PyThreadState_GetFrame(py_thread_state);

    // PyEval_GetFrame returns a borrowed reference so Py_DECREF is not needed
    // The PyFrameObject* corresponds to the current thread.
    py_last_frame_ = PyEval_GetFrame();
#elif PY_VERSION_HEX >= 0x030B0000  // >= Python 3.11.0
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

void PyApplication::reset_state() {
  // Ensure the Python Global Interpreter Lock (GIL) is held during execution.
  py::gil_scoped_acquire gil;
  Application::reset_state();

  // Clear the operator registry
  python_operator_registry_.clear();
  // Clear the service registry
  python_service_registry_.clear();
}

void PyApplication::attach_services_to_fragment(const std::shared_ptr<Fragment>& fragment) {
  py::gil_scoped_acquire gil;
  auto py_fragment = std::dynamic_pointer_cast<PyFragment>(fragment);
  if (!py_fragment) {
    throw py::value_error(
        fmt::format("Failed to cast fragment to PyFragment: '{}'", fragment->name()));
  }

  std::unordered_set<std::string> registered_service_ids;
  for (const auto& [service_key, service] : fragment_services_by_key()) {
    if (registered_service_ids.find(service_key.id) == registered_service_ids.end()) {
      HOLOSCAN_LOG_DEBUG(
          "Registering service '{}' with fragment '{}'", service_key.id, fragment->name());
      // Register service from the application to the fragment
      py_fragment->register_service_from(this, service_key.id);
      registered_service_ids.insert(service_key.id);
    }
  }

  // Copy the service from python_service_registry_ to each fragment
  for (const auto& [service_key, service] : python_service_registry_) {
    py_fragment->register_service_from(this, service_key);

    ServiceKey key{typeid(DefaultFragmentService), service_key};
    auto py_service = service.cast<std::shared_ptr<FragmentService>>();

    fragment_services_by_key_[key] = std::move(py_service);
  }
}

py::object PyApplication::get_python_service(const std::string& service_id) const {
  py::gil_scoped_acquire gil;
  auto it = python_service_registry_.find(service_id);
  if (it != python_service_registry_.end()) {
    return it->second;
  }
  return py::none();
}

void PyApplication::set_python_service(const std::string& service_id, py::object service) {
  py::gil_scoped_acquire gil;

  try {
    // In order for C++ code to detect and invoke ServiceDriverEndpoint and ServiceWorkerEndpoint
    // methods, we must cast the Python service object to DistributedAppService (which inherits from
    // both interfaces and FragmentService) and store it in the fragment_services_by_key_ map. For
    // example, functions like 'AppDriver::handle_driver_start' access this map and attempt to cast
    // entries to 'distributed::ServiceDriverEndpoint'. Without casting the Python object to
    // std::shared_ptr<DistributedAppService>, the stored pointer cannot be dynamically cast to
    // distributed::ServiceDriverEndpoint or distributed::ServiceWorkerEndpoint.
    auto shared = service.cast<std::shared_ptr<DistributedAppService>>();
    ServiceKey key{typeid(DistributedAppService), service_id};
    fragment_services_by_key_[key] = shared;
    HOLOSCAN_LOG_DEBUG("Fragment service '{}' is inherits from DistributedAppService", service_id);
  } catch (const py::cast_error&) {
    HOLOSCAN_LOG_DEBUG("Fragment service '{}' does not inherits from DistributedAppService",
                       service_id);
  }

  python_service_registry_[service_id] = std::move(service);
}

void PyApplication::clear_python_service(const std::string& service_id) {
  py::gil_scoped_acquire gil;
  python_service_registry_.erase(service_id);
}

}  // namespace holoscan
