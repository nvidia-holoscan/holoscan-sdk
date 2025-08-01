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

#include <fmt/format.h>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

#include "application.hpp"
#include "application_pydoc.hpp"
#include "fragment_pydoc.hpp"
#include "holoscan/core/application.hpp"
#include "holoscan/core/config.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/network_context.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/resource.hpp"
#include "holoscan/core/resources/gxf/system_resources.hpp"
#include "holoscan/core/scheduler.hpp"
#include "holoscan/logger/logger.hpp"
#include "kwarg_handling.hpp"
#include "operator.hpp"

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
            if (arg_list.size() == 1) {
              return py::cast(arg_list.args()[0]);
            }
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
          "port_pairs"_a)
      .def(  // note: virtual function
          "add_flow",
          py::overload_cast<const std::shared_ptr<Operator>&,
                            const std::shared_ptr<Operator>&,
                            IOSpec::ConnectorType>(&Fragment::add_flow),
          "upstream_op"_a,
          "downstream_op"_a,
          "connector_type"_a)
      .def(  // note: virtual function
          "add_flow",
          py::overload_cast<const std::shared_ptr<Operator>&,
                            const std::shared_ptr<Operator>&,
                            std::set<std::pair<std::string, std::string>>,
                            IOSpec::ConnectorType>(&Fragment::add_flow),
          "upstream_op"_a,
          "downstream_op"_a,
          "port_pairs"_a,
          "connector_type"_a,
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
      .def("add_data_logger",
           &Fragment::add_data_logger,
           "logger"_a,
           doc::Fragment::doc_add_data_logger)
      .def_property_readonly("data_loggers",
                             &Fragment::data_loggers,
                             doc::Fragment::doc_data_loggers,
                             py::return_value_policy::reference_internal)
      .def("make_thread_pool",
           &Fragment::make_thread_pool,
           "name"_a,
           "initialize_size"_a = 1,
           doc::Fragment::doc_make_thread_pool)
      .def("run",
           &Fragment::run,
           doc::Fragment::doc_run,
           py::call_guard<py::gil_scoped_release>())  // note: virtual function/should release GIL
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
      .def("stop_execution",
           &Fragment::stop_execution,
           "op_name"_a = "",
           doc::Fragment::doc_stop_execution)
      // Service registry methods
      .def(
          "register_service",
          [](py::object fragment_obj, py::object service_obj, const std::string& id) {
            // Helper function to generate a unique ID for Python services
            auto generate_python_service_id = [](py::object obj, const std::string& base_id) {
              if (!base_id.empty()) {
                return base_id;  // Use provided ID as-is
              }
              // Generate unique ID based on Python type's fully qualified name
              try {
                auto type_obj = py::type::of(obj);
                auto module = type_obj.attr("__module__").cast<std::string>();
                auto name = type_obj.attr("__qualname__").cast<std::string>();
                return fmt::format("__py__{}.{}", module, name);
              } catch (const py::error_already_set&) {
                // Fallback to a generic unique identifier
                return fmt::format("__py__service_{}", reinterpret_cast<uintptr_t>(obj.ptr()));
              }
            };

            // Get the Fragment from the Python object
            auto fragment = fragment_obj.cast<std::shared_ptr<Fragment>>();
            if (!fragment) {
              throw py::type_error("Invalid fragment object.");
            }

            // Store the original Python object in PyFragment's registry if it's a PyFragment
            std::string effective_id = generate_python_service_id(service_obj, id);

            // Check if this is a PyFragment or PyApplication and if the service is in the Python
            // registry
            auto py_fragment = std::dynamic_pointer_cast<PyFragment>(fragment);
            auto py_app = std::dynamic_pointer_cast<PyApplication>(fragment);

            // For storing in Python registry, we need to determine the actual storage key
            std::string storage_id(effective_id);
            try {
              auto fs_ptr = service_obj.cast<std::shared_ptr<FragmentService>>();
              if (auto resource_ptr = fs_ptr->resource()) {
                storage_id = resource_ptr->name();
              }
            } catch (const py::cast_error&) {
              try {
                auto resource_ptr = service_obj.cast<std::shared_ptr<Resource>>();
                storage_id = resource_ptr->name();
              } catch (const py::cast_error&) {
                // Not a resource or a service with a resource, use effective_id
              }
            } catch (...) {
              // resource() might throw an exception for services that don't implement it
            }

            if (py_fragment) {
              py_fragment->set_python_service(storage_id, service_obj);
            } else if (py_app) {
              py_app->set_python_service(storage_id, service_obj);
            } else {
              // For Python fragments that don't inherit from PyFragment, use Python-level registry
              try {
                if (py::hasattr(fragment_obj, "_python_service_registry")) {
                  auto registry = fragment_obj.attr("_python_service_registry");
                  registry.attr("__setitem__")(storage_id, service_obj);
                }
              } catch (const py::error_already_set&) {
                HOLOSCAN_LOG_DEBUG(
                    "Unable to register Python fragment service with id '{}' in custom registry. "
                    "Skipping.",
                    storage_id);
              }
            }

            // Helper lambda to clean up stored Python object on failure
            auto cleanup_on_failure = [&]() {
              if (py_fragment) {
                py_fragment->clear_python_service(storage_id);
              } else if (py_app) {
                py_app->clear_python_service(storage_id);
              } else {
                try {
                  if (py::hasattr(fragment_obj, "_python_service_registry")) {
                    auto registry = fragment_obj.attr("_python_service_registry");
                    if (py::hasattr(registry, "__delitem__")) {
                      registry.attr("__delitem__")(storage_id);
                    }
                  }
                } catch (const py::error_already_set&) {
                  HOLOSCAN_LOG_DEBUG(
                      "Unable to clear Python fragment service with id '{}'. Skipping.",
                      storage_id);
                }
              }
            };

            // Special handling for Python-defined services only
            if (storage_id.find("__py__") == 0) {
              try {
                // For Python services, always register as DefaultFragmentService to ensure proper
                // lookup
                auto fragment_service = service_obj.cast<std::shared_ptr<DefaultFragmentService>>();

                // Check if this DefaultFragmentService wraps a Resource
                std::string service_id = storage_id;
                try {
                  if (fragment_service->resource()) {
                    // If it wraps a resource, use empty ID (resource name will be used)
                    service_id = "";
                  }
                } catch (...) {
                  // If resource() throws, use the storage_id
                }

                if (!fragment->register_service(fragment_service, service_id)) {
                  cleanup_on_failure();
                  HOLOSCAN_LOG_ERROR("Failed to register Python fragment service with id '{}'",
                                     service_id);
                  throw std::runtime_error("Failed to register Python fragment service.");
                }
                return;
              } catch (const py::cast_error&) {
                // Python service doesn't inherit from DefaultFragmentService
                // Fall through to try other types
              }
            }

            // For C++ services, try the exact type casts in order to preserve the actual type
            // This is important for C++ code that uses service<MyService>() to retrieve services

            // Try FragmentService (for services that implement the interface)
            try {
              auto service = service_obj.cast<std::shared_ptr<FragmentService>>();

              // Check if this is a DefaultFragmentService wrapping a Resource
              bool has_resource = false;
              try {
                if (service->resource()) {
                  has_resource = true;
                }
              } catch (...) {
                // If resource() throws, it means there's no resource
                has_resource = false;
              }

              // If the service has a resource, use empty ID (resource name will be used)
              std::string service_id = has_resource ? "" : storage_id;

              if (!fragment->register_service(service, service_id)) {
                cleanup_on_failure();
                HOLOSCAN_LOG_ERROR("Failed to register fragment service with id '{}'", service_id);
                throw std::runtime_error("Failed to register fragment service.");
              }
              return;
            } catch (const py::cast_error&) {
              // Not an FragmentService, continue
            }

            // Try Resource
            try {
              auto resource = service_obj.cast<std::shared_ptr<Resource>>();

              // For resources, the ID must be empty as the resource's name is used as ID
              std::string resource_id = "";
              if (!storage_id.empty() && storage_id.find("__py__") != 0) {
                // If a non-empty ID was provided for a resource (and it's not a Python
                // auto-generated ID), log a debug message and use empty string instead
                HOLOSCAN_LOG_DEBUG(
                    "Resource '{}' registration: ID parameter '{}' will be ignored. "
                    "Resource name will be used as ID.",
                    resource->name(),
                    storage_id);
                resource_id = "";
              } else if (storage_id.find("__py__") == 0) {
                // For Python resources with auto-generated IDs, we still use empty string
                // but keep the storage_id for Python registry storage
                resource_id = "";
              }

              if (!fragment->register_service(resource, resource_id)) {
                cleanup_on_failure();
                HOLOSCAN_LOG_ERROR("Failed to register fragment resource '{}'", resource->name());
                throw std::runtime_error("Failed to register fragment resource.");
              }
              return;
            } catch (const py::cast_error&) {
              // Not a Resource
            }

            throw py::type_error(
                "Service must be a holoscan.Resource or holoscan.FragmentService instance.");
          },
          "service"_a,
          "id"_a = "",
          doc::Fragment::doc_register_service)
      .def(
          "service",
          [](py::object fragment_obj, py::type service_type, const std::string& id) -> py::object {
            // Helper function to generate the same unique ID used during registration
            auto generate_python_service_id = [&service_type](const std::string& base_id) {
              if (!base_id.empty()) {
                return base_id;  // Use provided ID as-is
              }
              // Generate unique ID based on Python type's fully qualified name
              try {
                auto module = service_type.attr("__module__").cast<std::string>();
                auto name = service_type.attr("__qualname__").cast<std::string>();
                return fmt::format("__py__{}.{}", module, name);
              } catch (const py::error_already_set&) {
                // Cannot generate type-based ID, just use empty string
                return std::string("");
              }
            };

            std::string effective_id = generate_python_service_id(id);

            // Get the Fragment from the Python object
            auto fragment = fragment_obj.cast<std::shared_ptr<Fragment>>();
            if (!fragment) {
              throw py::type_error("Invalid fragment object.");
            }

            // Check if this is a PyFragment or PyApplication and if the service is in the Python
            // registry
            auto py_fragment = std::dynamic_pointer_cast<PyFragment>(fragment);
            auto py_app = std::dynamic_pointer_cast<PyApplication>(fragment);

            // Helper lambda to check Python service registry
            auto check_python_registry = [](auto py_obj,
                                            const std::string& check_id) -> py::object {
              if (py_obj) {
                auto py_service = py_obj->get_python_service(check_id);
                if (!py_service.is_none()) {
                  return py_service;
                }
              }
              return py::none();
            };

            py::object py_service = py::none();
            // If looking up by resource name, first check with that name
            if (!id.empty()) {
              py_service = check_python_registry(py_fragment, id);
              if (py_service.is_none()) {
                py_service = check_python_registry(py_app, id);
              }
            }
            if (py_service.is_none() && effective_id != id) {
              // Otherwise check with the generated __py__ id
              py_service = check_python_registry(py_fragment, effective_id);
              if (py_service.is_none()) {
                py_service = check_python_registry(py_app, effective_id);
              }
            }

            // Check if we found anything
            if (!py_service.is_none()) {
              // We found something. Now check if it's the right type or needs unwrapping.
              bool is_req_type_resource = false;
              try {
                auto resource_py_type = py::type::of<Resource>();
                auto mro = service_type.attr("__mro__").cast<py::tuple>();
                for (const auto& base : mro) {
                  if (base.is(resource_py_type)) {
                    is_req_type_resource = true;
                    break;
                  }
                }
              } catch (const py::error_already_set& e) {
                throw py::type_error(
                    fmt::format("Unable to check MRO for service type: {}", e.what()));
              }

              if (is_req_type_resource) {
                // Requesting a resource type
                if (py::isinstance(py_service, service_type)) {
                  return py_service;
                }  // Direct hit

                // Try unwrapping
                try {
                  auto wrapper = py_service.cast<std::shared_ptr<FragmentService>>();
                  if (auto resource_sptr = wrapper->resource()) {
                    py::object py_underlying_resource = py::cast(resource_sptr);
                    if (py::isinstance(py_underlying_resource, service_type)) {
                      return py_underlying_resource;
                    }
                  }
                } catch (const py::cast_error&) {
                }
              } else {
                // Requesting a service (non-resource) type
                if (py::isinstance(py_service, service_type)) {
                  return py_service;
                }
              }
            }

            // Fallback to C++ lookup
            const std::type_info* lookup_type = nullptr;
            const auto* cpp_type = py::detail::get_type_info((PyTypeObject*)service_type.ptr());
            if (!cpp_type) {
              throw py::type_error("Unable to get C++ type info from Python type.");
            }
            lookup_type = cpp_type->cpptype;

            auto base_service = fragment->get_service_by_type_info(*lookup_type, id);
            if (!base_service) {
              return py::none();
            }

            bool is_resource = false;
            try {
              auto resource_py_type = py::type::of<Resource>();
              auto mro = service_type.attr("__mro__").cast<py::tuple>();
              for (const auto& base : mro) {
                if (base.is(resource_py_type)) {
                  is_resource = true;
                  break;
                }
              }
            } catch (const py::error_already_set& e) {
              throw py::type_error(
                  fmt::format("Unable to check MRO for service type: {}", e.what()));
            }

            if (is_resource) {
              auto resource = base_service->resource();
              if (!resource) {
                return py::none();
              }
              return py::cast(resource);
            } else {
              return py::cast(base_service);
            }
          },
          "service_type"_a,
          "id"_a = "",
          doc::Fragment::doc_service)
      .def(
          "__repr__",
          [](const py::object& obj) {
            // use py::object and obj.cast to avoid a segfault if object has not been initialized
            auto fragment = obj.cast<std::shared_ptr<Fragment>>();
            if (fragment) {
              return fmt::format("<holoscan.Fragment: name:{}>", fragment->name());
            }
            return std::string("<Fragment: None>");
          },
          R"doc(Return repr(self).)doc");
}

PyFragment::PyFragment(const py::object& op) {
  py::gil_scoped_acquire scope_guard;
  py_compose_ = py::getattr(op, "compose");
}

PyFragment::~PyFragment() {
  try {
    py::gil_scoped_acquire scope_guard;
    // We must not call the parent class destructor here because Pybind11 already handles this
    // through its holder mechanism. Specifically, Pybind11 calls destructors explicitly via
    // `v_h.holder<holder_type>().~holder_type();` in the `class_::dealloc()` method in pybind11.h.
    // Fragment::~Fragment(); // DO NOT CALL THIS - would cause double destruction

    // Clear the operator registry
    python_operator_registry_.clear();
    // Clear the service registry
    python_service_registry_.clear();
  } catch (const std::exception& e) {
    // Silently handle any exceptions during cleanup
    try {
      HOLOSCAN_LOG_ERROR("PyFragment destructor failed with {}", e.what());
    } catch (...) {
    }
  }
}

void PyFragment::add_operator(const std::shared_ptr<Operator>& op) {
  {
    pybind11::gil_scoped_acquire gil;
    // Store a reference to the Python operator in PyFragment's internal registry
    // to maintain the reference to the Python operator in case it's used by the
    // data flow tracker after `run()` or `run_async()` is called.
    // See the explanation in the `PyOperator::release_internal_resources()` method for details.
    python_operator_registry_[op.get()] = py::cast(op);
  }

  /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
  PYBIND11_OVERRIDE(void, Fragment, add_operator, op);
}

void PyFragment::add_flow(const std::shared_ptr<Operator>& upstream_op,
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

  /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
  PYBIND11_OVERRIDE(void, Fragment, add_flow, upstream_op, downstream_op);
}

void PyFragment::add_flow(const std::shared_ptr<Operator>& upstream_op,
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
  PYBIND11_OVERRIDE(void, Fragment, add_flow, upstream_op, downstream_op, io_map);
}

void PyFragment::add_flow(const std::shared_ptr<Operator>& upstream_op,
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
  PYBIND11_OVERRIDE(void, Fragment, add_flow, upstream_op, downstream_op, connector_type);
}

void PyFragment::add_flow(const std::shared_ptr<Operator>& upstream_op,
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
  PYBIND11_OVERRIDE(void, Fragment, add_flow, upstream_op, downstream_op, io_map, connector_type);
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
  {
    pybind11::gil_scoped_acquire gil;
    // Release all the Python operator references before run() is executed.
    // See the explanation in the `PyOperator::release_internal_resources()` method for details.
    python_operator_registry_.clear();
  }

  /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
  PYBIND11_OVERRIDE(void, Fragment, run);
}

void PyFragment::reset_state() {
  // Ensure the Python Global Interpreter Lock (GIL) is held during execution.
  py::gil_scoped_acquire gil;
  Fragment::reset_state();

  // Clear the operator registry
  python_operator_registry_.clear();
  // Clear the service registry
  python_service_registry_.clear();
}

bool PyFragment::register_service_from(Fragment* application, std::string_view id) {
  if (!application) {
    throw std::invalid_argument("application must not be nullptr");
  }

  py::gil_scoped_acquire gil;
  auto result = Fragment::register_service_from(application, id);
  if (!result) {
    throw py::value_error(fmt::format(
        "Failed to register service (id: '{}') from application: '{}'", id, application->name()));
  }
  auto* py_application = dynamic_cast<PyApplication*>(application);
  if (!py_application) {
    throw py::value_error(
        fmt::format("Unable to cast application to PyApplication: '{}'", application->name()));
  }

  // Copy the service from the source application to the current fragment
  py::object py_service = py_application->get_python_service(std::string(id));
  if (!py_service.is_none()) {
    std::string service_id = std::string(id);
    python_service_registry_[service_id] = py_service;

    // Lock the mutex before modifying fragment_services_by_key_
    std::unique_lock<std::shared_mutex> lock(fragment_service_registry_mutex_);

    // Check if the Python object is actually a FragmentService before casting
    try {
      auto service = py_service.cast<std::shared_ptr<FragmentService>>();

      // Use DefaultFragmentService for all Python services to ensure consistent lookup
      ServiceKey key{typeid(DefaultFragmentService), service_id};
      fragment_services_by_key_[key] = service;
    } catch (const py::cast_error&) {
      // If it's not a FragmentService, it might be a Resource or other type
      // The base class register_service_from should have already handled it
      HOLOSCAN_LOG_DEBUG("Python service '{}' is not a FragmentService type", service_id);
    }
  }
  return true;
}

py::object PyFragment::get_python_service(const std::string& service_id) const {
  py::gil_scoped_acquire gil;
  auto it = python_service_registry_.find(service_id);
  if (it != python_service_registry_.end()) {
    return it->second;
  }
  return py::none();
}

void PyFragment::set_python_service(const std::string& service_id, py::object service) {
  py::gil_scoped_acquire gil;
  python_service_registry_[service_id] = std::move(service);
}

void PyFragment::clear_python_service(const std::string& service_id) {
  py::gil_scoped_acquire gil;
  python_service_registry_.erase(service_id);
}
}  // namespace holoscan
