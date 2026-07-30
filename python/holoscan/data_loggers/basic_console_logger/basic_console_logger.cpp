/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for vector

#include <any>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <typeindex>
#include <variant>
#include <vector>

#include "./basic_console_logger.hpp"
#include "./pydoc.hpp"

#include <holoscan/core/arg.hpp>
#include <holoscan/core/component_spec.hpp>
#include <holoscan/core/component_traits.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/resource.hpp>
#include <holoscan/core/resources/async_data_logger.hpp>
#include <holoscan/core/resources/data_logger.hpp>
#include <holoscan/core/subgraph.hpp>
#include <holoscan/data_loggers/basic_console_logger/basic_console_logger.hpp>
#include <holoscan/data_loggers/basic_console_logger/gxf_console_logger.hpp>
#include <holoscan/data_loggers/basic_console_logger/simple_text_serializer.hpp>
#include "../../core/component_util.hpp"
#include "../../core/gil_guarded_pyobject.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan::data_loggers {

// Declaration of internal Python API for detecting interpreter finalization (Python < 3.13)
#if PY_VERSION_HEX < 0x030D0000
extern "C" int _Py_IsFinalizing(void);
#endif

namespace {

/**
 * @brief Check if the Python interpreter is finalizing (shutting down).
 *
 * During finalization, acquiring the GIL can cause deadlock if the main thread
 * holds the GIL while waiting for worker threads to join. This function allows
 * worker threads to detect finalization and skip Python object serialization.
 *
 * @return true if Python is finalizing, false otherwise
 */
inline bool is_python_finalizing() {
#if PY_VERSION_HEX >= 0x030D0000  // Python 3.13+: use public API
  return Py_IsFinalizing() != 0;
#else  // Python < 3.13: use internal API (available since 3.6)
  return _Py_IsFinalizing() != 0;
#endif
}

}  // namespace

/* Trampoline class for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the resource.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the resource's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_resource<ResourceT>
 */

PySimpleTextSerializer::PySimpleTextSerializer(
    const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph, int64_t max_elements,
    int64_t max_metadata_items, bool log_video_buffer_content, bool log_python_object_contents,
    const std::string& name)
    : SimpleTextSerializer(ArgList{Arg{"max_elements", max_elements},
                                   Arg{"max_metadata_items", max_metadata_items},
                                   Arg{"log_video_buffer_content", log_video_buffer_content},
                                   Arg{"log_python_object_contents", log_python_object_contents}}) {
  init_component_base(this, fragment_or_subgraph, name, "resource");
}

void PySimpleTextSerializer::initialize() {
  HOLOSCAN_LOG_INFO("in PySimpleTextSerializer::initialize");
  SimpleTextSerializer::initialize();

  // Support logging of GILGuardedPyObject
  register_gil_guarded_pyobject_encoder();
}

void PySimpleTextSerializer::setup(ComponentSpec& spec) {
  // Call parent setup first
  SimpleTextSerializer::setup(spec);

  spec.param(log_python_object_contents_,
             "log_python_object_contents",
             "Log Python object contents",
             "Log Python object contents (requires acquiring the GIL)",
             true);
}

void PySimpleTextSerializer::register_gil_guarded_pyobject_encoder() {
  HOLOSCAN_LOG_DEBUG("called register_gil_guarded_pyobject_encoder");
  this->register_encoder<std::shared_ptr<GILGuardedPyObject>>(
      [this](const std::any& value) -> std::string {
        HOLOSCAN_LOG_TRACE("in text serializer codec for std::shared_ptr<GILGuardedPyObject>");
        try {
          // Early return if content logging is disabled
          if (!log_python_object_contents_.get()) {
            return std::string{"Python Object"};
          }

          // Try to safely cast the value first
          std::shared_ptr<GILGuardedPyObject> gil_obj_ptr;
          try {
            gil_obj_ptr = std::any_cast<std::shared_ptr<GILGuardedPyObject>>(value);
          } catch (const std::bad_any_cast& e) {
            return std::string{"<invalid GILGuardedPyObject>"};
          }

          if (!gil_obj_ptr) {
            return std::string{"<null GILGuardedPyObject>"};
          }

          // Try to acquire GIL and access Python object safely
          // If this fails, we're likely in an unsafe threading context
          try {
            // Check if Python is finalizing or async logger shutdown is in progress.
            // Skip GIL acquisition to prevent deadlock where main thread holds GIL
            // while waiting for worker threads to join.
            // Check finalization first as it can be true before our shutdown flag is set.
            if (is_python_finalizing()) {
              return std::string{"Python Object (interpreter finalizing)"};
            }
            if (holoscan::async_logger_shutdown_in_progress().load(std::memory_order_acquire)) {
              return std::string{"Python Object (logger shutdown)"};
            }
            py::gil_scoped_acquire acquire;
            py::object py_obj = gil_obj_ptr->obj();

            // Get the type name
            std::string type_name;
            try {
              type_name = py::str(py_obj.get_type().attr("__name__"));
            } catch (...) {
              type_name = "Unknown";
            }

            // Get the repr
            std::string repr_str;
            try {
              py::object repr_obj = py::repr(py_obj);
              repr_str = py::str(repr_obj);
            } catch (...) {
              repr_str = "<repr failed>";
            }

            return fmt::format("Python({}): {}", type_name, repr_str);
          } catch (const py::error_already_set&) {
            // Python error - likely unsafe thread context
            return std::string{"Python Object (unsafe context)"};
          } catch (...) {
            // Any other error - unsafe to access Python objects
            return std::string{"Python Object (thread unsafe)"};
          }
        } catch (...) {
          // Catch-all for any other errors
          return std::string{"Python Object (error)"};
        }
      });
}

PyBasicConsoleLogger::PyBasicConsoleLogger(
    const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
    std::shared_ptr<SimpleTextSerializer> serializer, bool log_inputs, bool log_outputs,
    bool log_metadata, bool log_tensor_data_content, bool use_scheduler_clock,
    std::optional<std::shared_ptr<Resource>> clock,
    const std::vector<std::string>& allowlist_patterns,
    const std::vector<std::string>& denylist_patterns, const std::string& name)
    : BasicConsoleLogger(ArgList{Arg{"log_inputs", log_inputs},
                                 Arg{"log_outputs", log_outputs},
                                 Arg{"log_metadata", log_metadata},
                                 Arg{"log_tensor_data_content", log_tensor_data_content},
                                 Arg{"use_scheduler_clock", use_scheduler_clock},
                                 Arg{"allowlist_patterns", allowlist_patterns},
                                 Arg{"denylist_patterns", denylist_patterns}}) {
  if (serializer) {
    this->add_arg(Arg{"serializer", serializer});
  }
  if (clock.has_value()) {
    this->add_arg(Arg{"clock", clock.value()});
  }
  init_component_base(this, fragment_or_subgraph, name, "resource");
}

void PyBasicConsoleLogger::initialize() {
  HOLOSCAN_LOG_INFO("in PyBasicConsoleLogger::initialize");
  // Find if there is an argument for 'serializer'
  auto has_serializer = std::find_if(
      args().begin(), args().end(), [](const auto& arg) { return (arg.name() == "serializer"); });

  // Use PySimpleTextSerializer if no serializer was provided
  if (has_serializer == args().end()) {
    add_arg(Arg("serializer", fragment()->make_resource<PySimpleTextSerializer>("serializer")));
  }

  // call parent initialize after adding missing serializer arg above
  BasicConsoleLogger::initialize();
}

PyGXFConsoleLogger::PyGXFConsoleLogger(
    const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
    std::shared_ptr<SimpleTextSerializer> serializer, bool log_inputs, bool log_outputs,
    bool log_metadata, bool log_tensor_data_content, bool use_scheduler_clock,
    std::optional<std::shared_ptr<Resource>> clock,
    const std::vector<std::string>& allowlist_patterns,
    const std::vector<std::string>& denylist_patterns, const std::string& name)
    : GXFConsoleLogger(ArgList{Arg{"log_inputs", log_inputs},
                               Arg{"log_outputs", log_outputs},
                               Arg{"log_metadata", log_metadata},
                               Arg{"log_tensor_data_content", log_tensor_data_content},
                               Arg{"use_scheduler_clock", use_scheduler_clock},
                               Arg{"allowlist_patterns", allowlist_patterns},
                               Arg{"denylist_patterns", denylist_patterns}}) {
  if (serializer) {
    this->add_arg(Arg{"serializer", serializer});
  }
  if (clock.has_value()) {
    this->add_arg(Arg{"clock", clock.value()});
  }
  init_component_base(this, fragment_or_subgraph, name, "resource");
}

void PyGXFConsoleLogger::initialize() {
  HOLOSCAN_LOG_TRACE("PyGXFConsoleLogger::initialize called");
  // Find if there is an argument for 'serializer'
  auto has_serializer = std::find_if(
      args().begin(), args().end(), [](const auto& arg) { return (arg.name() == "serializer"); });

  // Use PySimpleTextSerializer if no serializer was provided
  if (has_serializer == args().end()) {
    add_arg(Arg("serializer", fragment()->make_resource<PySimpleTextSerializer>("serializer")));
  }

  // call parent initialize after adding missing serializer arg above
  GXFConsoleLogger::initialize();
}

/* The python module */

PYBIND11_MODULE(_basic_console_logger, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK BasicConsoleLogger Python Bindings
        -----------------------------------------------
        .. currentmodule:: _basic_console_logger
    )pbdoc";

  py::class_<SimpleTextSerializer,
             PySimpleTextSerializer,
             Resource,
             std::shared_ptr<SimpleTextSerializer>>(
      m, "SimpleTextSerializer", doc::SimpleTextSerializer::doc_SimpleTextSerializer)
      .def(py::init<const std::variant<Fragment*, Subgraph*>&,
                    int64_t,
                    int64_t,
                    bool,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "max_elements"_a = 10,
           "max_metadata_items"_a = 10,
           "log_video_buffer_content"_a = false,
           "log_python_object_contents"_a = true,
           "name"_a = std::string(resource_default_name_v<SimpleTextSerializer>),
           doc::SimpleTextSerializer::doc_SimpleTextSerializer);

  py::class_<BasicConsoleLogger,
             PyBasicConsoleLogger,
             DataLoggerResource,
             std::shared_ptr<BasicConsoleLogger>>(
      m, "BasicConsoleLogger", doc::BasicConsoleLogger::doc_BasicConsoleLogger)
      .def(py::init<const std::variant<Fragment*, Subgraph*>&,
                    std::shared_ptr<SimpleTextSerializer>,
                    bool,
                    bool,
                    bool,
                    bool,
                    bool,
                    std::optional<std::shared_ptr<Resource>>,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    const std::string&>(),
           "fragment"_a,
           "serializer"_a = py::none(),
           "log_inputs"_a = true,
           "log_outputs"_a = true,
           "log_metadata"_a = true,
           "log_tensor_data_content"_a = true,
           "use_scheduler_clock"_a = true,
           "clock"_a = py::none(),
           "allowlist_patterns"_a = py::list(),
           "denylist_patterns"_a = py::list(),
           "name"_a = std::string(resource_default_name_v<BasicConsoleLogger>),
           doc::BasicConsoleLogger::doc_BasicConsoleLogger);

  py::class_<GXFConsoleLogger,
             PyGXFConsoleLogger,
             BasicConsoleLogger,
             std::shared_ptr<GXFConsoleLogger>>(
      m, "GXFConsoleLogger", "GXF-specific extension of BasicConsoleLogger with Entity support")
      .def(py::init<const std::variant<Fragment*, Subgraph*>&,
                    std::shared_ptr<SimpleTextSerializer>,
                    bool,
                    bool,
                    bool,
                    bool,
                    bool,
                    std::optional<std::shared_ptr<Resource>>,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    const std::string&>(),
           "fragment"_a,
           "serializer"_a = py::none(),
           "log_inputs"_a = true,
           "log_outputs"_a = true,
           "log_metadata"_a = true,
           "log_tensor_data_content"_a = true,
           "use_scheduler_clock"_a = true,
           "clock"_a = py::none(),
           "allowlist_patterns"_a = py::list(),
           "denylist_patterns"_a = py::list(),
           "name"_a = std::string(resource_default_name_v<GXFConsoleLogger>),
           doc::GXFConsoleLogger::doc_GXFConsoleLogger);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::data_loggers
