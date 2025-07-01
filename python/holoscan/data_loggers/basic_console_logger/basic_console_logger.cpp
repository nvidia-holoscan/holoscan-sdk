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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for vector

#include <any>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <typeindex>
#include <vector>

#include "./pydoc.hpp"

#include "../../core/gil_guarded_pyobject.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/parameter.hpp"
#include "holoscan/core/resource.hpp"
#include "holoscan/core/resources/data_logger.hpp"
#include "holoscan/data_loggers/basic_console_logger/basic_console_logger.hpp"
#include "holoscan/data_loggers/basic_console_logger/simple_text_serializer.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan::data_loggers {

/* Trampoline class for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the resource.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the resource's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_resource<ResourceT>
 */

class PySimpleTextSerializer : public SimpleTextSerializer {
 public:
  /* Inherit the constructors */
  using SimpleTextSerializer::SimpleTextSerializer;

  // Define a constructor that fully initializes the object.
  PySimpleTextSerializer(Fragment* fragment, const py::args& args, int64_t max_elements = 10,
                         int64_t max_metadata_items = 10, bool log_python_object_contents = true,
                         const std::string& name = "simple_text_serializer")
      : SimpleTextSerializer(
            ArgList{Arg{"max_elements", max_elements},
                    Arg{"max_metadata_items", max_metadata_items},
                    Arg{"log_python_object_contents", log_python_object_contents}}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_);
  }

  void initialize() override {
    HOLOSCAN_LOG_INFO("in PySimpleTextSerializer::initialize");
    SimpleTextSerializer::initialize();

    // Support logging of GILGuardedPyObject
    register_gil_guarded_pyobject_encoder();
  }

  void setup(ComponentSpec& spec) override {
    // Call parent setup first
    SimpleTextSerializer::setup(spec);

    spec.param(log_python_object_contents_,
               "log_python_object_contents",
               "Log Python object contents",
               "Log Python object contents (requires acquiring the GIL)",
               true);
  }

 private:
  Parameter<bool> log_python_object_contents_;

  void register_gil_guarded_pyobject_encoder() {
    this->register_encoder<std::shared_ptr<GILGuardedPyObject>>(
        [this](const std::any& value) -> std::string {
          HOLOSCAN_LOG_INFO("in register_gil_guarded_pyobject_encoder");
          try {
            if (!log_python_object_contents_.get()) { return std::string{"Python Object"}; }

            auto gil_obj_ptr = std::any_cast<std::shared_ptr<GILGuardedPyObject>>(value);
            if (!gil_obj_ptr) { return "<null GILGuardedPyObject>"; }

            py::gil_scoped_acquire acquire;
            py::object py_obj = gil_obj_ptr->obj();

            // Get the type name
            std::string type_name = py::str(py_obj.get_type().attr("__name__"));

            // Get the repr
            std::string repr_str;
            try {
              py::object repr_obj = py::repr(py_obj);
              repr_str = py::str(repr_obj);
            } catch (const std::exception&) { repr_str = "<repr failed>"; }

            return fmt::format("Python({}): {}", type_name, repr_str);
          } catch (const std::exception& e) {
            return std::string("<error encoding GILGuardedPyObject: ") + e.what() + ">";
          }
        });
  }
};

class PyBasicConsoleLogger : public BasicConsoleLogger {
 public:
  /* Inherit the constructors */
  using BasicConsoleLogger::BasicConsoleLogger;

  // Define a constructor that fully initializes the object.
  PyBasicConsoleLogger(Fragment* fragment, const py::args& args,
                       std::shared_ptr<SimpleTextSerializer> serializer = nullptr,
                       bool log_inputs = true, bool log_outputs = true,
                       bool log_tensor_data_content = false, bool log_metadata = false,
                       const std::vector<std::string>& allowlist_patterns = {},
                       const std::vector<std::string>& denylist_patterns = {},
                       const std::string& name = "basic_console_logger")
      : BasicConsoleLogger(ArgList{Arg{"log_inputs", log_inputs},
                                   Arg{"log_outputs", log_outputs},
                                   Arg{"log_tensor_data_content", log_tensor_data_content},
                                   Arg{"log_metadata", log_metadata},
                                   Arg{"allowlist_patterns", allowlist_patterns},
                                   Arg{"denylist_patterns", denylist_patterns}}) {
    if (serializer) { this->add_arg(Arg{"serializer", serializer}); }
    // warn if non-empty allowlist and denylist are specified
    if (!allowlist_patterns.empty() && !denylist_patterns.empty()) {
      std::string warning_msg =
          "BasicConsoleLogger: Both allowlist_patterns and denylist_patterns are specified. "
          "Allowlist takes precedence and denylist will be ignored.";
      try {
        auto warnings = py::module_::import("warnings");
        warnings.attr("warn")(
            warning_msg, py::arg("category") = py::module_::import("builtins").attr("UserWarning"));
      } catch (const py::error_already_set&) {
        // If we can't import warnings or we're not in a Python context, just continue
      } catch (...) {
        // Ignore any other Python-related errors
      }
    }
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_);
  }

  void initialize() override {
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
};

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
      .def(py::init<Fragment*, const py::args&, int64_t, int64_t, bool, const std::string&>(),
           "fragment"_a,
           "max_elements"_a = 10,
           "max_metadata_items"_a = 10,
           "log_python_object_contents"_a = true,
           "name"_a = "simple_text_serializer"s,
           doc::SimpleTextSerializer::doc_SimpleTextSerializer);

  py::class_<BasicConsoleLogger,
             PyBasicConsoleLogger,
             DataLoggerResource,
             std::shared_ptr<BasicConsoleLogger>>(
      m, "BasicConsoleLogger", doc::BasicConsoleLogger::doc_BasicConsoleLogger)
      .def(py::init<Fragment*,
                    const py::args&,
                    std::shared_ptr<SimpleTextSerializer>,
                    bool,
                    bool,
                    bool,
                    bool,
                    const std::vector<std::string>&,
                    const std::vector<std::string>&,
                    const std::string&>(),
           "fragment"_a,
           "serializer"_a = py::none(),
           "log_inputs"_a = true,
           "log_outputs"_a = true,
           "log_tensor_data_content"_a = false,
           "log_metadata"_a = true,
           "allowlist_patterns"_a = py::list(),
           "denylist_patterns"_a = py::list(),
           "name"_a = "basic_console_logger"s,
           doc::BasicConsoleLogger::doc_BasicConsoleLogger);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::data_loggers
