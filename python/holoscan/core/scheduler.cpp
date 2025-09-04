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

#include <pybind11/pybind11.h>

#include <memory>
#include <string>

#include "holoscan/core/arg.hpp"
#include "holoscan/core/component.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/scheduler.hpp"
#include "kwarg_handling.hpp"
#include "scheduler_pydoc.hpp"

namespace py = pybind11;

namespace holoscan {

class PyScheduler : public Scheduler {
 public:
  /* Inherit the constructors */
  using Scheduler::Scheduler;

  // Define a kwargs-based constructor that can create an ArgList
  // for passing on to the variadic-template based constructor.
  PyScheduler(const py::args& args, const py::kwargs& kwargs) {
    using std::string_literals::operator""s;

    int n_fragments = 0;
    for (const auto& item : args) {
      auto arg_value = item.cast<py::object>();
      if (py::isinstance<Fragment>(arg_value)) {
        if (n_fragments > 0) {
          throw std::runtime_error("multiple Fragment objects provided");
        }
        fragment_ = arg_value.cast<Fragment*>();
        n_fragments += 1;
      } else {
        this->add_arg(py_object_to_arg(arg_value, ""s));
      }
    }
    for (const auto& [name, value] : kwargs) {
      auto kwarg_name = name.cast<std::string>();
      auto kwarg_value = value.cast<py::object>();
      if (kwarg_name == "name"s) {
        if (py::isinstance<py::str>(kwarg_value)) {
          name_ = kwarg_value.cast<std::string>();
        } else {
          throw std::runtime_error("name kwarg must be a string");
        }
      } else if (kwarg_name == "fragment"s) {
        if (py::isinstance<Fragment>(kwarg_value)) {
          if (n_fragments > 0) {
            throw std::runtime_error(
                "Cannot add kwarg fragment, when a Fragment was also provided positionally");
          }
          fragment_ = kwarg_value.cast<Fragment*>();
        } else {
          throw std::runtime_error("fragment kwarg must be a Fragment");
        }
      } else {
        this->add_arg(py_object_to_arg(kwarg_value, kwarg_name));
      }
    }
  }

  /* Trampolines (need one for each virtual function) */
  void initialize() override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Scheduler, initialize);
  }
  void setup(ComponentSpec& spec) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Scheduler, setup, spec);
  }
  std::shared_ptr<Clock> clock() override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<Clock>, Scheduler, clock);
  }
};

void init_scheduler(py::module_& m) {
  py::class_<Scheduler, Component, PyScheduler, std::shared_ptr<Scheduler>>(
      m, "Scheduler", doc::Scheduler::doc_Scheduler)
      .def(py::init<const py::args&, const py::kwargs&>(),
           doc::Scheduler::doc_Scheduler_args_kwargs)
      .def_property(
          "name",
          py::overload_cast<>(&Scheduler::name, py::const_),
          [](Scheduler& s, const std::string& name) -> Scheduler& { return s.name(name); },
          doc::Scheduler::doc_name)
      .def_property_readonly(
          "fragment", py::overload_cast<>(&Scheduler::fragment), doc::Scheduler::doc_fragment)
      .def_property("spec",
                    &Scheduler::spec_shared,
                    py::overload_cast<const std::shared_ptr<ComponentSpec>&>(&Scheduler::spec))
      .def("setup", &Scheduler::setup, doc::Scheduler::doc_setup)  // note: virtual
      .def("initialize",
           &Scheduler::initialize,
           doc::Scheduler::doc_initialize)  // note: virtual function
      .def_property_readonly("clock", &Scheduler::clock)
      .def(
          "__repr__",
          [](const py::object& obj) {
            // use py::object and obj.cast to avoid a segfault if object has not been initialized
            auto scheduler = obj.cast<std::shared_ptr<Scheduler>>();
            if (scheduler) {
              return scheduler->description();
            }
            return std::string("<Scheduler: None>");
          },
          R"doc(Return repr(self).)doc");
}

}  // namespace holoscan
