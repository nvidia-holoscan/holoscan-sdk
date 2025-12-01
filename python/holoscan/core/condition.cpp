/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <utility>

#include "component.hpp"
#include "condition_pydoc.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/component.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/condition.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/subgraph.hpp"
#include "kwarg_handling.hpp"

using pybind11::literals::operator""_a;  // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

class PyCondition : public Condition {
 public:
  /* Inherit the constructors */
  using Condition::Condition;

  // Define a kwargs-based constructor that can create an ArgList
  // for passing on to the variadic-template based constructor.
  PyCondition(const py::object& condition, Fragment* fragment, const py::args& args,
              const py::kwargs& kwargs)
      : py_condition_(condition),
        py_initialize_(py::getattr(condition, "initialize")),
        py_update_state_(py::getattr(condition, "update_state")),
        py_check_(py::getattr(condition, "check")),
        py_on_execute_(py::getattr(condition, "on_execute")) {
    using std::string_literals::operator""s;

    // Initialize the component's internal state by setting up the fragment and service provider
    // This mirrors what Fragment::setup_component_internals() does in C++, enabling service()
    // method access
    fragment_ = fragment;
    service_provider_ = fragment;

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

  // Override spec() method
  std::shared_ptr<PyComponentSpec> py_shared_spec() {
    auto spec_ptr = spec_shared();
    return std::static_pointer_cast<PyComponentSpec>(spec_ptr);
  }

  void setup(ComponentSpec& spec) override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE(void, Condition, setup, spec);
  }

  void initialize() override;

  void update_state(int64_t timestamp) override;

  void check(int64_t timestamp, holoscan::SchedulingStatusType* status_type,
             int64_t* target_timestamp) const override;

  void on_execute(int64_t timestamp) override;

 private:
  py::object py_condition_ = py::none();
  py::object py_initialize_ = py::none();    ///> cache the initialize method
  py::object py_update_state_ = py::none();  ///> cache the update_state method
  py::object py_check_ = py::none();         ///> cache the check method
  py::object py_on_execute_ = py::none();    ///> cache the on_execute method
};

void init_condition(py::module_& m) {
  py::enum_<ConditionType>(m, "ConditionType", doc::ConditionType::doc_ConditionType)
      .value("NONE", ConditionType::kNone)
      .value("MESSAGE_AVAILABLE", ConditionType::kMessageAvailable)
      .value("DOWNSTREAM_MESSAGE_AFFORDABLE", ConditionType::kDownstreamMessageAffordable)
      .value("COUNT", ConditionType::kCount)
      .value("BOOLEAN", ConditionType::kBoolean)
      .value("PERIODIC", ConditionType::kPeriodic)
      .value("ASYNCHRONOUS", ConditionType::kAsynchronous)
      .value("EXPIRING_MESSAGE_AVAILABLE", ConditionType::kExpiringMessageAvailable)
      .value("MULTI_MESSAGE_AVAILABLE", ConditionType::kMultiMessageAvailable)
      .value("MULTI_MESSAGE_AVAILABLE_TIMEOUT", ConditionType::kMultiMessageAvailableTimeout);

  py::enum_<SchedulingStatusType>(
      m, "SchedulingStatusType", doc::SchedulingStatusType::doc_SchedulingStatusType)
      .value("NEVER", SchedulingStatusType::kNever)
      .value("READY", SchedulingStatusType::kReady)
      .value("WAIT", SchedulingStatusType::kWait)
      .value("WAIT_TIME", SchedulingStatusType::kWaitTime)
      .value("WAIT_EVENT", SchedulingStatusType::kWaitEvent);

  py::class_<Condition, Component, PyCondition, std::shared_ptr<Condition>> condition_class(
      m, "ConditionBase", py::dynamic_attr(), doc::Condition::doc_Condition);

  condition_class
      .def(py::init<const py::object&, Fragment*, const py::args&, const py::kwargs&>(),
           doc::Condition::doc_Condition_args_kwargs)
      .def(py::init([](py::object condition,
                       std::shared_ptr<Subgraph>
                           subgraph,
                       const py::args& args,
                       const py::kwargs& kwargs) {
             // Extract the fragment from the subgraph
             Fragment* fragment = subgraph->fragment();

             // Create the PyCondition in the Subgraph's fragment
             auto py_condition = std::make_shared<PyCondition>(condition, fragment, args, kwargs);

             // Apply qualified naming using the subgraph's instance name
             std::string qualified_name =
                 subgraph->get_qualified_name(py_condition->name(), "condition");
             py_condition->name(qualified_name);

             return py_condition;
           }),
           "condition"_a,
           "subgraph"_a)
      .def_property(
          "name",
          py::overload_cast<>(&Condition::name, py::const_),
          [](Condition& c, const std::string& name) -> Condition& { return c.name(name); },
          doc::Condition::doc_name)
      .def_property_readonly(
          "fragment", py::overload_cast<>(&Condition::fragment), doc::Condition::doc_fragment)
      .def_property("spec",
                    &Condition::spec_shared,
                    py::overload_cast<const std::shared_ptr<ComponentSpec>&>(&Condition::spec))
      .def("setup", &Condition::setup, doc::Condition::doc_setup)  // note: virtual
      .def("initialize",
           &Condition::initialize,
           doc::Condition::doc_initialize)  // note: virtual function
      .def_property_readonly(
          "condition_type", &Condition::condition_type, doc::Condition::doc_condition_type)
      .def_property_readonly(
          "description", &Condition::description, doc::Condition::doc_description)
      .def("receiver",
           &Condition::receiver,
           "port_name"_a,
           doc::Condition::doc_receiver,
           py::return_value_policy::reference_internal)
      .def("transmitter",
           &Condition::transmitter,
           "port_name"_a,
           doc::Condition::doc_transmitter,
           py::return_value_policy::reference_internal)
      .def(
          "__repr__",
          [](const py::object& obj) {
            // use py::object and obj.cast to avoid a segfault if object has not been initialized
            auto condition = obj.cast<std::shared_ptr<Condition>>();
            if (condition) {
              return condition->description();
            }
            return std::string("<Condition: None>");
          },
          R"doc(Return repr(self).)doc");

  py::enum_<Condition::ConditionComponentType>(condition_class, "ConditionComponentType")
      .value("NATIVE", Condition::ConditionComponentType::kNative)
      .value("GXF", Condition::ConditionComponentType::kGXF);
}

void PyCondition::initialize() {
  // Get the initialize method of the Python Condition class and call it
  py::gil_scoped_acquire scope_guard;

  // Call the parent class's `initialize()` method to set up the condition arguments so that
  // parameters can be accessed in the `initialize()` method of the Python Condition class.
  //
  // In C++, this call is made in the `initialize()` method of the inheriting Condition class,
  // using `Condition::initialize()` call. In Python, the user doesn't have to call the parent
  // class's `initialize()` method explicitly. If there is a need to initialize something (such as
  // adding arguments), it can be done directly in the `__init__` method of the Python class
  // inheriting from the Condition class before calling the parent class's `__init__` method using
  // `super().__init__(fragment, *args, **kwargs)`.
  Condition::initialize();

  // TODO(unknown): add tracing support like in PyOperator
  // set_py_tracing();

  py_initialize_.operator()();
}

void PyCondition::update_state(int64_t timestamp) {
  // Get the update_state method of the Python Condition class and call it
  py::gil_scoped_acquire scope_guard;

  // TODO(unknown): add tracing support like in PyOperator
  // set_py_tracing();

  py_update_state_.operator()(timestamp);
}

void PyCondition::check(int64_t timestamp, holoscan::SchedulingStatusType* status_type,
                        int64_t* target_timestamp) const {
  // Get the check method of the Python Condition class and call it
  py::gil_scoped_acquire scope_guard;

  // TODO(unknown): add tracing support like in PyOperator
  // set_py_tracing();

  // The Python method returns a tuple containing Python objects corresponding to `status_type` and
  // `target_timestamp`. These objects are converted to C++ types and stored in the pointers passed
  // to the PyCondition::check method. The target timestamp is allowed to be None in which case no
  // value will be written to *target_timestamp.
  py::object check_result = py_check_.operator()(timestamp);
  if (py::isinstance<py::tuple>(check_result)) {
    auto tuple = check_result.cast<py::tuple>();
    if (tuple.size() == 2) {
      if (!py::isinstance<holoscan::SchedulingStatusType>(tuple[0])) {
        throw std::runtime_error(
            "The first element of the tuple returned by check must be a "
            "`holoscan.core.SchedulingStatusType` enum value");
      }
      *status_type = tuple[0].cast<holoscan::SchedulingStatusType>();
      if (py::isinstance<py::int_>(tuple[1])) {
        *target_timestamp = tuple[1].cast<int64_t>();
      } else if (!(tuple[1].is_none())) {
        throw std::runtime_error(
            "The second element of the tuple returned by check must be a Python int or None");
      }
    } else {
      throw std::runtime_error("check method must return a tuple of size 2");
    }
  } else {
    throw std::runtime_error("check method must return a tuple");
  }
}

void PyCondition::on_execute(int64_t timestamp) {
  // Get the on_execute method of the Python Condition class and call it
  py::gil_scoped_acquire scope_guard;

  // TODO(unknown): add tracing support like in PyOperator
  // set_py_tracing();

  py_on_execute_.operator()(timestamp);
}

}  // namespace holoscan
