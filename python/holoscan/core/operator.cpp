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

#include "operator.hpp"

#include <pybind11/functional.h>  // for lambda functions
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gil_guarded_pyobject.hpp"
#include "holoscan/core/component.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/condition.hpp"
#include "holoscan/core/expected.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/metadata.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resource.hpp"
#include "holoscan/core/subgraph.hpp"
#include "kwarg_handling.hpp"
#include "operator_pydoc.hpp"

using pybind11::literals::operator""_a;  // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

void init_operator(py::module_& m) {
  py::class_<MultiMessageConditionInfo, std::shared_ptr<MultiMessageConditionInfo>>(
      m,
      "MultiMessageConditionInfo",
      R"doc(Information associated with a multi-message condition.)doc")
      .def(py::init<>())
      .def_readwrite("kind", &MultiMessageConditionInfo::kind)
      .def_readwrite("port_names", &MultiMessageConditionInfo::port_names)
      .def_readwrite("args", &MultiMessageConditionInfo::args);

  py::class_<Operator::FlowInfo, std::shared_ptr<Operator::FlowInfo>>(
      m, "FlowInfo", doc::Operator::doc_FlowInfo)
      .def_readonly("curr_operator",
                    &Operator::FlowInfo::curr_operator,
                    "Current operator in the flow connection.")
      .def_readonly("output_port_name",
                    &Operator::FlowInfo::output_port_name,
                    "Name of the output port from the current operator.")
      .def_readonly("output_port_spec",
                    &Operator::FlowInfo::output_port_spec,
                    "Specification of the output port.")
      .def_readonly("next_operator",
                    &Operator::FlowInfo::next_operator,
                    "Next operator in the flow connection.")
      .def_readonly("input_port_name",
                    &Operator::FlowInfo::input_port_name,
                    "Name of the input port on the next operator.")
      .def_readonly("input_port_spec",
                    &Operator::FlowInfo::input_port_spec,
                    "Specification of the input port.");

  py::class_<OperatorSpec, ComponentSpec, std::shared_ptr<OperatorSpec>>(
      m, "OperatorSpec", R"doc(Operator specification class.)doc")
      .def(py::init<Fragment*>(), "fragment"_a, doc::OperatorSpec::doc_OperatorSpec)
      .def("input",
           py::overload_cast<>(&OperatorSpec::input<gxf::Entity>),
           doc::OperatorSpec::doc_input,
           py::return_value_policy::reference_internal)
      .def(
          "input",
          // Note: The return type needs to be specified explicitly because pybind11 can't
          // deduce it. Otherwise, this method will return a new IOSpec object instead of a
          // reference to the existing one.
          [](OperatorSpec& spec,
             const std::string& name,
             const py::object& size,
             std::optional<IOSpec::QueuePolicy> policy = std::nullopt) -> IOSpec& {
            // Check if 'size' is an int and convert to IOSpec::IOSize if necessary
            if (py::isinstance<py::int_>(size)) {
              auto size_int = size.cast<int>();
              // Assuming IOSpec::IOSize can be constructed from an int
              return spec.input<gxf::Entity>(name, IOSpec::IOSize(size_int), policy);
            }
            if (py::isinstance<IOSpec::IOSize>(size)) {
              // Directly pass IOSpec::IOSize if 'size' is already the correct type
              return spec.input<gxf::Entity>(name, size.cast<IOSpec::IOSize>(), policy);
            }
            throw std::runtime_error(
                "Invalid type for 'size'. Expected 'int' or 'holoscan.core.IOSpec.IOSize'.");
          },
          "name"_a,
          py::kw_only(),
          "size"_a = IOSpec::kSizeOne,
          "policy"_a = py::none(),
          doc::OperatorSpec::doc_input_kwargs,
          py::return_value_policy::reference_internal)
      .def_property_readonly("inputs",
                             &OperatorSpec::inputs,
                             doc::OperatorSpec::doc_inputs,
                             py::return_value_policy::reference_internal)
      .def("output",
           py::overload_cast<>(&OperatorSpec::output<gxf::Entity>),
           doc::OperatorSpec::doc_output,
           py::return_value_policy::reference_internal)
      .def(
          "output",
          // Note: The return type needs to be specified explicitly because pybind11 can't
          // deduce it. Otherwise, this method will return a new IOSpec object instead of a
          // reference to the existing one.
          [](OperatorSpec& spec,
             const std::string& name,
             const py::object& size,
             std::optional<IOSpec::QueuePolicy> policy = std::nullopt) -> IOSpec& {
            // Check if 'size' is an int and convert to IOSpec::IOSize if necessary
            if (py::isinstance<py::int_>(size)) {
              auto size_int = size.cast<int>();
              // Assuming IOSpec::IOSize can be constructed from an int
              return spec.output<gxf::Entity>(name, IOSpec::IOSize(size_int), policy);
            }
            if (py::isinstance<IOSpec::IOSize>(size)) {
              // Directly pass IOSpec::IOSize if 'size' is already the correct type
              return spec.output<gxf::Entity>(name, size.cast<IOSpec::IOSize>(), policy);
            }
            throw std::runtime_error(
                "Invalid type for 'size'. Expected 'int' or 'holoscan.core.IOSpec.IOSize'.");
          },
          "name"_a,
          py::kw_only(),
          "size"_a = IOSpec::kSizeOne,
          "policy"_a = py::none(),
          doc::OperatorSpec::doc_output_kwargs,
          py::return_value_policy::reference_internal)
      .def(
          "multi_port_condition",
          [](OperatorSpec& spec,
             ConditionType type,
             const std::vector<std::string>& port_names,
             const py::kwargs& kwargs) {
            // special handling of str -> YAML::Node conversion for sampling_mode argument
            ArgList extra_args{};
            for (const auto& [name, handle] : kwargs) {
              auto arg_name = name.cast<std::string>();
              auto arg_value = handle.cast<py::object>();
              if (arg_name == std::string("sampling_mode")) {
                if (py::isinstance<py::str>(arg_value)) {
                  auto mode_str = arg_value.cast<std::string>();
                  if (mode_str == "SumOfAll") {
                    extra_args.add(Arg("sampling_mode", YAML::Node("SumOfAll")));
                  } else if (mode_str == "PerReceiver") {
                    extra_args.add(Arg("sampling_mode", YAML::Node("PerReceiver")));
                  } else {
                    throw std::runtime_error("Invalid sampling mode: " + mode_str);
                  }
                } else {
                  throw std::runtime_error("Invalid type for 'sampling_mode'. Expected 'str'.");
                }
                kwargs.attr("pop")(arg_name);
              }
            }
            // automatically convert the remaining arguments
            ArgList args = kwargs_to_arglist(kwargs);
            // append any arguments such as sampling_mode that were handled separately
            args.add(extra_args);
            return spec.multi_port_condition(type, port_names, args);
          },
          "kind"_a,
          "port_names"_a,
          doc::OperatorSpec::doc_multi_port_condition)
      .def("multi_port_conditions",
           &OperatorSpec::multi_port_conditions,
           doc::OperatorSpec::doc_multi_port_conditions)
      .def("or_combine_port_conditions",
           &OperatorSpec::or_combine_port_conditions,
           "port_names"_a,
           doc::OperatorSpec::doc_or_combine_port_conditions)
      .def_property_readonly("outputs",
                             &OperatorSpec::outputs,
                             doc::OperatorSpec::doc_outputs,
                             py::return_value_policy::reference_internal)
      .def_property_readonly(
          "description", &OperatorSpec::description, doc::OperatorSpec::doc_description)
      .def(
          "__repr__",
          [](const OperatorSpec& spec) { return spec.description(); },
          R"doc(Return repr(self).)doc");

  // Note: In the case of OperatorSpec, InputContext, OutputContext, ExecutionContext,
  //       there are a separate, independent wrappers for PyOperatorSpec, PyInputContext,
  //       PyOutputContext, PyExecutionContext. These Py* variants are not exposed directly
  //       to end users of the API, but are used internally to enable native operators
  //       defined from python via inheritance from the `Operator` class as defined in
  //       core/__init__.py.

  py::class_<PyOperatorSpec, OperatorSpec, std::shared_ptr<PyOperatorSpec>>(
      m, "PyOperatorSpec", R"doc(Operator specification class.)doc")
      .def(py::init<Fragment*, py::object>(),
           "fragment"_a,
           "op"_a,
           doc::OperatorSpec::doc_OperatorSpec)
      .def("param",
           &PyOperatorSpec::py_param,
           "Register parameter",
           "name"_a,
           "default_value"_a = py::none(),
           "flag"_a = ParameterFlag::kNone,
           doc::OperatorSpec::doc_param);

  // note: added py::dynamic_attr() to allow dynamically adding attributes in a Python subclass
  py::class_<Operator, ComponentBase, PyOperator, std::shared_ptr<Operator>> operator_class(
      m, "Operator", py::dynamic_attr(), doc::Operator::doc_Operator_args_kwargs);

  operator_class
      .def(py::init<py::object, Fragment*, const py::args&, const py::kwargs&>(),
           doc::Operator::doc_Operator_args_kwargs)
      .def(py::init([](py::object op,
                       std::shared_ptr<Subgraph>
                           subgraph,
                       const py::args& args,
                       const py::kwargs& kwargs) {
             // Check if subgraph is nullptr (None from Python)
             if (!subgraph) {
               throw py::type_error("subgraph parameter cannot be None");
             }

             // Extract the fragment from the subgraph
             Fragment* fragment = subgraph->fragment();

             // Create the PyOperator in the Subgraph's fragment
             auto py_op = std::make_shared<PyOperator>(op, fragment, args, kwargs);

             // Apply qualified naming using the subgraph's instance name
             std::string qualified_name = subgraph->get_qualified_name(py_op->name(), "operator");
             py_op->name(qualified_name);

             return py_op;
           }),
           "op"_a,
           "subgraph"_a)
      .def_property(
          "name",
          py::overload_cast<>(&Operator::name, py::const_),
          [](Operator& op, const std::string& name) -> Operator& { return op.name(name); },
          doc::Operator::doc_name)
      .def_property_readonly(
          "fragment", py::overload_cast<>(&Operator::fragment), doc::Operator::doc_fragment)
      .def_property("spec",
                    &Operator::spec_shared,
                    py::overload_cast<const std::shared_ptr<OperatorSpec>&>(&Operator::spec),
                    doc::Operator::doc_spec)
      .def("receiver",
           &Operator::receiver,
           "port_name"_a,
           doc::Operator::doc_receiver,
           py::return_value_policy::reference_internal)
      .def("transmitter",
           &Operator::transmitter,
           "port_name"_a,
           doc::Operator::doc_transmitter,
           py::return_value_policy::reference_internal)
      .def("enable_metadata",
           &Operator::enable_metadata,
           "enable"_a,
           doc::Operator::doc_enable_metadata)
      .def("queue_policy",
           &Operator::queue_policy,
           "port_name"_a,
           "port_type"_a = IOSpec::IOType::kInput,
           "policy"_a = IOSpec::QueuePolicy::kFault,
           doc::Operator::doc_queue_policy,
           py::return_value_policy::reference_internal)
      .def_property_readonly("conditions", &Operator::conditions, doc::Operator::doc_conditions)
      .def_property_readonly("resources", &Operator::resources, doc::Operator::doc_resources)
      // TODO(unknown): sphinx API doc build complains if more than one overloaded add_dynamic_flow
      // method has a docstring specified. For now using the docstring defined for 2-argument
      // Operator-based version and describing the other variants in the Notes section.
      .def("add_dynamic_flow",
           py::overload_cast<const std::shared_ptr<Operator::FlowInfo>&>(
               &Operator::add_dynamic_flow),
           "flow"_a)
      .def("add_dynamic_flow",
           py::overload_cast<const std::vector<std::shared_ptr<Operator::FlowInfo>>&>(
               &Operator::add_dynamic_flow),
           "flows"_a)
      .def("add_dynamic_flow",
           py::overload_cast<const std::shared_ptr<Operator>&, const std::string&>(
               &Operator::add_dynamic_flow),
           "next_op"_a,
           "next_input_port_name"_a = "",
           doc::Operator::doc_add_dynamic_flow)
      .def("add_dynamic_flow",
           py::overload_cast<const std::string&,
                             const std::shared_ptr<Operator>&,
                             const std::string&>(&Operator::add_dynamic_flow),
           "curr_output_port_name"_a,
           "next_op"_a,
           "next_input_port_name"_a = "")
      .def_property_readonly("next_flows",
                             &Operator::next_flows,
                             doc::Operator::doc_next_flows,
                             py::return_value_policy::reference_internal)
      // Note: We don't need to expose `dynamic_flows()` here because it is used internally by the
      // GXFWrapper class
      .def("find_flow_info",
           &Operator::find_flow_info,
           "predicate"_a,
           doc::Operator::doc_find_flow_info,
           py::return_value_policy::reference_internal)
      .def("find_all_flow_info",
           &Operator::find_all_flow_info,
           "predicate"_a,
           doc::Operator::doc_find_all_flow_info,
           py::return_value_policy::reference_internal)
      .def_property_readonly(
          "operator_type", &Operator::operator_type, doc::Operator::doc_operator_type)
      .def_property_readonly(
          "metadata", py::overload_cast<>(&Operator::metadata), doc::Operator::doc_metadata)
      .def_property_readonly("is_metadata_enabled",
                             py::overload_cast<>(&Operator::is_metadata_enabled, py::const_),
                             doc::Operator::doc_is_metadata_enabled)
      .def_property("metadata_policy",
                    py::overload_cast<>(&Operator::metadata_policy, py::const_),
                    py::overload_cast<MetadataPolicy>(&Operator::metadata_policy),
                    doc::Operator::doc_metadata_policy)
      .def(
          "resource",
          [](Operator& op, const py::str& name) -> std::optional<py::object> {
            auto resources = op.resources();
            auto res = resources.find(name);
            if (res == resources.end()) {
              return py::none();
            }
            return py::cast(res->second);
          },
          "name"_a,
          doc::Operator::doc_resource)
      .def("add_arg",
           py::overload_cast<const Arg&>(&Operator::add_arg),
           "arg"_a,
           doc::Operator::doc_add_arg_Arg)
      .def("add_arg",
           py::overload_cast<const ArgList&>(&Operator::add_arg),
           "arg"_a,
           doc::Operator::doc_add_arg_ArgList)
      .def(
          "add_arg",
          [](Operator& op, const py::kwargs& kwargs) {
            return op.add_arg(kwargs_to_arglist(kwargs));
          },
          doc::Operator::doc_add_arg_kwargs)
      .def("add_arg",
           py::overload_cast<const std::shared_ptr<Condition>&>(&Operator::add_arg),
           "arg"_a,
           doc::Operator::doc_add_arg_condition)
      // Note: to avoid a doc build error, only list Parameters in docstring of the last overload
      .def("add_arg",
           py::overload_cast<const std::shared_ptr<Resource>&>(&Operator::add_arg),
           "arg"_a,
           doc::Operator::doc_add_arg_resource)
      .def("initialize",
           &Operator::initialize,
           doc::Operator::doc_initialize)                        // note: virtual function
      .def("setup", &Operator::setup, doc::Operator::doc_setup)  // note: virtual function
      .def("start",
           &Operator::start,  // note: virtual function
           doc::Operator::doc_start,
           py::call_guard<py::gil_scoped_release>())  // note: should release GIL
      .def("stop",
           &Operator::stop,  // note: virtual function
           doc::Operator::doc_stop,
           py::call_guard<py::gil_scoped_release>())  // note: should release GIL
      .def("compute",
           &Operator::compute,  // note: virtual function
           doc::Operator::doc_compute,
           py::call_guard<py::gil_scoped_release>())  // note: should release GIL
      .def_property_readonly("description", &Operator::description, doc::Operator::doc_description)
      .def_property_readonly_static(
          "INPUT_EXEC_PORT_NAME",
          [](py::object /* self */) { return Operator::kInputExecPortName; },
          "Default input execution port name.")
      .def_property_readonly_static(
          "OUTPUT_EXEC_PORT_NAME",
          [](py::object /* self */) { return Operator::kOutputExecPortName; },
          "Default output execution port name.")
      .def_property_readonly("async_condition",
                             &Operator::async_condition,
                             doc::Operator::doc_async_condition,
                             py::return_value_policy::reference_internal)
      .def("stop_execution", &Operator::stop_execution, doc::Operator::doc_stop_execution)
      .def_property_readonly("execution_context",
                             &Operator::execution_context,
                             doc::Operator::doc_execution_context,
                             py::return_value_policy::reference_internal)
      .def(
          "__repr__",
          [](const py::object& obj) {
            // use py::object and obj.cast to avoid a segfault if object has not been initialized
            auto op = obj.cast<std::shared_ptr<Operator>>();
            if (op) {
              return op->description();
            }
            return std::string("<Operator: None>");
          },
          R"doc(Return repr(self).)doc");

  py::enum_<Operator::OperatorType>(
      operator_class, "OperatorType", doc::OperatorType::doc_OperatorType)
      .value("NATIVE", Operator::OperatorType::kNative)
      .value("GXF", Operator::OperatorType::kGXF)
      .value("VIRTUAL", Operator::OperatorType::kVirtual);
}  // init_operator

PyOperatorSpec::PyOperatorSpec(Fragment* fragment, py::object op)
    : OperatorSpec(fragment), py_op_(py::cast<std::shared_ptr<PyOperator>>(op)) {}

void PyOperatorSpec::py_param(const std::string& name, const py::object& default_value,
                              const ParameterFlag& flag, const py::kwargs& kwargs) {
  using std::string_literals::operator""s;

  bool is_receivers = false;
  std::string headline{""s};
  std::string description{""s};
  for (const auto& [kw_name, value] : kwargs) {
    auto param_name = kw_name.cast<std::string>();
    if (param_name == "headline") {
      headline = value.cast<std::string>();
    } else if (param_name == "description") {
      description = value.cast<std::string>();
    } else if (param_name == "kind") {
      auto kind_val = value.cast<std::string>();
      if (kind_val == "receivers") {
        is_receivers = true;
      } else {
        throw std::runtime_error("unknown kind: '"s + kind_val +
                                 "'. Only `kind='receivers'` is currently supported."s);
      }
    } else {
      throw std::runtime_error("unsupported kwarg: "s + param_name);
    }
  }

  if (is_receivers) {
    // Create receivers object
    receivers_params_.emplace_back();

    // Register parameter
    auto& parameter = receivers_params_.back();
    param(parameter, name.c_str(), headline.c_str(), description.c_str(), {}, flag);
  } else {
    // Create parameter object.
    // Note that we create a weakref object to avoid incrementing/decrementing the
    // reference count for the object because owning the object would create a cyclic reference.
    py_params_.emplace_back(py::weakref(py_op()));

    // Register parameter
    auto& parameter = py_params_.back();
    // Please see register_py_type() in 'public/python/holoscan/core/arg.cpp' to see how
    // Parameter<py::object> is handled.
    param(parameter, name.c_str(), headline.c_str(), description.c_str(), default_value, flag);
  }
}

py::object PyOperatorSpec::py_op() const {
  if (auto py_op = py_op_.lock()) {
    return py::cast(py_op);
  }
  return py::none();
}

std::list<Parameter<py::object>>& PyOperatorSpec::py_params() {
  return py_params_;
}

std::list<Parameter<std::vector<IOSpec*>>>& PyOperatorSpec::py_receivers() {
  return receivers_params_;
}

// PyOperator
PyOperator::PyOperator(const py::object& op, Fragment* fragment, const py::args& args,
                       const py::kwargs& kwargs)
    : py_op_(op),
      py_compute_(py::getattr(op, "compute")),
      py_initialize_(py::getattr(op, "initialize")),
      py_start_(py::getattr(op, "start")),
      py_stop_(py::getattr(op, "stop")) {
  using std::string_literals::operator""s;

  HOLOSCAN_LOG_TRACE("PyOperator::PyOperator()");
  // Initialize the component's internal state by setting up the fragment and service provider
  // This mirrors what Fragment::setup_component_internals() does in C++, enabling service()
  // method access
  fragment_ = fragment;
  service_provider_ = fragment;

  // Store the application object to access the trace/profile functions
  auto* app = fragment_->application();
  py_app_ = dynamic_cast<PyApplication*>(app);

  // Parse args
  for (const auto& item : args) {
    auto arg_value = item.cast<py::object>();
    if (py::isinstance<Condition>(arg_value)) {
      this->add_arg(arg_value.cast<std::shared_ptr<Condition>>());
    } else if (py::isinstance<Resource>(arg_value)) {
      this->add_arg(arg_value.cast<std::shared_ptr<Resource>>());
    } else if (py::isinstance<Fragment>(arg_value)) {
      throw std::runtime_error("multiple Fragment objects provided");
    } else if (py::isinstance<ArgList>(arg_value)) {
      this->add_arg(arg_value.cast<ArgList>());
    } else if (py::isinstance<Arg>(arg_value)) {
      this->add_arg(arg_value.cast<Arg>());
    } else {
      this->add_arg(py_object_to_arg(arg_value, ""s));
    }
  }

  // Pars kwargs
  for (const auto& [name, value] : kwargs) {
    auto kwarg_name = name.cast<std::string>();
    auto kwarg_value = value.cast<py::object>();
    if (kwarg_name == "name"s) {
      if (py::isinstance<py::str>(kwarg_value)) {
        this->name(kwarg_value.cast<std::string>());
      } else {
        throw std::runtime_error("name kwarg must be a string");
      }
    } else if (kwarg_name == "fragment"s) {
      throw std::runtime_error("fragment cannot be passed via a kwarg, only positionally");
    } else if (py::isinstance<Condition>(kwarg_value)) {
      // Set the condition's name to the kwarg name
      auto cond = kwarg_value.cast<std::shared_ptr<Condition>>();
      cond->name(kwarg_name);
      this->add_arg(cond);
    } else if (py::isinstance<Resource>(kwarg_value)) {
      // Set the resource's name to the kwarg name
      auto resource = kwarg_value.cast<std::shared_ptr<Resource>>();
      resource->name(kwarg_name);
      this->add_arg(resource);
    } else {
      this->add_arg(py_object_to_arg(kwarg_value, kwarg_name));
    }
  }

  // Set name if needed
  if (name_.empty()) {
    static size_t op_number;
    op_number++;
    this->name("unnamed_operator_" + std::to_string(op_number));
  }
}

std::shared_ptr<PyOperatorSpec> PyOperator::py_shared_spec() {
  auto spec_ptr = spec_shared();
  return std::static_pointer_cast<PyOperatorSpec>(spec_ptr);
}

PyOperator::GILGuardedThreadLocal::GILGuardedThreadLocal() {
  py::gil_scoped_acquire scope_guard;
  data.pydevd_trace_func = py::none();
  data.pydevd_set_trace_to_threads_func = py::none();
  data.pydevd_thread_idents = py::none();
  data.dummy_thread = py::none();
}

PyOperator::GILGuardedThreadLocal::~GILGuardedThreadLocal() {
  // Since this destructor is called when exiting the thread, acquiring GIL and
  // decreasing reference count of Python objects in the thread-local data
  // may cause the interpreter to crash. So, we don't acquire GIL or handle
  // reference count of Python objects here.

  // Just release the Python objects.
  data.pydevd_trace_func.release();
  data.pydevd_set_trace_to_threads_func.release();
  data.pydevd_thread_idents.release();
  data.dummy_thread.release();

  data.py_last_frame = nullptr;

  data.c_profilefunc = nullptr;
  data.c_tracefunc = nullptr;
  data.c_profileobj = nullptr;
  data.c_traceobj = nullptr;
}

PyOperator::TracingThreadLocal& PyOperator::get_tracing_data() {
  // Define a thread-local object for storing tracing data.
  // Important: The type of a thread_local variable should be a pointer due to issues
  // with Thread-Local Storage (TLS) when dynamically loading libraries using dlopen().
  // The TLS space is limited to 2048 bytes.
  // For more details, refer to: https://fasterthanli.me/articles/a-dynamic-linker-murder-mystery.
  static thread_local std::unique_ptr<GILGuardedThreadLocal> gil_guarded_thread_local =
      std::make_unique<GILGuardedThreadLocal>();

  py::gil_scoped_acquire scope_guard;

  auto& data = gil_guarded_thread_local->data;

  // Return the cached thread-local data if it is already initialized
  if (data.is_initialized) {
    return data;
  }

  try {
    if (data.dummy_thread.is_none()) {
      // Create a dummy thread object for this thread by calling threading.current_thread()
      // so that debugger can recognize this thread as a Python thread.
      auto threading_module = py::module::import("threading");
      auto current_thread_func = py::getattr(threading_module, "current_thread");
      // Create py::object object having the result of current_thread_func()
      data.dummy_thread = current_thread_func();
    }

    // Check if the module name starts with '_pydevd_bundle' which means that it is using
    // PyDevd debugger. If so, then we need to store pydevd-specific data.

    // Get py_trace_func_'s class object using "__class__" attr
    auto trace_module = py_app_->py_trace_func_.attr("__class__").attr("__module__");
    // Check if the module name starts with '_pydevd_bundle' which means that it is using
    // PyDevd debugger. If so, then we need to set the trace function to the current frame.
    auto module_name = trace_module.cast<std::string>();
    // NOLINTNEXTLINE(abseil-string-find-str-contains)
    if (module_name.find("_pydevd_bundle") != std::string::npos) {
      if (data.pydevd_trace_func.is_none()) {
        // Get the trace function from the debugger
        auto pydevd_module = py::module::import("pydevd");
        auto debugger = py::getattr(pydevd_module, "GetGlobalDebugger")();
        // Get the trace function from the debugger
        data.pydevd_trace_func = py::getattr(debugger, "get_thread_local_trace_func")();
      }

      if (data.pydevd_set_trace_to_threads_func.is_none()) {
        auto pydevd_module = py::module::import("pydevd");

        data.pydevd_set_trace_to_threads_func =
            pydevd_module.attr("pydevd_tracing").attr("set_trace_to_threads");
      }

      if (data.pydevd_thread_idents.is_none()) {
        auto thread_module = py::module::import("_thread");
        auto get_ident_func = py::getattr(thread_module, "get_ident");
        // Create py::list object having the result of get_ident_func()
        auto thread_idents = py::list();
        thread_idents.append(get_ident_func());

        data.pydevd_thread_idents = thread_idents;
      }
    }

    data.is_initialized = true;
    data.in_tracing = (py_app_->c_tracefunc_ != nullptr) || (py_app_->c_profilefunc_ != nullptr);
    data.is_pydevd = (!data.pydevd_trace_func.is_none()) &&
                     (!data.pydevd_set_trace_to_threads_func.is_none()) &&
                     (!data.pydevd_thread_idents.is_none());
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_WARN("Exception occurredPyOperator:: while initializing tracing data: {}",
                      e.what());
    data.is_initialized = true;
    data.in_tracing = false;  // pretend that tracing is not enabled for this thread
    data.is_pydevd = false;
  }

  return data;
}

void PyOperator::set_py_tracing() {
  auto& tracing_data = get_tracing_data();

  try {
    // If tracing is not enabled, do nothing and return
    if (!tracing_data.in_tracing) {
      return;
    }

#if PY_VERSION_HEX >= 0x030D0000  // >= Python 3.13.0
    // Python 3.13 increased enforcement of thread safety
    auto* py_thread_state = PyThreadState_Get();
#else
    auto* py_thread_state = _PyThreadState_UncheckedGet();
#endif
    // If tracing_data.is_func_set is false, cache the current trace/profile functions for
    // the current thread.
    if (!tracing_data.is_func_set) {
      auto& py_last_frame = py_app_->py_last_frame_;
      auto& py_profile_func = py_app_->py_profile_func_;
      auto& py_trace_func = py_app_->py_trace_func_;
      auto& c_profilefunc = py_app_->c_profilefunc_;
      auto& c_profileobj = py_app_->c_profileobj_;
      auto& c_tracefunc = py_app_->c_tracefunc_;
      auto& c_traceobj = py_app_->c_traceobj_;

      tracing_data.py_last_frame = py_last_frame;

      // If pydevd is used, call pydevd.pydevd_tracing.set_trace_to_threads() to set
      // the trace function to the current thread.
      if (tracing_data.is_pydevd) {
        tracing_data.pydevd_set_trace_to_threads_func(
            tracing_data.pydevd_trace_func,
            py::arg("thread_idents") = tracing_data.pydevd_thread_idents,
            py::arg("create_dummy_thread") = py::bool_(false));

        tracing_data.c_profilefunc = py_thread_state->c_profilefunc;
        tracing_data.c_profileobj = py_thread_state->c_profileobj;

        tracing_data.c_tracefunc = py_thread_state->c_tracefunc;
        tracing_data.c_traceobj = py_thread_state->c_traceobj;
      } else {
        // If pydevd is not used, call sys.settrace/setprofile() to set
        // the trace/profile function to the current thread.
        auto sys_module = py::module::import("sys");

        // Check if py_profile_func is callable and call it.
        // In case of cProfile.Profile object, it is not callable so should not be called.
        if (!py_profile_func.is_none() && py::isinstance<py::function>(py_profile_func)) {
          sys_module.attr("setprofile")(py_profile_func);
          tracing_data.c_profilefunc = py_thread_state->c_profilefunc;
          tracing_data.c_profileobj = py_thread_state->c_profileobj;
        } else {
          HOLOSCAN_LOG_DEBUG("py_profile_func_ is not callable");
          tracing_data.c_profilefunc = c_profilefunc;
          tracing_data.c_profileobj = c_profileobj;
        }

        // Check if py_trace_func is callable and call it.
        if (!py_trace_func.is_none() && py::isinstance<py::function>(py_trace_func)) {
          sys_module.attr("settrace")(py_trace_func);
          tracing_data.c_tracefunc = py_thread_state->c_tracefunc;
          tracing_data.c_traceobj = py_thread_state->c_traceobj;
        } else {
          HOLOSCAN_LOG_DEBUG("py_trace_func_ is not callable");
          tracing_data.c_tracefunc = c_tracefunc;
          tracing_data.c_traceobj = c_traceobj;
        }
      }
      tracing_data.is_func_set = true;
    }

    // Set the trace/profile functions to the current thread.
    // Depending on the Python version, the way to set the trace/profile functions is different.

    // Set current frame to the last valid Python frame
#if PY_VERSION_HEX >= 0x030D0000  // >= Python 3.13.0
    // _PyThreadState_SetFrame is removed in Python 3.13 and there does not seem to
    // be any equivalent available.
#elif PY_VERSION_HEX >= 0x030B0000  // >= Python 3.11.0
    // https://github.com/python/cpython/blob/c184c6750e40ca4ffa4f62a5d145b892cbd066bc
    //   /Doc/whatsnew/3.11.rst#L2301
    // - tstate->frame is removed.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    py_thread_state->cframe->current_frame =
        reinterpret_cast<_PyInterpreterFrame*>(tracing_data.py_last_frame);
#else                               // < Python 3.11.0
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    py_thread_state->frame = reinterpret_cast<PyFrameObject*>(tracing_data.py_last_frame);
#endif

#if PY_VERSION_HEX >= 0x030D0000  // >= Python 3.13.0
    // set profile and tracing for the current thread using public API
    // (this API also exists even back in Python 3.9)
    PyEval_SetProfile(tracing_data.c_profilefunc, tracing_data.c_profileobj.ptr());
    PyEval_SetTrace(tracing_data.c_tracefunc, tracing_data.c_traceobj.ptr());

    // Python 3.12+ also has AllThreads variants of these
    // PyEval_SetProfileAllThreads(tracing_data.c_profilefunc, tracing_data.c_profileobj.ptr());
    // PyEval_SetTraceAllThreads(tracing_data.c_tracefunc, tracing_data.c_traceobj.ptr());
#elif PY_VERSION_HEX >= 0x030B0000  // >= Python 3.11.0
    // Recommended way to set the trace/profile functions in Python 3.11
    // (see
    // https://discuss.python.org/t/python-3-11-frame-structure-and-various-changes/17895/19)
    _PyEval_SetProfile(
        py_thread_state, tracing_data.c_profilefunc, tracing_data.c_profileobj.ptr());
    _PyEval_SetTrace(py_thread_state, tracing_data.c_tracefunc, tracing_data.c_traceobj.ptr());
#else                               // < Python 3.11.0
    py_thread_state->c_profilefunc = tracing_data.c_profilefunc;
    Py_XINCREF(tracing_data.c_profileobj.ptr());
    Py_XDECREF(py_thread_state->c_profileobj);
    py_thread_state->c_profileobj = tracing_data.c_profileobj.ptr();

    py_thread_state->c_tracefunc = tracing_data.c_tracefunc;
    Py_XINCREF(tracing_data.c_traceobj.ptr());
    Py_XDECREF(py_thread_state->c_traceobj);
    py_thread_state->c_traceobj = tracing_data.c_traceobj.ptr();
#if PY_VERSION_HEX >= 0x030A00B1    // >= Python 3.10.0 b1
    py_thread_state->cframe->use_tracing = 1;
#else                               // < Python 3.10.0 b1
    py_thread_state->use_tracing = 1;
#endif                              // about Python 3.10.0 b1
#endif                              // about Python 3.11.0
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_WARN("Exception occurred while setting trace/profile functions: {}", e.what());
    tracing_data.is_initialized = true;
    tracing_data.is_pydevd = false;
    tracing_data.in_tracing = false;  // pretend that tracing is not enabled for this thread
  }
}

void PyOperator::initialize() {
  // Get the initialize method of the Python Operator class and call it
  py::gil_scoped_acquire scope_guard;

  // Call the parent class's `initialize()` method to set up the operator arguments so that
  // parameters can be accessed in the `initialize()` method of the Python Operator class.
  //
  // In C++, this call is made in the `initialize()` method (usually at the end of the method)
  // of the inheriting Operator class, using `Operator::initialize()` call.
  // In Python, the user doesn't have to call the parent class's `initialize()` method explicitly.
  // If there is a need to initialize something (such as adding arguments), it can be done
  // directly in the `__init__` method of the Python class inheriting from the Operator class before
  // calling the parent class's `__init__` method using
  // `super().__init__(fragment, *args, **kwargs)`.
  Operator::initialize();

#if PY_VERSION_HEX < 0x030C0000  // < 3.12
  set_py_tracing();
#endif

  auto* context = fragment_->executor().context();

  // Create PyExecutionContext, PyInputContext, PyOutputContext objects and store them
  if (py_op_) {
    auto py_op = py_op_.cast<std::shared_ptr<PyOperator>>();
    py_context_ = std::make_shared<PyExecutionContext>(context, py_op);
    py_op_input_ = py_context_->py_input();
    py_op_output_ = py_context_->py_output();

    // Make sure CudaObjectHandler has been initialized for use by py_emit and py_receive
    py_context_->init_cuda_object_handler(this);
    py_op_input_->cuda_object_handler(py_context_->cuda_object_handler());
    py_op_output_->cuda_object_handler(py_context_->cuda_object_handler());
    HOLOSCAN_LOG_TRACE("PyOperator: py_context_->cuda_object_handler() for op '{}' is {}null",
                       name(),
                       py_context_->cuda_object_handler() == nullptr ? "" : "not ");
  } else {
    std::string err_msg = fmt::format("PyOperator '{}': py_op_ is not set", name());
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }
  if (py_initialize_.is_none()) {
    std::string err_msg = fmt::format("PyOperator '{}': initialize method is set to None", name());
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }

  py_initialize_.operator()();
}

void PyOperator::start() {
  // Get the start method of the Python Operator class and call it
  py::gil_scoped_acquire scope_guard;

#if PY_VERSION_HEX < 0x030C0000  // < 3.12
  set_py_tracing();
#endif
  if (py_start_.is_none()) {
    std::string err_msg = fmt::format("PyOperator '{}': start method is set to None", name());
    HOLOSCAN_LOG_ERROR(err_msg);
    throw std::runtime_error(err_msg);
  }
  py_start_.operator()();
}

void PyOperator::stop() {
  // Get the stop method of the Python Operator class and call it
  py::gil_scoped_acquire scope_guard;

#if PY_VERSION_HEX < 0x030C0000  // < 3.12
  set_py_tracing();
#endif

  // Guard: if the Python callable has been cleared during release_internal_resources(),
  // skip invoking it.
  if (py_stop_.is_none()) {
    HOLOSCAN_LOG_DEBUG(
        "PyOperator '{}' stop() skipped: py_stop_ is None (this might occur during "
        "application shutdown for some schedulers)",
        name());
    return;
  }
  py_stop_.operator()();
}

void PyOperator::compute(InputContext& op_input, OutputContext& op_output,
                         ExecutionContext& context) {
  // Get the compute method of the Python Operator class and call it
  py::gil_scoped_acquire scope_guard;

  py_context_->clear_received_streams();

#if PY_VERSION_HEX < 0x030C0000  // < 3.12
  set_py_tracing();
#endif

  // Guard: if the Python callable has been cleared during release_internal_resources(),
  // skip invoking it.
  if (py_compute_.is_none()) {
    HOLOSCAN_LOG_DEBUG(
        "PyOperator '{}' compute() skipped: py_compute_ is None (this might occur during "
        "application shutdown for some schedulers)",
        name());
    return;
  }

  py_compute_.operator()(py::cast(py_op_input_), py::cast(py_op_output_), py::cast(py_context_));
}

std::shared_ptr<holoscan::ExecutionContext> PyOperator::execution_context() const {
  return py_context_;
}

void PyOperator::release_internal_resources() {
  py::gil_scoped_acquire scope_guard;

  Operator::release_internal_resources();

  // Reset the Python objects

  // Note: We hold py_op_ (the Python object reference) in the C++ layer via
  //       PyApplication::python_operator_registry_ and
  //       PyFragment::python_operator_registry_
  //
  // Explanation:
  // py_op_ holds the py::object reference to the original Python wrapper object created
  // when this Operator's Python subclass was instantiated. Resetting it (e.g., via
  // 'py_op_ = py::none();') would release the C++ object's ownership of the Python
  // reference count for that wrapper.
  //
  // If no other Python code holds a reference to this specific Python wrapper object
  // at the time the C++ object is destroyed (or if py_op_ were reset earlier),
  // resetting py_op_ would allow Python's garbage collector to destroy the wrapper.
  //
  // When a pybind11 wrapper object is destroyed, its entry is removed from pybind11's
  // internal instance registry (which maps C++ pointers to existing Python wrappers).
  //
  // Consequently, if the same underlying C++ Operator is later retrieved from C++ code
  // and passed back to Python (e.g., via a call like fragment.graph.get_nodes()),
  // pybind11 would fail to find the original wrapper in its registry. It would then
  // be forced to create a *new*, base-class Python wrapper object ('holoscan.core._core.Operator').
  //
  // This new wrapper would:
  //   1. Have a different Python id().
  //   2. Lack any dynamically added attributes (like 'none_count' used in tests) that
  //      might have been set only on the original Python subclass instance.
  //
  // This behavior would break tests like 'test_ping_app_none_condition' in
  // 'python/tests/system/test_application_ping_none_condition.py', which depend on
  // dynamic attributes persisting on operators retrieved from the application graph.
  //
  // Therefore, we maintain the Python Operator object reference indefinitely within
  // the C++ object to guarantee the persistence and identity (id()) of the original Python wrapper
  // object throughout the lifetime of the C++ Operator instance it corresponds to.
  //
  // Current implementation is maintaining the Python objects in
  // PyApplication::python_operator_registry_ and PyFragment::python_operator_registry_ when
  // add_operator() or add_flow() method is called on PyApplication or PyFragment.
  //
  // We can safely reset py_op_ (the Python object reference) here because we maintain the reference
  // in `python_operator_registry_`. This reference will be properly reset via the `reset_state()`
  // method before the `run()` or `run_async()` method is called, or when the Application/Fragment
  // object's destructor executes. This approach enables consecutive runs while allowing operator
  // information to be accessed by the data flow tracker, and prevents memory leaks that could occur
  // from self-referencing Python objects in the PyOperator class.
  py_op_ = py::none();
  py_initialize_ = py::none();
  py_start_ = py::none();
  py_stop_ = py::none();
  py_compute_ = py::none();
  py_op_input_.reset();
  py_op_output_.reset();
  py_context_.reset();

  // Reset the Python application pointer
  py_app_ = nullptr;
}

}  // namespace holoscan
