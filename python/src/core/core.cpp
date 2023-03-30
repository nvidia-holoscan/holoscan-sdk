/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime.h>
#include <pybind11/complex.h>  // complex support
#include <pybind11/numpy.h>    // needed for py::array and py::dtype
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

#include <cstdint>
#include <memory>
#include <string>

#include "../kwarg_handling.hpp"
#include "core.hpp"
#include "core_pydoc.hpp"
#include "dl_converter.hpp"
#include "holoscan/core/application.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/argument_setter.hpp"
#include "holoscan/core/component.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/condition.hpp"
#include "holoscan/core/config.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/executors/gxf/gxf_parameter_adaptor.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/parameter.hpp"
#include "holoscan/core/resource.hpp"
#include "trampolines.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan {

template <typename typeT>
static void register_py_type() {
  auto& arg_setter = ArgumentSetter::get_instance();
  arg_setter.add_argument_setter<typeT>([](ParameterWrapper& param_wrap, Arg& arg) {
    std::any& any_param = param_wrap.value();
    // Note that the type of any_param is Parameter<typeT>*, not Parameter<typeT>.
    auto& param = *std::any_cast<Parameter<typeT>*>(any_param);

    // If arg has no name and value, that indicates that we want to set the default value for
    // the native operator if it is not specified.
    if (arg.name().empty() && !arg.has_value()) {
      const char* key = param.key().c_str();
      // If the attribute is not None, we do not need to set it.
      // Otherwise, we set it to the default value (if it exists).
      if (!py::hasattr(param.get(), key) ||
          (py::getattr(param.get(), key).is(py::none()) && param.has_default_value())) {
        py::setattr(param.get(), key, param.default_value());
      }
      return;
    }

    // `PyOperatorSpec.py_param` will have stored the actual Operator class from Python
    // into a Parameter<py::object> param, so `param.get()` here is `PyOperatorSpec.py_op`.
    // `param.key()` is the name of the attribute on the Python class.
    // We can then set that Class attribute's value to the value contained in Arg.
    // Here we use the `arg_to_py_object` utility to convert from `Arg` to `py::object`.
    py::setattr(param.get(), param.key().c_str(), arg_to_py_object(arg));
  });
}

PYBIND11_MODULE(_core, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _core
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::enum_<ArgElementType>(m, "ArgElementType", doc::ArgElementType::doc_ArgElementType)
      .value("CUSTOM", ArgElementType::kCustom)
      .value("BOOLEAN", ArgElementType::kBoolean)
      .value("INT8", ArgElementType::kInt8)
      .value("UNSIGNED8", ArgElementType::kUnsigned8)
      .value("INT16", ArgElementType::kInt16)
      .value("UNSIGNED16", ArgElementType::kUnsigned16)
      .value("INT32", ArgElementType::kInt32)
      .value("UNSIGNED32", ArgElementType::kUnsigned32)
      .value("INT64", ArgElementType::kInt64)
      .value("UNSIGNED64", ArgElementType::kUnsigned64)
      .value("FLOAT32", ArgElementType::kFloat32)
      .value("FLOAT64", ArgElementType::kFloat64)
      .value("STRING", ArgElementType::kString)
      .value("HANDLE", ArgElementType::kHandle)
      .value("YAML_NODE", ArgElementType::kYAMLNode)
      .value("IO_SPEC", ArgElementType::kIOSpec)
      .value("CONDITION", ArgElementType::kCondition)
      .value("RESOURCE", ArgElementType::kResource);

  py::enum_<ArgContainerType>(m, "ArgContainerType", doc::ArgContainerType::doc_ArgContainerType)
      .value("NATIVE", ArgContainerType::kNative)
      .value("VECTOR", ArgContainerType::kVector)
      .value("ARRAY", ArgContainerType::kArray);

  py::class_<Arg>(m, "Arg", R"doc(Class representing a typed argument.)doc")
      .def(py::init<const std::string&>(), "name"_a, doc::Arg::doc_Arg)
      // Do not implement overloaded initializers for object types here.
      //    Instead, see py_object_to_arg() utility for getting an Arg object from a Python one
      // Arg& operator=(const ArgT& value)
      // Arg&& operator=(ArgT&& value)
      .def_property_readonly("name", &Arg::name, doc::Arg::doc_name)
      .def_property_readonly("arg_type", &Arg::arg_type, doc::Arg::doc_arg_type)
      .def_property_readonly("has_value", &Arg::has_value, doc::Arg::doc_has_value)
      // std::any& value()
      .def("__int__",
           [](Arg& arg) -> py::object {
             auto result = arg_to_py_object(arg);
             if (!result.is_none()) {
               if (py::isinstance<py::int_>(result)) {
                 return result;
               } else if (py::isinstance<py::float_>(result)) {
                 return py::int_(static_cast<int64_t>(result.cast<double>()));
               }
             }
             return py::none();
           })
      .def("__float__",
           [](Arg& arg) -> py::object {
             auto result = arg_to_py_object(arg);
             if (!result.is_none()) {
               if (py::isinstance<py::float_>(result)) {
                 return result;
               } else if (py::isinstance<py::int_>(result)) {
                 return py::float_(static_cast<double>(result.cast<int64_t>()));
               }
             }
             return py::none();
           })
      .def("__bool__",
           [](Arg& arg) -> py::object {
             auto result = arg_to_py_object(arg);
             if (!result.is_none()) { return py::bool_(py::cast<bool>(result)); }
             return py::none();
           })
      .def("__str__",
           [](Arg& arg) -> py::object {
             auto result = arg_to_py_object(arg);
             if (py::isinstance<py::str>(result)) {
               return result;
             } else if (!result.is_none()) {
               return py::str(result);
             }
             return py::str();
           })
      .def_property_readonly("description", &Arg::description, doc::Arg::doc_description)
      .def(
          "__repr__",
          [](const Arg& arg) { return arg.description(); },
          py::call_guard<py::gil_scoped_release>(),
          R"doc(Return repr(self).)doc");

  py::class_<ArgList>(m, "ArgList", doc::ArgList::doc_ArgList)
      .def(py::init<>(), doc::ArgList::doc_ArgList)
      .def_property_readonly("name", &ArgList::name, doc::ArgList::doc_name)
      .def_property_readonly("size", &ArgList::size, doc::ArgList::doc_size)
      .def_property_readonly("args", &ArgList::args, doc::ArgList::doc_args)
      .def("clear", &ArgList::clear, doc::ArgList::doc_clear)
      .def("add", py::overload_cast<const Arg&>(&ArgList::add), "arg"_a, doc::ArgList::doc_add_Arg)
      .def("add",
           py::overload_cast<const ArgList&>(&ArgList::add),
           "arg"_a,
           doc::ArgList::doc_add_ArgList)
      .def_property_readonly("description", &ArgList::description, doc::ArgList::doc_description)
      .def(
          "__repr__",
          [](const ArgList& list) { return list.description(); },
          py::call_guard<py::gil_scoped_release>(),
          R"doc(Return repr(self).)doc");

  py::class_<ArgType>(m, "ArgType", doc::ArgType::doc_ArgType)
      .def(py::init<>(), doc::ArgType::doc_ArgType)
      .def(py::init<ArgElementType, ArgContainerType>(),
           "element_type"_a,
           "container_type"_a,
           doc::ArgType::doc_ArgType_kwargs)
      .def_property_readonly("element_type", &ArgType::element_type, doc::ArgType::doc_element_type)
      .def_property_readonly(
          "container_type", &ArgType::container_type, doc::ArgType::doc_container_type)
      .def_property_readonly("dimension", &ArgType::dimension, doc::ArgType::doc_dimension)
      .def_property_readonly("to_string", &ArgType::to_string, doc::ArgType::doc_to_string)
      .def(
          "__repr__",
          [](const ArgType& t) { return t.to_string(); },
          py::call_guard<py::gil_scoped_release>(),
          R"doc(Return repr(self).)doc");

  py::class_<Component, PyComponent, std::shared_ptr<Component>>(
      m, "Component", doc::Component::doc_Component)
      .def(py::init<>(), doc::Component::doc_Component)
      .def_property_readonly("id", &Component::id, doc::Component::doc_id)
      .def_property_readonly("name", &Component::name, doc::Component::doc_name)
      .def_property_readonly("fragment", &Component::fragment, doc::Component::doc_fragment)
      .def("add_arg",
           py::overload_cast<const Arg&>(&Component::add_arg),
           "arg"_a,
           doc::Component::doc_add_arg_Arg)
      .def("add_arg",
           py::overload_cast<const ArgList&>(&Component::add_arg),
           "arg"_a,
           doc::Component::doc_add_arg_ArgList)
      .def_property_readonly("args", &Component::args, doc::Component::doc_args)
      .def("initialize",
           &Component::initialize,
           doc::Component::doc_initialize)  // note: virtual function
      .def_property_readonly(
          "description", &Component::description, doc::Component::doc_description)
      .def(
          "__repr__",
          [](const Component& c) { return c.description(); },
          py::call_guard<py::gil_scoped_release>(),
          R"doc(Return repr(self).)doc");

  py::enum_<ConditionType>(m, "ConditionType", doc::ConditionType::doc_ConditionType)
      .value("NONE", ConditionType::kNone)
      .value("MESSAGE_AVAILABLE", ConditionType::kMessageAvailable)
      .value("DOWNSTREAM_MESSAGE_AFFORDABLE", ConditionType::kDownstreamMessageAffordable)
      .value("COUNT", ConditionType::kCount)
      .value("BOOLEAN", ConditionType::kBoolean);

  py::class_<Condition, Component, PyCondition, std::shared_ptr<Condition>>(
      m, "Condition", doc::Condition::doc_Condition)
      .def(py::init<const py::args&, const py::kwargs&>(),
           doc::Condition::doc_Condition_args_kwargs)
      .def_property("name",
                    py::overload_cast<>(&Condition::name, py::const_),
                    (Condition & (Condition::*)(const std::string&)&)&Condition::name,
                    doc::Condition::doc_name)
      .def_property("fragment",
                    py::overload_cast<>(&Condition::fragment),
                    py::overload_cast<Fragment*>(&Condition::fragment),
                    doc::Condition::doc_fragment)
      .def_property("spec",
                    &Condition::spec_shared,
                    py::overload_cast<const std::shared_ptr<ComponentSpec>&>(&Condition::spec))
      .def("setup", &Condition::setup, doc::Condition::doc_setup)  // note: virtual
      .def("initialize",
           &Condition::initialize,
           doc::Condition::doc_initialize)  // note: virtual function
      .def_property_readonly(
          "description", &Condition::description, doc::Condition::doc_description)
      .def(
          "__repr__",
          [](const Condition& c) { return c.description(); },
          py::call_guard<py::gil_scoped_release>(),
          R"doc(Return repr(self).)doc");

  py::class_<Resource, Component, PyResource, std::shared_ptr<Resource>>(
      m, "Resource", doc::Resource::doc_Resource)
      .def(py::init<const py::args&, const py::kwargs&>(), doc::Resource::doc_Resource_args_kwargs)
      .def_property("name",
                    py::overload_cast<>(&Resource::name, py::const_),
                    (Resource & (Resource::*)(const std::string&)&)&Resource::name,
                    doc::Resource::doc_name)
      .def_property("fragment",
                    py::overload_cast<>(&Resource::fragment),
                    py::overload_cast<Fragment*>(&Resource::fragment),
                    doc::Resource::doc_fragment)
      .def_property("spec",
                    &Resource::spec_shared,
                    py::overload_cast<const std::shared_ptr<ComponentSpec>&>(&Resource::spec))
      .def("setup", &Resource::setup, doc::Resource::doc_setup)  // note: virtual
      .def("initialize",
           &Resource::initialize,
           doc::Resource::doc_initialize)  // note: virtual function
      .def_property_readonly("description", &Resource::description, doc::Resource::doc_description)
      .def(
          "__repr__",
          [](const Resource& c) { return c.description(); },
          py::call_guard<py::gil_scoped_release>(),
          R"doc(Return repr(self).)doc");

  py::class_<InputContext, std::shared_ptr<InputContext>> input_context(
      m, "InputContext", doc::InputContext::doc_InputContext);

  input_context.def(
      "receive", [](const InputContext&, const std::string&) { return py::none(); }, "name"_a);

  py::class_<OutputContext, std::shared_ptr<OutputContext>> output_context(
      m, "OutputContext", doc::OutputContext::doc_OutputContext);

  output_context.def(
      "emit",
      [](const OutputContext&, py::object&, const std::string&) {},
      "data"_a,
      "name"_a = "");

  py::enum_<OutputContext::OutputType>(output_context, "OutputType")
      .value("SHARED_POINTER", OutputContext::OutputType::kSharedPointer)
      .value("GXF_ENTITY", OutputContext::OutputType::kGXFEntity);

  py::class_<ExecutionContext, std::shared_ptr<ExecutionContext>>(
      m, "ExecutionContext", doc::ExecutionContext::doc_ExecutionContext);

  py::class_<Message>(m, "Message", doc::Message::doc_Message);

  py::class_<ComponentSpec, std::shared_ptr<ComponentSpec>>(
      m, "ComponentSpec", R"doc(Component specification class.)doc")
      .def(py::init<Fragment*>(), "fragment"_a, doc::ComponentSpec::doc_ComponentSpec)
      .def_property_readonly("fragment",
                             py::overload_cast<>(&ComponentSpec::fragment),
                             doc::ComponentSpec::doc_fragment)
      .def_property_readonly("params", &ComponentSpec::params, doc::ComponentSpec::doc_params)
      .def_property_readonly(
          "description", &ComponentSpec::description, doc::ComponentSpec::doc_description)
      .def(
          "__repr__",
          [](const ComponentSpec& spec) { return spec.description(); },
          py::call_guard<py::gil_scoped_release>(),
          R"doc(Return repr(self).)doc");

  py::class_<IOSpec> iospec(m, "IOSpec", R"doc(I/O specification class.)doc");
  iospec
      .def(py::init<OperatorSpec*, const std::string&, IOSpec::IOType>(),
           "op_spec"_a,
           "name"_a,
           "io_type"_a,
           doc::IOSpec::doc_IOSpec)
      .def_property_readonly("name", &IOSpec::name, doc::IOSpec::doc_name)
      .def_property_readonly("io_type", &IOSpec::io_type, doc::IOSpec::doc_io_type)
      .def_property("resource",
                    py::overload_cast<>(&IOSpec::resource, py::const_),
                    py::overload_cast<std::shared_ptr<Resource>>(&IOSpec::resource),
                    doc::IOSpec::doc_resource)
      .def_property_readonly("conditions", &IOSpec::conditions, doc::IOSpec::doc_conditions)
      .def(
          "condition",
          [](IOSpec& io_spec, const ConditionType& kind, const py::kwargs& kwargs) {
            return io_spec.condition(kind, kwargs_to_arglist(kwargs));
          },
          doc::IOSpec::doc_condition)
      .def(
          "__repr__",
          [](const IOSpec& iospec) {
            std::string nm = iospec.name();
            return fmt::format(
                "<IOSpec: name={}, io_type={}>", nm, io_type_namemap.at(iospec.io_type()));
          },
          py::call_guard<py::gil_scoped_release>(),
          R"doc(Return repr(self).)doc");

  py::enum_<IOSpec::IOType>(iospec, "IOType", doc::IOType::doc_IOType)
      .value("INPUT", IOSpec::IOType::kInput)
      .value("OUTPUT", IOSpec::IOType::kOutput);

  py::class_<OperatorSpec, ComponentSpec, std::shared_ptr<OperatorSpec>>(
      m, "OperatorSpec", R"doc(Operator specification class.)doc")
      .def(py::init<Fragment*>(), "fragment"_a, doc::OperatorSpec::doc_OperatorSpec)
      .def("input",
           py::overload_cast<>(&OperatorSpec::input<gxf::Entity>),
           doc::OperatorSpec::doc_input,
           py::return_value_policy::reference_internal)
      .def("input",
           py::overload_cast<std::string>(&OperatorSpec::input<gxf::Entity>),
           "name"_a,
           doc::OperatorSpec::doc_input_name,
           py::return_value_policy::reference_internal)
      .def("output",
           py::overload_cast<>(&OperatorSpec::output<gxf::Entity>),
           doc::OperatorSpec::doc_output,
           py::return_value_policy::reference_internal)
      .def("output",
           py::overload_cast<std::string>(&OperatorSpec::output<gxf::Entity>),
           "name"_a,
           doc::OperatorSpec::doc_output_name,
           py::return_value_policy::reference_internal)
      .def_property_readonly(
          "description", &OperatorSpec::description, doc::OperatorSpec::doc_description)
      .def(
          "__repr__",
          [](const OperatorSpec& spec) { return spec.description(); },
          py::call_guard<py::gil_scoped_release>(),
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
           "default_value"_a = py::none());

  py::class_<PyInputContext, InputContext, std::shared_ptr<PyInputContext>>(
      m, "PyInputContext", R"doc(Input context class.)doc")
      .def("receive", &PyInputContext::py_receive);

  py::class_<PyOutputContext, OutputContext, std::shared_ptr<PyOutputContext>>(
      m, "PyOutputContext", R"doc(Output context class.)doc")
      .def("emit", &PyOutputContext::py_emit);

  py::class_<PyExecutionContext, ExecutionContext, std::shared_ptr<PyExecutionContext>>(
      m, "PyExecutionContext", R"doc(Execution context class.)doc")
      .def_property_readonly("input", &PyExecutionContext::py_input)
      .def_property_readonly("output", &PyExecutionContext::py_output);

  // note: added py::dynamic_attr() to allow dynamically adding attributes in a Python subclass
  py::class_<Operator, Component, PyOperator, std::shared_ptr<Operator>> operator_class(
      m, "Operator", py::dynamic_attr(), doc::Operator::doc_Operator_args_kwargs);

  operator_class
      .def(py::init<py::object, Fragment*, const py::args&, const py::kwargs&>(),
           doc::Operator::doc_Operator_args_kwargs)
      .def_property("name",
                    py::overload_cast<>(&Operator::name, py::const_),
                    (Operator & (Operator::*)(const std::string&)) & Operator::name,
                    doc::Operator::doc_name)
      .def_property("fragment",
                    py::overload_cast<>(&Operator::fragment),
                    py::overload_cast<Fragment*>(&Operator::fragment),
                    doc::Operator::doc_fragment)
      .def_property("spec",
                    &Operator::spec_shared,
                    py::overload_cast<const std::shared_ptr<OperatorSpec>&>(&Operator::spec))
      .def_property_readonly("conditions", &Operator::conditions, doc::Operator::doc_conditions)
      .def_property_readonly("resources", &Operator::resources, doc::Operator::doc_resources)
      .def_property_readonly(
          "operator_type", &Operator::operator_type, doc::Operator::doc_operator_type)
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
      .def(
          "__repr__",
          [](const Operator& op) { return op.description(); },
          py::call_guard<py::gil_scoped_release>(),
          R"doc(Return repr(self).)doc");

  py::enum_<Operator::OperatorType>(operator_class, "OperatorType")
      .value("NATIVE", Operator::OperatorType::kNative)
      .value("GXF", Operator::OperatorType::kGXF);

  py::class_<Config>(m, "Config", doc::Config::doc_Config)
      .def(py::init<const std::string&, const std::string&>(),
           "config_file"_a,
           "prefix"_a = "",
           doc::Config::doc_Config_kwargs)
      .def_property_readonly("config_file", &Config::config_file, doc::Config::doc_config_file)
      .def_property_readonly("prefix", &Config::prefix, doc::Config::doc_prefix);

  py::class_<Executor, PyExecutor, std::shared_ptr<Executor>>(
      m, "Executor", R"doc(Executor class.)doc")
      .def(py::init<Fragment*>(), "fragment"_a, doc::Executor::doc_Executor)
      .def("run", &Executor::run, doc::Executor::doc_run)  // note: virtual function
      .def_property_readonly(
          "fragment", py::overload_cast<>(&Executor::fragment), doc::Executor::doc_fragment)
      .def_property("context",
                    py::overload_cast<>(&Executor::context),
                    py::overload_cast<void*>(&Executor::context),
                    doc::Executor::doc_context)
      .def_property("context_uint64",
                    py::overload_cast<>(&Executor::context_uint64),
                    py::overload_cast<uint64_t>(&Executor::context_uint64),
                    doc::Executor::doc_context_uint64);

  // note: added py::dynamic_attr() to allow dynamically adding attributes in a Python subclass
  py::class_<Fragment, PyFragment>(m, "Fragment", py::dynamic_attr(), doc::Fragment::doc_Fragment)
      .def(py::init<>(), doc::Fragment::doc_Fragment)
      // notation for this name setter is a bit tricky (couldn't seem to do it with overload_cast)
      .def_property("name",
                    py::overload_cast<>(&Fragment::name, py::const_),
                    (Fragment & (Fragment::*)(const std::string&)&)&Fragment::name,
                    doc::Fragment::doc_name)
      .def("application", &Fragment::application, doc::Fragment::doc_application)
      .def("config",
           py::overload_cast<const std::string&, const std::string&>(&Fragment::config),
           "config_file"_a,
           "prefix"_a = "",
           doc::Fragment::doc_config_kwargs)
      .def("config", py::overload_cast<>(&Fragment::config), doc::Fragment::doc_config)
      .def_property_readonly("graph", &Fragment::graph, doc::Fragment::doc_graph)
      .def_property_readonly("executor", &Fragment::executor, doc::Fragment::doc_executor)
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
      .def(                                  // note: virtual function
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
      .def("run",
           &Fragment::run,
           doc::Fragment::doc_run,
           py::call_guard<py::gil_scoped_release>())  // note: virtual function
      .def(
          "__repr__",
          [](const Fragment& fragment) {
            return fmt::format("<holoscan.Fragment: name:{}>", fragment.name());
          },
          py::call_guard<py::gil_scoped_release>(),
          R"doc(Return repr(self).)doc");

  // note: added py::dynamic_attr() to allow dynamically adding attributes in a Python subclass
  py::class_<Application, Fragment, PyApplication>(
      m, "Application", py::dynamic_attr(), doc::Application::doc_Application)
      .def(py::init<>(), doc::Application::doc_Application)
      .def("add_operator",
           &Application::add_operator,
           "op"_a,
           doc::Application::doc_add_operator)  // note: virtual function
      .def(                                     // note: virtual function
          "add_flow",
          py::overload_cast<const std::shared_ptr<Operator>&, const std::shared_ptr<Operator>&>(
              &Application::add_flow),
          "upstream_op"_a,
          "downstream_op"_a)
      // TODO: sphinx API doc build complains if more than one of this add_flow
      //       overrides has a docstring specified.
      .def(  // note: virtual function
          "add_flow",
          py::overload_cast<const std::shared_ptr<Operator>&,
                            const std::shared_ptr<Operator>&,
                            std::set<std::pair<std::string, std::string>>>(&Application::add_flow),
          "upstream_op"_a,
          "downstream_op"_a,
          "port_pairs"_a,
          doc::Application::doc_add_flow_pair)
      .def(  // note: virtual function
          "add_flow",
          py::overload_cast<const std::shared_ptr<Fragment>&, const std::shared_ptr<Fragment>&>(
              &Application::add_flow),
          "upstream_frag"_a,
          "downstream_frag"_a)
      .def(  // note: virtual function
          "add_flow",
          py::overload_cast<const std::shared_ptr<Fragment>&,
                            const std::shared_ptr<Fragment>&,
                            const std::set<std::pair<std::string, std::string>>&>(
              &Application::add_flow),
          "upstream_frag"_a,
          "downstream_frag"_a,
          "port_pairs"_a)
      .def("compose",
           &Application::compose,
           doc::Application::doc_compose)  // note: virtual function
      .def("run",
           &Application::run,
           doc::Application::doc_run,
           py::call_guard<py::gil_scoped_release>())  // note: virtual function
      .def(
          "__repr__",
          [](const Application& app) {
            return fmt::format("<holoscan.Application: name:{}>", app.name());
          },
          py::call_guard<py::gil_scoped_release>(),
          R"doc(Return repr(self).)doc");
  // Note: unimplemented: templated make_fragment method

  // Note: currently not wrapping ArgumentSetter as it was not needed from Python
  // Note: currently not wrapping GXFParameterAdaptor as it was not needed from Python
  // Note: currently not wrapping individual MetaParameter class templates

  // DLPack data structures
  py::enum_<DLDeviceType>(m, "DLDeviceType", py::module_local())  //
      .value("DLCPU", kDLCPU)                                     //
      .value("DLCUDA", kDLCUDA)                                   //
      .value("DLCUDAHOST", kDLCUDAHost)                           //
      .value("DLCUDAMANAGED", kDLCUDAManaged);

  py::class_<DLDevice>(m, "DLDevice", py::module_local(), doc::DLDevice::doc_DLDevice)
      .def(py::init<DLDeviceType, int32_t>())
      .def_readonly("device_type",
                    &DLDevice::device_type,
                    doc::DLDevice::doc_device_type,
                    py::call_guard<py::gil_scoped_release>())
      .def_readonly("device_id",
                    &DLDevice::device_id,
                    doc::DLDevice::doc_device_id,
                    py::call_guard<py::gil_scoped_release>())
      .def(
          "__repr__",
          [](const DLDevice& device) {
            return fmt::format(
                "<DLDevice device_type:{} device_id:{}>", device.device_type, device.device_id);
          },
          py::call_guard<py::gil_scoped_release>(),
          R"doc(Return repr(self).)doc");

  // Tensor Class
  py::class_<Tensor, std::shared_ptr<Tensor>>(
      m, "Tensor", py::dynamic_attr(), doc::Tensor::doc_Tensor)
      .def(py::init<>(), doc::Tensor::doc_Tensor)
      .def_property_readonly(
          "ndim", &PyTensor::ndim, doc::Tensor::doc_ndim, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly(
          "shape",
          [](const Tensor& tensor) { return vector2pytuple<py::int_>(tensor.shape()); },
          doc::Tensor::doc_shape)
      .def_property_readonly(
          "strides",
          [](const Tensor& tensor) { return vector2pytuple<py::int_>(tensor.strides()); },
          doc::Tensor::doc_strides)
      .def_property_readonly(
          "size", &PyTensor::size, doc::Tensor::doc_size, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("dtype",
                             &PyTensor::dtype,
                             doc::Tensor::doc_dtype,
                             py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("itemsize",
                             &PyTensor::itemsize,
                             doc::Tensor::doc_itemsize,
                             py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("nbytes",
                             &PyTensor::nbytes,
                             doc::Tensor::doc_nbytes,
                             py::call_guard<py::gil_scoped_release>())
      .def_property_readonly(
          "data", &PyTensor::data, doc::Tensor::doc_data, py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("device",
                             &PyTensor::device,
                             doc::Tensor::doc_device,
                             py::call_guard<py::gil_scoped_release>());

  py::class_<PyTensor, Tensor, std::shared_ptr<PyTensor>>(m, "PyTensor", doc::Tensor::doc_Tensor)
      .def_static("as_tensor", &PyTensor::as_tensor, "obj"_a, doc::Tensor::doc_as_tensor);

  // Additional functions with no counterpart in the C++ API
  //    (e.g. helpers for converting Python objects to C++ API Arg/ArgList objects)
  //    These functions are defined in ../kwarg_handling.cpp
  m.def("py_object_to_arg",
        &py_object_to_arg,
        "obj"_a,
        "name"_a = "",
        doc::Core::doc_py_object_to_arg);
  m.def("kwargs_to_arglist", &kwargs_to_arglist, doc::Core::doc_kwargs_to_arglist);
  m.def("arg_to_py_object", &arg_to_py_object, "arg"_a, doc::Core::doc_arg_to_py_object);
  m.def("arglist_to_kwargs", &arglist_to_kwargs, "arglist"_a, doc::Core::doc_arglist_to_kwargs);

  py::enum_<DLDataTypeCode>(m, "DLDataTypeCode", py::module_local())
      .value("DLINT", kDLInt)
      .value("DLUINT", kDLUInt)
      .value("DLFLOAT", kDLFloat)
      .value("DLOPAQUEHANDLE", kDLOpaqueHandle)
      .value("DLBFLOAT", kDLBfloat)
      .value("DLCOMPLEX", kDLComplex);

  py::class_<DLDataType, std::shared_ptr<DLDataType>>(m, "DLDataType", py::module_local())
      .def_readwrite("code", &DLDataType::code)
      .def_readwrite("bits", &DLDataType::bits)
      .def_readwrite("lanes", &DLDataType::lanes)
      .def(
          "__repr__",
          [](const DLDataType& dtype) {
            return fmt::format("<DLDataType: code={}, bits={}, lanes={}>",
                               dldatatypecode_namemap.at(static_cast<DLDataTypeCode>(dtype.code)),
                               dtype.bits,
                               dtype.lanes);
          },
          py::call_guard<py::gil_scoped_release>(),
          R"doc(Return repr(self).)doc");

  // Register argument setter and gxf parameter adaptor for py::object
  register_py_type<py::object>();
}  // PYBIND11_MODULE NOLINT

////////////////////////////////////////////////////////////////////////////////////////////////////
// PyTensor definition
////////////////////////////////////////////////////////////////////////////////////////////////////

py::object PyTensor::as_tensor(const py::object& obj) {
  // This method could have been used as a constructor for the PyTensor class, but it was not
  // possible to get the py::object to be passed to the constructor. Instead, this method is used
  // to create a py::object from PyTensor object and set array interface on it.
  //
  //    // Note: this does not work, as the py::object is not passed to the constructor
  //    .def(py::init(&PyTensor::py_create), doc::Tensor::doc_Tensor);
  //
  //       include/pybind11/detail/init.h:86:19: error: static assertion failed: pybind11::init():
  //       init function must return a compatible pointer, holder, or value
  //       86 |     static_assert(!std::is_same<Class, Class>::value /* always false */,
  //
  //    // See https://github.com/pybind/pybind11/issues/2984 for more details
  std::shared_ptr<PyTensor> tensor;

  if (py::hasattr(obj, "__dlpack__") && py::hasattr(obj, "__dlpack_device__")) {
    tensor = PyTensor::from_dlpack(obj);
  } else if (py::hasattr(obj, "__cuda_array_interface__")) {
    tensor = PyTensor::from_cuda_array_interface(obj);
  } else if (py::hasattr(obj, "__array_interface__")) {
    tensor = PyTensor::from_array_interface(obj);
  } else {
    throw std::runtime_error("Unsupported Python object type");
  }
  py::object py_tensor = py::cast(tensor);

  // Set array interface attributes
  set_array_interface(py_tensor, tensor->dl_ctx());
  // Set dlpack attributes
  set_dlpack_interface(py_tensor, tensor->dl_ctx());
  return py_tensor;
}

std::shared_ptr<PyTensor> PyTensor::from_array_interface(const py::object& obj) {
  auto memory_buf = std::make_shared<ArrayInterfaceMemoryBuffer>();
  memory_buf->obj_ref = obj;  // hold obj to prevent it from being garbage collected

  auto array_interface = obj.attr("__array_interface__").cast<py::dict>();

  // Process mandatory entries
  memory_buf->dl_shape = array_interface["shape"].cast<std::vector<int64_t>>();
  auto& shape = memory_buf->dl_shape;
  auto typestr = array_interface["typestr"].cast<std::string>();
  if (!array_interface.contains("data")) {
    throw std::runtime_error(
        "Array interface data entry is missing (buffer interface) which is not supported ");
  }
  auto data_obj = array_interface["data"];
  if (data_obj.is_none()) {
    throw std::runtime_error(
        "Array interface data entry is None (buffer interface) which is not supported");
  }
  if (!py::isinstance<py::tuple>(data_obj)) {
    throw std::runtime_error(
        "Array interface data entry is not a tuple (buffer interface) which is not supported");
  }
  auto data_array = array_interface["data"].cast<std::vector<int64_t>>();
  auto data_ptr = reinterpret_cast<void*>(data_array[0]);
  bool data_readonly = data_array[1] > 0;
  auto version = array_interface["version"].cast<int64_t>();

  DLTensor local_dl_tensor{
      .data = data_ptr,
      .device = dldevice_from_pointer(data_ptr),
      .ndim = static_cast<int32_t>(shape.size()),
      .dtype = dldatatype_from_typestr(typestr),
      .shape = shape.data(),
      .strides = nullptr,
      .byte_offset = 0,
  };

  // Process 'optional' entries
  py::object strides_obj = py::none();
  if (array_interface.contains("strides")) { strides_obj = array_interface["strides"]; }
  auto& strides = memory_buf->dl_strides;
  if (strides_obj.is_none()) {
    calc_strides(local_dl_tensor, strides, true);
  } else {
    strides = strides_obj.cast<std::vector<int64_t>>();
    // The array interface's stride is using bytes, not element size, so we need to divide it by
    // the element size.
    int64_t elem_size = local_dl_tensor.dtype.bits / 8;
    for (auto& stride : strides) { stride /= elem_size; }
  }
  local_dl_tensor.strides = strides.data();

  // We do not process 'descr', 'mask', and 'offset' entries

  // Create DLManagedTensor object
  auto dl_managed_tensor_ctx = new DLManagedTensorCtx;
  auto& dl_managed_tensor = dl_managed_tensor_ctx->tensor;

  dl_managed_tensor_ctx->memory_ref = memory_buf;

  dl_managed_tensor.manager_ctx = dl_managed_tensor_ctx;
  dl_managed_tensor.deleter = [](DLManagedTensor* self) {
    auto dl_managed_tensor_ctx = static_cast<DLManagedTensorCtx*>(self->manager_ctx);
    // Note: since 'memory_ref' is maintaining python object reference, we should acquire GIL in
    // case this function is called from another non-python thread, before releasing 'memory_ref'.
    py::gil_scoped_acquire scope_guard;
    dl_managed_tensor_ctx->memory_ref.reset();
    delete dl_managed_tensor_ctx;
  };

  // Copy the DLTensor struct data
  DLTensor& dl_tensor = dl_managed_tensor.dl_tensor;
  dl_tensor = local_dl_tensor;

  // Create PyTensor
  std::shared_ptr<PyTensor> tensor = std::make_shared<PyTensor>(&dl_managed_tensor);

  return tensor;
}

std::shared_ptr<PyTensor> PyTensor::from_cuda_array_interface(const py::object& obj) {
  auto memory_buf = std::make_shared<ArrayInterfaceMemoryBuffer>();
  memory_buf->obj_ref = obj;  // hold obj to prevent it from being garbage collected

  auto array_interface = obj.attr("__cuda_array_interface__").cast<py::dict>();

  // Process mandatory entries
  memory_buf->dl_shape = array_interface["shape"].cast<std::vector<int64_t>>();
  auto& shape = memory_buf->dl_shape;
  auto typestr = array_interface["typestr"].cast<std::string>();
  auto data_array = array_interface["data"].cast<std::vector<int64_t>>();
  auto data_ptr = reinterpret_cast<void*>(data_array[0]);
  bool data_readonly = data_array[1] > 0;
  auto version = array_interface["version"].cast<int64_t>();

  DLTensor local_dl_tensor{
      .data = data_ptr,
      .device = dldevice_from_pointer(data_ptr),
      .ndim = static_cast<int32_t>(shape.size()),
      .dtype = dldatatype_from_typestr(typestr),
      .shape = shape.data(),
      .strides = nullptr,
      .byte_offset = 0,
  };

  // Process 'optional' entries
  py::object strides_obj = py::none();
  if (array_interface.contains("strides")) { strides_obj = array_interface["strides"]; }
  auto& strides = memory_buf->dl_strides;
  if (strides_obj.is_none()) {
    calc_strides(local_dl_tensor, strides, true);
  } else {
    strides = strides_obj.cast<std::vector<int64_t>>();
    // The array interface's stride is using bytes, not element size, so we need to divide it by
    // the element size.
    int64_t elem_size = local_dl_tensor.dtype.bits / 8;
    for (auto& stride : strides) { stride /= elem_size; }
  }
  local_dl_tensor.strides = strides.data();

  // We do not process 'descr' and 'mask' entries

  // Process 'stream' entry
  py::object stream_obj = py::none();
  if (array_interface.contains("stream")) { stream_obj = array_interface["stream"]; }

  int64_t stream_id = 1;  // legacy default stream
  cudaStream_t stream_ptr = nullptr;
  if (stream_obj.is_none()) {
    stream_id = -1;
  } else {
    stream_id = stream_obj.cast<int64_t>();
  }
  if (stream_id < -1) {
    throw std::runtime_error(
        "Invalid stream, valid stream should be  None (no synchronization), 1 (legacy default "
        "stream), 2 "
        "(per-thread defaultstream), or a positive integer (stream pointer)");
  } else if (stream_id <= 2) {
    stream_ptr = nullptr;
  } else {
    stream_ptr = reinterpret_cast<cudaStream_t>(stream_id);
  }

  cudaStream_t curr_stream_ptr = nullptr;  // legacy stream

  if (stream_id >= 0 && curr_stream_ptr != stream_ptr) {
    cudaEvent_t curr_stream_event;
    cudaEventCreateWithFlags(&curr_stream_event, cudaEventDisableTiming);
    cudaEventRecord(curr_stream_event, stream_ptr);
    // Make current stream (curr_stream_ptr) to wait until the given stream (stream_ptr)
    // is finished.
    // This is a reverse of py_dlpack() method (in dl_converter.hpp).
    cudaStreamWaitEvent(curr_stream_ptr, curr_stream_event, 0);
    cudaEventDestroy(curr_stream_event);
  }

  // Create DLManagedTensor object
  auto dl_managed_tensor_ctx = new DLManagedTensorCtx;
  auto& dl_managed_tensor = dl_managed_tensor_ctx->tensor;

  dl_managed_tensor_ctx->memory_ref = memory_buf;

  dl_managed_tensor.manager_ctx = dl_managed_tensor_ctx;
  dl_managed_tensor.deleter = [](DLManagedTensor* self) {
    auto dl_managed_tensor_ctx = static_cast<DLManagedTensorCtx*>(self->manager_ctx);
    // Note: since 'memory_ref' is maintaining python object reference, we should acquire GIL in
    // case this function is called from another non-python thread, before releasing 'memory_ref'.
    py::gil_scoped_acquire scope_guard;
    dl_managed_tensor_ctx->memory_ref.reset();
    delete dl_managed_tensor_ctx;
  };

  // Copy the DLTensor struct data
  DLTensor& dl_tensor = dl_managed_tensor.dl_tensor;
  dl_tensor = local_dl_tensor;

  // Create PyTensor
  std::shared_ptr<PyTensor> tensor = std::make_shared<PyTensor>(&dl_managed_tensor);

  return tensor;
}

std::shared_ptr<PyTensor> PyTensor::from_dlpack(const py::object& obj) {
  // Pybind11 doesn't have a way to get/set a pointer with a name so we have to use the C API
  // for efficiency.
  // auto dlpack_capsule = py::reinterpret_borrow<py::capsule>(obj.attr("__dlpack__")());
  auto dlpack_device_func = obj.attr("__dlpack_device__");

  // We don't handle backward compatibility with older versions of DLPack
  if (dlpack_device_func.is_none()) { throw std::runtime_error("DLPack device is not set"); }

  auto dlpack_device = py::cast<py::tuple>(dlpack_device_func());
  // https://dmlc.github.io/dlpack/latest/c_api.html#_CPPv48DLDevice
  DLDeviceType device_type = static_cast<DLDeviceType>(dlpack_device[0].cast<int>());
  int32_t device_id = dlpack_device[1].cast<int32_t>();

  DLDevice device = {device_type, device_id};

  auto dlpack_func = obj.attr("__dlpack__");
  py::capsule dlpack_capsule;

  // TOIMPROVE: need to get current stream pointer and call with the stream
  // https://github.com/dmlc/dlpack/issues/57 this thread was good to understand the differences
  // between __cuda_array_interface__ and __dlpack__ on life cycle/stream handling.
  // In DLPack, the client of the memory notify to the producer that the client will use the
  // client stream (`stream_ptr`) to consume the memory. It's the producer's responsibility to
  // make sure that the client stream wait for the producer stream to finish producing the memory.
  // The producer stream is the stream that the producer used to produce the memory. The producer
  // can then use this information to decide whether to use the same stream to produce the memory
  // or to use a different stream.
  // In __cuda_array_interface__, both producer and consumer are responsible for managing the
  // streams. The producer can use the `stream` field to specify the stream that the producer used
  // to produce the memory. The consumer can use the `stream` field to synchronize with the
  // producer stream. (please see
  // https://numba.readthedocs.io/en/latest/cuda/cuda_array_interface.html#synchronization)
  switch (device_type) {
    case kDLCUDA:
    case kDLCUDAManaged: {
      py::int_ stream_ptr(1);  // legacy stream
      dlpack_capsule = py::reinterpret_borrow<py::capsule>(dlpack_func(stream_ptr));
      break;
    }
    case kDLCPU:
    case kDLCUDAHost: {
      dlpack_capsule = py::reinterpret_borrow<py::capsule>(dlpack_func());
      break;
    }
    default:
      throw std::runtime_error(fmt::format("Unsupported device type: {}", device_type));
  }

  // Note: we should keep the reference to the capsule object (`dlpack_obj`) while working with
  // PyObject* pointer. Otherwise, the capsule can be deleted and the pointers will be invalid.
  py::object dlpack_obj = dlpack_func();

  PyObject* dlpack_capsule_ptr = dlpack_obj.ptr();

  if (!PyCapsule_IsValid(dlpack_capsule_ptr, "dltensor")) {
    const char* capsule_name = PyCapsule_GetName(dlpack_capsule_ptr);
    throw std::runtime_error(
        fmt::format("Received an invalid DLPack capsule ('{}'). You might have already consumed "
                    "the DLPack capsule.",
                    capsule_name));
  }

  DLManagedTensor* dl_managed_tensor =
      static_cast<DLManagedTensor*>(PyCapsule_GetPointer(dlpack_capsule_ptr, "dltensor"));

  // Set device
  dl_managed_tensor->dl_tensor.device = device;

  // Create PyTensor
  std::shared_ptr<PyTensor> tensor = std::make_shared<PyTensor>(dl_managed_tensor);

  // Set the capsule name to 'used_dltensor' so that it will not be consumed again.
  PyCapsule_SetName(dlpack_capsule_ptr, "used_dltensor");

  // Steal the ownership of the capsule so that it will not be destroyed when the capsule object
  // goes out of scope.
  PyCapsule_SetDestructor(dlpack_capsule_ptr, nullptr);

  return tensor;
}

}  // namespace holoscan
