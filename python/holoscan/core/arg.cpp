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
#include <pybind11/stl.h>

#include <cstdint>
#include <string>

#include "arg_pydoc.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/argument_setter.hpp"
#include "holoscan/core/executors/gxf/gxf_parameter_adaptor.hpp"
#include "kwarg_handling.hpp"

using pybind11::literals::operator""_a;  // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

template <typename typeT>
static void register_py_type() {
  auto& arg_setter = ArgumentSetter::get_instance();
  arg_setter.add_argument_setter<typeT>([](ParameterWrapper& param_wrap, Arg& arg) {
    std::any& any_param = param_wrap.value();
    // Note that the type of any_param is Parameter<typeT>*, not Parameter<typeT>.
    auto& param = *std::any_cast<Parameter<typeT>*>(any_param);
    // Acquire GIL because this method updates Python objects and this method is called when the
    // holoscan::gxf::GXFExecutor::run() method is called and we don't acquire GIL for the run()
    // method to allow the Python operator's compute() method calls to be concurrent.
    // (Otherwise, run() method calls GxfGraphWait() which blocks the Python thread.)
    py::gil_scoped_acquire scope_guard;

    // Get the operator object from the parameter (operator object is stored as a weakref object
    // in `PyOperatorSpec::py_params_` to prevent cyclic references).
    py::object op_obj = param.get()();  // call the object to get actual object
    if (op_obj.is_none()) {
      throw std::runtime_error(
          fmt::format("operator weak reference is no longer valid while accessing parameter '{}'",
                      param.key()));
    }

    // If arg has no name and value, that indicates that we want to set the default value for
    // the native operator if it is not specified.
    if (arg.name().empty() && !arg.has_value()) {
      const char* key = param.key().c_str();
      // If the attribute is not None, we do not need to set it.
      // Otherwise, we set it to the default value (if it exists).
      if (!py::hasattr(op_obj, key) ||
          (py::getattr(op_obj, key).is(py::none()) && param.has_default_value())) {
        py::setattr(op_obj, key, param.default_value());
      }
      return;
    }

    // `PyOperatorSpec.py_param` will have stored the actual Operator class from Python
    // into a Parameter<py::object> param (as a weakref object), so `op_obj` here is
    // `PyOperatorSpec::py_op_`.
    // `param.key()` is the name of the attribute on the Python class.
    // We can then set that Class attribute's value to the value contained in Arg.
    // Here we use the `arg_to_py_object` utility to convert from `Arg` to `py::object`.
    py::setattr(op_obj, param.key().c_str(), arg_to_py_object(arg));
  });
}

void init_arg(py::module_& m) {
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
      .def_property("name",
                    py::overload_cast<>(&Arg::name, py::const_),
                    py::overload_cast<const std::string&>(&Arg::name),
                    doc::Arg::doc_name)
      .def_property_readonly("arg_type", &Arg::arg_type, doc::Arg::doc_arg_type)
      .def_property_readonly("has_value", &Arg::has_value, doc::Arg::doc_has_value)
      // std::any& value()
      .def("__int__",
           [](Arg& arg) -> py::object {
             auto result = arg_to_py_object(arg);
             if (!result.is_none()) {
               if (py::isinstance<py::int_>(result)) {
                 return result;
               }
               if (py::isinstance<py::float_>(result)) {
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
               }
               if (py::isinstance<py::int_>(result)) {
                 return py::float_(static_cast<double>(result.cast<int64_t>()));
               }
             }
             return py::none();
           })
      .def("__bool__",
           [](Arg& arg) -> py::object {
             auto result = arg_to_py_object(arg);
             if (!result.is_none()) {
               return py::bool_(py::cast<bool>(result));
             }
             return py::none();
           })
      .def("__str__",
           [](Arg& arg) -> py::object {
             auto result = arg_to_py_object(arg);
             if (py::isinstance<py::str>(result)) {
               return result;
             }
             if (!result.is_none()) {
               return py::str(result);
             }
             return py::str();
           })
      .def_property_readonly("description", &Arg::description, doc::Arg::doc_description)
      .def(
          "__repr__",
          // use py::object and obj.cast to avoid a segfault if object has not been initialized
          [](const Arg& arg) { return arg.description(); },
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
          // use py::object and obj.cast to avoid a segfault if object has not been initialized
          [](const ArgList& list) { return list.description(); },
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
          // use py::object and obj.cast to avoid a segfault if object has not been initialized
          [](const ArgType& t) { return t.to_string(); },
          R"doc(Return repr(self).)doc");
  // Register argument setter and gxf parameter adaptor for py::object
  register_py_type<py::object>();

  // Note: currently not wrapping ArgumentSetter as it was not needed from Python
  // Note: currently not wrapping GXFParameterAdaptor as it was not needed from Python
  // Note: currently not wrapping individual MetaParameter class templates
}

}  // namespace holoscan
