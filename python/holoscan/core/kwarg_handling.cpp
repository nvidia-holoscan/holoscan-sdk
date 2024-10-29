/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <pybind11/numpy.h>  // py::dtype, py::array
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // needed for py::cast to work with std::vector types

#include <complex>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "holoscan/core/arg.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/condition.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/resource.hpp"
#include "kwarg_handling.hpp"
#include "kwarg_handling_pydoc.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

/// Convert a py::object to a `YAML::Node` type.
template <typename typeT>
inline static YAML::Node cast_to_yaml_node(const py::handle& obj) {
  YAML::Node yaml_node;
  yaml_node.push_back(obj.cast<typeT>());
  return yaml_node[0];
}

// Specialization for uint8_t
template <>
inline YAML::Node cast_to_yaml_node<uint8_t>(const py::handle& obj) {
  YAML::Node yaml_node;
  yaml_node.push_back(obj.cast<int64_t>());
  return yaml_node[0];
}

// Specialization for int8_t
template <>
inline YAML::Node cast_to_yaml_node<int8_t>(const py::handle& obj) {
  YAML::Node yaml_node;
  yaml_node.push_back(obj.cast<int64_t>());
  return yaml_node[0];
}

void set_scalar_arg_via_dtype(const py::object& obj, const py::dtype& dt, Arg& out) {
  auto dtype_name = dt.attr("name").cast<std::string>();
  if (dtype_name == "float16" || dtype_name == "float32") {
    // currently promoting float16 scalars to float
    out = cast_to_yaml_node<float>(obj);
  } else if (dtype_name == "float64") {
    out = cast_to_yaml_node<double>(obj);
  } else if (dtype_name == "bool") {
    out = cast_to_yaml_node<bool>(obj);
  } else if (dtype_name == "int8") {
    out = cast_to_yaml_node<int8_t>(obj);
  } else if (dtype_name == "int16") {
    out = cast_to_yaml_node<int16_t>(obj);
  } else if (dtype_name == "int32") {
    out = cast_to_yaml_node<int32_t>(obj);
  } else if (dtype_name == "int64") {
    out = cast_to_yaml_node<int64_t>(obj);
  } else if (dtype_name == "uint8") {
    out = cast_to_yaml_node<uint8_t>(obj);
  } else if (dtype_name == "uint16") {
    out = cast_to_yaml_node<uint16_t>(obj);
  } else if (dtype_name == "uint32") {
    out = cast_to_yaml_node<uint32_t>(obj);
  } else if (dtype_name == "uint64") {
    out = cast_to_yaml_node<uint64_t>(obj);
  } else {
    throw std::runtime_error("unsupported dtype: "s + dtype_name + ", leaving Arg uninitialized"s);
  }
}

template <typename T>
void set_vector_arg_via_numpy_array(const py::array& obj, Arg& out) {
  // not intended for images or other large tensors, just
  // for short arrays containing parameter settings to operators/resources
  if (obj.attr("ndim").cast<int>() == 1) {
    YAML::Node yaml_node = YAML::Load("[]");  // Create an empty sequence
    for (const auto& item : obj) { yaml_node.push_back(cast_to_yaml_node<T>(item)); }
    out = yaml_node;
  } else if (obj.attr("ndim").cast<int>() == 2) {
    YAML::Node yaml_node = YAML::Load("[]");  // Create an empty sequence
    for (const auto& item : obj) {
      YAML::Node inner_yaml_node = YAML::Load("[]");  // Create an empty sequence
      for (const auto& inner_item : item) {
        inner_yaml_node.push_back(cast_to_yaml_node<T>(inner_item));
      }
      if (inner_yaml_node.size() > 0) { yaml_node.push_back(inner_yaml_node); }
    }
    out = yaml_node;
  } else {
    throw std::runtime_error("Only 1d and 2d NumPy arrays are supported.");
  }
}

// NOLINTBEGIN(readability-function-cognitive-complexity)
template <typename T>
void set_vector_arg_via_py_sequence(const py::sequence& seq, Arg& out) {
  // not intended for images or other large tensors, just
  // for short arrays containing parameter settings to operators/resources

  if constexpr (std::is_same_v<T, std::shared_ptr<Resource>> ||
                std::is_same_v<T, std::shared_ptr<Condition>>) {
    auto first_item = seq[0];
    if (py::isinstance<py::sequence>(first_item) && !py::isinstance<py::str>(first_item)) {
      // Handle list of list and other sequence of sequence types.
      std::vector<std::vector<T>> v;
      v.reserve(static_cast<size_t>(py::len(seq)));
      for (const auto& item : seq) {
        std::vector<T> vv;
        vv.reserve(static_cast<size_t>(py::len(item)));
        for (const auto& inner_item : item) { vv.push_back(inner_item.cast<T>()); }
        v.push_back(vv);
      }
      out = v;
    } else {
      // 1d vector to handle a sequence of elements
      std::vector<T> v;
      size_t length = py::len(seq);
      v.reserve(length);
      for (const auto& item : seq) { v.push_back(item.cast<T>()); }
      out = v;
    }
  } else {
    auto first_item = seq[0];
    if (py::isinstance<py::sequence>(first_item) && !py::isinstance<py::str>(first_item)) {
      // Handle list of list and other sequence of sequence types.
      YAML::Node yaml_node = YAML::Load("[]");  // Create an empty sequence
      for (const auto& item : seq) {
        YAML::Node inner_yaml_node = YAML::Load("[]");  // Create an empty sequence
        for (const auto& inner_item : item) {
          inner_yaml_node.push_back(cast_to_yaml_node<T>(inner_item));
        }
        if (inner_yaml_node.size() > 0) { yaml_node.push_back(inner_yaml_node); }
      }
      out = yaml_node;
    } else {
      // 1d vector to handle a sequence of elements
      YAML::Node yaml_node = YAML::Load("[]");  // Create an empty sequence
      for (const auto& item : seq) { yaml_node.push_back(cast_to_yaml_node<T>(item)); }
      out = yaml_node;
    }
  }
}
// NOLINTEND(readability-function-cognitive-complexity)

void set_vector_arg_via_iterable(const py::object& obj, Arg& out) {
  py::sequence seq;
  if (py::isinstance<py::sequence>(obj)) {
    seq = obj;
  } else {
    // convert other iterables to a list first
    seq = py::list(obj);
  }

  if (py::len(seq) == 0) { throw std::runtime_error("sequences of length 0 are not supported."); }

  auto item0 = seq[0];
  if (py::isinstance<py::sequence>(item0) && !py::isinstance<py::str>(item0)) {
    py::sequence inner_seq = item0;
    if (py::len(inner_seq) == 0) {
      throw std::runtime_error("sequences of length 0 are not supported.");
    }
    auto item = inner_seq[0];
    if (py::isinstance<py::sequence>(item) && !py::isinstance<py::str>(item)) {
      throw std::runtime_error("Nested sequences of depth > 2 levels are not supported.");
    }
    if (py::isinstance<py::bool_>(item)) {
      set_vector_arg_via_py_sequence<bool>(seq, out);
    } else if (py::isinstance<py::int_>(item)) {
      set_vector_arg_via_py_sequence<int64_t>(seq, out);
    } else if (py::isinstance<py::float_>(item)) {
      set_vector_arg_via_py_sequence<double>(seq, out);
    } else if (py::isinstance<py::str>(item)) {
      set_vector_arg_via_py_sequence<std::string>(seq, out);
    } else {
      throw std::runtime_error("Nested sequence of unsupported type.");
    }
  } else {
    const auto& item = item0;
    if (py::isinstance<py::bool_>(item)) {
      set_vector_arg_via_py_sequence<bool>(seq, out);
    } else if (py::isinstance<py::int_>(item)) {
      set_vector_arg_via_py_sequence<int64_t>(seq, out);
    } else if (py::isinstance<py::float_>(item)) {
      set_vector_arg_via_py_sequence<double>(seq, out);
    } else if (py::isinstance<py::str>(item)) {
      set_vector_arg_via_py_sequence<std::string>(seq, out);
    } else if (py::isinstance<Resource>(item)) {
      set_vector_arg_via_py_sequence<std::shared_ptr<Resource>>(seq, out);
    } else if (py::isinstance<Condition>(item)) {
      set_vector_arg_via_py_sequence<std::shared_ptr<Condition>>(seq, out);
    }
  }
}

void set_vector_arg_via_dtype(const py::object& obj, const py::dtype& dt, Arg& out) {
  auto dtype_name = dt.attr("name").cast<std::string>();
  if (dtype_name == "float16" || dtype_name == "float32") {
    // currently promoting float16 scalars to float
    set_vector_arg_via_numpy_array<float>(obj, out);
  } else if (dtype_name == "float64") {
    set_vector_arg_via_numpy_array<double>(obj, out);
  } else if (dtype_name == "bool") {
    set_vector_arg_via_numpy_array<bool>(obj, out);
  } else if (dtype_name == "int8") {
    set_vector_arg_via_numpy_array<int8_t>(obj, out);
  } else if (dtype_name == "int16") {
    set_vector_arg_via_numpy_array<int16_t>(obj, out);
  } else if (dtype_name == "int32") {
    set_vector_arg_via_numpy_array<int32_t>(obj, out);
  } else if (dtype_name == "int64") {
    set_vector_arg_via_numpy_array<int64_t>(obj, out);
  } else if (dtype_name == "uint8") {
    set_vector_arg_via_numpy_array<uint8_t>(obj, out);
  } else if (dtype_name == "uint16") {
    set_vector_arg_via_numpy_array<uint16_t>(obj, out);
  } else if (dtype_name == "uint32") {
    set_vector_arg_via_numpy_array<uint32_t>(obj, out);
  } else if (dtype_name == "uint64") {
    set_vector_arg_via_numpy_array<uint64_t>(obj, out);
  } else if (dtype_name.find("str") == 0) {  // NOLINT(abseil-string-find-startswith)
    auto list_obj = obj.attr("tolist")().cast<py::list>();
    // TODO(grelee): set_vector_arg_via_seqeuence(list_obj, out);
  } else {
    throw std::runtime_error("unsupported dtype: "s + dtype_name + ", leaving Arg uninitialized"s);
  }
}

template <typename T>
py::object vector_arg_to_py_object(Arg& arg) {
  try {  // 1d:  std::vector<T>
    return py::cast(std::any_cast<std::vector<T>>(arg.value()));
  } catch (const std::bad_cast& e) {  // 2d:  std::vector<std::vector<T>>
    return py::cast(std::any_cast<std::vector<std::vector<T>>>(arg.value()));
  }
}

// NOLINTBEGIN(misc-no-recursion)
py::object yaml_node_to_py_object(const YAML::Node& node) {
  if (node.IsSequence()) {
    py::list list;
    for (const auto& item : node) { list.append(yaml_node_to_py_object(item)); }
    return list;
  }
  if (node.IsMap()) {
    py::dict dict;
    for (const auto& item : node) {
      dict[py::str(item.first.as<std::string>())] = yaml_node_to_py_object(item.second);
    }
    return dict;
  }
  if (node.IsScalar()) {
    // Check if it is null.
    if (node.IsNull()) { return py::none(); }
    // Check if it is an integer.
    {
      int64_t t{};
      if (YAML::convert<int64_t>::decode(node, t)) { return py::int_(t); }
    }
    // Check if it is a float.
    {
      double t{};
      if (YAML::convert<double>::decode(node, t)) { return py::float_(t); }
    }
    // Check if it is a boolean.
    {
      bool t{};
      if (YAML::convert<bool>::decode(node, t)) { return py::bool_(t); }
    }
    // Check if it is a string.
    {
      std::string t{};
      if (YAML::convert<std::string>::decode(node, t)) { return py::str(t); }
    }
  }
  return py::none();
}
// NOLINTEND(misc-no-recursion)

py::object arg_to_py_object(Arg& arg) {
  // Takes an Arg as input and returns an appropriate Python object equivalent.
  py::object out;
  auto t = arg.arg_type();
  auto container_type = t.container_type();
  auto element_type = t.element_type();
  // NOLINTBEGIN(clang-diagnostic-switch)
  if (container_type == ArgContainerType::kNative) {
    switch (element_type) {
      case ArgElementType::kBoolean:
        return py::cast(std::any_cast<bool>(arg.value()));
      case ArgElementType::kFloat32:
        return py::cast(std::any_cast<float>(arg.value()));
      case ArgElementType::kFloat64:
        return py::cast(std::any_cast<double>(arg.value()));
      case ArgElementType::kInt8:
        return py::cast(std::any_cast<int8_t>(arg.value()));
      case ArgElementType::kInt16:
        return py::cast(std::any_cast<int16_t>(arg.value()));
      case ArgElementType::kInt32:
        return py::cast(std::any_cast<int32_t>(arg.value()));
      case ArgElementType::kInt64:
        return py::cast(std::any_cast<int64_t>(arg.value()));
      case ArgElementType::kUnsigned8:
        return py::cast(std::any_cast<uint8_t>(arg.value()));
      case ArgElementType::kUnsigned16:
        return py::cast(std::any_cast<uint16_t>(arg.value()));
      case ArgElementType::kUnsigned32:
        return py::cast(std::any_cast<uint32_t>(arg.value()));
      case ArgElementType::kUnsigned64:
        return py::cast(std::any_cast<uint64_t>(arg.value()));
      case ArgElementType::kComplex64:
        return py::cast(std::any_cast<std::complex<float>>(arg.value()));
      case ArgElementType::kComplex128:
        return py::cast(std::any_cast<std::complex<double>>(arg.value()));
      case ArgElementType::kString:
        return py::cast(std::any_cast<std::string>(arg.value()));
      case ArgElementType::kYAMLNode: {
        auto node = std::any_cast<YAML::Node>(arg.value());
        return yaml_node_to_py_object(node);
      }
    }
    // Not handled here: kHandle, kCustom, kIOSpec, kCondition, kResource
  } else if (container_type == ArgContainerType::kVector) {
    switch (element_type) {
      case ArgElementType::kBoolean:
        return vector_arg_to_py_object<bool>(arg);
      case ArgElementType::kFloat32:
        return vector_arg_to_py_object<float>(arg);
      case ArgElementType::kFloat64:
        return vector_arg_to_py_object<double>(arg);
      case ArgElementType::kInt8:
        return vector_arg_to_py_object<int8_t>(arg);
      case ArgElementType::kInt16:
        return vector_arg_to_py_object<int16_t>(arg);
      case ArgElementType::kInt32:
        return vector_arg_to_py_object<int32_t>(arg);
      case ArgElementType::kInt64:
        return vector_arg_to_py_object<int64_t>(arg);
      case ArgElementType::kUnsigned8:
        return vector_arg_to_py_object<uint8_t>(arg);
      case ArgElementType::kUnsigned16:
        return vector_arg_to_py_object<uint16_t>(arg);
      case ArgElementType::kUnsigned32:
        return vector_arg_to_py_object<uint32_t>(arg);
      case ArgElementType::kUnsigned64:
        return vector_arg_to_py_object<uint64_t>(arg);
      case ArgElementType::kComplex64:
        return vector_arg_to_py_object<std::complex<float>>(arg);
      case ArgElementType::kComplex128:
        return vector_arg_to_py_object<std::complex<double>>(arg);
      case ArgElementType::kString:
        return vector_arg_to_py_object<std::string>(arg);
    }
    // Not handled here: kHandle, kCustom, kIOSpec, kCondition, kResource, kYAMLNode
  }
  // NOLINTEND(clang-diagnostic-switch)

  throw std::runtime_error(fmt::format(
      "Unable to convert Arg (name: {}, container_type: {}, element_type: {}) to Python object",
      arg.name(),
      static_cast<int>(container_type),
      static_cast<int>(element_type)));
}

Arg py_object_to_arg(py::object obj, const std::string& name = ""s) {
  Arg out(name);
  if (py::isinstance<py::str>(obj)) {
    out = cast_to_yaml_node<std::string>(obj);
  } else if (py::isinstance<py::array>(obj)) {
    // handle numpy arrays
    py::dtype array_dtype = obj.cast<py::array>().dtype();
    set_vector_arg_via_dtype(obj, array_dtype, out);
    return out;
  } else if (py::isinstance<py::iterable>(obj) && !py::isinstance<py::dict>(obj)) {
    // does not handle every possible type of iterable (e.g. dict)
    // will work for any that can be cast to py::list
    set_vector_arg_via_iterable(obj, out);
  } else if (py::isinstance<py::bool_>(obj)) {
    out = cast_to_yaml_node<bool>(obj);
  } else if (py::isinstance<py::int_>(obj) || PyLong_Check(obj.ptr())) {
    out = cast_to_yaml_node<int64_t>(obj);
  } else if (py::isinstance<py::float_>(obj)) {
    out = cast_to_yaml_node<double>(obj);
  } else if (PyComplex_Check(obj.ptr())) {
    throw std::runtime_error("complex value cannot be converted to Arg");
  } else if (PyNumber_Check(obj.ptr()) == 1) {
    py::module_ np = py::module_::import("numpy");
    auto numpy_generic = np.attr("generic");
    if (py::isinstance(obj, numpy_generic)) {
      // cast numpy scalars to appropriate dtype
      py::dtype dt = np.attr("dtype")(obj);
      set_scalar_arg_via_dtype(obj, dt, out);
      return out;
    }
    // cast any other unknown numeric type to double
    out = cast_to_yaml_node<double>(obj);
  } else if (py::isinstance<Resource>(obj)) {
    out = obj.cast<std::shared_ptr<Resource>>();
  } else if (py::isinstance<Condition>(obj)) {
    out = obj.cast<std::shared_ptr<Condition>>();
  } else {
    throw std::runtime_error("python object could not be converted to Arg");
  }
  return out;
}

ArgList kwargs_to_arglist(const py::kwargs& kwargs) {
  // Note: scalars will be kNative while any iterables will have type kNative.
  //       There is currently no option to choose conversion to kArray instead of kNative.
  ArgList arglist;
  if (kwargs) {
    for (const auto& [name, handle] : kwargs) {
      arglist.add(py_object_to_arg(handle.cast<py::object>(), name.cast<std::string>()));
    }
    /// .. do something with kwargs
  }
  return arglist;
}

py::dict arglist_to_kwargs(ArgList& arglist) {
  py::dict d;
  for (Arg& arg : arglist.args()) {
    py::object obj = arg_to_py_object(arg);
    d[arg.name().c_str()] = obj;
  }
  return d;
}

void init_kwarg_handling(py::module_& m) {
  // Additional functions with no counterpart in the C++ API
  //    (e.g. helpers for converting Python objects to C++ API Arg/ArgList objects)
  //    These functions are defined in ../kwarg_handling.cpp
  m.def("py_object_to_arg",
        &py_object_to_arg,
        "obj"_a,
        "name"_a = "",
        doc::KwargHandling::doc_py_object_to_arg);
  m.def("kwargs_to_arglist", &kwargs_to_arglist, doc::KwargHandling::doc_kwargs_to_arglist);
  m.def("arg_to_py_object", &arg_to_py_object, "arg"_a, doc::KwargHandling::doc_arg_to_py_object);
  m.def("arglist_to_kwargs",
        &arglist_to_kwargs,
        "arglist"_a,
        doc::KwargHandling::doc_arglist_to_kwargs);
}

}  // namespace holoscan
