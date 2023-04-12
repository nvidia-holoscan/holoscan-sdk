/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>  // py::dtype, py::array
#include <pybind11/stl.h>       // needed for py::cast to work with std::vector types

#include <cstdint>
#include <memory>
#include <string>

#include "kwarg_handling.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/condition.hpp"
#include "holoscan/core/io_spec.hpp"
#include "holoscan/core/resource.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

namespace py = pybind11;

namespace holoscan {

void set_scalar_arg_via_dtype(const py::object& obj, const py::dtype& dt, Arg& out) {
  std::string dtype_name = dt.attr("name").cast<std::string>();
  if (dtype_name == "float16") {  // currently promoting float16 scalars to float
    out = obj.cast<float>();
  } else if (dtype_name == "float32") {
    out = obj.cast<float>();
  } else if (dtype_name == "float64") {
    out = obj.cast<double>();
  } else if (dtype_name == "bool") {
    out = obj.cast<bool>();
  } else if (dtype_name == "int8") {
    out = obj.cast<int8_t>();
  } else if (dtype_name == "int16") {
    out = obj.cast<int16_t>();
  } else if (dtype_name == "int32") {
    out = obj.cast<int32_t>();
  } else if (dtype_name == "int64") {
    out = obj.cast<int64_t>();
  } else if (dtype_name == "uint8") {
    out = obj.cast<uint8_t>();
  } else if (dtype_name == "uint16") {
    out = obj.cast<uint16_t>();
  } else if (dtype_name == "uint32") {
    out = obj.cast<uint32_t>();
  } else if (dtype_name == "uint64") {
    out = obj.cast<uint64_t>();
  } else {
    throw std::runtime_error("unsupported dtype: "s + dtype_name + ", leaving Arg uninitialized"s);
  }
  return;
}

template <typename T>
void set_vector_arg_via_numpy_array(const py::array& obj, Arg& out) {
  // not intended for images or other large tensors, just
  // for short arrays containing parameter settings to operators
  if (obj.attr("ndim").cast<int>() == 1) {
    std::vector<T> v;
    v.reserve(obj.attr("size").cast<size_t>());
    for (auto item : obj) v.push_back(item.cast<T>());
    out = v;
  } else if (obj.attr("ndim").cast<int>() == 2) {
    std::vector<std::vector<T>> v;
    std::vector<py::ssize_t> shape = obj.attr("shape").cast<std::vector<py::ssize_t>>();
    v.reserve(static_cast<size_t>(shape[0]));
    for (auto item : obj) {
      std::vector<T> vv;
      vv.reserve(static_cast<size_t>(shape[1]));
      for (auto inner_item : item) { vv.push_back(inner_item.cast<T>()); }
      v.push_back(vv);
    }
    out = v;
  } else {
    throw std::runtime_error("Only 1d and 2d NumPy arrays are supported.");
  }
}

template <typename T>
void set_vector_arg_via_py_sequence(const py::sequence& seq, Arg& out) {
  // not intended for images or other large tensors, just
  // for short arrays containing parameter settings to operators

  auto first_item = seq[0];
  if (py::isinstance<py::sequence>(first_item) && !py::isinstance<py::str>(first_item)) {
    // Handle list of list and other sequence of sequence types.
    std::vector<std::vector<T>> v;
    v.reserve(static_cast<size_t>(py::len(seq)));
    for (auto item : seq) {
      std::vector<T> vv;
      vv.reserve(static_cast<size_t>(py::len(item)));
      for (auto inner_item : item) { vv.push_back(inner_item.cast<T>()); }
      v.push_back(vv);
    }
    out = v;
  } else {
    // 1d vector to handle a sequence of elements
    std::vector<T> v;
    size_t length = py::len(seq);
    v.reserve(length);
    for (auto item : seq) v.push_back(item.cast<T>());
    out = v;
  }
}

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
    auto item = item0;
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
  return;
}

void set_vector_arg_via_dtype(const py::object& obj, const py::dtype& dt, Arg& out) {
  std::string dtype_name = dt.attr("name").cast<std::string>();
  if (dtype_name == "float16") {  // currently promoting float16 scalars to float
    set_vector_arg_via_numpy_array<float>(obj, out);
  } else if (dtype_name == "float32") {
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
  } else if (dtype_name.find("str") == 0) {
    py::list list_obj = obj.attr("tolist")().cast<py::list>();
    // TODO(grelee): set_vector_arg_via_seqeuence(list_obj, out);
  } else {
    throw std::runtime_error("unsupported dtype: "s + dtype_name + ", leaving Arg uninitialized"s);
  }
  return;
}

template <typename T>
py::object vector_arg_to_py_object(Arg& arg) {
  try {  // 1d:  std::vector<T>
    return py::cast(std::any_cast<std::vector<T>>(arg.value()));
  } catch (const std::bad_cast& e) {  // 2d:  std::vector<std::vector<T>>
    return py::cast(std::any_cast<std::vector<std::vector<T>>>(arg.value()));
  }
}

py::object yaml_node_to_py_object(YAML::Node node) {
  if (node.IsSequence()) {
    py::list list;
    for (auto item : node) { list.append(yaml_node_to_py_object(item)); }
    return list;
  } else if (node.IsMap()) {
    py::dict dict;
    for (auto item : node) {
      dict[py::str(item.first.as<std::string>())] = yaml_node_to_py_object(item.second);
    }
    return dict;
  } else if (node.IsScalar()) {
    // Check if it is null.
    if (node.IsNull()) { return py::none(); }
    // Check if it is an integer.
    {
      int64_t t;
      if (YAML::convert<int64_t>::decode(node, t)) { return py::int_(t); }
    }
    // Check if it is a float.
    {
      double t;
      if (YAML::convert<double>::decode(node, t)) { return py::float_(t); }
    }
    // Check if it is a boolean.
    {
      bool t;
      if (YAML::convert<bool>::decode(node, t)) { return py::bool_(t); }
    }
    // Check if it is a string.
    {
      std::string t;
      if (YAML::convert<std::string>::decode(node, t)) { return py::str(t); }
    }
  }
  return py::none();
}

py::object arg_to_py_object(Arg& arg) {
  // Takes an Arg as input and returns an appropriate Python object equivalent.
  py::object out;
  auto t = arg.arg_type();
  auto container_type = t.container_type();
  auto element_type = t.element_type();
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
      case ArgElementType::kString:
        return vector_arg_to_py_object<std::string>(arg);
    }
    // Not handled here: kHandle, kCustom, kIOSpec, kCondition, kResource, kYAMLNode
  }

  throw std::runtime_error(fmt::format(
      "Unable to convert Arg (name: {}, container_type: {}, element_type: {}) to Python object",
      arg.name(),
      static_cast<int>(container_type),
      static_cast<int>(element_type)));
}

Arg py_object_to_arg(py::object obj, std::string name = "") {
  Arg out(name);
  if (py::isinstance<py::str>(obj)) {
    out = obj.cast<std::string>();
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
    out = obj.cast<bool>();
  } else if (py::isinstance<py::int_>(obj) || PyLong_Check(obj.ptr())) {
    out = obj.cast<int64_t>();
  } else if (py::isinstance<py::float_>(obj)) {
    out = obj.cast<double>();
  } else if (PyComplex_Check(obj.ptr())) {
    throw std::runtime_error("complex value cannot be converted to Arg");
  } else if (PyNumber_Check(obj.ptr())) {
    py::module_ np = py::module_::import("numpy");
    auto numpy_generic = np.attr("generic");
    if (py::isinstance(obj, numpy_generic)) {
      // cast numpy scalars to appropriate dtype
      py::dtype dt = np.attr("dtype")(obj);
      set_scalar_arg_via_dtype(obj, dt, out);
      return out;
    } else {
      // cast any other unknown numeric type to double
      out = obj.cast<double>();
    }
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
    for (auto& item : kwargs) {
      // item.first is the argument name
      // item.second has type py::handle so we cast it to py::object
      arglist.add(py_object_to_arg(item.second.cast<py::object>(), item.first.cast<std::string>()));
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

}  // namespace holoscan
