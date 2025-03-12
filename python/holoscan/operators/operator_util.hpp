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
#ifndef PYHOLOSCAN_OPERATORS_OPERATOR_UTIL_HPP
#define PYHOLOSCAN_OPERATORS_OPERATOR_UTIL_HPP

#include <pybind11/numpy.h>  // py::dtype, py::array
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // needed for py::cast to work with std::vector types

#include <memory>
#include <string>
#include <vector>

#include "holoscan/core/condition.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/resource.hpp"

namespace py = pybind11;

namespace holoscan {

void add_positional_condition_and_resource_args(Operator* op, const py::args& args) {
  for (auto it = args.begin(); it != args.end(); ++it) {
    if (py::isinstance<Condition>(*it)) {
      op->add_arg(it->cast<std::shared_ptr<Condition>>());
    } else if (py::isinstance<Resource>(*it)) {
      op->add_arg(it->cast<std::shared_ptr<Resource>>());
    } else {
      HOLOSCAN_LOG_WARN(
          "Unhandled positional argument detected (only Condition and Resource objects can be "
          "parsed positionally)");
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// The following methods have been adapted from kwarg_handling.hpp to enhance
// the `kwargs_to_arglist()` method with these updates:
// - Convert `py::object` to `Arg` using YAML::Node type, eliminating the need
//   for a uniform type in the `Arg` class.
//   - Except for `Resource` and `Condition` types, which are handled separately.
//
// `kwargs_to_arglist()` is used in `add_kwargs()` to convert Python kwargs to
// `ArgList` for use in the `ComponentBase::add_arg()` method.
////////////////////////////////////////////////////////////////////////////////

// TODO(gbae): Refactor these utility classes and implementations in kwarg_handling.hpp

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

template <typename T>
void set_vector_arg_via_numpy_array(const py::array& obj, Arg& out) {
  // not intended for images or other large tensors, just
  // for short arrays containing parameter settings to operators/resources
  if (obj.attr("ndim").cast<int>() == 1) {
    YAML::Node yaml_node = YAML::Load("[]");  // Create an empty sequence
    for (const auto& item : obj) yaml_node.push_back(cast_to_yaml_node<T>(item));
    out = yaml_node;
  } else if (obj.attr("ndim").cast<int>() == 2) {
    YAML::Node yaml_node = YAML::Load("[]");  // Create an empty sequence
    for (auto& item : obj) {
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
      for (const auto& item : seq) v.push_back(item.cast<T>());
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
      for (const auto& item : seq) yaml_node.push_back(cast_to_yaml_node<T>(item));
      out = yaml_node;
    }
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

void set_scalar_arg_via_dtype(const py::object& obj, const py::dtype& dt, Arg& out) {
  std::string dtype_name = dt.attr("name").cast<std::string>();
  if (dtype_name == "float16") {  // currently promoting float16 scalars to float
    out = cast_to_yaml_node<float>(obj);
  } else if (dtype_name == "float32") {
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
  return;
}

Arg py_object_to_arg(py::object obj, std::string name = "") {
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
      out = cast_to_yaml_node<double>(obj);
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
    for (auto& [name, handle] : kwargs) {
      arglist.add(py_object_to_arg(handle.cast<py::object>(), name.cast<std::string>()));
    }
    /// .. do something with kwargs
  }
  return arglist;
}

////////////////////////////////////////////////////////////////////////////////

void add_kwargs(ComponentBase* component, const py::kwargs& kwargs) {
  ArgList arg_list = kwargs_to_arglist(kwargs);
  component->add_arg(arg_list);
}

}  // namespace holoscan

#endif /* PYHOLOSCAN_OPERATORS_OPERATOR_UTIL_HPP */
