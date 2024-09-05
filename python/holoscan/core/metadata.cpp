/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gil_guarded_pyobject.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/metadata.hpp"
#include "kwarg_handling.hpp"
#include "metadata_pydoc.hpp"

using pybind11::literals::operator""_a;

namespace py = pybind11;

// use a special class to differentiate a default value from Python's None
class _NoneValue {};

namespace holoscan {

void set_scalar_metadata_via_dtype(const py::object& obj, const py::dtype& dt,
                                   MetadataObject& out) {
  std::string dtype_name = dt.attr("name").cast<std::string>();
  if (dtype_name == "float32") {
    out.set_value(obj.cast<float>());
  } else if (dtype_name == "float64") {
    out.set_value(obj.cast<double>());
  } else if (dtype_name == "bool") {
    out.set_value(obj.cast<bool>());
  } else if (dtype_name == "int8") {
    out.set_value(obj.cast<int8_t>());
  } else if (dtype_name == "int16") {
    out.set_value(obj.cast<int16_t>());
  } else if (dtype_name == "int32") {
    out.set_value(obj.cast<int32_t>());
  } else if (dtype_name == "int64") {
    out.set_value(obj.cast<int64_t>());
  } else if (dtype_name == "uint8") {
    out.set_value(obj.cast<uint8_t>());
  } else if (dtype_name == "uint16") {
    out.set_value(obj.cast<uint16_t>());
  } else if (dtype_name == "uint32") {
    out.set_value(obj.cast<uint32_t>());
  } else if (dtype_name == "uint64") {
    out.set_value(obj.cast<uint64_t>());
  } else if (dtype_name == "complex64") {
    out.set_value(obj.cast<std::complex<float>>());
  } else if (dtype_name == "complex128") {
    out.set_value(obj.cast<std::complex<double>>());
  } else {
    throw std::runtime_error("unsupported dtype: "s + dtype_name);
  }
  return;
}

template <typename T>
void set_vector_metadata_via_numpy_array(const py::array& obj, MetadataObject& out) {
  // not intended for images or other large tensors, just
  // for short arrays containing parameter settings to operators
  if (obj.attr("ndim").cast<int>() == 1) {
    std::vector<T> v;
    v.reserve(obj.attr("size").cast<size_t>());
    for (auto item : obj) v.push_back(item.cast<T>());
    out.set_value(v);
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
    out.set_value(v);
  } else {
    throw std::runtime_error("Only 1d and 2d NumPy arrays are supported.");
  }
}

template <typename T>
void set_vector_metadata_via_py_sequence(const py::sequence& seq, MetadataObject& out) {
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
    out.set_value(v);
  } else {
    // 1d vector to handle a sequence of elements
    std::vector<T> v;
    size_t length = py::len(seq);
    v.reserve(length);
    for (auto item : seq) v.push_back(item.cast<T>());
    out.set_value(v);
  }
}

void set_vector_metadata_via_iterable(const py::object& obj, MetadataObject& out) {
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
      set_vector_metadata_via_py_sequence<bool>(seq, out);
    } else if (py::isinstance<py::int_>(item)) {
      set_vector_metadata_via_py_sequence<int64_t>(seq, out);
    } else if (py::isinstance<py::float_>(item)) {
      set_vector_metadata_via_py_sequence<double>(seq, out);
    } else if (py::isinstance<py::str>(item)) {
      set_vector_metadata_via_py_sequence<std::string>(seq, out);
    } else {
      throw std::runtime_error("Nested sequence of unsupported type.");
    }
  } else {
    auto item = item0;
    if (py::isinstance<py::bool_>(item)) {
      set_vector_metadata_via_py_sequence<bool>(seq, out);
    } else if (py::isinstance<py::int_>(item)) {
      set_vector_metadata_via_py_sequence<int64_t>(seq, out);
    } else if (py::isinstance<py::float_>(item)) {
      set_vector_metadata_via_py_sequence<double>(seq, out);
    } else if (py::isinstance<py::str>(item)) {
      set_vector_metadata_via_py_sequence<std::string>(seq, out);
    }
  }
  return;
}

void set_vector_metadata_via_dtype(const py::object& obj, const py::dtype& dt,
                                   MetadataObject& out) {
  std::string dtype_name = dt.attr("name").cast<std::string>();
  if (dtype_name == "float32") {
    set_vector_metadata_via_numpy_array<float>(obj, out);
  } else if (dtype_name == "float64") {
    set_vector_metadata_via_numpy_array<double>(obj, out);
  } else if (dtype_name == "bool") {
    set_vector_metadata_via_numpy_array<bool>(obj, out);
  } else if (dtype_name == "int8") {
    set_vector_metadata_via_numpy_array<int8_t>(obj, out);
  } else if (dtype_name == "int16") {
    set_vector_metadata_via_numpy_array<int16_t>(obj, out);
  } else if (dtype_name == "int32") {
    set_vector_metadata_via_numpy_array<int32_t>(obj, out);
  } else if (dtype_name == "int64") {
    set_vector_metadata_via_numpy_array<int64_t>(obj, out);
  } else if (dtype_name == "uint8") {
    set_vector_metadata_via_numpy_array<uint8_t>(obj, out);
  } else if (dtype_name == "uint16") {
    set_vector_metadata_via_numpy_array<uint16_t>(obj, out);
  } else if (dtype_name == "uint32") {
    set_vector_metadata_via_numpy_array<uint32_t>(obj, out);
  } else if (dtype_name == "uint64") {
    set_vector_metadata_via_numpy_array<uint64_t>(obj, out);
  } else if (dtype_name == "complex64") {
    set_vector_metadata_via_numpy_array<std::complex<float>>(obj, out);
  } else if (dtype_name == "complex128") {
    set_vector_metadata_via_numpy_array<std::complex<double>>(obj, out);
  } else {
    throw std::runtime_error("unsupported dtype: "s + dtype_name);
  }
  return;
}

void py_object_to_metadata_object(MetadataObject& meta_obj, const py::object& value,
                                  const std::optional<py::dtype>& dtype = std::nullopt,
                                  bool cast_to_cpp = false) {
  if (cast_to_cpp) {
    if (py::isinstance<py::str>(value)) {
      meta_obj.set_value(value.cast<std::string>());
    } else if (py::isinstance<py::array>(value)) {
      // handle numpy arrays
      py::dtype array_dtype = value.cast<py::array>().dtype();
      set_vector_metadata_via_dtype(value, array_dtype, meta_obj);
    } else if (py::isinstance<py::iterable>(value) && !py::isinstance<py::dict>(value)) {
      // does not handle every possible type of iterable (e.g. dict)
      // will work for any that can be cast to py::list
      set_vector_metadata_via_iterable(value, meta_obj);
    } else if (py::isinstance<py::bool_>(value)) {
      meta_obj.set_value(value.cast<bool>());
    } else if (py::isinstance<py::int_>(value)) {
      if (dtype.has_value()) {
        set_scalar_metadata_via_dtype(value, dtype.value(), meta_obj);
      } else {
        meta_obj.set_value(value.cast<int64_t>());
      }
    } else if (py::isinstance<py::float_>(value)) {
      if (dtype.has_value()) {
        set_scalar_metadata_via_dtype(value, dtype.value(), meta_obj);
      } else {
        meta_obj.set_value(value.cast<double>());
      }
    } else {
      throw std::runtime_error(
          "Cast to C++ is unsupported for this type. Set cast_to_cpp to false to "
          "send as a Python object");
    }
  } else {
    auto data_ptr = std::make_shared<GILGuardedPyObject>(value);
    meta_obj.set_value(data_ptr);
  }
  return;
}

py::object metadata_obj_to_pyobject(MetadataObject& meta_obj) {
  std::any value = meta_obj.value();
  const auto& id = value.type();
  // Return a Python objects as-is.
  if (id == typeid(std::shared_ptr<GILGuardedPyObject>)) {
    return std::any_cast<std::shared_ptr<GILGuardedPyObject>>(value)->obj();
  }
  // For C++ types, support casting T, vector<T>, and vector<vector<<T>> types
  // where T is either std::string, bool or various integer or floating point types.
  if (id == typeid(std::string)) {
    return py::cast(std::any_cast<std::string>(value));
  } else if (id == typeid(float)) {
    return py::cast(std::any_cast<float>(value));
  } else if (id == typeid(double)) {
    return py::cast(std::any_cast<double>(value));
  } else if (id == typeid(bool)) {
    return py::cast(std::any_cast<bool>(value));
  } else if (id == typeid(int64_t)) {
    return py::cast(std::any_cast<int64_t>(value));
  } else if (id == typeid(uint64_t)) {
    return py::cast(std::any_cast<uint64_t>(value));
  } else if (id == typeid(int32_t)) {
    return py::cast(std::any_cast<int32_t>(value));
  } else if (id == typeid(uint32_t)) {
    return py::cast(std::any_cast<uint32_t>(value));
  } else if (id == typeid(int16_t)) {
    return py::cast(std::any_cast<int16_t>(value));
  } else if (id == typeid(uint16_t)) {
    return py::cast(std::any_cast<uint16_t>(value));
  } else if (id == typeid(int8_t)) {
    return py::cast(std::any_cast<int8_t>(value));
  } else if (id == typeid(uint8_t)) {
    return py::cast(std::any_cast<uint8_t>(value));
  } else if (id == typeid(std::complex<float>)) {
    return py::cast(std::any_cast<std::complex<float>>(value));
  } else if (id == typeid(std::complex<double>)) {
    return py::cast(std::any_cast<std::complex<double>>(value));
  } else if (id == typeid(std::vector<std::string>)) {
    return py::cast(std::any_cast<std::vector<std::string>>(value));
  } else if (id == typeid(std::vector<float>)) {
    return py::cast(std::any_cast<std::vector<float>>(value));
  } else if (id == typeid(std::vector<double>)) {
    return py::cast(std::any_cast<std::vector<double>>(value));
  } else if (id == typeid(std::vector<bool>)) {
    return py::cast(std::any_cast<std::vector<bool>>(value));
  } else if (id == typeid(std::vector<int64_t>)) {
    return py::cast(std::any_cast<std::vector<int64_t>>(value));
  } else if (id == typeid(std::vector<uint64_t>)) {
    return py::cast(std::any_cast<std::vector<uint64_t>>(value));
  } else if (id == typeid(std::vector<int32_t>)) {
    return py::cast(std::any_cast<std::vector<int32_t>>(value));
  } else if (id == typeid(std::vector<uint32_t>)) {
    return py::cast(std::any_cast<std::vector<uint32_t>>(value));
  } else if (id == typeid(std::vector<int16_t>)) {
    return py::cast(std::any_cast<std::vector<int16_t>>(value));
  } else if (id == typeid(std::vector<uint16_t>)) {
    return py::cast(std::any_cast<std::vector<uint16_t>>(value));
  } else if (id == typeid(std::vector<int8_t>)) {
    return py::cast(std::any_cast<std::vector<int8_t>>(value));
  } else if (id == typeid(std::vector<uint8_t>)) {
    return py::cast(std::any_cast<std::vector<uint8_t>>(value));
  } else if (id == typeid(std::vector<std::complex<float>>)) {
    return py::cast(std::any_cast<std::vector<std::complex<float>>>(value));
  } else if (id == typeid(std::vector<std::complex<double>>)) {
    return py::cast(std::any_cast<std::vector<std::complex<double>>>(value));
  } else if (id == typeid(std::vector<std::vector<std::string>>)) {
    return py::cast(std::any_cast<std::vector<std::vector<std::string>>>(value));
  } else if (id == typeid(std::vector<std::vector<float>>)) {
    return py::cast(std::any_cast<std::vector<std::vector<float>>>(value));
  } else if (id == typeid(std::vector<std::vector<double>>)) {
    return py::cast(std::any_cast<std::vector<std::vector<double>>>(value));
  } else if (id == typeid(std::vector<std::vector<bool>>)) {
    return py::cast(std::any_cast<std::vector<std::vector<bool>>>(value));
  } else if (id == typeid(std::vector<std::vector<int64_t>>)) {
    return py::cast(std::any_cast<std::vector<std::vector<int64_t>>>(value));
  } else if (id == typeid(std::vector<std::vector<uint64_t>>)) {
    return py::cast(std::any_cast<std::vector<std::vector<uint64_t>>>(value));
  } else if (id == typeid(std::vector<std::vector<int32_t>>)) {
    return py::cast(std::any_cast<std::vector<std::vector<int32_t>>>(value));
  } else if (id == typeid(std::vector<std::vector<uint32_t>>)) {
    return py::cast(std::any_cast<std::vector<std::vector<uint32_t>>>(value));
  } else if (id == typeid(std::vector<std::vector<int16_t>>)) {
    return py::cast(std::any_cast<std::vector<std::vector<int16_t>>>(value));
  } else if (id == typeid(std::vector<std::vector<uint16_t>>)) {
    return py::cast(std::any_cast<std::vector<std::vector<uint16_t>>>(value));
  } else if (id == typeid(std::vector<std::vector<int8_t>>)) {
    return py::cast(std::any_cast<std::vector<std::vector<int8_t>>>(value));
  } else if (id == typeid(std::vector<std::vector<uint8_t>>)) {
    return py::cast(std::any_cast<std::vector<std::vector<uint8_t>>>(value));
  } else if (id == typeid(std::vector<std::vector<std::complex<float>>>)) {
    return py::cast(std::any_cast<std::vector<std::vector<std::complex<float>>>>(value));
  } else if (id == typeid(std::vector<std::vector<std::complex<double>>>)) {
    return py::cast(std::any_cast<std::vector<std::vector<std::complex<double>>>>(value));
  } else {
    return py::none();
  }
}

void init_metadata(py::module_& m) {
  py::class_<_NoneValue>(m, "_NoneValue").def(py::init<>());

  py::enum_<MetadataPolicy>(m, "MetadataPolicy", doc::MetadataPolicy::doc_MetadataPolicy)
      .value("REJECT", MetadataPolicy::kReject)
      .value("UPDATE", MetadataPolicy::kUpdate)
      .value("RAISE", MetadataPolicy::kRaise);

  // MetadataDictionary provides a Python dict-like interface to the C++ MetadataDictionary.
  py::class_<MetadataDictionary, std::shared_ptr<MetadataDictionary>>(
      m, "MetadataDictionary", doc::MetadataDictionary::doc_MetadataDictionary)
      .def(py::init<>(), doc::MetadataDictionary::doc_MetadataDictionary)
      .def("has_key", &MetadataDictionary::has_key, "key"_a, doc::MetadataDictionary::doc_has_key)
      .def("__contains__", &MetadataDictionary::has_key)
      .def("keys", &MetadataDictionary::keys, doc::MetadataDictionary::doc_keys)
      .def(
          "get",
          [](MetadataDictionary& meta_dict,
             const std::string& key,
             const py::object& default_value = py::none()) -> py::object {
            if (!meta_dict.has_key(key)) { return default_value; }
            auto meta_obj = meta_dict.get(key);
            return metadata_obj_to_pyobject(*meta_obj);
          },
          "key"_a,
          "value"_a = py::none(),
          doc::MetadataDictionary::doc_get)
      .def(
          "__getitem__",
          [](MetadataDictionary& meta_dict, const std::string& key) -> py::object {
            if (!meta_dict.has_key(key)) { throw py::key_error(key); }
            auto meta_obj = meta_dict.get(key);
            return metadata_obj_to_pyobject(*meta_obj);
          },
          "key"_a)
      .def(
          "items",
          [](MetadataDictionary& meta_dict) -> std::vector<std::pair<std::string, py::object>> {
            std::vector<std::pair<std::string, py::object>> items;
            items.reserve(meta_dict.size());
            for (auto& [key, value] : meta_dict) {
              items.push_back({key, metadata_obj_to_pyobject(*value)});
            }
            return items;
          },
          doc::MetadataDictionary::doc_items)
      .def(
          "type_dict",
          [](MetadataDictionary& meta_dict) -> py::dict {
            py::dict type_dict;
            for (auto& [key, v] : meta_dict) {
              type_dict[py::str(key)] = py::str(v->value().type().name());
            }
            return type_dict;
          },
          doc::MetadataDictionary::doc_type_dict)
      .def(
          "pop",
          [](MetadataDictionary& meta_dict,
             const std::string& key,
             const py::object& default_value = py::none()) -> py::object {
            if (!meta_dict.has_key(key)) {
              if (py::isinstance<_NoneValue>(default_value)) {
                throw py::key_error(key);
              } else {
                return default_value;
              }
            }
            auto meta_obj = meta_dict.get(key);
            auto result = metadata_obj_to_pyobject(*meta_obj);
            meta_dict.erase(key);
            return result;
          },
          "key"_a,
          "default"_a = _NoneValue(),
          doc::MetadataDictionary::doc_pop)
      .def(
          "set",
          [](MetadataDictionary& meta_dict,
             const std::string& key,
             py::object& value,
             const std::optional<py::dtype>& dtype = std::nullopt,
             bool cast_to_cpp = false) {
            if (!cast_to_cpp) {
              auto data_ptr = std::make_shared<GILGuardedPyObject>(value);
              meta_dict.set<std::shared_ptr<GILGuardedPyObject>>(key, data_ptr);
            } else {
              auto meta_obj = std::make_shared<MetadataObject>();
              py_object_to_metadata_object(*meta_obj, value, dtype, cast_to_cpp);
              meta_dict.set(key, meta_obj);
            }
          },
          "key"_a,
          "value"_a,
          "dtype"_a = py::none(),
          "cast_to_cpp"_a = false,
          doc::MetadataDictionary::doc_set)
      .def("__setitem__",
           [](MetadataDictionary& meta_dict, const std::string& key, py::object& value) {
             auto data_ptr = std::make_shared<GILGuardedPyObject>(value);
             meta_dict.set<std::shared_ptr<GILGuardedPyObject>>(key, data_ptr);
           })
      .def_property("policy",
                    py::overload_cast<>(&MetadataDictionary::policy, py::const_),
                    py::overload_cast<const MetadataPolicy&>(&MetadataDictionary::policy),
                    doc::MetadataDictionary::doc_policy)
      .def("size", &MetadataDictionary::size, doc::MetadataDictionary::doc_size)
      .def("__len__", &MetadataDictionary::size)
      .def("erase", &MetadataDictionary::erase, "key"_a, doc::MetadataDictionary::doc_erase)
      .def("__delitem__", &MetadataDictionary::erase, "key"_a)
      .def("clear", &MetadataDictionary::clear, doc::MetadataDictionary::doc_clear)
      .def("merge", &MetadataDictionary::merge, "other"_a, doc::MetadataDictionary::doc_merge)
      .def("insert", &MetadataDictionary::insert, "other"_a, doc::MetadataDictionary::doc_insert)
      .def("swap", &MetadataDictionary::swap, "other"_a, doc::MetadataDictionary::doc_swap)
      .def("update", &MetadataDictionary::update, "other"_a, doc::MetadataDictionary::doc_update);
}

}  // namespace holoscan
