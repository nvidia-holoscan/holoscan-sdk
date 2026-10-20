/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "schema_value.hpp"

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>

#include "dl_converter.hpp"
#include "errors.hpp"
#include "imported_tensor.hpp"
#include "payload.hpp"

namespace holoscan {
namespace {

// Generated schema values nest far less deeply than this. The FlatBuffers verifier accepts at most
// 64 nested tables and vectors, and each of those takes at most a few converter levels. CPython's
// own C recursion limit is sized for its much smaller frames, so on its own it can let this
// converter exhaust a default 8 MiB thread stack before it raises.
constexpr int kMaxSchemaValueDepth = 256;

// Bounds recursive Python container and generated-dataclass conversion, raising RecursionError.
class PythonRecursionGuard {
 public:
  PythonRecursionGuard() {
    if (depth() >= kMaxSchemaValueDepth) {
      PyErr_Format(PyExc_RecursionError,
                   "maximum generated Holoscan schema nesting depth (%d) exceeded",
                   kMaxSchemaValueDepth);
      throw nb::python_error();
    }
    if (Py_EnterRecursiveCall(" while converting a generated Holoscan schema value") != 0) {
      throw nb::python_error();
    }
    ++depth();
  }

  ~PythonRecursionGuard() {
    --depth();
    Py_LeaveRecursiveCall();
  }

  PythonRecursionGuard(const PythonRecursionGuard&) = delete;
  PythonRecursionGuard& operator=(const PythonRecursionGuard&) = delete;

 private:
  [[nodiscard]] static int& depth() noexcept {
    thread_local int value{};
    return value;
  }
};

// Use the stable ABI's checked setters; each freshly allocated list slot steals one reference.
template <typename Convert>
nb::list python_schema_list(std::size_t size, Convert convert) {
  auto result = nb::steal<nb::list>(PyList_New(static_cast<Py_ssize_t>(size)));
  if (!result.is_valid()) {
    throw nb::python_error();
  }
  for (std::size_t index = 0; index < size; ++index) {
    nb::object item = convert(index);
    if (PyList_SetItem(result.ptr(), static_cast<Py_ssize_t>(index), item.release().ptr()) != 0) {
      throw nb::python_error();
    }
  }
  return result;
}

}  // namespace

// This recursive converter keeps the Python type precedence explicit (notably bool before int)
// and attaches schema-specific diagnostics at the point each representation is recognized.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
authoring::ForeignSchemaValue python_schema_value(nb::handle value) {
  const PythonRecursionGuard recursion_guard;
  if (value.is_none()) {
    return {};
  }
  if (nb::isinstance<nb::bool_>(value)) {
    return authoring::ForeignSchemaValue::boolean(value.ptr() == Py_True);
  }
  if (PyLong_Check(value.ptr()) != 0) {
    const nb::int_ zero{0};
    const int is_negative = PyObject_RichCompareBool(value.ptr(), zero.ptr(), Py_LT);
    if (is_negative < 0) {
      throw nb::python_error();
    }
    if (is_negative != 0) {
      const long long converted = PyLong_AsLongLong(value.ptr());  // NOLINT(runtime/int)
      if (converted == -1 && PyErr_Occurred() != nullptr) {
        PyErr_Clear();
        throw RuntimeError{
            Error{ErrorCode::kInvalidArgument, "PYTHON_SCHEMA_FIELD_VALUE_OUT_OF_RANGE"}};
      }
      return authoring::ForeignSchemaValue::signed_integer(static_cast<std::int64_t>(converted));
    }
    const unsigned long long converted =  // NOLINT(runtime/int)
        PyLong_AsUnsignedLongLong(value.ptr());
    if (converted == static_cast<unsigned long long>(-1) && PyErr_Occurred() != nullptr) {
      PyErr_Clear();
      throw RuntimeError{
          Error{ErrorCode::kInvalidArgument, "PYTHON_SCHEMA_FIELD_VALUE_OUT_OF_RANGE"}};
    }
    return authoring::ForeignSchemaValue::unsigned_integer(static_cast<std::uint64_t>(converted));
  }
  if (nb::isinstance<nb::float_>(value)) {
    return authoring::ForeignSchemaValue::floating_point(nb::cast<double>(value));
  }
  if (nb::isinstance<nb::str>(value)) {
    return authoring::ForeignSchemaValue::string(nb::cast<std::string>(value));
  }
  if (PyBytes_Check(value.ptr()) || PyByteArray_Check(value.ptr())) {
    const bool bytes = PyBytes_Check(value.ptr());
    const char* data = bytes ? PyBytes_AsString(value.ptr()) : PyByteArray_AsString(value.ptr());
    const auto size = bytes ? PyBytes_Size(value.ptr()) : PyByteArray_Size(value.ptr());
    const auto* begin = reinterpret_cast<const std::uint8_t*>(data);
    return authoring::ForeignSchemaValue::numeric_sequence(
        std::vector<std::uint8_t>(begin, begin + size));
  }
  const nb::object object = nb::borrow<nb::object>(value);
  if (is_tensor_like(object)) {
    return authoring::ForeignSchemaValue::tensor(*as_tensor(object));
  }
  if (nb::isinstance<nb::dict>(value)) {
    authoring::ForeignSchemaValue::Object fields;
    for (const auto& [key, item] : nb::cast<nb::dict>(value)) {
      if (!nb::isinstance<nb::str>(key)) {
        throw nb::type_error("generated schema field names must be strings");
      }
      fields.emplace(nb::cast<std::string>(key), python_schema_value(item));
    }
    return authoring::ForeignSchemaValue::object(std::move(fields));
  }
  if (nb::isinstance<nb::list>(value) || nb::isinstance<nb::tuple>(value)) {
    authoring::ForeignSchemaValue::Sequence elements;
    elements.reserve(static_cast<std::size_t>(
        PyList_Check(value.ptr()) ? PyList_Size(value.ptr()) : PyTuple_Size(value.ptr())));
    for (const nb::handle item : nb::cast<nb::iterable>(value)) {
      elements.push_back(python_schema_value(item));
    }
    return authoring::ForeignSchemaValue::sequence(std::move(elements));
  }
  if (nb::hasattr(value, "_holoscan_schema_values")) {
    return python_schema_value(value.attr("_holoscan_schema_values")());
  }
  if (nb::isinstance<PyPayloadType>(value)) {
    constexpr std::string_view schema_prefix{"holoscan.schema."};
    const std::string name = nb::cast<const PyPayloadType&>(value).name();
    if (name.starts_with(schema_prefix)) {
      const std::string_view value_name{name.data() + schema_prefix.size(),
                                        name.size() - schema_prefix.size()};
      throw nb::type_error(
          fmt::format("holoscan.payload.schema.{} is a payload identity token; emit an instance "
                      "of holoscan.schema.{} instead",
                      value_name,
                      value_name)
              .c_str());
    }
    throw nb::type_error(
        fmt::format("{} is a payload identity token and cannot be emitted as a value", name)
            .c_str());
  }
  throw nb::type_error("value cannot be converted to a generated Holoscan schema field");
}

nb::object python_schema_value(const authoring::ForeignSchemaValue& value) {
  if (value.is_null()) {
    return nb::none();
  }
  if (const auto* item = value.get<bool>()) {
    return nb::bool_(*item);
  }
  if (const auto* item = value.get<std::int64_t>()) {
    return nb::int_(*item);
  }
  if (const auto* item = value.get<std::uint64_t>()) {
    return nb::int_(*item);
  }
  if (const auto* item = value.get<double>()) {
    return nb::float_(*item);
  }
  if (const auto* item = value.get<std::string>()) {
    return nb::str(item->data(), item->size());
  }
  if (const auto* item = value.get<Tensor>()) {
    track_python_tensor(*item);
    return nb::cast(std::make_shared<Tensor>(*item));
  }
  if (const auto* items = value.get<authoring::ForeignSchemaValue::NumericSequence>()) {
    return std::visit(
        [](const auto& elements) -> nb::object {
          using Element = typename std::remove_cvref_t<decltype(elements)>::value_type;
          if constexpr (std::is_same_v<Element, std::uint8_t>) {
            if (elements.size() >
                static_cast<std::size_t>(std::numeric_limits<Py_ssize_t>::max())) {
              PyErr_SetString(PyExc_OverflowError,
                              "generated schema byte vector exceeds Python size limit");
              throw nb::python_error();
            }
            return nb::bytes(elements.empty() ? "" : reinterpret_cast<const char*>(elements.data()),
                             elements.size());
          } else {
            return python_schema_list(elements.size(), [&](std::size_t index) -> nb::object {
              if constexpr (std::is_floating_point_v<Element>) {
                return nb::float_(elements[index]);
              } else {
                return nb::int_(elements[index]);
              }
            });
          }
        },
        *items);
  }
  if (const auto* items = value.get<authoring::ForeignSchemaValue::Sequence>()) {
    return python_schema_list(
        items->size(), [&](std::size_t index) { return python_schema_value((*items)[index]); });
  }
  if (const auto* fields = value.get<authoring::ForeignSchemaValue::Object>()) {
    nb::dict result;
    for (const auto& [name, item] : *fields) {
      result[nb::str(name.c_str())] = python_schema_value(item);
    }
    return result;
  }
  throw RuntimeError{Error{ErrorCode::kProtocolError, "PYTHON_SCHEMA_VALUE_KIND_INVALID"}};
}

nb::object materialize_python_schema(const authoring::ForeignSchemaValue& value,
                                     const GilSafeObject& schema_type) {
  const nb::object fields = python_schema_value(value);
  if (!schema_type || schema_type.borrow().is_none()) {
    return fields;
  }
  return schema_type.borrow().attr("_from_holoscan_schema")(fields);
}

}  // namespace holoscan
