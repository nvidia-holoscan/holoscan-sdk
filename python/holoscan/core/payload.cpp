/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "payload.hpp"

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/schema/authoring_registry.hpp>

#include "graph_pydoc.hpp"
#include "module_name.hpp"

namespace holoscan {
namespace {

[[noreturn]] void throw_payload_error(std::string_view token) {
  throw RuntimeError{Error{ErrorCode::kInvalidArgument, std::string{token}}};
}

template <typename Integer>
[[nodiscard]] ParamScalarValue python_integer_scalar(nb::handle value) {
  static_assert(std::is_integral_v<Integer> && !std::same_as<Integer, bool>);
  if (!nb::isinstance<nb::int_>(value) || nb::isinstance<nb::bool_>(value)) {
    throw_payload_error("AUTHORING_PAYLOAD_VALUE_TYPE_MISMATCH");
  }
  if constexpr (std::is_signed_v<Integer>) {
    const long long decoded = PyLong_AsLongLong(value.ptr());
    if (PyErr_Occurred() != nullptr || decoded < std::numeric_limits<Integer>::min() ||
        decoded > std::numeric_limits<Integer>::max()) {
      PyErr_Clear();
      throw_payload_error("AUTHORING_PAYLOAD_VALUE_OUT_OF_RANGE");
    }
    return ParamScalarValue::from(static_cast<Integer>(decoded));
  } else {
    const unsigned long long decoded = PyLong_AsUnsignedLongLong(value.ptr());
    if (PyErr_Occurred() != nullptr || decoded > std::numeric_limits<Integer>::max()) {
      PyErr_Clear();
      throw_payload_error("AUTHORING_PAYLOAD_VALUE_OUT_OF_RANGE");
    }
    return ParamScalarValue::from(static_cast<Integer>(decoded));
  }
}

[[noreturn]] void throw_pod_field_error(std::string_view token, std::string_view field) {
  throw RuntimeError{
      Error{ErrorCode::kInvalidArgument, fmt::format("{}: field '{}'", token, field)}};
}

[[nodiscard]] std::size_t pod_element_size(PodScalarKind kind) noexcept {
  switch (kind) {
    case PodScalarKind::kByte:
    case PodScalarKind::kUInt8:
    case PodScalarKind::kInt8:
      return 1U;
    case PodScalarKind::kUInt16:
    case PodScalarKind::kInt16:
      return 2U;
    case PodScalarKind::kUInt32:
    case PodScalarKind::kInt32:
    case PodScalarKind::kFloat32:
      return 4U;
    case PodScalarKind::kUInt64:
    case PodScalarKind::kInt64:
    case PodScalarKind::kFloat64:
      return 8U;
  }
  return 0U;
}

template <typename Element>
[[nodiscard]] nb::object pod_element_to_python(const std::byte* address) {
  Element value{};
  std::memcpy(&value, address, sizeof(Element));
  if constexpr (std::is_floating_point_v<Element>) {
    return nb::float_(static_cast<double>(value));
  } else {
    return nb::int_(value);
  }
}

// Dispatch once per field, then convert the entire numeric array at its exact element type.
template <typename Element>
[[nodiscard]] nb::object pod_field_to_python(const std::byte* address, std::uint32_t count) {
  if (count == 0U) {
    return pod_element_to_python<Element>(address);
  }
  nb::object result = nb::steal(PyTuple_New(count));
  if (!result.is_valid()) {
    throw nb::python_error();
  }
  for (std::uint32_t index = 0U; index < count; ++index) {
    nb::object item = pod_element_to_python<Element>(address + index * sizeof(Element));
    if (PyTuple_SetItem(result.ptr(), index, item.release().ptr()) != 0) {
      throw nb::python_error();
    }
  }
  return result;
}

[[nodiscard]] nb::object pod_field_to_python(PodScalarKind kind, const std::byte* address,
                                             std::uint32_t count = 0U) {
  switch (kind) {
    case PodScalarKind::kByte:
    case PodScalarKind::kUInt8:
      return pod_field_to_python<std::uint8_t>(address, count);
    case PodScalarKind::kInt8:
      return pod_field_to_python<std::int8_t>(address, count);
    case PodScalarKind::kUInt16:
      return pod_field_to_python<std::uint16_t>(address, count);
    case PodScalarKind::kInt16:
      return pod_field_to_python<std::int16_t>(address, count);
    case PodScalarKind::kUInt32:
      return pod_field_to_python<std::uint32_t>(address, count);
    case PodScalarKind::kInt32:
      return pod_field_to_python<std::int32_t>(address, count);
    case PodScalarKind::kUInt64:
      return pod_field_to_python<std::uint64_t>(address, count);
    case PodScalarKind::kInt64:
      return pod_field_to_python<std::int64_t>(address, count);
    case PodScalarKind::kFloat32:
      return pod_field_to_python<float>(address, count);
    case PodScalarKind::kFloat64:
      return pod_field_to_python<double>(address, count);
  }
  throw_payload_error("AUTHORING_PAYLOAD_TYPE_UNSUPPORTED");
}

// Store an exact float, or an exact int that fits, without the generic scalar conversion and
// without running Python code. Anything else returns false, and the generic path then converts it
// or rejects it with its usual error.
template <typename Element>
[[nodiscard]] bool pod_element_fast_from_python(PyObject* value, std::byte* address) noexcept {
  Element stored{};
  if constexpr (std::is_floating_point_v<Element>) {
    if (PyFloat_CheckExact(value) == 0) {
      return false;
    }
    const double decoded = PyFloat_AsDouble(value);
    if constexpr (std::same_as<Element, float>) {
      if (std::isfinite(decoded) &&
          std::abs(decoded) > static_cast<double>(std::numeric_limits<float>::max())) {
        return false;
      }
    }
    stored = static_cast<Element>(decoded);
  } else if constexpr (std::is_signed_v<Element>) {
    if (PyLong_CheckExact(value) == 0) {
      return false;
    }
    int overflow = 0;
    const long long decoded = PyLong_AsLongLongAndOverflow(value, &overflow);
    if (overflow != 0 || decoded < std::numeric_limits<Element>::min() ||
        decoded > std::numeric_limits<Element>::max()) {
      return false;
    }
    stored = static_cast<Element>(decoded);
  } else {
    if (PyLong_CheckExact(value) == 0) {
      return false;
    }
    const unsigned long long decoded = PyLong_AsUnsignedLongLong(value);
    if (decoded == static_cast<unsigned long long>(-1) && PyErr_Occurred() != nullptr) {
      PyErr_Clear();
      return false;
    }
    if (decoded > std::numeric_limits<Element>::max()) {
      return false;
    }
    stored = static_cast<Element>(decoded);
  }
  std::memcpy(address, &stored, sizeof(Element));
  return true;
}

template <typename Element>
void pod_element_from_python(nb::handle value, ScalarIdentityKind kind, std::byte* address,
                             std::string_view field) {
  if (pod_element_fast_from_python<Element>(value.ptr(), address)) {
    return;
  }
  ParamScalarValue scalar;
  try {
    scalar = payload_scalar_from_python(value, kind);
  } catch (const RuntimeError& error) {
    // Keep the scalar conversion's token and add the field; what() leads with the code name.
    std::string_view token = error.what();
    if (const auto separator = token.find(": "); separator != std::string_view::npos) {
      token.remove_prefix(separator + 2U);
    }
    throw_pod_field_error(token, field);
  }
  Element decoded{};
  if (!scalar.try_as(decoded)) {
    throw_pod_field_error("AUTHORING_PAYLOAD_VALUE_OUT_OF_RANGE", field);
  }
  std::memcpy(address, &decoded, sizeof(Element));
}

[[nodiscard]] bool pod_element_fast_from_python(PodScalarKind kind, PyObject* value,
                                                std::byte* address) noexcept {
  switch (kind) {
    case PodScalarKind::kByte:
    case PodScalarKind::kUInt8:
      return pod_element_fast_from_python<std::uint8_t>(value, address);
    case PodScalarKind::kInt8:
      return pod_element_fast_from_python<std::int8_t>(value, address);
    case PodScalarKind::kUInt16:
      return pod_element_fast_from_python<std::uint16_t>(value, address);
    case PodScalarKind::kInt16:
      return pod_element_fast_from_python<std::int16_t>(value, address);
    case PodScalarKind::kUInt32:
      return pod_element_fast_from_python<std::uint32_t>(value, address);
    case PodScalarKind::kInt32:
      return pod_element_fast_from_python<std::int32_t>(value, address);
    case PodScalarKind::kUInt64:
      return pod_element_fast_from_python<std::uint64_t>(value, address);
    case PodScalarKind::kInt64:
      return pod_element_fast_from_python<std::int64_t>(value, address);
    case PodScalarKind::kFloat32:
      return pod_element_fast_from_python<float>(value, address);
    case PodScalarKind::kFloat64:
      return pod_element_fast_from_python<double>(value, address);
  }
  return false;
}

void pod_element_from_python(PodScalarKind kind, nb::handle value, std::byte* address,
                             std::string_view field) {
  switch (kind) {
    case PodScalarKind::kByte:
    case PodScalarKind::kUInt8:
      return pod_element_from_python<std::uint8_t>(
          value, ScalarIdentityKind::kUInt8, address, field);
    case PodScalarKind::kInt8:
      return pod_element_from_python<std::int8_t>(value, ScalarIdentityKind::kInt8, address, field);
    case PodScalarKind::kUInt16:
      return pod_element_from_python<std::uint16_t>(
          value, ScalarIdentityKind::kUInt16, address, field);
    case PodScalarKind::kInt16:
      return pod_element_from_python<std::int16_t>(
          value, ScalarIdentityKind::kInt16, address, field);
    case PodScalarKind::kUInt32:
      return pod_element_from_python<std::uint32_t>(
          value, ScalarIdentityKind::kUInt32, address, field);
    case PodScalarKind::kInt32:
      return pod_element_from_python<std::int32_t>(
          value, ScalarIdentityKind::kInt32, address, field);
    case PodScalarKind::kUInt64:
      return pod_element_from_python<std::uint64_t>(
          value, ScalarIdentityKind::kUInt64, address, field);
    case PodScalarKind::kInt64:
      return pod_element_from_python<std::int64_t>(
          value, ScalarIdentityKind::kInt64, address, field);
    case PodScalarKind::kFloat32:
      return pod_element_from_python<float>(value, ScalarIdentityKind::kFloat32, address, field);
    case PodScalarKind::kFloat64:
      return pod_element_from_python<double>(value, ScalarIdentityKind::kFloat64, address, field);
  }
  throw_payload_error("AUTHORING_PAYLOAD_TYPE_UNSUPPORTED");
}

// Hold one acquired Python buffer and release it on every exit, including an allocation failure.
class AcquiredPythonBuffer {
 public:
  explicit AcquiredPythonBuffer(nb::handle value) {
    if (PyObject_GetBuffer(value.ptr(), &view_, PyBUF_C_CONTIGUOUS) != 0) {
      throw nb::python_error();
    }
  }
  ~AcquiredPythonBuffer() { PyBuffer_Release(&view_); }
  AcquiredPythonBuffer(const AcquiredPythonBuffer&) = delete;
  AcquiredPythonBuffer& operator=(const AcquiredPythonBuffer&) = delete;
  AcquiredPythonBuffer(AcquiredPythonBuffer&&) = delete;
  AcquiredPythonBuffer& operator=(AcquiredPythonBuffer&&) = delete;

  [[nodiscard]] const Py_buffer& view() const noexcept { return view_; }

 private:
  Py_buffer view_{};
};

// Copy the exact native object bytes out of a buffer-protocol object.
[[nodiscard]] std::vector<std::byte> pod_bytes_from_buffer(nb::handle value,
                                                           const PodIdentityRecord& pod) {
  const AcquiredPythonBuffer buffer{value};
  const Py_ssize_t length = buffer.view().len;
  if (length < 0 || static_cast<std::uint64_t>(length) != pod.object_size) {
    throw RuntimeError{Error{
        ErrorCode::kInvalidArgument,
        fmt::format("AUTHORING_POD_PAYLOAD_SIZE_MISMATCH: '{}' needs {} bytes, the buffer has {}",
                    pod.canonical_name,
                    pod.object_size,
                    length)}};
  }
  std::vector<std::byte> native(pod.object_size);
  std::memcpy(native.data(), buffer.view().buf, native.size());
  return native;
}

// The field names of one raw-POD layout as interned Python strings, created on first use. A
// registry row is never freed, even when replaced, so its layout's address identifies it for the
// life of the process; the GIL serializes access, and the strings are kept for the same lifetime.
[[nodiscard]] std::span<PyObject* const> pod_field_keys(const PodIdentityRecord& pod) {
  static auto* const cache =
      new std::unordered_map<const PodIdentityRecord*, std::vector<PyObject*>>();
  if (const auto found = cache->find(&pod); found != cache->end()) {
    return found->second;
  }
  std::vector<PyObject*> keys;
  keys.reserve(pod.fields.size());
  for (const auto& field : pod.fields) {
    PyObject* key =
        PyUnicode_FromStringAndSize(field.name.data(), static_cast<Py_ssize_t>(field.name.size()));
    if (key == nullptr) {
      for (PyObject* const created : keys) {
        Py_DECREF(created);
      }
      throw nb::python_error();
    }
    PyUnicode_InternInPlace(&key);
    keys.push_back(key);
  }
  return cache->emplace(&pod, std::move(keys)).first->second;
}

// The element count cannot tell a one-element array from a scalar, so conversion needs the row's
// per-field array flags.
void require_pod_array_fields(const PodIdentityRecord& pod, std::span<const bool> array_fields) {
  if (array_fields.size() != pod.fields.size()) {
    throw_payload_error("AUTHORING_PAYLOAD_TYPE_UNSUPPORTED");
  }
}

// A mapping that has every raw-POD field and no other key has exactly one key per field.
void require_no_unknown_pod_keys(const PodIdentityRecord& pod, Py_ssize_t key_count) {
  if (key_count != static_cast<Py_ssize_t>(pod.fields.size())) {
    throw RuntimeError{Error{
        ErrorCode::kInvalidArgument,
        fmt::format("AUTHORING_POD_FIELD_UNKNOWN: '{}' has {} fields, the mapping has {} keys",
                    pod.canonical_name,
                    pod.fields.size(),
                    key_count)}};
  }
}

// Collect one value per raw-POD field from a mapping, in field order. A plain dict lookup runs no
// mapping code. For any other mapping, its actual keys are checked for missing and unknown fields
// before its first __getitem__, so a lookup hook such as defaultdict.__missing__ cannot supply or
// insert a missing field, a rejected mapping has no getter called, and each value is read once.
[[nodiscard]] std::vector<nb::object> pod_field_values(nb::handle value,
                                                       const PodIdentityRecord& pod) {
  const std::span<PyObject* const> keys = pod_field_keys(pod);
  std::vector<nb::object> items;
  items.reserve(pod.fields.size());
  if (PyDict_CheckExact(value.ptr())) {
    for (std::size_t index = 0U; index < pod.fields.size(); ++index) {
      const auto& field = pod.fields[index];
      PyObject* const item = PyDict_GetItemWithError(value.ptr(), keys[index]);
      if (item == nullptr) {
        if (PyErr_Occurred() != nullptr) {
          throw nb::python_error();
        }
        throw_pod_field_error("AUTHORING_POD_FIELD_MISSING", field.name);
      }
      items.push_back(nb::borrow(item));
    }
    require_no_unknown_pod_keys(pod, PyDict_Size(value.ptr()));
    return items;
  }
  const nb::object present_keys = nb::steal(PySet_New(value.attr("keys")().ptr()));
  if (!present_keys.is_valid()) {
    throw nb::python_error();
  }
  for (std::size_t index = 0U; index < pod.fields.size(); ++index) {
    const int present = PySet_Contains(present_keys.ptr(), keys[index]);
    if (present < 0) {
      throw nb::python_error();
    }
    if (present == 0) {
      throw_pod_field_error("AUTHORING_POD_FIELD_MISSING", pod.fields[index].name);
    }
  }
  require_no_unknown_pod_keys(pod, PySet_Size(present_keys.ptr()));
  for (PyObject* const key : keys) {
    nb::object item = nb::steal(PyObject_GetItem(value.ptr(), key));
    if (!item.is_valid()) {
      throw nb::python_error();
    }
    items.push_back(std::move(item));
  }
  return items;
}

void install_schema_payloads() {
  (void)schema::register_sdk_schema_payloads();
}

void install_payload_tokens(nb::module_& module) {
  constexpr std::array names{std::string_view{"boolean"},
                             std::string_view{"uint8"},
                             std::string_view{"int8"},
                             std::string_view{"uint16"},
                             std::string_view{"int16"},
                             std::string_view{"uint32"},
                             std::string_view{"int32"},
                             std::string_view{"uint64"},
                             std::string_view{"int64"},
                             std::string_view{"float32"},
                             std::string_view{"float64"}};
  for (const std::string_view name : names) {
    module.attr(std::string{name}.c_str()) = payload_type(std::string{name});
  }
  module.attr("pyobject") = payload_type("holoscan.PyObjectPayload");
}

}  // namespace

std::string PyPayloadType::name() const {
  return std::string{row_->name};
}

std::uint32_t PyPayloadType::epoch() const noexcept {
  if (row_->identity.schema) {
    return row_->identity.schema->compatibility_epoch;
  }
  if (row_->identity.pod) {
    return row_->identity.pod->version;
  }
  return 1U;
}

std::uint32_t PyPayloadType::transport_profile_tag() const noexcept {
  return schema_transport_profile_tag(row_->schema_transport_profile);
}

std::string PyPayloadType::repr() const {
  return fmt::format("<PayloadType {} (epoch {})>", row_->name, epoch());
}

nb::object PyPayloadType::equals(nb::handle other) const {
  if (!nb::isinstance<PyPayloadType>(other)) {
    return nb::borrow<nb::object>(Py_NotImplemented);
  }
  return nb::bool_(authoring::payload_identities_equal(
      row_->identity, nb::cast<const PyPayloadType&>(other).row_->identity));
}

Py_hash_t PyPayloadType::hash() const {
  return nb::hash(nb::str(row_->name.data(), row_->name.size()));
}

PyPayloadType payload_type(std::string name) {
  auto result = authoring::authoring_registry().payload_type(name);
  if (!result && name.find('.') == std::string::npos) {
    name.insert(0U, "holoscan.");
    result = authoring::authoring_registry().payload_type(name);
  }
  if (!result) {
    throw RuntimeError{result.error()};
  }
  return PyPayloadType{&result->get()};
}

const authoring::RegisteredPayloadType& python_payload_registration(nb::handle value) {
  if (nb::isinstance<PyPayloadType>(value)) {
    const auto& token = nb::cast<const PyPayloadType&>(value);
    // Tokens retain immutable historical rows; declarations need the active row's callbacks.
    auto current = authoring::authoring_registry().payload_type(token.row().identity);
    if (!current) {
      throw RuntimeError{current.error()};
    }
    return current->get();
  }
  if (value.ptr() == nb::type<Tensor>().ptr()) {
    return payload_type("holoscan.Tensor").row();
  }
  if (nb::hasattr(value, "_registry_name") && nb::hasattr(value, "_from_holoscan_schema")) {
    const authoring::RegisteredPayloadType& row =
        payload_type(nb::cast<std::string>(value.attr("_registry_name"))).row();
    if (row.identity.kind == PayloadIdentityKind::kSchema) {
      return row;
    }
  }
  throw RuntimeError{Error{ErrorCode::kInvalidArgument, "AUTHORING_PAYLOAD_TYPE_TOKEN_REQUIRED"}};
}

const authoring::RegisteredPayloadType& payload_row(nb::handle value) {
  if (nb::isinstance<PyPayloadType>(value)) {
    return nb::cast<const PyPayloadType&>(value).row();
  }
  throw RuntimeError{Error{ErrorCode::kInvalidArgument, "AUTHORING_PAYLOAD_TYPE_TOKEN_REQUIRED"}};
}

ParamScalarValue payload_scalar_from_python(nb::handle value, ScalarIdentityKind kind) {
  switch (kind) {
    case ScalarIdentityKind::kBoolean:
      if (!nb::isinstance<nb::bool_>(value)) {
        throw_payload_error("AUTHORING_PAYLOAD_VALUE_TYPE_MISMATCH");
      }
      return ParamScalarValue::from(value.ptr() == Py_True);
    case ScalarIdentityKind::kUInt8:
      return python_integer_scalar<std::uint8_t>(value);
    case ScalarIdentityKind::kInt8:
      return python_integer_scalar<std::int8_t>(value);
    case ScalarIdentityKind::kUInt16:
      return python_integer_scalar<std::uint16_t>(value);
    case ScalarIdentityKind::kInt16:
      return python_integer_scalar<std::int16_t>(value);
    case ScalarIdentityKind::kUInt32:
      return python_integer_scalar<std::uint32_t>(value);
    case ScalarIdentityKind::kInt32:
      return python_integer_scalar<std::int32_t>(value);
    case ScalarIdentityKind::kUInt64:
      return python_integer_scalar<std::uint64_t>(value);
    case ScalarIdentityKind::kInt64:
      return python_integer_scalar<std::int64_t>(value);
    case ScalarIdentityKind::kFloat32:
    case ScalarIdentityKind::kFloat64: {
      if ((!nb::isinstance<nb::float_>(value) && !nb::isinstance<nb::int_>(value)) ||
          nb::isinstance<nb::bool_>(value)) {
        throw_payload_error("AUTHORING_PAYLOAD_VALUE_TYPE_MISMATCH");
      }
      const double decoded = PyFloat_AsDouble(value.ptr());
      if (PyErr_Occurred() != nullptr) {
        if (PyErr_ExceptionMatches(PyExc_OverflowError) != 0) {
          PyErr_Clear();
          throw_payload_error("AUTHORING_PAYLOAD_VALUE_OUT_OF_RANGE");
        }
        throw nb::python_error();
      }
      if (kind == ScalarIdentityKind::kFloat32) {
        if (std::isfinite(decoded) &&
            std::abs(decoded) > static_cast<double>(std::numeric_limits<float>::max())) {
          throw_payload_error("AUTHORING_PAYLOAD_VALUE_OUT_OF_RANGE");
        }
        return ParamScalarValue::from(static_cast<float>(decoded));
      }
      return ParamScalarValue::from(decoded);
    }
    case ScalarIdentityKind::kLocalEnum:
      throw_payload_error("AUTHORING_PAYLOAD_TYPE_UNSUPPORTED");
  }
  throw_payload_error("AUTHORING_PAYLOAD_TYPE_UNSUPPORTED");
}

nb::object payload_scalar_to_python(ParamScalarValue value) {
  switch (value.kind) {
    case ParamScalarKind::kBool:
      return nb::bool_(value.raw_bits != 0U);
    case ParamScalarKind::kSignedInteger: {
      std::uint64_t bits = value.raw_bits;
      if (value.bit_width < 64U && value.bit_width > 0U &&
          (bits & (std::uint64_t{1} << (value.bit_width - 1U))) != 0U) {
        bits |= ~((std::uint64_t{1} << value.bit_width) - 1U);
      }
      return nb::int_(std::bit_cast<std::int64_t>(bits));
    }
    case ParamScalarKind::kUnsignedInteger:
      return nb::int_(value.raw_bits);
    case ParamScalarKind::kFloatingPoint:
      return nb::float_(value.bit_width == 32U ? static_cast<double>(std::bit_cast<float>(
                                                     static_cast<std::uint32_t>(value.raw_bits)))
                                               : std::bit_cast<double>(value.raw_bits));
    case ParamScalarKind::kDurationNanoseconds:
      return nb::int_(std::bit_cast<std::int64_t>(value.raw_bits));
    case ParamScalarKind::kNone:
      return nb::none();
  }
  return nb::none();
}

nb::dict payload_pod_to_python(const PodIdentityRecord& pod, std::span<const bool> array_fields,
                               std::span<const std::byte> native) {
  if (native.size() != pod.object_size) {
    throw RuntimeError{Error{ErrorCode::kProtocolError, "AUTHORING_POD_PAYLOAD_SIZE_MISMATCH"}};
  }
  require_pod_array_fields(pod, array_fields);
  const std::span<PyObject* const> keys = pod_field_keys(pod);
  nb::dict result;
  for (std::size_t index = 0U; index < pod.fields.size(); ++index) {
    const auto& field = pod.fields[index];
    const std::byte* const address = native.data() + field.byte_offset;
    const nb::handle name{keys[index]};
    if (!array_fields[index]) {
      result[name] = pod_field_to_python(field.scalar_kind, address);
      continue;
    }
    if (field.scalar_kind == PodScalarKind::kByte) {
      result[name] = nb::bytes(reinterpret_cast<const char*>(address), field.element_count);
      continue;
    }
    result[name] = pod_field_to_python(field.scalar_kind, address, field.element_count);
  }
  return result;
}

std::vector<std::byte> payload_pod_from_python(nb::handle value, const PodIdentityRecord& pod,
                                               std::span<const bool> array_fields) {
  require_pod_array_fields(pod, array_fields);
  if (!nb::isinstance<nb::str>(value) && PyObject_CheckBuffer(value.ptr()) != 0) {
    return pod_bytes_from_buffer(value, pod);
  }
  if (nb::isinstance<nb::str>(value) || PyMapping_Check(value.ptr()) == 0 ||
      !nb::hasattr(value, "keys")) {
    throw RuntimeError{Error{
        ErrorCode::kInvalidArgument,
        fmt::format("AUTHORING_PAYLOAD_VALUE_TYPE_MISMATCH: '{}' takes a mapping of its fields "
                    "or a buffer of its native bytes",
                    pod.canonical_name)}};
  }
  const std::vector<nb::object> items = pod_field_values(value, pod);
  std::vector<std::byte> native(pod.object_size);
  for (std::size_t index = 0U; index < pod.fields.size(); ++index) {
    const auto& field = pod.fields[index];
    const nb::object& item = items[index];
    std::byte* const address = native.data() + field.byte_offset;
    if (!array_fields[index]) {
      pod_element_from_python(field.scalar_kind, item, address, field.name);
      continue;
    }
    const std::size_t element_size = pod_element_size(field.scalar_kind);
    if ((field.scalar_kind == PodScalarKind::kUInt8 || field.scalar_kind == PodScalarKind::kByte) &&
        (PyBytes_Check(item.ptr()) || PyByteArray_Check(item.ptr()))) {
      const AcquiredPythonBuffer buffer{item};
      if (buffer.view().len != static_cast<Py_ssize_t>(field.element_count)) {
        throw_pod_field_error("AUTHORING_POD_FIELD_LENGTH_MISMATCH", field.name);
      }
      std::memcpy(address, buffer.view().buf, field.element_count);
      continue;
    }
    if (nb::isinstance<nb::str>(item) || nb::isinstance<nb::bytes>(item) ||
        PySequence_Check(item.ptr()) == 0) {
      throw_pod_field_error("AUTHORING_POD_FIELD_LENGTH_MISMATCH", field.name);
    }
    // A sequence whose length lookup raises keeps its own exception.
    const Py_ssize_t length = PySequence_Size(item.ptr());
    if (length < 0) {
      throw nb::python_error();
    }
    if (length != static_cast<Py_ssize_t>(field.element_count)) {
      throw_pod_field_error("AUTHORING_POD_FIELD_LENGTH_MISMATCH", field.name);
    }
    // An exact list or tuple lends its items. A borrowed item is converted only while no Python
    // code runs; one that needs the generic conversion, which can, is converted from an owned
    // reference.
    const bool exact_list = PyList_CheckExact(item.ptr()) != 0;
    const bool lends_items = exact_list || PyTuple_CheckExact(item.ptr()) != 0;
    for (std::uint32_t element = 0U; element < field.element_count; ++element) {
      const auto position = static_cast<Py_ssize_t>(element);
      nb::object entry;
      if (lends_items) {
        PyObject* const borrowed = exact_list ? PyList_GetItem(item.ptr(), position)
                                              : PyTuple_GetItem(item.ptr(), position);
        if (borrowed == nullptr) {
          throw nb::python_error();
        }
        if (pod_element_fast_from_python(
                field.scalar_kind, borrowed, address + element * element_size)) {
          continue;
        }
        entry = nb::borrow(borrowed);
      } else {
        entry = nb::steal(PySequence_GetItem(item.ptr(), position));
        if (!entry.is_valid()) {
          throw nb::python_error();
        }
      }
      pod_element_from_python(
          field.scalar_kind, entry, address + element * element_size, field.name);
    }
  }
  return native;
}

void init_payload(nb::module_& module) {
  auto payload_type_class = python::bind_class_in_module<PyPayloadType>(
      module, "holoscan.payload", "PayloadType", doc::graph::kPayloadType);
  payload_type_class.def_prop_ro("name", &PyPayloadType::name)
      .def_prop_ro("epoch", &PyPayloadType::epoch)
      .def_prop_ro("transport_profile_tag", &PyPayloadType::transport_profile_tag)
      .def("__repr__", &PyPayloadType::repr)
      .def("__eq__", &PyPayloadType::equals)
      .def("__hash__", &PyPayloadType::hash)
      .def("__call__",
           [](const PyPayloadType& token, const nb::args&, const nb::kwargs&) -> nb::object {
             constexpr std::string_view schema_prefix{"holoscan.schema."};
             const std::string name = token.name();
             if (name.starts_with(schema_prefix)) {
               const std::string_view value_name{name.data() + schema_prefix.size(),
                                                 name.size() - schema_prefix.size()};
               throw nb::type_error(
                   fmt::format("holoscan.payload.schema.{} is a payload identity token; "
                               "construct values with holoscan.schema.{}(...) instead",
                               value_name,
                               value_name)
                       .c_str());
             }
             throw nb::type_error(
                 fmt::format("{} is a payload identity token and cannot construct values", name)
                     .c_str());
           });
  install_schema_payloads();
  python::bind_functions_in_module(
      module, "holoscan.payload", [&] { module.def("resolve", &payload_type, nb::arg("name")); });
  module.attr("Header") = payload_type("holoscan.schema.Header");
  module.attr("String") = payload_type("holoscan.schema.String");
  module.attr("TensorPacket") = payload_type("holoscan.schema.TensorPacket");
  module.attr("Time") = payload_type("holoscan.schema.Time");
  module.attr("TensorType") = payload_type("holoscan.Tensor");
  install_payload_tokens(module);
}

}  // namespace holoscan
