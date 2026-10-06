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
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>

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
        if (PyErr_ExceptionMatches(PyExc_OverflowError)) {
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
