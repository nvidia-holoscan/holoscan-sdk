/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "native_operator.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/foreign_operator.hpp>
#include <holoscan/core/hosted_parameter_snapshot.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/tensor_contract.hpp>

#include "../../../src/runtime/runtime_callback_context.hpp"
#include "clocks.hpp"
#include "imported_tensor.hpp"
#include "lifecycle_authoring.hpp"
#include "module_name.hpp"
#include "native_io.hpp"
#include "payload.hpp"

namespace holoscan {

PyLifecycleHookBindings::PyLifecycleHookBindings(nb::handle instance) : instance_(instance) {}

void PyLifecycleHookBindings::reset() noexcept {
  for (GilSafeObject& callback : callbacks_) {
    callback = GilSafeObject{};
  }
  required_.fill(false);
}

void PyLifecycleHookBindings::bind(LifecycleStage stage, std::string_view method, bool required) {
  if (!is_valid_lifecycle_stage(stage) || method.empty()) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "PYTHON_LIFECYCLE_CALLBACK_INVALID"}};
  }
  if (stage == LifecycleStage::kStart || stage == LifecycleStage::kStop) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "PYTHON_LIFECYCLE_STAGE_NOT_HOSTABLE"}};
  }
  const std::size_t index = static_cast<std::size_t>(stage);
  if (callbacks_[index]) {
    throw RuntimeError{Error{ErrorCode::kDuplicateName, "PYTHON_LIFECYCLE_CALLBACK_DUPLICATE"}};
  }
  const nb::object instance = instance_.borrow();
  const std::string owned_method{method};
  PyObject* const attribute = PyObject_GetAttrString(instance.ptr(), owned_method.c_str());
  if (attribute == nullptr) {
    if (PyErr_ExceptionMatches(PyExc_AttributeError) == 0) {
      throw nb::python_error();
    }
    PyErr_Clear();
    throw RuntimeError{
        Error{ErrorCode::kNotFound, "PYTHON_LIFECYCLE_CALLBACK_UNKNOWN method=" + owned_method}};
  }
  nb::object callback = nb::steal<nb::object>(attribute);
  if (PyCallable_Check(callback.ptr()) == 0) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument,
                             "PYTHON_LIFECYCLE_CALLBACK_NOT_CALLABLE method=" + owned_method}};
  }
  try {
    nb::module_::import_("inspect").attr("signature")(callback).attr("bind")(nb::none());
  } catch (const nb::python_error& error) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument,
                             "PYTHON_LIFECYCLE_CALLBACK_SIGNATURE_INVALID method=" + owned_method +
                                 " detail=" + error.what()}};
  }
  callbacks_[index] = GilSafeObject{std::move(callback)};
  required_[index] = required;
}

bool PyLifecycleHookBindings::has(LifecycleStage stage) const noexcept {
  return is_valid_lifecycle_stage(stage) && callbacks_[static_cast<std::size_t>(stage)];
}

bool PyLifecycleHookBindings::required(LifecycleStage stage) const noexcept {
  return is_valid_lifecycle_stage(stage) && required_[static_cast<std::size_t>(stage)];
}

const GilSafeObject& PyLifecycleHookBindings::callback(LifecycleStage stage) const noexcept {
  return callbacks_[static_cast<std::size_t>(stage)];
}

namespace {

[[noreturn]] void throw_parameter_error(std::string_view token, std::string_view field = {}) {
  std::string message{token};
  if (!field.empty()) {
    message.append(" field_name=");
    message.append(field);
  }
  throw RuntimeError{Error{ErrorCode::kInvalidArgument, std::move(message)}};
}

// Implements the Python Param descriptor and validates values before graph capture.
class PyParamDescriptor {
 public:
  PyParamDescriptor(nb::handle default_value, std::string description, nb::handle value_range,
                    nb::handle allowed, bool recompile_required,
                    std::optional<std::size_t> max_elements)
      : default_(nb::borrow<nb::object>(default_value)),
        description_(std::move(description)),
        range_(nb::borrow<nb::object>(value_range)),
        allowed_(nb::borrow<nb::object>(allowed)),
        recompile_required_(recompile_required),
        max_elements_(max_elements) {}

  void set_name(std::string name) {
    if (!name_.empty() && name_ != name) {
      PyErr_Format(
          PyExc_TypeError,
          "Param descriptor already belongs to attribute '%s' and cannot be reused as '%s'",
          name_.c_str(),
          name.c_str());
      throw nb::python_error();
    }
    name_ = std::move(name);
  }
  [[nodiscard]] const std::string& name() const noexcept { return name_; }
  [[nodiscard]] const std::string& description() const noexcept { return description_; }
  [[nodiscard]] nb::object default_value() const { return default_; }
  [[nodiscard]] nb::object value_range() const { return range_; }
  [[nodiscard]] nb::object allowed() const { return allowed_; }
  [[nodiscard]] bool recompile_required() const noexcept { return recompile_required_; }
  [[nodiscard]] const std::optional<std::size_t>& max_elements() const noexcept {
    return max_elements_;
  }

  // Keep Python's scalar precedence, sequence rules, and authored constraints together so a
  // descriptor has one validation order and one diagnostic boundary.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  void validate(nb::handle value) const {
    const auto type_mismatch = [&] {
      throw_parameter_error("AUTHORING_PARAMETER_TYPE_MISMATCH", name_);
    };
    if (nb::isinstance<nb::bool_>(default_)) {
      if (!nb::isinstance<nb::bool_>(value)) {
        type_mismatch();
      }
    } else if (nb::isinstance<nb::int_>(default_)) {
      if (!nb::isinstance<nb::int_>(value) || nb::isinstance<nb::bool_>(value)) {
        type_mismatch();
      }
    } else if (nb::isinstance<nb::float_>(default_)) {
      if ((!nb::isinstance<nb::float_>(value) && !nb::isinstance<nb::int_>(value)) ||
          nb::isinstance<nb::bool_>(value)) {
        type_mismatch();
      }
    } else if (nb::isinstance<nb::str>(default_)) {
      if (!nb::isinstance<nb::str>(value)) {
        type_mismatch();
      }
    } else if (nb::isinstance<nb::list>(default_) || nb::isinstance<nb::tuple>(default_)) {
      if (!nb::isinstance<nb::list>(value) && !nb::isinstance<nb::tuple>(value)) {
        type_mismatch();
      }
      const nb::sequence sequence = nb::cast<nb::sequence>(value);
      if (max_elements_ && nb::len(sequence) > *max_elements_) {
        throw_parameter_error("PARAM_VALIDATION_FAILED category=cardinality_exceeded", name_);
      }
      if (!range_.is_none() || !allowed_.is_none()) {
        throw_parameter_error("AUTHORING_PARAMETER_CONSTRAINT_INVALID", name_);
      }
      if (declares_bounded_integral_sequence()) {
        for (const nb::handle item : sequence) {
          if (!nb::isinstance<nb::int_>(item) || nb::isinstance<nb::bool_>(item)) {
            type_mismatch();
          }
        }
      }
    } else if (Py_TYPE(value.ptr()) != Py_TYPE(default_.ptr())) {
      type_mismatch();
    }

    if (!range_.is_none()) {
      if (!nb::isinstance<nb::tuple>(range_) || nb::len(range_) != 2U) {
        throw_parameter_error("AUTHORING_PARAMETER_CONSTRAINT_INVALID", name_);
      }
      const nb::tuple bounds = nb::cast<nb::tuple>(range_);
      const int below = PyObject_RichCompareBool(value.ptr(), bounds[0].ptr(), Py_LT);
      const int above = PyObject_RichCompareBool(value.ptr(), bounds[1].ptr(), Py_GT);
      if (below < 0 || above < 0) {
        throw nb::python_error();
      }
      if (below != 0 || above != 0) {
        throw_parameter_error("PARAM_VALIDATION_FAILED category=out_of_range", name_);
      }
    }
    if (!allowed_.is_none()) {
      const int contains = PySequence_Contains(allowed_.ptr(), value.ptr());
      if (contains < 0) {
        throw nb::python_error();
      }
      if (contains == 0) {
        throw_parameter_error("PARAM_VALIDATION_FAILED category=not_allowed", name_);
      }
    }
  }

  [[nodiscard]] nb::object normalize(nb::handle value) const {
    validate(value);
    if (nb::isinstance<nb::float_>(default_) && nb::isinstance<nb::int_>(value)) {
      const double decoded = PyFloat_AsDouble(value.ptr());
      if (PyErr_Occurred() != nullptr) {
        PyErr_Clear();
        throw_parameter_error("PARAM_VALIDATION_FAILED category=out_of_range", name_);
      }
      return nb::float_(decoded);
    }
    return nb::borrow<nb::object>(value);
  }

  [[nodiscard]] bool declares_bounded_integral_sequence() const {
    if (!max_elements_ ||
        (!nb::isinstance<nb::list>(default_) && !nb::isinstance<nb::tuple>(default_))) {
      return false;
    }
    const nb::sequence sequence = nb::cast<nb::sequence>(default_);
    if (nb::len(sequence) == 0U) {
      return false;
    }
    for (const nb::handle item : sequence) {
      if (!nb::isinstance<nb::int_>(item) || nb::isinstance<nb::bool_>(item)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] nb::object get(nb::handle instance) const {
    if (instance.is_none()) {
      return nb::cast(const_cast<PyParamDescriptor*>(this), nb::rv_policy::reference);
    }
    if (!nb::hasattr(instance, "_holoscan_param_values")) {
      PyErr_Format(PyExc_AttributeError,
                   "parameter '%s' is available only on a hosted operator instance",
                   name_.c_str());
      throw nb::python_error();
    }
    const nb::dict values = nb::cast<nb::dict>(instance.attr("_holoscan_param_values"));
    const nb::str key{name_.c_str()};
    if (!values.contains(key)) {
      PyErr_Format(PyExc_AttributeError, "hosted operator has no parameter '%s'", name_.c_str());
      throw nb::python_error();
    }
    return nb::borrow<nb::object>(values[key]);
  }

 private:
  nb::object default_;
  std::string description_;
  nb::object range_;
  nb::object allowed_;
  std::string name_;
  bool recompile_required_{};
  std::optional<std::size_t> max_elements_;
};

// Associates an inherited Python attribute name with its resolved Param descriptor.
struct NativeParamDefinition {
  std::string attribute_name;
  PyParamDescriptor* descriptor{};
};

// Builds the stable diagnostic name of a hosted Python operator type.
[[nodiscard]] std::string native_operator_type_name(nb::handle type) {
  return nb::cast<std::string>(type.attr("__module__")) + "." +
         nb::cast<std::string>(type.attr("__qualname__"));
}

// Derives a portable parameter-schema identity from the hosted Python type name.
[[nodiscard]] std::string native_parameter_schema_name(std::string value) {
  value.insert(0U, "holoscan.python.");
  for (char& character : value) {
    const bool accepted = (character >= 'a' && character <= 'z') ||
                          (character >= 'A' && character <= 'Z') ||
                          (character >= '0' && character <= '9') || character == '.' ||
                          character == '_' || character == '-';
    if (!accepted) {
      character = '_';
    }
  }
  return value;
}

// Converts one Python constraint endpoint into Core's bounded scalar representation.
[[nodiscard]] ParamConstraintScalar python_param_constraint_scalar(nb::handle value,
                                                                   ParamLogicalType type,
                                                                   std::string_view field) {
  switch (type) {
    case ParamLogicalType::kBool:
      if (!nb::isinstance<nb::bool_>(value)) {
        throw_parameter_error("AUTHORING_PARAMETER_CONSTRAINT_INVALID", field);
      }
      return ParamConstraintScalar{.kind = ParamConstraintScalarKind::kUnsigned,
                                   .unsigned_value = value.ptr() == Py_True ? 1U : 0U};
    case ParamLogicalType::kSignedInteger: {
      if (!nb::isinstance<nb::int_>(value) || nb::isinstance<nb::bool_>(value)) {
        throw_parameter_error("AUTHORING_PARAMETER_CONSTRAINT_INVALID", field);
      }
      const long long decoded = PyLong_AsLongLong(value.ptr());
      if (PyErr_Occurred() != nullptr) {
        PyErr_Clear();
        throw_parameter_error("AUTHORING_PARAMETER_CONSTRAINT_INVALID", field);
      }
      return ParamConstraintScalar{.kind = ParamConstraintScalarKind::kSigned,
                                   .signed_value = static_cast<std::int64_t>(decoded)};
    }
    case ParamLogicalType::kFloatingPoint: {
      if ((!nb::isinstance<nb::float_>(value) && !nb::isinstance<nb::int_>(value)) ||
          nb::isinstance<nb::bool_>(value)) {
        throw_parameter_error("AUTHORING_PARAMETER_CONSTRAINT_INVALID", field);
      }
      const double decoded = PyFloat_AsDouble(value.ptr());
      if (PyErr_Occurred() != nullptr || !std::isfinite(decoded)) {
        PyErr_Clear();
        throw_parameter_error("AUTHORING_PARAMETER_CONSTRAINT_INVALID", field);
      }
      return ParamConstraintScalar{.kind = ParamConstraintScalarKind::kFloating,
                                   .floating_value = decoded};
    }
    case ParamLogicalType::kUnsignedInteger:
    case ParamLogicalType::kString:
    case ParamLogicalType::kSequence:
    case ParamLogicalType::kFilesystemPath:
    case ParamLogicalType::kDuration:
    case ParamLogicalType::kObject:
      throw_parameter_error("AUTHORING_PARAMETER_CONSTRAINT_INVALID", field);
  }
  throw_parameter_error("AUTHORING_PARAMETER_CONSTRAINT_INVALID", field);
}

// Projects compiler-visible Python parameters into an owning hosted-schema field.
// This conversion is an explicit Python-type-to-schema dispatch; separate helpers would repeat
// the shared field setup and make supported hosted shapes less visible.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] std::optional<HostedParameterField> hosted_parameter_field(
    const NativeParamDefinition& definition, nb::handle value) {
  HostedParameterField field;
  field.name = definition.attribute_name;
  field.description = definition.descriptor->description();
  field.mutability = definition.descriptor->recompile_required() ? ParamMutability::kRecompile
                                                                 : ParamMutability::kCompileTime;
  const nb::object default_value = definition.descriptor->default_value();
  if (nb::isinstance<nb::bool_>(default_value)) {
    field.type_shape = ParamFieldTypeShape{
        .type = ParamLogicalType::kBool, .bit_width = 1U, .has_stable_shape = true};
    field.values.push_back(ParamScalarValue::from(value.ptr() == Py_True));
  } else if (nb::isinstance<nb::int_>(default_value)) {
    const long long decoded = PyLong_AsLongLong(value.ptr());
    if (PyErr_Occurred() != nullptr) {
      PyErr_Clear();
      throw_parameter_error("PARAM_VALIDATION_FAILED category=out_of_range",
                            definition.attribute_name);
    }
    field.type_shape = ParamFieldTypeShape{
        .type = ParamLogicalType::kSignedInteger, .bit_width = 64U, .has_stable_shape = true};
    field.values.push_back(ParamScalarValue::from(static_cast<std::int64_t>(decoded)));
  } else if (nb::isinstance<nb::float_>(default_value)) {
    const double decoded = PyFloat_AsDouble(value.ptr());
    if (PyErr_Occurred() != nullptr) {
      PyErr_Clear();
      throw_parameter_error("PARAM_VALIDATION_FAILED category=out_of_range",
                            definition.attribute_name);
    }
    field.type_shape = ParamFieldTypeShape{
        .type = ParamLogicalType::kFloatingPoint, .bit_width = 64U, .has_stable_shape = true};
    field.values.push_back(ParamScalarValue::from(decoded));
  } else if (definition.descriptor->declares_bounded_integral_sequence()) {
    const auto& max_elements = definition.descriptor->max_elements();
    if (!max_elements.has_value()) {
      return std::nullopt;
    }
    const nb::sequence sequence = nb::cast<nb::sequence>(value);
    field.type_shape = ParamFieldTypeShape{
        .type = ParamLogicalType::kSequence,
        .element_or_rep_type = ParamLogicalType::kSignedInteger,
        .element_or_rep_bit_width = 64U,
        .has_stable_shape = true,
    };
    field.constraints.max_elements = max_elements.value();
    for (const nb::handle item : sequence) {
      if (!nb::isinstance<nb::int_>(item) || nb::isinstance<nb::bool_>(item)) {
        return std::nullopt;
      }
      const long long decoded = PyLong_AsLongLong(item.ptr());
      if (PyErr_Occurred() != nullptr) {
        PyErr_Clear();
        throw_parameter_error("PARAM_VALIDATION_FAILED category=out_of_range",
                              definition.attribute_name);
      }
      field.values.push_back(ParamScalarValue::from(static_cast<std::int64_t>(decoded)));
    }
  } else {
    return std::nullopt;
  }
  if (!definition.descriptor->value_range().is_none()) {
    const nb::tuple range = nb::cast<nb::tuple>(definition.descriptor->value_range());
    field.constraints.has_range = true;
    field.constraints.range_min =
        python_param_constraint_scalar(range[0], field.type_shape.type, definition.attribute_name);
    field.constraints.range_max =
        python_param_constraint_scalar(range[1], field.type_shape.type, definition.attribute_name);
  }
  if (!definition.descriptor->allowed().is_none()) {
    const nb::sequence allowed = nb::cast<nb::sequence>(definition.descriptor->allowed());
    if (nb::len(allowed) > kParamAllowedValueCapacity) {
      throw_parameter_error("AUTHORING_PARAMETER_CONSTRAINT_INVALID", definition.attribute_name);
    }
    for (const nb::handle candidate : allowed) {
      field.constraints.allowed_values[field.constraints.allowed_value_count++] =
          python_param_constraint_scalar(
              candidate, field.type_shape.type, definition.attribute_name);
    }
  }
  return field;
}

// Resolves a hosted Param descriptor to a compiler-visible Tensor byte-span dependency.
[[nodiscard]] TensorByteSpanExpression hosted_parameter_byte_span(
    const HostedParameterSnapshot& parameters,
    const std::vector<NativeParamDefinition>& definitions, nb::handle parameter, bool shape) {
  if (!nb::isinstance<PyParamDescriptor>(parameter)) {
    throw RuntimeError{
        Error{ErrorCode::kInvalidArgument,
              "PYTHON_TENSOR_PARAMETER_DESCRIPTOR_REQUIRED hint=use_type(self).parameter"}};
  }
  const auto* descriptor = &nb::cast<const PyParamDescriptor&>(parameter);
  const auto name = std::find_if(definitions.begin(), definitions.end(), [&](const auto& value) {
    return value.descriptor == descriptor;
  });
  const ParamSchemaDescriptor* schema = parameters.schema();
  if (name == definitions.end() || schema == nullptr) {
    throw RuntimeError{
        Error{ErrorCode::kInvalidArgument, "GRAPH_TENSOR_BYTE_SPAN_DEPENDENCY_INVALID"}};
  }
  const auto field =
      std::find_if(schema->fields.begin(), schema->fields.end(), [&](const auto& value) {
        return value.canonical_name == name->attribute_name;
      });
  if (field == schema->fields.end()) {
    throw RuntimeError{
        Error{ErrorCode::kInvalidArgument, "GRAPH_TENSOR_BYTE_SPAN_DEPENDENCY_INVALID"}};
  }
  return detail_make_hosted_parameter_byte_span(
      parameters.type().local_type_token, schema, &*field, shape);
}

[[nodiscard]] TensorMemoryKindExpression hosted_parameter_memory_kind(
    const HostedParameterSnapshot& parameters,
    const std::vector<NativeParamDefinition>& definitions, nb::handle parameter,
    MemoryKind when_false, MemoryKind when_true) {
  if (!nb::isinstance<PyParamDescriptor>(parameter)) {
    throw RuntimeError{
        Error{ErrorCode::kInvalidArgument,
              "PYTHON_TENSOR_PARAMETER_DESCRIPTOR_REQUIRED hint=use_type(self).parameter"}};
  }
  const auto* descriptor = &nb::cast<const PyParamDescriptor&>(parameter);
  const auto name = std::find_if(definitions.begin(), definitions.end(), [&](const auto& value) {
    return value.descriptor == descriptor;
  });
  const ParamSchemaDescriptor* schema = parameters.schema();
  if (name == definitions.end() || schema == nullptr) {
    throw RuntimeError{
        Error{ErrorCode::kInvalidArgument, "GRAPH_TENSOR_MEMORY_KIND_DEPENDENCY_INVALID"}};
  }
  const auto field =
      std::find_if(schema->fields.begin(), schema->fields.end(), [&](const auto& value) {
        return value.canonical_name == name->attribute_name;
      });
  if (field == schema->fields.end() || field->logical_type != ParamLogicalType::kBool) {
    throw RuntimeError{
        Error{ErrorCode::kInvalidArgument, "GRAPH_TENSOR_MEMORY_KIND_DEPENDENCY_INVALID"}};
  }
  const std::array mapping{
      detail::TensorMemoryKindAuthoringCase{ParamScalarValue::from(false), when_false},
      detail::TensorMemoryKindAuthoringCase{ParamScalarValue::from(true), when_true},
  };
  return detail_make_hosted_parameter_memory_kind(
      parameters.type().local_type_token, schema, &*field, mapping);
}

// Owns the immutable Python class state, parameters, and port declarations captured by Graph.op().
class PyStateCapture {
 public:
  using HoloscanOperatorParameterCapture = void;

  PyStateCapture(GilSafeObject type, GilSafeObject values, std::vector<NativePortDefinition> ports,
                 std::vector<NativeParamDefinition> parameter_definitions,
                 HostedParameterSnapshot parameters, bool explicit_baseline, bool gil_serialization,
                 std::string type_name)
      : type_(std::move(type)),
        values_(std::move(values)),
        ports_(std::move(ports)),
        parameter_definitions_(std::move(parameter_definitions)),
        parameters_(std::move(parameters)),
        explicit_baseline_(explicit_baseline),
        gil_serialization_(gil_serialization),
        type_name_(std::move(type_name)) {}

  [[nodiscard]] const ParamTypeDescriptor& parameter_type() const noexcept {
    return parameters_ ? parameters_.type() : param_type_descriptor<NoParams>();
  }
  [[nodiscard]] const ParamSchemaDescriptor* parameter_schema() const noexcept {
    return parameters_.schema();
  }
  [[nodiscard]] const void* initial_parameter_value() const noexcept {
    return parameters_ ? static_cast<const void*>(&parameters_)
                       : static_cast<const void*>(&detail::kNoParams);
  }
  [[nodiscard]] bool has_explicit_parameter_baseline() const noexcept { return explicit_baseline_; }
  [[nodiscard]] bool has_no_params() const noexcept { return !parameters_; }
  [[nodiscard]] const GilSafeObject& type_object() const noexcept { return type_; }
  [[nodiscard]] const GilSafeObject& values_object() const noexcept { return values_; }
  [[nodiscard]] const std::vector<NativePortDefinition>& ports() const noexcept { return ports_; }
  [[nodiscard]] const std::vector<NativeParamDefinition>& parameter_definitions() const noexcept {
    return parameter_definitions_;
  }
  [[nodiscard]] const HostedParameterSnapshot& parameters() const noexcept { return parameters_; }
  [[nodiscard]] bool gil_serialization() const noexcept { return gil_serialization_; }
  [[nodiscard]] const std::string& type_name() const noexcept { return type_name_; }

 private:
  GilSafeObject type_;
  GilSafeObject values_;
  std::vector<NativePortDefinition> ports_;
  std::vector<NativeParamDefinition> parameter_definitions_;
  HostedParameterSnapshot parameters_;
  bool explicit_baseline_{};
  bool gil_serialization_{};
  std::string type_name_;
};

// Captures inherited descriptors and normalized constructor values before Core stores the operator.
[[nodiscard]] PyStateCapture make_native_capture(nb::handle type, const nb::kwargs& kwargs) {
  std::vector<NativePortDefinition> ports;
  std::vector<NativeParamDefinition> params;
  const nb::tuple mro = nb::cast<nb::tuple>(type.attr("__mro__"));
  for (std::size_t offset = 0U; offset < mro.size(); ++offset) {
    const nb::handle base = mro[mro.size() - offset - 1U];
    for (const nb::handle item_handle :
         nb::cast<nb::iterable>(base.attr("__dict__").attr("items")())) {
      const nb::tuple item = nb::cast<nb::tuple>(item_handle);
      const std::string attribute_name = nb::cast<std::string>(item[0]);
      std::erase_if(ports,
                    [&](const auto& value) { return value.attribute_name == attribute_name; });
      std::erase_if(params,
                    [&](const auto& value) { return value.attribute_name == attribute_name; });
      if (auto port = native_port_definition(item[1], attribute_name)) {
        ports.push_back(std::move(*port));
      } else if (nb::isinstance<PyParamDescriptor>(item[1])) {
        auto& descriptor = nb::cast<PyParamDescriptor&>(item[1]);
        params.push_back(NativeParamDefinition{attribute_name, &descriptor});
      }
    }
  }
  for (std::size_t index = 0U; index < ports.size(); ++index) {
    if (std::ranges::any_of(std::span{ports}.first(index), [&](const auto& prior) {
          return prior.port_name == ports[index].port_name;
        })) {
      throw RuntimeError{Error{ErrorCode::kDuplicateName, "GRAPH_OPERATOR_PORT_NAME_DUPLICATE"}};
    }
  }

  const nb::dict supplied = nb::cast<nb::dict>(kwargs);
  const nb::dict values;
  const nb::object deepcopy = nb::module_::import_("copy").attr("deepcopy");
  const std::string type_name = native_operator_type_name(type);
  HostedParameterSnapshotBuilder builder{native_parameter_schema_name(type_name)};
  std::size_t hosted_count = 0U;
  for (const NativeParamDefinition& definition : params) {
    const nb::str key{definition.attribute_name.c_str()};
    const nb::handle value =
        supplied.contains(key) ? supplied[key] : definition.descriptor->default_value();
    const nb::object normalized = definition.descriptor->normalize(value);
    values[key] = deepcopy(normalized);
    if (auto hosted = hosted_parameter_field(definition, normalized)) {
      builder.field(std::move(*hosted));
      ++hosted_count;
    }
  }
  for (const nb::handle key : supplied.keys()) {
    if (!values.contains(key)) {
      throw_parameter_error("AUTHORING_PARAMETER_UNKNOWN", nb::cast<std::string>(key));
    }
  }
  HostedParameterSnapshot snapshot =
      hosted_count == 0U ? HostedParameterSnapshot{} : builder.build();
  // A GIL-enabled interpreter is one process-wide callback serialization resource. Mark every
  // native Python Operator with the same compiler-visible domain so automatic partitioning can
  // co-partition compatible callbacks instead of occupying independent lanes blocked on the GIL.
  // CPython 3.13+ documents `_is_gil_enabled` as an optional implementation detail, so feature-test
  // it: free-threaded CPython reports false and restores ordinary partitioning, while older or
  // unknown interpreters conservatively retain the serialization domain.
  bool gil_enabled = true;
  const nb::object sys = nb::module_::import_("sys");
  if (nb::hasattr(sys, "_is_gil_enabled")) {
    gil_enabled = nb::cast<bool>(sys.attr("_is_gil_enabled")());
  }
  return PyStateCapture{GilSafeObject{type},
                        GilSafeObject{values},
                        std::move(ports),
                        std::move(params),
                        std::move(snapshot),
                        supplied.size() != 0U,
                        gil_enabled,
                        type_name};
}

// Materializes the scalar and sequence subset of a hosted parameter snapshot as a Python dict.
[[nodiscard]] nb::dict python_hosted_parameters(const HostedParameterSnapshot& parameters) {
  nb::dict result;
  const ParamSchemaDescriptor* schema = parameters.schema();
  if (schema == nullptr) {
    return result;
  }
  for (const ParamFieldDescriptor& field : schema->fields) {
    const nb::str name{field.canonical_name.data(), field.canonical_name.size()};
    if (field.logical_type == ParamLogicalType::kSequence) {
      nb::list values;
      for (const ParamScalarValue value : parameters.read_scalar_sequence(field)) {
        values.append(payload_scalar_to_python(value));
      }
      result[name] = std::move(values);
      continue;
    }
    ParamScalarValue value;
    if (!parameters.read_scalar(field, value)) {
      throw RuntimeError{
          Error{ErrorCode::kNotSupported, "PYTHON_LIFECYCLE_PARAMETER_KIND_UNSUPPORTED"}};
    }
    result[name] = payload_scalar_to_python(value);
  }
  return result;
}

// Callback-scoped view of parameters, cancellation, transition phase, and restart targets.
class PyLifecycleContext {
 public:
  PyLifecycleContext(LifecycleContext& context, bool has_parameters)
      : context_(context), has_parameters_(has_parameters) {}

  [[nodiscard]] nb::dict params() const {
    return context_.with_value("PYTHON_LIFECYCLE_CONTEXT_EXPIRED", [&](LifecycleContext& context) {
      if (!has_parameters_) {
        return nb::dict{};
      }
      return python_hosted_parameters(context.params<HostedParameterSnapshot>());
    });
  }

  [[nodiscard]] std::uint64_t params_generation() const {
    return context_.with_value("PYTHON_LIFECYCLE_CONTEXT_EXPIRED", [](LifecycleContext& context) {
      return context.params_generation();
    });
  }

  [[nodiscard]] LifecycleTransitionPhase transition_phase() const {
    return context_.with_value("PYTHON_LIFECYCLE_CONTEXT_EXPIRED", [](LifecycleContext& context) {
      return context.transition_phase();
    });
  }

  [[nodiscard]] bool stop_requested() const {
    return context_.with_value("PYTHON_LIFECYCLE_CONTEXT_EXPIRED", [](LifecycleContext& context) {
      return context.cancellation().stop_requested();
    });
  }

  void report_error(ErrorCode code, const std::string& message) const {
    context_.with_value("PYTHON_LIFECYCLE_CONTEXT_EXPIRED",
                        [&](LifecycleContext& context) { context.report_error(code, message); });
  }

  [[nodiscard]] PyLifecycleOperatorRef restart_target(const std::string& path) const {
    return context_.with_value("PYTHON_LIFECYCLE_CONTEXT_EXPIRED", [&](LifecycleContext& context) {
      auto parsed = OperatorPath::parse(path);
      if (!parsed) {
        throw RuntimeError{parsed.error()};
      }
      auto target = context.restart_target(*parsed);
      if (!target) {
        throw RuntimeError{target.error()};
      }
      return PyLifecycleOperatorRef{*target};
    });
  }

  [[nodiscard]] ClockRef resolve_clock(std::optional<PyClockKey> clock) const {
    return context_.with_value("PYTHON_LIFECYCLE_CONTEXT_EXPIRED", [&](LifecycleContext& context) {
      const ClockKey key = clock ? clock->value() : ClockKey::default_clock();
      auto resolved = context.resolve_clock(key);
      if (!resolved) {
        throw RuntimeError{resolved.error()};
      }
      return *resolved;
    });
  }

  void invalidate() noexcept { context_.invalidate(); }

 private:
  PyCallbackBorrow<LifecycleContext> context_;
  bool has_parameters_{};
};

// Owns one runtime Python instance and the callbacks exposed through the C-shaped host table.
struct PyHostedCallbackContext {
  GilSafeObject instance;
  GilSafeObject setup;
  GilSafeObject contract;
  GilSafeObject compute;
  GilSafeObject start;
  GilSafeObject stop;
  std::shared_ptr<PyNativeOperatorBindings> bindings;
  PyLifecycleHookBindings lifecycle;
  bool has_parameters{};
  std::array<char, kForeignOperatorMessageCapacity> error{};
  std::size_t error_size{};
};

// Copies callback failure text into bounded storage owned by the callback context.
[[nodiscard]] ForeignOperatorResult python_callback_failure(
    PyHostedCallbackContext& context, std::string_view message,
    ErrorCode code = ErrorCode::kFailure) noexcept {
  context.error_size = std::min(message.size(), context.error.size());
  while (context.error_size != 0U && !detail::valid_utf8(message.substr(0U, context.error_size))) {
    --context.error_size;
  }
  if (context.error_size != 0U) {
    std::memcpy(context.error.data(), message.data(), context.error_size);
  }
  return ForeignOperatorResult::failure(code, context.error.data(), context.error_size);
}

// Preserves a non-success HoloscanError code while rejecting misleading success-valued failures.
[[nodiscard]] ErrorCode python_exception_error_code(const nb::python_error& error) noexcept {
  try {
    const nb::handle value = error.value();
    const nb::object holoscan_error = nb::module_::import_("holoscan.errors").attr("HoloscanError");
    if (value.is_valid() && nb::isinstance(value, holoscan_error) && nb::hasattr(value, "code")) {
      const ErrorCode code = nb::cast<ErrorCode>(value.attr("code"));
      if (code != ErrorCode::kSuccess) {
        return code;
      }
    }
  } catch (...) {
    PyErr_Clear();
  }
  return ErrorCode::kFailure;
}

// Set once interpreter exit has started, after the session drain has stopped and joined the
// sessions Python still owned; see mark_python_exit_started().
[[nodiscard]] std::atomic<bool>& python_exit_started() noexcept {
  static std::atomic<bool> started{false};
  return started;
}

// Keeps one Python thread state for each Runtime thread that runs hosted callbacks. Python creates
// a thread state when a thread it did not start first takes the GIL, and deletes it when that
// thread releases its last hold. Without a retained hold, every callback would create and delete a
// thread state: a fresh frame stack per call, and threading.local data lost between calls.
class RetainedPythonThreadState {
 public:
  RetainedPythonThreadState() = default;
  RetainedPythonThreadState(const RetainedPythonThreadState&) = delete;
  RetainedPythonThreadState& operator=(const RetainedPythonThreadState&) = delete;
  RetainedPythonThreadState(RetainedPythonThreadState&&) = delete;
  RetainedPythonThreadState& operator=(RetainedPythonThreadState&&) = delete;

  // Add one hold to the calling thread's state. Call with the GIL held, once the thread has taken
  // it for the first time.
  void retain() noexcept {
    if (state_ == nullptr) {
      static_cast<void>(PyGILState_Ensure());
      state_ = PyGILState_GetThisThreadState();
    }
  }

  // Whether the calling thread holds a retained state, that is, whether it is a Runtime thread that
  // has run a hosted callback.
  [[nodiscard]] bool retained() const noexcept { return state_ != nullptr; }

  // At thread exit, drop the retained hold, which deletes the state. Once interpreter exit has
  // started, Python reclaims thread states itself during finalization, and taking the GIL could
  // block or end this thread, so the state is left alone.
  ~RetainedPythonThreadState() {
    if (state_ == nullptr || python_exit_started().load(std::memory_order_acquire) ||
        !nb::is_alive() || PyGILState_GetThisThreadState() != state_) {
      return;
    }
    // Clearing the state can run Python finalizers while the controller joins this lane. Keep
    // Core's wait restrictions active so a final session owner transfers to the ordinary reaper.
    const detail::RuntimeCallbackContextScope cleanup_scope{
        detail::RuntimeCallbackContext::kLifecycle};
    PyEval_RestoreThread(state_);
    PyGILState_Release(PyGILState_UNLOCKED);
  }

 private:
  PyThreadState* state_{};
};

[[nodiscard]] RetainedPythonThreadState& retained_python_thread_state() noexcept {
  thread_local RetainedPythonThreadState state;
  return state;
}

// Apply shutdown admission before every hosted Runtime-to-Python entry, including reconstruction.
// A Runtime thread has either no state yet or the one it retains. Python-created threads keep
// their own state, so main-thread authoring remains available at exit.
[[nodiscard]] bool python_entry_allowed() noexcept {
  return nb::is_alive() && (!python_exit_started().load(std::memory_order_acquire) ||
                            (PyGILState_GetThisThreadState() != nullptr &&
                             !retained_python_thread_state().retained()));
}

// Contains every Python exception at the foreign callback boundary and returns an owning result.
template <typename Function>
[[nodiscard]] ForeignOperatorResult invoke_python_callback(PyHostedCallbackContext& context,
                                                           Function&& function) noexcept {
  // Once interpreter exit has started, a Runtime thread can be ended by finalization inside its GIL
  // acquisition, so its callback fails without entering Python. A Runtime thread has either no
  // Python thread state yet or the one it retains. Threads that Python created, including the
  // finalizing main thread, are unaffected.
  PyThreadState* const thread_state = PyGILState_GetThisThreadState();
  if (!python_entry_allowed()) {
    return python_callback_failure(context, "PYTHON_INTERPRETER_UNAVAILABLE");
  }
  // Only a Runtime thread's first callback finds no thread state; Python's own threads have one.
  const bool new_thread_state = thread_state == nullptr;
  const nb::gil_scoped_acquire gil;
  if (!gil.is_valid()) {
    return python_callback_failure(context, "PYTHON_INTERPRETER_UNAVAILABLE");
  }
  if (new_thread_state) {
    retained_python_thread_state().retain();
  }
  const PyImportedTensorScope tensor_scope{PyImportedTensorScope::stream()};
  try {
    std::forward<Function>(function)();
    return ForeignOperatorResult::success();
  } catch (const nb::python_error& error) {
    return python_callback_failure(context, error.what(), python_exception_error_code(error));
  } catch (const RuntimeError& error) {
    const Error translated = make_error(error);
    return python_callback_failure(context, translated.message, translated.code);
  } catch (const std::exception& error) {
    return python_callback_failure(context, error.what());
  } catch (...) {
    return python_callback_failure(context, "PYTHON_OPERATOR_CALLBACK_FAILED");
  }
}

// Constructs the runtime callback table and fresh Python operator instance for one plan run.
// Keep the callback ABI table adjacent to the context it owns. The repeated adapters are easier
// to audit for noexcept/error translation here than when split across per-callback factories.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] ForeignOperatorCallbackTable make_python_callback_table(
    const PyStateCapture& capture) {
  if (!python_entry_allowed()) {
    throw RuntimeError{Error{ErrorCode::kNotReady, "PYTHON_INTERPRETER_UNAVAILABLE"}};
  }
  const nb::gil_scoped_acquire gil;
  if (!gil.is_valid()) {
    throw RuntimeError{Error{ErrorCode::kNotReady, "PYTHON_INTERPRETER_UNAVAILABLE"}};
  }
  try {
    const nb::object type = capture.type_object().borrow();
    const nb::object instance = type.attr("__new__")(type);
    const nb::dict values = nb::cast<nb::dict>(
        nb::module_::import_("copy").attr("deepcopy")(capture.values_object().borrow()));
    instance.attr("_holoscan_param_values") = values;
    instance.attr("__init__")();
    auto bindings = make_native_operator_bindings(
        capture.ports(),
        capture.parameters(),
        [parameters = capture.parameters(), definitions = capture.parameter_definitions()](
            nb::handle parameter, bool shape) {
          return hosted_parameter_byte_span(parameters, definitions, parameter, shape);
        },
        [parameters = capture.parameters(), definitions = capture.parameter_definitions()](
            nb::handle parameter, MemoryKind when_false, MemoryKind when_true) {
          return hosted_parameter_memory_kind(
              parameters, definitions, parameter, when_false, when_true);
        });
    attach_native_operator_bindings(instance, bindings);
    if (!nb::hasattr(instance, "compute")) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "PYTHON_OPERATOR_COMPUTE_REQUIRED"}};
    }
    auto* context = new PyHostedCallbackContext{
        .instance = GilSafeObject{instance},
        .setup = nb::hasattr(instance, "setup") ? GilSafeObject{instance.attr("setup")}
                                                : GilSafeObject{},
        .contract = nb::hasattr(instance, "contract") ? GilSafeObject{instance.attr("contract")}
                                                      : GilSafeObject{},
        .compute = GilSafeObject{instance.attr("compute")},
        .start = nb::hasattr(instance, "start") ? GilSafeObject{instance.attr("start")}
                                                : GilSafeObject{},
        .stop =
            nb::hasattr(instance, "stop") ? GilSafeObject{instance.attr("stop")} : GilSafeObject{},
        .bindings = std::move(bindings),
        .lifecycle = PyLifecycleHookBindings{instance},
        .has_parameters = static_cast<bool>(capture.parameters()),
    };
    return ForeignOperatorCallbackTable{
        .context = context,
        .destroy =
            [](void* value) noexcept { delete static_cast<PyHostedCallbackContext*>(value); },
        .setup_cb =
            [](void* value, ForeignOperatorSpec* spec) noexcept {
              auto& context = *static_cast<PyHostedCallbackContext*>(value);
              return invoke_python_callback(context, [&] {
                setup_native_operator(context.setup ? context.setup.borrow() : nb::none(),
                                      *spec,
                                      context.bindings,
                                      context.lifecycle);
              });
            },
        .contract_cb =
            [](void* value, Contract* contract) noexcept {
              auto& context = *static_cast<PyHostedCallbackContext*>(value);
              if (!context.contract) {
                return ForeignOperatorResult::success();
              }
              return invoke_python_callback(context, [&] {
                const nb::object authored = context.contract.borrow()();
                if (!nb::isinstance<Contract>(authored)) {
                  throw RuntimeError{
                      Error{ErrorCode::kInvalidArgument, "PYTHON_OPERATOR_CONTRACT_REQUIRED"}};
                }
                *contract = nb::cast<const Contract&>(authored);
              });
            },
        .compute_cb =
            [](void* value, ExecutionContext* execution) noexcept {
              auto& context = *static_cast<PyHostedCallbackContext*>(value);
              return invoke_python_callback(context, [&] {
                compute_native_operator(context.compute.borrow(), *execution, context.bindings);
              });
            },
        .start_cb =
            [](void* value) noexcept {
              auto& context = *static_cast<PyHostedCallbackContext*>(value);
              return context.start
                         ? invoke_python_callback(context,
                                                  [&] {
                                                    const PyImportedTensorScope tensor_scope;
                                                    context.start.borrow()();
                                                  })
                         : ForeignOperatorResult::success();
            },
        .stop_cb =
            [](void* value) noexcept {
              auto& context = *static_cast<PyHostedCallbackContext*>(value);
              return context.stop
                         ? invoke_python_callback(context,
                                                  [&] {
                                                    const PyImportedTensorScope tensor_scope;
                                                    context.stop.borrow()();
                                                  })
                         : ForeignOperatorResult::success();
            },
        .has_lifecycle_cb =
            [](void* value, LifecycleStage stage) noexcept {
              const auto& context = *static_cast<const PyHostedCallbackContext*>(value);
              return context.lifecycle.has(stage);
            },
        .lifecycle_required_cb =
            [](void* value, LifecycleStage stage) noexcept {
              const auto& context = *static_cast<const PyHostedCallbackContext*>(value);
              return context.lifecycle.required(stage);
            },
        .lifecycle_cb =
            [](void* value,
               LifecycleStage stage,
               LifecycleContext* lifecycle,
               LifecycleStatus* status) noexcept {
              auto& context = *static_cast<PyHostedCallbackContext*>(value);
              if (lifecycle == nullptr || status == nullptr || !context.lifecycle.has(stage)) {
                return python_callback_failure(
                    context, "PYTHON_LIFECYCLE_CALLBACK_INVALID", ErrorCode::kInvalidArgument);
              }
              return invoke_python_callback(context, [&] {
                auto python_context =
                    std::make_shared<PyLifecycleContext>(*lifecycle, context.has_parameters);
                try {
                  const nb::object returned =
                      context.lifecycle.callback(stage).borrow()(nb::cast(python_context));
                  if (returned.is_none()) {
                    *status = LifecycleStatus::kOk;
                  } else {
                    if (!nb::isinstance<LifecycleStatus>(returned)) {
                      throw RuntimeError{
                          Error{ErrorCode::kInvalidArgument, "PYTHON_LIFECYCLE_STATUS_REQUIRED"}};
                    }
                    *status = nb::cast<LifecycleStatus>(returned);
                  }
                } catch (...) {
                  python_context->invalidate();
                  throw;
                }
                python_context->invalidate();
              });
            },
    };
  } catch (const nb::python_error& error) {
    throw RuntimeError{Error{python_exception_error_code(error), error.what()}};
  }
}

// Core operator adapter that hosts a fresh Python instance while exposing captured parameters. Its
// `python.gil` domain is a compiler placement request rather than a runtime lock; see the capture's
// `_is_gil_enabled` check for the free-threaded-Python opt-out.
class PyHostedOperator final : public ForeignOperatorCore {
 public:
  explicit PyHostedOperator(const PyStateCapture& capture)
      // Each argument expression is fully evaluated before another begins, so the callbacks owner
      // is constructed before option creation can throw; the analyzer does not model that lifetime.
      // NOLINTBEGIN(clang-analyzer-cplusplus.NewDeleteLeaks)
      : ForeignOperatorCore(
            ForeignOperatorCallbacks{make_python_callback_table(capture)},
            ForeignOperatorOptions{.serialization_domain =
                                       capture.gil_serialization()
                                           ? std::optional<std::string>{"python.gil"}
                                           : std::nullopt}),
        parameters_(capture.parameters()) {}
  // NOLINTEND(clang-analyzer-cplusplus.NewDeleteLeaks)

  void setup(OperatorSpec& spec) override {
    ForeignOperatorCore::setup(spec);
    register_lifecycle_callback(
        spec, LifecycleStage::kConfigure, &PyHostedOperator::configure_stage);
    register_lifecycle_callback(spec, LifecycleStage::kDiscover, &PyHostedOperator::discover_stage);
    register_lifecycle_callback(spec, LifecycleStage::kAllocate, &PyHostedOperator::allocate_stage);
    register_lifecycle_callback(spec, LifecycleStage::kWarmUp, &PyHostedOperator::warm_up_stage);
    register_lifecycle_callback(spec, LifecycleStage::kArm, &PyHostedOperator::arm_stage);
    register_lifecycle_callback(spec, LifecycleStage::kRelease, &PyHostedOperator::release_stage);
  }

 private:
  using LifecycleMethod = LifecycleStatus (PyHostedOperator::*)(LifecycleContext&) noexcept;

  void register_lifecycle_callback(OperatorSpec& spec, LifecycleStage stage,
                                   LifecycleMethod method) {
    if (!has_foreign_lifecycle_callback(stage)) {
      return;
    }
    if (foreign_lifecycle_callback_required(stage)) {
      spec.lifecycle().stage(stage, method);
    } else {
      spec.lifecycle().optional(stage, method);
    }
  }

  [[nodiscard]] LifecycleStatus configure_stage(LifecycleContext& context) noexcept {
    return invoke_foreign_lifecycle_callback(LifecycleStage::kConfigure, context);
  }
  [[nodiscard]] LifecycleStatus discover_stage(LifecycleContext& context) noexcept {
    return invoke_foreign_lifecycle_callback(LifecycleStage::kDiscover, context);
  }
  [[nodiscard]] LifecycleStatus allocate_stage(LifecycleContext& context) noexcept {
    return invoke_foreign_lifecycle_callback(LifecycleStage::kAllocate, context);
  }
  [[nodiscard]] LifecycleStatus warm_up_stage(LifecycleContext& context) noexcept {
    return invoke_foreign_lifecycle_callback(LifecycleStage::kWarmUp, context);
  }
  [[nodiscard]] LifecycleStatus arm_stage(LifecycleContext& context) noexcept {
    return invoke_foreign_lifecycle_callback(LifecycleStage::kArm, context);
  }
  [[nodiscard]] LifecycleStatus release_stage(LifecycleContext& context) noexcept {
    return invoke_foreign_lifecycle_callback(LifecycleStage::kRelease, context);
  }

  [[nodiscard]] const ParamTypeDescriptor& defining_param_type_impl() const noexcept override {
    return parameters_ ? parameters_.type() : param_type_descriptor<NoParams>();
  }
  [[nodiscard]] const ParamSchemaDescriptor* defining_param_schema_impl() const noexcept override {
    return parameters_.schema();
  }
  [[nodiscard]] const void* parameter_address_impl() const noexcept override {
    return parameters_ ? static_cast<const void*>(&parameters_)
                       : static_cast<const void*>(&detail::kNoParams);
  }
  void replace_runtime_parameter_value_impl(const void*) noexcept override {}
  [[nodiscard]] bool has_no_params_impl() const noexcept override { return !parameters_; }

  HostedParameterSnapshot parameters_;
};

}  // namespace

// Distinguishes native Python Operator subclasses from registry-backed C++ operator types.
[[nodiscard]] bool is_native_operator_type(nb::handle value) {
  if (PyType_Check(value.ptr()) == 0) {
    return false;
  }
  const int result = PyObject_IsSubclass(value.ptr(), nb::type<PyNativeOperatorBase>().ptr());
  if (result < 0) {
    throw nb::python_error();
  }
  return result != 0;
}

// Adds a native Python operator when applicable and returns its erased handle and captured values.
std::optional<PreparedNativeOperator> prepare_native_operator(nb::handle type,
                                                              const nb::kwargs& parameters) {
  if (!is_native_operator_type(type)) {
    return std::nullopt;
  }
  // Stored type-erased in PreparedNativeOperator::capture; add_native_operator() casts it back.
  auto capture = std::make_shared<const PyStateCapture>(make_native_capture(type, parameters));
  nb::dict values = nb::cast<nb::dict>(
      nb::module_::import_("builtins").attr("dict")(capture->values_object().borrow()));
  std::string type_name = capture->type_name();
  HostedParameterSnapshot snapshot = capture->parameters();
  return PreparedNativeOperator{
      std::move(capture), std::move(type_name), std::move(snapshot), std::move(values)};
}

authoring::RegisteredOperatorAddition add_native_operator(Graph& graph,
                                                          std::string_view instance_name,
                                                          const PreparedNativeOperator& prepared) {
  struct Context {
    Graph& graph;
    std::string_view instance_name;
    const PyStateCapture& capture;
    std::optional<authoring::RegisteredOperatorAddition> result;
  } context{graph,
            instance_name,
            // prepare_native_operator() is the only producer, so the erased type is known.
            *std::static_pointer_cast<const PyStateCapture>(prepared.capture),
            std::nullopt};

  const auto operation = [](void* opaque) -> expected<void, Error> {
    auto& call = *static_cast<Context*>(opaque);
    auto handle = call.graph.op<PyHostedOperator>(std::string{call.instance_name}, call.capture);
    auto reference = authoring::OpaqueOperatorRef::from_handle(handle);
    auto ports = reference.ports();
    call.result.emplace(
        authoring::RegisteredOperatorAddition{std::move(reference), std::move(ports)});
    return {};
  };
  auto applied = holoscan::detail::graph_apply_atomic_mutation(
      holoscan::detail::GraphPayloadRegistrationAccess::state(graph).lock(), &context, operation);
  if (!applied) {
    throw RuntimeError{applied.error()};
  }
  if (!context.result.has_value()) {
    throw RuntimeError{Error{ErrorCode::kProtocolError, "PYTHON_NATIVE_OPERATOR_RESULT_MISSING"}};
  }
  return std::move(*context.result);
}

void mark_python_exit_started() noexcept {
  python_exit_started().store(true, std::memory_order_release);
}

void init_native_operator(nb::module_& module) {
  auto native_operator_type = python::bind_class_in_module<PyNativeOperatorBase>(
      module,
      "holoscan",
      "Operator",
      "Base class for Python-defined operators. Each graph capture and run constructs a fresh "
      "instance; acquire run-scoped resources in start(), not __init__().",
      nb::dynamic_attr());
  native_operator_type.def(nb::init<>());

  auto lifecycle_operator_ref_type = python::bind_class_in_module<PyLifecycleOperatorRef>(
      module, "holoscan.lifecycle", "LifecycleOperatorRef");
  lifecycle_operator_ref_type.def_prop_ro("id", &PyLifecycleOperatorRef::id)
      .def_prop_ro("plan_generation", &PyLifecycleOperatorRef::plan_generation)
      .def_prop_ro("run_generation", &PyLifecycleOperatorRef::run_generation);
  auto lifecycle_context_type = python::bind_class_in_module<PyLifecycleContext>(
      module, "holoscan.lifecycle", "LifecycleContext");
  lifecycle_context_type.def_prop_ro("params", &PyLifecycleContext::params)
      .def_prop_ro("params_generation", &PyLifecycleContext::params_generation)
      .def_prop_ro("transition_phase", &PyLifecycleContext::transition_phase)
      .def_prop_ro("stop_requested", &PyLifecycleContext::stop_requested)
      .def("report_error",
           &PyLifecycleContext::report_error,
           nb::arg("code"),
           nb::arg("message"),
           "Supply bounded detail for a returned failure status. The first report wins; detail "
           "reported by a successful callback is discarded.")
      .def("restart_target", &PyLifecycleContext::restart_target, nb::arg("operator_path"))
      .def("resolve_clock",
           &PyLifecycleContext::resolve_clock,
           nb::arg("clock") = nb::none(),
           "Resolve the default or named clock in this callback Operator's owning Graph.");

  auto param_type = python::bind_class_in_module<PyParamDescriptor>(
      module,
      "holoscan",
      "Param",
      "Declare one validated hosted-operator parameter. Scalar and bounded nonempty integral "
      "sequence defaults are compiler-visible; other values remain callback-only.",
      nb::is_generic());
  param_type
      .def(
          "__init__",
          [](PyParamDescriptor* self,
             nb::handle default_value,
             std::string description,
             nb::handle value_range,
             nb::handle allowed,
             bool recompile_required,
             std::optional<std::size_t>
                 max_elements) {
            if (!value_range.is_none() &&
                (!nb::isinstance<nb::tuple>(value_range) || nb::len(value_range) != 2U)) {
              throw_parameter_error("AUTHORING_PARAMETER_CONSTRAINT_INVALID");
            }
            if (!allowed.is_none() && !nb::isinstance<nb::list>(allowed) &&
                !nb::isinstance<nb::tuple>(allowed)) {
              throw_parameter_error("AUTHORING_PARAMETER_CONSTRAINT_INVALID");
            }
            if (max_elements && *max_elements == 0U) {
              throw_parameter_error("AUTHORING_PARAMETER_CONSTRAINT_INVALID");
            }
            new (self) PyParamDescriptor(default_value,
                                         std::move(description),
                                         value_range,
                                         allowed,
                                         recompile_required,
                                         max_elements);
          },
          nb::arg("default"),
          nb::kw_only(),
          nb::arg("description") = "",
          nb::arg("range") = nb::none(),
          nb::arg("allowed") = nb::none(),
          nb::arg("recompile_required") = false,
          nb::arg("max_elements") = nb::none())
      .def_prop_ro("default", &PyParamDescriptor::default_value)
      .def_prop_ro("description", &PyParamDescriptor::description)
      .def_prop_ro("range", &PyParamDescriptor::value_range)
      .def_prop_ro("allowed", &PyParamDescriptor::allowed)
      .def_prop_ro("recompile_required", &PyParamDescriptor::recompile_required)
      .def_prop_ro("max_elements", &PyParamDescriptor::max_elements)
      .def(
          "__set_name__",
          [](PyParamDescriptor& descriptor, nb::handle, std::string name) {
            descriptor.set_name(std::move(name));
          },
          nb::arg("owner"),
          nb::arg("name"))
      .def("__get__",
           [](const PyParamDescriptor& descriptor, const nb::args& args) {
             if (args.size() != 2U) {
               throw nb::type_error("Param.__get__ expects an instance and owner");
             }
             return descriptor.get(args[0]);
           })
      .def("__set__", [](const PyParamDescriptor& descriptor, nb::handle, nb::handle) {
        PyErr_Format(
            PyExc_AttributeError, "parameter '%s' is read-only", descriptor.name().c_str());
        throw nb::python_error();
      });
}

}  // namespace holoscan
