/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "graph.hpp"

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/authoring/constructor_handle.hpp>
#include <holoscan/authoring/parameter_materializer.hpp>
#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/external_topic.hpp>
#include <holoscan/core/foreign_operator.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/hosted_parameter_snapshot.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/params/config.hpp>
#include <holoscan/python/constructor_handle_capsule.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "callback_ownership.hpp"
#include "clocks.hpp"
#include "execution_authoring.hpp"
#include "graph_pydoc.hpp"
#include "lifecycle_authoring.hpp"
#include "module_name.hpp"
#include "native_io.hpp"
#include "native_operator.hpp"
#include "payload.hpp"
#include "time.hpp"
#include "topics.hpp"

namespace holoscan {
namespace {

using authoring::ParameterAssignment;
using authoring::ParameterValue;
using authoring::RegisteredGraphType;
using authoring::RegisteredOperatorType;
using authoring::RegisteredPayloadType;

[[noreturn]] void throw_parameter_error(std::string_view token, std::string_view field = {}) {
  std::string message =
      field.empty() ? std::string{token} : fmt::format("{} field_name={}", token, field);
  throw RuntimeError{Error{ErrorCode::kInvalidArgument, std::move(message)}};
}

std::string python_repr(nb::handle value) {
  return nb::cast<std::string>(nb::repr(value));
}

// Owns recursive strings, sequences, and nested assignments while registry thunks consume borrowed
// ParameterValue views created from Python keyword arguments.
class AssignmentStorage {
 public:
  std::span<const ParameterAssignment> convert(const nb::dict& values,
                                               const ParamSchemaDescriptor* schema) {
    return convert_assignments(values, schema, {});
  }

 private:
  ParameterValue convert_value(nb::handle value, const ParamFieldDescriptor* field,
                               std::string_view path) {
    if (value.is_none()) {
      return ParameterValue::none();
    }
    if (field != nullptr && field->nested) {
      if (!nb::isinstance<nb::dict>(value) || field->local_operations.nested_schema == nullptr) {
        throw_parameter_error("AUTHORING_PARAMETER_TYPE_MISMATCH", path);
      }
      const ParamSchemaDescriptor* nested_schema = field->local_operations.nested_schema();
      if (nested_schema == nullptr) {
        throw_parameter_error("AUTHORING_PARAMETER_TYPE_MISMATCH", path);
      }
      const auto nested = convert_assignments(nb::cast<nb::dict>(value), nested_schema, path);
      return ParameterValue::from(nested);
    }
    if (field != nullptr && field->enum_choices.count != 0U && nb::isinstance<nb::str>(value)) {
      return convert_string(value, path);
    }

    const ParamLogicalType logical_type =
        field == nullptr ? infer_logical_type(value) : field->logical_type;
    switch (logical_type) {
      case ParamLogicalType::kBool:
        if (!nb::isinstance<nb::bool_>(value)) {
          throw_parameter_error("AUTHORING_PARAMETER_TYPE_MISMATCH", path);
        }
        return ParameterValue::from(value.ptr() == Py_True);
      case ParamLogicalType::kSignedInteger:
      case ParamLogicalType::kUnsignedInteger:
        return convert_integer(value, path);
      case ParamLogicalType::kFloatingPoint:
        return convert_floating(value, path);
      case ParamLogicalType::kString:
        return convert_string(value, path);
      case ParamLogicalType::kSequence:
        return convert_sequence(value, field, path);
      case ParamLogicalType::kDuration:
        return ParameterValue::from(convert_duration(value, path));
      case ParamLogicalType::kObject:
        if (nb::isinstance<nb::dict>(value)) {
          const auto nested = convert_assignments(nb::cast<nb::dict>(value), nullptr, path);
          return ParameterValue::from(nested);
        }
        throw_parameter_error("AUTHORING_PARAMETER_TYPE_MISMATCH", path);
      case ParamLogicalType::kFilesystemPath:
        return convert_string(value, path);
    }
    throw_parameter_error("AUTHORING_PARAMETER_TYPE_MISMATCH", path);
  }

  static ParamLogicalType infer_logical_type(nb::handle value) {
    if (nb::isinstance<nb::bool_>(value)) {
      return ParamLogicalType::kBool;
    }
    if (nb::isinstance<nb::int_>(value)) {
      return ParamLogicalType::kSignedInteger;
    }
    if (nb::isinstance<nb::float_>(value)) {
      return ParamLogicalType::kFloatingPoint;
    }
    if (nb::isinstance<nb::str>(value)) {
      return ParamLogicalType::kString;
    }
    if (nb::isinstance<nb::list>(value) || nb::isinstance<nb::tuple>(value)) {
      return ParamLogicalType::kSequence;
    }
    return ParamLogicalType::kObject;
  }

  static ParameterValue convert_integer(nb::handle value, std::string_view path) {
    if (!nb::isinstance<nb::int_>(value) || nb::isinstance<nb::bool_>(value)) {
      throw_parameter_error("AUTHORING_PARAMETER_TYPE_MISMATCH", path);
    }
    const long long signed_value = PyLong_AsLongLong(value.ptr());
    if (PyErr_Occurred() == nullptr) {
      return ParameterValue::from(static_cast<std::int64_t>(signed_value));
    }
    PyErr_Clear();
    const unsigned long long unsigned_value = PyLong_AsUnsignedLongLong(value.ptr());
    if (PyErr_Occurred() == nullptr) {
      return ParameterValue::from(static_cast<std::uint64_t>(unsigned_value));
    }
    PyErr_Clear();
    throw_parameter_error("AUTHORING_PARAMETER_TYPE_MISMATCH", path);
  }

  static ParameterValue convert_floating(nb::handle value, std::string_view path) {
    if ((!nb::isinstance<nb::float_>(value) && !nb::isinstance<nb::int_>(value)) ||
        nb::isinstance<nb::bool_>(value)) {
      throw_parameter_error("AUTHORING_PARAMETER_TYPE_MISMATCH", path);
    }
    const double converted = PyFloat_AsDouble(value.ptr());
    if (PyErr_Occurred() != nullptr) {
      PyErr_Clear();
      throw_parameter_error("AUTHORING_PARAMETER_TYPE_MISMATCH", path);
    }
    return ParameterValue::from(converted);
  }

  ParameterValue convert_string(nb::handle value, std::string_view path) {
    if (!nb::isinstance<nb::str>(value)) {
      throw_parameter_error("AUTHORING_PARAMETER_TYPE_MISMATCH", path);
    }
    strings_.push_back(nb::cast<std::string>(value));
    return ParameterValue::from(std::string_view{strings_.back()});
  }

  ParameterValue convert_sequence(nb::handle value, const ParamFieldDescriptor* field,
                                  std::string_view path) {
    if (!nb::isinstance<nb::list>(value) && !nb::isinstance<nb::tuple>(value)) {
      throw_parameter_error("AUTHORING_PARAMETER_TYPE_MISMATCH", path);
    }
    const nb::sequence sequence = nb::cast<nb::sequence>(value);
    const std::size_t sequence_size = nb::len(sequence);
    std::vector<ParameterValue> converted;
    converted.reserve(sequence_size);
    ParamFieldDescriptor element_field{};
    if (field != nullptr) {
      element_field.logical_type = field->type_shape.element_or_rep_type;
      element_field.type_shape.type = field->type_shape.element_or_rep_type;
      element_field.type_shape.bit_width = field->type_shape.element_or_rep_bit_width;
      element_field.enum_choices = field->enum_choices;
    }
    for (std::size_t index = 0U; index < sequence_size; ++index) {
      converted.push_back(
          convert_value(sequence[index], field == nullptr ? nullptr : &element_field, path));
    }
    sequences_.push_back(std::move(converted));
    return ParameterValue::from(std::span<const ParameterValue>{sequences_.back()});
  }

  static std::int64_t convert_duration(nb::handle value, std::string_view path) {
    return python_duration_nanoseconds(value, path);
  }

  std::span<const ParameterAssignment> convert_assignments(const nb::dict& values,
                                                           const ParamSchemaDescriptor* schema,
                                                           std::string_view prefix) {
    std::vector<ParameterAssignment> converted;
    converted.reserve(values.size());
    const nb::list items = values.items();
    for (const nb::handle item_handle : items) {
      const nb::tuple item = nb::cast<nb::tuple>(item_handle);
      if (!nb::isinstance<nb::str>(item[0])) {
        throw_parameter_error("AUTHORING_PARAMETER_UNKNOWN", prefix);
      }
      strings_.push_back(nb::cast<std::string>(item[0]));
      const std::string_view name = strings_.back();
      const auto lookup = schema == nullptr
                              ? authoring::detail::ParameterFieldLookup{}
                              : authoring::detail::lookup_parameter_field(*schema, name);
      if (lookup.ambiguous) {
        throw_parameter_error("AUTHORING_PARAMETER_AMBIGUOUS",
                              authoring::detail::parameter_field_path(prefix, name));
      }
      const ParamFieldDescriptor* field = lookup.field;
      if (schema != nullptr && field == nullptr) {
        throw_parameter_error("AUTHORING_PARAMETER_UNKNOWN",
                              authoring::detail::parameter_field_path(prefix, name));
      }
      const std::string path = authoring::detail::parameter_field_path(
          prefix, field == nullptr ? name : field->canonical_name);
      converted.push_back(ParameterAssignment{name, convert_value(item[1], field, path)});
    }
    assignments_.push_back(std::move(converted));
    return std::span<const ParameterAssignment>{assignments_.back()};
  }

  std::deque<std::string> strings_;
  std::deque<std::vector<ParameterValue>> sequences_;
  std::deque<std::vector<ParameterAssignment>> assignments_;
};

std::string safe_parameter_repr(nb::handle value, const ParamFieldDescriptor* field) {
  if (field != nullptr) {
    ParamFieldDocument document;
    document.persistence = field->persistence;
    document.visibility = field->visibility;
    document.sensitivity = field->sensitivity;
    const auto projection = classify_param_projection(
        document, ParamProjectionContext{.profile_allows = true, .codec_available = true});
    if (projection != params::ParamProjectionStatus::kIncluded) {
      return "<redacted>";
    }
  }
  if (field != nullptr && field->nested && nb::isinstance<nb::dict>(value) &&
      field->local_operations.nested_schema != nullptr) {
    const ParamSchemaDescriptor* nested_schema = field->local_operations.nested_schema();
    std::ostringstream output;
    output << '{';
    bool first = true;
    const nb::list items = nb::cast<nb::dict>(value).items();
    for (const nb::handle item_handle : items) {
      const nb::tuple item = nb::cast<nb::tuple>(item_handle);
      const std::string name = nb::cast<std::string>(item[0]);
      const ParamFieldDescriptor* nested_field =
          nested_schema == nullptr ? nullptr
                                   : authoring::detail::find_parameter_field(*nested_schema, name);
      if (!first) {
        output << ", ";
      }
      first = false;
      output << python_repr(item[0]) << ": " << safe_parameter_repr(item[1], nested_field);
    }
    output << '}';
    return output.str();
  }
  return python_repr(value);
}

std::string parameter_repr(const nb::dict& values, const ParamSchemaDescriptor* schema) {
  std::ostringstream output;
  bool first = true;
  const nb::list items = values.items();
  for (const nb::handle item_handle : items) {
    const nb::tuple item = nb::cast<nb::tuple>(item_handle);
    const std::string name = nb::cast<std::string>(item[0]);
    const ParamFieldDescriptor* field =
        schema == nullptr ? nullptr : authoring::detail::find_parameter_field(*schema, name);
    if (!first) {
      output << ", ";
    }
    first = false;
    output << name << '=' << safe_parameter_repr(item[1], field);
  }
  return output.str();
}

[[nodiscard]] RuntimeError constructor_handle_failure(ErrorCode code, std::string_view token,
                                                      std::string_view field) {
  return RuntimeError{authoring::detail::constructor_handle_error(code, token, "field", field)};
}

const authoring::ConstructorHandleInput* find_constructor_input(const RegisteredOperatorType& row,
                                                                std::string_view name) {
  if (row.constructor_inputs == nullptr) {
    return nullptr;
  }
  const auto& inputs = row.constructor_inputs->handles;
  const auto found = std::ranges::find(inputs, name, &authoring::ConstructorHandleInput::name);
  return found == inputs.end() ? nullptr : &*found;
}

// Adapts one Python callable passed for a declared callback input. The new state borrows the
// callable; the caller adopts it into the Graph's owner set once the Operator is added.
authoring::ConstructorHandle python_callback_handle(
    nb::handle value, const authoring::ConstructorHandleInput& input,
    std::vector<std::shared_ptr<PythonCallbackState>>& states) {
  if (PyCallable_Check(value.ptr()) == 0) {
    throw constructor_handle_failure(
        ErrorCode::kIncompatible, "AUTHORING_CONSTRUCTOR_HANDLE_TYPE_MISMATCH", input.name);
  }
  auto state = std::make_shared<PythonCallbackState>(value);
  states.push_back(state);
  auto callback = std::make_shared<const authoring::ConstructorCallback>(
      [state = std::move(state)] { (*state)(); });
  auto handle = authoring::ConstructorHandle::from_shared(authoring::callback_handle_type(),
                                                          std::move(callback));
  if (!handle) {
    throw RuntimeError{handle.error()};
  }
  return std::move(*handle);
}

[[nodiscard]] bool is_callback_input(const authoring::ConstructorHandleInput& input) noexcept {
  return input.type == &authoring::callback_handle_type().descriptor();
}

// Converts one declared constructor-handle keyword through the provider's capsule protocol. The
// provider's Python object is not retained; only the native owner inside the capsule is kept.
std::optional<authoring::ConstructorHandle> convert_constructor_handle(
    nb::handle value, const authoring::ConstructorHandleInput& input,
    std::vector<std::shared_ptr<PythonCallbackState>>& callback_states) {
  if (value.is_none()) {
    return std::nullopt;  // absence; the Core dispatcher enforces required inputs
  }
  if (!nb::hasattr(value, "__holoscan_constructor_handle__")) {
    // A provider may export a C++ callback through the protocol; any other callable is adapted.
    if (is_callback_input(input)) {
      return python_callback_handle(value, input, callback_states);
    }
    throw constructor_handle_failure(
        ErrorCode::kIncompatible, "AUTHORING_CONSTRUCTOR_HANDLE_PROTOCOL_MISSING", input.name);
  }
  nb::object exported;
  try {
    exported = value.attr("__holoscan_constructor_handle__")(
        nb::str(input.type->name.data(), input.type->name.size()));
  } catch (const nb::python_error&) {
    // Provider exception text may carry device details; report a bounded token instead.
    throw constructor_handle_failure(
        ErrorCode::kFailure, "AUTHORING_CONSTRUCTOR_HANDLE_EXPORT_FAILED", input.name);
  }
  if (exported.is(nb::handle{Py_NotImplemented})) {
    throw constructor_handle_failure(
        ErrorCode::kIncompatible, "AUTHORING_CONSTRUCTOR_HANDLE_TYPE_MISMATCH", input.name);
  }
  PyObject* capsule = exported.ptr();
  if (PyCapsule_IsValid(capsule, python::kConstructorHandleCapsuleName) == 0 ||
      PyCapsule_GetContext(capsule) != authoring::constructor_handle_box_identity()) {
    PyErr_Clear();
    throw constructor_handle_failure(
        ErrorCode::kInvalidArgument, "AUTHORING_CONSTRUCTOR_HANDLE_PROTOCOL_INVALID", input.name);
  }
  const auto* box = static_cast<const authoring::ConstructorHandle*>(
      PyCapsule_GetPointer(capsule, python::kConstructorHandleCapsuleName));
  // Copy before the temporary capsule, and the box it owns, is released.
  return *box;
}

std::string short_name(std::string_view name) {
  const std::size_t separator = name.rfind('.');
  return std::string{name.substr(separator == std::string_view::npos ? 0U : separator + 1U)};
}

class PyGraph;
class PyTensorProjection;

// Retains a hosted Graph interface declaration and the public-to-internal port binding pair.
struct HostedInterfaceEntry {
  PortDirection direction{PortDirection::kInput};
  const RegisteredPayloadType* payload{};
  authoring::InterfacePortPair ports;
};

// Shares one hosted Graph and its declared interface table across Python wrapper views.
struct HostedGraphControl {
  std::atomic<Graph*> graph{nullptr};
  std::map<std::string, HostedInterfaceEntry, std::less<>> interfaces;
  std::map<std::string, std::string, std::less<>> attribute_ports;
  std::string type_name;
  PyObject* composing_instance{};
  std::optional<nb::python_error> composition_error;
};

std::shared_ptr<HostedGraphControl> make_hosted_control(nb::handle instance);

// Retains the erased imported/exported endpoint that owns an external topic's hidden Graph port.
struct ExternalTopicControl {
  PortDirection direction{PortDirection::kOutput};
  authoring::OpaqueImportedEndpoint imported;
  authoring::OpaqueExportedEndpoint exported;

  [[nodiscard]] const authoring::OpaquePortRef& port() const noexcept {
    return direction == PortDirection::kOutput ? imported.port() : exported.port();
  }
};

// Wraps one declared port together with the weak graph-scoped Operator reference needed to recover
// its erased C++ authoring handle.
class PyPort {
 public:
  PyPort(authoring::OpaqueOperatorRef reference,
         std::shared_ptr<const std::vector<PortRecord>> owner_ports, PortRecord record,
         std::string owner_name, std::string owner_type_name)
      : reference_(std::move(reference)),
        operator_reference_(true),
        owner_ports_(std::move(owner_ports)),
        record_(std::move(record)),
        owner_name_(std::move(owner_name)),
        owner_type_name_(std::move(owner_type_name)) {}

  PyPort(std::weak_ptr<HostedGraphControl> control, PortDirection direction,
         const RegisteredPayloadType* payload, std::string name, std::string owner_name,
         std::string owner_type_name)
      : interface_control_(std::move(control)),
        record_{
            .direction = direction, .name = std::move(name), .payload_identity = payload->identity},
        owner_name_(std::move(owner_name)),
        owner_type_name_(std::move(owner_type_name)) {}

  PyPort(authoring::OpaqueGraphRef reference, const RegisteredGraphType* type, PortRecord record,
         std::string owner_name)
      : graph_reference_(std::move(reference)),
        graph_type_(type),
        record_(std::move(record)),
        owner_name_(std::move(owner_name)),
        owner_type_name_(short_name(type->name)) {}

  PyPort(authoring::OpaquePortRef reference, PortDirection direction,
         const RegisteredPayloadType* payload, std::string owner_name)
      : external_topic_reference_(std::move(reference)),
        record_{.direction = direction,
                .name = direction == PortDirection::kOutput ? "output" : "input",
                .payload_identity = payload->identity},
        owner_name_(std::move(owner_name)),
        owner_type_name_(direction == PortDirection::kOutput ? "ImportedTopic" : "ExportedTopic") {}

  [[nodiscard]] const std::string& name() const noexcept { return record_.name; }
  [[nodiscard]] PortDirection direction() const noexcept { return record_.direction; }
  [[nodiscard]] bool graph_valid() const noexcept {
    if (operator_reference_) {
      return reference_.graph_valid();
    }
    if (graph_type_ != nullptr) {
      return graph_reference_.graph_valid();
    }
    if (external_topic_reference_) {
      return static_cast<bool>(*external_topic_reference_);
    }
    auto control = interface_control_.lock();
    return control && control->graph.load(std::memory_order_acquire) != nullptr;
  }
  [[nodiscard]] PyPayloadType payload_type() const {
    auto row = authoring::authoring_registry().payload_type(record_.payload_identity);
    if (!row) {
      throw RuntimeError{row.error()};
    }
    return PyPayloadType{&row->get()};
  }
  [[nodiscard]] PyTensorProjection project(std::string field_name) const;
  [[nodiscard]] authoring::OpaquePortRef reference() const {
    if (operator_reference_) {
      return reference_.port(record_.name);
    }
    if (graph_type_ != nullptr) {
      return graph_reference_.interface_port(record_.name);
    }
    if (external_topic_reference_) {
      if (!static_cast<bool>(*external_topic_reference_)) {
        throw RuntimeError{
            Error{ErrorCode::kInvalidArgument, "GRAPH_EXTERNAL_TOPIC_HANDLE_INVALID"}};
      }
      return *external_topic_reference_;
    }
    auto control = interface_control_.lock();
    if (!control || control->graph.load(std::memory_order_acquire) == nullptr) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"}};
    }
    const auto found = control->interfaces.find(record_.name);
    if (found == control->interfaces.end() || found->second.direction != record_.direction) {
      throw RuntimeError{Error{ErrorCode::kNotFound, "GRAPH_PORT_NOT_FOUND"}};
    }
    return record_.direction == PortDirection::kInput ? found->second.ports.input()
                                                      : found->second.ports.output();
  }
  [[nodiscard]] std::string repr() const {
    auto payload = authoring::authoring_registry().payload_type(record_.payload_identity);
    if (!payload) {
      throw RuntimeError{payload.error()};
    }
    const std::string direction_name =
        record_.direction == PortDirection::kInput ? "Input" : "Output";
    return fmt::format("<{} '{}' of {} ({}): {}>",
                       direction_name,
                       record_.name,
                       owner_name_,
                       owner_type_name_,
                       short_name(payload->get().name));
  }

 private:
  authoring::OpaqueOperatorRef reference_;
  bool operator_reference_{};
  std::shared_ptr<const std::vector<PortRecord>> owner_ports_;
  authoring::OpaqueGraphRef graph_reference_;
  const RegisteredGraphType* graph_type_{};
  std::weak_ptr<HostedGraphControl> interface_control_;
  std::optional<authoring::OpaquePortRef> external_topic_reference_;
  PortRecord record_;
  std::string owner_name_;
  std::string owner_type_name_;

  friend class PyGraph;
  friend class PyFlow;
  friend class PyTensorProjection;
};

class PyTensorProjection {
 public:
  PyTensorProjection(PyPort source, std::string field_name)
      : source_(std::move(source)), field_name_(std::move(field_name)) {}

  [[nodiscard]] const PyPort& source() const noexcept { return source_; }
  [[nodiscard]] const std::string& field_name() const noexcept { return field_name_; }
  [[nodiscard]] bool graph_valid() const noexcept { return source_.graph_valid(); }
  [[nodiscard]] std::string repr() const {
    return fmt::format(
        "<TensorProjection {}.{} field='{}'>", source_.owner_name_, source_.name(), field_name_);
  }

 private:
  PyPort source_;
  std::string field_name_;

  friend class PyGraph;
};

PyTensorProjection PyPort::project(std::string field_name) const {
  if (direction() != PortDirection::kOutput) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_PORT_DIRECTION_MISMATCH"}};
  }
  auto row = authoring::authoring_registry().payload_type(record_.payload_identity);
  if (!row) {
    throw RuntimeError{row.error()};
  }
  const RegisteredPayloadType& payload = row->get();
  if (payload.add_tensor_projection == nullptr || payload.descriptor_tensor_fields.empty()) {
    throw RuntimeError{
        Error{ErrorCode::kNotSupported,
              fmt::format("GRAPH_TENSOR_PROJECTION_SOURCE_UNSUPPORTED port={}.{} field_name={}",
                          owner_name_,
                          name(),
                          field_name)}};
  }
  const auto field = std::ranges::find_if(
      payload.descriptor_tensor_fields,
      [&field_name](const authoring::RegisteredDescriptorTensorField& candidate) {
        return candidate.name == field_name;
      });
  if (field == payload.descriptor_tensor_fields.end()) {
    std::string known_fields;
    for (const auto& candidate : payload.descriptor_tensor_fields) {
      if (!known_fields.empty()) {
        known_fields.push_back(',');
      }
      known_fields.append(candidate.name);
    }
    throw RuntimeError{
        Error{ErrorCode::kInvalidArgument,
              fmt::format("GRAPH_PROJECTED_TENSOR_FIELD_UNKNOWN port={}.{} field_name={} "
                          "known_fields={}",
                          owner_name_,
                          name(),
                          field_name,
                          known_fields)}};
  }
  (void)reference();
  return PyTensorProjection{*this, std::move(field_name)};
}

bool python_keyword(std::string_view name) {
  const nb::str value{name.data(), name.size()};
  return nb::cast<bool>(nb::module_::import_("keyword").attr("iskeyword")(value));
}
bool valid_port_attribute(std::string_view name) {
  const nb::str value{name.data(), name.size()};
  return nb::cast<bool>(value.attr("isidentifier")()) && !python_keyword(name);
}

std::string keyword_port_name(std::string_view name) {
  if (name.size() > 1U && name.back() == '_' && python_keyword(name.substr(0U, name.size() - 1U))) {
    return std::string{name.substr(0U, name.size() - 1U)};
  }
  return std::string{name};
}

// Represents one graph-owned registered Operator instance and exposes its reflected ports and
// parameter binding route without constructing a detached Python Operator object.
class PyOperator {
 public:
  PyOperator(authoring::OpaqueOperatorRef reference, const RegisteredOperatorType* type,
             Graph* graph, std::vector<PortRecord> ports, std::string instance_name,
             nb::dict params)
      : reference_(std::move(reference)),
        type_(type),
        graph_(graph),
        ports_(std::make_shared<std::vector<PortRecord>>(std::move(ports))),
        instance_name_(std::move(instance_name)),
        type_name_(type->name),
        param_schema_(type->param_schema),
        params_(std::move(params)) {}

  PyOperator(authoring::OpaqueOperatorRef reference, Graph* graph, std::vector<PortRecord> ports,
             std::string instance_name, std::string type_name,
             HostedParameterSnapshot hosted_parameters, nb::dict params)
      : reference_(std::move(reference)),
        graph_(graph),
        ports_(std::make_shared<std::vector<PortRecord>>(std::move(ports))),
        instance_name_(std::move(instance_name)),
        type_name_(std::move(type_name)),
        hosted_parameters_(std::move(hosted_parameters)),
        param_schema_(hosted_parameters_.schema()),
        params_(std::move(params)) {}

  // Record inert constructor-handle summaries rendered by repr instead of the provider objects.
  void set_constructor_handle_summaries(std::vector<std::string> summaries) noexcept {
    handle_summaries_ = std::move(summaries);
  }

  // Bind a wrapper prepared before its registry mutation to the committed declaration.
  void publish(authoring::RegisteredOperatorAddition added) noexcept {
    reference_ = std::move(added.op);
    *ports_ = std::move(added.ports);
  }

  [[nodiscard]] const std::string& name() const { return instance_name_; }
  [[nodiscard]] std::string type_name() const { return type_name_; }
  [[nodiscard]] bool graph_valid() const noexcept { return reference_.graph_valid(); }
  [[nodiscard]] Graph& graph() const {
    if (!reference_.graph_valid() || graph_ == nullptr) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"}};
    }
    return *graph_;
  }
  [[nodiscard]] params::ParamConfigBinding config_binding(config::Path path,
                                                          params::ParamBindingMode mode) const {
    Graph& owner = graph();
    if (type_ == nullptr || type_->bind_config == nullptr) {
      throw RuntimeError{Error{ErrorCode::kNotSupported, "PARAM_CONFIG_BINDING_UNAVAILABLE"}};
    }
    return type_->bind_config(owner, instance_name_, std::move(path), mode);
  }
  [[nodiscard]] PyPort port(const std::string& name) const {
    const PortRecord* found = find_port(name);
    if (found == nullptr) {
      throw RuntimeError{
          Error{ErrorCode::kNotFound, "AUTHORING_OPERATOR_PORT_UNKNOWN name=" + std::string{name}}};
    }
    return make_port(*found);
  }
  [[nodiscard]] PyPort item_port(const std::string& name) const {
    const PortRecord* found = find_port(name);
    if (found == nullptr) {
      PyErr_SetObject(PyExc_KeyError, nb::str(name.data(), name.size()).ptr());
      throw nb::python_error();
    }
    return make_port(*found);
  }
  [[nodiscard]] PyPort attribute_port(const std::string& name) const {
    if (!valid_port_attribute(name)) {
      PyErr_Format(PyExc_AttributeError,
                   "port '%.*s' is not a valid attribute; use operator['%.*s']",
                   static_cast<int>(name.size()),
                   name.data(),
                   static_cast<int>(name.size()),
                   name.data());
      throw nb::python_error();
    }
    const bool exact = std::ranges::any_of(
        *ports_, [&](const PortRecord& port_record) { return port_record.name == name; });
    const std::string port_name = exact ? name : keyword_port_name(name);
    const PortRecord* found = find_port(port_name);
    if (found == nullptr) {
      PyErr_Format(PyExc_AttributeError,
                   "OperatorHandle '%.*s' has no port attribute '%.*s'",
                   static_cast<int>(instance_name_.size()),
                   instance_name_.data(),
                   static_cast<int>(name.size()),
                   name.data());
      throw nb::python_error();
    }
    return make_port(*found);
  }
  [[nodiscard]] nb::dict ports() const {
    nb::dict result;
    for (const PortRecord& port_record : *ports_) {
      result[port_record.name.c_str()] = nb::cast(port(port_record.name));
    }
    return result;
  }
  [[nodiscard]] std::string repr() const {
    std::string result{"<OperatorHandle '"};
    result.append(instance_name_);
    result.append("' (");
    result.append(type_name_);
    result.push_back(')');
    std::string rendered_params = parameter_repr(params_, param_schema_);
    for (const std::string& summary : handle_summaries_) {
      if (!rendered_params.empty()) {
        rendered_params.append(", ");
      }
      rendered_params.append(summary);
    }
    if (!rendered_params.empty()) {
      result.push_back(' ');
      result.append(rendered_params);
    }
    result.push_back('>');
    return result;
  }
  void add_to(RegionHandle& region) const {
    authoring::OpaqueOperatorRef::region_add(
        region, std::span<const authoring::OpaqueOperatorRef>{&reference_, 1U});
  }
  void add_to(PartitionHandle& partition) const {
    authoring::OpaqueOperatorRef::partition_add(
        partition, std::span<const authoring::OpaqueOperatorRef>{&reference_, 1U});
  }
  [[nodiscard]] const authoring::OpaqueOperatorRef& reference() const { return reference_; }
  void lifecycle_before(GraphLifecycleBuilder& lifecycle, const PyOperator& after,
                        LifecycleStage stage) const {
    authoring::OpaqueOperatorRef::lifecycle_before(
        lifecycle, reference(), after.reference(), stage);
  }
  void lifecycle_allow_self_restart(GraphLifecycleBuilder& lifecycle) const {
    authoring::OpaqueOperatorRef::lifecycle_allow_self_restart(lifecycle, reference());
  }
  void lifecycle_allow_external_restart(GraphLifecycleBuilder& lifecycle) const {
    authoring::OpaqueOperatorRef::lifecycle_allow_external_restart(lifecycle, reference());
  }
  void lifecycle_allow_restart_request(GraphLifecycleBuilder& lifecycle,
                                       const PyOperator& target) const {
    authoring::OpaqueOperatorRef::lifecycle_allow_restart_request(
        lifecycle, reference(), target.reference());
  }

 private:
  [[nodiscard]] const PortRecord* find_port(std::string_view name) const noexcept {
    const auto found = std::find_if(
        ports_->begin(), ports_->end(), [&](const auto& port) { return port.name == name; });
    return found == ports_->end() ? nullptr : &*found;
  }
  [[nodiscard]] PyPort make_port(const PortRecord& record) const {
    (void)reference_.port(record.name);
    return PyPort{reference_, ports_, record, instance_name_, short_name(type_name_)};
  }

  authoring::OpaqueOperatorRef reference_;
  const RegisteredOperatorType* type_{};
  Graph* graph_{};
  std::shared_ptr<std::vector<PortRecord>> ports_;
  std::string instance_name_;
  std::string type_name_;
  HostedParameterSnapshot hosted_parameters_;
  const ParamSchemaDescriptor* param_schema_{};
  nb::dict params_;
  std::vector<std::string> handle_summaries_;
};

// Converts Python Operator handles to the exact erased references required by policy builders.
std::vector<authoring::OpaqueOperatorRef> operator_references(const nb::args& operators) {
  std::vector<authoring::OpaqueOperatorRef> references;
  references.reserve(operators.size());
  for (std::size_t index = 0U; index < operators.size(); ++index) {
    const nb::handle value = operators[index];
    if (!nb::isinstance<PyOperator>(value)) {
      const nb::str actual_type = nb::type_name(value.type());
      PyErr_Format(PyExc_TypeError,
                   "argument %zu must be holoscan.OperatorHandle, not %.200s",
                   index + 1U,
                   actual_type.c_str());
      throw nb::python_error();
    }
    references.push_back(nb::cast<const PyOperator&>(value).reference());
  }
  return references;
}

// Adapts the erased C++ port configurator to Python's fluent mutation surface.
class PyPortConfigurator {
 public:
  explicit PyPortConfigurator(authoring::OpaquePortConfigurator configurator)
      : configurator_(std::move(configurator)) {}
  PyPortConfigurator& queue_depth(std::uint32_t depth) {
    configurator_.queue_depth(depth);
    return *this;
  }
  PyPortConfigurator& max_emits_per_compute(std::uint32_t count) {
    configurator_.max_emits_per_compute(count);
    return *this;
  }
  PyPortConfigurator& late_joiner_history_depth(std::uint32_t depth) {
    configurator_.late_joiner_history_depth(depth);
    return *this;
  }
  PyPortConfigurator& diagnostic_alias(const std::string& alias) {
    configurator_.diagnostic_alias(alias);
    return *this;
  }

 private:
  authoring::OpaquePortConfigurator configurator_;
};

// Adapts the erased C++ flow reference to Python's fluent edge-policy surface.
class PyFlowConfigurator {
 public:
  explicit PyFlowConfigurator(authoring::OpaqueFlowRef flow) : flow_(std::move(flow)) {}
  PyFlowConfigurator& queue_policy(QueuePolicy policy) {
    flow_.queue_policy(policy);
    return *this;
  }
  PyFlowConfigurator& queue_depth(std::uint32_t depth) {
    flow_.queue_depth(depth);
    return *this;
  }
  PyFlowConfigurator& topic_name(const std::string& name) {
    flow_.topic_name(name);
    return *this;
  }
  PyFlowConfigurator& memory_conversion(MemoryConversionPolicy policy) {
    flow_.memory_conversion(policy);
    return *this;
  }
  PyFlowConfigurator& max_inflight_samples(std::uint32_t count) {
    flow_.max_inflight_samples(count);
    return *this;
  }

 private:
  authoring::OpaqueFlowRef flow_;
};

// Keeps the erased flow and its endpoint wrappers together for inspection and later configuration.
class PyFlow {
 public:
  PyFlow(authoring::OpaqueFlowRef flow, PyPort source, PyPort target,
         std::optional<PyTensorProjection> tensor_projection = std::nullopt)
      : flow_(std::move(flow)),
        source_(std::move(source)),
        target_(std::move(target)),
        tensor_projection_(std::move(tensor_projection)) {}
  [[nodiscard]] bool graph_valid() const noexcept { return flow_.graph_valid(); }
  [[nodiscard]] const PyPort& source() const noexcept { return source_; }
  [[nodiscard]] const PyPort& target() const noexcept { return target_; }
  [[nodiscard]] const PyTensorProjection* tensor_projection() const noexcept {
    return tensor_projection_ ? &*tensor_projection_ : nullptr;
  }
  [[nodiscard]] std::string repr() const {
    if (tensor_projection_) {
      return fmt::format("<Flow {}.{} -> {}.{} tensor_projection='{}'>",
                         source_.owner_name_,
                         source_.name(),
                         target_.owner_name_,
                         target_.name(),
                         tensor_projection_->field_name());
    }
    return fmt::format("<Flow {}.{} -> {}.{}>",
                       source_.owner_name_,
                       source_.name(),
                       target_.owner_name_,
                       target_.name());
  }

 private:
  authoring::OpaqueFlowRef flow_;
  PyPort source_;
  PyPort target_;
  std::optional<PyTensorProjection> tensor_projection_;

  friend class PyGraph;
};

template <typename EnumT>
EnumT enum_option(nb::handle value, std::string_view field, std::string_view token);

template <typename T>
T unsigned_option(nb::handle value, std::string_view field, std::string_view token) {
  static_assert(std::is_unsigned_v<T>);
  if (!nb::isinstance<nb::int_>(value) || nb::isinstance<nb::bool_>(value)) {
    throw_parameter_error(token, field);
  }
  const unsigned long long result = PyLong_AsUnsignedLongLong(value.ptr());
  if (PyErr_Occurred() || result > std::numeric_limits<T>::max()) {
    PyErr_Clear();
    throw_parameter_error(token, field);
  }
  return static_cast<T>(result);
}

std::string registry_name(nb::handle type_or_name, std::string_view kind) {
  if (nb::isinstance<nb::str>(type_or_name)) {
    return nb::cast<std::string>(type_or_name);
  }
  if (nb::hasattr(type_or_name, "_registry_name")) {
    return nb::cast<std::string>(type_or_name.attr("_registry_name"));
  }
  throw RuntimeError{
      Error{ErrorCode::kInvalidArgument, fmt::format("AUTHORING_{}_TYPE_INVALID", kind)}};
}

template <typename EnumT>
EnumT enum_option(nb::handle value, std::string_view field, std::string_view token) {
  try {
    return nb::cast<EnumT>(value);
  } catch (const nb::cast_error&) {
    throw_parameter_error(token, field);
  }
}

// Owns values, such as topic names, that ConnectionOptions references while add_flow() consumes the
// translated Python keyword arguments.
class ConnectionOptionsStorage {
 public:
  // The keyword parser mirrors the public option vocabulary in one auditable dispatch chain.
  // Splitting individual cases would scatter lifetime-coupled storage from its borrowed views.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  explicit ConnectionOptionsStorage(const nb::kwargs& kwargs) {
    const nb::dict values = nb::cast<nb::dict>(kwargs);
    const nb::list items = values.items();
    for (const nb::handle item_handle : items) {
      const nb::tuple item = nb::cast<nb::tuple>(item_handle);
      const std::string name = nb::cast<std::string>(item[0]);
      const nb::handle value = item[1];
      if (name == "queue_policy") {
        if (!value.is_none()) {
          options_.queue_policy =
              enum_option<QueuePolicy>(value, name, "GRAPH_CONNECTION_OPTION_TYPE_MISMATCH");
        }
      } else if (name == "queue_depth") {
        if (!value.is_none()) {
          options_.queue_depth =
              unsigned_option<std::uint32_t>(value, name, "GRAPH_CONNECTION_OPTION_TYPE_MISMATCH");
        }
      } else if (name == "topic_name") {
        if (!value.is_none()) {
          if (!nb::isinstance<nb::str>(value)) {
            throw_parameter_error("GRAPH_CONNECTION_OPTION_TYPE_MISMATCH", name);
          }
          topic_name_ = nb::cast<std::string>(value);
          options_.topic_name = *topic_name_;
        }
      } else if (name == "memory_conversion") {
        if (!value.is_none()) {
          options_.memory_conversion = enum_option<MemoryConversionPolicy>(
              value, name, "GRAPH_CONNECTION_OPTION_TYPE_MISMATCH");
        }
      } else if (name == "max_inflight_samples") {
        if (!value.is_none()) {
          options_.max_inflight_samples =
              unsigned_option<std::uint32_t>(value, name, "GRAPH_CONNECTION_OPTION_TYPE_MISMATCH");
        }
      } else {
        throw_parameter_error("GRAPH_CONNECTION_OPTION_UNKNOWN", name);
      }
    }
  }
  [[nodiscard]] ConnectionOptions value() const {
    auto result = options_;
    result.topic_name = topic_name_ ? std::optional<std::string_view>{*topic_name_} : std::nullopt;
    return result;
  }

 private:
  ConnectionOptions options_;
  std::optional<std::string> topic_name_;
};

nb::object construct_python_operator(Graph& graph, nb::handle type_or_name,
                                     const std::string& instance_name, const nb::kwargs& params,
                                     nb::handle callback_owner = {}) {
  if (auto prepared = prepare_native_operator(type_or_name, params)) {
    // As for registry rows, prepare the wrapper before mutation and publish after commit.
    nb::object result = nb::cast(PyOperator{authoring::OpaqueOperatorRef{},
                                            &graph,
                                            {},
                                            instance_name,
                                            std::move(prepared->type_name),
                                            std::move(prepared->parameters),
                                            std::move(prepared->values)},
                                 nb::rv_policy::move);
    nb::inst_ptr<PyOperator>(result)->publish(add_native_operator(graph, instance_name, *prepared));
    return result;
  }
  const std::string type_name = registry_name(type_or_name, "OPERATOR");
  auto row_result = authoring::authoring_registry().operator_type(type_name);
  if (!row_result) {
    throw RuntimeError{row_result.error()};
  }
  const RegisteredOperatorType& row = row_result->get();
  // Route declared constructor-handle names first; every other keyword stays an ordinary value.
  // nanobind handles have shallow constness; the loop below fills this dictionary.
  // NOLINTNEXTLINE(misc-const-correctness)
  nb::dict values;
  std::vector<authoring::ConstructorHandleAssignment> handles;
  std::vector<std::shared_ptr<PythonCallbackState>> callback_states;
  std::vector<std::string> handle_summaries;
  for (const auto& [key, value] : nb::cast<nb::dict>(params)) {
    const std::string name = nb::cast<std::string>(key);
    const authoring::ConstructorHandleInput* input = find_constructor_input(row, name);
    if (input == nullptr) {
      values[key] = value;
      continue;
    }
    auto handle = convert_constructor_handle(value, *input, callback_states);
    if (!handle) {
      handle_summaries.push_back(fmt::format("{}=None", input->name));
      continue;
    }
    const bool python_callback =
        is_callback_input(*input) && !nb::hasattr(value, "__holoscan_constructor_handle__");
    handles.push_back(authoring::ConstructorHandleAssignment{input->name, std::move(*handle)});
    // A Python callback needs this interpreter when it runs; say so instead of hiding it.
    handle_summaries.push_back(
        fmt::format("{}=<{}: {}>",
                    input->name,
                    input->type->name,
                    python_callback ? "Python callable, process-local" : "process-local"));
  }
  std::vector<std::string_view> missing;
  for (const std::string_view required : row.required_parameters) {
    if (!values.contains(nb::str(required.data(), required.size()))) {
      missing.push_back(required);
    }
  }
  if (!missing.empty()) {
    std::ranges::sort(missing);
    std::ostringstream names;
    for (std::size_t index = 0; index < missing.size(); ++index) {
      if (index != 0U) {
        names << ", ";
      }
      names << missing[index];
    }
    throw nb::type_error(
        fmt::format("{} missing required parameters: {}", row.name, names.str()).c_str());
  }
  AssignmentStorage storage;
  const auto assignments = storage.convert(values, row.param_schema);
  // Prepare the Python wrapper before mutation so that nothing fallible follows the commit.
  PyOperator prepared{authoring::OpaqueOperatorRef{}, &row, &graph, {}, instance_name, values};
  prepared.set_constructor_handle_summaries(std::move(handle_summaries));
  nb::object result = nb::cast(std::move(prepared), nb::rv_policy::move);
  // This Graph object owns adapted Python callbacks where cyclic GC can see them.
  PythonCallbackOwners* callback_owners = nullptr;
  if (!callback_states.empty()) {
    if (!callback_owner.is_valid()) {
      throw RuntimeError{
          Error{ErrorCode::kNotSupported, "PYTHON_CONSTRUCTOR_CALLBACK_OWNER_REQUIRED"}};
    }
    callback_owners = python_callback_owners(callback_owner, true);
    callback_owners->reserve(callback_states.size());
  }
  auto added = authoring::add_registered_operator(
      graph,
      row,
      instance_name,
      authoring::OperatorConstructionInputs{.values = assignments, .handles = handles});
  if (!added) {
    throw RuntimeError{added.error()};
  }
  for (const auto& state : callback_states) {
    callback_owners->adopt(state);
  }
  nb::inst_ptr<PyOperator>(result)->publish(std::move(*added));
  return result;
}

class PyHostedGraph;
class PyEmbeddedGraph;

constexpr const char* kCompositionFailureAttribute = "__holoscan_composition_failure__";
constexpr const char* kEmbeddedInstances = "__holoscan_embedded_graph_instances__";

// Owns the C++ authoring Graph and mediates graph-scoped wrappers created through erased registry
// rows, preserving Core's native lifetime and diagnostic behavior.
class PyGraph {
 public:
  PyGraph(std::string name, nb::handle default_clock)
      : graph_(std::make_unique<Graph>(name)), name_(std::move(name)) {
    if (!default_clock.is_none()) {
      const PyClockKey key = add_clock_default(default_clock, "default_clock");
      graph().set_default_clock(key.value());
    }
  }

  [[nodiscard]] nb::object op(nb::handle type_or_name, const std::string& instance_name,
                              const nb::kwargs& params) {
    auto result =
        construct_python_operator(graph(), type_or_name, instance_name, params, nb::find(this));
    mark_local_authoring();
    ++operator_count_;
    return result;
  }

  [[nodiscard]] PyFlow add_flow(const PyPort& source, const PyPort& target,
                                const nb::kwargs& kwargs) {
    if (source.direction() != PortDirection::kOutput ||
        target.direction() != PortDirection::kInput) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_PORT_DIRECTION_MISMATCH"}};
    }
    const auto source_reference = source.reference();
    const auto target_reference = target.reference();
    const ConnectionOptionsStorage options{kwargs};
    auto& registry = authoring::authoring_registry();
    auto source_payload = registry.payload_type(source.record_.payload_identity);
    if (!source_payload) {
      throw RuntimeError{source_payload.error()};
    }
    auto target_payload = registry.payload_type(target.record_.payload_identity);
    if (!target_payload) {
      throw RuntimeError{target_payload.error()};
    }
    if (!authoring::payload_identities_equal(source.record_.payload_identity,
                                             target.record_.payload_identity)) {
      std::optional<std::string> suggestion;
      if (target.owner_ports_ != nullptr) {
        for (const PortRecord& candidate : *target.owner_ports_) {
          if (candidate.direction == PortDirection::kInput &&
              authoring::payload_identities_equal(source.record_.payload_identity,
                                                  candidate.payload_identity)) {
            suggestion = target.owner_name_;
            if (valid_port_attribute(candidate.name)) {
              suggestion->append(".");
              suggestion->append(candidate.name);
            } else {
              suggestion->append("[\"");
              suggestion->append(candidate.name);
              suggestion->append("\"]");
            }
            break;
          }
        }
      }
      std::string message = fmt::format(
          "GRAPH_PORT_TYPE_MISMATCH\n  source: {}.{} (Output, {}, epoch {})\n  target: {}.{} "
          "(Input, {}, epoch {})\n  Typed edges require identical payload identity.",
          source.owner_name_,
          source.record_.name,
          source_payload->get().name,
          PyPayloadType{&source_payload->get()}.epoch(),
          target.owner_name_,
          target.record_.name,
          target_payload->get().name,
          PyPayloadType{&target_payload->get()}.epoch());
      if (suggestion) {
        message += fmt::format(" Did you mean {}?", *suggestion);
      }
      throw RuntimeError{Error{ErrorCode::kIncompatible, std::move(message)}};
    }
    auto flow = source_payload->get().add_flow(
        graph(), source_reference, target_reference, options.value());
    mark_local_authoring();
    ++flow_count_;
    return PyFlow{std::move(flow), source, target};
  }

  [[nodiscard]] PyFlow add_flow(const PyTensorProjection& projection, const PyPort& target,
                                const nb::kwargs& kwargs) {
    const PyPort& source = projection.source_;
    if (source.direction() != PortDirection::kOutput ||
        target.direction() != PortDirection::kInput) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_PORT_DIRECTION_MISMATCH"}};
    }
    const auto source_reference = source.reference();
    const auto target_reference = target.reference();
    const ConnectionOptionsStorage options{kwargs};
    auto& registry = authoring::authoring_registry();
    auto source_payload = registry.payload_type(source.record_.payload_identity);
    if (!source_payload) {
      throw RuntimeError{source_payload.error()};
    }
    auto target_payload = registry.payload_type(target.record_.payload_identity);
    if (!target_payload) {
      throw RuntimeError{target_payload.error()};
    }
    if (target_payload->get().identity.kind != PayloadIdentityKind::kTensor ||
        source_payload->get().add_tensor_projection == nullptr) {
      throw RuntimeError{Error{
          ErrorCode::kIncompatible,
          fmt::format("GRAPH_TENSOR_PROJECTION_TARGET_UNSUPPORTED source={}.{} source_payload={} "
                      "field_name={} target={}.{} actual_target_payload={} "
                      "required_target_payload=holoscan.Tensor",
                      source.owner_name_,
                      source.record_.name,
                      source_payload->get().name,
                      projection.field_name_,
                      target.owner_name_,
                      target.record_.name,
                      target_payload->get().name)}};
    }
    auto flow = source_payload->get().add_tensor_projection(
        graph(), source_reference, target_reference, projection.field_name_, options.value());
    mark_local_authoring();
    ++flow_count_;
    return PyFlow{std::move(flow), source, target, projection};
  }

  [[nodiscard]] PyImportedTopic import_topic(std::string uri, const PyPayloadType& payload,
                                             const PyTopicOptions& options) {
    const RegisteredPayloadType& row = payload.row();
    const auto fields = descriptor_topic_fields(row, options);
    auto endpoint = fields ? row.import_descriptor_topic(graph(), uri, options.value(), *fields)
                           : row.import_topic(graph(), uri, options.value());
    mark_local_authoring();
    return PyImportedTopic{std::move(endpoint), &row, std::move(uri)};
  }

  [[nodiscard]] PyExportedTopic export_topic(std::string uri, const PyPayloadType& payload,
                                             const PyTopicOptions& options) {
    const RegisteredPayloadType& row = payload.row();
    const auto fields = descriptor_topic_fields(row, options);
    auto endpoint = fields ? row.export_descriptor_topic(graph(), uri, options.value(), *fields)
                           : row.export_topic(graph(), uri, options.value());
    mark_local_authoring();
    return PyExportedTopic{std::move(endpoint), &row, std::move(uri)};
  }

  [[nodiscard]] PyPortConfigurator configure(const PyPort& port) {
    const auto reference = port.reference();
    auto payload = authoring::authoring_registry().payload_type(port.record_.payload_identity);
    if (!payload) {
      throw RuntimeError{payload.error()};
    }
    mark_local_authoring();
    if (port.direction() == PortDirection::kInput) {
      return PyPortConfigurator{payload->get().configure_input(graph(), reference)};
    }
    return PyPortConfigurator{payload->get().configure_output(graph(), reference)};
  }
  [[nodiscard]] PyFlowConfigurator configure(const PyFlow& flow) {
    flow.flow_.validate_owner(graph());
    mark_local_authoring();
    return PyFlowConfigurator{flow.flow_};
  }

  [[nodiscard]] PyClockKey add_clock(nb::handle type_or_name, const std::string& name,
                                     const nb::kwargs& options) {
    std::string type_name = registry_name(type_or_name, "CLOCK");
    if (type_name.find('.') == std::string::npos) {
      type_name.insert(0U, "holoscan.");
    }
    if (options.size() != 0U) {
      if (type_name != "holoscan.RealtimeClock") {
        throw_parameter_error("GRAPH_CLOCK_OPTIONS_UNSUPPORTED", type_name);
      }
      RealtimeClock::Config config;
      const nb::dict values = nb::cast<nb::dict>(options);
      for (const nb::handle item_handle : values.items()) {
        const nb::tuple item = nb::cast<nb::tuple>(item_handle);
        const std::string field = nb::cast<std::string>(item[0]);
        try {
          if (field == "initial_time_offset") {
            config.initial_time_offset = nb::cast<double>(item[1]);
          } else if (field == "initial_time_scale") {
            config.initial_time_scale = nb::cast<double>(item[1]);
          } else if (field == "use_time_since_epoch") {
            config.use_time_since_epoch = nb::cast<bool>(item[1]);
          } else {
            throw_parameter_error("GRAPH_CLOCK_OPTION_UNKNOWN", field);
          }
        } catch (const nb::cast_error&) {
          throw_parameter_error("GRAPH_CLOCK_OPTION_TYPE_MISMATCH", field);
        }
      }
      PyClockKey result{graph().add_clock<RealtimeClock>(name, config)};
      mark_local_authoring();
      return result;
    }
    return add_clock_default(type_or_name, name);
  }
  [[nodiscard]] PyClockKey add_clock_default(nb::handle type_or_name, std::string_view name) {
    std::string type_name = registry_name(type_or_name, "CLOCK");
    if (type_name.find('.') == std::string::npos) {
      type_name.insert(0U, "holoscan.");
    }
    auto row = authoring::authoring_registry().clock_type(type_name);
    if (!row) {
      throw RuntimeError{row.error()};
    }
    PyClockKey result{row->get().add_to_graph(graph(), name)};
    mark_local_authoring();
    return result;
  }
  PyGraph& set_default_clock(const PyClockKey& key) {
    graph().set_default_clock(key.value());
    mark_local_authoring();
    return *this;
  }
  [[nodiscard]] PyClockKey default_clock() { return PyClockKey{graph().default_clock()}; }
  [[nodiscard]] PyClockKey import_clock(const std::string& name) {
    PyClockKey result{graph().import_clock(name)};
    mark_local_authoring();
    return result;
  }
  PyGraph& add_clock_conversion(const PyClockKey& from, const PyClockKey& to,
                                AffineClockConversion conversion) {
    graph().add_clock_conversion(from.value(), to.value(), conversion);
    mark_local_authoring();
    return *this;
  }
  [[nodiscard]] PyRegionHandle region(const std::string& name) {
    PyRegionHandle result{graph().region(name)};
    mark_local_authoring();
    return result;
  }
  [[nodiscard]] PyPartitionHandle partition(const std::string& name) {
    PyPartitionHandle result{graph().partition(name)};
    mark_local_authoring();
    return result;
  }
  void auto_partition_remaining() {
    graph().auto_partition_remaining();
    mark_local_authoring();
  }
  [[nodiscard]] PyGraphLifecycleBuilder lifecycle() {
    PyGraphLifecycleBuilder result{graph().lifecycle()};
    mark_local_authoring();
    return result;
  }

  [[nodiscard]] PyPort interface_port(std::string_view name, PortDirection direction,
                                      const RegisteredPayloadType& payload) const {
    auto control = hosted_control_ != nullptr ? hosted_control_ : top_level_control_;
    if (!control || control->graph.load(std::memory_order_acquire) == nullptr) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"}};
    }
    const auto found = control->interfaces.find(name);
    if (found == control->interfaces.end() || found->second.direction != direction ||
        found->second.payload != &payload) {
      throw RuntimeError{Error{ErrorCode::kNotFound, "GRAPH_PORT_NOT_FOUND"}};
    }
    return PyPort{control, direction, &payload, std::string{name}, name_, control->type_name};
  }

  [[nodiscard]] PyEmbeddedGraph embed(nb::handle owner, nb::handle graph_type,
                                      std::string instance_name, const nb::kwargs& params);
  void export_input(const PyPort& public_port, const PyPort& internal_port);
  void export_output(const PyPort& public_port, const PyPort& internal_port);
  PyGraph& bind_clock(const PyEmbeddedGraph& child, const PyClockKey& child_import,
                      const PyClockKey& parent_clock);

  void attach_host(const std::shared_ptr<HostedGraphControl>& control) {
    if (hosted_control_ != nullptr) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"}};
    }
    hosted_control_ = control;
  }

  void require_clean_embedded_constructor() const {
    if (local_authoring_) {
      throw RuntimeError{
          Error{ErrorCode::kInvalidArgument, "AUTHORING_GRAPH_CONSTRUCTOR_MUTATION_UNSUPPORTED"}};
    }
  }

  void prepare_for_compile(nb::handle instance) {
    throw_if_composition_failed();
    if (hosted_control_ != nullptr || python_composed_) {
      return;
    }
    if (python_composing_) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_COMPOSE_REENTRANT"}};
    }
    if (top_level_control_ == nullptr) {
      top_level_control_ = make_hosted_control(instance);
      top_level_control_->graph.store(graph_.get(), std::memory_order_release);
    }
    python_composing_ = true;
    try {
      instance.attr("compose")();
      python_composed_ = true;
      python_composing_ = false;
    } catch (const nb::python_error& error) {
      python_composing_ = false;
      try {
        retain_python_composition_failure(instance, error);
        discard_failed_draft();
      } catch (...) {
        discard_failed_draft();
        throw;
      }
      throw;
    } catch (...) {
      python_composing_ = false;
      native_composition_error_ = std::current_exception();
      discard_failed_draft();
      throw;
    }
  }

  [[nodiscard]] const std::string& name() const { return name_; }
  [[nodiscard]] Graph& graph() {
    throw_if_composition_failed();
    if (hosted_control_ != nullptr) {
      Graph* hosted_graph = hosted_control_->graph.load(std::memory_order_acquire);
      if (hosted_graph == nullptr) {
        throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"}};
      }
      return *hosted_graph;
    }
    return *graph_;
  }
  [[nodiscard]] std::string repr() const {
    return fmt::format("<Graph '{}' operators={} flows={}>", name_, operator_count_, flow_count_);
  }

 private:
  void discard_failed_draft() noexcept {
    if (top_level_control_ != nullptr) {
      top_level_control_->graph.store(nullptr, std::memory_order_release);
      top_level_control_.reset();
    }
    graph_.reset();
    operator_count_ = 0U;
    local_authoring_ = false;
  }

  void retain_python_composition_failure(nb::handle instance, const nb::python_error& error) {
    nb::weakref owner{instance};
    const nb::dict attributes = nb::cast<nb::dict>(instance.attr("__dict__"));
    attributes[kCompositionFailureAttribute] = error.value();
    python_composition_error_owner_.emplace(std::move(owner));
  }

  [[noreturn]] void rethrow_python_composition_failure() const {
    if (!python_composition_error_owner_.has_value()) {
      throw RuntimeError{Error{ErrorCode::kFailure, "GRAPH_COMPOSE_FAILURE_INVALID"}};
    }
    const nb::object owner = python_composition_error_owner_.value()();
    if (owner.is_none()) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"}};
    }
    const nb::dict attributes = nb::cast<nb::dict>(owner.attr("__dict__"));
    nb::object error = nb::cast<nb::object>(attributes[kCompositionFailureAttribute]);
    if (!PyExceptionInstance_Check(error.ptr())) {
      throw RuntimeError{Error{ErrorCode::kFailure, "GRAPH_COMPOSE_FAILURE_INVALID"}};
    }
    PyErr_SetRaisedException(error.release().ptr());
    throw nb::python_error();
  }

  void throw_if_composition_failed() const {
    if (python_composition_error_owner_) {
      rethrow_python_composition_failure();
    }
    if (native_composition_error_) {
      std::rethrow_exception(native_composition_error_);
    }
  }

  void mark_local_authoring() noexcept {
    if (hosted_control_ == nullptr) {
      local_authoring_ = true;
    }
  }

  std::unique_ptr<Graph> graph_;
  std::shared_ptr<HostedGraphControl> hosted_control_;
  std::shared_ptr<HostedGraphControl> top_level_control_;
  std::string name_;
  std::size_t operator_count_{};
  std::size_t flow_count_{};
  bool local_authoring_{};
  bool python_composing_{};
  bool python_composed_{};
  std::optional<nb::weakref> python_composition_error_owner_;
  std::exception_ptr native_composition_error_;
};

// Carries the Python instance and its shared control block through the retained C++ embed factory;
// the raw PyObject is borrowed only while compose() owns the GIL.
struct HostedGraphCapture {
  PyObject* python_graph{};
  std::shared_ptr<HostedGraphControl> control;
  const void* type_identity{};
};

// Normalizes one class-level Input/Output descriptor before Core interface ports are materialized.
struct HostedInterfaceDefinition {
  PortDirection direction{PortDirection::kInput};
  const RegisteredPayloadType* payload{};
  std::string port_name;
};

// Collects inherited Python port descriptors and builds the shared interface table used by the
// hosted Graph, its Python instance, and every weak GraphHandle wrapper.
std::shared_ptr<HostedGraphControl> make_hosted_control(nb::handle instance) {
  std::map<std::string, HostedInterfaceDefinition, std::less<>> attributes;
  const nb::object python_type = nb::borrow<nb::object>(Py_TYPE(instance.ptr()));
  const nb::tuple mro = nb::cast<nb::tuple>(python_type.attr("__mro__"));
  for (std::size_t offset = 0; offset < mro.size(); ++offset) {
    const nb::handle base = mro[mro.size() - offset - 1U];
    const nb::iterable items = nb::cast<nb::iterable>(base.attr("__dict__").attr("items")());
    for (const nb::handle item_handle : items) {
      const nb::tuple item = nb::cast<nb::tuple>(item_handle);
      const std::string attribute_name = nb::cast<std::string>(item[0]);
      const nb::handle value = item[1];
      if (auto port = native_port_definition(value, attribute_name)) {
        attributes[attribute_name] =
            HostedInterfaceDefinition{port->direction, port->payload, std::move(port->port_name)};
      } else {
        attributes.erase(attribute_name);
      }
    }
  }

  auto control = std::make_shared<HostedGraphControl>();
  control->type_name = nb::cast<std::string>(python_type.attr("__name__"));
  for (const auto& [attribute_name, definition] : attributes) {
    auto [_, inserted] = control->interfaces.try_emplace(
        definition.port_name,
        HostedInterfaceEntry{definition.direction,
                             definition.payload,
                             definition.payload->make_interface_ports(definition.port_name)});
    if (!inserted) {
      throw RuntimeError{Error{ErrorCode::kDuplicateName, "GRAPH_INTERFACE_NAME_DUPLICATE"}};
    }
    control->attribute_ports.emplace(attribute_name, definition.port_name);
  }
  return control;
}

// Adapts a Python Graph subclass to Core's retained Graph model while preserving its logical Python
// type identity, compose exception, and weak owner lifetime.
class PyHostedGraph final : public Graph {
 public:
  explicit PyHostedGraph(const HostedGraphCapture& capture)
      : control_(capture.control), type_identity_(capture.type_identity) {
    control_->composing_instance = capture.python_graph;
    control_->graph.store(this, std::memory_order_release);
    nb::cast<PyGraph&>(nb::handle{capture.python_graph}).attach_host(control_);
  }

  ~PyHostedGraph() override { control_->graph.store(nullptr, std::memory_order_release); }

  [[nodiscard]] const std::shared_ptr<HostedGraphControl>& control() const noexcept {
    return control_;
  }

 protected:
  [[nodiscard]] const void* embedding_type_identity() const noexcept override {
    return type_identity_;
  }

  void compose() override {
    const nb::gil_scoped_acquire gil;
    try {
      nb::borrow<nb::object>(nb::handle{control_->composing_instance}).attr("compose")();
      control_->composing_instance = nullptr;
    } catch (nb::python_error& error) {
      const nb::handle value = error.value();
      const nb::object holoscan_error =
          nb::module_::import_("holoscan.errors").attr("HoloscanError");
      if (nb::isinstance(value, holoscan_error) && nb::hasattr(value, "code") &&
          nb::hasattr(value, "token")) {
        const ErrorCode code = nb::cast<ErrorCode>(value.attr("code"));
        const std::string token = nb::cast<std::string>(value.attr("token"));
        control_->composing_instance = nullptr;
        throw RuntimeError{Error{code, token}};
      }
      control_->composition_error.emplace(std::move(error));
      control_->composing_instance = nullptr;
      throw std::runtime_error("Python Graph compose failed");
    }
  }

 private:
  std::shared_ptr<HostedGraphControl> control_;
  const void* type_identity_{};
};

// Presents either a hosted Python child or a registered C++ child through one weak Python
// GraphHandle surface without exposing the child's private authoring state.
class PyEmbeddedGraph {
 public:
  PyEmbeddedGraph(const GraphHandle<PyHostedGraph>& handle, nb::weakref python_graph,
                  std::shared_ptr<HostedGraphControl> control, std::string instance_name)
      : reference_(authoring::OpaqueGraphRef::from_handle(handle)),
        python_graph_(std::move(python_graph)),
        hosted_control_(std::move(control)),
        instance_name_(std::move(instance_name)) {}

  PyEmbeddedGraph(authoring::OpaqueGraphRef reference, const RegisteredGraphType* type,
                  std::vector<PortRecord> ports, std::string instance_name)
      : reference_(std::move(reference)),
        type_(type),
        ports_(std::move(ports)),
        instance_name_(std::move(instance_name)) {}

  [[nodiscard]] bool graph_valid() const noexcept { return reference_.graph_valid(); }
  [[nodiscard]] const std::string& name() const noexcept { return instance_name_; }
  [[nodiscard]] std::string type_name() const {
    if (type_ != nullptr) {
      return std::string{type_->name};
    }
    if (!python_graph_.has_value()) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"}};
    }
    const nb::object instance = python_graph_.value()();
    if (instance.is_none()) {
      return hosted_control_->type_name;
    }
    const nb::object python_type = nb::borrow<nb::object>(Py_TYPE(instance.ptr()));
    return fmt::format("{}.{}",
                       nb::cast<std::string>(python_type.attr("__module__")),
                       nb::cast<std::string>(python_type.attr("__qualname__")));
  }
  [[nodiscard]] nb::object path() const {
    const auto path = reference_.path();
    if (!path) {
      return nb::none();
    }
    nb::list result;
    for (const std::string& segment : path->segments()) {
      result.append(segment);
    }
    return nb::tuple{result};
  }
  [[nodiscard]] PyPort port(const std::string& name) const {
    if (!graph_valid()) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"}};
    }
    if (type_ != nullptr) {
      const auto found = std::find_if(
          ports_.begin(), ports_.end(), [&](const auto& port) { return port.name == name; });
      if (found == ports_.end()) {
        throw RuntimeError{
            Error{ErrorCode::kNotFound, "GRAPH_PORT_NOT_FOUND name=" + std::string{name}}};
      }
      return PyPort{reference_, type_, *found, instance_name_};
    }
    const auto found = hosted_control_->interfaces.find(name);
    if (found == hosted_control_->interfaces.end()) {
      throw RuntimeError{
          Error{ErrorCode::kNotFound, "GRAPH_PORT_NOT_FOUND name=" + std::string{name}}};
    }
    return PyPort{hosted_control_,
                  found->second.direction,
                  found->second.payload,
                  name,
                  instance_name_,
                  hosted_control_->type_name};
  }
  [[nodiscard]] PyPort item_port(const std::string& name) const {
    const bool registered_port =
        type_ != nullptr &&
        std::ranges::any_of(ports_, [&](const PortRecord& port) { return port.name == name; });
    const bool hosted_port = type_ == nullptr && hosted_control_->interfaces.contains(name);
    if (!registered_port && !hosted_port) {
      PyErr_SetObject(PyExc_KeyError, nb::str(name.data(), name.size()).ptr());
      throw nb::python_error();
    }
    return port(name);
  }
  [[nodiscard]] nb::dict ports() const {
    nb::dict result;
    if (type_ != nullptr) {
      for (const PortRecord& port_record : ports_) {
        result[port_record.name.c_str()] = nb::cast(port(port_record.name));
      }
      return result;
    }
    if (!reference_) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"}};
    }
    for (const auto& [port_name, _] : hosted_control_->interfaces) {
      result[port_name.c_str()] = nb::cast(port(port_name));
    }
    return result;
  }
  [[nodiscard]] PyClockKey imported_clock(const std::string& name) const {
    return PyClockKey{reference_.imported_clock(name)};
  }
  [[nodiscard]] nb::object attribute(const std::string& name) const {
    if (!graph_valid()) {
      PyErr_Format(PyExc_AttributeError,
                   "GraphHandle '%s' has no attribute '%s'",
                   instance_name_.c_str(),
                   name.c_str());
      throw nb::python_error();
    }
    if (type_ != nullptr) {
      if (!valid_port_attribute(name)) {
        PyErr_Format(PyExc_AttributeError,
                     "port '%.*s' is not a valid attribute; use graph['%.*s']",
                     static_cast<int>(name.size()),
                     name.data(),
                     static_cast<int>(name.size()),
                     name.data());
        throw nb::python_error();
      }
      const bool exact = std::ranges::any_of(
          ports_, [&](const PortRecord& port_record) { return port_record.name == name; });
      const std::string port_name = exact ? name : keyword_port_name(name);
      const bool found = std::ranges::any_of(
          ports_, [&](const PortRecord& port_record) { return port_record.name == port_name; });
      if (found) {
        return nb::cast(port(port_name));
      }
      PyErr_Format(PyExc_AttributeError,
                   "GraphHandle '%s' has no attribute '%s'",
                   instance_name_.c_str(),
                   name.c_str());
      throw nb::python_error();
    }

    const auto port_name = hosted_control_->attribute_ports.find(name);
    if (port_name != hosted_control_->attribute_ports.end()) {
      return nb::cast(port(port_name->second));
    }
    if (!python_graph_.has_value()) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"}};
    }
    const nb::object instance = python_graph_.value()();
    if (instance.is_none() || !nb::hasattr(instance, name.c_str())) {
      PyErr_Format(PyExc_AttributeError,
                   "GraphHandle '%s' has no attribute '%s'",
                   instance_name_.c_str(),
                   name.c_str());
      throw nb::python_error();
    }
    nb::object value = instance.attr(name.c_str());
    if (nb::isinstance<nb::callable>(value)) {
      PyErr_Format(PyExc_AttributeError,
                   "GraphHandle '%s' does not expose callable attribute '%s'",
                   instance_name_.c_str(),
                   name.c_str());
      throw nb::python_error();
    }
    return value;
  }
  [[nodiscard]] std::string repr() const {
    const std::string display_type =
        type_ != nullptr ? short_name(type_->name) : hosted_control_->type_name;
    return fmt::format("<GraphHandle '{}' ({}) valid={}>",
                       instance_name_,
                       display_type,
                       graph_valid() ? "True" : "False");
  }
  void add_to(RegionHandle& region) const { reference_.region_add(region); }
  [[nodiscard]] const authoring::OpaqueGraphRef& reference() const noexcept { return reference_; }

 private:
  authoring::OpaqueGraphRef reference_;
  std::optional<nb::weakref> python_graph_;
  std::shared_ptr<HostedGraphControl> hosted_control_;
  const RegisteredGraphType* type_{};
  std::vector<PortRecord> ports_;
  std::string instance_name_;

  friend class PyGraph;
};

PyEmbeddedGraph PyGraph::embed(nb::handle owner, nb::handle graph_type, std::string instance_name,
                               const nb::kwargs& params) {
  if (nb::isinstance<nb::str>(graph_type) || nb::hasattr(graph_type, "_registry_name")) {
    const std::string type_name = registry_name(graph_type, "GRAPH");
    auto row_result = authoring::authoring_registry().graph_type(type_name);
    if (!row_result) {
      throw RuntimeError{row_result.error()};
    }
    const RegisteredGraphType& row = row_result->get();
    const nb::dict values = nb::cast<nb::dict>(params);
    AssignmentStorage storage;
    const auto assignments = storage.convert(values, row.param_schema);
    auto embedded = authoring::embed_registered_graph(graph(), row, instance_name, assignments);
    if (!embedded) {
      throw RuntimeError{embedded.error()};
    }
    mark_local_authoring();
    return PyEmbeddedGraph{
        std::move(embedded->graph), &row, std::move(embedded->ports), std::move(instance_name)};
  }

  const nb::object constructor = nb::borrow<nb::object>(graph_type);
  nb::object instance = constructor(**params);
  if (!nb::isinstance<PyGraph>(instance)) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "AUTHORING_GRAPH_TYPE_INVALID"}};
  }
  nb::cast<const PyGraph&>(instance).require_clean_embedded_constructor();
  auto control = make_hosted_control(instance);

  const nb::dict owner_attributes = nb::cast<nb::dict>(owner.attr("__dict__"));
  nb::list instances;
  if (owner_attributes.contains(kEmbeddedInstances)) {
    instances = nb::cast<nb::list>(owner_attributes[kEmbeddedInstances]);
  } else {
    instances = nb::list();
    owner_attributes[kEmbeddedInstances] = instances;
  }
  instances.append(instance);

  std::optional<GraphHandle<PyHostedGraph>> handle;
  try {
    handle.emplace(graph().embed<PyHostedGraph>(
        instance_name, HostedGraphCapture{instance.ptr(), control, Py_TYPE(instance.ptr())}));
  } catch (const RuntimeError&) {
    (void)instances.attr("pop")();
    if (control->composition_error) {
      nb::python_error error = std::move(*control->composition_error);
      control->composition_error.reset();
      throw std::move(error);
    }
    throw;
  } catch (...) {
    (void)instances.attr("pop")();
    throw;
  }
  mark_local_authoring();
  return PyEmbeddedGraph{
      std::move(*handle), nb::weakref(instance), std::move(control), std::move(instance_name)};
}

void PyGraph::export_input(const PyPort& public_port, const PyPort& internal_port) {
  if (public_port.direction() != PortDirection::kInput ||
      internal_port.direction() != PortDirection::kInput) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_PORT_DIRECTION_MISMATCH"}};
  }
  if (!authoring::payload_identities_equal(public_port.record_.payload_identity,
                                           internal_port.record_.payload_identity)) {
    throw RuntimeError{Error{ErrorCode::kIncompatible, "GRAPH_PORT_TYPE_MISMATCH"}};
  }
  auto payload = authoring::authoring_registry().payload_type(public_port.record_.payload_identity);
  if (!payload) {
    throw RuntimeError{payload.error()};
  }
  payload->get().export_input(graph(), public_port.reference(), internal_port.reference());
  mark_local_authoring();
}

void PyGraph::export_output(const PyPort& public_port, const PyPort& internal_port) {
  if (public_port.direction() != PortDirection::kOutput ||
      internal_port.direction() != PortDirection::kOutput) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_PORT_DIRECTION_MISMATCH"}};
  }
  if (!authoring::payload_identities_equal(public_port.record_.payload_identity,
                                           internal_port.record_.payload_identity)) {
    throw RuntimeError{Error{ErrorCode::kIncompatible, "GRAPH_PORT_TYPE_MISMATCH"}};
  }
  auto payload = authoring::authoring_registry().payload_type(public_port.record_.payload_identity);
  if (!payload) {
    throw RuntimeError{payload.error()};
  }
  payload->get().export_output(graph(), public_port.reference(), internal_port.reference());
  mark_local_authoring();
}

PyGraph& PyGraph::bind_clock(const PyEmbeddedGraph& child, const PyClockKey& child_import,
                             const PyClockKey& parent_clock) {
  child.reference_.bind_clock(graph(), child_import.value(), parent_clock.value());
  mark_local_authoring();
  return *this;
}

}  // namespace

authoring::OpaqueOperatorRef add_python_operator(Graph& graph, nb::handle type,
                                                 std::string_view name, const nb::dict& params) {
  const nb::object result =
      construct_python_operator(graph, type, std::string{name}, nb::borrow<nb::kwargs>(params));
  return nb::inst_ptr<PyOperator>(result)->reference();
}

void prepare_python_graph_for_compile(nb::handle value) {
  if (!nb::isinstance<PyGraph>(value)) {
    throw nb::type_error("PARAM_CONFIG_GRAPH_REQUIRED");
  }
  nb::cast<PyGraph&>(value).prepare_for_compile(value);
}

void collect_python_callback_owners(nb::handle graph, PythonCallbackOwners& into) {
  if (const PythonCallbackOwners* owners = python_callback_owners(graph, false)) {
    into.adopt_all(*owners);
  }
  // Hosted child Graphs compose through their own Python objects, which keep their own callbacks.
  const nb::dict attributes = nb::cast<nb::dict>(graph.attr("__dict__"));
  if (attributes.contains(kEmbeddedInstances)) {
    for (const nb::handle child : nb::cast<nb::list>(attributes[kEmbeddedInstances])) {
      collect_python_callback_owners(child, into);
    }
  }
}

Graph& python_graph(nb::handle value) {
  if (!nb::isinstance<PyGraph>(value)) {
    throw nb::type_error("PARAM_CONFIG_GRAPH_REQUIRED");
  }
  return nb::cast<PyGraph&>(value).graph();
}

nb::object python_external_topic_port(authoring::OpaquePortRef reference, PortDirection direction,
                                      const RegisteredPayloadType& payload,
                                      std::string owner_name) {
  return nb::cast(PyPort{std::move(reference), direction, &payload, std::move(owner_name)});
}

nb::object python_graph_interface_port(nb::handle value, const std::string& name,
                                       PortDirection direction,
                                       const RegisteredPayloadType& payload) {
  if (!nb::isinstance<PyGraph>(value)) {
    throw nb::type_error("Input or Output descriptor owner must be a Graph or Operator instance");
  }
  return nb::cast(nb::cast<PyGraph&>(value).interface_port(name, direction, payload));
}

void python_region_add(RegionHandle& region, const nb::args& operators) {
  std::vector<authoring::OpaqueOperatorRef> operator_references;
  std::vector<authoring::OpaqueGraphRef> graph_references;
  operator_references.reserve(operators.size());
  graph_references.reserve(operators.size());
  for (std::size_t index = 0U; index < operators.size(); ++index) {
    const nb::handle value = operators[index];
    if (nb::isinstance<PyOperator>(value)) {
      operator_references.push_back(nb::cast<const PyOperator&>(value).reference());
    } else if (nb::isinstance<PyEmbeddedGraph>(value)) {
      graph_references.push_back(nb::cast<const PyEmbeddedGraph&>(value).reference());
    } else {
      const nb::str actual_type = nb::type_name(value.type());
      PyErr_Format(PyExc_TypeError,
                   "argument %zu must be holoscan.OperatorHandle or holoscan.GraphHandle, not "
                   "%.200s",
                   index + 1U,
                   actual_type.c_str());
      throw nb::python_error();
    }
  }

  authoring::OpaqueOperatorRef::region_add_members(region, operator_references, graph_references);
}

void python_partition_add(PartitionHandle& partition, const nb::args& operators) {
  const auto references = operator_references(operators);
  authoring::OpaqueOperatorRef::partition_add(partition, references);
}

void python_lifecycle_before(GraphLifecycleBuilder& lifecycle, nb::handle before, nb::handle after,
                             LifecycleStage stage) {
  const auto& before_operator = nb::cast<const PyOperator&>(before);
  const auto& after_operator = nb::cast<const PyOperator&>(after);
  before_operator.lifecycle_before(lifecycle, after_operator, stage);
}

void python_lifecycle_allow_self_restart(GraphLifecycleBuilder& lifecycle, nb::handle target) {
  nb::cast<const PyOperator&>(target).lifecycle_allow_self_restart(lifecycle);
}

void python_lifecycle_allow_external_restart(GraphLifecycleBuilder& lifecycle, nb::handle target) {
  nb::cast<const PyOperator&>(target).lifecycle_allow_external_restart(lifecycle);
}

void python_lifecycle_allow_restart_request(GraphLifecycleBuilder& lifecycle, nb::handle requester,
                                            nb::handle target) {
  nb::cast<const PyOperator&>(requester).lifecycle_allow_restart_request(
      lifecycle, nb::cast<const PyOperator&>(target));
}

std::string python_operator_name(nb::handle value) {
  if (!nb::isinstance<PyOperator>(value)) {
    throw nb::type_error("PARAM_CONFIG_OPERATOR_REQUIRED");
  }
  return nb::cast<const PyOperator&>(value).name();
}

params::ParamConfigBinding python_operator_config_binding(nb::handle value, config::Path path,
                                                          params::ParamBindingMode mode) {
  if (!nb::isinstance<PyOperator>(value)) {
    throw nb::type_error("PARAM_CONFIG_OPERATOR_REQUIRED");
  }
  return nb::cast<const PyOperator&>(value).config_binding(std::move(path), mode);
}

ExecutionPartitionView python_operator_partition(const ExecutionPartitionsView& partitions,
                                                 nb::handle value) {
  if (!nb::isinstance<PyOperator>(value)) {
    throw nb::type_error("PARAM_CONFIG_OPERATOR_REQUIRED");
  }
  return nb::cast<const PyOperator&>(value).reference().execution_partition(partitions);
}

std::string python_operator_path(nb::handle value, const char* type_error_token) {
  if (!nb::isinstance<PyOperator>(value)) {
    throw nb::type_error(type_error_token);
  }
  return nb::cast<const PyOperator&>(value).reference().path();
}

ConnectionOptions python_connection_options(nb::handle options) {
  return nb::cast<const ConnectionOptionsStorage&>(options).value();
}

void init_graph(nb::module_& module) {
  auto port_direction_type =
      python::bind_enum_in_module<PortDirection>(module, "holoscan", "PortDirection");
  port_direction_type.value("INPUT", PortDirection::kInput).value("OUTPUT", PortDirection::kOutput);
  auto queue_policy_type =
      python::bind_enum_in_module<QueuePolicy>(module, "holoscan", "QueuePolicy");
  queue_policy_type.value("BLOCKING", QueuePolicy::kBlocking)
      .value("DROP_OLDEST", QueuePolicy::kDropOldest);
  python::bind_class_in_module<ConnectionOptionsStorage>(module, "holoscan", "ConnectionOptions")
      .def(nb::init<const nb::kwargs&>())
      .def_prop_ro("queue_policy",
                   [](const ConnectionOptionsStorage& v) { return v.value().queue_policy; })
      .def_prop_ro("queue_depth",
                   [](const ConnectionOptionsStorage& v) { return v.value().queue_depth; })
      .def_prop_ro("topic_name",
                   [](const ConnectionOptionsStorage& v) {
                     const auto value = v.value();
                     return value.topic_name ? std::optional<std::string>{*value.topic_name}
                                             : std::nullopt;
                   })
      .def_prop_ro("memory_conversion",
                   [](const ConnectionOptionsStorage& v) { return v.value().memory_conversion; })
      .def_prop_ro("max_inflight_samples", [](const ConnectionOptionsStorage& v) {
        return v.value().max_inflight_samples;
      });
  auto port_type =
      python::bind_class_in_module<PyPort>(module, "holoscan", "Port", doc::graph::kPort);
  port_type.def_prop_ro("name", &PyPort::name)
      .def_prop_ro("direction", &PyPort::direction)
      .def_prop_ro("payload_type", &PyPort::payload_type)
      .def_prop_ro("graph_valid", &PyPort::graph_valid)
      .def("project", &PyPort::project, nb::arg("field_name"), doc::graph::kPortProject)
      .def("__repr__", &PyPort::repr);
  auto tensor_projection_type = python::bind_class_in_module<PyTensorProjection>(
      module, "holoscan.tensor", "TensorProjection", doc::graph::kTensorProjection);
  tensor_projection_type
      .def_prop_ro("source", &PyTensorProjection::source, nb::rv_policy::reference_internal)
      .def_prop_ro("field_name", &PyTensorProjection::field_name)
      .def_prop_ro("graph_valid", &PyTensorProjection::graph_valid)
      .def("__repr__", &PyTensorProjection::repr);

  auto operator_handle_type = python::bind_class_in_module<PyOperator>(
      module, "holoscan.operators", "OperatorHandle", doc::graph::kOperatorHandle);
  operator_handle_type.def_prop_ro("name", &PyOperator::name)
      .def_prop_ro("type_name", &PyOperator::type_name)
      .def_prop_ro("graph_valid", &PyOperator::graph_valid)
      .def_prop_ro("ports",
                   &PyOperator::ports,
                   nb::for_getter(nb::sig("def ports(self, /) -> dict[str, holoscan.Port]")))
      .def("__getitem__", &PyOperator::item_port, nb::arg("name"))
      .def("__getattr__", &PyOperator::attribute_port, nb::arg("name"))
      .def("__repr__", &PyOperator::repr);

  auto graph_handle_type = python::bind_class_in_module<PyEmbeddedGraph>(
      module, "holoscan", "GraphHandle", doc::graph::kGraphHandle);
  graph_handle_type.def_prop_ro("name", &PyEmbeddedGraph::name)
      .def_prop_ro("type_name", &PyEmbeddedGraph::type_name)
      .def_prop_ro("path",
                   &PyEmbeddedGraph::path,
                   nb::for_getter(nb::sig("def path(self, /) -> tuple[str, ...] | None")))
      .def_prop_ro("graph_valid", &PyEmbeddedGraph::graph_valid)
      .def_prop_ro("ports",
                   &PyEmbeddedGraph::ports,
                   nb::for_getter(nb::sig("def ports(self, /) -> dict[str, holoscan.Port]")))
      .def("imported_clock", &PyEmbeddedGraph::imported_clock, nb::arg("name"))
      .def("__getitem__", &PyEmbeddedGraph::item_port, nb::arg("name"))
      .def("__getattr__",
           &PyEmbeddedGraph::attribute,
           nb::arg("name"),
           nb::sig("def __getattr__(self, name: str) -> object"))
      .def("__repr__", &PyEmbeddedGraph::repr);
  auto flow_type =
      python::bind_class_in_module<PyFlow>(module, "holoscan", "Flow", doc::graph::kFlow);
  flow_type.def_prop_ro("source", &PyFlow::source, nb::rv_policy::reference_internal)
      .def_prop_ro("target", &PyFlow::target, nb::rv_policy::reference_internal)
      .def_prop_ro(
          "tensor_projection", &PyFlow::tensor_projection, nb::rv_policy::reference_internal)
      .def_prop_ro("graph_valid", &PyFlow::graph_valid)
      .def("__repr__", &PyFlow::repr);

  auto port_configurator_type = python::bind_class_in_module<PyPortConfigurator>(
      module, "holoscan", "PortConfigurator", doc::graph::kPortConfigurator);
  port_configurator_type
      .def("queue_depth",
           &PyPortConfigurator::queue_depth,
           nb::arg("depth"),
           nb::rv_policy::reference_internal)
      .def("max_emits_per_compute",
           &PyPortConfigurator::max_emits_per_compute,
           nb::arg("count") = 1U,
           nb::rv_policy::reference_internal)
      .def("late_joiner_history_depth",
           &PyPortConfigurator::late_joiner_history_depth,
           nb::arg("depth"),
           nb::rv_policy::reference_internal)
      .def("diagnostic_alias",
           &PyPortConfigurator::diagnostic_alias,
           nb::arg("alias"),
           nb::rv_policy::reference_internal);

  auto flow_configurator_type = python::bind_class_in_module<PyFlowConfigurator>(
      module, "holoscan", "FlowConfigurator", doc::graph::kFlowConfigurator);
  flow_configurator_type
      .def("queue_policy",
           &PyFlowConfigurator::queue_policy,
           nb::arg("policy"),
           nb::rv_policy::reference_internal)
      .def("queue_depth",
           &PyFlowConfigurator::queue_depth,
           nb::arg("depth"),
           nb::rv_policy::reference_internal)
      .def("topic_name",
           &PyFlowConfigurator::topic_name,
           nb::arg("name"),
           nb::rv_policy::reference_internal)
      .def("memory_conversion",
           &PyFlowConfigurator::memory_conversion,
           nb::arg("policy"),
           nb::rv_policy::reference_internal)
      .def("max_inflight_samples",
           &PyFlowConfigurator::max_inflight_samples,
           nb::arg("count"),
           nb::rv_policy::reference_internal);

  auto graph_type = python::bind_class_in_module<PyGraph>(module,
                                                          "holoscan",
                                                          "Graph",
                                                          doc::graph::kGraph,
                                                          nb::dynamic_attr(),
                                                          nb::is_weak_referenceable());
  graph_type
      .def(
          "__init__",
          [](PyGraph* self, std::string name, nb::handle default_clock) {
            new (self) PyGraph(std::move(name), default_clock);
          },
          nb::arg("name") = "",
          nb::kw_only(),
          nb::arg("default_clock").none() = nb::none(),
          nb::sig("def __init__(self, name: str = '', *, "
                  "default_clock: holoscan.clocks.ClockType | str | None = None) -> None"))
      .def("op",
           &PyGraph::op,
           nb::arg("type_or_name"),
           nb::arg("instance_name"),
           nb::arg("params"),
           doc::graph::kGraphOp,
           nb::sig("def op(self, type_or_name: holoscan.registry.OperatorType | "
                   "type[holoscan.Operator] | str, "
                   "instance_name: str, **params: object) -> "
                   "holoscan.operators.OperatorHandle"))
      .def("add_flow",
           nb::overload_cast<const PyPort&, const PyPort&, const nb::kwargs&>(&PyGraph::add_flow),
           nb::arg("source"),
           nb::arg("target"),
           nb::arg("options"),
           nb::sig("def add_flow(self, source: holoscan.Port, target: holoscan.Port, *, "
                   "queue_policy: holoscan.QueuePolicy = ..., queue_depth: int | None = None, "
                   "topic_name: str | None = None, "
                   "memory_conversion: holoscan.tensor.MemoryConversionPolicy | None = None, "
                   "max_inflight_samples: int | None = None) -> holoscan.Flow"))
      .def("add_flow",
           nb::overload_cast<const PyTensorProjection&, const PyPort&, const nb::kwargs&>(
               &PyGraph::add_flow),
           nb::arg("source"),
           nb::arg("target"),
           nb::arg("options"),
           nb::sig("def add_flow(self, source: holoscan.tensor.TensorProjection, "
                   "target: holoscan.Port, *, "
                   "queue_policy: holoscan.QueuePolicy = ..., queue_depth: int | None = None, "
                   "topic_name: str | None = None, "
                   "memory_conversion: holoscan.tensor.MemoryConversionPolicy | None = None, "
                   "max_inflight_samples: int | None = None) -> holoscan.Flow"))
      .def(
          "import_topic",
          [](PyGraph& graph,
             std::string uri,
             const PyPayloadType& payload,
             const PyTopicOptions& options) {
            return graph.import_topic(std::move(uri), payload, options);
          },
          nb::arg("uri"),
          nb::arg("payload_type"),
          nb::arg("options"),
          nb::sig("def import_topic(self, uri: str, payload_type: holoscan.payload.PayloadType, "
                  "options: holoscan.topics.TopicOptions) -> holoscan.topics.ImportedTopic"))
      .def(
          "import_topic",
          [](PyGraph& graph,
             std::string uri,
             const PyPayloadType& payload,
             const nb::kwargs& options) {
            return graph.import_topic(std::move(uri), payload, PyTopicOptions{options});
          },
          nb::arg("uri"),
          nb::arg("payload_type"),
          nb::arg("topic_options"),
          nb::sig("def import_topic(self, uri: str, payload_type: holoscan.payload.PayloadType, *, "
                  "endpoint_name: str | None = None, "
                  "memory_conversion: holoscan.tensor.MemoryConversionPolicy | None = None, "
                  "schema: holoscan.payload.PayloadType | None = None, "
                  "capture_time: holoscan.clocks.ClockKey | None = None, "
                  "clock_health: holoscan.topics.ClockHealthOptions | None = None, "
                  "attach: holoscan.topics.TopicAttachPolicy = ..., "
                  "bounds: holoscan.topics.EndpointBounds = ..., "
                  "tensor: holoscan.tensor.TensorBoundaryContract | None = None, "
                  "tensor_fields: dict[str, holoscan.tensor.TensorBoundaryContract] = ...) -> "
                  "holoscan.topics.ImportedTopic"))
      .def(
          "export_topic",
          [](PyGraph& graph,
             std::string uri,
             const PyPayloadType& payload,
             const PyTopicOptions& options) {
            return graph.export_topic(std::move(uri), payload, options);
          },
          nb::arg("uri"),
          nb::arg("payload_type"),
          nb::arg("options"),
          nb::sig("def export_topic(self, uri: str, payload_type: holoscan.payload.PayloadType, "
                  "options: holoscan.topics.TopicOptions) -> holoscan.topics.ExportedTopic"))
      .def(
          "export_topic",
          [](PyGraph& graph,
             std::string uri,
             const PyPayloadType& payload,
             const nb::kwargs& options) {
            return graph.export_topic(std::move(uri), payload, PyTopicOptions{options});
          },
          nb::arg("uri"),
          nb::arg("payload_type"),
          nb::arg("topic_options"),
          nb::sig("def export_topic(self, uri: str, payload_type: holoscan.payload.PayloadType, *, "
                  "endpoint_name: str | None = None, "
                  "memory_conversion: holoscan.tensor.MemoryConversionPolicy | None = None, "
                  "schema: holoscan.payload.PayloadType | None = None, "
                  "capture_time: holoscan.clocks.ClockKey | None = None, "
                  "clock_health: holoscan.topics.ClockHealthOptions | None = None, "
                  "attach: holoscan.topics.TopicAttachPolicy = ..., "
                  "bounds: holoscan.topics.EndpointBounds = ..., "
                  "tensor: holoscan.tensor.TensorBoundaryContract | None = None, "
                  "tensor_fields: dict[str, holoscan.tensor.TensorBoundaryContract] = ...) -> "
                  "holoscan.topics.ExportedTopic"))
      .def(
          "configure",
          [](PyGraph& graph, const PyPort& port) { return graph.configure(port); },
          nb::arg("port"))
      .def(
          "configure",
          [](PyGraph& graph, const PyFlow& flow) { return graph.configure(flow); },
          nb::arg("flow"))
      .def(
          "add_clock",
          [](PyGraph& graph,
             nb::handle type_or_name,
             const std::string& name,
             const nb::kwargs& options) { return graph.add_clock(type_or_name, name, options); },
          nb::arg("type_or_name"),
          nb::arg("name"),
          nb::arg("options"),
          nb::sig("def add_clock(self, type_or_name: holoscan.clocks.ClockType | str, "
                  "name: str, *, initial_time_offset: float = 0.0, "
                  "initial_time_scale: float = 1.0, use_time_since_epoch: bool = False) -> "
                  "holoscan.clocks.ClockKey"))
      .def("set_default_clock",
           &PyGraph::set_default_clock,
           nb::arg("key"),
           nb::rv_policy::reference_internal)
      .def("default_clock", &PyGraph::default_clock)
      .def("import_clock", &PyGraph::import_clock, nb::arg("name"))
      .def("add_clock_conversion",
           &PyGraph::add_clock_conversion,
           nb::arg("from_clock"),
           nb::arg("to_clock"),
           nb::arg("conversion"),
           nb::rv_policy::reference_internal)
      .def(
          "embed",
          [](nb::handle owner,
             nb::handle graph_type,
             std::string instance_name,
             const nb::kwargs& params) {
            return nb::cast<PyGraph&>(owner).embed(
                owner, graph_type, std::move(instance_name), params);
          },
          nb::arg("graph_type"),
          nb::arg("instance_name"),
          nb::arg("params"),
          doc::graph::kGraphEmbed,
          nb::sig("def embed(self, graph_type: holoscan.registry.GraphType | "
                  "type[holoscan.Graph] | str, instance_name: str, **params: object) -> "
                  "holoscan.GraphHandle"))
      .def("export_input", &PyGraph::export_input, nb::arg("public_port"), nb::arg("internal_port"))
      .def("export_output",
           &PyGraph::export_output,
           nb::arg("public_port"),
           nb::arg("internal_port"))
      .def("bind_clock",
           &PyGraph::bind_clock,
           nb::arg("child"),
           nb::arg("child_import"),
           nb::arg("parent_clock"),
           nb::rv_policy::reference_internal)
      .def("region", &PyGraph::region, nb::arg("name"))
      .def("partition", &PyGraph::partition, nb::arg("name"))
      .def("auto_partition_remaining", &PyGraph::auto_partition_remaining)
      .def("lifecycle", &PyGraph::lifecycle)
      .def(
          "compose", [](PyGraph&) {}, doc::graph::kGraphCompose)
      .def_prop_ro("name", &PyGraph::name)
      .def("__repr__", &PyGraph::repr);
}

}  // namespace holoscan
