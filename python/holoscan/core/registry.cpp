/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "registry.hpp"

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <string>
#include <utility>

#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/logger/logger.hpp>

#include "payload.hpp"

namespace holoscan {
namespace {

namespace nb = nanobind;

std::string scalar_python_type(ParamLogicalType type) {
  switch (type) {
    case ParamLogicalType::kBool:
      return "bool";
    case ParamLogicalType::kSignedInteger:
    case ParamLogicalType::kUnsignedInteger:
      return "int";
    case ParamLogicalType::kFloatingPoint:
      return "float";
    case ParamLogicalType::kString:
      return "str";
    case ParamLogicalType::kSequence:
      return "Never";
    case ParamLogicalType::kFilesystemPath:
      return "str";
    case ParamLogicalType::kDuration:
      return "holoscan.time.Duration | datetime.timedelta";
    case ParamLogicalType::kObject:
      return "Never";
  }
  return "Never";
}

std::string python_literal(nb::handle value) {
  return nb::cast<std::string>(nb::repr(value));
}

std::string enum_python_type(const ParamFieldDescriptor& field) {
  std::string result{"Literal["};
  for (std::size_t index = 0U; index < field.enum_choices.count; ++index) {
    if (index != 0U) {
      result.append(", ");
    }
    const ParamEnumChoice& choice = field.enum_choices.values[index];
    result.append(python_literal(nb::str{choice.name.data(), choice.name.size()}));
    result.append(", ");
    result.append(python_literal(payload_scalar_to_python(choice.value)));
  }
  result.push_back(']');
  return result;
}

std::string python_type_name(const ParamFieldDescriptor& field) {
  std::string result;
  if (field.nested) {
    result = "dict[str, object]";
  } else if (field.logical_type == ParamLogicalType::kSequence) {
    const std::string element = field.enum_choices.count == 0U
                                    ? scalar_python_type(field.type_shape.element_or_rep_type)
                                    : enum_python_type(field);
    if (element == "Never") {
      result = "Never";
    } else {
      result = fmt::format("list[{}] | tuple[{}, ...]", element, element);
    }
  } else if (field.enum_choices.count != 0U) {
    result = enum_python_type(field);
  } else {
    result = scalar_python_type(field.logical_type);
  }
  if (field.type_shape.optional) {
    result.append(" | None");
  }
  return result;
}

nb::object constraint_scalar(ParamConstraintScalar value) {
  switch (value.kind) {
    case ParamConstraintScalarKind::kSigned:
      return nb::cast(value.signed_value);
    case ParamConstraintScalarKind::kUnsigned:
      return nb::cast(value.unsigned_value);
    case ParamConstraintScalarKind::kFloating:
      return nb::cast(value.floating_value);
    case ParamConstraintScalarKind::kNone:
      return nb::none();
  }
  return nb::none();
}

nb::dict field_metadata(const ParamFieldDescriptor& field, bool required = false) {
  nb::dict result;
  result["name"] = std::string{field.canonical_name};
  result["alias"] = field.alias.empty() ? nb::none() : nb::cast(std::string{field.alias});
  result["python_type"] = python_type_name(field);
  result["description"] = std::string{field.description};
  result["unit"] = std::string{field.unit};
  result["optional"] = field.type_shape.optional;
  result["required"] = required;
  result["nested"] = field.nested;
  result["nested_schema_id"] = nb::none();
  result["nested_parameters"] = nb::none();
  if (field.nested && field.local_operations.nested_schema != nullptr) {
    const ParamSchemaDescriptor* nested_schema = field.local_operations.nested_schema();
    if (nested_schema != nullptr) {
      result["nested_schema_id"] = std::string{nested_schema->stable_id};
      nb::list nested_parameters;
      for (const ParamFieldDescriptor& nested_field : nested_schema->fields) {
        nested_parameters.append(field_metadata(nested_field));
      }
      result["nested_parameters"] = std::move(nested_parameters);
    }
  }
  result["sensitive"] = field.sensitivity == ParamSensitivity::kSensitive;
  result["range"] = nb::none();
  if (field.constraints.has_range) {
    result["range"] = nb::make_tuple(constraint_scalar(field.constraints.range_min),
                                     constraint_scalar(field.constraints.range_max));
  }
  nb::list choices;
  if (field.enum_choices.count != 0U) {
    for (std::size_t index = 0U; index < field.enum_choices.count; ++index) {
      const ParamEnumChoice& choice = field.enum_choices.values[index];
      nb::dict item;
      item["name"] = std::string{choice.name};
      item["value"] = payload_scalar_to_python(choice.value);
      choices.append(std::move(item));
    }
  } else {
    for (std::size_t index = 0U; index < field.constraints.allowed_value_count; ++index) {
      nb::dict item;
      item["name"] = nb::none();
      item["value"] = constraint_scalar(field.constraints.allowed_values[index]);
      choices.append(std::move(item));
    }
  }
  result["choices"] = std::move(choices);
  return result;
}

std::string introspection_name(authoring::GraphParameterIntrospection value) {
  switch (value) {
    case authoring::GraphParameterIntrospection::kNone:
      return "none";
    case authoring::GraphParameterIntrospection::kDeclared:
      return "declared";
    case authoring::GraphParameterIntrospection::kPassThrough:
      return "pass_through";
  }
  return "unknown";
}

std::string introspection_name(authoring::OperatorParameterIntrospection value) {
  switch (value) {
    case authoring::OperatorParameterIntrospection::kNone:
      return "none";
    case authoring::OperatorParameterIntrospection::kDeclared:
      return "declared";
    case authoring::OperatorParameterIntrospection::kConstructorReplay:
      return "constructor_replay";
    case authoring::OperatorParameterIntrospection::kPassThrough:
      return "pass_through";
  }
  return "unknown";
}

// Translates one immutable C++ registry description into Python-owned discovery metadata.
template <typename Description>
nb::dict description_metadata(const Description& description) {
  nb::dict result;
  result["name"] = description.name;
  result["description"] = description.param_schema == nullptr
                              ? std::string{}
                              : std::string{description.param_schema->description};
  result["parameter_introspection"] = introspection_name(description.parameter_introspection);
  result["parameter_summary"] = description.parameter_repr;

  nb::list required_parameters;
  if constexpr (requires { description.required_parameters; }) {
    for (const std::string& name : description.required_parameters) {
      required_parameters.append(name);
    }
  }
  result["required_parameters"] = std::move(required_parameters);

  nb::list ports;
  auto& registry = authoring::authoring_registry();
  for (const auto& port : description.ports) {
    auto payload = registry.payload_type(port.payload_identity);
    if (!payload) {
      throw RuntimeError{payload.error()};
    }
    nb::dict item;
    item["name"] = port.name;
    item["direction"] = std::string{port.direction == PortDirection::kInput ? "input" : "output"};
    item["payload_type"] = std::string{payload->get().name};
    ports.append(std::move(item));
  }
  result["ports"] = std::move(ports);

  nb::list parameters;
  if (description.param_schema != nullptr) {
    for (const ParamFieldDescriptor& field : description.param_schema->fields) {
      const bool required = [&] {
        if constexpr (requires { description.required_parameters; }) {
          return std::find(description.required_parameters.begin(),
                           description.required_parameters.end(),
                           field.canonical_name) != description.required_parameters.end();
        }
        return false;
      }();
      parameters.append(field_metadata(field, required));
    }
  }
  result["parameters"] = std::move(parameters);
  return result;
}

// Isolates provider failures so one malformed registered Graph cannot hide healthy rows.
nb::list registered_graph_descriptions() {
  nb::list result;
  auto& registry = authoring::authoring_registry();
  for (const auto& row : registry.graphs()) {
    auto description = registry.describe_graph(row.get().name);
    if (!description) {
      HOLOSCAN_LOG_WARN("Skipping registered Graph '{}' during discovery: {}",
                        row.get().name,
                        description.error().message);
      continue;
    }
    try {
      result.append(description_metadata(*description));
    } catch (const std::exception& error) {
      HOLOSCAN_LOG_WARN("Skipping registered Graph '{}' metadata during discovery: {}",
                        row.get().name,
                        error.what());
    } catch (...) {
      HOLOSCAN_LOG_WARN("Skipping registered Graph '{}' metadata during discovery: unknown error",
                        row.get().name);
    }
  }
  return result;
}

nb::list registered_operator_descriptions() {
  nb::list result;
  auto& registry = authoring::authoring_registry();
  for (const auto& row : registry.operators()) {
    auto description = registry.describe_operator(row.get().name);
    if (!description) {
      HOLOSCAN_LOG_WARN("Skipping registered Operator '{}' during discovery: {}",
                        row.get().name,
                        description.error().message);
      continue;
    }
    try {
      result.append(description_metadata(*description));
    } catch (const std::exception& error) {
      HOLOSCAN_LOG_WARN(
          "Skipping Operator '{}' metadata during discovery: {}", row.get().name, error.what());
    } catch (...) {
      HOLOSCAN_LOG_WARN("Skipping Operator '{}' metadata during discovery: unknown error",
                        row.get().name);
    }
  }
  return result;
}

nb::tuple registered_registry_counts() {
  auto& registry = authoring::authoring_registry();
  return nb::make_tuple(registry.graphs().size(),
                        registry.operators().size(),
                        registry.payloads().size(),
                        registry.clocks().size());
}

}  // namespace

void init_registry(nb::module_& module) {
  module.def("_registered_graph_descriptions", &registered_graph_descriptions);
  module.def("_registered_operator_descriptions", &registered_operator_descriptions);
  module.def("_registered_registry_counts", &registered_registry_counts);
}

}  // namespace holoscan
