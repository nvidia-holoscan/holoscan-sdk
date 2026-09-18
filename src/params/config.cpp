/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/params/config.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "compiler/parameter_identity.hpp"
#include "core/canonical_text_encoding.hpp"
#include "core/parameter_input_internal.hpp"

namespace holoscan::params {

namespace detail {

class ParamConfigBindingAccess {
 public:
  [[nodiscard]] static const OperatorPath& operator_path(
      const ParamConfigBinding& binding) noexcept {
    return binding.operator_path_;
  }

  [[nodiscard]] static const config::Path& config_path(const ParamConfigBinding& binding) noexcept {
    return binding.config_path_;
  }

  [[nodiscard]] static ParamBindingMode mode(const ParamConfigBinding& binding) noexcept {
    return binding.mode_;
  }

  [[nodiscard]] static const std::weak_ptr<holoscan::detail::GraphState>& state(
      const ParamConfigBinding& binding) noexcept {
    return binding.authoring_state_;
  }

  [[nodiscard]] static const ParamConfigTypeRoutes* type_routes(
      const ParamConfigBinding& binding) noexcept {
    return binding.type_routes_;
  }

  [[nodiscard]] static VertexId target(const ParamConfigBinding& binding) noexcept {
    return binding.target_;
  }

  [[nodiscard]] static const ParamSchemaDescriptor* schema(
      const ParamConfigBinding& binding) noexcept {
    return binding.schema_;
  }
};

}  // namespace detail

namespace {

constexpr std::uint32_t kMaxConfigBindings = 1024U;
constexpr std::uint32_t kMaxConfigFields = 65536U;
constexpr std::size_t kMaxConfigProvenancePathBytes = 512U;

[[nodiscard]] ParamToolError conversion_error(ParamToolErrorCode code,
                                              std::string_view token) noexcept {
  return ParamToolError{code, token};
}

template <typename Scalar>
[[nodiscard]] ParamResult<Scalar> decode_config_scalar(config::ConfigView value,
                                                       const config::DecodeOptions& options) {
  auto decoded = config::try_decode_config<Scalar>(value, options);
  if (!decoded) {
    return make_unexpected(
        from_config_error(std::move(decoded.error()), "PARAM_CONFIG_SCALAR_INVALID"));
  }
  return *decoded;
}

template <typename Scalar>
[[nodiscard]] ParamResult<ParamValue> decode_param_scalar(config::ConfigView value,
                                                          const config::DecodeOptions& options) {
  auto decoded = decode_config_scalar<Scalar>(value, options);
  if (!decoded) {
    return make_unexpected(std::move(decoded.error()));
  }
  return ParamValue::from(*decoded);
}

[[nodiscard]] ParamResult<ParamValue> decode_scalar(const config::ConfigView& value,
                                                    const ParamFieldTypeShape& shape,
                                                    const config::DecodeOptions& options) {
  switch (shape.type) {
    case ParamLogicalType::kBool: {
      auto decoded = decode_config_scalar<bool>(value, options);
      if (!decoded) {
        return make_unexpected(std::move(decoded.error()));
      }
      return ParamValue::from(*decoded);
    }
    case ParamLogicalType::kSignedInteger:
      switch (shape.bit_width) {
        case 8:
          return decode_param_scalar<std::int8_t>(value, options);
        case 16:
          return decode_param_scalar<std::int16_t>(value, options);
        case 32:
          return decode_param_scalar<std::int32_t>(value, options);
        case 64:
          return decode_param_scalar<std::int64_t>(value, options);
        default:
          break;
      }
      break;
    case ParamLogicalType::kUnsignedInteger:
      switch (shape.bit_width) {
        case 8:
          return decode_param_scalar<std::uint8_t>(value, options);
        case 16:
          return decode_param_scalar<std::uint16_t>(value, options);
        case 32:
          return decode_param_scalar<std::uint32_t>(value, options);
        case 64:
          return decode_param_scalar<std::uint64_t>(value, options);
        default:
          break;
      }
      break;
    case ParamLogicalType::kFloatingPoint: {
      if (shape.bit_width == 32U) {
        return decode_param_scalar<float>(value, options);
      }
      if (shape.bit_width == 64U) {
        return decode_param_scalar<double>(value, options);
      }
      break;
    }
    case ParamLogicalType::kDuration: {
      auto decoded = decode_config_scalar<std::int64_t>(value, options);
      if (!decoded) {
        return make_unexpected(std::move(decoded.error()));
      }
      return ParamValue::from(std::chrono::nanoseconds{*decoded});
    }
    default:
      break;
  }
  return make_unexpected(
      conversion_error(ParamToolErrorCode::kNotSupported, "PARAM_CONFIG_SCALAR_UNSUPPORTED"));
}

[[nodiscard]] ParamResult<config::ConfigValue> translate_encoded_scalar(
    config::ConfigResult<config::ConfigValue> encoded) {
  if (!encoded) {
    return make_unexpected(
        from_config_error(std::move(encoded.error()), "PARAM_CONFIG_SCALAR_ENCODE_FAILED"));
  }
  return std::move(*encoded);
}

[[nodiscard]] ParamResult<config::ConfigValue> invalid_scalar_value() {
  return make_unexpected(
      conversion_error(ParamToolErrorCode::kInvalidSnapshot, "PARAM_CONFIG_VALUE_INVALID"));
}

template <typename Scalar, typename Encoder>
[[nodiscard]] ParamResult<config::ConfigValue> encode_typed_scalar(ParamValue value,
                                                                   Encoder&& encoder) {
  auto decoded = value.as<Scalar>();
  if (!decoded) {
    return invalid_scalar_value();
  }
  return translate_encoded_scalar(std::invoke(std::forward<Encoder>(encoder), *decoded));
}

[[nodiscard]] ParamResult<config::ConfigValue> encode_signed_scalar(ParamValue value) {
  const auto encode = [](auto decoded) { return config::ConfigValue::try_signed_integer(decoded); };
  switch (value.bit_width()) {
    case 8:
      return encode_typed_scalar<std::int8_t>(value, encode);
    case 16:
      return encode_typed_scalar<std::int16_t>(value, encode);
    case 32:
      return encode_typed_scalar<std::int32_t>(value, encode);
    case 64:
      return encode_typed_scalar<std::int64_t>(value, encode);
    default:
      return invalid_scalar_value();
  }
}

[[nodiscard]] ParamResult<config::ConfigValue> encode_unsigned_scalar(ParamValue value) {
  const auto encode = [](auto decoded) {
    return config::ConfigValue::try_unsigned_integer(decoded);
  };
  switch (value.bit_width()) {
    case 8:
      return encode_typed_scalar<std::uint8_t>(value, encode);
    case 16:
      return encode_typed_scalar<std::uint16_t>(value, encode);
    case 32:
      return encode_typed_scalar<std::uint32_t>(value, encode);
    case 64:
      return encode_typed_scalar<std::uint64_t>(value, encode);
    default:
      return invalid_scalar_value();
  }
}

[[nodiscard]] ParamResult<config::ConfigValue> encode_floating_scalar(ParamValue value) {
  double decoded{};
  if (value.bit_width() == 32U) {
    auto scalar = value.as<float>();
    if (!scalar) {
      return invalid_scalar_value();
    }
    decoded = *scalar;
  } else if (value.bit_width() == 64U) {
    auto scalar = value.as<double>();
    if (!scalar) {
      return invalid_scalar_value();
    }
    decoded = *scalar;
  } else {
    return invalid_scalar_value();
  }
  return translate_encoded_scalar(config::ConfigValue::try_floating(decoded));
}

[[nodiscard]] ParamResult<config::ConfigValue> encode_scalar(ParamValue value) {
  switch (value.kind()) {
    case ParamScalarKind::kBool:
      return encode_typed_scalar<bool>(
          value, [](bool decoded) { return config::ConfigValue::try_boolean(decoded); });
    case ParamScalarKind::kSignedInteger:
      return encode_signed_scalar(value);
    case ParamScalarKind::kUnsignedInteger:
      return encode_unsigned_scalar(value);
    case ParamScalarKind::kFloatingPoint:
      return encode_floating_scalar(value);
    case ParamScalarKind::kDurationNanoseconds:
      return encode_typed_scalar<std::chrono::nanoseconds>(value, [](auto decoded) {
        return config::ConfigValue::try_signed_integer(decoded.count());
      });
    case ParamScalarKind::kNone:
      return invalid_scalar_value();
  }
  return invalid_scalar_value();
}

struct ConfigTree {
  std::optional<config::ConfigValue> value;
  std::map<std::string, ConfigTree, std::less<>> children;
};

[[nodiscard]] ParamResult<void> insert(ConfigTree& root, const ParamPath& path,
                                       config::ConfigValue value) {
  ConfigTree* node = &root;
  for (const auto& segment : path.segments()) {
    if (node->value) {
      return make_unexpected(
          conversion_error(ParamToolErrorCode::kInvalidSnapshot, "PARAM_CONFIG_PATH_OVERLAP"));
    }
    node = &node->children[segment];
  }
  if (node->value || !node->children.empty()) {
    return make_unexpected(
        conversion_error(ParamToolErrorCode::kInvalidSnapshot, "PARAM_CONFIG_PATH_DUPLICATE"));
  }
  node->value = std::move(value);
  return {};
}

[[nodiscard]] ParamResult<config::ConfigValue> materialize(const ConfigTree& node) {
  if (node.value) {
    return *node.value;
  }
  std::vector<std::string> keys;
  std::vector<config::ConfigValue> values;
  keys.reserve(node.children.size());
  values.reserve(node.children.size());
  for (const auto& [key, child] : node.children) {
    auto value = materialize(child);
    if (!value) {
      return make_unexpected(std::move(value.error()));
    }
    keys.push_back(key);
    values.push_back(std::move(*value));
  }
  std::vector<config::ConfigObjectEntry> entries;
  entries.reserve(keys.size());
  for (std::size_t index = 0; index < keys.size(); ++index) {
    entries.push_back(config::ConfigObjectEntry{.key = keys[index], .value = values[index]});
  }
  auto object = config::ConfigValue::try_object(entries);
  if (!object) {
    return make_unexpected(
        from_config_error(std::move(object.error()), "PARAM_CONFIG_OBJECT_ENCODE_FAILED"));
  }
  return std::move(*object);
}

[[nodiscard]] std::string render_path(const OperatorPath& path) {
  std::string result;
  for (const auto& segment : path.segments()) {
    if (!result.empty()) {
      result.push_back('/');
    }
    holoscan::detail::append_percent_encoded_uri_component(result, segment);
  }
  return result;
}

[[nodiscard]] ParamResult<std::string> config_source_path(const config::Path& base,
                                                          const ParamPath& field) {
  std::string result = base.to_string();
  if (result.size() > kMaxConfigProvenancePathBytes) {
    return make_unexpected(conversion_error(ParamToolErrorCode::kResourceExhausted,
                                            "PARAM_CONFIG_SOURCE_PATH_CAPACITY"));
  }
  for (const std::string& segment : field.segments()) {
    result.push_back('/');
    for (const char value : segment) {
      if (value == '~') {
        result += "~0";
      } else if (value == '/') {
        result += "~1";
      } else {
        result.push_back(value);
      }
    }
    if (result.size() > kMaxConfigProvenancePathBytes) {
      return make_unexpected(conversion_error(ParamToolErrorCode::kResourceExhausted,
                                              "PARAM_CONFIG_SOURCE_PATH_CAPACITY"));
    }
  }
  return result;
}

[[nodiscard]] ParamToolError with_config_source_context(ParamToolError error,
                                                        const config::Path& config_path,
                                                        const ParamPath& field_path,
                                                        std::string_view message = {}) {
  auto source_path = config_source_path(config_path, field_path);
  if (!source_path) {
    return error;
  }
  const auto& cause = error.config_cause();
  auto context = config::ConfigError::make(
      cause ? cause->code() : config::ConfigErrorCode::kInvalidArgument, message, *source_path);
  if (cause) {
    context = config::ConfigError::nest(std::move(context), *cause);
  }
  return ParamToolError{error.code(), error.token(), error.core_cause(), std::move(context)};
}

[[nodiscard]] ParamToolError with_config_binding_context(ParamToolError error,
                                                         const ParamConfigBinding& binding,
                                                         const ParamPath& field_path,
                                                         std::string_view operator_path) {
  std::string message{"while binding parameters to operator '"};
  message.append(operator_path);
  message.push_back('\'');
  return with_config_source_context(std::move(error),
                                    detail::ParamConfigBindingAccess::config_path(binding),
                                    field_path,
                                    message);
}

[[nodiscard]] ParamConfigDocumentProvenance document_provenance(
    const config::Document& document) noexcept {
  const config::ConfigDigest digest = document.digest();
  ParamConfigDocumentProvenance result{
      .semantic_version = digest.semantic.canonical_version,
      .semantic_digest = digest.semantic.sha256,
  };
  if (digest.source) {
    result.has_raw_source = true;
    result.raw_algorithm_version = digest.source->algorithm_version;
    result.raw_digest = digest.source->sha256;
    result.raw_byte_size = digest.source->byte_size;
  }
  if (digest.base_source) {
    result.has_base_raw_source = true;
    result.base_raw_algorithm_version = digest.base_source->algorithm_version;
    result.base_raw_digest = digest.base_source->sha256;
    result.base_raw_byte_size = digest.base_source->byte_size;
  }
  return result;
}

[[nodiscard]] bool same_path(const ParamPath& path, std::string_view encoded) {
  auto candidate = ParamPath::parse(encoded);
  return candidate && *candidate == path;
}

struct FieldMatch {
  const ParamFieldDescriptor* descriptor{};
  bool alias{};
  bool ambiguous{};
};

[[nodiscard]] FieldMatch match_field(const ParamSchemaDescriptor& schema, const ParamPath& path) {
  FieldMatch result;
  for (const ParamFieldDescriptor& candidate : schema.fields) {
    bool alias = false;
    if (!same_path(path, candidate.canonical_name)) {
      if (candidate.alias.empty() || !same_path(path, candidate.alias)) {
        continue;
      }
      alias = true;
    }
    if (result.descriptor != nullptr) {
      result.ambiguous = true;
      return result;
    }
    result.descriptor = &candidate;
    result.alias = alias;
  }
  return result;
}

// Runtime patch decoding admits only portable scalar catalog entries, so every
// object is structural and only non-object values are field contributions.
[[nodiscard]] ParamResult<void> flatten_config_values(
    const config::ConfigView& view, std::vector<std::string>& path,
    std::vector<std::pair<ParamPath, config::ConfigView>>& leaves, std::uint32_t max_values) {
  if (view.kind() != config::ConfigValueKind::kObject) {
    if (path.empty()) {
      return make_unexpected(conversion_error(ParamToolErrorCode::kInvalidArgument,
                                              "PARAM_CONFIG_VALUES_OBJECT_REQUIRED"));
    }
    if (leaves.size() >= max_values) {
      return make_unexpected(
          conversion_error(ParamToolErrorCode::kResourceExhausted, "PARAM_CONFIG_PATCH_CAPACITY"));
    }
    auto parameter_path = ParamPath::from_segments(path);
    if (!parameter_path) {
      return make_unexpected(
          from_core_error(std::move(parameter_path.error()), "PARAM_CONFIG_FIELD_PATH_INVALID"));
    }
    leaves.emplace_back(std::move(*parameter_path), view);
    return {};
  }
  for (std::size_t index = 0; index < view.size(); ++index) {
    auto key = view.try_key(index);
    auto child = view.try_element(index);
    if (!key || !child) {
      return make_unexpected(
          conversion_error(ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_OBJECT_INVALID"));
    }
    path.emplace_back(*key);
    auto flattened = flatten_config_values(*child, path, leaves, max_values);
    path.pop_back();
    if (!flattened) {
      return make_unexpected(std::move(flattened.error()));
    }
  }
  return {};
}

[[nodiscard]] ParamResult<void> flatten_config_values(
    const config::ConfigView& view, const ParamSchemaDescriptor& schema,
    std::vector<std::string>& path, std::vector<std::pair<ParamPath, config::ConfigView>>& leaves,
    std::uint32_t max_values) {
  std::optional<ParamPath> parameter_path;
  if (!path.empty()) {
    auto parsed = ParamPath::from_segments(path);
    if (!parsed) {
      return make_unexpected(
          from_core_error(std::move(parsed.error()), "PARAM_CONFIG_FIELD_PATH_INVALID"));
    }
    parameter_path = std::move(*parsed);
  }
  const bool declared_value =
      parameter_path && match_field(schema, *parameter_path).descriptor != nullptr;
  if (view.kind() != config::ConfigValueKind::kObject || declared_value) {
    if (path.empty()) {
      return make_unexpected(conversion_error(ParamToolErrorCode::kInvalidArgument,
                                              "PARAM_CONFIG_VALUES_OBJECT_REQUIRED"));
    }
    if (leaves.size() >= max_values) {
      return make_unexpected(
          conversion_error(ParamToolErrorCode::kResourceExhausted, "PARAM_CONFIG_PATCH_CAPACITY"));
    }
    leaves.emplace_back(std::move(*parameter_path), view);
    return {};
  }
  for (std::size_t index = 0; index < view.size(); ++index) {
    auto key = view.try_key(index);
    auto child = view.try_element(index);
    if (!key || !child) {
      return make_unexpected(
          conversion_error(ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_OBJECT_INVALID"));
    }
    path.emplace_back(*key);
    auto flattened = flatten_config_values(*child, schema, path, leaves, max_values);
    path.pop_back();
    if (!flattened) {
      return make_unexpected(std::move(flattened.error()));
    }
  }
  return {};
}

struct DecodedBindingField {
  ParamFieldId field{};
  ParamSchemaHash full_id{};
  ParamFieldRebindOrigin origin{ParamFieldRebindOrigin::kCaptured};
  std::optional<std::uint32_t> config_provenance_index;
};

struct DecodedBindingFields {
  std::vector<DecodedBindingField> contributions;
  ConfigTree normalized_patch;
  std::optional<ParamPath> codec_required_path;
};

[[nodiscard]] const detail::ParamConfigTypeRoutes* config_type_routes(
    const ParamSchemaDescriptor& schema, const detail::ParamConfigTypeRoutes* routes) noexcept {
  // The schema's local type describes the factory storage passed to captured callbacks.
  return routes != nullptr && schema.local_type != nullptr &&
                 routes->local_type_token == schema.local_type->local_type_token &&
                 routes->fields.size() == schema.fields.size() &&
                 routes->schema_id == schema.stable_id &&
                 routes->schema_version == schema.version &&
                 routes->structural_hash == schema.structural_hash
             ? routes
             : nullptr;
}

[[nodiscard]] const detail::ParamConfigTypeRoutes* config_type_routes(
    const ParamSchemaDescriptor& schema, const ParamConfigBinding& binding) noexcept {
  const detail::ParamConfigTypeRoutes* const routes =
      detail::ParamConfigBindingAccess::type_routes(binding);
  return routes != nullptr && schema.local_type != nullptr &&
                 schema.local_type->local_type_token != nullptr &&
                 routes->fields.size() == schema.fields.size() &&
                 routes->schema_id == schema.stable_id &&
                 routes->schema_version == schema.version &&
                 routes->structural_hash == schema.structural_hash
             ? routes
             : nullptr;
}

[[nodiscard]] const detail::ParamConfigFieldRoute* config_field_route(
    const ParamSchemaDescriptor& schema, const ParamFieldDescriptor& field,
    const detail::ParamConfigTypeRoutes* routes) noexcept {
  if (routes == nullptr) {
    return nullptr;
  }
  const std::size_t index = static_cast<std::size_t>(&field - schema.fields.data());
  return index < routes->fields.size() ? &routes->fields[index] : nullptr;
}

[[nodiscard]] ParamResult<void> append_binding_field(
    const ParamPath& field_path, const config::ConfigView& field_value,
    const ParamSchemaDescriptor& schema, const detail::ParamConfigTypeRoutes* routes,
    const config::Path& config_path, const ParamConfigDocumentProvenance& document_source,
    const ParamConfigDecodeOptions& options, std::string_view operator_path, void* aggregate_value,
    DecodedBindingFields& decoded_fields,
    std::vector<ParamConfigFieldProvenance>& config_provenance) {
  const FieldMatch match = match_field(schema, field_path);
  if (match.ambiguous) {
    return make_unexpected(
        conversion_error(ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_FIELD_AMBIGUOUS"));
  }
  if (match.descriptor == nullptr) {
    if (options.values.unknown_fields() == config::UnknownFieldPolicy::kIgnore) {
      return {};
    }
    return make_unexpected(
        conversion_error(ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_FIELD_UNKNOWN"));
  }
  const auto duplicate = std::ranges::find(
      decoded_fields.contributions, match.descriptor->id, &DecodedBindingField::field);
  if (duplicate != decoded_fields.contributions.end()) {
    return make_unexpected(
        conversion_error(ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_FIELD_DUPLICATE"));
  }
  auto canonical_path = ParamPath::parse(match.descriptor->canonical_name);
  if (!canonical_path) {
    return make_unexpected(
        from_core_error(std::move(canonical_path.error()), "PARAM_CONFIG_FIELD_PATH_INVALID"));
  }
  auto inserted = insert(decoded_fields.normalized_patch, *canonical_path, field_value.to_value());
  if (!inserted) {
    return make_unexpected(std::move(inserted.error()));
  }
  if (match.descriptor->local_operations.replace_member_from_scalar != nullptr) {
    auto decoded = decode_scalar(field_value, match.descriptor->type_shape, options.values);
    if (!decoded) {
      return make_unexpected(std::move(decoded.error()));
    }
    if (!match.descriptor->local_operations.replace_member_from_scalar(aggregate_value,
                                                                       decoded->scalar())) {
      return make_unexpected(conversion_error(ParamToolErrorCode::kInvalidArgument,
                                              "PARAM_CONFIG_FIELD_TYPE_MISMATCH"));
    }
  } else if (const detail::ParamConfigFieldRoute* const route =
                 config_field_route(schema, *match.descriptor, routes);
             route != nullptr && route->decode != nullptr) {
    auto decoded = route->decode(aggregate_value, field_value, options.values);
    if (!decoded) {
      return make_unexpected(std::move(decoded.error()));
    }
  } else {
    if (!decoded_fields.codec_required_path) {
      decoded_fields.codec_required_path = field_path;
    }
  }
  decoded_fields.contributions.push_back(DecodedBindingField{
      .field = match.descriptor->id,
      .full_id = match.descriptor->full_id,
      .origin = match.alias ? ParamFieldRebindOrigin::kAlias : ParamFieldRebindOrigin::kCaptured,
      .config_provenance_index = static_cast<std::uint32_t>(config_provenance.size()),
  });
  auto source_path = config_source_path(config_path, field_path);
  if (!source_path) {
    return make_unexpected(std::move(source_path.error()));
  }
  config_provenance.push_back(ParamConfigFieldProvenance{
      .document = document_source,
      .source_path = std::move(*source_path),
      .source_ordinal = static_cast<std::uint32_t>(config_provenance.size()),
      .target_operator_path = std::string{operator_path},
      .target_schema_id = std::string{schema.stable_id},
      .target_schema_version = schema.version,
      .target_field_id = match.descriptor->id,
      .target_full_id = match.descriptor->full_id,
      .origin = match.alias ? ParamFieldRebindOrigin::kAlias : ParamFieldRebindOrigin::kCaptured,
  });
  return {};
}

[[nodiscard]] ParamResult<DecodedBindingFields> decode_binding_fields(
    std::span<const std::pair<ParamPath, config::ConfigView>> leaves,
    const ParamSchemaDescriptor& schema, const detail::ParamConfigTypeRoutes* routes,
    const config::Path& config_path, const ParamConfigDocumentProvenance& document_source,
    const ParamConfigDecodeOptions& options, std::string_view operator_path, void* aggregate_value,
    std::vector<ParamConfigFieldProvenance>& config_provenance) {
  DecodedBindingFields result;
  result.contributions.reserve(leaves.size());
  for (const auto& [field_path, field_value] : leaves) {
    auto appended = append_binding_field(field_path,
                                         field_value,
                                         schema,
                                         routes,
                                         config_path,
                                         document_source,
                                         options,
                                         operator_path,
                                         aggregate_value,
                                         result,
                                         config_provenance);
    if (!appended) {
      return make_unexpected(
          with_config_source_context(std::move(appended.error()), config_path, field_path));
    }
  }
  return result;
}

template <typename Scalar>
[[nodiscard]] ParamConstraintScalar signed_constraint_scalar(ParamValue value) noexcept {
  auto decoded = value.as<Scalar>();
  return decoded ? ParamConstraintScalar{.kind = ParamConstraintScalarKind::kSigned,
                                         .signed_value = static_cast<std::int64_t>(*decoded)}
                 : ParamConstraintScalar{};
}

template <typename Scalar>
[[nodiscard]] ParamConstraintScalar unsigned_constraint_scalar(ParamValue value) noexcept {
  auto decoded = value.as<Scalar>();
  return decoded ? ParamConstraintScalar{.kind = ParamConstraintScalarKind::kUnsigned,
                                         .unsigned_value = static_cast<std::uint64_t>(*decoded)}
                 : ParamConstraintScalar{};
}

[[nodiscard]] ParamConstraintScalar integer_constraint_scalar(ParamValue value,
                                                              bool is_signed) noexcept {
  switch (value.bit_width()) {
    case 8:
      return is_signed ? signed_constraint_scalar<std::int8_t>(value)
                       : unsigned_constraint_scalar<std::uint8_t>(value);
    case 16:
      return is_signed ? signed_constraint_scalar<std::int16_t>(value)
                       : unsigned_constraint_scalar<std::uint16_t>(value);
    case 32:
      return is_signed ? signed_constraint_scalar<std::int32_t>(value)
                       : unsigned_constraint_scalar<std::uint32_t>(value);
    case 64:
      return is_signed ? signed_constraint_scalar<std::int64_t>(value)
                       : unsigned_constraint_scalar<std::uint64_t>(value);
    default:
      return {};
  }
}

[[nodiscard]] ParamConstraintScalar floating_constraint_scalar(ParamValue value) noexcept {
  double decoded{};
  if (value.bit_width() == 32U) {
    auto scalar = value.as<float>();
    if (!scalar) {
      return {};
    }
    decoded = *scalar;
  } else {
    auto scalar = value.as<double>();
    if (!scalar) {
      return {};
    }
    decoded = *scalar;
  }
  return ParamConstraintScalar{.kind = ParamConstraintScalarKind::kFloating,
                               .floating_value = decoded == 0.0 ? 0.0 : decoded};
}

[[nodiscard]] ParamConstraintScalar constraint_scalar(ParamValue value) noexcept {
  switch (value.kind()) {
    case ParamScalarKind::kBool:
      return unsigned_constraint_scalar<bool>(value);
    case ParamScalarKind::kSignedInteger:
      return integer_constraint_scalar(value, true);
    case ParamScalarKind::kUnsignedInteger:
      return integer_constraint_scalar(value, false);
    case ParamScalarKind::kFloatingPoint:
      return floating_constraint_scalar(value);
    case ParamScalarKind::kDurationNanoseconds: {
      auto decoded = value.as<std::chrono::nanoseconds>();
      return decoded ? ParamConstraintScalar{.kind = ParamConstraintScalarKind::kSigned,
                                             .signed_value = decoded->count()}
                     : ParamConstraintScalar{};
    }
    case ParamScalarKind::kNone:
      return {};
  }
  return {};
}

[[nodiscard]] bool scalar_less(ParamConstraintScalar left, ParamConstraintScalar right) noexcept {
  if (left.kind != right.kind) {
    return false;
  }
  switch (left.kind) {
    case ParamConstraintScalarKind::kSigned:
      return left.signed_value < right.signed_value;
    case ParamConstraintScalarKind::kUnsigned:
      return left.unsigned_value < right.unsigned_value;
    case ParamConstraintScalarKind::kFloating:
      return left.floating_value < right.floating_value;
    case ParamConstraintScalarKind::kNone:
      return false;
  }
  return false;
}

[[nodiscard]] bool scalar_equal(ParamConstraintScalar left, ParamConstraintScalar right) noexcept {
  if (left.kind != right.kind) {
    return false;
  }
  switch (left.kind) {
    case ParamConstraintScalarKind::kSigned:
      return left.signed_value == right.signed_value;
    case ParamConstraintScalarKind::kUnsigned:
      return left.unsigned_value == right.unsigned_value;
    case ParamConstraintScalarKind::kFloating:
      return left.floating_value == right.floating_value;
    case ParamConstraintScalarKind::kNone:
      return true;
  }
  return false;
}

[[nodiscard]] bool satisfies_constraints(ParamValue value,
                                         const ParamConstraints& constraints) noexcept {
  const ParamConstraintScalar observed = constraint_scalar(value);
  if (observed.kind == ParamConstraintScalarKind::kNone) {
    return false;
  }
  if (constraints.has_range &&
      (observed.kind != constraints.range_min.kind || observed.kind != constraints.range_max.kind ||
       scalar_less(observed, constraints.range_min) ||
       scalar_less(constraints.range_max, observed))) {
    return false;
  }
  if (constraints.allowed_value_count != 0U) {
    bool found = false;
    for (std::size_t index = 0; index < constraints.allowed_value_count; ++index) {
      found = found || scalar_equal(observed, constraints.allowed_values[index]);
    }
    if (!found) {
      return false;
    }
  }
  return true;
}

}  // namespace

namespace detail {

ParamResult<std::shared_ptr<const holoscan::detail::ParameterInputEvidenceState>>
decode_params_into(const config::Document& resolved_config, const config::Path& config_path,
                   void* baseline, const ParamSchemaDescriptor& schema,
                   const ParamConfigTypeRoutes& routes, bool explicit_baseline,
                   const config::DecodeOptions& options) {
  if (baseline == nullptr || schema.local_type == nullptr || schema.fields.empty()) {
    return make_unexpected(
        conversion_error(ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_BASELINE_INVALID"));
  }
  const ParamConfigTypeRoutes* const admitted_routes = config_type_routes(schema, &routes);
  if (admitted_routes == nullptr) {
    return make_unexpected(
        conversion_error(ParamToolErrorCode::kTargetMismatch, "PARAM_CONFIG_SCHEMA_MISMATCH"));
  }
  auto value = resolved_config.try_at(config_path);
  if (!value) {
    return make_unexpected(
        from_config_error(std::move(value.error()), "PARAM_CONFIG_PATH_INVALID"));
  }
  try {
    const auto fail = [&](ParamToolError error) {
      return make_unexpected(with_config_source_context(std::move(error), config_path, {}));
    };
    if (value->kind() != config::ConfigValueKind::kObject) {
      return fail(conversion_error(ParamToolErrorCode::kInvalidArgument,
                                   "PARAM_CONFIG_BINDING_OBJECT_REQUIRED"));
    }
    std::vector<std::pair<ParamPath, config::ConfigView>> leaves;
    leaves.reserve(std::min<std::size_t>(value->size(), kMaxConfigFields));
    std::vector<std::string> path;
    auto flattened = flatten_config_values(*value, schema, path, leaves, kMaxConfigFields);
    if (!flattened) {
      return fail(std::move(flattened.error()));
    }
    if (leaves.empty()) {
      return fail(
          conversion_error(ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_BINDING_EMPTY"));
    }

    std::vector<ParamConfigFieldProvenance> provenance;
    provenance.reserve(leaves.size());
    const ParamConfigDecodeOptions decode_options{
        .values = options,
        .max_bindings = 1U,
        .max_fields = kMaxConfigFields,
    };
    auto fields = decode_binding_fields(leaves,
                                        schema,
                                        admitted_routes,
                                        config_path,
                                        document_provenance(resolved_config),
                                        decode_options,
                                        {},
                                        baseline,
                                        provenance);
    if (!fields) {
      return make_unexpected(std::move(fields.error()));
    }
    if (fields->contributions.empty()) {
      return fail(
          conversion_error(ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_NO_CONTRIBUTIONS"));
    }
    if (fields->codec_required_path) {
      return make_unexpected(
          with_config_source_context(conversion_error(ParamToolErrorCode::kNotSupported,
                                                      "PARAM_CONFIG_FIELD_CODEC_UNAVAILABLE"),
                                     config_path,
                                     *fields->codec_required_path));
    }
    const ParamValidationResult validation = schema.local_type->validate_aggregate(baseline);
    if (!validation.ok()) {
      return fail(
          conversion_error(ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_VALIDATION_FAILED"));
    }

    auto evidence = std::make_shared<holoscan::detail::ParameterInputEvidenceState>();
    evidence->local_type_token = routes.local_type_token;
    evidence->schema_id = std::string{schema.stable_id};
    evidence->schema_version = schema.version;
    evidence->structural_hash = schema.structural_hash;
    evidence->contract_hash = schema.contract_hash;
    evidence->explicit_baseline = explicit_baseline;
    evidence->config_fields = std::move(provenance);
    return std::shared_ptr<const holoscan::detail::ParameterInputEvidenceState>{
        std::move(evidence)};
  } catch (const std::bad_alloc&) {
    return make_unexpected(
        conversion_error(ParamToolErrorCode::kResourceExhausted, "PARAM_CONFIG_ALLOC_FAILED"));
  }
}

}  // namespace detail

namespace {

[[nodiscard]] ParamResult<std::vector<holoscan::detail::ParameterTransactionFieldState>>
decode_transaction_fields(const config::ConfigView& value, const ParamSchemaDescriptor& schema,
                          const detail::ParamConfigTypeRoutes& routes,
                          const ParamConfigBinding& binding,
                          const ParamConfigDocumentProvenance& document_source,
                          const ParamConfigDecodeOptions& options, std::string_view operator_path,
                          std::uint32_t remaining_fields, std::uint32_t& decoded_field_count,
                          std::vector<ParamConfigFieldProvenance>& config_provenance) {
  const auto fail = [&](ParamToolError error, const ParamPath& field_path) {
    return make_unexpected(
        with_config_binding_context(std::move(error), binding, field_path, operator_path));
  };
  if (value.kind() != config::ConfigValueKind::kObject || schema.fields.empty()) {
    return fail(conversion_error(ParamToolErrorCode::kInvalidArgument,
                                 "PARAM_CONFIG_BINDING_OBJECT_REQUIRED"),
                {});
  }
  std::vector<std::pair<ParamPath, config::ConfigView>> leaves;
  leaves.reserve(std::min<std::size_t>(value.size(), remaining_fields));
  std::vector<std::string> path;
  auto flattened = flatten_config_values(value, schema, path, leaves, remaining_fields);
  if (!flattened) {
    return fail(std::move(flattened.error()), {});
  }

  std::vector<holoscan::detail::ParameterTransactionFieldState> fields;
  fields.reserve(leaves.size());
  for (const auto& [field_path, field_value] : leaves) {
    const FieldMatch match = match_field(schema, field_path);
    if (match.ambiguous) {
      return fail(
          conversion_error(ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_FIELD_AMBIGUOUS"),
          field_path);
    }
    if (match.descriptor == nullptr) {
      if (options.values.unknown_fields() == config::UnknownFieldPolicy::kIgnore) {
        continue;
      }
      return fail(
          conversion_error(ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_FIELD_UNKNOWN"),
          field_path);
    }
    if (std::ranges::find(fields,
                          match.descriptor->id,
                          &holoscan::detail::ParameterTransactionFieldState::field) !=
        fields.end()) {
      return fail(
          conversion_error(ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_FIELD_DUPLICATE"),
          field_path);
    }

    holoscan::detail::ParameterTransactionFieldState field{
        .field = match.descriptor->id,
        .full_id = match.descriptor->full_id,
        .origin = match.alias ? ParamFieldRebindOrigin::kAlias : ParamFieldRebindOrigin::kCaptured,
        .config_provenance_index = static_cast<std::uint32_t>(config_provenance.size()),
    };
    if (match.descriptor->local_operations.replace_member_from_scalar != nullptr) {
      auto decoded = decode_scalar(field_value, match.descriptor->type_shape, options.values);
      if (!decoded) {
        return fail(std::move(decoded.error()), field_path);
      }
      field.scalar_value = decoded->scalar();
      field.staged_value_digest =
          holoscan::detail::parameter_transaction_scalar_digest(*field.scalar_value);
    } else {
      const detail::ParamConfigFieldRoute* const route =
          config_field_route(schema, *match.descriptor, &routes);
      const ParamFieldLocalOperations& operations = match.descriptor->local_operations;
      if (route == nullptr || route->decode_staged == nullptr ||
          route->staged_value_digest == nullptr || operations.local_field_type_token == nullptr ||
          operations.object_size == 0U || operations.object_alignment == 0U ||
          operations.copy_construct_from_staged == nullptr ||
          operations.destroy_staged == nullptr ||
          operations.replace_member_from_staged == nullptr) {
        return fail(conversion_error(ParamToolErrorCode::kNotSupported,
                                     "PARAM_CONFIG_FIELD_CODEC_UNAVAILABLE"),
                    field_path);
      }
      auto staged = route->decode_staged(field_value, options.values);
      if (!staged) {
        return fail(std::move(staged.error()), field_path);
      }
      ParamSchemaHash digest{};
      if (!route->staged_value_digest(staged->get(), digest)) {
        return fail(conversion_error(ParamToolErrorCode::kNotSupported,
                                     "PARAM_CONFIG_FIELD_DIGEST_UNAVAILABLE"),
                    field_path);
      }
      field.staged_type_token = operations.local_field_type_token;
      field.local_value = std::move(*staged);
      field.local_value_digest = route->staged_value_digest;
      field.staged_value_digest = digest;
    }

    auto source_path =
        config_source_path(detail::ParamConfigBindingAccess::config_path(binding), field_path);
    if (!source_path) {
      return make_unexpected(std::move(source_path.error()));
    }
    config_provenance.push_back(ParamConfigFieldProvenance{
        .document = document_source,
        .source_path = std::move(*source_path),
        .source_ordinal = static_cast<std::uint32_t>(config_provenance.size()),
        .target_operator_path = std::string{operator_path},
        .target_schema_id = std::string{schema.stable_id},
        .target_schema_version = schema.version,
        .target_field_id = match.descriptor->id,
        .target_full_id = match.descriptor->full_id,
        .origin = field.origin,
    });
    fields.push_back(std::move(field));
  }
  if (fields.empty()) {
    return fail(
        conversion_error(ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_NO_CONTRIBUTIONS"),
        {});
  }
  decoded_field_count += static_cast<std::uint32_t>(fields.size());
  return fields;
}

}  // namespace

ParamResult<ParamRebindProvenance> set_params(Graph& graph, const config::Document& resolved_config,
                                              std::span<const ParamConfigBinding> bindings,
                                              ParamConfigDecodeOptions options) {
  if (bindings.empty()) {
    return make_unexpected(
        conversion_error(ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_BINDINGS_EMPTY"));
  }
  if (options.max_bindings == 0U || options.max_fields == 0U ||
      options.max_bindings > kMaxConfigBindings || options.max_fields > kMaxConfigFields) {
    return make_unexpected(
        conversion_error(ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_OPTIONS_INVALID"));
  }
  if (bindings.size() > options.max_bindings) {
    return make_unexpected(
        conversion_error(ParamToolErrorCode::kResourceExhausted, "PARAM_CONFIG_BINDING_CAPACITY"));
  }
  for (const ParamConfigBinding& binding : bindings) {
    const auto owner = detail::ParamConfigBindingAccess::state(binding).lock();
    if (!owner || owner != holoscan::detail::ParameterConfigBindingAccess::state(graph)) {
      return make_unexpected(
          conversion_error(ParamToolErrorCode::kTargetMismatch, "PARAM_CONFIG_GRAPH_MISMATCH"));
    }
    const ParamSchemaDescriptor* const schema = detail::ParamConfigBindingAccess::schema(binding);
    if (detail::ParamConfigBindingAccess::target(binding).value == 0U || schema == nullptr ||
        schema->local_type == nullptr || schema->local_type->local_type_token == nullptr ||
        detail::ParamConfigBindingAccess::operator_path(binding).segments().empty() ||
        detail::ParamConfigBindingAccess::mode(binding) != ParamBindingMode::kPatch) {
      return make_unexpected(
          conversion_error(ParamToolErrorCode::kTargetMismatch, "PARAM_CONFIG_BINDING_INVALID"));
    }
  }

  try {
    auto transaction = std::make_shared<holoscan::detail::ParameterTransactionState>();
    transaction->source = holoscan::detail::ParameterTransactionSource::kConfig;
    transaction->operators.reserve(bindings.size());
    std::vector<ParamConfigFieldProvenance> config_provenance;
    config_provenance.reserve(std::min<std::size_t>(options.max_fields, 4096U));
    std::vector<VertexId> seen_targets;
    seen_targets.reserve(bindings.size());
    std::uint32_t decoded_fields = 0U;
    const ParamConfigDocumentProvenance document_source = document_provenance(resolved_config);
    for (const ParamConfigBinding& binding : bindings) {
      const std::uint32_t provenance_begin = static_cast<std::uint32_t>(config_provenance.size());
      const VertexId target = detail::ParamConfigBindingAccess::target(binding);
      std::string target_path =
          render_path(detail::ParamConfigBindingAccess::operator_path(binding));
      const auto fail_binding = [&](ParamToolError error) {
        return make_unexpected(
            with_config_binding_context(std::move(error), binding, {}, target_path));
      };
      if (std::ranges::find(seen_targets, target) != seen_targets.end()) {
        return fail_binding(conversion_error(ParamToolErrorCode::kInvalidArgument,
                                             "PARAM_CONFIG_OPERATOR_DUPLICATE"));
      }
      auto value = resolved_config.try_at(detail::ParamConfigBindingAccess::config_path(binding));
      if (!value) {
        return fail_binding(
            from_config_error(std::move(value.error()), "PARAM_CONFIG_PATH_INVALID"));
      }
      const ParamSchemaDescriptor& schema = *detail::ParamConfigBindingAccess::schema(binding);
      const detail::ParamConfigTypeRoutes* const type_routes = config_type_routes(schema, binding);
      if (type_routes == nullptr) {
        return fail_binding(
            conversion_error(ParamToolErrorCode::kTargetMismatch, "PARAM_CONFIG_SCHEMA_MISMATCH"));
      }
      auto fields = decode_transaction_fields(*value,
                                              schema,
                                              *type_routes,
                                              binding,
                                              document_source,
                                              options,
                                              target_path,
                                              options.max_fields - decoded_fields,
                                              decoded_fields,
                                              config_provenance);
      if (!fields) {
        return make_unexpected(std::move(fields.error()));
      }
      transaction->operators.push_back(holoscan::detail::ParameterTransactionOperatorState{
          .target = target,
          .local_type_token = schema.local_type->local_type_token,
          .schema_id = std::string{schema.stable_id},
          .schema_version = schema.version,
          .structural_hash = schema.structural_hash,
          .contract_hash = schema.contract_hash,
          .fields = std::move(*fields),
          .provenance_begin = provenance_begin,
          .provenance_count =
              static_cast<std::uint32_t>(config_provenance.size()) - provenance_begin,
      });
      seen_targets.push_back(target);
    }
    transaction->provenance.mode = ParamRebindMode::kSparseContributions;
    transaction->provenance.config_fields = std::move(config_provenance);
    ParamRebindProvenance report = transaction->provenance;
    for (auto& source : transaction->provenance.config_fields) {
      source.target_operator_path.clear();
    }
    auto attached = holoscan::detail::ParameterInputAccess::attach_transaction(
        graph, std::shared_ptr<const holoscan::detail::ParameterTransactionState>{transaction});
    if (!attached) {
      return make_unexpected(
          from_core_error(std::move(attached.error()), "PARAM_CONFIG_ATTACH_FAILED"));
    }
    return report;
  } catch (const std::bad_alloc&) {
    return make_unexpected(
        conversion_error(ParamToolErrorCode::kResourceExhausted, "PARAM_CONFIG_ALLOC_FAILED"));
  }
}

ParamResult<config::Edit> to_config_edit(const ParamSnapshot& snapshot, const config::Path& path,
                                         ParamConfigWritePolicy policy) {
  if (policy != ParamConfigWritePolicy::kMergePersistedFields ||
      snapshot.operators().size() != 1U) {
    return make_unexpected(conversion_error(ParamToolErrorCode::kInvalidSnapshot,
                                            "PARAM_CONFIG_SNAPSHOT_SCOPE_INVALID"));
  }
  try {
    ConfigTree tree;
    for (const auto& field : snapshot.operators().front().fields) {
      if (field.projection != ParamProjectionStatus::kIncluded) {
        continue;
      }
      if (!field.value) {
        return make_unexpected(conversion_error(ParamToolErrorCode::kInvalidSnapshot,
                                                "PARAM_CONFIG_INCLUDED_VALUE_MISSING"));
      }
      auto value = encode_scalar(*field.value);
      if (!value) {
        return make_unexpected(std::move(value.error()));
      }
      auto inserted = insert(tree, field.path, std::move(*value));
      if (!inserted) {
        return make_unexpected(std::move(inserted.error()));
      }
    }
    auto object = materialize(tree);
    if (!object) {
      return make_unexpected(std::move(object.error()));
    }
    return config::Edit::merge_object(path, std::move(*object));
  } catch (const std::bad_alloc&) {
    return make_unexpected(
        conversion_error(ParamToolErrorCode::kResourceExhausted, "PARAM_CONFIG_ALLOC_FAILED"));
  }
}

namespace {

[[nodiscard]] bool runtime_profile_allows_adapter(
    const std::optional<Profile>& effective_profile) noexcept {
  const Profile profile = effective_profile.value_or(Profile::kRegulated);
  return profile != Profile::kRealTime && profile != Profile::kRegulated;
}

[[nodiscard]] bool runtime_field_allows_adapter(const ParamFieldCatalogRecord& record) noexcept {
  return record.adapter_eligible && record.visibility == ParamVisibility::kPublic &&
         record.external_access == ParamExternalAccess::kTuning &&
         record.sensitivity == ParamSensitivity::kNormal &&
         record.mutability == ParamMutability::kRuntime &&
         record.value_route == ParamValueRouteDisposition::kPortableScalar &&
         record.projection == ParamProjectionStatus::kIncluded &&
         record.update_disposition == ParameterFieldUpdateDisposition::kAdmitted;
}

[[nodiscard]] ParamResult<ParamPatch> decode_runtime_patch(
    const ParameterCatalogView& catalog, const RuntimeParamConfigBinding& binding,
    const ParamPath& field_path, const config::ConfigView& value,
    std::optional<ParamGeneration> expected_generation, const config::DecodeOptions& options) {
  auto address = catalog.resolve(binding.operator_path, field_path);
  if (!address) {
    return make_unexpected(
        from_core_error(std::move(address.error()), "PARAM_CONFIG_FIELD_UNKNOWN"));
  }
  auto field = catalog.field(*address);
  if (!field) {
    return make_unexpected(from_core_error(std::move(field.error()), "PARAM_CONFIG_FIELD_UNKNOWN"));
  }
  const ParamFieldCatalogRecord& record = **field;
  const auto owner = std::ranges::find(catalog.operators(),
                                       record.address.operator_instance,
                                       &OperatorParamRecord::operator_instance);
  if (owner == catalog.operators().end() ||
      !runtime_profile_allows_adapter(owner->effective_profile)) {
    return make_unexpected(
        conversion_error(ParamToolErrorCode::kPermissionDenied, "PARAM_CONFIG_PROFILE_DENIED"));
  }
  if (!runtime_field_allows_adapter(record)) {
    return make_unexpected(conversion_error(ParamToolErrorCode::kPermissionDenied,
                                            "PARAM_CONFIG_FIELD_NOT_ADAPTER_ELIGIBLE"));
  }
  auto decoded = decode_scalar(value, record.type_shape, options);
  if (!decoded) {
    return make_unexpected(std::move(decoded.error()));
  }
  // TODO: Convert duration bounds to the field domain here, as final aggregate validation does.
  if (!satisfies_constraints(*decoded, record.constraints)) {
    return make_unexpected(
        conversion_error(ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_FIELD_CONSTRAINT"));
  }
  return ParamPatch{
      .target = *address, .value = *decoded, .expected_generation = expected_generation};
}

}  // namespace

ParamResult<std::vector<ParamPatch>> decode_runtime_patches(
    const config::Document& document, const ParameterCatalogView& catalog,
    const RuntimeParamConfigBinding& binding, RuntimeParamDecodeOptions options) {
  if (options.max_patches == 0U || binding.operator_path.segments().empty() ||
      catalog.fields().empty() ||
      (options.expected_catalog_digest &&
       *options.expected_catalog_digest != catalog.compatibility_digest())) {
    return make_unexpected(conversion_error(ParamToolErrorCode::kTargetMismatch,
                                            "PARAM_CONFIG_RUNTIME_TARGET_INVALID"));
  }
  auto values = document.try_at(binding.values_path);
  if (!values) {
    return make_unexpected(
        from_config_error(std::move(values.error()), "PARAM_CONFIG_VALUES_PATH_INVALID"));
  }
  std::optional<ParamGeneration> expected_generation;
  if (binding.expected_generation_path) {
    auto generation = document.try_at(*binding.expected_generation_path);
    if (!generation) {
      return make_unexpected(
          from_config_error(std::move(generation.error()), "PARAM_CONFIG_GENERATION_PATH_INVALID"));
    }
    auto decoded = generation->try_as_unsigned_integer();
    if (!decoded) {
      return make_unexpected(
          from_config_error(std::move(decoded.error()), "PARAM_CONFIG_GENERATION_INVALID"));
    }
    expected_generation = *decoded;
  }
  try {
    std::vector<std::pair<ParamPath, config::ConfigView>> leaves;
    leaves.reserve(std::min<std::size_t>(values->size(), options.max_patches));
    std::vector<std::string> path;
    auto flattened = flatten_config_values(*values, path, leaves, options.max_patches);
    if (!flattened) {
      return make_unexpected(std::move(flattened.error()));
    }
    if (leaves.empty()) {
      return make_unexpected(
          conversion_error(ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_PATCH_EMPTY"));
    }
    std::vector<ParamPatch> patches;
    patches.reserve(leaves.size());
    for (const auto& [field_path, value] : leaves) {
      auto patch = decode_runtime_patch(
          catalog, binding, field_path, value, expected_generation, options.values);
      if (!patch) {
        return make_unexpected(std::move(patch.error()));
      }
      patches.push_back(std::move(*patch));
    }
    return patches;
  } catch (const std::bad_alloc&) {
    return make_unexpected(
        conversion_error(ParamToolErrorCode::kResourceExhausted, "PARAM_CONFIG_ALLOC_FAILED"));
  }
}

}  // namespace holoscan::params
