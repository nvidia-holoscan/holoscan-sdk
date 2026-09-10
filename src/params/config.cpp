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
#include "core/parameter_override_set_internal.hpp"
#include "core/parameter_rebind_support.hpp"

namespace holoscan::params {

namespace detail {

class ParamConfigBindingAccess {
 public:
  [[nodiscard]] static Graph* graph(const ParamConfigBinding& binding) noexcept {
    return binding.authoring_graph_;
  }

  [[nodiscard]] static const ParamConfigTypeRoutes* type_routes(
      const ParamConfigBinding& binding) noexcept {
    return binding.type_routes_;
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
  auto decoded = config::decode_config<Scalar>(value, options);
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
  const auto encode = [](auto decoded) { return config::ConfigValue::signed_integer(decoded); };
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
  const auto encode = [](auto decoded) { return config::ConfigValue::unsigned_integer(decoded); };
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
  return translate_encoded_scalar(config::ConfigValue::floating(decoded));
}

[[nodiscard]] ParamResult<config::ConfigValue> encode_scalar(ParamValue value) {
  switch (value.kind()) {
    case ParamScalarKind::kBool:
      return encode_typed_scalar<bool>(
          value, [](bool decoded) { return config::ConfigValue::boolean(decoded); });
    case ParamScalarKind::kSignedInteger:
      return encode_signed_scalar(value);
    case ParamScalarKind::kUnsignedInteger:
      return encode_unsigned_scalar(value);
    case ParamScalarKind::kFloatingPoint:
      return encode_floating_scalar(value);
    case ParamScalarKind::kDurationNanoseconds:
      return encode_typed_scalar<std::chrono::nanoseconds>(
          value, [](auto decoded) { return config::ConfigValue::signed_integer(decoded.count()); });
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
  auto object = config::ConfigValue::object(entries);
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
  return result;
}

[[nodiscard]] std::optional<std::string> operator_path(
    const holoscan::detail::GraphSnapshotData& snapshot, VertexId owner) {
  if (owner.value == 0U || owner.value > snapshot.vertices.size()) {
    return std::nullopt;
  }
  const VertexRecord& vertex = snapshot.vertices[owner.value - 1U];
  if (vertex.id != owner || vertex.kind != VertexKind::kOperator || vertex.graph.value == 0U ||
      vertex.graph.value > snapshot.graphs.size()) {
    return std::nullopt;
  }
  const GraphRecord& graph = snapshot.graphs[vertex.graph.value - 1U];
  if (graph.id != vertex.graph) {
    return std::nullopt;
  }
  std::string result;
  const auto append = [&](std::string_view segment) {
    if (!result.empty()) {
      result.push_back('/');
    }
    holoscan::detail::append_percent_encoded_uri_component(result, segment);
  };
  const auto& graph_path = graph.path;
  if (!graph_path) {
    append(vertex.name);
    return result;
  }
  for (const auto& segment : graph_path->segments()) {
    append(segment);
  }
  append(vertex.name);
  return result;
}

struct AggregateStorage {
  const ParamTypeDescriptor* type{};
  void* value{};

  AggregateStorage(const ParamTypeDescriptor* type_arg, void* value_arg) noexcept
      : type(type_arg), value(value_arg) {}
  AggregateStorage(const AggregateStorage&) = delete;
  AggregateStorage& operator=(const AggregateStorage&) = delete;
  AggregateStorage(AggregateStorage&& other) noexcept
      : type(std::exchange(other.type, nullptr)), value(std::exchange(other.value, nullptr)) {}
  AggregateStorage& operator=(AggregateStorage&&) = delete;
  ~AggregateStorage() {
    if (value != nullptr) {
      type->destroy(value);
      ::operator delete(value, std::align_val_t{type->object_alignment});
    }
  }

  [[nodiscard]] std::shared_ptr<const void> release_shared() {
    const ParamTypeDescriptor* const owned_type = std::exchange(type, nullptr);
    void* const owned_value = std::exchange(value, nullptr);
    return std::shared_ptr<const void>{
        owned_value, [owned_type](const void* pointer) noexcept {
          void* mutable_pointer = const_cast<void*>(pointer);
          owned_type->destroy(mutable_pointer);
          ::operator delete(mutable_pointer, std::align_val_t{owned_type->object_alignment});
        }};
  }
};

[[nodiscard]] ParamResult<AggregateStorage> copy_authoring_baseline(
    const holoscan::detail::OperatorFactoryBase& factory) {
  const ParamTypeDescriptor& type = factory.parameter_type();
  void* storage = nullptr;
  try {
    storage = ::operator new(type.object_size, std::align_val_t{type.object_alignment});
    type.copy_construct(storage, factory.initial_parameter_value());
    return AggregateStorage{&type, storage};
  } catch (const std::bad_alloc&) {
    if (storage != nullptr) {
      ::operator delete(storage, std::align_val_t{type.object_alignment});
    }
    return make_unexpected(
        conversion_error(ParamToolErrorCode::kResourceExhausted, "PARAM_CONFIG_ALLOC_FAILED"));
  } catch (...) {
    if (storage != nullptr) {
      ::operator delete(storage, std::align_val_t{type.object_alignment});
    }
    return make_unexpected(conversion_error(ParamToolErrorCode::kInvalidArgument,
                                            "PARAM_CONFIG_BASELINE_COPY_FAILED"));
  }
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
    auto key = view.key(index);
    auto child = view.element(index);
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
    auto key = view.key(index);
    auto child = view.element(index);
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

struct DecodedBindingFields {
  std::vector<holoscan::detail::ParamOverrideFieldOriginRecord> contributions;
  ConfigTree normalized_patch;
  bool codec_required{};
};

[[nodiscard]] const detail::ParamConfigTypeRoutes* config_type_routes(
    const ParamSchemaDescriptor& schema, const ParamConfigBinding& binding) noexcept {
  const detail::ParamConfigTypeRoutes* const routes =
      detail::ParamConfigBindingAccess::type_routes(binding);
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
    const ParamSchemaDescriptor& schema, const ParamConfigBinding& binding,
    const ParamConfigDocumentProvenance& document_source, const ParamConfigDecodeOptions& options,
    std::string_view operator_path, void* aggregate_value, DecodedBindingFields& decoded_fields,
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
  const detail::ParamConfigTypeRoutes* const routes = config_type_routes(schema, binding);
  const auto duplicate =
      std::ranges::find(decoded_fields.contributions,
                        match.descriptor->id,
                        &holoscan::detail::ParamOverrideFieldOriginRecord::field);
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
  const bool aggregate_codec_available = routes != nullptr && routes->decode_overlay != nullptr;
  const detail::ParamConfigFieldRoute* const route =
      aggregate_codec_available ? nullptr : config_field_route(schema, *match.descriptor, routes);
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
  } else if (aggregate_codec_available || route == nullptr || route->decode == nullptr) {
    // Preserve the existing whole-aggregate overlay whenever its codec is available. Mixing it
    // with field routes would decode the same normalized contribution twice. Without either
    // decoder, defer the complete aggregate to the caller's codec path.
    decoded_fields.codec_required = true;
  } else {
    auto decoded = route->decode(aggregate_value, field_value, options.values);
    if (!decoded) {
      return make_unexpected(std::move(decoded.error()));
    }
  }
  decoded_fields.contributions.push_back(holoscan::detail::ParamOverrideFieldOriginRecord{
      .field = match.descriptor->id,
      .full_id = match.descriptor->full_id,
      .origin = match.alias ? holoscan::detail::ParamOverrideContributionOrigin::kAlias
                            : holoscan::detail::ParamOverrideContributionOrigin::kCaptured,
      .config_provenance_index = static_cast<std::uint32_t>(config_provenance.size()),
  });
  auto source_path = config_source_path(binding.config_path, field_path);
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
    const ParamSchemaDescriptor& schema, const ParamConfigBinding& binding,
    const ParamConfigDocumentProvenance& document_source, const ParamConfigDecodeOptions& options,
    std::string_view operator_path, void* aggregate_value,
    std::vector<ParamConfigFieldProvenance>& config_provenance) {
  DecodedBindingFields result;
  result.contributions.reserve(leaves.size());
  for (const auto& [field_path, field_value] : leaves) {
    auto appended = append_binding_field(field_path,
                                         field_value,
                                         schema,
                                         binding,
                                         document_source,
                                         options,
                                         operator_path,
                                         aggregate_value,
                                         result,
                                         config_provenance);
    if (!appended) {
      return make_unexpected(std::move(appended.error()));
    }
  }
  return result;
}

[[nodiscard]] ParamResult<std::shared_ptr<const void>> resolve_binding_value(
    DecodedBindingFields& fields, AggregateStorage& aggregate,
    detail::ParamConfigDecodeOverlay decode_overlay, const ParamConfigDecodeOptions& options) {
  if (!fields.codec_required) {
    return aggregate.release_shared();
  }
  if (decode_overlay == nullptr) {
    return make_unexpected(conversion_error(ParamToolErrorCode::kNotSupported,
                                            "PARAM_CONFIG_FIELD_CODEC_UNAVAILABLE"));
  }
  auto patch = materialize(fields.normalized_patch);
  if (!patch) {
    return make_unexpected(std::move(patch.error()));
  }
  auto decoded = decode_overlay(patch->view(), aggregate.value, options.values);
  if (!decoded) {
    return make_unexpected(std::move(decoded.error()));
  }
  return std::move(*decoded);
}

[[nodiscard]] ParamResult<holoscan::detail::ParamOverrideOperatorState> decode_binding(
    const config::ConfigView& value, std::string operator_path,
    const holoscan::detail::OperatorFactoryBase& factory, const ParamConfigBinding& binding,
    const ParamConfigDocumentProvenance& document_source, const ParamConfigDecodeOptions& options,
    std::uint32_t remaining_fields, std::uint32_t& decoded_fields,
    std::vector<ParamConfigFieldProvenance>& config_provenance) {
  if (value.kind() != config::ConfigValueKind::kObject) {
    return make_unexpected(conversion_error(ParamToolErrorCode::kInvalidArgument,
                                            "PARAM_CONFIG_BINDING_OBJECT_REQUIRED"));
  }
  const ParamSchemaDescriptor* const schema = factory.parameter_schema();
  if (schema == nullptr || schema->fields.empty()) {
    return make_unexpected(
        conversion_error(ParamToolErrorCode::kNotSupported, "PARAM_CONFIG_SCHEMA_UNAVAILABLE"));
  }
  auto aggregate = copy_authoring_baseline(factory);
  if (!aggregate) {
    return make_unexpected(std::move(aggregate.error()));
  }
  const auto canonical_baseline = holoscan::detail::parameter_value_digest(
      factory.parameter_type(), schema, aggregate->value, factory.has_no_params());
  ParamSchemaHash baseline_digest{};
  const detail::ParamConfigTypeRoutes* const type_routes = config_type_routes(*schema, binding);
  const detail::ParamConfigValueDigest value_digest =
      type_routes == nullptr ? nullptr : type_routes->value_digest;
  bool baseline_digest_available = false;
  if (canonical_baseline.available()) {
    baseline_digest = canonical_baseline.digest;
    baseline_digest_available = true;
  } else if (value_digest != nullptr) {
    baseline_digest_available = value_digest(aggregate->value, baseline_digest);
  }
  // Decode first so an explicitly supplied unsupported field reports the field problem instead
  // of the less useful consequence that a complete value digest cannot be produced.

  std::vector<std::pair<ParamPath, config::ConfigView>> leaves;
  leaves.reserve(std::min<std::size_t>(value.size(), remaining_fields));
  std::vector<std::string> path;
  auto flattened = flatten_config_values(value, *schema, path, leaves, remaining_fields);
  if (!flattened) {
    return make_unexpected(std::move(flattened.error()));
  }
  if (leaves.empty()) {
    return make_unexpected(
        conversion_error(ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_BINDING_EMPTY"));
  }

  auto fields = decode_binding_fields(leaves,
                                      *schema,
                                      binding,
                                      document_source,
                                      options,
                                      operator_path,
                                      aggregate->value,
                                      config_provenance);
  if (!fields) {
    return make_unexpected(std::move(fields.error()));
  }
  if (fields->contributions.empty()) {
    return make_unexpected(
        conversion_error(ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_NO_CONTRIBUTIONS"));
  }
  if (fields->contributions.size() > remaining_fields) {
    return make_unexpected(
        conversion_error(ParamToolErrorCode::kResourceExhausted, "PARAM_CONFIG_FIELD_CAPACITY"));
  }
  decoded_fields += static_cast<std::uint32_t>(fields->contributions.size());

  auto resolved_value = resolve_binding_value(
      *fields, *aggregate, type_routes == nullptr ? nullptr : type_routes->decode_overlay, options);
  if (!resolved_value) {
    return make_unexpected(std::move(resolved_value.error()));
  }
  if (!baseline_digest_available) {
    return make_unexpected(conversion_error(ParamToolErrorCode::kNotSupported,
                                            "PARAM_CONFIG_VALUE_DIGEST_UNAVAILABLE"));
  }

  const ParamValidationResult validation =
      factory.parameter_type().validate_aggregate(resolved_value->get());
  if (!validation.ok()) {
    return make_unexpected(
        conversion_error(ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_VALIDATION_FAILED"));
  }
  const auto canonical_resolved = holoscan::detail::parameter_value_digest(
      factory.parameter_type(), schema, resolved_value->get(), factory.has_no_params());
  ParamSchemaHash resolved_digest{};
  if (canonical_resolved.available()) {
    resolved_digest = canonical_resolved.digest;
  } else if (value_digest == nullptr || !value_digest(resolved_value->get(), resolved_digest)) {
    return make_unexpected(conversion_error(ParamToolErrorCode::kNotSupported,
                                            "PARAM_CONFIG_VALUE_DIGEST_UNAVAILABLE"));
  }
  return holoscan::detail::ParamOverrideOperatorState{
      .operator_path = std::move(operator_path),
      .schema_id = std::string{schema->stable_id},
      .schema_version = schema->version,
      .structural_hash = schema->structural_hash,
      .contract_hash = schema->contract_hash,
      .local_type = &factory.parameter_type(),
      .local_value_digest = value_digest,
      .field_replacements_only = !fields->codec_required,
      .resolved_value = std::move(*resolved_value),
      .baseline_value_digest = baseline_digest,
      .resolved_value_digest = resolved_digest,
      .contributions = std::move(fields->contributions),
  };
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

ParamResult<ParamOverrideSet> decode_overrides(const config::Document& document,
                                               std::span<const ParamConfigBinding> bindings,
                                               ParamConfigDecodeOptions options) {
  if (bindings.empty()) {
    return make_unexpected(
        conversion_error(ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_BINDINGS_EMPTY"));
  }
  Graph* const graph = detail::ParamConfigBindingAccess::graph(bindings.front());
  if (graph == nullptr) {
    return make_unexpected(
        conversion_error(ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_GRAPH_REQUIRED"));
  }
  for (const ParamConfigBinding& binding : bindings) {
    if (detail::ParamConfigBindingAccess::graph(binding) != graph) {
      return make_unexpected(
          conversion_error(ParamToolErrorCode::kTargetMismatch, "PARAM_CONFIG_GRAPH_MISMATCH"));
    }
  }
  return decode_overrides(document, *graph, bindings, options);
}

ParamResult<ParamOverrideSet> decode_overrides(const config::Document& document, Graph& graph,
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
    Graph* const owner = detail::ParamConfigBindingAccess::graph(binding);
    if ((owner != nullptr && owner != &graph) || binding.operator_path.segments().empty() ||
        binding.mode != ParamBindingMode::kPatch) {
      return make_unexpected(
          conversion_error(ParamToolErrorCode::kTargetMismatch, "PARAM_CONFIG_BINDING_INVALID"));
    }
  }

  auto captured = holoscan::detail::capture_parameter_rebind_graph(graph);
  if (!captured) {
    return make_unexpected(
        from_core_error(std::move(captured.error()), "PARAM_CONFIG_TARGET_CAPTURE_FAILED"));
  }
  try {
    std::map<std::string, std::shared_ptr<holoscan::detail::OperatorFactoryBase>, std::less<>>
        targets;
    for (auto& item : captured->factories.operators) {
      auto path = operator_path(*captured->snapshot, item.vertex);
      if (!path || !item.factory || !targets.emplace(*path, std::move(item.factory)).second) {
        return make_unexpected(
            conversion_error(ParamToolErrorCode::kTargetMismatch, "PARAM_CONFIG_TARGET_INVALID"));
      }
    }

    std::vector<holoscan::detail::ParamOverrideOperatorState> operators;
    operators.reserve(bindings.size());
    std::vector<ParamConfigFieldProvenance> config_provenance;
    config_provenance.reserve(std::min<std::size_t>(options.max_fields, 4096U));
    std::vector<std::string> seen_targets;
    seen_targets.reserve(bindings.size());
    std::uint32_t decoded_fields = 0U;
    const ParamConfigDocumentProvenance document_source = document_provenance(document);
    for (const ParamConfigBinding& binding : bindings) {
      std::string target_path = render_path(binding.operator_path);
      if (std::ranges::find(seen_targets, target_path) != seen_targets.end()) {
        return make_unexpected(conversion_error(ParamToolErrorCode::kInvalidArgument,
                                                "PARAM_CONFIG_OPERATOR_DUPLICATE"));
      }
      const auto target = targets.find(target_path);
      if (target == targets.end()) {
        return make_unexpected(
            conversion_error(ParamToolErrorCode::kTargetMismatch, "PARAM_CONFIG_OPERATOR_UNKNOWN"));
      }
      auto value = document.at(binding.config_path);
      if (!value) {
        return make_unexpected(
            from_config_error(std::move(value.error()), "PARAM_CONFIG_PATH_INVALID"));
      }
      auto decoded = decode_binding(*value,
                                    target_path,
                                    *target->second,
                                    binding,
                                    document_source,
                                    options,
                                    options.max_fields - decoded_fields,
                                    decoded_fields,
                                    config_provenance);
      if (!decoded) {
        return make_unexpected(std::move(decoded.error()));
      }
      operators.push_back(std::move(*decoded));
      seen_targets.push_back(std::move(target_path));
    }
    ParamRebindProvenance provenance;
    provenance.mode = ParamRebindMode::kSparseContributions;
    provenance.target_profile = Profile::kStandard;
    provenance.config_fields = std::move(config_provenance);
    return holoscan::detail::ParamOverrideSetAccess::make_complete(
        captured->snapshot->identity,
        std::move(operators),
        {},
        std::move(provenance),
        ParamOverrideContextBinding::kCompile);
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
  auto values = document.at(binding.values_path);
  if (!values) {
    return make_unexpected(
        from_config_error(std::move(values.error()), "PARAM_CONFIG_VALUES_PATH_INVALID"));
  }
  std::optional<ParamGeneration> expected_generation;
  if (binding.expected_generation_path) {
    auto generation = document.at(*binding.expected_generation_path);
    if (!generation) {
      return make_unexpected(
          from_config_error(std::move(generation.error()), "PARAM_CONFIG_GENERATION_PATH_INVALID"));
    }
    auto decoded = generation->as_unsigned_integer();
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
