/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/holoscan_params_export.h>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/config/codec.hpp>
#include <holoscan/config/document.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/parameter_catalog.hpp>
#include <holoscan/core/parameter_provenance.hpp>
#include <holoscan/core/parameter_snapshot.hpp>
#include <holoscan/core/parameter_update.hpp>
#include <holoscan/params/error.hpp>

namespace holoscan::params {

namespace detail {
class ParamConfigBindingAccess;
struct ParamSchemaCodecMarker {};

using ParamConfigValueDigest = bool (*)(const void*, ParamSchemaHash&) noexcept;
using ParamConfigFieldDecode = ParamResult<void> (*)(void*, const config::ConfigView&,
                                                     const config::DecodeOptions&);
using ParamConfigFieldDecodeStaged = ParamResult<std::shared_ptr<const void>> (*)(
    const config::ConfigView&, const config::DecodeOptions&);

struct ParamConfigFieldRoute {
  ParamConfigFieldDecode decode{};
  ParamConfigFieldDecodeStaged decode_staged{};
  ParamConfigValueDigest staged_value_digest{};
};

struct ParamConfigTypeRoutes {
  const void* local_type_token{};
  std::string_view schema_id;
  ParamSchemaVersion schema_version{};
  ParamSchemaHash structural_hash{};
  std::span<const ParamConfigFieldRoute> fields;
};

[[nodiscard]] HOLOSCAN_PARAMS_EXPORT
    ParamResult<std::shared_ptr<const holoscan::detail::ParameterInputEvidenceState>>
    decode_params_into(const config::Document& resolved_config, const config::Path& config_path,
                       void* baseline, const ParamSchemaDescriptor& schema,
                       const ParamConfigTypeRoutes& routes, bool explicit_baseline,
                       const config::DecodeOptions& options);

template <typename ParamsT>
[[nodiscard]] ParamResult<config::ConfigValue> normalize_param_schema_aliases(
    const config::ConfigView& value);
template <typename Builder>
struct ParamConfigFieldBridge;

template <typename ParamsT, auto MemberPtr>
struct ParamConfigFieldBridge<ParamFieldBuilder<ParamsT, MemberPtr>> {
  using DeclaredFieldT = typename holoscan::detail::MemberPointerTraits<decltype(MemberPtr)>::Field;
  using FieldT = std::remove_cv_t<DeclaredFieldT>;

  [[nodiscard]] static ParamResult<void> decode(void* aggregate, const config::ConfigView& value,
                                                const config::DecodeOptions& options) {
    auto decoded = config::try_decode_config<FieldT>(value, options);
    if (!decoded) {
      return make_unexpected(
          from_config_error(std::move(decoded.error()), "PARAM_CONFIG_FIELD_DECODE_FAILED"));
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      static_cast<ParamsT*>(aggregate)->*MemberPtr = std::move(*decoded);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (const std::bad_alloc&) {
      return make_unexpected(
          ParamToolError{ParamToolErrorCode::kResourceExhausted, "PARAM_CONFIG_ALLOC_FAILED"});
    } catch (...) {
      return make_unexpected(ParamToolError{ParamToolErrorCode::kInvalidArgument,
                                            "PARAM_CONFIG_FIELD_REPLACEMENT_FAILED"});
    }
#endif
    return {};
  }

  [[nodiscard]] static ParamResult<void> decode_schema_derived(
      void* aggregate, const config::ConfigView& value, const config::DecodeOptions& options) {
    auto normalized = normalize_param_schema_aliases<FieldT>(value);
    if (!normalized) {
      return make_unexpected(std::move(normalized.error()));
    }
    return decode(aggregate, normalized->view(), options);
  }
  [[nodiscard]] static ParamResult<std::shared_ptr<const void>> decode_staged(
      const config::ConfigView& value, const config::DecodeOptions& options) {
    auto decoded = config::try_decode_config<FieldT>(value, options);
    if (!decoded) {
      return make_unexpected(
          from_config_error(std::move(decoded.error()), "PARAM_CONFIG_FIELD_DECODE_FAILED"));
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      return std::static_pointer_cast<const void>(
          std::make_shared<const FieldT>(std::move(*decoded)));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (const std::bad_alloc&) {
      return make_unexpected(
          ParamToolError{ParamToolErrorCode::kResourceExhausted, "PARAM_CONFIG_ALLOC_FAILED"});
    } catch (...) {
      return make_unexpected(ParamToolError{ParamToolErrorCode::kInvalidArgument,
                                            "PARAM_CONFIG_FIELD_STAGING_FAILED"});
    }
#endif
  }

  [[nodiscard]] static ParamResult<std::shared_ptr<const void>> decode_staged_schema_derived(
      const config::ConfigView& value, const config::DecodeOptions& options) {
    auto normalized = normalize_param_schema_aliases<FieldT>(value);
    if (!normalized) {
      return make_unexpected(std::move(normalized.error()));
    }
    return decode_staged(normalized->view(), options);
  }

  [[nodiscard]] static bool digest_staged(const void* value, ParamSchemaHash& result) noexcept {
    if (value == nullptr) {
      return false;
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      auto encoded = config::try_encode_config<FieldT>(*static_cast<const FieldT*>(value));
      if (!encoded) {
        return false;
      }
      auto document = config::Document::try_from_value(std::move(*encoded));
      if (!document) {
        return false;
      }
      result.value = document->digest().semantic.sha256;
      return true;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      return false;
    }
#endif
  }

  [[nodiscard]] static constexpr ParamConfigFieldRoute route(
      const ParamFieldDescriptor& descriptor) noexcept {
    ParamConfigFieldRoute result;
    if constexpr (config::ConfigCodable<FieldT>) {
      if constexpr (std::same_as<DeclaredFieldT, FieldT>) {
        result.decode_staged = &decode_staged;
        result.staged_value_digest = &digest_staged;
        if constexpr (std::is_base_of_v<ParamSchemaCodecMarker, config::ConfigCodec<FieldT>>) {
          if (descriptor.nested) {
            result.decode_staged = &decode_staged_schema_derived;
          }
        }
      }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      if constexpr (std::is_assignable_v<DeclaredFieldT&, FieldT>) {
#else
      if constexpr (std::is_nothrow_assignable_v<DeclaredFieldT&, FieldT>) {
#endif
        result.decode = &decode;
        if constexpr (std::is_base_of_v<ParamSchemaCodecMarker, config::ConfigCodec<FieldT>>) {
          if (descriptor.nested) {
            result.decode = &decode_schema_derived;
          }
        }
      }
    }
    return result;
  }

  [[nodiscard]] static constexpr bool matches(const ParamFieldDescriptor& field) noexcept {
    return field.local_member_token == &holoscan::detail::kMemberToken<MemberPtr>;
  }

  template <bool Required>
  [[nodiscard]] static constexpr auto object_field(std::string_view name) noexcept {
    return Required ? config::object_field(name, MemberPtr)
                    : config::defaulted_object_field(name, MemberPtr);
  }
};

template <typename ParamsT, bool Required, typename... Builders, std::size_t... Indices>
[[nodiscard]] constexpr auto make_param_schema_codec_fields(
    std::type_identity<std::tuple<Builders...>>, std::index_sequence<Indices...>) noexcept {
  constexpr const auto& schema = holoscan::detail::kParamSchema<ParamsT>;
  static_assert((ParamConfigFieldBridge<Builders>::matches(schema.fields[Indices]) && ...),
                "Schema-derived object codec internal schema field order mismatch");
  return std::tuple{ParamConfigFieldBridge<Builders>::template object_field<Required>(
      schema.fields[Indices].canonical_name)...};
}

[[nodiscard]] inline config::ConfigError param_schema_codec_error(config::ConfigErrorCode code,
                                                                  std::string_view message,
                                                                  std::string_view path) noexcept {
  return config::ConfigError::make(code, message, path);
}

[[nodiscard]] inline std::string param_schema_config_path(std::string_view base,
                                                          std::span<const std::string> segments) {
  std::string result{base};
  for (const std::string& segment : segments) {
    result = config::detail::append_path_token(result, segment);
  }
  return result;
}

[[nodiscard]] inline bool param_schema_path_prefix(std::span<const std::string> prefix,
                                                   const ParamPath& path) noexcept {
  const auto segments = path.segments();
  if (prefix.size() > segments.size()) {
    return false;
  }
  for (std::size_t index = 0; index < prefix.size(); ++index) {
    if (prefix[index] != segments[index]) {
      return false;
    }
  }
  return true;
}

struct ParamSchemaConfigRoute {
  ParamPath canonical;
  std::optional<ParamPath> alias;
};

struct ParamSchemaConfigInput {
  std::vector<std::string> path;
  config::ConfigValue value;
  bool declared_prefix{};
};

struct ParamSchemaConfigMatch {
  std::size_t index{};
  bool found{};
  bool ambiguous{};
};

template <typename SchemaT>
[[nodiscard]] inline config::ConfigResult<std::vector<ParamSchemaConfigRoute>>
param_schema_config_routes(const SchemaT& schema, bool include_aliases,
                           std::string_view context_path) {
  std::vector<ParamSchemaConfigRoute> result;
  result.reserve(schema.fields.size());
  for (const ParamFieldDescriptor& field : schema.fields) {
    auto canonical = ParamPath::parse(field.canonical_name);
    if (!canonical) {
      return tl::make_unexpected(param_schema_codec_error(
          config::ConfigErrorCode::kCodecFailure, "invalid parameter schema path", context_path));
    }
    std::optional<ParamPath> alias;
    if (include_aliases && !field.alias.empty()) {
      auto parsed = ParamPath::parse(field.alias);
      if (!parsed) {
        return tl::make_unexpected(param_schema_codec_error(config::ConfigErrorCode::kCodecFailure,
                                                            "invalid parameter schema alias",
                                                            context_path));
      }
      alias = std::move(*parsed);
    }
    result.push_back(
        ParamSchemaConfigRoute{.canonical = std::move(*canonical), .alias = std::move(alias)});
  }
  return result;
}

[[nodiscard]] inline bool param_schema_path_equal(std::span<const std::string> left,
                                                  const ParamPath& right) noexcept {
  const auto segments = right.segments();
  return left.size() == segments.size() && std::ranges::equal(left, segments);
}

[[nodiscard]] inline bool param_schema_paths_ambiguous(const ParamPath& left,
                                                       const ParamPath& right) noexcept {
  const auto left_segments = left.segments();
  const auto right_segments = right.segments();
  const std::size_t common_size = std::min(left_segments.size(), right_segments.size());
  return std::ranges::equal(left_segments.first(common_size), right_segments.first(common_size));
}

[[nodiscard]] inline bool has_param_schema_path_ambiguity(
    std::span<const ParamSchemaConfigRoute> routes) noexcept {
  for (std::size_t left = 0; left < routes.size(); ++left) {
    const ParamSchemaConfigRoute& left_route = routes[left];
    const ParamPath* left_alias = nullptr;
    if (left_route.alias.has_value()) {
      left_alias = &left_route.alias.value();
      if (param_schema_paths_ambiguous(left_route.canonical, *left_alias)) {
        return true;
      }
    }
    for (std::size_t right = left + 1U; right < routes.size(); ++right) {
      const ParamSchemaConfigRoute& right_route = routes[right];
      if (param_schema_paths_ambiguous(left_route.canonical, right_route.canonical) ||
          (left_alias != nullptr &&
           param_schema_paths_ambiguous(*left_alias, right_route.canonical))) {
        return true;
      }
      if (right_route.alias.has_value()) {
        const ParamPath& right_alias = right_route.alias.value();
        if (param_schema_paths_ambiguous(left_route.canonical, right_alias) ||
            (left_alias != nullptr && param_schema_paths_ambiguous(*left_alias, right_alias))) {
          return true;
        }
      }
    }
  }
  return false;
}

[[nodiscard]] inline ParamSchemaConfigMatch match_param_schema_route(
    std::span<const std::string> path, std::span<const ParamSchemaConfigRoute> routes) noexcept {
  ParamSchemaConfigMatch result;
  for (std::size_t index = 0; index < routes.size(); ++index) {
    if (!param_schema_path_equal(path, routes[index].canonical) &&
        (!routes[index].alias || !param_schema_path_equal(path, *routes[index].alias))) {
      continue;
    }
    if (result.found) {
      result.ambiguous = true;
      return result;
    }
    result = ParamSchemaConfigMatch{.index = index, .found = true};
  }
  return result;
}

template <typename SchemaT>
[[nodiscard]] inline ParamSchemaConfigMatch match_param_schema_literal_key(
    std::string_view key, const SchemaT& schema) noexcept {
  ParamSchemaConfigMatch result;
  for (std::size_t index = 0; index < schema.fields.size(); ++index) {
    const ParamFieldDescriptor& field = schema.fields[index];
    if (key != field.canonical_name && (field.alias.empty() || key != field.alias)) {
      continue;
    }
    if (result.found) {
      result.ambiguous = true;
      return result;
    }
    result = ParamSchemaConfigMatch{.index = index, .found = true};
  }
  return result;
}

template <typename SchemaT>
[[nodiscard]] inline config::ConfigError remap_param_schema_codec_error(
    const config::ConfigError& error, const SchemaT& schema,
    std::span<const ParamSchemaConfigRoute> routes, std::string_view context_path) {
  const std::string_view error_path = error.path();
  for (std::size_t index = 0; index < routes.size(); ++index) {
    const std::string literal_path =
        config::detail::append_path_token(context_path, schema.fields[index].canonical_name);
    if (error_path != literal_path &&
        !(error_path.starts_with(literal_path) && error_path.size() > literal_path.size() &&
          error_path[literal_path.size()] == '/')) {
      continue;
    }
    std::string structured_path =
        param_schema_config_path(context_path, routes[index].canonical.segments());
    structured_path.append(error_path.substr(literal_path.size()));
    return config::ConfigError::nest(config::ConfigError::make(error.code(),
                                                               error.message(),
                                                               structured_path,
                                                               error.source(),
                                                               error.excerpt(),
                                                               error.operation_stage(),
                                                               error.destination_state()),
                                     error);
  }
  return error;
}

[[nodiscard]] inline bool has_param_schema_path_prefix(
    std::span<const std::string> path, std::span<const ParamSchemaConfigRoute> routes) noexcept {
  return std::ranges::any_of(routes, [path](const ParamSchemaConfigRoute& route) {
    return param_schema_path_prefix(path, route.canonical) ||
           (route.alias && param_schema_path_prefix(path, *route.alias));
  });
}

[[nodiscard]] inline config::ConfigResult<void> collect_param_schema_config(
    const config::ConfigView& view, std::span<const ParamSchemaConfigRoute> routes,
    std::vector<std::string>& path, std::vector<ParamSchemaConfigInput>& values) {
  const ParamSchemaConfigMatch match = match_param_schema_route(path, routes);
  const bool prefix = has_param_schema_path_prefix(path, routes);
  if (!path.empty() && (match.found || view.kind() != config::ConfigValueKind::kObject ||
                        view.size() == 0U || !prefix)) {
    values.push_back(
        ParamSchemaConfigInput{.path = path, .value = view.to_value(), .declared_prefix = prefix});
    return {};
  }
  for (std::size_t index = 0; index < view.size(); ++index) {
    auto key = view.try_key(index);
    auto child = view.try_element(index);
    if (!key) {
      return tl::make_unexpected(key.error());
    }
    if (!child) {
      return tl::make_unexpected(child.error());
    }
    path.emplace_back(*key);
    auto collected = collect_param_schema_config(*child, routes, path, values);
    path.pop_back();
    if (!collected) {
      return tl::make_unexpected(collected.error());
    }
  }
  return {};
}

[[nodiscard]] inline config::ConfigResult<void> merge_param_schema_config(
    config::ConfigValue& root, std::vector<std::vector<std::string>>& existing_paths,
    std::span<const std::string> path, config::ConfigValue value, std::string_view context_path) {
  for (const auto& existing : existing_paths) {
    const std::size_t common = std::min(existing.size(), path.size());
    if (std::ranges::equal(std::span{existing}.first(common), path.first(common)) &&
        common == std::min(existing.size(), path.size())) {
      return tl::make_unexpected(param_schema_codec_error(
          config::ConfigErrorCode::kCodecFailure, "parameter schema paths overlap", context_path));
    }
  }
  for (auto segment = path.rbegin(); segment != path.rend(); ++segment) {
    const config::ConfigObjectEntry entry{*segment, std::move(value)};
    auto object = config::ConfigValue::try_object(std::span{&entry, 1U});
    if (!object) {
      return tl::make_unexpected(object.error());
    }
    value = std::move(*object);
  }
  auto merged = config::try_merge_config_values(root, value);
  if (!merged) {
    return tl::make_unexpected(merged.error());
  }
  root = std::move(*merged);
  existing_paths.emplace_back(path.begin(), path.end());
  return {};
}

template <std::size_t Index = 0, typename ParamsT>
[[nodiscard]] ParamResult<config::ConfigValue> normalize_param_schema_nested_field(
    std::size_t field_index, const config::ConfigView& value) {
  using SchemaT = std::remove_cvref_t<decltype(holoscan::detail::kParamSchema<ParamsT>)>;
  if constexpr (Index == std::tuple_size_v<typename SchemaT::FieldBuilders>) {
    return value.to_value();
  } else {
    if (field_index == Index) {
      using Builder = std::tuple_element_t<Index, typename SchemaT::FieldBuilders>;
      using FieldT = typename ParamConfigFieldBridge<Builder>::FieldT;
      if constexpr (config::ConfigCodable<FieldT>) {
        if constexpr (std::is_base_of_v<ParamSchemaCodecMarker, config::ConfigCodec<FieldT>>) {
          if (holoscan::detail::kParamSchema<ParamsT>.fields[Index].nested) {
            return normalize_param_schema_aliases<FieldT>(value);
          }
        }
      }
      return value.to_value();
    }
    return normalize_param_schema_nested_field<Index + 1, ParamsT>(field_index, value);
  }
}

template <typename ParamsT>
[[nodiscard]] ParamResult<config::ConfigValue> normalize_param_schema_aliases(
    const config::ConfigView& value) {
  if constexpr (!ParameterAggregate<ParamsT>) {
    return value.to_value();
  } else if constexpr (param_schema_availability<ParamsT>() != ParamSchemaAvailability::kDeclared) {
    return value.to_value();
  } else {
    if (value.kind() != config::ConfigValueKind::kObject) {
      return value.to_value();
    }
    constexpr const auto& schema = holoscan::detail::kParamSchema<ParamsT>;
    auto routes = param_schema_config_routes(schema, true, std::string_view{});
    if (!routes) {
      return make_unexpected(
          from_config_error(std::move(routes.error()), "PARAM_CONFIG_FIELD_PATH_INVALID"));
    }
    if (has_param_schema_path_ambiguity(*routes)) {
      std::vector<config::ConfigObjectEntry> entries;
      entries.reserve(value.size());
      std::vector<bool> contributed(schema.fields.size());
      for (std::size_t index = 0; index < value.size(); ++index) {
        auto key = value.try_key(index);
        auto child = value.try_element(index);
        if (!key) {
          return make_unexpected(
              from_config_error(std::move(key.error()), "PARAM_CONFIG_FIELD_NORMALIZATION_FAILED"));
        }
        if (!child) {
          return make_unexpected(from_config_error(std::move(child.error()),
                                                   "PARAM_CONFIG_FIELD_NORMALIZATION_FAILED"));
        }
        const ParamSchemaConfigMatch match = match_param_schema_literal_key(*key, schema);
        if (match.ambiguous) {
          return make_unexpected(
              ParamToolError{ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_FIELD_AMBIGUOUS"});
        }
        std::string_view target = *key;
        config::ConfigValue normalized = child->to_value();
        if (match.found) {
          if (contributed[match.index]) {
            return make_unexpected(ParamToolError{ParamToolErrorCode::kInvalidArgument,
                                                  "PARAM_CONFIG_FIELD_DUPLICATE"});
          }
          contributed[match.index] = true;
          target = schema.fields[match.index].canonical_name;
          auto nested = normalize_param_schema_nested_field<0, ParamsT>(match.index, *child);
          if (!nested) {
            return make_unexpected(std::move(nested.error()));
          }
          normalized = std::move(*nested);
        }
        entries.push_back(config::ConfigObjectEntry{target, std::move(normalized)});
      }
      auto normalized = config::ConfigValue::try_object(entries);
      if (!normalized) {
        return make_unexpected(from_config_error(std::move(normalized.error()),
                                                 "PARAM_CONFIG_FIELD_NORMALIZATION_FAILED"));
      }
      return std::move(*normalized);
    }
    std::vector<ParamSchemaConfigInput> values;
    values.reserve(value.size());
    std::vector<std::string> path;
    auto collected = collect_param_schema_config(value, *routes, path, values);
    if (!collected) {
      return make_unexpected(from_config_error(std::move(collected.error()),
                                               "PARAM_CONFIG_FIELD_NORMALIZATION_FAILED"));
    }

    auto root = config::ConfigValue::try_object(std::span<const config::ConfigObjectEntry>{});
    if (!root) {
      return make_unexpected(
          from_config_error(std::move(root.error()), "PARAM_CONFIG_FIELD_NORMALIZATION_FAILED"));
    }
    std::vector<std::vector<std::string>> normalized_paths;
    std::vector<bool> contributed(schema.fields.size());
    for (const ParamSchemaConfigInput& input : values) {
      const ParamSchemaConfigMatch match = match_param_schema_route(input.path, *routes);
      if (match.ambiguous) {
        return make_unexpected(
            ParamToolError{ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_FIELD_AMBIGUOUS"});
      }
      std::span<const std::string> target = input.path;
      config::ConfigValue normalized = input.value;
      if (match.found) {
        if (contributed[match.index]) {
          return make_unexpected(
              ParamToolError{ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_FIELD_DUPLICATE"});
        }
        contributed[match.index] = true;
        target = (*routes)[match.index].canonical.segments();
        auto nested =
            normalize_param_schema_nested_field<0, ParamsT>(match.index, input.value.view());
        if (!nested) {
          return make_unexpected(std::move(nested.error()));
        }
        normalized = std::move(*nested);
      }
      auto merged = merge_param_schema_config(
          *root, normalized_paths, target, std::move(normalized), std::string_view{});
      if (!merged) {
        return make_unexpected(from_config_error(std::move(merged.error()),
                                                 "PARAM_CONFIG_FIELD_NORMALIZATION_FAILED"));
      }
    }
    return std::move(*root);
  }
}

template <typename ParamsT, bool Required>
struct ParamSchemaCodecBase : private ParamSchemaCodecMarker, public config::ObjectCodec<ParamsT> {
 private:
  using SchemaT = std::remove_cvref_t<decltype(holoscan::detail::kParamSchema<ParamsT>)>;

 public:
  static_assert(holoscan::detail::kParamSchema<ParamsT>.header.coverage() ==
                    ParamSchemaCoverage::kComplete,
                "Schema-derived object codecs require a complete parameter schema");

  inline static constexpr auto fields = make_param_schema_codec_fields<ParamsT, Required>(
      std::type_identity<typename SchemaT::FieldBuilders>{},
      std::make_index_sequence<std::tuple_size_v<typename SchemaT::FieldBuilders>>{});

  static_assert(config::detail::validate_object_fields<ParamsT>(fields),
                "Schema-derived object codecs require ConfigCodable, writable schema fields");

  [[nodiscard]] static config::ConfigResult<ParamsT> decode(config::ConfigView view,
                                                            const config::DecodeOptions& options,
                                                            config::DecodeContext& context) {
    if (view.kind() != config::ConfigValueKind::kObject) {
      return tl::make_unexpected(param_schema_codec_error(config::ConfigErrorCode::kTypeMismatch,
                                                          "configuration value is not an object",
                                                          context.path()));
    }
    constexpr const auto& schema = holoscan::detail::kParamSchema<ParamsT>;
    auto routes = param_schema_config_routes(schema, true, context.path());
    if (!routes) {
      return tl::make_unexpected(routes.error());
    }
    if (has_param_schema_path_ambiguity(*routes)) {
      return config::ObjectCodec<ParamsT>::decode(std::move(view), options, context);
    }
    for (ParamSchemaConfigRoute& route : *routes) {
      route.alias.reset();
    }
    std::vector<ParamSchemaConfigInput> values;
    values.reserve(view.size());
    std::vector<std::string> path;
    auto collected = collect_param_schema_config(view, *routes, path, values);
    if (!collected) {
      return tl::make_unexpected(collected.error());
    }
    std::vector<config::ConfigObjectEntry> entries;
    entries.reserve(values.size());
    for (const ParamSchemaConfigInput& input : values) {
      const ParamSchemaConfigMatch match = match_param_schema_route(input.path, *routes);
      if (match.ambiguous) {
        return tl::make_unexpected(param_schema_codec_error(config::ConfigErrorCode::kCodecFailure,
                                                            "ambiguous parameter schema path",
                                                            context.path()));
      }
      if (match.found) {
        entries.push_back(
            config::ConfigObjectEntry{schema.fields[match.index].canonical_name, input.value});
      } else if (input.declared_prefix && input.value.kind() != config::ConfigValueKind::kObject) {
        return tl::make_unexpected(
            param_schema_codec_error(config::ConfigErrorCode::kTypeMismatch,
                                     "configuration path traverses a non-object value",
                                     param_schema_config_path(context.path(), input.path)));
      }
    }
    auto flat = config::ConfigValue::try_object(entries);
    if (!flat) {
      return tl::make_unexpected(flat.error());
    }
    auto decoded = config::ObjectCodec<ParamsT>::decode(flat->view(), options, context);
    if (!decoded) {
      return tl::make_unexpected(
          remap_param_schema_codec_error(decoded.error(), schema, *routes, context.path()));
    }
    if (options.unknown_fields() == config::UnknownFieldPolicy::kReject) {
      for (const ParamSchemaConfigInput& input : values) {
        if (!input.declared_prefix && !match_param_schema_route(input.path, *routes).found) {
          return tl::make_unexpected(
              param_schema_codec_error(config::ConfigErrorCode::kCodecFailure,
                                       "object contains an unconsumed member",
                                       context.path()));
        }
      }
    }
    return decoded;
  }

  [[nodiscard]] static config::ConfigResult<config::ConfigValue> encode(
      const ParamsT& value, const config::EncodeOptions& options, config::EncodeContext& context) {
    constexpr const auto& schema = holoscan::detail::kParamSchema<ParamsT>;
    auto routes = param_schema_config_routes(schema, true, context.path());
    if (!routes) {
      return tl::make_unexpected(routes.error());
    }
    if (has_param_schema_path_ambiguity(*routes)) {
      return config::ObjectCodec<ParamsT>::encode(value, options, context);
    }
    auto flat = config::ObjectCodec<ParamsT>::encode(value, options, context);
    if (!flat) {
      return tl::make_unexpected(
          remap_param_schema_codec_error(flat.error(), schema, *routes, context.path()));
    }
    auto root = config::ConfigValue::try_object(std::span<const config::ConfigObjectEntry>{});
    if (!root) {
      return tl::make_unexpected(root.error());
    }
    std::vector<std::vector<std::string>> encoded_paths;
    encoded_paths.reserve(schema.fields.size());
    for (std::size_t index = 0; index < schema.fields.size(); ++index) {
      auto member = flat->view().try_member(schema.fields[index].canonical_name);
      if (!member) {
        return tl::make_unexpected(member.error());
      }
      auto merged = merge_param_schema_config(*root,
                                              encoded_paths,
                                              (*routes)[index].canonical.segments(),
                                              member->to_value(),
                                              context.path());
      if (!merged) {
        return tl::make_unexpected(merged.error());
      }
    }
    return std::move(*root);
  }
};

}  // namespace detail

/// @brief Derive a default-preserving object codec from a complete parameter schema.
///
/// Specialize `config::ConfigCodec` for `ParamsT` by inheriting this type. Every declared
/// schema field becomes a default-preserving object member in declaration order, so absent keys
/// retain their C++ default-member values. The codec derives canonical names and member pointers.
/// Decoded canonical `ParamPath` segments materialize as nested configuration objects unless any
/// two canonical or declared-alias routes decode identically or one decoded route is a strict
/// prefix of another. Either ambiguity preserves the complete schema's literal-key representation.
/// Declared aliases participate in representation selection and are recursively normalized only
/// by Parameter binding; direct
/// `config::decode_config` does not normalize aliases. This whole-value codec does not apply
/// parameter projection or redaction policies.
///
/// @note Specialize nested schema field types before the enclosing type. Sensitive and transient
/// fields are encoded whenever this whole-value codec is used, so use parameter projection APIs
/// for policy-filtered output. Parameter binding uses this codec when the schema-backed type is a
/// non-scalar field with an exact visible field codec. Like `config::ObjectCodec`, this codec is
/// not ConfigCodable when exceptions are disabled.
///
/// @warning Public schemas are author-asserted. Every member that must survive whole-value decode
/// must be declared. C++ cannot expose omitted members for this helper to verify, and an undeclared
/// member resets to its C++ default value.
///
/// @tparam ParamsT Exact aggregate with a complete `HOLOSCAN_PARAM_SCHEMA` declaration whose
/// members are ConfigCodable and move-assignable.
template <typename ParamsT>
  requires ParameterAggregate<ParamsT> &&
           (param_schema_availability<ParamsT>() == ParamSchemaAvailability::kDeclared)
struct DefaultedParamSchemaCodec : detail::ParamSchemaCodecBase<ParamsT, false> {};

/// @brief Derive an object codec that requires every field in a complete parameter schema.
///
/// Specialize `config::ConfigCodec` for `ParamsT` by inheriting this type when every schema key
/// must be present during whole-value decode. This is useful for nested parameter objects that
/// field-local binding replaces as one value. Other behavior matches
/// `DefaultedParamSchemaCodec`. Sensitive and transient schema fields are still encoded.
/// The author-asserted completeness warning on `DefaultedParamSchemaCodec` also applies here.
///
/// @tparam ParamsT Exact aggregate accepted by `DefaultedParamSchemaCodec`.
template <typename ParamsT>
  requires ParameterAggregate<ParamsT> &&
           (param_schema_availability<ParamsT>() == ParamSchemaAvailability::kDeclared)
struct RequiredParamSchemaCodec : detail::ParamSchemaCodecBase<ParamsT, true> {};

namespace detail {

template <typename ParamsT, typename Builders>
struct ParamConfigFieldRouteTable;

template <typename ParamsT, typename... Builders>
struct ParamConfigFieldRouteTable<ParamsT, std::tuple<Builders...>> {
  template <std::size_t... Indices>
  [[nodiscard]] static constexpr auto make(std::index_sequence<Indices...>) noexcept {
    constexpr const auto& schema = holoscan::detail::kParamSchema<ParamsT>;
    return std::array<ParamConfigFieldRoute, sizeof...(Builders)>{
        ParamConfigFieldBridge<Builders>::route(schema.fields[Indices])...};
  }

  [[nodiscard]] static constexpr auto make() noexcept {
    return make(std::index_sequence_for<Builders...>{});
  }
};

template <typename ParamsT>
struct ParamConfigFieldRoutes {
  using SchemaT = std::remove_cvref_t<decltype(holoscan::detail::kParamSchema<ParamsT>)>;
  using FieldDescriptors = decltype(std::declval<SchemaT>().fields);

  [[nodiscard]] static constexpr auto make_fields() noexcept {
    // The schema and this table share the builders' declaration order.
    if constexpr (requires { typename SchemaT::FieldBuilders; }) {
      return ParamConfigFieldRouteTable<ParamsT, typename SchemaT::FieldBuilders>::make();
    } else {
      return std::array<ParamConfigFieldRoute, std::tuple_size_v<FieldDescriptors>>{};
    }
  }

  static constexpr auto fields = make_fields();

  [[nodiscard]] static constexpr ParamConfigTypeRoutes make_routes() noexcept {
    constexpr auto& schema = holoscan::detail::kParamSchema<ParamsT>;
    ParamConfigTypeRoutes result{
        .local_type_token = &holoscan::detail::kParameterTypeToken<ParamsT>,
        .schema_id = schema.header.stable_id(),
        .schema_version = schema.header.version(),
        .structural_hash = schema.structural_hash,
        .fields = fields,
    };
    return result;
  }

  static constexpr ParamConfigTypeRoutes routes = make_routes();
};
}  // namespace detail

/// @brief Select how a configuration object contributes to one Parameter aggregate.
///
/// baseline supports only a sparse patch. Members absent from the bound configuration object
/// preserve the Operator's typed authoring baseline (or its type-default baseline when none was
/// authored).
enum class ParamBindingMode : std::uint8_t {
  kPatch = 0,  ///< Apply only explicitly present declared fields.
};

class ParamConfigBinding;

template <std::derived_from<OperatorBase> OperatorT>
[[nodiscard]] ParamConfigBinding bind(OperatorHandle<OperatorT> handle, config::Path path,
                                      ParamBindingMode mode = ParamBindingMode::kPatch);

template <std::derived_from<OperatorBase> OperatorT>
[[nodiscard]] ParamConfigBinding bind(OperatorHandle<OperatorT> handle, std::string_view path,
                                      ParamBindingMode mode = ParamBindingMode::kPatch);

/// @brief Own one explicit Operator-path to configuration-path conversion request.
///
/// The two path values use deliberately different grammars. \ref OperatorPath owns decoded,
/// percent-escaped Graph path segments, while \ref config::Path owns RFC 6901 tokens. Construction
/// never converts between them or infers either path from the other.
///
/// A binding returned by \ref bind carries a non-owning, call-scoped association with the
/// handle's Graph. All bindings in one attachment must name that same live Graph. The association
/// is used only while attaching and is never retained by the resulting parameter input. A handle
/// binding also carries its stable local Operator ID and static schema metadata; graph-owned
/// attachment does not snapshot the Graph or read its authored baseline.
///
/// Distinct binding objects may be read concurrently. Assignment or destruction of a binding must
/// not overlap an attachment that borrows it.
class ParamConfigBinding {
 public:
  ParamConfigBinding(const ParamConfigBinding&) = default;
  ParamConfigBinding(ParamConfigBinding&&) noexcept = default;
  ParamConfigBinding& operator=(const ParamConfigBinding&) = default;
  ParamConfigBinding& operator=(ParamConfigBinding&&) noexcept = default;

 private:
  ParamConfigBinding(OperatorPath operator_path, config::Path config_path, ParamBindingMode mode)
      : operator_path_(std::move(operator_path)),
        config_path_(std::move(config_path)),
        mode_(mode) {}

  OperatorPath operator_path_;
  config::Path config_path_;
  ParamBindingMode mode_{ParamBindingMode::kPatch};
  std::weak_ptr<holoscan::detail::GraphState> authoring_state_;
  VertexId target_{};
  const ParamSchemaDescriptor* schema_{};
  const detail::ParamConfigTypeRoutes* type_routes_{};
  friend class detail::ParamConfigBindingAccess;
  template <std::derived_from<OperatorBase> OperatorT>
  friend ParamConfigBinding bind(OperatorHandle<OperatorT>, config::Path, ParamBindingMode);
};

/// @brief Create one handle-bound document conversion request.
/// @tparam OperatorT Exact authored Operator type.
/// @param handle Live handle whose source Graph supplies target identity.
/// @param path Explicit document location of the Operator's parameter object.
/// @param mode baseline patch mode.
/// @return Owning binding with a non-owning association to the handle's Graph.
/// @throws RuntimeError if \p handle is stale or its local name cannot form a valid Operator path.
/// @note The source Graph must remain the same live Graph through the later attachment call. The
/// handle need not outlive this call. The returned binding does not extend Graph lifetime and
/// grants no authoring or Runtime mutation authority.
/// Visible field codecs from the declared Parameter schema are captured for non-scalar fields.
/// Codec specializations must be visible where this function is instantiated. The module that
/// creates the binding must remain loaded through attachment and compilation. Captured routes are
/// accepted only when their stable schema identity and field table match the defining factory;
/// otherwise attachment reports `PARAM_CONFIG_SCHEMA_MISMATCH`. A contributed non-scalar field is
/// replaced as one value, so use a required-field codec when a partial nested object must fail
/// instead of applying C++ defaults.
/// In a build without exceptions, captured assignment and codec operations must not throw. No
/// callback or data address enters public provenance or plan identity.
template <std::derived_from<OperatorBase> OperatorT>
[[nodiscard]] ParamConfigBinding bind(OperatorHandle<OperatorT> handle, config::Path path,
                                      ParamBindingMode mode) {
  auto target = holoscan::detail::ParameterConfigBindingAccess::target(handle);
  if (!target) {
    holoscan::detail::throw_graph_error(std::move(target.error()));
  }
  std::array<std::string, 1> segments{std::string{handle.name()}};
  auto operator_path = OperatorPath::from_segments(segments);
  if (!operator_path) {
    holoscan::detail::throw_graph_error(std::move(operator_path.error()));
  }
  ParamConfigBinding result{std::move(*operator_path), std::move(path), mode};
  result.authoring_state_ = holoscan::detail::ParameterConfigBindingAccess::state(handle);
  result.target_ = target->vertex;
  using ParamsT = typename OperatorT::Params;
  if constexpr (param_schema_availability<ParamsT>() == ParamSchemaAvailability::kDeclared) {
    result.schema_ = target->schema;
    const ParamSchemaDescriptor* const caller_schema = param_schema_descriptor<ParamsT>();
    const auto& routes = detail::ParamConfigFieldRoutes<ParamsT>::routes;
    if (caller_schema != nullptr && target->schema != nullptr &&
        caller_schema->local_type != nullptr &&
        caller_schema->local_type->local_type_token == routes.local_type_token &&
        caller_schema->stable_id == target->schema->stable_id &&
        caller_schema->version == target->schema->version &&
        caller_schema->structural_hash == target->schema->structural_hash &&
        caller_schema->contract_hash == target->schema->contract_hash &&
        caller_schema->fields.size() == target->schema->fields.size()) {
      result.type_routes_ = &routes;
    }
  }
  return result;
}

/// @brief Create one handle-bound request from an RFC 6901 spelling.
/// @tparam OperatorT Exact authored Operator type.
/// @param handle Live handle whose source Graph supplies target identity and typed codec evidence.
/// @param path RFC 6901 path copied and validated before the binding is returned.
/// @param mode baseline patch mode.
/// @return Owning binding equivalent to the typed \ref config::Path overload.
/// @throws RuntimeError if \p path is invalid or the handle binding fails.
/// @note This convenience does not guess an Operator path or retain \p path.
template <std::derived_from<OperatorBase> OperatorT>
[[nodiscard]] ParamConfigBinding bind(OperatorHandle<OperatorT> handle, std::string_view path,
                                      ParamBindingMode mode) {
  auto parsed = config::Path::try_parse(path);
  if (!parsed) {
    holoscan::detail::throw_graph_error(
        Error{ErrorCode::kInvalidArgument, "PARAM_CONFIG_PATH_INVALID"});
  }
  return bind(handle, std::move(*parsed), mode);
}

/// @brief Configure strict document-to-override decoding.
struct ParamConfigDecodeOptions {
  /// Numeric and unknown-value conversion policy. Missing fields always preserve the baseline.
  config::DecodeOptions values{};
  /// Maximum number of bindings accepted by one all-or-nothing conversion.
  std::uint32_t max_bindings{1024U};
  /// Maximum total number of explicit field contributions accepted across all bindings.
  std::uint32_t max_fields{65536U};
};

/// @brief Try to decode one schema-backed parameter object without reading a Graph or Operator.
template <ParamSchemaDeclared ParamsT>
[[nodiscard]] ParamResult<PreparedParams<ParamsT>> try_decode(
    const config::Document& resolved_config, const config::Path& config_path, ParamsT baseline,
    config::DecodeOptions options = config::DecodeOptions{}) {
  auto evidence = detail::decode_params_into(resolved_config,
                                             config_path,
                                             &baseline,
                                             *param_schema_descriptor<ParamsT>(),
                                             detail::ParamConfigFieldRoutes<ParamsT>::routes,
                                             true,
                                             options);
  if (!evidence) {
    return make_unexpected(std::move(evidence.error()));
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    return holoscan::detail::PreparedParamsAccess::make(std::move(baseline), std::move(*evidence));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    return make_unexpected(
        ParamToolError{ParamToolErrorCode::kResourceExhausted, "PARAM_CONFIG_ALLOC_FAILED"});
  } catch (...) {
    return make_unexpected(
        ParamToolError{ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_VALUE_MOVE_FAILED"});
  }
#endif
}

template <ParamSchemaDeclared ParamsT>
[[nodiscard]] ParamResult<PreparedParams<ParamsT>> try_decode(
    const config::Document& resolved_config, const config::Path& config_path,
    config::DecodeOptions options = config::DecodeOptions{}) {
  ParamsT baseline{};
  auto evidence = detail::decode_params_into(resolved_config,
                                             config_path,
                                             &baseline,
                                             *param_schema_descriptor<ParamsT>(),
                                             detail::ParamConfigFieldRoutes<ParamsT>::routes,
                                             false,
                                             options);
  if (!evidence) {
    return make_unexpected(std::move(evidence.error()));
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    return holoscan::detail::PreparedParamsAccess::make(std::move(baseline), std::move(*evidence));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    return make_unexpected(
        ParamToolError{ParamToolErrorCode::kResourceExhausted, "PARAM_CONFIG_ALLOC_FAILED"});
  } catch (...) {
    return make_unexpected(
        ParamToolError{ParamToolErrorCode::kInvalidArgument, "PARAM_CONFIG_VALUE_MOVE_FAILED"});
  }
#endif
}

template <ParamSchemaDeclared ParamsT>
[[nodiscard]] ParamResult<PreparedParams<ParamsT>> try_decode(
    const config::Document& resolved_config, std::string_view config_path, ParamsT baseline,
    config::DecodeOptions options = config::DecodeOptions{}) {
  auto parsed = config::Path::try_parse(config_path);
  if (!parsed) {
    return make_unexpected(
        from_config_error(std::move(parsed.error()), "PARAM_CONFIG_PATH_INVALID"));
  }
  return try_decode<ParamsT>(resolved_config, *parsed, std::move(baseline), std::move(options));
}

template <ParamSchemaDeclared ParamsT>
[[nodiscard]] ParamResult<PreparedParams<ParamsT>> try_decode(
    const config::Document& resolved_config, std::string_view config_path,
    config::DecodeOptions options = config::DecodeOptions{}) {
  auto parsed = config::Path::try_parse(config_path);
  if (!parsed) {
    return make_unexpected(
        from_config_error(std::move(parsed.error()), "PARAM_CONFIG_PATH_INVALID"));
  }
  return try_decode<ParamsT>(resolved_config, *parsed, std::move(options));
}

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
/// @brief Decode one schema-backed parameter object or throw on failure.
/// @tparam ParamsT Schema-backed parameter type to decode.
/// @param resolved_config Configuration value to decode from.
/// @param config_path RFC 6901 path to the parameter value.
/// @param options Value conversion policy.
/// @return Prepared parameter value.
/// @throws ParamException if decoding fails.
template <ParamSchemaDeclared ParamsT>
[[nodiscard]] PreparedParams<ParamsT> decode(
    const config::Document& resolved_config, std::string_view config_path,
    config::DecodeOptions options = config::DecodeOptions{}) {
  auto decoded = try_decode<ParamsT>(resolved_config, config_path, std::move(options));
  if (!decoded) {
    throw ParamException(std::move(decoded.error()));
  }
  return std::move(*decoded);
}
#endif

/// @brief Attach one matching handle-bound configuration transaction to a Graph.
/// @return Owning conversion provenance, or an error without a partial Graph mutation.
/// @note Paths in the returned report are relative to the graph where this call occurs. The
/// compiler records final canonical paths after graph composition.
/// @note Every supplied non-scalar field requires an exact field codec. A whole-parameter codec
/// alone cannot stage a sparse transaction without reading an authored baseline.
/// @note This call mutates Graph authoring state and must not run concurrently with Graph mutation.
/// Configuration and snapshot attachment share one transaction slot, including across embedded
/// Graphs. A second transaction fails instead of replacing the first.
[[nodiscard]] HOLOSCAN_PARAMS_EXPORT ParamResult<ParamRebindProvenance> set_params(
    Graph& graph, const config::Document& resolved_config,
    std::span<const ParamConfigBinding> bindings, ParamConfigDecodeOptions options = {});

[[nodiscard]] inline ParamResult<ParamRebindProvenance> set_params(
    Graph& graph, const config::Document& resolved_config,
    std::initializer_list<ParamConfigBinding> bindings, ParamConfigDecodeOptions options = {}) {
  return set_params(graph,
                    resolved_config,
                    std::span<const ParamConfigBinding>{bindings.begin(), bindings.size()},
                    options);
}

/// @brief Select the safe persisted-field merge projection.
enum class ParamConfigWritePolicy : std::uint8_t {
  /// Merge only captured fields whose ordinary projection is \c kIncluded.
  kMergePersistedFields = 0,
};

/// @brief Encode captured included fields as one merge-object document edit.
/// @param snapshot Owning single-Operator capture or slice. It grants no live control authority.
/// @param path RFC 6901 destination at which the field object is merged.
/// @param policy Ordinary persisted/public projection policy.
/// @return One owning immutable edit, or a typed error without a partial projection.
/// @note The function never reads a live run, widens snapshot coverage, emits redacted/transient/
/// internal fields, or treats the application document as a snapshot archive. Only baseline fixed
/// scalar values can be projected.
[[nodiscard]] HOLOSCAN_PARAMS_EXPORT ParamResult<config::Edit> to_config_edit(
    const ParamSnapshot& snapshot, const config::Path& path,
    ParamConfigWritePolicy policy = ParamConfigWritePolicy::kMergePersistedFields);

/// @brief Bind one runtime update object to one restricted catalog Operator.
struct RuntimeParamConfigBinding {
  OperatorPath operator_path;  ///< Exact owning Operator path authorized by the catalog.
  config::Path values_path;    ///< RFC 6901 path to one object of field contributions.
  std::optional<config::Path>
      expected_generation_path;  ///< Optional RFC 6901 path to one unsigned generation.
};

/// @brief Configure one document-to-runtime-patch transaction.
struct RuntimeParamDecodeOptions {
  /// Numeric conversion policy. Missing/unknown fields always reject.
  config::DecodeOptions values{};
  /// Maximum number of copied patches returned by one decode.
  std::uint32_t max_patches{1024U};
  /// Optional exact catalog-CAS precondition.
  std::optional<ParamCatalogDigest> expected_catalog_digest;
};

/// @brief Decode one owned same-Operator patch batch through a restricted catalog.
/// @param document Immutable untrusted command document.
/// @param catalog Active policy-restricted catalog supplied by a Core control capability.
/// @param binding Exact Operator and document paths.
/// @param options Immutable conversion limits, numeric policy, and optional catalog precondition.
/// @return Owning fixed-scalar patches for one Operator, or a typed error and no patches.
/// @note Decoding never submits a patch or treats document text, labels, paths, or digests as actor
/// authority. Core repeats run/catalog/type/constraint/reservation validation on submission.
[[nodiscard]] HOLOSCAN_PARAMS_EXPORT ParamResult<std::vector<ParamPatch>> decode_runtime_patches(
    const config::Document& document, const ParameterCatalogView& catalog,
    const RuntimeParamConfigBinding& binding, RuntimeParamDecodeOptions options = {});

}  // namespace holoscan::params
