/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/holoscan_params_export.h>

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
#include <holoscan/core/graph.hpp>
#include <holoscan/core/parameter_catalog.hpp>
#include <holoscan/core/parameter_override_set.hpp>
#include <holoscan/core/parameter_snapshot.hpp>
#include <holoscan/core/parameter_update.hpp>
#include <holoscan/params/error.hpp>

namespace holoscan::params {

namespace detail {
class ParamConfigBindingAccess;

using ParamConfigDecodeOverlay = ParamResult<std::shared_ptr<const void>> (*)(
    const config::ConfigView&, const void*, const config::DecodeOptions&);
using ParamConfigValueDigest = bool (*)(const void*, ParamSchemaHash&) noexcept;
using ParamConfigFieldDecode = ParamResult<void> (*)(void*, const config::ConfigView&,
                                                     const config::DecodeOptions&);
using ParamConfigFieldEncode = bool (*)(const void*, config::ConfigValue&) noexcept;

struct ParamConfigFieldRoute {
  ParamConfigFieldDecode decode{};
  ParamConfigFieldEncode encode{};
};

struct ParamConfigTypeRoutes {
  const void* local_type_token{};
  std::string_view schema_id;
  ParamSchemaVersion schema_version{};
  ParamSchemaHash structural_hash{};
  std::span<const ParamConfigFieldRoute> fields;
  ParamConfigDecodeOverlay decode_overlay{};
  ParamConfigValueDigest value_digest{};
};

template <typename ParamsT>
struct ParamConfigCodecBridge {
  [[nodiscard]] static ParamResult<std::shared_ptr<const void>> decode_overlay(
      const config::ConfigView& patch, const void* baseline, const config::DecodeOptions& options) {
    if (baseline == nullptr || patch.kind() != config::ConfigValueKind::kObject) {
      return make_unexpected(ParamToolError{ParamToolErrorCode::kInvalidArgument,
                                            "PARAM_CONFIG_CODEC_BASELINE_INVALID"});
    }
    auto encoded = config::encode_config<ParamsT>(*static_cast<const ParamsT*>(baseline));
    if (!encoded) {
      return make_unexpected(
          from_config_error(std::move(encoded.error()), "PARAM_CONFIG_CODEC_BASELINE_FAILED"));
    }
    auto merged = config::merge_config_values(*encoded, patch.to_value());
    if (!merged) {
      return make_unexpected(
          from_config_error(std::move(merged.error()), "PARAM_CONFIG_CODEC_MERGE_FAILED"));
    }
    auto decoded = config::decode_config<ParamsT>(merged->view(), options);
    if (!decoded) {
      return make_unexpected(
          from_config_error(std::move(decoded.error()), "PARAM_CONFIG_CODEC_DECODE_FAILED"));
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      return std::static_pointer_cast<const void>(
          std::make_shared<const ParamsT>(std::move(*decoded)));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (const std::bad_alloc&) {
      return make_unexpected(
          ParamToolError{ParamToolErrorCode::kResourceExhausted, "PARAM_CONFIG_ALLOC_FAILED"});
    }
#endif
  }

  [[nodiscard]] static bool digest(const void* value, ParamSchemaHash& result) noexcept {
    if (value == nullptr) {
      return false;
    }
    auto encoded = config::encode_config<ParamsT>(*static_cast<const ParamsT*>(value));
    if (!encoded) {
      return false;
    }
    auto document = config::Document::from_value(std::move(*encoded));
    if (!document) {
      return false;
    }
    result.value = document->digest().semantic.sha256;
    return true;
  }
};

template <typename Builder>
struct ParamConfigFieldBridge;

template <typename ParamsT, auto MemberPtr>
struct ParamConfigFieldBridge<ParamFieldBuilder<ParamsT, MemberPtr>> {
  using FieldT =
      std::remove_cv_t<typename holoscan::detail::MemberPointerTraits<decltype(MemberPtr)>::Field>;

  [[nodiscard]] static ParamResult<void> decode(void* aggregate, const config::ConfigView& value,
                                                const config::DecodeOptions& options) {
    auto decoded = config::decode_config<FieldT>(value, options);
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

  [[nodiscard]] static bool encode(const void* aggregate, config::ConfigValue& result) noexcept {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      auto encoded =
          config::encode_config<FieldT>(static_cast<const ParamsT*>(aggregate)->*MemberPtr);
      if (!encoded) {
        return false;
      }
      result = std::move(*encoded);
      return true;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      return false;
    }
#endif
  }

  [[nodiscard]] static constexpr ParamConfigFieldRoute route() noexcept {
    ParamConfigFieldRoute result;
    using DeclaredFieldT =
        typename holoscan::detail::MemberPointerTraits<decltype(MemberPtr)>::Field;
    if constexpr (config::ConfigCodable<FieldT>) {
      result.encode = &encode;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      if constexpr (std::is_assignable_v<DeclaredFieldT&, FieldT>) {
#else
      if constexpr (std::is_nothrow_assignable_v<DeclaredFieldT&, FieldT>) {
#endif
        result.decode = &decode;
      }
    }
    return result;
  }
};

template <typename Builders>
struct ParamConfigFieldRouteTable;

template <typename... Builders>
struct ParamConfigFieldRouteTable<std::tuple<Builders...>> {
  [[nodiscard]] static constexpr auto make() noexcept {
    return std::array<ParamConfigFieldRoute, sizeof...(Builders)>{
        ParamConfigFieldBridge<Builders>::route()...};
  }
};

template <typename ParamsT>
struct ParamConfigFieldRoutes {
  using SchemaT = std::remove_cvref_t<decltype(holoscan::detail::kParamSchema<ParamsT>)>;
  using FieldDescriptors = decltype(std::declval<SchemaT>().fields);

  [[nodiscard]] static constexpr auto make_fields() noexcept {
    // The schema and this table share the builders' declaration order.
    if constexpr (requires { typename SchemaT::FieldBuilders; }) {
      return ParamConfigFieldRouteTable<typename SchemaT::FieldBuilders>::make();
    } else {
      return std::array<ParamConfigFieldRoute, std::tuple_size_v<FieldDescriptors>>{};
    }
  }

  static constexpr auto fields = make_fields();

  [[nodiscard]] static constexpr bool all_fields_encodable() noexcept {
    for (const ParamConfigFieldRoute& field_route : fields) {
      if (field_route.encode == nullptr) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] static bool digest(const void* value, ParamSchemaHash& result) noexcept {
    if (value == nullptr) {
      return false;
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      const auto& schema = param_schema<ParamsT>();
      std::array<config::ConfigValue, fields.size()> values{};
      std::array<config::ConfigObjectEntry, fields.size()> entries{};
      for (std::size_t index = 0; index < fields.size(); ++index) {
        if (fields[index].encode == nullptr || !fields[index].encode(value, values[index])) {
          return false;
        }
        entries[index] = config::ConfigObjectEntry{.key = schema.fields[index].canonical_name,
                                                   .value = values[index]};
      }
      auto object = config::ConfigValue::object(entries);
      if (!object) {
        return false;
      }
      auto document = config::Document::from_value(std::move(*object));
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

  [[nodiscard]] static constexpr ParamConfigTypeRoutes make_routes() noexcept {
    constexpr auto& schema = holoscan::detail::kParamSchema<ParamsT>;
    ParamConfigTypeRoutes result{
        .local_type_token = &holoscan::detail::kParameterTypeToken<ParamsT>,
        .schema_id = schema.header.stable_id(),
        .schema_version = schema.header.version(),
        .structural_hash = schema.structural_hash,
        .fields = fields,
    };
    if constexpr (config::ConfigCodable<ParamsT>) {
      result.decode_overlay = &ParamConfigCodecBridge<ParamsT>::decode_overlay;
    }
    if (schema.header.coverage() == ParamSchemaCoverage::kComplete && all_fields_encodable()) {
      result.value_digest = &digest;
    } else if constexpr (config::ConfigCodable<ParamsT>) {
      result.value_digest = &ParamConfigCodecBridge<ParamsT>::digest;
    }
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
/// A binding constructed directly is dynamic and must be supplied to the overload of
/// \ref decode_overrides that also receives its authoring \ref Graph. A binding returned by
/// \ref bind additionally carries a non-owning, call-scoped association with the handle's Graph;
/// all such bindings in one no-Graph decode call must name that same live Graph. The association
/// is used only while decoding and is never retained by the resulting \ref ParamOverrideSet.
///
/// Distinct binding objects may be read concurrently. Mutation, assignment, or destruction of a
/// binding must not overlap a decode that borrows it.
class ParamConfigBinding {
 public:
  /// @brief Construct an empty deferred-invalid dynamic binding.
  ///
  /// Both paths must be replaced with valid nonempty values before decoding. This constructor is
  /// intended for result containers and language bindings, not as a valid conversion request.
  ParamConfigBinding() = default;

  /// @brief Construct one owning dynamic binding.
  /// @param operator_path Exact decoded target Operator path.
  /// @param config_path Exact RFC 6901 path to the Parameter object in a document.
  /// @param mode Sparse contribution mode; baseline rejects every unrecognized value.
  /// @note This constructor does not retain a Graph and does not validate either target.
  /// Dynamic bindings can use Core's declared scalar replacement route. Typed bindings capture
  /// an aggregate codec when visible, otherwise they capture declared field codecs for fields
  /// without a Core scalar route.
  ParamConfigBinding(OperatorPath operator_path, config::Path config_path,
                     ParamBindingMode mode = ParamBindingMode::kPatch)
      : operator_path(std::move(operator_path)), config_path(std::move(config_path)), mode(mode) {}

  OperatorPath operator_path;  ///< Exact owning target path; empty is invalid for decoding.
  config::Path config_path;    ///< Exact owning document path; the root path is permitted.
  ParamBindingMode mode{ParamBindingMode::kPatch};  ///< Requested contribution semantics.

 private:
  Graph* authoring_graph_{};
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
/// @note The source Graph and handle must outlive the complete decode call. The returned binding
/// does not extend either lifetime and grants no authoring or Runtime mutation authority.
/// A visible whole-aggregate codec keeps priority for non-scalar fields. Otherwise, visible field
/// codecs from the declared Parameter schema are captured as a process-local bridge. Codec
/// specializations must be visible where this function is instantiated. The module that creates
/// the binding must remain loaded while the binding or resulting override set can be decoded,
/// compiled, or otherwise inspected. Captured routes are used only when the resolved target has
/// the same exact process-local Parameter type; otherwise non-scalar field decoding reports
/// `PARAM_CONFIG_FIELD_CODEC_UNAVAILABLE`. A complete schema needs every field to be encodable to
/// produce its value digest; otherwise provide a whole-aggregate codec.
/// When a whole-aggregate codec resolves a sparse patch, omitted non-scalar fields need a
/// non-throwing, Boolean-convertible equality comparison so compilation can verify that the codec
/// preserved them. Otherwise compilation reports `PARAMETER_OVERRIDE_UNDECLARED_CHANGE`.
/// Field-local decoding has no such requirement because HSDK copies the baseline and assigns only
/// contributed fields.
/// In a build without exceptions, captured assignment and codec operations must not throw. No
/// callback or data address enters public provenance or plan identity.
template <std::derived_from<OperatorBase> OperatorT>
[[nodiscard]] ParamConfigBinding bind(OperatorHandle<OperatorT> handle, config::Path path,
                                      ParamBindingMode mode) {
  Graph* const graph = holoscan::detail::ParameterConfigBindingAccess::graph(handle);
  if (graph == nullptr) {
    holoscan::detail::throw_graph_error(
        Error{ErrorCode::kInvalidArgument, "PARAM_CONFIG_HANDLE_INVALID"});
  }
  std::array<std::string, 1> segments{std::string{handle.name()}};
  auto operator_path = OperatorPath::from_segments(segments);
  if (!operator_path) {
    holoscan::detail::throw_graph_error(std::move(operator_path.error()));
  }
  ParamConfigBinding result{std::move(*operator_path), std::move(path), mode};
  result.authoring_graph_ = graph;
  using ParamsT = typename OperatorT::Params;
  if constexpr (param_schema_availability<ParamsT>() == ParamSchemaAvailability::kDeclared) {
    result.type_routes_ = &detail::ParamConfigFieldRoutes<ParamsT>::routes;
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
  auto parsed = config::Path::parse(path);
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

/// @brief Decode handle-bound bindings as one target-bound override transaction.
/// @param document Immutable source document. The conversion copies every retained value and
/// records its semantic/source identity without retaining a document view.
/// @param bindings Handle-bound requests; all must name one live authoring Graph.
/// @param options Immutable caller-lowered conversion limits and numeric policy.
/// @return One immutable Graph-bound override set, or a typed error with no partial result.
/// @note This cold function may compose the source Graph once. It does not compile, submit, or
/// mutate Runtime state. Duplicate Operator or field targets reject the whole call.
[[nodiscard]] HOLOSCAN_PARAMS_EXPORT ParamResult<ParamOverrideSet> decode_overrides(
    const config::Document& document, std::span<const ParamConfigBinding> bindings,
    ParamConfigDecodeOptions options = {});

/// @brief Decode a temporary list of handle-bound requests atomically.
/// @param document Immutable source document.
/// @param bindings Temporary owning bindings borrowed only for this call.
/// @param options Immutable conversion limits and numeric policy.
/// @return The same result as the span overload.
[[nodiscard]] inline ParamResult<ParamOverrideSet> decode_overrides(
    const config::Document& document, std::initializer_list<ParamConfigBinding> bindings,
    ParamConfigDecodeOptions options = {}) {
  return decode_overrides(
      document, std::span<const ParamConfigBinding>{bindings.begin(), bindings.size()}, options);
}

/// @brief Decode dynamic bindings against one explicit authoring Graph.
/// @param document Immutable source document.
/// @param graph Authoring Graph to compose and snapshot. The non-const reference makes that
/// one-time lazy composition explicit.
/// @param bindings Owning dynamic or matching handle-bound requests.
/// @param options Immutable caller-lowered conversion limits and numeric policy.
/// @return One immutable target-bound override set, or a typed error with no partial result.
/// @note Binding resolution uses only exact structured paths. It never interprets dots as Graph
/// separators, converts JSON Pointer tokens into Operator segments, or invents root authority.
/// Dynamically constructed bindings support declared scalar replacement only; use typed
/// \ref bind requests for fields that require a typed configuration codec.
[[nodiscard]] HOLOSCAN_PARAMS_EXPORT ParamResult<ParamOverrideSet> decode_overrides(
    const config::Document& document, Graph& graph, std::span<const ParamConfigBinding> bindings,
    ParamConfigDecodeOptions options = {});

/// @brief Decode a temporary list of dynamic requests against one Graph atomically.
/// @param document Immutable source document.
/// @param graph Authoring Graph to compose and snapshot once.
/// @param bindings Temporary owning bindings borrowed only for this call.
/// @param options Immutable conversion limits and numeric policy.
/// @return The same result as the span overload.
[[nodiscard]] inline ParamResult<ParamOverrideSet> decode_overrides(
    const config::Document& document, Graph& graph,
    std::initializer_list<ParamConfigBinding> bindings, ParamConfigDecodeOptions options = {}) {
  return decode_overrides(document,
                          graph,
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
