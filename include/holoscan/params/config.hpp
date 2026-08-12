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
#include <optional>
#include <span>
#include <string>
#include <string_view>
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
}  // namespace detail

/// @brief Select how a configuration object contributes to one Parameter aggregate.
///
/// EA1 supports only a sparse patch. Members absent from the bound configuration object preserve
/// the Operator's typed authoring baseline (or its type-default baseline when none was authored).
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
  /// @param mode Sparse contribution mode; EA1 rejects every unrecognized value.
  /// @note This constructor does not retain a Graph and does not validate either target.
  /// Dynamic bindings can use Core's declared scalar replacement route. A field requiring a
  /// whole-aggregate \ref config::ConfigCodec must use the typed \ref bind helper so the optional
  /// tooling package can erase that exact codec without a registry or RTTI lookup.
  ParamConfigBinding(OperatorPath operator_path, config::Path config_path,
                     ParamBindingMode mode = ParamBindingMode::kPatch)
      : operator_path(std::move(operator_path)), config_path(std::move(config_path)), mode(mode) {}

  OperatorPath operator_path;  ///< Exact owning target path; empty is invalid for decoding.
  config::Path config_path;    ///< Exact owning document path; the root path is permitted.
  ParamBindingMode mode{ParamBindingMode::kPatch};  ///< Requested contribution semantics.

 private:
  using DecodeOverlay = ParamResult<std::shared_ptr<const void>> (*)(const config::ConfigView&,
                                                                     const void*,
                                                                     const config::DecodeOptions&);
  using ValueDigest = bool (*)(const void*, ParamSchemaHash&) noexcept;

  Graph* authoring_graph_{};
  DecodeOverlay decode_overlay_{};
  ValueDigest value_digest_{};
  friend class detail::ParamConfigBindingAccess;
  template <std::derived_from<OperatorBase> OperatorT>
  friend ParamConfigBinding bind(OperatorHandle<OperatorT>, config::Path, ParamBindingMode);
};

/// @brief Create one handle-bound document conversion request.
/// @tparam OperatorT Exact authored Operator type.
/// @param handle Live handle whose source Graph supplies target identity.
/// @param path Explicit document location of the Operator's parameter object.
/// @param mode EA1 patch mode.
/// @return Owning binding with a non-owning association to the handle's Graph.
/// @throws RuntimeError if \p handle is stale or its local name cannot form a valid Operator path.
/// @note The source Graph and handle must outlive the complete decode call. The returned binding
/// does not extend either lifetime and grants no authoring or Runtime mutation authority.
/// A visible \ref config::ConfigCodec specialization for \c OperatorT::Params is captured as a
/// process-local call-only bridge; no callable address enters public provenance or plan identity.
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
  if constexpr (config::ConfigCodable<ParamsT>) {
    result.decode_overlay_ = &detail::ParamConfigCodecBridge<ParamsT>::decode_overlay;
    result.value_digest_ = &detail::ParamConfigCodecBridge<ParamsT>::digest;
  }
  return result;
}

/// @brief Create one handle-bound request from an RFC 6901 spelling.
/// @tparam OperatorT Exact authored Operator type.
/// @param handle Live handle whose source Graph supplies target identity and typed codec evidence.
/// @param path RFC 6901 path copied and validated before the binding is returned.
/// @param mode EA1 patch mode.
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
/// \ref bind requests for fields that require a whole-aggregate configuration codec.
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
/// internal fields, or treats the application document as a snapshot archive. Only EA1 fixed
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
