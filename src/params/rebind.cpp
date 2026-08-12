/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/params/rebind.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "compiler/parameter_plan_lowering.hpp"
#include "core/graph_compiler_access.hpp"
#include "core/parameter_override_set_internal.hpp"
#include "core/parameter_rebind_support.hpp"
#include "core/parameter_snapshot_archive_access.hpp"

namespace holoscan::params {
namespace detail {

struct ParamAdmissionContextState {
  Profile profile{Profile::kStandard};
};

struct TargetOperator {
  std::string path;
  std::shared_ptr<holoscan::detail::OperatorFactoryBase> factory;
};

struct ParamRebindTargetState {
  GraphAuthoringIdentity identity{};
  ParamSchemaHash admission_context_digest{};
  ParamSchemaHash selection_digest{};
  Profile profile{Profile::kStandard};
  ParamRebindTargetScopeKind scope_kind{ParamRebindTargetScopeKind::kWholeGraph};
  std::vector<std::string> scope_paths;
  std::vector<ParamRebindTargetOperatorProvenance> target_operators;
  std::vector<TargetOperator> operators;
};

}  // namespace detail
namespace {

constexpr std::size_t kMaxMigrationSteps = 16U;
constexpr std::size_t kMaxOperatorAliases = 1'024U;
constexpr std::size_t kMaxFieldAliases = 65'536U;
constexpr std::size_t kMaxOperators = 65'536U;
constexpr std::size_t kMaxTargetScopePaths = 65'536U;
constexpr std::size_t kMaxFields = 65'536U;
constexpr std::size_t kMaxPathBytes = 512U;
constexpr std::size_t kMaxSchemaIdBytes = 256U;
constexpr std::size_t kNoIndex = std::numeric_limits<std::size_t>::max();

[[nodiscard]] ParamRebindError rebind_error(
    ParamRebindErrorCode code, ParamToolErrorCode detail_code, std::string_view token,
    ParamRebindStage stage = ParamRebindStage::kSourceValidation, std::string_view source_path = {},
    std::string_view target_path = {}, std::string_view source_schema_id = {},
    std::string_view target_schema_id = {}, ParamFieldId source_field = {},
    ParamFieldId target_field = {}, std::optional<ParamSchemaHash> migration = std::nullopt,
    std::optional<Error> core = std::nullopt) {
  return ParamRebindError{code,
                          stage,
                          ParamToolError{detail_code, token, std::move(core)},
                          source_path,
                          target_path,
                          source_schema_id,
                          target_schema_id,
                          source_field,
                          target_field,
                          std::move(migration)};
}

[[nodiscard]] std::string percent_encode(std::string_view value) {
  constexpr std::string_view kHex = "0123456789ABCDEF";
  std::string result;
  result.reserve(value.size());
  for (const unsigned char byte : value) {
    const bool safe = (byte >= 'a' && byte <= 'z') || (byte >= 'A' && byte <= 'Z') ||
                      (byte >= '0' && byte <= '9') || byte == '-' || byte == '_' || byte == '.' ||
                      byte == '~';
    if (safe) {
      result.push_back(static_cast<char>(byte));
    } else {
      result.push_back('%');
      result.push_back(kHex[(byte >> 4U) & 0x0FU]);
      result.push_back(kHex[byte & 0x0FU]);
    }
  }
  return result;
}

[[nodiscard]] std::string render_segments(std::span<const std::string> segments) {
  std::string result;
  for (const auto& segment : segments) {
    if (!result.empty()) {
      result.push_back('/');
    }
    result += percent_encode(segment);
  }
  return result;
}

[[nodiscard]] expected<std::vector<std::string>, ParamRebindError> canonical_scope_paths(
    const ParamRebindTargetScope& scope) {
  const auto paths = scope.paths();
  switch (scope.kind()) {
    case ParamRebindTargetScopeKind::kWholeGraph:
      if (!paths.empty()) {
        return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidTarget,
                                            ParamToolErrorCode::kInvalidArgument,
                                            "PARAM_REBIND_TARGET_SCOPE_INVALID",
                                            ParamRebindStage::kTargetCapture));
      }
      return std::vector<std::string>{};
    case ParamRebindTargetScopeKind::kGraph:
    case ParamRebindTargetScopeKind::kOperator:
      if (paths.size() != 1U) {
        return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidTarget,
                                            ParamToolErrorCode::kInvalidArgument,
                                            "PARAM_REBIND_TARGET_SCOPE_INVALID",
                                            ParamRebindStage::kTargetCapture));
      }
      break;
    case ParamRebindTargetScopeKind::kOperatorSet:
      if (paths.empty() || paths.size() > kMaxTargetScopePaths) {
        return make_unexpected(rebind_error(
            ParamRebindErrorCode::kInvalidTarget,
            paths.empty() ? ParamToolErrorCode::kInvalidArgument
                          : ParamToolErrorCode::kResourceExhausted,
            paths.empty() ? "PARAM_REBIND_TARGET_SCOPE_EMPTY" : "PARAM_REBIND_TARGET_SCOPE_LIMIT",
            ParamRebindStage::kTargetCapture));
      }
      break;
    default:
      return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidTarget,
                                          ParamToolErrorCode::kInvalidArgument,
                                          "PARAM_REBIND_TARGET_SCOPE_INVALID",
                                          ParamRebindStage::kTargetCapture));
  }

  std::vector<std::string> result;
  result.reserve(paths.size());
  for (const auto& segments : paths) {
    const std::string rendered = render_segments(segments);
    if (segments.empty() || rendered.empty() || rendered.size() > kMaxPathBytes) {
      return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidTarget,
                                          ParamToolErrorCode::kInvalidArgument,
                                          "PARAM_REBIND_TARGET_SCOPE_PATH_INVALID",
                                          ParamRebindStage::kTargetCapture));
    }
    result.push_back(rendered);
  }
  std::ranges::sort(result);
  if (std::ranges::adjacent_find(result) != result.end()) {
    return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidTarget,
                                        ParamToolErrorCode::kInvalidArgument,
                                        "PARAM_REBIND_TARGET_SCOPE_DUPLICATE",
                                        ParamRebindStage::kTargetCapture));
  }
  return result;
}

[[nodiscard]] bool target_selected(ParamRebindTargetScopeKind kind,
                                   std::span<const std::string> scope_paths,
                                   std::string_view candidate) noexcept {
  switch (kind) {
    case ParamRebindTargetScopeKind::kWholeGraph:
      return true;
    case ParamRebindTargetScopeKind::kGraph:
      return candidate.size() > scope_paths.front().size() &&
             candidate.starts_with(scope_paths.front()) &&
             candidate[scope_paths.front().size()] == '/';
    case ParamRebindTargetScopeKind::kOperator:
      return candidate == scope_paths.front();
    case ParamRebindTargetScopeKind::kOperatorSet:
      return std::ranges::binary_search(scope_paths, candidate);
  }
  return false;
}

[[nodiscard]] bool graph_scope_exists(const holoscan::detail::GraphSnapshotData& snapshot,
                                      std::string_view scope_path) {
  return std::ranges::any_of(snapshot.graphs, [&](const GraphRecord& graph) {
    return graph.path && render_segments(graph.path->segments()) == scope_path;
  });
}

[[nodiscard]] std::optional<std::string> operator_path(
    const holoscan::detail::GraphSnapshotData& snapshot, const VertexRecord& vertex) {
  if (vertex.kind != VertexKind::kOperator || vertex.graph.value == 0U ||
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
    result += percent_encode(segment);
  };
  const auto& graph_path = graph.path;
  if (graph_path) {
    for (const auto& segment : graph_path->segments()) {
      append(segment);
    }
  }
  append(vertex.name);
  return result;
}

[[nodiscard]] std::optional<std::string> operator_path(
    const holoscan::detail::GraphSnapshotData& snapshot, VertexId owner) {
  if (owner.value == 0U || owner.value > snapshot.vertices.size()) {
    return std::nullopt;
  }
  const VertexRecord& vertex = snapshot.vertices[owner.value - 1U];
  return vertex.id == owner ? operator_path(snapshot, vertex) : std::nullopt;
}

struct SelectedTargetOperators {
  std::vector<VertexId> vertices;
  std::vector<std::string> paths;
};

[[nodiscard]] ParamRebindResult<SelectedTargetOperators> select_target_operators(
    const holoscan::detail::GraphSnapshotData& snapshot,
    const ParamRebindTargetScopeKind scope_kind, std::span<const std::string> scope_paths) {
  SelectedTargetOperators selected;
  selected.vertices.reserve(std::min(snapshot.vertices.size(), kMaxOperators + 1U));
  selected.paths.reserve(std::min(snapshot.vertices.size(), kMaxOperators + 1U));
  for (const VertexRecord& vertex : snapshot.vertices) {
    if (vertex.kind != VertexKind::kOperator) {
      continue;
    }
    auto path = operator_path(snapshot, vertex);
    if (!path) {
      return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidTarget,
                                          ParamToolErrorCode::kInvalidArgument,
                                          "PARAM_REBIND_TARGET_OPERATOR_INVALID",
                                          ParamRebindStage::kTargetCapture));
    }
    if (!target_selected(scope_kind, scope_paths, *path)) {
      continue;
    }
    if (path->size() > kMaxPathBytes) {
      return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidTarget,
                                          ParamToolErrorCode::kInvalidArgument,
                                          "PARAM_REBIND_TARGET_OPERATOR_INVALID",
                                          ParamRebindStage::kTargetCapture));
    }
    selected.vertices.push_back(vertex.id);
    selected.paths.push_back(std::move(*path));
  }
  if (selected.vertices.empty()) {
    const bool exact_operator_scope = scope_kind == ParamRebindTargetScopeKind::kOperator ||
                                      scope_kind == ParamRebindTargetScopeKind::kOperatorSet;
    return make_unexpected(rebind_error(ParamRebindErrorCode::kScopeNotFound,
                                        ParamToolErrorCode::kTargetMismatch,
                                        exact_operator_scope
                                            ? "PARAM_REBIND_TARGET_OPERATOR_UNKNOWN"
                                            : "PARAM_REBIND_TARGET_SCOPE_NOT_FOUND",
                                        ParamRebindStage::kTargetCapture));
  }
  if (selected.vertices.size() > kMaxOperators) {
    return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidTarget,
                                        ParamToolErrorCode::kResourceExhausted,
                                        "PARAM_REBIND_TARGET_OPERATOR_LIMIT",
                                        ParamRebindStage::kTargetCapture));
  }
  std::ranges::sort(selected.paths);
  const bool exact_operator_scope = scope_kind == ParamRebindTargetScopeKind::kOperator ||
                                    scope_kind == ParamRebindTargetScopeKind::kOperatorSet;
  if (exact_operator_scope && (selected.paths.size() != scope_paths.size() ||
                               !std::ranges::equal(selected.paths, scope_paths))) {
    return make_unexpected(rebind_error(ParamRebindErrorCode::kScopeNotFound,
                                        ParamToolErrorCode::kTargetMismatch,
                                        "PARAM_REBIND_TARGET_OPERATOR_UNKNOWN",
                                        ParamRebindStage::kTargetCapture));
  }
  return selected;
}

[[nodiscard]] ParamRebindResult<void> append_target_operator(
    detail::ParamRebindTargetState& state, const holoscan::detail::GraphSnapshotData& snapshot,
    holoscan::detail::ExecutableOperatorFactoryRecord& item) {
  auto path = operator_path(snapshot, item.vertex);
  if (!path || path->size() > kMaxPathBytes || !item.factory ||
      !target_selected(state.scope_kind, state.scope_paths, *path)) {
    return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidTarget,
                                        ParamToolErrorCode::kInvalidArgument,
                                        "PARAM_REBIND_TARGET_OPERATOR_INVALID",
                                        ParamRebindStage::kTargetCapture));
  }
  const ParamSchemaDescriptor* const schema = item.factory->parameter_schema();
  if (schema == nullptr) {
    return make_unexpected(rebind_error(ParamRebindErrorCode::kSchemaUnavailable,
                                        ParamToolErrorCode::kNotSupported,
                                        "PARAM_REBIND_SCHEMA_UNAVAILABLE",
                                        ParamRebindStage::kTargetCapture,
                                        {},
                                        *path));
  }
  if (schema->stable_id.empty() || schema->stable_id.size() > kMaxSchemaIdBytes ||
      schema->version == 0U) {
    return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidTarget,
                                        ParamToolErrorCode::kInvalidArgument,
                                        "PARAM_REBIND_TARGET_SCHEMA_INVALID",
                                        ParamRebindStage::kTargetCapture,
                                        {},
                                        *path));
  }
  state.target_operators.push_back(ParamRebindTargetOperatorProvenance{
      .operator_path = *path,
      .schema_id = std::string{schema->stable_id},
      .schema_version = schema->version,
      .structural_hash = schema->structural_hash,
      .contract_hash = schema->contract_hash,
  });
  state.operators.push_back(
      detail::TargetOperator{.path = std::move(*path), .factory = std::move(item.factory)});
  return {};
}

[[nodiscard]] ParamRebindResult<void> populate_target_operators(
    detail::ParamRebindTargetState& state, holoscan::detail::ParameterRebindGraphEvidence& captured,
    const std::size_t expected_count) {
  if (captured.factories.operators.size() != expected_count) {
    return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidTarget,
                                        ParamToolErrorCode::kInvalidArgument,
                                        "PARAM_REBIND_TARGET_OPERATOR_INVALID",
                                        ParamRebindStage::kTargetCapture));
  }
  state.operators.reserve(captured.factories.operators.size());
  state.target_operators.reserve(captured.factories.operators.size());
  for (auto& item : captured.factories.operators) {
    auto appended = append_target_operator(state, *captured.snapshot, item);
    if (!appended) {
      return make_unexpected(std::move(appended.error()));
    }
  }
  std::ranges::sort(state.operators, {}, &detail::TargetOperator::path);
  std::ranges::sort(
      state.target_operators, {}, &ParamRebindTargetOperatorProvenance::operator_path);
  if (std::ranges::adjacent_find(state.operators, {}, &detail::TargetOperator::path) !=
      state.operators.end()) {
    return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidTarget,
                                        ParamToolErrorCode::kInvalidArgument,
                                        "PARAM_REBIND_TARGET_OPERATOR_DUPLICATE",
                                        ParamRebindStage::kTargetCapture));
  }
  const bool exact_operator_scope = state.scope_kind == ParamRebindTargetScopeKind::kOperator ||
                                    state.scope_kind == ParamRebindTargetScopeKind::kOperatorSet;
  if (exact_operator_scope &&
      (state.operators.size() != state.scope_paths.size() ||
       !std::ranges::equal(
           state.operators, state.scope_paths, {}, &detail::TargetOperator::path))) {
    return make_unexpected(rebind_error(ParamRebindErrorCode::kScopeNotFound,
                                        ParamToolErrorCode::kTargetMismatch,
                                        "PARAM_REBIND_TARGET_OPERATOR_UNKNOWN",
                                        ParamRebindStage::kTargetCapture));
  }
  return {};
}

[[nodiscard]] std::string render_path(const OperatorPath& path) {
  std::string result;
  for (const auto& segment : path.segments()) {
    if (!result.empty()) {
      result.push_back('/');
    }
    result += percent_encode(segment);
  }
  return result;
}

[[nodiscard]] std::string render_path(const ParamPath& path) {
  std::string result;
  for (const auto& segment : path.segments()) {
    if (!result.empty()) {
      result.push_back('/');
    }
    result += percent_encode(segment);
  }
  return result;
}

struct SnapshotScopeSelector {
  ParamSnapshotScopeKind kind{ParamSnapshotScopeKind::kRoot};
  std::string path;
  bool valid{};

  [[nodiscard]] bool contains(std::string_view candidate) const noexcept {
    switch (kind) {
      case ParamSnapshotScopeKind::kRoot:
        return true;
      case ParamSnapshotScopeKind::kGraph:
        return candidate.size() > path.size() && candidate.starts_with(path) &&
               candidate[path.size()] == '/';
      case ParamSnapshotScopeKind::kOperator:
        return candidate == path;
    }
    return false;
  }
};

[[nodiscard]] SnapshotScopeSelector make_scope_selector(const ParamSnapshotScope& scope) {
  SnapshotScopeSelector result{.kind = scope.kind()};
  switch (scope.kind()) {
    case ParamSnapshotScopeKind::kRoot:
      result.valid = true;
      break;
    case ParamSnapshotScopeKind::kGraph:
    case ParamSnapshotScopeKind::kOperator:
      result.path = render_path(scope.path());
      result.valid = !result.path.empty();
      break;
  }
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

[[nodiscard]] expected<AggregateStorage, ParamRebindError> make_aggregate(
    const holoscan::detail::OperatorFactoryBase& factory, ParamRebindMode mode) {
  const ParamTypeDescriptor& type = factory.parameter_type();
  void* storage = nullptr;
  try {
    storage = ::operator new(type.object_size, std::align_val_t{type.object_alignment});
    if (mode == ParamRebindMode::kCompleteObject) {
      type.default_construct(storage);
    } else {
      type.copy_construct(storage, factory.initial_parameter_value());
    }
    return AggregateStorage{&type, storage};
  } catch (const std::bad_alloc&) {
    if (storage != nullptr) {
      ::operator delete(storage, std::align_val_t{type.object_alignment});
    }
    return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidTarget,
                                        ParamToolErrorCode::kResourceExhausted,
                                        "PARAM_REBIND_ALLOC_FAILED",
                                        mode == ParamRebindMode::kCompleteObject
                                            ? ParamRebindStage::kCompleteConstruction
                                            : ParamRebindStage::kSparseConstruction));
  } catch (...) {
    if (storage != nullptr) {
      ::operator delete(storage, std::align_val_t{type.object_alignment});
    }
    return make_unexpected(rebind_error(
        ParamRebindErrorCode::kInvalidTarget,
        ParamToolErrorCode::kInvalidArgument,
        mode == ParamRebindMode::kCompleteObject ? "PARAM_REBIND_DEFAULT_CONSTRUCT_FAILED"
                                                 : "PARAM_REBIND_BASELINE_COPY_FAILED",
        mode == ParamRebindMode::kCompleteObject ? ParamRebindStage::kCompleteConstruction
                                                 : ParamRebindStage::kSparseConstruction));
  }
}

struct SnapshotOperatorIndexEntry {
  std::string path;
  const ParamSnapshotOperatorRecord* record{};
  std::size_t ordinal{};
};

struct SnapshotOperatorIndex {
  std::vector<SnapshotOperatorIndexEntry> entries;
  bool valid{true};
};

[[nodiscard]] SnapshotOperatorIndex make_snapshot_operator_index(const ParamSnapshot& snapshot) {
  SnapshotOperatorIndex index;
  if (snapshot.operators().size() > kMaxOperators) {
    index.valid = false;
    return index;
  }
  index.entries.reserve(snapshot.operators().size());
  std::size_t field_count{};
  std::size_t ordinal{};
  for (const auto& op : snapshot.operators()) {
    std::string path = render_path(op.path);
    if (path.empty() || path.size() > kMaxPathBytes || op.schema_id.size() > kMaxSchemaIdBytes ||
        op.fields.size() > kMaxFields - field_count) {
      index.valid = false;
      return index;
    }
    field_count += op.fields.size();
    ParamFieldId previous{};
    for (const auto& field : op.fields) {
      if (field.address.field == 0U || field.address.field <= previous ||
          field.path.segments().empty() || field.type.id.empty() || field.type.version == 0U ||
          field.type.id.size() > kMaxSchemaIdBytes ||
          render_path(field.path).size() > kMaxPathBytes) {
        index.valid = false;
        return index;
      }
      previous = field.address.field;
    }
    index.entries.push_back(
        SnapshotOperatorIndexEntry{.path = std::move(path), .record = &op, .ordinal = ordinal++});
  }
  std::ranges::sort(index.entries, {}, &SnapshotOperatorIndexEntry::path);
  if (std::ranges::adjacent_find(index.entries, {}, &SnapshotOperatorIndexEntry::path) !=
      index.entries.end()) {
    index.valid = false;
  }
  return index;
}

[[nodiscard]] const SnapshotOperatorIndexEntry* find_snapshot_operator(
    const SnapshotOperatorIndex& index, std::string_view path) noexcept {
  const auto found =
      std::ranges::lower_bound(index.entries, path, {}, &SnapshotOperatorIndexEntry::path);
  return found != index.entries.end() && found->path == path ? &*found : nullptr;
}

struct OperatorAliasIndexEntry {
  std::string from_path;
  ParamSchemaVersion from_version{};
  std::string to_path;
  ParamSchemaVersion to_version{};
  const ParamOperatorAlias* alias{};
  std::size_t ordinal{};
};

struct OperatorAliasIndex {
  std::vector<OperatorAliasIndexEntry> entries;
  std::vector<std::size_t> by_source;
  std::vector<std::size_t> by_target;
  bool valid{true};
};

[[nodiscard]] int compare_text(std::string_view lhs, std::string_view rhs) noexcept {
  return lhs.compare(rhs);
}

[[nodiscard]] OperatorAliasIndex make_operator_alias_index(
    std::span<const ParamOperatorAlias> aliases) {
  OperatorAliasIndex index;
  index.entries.reserve(aliases.size());
  for (std::size_t ordinal = 0; ordinal < aliases.size(); ++ordinal) {
    const auto& alias = aliases[ordinal];
    if (alias.from_path.segments().empty() || alias.to_path.segments().empty() ||
        alias.from_schema_version == 0U || alias.to_schema_version == 0U) {
      index.valid = false;
      return index;
    }
    std::string from_path = render_path(alias.from_path);
    std::string to_path = render_path(alias.to_path);
    if (from_path.size() > kMaxPathBytes || to_path.size() > kMaxPathBytes) {
      index.valid = false;
      return index;
    }
    index.entries.push_back(OperatorAliasIndexEntry{.from_path = std::move(from_path),
                                                    .from_version = alias.from_schema_version,
                                                    .to_path = std::move(to_path),
                                                    .to_version = alias.to_schema_version,
                                                    .alias = &alias,
                                                    .ordinal = ordinal});
  }
  index.by_source.resize(index.entries.size());
  index.by_target.resize(index.entries.size());
  std::iota(index.by_source.begin(), index.by_source.end(), 0U);
  std::iota(index.by_target.begin(), index.by_target.end(), 0U);
  std::ranges::sort(index.by_source, [&](std::size_t lhs_index, std::size_t rhs_index) {
    const auto& lhs = index.entries[lhs_index];
    const auto& rhs = index.entries[rhs_index];
    return std::tie(lhs.from_path, lhs.from_version) < std::tie(rhs.from_path, rhs.from_version);
  });
  std::ranges::sort(index.by_target, [&](std::size_t lhs_index, std::size_t rhs_index) {
    const auto& lhs = index.entries[lhs_index];
    const auto& rhs = index.entries[rhs_index];
    return std::tie(lhs.to_path, lhs.to_version) < std::tie(rhs.to_path, rhs.to_version);
  });
  const auto duplicate_source = std::ranges::adjacent_find(
      index.by_source, [&](std::size_t lhs_index, std::size_t rhs_index) {
        const auto& lhs = index.entries[lhs_index];
        const auto& rhs = index.entries[rhs_index];
        return lhs.from_path == rhs.from_path && lhs.from_version == rhs.from_version;
      });
  const auto duplicate_target = std::ranges::adjacent_find(
      index.by_target, [&](std::size_t lhs_index, std::size_t rhs_index) {
        const auto& lhs = index.entries[lhs_index];
        const auto& rhs = index.entries[rhs_index];
        return lhs.to_path == rhs.to_path && lhs.to_version == rhs.to_version;
      });
  index.valid =
      duplicate_source == index.by_source.end() && duplicate_target == index.by_target.end();
  return index;
}

[[nodiscard]] const OperatorAliasIndexEntry* find_operator_alias_by_target(
    const OperatorAliasIndex& index, std::string_view path, ParamSchemaVersion version) noexcept {
  const auto found = std::lower_bound(
      index.by_target.begin(),
      index.by_target.end(),
      std::pair{path, version},
      [&](std::size_t entry_index, const auto& key) {
        const auto& entry = index.entries[entry_index];
        const int path_order = compare_text(entry.to_path, key.first);
        return path_order < 0 || (path_order == 0 && entry.to_version < key.second);
      });
  if (found == index.by_target.end()) {
    return nullptr;
  }
  const auto& entry = index.entries[*found];
  return entry.to_path == path && entry.to_version == version ? &entry : nullptr;
}

struct FieldAliasIndex {
  std::span<const ParamFieldAlias> aliases;
  std::vector<std::size_t> by_source;
  std::vector<std::size_t> by_target;
  bool valid{true};
};

[[nodiscard]] FieldAliasIndex make_field_alias_index(std::span<const ParamFieldAlias> aliases) {
  FieldAliasIndex index{.aliases = aliases};
  for (const auto& alias : aliases) {
    if (alias.from_schema_id.empty() || alias.to_schema_id.empty() ||
        alias.from_schema_id.size() > kMaxSchemaIdBytes ||
        alias.to_schema_id.size() > kMaxSchemaIdBytes || alias.from_schema_version == 0U ||
        alias.to_schema_version == 0U || alias.from_field_id == 0U || alias.to_field_id == 0U) {
      index.valid = false;
      return index;
    }
  }
  index.by_source.resize(aliases.size());
  index.by_target.resize(aliases.size());
  std::iota(index.by_source.begin(), index.by_source.end(), 0U);
  std::iota(index.by_target.begin(), index.by_target.end(), 0U);
  std::ranges::sort(index.by_source, [&](std::size_t lhs_index, std::size_t rhs_index) {
    const auto& lhs = aliases[lhs_index];
    const auto& rhs = aliases[rhs_index];
    return std::tie(lhs.from_schema_id, lhs.from_schema_version, lhs.from_field_id) <
           std::tie(rhs.from_schema_id, rhs.from_schema_version, rhs.from_field_id);
  });
  std::ranges::sort(index.by_target, [&](std::size_t lhs_index, std::size_t rhs_index) {
    const auto& lhs = aliases[lhs_index];
    const auto& rhs = aliases[rhs_index];
    return std::tie(lhs.to_schema_id, lhs.to_schema_version, lhs.to_field_id) <
           std::tie(rhs.to_schema_id, rhs.to_schema_version, rhs.to_field_id);
  });
  const auto duplicate_source = std::ranges::adjacent_find(
      index.by_source, [&](std::size_t lhs_index, std::size_t rhs_index) {
        const auto& lhs = aliases[lhs_index];
        const auto& rhs = aliases[rhs_index];
        return lhs.from_schema_id == rhs.from_schema_id &&
               lhs.from_schema_version == rhs.from_schema_version &&
               lhs.from_field_id == rhs.from_field_id;
      });
  const auto duplicate_target = std::ranges::adjacent_find(
      index.by_target, [&](std::size_t lhs_index, std::size_t rhs_index) {
        const auto& lhs = aliases[lhs_index];
        const auto& rhs = aliases[rhs_index];
        return lhs.to_schema_id == rhs.to_schema_id &&
               lhs.to_schema_version == rhs.to_schema_version && lhs.to_field_id == rhs.to_field_id;
      });
  index.valid =
      duplicate_source == index.by_source.end() && duplicate_target == index.by_target.end();
  return index;
}

[[nodiscard]] std::size_t find_field_alias_by_source(const FieldAliasIndex& index,
                                                     std::string_view schema_id,
                                                     ParamSchemaVersion version,
                                                     ParamFieldId field) noexcept {
  const auto found = std::lower_bound(index.by_source.begin(),
                                      index.by_source.end(),
                                      std::tuple{schema_id, version, field},
                                      [&](std::size_t alias_index, const auto& key) {
                                        const auto& alias = index.aliases[alias_index];
                                        const int schema_order =
                                            compare_text(alias.from_schema_id, std::get<0>(key));
                                        if (schema_order != 0) {
                                          return schema_order < 0;
                                        }
                                        if (alias.from_schema_version != std::get<1>(key)) {
                                          return alias.from_schema_version < std::get<1>(key);
                                        }
                                        return alias.from_field_id < std::get<2>(key);
                                      });
  if (found == index.by_source.end()) {
    return kNoIndex;
  }
  const auto& alias = index.aliases[*found];
  return alias.from_schema_id == schema_id && alias.from_schema_version == version &&
                 alias.from_field_id == field
             ? *found
             : kNoIndex;
}

struct TargetFieldIndexEntry {
  ParamFieldId id{};
  const ParamFieldDescriptor* descriptor{};
  std::size_t ordinal{};
};

struct TargetFieldIndex {
  std::vector<TargetFieldIndexEntry> entries;
};

[[nodiscard]] TargetFieldIndex make_target_field_index(const ParamSchemaDescriptor& schema) {
  TargetFieldIndex index;
  index.entries.reserve(schema.fields.size());
  std::size_t ordinal{};
  for (const auto& descriptor : schema.fields) {
    index.entries.push_back(TargetFieldIndexEntry{
        .id = descriptor.id, .descriptor = &descriptor, .ordinal = ordinal++});
  }
  std::ranges::sort(index.entries, {}, &TargetFieldIndexEntry::id);
  return index;
}

[[nodiscard]] const TargetFieldIndexEntry* unique_target_field(const TargetFieldIndex& index,
                                                               ParamFieldId id) noexcept {
  const auto found = std::ranges::lower_bound(index.entries, id, {}, &TargetFieldIndexEntry::id);
  if (found == index.entries.end() || found->id != id) {
    return nullptr;
  }
  const auto next = std::next(found);
  return next == index.entries.end() || next->id != id ? &*found : nullptr;
}

[[nodiscard]] const TargetFieldIndexEntry* target_field(const TargetFieldIndex& index,
                                                        const ParamSnapshotFieldRecord& source) {
  const TargetFieldIndexEntry* const match = unique_target_field(index, source.address.field);
  if (match == nullptr) {
    return nullptr;
  }
  auto target_path = ParamPath::parse(match->descriptor->canonical_name);
  if (!target_path || *target_path != source.path) {
    return nullptr;
  }
  const bool source_has_full_id = std::ranges::any_of(
      source.full_id.value, [](std::byte value) { return value != std::byte{}; });
  if (source_has_full_id && source.full_id != match->descriptor->full_id) {
    return nullptr;
  }
  return match;
}

struct SourceOperatorMatch {
  const SnapshotOperatorIndexEntry* source{};
  const ParamOperatorAlias* alias{};
  std::size_t alias_ordinal{kNoIndex};
  bool aliased{};
  bool conflict{};
};

[[nodiscard]] SourceOperatorMatch source_operator(const SnapshotOperatorIndex& snapshot,
                                                  std::string_view target_path,
                                                  ParamSchemaVersion target_version,
                                                  const OperatorAliasIndex& aliases) noexcept {
  const auto* direct = find_snapshot_operator(snapshot, target_path);
  const SnapshotOperatorIndexEntry* aliased = nullptr;
  const OperatorAliasIndexEntry* applied_alias =
      find_operator_alias_by_target(aliases, target_path, target_version);
  if (applied_alias != nullptr) {
    const auto* candidate = find_snapshot_operator(snapshot, applied_alias->from_path);
    if (candidate != nullptr && candidate->record->schema_version == applied_alias->from_version) {
      aliased = candidate;
    } else {
      applied_alias = nullptr;
    }
  }
  if (direct != nullptr && aliased != nullptr) {
    return {.source = direct, .aliased = false, .conflict = true};
  }
  if (direct != nullptr) {
    return {.source = direct, .aliased = false};
  }
  if (aliased != nullptr) {
    return {.source = aliased,
            .alias = applied_alias->alias,
            .alias_ordinal = applied_alias->ordinal,
            .aliased = true};
  }
  return {};
}

struct TargetFieldMatch {
  const ParamFieldDescriptor* descriptor{};
  const ParamFieldAlias* alias{};
  std::size_t descriptor_ordinal{kNoIndex};
  std::size_t alias_ordinal{kNoIndex};
  bool aliased{};
  bool conflict{};
};

[[nodiscard]] TargetFieldMatch resolve_target_field(
    const ParamSchemaDescriptor& target_schema, const ParamSnapshotOperatorRecord& source_operator,
    const ParamSnapshotFieldRecord& source_field, const FieldAliasIndex& aliases,
    const TargetFieldIndex& target_fields) {
  const TargetFieldIndexEntry* const direct = target_field(target_fields, source_field);
  const TargetFieldIndexEntry* aliased = nullptr;
  std::size_t applied_alias = find_field_alias_by_source(aliases,
                                                         source_operator.schema_id,
                                                         source_operator.schema_version,
                                                         source_field.address.field);
  if (applied_alias != kNoIndex) {
    const auto& alias = aliases.aliases[applied_alias];
    if (alias.to_schema_id == target_schema.stable_id &&
        alias.to_schema_version == target_schema.version) {
      aliased = unique_target_field(target_fields, alias.to_field_id);
    } else {
      applied_alias = kNoIndex;
    }
  }
  if (direct != nullptr && aliased != nullptr) {
    return {.descriptor = direct->descriptor,
            .descriptor_ordinal = direct->ordinal,
            .aliased = false,
            .conflict = true};
  }
  if (direct != nullptr) {
    return {
        .descriptor = direct->descriptor, .descriptor_ordinal = direct->ordinal, .aliased = false};
  }
  if (aliased != nullptr) {
    return {.descriptor = aliased->descriptor,
            .alias = &aliases.aliases[applied_alias],
            .descriptor_ordinal = aliased->ordinal,
            .alias_ordinal = applied_alias,
            .aliased = true};
  }
  return {};
}

[[nodiscard]] std::string_view integer_type_identity(bool is_signed,
                                                     std::uint16_t bit_width) noexcept {
  switch (bit_width) {
    case 8:
      return is_signed ? "holoscan.i8" : "holoscan.u8";
    case 16:
      return is_signed ? "holoscan.i16" : "holoscan.u16";
    case 32:
      return is_signed ? "holoscan.i32" : "holoscan.u32";
    case 64:
      return is_signed ? "holoscan.i64" : "holoscan.u64";
    default:
      return {};
  }
}

[[nodiscard]] std::string_view scalar_type_identity(const ParamFieldDescriptor& target) noexcept {
  switch (target.logical_type) {
    case ParamLogicalType::kBool:
      return "holoscan.bool";
    case ParamLogicalType::kSignedInteger:
      return integer_type_identity(true, target.type_shape.bit_width);
    case ParamLogicalType::kUnsignedInteger:
      return integer_type_identity(false, target.type_shape.bit_width);
    case ParamLogicalType::kFloatingPoint:
      if (target.type_shape.bit_width == 32U) {
        return "holoscan.f32";
      }
      return target.type_shape.bit_width == 64U ? "holoscan.f64" : std::string_view{};
    case ParamLogicalType::kDuration:
      return "holoscan.duration";
    case ParamLogicalType::kString:
    case ParamLogicalType::kSequence:
    case ParamLogicalType::kFilesystemPath:
    case ParamLogicalType::kObject:
      return {};
  }
  return {};
}

[[nodiscard]] bool scalar_type_identity_matches(const ParamSnapshotFieldRecord& source,
                                                const ParamFieldDescriptor& target) noexcept {
  if (!source.value || source.type.version != 1U || source.logical_type != target.logical_type ||
      source.type.shape.type != target.logical_type) {
    return false;
  }
  const auto scalar = source.value->scalar();
  const std::string_view expected_type = scalar_type_identity(target);
  if (expected_type.empty() || source.type.id != expected_type) {
    return false;
  }
  if (target.logical_type != ParamLogicalType::kDuration &&
      (source.type.shape.bit_width != target.type_shape.bit_width ||
       scalar.bit_width != target.type_shape.bit_width)) {
    return false;
  }
  if (!source.codec || source.codec->version != 1U) {
    return false;
  }
  constexpr std::string_view kTypePrefix = "holoscan.";
  const std::string_view expected_codec = expected_type.substr(kTypePrefix.size());
  return source.codec->id.starts_with("holoscan.config.") &&
         source.codec->id.substr(std::string_view{"holoscan.config."}.size()) == expected_codec;
}

[[nodiscard]] ParamRebindProvenance make_provenance(const ParamSnapshot& snapshot,
                                                    const detail::ParamRebindTargetState& target,
                                                    ParamRebindMode mode) {
  ParamRebindProvenance provenance;
  if (const auto archive =
          holoscan::detail::ParamSnapshotArchiveAccess::archive_identity(snapshot)) {
    provenance.source_archive_semantic_version = archive->semantic_version;
    provenance.source_archive_digest = archive->semantic_sha256;
  }
  provenance.source_plan_digest = snapshot.plan_digest();
  provenance.source_plan_generation = snapshot.plan_generation();
  provenance.source_run_generation = snapshot.run_generation();
  provenance.source_catalog_digest = snapshot.catalog_digest();
  provenance.source_scope_kind = snapshot.scope().kind();
  provenance.source_scope_path = render_path(snapshot.scope().path());
  provenance.mode = mode;
  provenance.target = target.identity;
  provenance.target_profile = target.profile;
  provenance.admission_context_digest = target.admission_context_digest;
  provenance.target_scope_kind = target.scope_kind;
  provenance.target_scope_paths = target.scope_paths;
  provenance.target_operators = target.target_operators;
  provenance.target_selection_digest = target.selection_digest;
  return provenance;
}

void append_operator_alias_provenance(ParamRebindProvenance& provenance,
                                      const ParamOperatorAlias& alias) {
  provenance.operator_aliases.push_back(ParamRebindOperatorAliasProvenance{
      .from_path = render_path(alias.from_path),
      .from_schema_version = alias.from_schema_version,
      .to_path = render_path(alias.to_path),
      .to_schema_version = alias.to_schema_version,
  });
}

void append_field_alias_provenance(ParamRebindProvenance& provenance,
                                   const ParamFieldAlias& alias) {
  provenance.field_aliases.push_back(ParamRebindFieldAliasProvenance{
      .from_schema_id = alias.from_schema_id,
      .from_schema_version = alias.from_schema_version,
      .from_field_id = alias.from_field_id,
      .to_schema_id = alias.to_schema_id,
      .to_schema_version = alias.to_schema_version,
      .to_field_id = alias.to_field_id,
  });
}

}  // namespace

ParamRebindError::ParamRebindError(ParamRebindErrorCode code, ParamRebindStage stage,
                                   ParamToolError detail, std::string_view source_path,
                                   std::string_view target_path, std::string_view source_schema_id,
                                   std::string_view target_schema_id, ParamFieldId source_field,
                                   ParamFieldId target_field,
                                   std::optional<ParamSchemaHash> migration_evidence) noexcept
    : code_(code),
      stage_(stage),
      source_field_(source_field),
      target_field_(target_field),
      migration_evidence_(std::move(migration_evidence)),
      detail_(std::move(detail)) {
  const auto copy_text = [&](std::string_view text, auto& storage, std::uint16_t& size) {
    const std::size_t retained = std::min(text.size(), storage.size());
    std::copy_n(text.begin(), retained, storage.begin());
    size = static_cast<std::uint16_t>(retained);
    context_truncated_ = context_truncated_ || retained != text.size();
  };
  copy_text(source_path, source_path_, source_path_size_);
  copy_text(target_path, target_path_, target_path_size_);
  copy_text(source_schema_id, source_schema_id_, source_schema_id_size_);
  copy_text(target_schema_id, target_schema_id_, target_schema_id_size_);
}

ParamRebindResult<ParamAdmissionContext> ParamAdmissionContext::standard() {
  try {
    return ParamAdmissionContext{std::make_shared<const detail::ParamAdmissionContextState>()};
  } catch (const std::bad_alloc&) {
    return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidTarget,
                                        ParamToolErrorCode::kResourceExhausted,
                                        "PARAM_REBIND_CONTEXT_ALLOC_FAILED",
                                        ParamRebindStage::kTargetCapture));
  }
}

Profile ParamAdmissionContext::profile() const noexcept {
  return state_ == nullptr ? Profile::kStandard : state_->profile;
}

ParamRebindTarget::~ParamRebindTarget() = default;

GraphAuthoringIdentity ParamRebindTarget::identity() const noexcept {
  return state_ == nullptr ? GraphAuthoringIdentity{} : state_->identity;
}

ParamRebindTargetScopeKind ParamRebindTarget::scope_kind() const noexcept {
  return state_ == nullptr ? ParamRebindTargetScopeKind::kWholeGraph : state_->scope_kind;
}

std::span<const std::string> ParamRebindTarget::scope_paths() const noexcept {
  return state_ == nullptr ? std::span<const std::string>{} : state_->scope_paths;
}

ParamSchemaHash ParamRebindTarget::selection_digest() const noexcept {
  return state_ == nullptr ? ParamSchemaHash{} : state_->selection_digest;
}

ParamRebindResult<ParamRebindTarget> make_param_rebind_target(Graph& graph,
                                                              const ParamAdmissionContext& context,
                                                              const ParamRebindTargetScope& scope) {
  if (context.state_ == nullptr || context.profile() != Profile::kStandard) {
    return make_unexpected(rebind_error(ParamRebindErrorCode::kProfileDenied,
                                        ParamToolErrorCode::kNotSupported,
                                        "PARAM_REBIND_PROFILE_UNSUPPORTED",
                                        ParamRebindStage::kTargetCapture));
  }
  try {
    auto scope_paths = canonical_scope_paths(scope);
    if (!scope_paths) {
      return make_unexpected(std::move(scope_paths.error()));
    }
    auto snapshot = holoscan::detail::GraphCompilerAccess::canonical_snapshot(graph);
    auto state = std::make_shared<detail::ParamRebindTargetState>();
    state->profile = context.profile();
    state->scope_kind = scope.kind();
    state->scope_paths = std::move(*scope_paths);
    if (state->scope_kind == ParamRebindTargetScopeKind::kGraph &&
        !graph_scope_exists(*snapshot, state->scope_paths.front())) {
      return make_unexpected(rebind_error(ParamRebindErrorCode::kScopeNotFound,
                                          ParamToolErrorCode::kTargetMismatch,
                                          "PARAM_REBIND_TARGET_GRAPH_UNKNOWN",
                                          ParamRebindStage::kTargetCapture,
                                          {},
                                          state->scope_paths.front()));
    }

    auto selected = select_target_operators(*snapshot, state->scope_kind, state->scope_paths);
    if (!selected) {
      return make_unexpected(std::move(selected.error()));
    }

    auto captured = holoscan::detail::capture_parameter_rebind_graph(
        graph, std::move(snapshot), selected->vertices);
    if (!captured) {
      return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidTarget,
                                          ParamToolErrorCode::kInvalidArgument,
                                          "PARAM_REBIND_TARGET_CAPTURE_FAILED",
                                          ParamRebindStage::kTargetCapture,
                                          {},
                                          {},
                                          {},
                                          {},
                                          {},
                                          {},
                                          std::nullopt,
                                          std::move(captured.error())));
    }
    state->identity = captured->snapshot->identity;
    state->admission_context_digest = captured->admission_context_digest;
    auto populated = populate_target_operators(*state, *captured, selected->vertices.size());
    if (!populated) {
      return make_unexpected(std::move(populated.error()));
    }
    state->selection_digest = holoscan::detail::ParamOverrideSetAccess::target_selection_digest(
        state->scope_kind, state->scope_paths, state->target_operators);
    return ParamRebindTarget{std::move(state)};
  } catch (const RuntimeError& error) {
    return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidTarget,
                                        ParamToolErrorCode::kInvalidArgument,
                                        "PARAM_REBIND_TARGET_CAPTURE_FAILED",
                                        ParamRebindStage::kTargetCapture,
                                        {},
                                        {},
                                        {},
                                        {},
                                        {},
                                        {},
                                        std::nullopt,
                                        make_error(error)));
  } catch (const std::bad_alloc&) {
    return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidTarget,
                                        ParamToolErrorCode::kResourceExhausted,
                                        "PARAM_REBIND_TARGET_ALLOC_FAILED",
                                        ParamRebindStage::kTargetCapture));
  } catch (const std::length_error&) {
    return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidTarget,
                                        ParamToolErrorCode::kResourceExhausted,
                                        "PARAM_REBIND_TARGET_ALLOC_FAILED",
                                        ParamRebindStage::kTargetCapture));
  }
}

// The conversion is intentionally a single fail-fast transaction: every branch records a distinct
// stable diagnostic and mutates only local staging state until the final override set is built.
// Splitting the orchestration further would obscure that ordering and duplicate the error context.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
ParamRebindResult<ParamOverrideSet> to_overrides(const ParamSnapshot& snapshot,
                                                 const ParamRebindTarget& target,
                                                 ParamRebindOptions options) {
  if (target.state_ == nullptr) {
    return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidTarget,
                                        ParamToolErrorCode::kInvalidArgument,
                                        "PARAM_REBIND_TARGET_EMPTY",
                                        ParamRebindStage::kTargetCapture));
  }
  if (target.state_->selection_digest !=
      holoscan::detail::ParamOverrideSetAccess::target_selection_digest(
          target.state_->scope_kind, target.state_->scope_paths, target.state_->target_operators)) {
    return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidTarget,
                                        ParamToolErrorCode::kTargetMismatch,
                                        "PARAM_REBIND_TARGET_SELECTION_INVALID",
                                        ParamRebindStage::kTargetCapture));
  }
  if (options.mode != ParamRebindMode::kSparseContributions &&
      options.mode != ParamRebindMode::kCompleteObject) {
    return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidSnapshot,
                                        ParamToolErrorCode::kInvalidArgument,
                                        "PARAM_REBIND_INPUT_LIMIT"));
  }
  if (options.migration_chain.size() > kMaxMigrationSteps) {
    return make_unexpected(rebind_error(ParamRebindErrorCode::kMigrationChainInvalid,
                                        ParamToolErrorCode::kResourceExhausted,
                                        "PARAM_REBIND_MIGRATION_LIMIT",
                                        ParamRebindStage::kMigration));
  }
  if (options.operator_aliases.size() > kMaxOperatorAliases) {
    return make_unexpected(rebind_error(ParamRebindErrorCode::kAliasConflict,
                                        ParamToolErrorCode::kResourceExhausted,
                                        "PARAM_REBIND_OPERATOR_ALIAS_LIMIT",
                                        ParamRebindStage::kOperatorAlias));
  }
  if (options.field_aliases.size() > kMaxFieldAliases) {
    return make_unexpected(rebind_error(ParamRebindErrorCode::kAliasConflict,
                                        ParamToolErrorCode::kResourceExhausted,
                                        "PARAM_REBIND_FIELD_ALIAS_LIMIT",
                                        ParamRebindStage::kFieldAlias));
  }
  if (!options.migration_chain.empty()) {
    return make_unexpected(rebind_error(ParamRebindErrorCode::kMigrationRequired,
                                        ParamToolErrorCode::kNotSupported,
                                        "PARAM_REBIND_MIGRATION_CATALOG_UNAVAILABLE",
                                        ParamRebindStage::kMigration));
  }
  try {
    const OperatorAliasIndex operator_aliases = make_operator_alias_index(options.operator_aliases);
    if (!operator_aliases.valid) {
      return make_unexpected(rebind_error(ParamRebindErrorCode::kAliasConflict,
                                          ParamToolErrorCode::kInvalidArgument,
                                          "PARAM_REBIND_OPERATOR_ALIAS_CONFLICT",
                                          ParamRebindStage::kOperatorAlias));
    }
    const FieldAliasIndex field_aliases = make_field_alias_index(options.field_aliases);
    if (!field_aliases.valid) {
      return make_unexpected(rebind_error(ParamRebindErrorCode::kAliasConflict,
                                          ParamToolErrorCode::kInvalidArgument,
                                          "PARAM_REBIND_FIELD_ALIAS_CONFLICT",
                                          ParamRebindStage::kFieldAlias));
    }
    if (snapshot.operators().empty()) {
      return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidSnapshot,
                                          ParamToolErrorCode::kInvalidSnapshot,
                                          "PARAM_REBIND_SNAPSHOT_EMPTY"));
    }
    const SnapshotScopeSelector scope = make_scope_selector(snapshot.scope());
    if (!scope.valid) {
      return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidSnapshot,
                                          ParamToolErrorCode::kInvalidSnapshot,
                                          "PARAM_REBIND_SOURCE_SCOPE_INVALID"));
    }
    const SnapshotOperatorIndex snapshot_index = make_snapshot_operator_index(snapshot);
    if (!snapshot_index.valid) {
      return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidSnapshot,
                                          ParamToolErrorCode::kInvalidSnapshot,
                                          "PARAM_REBIND_SNAPSHOT_IDENTITY_INVALID"));
    }
    const auto outside_scope = std::ranges::find_if(
        snapshot_index.entries, [&](const auto& entry) { return !scope.contains(entry.path); });
    if (outside_scope != snapshot_index.entries.end()) {
      return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidSnapshot,
                                          ParamToolErrorCode::kInvalidSnapshot,
                                          "PARAM_REBIND_SOURCE_OUTSIDE_SCOPE",
                                          ParamRebindStage::kSourceValidation,
                                          outside_scope->path));
    }
    ParamRebindProvenance provenance = make_provenance(snapshot, *target.state_, options.mode);
    provenance.fields.reserve(std::min<std::size_t>(kMaxFields, snapshot.operators().size() * 4U));
    std::vector<holoscan::detail::ParamOverrideOperatorState> operators;
    std::vector<unsigned char> matched_sources(snapshot.operators().size());
    std::vector<unsigned char> operator_alias_used(options.operator_aliases.size());
    std::vector<unsigned char> field_alias_used(options.field_aliases.size());
    std::size_t matched_source_count{};
    std::size_t target_field_count{};
    for (const auto& target_operator : target.state_->operators) {
      const ParamSchemaDescriptor* const schema = target_operator.factory->parameter_schema();
      if (schema == nullptr) {
        return make_unexpected(rebind_error(ParamRebindErrorCode::kSchemaUnavailable,
                                            ParamToolErrorCode::kNotSupported,
                                            "PARAM_REBIND_SCHEMA_UNAVAILABLE",
                                            ParamRebindStage::kTargetCapture,
                                            {},
                                            target_operator.path));
      }
      if (schema->stable_id.empty() || schema->stable_id.size() > kMaxSchemaIdBytes ||
          schema->fields.size() > kMaxFields - target_field_count) {
        return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidTarget,
                                            ParamToolErrorCode::kResourceExhausted,
                                            "PARAM_REBIND_TARGET_SCHEMA_LIMIT",
                                            ParamRebindStage::kTargetCapture,
                                            {},
                                            target_operator.path));
      }
      target_field_count += schema->fields.size();
      const TargetFieldIndex target_fields = make_target_field_index(*schema);
      const SourceOperatorMatch source_match =
          source_operator(snapshot_index, target_operator.path, schema->version, operator_aliases);
      if (source_match.conflict) {
        return make_unexpected(rebind_error(ParamRebindErrorCode::kAliasConflict,
                                            ParamToolErrorCode::kTargetMismatch,
                                            "PARAM_REBIND_OPERATOR_ALIAS_CONFLICT",
                                            ParamRebindStage::kOperatorAlias,
                                            {},
                                            target_operator.path));
      }
      const SnapshotOperatorIndexEntry* const source_entry = source_match.source;
      const ParamSnapshotOperatorRecord* const source =
          source_entry == nullptr ? nullptr : source_entry->record;
      if (source == nullptr) {
        if (options.mode == ParamRebindMode::kCompleteObject) {
          return make_unexpected(rebind_error(ParamRebindErrorCode::kUnknownOperator,
                                              ParamToolErrorCode::kTargetMismatch,
                                              "PARAM_REBIND_OPERATOR_MISSING",
                                              ParamRebindStage::kCompleteConstruction,
                                              {},
                                              target_operator.path));
        }
        for (const auto& field : schema->fields) {
          provenance.fields.push_back(ParamRebindFieldProvenance{
              .target_operator_path = target_operator.path,
              .target_schema_id = std::string{schema->stable_id},
              .target_schema_version = schema->version,
              .target_field_id = field.id,
              .target_full_id = field.full_id,
              .origin = ParamFieldRebindOrigin::kBaseline,
              .disposition = ParamFieldRebindDisposition::kRetainedBaseline,
          });
        }
        continue;
      }
      if (matched_sources[source_entry->ordinal] != 0U) {
        return make_unexpected(rebind_error(ParamRebindErrorCode::kAliasConflict,
                                            ParamToolErrorCode::kTargetMismatch,
                                            "PARAM_REBIND_OPERATOR_SOURCE_REUSED",
                                            ParamRebindStage::kOperatorAlias,
                                            source_entry->path,
                                            target_operator.path));
      }
      matched_sources[source_entry->ordinal] = 1U;
      ++matched_source_count;
      if (source_match.alias != nullptr) {
        if (operator_alias_used[source_match.alias_ordinal] == 0U) {
          append_operator_alias_provenance(provenance, *source_match.alias);
          operator_alias_used[source_match.alias_ordinal] = 1U;
        }
      }
      if (!source->effective_profile || *source->effective_profile != target.state_->profile) {
        return make_unexpected(rebind_error(ParamRebindErrorCode::kProfileIncompatible,
                                            ParamToolErrorCode::kTargetMismatch,
                                            "PARAM_REBIND_PROFILE_MISMATCH",
                                            ParamRebindStage::kFinalResolution,
                                            source_entry->path,
                                            target_operator.path,
                                            source->schema_id,
                                            schema->stable_id));
      }
      if (options.mode == ParamRebindMode::kCompleteObject &&
          (source->schema_availability != ParamSchemaAvailability::kDeclared ||
           source->schema_coverage != ParamSchemaCoverage::kComplete)) {
        return make_unexpected(rebind_error(ParamRebindErrorCode::kCoverageInsufficient,
                                            ParamToolErrorCode::kInvalidSnapshot,
                                            "PARAM_REBIND_COMPLETE_SCHEMA_REQUIRED",
                                            ParamRebindStage::kCompleteConstruction,
                                            source_entry->path,
                                            target_operator.path,
                                            source->schema_id,
                                            schema->stable_id));
      }
      if (source->schema_id != schema->stable_id || source->schema_version != schema->version ||
          source->structural_hash != schema->structural_hash ||
          source->contract_hash != schema->contract_hash) {
        const bool version_alias = source_match.aliased && source->schema_id == schema->stable_id &&
                                   source->structural_hash == schema->structural_hash &&
                                   source->contract_hash == schema->contract_hash;
        if (version_alias) {
          // An exact directed path/version alias may rename an otherwise identical schema. It does
          // not migrate a changed shape or policy contract.
        } else {
          return make_unexpected(rebind_error(ParamRebindErrorCode::kSchemaIncompatible,
                                              ParamToolErrorCode::kTargetMismatch,
                                              "PARAM_REBIND_SCHEMA_MISMATCH",
                                              ParamRebindStage::kFinalResolution,
                                              source_entry->path,
                                              target_operator.path,
                                              source->schema_id,
                                              schema->stable_id));
        }
      }
      auto aggregate = make_aggregate(*target_operator.factory, options.mode);
      if (!aggregate) {
        return make_unexpected(std::move(aggregate.error()));
      }
      const auto baseline =
          holoscan::detail::parameter_value_digest(target_operator.factory->parameter_type(),
                                                   schema,
                                                   aggregate->value,
                                                   target_operator.factory->has_no_params());
      if (!baseline.available()) {
        return make_unexpected(rebind_error(ParamRebindErrorCode::kFieldIncompatible,
                                            ParamToolErrorCode::kNotSupported,
                                            "PARAM_REBIND_VALUE_DIGEST_UNAVAILABLE",
                                            options.mode == ParamRebindMode::kCompleteObject
                                                ? ParamRebindStage::kCompleteConstruction
                                                : ParamRebindStage::kSparseConstruction,
                                            source_entry->path,
                                            target_operator.path,
                                            source->schema_id,
                                            schema->stable_id));
      }
      std::vector<holoscan::detail::ParamOverrideFieldOriginRecord> contributions;
      contributions.reserve(std::min(source->fields.size(), schema->fields.size()));
      std::vector<unsigned char> contributed_fields(schema->fields.size());
      for (const auto& field : source->fields) {
        if (field.projection != holoscan::params::ParamProjectionStatus::kIncluded) {
          continue;
        }
        if (!field.value) {
          return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidSnapshot,
                                              ParamToolErrorCode::kInvalidSnapshot,
                                              "PARAM_REBIND_INCLUDED_VALUE_MISSING",
                                              ParamRebindStage::kSourceValidation,
                                              source_entry->path,
                                              {},
                                              source->schema_id,
                                              {},
                                              field.address.field));
        }
        const TargetFieldMatch field_match =
            resolve_target_field(*schema, *source, field, field_aliases, target_fields);
        if (field_match.conflict) {
          return make_unexpected(rebind_error(ParamRebindErrorCode::kAliasConflict,
                                              ParamToolErrorCode::kTargetMismatch,
                                              "PARAM_REBIND_FIELD_ALIAS_CONFLICT",
                                              ParamRebindStage::kFieldAlias,
                                              source_entry->path,
                                              target_operator.path,
                                              source->schema_id,
                                              schema->stable_id,
                                              field.address.field));
        }
        const ParamFieldDescriptor* const descriptor = field_match.descriptor;
        if (descriptor == nullptr) {
          return make_unexpected(rebind_error(ParamRebindErrorCode::kUnknownField,
                                              ParamToolErrorCode::kTargetMismatch,
                                              "PARAM_REBIND_FIELD_UNKNOWN",
                                              ParamRebindStage::kFinalResolution,
                                              source_entry->path,
                                              target_operator.path,
                                              source->schema_id,
                                              schema->stable_id,
                                              field.address.field));
        }
        if (descriptor->visibility == ParamVisibility::kInternal ||
            descriptor->sensitivity == ParamSensitivity::kSensitive) {
          return make_unexpected(rebind_error(ParamRebindErrorCode::kRestrictedField,
                                              ParamToolErrorCode::kPermissionDenied,
                                              "PARAM_REBIND_FIELD_RESTRICTED",
                                              ParamRebindStage::kFinalResolution,
                                              source_entry->path,
                                              target_operator.path,
                                              source->schema_id,
                                              schema->stable_id,
                                              field.address.field,
                                              descriptor->id));
        }
        if (contributed_fields[field_match.descriptor_ordinal] != 0U) {
          return make_unexpected(rebind_error(ParamRebindErrorCode::kAliasConflict,
                                              ParamToolErrorCode::kTargetMismatch,
                                              "PARAM_REBIND_FIELD_TARGET_DUPLICATE",
                                              ParamRebindStage::kFieldAlias,
                                              source_entry->path,
                                              target_operator.path,
                                              source->schema_id,
                                              schema->stable_id,
                                              field.address.field,
                                              descriptor->id));
        }
        if (!scalar_type_identity_matches(field, *descriptor) ||
            descriptor->local_operations.replace_member_from_scalar == nullptr ||
            !descriptor->local_operations.replace_member_from_scalar(aggregate->value,
                                                                     field.value->scalar())) {
          return make_unexpected(rebind_error(ParamRebindErrorCode::kTypeMismatch,
                                              ParamToolErrorCode::kInvalidArgument,
                                              "PARAM_REBIND_FIELD_TYPE_MISMATCH",
                                              ParamRebindStage::kFinalResolution,
                                              source_entry->path,
                                              target_operator.path,
                                              source->schema_id,
                                              schema->stable_id,
                                              field.address.field,
                                              descriptor->id));
        }
        if (field_match.alias != nullptr) {
          if (field_alias_used[field_match.alias_ordinal] == 0U) {
            append_field_alias_provenance(provenance, *field_match.alias);
            field_alias_used[field_match.alias_ordinal] = 1U;
          }
        }
        const ParamFieldRebindOrigin origin = field_match.aliased
                                                  ? ParamFieldRebindOrigin::kAlias
                                                  : ParamFieldRebindOrigin::kCaptured;
        contributions.push_back(holoscan::detail::ParamOverrideFieldOriginRecord{
            .field = descriptor->id,
            .full_id = descriptor->full_id,
            .origin = origin == ParamFieldRebindOrigin::kAlias
                          ? holoscan::detail::ParamOverrideContributionOrigin::kAlias
                          : holoscan::detail::ParamOverrideContributionOrigin::kCaptured,
        });
        contributed_fields[field_match.descriptor_ordinal] = 1U;
        provenance.fields.push_back(ParamRebindFieldProvenance{
            .source_operator_path = source_entry->path,
            .source_schema_id = source->schema_id,
            .source_schema_version = source->schema_version,
            .source_field_id = field.address.field,
            .source_full_id = field.full_id,
            .target_operator_path = target_operator.path,
            .target_schema_id = std::string{schema->stable_id},
            .target_schema_version = schema->version,
            .target_field_id = descriptor->id,
            .target_full_id = descriptor->full_id,
            .origin = origin,
            .disposition = ParamFieldRebindDisposition::kContributed,
        });
      }
      if (options.mode == ParamRebindMode::kCompleteObject) {
        for (std::size_t field_index = 0; field_index < schema->fields.size(); ++field_index) {
          const auto& field = schema->fields[field_index];
          if (field.required_on_load && contributed_fields[field_index] == 0U) {
            return make_unexpected(rebind_error(ParamRebindErrorCode::kRequiredFieldMissing,
                                                ParamToolErrorCode::kInvalidSnapshot,
                                                "PARAM_REBIND_REQUIRED_FIELD_MISSING",
                                                ParamRebindStage::kCompleteConstruction,
                                                source_entry->path,
                                                target_operator.path,
                                                source->schema_id,
                                                schema->stable_id,
                                                {},
                                                field.id));
          }
        }
      }
      for (std::size_t field_index = 0; field_index < schema->fields.size(); ++field_index) {
        const auto& field = schema->fields[field_index];
        if (contributed_fields[field_index] != 0U) {
          continue;
        }
        const bool sparse = options.mode == ParamRebindMode::kSparseContributions;
        provenance.fields.push_back(ParamRebindFieldProvenance{
            .target_operator_path = target_operator.path,
            .target_schema_id = std::string{schema->stable_id},
            .target_schema_version = schema->version,
            .target_field_id = field.id,
            .target_full_id = field.full_id,
            .origin = sparse ? ParamFieldRebindOrigin::kBaseline : ParamFieldRebindOrigin::kDefault,
            .disposition = sparse ? ParamFieldRebindDisposition::kRetainedBaseline
                                  : ParamFieldRebindDisposition::kAppliedDefault,
        });
      }
      if (contributions.empty()) {
        continue;
      }
      const ParamValidationResult validation =
          target_operator.factory->parameter_type().validate_aggregate(aggregate->value);
      if (!validation.ok()) {
        return make_unexpected(rebind_error(ParamRebindErrorCode::kBoundsViolation,
                                            ParamToolErrorCode::kInvalidArgument,
                                            "PARAM_REBIND_VALIDATION_FAILED",
                                            options.mode == ParamRebindMode::kCompleteObject
                                                ? ParamRebindStage::kCompleteConstruction
                                                : ParamRebindStage::kSparseConstruction,
                                            source_entry->path,
                                            target_operator.path,
                                            source->schema_id,
                                            schema->stable_id,
                                            validation.field_id,
                                            validation.field_id));
      }
      const auto resolved =
          holoscan::detail::parameter_value_digest(target_operator.factory->parameter_type(),
                                                   schema,
                                                   aggregate->value,
                                                   target_operator.factory->has_no_params());
      if (!resolved.available()) {
        return make_unexpected(rebind_error(ParamRebindErrorCode::kFieldIncompatible,
                                            ParamToolErrorCode::kNotSupported,
                                            "PARAM_REBIND_VALUE_DIGEST_UNAVAILABLE",
                                            ParamRebindStage::kResultValidation,
                                            source_entry->path,
                                            target_operator.path,
                                            source->schema_id,
                                            schema->stable_id));
      }
      operators.push_back(holoscan::detail::ParamOverrideOperatorState{
          .operator_path = target_operator.path,
          .schema_id = std::string{schema->stable_id},
          .schema_version = schema->version,
          .structural_hash = schema->structural_hash,
          .contract_hash = schema->contract_hash,
          .local_type = &target_operator.factory->parameter_type(),
          .resolved_value = aggregate->release_shared(),
          .baseline_value_digest = baseline.digest,
          .resolved_value_digest = resolved.digest,
          .contributions = std::move(contributions),
      });
    }
    if (matched_source_count != snapshot.operators().size()) {
      return make_unexpected(rebind_error(ParamRebindErrorCode::kUnknownOperator,
                                          ParamToolErrorCode::kTargetMismatch,
                                          "PARAM_REBIND_SOURCE_OPERATOR_UNKNOWN",
                                          options.mode == ParamRebindMode::kCompleteObject
                                              ? ParamRebindStage::kCompleteConstruction
                                              : ParamRebindStage::kSparseConstruction));
    }
    if (operators.empty()) {
      return make_unexpected(rebind_error(ParamRebindErrorCode::kScopeNotFound,
                                          ParamToolErrorCode::kTargetMismatch,
                                          "PARAM_REBIND_NO_CONTRIBUTIONS",
                                          ParamRebindStage::kResultValidation));
    }
    std::ranges::sort(provenance.fields, [](const auto& lhs, const auto& rhs) {
      return std::tie(lhs.target_operator_path,
                      lhs.target_field_id,
                      lhs.source_operator_path,
                      lhs.source_field_id) < std::tie(rhs.target_operator_path,
                                                      rhs.target_field_id,
                                                      rhs.source_operator_path,
                                                      rhs.source_field_id);
    });
    return holoscan::detail::ParamOverrideSetAccess::make_complete(
        target.state_->identity,
        std::move(operators),
        target.state_->admission_context_digest,
        std::move(provenance));
  } catch (const std::bad_alloc&) {
    return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidSnapshot,
                                        ParamToolErrorCode::kResourceExhausted,
                                        "PARAM_REBIND_ALLOC_FAILED",
                                        ParamRebindStage::kResultValidation));
  }
}

ParamRebindResult<ParamOverrideSet> to_overrides(const ParamSnapshot& snapshot, Graph& graph,
                                                 ParamRebindOptions options) {
  return to_overrides(snapshot, graph, ParamRebindTargetScope::whole_graph(), options);
}

ParamRebindResult<ParamOverrideSet> to_overrides(const ParamSnapshot& snapshot, Graph& graph,
                                                 const ParamRebindTargetScope& target_scope,
                                                 ParamRebindOptions options) {
  auto context = ParamAdmissionContext::standard();
  if (!context) {
    return make_unexpected(std::move(context.error()));
  }
  auto target = make_param_rebind_target(graph, *context, target_scope);
  if (!target) {
    return make_unexpected(std::move(target.error()));
  }
  return to_overrides(snapshot, *target, options);
}

}  // namespace holoscan::params
