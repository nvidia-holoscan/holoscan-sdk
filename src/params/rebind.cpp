/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/parameter_input.hpp>
#include <holoscan/params/rebind.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <optional>
#include <ranges>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "compiler/parameter_identity.hpp"
#include "core/canonical_text_encoding.hpp"
#include "core/parameter_authoring_target.hpp"
#include "core/parameter_input_internal.hpp"
#include "core/parameter_snapshot_archive_access.hpp"

namespace holoscan::params {
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

[[nodiscard]] std::string render_segments(std::span<const std::string> segments) {
  std::string result;
  for (const auto& segment : segments) {
    if (!result.empty()) {
      result.push_back('/');
    }
    holoscan::detail::append_percent_encoded_uri_component(result, segment);
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

[[nodiscard]] std::string render_path(const ParamPath& path) {
  std::string result;
  for (const auto& segment : path.segments()) {
    if (!result.empty()) {
      result.push_back('/');
    }
    holoscan::detail::append_percent_encoded_uri_component(result, segment);
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

// Stage snapshot fields without reading a compiler snapshot or an authored parameter baseline.
ParamRebindResult<ParamRebindProvenance> set_params(Graph& graph, const ParamSnapshot& snapshot,
                                                    const ParamRebindTargetScope& target_scope,
                                                    ParamRebindOptions options) {
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
  auto scope_paths = canonical_scope_paths(target_scope);
  if (!scope_paths)
    return make_unexpected(std::move(scope_paths.error()));
  auto authored = holoscan::detail::ParameterInputAccess::authoring_targets(graph);
  if (!authored) {
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
                                        std::move(authored.error())));
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
    const SnapshotScopeSelector source_scope = make_scope_selector(snapshot.scope());
    const SnapshotOperatorIndex source_index = make_snapshot_operator_index(snapshot);
    if (!source_scope.valid || !source_index.valid || source_index.entries.empty()) {
      return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidSnapshot,
                                          ParamToolErrorCode::kInvalidSnapshot,
                                          "PARAM_REBIND_SNAPSHOT_IDENTITY_INVALID"));
    }
    const auto outside_source_scope =
        std::ranges::find_if(source_index.entries,
                             [&](const auto& entry) { return !source_scope.contains(entry.path); });
    if (outside_source_scope != source_index.entries.end()) {
      return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidSnapshot,
                                          ParamToolErrorCode::kInvalidSnapshot,
                                          "PARAM_REBIND_SOURCE_OUTSIDE_SCOPE",
                                          ParamRebindStage::kSourceValidation,
                                          outside_source_scope->path));
    }

    std::vector<const holoscan::detail::ParameterAuthoringTarget*> targets;
    for (const auto& target : authored->targets) {
      if (target_selected(target_scope.kind(), *scope_paths, target.path)) {
        targets.push_back(&target);
      }
    }
    std::ranges::sort(targets, {}, [](const auto* target) { return target->path; });
    if (targets.empty()) {
      return make_unexpected(rebind_error(ParamRebindErrorCode::kScopeNotFound,
                                          ParamToolErrorCode::kTargetMismatch,
                                          "PARAM_REBIND_TARGET_SCOPE_NOT_FOUND",
                                          ParamRebindStage::kTargetCapture));
    }
    const bool exact_scope = target_scope.kind() == ParamRebindTargetScopeKind::kOperator ||
                             target_scope.kind() == ParamRebindTargetScopeKind::kOperatorSet;
    if (exact_scope && (targets.size() != scope_paths->size() ||
                        !std::ranges::equal(targets, *scope_paths, {}, [](const auto* target) {
                          return target->path;
                        }))) {
      return make_unexpected(rebind_error(ParamRebindErrorCode::kScopeNotFound,
                                          ParamToolErrorCode::kTargetMismatch,
                                          "PARAM_REBIND_TARGET_OPERATOR_UNKNOWN",
                                          ParamRebindStage::kTargetCapture));
    }

    auto transaction = std::make_shared<holoscan::detail::ParameterTransactionState>();
    transaction->source = holoscan::detail::ParameterTransactionSource::kSnapshot;
    transaction->scope.kind = target_scope.kind();
    if (target_scope.kind() == ParamRebindTargetScopeKind::kWholeGraph) {
      transaction->scope.graph = authored->root_graph;
    } else if (target_scope.kind() == ParamRebindTargetScopeKind::kGraph) {
      const auto scope_graph = std::ranges::find(
          authored->graphs, scope_paths->front(), &holoscan::detail::ParameterAuthoringGraph::path);
      if (scope_graph == authored->graphs.end()) {
        return make_unexpected(rebind_error(ParamRebindErrorCode::kScopeNotFound,
                                            ParamToolErrorCode::kTargetMismatch,
                                            "PARAM_REBIND_TARGET_SCOPE_NOT_FOUND",
                                            ParamRebindStage::kTargetCapture));
      }
      transaction->scope.graph = scope_graph->graph;
    } else {
      transaction->scope.operators.reserve(targets.size());
      for (const auto* target : targets)
        transaction->scope.operators.push_back(target->vertex);
      std::ranges::sort(transaction->scope.operators, {}, &VertexId::value);
    }
    transaction->operators.reserve(targets.size());
    ParamRebindProvenance& provenance = transaction->provenance;
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
    provenance.mode = options.mode;
    provenance.target_scope_kind = target_scope.kind();
    provenance.target_scope_paths = *scope_paths;
    provenance.target_operators.reserve(targets.size());
    provenance.fields.reserve(std::min<std::size_t>(kMaxFields, snapshot.operators().size() * 4U));

    std::vector<unsigned char> matched_sources(snapshot.operators().size());
    std::vector<unsigned char> operator_alias_used(options.operator_aliases.size());
    std::vector<unsigned char> field_alias_used(options.field_aliases.size());
    std::size_t matched_source_count{};
    for (const auto* target : targets) {
      if (!target->type) {
        return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidTarget,
                                            ParamToolErrorCode::kInvalidArgument,
                                            "PARAM_REBIND_TARGET_OPERATOR_INVALID",
                                            ParamRebindStage::kTargetCapture,
                                            {},
                                            target->path));
      }
      const ParamSchemaDescriptor* const schema = target->schema;
      if (schema == nullptr) {
        return make_unexpected(rebind_error(ParamRebindErrorCode::kSchemaUnavailable,
                                            ParamToolErrorCode::kNotSupported,
                                            "PARAM_REBIND_SCHEMA_UNAVAILABLE",
                                            ParamRebindStage::kTargetCapture,
                                            {},
                                            target->path));
      }
      provenance.target_operators.push_back(ParamRebindTargetOperatorProvenance{
          .operator_path = target->path,
          .schema_id = std::string{schema->stable_id},
          .schema_version = schema->version,
          .structural_hash = schema->structural_hash,
          .contract_hash = schema->contract_hash,
      });
      const SourceOperatorMatch source_match =
          source_operator(source_index, target->path, schema->version, operator_aliases);
      if (source_match.conflict) {
        return make_unexpected(rebind_error(ParamRebindErrorCode::kAliasConflict,
                                            ParamToolErrorCode::kTargetMismatch,
                                            "PARAM_REBIND_OPERATOR_ALIAS_CONFLICT",
                                            ParamRebindStage::kOperatorAlias,
                                            {},
                                            target->path));
      }
      const SnapshotOperatorIndexEntry* const source = source_match.source;
      if (source == nullptr) {
        if (options.mode == ParamRebindMode::kCompleteObject) {
          return make_unexpected(rebind_error(ParamRebindErrorCode::kRequiredFieldMissing,
                                              ParamToolErrorCode::kTargetMismatch,
                                              "PARAM_REBIND_COMPLETE_TARGET_MISSING",
                                              ParamRebindStage::kCompleteConstruction,
                                              {},
                                              target->path));
        }
        continue;
      }
      if (matched_sources[source->ordinal] != 0U) {
        return make_unexpected(rebind_error(ParamRebindErrorCode::kAliasConflict,
                                            ParamToolErrorCode::kTargetMismatch,
                                            "PARAM_REBIND_OPERATOR_SOURCE_REUSED",
                                            ParamRebindStage::kOperatorAlias,
                                            source->path,
                                            target->path));
      }
      matched_sources[source->ordinal] = 1U;
      ++matched_source_count;
      if (source_match.alias != nullptr && operator_alias_used[source_match.alias_ordinal] == 0U) {
        append_operator_alias_provenance(provenance, *source_match.alias);
        operator_alias_used[source_match.alias_ordinal] = 1U;
      }
      const ParamSnapshotOperatorRecord& source_op = *source->record;
      if (!source_op.effective_profile) {
        return make_unexpected(rebind_error(ParamRebindErrorCode::kProfileIncompatible,
                                            ParamToolErrorCode::kTargetMismatch,
                                            "PARAM_REBIND_PROFILE_MISMATCH",
                                            ParamRebindStage::kFinalResolution,
                                            source->path,
                                            target->path));
      }
      if (options.mode == ParamRebindMode::kCompleteObject &&
          (source_op.schema_availability != ParamSchemaAvailability::kDeclared ||
           source_op.schema_coverage != ParamSchemaCoverage::kComplete)) {
        return make_unexpected(rebind_error(ParamRebindErrorCode::kCoverageInsufficient,
                                            ParamToolErrorCode::kInvalidSnapshot,
                                            "PARAM_REBIND_COMPLETE_SCHEMA_REQUIRED",
                                            ParamRebindStage::kCompleteConstruction,
                                            source->path,
                                            target->path,
                                            source_op.schema_id,
                                            schema->stable_id));
      }
      const bool version_alias = source_match.aliased && source_op.schema_id == schema->stable_id &&
                                 source_op.structural_hash == schema->structural_hash &&
                                 source_op.contract_hash == schema->contract_hash;
      if ((source_op.schema_id != schema->stable_id ||
           source_op.schema_version != schema->version ||
           source_op.structural_hash != schema->structural_hash ||
           source_op.contract_hash != schema->contract_hash) &&
          !version_alias) {
        return make_unexpected(rebind_error(ParamRebindErrorCode::kSchemaIncompatible,
                                            ParamToolErrorCode::kTargetMismatch,
                                            "PARAM_REBIND_SCHEMA_MISMATCH",
                                            ParamRebindStage::kFinalResolution,
                                            source->path,
                                            target->path,
                                            source_op.schema_id,
                                            schema->stable_id));
      }
      const TargetFieldIndex fields = make_target_field_index(*schema);
      std::vector<unsigned char> contributed(schema->fields.size());
      const std::uint32_t provenance_begin = static_cast<std::uint32_t>(provenance.fields.size());
      holoscan::detail::ParameterTransactionOperatorState staged{
          .target = target->vertex,
          .local_type_token = target->type->local_type_token,
          .schema_id = std::string{schema->stable_id},
          .schema_version = schema->version,
          .structural_hash = schema->structural_hash,
          .contract_hash = schema->contract_hash,
          .required_source_profile = source_op.effective_profile,
      };
      for (const ParamSnapshotFieldRecord& source_field : source_op.fields) {
        if (source_field.projection != ParamProjectionStatus::kIncluded)
          continue;
        if (!source_field.value) {
          return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidSnapshot,
                                              ParamToolErrorCode::kInvalidSnapshot,
                                              "PARAM_REBIND_INCLUDED_VALUE_MISSING",
                                              ParamRebindStage::kSourceValidation,
                                              source->path,
                                              {},
                                              source_op.schema_id,
                                              {},
                                              source_field.address.field));
        }
        const TargetFieldMatch field_match =
            resolve_target_field(*schema, source_op, source_field, field_aliases, fields);
        if (field_match.conflict) {
          return make_unexpected(rebind_error(ParamRebindErrorCode::kAliasConflict,
                                              ParamToolErrorCode::kTargetMismatch,
                                              "PARAM_REBIND_FIELD_ALIAS_CONFLICT",
                                              ParamRebindStage::kFieldAlias,
                                              source->path,
                                              target->path,
                                              source_op.schema_id,
                                              schema->stable_id,
                                              source_field.address.field));
        }
        const ParamFieldDescriptor* const descriptor = field_match.descriptor;
        if (descriptor == nullptr) {
          return make_unexpected(rebind_error(ParamRebindErrorCode::kUnknownField,
                                              ParamToolErrorCode::kTargetMismatch,
                                              "PARAM_REBIND_FIELD_UNKNOWN",
                                              ParamRebindStage::kFinalResolution,
                                              source->path,
                                              target->path,
                                              source_op.schema_id,
                                              schema->stable_id,
                                              source_field.address.field));
        }
        if (descriptor->visibility == ParamVisibility::kInternal ||
            descriptor->sensitivity == ParamSensitivity::kSensitive) {
          return make_unexpected(rebind_error(ParamRebindErrorCode::kRestrictedField,
                                              ParamToolErrorCode::kPermissionDenied,
                                              "PARAM_REBIND_FIELD_RESTRICTED",
                                              ParamRebindStage::kFinalResolution,
                                              source->path,
                                              target->path,
                                              source_op.schema_id,
                                              schema->stable_id,
                                              source_field.address.field,
                                              descriptor->id));
        }
        if (contributed[field_match.descriptor_ordinal] != 0U) {
          return make_unexpected(rebind_error(ParamRebindErrorCode::kAliasConflict,
                                              ParamToolErrorCode::kTargetMismatch,
                                              "PARAM_REBIND_FIELD_TARGET_DUPLICATE",
                                              ParamRebindStage::kFieldAlias,
                                              source->path,
                                              target->path,
                                              source_op.schema_id,
                                              schema->stable_id,
                                              source_field.address.field,
                                              descriptor->id));
        }
        if (!scalar_type_identity_matches(source_field, *descriptor) ||
            descriptor->local_operations.replace_member_from_scalar == nullptr) {
          const bool non_scalar = scalar_type_identity(*descriptor).empty();
          return make_unexpected(rebind_error(
              non_scalar ? ParamRebindErrorCode::kFieldIncompatible
                         : ParamRebindErrorCode::kTypeMismatch,
              non_scalar ? ParamToolErrorCode::kNotSupported : ParamToolErrorCode::kInvalidArgument,
              non_scalar ? "PARAM_REBIND_FIELD_CODEC_UNAVAILABLE"
                         : "PARAM_REBIND_FIELD_TYPE_MISMATCH",
              ParamRebindStage::kFinalResolution,
              source->path,
              target->path,
              source_op.schema_id,
              schema->stable_id,
              source_field.address.field,
              descriptor->id));
        }
        if (field_match.alias != nullptr && field_alias_used[field_match.alias_ordinal] == 0U) {
          append_field_alias_provenance(provenance, *field_match.alias);
          field_alias_used[field_match.alias_ordinal] = 1U;
        }
        const ParamFieldRebindOrigin origin = field_match.aliased
                                                  ? ParamFieldRebindOrigin::kAlias
                                                  : ParamFieldRebindOrigin::kCaptured;
        const std::uint32_t provenance_index = static_cast<std::uint32_t>(provenance.fields.size());
        staged.fields.push_back(holoscan::detail::ParameterTransactionFieldState{
            .field = descriptor->id,
            .full_id = descriptor->full_id,
            .origin = origin,
            .rebind_provenance_index = provenance_index,
            .scalar_value = source_field.value->scalar(),
            .staged_value_digest =
                holoscan::detail::parameter_transaction_scalar_digest(source_field.value->scalar()),
        });
        contributed[field_match.descriptor_ordinal] = 1U;
        provenance.fields.push_back(ParamRebindFieldProvenance{
            .source_operator_path = source->path,
            .source_schema_id = source_op.schema_id,
            .source_schema_version = source_op.schema_version,
            .source_field_id = source_field.address.field,
            .source_full_id = source_field.full_id,
            .target_operator_path = target->path,
            .target_schema_id = std::string{schema->stable_id},
            .target_schema_version = schema->version,
            .target_field_id = descriptor->id,
            .target_full_id = descriptor->full_id,
            .origin = origin,
            .disposition = ParamFieldRebindDisposition::kContributed,
        });
      }
      for (std::size_t index = 0; index < schema->fields.size(); ++index) {
        const ParamFieldDescriptor& field = schema->fields[index];
        if (contributed[index] != 0U)
          continue;
        if (options.mode == ParamRebindMode::kCompleteObject && field.required_on_load) {
          return make_unexpected(rebind_error(ParamRebindErrorCode::kRequiredFieldMissing,
                                              ParamToolErrorCode::kInvalidSnapshot,
                                              "PARAM_REBIND_REQUIRED_FIELD_MISSING",
                                              ParamRebindStage::kCompleteConstruction,
                                              source->path,
                                              target->path,
                                              source_op.schema_id,
                                              schema->stable_id,
                                              {},
                                              field.id));
        }
        provenance.fields.push_back(ParamRebindFieldProvenance{
            .target_operator_path = target->path,
            .target_schema_id = std::string{schema->stable_id},
            .target_schema_version = schema->version,
            .target_field_id = field.id,
            .target_full_id = field.full_id,
            .origin = options.mode == ParamRebindMode::kSparseContributions
                          ? ParamFieldRebindOrigin::kBaseline
                          : ParamFieldRebindOrigin::kDefault,
            .disposition = options.mode == ParamRebindMode::kSparseContributions
                               ? ParamFieldRebindDisposition::kRetainedBaseline
                               : ParamFieldRebindDisposition::kAppliedDefault,
        });
      }
      staged.provenance_begin = provenance_begin;
      staged.provenance_count =
          static_cast<std::uint32_t>(provenance.fields.size()) - provenance_begin;
      transaction->operators.push_back(std::move(staged));
    }
    if (matched_source_count != snapshot.operators().size()) {
      return make_unexpected(rebind_error(ParamRebindErrorCode::kUnknownOperator,
                                          ParamToolErrorCode::kTargetMismatch,
                                          "PARAM_REBIND_SOURCE_OPERATOR_UNKNOWN",
                                          options.mode == ParamRebindMode::kCompleteObject
                                              ? ParamRebindStage::kCompleteConstruction
                                              : ParamRebindStage::kSparseConstruction));
    }
    if (transaction->operators.empty()) {
      return make_unexpected(rebind_error(ParamRebindErrorCode::kScopeNotFound,
                                          ParamToolErrorCode::kTargetMismatch,
                                          "PARAM_REBIND_NO_CONTRIBUTIONS",
                                          ParamRebindStage::kResultValidation));
    }
    ParamRebindProvenance report = provenance;
    for (auto& field : provenance.fields)
      field.target_operator_path.clear();
    provenance.target_scope_paths.clear();
    provenance.target_operators.clear();
    auto attached = holoscan::detail::ParameterInputAccess::attach_transaction(
        graph, std::shared_ptr<const holoscan::detail::ParameterTransactionState>{transaction});
    if (!attached) {
      return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidTarget,
                                          ParamToolErrorCode::kTargetMismatch,
                                          "PARAM_REBIND_ATTACH_FAILED",
                                          ParamRebindStage::kResultValidation,
                                          {},
                                          {},
                                          {},
                                          {},
                                          {},
                                          {},
                                          std::nullopt,
                                          std::move(attached.error())));
    }
    return report;
  } catch (const std::bad_alloc&) {
    return make_unexpected(rebind_error(ParamRebindErrorCode::kInvalidSnapshot,
                                        ParamToolErrorCode::kResourceExhausted,
                                        "PARAM_REBIND_ALLOC_FAILED",
                                        ParamRebindStage::kResultValidation));
  }
}

}  // namespace holoscan::params
