/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Graph composition: embedded Graphs, regions, and partitions.

#include <holoscan/core/graph.hpp>
#include <holoscan/logger/logger.hpp>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "canonical_text_encoding.hpp"
#include "graph_internal.hpp"
#include "graph_mutation_rollback.hpp"
#include "graph_state.hpp"
#include "parameter_input_internal.hpp"

namespace holoscan::detail {
namespace {

constexpr std::size_t kMaximumPartitionContextFieldBytes = 192U;

[[nodiscard]] std::span<const std::string> partition_owner_path(const GraphState& state,
                                                                const PartitionRecord& partition) {
  const auto owner =
      std::ranges::find(state.embedded_graph_records_, partition.graph, &GraphRecord::id);
  if (owner == state.embedded_graph_records_.end())
    return {};
  const auto& path = owner->path;
  if (!path)
    return {};
  return path->segments();
}

// Bound each field independently and keep percent-encoded bytes intact when truncating.
[[nodiscard]] bool append_partition_context(std::string& output, std::string_view segment) {
  std::string encoded;
  append_percent_encoded_uri_component(encoded, segment);
  const std::size_t remaining = kMaximumPartitionContextFieldBytes - output.size();
  if (encoded.size() <= remaining) {
    output.append(encoded);
    return true;
  }
  std::size_t count = remaining;
  const auto escape = encoded.rfind('%', count);
  if (escape != std::string::npos && escape + 3U > count)
    count = escape;
  output.append(encoded, 0U, count);
  output.append("...");
  return false;
}

[[nodiscard]] Error internal_partition_graph_error(const EmbeddedGraphStorage& embedded) {
  const auto& state = *embedded.state;
  const auto declaration = std::ranges::min_element(
      state.partitions_, [&](const PartitionRecord& lhs, const PartitionRecord& rhs) {
        const auto lhs_path = partition_owner_path(state, lhs);
        const auto rhs_path = partition_owner_path(state, rhs);
        if (std::ranges::equal(lhs_path, rhs_path))
          return lhs.name < rhs.name;
        return std::lexicographical_compare(
            lhs_path.begin(), lhs_path.end(), rhs_path.begin(), rhs_path.end());
      });
  std::string owner;
  const auto append_path = [&](std::span<const std::string> segments) {
    for (const auto& segment : segments) {
      if (!owner.empty()) {
        if (owner.size() == kMaximumPartitionContextFieldBytes) {
          owner.append("...");
          return false;
        }
        owner.push_back('/');
      }
      if (!append_partition_context(owner, segment))
        return false;
    }
    return true;
  };
  if (append_path(embedded.path.segments()))
    static_cast<void>(append_path(partition_owner_path(state, *declaration)));
  std::string name;
  static_cast<void>(append_partition_context(name, declaration->name));
  return Error{
      ErrorCode::kInvalidArgument,
      "GRAPH_PARTITION_INTERNAL_DECLARATION (owner=" + owner + ", partition=" + name + ")"};
}

[[nodiscard]] bool region_handle_valid(const GraphState& state, RegionId id,
                                       const void* handle_identity) noexcept {
  if (!id || id.value > state.regions_.size() || id.value > state.region_handle_identities_.size())
    return false;
  const std::size_t index = static_cast<std::size_t>(id.value - 1U);
  return state.regions_[index].id == id &&
         state.region_handle_identities_[index].get() == handle_identity;
}

[[nodiscard]] bool partition_handle_valid(const GraphState& state, PartitionId id,
                                          const void* handle_identity) noexcept {
  if (!id || id.value > state.partitions_.size() ||
      id.value > state.partition_handle_identities_.size()) {
    return false;
  }
  const std::size_t index = static_cast<std::size_t>(id.value - 1U);
  return state.partitions_[index].id == id &&
         state.partition_handle_identities_[index].get() == handle_identity;
}

}  // namespace

expected<RegionRegistration, Error> graph_add_region(const std::shared_ptr<GraphState>& state,
                                                     std::string_view name) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return make_unexpected(std::move(prepared.error()));
  auto valid = validate_name(name, "GRAPH_REGION_NAME_INVALID");
  if (!valid)
    return make_unexpected(std::move(valid.error()));
  const std::lock_guard lock(state->mutex_);
  if (state->region_names_.contains(std::string(name))) {
    return make_unexpected(graph_error(ErrorCode::kDuplicateName, "GRAPH_REGION_NAME_DUPLICATE"));
  }
  GraphMutationRollback rollback(*state);
  const RegionId id = state->next_region_id();
  auto handle_identity = std::make_shared<std::byte>();
  state->region_handle_identities_.reserve(state->regions_.size() + 1U);
  state->regions_.push_back(
      RegionRecord{.id = id, .graph = state->graph_record_.id, .name = std::string(name)});
  state->region_handle_identities_.push_back(handle_identity);
  GraphMutationRollback::checkpoint(*state);
  state->region_names_.emplace(name);
  GraphMutationRollback::checkpoint(*state);
  state->mark_mutation();
  rollback.commit();
  return RegionRegistration{.id = id, .handle_identity = std::move(handle_identity)};
}

expected<void, Error> graph_region_add_operator(const std::shared_ptr<GraphState>& state,
                                                RegionId id, const void* region_handle_identity,
                                                std::size_t operator_index) {
  return graph_region_add_operators(
      state, id, region_handle_identity, std::span{&operator_index, 1U});
}

expected<void, Error> graph_region_add_operators(const std::shared_ptr<GraphState>& state,
                                                 RegionId id, const void* region_handle_identity,
                                                 std::span<const std::size_t> operator_indexes) {
  return graph_region_add_members(state, id, region_handle_identity, operator_indexes, {});
}

expected<void, Error> graph_region_add_embedded_graph(const std::shared_ptr<GraphState>& state,
                                                      RegionId id,
                                                      const void* region_handle_identity,
                                                      std::size_t embedded_index,
                                                      const void* graph_handle_identity) {
  if (!graph_embedded_valid(state, embedded_index, graph_handle_identity)) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_REGION_HANDLE_INVALID"));
  }
  return graph_region_add_members(
      state, id, region_handle_identity, {}, std::span{&embedded_index, 1U});
}

expected<void, Error> graph_region_add_members(const std::shared_ptr<GraphState>& state,
                                               RegionId id, const void* region_handle_identity,
                                               std::span<const std::size_t> operator_indexes,
                                               std::span<const std::size_t> embedded_indexes) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  const std::lock_guard lock(state->mutex_);
  if (!region_handle_valid(*state, id, region_handle_identity)) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_REGION_HANDLE_INVALID"));
  }
  if (operator_indexes.empty() && embedded_indexes.empty()) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_REGION_MEMBER_REQUIRED"));
  }

  std::vector<VertexId> added_operators;
  added_operators.reserve(operator_indexes.size());
  for (const std::size_t operator_index : operator_indexes) {
    if (operator_index >= state->operators_.size()) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_REGION_HANDLE_INVALID"));
    }
    added_operators.push_back(state->operators_[operator_index].vertex);
  }
  std::sort(added_operators.begin(), added_operators.end(), [](VertexId lhs, VertexId rhs) {
    return lhs.value < rhs.value;
  });
  if (std::adjacent_find(added_operators.begin(), added_operators.end()) != added_operators.end()) {
    return make_unexpected(graph_error(ErrorCode::kDuplicateName, "GRAPH_REGION_MEMBER_DUPLICATE"));
  }

  std::vector<GraphId> added_graphs;
  added_graphs.reserve(embedded_indexes.size());
  for (const std::size_t embedded_index : embedded_indexes) {
    if (embedded_index >= state->embedded_graphs_.size()) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_REGION_HANDLE_INVALID"));
    }
    added_graphs.push_back(state->embedded_graphs_[embedded_index].graph_id);
  }
  std::sort(added_graphs.begin(), added_graphs.end(), [](GraphId lhs, GraphId rhs) {
    return lhs.value < rhs.value;
  });
  if (std::adjacent_find(added_graphs.begin(), added_graphs.end()) != added_graphs.end()) {
    return make_unexpected(graph_error(ErrorCode::kDuplicateName, "GRAPH_REGION_MEMBER_DUPLICATE"));
  }

  auto& region = state->regions_[static_cast<std::size_t>(id.value - 1U)];
  auto members = region.members;
  const bool operator_assigned =
      std::any_of(added_operators.begin(), added_operators.end(), [&](VertexId vertex) {
        return std::binary_search(
            members.begin(), members.end(), vertex, [](VertexId lhs, VertexId rhs) {
              return lhs.value < rhs.value;
            });
      });
  auto graph_members = region.graph_members;
  const bool graph_assigned =
      std::any_of(added_graphs.begin(), added_graphs.end(), [&](GraphId graph) {
        return std::binary_search(
            graph_members.begin(), graph_members.end(), graph, [](GraphId lhs, GraphId rhs) {
              return lhs.value < rhs.value;
            });
      });
  if (operator_assigned || graph_assigned) {
    return make_unexpected(graph_error(ErrorCode::kDuplicateName, "GRAPH_REGION_MEMBER_DUPLICATE"));
  }

  members.reserve(members.size() + added_operators.size());
  members.insert(members.end(), added_operators.begin(), added_operators.end());
  std::sort(members.begin(), members.end(), [](VertexId lhs, VertexId rhs) {
    return lhs.value < rhs.value;
  });
  graph_members.reserve(graph_members.size() + added_graphs.size());
  graph_members.insert(graph_members.end(), added_graphs.begin(), added_graphs.end());
  std::sort(graph_members.begin(), graph_members.end(), [](GraphId lhs, GraphId rhs) {
    return lhs.value < rhs.value;
  });
  GraphMutationRollback::checkpoint(*state);
  region.members.swap(members);
  region.graph_members.swap(graph_members);
  // Preserve the established per-member generation for embedded-Graph batches while publishing
  // all membership only after every member has validated.
  const std::size_t mutation_count =
      embedded_indexes.empty() ? 1U : operator_indexes.size() + embedded_indexes.size();
  for (std::size_t index = 0U; index < mutation_count; ++index) {
    state->mark_mutation();
  }
  return {};
}

expected<void, Error> graph_region_add_constraint(const std::shared_ptr<GraphState>& state,
                                                  RegionId id, const void* region_handle_identity,
                                                  RegionConstraint constraint) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  if (!closed_enum(constraint.kind())) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_REGION_CONSTRAINT_INVALID"));
  }
  if (constraint.kind() == RegionConstraintKind::kLatencyBudget &&
      constraint.latency().count() <= 0) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_REGION_LATENCY_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  if (!region_handle_valid(*state, id, region_handle_identity)) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_REGION_HANDLE_INVALID"));
  }
  auto& region = state->regions_[static_cast<std::size_t>(id.value - 1U)];
  auto constraints = region.constraints;
  if (std::any_of(constraints.begin(), constraints.end(), [&](const RegionConstraint& existing) {
        return existing.kind() == constraint.kind();
      })) {
    return make_unexpected(
        graph_error(ErrorCode::kDuplicateName, "GRAPH_REGION_CONSTRAINT_DUPLICATE"));
  }
  constraints.push_back(constraint);
  std::sort(constraints.begin(), constraints.end(), [](const auto& lhs, const auto& rhs) {
    return static_cast<std::underlying_type_t<RegionConstraintKind>>(lhs.kind()) <
           static_cast<std::underlying_type_t<RegionConstraintKind>>(rhs.kind());
  });
  GraphMutationRollback::checkpoint(*state);
  region.constraints = std::move(constraints);
  state->mark_mutation();
  return {};
}

expected<void, Error> graph_region_set_profile(const std::shared_ptr<GraphState>& state,
                                               RegionId id, const void* region_handle_identity,
                                               Profile profile) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  if (!closed_enum(profile)) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_PROFILE_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  if (!region_handle_valid(*state, id, region_handle_identity)) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_REGION_HANDLE_INVALID"));
  }
  auto& region = state->regions_[static_cast<std::size_t>(id.value - 1U)];
  if (region.profile) {
    return make_unexpected(
        graph_error(ErrorCode::kDuplicateName, "GRAPH_REGION_PROFILE_DUPLICATE"));
  }
  region.profile = profile;
  state->mark_mutation();
  return {};
}

expected<PartitionRegistration, Error> graph_add_partition(const std::shared_ptr<GraphState>& state,
                                                           std::string_view name) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return make_unexpected(std::move(prepared.error()));
  auto valid = validate_name(name, "GRAPH_PARTITION_NAME_INVALID");
  if (!valid)
    return make_unexpected(std::move(valid.error()));
  const std::lock_guard lock(state->mutex_);
  if (std::any_of(state->partitions_.begin(),
                  state->partitions_.end(),
                  [&](const PartitionRecord& partition) {
                    return partition.graph == state->graph_record_.id && partition.include_all;
                  })) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_PARTITION_INCLUDE_ALL_CONFLICT"));
  }
  if (state->partition_names_.contains(std::string(name))) {
    return make_unexpected(
        graph_error(ErrorCode::kDuplicateName, "GRAPH_PARTITION_NAME_DUPLICATE"));
  }
  GraphMutationRollback rollback(*state);
  const PartitionId id = state->next_partition_id();
  auto handle_identity = std::make_shared<std::byte>();
  state->partition_handle_identities_.reserve(state->partitions_.size() + 1U);
  state->partitions_.push_back(
      PartitionRecord{.id = id, .graph = state->graph_record_.id, .name = std::string(name)});
  state->partition_handle_identities_.push_back(handle_identity);
  GraphMutationRollback::checkpoint(*state);
  state->partition_names_.emplace(name);
  GraphMutationRollback::checkpoint(*state);
  state->mark_mutation();
  rollback.commit();
  return PartitionRegistration{.id = id, .handle_identity = std::move(handle_identity)};
}

expected<void, Error> graph_partition_add_operator(const std::shared_ptr<GraphState>& state,
                                                   PartitionId id,
                                                   const void* partition_handle_identity,
                                                   std::size_t operator_index) {
  return graph_partition_add_operators(
      state, id, partition_handle_identity, std::span{&operator_index, 1U});
}

expected<void, Error> graph_partition_add_operators(const std::shared_ptr<GraphState>& state,
                                                    PartitionId id,
                                                    const void* partition_handle_identity,
                                                    std::span<const std::size_t> operator_indexes) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  const std::lock_guard lock(state->mutex_);
  if (!partition_handle_valid(*state, id, partition_handle_identity) || operator_indexes.empty()) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_PARTITION_HANDLE_INVALID"));
  }

  auto& partition = state->partitions_[static_cast<std::size_t>(id.value - 1U)];
  if (partition.include_all) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_PARTITION_INCLUDE_ALL_CONFLICT"));
  }

  std::vector<VertexId> added;
  added.reserve(operator_indexes.size());
  for (const std::size_t operator_index : operator_indexes) {
    if (operator_index >= state->operators_.size()) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_PARTITION_HANDLE_INVALID"));
    }
    added.push_back(state->operators_[operator_index].vertex);
  }
  std::sort(
      added.begin(), added.end(), [](VertexId lhs, VertexId rhs) { return lhs.value < rhs.value; });
  if (std::adjacent_find(added.begin(), added.end()) != added.end()) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_PARTITION_MEMBER_DUPLICATE"));
  }
  const bool assigned = std::any_of(
      state->partitions_.begin(), state->partitions_.end(), [&](const PartitionRecord& candidate) {
        if (candidate.graph != state->graph_record_.id)
          return false;
        return std::any_of(added.begin(), added.end(), [&](VertexId vertex) {
          return std::binary_search(
              candidate.members.begin(),
              candidate.members.end(),
              vertex,
              [](VertexId lhs, VertexId rhs) { return lhs.value < rhs.value; });
        });
      });
  if (assigned) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_PARTITION_MEMBER_DUPLICATE"));
  }

  auto members = partition.members;
  members.reserve(members.size() + added.size());
  members.insert(members.end(), added.begin(), added.end());
  std::sort(members.begin(), members.end(), [](VertexId lhs, VertexId rhs) {
    return lhs.value < rhs.value;
  });
  GraphMutationRollback::checkpoint(*state);
  partition.members = std::move(members);
  state->mark_mutation();
  return {};
}

expected<void, Error> graph_partition_add_embedded_graph(const std::shared_ptr<GraphState>& state,
                                                         PartitionId id,
                                                         const void* partition_handle_identity,
                                                         std::size_t embedded_index,
                                                         const void* graph_handle_identity) {
  if (!graph_embedded_valid(state, embedded_index, graph_handle_identity)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_PARTITION_HANDLE_INVALID"));
  }
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  const std::lock_guard lock(state->mutex_);
  if (!partition_handle_valid(*state, id, partition_handle_identity) ||
      embedded_index >= state->embedded_graphs_.size()) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_PARTITION_HANDLE_INVALID"));
  }
  auto& partition = state->partitions_[static_cast<std::size_t>(id.value - 1U)];
  if (partition.include_all) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_PARTITION_INCLUDE_ALL_CONFLICT"));
  }
  const auto& embedded = state->embedded_graphs_[embedded_index];
  // Child snapshots already include recursively embedded metadata. Never discard an inner
  // declaration, even if it is empty, when assigning this subtree to a parent owner.
  if (!embedded.state->partitions_.empty()) {
    GraphMutationRollback::checkpoint(*state);
    return make_unexpected(internal_partition_graph_error(embedded));
  }
  if (std::none_of(embedded.state->vertices_.begin(),
                   embedded.state->vertices_.end(),
                   [](const auto& vertex) { return vertex.kind == VertexKind::kOperator; })) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_PARTITION_GRAPH_EMPTY"));
  }
  for (const auto& candidate : state->partitions_) {
    if (std::find(candidate.graph_members.begin(),
                  candidate.graph_members.end(),
                  embedded.graph_id) != candidate.graph_members.end()) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_PARTITION_MEMBER_DUPLICATE"));
    }
  }
  auto members = partition.graph_members;
  members.push_back(embedded.graph_id);
  std::sort(members.begin(), members.end(), [](GraphId lhs, GraphId rhs) {
    return lhs.value < rhs.value;
  });
  GraphMutationRollback::checkpoint(*state);
  partition.graph_members = std::move(members);
  state->mark_mutation();
  return {};
}

expected<void, Error> graph_partition_include_all(const std::shared_ptr<GraphState>& state,
                                                  PartitionId id,
                                                  const void* partition_handle_identity) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  const std::lock_guard lock(state->mutex_);
  if (!partition_handle_valid(*state, id, partition_handle_identity)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_PARTITION_HANDLE_INVALID"));
  }

  auto& partition = state->partitions_[static_cast<std::size_t>(id.value - 1U)];
  if (partition.include_all)
    return {};
  if (state->automatic_partition_remaining_) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_PARTITION_AUTOMATIC_REMAINDER_CONFLICT"));
  }
  if (partition.graph != state->graph_record_.id || !partition.members.empty() ||
      !partition.graph_members.empty() || state->partitions_.size() != 1U) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_PARTITION_INCLUDE_ALL_CONFLICT"));
  }

  partition.include_all = true;
  state->mark_mutation();
  return {};
}

expected<void, Error> graph_auto_partition_remaining(const std::shared_ptr<GraphState>& state) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  const std::lock_guard lock(state->mutex_);
  if (std::any_of(state->partitions_.begin(),
                  state->partitions_.end(),
                  [&](const PartitionRecord& partition) {
                    return partition.graph == state->graph_record_.id && partition.include_all;
                  })) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_PARTITION_AUTOMATIC_REMAINDER_CONFLICT"));
  }
  if (state->automatic_partition_remaining_)
    return {};
  state->automatic_partition_remaining_ = true;
  state->mark_mutation();
  return {};
}

namespace {

template <typename Id>
using IdMap = std::unordered_map<std::uint64_t, Id>;

[[nodiscard]] const VertexRecord* find_vertex_record(const GraphSnapshotData& snapshot,
                                                     VertexId id) noexcept {
  const auto found = std::find_if(snapshot.vertices.begin(),
                                  snapshot.vertices.end(),
                                  [&](const VertexRecord& record) { return record.id == id; });
  return found == snapshot.vertices.end() ? nullptr : &*found;
}

}  // namespace

expected<void, Error> graph_prepare_embedded_type(const std::shared_ptr<GraphState>& parent,
                                                  const std::shared_ptr<GraphState>& child,
                                                  const void* child_type_token) {
  if (!parent || !child || parent == child || child_type_token == nullptr) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_EMBED_CHILD_INVALID"));
  }
  const std::scoped_lock lock(parent->mutex_, child->mutex_);
  std::vector<const void*> ancestry = parent->embedding_ancestry_;
  if (std::find(ancestry.begin(), ancestry.end(), child_type_token) != ancestry.end()) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_EMBED_CYCLE"));
  }
  // Each embedded state carries the complete type-token path through itself.
  // The next nested embed can therefore reject a repeated type without RTTI,
  // compiler-specific type names, or a dynamic token on the root Graph.
  ancestry.push_back(child_type_token);
  child->embedding_ancestry_ = std::move(ancestry);
  return {};
}

expected<EmbeddedGraphRegistration, Error>
graph_register_embedded(  // NOLINT(readability-function-cognitive-complexity)
    const std::shared_ptr<GraphState>& parent, std::string_view instance_name,
    std::unique_ptr<Graph> child, const std::shared_ptr<GraphState>& child_state,
    const std::shared_ptr<const GraphSnapshotData>& child_snapshot) {
  auto prepared = parent->prepare_mutation();
  if (!prepared)
    return make_unexpected(std::move(prepared.error()));
  auto valid_name = graph_prepare_embed(parent, instance_name);
  if (!valid_name)
    return make_unexpected(std::move(valid_name.error()));
  if (!child || !child_state || child_state.get() == parent.get() || !child_snapshot ||
      child_snapshot->graphs.empty()) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_EMBED_CHILD_INVALID"));
  }
  if (std::any_of(child_snapshot->partitions.begin(),
                  child_snapshot->partitions.end(),
                  [](const PartitionRecord& partition) { return partition.include_all; })) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_PARTITION_INCLUDE_ALL_CONFLICT"));
  }
  if (child_snapshot->automatic_partition_remaining) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_PARTITION_AUTOMATIC_REMAINDER_CONFLICT"));
  }
  std::shared_ptr<const void> handle_identity = std::make_shared<std::byte>(std::byte{});

  // Child composition and deep snapshot creation completed before this call,
  // so publication reuses that immutable snapshot without repeating its copy,
  // canonicalization, or hash. The terminal child is exclusively owned here
  // and cannot mutate, so parent commit need not hold the child mutex. This is
  // essential: rollback may destroy the child Graph, whose destructor locks
  // that mutex to detach its state.
  const std::lock_guard lock(parent->mutex_);
  if (parent->parameter_transaction_ && child_state->parameter_transaction_) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_PARAMETER_TRANSACTION_CONFLICT"));
  }
  if (!child_snapshot->partitions.empty() &&
      std::any_of(parent->partitions_.begin(),
                  parent->partitions_.end(),
                  [&](const PartitionRecord& partition) {
                    return partition.graph == parent->graph_record_.id && partition.include_all;
                  })) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_PARTITION_INCLUDE_ALL_CONFLICT"));
  }
  if (parent->embedded_instance_names_.contains(std::string(instance_name))) {
    return make_unexpected(
        graph_error(ErrorCode::kDuplicateName, "GRAPH_EMBED_INSTANCE_DUPLICATE"));
  }
  std::unordered_set<std::string_view> embedded_endpoint_names;
  embedded_endpoint_names.reserve(child_snapshot->external_topics.size());
  for (const auto& topic : child_snapshot->external_topics) {
    if (topic.endpoint_name && (parent->explicit_endpoint_names_.contains(*topic.endpoint_name) ||
                                !embedded_endpoint_names.emplace(*topic.endpoint_name).second)) {
      return make_unexpected(
          graph_error(ErrorCode::kDuplicateName, "GRAPH_ENDPOINT_NAME_DUPLICATE"));
    }
  }

  GraphMutationRollback rollback(*parent);
  const std::size_t embedded_index = parent->embedded_graphs_.size();
  IdMap<GraphId> graph_ids;
  IdMap<VertexId> vertex_ids;
  IdMap<GraphPortId> port_ids;

  const GraphId child_root = child_snapshot->graphs.front().id;
  GraphId embedded_root{};
  for (const GraphRecord& source : child_snapshot->graphs) {
    std::vector<std::string> segments{std::string(instance_name)};
    if (source.path) {
      segments.insert(
          segments.end(), source.path->segments().begin(), source.path->segments().end());
    }
    GraphInstancePath path = GraphState::make_path(std::move(segments));
    const GraphId id = parent->next_graph_id();
    graph_ids.emplace(source.id.value, id);
    if (source.id == child_root)
      embedded_root = id;
    const GraphId mapped_parent =
        source.parent ? graph_ids.at(source.parent->value) : parent->graph_record_.id;
    parent->embedded_graph_records_.push_back(GraphRecord{
        .id = id,
        .parent = mapped_parent,
        .name = source.id == child_root ? std::string(instance_name) : source.name,
        .path = std::move(path),
    });
    GraphMutationRollback::checkpoint(*parent);
  }

  for (const VertexRecord& source : child_snapshot->vertices) {
    const VertexId id = parent->next_vertex_id();
    vertex_ids.emplace(source.id.value, id);
    parent->vertices_.push_back(VertexRecord{
        .id = id,
        .graph = graph_ids.at(source.graph.value),
        .kind = source.kind,
        .name = source.name,
        .operator_host = source.operator_host,
        .serialization_domain = source.serialization_domain,
        .cuda_graph_capture_eligible = source.cuda_graph_capture_eligible,
    });
    parent->vertex_adjacency_.emplace_back();
    parent->cycle_visit_marks_.push_back(0U);
    GraphMutationRollback::checkpoint(*parent);
  }

  for (const PortStorage& source : child_state->ports_) {
    auto valid_catalog = validate_pod_catalog(*parent, source.record.payload_identity);
    if (!valid_catalog)
      return make_unexpected(std::move(valid_catalog.error()));
    const GraphPortId id = parent->next_port_id();
    port_ids.emplace(source.record.id.value, id);
    PortStorage copied = source;
    copied.record.id = id;
    copied.record.owner = vertex_ids.at(source.record.owner.value);
    copied.handle_identity = std::make_shared<std::byte>();
    copied.operator_index.reset();
    parent->ports_.push_back(std::move(copied));
    // The parent may wire only the child's declared Graph interface. Internal
    // Operator and external-boundary declarations remain owned by the child
    // state and are flattened as immutable compiler input, never as parent
    // authoring capabilities.
    const VertexRecord* source_owner = find_vertex_record(*child_snapshot, source.record.owner);
    if (source.record.owner_kind == PortOwnerKind::kGraphInterface && source_owner != nullptr &&
        source_owner->graph == child_root) {
      parent->port_by_declaration_.emplace(source.declaration, id);
    }
    record_pod_identity(*parent, parent->ports_.back().record.payload_identity);
    GraphMutationRollback::checkpoint(*parent);
  }
  const std::size_t copied_port_begin = parent->ports_.size() - child_state->ports_.size();
  for (std::size_t index = copied_port_begin; index < parent->ports_.size(); ++index) {
    auto& copied = parent->ports_[index].record;
    if (copied.setup.follows_input) {
      copied.setup.follows_input = port_ids.at(copied.setup.follows_input->value);
    }
    if (copied.graph_override.follows_input) {
      copied.graph_override.follows_input = port_ids.at(copied.graph_override.follows_input->value);
    }
  }

  for (std::size_t index = 0; index < child_state->operators_.size(); ++index) {
    parent->embedded_operator_mappings_.push_back(EmbeddedOperatorMapping{
        .source_state = child_state.get(),
        .source_index = index,
        .vertex = vertex_ids.at(child_state->operators_[index].vertex.value),
        .factory = child_state->operators_[index].factory->clone(),
        .parameter_input_evidence = child_state->operators_[index].parameter_input_evidence,
    });
    GraphMutationRollback::checkpoint(*parent);
  }
  for (const auto& mapping : child_state->embedded_operator_mappings_) {
    parent->embedded_operator_mappings_.push_back(EmbeddedOperatorMapping{
        .source_state = mapping.source_state,
        .source_index = mapping.source_index,
        .vertex = vertex_ids.at(mapping.vertex.value),
        .factory = mapping.factory->clone(),
        .parameter_input_evidence = mapping.parameter_input_evidence,
    });
    GraphMutationRollback::checkpoint(*parent);
  }

  for (const EdgeRecord& source : child_snapshot->edges) {
    EdgeRecord copied = source;
    copied.id = parent->next_edge_id();
    copied.source = port_ids.at(source.source.value);
    copied.target = port_ids.at(source.target.value);
    const PortStorage* const copied_source = find_port(*parent, copied.source);
    const PortStorage* const copied_target = find_port(*parent, copied.target);
    if (copied_source == nullptr || copied_target == nullptr) {
      return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_EMBED_CHILD_INVALID"));
    }
    reserve_adjacency(*parent, copied_source->record.owner);
    auto handle_identity = std::make_shared<std::byte>();
    parent->edge_handle_identities_.reserve(parent->edge_handle_identities_.size() + 1U);
    parent->edges_.push_back(std::move(copied));
    parent->edge_handle_identities_.push_back(std::move(handle_identity));
    append_adjacency(*parent, copied_source->record.owner, copied_target->record.owner);
    GraphMutationRollback::checkpoint(*parent);
  }

  std::vector<InterfaceBindingRecord> bindings = child_snapshot->interface_bindings;
  for (auto& binding : bindings) {
    binding.public_port = port_ids.at(binding.public_port.value);
    binding.internal_port = port_ids.at(binding.internal_port.value);
    const PortStorage* const public_port = find_port(*parent, binding.public_port);
    const PortStorage* const internal_port = find_port(*parent, binding.internal_port);
    if (public_port == nullptr || internal_port == nullptr) {
      return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_EMBED_CHILD_INVALID"));
    }
    const VertexId before = binding.direction == PortDirection::kInput
                                ? public_port->record.owner
                                : internal_port->record.owner;
    const VertexId after = binding.direction == PortDirection::kInput ? internal_port->record.owner
                                                                      : public_port->record.owner;
    reserve_adjacency(*parent, before);
    parent->interface_bindings_.push_back(std::move(binding));
    append_adjacency(*parent, before, after);
    GraphMutationRollback::checkpoint(*parent);
  }

  std::vector<ExternalTopicRecord> external_topics = child_snapshot->external_topics;
  for (auto& topic : external_topics) {
    topic.id = parent->next_external_topic_id();
    topic.vertex = vertex_ids.at(topic.vertex.value);
    topic.graph_port = port_ids.at(topic.graph_port.value);
    ExternalTopicStorage storage{.record = std::move(topic)};
    parent->external_topics_.push_back(std::move(storage));
    GraphMutationRollback::checkpoint(*parent);
    const auto& endpoint_name = parent->external_topics_.back().record.endpoint_name;
    if (endpoint_name) {
      const auto [unused, inserted] = parent->explicit_endpoint_names_.emplace(*endpoint_name);
      (void)unused;
      if (!inserted) {
        return make_unexpected(
            graph_error(ErrorCode::kDuplicateName, "GRAPH_ENDPOINT_NAME_DUPLICATE"));
      }
      GraphMutationRollback::checkpoint(*parent);
    }
  }

  std::vector<ReadinessSourceDeclarationRecord> readiness = child_snapshot->readiness_sources;
  for (auto& source : readiness) {
    source.owner = vertex_ids.at(source.owner.value);
    parent->readiness_sources_.push_back(std::move(source));
    GraphMutationRollback::checkpoint(*parent);
  }

  std::vector<RegionRecord> regions = child_snapshot->regions;
  for (auto& region : regions) {
    region.id = parent->next_region_id();
    region.graph = graph_ids.at(region.graph.value);
    for (auto& member : region.members)
      member = vertex_ids.at(member.value);
    for (auto& member : region.graph_members)
      member = graph_ids.at(member.value);
    auto region_handle_identity = std::make_shared<std::byte>();
    parent->region_handle_identities_.reserve(parent->regions_.size() + 1U);
    parent->regions_.push_back(std::move(region));
    parent->region_handle_identities_.push_back(std::move(region_handle_identity));
    GraphMutationRollback::checkpoint(*parent);
  }

  std::vector<PartitionRecord> partitions = child_snapshot->partitions;
  for (auto& partition : partitions) {
    partition.id = parent->next_partition_id();
    partition.graph = graph_ids.at(partition.graph.value);
    for (auto& member : partition.members)
      member = vertex_ids.at(member.value);
    for (auto& member : partition.graph_members)
      member = graph_ids.at(member.value);
    // Partition names are Graph-scoped. The root authoring index deliberately
    // excludes embedded Graph records, whose local names may repeat in sibling
    // instances without colliding with a root partition.
    auto partition_handle_identity = std::make_shared<std::byte>();
    parent->partition_handle_identities_.reserve(parent->partitions_.size() + 1U);
    parent->partitions_.push_back(std::move(partition));
    parent->partition_handle_identities_.push_back(std::move(partition_handle_identity));
    GraphMutationRollback::checkpoint(*parent);
  }

  for (const ClockStorage& source : child_state->clocks_) {
    ClockStorage copied{
        .record = source.record,
        .capabilities = source.capabilities,
        .factory = source.factory ? source.factory->clone() : nullptr,
        .shared = source.shared,
    };
    copied.record.graph = graph_ids.at(copied.record.graph.value);
    copied.capabilities.graph = copied.record.graph;
    parent->clocks_.push_back(std::move(copied));
    GraphMutationRollback::checkpoint(*parent);
  }

  std::vector<ClockImportAuthoringRecord> clock_imports = child_snapshot->clock_imports;
  for (auto& clock_import : clock_imports) {
    clock_import.graph = graph_ids.at(clock_import.graph.value);
    parent->clock_imports_.push_back(std::move(clock_import));
    GraphMutationRollback::checkpoint(*parent);
  }

  std::vector<EmbeddedClockBindingAuthoringRecord> clock_bindings = child_snapshot->clock_bindings;
  for (auto& binding : clock_bindings) {
    binding.child_graph = graph_ids.at(binding.child_graph.value);
    binding.parent_graph = graph_ids.at(binding.parent_graph.value);
    parent->clock_bindings_.push_back(std::move(binding));
    GraphMutationRollback::checkpoint(*parent);
  }

  std::vector<ClockConversionAuthoringRecord> conversions = child_snapshot->clock_conversions;
  for (auto& conversion : conversions) {
    conversion.graph = graph_ids.at(conversion.graph.value);
    parent->clock_conversions_.push_back(std::move(conversion));
    GraphMutationRollback::checkpoint(*parent);
  }

  std::vector<LifecycleDependencyAuthoringRecord> dependencies =
      child_snapshot->lifecycle_dependencies;
  for (auto& dependency : dependencies) {
    dependency.graph = graph_ids.at(dependency.graph.value);
    dependency.before = vertex_ids.at(dependency.before.value);
    dependency.after = vertex_ids.at(dependency.after.value);
    parent->lifecycle_dependencies_.push_back(std::move(dependency));
    GraphMutationRollback::checkpoint(*parent);
  }

  std::vector<LifecycleCallbackAuthoringRecord> callbacks = child_snapshot->lifecycle_callbacks;
  for (auto& callback : callbacks) {
    callback.owner = vertex_ids.at(callback.owner.value);
    parent->lifecycle_callbacks_.push_back(std::move(callback));
    GraphMutationRollback::checkpoint(*parent);
  }

  std::vector<LifecycleFactAuthoringRecord> facts = child_snapshot->lifecycle_facts;
  for (auto& fact : facts) {
    fact.owner = vertex_ids.at(fact.owner.value);
    parent->lifecycle_facts_.push_back(std::move(fact));
    GraphMutationRollback::checkpoint(*parent);
  }

  std::vector<LifecycleRestartPermissionAuthoringRecord> permissions =
      child_snapshot->lifecycle_restart_permissions;
  for (auto& permission : permissions) {
    if (permission.requester)
      permission.requester = vertex_ids.at(permission.requester->value);
    permission.target = vertex_ids.at(permission.target.value);
    parent->lifecycle_restart_permissions_.push_back(std::move(permission));
    GraphMutationRollback::checkpoint(*parent);
  }

  const GraphInstancePath handle_path = GraphState::make_path({std::string(instance_name)});
  parent->embedded_instance_names_.emplace(instance_name);
  GraphMutationRollback::checkpoint(*parent);
  parent->embedded_graphs_.push_back(EmbeddedGraphStorage{
      .graph = std::move(child),
      .state = child_state,
      .graph_id = embedded_root,
      .path = handle_path,
      .handle_identity = handle_identity,
  });
  GraphMutationRollback::checkpoint(*parent);
  if (child_state->parameter_transaction_) {
    try {
      auto remapped =
          std::make_shared<ParameterTransactionState>(*child_state->parameter_transaction_);
      for (auto& op : remapped->operators) {
        const auto target = vertex_ids.find(op.target.value);
        if (target == vertex_ids.end()) {
          return make_unexpected(
              graph_error(ErrorCode::kInvalidArgument, "GRAPH_PARAMETER_TRANSACTION_INVALID"));
        }
        op.target = target->second;
      }
      for (auto& scope_target : remapped->scope.operators) {
        const auto target = vertex_ids.find(scope_target.value);
        if (target == vertex_ids.end()) {
          return make_unexpected(
              graph_error(ErrorCode::kInvalidArgument, "GRAPH_PARAMETER_TRANSACTION_INVALID"));
        }
        scope_target = target->second;
      }
      if (remapped->scope.graph) {
        const auto target = graph_ids.find(remapped->scope.graph->value);
        if (target == graph_ids.end()) {
          return make_unexpected(
              graph_error(ErrorCode::kInvalidArgument, "GRAPH_PARAMETER_TRANSACTION_INVALID"));
        }
        remapped->scope.graph = target->second;
      }
      parent->parameter_transaction_ = std::move(remapped);
    } catch (const std::bad_alloc&) {
      return make_unexpected(
          graph_error(ErrorCode::kResourceExhausted, "GRAPH_PARAMETER_TRANSACTION_ALLOC_FAILED"));
    }
  }
  parent->mark_mutation();
  EmbeddedGraphRegistration registration{
      .index = embedded_index,
      .handle_identity = std::move(handle_identity),
  };
  rollback.commit();
  graph_log_noexcept([&] {
    HOLOSCAN_LOG_TRACE("Embedded Graph instance '{}' with {} flattened vertices",
                       instance_name,
                       child_snapshot->vertices.size());
  });
  return registration;
}

const Graph* graph_embedded_graph(const std::shared_ptr<GraphState>& state, std::size_t index,
                                  const void* handle_identity) noexcept {
  const std::lock_guard lock(state->mutex_);
  if (index >= state->embedded_graphs_.size() || handle_identity == nullptr ||
      state->embedded_graphs_[index].handle_identity.get() != handle_identity) {
    return nullptr;
  }
  return state->embedded_graphs_[index].graph.get();
}

bool graph_embedded_valid(const std::shared_ptr<GraphState>& state, std::size_t index,
                          const void* handle_identity) noexcept {
  return graph_embedded_graph(state, index, handle_identity) != nullptr;
}

std::optional<GraphInstancePath> graph_embedded_path(const std::shared_ptr<GraphState>& state,
                                                     std::size_t index,
                                                     const void* handle_identity) {
  const std::lock_guard lock(state->mutex_);
  if (index >= state->embedded_graphs_.size() || handle_identity == nullptr ||
      state->embedded_graphs_[index].handle_identity.get() != handle_identity) {
    return std::nullopt;
  }
  return state->embedded_graphs_[index].path;
}

expected<std::vector<PortRecord>, Error> graph_embedded_interface_ports(
    const std::shared_ptr<GraphState>& state, std::size_t index, const void* handle_identity) {
  if (!state) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  if (index >= state->embedded_graphs_.size() || handle_identity == nullptr ||
      state->embedded_graphs_[index].handle_identity.get() != handle_identity) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  }
  const GraphId child_graph = state->embedded_graphs_[index].graph_id;
  std::vector<PortRecord> result;
  for (const auto& entry : state->ports_) {
    if (entry.record.owner_kind != PortOwnerKind::kGraphInterface) {
      continue;
    }
    const auto owner = std::find_if(
        state->vertices_.begin(), state->vertices_.end(), [&](const VertexRecord& vertex) {
          return vertex.id == entry.record.owner;
        });
    if (owner != state->vertices_.end() && owner->graph == child_graph) {
      result.push_back(entry.record);
    }
  }
  return result;
}

expected<PortRecord, Error> graph_find_embedded_interface_port(
    const std::shared_ptr<GraphState>& state, std::size_t index, const void* handle_identity,
    std::string_view name) {
  auto ports = graph_embedded_interface_ports(state, index, handle_identity);
  if (!ports) {
    return make_unexpected(std::move(ports).error());
  }
  const auto port = std::find_if(
      ports->begin(), ports->end(), [&](const PortRecord& record) { return record.name == name; });
  if (port == ports->end()) {
    return make_unexpected(graph_error(ErrorCode::kNotFound, "GRAPH_PORT_NOT_FOUND"));
  }
  return *port;
}

expected<ClockKey, Error> graph_find_embedded_clock_import(const std::shared_ptr<GraphState>& state,
                                                           std::size_t index,
                                                           const void* handle_identity,
                                                           std::string_view name) {
  if (!state) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  if (index >= state->embedded_graphs_.size() || handle_identity == nullptr ||
      state->embedded_graphs_[index].handle_identity.get() != handle_identity) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  }
  const GraphId child_graph = state->embedded_graphs_[index].graph_id;
  const auto found = std::find_if(
      state->clock_imports_.begin(), state->clock_imports_.end(), [&](const auto& candidate) {
        return candidate.graph == child_graph && candidate.key.name() == name;
      });
  if (found == state->clock_imports_.end()) {
    return make_unexpected(graph_error(ErrorCode::kNotFound, "GRAPH_CLOCK_IMPORT_NOT_FOUND"));
  }
  return found->key;
}

}  // namespace holoscan::detail
