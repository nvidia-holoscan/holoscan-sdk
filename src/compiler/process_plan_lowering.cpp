/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "process_plan_lowering.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "core/diagnostic_registry.hpp"
#include "detail/canonical_utils.hpp"
#include "snapshot_lookup.hpp"

namespace holoscan::detail {
namespace {

constexpr std::string_view kProcessRejected = "PROCESS_PLACEMENT_REJECTED";
constexpr std::string_view kDefaultLaunchGroup = "holoscan.default_process";

[[nodiscard]] bool process_isolation_valid(ProcessIsolation isolation) noexcept {
  switch (isolation) {
    case ProcessIsolation::kSameProcess:
    case ProcessIsolation::kSeparateProcess:
    case ProcessIsolation::kExternalProcess:
      return true;
  }
  return false;
}

[[nodiscard]] std::string region_identity(const GraphSnapshotData& snapshot,
                                          const RegionRecord& region) {
  const GraphRecord* const graph = find_graph(snapshot, region.graph);
  std::string result = graph == nullptr ? std::string{"invalid"} : canonical_graph_path(*graph);
  result.push_back(':');
  result += region.name;
  return result;
}

[[nodiscard]] std::vector<VertexId> vertices_for_graph(const GraphSnapshotData& snapshot,
                                                       GraphId graph) {
  std::vector<VertexId> result;
  for (const auto& vertex : snapshot.vertices) {
    if (is_descendant_or_self(snapshot, vertex.graph, graph)) {
      result.push_back(vertex.id);
    }
  }
  return result;
}

[[nodiscard]] std::vector<VertexId> vertices_for_region(const GraphSnapshotData& snapshot,
                                                        const RegionRecord& region) {
  std::vector<VertexId> result = region.members;
  for (const GraphId graph : region.graph_members) {
    auto graph_vertices = vertices_for_graph(snapshot, graph);
    result.insert(result.end(), graph_vertices.begin(), graph_vertices.end());
  }
  std::sort(result.begin(), result.end(), [](VertexId lhs, VertexId rhs) {
    return lhs.value < rhs.value;
  });
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

[[nodiscard]] std::size_t selector_field_count(const GraphObjectSelectorRecord& selector) noexcept {
  return static_cast<std::size_t>(!selector.graph_path.empty()) +
         static_cast<std::size_t>(!selector.region_id.empty()) +
         static_cast<std::size_t>(!selector.external_topic.empty());
}

[[nodiscard]] auto selector_order(const GraphObjectSelectorRecord& selector) noexcept {
  return std::tie(selector.graph_path, selector.region_id, selector.external_topic);
}

[[nodiscard]] ProcessSelectorPlanRecord resolve_graph_selector(const GraphSnapshotData& snapshot,
                                                               ProcessRoleId role,
                                                               std::uint32_t ordinal,
                                                               std::string value) {
  ProcessSelectorPlanRecord result{
      .process_role = role,
      .ordinal = ordinal,
      .kind = ProcessSelectorKind::kGraphPath,
      .authored_value = std::move(value),
  };
  if (!bounded_identity_valid(result.authored_value, 1024U)) {
    result.reason = diagnostic_id(kProcessRejected);
    return result;
  }

  std::vector<const GraphRecord*> graphs;
  for (const auto& graph : snapshot.graphs) {
    const std::string canonical = canonical_graph_path(graph);
    if (result.authored_value == canonical ||
        (!graph.parent && !graph.name.empty() && result.authored_value == graph.name)) {
      graphs.push_back(&graph);
    }
  }
  std::vector<const VertexRecord*> vertices;
  for (const auto& vertex : snapshot.vertices) {
    if (result.authored_value == canonical_vertex_path(snapshot, vertex)) {
      vertices.push_back(&vertex);
    }
  }
  if (graphs.size() + vertices.size() != 1U) {
    result.reason = diagnostic_id(kProcessRejected);
    return result;
  }
  if (!graphs.empty()) {
    result.graph = graphs.front()->id;
    result.canonical_value = "graph:" + canonical_graph_path(*graphs.front());
    result.members = vertices_for_graph(snapshot, graphs.front()->id);
  } else {
    result.vertex = vertices.front()->id;
    result.canonical_value = "vertex:" + canonical_vertex_path(snapshot, *vertices.front());
    result.members.push_back(vertices.front()->id);
  }
  if (result.members.empty()) {
    result.reason = diagnostic_id(kProcessRejected);
    return result;
  }
  result.disposition = PlanDisposition::kAccepted;
  return result;
}

[[nodiscard]] ProcessSelectorPlanRecord resolve_region_selector(const GraphSnapshotData& snapshot,
                                                                ProcessRoleId role,
                                                                std::uint32_t ordinal,
                                                                std::string value) {
  ProcessSelectorPlanRecord result{
      .process_role = role,
      .ordinal = ordinal,
      .kind = ProcessSelectorKind::kRegion,
      .authored_value = std::move(value),
  };
  if (!bounded_identity_valid(result.authored_value, 1024U)) {
    result.reason = diagnostic_id(kProcessRejected);
    return result;
  }
  std::vector<const RegionRecord*> matches;
  for (const auto& region : snapshot.regions) {
    if (result.authored_value == region.name ||
        result.authored_value == region_identity(snapshot, region)) {
      matches.push_back(&region);
    }
  }
  if (matches.size() != 1U) {
    result.reason = diagnostic_id(kProcessRejected);
    return result;
  }
  result.region = matches.front()->id;
  result.canonical_value = "region:" + region_identity(snapshot, *matches.front());
  result.members = vertices_for_region(snapshot, *matches.front());
  if (result.members.empty()) {
    result.reason = diagnostic_id(kProcessRejected);
    return result;
  }
  result.disposition = PlanDisposition::kAccepted;
  return result;
}

[[nodiscard]] ProcessSelectorPlanRecord resolve_external_selector(const GraphSnapshotData& snapshot,
                                                                  ProcessRoleId role,
                                                                  std::uint32_t ordinal,
                                                                  std::string value) {
  ProcessSelectorPlanRecord result{
      .process_role = role,
      .ordinal = ordinal,
      .kind = ProcessSelectorKind::kExternalTopic,
      .authored_value = std::move(value),
  };
  if (!bounded_identity_valid(result.authored_value, 1024U)) {
    result.reason = diagnostic_id(kProcessRejected);
    return result;
  }
  std::vector<const ExternalTopicRecord*> matches;
  for (const auto& topic : snapshot.external_topics) {
    if (result.authored_value == topic.uri ||
        (topic.endpoint_name && result.authored_value == *topic.endpoint_name)) {
      matches.push_back(&topic);
    }
  }
  if (matches.size() != 1U) {
    result.reason = diagnostic_id(kProcessRejected);
    return result;
  }
  result.external_topic = matches.front()->id;
  result.vertex = matches.front()->vertex;
  result.canonical_value = "external:" + matches.front()->uri;
  result.members.push_back(matches.front()->vertex);
  result.disposition = PlanDisposition::kAccepted;
  return result;
}

[[nodiscard]] ProcessSelectorPlanRecord resolve_selector(
    const GraphSnapshotData& snapshot, ProcessRoleId role, std::uint32_t ordinal,
    const GraphObjectSelectorRecord& selector) {
  if (selector_field_count(selector) != 1U) {
    ProcessSelectorPlanRecord result{
        .process_role = role,
        .ordinal = ordinal,
        .authored_value = selector.graph_path + selector.region_id + selector.external_topic,
        .reason = diagnostic_id(kProcessRejected),
    };
    return result;
  }
  if (!selector.graph_path.empty()) {
    return resolve_graph_selector(snapshot, role, ordinal, selector.graph_path);
  }
  if (!selector.region_id.empty()) {
    return resolve_region_selector(snapshot, role, ordinal, selector.region_id);
  }
  return resolve_external_selector(snapshot, role, ordinal, selector.external_topic);
}

[[nodiscard]] bool external_vertex(const VertexRecord* vertex) noexcept {
  return vertex != nullptr &&
         (vertex->kind == VertexKind::kImportedTopic || vertex->kind == VertexKind::kExportedTopic);
}

void reject_role(ProcessRolePlanRecord& role) noexcept {
  role.accepted_isolation.reset();
  role.accepted_launch_group.clear();
  role.runtime_seal_required = false;
  role.disposition = PlanDisposition::kRejected;
  role.reason = diagnostic_id(kProcessRejected);
}

using ProcessGroupKey = std::tuple<ProcessIsolation, std::string, std::uint64_t>;

[[nodiscard]] ProcessGroupKey process_group_key(const ProcessRolePlanRecord& role,
                                                VertexId member) {
  return ProcessGroupKey{
      role.requested_isolation,
      role.accepted_launch_group,
      role.requested_isolation == ProcessIsolation::kSeparateProcess ? member.value : 0U};
}

}  // namespace

// Keep process identity resolution and locality publication in one deterministic pass.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
ProcessPlanLoweringResult lower_process_plan(const GraphSnapshotData& snapshot,
                                             const ProcessPlacementBinding& placement) {
  ProcessPlanLoweringResult result;
  result.process_group_by_vertex.resize(snapshot.vertices.size(), 1U);
  result.locality_by_edge.resize(snapshot.edges.size(), ChannelLocalityKind::kProcessLocal);

  std::vector<ProcessRoleBinding> authored{placement.process_roles().begin(),
                                           placement.process_roles().end()};
  for (auto& binding : authored) {
    std::sort(
        binding.placement.members.begin(),
        binding.placement.members.end(),
        [](const auto& lhs, const auto& rhs) { return selector_order(lhs) < selector_order(rhs); });
  }
  std::sort(authored.begin(), authored.end(), [](const auto& lhs, const auto& rhs) {
    const auto lhs_request = std::tuple{lhs.role,
                                        lhs.placement.isolation,
                                        lhs.placement.launch_group,
                                        lhs.placement.require_locality_probe_success};
    const auto rhs_request = std::tuple{rhs.role,
                                        rhs.placement.isolation,
                                        rhs.placement.launch_group,
                                        rhs.placement.require_locality_probe_success};
    if (lhs_request != rhs_request) {
      return lhs_request < rhs_request;
    }
    return std::lexicographical_compare(lhs.placement.members.begin(),
                                        lhs.placement.members.end(),
                                        rhs.placement.members.begin(),
                                        rhs.placement.members.end(),
                                        [](const auto& lhs_selector, const auto& rhs_selector) {
                                          return selector_order(lhs_selector) <
                                                 selector_order(rhs_selector);
                                        });
  });

  result.roles.reserve(authored.size());
  std::size_t role_index = 0U;
  while (role_index < authored.size()) {
    std::size_t duplicate_end = role_index + 1U;
    while (duplicate_end < authored.size() &&
           authored[duplicate_end].role == authored[role_index].role) {
      ++duplicate_end;
    }
    const bool duplicate_role = duplicate_end - role_index > 1U;
    for (; role_index < duplicate_end; ++role_index) {
      const auto& binding = authored[role_index];
      ProcessRolePlanRecord role{
          .id = ProcessRoleId{static_cast<std::uint32_t>(result.roles.size() + 1U)},
          .role = binding.role,
          .requested_isolation = binding.placement.isolation,
          .requested_launch_group = binding.placement.launch_group,
          .require_locality_probe_success = binding.placement.require_locality_probe_success,
      };
      bool valid = !duplicate_role && bounded_identity_valid(role.role, 1024U) &&
                   bounded_identity_valid(role.requested_launch_group, 1024U) &&
                   process_isolation_valid(role.requested_isolation) &&
                   !binding.placement.members.empty();
      std::set<std::uint64_t> selected_members;
      std::optional<std::tuple<std::string, std::string, std::string>> previous_selector;
      for (std::size_t selector_index = 0U; selector_index < binding.placement.members.size();
           ++selector_index) {
        const auto& selector = binding.placement.members[selector_index];
        const auto key =
            std::tuple{selector.graph_path, selector.region_id, selector.external_topic};
        const bool duplicate_selector = previous_selector && *previous_selector == key;
        previous_selector = key;
        auto resolved = resolve_selector(
            snapshot, role.id, static_cast<std::uint32_t>(selector_index), selector);
        if (duplicate_selector) {
          resolved.disposition = PlanDisposition::kRejected;
          resolved.reason = diagnostic_id(kProcessRejected);
        }
        valid = valid && resolved.disposition == PlanDisposition::kAccepted;
        for (const VertexId member : resolved.members) {
          if (!selected_members.insert(member.value).second) {
            valid = false;
          }
        }
        result.selectors.push_back(std::move(resolved));
      }
      for (const std::uint64_t member : selected_members) {
        role.members.push_back(VertexId{member});
      }
      if (role.requested_isolation == ProcessIsolation::kExternalProcess &&
          std::any_of(role.members.begin(), role.members.end(), [&](VertexId member) {
            return !external_vertex(find_vertex(snapshot, member));
          })) {
        valid = false;
      }
      if (!valid || role.members.empty()) {
        reject_role(role);
        result.error_tokens.emplace_back(kProcessRejected);
      } else {
        role.accepted_isolation = role.requested_isolation;
        role.accepted_launch_group = role.requested_launch_group;
        role.runtime_seal_required = role.require_locality_probe_success;
        role.disposition = PlanDisposition::kAccepted;
      }
      result.roles.push_back(std::move(role));
    }
  }

  std::map<std::uint64_t, std::vector<std::size_t>> roles_by_member;
  std::map<std::string, std::vector<std::size_t>> roles_by_launch_group;
  for (std::size_t index = 0U; index < result.roles.size(); ++index) {
    const auto& role = result.roles[index];
    if (role.disposition != PlanDisposition::kAccepted) {
      continue;
    }
    roles_by_launch_group[role.accepted_launch_group].push_back(index);
    for (const VertexId member : role.members) {
      roles_by_member[member.value].push_back(index);
    }
  }
  for (const auto& [unused, indexes] : roles_by_member) {
    (void)unused;
    if (indexes.size() > 1U) {
      for (const std::size_t index : indexes) {
        reject_role(result.roles[index]);
      }
      result.error_tokens.emplace_back(kProcessRejected);
    }
  }
  for (const auto& [unused, indexes] : roles_by_launch_group) {
    (void)unused;
    if (indexes.empty()) {
      continue;
    }
    const ProcessIsolation expected = result.roles[indexes.front()].requested_isolation;
    if (std::any_of(indexes.begin(), indexes.end(), [&](std::size_t index) {
          return result.roles[index].requested_isolation != expected;
        })) {
      for (const std::size_t index : indexes) {
        reject_role(result.roles[index]);
      }
      result.error_tokens.emplace_back(kProcessRejected);
    }
  }

  std::set<ProcessGroupKey> group_keys;
  for (const auto& role : result.roles) {
    if (role.disposition != PlanDisposition::kAccepted) {
      continue;
    }
    for (const VertexId member : role.members) {
      group_keys.insert(process_group_key(role, member));
    }
  }
  std::map<ProcessGroupKey, ProcessGroupId> group_ids;
  std::uint32_t next_group = 2U;
  for (const auto& key : group_keys) {
    group_ids.emplace(key, ProcessGroupId{next_group++});
  }

  std::vector<std::optional<ProcessRoleId>> role_by_vertex(snapshot.vertices.size());
  std::vector<ProcessIsolation> isolation_by_vertex(snapshot.vertices.size(),
                                                    ProcessIsolation::kSameProcess);
  std::vector<std::string> launch_group_by_vertex(snapshot.vertices.size(),
                                                  std::string{kDefaultLaunchGroup});
  std::vector<bool> seal_by_vertex(snapshot.vertices.size(), false);
  for (const auto& role : result.roles) {
    if (role.disposition != PlanDisposition::kAccepted) {
      continue;
    }
    for (const VertexId member : role.members) {
      const std::size_t index = static_cast<std::size_t>(member.value - 1U);
      result.process_group_by_vertex[index] = group_ids.at(process_group_key(role, member)).value;
      role_by_vertex[index] = role.id;
      isolation_by_vertex[index] = role.requested_isolation;
      launch_group_by_vertex[index] = role.accepted_launch_group;
      seal_by_vertex[index] = role.runtime_seal_required;
    }
  }

  result.vertices.reserve(snapshot.vertices.size());
  for (const auto& vertex : snapshot.vertices) {
    const std::size_t index = static_cast<std::size_t>(vertex.id.value - 1U);
    result.vertices.push_back(VertexProcessPlacementPlanRecord{
        .vertex = vertex.id,
        .process_group = ProcessGroupId{result.process_group_by_vertex[index]},
        .process_role = role_by_vertex[index],
        .isolation = isolation_by_vertex[index],
        .launch_group = launch_group_by_vertex[index],
        .runtime_seal_required = seal_by_vertex[index],
    });
  }

  for (const auto& edge : snapshot.edges) {
    const PortRecord* const source =
        find_port(snapshot, resolve_interface_port(snapshot, edge.source));
    const PortRecord* const target =
        find_port(snapshot, resolve_interface_port(snapshot, edge.target));
    if (source == nullptr || target == nullptr || !source->owner || !target->owner ||
        source->owner.value > result.process_group_by_vertex.size() ||
        target->owner.value > result.process_group_by_vertex.size()) {
      result.error_tokens.emplace_back(kProcessRejected);
      continue;
    }
    const auto source_group =
        result.process_group_by_vertex[static_cast<std::size_t>(source->owner.value - 1U)];
    const auto target_group =
        result.process_group_by_vertex[static_cast<std::size_t>(target->owner.value - 1U)];
    result.locality_by_edge[static_cast<std::size_t>(edge.id.value - 1U)] =
        source_group == target_group ? ChannelLocalityKind::kProcessLocal
                                     : ChannelLocalityKind::kCrossProcess;
  }
  canonicalize_tokens(result.error_tokens);
  return result;
}

void apply_process_edge_locality(std::span<const ChannelLocalityKind> locality_by_edge,
                                 std::span<PayloadEdgePlanRecord> payload_edges,
                                 std::span<ChannelLocalityRecord> channel_locality) noexcept {
  for (auto& edge : payload_edges) {
    if (!edge.edge || edge.edge.value > locality_by_edge.size() ||
        edge.locality != ChannelLocalityKind::kProcessLocal) {
      continue;
    }
    edge.locality = locality_by_edge[static_cast<std::size_t>(edge.edge.value - 1U)];
  }
  for (auto& edge : channel_locality) {
    if (!edge.edge || edge.edge.value > locality_by_edge.size() ||
        edge.accepted != ChannelLocalityKind::kProcessLocal) {
      continue;
    }
    const auto locality = locality_by_edge[static_cast<std::size_t>(edge.edge.value - 1U)];
    edge.requested = locality;
    edge.accepted = locality;
    edge.requires_runtime_attach = locality == ChannelLocalityKind::kCrossProcess;
  }
}

}  // namespace holoscan::detail
