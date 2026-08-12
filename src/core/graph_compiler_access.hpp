/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>

#include "../runtime/executable_factory_inventory.hpp"
#include "graph_state.hpp"
#include "temporal_contract_internal.hpp"

namespace holoscan::detail {

namespace graph_compiler_access_detail {

struct PendingOperatorFactory {
  VertexId authored_vertex{};
  std::shared_ptr<OperatorFactoryBase> factory;
};

struct AuthoredGraphLookup {
  std::vector<std::string> path;
  std::string name;
  GraphId id{};
};

struct AuthoredVertexLookup {
  GraphId graph{};
  std::string name;
  VertexId id{};
};

[[nodiscard]] inline bool schema_package_less(const SchemaPackagePlanRecord& lhs,
                                              const SchemaPackagePlanRecord& rhs) {
  return std::tie(lhs.schema_id.value,
                  lhs.binary_schema_size,
                  lhs.flatc_build,
                  lhs.ordered_flags,
                  lhs.include_closure_sha256,
                  lhs.generator_format_version,
                  lhs.package_format_version,
                  lhs.runtime_abi.major,
                  lhs.runtime_abi.minor,
                  lhs.runtime_abi.revision) < std::tie(rhs.schema_id.value,
                                                       rhs.binary_schema_size,
                                                       rhs.flatc_build,
                                                       rhs.ordered_flags,
                                                       rhs.include_closure_sha256,
                                                       rhs.generator_format_version,
                                                       rhs.package_format_version,
                                                       rhs.runtime_abi.major,
                                                       rhs.runtime_abi.minor,
                                                       rhs.runtime_abi.revision);
}

[[nodiscard]] inline std::vector<std::string> graph_path(const GraphRecord& record) {
  return record.path ? std::vector<std::string>{record.path->segments().begin(),
                                                record.path->segments().end()}
                     : std::vector<std::string>{};
}

[[nodiscard]] inline std::vector<AuthoredGraphLookup> make_graph_lookup(
    const std::vector<GraphRecord>& graphs) {
  std::vector<AuthoredGraphLookup> result;
  result.reserve(graphs.size());
  for (const GraphRecord& graph : graphs) {
    result.push_back(AuthoredGraphLookup{
        .path = graph_path(graph),
        .name = graph.name,
        .id = graph.id,
    });
  }
  std::ranges::sort(result, [](const auto& lhs, const auto& rhs) {
    return std::tie(lhs.path, lhs.name) < std::tie(rhs.path, rhs.name);
  });
  return result;
}

[[nodiscard]] inline std::vector<AuthoredVertexLookup> make_vertex_lookup(
    const std::vector<VertexRecord>& vertices) {
  std::vector<AuthoredVertexLookup> result;
  result.reserve(vertices.size());
  for (const VertexRecord& vertex : vertices) {
    if (vertex.kind == VertexKind::kOperator) {
      result.push_back(AuthoredVertexLookup{
          .graph = vertex.graph,
          .name = vertex.name,
          .id = vertex.id,
      });
    }
  }
  std::ranges::sort(result, [](const auto& lhs, const auto& rhs) {
    return std::tie(lhs.graph.value, lhs.name) < std::tie(rhs.graph.value, rhs.name);
  });
  return result;
}

[[nodiscard]] inline const AuthoredGraphLookup* find_graph(
    const std::vector<AuthoredGraphLookup>& graphs, const GraphRecord& canonical) {
  const AuthoredGraphLookup key{
      .path = graph_path(canonical),
      .name = canonical.name,
  };
  const auto less = [](const AuthoredGraphLookup& lhs, const AuthoredGraphLookup& rhs) {
    return std::tie(lhs.path, lhs.name) < std::tie(rhs.path, rhs.name);
  };
  const auto found = std::ranges::lower_bound(graphs, key, less);
  return found != graphs.end() && found->path == key.path && found->name == key.name ? &*found
                                                                                     : nullptr;
}

[[nodiscard]] inline const AuthoredVertexLookup* find_vertex(
    const std::vector<AuthoredVertexLookup>& vertices, GraphId graph, std::string_view name) {
  const AuthoredVertexLookup key{
      .graph = graph,
      .name = std::string{name},
  };
  const auto less = [](const AuthoredVertexLookup& lhs, const AuthoredVertexLookup& rhs) {
    return std::tie(lhs.graph.value, lhs.name) < std::tie(rhs.graph.value, rhs.name);
  };
  const auto found = std::ranges::lower_bound(vertices, key, less);
  return found != vertices.end() && found->graph == graph && found->name == name ? &*found
                                                                                 : nullptr;
}

[[nodiscard]] inline ExecutableFactoryInventory make_selected_factory_inventory(
    const std::vector<GraphRecord>& authored_graphs,
    const std::vector<VertexRecord>& authored_vertices,
    std::vector<PendingOperatorFactory> pending_operators, const GraphSnapshotData& canonical,
    std::span<const VertexId> selected_operators) {
  auto graphs = make_graph_lookup(authored_graphs);
  auto vertices = make_vertex_lookup(authored_vertices);
  std::ranges::sort(pending_operators, [](const auto& lhs, const auto& rhs) {
    return lhs.authored_vertex.value < rhs.authored_vertex.value;
  });

  std::vector<VertexId> selected{selected_operators.begin(), selected_operators.end()};
  std::ranges::sort(selected, {}, &VertexId::value);
  if (std::ranges::adjacent_find(selected) != selected.end()) {
    throw RuntimeError{Error{ErrorCode::kFailure, "GRAPH_OPERATOR_FACTORY_INVALID"}};
  }

  ExecutableFactoryInventory result;
  result.operators.reserve(selected.size());
  for (const VertexId selected_id : selected) {
    if (selected_id.value == 0U || selected_id.value > canonical.vertices.size()) {
      throw RuntimeError{Error{ErrorCode::kFailure, "GRAPH_OPERATOR_FACTORY_INVALID"}};
    }
    const VertexRecord& canonical_vertex = canonical.vertices[selected_id.value - 1U];
    if (canonical_vertex.id != selected_id || canonical_vertex.kind != VertexKind::kOperator ||
        canonical_vertex.graph.value == 0U ||
        canonical_vertex.graph.value > canonical.graphs.size()) {
      throw RuntimeError{Error{ErrorCode::kFailure, "GRAPH_OPERATOR_FACTORY_INVALID"}};
    }
    const GraphRecord& canonical_graph = canonical.graphs[canonical_vertex.graph.value - 1U];
    const AuthoredGraphLookup* const authored_graph = find_graph(graphs, canonical_graph);
    if (authored_graph == nullptr) {
      throw RuntimeError{Error{ErrorCode::kFailure, "GRAPH_OPERATOR_FACTORY_UNRESOLVED"}};
    }
    const AuthoredVertexLookup* const authored_vertex =
        find_vertex(vertices, authored_graph->id, canonical_vertex.name);
    if (authored_vertex == nullptr) {
      throw RuntimeError{Error{ErrorCode::kFailure, "GRAPH_OPERATOR_FACTORY_UNRESOLVED"}};
    }
    const auto pending = std::ranges::lower_bound(
        pending_operators, authored_vertex->id.value, {}, [](const PendingOperatorFactory& item) {
          return item.authored_vertex.value;
        });
    if (pending == pending_operators.end() || pending->authored_vertex != authored_vertex->id ||
        !pending->factory) {
      throw RuntimeError{Error{ErrorCode::kFailure, "GRAPH_OPERATOR_FACTORY_UNRESOLVED"}};
    }
    std::shared_ptr<OperatorFactoryBase> factory = pending->factory->clone();
    if (!factory) {
      throw RuntimeError{Error{ErrorCode::kFailure, "GRAPH_OPERATOR_FACTORY_INVALID"}};
    }
    result.operators.push_back(ExecutableOperatorFactoryRecord{
        .vertex = selected_id,
        .operator_id = static_cast<OperatorInstanceId>(result.operators.size() + 1U),
        .factory = std::move(factory),
        .lifecycle_callbacks = {},
    });
  }
  return result;
}

}  // namespace graph_compiler_access_detail

/// @brief Expose immutable Graph composition evidence to the cold compiler and source-tree tests.
///
/// This header is not installed. Returning an owning copy lets a later compiler retain rejected-
/// plan diagnostics without borrowing Graph state, while public authoring operations continue to
/// replay only the exact stable primary token.
class GraphCompilerAccess {
 public:
  /// @brief Preserve one replayed port's compile-only type and identity facts.
  struct TemporalReplayPort {
    PortId id{};                                     ///< Dense canonical plan-local port identity.
    PortDirection direction{PortDirection::kInput};  ///< Declared direction.
    std::string name;                                ///< Owned declaration-local name.
    const void* local_type_token{};                  ///< Process-local validation token only.
  };

  /// @brief Preserve one factory-replayed temporal contract for cold lowering.
  struct TemporalReplay {
    VertexId owner{};                                 ///< Canonical owning Operator vertex.
    OperatorInstanceId operator_id{};                 ///< Dense canonical executable identity.
    std::string operator_name;                        ///< Owned canonical local name.
    TemporalContractSnapshot contract;                ///< Normalized compile-only descriptor.
    std::vector<TemporalReplayPort> ports;            ///< Canonical port/type inventory.
    const ParamTypeDescriptor* parameter_type{};      ///< Process-local aggregate identity.
    const ParamSchemaDescriptor* parameter_schema{};  ///< Process-local declared schema.
    const void* initial_parameter_value{};            ///< Borrowed factory-owned generation zero.
  };

  /// @brief Copy the sole canonical pointer-free compiler handoff for one Graph.
  /// @param graph Fully composed Graph.
  /// @return Deep-owned normalized records with path-derived local identities.
  /// @note Compilation and admission consume this normalized copy. A compiler
  /// may publish the same snapshot through \ref canonical_view so every plan
  /// record and its exposed Graph use one canonical identity space.
  [[nodiscard]] static std::shared_ptr<const GraphSnapshotData> canonical_snapshot(Graph& graph) {
    auto authored = graph_snapshot(graph.state_);
    return std::make_shared<const GraphSnapshotData>(canonicalize_graph_snapshot(*authored));
  }

  /// @brief Create a public immutable view over one canonical compiler snapshot.
  /// @param snapshot Canonical deep-owned snapshot returned by \ref canonical_snapshot.
  /// @return Copyable view sharing the exact canonical record storage.
  /// @note The returned IDs are plan-local canonical IDs, not the authored
  /// snapshot-local IDs returned by \ref Graph::view.
  [[nodiscard]] static GraphView canonical_view(
      std::shared_ptr<const GraphSnapshotData> snapshot) noexcept {
    return GraphView{std::move(snapshot)};
  }

  /// @brief Deep-copy canonical generated-package provenance used by one Graph.
  /// @param graph Composed source Graph whose cold package evidence is captured.
  /// @param canonical Canonical snapshot whose generation must still match.
  /// @return Records ordered by SchemaId and complete evidence with exact duplicates coalesced.
  /// @throws RuntimeError when the Graph changes during capture.
  /// @note Distinct honest evidence for one SchemaId remains as distinct diagnostic records and
  /// never changes compile admission or semantic identity.
  [[nodiscard]] static std::vector<SchemaPackagePlanRecord> schema_packages(
      const Graph& graph, const GraphSnapshotData& canonical) {
    std::vector<SchemaPackagePlanRecord> result;
    {
      const std::shared_ptr<GraphState>& state = graph.state_;
      std::lock_guard lock(state->mutex_);
      if (state->generation_ != canonical.generation) {
        throw RuntimeError{Error{ErrorCode::kIncompatible, "GRAPH_MUTATED_DURING_COMPILE"}};
      }
      result.reserve(state->ports_.size());
      for (const PortStorage& port : state->ports_) {
        if (port.schema_package) {
          result.push_back(*port.schema_package);
        }
      }
    }
    std::ranges::sort(result, graph_compiler_access_detail::schema_package_less);
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
  }

  /// @brief Retain the immutable canonical storage behind one compiler Graph view.
  /// @param view Canonical Graph view published in an execution plan.
  /// @return Shared immutable storage, or null for an empty view.
  /// @note This cold source-private accessor supports compiler handoff validation
  /// and source-tree malformed-plan tests; it grants no public mutation API.
  [[nodiscard]] static std::shared_ptr<const GraphSnapshotData> retain_snapshot(
      const GraphView& view) noexcept {
    return view.snapshot_;
  }

  /// @brief Clone only explicitly selected Operator factories for cold Parameter tooling.
  /// @param graph Composed source Graph whose selected process-local factories are captured.
  /// @param canonical Canonical snapshot used to resolve exact selected identities.
  /// @param selected_operators Non-empty canonical Operator vertex set.
  /// @return Deep-owned selected factories with no clocks or lifecycle callback inventory.
  /// @throws RuntimeError when the Graph changes or a selected factory cannot be resolved.
  /// @throws std::bad_alloc when cold index or factory storage cannot be allocated.
  /// @note Unselected factories are neither retained, cloned, nor schema-queried. This seam is
  /// source-private and adds no runtime dispatch state or lookup.
  [[nodiscard]] static ExecutableFactoryInventory parameter_factories(
      const Graph& graph, const GraphSnapshotData& canonical,
      std::span<const VertexId> selected_operators) {
    if (selected_operators.empty()) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_OPERATOR_SELECTION_EMPTY"}};
    }
    std::vector<GraphRecord> authored_graphs;
    std::vector<VertexRecord> authored_vertices;
    std::vector<graph_compiler_access_detail::PendingOperatorFactory> pending_operators;
    {
      const std::shared_ptr<GraphState>& state = graph.state_;
      std::lock_guard lock(state->mutex_);
      if (state->generation_ != canonical.generation) {
        throw RuntimeError{Error{ErrorCode::kIncompatible, "GRAPH_MUTATED_DURING_COMPILE"}};
      }
      authored_graphs.reserve(1U + state->embedded_graph_records_.size());
      authored_graphs.push_back(state->graph_record_);
      authored_graphs.insert(authored_graphs.end(),
                             state->embedded_graph_records_.begin(),
                             state->embedded_graph_records_.end());
      authored_vertices = state->vertices_;
      pending_operators.reserve(state->operators_.size() +
                                state->embedded_operator_mappings_.size());
      for (const auto& op : state->operators_) {
        pending_operators.push_back(
            graph_compiler_access_detail::PendingOperatorFactory{op.vertex, op.factory});
      }
      for (const auto& mapping : state->embedded_operator_mappings_) {
        pending_operators.push_back(
            graph_compiler_access_detail::PendingOperatorFactory{mapping.vertex, mapping.factory});
      }
    }
    return graph_compiler_access_detail::make_selected_factory_inventory(
        authored_graphs,
        authored_vertices,
        std::move(pending_operators),
        canonical,
        selected_operators);
  }

  /// @brief Clone every executable factory and bind it to canonical plan identities.
  /// @param graph Composed source Graph whose process-local factory inventory is captured.
  /// @param canonical Canonical snapshot published through the resulting ExecutionPlan.
  /// @return Deep-owned private inventory safe after the source Graph is destroyed.
  /// @throws RuntimeError when the Graph changes during capture or a factory cannot be resolved.
  /// @throws std::bad_alloc when cold factory or identity storage cannot be allocated.
  /// @note The inventory is process-local executable state. It never enters serialized
  /// artifacts, digests, diagnostics, GraphView, or transport records.
  [[nodiscard]] static ExecutableFactoryInventory executable_factories(
      const Graph& graph, const GraphSnapshotData& canonical) {
    struct PendingClock {
      ClockDeclarationRecord declaration;
      std::shared_ptr<ClockFactoryBase> factory;
      std::shared_ptr<Clock> shared;
    };

    std::vector<GraphRecord> authored_graphs;
    std::vector<VertexRecord> authored_vertices;
    std::vector<graph_compiler_access_detail::PendingOperatorFactory> pending_operators;
    std::vector<PendingClock> pending_clocks;
    {
      const std::shared_ptr<GraphState>& state = graph.state_;
      std::lock_guard lock(state->mutex_);
      if (state->generation_ != canonical.generation) {
        throw RuntimeError{Error{ErrorCode::kIncompatible, "GRAPH_MUTATED_DURING_COMPILE"}};
      }
      authored_graphs.reserve(1U + state->embedded_graph_records_.size());
      authored_graphs.push_back(state->graph_record_);
      authored_graphs.insert(authored_graphs.end(),
                             state->embedded_graph_records_.begin(),
                             state->embedded_graph_records_.end());
      authored_vertices = state->vertices_;
      pending_operators.reserve(state->operators_.size() +
                                state->embedded_operator_mappings_.size());
      for (const auto& op : state->operators_) {
        pending_operators.push_back(
            graph_compiler_access_detail::PendingOperatorFactory{op.vertex, op.factory});
      }
      for (const auto& mapping : state->embedded_operator_mappings_) {
        pending_operators.push_back(
            graph_compiler_access_detail::PendingOperatorFactory{mapping.vertex, mapping.factory});
      }
      pending_clocks.reserve(state->clocks_.size());
      for (const auto& clock : state->clocks_) {
        pending_clocks.push_back(PendingClock{clock.record, clock.factory, clock.shared});
      }
    }

    const auto graph_path = [](const GraphRecord& record) {
      return record.path ? std::vector<std::string>{record.path->segments().begin(),
                                                    record.path->segments().end()}
                         : std::vector<std::string>{};
    };
    const auto authored_graph = [&](GraphId id) -> const GraphRecord* {
      const auto found = std::ranges::find(authored_graphs, id, &GraphRecord::id);
      return found == authored_graphs.end() ? nullptr : &*found;
    };
    const auto canonical_graph = [&](const GraphRecord& authored) -> const GraphRecord* {
      const auto path = graph_path(authored);
      const auto found = std::ranges::find_if(canonical.graphs, [&](const GraphRecord& candidate) {
        return candidate.name == authored.name && graph_path(candidate) == path;
      });
      return found == canonical.graphs.end() ? nullptr : &*found;
    };

    ExecutableFactoryInventory result;
    result.operators.reserve(pending_operators.size());
    for (const graph_compiler_access_detail::PendingOperatorFactory& pending : pending_operators) {
      const auto vertex =
          std::ranges::find(authored_vertices, pending.authored_vertex, &VertexRecord::id);
      if (vertex == authored_vertices.end()) {
        throw RuntimeError{Error{ErrorCode::kFailure, "GRAPH_OPERATOR_FACTORY_INVALID"}};
      }
      const GraphRecord* const authored = authored_graph(vertex->graph);
      const GraphRecord* const graph_record =
          authored == nullptr ? nullptr : canonical_graph(*authored);
      if (graph_record == nullptr) {
        throw RuntimeError{Error{ErrorCode::kFailure, "GRAPH_OPERATOR_FACTORY_UNRESOLVED"}};
      }
      const auto canonical_vertex =
          std::ranges::find_if(canonical.vertices, [&](const VertexRecord& candidate) {
            return candidate.graph == graph_record->id && candidate.kind == VertexKind::kOperator &&
                   candidate.name == vertex->name;
          });
      if (canonical_vertex == canonical.vertices.end()) {
        throw RuntimeError{Error{ErrorCode::kFailure, "GRAPH_OPERATOR_FACTORY_UNRESOLVED"}};
      }
      if (!pending.factory) {
        throw RuntimeError{Error{ErrorCode::kFailure, "GRAPH_OPERATOR_FACTORY_INVALID"}};
      }
      std::shared_ptr<OperatorFactoryBase> factory = pending.factory->clone();
      if (!factory) {
        throw RuntimeError{Error{ErrorCode::kFailure, "GRAPH_OPERATOR_FACTORY_INVALID"}};
      }
      std::vector<ExecutableLifecycleCallbackRecord> lifecycle_callbacks;
      for (const LifecycleCallbackAuthoringRecord& callback : canonical.lifecycle_callbacks) {
        if (callback.owner != canonical_vertex->id) {
          continue;
        }
        ExecutableLifecycleCallbackOrigin origin{};
        switch (callback.origin) {
          case LifecycleCallbackOrigin::kAdvancedMember:
            origin = ExecutableLifecycleCallbackOrigin::kAdvancedMember;
            break;
          case LifecycleCallbackOrigin::kLegacyStart:
            origin = ExecutableLifecycleCallbackOrigin::kLegacyStart;
            break;
          case LifecycleCallbackOrigin::kLegacyStop:
            origin = ExecutableLifecycleCallbackOrigin::kLegacyStop;
            break;
        }
        lifecycle_callbacks.push_back(ExecutableLifecycleCallbackRecord{
            .stage = callback.stage,
            .required = callback.required,
            .method = callback.method,
            .origin = origin,
        });
      }
      result.operators.push_back(ExecutableOperatorFactoryRecord{
          .vertex = canonical_vertex->id,
          .operator_id = 0U,
          .factory = std::move(factory),
          .lifecycle_callbacks = std::move(lifecycle_callbacks),
      });
    }
    std::ranges::sort(result.operators, [](const auto& lhs, const auto& rhs) {
      return lhs.vertex.value < rhs.vertex.value;
    });
    for (std::size_t index = 0; index < result.operators.size(); ++index) {
      if (index >= std::numeric_limits<OperatorInstanceId>::max()) {
        throw RuntimeError{Error{ErrorCode::kResourceExhausted, "GRAPH_OPERATOR_ID_EXHAUSTED"}};
      }
      result.operators[index].operator_id = static_cast<OperatorInstanceId>(index + 1U);
    }

    result.clocks.reserve(pending_clocks.size());
    for (const PendingClock& pending : pending_clocks) {
      const GraphRecord* const authored = authored_graph(pending.declaration.graph);
      const GraphRecord* const graph_record =
          authored == nullptr ? nullptr : canonical_graph(*authored);
      if (graph_record == nullptr || (pending.factory == nullptr) == (pending.shared == nullptr)) {
        throw RuntimeError{Error{ErrorCode::kFailure, "GRAPH_CLOCK_FACTORY_INVALID"}};
      }
      std::shared_ptr<ClockFactoryBase> factory =
          pending.factory == nullptr ? nullptr : pending.factory->clone();
      if (pending.factory != nullptr && factory == nullptr) {
        throw RuntimeError{Error{ErrorCode::kFailure, "GRAPH_CLOCK_FACTORY_INVALID"}};
      }
      result.clocks.push_back(ExecutableClockFactoryRecord{
          .graph = graph_record->id,
          .key = pending.declaration.key,
          .factory = std::move(factory),
          .shared = pending.shared,
      });
    }
    std::ranges::sort(result.clocks, [](const auto& lhs, const auto& rhs) {
      return std::tuple{lhs.graph.value, lhs.key.name()} <
             std::tuple{rhs.graph.value, rhs.key.name()};
    });
    return result;
  }

  /// @brief Recreate every root and embedded Operator from retained factories.
  /// @param graph Graph whose private factory inventory is exercised.
  /// @return Number of independently recreated Operator instances.
  [[nodiscard]] static expected<std::size_t, Error> instantiate_operator_factories(
      const Graph& graph) {
    std::vector<std::shared_ptr<OperatorFactoryBase>> factories;
    {
      const std::shared_ptr<GraphState>& state = graph.state_;
      std::lock_guard lock(state->mutex_);
      factories.reserve(state->operators_.size() + state->embedded_operator_mappings_.size());
      for (const auto& op : state->operators_)
        factories.push_back(op.factory);
      for (const auto& mapping : state->embedded_operator_mappings_)
        factories.push_back(mapping.factory);
    }
    for (const auto& factory : factories) {
      if (!factory)
        return make_unexpected(Error{ErrorCode::kFailure, "GRAPH_OPERATOR_FACTORY_INVALID"});
      auto created = factory->create();
      if (!created)
        return make_unexpected(std::move(created.error()));
      if (!created.value())
        return make_unexpected(Error{ErrorCode::kFailure, "GRAPH_OPERATOR_FACTORY_INVALID"});
    }
    return factories.size();
  }

  /// @brief Replay and normalize every Operator temporal contract.
  /// @param factories Frozen plan-owned executable factories and identity map.
  /// @param snapshot Canonical pointer-free snapshot used by the destination plan.
  /// @param plan_generation Nonzero provenance bound into readiness references.
  /// @param readiness Already resolved logical readiness catalog.
  /// @return Canonical Operator-order normalized contracts and compile-only type facts.
  /// @note Callable and payload type tokens remain process-local validation
  /// evidence and never enter the returned public plan records or serializers.
  [[nodiscard]] static expected<std::vector<TemporalReplay>, Error> replay_temporal_contracts(
      const ExecutableFactoryInventory& factories, const GraphSnapshotData& snapshot,
      PlanGenerationId plan_generation, std::span<const ReadinessPlanRecord> readiness);

  /// @brief Recreate or resolve every root and embedded clock declaration.
  /// @param graph Graph whose private clock ownership inventory is exercised.
  /// @return Number of independently replayable or explicitly shared clocks.
  [[nodiscard]] static expected<std::size_t, Error> instantiate_clock_factories(
      const Graph& graph) {
    std::vector<std::shared_ptr<ClockFactoryBase>> factories;
    std::size_t shared_count = 0;
    {
      const std::shared_ptr<GraphState>& state = graph.state_;
      std::lock_guard lock(state->mutex_);
      factories.reserve(state->clocks_.size());
      for (const auto& clock : state->clocks_) {
        if (clock.factory)
          factories.push_back(clock.factory);
        else if (clock.shared)
          ++shared_count;
        else
          return make_unexpected(Error{ErrorCode::kFailure, "GRAPH_CLOCK_FACTORY_INVALID"});
      }
    }
    for (const auto& factory : factories) {
      auto created = factory->create();
      if (!created)
        return make_unexpected(std::move(created.error()));
      if (!created.value())
        return make_unexpected(Error{ErrorCode::kFailure, "GRAPH_CLOCK_FACTORY_INVALID"});
    }
    return factories.size() + shared_count;
  }

  /// @brief Copy the terminal composition diagnostic when composition has failed.
  /// @param graph Graph whose private composition state is inspected.
  /// @return Owning primary/cause/context record, or no value before failure.
  [[nodiscard]] static std::optional<GraphCompositionDiagnostic> composition_diagnostic(
      const Graph& graph) {
    const std::shared_ptr<GraphState>& state = graph.state_;
    std::lock_guard lock(state->mutex_);
    return state->composition_diagnostic_;
  }

  /// @brief Register an import with a caller-supplied pointer-free payload identity.
  /// @tparam T Valid local port type used only to construct the declaration object.
  /// @param graph Graph whose defensive record validation is exercised.
  /// @param payload_identity Untrusted identity record injected before publication.
  /// @return Registered port or the defensive Graph validation error.
  /// @note This source-tree-only seam lets tests exercise malformed erased records without
  /// weakening compile-time validation on the public typed authoring APIs.
  template <PublishedPayload T>
  [[nodiscard]] static expected<GraphPortId, Error> register_import_with_payload_identity(
      Graph& graph, PayloadIdentityRecord payload_identity) {
    auto port = std::make_unique<Output<T>>("external");
    return graph_register_import(graph.state_,
                                 "holomq://test/malformed-pod",
                                 {},
                                 std::unique_ptr<OutputBase>{std::move(port)},
                                 std::move(payload_identity),
                                 std::nullopt,
                                 0U);
  }

  /// @brief Inject one allocation-like failure at a later logical mutation checkpoint.
  /// @param graph Graph whose private cold mutation path is instrumented.
  /// @param checkpoints Number of checkpoints to pass before throwing; zero throws at the next.
  /// @note This source-tree-only seam never enters installed headers or production control flow
  /// unless a test explicitly arms it.
  static void fail_mutation_after(Graph& graph, std::size_t checkpoints) {
    const std::shared_ptr<GraphState>& state = graph.state_;
    std::lock_guard lock(state->mutex_);
    state->mutation_fault_countdown_ = checkpoints;
  }

  /// @brief Inject one allocation failure while enriching a composition failure diagnostic.
  /// @param graph Graph whose private composition failure path is instrumented.
  /// @note This source-tree-only seam proves terminal publication remains allocation-safe.
  static void fail_composition_diagnostic_capture(Graph& graph) {
    const std::shared_ptr<GraphState>& state = graph.state_;
    std::lock_guard lock(state->mutex_);
    state->fail_composition_diagnostic_capture_ = true;
  }

  /// @brief Report whether positional port and edge IDs match their canonical records.
  /// @param graph Graph whose private cold authoring indexes are inspected.
  /// @return True when the dense-ID invariant and operator indexes are consistent.
  [[nodiscard]] static bool dense_ids_valid(const Graph& graph) noexcept {
    const std::shared_ptr<GraphState>& state = graph.state_;
    std::lock_guard lock(state->mutex_);
    return state->dense_ids_valid();
  }
};

}  // namespace holoscan::detail
