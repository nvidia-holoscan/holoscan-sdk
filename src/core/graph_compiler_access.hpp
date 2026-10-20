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
  std::optional<std::size_t> root_operator_index;
  std::shared_ptr<const ParameterInputEvidenceState> parameter_input_evidence;
  std::vector<ExecutableLocalEnumPortRecord> local_enum_ports;
};

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
    const RuntimePortTypeOps* runtime_type_ops{};    ///< Static codec retained outside artifacts.
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

  /// @brief Retain the immutable canonical storage behind one compiler Graph view.
  /// @param view Canonical Graph view published in an execution plan.
  /// @return Shared immutable storage, or null for an empty view.
  /// @note This cold source-private accessor supports compiler handoff validation
  /// and source-tree malformed-plan tests; it grants no public mutation API.
  [[nodiscard]] static std::shared_ptr<const GraphSnapshotData> retain_snapshot(
      const GraphView& view) noexcept {
    return view.snapshot_;
  }

  /// @brief Clone every executable factory and bind it to canonical plan identities.
  /// @param graph Composed source Graph whose process-local factory inventory is captured.
  /// @param canonical Canonical snapshot published through the resulting ExecutionPlan.
  /// @return Deep-owned private inventory safe after the source Graph is destroyed.
  /// @throws RuntimeError when the Graph changes during capture, a factory cannot be resolved, or
  /// a parameter transaction target is not present in the captured operator inventory.
  /// @throws std::bad_alloc when cold factory or identity storage cannot be allocated.
  /// @note The inventory is process-local executable state. It never enters serialized
  /// artifacts, digests, diagnostics, GraphView, or transport records.
  // Factory capture authenticates root and embedded ownership in one cold transaction before any
  // executable inventory is published.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
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
    std::shared_ptr<const ParameterTransactionState> parameter_transaction;
    {
      const std::shared_ptr<GraphState>& state = graph.state_;
      const std::lock_guard lock(state->mutex_);
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
      for (std::size_t index = 0U; index < state->operators_.size(); ++index) {
        const auto& op = state->operators_[index];
        pending_operators.push_back(graph_compiler_access_detail::PendingOperatorFactory{
            op.vertex, op.factory, index, op.parameter_input_evidence, {}});
      }
      for (const auto& mapping : state->embedded_operator_mappings_) {
        pending_operators.push_back(graph_compiler_access_detail::PendingOperatorFactory{
            mapping.vertex, mapping.factory, std::nullopt, mapping.parameter_input_evidence, {}});
      }
      for (auto& pending : pending_operators) {
        for (const auto& port : state->ports_) {
          if (port.record.owner == pending.authored_vertex &&
              port.record.owner_kind == PortOwnerKind::kOperator &&
              port.record.payload_identity.scalar &&
              port.record.payload_identity.scalar->kind == ScalarIdentityKind::kLocalEnum) {
            pending.local_enum_ports.push_back(ExecutableLocalEnumPortRecord{
                port.record.direction, port.record.name, port.local_type_token});
          }
        }
      }
      parameter_transaction = state->parameter_transaction_;
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
    std::vector<std::pair<VertexId, VertexId>> canonical_vertices;
    canonical_vertices.reserve(pending_operators.size());
    result.operators.reserve(pending_operators.size());
    for (graph_compiler_access_detail::PendingOperatorFactory& pending : pending_operators) {
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
      canonical_vertices.emplace_back(pending.authored_vertex, canonical_vertex->id);
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
          .cuda_graph_capture_eligible = canonical_vertex->cuda_graph_capture_eligible,
          .factory = std::move(factory),
          .parameter_input_evidence = pending.parameter_input_evidence,
          .lifecycle_callbacks = std::move(lifecycle_callbacks),
          .root_operator_index = pending.root_operator_index,
          .local_enum_ports = std::move(pending.local_enum_ports),
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
    if (parameter_transaction) {
      auto remapped = std::make_shared<ParameterTransactionState>(*parameter_transaction);
      for (auto& op : remapped->operators) {
        const auto target = std::ranges::find_if(canonical_vertices, [&](const auto& candidate) {
          return candidate.first == op.target;
        });
        if (target == canonical_vertices.end()) {
          throw RuntimeError{Error{ErrorCode::kFailure, "GRAPH_PARAMETER_TRANSACTION_INVALID"}};
        }
        op.target = target->second;
      }
      for (auto& scope_target : remapped->scope.operators) {
        const auto target = std::ranges::find_if(canonical_vertices, [&](const auto& candidate) {
          return candidate.first == scope_target;
        });
        if (target == canonical_vertices.end()) {
          throw RuntimeError{Error{ErrorCode::kFailure, "GRAPH_PARAMETER_TRANSACTION_INVALID"}};
        }
        scope_target = target->second;
      }
      if (remapped->scope.graph) {
        const GraphRecord* const authored = authored_graph(*remapped->scope.graph);
        const GraphRecord* const mapped =
            authored == nullptr ? nullptr : canonical_graph(*authored);
        if (mapped == nullptr) {
          throw RuntimeError{Error{ErrorCode::kFailure, "GRAPH_PARAMETER_TRANSACTION_INVALID"}};
        }
        remapped->scope.graph = mapped->id;
      }
      result.parameter_transaction = std::move(remapped);
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
      const std::lock_guard lock(state->mutex_);
      factories.reserve(state->operators_.size() + state->embedded_operator_mappings_.size());
      for (const auto& op : state->operators_)
        factories.push_back(op.factory);
      for (const auto& mapping : state->embedded_operator_mappings_)
        factories.push_back(mapping.factory);
    }
    for (const auto& factory : factories) {
      if (!factory)
        return make_unexpected(Error{ErrorCode::kFailure, "GRAPH_OPERATOR_FACTORY_INVALID"});
      auto created = factory->create(nullptr);
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
      PlanGenerationId plan_generation, std::span<const ReadinessPlanRecord> readiness,
      Diagnostic* failure = nullptr);

  /// @brief Recreate or resolve every root and embedded clock declaration.
  /// @param graph Graph whose private clock ownership inventory is exercised.
  /// @return Number of independently replayable or explicitly shared clocks.
  [[nodiscard]] static expected<std::size_t, Error> instantiate_clock_factories(
      const Graph& graph) {
    std::vector<std::shared_ptr<ClockFactoryBase>> factories;
    std::size_t shared_count = 0;
    {
      const std::shared_ptr<GraphState>& state = graph.state_;
      const std::lock_guard lock(state->mutex_);
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
      auto created = factory->create(nullptr);
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
    const std::lock_guard lock(state->mutex_);
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
                                 0U);
  }

  /// @brief Inject one allocation-like failure at a later logical mutation checkpoint.
  /// @param graph Graph whose private cold mutation path is instrumented.
  /// @param checkpoints Number of checkpoints to pass before throwing; zero throws at the next.
  /// @note This source-tree-only seam never enters installed headers or production control flow
  /// unless a test explicitly arms it.
  static void fail_mutation_after(Graph& graph, std::size_t checkpoints) {
    const std::shared_ptr<GraphState>& state = graph.state_;
    const std::lock_guard lock(state->mutex_);
    state->mutation_fault_countdown_ = checkpoints;
  }

  /// @brief Inject one allocation failure while enriching a composition failure diagnostic.
  /// @param graph Graph whose private composition failure path is instrumented.
  /// @note This source-tree-only seam proves terminal publication remains allocation-safe.
  static void fail_composition_diagnostic_capture(Graph& graph) {
    const std::shared_ptr<GraphState>& state = graph.state_;
    const std::lock_guard lock(state->mutex_);
    state->fail_composition_diagnostic_capture_ = true;
  }

  /// @brief Report whether positional port and edge IDs match their canonical records.
  /// @param graph Graph whose private cold authoring indexes are inspected.
  /// @return True when the dense-ID invariant and operator indexes are consistent.
  [[nodiscard]] static bool dense_ids_valid(const Graph& graph) noexcept {
    const std::shared_ptr<GraphState>& state = graph.state_;
    const std::lock_guard lock(state->mutex_);
    return state->dense_ids_valid();
  }
};

}  // namespace holoscan::detail
