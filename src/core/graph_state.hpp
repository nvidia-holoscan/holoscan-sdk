/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <holoscan/core/graph.hpp>

#include "graph_snapshot.hpp"
#include "parameter_input_internal.hpp"

namespace holoscan::detail {

class GraphMutationRollback;

enum class CompositionState : std::uint8_t {
  kUncomposed,
  kComposing,
  kComposed,
  kFailed,
};

/// @brief Own the complete terminal composition diagnostic retained for compiler consumption.
///
/// Public Graph authoring operations replay only \ref primary. The private compiler/test seam
/// consumes the optional typed \ref cause or bounded non-normative \ref context when it builds a
/// rejected-plan diagnostic in a later component.
struct GraphCompositionDiagnostic {
  Error primary;
  std::optional<Error> cause;
  std::string context;
};

struct PortStorage {
  PortRecord record;
  std::shared_ptr<const void> handle_identity;
  const void* declaration{};
  const void* local_type_token{};
  std::optional<std::size_t> operator_index;
  std::optional<TensorPoolBounds> setup_tensor_allocation;
  const std::type_info* local_type_info{};
};

struct OperatorStorage {
  std::string name;
  std::shared_ptr<const void> handle_identity;
  const void* operator_type_token{};
  std::shared_ptr<OperatorFactoryBase> factory;
  std::shared_ptr<OperatorBase> prototype;
  std::shared_ptr<const ParameterInputEvidenceState> parameter_input_evidence;
  VertexId vertex{};
  std::vector<GraphPortId> ports;
  OperatorRef plan_ref{};
};

struct ExternalTopicStorage {
  ExternalTopicRecord record;
  std::unique_ptr<InputBase> input;
  std::unique_ptr<OutputBase> output;
};

struct ClockStorage {
  ClockDeclarationRecord record;
  ClockCapabilityAuthoringRecord capabilities;
  std::shared_ptr<ClockFactoryBase> factory;
  std::shared_ptr<Clock> shared;
};

struct EmbeddedOperatorMapping {
  const GraphState* source_state{};
  std::size_t source_index{};
  VertexId vertex{};
  std::shared_ptr<OperatorFactoryBase> factory;
  std::shared_ptr<const ParameterInputEvidenceState> parameter_input_evidence;
};

struct EmbeddedGraphStorage {
  std::unique_ptr<Graph> graph;
  std::shared_ptr<GraphState> state;
  GraphId graph_id{};
  GraphInstancePath path;
  std::shared_ptr<const void> handle_identity;  ///< Shared only with handles for this registration.
};

class GraphState {
 public:
  GraphState(Graph* owner, std::string name);
  ~GraphState() = default;

  GraphState(const GraphState&) = delete;
  GraphState& operator=(const GraphState&) = delete;

  [[nodiscard]] expected<void, Error> prepare_mutation();
  [[nodiscard]] expected<void, Error> prepare_observation();
  void mark_mutation() noexcept;
  void detach() noexcept;
  void clear_failed_draft() noexcept;
  [[nodiscard]] bool dense_ids_valid() const noexcept {
    if (vertex_adjacency_.size() != vertices_.size() ||
        cycle_visit_marks_.size() != vertices_.size() || next_vertex_id_ != vertices_.size() + 1U ||
        next_port_id_ != ports_.size() + 1U || next_edge_id_ != edges_.size() + 1U ||
        next_region_id_ != regions_.size() + 1U || next_partition_id_ != partitions_.size() + 1U) {
      return false;
    }
    for (std::size_t index = 0; index < vertices_.size(); ++index) {
      if (vertices_[index].id.value != index + 1U)
        return false;
      if (std::any_of(vertex_adjacency_[index].begin(),
                      vertex_adjacency_[index].end(),
                      [&](std::size_t target) { return target >= vertices_.size(); })) {
        return false;
      }
    }
    for (std::size_t index = 0; index < ports_.size(); ++index) {
      if (ports_[index].record.id.value != index + 1U)
        return false;
    }
    if (edges_.size() != edge_handle_identities_.size())
      return false;
    for (std::size_t index = 0; index < edges_.size(); ++index) {
      if (edges_[index].id.value != index + 1U || !edge_handle_identities_[index])
        return false;
    }
    if (regions_.size() != region_handle_identities_.size())
      return false;
    for (std::size_t index = 0; index < regions_.size(); ++index) {
      if (regions_[index].id.value != index + 1U || !region_handle_identities_[index])
        return false;
    }
    if (partitions_.size() != partition_handle_identities_.size())
      return false;
    for (std::size_t index = 0; index < partitions_.size(); ++index) {
      if (partitions_[index].id.value != index + 1U || !partition_handle_identities_[index])
        return false;
    }
    return true;
  }

  [[nodiscard]] GraphPortId next_port_id() noexcept { return GraphPortId{next_port_id_++}; }
  [[nodiscard]] VertexId next_vertex_id() noexcept { return VertexId{next_vertex_id_++}; }
  [[nodiscard]] EdgeId next_edge_id() noexcept { return EdgeId{next_edge_id_++}; }
  [[nodiscard]] RegionId next_region_id() noexcept { return RegionId{next_region_id_++}; }
  [[nodiscard]] PartitionId next_partition_id() noexcept {
    return PartitionId{next_partition_id_++};
  }
  [[nodiscard]] ExternalTopicId next_external_topic_id() noexcept {
    return ExternalTopicId{next_external_topic_id_++};
  }
  [[nodiscard]] GraphId next_graph_id() noexcept { return GraphId{next_graph_id_++}; }
  [[nodiscard]] static GraphInstancePath make_path(std::vector<std::string> segments) {
    return GraphInstancePath{std::move(segments)};
  }

  Graph* owner_{};
  mutable std::recursive_mutex mutex_;
  std::condition_variable_any composition_cv_;
  CompositionState composition_state_{CompositionState::kUncomposed};
  std::thread::id composing_thread_;
  std::optional<GraphCompositionDiagnostic> composition_diagnostic_;
  GraphGeneration generation_{};
  GraphInstanceId instance_id_{};
  GraphRecord graph_record_;
  std::vector<GraphRecord> embedded_graph_records_;
  std::vector<EmbeddedGraphStorage> embedded_graphs_;
  std::vector<EmbeddedOperatorMapping> embedded_operator_mappings_;
  std::shared_ptr<const ParameterTransactionState> parameter_transaction_;
  std::vector<const void*> embedding_ancestry_;
  std::deque<OperatorStorage> operators_;
  std::vector<std::vector<std::size_t>> vertex_adjacency_;
  std::vector<std::uint64_t> cycle_visit_marks_;
  std::uint64_t cycle_visit_epoch_{};
  std::vector<VertexRecord> vertices_;
  std::deque<PortStorage> ports_;
  std::unordered_map<const void*, GraphPortId> port_by_declaration_;
  std::vector<EdgeRecord> edges_;
  std::vector<std::shared_ptr<const void>> edge_handle_identities_;
  std::vector<InterfaceBindingRecord> interface_bindings_;
  std::deque<ExternalTopicStorage> external_topics_;
  std::vector<ReadinessSourceDeclarationRecord> readiness_sources_;
  std::vector<RegionRecord> regions_;
  std::vector<std::shared_ptr<const void>> region_handle_identities_;
  std::vector<PartitionRecord> partitions_;
  std::vector<std::shared_ptr<const void>> partition_handle_identities_;
  bool automatic_partition_remaining_{};
  std::deque<ClockStorage> clocks_;
  std::vector<ClockImportAuthoringRecord> clock_imports_;
  std::vector<EmbeddedClockBindingAuthoringRecord> clock_bindings_;
  std::vector<ClockConversionAuthoringRecord> clock_conversions_;
  std::vector<LifecycleDependencyAuthoringRecord> lifecycle_dependencies_;
  std::vector<LifecycleCallbackAuthoringRecord> lifecycle_callbacks_;
  std::vector<LifecycleFactAuthoringRecord> lifecycle_facts_;
  std::vector<LifecycleRestartPermissionAuthoringRecord> lifecycle_restart_permissions_;
  std::unordered_set<std::string> vertex_names_;
  std::unordered_set<std::string> interface_port_names_;
  std::unordered_set<std::string> region_names_;
  std::unordered_set<std::string> partition_names_;
  std::unordered_set<std::string> clock_names_;
  std::unordered_set<std::string> explicit_endpoint_names_;
  std::unordered_set<std::string> embedded_instance_names_;
  std::vector<PayloadIdentityRecord> pod_identity_catalog_;
  ClockKey default_clock_;
  std::deque<TensorOutputPortConfigurator> tensor_configurators_;
  std::uint32_t next_boundary_ordinal_{};

 private:
  friend class GraphMutationRollback;
  friend class GraphCompilerAccess;
  std::uint64_t next_vertex_id_{1};
  std::uint64_t next_graph_id_{2};
  std::uint64_t next_port_id_{1};
  std::uint64_t next_edge_id_{1};
  std::uint64_t next_region_id_{1};
  std::uint64_t next_partition_id_{1};
  std::uint64_t next_external_topic_id_{1};
  std::optional<std::size_t> mutation_fault_countdown_;
  bool fail_composition_diagnostic_capture_{};
};

}  // namespace holoscan::detail
