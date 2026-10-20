/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "graph_snapshot.hpp"
#include "graph_state.hpp"

namespace holoscan::detail {

/// Restore appended Graph records and their derived side-index entries after a cold mutation
/// throws. Ordinary transactions capture only scalar sizes and counters; the compound mode also
/// owns pre-mutation region and partition records because nested public mutations can commit into
/// those existing records. Rollback may scan or swap prepared storage but never allocates.
class GraphMutationRollback {
 public:
  struct CaptureExistingRecords {};

  struct ExistingRecords {
    explicit ExistingRecords(const GraphState& state)
        : embedded_graph_records(state.embedded_graph_records_),
          embedded_operator_mappings(state.embedded_operator_mappings_),
          operators(state.operators_),
          vertex_adjacency(state.vertex_adjacency_),
          cycle_visit_marks(state.cycle_visit_marks_),
          cycle_visit_epoch(state.cycle_visit_epoch_),
          vertices(state.vertices_),
          ports(state.ports_),
          port_by_declaration(state.port_by_declaration_),
          edges(state.edges_),
          edge_handle_identities(state.edge_handle_identities_),
          interface_bindings(state.interface_bindings_),
          readiness_sources(state.readiness_sources_),
          regions(state.regions_),
          region_handle_identities(state.region_handle_identities_),
          partitions(state.partitions_),
          partition_handle_identities(state.partition_handle_identities_),
          automatic_partition_remaining(state.automatic_partition_remaining_),
          clocks(state.clocks_),
          clock_imports(state.clock_imports_),
          clock_bindings(state.clock_bindings_),
          clock_conversions(state.clock_conversions_),
          lifecycle_dependencies(state.lifecycle_dependencies_),
          lifecycle_callbacks(state.lifecycle_callbacks_),
          lifecycle_facts(state.lifecycle_facts_),
          lifecycle_restart_permissions(state.lifecycle_restart_permissions_),
          vertex_names(state.vertex_names_),
          interface_port_names(state.interface_port_names_),
          region_names(state.region_names_),
          partition_names(state.partition_names_),
          clock_names(state.clock_names_),
          explicit_endpoint_names(state.explicit_endpoint_names_),
          embedded_instance_names(state.embedded_instance_names_),
          pod_identity_catalog(state.pod_identity_catalog_),
          default_clock(state.default_clock_) {}

    std::vector<GraphRecord> embedded_graph_records;
    std::vector<EmbeddedOperatorMapping> embedded_operator_mappings;
    std::deque<OperatorStorage> operators;
    std::vector<std::vector<std::size_t>> vertex_adjacency;
    std::vector<std::uint64_t> cycle_visit_marks;
    std::uint64_t cycle_visit_epoch{};
    std::vector<VertexRecord> vertices;
    std::deque<PortStorage> ports;
    std::unordered_map<const void*, GraphPortId> port_by_declaration;
    std::vector<EdgeRecord> edges;
    std::vector<std::shared_ptr<const void>> edge_handle_identities;
    std::vector<InterfaceBindingRecord> interface_bindings;
    std::vector<ReadinessSourceDeclarationRecord> readiness_sources;
    std::vector<RegionRecord> regions;
    std::vector<std::shared_ptr<const void>> region_handle_identities;
    std::vector<PartitionRecord> partitions;
    std::vector<std::shared_ptr<const void>> partition_handle_identities;
    bool automatic_partition_remaining{};
    std::deque<ClockStorage> clocks;
    std::vector<ClockImportAuthoringRecord> clock_imports;
    std::vector<EmbeddedClockBindingAuthoringRecord> clock_bindings;
    std::vector<ClockConversionAuthoringRecord> clock_conversions;
    std::vector<LifecycleDependencyAuthoringRecord> lifecycle_dependencies;
    std::vector<LifecycleCallbackAuthoringRecord> lifecycle_callbacks;
    std::vector<LifecycleFactAuthoringRecord> lifecycle_facts;
    std::vector<LifecycleRestartPermissionAuthoringRecord> lifecycle_restart_permissions;
    std::unordered_set<std::string> vertex_names;
    std::unordered_set<std::string> interface_port_names;
    std::unordered_set<std::string> region_names;
    std::unordered_set<std::string> partition_names;
    std::unordered_set<std::string> clock_names;
    std::unordered_set<std::string> explicit_endpoint_names;
    std::unordered_set<std::string> embedded_instance_names;
    std::vector<PayloadIdentityRecord> pod_identity_catalog;
    ClockKey default_clock;
  };

  explicit GraphMutationRollback(GraphState& state) noexcept
      : state_(state),
        generation_(state.generation_),
        next_vertex_id_(state.next_vertex_id_),
        next_graph_id_(state.next_graph_id_),
        next_port_id_(state.next_port_id_),
        next_edge_id_(state.next_edge_id_),
        next_region_id_(state.next_region_id_),
        next_partition_id_(state.next_partition_id_),
        next_external_topic_id_(state.next_external_topic_id_),
        next_boundary_ordinal_(state.next_boundary_ordinal_),
        operators_size_(state.operators_.size()),
        embedded_graph_records_size_(state.embedded_graph_records_.size()),
        embedded_graphs_size_(state.embedded_graphs_.size()),
        embedded_operator_mappings_size_(state.embedded_operator_mappings_.size()),
        parameter_transaction_(state.parameter_transaction_),
        vertices_size_(state.vertices_.size()),
        ports_size_(state.ports_.size()),
        edges_size_(state.edges_.size()),
        interface_bindings_size_(state.interface_bindings_.size()),
        external_topics_size_(state.external_topics_.size()),
        readiness_sources_size_(state.readiness_sources_.size()),
        regions_size_(state.regions_.size()),
        partitions_size_(state.partitions_.size()),
        clocks_size_(state.clocks_.size()),
        clock_imports_size_(state.clock_imports_.size()),
        clock_bindings_size_(state.clock_bindings_.size()),
        clock_conversions_size_(state.clock_conversions_.size()),
        lifecycle_dependencies_size_(state.lifecycle_dependencies_.size()),
        lifecycle_callbacks_size_(state.lifecycle_callbacks_.size()),
        lifecycle_facts_size_(state.lifecycle_facts_.size()),
        lifecycle_permissions_size_(state.lifecycle_restart_permissions_.size()),
        pod_identity_catalog_size_(state.pod_identity_catalog_.size()) {
    assert(state_.dense_ids_valid());
  }

  GraphMutationRollback(GraphState& state, CaptureExistingRecords);

  GraphMutationRollback(const GraphMutationRollback&) = delete;
  GraphMutationRollback& operator=(const GraphMutationRollback&) = delete;

  ~GraphMutationRollback() noexcept;

  void rollback() noexcept;

  void restore_boundary_ordinal(std::uint32_t ordinal) noexcept {
    next_boundary_ordinal_ = ordinal;
  }
  void commit() noexcept {
    assert(state_.dense_ids_valid());
    committed_ = true;
  }
  static void checkpoint(GraphState& state) {
    if (!state.mutation_fault_countdown_)
      return;
    if (*state.mutation_fault_countdown_ == 0U) {
      state.mutation_fault_countdown_.reset();
      throw std::bad_alloc{};
    }
    --*state.mutation_fault_countdown_;
  }

 private:
  void rollback_complete_mutation() noexcept;
  void release_root_port_ownership() noexcept;
  void claim_root_port_ownership() noexcept;
  void rollback_ports() noexcept;
  void rollback_name_indexes() noexcept;
  void rollback_adjacency() noexcept;
  void rollback_embedded_names() noexcept;
  void truncate_records() noexcept;
  void restore_identity() noexcept;
  static bool is_root_vertex(const GraphState& state, VertexId id) noexcept;
  static void remove_adjacency(GraphState& state, VertexId source, VertexId target) noexcept;
  static const PortStorage* rollback_port(const GraphState& state, GraphPortId id) noexcept;
  static void remove_edge_adjacency(GraphState& state, const EdgeRecord& edge) noexcept;
  static void remove_binding_adjacency(GraphState& state,
                                       const InterfaceBindingRecord& binding) noexcept;
  static void erase_declaration(std::unordered_map<const void*, GraphPortId>& declarations,
                                const void* declaration) noexcept;
  static void erase_name(std::unordered_set<std::string>& names, std::string_view name) noexcept;

  GraphState& state_;
  GraphGeneration generation_;
  std::uint64_t next_vertex_id_;
  std::uint64_t next_graph_id_;
  std::uint64_t next_port_id_;
  std::uint64_t next_edge_id_;
  std::uint64_t next_region_id_;
  std::uint64_t next_partition_id_;
  std::uint64_t next_external_topic_id_;
  std::uint32_t next_boundary_ordinal_;
  std::size_t operators_size_;
  std::size_t embedded_graph_records_size_;
  std::size_t embedded_graphs_size_;
  std::size_t embedded_operator_mappings_size_;
  std::shared_ptr<const ParameterTransactionState> parameter_transaction_;
  std::size_t vertices_size_;
  std::size_t ports_size_;
  std::size_t edges_size_;
  std::size_t interface_bindings_size_;
  std::size_t external_topics_size_;
  std::size_t readiness_sources_size_;
  std::size_t regions_size_;
  std::size_t partitions_size_;
  std::size_t clocks_size_;
  std::size_t clock_imports_size_;
  std::size_t clock_bindings_size_;
  std::size_t clock_conversions_size_;
  std::size_t lifecycle_dependencies_size_;
  std::size_t lifecycle_callbacks_size_;
  std::size_t lifecycle_facts_size_;
  std::size_t lifecycle_permissions_size_;
  std::size_t pod_identity_catalog_size_;
  std::unique_ptr<ExistingRecords> existing_records_;
  bool committed_{};
};

}  // namespace holoscan::detail
