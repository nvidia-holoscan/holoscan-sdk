/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/graph.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/time/synthetic_clock.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "graph_state.hpp"
#include "operator_spec_compiler_access.hpp"
#include "tensor_field_configuration_internal.hpp"

namespace holoscan {
namespace detail {

[[nodiscard]] ClockCapabilityAuthoringRecord clock_capabilities(const Clock& clock, GraphId graph,
                                                                const ClockKey& key) {
  ClockCapabilityAuthoringRecord result{
      .graph = graph, .key = key, .can_read_now = true, .can_wait_until = true};
  if (dynamic_cast<const RealtimeClock*>(&clock) != nullptr) {
    result.steady = true;
    result.independent_progress = true;
    result.finite_deadline = true;
    result.evidence_id = "holoscan.clock.realtime.v1";
  } else if (dynamic_cast<const ManualClock*>(&clock) != nullptr) {
    result.steady = true;
    result.evidence_id = "holoscan.clock.manual.external-progress-required.v1";
  } else if (dynamic_cast<const SyntheticClock*>(&clock) != nullptr) {
    result.steady = true;
    result.evidence_id = "holoscan.clock.synthetic.external-progress-required.v1";
  } else {
    // Clock's base interface proves readable/waitable operations, but it does
    // not prove monotonic or independently progressing deadline behavior.
    result.evidence_id = "holoscan.clock.custom.unqualified.v1";
  }
  return result;
}

void graph_log_factory_exception(std::string_view context, std::string_view detail) noexcept {
  try {
    HOLOSCAN_LOG_DEBUG("Graph {} factory exception detail: {}", context, detail);
  } catch (...) {  // NOLINT(bugprone-empty-catch)
    // Logging is diagnostic-only and must never replace the stable typed factory failure.
  }
}

/// Restore appended Graph records and their derived side-index entries after a cold mutation
/// throws. Successful transactions capture only scalar sizes and counters; rollback may scan the
/// newly appended records but never allocates. Existing nested records are prepared off to the
/// side and swapped separately by their mutators.
class GraphMutationRollback {
 public:
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

  GraphMutationRollback(const GraphMutationRollback&) = delete;
  GraphMutationRollback& operator=(const GraphMutationRollback&) = delete;

  ~GraphMutationRollback() noexcept {
    if (committed_)
      return;
    rollback_ports();
    rollback_name_indexes();
    rollback_adjacency();
    rollback_embedded_names();
    truncate_records();
    restore_identity();
    assert(state_.dense_ids_valid());
  }

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
  void rollback_ports() noexcept {
    for (std::size_t index = ports_size_; index < state_.ports_.size(); ++index) {
      const auto& port = state_.ports_[index];
      erase_declaration(state_.port_by_declaration_, port.declaration);
      const bool root_owned = is_root_vertex(state_, port.record.owner);
      const bool parent_owned = port.declaration != nullptr && root_owned;
      if (root_owned && port.record.owner_kind == PortOwnerKind::kGraphInterface) {
        erase_name(state_.interface_port_names_, port.record.name);
      }
      if (!parent_owned)
        continue;
      if (port.record.direction == PortDirection::kInput) {
        GraphPortOwnershipAccess::claim(*static_cast<const InputBase*>(port.declaration), nullptr);
      } else {
        GraphPortOwnershipAccess::claim(*static_cast<const OutputBase*>(port.declaration), nullptr);
      }
    }
  }

  void rollback_name_indexes() noexcept {
    for (std::size_t index = vertices_size_; index < state_.vertices_.size(); ++index) {
      const auto& vertex = state_.vertices_[index];
      if (vertex.graph == state_.graph_record_.id && vertex.kind == VertexKind::kOperator) {
        erase_name(state_.vertex_names_, vertex.name);
      }
    }
    for (std::size_t index = regions_size_; index < state_.regions_.size(); ++index) {
      if (state_.regions_[index].graph == state_.graph_record_.id) {
        erase_name(state_.region_names_, state_.regions_[index].name);
      }
    }
    for (std::size_t index = partitions_size_; index < state_.partitions_.size(); ++index) {
      if (state_.partitions_[index].graph == state_.graph_record_.id) {
        erase_name(state_.partition_names_, state_.partitions_[index].name);
      }
    }
    for (std::size_t index = clocks_size_; index < state_.clocks_.size(); ++index) {
      if (state_.clocks_[index].record.graph == state_.graph_record_.id) {
        erase_name(state_.clock_names_, state_.clocks_[index].record.key.name());
      }
    }
    for (std::size_t index = clock_imports_size_; index < state_.clock_imports_.size(); ++index) {
      if (state_.clock_imports_[index].graph == state_.graph_record_.id) {
        erase_name(state_.clock_names_, state_.clock_imports_[index].key.name());
      }
    }
    for (std::size_t index = external_topics_size_; index < state_.external_topics_.size();
         ++index) {
      const auto& endpoint_name = state_.external_topics_[index].record.endpoint_name;
      if (endpoint_name) {
        erase_name(state_.explicit_endpoint_names_, *endpoint_name);
      }
    }
  }

  void rollback_adjacency() noexcept {
    for (std::size_t index = state_.interface_bindings_.size(); index > interface_bindings_size_;
         --index) {
      remove_binding_adjacency(state_, state_.interface_bindings_[index - 1U]);
    }
    for (std::size_t index = state_.edges_.size(); index > edges_size_; --index) {
      remove_edge_adjacency(state_, state_.edges_[index - 1U]);
    }
  }

  void rollback_embedded_names() noexcept {
    for (std::size_t index = embedded_graph_records_size_;
         index < state_.embedded_graph_records_.size();
         ++index) {
      const auto& graph = state_.embedded_graph_records_[index];
      if (graph.parent == state_.graph_record_.id) {
        erase_name(state_.embedded_instance_names_, graph.name);
      }
    }
  }

  void truncate_records() noexcept {
    state_.interface_bindings_.resize(interface_bindings_size_);
    state_.edges_.resize(edges_size_);
    state_.readiness_sources_.resize(readiness_sources_size_);
    state_.clock_bindings_.resize(clock_bindings_size_);
    state_.clock_imports_.resize(clock_imports_size_);
    state_.clock_conversions_.resize(clock_conversions_size_);
    state_.lifecycle_dependencies_.resize(lifecycle_dependencies_size_);
    state_.lifecycle_callbacks_.resize(lifecycle_callbacks_size_);
    state_.lifecycle_facts_.resize(lifecycle_facts_size_);
    state_.lifecycle_restart_permissions_.resize(lifecycle_permissions_size_);
    state_.ports_.resize(ports_size_);
    state_.vertices_.resize(vertices_size_);
    state_.external_topics_.resize(external_topics_size_);
    state_.operators_.resize(operators_size_);
    state_.embedded_operator_mappings_.resize(embedded_operator_mappings_size_);
    state_.embedded_graph_records_.resize(embedded_graph_records_size_);
    state_.embedded_graphs_.erase(
        state_.embedded_graphs_.begin() + static_cast<std::ptrdiff_t>(embedded_graphs_size_),
        state_.embedded_graphs_.end());
    state_.vertex_adjacency_.resize(vertices_size_);
    state_.cycle_visit_marks_.resize(vertices_size_);
    state_.regions_.resize(regions_size_);
    state_.partitions_.resize(partitions_size_);
    state_.clocks_.resize(clocks_size_);
    state_.pod_identity_catalog_.resize(pod_identity_catalog_size_);
  }

  void restore_identity() noexcept {
    state_.generation_ = generation_;
    state_.next_vertex_id_ = next_vertex_id_;
    state_.next_graph_id_ = next_graph_id_;
    state_.next_port_id_ = next_port_id_;
    state_.next_edge_id_ = next_edge_id_;
    state_.next_region_id_ = next_region_id_;
    state_.next_partition_id_ = next_partition_id_;
    state_.next_external_topic_id_ = next_external_topic_id_;
    state_.next_boundary_ordinal_ = next_boundary_ordinal_;
  }

  static bool is_root_vertex(const GraphState& state, VertexId id) noexcept {
    if (!id || id.value > state.vertices_.size())
      return false;
    const auto& vertex = state.vertices_[static_cast<std::size_t>(id.value - 1U)];
    return vertex.id == id && vertex.graph == state.graph_record_.id;
  }

  static void remove_adjacency(GraphState& state, VertexId source, VertexId target) noexcept {
    if (!source || !target || source.value > state.vertex_adjacency_.size())
      return;
    auto& outgoing = state.vertex_adjacency_[static_cast<std::size_t>(source.value - 1U)];
    const std::size_t target_index = static_cast<std::size_t>(target.value - 1U);
    if (!outgoing.empty() && outgoing.back() == target_index) {
      outgoing.pop_back();
    }
  }

  static const PortStorage* rollback_port(const GraphState& state, GraphPortId id) noexcept {
    if (!id || id.value > state.ports_.size())
      return nullptr;
    const auto& port = state.ports_[static_cast<std::size_t>(id.value - 1U)];
    return port.record.id == id ? &port : nullptr;
  }

  static void remove_edge_adjacency(GraphState& state, const EdgeRecord& edge) noexcept {
    const PortStorage* source = rollback_port(state, edge.source);
    const PortStorage* target = rollback_port(state, edge.target);
    if (source != nullptr && target != nullptr) {
      remove_adjacency(state, source->record.owner, target->record.owner);
    }
  }

  static void remove_binding_adjacency(GraphState& state,
                                       const InterfaceBindingRecord& binding) noexcept {
    const PortStorage* public_port = rollback_port(state, binding.public_port);
    const PortStorage* internal_port = rollback_port(state, binding.internal_port);
    if (public_port == nullptr || internal_port == nullptr)
      return;
    if (binding.direction == PortDirection::kInput) {
      remove_adjacency(state, public_port->record.owner, internal_port->record.owner);
    } else {
      remove_adjacency(state, internal_port->record.owner, public_port->record.owner);
    }
  }

  static void erase_declaration(std::unordered_map<const void*, GraphPortId>& declarations,
                                const void* declaration) noexcept {
    const auto found = std::find_if(declarations.begin(),
                                    declarations.end(),
                                    [&](const auto& item) { return item.first == declaration; });
    if (found != declarations.end()) {
      declarations.erase(found);
    }
  }

  static void erase_name(std::unordered_set<std::string>& names, std::string_view name) noexcept {
    const auto found = std::find_if(
        names.begin(), names.end(), [&](const std::string& item) { return item == name; });
    if (found != names.end()) {
      names.erase(found);
    }
  }

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
  bool committed_{};
};

/// Restore a deliberately pre-validation external-boundary ordinal only when
/// control leaves by exception. Ordinary typed validation failures return
/// normally and therefore retain their GRAPH-011 declaration slot.
class ExceptionalBoundaryOrdinalRollback {
 public:
  ExceptionalBoundaryOrdinalRollback(std::uint32_t& next_ordinal,
                                     std::uint32_t reserved_ordinal) noexcept
      : next_ordinal_(next_ordinal),
        reserved_ordinal_(reserved_ordinal),
        uncaught_exceptions_(std::uncaught_exceptions()) {}

  ExceptionalBoundaryOrdinalRollback(const ExceptionalBoundaryOrdinalRollback&) = delete;
  ExceptionalBoundaryOrdinalRollback& operator=(const ExceptionalBoundaryOrdinalRollback&) = delete;

  ~ExceptionalBoundaryOrdinalRollback() noexcept {
    if (std::uncaught_exceptions() > uncaught_exceptions_)
      next_ordinal_ = reserved_ordinal_;
  }

 private:
  std::uint32_t& next_ordinal_;
  std::uint32_t reserved_ordinal_;
  int uncaught_exceptions_;
};

namespace {

constexpr std::string_view kGeneratedEndpointNamespace{"__holoscan_auto_endpoint/"};
constexpr std::string_view kHoloMqTopicPrefix{"holomq://"};

std::atomic<std::uint64_t> g_next_graph_instance{1U};

[[nodiscard]] Error graph_error(ErrorCode code, std::string_view token) {
  return Error{code, std::string(token)};
}

[[nodiscard]] Error tensor_field_graph_error(ErrorCode code, std::string_view token,
                                             const TensorFieldConfigurationRecord& field) {
  std::string message(token);
  message.append(" (field_id=");
  message.append(std::to_string(field.field_id));
  message.append(", tensor_index=");
  message.append(std::to_string(field.tensor_index));
  message.push_back(')');
  return Error{code, std::move(message)};
}

[[nodiscard]] std::string bounded_exception_context(std::string_view context) {
  constexpr std::size_t kMaximumContextBytes = 160U;
  const std::size_t count = std::min(context.size(), kMaximumContextBytes);
  std::string result;
  result.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const unsigned char value = static_cast<unsigned char>(context[index]);
    result.push_back(value >= 0x20U && value != 0x7FU ? static_cast<char>(value) : '?');
  }
  return result;
}

template <typename Callable>
void graph_log_noexcept(Callable&& callable) noexcept {
  try {
    std::forward<Callable>(callable)();
  } catch (...) {  // NOLINT(bugprone-empty-catch)
    // Logging is diagnostic-only. Formatting, allocation, and logger backend
    // failures must not replace an authoring result or make committed state
    // appear to have failed.
  }
}

template <typename Enum>
[[nodiscard]] constexpr bool closed_enum(Enum value) noexcept;

// clang-format off
#define HOLOSCAN_GRAPH_CLOSED_ENUM(EnumType, ...)                     \
  template <>                                                         \
  [[nodiscard]] constexpr bool closed_enum(EnumType value) noexcept { \
    switch (value) {                                                  \
      __VA_ARGS__                                                     \
    }                                                                 \
    return false;                                                     \
  }
#define HOLOSCAN_GRAPH_ENUM_CASE(Value) \
  case Value:                           \
    return true;
// clang-format on

// The generated switches deliberately return the same value for every listed enumerator: the
// repetition is the explicit whitelist that rejects fabricated closed-enum values.
// NOLINTBEGIN(bugprone-branch-clone)
HOLOSCAN_GRAPH_CLOSED_ENUM(QueuePolicy, HOLOSCAN_GRAPH_ENUM_CASE(QueuePolicy::kBlocking)
                                            HOLOSCAN_GRAPH_ENUM_CASE(QueuePolicy::kDropOldest))
HOLOSCAN_GRAPH_CLOSED_ENUM(MemoryConversionPolicy,
                           HOLOSCAN_GRAPH_ENUM_CASE(MemoryConversionPolicy::kDefault)
                               HOLOSCAN_GRAPH_ENUM_CASE(MemoryConversionPolicy::kDisable)
                                   HOLOSCAN_GRAPH_ENUM_CASE(MemoryConversionPolicy::kAllowIfSafe)
                                       HOLOSCAN_GRAPH_ENUM_CASE(MemoryConversionPolicy::kRequire))
HOLOSCAN_GRAPH_CLOSED_ENUM(
    InterfaceTensorEmitPolicy,
    HOLOSCAN_GRAPH_ENUM_CASE(InterfaceTensorEmitPolicy::kRequireExportable)
        HOLOSCAN_GRAPH_ENUM_CASE(InterfaceTensorEmitPolicy::kCopyIfNeeded)
            HOLOSCAN_GRAPH_ENUM_CASE(InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded))
HOLOSCAN_GRAPH_CLOSED_ENUM(TensorStridePolicy,
                           HOLOSCAN_GRAPH_ENUM_CASE(TensorStridePolicy::kContiguousOnly)
                               HOLOSCAN_GRAPH_ENUM_CASE(TensorStridePolicy::kBoundedStrided))
HOLOSCAN_GRAPH_CLOSED_ENUM(PeerRequirement,
                           HOLOSCAN_GRAPH_ENUM_CASE(PeerRequirement::kOptional)
                               HOLOSCAN_GRAPH_ENUM_CASE(PeerRequirement::kRequired))
HOLOSCAN_GRAPH_CLOSED_ENUM(ClockHealthCapability,
                           HOLOSCAN_GRAPH_ENUM_CASE(ClockHealthCapability::kUnavailable)
                               HOLOSCAN_GRAPH_ENUM_CASE(ClockHealthCapability::kStatusReports))
HOLOSCAN_GRAPH_CLOSED_ENUM(
    ClockHealthRequirement,
    HOLOSCAN_GRAPH_ENUM_CASE(ClockHealthRequirement::kNotRequired)
        HOLOSCAN_GRAPH_ENUM_CASE(ClockHealthRequirement::kFreshHealthyReport))
HOLOSCAN_GRAPH_CLOSED_ENUM(MemoryKind,
                           HOLOSCAN_GRAPH_ENUM_CASE(MemoryKind::kUnknown)
                               HOLOSCAN_GRAPH_ENUM_CASE(MemoryKind::kHost)
                                   HOLOSCAN_GRAPH_ENUM_CASE(MemoryKind::kPinnedHost)
                                       HOLOSCAN_GRAPH_ENUM_CASE(MemoryKind::kCudaDevice)
                                           HOLOSCAN_GRAPH_ENUM_CASE(MemoryKind::kCudaManaged))
HOLOSCAN_GRAPH_CLOSED_ENUM(
    ScalarIdentityKind,
    HOLOSCAN_GRAPH_ENUM_CASE(ScalarIdentityKind::kBoolean) HOLOSCAN_GRAPH_ENUM_CASE(
        ScalarIdentityKind::kUInt8) HOLOSCAN_GRAPH_ENUM_CASE(ScalarIdentityKind::kInt8)
        HOLOSCAN_GRAPH_ENUM_CASE(ScalarIdentityKind::kUInt16)
            HOLOSCAN_GRAPH_ENUM_CASE(ScalarIdentityKind::kInt16)
                HOLOSCAN_GRAPH_ENUM_CASE(ScalarIdentityKind::kUInt32)
                    HOLOSCAN_GRAPH_ENUM_CASE(ScalarIdentityKind::kInt32)
                        HOLOSCAN_GRAPH_ENUM_CASE(ScalarIdentityKind::kUInt64)
                            HOLOSCAN_GRAPH_ENUM_CASE(ScalarIdentityKind::kInt64)
                                HOLOSCAN_GRAPH_ENUM_CASE(ScalarIdentityKind::kFloat32)
                                    HOLOSCAN_GRAPH_ENUM_CASE(ScalarIdentityKind::kFloat64)
                                        HOLOSCAN_GRAPH_ENUM_CASE(ScalarIdentityKind::kLocalEnum))
HOLOSCAN_GRAPH_CLOSED_ENUM(PayloadKind,
                           HOLOSCAN_GRAPH_ENUM_CASE(PayloadKind::kInlineTyped)
                               HOLOSCAN_GRAPH_ENUM_CASE(PayloadKind::kSerializedBytes)
                                   HOLOSCAN_GRAPH_ENUM_CASE(PayloadKind::kDescriptorBacked)
                                       HOLOSCAN_GRAPH_ENUM_CASE(PayloadKind::kCompatibilityPayload)
                                           HOLOSCAN_GRAPH_ENUM_CASE(PayloadKind::kControlEmpty))
HOLOSCAN_GRAPH_CLOSED_ENUM(PayloadIdentityKind,
                           HOLOSCAN_GRAPH_ENUM_CASE(PayloadIdentityKind::kScalar)
                               HOLOSCAN_GRAPH_ENUM_CASE(PayloadIdentityKind::kTensor)
                                   HOLOSCAN_GRAPH_ENUM_CASE(PayloadIdentityKind::kSchema)
                                       HOLOSCAN_GRAPH_ENUM_CASE(PayloadIdentityKind::kPod))
HOLOSCAN_GRAPH_CLOSED_ENUM(Profile, HOLOSCAN_GRAPH_ENUM_CASE(Profile::kStandard)
                                        HOLOSCAN_GRAPH_ENUM_CASE(Profile::kRobotics)
                                            HOLOSCAN_GRAPH_ENUM_CASE(Profile::kRealTime)
                                                HOLOSCAN_GRAPH_ENUM_CASE(Profile::kRegulated)
                                                    HOLOSCAN_GRAPH_ENUM_CASE(Profile::kResearch))
HOLOSCAN_GRAPH_CLOSED_ENUM(
    RegionConstraintKind,
    HOLOSCAN_GRAPH_ENUM_CASE(RegionConstraintKind::kStaticTopology)
        HOLOSCAN_GRAPH_ENUM_CASE(RegionConstraintKind::kBoundedMemory)
            HOLOSCAN_GRAPH_ENUM_CASE(RegionConstraintKind::kDeviceCompatibleOperators)
                HOLOSCAN_GRAPH_ENUM_CASE(RegionConstraintKind::kLatencyBudget))
HOLOSCAN_GRAPH_CLOSED_ENUM(OptimizationHint,
                           HOLOSCAN_GRAPH_ENUM_CASE(OptimizationHint::kCudaGraphFusion))
HOLOSCAN_GRAPH_CLOSED_ENUM(
    LifecycleStage, HOLOSCAN_GRAPH_ENUM_CASE(LifecycleStage::kConfigure)
                        HOLOSCAN_GRAPH_ENUM_CASE(LifecycleStage::kDiscover)
                            HOLOSCAN_GRAPH_ENUM_CASE(LifecycleStage::kAllocate)
                                HOLOSCAN_GRAPH_ENUM_CASE(LifecycleStage::kWarmUp)
                                    HOLOSCAN_GRAPH_ENUM_CASE(LifecycleStage::kArm)
                                        HOLOSCAN_GRAPH_ENUM_CASE(LifecycleStage::kStart)
                                            HOLOSCAN_GRAPH_ENUM_CASE(LifecycleStage::kStop)
                                                HOLOSCAN_GRAPH_ENUM_CASE(LifecycleStage::kRelease))
HOLOSCAN_GRAPH_CLOSED_ENUM(
    OperatorSpecCompilerAccess::LifecycleFactKind,
    HOLOSCAN_GRAPH_ENUM_CASE(OperatorSpecCompilerAccess::LifecycleFactKind::kProduces)
        HOLOSCAN_GRAPH_ENUM_CASE(OperatorSpecCompilerAccess::LifecycleFactKind::kRequires))
HOLOSCAN_GRAPH_CLOSED_ENUM(
    LifecycleRestartOrigin,
    HOLOSCAN_GRAPH_ENUM_CASE(LifecycleRestartOrigin::kSelf)
        HOLOSCAN_GRAPH_ENUM_CASE(LifecycleRestartOrigin::kExternal)
            HOLOSCAN_GRAPH_ENUM_CASE(LifecycleRestartOrigin::kOperatorRequest))
HOLOSCAN_GRAPH_CLOSED_ENUM(ReadinessSourceKind,
                           HOLOSCAN_GRAPH_ENUM_CASE(ReadinessSourceKind::kReadable)
                               HOLOSCAN_GRAPH_ENUM_CASE(ReadinessSourceKind::kNotification)
                                   HOLOSCAN_GRAPH_ENUM_CASE(ReadinessSourceKind::kDoorbell)
                                       HOLOSCAN_GRAPH_ENUM_CASE(ReadinessSourceKind::kSpin))
// NOLINTEND(bugprone-branch-clone)

#undef HOLOSCAN_GRAPH_ENUM_CASE
#undef HOLOSCAN_GRAPH_CLOSED_ENUM

[[nodiscard]] bool valid_utf8(std::string_view value) noexcept {
  std::size_t index = 0;
  while (index < value.size()) {
    const auto first = static_cast<unsigned char>(value[index]);
    if (first < 0x80U) {
      ++index;
      continue;
    }
    std::size_t remaining = 0;
    std::uint32_t codepoint = 0;
    if ((first & 0xE0U) == 0xC0U) {
      remaining = 1;
      codepoint = first & 0x1FU;
      if (codepoint < 2U)
        return false;
    } else if ((first & 0xF0U) == 0xE0U) {
      remaining = 2;
      codepoint = first & 0x0FU;
    } else if ((first & 0xF8U) == 0xF0U) {
      remaining = 3;
      codepoint = first & 0x07U;
    } else {
      return false;
    }
    if (index + remaining >= value.size())
      return false;
    for (std::size_t offset = 1; offset <= remaining; ++offset) {
      const auto continuation = static_cast<unsigned char>(value[index + offset]);
      if ((continuation & 0xC0U) != 0x80U)
        return false;
      codepoint = (codepoint << 6U) | (continuation & 0x3FU);
    }
    if ((remaining == 2 && codepoint < 0x800U) || (remaining == 3 && codepoint < 0x10000U) ||
        (codepoint >= 0xD800U && codepoint <= 0xDFFFU) || codepoint > 0x10FFFFU) {
      return false;
    }
    index += remaining + 1U;
  }
  return true;
}

[[nodiscard]] bool edge_whitespace(std::string_view value) noexcept {
  if (value.empty())
    return false;
  const auto decode = [&](std::size_t index) {
    const auto first = static_cast<unsigned char>(value[index]);
    if (first < 0x80U)
      return static_cast<std::uint32_t>(first);
    const std::size_t remaining = (first & 0xE0U) == 0xC0U   ? 1U
                                  : (first & 0xF0U) == 0xE0U ? 2U
                                                             : 3U;
    std::uint32_t codepoint = first & (remaining == 1U ? 0x1FU : remaining == 2U ? 0x0FU : 0x07U);
    for (std::size_t offset = 1; offset <= remaining; ++offset) {
      codepoint = (codepoint << 6U) | (static_cast<unsigned char>(value[index + offset]) & 0x3FU);
    }
    return codepoint;
  };
  const auto whitespace = [](std::uint32_t codepoint) {
    return (codepoint >= 0x0009U && codepoint <= 0x000DU) || codepoint == 0x0020U ||
           codepoint == 0x0085U || codepoint == 0x00A0U || codepoint == 0x1680U ||
           (codepoint >= 0x2000U && codepoint <= 0x200AU) || codepoint == 0x2028U ||
           codepoint == 0x2029U || codepoint == 0x202FU || codepoint == 0x205FU ||
           codepoint == 0x3000U;
  };
  std::size_t last = value.size() - 1U;
  while (last > 0U && (static_cast<unsigned char>(value[last]) & 0xC0U) == 0x80U) {
    --last;
  }
  return whitespace(decode(0)) || whitespace(decode(last));
}

[[nodiscard]] expected<void, Error> validate_name(std::string_view name, std::string_view token,
                                                  bool allow_empty = false) {
  if ((!allow_empty && name.empty()) || name.size() > 255U || !valid_utf8(name) ||
      edge_whitespace(name) || std::any_of(name.begin(), name.end(), [](char value) {
        const auto byte = static_cast<unsigned char>(value);
        return byte == 0U || byte < 0x20U || byte == 0x7FU;
      })) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, token));
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_endpoint_name(std::string_view name) {
  auto valid = validate_name(name, "GRAPH_ENDPOINT_NAME_INVALID");
  if (!valid)
    return valid;
  if (name.starts_with(kGeneratedEndpointNamespace)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_RESERVED_ENDPOINT_NAMESPACE"));
  }
  return {};
}

[[nodiscard]] std::uint64_t pod_scalar_size(PodScalarKind kind) noexcept {
  switch (kind) {
    case PodScalarKind::kUInt8:
    case PodScalarKind::kInt8:
      return 1;
    case PodScalarKind::kUInt16:
    case PodScalarKind::kInt16:
      return 2;
    case PodScalarKind::kUInt32:
    case PodScalarKind::kInt32:
    case PodScalarKind::kFloat32:
      return 4;
    case PodScalarKind::kUInt64:
    case PodScalarKind::kInt64:
    case PodScalarKind::kFloat64:
      return 8;
  }
  return 0;
}

[[nodiscard]] expected<void, Error> validate_schema(const SchemaIdentityRecord& schema) {
  auto name = validate_name(schema.canonical_name, "GRAPH_SCHEMA_NAME_INVALID");
  if (!name)
    return name;
  if (schema.version == 0U) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_SCHEMA_VERSION_INVALID"));
  }
  if (std::all_of(schema.schema_sha256.begin(), schema.schema_sha256.end(), [](std::byte value) {
        return value == std::byte{};
      })) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_SCHEMA_DIGEST_INVALID"));
  }
  return {};
}

[[nodiscard]] bool schema_identity_equal(const SchemaIdentityRecord& lhs,
                                         const SchemaIdentityRecord& rhs) noexcept {
  return lhs.canonical_name == rhs.canonical_name && lhs.version == rhs.version &&
         lhs.schema_sha256 == rhs.schema_sha256 && lhs.id == rhs.id;
}

[[nodiscard]] bool valid_dtype(DLDataType dtype) noexcept {
  if (dtype.bits == 0U || dtype.lanes == 0U)
    return false;
  switch (dtype.code) {
    case kDLInt:
    case kDLUInt:
    case kDLFloat:
    case kDLOpaqueHandle:
    case kDLBfloat:
    case kDLComplex:
    case kDLBool:
      return true;
    default:
      return false;
  }
}

[[nodiscard]] expected<void, Error> validate_pod(const PodIdentityRecord& pod) {
  auto name = validate_name(pod.canonical_name, "GRAPH_POD_NAME_INVALID");
  if (!name)
    return name;
  if (pod.version == 0U || pod.object_size == 0U || pod.object_alignment == 0U ||
      (pod.object_alignment & (pod.object_alignment - 1U)) != 0U || pod.fields.empty()) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_POD_HEADER_INVALID"));
  }
  std::unordered_set<std::string> field_names;
  std::uint64_t previous_end = 0;
  for (const auto& field : pod.fields) {
    auto field_name = validate_name(field.name, "GRAPH_POD_FIELD_NAME_INVALID");
    if (!field_name)
      return field_name;
    if (!field_names.emplace(field.name).second || field.element_count == 0U) {
      return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_POD_FIELD_INVALID"));
    }
    const std::uint64_t element_size = pod_scalar_size(field.scalar_kind);
    const std::uint64_t byte_count = element_size * field.element_count;
    const std::uint64_t end = static_cast<std::uint64_t>(field.byte_offset) + byte_count;
    if (element_size == 0U || end < field.byte_offset || end > pod.object_size ||
        field.byte_offset != previous_end) {
      return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_POD_LAYOUT_INVALID"));
    }
    previous_end = end;
  }
  if (previous_end != pod.object_size) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_POD_LAYOUT_INVALID"));
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_payload_identity(
    const PayloadIdentityRecord& identity) {
  if (!closed_enum(identity.kind) || !closed_enum(identity.payload_kind)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_PAYLOAD_IDENTITY_ENUM_INVALID"));
  }
  switch (identity.kind) {
    case PayloadIdentityKind::kScalar:
      if (!identity.scalar || !closed_enum(identity.scalar->kind) || identity.schema ||
          identity.pod) {
        return make_unexpected(
            graph_error(ErrorCode::kInvalidArgument, "GRAPH_PAYLOAD_IDENTITY_INVALID"));
      }
      break;
    case PayloadIdentityKind::kTensor:
      if (identity.scalar || identity.schema || identity.pod ||
          identity.payload_kind != PayloadKind::kDescriptorBacked) {
        return make_unexpected(
            graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_IDENTITY_INVALID"));
      }
      break;
    case PayloadIdentityKind::kSchema: {
      const bool supported_schema_payload =
          identity.payload_kind == PayloadKind::kSerializedBytes ||
          identity.payload_kind == PayloadKind::kDescriptorBacked;
      if (identity.scalar || !identity.schema || identity.pod || !supported_schema_payload) {
        return make_unexpected(
            graph_error(ErrorCode::kInvalidArgument, "GRAPH_SCHEMA_IDENTITY_INVALID"));
      }
      return validate_schema(*identity.schema);
    }
    case PayloadIdentityKind::kPod:
      if (identity.scalar || identity.schema || !identity.pod) {
        return make_unexpected(
            graph_error(ErrorCode::kInvalidArgument, "GRAPH_POD_IDENTITY_INVALID"));
      }
      return validate_pod(*identity.pod);
  }
  return {};
}

[[nodiscard]] bool tensor_bearing(const PayloadIdentityRecord& identity,
                                  const std::optional<TensorPortLayout>& layout) noexcept {
  return identity.kind == PayloadIdentityKind::kTensor || layout.has_value();
}

[[nodiscard]] expected<void, Error> validate_payload_options(
    const PayloadOptionsRecord& options, const PayloadIdentityRecord& identity,
    const std::optional<TensorPortLayout>& layout) {
  const bool inline_policy = options.interface_tensor_emit_policy ==
                             InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded;
  if (!closed_enum(options.interface_tensor_emit_policy) ||
      (options.max_inline_bytes && *options.max_inline_bytes == 0U) ||
      (options.max_tensor_byte_span && *options.max_tensor_byte_span == 0U) ||
      (inline_policy != options.max_inline_bytes.has_value())) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_INLINE_BOUND_INVALID"));
  }
  if (options.schema) {
    auto valid = validate_schema(*options.schema);
    if (!valid)
      return valid;
    if (identity.kind == PayloadIdentityKind::kSchema) {
      if (!identity.schema || !schema_identity_equal(*options.schema, *identity.schema)) {
        return make_unexpected(
            graph_error(ErrorCode::kIncompatible, "GRAPH_PAYLOAD_SCHEMA_MISMATCH"));
      }
    } else if (identity.kind != PayloadIdentityKind::kTensor) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_PAYLOAD_SCHEMA_IRRELEVANT"));
    }
  }
  if (identity.kind == PayloadIdentityKind::kSchema &&
      (!options.schema || !identity.schema ||
       !schema_identity_equal(*options.schema, *identity.schema))) {
    return make_unexpected(graph_error(ErrorCode::kIncompatible, "GRAPH_PAYLOAD_SCHEMA_MISMATCH"));
  }
  const bool nondefault_tensor_policy =
      options.interface_tensor_emit_policy != InterfaceTensorEmitPolicy::kRequireExportable ||
      options.max_inline_bytes.has_value() || options.max_tensor_byte_span.has_value();
  if (identity.kind == PayloadIdentityKind::kSchema && nondefault_tensor_policy && !layout) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_NON_TENSOR_PAYLOAD_OPTIONS"));
  }
  if (identity.kind != PayloadIdentityKind::kSchema && !tensor_bearing(identity, layout)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_NON_TENSOR_PAYLOAD_OPTIONS"));
  }
  return {};
}

[[nodiscard]] bool payload_identity_equal(const PayloadIdentityRecord& lhs,
                                          const PayloadIdentityRecord& rhs) {
  if (lhs.kind != rhs.kind || lhs.payload_kind != rhs.payload_kind)
    return false;
  if (lhs.scalar.has_value() != rhs.scalar.has_value() ||
      lhs.schema.has_value() != rhs.schema.has_value() ||
      lhs.pod.has_value() != rhs.pod.has_value()) {
    return false;
  }
  if (lhs.scalar && lhs.scalar->kind != rhs.scalar->kind)
    return false;
  if (lhs.schema && (lhs.schema->canonical_name != rhs.schema->canonical_name ||
                     lhs.schema->version != rhs.schema->version ||
                     lhs.schema->schema_sha256 != rhs.schema->schema_sha256 ||
                     lhs.schema->id != rhs.schema->id)) {
    return false;
  }
  if (lhs.pod) {
    if (lhs.pod->canonical_name != rhs.pod->canonical_name ||
        lhs.pod->version != rhs.pod->version || lhs.pod->object_size != rhs.pod->object_size ||
        lhs.pod->object_alignment != rhs.pod->object_alignment ||
        lhs.pod->fields.size() != rhs.pod->fields.size()) {
      return false;
    }
    for (std::size_t index = 0; index < lhs.pod->fields.size(); ++index) {
      const auto& left = lhs.pod->fields[index];
      const auto& right = rhs.pod->fields[index];
      if (left.name != right.name || left.byte_offset != right.byte_offset ||
          left.scalar_kind != right.scalar_kind || left.element_count != right.element_count) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] expected<void, Error> validate_pod_catalog(const GraphState& state,
                                                         const PayloadIdentityRecord& identity) {
  if (!identity.pod)
    return {};
  const auto& candidate = identity.pod.value();
  for (const auto& existing : state.pod_identity_catalog_) {
    if (!existing.pod)
      continue;
    const auto& prior = existing.pod.value();
    if (prior.canonical_name == candidate.canonical_name && prior.version == candidate.version) {
      if (!payload_identity_equal(existing, identity)) {
        return make_unexpected(
            graph_error(ErrorCode::kIncompatible, "GRAPH_POD_IDENTITY_CONFLICT"));
      }
      return {};
    }
  }
  return {};
}

void record_pod_identity(GraphState& state, const PayloadIdentityRecord& identity) {
  if (!identity.pod)
    return;
  const auto& candidate = identity.pod.value();
  const bool present = std::any_of(state.pod_identity_catalog_.begin(),
                                   state.pod_identity_catalog_.end(),
                                   [&](const auto& item) {
                                     return item.pod &&
                                            item.pod->canonical_name == candidate.canonical_name &&
                                            item.pod->version == candidate.version;
                                   });
  if (!present)
    state.pod_identity_catalog_.push_back(identity);
}

[[nodiscard]] std::optional<SchemaIdentityRecord> copy_schema(
    const std::optional<SchemaIdentity>& source) {
  if (!source)
    return std::nullopt;
  return SchemaIdentityRecord{
      .canonical_name = std::string(source->canonical_name),
      .version = source->version,
      .schema_sha256 = source->schema_sha256,
      .id = source->id,
  };
}

[[nodiscard]] ConnectionOptionsRecord copy_connection_options(const ConnectionOptions& source) {
  return ConnectionOptionsRecord{
      .queue_policy = source.queue_policy,
      .queue_depth = source.queue_depth,
      .topic_name =
          source.topic_name ? std::optional<std::string>{*source.topic_name} : std::nullopt,
      .memory_conversion = source.memory_conversion,
      .max_inflight_samples = source.max_inflight_samples,
  };
}

[[nodiscard]] expected<void, Error> validate_connection_options(const ConnectionOptions& options) {
  if (!closed_enum(options.queue_policy) ||
      (options.memory_conversion && !closed_enum(*options.memory_conversion)) ||
      (options.queue_depth && *options.queue_depth == 0U) ||
      (options.max_inflight_samples && *options.max_inflight_samples == 0U)) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_FLOW_BOUND_INVALID"));
  }
  if (options.topic_name) {
    return validate_name(*options.topic_name, "GRAPH_FLOW_TOPIC_NAME_INVALID");
  }
  return {};
}

[[nodiscard]] PortStorage* find_port(GraphState& state, GraphPortId id) noexcept {
  if (!id || id.value > state.ports_.size())
    return nullptr;
  PortStorage& port = state.ports_[static_cast<std::size_t>(id.value - 1U)];
  if (port.record.id != id) {
    assert(false && "Graph port IDs must remain dense");
    return nullptr;
  }
  return &port;
}

[[nodiscard]] const PortStorage* find_port(const GraphState& state, GraphPortId id) noexcept {
  if (!id || id.value > state.ports_.size())
    return nullptr;
  const PortStorage& port = state.ports_[static_cast<std::size_t>(id.value - 1U)];
  if (port.record.id != id) {
    assert(false && "Graph port IDs must remain dense");
    return nullptr;
  }
  return &port;
}

[[nodiscard]] EdgeRecord* find_edge(GraphState& state, EdgeId id) noexcept {
  if (!id || id.value > state.edges_.size())
    return nullptr;
  EdgeRecord& edge = state.edges_[static_cast<std::size_t>(id.value - 1U)];
  if (edge.id != id) {
    assert(false && "Graph edge IDs must remain dense");
    return nullptr;
  }
  return &edge;
}

[[nodiscard]] expected<GraphPortId, Error> find_port_by_declaration(
    GraphState& state, const void* declaration, PortDirection direction,
    const PayloadIdentityRecord& payload_identity) {
  const auto found = state.port_by_declaration_.find(declaration);
  if (found == state.port_by_declaration_.end()) {
    return make_unexpected(graph_error(ErrorCode::kNotFound, "GRAPH_PORT_NOT_FOUND"));
  }
  PortStorage* port = find_port(state, found->second);
  if (port == nullptr || port->record.direction != direction ||
      !payload_identity_equal(port->record.payload_identity, payload_identity)) {
    return make_unexpected(graph_error(ErrorCode::kIncompatible, "GRAPH_PORT_TYPE_MISMATCH"));
  }
  return found->second;
}

[[nodiscard]] expected<void, Error> run_setup(OperatorBase& prototype, OperatorSpec& spec) {
  try {
    prototype.setup(spec);
  } catch (const RuntimeError& error) {
    graph_log_noexcept(
        [&] { HOLOSCAN_LOG_DEBUG("Operator setup RuntimeError detail: {}", error.what()); });
    return make_unexpected(Error{error.code(), "GRAPH_OPERATOR_SETUP_RUNTIME_ERROR"});
  } catch (const std::exception& exception) {
    graph_log_noexcept(
        [&] { HOLOSCAN_LOG_DEBUG("Operator setup exception detail: {}", exception.what()); });
    return make_unexpected(Error{ErrorCode::kFailure, "GRAPH_OPERATOR_SETUP_FAILED"});
  } catch (...) {
    return make_unexpected(graph_error(ErrorCode::kFailure, "GRAPH_OPERATOR_SETUP_FAILED"));
  }
  if (OperatorSpecCompilerAccess::declaration_error(spec).code != ErrorCode::kSuccess) {
    return make_unexpected(OperatorSpecCompilerAccess::declaration_error(spec));
  }
  return {};
}

}  // namespace

GraphState::GraphState(Graph* owner, std::string name)
    : owner_(owner),
      graph_record_{.id = GraphId{1}, .parent = std::nullopt, .name = std::move(name)} {
  const std::uint64_t sequence = g_next_graph_instance.fetch_add(1U, std::memory_order_relaxed);
  for (std::size_t index = 0; index < sizeof(sequence); ++index) {
    instance_id_.value[index] =
        static_cast<std::byte>((sequence >> static_cast<unsigned>(index * 8U)) & 0xFFU);
  }
  // The second word is a fixed domain discriminator, not an address or implementation token.
  constexpr std::array<std::byte, 8> kDomain{std::byte{'H'},
                                             std::byte{'S'},
                                             std::byte{'D'},
                                             std::byte{'K'},
                                             std::byte{'G'},
                                             std::byte{'R'},
                                             std::byte{'P'},
                                             std::byte{'H'}};
  std::copy(kDomain.begin(), kDomain.end(), instance_id_.value.begin() + sizeof(sequence));
}

expected<void, Error> GraphState::prepare_mutation() {
  std::unique_lock lock(mutex_);
  while (composition_state_ == CompositionState::kComposing &&
         composing_thread_ != std::this_thread::get_id()) {
    composition_cv_.wait(lock);
  }
  if (composition_state_ == CompositionState::kComposed)
    return {};
  if (composition_state_ == CompositionState::kFailed) {
    return make_unexpected(composition_diagnostic_->primary);
  }
  if (composition_state_ == CompositionState::kComposing)
    return {};
  if (owner_ == nullptr) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  }
  composition_state_ = CompositionState::kComposing;
  composing_thread_ = std::this_thread::get_id();
  Graph* owner = owner_;
  lock.unlock();

  Error failure{ErrorCode::kSuccess};
  std::optional<Error> failure_cause;
  std::string failure_context;
  const auto capture_failure = [&](auto&& capture) noexcept {
    try {
      if (fail_composition_diagnostic_capture_) {
        fail_composition_diagnostic_capture_ = false;
        throw std::bad_alloc{};
      }
      std::forward<decltype(capture)>(capture)();
    } catch (...) {
      failure.code = ErrorCode::kFailure;
      failure.message.clear();
      failure_cause.reset();
      failure_context.clear();
    }
  };
  try {
    owner->compose();
  } catch (const RuntimeError& error) {
    capture_failure([&] {
      graph_log_noexcept(
          [&] { HOLOSCAN_LOG_DEBUG("Graph compose RuntimeError detail: {}", error.what()); });
      Error cause = make_error(error);
      if (cause.message == "GRAPH_COMPOSE_REENTRANT" || cause.message == "GRAPH_EMBED_CYCLE") {
        failure = std::move(cause);
      } else {
        failure = graph_error(ErrorCode::kFailure, "GRAPH_COMPOSE_FAILED");
        failure_cause = std::move(cause);
      }
    });
  } catch (const std::exception& exception) {
    capture_failure([&] {
      failure_context = bounded_exception_context(exception.what());
      graph_log_noexcept(
          [&] { HOLOSCAN_LOG_DEBUG("Graph compose exception detail: {}", failure_context); });
      failure = graph_error(ErrorCode::kFailure, "GRAPH_COMPOSE_EXCEPTION");
    });
  } catch (...) {
    capture_failure(
        [&] { failure = graph_error(ErrorCode::kFailure, "GRAPH_COMPOSE_UNKNOWN_EXCEPTION"); });
  }

  lock.lock();
  composing_thread_ = {};
  if (failure.code != ErrorCode::kSuccess) {
    static_assert(std::is_nothrow_move_constructible_v<GraphCompositionDiagnostic>);
    GraphCompositionDiagnostic diagnostic{
        .primary = std::move(failure),
        .cause = std::move(failure_cause),
        .context = std::move(failure_context),
    };
    composition_diagnostic_.emplace(std::move(diagnostic));
    composition_state_ = CompositionState::kFailed;
    clear_failed_draft();
    composition_cv_.notify_all();
    graph_log_noexcept([&] {
      HOLOSCAN_LOG_ERROR("Graph composition failed: {}",
                         runtime_error_message(composition_diagnostic_->primary));
    });
    return make_unexpected(composition_diagnostic_->primary);
  }
  composition_state_ = CompositionState::kComposed;
  ++generation_;
  composition_cv_.notify_all();
  graph_log_noexcept(
      [&] { HOLOSCAN_LOG_TRACE("Graph composition completed at generation {}", generation_); });
  return {};
}

expected<void, Error> GraphState::prepare_observation() {
  {
    const std::lock_guard lock(mutex_);
    if (composition_state_ == CompositionState::kComposing &&
        composing_thread_ == std::this_thread::get_id()) {
      return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_COMPOSE_REENTRANT"));
    }
  }
  return prepare_mutation();
}

void GraphState::mark_mutation() noexcept {
  if (!(composition_state_ == CompositionState::kComposing &&
        composing_thread_ == std::this_thread::get_id())) {
    ++generation_;
  }
}

void GraphState::detach() noexcept {
  const std::lock_guard lock(mutex_);
  owner_ = nullptr;
}

void GraphState::clear_failed_draft() noexcept {
  operators_.clear();
  vertex_adjacency_.clear();
  cycle_visit_marks_.clear();
  cycle_visit_epoch_ = 0U;
  embedded_graph_records_.clear();
  embedded_operator_mappings_.clear();
  embedded_graphs_.clear();
  vertices_.clear();
  ports_.clear();
  port_by_declaration_.clear();
  edges_.clear();
  interface_bindings_.clear();
  external_topics_.clear();
  readiness_sources_.clear();
  regions_.clear();
  partitions_.clear();
  clocks_.clear();
  clock_imports_.clear();
  clock_bindings_.clear();
  clock_conversions_.clear();
  lifecycle_dependencies_.clear();
  lifecycle_callbacks_.clear();
  lifecycle_facts_.clear();
  lifecycle_restart_permissions_.clear();
  vertex_names_.clear();
  interface_port_names_.clear();
  region_names_.clear();
  partition_names_.clear();
  clock_names_.clear();
  default_clock_ = ClockKey{};
  explicit_endpoint_names_.clear();
  embedded_instance_names_.clear();
  pod_identity_catalog_.clear();
}

namespace {

[[nodiscard]] PortConfigurationRecord setup_configuration(
    const OperatorSpecCompilerAccess::PortMetadata& metadata) {
  PortConfigurationRecord result;
  if (metadata.direction == PortDirection::kInput) {
    result.queue_depth = metadata.queue_depth;
    result.memory_conversion = metadata.memory_conversion;
    result.payload_options = metadata.payload_options;
  } else {
    result.max_emits_per_compute = metadata.max_emits_per_compute;
    result.late_joiner_history_depth = metadata.late_joiner_history_depth;
    result.tensor_allocation = metadata.tensor_allocation;
    result.payload_options = metadata.payload_options;
  }
  if (!metadata.diagnostic_alias.empty()) {
    result.diagnostic_alias = std::string(metadata.diagnostic_alias);
  }
  result.tensor_layout = metadata.tensor_layout;
  if (metadata.descriptor_tensor_field_count == 1U && metadata.tensor_fields.size() == 1U) {
    const auto& field = metadata.tensor_fields.front().configuration;
    result.tensor_layout = field.tensor_layout;
    if (field.payload_options) {
      result.payload_options = PayloadOptionsRecord{
          .max_inline_bytes = field.payload_options->max_inline_bytes
                                  ? std::optional{static_cast<std::uint32_t>(
                                        *field.payload_options->max_inline_bytes)}
                                  : std::nullopt,
          .max_tensor_byte_span = field.payload_options->max_tensor_byte_span,
          .interface_tensor_emit_policy = field.payload_options->interface_tensor_emit_policy,
      };
    }
    if (metadata.direction == PortDirection::kInput) {
      result.memory_conversion = field.memory_conversion;
    } else {
      result.tensor_allocation = field.tensor_allocation;
    }
    return result;
  }
  result.tensor_fields.reserve(metadata.tensor_fields.size());
  for (const auto& field : metadata.tensor_fields) {
    result.tensor_fields.push_back(field.configuration);
  }
  std::sort(result.tensor_fields.begin(),
            result.tensor_fields.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.field_id < rhs.field_id; });
  return result;
}

[[nodiscard]] bool valid_tensor_field_payload_options(
    const TensorFieldPayloadOptions& options) noexcept {
  const bool inline_policy = options.interface_tensor_emit_policy ==
                             InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded;
  return closed_enum(options.interface_tensor_emit_policy) &&
         options.max_inline_bytes.value_or(1U) != 0U &&
         options.max_tensor_byte_span.value_or(1U) != 0U &&
         inline_policy == options.max_inline_bytes.has_value();
}

[[nodiscard]] expected<void, Error> validate_one_tensor_field_metadata(
    const OperatorSpecCompilerAccess::PortMetadata& metadata,
    const AuthoredTensorFieldConfiguration& authored, std::vector<std::uint16_t>& field_ids,
    std::vector<std::uint16_t>& tensor_indices) {
  const auto& field = authored.configuration;
  if (authored.schema_type_token != metadata.local_type_token) {
    return make_unexpected(tensor_field_graph_error(
        ErrorCode::kInvalidArgument, "GRAPH_TENSOR_FIELD_SCHEMA_MISMATCH", field));
  }
  if (!authored.selector_valid || field.tensor_index >= metadata.descriptor_tensor_field_count) {
    return make_unexpected(
        tensor_field_graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_FIELD_UNKNOWN", field));
  }
  if (std::find(field_ids.begin(), field_ids.end(), field.field_id) != field_ids.end() ||
      std::find(tensor_indices.begin(), tensor_indices.end(), field.tensor_index) !=
          tensor_indices.end()) {
    return make_unexpected(
        tensor_field_graph_error(ErrorCode::kDuplicateName, "GRAPH_TENSOR_FIELD_DUPLICATE", field));
  }
  field_ids.push_back(field.field_id);
  tensor_indices.push_back(field.tensor_index);

  if ((field.memory_conversion && !closed_enum(*field.memory_conversion)) ||
      (field.tensor_layout && field.tensor_layout->memory_kind &&
       !closed_enum(*field.tensor_layout->memory_kind)) ||
      (field.payload_options && !valid_tensor_field_payload_options(*field.payload_options))) {
    return make_unexpected(tensor_field_graph_error(
        ErrorCode::kInvalidArgument, "GRAPH_TENSOR_FIELD_ENUM_INVALID", field));
  }
  if (metadata.descriptor_tensor_field_count == 1U && field.payload_options &&
      field.payload_options->max_inline_bytes &&
      *field.payload_options->max_inline_bytes > std::numeric_limits<std::uint32_t>::max()) {
    return make_unexpected(tensor_field_graph_error(
        ErrorCode::kInvalidArgument, "GRAPH_TENSOR_FIELD_ENUM_INVALID", field));
  }
  if (field.tensor_layout && field.tensor_layout->dtype &&
      !valid_dtype(*field.tensor_layout->dtype)) {
    return make_unexpected(tensor_field_graph_error(
        ErrorCode::kInvalidArgument, "GRAPH_TENSOR_FIELD_DTYPE_INVALID", field));
  }
  if (field.tensor_allocation &&
      (field.tensor_allocation->max_rank == 0U || field.tensor_allocation->max_byte_span == 0U)) {
    if (field.tensor_allocation->max_rank == 0U && field.tensor_layout &&
        !field.tensor_layout->rank) {
      return make_unexpected(tensor_field_graph_error(
          ErrorCode::kInvalidArgument, "GRAPH_TENSOR_FIELD_RANK_BOUND_UNRESOLVED", field));
    }
    return make_unexpected(tensor_field_graph_error(
        ErrorCode::kInvalidArgument, "GRAPH_TENSOR_FIELD_ALLOCATION_INVALID", field));
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_tensor_field_metadata(
    const OperatorSpecCompilerAccess::PortMetadata& metadata) {
  if (metadata.descriptor_tensor_field_count == 0U) {
    if (!metadata.tensor_fields.empty()) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_FIELD_SCHEMA_INVALID"));
    }
    return {};
  }
  if (metadata.descriptor_tensor_field_count > 1U &&
      (metadata.tensor_layout || metadata.memory_conversion || metadata.tensor_allocation ||
       metadata.payload_options)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_FIELD_SCOPE_REQUIRED"));
  }
  if (metadata.descriptor_tensor_field_count == 1U && !metadata.tensor_fields.empty() &&
      (metadata.tensor_layout || metadata.memory_conversion || metadata.tensor_allocation ||
       metadata.payload_options)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_FIELD_SCOPE_CONFLICT"));
  }

  std::vector<std::uint16_t> field_ids;
  std::vector<std::uint16_t> tensor_indices;
  field_ids.reserve(metadata.tensor_fields.size());
  tensor_indices.reserve(metadata.tensor_fields.size());
  for (const auto& authored : metadata.tensor_fields) {
    auto valid = validate_one_tensor_field_metadata(metadata, authored, field_ids, tensor_indices);
    if (!valid) {
      return valid;
    }
  }
  return {};
}

// Port metadata validation is kept as one ordered fail-closed funnel so each declaration is
// rejected before any Graph-owned state is committed.
[[nodiscard]] expected<void, Error>
validate_port_metadata(  // NOLINT(readability-function-cognitive-complexity)
    const OperatorSpecCompilerAccess::PortMetadata& metadata) {
  auto tensor_fields = validate_tensor_field_metadata(metadata);
  if (!tensor_fields)
    return tensor_fields;
  auto name = validate_name(metadata.name, "GRAPH_PORT_NAME_INVALID");
  if (!name)
    return name;
  auto identity = validate_payload_identity(metadata.payload_identity);
  if (!identity)
    return identity;
  const bool schema = metadata.payload_identity.kind == PayloadIdentityKind::kSchema;
  if (schema != (metadata.serialized_payload_capacity_bytes != 0U) ||
      (schema &&
       (metadata.runtime_type_ops == nullptr ||
        (metadata.runtime_type_ops->codec_family != RuntimePayloadCodecFamily::kGeneratedSchema &&
         metadata.runtime_type_ops->codec_family !=
             RuntimePayloadCodecFamily::kDescriptorTensorSchema) ||
        metadata.runtime_type_ops->wire_capacity != metadata.serialized_payload_capacity_bytes))) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_SCHEMA_CAPACITY_INVALID"));
  }
  if ((metadata.direction == PortDirection::kInput && metadata.queue_depth == 0U) ||
      (metadata.direction == PortDirection::kOutput && metadata.max_emits_per_compute == 0U)) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_PORT_BOUND_INVALID"));
  }
  if ((metadata.memory_conversion && !closed_enum(*metadata.memory_conversion)) ||
      (metadata.tensor_layout && metadata.tensor_layout->memory_kind &&
       !closed_enum(*metadata.tensor_layout->memory_kind)) ||
      (metadata.payload_options &&
       !closed_enum(metadata.payload_options->interface_tensor_emit_policy))) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_PORT_ENUM_INVALID"));
  }
  if (metadata.tensor_layout && metadata.tensor_layout->dtype &&
      !valid_dtype(*metadata.tensor_layout->dtype)) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_DTYPE_INVALID"));
  }
  if (!metadata.diagnostic_alias.empty()) {
    auto alias = validate_name(metadata.diagnostic_alias, "GRAPH_PORT_ALIAS_INVALID");
    if (!alias)
      return alias;
  }
  if (metadata.tensor_allocation && (metadata.tensor_allocation->max_rank == 0U ||
                                     metadata.tensor_allocation->max_byte_span == 0U)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_ALLOCATION_INVALID"));
  }
  if (metadata.payload_options) {
    auto options = validate_payload_options(
        *metadata.payload_options, metadata.payload_identity, metadata.tensor_layout);
    if (!options)
      return options;
  }
  return {};
}

[[nodiscard]] bool schema_equal(const std::optional<SchemaIdentityRecord>& lhs,
                                const std::optional<SchemaIdentityRecord>& rhs) {
  if (lhs.has_value() != rhs.has_value())
    return false;
  return !lhs || (lhs->canonical_name == rhs->canonical_name && lhs->version == rhs->version &&
                  lhs->schema_sha256 == rhs->schema_sha256 && lhs->id == rhs->id);
}

[[nodiscard]] bool payload_options_equal(const std::optional<PayloadOptionsRecord>& lhs,
                                         const std::optional<PayloadOptionsRecord>& rhs) {
  if (lhs.has_value() != rhs.has_value())
    return false;
  return !lhs || (schema_equal(lhs->schema, rhs->schema) &&
                  lhs->max_inline_bytes == rhs->max_inline_bytes &&
                  lhs->max_tensor_byte_span == rhs->max_tensor_byte_span &&
                  lhs->interface_tensor_emit_policy == rhs->interface_tensor_emit_policy);
}

[[nodiscard]] bool port_setup_equal(const PortStorage& existing,
                                    const OperatorSpecCompilerAccess::PortMetadata& candidate) {
  const auto candidate_setup = setup_configuration(candidate);
  const auto& setup = existing.record.setup;
  return existing.record.direction == candidate.direction &&
         existing.record.name == candidate.name &&
         existing.local_type_token == candidate.local_type_token &&
         payload_identity_equal(existing.record.payload_identity, candidate.payload_identity) &&
         existing.record.serialized_payload_capacity_bytes ==
             candidate.serialized_payload_capacity_bytes &&
         existing.record.generic_loan_supported ==
             (candidate.direction == PortDirection::kOutput &&
              candidate.runtime_type_ops != nullptr &&
              candidate.runtime_type_ops->generic_loan_supported) &&
         existing.record.generic_loan_slot_size_bytes ==
             (candidate.direction == PortDirection::kOutput &&
                      candidate.runtime_type_ops != nullptr &&
                      candidate.runtime_type_ops->generic_loan_supported
                  ? static_cast<std::uint64_t>(
                        candidate.runtime_type_ops->generic_loan_slot_size_bytes)
                  : 0U) &&
         existing.record.descriptor_tensor_field_count == candidate.descriptor_tensor_field_count &&
         setup.queue_depth == candidate_setup.queue_depth &&
         setup.max_emits_per_compute == candidate_setup.max_emits_per_compute &&
         setup.late_joiner_history_depth == candidate_setup.late_joiner_history_depth &&
         setup.diagnostic_alias == candidate_setup.diagnostic_alias &&
         tensor_layout_equal(setup.tensor_layout, candidate_setup.tensor_layout) &&
         setup.memory_conversion == candidate_setup.memory_conversion &&
         setup.tensor_allocation.has_value() == candidate_setup.tensor_allocation.has_value() &&
         (!setup.tensor_allocation ||
          (setup.tensor_allocation->max_rank == candidate_setup.tensor_allocation->max_rank &&
           setup.tensor_allocation->max_byte_span ==
               candidate_setup.tensor_allocation->max_byte_span)) &&
         payload_options_equal(setup.payload_options, candidate_setup.payload_options) &&
         tensor_field_configuration_equal(setup.tensor_fields, candidate_setup.tensor_fields);
}

[[nodiscard]] bool clock_key_equal(const ClockKey& lhs, const ClockKey& rhs) {
  return lhs == rhs;
}

[[nodiscard]] bool source_equal(const ReadinessSourceDeclarationRecord& existing,
                                const OperatorSpecCompilerAccess::SourceMetadata& candidate) {
  return existing.kind == candidate.kind && existing.name == candidate.name &&
         existing.requested_capacity == candidate.capacity &&
         existing.requested_sender_reference_capacity == candidate.sender_reference_capacity &&
         clock_key_equal(existing.requested_clock, candidate.clock) &&
         existing.wake_requirement == (candidate.kind == ReadinessSourceKind::kSpin
                                           ? ReadinessWakeRequirement::kBusyPollRequired
                                           : ReadinessWakeRequirement::kDeploymentSelected);
}

[[nodiscard]] bool lifecycle_callback_equal(
    const LifecycleCallbackAuthoringRecord& existing,
    const OperatorSpecCompilerAccess::LifecycleCallbackMetadata& candidate) noexcept {
  return existing.stage == candidate.stage && existing.required == candidate.required &&
         existing.method == candidate.method &&
         existing.origin == LifecycleCallbackOrigin::kAdvancedMember;
}

[[nodiscard]] constexpr LifecycleMethodId lifecycle_method_id(LifecycleStage stage) noexcept {
  return static_cast<LifecycleMethodId>(static_cast<std::uint8_t>(stage) + 1U);
}

[[nodiscard]] bool lifecycle_callback_equal_legacy(const LifecycleCallbackAuthoringRecord& existing,
                                                   LifecycleStage stage,
                                                   LifecycleCallbackOrigin origin) noexcept {
  return existing.stage == stage && existing.required &&
         existing.method == lifecycle_method_id(stage) && existing.origin == origin;
}

[[nodiscard]] bool has_lifecycle_callback(const OperatorSpec& spec, LifecycleStage stage) noexcept {
  for (std::size_t index = 0; index < OperatorSpecCompilerAccess::lifecycle_callback_count(spec);
       ++index) {
    if (OperatorSpecCompilerAccess::lifecycle_callback(spec, index).stage == stage)
      return true;
  }
  return false;
}

[[nodiscard]] bool lifecycle_fact_equal(
    const LifecycleFactAuthoringRecord& existing,
    const OperatorSpecCompilerAccess::LifecycleFactMetadata& candidate) noexcept {
  static_assert(
      static_cast<std::uint8_t>(LifecycleFactKind::kProduces) ==
      static_cast<std::uint8_t>(OperatorSpecCompilerAccess::LifecycleFactKind::kProduces));
  static_assert(
      static_cast<std::uint8_t>(LifecycleFactKind::kRequires) ==
      static_cast<std::uint8_t>(OperatorSpecCompilerAccess::LifecycleFactKind::kRequires));
  const auto candidate_kind = static_cast<LifecycleFactKind>(candidate.kind);
  return existing.stage == candidate.stage && existing.kind == candidate_kind &&
         existing.stable_name == candidate.stable_name;
}

[[nodiscard]] expected<void, Error> validate_lifecycle_metadata(const OperatorSpec& spec,
                                                                const void* operator_type_token) {
  for (std::size_t index = 0; index < OperatorSpecCompilerAccess::lifecycle_callback_count(spec);
       ++index) {
    const auto callback = OperatorSpecCompilerAccess::lifecycle_callback(spec, index);
    if (!closed_enum(callback.stage)) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_STAGE_INVALID"));
    }
    if (callback.declaring_type_token != operator_type_token) {
      return make_unexpected(
          graph_error(ErrorCode::kIncompatible, "GRAPH_LIFECYCLE_CALLBACK_OWNER_MISMATCH"));
    }
  }
  for (std::size_t index = 0; index < OperatorSpecCompilerAccess::lifecycle_fact_count(spec);
       ++index) {
    const auto fact = OperatorSpecCompilerAccess::lifecycle_fact(spec, index);
    if (!closed_enum(fact.stage)) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_STAGE_INVALID"));
    }
    if (!closed_enum(fact.kind)) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_FACT_KIND_INVALID"));
    }
    auto fact_name = validate_name(fact.stable_name, "GRAPH_LIFECYCLE_KEY_NAME_INVALID");
    if (!fact_name)
      return make_unexpected(std::move(fact_name.error()));
  }
  return {};
}

}  // namespace

expected<void, Error> graph_prepare_authoring(const std::shared_ptr<GraphState>& state) {
  if (!state)
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  return state->prepare_mutation();
}

expected<void, Error> graph_prepare_embed(const std::shared_ptr<GraphState>& state,
                                          std::string_view instance_name) {
  if (!state)
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
  auto valid = validate_name(instance_name, "GRAPH_EMBED_INSTANCE_NAME_INVALID");
  if (!valid)
    return valid;
  // Structured paths retain decoded segments. Dot segments would invite
  // normalization aliases, while the exact top-level `_root` spelling is
  // reserved for rendering the root boundary.
  if (instance_name == "." || instance_name == ".." || instance_name == "_root") {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_EMBED_INSTANCE_PATH_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  if (state->embedded_instance_names_.contains(std::string(instance_name))) {
    return make_unexpected(
        graph_error(ErrorCode::kDuplicateName, "GRAPH_EMBED_INSTANCE_DUPLICATE"));
  }
  return {};
}

// Registration intentionally keeps validation, ownership transfer, and rollback-sensitive state
// publication in one ordered transaction.
expected<OperatorRegistration, Error>
graph_register_operator(  // NOLINT(readability-function-cognitive-complexity)
    const std::shared_ptr<GraphState>& state, std::string_view name,
    std::shared_ptr<OperatorFactoryBase> factory, std::unique_ptr<OperatorBase> prototype,
    const void* operator_type_token) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return make_unexpected(std::move(prepared.error()));
  auto valid_name = validate_name(name, "GRAPH_OPERATOR_NAME_INVALID");
  if (!valid_name)
    return make_unexpected(std::move(valid_name.error()));
  if (!factory || !prototype || operator_type_token == nullptr) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_OPERATOR_FACTORY_INVALID"));
  }
  const OperatorFactoryBase::LifecycleHookSet lifecycle_hooks = factory->lifecycle_hooks();

  OperatorSpec spec;
  auto setup = run_setup(*prototype, spec);
  if (!setup)
    return make_unexpected(std::move(setup.error()));
  for (std::size_t index = 0; index < OperatorSpecCompilerAccess::port_count(spec); ++index) {
    auto valid = validate_port_metadata(OperatorSpecCompilerAccess::port(spec, index));
    if (!valid)
      return make_unexpected(std::move(valid.error()));
  }
  for (std::size_t index = 0; index < OperatorSpecCompilerAccess::source_count(spec); ++index) {
    const auto source = OperatorSpecCompilerAccess::source(spec, index);
    auto source_name = validate_name(source.name, "GRAPH_SOURCE_NAME_INVALID");
    if (!source_name)
      return make_unexpected(std::move(source_name.error()));
    if (!closed_enum(source.kind)) {
      return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_SOURCE_KIND_INVALID"));
    }
    const bool single =
        source.kind == ReadinessSourceKind::kReadable || source.kind == ReadinessSourceKind::kSpin;
    if ((single && source.capacity != 1U) || (!single && source.capacity == 0U) ||
        (source.kind == ReadinessSourceKind::kNotification &&
         source.sender_reference_capacity == 0U)) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_SOURCE_BOUND_INVALID"));
    }
  }
  auto valid_lifecycle = validate_lifecycle_metadata(spec, operator_type_token);
  if (!valid_lifecycle)
    return make_unexpected(std::move(valid_lifecycle.error()));
  if ((lifecycle_hooks.start && has_lifecycle_callback(spec, LifecycleStage::kStart)) ||
      (lifecycle_hooks.stop && has_lifecycle_callback(spec, LifecycleStage::kStop))) {
    return make_unexpected(
        graph_error(ErrorCode::kDuplicateName, "GRAPH_LIFECYCLE_CALLBACK_DUPLICATE"));
  }
  const std::lock_guard lock(state->mutex_);
  if (state->vertex_names_.contains(std::string(name))) {
    return make_unexpected(graph_error(ErrorCode::kDuplicateName, "GRAPH_OPERATOR_NAME_DUPLICATE"));
  }
  for (std::size_t index = 0; index < OperatorSpecCompilerAccess::port_count(spec); ++index) {
    const auto identity = OperatorSpecCompilerAccess::port(spec, index).payload_identity;
    auto catalog = validate_pod_catalog(*state, identity);
    if (!catalog)
      return make_unexpected(std::move(catalog.error()));
    if (identity.pod) {
      const auto& candidate = identity.pod.value();
      for (std::size_t prior = 0; prior < index; ++prior) {
        const auto prior_identity = OperatorSpecCompilerAccess::port(spec, prior).payload_identity;
        if (prior_identity.pod && prior_identity.pod->canonical_name == candidate.canonical_name &&
            prior_identity.pod->version == candidate.version &&
            !payload_identity_equal(prior_identity, identity)) {
          return make_unexpected(
              graph_error(ErrorCode::kIncompatible, "GRAPH_POD_IDENTITY_CONFLICT"));
        }
      }
    }
  }

  std::unordered_set<const void*> candidate_declarations;
  for (std::size_t index = 0; index < OperatorSpecCompilerAccess::port_count(spec); ++index) {
    const auto metadata = OperatorSpecCompilerAccess::port(spec, index);
    if (!candidate_declarations.emplace(metadata.declaration).second) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_PORT_DECLARATION_DUPLICATE"));
    }
    if (metadata.direction == PortDirection::kInput) {
      const auto& input = *static_cast<const InputBase*>(metadata.declaration);
      if (GraphPortOwnershipAccess::owner(input) != nullptr) {
        return make_unexpected(
            graph_error(ErrorCode::kInvalidArgument, "GRAPH_PORT_ALREADY_OWNED"));
      }
    } else {
      const auto& output = *static_cast<const OutputBase*>(metadata.declaration);
      if (GraphPortOwnershipAccess::owner(output) != nullptr) {
        return make_unexpected(
            graph_error(ErrorCode::kInvalidArgument, "GRAPH_PORT_ALREADY_OWNED"));
      }
    }
  }

  OperatorStorage storage;
  GraphMutationRollback rollback(*state);
  const std::size_t operator_index = state->operators_.size();
  const VertexId vertex = state->next_vertex_id();
  storage = OperatorStorage{
      .name = std::string(name),
      .operator_type_token = operator_type_token,
      .factory = std::move(factory),
      .prototype = std::move(prototype),
      .vertex = vertex,
  };
  storage.ports.reserve(OperatorSpecCompilerAccess::port_count(spec));

  state->vertices_.push_back(VertexRecord{
      .id = vertex,
      .graph = state->graph_record_.id,
      .kind = VertexKind::kOperator,
      .name = std::string(name),
  });
  GraphMutationRollback::checkpoint(*state);
  state->vertex_names_.emplace(name);
  GraphMutationRollback::checkpoint(*state);
  for (std::size_t index = 0; index < OperatorSpecCompilerAccess::port_count(spec); ++index) {
    const auto metadata = OperatorSpecCompilerAccess::port(spec, index);
    const GraphPortId port_id = state->next_port_id();
    PortStorage port{
        .record =
            PortRecord{
                .id = port_id,
                .owner = vertex,
                .owner_kind = PortOwnerKind::kOperator,
                .direction = metadata.direction,
                .name = std::string(metadata.name),
                .payload_identity = metadata.payload_identity,
                .serialized_payload_capacity_bytes = metadata.serialized_payload_capacity_bytes,
                .generic_loan_supported = metadata.direction == PortDirection::kOutput &&
                                          metadata.runtime_type_ops != nullptr &&
                                          metadata.runtime_type_ops->generic_loan_supported,
                .generic_loan_slot_size_bytes =
                    metadata.direction == PortDirection::kOutput &&
                            metadata.runtime_type_ops != nullptr &&
                            metadata.runtime_type_ops->generic_loan_supported
                        ? static_cast<std::uint64_t>(
                              metadata.runtime_type_ops->generic_loan_slot_size_bytes)
                        : 0U,
                .descriptor_tensor_field_count = metadata.descriptor_tensor_field_count,
                .setup = setup_configuration(metadata),
            },
        .declaration = metadata.declaration,
        .local_type_token = metadata.local_type_token,
        .operator_index = operator_index,
        .setup_tensor_allocation = metadata.tensor_allocation,
        .schema_package = metadata.schema_package,
    };
    state->ports_.push_back(std::move(port));
    GraphMutationRollback::checkpoint(*state);
    state->port_by_declaration_.emplace(metadata.declaration, port_id);
    GraphMutationRollback::checkpoint(*state);
    storage.ports.push_back(port_id);
    if (metadata.direction == PortDirection::kInput) {
      GraphPortOwnershipAccess::claim(*static_cast<const InputBase*>(metadata.declaration),
                                      state.get());
    } else {
      GraphPortOwnershipAccess::claim(*static_cast<const OutputBase*>(metadata.declaration),
                                      state.get());
    }
    GraphMutationRollback::checkpoint(*state);
  }
  for (std::size_t index = 0; index < OperatorSpecCompilerAccess::port_count(spec); ++index) {
    const auto identity = OperatorSpecCompilerAccess::port(spec, index).payload_identity;
    record_pod_identity(*state, identity);
    if (identity.pod)
      GraphMutationRollback::checkpoint(*state);
  }
  for (std::size_t index = 0; index < OperatorSpecCompilerAccess::source_count(spec); ++index) {
    const auto source = OperatorSpecCompilerAccess::source(spec, index);
    state->readiness_sources_.push_back(ReadinessSourceDeclarationRecord{
        .owner = vertex,
        .declaration_ordinal = static_cast<std::uint32_t>(index),
        .kind = source.kind,
        .name = std::string(source.name),
        .requested_capacity = source.capacity,
        .requested_sender_reference_capacity = source.kind == ReadinessSourceKind::kNotification
                                                   ? source.sender_reference_capacity
                                                   : 0U,
        .requested_clock = source.clock,
        .wake_requirement = source.kind == ReadinessSourceKind::kSpin
                                ? ReadinessWakeRequirement::kBusyPollRequired
                                : ReadinessWakeRequirement::kDeploymentSelected,
        .referenced = false,
    });
    GraphMutationRollback::checkpoint(*state);
  }
  for (std::size_t index = 0; index < OperatorSpecCompilerAccess::lifecycle_callback_count(spec);
       ++index) {
    const auto callback = OperatorSpecCompilerAccess::lifecycle_callback(spec, index);
    state->lifecycle_callbacks_.push_back(LifecycleCallbackAuthoringRecord{
        .owner = vertex,
        .stage = callback.stage,
        .required = callback.required,
        .method = callback.method,
        .origin = LifecycleCallbackOrigin::kAdvancedMember,
    });
    GraphMutationRollback::checkpoint(*state);
  }
  if (lifecycle_hooks.start) {
    state->lifecycle_callbacks_.push_back(LifecycleCallbackAuthoringRecord{
        .owner = vertex,
        .stage = LifecycleStage::kStart,
        .required = true,
        .method = lifecycle_method_id(LifecycleStage::kStart),
        .origin = LifecycleCallbackOrigin::kLegacyStart,
    });
    GraphMutationRollback::checkpoint(*state);
  }
  if (lifecycle_hooks.stop) {
    state->lifecycle_callbacks_.push_back(LifecycleCallbackAuthoringRecord{
        .owner = vertex,
        .stage = LifecycleStage::kStop,
        .required = true,
        .method = lifecycle_method_id(LifecycleStage::kStop),
        .origin = LifecycleCallbackOrigin::kLegacyStop,
    });
    GraphMutationRollback::checkpoint(*state);
  }
  for (std::size_t index = 0; index < OperatorSpecCompilerAccess::lifecycle_fact_count(spec);
       ++index) {
    const auto fact = OperatorSpecCompilerAccess::lifecycle_fact(spec, index);
    state->lifecycle_facts_.push_back(LifecycleFactAuthoringRecord{
        .owner = vertex,
        .stage = fact.stage,
        .kind = static_cast<LifecycleFactKind>(fact.kind),
        .stable_name = std::string(fact.stable_name),
    });
    GraphMutationRollback::checkpoint(*state);
  }
  state->operators_.push_back(std::move(storage));
  state->vertex_adjacency_.emplace_back();
  state->cycle_visit_marks_.push_back(0U);
  state->mark_mutation();
  GraphMutationRollback::checkpoint(*state);
  rollback.commit();
  graph_log_noexcept([&] { HOLOSCAN_LOG_TRACE("Registered Graph operator '{}'", name); });
  return OperatorRegistration{.index = operator_index};
}

const OperatorBase* graph_operator_prototype(const std::shared_ptr<GraphState>& state,
                                             std::size_t index,
                                             const void* operator_type_token) noexcept {
  const std::lock_guard lock(state->mutex_);
  if (index >= state->operators_.size())
    return nullptr;
  const auto& op = state->operators_[index];
  if (op.operator_type_token != operator_type_token)
    return nullptr;
  return op.prototype.get();
}

bool graph_operator_valid(const std::shared_ptr<GraphState>& state, std::size_t index,
                          const void* operator_type_token) noexcept {
  return graph_operator_prototype(state, index, operator_type_token) != nullptr;
}

std::string_view graph_operator_name(const std::shared_ptr<GraphState>& state,
                                     std::size_t index) noexcept {
  const std::lock_guard lock(state->mutex_);
  if (index >= state->operators_.size())
    return {};
  return state->operators_[index].name;
}

OperatorRef graph_operator_ref(const std::shared_ptr<GraphState>& state,
                               std::size_t index) noexcept {
  const std::lock_guard lock(state->mutex_);
  if (index >= state->operators_.size())
    return {};
  return state->operators_[index].plan_ref;
}

// Parameter replacement validates a complete candidate and its port/source invariants before
// atomically transferring ownership into the existing logical Operator identity.
expected<void, Error>
graph_set_operator_params(  // NOLINT(readability-function-cognitive-complexity)
    const std::shared_ptr<GraphState>& state, std::size_t index, const void* params,
    const void* parameter_type_token) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return make_unexpected(std::move(prepared.error()));

  std::shared_ptr<OperatorFactoryBase> factory;
  VertexId vertex;
  const void* operator_type_token{};
  {
    const std::lock_guard lock(state->mutex_);
    if (index >= state->operators_.size()) {
      return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"));
    }
    factory = state->operators_[index].factory;
    vertex = state->operators_[index].vertex;
    operator_type_token = state->operators_[index].operator_type_token;
  }
  auto updated_factory = factory->clone_with_params(params, parameter_type_token);
  if (!updated_factory) {
    return make_unexpected(std::move(updated_factory.error()));
  }
  auto candidate = updated_factory.value()->create();
  if (!candidate)
    return make_unexpected(std::move(candidate.error()));
  OperatorSpec spec;
  auto setup = run_setup(*candidate.value(), spec);
  if (!setup)
    return setup;
  auto valid_lifecycle = validate_lifecycle_metadata(spec, operator_type_token);
  if (!valid_lifecycle)
    return make_unexpected(std::move(valid_lifecycle.error()));
  const OperatorFactoryBase::LifecycleHookSet lifecycle_hooks =
      updated_factory.value()->lifecycle_hooks();
  if ((lifecycle_hooks.start && has_lifecycle_callback(spec, LifecycleStage::kStart)) ||
      (lifecycle_hooks.stop && has_lifecycle_callback(spec, LifecycleStage::kStop))) {
    return make_unexpected(
        graph_error(ErrorCode::kDuplicateName, "GRAPH_LIFECYCLE_CALLBACK_DUPLICATE"));
  }

  const std::lock_guard lock(state->mutex_);
  if (index >= state->operators_.size() || state->operators_[index].factory != factory) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_OPERATOR_CHANGED"));
  }
  auto& op = state->operators_[index];
  const auto vertex_record = std::find_if(
      state->vertices_.begin(), state->vertices_.end(), [vertex](const auto& candidate) {
        return candidate.id == vertex;
      });
  if (vertex_record == state->vertices_.end())
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_OPERATOR_CHANGED"));
  if (OperatorSpecCompilerAccess::port_count(spec) != op.ports.size()) {
    return make_unexpected(
        graph_error(ErrorCode::kIncompatible, "GRAPH_PARAMETER_PORT_SHAPE_CHANGED"));
  }
  std::vector<const void*> declarations;
  declarations.reserve(op.ports.size());
  for (std::size_t port_index = 0; port_index < op.ports.size(); ++port_index) {
    const auto metadata = OperatorSpecCompilerAccess::port(spec, port_index);
    PortStorage* existing = find_port(*state, op.ports[port_index]);
    if (existing == nullptr || !port_setup_equal(*existing, metadata)) {
      return make_unexpected(
          graph_error(ErrorCode::kIncompatible, "GRAPH_PARAMETER_PORT_SHAPE_CHANGED"));
    }
    const void* owner =
        metadata.direction == PortDirection::kInput
            ? GraphPortOwnershipAccess::owner(*static_cast<const InputBase*>(metadata.declaration))
            : GraphPortOwnershipAccess::owner(
                  *static_cast<const OutputBase*>(metadata.declaration));
    if (owner != nullptr && owner != state.get()) {
      return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_PORT_ALREADY_OWNED"));
    }
    declarations.push_back(metadata.declaration);
  }
  std::vector<ReadinessSourceDeclarationRecord*> existing_sources;
  for (auto& source : state->readiness_sources_) {
    if (source.owner == vertex)
      existing_sources.push_back(&source);
  }
  if (OperatorSpecCompilerAccess::source_count(spec) != existing_sources.size()) {
    return make_unexpected(
        graph_error(ErrorCode::kIncompatible, "GRAPH_PARAMETER_SOURCE_SHAPE_CHANGED"));
  }
  for (std::size_t source_index = 0; source_index < existing_sources.size(); ++source_index) {
    if (!source_equal(*existing_sources[source_index],
                      OperatorSpecCompilerAccess::source(spec, source_index))) {
      return make_unexpected(
          graph_error(ErrorCode::kIncompatible, "GRAPH_PARAMETER_SOURCE_SHAPE_CHANGED"));
    }
  }
  std::vector<const LifecycleCallbackAuthoringRecord*> existing_callbacks;
  for (const auto& callback : state->lifecycle_callbacks_) {
    if (callback.owner == vertex)
      existing_callbacks.push_back(&callback);
  }
  const std::size_t expected_callback_count =
      OperatorSpecCompilerAccess::lifecycle_callback_count(spec) +
      static_cast<std::size_t>(lifecycle_hooks.start) +
      static_cast<std::size_t>(lifecycle_hooks.stop);
  if (expected_callback_count != existing_callbacks.size()) {
    return make_unexpected(
        graph_error(ErrorCode::kIncompatible, "GRAPH_PARAMETER_LIFECYCLE_SHAPE_CHANGED"));
  }
  for (std::size_t callback_index = 0; callback_index < existing_callbacks.size();
       ++callback_index) {
    bool matches = false;
    const std::size_t advanced_count = OperatorSpecCompilerAccess::lifecycle_callback_count(spec);
    if (callback_index < advanced_count) {
      matches = lifecycle_callback_equal(
          *existing_callbacks[callback_index],
          OperatorSpecCompilerAccess::lifecycle_callback(spec, callback_index));
    } else {
      std::size_t legacy_index = callback_index - advanced_count;
      if (lifecycle_hooks.start) {
        if (legacy_index == 0U) {
          matches = lifecycle_callback_equal_legacy(*existing_callbacks[callback_index],
                                                    LifecycleStage::kStart,
                                                    LifecycleCallbackOrigin::kLegacyStart);
        } else {
          --legacy_index;
        }
      }
      if (!matches && lifecycle_hooks.stop && legacy_index == 0U) {
        matches = lifecycle_callback_equal_legacy(*existing_callbacks[callback_index],
                                                  LifecycleStage::kStop,
                                                  LifecycleCallbackOrigin::kLegacyStop);
      }
    }
    if (!matches) {
      return make_unexpected(
          graph_error(ErrorCode::kIncompatible, "GRAPH_PARAMETER_LIFECYCLE_SHAPE_CHANGED"));
    }
  }
  std::vector<const LifecycleFactAuthoringRecord*> existing_facts;
  for (const auto& fact : state->lifecycle_facts_) {
    if (fact.owner == vertex)
      existing_facts.push_back(&fact);
  }
  if (OperatorSpecCompilerAccess::lifecycle_fact_count(spec) != existing_facts.size()) {
    return make_unexpected(
        graph_error(ErrorCode::kIncompatible, "GRAPH_PARAMETER_LIFECYCLE_SHAPE_CHANGED"));
  }
  for (std::size_t fact_index = 0; fact_index < existing_facts.size(); ++fact_index) {
    if (!lifecycle_fact_equal(*existing_facts[fact_index],
                              OperatorSpecCompilerAccess::lifecycle_fact(spec, fact_index))) {
      return make_unexpected(
          graph_error(ErrorCode::kIncompatible, "GRAPH_PARAMETER_LIFECYCLE_SHAPE_CHANGED"));
    }
  }

  auto replacement_map = state->port_by_declaration_;
  for (const GraphPortId port_id : op.ports) {
    PortStorage* existing = find_port(*state, port_id);
    replacement_map.erase(existing->declaration);
  }
  for (std::size_t port_index = 0; port_index < op.ports.size(); ++port_index) {
    const auto [unused, inserted] =
        replacement_map.emplace(declarations[port_index], op.ports[port_index]);
    (void)unused;
    if (!inserted) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_PORT_DECLARATION_DUPLICATE"));
    }
  }
  using DeclarationMap = decltype(state->port_by_declaration_);
  static_assert(noexcept(std::declval<DeclarationMap&>().swap(std::declval<DeclarationMap&>())));
  static_assert(std::is_nothrow_move_assignable_v<std::shared_ptr<OperatorFactoryBase>>);
  static_assert(std::is_nothrow_move_assignable_v<std::unique_ptr<OperatorBase>>);
  GraphMutationRollback::checkpoint(*state);
  for (std::size_t port_index = 0; port_index < op.ports.size(); ++port_index) {
    const auto metadata = OperatorSpecCompilerAccess::port(spec, port_index);
    PortStorage* existing = find_port(*state, op.ports[port_index]);
    existing->declaration = declarations[port_index];
    if (metadata.direction == PortDirection::kInput) {
      GraphPortOwnershipAccess::claim(*static_cast<const InputBase*>(metadata.declaration),
                                      state.get());
    } else {
      GraphPortOwnershipAccess::claim(*static_cast<const OutputBase*>(metadata.declaration),
                                      state.get());
    }
  }
  state->port_by_declaration_.swap(replacement_map);
  op.factory = std::move(updated_factory.value());
  op.prototype = std::move(candidate.value());
  state->mark_mutation();
  return {};
}

namespace {

[[nodiscard]] bool creates_directed_cycle(GraphState& state, VertexId source_vertex,
                                          VertexId target_vertex) {
  if (!source_vertex || !target_vertex || source_vertex.value > state.vertex_adjacency_.size() ||
      target_vertex.value > state.vertex_adjacency_.size()) {
    return true;
  }
  const std::size_t source_index = static_cast<std::size_t>(source_vertex.value - 1U);
  const std::size_t target_index = static_cast<std::size_t>(target_vertex.value - 1U);
  if (source_index == target_index)
    return true;
  if (state.cycle_visit_epoch_ == std::numeric_limits<std::uint64_t>::max()) {
    std::fill(state.cycle_visit_marks_.begin(), state.cycle_visit_marks_.end(), 0U);
    state.cycle_visit_epoch_ = 1U;
  } else {
    ++state.cycle_visit_epoch_;
  }
  const std::uint64_t visit_epoch = state.cycle_visit_epoch_;
  std::vector<std::size_t> pending{target_index};
  while (!pending.empty()) {
    const std::size_t current = pending.back();
    pending.pop_back();
    if (current == source_index)
      return true;
    if (current >= state.cycle_visit_marks_.size() ||
        state.cycle_visit_marks_[current] == visit_epoch) {
      continue;
    }
    state.cycle_visit_marks_[current] = visit_epoch;
    const auto& outgoing = state.vertex_adjacency_[current];
    pending.insert(pending.end(), outgoing.begin(), outgoing.end());
  }
  return false;
}

void reserve_adjacency(GraphState& state, VertexId source) {
  assert(source && source.value <= state.vertex_adjacency_.size());
  auto& outgoing = state.vertex_adjacency_[static_cast<std::size_t>(source.value - 1U)];
  outgoing.reserve(outgoing.size() + 1U);
}

void append_adjacency(GraphState& state, VertexId source, VertexId target) {
  assert(source && source.value <= state.vertex_adjacency_.size());
  assert(target && target.value <= state.vertex_adjacency_.size());
  state.vertex_adjacency_[static_cast<std::size_t>(source.value - 1U)].push_back(
      static_cast<std::size_t>(target.value - 1U));
}

[[nodiscard]] std::optional<PayloadOptionsRecord> boundary_tensor_payload(const GraphState& state,
                                                                          GraphPortId port) {
  for (const auto& topic : state.external_topics_) {
    if (topic.record.graph_port == port && topic.record.tensor) {
      const PortStorage* const boundary_port = find_port(state, port);
      const bool descriptor_schema =
          boundary_port != nullptr &&
          boundary_port->record.payload_identity.kind == PayloadIdentityKind::kSchema &&
          boundary_port->record.payload_identity.payload_kind == PayloadKind::kDescriptorBacked;
      return PayloadOptionsRecord{
          .schema = topic.record.schema,
          .max_inline_bytes = topic.record.tensor->max_inline_bytes,
          .max_tensor_byte_span =
              descriptor_schema ? topic.record.tensor->max_byte_span : std::nullopt,
          .interface_tensor_emit_policy = topic.record.tensor->foreign_emit_policy,
      };
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<TensorPortLayout> effective_tensor_layout(const PortStorage& port) {
  return port.record.graph_override.tensor_layout ? port.record.graph_override.tensor_layout
                                                  : port.record.setup.tensor_layout;
}

[[nodiscard]] bool tensor_layouts_compatible(
    const std::optional<TensorPortLayout>& source,
    const std::optional<TensorPortLayout>& target) noexcept {
  if (!source || !target) {
    return true;
  }
  if (source->memory_kind && target->memory_kind && source->memory_kind != target->memory_kind) {
    return false;
  }
  if (source->rank && target->rank && source->rank != target->rank) {
    return false;
  }
  return !source->dtype || !target->dtype ||
         (source->dtype->code == target->dtype->code &&
          source->dtype->bits == target->dtype->bits &&
          source->dtype->lanes == target->dtype->lanes);
}

[[nodiscard]] bool declares_tensor_semantics(const PortStorage& port) noexcept {
  return port.record.payload_identity.kind == PayloadIdentityKind::kTensor ||
         port.record.setup.tensor_layout.has_value() || !port.record.setup.tensor_fields.empty();
}

[[nodiscard]] bool tensor_field_layouts_compatible(const PortStorage& source,
                                                   const PortStorage& target) noexcept {
  const auto& source_fields = source.record.setup.tensor_fields;
  const auto& target_fields = target.record.setup.tensor_fields;
  for (const auto& target_field : target_fields) {
    const auto source_field =
        std::find_if(source_fields.begin(), source_fields.end(), [&](const auto& candidate) {
          return candidate.field_id == target_field.field_id &&
                 candidate.tensor_index == target_field.tensor_index;
        });
    if (source_field != source_fields.end() &&
        !tensor_layouts_compatible(source_field->tensor_layout, target_field.tensor_layout)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] expected<PayloadOptionsRecord, Error> effective_tensor_payload(
    const GraphState& state, const PortStorage& source, const PortStorage& target,
    std::optional<PayloadOptionsRecord> source_override = std::nullopt) {
  if (!tensor_layouts_compatible(effective_tensor_layout(source),
                                 effective_tensor_layout(target))) {
    return make_unexpected(
        graph_error(ErrorCode::kIncompatible, "GRAPH_TENSOR_LAYOUT_EDGE_CONFLICT"));
  }
  if (!tensor_field_layouts_compatible(source, target)) {
    return make_unexpected(
        graph_error(ErrorCode::kIncompatible, "GRAPH_TENSOR_FIELD_LAYOUT_EDGE_CONFLICT"));
  }
  std::optional<PayloadOptionsRecord> producer = std::move(source_override);
  if (!producer) {
    producer = source.record.graph_override.payload_options
                   ? source.record.graph_override.payload_options
                   : source.record.setup.payload_options;
  }
  std::optional<PayloadOptionsRecord> boundary = boundary_tensor_payload(state, target.record.id);
  if (!boundary)
    boundary = boundary_tensor_payload(state, source.record.id);

  if (producer && boundary &&
      (producer->interface_tensor_emit_policy != boundary->interface_tensor_emit_policy ||
       producer->max_inline_bytes != boundary->max_inline_bytes ||
       producer->max_tensor_byte_span != boundary->max_tensor_byte_span)) {
    return make_unexpected(
        graph_error(ErrorCode::kIncompatible, "GRAPH_TENSOR_PAYLOAD_POLICY_CONFLICT"));
  }
  if (producer)
    return *producer;
  if (boundary)
    return *boundary;
  return PayloadOptionsRecord{};
}

using EdgePayloadUpdates = std::vector<std::pair<std::size_t, std::optional<PayloadOptionsRecord>>>;

[[nodiscard]] expected<EdgePayloadUpdates, Error> recompute_incident_tensor_payloads(
    const GraphState& state, GraphPortId changed_port, const PortStorage& candidate) {
  EdgePayloadUpdates updates;
  for (std::size_t index = 0; index < state.edges_.size(); ++index) {
    const auto& edge = state.edges_[index];
    if (edge.source != changed_port && edge.target != changed_port)
      continue;
    const PortStorage* stored_source = find_port(state, edge.source);
    const PortStorage* stored_target = find_port(state, edge.target);
    if (stored_source == nullptr || stored_target == nullptr) {
      return make_unexpected(graph_error(ErrorCode::kFailure, "GRAPH_EDGE_PORT_MISSING"));
    }
    const PortStorage& source = edge.source == changed_port ? candidate : *stored_source;
    const PortStorage& target = edge.target == changed_port ? candidate : *stored_target;
    const bool tensor_edge = declares_tensor_semantics(source) || declares_tensor_semantics(target);
    if (!tensor_edge) {
      updates.emplace_back(index, std::nullopt);
      continue;
    }
    auto normalized = effective_tensor_payload(state, source, target);
    if (!normalized)
      return make_unexpected(std::move(normalized.error()));
    updates.emplace_back(index, std::move(normalized.value()));
  }
  return updates;
}

template <typename PortBaseT>
[[nodiscard]] expected<GraphPortId, Error> register_interface_port(
    GraphState& state, const PortBaseT& port, PortDirection direction,
    PayloadIdentityRecord payload_identity, std::uint64_t serialized_payload_capacity_bytes) {
  const void* owner = GraphPortOwnershipAccess::owner(port);
  if (owner != nullptr && owner != &state) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_CROSS_GRAPH_PORT"));
  }
  const auto found = state.port_by_declaration_.find(&port);
  if (found != state.port_by_declaration_.end()) {
    PortStorage* existing = find_port(state, found->second);
    if (existing == nullptr || existing->record.owner_kind != PortOwnerKind::kGraphInterface ||
        existing->record.direction != direction ||
        !payload_identity_equal(existing->record.payload_identity, payload_identity) ||
        existing->record.serialized_payload_capacity_bytes != serialized_payload_capacity_bytes) {
      return make_unexpected(
          graph_error(ErrorCode::kIncompatible, "GRAPH_INTERFACE_PORT_CONFLICT"));
    }
    return found->second;
  }
  auto name = validate_name(port.name(), "GRAPH_INTERFACE_PORT_NAME_INVALID");
  if (!name)
    return make_unexpected(std::move(name.error()));
  auto identity = validate_payload_identity(payload_identity);
  if (!identity)
    return make_unexpected(std::move(identity.error()));
  const bool schema = payload_identity.kind == PayloadIdentityKind::kSchema;
  if (schema != (serialized_payload_capacity_bytes != 0U) ||
      serialized_payload_capacity_bytes > std::numeric_limits<std::uint32_t>::max()) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_SCHEMA_CAPACITY_INVALID"));
  }
  auto catalog = validate_pod_catalog(state, payload_identity);
  if (!catalog)
    return make_unexpected(std::move(catalog.error()));
  if (state.interface_port_names_.contains(std::string(port.name()))) {
    return make_unexpected(
        graph_error(ErrorCode::kDuplicateName, "GRAPH_INTERFACE_NAME_DUPLICATE"));
  }
  const VertexId vertex = state.next_vertex_id();
  const GraphPortId port_id = state.next_port_id();
  state.vertices_.push_back(VertexRecord{
      .id = vertex,
      .graph = state.graph_record_.id,
      .kind = VertexKind::kGraphInterface,
      .name = std::string(port.name()),
  });
  state.vertex_adjacency_.emplace_back();
  state.cycle_visit_marks_.push_back(0U);
  GraphMutationRollback::checkpoint(state);
  state.ports_.push_back(PortStorage{
      .record =
          PortRecord{
              .id = port_id,
              .owner = vertex,
              .owner_kind = PortOwnerKind::kGraphInterface,
              .direction = direction,
              .name = std::string(port.name()),
              .payload_identity = std::move(payload_identity),
              .serialized_payload_capacity_bytes = serialized_payload_capacity_bytes,
          },
      .declaration = &port,
  });
  GraphMutationRollback::checkpoint(state);
  state.port_by_declaration_.emplace(&port, port_id);
  GraphMutationRollback::checkpoint(state);
  state.interface_port_names_.emplace(port.name());
  GraphMutationRollback::checkpoint(state);
  record_pod_identity(state, state.ports_.back().record.payload_identity);
  if (state.ports_.back().record.payload_identity.pod)
    GraphMutationRollback::checkpoint(state);
  GraphPortOwnershipAccess::claim(port, &state);
  GraphMutationRollback::checkpoint(state);
  return port_id;
}

}  // namespace

// Flow authoring validates and commits every related graph index atomically.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<FlowRegistration, Error> graph_add_flow(const std::shared_ptr<GraphState>& state,
                                                 const OutputBase* source, const InputBase* target,
                                                 const PayloadIdentityRecord& source_identity,
                                                 const PayloadIdentityRecord& target_identity,
                                                 ConnectionOptions options) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return make_unexpected(std::move(prepared.error()));
  if (source == nullptr || target == nullptr) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_FLOW_PORT_INVALID"));
  }
  auto options_valid = validate_connection_options(options);
  if (!options_valid) {
    return make_unexpected(std::move(options_valid.error()));
  }
  const void* source_owner = GraphPortOwnershipAccess::owner(*source);
  const void* target_owner = GraphPortOwnershipAccess::owner(*target);
  const std::lock_guard lock(state->mutex_);
  const auto owns_embedded_state =
      [&](const auto& self, const GraphState& candidate, const void* owner) -> bool {
    for (const auto& embedded : candidate.embedded_graphs_) {
      if (embedded.state.get() == owner ||
          (embedded.state != nullptr && self(self, *embedded.state, owner))) {
        return true;
      }
    }
    return false;
  };
  // Embedding copies a child root interface into the parent declaration map so
  // parent authoring can connect that public boundary. Other declarations from
  // the embedded hierarchy intentionally fall through to GRAPH_PORT_NOT_FOUND;
  // a declaration owned by an unrelated Graph remains a cross-Graph error.
  const auto local_or_embedded_interface = [&](const void* declaration, const void* owner) {
    if (owner == nullptr || owner == state.get()) {
      return true;
    }
    const auto found = state->port_by_declaration_.find(declaration);
    if (found == state->port_by_declaration_.end()) {
      return owns_embedded_state(owns_embedded_state, *state, owner);
    }
    const PortStorage* port = find_port(*state, found->second);
    return port != nullptr && port->record.owner_kind == PortOwnerKind::kGraphInterface;
  };
  if (!local_or_embedded_interface(source, source_owner) ||
      !local_or_embedded_interface(target, target_owner)) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_CROSS_GRAPH_PORT"));
  }
  auto source_id =
      find_port_by_declaration(*state, source, PortDirection::kOutput, source_identity);
  if (!source_id)
    return make_unexpected(std::move(source_id.error()));
  auto target_id = find_port_by_declaration(*state, target, PortDirection::kInput, target_identity);
  if (!target_id)
    return make_unexpected(std::move(target_id.error()));
  if (!payload_identity_equal(source_identity, target_identity)) {
    return make_unexpected(graph_error(ErrorCode::kIncompatible, "GRAPH_PORT_TYPE_MISMATCH"));
  }
  if (std::any_of(state->edges_.begin(), state->edges_.end(), [&](const EdgeRecord& edge) {
        return edge.target == target_id.value();
      })) {
    return make_unexpected(graph_error(ErrorCode::kIncompatible, "GRAPH_INPUT_FANIN_UNSUPPORTED"));
  }
  const PortStorage* source_port = find_port(*state, source_id.value());
  const PortStorage* target_port = find_port(*state, target_id.value());
  if (source_port->record.serialized_payload_capacity_bytes !=
      target_port->record.serialized_payload_capacity_bytes) {
    return make_unexpected(graph_error(ErrorCode::kIncompatible, "GRAPH_SCHEMA_CAPACITY_MISMATCH"));
  }
  if ((source_owner != state.get() &&
       source_port->record.owner_kind != PortOwnerKind::kGraphInterface) ||
      (target_owner != state.get() &&
       target_port->record.owner_kind != PortOwnerKind::kGraphInterface)) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_CROSS_GRAPH_PORT"));
  }
  if (options.memory_conversion && !declares_tensor_semantics(*source_port) &&
      !declares_tensor_semantics(*target_port)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_NON_TENSOR_MEMORY_CONVERSION"));
  }
  if (creates_directed_cycle(*state, source_port->record.owner, target_port->record.owner)) {
    return make_unexpected(graph_error(ErrorCode::kIncompatible, "GRAPH_DIRECT_CYCLE"));
  }
  std::optional<PayloadOptionsRecord> effective_payload;
  if (declares_tensor_semantics(*source_port) || declares_tensor_semantics(*target_port)) {
    auto normalized = effective_tensor_payload(*state, *source_port, *target_port);
    if (!normalized)
      return make_unexpected(std::move(normalized.error()));
    effective_payload = std::move(normalized.value());
  }
  GraphMutationRollback rollback(*state);
  const EdgeId edge = state->next_edge_id();
  // Allocate before recording the edge so rollback cannot confuse this flow with a prior parallel
  // flow between the same vertices.
  reserve_adjacency(*state, source_port->record.owner);
  state->edges_.push_back(EdgeRecord{
      .id = edge,
      .source = source_id.value(),
      .target = target_id.value(),
      .add_flow_options = copy_connection_options(options),
      .effective_tensor_payload = std::move(effective_payload),
  });
  append_adjacency(*state, source_port->record.owner, target_port->record.owner);
  state->mark_mutation();
  GraphMutationRollback::checkpoint(*state);
  rollback.commit();
  graph_log_noexcept([&] { HOLOSCAN_LOG_TRACE("Registered Graph flow {}", edge.value); });
  return FlowRegistration{
      .edge = edge,
      .source = source_id.value(),
      .target = target_id.value(),
  };
}

expected<void, Error> graph_bind_input_interface(const std::shared_ptr<GraphState>& state,
                                                 const InputBase* public_input,
                                                 const InputBase* internal_input,
                                                 PayloadIdentityRecord payload_identity) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  if (public_input == nullptr || internal_input == nullptr || public_input == internal_input) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_INTERFACE_PORT_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  auto internal =
      find_port_by_declaration(*state, internal_input, PortDirection::kInput, payload_identity);
  if (!internal)
    return make_unexpected(std::move(internal.error()));
  const PortStorage* internal_port = find_port(*state, internal.value());
  if (GraphPortOwnershipAccess::owner(*internal_input) != state.get() &&
      internal_port->record.owner_kind != PortOwnerKind::kGraphInterface) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_CROSS_GRAPH_PORT"));
  }
  GraphMutationRollback rollback(*state);
  auto public_id = register_interface_port(*state,
                                           *public_input,
                                           PortDirection::kInput,
                                           std::move(payload_identity),
                                           internal_port->record.serialized_payload_capacity_bytes);
  if (!public_id)
    return make_unexpected(std::move(public_id.error()));
  if (std::any_of(state->interface_bindings_.begin(),
                  state->interface_bindings_.end(),
                  [&](const InterfaceBindingRecord& binding) {
                    return binding.public_port == public_id.value();
                  })) {
    return make_unexpected(
        graph_error(ErrorCode::kDuplicateName, "GRAPH_INTERFACE_BINDING_DUPLICATE"));
  }
  const PortStorage* public_port = find_port(*state, public_id.value());
  reserve_adjacency(*state, public_port->record.owner);
  state->interface_bindings_.push_back(InterfaceBindingRecord{
      .public_port = public_id.value(),
      .internal_port = internal.value(),
      .direction = PortDirection::kInput,
  });
  append_adjacency(*state, public_port->record.owner, internal_port->record.owner);
  state->mark_mutation();
  GraphMutationRollback::checkpoint(*state);
  rollback.commit();
  return {};
}

expected<void, Error> graph_bind_output_interface(const std::shared_ptr<GraphState>& state,
                                                  const OutputBase* internal_output,
                                                  const OutputBase* public_output,
                                                  PayloadIdentityRecord payload_identity) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  if (public_output == nullptr || internal_output == nullptr || public_output == internal_output) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_INTERFACE_PORT_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  auto internal =
      find_port_by_declaration(*state, internal_output, PortDirection::kOutput, payload_identity);
  if (!internal)
    return make_unexpected(std::move(internal.error()));
  const PortStorage* internal_port = find_port(*state, internal.value());
  if (GraphPortOwnershipAccess::owner(*internal_output) != state.get() &&
      internal_port->record.owner_kind != PortOwnerKind::kGraphInterface) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_CROSS_GRAPH_PORT"));
  }
  GraphMutationRollback rollback(*state);
  auto public_id = register_interface_port(*state,
                                           *public_output,
                                           PortDirection::kOutput,
                                           std::move(payload_identity),
                                           internal_port->record.serialized_payload_capacity_bytes);
  if (!public_id)
    return make_unexpected(std::move(public_id.error()));
  if (std::any_of(state->interface_bindings_.begin(),
                  state->interface_bindings_.end(),
                  [&](const InterfaceBindingRecord& binding) {
                    return binding.public_port == public_id.value();
                  })) {
    return make_unexpected(
        graph_error(ErrorCode::kDuplicateName, "GRAPH_INTERFACE_BINDING_DUPLICATE"));
  }
  const PortStorage* public_port = find_port(*state, public_id.value());
  reserve_adjacency(*state, internal_port->record.owner);
  state->interface_bindings_.push_back(InterfaceBindingRecord{
      .public_port = public_id.value(),
      .internal_port = internal.value(),
      .direction = PortDirection::kOutput,
  });
  append_adjacency(*state, internal_port->record.owner, public_port->record.owner);
  state->mark_mutation();
  GraphMutationRollback::checkpoint(*state);
  rollback.commit();
  return {};
}

expected<GraphPortId, Error> graph_find_port(const std::shared_ptr<GraphState>& state,
                                             const void* declaration, PortDirection direction,
                                             const PayloadIdentityRecord& payload_identity) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return make_unexpected(std::move(prepared.error()));
  const std::lock_guard lock(state->mutex_);
  return find_port_by_declaration(*state, declaration, direction, payload_identity);
}

const void* graph_port_declaration(const std::shared_ptr<GraphState>& state, GraphPortId port,
                                   PortDirection direction) noexcept {
  const std::lock_guard lock(state->mutex_);
  const PortStorage* found = find_port(*state, port);
  if (found == nullptr || found->record.direction != direction)
    return nullptr;
  return found->declaration;
}

const void* graph_port_declaration(const std::shared_ptr<GraphState>& state, GraphPortId port,
                                   PortDirection direction,
                                   const PayloadIdentityRecord& payload_identity) noexcept {
  const std::lock_guard lock(state->mutex_);
  const PortStorage* found = find_port(*state, port);
  if (found == nullptr || found->record.direction != direction ||
      !payload_identity_equal(found->record.payload_identity, payload_identity)) {
    return nullptr;
  }
  return found->declaration;
}

bool graph_flow_valid(const std::shared_ptr<GraphState>& state, EdgeId edge, GraphPortId source,
                      GraphPortId target) noexcept {
  if (!state || !edge)
    return false;
  const std::lock_guard lock(state->mutex_);
  const EdgeRecord* record = find_edge(*state, edge);
  return record != nullptr && record->source == source && record->target == target;
}

namespace {

template <typename Mutation>
expected<void, Error> configure_port(const std::shared_ptr<GraphState>& state, GraphPortId id,
                                     Mutation&& mutation) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  const std::lock_guard lock(state->mutex_);
  PortStorage* port = find_port(*state, id);
  if (port == nullptr) {
    return make_unexpected(graph_error(ErrorCode::kNotFound, "GRAPH_PORT_NOT_FOUND"));
  }
  auto result = std::forward<Mutation>(mutation)(*port);
  if (!result)
    return result;
  state->mark_mutation();
  return {};
}

template <typename Mutation>
expected<void, Error> configure_edge(const std::shared_ptr<GraphState>& state, EdgeId id,
                                     Mutation&& mutation) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  const std::lock_guard lock(state->mutex_);
  EdgeRecord* edge = find_edge(*state, id);
  if (edge == nullptr) {
    return make_unexpected(graph_error(ErrorCode::kNotFound, "GRAPH_FLOW_NOT_FOUND"));
  }
  std::forward<Mutation>(mutation)(*edge);
  state->mark_mutation();
  return {};
}

}  // namespace

expected<void, Error> graph_configure_input_queue(const std::shared_ptr<GraphState>& state,
                                                  GraphPortId id, std::uint32_t depth) {
  return configure_port(state, id, [depth](PortStorage& port) -> expected<void, Error> {
    if (port.record.direction != PortDirection::kInput || depth == 0U) {
      return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_INPUT_QUEUE_INVALID"));
    }
    port.record.graph_override.queue_depth = depth;
    return {};
  });
}

expected<void, Error> graph_configure_input_tensor(const std::shared_ptr<GraphState>& state,
                                                   GraphPortId id, TensorPortLayout layout) {
  if ((layout.memory_kind && !closed_enum(*layout.memory_kind)) ||
      (layout.dtype && !valid_dtype(*layout.dtype))) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_LAYOUT_INVALID"));
  }
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  const std::lock_guard lock(state->mutex_);
  PortStorage* port = find_port(*state, id);
  if (port == nullptr || port->record.direction != PortDirection::kInput) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_INPUT_CONFIG_INVALID"));
  }
  if (!declares_tensor_semantics(*port)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_NON_TENSOR_LAYOUT_MUTATION"));
  }
  PortStorage candidate = *port;
  candidate.record.graph_override.tensor_layout = layout;
  const auto options = candidate.record.graph_override.payload_options
                           ? candidate.record.graph_override.payload_options
                           : candidate.record.setup.payload_options;
  if (options) {
    auto valid = validate_payload_options(
        *options, candidate.record.payload_identity, effective_tensor_layout(candidate));
    if (!valid)
      return valid;
  }
  auto edge_updates = recompute_incident_tensor_payloads(*state, id, candidate);
  if (!edge_updates)
    return make_unexpected(std::move(edge_updates.error()));
  using TensorLayoutOverride = decltype(port->record.graph_override.tensor_layout);
  using EffectivePayload = decltype(state->edges_[0].effective_tensor_payload);
  static_assert(std::is_nothrow_copy_assignable_v<TensorLayoutOverride>);
  static_assert(std::is_nothrow_move_assignable_v<EffectivePayload>);
  GraphMutationRollback::checkpoint(*state);
  port->record.graph_override.tensor_layout = layout;
  for (auto& [index, effective] : edge_updates.value())
    state->edges_[index].effective_tensor_payload = std::move(effective);
  state->mark_mutation();
  return {};
}

expected<void, Error> graph_configure_input_memory(const std::shared_ptr<GraphState>& state,
                                                   GraphPortId id, MemoryConversionPolicy policy) {
  if (!closed_enum(policy)) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_MEMORY_POLICY_INVALID"));
  }
  return configure_port(state, id, [policy](PortStorage& port) -> expected<void, Error> {
    if (port.record.direction != PortDirection::kInput) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_INPUT_CONFIG_INVALID"));
    }
    if (!declares_tensor_semantics(port)) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_NON_TENSOR_MEMORY_CONVERSION"));
    }
    port.record.graph_override.memory_conversion = policy;
    return {};
  });
}

expected<void, Error> graph_configure_port_alias(const std::shared_ptr<GraphState>& state,
                                                 GraphPortId id, std::string_view alias) {
  auto valid = validate_name(alias, "GRAPH_PORT_ALIAS_INVALID");
  if (!valid)
    return valid;
  return configure_port(state, id, [alias](PortStorage& port) -> expected<void, Error> {
    port.record.graph_override.diagnostic_alias = std::string(alias);
    return {};
  });
}

expected<void, Error> graph_configure_output_emits(const std::shared_ptr<GraphState>& state,
                                                   GraphPortId id, std::uint32_t count) {
  return configure_port(state, id, [count](PortStorage& port) -> expected<void, Error> {
    if (port.record.direction != PortDirection::kOutput || count == 0U) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_OUTPUT_BOUND_INVALID"));
    }
    port.record.graph_override.max_emits_per_compute = count;
    return {};
  });
}

expected<void, Error> graph_configure_output_history(const std::shared_ptr<GraphState>& state,
                                                     GraphPortId id, std::uint32_t depth) {
  return configure_port(state, id, [depth](PortStorage& port) -> expected<void, Error> {
    if (port.record.direction != PortDirection::kOutput || depth == 0U) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_OUTPUT_CONFIG_INVALID"));
    }
    port.record.graph_override.late_joiner_history_depth = depth;
    return {};
  });
}

expected<void, Error> graph_configure_output_tensor(const std::shared_ptr<GraphState>& state,
                                                    GraphPortId id, TensorPortLayout layout) {
  if ((layout.memory_kind && !closed_enum(*layout.memory_kind)) ||
      (layout.dtype && !valid_dtype(*layout.dtype))) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_LAYOUT_INVALID"));
  }
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  const std::lock_guard lock(state->mutex_);
  PortStorage* port = find_port(*state, id);
  if (port == nullptr || port->record.direction != PortDirection::kOutput) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_OUTPUT_CONFIG_INVALID"));
  }
  if (!declares_tensor_semantics(*port)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_NON_TENSOR_LAYOUT_MUTATION"));
  }
  PortStorage candidate = *port;
  candidate.record.graph_override.tensor_layout = layout;
  const auto options = candidate.record.graph_override.payload_options
                           ? candidate.record.graph_override.payload_options
                           : candidate.record.setup.payload_options;
  if (options) {
    auto valid = validate_payload_options(
        *options, candidate.record.payload_identity, effective_tensor_layout(candidate));
    if (!valid)
      return valid;
  }
  auto edge_updates = recompute_incident_tensor_payloads(*state, id, candidate);
  if (!edge_updates)
    return make_unexpected(std::move(edge_updates.error()));
  using TensorLayoutOverride = decltype(port->record.graph_override.tensor_layout);
  using EffectivePayload = decltype(state->edges_[0].effective_tensor_payload);
  static_assert(std::is_nothrow_copy_assignable_v<TensorLayoutOverride>);
  static_assert(std::is_nothrow_move_assignable_v<EffectivePayload>);
  GraphMutationRollback::checkpoint(*state);
  port->record.graph_override.tensor_layout = layout;
  for (auto& [index, effective] : edge_updates.value())
    state->edges_[index].effective_tensor_payload = std::move(effective);
  state->mark_mutation();
  return {};
}

expected<void, Error> graph_configure_output_payload(const std::shared_ptr<GraphState>& state,
                                                     GraphPortId id, PayloadOptions options) {
  PayloadOptionsRecord copied = holoscan::detail::copy_payload_options(options);
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  const std::lock_guard lock(state->mutex_);
  PortStorage* port = find_port(*state, id);
  if (port == nullptr || port->record.direction != PortDirection::kOutput) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_OUTPUT_CONFIG_INVALID"));
  }
  const auto layout = effective_tensor_layout(*port);
  auto valid = validate_payload_options(copied, port->record.payload_identity, layout);
  if (!valid)
    return valid;
  std::vector<std::pair<std::size_t, PayloadOptionsRecord>> normalized;
  if (declares_tensor_semantics(*port)) {
    for (std::size_t index = 0; index < state->edges_.size(); ++index) {
      auto& edge = state->edges_[index];
      if (edge.source != id)
        continue;
      const PortStorage* target = find_port(*state, edge.target);
      auto effective = effective_tensor_payload(*state, *port, *target, copied);
      if (!effective)
        return make_unexpected(std::move(effective.error()));
      normalized.emplace_back(index, std::move(effective.value()));
    }
  }
  using PayloadOverride = decltype(port->record.graph_override.payload_options);
  using EffectivePayload = decltype(state->edges_[0].effective_tensor_payload);
  static_assert(std::is_nothrow_move_assignable_v<PayloadOverride>);
  static_assert(std::is_nothrow_move_assignable_v<EffectivePayload>);
  GraphMutationRollback::checkpoint(*state);
  port->record.graph_override.payload_options = std::move(copied);
  for (auto& [index, effective] : normalized)
    state->edges_[index].effective_tensor_payload = std::move(effective);
  state->mark_mutation();
  return {};
}

expected<void, Error> graph_configure_tensor_allocation(const std::shared_ptr<GraphState>& state,
                                                        GraphPortId id,
                                                        TensorAllocationBounds bounds) {
  return configure_port(state, id, [bounds](PortStorage& port) -> expected<void, Error> {
    if (port.record.direction != PortDirection::kOutput ||
        port.record.payload_identity.kind != PayloadIdentityKind::kTensor ||
        bounds.max_rank == 0U || bounds.max_byte_span == 0U || !port.setup_tensor_allocation ||
        bounds.max_rank > port.setup_tensor_allocation->max_rank ||
        bounds.max_byte_span > port.setup_tensor_allocation->max_byte_span) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_ALLOCATION_NOT_TIGHTENING"));
    }
    port.record.graph_override.tensor_allocation = bounds;
    return {};
  });
}

expected<void, Error> graph_configure_flow_policy(const std::shared_ptr<GraphState>& state,
                                                  EdgeId id, QueuePolicy policy) {
  if (!closed_enum(policy)) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_QUEUE_POLICY_INVALID"));
  }
  return configure_edge(
      state, id, [policy](EdgeRecord& edge) { edge.graph_override.queue_policy = policy; });
}

expected<void, Error> graph_configure_flow_depth(const std::shared_ptr<GraphState>& state,
                                                 EdgeId id, std::uint32_t depth) {
  if (depth == 0U) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_FLOW_BOUND_INVALID"));
  }
  return configure_edge(
      state, id, [depth](EdgeRecord& edge) { edge.graph_override.queue_depth = depth; });
}

expected<void, Error> graph_configure_flow_topic(const std::shared_ptr<GraphState>& state,
                                                 EdgeId id, std::string_view name) {
  auto valid = validate_name(name, "GRAPH_FLOW_TOPIC_NAME_INVALID");
  if (!valid)
    return valid;
  return configure_edge(
      state, id, [name](EdgeRecord& edge) { edge.graph_override.topic_name = std::string(name); });
}

expected<void, Error> graph_configure_flow_memory(const std::shared_ptr<GraphState>& state,
                                                  EdgeId id, MemoryConversionPolicy policy) {
  if (!closed_enum(policy)) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_MEMORY_POLICY_INVALID"));
  }
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  const std::lock_guard lock(state->mutex_);
  EdgeRecord* edge = find_edge(*state, id);
  if (edge == nullptr) {
    return make_unexpected(graph_error(ErrorCode::kNotFound, "GRAPH_FLOW_NOT_FOUND"));
  }
  const PortStorage* source = find_port(*state, edge->source);
  const PortStorage* target = find_port(*state, edge->target);
  if (source == nullptr || target == nullptr ||
      (!declares_tensor_semantics(*source) && !declares_tensor_semantics(*target))) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_NON_TENSOR_MEMORY_CONVERSION"));
  }
  edge->graph_override.memory_conversion = policy;
  state->mark_mutation();
  return {};
}

expected<void, Error> graph_configure_flow_inflight(const std::shared_ptr<GraphState>& state,
                                                    EdgeId id, std::uint32_t count) {
  if (count == 0U) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_FLOW_BOUND_INVALID"));
  }
  return configure_edge(
      state, id, [count](EdgeRecord& edge) { edge.graph_override.max_inflight_samples = count; });
}

namespace {

[[nodiscard]] expected<void, Error> validate_topic_uri(std::string_view uri) {
  if (!uri.starts_with(kHoloMqTopicPrefix) || uri.size() == kHoloMqTopicPrefix.size() ||
      uri.size() > 4096U || !valid_utf8(uri) || edge_whitespace(uri) ||
      std::any_of(uri.begin(), uri.end(), [](char value) {
        const auto byte = static_cast<unsigned char>(value);
        return byte == 0U || byte < 0x20U || byte == 0x7FU;
      })) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_TOPIC_URI_INVALID"));
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_clock_health(const ClockHealthOptions& health,
                                                          bool imported) {
  if (!closed_enum(health.capability) || !closed_enum(health.requirement)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "ENDPOINT_CLOCK_HEALTH_ENUM_INVALID"));
  }
  const bool ordinary = health.capability == ClockHealthCapability::kUnavailable &&
                        health.requirement == ClockHealthRequirement::kNotRequired &&
                        !health.max_report_age;
  const bool monitored = imported && health.capability == ClockHealthCapability::kStatusReports &&
                         health.requirement == ClockHealthRequirement::kFreshHealthyReport &&
                         health.max_report_age && health.max_report_age->count() > 0;
  if (!ordinary && !monitored) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "ENDPOINT_CLOCK_HEALTH_CONTRACT_INVALID"));
  }
  return {};
}

[[nodiscard]] expected<std::optional<LogicalTensorRequirementsRecord>, Error>
copy_tensor_requirements(const TopicOptions& options, const PayloadIdentityRecord& identity) {
  if (identity.kind != PayloadIdentityKind::kTensor) {
    if (options.tensor) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_NON_TENSOR_OPTIONS_INVALID"));
    }
    return std::nullopt;
  }
  if (!options.tensor)
    return std::nullopt;
  const auto& tensor = *options.tensor;
  if (!closed_enum(tensor.strides) || !closed_enum(tensor.foreign_emit_policy) ||
      (tensor.memory_kind && !closed_enum(*tensor.memory_kind))) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_ENUM_INVALID"));
  }
  if (tensor.dtype && !valid_dtype(*tensor.dtype)) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_DTYPE_INVALID"));
  }
  LogicalTensorRequirementsRecord result{
      .memory_kind = tensor.memory_kind,
      .dtype = tensor.dtype,
      .rank = tensor.rank,
      .strides = tensor.strides,
      .foreign_emit_policy = tensor.foreign_emit_policy,
      .max_inline_bytes = tensor.max_inline_bytes,
  };
  if (tensor.logical_bounds) {
    const auto& bounds = *tensor.logical_bounds;
    if (bounds.max_rank == 0U || bounds.max_byte_span == 0U ||
        bounds.axes.size() > bounds.max_rank || (tensor.rank && *tensor.rank > bounds.max_rank)) {
      return make_unexpected(
          graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_LOGICAL_BOUNDS_INVALID"));
    }
    result.max_rank = bounds.max_rank;
    result.max_byte_span = bounds.max_byte_span;
    result.axes.assign(bounds.axes.begin(), bounds.axes.end());
  }
  if (tensor.strides == TensorStridePolicy::kBoundedStrided &&
      (!result.max_rank || result.axes.size() != *result.max_rank ||
       std::any_of(result.axes.begin(), result.axes.end(), [](const TensorAxisBounds& axis) {
         return axis.max_abs_stride_bytes == 0U;
       }))) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_STRIDE_BOUNDS_INCOMPLETE"));
  }
  const bool inline_policy =
      tensor.foreign_emit_policy == InterfaceTensorEmitPolicy::kInlineSmallHostThenCopyIfNeeded;
  if ((inline_policy && (!tensor.max_inline_bytes || *tensor.max_inline_bytes == 0U)) ||
      (!inline_policy && tensor.max_inline_bytes)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_TENSOR_INLINE_POLICY_INVALID"));
  }
  return std::optional<LogicalTensorRequirementsRecord>{std::move(result)};
}

[[nodiscard]] expected<void, Error> validate_endpoint_bounds(const EndpointBounds& bounds,
                                                             bool imported) {
  if ((imported && bounds.subscriber) || (!imported && bounds.publisher) ||
      (bounds.publisher && bounds.subscriber)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_TOPIC_DIRECTION_BOUNDS_INVALID"));
  }
  if (bounds.publisher && bounds.publisher->max_loaned_samples &&
      *bounds.publisher->max_loaned_samples == 0U) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_TOPIC_PUBLISHER_BOUND_INVALID"));
  }
  if (bounds.subscriber &&
      ((bounds.subscriber->queue_depth && *bounds.subscriber->queue_depth == 0U) ||
       (bounds.subscriber->max_borrowed_samples &&
        *bounds.subscriber->max_borrowed_samples == 0U))) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_TOPIC_SUBSCRIBER_BOUND_INVALID"));
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_topic_schema(
    const TopicOptions& options, const PayloadIdentityRecord& identity,
    std::optional<SchemaIdentityRecord>& copied) {
  copied = copy_schema(options.schema);
  if (copied) {
    auto valid = validate_schema(*copied);
    if (!valid)
      return valid;
  }
  if (identity.kind == PayloadIdentityKind::kSchema) {
    if (copied && !schema_equal(copied, identity.schema)) {
      return make_unexpected(graph_error(ErrorCode::kIncompatible, "GRAPH_TOPIC_SCHEMA_MISMATCH"));
    }
    copied = identity.schema;
  } else if (identity.kind != PayloadIdentityKind::kTensor && copied) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_TOPIC_SCHEMA_IRRELEVANT"));
  }
  return {};
}

template <typename PortBaseT>
// External-topic registration keeps ordinal consumption, deep copying, contract validation, and
// publication in one fail-closed path so failed declarations cannot partially enter a snapshot.
expected<GraphPortId, Error>
register_external_topic(  // NOLINT(readability-function-cognitive-complexity)
    const std::shared_ptr<GraphState>& state, std::string_view uri, TopicOptions options,
    std::unique_ptr<PortBaseT> port, PayloadIdentityRecord payload_identity,
    std::optional<SchemaPackagePlanRecord>
        schema_package,  // NOLINT(performance-unnecessary-value-param)
    std::uint64_t serialized_payload_capacity_bytes, bool imported) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return make_unexpected(std::move(prepared.error()));
  if (!port) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_TOPIC_PORT_INVALID"));
  }

  const std::unique_lock lock(state->mutex_);
  if (state->next_boundary_ordinal_ == std::numeric_limits<std::uint32_t>::max()) {
    return make_unexpected(
        graph_error(ErrorCode::kResourceExhausted, "GRAPH_TOPIC_ORDINAL_EXHAUSTED"));
  }
  // GRAPH-011 intentionally consumes the mixed import/export declaration ordinal before
  // semantic validation. The exceptional-only guard distinguishes a normal typed rejection
  // from allocation/copy failure anywhere before the complete mutation transaction exists.
  const std::uint32_t ordinal = state->next_boundary_ordinal_++;
  const ExceptionalBoundaryOrdinalRollback ordinal_rollback{state->next_boundary_ordinal_, ordinal};
  GraphMutationRollback::checkpoint(*state);

  auto valid_uri = validate_topic_uri(uri);
  if (!valid_uri)
    return make_unexpected(std::move(valid_uri.error()));
  auto identity_valid = validate_payload_identity(payload_identity);
  if (!identity_valid)
    return make_unexpected(std::move(identity_valid.error()));
  const bool schema = payload_identity.kind == PayloadIdentityKind::kSchema;
  if (schema != (serialized_payload_capacity_bytes != 0U) ||
      serialized_payload_capacity_bytes > std::numeric_limits<std::uint32_t>::max()) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_SCHEMA_CAPACITY_INVALID"));
  }
  if (payload_identity.kind == PayloadIdentityKind::kScalar &&
      payload_identity.scalar->kind == ScalarIdentityKind::kLocalEnum) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_EXTERNAL_ENUM_IDENTITY_REQUIRED"));
  }
  std::optional<std::string> endpoint_name;
  if (options.endpoint_name) {
    auto valid_name = validate_endpoint_name(*options.endpoint_name);
    if (!valid_name)
      return make_unexpected(std::move(valid_name.error()));
    endpoint_name = std::string(*options.endpoint_name);
  }
  if (options.attach.max_wait && options.attach.max_wait->count() <= 0) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_TOPIC_ATTACH_BOUND_INVALID"));
  }
  if (!closed_enum(options.attach.peer) ||
      (options.memory_conversion && !closed_enum(*options.memory_conversion))) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_TOPIC_ENUM_INVALID"));
  }
  if (options.capture_time && !options.capture_time->clock.is_default() &&
      options.capture_time->clock.name().empty()) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_CAPTURE_CLOCK_INVALID"));
  }
  const ClockHealthOptions health = options.clock_health.value_or(ClockHealthOptions{});
  auto valid_health = validate_clock_health(health, imported);
  if (!valid_health)
    return make_unexpected(std::move(valid_health.error()));
  auto valid_bounds = validate_endpoint_bounds(options.bounds, imported);
  if (!valid_bounds)
    return make_unexpected(std::move(valid_bounds.error()));
  auto tensor = copy_tensor_requirements(options, payload_identity);
  if (!tensor)
    return make_unexpected(std::move(tensor.error()));
  std::optional<SchemaIdentityRecord> topic_schema;
  auto valid_schema = validate_topic_schema(options, payload_identity, topic_schema);
  if (!valid_schema)
    return make_unexpected(std::move(valid_schema.error()));
  if (options.memory_conversion && payload_identity.kind != PayloadIdentityKind::kTensor) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_TOPIC_MEMORY_CONVERSION_IRRELEVANT"));
  }

  if (endpoint_name && state->explicit_endpoint_names_.contains(*endpoint_name)) {
    return make_unexpected(graph_error(ErrorCode::kDuplicateName, "GRAPH_ENDPOINT_NAME_DUPLICATE"));
  }
  if (GraphPortOwnershipAccess::owner(*port) != nullptr) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_PORT_ALREADY_OWNED"));
  }
  auto catalog = validate_pod_catalog(*state, payload_identity);
  if (!catalog)
    return make_unexpected(std::move(catalog.error()));
  ExternalTopicStorage storage;
  GraphMutationRollback rollback(*state);
  rollback.restore_boundary_ordinal(ordinal);
  const VertexId vertex = state->next_vertex_id();
  const GraphPortId port_id = state->next_port_id();
  const ExternalTopicId topic_id = state->next_external_topic_id();
  const PortDirection direction = imported ? PortDirection::kOutput : PortDirection::kInput;
  const PortOwnerKind owner_kind =
      imported ? PortOwnerKind::kImportedTopic : PortOwnerKind::kExportedTopic;
  ExternalTopicRecord record{
      .id = topic_id,
      .vertex = vertex,
      .graph_port = port_id,
      .imported = imported,
      .uri = std::string(uri),
      .endpoint_name = endpoint_name,
      .generated_name_requested = !endpoint_name,
      .declaration_ordinal = ordinal,
      .payload_identity = std::move(payload_identity),
      .serialized_payload_capacity_bytes = serialized_payload_capacity_bytes,
      .schema = std::move(topic_schema),
      .memory_conversion = options.memory_conversion,
      .capture_time = std::move(options.capture_time),
      .clock_health = health,
      .clock_health_authored = options.clock_health.has_value(),
      .attach = options.attach,
      .tensor = std::move(tensor.value()),
      .bounds = options.bounds,
  };
  storage.record = std::move(record);
  if constexpr (std::same_as<PortBaseT, OutputBase>) {
    storage.output = std::move(port);
  } else {
    storage.input = std::move(port);
  }
  const PortBaseT* owned_port = nullptr;
  if constexpr (std::same_as<PortBaseT, OutputBase>) {
    owned_port = storage.output.get();
  } else {
    owned_port = storage.input.get();
  }
  state->vertices_.push_back(VertexRecord{
      .id = vertex,
      .graph = state->graph_record_.id,
      .kind = imported ? VertexKind::kImportedTopic : VertexKind::kExportedTopic,
      .name = endpoint_name.value_or(std::string(uri)),
  });
  state->vertex_adjacency_.emplace_back();
  state->cycle_visit_marks_.push_back(0U);
  GraphMutationRollback::checkpoint(*state);
  state->ports_.push_back(PortStorage{
      .record =
          PortRecord{
              .id = port_id,
              .owner = vertex,
              .owner_kind = owner_kind,
              .direction = direction,
              .name = "external",
              .payload_identity = storage.record.payload_identity,
              .serialized_payload_capacity_bytes = serialized_payload_capacity_bytes,
          },
      .declaration = owned_port,
      .schema_package = std::move(schema_package),
  });
  GraphMutationRollback::checkpoint(*state);
  state->port_by_declaration_.emplace(owned_port, port_id);
  GraphMutationRollback::checkpoint(*state);
  GraphPortOwnershipAccess::claim(*owned_port, state.get());
  GraphMutationRollback::checkpoint(*state);
  state->external_topics_.push_back(std::move(storage));
  GraphMutationRollback::checkpoint(*state);
  if (endpoint_name) {
    state->explicit_endpoint_names_.emplace(*endpoint_name);
    GraphMutationRollback::checkpoint(*state);
  }
  record_pod_identity(*state, state->external_topics_.back().record.payload_identity);
  if (state->external_topics_.back().record.payload_identity.pod)
    GraphMutationRollback::checkpoint(*state);
  state->mark_mutation();
  rollback.commit();
  return port_id;
}

}  // namespace

expected<GraphPortId, Error> graph_register_import(
    const std::shared_ptr<GraphState>& state, std::string_view uri, TopicOptions options,
    std::unique_ptr<OutputBase> port, PayloadIdentityRecord payload_identity,
    std::optional<SchemaPackagePlanRecord> schema_package,
    std::uint64_t serialized_payload_capacity_bytes) {
  return register_external_topic(state,
                                 uri,
                                 std::move(options),
                                 std::move(port),
                                 std::move(payload_identity),
                                 std::move(schema_package),
                                 serialized_payload_capacity_bytes,
                                 true);
}

expected<GraphPortId, Error> graph_register_export(
    const std::shared_ptr<GraphState>& state, std::string_view uri, TopicOptions options,
    std::unique_ptr<InputBase> port, PayloadIdentityRecord payload_identity,
    std::optional<SchemaPackagePlanRecord> schema_package,
    std::uint64_t serialized_payload_capacity_bytes) {
  return register_external_topic(state,
                                 uri,
                                 std::move(options),
                                 std::move(port),
                                 std::move(payload_identity),
                                 std::move(schema_package),
                                 serialized_payload_capacity_bytes,
                                 false);
}

expected<RegionId, Error> graph_add_region(const std::shared_ptr<GraphState>& state,
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
  state->regions_.push_back(
      RegionRecord{.id = id, .graph = state->graph_record_.id, .name = std::string(name)});
  GraphMutationRollback::checkpoint(*state);
  state->region_names_.emplace(name);
  GraphMutationRollback::checkpoint(*state);
  state->mark_mutation();
  rollback.commit();
  return id;
}

expected<void, Error> graph_region_add_operator(const std::shared_ptr<GraphState>& state,
                                                RegionId id, std::size_t operator_index) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  const std::lock_guard lock(state->mutex_);
  if (!id || id.value > state->regions_.size() ||
      state->regions_[static_cast<std::size_t>(id.value - 1U)].id != id ||
      operator_index >= state->operators_.size()) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_REGION_HANDLE_INVALID"));
  }
  auto& region = state->regions_[static_cast<std::size_t>(id.value - 1U)];
  auto members = region.members;
  const VertexId vertex = state->operators_[operator_index].vertex;
  if (std::find(members.begin(), members.end(), vertex) != members.end()) {
    return make_unexpected(graph_error(ErrorCode::kDuplicateName, "GRAPH_REGION_MEMBER_DUPLICATE"));
  }
  members.push_back(vertex);
  std::sort(members.begin(), members.end(), [](VertexId lhs, VertexId rhs) {
    return lhs.value < rhs.value;
  });
  GraphMutationRollback::checkpoint(*state);
  region.members = std::move(members);
  state->mark_mutation();
  return {};
}

expected<void, Error> graph_region_add_embedded_graph(const std::shared_ptr<GraphState>& state,
                                                      RegionId id, std::size_t embedded_index) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  const std::lock_guard lock(state->mutex_);
  if (!id || id.value > state->regions_.size() ||
      state->regions_[static_cast<std::size_t>(id.value - 1U)].id != id ||
      embedded_index >= state->embedded_graphs_.size()) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_REGION_HANDLE_INVALID"));
  }
  auto& region = state->regions_[static_cast<std::size_t>(id.value - 1U)];
  auto members = region.graph_members;
  const GraphId graph = state->embedded_graphs_[embedded_index].graph_id;
  if (std::find(members.begin(), members.end(), graph) != members.end()) {
    return make_unexpected(graph_error(ErrorCode::kDuplicateName, "GRAPH_REGION_MEMBER_DUPLICATE"));
  }
  members.push_back(graph);
  std::sort(members.begin(), members.end(), [](GraphId lhs, GraphId rhs) {
    return lhs.value < rhs.value;
  });
  GraphMutationRollback::checkpoint(*state);
  region.graph_members = std::move(members);
  state->mark_mutation();
  return {};
}

expected<void, Error> graph_region_add_constraint(const std::shared_ptr<GraphState>& state,
                                                  RegionId id, RegionConstraint constraint) {
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
  if (!id || id.value > state->regions_.size() ||
      state->regions_[static_cast<std::size_t>(id.value - 1U)].id != id) {
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

expected<void, Error> graph_region_add_hint(const std::shared_ptr<GraphState>& state, RegionId id,
                                            OptimizationHint hint) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  if (!closed_enum(hint)) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_REGION_HINT_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  if (!id || id.value > state->regions_.size() ||
      state->regions_[static_cast<std::size_t>(id.value - 1U)].id != id) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_REGION_HANDLE_INVALID"));
  }
  auto& region = state->regions_[static_cast<std::size_t>(id.value - 1U)];
  auto hints = region.hints;
  if (std::find(hints.begin(), hints.end(), hint) != hints.end()) {
    return make_unexpected(graph_error(ErrorCode::kDuplicateName, "GRAPH_REGION_HINT_DUPLICATE"));
  }
  hints.push_back(hint);
  std::sort(hints.begin(), hints.end(), [](OptimizationHint lhs, OptimizationHint rhs) {
    return static_cast<std::underlying_type_t<OptimizationHint>>(lhs) <
           static_cast<std::underlying_type_t<OptimizationHint>>(rhs);
  });
  GraphMutationRollback::checkpoint(*state);
  region.hints = std::move(hints);
  state->mark_mutation();
  return {};
}

expected<void, Error> graph_region_set_profile(const std::shared_ptr<GraphState>& state,
                                               RegionId id, Profile profile) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  if (!closed_enum(profile)) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_PROFILE_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  if (!id || id.value > state->regions_.size() ||
      state->regions_[static_cast<std::size_t>(id.value - 1U)].id != id) {
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

expected<PartitionId, Error> graph_add_partition(const std::shared_ptr<GraphState>& state,
                                                 std::string_view name) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return make_unexpected(std::move(prepared.error()));
  auto valid = validate_name(name, "GRAPH_PARTITION_NAME_INVALID");
  if (!valid)
    return make_unexpected(std::move(valid.error()));
  const std::lock_guard lock(state->mutex_);
  if (state->partition_names_.contains(std::string(name))) {
    return make_unexpected(
        graph_error(ErrorCode::kDuplicateName, "GRAPH_PARTITION_NAME_DUPLICATE"));
  }
  GraphMutationRollback rollback(*state);
  const PartitionId id = state->next_partition_id();
  state->partitions_.push_back(
      PartitionRecord{.id = id, .graph = state->graph_record_.id, .name = std::string(name)});
  GraphMutationRollback::checkpoint(*state);
  state->partition_names_.emplace(name);
  GraphMutationRollback::checkpoint(*state);
  state->mark_mutation();
  rollback.commit();
  return id;
}

expected<void, Error> graph_partition_add_operator(const std::shared_ptr<GraphState>& state,
                                                   PartitionId id, std::size_t operator_index) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  const std::lock_guard lock(state->mutex_);
  if (!id || id.value > state->partitions_.size() ||
      state->partitions_[static_cast<std::size_t>(id.value - 1U)].id != id ||
      operator_index >= state->operators_.size()) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_PARTITION_HANDLE_INVALID"));
  }

  const VertexId vertex = state->operators_[operator_index].vertex;
  const bool assigned = std::any_of(
      state->partitions_.begin(), state->partitions_.end(), [&](const PartitionRecord& partition) {
        return partition.graph == state->graph_record_.id &&
               std::find(partition.members.begin(), partition.members.end(), vertex) !=
                   partition.members.end();
      });
  if (assigned) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_PARTITION_MEMBER_DUPLICATE"));
  }

  auto& partition = state->partitions_[static_cast<std::size_t>(id.value - 1U)];
  auto members = partition.members;
  members.push_back(vertex);
  std::sort(members.begin(), members.end(), [](VertexId lhs, VertexId rhs) {
    return lhs.value < rhs.value;
  });
  GraphMutationRollback::checkpoint(*state);
  partition.members = std::move(members);
  state->mark_mutation();
  return {};
}

namespace {

[[nodiscard]] bool lifecycle_cycle(const GraphState& state, VertexId before, VertexId after,
                                   LifecycleStage stage) {
  std::vector<VertexId> pending{after};
  std::unordered_set<std::uint64_t> visited;
  while (!pending.empty()) {
    const VertexId current = pending.back();
    pending.pop_back();
    if (current == before)
      return true;
    if (!visited.emplace(current.value).second)
      continue;
    for (const auto& dependency : state.lifecycle_dependencies_) {
      if (dependency.stage == stage && dependency.before == current) {
        pending.push_back(dependency.after);
      }
    }
  }
  return false;
}

[[nodiscard]] std::optional<VertexId> resolve_operator_vertex(const GraphState& parent,
                                                              const GraphState& source,
                                                              std::size_t source_index) noexcept {
  if (&parent == &source) {
    if (source_index >= parent.operators_.size())
      return std::nullopt;
    return parent.operators_[source_index].vertex;
  }
  const auto found =
      std::find_if(parent.embedded_operator_mappings_.begin(),
                   parent.embedded_operator_mappings_.end(),
                   [&](const EmbeddedOperatorMapping& mapping) {
                     return mapping.source_state == &source && mapping.source_index == source_index;
                   });
  return found == parent.embedded_operator_mappings_.end() ? std::nullopt
                                                           : std::optional<VertexId>{found->vertex};
}

}  // namespace

expected<void, Error> graph_lifecycle_before(const std::shared_ptr<GraphState>& state,
                                             std::size_t before_index, std::size_t after_index,
                                             LifecycleStage stage) {
  return graph_lifecycle_before_resolved(state, state, before_index, state, after_index, stage);
}

expected<void, Error> graph_lifecycle_before_resolved(
    const std::shared_ptr<GraphState>& state, const std::shared_ptr<GraphState>& before_state,
    std::size_t before_index, const std::shared_ptr<GraphState>& after_state,
    std::size_t after_index, LifecycleStage stage) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  if (!closed_enum(stage)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_STAGE_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  const std::optional<VertexId> before =
      resolve_operator_vertex(*state, *before_state, before_index);
  const std::optional<VertexId> after = resolve_operator_vertex(*state, *after_state, after_index);
  if (!before || !after) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_HANDLE_INVALID"));
  }
  if (*before == *after || lifecycle_cycle(*state, *before, *after, stage)) {
    return make_unexpected(graph_error(ErrorCode::kIncompatible, "GRAPH_LIFECYCLE_CYCLE"));
  }
  const LifecycleDependencyAuthoringRecord candidate{
      .graph = state->graph_record_.id,
      .before = *before,
      .after = *after,
      .stage = stage,
  };
  if (std::any_of(state->lifecycle_dependencies_.begin(),
                  state->lifecycle_dependencies_.end(),
                  [&](const auto& existing) {
                    return existing.before == candidate.before &&
                           existing.after == candidate.after && existing.stage == candidate.stage;
                  })) {
    return make_unexpected(
        graph_error(ErrorCode::kDuplicateName, "GRAPH_LIFECYCLE_DEPENDENCY_DUPLICATE"));
  }
  state->lifecycle_dependencies_.push_back(candidate);
  state->mark_mutation();
  return {};
}

expected<void, Error> graph_lifecycle_permission(const std::shared_ptr<GraphState>& state,
                                                 LifecycleRestartOrigin origin,
                                                 std::optional<std::size_t> requester,
                                                 std::size_t target_index) {
  return graph_lifecycle_permission_resolved(state, origin, state, requester, state, target_index);
}

expected<void, Error> graph_lifecycle_permission_resolved(
    const std::shared_ptr<GraphState>& state, LifecycleRestartOrigin origin,
    const std::shared_ptr<GraphState>& requester_state, std::optional<std::size_t> requester,
    const std::shared_ptr<GraphState>& target_state, std::size_t target_index) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  if (!closed_enum(origin)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_ORIGIN_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  const std::optional<VertexId> target =
      resolve_operator_vertex(*state, *target_state, target_index);
  const std::optional<VertexId> requester_vertex =
      requester ? resolve_operator_vertex(*state, *requester_state, *requester) : std::nullopt;
  if (!target || (requester && !requester_vertex)) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_HANDLE_INVALID"));
  }
  if ((requester_vertex && *requester_vertex == *target) ||
      ((origin == LifecycleRestartOrigin::kOperatorRequest) != requester.has_value())) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_PERMISSION_INVALID"));
  }
  LifecycleRestartPermissionAuthoringRecord candidate{
      .origin = origin,
      .requester = requester_vertex,
      .target = *target,
  };
  if (std::any_of(state->lifecycle_restart_permissions_.begin(),
                  state->lifecycle_restart_permissions_.end(),
                  [&](const auto& existing) {
                    return existing.origin == candidate.origin &&
                           existing.requester == candidate.requester &&
                           existing.target == candidate.target;
                  })) {
    return make_unexpected(
        graph_error(ErrorCode::kDuplicateName, "GRAPH_LIFECYCLE_PERMISSION_DUPLICATE"));
  }
  state->lifecycle_restart_permissions_.push_back(std::move(candidate));
  state->mark_mutation();
  return {};
}

expected<ClockKey, Error> graph_add_clock(const std::shared_ptr<GraphState>& state,
                                          std::string_view name,
                                          std::shared_ptr<ClockFactoryBase> factory) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return make_unexpected(std::move(prepared.error()));
  auto valid_name = validate_name(name, "GRAPH_CLOCK_NAME_INVALID");
  if (!valid_name)
    return make_unexpected(std::move(valid_name.error()));
  if (!factory) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_CLOCK_FACTORY_INVALID"));
  }
  auto candidate = factory->create();
  if (!candidate)
    return make_unexpected(std::move(candidate.error()));
  if (!candidate.value()) {
    return make_unexpected(graph_error(ErrorCode::kFailure, "GRAPH_CLOCK_FACTORY_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  if (state->clock_names_.contains(std::string(name))) {
    return make_unexpected(graph_error(ErrorCode::kDuplicateName, "GRAPH_CLOCK_NAME_DUPLICATE"));
  }
  ClockKey key = ClockKey::named(name);
  ClockStorage storage{
      .record =
          ClockDeclarationRecord{
              .graph = state->graph_record_.id,
              .key = key,
              .shared = false,
              .is_default = false,
          },
      .capabilities = clock_capabilities(*candidate.value(), state->graph_record_.id, key),
      .factory = std::move(factory),
  };
  GraphMutationRollback rollback(*state);
  state->clocks_.push_back(std::move(storage));
  GraphMutationRollback::checkpoint(*state);
  state->clock_names_.emplace(name);
  GraphMutationRollback::checkpoint(*state);
  state->mark_mutation();
  rollback.commit();
  return key;
}

expected<ClockKey, Error> graph_bind_shared_clock(const std::shared_ptr<GraphState>& state,
                                                  std::string_view name,
                                                  std::shared_ptr<Clock> clock) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return make_unexpected(std::move(prepared.error()));
  auto valid_name = validate_name(name, "GRAPH_CLOCK_NAME_INVALID");
  if (!valid_name)
    return make_unexpected(std::move(valid_name.error()));
  if (!clock) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_SHARED_CLOCK_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  if (state->clock_names_.contains(std::string(name))) {
    return make_unexpected(graph_error(ErrorCode::kDuplicateName, "GRAPH_CLOCK_NAME_DUPLICATE"));
  }
  ClockKey key = ClockKey::named(name);
  ClockStorage storage{
      .record =
          ClockDeclarationRecord{
              .graph = state->graph_record_.id,
              .key = key,
              .shared = true,
              .is_default = false,
          },
      .capabilities = clock_capabilities(*clock, state->graph_record_.id, key),
      .shared = std::move(clock),
  };
  GraphMutationRollback rollback(*state);
  state->clocks_.push_back(std::move(storage));
  GraphMutationRollback::checkpoint(*state);
  state->clock_names_.emplace(name);
  GraphMutationRollback::checkpoint(*state);
  state->mark_mutation();
  rollback.commit();
  return key;
}

expected<ClockKey, Error> graph_import_clock(const std::shared_ptr<GraphState>& state,
                                             std::string_view name) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return make_unexpected(std::move(prepared.error()));
  auto valid_name = validate_name(name, "GRAPH_CLOCK_IMPORT_NAME_INVALID");
  if (!valid_name)
    return make_unexpected(std::move(valid_name.error()));
  const std::lock_guard lock(state->mutex_);
  if (state->clock_names_.contains(std::string(name))) {
    return make_unexpected(graph_error(ErrorCode::kDuplicateName, "GRAPH_CLOCK_NAME_DUPLICATE"));
  }
  ClockKey key = ClockKey::named(name);
  GraphMutationRollback rollback(*state);
  state->clock_imports_.push_back(ClockImportAuthoringRecord{
      .graph = state->graph_record_.id,
      .key = key,
      .is_default = false,
  });
  GraphMutationRollback::checkpoint(*state);
  state->clock_names_.emplace(name);
  GraphMutationRollback::checkpoint(*state);
  state->mark_mutation();
  rollback.commit();
  return key;
}

expected<void, Error> graph_bind_clock_import(const std::shared_ptr<GraphState>& state,
                                              std::size_t child_index, ClockKey child_import,
                                              ClockKey parent_clock) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  if (child_import.is_default() || child_import.name().empty() || parent_clock.is_default() ||
      parent_clock.name().empty()) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_CLOCK_BINDING_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  if (child_index >= state->embedded_graphs_.size()) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_CLOCK_BINDING_HANDLE_INVALID"));
  }
  const GraphId child_graph = state->embedded_graphs_[child_index].graph_id;
  const auto imported = std::find_if(
      state->clock_imports_.begin(), state->clock_imports_.end(), [&](const auto& candidate) {
        return candidate.graph == child_graph && candidate.key == child_import;
      });
  if (imported == state->clock_imports_.end()) {
    return make_unexpected(graph_error(ErrorCode::kNotFound, "GRAPH_CLOCK_IMPORT_NOT_FOUND"));
  }
  if (!state->clock_names_.contains(std::string(parent_clock.name()))) {
    return make_unexpected(graph_error(ErrorCode::kNotFound, "GRAPH_CLOCK_NOT_FOUND"));
  }
  if (std::any_of(
          state->clock_bindings_.begin(), state->clock_bindings_.end(), [&](const auto& existing) {
            return existing.child_graph == child_graph && existing.child_clock == child_import;
          })) {
    return make_unexpected(graph_error(ErrorCode::kDuplicateName, "GRAPH_CLOCK_BINDING_DUPLICATE"));
  }
  GraphMutationRollback rollback(*state);
  state->clock_bindings_.push_back(EmbeddedClockBindingAuthoringRecord{
      .child_graph = child_graph,
      .child_clock = std::move(child_import),
      .parent_graph = state->graph_record_.id,
      .parent_clock = std::move(parent_clock),
  });
  state->mark_mutation();
  rollback.commit();
  return {};
}

expected<void, Error> graph_set_default_clock(const std::shared_ptr<GraphState>& state,
                                              ClockKey key) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  if (key.is_default() || key.name().empty()) {
    return make_unexpected(graph_error(ErrorCode::kInvalidArgument, "GRAPH_DEFAULT_CLOCK_INVALID"));
  }
  const std::lock_guard lock(state->mutex_);
  if (!state->clock_names_.contains(std::string(key.name()))) {
    return make_unexpected(graph_error(ErrorCode::kNotFound, "GRAPH_CLOCK_NOT_FOUND"));
  }
  static_assert(std::is_nothrow_move_assignable_v<ClockKey>);
  GraphMutationRollback::checkpoint(*state);
  for (auto& clock : state->clocks_) {
    if (clock.record.graph == state->graph_record_.id) {
      clock.record.is_default = clock.record.key == key;
    }
  }
  for (auto& clock_import : state->clock_imports_) {
    if (clock_import.graph == state->graph_record_.id) {
      clock_import.is_default = clock_import.key == key;
    }
  }
  state->default_clock_ = std::move(key);
  state->mark_mutation();
  return {};
}

expected<ClockKey, Error> graph_default_clock(const std::shared_ptr<GraphState>& state) {
  auto prepared = state->prepare_observation();
  if (!prepared)
    return make_unexpected(std::move(prepared.error()));
  const std::lock_guard lock(state->mutex_);
  return state->default_clock_;
}

expected<void, Error> graph_add_clock_conversion(const std::shared_ptr<GraphState>& state,
                                                 ClockKey from, ClockKey to,
                                                 AffineClockConversion conversion) {
  auto prepared = state->prepare_mutation();
  if (!prepared)
    return prepared;
  if (from.is_default() || to.is_default() || from.name().empty() || to.name().empty() ||
      from == to || conversion.scale.numerator <= 0 || conversion.scale.denominator == 0U) {
    return make_unexpected(
        graph_error(ErrorCode::kInvalidArgument, "GRAPH_CLOCK_CONVERSION_INVALID"));
  }
  const auto divisor = std::gcd(static_cast<std::uint64_t>(conversion.scale.numerator),
                                conversion.scale.denominator);
  conversion.scale.numerator /= static_cast<std::int64_t>(divisor);
  conversion.scale.denominator /= divisor;
  const std::lock_guard lock(state->mutex_);
  if (!state->clock_names_.contains(std::string(from.name())) ||
      !state->clock_names_.contains(std::string(to.name()))) {
    return make_unexpected(graph_error(ErrorCode::kNotFound, "GRAPH_CLOCK_NOT_FOUND"));
  }
  for (const auto& existing : state->clock_conversions_) {
    if (existing.graph != state->graph_record_.id)
      continue;
    if (existing.from == from && existing.to == to) {
      if (existing.conversion == conversion)
        return {};
      return make_unexpected(
          graph_error(ErrorCode::kIncompatible, "GRAPH_CLOCK_CONVERSION_CONFLICT"));
    }
  }
  std::vector<ClockKey> pending{to};
  std::unordered_set<std::string> visited;
  while (!pending.empty()) {
    const ClockKey current = std::move(pending.back());
    pending.pop_back();
    if (current == from) {
      return make_unexpected(graph_error(ErrorCode::kIncompatible, "GRAPH_CLOCK_CONVERSION_CYCLE"));
    }
    if (!visited.emplace(current.name()).second)
      continue;
    for (const auto& existing : state->clock_conversions_) {
      if (existing.graph == state->graph_record_.id && existing.from == current)
        pending.push_back(existing.to);
    }
  }
  state->clock_conversions_.push_back(ClockConversionAuthoringRecord{
      .graph = state->graph_record_.id,
      .from = std::move(from),
      .to = std::move(to),
      .conversion = conversion,
  });
  state->mark_mutation();
  return {};
}

namespace {

using HashBuilder = holoscan::detail::StableHashBuilder;

template <typename T>
void hash_optional(HashBuilder& hash, const std::optional<T>& value) {
  hash.add(value.has_value());
  if (value)
    hash.add(*value);
}

void hash_bytes(HashBuilder& hash, std::span<const std::byte> bytes) {
  hash.add(static_cast<std::uint64_t>(bytes.size()));
  for (const std::byte value : bytes)
    hash.add_byte(static_cast<std::uint8_t>(value));
}

void hash_schema(HashBuilder& hash, const std::optional<SchemaIdentityRecord>& schema) {
  hash.add(schema.has_value());
  if (!schema)
    return;
  hash.add(schema->canonical_name);
  hash.add(schema->version);
  hash_bytes(hash, schema->schema_sha256);
  hash_bytes(hash, schema->id.value);
}

void hash_payload_identity(HashBuilder& hash, const PayloadIdentityRecord& identity) {
  hash.add(identity.kind);
  hash.add(identity.payload_kind);
  hash.add(identity.scalar.has_value());
  if (identity.scalar)
    hash.add(identity.scalar->kind);
  hash_schema(hash, identity.schema);
  hash.add(identity.pod.has_value());
  if (!identity.pod)
    return;
  hash.add(identity.pod->canonical_name);
  hash.add(identity.pod->version);
  hash.add(identity.pod->object_size);
  hash.add(identity.pod->object_alignment);
  hash.add(static_cast<std::uint64_t>(identity.pod->fields.size()));
  for (const auto& field : identity.pod->fields) {
    hash.add(field.name);
    hash.add(field.byte_offset);
    hash.add(field.scalar_kind);
    hash.add(field.element_count);
  }
}

void hash_tensor_layout(HashBuilder& hash, const std::optional<TensorPortLayout>& layout) {
  hash.add(layout.has_value());
  if (!layout)
    return;
  hash_optional(hash, layout->memory_kind);
  hash.add(layout->dtype.has_value());
  if (layout->dtype) {
    hash.add(layout->dtype->code);
    hash.add(layout->dtype->bits);
    hash.add(layout->dtype->lanes);
  }
  hash_optional(hash, layout->rank);
}

void hash_payload_options(HashBuilder& hash, const std::optional<PayloadOptionsRecord>& options) {
  hash.add(options.has_value());
  if (!options)
    return;
  hash_schema(hash, options->schema);
  hash_optional(hash, options->max_inline_bytes);
  hash_optional(hash, options->max_tensor_byte_span);
  hash.add(options->interface_tensor_emit_policy);
}

void hash_tensor_fields(HashBuilder& hash,
                        const std::vector<TensorFieldConfigurationRecord>& fields) {
  if (fields.empty()) {
    return;
  }
  hash.add(std::string_view{"tensor-fields-v1"});
  hash.add(static_cast<std::uint64_t>(fields.size()));
  for (const auto& field : fields) {
    hash.add(field.field_id);
    hash.add(field.tensor_index);
    hash_tensor_layout(hash, field.tensor_layout);
    hash_optional(hash, field.memory_conversion);
    hash.add(field.tensor_allocation.has_value());
    if (field.tensor_allocation) {
      hash.add(field.tensor_allocation->max_rank);
      hash.add(static_cast<std::uint64_t>(field.tensor_allocation->max_byte_span));
    }
    hash.add(field.payload_options.has_value());
    if (field.payload_options) {
      hash_optional(hash, field.payload_options->max_inline_bytes);
      hash_optional(hash, field.payload_options->max_tensor_byte_span);
      hash.add(field.payload_options->interface_tensor_emit_policy);
    }
  }
}

void hash_port_configuration(HashBuilder& hash, const PortConfigurationRecord& config) {
  hash_optional(hash, config.queue_depth);
  hash_optional(hash, config.max_emits_per_compute);
  hash_optional(hash, config.late_joiner_history_depth);
  hash.add(config.diagnostic_alias.has_value());
  if (config.diagnostic_alias)
    hash.add(*config.diagnostic_alias);
  hash_tensor_layout(hash, config.tensor_layout);
  hash_optional(hash, config.memory_conversion);
  hash.add(config.tensor_allocation.has_value());
  if (config.tensor_allocation) {
    hash.add(config.tensor_allocation->max_rank);
    hash.add(static_cast<std::uint64_t>(config.tensor_allocation->max_byte_span));
  }
  hash_payload_options(hash, config.payload_options);
  hash_tensor_fields(hash, config.tensor_fields);
}

void hash_connection(HashBuilder& hash, const ConnectionOptionsRecord& options) {
  hash.add(options.queue_policy);
  hash_optional(hash, options.queue_depth);
  hash.add(options.topic_name.has_value());
  if (options.topic_name)
    hash.add(*options.topic_name);
  hash_optional(hash, options.memory_conversion);
  hash_optional(hash, options.max_inflight_samples);
}

void hash_connection_override(HashBuilder& hash, const ConnectionOverrideRecord& options) {
  hash_optional(hash, options.queue_policy);
  hash_optional(hash, options.queue_depth);
  hash.add(options.topic_name.has_value());
  if (options.topic_name)
    hash.add(*options.topic_name);
  hash_optional(hash, options.memory_conversion);
  hash_optional(hash, options.max_inflight_samples);
}

void hash_clock_key(HashBuilder& hash, const ClockKey& key) {
  hash.add(key.is_default());
  hash.add(key.name());
}

// The canonical hash serializer remains visibly ordered with the GraphView record layout. Splitting
// the record sequence across helpers would obscure the compatibility-sensitive field order.
GraphAuthoringHash compute_graph_hash(  // NOLINT(readability-function-cognitive-complexity)
    const GraphSnapshotData& snapshot) {
  HashBuilder hash;
  hash.add("holoscan.graph.authoring.v1");
  hash.add(static_cast<std::uint64_t>(snapshot.graphs.size()));
  for (const auto& graph : snapshot.graphs) {
    hash.add(graph.id.value);
    hash.add(graph.parent.has_value());
    if (graph.parent)
      hash.add(graph.parent->value);
    hash.add(graph.name);
    hash.add(graph.path.has_value());
    if (graph.path) {
      hash.add(static_cast<std::uint64_t>(graph.path->segments().size()));
      for (const auto& segment : graph.path->segments())
        hash.add(segment);
    }
  }
  hash.add(static_cast<std::uint64_t>(snapshot.vertices.size()));
  for (const auto& vertex : snapshot.vertices) {
    hash.add(vertex.id.value);
    hash.add(vertex.graph.value);
    hash.add(vertex.kind);
    hash.add(vertex.name);
  }
  hash.add(static_cast<std::uint64_t>(snapshot.ports.size()));
  for (const auto& port : snapshot.ports) {
    hash.add(port.id.value);
    hash.add(port.owner.value);
    hash.add(port.owner_kind);
    hash.add(port.direction);
    hash.add(port.name);
    hash_payload_identity(hash, port.payload_identity);
    hash.add(port.serialized_payload_capacity_bytes);
    hash.add(port.generic_loan_supported);
    hash.add(port.generic_loan_slot_size_bytes);
    hash_port_configuration(hash, port.setup);
    hash_port_configuration(hash, port.graph_override);
  }
  hash.add(static_cast<std::uint64_t>(snapshot.edges.size()));
  for (const auto& edge : snapshot.edges) {
    hash.add(edge.id.value);
    hash.add(edge.source.value);
    hash.add(edge.target.value);
    hash_connection(hash, edge.add_flow_options);
    hash_connection_override(hash, edge.graph_override);
    hash_payload_options(hash, edge.effective_tensor_payload);
  }
  hash.add(static_cast<std::uint64_t>(snapshot.interface_bindings.size()));
  for (const auto& binding : snapshot.interface_bindings) {
    hash.add(binding.public_port.value);
    hash.add(binding.internal_port.value);
    hash.add(binding.direction);
  }
  hash.add(static_cast<std::uint64_t>(snapshot.external_topics.size()));
  for (const auto& topic : snapshot.external_topics) {
    hash.add(topic.id.value);
    hash.add(topic.vertex.value);
    hash.add(topic.graph_port.value);
    hash.add(topic.imported);
    hash.add(topic.uri);
    hash.add(topic.endpoint_name.has_value());
    if (topic.endpoint_name)
      hash.add(*topic.endpoint_name);
    hash.add(topic.generated_name_requested);
    hash.add(topic.declaration_ordinal);
    hash_payload_identity(hash, topic.payload_identity);
    hash.add(topic.serialized_payload_capacity_bytes);
    hash_schema(hash, topic.schema);
    hash_optional(hash, topic.memory_conversion);
    hash.add(topic.capture_time.has_value());
    if (topic.capture_time)
      hash_clock_key(hash, topic.capture_time->clock);
    hash.add(topic.clock_health.capability);
    hash.add(topic.clock_health.requirement);
    hash.add(topic.clock_health.max_report_age.has_value());
    if (topic.clock_health.max_report_age)
      hash.add(topic.clock_health.max_report_age->count());
    hash.add(topic.clock_health_authored);
    hash.add(topic.attach.peer);
    hash.add(topic.attach.max_wait.has_value());
    if (topic.attach.max_wait)
      hash.add(topic.attach.max_wait->count());
    hash.add(topic.tensor.has_value());
    if (topic.tensor) {
      hash_optional(hash, topic.tensor->memory_kind);
      hash.add(topic.tensor->dtype.has_value());
      if (topic.tensor->dtype) {
        hash.add(topic.tensor->dtype->code);
        hash.add(topic.tensor->dtype->bits);
        hash.add(topic.tensor->dtype->lanes);
      }
      hash_optional(hash, topic.tensor->rank);
      hash_optional(hash, topic.tensor->max_rank);
      hash_optional(hash, topic.tensor->max_byte_span);
      hash.add(static_cast<std::uint64_t>(topic.tensor->axes.size()));
      for (const auto& axis : topic.tensor->axes) {
        hash.add(axis.max_extent);
        hash.add(axis.max_abs_stride_bytes);
      }
      hash.add(topic.tensor->strides);
      hash.add(topic.tensor->foreign_emit_policy);
      hash_optional(hash, topic.tensor->max_inline_bytes);
    }
    hash.add(topic.bounds.publisher.has_value());
    if (topic.bounds.publisher)
      hash_optional(hash, topic.bounds.publisher->max_loaned_samples);
    hash.add(topic.bounds.subscriber.has_value());
    if (topic.bounds.subscriber) {
      hash_optional(hash, topic.bounds.subscriber->queue_depth);
      hash_optional(hash, topic.bounds.subscriber->max_borrowed_samples);
    }
  }
  hash.add(static_cast<std::uint64_t>(snapshot.readiness_sources.size()));
  for (const auto& source : snapshot.readiness_sources) {
    hash.add(source.owner.value);
    hash.add(source.declaration_ordinal);
    hash.add(source.kind);
    hash.add(source.name);
    hash.add(source.requested_capacity);
    hash.add(source.requested_sender_reference_capacity);
    hash_clock_key(hash, source.requested_clock);
    hash.add(source.wake_requirement);
    hash.add(source.referenced);
  }
  hash.add(static_cast<std::uint64_t>(snapshot.regions.size()));
  for (const auto& region : snapshot.regions) {
    hash.add(region.id.value);
    hash.add(region.graph.value);
    hash.add(region.name);
    hash.add(static_cast<std::uint64_t>(region.members.size()));
    for (const VertexId member : region.members)
      hash.add(member.value);
    hash.add(static_cast<std::uint64_t>(region.graph_members.size()));
    for (const GraphId member : region.graph_members)
      hash.add(member.value);
    hash.add(static_cast<std::uint64_t>(region.constraints.size()));
    for (const auto& constraint : region.constraints) {
      hash.add(constraint.kind());
      hash.add(constraint.latency().count());
    }
    hash.add(static_cast<std::uint64_t>(region.hints.size()));
    for (auto hint : region.hints)
      hash.add(hint);
    hash_optional(hash, region.profile);
  }
  hash.add(static_cast<std::uint64_t>(snapshot.partitions.size()));
  for (const auto& partition : snapshot.partitions) {
    hash.add(partition.id.value);
    hash.add(partition.graph.value);
    hash.add(partition.name);
    hash.add(static_cast<std::uint64_t>(partition.members.size()));
    for (const VertexId member : partition.members)
      hash.add(member.value);
  }
  hash.add(static_cast<std::uint64_t>(snapshot.clocks.size()));
  for (const auto& clock : snapshot.clocks) {
    hash.add(clock.graph.value);
    hash_clock_key(hash, clock.key);
    hash.add(clock.shared);
    hash.add(clock.is_default);
  }
  hash.add(static_cast<std::uint64_t>(snapshot.clock_capabilities.size()));
  for (const auto& capability : snapshot.clock_capabilities) {
    hash.add(capability.graph.value);
    hash_clock_key(hash, capability.key);
    hash.add(capability.can_read_now);
    hash.add(capability.can_wait_until);
    hash.add(capability.steady);
    hash.add(capability.independent_progress);
    hash.add(capability.finite_deadline);
    hash.add(capability.external_driver_admitted);
    hash.add(capability.evidence_id);
  }
  hash.add(static_cast<std::uint64_t>(snapshot.clock_imports.size()));
  for (const auto& clock_import : snapshot.clock_imports) {
    hash.add(clock_import.graph.value);
    hash_clock_key(hash, clock_import.key);
    hash.add(clock_import.is_default);
  }
  hash.add(static_cast<std::uint64_t>(snapshot.clock_bindings.size()));
  for (const auto& binding : snapshot.clock_bindings) {
    hash.add(binding.child_graph.value);
    hash_clock_key(hash, binding.child_clock);
    hash.add(binding.parent_graph.value);
    hash_clock_key(hash, binding.parent_clock);
  }
  auto clock_conversions = snapshot.clock_conversions;
  std::sort(
      clock_conversions.begin(), clock_conversions.end(), [](const auto& lhs, const auto& rhs) {
        return std::tuple{lhs.from.is_default(),
                          lhs.graph.value,
                          lhs.from.name(),
                          lhs.to.is_default(),
                          lhs.to.name(),
                          lhs.conversion.scale.numerator,
                          lhs.conversion.scale.denominator,
                          lhs.conversion.offset_ns} < std::tuple{rhs.from.is_default(),
                                                                 rhs.graph.value,
                                                                 rhs.from.name(),
                                                                 rhs.to.is_default(),
                                                                 rhs.to.name(),
                                                                 rhs.conversion.scale.numerator,
                                                                 rhs.conversion.scale.denominator,
                                                                 rhs.conversion.offset_ns};
      });
  hash.add(static_cast<std::uint64_t>(clock_conversions.size()));
  for (const auto& conversion : clock_conversions) {
    hash.add(conversion.graph.value);
    hash_clock_key(hash, conversion.from);
    hash_clock_key(hash, conversion.to);
    hash.add(conversion.conversion.scale.numerator);
    hash.add(conversion.conversion.scale.denominator);
    hash.add(conversion.conversion.offset_ns);
  }
  auto lifecycle_dependencies = snapshot.lifecycle_dependencies;
  std::sort(lifecycle_dependencies.begin(),
            lifecycle_dependencies.end(),
            [](const auto& lhs, const auto& rhs) {
              return std::tuple{lhs.graph.value, lhs.stage, lhs.before.value, lhs.after.value} <
                     std::tuple{rhs.graph.value, rhs.stage, rhs.before.value, rhs.after.value};
            });
  hash.add(static_cast<std::uint64_t>(lifecycle_dependencies.size()));
  for (const auto& dependency : lifecycle_dependencies) {
    hash.add(dependency.graph.value);
    hash.add(dependency.before.value);
    hash.add(dependency.after.value);
    hash.add(dependency.stage);
  }
  auto lifecycle_callbacks = snapshot.lifecycle_callbacks;
  std::sort(
      lifecycle_callbacks.begin(), lifecycle_callbacks.end(), [](const auto& lhs, const auto& rhs) {
        return std::tuple{lhs.owner.value, lhs.stage, lhs.required, lhs.method, lhs.origin} <
               std::tuple{rhs.owner.value, rhs.stage, rhs.required, rhs.method, rhs.origin};
      });
  hash.add(static_cast<std::uint64_t>(lifecycle_callbacks.size()));
  for (const auto& callback : lifecycle_callbacks) {
    hash.add(callback.owner.value);
    hash.add(callback.stage);
    hash.add(callback.required);
    hash.add(callback.method);
    hash.add(callback.origin);
  }
  auto lifecycle_facts = snapshot.lifecycle_facts;
  std::sort(lifecycle_facts.begin(), lifecycle_facts.end(), [](const auto& lhs, const auto& rhs) {
    return std::tuple{lhs.owner.value, lhs.stage, lhs.kind, lhs.stable_name} <
           std::tuple{rhs.owner.value, rhs.stage, rhs.kind, rhs.stable_name};
  });
  hash.add(static_cast<std::uint64_t>(lifecycle_facts.size()));
  for (const auto& fact : lifecycle_facts) {
    hash.add(fact.owner.value);
    hash.add(fact.stage);
    hash.add(fact.kind);
    hash.add(fact.stable_name);
  }
  auto lifecycle_permissions = snapshot.lifecycle_restart_permissions;
  std::sort(lifecycle_permissions.begin(),
            lifecycle_permissions.end(),
            [](const auto& lhs, const auto& rhs) {
              return std::tuple{lhs.origin,
                                lhs.requester.has_value(),
                                lhs.requester ? lhs.requester->value : 0U,
                                lhs.target.value} <
                     std::tuple{rhs.origin,
                                rhs.requester.has_value(),
                                rhs.requester ? rhs.requester->value : 0U,
                                rhs.target.value};
            });
  hash.add(static_cast<std::uint64_t>(lifecycle_permissions.size()));
  for (const auto& permission : lifecycle_permissions) {
    hash.add(permission.origin);
    hash.add(permission.requester.has_value());
    if (permission.requester)
      hash.add(permission.requester->value);
    hash.add(permission.target.value);
  }
  return GraphAuthoringHash{.value = hash.finish().value};
}

}  // namespace

std::shared_ptr<GraphState> make_graph_state(Graph* owner, std::string_view name) {
  auto valid = validate_name(name, "GRAPH_NAME_INVALID", true);
  if (!valid)
    throw RuntimeError(valid.error());
  return std::make_shared<GraphState>(owner, std::string(name));
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

  // Child composition and deep snapshot creation completed before this call,
  // so publication reuses that immutable snapshot without repeating its copy,
  // canonicalization, or hash. The terminal child is exclusively owned here
  // and cannot mutate, so parent commit need not hold the child mutex. This is
  // essential: rollback may destroy the child Graph, whose destructor locks
  // that mutex to detach its state.
  const std::lock_guard lock(parent->mutex_);
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

  for (std::size_t index = 0; index < child_state->operators_.size(); ++index) {
    parent->embedded_operator_mappings_.push_back(EmbeddedOperatorMapping{
        .source_state = child_state.get(),
        .source_index = index,
        .vertex = vertex_ids.at(child_state->operators_[index].vertex.value),
        .factory = child_state->operators_[index].factory->clone(),
    });
    GraphMutationRollback::checkpoint(*parent);
  }
  for (const auto& mapping : child_state->embedded_operator_mappings_) {
    parent->embedded_operator_mappings_.push_back(EmbeddedOperatorMapping{
        .source_state = mapping.source_state,
        .source_index = mapping.source_index,
        .vertex = vertex_ids.at(mapping.vertex.value),
        .factory = mapping.factory->clone(),
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
    parent->edges_.push_back(std::move(copied));
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
    parent->regions_.push_back(std::move(region));
    GraphMutationRollback::checkpoint(*parent);
  }

  std::vector<PartitionRecord> partitions = child_snapshot->partitions;
  for (auto& partition : partitions) {
    partition.id = parent->next_partition_id();
    partition.graph = graph_ids.at(partition.graph.value);
    for (auto& member : partition.members)
      member = vertex_ids.at(member.value);
    // Partition names are Graph-scoped. The root authoring index deliberately
    // excludes embedded Graph records, whose local names may repeat in sibling
    // instances without colliding with a root partition.
    parent->partitions_.push_back(std::move(partition));
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
  });
  parent->mark_mutation();
  GraphMutationRollback::checkpoint(*parent);
  EmbeddedGraphRegistration registration{
      .index = embedded_index,
  };
  rollback.commit();
  graph_log_noexcept([&] {
    HOLOSCAN_LOG_TRACE("Embedded Graph instance '{}' with {} flattened vertices",
                       instance_name,
                       child_snapshot->vertices.size());
  });
  return registration;
}

const Graph* graph_embedded_graph(const std::shared_ptr<GraphState>& state,
                                  std::size_t index) noexcept {
  const std::lock_guard lock(state->mutex_);
  if (index >= state->embedded_graphs_.size())
    return nullptr;
  return state->embedded_graphs_[index].graph.get();
}

bool graph_embedded_valid(const std::shared_ptr<GraphState>& state, std::size_t index) noexcept {
  return graph_embedded_graph(state, index) != nullptr;
}

std::optional<GraphInstancePath> graph_embedded_path(const std::shared_ptr<GraphState>& state,
                                                     std::size_t index) {
  const std::lock_guard lock(state->mutex_);
  if (index >= state->embedded_graphs_.size())
    return std::nullopt;
  return state->embedded_graphs_[index].path;
}

std::shared_ptr<const GraphSnapshotData> graph_snapshot(const std::shared_ptr<GraphState>& state) {
  auto prepared = state->prepare_observation();
  if (!prepared)
    throw RuntimeError(prepared.error());
  const std::lock_guard lock(state->mutex_);
  auto snapshot = std::make_shared<GraphSnapshotData>();
  snapshot->generation = state->generation_;
  snapshot->graphs.push_back(state->graph_record_);
  snapshot->graphs.insert(snapshot->graphs.end(),
                          state->embedded_graph_records_.begin(),
                          state->embedded_graph_records_.end());
  snapshot->vertices = state->vertices_;
  snapshot->ports.reserve(state->ports_.size());
  for (const auto& port : state->ports_)
    snapshot->ports.push_back(port.record);
  snapshot->edges = state->edges_;
  snapshot->interface_bindings = state->interface_bindings_;
  snapshot->external_topics.reserve(state->external_topics_.size());
  for (const auto& topic : state->external_topics_) {
    snapshot->external_topics.push_back(topic.record);
  }
  snapshot->readiness_sources = state->readiness_sources_;
  snapshot->regions = state->regions_;
  snapshot->partitions = state->partitions_;
  snapshot->clocks.reserve(state->clocks_.size());
  snapshot->clock_capabilities.reserve(state->clocks_.size());
  for (const auto& clock : state->clocks_) {
    snapshot->clocks.push_back(clock.record);
    snapshot->clock_capabilities.push_back(clock.capabilities);
  }
  snapshot->clock_imports = state->clock_imports_;
  snapshot->clock_bindings = state->clock_bindings_;
  snapshot->clock_conversions = state->clock_conversions_;
  snapshot->lifecycle_dependencies = state->lifecycle_dependencies_;
  snapshot->lifecycle_callbacks = state->lifecycle_callbacks_;
  snapshot->lifecycle_facts = state->lifecycle_facts_;
  snapshot->lifecycle_restart_permissions = state->lifecycle_restart_permissions_;
  const GraphSnapshotData canonical = canonicalize_graph_snapshot(*snapshot);
  snapshot->identity = GraphAuthoringIdentity{
      .graph = state->instance_id_,
      .version = snapshot->generation,
      .hash = compute_graph_hash(canonical),
  };
  return snapshot;
}

expected<TensorOutputPortConfigurator*, Error> graph_tensor_configurator(
    const std::shared_ptr<GraphState>& state, GraphPortId id) {
  const std::lock_guard lock(state->mutex_);
  for (auto& configurator : state->tensor_configurators_) {
    if (configurator.port_ == id)
      return &configurator;
  }
  try {
    GraphMutationRollback::checkpoint(*state);
    state->tensor_configurators_.push_back(TensorOutputPortConfigurator{state, id});
  } catch (const std::bad_alloc&) {
    // Preserve the nonthrowing expected path even while the allocator is
    // exhausted: constructing an optional diagnostic string could throw here.
    return make_unexpected(Error{.code = ErrorCode::kResourceExhausted});
  }
  return &state->tensor_configurators_.back();
}

}  // namespace detail

RegionConstraint RegionConstraint::static_topology() noexcept {
  return RegionConstraint{RegionConstraintKind::kStaticTopology, {}};
}

RegionConstraint RegionConstraint::bounded_memory() noexcept {
  return RegionConstraint{RegionConstraintKind::kBoundedMemory, {}};
}

RegionConstraint RegionConstraint::device_compatible_ops() noexcept {
  return RegionConstraint{RegionConstraintKind::kDeviceCompatibleOperators, {}};
}

RegionConstraint RegionConstraint::latency_budget(std::chrono::nanoseconds budget) {
  if (budget.count() <= 0) {
    throw RuntimeError(Error{ErrorCode::kInvalidArgument, "GRAPH_REGION_LATENCY_INVALID"});
  }
  return RegionConstraint{RegionConstraintKind::kLatencyBudget, budget};
}

TensorOutputPortConfigurator& TensorOutputPortConfigurator::max_emits_per_compute(
    std::uint32_t count) {
  apply(detail::graph_configure_output_emits(state(), port_, count));
  return *this;
}

TensorOutputPortConfigurator& TensorOutputPortConfigurator::late_joiner_history_depth(
    std::uint32_t depth) {
  apply(detail::graph_configure_output_history(state(), port_, depth));
  return *this;
}

TensorOutputPortConfigurator& TensorOutputPortConfigurator::produces_tensor(
    TensorPortLayout layout) {
  apply(detail::graph_configure_output_tensor(state(), port_, layout));
  return *this;
}

TensorOutputPortConfigurator& TensorOutputPortConfigurator::payload_options(
    PayloadOptions options) {
  apply(detail::graph_configure_output_payload(state(), port_, std::move(options)));
  return *this;
}

TensorOutputPortConfigurator& TensorOutputPortConfigurator::diagnostic_alias(
    std::string_view alias) {
  apply(detail::graph_configure_port_alias(state(), port_, alias));
  return *this;
}

TensorOutputPortConfigurator& TensorOutputPortConfigurator::tensor_allocation(
    TensorAllocationBounds bounds) {
  apply(detail::graph_configure_tensor_allocation(state(), port_, bounds));
  return *this;
}

std::shared_ptr<detail::GraphState> TensorOutputPortConfigurator::state() const {
  auto value = state_.lock();
  if (!value) {
    detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"});
  }
  return value;
}

void TensorOutputPortConfigurator::apply(expected<void, Error> result) {
  if (!result)
    detail::throw_graph_error(std::move(result.error()));
}

RegionHandle& RegionHandle::constrain(RegionConstraint constraint) {
  auto state = state_.lock();
  if (!state) {
    detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_REGION_HANDLE_INVALID"});
  }
  auto result = detail::graph_region_add_constraint(state, id_, constraint);
  if (!result)
    detail::throw_graph_error(std::move(result.error()));
  return *this;
}

RegionHandle& RegionHandle::optimize(OptimizationHint hint) {
  auto state = state_.lock();
  if (!state) {
    detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_REGION_HANDLE_INVALID"});
  }
  auto result = detail::graph_region_add_hint(state, id_, hint);
  if (!result)
    detail::throw_graph_error(std::move(result.error()));
  return *this;
}

RegionHandle& RegionHandle::profile(Profile profile) {
  auto state = state_.lock();
  if (!state) {
    detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_REGION_HANDLE_INVALID"});
  }
  auto result = detail::graph_region_set_profile(state, id_, profile);
  if (!result)
    detail::throw_graph_error(std::move(result.error()));
  return *this;
}

GraphGeneration GraphView::generation() const noexcept {
  return snapshot_ ? snapshot_->generation : 0U;
}

GraphAuthoringIdentity GraphView::identity() const noexcept {
  return snapshot_ ? snapshot_->identity : GraphAuthoringIdentity{};
}

std::span<const GraphRecord> GraphView::graphs() const noexcept {
  return snapshot_ ? std::span<const GraphRecord>{snapshot_->graphs}
                   : std::span<const GraphRecord>{};
}

std::span<const VertexRecord> GraphView::vertices() const noexcept {
  return snapshot_ ? std::span<const VertexRecord>{snapshot_->vertices}
                   : std::span<const VertexRecord>{};
}

std::span<const PortRecord> GraphView::ports() const noexcept {
  return snapshot_ ? std::span<const PortRecord>{snapshot_->ports} : std::span<const PortRecord>{};
}

std::span<const EdgeRecord> GraphView::edges() const noexcept {
  return snapshot_ ? std::span<const EdgeRecord>{snapshot_->edges} : std::span<const EdgeRecord>{};
}

std::span<const InterfaceBindingRecord> GraphView::interface_bindings() const noexcept {
  return snapshot_ ? std::span<const InterfaceBindingRecord>{snapshot_->interface_bindings}
                   : std::span<const InterfaceBindingRecord>{};
}

std::span<const ExternalTopicRecord> GraphView::external_topics() const noexcept {
  return snapshot_ ? std::span<const ExternalTopicRecord>{snapshot_->external_topics}
                   : std::span<const ExternalTopicRecord>{};
}

std::span<const ReadinessSourceDeclarationRecord> GraphView::readiness_sources() const noexcept {
  return snapshot_ ? std::span<const ReadinessSourceDeclarationRecord>{snapshot_->readiness_sources}
                   : std::span<const ReadinessSourceDeclarationRecord>{};
}

std::span<const RegionRecord> GraphView::regions() const noexcept {
  return snapshot_ ? std::span<const RegionRecord>{snapshot_->regions}
                   : std::span<const RegionRecord>{};
}

std::span<const PartitionRecord> GraphView::partitions() const noexcept {
  return snapshot_ ? std::span<const PartitionRecord>{snapshot_->partitions}
                   : std::span<const PartitionRecord>{};
}

std::span<const ClockDeclarationRecord> GraphView::clocks() const noexcept {
  return snapshot_ ? std::span<const ClockDeclarationRecord>{snapshot_->clocks}
                   : std::span<const ClockDeclarationRecord>{};
}

std::span<const ClockImportAuthoringRecord> GraphView::clock_imports() const noexcept {
  return snapshot_ ? std::span<const ClockImportAuthoringRecord>{snapshot_->clock_imports}
                   : std::span<const ClockImportAuthoringRecord>{};
}

std::span<const EmbeddedClockBindingAuthoringRecord> GraphView::clock_bindings() const noexcept {
  return snapshot_ ? std::span<const EmbeddedClockBindingAuthoringRecord>{snapshot_->clock_bindings}
                   : std::span<const EmbeddedClockBindingAuthoringRecord>{};
}

std::span<const ClockConversionAuthoringRecord> GraphView::clock_conversions() const noexcept {
  return snapshot_ ? std::span<const ClockConversionAuthoringRecord>{snapshot_->clock_conversions}
                   : std::span<const ClockConversionAuthoringRecord>{};
}

std::span<const LifecycleDependencyAuthoringRecord> GraphView::lifecycle_dependencies()
    const noexcept {
  return snapshot_
             ? std::span<const LifecycleDependencyAuthoringRecord>{snapshot_
                                                                       ->lifecycle_dependencies}
             : std::span<const LifecycleDependencyAuthoringRecord>{};
}

std::span<const LifecycleRestartPermissionAuthoringRecord>
GraphView::lifecycle_restart_permissions() const noexcept {
  if (snapshot_) {
    return std::span<const LifecycleRestartPermissionAuthoringRecord>{
        snapshot_->lifecycle_restart_permissions};
  }
  return {};
}

Graph::Graph(std::string_view name) : state_(detail::make_graph_state(this, name)) {}

Graph::~Graph() {
  state_->detach();
}

RegionHandle Graph::region(std::string_view name) {
  auto result = detail::graph_add_region(state_, name);
  if (!result)
    detail::throw_graph_error(std::move(result.error()));
  return RegionHandle{state_, result.value()};
}

PartitionHandle Graph::partition(std::string_view name) {
  auto result = detail::graph_add_partition(state_, name);
  if (!result)
    detail::throw_graph_error(std::move(result.error()));
  return PartitionHandle{state_, result.value()};
}

GraphLifecycleBuilder Graph::lifecycle() {
  auto prepared = state_->prepare_mutation();
  if (!prepared)
    detail::throw_graph_error(std::move(prepared.error()));
  return GraphLifecycleBuilder{state_};
}

GraphView Graph::view() {
  return GraphView{detail::graph_snapshot(state_)};
}

ClockKey Graph::bind_shared_clock(std::string_view name, std::shared_ptr<Clock> clock) {
  auto result = detail::graph_bind_shared_clock(state_, name, std::move(clock));
  if (!result)
    detail::throw_graph_error(std::move(result.error()));
  return std::move(result.value());
}

ClockKey Graph::import_clock(std::string_view name) {
  auto result = detail::graph_import_clock(state_, name);
  if (!result)
    detail::throw_graph_error(std::move(result.error()));
  return std::move(result.value());
}

Graph& Graph::set_default_clock(ClockKey key) {
  auto result = detail::graph_set_default_clock(state_, std::move(key));
  if (!result)
    detail::throw_graph_error(std::move(result.error()));
  return *this;
}

ClockKey Graph::default_clock() {
  auto result = detail::graph_default_clock(state_);
  if (!result)
    detail::throw_graph_error(std::move(result.error()));
  return std::move(result.value());
}

Graph& Graph::add_clock_conversion(ClockKey from, ClockKey to, AffineClockConversion conversion) {
  auto result =
      detail::graph_add_clock_conversion(state_, std::move(from), std::move(to), conversion);
  if (!result)
    detail::throw_graph_error(std::move(result.error()));
  return *this;
}

}  // namespace holoscan
