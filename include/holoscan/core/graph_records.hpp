/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <dlpack/dlpack.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/channel/clock_health.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/external_topic_types.hpp>
#include <holoscan/core/graph_ids.hpp>
#include <holoscan/core/lifecycle_types.hpp>
#include <holoscan/core/memory_kind.hpp>
#include <holoscan/core/operator_types.hpp>
#include <holoscan/core/payload_identity.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/port_types.hpp>
#include <holoscan/core/profile.hpp>
#include <holoscan/core/rate.hpp>
#include <holoscan/core/readiness_source.hpp>
#include <holoscan/core/tensor_contract_types.hpp>
#include <holoscan/core/tensor_field.hpp>
#include <holoscan/time/clock_registry.hpp>

namespace holoscan {

namespace detail {
class GraphState;
class GraphCompilerAccess;
struct GraphSnapshotData;
}  // namespace detail

/// @brief Preserve logical Tensor boundary requirements without retaining spans.
struct TensorBoundaryContractRecord {
  std::optional<MemoryKind> memory_kind;       ///< Exact required memory kind when present.
  std::optional<DLDataType> dtype;             ///< Exact required element type when present.
  std::optional<std::uint32_t> rank;           ///< Exact logical rank when present; zero is scalar.
  std::optional<std::uint32_t> max_rank;       ///< Positive maximum rank when bounded.
  std::optional<std::uint64_t> max_byte_span;  ///< Positive maximum logical byte span.
  std::vector<TensorAxisBounds> axes;          ///< Owned per-axis extent and stride bounds.
  TensorStridePolicy strides{TensorStridePolicy::kContiguousOnly};  ///< Layout policy.
  InterfaceTensorEmitPolicy foreign_emit_policy{
      InterfaceTensorEmitPolicy::kRequireExportable};  ///< Foreign storage policy.
  std::optional<std::uint32_t> max_inline_bytes;       ///< Positive inline cap when authored.
  /// Positive foreign-storage staging cap when copying is admitted.
  std::optional<std::uint64_t> max_staging_byte_span;
  /// Consumer-side memory conversion policy at this external boundary.
  std::optional<MemoryConversionPolicy> memory_conversion;
  TensorContractOrigins origins;  ///< Cold diagnostic authored-versus-derived provenance.
};

/// @brief Preserve one canonical descriptor-schema external Tensor field contract.
struct DescriptorTensorTopicFieldRecord {
  std::uint16_t field_id{};             ///< Canonical FlatBuffers field id.
  std::uint16_t tensor_index{};         ///< Dense index in ascending field-id order.
  TensorBoundaryContractRecord tensor;  ///< Complete finite field boundary contract.
};

/// @brief Preserve one copied edge-options record and its authored presence bits.
struct ConnectionOptionsRecord {
  QueuePolicy queue_policy{QueuePolicy::kBlocking};         ///< Authored finite-queue behavior.
  std::optional<std::uint32_t> queue_depth;                 ///< Positive authored queue depth.
  std::optional<std::string> topic_name;                    ///< Owned logical flow alias.
  std::optional<MemoryConversionPolicy> memory_conversion;  ///< Authored Tensor conversion rule.
  std::optional<std::uint32_t> max_inflight_samples;        ///< Positive authored retention bound.
};

/// @brief Preserve only flow options explicitly changed after edge creation.
struct ConnectionOverrideRecord {
  std::optional<QueuePolicy> queue_policy;   ///< Replacement queue policy when authored.
  std::optional<std::uint32_t> queue_depth;  ///< Positive replacement depth.
  std::optional<std::string> topic_name;     ///< Owned replacement logical alias.
  std::optional<MemoryConversionPolicy> memory_conversion;  ///< Replacement conversion request.
  std::optional<std::uint32_t> max_inflight_samples;  ///< Positive replacement retention bound.
};

/// @brief Preserve one authored port-configuration layer.
///
/// Setup defaults and Graph overrides occupy separate instances so compilation can
/// resolve precedence without reconstructing author intent.
struct PortConfigurationRecord {
  std::optional<std::uint32_t> queue_depth;                ///< Input history depth when present.
  std::optional<std::uint32_t> max_emits_per_compute;      ///< Output burst bound when present.
  std::optional<std::uint32_t> late_joiner_history_depth;  ///< Output retained history.
  PortRateKind rate_kind{PortRateKind::kUnspecified};      ///< Normalized setup rate form.
  std::optional<Rate> declared_nominal_rate;               ///< Fixed output nominal rate.
  std::optional<Rate> declared_expected_rate;              ///< Advisory input expectation.
  std::optional<GraphPortId> follows_input;                ///< Canonical owner-local input.
  std::optional<RateRatio> follows_ratio;                  ///< Exact follows multiplier.
  std::optional<LifecycleStage> lifecycle_binding_stage;   ///< Output binding deadline.
  /// Maximum selectable nominal average for a lifecycle-selected root output.
  std::optional<Rate> declared_max_nominal_rate;
  std::optional<std::string> diagnostic_alias;              ///< Owned diagnostic-only alias.
  std::optional<TensorRepresentation> tensor_layout;        ///< Logical Tensor facts when authored.
  std::optional<TensorContractRecord> tensor_contract;      ///< Owned normalized role contract.
  std::optional<MemoryConversionPolicy> memory_conversion;  ///< Port conversion request.
  std::optional<TensorPoolBounds> tensor_allocation;        ///< Direct Tensor allocation bound.
  /// Immutable compile-derived Tensor slot-capacity expression when authored.
  std::optional<TensorByteSpanExpressionRecord> tensor_byte_span_expression;
  /// Immutable compile-derived Tensor memory-kind expression when authored.
  std::optional<TensorMemoryKindExpressionRecord> tensor_memory_kind_expression;
  std::optional<PayloadOptionsRecord> payload_options;  ///< Deep-owned producer options.
  /// True when a Tensor input admits a Tensor projected out of a producer's schema payload.
  bool accepts_projected_tensor{};
  /// Field-scoped Tensor facts in canonical ascending field-id order.
  std::vector<TensorFieldConfigurationRecord> tensor_fields;
};

/// @brief Describe one Graph instance in an immutable authoring snapshot.
struct GraphRecord {
  GraphId id{};                   ///< Nonzero identity within this authoring snapshot.
  std::optional<GraphId> parent;  ///< Parent reusable Graph when nested; absent for root.
  std::string name;               ///< Owned local Graph name, possibly empty for an unnamed root.
  std::optional<GraphInstancePath> path;  ///< Canonical path; absent only for root scope.
};

/// @brief Describe one logical authoring vertex.
struct VertexRecord {
  VertexId id{};                           ///< Nonzero snapshot-local vertex identity.
  GraphId graph{};                         ///< Owning Graph record.
  VertexKind kind{VertexKind::kOperator};  ///< Logical authoring kind before lowering.
  std::string name;                        ///< Owned authoring name.
  OperatorHostKind operator_host{OperatorHostKind::kNative};  ///< Callback implementation kind.
  /// Owned callback serialization-domain key copied from host metadata. Equal nonempty keys request
  /// compatible co-partitioning (for example, Python callbacks sharing the GIL); an empty key
  /// permits independent partitioning.
  std::string serialization_domain;
  bool cuda_graph_capture_eligible{};  ///< Operator `compute()` capture eligibility declaration.
};

/// @brief Describe one copied port declaration.
struct PortRecord {
  GraphPortId id{};  ///< Snapshot-local authoring identity, never a plan \c PortId.
  VertexId owner{};  ///< Owning logical vertex.
  PortOwnerKind owner_kind{PortOwnerKind::kOperator};  ///< Authoring owner classification.
  PortDirection direction{PortDirection::kInput};      ///< Input or output direction.
  bool optional{};   ///< Operator-owned permission for eligible Exact/SafeDefault input absence.
  std::string name;  ///< Owned declaration-local name.
  PayloadIdentityRecord payload_identity;  ///< Complete pointer-free semantic identity.
  /// Positive generated-package serialized bound; zero for every other payload family.
  std::uint64_t serialized_payload_capacity_bytes{};
  /// Whether the exact typed output supports bounded generic-loan construction.
  bool generic_loan_supported{};
  /// Exact native value-slot size when \ref generic_loan_supported is true; otherwise zero.
  std::uint64_t generic_loan_slot_size_bytes{};
  /// Direct Tensor field count for descriptor-backed schemas; zero for other payload families.
  std::uint16_t descriptor_tensor_field_count{};
  PortConfigurationRecord setup;           ///< OperatorSpec declaration defaults.
  PortConfigurationRecord graph_override;  ///< Later Graph configurator requests.
};

/// @brief Describe one typed authored edge.
struct EdgeRecord {
  EdgeId id{};                               ///< Nonzero snapshot-local edge identity.
  GraphPortId source{};                      ///< Snapshot-local source output identity.
  GraphPortId target{};                      ///< Snapshot-local target input identity.
  ConnectionOptionsRecord add_flow_options;  ///< Options copied at edge creation.
  ConnectionOverrideRecord graph_override;   ///< Later flow-configurator requests.
  std::optional<PayloadOptionsRecord> effective_tensor_payload;  ///< Normalized Tensor policy.
  /// Producer Tensor field delivered when this edge narrows a schema payload to a Tensor.
  std::optional<TensorProjectionRecord> tensor_projection;
};

/// @brief Describe one reusable-Graph interface alias.
struct InterfaceBindingRecord {
  GraphPortId public_port{};    ///< Graph-interface declaration exposed to the parent Graph.
  GraphPortId internal_port{};  ///< Internal Operator declaration represented by the alias.
  PortDirection direction{PortDirection::kInput};  ///< Shared direction of both declarations.
};

/// @brief Describe one intrinsic external topic without transport/runtime state.
struct ExternalTopicRecord {
  ExternalTopicId id{};      ///< Nonzero snapshot-local boundary identity.
  VertexId vertex{};         ///< Intrinsic logical boundary vertex.
  GraphPortId graph_port{};  ///< Stable Graph-owned boundary port identity.
  bool imported{};           ///< True for external publisher to Graph; false for Graph export.
  std::string uri;           ///< Owned backend-neutral external topic URI.
  std::optional<std::string> endpoint_name;  ///< Owned explicit endpoint name when authored.
  bool generated_name_requested{};  ///< True when later lowering must generate an endpoint name.
  std::uint32_t declaration_ordinal{};     ///< Mixed import/export declaration attempt ordinal.
  PayloadIdentityRecord payload_identity;  ///< Complete pointer-free boundary identity.
  std::uint64_t serialized_payload_capacity_bytes{};  ///< Positive schema serialized-byte bound.
  std::optional<SchemaIdentityRecord> schema;  ///< Effective owned topic schema when declared.
  std::optional<MemoryConversionPolicy> memory_conversion;  ///< Tensor-only conversion request.
  std::optional<RequireCaptureTime> capture_time;  ///< Required capture-time domain when authored.
  ClockHealthOptions clock_health{};  ///< Normalized ordinary or monitored health contract.
  bool clock_health_authored{};       ///< True when health options were explicitly supplied.
  TopicAttachPolicy attach{};         ///< External peer presence and finite wait request.
  std::optional<TensorBoundaryContractRecord> tensor;  ///< Deep-owned Tensor contract.
  /// Complete descriptor-schema field inventory in canonical ascending field-id order.
  std::vector<DescriptorTensorTopicFieldRecord> tensor_fields;
  EndpointBounds bounds{};  ///< Direction-specific external peer resource bounds.
};

/// @brief Classify the wake requirement implied by one readiness source declaration.
enum class ReadinessWakeRequirement : std::uint8_t {
  kDeploymentSelected = 0,  ///< Compiler/runtime policy may select an admitted wake mechanism.
  kBusyPollRequired = 1,    ///< The source explicitly requires busy polling.
};

/// @brief Describe one logical readiness declaration after Graph composition.
///
/// This record is pointer-free authoring provenance. Graph composition assigns no physical
/// descriptor, CUDA event, HoloMQ handle, callback adapter, or plan-local
/// \c SourceId; compilation and runtime binding supply those resources.
struct ReadinessSourceDeclarationRecord {
  VertexId owner{};  ///< Logical Operator vertex that declared the readiness source.
  std::uint32_t declaration_ordinal{};  ///< Stable source order within the owning Operator.
  ReadinessSourceKind kind{ReadinessSourceKind::kReadable};  ///< Closed declaration kind.
  std::string name;                    ///< Owned unique declaration name within the Operator.
  std::uint32_t requested_capacity{};  ///< Requested bounded capacity, or zero when not applicable.
  std::uint32_t requested_sender_reference_capacity{};  ///< Requested sender-reference capacity.
  ClockKey requested_clock;  ///< Owned named/default clock key for time-based readiness.
  ReadinessWakeRequirement wake_requirement{
      ReadinessWakeRequirement::kDeploymentSelected};  ///< Source-derived wake requirement.
  /// Always false in Graph authoring snapshots. Inspect \c ReadinessPlanRecord::referenced
  /// through \c ExecutionPlan::readiness() for compiled contract usage.
  bool referenced{};
};

/// @brief Classify a backend-neutral region constraint.
enum class RegionConstraintKind : std::uint8_t {
  kStaticTopology,             ///< Require topology that does not change during execution.
  kBoundedMemory,              ///< Require finite admitted memory use.
  kDeviceCompatibleOperators,  ///< Require operators compatible with selected device lowering.
  kLatencyBudget,              ///< Carry a positive end-to-end region latency budget.
  kCudaGraph,                  ///< Require CUDA Graph capture and replay lowering.
};

/// @brief Carry one backend-neutral region constraint.
class HOLOSCAN_CORE_EXPORT RegionConstraint {
 public:
  /// @brief Create a static-topology constraint.
  /// @return Constraint with no associated duration.
  [[nodiscard]] static RegionConstraint static_topology() noexcept;

  /// @brief Create a bounded-memory constraint.
  /// @return Constraint with no associated duration.
  [[nodiscard]] static RegionConstraint bounded_memory() noexcept;

  /// @brief Create a device-compatible-operators constraint.
  /// @return Constraint with no associated duration.
  [[nodiscard]] static RegionConstraint device_compatible_ops() noexcept;

  /// @brief Create a CUDA Graph capture-and-replay constraint.
  /// @return Constraint with no associated duration.
  /// @note This is a hard lowering request, not a best-effort optimization hint. Compilation fails
  /// if the requested region cannot be lowered. Every member Operator must opt in through
  /// `OperatorSpec::cuda_graph_capture_eligible()`.
  /// @note The CUDA Graph constraint must be the region's only `RegionConstraint`. Compilation
  /// rejects a region that combines it with static-topology, bounded-memory,
  /// device-compatible-operator, or latency-budget constraints. A region profile is separate from
  /// this closed constraint set and remains permitted.
  /// @note The current compiler accepts only regions declared in the root Graph with directly
  /// listed operators owned by that root Graph. Embedded Graph members, nested operators, and CUDA
  /// Graph regions declared inside embedded Graphs are rejected. Nested Graph support might be
  /// added in a future release.
  /// @note Every Tensor output feeding a captured edge must use Runtime-owned storage and declare a
  /// complete maximum shape. Within the CUDA Graph region that shape is exact for capture and every
  /// activation, while ordinary Tensor edges continue to interpret it as an upper bound.
  /// @note An outside ingress producer or egress consumer must be assigned to the same process
  /// group as the CUDA Graph region.
  /// Region boundary and internal Tensor edges must resolve to one RuntimePool slot so the
  /// captured device address remains fixed. Queue depths and retention requirements needing more
  /// than one live slot are rejected during compilation. The ingress producer and downstream
  /// consumer apply backpressure until the slot is released, and a pending source trigger is
  /// retained while the region's output slot is unavailable.
  /// @see OperatorSpec::cuda_graph_capture_eligible()
  [[nodiscard]] static RegionConstraint cuda_graph() noexcept;

  /// @brief Create a positive region latency-budget constraint.
  /// @param budget Positive latency budget.
  /// @return Latency-budget constraint.
  /// @throws RuntimeError if \p budget is not positive.
  [[nodiscard]] static RegionConstraint latency_budget(std::chrono::nanoseconds budget);

  /// @brief Return the closed constraint kind.
  /// @return Constraint classification.
  [[nodiscard]] RegionConstraintKind kind() const noexcept { return kind_; }

  /// @brief Return the authored latency budget.
  /// @return Positive duration for `kLatencyBudget`; zero for every other kind.
  [[nodiscard]] std::chrono::nanoseconds latency() const noexcept { return latency_; }

 private:
  RegionConstraint(RegionConstraintKind kind, std::chrono::nanoseconds latency) noexcept
      : kind_(kind), latency_(latency) {}
  RegionConstraintKind kind_;
  std::chrono::nanoseconds latency_{};
};

/// @brief Describe one canonical Graph region declaration.
struct RegionRecord {
  RegionId id{};                              ///< Nonzero snapshot-local region identity.
  GraphId graph{};                            ///< Exact owning Graph instance.
  std::string name;                           ///< Owned unique region name.
  std::vector<VertexId> members;              ///< Canonical logical operator membership.
  std::vector<GraphId> graph_members;         ///< Canonical embedded Graph membership.
  std::vector<RegionConstraint> constraints;  ///< Unique backend-neutral constraints.
  std::optional<Profile> profile;             ///< Optional region-level profile request.
};

/// @brief Describe one developer-preview manual execution partition.
///
/// Partition declarations are distinct from overlapping \ref RegionRecord
/// scopes. Each record owns its name, direct Operator and atomic child-instance membership, or
/// the mutually exclusive compile-root \ref PartitionHandle::include_all
/// directive. Without an explicit automatic-remainder request, the compiler
/// requires ordinary authored membership to form a nonempty, disjoint, exact
/// cover before lowering it into execution-plan partitions.
struct PartitionRecord {
  PartitionId id{};               ///< Nonzero snapshot-local partition identity.
  GraphId graph{};                ///< Exact owning Graph instance.
  std::string name;               ///< Owned unique partition name in the Graph scope.
  std::vector<VertexId> members;  ///< Canonical direct-Operator membership.
  bool include_all{};             ///< Expand to the complete compile-root application at compile.
  std::vector<GraphId>
      graph_members;  ///< Explicit atomic child roots; includes descendant Operators.
};

/// @brief Describe one owned/replayable clock declaration.
struct ClockDeclarationRecord {
  GraphId graph{};    ///< Exact owning Graph instance.
  ClockKey key;       ///< Owned named clock key.
  bool shared{};      ///< True only for explicit `bind_shared_clock` authoring.
  bool is_default{};  ///< True only when this clock is the current Graph default.
};

/// @brief Describe one explicit clock dependency declared by a reusable Graph.
///
/// An import is a scoped authoring key, not a runtime clock declaration. A
/// direct parent must bind it to one exact parent clock before compilation.
struct ClockImportAuthoringRecord {
  GraphId graph{};    ///< Reusable Graph instance declaring the dependency.
  ClockKey key;       ///< Owned named import key in that Graph's clock namespace.
  bool is_default{};  ///< True only when this import is the Graph default.
};

/// @brief Describe one exact parent-to-child clock import binding.
///
/// Bindings follow Graph instance identity rather than clock-name lookup. The
/// parent target may itself be an imported clock, allowing explicit forwarding
/// through nested reusable Graph boundaries.
struct EmbeddedClockBindingAuthoringRecord {
  GraphId child_graph{};   ///< Immediate embedded child declaring the import.
  ClockKey child_clock;    ///< Exact imported key in the child scope.
  GraphId parent_graph{};  ///< Immediate parent Graph providing the target.
  ClockKey parent_clock;   ///< Exact named clock or import in the parent scope.
};

/// @brief Describe one immutable directed clock conversion declaration.
///
/// The stored positive ratio is reduced to canonical lowest terms. Reauthoring
/// the same directed pair with an equivalent ratio and offset is idempotent;
/// conflicting pairs and directed conversion cycles are rejected.
struct ClockConversionAuthoringRecord {
  GraphId graph{};                   ///< Exact owning Graph instance.
  ClockKey from;                     ///< Declared source clock.
  ClockKey to;                       ///< Distinct declared destination clock.
  AffineClockConversion conversion;  ///< Positive-scale immutable conversion.
};

/// @brief Describe one same-stage lifecycle ordering edge.
struct LifecycleDependencyAuthoringRecord {
  GraphId graph{};    ///< Graph instance whose lifecycle builder declared the edge.
  VertexId before{};  ///< Logical checkpoint that must complete first.
  VertexId after{};   ///< Logical checkpoint released after `before`.
  LifecycleStage stage{LifecycleStage::kConfigure};  ///< Exact same-stage ordering scope.
};

/// @brief Classify one independently authorized lifecycle restart origin.
enum class LifecycleRestartOrigin : std::uint8_t {
  kSelf,             ///< Operator may request its own restart.
  kExternal,         ///< Authorized external run control may request restart.
  kOperatorRequest,  ///< One exact requester Operator may target another.
};

/// @brief Describe one deny-by-default lifecycle restart permission.
struct LifecycleRestartPermissionAuthoringRecord {
  LifecycleRestartOrigin origin{LifecycleRestartOrigin::kSelf};  ///< Authorized request origin.
  std::optional<VertexId> requester;  ///< Exact requester for operator-origin permission.
  VertexId target{};                  ///< Restart target; grants no state access.
};

/// @brief Inspect one immutable deep-owning Graph authoring snapshot.
///
/// Copies share immutable snapshot storage and remain valid after later source
/// Graph mutation or destruction. Concurrent reads of one frozen view are safe;
/// the view exposes no mutation API. Record spans preserve authored order and
/// snapshot-local identities so they continue to correspond to authoring
/// handles. The identity hash is computed from canonical semantic content;
/// compiler-local normalized identities and ordering are exposed only through
/// the non-installed compiler handoff.
class HOLOSCAN_CORE_EXPORT GraphView {
 public:
  /// @brief Create an empty immutable view.
  GraphView() = default;

  /// @brief Return the source Graph authoring generation.
  /// @return Zero for an empty view; otherwise the frozen generation.
  [[nodiscard]] GraphGeneration generation() const noexcept;
  /// @brief Return the stable source/version/content identity of this snapshot.
  /// @return Zero-initialized identity for an empty view; otherwise the exact
  /// process-local Graph instance, frozen authoring version, and deterministic
  /// canonical content hash.
  [[nodiscard]] GraphAuthoringIdentity identity() const noexcept;
  /// @brief Return frozen Graph records.
  /// @return Borrowed immutable records, or an empty span for an empty view.
  /// @note The span remains valid only while a \ref GraphView sharing this snapshot remains alive.
  [[nodiscard]] std::span<const GraphRecord> graphs() const noexcept;
  /// @brief Return frozen logical vertex records.
  /// @return Borrowed immutable records, or an empty span for an empty view.
  /// @note The span remains valid only while a \ref GraphView sharing this snapshot remains alive.
  [[nodiscard]] std::span<const VertexRecord> vertices() const noexcept;
  /// @brief Return frozen pointer-free port records.
  /// @return Borrowed immutable records, or an empty span for an empty view.
  /// @note The span remains valid only while a \ref GraphView sharing this snapshot remains alive.
  [[nodiscard]] std::span<const PortRecord> ports() const noexcept;
  /// @brief Return frozen typed queue-edge records.
  /// @return Borrowed immutable records, or an empty span for an empty view.
  /// @note The span remains valid only while a \ref GraphView sharing this snapshot remains alive.
  [[nodiscard]] std::span<const EdgeRecord> edges() const noexcept;
  /// @brief Return frozen same-direction interface aliases.
  /// @return Borrowed immutable records, or an empty span for an empty view.
  /// @note The span remains valid only while a \ref GraphView sharing this snapshot remains alive.
  [[nodiscard]] std::span<const InterfaceBindingRecord> interface_bindings() const noexcept;
  /// @brief Return frozen intrinsic external topic records.
  /// @return Borrowed immutable records, or an empty span for an empty view.
  /// @note The span remains valid only while a \ref GraphView sharing this snapshot remains alive.
  [[nodiscard]] std::span<const ExternalTopicRecord> external_topics() const noexcept;
  /// @brief Return frozen readiness-source declarations.
  /// @return Borrowed immutable records, or an empty span for an empty view.
  /// @note The span remains valid only while a \ref GraphView sharing this snapshot remains alive.
  [[nodiscard]] std::span<const ReadinessSourceDeclarationRecord> readiness_sources()
      const noexcept;
  /// @brief Return frozen region declarations.
  /// @return Borrowed immutable records, or an empty span for an empty view.
  /// @note The span remains valid only while a \ref GraphView sharing this snapshot remains alive.
  [[nodiscard]] std::span<const RegionRecord> regions() const noexcept;
  /// @brief Return frozen developer-preview manual partition declarations.
  /// @return Borrowed immutable records, or an empty span for an empty view.
  /// @note The span remains valid only while a \ref GraphView sharing this snapshot remains alive.
  [[nodiscard]] std::span<const PartitionRecord> partitions() const noexcept;
  /// @brief Return whether automatic partitioning was requested for uncovered executable members.
  /// @return `true` when the frozen Graph authored \ref Graph::auto_partition_remaining.
  [[nodiscard]] bool automatic_partition_remaining() const noexcept;
  /// @brief Return frozen owned/shared clock declarations.
  /// @return Borrowed immutable records, or an empty span for an empty view.
  /// @note The span remains valid only while a \ref GraphView sharing this snapshot remains alive.
  [[nodiscard]] std::span<const ClockDeclarationRecord> clocks() const noexcept;
  /// @brief Return frozen reusable-Graph clock imports.
  /// @return Borrowed immutable records, or an empty span for an empty view.
  /// @note The span remains valid only while a \ref GraphView sharing this snapshot remains alive.
  [[nodiscard]] std::span<const ClockImportAuthoringRecord> clock_imports() const noexcept;
  /// @brief Return frozen exact embedded clock bindings.
  /// @return Borrowed immutable records, or an empty span for an empty view.
  /// @note The span remains valid only while a \ref GraphView sharing this snapshot remains alive.
  [[nodiscard]] std::span<const EmbeddedClockBindingAuthoringRecord> clock_bindings()
      const noexcept;
  /// @brief Return frozen directed affine clock conversions.
  /// @return Borrowed immutable records, or an empty span for an empty view.
  /// @note The span remains valid only while a \ref GraphView sharing this snapshot remains alive.
  [[nodiscard]] std::span<const ClockConversionAuthoringRecord> clock_conversions() const noexcept;
  /// @brief Return frozen same-stage lifecycle dependencies.
  /// @return Borrowed immutable records, or an empty span for an empty view.
  /// @note The span remains valid only while a \ref GraphView sharing this snapshot remains alive.
  [[nodiscard]] std::span<const LifecycleDependencyAuthoringRecord> lifecycle_dependencies()
      const noexcept;
  /// @brief Return frozen deny-by-default restart permissions.
  /// @return Borrowed immutable records, or an empty span for an empty view.
  /// @note The span remains valid only while a \ref GraphView sharing this snapshot remains alive.
  [[nodiscard]] std::span<const LifecycleRestartPermissionAuthoringRecord>
  lifecycle_restart_permissions() const noexcept;

 private:
  explicit GraphView(std::shared_ptr<const detail::GraphSnapshotData> snapshot) noexcept
      : snapshot_(std::move(snapshot)) {}
  std::shared_ptr<const detail::GraphSnapshotData> snapshot_;
  friend class Graph;
  friend class detail::GraphCompilerAccess;
};

namespace detail {

/// @brief Carry one immutable root-Graph Operator-handle lookup identity.
///
/// This private bridge deliberately excludes embedded Graph handles. The execution plan combines
/// these source-Graph facts with its own immutable root-index table without mutating authoring
/// storage or publishing a plan-bound \ref OperatorRef back into the Graph.
struct GraphOperatorLocator {
  GraphInstanceId graph{};
  GraphGeneration generation{};
  std::size_t root_operator_index{};
};

/// @cond INTERNAL_GRAPH_BRIDGES
HOLOSCAN_CORE_EXPORT expected<GraphOperatorLocator, Error> graph_operator_locator(
    const std::shared_ptr<GraphState>& state, std::size_t index,
    const void* handle_identity) noexcept;
/// @endcond

}  // namespace detail

}  // namespace holoscan
