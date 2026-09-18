/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/channel/pod_type.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/external_topic.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_input.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/payload_identity.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/profile.hpp>
#include <holoscan/core/rate.hpp>
#include <holoscan/core/readiness_source.hpp>
#include <holoscan/core/tensor_field.hpp>
#include <holoscan/time/clock.hpp>
#include <holoscan/time/clock_registry.hpp>

namespace holoscan {

namespace authoring {
class HOLOSCAN_CORE_EXPORT OpaqueOperatorRef;
}  // namespace authoring

class Graph;
class HOLOSCAN_CORE_EXPORT TensorOutputPortConfigurator;
class GraphInstancePath;
class ExecutionPartitionsView;

namespace detail {
class GraphState;
class GraphCompilerAccess;
class GraphOperatorRegistrationAccess;
class GraphPayloadRegistrationAccess;
class ParameterConfigBindingAccess;
class ParameterInputAccess;
struct GraphSnapshotData;
struct ParameterInputEvidenceState;
}  // namespace detail

template <typename G>
class GraphHandle;

/// @brief Identify one immutable Graph authoring snapshot generation.
using GraphGeneration = std::uint64_t;

/// @brief Identify one live Graph authoring instance without retaining an address.
///
/// The value is unique within the current process lifetime and remains stable
/// across snapshots of that Graph. It is not a pointer, persistent deployment
/// identity, or cross-process interchange key.
struct GraphInstanceId {
  std::array<std::byte, 16> value{};  ///< Complete process-local instance identity bytes.
  friend constexpr bool operator==(GraphInstanceId, GraphInstanceId) noexcept = default;
};

/// @brief Hold one deterministic 256-bit Graph authoring fingerprint.
///
/// Equal canonical pointer-free authoring records produce equal values
/// independent of Graph object address. The value is an authoring-content
/// fingerprint, not a cryptographic authenticity claim.
struct GraphAuthoringHash {
  std::array<std::byte, 32> value{};  ///< Complete canonical authoring digest bytes.
  friend constexpr bool operator==(GraphAuthoringHash, GraphAuthoringHash) noexcept = default;
};

/// @brief Identify the exact immutable Graph authoring state behind one snapshot.
///
/// The tuple distinguishes the live source Graph, its monotonically increasing
/// authoring version, and its canonical content. Separate Graph instances may
/// have equal \ref hash values while retaining different \ref graph identities.
struct GraphAuthoringIdentity {
  GraphInstanceId graph;      ///< Stable identity of the source Graph object.
  GraphGeneration version{};  ///< Source authoring generation frozen by the snapshot.
  GraphAuthoringHash hash;    ///< Deterministic hash of canonical pointer-free records.
  friend constexpr bool operator==(GraphAuthoringIdentity,
                                   GraphAuthoringIdentity) noexcept = default;
};

/// @brief Own the decoded canonical path of one embedded Graph instance.
///
/// Every value contains at least one embedded-instance segment. Root scope is
/// represented separately by an absent path in \ref GraphRecord; `_root` is an
/// endpoint-name rendering token and never a path segment. Identity comparisons
/// operate on decoded segments rather than a rendered slash-delimited string,
/// so local names may repeat in distinct parent scopes without aliasing.
class HOLOSCAN_CORE_EXPORT GraphInstancePath {
 public:
  GraphInstancePath() = delete;

  /// @brief Return the decoded path segments in root-to-leaf order.
  /// @return Borrowed immutable segments valid for this object's lifetime.
  [[nodiscard]] std::span<const std::string> segments() const noexcept { return segments_; }

  /// @brief Return the number of embedded-instance segments.
  /// @return Segment count.
  [[nodiscard]] std::size_t size() const noexcept { return segments_.size(); }

  friend bool operator==(const GraphInstancePath&, const GraphInstancePath&) noexcept = default;

 private:
  explicit GraphInstancePath(std::vector<std::string> segments) : segments_(std::move(segments)) {}
  std::vector<std::string> segments_;
  friend class detail::GraphState;
  friend class detail::GraphCompilerAccess;
  friend struct detail::GraphSnapshotData;
};

/// @brief Identify one Graph record within a GraphView snapshot.
struct GraphId {
  std::uint64_t value{};  ///< Nonzero snapshot-local identity.
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0; }
  friend constexpr bool operator==(GraphId, GraphId) noexcept = default;
};

/// @brief Identify one vertex record within a GraphView snapshot.
struct VertexId {
  std::uint64_t value{};  ///< Nonzero snapshot-local identity.
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0; }
  friend constexpr bool operator==(VertexId, VertexId) noexcept = default;
};

/// @brief Identify one edge record within a GraphView snapshot.
struct EdgeId {
  std::uint64_t value{};  ///< Nonzero snapshot-local identity.
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0; }
  friend constexpr bool operator==(EdgeId, EdgeId) noexcept = default;
};

/// @brief Identify one region record within a GraphView snapshot.
struct RegionId {
  std::uint64_t value{};  ///< Nonzero snapshot-local identity.
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0; }
  friend constexpr bool operator==(RegionId, RegionId) noexcept = default;
};

/// @brief Identify one manual execution-partition record within a GraphView snapshot.
///
/// @note This developer-preview identity is local to one authoring snapshot.
/// Compiler lowering assigns the corresponding immutable execution-plan
/// partition identity.
struct PartitionId {
  std::uint64_t value{};  ///< Nonzero snapshot-local identity.
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0; }
  friend constexpr bool operator==(PartitionId, PartitionId) noexcept = default;
};

/// @brief Identify one external topic record within a GraphView snapshot.
struct ExternalTopicId {
  std::uint64_t value{};  ///< Nonzero snapshot-local identity.
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0; }
  friend constexpr bool operator==(ExternalTopicId, ExternalTopicId) noexcept = default;
};

/// @brief Identify one port record within a GraphView authoring snapshot.
///
/// This value is never a compiler-assigned \c PortId. HSDK5-009 remaps it to a
/// plan-local identity when compilation freezes an authoring snapshot.
struct GraphPortId {
  std::uint64_t value{};  ///< Nonzero snapshot-local authoring identity.
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0; }
  friend constexpr bool operator==(GraphPortId, GraphPortId) noexcept = default;
};

/// @brief Classify a logical Graph vertex before compiler-inserted lowering.
enum class VertexKind : std::uint8_t {
  kOperator = 0,        ///< User-authored Operator declaration.
  kGraphInterface = 1,  ///< Public reusable-Graph interface declaration.
  kImportedTopic = 2,   ///< Intrinsic external-to-Graph boundary.
  kExportedTopic = 3,   ///< Intrinsic Graph-to-external boundary.
};

/// @brief Classify which authoring object owns one logical port.
enum class PortOwnerKind : std::uint8_t {
  kOperator = 0,        ///< Operator member declared through `OperatorSpec`.
  kGraphInterface = 1,  ///< Reusable-Graph interface member.
  kImportedTopic = 2,   ///< Imported-topic Graph-side output.
  kExportedTopic = 3,   ///< Exported-topic Graph-side input.
};

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
/// Setup defaults and Graph overrides occupy separate instances so HSDK5-009 can
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
/// This record is pointer-free authoring provenance. HSDK5-006 assigns no physical
/// descriptor, CUDA event, HoloMQ handle, callback adapter, or plan-local
/// `SourceId`; compiler/runtime binding belongs to later components.
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
  bool referenced{};  ///< Reserved for later Contract lowering; false in HSDK5-006 snapshots.
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

namespace detail {

class GraphState;
class GraphCompilerAccess;
struct GraphSnapshotData;
struct DiagnosticCapture;

/// @brief Log cold user-factory exception detail without placing logger macros in public headers.
/// @param context Stable bounded construction context token.
/// @param detail Arbitrary exception text used only for cold diagnostics.
HOLOSCAN_CORE_EXPORT void graph_log_factory_exception(std::string_view context,
                                                      std::string_view detail) noexcept;

/// @brief Capture the currently handled factory exception with bounded explanatory context.
/// @param token Stable construction or declaration failure token, or empty for cause text only.
/// @return Original category and contextual message, or a code-only allocation-safe fallback.
HOLOSCAN_CORE_EXPORT Error graph_factory_error(std::string_view token) noexcept;

/// @brief Capture into borrowed fixed storage, or render the ordinary authoring error.
/// @param token Stable authoring failure token when no capture is supplied.
/// @param capture Storage borrowed for this call only; nonnull capture returns a code-only Error.
/// @return Original error category with independently retained evidence.
HOLOSCAN_CORE_EXPORT Error graph_factory_error(std::string_view token,
                                               DiagnosticCapture* capture) noexcept;

class GraphPortOwnershipAccess {
 public:
  [[nodiscard]] static const void* owner(const InputBase& port) noexcept {
    return port.graph_owner_token_;
  }
  [[nodiscard]] static const void* owner(const OutputBase& port) noexcept {
    return port.graph_owner_token_;
  }
  static void claim(const InputBase& port, const void* owner) noexcept {
    port.graph_owner_token_ = owner;
  }
  static void claim(const OutputBase& port, const void* owner) noexcept {
    port.graph_owner_token_ = owner;
  }
};

template <typename T>
inline constexpr const std::byte& kGraphOperatorTypeToken = kLifecycleDeclaringTypeToken<T>;

template <typename T>
inline constexpr std::byte kGraphEmbeddingTypeToken{};

class GraphOperatorAccess {
 public:
  template <typename Op>
  [[nodiscard]] static expected<void, Error> bind(Op& op, typename Op::Params params) {
    const ParamValidationResult validation = validate_params(params);
    if (!validation.ok())
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, parameter_validation_error_message(validation)});
    if constexpr (std::same_as<typename Op::Params, NoParams>) {
      return {};
    } else if constexpr (std::is_nothrow_swappable_v<typename Op::Params>) {
      op.bind_initial_params(params);
      return {};
    } else {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
  }

 private:
  static constexpr std::size_t kMaximumRenderedIdentityBytes = 96U;

  [[nodiscard]] static constexpr std::string_view validation_code_name(
      ParamValidationCode code) noexcept {
    switch (code) {
      case ParamValidationCode::kOk:
        return "ok";
      case ParamValidationCode::kOutOfRange:
        return "out_of_range";
      case ParamValidationCode::kNotAllowed:
        return "not_allowed";
      case ParamValidationCode::kCardinalityExceeded:
        return "cardinality_exceeded";
      case ParamValidationCode::kInvalidEncoding:
        return "invalid_encoding";
      case ParamValidationCode::kValidatorRejected:
        return "validator_rejected";
    }
    return "unknown";
  }

  static void append_bounded_identity(std::string& message, std::string_view identity) {
    if (identity.size() <= kMaximumRenderedIdentityBytes) {
      message.append(identity);
      return;
    }

    detail::StableHashBuilder hash;
    hash.add("holoscan.param.validation-diagnostic.identity/v1");
    hash.add(identity);
    const ParamSchemaHash digest = hash.finish();
    constexpr std::string_view kHexDigits{"0123456789abcdef"};

    std::size_t prefix_bytes = kMaximumRenderedIdentityBytes;
    while (prefix_bytes != 0U && !detail::valid_utf8(identity.substr(0U, prefix_bytes))) {
      --prefix_bytes;
    }
    message.append(identity.substr(0U, prefix_bytes));
    message.append("...;bytes=");
    message.append(std::to_string(identity.size()));
    message.append(";sha256=");
    for (const std::byte byte : digest.value) {
      const auto value = static_cast<std::uint8_t>(byte);
      message.push_back(kHexDigits[value >> 4U]);
      message.push_back(kHexDigits[value & 0x0FU]);
    }
  }

  [[nodiscard]] static std::string parameter_validation_error_message(
      const ParamValidationResult& validation) {
    std::string message{"PARAM_VALIDATION_FAILED category="};
    message.append(validation_code_name(validation.code));
    message.append(" field_id=");
    message.append(std::to_string(validation.field_id));
    if (!validation.field_name.empty()) {
      message.append(" field_name=");
      append_bounded_identity(message, validation.field_name);
    }
    if (!validation.validator_id.empty()) {
      message.append(" validator_id=");
      append_bounded_identity(message, validation.validator_id);
    }
    return message;
  }
};

class OperatorFactoryBase {
 public:
  virtual ~OperatorFactoryBase() = default;
  [[nodiscard]] virtual expected<std::unique_ptr<OperatorBase>, Error> create(
      DiagnosticCapture* capture) const = 0;
  [[nodiscard]] virtual std::shared_ptr<OperatorFactoryBase> clone() const = 0;
  [[nodiscard]] virtual expected<std::shared_ptr<OperatorFactoryBase>, Error> clone_with_params(
      const void* params, const void* type_token) const = 0;

  /// @brief Return process-local whole-aggregate Parameter operations.
  /// @return Static descriptor for the concrete Operator's exact parameter type.
  /// @note The compiler copies only pointer-free descriptor fields into a plan.
  [[nodiscard]] virtual const ParamTypeDescriptor& parameter_type() const noexcept = 0;

  /// @brief Return the concrete Operator's declared Parameter schema.
  /// @return Static local descriptor, or \c nullptr for opaque-local mode.
  /// @note Local operation pointers remain private executable-factory metadata.
  [[nodiscard]] virtual const ParamSchemaDescriptor* parameter_schema() const noexcept = 0;

  /// @brief Return the immutable generation-zero aggregate used by this factory.
  /// @return Process-local typed value matching \ref parameter_type, never \c nullptr.
  /// @note Compilation may read declared fields through schema-owned local thunks. The address and
  /// object representation never enter an ExecutionPlan or portable artifact.
  [[nodiscard]] virtual const void* initial_parameter_value() const noexcept = 0;

  /// @brief Return whether authoring supplied a whole typed generation-zero baseline.
  /// @return \c true only when `OperatorHandle::set_params()` replaced `ParamsT{}`.
  /// @note Compilation uses this fact to preserve default-versus-baseline provenance without
  /// comparing values or inferring intent from aggregate equality.
  [[nodiscard]] virtual bool has_explicit_parameter_baseline() const noexcept = 0;

  /// @brief Return whether this factory's Operator uses \ref NoParams.
  /// @return \c true only for the static no-storage generation-zero contract.
  [[nodiscard]] virtual bool has_no_params() const noexcept = 0;
  struct LifecycleHookSet {
    bool start{};
    bool stop{};
  };
  [[nodiscard]] virtual LifecycleHookSet lifecycle_hooks() const noexcept = 0;
};

template <typename Owner>
auto lifecycle_start_owner(void (Owner::*)()) -> std::type_identity<Owner>;

template <typename Owner>
auto lifecycle_start_owner(void (Owner::*)() noexcept) -> std::type_identity<Owner>;

template <typename Owner>
auto lifecycle_stop_owner(void (Owner::*)()) -> std::type_identity<Owner>;

template <typename Owner>
auto lifecycle_stop_owner(void (Owner::*)() noexcept) -> std::type_identity<Owner>;

template <typename Op>
concept SelectableOperatorLifecycleHooks = requires {
  typename decltype(lifecycle_start_owner(&Op::start))::type;
  typename decltype(lifecycle_stop_owner(&Op::stop))::type;
};

template <typename Op>
  requires SelectableOperatorLifecycleHooks<Op>
[[nodiscard]] constexpr OperatorFactoryBase::LifecycleHookSet operator_lifecycle_hooks() noexcept {
  using StartOwner = typename decltype(lifecycle_start_owner(&Op::start))::type;
  using StopOwner = typename decltype(lifecycle_stop_owner(&Op::stop))::type;
  return {
      .start = !std::same_as<StartOwner, OperatorBase>,
      .stop = !std::same_as<StopOwner, OperatorBase>,
  };
}

template <typename T>
struct IsSpan : std::false_type {};
template <typename T, std::size_t Extent>
struct IsSpan<std::span<T, Extent>> : std::true_type {};
template <typename T>
struct IsReferenceWrapper : std::false_type {};
template <typename T>
struct IsReferenceWrapper<std::reference_wrapper<T>> : std::true_type {};
template <typename T>
struct IsInitializerList : std::false_type {};
template <typename T>
struct IsInitializerList<std::initializer_list<T>> : std::true_type {};
template <typename T>
inline constexpr bool kStringCapture =
    std::same_as<std::remove_cvref_t<T>, std::string> ||
    std::same_as<std::remove_cvref_t<T>, std::string_view> ||
    (std::is_array_v<std::remove_reference_t<T>> &&
     std::same_as<std::remove_extent_t<std::remove_reference_t<T>>, const char>);

template <std::size_t N>
// A bounded C-array reference is the only portable way to distinguish a string literal from a
// borrowed character pointer before converting it into Graph-owned storage.
[[nodiscard]] std::string normalize_graph_capture(
    const char (&value)[N]) {  // NOLINT(cppcoreguidelines-avoid-c-arrays)
  return std::string(value, N == 0 ? 0 : N - 1);
}

[[nodiscard]] inline std::string normalize_graph_capture(std::string_view value) {
  return std::string(value);
}

template <typename T>
  requires(!std::is_array_v<std::remove_reference_t<T>> &&
           !std::same_as<std::remove_cvref_t<T>, std::string_view>)
[[nodiscard]] std::remove_cvref_t<T> normalize_graph_capture(T&& value) {
  return std::forward<T>(value);
}

template <typename T>
using NormalizedGraphCapture = decltype(normalize_graph_capture(std::declval<T>()));

template <typename Op, typename... Captures>
class OperatorFactoryModel final : public OperatorFactoryBase {
 public:
  explicit OperatorFactoryModel(Captures... captures) : captures_(std::move(captures)...) {}

  [[nodiscard]] expected<std::unique_ptr<OperatorBase>, Error> create(
      DiagnosticCapture* capture) const override {
    try {
      auto op = std::apply([](const auto&... values) { return std::make_unique<Op>(values...); },
                           captures_);
      if (params_) {
        auto params = *params_;
        auto bound = GraphOperatorAccess::bind(*op, std::move(params));
        if (!bound)
          return make_unexpected(std::move(bound.error()));
      }
      return std::unique_ptr<OperatorBase>(std::move(op));
    } catch (const std::exception& exception) {
      auto error = graph_factory_error("GRAPH_OPERATOR_CONSTRUCTION_FAILED", capture);
      graph_log_factory_exception("operator", exception.what());
      return make_unexpected(std::move(error));
    } catch (...) {
      auto error = graph_factory_error("GRAPH_OPERATOR_CONSTRUCTION_FAILED", capture);
      graph_log_factory_exception("operator", "non-standard exception");
      return make_unexpected(std::move(error));
    }
  }

  [[nodiscard]] std::shared_ptr<OperatorFactoryBase> clone() const override {
    return std::make_shared<OperatorFactoryModel>(*this);
  }

  [[nodiscard]] expected<std::shared_ptr<OperatorFactoryBase>, Error> clone_with_params(
      const void* params, const void* type_token) const override {
    if (params == nullptr ||
        type_token != param_type_descriptor<typename Op::Params>().local_type_token) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument, "GRAPH_PARAMETER_TYPE_MISMATCH"});
    }
    auto copy = std::make_shared<OperatorFactoryModel>(*this);
    copy->params_.reset();
    copy->params_.emplace(*static_cast<const typename Op::Params*>(params));
    return std::shared_ptr<OperatorFactoryBase>(std::move(copy));
  }

  [[nodiscard]] const ParamTypeDescriptor& parameter_type() const noexcept override {
    return param_type_descriptor<typename Op::Params>();
  }

  [[nodiscard]] const ParamSchemaDescriptor* parameter_schema() const noexcept override {
    return param_schema_descriptor<typename Op::Params>();
  }

  [[nodiscard]] const void* initial_parameter_value() const noexcept override {
    if (params_) {
      return &*params_;
    }
    if constexpr (std::same_as<typename Op::Params, NoParams>) {
      return &kNoParams;
    } else {
      static const typename Op::Params default_value{};
      return &default_value;
    }
  }

  [[nodiscard]] bool has_explicit_parameter_baseline() const noexcept override {
    return params_.has_value();
  }

  [[nodiscard]] bool has_no_params() const noexcept override {
    return std::same_as<typename Op::Params, NoParams>;
  }

  [[nodiscard]] LifecycleHookSet lifecycle_hooks() const noexcept override {
    return operator_lifecycle_hooks<Op>();
  }

 private:
  std::tuple<Captures...> captures_;
  std::optional<typename Op::Params> params_;
};

class ClockFactoryBase {
 public:
  virtual ~ClockFactoryBase() = default;
  [[nodiscard]] virtual expected<std::unique_ptr<Clock>, Error> create(
      DiagnosticCapture* capture) const = 0;
  [[nodiscard]] virtual std::shared_ptr<ClockFactoryBase> clone() const = 0;
};

template <typename ClockT, typename... Captures>
class ClockFactoryModel final : public ClockFactoryBase {
 public:
  explicit ClockFactoryModel(Captures... captures) : captures_(std::move(captures)...) {}
  [[nodiscard]] expected<std::unique_ptr<Clock>, Error> create(
      DiagnosticCapture* capture) const override {
    try {
      auto clock = std::apply(
          [](const auto&... values) { return std::make_unique<ClockT>(values...); }, captures_);
      return std::unique_ptr<Clock>(std::move(clock));
    } catch (const std::exception& exception) {
      auto error = graph_factory_error("GRAPH_CLOCK_CONSTRUCTION_FAILED", capture);
      graph_log_factory_exception("clock", exception.what());
      return make_unexpected(std::move(error));
    } catch (...) {
      auto error = graph_factory_error("GRAPH_CLOCK_CONSTRUCTION_FAILED", capture);
      graph_log_factory_exception("clock", "non-standard exception");
      return make_unexpected(std::move(error));
    }
  }
  [[nodiscard]] std::shared_ptr<ClockFactoryBase> clone() const override {
    return std::make_shared<ClockFactoryModel>(*this);
  }

 private:
  std::tuple<Captures...> captures_;
};

template <typename T>
inline constexpr bool kStableExternalScalar =
    std::same_as<T, bool> || std::same_as<T, std::uint8_t> || std::same_as<T, std::int8_t> ||
    std::same_as<T, std::uint16_t> || std::same_as<T, std::int16_t> ||
    std::same_as<T, std::uint32_t> || std::same_as<T, std::int32_t> ||
    std::same_as<T, std::uint64_t> || std::same_as<T, std::int64_t> ||
    (std::same_as<T, float> && std::numeric_limits<float>::is_iec559 && sizeof(float) == 4) ||
    (std::same_as<T, double> && std::numeric_limits<double>::is_iec559 && sizeof(double) == 8);

template <typename T>
concept ExternalTopicPayload =
    PublishedPayload<T> && (kStableExternalScalar<T> || std::same_as<T, Tensor> ||
                            (HasSchemaIdentity<T> != HasChannelPodContract<T>));

struct OperatorRegistration {
  std::size_t index{};
};

struct InitialOperatorParameters {
  const void* value{};
  const void* local_type_token{};
  const ParamSchemaDescriptor* schema{};
};

struct FlowRegistration {
  EdgeId edge{};
  GraphPortId source{};
  GraphPortId target{};
};

struct EmbeddedGraphRegistration {
  std::size_t index{};
};

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

/// @brief Carry one defining-module parameter target for optional authoring tooling.
struct GraphOperatorParameterTarget {
  VertexId vertex{};
  const ParamSchemaDescriptor* schema{};
};

HOLOSCAN_CORE_EXPORT expected<OperatorRegistration, Error> graph_register_operator(
    const std::shared_ptr<GraphState>& state, std::string_view name,
    std::shared_ptr<OperatorFactoryBase> factory, std::unique_ptr<OperatorBase> prototype,
    const void* operator_type_token, const InitialOperatorParameters* initial_params = nullptr);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_prepare_authoring(
    const std::shared_ptr<GraphState>& state);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_prepare_embed(
    const std::shared_ptr<GraphState>& state, std::string_view instance_name);
HOLOSCAN_CORE_EXPORT const OperatorBase* graph_operator_prototype(
    const std::shared_ptr<GraphState>& state, std::size_t index,
    const void* operator_type_token) noexcept;
HOLOSCAN_CORE_EXPORT const void* graph_operator_port_declaration(
    const std::shared_ptr<GraphState>& state, std::size_t index, std::string_view name) noexcept;
HOLOSCAN_CORE_EXPORT expected<const void*, Error> graph_resolve_operator_port(
    const std::shared_ptr<GraphState>& state, std::size_t index, const void* operator_type_token,
    std::string_view name, PortDirection direction, const PayloadIdentityRecord& payload_identity);
HOLOSCAN_CORE_EXPORT bool graph_operator_valid(const std::shared_ptr<GraphState>& state,
                                               std::size_t index,
                                               const void* operator_type_token) noexcept;
HOLOSCAN_CORE_EXPORT std::string_view graph_operator_name(const std::shared_ptr<GraphState>& state,
                                                          std::size_t index) noexcept;
HOLOSCAN_CORE_EXPORT expected<std::size_t, Error> graph_find_operator(
    const std::shared_ptr<GraphState>& state, std::string_view name) noexcept;
HOLOSCAN_CORE_EXPORT OperatorRef graph_operator_ref(const std::shared_ptr<GraphState>& state,
                                                    std::size_t index) noexcept;
HOLOSCAN_CORE_EXPORT expected<GraphOperatorLocator, Error> graph_operator_locator(
    const std::shared_ptr<GraphState>& state, std::size_t index) noexcept;
HOLOSCAN_CORE_EXPORT expected<VertexId, Error> graph_operator_vertex(
    const std::shared_ptr<GraphState>& state, std::size_t index, const void* operator_type_token);
HOLOSCAN_CORE_EXPORT expected<GraphOperatorParameterTarget, Error> graph_operator_parameter_target(
    const std::shared_ptr<GraphState>& state, std::size_t index, const void* operator_type_token);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_set_operator_params(
    const std::shared_ptr<GraphState>& state, std::size_t index, const void* params,
    const void* operator_type_token, const void* caller_parameter_type_token,
    const ParamSchemaDescriptor* caller_schema,
    std::shared_ptr<const ParameterInputEvidenceState> evidence = nullptr);

HOLOSCAN_CORE_EXPORT expected<FlowRegistration, Error> graph_add_flow(
    const std::shared_ptr<GraphState>& state, const OutputBase* source, const InputBase* target,
    const PayloadIdentityRecord& source_identity, const PayloadIdentityRecord& target_identity,
    ConnectionOptions options);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_bind_input_interface(
    const std::shared_ptr<GraphState>& state, const InputBase* first, const InputBase* second,
    PayloadIdentityRecord payload_identity);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_bind_output_interface(
    const std::shared_ptr<GraphState>& state, const OutputBase* first, const OutputBase* second,
    PayloadIdentityRecord payload_identity);
HOLOSCAN_CORE_EXPORT expected<GraphPortId, Error> graph_find_port(
    const std::shared_ptr<GraphState>& state, const void* declaration, PortDirection direction,
    const PayloadIdentityRecord& payload_identity);
HOLOSCAN_CORE_EXPORT const void* graph_port_declaration(const std::shared_ptr<GraphState>& state,
                                                        GraphPortId port,
                                                        PortDirection direction) noexcept;
HOLOSCAN_CORE_EXPORT const void* graph_port_declaration(
    const std::shared_ptr<GraphState>& state, GraphPortId port, PortDirection direction,
    const PayloadIdentityRecord& payload_identity) noexcept;
HOLOSCAN_CORE_EXPORT bool graph_flow_valid(const std::shared_ptr<GraphState>& state, EdgeId edge,
                                           GraphPortId source, GraphPortId target) noexcept;

HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_input_queue(
    const std::shared_ptr<GraphState>&, GraphPortId, std::uint32_t);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_input_tensor_spec(
    const std::shared_ptr<GraphState>&, GraphPortId, const TensorInputSpec&);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_port_alias(
    const std::shared_ptr<GraphState>&, GraphPortId, std::string_view);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_output_emits(
    const std::shared_ptr<GraphState>&, GraphPortId, std::uint32_t);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_output_history(
    const std::shared_ptr<GraphState>&, GraphPortId, std::uint32_t);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_output_tensor_spec(
    const std::shared_ptr<GraphState>&, GraphPortId, const TensorOutputSpec&);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_output_payload(
    const std::shared_ptr<GraphState>&, GraphPortId, PayloadOptions);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_flow_policy(
    const std::shared_ptr<GraphState>&, EdgeId, QueuePolicy);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_flow_depth(
    const std::shared_ptr<GraphState>&, EdgeId, std::uint32_t);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_flow_topic(
    const std::shared_ptr<GraphState>&, EdgeId, std::string_view);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_flow_memory(
    const std::shared_ptr<GraphState>&, EdgeId, MemoryConversionPolicy);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_flow_inflight(
    const std::shared_ptr<GraphState>&, EdgeId, std::uint32_t);

HOLOSCAN_CORE_EXPORT expected<GraphPortId, Error> graph_register_import(
    const std::shared_ptr<GraphState>&, std::string_view, TopicOptions, std::unique_ptr<OutputBase>,
    PayloadIdentityRecord, std::uint64_t serialized_payload_capacity_bytes,
    std::vector<AuthoredDescriptorTensorTopicField> tensor_fields = {},
    std::uint16_t descriptor_tensor_field_count = 0U);
HOLOSCAN_CORE_EXPORT expected<GraphPortId, Error> graph_register_export(
    const std::shared_ptr<GraphState>&, std::string_view, TopicOptions, std::unique_ptr<InputBase>,
    PayloadIdentityRecord, std::uint64_t serialized_payload_capacity_bytes,
    std::vector<AuthoredDescriptorTensorTopicField> tensor_fields = {},
    std::uint16_t descriptor_tensor_field_count = 0U);

HOLOSCAN_CORE_EXPORT expected<RegionId, Error> graph_add_region(const std::shared_ptr<GraphState>&,
                                                                std::string_view);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_region_add_operator(
    const std::shared_ptr<GraphState>&, RegionId, std::size_t);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_region_add_embedded_graph(
    const std::shared_ptr<GraphState>&, RegionId, std::size_t);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_region_add_constraint(
    const std::shared_ptr<GraphState>&, RegionId, RegionConstraint);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_region_set_profile(
    const std::shared_ptr<GraphState>&, RegionId, Profile);

HOLOSCAN_CORE_EXPORT expected<PartitionId, Error> graph_add_partition(
    const std::shared_ptr<GraphState>&, std::string_view);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_partition_add_operator(
    const std::shared_ptr<GraphState>&, PartitionId, std::size_t);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_partition_add_operators(
    const std::shared_ptr<GraphState>&, PartitionId, std::span<const std::size_t>);
/// @brief Assign one direct-child instance without partial partition membership updates.
/// @pre The non-null GraphState comes from matching live partition and Graph handles.
/// @return An error for invalid identities, conflicting assignments, an empty subtree or
/// internal partition declarations; otherwise publish the complete instance selection.
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_partition_add_embedded_graph(
    const std::shared_ptr<GraphState>&, PartitionId, std::size_t);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_partition_include_all(
    const std::shared_ptr<GraphState>&, PartitionId);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_auto_partition_remaining(
    const std::shared_ptr<GraphState>&);

HOLOSCAN_CORE_EXPORT expected<void, Error> graph_lifecycle_before(
    const std::shared_ptr<GraphState>&, std::size_t, std::size_t, LifecycleStage);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_lifecycle_permission(
    const std::shared_ptr<GraphState>&, LifecycleRestartOrigin, std::optional<std::size_t>,
    std::size_t);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_lifecycle_before_resolved(
    const std::shared_ptr<GraphState>&, const std::shared_ptr<GraphState>&, std::size_t,
    const std::shared_ptr<GraphState>&, std::size_t, LifecycleStage);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_lifecycle_permission_resolved(
    const std::shared_ptr<GraphState>&, LifecycleRestartOrigin, const std::shared_ptr<GraphState>&,
    std::optional<std::size_t>, const std::shared_ptr<GraphState>&, std::size_t);

HOLOSCAN_CORE_EXPORT expected<ClockKey, Error> graph_add_clock(const std::shared_ptr<GraphState>&,
                                                               std::string_view,
                                                               std::shared_ptr<ClockFactoryBase>);
HOLOSCAN_CORE_EXPORT expected<ClockKey, Error> graph_bind_shared_clock(
    const std::shared_ptr<GraphState>&, std::string_view, std::shared_ptr<Clock>);
HOLOSCAN_CORE_EXPORT expected<ClockKey, Error> graph_import_clock(
    const std::shared_ptr<GraphState>&, std::string_view);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_bind_clock_import(
    const std::shared_ptr<GraphState>&, std::size_t, ClockKey, ClockKey);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_set_default_clock(
    const std::shared_ptr<GraphState>&, ClockKey);
HOLOSCAN_CORE_EXPORT expected<ClockKey, Error> graph_default_clock(
    const std::shared_ptr<GraphState>&);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_add_clock_conversion(
    const std::shared_ptr<GraphState>&, ClockKey, ClockKey, AffineClockConversion);

HOLOSCAN_CORE_EXPORT std::shared_ptr<GraphState> make_graph_state(Graph* owner,
                                                                  std::string_view name);
HOLOSCAN_CORE_EXPORT expected<EmbeddedGraphRegistration, Error> graph_register_embedded(
    const std::shared_ptr<GraphState>& parent, std::string_view instance_name,
    std::unique_ptr<Graph> child, const std::shared_ptr<GraphState>& child_state,
    const std::shared_ptr<const GraphSnapshotData>& child_snapshot);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_prepare_embedded_type(
    const std::shared_ptr<GraphState>& parent, const std::shared_ptr<GraphState>& child,
    const void* child_type_token);
HOLOSCAN_CORE_EXPORT const Graph* graph_embedded_graph(const std::shared_ptr<GraphState>&,
                                                       std::size_t) noexcept;
HOLOSCAN_CORE_EXPORT bool graph_embedded_valid(const std::shared_ptr<GraphState>&,
                                               std::size_t) noexcept;
HOLOSCAN_CORE_EXPORT std::optional<GraphInstancePath> graph_embedded_path(
    const std::shared_ptr<GraphState>&, std::size_t);
HOLOSCAN_CORE_EXPORT std::shared_ptr<const GraphSnapshotData> graph_snapshot(
    const std::shared_ptr<GraphState>& state);
HOLOSCAN_CORE_EXPORT expected<TensorOutputPortConfigurator*, Error> graph_tensor_configurator(
    const std::shared_ptr<GraphState>&, GraphPortId);

inline void throw_graph_error(Error&& error) {
  throw RuntimeError{std::move(error)};
}

}  // namespace detail

template <PublishedPayload T>
Output<T>& ImportedTopic<T>::output() const {
  auto state = state_.lock();
  const auto* declaration = state ? detail::graph_port_declaration(
                                        state, GraphPortId{graph_port_}, PortDirection::kOutput)
                                  : nullptr;
  if (declaration == nullptr) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_EXTERNAL_TOPIC_HANDLE_INVALID"}};
  }
  return *static_cast<Output<T>*>(const_cast<void*>(declaration));
}

template <PublishedPayload T>
bool ImportedTopic<T>::graph_valid() const noexcept {
  auto state = state_.lock();
  return state && graph_port_ != 0U &&
         detail::graph_port_declaration(state, GraphPortId{graph_port_}, PortDirection::kOutput) !=
             nullptr;
}

template <PublishedPayload T>
Input<T>& ExportedTopic<T>::input() const {
  auto state = state_.lock();
  const auto* declaration =
      state ? detail::graph_port_declaration(state, GraphPortId{graph_port_}, PortDirection::kInput)
            : nullptr;
  if (declaration == nullptr) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_EXTERNAL_TOPIC_HANDLE_INVALID"}};
  }
  return *static_cast<Input<T>*>(const_cast<void*>(declaration));
}

template <PublishedPayload T>
bool ExportedTopic<T>::graph_valid() const noexcept {
  auto state = state_.lock();
  return state && graph_port_ != 0U &&
         detail::graph_port_declaration(state, GraphPortId{graph_port_}, PortDirection::kInput) !=
             nullptr;
}

template <std::derived_from<OperatorBase> Op>
class OperatorPortProxy {
 public:
  /// @brief Opt this authoring handle out of replayable Graph captures.
  using graph_authoring_handle_tag = void;

  /// @brief Resolve const access to the current sealed authoring prototype.
  /// @return Non-null current prototype pointer.
  /// @throws RuntimeError with `GRAPH_HANDLE_INVALID` if the Graph or declaration is retired.
  /// @note Resolution occurs on every access so parameter reconstruction cannot
  /// leave a cached pointer to a retired prototype.
  [[nodiscard]] const Op* operator->() const {
    auto state = state_.lock();
    const auto* base = state ? detail::graph_operator_prototype(
                                   state, index_, &detail::kGraphOperatorTypeToken<Op>)
                             : nullptr;
    if (base == nullptr) {
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"});
    }
    return static_cast<const Op*>(base);
  }

 private:
  OperatorPortProxy(std::weak_ptr<detail::GraphState> state, std::size_t index) noexcept
      : state_(std::move(state)), index_(index) {}
  std::weak_ptr<detail::GraphState> state_;
  std::size_t index_{};
  template <std::derived_from<OperatorBase>>
  friend class OperatorHandle;
};

/// @brief Hold copyable Graph declaration identity for one Operator type.
template <std::derived_from<OperatorBase> Op>
class OperatorHandle {
 public:
  /// @brief Opt this authoring handle out of replayable Graph captures.
  using graph_authoring_handle_tag = void;

  /// @brief Name the exact concrete Operator type represented by this handle.
  using Operator = Op;

  /// @brief Name the complete generation-zero parameter aggregate for \ref Operator.
  using Params = typename Op::Params;

  /// @brief Name the restricted dynamic prototype accessor used for typed port wiring.
  using PortAccess = OperatorPortProxy<Op>;

  /// @brief Return restricted const prototype access for typed port wiring.
  [[nodiscard]] PortAccess operator->() const noexcept { return PortAccess{state_, index_}; }

  /// @brief Replace the complete generation-zero parameter aggregate.
  /// @param params Valid candidate aggregate copied into replayable factory state.
  /// @throws RuntimeError if validation/reconstruction fails or this handle is stale.
  void set_params(Params params) {
    auto state = state_.lock();
    if (!state)
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"});
    auto result =
        detail::graph_set_operator_params(state,
                                          index_,
                                          &params,
                                          operator_type_token_,
                                          param_type_descriptor<Params>().local_type_token,
                                          param_schema_descriptor<Params>(),
                                          nullptr);
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
  }

  /// @brief Replace parameters and retain their immutable producer evidence.
  /// @param params Prepared value for this exact Operator parameter type.
  /// @throws RuntimeError if validation/reconstruction fails or this handle is stale.
  void set_params(PreparedParams<Params> params) {
    auto state = state_.lock();
    if (!state)
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"});
    auto evidence = detail::PreparedParamsAccess::evidence(params);
    Params value = std::move(params).take();
    auto result =
        detail::graph_set_operator_params(state,
                                          index_,
                                          &value,
                                          operator_type_token_,
                                          param_type_descriptor<Params>().local_type_token,
                                          param_schema_descriptor<Params>(),
                                          std::move(evidence));
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
  }

  /// @brief Return the owned local declaration name.
  [[nodiscard]] std::string_view name() const noexcept {
    auto state = state_.lock();
    return state ? detail::graph_operator_name(state, index_) : std::string_view{};
  }

  /// @brief Return whether the source Graph still owns the exact typed declaration.
  [[nodiscard]] bool graph_valid() const noexcept {
    auto state = state_.lock();
    return state && detail::graph_operator_valid(state, index_, operator_type_token_);
  }

  /// @brief Return whether this handle names a live Graph declaration.
  [[nodiscard]] explicit operator bool() const noexcept { return graph_valid(); }

  /// @brief Return the compiled operator reference when one exists.
  /// @return Invalid zero reference in HSDK5-006; HSDK5-009 binds plan identity.
  [[nodiscard]] OperatorRef ref() const noexcept {
    auto state = state_.lock();
    return state ? detail::graph_operator_ref(state, index_) : OperatorRef{};
  }

  /// @brief Return the compiled operator identity when one exists.
  /// @return Zero in HSDK5-006 because authoring handles never fabricate plan IDs.
  [[nodiscard]] OperatorInstanceId id() const noexcept { return ref().id; }

  /// @brief Return the immutable compiled-plan generation when one exists.
  /// @return Zero until HSDK5-009 freezes this declaration into a plan.
  [[nodiscard]] PlanGenerationId plan_generation() const noexcept { return ref().plan_generation; }

  /// @brief Return whether this handle is bound to an immutable compiled plan.
  /// @return Always false for the HSDK5-006 authoring-only handle.
  [[nodiscard]] bool plan_bound() const noexcept { return ref().plan_bound(); }

 private:
  OperatorHandle(const void* operator_type_token, std::weak_ptr<detail::GraphState> state,
                 std::size_t index) noexcept
      : operator_type_token_(operator_type_token), state_(std::move(state)), index_(index) {}
  const void* operator_type_token_{};
  std::weak_ptr<detail::GraphState> state_;
  std::size_t index_{};
  friend class Graph;
  friend class RegionHandle;
  friend class PartitionHandle;
  friend class GraphLifecycleBuilder;
  friend class ExecutionPartitionsView;
  friend class detail::ParameterConfigBindingAccess;
  friend class authoring::OpaqueOperatorRef;
};

namespace detail {

/// @brief Expose call-scoped authoring ownership to optional Parameter tooling.
class ParameterConfigBindingAccess {
 public:
  [[nodiscard]] static const std::shared_ptr<GraphState>& state(Graph& graph) noexcept;

  template <std::derived_from<OperatorBase> Op>
  [[nodiscard]] static const std::weak_ptr<GraphState>& state(
      const OperatorHandle<Op>& handle) noexcept {
    return handle.state_;
  }

  template <std::derived_from<OperatorBase> Op>
  [[nodiscard]] static expected<GraphOperatorParameterTarget, Error> target(
      const OperatorHandle<Op>& handle) {
    auto state = handle.state_.lock();
    if (!state) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument, "PARAM_CONFIG_HANDLE_INVALID"});
    }
    return graph_operator_parameter_target(state, handle.index_, handle.operator_type_token_);
  }

  template <std::derived_from<OperatorBase> Op>
  [[nodiscard]] static expected<VertexId, Error> vertex(const OperatorHandle<Op>& handle) {
    auto resolved = target(handle);
    if (!resolved) {
      return make_unexpected(std::move(resolved.error()));
    }
    return resolved->vertex;
  }
};

}  // namespace detail

/// @brief Hold read-only typed identity for one independently owned embedded Graph.
///
/// The direct parent Graph owns the exact child object and its complete authoring
/// state. A handle never extends that lifetime, never permits child mutation, and
/// never aliases a separately embedded instance of the same reusable Graph type.
/// Its path is relative to that direct parent; canonical root-qualified instance
/// identity after any outer embedding is exposed by \ref GraphView.
///
/// @tparam G Concrete reusable Graph type.
template <typename G>
class GraphHandle {
 public:
  /// @brief Opt this authoring handle out of replayable Graph captures.
  using graph_authoring_handle_tag = void;

  static_assert(std::derived_from<G, Graph>, "GraphHandle requires a Graph-derived type");
  /// @brief Name the exact reusable Graph type represented by this handle.
  using GraphType = G;

  /// @brief Resolve read-only access to the embedded Graph's public interface members.
  /// @return Non-null exact child instance.
  /// @throws RuntimeError with `GRAPH_HANDLE_INVALID` if the parent or embedded declaration is
  /// retired.
  /// @note The returned pointer grants no Graph authoring mutation authority.
  [[nodiscard]] const G* operator->() const {
    auto state = state_.lock();
    const auto* child =
        state ? static_cast<const G*>(detail::graph_embedded_graph(state, index_)) : nullptr;
    if (child == nullptr) {
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"});
    }
    return child;
  }

  /// @brief Return an owning structured path relative to the direct parent.
  /// @return Parent-relative path while the parent owns this registration, or
  /// no value after parent retirement or if the registration is invalid.
  /// @throws std::bad_alloc if allocating the owning path copy fails.
  /// @note A containing GraphView supplies the root-qualified path after this
  /// Graph is itself embedded in an outer Graph.
  [[nodiscard]] std::optional<GraphInstancePath> path() const {
    auto state = state_.lock();
    return state ? detail::graph_embedded_path(state, index_) : std::nullopt;
  }

  /// @brief Return whether the parent still owns this exact typed child instance.
  /// @return True only while the parent state and typed registration remain live.
  [[nodiscard]] bool graph_valid() const noexcept {
    auto state = state_.lock();
    return state && detail::graph_embedded_valid(state, index_);
  }

  /// @brief Return whether this handle names a live embedded Graph.
  [[nodiscard]] explicit operator bool() const noexcept { return graph_valid(); }

 private:
  GraphHandle(std::weak_ptr<detail::GraphState> state, std::size_t index)
      : state_(std::move(state)), index_(index) {}
  std::weak_ptr<detail::GraphState> state_;
  std::size_t index_{};
  friend class Graph;
  friend class RegionHandle;
  friend class PartitionHandle;
};

/// @brief Hold one semantically compatible typed Graph edge declaration.
/// @tparam SourceT Exact C++ source payload type.
/// @tparam TargetT Exact C++ target payload type; defaults to \c SourceT.
///
/// Different endpoint types are admitted only when both declare explicit raw-POD
/// contracts and their complete canonical name, version, object layout, and field
/// records compare equal. Every other payload family remains exact-C++-type only.
template <PublishedPayload SourceT, PublishedPayload TargetT = SourceT>
class FlowHandle {
 public:
  /// @brief Opt this authoring handle out of replayable Graph captures.
  using graph_authoring_handle_tag = void;

  /// @brief Return the snapshot-local authored edge identity.
  /// @return Nonzero edge identity while the source Graph exists.
  [[nodiscard]] EdgeId id() const noexcept { return id_; }

  /// @brief Resolve the current source declaration.
  /// @return Current output member after any atomic parameter reconstruction.
  /// @throws RuntimeError if the Graph retired or the structural declaration changed.
  [[nodiscard]] const Output<SourceT>& source() const {
    auto state = state_.lock();
    if (!state)
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_FLOW_HANDLE_INVALID"});
    const void* declaration = detail::graph_port_declaration(
        state, source_, PortDirection::kOutput, detail::operator_spec_payload_identity<SourceT>());
    if (declaration == nullptr) {
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_FLOW_SOURCE_INVALID"});
    }
    return *static_cast<const Output<SourceT>*>(declaration);
  }

  /// @brief Resolve the current target declaration.
  /// @return Current input member after any atomic parameter reconstruction.
  /// @throws RuntimeError if the Graph retired or the structural declaration changed.
  [[nodiscard]] const Input<TargetT>& target() const {
    auto state = state_.lock();
    if (!state)
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_FLOW_HANDLE_INVALID"});
    const void* declaration = detail::graph_port_declaration(
        state, target_, PortDirection::kInput, detail::operator_spec_payload_identity<TargetT>());
    if (declaration == nullptr) {
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_FLOW_TARGET_INVALID"});
    }
    return *static_cast<const Input<TargetT>*>(declaration);
  }

  /// @brief Return whether this edge remains owned by its source Graph.
  /// @return True while the Graph and nonzero authoring identity remain live.
  [[nodiscard]] bool graph_valid() const noexcept {
    auto state = state_.lock();
    return state && detail::graph_flow_valid(state, id_, source_, target_);
  }

 private:
  FlowHandle(std::weak_ptr<detail::GraphState> state, EdgeId id, GraphPortId source,
             GraphPortId target) noexcept
      : state_(std::move(state)), id_(id), source_(source), target_(target) {}
  std::weak_ptr<detail::GraphState> state_;
  EdgeId id_{};
  GraphPortId source_{};
  GraphPortId target_{};
  friend class Graph;
};

/// @brief Configure cold authoring policy for one Graph-owned input port.
///
/// A configurator is a non-owning handle. It remains usable only while the
/// originating Graph and exact port declaration remain live. Each successful
/// call atomically updates cold authoring state; a rejected call leaves the
/// port and its incident edges unchanged. Configurators do not make Graph
/// authoring thread-safe, so callers must serialize them with all other reads
/// and mutations of the same Graph.
///
/// @tparam T Exact published payload type of the input declaration.
template <PublishedPayload T>
class InputPortConfigurator {
 public:
  /// @brief Opt this authoring handle out of replayable Graph captures.
  using graph_authoring_handle_tag = void;

  /// @brief Override the finite input history depth.
  /// @param depth Positive number of samples retained for this logical input.
  /// @return This configurator for fluent authoring.
  /// @throws RuntimeError if \p depth is zero or the source Graph is unavailable.
  InputPortConfigurator& queue_depth(std::uint32_t depth) {
    apply(detail::graph_configure_input_queue(state(), port_, depth));
    return *this;
  }

  /// @brief Override this input with one common consumer Tensor spec.
  /// @param spec Representation, finite bounds, and conversion requirements.
  /// @return This configurator for fluent authoring.
  InputPortConfigurator& expects_tensor(const TensorInputSpec& spec) {
    apply(detail::graph_configure_input_tensor_spec(state(), port_, spec));
    return *this;
  }

  /// @brief Assign an owned diagnostic alias without changing port identity.
  /// @param alias Nonempty bounded UTF-8 alias copied before return.
  /// @return This configurator for fluent authoring.
  /// @throws RuntimeError if \p alias is invalid or the source Graph is unavailable.
  InputPortConfigurator& diagnostic_alias(std::string_view alias) {
    apply(detail::graph_configure_port_alias(state(), port_, alias));
    return *this;
  }

 private:
  InputPortConfigurator(std::weak_ptr<detail::GraphState> state, GraphPortId port) noexcept
      : state_(std::move(state)), port_(port) {}
  [[nodiscard]] std::shared_ptr<detail::GraphState> state() const {
    auto value = state_.lock();
    if (!value)
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"});
    return value;
  }
  static void apply(expected<void, Error> result) {
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
  }
  std::weak_ptr<detail::GraphState> state_;
  GraphPortId port_{};
  friend class Graph;
};

/// @brief Configure cold authoring policy for one Graph-owned output port.
///
/// A configurator is a non-owning handle. It remains usable only while the
/// originating Graph and exact port declaration remain live. Each successful
/// call atomically updates cold authoring state; a rejected call leaves the
/// port and its incident edges unchanged. Configurators do not make Graph
/// authoring thread-safe, so callers must serialize them with all other reads
/// and mutations of the same Graph.
///
/// @tparam T Exact published payload type of the output declaration.
template <PublishedPayload T>
class OutputPortConfigurator {
 public:
  /// @brief Opt this authoring handle out of replayable Graph captures.
  using graph_authoring_handle_tag = void;

  /// @brief Override the positive per-compute emission bound.
  /// @param count Positive maximum accepted emissions per compute call.
  /// @return This configurator for fluent authoring.
  /// @throws RuntimeError if \p count is zero or the source Graph is unavailable.
  OutputPortConfigurator& max_emits_per_compute(std::uint32_t count = 1) {
    apply(detail::graph_configure_output_emits(state(), port_, count));
    return *this;
  }

  /// @brief Override the finite late-joiner history depth.
  /// @param depth Positive retained-sample depth.
  /// @return This configurator for fluent authoring.
  /// @throws RuntimeError if \p depth is zero or the source Graph is unavailable.
  OutputPortConfigurator& late_joiner_history_depth(std::uint32_t depth) {
    apply(detail::graph_configure_output_history(state(), port_, depth));
    return *this;
  }

  /// @brief Override this output with one Tensor spec.
  /// @param spec Representation, finite bounds, and transfer promises.
  /// @return This configurator for fluent authoring.
  /// @note kPassThrough does not remove Runtime-pool storage declared during Operator setup;
  /// Graph overrides may tighten setup resource declarations but do not retract them.
  OutputPortConfigurator& produces_tensor(const TensorOutputSpec& spec) {
    apply(detail::graph_configure_output_tensor_spec(state(), port_, spec));
    return *this;
  }

  /// @brief Override schema identity and payload options for this output.
  /// @param options Options copied before return.
  /// @return This configurator for fluent authoring.
  OutputPortConfigurator& payload_options(PayloadOptions options) {
    apply(detail::graph_configure_output_payload(state(), port_, options));
    return *this;
  }

  /// @brief Assign an owned diagnostic alias without changing port identity.
  /// @param alias Nonempty bounded UTF-8 alias copied before return.
  /// @return This configurator for fluent authoring.
  /// @throws RuntimeError if \p alias is invalid or the source Graph is unavailable.
  OutputPortConfigurator& diagnostic_alias(std::string_view alias) {
    apply(detail::graph_configure_port_alias(state(), port_, alias));
    return *this;
  }

 private:
  OutputPortConfigurator(std::weak_ptr<detail::GraphState> state, GraphPortId port) noexcept
      : state_(std::move(state)), port_(port) {}
  [[nodiscard]] std::shared_ptr<detail::GraphState> state() const {
    auto value = state_.lock();
    if (!value)
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"});
    return value;
  }
  static void apply(expected<void, Error> result) {
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
  }
  std::weak_ptr<detail::GraphState> state_;
  GraphPortId port_{};
  friend class Graph;
};

/// @brief Configure a direct Tensor output while preserving specialized fluent type.
class HOLOSCAN_CORE_EXPORT TensorOutputPortConfigurator {
 public:
  /// @brief Opt this authoring handle out of replayable Graph captures.
  using graph_authoring_handle_tag = void;

  /// @brief Override the positive per-compute emission bound.
  /// @param count Positive maximum accepted emissions per compute call.
  /// @return This stable Graph-owned configurator.
  /// @throws RuntimeError if \p count is zero or the source Graph is unavailable.
  TensorOutputPortConfigurator& max_emits_per_compute(std::uint32_t count = 1);

  /// @brief Override the finite late-joiner history depth.
  /// @param depth Positive retained-sample depth.
  /// @return This stable Graph-owned configurator.
  /// @throws RuntimeError if \p depth is zero or the source Graph is unavailable.
  TensorOutputPortConfigurator& late_joiner_history_depth(std::uint32_t depth);

  /// @brief Override this output with one common producer Tensor spec.
  /// @param spec Representation, finite bounds, storage intent, and transfer policy.
  /// @return This stable Graph-owned configurator.
  TensorOutputPortConfigurator& produces_tensor(const TensorOutputSpec& spec);

  /// @brief Override schema identity and payload options for this output.
  /// @param options Options copied before return.
  /// @return This stable Graph-owned configurator.
  TensorOutputPortConfigurator& payload_options(PayloadOptions options);

  /// @brief Assign an owned diagnostic alias without changing port identity.
  /// @param alias Nonempty bounded UTF-8 alias copied before return.
  /// @return This stable Graph-owned configurator.
  /// @throws RuntimeError if \p alias is invalid or the source Graph is unavailable.
  TensorOutputPortConfigurator& diagnostic_alias(std::string_view alias);

 private:
  TensorOutputPortConfigurator(std::weak_ptr<detail::GraphState> state, GraphPortId port) noexcept
      : state_(std::move(state)), port_(port) {}
  [[nodiscard]] std::shared_ptr<detail::GraphState> state() const;
  static void apply(expected<void, Error> result);
  std::weak_ptr<detail::GraphState> state_;
  GraphPortId port_{};
  friend class Graph;
  friend class detail::GraphState;
  friend expected<TensorOutputPortConfigurator*, Error> detail::graph_tensor_configurator(
      const std::shared_ptr<detail::GraphState>&, GraphPortId);
};

/// @brief Configure cold queue and transport policy for one Graph-owned flow.
///
/// A configurator is a non-owning handle to an already-authored edge. It
/// remains usable only while the originating Graph and edge remain live. Each
/// successful call atomically updates the edge; rejected calls preserve the
/// previous declaration. Configurators do not make Graph authoring thread-safe,
/// so callers must serialize them with all other reads and mutations of the
/// same Graph.
///
/// @tparam SourceT Exact admitted source payload type.
/// @tparam TargetT Exact admitted target payload type. A distinct type is
/// admitted only through an equal explicit raw-POD contract.
template <PublishedPayload SourceT, PublishedPayload TargetT = SourceT>
class FlowConfigurator {
 public:
  /// @brief Opt this authoring handle out of replayable Graph captures.
  using graph_authoring_handle_tag = void;

  /// @brief Override the authored finite-queue behavior.
  /// @param policy Closed queue policy.
  /// @return This configurator for fluent authoring.
  /// @throws RuntimeError if \p policy is invalid or the flow is unavailable.
  FlowConfigurator& queue_policy(QueuePolicy policy) {
    apply(detail::graph_configure_flow_policy(state(), edge_, policy));
    return *this;
  }
  /// @brief Override the positive destination queue depth.
  /// @param depth Positive queue depth.
  /// @return This configurator for fluent authoring.
  /// @throws RuntimeError if \p depth is zero or the flow is unavailable.
  FlowConfigurator& queue_depth(std::uint32_t depth) {
    apply(detail::graph_configure_flow_depth(state(), edge_, depth));
    return *this;
  }
  /// @brief Override the owned logical flow alias.
  /// @param name Nonempty bounded UTF-8 alias copied before return.
  /// @return This configurator for fluent authoring.
  /// @throws RuntimeError if \p name is invalid or the flow is unavailable.
  FlowConfigurator& topic_name(std::string_view name) {
    apply(detail::graph_configure_flow_topic(state(), edge_, name));
    return *this;
  }
  /// @brief Override the Tensor memory-conversion request.
  /// @param policy Closed conversion policy.
  /// @return This configurator for fluent authoring.
  /// @throws RuntimeError if \p policy is invalid, the flow is unavailable, or
  /// neither endpoint has direct or setup-declared Tensor semantics.
  FlowConfigurator& memory_conversion(MemoryConversionPolicy policy) {
    apply(detail::graph_configure_flow_memory(state(), edge_, policy));
    return *this;
  }
  /// @brief Override the positive in-flight retention bound.
  /// @param count Positive maximum in-flight sample count.
  /// @return This configurator for fluent authoring.
  /// @throws RuntimeError if \p count is zero or the flow is unavailable.
  FlowConfigurator& max_inflight_samples(std::uint32_t count) {
    apply(detail::graph_configure_flow_inflight(state(), edge_, count));
    return *this;
  }

 private:
  FlowConfigurator(std::weak_ptr<detail::GraphState> state, EdgeId edge) noexcept
      : state_(std::move(state)), edge_(edge) {}
  [[nodiscard]] std::shared_ptr<detail::GraphState> state() const {
    auto value = state_.lock();
    if (!value)
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_HANDLE_INVALID"});
    return value;
  }
  static void apply(expected<void, Error> result) {
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
  }
  std::weak_ptr<detail::GraphState> state_;
  EdgeId edge_{};
  friend class Graph;
};

/// @brief Author one named logical region and its cold placement constraints.
///
/// This non-owning handle refers to one region in the originating Graph. It
/// remains usable only while that Graph and declaration remain live. Region
/// mutation is atomic on failure but is not thread-safe with any other read or
/// mutation of the same Graph.
class HOLOSCAN_CORE_EXPORT RegionHandle {
 public:
  /// @brief Opt this authoring handle out of replayable Graph captures.
  using graph_authoring_handle_tag = void;

  /// @brief Add one logical Operator to this region.
  /// @tparam Op Exact Operator type.
  /// @param op Graph-owned Operator declaration.
  /// @return This handle for fluent authoring.
  /// @throws RuntimeError if the handle is foreign, stale, or already a member.
  template <std::derived_from<OperatorBase> Op>
  RegionHandle& add(const OperatorHandle<Op>& op) {
    auto state = state_.lock();
    auto op_state = op.state_.lock();
    if (!state || state != op_state) {
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_REGION_HANDLE_INVALID"});
    }
    auto result = detail::graph_region_add_operator(state, id_, op.index_);
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return *this;
  }

  /// @brief Add every flattened Operator from one embedded Graph to this region.
  /// @tparam G Exact reusable Graph type.
  /// @param graph Parent-owned embedded Graph instance.
  /// @return This handle for fluent authoring.
  /// @throws RuntimeError if the handle is foreign, stale, or already a member.
  /// @note A region containing an embedded Graph cannot use
  /// \ref RegionConstraint::cuda_graph(). CUDA Graph regions are currently limited to directly
  /// listed operators in the root Graph.
  template <typename G>
    requires std::derived_from<G, Graph>
  RegionHandle& add(const GraphHandle<G>& graph) {
    auto state = state_.lock();
    if (!state || state != graph.state_.lock() || !graph.graph_valid()) {
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_REGION_HANDLE_INVALID"});
    }
    auto result = detail::graph_region_add_embedded_graph(state, id_, graph.index_);
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return *this;
  }
  /// @brief Add one unique backend-neutral region constraint.
  /// @param constraint Valid constraint not already present by kind.
  /// @return This handle for fluent authoring.
  /// @throws RuntimeError if the constraint is invalid or duplicated.
  RegionHandle& constrain(RegionConstraint constraint);

  /// @brief Set this region's capability profile request.
  /// @param profile Closed profile value.
  /// @return This handle for fluent authoring.
  /// @throws RuntimeError if \p profile is invalid or already set.
  RegionHandle& profile(Profile profile);

  /// @brief Return this region's snapshot-local authoring identity.
  /// @return Nonzero region identity while the declaration is live.
  [[nodiscard]] RegionId id() const noexcept { return id_; }

 private:
  RegionHandle(std::weak_ptr<detail::GraphState> state, RegionId id) noexcept
      : state_(std::move(state)), id_(id) {}
  std::weak_ptr<detail::GraphState> state_;
  RegionId id_{};
  friend class Graph;
};

/// @brief Author one developer-preview manual execution partition.
///
/// This non-owning handle refers to one partition declaration in the
/// originating Graph. Explicit membership accepts direct Operator declarations
/// and complete embedded instances from that Graph. The compile-root \ref include_all directive
/// instead expands to all composed Operators. A compiler validates the final
/// cover according to the Graph's exact manual or automatic-remainder intent.
///
/// The handle remains usable only while the originating Graph and declaration
/// remain live. Mutation is atomic on failure but is not thread-safe with any
/// other read or mutation of the same Graph.
class HOLOSCAN_CORE_EXPORT PartitionHandle {
 public:
  /// @brief Opt this authoring handle out of replayable Graph captures.
  using graph_authoring_handle_tag = void;

  /// @brief Add one direct logical Operator to this partition.
  /// @tparam Op Exact Operator type.
  /// @param op Graph-owned direct Operator declaration.
  /// @return This handle for fluent authoring.
  /// @throws RuntimeError if either handle is foreign or stale, or if the
  /// Operator is already assigned to any manual partition in this Graph.
  template <std::derived_from<OperatorBase> Op>
  PartitionHandle& add(const OperatorHandle<Op>& op) {
    auto state = state_.lock();
    auto op_state = op.state_.lock();
    if (!state || !op_state) {
      detail::throw_graph_error(
          Error{ErrorCode::kInvalidArgument, "GRAPH_PARTITION_HANDLE_INVALID"});
    }
    if (state != op_state) {
      detail::throw_graph_error(
          Error{ErrorCode::kInvalidArgument, "GRAPH_PARTITION_MEMBER_FOREIGN"});
    }
    auto result = detail::graph_partition_add_operator(state, id_, op.index_);
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return *this;
  }

  /// @brief Add a complete batch of direct logical Operators to this partition.
  /// @tparam Ops Exact Operator types.
  /// @param operators Graph-owned direct Operator declarations.
  /// @return This handle for fluent authoring.
  /// @throws RuntimeError if a handle is foreign or stale, if any Operator is
  /// duplicated within the batch or already assigned, or if this declaration
  /// uses \ref include_all. A rejected batch leaves membership unchanged.
  template <std::derived_from<OperatorBase>... Ops>
    requires(sizeof...(Ops) > 1U)
  PartitionHandle& add(const OperatorHandle<Ops>&... operators) {
    auto state = state_.lock();
    if (!state) {
      detail::throw_graph_error(
          Error{ErrorCode::kInvalidArgument, "GRAPH_PARTITION_HANDLE_INVALID"});
    }

    std::array<std::size_t, sizeof...(Ops)> indexes{};
    std::size_t index{};
    bool invalid{};
    bool foreign{};
    const auto collect = [&](const auto& op) {
      auto op_state = op.state_.lock();
      if (!op_state) {
        invalid = true;
      } else if (state != op_state) {
        foreign = true;
      }
      indexes[index++] = op.index_;
    };
    (collect(operators), ...);
    if (invalid) {
      detail::throw_graph_error(
          Error{ErrorCode::kInvalidArgument, "GRAPH_PARTITION_HANDLE_INVALID"});
    }
    if (foreign) {
      detail::throw_graph_error(
          Error{ErrorCode::kInvalidArgument, "GRAPH_PARTITION_MEMBER_FOREIGN"});
    }

    auto result = detail::graph_partition_add_operators(state, id_, indexes);
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return *this;
  }

  /// @brief Assign one embedded Graph instance and all descendant Operators atomically.
  /// @tparam G Exact embedded Graph type.
  /// @param graph Live direct-child instance owned by this partition's Graph.
  /// @return This handle for fluent authoring with other instances or direct Operators.
  /// @throws RuntimeError if either handle is stale or foreign, the instance is already
  /// assigned, the subtree has no Operators or declares any internal partition (even empty),
  /// or this partition uses \ref include_all. Rejection leaves Graph identity and membership
  /// unchanged. Compilation preserves the exact Operator set and validates placement constraints.
  /// @note Independent sources in the selected subtree share one serial partition owner.
  template <std::derived_from<Graph> G>
  PartitionHandle& add(const GraphHandle<G>& graph) {
    auto state = state_.lock();
    auto graph_state = graph.state_.lock();
    if (!state || !graph_state) {
      detail::throw_graph_error(
          Error{ErrorCode::kInvalidArgument, "GRAPH_PARTITION_HANDLE_INVALID"});
    }
    if (state != graph_state) {
      detail::throw_graph_error(
          Error{ErrorCode::kInvalidArgument, "GRAPH_PARTITION_MEMBER_FOREIGN"});
    }
    auto result = detail::graph_partition_add_embedded_graph(state, id_, graph.index_);
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return *this;
  }

  /// @brief Expand this partition to the complete compile-root application.
  /// @return This handle for fluent authoring.
  /// @throws RuntimeError if the declaration has explicit members, another
  /// partition exists, automatic remainder authoring was requested, or the
  /// declaration is later embedded below the compile root.
  /// @note Expansion occurs from the final composed snapshot at compile time,
  /// so Operators and embedded Graphs authored after this call are included.
  PartitionHandle& include_all();

 private:
  PartitionHandle(std::weak_ptr<detail::GraphState> state, PartitionId id) noexcept
      : state_(std::move(state)), id_(id) {}
  std::weak_ptr<detail::GraphState> state_;
  PartitionId id_{};
  friend class Graph;
};

/// @brief Author lifecycle ordering and restart permissions for one Graph.
///
/// The builder is a non-owning view of the originating Graph's cold lifecycle
/// declaration. It remains usable only while that Graph remains live. Each
/// accepted call records one dependency or permission and each rejected call
/// leaves the declaration unchanged. The builder does not make Graph authoring
/// thread-safe; callers must serialize it with all other Graph access.
class HOLOSCAN_CORE_EXPORT GraphLifecycleBuilder {
 public:
  /// @brief Opt this authoring handle out of replayable Graph captures.
  using graph_authoring_handle_tag = void;

  /// @brief Order two logical checkpoints within one lifecycle stage.
  /// @tparam BeforeT Predecessor Operator type.
  /// @tparam AfterT Successor Operator type.
  /// @param before Operator whose checkpoint must complete first.
  /// @param after Operator released after \p before.
  /// @param stage Exact same-stage ordering scope.
  /// @return This builder for fluent authoring.
  /// @throws RuntimeError for foreign/stale handles, duplicates, cycles, or invalid stages.
  /// @note The declaring Graph scope may order Operators in that scope or any descendant scope.
  /// Both endpoints must be descendants of the declaring scope; sibling Graphs cannot author an
  /// edge over one another's Operators.
  template <std::derived_from<OperatorBase> BeforeT, std::derived_from<OperatorBase> AfterT>
  GraphLifecycleBuilder& before(const OperatorHandle<BeforeT>& before,
                                const OperatorHandle<AfterT>& after, LifecycleStage stage) {
    auto state = state_.lock();
    auto before_state = before.state_.lock();
    auto after_state = after.state_.lock();
    if (!state || !before_state || !after_state) {
      detail::throw_graph_error(
          Error{ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_HANDLE_INVALID"});
    }
    auto result = detail::graph_lifecycle_before_resolved(
        state, before_state, before.index_, after_state, after.index_, stage);
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return *this;
  }

  /// @brief Authorize one Operator to request its own lifecycle restart.
  /// @tparam TargetT Target Operator type.
  /// @param target Exact restart target.
  /// @return This builder for fluent authoring.
  /// @throws RuntimeError if \p target is foreign/stale or permission is duplicated.
  template <std::derived_from<OperatorBase> TargetT>
  GraphLifecycleBuilder& allow_self_restart(const OperatorHandle<TargetT>& target) {
    return permission(LifecycleRestartOrigin::kSelf, std::nullopt, target);
  }

  /// @brief Authorize external run control to restart one Operator.
  /// @tparam TargetT Target Operator type.
  /// @param target Exact restart target.
  /// @return This builder for fluent authoring.
  /// @throws RuntimeError if \p target is foreign/stale or permission is duplicated.
  template <std::derived_from<OperatorBase> TargetT>
  GraphLifecycleBuilder& allow_external_restart(const OperatorHandle<TargetT>& target) {
    return permission(LifecycleRestartOrigin::kExternal, std::nullopt, target);
  }

  /// @brief Authorize one exact Operator to request restart of another.
  /// @tparam RequesterT Requesting Operator type.
  /// @tparam TargetT Target Operator type.
  /// @param requester Exact authorized requester.
  /// @param target Exact restart target.
  /// @return This builder for fluent authoring.
  /// @throws RuntimeError for foreign/stale handles or duplicate permission.
  template <std::derived_from<OperatorBase> RequesterT, std::derived_from<OperatorBase> TargetT>
  GraphLifecycleBuilder& allow_restart_request(const OperatorHandle<RequesterT>& requester,
                                               const OperatorHandle<TargetT>& target) {
    auto state = state_.lock();
    auto requester_state = requester.state_.lock();
    auto target_state = target.state_.lock();
    if (!state || !requester_state || !target_state) {
      detail::throw_graph_error(
          Error{ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_HANDLE_INVALID"});
    }
    auto result =
        detail::graph_lifecycle_permission_resolved(state,
                                                    LifecycleRestartOrigin::kOperatorRequest,
                                                    requester_state,
                                                    requester.index_,
                                                    target_state,
                                                    target.index_);
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return *this;
  }

 private:
  explicit GraphLifecycleBuilder(std::weak_ptr<detail::GraphState> state) noexcept
      : state_(std::move(state)) {}
  template <std::derived_from<OperatorBase> TargetT>
  GraphLifecycleBuilder& permission(LifecycleRestartOrigin origin,
                                    std::optional<std::size_t> requester,
                                    const OperatorHandle<TargetT>& target) {
    auto state = state_.lock();
    auto target_state = target.state_.lock();
    if (!state || !target_state) {
      detail::throw_graph_error(
          Error{ErrorCode::kInvalidArgument, "GRAPH_LIFECYCLE_HANDLE_INVALID"});
    }
    auto result = detail::graph_lifecycle_permission_resolved(
        state, origin, state, requester, target_state, target.index_);
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return *this;
  }
  std::weak_ptr<detail::GraphState> state_;
  friend class Graph;
  friend class authoring::OpaqueOperatorRef;
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

template <typename T>
concept GraphAuthoringHandle =
    requires { typename std::remove_cvref_t<T>::graph_authoring_handle_tag; };

template <typename T>
concept ShallowCapture =
    (!std::is_pointer_v<std::remove_cvref_t<T>> ||
     std::is_function_v<std::remove_pointer_t<std::remove_cvref_t<T>>>) &&
    !IsSpan<std::remove_cvref_t<T>>::value && !IsReferenceWrapper<std::remove_cvref_t<T>>::value &&
    !IsInitializerList<std::remove_cvref_t<T>>::value && !GraphAuthoringHandle<T> &&
    (!std::ranges::view<std::remove_cvref_t<T>> || kStringCapture<T>) &&
    (!(std::ranges::range<std::remove_cvref_t<T>> &&
       std::ranges::borrowed_range<std::remove_cvref_t<T>>) ||
     kStringCapture<T>);

template <typename T>
concept ReplayableCapture = ShallowCapture<T> && std::copy_constructible<NormalizedGraphCapture<T>>;

}  // namespace detail

/// @brief Own an editable cold graph declaration and replayable construction factories.
///
/// Graph is the only HSDK 5 root authoring abstraction. It performs lazy
/// composition and cold validation but does not compile or execute a graph.
/// The lazy-composition gate is internally synchronized, so concurrent observers
/// share one composer and receive its terminal result. This synchronization does
/// not make general authoring thread-safe: after composition, callers must
/// serialize mutation with every other mutation or observation of the same
/// Graph. Handles do not extend Graph lifetime and become invalid when the Graph
/// is destroyed.
///
/// @warning The complete most-derived Graph object must remain alive until all
/// authoring, observation, and lazy-composition calls have returned. Destruction
/// must not race any Graph access; handle invalidation does not provide concurrent
/// destruction synchronization.
///
/// Every successful public compound mutation is atomic. If allocation or an
/// injected exceptional boundary interrupts an in-progress mutation, Graph
/// restores the prior snapshot identity, generation, IDs, ownership claims,
/// and declaration indexes before the exception escapes. A rejected external
/// topic declaration is the deliberate exception to counter rollback: ordinary
/// typed validation failures consume their mixed import/export declaration
/// ordinal, while exceptional failures restore the reserved ordinal.
class HOLOSCAN_CORE_EXPORT Graph {
 public:
  /// @brief Create an empty named Graph authoring root.
  /// @param name Optional bounded UTF-8 name copied before return.
  /// @throws RuntimeError if \p name is invalid.
  explicit Graph(std::string_view name = "");

  /// @brief Destroy Graph-owned prototypes, factories, and authoring state.
  /// @warning Destruction must not overlap any authoring, observation, composition,
  /// or handle operation associated with this Graph.
  virtual ~Graph();

  /// @brief Reject copying because authoring identities and ownership claims are instance-local.
  Graph(const Graph&) = delete;

  /// @brief Reject copy assignment because Graph-owned declarations cannot change roots.
  Graph& operator=(const Graph&) = delete;

  /// @brief Reject moving because handles retain the source Graph's stable instance identity.
  Graph(Graph&&) = delete;

  /// @brief Reject move assignment because existing handles cannot be rebound to another root.
  Graph& operator=(Graph&&) = delete;

  /// @brief Add one typed Operator declaration and replayable factory.
  /// @tparam Op Concrete Operator type.
  /// @tparam Args Constructor capture argument types.
  /// @param name Nonempty unique bounded UTF-8 declaration name.
  /// @param args Constructor values normalized and copied into replayable state.
  /// @return Copyable authoring handle that dynamically resolves current ports.
  /// @throws RuntimeError if construction, setup, naming, or registration fails.
  /// @note If this is the first external authoring operation on a derived Graph, Core
  /// completes or terminally fails `compose()` before constructing the requested
  /// Operator factory or prototype.
  /// @pre Each normalized capture must be copy-constructible. Copies follow the
  /// captured type's ordinary C++ semantics, including shared identity for
  /// `std::shared_ptr` and non-owning observation for `std::weak_ptr`.
  /// @note Obvious top-level borrowed forms, move-only values, and Graph authoring
  /// handles are rejected at compile time. Function pointers and copyable callables
  /// are accepted. String views and string literals normalize to owned strings.
  /// Checks are intentionally shallow: nested container elements, user-defined
  /// fields, and callable captures remain the application's lifetime responsibility.
  /// @warning Do not call Graph authoring verbs from a derived Graph constructor;
  /// author the reusable body only in the Core-managed \ref compose override.
  /// @pre Any legacy \c Operator::start or \c Operator::stop hook must be public, unambiguous,
  /// and have the exact zero-argument \c void signature. Any advanced lifecycle member callback
  /// registered by \c setup() must be declared by \p Op itself rather than by a base or unrelated
  /// class.
  template <std::derived_from<OperatorBase> Op, typename... Args>
    requires((detail::ReplayableCapture<Args> && ...) &&
             std::constructible_from<Op, const detail::NormalizedGraphCapture<Args>&...>)
  OperatorHandle<Op> op(std::string_view name, Args&&... args) {
    if constexpr (!detail::SelectableOperatorLifecycleHooks<Op>) {
      static_assert(detail::SelectableOperatorLifecycleHooks<Op>,
                    "Operator lifecycle hooks must be public, unambiguous, and have the exact "
                    "void() signature");
    } else {
      auto prepared = detail::graph_prepare_authoring(state_);
      if (!prepared)
        detail::throw_graph_error(std::move(prepared.error()));
      using Factory = detail::OperatorFactoryModel<Op, detail::NormalizedGraphCapture<Args>...>;
      auto factory =
          std::make_shared<Factory>(detail::normalize_graph_capture(std::forward<Args>(args))...);
      auto prototype = factory->create(nullptr);
      if (!prototype) {
        prototype.error().message += "; operator=" + std::string{name};
        detail::throw_graph_error(std::move(prototype.error()));
      }
      auto registered = detail::graph_register_operator(state_,
                                                        name,
                                                        factory,
                                                        std::move(prototype.value()),
                                                        &detail::kGraphOperatorTypeToken<Op>);
      if (!registered)
        detail::throw_graph_error(std::move(registered.error()));
      return OperatorHandle<Op>{
          &detail::kGraphOperatorTypeToken<Op>, state_, registered.value().index};
    }
  }

  /// @brief Add one typed Operator declaration using the exact `op` alias.
  /// @tparam Op Concrete Operator type.
  /// @tparam Args Constructor capture argument types.
  /// @param name Nonempty unique bounded UTF-8 declaration name.
  /// @param args Constructor values copied under the same ownership precondition as `op`.
  /// @return Copyable typed authoring handle.
  /// @throws RuntimeError under the same conditions as `op`.
  template <std::derived_from<OperatorBase> Op, typename... Args>
    requires((detail::ReplayableCapture<Args> && ...) &&
             std::constructible_from<Op, const detail::NormalizedGraphCapture<Args>&...>)
  OperatorHandle<Op> add_operator(std::string_view name, Args&&... args) {
    return op<Op>(name, std::forward<Args>(args)...);
  }

  /// @brief Resolve one existing graph-local root Operator by name and exact concrete type.
  /// @tparam Op Expected concrete Operator type.
  /// @param name Exact local declaration name from this Graph's root authoring inventory.
  /// @return Copyable typed authoring handle interchangeable with the creation handle.
  /// @throws RuntimeError with `GRAPH_OPERATOR_NOT_FOUND` when no root declaration has this name.
  /// @throws RuntimeError with `GRAPH_OPERATOR_TYPE_MISMATCH` when the named root declaration has
  /// another concrete Operator type.
  /// @note This is a cold authoring-time lookup. It does not trigger lazy composition, inspect
  /// embedded Graph operators, or participate in per-activation execution.
  /// @note Whole-aggregate parameter reconstruction does not invalidate either the creation handle
  /// or a handle returned by this lookup.
  template <std::derived_from<OperatorBase> Op>
  [[nodiscard]] OperatorHandle<Op> handle(std::string_view name) {
    auto found = detail::graph_find_operator(state_, name);
    if (!found) {
      detail::throw_graph_error(std::move(found.error()));
    }
    if (!detail::graph_operator_valid(state_, *found, &detail::kGraphOperatorTypeToken<Op>)) {
      detail::throw_graph_error(Error{ErrorCode::kIncompatible, "GRAPH_OPERATOR_TYPE_MISMATCH"});
    }
    return OperatorHandle<Op>{&detail::kGraphOperatorTypeToken<Op>, state_, *found};
  }

  /// @brief Independently construct, compose, and own one reusable Graph instance.
  /// @tparam G Concrete reusable Graph type.
  /// @tparam Args Constructor capture argument types.
  /// @param instance_name Nonempty unique local embedded-instance name.
  /// @param args Constructor values normalized and forwarded once into the child object.
  /// @return Read-only typed handle exposing the child's public interface members.
  /// @throws RuntimeError if parent/child composition, construction, canonical
  /// remapping, name validation, or publication fails.
  /// @note Child composition reaches a terminal state before one atomic parent
  /// publication. The child Graph is never copied or moved and remains owned by
  /// the parent until parent retirement.
  /// @note Unlike replayed Operator and clock factories, embedding constructs the
  /// child once, so constructor arguments may transfer move-only ownership. The
  /// same shallow top-level borrow checks still apply, and Graph authoring handles
  /// remain rejected because they are composition-time identity.
  /// @warning The handle permits interface inspection only. It does not grant
  /// mutable access to the child authoring Graph.
  template <typename G, typename... Args>
    requires std::derived_from<G, Graph> && (detail::ShallowCapture<Args> && ...) &&
             (std::constructible_from<detail::NormalizedGraphCapture<Args>, Args> && ...) &&
             std::constructible_from<G, detail::NormalizedGraphCapture<Args>...>
  GraphHandle<G> embed(std::string_view instance_name, Args&&... args) {
    auto prepared = detail::graph_prepare_authoring(state_);
    if (!prepared)
      detail::throw_graph_error(std::move(prepared.error()));
    auto valid_instance = detail::graph_prepare_embed(state_, instance_name);
    if (!valid_instance)
      detail::throw_graph_error(std::move(valid_instance.error()));

    std::unique_ptr<G> child;
    try {
      child = std::make_unique<G>(detail::normalize_graph_capture(std::forward<Args>(args))...);
    } catch (const std::exception& exception) {
      detail::graph_log_factory_exception("embedded_graph", exception.what());
      auto error = detail::graph_factory_error("GRAPH_EMBED_CONSTRUCTION_FAILED");
      error.message += "; graph instance=" + std::string(instance_name);
      detail::throw_graph_error(std::move(error));
    } catch (...) {
      detail::graph_log_factory_exception("embedded_graph", "non-standard exception");
      auto error = detail::graph_factory_error("GRAPH_EMBED_CONSTRUCTION_UNKNOWN_EXCEPTION");
      error.message += "; graph instance=" + std::string(instance_name);
      detail::throw_graph_error(std::move(error));
    }
    const std::shared_ptr<detail::GraphState> child_state = child->state_;
    auto typed = detail::graph_prepare_embedded_type(
        state_, child_state, &detail::kGraphEmbeddingTypeToken<G>);
    if (!typed)
      detail::throw_graph_error(std::move(typed.error()));
    std::shared_ptr<const detail::GraphSnapshotData> child_snapshot;
    try {
      child_snapshot = detail::graph_snapshot(child_state);
    } catch (const RuntimeError& exception) {
      auto error = make_error(exception);
      if (error.message == "GRAPH_EMBED_CYCLE" || error.message == "GRAPH_COMPOSE_REENTRANT") {
        throw;
      }
      error.message += "; graph instance=" + std::string(instance_name);
      detail::throw_graph_error(std::move(error));
    }
    auto registered = detail::graph_register_embedded(state_,
                                                      instance_name,
                                                      std::unique_ptr<Graph>(std::move(child)),
                                                      child_state,
                                                      child_snapshot);
    if (!registered)
      detail::throw_graph_error(std::move(registered.error()));
    return GraphHandle<G>{state_, registered.value().index};
  }

  /// @brief Add one exact-type or exact-contract raw-POD directed queue edge.
  /// @tparam SourceT Exact admitted source payload type.
  /// @tparam TargetT Exact admitted target payload type.
  /// @param source Graph-owned output declaration.
  /// @param target Graph-owned input declaration.
  /// @param options Cold edge options copied before return.
  /// @return Typed dynamic flow handle.
  /// @throws RuntimeError for foreign/stale ports, fan-in, cycles, duplicate
  /// topology, unequal complete POD contracts, or invalid options.
  /// @note Distinct C++ endpoint types participate only when both have explicit
  /// raw-POD contracts. Scalar, enum, Tensor, and schema-root edges require the
  /// same C++ payload type.
  template <PublishedPayload SourceT, PublishedPayload TargetT>
    requires(std::same_as<SourceT, TargetT> ||
             (detail::HasChannelPodContract<SourceT> && detail::HasChannelPodContract<TargetT>))
  FlowHandle<SourceT, TargetT> add_flow(const Output<SourceT>& source, const Input<TargetT>& target,
                                        ConnectionOptions options = {}) {
    auto result = detail::graph_add_flow(state_,
                                         &source,
                                         &target,
                                         detail::operator_spec_payload_identity<SourceT>(),
                                         detail::operator_spec_payload_identity<TargetT>(),
                                         options);
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return FlowHandle<SourceT, TargetT>{
        state_, result.value().edge, result.value().source, result.value().target};
  }

  /// @brief Bind a reusable Graph public input as an alias of an internal input.
  /// @tparam T Exact admitted local payload type.
  /// @param public_input Stable public Graph input declaration.
  /// @param internal_input Stable internal Operator input declaration.
  /// @throws RuntimeError if either declaration belongs to another Graph or the
  /// payload/direction contract differs.
  /// @note This is an interface alias, not a runtime queue edge, and therefore
  /// accepts no \ref ConnectionOptions and returns no \ref FlowHandle.
  template <PublishedPayload T>
  void add_flow(const Input<T>& public_input, const Input<T>& internal_input) {
    auto result = detail::graph_bind_input_interface(
        state_, &public_input, &internal_input, detail::operator_spec_payload_identity<T>());
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
  }

  /// @brief Bind a reusable Graph public output as an alias of an internal output.
  /// @tparam T Exact admitted local payload type.
  /// @param internal_output Stable internal Operator output declaration.
  /// @param public_output Stable public Graph output declaration.
  /// @throws RuntimeError if either declaration belongs to another Graph or the
  /// payload/direction contract differs.
  /// @note This is an interface alias, not a runtime queue edge, and therefore
  /// accepts no \ref ConnectionOptions and returns no \ref FlowHandle.
  template <PublishedPayload T>
  void add_flow(const Output<T>& internal_output, const Output<T>& public_output) {
    auto result = detail::graph_bind_output_interface(
        state_, &internal_output, &public_output, detail::operator_spec_payload_identity<T>());
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
  }

  /// @brief Expose an internal input through a reusable Graph input alias.
  /// @tparam T Exact admitted local payload type.
  /// @param public_port Public Graph input declaration.
  /// @param internal_port Internal Operator input declaration.
  /// @throws RuntimeError if alias validation fails.
  template <PublishedPayload T>
  void export_input(const Input<T>& public_port, const Input<T>& internal_port) {
    (void)add_flow(public_port, internal_port);
  }

  /// @brief Expose an internal output through a reusable Graph output alias.
  /// @tparam T Exact admitted local payload type.
  /// @param public_port Public Graph output declaration.
  /// @param internal_port Internal Operator output declaration.
  /// @throws RuntimeError if alias validation fails.
  template <PublishedPayload T>
  void export_output(const Output<T>& public_port, const Output<T>& internal_port) {
    (void)add_flow(internal_port, public_port);
  }

  /// @brief Create a throwing configurator for one Graph-owned input.
  /// @tparam T Exact admitted local payload type.
  /// @param port Exact input declaration.
  /// @return Short-lived fluent configurator referencing stable authoring identity.
  /// @throws RuntimeError if \p port is foreign, stale, or structurally unknown.
  template <PublishedPayload T>
  InputPortConfigurator<T> configure(const Input<T>& port) {
    auto id = detail::graph_find_port(
        state_, &port, PortDirection::kInput, detail::operator_spec_payload_identity<T>());
    if (!id)
      detail::throw_graph_error(std::move(id.error()));
    return InputPortConfigurator<T>{state_, id.value()};
  }

  /// @brief Create a throwing configurator for one non-Tensor output.
  /// @tparam T Exact admitted non-Tensor local payload type.
  /// @param port Exact output declaration.
  /// @return Short-lived fluent configurator referencing stable authoring identity.
  /// @throws RuntimeError if \p port is foreign, stale, or structurally unknown.
  template <PublishedPayload T>
    requires(!std::same_as<T, Tensor> && !detail::kRuntimeDescriptorTensorSchemaPayload<T>)
  OutputPortConfigurator<T> configure(const Output<T>& port) {
    auto id = detail::graph_find_port(
        state_, &port, PortDirection::kOutput, detail::operator_spec_payload_identity<T>());
    if (!id)
      detail::throw_graph_error(std::move(id.error()));
    return OutputPortConfigurator<T>{state_, id.value()};
  }

  /// @brief Return the stable Graph-owned configurator for a descriptor-backed Tensor output.
  /// @tparam T Admitted descriptor-backed Tensor schema payload.
  /// @param port Exact output declaration.
  /// @return Stable configurator reference valid until Graph destruction.
  template <PublishedPayload T>
    requires detail::kRuntimeDescriptorTensorSchemaPayload<T>
  TensorOutputPortConfigurator& configure(const Output<T>& port) {
    auto id = detail::graph_find_port(
        state_, &port, PortDirection::kOutput, detail::operator_spec_payload_identity<T>());
    if (!id)
      detail::throw_graph_error(std::move(id.error()));
    auto configurator = detail::graph_tensor_configurator(state_, id.value());
    if (!configurator)
      detail::throw_graph_error(std::move(configurator.error()));
    return *configurator.value();
  }

  /// @brief Return the stable Graph-owned configurator for a Tensor output.
  /// @param port Exact mutable output declaration.
  /// @return Stable configurator reference valid until Graph destruction.
  /// @throws RuntimeError if \p port is foreign, stale, or structurally unknown.
  TensorOutputPortConfigurator& configure(Output<Tensor>& port) {
    return configure(static_cast<const Output<Tensor>&>(port));
  }

  /// @brief Return the stable Graph-owned configurator for a Tensor output.
  /// @param port Exact output declaration.
  /// @return Stable configurator reference valid until Graph destruction.
  /// @throws RuntimeError if \p port is foreign, stale, or structurally unknown.
  TensorOutputPortConfigurator& configure(const Output<Tensor>& port) {
    auto id = detail::graph_find_port(
        state_, &port, PortDirection::kOutput, detail::operator_spec_payload_identity<Tensor>());
    if (!id)
      detail::throw_graph_error(std::move(id.error()));
    auto configurator = detail::graph_tensor_configurator(state_, id.value());
    if (!configurator)
      detail::throw_graph_error(std::move(configurator.error()));
    return *configurator.value();
  }

  /// @brief Create a throwing configurator for one authored flow.
  /// @tparam SourceT Exact source payload type retained by the flow handle.
  /// @tparam TargetT Exact target payload type retained by the flow handle.
  /// @param flow Exact flow handle owned by this Graph.
  /// @return Short-lived fluent configurator referencing stable edge identity.
  /// @throws RuntimeError if \p flow is foreign or stale.
  template <PublishedPayload SourceT, PublishedPayload TargetT>
  FlowConfigurator<SourceT, TargetT> configure(FlowHandle<SourceT, TargetT> flow) {
    if (!flow.graph_valid() || flow.state_.lock() != state_) {
      detail::throw_graph_error(Error{ErrorCode::kInvalidArgument, "GRAPH_FLOW_HANDLE_INVALID"});
    }
    return FlowConfigurator<SourceT, TargetT>{state_, flow.id()};
  }

  /// @brief Try to create an input configurator without throwing.
  /// @tparam T Exact admitted local payload type.
  /// @param port Exact input declaration.
  /// @return Configurator or typed lookup/ownership error.
  template <PublishedPayload T>
  expected<InputPortConfigurator<T>, Error> try_configure(const Input<T>& port) {
    auto id = detail::graph_find_port(
        state_, &port, PortDirection::kInput, detail::operator_spec_payload_identity<T>());
    if (!id)
      return make_unexpected(std::move(id.error()));
    return InputPortConfigurator<T>{state_, id.value()};
  }

  /// @brief Try to create a non-Tensor output configurator without throwing.
  /// @tparam T Exact admitted non-Tensor local payload type.
  /// @param port Exact output declaration.
  /// @return Configurator or typed lookup/ownership error.
  template <PublishedPayload T>
    requires(!std::same_as<T, Tensor> && !detail::kRuntimeDescriptorTensorSchemaPayload<T>)
  expected<OutputPortConfigurator<T>, Error> try_configure(const Output<T>& port) {
    auto id = detail::graph_find_port(
        state_, &port, PortDirection::kOutput, detail::operator_spec_payload_identity<T>());
    if (!id)
      return make_unexpected(std::move(id.error()));
    return OutputPortConfigurator<T>{state_, id.value()};
  }

  /// @brief Try to create a descriptor-backed Tensor output configurator.
  /// @tparam T Admitted descriptor-backed Tensor schema payload.
  /// @param port Exact output declaration.
  /// @return Configurator value or typed lookup/ownership error.
  template <PublishedPayload T>
    requires detail::kRuntimeDescriptorTensorSchemaPayload<T>
  expected<TensorOutputPortConfigurator, Error> try_configure(const Output<T>& port) {
    auto id = detail::graph_find_port(
        state_, &port, PortDirection::kOutput, detail::operator_spec_payload_identity<T>());
    if (!id)
      return make_unexpected(std::move(id.error()));
    auto configurator = detail::graph_tensor_configurator(state_, id.value());
    if (!configurator)
      return make_unexpected(std::move(configurator.error()));
    return *configurator.value();
  }

  /// @brief Try to create a value Tensor output configurator without throwing.
  /// @param port Exact mutable Tensor output declaration.
  /// @return Configurator value or typed lookup/ownership error.
  /// @retval ErrorCode::kResourceExhausted if stable configurator storage cannot
  /// be allocated. This result carries no diagnostic string so the error path
  /// itself remains allocation-free.
  expected<TensorOutputPortConfigurator, Error> try_configure(Output<Tensor>& port) {
    return try_configure(static_cast<const Output<Tensor>&>(port));
  }

  /// @brief Try to create a value Tensor output configurator without throwing.
  /// @param port Exact Tensor output declaration.
  /// @return Configurator value or typed lookup/ownership error.
  /// @retval ErrorCode::kResourceExhausted if stable configurator storage cannot
  /// be allocated. This result carries no diagnostic string so the error path
  /// itself remains allocation-free.
  expected<TensorOutputPortConfigurator, Error> try_configure(const Output<Tensor>& port) {
    auto id = detail::graph_find_port(
        state_, &port, PortDirection::kOutput, detail::operator_spec_payload_identity<Tensor>());
    if (!id)
      return make_unexpected(std::move(id.error()));
    auto configurator = detail::graph_tensor_configurator(state_, id.value());
    if (!configurator)
      return make_unexpected(std::move(configurator.error()));
    return *configurator.value();
  }

  /// @brief Try to create a flow configurator without throwing.
  /// @tparam SourceT Exact source payload type retained by the flow handle.
  /// @tparam TargetT Exact target payload type retained by the flow handle.
  /// @param flow Exact flow handle owned by this Graph.
  /// @return Configurator or typed ownership/staleness error.
  template <PublishedPayload SourceT, PublishedPayload TargetT>
  expected<FlowConfigurator<SourceT, TargetT>, Error> try_configure(
      FlowHandle<SourceT, TargetT> flow) {
    if (!flow.graph_valid() || flow.state_.lock() != state_) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument, "GRAPH_FLOW_HANDLE_INVALID"});
    }
    return FlowConfigurator<SourceT, TargetT>{state_, flow.id()};
  }

  /// @brief Declare an intrinsic external-to-Graph typed topic boundary.
  /// @tparam T Exact portable scalar, Tensor, schema-root, or explicit POD payload type.
  /// @param uri Nonempty bounded `holomq://` topic URI copied before return.
  /// @param options Complete cold boundary contract copied before return.
  /// @return Typed handle exposing the stable Graph-side output declaration.
  /// @throws RuntimeError if identity, URI, options, bounds, or name validation fails.
  /// @note On first external use of a derived Graph, Core completes or terminally
  /// fails composition before constructing the requested boundary port.
  /// @note The value parameter is intentional: it defines one immutable authoring snapshot while
  /// Core deep-copies every view-bearing field before this call returns.
  /// @note Ordinary typed validation rejection consumes the next mixed external
  /// declaration ordinal. An exception before publication restores that ordinal.
  template <detail::ExternalTopicPayload T>
    requires(!detail::kRuntimeDescriptorTensorSchemaPayload<T>)
  ImportedTopic<T> import_topic(
      std::string_view uri,
      TopicOptions options = {}) {  // NOLINT(performance-unnecessary-value-param)
    auto prepared = detail::graph_prepare_authoring(state_);
    if (!prepared)
      detail::throw_graph_error(std::move(prepared.error()));
    auto port = std::make_unique<Output<T>>("external");
    auto result = detail::graph_register_import(state_,
                                                uri,
                                                options,
                                                std::unique_ptr<OutputBase>(std::move(port)),
                                                detail::operator_spec_payload_identity<T>(),
                                                detail::serialized_payload_capacity_bytes<T>());
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return ImportedTopic<T>{state_, result.value().value};
  }

  /// @brief Declare a descriptor-backed schema external-to-Graph topic boundary.
  /// @tparam T Owning singular or multi-Tensor descriptor-backed schema type.
  /// @param uri Nonempty bounded `holomq://` topic URI copied before return.
  /// @param options Common cold boundary contract copied before return.
  /// @param tensors Complete typed Tensor field inventory copied before return.
  /// @return Typed handle exposing the stable Graph-side output declaration.
  /// @throws RuntimeError if identity, URI, field inventory, bounds, or name validation fails.
  template <detail::ExternalTopicPayload T>
    requires detail::kRuntimeDescriptorTensorSchemaPayload<T>
  ImportedTopic<T> import_topic(
      std::string_view uri,
      TopicOptions options,  // NOLINT(performance-unnecessary-value-param)
      DescriptorTensorTopicContract<T> tensors) {
    auto prepared = detail::graph_prepare_authoring(state_);
    if (!prepared)
      detail::throw_graph_error(std::move(prepared.error()));
    auto port = std::make_unique<Output<T>>("external");
    auto result = detail::graph_register_import(
        state_,
        uri,
        std::move(options),
        std::unique_ptr<OutputBase>(std::move(port)),
        detail::operator_spec_payload_identity<T>(),
        detail::external_topic_payload_capacity_bytes<T>(),
        std::move(tensors.fields_),
        static_cast<std::uint16_t>(detail::descriptor_tensor_schema_tensor_count_v<T>));
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return ImportedTopic<T>{state_, result.value().value};
  }

  /// @brief Declare an intrinsic Graph-to-external typed topic boundary.
  /// @tparam T Exact portable scalar, Tensor, schema-root, or explicit POD payload type.
  /// @param uri Nonempty bounded `holomq://` topic URI copied before return.
  /// @param options Complete cold boundary contract copied before return.
  /// @return Typed handle exposing the stable Graph-side input declaration.
  /// @throws RuntimeError if identity, URI, options, bounds, or name validation fails.
  /// @note On first external use of a derived Graph, Core completes or terminally
  /// fails composition before constructing the requested boundary port.
  /// @note The value parameter is intentional: it defines one immutable authoring snapshot while
  /// Core deep-copies every view-bearing field before this call returns.
  /// @note Ordinary typed validation rejection consumes the next mixed external
  /// declaration ordinal. An exception before publication restores that ordinal.
  template <detail::ExternalTopicPayload T>
    requires(!detail::kRuntimeDescriptorTensorSchemaPayload<T>)
  ExportedTopic<T> export_topic(
      std::string_view uri,
      TopicOptions options = {}) {  // NOLINT(performance-unnecessary-value-param)
    auto prepared = detail::graph_prepare_authoring(state_);
    if (!prepared)
      detail::throw_graph_error(std::move(prepared.error()));
    auto port = std::make_unique<Input<T>>("external");
    auto result = detail::graph_register_export(state_,
                                                uri,
                                                options,
                                                std::unique_ptr<InputBase>(std::move(port)),
                                                detail::operator_spec_payload_identity<T>(),
                                                detail::serialized_payload_capacity_bytes<T>());
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return ExportedTopic<T>{state_, result.value().value};
  }

  /// @brief Declare a descriptor-backed schema Graph-to-external topic boundary.
  /// @tparam T Owning singular or multi-Tensor descriptor-backed schema type.
  /// @param uri Nonempty bounded `holomq://` topic URI copied before return.
  /// @param options Common cold boundary contract copied before return.
  /// @param tensors Complete typed Tensor field inventory copied before return.
  /// @return Typed handle exposing the stable Graph-side input declaration.
  /// @throws RuntimeError if identity, URI, field inventory, bounds, or name validation fails.
  template <detail::ExternalTopicPayload T>
    requires detail::kRuntimeDescriptorTensorSchemaPayload<T>
  ExportedTopic<T> export_topic(
      std::string_view uri,
      TopicOptions options,  // NOLINT(performance-unnecessary-value-param)
      DescriptorTensorTopicContract<T> tensors) {
    auto prepared = detail::graph_prepare_authoring(state_);
    if (!prepared)
      detail::throw_graph_error(std::move(prepared.error()));
    auto port = std::make_unique<Input<T>>("external");
    auto result = detail::graph_register_export(
        state_,
        uri,
        std::move(options),
        std::unique_ptr<InputBase>(std::move(port)),
        detail::operator_spec_payload_identity<T>(),
        detail::external_topic_payload_capacity_bytes<T>(),
        std::move(tensors.fields_),
        static_cast<std::uint16_t>(detail::descriptor_tensor_schema_tensor_count_v<T>));
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return ExportedTopic<T>{state_, result.value().value};
  }

  /// @brief Add one uniquely named backend-neutral policy region.
  /// @param name Nonempty unique bounded UTF-8 name copied before return.
  /// @return Fluent region handle.
  /// @throws RuntimeError if \p name is invalid or duplicated.
  RegionHandle region(std::string_view name);

  /// @brief Add one uniquely named developer-preview manual execution partition.
  /// @param name Nonempty unique bounded UTF-8 name copied before return.
  /// @return Fluent partition handle accepting direct Operators and complete embedded instances.
  /// @throws RuntimeError if \p name is invalid or duplicated.
  /// @note Without \ref auto_partition_remaining or
  /// \ref PartitionHandle::include_all, compile requires all partitions to be
  /// nonempty and to cover every executable Operator exactly once.
  /// @note Manual partitions and \ref region declarations are orthogonal;
  /// regions may overlap partition boundaries.
  /// @note Direct root Operators and selected child subtrees may anchor automatic remainder.
  /// Selected subtrees include every descendant Operator and must not declare partitions
  /// internally. A root \ref PartitionHandle::include_all covers the complete application.
  PartitionHandle partition(std::string_view name);

  /// @brief Automatically partition Operators omitted from exact authored partitions.
  /// @throws RuntimeError if a partition already uses \ref PartitionHandle::include_all.
  /// @note The request is idempotent and may precede or follow exact partition authoring.
  /// Compilation rejects a request with no nonempty exact partition anchor.
  void auto_partition_remaining();

  /// @brief Create a lifecycle ordering/permission authoring builder.
  /// @return Short-lived builder tied to this Graph.
  GraphLifecycleBuilder lifecycle();

  /// @brief Freeze and return one immutable deep-owning authoring snapshot.
  /// @return Snapshot that remains valid across Graph mutation and destruction.
  /// @throws RuntimeError if lazy composition failed or observation is reentrant.
  [[nodiscard]] GraphView view();

  /// @brief Add one Graph-owned replayable clock declaration.
  /// @tparam ClockT Concrete Clock type.
  /// @tparam Args Constructor capture argument types.
  /// @param name Nonempty unique bounded UTF-8 clock name.
  /// @param args Constructor values copied under the same ownership precondition as `op`.
  /// @return Owned named cold clock key.
  /// @throws RuntimeError if construction, naming, or registration fails.
  /// @note On first external use of a derived Graph, Core completes or terminally
  /// fails composition before constructing the requested clock factory.
  /// @note Registration constructs one Clock for validation and discards it. A
  /// later compiler or RunSession replays the retained factory to create its own
  /// ordinary Clock; only `bind_shared_clock()` retains a live instance.
  /// @pre Every normalized capture is copy-constructible and follows its ordinary
  /// C++ copy semantics. Obvious top-level borrowed forms and Graph authoring handles
  /// are rejected. Checks are shallow; nested borrows remain the application's lifetime
  /// responsibility. Construction must be repeatable, and constructor side effects are
  /// not authoritative runtime state.
  template <std::derived_from<Clock> ClockT, typename... Args>
    requires((detail::ReplayableCapture<Args> && ...) &&
             std::constructible_from<ClockT, const detail::NormalizedGraphCapture<Args>&...>)
  ClockKey add_clock(std::string_view name, Args&&... args) {
    auto prepared = detail::graph_prepare_authoring(state_);
    if (!prepared)
      detail::throw_graph_error(std::move(prepared.error()));
    using Factory = detail::ClockFactoryModel<ClockT, detail::NormalizedGraphCapture<Args>...>;
    auto factory =
        std::make_shared<Factory>(detail::normalize_graph_capture(std::forward<Args>(args))...);
    auto result = detail::graph_add_clock(state_, name, std::move(factory));
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return std::move(result.value());
  }

  /// @brief Bind one explicitly shared live clock into cold Graph ownership.
  /// @param name Nonempty unique bounded UTF-8 clock name.
  /// @param clock Non-null shared clock retained by Graph state.
  /// @return Owned named cold clock key.
  /// @throws RuntimeError if naming or pointer validation fails.
  ClockKey bind_shared_clock(std::string_view name, std::shared_ptr<Clock> clock);

  /// @brief Declare one explicit clock dependency for a reusable Graph.
  /// @param name Nonempty unique bounded UTF-8 clock import name.
  /// @return Owned named import key that contracts and conversions may reference.
  /// @throws RuntimeError if naming or registration fails.
  /// @note An import creates no runtime clock. Every embedded instance must be
  /// bound explicitly by its direct parent before compilation.
  ClockKey import_clock(std::string_view name);

  /// @brief Bind one embedded Graph clock import to one exact parent clock.
  /// @tparam G Concrete reusable Graph type.
  /// @param child Exact live child instance owned by this Graph.
  /// @param child_import Named clock import declared by that child root.
  /// @param parent_clock Named clock declaration or import in this Graph.
  /// @return This Graph.
  /// @throws RuntimeError for a foreign/stale child handle, a non-import child
  /// key, an unknown parent key, or a duplicate/conflicting binding.
  /// @note Resolution follows only these instance-qualified records. It never
  /// searches an ancestor scope by clock name.
  template <typename G>
    requires std::derived_from<G, Graph>
  // ClockKey is an owning cold-authoring value. Taking both keys by value preserves efficient
  // rvalue transfer into the type-erased implementation while giving lvalue callers one
  // explicit ownership copy at this public boundary.
  // NOLINTNEXTLINE(performance-unnecessary-value-param)
  Graph& bind_clock(const GraphHandle<G>& child, ClockKey child_import, ClockKey parent_clock) {
    auto child_owner = child.state_.lock();
    if (!child_owner || child_owner.get() != state_.get()) {
      detail::throw_graph_error(
          Error{ErrorCode::kInvalidArgument, "GRAPH_CLOCK_BINDING_HANDLE_INVALID"});
    }
    auto result = detail::graph_bind_clock_import(
        state_, child.index_, std::move(child_import), std::move(parent_clock));
    if (!result)
      detail::throw_graph_error(std::move(result.error()));
    return *this;
  }

  /// @brief Select one previously declared named clock as Graph default.
  /// @param key Existing non-default named key.
  /// @return This Graph.
  /// @throws RuntimeError if \p key is default, empty, or not declared.
  Graph& set_default_clock(ClockKey key);

  /// @brief Return the selected default clock key.
  /// @return Named key, or the distinct default/omitted key when none was selected.
  /// @throws RuntimeError if lazy composition failed or observation is reentrant.
  [[nodiscard]] ClockKey default_clock();

  /// @brief Add one immutable directed affine clock conversion.
  /// @param from Existing named source clock.
  /// @param to Existing distinct named destination clock.
  /// @param conversion Positive exact rate and signed nanosecond offset.
  /// @return This Graph.
  /// @throws RuntimeError for missing/self clocks, an invalid scale, a
  /// conflicting conversion for the same directed pair, or a directed cycle.
  /// @note Core reduces the positive ratio to lowest terms. Repeating the same
  /// pair with an equivalent canonical ratio and equal offset is idempotent.
  Graph& add_clock_conversion(ClockKey from, ClockKey to, AffineClockConversion conversion);

 protected:
  /// @brief Author a derived Graph body during Core-managed lazy composition.
  ///
  /// Core invokes this override at most once after derived construction. Normal
  /// authoring verbs are valid inside the active transaction. Topology observers
  /// and other same-Graph paths that re-enter the composition gate fail with
  /// `GRAPH_COMPOSE_REENTRANT`; a terminal failure is retained and replayed.
  /// @warning Do not author from the derived constructor and do not call this
  /// virtual function directly. Constructor misuse and direct self-recursion
  /// have no promised runtime diagnosis.
  /// @warning The complete most-derived Graph must remain alive for the entire
  /// callback; concurrent destruction is outside the Graph lifetime contract.
  virtual void compose() {}

 private:
  template <std::derived_from<OperatorBase> Op>
  OperatorHandle<Op> add_operator_with_initial_params(std::string_view name,
                                                      const typename Op::Params& params) {
    if constexpr (!detail::SelectableOperatorLifecycleHooks<Op>) {
      static_assert(detail::SelectableOperatorLifecycleHooks<Op>,
                    "Operator lifecycle hooks must be public, unambiguous, and have the exact "
                    "void() signature");
    } else {
      auto prepared = detail::graph_prepare_authoring(state_);
      if (!prepared)
        detail::throw_graph_error(std::move(prepared.error()));
      using Factory = detail::OperatorFactoryModel<Op>;
      auto factory = std::make_shared<Factory>();
      auto prototype = factory->create(nullptr);
      if (!prototype) {
        prototype.error().message += "; operator=" + std::string{name};
        detail::throw_graph_error(std::move(prototype.error()));
      }
      const detail::InitialOperatorParameters initial_params{
          .value = &params,
          .local_type_token = param_type_descriptor<typename Op::Params>().local_type_token,
          .schema = param_schema_descriptor<typename Op::Params>(),
      };
      auto registered = detail::graph_register_operator(state_,
                                                        name,
                                                        factory,
                                                        std::move(prototype.value()),
                                                        &detail::kGraphOperatorTypeToken<Op>,
                                                        &initial_params);
      if (!registered)
        detail::throw_graph_error(std::move(registered.error()));
      return OperatorHandle<Op>{
          &detail::kGraphOperatorTypeToken<Op>, state_, registered.value().index};
    }
  }

  std::shared_ptr<detail::GraphState> state_;
  friend class detail::GraphCompilerAccess;
  friend class detail::GraphOperatorRegistrationAccess;
  friend class detail::GraphPayloadRegistrationAccess;
  friend class detail::GraphState;
  friend class detail::ParameterConfigBindingAccess;
  friend class detail::ParameterInputAccess;
};

namespace detail {

class GraphOperatorRegistrationAccess {
 public:
  template <std::derived_from<OperatorBase> Op>
  static OperatorHandle<Op> add_with_params(Graph& graph, std::string_view name,
                                            const typename Op::Params& params) {
    return graph.add_operator_with_initial_params<Op>(name, params);
  }
};

/// @brief Route type-erased registry declarations through Graph-owned validation.
class GraphPayloadRegistrationAccess {
 public:
  static void add_flow(Graph& graph, const OutputBase& source, const InputBase& target,
                       const PayloadIdentityRecord& source_identity,
                       const PayloadIdentityRecord& target_identity, ConnectionOptions options) {
    auto result = graph_add_flow(
        graph.state_, &source, &target, source_identity, target_identity, std::move(options));
    if (!result) {
      throw_graph_error(std::move(result.error()));
    }
  }

  static void export_input(Graph& graph, const InputBase& public_port,
                           const InputBase& internal_port, PayloadIdentityRecord identity) {
    auto result =
        graph_bind_input_interface(graph.state_, &public_port, &internal_port, std::move(identity));
    if (!result) {
      throw_graph_error(std::move(result.error()));
    }
  }

  static void export_output(Graph& graph, const OutputBase& public_port,
                            const OutputBase& internal_port, PayloadIdentityRecord identity) {
    auto result = graph_bind_output_interface(
        graph.state_, &internal_port, &public_port, std::move(identity));
    if (!result) {
      throw_graph_error(std::move(result.error()));
    }
  }
};

}  // namespace detail

inline const std::shared_ptr<detail::GraphState>& detail::ParameterConfigBindingAccess::state(
    Graph& graph) noexcept {
  return graph.state_;
}

}  // namespace holoscan
