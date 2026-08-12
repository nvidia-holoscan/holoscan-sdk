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
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/payload_identity.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/profile.hpp>
#include <holoscan/core/readiness_source.hpp>
#include <holoscan/core/tensor_field.hpp>
#include <holoscan/time/clock.hpp>
#include <holoscan/time/clock_registry.hpp>

namespace holoscan {

class Graph;
class HOLOSCAN_CORE_EXPORT TensorOutputPortConfigurator;
class GraphInstancePath;

namespace detail {
class GraphState;
class GraphCompilerAccess;
class ParameterConfigBindingAccess;
struct GraphSnapshotData;
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
/// This value is never a compiler-assigned \c PortId. EA1-009 remaps it to a
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
struct LogicalTensorRequirementsRecord {
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
/// Setup defaults and Graph overrides occupy separate instances so EA1-009 can
/// resolve precedence without reconstructing author intent.
struct PortConfigurationRecord {
  std::optional<std::uint32_t> queue_depth;                 ///< Input history depth when present.
  std::optional<std::uint32_t> max_emits_per_compute;       ///< Output burst bound when present.
  std::optional<std::uint32_t> late_joiner_history_depth;   ///< Output retained history.
  std::optional<std::string> diagnostic_alias;              ///< Owned diagnostic-only alias.
  std::optional<TensorPortLayout> tensor_layout;            ///< Logical Tensor facts when authored.
  std::optional<MemoryConversionPolicy> memory_conversion;  ///< Port conversion request.
  std::optional<TensorAllocationBounds> tensor_allocation;  ///< Direct Tensor allocation bound.
  std::optional<PayloadOptionsRecord> payload_options;      ///< Deep-owned producer options.
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
};

/// @brief Describe one copied port declaration.
struct PortRecord {
  GraphPortId id{};  ///< Snapshot-local authoring identity, never a plan \c PortId.
  VertexId owner{};  ///< Owning logical vertex.
  PortOwnerKind owner_kind{PortOwnerKind::kOperator};  ///< Authoring owner classification.
  PortDirection direction{PortDirection::kInput};      ///< Input or output direction.
  std::string name;                                    ///< Owned declaration-local name.
  PayloadIdentityRecord payload_identity;              ///< Complete pointer-free semantic identity.
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
  std::optional<LogicalTensorRequirementsRecord> tensor;  ///< Deep-owned Tensor contract.
  EndpointBounds bounds{};  ///< Direction-specific external peer resource bounds.
};

/// @brief Classify the wake requirement implied by one readiness source declaration.
enum class ReadinessWakeRequirement : std::uint8_t {
  kDeploymentSelected = 0,  ///< Compiler/runtime policy may select an admitted wake mechanism.
  kBusyPollRequired = 1,    ///< The source explicitly requires busy polling.
};

/// @brief Describe one logical readiness declaration after Graph composition.
///
/// This record is pointer-free authoring provenance. EA1-006 assigns no physical
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
  bool referenced{};  ///< Reserved for later Contract lowering; false in EA1-006 snapshots.
};

/// @brief Classify a backend-neutral region constraint.
enum class RegionConstraintKind : std::uint8_t {
  kStaticTopology,             ///< Require topology that does not change during execution.
  kBoundedMemory,              ///< Require finite admitted memory use.
  kDeviceCompatibleOperators,  ///< Require operators compatible with selected device lowering.
  kLatencyBudget,              ///< Carry a positive end-to-end region latency budget.
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

/// @brief Select one backend-neutral optimization hint.
enum class OptimizationHint : std::uint8_t {
  kCudaGraphFusion = 0,  ///< Request analysis for CUDA Graph fusion; never select a backend.
};

/// @brief Describe one canonical Graph region declaration.
struct RegionRecord {
  RegionId id{};                              ///< Nonzero snapshot-local region identity.
  GraphId graph{};                            ///< Exact owning Graph instance.
  std::string name;                           ///< Owned unique region name.
  std::vector<VertexId> members;              ///< Canonical logical operator membership.
  std::vector<GraphId> graph_members;         ///< Canonical embedded Graph membership.
  std::vector<RegionConstraint> constraints;  ///< Unique backend-neutral constraints.
  std::vector<OptimizationHint> hints;        ///< Unique optimization requests.
  std::optional<Profile> profile;             ///< Optional region-level profile request.
};

/// @brief Describe one developer-preview manual execution partition.
///
/// Partition declarations are distinct from overlapping \ref RegionRecord
/// scopes. Each record owns its name and direct Operator membership. The
/// compiler later requires authored partitions to form a nonempty, disjoint,
/// exact cover before lowering them into execution-plan partitions.
struct PartitionRecord {
  PartitionId id{};               ///< Nonzero snapshot-local partition identity.
  GraphId graph{};                ///< Exact owning Graph instance.
  std::string name;               ///< Owned unique partition name in the Graph scope.
  std::vector<VertexId> members;  ///< Canonical direct-Operator membership.
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

/// @brief Log cold user-factory exception detail without placing logger macros in public headers.
/// @param context Stable bounded construction context token.
/// @param detail Arbitrary exception text used only for cold diagnostics.
HOLOSCAN_CORE_EXPORT void graph_log_factory_exception(std::string_view context,
                                                      std::string_view detail) noexcept;

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
  [[nodiscard]] virtual expected<std::unique_ptr<OperatorBase>, Error> create() const = 0;
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
struct IsUniquePtr : std::false_type {};
template <typename T, typename D>
struct IsUniquePtr<std::unique_ptr<T, D>> : std::true_type {};
template <typename T>
struct IsSharedPtr : std::false_type {};
template <typename T>
struct IsSharedPtr<std::shared_ptr<T>> : std::true_type {};
template <typename T>
struct IsWeakPtr : std::false_type {};
template <typename T>
struct IsWeakPtr<std::weak_ptr<T>> : std::true_type {};
template <typename T>
struct IsInitializerList : std::false_type {};
template <typename T>
struct IsInitializerList<std::initializer_list<T>> : std::true_type {};
template <typename T>
struct IsStdFunction : std::false_type {};
template <typename Result, typename... Args>
struct IsStdFunction<std::function<Result(Args...)>> : std::true_type {};
template <typename T>
struct IsGraphHandle : std::false_type {};
template <typename G>
struct IsGraphHandle<GraphHandle<G>> : std::true_type {};
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

template <typename T>
concept GraphOwnedCapture =
    !std::is_pointer_v<std::remove_cvref_t<T>> && !IsSpan<std::remove_cvref_t<T>>::value &&
    !IsReferenceWrapper<std::remove_cvref_t<T>>::value &&
    !IsUniquePtr<std::remove_cvref_t<T>>::value && !IsSharedPtr<std::remove_cvref_t<T>>::value &&
    !IsWeakPtr<std::remove_cvref_t<T>>::value &&
    !IsInitializerList<std::remove_cvref_t<T>>::value &&
    !IsStdFunction<std::remove_cvref_t<T>>::value &&
    !IsGraphHandle<std::remove_cvref_t<T>>::value &&
    (!std::ranges::view<std::remove_cvref_t<T>> || kStringCapture<T>) &&
    (!(std::ranges::range<std::remove_cvref_t<T>> &&
       std::ranges::borrowed_range<std::remove_cvref_t<T>>) ||
     kStringCapture<T>) &&
    std::copy_constructible<NormalizedGraphCapture<T>>;

template <typename Op, typename... Captures>
class OperatorFactoryModel final : public OperatorFactoryBase {
 public:
  explicit OperatorFactoryModel(Captures... captures) : captures_(std::move(captures)...) {}

  [[nodiscard]] expected<std::unique_ptr<OperatorBase>, Error> create() const override {
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
      graph_log_factory_exception("operator", exception.what());
      return make_unexpected(Error{ErrorCode::kFailure, "GRAPH_OPERATOR_CONSTRUCTION_FAILED"});
    } catch (...) {
      graph_log_factory_exception("operator", "non-standard exception");
      return make_unexpected(Error{ErrorCode::kFailure, "GRAPH_OPERATOR_CONSTRUCTION_FAILED"});
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
  [[nodiscard]] virtual expected<std::unique_ptr<Clock>, Error> create() const = 0;
  [[nodiscard]] virtual std::shared_ptr<ClockFactoryBase> clone() const = 0;
};

template <typename ClockT, typename... Captures>
class ClockFactoryModel final : public ClockFactoryBase {
 public:
  explicit ClockFactoryModel(Captures... captures) : captures_(std::move(captures)...) {}
  [[nodiscard]] expected<std::unique_ptr<Clock>, Error> create() const override {
    try {
      auto clock = std::apply(
          [](const auto&... values) { return std::make_unique<ClockT>(values...); }, captures_);
      return std::unique_ptr<Clock>(std::move(clock));
    } catch (const std::exception& exception) {
      graph_log_factory_exception("clock", exception.what());
      return make_unexpected(Error{ErrorCode::kFailure, "GRAPH_CLOCK_CONSTRUCTION_FAILED"});
    } catch (...) {
      graph_log_factory_exception("clock", "non-standard exception");
      return make_unexpected(Error{ErrorCode::kFailure, "GRAPH_CLOCK_CONSTRUCTION_FAILED"});
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

struct FlowRegistration {
  EdgeId edge{};
  GraphPortId source{};
  GraphPortId target{};
};

struct EmbeddedGraphRegistration {
  std::size_t index{};
};

HOLOSCAN_CORE_EXPORT expected<OperatorRegistration, Error> graph_register_operator(
    const std::shared_ptr<GraphState>& state, std::string_view name,
    std::shared_ptr<OperatorFactoryBase> factory, std::unique_ptr<OperatorBase> prototype,
    const void* operator_type_token);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_prepare_authoring(
    const std::shared_ptr<GraphState>& state);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_prepare_embed(
    const std::shared_ptr<GraphState>& state, std::string_view instance_name);
HOLOSCAN_CORE_EXPORT const OperatorBase* graph_operator_prototype(
    const std::shared_ptr<GraphState>& state, std::size_t index,
    const void* operator_type_token) noexcept;
HOLOSCAN_CORE_EXPORT bool graph_operator_valid(const std::shared_ptr<GraphState>& state,
                                               std::size_t index,
                                               const void* operator_type_token) noexcept;
HOLOSCAN_CORE_EXPORT std::string_view graph_operator_name(const std::shared_ptr<GraphState>& state,
                                                          std::size_t index) noexcept;
HOLOSCAN_CORE_EXPORT OperatorRef graph_operator_ref(const std::shared_ptr<GraphState>& state,
                                                    std::size_t index) noexcept;
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_set_operator_params(
    const std::shared_ptr<GraphState>& state, std::size_t index, const void* params,
    const void* parameter_type_token);

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
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_input_tensor(
    const std::shared_ptr<GraphState>&, GraphPortId, TensorPortLayout);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_input_memory(
    const std::shared_ptr<GraphState>&, GraphPortId, MemoryConversionPolicy);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_port_alias(
    const std::shared_ptr<GraphState>&, GraphPortId, std::string_view);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_output_emits(
    const std::shared_ptr<GraphState>&, GraphPortId, std::uint32_t);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_output_history(
    const std::shared_ptr<GraphState>&, GraphPortId, std::uint32_t);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_output_tensor(
    const std::shared_ptr<GraphState>&, GraphPortId, TensorPortLayout);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_output_payload(
    const std::shared_ptr<GraphState>&, GraphPortId, PayloadOptions);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_configure_tensor_allocation(
    const std::shared_ptr<GraphState>&, GraphPortId, TensorAllocationBounds);
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
    PayloadIdentityRecord, std::optional<SchemaPackagePlanRecord>,
    std::uint64_t serialized_payload_capacity_bytes);
HOLOSCAN_CORE_EXPORT expected<GraphPortId, Error> graph_register_export(
    const std::shared_ptr<GraphState>&, std::string_view, TopicOptions, std::unique_ptr<InputBase>,
    PayloadIdentityRecord, std::optional<SchemaPackagePlanRecord>,
    std::uint64_t serialized_payload_capacity_bytes);

HOLOSCAN_CORE_EXPORT expected<RegionId, Error> graph_add_region(const std::shared_ptr<GraphState>&,
                                                                std::string_view);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_region_add_operator(
    const std::shared_ptr<GraphState>&, RegionId, std::size_t);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_region_add_embedded_graph(
    const std::shared_ptr<GraphState>&, RegionId, std::size_t);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_region_add_constraint(
    const std::shared_ptr<GraphState>&, RegionId, RegionConstraint);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_region_add_hint(const std::shared_ptr<GraphState>&,
                                                                 RegionId, OptimizationHint);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_region_set_profile(
    const std::shared_ptr<GraphState>&, RegionId, Profile);

HOLOSCAN_CORE_EXPORT expected<PartitionId, Error> graph_add_partition(
    const std::shared_ptr<GraphState>&, std::string_view);
HOLOSCAN_CORE_EXPORT expected<void, Error> graph_partition_add_operator(
    const std::shared_ptr<GraphState>&, PartitionId, std::size_t);

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
    auto result = detail::graph_set_operator_params(
        state, index_, &params, param_type_descriptor<Params>().local_type_token);
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
    return state &&
           detail::graph_operator_valid(state, index_, &detail::kGraphOperatorTypeToken<Op>);
  }

  /// @brief Return whether this handle names a live Graph declaration.
  [[nodiscard]] explicit operator bool() const noexcept { return graph_valid(); }

  /// @brief Return the compiled operator reference when one exists.
  /// @return Invalid zero reference in EA1-006; EA1-009 binds plan identity.
  [[nodiscard]] OperatorRef ref() const noexcept {
    auto state = state_.lock();
    return state ? detail::graph_operator_ref(state, index_) : OperatorRef{};
  }

  /// @brief Return the compiled operator identity when one exists.
  /// @return Zero in EA1-006 because authoring handles never fabricate plan IDs.
  [[nodiscard]] OperatorInstanceId id() const noexcept { return ref().id; }

  /// @brief Return the immutable compiled-plan generation when one exists.
  /// @return Zero until EA1-009 freezes this declaration into a plan.
  [[nodiscard]] PlanGenerationId plan_generation() const noexcept { return ref().plan_generation; }

  /// @brief Return whether this handle is bound to an immutable compiled plan.
  /// @return Always false for the EA1-006 authoring-only handle.
  [[nodiscard]] bool plan_bound() const noexcept { return ref().plan_bound(); }

 private:
  OperatorHandle(Graph* graph, std::weak_ptr<detail::GraphState> state, std::size_t index) noexcept
      : graph_(graph), state_(std::move(state)), index_(index) {}
  Graph* graph_{};
  std::weak_ptr<detail::GraphState> state_;
  std::size_t index_{};
  friend class Graph;
  friend class RegionHandle;
  friend class PartitionHandle;
  friend class GraphLifecycleBuilder;
  friend class detail::ParameterConfigBindingAccess;
};

namespace detail {

/// @brief Expose call-scoped authoring ownership to optional Parameter tooling.
class ParameterConfigBindingAccess {
 public:
  template <std::derived_from<OperatorBase> Op>
  [[nodiscard]] static Graph* graph(const OperatorHandle<Op>& handle) noexcept {
    return handle.state_.expired() ? nullptr : handle.graph_;
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
  /// @brief Override the finite input history depth.
  /// @param depth Positive number of samples retained for this logical input.
  /// @return This configurator for fluent authoring.
  /// @throws RuntimeError if \p depth is zero or the source Graph is unavailable.
  InputPortConfigurator& queue_depth(std::uint32_t depth) {
    apply(detail::graph_configure_input_queue(state(), port_, depth));
    return *this;
  }

  /// @brief Override logical Tensor requirements for this input.
  /// @param layout Backend-neutral logical Tensor facts.
  /// @return This configurator for fluent authoring.
  /// @throws RuntimeError if the layout contains an invalid closed value, the
  /// source Graph is unavailable, or the port is neither a direct Tensor nor an
  /// Operator-setup-declared Tensor carrier.
  /// @note Tensor facts may describe a typed carrier port; this method does not
  /// grant direct Tensor allocation capability.
  InputPortConfigurator& expects_tensor(TensorPortLayout layout) {
    apply(detail::graph_configure_input_tensor(state(), port_, layout));
    return *this;
  }

  /// @brief Override the input Tensor memory-conversion request.
  /// @param policy Closed conversion policy.
  /// @return This configurator for fluent authoring.
  /// @throws RuntimeError if \p policy is invalid or this is not a Tensor port.
  InputPortConfigurator& memory_conversion(MemoryConversionPolicy policy) {
    apply(detail::graph_configure_input_memory(state(), port_, policy));
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

  /// @brief Override logical Tensor production facts for this output.
  /// @param layout Backend-neutral logical Tensor facts.
  /// @return This configurator for fluent authoring.
  /// @throws RuntimeError if the layout contains an invalid closed value, the
  /// source Graph is unavailable, or the port is neither a direct Tensor nor an
  /// Operator-setup-declared Tensor carrier.
  /// @note Tensor facts may describe a typed carrier output; this method does
  /// not add the direct-Tensor allocation configurator.
  OutputPortConfigurator& produces_tensor(TensorPortLayout layout) {
    apply(detail::graph_configure_output_tensor(state(), port_, layout));
    return *this;
  }

  /// @brief Override producer-side payload materialization options.
  /// @param options Schema and interface Tensor policy copied before return.
  /// @return This configurator for fluent authoring.
  /// @throws RuntimeError if \p options is invalid for the payload type.
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

  /// @brief Override logical Tensor production facts.
  /// @param layout Backend-neutral logical Tensor facts.
  /// @return This stable Graph-owned configurator.
  /// @throws RuntimeError if the layout contains an invalid closed value or the
  /// source Graph is unavailable.
  TensorOutputPortConfigurator& produces_tensor(TensorPortLayout layout);

  /// @brief Override producer-side Tensor materialization options.
  /// @param options Schema and interface Tensor policy copied before return.
  /// @return This stable Graph-owned configurator.
  /// @throws RuntimeError if policy/cap validation or effective-boundary
  /// normalization fails, or the source Graph is unavailable. A rejected update
  /// leaves the port and all existing edges unchanged.
  TensorOutputPortConfigurator& payload_options(PayloadOptions options);

  /// @brief Assign an owned diagnostic alias without changing port identity.
  /// @param alias Nonempty bounded UTF-8 alias copied before return.
  /// @return This stable Graph-owned configurator.
  /// @throws RuntimeError if \p alias is invalid or the source Graph is unavailable.
  TensorOutputPortConfigurator& diagnostic_alias(std::string_view alias);

  /// @brief Tighten the allocation bounds declared by `OperatorSpec`.
  /// @param bounds Positive bounds no wider than the declaration capability.
  /// @return This stable Graph-owned configurator.
  /// @throws RuntimeError if allocation was not declared or \p bounds widens it.
  TensorOutputPortConfigurator& tensor_allocation(TensorAllocationBounds bounds);

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

  /// @brief Add one unique backend-neutral optimization hint.
  /// @param hint Closed hint not already present.
  /// @return This handle for fluent authoring.
  /// @throws RuntimeError if \p hint is invalid or duplicated.
  RegionHandle& optimize(OptimizationHint hint);

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
/// originating Graph. It accepts only direct Operator declarations from that
/// Graph; embedded-Graph membership is intentionally outside the EA1 surface.
/// A compiler validates that the complete authored partition set is nonempty,
/// disjoint, and covers every executable Operator exactly once.
///
/// The handle remains usable only while the originating Graph and declaration
/// remain live. Mutation is atomic on failure but is not thread-safe with any
/// other read or mutation of the same Graph.
class HOLOSCAN_CORE_EXPORT PartitionHandle {
 public:
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
  /// @pre Each arbitrary user-defined captured value must establish independent
  /// owning replayable state when copied; hidden borrowed fields and lambda
  /// captures cannot be inspected mechanically by C++20 concepts.
  /// @note Known top-level borrowed, pointer, smart-pointer, type-erased callable,
  /// and noncopyable capture forms are rejected at compile time. String views and
  /// string literals normalize to owned strings.
  /// @warning Do not call Graph authoring verbs from a derived Graph constructor;
  /// author the reusable body only in the Core-managed \ref compose override.
  /// @pre Any legacy \c Operator::start or \c Operator::stop hook must be public, unambiguous,
  /// and have the exact zero-argument \c void signature. Any advanced lifecycle member callback
  /// registered by \c setup() must be declared by \p Op itself rather than by a base or unrelated
  /// class.
  template <std::derived_from<OperatorBase> Op, typename... Args>
    requires((detail::GraphOwnedCapture<Args> && ...) &&
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
      auto prototype = factory->create();
      if (!prototype)
        detail::throw_graph_error(std::move(prototype.error()));
      auto registered = detail::graph_register_operator(state_,
                                                        name,
                                                        factory,
                                                        std::move(prototype.value()),
                                                        &detail::kGraphOperatorTypeToken<Op>);
      if (!registered)
        detail::throw_graph_error(std::move(registered.error()));
      return OperatorHandle<Op>{this, state_, registered.value().index};
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
    requires((detail::GraphOwnedCapture<Args> && ...) &&
             std::constructible_from<Op, const detail::NormalizedGraphCapture<Args>&...>)
  OperatorHandle<Op> add_operator(std::string_view name, Args&&... args) {
    return op<Op>(name, std::forward<Args>(args)...);
  }

  /// @brief Independently construct, compose, and own one reusable Graph instance.
  /// @tparam G Concrete reusable Graph type.
  /// @tparam Args Constructor capture argument types.
  /// @param instance_name Nonempty unique local embedded-instance name.
  /// @param args Constructor values normalized and copied into the child object.
  /// @return Read-only typed handle exposing the child's public interface members.
  /// @throws RuntimeError if parent/child composition, construction, canonical
  /// remapping, name validation, or publication fails.
  /// @note Child composition reaches a terminal state before one atomic parent
  /// publication. The child Graph is never copied or moved and remains owned by
  /// the parent until parent retirement.
  /// @warning The handle permits interface inspection only. It does not grant
  /// mutable access to the child authoring Graph.
  template <typename G, typename... Args>
    requires std::derived_from<G, Graph> && (detail::GraphOwnedCapture<Args> && ...) &&
             std::constructible_from<G, const detail::NormalizedGraphCapture<Args>&...>
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
      detail::throw_graph_error(Error{ErrorCode::kFailure, "GRAPH_EMBED_CONSTRUCTION_FAILED"});
    } catch (...) {
      detail::graph_log_factory_exception("embedded_graph", "non-standard exception");
      detail::throw_graph_error(
          Error{ErrorCode::kFailure, "GRAPH_EMBED_CONSTRUCTION_UNKNOWN_EXCEPTION"});
    }
    const std::shared_ptr<detail::GraphState> child_state = child->state_;
    auto typed = detail::graph_prepare_embedded_type(
        state_, child_state, &detail::kGraphEmbeddingTypeToken<G>);
    if (!typed)
      detail::throw_graph_error(std::move(typed.error()));
    const auto child_snapshot = detail::graph_snapshot(child_state);
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
  ImportedTopic<T> import_topic(
      std::string_view uri,
      TopicOptions options = {}) {  // NOLINT(performance-unnecessary-value-param)
    auto prepared = detail::graph_prepare_authoring(state_);
    if (!prepared)
      detail::throw_graph_error(std::move(prepared.error()));
    auto port = std::make_unique<Output<T>>("external");
    auto result =
        detail::graph_register_import(state_,
                                      uri,
                                      options,
                                      std::unique_ptr<OutputBase>(std::move(port)),
                                      detail::operator_spec_payload_identity<T>(),
                                      detail::operator_spec_schema_package_plan_record<T>(),
                                      detail::serialized_payload_capacity_bytes<T>());
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
  ExportedTopic<T> export_topic(
      std::string_view uri,
      TopicOptions options = {}) {  // NOLINT(performance-unnecessary-value-param)
    auto prepared = detail::graph_prepare_authoring(state_);
    if (!prepared)
      detail::throw_graph_error(std::move(prepared.error()));
    auto port = std::make_unique<Input<T>>("external");
    auto result =
        detail::graph_register_export(state_,
                                      uri,
                                      options,
                                      std::unique_ptr<InputBase>(std::move(port)),
                                      detail::operator_spec_payload_identity<T>(),
                                      detail::operator_spec_schema_package_plan_record<T>(),
                                      detail::serialized_payload_capacity_bytes<T>());
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
  /// @return Fluent partition handle accepting direct Operator members.
  /// @throws RuntimeError if \p name is invalid or duplicated.
  /// @note If any partition is authored, compile requires all partitions to be
  /// nonempty and to cover every executable Operator exactly once.
  /// @note Manual partitions and \ref region declarations are orthogonal;
  /// regions may overlap partition boundaries.
  /// @note Manual partitions are incompatible with embedded sub-graphs.
  /// Compiling a Graph that has both reports
  /// `PARTITION_MANUAL_EMBEDDED_GRAPH_UNSUPPORTED`; remove `partition()`
  /// declarations or avoid `embed()` to recover automatic partitioning.
  PartitionHandle partition(std::string_view name);

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
  /// @pre Every arbitrary user-defined capture copy independently owns replayable
  /// state, and construction is repeatable. Constructor side effects are not
  /// authoritative runtime state.
  template <std::derived_from<Clock> ClockT, typename... Args>
    requires((detail::GraphOwnedCapture<Args> && ...) &&
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
  // rvalue transfer into the type-erased implementation while giving lvalue callers one explicit
  // ownership copy at this public boundary.
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
  std::shared_ptr<detail::GraphState> state_;
  friend class detail::GraphCompilerAccess;
  friend class detail::GraphState;
};

}  // namespace holoscan
