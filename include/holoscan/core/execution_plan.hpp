/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/holoscan_core_export.h>

#include <array>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/channel/clock_health.hpp>
#include <holoscan/channel/endpoint_catalog.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/operator_ref.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/parameter_catalog.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/rate.hpp>
#include <holoscan/core/readiness_source.hpp>
#include <holoscan/core/source_id.hpp>
#include <holoscan/core/temporal_plan.hpp>
#include <holoscan/threading/thread_placement.hpp>
#include <holoscan/time/clock_registry.hpp>

namespace holoscan {

namespace detail {
struct ExecutionPlanState;
class ExecutionPlanAccess;
}  // namespace detail

/// @brief Inspect the immutable canonical Graph stored by one execution plan.
///
/// This view has a deliberately distinct type from \ref GraphView. GraphView
/// preserves authored order and snapshot-local identities that correspond to
/// authoring handles, whereas this view exposes compiler-canonical order and
/// dense plan-local identities that join directly to other ExecutionPlan
/// records. Copies retain the immutable plan snapshot and remain valid after
/// the source Graph or the ExecutionPlan that produced the view is destroyed.
/// Concurrent reads of separate copies are safe.
class HOLOSCAN_CORE_EXPORT ExecutionGraphView {
 public:
  /// @brief Create an empty canonical plan view.
  ExecutionGraphView() = default;

  /// @brief Return the frozen source Graph authoring generation.
  /// @return Zero for an empty view; otherwise the frozen generation.
  [[nodiscard]] GraphGeneration generation() const noexcept;

  /// @brief Return the retained source Graph authoring identity.
  /// @return Zero-initialized identity for an empty view.
  [[nodiscard]] GraphAuthoringIdentity identity() const noexcept;

  /// @brief Return canonical Graph records.
  /// @return Records whose IDs share the ExecutionPlan identity domain.
  [[nodiscard]] std::span<const GraphRecord> graphs() const noexcept;

  /// @brief Return canonical logical vertex records.
  /// @return Records whose IDs join directly to plan vertex records.
  [[nodiscard]] std::span<const VertexRecord> vertices() const noexcept;

  /// @brief Return canonical pointer-free port records.
  /// @return Records whose IDs join directly to plan port records.
  [[nodiscard]] std::span<const PortRecord> ports() const noexcept;

  /// @brief Return canonical typed queue-edge records.
  /// @return Records whose IDs join directly to plan edge records.
  [[nodiscard]] std::span<const EdgeRecord> edges() const noexcept;

  /// @brief Return canonical same-direction interface aliases.
  /// @return Immutable interface-binding records.
  [[nodiscard]] std::span<const InterfaceBindingRecord> interface_bindings() const noexcept;

  /// @brief Return canonical intrinsic external topic records.
  /// @return Immutable external-topic records.
  [[nodiscard]] std::span<const ExternalTopicRecord> external_topics() const noexcept;

  /// @brief Return canonical readiness-source declarations.
  /// @return Immutable logical readiness declarations.
  [[nodiscard]] std::span<const ReadinessSourceDeclarationRecord> readiness_sources()
      const noexcept;

  /// @brief Return canonical region declarations.
  /// @return Immutable region records.
  [[nodiscard]] std::span<const RegionRecord> regions() const noexcept;

  /// @brief Return canonical clock declarations.
  /// @return Immutable cold clock declarations.
  [[nodiscard]] std::span<const ClockDeclarationRecord> clocks() const noexcept;

  /// @brief Return canonical reusable-Graph clock imports.
  /// @return Immutable scoped clock dependencies.
  [[nodiscard]] std::span<const ClockImportAuthoringRecord> clock_imports() const noexcept;

  /// @brief Return canonical exact embedded clock bindings.
  /// @return Immutable parent-to-child clock bindings.
  [[nodiscard]] std::span<const EmbeddedClockBindingAuthoringRecord> clock_bindings()
      const noexcept;

  /// @brief Return canonical directed clock-conversion declarations.
  /// @return Immutable authored conversion records.
  [[nodiscard]] std::span<const ClockConversionAuthoringRecord> clock_conversions() const noexcept;

  /// @brief Return canonical same-stage lifecycle dependencies.
  /// @return Immutable authored lifecycle dependency records.
  [[nodiscard]] std::span<const LifecycleDependencyAuthoringRecord> lifecycle_dependencies()
      const noexcept;

  /// @brief Return canonical deny-by-default restart permissions.
  /// @return Immutable authored restart-permission records.
  [[nodiscard]] std::span<const LifecycleRestartPermissionAuthoringRecord>
  lifecycle_restart_permissions() const noexcept;

 private:
  explicit ExecutionGraphView(GraphView view) noexcept : view_(std::move(view)) {}

  GraphView view_;
  friend class ExecutionPlan;
};

/// @brief Hold one deterministic 256-bit execution-plan digest.
///
/// This digest covers versioned normalized plan semantics. It is distinct from
/// \ref GraphAuthoringHash and is not a cryptographic authenticity claim.
struct PlanDigest {
  std::array<std::byte, 32> value{};  ///< Digest bytes in canonical byte order.
  friend constexpr bool operator==(PlanDigest, PlanDigest) noexcept = default;
};

/// @brief Identify one compiler or runtime policy algorithm.
struct AlgorithmKey {
  std::string id;           ///< Owned stable algorithm identifier.
  std::uint32_t version{};  ///< Observable semantic version; zero is invalid.
  friend bool operator==(const AlgorithmKey&, const AlgorithmKey&) noexcept = default;
};

/// @brief Classify the support maturity of an accepted algorithm.
enum class AlgorithmMaturity : std::uint8_t {
  kProduction = 0,    ///< Supported default behavior for this release.
  kExperimental = 1,  ///< Private developer-preview behavior with no compatibility promise.
};

/// @brief Report whether a requested planning decision was admitted.
enum class PlanDisposition : std::uint8_t {
  kAccepted = 0,          ///< The requested algorithm produced the accepted cover.
  kRejected = 1,          ///< Selection or planning failed without fallback.
  kExplicitFallback = 2,  ///< Reserved schema value; baseline never produces it.
};

/// @brief Record the authority that selected one planning algorithm.
enum class PlanSelectionSource : std::uint8_t {
  kCoreDefault = 0,          ///< Core selected the production default.
  kCompilerPolicy = 1,       ///< Reserved compiler-policy selection.
  kCompilerService = 2,      ///< A private compiler service supplied the request.
  kTestOverride = 3,         ///< A source-tree conformance seam supplied the request.
  kEnvironmentOverride = 4,  ///< The exact baseline experimental environment path supplied it.
  kExplicitFallback = 5,     ///< Reserved schema value; baseline never produces it.
};

/// @brief Identify one compiler-produced execution partition within a plan.
struct ExecutionPartitionId {
  std::uint32_t value{};  ///< Dense nonzero canonical partition identity.

  /// @brief Return whether this identity names an accepted partition.
  /// @return \c true when \ref value is nonzero.
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0U; }
  friend constexpr bool operator==(ExecutionPartitionId, ExecutionPartitionId) noexcept = default;
};

/// @brief Classify one compiler partition decision for a vertex.
enum class PartitionDecisionKind : std::uint8_t {
  kStart = 0,             ///< Start the first or independently rooted partition.
  kJoin = 1,              ///< Join a preceding compatible partition.
  kBoundarySplit = 2,     ///< Split at a mandatory execution-domain boundary.
  kFanInSplit = 3,        ///< Split at a join of distinct predecessor partitions.
  kFanOutLegSplit = 4,    ///< Split one distinct fan-out successor leg.
  kCollapseRejected = 5,  ///< Split because a proposed collapse was illegal.
  kAuthored = 6,          ///< Preserve an exact developer-authored partition anchor.
};

/// @brief Explain one partition start, join, or rejected collapse.
///
/// Enumerator order is the canonical reason order used in plan serialization.
enum class PartitionDecisionReason : std::uint8_t {
  kProcessBoundary = 0,                ///< Separate incompatible runtime process domains.
  kExternalEndpointBoundary = 1,       ///< Preserve an external client boundary.
  kDeviceTransferIncompatibility = 2,  ///< Separate incompatible device movement domains.
  kRegionBackendAdmission = 3,         ///< Preserve region/backend admission boundaries.
  kExplicitThreadPlacement = 4,        ///< Preserve explicitly distinct thread placement.
  kLifecycleIsolation = 5,        ///< Reserved; the source-chain planner does not emit this reason.
  kReadinessIsolation = 6,        ///< Preserve incompatible readiness behavior.
  kIndependentReadinessRoot = 7,  ///< Start at an independently triggered root.
  kSourceRoot = 8,                ///< Start at a source vertex with no executable predecessor.
  kFanIn = 9,                     ///< Split at predecessors belonging to distinct partitions.
  kFanOutLeg = 10,                ///< Give one fan-out successor leg a distinct partition.
  kCollapseRejected = 11,         ///< Preserve a validator-rejected collapse boundary.
  kLinearChainJoin = 12,          ///< Join one compatible serial linear-chain successor.
  kAuthoredPartitionBoundary = 13,  ///< Preserve an exact authored-anchor boundary.
};

/// @brief Identify the authority that produced one accepted partition cover.
enum class PartitioningMode : std::uint8_t {
  kAutomatic = 0,             ///< The production planner produced the complete cover.
  kConstrainedAutomatic = 1,  ///< Authored exact anchors constrain an automatic remainder.
  kManual = 2,                ///< Authored declarations produced the complete exact cover.
};

/// @brief Identify whether one accepted execution partition was authored or automatic.
enum class ExecutionPartitionOrigin : std::uint8_t {
  kAuthored = 0,   ///< The partition has an exact developer-authored name and membership.
  kAutomatic = 1,  ///< The partition is an unnamed planner result.
};

/// @brief Preserve selection and admission evidence for the partition planner.
struct PartitioningPlanRecord {
  std::optional<AlgorithmKey> requested_algorithm;  ///< Absent only before valid key parsing.
  std::optional<AlgorithmKey> accepted_algorithm;   ///< Absent for every rejected plan.
  std::uint32_t interface_version{};                ///< Private planner interface version.
  std::optional<AlgorithmMaturity> maturity;        ///< Classified requested algorithm maturity.
  PlanDisposition disposition{PlanDisposition::kRejected};  ///< Total selection result.
  PlanSelectionSource selection_source{PlanSelectionSource::kCoreDefault};  ///< Provenance.
  PlanDigest config_digest{};  ///< Digest of normalized selector/profile configuration.
  DiagnosticId reason{};       ///< Primary stable selection or rejection reason.
  /// Accepted-cover authority. Meaningful only when \ref disposition is \c kAccepted; rejected
  /// records retain the aggregate default as storage, not as accepted partitioning evidence.
  PartitioningMode mode{PartitioningMode::kAutomatic};
  std::uint32_t authored_partition_count{};   ///< Number of exact authored partitions.
  std::uint32_t automatic_partition_count{};  ///< Number of unnamed planner partitions.
};

/// @brief Record the canonical partition decision for one executable vertex.
struct PartitionDecisionRecord {
  VertexId vertex{};                                              ///< Canonical logical vertex.
  ExecutionPartitionId partition{};                               ///< Final accepted partition.
  PartitionDecisionKind decision{PartitionDecisionKind::kStart};  ///< Decision category.
  std::vector<PartitionDecisionReason> reasons;  ///< Canonically ordered complete reasons.
};

/// @brief Describe one accepted execution partition.
struct ExecutionPartitionRecord {
  ExecutionPartitionId id{};                 ///< Dense final partition identity.
  std::optional<std::string> authored_name;  ///< Manual name, absent for implicit partitions.
  std::vector<VertexId> members;             ///< Canonically ordered executable vertices.
  /// Canonical logical Operator lookup keys in plan-Graph order; excludes synthetic members.
  /// Each decoded Graph/Operator UTF-8 segment is URI-percent-encoded bytewise with uppercase
  /// hexadecimal before slash joining. Keys use the no-leading-slash OperatorPath grammar.
  std::vector<std::string> operator_paths;
  std::uint32_t execution_lane{};  ///< Dense serial lane index, starting at zero.
};

/// @brief Retain immutable inspection of one accepted execution partition.
///
/// Copies share immutable plan-owned state. Accessor spans and string views remain valid for this
/// element-view lifetime, independently of the source plan, collection, or iterator.
class HOLOSCAN_CORE_EXPORT ExecutionPartitionView {
 public:
  ExecutionPartitionView() = delete;

  /// @brief Return the dense accepted partition identity.
  [[nodiscard]] ExecutionPartitionId id() const noexcept;

  /// @brief Return whether exact authoring or automatic planning produced this partition.
  [[nodiscard]] ExecutionPartitionOrigin origin() const noexcept;

  /// @brief Return the exact authored name, absent for automatic partitions.
  [[nodiscard]] std::optional<std::string_view> authored_name() const noexcept;

  /// @brief Return the complete canonical executable member set.
  [[nodiscard]] std::span<const VertexId> members() const noexcept;

  /// @brief Return exact canonical logical Operator lookup keys, excluding synthetic members.
  /// @note Reserved and non-ASCII bytes are percent-encoded per decoded path segment. Keys use the
  /// no-leading-slash OperatorPath grammar shared by Parameter records and Tensor placement.
  [[nodiscard]] std::span<const std::string> operator_paths() const noexcept;

  /// @brief Return the dense serial execution lane.
  [[nodiscard]] std::uint32_t execution_lane() const noexcept;

 private:
  ExecutionPartitionView(std::shared_ptr<const detail::ExecutionPlanState> state,
                         std::size_t index) noexcept
      : state_(std::move(state)), index_(index) {}

  std::shared_ptr<const detail::ExecutionPlanState> state_;
  std::size_t index_{};
  friend class ExecutionPartitionsView;
};

/// @brief Retain immutable range and lookup access to an accepted partition cover.
///
/// The range is intentionally not borrowed. Copies and copied iterators independently retain the
/// immutable state, and iterator dereference returns an independently retained element value.
class HOLOSCAN_CORE_EXPORT ExecutionPartitionsView {
 public:
  /// @brief Iterate retained partition element values in canonical ID order.
  class const_iterator {
   public:
    using iterator_concept = std::forward_iterator_tag;
    using iterator_category = std::input_iterator_tag;
    using value_type = ExecutionPartitionView;
    using difference_type = std::ptrdiff_t;

    const_iterator() noexcept = default;

    [[nodiscard]] ExecutionPartitionView operator*() const noexcept {
      return ExecutionPartitionView{state_, index_};
    }
    const_iterator& operator++() noexcept {
      ++index_;
      return *this;
    }
    const_iterator operator++(int) noexcept {
      const_iterator previous = *this;
      ++*this;
      return previous;
    }
    friend bool operator==(const const_iterator&, const const_iterator&) noexcept = default;

   private:
    const_iterator(std::shared_ptr<const detail::ExecutionPlanState> state,
                   std::size_t index) noexcept
        : state_(std::move(state)), index_(index) {}

    std::shared_ptr<const detail::ExecutionPlanState> state_;
    std::size_t index_{};
    friend class ExecutionPartitionsView;
  };

  ExecutionPartitionsView() = delete;

  /// @brief Return the accepted cover authority.
  [[nodiscard]] PartitioningMode mode() const noexcept;

  /// @brief Return the number of exact authored partitions.
  [[nodiscard]] std::uint32_t authored_count() const noexcept;

  /// @brief Return the number of unnamed automatic partitions.
  [[nodiscard]] std::uint32_t automatic_count() const noexcept;

  /// @brief Return the number of accepted execution partitions.
  [[nodiscard]] std::size_t size() const noexcept;

  /// @brief Return whether the accepted cover contains no partitions.
  [[nodiscard]] bool empty() const noexcept;

  [[nodiscard]] const_iterator begin() const noexcept { return const_iterator{state_, 0U}; }
  [[nodiscard]] const_iterator end() const noexcept { return const_iterator{state_, size()}; }

  /// @brief Return the unique partition containing one root-Graph Operator handle.
  /// @throws RuntimeError with a typed code for unusable, foreign, stale, or unknown identity.
  template <std::derived_from<OperatorBase> Op>
  [[nodiscard]] ExecutionPartitionView of(const OperatorHandle<Op>& op) const {
    if (!usable()) {
      throw RuntimeError{ErrorCode::kInvalidArgument};
    }
    const auto graph_state = op.state_.lock();
    if (!graph_state) {
      throw RuntimeError{ErrorCode::kInvalidArgument};
    }
    auto locator = detail::graph_operator_locator(graph_state, op.index_);
    if (!locator) {
      throw RuntimeError{locator.error()};
    }
    return of_root_locator(locator.value());
  }

  /// @brief Return the unique partition containing one plan-bound Operator reference.
  /// @throws RuntimeError with a typed code for unusable, invalid, stale, or unknown identity.
  [[nodiscard]] ExecutionPartitionView of(OperatorRef plan_operator) const;

  /// @brief Look up one partition by exact nonzero dense ID.
  [[nodiscard]] expected<ExecutionPartitionView, Error> lookup(
      ExecutionPartitionId id) const noexcept;

  /// @brief Look up one partition by exact authored name.
  [[nodiscard]] expected<ExecutionPartitionView, Error> lookup_named(
      std::string_view authored_name) const noexcept;

  /// @brief Look up one partition by an exact published canonical logical Operator key.
  /// @note Input is compared byte-for-byte without decoding or normalization. Obtain the key from
  /// \ref ExecutionPartitionView::operator_paths when an Operator or Graph name needs encoding.
  [[nodiscard]] expected<ExecutionPartitionView, Error> lookup_for_path(
      std::string_view canonical_operator_path) const noexcept;

 private:
  explicit ExecutionPartitionsView(std::shared_ptr<const detail::ExecutionPlanState> state) noexcept
      : state_(std::move(state)) {}
  [[nodiscard]] bool usable() const noexcept;
  [[nodiscard]] ExecutionPartitionView of_root_locator(
      const detail::GraphOperatorLocator& locator) const;

  std::shared_ptr<const detail::ExecutionPlanState> state_;
  friend class ExecutionPlan;
};

/// @brief Preserve one original graph edge that crosses accepted partitions.
struct ScheduleEdgeRecord {
  EdgeId edge{};                       ///< Canonical original graph edge.
  ExecutionPartitionId source{};       ///< Source partition.
  ExecutionPartitionId destination{};  ///< Destination partition.
};

/// @brief Describe the aggregate runtime scheduling contract selected by compilation.
///
/// Each callback-bearing execution partition contributes one independently runnable callback
/// owner. Interface-only partitions retain topology lanes but materialize no empty Runtime worker.
/// Callback execution within each owner remains deterministic and serial, while distinct owners
/// may make progress concurrently. \ref intra_path_parallelism remains one whenever at least one
/// callback owner exists because no individual partition executes more than one callback at a
/// time.
struct SchedulerPlanRecord {
  AlgorithmKey ready_policy{};           ///< Accepted partition-local READY-policy identity.
  std::uint32_t execution_lane_count{};  ///< Number of accepted partition topology lanes.
  /// Maximum concurrently active callback owners; zero for zero-work or endpoint-only plans.
  std::uint32_t max_concurrent_activations{};
  /// True exactly when distinct partition owners may overlap.
  bool activation_overlap{};
  /// Number of callback-bearing partition owners; interface-only lanes do not contribute.
  std::uint32_t worker_count{};
  /// Maximum callback parallelism inside one partition; zero when no callback owner exists.
  std::uint32_t intra_path_parallelism{};
  DiagnosticId reason{};  ///< Stable scheduling-policy reason.
};

/// @brief Classify one compile-time placement proof without applying operating-system state.
enum class PlacementProbeStatus : std::uint8_t {
  kNotRequested = 0,      ///< The corresponding placement property was not requested.
  kPassed = 1,            ///< Static compile-time evidence proves the requested property.
  kRuntimeRequired = 2,   ///< Runtime must apply and read back the accepted property.
  kFallbackRecorded = 3,  ///< Policy admitted an explicit bounded fallback.
  kRejected = 4,          ///< Required evidence was unavailable and admission failed.
};

/// @brief Preserve one bounded placement proof or deferred readback obligation.
struct PlacementProbePlanRecord {
  ExecutionPartitionId partition{};  ///< Canonical execution partition being checked.
  std::string probe_id;              ///< Stable probe identity.
  PlacementProbeStatus status{PlacementProbeStatus::kNotRequested};  ///< Total result.
  std::string evidence_id;  ///< Stable evidence or obligation id.
  DiagnosticId reason{};    ///< Stable rejection/fallback reason.
};

/// @brief Freeze requested and accepted placement for one named execution partition.
///
/// A request that resolves a unique partition name carries its canonical
/// nonzero @ref partition identity and dense execution lane even if later
/// placement validation rejects it. An unknown name carries a zero partition
/// identity, and its execution lane has no semantic meaning. Every rejected
/// record has an empty @ref accepted value and a stable diagnostic.
struct PartitionPlacementPlanRecord {
  ExecutionPartitionId partition{};                         ///< Canonical execution partition.
  std::string authored_name;                                ///< Exact owned manual partition name.
  std::uint32_t execution_lane{};                           ///< Dense lane owned by the partition.
  ThreadPlacement requested;                                ///< Normalized deployment request.
  std::optional<ThreadPlacement> accepted;                  ///< Frozen accepted placement.
  PlanDisposition disposition{PlanDisposition::kRejected};  ///< Total admission result.
  DiagnosticId reason{};                                    ///< Stable primary reason, or zero.
};

/// @brief Map one deterministic serial execution lane to optional placement evidence.
struct ExecutionLanePlanRecord {
  std::uint32_t execution_lane{};        ///< Dense zero-based lane identity.
  ExecutionPartitionId partition{};      ///< Exact owning execution partition.
  ThreadPlacement accepted_placement{};  ///< Immutable lane policy consumed by Runtime.
};

/// @brief Identify one compile-admitted process-equivalence group.
struct ProcessGroupId {
  std::uint32_t value{};  ///< Dense nonzero identity in canonical group order.
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0U; }
  friend constexpr bool operator==(ProcessGroupId, ProcessGroupId) noexcept = default;
};

/// @brief Classify the physical communication boundary represented by one payload edge.
enum class ChannelLocalityKind : std::uint8_t {
  kProcessLocal = 0,    ///< Both endpoints execute inside this Runtime process.
  kExternalImport = 1,  ///< An external publisher supplies the Graph-side source.
  kExternalExport = 2,  ///< A Graph-side producer supplies an external subscriber.
  kCrossProcess = 3,    ///< An internal edge crosses compile-admitted process groups.
};

/// @brief Preserve one canonical compiled port and its resolved finite bounds.
///
/// The record deep-owns semantic identity and contains no declaration address,
/// queue object, pool handle, or Runtime binding. Its \ref id is scoped by the
/// enclosing execution plan and remains valid for that plan's lifetime.
struct PortPlanRecord {
  PortId id{};                                         ///< Dense nonzero plan-local port identity.
  VertexId owner{};                                    ///< Canonical owning logical vertex.
  PortOwnerKind owner_kind{PortOwnerKind::kOperator};  ///< Owning object family.
  PortDirection direction{PortDirection::kInput};      ///< Accepted data-flow direction.
  std::string local_name;                              ///< Owned declaration-local port name.
  PayloadIdentityRecord payload_identity;              ///< Complete pointer-free payload identity.
  /// Positive generated-package serialized bound; zero for every other payload family.
  std::uint64_t serialized_payload_capacity_bytes{};
  /// Whether compile admitted bounded direct generic-loan construction for this output.
  bool generic_loan_supported{};
  /// Exact native constructed-value slot size when generic loans are supported; otherwise zero.
  std::uint64_t generic_loan_slot_size_bytes{};
  std::uint32_t queue_depth{};                        ///< Input queue depth; zero for outputs.
  std::uint32_t max_emits_per_compute{};              ///< Output burst bound; zero for inputs.
  std::uint32_t late_joiner_history_depth{};          ///< Output history; zero in the baseline.
  std::optional<TensorRepresentation> tensor_layout;  ///< Accepted logical Tensor facts.
  /// Complete normalized input requirements or output promises when authored.
  std::optional<TensorContractRecord> tensor_contract;
  std::optional<Profile> effective_profile;  ///< Resolved owner profile, absent on rejection.
};

/// @brief Preserve a reusable-Graph interface alias in plan-local port identities.
struct InterfacePortBindingPlanRecord {
  PortId public_port{};    ///< Public reusable-Graph interface port.
  PortId internal_port{};  ///< Internal Operator port represented by the alias.
  PortDirection direction{PortDirection::kInput};  ///< Shared validated direction.
};

/// @brief Preserve one resolved typed payload edge and its finite retention geometry.
struct PayloadEdgePlanRecord {
  EdgeId edge{};                                     ///< Canonical source Graph edge.
  PortId source{};                                   ///< Plan-local output identity.
  PortId target{};                                   ///< Plan-local input identity.
  PayloadIdentityRecord payload_identity;            ///< Complete accepted payload identity.
  QueuePolicy queue_policy{QueuePolicy::kBlocking};  ///< Resolved edge queue policy.
  std::uint32_t queue_depth{};                       ///< Positive resolved destination depth.
  /// Positive finite diagnostic bound admitted for this flow.
  ///
  /// This value records the authored or derived maximum for validation and
  /// inspection. It does not independently reserve payload slots; physical
  /// capacity is derived from queue, cache, window, history, publication, and
  /// retirement claims recorded elsewhere in the plan.
  std::uint32_t max_inflight_samples{};
  std::uint32_t producer_max_emits_per_compute{};  ///< Positive producer burst bound.
  std::uint32_t fanout{};                          ///< Positive number of source successors.
  std::uint32_t cache_capacity{};                  ///< Selector/cache retention for the target.
  ChannelLocalityKind locality{ChannelLocalityKind::kProcessLocal};  ///< Frozen locality.
  PlanDisposition disposition{PlanDisposition::kRejected};  ///< Total edge admission result.
  DiagnosticId reason{};                                    ///< Stable rejection reason, or zero.
};

/// @brief Identify one immutable queue-sizing recipe.
struct QueueSizingRecipeId {
  std::uint32_t value{};
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0U; }
  friend constexpr bool operator==(QueueSizingRecipeId, QueueSizingRecipeId) noexcept = default;
};

/// @brief Identify one immutable accumulation-horizon record.
struct HorizonRecordId {
  std::uint32_t value{};
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0U; }
  friend constexpr bool operator==(HorizonRecordId, HorizonRecordId) noexcept = default;
};

/// @brief Identify one immutable hard-floor contribution.
struct HardFloorContributorId {
  std::uint32_t value{};
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0U; }
  friend constexpr bool operator==(HardFloorContributorId,
                                   HardFloorContributorId) noexcept = default;
};

struct HardFloorScenarioId {
  std::uint32_t value{};
  friend constexpr bool operator==(HardFloorScenarioId, HardFloorScenarioId) noexcept = default;
};

struct HardFloorAlternativeGroupId {
  std::uint32_t value{};
  friend constexpr bool operator==(HardFloorAlternativeGroupId,
                                   HardFloorAlternativeGroupId) noexcept = default;
};

/// @brief Preserve an exact positive rational duration in nanoseconds.
struct ExactDuration {
  std::uint64_t numerator_ns{};
  std::uint64_t denominator{1U};
  friend constexpr bool operator==(ExactDuration, ExactDuration) noexcept = default;
};

enum class AccumulationHorizonOrigin : std::uint8_t {
  kProducerInterarrival = 0,
  kDriverInterarrival = 1,
  kClockPeriod = 2,
  kTemporalWindowBound = 3,
  kExternalCadence = 4,
};

enum class HardFloorOrigin : std::uint8_t {
  kMinimumQueue = 0,
  kProducerBurst = 1,
  kSelectorRetention = 2,
  kWindowRetention = 3,
  kPendingActivation = 4,
  kProviderMinimum = 5,
};

/// @brief Identify the authority that selected an effective logical queue depth.
enum class QueueDepthOrigin : std::uint8_t {
  kGraphEdgeOverride = 0,
  kAddFlowOption = 1,
  kGraphInputOverride = 2,
  kOperatorSetup = 3,
  kRateAndContractDerived = 4,
  kContractDerived = 5,
  kFrameworkFallback = 6,
};

/// @brief Classify nominal stability independently of depth authority.
enum class StabilityClass : std::uint8_t {
  kStableNominalEstimate = 0,
  kPredictedSustainedBlocking = 1,
  kPredictedReplacementOrDrop = 2,
  kUnresolvedBoundedEvidence = 3,
};

enum class RateComparison : std::uint8_t {
  kNotCompared = 0,
  kEqual = 1,
  kMismatch = 2,
};

/// @brief Freeze pointer-free rate declaration and selection-domain authority for one port.
struct PortRatePlanRecord {
  PortId port{};
  PortRateKind kind{PortRateKind::kUnspecified};
  std::optional<Rate> declared_nominal_rate;
  std::optional<Rate> declared_expected_rate;
  std::optional<PortId> follows_input;
  std::optional<RateRatio> follows_ratio;
  std::optional<LifecycleStage> lifecycle_binding_stage;
  std::optional<Rate> declared_max_nominal_rate;
  PlanDisposition disposition{PlanDisposition::kRejected};
  DiagnosticId reason{};
};

/// @brief Preserve one exact finite accumulation horizon used by a sizing recipe.
struct AccumulationHorizonPlanRecord {
  HorizonRecordId id{};
  EdgeId edge{};
  std::optional<ExactDuration> horizon;
  std::optional<PortId> inverse_rate_port;
  AccumulationHorizonOrigin origin{AccumulationHorizonOrigin::kProducerInterarrival};
  std::uint32_t source_record{};
};

/// @brief Preserve one reviewable hard-floor contribution and its composition identity.
struct HardFloorContributorPlanRecord {
  HardFloorContributorId id{};
  EdgeId edge{};
  HardFloorOrigin origin{HardFloorOrigin::kMinimumQueue};
  std::uint32_t required_slots{};
  HardFloorScenarioId scenario{};
  HardFloorAlternativeGroupId alternative_group{};
};

enum class ProviderReusePolicy : std::uint8_t {
  kCompatibleExactOrLarger = 0,
  kFreshOnly = 1,
};

enum class AllocationGeometryMode : std::uint8_t {
  kCompileConcrete = 0,
  kLifecycleCandidate = 1,
};

/// @brief Freeze the immutable sizing formula and authoring authority for one logical edge.
struct QueueSizingRecipePlanRecord {
  QueueSizingRecipeId id{};
  EdgeId edge{};
  QueueDepthOrigin origin{QueueDepthOrigin::kFrameworkFallback};
  std::optional<std::uint32_t> explicit_depth;
  RateRatio headroom_ratio{};
  std::optional<LifecycleStage> latest_binding_stage;
  ProviderReusePolicy reuse_policy{ProviderReusePolicy::kCompatibleExactOrLarger};
  AllocationGeometryMode geometry_mode{AllocationGeometryMode::kCompileConcrete};
  PlanDisposition disposition{PlanDisposition::kRejected};
  DiagnosticId reason{};
};

/// @brief Record requested and accepted locality for one payload edge.
struct ChannelLocalityRecord {
  EdgeId edge{};  ///< Canonical source Graph edge.
  ChannelLocalityKind requested{ChannelLocalityKind::kProcessLocal};  ///< Authored boundary.
  ChannelLocalityKind accepted{ChannelLocalityKind::kProcessLocal};   ///< Frozen realization.
  bool requires_runtime_attach{};  ///< Whether startup must attach an external endpoint.
};

/// @brief Hold service-wide Channel requirements that either startup role must verify.
///
/// Publisher and subscriber endpoint-local requests live in distinct records;
/// neither may redefine this service-static contract at startup.
struct ChannelServiceStaticPlanRecord {
  ExternalTopicId topic{};                 ///< Canonical external boundary identity.
  std::string service_id;                  ///< Stable plan-owned service identity.
  PayloadIdentityRecord payload_identity;  ///< Complete wire/payload identity.
  /// Canonical logical field contracts for a descriptor-backed schema, otherwise empty.
  std::vector<DescriptorTensorTopicFieldRecord> tensor_fields;
  QueuePolicy queue_policy{QueuePolicy::kBlocking};  ///< Accepted service queue family.
  std::uint32_t max_subscriber_queue_depth{};        ///< Positive service-wide maximum.
  std::uint32_t max_subscriber_borrowed_samples{};   ///< Positive service-wide maximum.
  std::uint32_t publisher_history_depth{};           ///< Exactly zero for baseline.
  bool event_wake{};                                 ///< Whether event wake is required.
  bool tensor_payload{};                             ///< Whether Tensor wire policy applies.
  InterfaceTensorEmitPolicy tensor_emit_policy{
      InterfaceTensorEmitPolicy::kRequireExportable};  ///< Frozen Tensor materialization policy.
};

/// @brief Hold publisher-local endpoint requirements without service maxima.
struct ChannelPublisherEndpointPlanRecord {
  ExternalTopicId topic{};                  ///< Canonical external boundary identity.
  std::string service_id;                   ///< Referenced service-static record.
  std::uint32_t max_loaned_samples{};       ///< Positive finite publisher-local credit.
  bool replacement_attribution_required{};  ///< Whether lossy fanout needs attribution.
};

/// @brief Hold subscriber-local endpoint requests without service authority.
struct ChannelSubscriberEndpointPlanRecord {
  ExternalTopicId topic{};                ///< Canonical external boundary identity.
  std::string service_id;                 ///< Referenced service-static record.
  std::uint32_t requested_queue_depth{};  ///< Positive endpoint-local queue request.
  std::uint32_t max_borrowed_samples{};   ///< Positive endpoint-local read credit.
  bool event_wake{};                      ///< Endpoint-local wait integration request.
};

/// @brief Classify evidence used to admit a generic non-Tensor output loan.
enum class GenericLoanTraitEvidence : std::uint8_t {
  kStaticallyBoundedScalar = 0,         ///< Fixed scalar construction/reset is bounded.
  kNothrowDefaultConstructiblePOD = 1,  ///< POD value construction/destruction is nonthrowing.
  kNothrowDefaultConstructibleSchema =
      2,  ///< Owning schema value construction/destruction is nonthrowing.
};

/// @brief Freeze one generic non-Tensor output-loan capability.
struct GenericOutputLoanPlanRecord {
  PortId output{};                     ///< Direct non-Tensor output identity.
  PlanGenerationId plan_generation{};  ///< Nonzero enclosing plan identity.
  RunGenerationId run_generation{};    ///< Zero for this immutable plan-bound record.
  std::string pool_id;                 ///< Stable logical pool identity, never a handle.
  GenericLoanTraitEvidence trait_evidence{
      GenericLoanTraitEvidence::kStaticallyBoundedScalar};  ///< Payload qualification family.
  std::uint64_t slot_size_bytes{};          ///< Exact constructed value size for capacity math.
  std::uint64_t aggregate_byte_capacity{};  ///< Checked product of slot count and slot size.
  std::uint32_t slot_count{};               ///< Positive finite constructed-slot capacity.
  std::uint32_t max_live_loans{};           ///< Positive unpublished-loan credit.
  std::uint32_t max_emits_per_compute{};    ///< Positive output burst bound.
  std::uint32_t retirement_capacity{};      ///< Fixed retirement-record capacity.
  LifecycleStage materialize_before{LifecycleStage::kArm};  ///< Latest materialization stage.
  std::optional<Profile> effective_profile;                 ///< Resolved owner profile.
  PlanDisposition disposition{PlanDisposition::kRejected};  ///< Total capability admission.
  bool pre_submit_preserves_same_activation_loan{true};     ///< Explicit caller retry rule.
  bool runtime_retry_enabled{false};                        ///< Always false in baseline.
  bool fallback_rejected{true};                             ///< No heap/growth fallback.
};

/// @brief Identify one canonical Tensor-slot retention reason.
///
/// These values have stable order for plan and evidence serialization. Multiple
/// reasons on one slot do not consume multiple physical-slot capacity units.
enum class TensorSlotClaimReason : std::uint8_t {
  kProducerLoan = 0,
  kProducerWriter = 1,
  kProducerReadiness = 2,
  kAcceptedPublication = 3,
  kHolomqTransportOrFanout = 4,
  kSubscriberBorrowOrImport = 5,
  kConsumerCudaRelease = 6,
  kRelayOrReexport = 7,
  kTemporalCache = 8,
  kPendingOrSelectedWindow = 9,
  kReuseLast = 10,
  kDrainAllActivation = 11,
  kDlpackOrFrameworkBorrow = 12,
  kCopyOrRecorderMaterialization = 13,
  kCancellation = 14,
  kQuarantine = 15,
  kRetirementService = 16,
};

/// @brief Return the stable snake-case token for one Tensor claim reason.
/// @param reason Closed claim-reason value.
/// @return Stable schema token, or an empty view for an unknown value.
[[nodiscard]] HOLOSCAN_CORE_EXPORT std::string_view tensor_slot_claim_reason_name(
    TensorSlotClaimReason reason) noexcept;

/// @brief Record one unique-slot capacity contributor and all reasons it covers.
struct PoolCapacityContributorRecord {
  std::string pool_id;                         ///< Stable owning logical pool identity.
  std::string contributor_id;                  ///< Stable bound-derivation identity.
  std::uint32_t maximum_distinct_slots{};      ///< Unique slots, never claim-counter sum.
  std::vector<TensorSlotClaimReason> reasons;  ///< Canonically sorted covered reasons.
};

/// @brief Identify one immutable stable-pool capacity dependency.
struct PoolCapacityDependencyId {
  std::uint32_t value{};
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0U; }
  friend constexpr bool operator==(PoolCapacityDependencyId,
                                   PoolCapacityDependencyId) noexcept = default;
};

/// @brief Identify one immutable provider-allocation recipe.
struct ProviderAllocationPlanRecordId {
  std::uint32_t value{};
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0U; }
  friend constexpr bool operator==(ProviderAllocationPlanRecordId,
                                   ProviderAllocationPlanRecordId) noexcept = default;
};

struct PoolCapacityScenarioId {
  std::uint32_t value{};
  friend constexpr bool operator==(PoolCapacityScenarioId,
                                   PoolCapacityScenarioId) noexcept = default;
};

struct PoolCapacityAlternativeGroupId {
  std::uint32_t value{};
  friend constexpr bool operator==(PoolCapacityAlternativeGroupId,
                                   PoolCapacityAlternativeGroupId) noexcept = default;
};

/// @brief Classify the logical lifetime represented by one stable-pool dependency.
enum class PoolClaimOrigin : std::uint8_t {
  kEdgeDelivery = 0,
  kTransportRetention = 1,
  kTemporalRetention = 2,
  kProducerLoan = 3,
  kHistory = 4,
  kCopyOrExport = 5,
  kCancellationOrQuarantine = 6,
  kRetirement = 7,
  kProviderSpecific = 8,
};

/// @brief Select the sole immutable authority for one pool-capacity dependency.
enum class PoolCapacityDependencySource : std::uint8_t {
  kQueueSizingRecipe = 0,
  kPredecessorContributor = 1,
};

/// @brief Join one logical queue or predecessor claim to one stable provider pool.
struct PoolCapacityDependencyPlanRecord {
  PoolCapacityDependencyId id{};
  std::string pool_id;
  PoolClaimOrigin origin{PoolClaimOrigin::kEdgeDelivery};
  PoolCapacityDependencySource source{PoolCapacityDependencySource::kQueueSizingRecipe};
  QueueSizingRecipeId queue_sizing_recipe{};
  std::string predecessor_contributor_id;
  PoolCapacityScenarioId scenario{};
  PoolCapacityAlternativeGroupId alternative_group{};
  std::uint32_t mapping_multiplicity{1U};
  /// Queue-recipe dependencies own their canonical claim-reason evidence. Fixed dependencies
  /// inherit it from the referenced predecessor contributor and leave this vector empty.
  std::vector<TensorSlotClaimReason> reasons;
};

enum class ProviderAllocationKind : std::uint8_t {
  kGenericOutputLoan = 0,
  kTensorOutput = 1,
  kTensorFieldOutput = 2,
  kTensorInterface = 3,
};

/// @brief Freeze static Tensor-provider facts without inventing allocation geometry.
struct TensorProviderTemplatePlanRecord {
  std::string allocator_root_id;
  ProcessGroupId allocator_process_group{};
  TensorPoolBounds bounds{};
  std::vector<TensorAxisBounds> axes;
  std::optional<TensorRepresentation> layout;
  MemoryKind memory_kind{MemoryKind::kUnknown};
  DeviceId device{};
  std::uint32_t max_outstanding_loans{};
  std::uint32_t max_emits_per_compute{};
  std::optional<Profile> effective_profile;
  bool fallback_rejected{true};
};

/// @brief Freeze the dependency range and stable join for one provider allocation.
struct ProviderAllocationPlanRecord {
  ProviderAllocationPlanRecordId id{};
  std::string pool_id;
  PoolCapacityDependencyId first_sizing_dependency{};
  std::uint32_t sizing_dependency_count{};
  ProviderAllocationKind allocation_kind{ProviderAllocationKind::kTensorOutput};
  AllocationGeometryMode geometry_mode{AllocationGeometryMode::kCompileConcrete};
  std::optional<PortId> output;
  std::optional<std::uint16_t> tensor_field_id;
  std::optional<std::uint16_t> tensor_index;
  std::optional<ExternalTopicId> interface_topic;
  std::optional<TensorProviderTemplatePlanRecord> tensor_template;
  LifecycleStage complete_before{LifecycleStage::kAllocate};
  PlanDisposition disposition{PlanDisposition::kRejected};
  DiagnosticId reason{};
};

/// @brief Classify the strongest update boundary of a derived Tensor byte-span dependency.
enum class TensorByteSpanDerivationUpdateDisposition : std::uint8_t {
  kFrozenCompileTime = 0,  ///< Every Parameter dependency is compile-time immutable.
  kRecompileRequired = 1,  ///< At least one dependency requires compiling a new plan.
};

/// @brief Preserve one evaluated node of a derived Tensor byte-span expression.
struct TensorByteSpanDerivationNodePlanRecord {
  TensorByteSpanOperation operation{TensorByteSpanOperation::kPositiveConstant};  ///< Node kind.
  std::uint32_t left{};            ///< Left child index for compound operations.
  std::uint32_t right{};           ///< Right child index for binary operations.
  std::uint64_t operand{};         ///< Literal or alignment operand, otherwise zero.
  ParamFieldId parameter_field{};  ///< Parameter dependency, or zero.
  PortId input{};                  ///< Input dependency, or zero.
  std::uint64_t resolved_value{};  ///< Checked positive node result in bytes.
};

/// @brief Record compiler evidence for one derived Tensor output-pool slot capacity.
struct TensorByteSpanDerivationPlanRecord {
  PortId output{};  ///< Owning runtime-pooled Tensor output.
  /// Descriptor-schema field id, or no value for a raw Tensor output.
  std::optional<std::uint16_t> field_id;
  /// Dense descriptor-schema Tensor index, or no value for a raw Tensor output.
  std::optional<std::uint16_t> tensor_index;
  PlanGenerationId plan_generation{};      ///< Nonzero enclosing plan identity.
  RunGenerationId run_generation{};        ///< Zero for this immutable plan-bound evidence.
  std::uint64_t resolved_max_byte_span{};  ///< Concrete per-slot byte capacity.
  std::optional<std::uint64_t>
      resolved_max_staging_byte_span;  ///< Concrete staging capacity when copy-capable.
  TensorByteSpanDerivationUpdateDisposition update_disposition{
      TensorByteSpanDerivationUpdateDisposition::kFrozenCompileTime};  ///< Strongest dependency.
  std::vector<TensorByteSpanDerivationNodePlanRecord> nodes;  ///< Canonical evaluated postorder.
};

/// @brief Freeze one Tensor-backed output allocation capability and capacity.
struct TensorAllocationPlanRecord {
  PortId output{};                               ///< Tensor-backed output identity.
  PlanGenerationId plan_generation{};            ///< Nonzero enclosing plan identity.
  RunGenerationId run_generation{};              ///< Zero for this immutable plan-bound record.
  std::string pool_id;                           ///< Stable output pool identity.
  std::string allocator_root_id;                 ///< Stable process/device allocator lineage.
  ProcessGroupId allocator_process_group{};      ///< Compile-admitted process lineage.
  TensorPoolBounds bounds{};                     ///< Tightened maximum rank and byte span.
  std::vector<TensorAxisBounds> axes;            ///< Per-axis extent and byte-stride maxima.
  std::optional<TensorRepresentation> layout;    ///< Accepted logical output constraints.
  MemoryKind memory_kind{MemoryKind::kUnknown};  ///< Frozen physical storage class.
  DeviceId device{};                             ///< Resolved device, or invalid for host storage.
  std::uint32_t slot_count{};                    ///< Peak distinct live-slot capacity.
  std::uint64_t aggregate_byte_capacity{};       ///< Checked slots-times-byte-span maximum.
  std::uint32_t max_outstanding_loans{};         ///< Positive producer-loan credit.
  std::uint32_t max_emits_per_compute{};         ///< Positive output burst bound.
  std::uint32_t control_capacity{};              ///< Pre-materialized slot-control records.
  std::uint32_t readiness_capacity{};            ///< One concrete reservation per successful draw.
  std::uint32_t release_capacity{};              ///< Consumer/CUDA release records.
  std::uint32_t retirement_capacity{};           ///< Bounded recycler queue.
  std::uint32_t quarantine_capacity{};           ///< Bounded safety-hold records.
  LifecycleStage materialize_before{LifecycleStage::kArm};  ///< Latest materialization stage.
  std::optional<Profile> effective_profile;                 ///< Resolved owner profile.
  PlanDisposition disposition{PlanDisposition::kRejected};  ///< Total capability admission.
  bool fallback_rejected{true};                             ///< No allocator/copy fallback.
};

/// @brief Freeze one field-specific allocation capability for a multi-Tensor schema output.
///
/// Records are emitted in ascending FlatBuffers field-id order. Each field owns an independent
/// pool and complete capacity-contributor set; validation remains pool-keyed and order-insensitive.
struct TensorFieldAllocationPlanRecord {
  PortId output{};                               ///< Owning descriptor-schema output.
  std::uint16_t field_id{};                      ///< Canonical FlatBuffers field id.
  std::uint16_t tensor_index{};                  ///< Dense ascending-field-id index.
  PlanGenerationId plan_generation{};            ///< Nonzero enclosing plan identity.
  RunGenerationId run_generation{};              ///< Zero for immutable plan-bound records.
  std::string pool_id;                           ///< Stable field-specific logical pool identity.
  std::string allocator_root_id;                 ///< Stable process/device allocator lineage.
  ProcessGroupId allocator_process_group{};      ///< Compile-admitted process lineage.
  TensorPoolBounds bounds{};                     ///< Field-specific rank and byte-span bounds.
  std::vector<TensorAxisBounds> axes;            ///< Field-specific axis/stride maxima.
  std::optional<TensorRepresentation> layout;    ///< Accepted logical field constraints.
  MemoryKind memory_kind{MemoryKind::kUnknown};  ///< Frozen physical storage class.
  DeviceId device{};                             ///< Shared port device, or invalid for host.
  std::uint32_t slot_count{};                    ///< Peak distinct live slots for this field.
  std::uint64_t aggregate_byte_capacity{};       ///< Checked slots-times-byte-span maximum.
  std::uint32_t max_outstanding_loans{};         ///< Positive producer-loan credit.
  std::uint32_t max_emits_per_compute{};         ///< Positive output burst bound.
  std::uint32_t control_capacity{};              ///< Pre-materialized slot controls.
  std::uint32_t readiness_capacity{};            ///< Readiness records for this field pool.
  std::uint32_t release_capacity{};              ///< Consumer/CUDA release records.
  std::uint32_t retirement_capacity{};           ///< Bounded recycler queue.
  std::uint32_t quarantine_capacity{};           ///< Bounded safety-hold records.
  LifecycleStage materialize_before{LifecycleStage::kArm};  ///< Latest materialization stage.
  std::optional<Profile> effective_profile;                 ///< Resolved owner profile.
  PlanDisposition disposition{PlanDisposition::kRejected};  ///< Total field admission.
  bool fallback_rejected{true};                             ///< No allocator/copy fallback.
};

/// Maximum aggregate field-pool slots admitted for one multi-Tensor output.
inline constexpr std::uint32_t kMaximumDescriptorTensorFieldPoolSlots = 4096U;

/// @brief Freeze one logical Tensor boundary pool and exporter-generation lineage.
struct TensorInterfacePoolPlanRecord {
  ExternalTopicId topic{};  ///< Canonical external Tensor boundary.
  PortId graph_port{};      ///< Plan-local Graph-side boundary port.
  /// Canonical descriptor-schema field id, or no value for a raw Tensor topic.
  std::optional<std::uint16_t> field_id;
  /// Dense descriptor-schema Tensor index, or no value for a raw Tensor topic.
  std::optional<std::uint16_t> tensor_index;
  PlanGenerationId plan_generation{};            ///< Nonzero enclosing plan identity.
  RunGenerationId run_generation{};              ///< Zero for this immutable plan-bound record.
  std::string pool_id;                           ///< Stable endpoint-generation pool identity.
  std::string allocator_root_id;                 ///< Stable process/device allocator root.
  ProcessGroupId allocator_process_group{};      ///< Compile-admitted process lineage.
  std::string exporter_subpool_id;               ///< Stable endpoint exporter-sub-pool lineage.
  TensorBoundaryContractRecord logical;          ///< Complete finite logical Tensor contract.
  MemoryKind memory_kind{MemoryKind::kUnknown};  ///< Frozen accepted storage class.
  DeviceId device{};                             ///< Resolved device, or invalid for host storage.
  std::uint32_t slot_count{};                    ///< Peak distinct live-slot capacity.
  std::uint64_t aggregate_byte_capacity{};       ///< Checked slots-times-logical-span maximum.
  std::uint32_t publisher_loan_capacity{};       ///< Positive publisher-local credit.
  std::uint32_t control_capacity{};              ///< Pre-materialized control/descriptor records.
  std::uint32_t readiness_capacity{};            ///< One readiness token per successful draw.
  std::uint32_t release_capacity{};              ///< Fixed release-callback capacity.
  std::uint32_t retirement_capacity{};           ///< Fixed retirement queue capacity.
  std::uint32_t quarantine_capacity{};           ///< Fixed quarantined-slot capacity.
  InterfaceTensorEmitPolicy emit_policy{
      InterfaceTensorEmitPolicy::kRequireExportable};  ///< Accepted export/copy policy.
  bool imported{};                                     ///< True for external-to-Graph direction.
  std::optional<Profile> effective_profile;            ///< Resolved boundary profile.
};

/// @brief Request or record one exact structured SM allocation.
///
/// The CUDA ordinal and SM count are target-specific deployment facts. An unresolved/negative
/// device or a zero SM count is invalid and is never interpreted as device zero or automatic
/// sizing policy.
struct CudaGreenContextPlacement {
  DeviceId device;           ///< Explicit nonnegative process-local CUDA device ordinal.
  std::uint32_t sm_count{};  ///< Exact positive structured SM count.

  /// @brief Compare exact device and SM-count placement semantics.
  friend constexpr bool operator==(CudaGreenContextPlacement lhs,
                                   CudaGreenContextPlacement rhs) noexcept = default;
};

/// @brief Record the authority that requested a Green Context assignment.
enum class CudaGreenContextAssignmentSource : std::uint8_t {
  kDeveloperBinding = 0,  ///< Exact public deployment binding.
  kAutomaticPolicy = 1,   ///< Reserved for phase-2 automatic partitioning.
  kResidualPolicy = 2,    ///< Compiler-defined residual service assignment.
};

/// @brief Classify one materialized Green Context resource.
enum class CudaGreenContextResourceRole : std::uint8_t {
  kDedicatedPartition = 0,     ///< Dedicated exact manual-partition resource.
  kSharedServiceResidual = 1,  ///< Structured residual used by unbound HSDK CUDA work.
  kReservedResidual = 2,       ///< Structured residual reserved without assigned work.
};

/// @brief Record the admitted SM co-scheduling policy.
enum class CudaGreenContextCoScheduling : std::uint8_t {
  kPreserveDeviceAlignment = 0,  ///< Preserve driver-reported structured alignment.
  kRelaxed = 1,                  ///< Reserved unsupported relaxed policy.
};

/// @brief Record Green Context workqueue ownership scope.
enum class CudaGreenContextWorkqueueScope : std::uint8_t {
  kDeviceContext = 0,         ///< Workqueues remain device-context scoped.
  kGreenContextBalanced = 1,  ///< Reserved future balanced workqueue policy.
};

/// @brief Record the strength of the admitted SM-set guarantee.
enum class CudaGreenContextSmSetGuarantee : std::uint8_t {
  /// CUDA-documented SM set, including documented superset exceptions; no concurrency/QoS claim.
  kDriverDocumented = 0,
  kStrictNoKnownSuperset = 1,  ///< Reserved stronger deployment guarantee.
};

/// @brief Freeze one process/device Green Context pool declaration.
struct CudaGreenContextPoolPlanRecord {
  PlanGenerationId plan_generation{};            ///< Nonzero enclosing plan identity.
  RunGenerationId run_generation{};              ///< Zero for immutable plan-bound evidence.
  ProcessGroupId process_group{};                ///< Runtime process owning the native split.
  DeviceId device{};                             ///< Process-local CUDA ordinal.
  std::string deployment_device_id;              ///< Stable deployment physical-device identity.
  std::string pool_id;                           ///< Stable plan-local pool lineage.
  std::uint32_t device_sm_count{};               ///< Qualified total SM count.
  std::uint32_t min_sm_partition_size{};         ///< Qualified driver minimum.
  std::uint32_t sm_coscheduled_alignment{};      ///< Qualified co-scheduling alignment.
  std::uint32_t structured_sm_quantum{};         ///< LCM of minimum size and alignment.
  std::uint32_t max_structured_context_count{};  ///< Qualified structured-context budget.
  std::uint32_t unprovisioned_sm_count{};        ///< Tail not admitted for HSDK work.
  CudaGreenContextCoScheduling co_scheduling{
      CudaGreenContextCoScheduling::kPreserveDeviceAlignment};  ///< Split policy.
  CudaGreenContextWorkqueueScope workqueue_scope{
      CudaGreenContextWorkqueueScope::kDeviceContext};  ///< Workqueue ownership contract.
  std::uint32_t workqueue_concurrency_limit{};          ///< Zero selects driver default.
  CudaGreenContextSmSetGuarantee sm_set_guarantee{
      CudaGreenContextSmSetGuarantee::kDriverDocumented};   ///< Qualified SM-set strength.
  LifecycleStage materialize_before{LifecycleStage::kArm};  ///< Latest creation stage.
  bool fallback_rejected{true};                             ///< No ordinary-context fallback.

  friend bool operator==(const CudaGreenContextPoolPlanRecord&,
                         const CudaGreenContextPoolPlanRecord&) = default;
};

/// @brief Freeze one context resource produced by a pool split.
struct CudaGreenContextResourcePlanRecord {
  PlanGenerationId plan_generation{};  ///< Nonzero enclosing plan identity.
  RunGenerationId run_generation{};    ///< Zero for immutable plan-bound evidence.
  ProcessGroupId process_group{};      ///< Owning runtime process.
  DeviceId device{};                   ///< Process-local CUDA ordinal.
  std::string pool_id;                 ///< Owning pool lineage.
  std::string green_context_id;        ///< Stable plan-local context lineage.
  std::uint32_t accepted_sm_count{};   ///< Exact admitted structured SM count.
  CudaGreenContextResourceRole role{
      CudaGreenContextResourceRole::kDedicatedPartition};  ///< Resource purpose.

  friend bool operator==(const CudaGreenContextResourcePlanRecord&,
                         const CudaGreenContextResourcePlanRecord&) = default;
};

/// @brief Freeze one accepted or rejected authored Green Context request.
struct CudaGreenContextRequestPlanRecord {
  PlanGenerationId plan_generation{};     ///< Nonzero enclosing plan identity.
  RunGenerationId run_generation{};       ///< Zero for immutable plan-bound evidence.
  ExecutionPartitionId partition{};       ///< Resolved partition, or zero when unknown.
  std::string authored_name;              ///< Deep-owned exact author selector.
  std::uint32_t execution_lane{};         ///< Dense lane identity, or zero if unresolved.
  CudaGreenContextPlacement requested{};  ///< Exact authored target request.
  std::optional<CudaGreenContextPlacement> accepted;  ///< Equal accepted placement, or empty.
  CudaGreenContextAssignmentSource source{
      CudaGreenContextAssignmentSource::kDeveloperBinding};  ///< Request authority.
  PlanDisposition disposition{PlanDisposition::kRejected};   ///< Total admission result.
  DiagnosticId reason{};                                     ///< Stable rejection reason.

  friend bool operator==(const CudaGreenContextRequestPlanRecord&,
                         const CudaGreenContextRequestPlanRecord&) = default;
};

/// @brief Join one execution partition to one admitted Green Context resource.
struct CudaGreenContextAssignmentPlanRecord {
  PlanGenerationId plan_generation{};  ///< Nonzero enclosing plan identity.
  RunGenerationId run_generation{};    ///< Zero for immutable plan-bound evidence.
  ExecutionPartitionId partition{};    ///< Accepted execution partition.
  std::uint32_t execution_lane{};      ///< Dense partition lane identity.
  ProcessGroupId process_group{};      ///< Owning runtime process.
  DeviceId device{};                   ///< Process-local CUDA ordinal.
  std::string pool_id;                 ///< Owning pool lineage.
  std::string green_context_id;        ///< Assigned context lineage.
  CudaGreenContextAssignmentSource source{
      CudaGreenContextAssignmentSource::kDeveloperBinding};  ///< Assignment authority.

  friend bool operator==(const CudaGreenContextAssignmentPlanRecord&,
                         const CudaGreenContextAssignmentPlanRecord&) = default;
};

/// @brief Freeze one process-owned CUDA compute-stream materialization contract.
struct CudaComputeStreamPlanRecord {
  PlanGenerationId plan_generation{};             ///< Nonzero enclosing plan identity.
  RunGenerationId run_generation{};               ///< Zero for immutable plan-bound evidence.
  ProcessGroupId process_group{};                 ///< Owning runtime process.
  DeviceId device{};                              ///< Process-local CUDA ordinal.
  std::string owner_id;                           ///< Stable partition or compiler-service owner.
  std::optional<ExecutionPartitionId> partition;  ///< Partition owner, or empty for a service.
  std::string stream_id;                          ///< Stable plan-local stream lineage.
  std::optional<std::string> green_context_id;    ///< Assigned Green Context, or ordinary stream.
  std::vector<EdgeId> fused_linear_edges;         ///< Canonical same-stream Tensor edges.
  LifecycleStage materialize_before{LifecycleStage::kArm};  ///< Latest creation stage.

  friend bool operator==(const CudaComputeStreamPlanRecord&,
                         const CudaComputeStreamPlanRecord&) = default;
};

/// @brief Freeze one process/device Tensor resource materialization contract.
///
/// The compiler aggregates accepted direct, descriptor-field, and interface
/// Tensor pools by process and explicit device. A cross-process CUDA-device or
/// pinned-host receiver with no local Tensor pool uses a capacity-only record
/// with zero pool and capacity fields. The pinned Tensor's public DLPack device
/// remains zero; this private device selects only its CUDA synchronization
/// context.
/// Compute-stream identity and fusion belong exclusively to
/// \ref CudaComputeStreamPlanRecord. Logical partitions do not multiply these
/// aggregate capacity resources. This record contains no CUDA handle, context,
/// event, allocator object, or Runtime-owned address.
struct TensorDeviceResourcePlanRecord {
  PlanGenerationId plan_generation{};  ///< Nonzero enclosing plan identity.
  RunGenerationId run_generation{};    ///< Zero for immutable plan-bound evidence.
  ProcessGroupId process_group{};      ///< Compile-admitted process ownership.
  DeviceId device{};                   ///< Explicit nonnegative CUDA device identity.
  std::string allocator_root_id;       ///< Stable process/device resource lineage.
  std::uint32_t pool_count{};     ///< Accepted pools, or zero for a receive-only stream resource.
  std::uint32_t slot_capacity{};  ///< Sum of finite distinct-slot capacities.
  std::uint32_t loan_credit_capacity{};  ///< Sum of producer/publisher loan credits.
  std::uint32_t control_capacity{};      ///< Sum of pre-materialized control records.
  std::uint32_t readiness_capacity{};    ///< Sum of readiness reservations.
  std::uint32_t release_capacity{};      ///< Sum of release records.
  std::uint32_t retirement_capacity{};   ///< Sum of retirement records.
  std::uint32_t quarantine_capacity{};   ///< Sum of safety-hold records.
  LifecycleStage materialize_before{LifecycleStage::kArm};  ///< Latest construction stage.
  bool gpu_resident_backend_selected{};  ///< False until a separately admitted backend exists.
  bool fallback_rejected{true};          ///< No implicit device/copy/backend fallback.

  /// @brief Compare every immutable materialization field.
  friend bool operator==(const TensorDeviceResourcePlanRecord&,
                         const TensorDeviceResourcePlanRecord&) = default;
};

/// @brief Freeze one generated-provider inline Tensor record bucket.
///
/// This compile-owned record identifies the single versioned discriminated
/// service representation and its exact bounded inline bucket. It contains no
/// provider object, record storage, payload pointer, descriptor, or runtime
/// availability observation. Runtime shall materialize both recorded arms
/// before \ref LifecycleStage::kArm without changing the admitted byte cap.
struct TensorInlineRecordPlanRecord {
  ExternalTopicId topic{};                ///< Canonical external Tensor boundary.
  std::string service_id;                 ///< Canonical endpoint service compatibility identity.
  std::string fixed_record_bucket_id;     ///< Stable private logical bucket identity.
  std::string provider_id;                ///< Generated package provider identity.
  std::string service_family_id;          ///< One inline/descriptor record service family.
  std::string capability_id;              ///< Exact generated capability identity.
  std::uint32_t capability_version{};     ///< Exact nonzero capability version.
  std::string capability_evidence_id;     ///< Exact package evidence identity.
  std::string record_format_id;           ///< Versioned discriminated record format identity.
  std::uint32_t record_format_version{};  ///< Exact supported record format version.
  std::string inline_discriminator;       ///< Exact inline-arm discriminator token.
  std::string opaque_descriptor_discriminator;        ///< Exact descriptor-arm discriminator token.
  std::uint32_t fixed_metadata_overhead_bytes{};      ///< Version/discriminator/metadata bytes.
  std::uint32_t provider_max_tensor_rank{};           ///< Generated provider metadata rank bound.
  std::uint64_t provider_max_inline_logical_bytes{};  ///< Provider logical-byte maximum.
  std::uint64_t profile_max_inline_logical_bytes{};   ///< Effective profile logical maximum.
  std::uint64_t effective_max_inline_logical_bytes{};  ///< Minimum provider/profile maximum.
  std::uint64_t accepted_max_inline_logical_bytes{};   ///< Exact authored/admitted cap.
  std::uint64_t provider_max_fixed_record_bytes{};     ///< Provider total record maximum.
  std::uint64_t fixed_record_bucket_bytes{};           ///< Checked metadata plus accepted cap.
  std::string inline_arm_schema_id;                    ///< Exact inline-arm schema evidence.
  std::string opaque_descriptor_arm_schema_id;         ///< Exact descriptor-arm schema evidence.
  bool inline_arm_materialized{true};                  ///< Fixed inline arm is required before arm.
  bool descriptor_arm_materialized{true};    ///< Fixed opaque descriptor arm is also required.
  bool fallback_rejected{true};              ///< No truncation, alias, or pressure fallback.
  std::optional<Profile> effective_profile;  ///< Profile used for maximum admission.
  PlanDisposition disposition{PlanDisposition::kRejected};  ///< Total provider admission.
};

/// @brief Freeze finite imported endpoint-generation retention and ABA policy.
struct ImportedGenerationCapacityPlanRecord {
  ExternalTopicId topic{};                         ///< Canonical external import boundary.
  std::string runtime_scope_id;                    ///< Stable Runtime scope identity.
  std::string endpoint_or_service_id;              ///< Stable endpoint/service identity.
  std::uint32_t endpoint_max_generations{};        ///< Positive endpoint-local maximum.
  std::uint32_t service_max_generations{};         ///< Positive service-wide maximum.
  std::uint32_t runtime_max_generations{};         ///< Positive Runtime aggregate contribution.
  std::uint32_t candidate_reservation_capacity{};  ///< Zero or one unpublished candidate.
  std::uint32_t release_record_capacity{};         ///< Fixed claim-free release records.
  std::uint32_t quarantine_record_capacity{};      ///< Fixed safety-hold records.
  std::optional<Profile> effective_profile;        ///< Profile used for restart admission.
  PeerRequirement peer_requirement{PeerRequirement::kOptional};  ///< Frozen attach requirement.
  std::optional<std::uint64_t> attach_max_wait_ns;  ///< Positive finite attach bound, if any.
  bool restart_capable_minimum_two{};               ///< Whether the minimum-two rule applies.
  bool no_wrap_aba_generation{true};                ///< Exhaustion is terminal, never wrap/reuse.
  bool fallback_rejected{true};                     ///< No eviction, wait, rollover, or retry.
};

/// @brief Classify one compile-visible Tensor source family at an external boundary.
///
/// The class is logical admission evidence, not a pointer, allocator, backend
/// descriptor, or claim of an observed runtime Tensor. Values have stable plan
/// schema order and must not be renumbered.
enum class TensorSourceClass : std::uint8_t {
  kOrdinaryHostContiguous = 0,  ///< Pageable host storage with canonical contiguous layout.
  kOrdinaryHostStrided = 1,     ///< Pageable host storage requiring bounded materialization.
  kPinnedHost = 2,              ///< Page-locked host storage; never inline-small-host.
  kCudaDevice = 3,              ///< CUDA device storage with separately resolved device identity.
  kCudaManaged = 4,             ///< CUDA managed storage with separately admitted movement.
  kImportedDescriptor = 5,      ///< Storage already owned through an imported read-only lease.
  kAlreadyExportable = 6,       ///< Source proven compatible with the accepted descriptor service.
};

/// @brief Identify one successful external Tensor transfer representation.
///
/// These values name only successful realizations. Resource, validation, and
/// publication failures remain separate typed outcomes.
enum class TensorTransferRealization : std::uint8_t {
  kInline = 0,                  ///< Copy host-ready logical bytes into the bounded inline arm.
  kCopiedToExportablePool = 1,  ///< Copy into a plan-owned exportable destination pool.
  kZeroCopyExportable = 2,      ///< Publish an already compatible exportable descriptor.
};

/// @brief Freeze one source-class-to-realization admission cell for a Tensor topic.
///
/// One or more records are emitted for every closed source class. When byte
/// size changes the realization, records partition the complete finite logical
/// byte domain into non-overlapping ranges. An accepted record names the only
/// realization permitted for that cell; a rejected record proves that runtime
/// must return a typed failure instead of selecting a fallback. Every delivered
/// external Tensor is read-only under the Topic 31 lease contract regardless of
/// whether its backing bytes were copied.
struct TensorTransferPlanRecord {
  ExternalTopicId topic{};  ///< Canonical external Tensor boundary.
  /// Canonical descriptor-schema field id, or no value for a raw Tensor topic.
  std::optional<std::uint16_t> field_id;
  /// Dense descriptor-schema Tensor index, or no value for a raw Tensor topic.
  std::optional<std::uint16_t> tensor_index;
  /// Process group that owns this transfer's copy-pool attachment, when one is required.
  ProcessGroupId interface_process_group{};
  /// Resolved CUDA copy-pool device, invalid for ordinary host or placement-free transfers.
  DeviceId interface_device{};
  TensorSourceClass source_class{TensorSourceClass::kOrdinaryHostContiguous};
  std::optional<TensorTransferRealization> realization;  ///< Exact success arm when admitted.
  std::uint64_t min_logical_bytes{};  ///< Inclusive logical byte lower bound; zero is valid.
  std::uint64_t max_logical_bytes{};  ///< Inclusive positive finite logical byte upper bound.
  std::string evidence_id;            ///< Stable capability or rejection-evidence identity.
  std::string service_owner_id;       ///< Stable CUDA transfer-service stream lineage, if needed.
  bool source_must_be_host_ready{};   ///< Inline performs no wait or async copy.
  bool source_must_be_exportable{};   ///< Zero-copy requires compatible exportability evidence.
  bool source_retained_until_materialization{};  ///< Source claim outlives inline/copy completion.
  bool delivered_read_only{true};        ///< Imported/selected publication never grants write.
  bool requires_inline_record{};         ///< Fixed-capacity inline arm must be materialized.
  bool requires_exportable_copy_pool{};  ///< Fixed-capacity copy pool must be materialized.
  PlanDisposition disposition{PlanDisposition::kRejected};  ///< Total admission for this cell.
};

/// @brief Classify unpublished authority after a proven pre-submit failure.
enum class PreSubmitAuthorityDisposition : std::uint8_t {
  kRetainedSameActivation = 0,  ///< Generic loan may be explicitly corrected/resubmitted.
  kConsumedToRetirement = 1,    ///< Tensor loan attempt consumes public authority.
};

/// @brief Freeze publication-boundary authority and retry semantics for one output.
struct PublicationOutcomePlanRecord {
  PortId output{};  ///< Canonical publishing output.
  bool tensor{};    ///< Whether Tensor consume-on-attempt rules apply.
  ErrorCode pre_submit_error{ErrorCode::kBackpressured};  ///< Proven zero-delivery result.
  ErrorCode post_submit_error{
      ErrorCode::kPublicationIndeterminate};  ///< Unknown-visibility post-submit result.
  ErrorCode post_submit_partial_error{
      ErrorCode::kPublicationPartial};  ///< Known strict-subset post-submit result.
  PreSubmitAuthorityDisposition pre_submit_authority{
      PreSubmitAuthorityDisposition::kRetainedSameActivation};  ///< Unsubmitted authority.
  bool pre_submit_assigns_sequence{};                           ///< Always false.
  bool post_submit_assigns_sequence{true};    ///< Submission consumes one sequence.
  bool post_submit_consumes_authority{true};  ///< No retry after uncertainty.
  bool retain_claims_until_retirement{true};  ///< Preserve applicable canonical claims.
  bool runtime_retry_enabled{};               ///< Always false in baseline.
};

/// @brief Report whether profile admission accepted the normalized plan.
enum class ProfileAdmissionDisposition : std::uint8_t {
  kAccepted = 0,  ///< Every effective scope satisfies the selected policy.
  kRejected = 1,  ///< At least one scope or capability violates policy.
};

/// @brief Identify whether one region profile was inherited or explicit.
enum class RegionProfileOrigin : std::uint8_t {
  kInheritedGraph = 0,  ///< The region inherited the root or parent scope profile.
  kExplicitRegion = 1,  ///< The region explicitly requested its effective profile.
};

/// @brief Identify the versioned profile policy used during admission.
struct ProfilePolicyId {
  std::uint32_t major{};  ///< Policy major version.
  std::uint32_t minor{};  ///< Policy minor version.
  PlanDigest digest{};    ///< Digest of canonical policy data.
};

/// @brief Preserve one region's requested and effective profile.
struct RegionProfileRecord {
  RegionId region_id{};                      ///< Canonical region identity.
  std::optional<Profile> requested_profile;  ///< Explicit request, when present.
  std::optional<Profile> effective_profile;  ///< Resolved profile, absent on invalid input.
  RegionProfileOrigin origin{RegionProfileOrigin::kInheritedGraph};  ///< Request provenance.
};

/// @brief Preserve the effective profile and contributing regions for one plan object.
struct ProfileScopeResolutionRecord {
  std::string object_id;                          ///< Owned canonical plan object identity.
  std::string object_kind;                        ///< Closed schema-v1 object-kind token.
  Profile effective_profile{Profile::kStandard};  ///< Resolved effective profile.
  std::vector<RegionId> contributing_regions;     ///< Sorted explicit contributors only.
};

/// @brief Classify one profile-controlled feature outcome.
enum class ProfileFeatureDisposition : std::uint8_t {
  kAdmitted = 0,    ///< Feature and required evidence are accepted.
  kRestricted = 1,  ///< Feature is admitted only with recorded restrictions.
  kDisabled = 2,    ///< Feature is explicitly absent from the accepted plan.
  kRejected = 3,    ///< Feature causes profile admission failure.
};

/// @brief Record one exact executing-package capability requirement.
struct ProfileCapabilityRequirementRecord {
  std::string capability_id;  ///< Stable capability identity.
  std::uint32_t version{};    ///< Exact required version; no numeric-newer inference.
  std::string evidence_id;    ///< Stable package/plan evidence reference.
};

/// @brief Record one stable evidence identity referenced by profile decisions.
struct ProfileEvidenceReferenceRecord {
  std::string id;           ///< Stable evidence identity.
  std::string kind;         ///< Canonical bounded evidence-kind token.
  std::uint32_t version{};  ///< Exact evidence schema version.
};

/// @brief Record one total profile feature decision.
struct ProfileFeatureDispositionRecord {
  std::string feature_id;  ///< Stable plan-scoped feature identity.
  ProfileFeatureDisposition disposition{ProfileFeatureDisposition::kRejected};  ///< Outcome.
  std::string rule_id;      ///< Versioned policy rule that produced the outcome.
  std::string evidence_id;  ///< Referenced evidence; empty only when none is required.
};

/// @brief Record one compiler-derived finite capacity bound.
struct ProfileDerivedCapacityRecord {
  std::string object_id;    ///< Canonical owning plan object.
  std::string capacity_id;  ///< Stable capacity dimension.
  std::uint64_t limit{};    ///< Finite accepted upper bound.
  std::string unit;         ///< Canonical unit such as count, bytes, or nanoseconds.
};

/// @brief Record one resource that Runtime must materialize before a lifecycle stage.
struct ProfileMaterializationObligationRecord {
  std::string object_id;    ///< Canonical owning plan object or resource.
  std::string strategy_id;  ///< Stable materialization strategy identity.
  LifecycleStage complete_before_stage{LifecycleStage::kConfigure};  ///< Deadline barrier.
  std::uint8_t cleanup_stage_mask{0xC0U};  ///< Required bounded `kStop`/`kRelease` unwind stages.
  std::uint32_t max_cleanup_actions{1U};   ///< Finite dependency-ordered cleanup action bound.
  std::string evidence_id;                 ///< Stable plan evidence proving the finite obligation.
};

/// @brief Record one policy-authorized explicit fallback.
struct ProfileFallbackRecord {
  std::string feature_id;   ///< Feature receiving the fallback.
  std::string fallback_id;  ///< Stable bounded fallback behavior.
  std::string reason_id;    ///< Stable policy reason.
  std::string evidence_id;  ///< Stable evidence supporting the fallback.
};

/// @brief Identify the authority that admitted one generated endpoint name.
enum class GeneratedEndpointNameAdmissionSource : std::uint8_t {
  kNone = 0,                 ///< No exact deployment authority matched.
  kSuppliedCatalog = 1,      ///< The caller-supplied immutable catalog matched.
  kDeploymentAllowlist = 2,  ///< An exact typed DeploymentBinding pin matched.
};

/// @brief Report generated endpoint-name admission in the profile manifest.
struct GeneratedEndpointNameAdmissionRecord {
  std::string final_name;   ///< Exact plan-owned client lookup identity.
  std::string algorithm;    ///< Stable generation algorithm identity.
  std::uint32_t version{};  ///< Exact observable algorithm version.
  GeneratedEndpointNameAdmissionSource source{GeneratedEndpointNameAdmissionSource::kNone};
  ProfileFeatureDisposition disposition{ProfileFeatureDisposition::kRejected};
  std::string evidence_id;  ///< Stable matched-authority reference.
};

/// @brief Own the minimum immutable profile-admission evidence for an execution plan.
struct ProfileManifest {
  ProfileAdmissionDisposition disposition{
      ProfileAdmissionDisposition::kRejected};         ///< Total profile admission result.
  std::optional<Profile> requested_profile;            ///< Valid root request, when known.
  std::optional<Profile> effective_graph_profile;      ///< Effective root policy when resolved.
  std::vector<RegionProfileRecord> effective_regions;  ///< Canonical region records.
  std::vector<ProfileScopeResolutionRecord> scope_resolutions;  ///< Canonical object scopes.
  ProfilePolicyId profile_policy_id{};                          ///< Versioned policy authority.
  std::string plan_schema_version;  ///< Stable plan schema identifier/version.
  std::string runtime_abi_version;  ///< Required Runtime ABI identity.
  std::string target_triple;        ///< Compile-target identity used for admission.
  std::string producer_build_id;    ///< Diagnostic-only producer build identity.
  std::vector<ProfileCapabilityRequirementRecord> required_capabilities;
  std::vector<ProfileEvidenceReferenceRecord> evidence_references;
  std::vector<ProfileFeatureDispositionRecord> feature_dispositions;
  std::vector<ProfileDerivedCapacityRecord> derived_capacities;
  std::vector<ProfileMaterializationObligationRecord> materialization_obligations;
  std::vector<ProfileFallbackRecord> explicit_fallbacks;
  std::vector<GeneratedEndpointNameAdmissionRecord> generated_endpoint_names;
};

/// @brief Describe one clock declaration in the immutable plan-local catalog.
///
/// The numeric \ref id is meaningful only together with the enclosing
/// \ref ClockCatalogView plan generation. The record owns its cold key and
/// contains no live clock, factory, registry pointer, or backend handle.
struct ClockPlanRecord {
  ClockId id{};           ///< Dense nonzero identity within this plan catalog.
  GraphId graph{};        ///< Canonical owning Graph scope.
  ClockKey key;           ///< Owned authored name; never the omitted/default key.
  bool shared{};          ///< True only for explicit shared-clock authoring.
  bool is_default{};      ///< True for the default clock of the owning Graph.
  bool can_read_now{};    ///< Whether immutable capability evidence admits current-time reads.
  bool can_wait_until{};  ///< Whether scheduler infrastructure may wait until a coordinate.
  bool steady{};          ///< Whether timestamps are proven monotonic within a clock generation.
  bool independent_progress{};      ///< Whether time advances without an external test driver.
  bool finite_deadline{};           ///< Whether finite wait deadlines are supported and bounded.
  bool external_driver_admitted{};  ///< Whether a non-independent clock has an admitted driver.
  std::string
      capability_evidence_id;  ///< Stable evidence identity for the complete capability set.
};

/// @brief Retain one immutable plan-local clock catalog view.
///
/// Copies share ownership of the complete immutable plan state, so \ref clocks
/// remains valid after the ExecutionPlan that produced this value is destroyed.
/// Persisted, trace, and transport records must still retain
/// \ref plan_generation before interpreting any numeric \ref ClockId from this
/// catalog. The view exposes no live clock, registry, or mutation authority.
class HOLOSCAN_CORE_EXPORT ClockCatalogView {
 public:
  /// @brief Create an empty non-authorizing catalog view.
  ClockCatalogView() noexcept = default;

  PlanGenerationId plan_generation{};       ///< Nonzero enclosing plan identity.
  ClockId root_default_clock{};             ///< Root Graph default, or invalid when absent.
  std::span<const ClockPlanRecord> clocks;  ///< Dense ClockId assignment order.

 private:
  ClockCatalogView(PlanGenerationId generation, ClockId root_default,
                   std::span<const ClockPlanRecord> records,
                   std::shared_ptr<const detail::ExecutionPlanState> owner) noexcept
      : plan_generation(generation),
        root_default_clock(root_default),
        clocks(records),
        owner_(std::move(owner)) {}

  std::shared_ptr<const detail::ExecutionPlanState> owner_;
  friend class ExecutionPlan;
};

/// @brief Describe one admitted direct affine conversion between clock domains.
///
/// baseline records only explicitly authored directions. It never synthesizes an
/// inverse or multi-hop conversion.
struct ClockConversionRecord {
  std::uint64_t id{};                  ///< Dense nonzero conversion identity.
  PlanGenerationId plan_generation{};  ///< Enclosing clock-catalog identity.
  GraphId graph{};                     ///< Canonical declaration scope.
  ClockId from{};                      ///< Direct source domain.
  ClockId to{};                        ///< Direct destination domain.
  ClockRateRatio scale{};              ///< Positive reduced exact ratio.
  std::int64_t offset_ns{};            ///< Signed target-domain offset.
  /// Deterministic integer materialization policy.
  ClockRoundingMode rounding{ClockRoundingMode::kNearestTiesToEven};
  /// Checked conversion-overflow policy.
  ClockOverflowPolicy overflow{ClockOverflowPolicy::kError};
};

/// @brief Identify the plan-owned epoch that qualifies endpoint health state.
enum class EndpointProducerEpochSource : std::uint8_t {
  kNotApplicable = 0,         ///< Ordinary endpoints allocate no health epoch state.
  kAttachmentGeneration = 1,  ///< Endpoint attachment/reconstruction advances the epoch.
};

/// @brief Identify the origin of one finite private clock-health resource bound.
enum class EndpointClockHealthBoundOrigin : std::uint8_t {
  kNotApplicable = 0,    ///< Ordinary endpoints allocate no report resources.
  kPortableMinimum = 1,  ///< Core selected the documented portable capacity of one.
};

/// @brief Classify the private monitor-to-publisher handoff contract.
enum class EndpointClockHealthMonitorPathKind : std::uint8_t {
  kNotApplicable = 0,        ///< Ordinary endpoints have no monitor path.
  kPublisherSerialized = 1,  ///< The integration serializes reports with publisher data calls.
};

/// @brief Freeze the consumer action for one non-nominal clock-health state.
enum class EndpointClockHealthDisposition : std::uint8_t {
  kNotApplicable = 0,  ///< Clock health does not gate this endpoint.
  kSkip = 1,           ///< Suppress the affected candidate without invoking user compute.
  kDegraded = 2,       ///< Admit only the plan's explicitly degraded compute path.
  kFailRun = 3,        ///< Terminate the affected run through typed failure handling.
};

/// @brief Freeze the selected profile rule for `ClockHealthReason::kUnspecified`.
enum class ClockHealthUnspecifiedDisposition : std::uint8_t {
  kAdmitted = 0,       ///< Accept the frozen reason under the selected profile.
  kQualifiedOnly = 1,  ///< Accept only with the plan-recorded qualification evidence.
  kRejected = 2,       ///< Reject the reason under the selected profile.
};

/// @brief Freeze behavior when required structural health evidence is lost.
enum class EndpointClockHealthTraceLossDisposition : std::uint8_t {
  kNotApplicable = 0,  ///< Ordinary endpoints produce no clock-health evidence.
  kFailClosed = 1,     ///< Required loss makes execution/replay decision-incomplete.
};

/// @brief Preserve one external endpoint's immutable clock-health contract and bounds.
///
/// One record exists for every external endpoint. Ordinary records explicitly
/// contain `kUnavailable` plus `kNotRequired` and allocate no report state.
/// Monitored records bind every numeric clock identity to `plan_generation`,
/// own only cold policy/evidence, and contain no current assessment, acceptance
/// time, expiry, backend object, or mutable Runtime state. Copies of an
/// ExecutionPlan share immutable storage and are safe for concurrent reads.
struct EndpointClockHealthPlanRecord {
  PortId endpoint{};                         ///< Dense plan-local boundary endpoint.
  PlanGenerationId plan_generation{};        ///< Provenance for every numeric clock identity.
  ClockId capture_clock{};                   ///< Catalog-resolved capture clock, or invalid.
  std::uint64_t capture_conversion_id{};     ///< Direct capture/reference conversion, or zero.
  PlanGenerationId conversion_generation{};  ///< Conversion catalog generation, or zero.
  EndpointProducerEpochSource producer_epoch_source{
      EndpointProducerEpochSource::kNotApplicable};  ///< Health epoch/reset authority.

  ClockHealthCapability capability{ClockHealthCapability::kUnavailable};  ///< Static capability.
  ClockHealthRequirement requirement{
      ClockHealthRequirement::kNotRequired};               ///< Static evidence requirement.
  std::optional<std::chrono::nanoseconds> max_report_age;  ///< Positive monitored freshness bound.
  ClockId health_age_clock{};  ///< Independently progressing monotonic Runtime clock.

  std::uint32_t fixed_report_ingress_capacity{};  ///< Preallocated pending-report capacity.
  std::uint32_t fixed_update_capacity{};  ///< Preallocated structural health-update capacity.
  EndpointClockHealthBoundOrigin ingress_capacity_origin{
      EndpointClockHealthBoundOrigin::kNotApplicable};  ///< Capacity derivation source.
  std::uint32_t per_dispatch_report_budget{};   ///< Maximum reports applied per Dispatcher pass.
  std::string publisher_operation_sequence_id;  ///< Report/data causal order schema identity.
  std::string report_sequence_id;               ///< Accepted report-order schema identity.
  std::string health_update_sequence_id;        ///< Receiver state-update schema identity.

  EndpointClockHealthMonitorPathKind monitor_path_kind{
      EndpointClockHealthMonitorPathKind::kNotApplicable};     ///< Handoff protocol category.
  std::optional<std::chrono::nanoseconds> max_monitor_period;  ///< Qualified cadence bound.
  std::optional<std::chrono::nanoseconds>
      max_observation_to_call_delay;  ///< Qualified observation/handoff/call bound.
  std::optional<std::chrono::nanoseconds>
      max_report_delivery_delay;  ///< Qualified call-to-receiver bound.
  std::optional<std::chrono::nanoseconds> admitted_jitter_margin;  ///< Qualified jitter bound.
  std::string monitor_evidence_reference;  ///< Stable qualification evidence identity.

  EndpointClockHealthDisposition no_report_disposition{
      EndpointClockHealthDisposition::kNotApplicable};  ///< Initial/reset state action.
  EndpointClockHealthDisposition degraded_disposition{
      EndpointClockHealthDisposition::kNotApplicable};  ///< Fresh degraded assessment action.
  EndpointClockHealthDisposition unhealthy_disposition{
      EndpointClockHealthDisposition::kNotApplicable};  ///< Fresh unhealthy assessment action.
  EndpointClockHealthDisposition stale_disposition{
      EndpointClockHealthDisposition::kNotApplicable};  ///< Expired evidence action.
  ClockHealthUnspecifiedDisposition unspecified_reason_disposition{
      ClockHealthUnspecifiedDisposition::kAdmitted};  ///< Selected profile rule.
  std::string clock_health_update_witness_id;         ///< Stable structural witness identity.
  EndpointClockHealthTraceLossDisposition trace_loss_disposition{
      EndpointClockHealthTraceLossDisposition::kNotApplicable};  ///< Missing evidence behavior.
};

/// @brief Record one canonical logical readiness source and its finite admission facts.
struct ReadinessPlanRecord {
  SourceId source_id{};                 ///< Nonzero plan-local source identity.
  PlanGenerationId plan_generation{};   ///< Enclosing clock-catalog identity.
  VertexId owner{};                     ///< Canonical owning Operator vertex.
  std::string owner_path;               ///< Fully qualified canonical Graph/Operator path.
  std::uint32_t declaration_ordinal{};  ///< Stable source order within the owning Operator.
  ReadinessSourceKind kind{ReadinessSourceKind::kReadable};  ///< Logical source family.
  std::string name;                          ///< Owned declaration-local source name.
  std::optional<Profile> effective_profile;  ///< Resolved owner profile, or absent on rejection.
  ClockKey requested_clock;                  ///< Deep-owned cold authoring request.
  ClockId resolved_clock{};                  ///< Dense plan/catalog-local clock domain.
  std::uint32_t requested_capacity{};        ///< Exact authored bounded witness capacity.
  std::uint32_t requested_sender_reference_capacity{};  ///< Exact authored sender bound.
  ReadinessWakeRequirement wake_requirement{
      ReadinessWakeRequirement::kDeploymentSelected};  ///< Authored logical wake requirement.
  std::uint32_t accepted_capacity{};                   ///< Positive admitted witness capacity.
  std::uint32_t accepted_sender_reference_capacity{};  ///< Sender-reference capacity.
  bool referenced{};                                   ///< Whether a Contract consumes the source.
  bool materialized{};  ///< Whether Runtime must allocate an adapter; false when unused.
  std::uint64_t initial_source_generation{1};  ///< Initial lifecycle source generation.
  std::uint64_t initial_source_sequence{};     ///< Zero before the first acceptance.
  DiagnosticId disposition_reason{};           ///< Stable admission/materialization reason.
};

/// @brief Classify one edge in a lifecycle-stage dependency DAG.
enum class LifecycleEdgeKind : std::uint8_t {
  kBefore = 0,         ///< Direct Graph-level \c before edge.
  kDependencyKey = 1,  ///< Edge resolved from produces/requires declarations.
};

/// @brief Describe one operator's logical checkpoint at one lifecycle stage.
struct LifecycleCheckpointPlanRecord {
  OperatorRef operator_instance{};                   ///< Plan-generation-stamped owner.
  LifecycleStage stage{LifecycleStage::kConfigure};  ///< Global stage barrier.
  bool callback_present{};                   ///< False denotes a logical skipped checkpoint.
  bool required{};                           ///< Whether callback absence rejects admission.
  std::uint32_t stable_topological_level{};  ///< Deterministic level within the stage.
};

/// @brief Describe one validated same-stage lifecycle dependency.
struct LifecycleDependencyPlanRecord {
  OperatorRef before{};                                ///< Checkpoint that must complete first.
  OperatorRef after{};                                 ///< Checkpoint permitted to follow.
  LifecycleStage stage{LifecycleStage::kConfigure};    ///< Shared dependency stage.
  LifecycleEdgeKind kind{LifecycleEdgeKind::kBefore};  ///< Direct or key-derived origin.
  std::string dependency_key;  ///< Owned stable key; empty for direct edges.
};

/// @brief Record one deny-by-default compute cross-target restart authorization.
struct LifecycleRestartPermissionPlanRecord {
  OperatorRef requester{};  ///< Plan-bound requesting Operator.
  OperatorRef target{};     ///< Plan-bound authorized target.
};

/// @brief Report compile disposition for one operator's active restart target.
enum class LifecycleRestartTargetDisposition : std::uint8_t {
  kDisabled = 0,      ///< Active lifecycle transactions were not enabled.
  kNotRequested = 1,  ///< Enabled globally but no active origin targets this Operator.
  kAdmitted = 2,      ///< Requested origins/stages passed admission.
  kRejected = 3,      ///< Requested behavior failed admission.
};

/// @brief Record the complete runtime restriction baseline for one restart target.
struct LifecycleRestartTargetPlanRecord {
  OperatorRef target{};                      ///< Plan-bound target; run generation is zero.
  std::string effective_scope_id;            ///< Owned canonical Operator scope identity.
  std::optional<Profile> effective_profile;  ///< Resolved target profile, or absent on rejection.
  bool self_allowed{};                       ///< Compute-self restart authority.
  bool external_control_allowed{};           ///< Run control authority.
  bool parameter_origin_allowed{};           ///< Lifecycle-mutable field authority.
  std::uint8_t allowed_from_stage_mask{};    ///< Bits for configure through start.
  LifecycleRestartTargetDisposition disposition{
      LifecycleRestartTargetDisposition::kDisabled};  ///< Total target disposition.
  ReasonId disposition_reason{};                      ///< Stable bounded disposition reason.
};

/// @brief Explain why compilation widened an explicit restart toward \c kConfigure.
///
/// Numeric values are schema-v1 plan values and must not be reordered.
enum class LifecycleRestartWideningReason : std::uint8_t {
  kNone = 0,             ///< The requested stage already covers the complete closure.
  kDependencyStage = 1,  ///< A closure dependency requires an earlier forward stage.
};

/// @brief Resolve one valid requested forward stage for one restart target.
struct LifecycleRestartStageResolutionPlanRecord {
  LifecycleStage requested_stage{LifecycleStage::kConfigure};  ///< Exact request alternative.
  LifecycleStage effective_stage{LifecycleStage::kConfigure};  ///< Same or earlier admitted stage.
  LifecycleRestartWideningReason reason{
      LifecycleRestartWideningReason::kNone};         ///< Stable widening explanation.
  std::uint32_t required_structural_event_records{};  ///< Retained event delta per request.
  std::uint32_t required_quiesce_records{};   ///< Exact callback-bearing member journal bound.
  std::uint32_t required_rollback_records{};  ///< Exact present forward-callback journal bound.
};

/// @brief Freeze one target's complete dependency closure and stage-widening table.
///
/// Records are plan-generation-local, sorted by target identity, and deep-own the
/// closure in increasing dense Operator identity. Exactly six stage resolutions
/// appear in configure-through-start order. This record grants no restart
/// authority; callers must intersect it with the matching
/// LifecycleRestartTargetPlanRecord. Copies of ExecutionPlan share immutable
/// ownership and concurrent read-only access is safe.
struct LifecycleRestartClosurePlanRecord {
  OperatorRef target{};                       ///< Plan-bound target with zero run generation.
  std::vector<OperatorRef> closure;           ///< Nonempty sorted target/dependent set.
  std::uint8_t valid_requested_stage_mask{};  ///< Bits 0..5 for configure through start.
  std::array<LifecycleRestartStageResolutionPlanRecord, 6> stage_resolutions{};
  std::uint32_t required_fenced_operators{};  ///< Exact closure membership bound.
  std::uint32_t required_quiesce_records{};   ///< Exact maximum across six requested stages.
  std::uint32_t required_rollback_records{};  ///< Exact maximum across six requested stages.
};

/// @brief Record one lifecycle-mutable field's earliest invalidated stage.
struct LifecycleMutableFieldPlanRecord {
  OperatorRef owner{};                                    ///< Plan-bound owning Operator.
  ParamFieldId field{};                                   ///< Stable schema field identity.
  LifecycleStage from_stage{LifecycleStage::kConfigure};  ///< Earliest forward stage.
};

/// @brief Group exact finite lifecycle transaction capacities.
struct LifecycleTransactionCapacityValues {
  std::uint32_t max_pending_requests{};       ///< Accepted submissions before terminalization.
  std::uint32_t max_pending_transactions{};   ///< Active plus successor transaction bound.
  std::uint32_t max_receipts{};               ///< Retained per-submission receipt bound.
  std::uint32_t max_fenced_operators{};       ///< Maximum union-closure membership.
  std::uint32_t max_candidate_generations{};  ///< Private complete candidate bound.
  std::uint32_t max_events{};                 ///< Structural event retention capacity.
  std::uint32_t max_audit_records{};          ///< Required bounded audit capacity.
  std::uint32_t max_quiesce_records{};        ///< Quiesce-obligation journal bound.
  std::uint32_t max_rollback_records{};       ///< Completed-work rollback journal bound.
};

/// @brief Report the resolved Parameter active-update policy.
enum class ParamRuntimeUpdatePolicy : std::uint8_t {
  kProfileDefault = 0,       ///< Resolve from the selected profile.
  kDisabled = 1,             ///< Disable active runtime updates.
  kEnabledIfAdmitted = 2,    ///< Enable only when every required proof is admitted.
  kRequirePreallocated = 3,  ///< Require bounded preconstructed generation storage.
};

/// @brief Preserve configured and compiler-derived lifecycle capacities.
struct LifecycleTransactionCapacityPlanRecord {
  LifecycleTransactionCapacityValues configured{};        ///< Exact accepted option values.
  LifecycleTransactionCapacityValues required_minimum{};  ///< Derived finite minima.
  ParamRuntimeUpdatePolicy resolved_parameter_update_policy{
      ParamRuntimeUpdatePolicy::kProfileDefault};          ///< Resolved Parameter policy.
  bool parameter_origin_enabled{};                         ///< Whether fields may submit restarts.
  std::uint32_t required_generation_slots_per_operator{};  ///< Candidate/rollback slots.
  std::uint64_t required_parameter_arena_bytes{};          ///< Parameter arena minimum.
  std::uint32_t required_parameter_audit_records{};        ///< Parameter audit minimum.
  ReasonId derivation_reason{};                            ///< Stable derivation reason.
};

/// @brief Identify the plan-owned finite generation-storage topology.
enum class ParameterArenaStrategy : std::uint8_t {
  kSeparatePerOperator = 0,  ///< Allocate one independently aligned arena per enabled Operator.
};

/// @brief Describe one pointer-free parameter field in an immutable plan.
///
/// The record owns every string and contains no member token, operation
/// pointer, aggregate address, or value. It is interpreted only with the
/// enclosing plan generation and matching \ref OperatorParameterPlanRecord.
struct ParameterFieldPlanRecord {
  OperatorRef owner{};         ///< Plan-bound Operator owning the declared field.
  ParamInstanceId instance{};  ///< Collision-checked plan-scoped field-instance identity.
  ParamSchemaHash
      instance_identity_digest{};  ///< Complete Operator-path-plus-field identity digest.
  ParamFieldId field{};            ///< Stable schema-lineage-local field identity.
  ParamSchemaHash full_id{};       ///< Complete canonical field identity.
  std::string canonical_name;      ///< Owned canonical structured field path.
  ParamLogicalType logical_type{ParamLogicalType::kObject};  ///< Portable logical family.
  ParamFieldTypeShape type_shape{};  ///< Frozen portable logical shape and scalar width.
  ParamMutability mutability{ParamMutability::kCompileTime};  ///< Declared update boundary.
  LifecycleStage earliest_invalidated_stage{
      LifecycleStage::kConfigure};  ///< Earliest forward stage invalidated by a change.
  ParamPersistence persistence{ParamPersistence::kPersisted};       ///< Durable projection policy.
  ParamVisibility visibility{ParamVisibility::kPublic};             ///< Ordinary discovery policy.
  ParamExternalAccess external_access{ParamExternalAccess::kNone};  ///< External eligibility.
  ParamSensitivity sensitivity{ParamSensitivity::kNormal};          ///< Redaction policy.
  bool required_on_load{};         ///< Whether complete import requires an archived value.
  std::string alias;               ///< Prior canonical input spelling; never output identity.
  std::string validator_id;        ///< Stable validator identity, or empty.
  ParamConstraints constraints{};  ///< Portable declared bounds and choices.
  ParamStringEncoding string_encoding{ParamStringEncoding::kArbitraryBytes};
  ParamStringEncoding element_string_encoding{ParamStringEncoding::kArbitraryBytes};
  ParamValueRouteDisposition value_route{
      ParamValueRouteDisposition::kUnsupported};  ///< Frozen generic value route.
  ParamStorageTraitsSummary storage{};            ///< Pointer-free intrinsic storage evidence.
  ParameterMaterializationStrategy materialization{
      ParameterMaterializationStrategy::kRejected};  ///< Aggregate-backed field strategy.
  params::ParamProjectionStatus projection{
      params::ParamProjectionStatus::kExcludedUnsupportedCodec};  ///< Public projection outcome.
  ParameterFieldUpdateDisposition update_disposition{
      ParameterFieldUpdateDisposition::kPolicyDisabled};  ///< Total active-update disposition.
  bool adapter_eligible{};  ///< Whether stable identity/type/codec policy permits an adapter key.
};

/// @brief Identify one compile-resolved temporal term derived from a Parameter field.
enum class ParameterDerivedTermKind : std::uint8_t {
  kLatestWithinMaxAge = 0,  ///< Inclusive scalar selector age for one input.
  kContractDeadline = 1,    ///< Latest permitted compute-start budget for one Operator.
};

/// @brief Freeze whether a derived term can follow later Parameter generations.
enum class ParameterDerivedTermUpdateDisposition : std::uint8_t {
  kRuntimeAtomic = 0,      ///< Runtime may atomically publish a bounded derived-term update.
  kFrozenCompileTime = 1,  ///< The term is fixed to generation zero for this plan.
  kRecompileRequired = 2,  ///< A field change requires a newly compiled plan.
};

/// @brief Describe one pointer-free bounded duration term derived during compilation.
///
/// The compiler resolves the process-local \ref ParamRef authoring token against the owning
/// Operator factory, validates exact integral-duration conversion, and publishes only canonical
/// signed nanoseconds and stable schema identity. Runtime never receives a member pointer, object
/// address, local operation thunk, or C++ duration representation through this record.
struct ParameterDerivedTermPlanRecord {
  std::string term_id;  ///< Canonical plan-local semantic term identity.
  OperatorRef owner{};  ///< Plan-bound Operator owning the dependency.
  ParameterDerivedTermKind kind{ParameterDerivedTermKind::kLatestWithinMaxAge};
  PortId input{};                   ///< Selected input for max-age terms, or zero for deadlines.
  ParamFieldId field{};             ///< Stable field projection identity.
  ParamSchemaHash field_full_id{};  ///< Complete canonical field identity.
  std::string field_path;           ///< Owned declared nested path.
  std::string schema_id;            ///< Owning root schema identity.
  ParamSchemaVersion schema_version{};                        ///< Owning root schema version.
  ParamMutability mutability{ParamMutability::kCompileTime};  ///< Declared update boundary.
  ParameterDerivedTermUpdateDisposition update_disposition{
      ParameterDerivedTermUpdateDisposition::kFrozenCompileTime};
  std::int64_t initial_value_ns{};  ///< Positive generation-zero canonical value.
  std::int64_t minimum_value_ns{};  ///< Positive inclusive admitted lower bound.
  std::int64_t maximum_value_ns{};  ///< Inclusive admitted upper bound.
};

/// @brief Describe one Operator's immutable generation-zero Parameter contract.
///
/// Records are in dense Operator identity order and own their schema/path
/// strings. Object size/alignment are process-ABI materialization facts, not a
/// portable schema or permission to byte-copy an object.
struct OperatorParameterPlanRecord {
  OperatorRef owner{};                       ///< Plan-bound Operator identity.
  std::string operator_path;                 ///< Canonical structured identity spelling.
  std::optional<Profile> effective_profile;  ///< Resolved profile, or absent on rejection.
  ParamGeneration initial_generation{};      ///< Always zero for generation zero.
  bool no_params{};                          ///< True only for the static \ref NoParams value.
  ParamSchemaAvailability schema_availability{
      ParamSchemaAvailability::kOpaqueLocal};  ///< Declared-schema availability.
  ParameterSchemaAdmissionDisposition schema_disposition{
      ParameterSchemaAdmissionDisposition::kOpaqueLocal};  ///< Total schema admission outcome.
  ParameterMaterializationStrategy materialization{
      ParameterMaterializationStrategy::kRejected};  ///< Accepted generation-zero strategy.
  std::string schema_id;                             ///< Owned stable schema ID, or empty.
  ParamSchemaVersion schema_version{};               ///< Declared schema version, or zero.
  ParamSchemaHash structural_hash{};                 ///< Decoding-shape identity, or zero.
  ParamSchemaHash contract_hash{};                   ///< Enforced-policy identity, or zero.
  ParamSchemaHash metadata_hash{};  ///< Contract-plus-description identity, or zero.
  ParamSchemaCoverage schema_coverage{
      ParamSchemaCoverage::kDeclaredFieldsOnly};  ///< Coverage claim or conservative default.
  ParamSchemaProof schema_proof{ParamSchemaProof::kAuthorAsserted};  ///< Coverage evidence.
  std::uint64_t object_size{};              ///< Process ABI object bytes for finite derivation.
  std::uint64_t object_alignment{};         ///< Required separate-arena alignment.
  ParamStorageTraitsSummary storage{};      ///< Pointer-free whole-aggregate storage evidence.
  bool fixed_scalar_generation_strategy{};  ///< Complete fixed scalar cloning is admitted.
  bool runtime_updates_enabled{};           ///< At least one runtime field is admitted.
  bool lifecycle_updates_enabled{};         ///< At least one lifecycle field is admitted.
};

/// @brief Preserve finite Parameter storage and evidence capacities.
///
/// \ref parameter_arena_bytes is the sum of independently aligned per-Operator
/// arenas. No inter-Operator padding is implied by this record.
struct ParameterRuntimeCapacityPlanRecord {
  ParamRuntimeUpdatePolicy resolved_update_policy{
      ParamRuntimeUpdatePolicy::kDisabled};  ///< Total active-update policy.
  ParameterArenaStrategy arena_strategy{
      ParameterArenaStrategy::kSeparatePerOperator};  ///< Frozen allocation topology.
  std::uint32_t enabled_operator_count{};             ///< Operators owning finite arenas.
  std::uint32_t max_pending_updates_per_operator{};   ///< Accepted per-Operator pending bound.
  std::uint32_t generation_slots_per_operator{};      ///< Published/candidate/retirement slots.
  std::uint32_t compute_ingress_capacity{};           ///< Preallocated compute-origin slots.
  std::uint32_t max_compute_patches_per_request{};    ///< Inline patches copied into one slot.
  std::uint32_t audit_capacity{};                     ///< Global retained audit capacity.
  std::uint32_t event_capacity{};                     ///< Global retained event capacity.
  std::uint32_t max_event_cursors{};                  ///< Independent observer cursor capacity.
  /// Minimum lifecycle queue, receipt, fence, generation, event, audit,
  /// quiesce, and rollback capacities contributed by admitted Parameter updates.
  LifecycleTransactionCapacityValues required_lifecycle_capacity{};
  std::uint64_t parameter_arena_bytes{};  ///< Sum of all separate per-Operator arenas.
  std::uint64_t max_arena_alignment{};    ///< Largest required individual arena alignment.
  bool runtime_updates_enabled{};         ///< Whether any Operator admits next-compute publication.
  bool lifecycle_updates_enabled{};       ///< Whether any Operator admits a lifecycle candidate.
};

/// @brief Own an immutable inspectable result of Graph compilation.
///
/// Copies share one immutable state. The state deep-owns all records and remains
/// valid after the source Graph is mutated or destroyed. A rejected plan remains
/// queryable but cannot be executed by later runtime entry points.
class HOLOSCAN_CORE_EXPORT ExecutionPlan {
 public:
  /// @brief Create an empty rejected plan handle.
  ExecutionPlan() noexcept = default;

  /// @brief Return canonical schema-versioned JSON.
  /// @return Plan-owned bytes, or an empty view for an empty handle.
  [[nodiscard]] std::string_view json() const noexcept;

  /// @brief Return the frozen canonical compiler snapshot.
  /// @return Shared immutable canonical view using the same dense IDs as all
  /// plan records, or an empty view.
  /// @note These IDs need not equal the snapshot-local authoring IDs exposed by
  /// \ref GraphView. Match plan records only against this view.
  [[nodiscard]] ExecutionGraphView graph() const noexcept;

  /// @brief Return the complete source Graph identity.
  /// @return Frozen process-local instance/version/hash tuple, or zeros.
  [[nodiscard]] GraphAuthoringIdentity graph_identity() const noexcept;

  /// @brief Return the frozen source Graph generation.
  /// @return Nonzero generation for a composed plan, or zero.
  [[nodiscard]] GraphGeneration generation() const noexcept;

  /// @brief Return the canonical source Graph authoring hash.
  /// @return Authoring-content hash, or all-zero bytes.
  [[nodiscard]] GraphAuthoringHash graph_hash() const noexcept;

  /// @brief Return the semantic execution-plan digest.
  /// @return Versioned normalized plan digest, or all-zero bytes.
  [[nodiscard]] PlanDigest digest() const noexcept;

  /// @brief Return whether admission completed without an error diagnostic.
  /// @return \c true for an admitted plan with no error, including a
  /// compiler-accepted empty Graph's zero-work plan.
  [[nodiscard]] bool ok() const noexcept;

  /// @brief Return stable compile diagnostics in canonical order.
  /// @return Plan-owned immutable records, or an empty span.
  [[nodiscard]] std::span<const Diagnostic> diagnostics() const noexcept;

  /// @brief Require an admitted plan before continuing cold setup.
  /// @throws CompileError when the handle is empty or the plan has an error.
  void require_no_errors() const;

  /// @brief Return immutable profile admission evidence.
  /// @return Plan-owned manifest, or a stable empty rejected manifest.
  [[nodiscard]] const ProfileManifest& profile_manifest() const noexcept;

  /// @brief Return partition algorithm selection evidence.
  /// @return Plan-owned record, or a stable empty rejected record.
  /// @note Inspect \ref PartitioningPlanRecord::disposition before interpreting accepted-cover
  /// fields such as \ref PartitioningPlanRecord::mode and the partition counts.
  [[nodiscard]] const PartitioningPlanRecord& partitioning_plan() const noexcept;

  /// @brief Return the retained high-level partition inspection range.
  /// @return Immutable range and lookup facade retaining plan state independently.
  /// @throws RuntimeError with ErrorCode::kInvalidArgument when this plan is default-constructed
  /// or rejected.
  [[nodiscard]] ExecutionPartitionsView partitions() const;

  /// @brief Return canonical per-vertex partition decisions.
  /// @return Plan-owned immutable records, or an empty span.
  [[nodiscard]] std::span<const PartitionDecisionRecord> partition_decisions() const noexcept;

  /// @brief Return the accepted disjoint execution-partition cover.
  /// @return Plan-owned immutable records, or an empty span.
  [[nodiscard]] std::span<const ExecutionPartitionRecord> execution_partitions() const noexcept;

  /// @brief Return every original graph edge crossing accepted partitions.
  /// @return Plan-owned immutable records, or an empty span.
  [[nodiscard]] std::span<const ScheduleEdgeRecord> schedule_edges() const noexcept;

  /// @brief Return aggregate partition-owner scheduler evidence.
  ///
  /// The record permits concurrency across execution partitions while preserving deterministic
  /// serial callback execution within every partition.
  /// @return Accepted scheduler record, or a zero-initialized record.
  [[nodiscard]] SchedulerPlanRecord scheduler_plan() const noexcept;

  /// @brief Return compile-time placement proofs and Runtime readback obligations.
  /// @return Plan-owned immutable records in partition/probe order.
  [[nodiscard]] std::span<const PlacementProbePlanRecord> placement_probes() const noexcept;

  /// @brief Return requested and accepted placement for named execution partitions.
  /// @return Plan-owned immutable records in canonical authored-name and request order.
  [[nodiscard]] std::span<const PartitionPlacementPlanRecord> partition_placements() const noexcept;

  /// @brief Return every dense partition-local serial callback owner consumed by Runtime.
  /// @return Plan-owned records ordered by zero-based lane identity.
  [[nodiscard]] std::span<const ExecutionLanePlanRecord> execution_lanes() const noexcept;

  /// @brief Return canonical compiled ports with resolved finite bounds.
  /// @return Plan-owned records in dense \ref PortId order.
  [[nodiscard]] std::span<const PortPlanRecord> ports() const noexcept;

  /// @brief Return pointer-free rate declarations and lifecycle selection domains.
  /// @return Plan-owned records in dense \ref PortId order.
  [[nodiscard]] std::span<const PortRatePlanRecord> port_rates() const noexcept;

  /// @brief Return exact finite accumulation horizons used by queue sizing.
  /// @return Plan-owned records in canonical edge and horizon order.
  [[nodiscard]] std::span<const AccumulationHorizonPlanRecord> accumulation_horizons()
      const noexcept;

  /// @brief Return reviewable hard-floor contributions and composition identities.
  /// @return Plan-owned records in canonical edge and contributor order.
  [[nodiscard]] std::span<const HardFloorContributorPlanRecord> hard_floor_contributors()
      const noexcept;

  /// @brief Return immutable logical queue-sizing recipes.
  /// @return Plan-owned records in canonical \ref EdgeId order.
  [[nodiscard]] std::span<const QueueSizingRecipePlanRecord> queue_sizing_recipes() const noexcept;

  /// @brief Return deep-owned generated-package provenance used by this local plan.
  /// @return Records ordered by SchemaId and complete evidence, or an empty span.
  /// @note These diagnostic records are serialized in the public artifact but do not participate
  /// in Graph identity, the execution-plan digest, or wire compatibility.
  [[nodiscard]] std::span<const SchemaPackagePlanRecord> schema_packages() const noexcept;

  /// @brief Return reusable-Graph aliases projected into plan-local port identities.
  /// @return Plan-owned records in public/internal port order.
  [[nodiscard]] std::span<const InterfacePortBindingPlanRecord> interface_port_bindings()
      const noexcept;

  /// @brief Return resolved typed payload edges and queue/cache geometry.
  /// @return Plan-owned records in canonical \ref EdgeId order.
  [[nodiscard]] std::span<const PayloadEdgePlanRecord> payload_edges() const noexcept;

  /// @brief Return requested and accepted communication locality per payload edge.
  /// @return Plan-owned records in canonical \ref EdgeId order.
  [[nodiscard]] std::span<const ChannelLocalityRecord> channel_locality() const noexcept;

  /// @brief Return external Channel service-static requirements.
  /// @return Plan-owned records in canonical external-topic order.
  [[nodiscard]] std::span<const ChannelServiceStaticPlanRecord> channel_services() const noexcept;

  /// @brief Return publisher-local external endpoint requests.
  /// @return Plan-owned records in canonical external-topic order.
  [[nodiscard]] std::span<const ChannelPublisherEndpointPlanRecord> channel_publishers()
      const noexcept;

  /// @brief Return subscriber-local external endpoint requests.
  /// @return Plan-owned records in canonical external-topic order.
  [[nodiscard]] std::span<const ChannelSubscriberEndpointPlanRecord> channel_subscribers()
      const noexcept;

  /// @brief Return admitted generic non-Tensor output-loan capabilities.
  /// @return Plan-owned records in output \ref PortId order.
  [[nodiscard]] std::span<const GenericOutputLoanPlanRecord> generic_output_loans() const noexcept;

  /// @brief Return direct Tensor output-allocation capabilities.
  /// @return Plan-owned records only for outputs admitted with runtime-pool storage intent.
  [[nodiscard]] std::span<const TensorAllocationPlanRecord> tensor_allocations() const noexcept;

  /// @brief Return compile-time evidence for Parameter/input-derived Tensor slot capacities.
  /// @return Immutable records in output PortId order.
  [[nodiscard]] std::span<const TensorByteSpanDerivationPlanRecord> tensor_byte_span_derivations()
      const noexcept;

  /// @brief Find derived Tensor capacity evidence by authored Operator and output names.
  /// @param operator_path Canonical Operator path, such as `resize` or `child/resize`.
  /// @param output_name Declaration-local output name.
  /// @param field_id Optional descriptor-schema field identity.
  /// @param tensor_index Optional descriptor-schema Tensor index paired with `field_id`.
  /// @return A plan-owned record, or `nullptr` when the names are absent, the field selector is
  /// incomplete, or an omitted field selector would be ambiguous for a multi-Tensor output.
  [[nodiscard]] const TensorByteSpanDerivationPlanRecord* find_tensor_byte_span_derivation(
      std::string_view operator_path, std::string_view output_name,
      std::optional<std::uint16_t> field_id = std::nullopt,
      std::optional<std::uint16_t> tensor_index = std::nullopt) const noexcept;

  /// @brief Return field-specific allocation records for multi-Tensor schema outputs.
  /// @return Immutable records in output then canonical field-id order.
  [[nodiscard]] std::span<const TensorFieldAllocationPlanRecord> tensor_field_allocations()
      const noexcept;

  /// @brief Return physical Tensor interface-pool contracts.
  /// @return Plan-owned records in canonical external-topic order.
  [[nodiscard]] std::span<const TensorInterfacePoolPlanRecord> tensor_interface_pools()
      const noexcept;

  /// @brief Return aggregated process/device Tensor materialization contracts.
  /// @return Plan-owned records in process-group then device order.
  [[nodiscard]] std::span<const TensorDeviceResourcePlanRecord> tensor_device_resources()
      const noexcept;

  /// @brief Return Green Context pool declarations in process/device order.
  /// @return Plan-owned records in canonical process-group then device order.
  [[nodiscard]] std::span<const CudaGreenContextPoolPlanRecord> cuda_green_context_pools()
      const noexcept;

  /// @brief Return Green Context resources in pool/context order.
  /// @return Plan-owned records in canonical process-group, device, then context order.
  [[nodiscard]] std::span<const CudaGreenContextResourcePlanRecord> cuda_green_context_resources()
      const noexcept;

  /// @brief Return accepted and rejected Green Context requests in canonical selector order.
  /// @return Plan-owned records ordered by authored selector, device, then requested SM count.
  [[nodiscard]] std::span<const CudaGreenContextRequestPlanRecord> cuda_green_context_requests()
      const noexcept;

  /// @brief Return partition-to-context assignments in canonical partition order.
  /// @return Plan-owned records ordered by process group, device, then execution partition.
  [[nodiscard]] std::span<const CudaGreenContextAssignmentPlanRecord>
  cuda_green_context_assignments() const noexcept;

  /// @brief Return generalized CUDA compute streams in canonical owner order.
  /// @return Plan-owned records ordered by process group, device, then stable owner identity.
  [[nodiscard]] std::span<const CudaComputeStreamPlanRecord> cuda_compute_streams() const noexcept;

  /// @brief Return generated-provider inline Tensor record bucket contracts.
  /// @return Plan-owned records ordered by external topic identity.
  [[nodiscard]] std::span<const TensorInlineRecordPlanRecord> tensor_inline_records()
      const noexcept;

  /// @brief Return finite imported exporter-generation capacities.
  /// @return Plan-owned records in canonical external-topic order.
  [[nodiscard]] std::span<const ImportedGenerationCapacityPlanRecord>
  imported_generation_capacities() const noexcept;

  /// @brief Return the closed Tensor source-class-to-realization matrix.
  /// @return Plan-owned records ordered by topic and stable source-class value.
  [[nodiscard]] std::span<const TensorTransferPlanRecord> tensor_transfer_plans() const noexcept;

  /// @brief Return canonical unique-slot capacity contributors.
  /// @return Plan-owned records ordered by pool and contributor identity.
  [[nodiscard]] std::span<const PoolCapacityContributorRecord> pool_capacity_contributors()
      const noexcept;

  /// @brief Return stable many-to-many logical-claim to provider-pool dependencies.
  /// @return Plan-owned records ordered by pool, scenario, group, and record identity.
  [[nodiscard]] std::span<const PoolCapacityDependencyPlanRecord> pool_capacity_dependencies()
      const noexcept;

  /// @brief Return immutable provider allocation recipes and dependency ranges.
  /// @return Plan-owned records ordered by stable pool identity.
  [[nodiscard]] std::span<const ProviderAllocationPlanRecord> provider_allocation_plans()
      const noexcept;

  /// @brief Return per-output publication-boundary retry and authority contracts.
  /// @return Plan-owned records in output \ref PortId order.
  [[nodiscard]] std::span<const PublicationOutcomePlanRecord> publication_outcomes() const noexcept;

  /// @brief Return canonical readiness declarations and admitted capacities.
  /// @return Plan-owned immutable records, or an empty span.
  [[nodiscard]] std::span<const ReadinessPlanRecord> readiness() const noexcept;

  /// @brief Return the immutable plan-local clock catalog.
  /// @return Owning view carrying the nonzero plan generation and dense clocks.
  /// The returned value remains valid after this ExecutionPlan is destroyed.
  [[nodiscard]] ClockCatalogView clock_catalog() const noexcept;

  /// @brief Return explicitly authored direct clock conversions.
  /// @return Plan-owned immutable records in canonical declaration order.
  [[nodiscard]] std::span<const ClockConversionRecord> clock_conversions() const noexcept;

  /// @brief Return normalized clock-health contracts for external endpoints.
  /// @return One plan-owned record per external endpoint in boundary-port order.
  [[nodiscard]] std::span<const EndpointClockHealthPlanRecord> endpoint_clock_health()
      const noexcept;

  /// @brief Return compile-admitted Operator completion rules.
  /// @return Plan-owned records in canonical logical Operator order.
  [[nodiscard]] std::span<const OperatorCompletionPlanRecord> operator_completion_rules()
      const noexcept;

  /// @brief Return one total pointer-free record per admitted temporal trigger.
  /// @return Plan-owned records in canonical Operator order.
  [[nodiscard]] std::span<const TemporalTriggerPlanRecord> temporal_triggers() const noexcept;

  /// @brief Return deterministic oldest-head aligned-join policies.
  /// @return Plan-owned records in canonical Operator order.
  [[nodiscard]] std::span<const AlignedTriggerPlanRecord> aligned_triggers() const noexcept;

  /// @brief Return fixed-phase clock-trigger schedules.
  /// @return Plan-owned records in canonical Operator order.
  [[nodiscard]] std::span<const ClockTriggerPlanRecord> clock_triggers() const noexcept;

  /// @brief Return explicit latest-start deadline policies.
  /// @return Plan-owned records in canonical Operator order.
  [[nodiscard]] std::span<const DeadlinePlanRecord> deadlines() const noexcept;

  /// @brief Return level-readable trigger policies.
  /// @return Plan-owned records in canonical Operator order.
  [[nodiscard]] std::span<const ReadableTriggerPlanRecord> readable_triggers() const noexcept;

  /// @brief Return complete phase-local missing-input aggregation policies.
  /// @return Plan-owned records grouped by Operator and selection phase.
  [[nodiscard]] std::span<const MissingAggregatePlanRecord> missing_aggregates() const noexcept;

  /// @brief Return admitted bounded window-trigger state machines.
  /// @return Plan-owned records in canonical Operator order.
  [[nodiscard]] std::span<const WindowTriggerPlanRecord> window_triggers() const noexcept;

  /// @brief Return admitted bracketed-input retention contracts.
  /// @return Plan-owned records in canonical input order.
  [[nodiscard]] std::span<const BracketedWindowPlanRecord> bracketed_windows() const noexcept;

  /// @brief Return one fully resolved record per explicitly selected input.
  /// @return Plan-owned records in canonical Operator/input order.
  [[nodiscard]] std::span<const InputSelectionPlanRecord> input_selections() const noexcept;

  /// @brief Return normalized predicate-trigger records.
  /// @return Plan-owned records in canonical Operator order.
  [[nodiscard]] std::span<const PredicateTriggerPlanRecord> predicate_triggers() const noexcept;

  /// @brief Return ordered scalar predicate-input records.
  /// @return Plan-owned records grouped by predicate and argument ordinal.
  [[nodiscard]] std::span<const PredicateInputPlanRecord> predicate_inputs() const noexcept;

  /// @brief Return pointer-free predicate callable admission records.
  /// @return Plan-owned records joined to triggers by predicate identity.
  [[nodiscard]] std::span<const PredicateCallablePlanRecord> predicate_callables() const noexcept;

  /// @brief Return every logical lifecycle checkpoint.
  /// @return Plan-owned immutable records, or an empty span.
  [[nodiscard]] std::span<const LifecycleCheckpointPlanRecord> lifecycle_checkpoints()
      const noexcept;

  /// @brief Return validated lifecycle dependencies.
  /// @return Plan-owned immutable records, or an empty span.
  [[nodiscard]] std::span<const LifecycleDependencyPlanRecord> lifecycle_dependencies()
      const noexcept;

  /// @brief Return compile-admitted cross-target restart permissions.
  /// @return Plan-owned immutable records, or an empty span.
  [[nodiscard]] std::span<const LifecycleRestartPermissionPlanRecord>
  lifecycle_restart_permissions() const noexcept;

  /// @brief Return one total restart-target disposition per Operator.
  /// @return Plan-owned immutable records, or an empty span.
  [[nodiscard]] std::span<const LifecycleRestartTargetPlanRecord> lifecycle_restart_targets()
      const noexcept;

  /// @brief Return compiler-derived per-target restart closures and widening tables.
  /// @return Plan-owned records sorted by dense target identity. The span remains valid while any
  /// copy of this ExecutionPlan owns the immutable state and is safe for concurrent read-only use.
  [[nodiscard]] std::span<const LifecycleRestartClosurePlanRecord> lifecycle_restart_closures()
      const noexcept;

  /// @brief Return lifecycle-mutable Parameter field records.
  /// @return Plan-owned immutable records, or an empty span.
  [[nodiscard]] std::span<const LifecycleMutableFieldPlanRecord> lifecycle_mutable_fields()
      const noexcept;

  /// @brief Return finite active lifecycle transaction capacity evidence.
  /// @return Accepted record, or empty when transactions are disabled/unavailable.
  [[nodiscard]] std::optional<LifecycleTransactionCapacityPlanRecord>
  lifecycle_transaction_capacity() const noexcept;

  /// @brief Return one generation-zero Parameter record per compiled Operator.
  /// @return Plan-owned records in dense Operator identity order, or an empty span.
  [[nodiscard]] std::span<const OperatorParameterPlanRecord> operator_parameters() const noexcept;

  /// @brief Return every declared Parameter field admission record.
  /// @return Plan-owned records ordered by Operator identity then \ref ParamFieldId.
  [[nodiscard]] std::span<const ParameterFieldPlanRecord> parameter_fields() const noexcept;

  /// @brief Return an owning policy-filtered plan-bound Parameter catalog.
  /// @param policy Public policy projection. The baseline exposes no privileged catalog policy.
  /// @return Immutable catalog copy with no startup values, value digests, or local operations.
  /// @note Copying a nonempty catalog may allocate. The returned copy owns its records and remains
  /// valid independently of this ExecutionPlan. An empty or unsupported-policy request returns an
  /// empty catalog.
  [[nodiscard]] ParameterCatalogView parameter_catalog(
      ParamCatalogAccessPolicy policy = ParamCatalogAccessPolicy::kPublic) const;

  /// @brief Return bounded temporal terms derived from declared Parameter fields.
  /// @return Plan-owned immutable records in canonical term-identity order.
  [[nodiscard]] std::span<const ParameterDerivedTermPlanRecord> parameter_derived_terms()
      const noexcept;

  /// @brief Return finite Parameter update capacity evidence.
  /// @return Compiler-selected capacities, or empty for an empty plan handle.
  [[nodiscard]] std::optional<ParameterRuntimeCapacityPlanRecord> parameter_runtime_capacity()
      const noexcept;

  /// @brief Return the immutable external endpoint catalog.
  /// @return Plan-owned attempted catalog, including normalized rows retained
  /// by a rejected plan. The catalog never makes a rejected plan executable.
  [[nodiscard]] const ChannelEndpointCatalog& endpoint_catalog() const noexcept;

 private:
  explicit ExecutionPlan(std::shared_ptr<const detail::ExecutionPlanState> state) noexcept
      : state_(std::move(state)) {}

  std::shared_ptr<const detail::ExecutionPlanState> state_;
  friend class detail::ExecutionPlanAccess;
};

}  // namespace holoscan
