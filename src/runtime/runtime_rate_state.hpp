/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/rate.hpp>
#include <holoscan/core/rate_generation.hpp>

#include "../core/pool_capacity_reducer.hpp"

namespace holoscan::detail {

struct ExecutionPlanState;
struct RuntimeOperatorInstance;
class RuntimeRateState;
class RuntimeRateStateTestAccess;

/// One candidate's active logical queue depth for a canonical payload edge.
struct RuntimeQueueDepthGeneration {
  EdgeId edge{};
  QueueSizingRecipeId recipe{};
  PortId producer{};
  HorizonRecordId accumulation_horizon{};
  HardFloorContributorId first_hard_floor_contributor{};
  std::uint32_t hard_floor_contributor_count{};
  std::uint32_t hard_floor{};
  std::optional<std::uint32_t> nominal_arrivals;
  std::optional<std::uint32_t> rate_recommendation;
  std::optional<std::uint32_t> admitted_depth_ceiling;
  StabilityClass stability{StabilityClass::kUnresolvedBoundedEvidence};
  DiagnosticId stability_reason{};
  std::uint32_t active_logical_depth{};
  /// True when this candidate must publish a different lifecycle-selected active limit.
  bool active_depth_update_required{};
};

/// One evaluated contribution from an immutable stable-pool dependency.
struct RuntimePoolCapacityContributionGeneration {
  PoolCapacityDependencyId dependency{};
  QueueSizingRecipeId queue_sizing_recipe{};
  std::uint32_t mapped_entries{};
};

/// One candidate's composed active capacity for a stable provider pool.
struct RuntimePoolCapacityGeneration {
  ProviderAllocationPlanRecordId provider{};
  RunGenerationId run_generation{};
  RateGenerationId rate_generation{};
  std::uint32_t required_entries{};
  std::optional<std::uint32_t> admitted_entry_ceiling;
  PoolCapacityDependencyId first_contribution{};
  std::uint32_t contribution_count{};
};

/// One candidate's concrete nonzero Tensor allocation adapter.
struct RuntimeTensorAllocationGeneration {
  ProviderAllocationPlanRecordId provider{};
  RunGenerationId run_generation{};
  RateGenerationId rate_generation{};
  TensorAllocationPlanRecord allocation;
};

/// One provider's cold materialization result for the candidate being evaluated.
struct RuntimeProviderMaterializationResult {
  ProviderAllocationPlanRecordId provider{};
  AllocationAction action{AllocationAction::kRejected};
  DiagnosticId compatibility_reason{};
  std::optional<std::uint32_t> materialized_entries;
  std::optional<std::uint64_t> stride_bytes_per_entry;
  std::optional<std::uint64_t> reserved_bytes_total;
  bool reported{};
};

/// Borrow the complete evaluated materialization view while lifecycle is fenced.
struct RuntimeRateMaterializationView {
  std::span<const RuntimeQueueDepthGeneration> queue_depths;
  std::span<const RuntimePoolCapacityGeneration> pool_capacities;
  std::span<const RuntimeTensorAllocationGeneration> tensor_allocations;
  std::span<RuntimeProviderMaterializationResult> provider_results;
};

/// Borrowed RunSession adapter for rate-dependent Core resource materialization.
struct RuntimeRateMaterializationBinding {
  void* state{};
  expected<void, Error> (*ensure)(void*, const RuntimeRateMaterializationView&, LifecycleStage,
                                  LifecycleTransitionPhase) noexcept {};
};

/// Pin one immutable published rate generation through a compute callback.
class RuntimeRateGenerationPin {
 public:
  RuntimeRateGenerationPin() noexcept = default;
  RuntimeRateGenerationPin(const RuntimeRateGenerationPin&) = delete;
  RuntimeRateGenerationPin& operator=(const RuntimeRateGenerationPin&) = delete;
  RuntimeRateGenerationPin(RuntimeRateGenerationPin&& other) noexcept;
  RuntimeRateGenerationPin& operator=(RuntimeRateGenerationPin&& other) noexcept;
  ~RuntimeRateGenerationPin();

  /// Resolve one declaration against the exact generation retained by this pin.
  [[nodiscard]] static expected<PortRateView, Error> resolve(void* state, const void* declaration,
                                                             bool output) noexcept;

 private:
  friend class RuntimeRateCallbackBinding;
  friend class RuntimeRateState;

  RuntimeRateGenerationPin(RuntimeRateState* state, std::size_t slot,
                           OperatorInstanceId owner) noexcept;
  void release() noexcept;

  RuntimeRateState* state_{};
  std::size_t slot_{std::numeric_limits<std::size_t>::max()};
  OperatorInstanceId owner_{};
};

/// Lazily pin one immutable rate generation only when a compute callback queries a port rate.
///
/// Constructing this callback-local adapter performs no atomic operation. Its first successful
/// query acquires one generation pin, and every later query in that callback resolves against the
/// same immutable generation. Destruction releases the pin when the callback scope ends.
class RuntimeRateCallbackBinding {
 public:
  RuntimeRateCallbackBinding(RuntimeRateState* state, OperatorInstanceId owner) noexcept;
  RuntimeRateCallbackBinding(const RuntimeRateCallbackBinding&) = delete;
  RuntimeRateCallbackBinding& operator=(const RuntimeRateCallbackBinding&) = delete;
  RuntimeRateCallbackBinding(RuntimeRateCallbackBinding&&) = delete;
  RuntimeRateCallbackBinding& operator=(RuntimeRateCallbackBinding&&) = delete;

  /// Resolve one declaration, acquiring the callback's generation pin on its first query.
  [[nodiscard]] static expected<PortRateView, Error> resolve(void* state, const void* declaration,
                                                             bool output) noexcept;

 private:
  RuntimeRateState* state_{};
  OperatorInstanceId owner_{};
  RuntimeRateGenerationPin pin_;
};

/// Own private and atomically published rate generations for one RunSession.
class RuntimeRateState {
 public:
  RuntimeRateState(const RuntimeRateState&) = delete;
  RuntimeRateState& operator=(const RuntimeRateState&) = delete;
  RuntimeRateState(RuntimeRateState&&) = delete;
  RuntimeRateState& operator=(RuntimeRateState&&) = delete;
  ~RuntimeRateState();

  /// Build the bounded declaration index and preallocate every generation slot.
  [[nodiscard]] static expected<std::unique_ptr<RuntimeRateState>, Error> create(
      const ExecutionPlanState& plan, std::span<RuntimeOperatorInstance> operators,
      RunGenerationId run_generation, std::size_t execution_lane_count,
      ProcessGroupId process_group = {});

  /// Prepare generation zero before the first startup lifecycle callback.
  [[nodiscard]] expected<void, Error> begin_startup_candidate() noexcept;

  /// Prepare a fresh restart candidate while retaining unaffected published bindings.
  [[nodiscard]] expected<void, Error> begin_restart_candidate(
      std::span<const OperatorInstanceId> members, LifecycleStage from_stage) noexcept;

  /// Bind one lifecycle-selected output in its exact owner/stage/forward callback.
  [[nodiscard]] expected<void, Error> bind_nominal_rate(OperatorInstanceId owner,
                                                        LifecycleStage stage,
                                                        LifecycleTransitionPhase phase,
                                                        const void* declaration,
                                                        Rate value) noexcept;

  /// Query the phase-selected lifecycle generation for one owner-local declaration.
  [[nodiscard]] expected<PortRateView, Error> resolve_lifecycle(OperatorInstanceId owner,
                                                                LifecycleStage stage,
                                                                LifecycleTransitionPhase phase,
                                                                const void* declaration,
                                                                bool output) noexcept;

  /// Close one Core-owned global rate barrier after all user callbacks in that stage.
  [[nodiscard]] expected<void, Error> close_stage_barrier(LifecycleStage stage,
                                                          LifecycleTransitionPhase phase) noexcept;

  /// Prove that the complete candidate is publishable before an irreversible runtime commit.
  [[nodiscard]] expected<void, Error> preflight_candidate_publication() noexcept;

  /// Publish the already-preflighted candidate as a checked pointer-sized atomic exchange.
  [[nodiscard]] expected<void, Error> publish_candidate() noexcept;

  /// Discard a private candidate without changing the old published generation.
  void discard_candidate() noexcept;

  /// Retain the private candidate as the latest rejected generation.
  void reject_candidate(Error error = Error{ErrorCode::kFailure, "RATE_GENERATION_REJECTED"},
                        DiagnosticSource source = {}) noexcept;

  /// Capture an owning coherent published/latest-rejected evidence snapshot.
  [[nodiscard]] expected<RateGenerationSnapshot, Error> capture_snapshot() const;

  /// Pin the current published generation for one compute callback.
  [[nodiscard]] expected<RuntimeRateGenerationPin, Error> pin_published(
      OperatorInstanceId owner) noexcept;

  /// Return whether this plan needs a synthetic barrier at one forward stage.
  [[nodiscard]] bool requires_barrier(LifecycleStage stage) const noexcept;

  /// Return whether rate-dependent resources must be ensured as first kAllocate work.
  [[nodiscard]] bool requires_resource_materialization() const noexcept;

  /// Return whether startup storage geometry is unavailable until a lifecycle barrier closes.
  [[nodiscard]] bool requires_deferred_resource_materialization() const noexcept;

  /// Visit the complete evaluated candidate queue-depth table while lifecycle is fenced.
  [[nodiscard]] expected<void, Error> visit_candidate_queue_depths(
      void* state,
      expected<void, Error> (*visitor)(void*,
                                       std::span<const RuntimeQueueDepthGeneration>) noexcept,
      bool require_all_resolved = true) noexcept;

  /// Visit the complete candidate queue, pool, and provider geometry while lifecycle is fenced.
  [[nodiscard]] expected<void, Error> visit_candidate_materialization(
      void* state,
      expected<void, Error> (*visitor)(void*, const RuntimeRateMaterializationView&) noexcept,
      bool require_all_resolved = true) noexcept;

 private:
  friend class RuntimeRateGenerationPin;
  friend class RuntimeRateStateTestAccess;

  static constexpr std::size_t kForwardStageCount = 6U;
  static constexpr std::size_t kInvalidSlot = std::numeric_limits<std::size_t>::max();

  struct DeclarationEntry {
    const void* declaration{};
    PortId port{};
    OperatorInstanceId owner{};
    PortDirection direction{PortDirection::kInput};
  };

  struct GenerationSlot {
    std::atomic<std::uint32_t> readers{};
    std::vector<PortRateView> views;
    std::vector<std::optional<Rate>> bindings;
    std::vector<std::uint8_t> resolution_state;
    std::vector<PortId> immediate_provenance;
    std::vector<PortId> root_provenance;
    std::vector<RateComparison> expectation_comparisons;
    std::vector<RuntimeQueueDepthGeneration> queue_depths;
    std::vector<RuntimePoolCapacityContributionGeneration> pool_contributions;
    std::vector<RuntimePoolCapacityGeneration> pool_capacities;
    std::vector<PoolCapacityTerm> pool_capacity_terms;
    std::vector<MappedPoolCapacityTerm> mapped_pool_capacity_terms;
    std::vector<RuntimeTensorAllocationGeneration> tensor_allocations;
    std::vector<RuntimeProviderMaterializationResult> provider_results;
    std::vector<ProviderAllocationGenerationRecord> provider_records;
    std::vector<std::uint8_t> provider_decision_required;
    std::array<bool, kForwardStageCount> closed_stages{};
    std::size_t predecessor_slot{kInvalidSlot};
    RateGenerationId generation{};
    Error failure{ErrorCode::kSuccess};
    DiagnosticSource failure_source{};
    bool preflighted{};
    bool provider_records_sealed{};
  };

  struct ProviderPublishedIdentity {
    ProviderAllocationGenerationRecordId decision{};
    AllocationGenerationId allocation{};
  };

  explicit RuntimeRateState(const ExecutionPlanState& plan, RunGenerationId run_generation,
                            ProcessGroupId process_group) noexcept;

  [[nodiscard]] const DeclarationEntry* find_declaration(const void* declaration) const noexcept;
  [[nodiscard]] expected<std::size_t, Error> acquire_candidate_slot_locked() noexcept;
  [[nodiscard]] expected<RateGenerationId, Error> reserve_generation_locked() noexcept;
  void initialize_candidate_locked(GenerationSlot& slot, RateGenerationId generation) noexcept;
  [[nodiscard]] expected<void, Error> resolve_candidate_locked(GenerationSlot& slot) noexcept;
  [[nodiscard]] expected<void, Error> evaluate_queue_depths_locked(GenerationSlot& slot) noexcept;
  [[nodiscard]] expected<void, Error> evaluate_pool_capacities_locked(
      GenerationSlot& slot) noexcept;
  void fail_candidate_locked(GenerationSlot& slot, Error error) noexcept;
  void reject_candidate_locked(Error error, DiagnosticSource source) noexcept;
  [[nodiscard]] expected<void, Error> seal_provider_records_locked(GenerationSlot& slot) noexcept;
  [[nodiscard]] expected<void, Error> append_snapshot_generation_locked(
      RateGenerationSnapshot& snapshot, const GenerationSlot& slot,
      RateGenerationDisposition disposition) const;
  [[nodiscard]] expected<PortRateView, Error> resolve_pinned(std::size_t slot,
                                                             OperatorInstanceId owner,
                                                             const void* declaration,
                                                             bool output) const noexcept;
  void release_pin(std::size_t slot) noexcept;

  const ExecutionPlanState* plan_{};
  RunGenerationId run_generation_{};
  ProcessGroupId process_group_{};
  std::vector<DeclarationEntry> declaration_index_;
  std::vector<OperatorInstanceId> port_owners_;
  std::vector<PortDirection> port_directions_;
  std::vector<PortId> input_sources_;
  std::vector<std::uint8_t> local_ports_;
  std::vector<std::uint8_t> local_edges_;
  std::vector<std::uint8_t> local_providers_;
  std::vector<std::uint8_t> local_dependencies_;
  std::vector<std::uint8_t> evidence_ports_;
  std::vector<std::uint8_t> evidence_queues_;
  std::array<bool, kForwardStageCount> barrier_stages_{};
  bool resource_materialization_required_{};
  bool deferred_resource_materialization_required_{};
  std::vector<std::unique_ptr<GenerationSlot>> slots_;
  std::atomic<std::size_t> published_slot_{kInvalidSlot};
  std::optional<std::size_t> candidate_slot_;
  std::optional<std::size_t> rejected_slot_;
  std::vector<std::optional<LifecycleStage>> provider_latest_binding_stages_;
  std::vector<ProviderPublishedIdentity> published_provider_identities_;
  RateGenerationId next_generation_{1U};
  std::uint64_t next_allocation_generation_{1U};
  std::uint64_t next_storage_binding_generation_{1U};
  mutable std::mutex lifecycle_mutex_;
};

/// Test-only access to bounded generation-counter failure seams.
class RuntimeRateStateTestAccess {
 public:
  static void set_next_storage_binding_generation(RuntimeRateState& state,
                                                  std::uint64_t next) noexcept;
};

}  // namespace holoscan::detail
