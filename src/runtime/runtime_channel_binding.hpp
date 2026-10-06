/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include <holoscan/channel/endpoint_interfaces.hpp>
#include <holoscan/core/activation.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/rate_generation.hpp>
#include <holoscan/time/clock_registry.hpp>

#include "../core/diagnostic_registry.hpp"
#include "port_binding.hpp"
#include "temporal_input_cache.hpp"

namespace holoscan::channel_internal {
enum class CatalogSchemaFrameCorruption : std::uint8_t;
}  // namespace holoscan::channel_internal

namespace holoscan::detail {

struct ExecutionPlanState;
class EndpointClockHealthRuntimeCoordinator;
class RuntimeChannelSession;
class RuntimeChannelSessionTestAccess;
class RuntimeFixtureObservation;
class RuntimeCudaExecutionResources;
class RuntimeRateState;
struct RuntimeSupportThreadPlacementFailure;

using RuntimeRateSnapshotCapture =
    expected<RateGenerationSnapshot, Error> (*)(const RuntimeRateState&) noexcept;

enum class RuntimeChannelTeardownPhase : std::uint8_t {
  kPumpJoined,
  kEndpointChildrenDestroyed,
  kRuntimeDestroyed,
};

using RuntimeChannelTeardownObserver = void (*)(void*, RuntimeChannelTeardownPhase) noexcept;
using RuntimeChannelInputLockObserver = void (*)(void*) noexcept;
using RuntimeChannelInputPollContentionObserver = void (*)(void*, VertexId, PortId,
                                                           SourceId) noexcept;
using RuntimeChannelActivationRetirementObserver = expected<void, Error> (*)(void*, VertexId,
                                                                             std::uint64_t,
                                                                             bool) noexcept;
using RuntimeChannelPollSelectionObserver = void (*)(void*) noexcept;
using RuntimeChannelClockHealthLockObserver = void (*)(void*) noexcept;

/// Source-tree observer that records whether a RunSession-owned Channel session was destroyed.
class RuntimeChannelTeardownTestControl final {
 public:
  static void observe(void* opaque, RuntimeChannelTeardownPhase phase) noexcept {
    if (opaque != nullptr && phase == RuntimeChannelTeardownPhase::kRuntimeDestroyed) {
      static_cast<RuntimeChannelTeardownTestControl*>(opaque)->runtime_destroyed_.store(
          true, std::memory_order_release);
    }
  }

  [[nodiscard]] bool runtime_destroyed() const noexcept {
    return runtime_destroyed_.load(std::memory_order_acquire);
  }

 private:
  std::atomic_bool runtime_destroyed_{false};
};

/// Select which Channel poll flavour a \ref RuntimeChannelIncompletePollTestControl forces.
///
/// Each flavour is also the acquisition the Channel readiness binding performs for that poll.
enum class RuntimeChannelIncompletePollKind : std::uint8_t {
  /// Blocking recovery polls (\ref PortBinding::poll_blocking).
  kBlocking,
  /// Ordinary nonblocking readiness polls (\ref PortBinding::poll).
  kNonblocking,
  /// The bounded retry after a contended decline (\ref PortBinding::poll_after_contention).
  kContentionRetry,
};

/// Budget for one contended input cache acquisition during \ref PortBinding::poll_after_contention.
///
/// Every holder of an input cache mutex is a short, non-blocking critical section (a slot
/// reservation or a bounded store, measured at roughly 100-200 ns), so 50 us leaves a wide margin
/// per acquisition while staying far below the 1 ms fallback park the retry avoids. The Channel
/// readiness binding applies the budget per contended acquisition, not per poll, and reports an
/// expired budget exactly like a declined nonblocking poll.
constexpr std::chrono::nanoseconds kContentionRetryBudget = std::chrono::microseconds{50};

/// Source-tree control that makes a bounded number of Channel polls report incomplete.
///
/// A forced nonblocking decline models an ordinary readiness poll that lost its input cache lock
/// to a concurrent bounded holder; the contention retry then completes, so a test can prove the
/// retry converts the published revision promptly. A forced contention-retry decline models a
/// holder that outlives the retry, so a round can still end with an incomplete ordinary scan.
class RuntimeChannelIncompletePollTestControl final {
 public:
  explicit RuntimeChannelIncompletePollTestControl(
      std::uint32_t count,
      RuntimeChannelIncompletePollKind kind = RuntimeChannelIncompletePollKind::kBlocking,
      VertexId target = {}) noexcept
      : remaining_(count), kind_(kind), target_(target) {}

  /// Return whether this control targets the given poll flavour.
  [[nodiscard]] bool applies_to(const RuntimeChannelIncompletePollKind flavour) const noexcept {
    return flavour == kind_;
  }

  /// Replace the remaining forced-incomplete budget, for example after a warm-up delivery.
  void arm(const std::uint32_t count) noexcept {
    remaining_.store(count, std::memory_order_release);
  }

  [[nodiscard]] bool consume(VertexId target) noexcept {
    if (target_ && target_ != target) {
      return false;
    }
    std::uint32_t remaining = remaining_.load(std::memory_order_acquire);
    while (remaining != 0U) {
      if (remaining_.compare_exchange_weak(
              remaining, remaining - 1U, std::memory_order_acq_rel, std::memory_order_acquire)) {
        forced_.fetch_add(1U, std::memory_order_release);
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] std::uint32_t forced() const noexcept {
    return forced_.load(std::memory_order_acquire);
  }

 private:
  std::atomic<std::uint32_t> remaining_{};
  std::atomic<std::uint32_t> forced_{};
  RuntimeChannelIncompletePollKind kind_{RuntimeChannelIncompletePollKind::kBlocking};
  VertexId target_{};
};

/// Source-tree hook invoked after RunSession creates its Channel session and before the run starts.
class RuntimeChannelSessionBindTestControl {
 public:
  virtual ~RuntimeChannelSessionBindTestControl() = default;

  [[nodiscard]] virtual expected<void, Error> bind(RuntimeChannelSession& session) noexcept = 0;
};

/// Borrow one nonblocking Runtime-wake callback for Channel progress.
struct RuntimeChannelReadinessWake {
  void* state{};
  /// Publish one scan hint after a joined pump scan when exact owner routing is unavailable.
  void (*signal)(void*) noexcept {};
  /// Publish a coalesced rescan hint to one exact compiler-frozen partition owner.
  ///
  /// Direct-retained progress and typed readiness materialized by either Channel drain owner use
  /// this route when distinct partition owners can otherwise remain parked.
  void (*signal_owner_progress)(void*, VertexId) noexcept {};
};

/// Borrow one cold session-local queue-depth overlay without mutating the immutable plan.
///
/// A lifecycle-candidate edge supplies its concrete depth only after the owning rate barrier has
/// closed. The resolver is consulted during unpublished Channel reconstruction at the first Core
/// kAllocate work item; it must return no value for every compile-concrete edge.
struct RuntimeChannelSizingOverride {
  void* state{};
  expected<std::optional<std::uint32_t>, Error> (*queue_depth)(void*, EdgeId) noexcept {};
};

struct RuntimeChannelInputResetState {
  std::uint64_t active_generation{};
  std::uint64_t view_generation{};
  std::uint64_t producer_epoch{};
  std::uint64_t next_arrival_sequence{};
  std::uint64_t next_global_arrival_order{};
  std::uint64_t last_producer_sequence{};
  std::uint64_t end_of_stream_transport_sequence{};
  bool reuse_retained{};
  bool temporal_retained{};
  bool reuse_slot_matches_single_occupied{};
  std::size_t occupied_samples{};
  std::size_t pending_direct_ingress_samples{};
  /// Number of slots retaining any payload, witness, selection, cache, ingress, or temporal state.
  std::size_t uncleared_sample_slots{};
  ClockId catalog_capture_clock{};
  ClockId plan_capture_clock{};
  /// Producer-incarnation token latched by a catalog import endpoint; zero when the endpoint has
  /// not accepted any operation or when the subscriber is not a catalog import.
  std::uint64_t latched_producer_generation{};
  bool end_of_stream_received{};
  /// Whether every subscriber-level sequence and terminal field has its cold-reset value.
  bool subscriber_sequence_state_reset{};
};

struct RuntimeChannelPumpIndexState {
  std::size_t wait_handle_count{};
  std::size_t drain_subscriber_count{};
  bool background_thread_required{};
  bool background_thread_running{};
  bool exact{};
};

struct RuntimeChannelProcessLocalProgressState {
  std::uint64_t published_deliveries{};
  std::uint64_t received_deliveries{};
  std::uint64_t publications_in_flight{};
  bool tracking_valid{};
};

/// Source-tree-only queue observation joined to one published rate generation.
///
/// Optional measurement fields remain empty until the corresponding queue state or attempt has
/// actually been sampled. In particular, an empty counter is unavailable evidence, not an
/// implied zero.
struct RuntimeChannelQueueObservation {
  PlanDigest plan_digest{};
  PlanGenerationId plan_generation{};
  EdgeId edge{};
  RunGenerationId run_generation{};
  RateGenerationId rate_generation{};
  std::uint32_t selected_logical_depth{};
  std::optional<std::uint32_t> rate_recommendation;
  std::optional<std::uint32_t> observed_occupancy;
  std::optional<std::uint32_t> observed_occupancy_high_water;
  std::optional<std::uint64_t> observed_backpressure_count;
  std::optional<std::uint64_t> observed_replacement_or_drop_count;
};

/// Source-tree-only owner for one bounded, explicitly sampled queue observation.
///
/// The control is never installed in an ExecutionPlan or Runtime hot path. A focused test owns it
/// and asks RuntimeChannelSessionTestAccess to sample one real typed-retention transition at a
/// time.
class RuntimeChannelQueueObservationTestControl final {
 public:
  [[nodiscard]] RuntimeChannelQueueObservation snapshot() const {
    const std::scoped_lock lock{mutex_};
    return observation_;
  }

 private:
  friend class RuntimeChannelSessionTestAccess;

  mutable std::mutex mutex_;
  RuntimeChannelQueueObservation observation_;
  std::optional<std::uint32_t> previous_occupancy_;
  std::optional<std::uint32_t> previous_staged_occupancy_;
  std::optional<std::uint64_t> previous_arrival_sequence_;
  std::optional<std::uint64_t> previous_terminal_sequence_;
  std::optional<bool> previous_terminal_received_;
  const RuntimeRateState* rate_state_{};
  bool bound_{};
};

/// @brief Describe the private endpoint families materialized for one Tensor edge.
struct RuntimeChannelTensorTransferState {
  std::size_t buffer_handle_publishers{};  ///< BufferHandle publisher endpoint count.
  std::size_t inline_publishers{};         ///< Inline-payload publisher endpoint count.
  bool buffer_handle_subscriber{};         ///< Whether the input uses a BufferHandle subscriber.
  bool inline_subscriber{};                ///< Whether the input uses an inline subscriber.
};

/// Describe one process-local replay declaration without placing C++ type state in the plan.
struct TensorContractMismatchCapture;

struct RuntimeChannelPortReplayRecord {
  PortId port{};
  VertexId owner{};
  PortDirection direction{PortDirection::kInput};
  void* declaration{};
  const RuntimePortTypeOps* type_ops{};
  TensorContractMismatchCapture* tensor_failure{};
};

/// Select retained typed samples for one input during one activation.
struct RuntimeChannelActivationSelection {
  PortId port{};
  InputSelection selection{};
  std::span<const TemporalSampleRecord> samples;
  bool batch{};
  bool destructive{};
};

/// Choose which eligible staged sample one freshness handoff reserves.
enum class RuntimeChannelFreshnessSelectionMode : std::uint8_t {
  kNewest,
  kNewestWithArrivalPrefix,
  kOldestProducerSuccessor,
};

/// Identify the exact staged sample and subscriber selected for one freshness handoff.
///
/// The generation is opaque to callers. RuntimeChannelSession validates it against the live
/// subscriber-scoped reservation before releasing the retained claim, so an ordinary promotion or
/// a later reservation cannot consume a predecessor's handoff.
struct RuntimeChannelFreshnessReservation {
  PortId port{};
  TemporalSampleRecord successor{};
  TimePoint successor_reference{};
  TemporalSampleRecord target{};
  TimePoint target_reference{};
  std::uint64_t generation{};
};

/// Bind one transported ReuseLast probe to the exact staged head it observed.
///
/// The subscriber index is private plan materialization order. Sequence and capture are copied
/// from the transport while its consumer lock is held; commit revalidates both under that same
/// lock before releasing any retained selector claim.
struct RuntimeChannelStagedReuseCandidate {
  std::uint32_t subscriber_index{};
  std::uint64_t source_sequence{};
  TimePoint source_capture{};
  TimePoint successor_reference{};
};

/// Describe one cold-proven destination that may consume a direct continuation poll in-place.
///
/// The record owns no state. Every identity is copied from the accepted plan or the live
/// generation-bound readiness binding and must be revalidated after the source callback.
struct RuntimeChannelDirectContinuationTarget {
  PlanGenerationId plan_generation{};
  RunGenerationId run_generation{};
  VertexId producer{};
  VertexId destination{};
  PortId source_output{};
  PortId destination_input{};
  std::string_view service;
  SourceId readiness_source{};
  ClockId reference_clock{};
  std::uint64_t source_generation{};
  std::uint64_t readiness_sequence{};
  std::uint64_t destination_revision{};
  std::uint64_t observed_destination_revision{};
  std::uint64_t source_activation_generation{};
  std::uint32_t execution_lane{};
  std::size_t readiness_capacity{};
};

/// Carry allocation-free evidence that one direct publication completed its successor poll.
///
/// This receipt is evidence only. It never grants activation authority and cannot bypass
/// dispatcher ingestion, selection, parameter observation, lifecycle, or activation retirement.
struct RuntimeChannelDirectContinuationReceipt {
  PlanGenerationId plan_generation{};
  RunGenerationId run_generation{};
  VertexId producer{};
  VertexId destination{};
  PortId source_output{};
  PortId destination_input{};
  std::string_view service;
  SourceId readiness_source{};
  ClockId reference_clock{};
  std::uint64_t source_generation{};
  std::uint64_t readiness_sequence{};
  std::uint64_t producer_epoch{};
  std::uint64_t publication_sequence{};
  std::uint64_t prior_destination_revision{};
  std::uint64_t destination_revision{};
  std::uint64_t source_activation_generation{};
  std::uint32_t execution_lane{};
  std::uint32_t successor_publications{};
  ErrorCode publication_status{ErrorCode::kSuccess};
};

/// Return whether one receipt exactly matches its cold target and non-consuming publication.
[[nodiscard]] bool runtime_channel_direct_continuation_matches(
    const RuntimeChannelDirectContinuationReceipt& receipt,
    const RuntimeChannelDirectContinuationTarget& target) noexcept;

/// Identify a cold fixture-profile rejection without allocating diagnostic text.
struct RuntimeFixtureObservationBindingFailure {
  ErrorCode code{ErrorCode::kIncompatible};
  VertexId owner{};
  PortId port{};
};

/// Own the process-local Channel endpoints and typed sample retention for one RunSession.
///
/// Construction performs all endpoint creation, WaitSet attachment, payload-storage allocation,
/// generic-loan construction, and replay-port binding before lifecycle entry. `arm()` is the sole
/// transition that starts readiness observation; `cancel()` followed by `join()` must complete
/// before lifecycle stop/release destroys replay declarations. The class is private Runtime
/// substrate and exposes no Channel or HoloMQ object to Operator code.
class RuntimeChannelSession {
 public:
  /// Select scalar receipt thunks before Arm and retain their owner through channel destruction.
  /// Validate every output before changing any binding; rejection identifies the first port.
  [[nodiscard]] expected<void, RuntimeFixtureObservationBindingFailure>
  observe_fixture_publications(std::shared_ptr<RuntimeFixtureObservation> observation) noexcept;
  RuntimeChannelSession(const RuntimeChannelSession&) = delete;
  RuntimeChannelSession& operator=(const RuntimeChannelSession&) = delete;
  RuntimeChannelSession(RuntimeChannelSession&&) = delete;
  RuntimeChannelSession& operator=(RuntimeChannelSession&&) = delete;
  ~RuntimeChannelSession();

  /// @brief Materialize bounded canonical-scalar and private constructed-loan bindings.
  ///
  /// @param plan Accepted immutable plan state. It must outlive the returned session because the
  ///        session borrows cold support-thread placement records through join.
  /// @param replay_ports Cold replay declarations for ports materialized in this process.
  /// @param process_group Process group selected for this Runtime worker, or invalid for the
  ///        complete local plan.
  /// @param readiness_wake Optional borrowed callback invoked after a joined pump scan publishes
  ///        typed input readiness. Its state must remain valid through \ref join.
  /// @param run_generation Stable run identity that owns this session. Source-tree tests that do
  ///        not join session state to generation evidence may omit it.
  /// @return The fully reconstructed Channel session, or a typed reconstruction error.
  [[nodiscard]] static expected<std::unique_ptr<RuntimeChannelSession>, Error> create(
      const ExecutionPlanState& plan, std::span<const RuntimeChannelPortReplayRecord> replay_ports,
      ProcessGroupId process_group = {}, RuntimeChannelReadinessWake readiness_wake = {},
      const RuntimeCudaExecutionResources* cuda_resources = nullptr,
      RuntimeChannelSizingOverride sizing_override = {}, RunGenerationId run_generation = {});

  /// Return whether reconstruction materialized any Channel activation state for an Operator.
  ///
  /// This cold query distinguishes bound Channel endpoints from replay-only port declarations.
  [[nodiscard]] bool has_materialized_activation_state(VertexId owner) const noexcept;

  /// Return whether reconstruction materialized subscriber-backed Channel input state.
  ///
  /// Output-only owners and channel-free optional declarations may still require activation state
  /// but must not acquire an input-arrival readiness source.
  [[nodiscard]] bool has_materialized_input_state(VertexId owner) const noexcept;

  /// Restore the session's original binding after startup CUDA Graph capture temporarily bound an
  /// input to a fixed region buffer. Returns false when this session does not own the input.
  [[nodiscard]] bool restore_input_binding(PortId input) noexcept;

  /// Restore the session's original publication binding after startup CUDA Graph capture.
  [[nodiscard]] bool restore_output_binding(PortId output) noexcept;

  /// Require one Tensor input to carry the fixed address captured for a CUDA Graph region.
  [[nodiscard]] bool require_fixed_tensor(PortId input, const void* address,
                                          std::span<const std::uint64_t> shape) noexcept;

  /// Clear a CUDA Graph fixed-address requirement when its owning region storage is detached.
  [[nodiscard]] bool clear_fixed_tensor_requirement(PortId input, const void* address) noexcept;

  /// Create one event-ring facade for all Channel inputs owned by an Operator.
  [[nodiscard]] expected<std::unique_ptr<PortBinding>, Error> create_readiness_binding(
      VertexId owner, RuntimePortBindingRequest request) noexcept;

  /// Return the exact finite ingress-witness capacity materialized for an Operator.
  [[nodiscard]] expected<std::uint32_t, Error> readiness_capacity(VertexId owner) const noexcept;

  /// Bind monitored external imports to the session's preallocated health coordinator.
  ///
  /// This cold operation must complete before arm. The borrowed mutex serializes monitored
  /// report dispatch and snapshotting with RunSession expiry and generation-fenced reset. It must
  /// remain valid through \ref join. Ordinary sessions with no monitored receivers retain no
  /// pointer to it and incur no lock.
  [[nodiscard]] expected<void, Error> bind_clock_health(
      EndpointClockHealthRuntimeCoordinator& coordinator, std::mutex& serialization_mutex) noexcept;

  /// @brief Bind the clock this session stamps publication coordinates from.
  ///
  /// This cold operation must complete before arm. The reference is borrowed from the session's
  /// GraphExec clock registry generation, because every publication stamps
  /// \c SampleMetadata::publish_timestamp_ns from it at the submission boundary. An inert
  /// reference is accepted and leaves publication time canonically absent, which is what a
  /// clockless plan requires. A session left unbound behaves the same way.
  ///
  /// @param clock Resolved plan default clock for this run generation, or an inert reference.
  /// @return Success when the clock is bound or deliberately absent, \c kInvalidArgument when the
  ///         session is already armed or already bound, \c kNotSupported when \p clock is resolved
  ///         but does not admit current-time reads, or \c kProtocolError when \p clock is not the
  ///         plan root default clock the endpoint catalog projected its publication record for.
  [[nodiscard]] expected<void, Error> bind_publish_clock(ClockRef clock) noexcept;

  /// Refresh publisher connection state before the global start barrier.
  [[nodiscard]] expected<void, Error> refresh_connections() noexcept;

  /// Start the single joined WaitSet pump after every Operator completed `kStart`.
  [[nodiscard]] expected<void, Error> arm() noexcept;

  /// Prepare one lifecycle candidate's active logical depth without publishing it.
  [[nodiscard]] expected<void, Error> prepare_rate_candidate_active_depth(
      EdgeId edge, std::uint32_t depth) noexcept;

  /// Publish every prepared active depth after the complete restart/start transaction succeeds.
  [[nodiscard]] expected<void, Error> commit_rate_candidate_active_depths() noexcept;

  /// Discard prepared active depths without changing the published queue behavior.
  void discard_rate_candidate_active_depths() noexcept;

  /// Flush one bounded process-local activation cut into the typed input stores.
  ///
  /// The serial controller invokes this after a successful compute and before freezing another
  /// activation or publishing a pending restart. The operation is serialized with the background
  /// pump, excludes externally driven imports, and leaves transport queued when a typed store is
  /// normally backpressured.
  [[nodiscard]] expected<void, Error> flush_process_local_activation_cut() noexcept;

  /// Arm a clean same-lane direct successor that may complete its poll in the source task.
  ///
  /// An empty result is an ordinary-path disposition. Arming is allocation-free, reserves no
  /// readiness credit, and adds one predictable capture branch to the selected output only for
  /// this activation.
  [[nodiscard]] std::optional<RuntimeChannelDirectContinuationTarget> arm_direct_continuation(
      VertexId producer, std::uint32_t execution_lane) noexcept;

  /// Complete one exact direct-successor poll on the already-active source execution lane.
  ///
  /// Dynamic mismatch returns an empty result and leaves the ordinary next-scan poll authoritative.
  /// A nonempty receipt proves that the full existing readiness poll minted exactly one event.
  [[nodiscard]] expected<std::optional<RuntimeChannelDirectContinuationReceipt>, Error>
  poll_direct_continuation(VertexId producer, std::uint32_t execution_lane,
                           TimePoint observed_at) noexcept;

  /// Revalidate one completion receipt against live plan, run, producer, and readiness generations.
  [[nodiscard]] bool validate_direct_continuation_receipt(
      const RuntimeChannelDirectContinuationReceipt& receipt) const noexcept;

  /// Request bounded pump cancellation. This function does not join the pump.
  void cancel() noexcept;

  /// Join the pump and return its first terminal transport error, when any.
  [[nodiscard]] expected<void, Error> join() noexcept;

  /// Return the typed selector record paired with one drained readiness witness.
  [[nodiscard]] expected<TemporalSampleRecord, Error> temporal_sample(
      PortId port, std::uint64_t readiness_sequence) const noexcept;

  /// Bind one Operator's frozen typed selection before entering its callback.
  ///
  /// The first call opens the generation. A later call for the same owner and generation may add
  /// disjoint, previously unprepared inputs after predicate evaluation; existing typed selections
  /// and predicate views remain pinned. Duplicate or conflicting ports are rejected, and a failed
  /// incremental call rolls back only that call. `retire_activation(..., false)` aborts the
  /// complete accumulated generation.
  [[nodiscard]] expected<void, Error> prepare_activation(
      VertexId owner, std::uint64_t generation,
      std::span<const RuntimeChannelActivationSelection> selections) noexcept;

  /// @brief Enforce Tensor read authority and stream ordering for one prepared activation.
  ///
  /// The operation acquires one finite private read lease for each newly prepared Tensor sample.
  /// CUDA-like samples additionally enqueue their producer-readiness wait and bind their final
  /// release to \p consumer_stream. Repeated calls for incremental predicate/compute selection
  /// prepare only newly added samples. All leases retire with the activation.
  ///
  /// @param owner Vertex owning the prepared activation.
  /// @param generation Activation generation opened by \ref prepare_activation.
  /// @param consumer_stream CUDA stream that consumes CUDA-like samples, or \c nullptr when no
  ///        consumer stream is required.
  /// @return Success after all newly selected Tensor samples are prepared, or an error if
  ///         activation identity, Tensor authority, or stream-ordering validation fails.
  [[nodiscard]] expected<void, Error> prepare_tensor_access(VertexId owner,
                                                            std::uint64_t generation,
                                                            cudaStream_t consumer_stream) noexcept;

  /// Return one concrete predicate input-view object for the active scalar selection.
  [[nodiscard]] expected<const void*, Error> predicate_view(
      PortId port, std::uint64_t generation) const noexcept;

  /// Retire an Operator callback and commit only its accepted destructive selections.
  [[nodiscard]] expected<void, Error> retire_activation(VertexId owner, std::uint64_t generation,
                                                        bool activation_accepted) noexcept;

  /// Retire the finite input-generation state owned by one downstream Operator.
  ///
  /// The caller invokes this only after every input producer delivered EOS and the final
  /// readiness witness retired. A `false` result is ordinary bounded drain pressure: an accepted
  /// temporal/cache claim, publication, output loan, or other owner-local use still holds the
  /// generation. Success releases finite ReuseLast retention and proves the owner can propagate
  /// EOS without inferring completion from temporary queue emptiness.
  [[nodiscard]] expected<bool, Error> retire_terminal_input_generation(VertexId owner) noexcept;

  /// Close every output owned by one completed operator generation.
  ///
  /// A transient transport error leaves closure pending; the controller refreshes and retries.
  [[nodiscard]] expected<void, Error> close_owner_outputs(VertexId owner) noexcept;

  /// Retire cache history that the temporal selector has proved no longer observable.
  ///
  /// This releases the selector's payload claim atomically for the supplied records. Physical
  /// storage remains stable while a readiness witness, active callback, or finite ReuseLast claim
  /// still owns it, and is reclaimed when the final independent claim retires.
  /// @param promoted When supplied, receives the number of staged samples the freed capacity
  /// admitted. A caller that released a claim specifically to make room for a successor needs this
  /// to tell "the successor is now in flight" from "the release freed nothing", which are opposite
  /// outcomes and indistinguishable from the return value alone.
  /// @param freshness_reservation Exact subscriber-scoped handoff to commit with the retirement.
  /// When supplied, the method preflights that the retained slot will become reclaimable, releases
  /// it, and promotes only the reserved successor while holding the same subscriber mutex.
  [[nodiscard]] expected<void, Error> retire_samples(
      PortId port, std::span<const TemporalSampleRecord> samples, std::uint32_t* promoted = nullptr,
      const RuntimeChannelFreshnessReservation* freshness_reservation = nullptr,
      std::int64_t* promoted_capture = nullptr) noexcept;

  /// Retry auxiliary process-local transport only after both sides of a freshness handoff commit.
  ///
  /// The Channel payload replacement is irreversible once \ref retire_samples succeeds with a
  /// freshness reservation. Keeping this fallible retry separate lets the controller commit the
  /// matching selector transaction before it propagates any retry failure.
  [[nodiscard]] expected<void, Error> retry_after_freshness_handoff() noexcept;

  /// Qualify and reserve one staged record for a freshness eviction that has not happened yet.
  ///
  /// Qualification and admission must be one transaction. Deciding on a record and then admitting
  /// whatever staging holds later is unsound however sound the decision was: at a fast producer
  /// the qualified record is replaced in between, and its replacement is captured after the
  /// witness the eviction was justified against, so it sorts behind that witness instead of ahead
  /// of it. A reservation is skipped by staging drop-oldest until it is promoted.
  ///
  /// @param retained Identity the caller intends to release.
  /// @param floor Bound on the reference coordinate of every witness the round can produce, or
  /// \c std::nullopt when the round can produce none. No witness means nothing to strand, so the
  /// coordinate test is skipped and any strictly newer staged record qualifies.
  /// @param selection_mode Producer-order selection for a single handoff, or newest-causal target
  /// selection with its producer-order arrival prefix preserved.
  /// @return The exact reservation when one subscriber owns both the retained claim and an
  ///         admissible successor; \c std::nullopt when the caller must decline the eviction.
  [[nodiscard]] expected<std::optional<RuntimeChannelFreshnessReservation>, Error>
  reserve_freshness_refill(PortId port, const TemporalSampleRecord& retained,
                           std::optional<TimePoint> floor,
                           RuntimeChannelFreshnessSelectionMode selection_mode =
                               RuntimeChannelFreshnessSelectionMode::kNewest) noexcept;

  /// Return whether an exact subscriber owns a reclaimable retained claim and newer staging.
  ///
  /// This cheap probe keeps clock-floor resolution off idle dispatch rounds. It does not reserve
  /// or otherwise mutate a slot; the caller must still use \ref reserve_freshness_refill after
  /// resolving the witness floor.
  [[nodiscard]] expected<bool, Error> has_freshness_refill_candidate(
      PortId port, const TemporalSampleRecord& retained) const noexcept;

  /// Return an exact transported-head token for one retained selector slot.
  [[nodiscard]] expected<std::optional<RuntimeChannelStagedReuseCandidate>, Error>
  staged_reuse_replacement_candidate(PortId port,
                                     const TemporalSampleRecord& retained) const noexcept;

  /// Return whether one exact input's current generation has a non-consuming terminal head.
  [[nodiscard]] expected<bool, Error> input_terminal_observed(PortId port) const noexcept;

  /// Replace one exact retained selector slot with its already-staged newer successor.
  ///
  /// @return The immutable temporal identity assigned to the committed replacement.
  [[nodiscard]] expected<TemporalSampleRecord, Error> replace_staged_reuse(
      PortId port, const TemporalSampleRecord& retained,
      const RuntimeChannelStagedReuseCandidate& candidate) noexcept;

  /// Return whether this input has a bounded freshness-refill path.
  ///
  /// Direct drop-oldest subscribers and transported ReuseLast staging can use the handoff.
  [[nodiscard]] expected<bool, Error> supports_freshness_refill(PortId port) const noexcept;

  /// Return whether this input has a drop-oldest subscriber with bounded refill staging.
  [[nodiscard]] expected<bool, Error> supports_drop_oldest_freshness_refill(
      PortId port) const noexcept;

  /// Recover an abandoned reservation before this input's next reconcile decision.
  ///
  /// The operation promotes newly writable staging while holding the subscriber cache lock. It
  /// returns false when a producer owns that lock, leaving the generation-fenced reservation for
  /// another non-blocking attempt. A caller that will immediately re-reserve may preserve an
  /// uncommitted target while its retained claim remains live.
  [[nodiscard]] expected<bool, Error> recover_abandoned_freshness_reservation(
      PortId port, bool preserve_uncommitted_target = false) noexcept;

  /// Abandon an exact reservation whose eviction did not proceed.
  void release_freshness_reservation(
      const RuntimeChannelFreshnessReservation& reservation) noexcept;

  /// Return the fate of the most recent record promoted for a released freshness claim.
  ///
  /// 0 none or pending, 1 published, 2 protected from a pre-witness replacement. Presence alone
  /// cannot answer this because publication clears the marking.
  [[nodiscard]] expected<std::uint8_t, Error> last_refill_outcome(PortId port) const noexcept;

  /// Retire selector history after the input's physical EOS witness has retired.
  ///
  /// Unlike ordinary selector retirement, this terminal boundary may also release the finite
  /// ReuseLast claim on the supplied records. The operation fails closed unless every incident
  /// subscriber has published EOS and no activation, ingress reservation, or unlisted temporal
  /// claim remains.
  [[nodiscard]] expected<void, Error> retire_terminal_selector_history_samples(
      VertexId owner, PortId port, std::span<const TemporalSampleRecord> samples) noexcept;

  /// Clear retained samples, activation state, loans, and readiness generations while disarmed.
  ///
  /// A session containing an externally driven import returns \ref ErrorCode::kNotSupported before
  /// mutating state. A future catalog-owned atomic generation handoff is required before reset can
  /// safely detach and replace an independently driven peer.
  [[nodiscard]] expected<void, Error> reset() noexcept;

  /// Reset one disarmed, quiescent Operator's Channel state without disturbing unrelated owners.
  ///
  /// An incident externally driven import returns \ref ErrorCode::kNotSupported before mutating
  /// state. Unrelated external imports do not prevent a scoped reset.
  [[nodiscard]] expected<void, Error> reset(VertexId owner) noexcept;

  /// Return whether a lifecycle restart union owns any bound Channel state.
  ///
  /// This cold, read-only preflight uses the same exact owner identities as \ref commit_restart,
  /// including Channel consumers whose readiness source is a clock or physical event rather than
  /// Channel arrival. Invalid or duplicate owner sets return `kInvalidArgument`.
  [[nodiscard]] expected<bool, Error> restart_affects(
      std::span<const VertexId> restarted_owners,
      std::span<const VertexId> affected_receivers) const noexcept;

  /// Return whether one receiver has a terminal external import awaiting lifecycle replacement.
  ///
  /// The query becomes false after the plan-admitted generation budget is exhausted. RunSession
  /// uses it only after the receiver's terminal input drain closes, so an explicitly
  /// restart-capable import remains live for a control request instead of racing natural run
  /// completion.
  [[nodiscard]] bool awaits_external_restart(VertexId receiver) const noexcept;

  /// Commit one successful lifecycle restart union after the readiness pump has joined.
  ///
  /// Every restarted owner advances each owned output generation exactly once. Each affected
  /// receiver advances one Channel readiness generation invalidated by a restarted producer or by
  /// its own lifecycle restart. With all publishers fenced, the operation discards at most each
  /// affected process-local subscriber's frozen queue depth, clears retained typed state exactly
  /// once, and leaves non-incident Operators untouched. Both spans must be nonempty, contain valid
  /// unique vertices, and every restarted owner must also appear in \p affected_receivers.
  ///
  /// An incident externally driven import returns \ref ErrorCode::kNotSupported before transport
  /// consumption or generation publication. Failure after process-local transport consumption
  /// terminalizes the session before publishing any new generation.
  ///
  /// @param restarted_owners Operators whose lifecycle transaction publishes a successor
  ///     generation.
  /// @param affected_receivers Operators whose Channel input/readiness state is invalidated by the
  ///     restart union.
  /// @return Success when the complete union is committed atomically, or a typed error otherwise.
  [[nodiscard]] expected<void, Error> commit_restart(
      std::span<const VertexId> restarted_owners,
      std::span<const VertexId> affected_receivers) noexcept;

  /// Commit a single-owner restart through the union implementation.
  [[nodiscard]] expected<void, Error> commit_restart(
      VertexId target, std::span<const VertexId> affected_receivers) noexcept {
    const std::array restarted{target};
    return commit_restart(restarted, affected_receivers);
  }

  /// Return the first pump or binding failure without mutating session state.
  [[nodiscard]] expected<void, Error> status() const noexcept;

  /// @brief Return the registered structural identity of the first terminal Channel failure.
  ///
  /// An empty result means the terminal error has no more-specific registered identity. The
  /// source-private value lets RunSession select a diagnostic without parsing or allocating text.
  ///
  /// @return The registered diagnostic identity, or an empty optional when no more-specific
  /// identity is available.
  [[nodiscard]] std::optional<StructuralDiagnosticId> terminal_diagnostic() const noexcept;

  /// Return cold channel-pump affinity failure context, when placement caused startup failure.
  [[nodiscard]] const RuntimeSupportThreadPlacementFailure* support_thread_placement_failure()
      const noexcept;

  /// Return the owning Operator of the first terminal Channel input failure.
  ///
  /// This source-private identity is zero when the failure is not input-scoped. It allows
  /// RunSession to retain exact diagnostic ownership without parsing text or searching plan state.
  [[nodiscard]] VertexId terminal_owner() const noexcept;

  /// Attach the source-tree teardown observer carried by a test execution inventory.
  ///
  /// RunSession owns this one cold forwarding seam because production translation units must not
  /// depend on the broader Channel mutation API used by unit tests.
  [[nodiscard]] expected<void, Error> set_teardown_observer_for_test(
      void* state, RuntimeChannelTeardownObserver observer) noexcept;

  /// Inject one source-tree activation-retirement failure carried by a test inventory.
  [[nodiscard]] expected<void, Error> fail_next_activation_retirement_for_test() noexcept;

 private:
  friend class RuntimeChannelSessionTestAccess;

  class Impl;

  enum class SampleRetirementKind : std::uint8_t {
    kSelectorHistory,
    kTerminalHistory,
  };

  explicit RuntimeChannelSession(std::unique_ptr<Impl> impl) noexcept;

  [[nodiscard]] expected<void, Error> retire_samples_impl(
      PortId port, std::span<const TemporalSampleRecord> samples,
      SampleRetirementKind retirement_kind, std::uint32_t* promoted,
      const RuntimeChannelFreshnessReservation* freshness_reservation,
      std::int64_t* promoted_capture) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::detail
