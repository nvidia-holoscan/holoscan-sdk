/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>

#include <array>
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

#include "../core/diagnostic_registry.hpp"
#include "port_binding.hpp"
#include "temporal_input_cache.hpp"

namespace holoscan::channel_internal {
enum class CatalogSchemaFrameCorruption : std::uint8_t;
}

namespace holoscan::detail {

struct ExecutionPlanState;
class EndpointClockHealthRuntimeCoordinator;
class RuntimeChannelSessionTestAccess;

enum class RuntimeChannelTeardownPhase : std::uint8_t {
  kPumpJoined,
  kEndpointChildrenDestroyed,
  kRuntimeDestroyed,
};

using RuntimeChannelTeardownObserver = void (*)(void*, RuntimeChannelTeardownPhase) noexcept;
using RuntimeChannelInputLockObserver = void (*)(void*) noexcept;
using RuntimeChannelPollSelectionObserver = void (*)(void*) noexcept;
using RuntimeChannelClockHealthLockObserver = void (*)(void*) noexcept;

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

struct RuntimeChannelInputResetState {
  std::uint64_t active_generation{};
  std::uint64_t view_generation{};
  std::uint64_t producer_epoch{};
  std::uint64_t next_arrival_sequence{};
  std::uint64_t next_global_arrival_order{};
  std::size_t occupied_samples{};
  std::size_t pending_direct_ingress_samples{};
  ClockId catalog_capture_clock{};
  ClockId plan_capture_clock{};
  /// Producer-incarnation token latched by a catalog import endpoint; zero when the endpoint has
  /// not accepted any operation or when the subscriber is not a catalog import.
  std::uint64_t latched_producer_generation{};
  bool end_of_stream_received{};
};

struct RuntimeChannelPumpIndexState {
  std::size_t wait_handle_count{};
  std::size_t drain_subscriber_count{};
  bool exact{};
};

/// @brief Describe the private endpoint families materialized for one Tensor edge.
struct RuntimeChannelTensorTransferState {
  std::size_t buffer_handle_publishers{};  ///< BufferHandle publisher endpoint count.
  std::size_t inline_publishers{};         ///< Inline-payload publisher endpoint count.
  bool buffer_handle_subscriber{};         ///< Whether the input uses a BufferHandle subscriber.
  bool inline_subscriber{};                ///< Whether the input uses an inline subscriber.
};

/// Describe one process-local replay declaration without placing C++ type state in the plan.
struct RuntimeChannelPortReplayRecord {
  PortId port{};
  VertexId owner{};
  PortDirection direction{PortDirection::kInput};
  void* declaration{};
  const RuntimePortTypeOps* type_ops{};
};

/// Select retained typed samples for one input during one activation.
struct RuntimeChannelActivationSelection {
  PortId port{};
  InputSelection selection{};
  std::span<const TemporalSampleRecord> samples;
  bool batch{};
  bool destructive{};
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

/// Own the process-local Channel endpoints and typed sample retention for one RunSession.
///
/// Construction performs all endpoint creation, WaitSet attachment, payload-storage allocation,
/// generic-loan construction, and replay-port binding before lifecycle entry. `arm()` is the sole
/// transition that starts readiness observation; `cancel()` followed by `join()` must complete
/// before lifecycle stop/release destroys replay declarations. The class is private Runtime
/// substrate and exposes no Channel or HoloMQ object to Operator code.
class RuntimeChannelSession {
 public:
  RuntimeChannelSession(const RuntimeChannelSession&) = delete;
  RuntimeChannelSession& operator=(const RuntimeChannelSession&) = delete;
  RuntimeChannelSession(RuntimeChannelSession&&) = delete;
  RuntimeChannelSession& operator=(RuntimeChannelSession&&) = delete;
  ~RuntimeChannelSession();

  /// @brief Materialize bounded canonical-scalar and private constructed-loan bindings.
  ///
  /// @param plan Accepted immutable execution plan.
  /// @param replay_ports Cold replay declarations for ports materialized in this process.
  /// @param process_group Process group selected for this Runtime worker, or invalid for the
  ///        complete local plan.
  /// @param readiness_wake Optional borrowed callback invoked after a joined pump scan publishes
  ///        typed input readiness. Its state must remain valid through \ref join.
  /// @return The fully reconstructed Channel session, or a typed reconstruction error.
  [[nodiscard]] static expected<std::unique_ptr<RuntimeChannelSession>, Error> create(
      const ExecutionPlanState& plan, std::span<const RuntimeChannelPortReplayRecord> replay_ports,
      ProcessGroupId process_group = {}, RuntimeChannelReadinessWake readiness_wake = {});

  /// Return whether reconstruction materialized any Channel activation state for an Operator.
  ///
  /// This cold query distinguishes bound Channel endpoints from replay-only port declarations.
  [[nodiscard]] bool has_materialized_activation_state(VertexId owner) const noexcept;

  /// Return whether reconstruction materialized Channel input state for an Operator.
  ///
  /// Output-only owners still require activation state for publication but must not acquire an
  /// input-arrival readiness source.
  [[nodiscard]] bool has_materialized_input_state(VertexId owner) const noexcept;

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

  /// Refresh publisher connection state before the global start barrier.
  [[nodiscard]] expected<void, Error> refresh_connections() noexcept;

  /// Start the single joined WaitSet pump after every Operator completed `kStart`.
  [[nodiscard]] expected<void, Error> arm() noexcept;

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

  /// Retire cache history that the temporal selector has proved no longer observable.
  ///
  /// This releases the selector's payload claim atomically for the supplied records. Physical
  /// storage remains stable while a readiness witness, active callback, or finite ReuseLast claim
  /// still owns it, and is reclaimed when the final independent claim retires.
  [[nodiscard]] expected<void, Error> retire_samples(
      PortId port, std::span<const TemporalSampleRecord> samples) noexcept;

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

  /// Return the owning Operator of the first terminal Channel input failure.
  ///
  /// This source-private identity is zero when the failure is not input-scoped. It allows
  /// RunSession to retain exact diagnostic ownership without parsing text or searching plan state.
  [[nodiscard]] VertexId terminal_owner() const noexcept;

 private:
  friend class RuntimeChannelSessionTestAccess;

  class Impl;

  explicit RuntimeChannelSession(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

/// Give source-tree tests a narrow seam for scripted publication-phase conformance.
///
/// This type is declared only in the private Runtime source tree. It cannot enter an installed
/// header or plan artifact and may replace publishers only while the session is disarmed.
class RuntimeChannelSessionTestAccess {
 public:
  [[nodiscard]] static expected<void, Error> replace_output_publishers(
      RuntimeChannelSession& session, PortId output,
      std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers,
      std::span<const std::uint32_t> expected_recipient_counts = {},
      bool refresh_before_emit = false) noexcept;

  /// Throw once before submission or after the requested number of Tensor fanout accepts.
  [[nodiscard]] static expected<void, Error> throw_tensor_fanout_after_accepts(
      RuntimeChannelSession& session, PortId output, std::size_t accepted_count) noexcept;

  /// Replace one disarmed process-local subscriber below its wait-set attachment for a direct
  /// source-tree receive test.
  [[nodiscard]] static expected<void, Error> replace_input_subscriber(
      RuntimeChannelSession& session, PortId input,
      std::unique_ptr<channel_internal::InlinePayloadSubscriber> replacement) noexcept;

  /// Execute one direct receive through a source-tree subscriber replacement while disarmed.
  [[nodiscard]] static expected<void, Error> receive_one(RuntimeChannelSession& session,
                                                         PortId input) noexcept;

  /// Return the number of typed slots reserved by an in-progress transport receive or import.
  [[nodiscard]] static expected<std::size_t, Error> typed_ingress_reservations(
      const RuntimeChannelSession& session, PortId input) noexcept;

  /// Attempt ordinary unclaimed-slot reclamation on the one reserved typed destination.
  [[nodiscard]] static expected<void, Error> reclaim_reserved_input_slot(
      RuntimeChannelSession& session, PortId input) noexcept;

  /// Install a source-private observer after transport data is acquired but before typed commit.
  [[nodiscard]] static expected<void, Error> set_typed_ingress_observer(
      RuntimeChannelSession& session, PortId input, void* state,
      void (*observer)(void*) noexcept) noexcept;

  /// Drain one externally driven import through its ordered catalog transaction.
  [[nodiscard]] static expected<void, Error> drain_external_inputs(RuntimeChannelSession& session,
                                                                   VertexId owner) noexcept;

  /// Exercise ordered terminal retention directly while the session is disarmed.
  [[nodiscard]] static expected<void, Error> store_end_of_stream(
      RuntimeChannelSession& session, PortId input, std::uint64_t transport_sequence) noexcept;

  /// Corrupt the next process-local generated-schema frame below its trusted encoder.
  [[nodiscard]] static expected<void, Error> corrupt_next_process_local_schema_publication(
      RuntimeChannelSession& session, PortId output,
      channel_internal::CatalogSchemaFrameCorruption corruption) noexcept;

  [[nodiscard]] static expected<void, Error> set_readiness_sequence(
      PortBinding& binding, std::uint64_t sequence) noexcept;

  [[nodiscard]] static expected<void, Error> set_single_occupied_witness_state(
      RuntimeChannelSession& session, PortId input, std::uint64_t readiness_sequence,
      bool event_emitted, bool witness_retired) noexcept;

  /// Configure whether one direct lossy subscriber may replace a sample before its witness emits.
  [[nodiscard]] static expected<void, Error> set_pre_witness_replacement_allowed(
      RuntimeChannelSession& session, PortId input, bool allowed) noexcept;

  /// Set the four independent claims that make one occupied typed slot ineligible for replacement.
  [[nodiscard]] static expected<void, Error> set_single_occupied_claim_state(
      RuntimeChannelSession& session, PortId input, bool active_selected, bool reuse_retained,
      bool temporal_retained, bool retirement_selected) noexcept;

  /// Inject one direct inline payload with caller-selected metadata through the production store.
  ///
  /// The session must remain unarmed because this test seam bypasses the runtime event loop.
  [[nodiscard]] static expected<void, Error> inject_direct_inline_payload(
      RuntimeChannelSession& session, PortId input, std::span<const std::byte> payload,
      SampleMetadata metadata) noexcept;

  [[nodiscard]] static expected<void, Error> with_input_lock(
      RuntimeChannelSession& session, PortId input, void* state,
      RuntimeChannelInputLockObserver observer) noexcept;

  /// Pause one readiness poll after it saves the oldest candidate and releases all input locks.
  ///
  /// The source-tree HoloMQ concurrency test uses this one-shot seam to replace a drop-oldest slot
  /// before the poll reacquires its owning lock. The observer must be installed before the tested
  /// poll and cannot be installed while another observer remains pending.
  [[nodiscard]] static expected<void, Error> set_poll_selection_observer(
      RuntimeChannelSession& session, void* state,
      RuntimeChannelPollSelectionObserver observer) noexcept;

  [[nodiscard]] static expected<void, Error> set_teardown_observer(
      RuntimeChannelSession& session, void* state,
      RuntimeChannelTeardownObserver observer) noexcept;

  /// Pause one clock-health reset immediately before it acquires the shared serialization mutex.
  ///
  /// The one-shot source-tree seam lets a concurrency regression prove that a Channel reset and
  /// RunSession expiry/report transaction cannot enter the coordinator together.
  [[nodiscard]] static expected<void, Error> set_clock_health_lock_observer(
      RuntimeChannelSession& session, void* state,
      RuntimeChannelClockHealthLockObserver observer) noexcept;

  /// Return whether this session retained the monitored-only serialization boundary.
  [[nodiscard]] static bool clock_health_serialization_bound(
      const RuntimeChannelSession& session) noexcept;

  [[nodiscard]] static expected<void, Error> set_input_reset_counters(
      RuntimeChannelSession& session, PortId input, std::uint64_t active_generation,
      std::uint64_t view_generation, std::uint64_t producer_epoch) noexcept;

  [[nodiscard]] static expected<void, Error> set_arrival_counters(
      RuntimeChannelSession& session, PortId input, std::uint64_t next_arrival_sequence,
      std::uint64_t next_global_arrival_order) noexcept;

  /// Model coincident session-wide progress totals while preserving a real blocked transport head.
  [[nodiscard]] static expected<void, Error> set_process_local_progress_counts(
      RuntimeChannelSession& session, std::uint64_t published, std::uint64_t received) noexcept;

  /// @brief Inject one metadata record at the typed-retention boundary while the pump is disarmed.
  ///
  /// Source-tree conformance tests use this seam to prove malformed transport metadata is rejected
  /// before sample construction, counter reservation, or cache mutation. The payload scratch is
  /// deterministically zeroed and remains private.
  ///
  /// @param session The cold, disarmed Runtime Channel session under test.
  /// @param input The input port whose typed-retention boundary receives the metadata.
  /// @param metadata The transport metadata record to validate and retain.
  /// @param terminal_diagnostic Optional output for the registered terminal diagnostic identity.
  /// @return Success when the metadata is accepted; otherwise the production validation error.
  [[nodiscard]] static expected<void, Error> inject_received_metadata(
      RuntimeChannelSession& session, PortId input, SampleMetadata metadata,
      std::optional<StructuralDiagnosticId>* terminal_diagnostic = nullptr) noexcept;

  /// @brief Inject one Tensor sample at the typed-retention boundary.
  ///
  /// The session must be cold and disarmed, and \p input must identify a Tensor subscriber. The
  /// sample passes through the production physical, logical, metadata, ordering, and bounded-cache
  /// validation path before retention. Contract rejection occurs before arrival-counter or cache
  /// mutation.
  ///
  /// @param session Runtime Channel session whose private subscriber is exercised.
  /// @param input Exact nonzero input \ref PortId.
  /// @param sample Tensor sample consumed by the attempted injection.
  /// @return Success when the sample is retained, or a checked session, identity, contract,
  ///         ordering, or capacity error.
  [[nodiscard]] static expected<void, Error> inject_received_tensor(RuntimeChannelSession& session,
                                                                    PortId input,
                                                                    Sample<Tensor> sample) noexcept;

  /// @brief Configure a frozen host-Tensor import contract for a conformance test.
  ///
  /// The session must be cold and disarmed, \p input must identify a Tensor subscriber, and no
  /// sample may already occupy that subscriber. EA1 accepts a nonzero maximum byte span,
  /// \c MemoryKind::kHost, and \c TensorStridePolicy::kContiguousOnly. This seam changes only the
  /// private receive-validation state; it does not construct, refresh, or replace a transport
  /// endpoint.
  ///
  /// @param session Runtime Channel session whose private subscriber is configured.
  /// @param input Exact nonzero input \ref PortId.
  /// @param requirements Complete logical Tensor requirements to install.
  /// @return Success when the contract is installed; \c kInvalidArgument for an invalid session,
  ///         state, port type, or unsupported contract, \c kNotFound when no subscriber owns
  ///         \p input, or \c kBackpressured when retained storage is occupied.
  [[nodiscard]] static expected<void, Error> set_external_tensor_contract(
      RuntimeChannelSession& session, PortId input,
      LogicalTensorRequirementsRecord requirements) noexcept;

  /// Configure one disarmed test subscriber to exercise the external metadata boundary.
  ///
  /// Both clock identifiers must be valid or both absent. The transport endpoint itself remains
  /// unchanged; only source-tree metadata-admission tests may use this seam.
  [[nodiscard]] static expected<void, Error> set_external_metadata_contract(
      RuntimeChannelSession& session, PortId input, ClockId catalog_capture_clock,
      ClockId plan_capture_clock) noexcept;

  [[nodiscard]] static expected<RuntimeChannelInputResetState, Error> input_reset_state(
      const RuntimeChannelSession& session, PortId input) noexcept;

  [[nodiscard]] static expected<RuntimeChannelPumpIndexState, Error> pump_index_state(
      const RuntimeChannelSession& session) noexcept;

  /// Initialize one cold raw-Tensor subscriber's controller import device.
  ///
  /// This source-tree-only seam supplies the requested frozen device to an existing descriptor
  /// endpoint, invokes the production controller initialization path, and leaves transport
  /// disarmed. It is used to prove exact-device selection, unavailable-device failure, and caller
  /// restoration without racing the background pump.
  [[nodiscard]] static expected<void, Error> initialize_controller_descriptor_import(
      RuntimeChannelSession& session, PortId input, DeviceId device) noexcept;

  /// Configure one cold raw-Tensor subscriber's frozen import device without initializing it.
  [[nodiscard]] static expected<void, Error> configure_descriptor_import_device(
      RuntimeChannelSession& session, PortId input, DeviceId device) noexcept;

  /// Return how many real controller-thread descriptor initializations completed.
  [[nodiscard]] static expected<std::uint64_t, Error>
  controller_descriptor_import_initialization_count(const RuntimeChannelSession& session) noexcept;

  /// Select two raw-Tensor import devices and report the resulting CUDA device.
  ///
  /// An invalid second device represents a host/unspecified subscriber and must return the cursor
  /// to its ambient device. This source-tree-only seam also restores the caller before returning.
  [[nodiscard]] static expected<DeviceId, Error> cuda_device_after_import_selection(
      DeviceId selected, DeviceId fallback) noexcept;

  /// @brief Return the private endpoint families selected for one Tensor edge.
  /// @param session Cold or disarmed Runtime Channel session to inspect.
  /// @param output Exact output \ref PortId of the edge.
  /// @param input Exact input \ref PortId of the edge.
  /// @return Endpoint-family snapshot, or a checked identity/session error.
  [[nodiscard]] static expected<RuntimeChannelTensorTransferState, Error> tensor_transfer_state(
      const RuntimeChannelSession& session, PortId output, PortId input) noexcept;

  [[nodiscard]] static expected<void, Error> set_output_active_generation(
      RuntimeChannelSession& session, PortId output, std::uint64_t active_generation) noexcept;

  [[nodiscard]] static expected<std::uint64_t, Error> output_active_generation(
      const RuntimeChannelSession& session, PortId output) noexcept;

  [[nodiscard]] static expected<void, Error> fail_next_restart_after_transport_discard(
      RuntimeChannelSession& session, ErrorCode code) noexcept;
};

}  // namespace holoscan::detail
