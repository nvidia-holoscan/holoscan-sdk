/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <holoscan/channel/endpoint_interfaces.hpp>

#include "runtime_channel_binding.hpp"

namespace holoscan::detail {

/// Give source-tree tests a narrow seam for scripted publication-phase conformance.
///
/// This declaration is compiled only into source-tree Runtime tests and benchmarks. It cannot
/// enter an installed header or plan artifact and may replace publishers only while the session
/// is disarmed.
class RuntimeChannelSessionTestAccess {
 public:
  /// Describe allocated input resources independently of live payload occupancy.
  struct InputResourceState {
    std::size_t subscriber_count{};
    std::size_t payload_slot_count{};
    bool claim_pool_present{};
  };

  /// Return an existing input's resource inventory while the session is disarmed.
  [[nodiscard]] static expected<InputResourceState, Error> input_resource_state(
      const RuntimeChannelSession& session, PortId input) noexcept;

  /// Return the published active logical queue depth for one payload edge.
  [[nodiscard]] static expected<std::uint32_t, Error> active_queue_depth(
      const RuntimeChannelSession& session, EdgeId edge) noexcept;

  /// Bind one source-tree observation control to an exact published rate-generation queue row.
  ///
  /// The join authenticates plan digest/generation, run/rate generation, recipe-to-edge identity,
  /// and the live selected logical depth. It records no occupancy or counter merely by binding.
  [[nodiscard]] static expected<void, Error> bind_queue_observation(
      const RuntimeChannelSession& session, const ExecutionPlanState& plan,
      const RuntimeRateState& rates, RuntimeRateSnapshotCapture capture, EdgeId edge,
      const std::shared_ptr<RuntimeChannelQueueObservationTestControl>& control) noexcept;

  /// Sample one edge without attributing a queue-facing attempt.
  ///
  /// This establishes or refreshes the typed-retention high-water baseline without fabricating
  /// counter evidence. The control must still name the currently published rate generation.
  [[nodiscard]] static expected<void, Error> sample_queue_observation(
      const RuntimeChannelSession& session, const RuntimeRateState& rates,
      RuntimeRateSnapshotCapture capture, EdgeId edge,
      const std::shared_ptr<RuntimeChannelQueueObservationTestControl>& control) noexcept;

  /// Return the HoloMQ name a scripted peer uses to reach one process-local service.
  [[nodiscard]] static expected<std::string, Error> process_local_transport_name(
      const RuntimeChannelSession& session, std::string_view service_id);

  /// Format the process-local transport scope from explicit identity inputs.
  [[nodiscard]] static std::string process_local_transport_scope(std::uint64_t process,
                                                                 std::uint64_t ordinal,
                                                                 std::uint64_t incarnation);

  /// Fail the next validated activation retirement before mutating Channel state.
  [[nodiscard]] static expected<void, Error> fail_next_activation_retirement(
      RuntimeChannelSession& session) noexcept;

  /// Fail the session as a pump or receive failure would.
  static void fail(RuntimeChannelSession& session, ErrorCode code) noexcept;

  /// Fail one fixed-Tensor shape allocation after the requested number of successful copies.
  [[nodiscard]] static expected<void, Error> fail_fixed_tensor_shape_allocation_after_successes(
      RuntimeChannelSession& session, std::size_t successful_allocations) noexcept;

  /// Observe validated activation retirement before Channel mutates any activation-owned state.
  [[nodiscard]] static expected<void, Error> set_activation_retirement_observer(
      RuntimeChannelSession& session, void* state,
      RuntimeChannelActivationRetirementObserver observer) noexcept;

  /// Install scripted peer policy and publication outcomes for cold-session tests such as
  /// ExhaustedOptionalNoPeerCloseResolvesLocally.
  [[nodiscard]] static expected<void, Error> replace_output_publishers(
      RuntimeChannelSession& session, PortId output,
      std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers,
      std::span<const std::uint32_t> expected_recipient_counts = {},
      bool refresh_before_emit = false, bool peer_required = true) noexcept;

  /// Count resolved terminal arms in a cold output, for example after required-peer abandonment
  /// in ExhaustedRequiredExportAbandonmentReturnsClosed. This observes resolution, not EOS receipt.
  [[nodiscard]] static expected<std::size_t, Error> resolved_output_terminal_arms(
      const RuntimeChannelSession& session, PortId output) noexcept;

  /// Replace one descriptor-schema output with a scripted non-local publisher while disarmed.
  [[nodiscard]] static expected<void, Error> replace_output_with_nonlocal_tensor_schema_publisher(
      RuntimeChannelSession& session, PortId output,
      std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher> publisher) noexcept;

  /// Throw once before submission or after the requested number of Tensor fanout accepts.
  [[nodiscard]] static expected<void, Error> throw_tensor_fanout_after_accepts(
      RuntimeChannelSession& session, PortId output, std::size_t accepted_count) noexcept;

  /// Exhaust one exact catalog Tensor output's terminal loan before backend submission.
  [[nodiscard]] static expected<void, Error> fail_next_catalog_tensor_terminal_pre_submit(
      RuntimeChannelSession& session, PortId output) noexcept;

  /// Set one exact catalog Tensor output to permanent endpoint operation-sequence exhaustion.
  [[nodiscard]] static expected<void, Error> exhaust_catalog_tensor_operation_sequence(
      RuntimeChannelSession& session, PortId output) noexcept;

  /// Replace one disarmed process-local subscriber, below any wait-set attachment, for a direct
  /// source-tree receive test.
  [[nodiscard]] static expected<void, Error> replace_input_subscriber(
      RuntimeChannelSession& session, PortId input,
      std::unique_ptr<channel_internal::InlinePayloadSubscriber> replacement) noexcept;

  /// Execute one direct receive through a source-tree subscriber replacement while disarmed.
  [[nodiscard]] static expected<void, Error> receive_one(RuntimeChannelSession& session,
                                                         PortId input) noexcept;

  /// Select concurrent-admission blocked-head accounting on one disarmed session.
  [[nodiscard]] static expected<void, Error> enable_concurrent_admission(
      RuntimeChannelSession& session) noexcept;

  /// Record one subscriber's blocked-head transition under its transport mutex, as a receive does.
  [[nodiscard]] static expected<void, Error> set_transport_head_blocked(
      RuntimeChannelSession& session, PortId input, bool blocked) noexcept;

  /// Store EOS on one subscriber of an armed session under its transport and cache mutexes, as a
  /// concurrently completing receive would.
  [[nodiscard]] static expected<void, Error> receive_end_of_stream_concurrently(
      RuntimeChannelSession& session, PortId input, std::uint64_t transport_sequence) noexcept;

  /// Run one emit-time admission over these recipient inputs of a disarmed session, as an
  /// accepted publication under the emit-recipients experiment does.
  [[nodiscard]] static expected<void, Error> admit_recipients(
      RuntimeChannelSession& session, std::span<const PortId> inputs) noexcept;

  /// Run one retirement retry over these released inputs of a disarmed session. Repeats are
  /// allowed, so a small fixture can exercise released lists of any length.
  [[nodiscard]] static expected<void, Error> retry_released_inputs(
      RuntimeChannelSession& session, std::span<const PortId> inputs) noexcept;

  /// Whether the session's retry signal reports a possibly blocked transport head.
  [[nodiscard]] static bool blocked_transport_retry_pending(
      const RuntimeChannelSession& session) noexcept;

  /// Whether one input's staged transport head is hidden from the pump's wait.
  [[nodiscard]] static expected<bool, Error> blocked_head_masked(
      const RuntimeChannelSession& session, PortId input) noexcept;

  /// Run one pump wait, as the pump does, on a session without a running pump.
  [[nodiscard]] static expected<RuntimeChannelPumpWaitObservation, Error> wait_as_pump(
      RuntimeChannelSession& session, PortId input, std::chrono::nanoseconds timeout) noexcept;

  /// Execute and classify one real disarmed transport receive for an authenticated observation.
  ///
  /// The returned status is the receive status. Counter evidence is derived from its actual
  /// result plus exact before/after typed and staged inventories; no caller supplies an outcome.
  [[nodiscard]] static expected<void, Error> receive_one_observed(
      RuntimeChannelSession& session, const RuntimeRateState& rates,
      RuntimeRateSnapshotCapture capture, PortId input, EdgeId edge,
      const std::shared_ptr<RuntimeChannelQueueObservationTestControl>& control) noexcept;

  /// Temporarily alias one disarmed subscriber's target for exact-edge observation tests.
  ///
  /// Authored Graphs reject input fan-in. This source-tree-only seam can model an ambiguous target
  /// so receive observation proves it consumes the subscriber already authenticated by edge.
  [[nodiscard]] static expected<void, Error> set_observation_target_alias(
      RuntimeChannelSession& session, EdgeId edge, PortId target) noexcept;

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

  /// Hold one real reuse replacement reservation around a scripted cold-session owner action.
  ///
  /// No data is received or committed. Destruction exercises the production rollback protocol
  /// after the callback returns, without adding an observer to a production execution path.
  [[nodiscard]] static expected<void, Error> with_reuse_replacement_reservation(
      RuntimeChannelSession& session, PortId input, void* state,
      void (*callback)(void*) noexcept) noexcept;

  /// Fail one staged-newer-data reuse replacement after reserving its exact destination.
  [[nodiscard]] static expected<void, Error> fail_next_reuse_replacement_receive(
      RuntimeChannelSession& session, PortId input) noexcept;

  /// Stop one Tensor input staging arrivals, as an external import does not, so a newer arrival
  /// behind a pinned typed slot stays at the endpoint head.
  [[nodiscard]] static expected<void, Error> stop_tensor_ingress_staging(
      RuntimeChannelSession& session, PortId input) noexcept;

  /// Reject one staged-newer-data reuse replacement with one retryable source-tree error.
  [[nodiscard]] static expected<void, Error> reject_next_reuse_replacement_receive(
      RuntimeChannelSession& session, PortId input, ErrorCode error) noexcept;

  /// Observe the exact transport-staged data sequence for one source-tree input.
  [[nodiscard]] static expected<std::optional<std::uint64_t>, Error> staged_data_sequence(
      const RuntimeChannelSession& session, PortId input) noexcept;

  /// Discard one exact process-local transport data head while holding its ingress serialization.
  ///
  /// This cold-session seam lets a test deterministically advance from a probed staged identity to
  /// its successor without opening a production scheduler or pump race.
  [[nodiscard]] static expected<void, Error> discard_staged_data_head(
      RuntimeChannelSession& session, PortId input) noexcept;

  /// Fail the auxiliary process-local retry after one committed freshness handoff.
  [[nodiscard]] static expected<void, Error> fail_next_freshness_handoff_retry(
      RuntimeChannelSession& session) noexcept;

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

  /// Model a later typed receive overtaking one already-staged survivor on a lossy input.
  [[nodiscard]] static expected<void, Error> make_staged_sample_older_than_typed_sample(
      RuntimeChannelSession& session, PortId input) noexcept;

  /// Exercise deferred promotion and post-commit retry for a live stale staging reservation.
  [[nodiscard]] static expected<void, Error> commit_staged_reservation_after_newer_typed_overtake(
      RuntimeChannelSession& session, PortId input,
      bool staged_is_committed_freshness_target = false) noexcept;

  /// Temporarily include a second input's subscriber in one freshness-identity scan.
  [[nodiscard]] static expected<void, Error> set_freshness_subscriber_alias(
      RuntimeChannelSession& session, PortId input, PortId subscriber_input, bool include) noexcept;

  /// Saturate every independently reset slot and subscriber field for reset-equivalence tests.
  [[nodiscard]] static expected<void, Error> saturate_single_occupied_reset_state(
      RuntimeChannelSession& session, PortId input) noexcept;

  /// Inject one direct inline payload with caller-selected metadata through the production store.
  ///
  /// The session must remain unarmed because this test seam bypasses the runtime event loop.
  [[nodiscard]] static expected<void, Error> inject_direct_inline_payload(
      RuntimeChannelSession& session, PortId input, std::span<const std::byte> payload,
      SampleMetadata metadata) noexcept;

  [[nodiscard]] static expected<void, Error> with_input_lock(
      RuntimeChannelSession& session, PortId input, void* state,
      RuntimeChannelInputLockObserver observer) noexcept;

  /// Run `observer` while holding the session's pump scan lock, as the pump does during a scan.
  [[nodiscard]] static expected<void, Error> with_pump_scan_lock(
      RuntimeChannelSession& session, void* state,
      RuntimeChannelInputLockObserver observer) noexcept;

  /// Receive every transported delivery into typed storage, as one pump scan does. The caller holds
  /// the pump scan lock through with_pump_scan_lock().
  [[nodiscard]] static expected<void, Error> receive_as_pump(
      RuntimeChannelSession& session) noexcept;

  /// Observe an ordinary readiness poll that cannot acquire one exact input cache mutex.
  ///
  /// The observer is installed before the session is armed and remains immutable until teardown.
  [[nodiscard]] static expected<void, Error> set_input_poll_contention_observer(
      RuntimeChannelSession& session, PortId input, void* state,
      RuntimeChannelInputPollContentionObserver observer) noexcept;

  /// Observe each failed try_lock that one bounded contention retry follows with another attempt
  /// on one exact input; the final failure that expires the budget is not reported.
  ///
  /// A source-tree test uses this to release a real holder only after the retry loop has actually
  /// re-armed, which makes the release-during-budget success path deterministic. Installed before
  /// the session is armed and immutable until teardown.
  [[nodiscard]] static expected<void, Error> set_input_contention_retry_attempt_observer(
      RuntimeChannelSession& session, PortId input, void* state,
      RuntimeChannelInputPollContentionObserver observer) noexcept;

  /// Replace the bounded contention retry budget of an unarmed session.
  ///
  /// Production keeps \ref kContentionRetryBudget; a test widens it so a scheduler pause cannot
  /// expire the retry whose success path it is proving.
  [[nodiscard]] static expected<void, Error> set_contention_retry_budget(
      RuntimeChannelSession& session, std::chrono::nanoseconds budget) noexcept;

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

  /// Return the private activation-cut progress evidence for source-tree conformance tests.
  [[nodiscard]] static expected<RuntimeChannelProcessLocalProgressState, Error>
  process_local_progress_state(const RuntimeChannelSession& session) noexcept;

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
  /// sample may already occupy that subscriber. Baseline accepts a nonzero maximum byte span,
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
      TensorBoundaryContractRecord requirements) noexcept;

  /// Configure one disarmed test subscriber to exercise the external metadata boundary.
  ///
  /// Both clock identifiers must be valid or both absent. The transport endpoint itself remains
  /// unchanged; only source-tree metadata-admission tests may use this seam.
  [[nodiscard]] static expected<void, Error> set_external_metadata_contract(
      RuntimeChannelSession& session, PortId input, ClockId catalog_capture_clock,
      ClockId plan_capture_clock) noexcept;

  [[nodiscard]] static expected<RuntimeChannelInputResetState, Error> input_reset_state(
      const RuntimeChannelSession& session, PortId input) noexcept;

  /// Observe the production no-claim scalar binding without copying its selected sample.
  [[nodiscard]] static expected<RuntimeScalarReadBinding, Error> selected_scalar_binding(
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

  /// Run the controller descriptor-import initialization on the calling thread without changing
  /// the frozen import inventory, as an activation cut or owner poll on that thread would.
  [[nodiscard]] static expected<void, Error> initialize_descriptor_import_on_this_thread(
      RuntimeChannelSession& session) noexcept;

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

  /// Inspect or seed exhaustion only while the session is disarmed and the output is inactive.
  [[nodiscard]] static expected<void, Error> set_output_next_sequence(
      RuntimeChannelSession& session, PortId output, std::uint64_t next_sequence) noexcept;

  [[nodiscard]] static expected<std::uint64_t, Error> output_next_sequence(
      const RuntimeChannelSession& session, PortId output) noexcept;

  [[nodiscard]] static expected<void, Error> set_output_active_generation(
      RuntimeChannelSession& session, PortId output, std::uint64_t active_generation) noexcept;

  [[nodiscard]] static expected<std::uint64_t, Error> output_active_generation(
      const RuntimeChannelSession& session, PortId output) noexcept;

  [[nodiscard]] static expected<void, Error> fail_next_restart_after_transport_discard(
      RuntimeChannelSession& session, ErrorCode code) noexcept;
};

}  // namespace holoscan::detail
