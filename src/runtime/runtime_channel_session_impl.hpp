/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <semaphore>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>
#include <nvtx3/nvToolsExt.h>

#include <holoscan/channel/endpoint_interfaces.hpp>
#include <holoscan/channel/manual_tensor_publisher.hpp>
#include <holoscan/channel/runtime.hpp>
#include <holoscan/channel/tensor_endpoints.hpp>
#include <holoscan/channel/types.hpp>

#include "../channel/blocked_head_wait_set.hpp"
#include "../channel/channel_catalog_binding.hpp"
#include "../channel/channel_tensor_loan_internal.hpp"
#include "../channel/schema_envelope_codec.hpp"
#include "../compiler/execution_plan_lookup.hpp"
#include "../compiler/execution_plan_state.hpp"
#include "../core/domain/tensor_access.hpp"
#include "../time/clock_registry_internal.hpp"
#include "bounded_lock.hpp"
#include "channel_runtime_binding_access.hpp"
#include "core/diagnostic_reporting.hpp"
#include "endpoint_clock_health_runtime.hpp"
#include "runtime_channel_binding.hpp"
#include "runtime_channel_storage.hpp"
#include "runtime_fanout_test_control.hpp"
#include "runtime_phase_counters.hpp"
#include "runtime_support_thread_control.hpp"
#include "runtime_support_thread_placement.hpp"
#include "runtime_tensor_resources.hpp"
#include "temporal_preselection.hpp"

namespace holoscan::detail {

inline constexpr auto kPumpWait = std::chrono::milliseconds{10};
// How long the pump keeps probing a refused head before hiding it. A consumer that frees its slot
// within this spin is picked up by the pump's next probe, with no wake from the consumer's thread.
inline constexpr auto kBlockedHeadSpin = std::chrono::microseconds{20};

inline void count_phase(RuntimePhaseCounters* counters, RuntimePhaseCounter counter) noexcept {
  if (counters != nullptr) {
    counters->increment(counter);
  }
}

inline void trace_phase(RuntimePhaseCounters* counters, RuntimePhaseTracePoint point,
                        std::uint64_t source_sequence = 0U) noexcept {
  if (counters != nullptr) {
    counters->trace(point, kNoRuntimePhaseReceiver, {}, source_sequence);
  }
}

inline void trace_output_emit_phase(RuntimePhaseCounters* counters, RuntimePhaseTracePoint point,
                                    PortId output_port, std::uint64_t source_sequence) noexcept {
  if (counters != nullptr) {
    counters->trace(point, kNoRuntimePhaseReceiver, {}, source_sequence, output_port.value());
  }
}

[[nodiscard]] inline Error normalize_terminal_pre_submit_error(Error error) noexcept {
  if (error.code == ErrorCode::kResourceExhausted) {
    error.code = ErrorCode::kBackpressured;
  }
  return error;
}

inline void corrupt_process_local_schema_frame(
    std::span<std::byte> frame,
    const channel_internal::CatalogSchemaFrameCorruption corruption) noexcept {
  constexpr std::size_t kSchemaIdOffset = 8U;
  constexpr std::size_t kPayloadSizeOffset = 24U;
  constexpr std::size_t kEndpointGenerationOffset = 32U;
  constexpr std::size_t kPayloadOffset = kSchemaEnvelopeHeaderSize;

  switch (corruption) {
    case channel_internal::CatalogSchemaFrameCorruption::kMalformedEnvelope:
      channel_internal::schema_envelope_detail::store_little<std::uint32_t>(frame, 4U, 0U);
      break;
    case channel_internal::CatalogSchemaFrameCorruption::kSchemaId:
      frame[kSchemaIdOffset] ^= std::byte{0xffU};
      break;
    case channel_internal::CatalogSchemaFrameCorruption::kPayloadSize:
      channel_internal::schema_envelope_detail::store_little<std::uint32_t>(
          frame, kPayloadSizeOffset, 0U);
      break;
    case channel_internal::CatalogSchemaFrameCorruption::kEndpointGeneration:
      channel_internal::schema_envelope_detail::store_little<std::uint64_t>(
          frame,
          kEndpointGenerationOffset,
          channel_internal::kInitialSchemaEndpointGeneration + 1U);
      break;
    case channel_internal::CatalogSchemaFrameCorruption::kPayloadVerifier:
      std::fill(frame.begin() + kPayloadOffset, frame.end(), std::byte{0xffU});
      break;
    case channel_internal::CatalogSchemaFrameCorruption::kEnvelopeAbi:
      channel_internal::schema_envelope_detail::store_little<std::uint32_t>(frame, 0U, 0U);
      break;
  }
}

[[nodiscard]] inline bool valid_input_selection(const InputSelection selection) noexcept {
  switch (selection.state) {
    case InputState::kSelected:
      return selection.reason == SelectionReason::kNone;
    case InputState::kReused:
    case InputState::kSafeDefault:
    case InputState::kAbsent:
      return selection.reason != SelectionReason::kNone &&
             selection.reason != SelectionReason::kNotSelectedForActivation;
    case InputState::kNotSelected:
      return selection.reason == SelectionReason::kNotSelectedForActivation;
  }
  return false;
}

[[nodiscard]] inline SampleMetadata metadata_from_options(const EmitOptions& options,
                                                          std::size_t payload_size) noexcept {
  SampleMetadata metadata{};
  if (options.capture_time) {
    metadata.capture_timestamp_ns = options.capture_time->timestamp_ns;
    metadata.source_clock_id = options.capture_time->clock_id;
  }
  if (options.frame_id) {
    metadata.frame_id = *options.frame_id;
  }
  if (options.trace) {
    metadata.trace_id = options.trace->trace_id;
  }
  metadata.byte_size = payload_size;
  metadata.payload_kind = PayloadKind::kInlineTyped;
  metadata.flags = options.flags;
  return metadata;
}

/// What one successful Channel receive transition admitted.
enum class ChannelReceiveKind : std::uint8_t {
  kData,
  kEndOfStream,
};

/// Tag a completed receive transition with what it admitted.
[[nodiscard]] inline expected<ChannelReceiveKind, Error> received_as(
    expected<void, Error> completed, const ChannelReceiveKind kind) noexcept {
  if (!completed) {
    return make_unexpected(std::move(completed).error());
  }
  return kind;
}

/// A std::mutex that counts its acquisitions, and the lock() calls that found it held, once phase
/// counters are attached. try_lock() counts only a successful acquisition: a failed nonblocking or
/// bounded-retry probe is not counted as contention, so a site that probes, backs off, and later
/// calls lock() reports contention only if that lock() itself found the mutex held. Site-specific
/// counters record failed probes. Unattached, as in production, a lock costs one extra predictable
/// branch.
class CountedMutex {
 public:
  void lock() {
    if (counters_ == nullptr) {
      mutex_.lock();
      return;
    }
    counters_->increment(acquisitions_);
    if (!mutex_.try_lock()) {
      counters_->increment(contended_);
      mutex_.lock();
    }
  }

  [[nodiscard]] bool try_lock() noexcept {
    const bool locked = mutex_.try_lock();
    if (locked && counters_ != nullptr) {
      counters_->increment(acquisitions_);
    }
    return locked;
  }

  void unlock() noexcept { mutex_.unlock(); }

  // Call before any thread can lock this mutex.
  void attach(RuntimePhaseCounters* counters, const RuntimePhaseCounter acquisitions,
              const RuntimePhaseCounter contended) noexcept {
    counters_ = counters;
    acquisitions_ = acquisitions;
    contended_ = contended;
  }

 private:
  std::mutex mutex_;
  RuntimePhaseCounters* counters_{};
  RuntimePhaseCounter acquisitions_{};
  RuntimePhaseCounter contended_{};
};

/// Own all materialized Channel state for one run and arbitrate its hot-path protocols.
///
/// The declaration order is also destruction order in reverse: bindings and worker state retire
/// before the underlying Channel runtime. The type is source-private and shared only with the
/// source-tree test-access translation unit.
// Preserve reverse-order teardown of bindings and worker state before the Channel runtime.
// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
class RuntimeChannelSession::Impl {
 public:
  /// Own unpublished Channel construction state until one complete session can be committed.
  class MaterializationTransaction;

#if !defined(NDEBUG)
  /// Assert that one activation owner has only one mutator at a time.
  ///
  /// Different persistent partition owners may mutate their disjoint activation records in
  /// parallel. The shared barrier still rejects overlap with a whole-session control transaction.
  class OwnerMutationGuard {
   public:
    OwnerMutationGuard(std::shared_mutex& barrier, std::atomic_flag& active) noexcept
        : barrier_(barrier, std::try_to_lock), active_(&active) {
      assert(barrier_.owns_lock());
      const bool already_active = active_->test_and_set(std::memory_order_acquire);
      assert(!already_active);
    }

    OwnerMutationGuard(const OwnerMutationGuard&) = delete;
    OwnerMutationGuard& operator=(const OwnerMutationGuard&) = delete;

    ~OwnerMutationGuard() noexcept { active_->clear(std::memory_order_release); }

   private:
    std::shared_lock<std::shared_mutex> barrier_;
    std::atomic_flag* active_;
  };

  /// Assert that a reset or restart transaction does not overlap any activation-owner mutation.
  class ControlMutationGuard {
   public:
    explicit ControlMutationGuard(std::shared_mutex& barrier) noexcept
        : barrier_(barrier, std::try_to_lock) {
      assert(barrier_.owns_lock());
    }

    ControlMutationGuard(const ControlMutationGuard&) = delete;
    ControlMutationGuard& operator=(const ControlMutationGuard&) = delete;

    ~ControlMutationGuard() = default;

   private:
    std::unique_lock<std::shared_mutex> barrier_;
  };
#endif

  struct SubscriberState;
  class ReadinessBinding;

  /// Keep language-owned payload references alive until the enclosing channel lock has unwound.
  class DeferredSampleReleases {
   public:
    DeferredSampleReleases(const RuntimePortTypeOps& type_ops, std::size_t maximum_releases)
        : retain_(type_ops.retain_sample_payload_for_deferred_release) {
      if (retain_ != nullptr) {
        releases_.reserve(maximum_releases);
      }
    }

    void retain(const void* sample) noexcept {
      if (retain_ == nullptr) {
        return;
      }
      releases_.emplace_back();
      retain_(sample, &releases_.back());
    }

   private:
    void (*retain_)(const void*, std::shared_ptr<void>*) noexcept {};
    std::vector<std::shared_ptr<void>> releases_;
  };

  struct SampleSlot {
    SubscriberState* subscriber{};
    AlignedBlock sample;
    TemporalSampleRecord temporal{};
    std::uint64_t readiness_sequence{};
    std::uint64_t arrival_order{};
    EndpointClockHealthDisposition clock_health_disposition{
        EndpointClockHealthDisposition::kNotApplicable};
    std::uint64_t clock_health_generation{};
    bool occupied{};
    bool event_emitted{};
    bool witness_retired{};
    bool active_selected{};
    bool reuse_retained{};
    bool temporal_retained{};
    bool retirement_selected{};
    bool typed_ingress_reserved{};
    bool sample_storage_prepared{};
    bool direct_ingress_unread{};
    // A capture-absent staged payload can be a selector-visible missing-record arrival without
    // becoming the callback payload. It keeps direct-ingress ownership until that readiness
    // witness and its temporal-cache evidence both retire, preserving the independent ReuseLast
    // incumbent throughout the callback.
    bool staged_arrival_evidence{};
    // Set while this staged record is the one a freshness eviction has qualified but not yet
    // admitted, and cleared when it is promoted or the reservation is abandoned. Staging
    // drop-oldest skips it: the gate's argument is about this record's coordinate, so admitting a
    // different one makes the eviction unsound however sound the gate was.
    bool refill_reserved{};
    // Set on the oldest staged predecessor selected for the current step toward a separately
    // fenced newest refill target. Unlike refill_reserved this record still counts against the
    // authored staging depth; it is protected only across the selector/channel commit boundary.
    bool refill_prefix_reserved{};
    // Set while this record is what a released freshness claim was traded for, and cleared once
    // its readiness event is published or the slot is reset.
    //
    // Pre-witness replacement is declined while it is set. Without that, the record carries no
    // claim between promotion and publication, so a fast producer can take back the very sample
    // the freshness eviction gave up a retained record to obtain -- and the replacement's capture
    // is later than the pending witness, so it sorts behind and the activation selects nothing.
    // The exclusion is bounded: the next poll publishes the record and clears this, and a producer
    // that cannot replace stages into direct ingress rather than blocking.
    bool owed_to_released_claim{};
    // Readiness coordinate of a capture-less record, published by every witness for it however
    // late the poll runs. It is the record's publication coordinate on an input ordered by
    // publication, or else the reading a drop-oldest DrainAll trade qualified it with, which lies
    // at or before that trade's trigger. Invalid otherwise.
    TimePoint capture_absent_reference{};
  };

  static void reset_sample_slot(SampleSlot& slot, DeferredSampleReleases& releases) noexcept;

  // Keep clock, selection, and freshness state grouped with their storage and lifetime flags.
  // NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
  struct InputPortState {
    PortId port{};
    VertexId owner{};
    void* declaration{};
    const RuntimePortTypeOps* type_ops{};
    std::vector<SubscriberState*> subscribers;
    std::vector<SampleSlot*> active_slots;
    std::vector<SampleSlot*> retirement_slots;
    std::vector<SampleSlot*> tensor_access_slots;
    std::vector<TensorReadLease> tensor_read_leases;
    ReadClaimPool* claim_pool{};
    RuntimeInputBinding binding{};
    AlignedBlock predicate_view_storage;
    ClockId capture_clock{};
    ClockId reference_clock{};
    std::uint64_t conversion_id{};
    std::optional<DirectAffineClockEdge> capture_to_reference;
    RuntimePhaseCounters* phase_counters{};
    ReadinessBinding* readiness_binding{};
    // A producer publishes this equality-only dirty epoch after completing typed cache mutation.
    // Aliasing an observed value would require one complete uint64_t cycle between polls.
    std::atomic<std::uint64_t> readiness_revision{};
    bool capture_time_required{};
    // Capture-less arrivals are ordered by their publication coordinate, which is on this input's
    // reference clock, instead of by when the owner happens to poll them (side-only inputs).
    bool publication_ordered{};
    // Data arrivals cannot activate the owner, so they publish their readiness revision without
    // waking it; the owner's next trigger scan ingests them. End of stream still wakes it.
    bool side_only{};
    // The pointer spans subscriber caches; slot contents remain protected by
    // their own cache mutex. Addresses and subscriber identities are stable
    // until the joined session is destroyed.
    std::atomic<SampleSlot*> reuse_slot{};
    InputSelection active_selection{InputState::kAbsent, SelectionReason::kNeverSeen};
    std::uint64_t active_generation{1U};
    std::uint64_t view_generation{1U};
    bool active{};
    bool batch{};
    bool destructive{};
    bool reuse_last{};
    bool private_depth_one_reuse_slot{};
    // A DrainAll selection. Its drop-oldest subscribers provision one staging slot beyond the
    // authored depth, which startup enables once the owner proves tick-reconcilable.
    bool drain_all{};
    // Its owner reconciles it once per tick, so a trade waits for the subscriber mutex rather than
    // declining when the producer holds it.
    bool tick_reconciled{};
    // The owner's next tick, in reference-clock nanoseconds, published before it parks. Promotion
    // keeps an unwitnessed typed record at or before it rather than replacing it with a record
    // published after it, so the tick still finds its newest eligible sample.
    std::atomic<std::int64_t> tick_fence_ns{std::numeric_limits<std::int64_t>::max()};
    bool selection_prepared{};
    bool staged_now{};
    bool predicate_view_constructed{};
    SubscriberState* freshness_reservation_subscriber{};
    SampleSlot* freshness_reservation_retained{};
    SampleSlot* freshness_reservation_successor{};
    // The newest witness-qualified staged record remains fenced while predecessor records are
    // admitted one at a time in arrival order. The target reservation consumes the existing one
    // record of freshness headroom; predecessor reservations do not.
    SampleSlot* freshness_refill_target{};
    TimePoint freshness_refill_target_reference{};
    bool freshness_refill_prefix_committed{};
    bool freshness_refill_drain_cut{};
    std::uint64_t next_freshness_reservation_generation{1U};
    std::uint64_t freshness_reservation_generation{};
    bool batch_receive_nvtx_enabled{};
    const void* required_tensor_address{};
    std::vector<std::uint64_t> required_tensor_shape;

    ~InputPortState();

    // Fate of the most recent record promoted for a released freshness claim: 0 none, 1 published,
    // 2 protected from replacement before it could publish. "Cleared" alone cannot say which, and
    // they are opposite outcomes -- one completed the trade, the other observed its guard fire.
    std::atomic<std::uint8_t> last_refill_outcome{};

    [[nodiscard]] std::uint64_t publish_readiness_change() noexcept;

    [[nodiscard]] static InputSelection selection(const void* opaque,
                                                  std::uint64_t generation) noexcept;

    [[nodiscard]] static expected<void, Error> classify(const InputPortState& state,
                                                        std::uint64_t generation,
                                                        bool batch_accessor) noexcept;

    [[nodiscard]] static expected<RuntimeScalarReadBinding, Error> scalar(
        const void* opaque, std::uint64_t generation, bool acquire_claim) noexcept;

    [[nodiscard]] static expected<RuntimeBatchReadBinding, Error> batch_read(
        const void* opaque, std::uint64_t generation) noexcept;

    inline static const RuntimeInputBindingOps kBindingOps{
        .selection = &selection,
        .scalar = &scalar,
        .batch = &batch_read,
    };
  };

  // The readiness coordinate a newly stored record carries: its publication coordinate when it
  // has no capture time and its input is ordered by publication; invalid otherwise.
  [[nodiscard]] static TimePoint stored_capture_absent_reference(
      const InputPortState& input, const SampleMetadata& metadata) noexcept {
    const TimePoint published = metadata.publish_time();
    return input.publication_ordered && !metadata.capture_time().valid() && published.valid() &&
                   published.clock_id == input.reference_clock
               ? published
               : TimePoint{};
  }

  // The coordinate a tick fence compares: the publication coordinate of a capture-less record on
  // a publication-ordered input, or the capture time in the input's reference clock.
  [[nodiscard]] static std::optional<std::int64_t> tick_fence_coordinate(
      const InputPortState& input, const SampleSlot& slot) noexcept {
    if (slot.capture_absent_reference.valid()) {
      return slot.capture_absent_reference.clock_id == input.reference_clock
                 ? std::optional<std::int64_t>{slot.capture_absent_reference.timestamp_ns}
                 : std::nullopt;
    }
    const TimePoint capture = slot.temporal.capture_time;
    if (!capture.valid() || !input.capture_clock.valid() || !input.reference_clock.valid() ||
        capture.clock_id != input.capture_clock) {
      return std::nullopt;
    }
    auto converted = convert_identical_or_direct(
        capture,
        input.reference_clock,
        input.capture_to_reference ? &*input.capture_to_reference : nullptr);
    return converted ? std::optional<std::int64_t>{converted->timestamp_ns} : std::nullopt;
  }

  // Field order groups transport, retention, and lifecycle state by responsibility. Reordering
  // this hot private state solely to minimize padding would obscure those ownership boundaries.
  // NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
  struct SubscriberState {
    // Message handle for receive annotations.
    nvtxStringHandle_t receive_nvtx_message{};
    nvtxStringHandle_t source_identity_nvtx_string{};
    EdgeId edge{};
    PortId source{};
    PortId target{};
    VertexId source_owner{};
    VertexId owner{};
    // The plan can be released or rebuilt after session construction. Keep the service identity
    // with the subscriber because direct-continuation targets expose it for the lifetime of this
    // session.
    std::string service;
    QueuePolicy queue_policy{QueuePolicy::kBlocking};
    // Set, under mutex, once the producer owner starts its terminal output close. A non-blocking
    // producer can wait for this edge's capacity only then, so only then do retirements wake it.
    bool producer_awaits_capacity{};
    std::uint32_t readiness_capacity{};
    std::uint32_t drain_capacity{};
    std::uint32_t read_claim_capacity{};
    std::uint32_t typed_slot_capacity{};
    std::uint32_t direct_ingress_capacity{};
    std::uint32_t active_typed_slot_capacity{};
    std::uint32_t active_direct_ingress_capacity{};
    // One physical staging slot beyond the authored depth is provisioned for a direct-retained
    // DrainAll drop-oldest subscriber; extra_staging_survivor enables it, so staging keeps N + 1
    // survivors.
    bool tick_staging_reserve{};
    bool extra_staging_survivor{};
    std::optional<std::uint32_t> candidate_typed_slot_capacity;
    std::optional<std::uint32_t> candidate_direct_ingress_capacity;
    std::uint32_t health_dispatch_budget{};
    bool pre_witness_replacement_allowed{};
    bool direct_process_local{};
    // In pump_subscribers_: it receives through its transport endpoint, the only path that can set
    // transport_head_blocked. Set once by initialize_pump_indices().
    bool transport_inventory{};
    // A blocking direct-retained subscriber whose consumer runs on a different execution lane
    // than its producer. Publications wait for its typed capacity within the graph blocking budget.
    bool direct_cross_lane{};
    // A blocking transported subscriber whose consumer runs on a different execution lane than its
    // producer. A credited publication may wait for its transport capacity.
    bool credit_cross_lane{};
    bool cross_process_transport{};
    bool process_local_transport{};
    // A process-local HoloMQ Tensor-bearing subscriber, whose staged arrivals the compiler budgets
    // in its producer's pools. Cross-process and imported Tensors do not stage.
    bool tensor_ingress_staging{};
    // A recipient of an emit-admitted service: admitted inside its producer's emit(), retried on
    // retirement, and never scanned by the pump.
    bool emit_admitted{};
    bool externally_driven_import{};
    bool catalog_tensor_import{};
    InputPortState* input{};
    std::unique_ptr<channel_internal::InlinePayloadSubscriber> endpoint;
    std::optional<ChannelSubscriber<Tensor>> tensor_endpoint;
    std::unique_ptr<channel_internal::TensorSchemaPayloadSubscriber> tensor_schema_endpoint;
    // Bind the source-private staged-head interface once per endpoint generation. Ordinary
    // receives can then reject ReuseLast replacement with pointer/plan checks and no RTTI.
    channel_internal::StagedDataWaitSetSubscriber* staged_data_endpoint{};
    std::uint64_t tensor_max_byte_span{};
    MemoryKind tensor_memory_kind{MemoryKind::kUnknown};
    DeviceId tensor_device{};
    std::vector<TensorFieldServiceStorageContract> tensor_fields;
    std::vector<DescriptorTensorTopicFieldRecord> tensor_logical_fields;
    std::vector<PayloadDescriptor> descriptor_tensor_scratch;
    std::vector<Tensor> tensor_schema_scratch;
    // Occupied-slot replacement is prepared away from the incumbent so every fallible decode or
    // construction path can leave the prior payload and its transport attribution intact.
    AlignedBlock replacement_sample;
    bool replacement_sample_constructed{};
    bool replacement_sample_prepared{};
    AlignedBlock descriptor_replacement_sample;
    bool descriptor_replacement_sample_prepared{};
    // Producer Tensor field this subscriber takes when its edge narrows the schema payload to a
    // Tensor. Sibling subscribers on the same service still receive the schema root.
    std::optional<TensorProjectionRecord> tensor_projection;
    // The consumer is Tensor-typed; transported envelopes use the frozen producer schema codec.
    const RuntimePortTypeOps* projected_schema_ops{};
    std::optional<TensorRepresentation> pass_through_tensor_layout;
    std::optional<TensorContractRecord> pass_through_tensor_contract;
    std::optional<TensorBoundaryContractRecord> tensor_logical_requirements;
    std::uint64_t wait_handle{};
    std::uint64_t terminal_wait_handle{};
    AlignedBlock scratch;
    std::size_t scratch_size{};
    std::vector<std::byte> native_scratch;
    std::optional<PodIdentityRecord> pod_identity;
    std::optional<SchemaIdentityRecord> process_local_schema_identity;
    std::optional<SchemaIdentityRecord> catalog_schema_identity;
    std::vector<SampleSlot> slots;
    void* typed_ingress_observer_state{};
    void (*typed_ingress_observer)(void*) noexcept {};
    void* input_poll_contention_observer_state{};
    RuntimeChannelInputPollContentionObserver input_poll_contention_observer{};
    // Test seam: observe each failed try_lock inside one bounded contention retry on this input.
    void* input_contention_retry_attempt_observer_state{};
    RuntimeChannelInputPollContentionObserver input_contention_retry_attempt_observer{};
    bool fail_next_reuse_replacement_receive_for_test{};
    ErrorCode retryable_reuse_replacement_receive_error_for_test{ErrorCode::kSuccess};
    // Serialize the single transport consumer and its scratch independently from retained typed
    // state. First-use Tensor imports may map shared resources, so they must not hold the cache
    // mutex needed by readiness polling and activation retirement.
    mutable CountedMutex transport_mutex;
    mutable CountedMutex mutex;
    std::uint64_t next_arrival_sequence{1U};
    std::uint64_t producer_epoch{1U};
    std::uint64_t last_producer_sequence{};
    // The newest sequence that has left transport, typed or direct-ingress staged, published for
    // lock-free publication credit checks by the producer. Written under the cache mutex.
    std::atomic<std::uint64_t> admitted_sequence{};
    std::uint64_t last_direct_ingress_sequence{};
    std::uint64_t end_of_stream_arrival_order{};
    std::uint64_t end_of_stream_transport_sequence{};
    bool end_of_stream_received{};
    bool end_of_stream_event_emitted{};
    ClockId catalog_capture_clock{};
    ClockId plan_capture_clock{};
    PortId clock_health_endpoint{};
    EndpointClockHealthRuntimeCoordinator* clock_health_coordinator{};
    // Generation-fenced rebind state for externally driven imports. The catalog copy shares the
    // plan's immutable endpoint-catalog state so the same catalog record can be re-resolved for a
    // replacement attachment. The cold attachment consumes slot one; each successful rebind
    // monotonically consumes another plan-admitted slot. CHAN-087 deliberately never decrements or
    // reuses this no-wrap lifetime budget (unlike HSDK5-014's reusable Tensor physical slots).
    ChannelEndpointCatalog rebind_catalog{};
    std::string rebind_endpoint_name;
    std::uint32_t import_generation_capacity{};
    std::uint32_t import_generations_used{1U};
    bool import_restart_capable{};
    // Snapshot consumed source authority at cutover, independently of data received in this epoch.
    // EOS may follow an empty replacement epoch; data admission still uses only observed records.
    std::uint64_t cutover_sequence_floor{};
    // Concurrent-admission state, written only under this subscriber's transport_mutex: the
    // transport head last met full typed retention and has not been admitted since.
    std::atomic<bool> transport_head_blocked{};
    // Set, in either admission mode, while the staged transport head is blocked by full typed
    // retention: it hides the samples queued behind that head from the pump's wait. Set only
    // under transport_mutex. Null when no pump wait-set attachment can be masked, or when the
    // head can replace a retained sample instead of waiting for one to retire.
    std::shared_ptr<std::atomic<bool>> blocked_head_mask;
    // Set while the pump's last wait reported this subscriber's data attachment ready, which means
    // a sample is queued behind any head the endpoint has staged.
    std::atomic<bool> pump_reported_ready{};
    // When a plain receive first found the current staged head refused, or zero. Written only under
    // transport_mutex.
    std::int64_t blocked_since_ns{};

    ~SubscriberState();

    [[nodiscard]] std::span<std::byte> scratch_bytes() noexcept {
      return {static_cast<std::byte*>(scratch.data()), scratch_size};
    }

    [[nodiscard]] std::span<const std::byte> scratch_bytes() const noexcept {
      return {static_cast<const std::byte*>(scratch.data()), scratch_size};
    }

    void publish_admitted_sequence() noexcept {
      admitted_sequence.store(std::max(last_producer_sequence, last_direct_ingress_sequence),
                              std::memory_order_release);
    }

    [[nodiscard]] bool requires_pump_vmm_import_context() const noexcept {
      return tensor_endpoint.has_value() ||
             (tensor_schema_endpoint != nullptr && cross_process_transport);
    }

    [[nodiscard]] bool has_pending_direct_ingress() const noexcept;

    // Show the samples queued behind this subscriber's transport head to the pump's wait again, and
    // start a new spin at the next refusal.
    void unmask_blocked_head() noexcept {
      blocked_since_ns = 0;
      if (blocked_head_mask != nullptr) {
        blocked_head_mask->store(false, std::memory_order_release);
      }
    }

    [[nodiscard]] bool uses_device_affine_descriptor_import() const noexcept;

    [[nodiscard]] expected<std::uint64_t, Error> attach_wait_set(
        channel_internal::InlinePayloadWaitSet& wait_set) {
      const std::uint32_t endpoint_count =
          static_cast<std::uint32_t>(endpoint != nullptr) +
          static_cast<std::uint32_t>(tensor_endpoint.has_value()) +
          static_cast<std::uint32_t>(tensor_schema_endpoint != nullptr);
      if (direct_process_local || endpoint_count != 1U) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      if (endpoint != nullptr) {
        return wait_set.attach(*endpoint);
      }
      if (tensor_endpoint.has_value()) {
        return RuntimeChannelBindingAccess::attach_wait_set(wait_set, *tensor_endpoint);
      }
      return wait_set.attach(*tensor_schema_endpoint);
    }

    [[nodiscard]] std::optional<std::uint64_t> secondary_wait_set_handle() noexcept {
      if (endpoint != nullptr) {
        return channel_internal::protected_terminal_wait_handle(*endpoint);
      }
      if (tensor_endpoint.has_value()) {
        return RuntimeChannelBindingAccess::secondary_wait_set_handle(*tensor_endpoint);
      }
      if (tensor_schema_endpoint != nullptr) {
        return channel_internal::protected_terminal_wait_handle(*tensor_schema_endpoint);
      }
      return std::nullopt;
    }

    [[nodiscard]] std::optional<channel_internal::StagedDataState> staged_data_state() noexcept {
      return staged_data_endpoint != nullptr ? staged_data_endpoint->staged_data_state()
                                             : std::nullopt;
    }

    [[nodiscard]] std::optional<std::uint64_t> staged_terminal_sequence() noexcept {
      return staged_data_endpoint != nullptr ? staged_data_endpoint->staged_terminal_sequence()
                                             : std::nullopt;
    }

    [[nodiscard]] bool supports_staged_data_replacement() const noexcept {
      return staged_data_endpoint != nullptr;
    }

    [[nodiscard]] std::optional<channel_internal::StagedDataState>
    staged_data_before_reuse_replacement(const bool has_typed_reservation) noexcept {
      return !has_typed_reservation && input != nullptr && input->reuse_last &&
                     supports_staged_data_replacement()
                 ? staged_data_state()
                 : std::nullopt;
    }

    void refresh_staged_data_endpoint() noexcept {
      staged_data_endpoint = nullptr;
      if (endpoint != nullptr) {
        staged_data_endpoint =
            dynamic_cast<channel_internal::StagedDataWaitSetSubscriber*>(endpoint.get());
      } else if (tensor_endpoint.has_value()) {
        staged_data_endpoint =
            RuntimeChannelBindingAccess::staged_tensor_subscriber(*tensor_endpoint);
      } else if (tensor_schema_endpoint != nullptr) {
        staged_data_endpoint = dynamic_cast<channel_internal::StagedDataWaitSetSubscriber*>(
            tensor_schema_endpoint.get());
      }
    }

    [[nodiscard]] bool supports_cold_sample_replacement() noexcept {
      return queue_policy == QueuePolicy::kDropOldest ||
             (input != nullptr && input->reuse_last && supports_staged_data_replacement());
    }

    [[nodiscard]] expected<bool, Error> has_pending_transport() const noexcept;

    [[nodiscard]] expected<channel_internal::CatalogReceivedOperation, Error>
    receive_catalog_operation(const bool accept_data) noexcept {
      if (!externally_driven_import) {
        return make_unexpected(Error{ErrorCode::kNotSupported});
      }
      if (tensor_endpoint.has_value()) {
        return RuntimeChannelBindingAccess::receive_catalog_tensor_operation(*tensor_endpoint,
                                                                             accept_data);
      }
      if (tensor_schema_endpoint != nullptr) {
        return RuntimeChannelBindingAccess::receive_catalog_tensor_schema_operation(
            *tensor_schema_endpoint, accept_data);
      }
      if (endpoint == nullptr) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      return channel_internal::CatalogChannelBindingAccess::receive_next(
          *endpoint, scratch_bytes(), accept_data, true);
    }

    [[nodiscard]] expected<channel_internal::CatalogImportQuiescence, Error>
    catalog_import_quiescence() noexcept {
      if (tensor_endpoint.has_value()) {
        return RuntimeChannelBindingAccess::catalog_tensor_import_quiescence(*tensor_endpoint);
      }
      if (tensor_schema_endpoint != nullptr) {
        return RuntimeChannelBindingAccess::catalog_tensor_schema_import_quiescence(
            *tensor_schema_endpoint);
      }
      if (endpoint == nullptr) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      return channel_internal::CatalogChannelBindingAccess::import_quiescence(*endpoint);
    }

    [[nodiscard]] expected<std::uint64_t, Error> latched_catalog_producer_generation()
        const noexcept {
      if (tensor_endpoint.has_value()) {
        return RuntimeChannelBindingAccess::latched_catalog_tensor_producer_generation(
            *tensor_endpoint);
      }
      if (tensor_schema_endpoint != nullptr) {
        return RuntimeChannelBindingAccess::latched_catalog_tensor_schema_producer_generation(
            *tensor_schema_endpoint);
      }
      if (endpoint == nullptr) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      return channel_internal::CatalogChannelBindingAccess::latched_producer_generation(*endpoint);
    }

    [[nodiscard]] expected<void, Error> supersede_catalog_producer_generation(
        const std::uint64_t generation) noexcept {
      if (tensor_endpoint.has_value()) {
        return RuntimeChannelBindingAccess::supersede_catalog_tensor_producer_generation(
            *tensor_endpoint, generation);
      }
      if (tensor_schema_endpoint != nullptr) {
        return RuntimeChannelBindingAccess::supersede_catalog_tensor_schema_producer_generation(
            *tensor_schema_endpoint, generation);
      }
      if (endpoint == nullptr) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      return channel_internal::CatalogChannelBindingAccess::supersede_producer_generation(
          *endpoint, generation);
    }

    [[nodiscard]] expected<void, Error> detach_catalog_wait_set() noexcept {
      if (tensor_endpoint.has_value()) {
        return RuntimeChannelBindingAccess::detach_catalog_tensor_wait_set(*tensor_endpoint);
      }
      if (tensor_schema_endpoint != nullptr) {
        return RuntimeChannelBindingAccess::detach_catalog_tensor_schema_wait_set(
            *tensor_schema_endpoint);
      }
      if (endpoint == nullptr) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      return endpoint->detach_wait_set();
    }

    [[nodiscard]] bool catalog_has_publishers() const noexcept {
      if (tensor_endpoint.has_value()) {
        return tensor_endpoint->has_publishers();
      }
      if (tensor_schema_endpoint != nullptr) {
        return tensor_schema_endpoint->has_publishers();
      }
      return endpoint != nullptr && endpoint->has_publishers();
    }

    [[nodiscard]] expected<std::uint32_t, Error> discard_process_local_transport(
        CudaDeviceCursor* scan_device) noexcept;

    [[nodiscard]] expected<void, Error> discard_one_process_local_transport(
        CudaDeviceCursor* scan_device) noexcept;

    void clear_samples(std::uint64_t sequence_floor) noexcept;
  };

  // Whether a producer can be waiting for this direct edge's typed capacity: a blocking edge, or a
  // non-blocking one whose producer started its terminal close. Call with the subscriber mutex
  // held.
  [[nodiscard]] static bool producer_may_await_capacity(
      const SubscriberState& subscriber) noexcept {
    return subscriber.queue_policy == QueuePolicy::kBlocking || subscriber.producer_awaits_capacity;
  }

  // Serialize an input-wide retention transition against every incident
  // producer. The cold subscriber order is stable and disjoint between inputs.
  // Never acquire a transport mutex or re-enter a cache lock while holding this
  // allocation-free guard.
  class InputCacheLock {
   public:
    explicit InputCacheLock(const InputPortState& input) noexcept
        : subscribers_(input.subscribers) {
      for (auto* subscriber : subscribers_) {
        subscriber->mutex.lock();
      }
    }
    InputCacheLock(const InputCacheLock&) = delete;
    InputCacheLock& operator=(const InputCacheLock&) = delete;
    ~InputCacheLock() {
      for (auto it = subscribers_.rbegin(); it != subscribers_.rend(); ++it) {
        (*it)->mutex.unlock();
      }
    }

   private:
    std::span<SubscriberState* const> subscribers_;
  };

  // Own one exact typed destination across transport receive and first-use import work performed
  // without the cache mutex. Every non-commit exit restores the prior cache visibility. release()
  // acquires the cache mutex; never let an armed reservation die while holding that mutex.
  class TypedIngressReservation {
   public:
    TypedIngressReservation() = default;

    TypedIngressReservation(SubscriberState& subscriber, SampleSlot& destination, const bool staged,
                            const bool reuse_replacement = false,
                            const bool prior_temporal_retained = false,
                            const bool released_claim_replacement = false) noexcept
        : subscriber_(&subscriber),
          destination_(&destination),
          staged_(staged),
          reuse_replacement_(reuse_replacement),
          prior_temporal_retained_(prior_temporal_retained),
          released_claim_replacement_(released_claim_replacement),
          prior_arrival_order_(destination.arrival_order),
          prior_source_sequence_(destination.temporal.source_sequence) {}

    /// Whether the reserved slot belongs to the staged population rather than the typed one.
    ///
    /// This is a property of the slot, not of how the sample reaches it. Inferring it from the
    /// delivery mode is what previously made a staged transported sample count as typed.
    [[nodiscard]] bool staged() const noexcept { return staged_; }

    TypedIngressReservation(const TypedIngressReservation&) = delete;
    TypedIngressReservation& operator=(const TypedIngressReservation&) = delete;

    TypedIngressReservation(TypedIngressReservation&& other) noexcept
        : subscriber_(other.subscriber_),
          destination_(other.destination_),
          staged_(other.staged_),
          reuse_replacement_(other.reuse_replacement_),
          prior_temporal_retained_(other.prior_temporal_retained_),
          released_claim_replacement_(other.released_claim_replacement_),
          prior_arrival_order_(other.prior_arrival_order_),
          prior_source_sequence_(other.prior_source_sequence_) {
      other.subscriber_ = nullptr;
      other.destination_ = nullptr;
    }

    TypedIngressReservation& operator=(TypedIngressReservation&&) = delete;

    ~TypedIngressReservation() { release(); }

    // Borrow the protected destination while its cache mutex is held. The reservation remains
    // armed until the complete store succeeds, so a fallible decode or construction can restore
    // the exact incumbent.
    [[nodiscard]] SampleSlot* destination_locked() noexcept {
      assert(subscriber_ != nullptr);
      assert(destination_ != nullptr);
      assert(destination_->typed_ingress_reserved);
      return destination_;
    }

    // Commit the protected destination only after the complete store has succeeded.
    void commit_locked() noexcept {
      assert(subscriber_ != nullptr);
      assert(destination_ != nullptr);
      assert(destination_->typed_ingress_reserved);
      if (released_claim_replacement_) {
        // The released ReuseLast claim commits this exact transported successor to readiness.
        // Publish the guard before dropping the typed-ingress reservation so no producer can
        // observe an unclaimed, pre-witness replacement window.
        destination_->owed_to_released_claim = true;
        subscriber_->input->last_refill_outcome.store(0U, std::memory_order_relaxed);
      }
      destination_->typed_ingress_reserved = false;
      subscriber_ = nullptr;
      destination_ = nullptr;
    }

    void release() noexcept {
      if (subscriber_ == nullptr) {
        return;
      }
      DeferredSampleReleases releases{*subscriber_->input->type_ops, 1U};
      const std::scoped_lock cache_lock{subscriber_->mutex};
      assert(destination_ != nullptr);
      assert(destination_->typed_ingress_reserved);
      destination_->typed_ingress_reserved = false;
      if (reuse_replacement_) {
        const bool same_slot_identity =
            destination_->occupied && destination_->subscriber == subscriber_ &&
            destination_->arrival_order == prior_arrival_order_ &&
            destination_->temporal.source_sequence == prior_source_sequence_;
        SampleSlot* empty = nullptr;
        if (same_slot_identity && !destination_->reuse_retained &&
            destination_->temporal_retained == prior_temporal_retained_ &&
            subscriber_->input->reuse_slot.compare_exchange_strong(
                empty, destination_, std::memory_order_acq_rel, std::memory_order_acquire)) {
          destination_->reuse_retained = true;
          destination_->temporal_retained = prior_temporal_retained_;
        } else {
          // The reservation fence makes identity drift unreachable through production cache
          // mutation. If another owner nevertheless installed a claim/pointer while transport was
          // outside the mutex, preserve that incumbent and reclaim only this now-unclaimed slot.
          assert(same_slot_identity);
          reclaim_sample_slot_if_unclaimed(*destination_, releases);
        }
      }
      subscriber_ = nullptr;
      destination_ = nullptr;
    }

   private:
    SubscriberState* subscriber_{};
    SampleSlot* destination_{};
    bool staged_{};
    bool reuse_replacement_{};
    bool prior_temporal_retained_{};
    bool released_claim_replacement_{};
    std::uint64_t prior_arrival_order_{};
    std::uint64_t prior_source_sequence_{};
  };

  /// Prepare the persistent descriptor sample used during restart attachment replacement.
  [[nodiscard]] static expected<void, Error> prepare_descriptor_replacement_sample(
      SubscriberState& subscriber) noexcept {
    const RuntimePortTypeOps& ops = *subscriber.input->type_ops;
    if (ops.codec_family != RuntimePayloadCodecFamily::kDescriptorTensorSchema) {
      return {};
    }
    if (!ops.persistent_sample_storage || ops.prepare_sample_storage == nullptr ||
        ops.reset_sample_storage == nullptr || ops.destroy_sample_storage == nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    auto storage = AlignedBlock::create(ops.sample_size, ops.sample_alignment);
    if (!storage) {
      return make_unexpected(std::move(storage).error());
    }
    auto prepared = ops.prepare_sample_storage(storage->data());
    if (!prepared) {
      return make_unexpected(std::move(prepared).error());
    }
    subscriber.descriptor_replacement_sample = std::move(*storage);
    subscriber.descriptor_replacement_sample_prepared = true;
    return {};
  }

  /// Reclaim storage only after transport attribution and every runtime payload claim have ended.
  ///
  /// The caller must hold the owning subscriber mutex. A readiness event transfers a temporal
  /// claim to the selector before the witness can retire, so witness retirement alone is never
  /// sufficient to recycle a slot retained by a cache, pending window, closed window, or reuse
  /// policy.
  static void reclaim_sample_slot_if_unclaimed(SampleSlot& slot,
                                               DeferredSampleReleases& releases) noexcept {
    if (slot.occupied && !slot.temporal_retained && !slot.retirement_selected &&
        !slot.typed_ingress_reserved && !slot.active_selected && !slot.reuse_retained &&
        !slot.refill_reserved && !slot.refill_prefix_reserved && !slot.owed_to_released_claim &&
        (!slot.event_emitted || slot.witness_retired)) {
      reset_sample_slot(slot, releases);
    }
  }

  // Called under the subscriber mutex. Producer traffic cannot replace an unread member of the
  // finite drain cut while its owner is waiting for admission credit or an earlier handoff.
  [[nodiscard]] static bool drain_cut_retains(const SampleSlot& slot) noexcept;

  struct OutputState;

  struct LoanControl {
    OutputState* owner{};
    std::atomic<std::uint32_t>* live_credit{};
    const RuntimePortTypeOps* type_ops{};
    AlignedBlock value;
    RuntimeLoanBinding binding{};
    std::atomic<std::uint32_t> references{1U};
    std::uint64_t activation_generation{};
    bool live{};

    void release_reference() noexcept;

    void reset_value() noexcept;

    void retire_live_authority() noexcept;

    static void cancel(void* opaque) noexcept;
  };

  class ProcessLocalPublicationScope {
   public:
    // A process-local scope also marks `owner` as having transported since its last cut, even when
    // emit-admitted: delivery inside emit() may still leave a blocked or pending head. With no
    // activation owner (a terminal close, for example), no owner cut will settle the publication,
    // so an accepted or indeterminate one falls back to the next cut of any owner.
    ProcessLocalPublicationScope(Impl& session, bool process_local, VertexId owner,
                                 bool emit_admitted = false) noexcept;

    ProcessLocalPublicationScope(const ProcessLocalPublicationScope&) = delete;
    ProcessLocalPublicationScope& operator=(const ProcessLocalPublicationScope&) = delete;

    ~ProcessLocalPublicationScope() noexcept;

    void accept(std::uint32_t deliveries,
                std::span<SubscriberState* const> activation_cut_subscribers = {},
                std::span<SubscriberState* const> emit_recipients = {},
                std::uint64_t publication_sequence = 0U) noexcept;

    void reject() noexcept;

    void invalidate() noexcept;

   private:
    void finish() noexcept;

    Impl* session_{};
    bool in_flight_{};
    // The endpoint's service is delivered inside emit(): accept() admits its recipients and an
    // uncertain send drains the whole inventory before emit() returns.
    bool emit_admitted_{};
    bool unowned_{};
  };

  /// @brief Return the clock this session stamps publication coordinates from, or zero.
  [[nodiscard]] ClockId publish_clock_id() const noexcept { return publish_clock_.id(); }

  /// @brief Stamp one sample's publication coordinate at the Runtime submission boundary.
  ///
  /// The coordinate is read once per publication rather than once per endpoint, so every fanout
  /// copy of one sample shares a single publication instant. That is what lets a consumer compare
  /// two recipients' latency against the same event instead of against two unrelated clock reads.
  ///
  /// A session with no bound publish clock leaves the canonical absent coordinate in place.
  ///
  /// @param metadata Metadata record stamped in place.
  /// @return Success, or the clock-read failure that aborts this publication before submission.
  [[nodiscard]] expected<void, Error> stamp_publication_coordinate(
      SampleMetadata& metadata) noexcept {
    if (!publish_clock_.id().valid()) {
      metadata.publish_timestamp_ns = 0;
      metadata.publish_clock_id = {};
      return {};
    }
    auto published_at = publish_clock_.now();
    if (!published_at) {
      HOLOSCAN_LOG_AUTOMATIC_ERROR("Channel publication could not read publish clock {}",
                                   publish_clock_.id().value);
      return make_unexpected(std::move(published_at).error());
    }
    metadata.publish_timestamp_ns = published_at->timestamp_ns;
    metadata.publish_clock_id = published_at->clock_id;
    return {};
  }

  /// @brief Translate one outbound sample's publication coordinate into the catalog namespace.
  ///
  /// The coordinate Runtime stamped names a plan clock, which no external client can resolve. This
  /// rewrites the identity to the catalog record for the same clock key and leaves the coordinate
  /// itself untouched, because both namespaces describe the same instant on the same clock.
  ///
  /// When this plan projects no publication domain the pair is cleared together. Exporting a
  /// coordinate whose clock a client cannot resolve would be worse than absence: the client would
  /// read a plausible number with no way to discover it means nothing to it.
  ///
  /// @param metadata Endpoint-bound metadata copy rewritten in place.
  void project_publication_coordinate(SampleMetadata& metadata) const noexcept {
    if (!catalog_publish_clock_.valid() || !metadata.publish_clock_id.valid()) {
      metadata.publish_clock_id = {};
      metadata.publish_timestamp_ns = 0;
      return;
    }
    metadata.publish_clock_id = catalog_publish_clock_;
  }

  /// @brief Return whether one received sample's publication coordinate is admissible here.
  ///
  /// A producer materialized from this plan and its consumer share one plan clock catalog, so the
  /// sample must carry exactly the clock declaration this session stamps from. Merely requiring
  /// internal coherence would admit a dropped stamp as canonical absence, which is
  /// indistinguishable to a consumer from a producer that legitimately has no publish clock. An
  /// external import carries no publication coordinate because the exporting catalog owns a
  /// separate clock namespace.
  ///
  /// Naming the same declaration is not the same as sharing a live clock. A cross-process producer
  /// satisfies this rule and is still attributed to a different generation, so admission is paired
  /// with \ref withdraw_foreign_publication_coordinate at the retention boundary.
  ///
  /// @param metadata Received metadata under validation.
  /// @param externally_imported Whether the sample crossed an external endpoint boundary.
  /// @return \c true when the coordinate matches what this session admits for that origin.
  [[nodiscard]] bool publication_coordinate_admitted(const SampleMetadata& metadata,
                                                     bool externally_imported) const noexcept {
    if (externally_imported || !publish_clock_.id().valid()) {
      return !metadata.publish_clock_id.valid() && metadata.publish_timestamp_ns == 0;
    }
    return metadata.publish_clock_id == publish_clock_.id();
  }

  /// @brief Withdraw a retained coordinate that was measured in another clock generation.
  ///
  /// A cross-process peer reconstructs this same plan in its own worker, so it resolves the plan's
  /// root default clock declaration to its own live instance. Both workers number that declaration
  /// identically, which is why the arriving identity passes
  /// \ref publication_coordinate_admitted, but the coordinate was measured against an origin this
  /// worker cannot place: the ordinary default clock counts from its own construction. Retaining
  /// the pair would let a consumer subtract two unrelated origins and read the difference as
  /// transit latency, including a negative one when the peer's worker started first.
  ///
  /// Canonical absence is the honest answer for that hop, and it is the same answer an external
  /// import already receives for the same underlying reason.
  ///
  /// @param subscriber Borrowed receiving endpoint.
  /// @param metadata Retained metadata copy cleared in place when the hop crossed processes.
  static void withdraw_foreign_publication_coordinate(const SubscriberState& subscriber,
                                                      SampleMetadata& metadata) noexcept {
    if (!subscriber.cross_process_transport) {
      return;
    }
    metadata.publish_clock_id = {};
    metadata.publish_timestamp_ns = 0;
  }

  struct OutputState {
    struct EndpointState {
      std::unique_ptr<channel_internal::InlinePayloadPublisher> publisher;
      std::string process_local_service;
      std::vector<SubscriberState*> activation_cut_subscribers;
      // Every handle-less subscriber of this service, of either queue policy, admitted inside
      // emit() when the service is emit-admitted. activation_cut_subscribers stays blocking-only
      // for the single-runner indexed cut.
      std::vector<SubscriberState*> emit_recipients;
      // Publication credits: blocking transported recipients, and the last sequence submitted to
      // them. Only the producer writes it.
      std::vector<SubscriberState*> credit_subscribers;
      std::uint64_t last_sent_sequence{};
      ClockId plan_capture_clock{};
      ClockId catalog_capture_clock{};
      std::uint32_t expected_recipient_count{1U};
      bool refresh_before_emit{};
      bool peer_required{true};
      bool process_local_transport{};
      bool emit_admitted{};
      bool catalog_encodes_custom_pod{};
      bool process_local_schema_envelope{};
      bool terminal_obligated{};
      bool terminal_resolved{};
      channel_internal::CatalogCanonicalPodPublisherBinding canonical_pod_publisher;
    };

    // Emit annotation for Operator Centric Logging.
    nvtxStringHandle_t emit_nvtx_message{};
    nvtxStringHandle_t source_identity_nvtx_string{};
    PortId port{};
    VertexId owner{};
    Impl* session{};
    TensorContractMismatchCapture* tensor_failure{};
    SubscriberState* direct_subscriber{};
    // Every direct-retained recipient of a drop-oldest fanout family, cross-lane recipients first;
    // empty for fanout one. `direct_subscriber` is the first recipient either way. A mixed-locality
    // family also keeps its cross-process and exported legs in `endpoints`.
    std::vector<SubscriberState*> direct_fanout;
    // Whether any direct fanout recipient is blocking, so emit() checks capacity first.
    bool direct_fanout_blocking{};

    // Every direct-retained recipient: the fanout family, the single fanout-one recipient, or none.
    [[nodiscard]] std::span<SubscriberState* const> direct_recipients() const noexcept {
      if (!direct_fanout.empty()) {
        return direct_fanout;
      }
      return {&direct_subscriber, direct_subscriber != nullptr ? 1U : 0U};
    }
    // Whether the output publishes through any endpoint, beside any direct recipients: a
    // mixed-locality family's transported legs.
    [[nodiscard]] bool has_endpoints() const noexcept {
      return !endpoints.empty() || !tensor_endpoints.empty() || !catalog_tensor_endpoints.empty() ||
             !tensor_schema_endpoints.empty();
    }
    void* declaration{};
    const RuntimePortTypeOps* type_ops{};
    std::vector<EndpointState> endpoints;
    struct TensorEndpointState {
      ManualTensorPublisher publisher;
      std::string process_local_service;
      std::vector<SubscriberState*> activation_cut_subscribers;
      std::vector<SubscriberState*> emit_recipients;
      std::vector<SubscriberState*> credit_subscribers;
      std::uint64_t last_sent_sequence{};
      std::uint32_t expected_recipient_count{1U};
      bool process_local_transport{};
      bool emit_admitted{};
      bool terminal_resolved{};
    };
    std::vector<TensorEndpointState> tensor_endpoints;
    struct CatalogTensorEndpointState {
      ChannelPublisher<Tensor> publisher;
      cudaStream_t staging_stream{};
      std::shared_ptr<const holomq::cuda::StreamLifetime> staging_lifetime;
      ClockId plan_capture_clock{};
      ClockId catalog_capture_clock{};
      bool peer_required{true};
      bool active_for_current_fanout{};
      bool terminal_obligated{};
      bool terminal_resolved{};
    };
    std::vector<CatalogTensorEndpointState> catalog_tensor_endpoints;
    struct TensorSchemaEndpointState {
      std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher> publisher;
      std::string process_local_service;
      std::vector<SubscriberState*> activation_cut_subscribers;
      std::vector<SubscriberState*> emit_recipients;
      std::vector<SubscriberState*> credit_subscribers;
      std::uint64_t last_sent_sequence{};
      ClockId plan_capture_clock{};
      ClockId catalog_capture_clock{};
      std::uint32_t expected_recipient_count{1U};
      bool refresh_before_emit{};
      bool peer_required{true};
      bool process_local_transport{};
      bool emit_admitted{};
      bool terminal_obligated{};
      bool terminal_resolved{};
      // A mixed-locality publication's reservation and metadata for this leg, held from before
      // the direct stores until the leg publishes.
      std::optional<channel_internal::TensorSchemaPayloadReservation> pending_reservation;
      SampleMetadata pending_metadata{};
    };
    std::vector<TensorSchemaEndpointState> tensor_schema_endpoints;
    SchemaId tensor_schema_id{};
    std::uint64_t tensor_max_byte_span{};
    MemoryKind tensor_memory_kind{MemoryKind::kUnknown};
    DeviceId tensor_device{};
    std::optional<TensorRepresentation> tensor_layout;
    std::optional<TensorContractRecord> tensor_contract;
    std::vector<TensorFieldServiceStorageContract> tensor_fields;
    std::vector<DescriptorTensorTopicFieldRecord> tensor_logical_fields;
    std::vector<Tensor> tensor_schema_scratch;
    std::optional<TensorBoundaryContractRecord> catalog_tensor_requirements;
    RuntimeOutputBinding binding{};
    AlignedBlock encode_scratch;
    AlignedBlock process_local_schema_scratch;
    std::optional<PodIdentityRecord> pod_identity;
    std::optional<SchemaIdentityRecord> process_local_schema_identity;
    std::optional<channel_internal::CatalogSchemaFrameCorruption>
        next_process_local_schema_corruption;
    RuntimePhaseCounters* phase_counters{};
    RuntimeTensorOutputTestControl* tensor_test_control{};
    std::optional<std::size_t> throw_tensor_fanout_after_accepts_for_test;
    std::vector<LoanControl*> loans;
    // S2: the error that the latest publication on this output returned after the accepted stop
    // cancelled its wait, and that publication's activation generation (zero when none has).
    // Written only by the owner's activation, like next_sequence.
    Error stop_cancelled_error{};
    std::uint64_t next_sequence{1U};
    std::uint64_t active_generation{1U};
    std::uint64_t stop_cancelled_generation{};
    std::uint32_t max_emits_per_activation{};
    std::uint32_t emitted_this_activation{};
    std::uint32_t max_live_loans{};
    std::atomic<std::uint32_t> live_loan_count{};
    // The recipient a direct continuation polls: the fanout-one recipient, or the one fanout
    // recipient a causal chain continues into. Null when this output never continues.
    SubscriberState* direct_continuation_subscriber{};
    bool direct_continuation_capture{};
    bool active{};
    // Remember authority consumed by an earlier partial close until every arm has resolved.
    // Local no-peer resolution and recipient abandonment consume no terminal coordinate.
    enum class TerminalCloseState : std::uint8_t { kPending, kAuthorityConsumed, kComplete };
    TerminalCloseState terminal_close_state{TerminalCloseState::kPending};

    // Resolve each transport family in order, retaining consumed terminal authority across retries.
    [[nodiscard]] expected<void, Error> close_terminal() noexcept;
    [[nodiscard]] expected<void, Error> close_inline_endpoints() noexcept;
    [[nodiscard]] expected<void, Error> close_tensor_endpoints() noexcept;
    [[nodiscard]] expected<void, Error> close_catalog_tensor_endpoints() noexcept;
    [[nodiscard]] expected<void, Error> close_tensor_schema_endpoints() noexcept;
    void resolve_terminal_publication(bool& terminal_resolved) noexcept;
    [[nodiscard]] expected<void, Error> reject_terminal_publication(
        ProcessLocalPublicationScope& publication, Error error, bool& terminal_resolved);

    [[nodiscard]] bool terminal_close_started() const noexcept {
      return terminal_close_state != TerminalCloseState::kPending ||
             std::ranges::any_of(endpoints, &EndpointState::terminal_resolved) ||
             std::ranges::any_of(tensor_endpoints, &TensorEndpointState::terminal_resolved) ||
             std::ranges::any_of(catalog_tensor_endpoints,
                                 &CatalogTensorEndpointState::terminal_resolved) ||
             std::ranges::any_of(tensor_schema_endpoints,
                                 &TensorSchemaEndpointState::terminal_resolved);
    }

    void reset_terminal_close_state() noexcept {
      for (auto& endpoint : endpoints) {
        endpoint.terminal_obligated = endpoint.peer_required;
        endpoint.terminal_resolved = false;
      }
      for (auto& endpoint : tensor_endpoints) {
        endpoint.terminal_resolved = false;
      }
      for (auto& endpoint : catalog_tensor_endpoints) {
        endpoint.active_for_current_fanout = false;
        endpoint.terminal_obligated = endpoint.peer_required;
        endpoint.terminal_resolved = false;
      }
      for (auto& endpoint : tensor_schema_endpoints) {
        endpoint.terminal_obligated = endpoint.peer_required;
        endpoint.terminal_resolved = false;
      }
      terminal_close_state = TerminalCloseState::kPending;
    }

    [[nodiscard]] bool tensor_payload() const noexcept;

    [[nodiscard]] bool tensor_schema_payload() const noexcept;

    void mark_emit_nvtx(std::uint64_t sequence) const noexcept;

    void record_emit() noexcept;

    ~OutputState();

    [[nodiscard]] expected<void, Error> classify_connection_refresh(expected<void, Error> refreshed,
                                                                    bool submitted) noexcept;

    template <typename RetainOperation>
    [[nodiscard]] static expected<void, Error> commit_direct_publication(
        OutputState& state, SubscriberState& subscriber, std::uint64_t publication_sequence,
        RetainOperation&& retain) noexcept;

    // Store one publication into every direct recipient without recording the emit. The sequence
    // and coordinate are already stamped once; a failure at the first recipient is pre-submit, and
    // a later one records the emit, fails the session, and reports a partial publication.
    template <typename RetainOperation>
    [[nodiscard]] static expected<void, Error> store_direct_recipients(
        OutputState& state, std::uint64_t publication_sequence, RetainOperation& retain) noexcept;

    // Store one publication into every recipient of a direct fanout family and record the emit.
    template <typename RetainOperation>
    [[nodiscard]] static expected<void, Error> commit_direct_fanout_publication(
        OutputState& state, std::uint64_t publication_sequence, RetainOperation& retain) noexcept;

    [[nodiscard]] expected<void, Error> refresh_connections() noexcept {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        for (auto& endpoint : endpoints) {
          if (endpoint.terminal_resolved) {
            continue;
          }
          auto refreshed = endpoint.publisher->update_connections();
          if (!refreshed) {
            return make_unexpected(std::move(refreshed).error());
          }
        }
        for (auto& endpoint : tensor_endpoints) {
          if (endpoint.terminal_resolved) {
            continue;
          }
          auto refreshed = endpoint.publisher.update_connections();
          if (!refreshed) {
            return make_unexpected(std::move(refreshed).error());
          }
        }
        for (auto& endpoint : catalog_tensor_endpoints) {
          if (endpoint.terminal_resolved) {
            continue;
          }
          auto refreshed = endpoint.publisher.update_connections();
          if (!refreshed) {
            return make_unexpected(std::move(refreshed).error());
          }
        }
        for (auto& endpoint : tensor_schema_endpoints) {
          if (endpoint.terminal_resolved) {
            continue;
          }
          auto refreshed = endpoint.publisher->update_connections();
          if (!refreshed) {
            return make_unexpected(std::move(refreshed).error());
          }
        }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (...) {
        // Refresh precedes terminal submission, so retain every endpoint's retry authority and
        // convert the backend exception into a checked pre-submit failure.
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
      return {};
    }

    // Publication distinguishes every pre-submit, accepted, and indeterminate fanout phase in
    // branch order so partial publication can never be retried as an ordinary failure.
    [[nodiscard]] expected<void, Error> emit_payload(
        std::span<const std::byte> payload, SampleMetadata metadata,

        const channel_internal::InlinePayloadEncoder* encoder = nullptr) noexcept;

    [[nodiscard]] static expected<void, Error> validate_emit_authority(
        const OutputState& state) noexcept;

    [[nodiscard]] static expected<void, Error> classify_tensor_publication(
        OutputState& state, expected<std::uint32_t, Error> published, bool& submitted) noexcept;

    [[nodiscard]] static expected<void, Error> classify_tensor_failure(OutputState& state,
                                                                       Error error,
                                                                       bool submitted) noexcept;

    static void inject_tensor_fanout_failure(OutputState& state, std::size_t accepted_count);

    [[nodiscard]] static expected<void, Error> validate_catalog_tensor_metadata(
        const OutputState& state, const SampleMetadata& metadata) noexcept;

    [[nodiscard]] static expected<void, Error> emit_internal_tensor_endpoints(
        OutputState& state, Sample<Tensor>& normalized, bool& submitted);

    struct CatalogTensorFanoutPreparation {
      bool copy_required{};
      std::size_t staging_index{std::numeric_limits<std::size_t>::max()};
      std::optional<channel_internal::StagedTensorCopy> staged_copy;
    };

    [[nodiscard]] static expected<CatalogTensorFanoutPreparation, Error>
    prepare_catalog_tensor_fanout(OutputState& state, const Tensor& supplied);

    [[nodiscard]] static expected<void, Error> emit_catalog_tensor_endpoint(
        OutputState& state, CatalogTensorEndpointState& endpoint, const Tensor& supplied,
        const SampleMetadata& normalized_metadata, CatalogTensorFanoutPreparation& preparation,

        bool& submitted);

    [[nodiscard]] static expected<void, Error> contain_tensor_fanout_exception(
        OutputState& state, bool submitted) noexcept;

    // Run every transported leg's preparation that delivers nothing: the catalog clock
    // projections, then each catalog leg's refresh, classification, and interface-pool copy.
    [[nodiscard]] static expected<CatalogTensorFanoutPreparation, Error> prepare_tensor_transport(
        OutputState& state, const Sample<Tensor>& supplied,
        const SampleMetadata& metadata) noexcept;

    // Publish to every transported leg. `submitted` is whether direct recipients already hold the
    // publication, which makes a later rejection indeterminate.
    [[nodiscard]] static expected<void, Error> publish_tensor_transport(
        OutputState& state, const Sample<Tensor>& supplied, Sample<Tensor> normalized,
        CatalogTensorFanoutPreparation& preparation, bool submitted) noexcept;

    [[nodiscard]] static expected<void, Error> emit_tensor_fanout(
        OutputState& state, const Sample<Tensor>& supplied, Sample<Tensor> normalized) noexcept;

    [[nodiscard]] static expected<void, Error> emit_mixed_locality_tensor(
        OutputState& state, const Sample<Tensor>& supplied, Sample<Tensor> normalized) noexcept;

    void capture_tensor_rejection(const Tensor& tensor, const TensorAccessSnapshot& snapshot,
                                  bool native_contract_matches,
                                  bool port_contract_matches) noexcept;

    [[nodiscard]] static expected<void, Error> emit_tensor_sample(
        OutputState& state, const Sample<Tensor>& supplied) noexcept;

    [[nodiscard]] static expected<void, Error> emit_tensor_schema_fanout(
        OutputState& state, const void* payload, std::span<const Tensor> tensors,
        std::span<std::byte> destination, const SampleMetadata& metadata,
        bool require_direct_export) noexcept;

    struct TensorSchemaEncodeContext {
      OutputState* output{};
      const void* payload{};
      std::span<const Tensor> tensors{};
      SampleMetadata metadata{};
      std::uint64_t endpoint_generation{};
      bool separate_tensor{};
    };

    [[nodiscard]] static expected<std::size_t, Error> encode_tensor_schema_transport(
        void* opaque, std::span<std::byte> destination,
        const channel_internal::TensorSchemaPayloadReservation& reservation) noexcept;

    [[nodiscard]] static expected<void, Error> emit_tensor_schema_direct_loan(
        OutputState& state, const void* payload, std::span<const Tensor> tensors,
        SampleMetadata metadata, bool require_direct_export) noexcept;

    // Schema emission is one acceptance transaction across direct and endpoint fanout paths.
    [[nodiscard]] static expected<void, Error> emit_mixed_locality_tensor_schema(
        OutputState& state, const void* payload, std::span<const Tensor> tensors,
        std::span<std::byte> destination, std::size_t direct_encoded_size,
        const SampleMetadata& metadata, bool separate_tensor) noexcept;

    [[nodiscard]] static expected<void, Error> emit_tensor_schema_value(
        OutputState& state, const void* payload, std::span<const Tensor> tensors,
        const SampleMetadata& metadata, bool separate_tensor) noexcept;

    [[nodiscard]] static expected<void, Error> emit_tensor_schema_sample(
        OutputState& state, const void* payload, const SampleMetadata& metadata) noexcept;

    struct GeneratedSchemaEncodeContext {
      OutputState* output{};
      const void* payload{};
      std::uint64_t sequence{};
    };

    [[nodiscard]] static expected<std::size_t, Error> encode_generated_schema(
        void* opaque, std::span<std::byte> destination) noexcept;

    [[nodiscard]] static expected<void, Error> emit_generated_schema_payload(
        OutputState& state, const void* payload, const SampleMetadata& metadata,
        std::uint64_t emitted_sequence) noexcept;

    [[nodiscard]] static expected<void, Error> emit_process_local_object(
        OutputState& state, const void* payload, SampleMetadata metadata) noexcept;

    [[nodiscard]] static expected<void, Error> emit_non_tensor_sample(
        OutputState& state, const void* payload, const SampleMetadata& metadata) noexcept;

    [[nodiscard]] static expected<void, Error> emit_sample(void* opaque,
                                                           const void* sample) noexcept;

    [[nodiscard]] static expected<void, Error> observed_emit_sample(void* opaque,
                                                                    const void* sample) noexcept;
    [[nodiscard]] static expected<void, Error> observed_emit_loan(void* opaque,
                                                                  RuntimeLoanBinding* binding,
                                                                  EmitOptions options) noexcept;
    [[nodiscard]] static expected<void, Error> emit_ref(void* opaque, const void* payload,
                                                        EmitOptions options) noexcept;

    [[nodiscard]] static expected<void, Error> emit_descriptor_tensor(void* opaque,
                                                                      const void* descriptor,
                                                                      const Tensor* tensor,
                                                                      EmitOptions options) noexcept;

    [[nodiscard]] static expected<void, Error> emit_descriptor_tensors(
        void* opaque, const void* descriptor, std::span<const Tensor> tensors,
        EmitOptions options) noexcept;

    [[nodiscard]] static expected<RuntimeLoanAllocation, Error> allocate(void* opaque) noexcept;

    [[nodiscard]] static expected<void, Error> emit_loan(void* opaque, RuntimeLoanBinding* binding,
                                                         EmitOptions options) noexcept;

    /// S2: when the accepted stop cancelled this failed publication's wait, mark `error` as
    /// stopped and record it. Takes the per-thread cancellation record, so call it after every
    /// failed publication.
    void note_publication_failure(Error& error) noexcept;

    static const RuntimeOutputBindingOps kBindingOps;
    static const RuntimeOutputBindingOps kObservedBindingOps;
  };

  struct ActivationOwnerState {
    struct DirectContinuationPublication {
      OutputState* output{};
      SubscriberState* subscriber{};
      std::uint64_t publication_sequence{};
      std::uint64_t producer_epoch{};
      std::uint64_t prior_destination_revision{};
      std::uint64_t destination_revision{};
      std::uint64_t source_activation_generation{};
      std::uint32_t successor_publications{};
    };

    std::vector<InputPortState*> inputs;
    std::vector<SubscriberState*> subscribers;
    std::vector<OutputState*> outputs;
    // Cold, session-local topology decision populated after the destination readiness binding
    // exists. A null pointer is the permanent dormant state for this RunSession; live generation,
    // revision, closure, and capacity predicates remain checked when a continuation is armed.
    OutputState* direct_continuation_output{};
    std::optional<DirectContinuationPublication> direct_continuation;
    // The target armed for this activation. Its identities are fixed for the activation, so the
    // continuation poll reuses it and rechecks only the binding's live state.
    std::optional<RuntimeChannelDirectContinuationTarget> armed_continuation;
    // Set when this owner submits a process-local transported publication, and consumed by the
    // owner's next post-compute activation cut. Emits and that cut are serialized by the owner's
    // activation, so the flag needs no synchronization of its own.
    bool transported_since_cut{};
  };

  /// Carry one allocation-free activation-prepare transaction across its ordered phases.
  struct ActivationPrepareTransaction {
    ActivationOwnerState* owner{};
    bool initial_prepare{};
  };

  [[nodiscard]] expected<void, Error> bind_activation_owners(const ExecutionPlanState& plan);

  [[nodiscard]] ActivationOwnerState* find_activation_owner(VertexId owner) noexcept;

  [[nodiscard]] const ActivationOwnerState* find_activation_owner(VertexId owner) const noexcept;

  [[nodiscard]] expected<ActivationPrepareTransaction, Error> begin_activation_prepare(
      ActivationOwnerState& owner, std::uint64_t generation,
      std::span<const RuntimeChannelActivationSelection> selections) noexcept;

  void rollback_activation_prepare(ActivationPrepareTransaction& transaction) noexcept;

  [[nodiscard]] expected<void, Error> stage_activation_selections(
      ActivationPrepareTransaction& transaction,
      std::span<const RuntimeChannelActivationSelection> selections) noexcept;

  [[nodiscard]] expected<void, Error> construct_activation_predicate_views(
      ActivationPrepareTransaction& transaction) noexcept;

  static void commit_activation_prepare(ActivationPrepareTransaction& transaction) noexcept;

  [[nodiscard]] expected<void, Error> validate_activation_retirement(
      const ActivationOwnerState& owner, std::uint64_t generation,
      bool activation_accepted) const noexcept;

  [[nodiscard]] expected<void, Error> retire_activation_inputs(
      ActivationOwnerState& owner, bool activation_accepted, bool signal_direct_progress) noexcept;

  static void finish_activation_retirement(ActivationOwnerState& owner) noexcept;

  static void rollback_sample_retirement(InputPortState& input) noexcept;

  [[nodiscard]] static expected<void, Error> select_sample_retirement(
      InputPortState& input, std::span<const TemporalSampleRecord> samples,
      bool terminal_history_release) noexcept;

  [[nodiscard]] static expected<void, Error> validate_sample_retirement(
      const InputPortState& input, bool terminal_history_release) noexcept;

  [[nodiscard]] expected<void, Error> commit_sample_retirement(
      InputPortState& input, bool terminal_history_release, bool signal_direct_progress,
      std::uint32_t* promoted_out) noexcept;

  // Select, validate, retire, and promote an ordinary selector-history retirement of a
  // single-subscriber input under one subscriber-mutex section. It has the same outcomes as
  // select_sample_retirement(), validate_sample_retirement(), and commit_sample_retirement() in
  // turn, with no window between them.
  [[nodiscard]] expected<void, Error> retire_single_subscriber_samples(
      InputPortState& input, std::span<const TemporalSampleRecord> samples,
      bool signal_direct_progress, std::uint32_t* promoted_out) noexcept;

#if !defined(NDEBUG)
  [[nodiscard]] std::atomic_flag& owner_mutation_flag(VertexId owner) noexcept;
#endif

  void cache_direct_continuation_candidate(VertexId destination,
                                           ReadinessBinding& binding) noexcept;

  // Bind `producer`'s direct continuation to the recipient that feeds `destination_input`, through
  // a fanout output or into an owner with other inputs. Returns false and changes nothing unless
  // that recipient is a same-session direct subscriber on the producer's only direct output.
  [[nodiscard]] bool enable_chain_continuation(VertexId producer,
                                               PortId destination_input) noexcept;

  [[nodiscard]] std::optional<RuntimeChannelDirectContinuationTarget> direct_continuation_target(
      VertexId producer, std::uint32_t execution_lane, bool require_clean_revision,
      bool require_empty_binding) const noexcept;

  void note_direct_continuation_publication(OutputState& output, SubscriberState& subscriber,
                                            std::uint64_t publication_sequence,
                                            std::uint64_t prior_revision,
                                            std::uint64_t destination_revision) noexcept;

  enum class PumpScanLockSite : std::uint8_t { kEmit, kCut, kPump };

  // Acquire pump_scan_mutex_. Uninstrumented sessions lock directly; instrumented sessions also
  // record contended acquisitions and the time spent waiting for the lock at each site. A
  // contended acquisition is counted as soon as the first attempt fails, before the wait.
  [[nodiscard]] std::unique_lock<std::timed_mutex> acquire_pump_scan_lock(
      const PumpScanLockSite site) noexcept {
    if (phase_counters_ == nullptr) {
      return std::unique_lock<std::timed_mutex>{pump_scan_mutex_};
    }
    const auto index = static_cast<std::size_t>(site);
    constexpr std::array acquisitions{RuntimePhaseCounter::kChannelScanLockAcquisitionsEmit,
                                      RuntimePhaseCounter::kChannelScanLockAcquisitionsCut,
                                      RuntimePhaseCounter::kChannelScanLockAcquisitionsPump};
    constexpr std::array contended{RuntimePhaseCounter::kChannelScanLockContendedEmit,
                                   RuntimePhaseCounter::kChannelScanLockContendedCut,
                                   RuntimePhaseCounter::kChannelScanLockContendedPump};
    constexpr std::array waited{RuntimePhaseCounter::kChannelScanLockWaitNsEmit,
                                RuntimePhaseCounter::kChannelScanLockWaitNsCut,
                                RuntimePhaseCounter::kChannelScanLockWaitNsPump};
    phase_counters_->increment(acquisitions[index]);
    std::unique_lock<std::timed_mutex> lock{pump_scan_mutex_, std::try_to_lock};
    if (!lock.owns_lock()) {
      const auto started = std::chrono::steady_clock::now();
      phase_counters_->increment(contended[index]);
      lock.lock();
      const auto wait = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - started);
      phase_counters_->add(waited[index], static_cast<std::uint64_t>(wait.count()));
    }
    return lock;
  }

  // Acquire pump_scan_mutex_ for the post-compute cut, or return an unowned lock once `settled()`
  // shows that the holder has already received this activation's deliveries. The cut never waits
  // without a timeout (lock_unless_settled); instrumented sessions record contention as above.
  template <typename Settled>
  [[nodiscard]] std::unique_lock<std::timed_mutex> acquire_cut_scan_lock(
      Settled&& settled) noexcept {
    if (phase_counters_ == nullptr) {
      return lock_unless_settled(pump_scan_mutex_, std::forward<Settled>(settled));
    }
    phase_counters_->increment(RuntimePhaseCounter::kChannelScanLockAcquisitionsCut);
    std::unique_lock<std::timed_mutex> lock{pump_scan_mutex_, std::try_to_lock};
    if (lock.owns_lock()) {
      return lock;
    }
    const auto started = std::chrono::steady_clock::now();
    phase_counters_->increment(RuntimePhaseCounter::kChannelScanLockContendedCut);
    lock = lock_unless_settled(pump_scan_mutex_, std::forward<Settled>(settled));
    const auto wait = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started);
    phase_counters_->add(RuntimePhaseCounter::kChannelScanLockWaitNsCut,
                         static_cast<std::uint64_t>(wait.count()));
    return lock;
  }

  enum class InputCacheAdmissionSite : std::uint8_t { kDirect, kTransport };

  // Acquire one input cache mutex on an admission path: a producer's direct store, or a transport
  // receive's destination reservation and store. Uninstrumented sessions lock directly;
  // instrumented sessions also record contended acquisitions (counted when the first attempt
  // fails) and the time spent waiting, which is how a producer or the pump waiting on a consumer
  // lane's cache shows up.
  [[nodiscard]] std::unique_lock<CountedMutex> acquire_admission_cache_lock(
      const SubscriberState& subscriber, const InputCacheAdmissionSite site) noexcept {
    if (phase_counters_ == nullptr) {
      return std::unique_lock<CountedMutex>{subscriber.mutex};
    }
    std::unique_lock<CountedMutex> lock{subscriber.mutex, std::try_to_lock};
    if (!lock.owns_lock()) {
      const auto index = static_cast<std::size_t>(site);
      constexpr std::array contended{RuntimePhaseCounter::kChannelInputCacheLockContendedDirect,
                                     RuntimePhaseCounter::kChannelInputCacheLockContendedTransport};
      constexpr std::array waited{RuntimePhaseCounter::kChannelInputCacheLockWaitNsDirect,
                                  RuntimePhaseCounter::kChannelInputCacheLockWaitNsTransport};
      const auto started = std::chrono::steady_clock::now();
      phase_counters_->increment(contended[index]);
      lock.lock();
      const auto wait = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - started);
      phase_counters_->add(waited[index], static_cast<std::uint64_t>(wait.count()));
    }
    return lock;
  }

  // Leave transported publications that no owner cut will settle to the next cut of any owner.
  // Called after the progress counts are recorded, so a cut that clears the flag before reading
  // them either sees those deliveries unsettled or leaves the flag set for the next cut.
  void require_session_cut() noexcept {
    transport_settle_fallback_.store(true, std::memory_order_release);
  }

  // An owner whose cut will not run (its callback failed) hands its obligation to any later cut.
  void defer_activation_cut(const VertexId producer) noexcept {
    ActivationOwnerState* const owner = find_activation_owner(producer);
    if (owner != nullptr && std::exchange(owner->transported_since_cut, false)) {
      require_session_cut();
    }
  }

  // Name the first quiescence condition that still requires a drain, or none once the cut holds.
  [[nodiscard]] std::optional<RuntimePhaseCounter> process_local_activation_cut_miss()
      const noexcept {
    if (blocked_transport_retry_pending()) {
      return RuntimePhaseCounter::kChannelActivationCutMissBlockedLatch;
    }
    if (!process_local_progress_tracking_valid_.load(std::memory_order_acquire)) {
      return RuntimePhaseCounter::kChannelActivationCutMissTrackingInvalid;
    }
    const auto in_flight = [&]() noexcept {
      return track_concurrent_process_local_publications_ &&
             process_local_publications_in_flight_.load(std::memory_order_acquire) != 0U;
    };
    if (in_flight()) {
      return RuntimePhaseCounter::kChannelActivationCutMissInFlight;
    }
    const std::uint64_t published =
        process_local_published_deliveries_.load(std::memory_order_acquire);
    if (published != settled_process_local_deliveries()) {
      return RuntimePhaseCounter::kChannelActivationCutMissUndelivered;
    }
    if (in_flight()) {
      return RuntimePhaseCounter::kChannelActivationCutMissInFlight;
    }
    return std::nullopt;
  }

  [[nodiscard]] expected<void, Error> ensure_process_local_activation_cut(
      const RuntimeChannelActivationCutOrigin origin, const VertexId producer = {}) noexcept {
    if (!armed_.load(std::memory_order_acquire)) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto current = status();
    if (!current) {
      return make_unexpected(std::move(current).error());
    }
    if (pump_subscribers_.empty()) {
      return {};
    }
    count_phase(phase_counters_, RuntimePhaseCounter::kChannelActivationCuts);
    count_phase(phase_counters_, activation_cut_counter(origin, false));
    // Delivery inside emit() settles publications before emit() returns, so the progress check
    // below already skips cheaply. There it is also the only retry for a head that a drain left
    // pending without latching (a catalog staged head, a freshness handoff): no pump scans for it.
    if (producer && !emit_admission_) {
      // The cut settles what this activation published. One that transported nothing has nothing
      // to settle, so it skips without reading session-wide progress, which other lanes' in-flight
      // publications would otherwise make it drain. A latched blocked head still takes the full
      // path below, which retries it.
      ActivationOwnerState* const owner = find_activation_owner(producer);
      const bool transported =
          owner != nullptr && std::exchange(owner->transported_since_cut, false);
      if (!transported && !blocked_transport_retry_pending() &&
          !transport_settle_fallback_.load(std::memory_order_acquire)) {
        count_phase(phase_counters_, RuntimePhaseCounter::kChannelActivationCutSkips);
        count_phase(phase_counters_, RuntimePhaseCounter::kChannelActivationCutSkipsNoTransport);
        return {};
      }
    }
    // This cut settles everything counted so far, including publications no owner cut covers.
    // The acquire exchange keeps the progress reads below after it (see require_session_cut()).
    static_cast<void>(transport_settle_fallback_.exchange(false, std::memory_order_acq_rel));

    if (!process_local_activation_cut_miss()) {
      count_phase(phase_counters_, RuntimePhaseCounter::kChannelActivationCutSkips);
      return {};
    }

    // The pump may already be inside its finite scan when compute retires. Joining that scan and
    // completing one more bounded pass establishes the activation cut: every process-local sample
    // published by the just-finished serial compute is either in typed storage or remains queued
    // behind ordinary typed-store backpressure before the controller can freeze another candidate.
    const auto cut_lock = acquire_cut_scan_lock(
        [&]() noexcept { return !process_local_activation_cut_miss().has_value(); });
    if (!cut_lock.owns_lock()) {
      // The holder, usually the pump, received this cut's deliveries while it waited.
      count_phase(phase_counters_, RuntimePhaseCounter::kChannelActivationCutSkips);
      count_phase(phase_counters_, RuntimePhaseCounter::kChannelActivationCutSkipsWhileWaiting);
      return {};
    }
    const std::optional<RuntimePhaseCounter> locked_miss = process_local_activation_cut_miss();
    if (!locked_miss) {
      count_phase(phase_counters_, RuntimePhaseCounter::kChannelActivationCutSkips);
      count_phase(phase_counters_, RuntimePhaseCounter::kChannelActivationCutSkipsAfterLock);
      return {};
    }
    if (!initialize_controller_descriptor_import()) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    count_phase(phase_counters_, RuntimePhaseCounter::kChannelActivationCutDrains);
    count_phase(phase_counters_, activation_cut_counter(origin, true));
    count_phase(phase_counters_, *locked_miss);
    // A pump-observed blocked head remains authoritative even when unrelated inbound and outbound
    // progress totals happen to cancel in a relay process. The serialized retry below either
    // materializes that head or relatches the condition if typed retention is still full.
    PortId failed_port;
    VertexId failed_owner;
    std::optional<StructuralDiagnosticId> terminal_diagnostic;
    if (activation_cut_index_ready()) {
      auto indexed = drain_indexed_activation_cut(failed_port, failed_owner, terminal_diagnostic);
      if (!indexed) {
        HOLOSCAN_LOG_AUTOMATIC_ERROR(
            "Indexed post-compute Channel activation cut failed with {} on port {}",
            error_code_name(indexed.error().code),
            failed_port.value());
        fail(indexed.error(), failed_owner, terminal_diagnostic);
        return make_unexpected(std::move(indexed).error());
      }
      if (*indexed) {
        return status();
      }
    }
    // Concurrent admission: only the capacity release that can unblock a head retries it; complete
    // retirement and terminal cuts still try every head.
    auto drained = drain_unmonitored_inputs(
        failed_port,
        failed_owner,
        terminal_diagnostic,
        DrainScope::kInventory,
        nullptr,
        true,
        {},
        concurrent_admission_ && origin == RuntimeChannelActivationCutOrigin::kPostCompute);
    if (!drained) {
      HOLOSCAN_LOG_AUTOMATIC_ERROR("Post-compute Channel activation cut failed with {} on port {}",
                                   error_code_name(drained.error().code),
                                   failed_port.value());
      fail(drained.error(), failed_owner, terminal_diagnostic);
      return make_unexpected(std::move(drained).error());
    }
    rearm_activation_cut_index_if_drained();
    return status();
  }

  [[nodiscard]] static constexpr RuntimePhaseCounter activation_cut_counter(
      const RuntimeChannelActivationCutOrigin origin, const bool drained) noexcept {
    switch (origin) {
      case RuntimeChannelActivationCutOrigin::kRetirementRetry:
        return drained ? RuntimePhaseCounter::kChannelActivationCutDrainsRetirementRetry
                       : RuntimePhaseCounter::kChannelActivationCutsRetirementRetry;
      case RuntimeChannelActivationCutOrigin::kTerminal:
        return drained ? RuntimePhaseCounter::kChannelActivationCutDrainsTerminal
                       : RuntimePhaseCounter::kChannelActivationCutsTerminal;
      case RuntimeChannelActivationCutOrigin::kPostCompute:
        break;
    }
    return drained ? RuntimePhaseCounter::kChannelActivationCutDrainsPostCompute
                   : RuntimePhaseCounter::kChannelActivationCutsPostCompute;
  }

  [[nodiscard]] expected<std::optional<RuntimeChannelDirectContinuationReceipt>, Error>
  poll_direct_continuation(VertexId producer, std::uint32_t execution_lane,
                           TimePoint observed_at) noexcept;

  [[nodiscard]] bool validate_direct_continuation_receipt(
      const RuntimeChannelDirectContinuationReceipt& receipt) const noexcept;

  [[nodiscard]] static expected<void, Error> rollback_tensor_access(InputPortState& input,
                                                                    std::size_t access_checkpoint,
                                                                    std::size_t lease_checkpoint,
                                                                    Error error) noexcept;

  [[nodiscard]] static Tensor* tensor_for_access(InputPortState& input, SampleSlot& slot,
                                                 bool direct_tensor, std::uint32_t index) noexcept;

  [[nodiscard]] static bool tensor_device_requires_consumer_stream(
      DLDeviceType device_type) noexcept;

  [[nodiscard]] static expected<void, Error> prepare_tensor_stream_access(
      Tensor& tensor, cudaStream_t consumer_stream,
      const std::shared_ptr<const holomq::cuda::StreamLifetime>& lifetime) noexcept;

  [[nodiscard]] expected<void, Error> initialize_pump_indices();

  [[nodiscard]] expected<void, Error> initialize_activation_cut_subscriber_index();
  // Publication credits: list each process-local endpoint's blocking transported recipients.
  [[nodiscard]] expected<void, Error> initialize_publication_credit_index();

  // Freeze each emit-admitted endpoint's exact emit_recipients.
  [[nodiscard]] expected<void, Error> initialize_emit_recipient_index();

  // Deliveries that no drain will receive again: received, replaced in transport, or discarded by
  // a restart.
  [[nodiscard]] std::uint64_t settled_process_local_deliveries() const noexcept {
    constexpr std::uint64_t kMaximum = std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t received =
        process_local_received_deliveries_.load(std::memory_order_acquire);
    const std::uint64_t replaced =
        process_local_replaced_deliveries_.load(std::memory_order_acquire);
    const std::uint64_t discarded =
        process_local_discarded_deliveries_.load(std::memory_order_acquire);
    if (replaced > kMaximum - received) {
      return kMaximum;
    }
    return discarded > kMaximum - received - replaced ? kMaximum : received + replaced + discarded;
  }

  // Count one delivery that HoloMQ replaced before any drain received it.
  void record_process_local_replacement() noexcept;

  // Count deliveries that a restart consumed from transport before any drain received them.
  void record_process_local_discards(std::uint32_t deliveries) noexcept;

  // The replacement observer for a publisher of a process-local service; a cross-process service's
  // replaced receivers belong to other processes and are not counted here.
  [[nodiscard]] channel_internal::ReplacementObserver transport_replacement_observer(
      const bool cross_process_transport) noexcept {
    if (cross_process_transport) {
      return {};
    }
    return channel_internal::ReplacementObserver{
        .state = this,
        .replaced =
            [](void* state) noexcept {
              auto& session = *static_cast<Impl*>(state);
              session.record_process_local_replacement();
              count_phase(session.phase_counters_,
                          RuntimePhaseCounter::kChannelTransportReplacements);
            },
    };
  }

  [[nodiscard]] bool background_pump_required() const noexcept;

  [[nodiscard]] static expected<void, Error> prepare_tensor_input_access(
      InputPortState& input, cudaStream_t consumer_stream,
      const std::shared_ptr<const holomq::cuda::StreamLifetime>& lifetime) noexcept;

  [[nodiscard]] bool replace_wait_handles(
      SubscriberState& subscriber, const std::uint64_t replacement,
      const std::optional<std::uint64_t> terminal_replacement) noexcept {
    if (subscriber.wait_handle == 0U || replacement == 0U ||
        (subscriber.terminal_wait_handle != 0U) != terminal_replacement.has_value() ||
        (terminal_replacement &&
         (*terminal_replacement == 0U || replacement == *terminal_replacement))) {
      return false;
    }
    const std::uint64_t original = subscriber.wait_handle;
    const std::uint64_t original_terminal = subscriber.terminal_wait_handle;
    const auto owned_handle = [&](const std::uint64_t handle) {
      const auto found = wait_subscriber_by_handle_.find(handle);
      return found != wait_subscriber_by_handle_.end() && found->second == &subscriber;
    };
    if (!owned_handle(original) || (original_terminal != 0U && !owned_handle(original_terminal))) {
      return false;
    }
    const auto replacement_available = [&](const std::uint64_t handle) {
      return !wait_subscriber_by_handle_.contains(handle) || handle == original ||
             handle == original_terminal;
    };
    if (!replacement_available(replacement) ||
        (terminal_replacement && !replacement_available(*terminal_replacement))) {
      return false;
    }

    using WaitHandleNode = decltype(wait_subscriber_by_handle_)::node_type;
    WaitHandleNode node = wait_subscriber_by_handle_.extract(original);
    WaitHandleNode terminal_node;
    if (original_terminal != 0U) {
      terminal_node = wait_subscriber_by_handle_.extract(original_terminal);
    }
    const auto restore = [&](WaitHandleNode&& restore_node, const std::uint64_t handle) noexcept {
      if (restore_node.empty()) {
        return false;
      }
      restore_node.key() = handle;
      return wait_subscriber_by_handle_.insert(std::move(restore_node)).inserted;
    };
    if (node.empty() || node.mapped() != &subscriber ||
        (original_terminal != 0U &&
         (terminal_node.empty() || terminal_node.mapped() != &subscriber))) {
      static_cast<void>(restore(std::move(node), original));
      static_cast<void>(restore(std::move(terminal_node), original_terminal));
      return false;
    }

    node.key() = replacement;
    auto inserted = wait_subscriber_by_handle_.insert(std::move(node));
    if (!inserted.inserted) {
      static_cast<void>(restore(std::move(inserted.node), original));
      static_cast<void>(restore(std::move(terminal_node), original_terminal));
      return false;
    }
    if (original_terminal == 0U) {
      return true;
    }

    // A nonzero original terminal key establishes the replacement above.
    terminal_node.key() = *terminal_replacement;  // NOLINT(bugprone-unchecked-optional-access)
    auto terminal_inserted = wait_subscriber_by_handle_.insert(std::move(terminal_node));
    if (terminal_inserted.inserted) {
      return true;
    }
    auto inserted_node = wait_subscriber_by_handle_.extract(replacement);
    static_cast<void>(restore(std::move(inserted_node), original));
    static_cast<void>(restore(std::move(terminal_inserted.node), original_terminal));
    return false;
  }

  [[nodiscard]] static InputPortState* find_activation_input(ActivationOwnerState& owner,
                                                             PortId port) noexcept;

  class ReadinessBinding final : public PortBinding {
   public:
    ReadinessBinding(Impl* owner, VertexId vertex, RuntimePortBindingRequest request,
                     std::size_t candidate_capacity, std::size_t terminal_source_count,
                     std::span<InputPortState* const> inputs, bool polls_external_transport);

    ~ReadinessBinding() override;

    [[nodiscard]] expected<void, Error> poll(TimePoint observed_at) noexcept override;

    [[nodiscard]] expected<void, Error> poll_blocking(TimePoint observed_at) noexcept override;

    [[nodiscard]] expected<void, Error> poll_after_contention(
        TimePoint observed_at) noexcept override;

    [[nodiscard]] expected<void, Error> poll_with_acquisition(
        TimePoint observed_at, RuntimeChannelIncompletePollKind flavour) noexcept;

    [[nodiscard]] SourceId source() const noexcept override;

    [[nodiscard]] bool last_poll_incomplete() const noexcept override {
      return last_poll_incomplete_;
    }

    [[nodiscard]] bool last_poll_lock_contended() const noexcept override {
      return last_poll_lock_contended_;
    }

    [[nodiscard]] std::optional<PortBindingPollTarget> poll_target() const noexcept override;

    [[nodiscard]] expected<void, Error> enqueue_window_timeout(
        PortId port, TimePoint terminal_time) noexcept override;

    [[nodiscard]] expected<void, Error> retire(std::uint64_t sequence) noexcept override;
    void cancel() noexcept override;
    [[nodiscard]] expected<std::uint64_t, Error> advance_source_generation() noexcept override;

    [[nodiscard]] expected<std::size_t, Error> drain(
        std::span<RuntimeEvent> destination) noexcept override;

    [[nodiscard]] std::size_t capacity() const noexcept override;
    [[nodiscard]] bool closed() const noexcept override;
    [[nodiscard]] expected<void, Error> terminal_status() const noexcept override;
    // Process-local deliveries and their terminal records wake the owner; external imports poll.
    [[nodiscard]] bool progress_announced() const noexcept override {
      return !polls_external_transport_;
    }

   private:
    struct InputRevision {
      InputPortState* input{};
      std::uint64_t observed{};
      std::uint64_t pending{};
    };

    struct SlotCandidate {
      SubscriberState* subscriber{};
      SampleSlot* slot{};
      std::uint64_t arrival_order{};
    };

    [[nodiscard]] static bool later_candidate(const SlotCandidate& lhs,
                                              const SlotCandidate& rhs) noexcept;

    [[nodiscard]] bool owns_input(const InputPortState* input) const noexcept;

    // The revision last observed for `input`, or nullptr when this binding does not own it.
    [[nodiscard]] const InputRevision* revision_of(const InputPortState* input) const noexcept;

    // Whether every owned input other than `except` has no unobserved readiness change.
    [[nodiscard]] bool others_observed(const InputPortState* except) const noexcept;

    void detach_from_owner() noexcept;

    void publish_deferred_ring_retry() noexcept {
      if (!ring_retry_pending_) {
        return;
      }
      ring_retry_pending_ = false;
      for (const InputRevision& revision : input_revisions_) {
        static_cast<void>(revision.input->publish_readiness_change());
      }
    }

    [[nodiscard]] expected<void, Error> push_end_of_stream(PortId port,
                                                           TimePoint observed_at) noexcept;

    [[nodiscard]] expected<void, Error> push(SampleSlot& slot, std::uint64_t& sequence,
                                             TimePoint observed_at) noexcept;

    Impl* owner_{};
    VertexId vertex_{};
    RuntimePortBindingRequest request_{};
    std::vector<RuntimeEvent> storage_;
    std::vector<std::uint64_t> timeout_sequences_;
    std::vector<SlotCandidate> candidates_;
    std::vector<std::uint64_t> closure_sequences_;
    std::vector<InputRevision> input_revisions_;
    std::size_t head_{};
    std::size_t tail_{};
    std::size_t size_{};
    std::size_t timeout_count_{};
    std::size_t closure_count_{};
    std::size_t terminal_source_count_{};
    std::size_t terminal_sources_retired_{};
    std::uint64_t next_source_sequence_{request_.initial_source_sequence};
    bool polls_external_transport_{};
    bool ring_retry_pending_{};
    // Written by the owning partition; another partition's direct-continuation arming reads it.
    std::atomic<bool> closed_{};
    bool last_poll_incomplete_{};
    bool last_poll_lock_contended_{};

    friend class Impl;
    friend class RuntimeChannelSessionTestAccess;
  };

  Impl(ChannelRuntime runtime, PlanGenerationId plan_generation, RunGenerationId run_generation,
       std::span<const ClockConversionRecord> clock_conversions,
       std::span<const SupportThreadAffinityPlanRecord> support_thread_affinities,
       std::shared_ptr<RuntimeSupportThreadControl> support_thread_control,
       std::shared_ptr<RuntimeChannelIncompletePollTestControl> incomplete_poll_test_control,
       const bool force_process_local_channel_pump,
       const bool track_concurrent_process_local_publications, RuntimePhaseCounters* phase_counters,
       RuntimeChannelReadinessWake readiness_wake)
      : runtime_(std::move(runtime)),
        plan_generation_(plan_generation),
        run_generation_(run_generation),
        clock_conversions_(clock_conversions.begin(), clock_conversions.end()),
        support_thread_affinities_(support_thread_affinities),
        support_thread_control_(std::move(support_thread_control)),
        incomplete_poll_test_control_(std::move(incomplete_poll_test_control)),
        force_process_local_channel_pump_(force_process_local_channel_pump),
        track_concurrent_process_local_publications_(track_concurrent_process_local_publications),
        phase_counters_(phase_counters),
        readiness_wake_(readiness_wake) {}

  ~Impl() {
    cancel();
    static_cast<void>(join());
    observe_teardown(RuntimeChannelTeardownPhase::kPumpJoined);
    detach_readiness_bindings();
    for (const auto& input : inputs_) {
      input->type_ops->unbind_input(input->declaration, &input->binding);
    }
    for (const auto& output : outputs_) {
      output->type_ops->unbind_output(output->declaration, &output->binding);
    }
    blocked_head_wait_set_ = nullptr;
    wait_set_.reset();
    wait_subscriber_by_handle_.clear();
    pump_subscribers_.clear();
    subscribers_.clear();
    activation_owners_.clear();
    // The cold observation owner remains alive throughout this destructor body; destroy every
    // OutputState borrower before member destruction can release that trailing shared owner.
    outputs_.clear();
    inputs_.clear();
    observe_teardown(RuntimeChannelTeardownPhase::kEndpointChildrenDestroyed);
    runtime_.reset();
    observe_teardown(RuntimeChannelTeardownPhase::kRuntimeDestroyed);
  }

  void unregister_readiness_binding(ReadinessBinding& binding) noexcept;

  void detach_readiness_bindings() noexcept;

  void observe_teardown(const RuntimeChannelTeardownPhase phase) noexcept {
    if (teardown_observer_ != nullptr) {
      teardown_observer_(teardown_observer_state_, phase);
    }
  }

  [[nodiscard]] expected<void, Error> set_teardown_observer_for_test(
      void* state, RuntimeChannelTeardownObserver observer) noexcept {
    if (armed_.load(std::memory_order_acquire) || state == nullptr || observer == nullptr ||
        teardown_observer_ != nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    teardown_observer_state_ = state;
    teardown_observer_ = observer;
    return {};
  }

  [[nodiscard]] expected<void, Error> fail_next_activation_retirement_for_test(
      const bool failed_callback_only) noexcept {
    if (armed_.load(std::memory_order_acquire) || activation_retirement_observer_ != nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    activation_retirement_observer_state_ = this;
    if (failed_callback_only) {
      // Other owners may retire concurrently, so the observer stays installed and a one-shot
      // atomic selects the first retirement of a failed callback.
      failed_callback_retirement_fault_.store(true, std::memory_order_release);
      activation_retirement_observer_ =
          [](void* const state,
             VertexId,
             std::uint64_t,
             const bool activation_accepted) noexcept -> expected<void, Error> {
        auto& self = *static_cast<Impl*>(state);
        if (activation_accepted ||
            !self.failed_callback_retirement_fault_.exchange(false, std::memory_order_acq_rel)) {
          return {};
        }
        return make_unexpected(
            Error{ErrorCode::kFailure, "RUNTIME_TEST_CHANNEL_ACTIVATION_RETIREMENT_FAILED"});
      };
      return {};
    }
    activation_retirement_observer_ =
        [](void* const state, VertexId, std::uint64_t, bool) noexcept -> expected<void, Error> {
      auto& self = *static_cast<Impl*>(state);
      self.activation_retirement_observer_state_ = nullptr;
      self.activation_retirement_observer_ = nullptr;
      return make_unexpected(
          Error{ErrorCode::kFailure, "RUNTIME_TEST_CHANNEL_ACTIVATION_RETIREMENT_FAILED"});
    };
    return {};
  }

  [[nodiscard]] expected<void, Error> fail_fixed_tensor_shape_allocation_after_successes_for_test(
      const std::size_t successful_allocations) noexcept {
    fixed_tensor_shape_allocations_before_failure_for_test_ = successful_allocations;
    return {};
  }

  [[nodiscard]] expected<void, Error> set_activation_retirement_observer_for_test(
      void* const state, const RuntimeChannelActivationRetirementObserver observer) noexcept {
    if (armed_.load(std::memory_order_acquire) || state == nullptr || observer == nullptr ||
        activation_retirement_observer_ != nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    activation_retirement_observer_state_ = state;
    activation_retirement_observer_ = observer;
    return {};
  }

  // Each branch proves one compiler-frozen clock-domain relation before exposing coordinates.
  [[nodiscard]] expected<std::pair<TimePoint, TimePoint>, Error> event_coordinates(
      const SampleSlot& slot, const RuntimePortBindingRequest& request,
      TimePoint observed_at) const noexcept;

  // The visible branch sequence preserves the bounded snapshot, heap ordering, and revalidation
  // protocol across independently locked input caches.
  [[nodiscard]] expected<bool, Error> poll_inputs(
      ReadinessBinding& binding, TimePoint observed_at,
      RuntimeChannelIncompletePollKind flavour) noexcept;

  template <typename Initialize>
  [[nodiscard]] expected<std::pair<OutputState*, bool>, Error> ensure_output_state(
      const RuntimeChannelPortReplayRecord& replay, const PortPlanRecord& port_plan,
      RuntimeTensorOutputTestControl* const tensor_test_control, Initialize&& initialize) {
    const auto existing =
        std::find_if(outputs_.begin(), outputs_.end(), [&](const auto& candidate) {
          return candidate->port == replay.port;
        });
    if (existing != outputs_.end()) {
      OutputState* const output = existing->get();
      if (output->owner != replay.owner || output->declaration != replay.declaration ||
          output->type_ops != replay.type_ops || output->tensor_failure != replay.tensor_failure) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      return std::pair{output, false};
    }

    auto created = std::make_unique<OutputState>();
    created->port = replay.port;
    created->owner = replay.owner;
    created->session = this;
    created->declaration = replay.declaration;
    created->type_ops = replay.type_ops;
    created->tensor_failure = replay.tensor_failure;
    created->phase_counters = phase_counters_;
    created->tensor_test_control = tensor_test_control;
    created->max_emits_per_activation = port_plan.max_emits_per_compute;
    if (replay.type_ops->codec_family == RuntimePayloadCodecFamily::kCustomPod) {
      if (!port_plan.payload_identity.pod) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      created->pod_identity = port_plan.payload_identity.pod;
    }
    // Caller-specific construction and validation complete before the ownership transaction is
    // committed to outputs_. A failed initializer therefore cannot expose a partial output to
    // teardown or later fanout arms.
    auto initialized = std::forward<Initialize>(initialize)(*created);
    if (!initialized) {
      return make_unexpected(std::move(initialized).error());
    }
    OutputState* const output = created.get();
    outputs_.push_back(std::move(created));
    return std::pair{output, true};
  }

  [[nodiscard]] expected<InputPortState*, Error> ensure_input_state(
      const ExecutionPlanState& plan, const RuntimeChannelPortReplayRecord& replay) {
    if (InputPortState* const existing = find_input(replay.port); existing != nullptr) {
      if (existing->owner != replay.owner || existing->declaration != replay.declaration ||
          existing->type_ops != replay.type_ops) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      return existing;
    }

    auto created = std::make_unique<InputPortState>();
    created->port = replay.port;
    created->owner = replay.owner;
    created->declaration = replay.declaration;
    created->type_ops = replay.type_ops;
    created->phase_counters = phase_counters_;
    const InputSelectionPlanRecord* const selection =
        find_input_selection_by_input(plan, replay.port);
    if (selection != nullptr) {
      if (!selection->reference_clock.valid() || selection->cache_capacity == 0U ||
          (selection->missing == TemporalRecordMissingKind::kReuseLast &&
           (selection->cardinality != InputSelectionCardinality::kSingleSample ||
            selection->batch_selection_mode != BatchSelectionMode::kNotApplicable ||
            selection->reuse_max_age_ns <= 0))) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      created->reference_clock = selection->reference_clock;
      created->conversion_id = selection->conversion_id;
      created->capture_time_required = selection->selector != InputSelectorKind::kDrainAll;
      created->reuse_last = selection->missing == TemporalRecordMissingKind::kReuseLast;
      // ReuseLast is independent retention, so keep its exact fallback outside the authored typed
      // depth while a causal successor waits for its trigger.
      created->private_depth_one_reuse_slot =
          selection->selector == InputSelectorKind::kLatestWithin && created->reuse_last &&
          selection->cache_capacity == 1U;
      created->drain_all = selection->selector == InputSelectorKind::kDrainAll;
      if (selection->conversion_id == 0U) {
        created->capture_clock = selection->reference_clock;
      } else {
        const ClockConversionRecord* const conversion =
            find_clock_conversion_by_id(plan, selection->conversion_id);
        if (conversion == nullptr || conversion->plan_generation != plan.plan_generation ||
            !conversion->from.valid() || conversion->to != selection->reference_clock ||
            conversion->rounding != ClockRoundingMode::kNearestTiesToEven ||
            conversion->overflow != ClockOverflowPolicy::kError) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        created->capture_clock = conversion->from;
        created->capture_to_reference = DirectAffineClockEdge{
            .source = conversion->from,
            .target = conversion->to,
            .conversion = AffineClockConversion{.scale = conversion->scale,
                                                .offset_ns = conversion->offset_ns},
        };
      }
    }
    created->binding = RuntimeInputBinding{
        .state = created.get(),
        .ops = &InputPortState::kBindingOps,
        .generation = created->active_generation,
    };
    InputPortState* const input = created.get();
    inputs_.push_back(std::move(created));
    return input;
  }

  // Materialize every typed slot before publishing subscriber storage to Runtime.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<void, Error> prepare_subscriber_storage(
      SubscriberState& subscriber, const std::uint32_t typed_slot_capacity,
      const std::uint32_t sample_slot_capacity, const std::size_t tensor_fields_per_slot,
      const bool attach_wait_set) {
    if (typed_slot_capacity == 0U || sample_slot_capacity < typed_slot_capacity) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    subscriber.refresh_staged_data_endpoint();
    subscriber.typed_slot_capacity = typed_slot_capacity;
    subscriber.active_typed_slot_capacity = typed_slot_capacity;
    subscriber.slots.reserve(sample_slot_capacity);
    for (std::uint32_t index = 0U; index < sample_slot_capacity; ++index) {
      auto storage = AlignedBlock::create(subscriber.input->type_ops->sample_size,
                                          subscriber.input->type_ops->sample_alignment);
      if (!storage) {
        return make_unexpected(std::move(storage).error());
      }
      SampleSlot slot;
      slot.subscriber = &subscriber;
      slot.sample = std::move(*storage);
      subscriber.slots.push_back(std::move(slot));
      if (subscriber.input->type_ops->persistent_sample_storage) {
        if (subscriber.input->type_ops->prepare_sample_storage == nullptr ||
            subscriber.input->type_ops->reset_sample_storage == nullptr ||
            subscriber.input->type_ops->destroy_sample_storage == nullptr ||
            (subscriber.input->type_ops->codec_family ==
                 RuntimePayloadCodecFamily::kGeneratedSchema &&
             subscriber.input->type_ops->assign_sample_storage == nullptr)) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        auto prepared = subscriber.input->type_ops->prepare_sample_storage(
            subscriber.slots.back().sample.data());
        if (!prepared) {
          return make_unexpected(std::move(prepared).error());
        }
        subscriber.slots.back().sample_storage_prepared = true;
      }
    }
    // Cold storage is needed only where an occupied slot can be replaced transactionally: a
    // lossy DropOldest subscriber or the transported ReuseLast handoff. Other subscribers do not
    // pay one full sample allocation.
    if (subscriber.supports_cold_sample_replacement()) {
      auto replacement_storage = AlignedBlock::create(subscriber.input->type_ops->sample_size,
                                                      subscriber.input->type_ops->sample_alignment);
      if (!replacement_storage) {
        return make_unexpected(std::move(replacement_storage).error());
      }
      subscriber.replacement_sample = std::move(*replacement_storage);
      if (subscriber.input->type_ops->persistent_sample_storage &&
          subscriber.input->type_ops->codec_family == RuntimePayloadCodecFamily::kGeneratedSchema) {
        auto prepared = subscriber.input->type_ops->prepare_sample_storage(
            subscriber.replacement_sample.data());
        if (!prepared)
          return make_unexpected(std::move(prepared).error());
        subscriber.replacement_sample_prepared = true;
      }
    }
    auto replacement_prepared = prepare_descriptor_replacement_sample(subscriber);
    if (!replacement_prepared) {
      return make_unexpected(std::move(replacement_prepared).error());
    }
    if (!subscriber.direct_process_local && attach_wait_set) {
      auto attached = subscriber.attach_wait_set(*wait_set_);
      if (!attached) {
        return make_unexpected(std::move(attached).error());
      }
      subscriber.wait_handle = *attached;
      subscriber.terminal_wait_handle = subscriber.secondary_wait_set_handle().value_or(0U);
      wait_attachment_count_ += subscriber.terminal_wait_handle == 0U ? 1U : 2U;
    }

    InputPortState& input = *subscriber.input;
    if (input.active_slots.max_size() - input.active_slots.capacity() < typed_slot_capacity) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    input.active_slots.reserve(input.active_slots.capacity() + typed_slot_capacity);
    const std::uint32_t reuse_retirement_headroom =
        subscriber.queue_policy == QueuePolicy::kDropOldest && input.private_depth_one_reuse_slot
            ? 1U
            : 0U;
    if (typed_slot_capacity >
            std::numeric_limits<std::uint32_t>::max() - reuse_retirement_headroom ||
        input.retirement_slots.max_size() - input.retirement_slots.capacity() <
            typed_slot_capacity + reuse_retirement_headroom) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    input.retirement_slots.reserve(input.retirement_slots.capacity() + typed_slot_capacity +
                                   reuse_retirement_headroom);
    if (tensor_fields_per_slot == 0U ||
        tensor_fields_per_slot > std::numeric_limits<std::size_t>::max() / typed_slot_capacity) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    const std::size_t tensor_lease_capacity =
        tensor_fields_per_slot * static_cast<std::size_t>(typed_slot_capacity);
    if (input.tensor_access_slots.max_size() - input.tensor_access_slots.capacity() <
            typed_slot_capacity ||
        input.tensor_read_leases.max_size() - input.tensor_read_leases.capacity() <
            tensor_lease_capacity) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    input.tensor_access_slots.reserve(input.tensor_access_slots.capacity() + typed_slot_capacity);
    input.tensor_read_leases.reserve(input.tensor_read_leases.capacity() + tensor_lease_capacity);
    return {};
  }

  [[nodiscard]] InputPortState* find_input(PortId port) noexcept {
    if (!input_by_port_.empty()) {
      return !port || port.value() > input_by_port_.size()
                 ? nullptr
                 : input_by_port_[static_cast<std::size_t>(port.value() - 1U)];
    }
    const auto found = std::find_if(
        inputs_.begin(), inputs_.end(), [&](const auto& input) { return input->port == port; });
    return found == inputs_.end() ? nullptr : found->get();
  }

  [[nodiscard]] const InputPortState* find_input(PortId port) const noexcept {
    if (!input_by_port_.empty()) {
      return !port || port.value() > input_by_port_.size()
                 ? nullptr
                 : input_by_port_[static_cast<std::size_t>(port.value() - 1U)];
    }
    const auto found = std::find_if(
        inputs_.begin(), inputs_.end(), [&](const auto& input) { return input->port == port; });
    return found == inputs_.end() ? nullptr : found->get();
  }

  [[nodiscard]] OutputState* find_output(PortId port) noexcept {
    const auto found = std::find_if(
        outputs_.begin(), outputs_.end(), [&](const auto& output) { return output->port == port; });
    return found == outputs_.end() ? nullptr : found->get();
  }

  [[nodiscard]] expected<void, Error> replace_output_publishers_for_test(
      PortId port,
      std::vector<std::unique_ptr<channel_internal::InlinePayloadPublisher>> publishers,
      const std::span<const std::uint32_t> expected_recipient_counts,
      const bool refresh_before_emit, const bool peer_required) noexcept {
    if (armed_.load(std::memory_order_acquire) || !port || publishers.empty() ||
        (!expected_recipient_counts.empty() &&
         expected_recipient_counts.size() != publishers.size()) ||
        std::any_of(publishers.begin(),
                    publishers.end(),
                    [](const auto& publisher) { return publisher == nullptr; }) ||
        std::any_of(expected_recipient_counts.begin(),
                    expected_recipient_counts.end(),
                    [](const auto count) { return count == 0U; })) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    const auto output = std::find_if(outputs_.begin(), outputs_.end(), [&](const auto& candidate) {
      return candidate->port == port;
    });
    if (output == outputs_.end() || (*output)->active) {
      return make_unexpected(
          Error{output == outputs_.end() ? ErrorCode::kNotFound : ErrorCode::kBackpressured});
    }
    auto original_endpoints = std::move((*output)->endpoints);
    (*output)->endpoints.clear();
    (*output)->endpoints.reserve(publishers.size());
    for (std::size_t index = 0U; index < publishers.size(); ++index) {
      (*output)->endpoints.push_back(OutputState::EndpointState{
          .publisher = std::move(publishers[index]),
          .expected_recipient_count =
              expected_recipient_counts.empty() ? 1U : expected_recipient_counts[index],
          .refresh_before_emit = refresh_before_emit,
          .peer_required = peer_required,
          .process_local_transport = !refresh_before_emit,
      });
      // Replacing only the backend preserves the cold recipient proof for matching private
      // emit-experiment fixtures. A changed shape deliberately has no proof and must fail closed.
      auto& endpoint = (*output)->endpoints.back();
      if (index < original_endpoints.size() && endpoint.process_local_transport) {
        endpoint.emit_admitted = original_endpoints[index].emit_admitted;
      }
      if (endpoint.emit_admitted && original_endpoints[index].process_local_transport &&
          endpoint.expected_recipient_count == original_endpoints[index].expected_recipient_count) {
        endpoint.process_local_service = std::move(original_endpoints[index].process_local_service);
        endpoint.activation_cut_subscribers =
            std::move(original_endpoints[index].activation_cut_subscribers);
        endpoint.emit_recipients = std::move(original_endpoints[index].emit_recipients);
      }
    }
    return {};
  }

  [[nodiscard]] expected<void, Error> throw_tensor_fanout_after_accepts_for_test(
      const PortId port, const std::size_t accepted_count) noexcept {
    if (armed_.load(std::memory_order_acquire) || !port) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    const auto output = std::find_if(outputs_.begin(), outputs_.end(), [&](const auto& candidate) {
      return candidate->port == port;
    });
    if (output == outputs_.end()) {
      return make_unexpected(Error{ErrorCode::kNotFound});
    }
    auto& state = **output;
    if (state.active || !state.tensor_payload() || accepted_count > state.tensor_endpoints.size() ||
        state.throw_tensor_fanout_after_accepts_for_test) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    state.throw_tensor_fanout_after_accepts_for_test = accepted_count;
    return {};
#else
    (void)accepted_count;
    return make_unexpected(Error{ErrorCode::kNotSupported});
#endif
  }

  // Witness retirement preserves ordered cache, attribution, and readiness transitions.
  [[nodiscard]] expected<void, Error> retire_witness(VertexId owner,
                                                     std::uint64_t sequence) noexcept;

  enum class WritableSlotIntent : std::uint8_t { kProbe, kClaim };

  // Typed slots a blocking publication could still claim, counted as writable_slot() counts them.
  // Requires subscriber.mutex.
  [[nodiscard]] static std::size_t blocking_typed_capacity_locked(
      const SubscriberState& subscriber) noexcept {
    std::size_t occupied = 0U;
    std::size_t free_slots = 0U;
    for (const auto& slot : subscriber.slots) {
      if ((slot.occupied && !slot.direct_ingress_unread) || slot.typed_ingress_reserved) {
        ++occupied;
      }
      if (!slot.occupied && !slot.typed_ingress_reserved) {
        ++free_slots;
      }
    }
    if (occupied >= subscriber.active_typed_slot_capacity) {
      return 0U;
    }
    return std::min<std::size_t>(subscriber.active_typed_slot_capacity - occupied, free_slots);
  }

  // A blocking direct-retained edge has no transport queue. When its consumer runs concurrently on
  // another lane, lag is ordinary pressure: wait for a typed slot within the same per-recipient
  // budget as a transported blocking send, without consuming the publication. Only this producer
  // fills the store and its consumer only frees it, so observed capacity remains until the store.
  // Returns false when the budget expires or the session is stopping or failing.
  [[nodiscard]] bool wait_for_direct_capacity(SubscriberState& subscriber) noexcept;
  // Whether a blocking direct recipient can take one more sample: waits within the graph blocking
  // budget for a recipient on another lane, and only reads a recipient on the producer's lane,
  // whose consumer cannot run before the producer's callback returns.
  [[nodiscard]] bool direct_recipient_has_capacity(SubscriberState& subscriber) noexcept;

  // Require transport credit, sent minus admitted samples below the queue depth, at
  // every credited recipient of these endpoints before any is submitted. Only this producer sends
  // to them, so granted credit stays granted until its own next send. A recipient on another lane
  // is waited for within the graph blocking budget; otherwise, or on expiry, the publication fails
  // pre-submit with kBackpressured and nothing delivered.
  template <typename Endpoints>
  [[nodiscard]] expected<void, Error> await_publication_credits(
      const Endpoints& endpoints) noexcept;

  template <bool HasReplacementCandidate = false>
  // Slot selection keeps capacity, protection, replacement, and retention policy together.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] SampleSlot* writable_slot(
      SubscriberState& subscriber, WritableSlotIntent intent,
      const SampleSlot* const replacement_candidate = nullptr) noexcept {
    if constexpr (HasReplacementCandidate) {
      if (replacement_candidate == nullptr) {
        return nullptr;
      }
    }
    const bool private_reuse_slot = subscriber.queue_policy == QueuePolicy::kDropOldest &&
                                    subscriber.input->private_depth_one_reuse_slot;
    const std::size_t typed_occupied = static_cast<std::size_t>(
        std::count_if(subscriber.slots.begin(), subscriber.slots.end(), [&](const auto& slot) {
          const bool private_reuse = private_reuse_slot && slot.reuse_retained;
          return (slot.occupied && !slot.direct_ingress_unread && !private_reuse) ||
                 (slot.typed_ingress_reserved &&
                  (!HasReplacementCandidate || &slot != replacement_candidate));
        }));
    if (typed_occupied < subscriber.active_typed_slot_capacity) {
      const auto free =
          std::find_if(subscriber.slots.begin(), subscriber.slots.end(), [](const auto& slot) {
            return !slot.occupied && !slot.typed_ingress_reserved;
          });
      return free == subscriber.slots.end() ? nullptr : &*free;
    }
    if (typed_occupied > subscriber.active_typed_slot_capacity) {
      return nullptr;
    }
    if (subscriber.queue_policy == QueuePolicy::kBlocking) {
      return nullptr;
    }
    const auto eligible = [&](const SampleSlot& slot) {
      return slot.occupied && !slot.direct_ingress_unread && !slot.active_selected &&
             !slot.reuse_retained && !slot.temporal_retained && !slot.retirement_selected &&
             !slot.typed_ingress_reserved &&
             ((slot.event_emitted && slot.witness_retired) ||
              (!slot.event_emitted && !slot.owed_to_released_claim &&
               subscriber.pre_witness_replacement_allowed));
    };
    SampleSlot* oldest = nullptr;
    SampleSlot* protected_refill = nullptr;
    for (SampleSlot& slot : subscriber.slots) {
      const bool refill_is_protected =
          slot.occupied && !slot.direct_ingress_unread && !slot.active_selected &&
          !slot.reuse_retained && !slot.temporal_retained && !slot.retirement_selected &&
          !slot.typed_ingress_reserved && !slot.event_emitted && slot.owed_to_released_claim &&
          subscriber.pre_witness_replacement_allowed;
      if (refill_is_protected &&
          (protected_refill == nullptr || slot.arrival_order < protected_refill->arrival_order)) {
        protected_refill = &slot;
      }
      if (eligible(slot) && (oldest == nullptr || slot.arrival_order < oldest->arrival_order)) {
        oldest = &slot;
      }
    }
    // A promoted refill carries no claim until its readiness event is published, but the retained
    // record was already given up on the promise that this exact successor reaches the cache.
    // Record the replacement the guard declined rather than the now-impossible loss it prevented.
    if (protected_refill != nullptr &&
        (oldest == nullptr || protected_refill->arrival_order < oldest->arrival_order)) {
      if (intent == WritableSlotIntent::kClaim &&
          (!HasReplacementCandidate ||
           (is_strictly_newer_sample(replacement_candidate->temporal, protected_refill->temporal) &&
            (oldest == nullptr ||
             is_strictly_newer_sample(replacement_candidate->temporal, oldest->temporal))))) {
        count_phase(phase_counters_,
                    RuntimePhaseCounter::kChannelReleasedClaimRefillReplacementDeclined);
        subscriber.input->last_refill_outcome.store(2U, std::memory_order_relaxed);
      }
    }
    if (oldest != nullptr && !oldest->event_emitted && intent == WritableSlotIntent::kClaim &&
        (!HasReplacementCandidate ||
         is_strictly_newer_sample(replacement_candidate->temporal, oldest->temporal))) {
      count_phase(phase_counters_, RuntimePhaseCounter::kChannelPreWitnessReplacements);
    }
    return oldest;
  }

  [[nodiscard]] SampleSlot* writable_direct_ingress_slot(SubscriberState& subscriber) noexcept;

  /// Restore the authored direct-ingress depth before abandoning an uncommitted refill target.
  ///
  /// A fenced target occupies the one preallocated headroom slot and is excluded from the
  /// authored staging depth. If a producer used that headroom, releasing the fence must discard
  /// the oldest staged record so the newest-N survivor invariant remains true. A concurrent typed
  /// ingress reservation defers the release instead of reclaiming storage still being written.
  [[nodiscard]] static bool normalize_direct_ingress_after_freshness_release(
      SubscriberState& subscriber, DeferredSampleReleases& releases) noexcept;

  [[nodiscard]] std::optional<TypedIngressReservation> reserve_typed_ingress_destination_locked(
      SubscriberState& subscriber) noexcept;

  [[nodiscard]] std::optional<TypedIngressReservation> reserve_typed_ingress_destination(
      SubscriberState& subscriber) noexcept;

  // A capture-less process-local successor is missing-record evidence, not a replacement payload.
  // Publish its staged identity without consuming it so ReuseLast can return the incumbent and
  // selector retirement can later reclaim this exact successor before ordered EOS advances.
  [[nodiscard]] bool staged_arrival_evidence_eligible(
      const SubscriberState& subscriber, const SampleSlot& slot,
      const TemporalSampleRecord& temporal, const bool exact_store_reservation) const noexcept {
    const InputPortState& input = *subscriber.input;
    const SampleSlot* const retained = input.reuse_slot.load(std::memory_order_acquire);
    return !subscriber.externally_driven_import && input.reuse_last && retained != nullptr &&
           &slot != retained && !slot.staged_arrival_evidence && !slot.event_emitted &&
           (!slot.typed_ingress_reserved || exact_store_reservation) && !slot.refill_reserved &&
           !slot.refill_prefix_reserved && !temporal.capture_time.valid() &&
           retained->subscriber == &subscriber && retained->occupied &&
           !retained->direct_ingress_unread && retained->reuse_retained &&
           !retained->active_selected && is_strictly_newer_sample(temporal, retained->temporal);
  }

  [[nodiscard]] expected<std::optional<std::uint64_t>, Error>
  reserve_staged_arrival_evidence_order_if_eligible(SubscriberState& subscriber, SampleSlot& slot,
                                                    const TemporalSampleRecord& temporal,
                                                    const bool exact_store_reservation) noexcept {
    if (!staged_arrival_evidence_eligible(subscriber, slot, temporal, exact_store_reservation)) {
      return std::nullopt;
    }
    auto arrival_order = reserve_arrival_order();
    if (!arrival_order) {
      return make_unexpected(std::move(arrival_order).error());
    }
    return std::optional<std::uint64_t>{*arrival_order};
  }

  [[nodiscard]] expected<bool, Error> publish_staged_arrival_evidence_if_eligible(
      SubscriberState& subscriber, SampleSlot& slot, const bool exact_store_reservation = false,
      std::optional<std::uint64_t> reserved_arrival_order = std::nullopt) noexcept {
    InputPortState& input = *subscriber.input;
    if (!slot.occupied || !slot.direct_ingress_unread ||
        !staged_arrival_evidence_eligible(
            subscriber, slot, slot.temporal, exact_store_reservation)) {
      return false;
    }
    if (!reserved_arrival_order) {
      if (exact_store_reservation) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      auto arrival_order = reserve_arrival_order();
      if (!arrival_order) {
        return make_unexpected(std::move(arrival_order).error());
      }
      reserved_arrival_order = *arrival_order;
    }
    slot.arrival_order = *reserved_arrival_order;
    slot.staged_arrival_evidence = true;
    static_cast<void>(input.publish_readiness_change());
    return true;
  }

  [[nodiscard]] std::optional<TypedIngressReservation> reserve_staged_newer_data_replacement(
      SubscriberState& subscriber,
      const std::optional<TemporalSampleRecord> retained_handoff = std::nullopt,
      const std::optional<channel_internal::StagedDataState> staged_data_snapshot =
          std::nullopt) noexcept {
    const auto staged_data =
        staged_data_snapshot.has_value() ? staged_data_snapshot : subscriber.staged_data_state();
    if (!staged_data || staged_data->sequence == 0U) {
      return std::nullopt;
    }
    const auto cache_lock =
        acquire_admission_cache_lock(subscriber, InputCacheAdmissionSite::kTransport);
    InputPortState& input = *subscriber.input;
    SampleSlot* destination = input.reuse_slot.load(std::memory_order_acquire);
    if (retained_handoff) {
      const auto found =
          std::find_if(subscriber.slots.begin(), subscriber.slots.end(), [&](const auto& slot) {
            return slot.occupied && !slot.direct_ingress_unread && slot.temporal_retained &&
                   slot.temporal == *retained_handoff;
          });
      destination = found == subscriber.slots.end() ? nullptr : &*found;
    }
    // A different subscriber's pointer may be published concurrently. Its
    // immutable identity can be inspected here, but none of its mutable slot
    // fields are protected by this lock.
    if (destination == nullptr || destination->subscriber != &subscriber) {
      return std::nullopt;
    }
    const bool replacing_reuse = destination->reuse_retained;
    if (!destination->occupied || destination->direct_ingress_unread ||
        (replacing_reuse && input.reuse_slot.load(std::memory_order_acquire) != destination) ||
        destination->active_selected || destination->retirement_selected ||
        destination->typed_ingress_reserved ||
        (!retained_handoff && (!replacing_reuse || destination->temporal_retained)) ||
        (destination->event_emitted && !destination->witness_retired) ||
        destination->temporal.source_sequence >= staged_data->sequence) {
      return std::nullopt;
    }
    // A controller transaction has released this exact retained claim. ReuseLast additionally
    // owns the private fallback pointer; an ordinary cache replacement leaves it unchanged.
    if (replacing_reuse) {
      destination->reuse_retained = false;
      input.reuse_slot.store(nullptr, std::memory_order_release);
    }
    destination->typed_ingress_reserved = true;
    const bool prior_temporal_retained = destination->temporal_retained;
    return TypedIngressReservation{subscriber,
                                   *destination,
                                   false,
                                   replacing_reuse,
                                   prior_temporal_retained,
                                   retained_handoff.has_value()};
  }

  static void observe_typed_ingress_for_test(SubscriberState& subscriber) noexcept {
    if (subscriber.typed_ingress_observer != nullptr) {
      subscriber.typed_ingress_observer(subscriber.typed_ingress_observer_state);
    }
  }

  /// @param freshness_refill Exact reserved record this promotion admits for a released claim.
  ///
  /// A promotion driven by the freshness reconcile is not speculative: the retained record was
  /// already given up on the promise that this one reaches the cache. Marking it is what lets a
  /// replacement of that record be told from the ordinary drop-oldest churn around it.
  [[nodiscard]] expected<std::uint32_t, Error> promote_direct_ingress(
      SubscriberState& subscriber, DeferredSampleReleases& releases,
      std::optional<std::uint64_t> first_arrival_order = std::nullopt,
      SampleSlot* freshness_refill = nullptr, std::int64_t* promoted_capture_ns = nullptr,
      SampleSlot* released_typed_capacity = nullptr) noexcept;

  [[nodiscard]] expected<void, Error> commit_typed_ingress_reservation(
      SubscriberState& subscriber, TypedIngressReservation& reservation,
      DeferredSampleReleases& releases) noexcept {
    const bool staged = reservation.staged();
    reservation.commit_locked();
    if (!staged) {
      return {};
    }
    auto promoted = promote_direct_ingress(subscriber, releases);
    if (!promoted) {
      return make_unexpected(std::move(promoted).error());
    }
    return {};
  }

  [[nodiscard]] expected<std::optional<std::uint64_t>, Error>
  reserve_immediate_direct_ingress_promotion(SubscriberState& subscriber,
                                             DeferredSampleReleases& releases) noexcept;

  [[nodiscard]] expected<void, Error> validate_and_translate_catalog_capture(
      const SubscriberState& subscriber, SampleMetadata& metadata,
      std::optional<StructuralDiagnosticId>* terminal_diagnostic) const noexcept;

  // Retention validates the complete metadata, clock, sequence, capacity, and claim contract
  // before committing one sample slot; the fail-closed branches intentionally remain explicit.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<void, Error> store_received(
      SubscriberState& subscriber, DeferredSampleReleases& releases, SampleMetadata received,
      EndpointClockHealthDisposition clock_health_disposition,
      std::uint64_t clock_health_generation, std::span<const std::byte> borrowed_payload = {},
      std::optional<StructuralDiagnosticId>* terminal_diagnostic = nullptr,
      SampleSlot* reserved_destination = nullptr, bool reserved_destination_staged = false,
      const void* borrowed_value = nullptr) noexcept;

  // Payload-family retention primitives do not publish owner progress or output accounting.
  // Successful Output publication must wrap them in commit_direct_publication().
  [[nodiscard]] expected<void, Error> retain_direct_inline_payload(
      SubscriberState& subscriber, std::span<const std::byte> payload,
      SampleMetadata metadata) noexcept;

  [[nodiscard]] expected<void, Error> store_direct_object_payload(SubscriberState& subscriber,
                                                                  DeferredSampleReleases& releases,
                                                                  const void* payload,
                                                                  SampleMetadata metadata) noexcept;

  // Keep receive admission, metadata fencing, and cache mutation ordered as one transaction.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<void, Error> store_received_tensor(
      SubscriberState& subscriber, DeferredSampleReleases& releases, Sample<Tensor> received,
      std::optional<StructuralDiagnosticId>* terminal_diagnostic = nullptr,
      SampleSlot* reserved_destination = nullptr, bool reserved_destination_staged = false,
      EndpointClockHealthDisposition clock_health_disposition =
          EndpointClockHealthDisposition::kNotApplicable,
      std::uint64_t clock_health_generation = 0U) noexcept;

  /// Whether a Tensor sample meets the parts of its frozen contract that one recipient adds:
  /// capture domain, planned storage, pass-through contract or layout, and logical requirements.
  /// A direct fanout checks every recipient before it stores into any.
  [[nodiscard]] static bool tensor_recipient_admits(const SubscriberState& subscriber,
                                                    const Tensor& tensor,
                                                    const TensorAccessSnapshot& snapshot,
                                                    const SampleMetadata& metadata) noexcept;
  /// The same for the Tensor fields of a descriptor-schema sample.
  [[nodiscard]] static bool tensor_schema_recipient_admits(const SubscriberState& subscriber,
                                                           std::span<const Tensor> tensors,
                                                           const SampleMetadata& metadata) noexcept;

  [[nodiscard]] expected<void, Error> retain_direct_tensor_sample(SubscriberState& subscriber,
                                                                  Sample<Tensor> received) noexcept;

  // Keep schema decoding, field validation, and cache publication in one receive transaction.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<void, Error> store_received_tensor_schema(
      SubscriberState& subscriber, DeferredSampleReleases& releases,
      std::span<const std::byte> encoded, std::span<Tensor> received, SampleMetadata metadata,
      SampleSlot* reserved_destination = nullptr, bool reserved_destination_staged = false,
      EndpointClockHealthDisposition clock_health_disposition =
          EndpointClockHealthDisposition::kNotApplicable,
      std::uint64_t clock_health_generation = 0U,
      std::optional<StructuralDiagnosticId>* terminal_diagnostic = nullptr) noexcept;

  [[nodiscard]] expected<void, Error> retain_direct_tensor_schema_sample(
      SubscriberState& subscriber, std::span<const std::byte> encoded,
      std::span<const Tensor> received) noexcept;

  [[nodiscard]] bool owner_progress_wake_enabled() const noexcept;

  /// Whether the session accepted a stop request (S3). Monotonic: restart and re-arm never clear
  /// it. Publication waits abort once it is set, and a pressure failure observed after it is set
  /// is attributed to the stop.
  [[nodiscard]] bool publication_stop_requested() const noexcept {
    return readiness_wake_.publication_stop != nullptr &&
           readiness_wake_.publication_stop->load(std::memory_order_acquire);
  }

  void signal_owner_progress(VertexId owner) noexcept;

  // Wake the subscriber's owner for an admitted data sample unless the input is side-only.
  void signal_owner_data_progress(const SubscriberState& subscriber) noexcept;

  // Wake the subscriber's owner for one receive: always for end of stream, else as for data.
  void signal_owner_receive_progress(const SubscriberState& subscriber,
                                     ChannelReceiveKind received) noexcept;

  [[nodiscard]] expected<void, Error> store_end_of_stream(
      SubscriberState& subscriber, std::uint64_t transport_sequence) noexcept;

  [[nodiscard]] expected<ChannelReceiveKind, Error> receive_tensor(
      SubscriberState& subscriber, ChannelSubscriber<Tensor>& endpoint,
      CudaDeviceCursor* scan_device, TypedIngressReservation* reservation,
      std::optional<StructuralDiagnosticId>* terminal_diagnostic) noexcept;

  [[nodiscard]] expected<void, Error> store_received_tensor_schema_payload(
      SubscriberState& subscriber, std::span<const std::byte> wire_payload,
      std::optional<SampleMetadata> metadata, TypedIngressReservation& reservation,
      CudaDeviceCursor* scan_device,
      EndpointClockHealthDisposition clock_health_disposition =
          EndpointClockHealthDisposition::kNotApplicable,
      std::uint64_t clock_health_generation = 0U,
      std::optional<StructuralDiagnosticId>* terminal_diagnostic = nullptr) noexcept;

  // What one receive admitted, captured inside its ordered transport transaction. Post-lock wake
  // and observation callbacks read this instead of mutable subscriber state, which a concurrent
  // receive may already have advanced (another data sample, or EOS).
  struct ReceiveReceipt {
    std::uint64_t data_sequence{};  ///< Admitted data publication sequence, or 0.
  };

  [[nodiscard]] expected<ChannelReceiveKind, Error> receive_one(
      SubscriberState& subscriber,
      std::optional<StructuralDiagnosticId>* terminal_diagnostic = nullptr,
      CudaDeviceCursor* scan_device = nullptr,
      std::optional<TemporalSampleRecord> reuse_handoff = std::nullopt,
      const channel_internal::StagedDataState* expected_staged_data = nullptr,
      ReceiveReceipt* receipt = nullptr) noexcept;

  // receive_one() for a caller that already holds subscriber.transport_mutex, for example to keep
  // a readiness probe and its receive in one transport transaction.
  [[nodiscard]] expected<ChannelReceiveKind, Error> receive_one_with_transport_held(
      SubscriberState& subscriber, std::optional<StructuralDiagnosticId>* terminal_diagnostic,
      CudaDeviceCursor* scan_device, std::optional<TemporalSampleRecord> reuse_handoff,
      const channel_internal::StagedDataState* expected_staged_data,
      ReceiveReceipt* receipt) noexcept;

  // Requires subscriber.transport_mutex.
  [[nodiscard]] expected<ChannelReceiveKind, Error> receive_one_locked(
      SubscriberState& subscriber, std::optional<StructuralDiagnosticId>* terminal_diagnostic,
      CudaDeviceCursor* scan_device, std::optional<TemporalSampleRecord> reuse_handoff,
      const channel_internal::StagedDataState* expected_staged_data) noexcept;

  // Catalog draining preserves ordered health/data heads and the finite per-dispatch budget.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<void, Error> drain_monitored_inputs(
      const ActivationOwnerState& owner) noexcept;

  // kInventory drains every pump-inventory subscriber (cuts, uncertain sends); kPumpScan the
  // pump's own subscribers; kRecipients an explicit list (emit admission, retirement retries).
  enum class DrainScope : std::uint8_t { kInventory, kPumpScan, kRecipients };

  [[nodiscard]] expected<bool, Error> drain_unmonitored_inputs(
      PortId& failed_port, VertexId& failed_owner,
      std::optional<StructuralDiagnosticId>& terminal_diagnostic, DrainScope scope,
      CudaDeviceCursor* scan_device = nullptr, bool activation_cut = false,
      std::span<SubscriberState* const> recipients = {}, bool skip_known_blocked = false) noexcept;

  // Prepare the calling lane, which is not the pump, to import Tensor descriptors for these
  // subscribers: initialize this thread's descriptor-import state once and, when a subscriber is
  // device-affine, open one cursor so its frozen device is applied and restored across the drain.
  // Inline-only sessions make no CUDA call. Returns false after failing the session.
  [[nodiscard]] bool prepare_lane_descriptor_import(
      std::span<SubscriberState* const> subscribers,
      std::optional<CudaDeviceCursor>& import_device) noexcept;

  // Private emit experiment: retain the scan/cache locks and probe a cold exact-recipient list.
  // Uncertain submissions explicitly select the complete inventory; an empty exact list is invalid.
  void materialize_experimental_publication(DrainScope scope,
                                            std::span<SubscriberState* const> recipients = {},
                                            std::uint64_t publication_sequence = 0U) noexcept;

  // Record test-control checkpoints around the owner wake for one receive (see
  // signal_owner_receive_progress()).
  void signal_received_progress(SubscriberState& subscriber, const ReceiveReceipt& receipt,
                                ChannelReceiveKind received) noexcept;

  // Receipts are captured only for private fanout attribution; ordinary receives skip the reads.
  [[nodiscard]] bool fanout_observation_enabled() const noexcept {
    return fanout_test_control_ != nullptr && fanout_test_control_->observe != nullptr;
  }

  [[nodiscard]] bool activation_cut_index_ready() const noexcept;

  // The accepted process-local publication already names every physical blocking subscriber and
  // HoloMQ makes each submitted head immediately visible. Drain those exact subscribers without
  // rescanning unrelated services or asking the backend whether the known head exists. Any typed
  // retention backpressure falls back to the complete inventory path and its autonomous retry.
  [[nodiscard]] expected<bool, Error> drain_indexed_activation_cut(
      PortId& failed_port, VertexId& failed_owner,
      std::optional<StructuralDiagnosticId>& terminal_diagnostic) noexcept;

  void rearm_activation_cut_index_if_drained() noexcept;

  // `released` names the subscribers whose typed capacity was released. An empty list with
  // `unknown_release` means capacity may have been released on any subscriber.
  [[nodiscard]] expected<void, Error> retry_blocked_process_local_transport(
      std::span<SubscriberState* const> released = {}, bool unknown_release = false) noexcept;

  // Concurrent admission: read the flag under the transport mutex that orders the outcomes it
  // records. A publication enqueued before this check is then either behind that blocked head or
  // observed by the retry that cleared the flag.
  [[nodiscard]] static bool transport_head_known_blocked(SubscriberState& subscriber) noexcept {
    const std::scoped_lock transport_lock{subscriber.transport_mutex};
    return subscriber.transport_head_blocked.load(std::memory_order_relaxed);
  }

  // Requires subscriber.transport_mutex, so each subscriber's flag follows its receive outcomes in
  // order. With concurrent admission the count of blocked heads is the only retry signal: a
  // separate latch derived from it could be re-set from a stale count observation by a concurrent
  // call on another subscriber, and nothing would clear it again.
  void set_transport_head_blocked(SubscriberState& subscriber, const bool blocked) noexcept {
    if (subscriber.transport_head_blocked.load(std::memory_order_relaxed) == blocked) {
      return;
    }
    subscriber.transport_head_blocked.store(blocked, std::memory_order_seq_cst);
    if (blocked) {
      blocked_head_count_.fetch_add(1U, std::memory_order_seq_cst);
    } else {
      blocked_head_count_.fetch_sub(1U, std::memory_order_seq_cst);
    }
  }

  // Attach the pump wait set's release wake and bind the readiness mask of every pump-served
  // subscriber whose head can block. Without the capability, or with the private A/B control,
  // every blocked head stays visible to the pump's wait.
  [[nodiscard]] expected<void, Error> bind_blocked_head_wake();

  // Install one subscriber's mask, cleared, on its current data attachment, as after a rebind.
  [[nodiscard]] expected<void, Error> bind_blocked_head_mask(SubscriberState& subscriber) noexcept;

  // Whether a plain receive that found the staged head refused should hide it: only when the pump's
  // wait reports samples queued behind it, so the pump would otherwise keep waking, and only after
  // the spin. Requires subscriber.transport_mutex.
  [[nodiscard]] bool blocked_head_hide_due(SubscriberState& subscriber) noexcept;

  // Hide a staged head that a plain receive found blocked. Returns true only when this call set
  // the mask, so its caller retries once: a release that freed capacity before the mask was
  // visible posted no wake.
  [[nodiscard]] bool mask_blocked_head(SubscriberState& subscriber) noexcept;

  // Mark the subscribers whose data attachments the pump's last wait reported ready, clearing the
  // previous wait's marks. Runs on the waiting thread.
  void mark_pump_reported_ready(std::span<const std::uint64_t> handles) noexcept;

  // Show again every head this release may have freed capacity for, and wake the pump once if any
  // was hidden.
  void wake_pump_for_blocked_heads(std::span<SubscriberState* const> released,
                                   bool unknown_release) noexcept;

  // Whether a process-local transport head may be waiting behind full typed retention. Every
  // retry, cut, and terminal-retirement decision reads this instead of either representation.
  [[nodiscard]] bool blocked_transport_retry_pending() const noexcept {
    return concurrent_admission_
               ? blocked_head_count_.load(std::memory_order_seq_cst) != 0U
               : blocked_transport_retry_required_.load(std::memory_order_acquire);
  }

  [[nodiscard]] expected<void, Error> retry_after_freshness_handoff() noexcept {
    if (fail_next_freshness_handoff_retry_.exchange(false, std::memory_order_acq_rel)) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    return retry_blocked_process_local_transport({}, true);
  }

  void record_process_local_publication(
      std::uint32_t deliveries,
      std::span<SubscriberState* const> activation_cut_subscribers = {}) noexcept;

  void record_process_local_receive() noexcept;

  void invalidate_process_local_progress() noexcept;

  void record_process_local_progress(std::atomic<std::uint64_t>& counter,
                                     std::uint32_t delta) noexcept;

  [[nodiscard]] expected<void, Error> wait_for_transport_activity() noexcept;

  [[nodiscard]] bool enter_pump_support_thread() noexcept;

  void freeze_descriptor_import_inventory();

  [[nodiscard]] bool initialize_descriptor_import(CudaDeviceCursor& device) noexcept;

  [[nodiscard]] bool initialize_pump_descriptor_import(
      std::optional<CudaDeviceCursor>& pump_device) noexcept;

  [[nodiscard]] bool initialize_controller_descriptor_import() noexcept;

  void pump() noexcept;

  void pump_with_thread_usage() noexcept;

  void fail(const Error& error, const VertexId owner = {},
            const std::optional<StructuralDiagnosticId> terminal_diagnostic = {}) noexcept {
    bool expected = false;
    const bool first = terminal_failure_claimed_.compare_exchange_strong(
        expected, true, std::memory_order_acq_rel, std::memory_order_acquire);
    if (first) {
      terminal_owner_.store(owner.value, std::memory_order_relaxed);
      terminal_diagnostic_.store(
          terminal_diagnostic ? static_cast<std::uint32_t>(*terminal_diagnostic) : 0U,
          std::memory_order_relaxed);
      // Publish the code only after the winner's bounded detail is complete. Readers acquire this
      // store before observing the reason/owner, while losing failures cannot overwrite either.
      terminal_error_.store(error.code, std::memory_order_release);
    }
    closed_.store(true, std::memory_order_release);
    cancel_requested_.store(true, std::memory_order_release);
    // A parked owner must not learn of the failure only at its next deadline: wake every owner.
    if (first && readiness_wake_.signal != nullptr) {
      readiness_wake_.signal(readiness_wake_.state);
    }
  }

  void fail(const ErrorCode code) noexcept { fail(Error{code}); }

  [[nodiscard]] expected<void, Error> status() const noexcept {
    const ErrorCode code = terminal_error_.load(std::memory_order_acquire);
    if (code == ErrorCode::kSuccess) {
      return {};
    }
    return make_unexpected(Error{code});
  }

  [[nodiscard]] expected<std::uint64_t, Error> reserve_arrival_order() noexcept {
    // The pump is the sole steady-state writer. Controller-side monitored drains execute only
    // while that pump is joined, so the atomic protects bounded test/control transitions rather
    // than authorizing parallel ingress. A future parallel drain needs prebound/sharded ordering,
    // not additional writers contending on this session-global sequence.
    std::uint64_t current = next_arrival_order_.load(std::memory_order_relaxed);
    for (;;) {
      if (current == std::numeric_limits<std::uint64_t>::max()) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      if (next_arrival_order_.compare_exchange_weak(
              current, current + 1U, std::memory_order_relaxed, std::memory_order_relaxed)) {
        return current;
      }
    }
  }

  template <typename Predicate>
  [[nodiscard]] expected<void, Error> reset_clock_health(const Predicate& affected) noexcept {
    const bool has_affected_health = std::ranges::any_of(subscribers_, [&](const auto& subscriber) {
      return affected(*subscriber) && static_cast<bool>(subscriber->clock_health_endpoint);
    });
    if (!has_affected_health) {
      return {};
    }
    if (clock_health_serialization_mutex_ == nullptr) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    if (clock_health_lock_observer_ != nullptr) {
      const auto observer = std::exchange(clock_health_lock_observer_, nullptr);
      void* const state = std::exchange(clock_health_lock_observer_state_, nullptr);
      observer(state);
    }
    // The Runtime joins ingress before reset. Serialize the complete preflight and commit against
    // report dispatch, expiry, and data snapshotting so the receiver binding cannot change after
    // validation and before reset_batch() publishes the next generation.
    const std::lock_guard clock_health_lock{*clock_health_serialization_mutex_};
    clock_health_reset_requests_.clear();
    EndpointClockHealthRuntimeCoordinator* coordinator = nullptr;
    for (const auto& subscriber : subscribers_) {
      if (!affected(*subscriber) || !subscriber->clock_health_endpoint) {
        continue;
      }
      if (subscriber->clock_health_coordinator == nullptr) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      if (coordinator == nullptr) {
        coordinator = subscriber->clock_health_coordinator;
      } else if (coordinator != subscriber->clock_health_coordinator) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      EndpointClockHealthReceiver* const receiver =
          coordinator->receiver(subscriber->clock_health_endpoint);
      if (receiver == nullptr ||
          receiver->state().binding.producer_epoch != subscriber->producer_epoch ||
          subscriber->producer_epoch == std::numeric_limits<std::uint64_t>::max()) {
        return make_unexpected(
            Error{receiver == nullptr ||
                          receiver->state().binding.producer_epoch != subscriber->producer_epoch
                      ? ErrorCode::kProtocolError
                      : ErrorCode::kResourceExhausted});
      }
      EndpointClockHealthBindingIdentity next = receiver->state().binding;
      ++next.producer_epoch;
      clock_health_reset_requests_.push_back(EndpointClockHealthResetRequest{
          .endpoint = subscriber->clock_health_endpoint,
          .origin = EndpointClockHealthUpdateOrigin::kProducerEpochReset,
          .binding = next,
      });
    }
    if (clock_health_reset_requests_.empty()) {
      return {};
    }
    auto reset = coordinator->reset_batch(clock_health_reset_requests_);
    if (!reset) {
      const ErrorCode code = reset.error().code == ErrorCode::kNotReady ? ErrorCode::kBackpressured
                                                                        : reset.error().code;
      return make_unexpected(Error{code});
    }
    return {};
  }

  /// Gate one externally driven import for a generation-fenced rebind.
  ///
  /// Caller holds the subscriber mutex with the pump joined. The gate stays kBackpressured while
  /// pending transport of the current latched incarnation exists or a live publisher is present
  /// without proof that the only observed traffic already belongs to a new incarnation. Advisory
  /// absence alone never fabricates a replacement generation: the cutover is admitted only
  /// because the wire generation fence rejects post-cut old-incarnation frames typed. A
  /// non-restart-capable import keeps failing closed with kNotSupported, and exceeding the
  /// plan-admitted retained-generation capacity fails with kResourceExhausted before any
  /// destructive step (no eviction, wait, or growth).
  [[nodiscard]] expected<void, Error> import_rebind_gate(SubscriberState& subscriber) noexcept;

  /// Atomically cut one gated import over to a replacement catalog attachment.
  ///
  /// Caller holds the subscriber mutex with the pump joined and every preflight (gate, capacity,
  /// epoch exhaustion, clock-health reset) already succeeded. The endpoint services admit one
  /// subscriber, so the old attachment must be detached and dropped before the replacement is
  /// created; any failure after that first destructive step cannot be rolled back and
  /// terminalizes this session, mirroring commit_restart's discard semantics. The replacement
  /// generation slot is consumed only after the cut completes.
  [[nodiscard]] expected<void, Error> rebind_import(SubscriberState& subscriber) noexcept;

  void cancel() noexcept;

  [[nodiscard]] expected<void, Error> join() noexcept;

  /// Snapshot only at a committed lifecycle boundary with producers and the pump quiescent.
  [[nodiscard]] std::uint64_t cutover_sequence_floor(
      const SubscriberState& subscriber) const noexcept;

  // Return this session's HoloMQ name for one process-local service identity.
  [[nodiscard]] std::string process_local_transport_name(std::string_view service_id) const;

  // Format a session scope from the namespace-local process id, the per-process session ordinal,
  // and a process-incarnation token that keeps equal ids in separate PID namespaces distinct.
  [[nodiscard]] static std::string process_local_transport_scope(std::uint64_t process,
                                                                 std::uint64_t ordinal,
                                                                 std::uint64_t incarnation);

  std::optional<ChannelRuntime> runtime_;
  // Session-unique suffix for process-local HoloMQ service names; see materialization.
  std::string process_local_transport_scope_;
  PlanGenerationId plan_generation_{};
  RunGenerationId run_generation_{};
  std::vector<ClockConversionRecord> clock_conversions_;
  // Borrowed from the retained ExecutionPlanState, which outlives every RunSession channel.
  std::span<const SupportThreadAffinityPlanRecord> support_thread_affinities_;
  // Borrowed from the RunSession clock registry generation. Invalid until RunSession binds it,
  // which keeps hand-authored test plans publishing canonical absent publication time.
  // Read at every publication; the session is destroyed before its registry generation.
  RetainedClock publish_clock_;
  // Catalog-namespace identity of publish_clock_, resolved once from the plan.
  ClockId catalog_publish_clock_{};
  // Plan-namespace identity the catalog record above was projected for.
  ClockId plan_publish_clock_{};
  RuntimeSupportThreadPlacementFailure support_thread_placement_failure_;
  std::shared_ptr<RuntimeSupportThreadControl> support_thread_control_;
  std::shared_ptr<RuntimeChannelIncompletePollTestControl> incomplete_poll_test_control_;
  // Per-acquisition budget of the bounded contention retry: kContentionRetryBudget unless a
  // source-tree test widens it before the session is armed.
  std::chrono::nanoseconds contention_retry_budget_{kContentionRetryBudget};
  bool force_process_local_channel_pump_{};
  // Some process-local service is emit-admitted. Admission then runs concurrently, without the
  // scan lock, on producer lanes, beside any pump that serves the other services.
  bool emit_admission_{};
  const RuntimeFanoutTestControl* fanout_test_control_{};
  bool track_concurrent_process_local_publications_{};
  RuntimePhaseCounters* phase_counters_{};
  RuntimeChannelReadinessWake readiness_wake_{};
  std::optional<std::size_t> fixed_tensor_shape_allocations_before_failure_for_test_;
  std::unique_ptr<channel_internal::InlinePayloadWaitSet> wait_set_;
  // wait_set_'s blocked-head capability once its wake is attached; null leaves every blocked head
  // visible to the pump's wait.
  channel_internal::BlockedHeadWaitSet* blocked_head_wait_set_{};
  std::vector<std::unique_ptr<InputPortState>> inputs_;
  std::vector<InputPortState*> input_by_port_;
  // PortBinding instances are owned by RunSession (or a test caller), not by this session. Track
  // their borrowed back-pointers so session teardown can detach them before InputPortState dies.
  std::vector<ReadinessBinding*> readiness_bindings_;
  std::vector<std::unique_ptr<SubscriberState>> subscribers_;
  std::vector<DeviceId> descriptor_import_devices_;
  std::unordered_map<std::uint64_t, SubscriberState*> wait_subscriber_by_handle_;
  std::vector<SubscriberState*> pump_subscribers_;
  // pump_subscribers_ without emit-admitted recipients: what the background pump scans. Cuts and
  // uncertain sends still drain the complete pump_subscribers_ inventory.
  std::vector<SubscriberState*> pump_scan_subscribers_;
  std::vector<SubscriberState*> activation_cut_pending_subscribers_;
  std::vector<std::unique_ptr<OutputState>> outputs_;
  std::vector<ActivationOwnerState> activation_owners_;
  // Null for the overwhelmingly common unmonitored session. Monitored report dispatch, state
  // snapshotting, expiry, and reset share this one externally owned serialization boundary.
  std::mutex* clock_health_serialization_mutex_{};
  std::vector<EndpointClockHealthResetRequest> clock_health_reset_requests_;
  std::vector<std::uint64_t> ready_handles_;
  // Subscribers marked by mark_pump_reported_ready(); reserved for every wait attachment.
  std::vector<SubscriberState*> pump_reported_ready_;
  std::chrono::nanoseconds blocked_head_spin_{kBlockedHeadSpin};
  std::size_t wait_attachment_count_{};
  // Timed so that the post-compute cut can wait for it in bounded slices (lock_unless_settled).
  std::timed_mutex pump_scan_mutex_;
  // Set by transported publications that no owner's activation cut will settle; any cut that
  // sees it takes the full path.
  std::atomic<bool> transport_settle_fallback_{false};
  std::binary_semaphore pump_startup_{0};
  std::thread pump_thread_;
  // Identifies the frozen descriptor-import inventory. Threads that have initialized it remember
  // this token, so a changed inventory is initialized again on every thread.
  std::uint64_t descriptor_import_token_{};
  bool descriptor_import_present_{};
  std::atomic<std::uint64_t> controller_descriptor_import_initialization_count_{};
  std::atomic<bool> armed_{};
  std::atomic<bool> cancel_requested_{};
  std::atomic<bool> closed_{};
  std::atomic<ErrorCode> terminal_error_{ErrorCode::kSuccess};
  std::atomic_bool terminal_failure_claimed_{false};
  std::atomic<std::uint32_t> terminal_diagnostic_{};
  std::atomic<std::uint64_t> terminal_owner_{0U};
  std::atomic<std::uint64_t> next_arrival_order_{1U};
  std::atomic<std::uint64_t> process_local_published_deliveries_{};
  std::atomic<std::uint64_t> process_local_received_deliveries_{};
  // Deliveries that HoloMQ replaced in a drop-oldest subscriber queue before any drain received
  // them. Each was counted as published and will never be received.
  std::atomic<std::uint64_t> process_local_replaced_deliveries_{};
  // Deliveries that a restart discarded from the old generation's transport. Each was counted as
  // published and will never be received.
  std::atomic<std::uint64_t> process_local_discarded_deliveries_{};
  std::atomic<std::uint64_t> process_local_publications_in_flight_{};
  std::atomic<bool> process_local_progress_tracking_valid_{true};
  // Retry latch for sessions without concurrent admission; read through
  // blocked_transport_retry_pending().
  std::atomic<bool> blocked_transport_retry_required_{};
  // Private concurrent-admission state; see RuntimeFanoutTestControl::concurrent_admission and
  // retry_blocked_process_local_transport(). The scratch list is guarded by pump_scan_mutex_ and
  // reserved for the whole pump inventory when armed.
  bool concurrent_admission_{};
  // Check transport credit at every blocking process-local HoloMQ recipient before submission.
  // Off for plans with restart authority and under the private disable control.
  bool publication_credits_{};
  std::atomic<std::size_t> blocked_head_count_{};
  std::vector<SubscriberState*> scoped_retry_subscribers_;
  std::atomic<bool> fail_next_freshness_handoff_retry_{};
  bool activation_cut_index_available_{};
  bool activation_cut_index_enabled_{};
  bool activation_cut_index_current_{true};
#if !defined(NDEBUG)
  std::shared_mutex controller_mutation_barrier_;
  std::unique_ptr<std::atomic_flag[]> owner_mutation_active_;
#endif
  ErrorCode restart_post_discard_failure_for_test_{ErrorCode::kSuccess};
  void* activation_retirement_observer_state_{};
  RuntimeChannelActivationRetirementObserver activation_retirement_observer_{};
  std::atomic<bool> failed_callback_retirement_fault_{};
  void* teardown_observer_state_{};
  RuntimeChannelTeardownObserver teardown_observer_{};
  void* poll_selection_observer_state_{};
  RuntimeChannelPollSelectionObserver poll_selection_observer_{};
  void* clock_health_lock_observer_state_{};
  RuntimeChannelClockHealthLockObserver clock_health_lock_observer_{};
  // Cold ownership stays after ordinary fields to preserve their offsets. ~Impl explicitly
  // clears outputs_ while this owner is still alive; quarantine retains the complete Impl.
  std::shared_ptr<RuntimeFixtureObservation> fixture_observation_;
};

}  // namespace holoscan::detail
