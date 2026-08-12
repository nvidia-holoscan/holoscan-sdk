/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include <holoscan/core/activation.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/time/time_point.hpp>

#include "core/tag_internal.hpp"
#include "port_binding.hpp"
#include "scheduling/ready_policy.hpp"
#include "temporal_input_cache.hpp"
#include "temporal_preselection.hpp"

namespace holoscan::detail {

class EndpointClockHealthReceiver;

/// Execute one compiler-frozen input selector against bounded runtime history.
///
/// This private state is the payload-independent half of direct Runtime input
/// selection. It consumes the immutable plan record verbatim, owns only copied
/// temporal metadata, and leaves typed Sample/Channel claims to EA1-012.
class RuntimeInputSelectionState {
 public:
  class Transaction {
   public:
    Transaction() = default;
    Transaction(const Transaction&) = delete;
    Transaction& operator=(const Transaction&) = delete;
    Transaction(Transaction&& other) noexcept;
    Transaction& operator=(Transaction&&) = delete;
    ~Transaction();

    [[nodiscard]] InputSelection selection() const noexcept { return selection_; }
    [[nodiscard]] std::span<const TemporalSampleRecord> selected() const noexcept;

    /// Return the selector's semantic reference coordinate when one exists.
    ///
    /// Bracketed windows retain the current anchor as their semantic
    /// reference even though a later successor/timeout witness closes the
    /// candidate. Other selectors use the source candidate reference.
    [[nodiscard]] TimePoint semantic_reference_time() const noexcept;

    /// Retire the frozen decision and commit destructive work only when accepted.
    [[nodiscard]] expected<void, Error> finish(bool activation_accepted) noexcept;

   private:
    explicit Transaction(RuntimeInputSelectionState* owner) noexcept : owner_(owner) {}

    RuntimeInputSelectionState* owner_{};
    std::optional<TemporalInputCache::ScalarTransaction> scalar_;
    std::optional<TemporalInputCache::DrainTransaction> drain_;
    std::optional<TemporalSampleRecord> scalar_sample_;
    InputSelection selection_{};
    bool bracketed_{};

    friend class RuntimeInputSelectionState;
  };

  RuntimeInputSelectionState(const RuntimeInputSelectionState&) = delete;
  RuntimeInputSelectionState& operator=(const RuntimeInputSelectionState&) = delete;
  RuntimeInputSelectionState(RuntimeInputSelectionState&&) noexcept = default;
  RuntimeInputSelectionState& operator=(RuntimeInputSelectionState&&) noexcept = default;
  ~RuntimeInputSelectionState() = default;

  [[nodiscard]] static expected<RuntimeInputSelectionState, Error> create(
      const InputSelectionPlanRecord& selection,
      const BracketedWindowPlanRecord* bracketed_window = nullptr,
      std::uint32_t max_pending_windows = 0U);

  /// Retain one input sample under the selector's admitted finite capacity.
  [[nodiscard]] expected<void, Error> push(TemporalSampleRecord sample) noexcept;

  /// Advance the anchor state of a bracketed-window selector.
  [[nodiscard]] expected<WindowAnchorResult, Error> push_anchor(
      TemporalSampleRecord anchor) noexcept;

  /// Close the oldest bracketed window from one admitted terminal fact.
  [[nodiscard]] expected<void, Error> close_oldest(SelectionReason reason) noexcept;

  /// @brief Bind the current generation's prevalidated LatestWithin age.
  /// @param max_age Positive canonical duration from the same activation Parameter pin.
  /// @return Success, or an error for a non-LatestWithin selector or active transaction.
  [[nodiscard]] expected<void, Error> set_latest_within_max_age(
      std::chrono::nanoseconds max_age) noexcept;

  /// Close this input generation with an explicit transport EOS witness.
  ///
  /// Scalar and drain selectors retain the closure reason for their next empty decision.
  /// Bracketed selectors close their oldest pending window when present and otherwise accept the
  /// empty pending set. RunSession stops forwarding later anchors to the terminal selector.
  [[nodiscard]] expected<void, Error> close_end_of_stream() noexcept;

  /// Freeze one selector decision at the candidate's immutable reference time.
  [[nodiscard]] expected<Transaction, Error> begin(TimePoint reference_time,
                                                   std::uint64_t arrival_watermark,
                                                   std::uint64_t producer_epoch) noexcept;

  /// Clear history at a run/restart/rebind/epoch/discontinuity boundary.
  [[nodiscard]] expected<void, Error> reset() noexcept;

  [[nodiscard]] const InputSelectionPlanRecord& plan() const noexcept { return plan_; }
  [[nodiscard]] std::size_t retained_count() const noexcept;

  /// Return the preallocated scratch bound for typed payload claims.
  [[nodiscard]] std::size_t payload_claim_capacity() const noexcept;

  /// Copy records whose typed payloads remain observable by this selector.
  ///
  /// The returned prefix can contain duplicates for overlapping closed
  /// windows. Runtime sorts and deduplicates it before retiring Channel slots.
  [[nodiscard]] expected<std::size_t, Error> copy_payload_claims(
      std::span<TemporalSampleRecord> output) const noexcept;

  /// Return the number of bracketed windows awaiting a terminal close.
  [[nodiscard]] std::size_t pending_window_count() const noexcept;

  /// Return the oldest pending window's semantic anchor, when present.
  [[nodiscard]] std::optional<TimePoint> oldest_pending_window_reference() const noexcept;

 private:
  RuntimeInputSelectionState(InputSelectionPlanRecord plan,
                             std::optional<BracketedWindowEngine> bracketed);

  [[nodiscard]] static bool valid_plan(const InputSelectionPlanRecord& selection,
                                       const BracketedWindowPlanRecord* bracketed_window,
                                       std::uint32_t max_pending_windows) noexcept;
  [[nodiscard]] ScalarSelectionResult apply_missing(ScalarSelectionResult result,
                                                    std::uint64_t producer_epoch,
                                                    TimePoint reference_time) const noexcept;
  void release(Transaction& transaction) noexcept;

  InputSelectionPlanRecord plan_;
  TemporalInputCache cache_;
  std::optional<BracketedWindowEngine> bracketed_;
  std::optional<TemporalSampleRecord> retained_;
  std::chrono::nanoseconds latest_within_max_age_{};
  std::optional<SelectionReason> closure_reason_;
  bool transaction_active_{};
};

/// Execute one compiler-frozen oldest-head aligned join.
///
/// This private coordinator preserves the exact plan member order while
/// delegating deterministic FIFO-head matching and retirement to
/// \ref AlignedJoinEngine. Samples passed to \ref push must already be
/// normalized into the plan's comparison clock by the admitted direct
/// conversion. Typed Channel claims remain owned by the Runtime Channel
/// session and are joined to these payload-independent records by source
/// identity.
class RuntimeAlignedSelectionState {
 public:
  class Transaction {
   public:
    Transaction() = default;
    Transaction(const Transaction&) = delete;
    Transaction& operator=(const Transaction&) = delete;
    Transaction(Transaction&& other) noexcept;
    Transaction& operator=(Transaction&&) = delete;
    ~Transaction();

    [[nodiscard]] AlignedCandidateDisposition disposition() const noexcept;
    [[nodiscard]] TimePoint reference_time() const noexcept;
    [[nodiscard]] SelectionReason reason() const noexcept;

    /// Return one selection state per member in immutable-plan order.
    [[nodiscard]] std::span<const InputSelection> selections() const noexcept;

    /// Return the selected/reused member records frozen by the engine.
    [[nodiscard]] std::span<const TemporalSampleRecord> selected() const noexcept;

    /// Complete this decision.
    ///
    /// Nominal and degraded heads are consumed only for an accepted
    /// activation. A deterministic window miss always retires every minimum
    /// head even though it does not invoke compute. Other no-fire decisions
    /// abort without consuming provisional heads.
    [[nodiscard]] expected<void, Error> finish(bool activation_accepted) noexcept;

   private:
    Transaction(RuntimeAlignedSelectionState* owner,
                AlignedJoinEngine::Transaction transaction) noexcept;

    RuntimeAlignedSelectionState* owner_{};
    std::optional<AlignedJoinEngine::Transaction> transaction_;

    friend class RuntimeAlignedSelectionState;
  };

  RuntimeAlignedSelectionState(const RuntimeAlignedSelectionState&) = delete;
  RuntimeAlignedSelectionState& operator=(const RuntimeAlignedSelectionState&) = delete;
  RuntimeAlignedSelectionState(RuntimeAlignedSelectionState&&) noexcept = default;
  RuntimeAlignedSelectionState& operator=(RuntimeAlignedSelectionState&&) noexcept = default;
  ~RuntimeAlignedSelectionState() = default;

  /// Validate and materialize one immutable aligned-trigger record.
  [[nodiscard]] static expected<RuntimeAlignedSelectionState, Error> create(
      const AlignedTriggerPlanRecord& aligned,
      std::span<const InputSelectionPlanRecord> selections);

  /// Retain one already-normalized member sample under its exact plan bound.
  [[nodiscard]] expected<void, Error> push(PortId input, TemporalSampleRecord sample) noexcept;

  /// Mark one member generation closed by an explicit EOS witness.
  [[nodiscard]] expected<void, Error> close_end_of_stream(PortId input,
                                                          TimePoint closure_time) noexcept;

  /// Freeze the deterministic oldest-head decision.
  ///
  /// An empty closure span means every member remains open. A nonempty span
  /// must contain exactly one proof per member in immutable-plan order.
  [[nodiscard]] expected<Transaction, Error> begin(
      std::span<const AlignmentClosureProof> closure = {}, TimePoint closure_time = {}) noexcept;

  /// Clear all retained heads and fallback state at a generation boundary.
  [[nodiscard]] expected<void, Error> reset() noexcept;

  [[nodiscard]] const AlignedTriggerPlanRecord& plan() const noexcept { return plan_; }
  [[nodiscard]] std::span<const PortId> member_inputs() const noexcept { return member_inputs_; }
  [[nodiscard]] std::size_t queued(PortId input) const noexcept;

  /// Return the preallocated scratch bound for one member's typed claims.
  [[nodiscard]] std::size_t payload_claim_capacity(PortId input) const noexcept;

  /// Copy queued and finite-reuse records that can still select one member.
  [[nodiscard]] expected<std::size_t, Error> copy_payload_claims(
      PortId input, std::span<TemporalSampleRecord> output) const noexcept;

 private:
  RuntimeAlignedSelectionState(AlignedTriggerPlanRecord plan, std::vector<PortId> member_inputs,
                               std::vector<std::uint32_t> member_capacities,
                               std::vector<AlignedMissingPolicy> missing,
                               std::size_t engine_capacity);

  [[nodiscard]] std::size_t member_index(PortId input) const noexcept;
  void release(Transaction& transaction) noexcept;

  AlignedTriggerPlanRecord plan_;
  std::vector<PortId> member_inputs_;
  std::vector<std::uint32_t> member_capacities_;
  std::vector<AlignedMissingPolicy> missing_;
  std::vector<AlignmentClosureProof> open_closure_;
  TimePoint closure_time_{};
  AlignedJoinEngine engine_;
  bool transaction_active_{};
};

/// Apply one compiler-frozen latest-start rule at the final invocation gate.
[[nodiscard]] expected<DeadlineGateResult, Error> evaluate_plan_deadline(
    const DeadlinePlanRecord& plan, TimePoint source_coordinate,
    TimePoint dispatch_coordinate) noexcept;

/// Freeze one latest-start rule together with its exact admitted clock conversion.
///
/// This private record prevents Runtime from re-looking-up or re-inferring a
/// conversion after cold startup. A missing \ref source_to_reference denotes
/// a same-clock rule and therefore requires \ref source_clock and
/// \ref reference_clock to match.
struct RuntimeDeadlineBindingRecord {
  const DeadlinePlanRecord* deadline{};  ///< Borrowed from the immutable session plan.
  PlanGenerationId plan_generation{};
  ClockId source_clock{};
  ClockId reference_clock{};
  std::uint64_t conversion_id{};
  ClockRoundingMode rounding{ClockRoundingMode::kNearestTiesToEven};
  ClockOverflowPolicy overflow{ClockOverflowPolicy::kError};
  std::optional<DirectAffineClockEdge> source_to_reference;
};

/// Validate and bind the unique deadline owned by one temporal trigger.
[[nodiscard]] expected<std::optional<RuntimeDeadlineBindingRecord>, Error> bind_runtime_deadline(
    std::span<const DeadlinePlanRecord> deadlines, const TemporalTriggerPlanRecord& trigger,
    std::span<const ClockConversionRecord> conversions, PlanGenerationId plan_generation,
    const PredicateTriggerPlanRecord* predicate = nullptr) noexcept;

/// Validate that every frozen deadline is consumed exactly once by its owning trigger.
[[nodiscard]] expected<void, Error> validate_runtime_deadline_inventory(
    std::span<const DeadlinePlanRecord> deadlines,
    std::span<const TemporalTriggerPlanRecord> triggers,
    std::span<const PredicateTriggerPlanRecord> predicates,
    std::span<const ClockConversionRecord> conversions, PlanGenerationId plan_generation) noexcept;

/// Apply one bound latest-start rule using the event's two frozen coordinates.
///
/// The budget is added in the source domain. The resulting latest source
/// coordinate is compared exactly in the trigger reference domain without
/// materializing a rounded intermediate value.
[[nodiscard]] expected<DeadlineGateResult, Error> evaluate_runtime_deadline(
    const RuntimeDeadlineBindingRecord& binding, TimePoint source_decision_time,
    TimePoint reference_time, TimePoint dispatch_reference_time) noexcept;

/// Apply one bound latest-start rule with a generation-pinned budget.
[[nodiscard]] expected<DeadlineGateResult, Error> evaluate_runtime_deadline(
    const RuntimeDeadlineBindingRecord& binding, TimePoint source_decision_time,
    TimePoint reference_time, TimePoint dispatch_reference_time,
    std::chrono::nanoseconds budget) noexcept;

/// Preserve an earlier degraded activation unless the deadline itself is violated.
///
/// A nominal latest-start decision is not a recovery decision and therefore
/// cannot erase a SafeDefault/invariant status already frozen by selection or
/// predicate processing. A deadline violation remains the final invocation
/// action and supplies its own status.
[[nodiscard]] ActivationStatus merge_deadline_activation_status(
    ActivationStatus current, const DeadlineGateResult& deadline) noexcept;

/// Associate one already-frozen READY candidate with its runtime receiver.
struct DispatcherCandidate {
  scheduling::ReadyCandidateView ready;
  Tag tag;
  std::size_t receiver_index{};
  EndpointClockHealthDisposition clock_health_disposition{
      EndpointClockHealthDisposition::kNotApplicable};
};

/// Identify one candidate selected under the process-local serial token.
struct DispatcherSelection {
  std::size_t slot{};
  std::size_t receiver_index{};
  scheduling::ReadyCandidateView ready;
  Tag tag;
  EndpointClockHealthDisposition clock_health_disposition{
      EndpointClockHealthDisposition::kNotApplicable};
};

/// Bind one immutable-plan readiness source to its Dispatcher receiver.
///
/// This private record is materialized before execution. All identities and
/// the reference clock must come from one validated immutable plan; numeric
/// identity equality alone does not establish that provenance.
struct RuntimeSourceBindingRecord {
  RuntimeEventKind candidate_kind{RuntimeEventKind::kInputArrival};
  SourceId source;
  PortId port;
  ClockId source_clock;
  ClockId reference_clock;
  std::uint64_t source_to_reference_conversion_id{};
  std::uint32_t source_order{};
  std::uint32_t capacity{};
  std::uint64_t source_generation{1U};
  std::uint32_t tag_generation{1U};
  std::size_t receiver_index{};
  ExecutionPartitionId partition;
  std::uint32_t execution_lane{};
  VertexId vertex;
  std::uint32_t plan_vertex_order{};
  std::uint64_t stable_candidate_tie{};
  std::optional<RuntimeDeadlineBindingRecord> deadline;
  /// Borrow the preallocated receiver for this timing-sensitive source, or null.
  ///
  /// The receiver must outlive the coordinator and is mutated only under the
  /// same Dispatcher serialization domain. External attachment ownership is
  /// supplied by EA1-016; this pointer is only the transport-neutral Runtime
  /// eligibility hook.
  EndpointClockHealthReceiver* clock_health{};
};

/// Preserve the exact source witness and candidate identity frozen together.
struct FrozenRuntimeCandidate {
  RuntimeEvent witness;
  DispatcherCandidate candidate;
  std::size_t source_slot{};
  std::uint64_t clock_health_generation{};
  bool source_retired{};
};

/// Drain bounded source events and form deterministic activation candidates.
///
/// Witness acceptance and candidate formation are deliberately separate.
/// `ingest()` validates and retains source stamps and reference time without
/// assigning a `Tag`. After every binding in the current polling round has
/// been drained, repeated `freeze_next()` calls choose the deterministic
/// minimum witness, reserve the receiver's `Tag`, and assign `TagTime`.
/// Rejected and no-fire candidates must still be passed to `retire()` so their
/// source credit is released without reusing the consumed `Tag`.
class RuntimeCandidateCoordinator {
 public:
  RuntimeCandidateCoordinator(const RuntimeCandidateCoordinator&) = delete;
  RuntimeCandidateCoordinator& operator=(const RuntimeCandidateCoordinator&) = delete;
  RuntimeCandidateCoordinator(RuntimeCandidateCoordinator&&) noexcept = default;
  RuntimeCandidateCoordinator& operator=(RuntimeCandidateCoordinator&&) noexcept = default;
  ~RuntimeCandidateCoordinator() = default;

  [[nodiscard]] static expected<RuntimeCandidateCoordinator, Error> create(
      std::span<const RuntimeSourceBindingRecord> sources, std::size_t receiver_count,
      std::size_t max_outstanding);
  [[nodiscard]] static expected<RuntimeCandidateCoordinator, Error> create_with_tags(
      std::span<const RuntimeSourceBindingRecord> sources,
      std::span<TagAllocator* const> receiver_tags, std::size_t max_outstanding);

  /// Drain one bounded ingress prefix without allocating or invoking user code.
  [[nodiscard]] expected<std::size_t, Error> ingest(PortBinding& binding) noexcept;

  /// Drain at most the already-inspected source-local credit from one binding.
  ///
  /// The caller obtains `source_credit` from `source_credit_remaining()` immediately before
  /// polling the same binding. The Dispatcher serialization domain keeps that snapshot stable
  /// until this call. Bounding the destructive drain prevents a queued successor from being
  /// removed while an earlier witness from the same source still owns the remaining credit.
  [[nodiscard]] expected<std::size_t, Error> ingest(PortBinding& binding, SourceId source,
                                                    std::size_t source_credit) noexcept;

  /// Form the deterministic minimum pending candidate.
  ///
  /// The caller must first drain every binding participating in the current
  /// freeze round. This whole-round precondition ensures adapter polling order
  /// cannot affect equal-time microsteps.
  [[nodiscard]] expected<FrozenRuntimeCandidate, Error> freeze_next() noexcept;

  /// Freeze one compiler-proven causal successor at the next parent microstep.
  ///
  /// The caller must prove that the expected source is the sole newly ingested
  /// witness, that no competing root is due, and that no other source can
  /// become visible in the closed chain. Failure leaves no alternate
  /// scheduling authority: once ingress is destructively accepted, a
  /// mismatch is terminal rather than an ordinary whole-round fallback.
  [[nodiscard]] expected<FrozenRuntimeCandidate, Error> freeze_causal_next(SourceId expected_source,
                                                                           TagTime parent) noexcept;

  /// Release source credit after no-fire or complete activation retirement.
  [[nodiscard]] expected<void, Error> retire(const FrozenRuntimeCandidate& candidate) noexcept;

  /// Publish one receiver's successful lifecycle restart.
  [[nodiscard]] expected<void, Error> publish_restart(std::size_t receiver_index) noexcept;

  /// Remove one receiver's pending witnesses before publishing a restart generation.
  ///
  /// Frozen witnesses must already be retired. The caller receives every removed event so its
  /// physical binding credit can be retired exactly once. Insufficient destination capacity or a
  /// still-frozen witness fails without mutation.
  [[nodiscard]] expected<std::size_t, Error> cancel_receiver(
      std::size_t receiver_index, std::span<RuntimeEvent> canceled) noexcept;

  /// Return the currently published generation for one frozen source identity.
  [[nodiscard]] expected<std::uint64_t, Error> source_generation(SourceId source) const noexcept;

  /// Return the receiver index bound to one frozen source identity.
  [[nodiscard]] expected<std::size_t, Error> source_receiver(SourceId source) const noexcept;

  /// Return how many additional drained witnesses one source can accept.
  ///
  /// The caller uses this preflight before polling or draining a physical binding. Zero
  /// is ordinary bounded pressure: an older witness from the same source must be retired before
  /// the binding is observed again. This prevents removing a successor from binding-owned ingress
  /// only to reject it against the source-local admitted capacity.
  [[nodiscard]] expected<std::size_t, Error> source_credit_remaining(
      SourceId source) const noexcept;

  /// Release every pending/frozen source credit during terminal teardown.
  void cancel_all() noexcept;

  [[nodiscard]] std::size_t pending_count() const noexcept { return pending_.size(); }
  [[nodiscard]] std::size_t outstanding_count() const noexcept { return outstanding_count_; }
  [[nodiscard]] std::size_t max_outstanding() const noexcept { return max_outstanding_; }

 private:
  struct SourceState {
    RuntimeSourceBindingRecord binding;
    std::uint64_t last_source_sequence{};
    std::uint32_t outstanding{};
    bool closed{};
    std::vector<std::uint64_t> frozen_sequences;
  };

  struct PendingEvent {
    RuntimeEvent event;
    std::size_t source_slot{};
  };

  RuntimeCandidateCoordinator(std::vector<SourceState> sources,
                              std::vector<std::unique_ptr<TagAllocator>> owned_tags,
                              std::size_t max_outstanding);

  [[nodiscard]] expected<void, Error> accept_event(RuntimeEvent event) noexcept;
  [[nodiscard]] expected<FrozenRuntimeCandidate, Error> freeze_pending(std::size_t pending_index,
                                                                       TagTime tag_time) noexcept;
  [[nodiscard]] std::size_t find_source(SourceId source) const noexcept;

  std::vector<SourceState> sources_;
  std::vector<std::unique_ptr<TagAllocator>> owned_tags_;
  std::vector<TagAllocator*> tags_;
  std::vector<RuntimeEvent> ingress_scratch_;
  std::vector<PendingEvent> pending_;
  std::size_t max_outstanding_{};
  std::size_t outstanding_count_{};
  std::int64_t last_logical_ns_{};
  std::uint32_t last_microstep_{};
  bool tag_time_initialized_{};
};

/// Borrow one activation's payload-independent selected metadata.
struct ActivationSelectionView {
  Tag tag;
  TagTime tag_time;
  /// Semantic selection coordinate, such as the current camera anchor.
  TimePoint reference_time;
  /// Later source witness that closed and activated the decision.
  TimePoint close_time;
  MissingAggregateDisposition disposition{MissingAggregateDisposition::kNoFire};
  EndpointClockHealthDisposition clock_health_disposition{
      EndpointClockHealthDisposition::kNotApplicable};
  std::span<const InputSelection> inputs;
  std::span<const TemporalSampleRecord> selected;
  std::uint64_t generation{};
};

/// Own a preallocated, activation-scoped copy of selected temporal metadata.
///
/// Queue transactions remain owned by the selector. The caller commits a
/// scalar or DrainAll transaction only for nominal/degraded acceptance and
/// lets its destructor abort on no-fire. Copying the selected metadata before
/// destructive commit keeps callback-visible selection stable through
/// retirement. EA1-012 attaches typed `Sample<T>` claims to the same slots.
class ActivationSelectionState {
 public:
  ActivationSelectionState(const ActivationSelectionState&) = delete;
  ActivationSelectionState& operator=(const ActivationSelectionState&) = delete;
  ActivationSelectionState(ActivationSelectionState&&) noexcept = default;
  ActivationSelectionState& operator=(ActivationSelectionState&&) noexcept = default;
  ~ActivationSelectionState() = default;

  [[nodiscard]] static expected<ActivationSelectionState, Error> create(std::size_t max_inputs,
                                                                        std::size_t max_selected);

  [[nodiscard]] expected<ActivationSelectionView, Error> begin(
      const FrozenRuntimeCandidate& candidate, std::span<const InputSelection> inputs,
      std::span<const TemporalSampleRecord> selected,
      TimePoint semantic_reference_time = {}) noexcept;
  [[nodiscard]] expected<ActivationSelectionView, Error> view(
      std::uint64_t generation) const noexcept;
  [[nodiscard]] expected<void, Error> retire(Tag tag) noexcept;

  [[nodiscard]] bool active() const noexcept { return active_; }
  [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
  [[nodiscard]] std::size_t max_inputs() const noexcept { return max_inputs_; }
  [[nodiscard]] std::size_t max_selected() const noexcept { return max_selected_; }

 private:
  ActivationSelectionState(std::size_t max_inputs, std::size_t max_selected);

  std::vector<InputSelection> inputs_;
  std::vector<TemporalSampleRecord> selected_;
  Tag tag_;
  TagTime tag_time_;
  TimePoint reference_time_;
  TimePoint close_time_;
  MissingAggregateDisposition disposition_{MissingAggregateDisposition::kNoFire};
  EndpointClockHealthDisposition clock_health_disposition_{
      EndpointClockHealthDisposition::kNotApplicable};
  std::size_t max_inputs_{};
  std::size_t max_selected_{};
  std::uint64_t generation_{1U};
  bool active_{};
};

/// Own one run's deterministic READY set and policy state.
///
/// Construction and `freeze()` are cold and may allocate up to the accepted
/// bound. `select()` and `complete()` perform no allocation or graph traversal.
/// At most one selection may be outstanding, which is the process-local
/// global-serial activation token even when the accepted plan owns multiple
/// execution lanes.
class Dispatcher {
 public:
  Dispatcher(const Dispatcher&) = delete;
  Dispatcher& operator=(const Dispatcher&) = delete;
  Dispatcher(Dispatcher&&) noexcept = default;
  Dispatcher& operator=(Dispatcher&&) noexcept = default;
  ~Dispatcher() = default;

  [[nodiscard]] static expected<Dispatcher, Error> create(const SchedulerPlanRecord& plan,
                                                          std::size_t max_candidates);

  /// Freeze one source-eligible set after applying captured health disposition.
  ///
  /// `kSkip` entries remain inactive and must have their coordinator credit
  /// retired without compute. `kDegraded` entries remain selectable and carry
  /// that disposition into the activation. `kFailRun` rejects the whole set.
  [[nodiscard]] expected<void, Error> freeze(std::span<const DispatcherCandidate> candidates);
  [[nodiscard]] expected<DispatcherSelection, Error> select() noexcept;
  [[nodiscard]] expected<void, Error> complete(const DispatcherSelection& selection) noexcept;

  /// Cancel every frozen candidate owned by one receiver.
  ///
  /// Receiver cancellation is valid only between activations. It is used by
  /// lifecycle restart fencing so an idle target cannot enter compute with a
  /// candidate frozen under its previous source and Tag generations.
  [[nodiscard]] expected<std::size_t, Error> cancel_receiver(std::size_t receiver_index) noexcept;
  void cancel_all() noexcept;

  [[nodiscard]] std::size_t active_count() const noexcept { return active_count_; }
  [[nodiscard]] std::size_t max_candidates() const noexcept { return max_candidates_; }
  [[nodiscard]] bool activation_in_progress() const noexcept {
    return policy_state_.activation_in_progress;
  }

 private:
  Dispatcher(const SchedulerPlanRecord& plan, std::unique_ptr<scheduling::ReadyPolicy> policy,
             scheduling::ReadyPolicyState policy_state, std::size_t max_candidates);

  const SchedulerPlanRecord* plan_{};
  std::unique_ptr<scheduling::ReadyPolicy> policy_;
  scheduling::ReadyPolicyState policy_state_;
  std::vector<scheduling::ReadyCandidateView> ready_;
  std::vector<Tag> tags_;
  std::vector<std::size_t> receiver_indices_;
  std::vector<EndpointClockHealthDisposition> clock_health_dispositions_;
  std::size_t max_candidates_{};
  std::size_t active_count_{};
};

}  // namespace holoscan::detail
