/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <limits>
#include <utility>

#include "run_session_internal.hpp"

#include "runtime_test_access.hpp"

namespace holoscan::detail {

// Temporal ingestion, eligibility, and frozen-candidate construction.

namespace {

[[nodiscard]] constexpr std::string_view channel_terminal_token(
    const std::optional<StructuralDiagnosticId> diagnostic) noexcept {
  if (!diagnostic) {
    return "RUNTIME_READINESS_BINDING_UNAVAILABLE";
  }
  if (const auto* descriptor = find_structural_diagnostic(*diagnostic); descriptor != nullptr) {
    return descriptor->token;
  }
  // A new Channel producer must register its typed structural identity instead of silently
  // degrading to a generic readiness failure.
  return "COMPILER_DIAGNOSTIC_TOKEN_UNREGISTERED";
}

}  // namespace

[[nodiscard]] bool RunSessionState::witness_reference_floor(
    ReadyRunState& state, std::optional<TimePoint>& floor) noexcept {
  floor =
      state.coordinator != nullptr ? state.coordinator->oldest_pending_reference() : std::nullopt;
  for (const ClockTriggerPortBinding* const clock : state.clock_bindings) {
    if (clock == nullptr) {
      continue;
    }
    auto next = clock->next_reference();
    if (!next) {
      return false;
    }
    const auto& next_reference = *next;
    if (!next_reference.has_value()) {
      continue;
    }
    if (!floor || next_reference->timestamp_ns < floor->timestamp_ns) {
      floor = *next_reference;
    }
  }
  return true;
}

// The reserve, evict, promote, and retry sequence is one hot ownership-transfer transaction.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool RunSessionState::reconcile_superseded_retention(
    ReadyRunState& state, SourceId readiness_source, std::size_t& refill_credit,
    std::size_t& recovered_count) noexcept {
  if (state.channels == nullptr || !readiness_source) {
    return true;
  }
  // Lazily resolved on the first selector that reaches an eviction, then reused for this
  // source-local recovery pass.
  std::optional<TimePoint> floor;
  bool floor_known = false;
  for (auto& slot : state.input_selectors) {
    // A bracketed selector owns its retained set through window closure and a drain selector
    // delivers an ordered FIFO batch; removing one record from either would change what the
    // contract author asked for. Aligned members retire through their join engine instead.
    //
    // Exact cache rows remain ineligible: a later sample cannot substitute for the one exact
    // coordinate they satisfy. Its independent ReuseLast copy is different. Once the selector
    // has already used N and transport stages strictly newer N+1, the controller may hand off
    // that exact reuse record without evicting any Exact cache row.
    const InputSelectorKind selector = slot.selector.plan().selector;
    if (slot.readiness_source != readiness_source || slot.aligned_member || !slot.input ||
        (selector != InputSelectorKind::kLatestWithin &&
         slot.selector.plan().missing != TemporalRecordMissingKind::kReuseLast)) {
      continue;
    }
    // A committed replacement remains fenced until its exact prerequisite event is published,
    // ingested, and retired. Do not start another reconcile transaction for the same selector
    // while a deferred recovery poll still owns that obligation.
    if (slot.freshness_refill_prerequisite) {
      continue;
    }
    // A release that lost the producer lock leaves its exact successor fenced. Recover it before
    // every decision gate below, including exhausted credit and an empty selector, so an idle
    // port cannot depend on new traffic to revisit already-staged work.
    auto recovered = state.channels->recover_abandoned_freshness_reservation(slot.input);
    if (!recovered) {
      append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", slot.owner);
      return false;
    }
    if (!*recovered) {
      continue;
    }
    if (refill_credit == 0U) {
      continue;
    }
    // Payload claims are dispatcher-local, so testing them before the staged-sample query keeps
    // an otherwise idle or empty selector off the remaining subscriber mutex probes.
    auto claims = slot.selector.copy_payload_claims(slot.channel_payload_claim_scratch);
    if (!claims) {
      // The preallocated scratch cannot hold what the selector retains. This is the same
      // failure synchronize_temporal_payload_claims treats as terminal, so it fails the same
      // way here rather than being indistinguishable from an idle selector.
      append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", slot.owner);
      return false;
    }
    // Zero retained records means the ordinary ingest path will admit the staged sample without
    // help. Any positive count is in scope: drop-oldest retains the newest N unread survivors,
    // so at depth N the record a newer sample should displace is the oldest of the N held, not
    // the sole one held at depth one.
    if (*claims == 0U) {
      continue;
    }
    // Counted here rather than at the top of the loop: everything above is a structural property
    // of the selector that cannot change within a run, so counting it would report the graph's
    // shape instead of the reconcile's behavior. From here on every exit is a decision about
    // this round, and the eviction plus the decline branches account for this count exactly --
    // apart from the inconsistency below, which ends the session rather than the round.
    count_phase(RuntimePhaseCounter::kTemporalRetentionCandidates);
    const auto claim_span =
        std::span<const TemporalSampleRecord>{slot.channel_payload_claim_scratch}.first(*claims);
    const auto reuse_retained_record = [&]() noexcept {
      count_phase(RuntimePhaseCounter::kTemporalReuseLastRetainedRecordQueries);
      return slot.selector.reuse_retained_record();
    };
    const auto relevant_witness = [&]() -> std::optional<TimePoint> {
      auto witness =
          selector_witness_reference(state, slot.owner, slot.witness_source, slot.witness_port);
      if (witness) {
        return witness;
      }
      if (!slot.upcoming_clock_freshness || slot.receiver_index >= state.clock_bindings.size()) {
        return std::nullopt;
      }
      // Between periodic firings the authored next wake is an exact future witness. The pending
      // lookup above wins while the current firing is still observable.
      const ClockTriggerPortBinding* const clock = state.clock_bindings[slot.receiver_index];
      return clock == nullptr || clock->source() != slot.witness_source ? std::nullopt
                                                                        : clock->next_wake();
    };
    const auto reuse_retained = reuse_retained_record();
    std::optional<TimePoint> reuse_cache_witness;
    if (reuse_retained && !slot.upcoming_clock_freshness &&
        (selector != InputSelectorKind::kLatestWithin ||
         slot.selector.plan().cache_capacity != 1U || slot.selector.retained_count() == 0U)) {
      auto staged = state.channels->staged_reuse_replacement_candidate(slot.input, *reuse_retained);
      if (!staged) {
        append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", slot.owner);
        return false;
      }
      if (staged->has_value()) {
        const std::optional<TimePoint> witness_reference = relevant_witness();
        bool trigger_terminal = false;
        if (!witness_reference && slot.witness_port) {
          auto observed = state.channels->input_terminal_observed(slot.witness_port);
          if (!observed) {
            append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", slot.owner);
            return false;
          }
          trigger_terminal = *observed;
        }
        const bool terminal_unobservable =
            !witness_reference && trigger_terminal && state.dispatcher != nullptr &&
            state.dispatcher->active_count() == 0U && state.selection.frozen_candidates.empty() &&
            state.coordinator != nullptr &&
            state.coordinator->no_pending_trigger_witness(slot.owner, slot.witness_port);
        // For a sole input-trigger lane, the staged data record is itself the next immutable
        // witness once published. Exact matches its own capture coordinate and LatestWithin
        // admits it at age zero, so this is selector eligibility rather than an absent-witness
        // guess. Independent clock/notification/driver triggers do not satisfy this proof.
        const bool self_trigger_successor =
            !witness_reference && slot.input == slot.witness_port &&
            slot.readiness_source == slot.witness_source && state.dispatcher != nullptr &&
            state.dispatcher->active_count() == 0U && state.selection.frozen_candidates.empty() &&
            state.coordinator != nullptr &&
            state.coordinator->no_pending_trigger_witness(slot.owner, slot.witness_port);
        if (!terminal_unobservable && !self_trigger_successor &&
            (!witness_reference || !slot.selector.staged_reuse_eligible(
                                       (**staged).successor_reference, *witness_reference))) {
          count_phase(RuntimePhaseCounter::kTemporalRetentionDeclinedAboveFloor);
          continue;
        }
        // This input's immutable trigger is the only witness that can consume its selector
        // state. Other pending candidates may sort earlier, but cannot observe this retained
        // record; eligibility against the exact trigger coordinate is therefore the complete
        // handoff gate. In particular, an absent trigger still fails closed above.
        const auto prior = std::find(slot.channel_payload_claims.begin(),
                                     slot.channel_payload_claims.end(),
                                     *reuse_retained);
        if (prior == slot.channel_payload_claims.end()) {
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", slot.owner);
          return false;
        }
        auto selector_reservation = slot.selector.reserve_reuse_replacement(*reuse_retained);
        if (!selector_reservation) {
          count_phase(RuntimePhaseCounter::kTemporalRetentionDeclinedEvictionRefused);
          continue;
        }
        auto replaced = state.channels->replace_staged_reuse(slot.input, *reuse_retained, **staged);
        if (!replaced) {
          if (replaced.error().code == ErrorCode::kBackpressured ||
              replaced.error().code == ErrorCode::kNotReady) {
            count_phase(RuntimePhaseCounter::kTemporalRetentionDeclinedEvictionRefused);
            continue;
          }
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", slot.owner);
          return false;
        }
        selector_reservation->commit();
        slot.freshness_promoted_this_round = true;
        slot.freshness_promoted_capture_ns = (**staged).successor_reference.timestamp_ns;
        if (slot.nontrigger_prerequisite) {
          TemporalSampleRecord prerequisite = *replaced;
          prerequisite.capture_time = (**staged).successor_reference;
          slot.freshness_refill_prerequisite = prerequisite;
          slot.freshness_refill_target = prerequisite;
          state.freshness_evicted_this_round = true;
        }
        slot.channel_payload_claims.erase(prior);
        // The Channel receive and selector retirement are now irreversible and both ledgers
        // agree. Only after that commit may the auxiliary process-local retry propagate an
        // error; otherwise reservation rollback would restore metadata for the replaced payload.
        if (auto retried = state.channels->retry_after_freshness_handoff(); !retried) {
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", slot.owner);
          return false;
        }
        --refill_credit;
        ++recovered_count;
        count_phase(RuntimePhaseCounter::kTemporalRetentionEvictions);
        continue;
      }
      // Direct ingress already has a cold successor slot and uses the established reversible
      // freshness reservation. Apply that transaction to the exact ReuseLast record as well;
      // choosing the minimum over all claims would select a cache row at depth greater than one.
      auto refill_candidate =
          state.channels->has_freshness_refill_candidate(slot.input, *reuse_retained);
      if (!refill_candidate) {
        append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", slot.owner);
        return false;
      }
      if (*refill_candidate) {
        const std::optional<TimePoint> witness_reference = relevant_witness();
        bool trigger_terminal = false;
        if (!witness_reference && slot.witness_port) {
          auto observed = state.channels->input_terminal_observed(slot.witness_port);
          if (!observed) {
            append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", slot.owner);
            return false;
          }
          trigger_terminal = *observed;
        }
        const bool terminal_trigger_unobservable =
            !witness_reference && trigger_terminal && state.dispatcher != nullptr &&
            state.dispatcher->active_count() == 0U && state.selection.frozen_candidates.empty() &&
            state.coordinator != nullptr &&
            state.coordinator->no_pending_trigger_witness(slot.owner, slot.witness_port);
        const bool self_trigger_successor =
            !witness_reference && slot.input == slot.witness_port &&
            slot.readiness_source == slot.witness_source && state.dispatcher != nullptr &&
            state.dispatcher->active_count() == 0U && state.selection.frozen_candidates.empty() &&
            state.coordinator != nullptr &&
            state.coordinator->no_pending_trigger_witness(slot.owner, slot.witness_port);
        const bool no_witness_eligible = terminal_trigger_unobservable || self_trigger_successor;
        if (!witness_reference && !no_witness_eligible) {
          count_phase(RuntimePhaseCounter::kTemporalRetentionDeclinedAboveFloor);
          continue;
        }
        // The successor is reserved only through the coordinate of the immutable trigger that
        // can observe this selector. A global coordinator floor can contain unrelated data or
        // another operator's event; neither is a consumer of this ReuseLast record.
        const bool nontrigger_prerequisite = slot.nontrigger_prerequisite;
        RuntimeChannelFreshnessSelectionMode selection_mode =
            RuntimeChannelFreshnessSelectionMode::kNewest;
        if (nontrigger_prerequisite && selector == InputSelectorKind::kLatestWithin) {
          selection_mode = RuntimeChannelFreshnessSelectionMode::kNewestWithArrivalPrefix;
        } else if (nontrigger_prerequisite && selector == InputSelectorKind::kExact) {
          selection_mode = RuntimeChannelFreshnessSelectionMode::kOldestProducerSuccessor;
        }
        auto reservation = state.channels->reserve_freshness_refill(
            slot.input, *reuse_retained, witness_reference, selection_mode);
        if (!reservation) {
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", slot.owner);
          return false;
        }
        if (!reservation->has_value()) {
          count_phase(RuntimePhaseCounter::kTemporalRetentionDeclinedAboveFloor);
          continue;
        }
        // A non-trigger LatestWithin input qualifies the fenced newest target. Its earlier
        // producer-order prefix need not individually satisfy the current age window. Single
        // successor modes qualify the exact record they will admit.
        const TimePoint eligible_reference =
            selection_mode == RuntimeChannelFreshnessSelectionMode::kNewestWithArrivalPrefix
                ? (**reservation).target_reference
                : (**reservation).successor_reference;
        if (!no_witness_eligible &&
            (!witness_reference ||
             !slot.selector.staged_reuse_eligible(eligible_reference, *witness_reference))) {
          state.channels->release_freshness_reservation(**reservation);
          count_phase(RuntimePhaseCounter::kTemporalRetentionDeclinedAboveFloor);
          continue;
        }
        auto selector_reservation = slot.selector.reserve_reuse_replacement(*reuse_retained);
        if (!selector_reservation) {
          state.channels->release_freshness_reservation(**reservation);
          count_phase(RuntimePhaseCounter::kTemporalRetentionDeclinedEvictionRefused);
          continue;
        }
        std::uint32_t promoted = 0U;
        std::int64_t promoted_capture_ns = 0;
        if (!synchronize_temporal_payload_claims(
                state, slot, &promoted, &**reservation, &promoted_capture_ns)) {
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", slot.owner);
          return false;
        }
        if (promoted != 1U) {
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", slot.owner);
          return false;
        }
        selector_reservation->commit();
        slot.freshness_promoted_this_round = true;
        slot.freshness_promoted_capture_ns = promoted_capture_ns;
        if (nontrigger_prerequisite) {
          TemporalSampleRecord prerequisite = (**reservation).successor;
          prerequisite.capture_time = (**reservation).successor_reference;
          slot.freshness_refill_prerequisite = prerequisite;
          TemporalSampleRecord target = (**reservation).target;
          target.capture_time = (**reservation).target_reference;
          slot.freshness_refill_target = target;
          state.freshness_evicted_this_round = true;
        }
        // Both ownership ledgers now agree that the old ReuseLast payload is gone. Only after
        // that irreversible two-party commit may an auxiliary process-local retry fail the run;
        // otherwise the selector reservation destructor would restore metadata for a payload
        // that Channel already replaced.
        if (auto retried = state.channels->retry_after_freshness_handoff(); !retried) {
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", slot.owner);
          return false;
        }
        --refill_credit;
        ++recovered_count;
        count_phase(RuntimePhaseCounter::kTemporalRetentionEvictions);
        count_phase(RuntimePhaseCounter::kTemporalRetentionEvictionPromoted);
        continue;
      }
    }
    if (selector != InputSelectorKind::kLatestWithin) {
      count_phase(RuntimePhaseCounter::kTemporalRetentionDeclinedNotSuperseded);
      continue;
    }
    // The oldest ordinary cache claim is the one LatestWithin drop-oldest displaces. ReuseLast
    // was handled above through its exact independent record and must never be selected by this
    // minimum over all claims.
    const auto retained_entry = std::min_element(
        claim_span.begin(),
        claim_span.end(),
        [&](const TemporalSampleRecord& left, const TemporalSampleRecord& right) noexcept {
          count_phase(RuntimePhaseCounter::kTemporalReuseLastComparatorCalls);
          const bool left_reuse = reuse_retained_record().value_or(TemporalSampleRecord{}) == left;
          const bool right_reuse =
              reuse_retained_record().value_or(TemporalSampleRecord{}) == right;
          if (left_reuse != right_reuse) {
            return !left_reuse;
          }
          return std::tie(left.producer_epoch, left.arrival_sequence) <
                 std::tie(right.producer_epoch, right.arrival_sequence);
        });
    if (retained_entry == claim_span.end() ||
        (reuse_retained_record() && *retained_entry == *reuse_retained_record())) {
      count_phase(RuntimePhaseCounter::kTemporalRetentionDeclinedNotSuperseded);
      continue;
    }
    const TemporalSampleRecord retained = *retained_entry;
    if (reuse_retained && slot.upcoming_clock_freshness &&
        slot.selector.plan().cache_capacity == 1U) {
      auto staged = state.channels->staged_reuse_replacement_candidate(slot.input, retained);
      if (!staged) {
        append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", slot.owner);
        return false;
      }
      if (staged->has_value()) {
        const std::optional<TimePoint> witness_reference = relevant_witness();
        if (!witness_reference || !slot.selector.staged_reuse_eligible(
                                      (**staged).successor_reference, *witness_reference)) {
          count_phase(RuntimePhaseCounter::kTemporalRetentionDeclinedAboveFloor);
          continue;
        }
        const auto prior = std::find(
            slot.channel_payload_claims.begin(), slot.channel_payload_claims.end(), retained);
        if (prior == slot.channel_payload_claims.end()) {
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", slot.owner);
          return false;
        }
        auto selector_reservation = slot.selector.reserve_cache_replacement(retained);
        if (!selector_reservation) {
          count_phase(RuntimePhaseCounter::kTemporalRetentionDeclinedEvictionRefused);
          continue;
        }
        auto replaced = state.channels->replace_staged_reuse(slot.input, retained, **staged);
        if (!replaced) {
          if (replaced.error().code == ErrorCode::kBackpressured ||
              replaced.error().code == ErrorCode::kNotReady) {
            count_phase(RuntimePhaseCounter::kTemporalRetentionDeclinedEvictionRefused);
            continue;
          }
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", slot.owner);
          return false;
        }
        selector_reservation->commit();
        slot.freshness_promoted_this_round = true;
        slot.freshness_promoted_capture_ns = (**staged).successor_reference.timestamp_ns;
        if (slot.nontrigger_prerequisite) {
          TemporalSampleRecord prerequisite = *replaced;
          prerequisite.capture_time = (**staged).successor_reference;
          slot.freshness_refill_prerequisite = prerequisite;
          slot.freshness_refill_target = prerequisite;
          state.freshness_evicted_this_round = true;
        }
        slot.channel_payload_claims.erase(prior);
        if (auto retried = state.channels->retry_after_freshness_handoff(); !retried) {
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", slot.owner);
          return false;
        }
        --refill_credit;
        ++recovered_count;
        count_phase(RuntimePhaseCounter::kTemporalRetentionEvictions);
        continue;
      }
    }
    auto refill_candidate = state.channels->has_freshness_refill_candidate(slot.input, retained);
    if (!refill_candidate) {
      append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", slot.owner);
      return false;
    }
    if (!*refill_candidate) {
      count_phase(RuntimePhaseCounter::kTemporalRetentionDeclinedNotSuperseded);
      continue;
    }
    if (reuse_retained && selector == InputSelectorKind::kLatestWithin &&
        slot.selector.plan().cache_capacity == 1U && slot.selector.retained_count() != 0U) {
      reuse_cache_witness = relevant_witness();
      if (!reuse_cache_witness) {
        count_phase(RuntimePhaseCounter::kTemporalRetentionDeclinedAboveFloor);
        continue;
      }
    }
    // Computed at most once per round, and only for a selector that has already earned an
    // possible eviction. Hoisting it above the loop would read every clock on every dispatch
    // iteration even when no selector retains a payload, which is the common case.
    if (!reuse_cache_witness && !floor_known) {
      if (!witness_reference_floor(state, floor)) {
        slot.freshness_gate_inert_this_round = false;
        // A clock that cannot be read leaves the round's witnesses unbounded. Declining costs a
        // freshness refresh; proceeding could strand a witness this record would have satisfied.
        count_phase(RuntimePhaseCounter::kTemporalRetentionDeclinedUnboundedRound);
        return true;
      }
      floor_known = true;
      count_phase(floor ? RuntimePhaseCounter::kTemporalRetentionFloorResolved
                        : RuntimePhaseCounter::kTemporalRetentionFloorAbsent);
    }
    const std::optional<TimePoint>& eviction_floor =
        reuse_cache_witness ? reuse_cache_witness : floor;
    // No floor means no witness is possible this round as seen from here, so the gate does not
    // apply. A selector with a reuse fallback instead uses its exact relevant witness above.
    slot.freshness_gate_inert_this_round = !eviction_floor.has_value();
    // Reserve the record this eviction will admit, before releasing anything. Qualifying here
    // and admitting whatever staging holds by the time the promotion runs is unsound however
    // sound the qualification was: at a fast producer the qualified record is replaced in
    // between, and its replacement is captured after the witness this eviction was justified
    // against, so it sorts behind that witness rather than ahead of it.
    //
    // The reservation subsumes the floor comparison that used to stand here. Coordinates from
    // different clock domains are not comparable, so an unnormalized capture declines inside it
    // rather than being compared by raw nanoseconds.
    const bool nontrigger_prerequisite = slot.nontrigger_prerequisite;
    const RuntimeChannelFreshnessSelectionMode selection_mode =
        nontrigger_prerequisite ? RuntimeChannelFreshnessSelectionMode::kNewestWithArrivalPrefix
                                : RuntimeChannelFreshnessSelectionMode::kNewest;
    auto reservation = state.channels->reserve_freshness_refill(
        slot.input, retained, eviction_floor, selection_mode);
    if (!reservation) {
      append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", slot.owner);
      return false;
    }
    if (!reservation->has_value()) {
      count_phase(RuntimePhaseCounter::kTemporalRetentionDeclinedAboveFloor);
      continue;
    }
    // Unlike the two queries above, every failure here is a documented refusal rather than an
    // inconsistency: kNotSupported for a bracketed selector, kBackpressured while a selection
    // transaction is reading the retained set, kNotFound for an identity the cache no longer
    // holds. Each leaves the candidate exactly as it was, so a later pass can reconcile it.
    const expected<void, Error> evicted = slot.selector.evict_retained(retained.arrival_sequence);
    if (!evicted) {
      // The reservation exists only to protect this eviction. Abandoning it keeps the staging
      // pool's ordinary drop-oldest from being pinned by a trade that did not happen.
      state.channels->release_freshness_reservation(**reservation);
      count_phase(RuntimePhaseCounter::kTemporalRetentionDeclinedEvictionRefused);
      continue;
    }
    count_phase(RuntimePhaseCounter::kTemporalRetentionEvictions);
    slot.freshness_evicted_this_round = true;
    state.freshness_evicted_this_round = true;
    slot.freshness_successor_capture_ns = (*reservation)->successor.capture_time.timestamp_ns;
    slot.freshness_floor_ns = eviction_floor ? eviction_floor->timestamp_ns : 0;
    // Ordering is mandatory: until the eviction above, the cached record named a live claim.
    // Retiring it now is no longer optional either, because the selector has already given it
    // up -- claim synchronization is what hands it back and frees the slot for promotion.
    std::uint32_t promoted = 0U;
    std::int64_t promoted_capture_ns = 0;
    if (!synchronize_temporal_payload_claims(
            state, slot, &promoted, &**reservation, &promoted_capture_ns)) {
      append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", slot.owner);
      return false;
    }
    // The release is only half the trade. If it promoted nothing, the successor this eviction
    // was performed for is still staged and the cache is now empty with nothing in flight to
    // refill it -- the exact shape of the regression, and distinguishable here from the case
    // where the successor is on its way through the poll below.
    slot.freshness_promoted_this_round = promoted != 0U;
    slot.freshness_promoted_capture_ns = promoted_capture_ns;
    if (promoted != 1U) {
      append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", slot.owner);
      return false;
    }
    if (nontrigger_prerequisite) {
      TemporalSampleRecord prerequisite = (**reservation).successor;
      prerequisite.capture_time = (**reservation).successor_reference;
      slot.freshness_refill_prerequisite = prerequisite;
      TemporalSampleRecord target = (**reservation).target;
      target.capture_time = (**reservation).target_reference;
      slot.freshness_refill_target = target;
    }
    // Channel and selector ownership now agree on the promoted successor. Only after that
    // irreversible commit may the auxiliary process-local retry propagate an error.
    if (auto retried = state.channels->retry_after_freshness_handoff(); !retried) {
      append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", slot.owner);
      return false;
    }
    --refill_credit;
    ++recovered_count;
    count_phase(RuntimePhaseCounter::kTemporalRetentionEvictionPromoted);
  }
  return true;
}

// Validate every member's pending-window state before publishing a timeout witness for the
// operator, preserving one coherent closure and clock coordinate.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool RunSessionState::enqueue_due_window_timeouts(GraphExec& exec,
                                                                ReadyRunState& state) noexcept {
  for (const WindowTriggerPlanRecord& window : plan_->window_triggers) {
    if (window.owner.plan_generation != plan_->plan_generation) {
      continue;
    }
    RuntimeOperatorInstance* const instance = exec.operator_by_id(window.owner.id);
    const auto receiver = exec.receiver_index(window.owner.id);
    if (instance == nullptr || !receiver) {
      continue;
    }
    if (!owns_receiver(state, *receiver)) {
      continue;
    }
    if (window.bracketed_inputs.empty() || window.max_wait_ns <= 0 || !window.wait_clock.valid()) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
      return false;
    }

    std::optional<std::size_t> pending_count;
    std::optional<TimePoint> oldest_reference;
    for (const PortId input : window.bracketed_inputs) {
      const auto selector = std::find_if(
          state.input_selectors.begin(), state.input_selectors.end(), [&](const auto& slot) {
            return slot.owner == instance->vertex && slot.input == input;
          });
      if (selector == state.input_selectors.end()) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance->vertex);
        return false;
      }
      const std::size_t count = selector->selector.pending_window_count();
      const std::optional<TimePoint> reference =
          selector->selector.oldest_pending_window_reference();
      if ((count == 0U) != !reference || (pending_count && *pending_count != count) ||
          (oldest_reference && reference && *oldest_reference != *reference)) {
        append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance->vertex);
        return false;
      }
      pending_count = count;
      if (reference) {
        oldest_reference = *reference;
      }
    }
    if (!pending_count || *pending_count == 0U) {
      continue;
    }
    if (!oldest_reference || oldest_reference->clock_id != window.wait_clock ||
        oldest_reference->timestamp_ns >
            std::numeric_limits<std::int64_t>::max() - window.max_wait_ns) {
      append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance->vertex);
      return false;
    }
    auto clock = exec.resolve_clock(window.wait_clock);
    auto now = clock ? clock->now() : expected<TimePoint, Error>{make_unexpected(clock.error())};
    if (!now || now->clock_id != window.wait_clock) {
      append_diagnostic("RUNTIME_CLOCK_READ_FAILED", instance->vertex);
      return false;
    }
    const std::int64_t deadline_ns = oldest_reference->timestamp_ns + window.max_wait_ns;
    if (now->timestamp_ns < deadline_ns) {
      continue;
    }
    const TemporalTriggerPlanRecord* const trigger = instance->temporal_trigger;
    PortBinding* const binding =
        trigger == nullptr ? nullptr : physical_binding_for(state, trigger->readiness_source);
    if (trigger == nullptr || trigger->kind != TemporalRecordTriggerKind::kWindow ||
        trigger->trigger_input != window.anchor || binding == nullptr) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance->vertex);
      return false;
    }
    auto enqueued = binding->enqueue_window_timeout(window.anchor, *now);
    if (!enqueued && enqueued.error().code != ErrorCode::kBackpressured) {
      append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance->vertex);
      return false;
    }
    if (enqueued && plan_->executable_factories.test_window_timeout_control != nullptr) {
      plan_->executable_factories.test_window_timeout_control->evaluated(
          state.runner != nullptr ? state.runner->partition : ExecutionPartitionId{},
          instance->vertex);
    }
  }
  return true;
}

[[nodiscard]] bool RunSessionState::arrival_trigger_pin_matches(
    const RuntimeOperatorInstance& instance,
    const FrozenRuntimeCandidate& candidate) const noexcept {
  if (candidate.witness.kind != RuntimeEventKind::kInputArrival || !candidate.witness.port ||
      candidate.witness.source_sequence == 0U) {
    return false;
  }
  const TemporalTriggerPlanRecord* const trigger = instance.temporal_trigger;
  if (trigger == nullptr || trigger->owner.plan_generation != plan_->plan_generation) {
    return false;
  }
  if (trigger->kind == TemporalRecordTriggerKind::kEach) {
    return trigger->trigger_input == candidate.witness.port;
  }
  if (trigger->kind != TemporalRecordTriggerKind::kAnyInput) {
    return false;
  }
  // OnAnyInput is defined over every connected input, including ports without an explicit
  // temporal selector. Those ports intentionally do not appear in declared_inputs, so validate
  // the witness against the frozen physical input inventory instead of the selector inventory.
  const auto* port =
      find_port_by_id_and_direction(*plan_, candidate.witness.port, PortDirection::kInput);
  return port != nullptr && port->owner == instance.vertex &&
         port->owner_kind == PortOwnerKind::kOperator;
}

[[nodiscard]] bool RunSessionState::arrival_trigger_pin_matches(
    const GraphExec& exec, const FrozenRuntimeCandidate& candidate) const noexcept {
  return candidate.candidate.receiver_index < exec.operators().size() &&
         arrival_trigger_pin_matches(exec.operators()[candidate.candidate.receiver_index],
                                     candidate);
}

[[nodiscard]] expected<void, Error> RunSessionState::synchronize_temporal_payload_claims(
    ReadyRunState& state, ReadyRunState::InputSelectorSlot& slot, std::uint32_t* promoted_out,
    const RuntimeChannelFreshnessReservation* freshness_reservation,
    std::int64_t* promoted_capture_out) noexcept {
  if (state.channels == nullptr) {
    slot.channel_payload_claims.clear();
    return {};
  }
  auto claims = [&]() -> expected<std::size_t, Error> {
    if (!slot.aligned_member) {
      return slot.selector.copy_payload_claims(slot.channel_payload_claim_scratch);
    }
    const auto aligned = std::find_if(state.aligned_selectors.begin(),
                                      state.aligned_selectors.end(),
                                      [&](const auto& row) { return row.owner == slot.owner; });
    return aligned == state.aligned_selectors.end()
               ? expected<std::size_t, Error>{make_unexpected(Error{ErrorCode::kInvalidArgument})}
               : aligned->selector.copy_payload_claims(slot.input,
                                                       slot.channel_payload_claim_scratch);
  }();
  if (!claims || *claims > slot.channel_payload_claim_scratch.size()) {
    return make_unexpected(claims ? Error{ErrorCode::kProtocolError} : std::move(claims).error());
  }

  const auto less = [](const TemporalSampleRecord& left,
                       const TemporalSampleRecord& right) noexcept {
    return std::tie(left.producer_epoch,
                    left.arrival_sequence,
                    left.source_sequence,
                    left.capture_time.clock_id.value,
                    left.capture_time.timestamp_ns) < std::tie(right.producer_epoch,
                                                               right.arrival_sequence,
                                                               right.source_sequence,
                                                               right.capture_time.clock_id.value,
                                                               right.capture_time.timestamp_ns);
  };
  auto scratch_begin = slot.channel_payload_claim_scratch.begin();
  auto scratch_end = scratch_begin + static_cast<std::ptrdiff_t>(*claims);
  std::sort(scratch_begin, scratch_end, less);
  scratch_end = std::unique(scratch_begin, scratch_end);

  std::size_t retired_count = 0U;
  for (const TemporalSampleRecord& prior : slot.channel_payload_claims) {
    if (!std::binary_search(scratch_begin, scratch_end, prior, less)) {
      if (retired_count == slot.channel_payload_retire_scratch.size()) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      slot.channel_payload_retire_scratch[retired_count++] = prior;
    }
  }
  if (retired_count != 0U) {
    // Channel validates and retires the complete delta atomically. Supplying one preallocated
    // batch prevents a malformed later record from releasing any earlier selector claim.
    auto released = state.channels->retire_samples(
        slot.input,
        std::span<const TemporalSampleRecord>{slot.channel_payload_retire_scratch}.first(
            retired_count),
        promoted_out,
        freshness_reservation,
        promoted_capture_out);
    if (!released) {
      return make_unexpected(std::move(released).error());
    }
  }
  slot.channel_payload_claims.clear();
  slot.channel_payload_claims.insert(slot.channel_payload_claims.end(), scratch_begin, scratch_end);
  return {};
}

[[nodiscard]] bool RunSessionState::synchronize_temporal_payload_claims(ReadyRunState& state,
                                                                        VertexId owner) noexcept {
  for (auto& slot : state.input_selectors) {
    if (slot.owner != owner) {
      continue;
    }
    if (!synchronize_temporal_payload_claims(state, slot)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool RunSessionState::synchronize_temporal_payload_claims(
    ReadyRunState& state) noexcept {
  for (auto& slot : state.input_selectors) {
    if (!synchronize_temporal_payload_claims(state, slot)) {
      return false;
    }
  }
  return true;
}

// Channel ingestion atomically reconciles producer epochs, temporal selectors, bracket anchors,
// predicate state, and payload claims for one frozen witness.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool RunSessionState::ingest_channel_candidate(
    GraphExec& exec, ReadyRunState& state, const FrozenRuntimeCandidate& candidate,
    const TemporalSampleRecord* pre_resolved_sample) {
  if (candidate.witness.kind == RuntimeEventKind::kInputEndOfStream) {
    if (!candidate.witness.port || !candidate.witness.reference_time.valid() ||
        candidate.candidate.receiver_index >= exec.operators().size()) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
      return false;
    }
    const RuntimeOperatorInstance& instance = exec.operators()[candidate.candidate.receiver_index];
    bool closed = false;
    for (auto& selector : state.input_selectors) {
      if (selector.owner != instance.vertex || selector.input != candidate.witness.port) {
        continue;
      }
      if (selector.aligned_member) {
        const auto aligned = std::find_if(
            state.aligned_selectors.begin(), state.aligned_selectors.end(), [&](const auto& slot) {
              return slot.owner == selector.owner;
            });
        if (aligned == state.aligned_selectors.end()) {
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", selector.owner);
          return false;
        }
        selector.end_of_stream_decision_pending =
            selector.end_of_stream_decision_pending ||
            std::any_of(aligned->selector.member_inputs().begin(),
                        aligned->selector.member_inputs().end(),
                        [&](const PortId input) { return aligned->selector.queued(input) != 0U; });
        if (!aligned->selector.close_end_of_stream(selector.input,
                                                   candidate.witness.reference_time)) {
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", selector.owner);
          return false;
        }
      } else {
        const bool decision_pending =
            selector.selector.plan().selector == InputSelectorKind::kBracketedWindow &&
            selector.selector.pending_window_count() != 0U;
        if (!selector.selector.close_end_of_stream()) {
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", selector.owner);
          return false;
        }
        selector.end_of_stream_decision_pending =
            selector.end_of_stream_decision_pending || decision_pending;
      }
      selector.end_of_stream_received = true;
      closed = true;
    }
    if (closed && !synchronize_temporal_payload_claims(state, instance.vertex)) {
      append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
      return false;
    }
    return true;
  }
  if (candidate.witness.kind == RuntimeEventKind::kWindowTimeout) {
    if (!candidate.witness.port || candidate.candidate.receiver_index >= exec.operators().size()) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
      return false;
    }
    const RuntimeOperatorInstance& instance = exec.operators()[candidate.candidate.receiver_index];
    const auto window =
        std::find_if(plan_->window_triggers.begin(),
                     plan_->window_triggers.end(),
                     [&](const WindowTriggerPlanRecord& record) {
                       return record.owner.id == instance.operator_id &&
                              record.owner.plan_generation == plan_->plan_generation &&
                              record.anchor == candidate.witness.port;
                     });
    if (window == plan_->window_triggers.end()) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
      return false;
    }
    std::size_t closed_count = 0U;
    for (auto& selector : state.input_selectors) {
      if (selector.owner != instance.vertex ||
          std::find(window->bracketed_inputs.begin(),
                    window->bracketed_inputs.end(),
                    selector.input) == window->bracketed_inputs.end()) {
        continue;
      }
      if (!selector.selector.close_oldest(SelectionReason::kWindowTimeout)) {
        append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
        return false;
      }
      ++closed_count;
    }
    if (closed_count != window->bracketed_inputs.size()) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
      return false;
    }
    if (!synchronize_temporal_payload_claims(state, instance.vertex)) {
      append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
      return false;
    }
    return true;
  }
  if (candidate.witness.kind != RuntimeEventKind::kInputArrival) {
    return true;
  }
  if (state.channels == nullptr || !candidate.witness.port ||
      candidate.witness.source_sequence == 0U) {
    append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
    return false;
  }
  expected<TemporalSampleRecord, Error> resolved_sample =
      pre_resolved_sample != nullptr
          ? expected<TemporalSampleRecord, Error>{*pre_resolved_sample}
          : state.channels->temporal_sample(candidate.witness.port,
                                            candidate.witness.source_sequence);
  if (!resolved_sample) {
    append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
    return false;
  }
  const TemporalSampleRecord& sample = *resolved_sample;
  bool consumed = false;
  for (auto& selector : state.input_selectors) {
    if (selector.input == candidate.witness.port) {
      if (selector.arrival_watermark != 0U && selector.producer_epoch != sample.producer_epoch) {
        if (selector.aligned_member) {
          const auto aligned =
              std::find_if(state.aligned_selectors.begin(),
                           state.aligned_selectors.end(),
                           [&](const auto& slot) { return slot.owner == selector.owner; });
          if (aligned == state.aligned_selectors.end() || !aligned->selector.reset()) {
            append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", selector.owner);
            return false;
          }
          for (auto& member : state.input_selectors) {
            if (member.owner == selector.owner && member.aligned_member) {
              member.arrival_watermark = 0U;
              member.end_of_stream_received = false;
              member.end_of_stream_decision_pending = false;
              member.freshness_refill_prerequisite.reset();
              member.freshness_refill_target.reset();
            }
          }
        } else if (!selector.selector.reset()) {
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", selector.owner);
          return false;
        } else {
          selector.end_of_stream_received = false;
          selector.end_of_stream_decision_pending = false;
          selector.freshness_refill_prerequisite.reset();
          selector.freshness_refill_target.reset();
        }
        RuntimeOperatorInstance* const instance = exec.operator_by_vertex(selector.owner);
        const auto receiver = exec.receiver_index(selector.owner);
        if (instance == nullptr || !receiver) {
          append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", selector.owner);
          return false;
        }
        if (state.predicates[*receiver]) {
          state.predicates[*receiver]->reset(PredicateResetReason::kProducerEpoch,
                                             instance->tags->generation());
        }
      }
      const auto& selection = selector.selector.plan();
      if (!selector.aligned_member && selector.freshness_reconcile &&
          selection.selector == InputSelectorKind::kLatestWithin &&
          selection.missing == TemporalRecordMissingKind::kReuseLast &&
          selection.cache_capacity == 1U && selector.selector.retained_count() == 1U &&
          !sample.capture_time.valid()) {
        // LatestWithin can never select a capture-less arrival. Leave the causal cache row in
        // place and release this arrival through the ordinary unretained-payload path below.
        consumed = true;
        continue;
      }
      const auto pushed = [&]() -> expected<void, Error> {
        if (!selector.aligned_member) {
          if (selector.freshness_reconcile &&
              selection.selector == InputSelectorKind::kLatestWithin &&
              selection.missing == TemporalRecordMissingKind::kReuseLast &&
              selection.cache_capacity == 1U && selector.selector.retained_count() == 1U) {
            auto claims =
                selector.selector.copy_payload_claims(selector.channel_payload_claim_scratch);
            if (!claims || *claims == 0U) {
              return make_unexpected(claims ? Error{ErrorCode::kProtocolError}
                                            : std::move(claims).error());
            }
            auto evicted = selector.selector.evict_retained(
                selector.channel_payload_claim_scratch.front().arrival_sequence);
            if (!evicted) {
              // Ingestion owns no active selector transaction; push() below enforces the same
              // invariant. A refusal here therefore denotes inconsistent frozen-candidate state.
              return evicted;
            }
          }
          return selector.selector.push(sample);
        }
        const auto aligned = std::find_if(
            state.aligned_selectors.begin(), state.aligned_selectors.end(), [&](const auto& slot) {
              return slot.owner == selector.owner;
            });
        return aligned == state.aligned_selectors.end()
                   ? expected<void, Error>{make_unexpected(Error{ErrorCode::kInvalidArgument})}
                   : aligned->selector.push(selector.input, sample);
      }();
      if (!pushed) {
        append_callback_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE",
                                   selector.owner,
                                   {},
                                   pushed.error(),
                                   RuntimeCallbackIdentity{
                                       .source = candidate.witness.source,
                                       .source_sequence = candidate.witness.source_sequence,
                                       .reference_time = candidate.witness.reference_time,
                                   });
        return false;
      }
      // The last link of the refill: a record reaching the cache for this selector in this
      // round. Recording it here closes the chain the eviction started, so a loss can name the
      // step that did not happen instead of the step that was assumed not to.
      selector.freshness_ingested_this_round = true;
      selector.arrival_watermark = sample.arrival_sequence;
      selector.producer_epoch = sample.producer_epoch;
      consumed = true;
      if (!synchronize_temporal_payload_claims(state, selector.owner)) {
        append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", selector.owner);
        return false;
      }
    }
    const auto bracketed = std::find_if(
        plan_->bracketed_windows.begin(), plan_->bracketed_windows.end(), [&](const auto& row) {
          return row.input == selector.input && row.anchor == candidate.witness.port;
        });
    if (bracketed != plan_->bracketed_windows.end()) {
      consumed = true;
      // EOS terminates this bracketed input generation. Later anchors remain valid for their
      // own selectors, but must not create windows that have no remaining EOS event to close.
      if (!selector.end_of_stream_received) {
        auto advanced = selector.selector.push_anchor(sample);
        if (!advanced) {
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", selector.owner);
          return false;
        }
        if (!synchronize_temporal_payload_claims(state, selector.owner)) {
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", selector.owner);
          return false;
        }
      }
    }
  }
  if (!consumed && !arrival_trigger_pin_matches(exec, candidate)) {
    append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
    return false;
  }
  const bool payload_retained = std::any_of(
      state.input_selectors.begin(), state.input_selectors.end(), [&](const auto& selector) {
        return selector.input == candidate.witness.port &&
               std::find(selector.channel_payload_claims.begin(),
                         selector.channel_payload_claims.end(),
                         sample) != selector.channel_payload_claims.end();
      });
  if (!payload_retained) {
    // Anchor-only arrivals and direct OnEach/OnAny trigger pins retain only metadata in the
    // selector layer. Release their temporal claim now; the unretired readiness witness keeps
    // direct-pin storage alive through prepare/compute, then permits bounded reclamation.
    const std::array unretained{sample};
    auto released = state.channels->retire_samples(candidate.witness.port, unretained);
    if (!released) {
      append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE");
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool RunSessionState::channel_candidate_is_eligible(
    const GraphExec& exec, const ReadyRunState& state,
    const FrozenRuntimeCandidate& candidate) const noexcept {
  if (candidate.candidate.receiver_index >= exec.operators().size()) {
    return false;
  }
  const RuntimeOperatorInstance& instance = exec.operators()[candidate.candidate.receiver_index];
  const TemporalTriggerPlanRecord* const trigger = instance.temporal_trigger;
  if (trigger == nullptr) {
    return false;
  }
  if (candidate.witness.kind == RuntimeEventKind::kWindowTimeout) {
    return trigger->kind == TemporalRecordTriggerKind::kWindow &&
           candidate.witness.port == trigger->trigger_input;
  }
  if (candidate.witness.kind == RuntimeEventKind::kInputEndOfStream) {
    const bool witness_matches = std::any_of(
        state.input_selectors.begin(), state.input_selectors.end(), [&](const auto& selector) {
          return selector.owner == instance.vertex && selector.input == candidate.witness.port;
        });
    const bool every_input_closed = std::all_of(
        state.input_selectors.begin(), state.input_selectors.end(), [&](const auto& selector) {
          return selector.owner != instance.vertex || !selector.connected ||
                 selector.end_of_stream_received;
        });
    if (!witness_matches || !every_input_closed) {
      return false;
    }
    return std::any_of(
        state.input_selectors.begin(), state.input_selectors.end(), [&](const auto& selector) {
          if (selector.owner != instance.vertex || !selector.end_of_stream_decision_pending) {
            return false;
          }
          if (trigger->kind == TemporalRecordTriggerKind::kAligned) {
            return selector.aligned_member;
          }
          return trigger->kind == TemporalRecordTriggerKind::kWindow && !selector.aligned_member &&
                 selector.selector.plan().selector == InputSelectorKind::kBracketedWindow;
        });
  }
  if (candidate.witness.kind != RuntimeEventKind::kInputArrival) {
    return true;
  }
  // Only input-backed trigger families may turn an arrival witness into an activation.
  // Clock and readiness-driven Operators still ingest side-input state and retire its readiness
  // credit here, but the independent source remains their sole activation authority. Window
  // successors remain eligible because an input witness can close the frozen bracket.
  return input_arrival_can_activate(*trigger, instance.predicate_trigger, candidate.witness.port);
}

// Form the bounded activation snapshot, health disposition, and retirement obligations in one
// pass so dispatch never observes a partially frozen candidate set.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool RunSessionState::freeze_ready_candidates(
    GraphExec& exec, ReadyRunState& state, std::optional<std::uint32_t> current_execution_lane) {
  if (!exec.has_process_local_callback_owners()) {
    return true;
  }
  if (std::any_of(state.selection.frozen_candidates.begin(),
                  state.selection.frozen_candidates.end(),
                  [](const auto& candidate) { return !candidate.source_retired; })) {
    append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
    static_cast<void>(cancel_ready_sources(state));
    return false;
  }
  state.selection.candidates.clear();
  state.selection.frozen_candidates.clear();
  expected<std::size_t, EndpointClockHealthError> expired{0U};
  if (state.clock_health->receiver_count() != 0U) {
    const std::lock_guard clock_health_lock(clock_health_mutex_);
    expired = state.clock_health->expire_due();
  }
  if (!expired) {
    append_diagnostic(expired.error().token.empty() ? std::string_view{"RUNTIME_CLOCK_READ_FAILED"}
                                                    : expired.error().token);
    return false;
  }
  if (state.channels != nullptr) {
    auto channel_status = state.channels->status();
    if (!channel_status) {
      append_callback_diagnostic(channel_terminal_token(state.channels->terminal_diagnostic()),
                                 state.channels->terminal_owner(),
                                 {},
                                 channel_status.error(),
                                 {});
      return false;
    }
  }
  std::size_t ingested_count = 0U;
  // Rewritten every round before any structural refusal, so diagnostics formed later in this
  // dispatcher iteration describe only the current attempt.
  state.freshness_evicted_this_round = false;
  state.poll_incomplete_this_round = false;
  for (auto& slot : state.input_selectors) {
    slot.freshness_evicted_this_round = false;
    slot.freshness_promoted_this_round = false;
    slot.freshness_gate_inert_this_round = false;
    slot.freshness_ingested_this_round = false;
  }
  for (auto& freshness : state.freshness_bindings) {
    freshness.poll_complete_this_round = false;
    freshness.ordinary_poll_incomplete_this_round = false;
  }
  for (std::size_t binding_index = 0U; binding_index < state.bindings.size(); ++binding_index) {
    auto& binding = state.bindings[binding_index];
    // The coordinator bound is the sum of all source-local bounds. Once it is full, every
    // source already has its admitted prefix represented, so polling another binding can only
    // expose a successor that must remain behind that prefix. Freeze the deterministic minimum
    // first instead of treating the expected bounded condition as a Runtime failure.
    if (state.coordinator->outstanding_count() == state.coordinator->max_outstanding()) {
      // Counted because this loop is what the reconcile above depends on: a record it released
      // reaches the selector cache only if this poll runs for that binding in this same pass.
      // Both skips leave the successor unpolled, so they are the two ways an eviction can be
      // correct in its ordering argument and still cost the activation.
      count_phase(RuntimePhaseCounter::kReadinessPollSkippedByCoordinatorBound);
      state.poll_incomplete_this_round = true;
      break;
    }
    // A different source may leave room in the global bound while this binding's complete
    // source-local prefix is already pending. Do not poll or drain a successor until that older
    // witness retires: drain is destructive, and rejecting the removed successor as
    // backpressured would lose its credit and fail an otherwise valid mixed-source run.
    const auto source_credit = state.coordinator->source_credit_remaining(binding->source());
    if (!source_credit) {
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
      static_cast<void>(cancel_ready_sources(state));
      return false;
    }
    if (*source_credit == 0U) {
      count_phase(RuntimePhaseCounter::kReadinessPollSkippedBySourceCredit);
      state.poll_incomplete_this_round = true;
      continue;
    }
    ReadyRunState::FreshnessBindingSlot* freshness = nullptr;
    if (!state.freshness_binding_index_by_binding.empty()) {
      const std::size_t freshness_index = state.freshness_binding_index_by_binding[binding_index];
      if (freshness_index != std::numeric_limits<std::size_t>::max()) {
        freshness = &state.freshness_bindings[freshness_index];
      }
    }
    bool freshness_poll_complete = false;
    bool direct_continuation_poll_elided = false;
    if (state.direct_chain.continuation_receipt &&
        state.direct_chain.continuation_receipt->readiness_source == binding->source()) {
      direct_continuation_poll_elided = state.channels != nullptr &&
                                        state.channels->validate_direct_continuation_receipt(
                                            *state.direct_chain.continuation_receipt) &&
                                        !binding->poll_target().has_value();
      state.direct_chain.continuation_receipt.reset();
      if (direct_continuation_poll_elided) {
        count_phase(RuntimePhaseCounter::kDirectContinuationReceiptsAccepted);
        count_phase(RuntimePhaseCounter::kDirectContinuationPollElisions);
      }
    }
    const std::optional<PortBindingPollTarget> target =
        direct_continuation_poll_elided ? std::nullopt : binding->poll_target();
    if (!target) {
      count_phase(RuntimePhaseCounter::kReadinessPollSkippedByNoTarget);
    }
    if (target) {
      auto clock = exec.resolve_clock(target->reference_clock);
      auto observed_at =
          clock ? clock->now() : expected<TimePoint, Error>{make_unexpected(clock.error())};
      if (!observed_at || observed_at->clock_id != target->reference_clock) {
        append_diagnostic("RUNTIME_CLOCK_READ_FAILED");
        static_cast<void>(cancel_ready_sources(state));
        return false;
      }
      struct PollInvocation {
        PortBinding* binding{};
        TimePoint observed_at{};
        bool after_contention{};
      } invocation{.binding = binding.get(), .observed_at = *observed_at};
      if (current_execution_lane && *current_execution_lane != target->execution_lane) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
        static_cast<void>(cancel_ready_sources(state));
        return false;
      }
      const auto invoke_poll = [&]() noexcept -> expected<void, Error> {
        if (current_execution_lane) {
          return invocation.after_contention ? binding->poll_after_contention(*observed_at)
                                             : binding->poll(*observed_at);
        }
        count_phase(RuntimePhaseCounter::kReadinessPollDispatches);
        return state.lanes->dispatch(
            target->execution_lane,
            ExecutionLaneTask{
                .context = &invocation,
                .invoke = [](void* state) noexcept -> expected<void, Error> {
                  auto& task = *static_cast<PollInvocation*>(state);
                  return task.after_contention
                             ? task.binding->poll_after_contention(task.observed_at)
                             : task.binding->poll(task.observed_at);
                },
                .callback_context = RuntimeCallbackContext::kDispatcher,
            });
      };
      const bool contention_retry_eligible =
          !state.contention_retry_eligible_by_binding.empty() &&
          state.contention_retry_eligible_by_binding[binding_index] != 0U;
      expected<void, Error> polled = invoke_poll();
      if (polled && contention_retry_eligible && binding->last_poll_lock_contended()) {
        // The nonblocking poll lost the input cache lock to another bounded holder: the channel
        // pump reserving a typed slot, or a sibling owner's post-compute cut probing this input.
        // The published revision stays unobserved, but the wake token that announced it was
        // consumed before this scan and no holder signals again when it releases the lock, so
        // parking now would wait out the whole bounded fallback with a retained sample pending.
        // Measured on graph_intraprocess_docs_pod_1to2_per_sink_latency, that was the residual
        // ~1.06 ms p999 tail. Retry once, waiting out that short critical section under a bounded
        // budget instead; an expired budget or a still-incomplete retry keeps the dirty revision
        // for the next round exactly as before.
        count_phase(RuntimePhaseCounter::kReadinessPollContentionRetries);
        invocation.after_contention = true;
        polled = invoke_poll();
      }
      if (!polled) {
        const auto terminal_diagnostic =
            state.channels != nullptr ? state.channels->terminal_diagnostic() : std::nullopt;
        VertexId vertex{};
        if (terminal_diagnostic) {
          vertex = state.channels->terminal_owner();
        } else {
          const auto receiver = state.coordinator->source_receiver(binding->source());
          vertex = receiver && *receiver < exec.operators().size()
                       ? exec.operators()[*receiver].vertex
                       : VertexId{};
        }
        const std::string_view fallback = channel_terminal_token(terminal_diagnostic);
        const std::string_view token =
            terminal_diagnostic ? fallback : registered_token_or(polled.error().message, fallback);
        append_callback_diagnostic(token,
                                   vertex,
                                   {},
                                   polled.error(),
                                   RuntimeCallbackIdentity{.source = binding->source()});
        static_cast<void>(cancel_ready_sources(state));
        return false;
      }
      if (freshness != nullptr) {
        freshness->ordinary_poll_incomplete_this_round = binding->last_poll_incomplete();
        freshness_poll_complete = !freshness->ordinary_poll_incomplete_this_round;
        state.poll_incomplete_this_round =
            state.poll_incomplete_this_round || !freshness_poll_complete;
      } else if (contention_retry_eligible && binding->last_poll_incomplete()) {
        // An ordinary binding that stayed incomplete after the contention retry still owns a
        // dirty revision; record the round so a missed selection is classified as unpolled.
        state.poll_incomplete_this_round = true;
      }
      if (!current_execution_lane) {
        trace_phase(RuntimePhaseTracePoint::kReadinessPollReturned,
                    kNoRuntimePhaseReceiver,
                    binding->source(),
                    0U);
      }
    }
    if (!target && freshness != nullptr) {
      freshness_poll_complete = true;
    }
    auto ingested = state.coordinator->ingest(*binding, binding->source(), *source_credit);
    if (!ingested || ingested_count > std::numeric_limits<std::size_t>::max() - *ingested) {
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
      static_cast<void>(cancel_ready_sources(state));
      return false;
    }
    ingested_count += *ingested;
    if (freshness != nullptr) {
      freshness->poll_complete_this_round = freshness_poll_complete;
    }
  }
  // An independent activation witness must not be frozen against stale ReuseLast state when
  // the exact Channel source that supplies its selector just declined a nonblocking poll. The
  // witness remains coordinator-owned and is retried on the next bounded round after Channel
  // serialization is available. A skipped poll is deliberately excluded: coordinator/source
  // bounds already preserve their own prefix and may need the pending witness to make progress.
  bool independent_reuse_witness_waits_for_poll = false;
  for (auto& freshness : state.freshness_bindings) {
    if (!freshness.ordinary_poll_incomplete_this_round || freshness.binding == nullptr) {
      continue;
    }
    const SourceId readiness_source = freshness.binding->source();
    independent_reuse_witness_waits_for_poll = std::any_of(
        state.input_selectors.begin(), state.input_selectors.end(), [&](const auto& slot) {
          return slot.freshness_reconcile && slot.readiness_source == readiness_source &&
                 slot.selector.plan().missing == TemporalRecordMissingKind::kReuseLast &&
                 slot.readiness_source != slot.witness_source && slot.input != slot.witness_port &&
                 selector_witness_reference(
                     state, slot.owner, slot.witness_source, slot.witness_port)
                     .has_value();
        });
    if (independent_reuse_witness_waits_for_poll) {
      break;
    }
  }

  // Freshness recovery is a second, cold phase over only the Channel sources that own
  // LatestWithin selectors. The ordinary poll and ingest above establish a complete existing
  // prefix before any retained record is released. Rechecking both coordinator bounds here
  // reserves enough admission credit for every successor that the reconcile promotes.
  for (auto& freshness : state.freshness_bindings) {
    if (!freshness.poll_complete_this_round || freshness.binding == nullptr) {
      continue;
    }
    PortBinding& binding = *freshness.binding;
    const auto source_credit = state.coordinator->source_credit_remaining(binding.source());
    const std::size_t outstanding = state.coordinator->outstanding_count();
    const std::size_t maximum = state.coordinator->max_outstanding();
    if (!source_credit || outstanding > maximum) {
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
      static_cast<void>(cancel_ready_sources(state));
      return false;
    }
    std::size_t refill_credit = std::min(*source_credit, maximum - outstanding);
    std::size_t recovered_count = 0U;
    if (!reconcile_superseded_retention(state, binding.source(), refill_credit, recovered_count)) {
      static_cast<void>(cancel_ready_sources(state));
      return false;
    }
    if (recovered_count == 0U) {
      continue;
    }

    // The exact binding whose selector released a claim must now observe the revision published
    // by that promotion. Waiting is justified only on this source and runs through the same
    // compiler-selected lane serialization used by an ordinary poll.
    const auto target = binding.poll_target();
    if (!target) {
      // The successor's revision remains published for the next round. Treat a transiently
      // absent target like every other incomplete pass instead of converting one missed
      // activation into a failed run.
      count_phase(RuntimePhaseCounter::kReadinessPollSkippedByNoTarget);
      state.poll_incomplete_this_round = true;
      continue;
    }
    auto clock = exec.resolve_clock(target->reference_clock);
    auto observed_at =
        clock ? clock->now() : expected<TimePoint, Error>{make_unexpected(clock.error())};
    if (!observed_at || observed_at->clock_id != target->reference_clock) {
      append_diagnostic("RUNTIME_CLOCK_READ_FAILED");
      static_cast<void>(cancel_ready_sources(state));
      return false;
    }
    struct BlockingPollInvocation {
      PortBinding* binding{};
      TimePoint observed_at{};
    } invocation{.binding = &binding, .observed_at = *observed_at};
    if (!current_execution_lane) {
      count_phase(RuntimePhaseCounter::kReadinessPollDispatches);
    }
    expected<void, Error> polled;
    if (current_execution_lane) {
      if (*current_execution_lane != target->execution_lane) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
        static_cast<void>(cancel_ready_sources(state));
        return false;
      }
      polled = binding.poll_blocking(*observed_at);
    } else {
      polled =
          state.lanes->dispatch(target->execution_lane,
                                ExecutionLaneTask{
                                    .context = &invocation,
                                    .invoke = [](void* state) noexcept -> expected<void, Error> {
                                      auto& task = *static_cast<BlockingPollInvocation*>(state);
                                      return task.binding->poll_blocking(task.observed_at);
                                    },
                                    .callback_context = RuntimeCallbackContext::kDispatcher,
                                });
    }
    if (!polled) {
      const auto terminal_diagnostic =
          state.channels != nullptr ? state.channels->terminal_diagnostic() : std::nullopt;
      VertexId vertex{};
      if (terminal_diagnostic) {
        vertex = state.channels->terminal_owner();
      } else {
        const auto receiver = state.coordinator->source_receiver(binding.source());
        vertex = receiver && *receiver < exec.operators().size()
                     ? exec.operators()[*receiver].vertex
                     : VertexId{};
      }
      const std::string_view fallback = channel_terminal_token(terminal_diagnostic);
      const std::string_view token =
          terminal_diagnostic ? fallback : registered_token_or(polled.error().message, fallback);
      append_callback_diagnostic(
          token, vertex, {}, polled.error(), RuntimeCallbackIdentity{.source = binding.source()});
      static_cast<void>(cancel_ready_sources(state));
      return false;
    }
    if (!current_execution_lane) {
      trace_phase(RuntimePhaseTracePoint::kReadinessPollReturned,
                  kNoRuntimePhaseReceiver,
                  binding.source(),
                  0U);
    }
    if (binding.last_poll_incomplete()) {
      // poll_blocking() waits for subscriber cache locks, but whole-owner polling can still
      // decline on a contended external-import transport lock or an in-progress typed ingress.
      // The promoted successor remains published and is observed by the next dispatch round.
      state.poll_incomplete_this_round = true;
      continue;
    }

    const auto recovery_credit = state.coordinator->source_credit_remaining(binding.source());
    if (!recovery_credit || *recovery_credit < recovered_count) {
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
      static_cast<void>(cancel_ready_sources(state));
      return false;
    }
    auto ingested = state.coordinator->ingest(binding, binding.source(), *recovery_credit);
    if (!ingested || *ingested < recovered_count ||
        ingested_count > std::numeric_limits<std::size_t>::max() - *ingested) {
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
      static_cast<void>(cancel_ready_sources(state));
      return false;
    }
    ingested_count += *ingested;
  }
  state.round_ingested_count = static_cast<std::uint32_t>(ingested_count);
  if (independent_reuse_witness_waits_for_poll) {
    // This pre-existing one-round gate protects a ReuseLast handoff whose own source could not be
    // observed. No prefix candidate has been frozen yet, so every coordinator witness remains
    // available for the retry.
    return true;
  }

  // A non-trigger side input may have promoted the oldest member of a source-sequence prefix
  // while keeping its newest causal member reserved as the final refill target. Fence only that
  // owning receiver, expose one oldest candidate per fenced input, and leave unrelated owners in
  // the ordinary global ordering. The vector was sized at startup, so rebuilding it never grows
  // storage in the dispatch hot path.
  const auto rebuild_freshness_input_fences = [&]() noexcept {
    state.freshness_input_fences.clear();
    for (const auto& slot : state.input_selectors) {
      if (!slot.freshness_refill_target || !slot.nontrigger_prerequisite) {
        continue;
      }
      state.freshness_input_fences.push_back(
          RuntimeCandidateInputFence{slot.owner, slot.readiness_source, slot.input});
      state.poll_incomplete_this_round = true;
    }
  };
  if (state.freshness_evicted_this_round || !state.freshness_input_fences.empty()) {
    rebuild_freshness_input_fences();
  }
  bool freeze_failed = false;
  const auto freeze_one_eligible = [&]() -> bool {
    while (state.coordinator->pending_count() != 0U) {
      const bool constrained = !state.freshness_input_fences.empty();
      auto candidate =
          constrained ? state.coordinator->freeze_next_constrained(state.freshness_input_fences)
                      : state.coordinator->freeze_next();
      if (!candidate) {
        if (candidate.error().code == ErrorCode::kNotReady && constrained) {
          return false;
        }
        HOLOSCAN_LOG_ERROR("Runtime candidate freeze failed with {}: pending={} outstanding={}",
                           error_code_name(candidate.error().code),
                           state.coordinator->pending_count(),
                           state.coordinator->outstanding_count());
        append_callback_diagnostic(candidate.error().code == ErrorCode::kResourceExhausted
                                       ? std::string_view{"RUNTIME_TAG_CAPACITY_EXHAUSTED"}
                                       : std::string_view{"RUNTIME_READINESS_BINDING_UNAVAILABLE"},
                                   {},
                                   "candidate_freeze",
                                   candidate.error(),
                                   {});
        freeze_failed = true;
        return false;
      }
      if (!constrained) {
        if (!ingest_channel_candidate(exec, state, *candidate)) {
          freeze_failed = true;
          return false;
        }
      } else {
        ReadyRunState::InputSelectorSlot* fenced_slot = nullptr;
        std::optional<TemporalSampleRecord> fenced_sample;
        bool reached_prerequisite = false;
        bool reached_final_target = false;
        if (candidate->witness.kind == RuntimeEventKind::kInputArrival &&
            candidate->candidate.receiver_index < exec.operators().size()) {
          const VertexId owner = exec.operators()[candidate->candidate.receiver_index].vertex;
          const auto found = std::find_if(
              state.input_selectors.begin(),
              state.input_selectors.end(),
              [&](const ReadyRunState::InputSelectorSlot& slot) {
                return slot.owner == owner && slot.readiness_source == candidate->witness.source &&
                       slot.input == candidate->witness.port && slot.freshness_refill_target &&
                       slot.nontrigger_prerequisite;
              });
          if (found != state.input_selectors.end()) {
            const TemporalSampleRecord final_target = *found->freshness_refill_target;
            const std::optional<TemporalSampleRecord> expected_prerequisite =
                found->freshness_refill_prerequisite;
            auto sample = state.channels->temporal_sample(candidate->witness.port,
                                                          candidate->witness.source_sequence);
            const auto same_identity = [](const TemporalSampleRecord& left,
                                          const TemporalSampleRecord& right) noexcept {
              return left.producer_epoch == right.producer_epoch &&
                     left.source_sequence == right.source_sequence &&
                     left.arrival_sequence == right.arrival_sequence;
            };
            const bool newer_epoch = sample && sample->producer_epoch > final_target.producer_epoch;
            const bool at_or_before_final_target =
                sample && sample->producer_epoch == final_target.producer_epoch &&
                (same_identity(*sample, final_target) ||
                 is_strictly_newer_sample(final_target, *sample));
            const bool at_or_before_expected =
                !expected_prerequisite ||
                (sample && sample->producer_epoch == expected_prerequisite->producer_epoch &&
                 (same_identity(*sample, *expected_prerequisite) ||
                  is_strictly_newer_sample(*expected_prerequisite, *sample)));
            if (!sample ||
                (!newer_epoch && (!at_or_before_final_target || !at_or_before_expected))) {
              append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE", found->owner);
              freeze_failed = true;
              return false;
            }
            fenced_sample = *sample;
            fenced_slot = &*found;
            reached_prerequisite =
                expected_prerequisite && same_identity(*sample, *expected_prerequisite);
            reached_final_target = same_identity(*sample, final_target);
          }
        }
        if (!ingest_channel_candidate(
                exec, state, *candidate, fenced_sample ? &*fenced_sample : nullptr)) {
          freeze_failed = true;
          return false;
        }
        if (fenced_slot != nullptr) {
          // A producer-epoch reset clears both markers inside ingest_channel_candidate(). For the
          // same epoch, advance them only after the exact prerequisite has entered the cache.
          if (reached_prerequisite) {
            fenced_slot->freshness_refill_prerequisite.reset();
          }
          if (reached_final_target) {
            fenced_slot->freshness_refill_prerequisite.reset();
            fenced_slot->freshness_refill_target.reset();
          }
          rebuild_freshness_input_fences();
        }
      }
      if (!channel_candidate_is_eligible(exec, state, *candidate)) {
        auto retired = retire_frozen_source(state, *candidate);
        if (!retired) {
          append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
          freeze_failed = true;
          return false;
        }
        continue;
      }
      // Serial-baseline dispatch exposes exactly the selector state established by this
      // witness. Leaving later accepted events in the coordinator prevents a successor or
      // timeout from closing a window before the earlier candidate has committed or rolled back.
      state.selection.candidates.push_back(candidate->candidate);
      state.selection.frozen_candidates.push_back(std::move(*candidate));
      count_phase(RuntimePhaseCounter::kCandidatesFrozen);
      trace_phase(RuntimePhaseTracePoint::kCandidateFrozen,
                  state.selection.frozen_candidates.back().candidate.receiver_index,
                  state.selection.frozen_candidates.back().witness.source,
                  state.selection.frozen_candidates.back().witness.source_sequence);
      return true;
    }
    return false;
  };

  bool candidate_ready = freeze_one_eligible();
  if (!candidate_ready && !freeze_failed) {
    if (plan_->executable_factories.test_late_readiness_drain_control != nullptr) {
      plan_->executable_factories.test_late_readiness_drain_control->before_late_drain();
    }
    if (!enqueue_due_window_timeouts(exec, state)) {
      freeze_failed = true;
    } else {
      bool late_ingested_any = false;
      for (auto& binding : state.bindings) {
        if (state.coordinator->outstanding_count() == state.coordinator->max_outstanding()) {
          break;
        }
        const auto source_credit = state.coordinator->source_credit_remaining(binding->source());
        if (!source_credit) {
          append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
          freeze_failed = true;
          break;
        }
        if (*source_credit == 0U) {
          continue;
        }
        auto ingested = state.coordinator->ingest(*binding, binding->source(), *source_credit);
        if (!ingested) {
          append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
          freeze_failed = true;
          break;
        }
        late_ingested_any = late_ingested_any || *ingested != 0U;
      }
      const bool late_reuse_witness_requires_rescan =
          !freeze_failed && late_ingested_any &&
          std::any_of(
              state.input_selectors.begin(), state.input_selectors.end(), [&](const auto& slot) {
                return slot.freshness_reconcile &&
                       slot.selector.plan().missing == TemporalRecordMissingKind::kReuseLast &&
                       slot.nontrigger_prerequisite &&
                       selector_witness_reference(
                           state, slot.owner, slot.witness_source, slot.witness_port)
                           .has_value();
              });
      if (late_reuse_witness_requires_rescan) {
        // This witness arrived after the ordinary Channel poll and freshness reconciliation.
        // Preserve its exact coordinator record and finite credit so the next bounded round can
        // rerun both phases before evaluating ReuseLast state.
        return true;
      }
      candidate_ready = !freeze_failed && freeze_one_eligible();
    }
  }
  if (freeze_failed) {
    static_cast<void>(cancel_ready_sources(state));
    return false;
  }
  if (!candidate_ready) {
    return true;
  }
  const RuntimeCallbackContextScope callback_scope{RuntimeCallbackContext::kDispatcher};
  auto frozen = state.dispatcher->freeze(state.selection.candidates);
  if (!frozen) {
    static_cast<void>(cancel_ready_sources(state));
    append_diagnostic("READY_POLICY_PLAN_INVALID");
    return false;
  }
  // Health-suppressed candidates reserve their receiver-local Tag before
  // eligibility is known, but never enter compute. Release source credit now
  // so a later direct-runtime freeze cannot inherit a phantom outstanding
  // event; the Tag itself remains consumed.
  for (auto& candidate : state.selection.frozen_candidates) {
    if (candidate.candidate.clock_health_disposition == EndpointClockHealthDisposition::kSkip &&
        !retire_frozen_source(state, candidate)) {
      static_cast<void>(cancel_ready_sources(state));
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
      return false;
    }
  }
  return true;
}

// Causal continuation validation and fallback form one ordered frontier transition.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] RunSessionState::CausalFrontierDisposition RunSessionState::freeze_causal_successor(
    GraphExec& exec, ReadyRunState& state, std::size_t producer_receiver,
    TagTime parent_tag_time) noexcept {
  constexpr std::size_t kNoReceiver = std::numeric_limits<std::size_t>::max();
  if (!state.direct_chain.continuation_receipt ||
      producer_receiver >= state.direct_chain.successor_by_receiver.size() ||
      state.direct_chain.successor_by_receiver[producer_receiver] == kNoReceiver) {
    return CausalFrontierDisposition::kFallback;
  }
  count_phase(RuntimePhaseCounter::kPartitionCausalFrontierAttempts);
  const auto fallback = [&]() noexcept {
    count_phase(RuntimePhaseCounter::kPartitionCausalFrontierFallbacks);
    return CausalFrontierDisposition::kFallback;
  };
  if (stop_requested_.load(std::memory_order_acquire) || lifecycle_control_->has_pending() ||
      state.channels == nullptr || state.dispatcher->active_count() != 0U ||
      state.coordinator->pending_count() != 0U || state.coordinator->outstanding_count() != 0U ||
      !state.direct_chain.root_source || !state.direct_chain.root_clock.valid() ||
      state.direct_chain.root_binding == nullptr || !parent_tag_time.valid()) {
    return fallback();
  }

  const RuntimeChannelDirectContinuationReceipt receipt = *state.direct_chain.continuation_receipt;
  const std::size_t successor_receiver =
      state.direct_chain.successor_by_receiver[producer_receiver];
  PortBinding* const successor_binding =
      producer_receiver < state.direct_chain.binding_by_producer.size()
          ? state.direct_chain.binding_by_producer[producer_receiver]
          : nullptr;
  const auto receipt_receiver = state.coordinator->source_receiver(receipt.readiness_source);
  if (successor_receiver >= exec.operators().size() || !receipt_receiver ||
      *receipt_receiver != successor_receiver ||
      receipt.destination != exec.operators()[successor_receiver].vertex ||
      successor_binding == nullptr || successor_binding->source() != receipt.readiness_source ||
      successor_binding->poll_target().has_value() ||
      !state.channels->validate_direct_continuation_receipt(receipt)) {
    return fallback();
  }

  // The compiler-proven chain gives the receipt-proven successor the next
  // microstep at its parent's logical coordinate. Confirm that the root's
  // next scheduled coordinate is strictly later before ingesting the
  // receipt; otherwise leave the receipt intact for the ordinary
  // deterministic READY-set path.
  if (producer_receiver == state.direct_chain.root_receiver || !state.direct_chain.next_root_wake) {
    state.direct_chain.next_root_wake = state.direct_chain.root_binding->next_wake();
  }
  if (!state.direct_chain.next_root_wake) {
    return fallback();
  }
  if (state.direct_chain.next_root_wake->clock_id != state.direct_chain.root_clock ||
      state.direct_chain.next_root_wake->timestamp_ns <= parent_tag_time.logical_ns) {
    return fallback();
  }

  // The cold chain proof binds the sole successor source. Keep the common
  // health/terminal prelude, then ingest that exact binding instead of
  // rescanning every partition source and validating the receipt twice.
  if (std::ranges::any_of(state.selection.frozen_candidates,
                          [](const auto& candidate) { return !candidate.source_retired; })) {
    append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
    static_cast<void>(cancel_ready_sources(state));
    return CausalFrontierDisposition::kFailed;
  }
  state.selection.candidates.clear();
  state.selection.frozen_candidates.clear();
  expected<std::size_t, EndpointClockHealthError> expired{0U};
  if (state.clock_health->receiver_count() != 0U) {
    const std::lock_guard clock_health_lock(clock_health_mutex_);
    expired = state.clock_health->expire_due();
  }
  if (!expired) {
    append_diagnostic(expired.error().token.empty() ? std::string_view{"RUNTIME_CLOCK_READ_FAILED"}
                                                    : expired.error().token);
    return CausalFrontierDisposition::kFailed;
  }
  auto channel_status = state.channels->status();
  if (!channel_status) {
    append_callback_diagnostic(channel_terminal_token(state.channels->terminal_diagnostic()),
                               state.channels->terminal_owner(),
                               {},
                               channel_status.error(),
                               {});
    return CausalFrontierDisposition::kFailed;
  }
  const auto source_credit = state.coordinator->source_credit_remaining(receipt.readiness_source);
  if (!source_credit || *source_credit == 0U) {
    append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE",
                      exec.operators()[successor_receiver].vertex,
                      "partition_causal_frontier_credit");
    static_cast<void>(cancel_ready_sources(state));
    return CausalFrontierDisposition::kFailed;
  }
  // Consume the receipt before freeze_causal_next() because that operation assumes exclusive
  // authority over the sole pending event. Ingress is destructive: any failure below is terminal,
  // and cancel_ready_sources() retires coordinator credit rather than attempting an ordinary
  // poll that cannot recover the already-accepted continuation witness.
  state.direct_chain.continuation_receipt.reset();
  count_phase(RuntimePhaseCounter::kDirectContinuationReceiptsAccepted);
  count_phase(RuntimePhaseCounter::kDirectContinuationPollElisions);
  const auto ingested =
      state.coordinator->ingest(*successor_binding, receipt.readiness_source, *source_credit);
  if (!ingested || *ingested != 1U) {
    append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH",
                      exec.operators()[successor_receiver].vertex,
                      "partition_causal_frontier_ingress");
    static_cast<void>(cancel_ready_sources(state));
    return CausalFrontierDisposition::kFailed;
  }
  const bool force_causal_freeze_failure =
      plan_->executable_factories.test_direct_continuation_control != nullptr &&
      plan_->executable_factories.test_direct_continuation_control->consume_causal_freeze_failure();
  auto candidate = state.coordinator->freeze_causal_next(
      force_causal_freeze_failure ? SourceId{} : receipt.readiness_source, parent_tag_time);
  if (!candidate) {
    append_callback_diagnostic(candidate.error().code == ErrorCode::kResourceExhausted
                                   ? std::string_view{"RUNTIME_TAG_CAPACITY_EXHAUSTED"}
                                   : std::string_view{"RUNTIME_READINESS_BINDING_UNAVAILABLE"},
                               exec.operators()[successor_receiver].vertex,
                               "partition_causal_frontier_freeze",
                               candidate.error(),
                               {});
    const std::size_t pending_before_cancel = state.coordinator->pending_count();
    const std::size_t outstanding_before_cancel = state.coordinator->outstanding_count();
    const bool retired_all = cancel_ready_sources(state);
    if (force_causal_freeze_failure) {
      plan_->executable_factories.test_direct_continuation_control
          ->record_causal_freeze_failure_cleanup(
              pending_before_cancel,
              outstanding_before_cancel,
              state.coordinator->pending_count(),
              state.coordinator->outstanding_count(),
              state.direct_chain.continuation_receipt.has_value(),
              retired_all);
    }
    return CausalFrontierDisposition::kFailed;
  }
  if (!ingest_channel_candidate(exec, state, *candidate)) {
    static_cast<void>(cancel_ready_sources(state));
    return CausalFrontierDisposition::kFailed;
  }
  if (!channel_candidate_is_eligible(exec, state, *candidate)) {
    if (!retire_frozen_source(state, *candidate)) {
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
      static_cast<void>(cancel_ready_sources(state));
      return CausalFrontierDisposition::kFailed;
    }
    return CausalFrontierDisposition::kHandled;
  }
  state.selection.candidates.push_back(candidate->candidate);
  state.selection.frozen_candidates.push_back(std::move(*candidate));
  count_phase(RuntimePhaseCounter::kCandidatesFrozen);
  trace_phase(RuntimePhaseTracePoint::kCandidateFrozen,
              state.selection.frozen_candidates.back().candidate.receiver_index,
              state.selection.frozen_candidates.back().witness.source,
              state.selection.frozen_candidates.back().witness.source_sequence);
  {
    const RuntimeCallbackContextScope callback_scope{RuntimeCallbackContext::kDispatcher};
    auto frozen = state.dispatcher->freeze(state.selection.candidates);
    if (!frozen) {
      static_cast<void>(cancel_ready_sources(state));
      append_diagnostic("READY_POLICY_PLAN_INVALID");
      return CausalFrontierDisposition::kFailed;
    }
  }
  for (auto& frozen : state.selection.frozen_candidates) {
    if (frozen.candidate.clock_health_disposition == EndpointClockHealthDisposition::kSkip &&
        !retire_frozen_source(state, frozen)) {
      static_cast<void>(cancel_ready_sources(state));
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
      return CausalFrontierDisposition::kFailed;
    }
  }
  if (state.dispatcher->active_count() == 0U) {
    if (state.direct_chain.continuation_receipt || state.coordinator->pending_count() != 0U ||
        state.coordinator->outstanding_count() != 0U) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH",
                        exec.operators()[successor_receiver].vertex,
                        "partition_causal_frontier_suppression");
      static_cast<void>(cancel_ready_sources(state));
      return CausalFrontierDisposition::kFailed;
    }
    return CausalFrontierDisposition::kHandled;
  }
  if (state.selection.frozen_candidates.size() != 1U ||
      state.selection.frozen_candidates.front().candidate.receiver_index != successor_receiver ||
      state.selection.frozen_candidates.front().witness.source != receipt.readiness_source) {
    append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH",
                      exec.operators()[successor_receiver].vertex,
                      "partition_causal_frontier");
    static_cast<void>(cancel_ready_sources(state));
    return CausalFrontierDisposition::kFailed;
  }
  count_phase(RuntimePhaseCounter::kPartitionCausalFrontierAccepted);
  return CausalFrontierDisposition::kAccepted;
}

[[nodiscard]] const RunSessionState::ReadyRunState::CompletionReceiverState*
RunSessionState::completion_receiver_state(const ReadyRunState& state,
                                           const std::size_t receiver) noexcept {
  const auto found =
      std::lower_bound(state.completion_receivers.begin(),
                       state.completion_receivers.end(),
                       receiver,
                       [](const ReadyRunState::CompletionReceiverState& candidate,
                          const std::size_t value) { return candidate.receiver < value; });
  return found != state.completion_receivers.end() && found->receiver == receiver ? &*found
                                                                                  : nullptr;
}

[[nodiscard]] RunSessionState::ReadyRunState::InputTerminalDrainState*
RunSessionState::input_terminal_receiver_state(ReadyRunState& state,
                                               const std::size_t receiver) noexcept {
  const auto found =
      std::lower_bound(state.input_terminal_receivers.begin(),
                       state.input_terminal_receivers.end(),
                       receiver,
                       [](const ReadyRunState::InputTerminalDrainState& candidate,
                          const std::size_t value) { return candidate.receiver < value; });
  return found != state.input_terminal_receivers.end() && found->receiver == receiver ? &*found
                                                                                      : nullptr;
}

[[nodiscard]] const RunSessionState::ReadyRunState::InputTerminalDrainState*
RunSessionState::input_terminal_receiver_state(const ReadyRunState& state,
                                               const std::size_t receiver) noexcept {
  const auto found =
      std::lower_bound(state.input_terminal_receivers.begin(),
                       state.input_terminal_receivers.end(),
                       receiver,
                       [](const ReadyRunState::InputTerminalDrainState& candidate,
                          const std::size_t value) { return candidate.receiver < value; });
  return found != state.input_terminal_receivers.end() && found->receiver == receiver ? &*found
                                                                                      : nullptr;
}

[[nodiscard]] bool RunSessionState::cancel_completion_receiver(ReadyRunState& state,
                                                               std::size_t receiver) noexcept {
  if (plan_->executable_factories.test_operator_completion_final_entry_control != nullptr) {
    // Source-tree tests retain their already-bounded finite witnesses so the lane-side final
    // entry fence, rather than eager receiver cancellation, rejects each stale selection.
    return true;
  }
  auto canceled_dispatcher = state.dispatcher->cancel_receiver(receiver);
  if (!canceled_dispatcher) {
    return false;
  }
  std::size_t retired = 0U;
  for (auto& candidate : state.selection.frozen_candidates) {
    if (candidate.candidate.receiver_index != receiver || candidate.source_retired) {
      continue;
    }
    if (!retire_frozen_source(state, candidate)) {
      return false;
    }
    ++retired;
  }
  if (retired != *canceled_dispatcher) {
    return false;
  }
  auto canceled = state.coordinator->cancel_receiver(receiver, state.restart.events);
  if (!canceled) {
    return false;
  }
  for (std::size_t index = 0U; index < *canceled; ++index) {
    const RuntimeEvent& event = state.restart.events[index];
    PortBinding* const binding = physical_binding_for(state, event.source);
    if (binding != nullptr && !binding->retire(event.source_sequence)) {
      return false;
    }
  }
  for (auto& binding : state.bindings) {
    const SourceId source = binding->source();
    if (!source) {
      continue;
    }
    const auto source_receiver = state.coordinator->source_receiver(source);
    if (source_receiver && *source_receiver == receiver) {
      binding->cancel();
    }
  }
  return true;
}

[[nodiscard]] bool RunSessionState::begin_zero_completion_closures(ReadyRunState& state) noexcept {
  for (const ReadyRunState::CompletionReceiverState& completion : state.completion_receivers) {
    OperatorCompletionProgressRecord& progress =
        operator_completion_progress_[completion.completion_index];
    OperatorCompletionPhase& phase = operator_completion_phases_[completion.completion_index];
    if (progress.configured_limit != 0U || phase != OperatorCompletionPhase::kOpen) {
      continue;
    }
    if (!cancel_completion_receiver(state, completion.receiver)) {
      return false;
    }
    progress.limit_reached = true;
    phase = OperatorCompletionPhase::kClosing;
    state.active_completion_receivers.push_back(completion);
  }
  return true;
}

template <bool HandlesTerminal>
[[nodiscard]] expected<void, Error> RunSessionState::retire_frozen_source_impl(
    ReadyRunState& state, FrozenRuntimeCandidate& candidate) noexcept {
  if (candidate.source_retired) {
    return make_unexpected(Error{ErrorCode::kStaleEndpoint});
  }
  auto retired = state.coordinator->retire(candidate);
  if (!retired) {
    return make_unexpected(std::move(retired.error()));
  }
  // Once coordinator credit is returned, retrying this candidate would be a
  // double retirement even if the binding reports a terminal failure.
  candidate.source_retired = true;
  PortBinding* const binding = physical_binding_for(state, candidate.witness.source);
  if constexpr (!HandlesTerminal) {
    return binding == nullptr ? expected<void, Error>{}
                              : binding->retire(candidate.witness.source_sequence);
  }
  if (candidate.witness.kind != RuntimeEventKind::kInputEndOfStream) {
    return binding == nullptr ? expected<void, Error>{}
                              : binding->retire(candidate.witness.source_sequence);
  }
  auto physical_retired = binding == nullptr ? expected<void, Error>{}
                                             : binding->retire(candidate.witness.source_sequence);
  if (!physical_retired) {
    return physical_retired;
  }
  ReadyRunState::InputTerminalDrainState* const terminal =
      input_terminal_receiver_state(state, candidate.candidate.receiver_index);
  if (terminal == nullptr || terminal->phase != InputTerminalDrainPhase::kOpen) {
    return {};
  }
  // A plain OnEach/OnAnyInput trigger pin need not have an explicit temporal selector, so the
  // selector inventory cannot prove that every physical input generation is terminal. The
  // binding closes only after retiring one EOS witness for every materialized input source;
  // Channel then performs the stronger retained-work proof before outputs are closed.
  if (binding != nullptr && binding->closed()) {
    terminal->phase = InputTerminalDrainPhase::kDraining;
    state.active_input_terminal_receivers.push_back(candidate.candidate.receiver_index);
  }
  return {};
}

[[nodiscard]] expected<void, Error> RunSessionState::retire_nonterminal_frozen_source(
    ReadyRunState& state, FrozenRuntimeCandidate& candidate) noexcept {
  return retire_frozen_source_impl<false>(state, candidate);
}

[[nodiscard]] std::optional<TimePoint> RunSessionState::selector_witness_reference(
    ReadyRunState& state, const VertexId owner, const SourceId witness_source,
    const PortId witness_port) noexcept {
  return state.coordinator != nullptr
             ? state.coordinator->oldest_pending_nonterminal_trigger_reference(
                   owner, witness_source, witness_port)
             : std::nullopt;
}

[[nodiscard]] bool RunSessionState::input_arrival_can_activate(
    const TemporalTriggerPlanRecord& trigger, const PredicateTriggerPlanRecord* const predicate,
    const PortId input) noexcept {
  if (trigger.kind == TemporalRecordTriggerKind::kEach) {
    return input == trigger.trigger_input;
  }
  if (trigger.kind == TemporalRecordTriggerKind::kPredicate) {
    return predicate != nullptr &&
           predicate->source == PredicateEvaluationSourceKind::kInputUpdate &&
           input == predicate->driver;
  }
  return trigger.kind == TemporalRecordTriggerKind::kAnyInput ||
         trigger.kind == TemporalRecordTriggerKind::kAligned ||
         trigger.kind == TemporalRecordTriggerKind::kWindow;
}

[[nodiscard]] bool RunSessionState::terminal_successor_ready(const ReadyRunState& state,
                                                             VertexId owner,
                                                             bool include_aligned) const noexcept {
  const bool bracketed_ready =
      std::any_of(state.input_selectors.begin(),
                  state.input_selectors.end(),
                  [&](const ReadyRunState::InputSelectorSlot& slot) {
                    return slot.owner == owner && !slot.aligned_member &&
                           slot.end_of_stream_received && slot.end_of_stream_decision_pending &&
                           slot.selector.plan().selector == InputSelectorKind::kBracketedWindow &&
                           slot.selector.terminal_decision_ready();
                  });
  if (bracketed_ready || !include_aligned) {
    return bracketed_ready;
  }
  const auto aligned = std::find_if(
      state.aligned_selectors.begin(), state.aligned_selectors.end(), [&](const auto& slot) {
        return slot.owner == owner && slot.selector.terminal_decision_ready();
      });
  return aligned != state.aligned_selectors.end();
}

[[nodiscard]] expected<void, Error> RunSessionState::retire_or_rearm_terminal_source(
    ReadyRunState& state, FrozenRuntimeCandidate& frozen, VertexId owner,
    bool include_aligned) noexcept {
  if (frozen.witness.kind != RuntimeEventKind::kInputEndOfStream ||
      !terminal_successor_ready(state, owner, include_aligned)) {
    return retire_frozen_source(state, frozen);
  }
  if (frozen.source_retired || state.dispatcher->active_count() != 0U ||
      state.selection.frozen_candidates.size() != 1U ||
      &state.selection.frozen_candidates.front() != &frozen) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  auto rearmed = state.coordinator->rearm_terminal_successor(frozen);
  if (!rearmed) {
    return make_unexpected(std::move(rearmed.error()));
  }
  state.selection.candidates.clear();
  state.selection.candidates.push_back(frozen.candidate);
  auto frozen_set = state.dispatcher->freeze(state.selection.candidates);
  if (!frozen_set) {
    return make_unexpected(std::move(frozen_set.error()));
  }
  count_phase(RuntimePhaseCounter::kCandidatesFrozen);
  trace_phase(RuntimePhaseTracePoint::kCandidateFrozen,
              frozen.candidate.receiver_index,
              frozen.witness.source,
              frozen.witness.source_sequence);
  return {};
}

[[nodiscard]] expected<bool, Error> RunSessionState::advance_owner_output_drain(
    ReadyRunState& state, const RuntimeOperatorInstance& instance) noexcept {
  if (tensor_resources_ != nullptr) {
    auto close_ready = tensor_resources_->preflight_owner_terminal_close(instance.operator_id);
    if (!close_ready) {
      return make_unexpected(std::move(close_ready).error());
    }
    if (!*close_ready) {
      return false;
    }
  }
  if (state.channels != nullptr) {
    auto closed = state.channels->close_owner_outputs(instance.vertex);
    if (!closed) {
      const ErrorCode code = closed.error().code;
      if (code == ErrorCode::kNotReady || code == ErrorCode::kBackpressured) {
        return false;
      }
      return make_unexpected(std::move(closed).error());
    }
  }
  if (tensor_resources_ != nullptr) {
    auto retired = tensor_resources_->advance_owner_terminal_retirement(instance.operator_id);
    if (!retired) {
      return make_unexpected(std::move(retired).error());
    }
    if (!*retired) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool RunSessionState::advance_completion_closures(GraphExec& exec,
                                                                ReadyRunState& state) noexcept {
  if (state.active_completion_receivers.empty()) {
    return true;
  }
  // Compact pending work in place only after giving every active receiver one attempt. This
  // prevents one backpressured owner from starving independent terminal closures.
  std::size_t pending_count = 0U;
  for (const ReadyRunState::CompletionReceiverState& completion :
       state.active_completion_receivers) {
    if (completion.receiver >= exec.operators().size() ||
        completion.completion_index >= operator_completion_progress_.size() ||
        completion.completion_index >= operator_completion_phases_.size() ||
        operator_completion_phases_[completion.completion_index] !=
            OperatorCompletionPhase::kClosing) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
      return false;
    }
    const RuntimeOperatorInstance& instance = exec.operators()[completion.receiver];
    auto drained = advance_owner_output_drain(state, instance);
    if (!drained) {
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE",
                        instance.vertex,
                        "compute_call_completion_end_of_stream");
      return false;
    }
    if (!*drained) {
      state.active_completion_receivers[pending_count++] = completion;
      continue;
    }
    operator_completion_phases_[completion.completion_index] = OperatorCompletionPhase::kClosed;
  }
  state.active_completion_receivers.resize(pending_count);
  return true;
}

[[nodiscard]] bool RunSessionState::advance_input_terminal_drains(GraphExec& exec,
                                                                  ReadyRunState& state) noexcept {
  if (state.active_input_terminal_receivers.empty()) {
    return true;
  }
  // Give every active receiver one bounded drain attempt before compacting pending work.
  // Compact pending work in place only after giving every active receiver one attempt. This
  // prevents one retained input generation from starving independent terminal drains.
  std::size_t pending_count = 0U;
  for (const std::size_t receiver : state.active_input_terminal_receivers) {
    ReadyRunState::InputTerminalDrainState* const terminal =
        input_terminal_receiver_state(state, receiver);
    if (receiver >= exec.operators().size() || !owns_receiver(state, receiver) ||
        state.channels == nullptr || terminal == nullptr ||
        terminal->phase != InputTerminalDrainPhase::kDraining) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
      return false;
    }
    const RuntimeOperatorInstance& instance = exec.operators()[receiver];
    auto cut = state.channels->flush_process_local_activation_cut();
    if (!cut) {
      const ErrorCode code = cut.error().code;
      if (code == ErrorCode::kNotReady || code == ErrorCode::kBackpressured) {
        state.active_input_terminal_receivers[pending_count++] = receiver;
        continue;
      }
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE",
                        instance.vertex,
                        "terminal_input_activation_cut");
      return false;
    }
    auto selector_history_released = release_terminal_selector_history(state, instance.vertex);
    if (!selector_history_released) {
      append_diagnostic(
          "RUNTIME_READINESS_BINDING_UNAVAILABLE", instance.vertex, "terminal_selector_history");
      return false;
    }
    auto drained = state.channels->retire_terminal_input_generation(instance.vertex);
    if (!drained) {
      append_diagnostic(
          "RUNTIME_READINESS_BINDING_UNAVAILABLE", instance.vertex, "terminal_input_generation");
      return false;
    }
    if (!*drained) {
      state.active_input_terminal_receivers[pending_count++] = receiver;
      continue;
    }
    // A closed external-import generation can still own an admitted lifecycle replacement.
    // Retire that input generation now, but preserve the Operator's output authority until the
    // final replacement generation reaches EOS. The successful restart path reopens this input
    // terminal phase alongside its newly published generation.
    if (!state.channels->awaits_external_restart(instance.vertex)) {
      auto drained_outputs = advance_owner_output_drain(state, instance);
      if (!drained_outputs) {
        append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE",
                          instance.vertex,
                          "terminal_input_end_of_stream");
        return false;
      }
      if (!*drained_outputs) {
        state.active_input_terminal_receivers[pending_count++] = receiver;
        continue;
      }
    }
    terminal->phase = InputTerminalDrainPhase::kClosed;
    state.closed_input_terminal_receivers.push_back(receiver);
  }
  state.active_input_terminal_receivers.resize(pending_count);
  return true;
}

template expected<void, Error> RunSessionState::retire_frozen_source_impl<true>(
    ReadyRunState& state, FrozenRuntimeCandidate& candidate) noexcept;

}  // namespace holoscan::detail
