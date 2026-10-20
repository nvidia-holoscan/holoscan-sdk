/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <string>

#include "compiler/execution_plan_lookup.hpp"
#include "run_session_internal.hpp"
#include "temporal_depth_publication.hpp"

namespace holoscan::detail {

// Dependency-scoped restart fencing, publication, and rollback.

// Freeze the exact lifecycle-member and Channel-dependent receiver closure, including saved
// clock phases, before any restart fence mutates dispatcher or transport state.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool RunSessionState::prepare_runtime_restart_scope(
    GraphExec& exec, ReadyRunState& state, std::span<const OperatorInstanceId> members,
    std::span<const VertexId> member_vertices) noexcept {
  if (members.empty() || members.size() != member_vertices.size()) {
    return false;
  }
  state.restart.receivers.clear();
  state.restart.member_vertices.clear();
  state.restart.vertices.clear();
  const auto append_affected = [&](std::size_t receiver, VertexId vertex, bool lifecycle_member) {
    if (receiver >= exec.operators().size() || !vertex ||
        std::find(state.restart.receivers.begin(), state.restart.receivers.end(), receiver) !=
            state.restart.receivers.end()) {
      return false;
    }
    state.restart.receivers.push_back(receiver);
    state.restart.vertices.push_back(vertex);
    if (lifecycle_member) {
      state.restart.member_vertices.push_back(vertex);
    }
    return true;
  };
  for (const OperatorInstanceId member : members) {
    RuntimeOperatorInstance* const instance = exec.operator_by_id(member);
    const auto receiver = exec.receiver_index(member);
    if (instance == nullptr || !receiver) {
      return false;
    }
    if (!owns_receiver(state, *receiver)) {
      continue;
    }
    if (!append_affected(*receiver, instance->vertex, true)) {
      return false;
    }
  }
  for (const auto& subscriber : plan_->process_local_channel_subscribers) {
    const auto* source =
        find_port_by_id_and_direction(*plan_, subscriber.source, PortDirection::kOutput);
    const auto* input =
        find_port_by_id_and_direction(*plan_, subscriber.target, PortDirection::kInput);
    if (source == nullptr || input == nullptr) {
      return false;
    }
    if (std::find(member_vertices.begin(), member_vertices.end(), source->owner) ==
            member_vertices.end() ||
        std::find(state.restart.vertices.begin(), state.restart.vertices.end(), input->owner) !=
            state.restart.vertices.end()) {
      continue;
    }
    RuntimeOperatorInstance* const affected = exec.operator_by_vertex(input->owner);
    const auto receiver = exec.receiver_index(input->owner);
    if (affected == nullptr || !receiver) {
      return false;
    }
    if (!owns_receiver(state, *receiver)) {
      continue;
    }
    if (!append_affected(*receiver, affected->vertex, false)) {
      return false;
    }
  }
  for (const std::size_t receiver : state.restart.receivers) {
    ClockTriggerPortBinding* const clock = state.clock_bindings[receiver];
    state.restart.phases[receiver].reset();
    if (clock == nullptr) {
      continue;
    }
    auto phase = clock->restart_phase();
    if (!phase) {
      return false;
    }
    state.restart.phases[receiver] = *phase;
  }
  return true;
}

[[nodiscard]] bool RunSessionState::prepare_runtime_restart_aggregate(
    GraphExec& exec, std::span<ReadyRunState> states,
    std::span<const OperatorInstanceId> members) noexcept {
  if (states.empty() || members.empty()) {
    return false;
  }
  ReadyRunState& aggregate = states.front();
  aggregate.restart.aggregate_member_vertices.clear();
  aggregate.restart.aggregate_vertices.clear();
  for (const OperatorInstanceId member : members) {
    RuntimeOperatorInstance* const instance = exec.operator_by_id(member);
    if (instance == nullptr ||
        std::find(aggregate.restart.aggregate_member_vertices.begin(),
                  aggregate.restart.aggregate_member_vertices.end(),
                  instance->vertex) != aggregate.restart.aggregate_member_vertices.end()) {
      return false;
    }
    aggregate.restart.aggregate_member_vertices.push_back(instance->vertex);
  }
  for (ReadyRunState& state : states) {
    if (!prepare_runtime_restart_scope(
            exec, state, members, aggregate.restart.aggregate_member_vertices)) {
      return false;
    }
    for (const VertexId vertex : state.restart.vertices) {
      if (std::find(aggregate.restart.aggregate_vertices.begin(),
                    aggregate.restart.aggregate_vertices.end(),
                    vertex) == aggregate.restart.aggregate_vertices.end()) {
        aggregate.restart.aggregate_vertices.push_back(vertex);
      }
    }
  }
  return !aggregate.restart.aggregate_vertices.empty();
}

[[nodiscard]] expected<bool, Error> RunSessionState::restart_scope_uses_channels(
    const ReadyRunState& aggregate) const noexcept {
  return aggregate.channels == nullptr
             ? expected<bool, Error>{false}
             : aggregate.channels->restart_affects(aggregate.restart.aggregate_member_vertices,
                                                   aggregate.restart.aggregate_vertices);
}

[[nodiscard]] bool RunSessionState::fence_runtime_restart_state(ReadyRunState& state) noexcept {
  for (const std::size_t receiver : state.restart.receivers) {
    auto canceled = state.dispatcher->cancel_receiver(receiver);
    if (!canceled) {
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
    // Every active Dispatcher slot has exactly one coordinator witness. A mismatch means the
    // restart fence could not establish an exact-once retirement boundary.
    if (retired != *canceled) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool RunSessionState::fence_runtime_restart(
    GraphExec& exec, std::span<ReadyRunState> states, std::span<const OperatorInstanceId> members,
    LifecycleStage from_stage, bool& channels_quiesced) noexcept {
  channels_quiesced = false;
  // A lifecycle request can become visible immediately after an input EOS witness retires,
  // before the ordinary owner loop gets another chance to finish its terminal drain.  Complete
  // that already-admitted work at the same serialized boundary before deciding whether restart
  // is safe.  An incomplete drain remains fenced below, while a fully closed external import
  // can publish a replacement generation.
  for (ReadyRunState& state : states) {
    if (!state.active_input_terminal_receivers.empty() &&
        !advance_input_terminal_drains(exec, state)) {
      return false;
    }
  }
  if (!prepare_runtime_restart_aggregate(exec, states, members)) {
    return false;
  }
  ReadyRunState& aggregate = states.front();
  if (aggregate.notifications != nullptr) {
    aggregate.notifications->begin_restart(aggregate.restart.aggregate_vertices);
  }
  const auto fail_restart_fence = [&]() noexcept {
    if (aggregate.notifications != nullptr) {
      aggregate.notifications->rollback_restart(aggregate.restart.aggregate_vertices);
    }
    return false;
  };
  if (std::any_of(states.begin(), states.end(), [this](const ReadyRunState& state) {
        return std::any_of(
            state.restart.receivers.begin(),
            state.restart.receivers.end(),
            [&](const std::size_t receiver) {
              const ReadyRunState::InputTerminalDrainState* const terminal =
                  input_terminal_receiver_state(state, receiver);
              if (terminal != nullptr && terminal->phase == InputTerminalDrainPhase::kDraining) {
                return true;
              }
              const ReadyRunState::CompletionReceiverState* const completion =
                  completion_receiver_state(state, receiver);
              return completion != nullptr &&
                     operator_completion_phases_[completion->completion_index] !=
                         OperatorCompletionPhase::kOpen;
            });
      })) {
    return fail_restart_fence();
  }
  const auto uses_channels = restart_scope_uses_channels(aggregate);
  if (!uses_channels) {
    return fail_restart_fence();
  }
  if (*uses_channels) {
    aggregate.channels->cancel();
    if (!aggregate.channels->join()) {
      return fail_restart_fence();
    }
    channels_quiesced = true;
  }
  for (ReadyRunState& state : states) {
    if (!fence_runtime_restart_state(state)) {
      return fail_restart_fence();
    }
  }
  if (tensor_resources_ != nullptr) {
    auto fenced = tensor_resources_->begin_restart(members, from_stage);
    if (!fenced) {
      return fail_restart_fence();
    }
  }
  return true;
}

// Retire the coordinator and binding prefixes for the complete prepared restart scope before any
// source generation is advanced.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool RunSessionState::retire_restart_witnesses(ReadyRunState& state) noexcept {
  for (const std::size_t receiver : state.restart.receivers) {
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
  }

  // Joined Channel ingress cannot append while the remaining binding-ring prefix is retired.
  // Resetting each affected source generation after this loop therefore cannot strand an
  // un-ingested old-generation witness.
  for (auto& binding : state.bindings) {
    const SourceId source = binding->source();
    if (!source) {
      continue;
    }
    const auto receiver = state.coordinator->source_receiver(source);
    if (!receiver ||
        std::find(state.restart.receivers.begin(), state.restart.receivers.end(), *receiver) ==
            state.restart.receivers.end()) {
      continue;
    }
    RuntimeEvent event;
    for (;;) {
      auto drained = binding->drain({&event, 1U});
      if (!drained || *drained > 1U) {
        return false;
      }
      if (*drained == 0U) {
        break;
      }
      if (!binding->retire(event.source_sequence)) {
        return false;
      }
    }
  }
  return true;
}

// Publish Channel, selector, readiness-source, clock, and predicate generations as one ordered
// restart transaction after the exact affected scope has been fenced.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool RunSessionState::publish_runtime_restart_state(GraphExec& exec,
                                                                  ReadyRunState& state) noexcept {
  for (auto& selector : state.input_selectors) {
    if (std::find(state.restart.vertices.begin(), state.restart.vertices.end(), selector.owner) !=
        state.restart.vertices.end()) {
      if (!selector.selector.reset()) {
        return false;
      }
      selector.end_of_stream_received = false;
      selector.end_of_stream_decision_pending = false;
      selector.freshness_refill_prerequisite.reset();
      selector.freshness_refill_target.reset();
      selector.deferred_drain_refill.reset();
    }
  }
  for (auto& aligned : state.aligned_selectors) {
    if (std::find(state.restart.vertices.begin(), state.restart.vertices.end(), aligned.owner) !=
            state.restart.vertices.end() &&
        !aligned.selector.reset()) {
      return false;
    }
  }
  // Channel restart retires the affected typed slots before the metadata
  // selectors reset. Drop the mirrored claim sets without calling back into
  // the now-new Channel generation.
  for (auto& selector : state.input_selectors) {
    if (std::find(state.restart.vertices.begin(), state.restart.vertices.end(), selector.owner) !=
        state.restart.vertices.end()) {
      selector.channel_payload_claims.clear();
    }
  }

  for (const std::size_t receiver : state.restart.receivers) {
    if (!state.coordinator->publish_restart(receiver)) {
      return false;
    }
    std::size_t advanced_sources = 0U;
    for (auto& binding : state.bindings) {
      const SourceId source = binding->source();
      if (!source) {
        continue;
      }
      const auto source_receiver = state.coordinator->source_receiver(source);
      if (!source_receiver || source_receiver.value() != receiver) {
        continue;
      }
      const auto expected_generation = state.coordinator->source_generation(source);
      if (!expected_generation) {
        return false;
      }
      ClockTriggerPortBinding* const clock = state.clock_bindings[receiver];
      const auto& restart_phase = state.restart.phases[receiver];
      if (binding.get() == clock && !restart_phase) {
        return false;
      }
      const expected<std::uint64_t, Error> advanced_generation =
          binding.get() == clock ? expected<std::uint64_t, Error>{clock->publish_restart(
                                       restart_phase.value_or(TimePoint{}))}
                                 : binding->advance_source_generation();
      if (!advanced_generation || *advanced_generation != *expected_generation) {
        return false;
      }
      ++advanced_sources;
    }
    if (advanced_sources == 0U) {
      return false;
    }
    ReadyRunState::InputTerminalDrainState* const terminal =
        input_terminal_receiver_state(state, receiver);
    if (terminal != nullptr) {
      terminal->phase = InputTerminalDrainPhase::kOpen;
      const auto closed = std::find(state.closed_input_terminal_receivers.begin(),
                                    state.closed_input_terminal_receivers.end(),
                                    receiver);
      if (closed != state.closed_input_terminal_receivers.end()) {
        state.closed_input_terminal_receivers.erase(closed);
      }
    }
    if (state.predicates[receiver]) {
      const bool lifecycle_member =
          std::find(state.restart.member_vertices.begin(),
                    state.restart.member_vertices.end(),
                    exec.operators()[receiver].vertex) != state.restart.member_vertices.end();
      state.predicates[receiver]->reset(lifecycle_member ? PredicateResetReason::kOperatorRestart
                                                         : PredicateResetReason::kProducerEpoch,
                                        exec.operators()[receiver].tags->generation());
    }
  }
  return true;
}

// Restart publication validates and updates the aggregate lifecycle as one ordered transition.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool RunSessionState::publish_runtime_restart(GraphExec& exec,
                                                            std::span<ReadyRunState> states,
                                                            OperatorRef target,
                                                            bool channels_quiesced) noexcept {
  if (states.empty()) {
    return false;
  }
  ReadyRunState& aggregate = states.front();
  RuntimeOperatorInstance* const target_instance = exec.operator_by_id(target.id);
  if (target_instance == nullptr ||
      std::find(aggregate.restart.aggregate_member_vertices.begin(),
                aggregate.restart.aggregate_member_vertices.end(),
                target_instance->vertex) == aggregate.restart.aggregate_member_vertices.end()) {
    return false;
  }
  for (ReadyRunState& state : states) {
    if (!retire_restart_witnesses(state)) {
      return false;
    }
  }
  if (tensor_resources_ != nullptr) {
    auto preflight = tensor_resources_->preflight_restart_publication();
    if (!preflight) {
      return false;
    }
  }
  // One run-wide rate candidate backs every affected partition state. Validate it once before
  // Channel commits the irreversible transport discard below.
  if (rate_state_ == nullptr || !rate_state_->preflight_candidate_publication()) {
    return false;
  }
  if (channels_quiesced) {
    // Channel commit is the irreversible transport-discard boundary. Every failure after this
    // point is fail-stopped by process_pending_restarts(); no runner may resume an activation
    // against a partially published generation, and rollback must not resurrect discarded
    // ingress.
    auto committed = aggregate.channels->commit_restart(aggregate.restart.aggregate_member_vertices,
                                                        aggregate.restart.aggregate_vertices);
    // `restart_scope_uses_channels()` proved that this lifecycle incident owns at least one
    // frozen Channel binding. A missing affected Channel state is therefore an inconsistent
    // session, not an ignorable empty scope. Fail before publishing selector, readiness, or
    // producer generations so old samples can never be relabeled as post-restart state.
    if (!committed) {
      return false;
    }
  }
  if (channel_session_ != nullptr && !channel_session_->commit_rate_candidate_active_depths()) {
    return false;
  }
  for (ReadyRunState& state : states) {
    if (!publish_runtime_restart_state(exec, state)) {
      return false;
    }
  }
  if (!runtime_internal::commit_rate_candidate_temporal_depths(states)) {
    return false;
  }
  if (channels_quiesced && !arm_channels_with_diagnostics(aggregate.channels)) {
    return false;
  }
  // Direct Tensor pools are the final publication participant. Every earlier Channel,
  // selector, readiness, and coordinator transition can still fail; publishing a replacement
  // pool before them would consume the old pool generation even when the restart is ultimately
  // rejected. Tensor publication is preflighted by begin_restart() and cannot expose user work
  // until this safe point returns.
  if (tensor_resources_ != nullptr) {
    auto published = tensor_resources_->publish_restart();
    if (!published) {
      return false;
    }
  }
  if (aggregate.notifications != nullptr &&
      !aggregate.notifications->publish_restart(aggregate.restart.aggregate_vertices)) {
    return false;
  }
  return static_cast<bool>(rate_state_->publish_candidate());
}

[[nodiscard]] bool RunSessionState::rollback_runtime_restart(std::span<ReadyRunState> states,
                                                             bool channels_quiesced) noexcept {
  if (states.empty()) {
    return false;
  }
  ReadyRunState& aggregate = states.front();
  const bool tensor_rolled_back =
      tensor_resources_ == nullptr || tensor_resources_->rollback_restart().has_value();
  const bool channels_rearmed =
      !channels_quiesced || arm_channels_with_diagnostics(aggregate.channels);
  if (aggregate.notifications != nullptr) {
    aggregate.notifications->rollback_restart(aggregate.restart.aggregate_vertices);
  }
  if (channel_session_ != nullptr) {
    channel_session_->discard_rate_candidate_active_depths();
  }
  discard_rate_candidate_temporal_depths(states);
  return tensor_rolled_back && channels_rearmed;
}

// Restart processing is one rollback-capable transaction across all runtime subsystems.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool RunSessionState::process_pending_restarts(
    GraphExec& exec, std::span<ReadyRunState> states) noexcept {
  if (states.empty()) {
    return false;
  }
  const RuntimeCallbackContextScope control_scope{RuntimeCallbackContext::kDispatcher};
  struct RestartPublicationState {
    RunSessionState* session{};
    GraphExec* exec{};
    std::span<ReadyRunState> ready_states;
  } publication{.session = this, .exec = &exec, .ready_states = states};
  return lifecycle_control_->process_pending(
      // Keep lifecycle candidate state and subsystem publication in one rollback-capable callback.
      // NOLINTNEXTLINE(readability-function-cognitive-complexity)
      [](void* state, const LifecycleRestartExecutionRequest& request) noexcept {
        auto& publication = *static_cast<RestartPublicationState*>(state);
        publication.exec->clear_lifecycle_failure_evidence();
        const auto transition_candidates = [&](LifecycleTransactionState lifecycle_state) {
          for (const auto& candidate : request.parameter_candidates) {
            if (!publication.exec->parameter_updates().transition_lifecycle_candidate(
                    candidate.owner, candidate.generation, request.transaction, lifecycle_state)) {
              return false;
            }
          }
          return true;
        };
        if (!transition_candidates(LifecycleTransactionState::kQuiescing) ||
            !transition_candidates(LifecycleTransactionState::kRestarting)) {
          static_cast<void>(transition_candidates(LifecycleTransactionState::kFailedStopped));
          return LifecycleRestartExecutionOutcome{};
        }
        bool channels_quiesced = false;
        if (!publication.session->fence_runtime_restart(*publication.exec,
                                                        publication.ready_states,
                                                        request.members,
                                                        request.stage,
                                                        channels_quiesced)) {
          static_cast<void>(transition_candidates(LifecycleTransactionState::kFailedStopped));
          return LifecycleRestartExecutionOutcome{};
        }
        if (publication.session->rate_state_ == nullptr ||
            !publication.session->rate_state_->begin_restart_candidate(request.members,
                                                                       request.stage)) {
          static_cast<void>(transition_candidates(LifecycleTransactionState::kFailedStopped));
          return LifecycleRestartExecutionOutcome{};
        }
        LifecycleRestartExecutionOutcome outcome = publication.exec->lifecycle_restart(request);
        // A Parameter-origin restart shares the Parameter terminal fence:
        // do not publish its candidate or source generation after stop.
        // Plain lifecycle work instead preserves the exact outcome already
        // produced by the executor, as required by the lifecycle contract.
        const bool parameter_stop_fence =
            !request.parameter_candidates.empty() &&
            publication.session->stop_requested_.load(std::memory_order_acquire);
        const bool restart_publication_failed =
            !parameter_stop_fence &&
            ((outcome.disposition == LifecycleRestartExecutionDisposition::kPublished &&
              !publication.session->publish_runtime_restart(*publication.exec,
                                                            publication.ready_states,
                                                            request.target,
                                                            channels_quiesced)) ||
             (outcome.disposition == LifecycleRestartExecutionDisposition::kRolledBack &&
              !publication.session->rollback_runtime_restart(publication.ready_states,
                                                             channels_quiesced)));
        if (parameter_stop_fence || restart_publication_failed) {
          outcome.disposition = LifecycleRestartExecutionDisposition::kFailedStopped;
        }
        if (outcome.disposition == LifecycleRestartExecutionDisposition::kFailedStopped &&
            publication.session->channel_session_ != nullptr) {
          publication.session->channel_session_->discard_rate_candidate_active_depths();
        }
        if (outcome.disposition == LifecycleRestartExecutionDisposition::kFailedStopped) {
          publication.session->discard_rate_candidate_temporal_depths(publication.ready_states);
        }
        const LifecycleTransactionState terminal_state = [&] {
          switch (outcome.disposition) {
            case LifecycleRestartExecutionDisposition::kPublished:
              return LifecycleTransactionState::kPublished;
            case LifecycleRestartExecutionDisposition::kRolledBack:
              return LifecycleTransactionState::kRolledBack;
            case LifecycleRestartExecutionDisposition::kFailedStopped:
              return LifecycleTransactionState::kFailedStopped;
          }
          return LifecycleTransactionState::kFailedStopped;
        }();
        if (!transition_candidates(terminal_state)) {
          outcome.disposition = LifecycleRestartExecutionDisposition::kFailedStopped;
          static_cast<void>(transition_candidates(LifecycleTransactionState::kFailedStopped));
        }
        if (publication.session->rate_state_ != nullptr) {
          if (parameter_stop_fence) {
            publication.session->rate_state_->discard_candidate();
          } else if (outcome.disposition != LifecycleRestartExecutionDisposition::kPublished) {
            const std::string_view failure_token = publication.exec->lifecycle_failure_token();
            publication.session->rate_state_->reject_candidate(
                Error{ErrorCode::kLifecycleStageFailed, std::string{failure_token}});
          }
        }
        if (outcome.disposition == LifecycleRestartExecutionDisposition::kRolledBack) {
          publication.session->append_lifecycle_diagnostics(
              *publication.exec, ::holoscan::DiagnosticSeverity::kWarning);
        }
        return outcome;
      },
      &publication,
      stop_requested_);
}

[[nodiscard]] bool RunSessionState::process_pending_restarts(GraphExec& exec,
                                                             ReadyRunState& state) noexcept {
  return process_pending_restarts(exec, std::span<ReadyRunState>{&state, 1U});
}

}  // namespace holoscan::detail
