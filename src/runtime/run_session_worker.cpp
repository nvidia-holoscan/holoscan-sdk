/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "run_session_internal.hpp"

#include "../core/canonical_text_encoding.hpp"
#include "busy_wait_pause.hpp"
#include "core/diagnostic_reporting.hpp"
#include "runtime_cuda_context_placement.hpp"
#include "runtime_cuda_graph_diagnostics.hpp"
#include "runtime_cuda_graph_region_resources.hpp"
#include "runtime_fanout_test_control.hpp"
#include "runtime_support_thread_control.hpp"
#include "runtime_support_thread_placement.hpp"
#include "runtime_test_controls.hpp"
#include "temporal_depth_publication.hpp"

namespace holoscan::detail {

// Partition-owner execution and terminal run cleanup.

static_assert(std::is_nothrow_move_constructible_v<Error>,
              "quiesce failure cleanup requires allocation-free Error transfer");

namespace {

// Mark the calling thread as a partition owner for the scope of one owner loop.
class PartitionOwnerThreadScope final {
 public:
  explicit PartitionOwnerThreadScope(const RuntimeWakeState* wake) noexcept
      : previous_(current_partition_owner_wake()) {
    current_partition_owner_wake() = wake;
    current_partition_owner_rescan() = false;
  }
  ~PartitionOwnerThreadScope() {
    current_partition_owner_wake() = previous_;
    current_partition_owner_rescan() = false;
  }
  PartitionOwnerThreadScope(const PartitionOwnerThreadScope&) = delete;
  PartitionOwnerThreadScope& operator=(const PartitionOwnerThreadScope&) = delete;

 private:
  const RuntimeWakeState* previous_;
};

// The identity recorded for a partition owner's thread: the Linux thread id, which traces show,
// or elsewhere a nonzero opaque value derived from std::thread::id. Zero means "no owner".
[[nodiscard]] std::uint64_t owner_thread_identity() noexcept {
#if defined(__linux__)
  return static_cast<std::uint64_t>(syscall(SYS_gettid));
#else
  const auto hashed =
      static_cast<std::uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
  return hashed == 0U ? 1U : hashed;
#endif
}

class RunOwnedThreadScope final {
 public:
  explicit RunOwnedThreadScope(const void* owner) noexcept
      : previous_owner_(current_run_owned_session()) {
    set_run_owned_session(owner);
  }

  ~RunOwnedThreadScope() noexcept { set_run_owned_session(previous_owner_); }

 private:
  const void* previous_owner_;
};

}  // namespace

// Keep cleanup and quarantine in ownership order so every failed CUDA barrier retains
// the same complete owner closure before terminal publication.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] expected<void, Error> RunSessionState::release_execution_resources(
    RuntimeCudaExecutionDiagnostic* cuda_diagnostic) {
  if (tensor_resources_ != nullptr) {
    tensor_resources_->close();
    tensor_resources_->detach_ports();
  }
  // Retain the terminal ParameterUpdateCoordinator separately so existing application handles
  // can inspect final update results until their RunSession owner is destroyed.
  if (graph_exec_ != nullptr && parameter_updates_ != nullptr) {
    retained_parameter_updates_ = graph_exec_->take_parameter_updates();
  } else {
    // Startup failures that never attached the control link must wake concurrent handle minting
    // with kClosed; there is no terminal coordinator state to preserve.
    retire_parameter_control();
  }
  if (cuda_graph_quiesce_failure_) {
    if (plan_ != nullptr && plan_->executable_factories.test_cuda_execution_control != nullptr) {
      plan_->executable_factories.test_cuda_execution_control->notify_failure_observer(
          RuntimeCudaExecutionObserverPoint::kQuiesceFailureRelease);
    }
    // This cleanup runs outside worker_main()'s catch boundary. Move the owning message through
    // the result path so host allocation failure cannot bypass quarantine or terminal publication.
    Error failure = std::move(*cuda_graph_quiesce_failure_);
    cuda_graph_quiesce_failure_.reset();
    if (cuda_execution_resources_ != nullptr) {
      cuda_execution_resources_->quarantine_after_external_failure();
    }
    quarantine_run_execution_owners(std::move(graph_exec_),
                                    std::move(tensor_resources_),
                                    std::move(channel_session_),
                                    std::move(cuda_graph_region_resources_),
                                    std::move(cuda_execution_resources_));
    if (plan_ != nullptr && plan_->executable_factories.test_cuda_execution_control != nullptr) {
      plan_->executable_factories.test_cuda_execution_control->notify_failure_observer(
          RuntimeCudaExecutionObserverPoint::kQuiesceFailureReturn);
    }
    return make_unexpected(std::move(failure));
  }
  if (cuda_graph_region_resources_ != nullptr) {
    if (auto synchronized = cuda_graph_region_resources_->quiesce(cuda_diagnostic); !synchronized) {
      if (cuda_execution_resources_ != nullptr) {
        cuda_execution_resources_->quarantine_after_external_failure();
      }
      quarantine_run_execution_owners(std::move(graph_exec_),
                                      std::move(tensor_resources_),
                                      std::move(channel_session_),
                                      std::move(cuda_graph_region_resources_),
                                      std::move(cuda_execution_resources_));
      return synchronized;
    }
  }
  // Quiesce while reconstructed Operators, unpublished loans, Tensor backing, fixed CUDA events,
  // and Channel-owned imported-VMM mappings remain alive. A failed barrier makes their native
  // dependencies uncertain, so retain the complete owner closure and require process restart
  // rather than destroying beneath work that CUDA did not prove complete.
  if (cuda_execution_resources_ != nullptr) {
    if (auto synchronized = cuda_execution_resources_->quiesce(cuda_diagnostic); !synchronized) {
      quarantine_run_execution_owners(std::move(graph_exec_),
                                      std::move(tensor_resources_),
                                      std::move(channel_session_),
                                      std::move(cuda_graph_region_resources_),
                                      std::move(cuda_execution_resources_));
      return synchronized;
    }
  }
  if (cuda_graph_region_resources_ != nullptr) {
    if (auto shutdown = cuda_graph_region_resources_->shutdown(cuda_diagnostic); !shutdown) {
      if (cuda_execution_resources_ != nullptr) {
        cuda_execution_resources_->quarantine_after_external_failure();
      }
      quarantine_run_execution_owners(std::move(graph_exec_),
                                      std::move(tensor_resources_),
                                      std::move(channel_session_),
                                      std::move(cuda_graph_region_resources_),
                                      std::move(cuda_execution_resources_));
      return shutdown;
    }
  }
  if (cuda_graph_region_resources_ != nullptr) {
    cuda_graph_region_resources_->detach_session_state();
  }
  // Channel bindings borrow reconstructed Operator ports, while Operators can retain unpublished
  // Tensor loans. After stream quiescence, detach and destroy the Channel session before its
  // borrowed ports, then destroy Operators before Tensor pools and native CUDA resources.
  channel_session_.reset();
  cuda_graph_region_resources_.reset();
  graph_exec_.reset();
  tensor_resources_.reset();
  expected<void, Error> shutdown;
  if (cuda_execution_resources_ != nullptr) {
    shutdown = cuda_execution_resources_->shutdown(cuda_diagnostic);
  }
  cuda_execution_resources_.reset();
  return shutdown;
}

void RunSessionState::worker_main() noexcept {
  const RunOwnedThreadScope run_owned_thread{this};
  RunTermination outcome{RunTermination::kFailed};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    outcome = worker_main_impl();
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    append_diagnostic("RUNTIME_STARTUP_FAILED");
  }
#endif
  // Once the worker has passed every possible callback publication boundary, accepted
  // candidates cannot remain pending. Terminalize them before releasing Graph-owned state.
  terminalize_parameter_updates_at_run_end();
  RuntimeCudaExecutionDiagnostic cuda_diagnostic;
  if (auto released = release_execution_resources(&cuda_diagnostic); !released) {
    RuntimeDiagnosticDetail detail{
        .present = true,
        .cuda_execution = std::move(cuda_diagnostic),
        .operation = DiagnosticOperation::kRuntime,
        .role = DiagnosticFailureRole::kTeardown,
    };
    detail.capture.emplace();
    detail.capture->append(released.error().code, released.error().message);
    append_diagnostic(
        registered_token_or(released.error().message, "CUDA_GREEN_CONTEXT_MATERIALIZATION_FAILED"),
        {},
        {},
        &detail);
    // Mandatory cleanup is part of whole-run success. Preserve any earlier failure and promote
    // an otherwise successful or stopped execution to the strongest terminal outcome.
    outcome = RunTermination::kFailed;
  }
  finish(outcome);
}

void RunSessionState::worker_main_with_thread_usage() noexcept {
  const RuntimeThreadUsageScope thread_usage{phase_counters_, RuntimeThreadUsageRole::kController};
  worker_main();
}

template <bool TracksTerminalWork>
[[nodiscard]] bool RunSessionState::all_sources_closed(GraphExec& exec,
                                                       const ReadyRunState& state) const noexcept {
  if constexpr (!TracksTerminalWork) {
    return std::all_of(state.bindings.begin(), state.bindings.end(), [](const auto& binding) {
      return binding->closed();
    });
  }
  bool awaiting_external_restart = false;
  if (lifecycle_control_->enabled() && state.channels != nullptr) {
    for (const std::size_t receiver : state.closed_input_terminal_receivers) {
      if (receiver < exec.operators().size() &&
          state.channels->awaits_external_restart(exec.operators()[receiver].vertex)) {
        awaiting_external_restart = true;
        break;
      }
    }
  }
  return state.active_completion_receivers.empty() &&
         state.active_input_terminal_receivers.empty() && !awaiting_external_restart &&
         std::all_of(state.bindings.begin(), state.bindings.end(), [](const auto& binding) {
           return binding->closed();
         });
}

// A source can fail after the owner's last drain, and closed() does not tell failure from a
// normal close, so report a terminal failure here instead of completing.
[[nodiscard]] bool RunSessionState::sources_closed_without_failure(
    const ReadyRunState& state) noexcept {
  for (const auto& binding : state.bindings) {
    if (!binding->terminal_status()) {
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
      return false;
    }
  }
  return true;
}

// Publish each tick-reconciled input's next tick, so a publication racing that tick cannot displace
// the newest sample at or before it. The owner publishes it before parking and as soon as a scan
// freezes work: an owner running behind its clock may not park between two ticks, and until the
// fence moves past the tick just frozen, promotion defers every newer arrival to staging, whose
// N + 1 survivors can all be past the next tick by the time a delayed scan reaches it.
void RunSessionState::publish_tick_fences(ReadyRunState& state) noexcept {
  if (!state.publishes_tick_fences) {
    return;
  }
  for (const auto& selector : state.input_selectors) {
    if (!selector.tick_fenced || selector.receiver_index >= state.clock_bindings.size()) {
      continue;
    }
    const ClockTriggerPortBinding* const clock = state.clock_bindings[selector.receiver_index];
    if (clock == nullptr) {
      continue;
    }
    if (const std::optional<TimePoint> next_tick = clock->next_wake(); next_tick) {
      state.channels->set_tick_fence(selector.input, *next_tick);
    }
  }
}

// Keep the park-admission predicates, tick fences, and wait accounting together so the
// deadline is armed only after every source of pending work has been checked.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] bool RunSessionState::wait_for_source_progress(GraphExec& exec,
                                                             ReadyRunState& state) noexcept {
  using namespace std::chrono_literals;
  // A long park needs every progress source to wake the owner, and nothing this owner itself
  // must revisit: an incomplete poll, a pending event, a refill fence, or terminal work.
  const bool long_park =
      state.long_park_eligible && !state.poll_incomplete_this_round &&
      state.coordinator->pending_count() == 0U && state.freshness_input_fences.empty() &&
      state.active_completion_receivers.empty() && state.active_input_terminal_receivers.empty();
  const std::chrono::nanoseconds park_cap = long_park ? owner_park_cap() : 1ms;
  std::chrono::nanoseconds wait_duration = park_cap;
  for (const auto& binding : state.bindings) {
    const std::optional<PortBindingPollTarget> target = binding->poll_target();
    if (target && target->busy_poll) {
      // One controller loop performs one bounded poll. Returning here keeps
      // stop checks cooperative without parking the admitted busy-poll lane.
      return true;
    }
    const std::optional<TimePoint> wake = binding->next_wake();
    if (!wake) {
      continue;
    }
    auto now = exec.now(wake->clock_id);
    if (!now) {
      append_diagnostic("RUNTIME_CLOCK_READ_FAILED");
      return false;
    }
    if (now->timestamp_ns >= wake->timestamp_ns) {
      return true;
    }
    wait_duration =
        bounded_source_wait_duration(now->timestamp_ns, wake->timestamp_ns, wait_duration);
  }
  if (stop_requested_.load(std::memory_order_acquire)) {
    return true;
  }
  if (state.owner_busy_poll) {
    // Channel bindings expose revision changes, not the lane's waiting policy. Honor the
    // frozen dedicated-lane request even when no Channel revision changed on this scan.
    busy_wait_pause();
    return true;
  }
  publish_tick_fences(state);
  RuntimeWakeState& wake = state.wake != nullptr ? *state.wake : readiness_wake_;
  // Arm the absolute deadline at once: nothing between this read and the kernel wait may delay
  // the timer, and a spurious or interrupted return waits again for the same deadline.
  const auto deadline = std::chrono::steady_clock::now() + wait_duration;
  if (phase_counters_ == nullptr) {
    static_cast<void>(wake.wait_until(deadline));
    return !wake_wait_failed(wake);
  }
  phase_counters_->increment(RuntimePhaseCounter::kWaitCalls);
  phase_counters_->increment(state.accounting_partition, RuntimePartitionCounter::kWaits);
  if (long_park) {
    phase_counters_->increment(state.accounting_partition, RuntimePartitionCounter::kLongParks);
  }
  phase_counters_->trace(
      RuntimePhaseTracePoint::kControllerParkStarted, kNoRuntimePhaseReceiver, {}, 0U);
  const bool capped = wait_duration == park_cap;
  if (wake.wait_until(deadline)) {
    phase_counters_->increment(RuntimePhaseCounter::kWakeReturns);
    phase_counters_->increment(state.accounting_partition, RuntimePartitionCounter::kTokenReturns);
    phase_counters_->trace(
        RuntimePhaseTracePoint::kControllerWakeReturned, kNoRuntimePhaseReceiver, {}, 0U);
  } else {
    phase_counters_->increment(RuntimePhaseCounter::kTimeoutReturns);
    phase_counters_->increment(
        state.accounting_partition,
        capped ? RuntimePartitionCounter::kCapReturns : RuntimePartitionCounter::kDeadlineReturns);
    state.long_park_ended_at_cap = long_park && capped;
    phase_counters_->trace(
        RuntimePhaseTracePoint::kControllerTimeoutReturned, kNoRuntimePhaseReceiver, {}, 0U);
  }
  return !wake_wait_failed(wake);
}

// The long cap bounds the cost of a progress source that fails to wake its owner.
[[nodiscard]] std::chrono::nanoseconds RunSessionState::owner_park_cap() const noexcept {
  constexpr std::chrono::nanoseconds kOwnerParkCap = std::chrono::milliseconds{100};
  const std::int64_t override_ns = plan_->executable_factories.test_owner_park_cap_ns;
  return override_ns > 0 ? std::chrono::nanoseconds{override_ns} : kOwnerParkCap;
}

// A wait that failed for any reason other than its deadline fails the run closed: treating it as
// a timeout would make the owner spin at full CPU.
[[nodiscard]] bool RunSessionState::wake_wait_failed(const RuntimeWakeState& wake) noexcept {
  if (!wake.failed()) {
    return false;
  }
  append_diagnostic("RUNTIME_WAKE_WAIT_FAILED");
  return true;
}

// The owner loop serializes lifecycle, readiness, activation, and termination transitions.
// The private observation entry point admits exactly one partition, which runs through the shared
// owner below. These multi-partition owners therefore call the unobserved dispatch specialization.
template <bool TracksTerminalWork>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] RunTermination RunSessionState::run_partition_owner(
    PartitionOwnerContext& context) noexcept {
  GraphExec& exec = *context.exec;
  ReadyRunState& ready = *context.ready;
  const PartitionOwnerThreadScope owner_thread{ready.wake};
  if (phase_counters_ != nullptr) {
    phase_counters_->record_partition_owner(ready.accounting_partition, owner_thread_identity());
  }
  ready.direct_chain.group_failed = &context.group->failed;
  while (!context.group->failed.load(std::memory_order_acquire)) {
    if (stop_requested_.load(std::memory_order_acquire)) {
      return RunTermination::kStopped;
    }
    if (lifecycle_control_->has_pending()) {
      return RunTermination::kCompleted;
    }
    if constexpr (TracksTerminalWork) {
      // Like every failure below, wake the other owners: one parked until its next deadline would
      // otherwise learn of the failure only when that park ends.
      if (!ready.active_completion_receivers.empty() && !advance_completion_closures(exec, ready)) {
        context.group->failed.store(true, std::memory_order_release);
        signal_controller_wake(RuntimePhaseCounter::kWakeSignals);
        return RunTermination::kFailed;
      }
      if (!ready.active_input_terminal_receivers.empty() &&
          !advance_input_terminal_drains(exec, ready)) {
        context.group->failed.store(true, std::memory_order_release);
        signal_controller_wake(RuntimePhaseCounter::kWakeSignals);
        return RunTermination::kFailed;
      }
    }
    if (ready.dispatcher->active_count() != 0U) {
      const RunTermination dispatched = dispatch_ready_impl<false>(exec, ready);
      if (dispatched != RunTermination::kCompleted) {
        if (dispatched == RunTermination::kFailed) {
          context.group->failed.store(true, std::memory_order_release);
          signal_controller_wake(RuntimePhaseCounter::kWakeSignals);
        }
        return dispatched;
      }
      if (lifecycle_control_->has_pending()) {
        return RunTermination::kCompleted;
      }
      count_phase(RuntimePhaseCounter::kPartitionTailDispatchAttempts);
      count_phase((ready.partition_owner_execution || ready.direct_chain.continuation_receipt)
                      ? RuntimePhaseCounter::kPartitionTailDispatchAccepted
                      : RuntimePhaseCounter::kPartitionTailDispatchFallbacks);
      continue;
    }
    RuntimeWakeState& wake = ready.wake != nullptr ? *ready.wake : readiness_wake_;
    static_cast<void>(wake.drain());
    // The scan below observes everything this owner published before it.
    current_partition_owner_rescan() = false;
    count_phase(RuntimePhaseCounter::kReadinessScans);
    count_phase(RuntimePhaseCounter::kPartitionRunnerSweeps);
    if (phase_counters_ != nullptr) {
      phase_counters_->increment(ready.accounting_partition,
                                 RuntimePartitionCounter::kReadinessScans);
    }
    if (!freeze_ready_candidates(exec, ready, ready.runner->execution_lane)) {
      context.group->failed.store(true, std::memory_order_release);
      signal_controller_wake(RuntimePhaseCounter::kWakeSignals);
      return RunTermination::kFailed;
    }
    if (lifecycle_control_->has_pending()) {
      return RunTermination::kCompleted;
    }
    if (ready.dispatcher->active_count() != 0U) {
      publish_tick_fences(ready);
      const RunTermination dispatched = dispatch_ready_impl<false>(exec, ready);
      if (dispatched != RunTermination::kCompleted) {
        if (dispatched == RunTermination::kFailed) {
          context.group->failed.store(true, std::memory_order_release);
          signal_controller_wake(RuntimePhaseCounter::kWakeSignals);
        }
        return dispatched;
      }
      if (lifecycle_control_->has_pending()) {
        return RunTermination::kCompleted;
      }
      count_phase(RuntimePhaseCounter::kPartitionTailDispatchAttempts);
      count_phase((ready.partition_owner_execution || ready.direct_chain.continuation_receipt)
                      ? RuntimePhaseCounter::kPartitionTailDispatchAccepted
                      : RuntimePhaseCounter::kPartitionTailDispatchFallbacks);
      continue;
    }
    if (all_sources_closed<TracksTerminalWork>(exec, ready)) {
      if (!sources_closed_without_failure(ready)) {
        context.group->failed.store(true, std::memory_order_release);
        signal_controller_wake(RuntimePhaseCounter::kWakeSignals);
        return RunTermination::kFailed;
      }
      return RunTermination::kCompleted;
    }
    count_phase(RuntimePhaseCounter::kNoProgressScans);
    if (phase_counters_ != nullptr) {
      phase_counters_->increment(ready.accounting_partition,
                                 RuntimePartitionCounter::kNoProgressScans);
    }
    if (std::exchange(current_partition_owner_rescan(), false)) {
      continue;  // this owner signaled itself during the scan; scan again instead of parking
    }
    if (!wait_for_source_progress(exec, ready)) {
      context.group->failed.store(true, std::memory_order_release);
      signal_controller_wake(RuntimePhaseCounter::kWakeSignals);
      return RunTermination::kFailed;
    }
  }
  return RunTermination::kFailed;
}

template <bool TracksTerminalWork>
[[nodiscard]] expected<void, Error> RunSessionState::invoke_partition_owner(void* opaque) noexcept {
  if (opaque == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "RUNTIME_PARTITION_RUNNER_CONTEXT_INVALID"});
  }
  auto& context = *static_cast<PartitionOwnerContext*>(opaque);
  if (context.session == nullptr || context.exec == nullptr || context.ready == nullptr ||
      context.group == nullptr || context.ready->runner == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "RUNTIME_PARTITION_RUNNER_CONTEXT_INVALID"});
  }
  context.outcome = context.session->run_partition_owner<TracksTerminalWork>(context);
  if (context.outcome == RunTermination::kFailed &&
      context.group->failed.load(std::memory_order_acquire)) {
    if (const auto& control =
            context.session->plan_->executable_factories.test_direct_continuation_control;
        control) {
      control->group_failure_published();
    }
  }
  // Execution failures already set group.failed and retain their runtime diagnostics. The lane
  // task itself completed correctly; reserve its Error channel for invalid runner context.
  return {};
}

RunSessionState::PendingStartupRateCandidate::~PendingStartupRateCandidate() {
  if (state != nullptr && !terminal) {
    state->reject_candidate(std::move(failure));
  }
}

void RunSessionState::PendingStartupRateCandidate::set_failure(Error error) {
  failure = std::move(error);
}

void RunSessionState::PendingStartupRateCandidate::publish() noexcept {
  terminal = true;
}

void RunSessionState::PendingStartupRateCandidate::cancel() noexcept {
  if (state != nullptr && !terminal) {
    state->discard_candidate();
  }
  terminal = true;
}

/// Materialize compile-concrete startup resources before lifecycle callbacks can run.
[[nodiscard]] bool RunSessionState::materialize_startup_resources(
    GraphExec& exec, std::vector<ReadyRunState>& ready_states, ReadySharedResources& shared,
    RuntimeCudaExecutionResources& cuda, RuntimeRateState& rates,
    PendingStartupRateCandidate& pending_rate_candidate) {
  ReadyRunState& shared_ready = ready_states.front();
  if (!initialize_channel_session(exec, shared_ready, shared, cuda)) {
    return false;
  }
  // ReadySharedResources is local to worker_main_impl(). Retain the Channel session separately
  // so its imported CUDA mappings cannot be released before terminal stream quiescence.
  channel_session_ = shared.channels;
  for (ReadyRunState& ready : ready_states) {
    bind_shared_resources(ready, shared);
  }
  auto tensor_resources = RuntimeTensorResources::create(
      *plan_, exec.operators(), run_generation_, exec.process_group(), cuda);
  if (!tensor_resources) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR("Runtime Tensor resource materialization failed: {}",
                                 runtime_error_message(tensor_resources.error()));
    append_diagnostic("PROFILE_MATERIALIZATION_FAILED");
    return false;
  }
  tensor_resources_ = std::move(*tensor_resources);
  auto provider_evidence = rates.visit_candidate_materialization(
      tensor_resources_.get(),
      [](void* state,
         const RuntimeRateMaterializationView& materialization) noexcept -> expected<void, Error> {
        if (state == nullptr) {
          return make_unexpected(
              Error{ErrorCode::kInvalidArgument, "QUEUE_RATE_MATERIALIZATION_INVALID"});
        }
        return static_cast<RuntimeTensorResources*>(state)->prepare_rate_candidate_startup(
            materialization);
      },
      false);
  if (!provider_evidence) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR("Runtime Tensor provider evidence failed: {}",
                                 runtime_error_message(provider_evidence.error()));
    append_diagnostic(
        registered_token_or(provider_evidence.error().message, "PROFILE_MATERIALIZATION_FAILED"));
    pending_rate_candidate.set_failure(provider_evidence.error());
    return false;
  }
  for (ReadyRunState& ready : ready_states) {
    if (!initialize_ready_state(exec, ready, shared)) {
      return false;
    }
  }
  return bind_shared_production_notifications(exec, ready_states, shared);
}

/// Bind lifecycle, publish the accepted rate candidate, then arm every startup source.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] RunTermination RunSessionState::arm_startup_sources(
    GraphExec& exec, std::vector<ReadyRunState>& ready_states, ReadySharedResources& shared,
    RateResourceMaterializationState& rate_materialization,
    PendingStartupRateCandidate& pending_rate_candidate) {
  if (auto lifecycle =
          exec.initialize_lifecycle(run_generation_,
                                    &stop_requested_,
                                    shared.lanes.get(),
                                    shared.notifications.get(),
                                    rate_state_.get(),
                                    RuntimeRateMaterializationBinding{
                                        .state = &rate_materialization,
                                        .ensure = &RunSessionState::ensure_rate_candidate_resources,
                                    });
      !lifecycle) {
    append_diagnostic(
        registered_token_or(lifecycle.error().message, "RUNTIME_LIFECYCLE_PLAN_INVALID"));
    return RunTermination::kFailed;
  }

  RunTermination outcome = RunTermination::kCompleted;
  if (stop_requested_.load(std::memory_order_acquire)) {
    outcome = RunTermination::kStopped;
  } else if (!exec.lifecycle_startup()) {
    append_lifecycle_diagnostics(exec);
    pending_rate_candidate.set_failure(
        Error{ErrorCode::kLifecycleStageFailed, std::string{exec.lifecycle_failure_token()}});
    outcome = RunTermination::kFailed;
  }
  if (outcome == RunTermination::kCompleted && !plan_->cuda_graph_execution_regions.empty()) {
    CUDAGraphRuntimeRegionIdentity failure_region;
    auto resources = RuntimeCUDAGraphRegionResources::create(
        *plan_,
        exec.operators(),
        RuntimeCUDAGraphTensorPoolBinding{
            .state = tensor_resources_.get(),
            .fixed_output_address =
                [](void* state, const PortId output) noexcept {
                  return static_cast<RuntimeTensorResources*>(state)->fixed_output_address(output);
                },
            .fixed_output_available =
                [](void* state, const PortId output) noexcept {
                  return static_cast<RuntimeTensorResources*>(state)->fixed_output_available(
                      output);
                },
            .restore_output_binding =
                [](void* state, const PortId output) noexcept {
                  return static_cast<RuntimeTensorResources*>(state)->restore_output_binding(
                      output);
                },
        },
        exec.process_group(),
        RuntimeCUDAGraphChannelBindings{
            .state = channel_session_.get(),
            .restore_input =
                [](void* state, const PortId input) noexcept {
                  return static_cast<RuntimeChannelSession*>(state)->restore_input_binding(input);
                },
            .restore_output =
                [](void* state, const PortId output) noexcept {
                  return static_cast<RuntimeChannelSession*>(state)->restore_output_binding(output);
                },
            .require_fixed_tensor =
                [](void* state,
                   const PortId input,
                   const void* address,
                   std::span<const std::uint64_t>
                       shape) noexcept {
                  return static_cast<RuntimeChannelSession*>(state)->require_fixed_tensor(
                      input, address, shape);
                },
            .clear_fixed_tensor =
                [](void* state, const PortId input, const void* address) noexcept {
                  return static_cast<RuntimeChannelSession*>(state)->clear_fixed_tensor_requirement(
                      input, address);
                },
        },
        &failure_region,
        plan_->executable_factories.test_cuda_execution_control);
    if (!resources) {
      append_diagnostic(
          registered_token_or(resources.error().message, kCUDAGraphRuntimeInitializationFailed),
          failure_region.fused_operator,
          "cuda_graph_region_construction",
          nullptr,
          {},
          cuda_graph_region_failure_evidence(failure_region, resources.error()),
          ::holoscan::DiagnosticSeverity::kError,
          failure_region.graph);
      outcome = RunTermination::kFailed;
    } else {
      cuda_graph_region_resources_ = std::move(*resources);
      auto captured = cuda_graph_region_resources_->capture_all(
          this, &RunSessionState::invoke_cuda_graph_capture_member, &failure_region);
      if (!captured) {
        if (captured.error().code == ErrorCode::kInterrupted &&
            stop_requested_.load(std::memory_order_acquire)) {
          outcome = RunTermination::kStopped;
        } else {
          append_diagnostic(
              registered_token_or(captured.error().message, kCUDAGraphRuntimeInitializationFailed),
              failure_region.fused_operator,
              "cuda_graph_capture",
              nullptr,
              {},
              cuda_graph_region_failure_evidence(failure_region, captured.error()),
              ::holoscan::DiagnosticSeverity::kError,
              failure_region.graph);
          outcome = RunTermination::kFailed;
        }
      }
    }
  }
  bool rate_candidate_ready = false;
  if (outcome == RunTermination::kCompleted) {
    auto preflight = rate_state_->preflight_candidate_publication();
    if (!preflight) {
      append_diagnostic(
          registered_token_or(preflight.error().message, "RUNTIME_LIFECYCLE_CALLBACK_FAILED"));
      pending_rate_candidate.set_failure(preflight.error());
      outcome = RunTermination::kFailed;
    } else if ((channel_session_ != nullptr &&
                !channel_session_->commit_rate_candidate_active_depths()) ||
               !runtime_internal::commit_rate_candidate_temporal_depths(ready_states)) {
      append_diagnostic("QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE");
      pending_rate_candidate.set_failure(
          Error{ErrorCode::kFailure, "QUEUE_ACTIVE_DEPTH_NOT_ENFORCEABLE"});
      outcome = RunTermination::kFailed;
    } else {
      rate_candidate_ready = true;
    }
  }
  if (outcome == RunTermination::kCompleted && shared.notifications != nullptr &&
      !shared.notifications->arm_startup()) {
    append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
    outcome = RunTermination::kFailed;
  }
  if (outcome == RunTermination::kCompleted) {
    if (ready_states.size() == 1U) {
      if (!bind_ready_sources(exec, ready_states.front())) {
        outcome = RunTermination::kFailed;
      }
    } else {
      for (ReadyRunState& ready : ready_states) {
        if (!bind_ready_sources(exec, ready, false)) {
          outcome = RunTermination::kFailed;
          break;
        }
      }
      if (outcome == RunTermination::kCompleted &&
          !refresh_and_arm_channels(shared.channels.get())) {
        outcome = RunTermination::kFailed;
      }
    }
  }
  if (outcome == RunTermination::kCompleted && cuda_graph_region_resources_ != nullptr) {
    auto validated = cuda_graph_region_resources_->validate_output_readiness_bindings();
    if (!validated) {
      append_diagnostic(
          registered_token_or(validated.error().message, "RUNTIME_READINESS_BINDING_UNAVAILABLE"));
      outcome = RunTermination::kFailed;
    }
  }
  if (outcome == RunTermination::kCompleted) {
    for (ReadyRunState& ready : ready_states) {
      if (!begin_zero_completion_closures(ready)) {
        append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
        outcome = RunTermination::kFailed;
        break;
      }
    }
  }
  if (outcome == RunTermination::kCompleted && !lifecycle_control_->open_ingress()) {
    if (stop_requested_.load(std::memory_order_acquire)) {
      outcome = RunTermination::kStopped;
    } else {
      append_diagnostic("RUNTIME_STARTUP_FAILED");
      outcome = RunTermination::kFailed;
    }
  }
  if (outcome == RunTermination::kCompleted && rate_candidate_ready) {
    auto published = rate_state_->publish_candidate();
    if (published) {
      pending_rate_candidate.publish();
    } else {
      append_diagnostic(
          registered_token_or(published.error().message, "RUNTIME_LIFECYCLE_CALLBACK_FAILED"));
      pending_rate_candidate.set_failure(published.error());
      outcome = RunTermination::kFailed;
    }
  }
  if (outcome == RunTermination::kStopped) {
    pending_rate_candidate.cancel();
  } else if (outcome != RunTermination::kCompleted) {
    rate_state_->reject_candidate(std::move(pending_rate_candidate.failure));
    pending_rate_candidate.terminal = true;
    if (channel_session_ != nullptr) {
      channel_session_->discard_rate_candidate_active_depths();
    }
    discard_rate_candidate_temporal_depths(ready_states);
  }
  return outcome;
}

// Keep startup, dispatch, and terminal cleanup in one transaction so ReadyRunState ownership and
// shutdown ordering remain visible at the call site.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] RunTermination RunSessionState::worker_main_impl() {
  // Before this controller or any lane is pinned, so CUDA's own threads do not inherit the pin.
  try {
    create_cuda_contexts_off_isolated_cpus(runtime_cuda_devices(*plan_, process_group_));
  } catch (...) {
    // A device list that cannot be allocated leaves context creation to the first CUDA use.
  }
  RuntimeSupportThreadPlacementFailure placement_failure;
  auto placement = apply_support_thread_affinity(plan_->support_thread_affinities,
                                                 SupportThreadRole::kController,
                                                 &placement_failure,
                                                 &pre_controller_affinity_);
  if (!placement) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR("Runtime controller thread placement failed with {}: {}",
                                 error_code_name(placement.error().code),
                                 placement.error().message);
    const RuntimeDiagnosticDetail detail{
        .present = true,
        .error = placement.error(),
        .operation = DiagnosticOperation::kRuntime,
        .component_path = "support-thread/controller",
    };
    append_diagnostic("RUNTIME_STARTUP_FAILED",
                      {},
                      {},
                      &detail,
                      runtime_support_thread_placement_required(placement_failure),
                      runtime_support_thread_placement_provided(placement_failure));
    return RunTermination::kFailed;
  }
  if (plan_->executable_factories.support_thread_control != nullptr) {
    auto entered =
        plan_->executable_factories.support_thread_control->enter(SupportThreadRole::kController);
    if (!entered) {
      HOLOSCAN_LOG_AUTOMATIC_ERROR("Runtime controller thread setup failed with {}: {}",
                                   error_code_name(entered.error().code),
                                   entered.error().message);
      append_diagnostic("RUNTIME_STARTUP_FAILED");
      return RunTermination::kFailed;
    }
  }
  RuntimeReplayMismatchDetail replay_mismatch;
  RuntimeDiagnosticDetail reconstruction_failure;
  auto reconstructed = [&]() -> expected<std::unique_ptr<GraphExec>, Error> {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      return GraphExec::reconstruct(
          plan_, run_generation_, process_group_, &replay_mismatch, &reconstruction_failure);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      if (!reconstruction_failure.capture || reconstruction_failure.capture->count == 0U) {
        reconstruction_failure.capture.emplace();
        reconstruction_failure.capture->capture_exception();
        if (reconstruction_failure.operation == DiagnosticOperation::kUnknown) {
          reconstruction_failure.operation = DiagnosticOperation::kRuntime;
        }
      } else {
        reconstruction_failure.capture->truncated = true;
        reconstruction_failure.component_path.clear();
      }
      reconstruction_failure.present = true;
      return make_unexpected(Error{ErrorCode::kFailure});
    }
#endif
  }();
  if (!reconstructed) {
    if (!reconstruction_failure.present && !replay_mismatch.present) {
      reconstruction_failure.present = true;
      reconstruction_failure.operation = DiagnosticOperation::kRuntime;
      reconstruction_failure.capture.emplace();
      reconstruction_failure.capture->append(reconstructed.error().code,
                                             reconstructed.error().message);
    }
    const std::string_view fallback =
        reconstruction_failure.operation == DiagnosticOperation::kSetup
            ? "RUNTIME_OPERATOR_SETUP_FAILED"
        : reconstruction_failure.operation == DiagnosticOperation::kContract
            ? "RUNTIME_OPERATOR_CALLBACK_FAILED"
        : reconstruction_failure.operation == DiagnosticOperation::kConstruct &&
                reconstruction_failure.vertex
            ? "RUNTIME_OPERATOR_RECONSTRUCTION_FAILED"
        : reconstruction_failure.operation == DiagnosticOperation::kConstruct &&
                reconstruction_failure.component_graph
            ? "RUNTIME_CLOCK_RECONSTRUCTION_FAILED"
            : "RUNTIME_STARTUP_FAILED";
    const std::string_view token = registered_token_or(reconstructed.error().message, fallback);
    if (replay_mismatch.present) {
      auto detail = std::move(reconstruction_failure);
      detail.present = true;
      detail.error = std::move(reconstructed.error());
      detail.operator_id = replay_mismatch.operator_id;
      detail.replay_mismatch = replay_mismatch;
      std::string expected;
      std::string observed;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        expected = lowercase_hex_text(replay_mismatch.expected.value, HexPrefix::kSha256);
        observed = lowercase_hex_text(replay_mismatch.observed.value, HexPrefix::kSha256);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (...) {
        if (!detail.capture) {
          detail.capture.emplace();
          detail.capture->append(detail.error.code, detail.error.message);
        }
        detail.capture->truncated = true;
      }
#endif
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_FINGERPRINT_MISMATCH",
                        replay_mismatch.vertex,
                        {},
                        &detail,
                        expected,
                        observed);
    } else {
      append_diagnostic(token,
                        reconstruction_failure.vertex,
                        {},
                        reconstruction_failure.present ? &reconstruction_failure : nullptr);
    }
    return RunTermination::kFailed;
  }
  graph_exec_ = std::move(reconstructed).value();
  if (graph_exec_->operators().empty()) {
    retire_parameter_control();
    return RunTermination::kCompleted;
  }
  if (auto bound = graph_exec_->parameter_updates().bind_lifecycle_submitter(
          lifecycle_control_->parameter_submit_binding());
      !bound) {
    append_diagnostic("RUNTIME_PARAMETER_CONTROL_BINDING_FAILED");
    retire_parameter_control();
    return RunTermination::kFailed;
  }
  if (auto attached = attach_parameter_control(graph_exec_->parameter_updates()); !attached) {
    append_diagnostic(
        registered_token_or(attached.error().message, "RUNTIME_PARAMETER_CONTROL_BINDING_FAILED"));
    retire_parameter_control();
    return RunTermination::kFailed;
  }
  GraphExec& exec = *graph_exec_;
  // Declaration order is load-bearing: ready states borrow pointers and spans from the shared
  // owner and therefore must be destroyed first on every return path.
  ReadySharedResources shared_ready_resources;
  std::vector<ReadyRunState> ready_states;
  if (!materialize_partition_runner_states(exec, ready_states)) {
    return RunTermination::kFailed;
  }
  ReadyRunState& shared_ready = ready_states.front();
  StartupRuntimeTeardown startup_teardown{*this, shared_ready_resources};
  if (!initialize_lanes(exec, shared_ready, shared_ready_resources)) {
    return RunTermination::kFailed;
  }
  auto rates = RuntimeRateState::create(*plan_,
                                        exec.operators(),
                                        run_generation_,
                                        plan_->scheduler_plan.execution_lane_count,
                                        exec.process_group());
  if (!rates) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR("Runtime rate-state construction failed with {}",
                                 rates.error().message);
    append_diagnostic(registered_token_or(rates.error().message, "RUNTIME_LIFECYCLE_PLAN_INVALID"));
    return RunTermination::kFailed;
  }
  rate_state_ = std::move(*rates);
  attach_rate_snapshot_state(*rate_state_);
  if (auto candidate = rate_state_->begin_startup_candidate(); !candidate) {
    append_diagnostic(
        registered_token_or(candidate.error().message, "RUNTIME_LIFECYCLE_PLAN_INVALID"));
    return RunTermination::kFailed;
  }
  PendingStartupRateCandidate pending_rate_candidate{.state = rate_state_.get()};
  const auto tensor_test_control = plan_->executable_factories.test_tensor_output_control;
  if (tensor_test_control != nullptr) {
    tensor_test_control->prepare(
        plan_->tensor_allocations.size() + plan_->tensor_field_allocations.size() +
        static_cast<std::size_t>(
            std::ranges::count_if(plan_->provider_allocation_plans, [](const auto& provider) {
              return provider.geometry_mode == AllocationGeometryMode::kLifecycleCandidate &&
                     provider.allocation_kind == ProviderAllocationKind::kTensorOutput &&
                     provider.disposition == PlanDisposition::kAccepted;
            })));
  }
  RuntimeCudaExecutionDiagnostic cuda_diagnostic;
  auto cuda_resources =
      RuntimeCudaExecutionResources::create(*plan_,
                                            exec.operators(),
                                            run_generation_,
                                            exec.process_group(),
                                            tensor_test_control,
                                            plan_->executable_factories.test_cuda_execution_control,
                                            &cuda_diagnostic);
  if (!cuda_resources) {
    HOLOSCAN_LOG_AUTOMATIC_ERROR("Runtime CUDA execution resource materialization failed: {}",
                                 runtime_error_message(cuda_resources.error()));
    const RuntimeDiagnosticDetail detail{
        .present = true,
        .error = cuda_resources.error(),
        .cuda_execution = std::move(cuda_diagnostic),
    };
    append_diagnostic(
        registered_token_or(cuda_resources.error().message, "PROFILE_MATERIALIZATION_FAILED"),
        {},
        {},
        &detail);
    return RunTermination::kFailed;
  }
  cuda_execution_resources_ = std::move(*cuda_resources);
  const bool defer_rate_resources = rate_state_->requires_deferred_resource_materialization();
  RateResourceMaterializationState rate_materialization{
      .session = this,
      .exec = &exec,
      .shared = &shared_ready_resources,
      .ready_states = std::span<ReadyRunState>{ready_states},
      .cuda = cuda_execution_resources_.get(),
      .defer_startup = defer_rate_resources,
      .resources_materialized = !defer_rate_resources,
  };
  if (!defer_rate_resources && !materialize_startup_resources(exec,
                                                              ready_states,
                                                              shared_ready_resources,
                                                              *cuda_execution_resources_,
                                                              *rate_state_,
                                                              pending_rate_candidate)) {
    return RunTermination::kFailed;
  }
  RunTermination outcome = arm_startup_sources(
      exec, ready_states, shared_ready_resources, rate_materialization, pending_rate_candidate);
  if (outcome == RunTermination::kCompleted && shared_ready.runner != nullptr) {
    count_phase(RuntimePhaseCounter::kPartitionControlBarriers);
  }
  if (outcome == RunTermination::kCompleted && ready_states.size() == 1U) {
    ReadyRunState& ready = shared_ready;
    const bool tracks_terminal_work =
        !ready.completion_receivers.empty() || !ready.input_terminal_receivers.empty();
    // Observation ownership is fixed before start(). Select the entire shared-owner loop once,
    // preserving direct calls at every dispatch batch for both observed and ordinary execution.
    if (fixture_observation_ == nullptr) {
      outcome = tracks_terminal_work ? run_shared_owner<true, false>(exec, ready)
                                     : run_shared_owner<false, false>(exec, ready);
    } else {
      outcome = tracks_terminal_work ? run_shared_owner<true, true>(exec, ready)
                                     : run_shared_owner<false, true>(exec, ready);
    }
  } else if (outcome == RunTermination::kCompleted) {
    PartitionOwnerGroup group;
    std::vector<PartitionOwnerContext> contexts;
    std::vector<ExecutionLaneOwnerTask> tasks;
    contexts.reserve(ready_states.size());
    tasks.reserve(ready_states.size());
    for (ReadyRunState& ready : ready_states) {
      contexts.push_back(PartitionOwnerContext{
          .session = this,
          .exec = &exec,
          .ready = &ready,
          .group = &group,
      });
    }
    for (PartitionOwnerContext& context : contexts) {
      const bool tracks_terminal_work = !context.ready->completion_receivers.empty() ||
                                        !context.ready->input_terminal_receivers.empty();
      tasks.push_back(ExecutionLaneOwnerTask{
          .execution_lane = context.ready->runner->execution_lane,
          .context = &context,
          .invoke =
              tracks_terminal_work ? &invoke_partition_owner<true> : &invoke_partition_owner<false>,
      });
    }
    while (outcome == RunTermination::kCompleted) {
      group.failed.store(false, std::memory_order_release);
      for (PartitionOwnerContext& context : contexts) {
        context.outcome = RunTermination::kCompleted;
      }
      auto launched = shared_ready_resources.lanes->launch_owner_tasks(tasks);
      auto waited = launched ? shared_ready_resources.lanes->wait_owner_tasks()
                             : expected<void, Error>{make_unexpected(launched.error())};
      if (!waited || group.failed.load(std::memory_order_acquire)) {
        if (!waited) {
          append_diagnostic(
              registered_token_or(waited.error().message, "RUNTIME_EXECUTION_LANE_PLAN_INVALID"));
        }
        outcome = RunTermination::kFailed;
        break;
      }
      if (stop_requested_.load(std::memory_order_acquire) ||
          std::ranges::any_of(contexts, [](const PartitionOwnerContext& context) {
            return context.outcome == RunTermination::kStopped;
          })) {
        outcome = RunTermination::kStopped;
        break;
      }
      if (!lifecycle_control_->has_pending()) {
        break;
      }
      count_phase(RuntimePhaseCounter::kPartitionControlBarriers);
      if (!process_pending_restarts(exec, ready_states)) {
        append_lifecycle_diagnostics(exec);
        outcome = RunTermination::kFailed;
        break;
      }
    }
  }
  // From here onward the common shutdown sequence owns every resource covered by the startup
  // guard, including failure outcomes discovered during lifecycle startup or dispatch.
  startup_teardown.dismiss();
  for (ReadyRunState& ready : ready_states) {
    if (!cancel_ready_sources(ready)) {
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
      outcome = RunTermination::kFailed;
    }
    // A source failure outranks a stop or requested completion. An idle round skips the late drain
    // that would have reported it, and an owner audits its sources only once all of them closed,
    // so audit them here. Cancellation keeps a source's failure.
    if (outcome != RunTermination::kFailed && !sources_closed_without_failure(ready)) {
      outcome = RunTermination::kFailed;
    }
  }
  if (ready_states.size() > 1U && shared_ready_resources.notifications != nullptr) {
    shared_ready_resources.notifications->close();
  }
  RuntimeCudaExecutionDiagnostic cuda_graph_diagnostic;
  expected<void, Error> graph_quiesced =
      cuda_graph_region_resources_ == nullptr
          ? expected<void, Error>{}
          : cuda_graph_region_resources_->quiesce(&cuda_graph_diagnostic);
  if (!graph_quiesced) {
    if (plan_->executable_factories.test_cuda_execution_control != nullptr) {
      plan_->executable_factories.test_cuda_execution_control->notify_failure_observer(
          RuntimeCudaExecutionObserverPoint::kQuiesceFailureRetention);
    }
    // Retain the one owning error without recopying its message on this failure-cleanup path.
    cuda_graph_quiesce_failure_.emplace(std::move(graph_quiesced).error());
    if (cuda_execution_resources_ != nullptr) {
      cuda_execution_resources_->quarantine_after_external_failure();
    }
    append_cuda_graph_quiesce_failure(*cuda_graph_quiesce_failure_,
                                      std::move(cuda_graph_diagnostic));
    outcome = RunTermination::kFailed;
  } else {
    if (!stop_channels(shared_ready_resources.channels.get())) {
      outcome = RunTermination::kFailed;
    }
    if (tensor_resources_ != nullptr) {
      tensor_resources_->close();
    }
    if (!exec.lifecycle_shutdown()) {
      append_lifecycle_diagnostics(exec);
      outcome = RunTermination::kFailed;
    }
  }
  if (!stop_lanes(shared_ready_resources.lanes.get())) {
    outcome = RunTermination::kFailed;
  }
  if (outcome == RunTermination::kCompleted && stop_requested_.load(std::memory_order_acquire)) {
    outcome = RunTermination::kStopped;
  }
  return outcome;
}

void RunSessionState::finish(RunTermination outcome) noexcept {
  if (lifecycle_control_ != nullptr) {
    // worker_main() reaches finish only after worker_main_impl() has unwound
    // every callback, readiness binding, and lane owner. The same idempotent
    // finalizer closes a worker-construction failure before its terminal
    // session can mint a still-starting lifecycle capability.
    lifecycle_control_->finalize_close();
  }
  if (terminal_participant_ != nullptr && !terminal_participant_->before_terminal_freeze(outcome)) {
    outcome = RunTermination::kFailed;
  }
  finish_after_terminal_participant(outcome);
}

// Callback identity stays compact on the activation path. Construct optional owning evidence only
// here so the reserved cause capacity cannot introduce stack-page probes into successful dispatch.
void RunSessionState::append_callback_diagnostic(
    std::string_view token, VertexId vertex, std::string_view service, const Error& error,
    const RuntimeCallbackIdentity& identity, std::string_view predicate_id,
    const RuntimeMaxStartDelayMissDiagnostic* max_start_delay,
    ::holoscan::DiagnosticSeverity severity, const DiagnosticOperation operation) noexcept {
  RuntimeDiagnosticDetail detail{
      .present = true,
      .error = Error{error.code},
      .operator_id = identity.operator_id,
      .source = identity.source,
      .source_sequence = identity.source_sequence,
      .tag = identity.tag,
      .tag_time = identity.tag_time,
      .reference_time = identity.reference_time,
      .state_generation = identity.state_generation,
  };
  detail.operation = operation;
  if (error.code != ErrorCode::kSuccess || !error.message.empty()) {
    detail.capture.emplace();
    detail.capture->append(error.code, error.message);
  }
  if (max_start_delay != nullptr) {
    detail.max_start_delay_miss = *max_start_delay;
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    detail.predicate_id.assign(predicate_id);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    if (detail.capture) {
      detail.capture->truncated = true;
    }
  }
#endif
  append_diagnostic(token, vertex, service, &detail, {}, {}, severity);
}

// Keep the large owning failure snapshots off recurring dispatch stack frames. This translation
// unit contains terminal publication; successful callbacks never call this function.
bool RunSessionState::compute_failure_caused_by_stop(const RuntimeChannelSession* const channels,
                                                     const RuntimeOperatorInstance& instance,
                                                     const ComputeLaneInvocation& invocation,
                                                     const std::uint64_t activation_generation,
                                                     const Error& error) const noexcept {
  const ErrorCode code = error.code;
  return invocation.callback_failed && !invocation.cuda_device_restore_failed &&
         error.provenance != 0U &&
         (code == ErrorCode::kBackpressured || code == ErrorCode::kPublicationPartial ||
          code == ErrorCode::kPublicationIndeterminate) &&
         (termination_state_.load(std::memory_order_acquire) & kRequestedStop) != 0U &&
         channels != nullptr &&
         channels->publication_cancelled_by_stop(instance.vertex, activation_generation, error);
}

void RunSessionState::append_compute_diagnostics(const RuntimeOperatorInstance& instance,
                                                 const ComputeLaneInvocation& invocation,
                                                 const Error& error,
                                                 const RuntimeCallbackIdentity& identity,
                                                 const bool stopped_publication) noexcept {
  if (!invocation.callback_failed && !invocation.cuda_device_restore_failed) {
    DiagnosticCapture dispatch_failure;
    dispatch_failure.append(error.code, error.message);
    const RuntimeDiagnosticDetail detail{
        .present = true,
        .error = Error{error.code},
        .operator_id = instance.operator_id,
        .capture = dispatch_failure,
        .operation = DiagnosticOperation::kRuntime,
    };
    append_diagnostic("RUNTIME_COMPUTE_DISPATCH_FAILED", instance.vertex, {}, &detail);
  }
  if (invocation.callback_failed) {
    const RuntimeDiagnosticDetail detail{
        .present = true,
        .error = Error{error.code},
        .operator_id = instance.operator_id,
        .source = identity.source,
        .source_sequence = identity.source_sequence,
        .tag = identity.tag,
        .tag_time = identity.tag_time,
        .reference_time = identity.reference_time,
        .state_generation = identity.state_generation,
        .capture = instance.failure_capture,
        .tensor_failure = instance.tensor_failure,
        .operation = DiagnosticOperation::kCompute,
    };
    if (stopped_publication) {
      // The run ends stopped: this callback failed only because the stop closed admission while
      // one of its publications waited for a recipient that will never run again.
      append_diagnostic("RUNTIME_OPERATOR_PUBLICATION_STOPPED",
                        instance.vertex,
                        "operator_compute",
                        &detail,
                        {},
                        {},
                        ::holoscan::DiagnosticSeverity::kWarning);
    } else {
      append_compute_failure_diagnostic(instance, detail, error);
    }
  }
  if (invocation.cuda_device_restore_failed) {
    const RuntimeDiagnosticDetail detail{
        .present = true,
        .error = Error{ErrorCode::kNotSupported},
        .operator_id = instance.operator_id,
        .operation = DiagnosticOperation::kRuntime,
        .role = invocation.callback_failed ? DiagnosticFailureRole::kTeardown
                                           : DiagnosticFailureRole::kPrimary,
        .follows_compute_failure = invocation.callback_failed,
    };
    append_diagnostic("RUNTIME_COMPUTE_CUDA_DEVICE_RESTORE_FAILED", instance.vertex, {}, &detail);
  }
}

void RunSessionState::append_lifecycle_diagnostics(
    const GraphExec& exec, ::holoscan::DiagnosticSeverity severity) noexcept {
  const auto failures = exec.lifecycle_failures();
  if (failures.empty()) {
    if (severity != ::holoscan::DiagnosticSeverity::kError) {
      return;
    }
    const auto* result = exec.lifecycle_result();
    if (result == nullptr || result->disposition == LifecycleExecutionDisposition::kCompleted) {
      append_diagnostic(exec.lifecycle_failure_token());
      return;
    }
    constexpr std::array reasons{"completed",
                                 "lifecycle transition was rolled back",
                                 "invalid lifecycle state",
                                 "invalid lifecycle restart stage",
                                 "invalid lifecycle callback binding",
                                 "lifecycle stage failed",
                                 "lifecycle rollback failed",
                                 "lifecycle capacity exhausted"};
    const auto index = static_cast<std::size_t>(result->disposition);
    RuntimeDiagnosticDetail detail{.present = true, .operation = DiagnosticOperation::kLifecycle};
    detail.capture.emplace();
    detail.capture->append(result->disposition == LifecycleExecutionDisposition::kCapacityExceeded
                               ? ErrorCode::kResourceExhausted
                               : ErrorCode::kLifecycleStageFailed,
                           index < reasons.size() ? reasons[index] : "unknown lifecycle failure");
    if (result->primary_failure.present) {
      detail.stage = result->primary_failure.stage;
      detail.phase = result->primary_failure.phase;
      if (const auto* owner = exec.operator_by_id(result->primary_failure.node_id)) {
        detail.vertex = owner->vertex;
      }
    }
    append_diagnostic(exec.lifecycle_failure_token(), detail.vertex, "lifecycle", &detail);
    return;
  }
  for (const auto& failure : failures) {
    RuntimeDiagnosticDetail detail{
        .present = true,
        .capture = failure.capture,
        .operation = DiagnosticOperation::kLifecycle,
        .stage = failure.failure.stage,
        .phase = failure.failure.phase,
        .transaction_id = failure.transaction_id,
    };
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      detail.component_path = failure.component_path;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      detail.capture->truncated = true;
    }
#endif
    append_diagnostic(failure.token, failure.vertex, "lifecycle", &detail, {}, {}, severity);
  }
}

void RunSessionState::finish_staged_start_failure() noexcept {
  if (lifecycle_control_ != nullptr) {
    lifecycle_control_->finalize_close();
  }
  finish_after_terminal_participant(RunTermination::kFailed);
}

void RunSessionState::finish_after_terminal_participant(RunTermination outcome) noexcept {
  // This fetch-or is the final publication boundary. It closes an otherwise-open gate, prevents
  // later request mutation, and snapshots all request bits after every lane owner has retired.
  const std::uint64_t termination_state =
      termination_state_.fetch_or(kCallbackAdmissionClosed | kTerminationFrozen,
                                  std::memory_order_acq_rel) |
      kCallbackAdmissionClosed | kTerminationFrozen;
  if ((termination_state & kCallbackClaimCountMask) != 0U) {
    append_diagnostic(
        "RUNTIME_OPERATOR_REPLAY_MISMATCH", {}, "termination_admission_claims_not_retired");
    outcome = RunTermination::kFailed;
  }
  if (outcome != RunTermination::kFailed) {
    if ((termination_state & kRequestedStop) != 0U) {
      outcome = RunTermination::kStopped;
    } else if ((termination_state & kRequestedCompletion) != 0U) {
      outcome = RunTermination::kRequestedCompleted;
    }
  }
  {
    const std::lock_guard diagnostic_lock(diagnostic_mutex_);
    diagnostic_buffer_.finalize();
  }
  if (plan_ != nullptr && plan_->runtime_diagnostic_sink != nullptr) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      plan_->runtime_diagnostic_sink->publish(diagnostics());
      plan_->runtime_diagnostic_sink->publish_runtime_details(diagnostic_buffer_.details());
      plan_->runtime_diagnostic_sink->publish_lane_startup_evidence(lane_startup_evidence_);
      plan_->runtime_diagnostic_sink->publish_operator_completion_progress(
          operator_completion_progress_);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      // Observability is subordinate to execution. A private sink cannot
      // replace or weaken the already-determined cleanup and run outcome.
      static_cast<void>(0);
    }
#endif
  }
  {
    const std::lock_guard lock(join_mutex_);
    termination_.store(outcome, std::memory_order_release);
  }
  join_cv_.notify_all();
  rate_snapshot_cv_.notify_all();
  if (terminal_participant_ != nullptr) {
    terminal_participant_->after_terminal_freeze();
  }
  if (outcome == RunTermination::kFailed) {
    publish_runtime_terminal_summary(
        run_generation_,
        diagnostics(),
        plan_ != nullptr ? plan_->executable_factories.terminal_summary_sink.get() : nullptr);
  }
}

template <bool TracksTerminalWork, bool Observed>
// The shared owner preserves the serialized lifecycle/readiness transaction used by partitions.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] RunTermination RunSessionState::run_shared_owner(GraphExec& exec,
                                                               ReadyRunState& ready) noexcept {
  RunTermination outcome = RunTermination::kCompleted;
  while (outcome == RunTermination::kCompleted) {
    if (stop_requested_.load(std::memory_order_acquire)) {
      return RunTermination::kStopped;
    }
    if (!process_pending_restarts(exec, ready)) {
      append_lifecycle_diagnostics(exec);
      return RunTermination::kFailed;
    }
    if constexpr (TracksTerminalWork) {
      if (!ready.active_completion_receivers.empty() && !advance_completion_closures(exec, ready)) {
        return RunTermination::kFailed;
      }
      if (!ready.active_input_terminal_receivers.empty() &&
          !advance_input_terminal_drains(exec, ready)) {
        return RunTermination::kFailed;
      }
    }
    // Consume signals that preceded this readiness scan. A concurrent signal remains as a
    // durable token, so the timed wait below cannot miss the scan-to-wait transition.
    RuntimeWakeState& wake = ready.wake != nullptr ? *ready.wake : readiness_wake_;
    static_cast<void>(wake.drain());
    count_phase(RuntimePhaseCounter::kReadinessScans);
    count_phase(RuntimePhaseCounter::kPartitionRunnerSweeps);
    if (!freeze_ready_candidates(exec, ready, ready.inline_execution_lane)) {
      return RunTermination::kFailed;
    }
    if (ready.dispatcher->active_count() != 0U) {
      publish_tick_fences(ready);
      outcome = dispatch_ready_impl<Observed>(exec, ready);
      if (outcome != RunTermination::kCompleted) {
        return outcome;
      }
      count_phase(RuntimePhaseCounter::kPartitionTailDispatchAttempts);
      count_phase(ready.direct_chain.continuation_receipt
                      ? RuntimePhaseCounter::kPartitionTailDispatchAccepted
                      : RuntimePhaseCounter::kPartitionTailDispatchFallbacks);
      continue;
    }
    if (all_sources_closed<TracksTerminalWork>(exec, ready)) {
      return sources_closed_without_failure(ready) ? RunTermination::kCompleted
                                                   : RunTermination::kFailed;
    }
    count_phase(RuntimePhaseCounter::kNoProgressScans);
    if (!wait_for_source_progress(exec, ready)) {
      return RunTermination::kFailed;
    }
  }
  return outcome;
}

}  // namespace holoscan::detail
