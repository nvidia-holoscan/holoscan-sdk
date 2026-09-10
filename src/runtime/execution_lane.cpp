/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "execution_lane.hpp"

#include "runtime_phase_counters.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#include <intrin.h>
#endif

#ifdef __linux__
#include "../threading/linux_thread_controller.hpp"
#endif

#include "../threading/thread_placement_contracts.hpp"

namespace holoscan::detail {
namespace {

void count_phase(RuntimePhaseCounters* counters, RuntimePhaseCounter counter) noexcept {
  if (counters != nullptr) {
    counters->increment(counter);
  }
}

[[nodiscard]] expected<void, Error> invoke_lane_task(const ExecutionLaneTask& task,
                                                     RuntimePhaseCounters* counters) {
  count_phase(counters, RuntimePhaseCounter::kLaneCallbacksEntered);
  struct ExitCounter {
    RuntimePhaseCounters* counters;
    ~ExitCounter() { count_phase(counters, RuntimePhaseCounter::kLaneCallbacksExited); }
  } const exit_counter{counters};
  return task.invoke(task.context);
}

[[nodiscard]] unexpected<Error> lane_error(ErrorCode code, std::string message) {
  return make_unexpected(Error{code, std::move(message)});
}

[[nodiscard]] unexpected<ExecutionLanePlacementError> placement_error(
    ExecutionLanePlacementErrorKind kind, Error error) {
  return make_unexpected(ExecutionLanePlacementError{kind, std::move(error)});
}

[[nodiscard]] ExecutionLanePlacementErrorKind classify_system_error(ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::kNotImplemented:
    case ErrorCode::kNotSupported:
    case ErrorCode::kNotReady:
    case ErrorCode::kPermissionDenied:
      return ExecutionLanePlacementErrorKind::kUnavailable;
    default:
      return ExecutionLanePlacementErrorKind::kFailed;
  }
}

[[nodiscard]] expected<void, Error> verify_snapshot(
    const ExecutionLaneOsPlacement& accepted, const ExecutionLanePlacementSnapshot& observed) {
  if (accepted.affinity && accepted.affinity->cores != observed.affinity.cores) {
    return lane_error(ErrorCode::kIncompatible, "RUNTIME_LANE_AFFINITY_READBACK_MISMATCH");
  }
  if (accepted.scheduling.policy != observed.policy ||
      accepted.scheduling.priority != observed.priority ||
      !deadline_scheduling_equal(accepted.scheduling.deadline, observed.deadline)) {
    return lane_error(ErrorCode::kIncompatible, "RUNTIME_LANE_SCHEDULING_READBACK_MISMATCH");
  }
  if (accepted.scheduling.policy != SchedulingPolicy::kOther && !observed.reset_on_fork) {
    return lane_error(ErrorCode::kIncompatible, "RUNTIME_LANE_RESET_ON_FORK_READBACK_MISMATCH");
  }
  return {};
}

class SystemExecutionLanePlacementAdapter final : public ExecutionLanePlacementAdapter {
 public:
  [[nodiscard]] expected<void, ExecutionLanePlacementError> apply(
      const ExecutionLaneOsPlacement& placement) override {
#ifdef __linux__
    const ThreadPlacement requested{
        .affinity = placement.affinity,
        .scheduling = placement.scheduling,
    };
    auto result = LinuxThreadController::apply(requested);
    if (!result) {
      return placement_error(classify_system_error(result.error().code), result.error());
    }
    return {};
#else
    return validate_execution_lane_platform_placement(false, placement);
#endif
  }

  [[nodiscard]] expected<ExecutionLanePlacementSnapshot, ExecutionLanePlacementError> snapshot()
      override {
#ifdef __linux__
    auto result = LinuxThreadController::snapshot();
    if (!result) {
      return placement_error(classify_system_error(result.error().code), result.error());
    }
    return ExecutionLanePlacementSnapshot{
        .affinity = std::move(result->affinity),
        .policy = result->policy,
        .priority = result->priority,
        .deadline = result->deadline,
        .reset_on_fork = result->reset_on_fork,
    };
#else
    return ExecutionLanePlacementSnapshot{};
#endif
  }

  [[nodiscard]] expected<void, ExecutionLanePlacementError> restore(
      const ExecutionLanePlacementSnapshot& snapshot) override {
#ifdef __linux__
    auto result = LinuxThreadController::restore(LinuxThreadSnapshot{
        .affinity = snapshot.affinity,
        .policy = snapshot.policy,
        .priority = snapshot.priority,
        .deadline = snapshot.deadline,
        .reset_on_fork = snapshot.reset_on_fork,
    });
    if (!result) {
      return placement_error(classify_system_error(result.error().code), result.error());
    }
    return {};
#else
    (void)snapshot;
    return {};
#endif
  }
};

[[nodiscard]] ExecutionLaneOsPlacement os_placement(const ThreadPlacement& placement) {
  return ExecutionLaneOsPlacement{
      .affinity = placement.affinity,
      .scheduling = placement.scheduling,
  };
}

inline void busy_wait_pause() noexcept {
#if defined(__i386__) || defined(__x86_64__)
  __builtin_ia32_pause();
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
  _mm_pause();
#elif defined(__arm__) || defined(__aarch64__)
  __asm__ __volatile__("yield");
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

}  // namespace

std::string_view execution_lane_startup_diagnostic_token(
    const ExecutionLaneStartupEvidence& evidence) noexcept {
  if (evidence.disposition == ExecutionLaneStartupDisposition::kUnavailable && evidence.error &&
      evidence.error->code == ErrorCode::kNotSupported) {
    return "PLATFORM_CAPABILITY_UNAVAILABLE";
  }
  if (evidence.error && !evidence.error->message.empty()) {
    return evidence.error->message;
  }
  return "RUNTIME_LANE_STARTUP_FAILED";
}

expected<void, ExecutionLanePlacementError> validate_execution_lane_platform_placement(
    bool platform_supports_os_placement, const ExecutionLaneOsPlacement& placement) {
  const bool requests_os_placement = placement.affinity ||
                                     placement.scheduling.policy != SchedulingPolicy::kOther ||
                                     placement.scheduling.priority || placement.scheduling.deadline;
  if (platform_supports_os_placement || !requests_os_placement) {
    return {};
  }
  return placement_error(ExecutionLanePlacementErrorKind::kUnavailable,
                         Error{ErrorCode::kNotSupported, "RUNTIME_LANE_OS_PLACEMENT_UNAVAILABLE"});
}

struct ExecutionLaneSet::Impl {
  enum class State : std::uint8_t {
    kCreated = 0,
    kStarting = 1,
    kReady = 2,
    kStopping = 3,
    kFailed = 4,
    kJoined = 5,
  };

  enum class BusyTaskState : std::uint8_t {
    kIdle = 0,
    kPreparing = 1,
    kPending = 2,
    kRunning = 3,
    kComplete = 4,
    kStopped = 5,
  };

  enum class OwnerTaskState : std::uint8_t {
    kUnassigned = 0,
    kPending = 1,
    kRunning = 2,
    kComplete = 3,
  };

  struct Lane {
    explicit Lane(ExecutionLanePlanRecord source, bool inline_owner)
        : plan(std::move(source)), inline_owner(inline_owner) {}

    ExecutionLanePlanRecord plan;
    bool inline_owner{};
    std::atomic_bool inline_dispatch_active{};
    std::thread::id inline_owner_thread;
    std::mutex inline_mutex;
    std::condition_variable inline_cv;
    bool inline_stop_requested{};
    std::optional<ExecutionLanePlacementSnapshot> inline_original_placement;
    std::optional<Error> inline_restore_error;
    std::thread worker;
    std::mutex mutex;
    std::condition_variable cv;
    bool stop_requested{};
    bool task_pending{};
    bool task_running{};
    bool task_complete{};
    std::atomic<bool> busy_stop_requested{false};
    std::atomic<BusyTaskState> busy_task_state{BusyTaskState::kIdle};
    ExecutionLaneTask task;
    std::optional<Error> task_error;
    std::mutex owner_task_mutex;
    std::condition_variable owner_task_cv;
    std::atomic<OwnerTaskState> owner_task_state{OwnerTaskState::kUnassigned};
    ExecutionLaneOwnerTask owner_task;
    std::optional<Error> owner_task_error;
    ExecutionLaneStartupPhase startup_phase{ExecutionLaneStartupPhase::kWorkerStart};
  };

  Impl(SchedulerPlanRecord scheduler, std::span<const ExecutionLanePlanRecord> source,
       std::shared_ptr<ExecutionLanePlacementAdapter> adapter, RuntimePhaseCounters* counters,
       std::optional<std::uint32_t> inline_execution_lane)
      : scheduler(std::move(scheduler)),
        placement_adapter(std::move(adapter)),
        phase_counters(counters),
        lane_index_by_execution_lane(this->scheduler.execution_lane_count,
                                     std::numeric_limits<std::size_t>::max()) {
    lanes.reserve(source.size());
    evidence.reserve(source.size());
    for (const auto& record : source) {
      lane_index_by_execution_lane[record.execution_lane] = lanes.size();
      lanes.push_back(std::make_unique<Lane>(
          record, inline_execution_lane && *inline_execution_lane == record.execution_lane));
      evidence.push_back(ExecutionLaneStartupEvidence{
          .execution_lane = record.execution_lane,
          .partition = record.partition,
          .wake_mode = record.accepted_placement.wake_mode,
          .irq_isolation = record.accepted_placement.irq_isolation,
          .applied = std::nullopt,
          .error = std::nullopt,
      });
    }
  }

  [[nodiscard]] expected<void, Error> restore_inline_placement(Lane& lane) {
    std::optional<ExecutionLanePlacementSnapshot> original;
    {
      const std::lock_guard inline_lock(lane.inline_mutex);
      if (!lane.inline_original_placement) {
        if (lane.inline_restore_error) {
          return make_unexpected(*lane.inline_restore_error);
        }
        return {};
      }
      if (lane.inline_owner_thread != std::this_thread::get_id()) {
        return lane_error(ErrorCode::kInvalidArgument, "RUNTIME_INLINE_LANE_OWNER_THREAD_REQUIRED");
      }
      original = std::move(lane.inline_original_placement);
      lane.inline_original_placement.reset();
    }

    std::optional<Error> failure;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      auto restored = placement_adapter->restore(*original);
      if (!restored) {
        failure = std::move(restored).error().error;
      }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (const std::exception& exception) {
      failure = Error{ErrorCode::kFailure, exception.what()};
    } catch (...) {
      failure = Error{ErrorCode::kFailure, "RUNTIME_LANE_PLACEMENT_RESTORE_FAILED"};
    }
#endif

    const std::lock_guard inline_lock(lane.inline_mutex);
    if (failure) {
      lane.inline_original_placement = std::move(original);
      lane.inline_restore_error = *failure;
      return make_unexpected(std::move(*failure));
    }
    lane.inline_restore_error.reset();
    return {};
  }

  [[nodiscard]] bool publish_startup_failure(std::size_t index,
                                             ExecutionLaneStartupDisposition disposition,
                                             ExecutionLaneStartupPhase phase, Error error) {
    Lane& lane = *lanes[index];
    if (lane.inline_owner) {
      auto restored = restore_inline_placement(lane);
      if (!restored) {
        error = std::move(restored).error();
      }
    }
    static_cast<void>(
        publish_startup(index, make_failure(index, disposition, phase, std::move(error))));
    return false;
  }

  [[nodiscard]] bool prepare_lane_startup(std::size_t index) {
    Lane& lane = *lanes[index];
    const ExecutionLaneOsPlacement accepted = os_placement(lane.plan.accepted_placement);

    if (lane.inline_owner) {
      lane.startup_phase = ExecutionLaneStartupPhase::kPlacementReadback;
      auto original = placement_adapter->snapshot();
      if (!original) {
        const auto disposition =
            original.error().kind == ExecutionLanePlacementErrorKind::kUnavailable
                ? ExecutionLaneStartupDisposition::kUnavailable
                : ExecutionLaneStartupDisposition::kFailed;
        return publish_startup_failure(index,
                                       disposition,
                                       ExecutionLaneStartupPhase::kPlacementReadback,
                                       std::move(original).error().error);
      }
      bool stopped = false;
      {
        const std::lock_guard inline_lock(lane.inline_mutex);
        stopped = lane.inline_stop_requested;
        if (!stopped) {
          lane.inline_original_placement = std::move(*original);
        }
      }
      if (stopped) {
        return publish_startup_failure(
            index,
            ExecutionLaneStartupDisposition::kStopped,
            ExecutionLaneStartupPhase::kPlacementReadback,
            Error{ErrorCode::kInterrupted, "RUNTIME_LANE_STARTUP_CANCELLED"});
      }
    }

    lane.startup_phase = ExecutionLaneStartupPhase::kPlacementApply;
    auto applied = placement_adapter->apply(accepted);
    if (!applied) {
      const auto disposition = applied.error().kind == ExecutionLanePlacementErrorKind::kUnavailable
                                   ? ExecutionLaneStartupDisposition::kUnavailable
                                   : ExecutionLaneStartupDisposition::kFailed;
      return publish_startup_failure(index,
                                     disposition,
                                     ExecutionLaneStartupPhase::kPlacementApply,
                                     std::move(applied).error().error);
    }

    lane.startup_phase = ExecutionLaneStartupPhase::kPlacementReadback;
    auto snapshot = placement_adapter->snapshot();
    if (!snapshot) {
      const auto disposition =
          snapshot.error().kind == ExecutionLanePlacementErrorKind::kUnavailable
              ? ExecutionLaneStartupDisposition::kUnavailable
              : ExecutionLaneStartupDisposition::kFailed;
      return publish_startup_failure(index,
                                     disposition,
                                     ExecutionLaneStartupPhase::kPlacementReadback,
                                     std::move(snapshot).error().error);
    }

    lane.startup_phase = ExecutionLaneStartupPhase::kPlacementVerify;
    auto verified = verify_snapshot(accepted, *snapshot);
    if (!verified) {
      return publish_startup_failure(index,
                                     ExecutionLaneStartupDisposition::kFailed,
                                     ExecutionLaneStartupPhase::kPlacementVerify,
                                     std::move(verified).error());
    }

    lane.startup_phase = ExecutionLaneStartupPhase::kAllLanesBarrier;
    ExecutionLaneStartupEvidence ready{
        .execution_lane = lane.plan.execution_lane,
        .partition = lane.plan.partition,
        .wake_mode = lane.plan.accepted_placement.wake_mode,
        .irq_isolation = lane.plan.accepted_placement.irq_isolation,
        .disposition = ExecutionLaneStartupDisposition::kReady,
        .phase = ExecutionLaneStartupPhase::kAllLanesBarrier,
        .applied = std::move(*snapshot),
        .error = std::nullopt,
    };
    return publish_startup(index, std::move(ready));
  }

  void count_dispatch_context(const ExecutionLaneTask& task) const noexcept {
    count_phase(phase_counters, RuntimePhaseCounter::kLaneDispatches);
    RuntimeCallbackContext context = task.callback_identity.context;
    if (context == RuntimeCallbackContext::kNone) {
      context = task.callback_context;
    }
    switch (context) {
      case RuntimeCallbackContext::kCompute:
        count_phase(phase_counters, RuntimePhaseCounter::kLaneComputeDispatches);
        break;
      case RuntimeCallbackContext::kDispatcher:
        count_phase(phase_counters, RuntimePhaseCounter::kLaneDispatcherDispatches);
        break;
      case RuntimeCallbackContext::kLifecycle:
        count_phase(phase_counters, RuntimePhaseCounter::kLaneLifecycleDispatches);
        break;
      case RuntimeCallbackContext::kNone:
      default:
        count_phase(phase_counters, RuntimePhaseCounter::kLaneUnknownDispatches);
        break;
    }
  }

  [[nodiscard]] ExecutionLaneStartupEvidence make_failure(
      std::size_t index, ExecutionLaneStartupDisposition disposition,
      ExecutionLaneStartupPhase phase, Error error) const {
    ExecutionLaneStartupEvidence result{
        .execution_lane = lanes[index]->plan.execution_lane,
        .partition = lanes[index]->plan.partition,
        .wake_mode = lanes[index]->plan.accepted_placement.wake_mode,
        .irq_isolation = lanes[index]->plan.accepted_placement.irq_isolation,
        .disposition = disposition,
        .phase = phase,
        .applied = std::nullopt,
        .error = std::move(error),
    };
    return result;
  }

  [[nodiscard]] bool publish_startup(std::size_t index,
                                     ExecutionLaneStartupEvidence result) noexcept {
    const std::lock_guard lock(startup_mutex);
    if (index >= evidence.size() ||
        evidence[index].disposition != ExecutionLaneStartupDisposition::kPending) {
      return false;
    }
    evidence[index] = std::move(result);
    ++startup_completed;
    if (evidence[index].disposition != ExecutionLaneStartupDisposition::kReady &&
        !first_startup_failure) {
      first_startup_failure = index;
    }
    startup_cv.notify_all();
    return true;
  }

  void handle_lane_exception(std::size_t index, Error error) noexcept {
    if (publish_startup(index,
                        make_failure(index,
                                     ExecutionLaneStartupDisposition::kFailed,
                                     lanes[index]->startup_phase,
                                     error))) {
      return;
    }

    // Startup evidence is immutable after the all-lanes barrier. If a worker
    // nevertheless fails later, close the complete lane set and release any
    // caller blocked on that lane instead of republishing startup evidence or
    // leaving a dispatch waiting for a worker that has exited.
    Lane& lane = *lanes[index];
    if (lane.plan.accepted_placement.wake_mode == WakeMode::kBusyPoll) {
      lane.busy_stop_requested.store(true, std::memory_order_release);
      const BusyTaskState task_state = lane.busy_task_state.load(std::memory_order_acquire);
      if (task_state == BusyTaskState::kPending || task_state == BusyTaskState::kRunning) {
        lane.task_error = std::move(error);
        lane.task.context = nullptr;
        lane.task.invoke = nullptr;
        lane.busy_task_state.store(BusyTaskState::kComplete, std::memory_order_release);
      }
    } else {
      const std::lock_guard lane_lock(lane.mutex);
      lane.stop_requested = true;
      if (lane.task_pending || lane.task_running) {
        lane.task_pending = false;
        lane.task_running = false;
        lane.task_error = std::move(error);
        lane.task_complete = true;
      }
      lane.cv.notify_all();
    }
    {
      const std::lock_guard state_lock(state_mutex);
      if (state != State::kJoined && state != State::kFailed) {
        state = State::kStopping;
      }
      barrier_open.store(false, std::memory_order_release);
    }
    stop_lanes();
  }

  void complete_owner_task(Lane& lane, std::optional<Error> error) noexcept {
    {
      const std::lock_guard owner_lock(lane.owner_task_mutex);
      lane.owner_task.context = nullptr;
      lane.owner_task.invoke = nullptr;
      lane.owner_task_error = std::move(error);
      lane.owner_task_state.store(OwnerTaskState::kComplete, std::memory_order_release);
    }
    lane.owner_task_cv.notify_all();
  }

  [[nodiscard]] bool cancel_pending_owner_task(Lane& lane) noexcept {
    OwnerTaskState pending = OwnerTaskState::kPending;
    if (!lane.owner_task_state.compare_exchange_strong(pending,
                                                       OwnerTaskState::kRunning,
                                                       std::memory_order_acq_rel,
                                                       std::memory_order_acquire)) {
      return false;
    }
    complete_owner_task(lane, Error{ErrorCode::kInterrupted, "RUNTIME_LANE_OWNER_TASK_CANCELLED"});
    return true;
  }

  [[nodiscard]] bool run_owner_task(Lane& lane) noexcept {
    OwnerTaskState pending = OwnerTaskState::kPending;
    if (!lane.owner_task_state.compare_exchange_strong(pending,
                                                       OwnerTaskState::kRunning,
                                                       std::memory_order_acq_rel,
                                                       std::memory_order_acquire)) {
      return false;
    }

    const ExecutionLaneOwnerTask task = lane.owner_task;
    std::optional<Error> error;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      auto result = task.invoke(task.context);
      if (!result) {
        error = std::move(result).error();
      }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (const std::exception& exception) {
      error = Error{ErrorCode::kFailure, exception.what()};
    } catch (...) {
      error = Error{ErrorCode::kFailure, "RUNTIME_LANE_OWNER_TASK_EXCEPTION"};
    }
#endif
    complete_owner_task(lane, std::move(error));
    return true;
  }

  void busy_lane_loop(Lane& lane) {
    // The task-state release/acquire handoff owns the borrowed task and result while this
    // accepted busy-poll lane spins. Sleeping lanes retain the mutex/condition-variable path.
    for (;;) {
      if (lane.owner_task_state.load(std::memory_order_acquire) == OwnerTaskState::kPending) {
        if (lane.busy_stop_requested.load(std::memory_order_acquire)) {
          static_cast<void>(cancel_pending_owner_task(lane));
          return;
        }
        static_cast<void>(run_owner_task(lane));
        continue;
      }

      const BusyTaskState task_state = lane.busy_task_state.load(std::memory_order_acquire);
      if (task_state != BusyTaskState::kPending) {
        if (lane.busy_stop_requested.load(std::memory_order_acquire) &&
            task_state == BusyTaskState::kIdle) {
          BusyTaskState idle = BusyTaskState::kIdle;
          if (lane.busy_task_state.compare_exchange_strong(idle,
                                                           BusyTaskState::kStopped,
                                                           std::memory_order_acq_rel,
                                                           std::memory_order_acquire)) {
            return;
          }
        }
        busy_wait_pause();
        continue;
      }

      if (lane.busy_stop_requested.load(std::memory_order_acquire)) {
        lane.task.context = nullptr;
        lane.task.invoke = nullptr;
        lane.task_error = Error{ErrorCode::kInterrupted, "RUNTIME_LANE_TASK_CANCELLED"};
        lane.busy_task_state.store(BusyTaskState::kComplete, std::memory_order_release);
        continue;
      }

      lane.busy_task_state.store(BusyTaskState::kRunning, std::memory_order_relaxed);
      const ExecutionLaneTask task = lane.task;
      RuntimeCallbackIdentity identity = task.callback_identity;
      if (identity.context == RuntimeCallbackContext::kNone) {
        identity.context = task.callback_context;
      }
      const RuntimeCallbackContextScope callback_scope{identity};
      expected<void, Error> result = invoke_lane_task(task, phase_counters);
      lane.task.context = nullptr;
      lane.task.invoke = nullptr;
      lane.task_error =
          result ? std::optional<Error>{} : std::optional<Error>{std::move(result).error()};
      lane.busy_task_state.store(BusyTaskState::kComplete, std::memory_order_release);
    }
  }

  // Keep worker placement, startup evidence, barrier admission, and task-loop cleanup in one lane
  // lifetime so no extracted helper can outlive its owning Lane state.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  void lane_main_impl(std::size_t index) {
    Lane& lane = *lanes[index];
    if (!prepare_lane_startup(index)) {
      return;
    }

    std::unique_lock lane_lock(lane.mutex);
    lane.cv.wait(lane_lock, [&] {
      return barrier_open.load(std::memory_order_acquire) || lane.stop_requested;
    });
    if (lane.stop_requested) {
      return;
    }

    if (lane.plan.accepted_placement.wake_mode == WakeMode::kBusyPoll) {
      lane_lock.unlock();
      busy_lane_loop(lane);
      return;
    }

    for (;;) {
      lane.cv.wait(lane_lock, [&] {
        return lane.task_pending || lane.stop_requested ||
               lane.owner_task_state.load(std::memory_order_acquire) == OwnerTaskState::kPending;
      });

      if (lane.owner_task_state.load(std::memory_order_acquire) == OwnerTaskState::kPending) {
        if (lane.stop_requested) {
          static_cast<void>(cancel_pending_owner_task(lane));
          return;
        }
        lane_lock.unlock();
        static_cast<void>(run_owner_task(lane));
        lane_lock.lock();
        continue;
      }

      if (lane.stop_requested && !lane.task_pending) {
        return;
      }
      if (lane.stop_requested) {
        lane.task_pending = false;
        lane.task_error = Error{ErrorCode::kInterrupted, "RUNTIME_LANE_TASK_CANCELLED"};
        lane.task_complete = true;
        lane.cv.notify_all();
        return;
      }

      const ExecutionLaneTask task = lane.task;
      lane.task_pending = false;
      lane.task_running = true;
      lane_lock.unlock();
      RuntimeCallbackIdentity identity = task.callback_identity;
      if (identity.context == RuntimeCallbackContext::kNone) {
        identity.context = task.callback_context;
      }
      const RuntimeCallbackContextScope callback_scope{identity};
      expected<void, Error> result = invoke_lane_task(task, phase_counters);
      lane_lock.lock();
      lane.task_running = false;
      lane.task.context = nullptr;
      lane.task.invoke = nullptr;
      lane.task_error =
          result ? std::optional<Error>{} : std::optional<Error>{std::move(result).error()};
      lane.task_complete = true;
      lane.cv.notify_all();
      if (lane.stop_requested) {
        return;
      }
    }
  }

  void lane_main(std::size_t index) noexcept {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      lane_main_impl(index);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (const std::exception& exception) {
      handle_lane_exception(index, Error{ErrorCode::kFailure, exception.what()});
    } catch (...) {
      handle_lane_exception(index, Error{ErrorCode::kFailure, "RUNTIME_LANE_WORKER_EXCEPTION"});
    }
#endif
  }

  void lane_main_with_thread_usage(std::size_t index) noexcept {
    const RuntimeThreadUsageScope thread_usage{phase_counters,
                                               RuntimeThreadUsageRole::kExecutionLane};
    lane_main(index);
  }

  void stop_lanes() noexcept {
    {
      const std::lock_guard lock(startup_mutex);
      startup_cancelled = true;
    }
    startup_cv.notify_all();
    for (const auto& lane_owner : lanes) {
      Lane& lane = *lane_owner;
      static_cast<void>(cancel_pending_owner_task(lane));
      if (lane.inline_owner) {
        {
          const std::lock_guard inline_lock(lane.inline_mutex);
          lane.inline_stop_requested = true;
        }
        lane.inline_cv.notify_all();
        continue;
      }
      if (lane.plan.accepted_placement.wake_mode == WakeMode::kBusyPoll) {
        lane.busy_stop_requested.store(true, std::memory_order_release);
        {
          const std::lock_guard lane_lock(lane.mutex);
          lane.stop_requested = true;
        }
        lane.cv.notify_all();
        continue;
      }
      const std::lock_guard lane_lock(lane.mutex);
      lane.stop_requested = true;
      if (lane.task_pending && !lane.task_running) {
        lane.task_pending = false;
        lane.task_error = Error{ErrorCode::kInterrupted, "RUNTIME_LANE_TASK_CANCELLED"};
        lane.task_complete = true;
      }
      lane.cv.notify_all();
    }
  }

  // Lane shutdown keeps ownership, placement restoration, and join outcomes in one transaction.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<void, Error> join_lanes() noexcept {
    for (const auto& lane_owner : lanes) {
      Lane& lane = *lane_owner;
      if (lane.inline_owner) {
        std::unique_lock inline_lock(lane.inline_mutex);
        if (lane.inline_dispatch_active.load(std::memory_order_acquire) &&
            lane.inline_owner_thread == std::this_thread::get_id()) {
          return lane_error(ErrorCode::kInvalidArgument, "RUNTIME_LANE_SELF_JOIN_FORBIDDEN");
        }
        lane.inline_cv.wait(inline_lock, [&] {
          return !lane.inline_dispatch_active.load(std::memory_order_acquire);
        });
        const bool owner_thread = lane.inline_owner_thread == std::this_thread::get_id();
        const bool restore_pending = lane.inline_original_placement.has_value();
        const std::optional<Error> restore_error = lane.inline_restore_error;
        inline_lock.unlock();
        if (restore_pending) {
          if (!owner_thread) {
            return lane_error(ErrorCode::kInvalidArgument,
                              "RUNTIME_INLINE_LANE_OWNER_THREAD_REQUIRED");
          }
          auto restored = restore_inline_placement(lane);
          if (!restored) {
            return restored;
          }
        } else if (restore_error) {
          return make_unexpected(*restore_error);
        }
        continue;
      }
      std::thread& worker = lane_owner->worker;
      if (!worker.joinable()) {
        continue;
      }
      if (worker.get_id() == std::this_thread::get_id()) {
        return lane_error(ErrorCode::kInvalidArgument, "RUNTIME_LANE_SELF_JOIN_FORBIDDEN");
      }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        worker.join();
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const std::system_error& error) {
        return lane_error(ErrorCode::kFailure, error.what());
      } catch (...) {
        return lane_error(ErrorCode::kFailure, "RUNTIME_LANE_JOIN_FAILED");
      }
#endif
    }
    return {};
  }

  SchedulerPlanRecord scheduler;
  std::shared_ptr<ExecutionLanePlacementAdapter> placement_adapter;
  RuntimePhaseCounters* phase_counters{};
  std::vector<std::unique_ptr<Lane>> lanes;
  std::vector<ExecutionLaneStartupEvidence> evidence;
  std::vector<std::size_t> lane_index_by_execution_lane;

  mutable std::mutex state_mutex;
  State state{State::kCreated};
  std::mutex dispatch_serial_mutex;
  std::mutex join_mutex;
  std::mutex owner_round_wait_mutex;
  bool owner_round_active{};

  std::mutex startup_mutex;
  std::condition_variable startup_cv;
  std::size_t startup_completed{};
  std::optional<std::size_t> first_startup_failure;
  bool startup_cancelled{};
  std::atomic<bool> barrier_open{false};
};

std::shared_ptr<ExecutionLanePlacementAdapter> make_system_execution_lane_placement_adapter() {
  return std::make_shared<SystemExecutionLanePlacementAdapter>();
}

ExecutionLaneSet::ExecutionLaneSet(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

ExecutionLaneSet::~ExecutionLaneSet() {
  if (impl_) {
    request_stop();
    if (!join()) {
      // Reclaiming lane state after a failed or self-referential join would leave an active
      // callback or worker with dangling storage. Destruction cannot report that error safely.
      std::terminate();
    }
  }
}

expected<std::unique_ptr<ExecutionLaneSet>, Error> ExecutionLaneSet::create(
    const SchedulerPlanRecord& scheduler, std::span<const ExecutionLanePlanRecord> lanes,
    std::shared_ptr<ExecutionLanePlacementAdapter> placement_adapter,
    RuntimePhaseCounters* phase_counters, std::optional<std::uint32_t> inline_execution_lane) {
  const bool serial_geometry =
      scheduler.max_concurrent_activations == 1U && !scheduler.activation_overlap;
  const bool parallel_owner_geometry =
      scheduler.worker_count > 1U &&
      scheduler.max_concurrent_activations == scheduler.worker_count &&
      scheduler.activation_overlap && !inline_execution_lane;
  if (lanes.empty() || scheduler.execution_lane_count == 0U ||
      lanes.size() != scheduler.worker_count || lanes.size() > scheduler.execution_lane_count ||
      (!serial_geometry && !parallel_owner_geometry)) {
    return lane_error(ErrorCode::kInvalidArgument, "RUNTIME_EXECUTION_LANE_PLAN_INVALID");
  }
  if (lanes.size() > std::numeric_limits<std::uint32_t>::max()) {
    return lane_error(ErrorCode::kResourceExhausted, "RUNTIME_EXECUTION_LANE_COUNT_EXHAUSTED");
  }
  for (std::size_t index = 0; index < lanes.size(); ++index) {
    const auto& lane = lanes[index];
    if (lane.execution_lane >= scheduler.execution_lane_count ||
        (index != 0U && lanes[index - 1U].execution_lane >= lane.execution_lane) ||
        !lane.partition) {
      return lane_error(ErrorCode::kInvalidArgument, "RUNTIME_EXECUTION_LANE_PLAN_INVALID");
    }
    for (std::size_t previous = 0U; previous < index; ++previous) {
      if (lanes[previous].partition == lane.partition) {
        return lane_error(ErrorCode::kInvalidArgument, "PARTITION_PLACEMENT_UNKNOWN_OR_DUPLICATE");
      }
    }
  }
  if (inline_execution_lane &&
      (lanes.size() != 1U || *inline_execution_lane >= scheduler.execution_lane_count ||
       std::none_of(lanes.begin(), lanes.end(), [&](const ExecutionLanePlanRecord& lane) {
         return lane.execution_lane == *inline_execution_lane;
       }))) {
    return lane_error(ErrorCode::kInvalidArgument, "RUNTIME_INLINE_EXECUTION_LANE_INVALID");
  }

  if (!placement_adapter) {
    placement_adapter = make_system_execution_lane_placement_adapter();
  }
  if (!placement_adapter) {
    return lane_error(ErrorCode::kResourceExhausted, "RUNTIME_LANE_PLACEMENT_ADAPTER_UNAVAILABLE");
  }

  auto impl = std::make_unique<Impl>(
      scheduler, lanes, std::move(placement_adapter), phase_counters, inline_execution_lane);
  return std::unique_ptr<ExecutionLaneSet>{new ExecutionLaneSet(std::move(impl))};
}

// Startup must preserve one ordered rollback path across every lane and placement mode.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, ExecutionLaneStartupEvidence> ExecutionLaneSet::start() {
  {
    const std::lock_guard state_lock(impl_->state_mutex);
    if (impl_->state != Impl::State::kCreated) {
      return make_unexpected(impl_->make_failure(
          0U,
          ExecutionLaneStartupDisposition::kFailed,
          ExecutionLaneStartupPhase::kAllLanesBarrier,
          Error{ErrorCode::kInvalidArgument, "RUNTIME_LANE_SET_ALREADY_STARTED"}));
    }
    impl_->state = Impl::State::kStarting;
  }

  for (std::size_t index = 0; index < impl_->lanes.size(); ++index) {
    Impl::Lane& lane = *impl_->lanes[index];
    if (lane.inline_owner) {
      {
        const std::lock_guard inline_lock(lane.inline_mutex);
        lane.inline_owner_thread = std::this_thread::get_id();
      }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        if (!impl_->prepare_lane_startup(index)) {
          break;
        }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (const std::exception& exception) {
        Error error{ErrorCode::kFailure, exception.what()};
        auto restored = impl_->restore_inline_placement(lane);
        if (!restored) {
          error = std::move(restored).error();
        }
        static_cast<void>(
            impl_->publish_startup(index,
                                   impl_->make_failure(index,
                                                       ExecutionLaneStartupDisposition::kFailed,
                                                       lane.startup_phase,
                                                       std::move(error))));
        break;
      } catch (...) {
        Error error{ErrorCode::kFailure, "RUNTIME_INLINE_LANE_START_FAILED"};
        auto restored = impl_->restore_inline_placement(lane);
        if (!restored) {
          error = std::move(restored).error();
        }
        static_cast<void>(
            impl_->publish_startup(index,
                                   impl_->make_failure(index,
                                                       ExecutionLaneStartupDisposition::kFailed,
                                                       lane.startup_phase,
                                                       std::move(error))));
        break;
      }
#endif
      continue;
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      // Keep ordinary lanes on the original entry point. Thread-usage accounting is selected once
      // during cold startup and adds no scope or sampling work to the uninstrumented loop.
      if (impl_->phase_counters != nullptr) {
        impl_->lanes[index]->worker =
            std::thread([impl = impl_.get(), index] { impl->lane_main_with_thread_usage(index); });
      } else {
        impl_->lanes[index]->worker =
            std::thread([impl = impl_.get(), index] { impl->lane_main(index); });
      }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (const std::system_error& error) {
      const std::lock_guard startup_lock(impl_->startup_mutex);
      impl_->evidence[index] = impl_->make_failure(index,
                                                   ExecutionLaneStartupDisposition::kUnavailable,
                                                   ExecutionLaneStartupPhase::kWorkerStart,
                                                   Error{ErrorCode::kNotReady, error.what()});
      impl_->first_startup_failure = index;
      impl_->startup_cancelled = true;
      break;
    } catch (...) {
      const std::lock_guard startup_lock(impl_->startup_mutex);
      impl_->evidence[index] =
          impl_->make_failure(index,
                              ExecutionLaneStartupDisposition::kFailed,
                              ExecutionLaneStartupPhase::kWorkerStart,
                              Error{ErrorCode::kFailure, "RUNTIME_LANE_WORKER_START_FAILED"});
      impl_->first_startup_failure = index;
      impl_->startup_cancelled = true;
      break;
    }
#endif
  }

  std::optional<std::size_t> failed;
  bool cancelled = false;
  {
    std::unique_lock startup_lock(impl_->startup_mutex);
    impl_->startup_cv.wait(startup_lock, [&] {
      return impl_->first_startup_failure || impl_->startup_cancelled ||
             impl_->startup_completed == impl_->lanes.size();
    });
    failed = impl_->first_startup_failure;
    cancelled = impl_->startup_cancelled;
  }

  if (failed || cancelled) {
    impl_->stop_lanes();
    static_cast<void>(impl_->join_lanes());
    {
      // Every accepted lane receives a terminal startup record even when a
      // peer fails first. Workers have joined, so no lane can race this final
      // conversion from pending to peer-cancelled evidence.
      const std::lock_guard startup_lock(impl_->startup_mutex);
      for (std::size_t index = 0U; index < impl_->evidence.size(); ++index) {
        if (impl_->evidence[index].disposition == ExecutionLaneStartupDisposition::kPending) {
          impl_->evidence[index] =
              impl_->make_failure(index,
                                  ExecutionLaneStartupDisposition::kStopped,
                                  ExecutionLaneStartupPhase::kAllLanesBarrier,
                                  Error{ErrorCode::kInterrupted, "RUNTIME_LANE_STARTUP_CANCELLED"});
          ++impl_->startup_completed;
        }
      }
    }
    const std::lock_guard state_lock(impl_->state_mutex);
    impl_->state = Impl::State::kFailed;
    if (!failed) {
      return make_unexpected(
          impl_->make_failure(0U,
                              ExecutionLaneStartupDisposition::kStopped,
                              ExecutionLaneStartupPhase::kAllLanesBarrier,
                              Error{ErrorCode::kInterrupted, "RUNTIME_LANE_STARTUP_CANCELLED"}));
    }
    return make_unexpected(impl_->evidence[failed.value_or(0U)]);
  }

  bool barrier_opened = false;
  {
    const std::lock_guard state_lock(impl_->state_mutex);
    if (impl_->state == Impl::State::kStarting) {
      impl_->barrier_open.store(true, std::memory_order_release);
      impl_->state = Impl::State::kReady;
      barrier_opened = true;
    }
  }
  if (!barrier_opened) {
    impl_->stop_lanes();
    static_cast<void>(impl_->join_lanes());
    const std::lock_guard state_lock(impl_->state_mutex);
    impl_->state = Impl::State::kFailed;
    return make_unexpected(
        impl_->make_failure(0U,
                            ExecutionLaneStartupDisposition::kStopped,
                            ExecutionLaneStartupPhase::kAllLanesBarrier,
                            Error{ErrorCode::kInterrupted, "RUNTIME_LANE_STARTUP_CANCELLED"}));
  }
  for (const auto& lane : impl_->lanes) {
    // Synchronize the shared barrier predicate with the mutex used by each worker's wait. A
    // worker may publish ready evidence before it enters that wait; taking its mutex here closes
    // the check-to-park interval so the notification below cannot be lost. This is required even
    // though the predicate is atomic: busy-poll dispatches do not otherwise signal lane.cv.
    const std::lock_guard lane_lock(lane->mutex);
    lane->cv.notify_all();
  }
  return {};
}

// Dispatch is one state transition spanning inline, busy-poll, and sleeping lane modes.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> ExecutionLaneSet::dispatch(std::uint32_t execution_lane,
                                                 ExecutionLaneTask task) {
  if (task.invoke == nullptr || execution_lane >= impl_->lane_index_by_execution_lane.size()) {
    return lane_error(ErrorCode::kInvalidArgument, "RUNTIME_LANE_DISPATCH_INVALID");
  }
  const std::size_t lane_index = impl_->lane_index_by_execution_lane[execution_lane];
  if (lane_index == std::numeric_limits<std::size_t>::max()) {
    return lane_error(ErrorCode::kInvalidArgument, "RUNTIME_LANE_DISPATCH_INVALID");
  }
  Impl::Lane& lane = *impl_->lanes[lane_index];
  if (lane.inline_owner) {
    const std::lock_guard inline_lock(lane.inline_mutex);
    if (lane.inline_owner_thread != std::this_thread::get_id()) {
      return lane_error(ErrorCode::kInvalidArgument, "RUNTIME_INLINE_LANE_OWNER_THREAD_REQUIRED");
    }
    if (lane.inline_dispatch_active.load(std::memory_order_acquire)) {
      return lane_error(ErrorCode::kInvalidArgument, "RUNTIME_LANE_RECURSIVE_DISPATCH_FORBIDDEN");
    }
  }
  for (const auto& lane : impl_->lanes) {
    if (lane->worker.joinable() && lane->worker.get_id() == std::this_thread::get_id()) {
      return lane_error(ErrorCode::kInvalidArgument, "RUNTIME_LANE_RECURSIVE_DISPATCH_FORBIDDEN");
    }
  }

  // This mutex is the process-local RunSession dispatch token. It remains held
  // while the selected lane executes the complete callback/commit unit.
  std::unique_lock serial_lock(impl_->dispatch_serial_mutex, std::defer_lock);
  if (impl_->phase_counters == nullptr) {
    serial_lock.lock();
  } else if (!serial_lock.try_lock()) {
    count_phase(impl_->phase_counters, RuntimePhaseCounter::kLaneSerializationContentions);
    serial_lock.lock();
  }
  {
    const std::lock_guard state_lock(impl_->state_mutex);
    if (impl_->state != Impl::State::kReady ||
        !impl_->barrier_open.load(std::memory_order_acquire)) {
      return lane_error(ErrorCode::kNotReady, "RUNTIME_LANES_NOT_READY");
    }
    if (impl_->owner_round_active) {
      return lane_error(ErrorCode::kClosed, "RUNTIME_LANE_OWNER_TASKS_ACTIVE");
    }
  }

  if (lane.inline_owner) {
    {
      // Stop admission and callback admission share this lock. Once stop
      // publishes its terminal flag, no inline callback can enter user code.
      const std::lock_guard inline_lock(lane.inline_mutex);
      if (lane.inline_stop_requested) {
        return lane_error(ErrorCode::kClosed, "RUNTIME_LANE_STOPPED");
      }
      if (lane.inline_dispatch_active.exchange(true, std::memory_order_acq_rel)) {
        return lane_error(ErrorCode::kInvalidArgument, "RUNTIME_LANE_RECURSIVE_DISPATCH_FORBIDDEN");
      }
    }
    struct InlineDispatchGuard {
      Impl& impl;
      Impl::Lane& lane;
      ~InlineDispatchGuard() {
        bool restore_placement = false;
        {
          const std::lock_guard inline_lock(lane.inline_mutex);
          restore_placement = lane.inline_stop_requested;
          if (!restore_placement) {
            lane.inline_dispatch_active.store(false, std::memory_order_release);
          }
        }
        if (restore_placement) {
          static_cast<void>(impl.restore_inline_placement(lane));
          const std::lock_guard inline_lock(lane.inline_mutex);
          lane.inline_dispatch_active.store(false, std::memory_order_release);
        }
        lane.inline_cv.notify_all();
      }
    } const inline_dispatch_guard{*impl_, lane};

    RuntimeCallbackIdentity identity = task.callback_identity;
    if (identity.context == RuntimeCallbackContext::kNone) {
      identity.context = task.callback_context;
    }
    const RuntimeCallbackContextScope callback_scope{identity};
    count_phase(impl_->phase_counters, RuntimePhaseCounter::kLaneInlineOwnerInvocations);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      return invoke_lane_task(task, impl_->phase_counters);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (const std::exception& exception) {
      request_stop();
      return lane_error(ErrorCode::kFailure, exception.what());
    } catch (...) {
      request_stop();
      return lane_error(ErrorCode::kFailure, "RUNTIME_INLINE_LANE_CALLBACK_FAILED");
    }
#endif
  }
  if (lane.plan.accepted_placement.wake_mode == WakeMode::kBusyPoll) {
    if (lane.busy_stop_requested.load(std::memory_order_acquire)) {
      return lane_error(ErrorCode::kClosed, "RUNTIME_LANE_STOPPED");
    }
    Impl::BusyTaskState idle = Impl::BusyTaskState::kIdle;
    if (!lane.busy_task_state.compare_exchange_strong(idle,
                                                      Impl::BusyTaskState::kPreparing,
                                                      std::memory_order_acq_rel,
                                                      std::memory_order_acquire)) {
      return lane_error(ErrorCode::kClosed, "RUNTIME_LANE_STOPPED");
    }
    lane.task = task;
    lane.task_error.reset();
    // kPreparing prevents a concurrent stop from retiring the worker while this borrowed task is
    // filled. Publishing kPending transfers task access to the lane; observing kComplete transfers
    // the result back. The process-wide serial token prevents a second publisher from racing it.
    impl_->count_dispatch_context(task);
    count_phase(impl_->phase_counters, RuntimePhaseCounter::kLaneBusyTaskPublications);
    lane.busy_task_state.store(Impl::BusyTaskState::kPending, std::memory_order_release);
    while (lane.busy_task_state.load(std::memory_order_acquire) != Impl::BusyTaskState::kComplete) {
      busy_wait_pause();
    }
    std::optional<Error> task_error = std::move(lane.task_error);
    count_phase(impl_->phase_counters, RuntimePhaseCounter::kLaneCompletionsObserved);
    lane.task_error.reset();
    lane.busy_task_state.store(Impl::BusyTaskState::kIdle, std::memory_order_release);
    if (task_error) {
      return make_unexpected(std::move(*task_error));
    }
    return {};
  }
  std::unique_lock lane_lock(lane.mutex);
  if (lane.stop_requested || lane.task_pending || lane.task_running) {
    return lane_error(ErrorCode::kClosed, "RUNTIME_LANE_STOPPED");
  }
  lane.task = task;
  lane.task_error.reset();
  lane.task_complete = false;
  lane.task_pending = true;
  impl_->count_dispatch_context(task);
  count_phase(impl_->phase_counters, RuntimePhaseCounter::kLaneSleepingWakeSignals);
  lane.cv.notify_one();
  lane.cv.wait(lane_lock, [&] { return lane.task_complete; });
  count_phase(impl_->phase_counters, RuntimePhaseCounter::kLaneCompletionsObserved);
  if (lane.task_error) {
    Error error = std::move(*lane.task_error);
    lane.task_error.reset();
    return make_unexpected(std::move(error));
  }
  return {};
}

expected<void, Error> ExecutionLaneSet::launch_owner_tasks(
    std::span<const ExecutionLaneOwnerTask> tasks) {
  for (const auto& lane : impl_->lanes) {
    if (lane->inline_owner) {
      return lane_error(ErrorCode::kInvalidArgument, "RUNTIME_LANE_OWNER_TASK_INLINE_FORBIDDEN");
    }
    if (lane->worker.joinable() && lane->worker.get_id() == std::this_thread::get_id()) {
      return lane_error(ErrorCode::kInvalidArgument, "RUNTIME_LANE_OWNER_TASK_SELF_LAUNCH");
    }
  }
  if (tasks.size() != impl_->lanes.size()) {
    return lane_error(ErrorCode::kInvalidArgument, "RUNTIME_LANE_OWNER_TASK_COVER_INVALID");
  }
  if (!impl_->scheduler.activation_overlap ||
      impl_->scheduler.max_concurrent_activations != impl_->scheduler.worker_count) {
    return lane_error(ErrorCode::kInvalidArgument, "RUNTIME_LANE_OWNER_TASK_GEOMETRY_INVALID");
  }
  for (std::size_t index = 0U; index < tasks.size(); ++index) {
    if (tasks[index].invoke == nullptr ||
        tasks[index].execution_lane != impl_->lanes[index]->plan.execution_lane) {
      return lane_error(ErrorCode::kInvalidArgument, "RUNTIME_LANE_OWNER_TASK_COVER_INVALID");
    }
  }

  // Drain any accepted legacy callback before atomically switching the complete
  // lane set into owner mode. The token is released before workers observe the
  // tasks and is never acquired by an owner task.
  const std::lock_guard dispatch_lock(impl_->dispatch_serial_mutex);
  {
    const std::lock_guard state_lock(impl_->state_mutex);
    if (impl_->state != Impl::State::kReady ||
        !impl_->barrier_open.load(std::memory_order_acquire)) {
      return lane_error(ErrorCode::kNotReady, "RUNTIME_LANES_NOT_READY");
    }
    if (impl_->owner_round_active) {
      return lane_error(ErrorCode::kInvalidArgument, "RUNTIME_LANE_OWNER_TASKS_ACTIVE");
    }
    for (const auto& lane : impl_->lanes) {
      if (lane->owner_task_state.load(std::memory_order_acquire) !=
          Impl::OwnerTaskState::kUnassigned) {
        return lane_error(ErrorCode::kInvalidArgument, "RUNTIME_LANE_OWNER_TASKS_ACTIVE");
      }
    }
    for (std::size_t index = 0U; index < tasks.size(); ++index) {
      Impl::Lane& lane = *impl_->lanes[index];
      const std::lock_guard owner_lock(lane.owner_task_mutex);
      lane.owner_task = tasks[index];
      lane.owner_task_error.reset();
      lane.owner_task_state.store(Impl::OwnerTaskState::kPending, std::memory_order_release);
    }
    impl_->owner_round_active = true;
  }
  for (const auto& lane : impl_->lanes) {
    lane->cv.notify_one();
  }
  return {};
}

expected<void, Error> ExecutionLaneSet::wait_owner_tasks() noexcept {
  for (const auto& lane : impl_->lanes) {
    if (lane->worker.joinable() && lane->worker.get_id() == std::this_thread::get_id()) {
      return lane_error(ErrorCode::kInvalidArgument, "RUNTIME_LANE_OWNER_TASK_SELF_WAIT");
    }
  }
  const std::lock_guard wait_lock(impl_->owner_round_wait_mutex);
  {
    const std::lock_guard state_lock(impl_->state_mutex);
    if (!impl_->owner_round_active) {
      return lane_error(ErrorCode::kInvalidArgument, "RUNTIME_LANE_OWNER_TASKS_NOT_LAUNCHED");
    }
  }

  std::optional<Error> first_error;
  for (const auto& lane : impl_->lanes) {
    std::unique_lock owner_lock(lane->owner_task_mutex);
    lane->owner_task_cv.wait(owner_lock, [&] {
      return lane->owner_task_state.load(std::memory_order_acquire) ==
             Impl::OwnerTaskState::kComplete;
    });
    if (!first_error && lane->owner_task_error) {
      first_error = lane->owner_task_error;
    }
  }

  // Reopen legacy dispatch only after every owner task has retired and the
  // entire round has been reset. Serializing this transition with legacy
  // dispatch prevents a callback from observing a partially reset cover.
  const std::lock_guard dispatch_lock(impl_->dispatch_serial_mutex);
  {
    const std::lock_guard state_lock(impl_->state_mutex);
    for (const auto& lane : impl_->lanes) {
      const std::lock_guard owner_lock(lane->owner_task_mutex);
      lane->owner_task = {};
      lane->owner_task_error.reset();
      lane->owner_task_state.store(Impl::OwnerTaskState::kUnassigned, std::memory_order_release);
    }
    impl_->owner_round_active = false;
  }
  if (first_error) {
    return make_unexpected(std::move(first_error).value_or(Error{ErrorCode::kFailure}));
  }
  return {};
}

void ExecutionLaneSet::request_stop() noexcept {
  {
    const std::lock_guard state_lock(impl_->state_mutex);
    if (impl_->state == Impl::State::kJoined || impl_->state == Impl::State::kFailed) {
      return;
    }
    impl_->state = Impl::State::kStopping;
    impl_->barrier_open.store(false, std::memory_order_release);
  }
  impl_->stop_lanes();
}

expected<void, Error> ExecutionLaneSet::join() noexcept {
  const std::lock_guard join_lock(impl_->join_mutex);
  {
    const std::lock_guard state_lock(impl_->state_mutex);
    if (impl_->state == Impl::State::kJoined) {
      return {};
    }
  }
  request_stop();
  auto result = impl_->join_lanes();
  if (!result) {
    return result;
  }
  // Inline callbacks do not own a joinable worker. Draining the same serial
  // token used by dispatch closes the interval between the callback returning
  // and its caller releasing the complete borrowed task.
  const std::lock_guard dispatch_lock(impl_->dispatch_serial_mutex);
  const std::lock_guard state_lock(impl_->state_mutex);
  impl_->state = Impl::State::kJoined;
  return {};
}

bool ExecutionLaneSet::ready() const noexcept {
  const std::lock_guard state_lock(impl_->state_mutex);
  return impl_->state == Impl::State::kReady && impl_->barrier_open.load(std::memory_order_acquire);
}

ExecutionLaneSet::StartupEvidenceView ExecutionLaneSet::startup_evidence() const noexcept {
  return impl_->evidence;
}

std::vector<ExecutionLaneStartupEvidence> ExecutionLaneSet::take_startup_evidence() noexcept {
  return std::move(impl_->evidence);
}

}  // namespace holoscan::detail
