/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_test_graph.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

#include "channel/process_identity.hpp"
#include "runtime/runtime_test_access.hpp"

namespace holoscan {
[[nodiscard]] expected<std::vector<std::byte>, Error> PredicateConfigCodec<
    RuntimeInputPredicateTestConfig>::encode(const RuntimeInputPredicateTestConfig& value) {
  return std::vector<std::byte>{
      std::byte(value.minimum_driver_value & 0xFFU),
      std::byte((value.minimum_driver_value >> 8U) & 0xFFU),
      std::byte((value.minimum_driver_value >> 16U) & 0xFFU),
      std::byte((value.minimum_driver_value >> 24U) & 0xFFU),
  };
}

}  // namespace holoscan

namespace holoscan::test {
namespace {

// Runtime-materializing tests publish into the host-wide HoloMQ namespace. Give every invocation
// its own graph identity so concurrent CTest processes and SDK containers cannot derive the same
// process-local Channel service names.
[[nodiscard]] std::string unique_runtime_compute_call_graph_name() {
  static std::atomic<std::uint64_t> ordinal{};
  const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::string("runtime-compute-call-completion-channel-") +
         std::to_string(channel_internal::current_process_identity()) + "-" +
         std::to_string(timestamp) + "-" +
         std::to_string(ordinal.fetch_add(1U, std::memory_order_relaxed));
}

}  // namespace

std::mutex RuntimeProbeState::mutex{};

std::condition_variable RuntimeProbeState::cv{};

std::vector<std::int32_t> RuntimeProbeState::events{};

std::vector<std::thread::id> RuntimeProbeState::event_threads{};

std::atomic<RuntimeProbeMode> RuntimeProbeState::mode{RuntimeProbeMode::kComplete};

std::atomic<std::uint32_t> RuntimeProbeState::constructed{};

std::atomic<std::uint32_t> RuntimeProbeState::active{};

std::atomic<std::uint32_t> RuntimeProbeState::max_active{};

std::atomic_bool RuntimeProbeState::setup_entered{};

std::atomic_bool RuntimeProbeState::compute_entered{};

std::atomic<std::uint64_t> RuntimeProbeState::observed_clock_id{};

std::atomic<std::uint64_t> RuntimeProbeState::observed_activation_clock_id{};

std::atomic<std::int64_t> RuntimeProbeState::observed_activation_time_ns{};

std::atomic<std::int64_t> RuntimeProbeState::observed_clock_time_ns{};

std::atomic_bool RuntimeProbeState::observed_clock_read_succeeded{};

std::atomic_bool RuntimeProbeState::observed_clock_can_read_now{};

std::atomic_bool RuntimeProbeState::observed_clock_can_wait_until{};

std::atomic_bool RuntimeProbeState::observed_clock_steady{};

std::atomic<std::int64_t> RuntimeProbeState::runtime_clock_ns{};

std::atomic<std::uint32_t> RuntimeProbeState::clocks_constructed{};

std::atomic<RunSession*> RuntimeProbeState::callback_session{};

std::atomic_bool RuntimeProbeState::callback_session_action_returned{};

std::atomic<std::int64_t> RuntimeProbeState::stop_observed_time_ns{};

std::atomic<std::int64_t> RuntimeProbeState::callback_return_time_ns{};

std::atomic<int> RuntimeProbeState::external_ready_fd{-1};

std::atomic<int> RuntimeProbeState::external_stop_gate_fd{-1};

std::mutex RuntimeClockProbeState::mutex{};

std::condition_variable RuntimeClockProbeState::cv{};

std::atomic<std::size_t> RuntimeClockProbeState::compute_count{};

std::atomic<RuntimeClockProbeState::TerminationRequestMode>
    RuntimeClockProbeState::termination_request_mode{};

std::array<Tag, RuntimeClockProbeState::kRecordedActivations> RuntimeClockProbeState::tags{};

std::array<TagTime, RuntimeClockProbeState::kRecordedActivations>
    RuntimeClockProbeState::tag_times{};

std::array<TimePoint, RuntimeClockProbeState::kRecordedActivations>
    RuntimeClockProbeState::activation_times{};

std::mutex RuntimeNotificationProbeState::mutex{};

std::condition_variable RuntimeNotificationProbeState::cv{};

std::optional<NotificationSender> RuntimeNotificationProbeState::sender{};

ErrorCode RuntimeNotificationProbeState::arm_post_error{ErrorCode::kSuccess};

ErrorCode RuntimeNotificationProbeState::second_acquire_error{ErrorCode::kSuccess};

std::size_t RuntimeNotificationProbeState::compute_count{};

bool RuntimeNotificationProbeState::compute_entered{};

bool RuntimeNotificationProbeState::release_compute{};

std::mutex RuntimeClockPredicateProbeState::mutex{};

std::condition_variable RuntimeClockPredicateProbeState::cv{};

std::atomic<std::size_t> RuntimeClockPredicateProbeState::evaluation_count{};

std::atomic<std::size_t> RuntimeClockPredicateProbeState::compute_count{};

std::atomic_bool RuntimeClockPredicateProbeState::restart_accepted{};

std::atomic<ErrorCode> RuntimeClockPredicateProbeState::restart_error{ErrorCode::kSuccess};

std::atomic<std::uint64_t> RuntimeClockPredicateProbeState::restart_receipt_id{};

std::atomic<LifecycleStage> RuntimeClockPredicateProbeState::restart_requested_stage{
    LifecycleStage::kConfigure};

std::array<RuntimeClockPredicateEvaluationEvidence,
           RuntimeClockPredicateProbeState::kRecordedEvaluations>
    RuntimeClockPredicateProbeState::evaluations{};

std::array<RuntimeClockPredicateComputeEvidence,
           RuntimeClockPredicateProbeState::kRecordedEvaluations>
    RuntimeClockPredicateProbeState::computes{};

std::mutex LifecycleParamProbeState::mutex{};

std::condition_variable LifecycleParamProbeState::cv{};

std::vector<LifecycleParamObservation> LifecycleParamProbeState::lifecycle{};

std::vector<LifecycleParamComputeObservation> LifecycleParamProbeState::compute_observations{};

bool LifecycleParamProbeState::hold_first{};

bool LifecycleParamProbeState::release_first{true};

bool LifecycleParamProbeState::fail_start{};

std::int32_t LifecycleParamProbeState::fail_start_identity{1};

bool LifecycleParamProbeState::failed_start{};

bool LifecycleParamProbeState::hold_restart_start{};

bool LifecycleParamProbeState::release_restart_start{true};

bool LifecycleParamProbeState::restart_start_entered{};

bool LifecycleParamProbeState::restart_start_observed_cancellation{};

std::mutex RuntimeParamProbeState::mutex{};

std::condition_variable RuntimeParamProbeState::cv{};

std::vector<RuntimeParamObservation> RuntimeParamProbeState::observations{};

bool RuntimeParamProbeState::hold_first{};

bool RuntimeParamProbeState::release_first{true};

bool RuntimeParamProbeState::hold_second{};

bool RuntimeParamProbeState::release_second{true};

bool RuntimeParamProbeState::hold_start{};

bool RuntimeParamProbeState::start_entered{};

bool RuntimeParamProbeState::release_start{true};

bool RuntimeParamProbeState::hold_stop{};

bool RuntimeParamProbeState::stop_entered{};

bool RuntimeParamProbeState::release_stop{true};

std::atomic<std::int32_t> RuntimeParamProbeState::stop_observed_value{};

std::atomic_bool RuntimeParamProbeState::stop_observed_params{};

std::atomic<std::uint32_t> RuntimeDeadlineParamProbeState::compute_count{};

std::atomic<std::int64_t> RuntimeDeadlineParamProbeState::observed_deadline_ns{};

std::atomic<ParamGeneration> RuntimeDeadlineParamProbeState::observed_generation{};

std::atomic<ActivationReason> RuntimeDeadlineParamProbeState::observed_reason{
    ActivationReason::kNominal};

std::atomic<OnViolation> RuntimeDeadlineParamProbeState::replay_violation{OnViolation::kFailRun};

std::mutex RuntimeAlignedChannelProbeState::mutex{};

std::condition_variable RuntimeAlignedChannelProbeState::cv{};

std::atomic<RuntimeAlignedChannelScript> RuntimeAlignedChannelProbeState::script{
    RuntimeAlignedChannelScript::kExactBoundary};

std::atomic<std::size_t> RuntimeAlignedChannelProbeState::next_step{};

std::atomic<std::size_t> RuntimeAlignedChannelProbeState::source_compute_count{};

std::atomic<std::size_t> RuntimeAlignedChannelProbeState::sink_compute_count{};

std::atomic<std::uint32_t> RuntimeAlignedChannelProbeState::observed_first{};

std::atomic<std::uint32_t> RuntimeAlignedChannelProbeState::observed_second{};

std::atomic<std::int64_t> RuntimeAlignedChannelProbeState::observed_first_capture{};

std::atomic<std::int64_t> RuntimeAlignedChannelProbeState::observed_second_capture{};

std::mutex RuntimeUnconnectedPortProbeState::mutex{};

std::condition_variable RuntimeUnconnectedPortProbeState::cv{};

std::atomic<std::uint32_t> RuntimeUnconnectedPortProbeState::compute_count{};

std::mutex RuntimeActivationCutProbeState::mutex{};

std::condition_variable RuntimeActivationCutProbeState::cv{};

std::vector<RuntimeActivationCutObservation> RuntimeActivationCutProbeState::observations{};

std::mutex RuntimeWindowChannelProbeState::mutex{};

std::condition_variable RuntimeWindowChannelProbeState::cv{};

std::atomic<RuntimeWindowChannelScript> RuntimeWindowChannelProbeState::script{
    RuntimeWindowChannelScript::kSuccessorClose};

std::atomic<std::size_t> RuntimeWindowChannelProbeState::compute_count{};

std::atomic<std::size_t> RuntimeWindowChannelProbeState::safe_default_end_of_stream_count{};

std::atomic<std::size_t> RuntimeWindowChannelProbeState::downstream_count{};

std::array<Tag, 2U> RuntimeWindowChannelProbeState::terminal_tags{};

std::array<TagTime, 2U> RuntimeWindowChannelProbeState::terminal_tag_times{};

std::atomic<InputState> RuntimeWindowChannelProbeState::selection_state{InputState::kNotSelected};

std::atomic<SelectionReason> RuntimeWindowChannelProbeState::selection_reason{
    SelectionReason::kNotSelectedForActivation};

std::atomic<std::int64_t> RuntimeWindowChannelProbeState::anchor_capture_ns{};

std::atomic<std::int64_t> RuntimeWindowChannelProbeState::activation_time_ns{};

std::atomic<std::size_t> RuntimeWindowChannelProbeState::batch_size{};

std::atomic<std::int64_t> RuntimeWindowChannelProbeState::batch_first_capture_ns{};

std::atomic<std::int64_t> RuntimeWindowChannelProbeState::batch_last_capture_ns{};

std::atomic<RuntimeInputTriggerReplayKind> RuntimeInputTriggerReplayOperator::kind{
    RuntimeInputTriggerReplayKind::kAny};

std::atomic<RuntimeInputContractReplayMode> RuntimeInputContractPredicateOperator::mode{
    RuntimeInputContractReplayMode::kMatch};

std::atomic<std::uint32_t> RuntimeClockInputReplayOperator::mismatch_ordinal{};

std::mutex RuntimeInputPredicateChannelState::mutex{};

std::condition_variable RuntimeInputPredicateChannelState::cv{};

std::atomic<std::size_t> RuntimeInputPredicateChannelState::evaluator_count{};

std::atomic<std::size_t> RuntimeInputPredicateChannelState::compute_count{};

std::atomic<std::size_t> RuntimeInputPredicateChannelState::source_count{};

std::atomic<std::uint32_t> RuntimeInputPredicateChannelState::evaluator_driver_value{};

std::atomic<std::uint32_t> RuntimeInputPredicateChannelState::evaluator_additional_value{};

std::atomic<std::uint64_t> RuntimeInputPredicateChannelState::evaluator_driver_sequence{};

std::atomic<std::uint64_t> RuntimeInputPredicateChannelState::evaluator_additional_sequence{};

std::atomic<std::uint32_t> RuntimeInputPredicateChannelState::compute_driver_value{};

std::atomic<std::uint32_t> RuntimeInputPredicateChannelState::compute_additional_value{};

std::atomic<std::uint64_t> RuntimeInputPredicateChannelState::compute_driver_sequence{};

std::atomic<std::uint64_t> RuntimeInputPredicateChannelState::compute_additional_sequence{};

std::vector<RuntimeInputPredicateEvaluationObservation>
    RuntimeInputPredicateChannelState::evaluation_observations{};

std::vector<RuntimeInputPredicateComputeObservation>
    RuntimeInputPredicateChannelState::compute_observations{};

void RuntimeProbeState::reset(RuntimeProbeMode requested_mode) {
  std::lock_guard lock(mutex);
  events.clear();
  event_threads.clear();
  constructed.store(0, std::memory_order_relaxed);
  active.store(0, std::memory_order_relaxed);
  max_active.store(0, std::memory_order_relaxed);
  setup_entered.store(false, std::memory_order_relaxed);
  compute_entered.store(false, std::memory_order_relaxed);
  observed_clock_id.store(0U, std::memory_order_relaxed);
  observed_activation_clock_id.store(0U, std::memory_order_relaxed);
  observed_activation_time_ns.store(0, std::memory_order_relaxed);
  observed_clock_time_ns.store(0, std::memory_order_relaxed);
  observed_clock_read_succeeded.store(false, std::memory_order_relaxed);
  observed_clock_can_read_now.store(false, std::memory_order_relaxed);
  observed_clock_can_wait_until.store(false, std::memory_order_relaxed);
  observed_clock_steady.store(false, std::memory_order_relaxed);
  runtime_clock_ns.store(0, std::memory_order_relaxed);
  clocks_constructed.store(0U, std::memory_order_relaxed);
  callback_session.store(nullptr, std::memory_order_relaxed);
  callback_session_action_returned.store(false, std::memory_order_relaxed);
  stop_observed_time_ns.store(0, std::memory_order_relaxed);
  callback_return_time_ns.store(0, std::memory_order_relaxed);
  external_ready_fd.store(-1, std::memory_order_relaxed);
  external_stop_gate_fd.store(-1, std::memory_order_relaxed);
  mode.store(requested_mode, std::memory_order_relaxed);
}

void RuntimeProbeState::bind_external_ready_fd(const int fd) noexcept {
  external_ready_fd.store(fd, std::memory_order_release);
}

void RuntimeProbeState::bind_external_stop_gate_fd(const int fd) noexcept {
  external_stop_gate_fd.store(fd, std::memory_order_release);
}

void RuntimeProbeState::bind_callback_session(RunSession* session) noexcept {
  callback_session.store(session, std::memory_order_release);
}

void RuntimeProbeState::invoke_session_action_from_callback(RuntimeProbeMode requested_mode) {
  if (mode.load(std::memory_order_relaxed) != requested_mode) {
    return;
  }
  RunSession* session = nullptr;
  while ((session = callback_session.load(std::memory_order_acquire)) == nullptr) {
    std::this_thread::yield();
  }
  if (requested_mode == RuntimeProbeMode::kWaitFromCompute ||
      requested_mode == RuntimeProbeMode::kWaitFromLifecycle) {
    session->wait();
    return;
  }
  delete session;
  callback_session_action_returned.store(true, std::memory_order_release);
}

[[nodiscard]] bool RuntimeProbeState::callback_action_returned() noexcept {
  return callback_session_action_returned.load(std::memory_order_acquire);
}

void RuntimeProbeState::record(std::int32_t value) {
  std::lock_guard lock(mutex);
  events.push_back(value);
  event_threads.push_back(std::this_thread::get_id());
  cv.notify_all();
}

[[nodiscard]] std::vector<std::int32_t> RuntimeProbeState::snapshot() {
  std::lock_guard lock(mutex);
  return events;
}

[[nodiscard]] std::vector<std::thread::id> RuntimeProbeState::snapshot_threads() {
  std::lock_guard lock(mutex);
  return event_threads;
}

[[nodiscard]] bool RuntimeProbeState::wait_until_compute_enters() {
  std::unique_lock lock(mutex);
  return cv.wait_for(lock, std::chrono::seconds{5}, [] {
    return compute_entered.load(std::memory_order_acquire);
  });
}

[[nodiscard]] bool RuntimeProbeState::wait_until_active(std::uint32_t expected) {
  std::unique_lock lock(mutex);
  return cv.wait_for(lock, std::chrono::seconds{5}, [expected] {
    return active.load(std::memory_order_acquire) == expected;
  });
}

void RuntimeClockProbeState::reset() noexcept {
  compute_count.store(0U, std::memory_order_relaxed);
  termination_request_mode.store(TerminationRequestMode::kNone, std::memory_order_relaxed);
  tags.fill({});
  tag_times.fill({});
  activation_times.fill({});
}

[[nodiscard]] bool RuntimeClockProbeState::wait_until_compute_count(std::size_t expected) {
  std::unique_lock lock(mutex);
  return cv.wait_for(lock, std::chrono::seconds{5}, [expected] {
    return compute_count.load(std::memory_order_acquire) >= expected;
  });
}

void RuntimeNotificationProbeState::reset() {
  const std::lock_guard lock(mutex);
  sender.reset();
  arm_post_error = ErrorCode::kSuccess;
  second_acquire_error = ErrorCode::kSuccess;
  compute_count = 0U;
  compute_entered = false;
  release_compute = false;
}

void RuntimeNotificationProbeState::publish_sender(NotificationSender value, ErrorCode post_error,
                                                   ErrorCode acquire_error) {
  {
    const std::lock_guard lock(mutex);
    sender = std::move(value);
    arm_post_error = post_error;
    second_acquire_error = acquire_error;
  }
  cv.notify_all();
}

[[nodiscard]] bool RuntimeNotificationProbeState::wait_until_sender() {
  std::unique_lock lock(mutex);
  return cv.wait_for(lock, std::chrono::seconds{5}, [] { return sender.has_value(); });
}

[[nodiscard]] NotificationSender RuntimeNotificationProbeState::sender_copy() {
  const std::lock_guard lock(mutex);
  return sender.value_or(NotificationSender{});
}

[[nodiscard]] bool RuntimeNotificationProbeState::wait_until_compute_enters() {
  std::unique_lock lock(mutex);
  return cv.wait_for(lock, std::chrono::seconds{5}, [] { return compute_entered; });
}

[[nodiscard]] bool RuntimeNotificationProbeState::wait_until_compute_count(std::size_t expected) {
  std::unique_lock lock(mutex);
  return cv.wait_for(
      lock, std::chrono::seconds{5}, [expected] { return compute_count >= expected; });
}

void RuntimeNotificationProbeState::release_blocked_compute() {
  {
    const std::lock_guard lock(mutex);
    release_compute = true;
  }
  cv.notify_all();
}

void RuntimeClockPredicateProbeState::reset() noexcept {
  evaluation_count.store(0U, std::memory_order_relaxed);
  compute_count.store(0U, std::memory_order_relaxed);
  restart_accepted.store(false, std::memory_order_relaxed);
  restart_error.store(ErrorCode::kSuccess, std::memory_order_relaxed);
  restart_receipt_id.store(0U, std::memory_order_relaxed);
  restart_requested_stage.store(LifecycleStage::kConfigure, std::memory_order_relaxed);
  evaluations.fill({});
  computes.fill({});
}

void RuntimeClockPredicateProbeState::record_evaluation(const PredicateEvaluationContext& context,
                                                        std::uint32_t state_count,
                                                        bool result) noexcept {
  const std::size_t index = evaluation_count.load(std::memory_order_relaxed);
  if (index < evaluations.size()) {
    evaluations[index] = {
        .reference_time = context.reference_time(),
        .tag = context.tag(),
        .tag_time = context.tag_time(),
        .state_count = state_count,
        .selection_count = context.selections().size(),
        .result = result,
    };
  }
  RuntimeProbeState::runtime_clock_ns.fetch_add(RuntimeClockProbeState::kPeriodNs,
                                                std::memory_order_relaxed);
  evaluation_count.store(index + 1U, std::memory_order_release);
  cv.notify_all();
}

void RuntimeClockPredicateProbeState::record_compute(ExecutionContext& context,
                                                     bool request_restart) noexcept {
  const std::size_t index = compute_count.load(std::memory_order_relaxed);
  if (index < computes.size()) {
    computes[index] = {
        .activation_time = context.activation_time(),
        .tag = context.tag(),
        .tag_time = context.tag_time(),
    };
  }
  if (request_restart) {
    auto receipt = context.request_restart();
    if (receipt) {
      restart_receipt_id.store(receipt->receipt_id, std::memory_order_relaxed);
      restart_requested_stage.store(receipt->requested_stage, std::memory_order_relaxed);
      restart_accepted.store(true, std::memory_order_release);
    } else {
      restart_error.store(receipt.error().code, std::memory_order_relaxed);
    }
  }
  compute_count.store(index + 1U, std::memory_order_release);
  cv.notify_all();
}

[[nodiscard]] bool RuntimeClockPredicateProbeState::wait_until_compute_count(std::size_t expected) {
  std::unique_lock lock(mutex);
  return cv.wait_for(lock, std::chrono::seconds{5}, [expected] {
    return compute_count.load(std::memory_order_acquire) >= expected;
  });
}

RuntimeProbeClock::RuntimeProbeClock() {
  RuntimeProbeState::clocks_constructed.fetch_add(1U, std::memory_order_relaxed);
}

[[nodiscard]] double RuntimeProbeClock::time() const {
  return timestamp_to_time(timestamp());
}

[[nodiscard]] std::int64_t RuntimeProbeClock::timestamp() const {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  if (RuntimeProbeState::mode.load(std::memory_order_relaxed) ==
      RuntimeProbeMode::kThrowClockRead) {
    throw std::runtime_error{"runtime test clock read failure"};
  }
#endif
  return RuntimeProbeState::runtime_clock_ns.load(std::memory_order_relaxed);
}

[[nodiscard]] expected<void, Error> RuntimeProbeClock::sleep_for(std::int64_t duration_ns) {
  RuntimeProbeState::runtime_clock_ns.fetch_add(duration_ns, std::memory_order_relaxed);
  return {};
}

[[nodiscard]] expected<void, Error> RuntimeProbeClock::sleep_until(std::int64_t target_time_ns) {
  RuntimeProbeState::runtime_clock_ns.store(target_time_ns, std::memory_order_relaxed);
  return {};
}

RuntimeProbeOperator::RuntimeProbeOperator(std::int32_t identity, bool) : identity_(identity) {
  RuntimeProbeState::constructed.fetch_add(1, std::memory_order_relaxed);
}

void RuntimeProbeOperator::setup(OperatorSpec& spec) {
  RuntimeProbeState::setup_entered.store(true, std::memory_order_release);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  if (RuntimeProbeState::mode.load(std::memory_order_relaxed) == RuntimeProbeMode::kThrowSetup) {
    throw std::runtime_error{"runtime test setup failure"};
  }
#endif
  spec.spin_source(spin_, "activation").capacity(1);
}

[[nodiscard]] Contract RuntimeProbeOperator::contract() const {
  Contract result;
  if (RuntimeProbeState::mode.load(std::memory_order_relaxed) ==
      RuntimeProbeMode::kMismatchedContract) {
    return result;
  }
  result.trigger(OnSpin{.event = spin_});
  return result;
}

void RuntimeProbeOperator::start() {
  const RuntimeProbeMode callback_mode = RuntimeProbeState::mode.load(std::memory_order_relaxed);
  if (callback_mode == RuntimeProbeMode::kWaitFromLifecycle ||
      callback_mode == RuntimeProbeMode::kDestroyFromLifecycle) {
    RuntimeProbeState::invoke_session_action_from_callback(callback_mode);
  }
  if (RuntimeProbeState::mode.load(std::memory_order_relaxed) ==
      RuntimeProbeMode::kAdvanceClockInStart) {
    RuntimeProbeState::runtime_clock_ns.store(37, std::memory_order_relaxed);
  }
  RuntimeProbeState::record(identity_ * 10 + 1);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  if (identity_ == 2 &&
      RuntimeProbeState::mode.load(std::memory_order_relaxed) == RuntimeProbeMode::kThrowStart) {
    throw std::runtime_error{"runtime test start failure"};
  }
#endif
}

[[nodiscard]] expected<void, Error> RuntimeProbeOperator::compute(ExecutionContext& context) {
  const RuntimeProbeMode callback_mode = RuntimeProbeState::mode.load(std::memory_order_relaxed);
  if (callback_mode == RuntimeProbeMode::kWaitFromCompute ||
      callback_mode == RuntimeProbeMode::kDestroyFromCompute) {
    RuntimeProbeState::invoke_session_action_from_callback(callback_mode);
  }
  if (callback_mode == RuntimeProbeMode::kRequestStop ||
      callback_mode == RuntimeProbeMode::kRequestStopTwice) {
    context.request_stop();
    if (callback_mode == RuntimeProbeMode::kRequestStopTwice) {
      context.request_stop();
    }
  }
  if (callback_mode == RuntimeProbeMode::kRequestCompletionThenWaitForStop) {
    context.request_completion();
  }
  const ClockRef clock = context.clock();
  const ClockCapabilities capabilities = clock.capabilities();
  const TimePoint activation_time = context.activation_time();
  RuntimeProbeState::observed_clock_id.store(clock.id().value, std::memory_order_relaxed);
  RuntimeProbeState::observed_activation_clock_id.store(activation_time.clock_id.value,
                                                        std::memory_order_relaxed);
  RuntimeProbeState::observed_activation_time_ns.store(activation_time.timestamp_ns,
                                                       std::memory_order_relaxed);
  const auto clock_time = clock.now();
  RuntimeProbeState::observed_clock_read_succeeded.store(static_cast<bool>(clock_time),
                                                         std::memory_order_relaxed);
  RuntimeProbeState::observed_clock_can_read_now.store(capabilities.can_read_now(),
                                                       std::memory_order_relaxed);
  RuntimeProbeState::observed_clock_can_wait_until.store(capabilities.can_wait_until(),
                                                         std::memory_order_relaxed);
  RuntimeProbeState::observed_clock_steady.store(capabilities.steady(), std::memory_order_relaxed);
  if (clock_time) {
    RuntimeProbeState::observed_clock_time_ns.store(clock_time->timestamp_ns,
                                                    std::memory_order_relaxed);
  }

  const auto now_active = RuntimeProbeState::active.fetch_add(1, std::memory_order_acq_rel) + 1U;
  auto maximum = RuntimeProbeState::max_active.load(std::memory_order_relaxed);
  while (maximum < now_active && !RuntimeProbeState::max_active.compare_exchange_weak(
                                     maximum, now_active, std::memory_order_relaxed)) {
  }
  RuntimeProbeState::compute_entered.store(true, std::memory_order_release);
#if defined(__linux__)
  if (const int ready_fd =
          RuntimeProbeState::external_ready_fd.exchange(-1, std::memory_order_acq_rel);
      ready_fd >= 0) {
    const std::uint8_t ready{1U};
    [[maybe_unused]] const ssize_t written = write(ready_fd, &ready, sizeof(ready));
  }
#endif
  RuntimeProbeState::record(identity_ * 10 + 2);

  const RuntimeProbeMode mode = RuntimeProbeState::mode.load(std::memory_order_relaxed);
  if (mode == RuntimeProbeMode::kWaitForStop || mode == RuntimeProbeMode::kPollForStop ||
      mode == RuntimeProbeMode::kIgnoreStopUntilReleased ||
      mode == RuntimeProbeMode::kWaitForExternalRelease ||
      mode == RuntimeProbeMode::kWaitForStopThenFail ||
      mode == RuntimeProbeMode::kRequestCompletionThenWaitForStop) {
#if defined(__linux__)
    const int gate_fd =
        RuntimeProbeState::external_stop_gate_fd.exchange(-1, std::memory_order_acq_rel);
    if ((mode == RuntimeProbeMode::kIgnoreStopUntilReleased ||
         mode == RuntimeProbeMode::kWaitForExternalRelease) &&
        gate_fd >= 0) {
      std::uint8_t release{};
      [[maybe_unused]] const ssize_t read_count = read(gate_fd, &release, sizeof(release));
    }
#endif
    if (mode != RuntimeProbeMode::kWaitForExternalRelease) {
      while (!context.cancellation().stop_requested()) {
        if (mode == RuntimeProbeMode::kPollForStop) {
          std::this_thread::sleep_for(std::chrono::milliseconds{2});
        } else {
          std::this_thread::yield();
        }
      }
      RuntimeProbeState::stop_observed_time_ns.store(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now().time_since_epoch())
              .count(),
          std::memory_order_release);
    }
#if defined(__linux__)
    if (mode != RuntimeProbeMode::kIgnoreStopUntilReleased &&
        mode != RuntimeProbeMode::kWaitForExternalRelease && gate_fd >= 0) {
      std::uint8_t release{};
      [[maybe_unused]] const ssize_t read_count = read(gate_fd, &release, sizeof(release));
    }
#endif
  }
  RuntimeProbeState::active.fetch_sub(1, std::memory_order_acq_rel);
  RuntimeProbeState::callback_return_time_ns.store(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count(),
      std::memory_order_release);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  if (mode == RuntimeProbeMode::kThrowCompute) {
    throw std::runtime_error{"runtime test compute failure"};
  }
#endif
  if (mode == RuntimeProbeMode::kFail || mode == RuntimeProbeMode::kWaitForStopThenFail) {
    return make_unexpected(Error{ErrorCode::kFailure, "RUNTIME_TEST_COMPUTE_FAILED"});
  }
  return {};
}

void RuntimeProbeOperator::stop() {
  RuntimeProbeState::record(identity_ * 10 + 3);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  if (RuntimeProbeState::mode.load(std::memory_order_relaxed) == RuntimeProbeMode::kThrowStop) {
    throw std::runtime_error{"runtime test stop failure"};
  }
#endif
}

[[nodiscard]] expected<void, Error> EventPolicyParamOperator::compute(ExecutionContext&) {
  return {};
}

void LifecycleParamProbeState::reset(bool hold_first_callback, bool fail_restart_start,
                                     std::int32_t failure_identity,
                                     bool hold_restart_start_callback) {
  std::lock_guard lock(mutex);
  lifecycle.clear();
  compute_observations.clear();
  hold_first = hold_first_callback;
  release_first = !hold_first_callback;
  fail_start = fail_restart_start;
  fail_start_identity = failure_identity;
  failed_start = false;
  hold_restart_start = hold_restart_start_callback;
  release_restart_start = !hold_restart_start_callback;
  restart_start_entered = false;
  restart_start_observed_cancellation = false;
}

void LifecycleParamProbeState::record_lifecycle(LifecycleContext& context, LifecycleStage stage,
                                                std::int32_t operator_identity) {
  const auto& params = context.params<LifecycleParamProbeParams>();
  std::unique_lock lock(mutex);
  lifecycle.push_back({
      .operator_identity = operator_identity,
      .stage = stage,
      .phase = context.transition_phase(),
      .value = params.value,
      .generation = context.params_generation(),
      .address = &params,
  });
  cv.notify_all();
  if (stage == LifecycleStage::kStart &&
      context.transition_phase() == LifecycleTransitionPhase::kRestartForward &&
      hold_restart_start) {
    restart_start_entered = true;
    cv.notify_all();
    cv.wait_for(lock, std::chrono::seconds{5}, [] { return release_restart_start; });
    restart_start_observed_cancellation = context.cancellation().stop_requested();
    cv.notify_all();
  }
}

[[nodiscard]] bool LifecycleParamProbeState::should_fail_start(LifecycleContext& context,
                                                               std::int32_t operator_identity) {
  std::lock_guard lock(mutex);
  if (fail_start && !failed_start && operator_identity == fail_start_identity &&
      context.transition_phase() == LifecycleTransitionPhase::kRestartForward) {
    failed_start = true;
    return true;
  }
  return false;
}

void LifecycleParamProbeState::record_compute(ExecutionContext& context,
                                              std::int32_t operator_identity) {
  const auto& params = context.params<LifecycleParamProbeParams>();
  std::unique_lock lock(mutex);
  const std::size_t index = compute_observations.size();
  compute_observations.push_back({
      .operator_identity = operator_identity,
      .value = params.value,
      .generation = context.params_generation(),
  });
  cv.notify_all();
  if (index == 0U && hold_first) {
    cv.wait_for(lock, std::chrono::seconds{5}, [] { return release_first; });
  }
}

[[nodiscard]] bool LifecycleParamProbeState::wait_until_compute_count(std::size_t count) {
  std::unique_lock lock(mutex);
  return cv.wait_for(
      lock, std::chrono::seconds{5}, [count] { return compute_observations.size() >= count; });
}

void LifecycleParamProbeState::release_first_callback() {
  std::lock_guard lock(mutex);
  release_first = true;
  cv.notify_all();
}

[[nodiscard]] bool LifecycleParamProbeState::wait_until_restart_start() {
  std::unique_lock lock(mutex);
  return cv.wait_for(lock, std::chrono::seconds{5}, [] { return restart_start_entered; });
}

void LifecycleParamProbeState::release_restart_start_callback() {
  std::lock_guard lock(mutex);
  release_restart_start = true;
  cv.notify_all();
}

[[nodiscard]] bool LifecycleParamProbeState::restart_start_saw_cancellation() {
  std::lock_guard lock(mutex);
  return restart_start_observed_cancellation;
}

[[nodiscard]] std::vector<LifecycleParamObservation>
LifecycleParamProbeState::lifecycle_snapshot() {
  std::lock_guard lock(mutex);
  return lifecycle;
}

[[nodiscard]] std::vector<LifecycleParamComputeObservation>
LifecycleParamProbeState::compute_snapshot() {
  std::lock_guard lock(mutex);
  return compute_observations;
}

LifecycleParamProbeOperator::LifecycleParamProbeOperator(std::int32_t identity) noexcept
    : identity_(identity) {}

void LifecycleParamProbeOperator::setup(OperatorSpec& spec) {
  spec.spin_source(spin_, "activation").capacity(1);
  spec.lifecycle()
      .stage(LifecycleStage::kAllocate, &LifecycleParamProbeOperator::allocate)
      .stage(LifecycleStage::kWarmUp, &LifecycleParamProbeOperator::warm_up)
      .stage(LifecycleStage::kArm, &LifecycleParamProbeOperator::arm)
      .stage(LifecycleStage::kStart, &LifecycleParamProbeOperator::start_stage)
      .stage(LifecycleStage::kStop, &LifecycleParamProbeOperator::stop_stage)
      .stage(LifecycleStage::kRelease, &LifecycleParamProbeOperator::release);
}

[[nodiscard]] Contract LifecycleParamProbeOperator::contract() const {
  Contract result;
  result.trigger(OnSpin{.event = spin_});
  return result;
}

[[nodiscard]] expected<void, Error> LifecycleParamProbeOperator::compute(
    ExecutionContext& context) {
  LifecycleParamProbeState::record_compute(context, identity_);
  return {};
}

LifecycleStatus LifecycleParamProbeOperator::allocate(LifecycleContext& context) noexcept {
  return record(context, LifecycleStage::kAllocate);
}

LifecycleStatus LifecycleParamProbeOperator::warm_up(LifecycleContext& context) noexcept {
  return record(context, LifecycleStage::kWarmUp);
}

LifecycleStatus LifecycleParamProbeOperator::arm(LifecycleContext& context) noexcept {
  return record(context, LifecycleStage::kArm);
}

LifecycleStatus LifecycleParamProbeOperator::start_stage(LifecycleContext& context) noexcept {
  const LifecycleStatus result = record(context, LifecycleStage::kStart);
  return LifecycleParamProbeState::should_fail_start(context, identity_)
             ? LifecycleStatus::kFatalFailure
             : result;
}

LifecycleStatus LifecycleParamProbeOperator::stop_stage(LifecycleContext& context) noexcept {
  return record(context, LifecycleStage::kStop);
}

LifecycleStatus LifecycleParamProbeOperator::release(LifecycleContext& context) noexcept {
  return record(context, LifecycleStage::kRelease);
}

LifecycleStatus LifecycleParamProbeOperator::record(LifecycleContext& context,
                                                    LifecycleStage stage) const noexcept {
  LifecycleParamProbeState::record_lifecycle(context, stage, identity_);
  return LifecycleStatus::kOk;
}

[[nodiscard]] expected<void, Error> RuntimeDurationParamProbeOperator::compute(ExecutionContext&) {
  return {};
}

[[nodiscard]] expected<void, Error> RuntimeValidationParamProbeOperator::compute(
    ExecutionContext&) {
  return {};
}

[[nodiscard]] expected<void, Error> RuntimeFloatParamProbeOperator::compute(ExecutionContext&) {
  return {};
}

[[nodiscard]] expected<void, Error> MixedMutabilityParamProbeOperator::compute(ExecutionContext&) {
  return {};
}

void RuntimeParamProbeState::reset(bool hold_first_callback, bool hold_stop_callback,
                                   bool hold_second_callback) {
  std::lock_guard lock(mutex);
  observations.clear();
  hold_first = hold_first_callback;
  release_first = !hold_first_callback;
  hold_second = hold_second_callback;
  release_second = !hold_second_callback;
  hold_start = false;
  start_entered = false;
  release_start = true;
  hold_stop = hold_stop_callback;
  stop_entered = false;
  release_stop = !hold_stop_callback;
  stop_observed_value.store(0, std::memory_order_relaxed);
  stop_observed_params.store(false, std::memory_order_relaxed);
}

[[nodiscard]] bool RuntimeParamProbeState::wait_until_callback_enters(std::size_t count) {
  std::unique_lock lock(mutex);
  return cv.wait_for(
      lock, std::chrono::seconds{5}, [count] { return observations.size() >= count; });
}

void RuntimeParamProbeState::release_first_callback() {
  std::lock_guard lock(mutex);
  release_first = true;
  cv.notify_all();
}

void RuntimeParamProbeState::release_second_callback() {
  std::lock_guard lock(mutex);
  release_second = true;
  cv.notify_all();
}

void RuntimeParamProbeState::hold_start_callback() {
  std::lock_guard lock(mutex);
  hold_start = true;
  start_entered = false;
  release_start = false;
}

[[nodiscard]] bool RuntimeParamProbeState::wait_until_start_enters() {
  std::unique_lock lock(mutex);
  return cv.wait_for(lock, std::chrono::seconds{5}, [] { return start_entered; });
}

void RuntimeParamProbeState::release_start_callback() {
  std::lock_guard lock(mutex);
  release_start = true;
  cv.notify_all();
}

[[nodiscard]] bool RuntimeParamProbeState::wait_until_stop_enters() {
  std::unique_lock lock(mutex);
  return cv.wait_for(lock, std::chrono::seconds{5}, [] { return stop_entered; });
}

void RuntimeParamProbeState::release_stop_callback() {
  std::lock_guard lock(mutex);
  release_stop = true;
  cv.notify_all();
}

[[nodiscard]] std::vector<RuntimeParamObservation> RuntimeParamProbeState::snapshot() {
  std::lock_guard lock(mutex);
  return observations;
}

void RuntimeParamProbeOperator::setup(OperatorSpec& spec) {
  spec.spin_source(spin_, "activation").capacity(1);
}

[[nodiscard]] Contract RuntimeParamProbeOperator::contract() const {
  Contract result;
  result.trigger(OnSpin{.event = spin_});
  return result;
}

void RuntimeParamProbeOperator::start() {
  {
    std::unique_lock lock(RuntimeParamProbeState::mutex);
    RuntimeParamProbeState::start_entered = true;
    RuntimeParamProbeState::cv.notify_all();
    if (RuntimeParamProbeState::hold_start) {
      RuntimeParamProbeState::cv.wait_for(
          lock, std::chrono::seconds{5}, [] { return RuntimeParamProbeState::release_start; });
    }
  }
  RuntimeProbeState::record(901);
}

[[nodiscard]] expected<void, Error> RuntimeParamProbeOperator::compute(ExecutionContext& context) {
  const RuntimeProbeParams& before = context.params<RuntimeProbeParams>();
  const ParamGeneration before_generation = context.params_generation();
  const Tag before_tag = context.tag();
  const TagTime before_tag_time = context.tag_time();
  const TimePoint before_activation_time = context.activation_time();
  const ActivationStatus before_activation = context.activation();
  std::size_t observation_index{};
  {
    std::unique_lock lock(RuntimeParamProbeState::mutex);
    observation_index = RuntimeParamProbeState::observations.size();
    RuntimeParamProbeState::observations.push_back(RuntimeParamObservation{
        .before_value = before.value,
        .after_value = before.value,
        .before_generation = before_generation,
        .after_generation = before_generation,
        .before_address = &before,
        .after_address = &before,
        .before_tag = before_tag,
        .after_tag = before_tag,
        .before_tag_time = before_tag_time,
        .after_tag_time = before_tag_time,
        .before_activation_time = before_activation_time,
        .after_activation_time = before_activation_time,
        .before_activation = before_activation,
        .after_activation = before_activation,
    });
    RuntimeParamProbeState::cv.notify_all();
    if (observation_index == 0U && RuntimeParamProbeState::hold_first) {
      RuntimeParamProbeState::cv.wait_for(
          lock, std::chrono::seconds{5}, [] { return RuntimeParamProbeState::release_first; });
    }
    if (observation_index == 1U && RuntimeParamProbeState::hold_second) {
      RuntimeParamProbeState::cv.wait_for(
          lock, std::chrono::seconds{5}, [] { return RuntimeParamProbeState::release_second; });
    }
  }

  const RuntimeProbeParams& after = context.params<RuntimeProbeParams>();
  const ParamGeneration after_generation = context.params_generation();
  const Tag after_tag = context.tag();
  const TagTime after_tag_time = context.tag_time();
  const TimePoint after_activation_time = context.activation_time();
  const ActivationStatus after_activation = context.activation();
  {
    std::lock_guard lock(RuntimeParamProbeState::mutex);
    auto& observation = RuntimeParamProbeState::observations[observation_index];
    observation.after_value = after.value;
    observation.after_generation = after_generation;
    observation.after_address = &after;
    observation.after_tag = after_tag;
    observation.after_tag_time = after_tag_time;
    observation.after_activation_time = after_activation_time;
    observation.after_activation = after_activation;
    RuntimeParamProbeState::cv.notify_all();
  }
  RuntimeProbeState::record(902);
  return {};
}

void RuntimeParamProbeOperator::stop() {
  RuntimeParamProbeState::stop_observed_value.store(params().value, std::memory_order_relaxed);
  RuntimeParamProbeState::stop_observed_params.store(true, std::memory_order_release);
  {
    std::unique_lock lock(RuntimeParamProbeState::mutex);
    RuntimeParamProbeState::stop_entered = true;
    RuntimeParamProbeState::cv.notify_all();
    if (RuntimeParamProbeState::hold_stop) {
      RuntimeParamProbeState::cv.wait_for(
          lock, std::chrono::seconds{5}, [] { return RuntimeParamProbeState::release_stop; });
    }
  }
  RuntimeProbeState::record(903);
}

void RuntimeDeadlineParamProbeState::reset() noexcept {
  compute_count.store(0U, std::memory_order_relaxed);
  observed_deadline_ns.store(0, std::memory_order_relaxed);
  observed_generation.store(0U, std::memory_order_relaxed);
  observed_reason.store(ActivationReason::kNominal, std::memory_order_relaxed);
  replay_violation.store(OnViolation::kFailRun, std::memory_order_relaxed);
}

void RuntimeDeadlineParamProbeOperator::setup(OperatorSpec& spec) {
  spec.spin_source(spin_, "activation").capacity(1);
}

[[nodiscard]] Contract RuntimeDeadlineParamProbeOperator::contract() const {
  Contract result;
  result.trigger(OnSpin{.event = spin_})
      .deadline(param_ref<&RuntimeDeadlineProbeParams::deadline>())
      .on_violation(
          RuntimeDeadlineParamProbeState::replay_violation.load(std::memory_order_relaxed));
  return result;
}

void RuntimeDeadlineParamProbeOperator::start() {
  std::unique_lock lock(RuntimeParamProbeState::mutex);
  RuntimeParamProbeState::start_entered = true;
  RuntimeParamProbeState::cv.notify_all();
  if (RuntimeParamProbeState::hold_start) {
    RuntimeParamProbeState::cv.wait_for(
        lock, std::chrono::seconds{5}, [] { return RuntimeParamProbeState::release_start; });
  }
}

[[nodiscard]] expected<void, Error> RuntimeDeadlineParamProbeOperator::compute(
    ExecutionContext& context) {
  RuntimeDeadlineParamProbeState::observed_deadline_ns.store(
      context.params<RuntimeDeadlineProbeParams>().deadline.count(), std::memory_order_relaxed);
  RuntimeDeadlineParamProbeState::observed_generation.store(context.params_generation(),
                                                            std::memory_order_relaxed);
  RuntimeDeadlineParamProbeState::observed_reason.store(context.activation().reason,
                                                        std::memory_order_relaxed);
  RuntimeDeadlineParamProbeState::compute_count.fetch_add(1U, std::memory_order_release);
  return {};
}

void RuntimeClockProbeOperator::setup(OperatorSpec&) {}

[[nodiscard]] Contract RuntimeClockProbeOperator::contract() const {
  Contract result;
  result.trigger(OnClock{.period = std::chrono::nanoseconds{RuntimeClockProbeState::kPeriodNs}});
  return result;
}

[[nodiscard]] expected<void, Error> RuntimeClockProbeOperator::compute(ExecutionContext& context) {
  switch (RuntimeClockProbeState::termination_request_mode.exchange(
      RuntimeClockProbeState::TerminationRequestMode::kNone, std::memory_order_acq_rel)) {
    case RuntimeClockProbeState::TerminationRequestMode::kCompletion:
      context.request_completion();
      break;
    case RuntimeClockProbeState::TerminationRequestMode::kStop:
      context.request_stop();
      break;
    case RuntimeClockProbeState::TerminationRequestMode::kCompletionThenStop:
      context.request_completion();
      context.request_stop();
      break;
    case RuntimeClockProbeState::TerminationRequestMode::kStopThenCompletion:
      context.request_stop();
      context.request_completion();
      break;
    case RuntimeClockProbeState::TerminationRequestMode::kCompletionThenFailure:
      context.request_completion();
      return make_unexpected(Error{ErrorCode::kFailure, "intentional termination request failure"});
    case RuntimeClockProbeState::TerminationRequestMode::kNone:
      break;
  }
  // The deterministic-serial Runtime guarantees one active callback at a
  // time. Publish the fixed evidence prefix before the release-store so the
  // external test can request stop without a mutex in this hot callback.
  const std::size_t index = RuntimeClockProbeState::compute_count.load(std::memory_order_relaxed);
  if (index < RuntimeClockProbeState::kRecordedActivations) {
    RuntimeClockProbeState::tags[index] = context.tag();
    RuntimeClockProbeState::tag_times[index] = context.tag_time();
    RuntimeClockProbeState::activation_times[index] = context.activation_time();
  }
  RuntimeProbeState::runtime_clock_ns.fetch_add(RuntimeClockProbeState::kPeriodNs,
                                                std::memory_order_relaxed);
  RuntimeClockProbeState::compute_count.store(index + 1U, std::memory_order_release);
  RuntimeClockProbeState::cv.notify_all();
  return {};
}

void RuntimeNotificationProbeOperator::setup(OperatorSpec& spec) {
  spec.notification_source(notification_, "notification")
      .capacity(1U)
      .sender_reference_capacity(1U);
  spec.lifecycle().stage(LifecycleStage::kArm, &RuntimeNotificationProbeOperator::arm);
}

[[nodiscard]] Contract RuntimeNotificationProbeOperator::contract() const {
  Contract result;
  result.trigger(OnNotified{.event = notification_});
  return result;
}

[[nodiscard]] expected<void, Error> RuntimeNotificationProbeOperator::compute(ExecutionContext&) {
  std::unique_lock lock(RuntimeNotificationProbeState::mutex);
  RuntimeNotificationProbeState::compute_entered = true;
  RuntimeNotificationProbeState::cv.notify_all();
  const bool released = RuntimeNotificationProbeState::cv.wait_for(
      lock, std::chrono::seconds{5}, [] { return RuntimeNotificationProbeState::release_compute; });
  if (!released) {
    RuntimeNotificationProbeState::compute_entered = false;
    RuntimeNotificationProbeState::cv.notify_all();
    return make_unexpected(Error{ErrorCode::kTimeout});
  }
  ++RuntimeNotificationProbeState::compute_count;
  RuntimeNotificationProbeState::compute_entered = false;
  RuntimeNotificationProbeState::cv.notify_all();
  return {};
}

LifecycleStatus RuntimeNotificationProbeOperator::arm(LifecycleContext& context) noexcept {
  auto sender = context.notification_sender(notification_);
  if (!sender) {
    return LifecycleStatus::kFatalFailure;
  }
  auto before_start = sender->post();
  auto second = context.notification_sender(notification_);
  RuntimeNotificationProbeState::publish_sender(
      std::move(*sender),
      before_start ? ErrorCode::kSuccess : before_start.error().code,
      second ? ErrorCode::kSuccess : second.error().code);
  return LifecycleStatus::kOk;
}

void RuntimeClockPredicateErrorOperator::setup(OperatorSpec&) {}

[[nodiscard]] Contract RuntimeClockPredicateErrorOperator::contract() const {
  auto evaluator =
      predicate("runtime-clock-predicate",
                [](const PredicateEvaluationContext&) noexcept -> expected<bool, Error> {
                  return make_unexpected(
                      Error{ErrorCode::kProtocolError, "runtime predicate failure detail"});
                });
  Contract result;
  result
      .trigger(OnPredicate{
          .evaluate_on =
              OnClock{.period = std::chrono::nanoseconds{RuntimeClockProbeState::kPeriodNs}},
          .predicate = std::move(evaluator),
          .firing = PredicateFiring::kOnEveryTrueEvaluation,
      })
      .deadline(std::chrono::milliseconds{1})
      .on_violation(OnViolation::kEmitWitness);
  return result;
}

[[nodiscard]] expected<void, Error> RuntimeClockPredicateErrorOperator::compute(ExecutionContext&) {
  return {};
}

RuntimeClockPredicateProbeOperator::RuntimeClockPredicateProbeOperator(
    RuntimeClockPredicatePattern pattern, PredicateFiring firing, bool restart_after_first_compute)
    : pattern_(pattern),
      firing_(firing),
      restart_after_first_compute_(restart_after_first_compute) {}

void RuntimeClockPredicateProbeOperator::setup(OperatorSpec&) {}

[[nodiscard]] Contract RuntimeClockPredicateProbeOperator::contract() const {
  struct State {
    std::uint32_t evaluation_count{};
  };
  const RuntimeClockPredicatePattern pattern = pattern_;
  auto evaluator = predicate<State>(
      "runtime-clock-predicate-success",
      [pattern](const PredicateEvaluationContext& context, State& state) noexcept {
        ++state.evaluation_count;
        const bool result = pattern == RuntimeClockPredicatePattern::kAlwaysTrue ||
                            (state.evaluation_count & 1U) != 0U;
        RuntimeClockPredicateProbeState::record_evaluation(context, state.evaluation_count, result);
        return result;
      });
  Contract result;
  result.trigger(OnPredicate{
      .evaluate_on = OnClock{.period = std::chrono::nanoseconds{RuntimeClockProbeState::kPeriodNs}},
      .predicate = std::move(evaluator),
      .firing = firing_,
  });
  return result;
}

[[nodiscard]] expected<void, Error> RuntimeClockPredicateProbeOperator::compute(
    ExecutionContext& context) {
  const std::size_t index =
      RuntimeClockPredicateProbeState::compute_count.load(std::memory_order_relaxed);
  RuntimeClockPredicateProbeState::record_compute(context,
                                                  restart_after_first_compute_ && index == 0U);
  if (index == 1U) {
    while (!context.cancellation().stop_requested()) {
      std::this_thread::yield();
    }
  }
  return {};
}

void RuntimeInputContractSourceOperator::setup(OperatorSpec& spec) {
  spec.output(driver, "driver");
  spec.output(additional, "additional");
  spec.output(compute_only, "compute_only");
}

[[nodiscard]] Contract RuntimeInputContractSourceOperator::contract() const {
  Contract result;
  result.trigger(OnClock{.period = std::chrono::milliseconds{1}});
  return result;
}

[[nodiscard]] expected<void, Error> RuntimeInputContractSourceOperator::compute(ExecutionContext&) {
  return {};
}

void RuntimeAlignedChannelProbeState::reset(RuntimeAlignedChannelScript requested) {
  std::lock_guard lock(mutex);
  script.store(requested, std::memory_order_relaxed);
  next_step.store(0U, std::memory_order_relaxed);
  source_compute_count.store(0U, std::memory_order_relaxed);
  sink_compute_count.store(0U, std::memory_order_relaxed);
  observed_first.store(0U, std::memory_order_relaxed);
  observed_second.store(0U, std::memory_order_relaxed);
  observed_first_capture.store(0, std::memory_order_relaxed);
  observed_second_capture.store(0, std::memory_order_relaxed);
}

[[nodiscard]] bool RuntimeAlignedChannelProbeState::wait_for_source(std::size_t expected) {
  std::unique_lock lock(mutex);
  return cv.wait_for(lock, std::chrono::seconds{5}, [expected] {
    return source_compute_count.load(std::memory_order_acquire) >= expected;
  });
}

[[nodiscard]] bool RuntimeAlignedChannelProbeState::wait_for_sink(std::size_t expected) {
  std::unique_lock lock(mutex);
  return cv.wait_for(lock, std::chrono::seconds{5}, [expected] {
    return sink_compute_count.load(std::memory_order_acquire) >= expected;
  });
}

void RuntimeAlignedChannelProbeState::record_source_compute() {
  source_compute_count.fetch_add(1U, std::memory_order_release);
  cv.notify_all();
}

void RuntimeAlignedChannelProbeState::record_sink_compute(const Sample<std::uint32_t>& first,
                                                          const Sample<std::uint32_t>& second) {
  observed_first.store(first.data, std::memory_order_relaxed);
  observed_second.store(second.data, std::memory_order_relaxed);
  observed_first_capture.store(first.metadata.capture_timestamp_ns, std::memory_order_relaxed);
  observed_second_capture.store(second.metadata.capture_timestamp_ns, std::memory_order_relaxed);
  sink_compute_count.fetch_add(1U, std::memory_order_release);
  cv.notify_all();
}

void RuntimeAlignedChannelSourceOperator::setup(OperatorSpec& spec) {
  spec.output(first, "first").max_emits_per_compute(2U);
  spec.output(second, "second").max_emits_per_compute(1U);
}

[[nodiscard]] Contract RuntimeAlignedChannelSourceOperator::contract() const {
  Contract result;
  // Keep the producer alive while the Channel-backed sink is pending without
  // introducing a second source activation into this one-shot selection test.
  result.trigger(OnClock{.period = std::chrono::seconds{30}});
  return result;
}

[[nodiscard]] expected<void, Error> RuntimeAlignedChannelSourceOperator::compute(
    ExecutionContext& context) {
  const auto emit_captured =
      [](Output<std::uint32_t>& output, std::uint32_t value, std::int64_t capture_timestamp_ns) {
        return output.emit(Sample<std::uint32_t>{
            value,
            SampleMetadata{.capture_timestamp_ns = capture_timestamp_ns,
                           .source_clock_id = ClockId{1U}},
        });
      };

  const std::size_t step =
      RuntimeAlignedChannelProbeState::next_step.fetch_add(1U, std::memory_order_relaxed);
  const auto script = RuntimeAlignedChannelProbeState::script.load(std::memory_order_relaxed);
  if (step != 0U) {
    RuntimeAlignedChannelProbeState::record_source_compute();
    return {};
  }

  const TimePoint activation_time = context.activation_time();
  if (!activation_time.valid() || activation_time.clock_id != ClockId{1U}) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  const std::int64_t base = activation_time.timestamp_ns;
  auto emitted = emit_captured(first, 1U, base);
  if (!emitted) {
    return emitted;
  }
  if (script == RuntimeAlignedChannelScript::kMisalignmentProgress) {
    emitted = emit_captured(first, 3U, base + 10);
    if (!emitted) {
      return emitted;
    }
    emitted = emit_captured(second, 2U, base + 20);
  } else if (script == RuntimeAlignedChannelScript::kExactBoundary) {
    emitted = emit_captured(second, 2U, base + 10);
  }
  RuntimeAlignedChannelProbeState::record_source_compute();
  return emitted;
}

void RuntimeAlignedChannelSinkOperator::setup(OperatorSpec& spec) {
  spec.input(first, "first").queue_depth(8U);
  spec.input(second, "second").queue_depth(8U);
}

[[nodiscard]] Contract RuntimeAlignedChannelSinkOperator::contract() const {
  Contract result;
  result.trigger(OnAligned{.ports = ports(first, second), .window = std::chrono::nanoseconds{10}})
      .input(first, Exact{})
      .input(second, Exact{});
  return result;
}

[[nodiscard]] expected<void, Error> RuntimeAlignedChannelSinkOperator::compute(ExecutionContext&) {
  auto first_sample = first.receive();
  if (!first_sample) {
    return make_unexpected(std::move(first_sample.error()));
  }
  auto second_sample = second.receive();
  if (!second_sample) {
    return make_unexpected(std::move(second_sample.error()));
  }
  RuntimeAlignedChannelProbeState::record_sink_compute(*first_sample, *second_sample);
  return {};
}

void RuntimeUnconnectedPortProbeState::reset() noexcept {
  compute_count.store(0U, std::memory_order_relaxed);
}

[[nodiscard]] bool RuntimeUnconnectedPortProbeState::wait_for_compute() {
  std::unique_lock lock(mutex);
  return cv.wait_for(lock, std::chrono::seconds{5}, [] {
    return compute_count.load(std::memory_order_acquire) != 0U;
  });
}

void RuntimeUnconnectedPortClockOperator::setup(OperatorSpec& spec) {
  spec.input(input, "unused-input").queue_depth(1U);
  spec.output(output, "unused-output").max_emits_per_compute(1U);
}

[[nodiscard]] Contract RuntimeUnconnectedPortClockOperator::contract() const {
  Contract result;
  result.trigger(OnClock{.period = std::chrono::seconds{30}});
  return result;
}

[[nodiscard]] expected<void, Error> RuntimeUnconnectedPortClockOperator::compute(
    ExecutionContext&) {
  RuntimeUnconnectedPortProbeState::compute_count.fetch_add(1U, std::memory_order_release);
  RuntimeUnconnectedPortProbeState::cv.notify_all();
  return {};
}

void RuntimeActivationCutProbeState::reset() {
  std::lock_guard lock(mutex);
  observations.clear();
}

[[nodiscard]] bool RuntimeActivationCutProbeState::wait_for_sinks(std::size_t expected) {
  std::unique_lock lock(mutex);
  return cv.wait_for(
      lock, std::chrono::seconds{5}, [expected] { return observations.size() >= expected; });
}

void RuntimeActivationCutProbeState::record(const Sample<std::uint32_t>& sample, TagTime tag_time) {
  {
    std::lock_guard lock(mutex);
    observations.push_back(RuntimeActivationCutObservation{
        .value = sample.data,
        .capture_timestamp_ns = sample.metadata.capture_timestamp_ns,
        .tag_time = tag_time,
    });
  }
  cv.notify_all();
}

[[nodiscard]] std::vector<RuntimeActivationCutObservation>
RuntimeActivationCutProbeState::snapshot() {
  std::lock_guard lock(mutex);
  return observations;
}

void RuntimeActivationCutSourceOperator::setup(OperatorSpec& spec) {
  // Canonical port ordering is lexical, so the prefixes deliberately keep
  // the later TagTime first in the frozen subscriber scan.
  spec.output(later, "a-later").max_emits_per_compute(1U);
  spec.output(earlier, "z-earlier").max_emits_per_compute(1U);
}

[[nodiscard]] Contract RuntimeActivationCutSourceOperator::contract() const {
  Contract result;
  result.trigger(OnClock{.period = std::chrono::seconds{30}});
  return result;
}

[[nodiscard]] expected<void, Error> RuntimeActivationCutSourceOperator::compute(
    ExecutionContext& context) {
  if (emitted_) {
    return {};
  }
  emitted_ = true;
  const TimePoint activation_time = context.activation_time();
  if (!activation_time.valid() || activation_time.clock_id != ClockId{1U}) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  const auto emit_captured =
      [](Output<std::uint32_t>& output, std::uint32_t value, std::int64_t capture_timestamp_ns) {
        return output.emit(Sample<std::uint32_t>{
            value,
            SampleMetadata{.capture_timestamp_ns = capture_timestamp_ns,
                           .source_clock_id = ClockId{1U}},
        });
      };
  const std::int64_t base = activation_time.timestamp_ns;
  auto emitted = emit_captured(later, 10U, base + 10);
  if (!emitted) {
    return emitted;
  }
  // Give the asynchronous pump time to retain the first subscriber's later TagTime before the
  // second publication. Correctness must not depend on which side wins the final post-emit race.
  std::this_thread::sleep_for(std::chrono::milliseconds{50});
  return emit_captured(earlier, 5U, base + 5);
}

void RuntimeActivationCutSinkOperator::setup(OperatorSpec& spec) {
  spec.input(input, "input").queue_depth(1U);
}

[[nodiscard]] Contract RuntimeActivationCutSinkOperator::contract() const {
  Contract result;
  // OnEach already fixes its driver to the exact triggering sample. Re-selecting that same port
  // is invalid temporal authoring and would prevent this runtime-ordering fixture from compiling.
  result.trigger(OnEach{input});
  return result;
}

[[nodiscard]] expected<void, Error> RuntimeActivationCutSinkOperator::compute(
    ExecutionContext& context) {
  auto sample = input.receive();
  if (!sample) {
    return make_unexpected(std::move(sample.error()));
  }
  RuntimeActivationCutProbeState::record(*sample, context.tag_time());
  return {};
}

void RuntimeWindowChannelProbeState::reset(RuntimeWindowChannelScript requested) {
  std::lock_guard lock(mutex);
  script.store(requested, std::memory_order_relaxed);
  compute_count.store(0U, std::memory_order_relaxed);
  safe_default_end_of_stream_count.store(0U, std::memory_order_relaxed);
  downstream_count.store(0U, std::memory_order_relaxed);
  terminal_tags = {};
  terminal_tag_times = {};
  selection_state.store(InputState::kNotSelected, std::memory_order_relaxed);
  selection_reason.store(SelectionReason::kNotSelectedForActivation, std::memory_order_relaxed);
  anchor_capture_ns.store(0, std::memory_order_relaxed);
  activation_time_ns.store(0, std::memory_order_relaxed);
  batch_size.store(0U, std::memory_order_relaxed);
  batch_first_capture_ns.store(0, std::memory_order_relaxed);
  batch_last_capture_ns.store(0, std::memory_order_relaxed);
}

[[nodiscard]] bool RuntimeWindowChannelProbeState::wait_for_compute() {
  std::unique_lock lock(mutex);
  return cv.wait_for(lock, std::chrono::seconds{5}, [] {
    return compute_count.load(std::memory_order_acquire) != 0U;
  });
}

[[nodiscard]] bool RuntimeWindowChannelProbeState::wait_for_compute_count(std::size_t expected) {
  std::unique_lock lock(mutex);
  return cv.wait_for(lock, std::chrono::seconds{5}, [expected] {
    return compute_count.load(std::memory_order_acquire) >= expected &&
           downstream_count.load(std::memory_order_acquire) >= expected;
  });
}

void RuntimeWindowChannelProbeState::record(InputSelection selection,
                                            const Sample<std::uint32_t>& anchor_sample,
                                            TimePoint activation_time, Tag tag, TagTime tag_time,
                                            std::size_t selected_size,
                                            std::int64_t first_capture_ns,
                                            std::int64_t last_capture_ns) {
  const std::size_t index = compute_count.load(std::memory_order_relaxed);
  if (index < terminal_tags.size()) {
    terminal_tags[index] = tag;
    terminal_tag_times[index] = tag_time;
  }
  selection_state.store(selection.state, std::memory_order_relaxed);
  selection_reason.store(selection.reason, std::memory_order_relaxed);
  anchor_capture_ns.store(anchor_sample.metadata.capture_timestamp_ns, std::memory_order_relaxed);
  activation_time_ns.store(activation_time.timestamp_ns, std::memory_order_relaxed);
  batch_size.store(selected_size, std::memory_order_relaxed);
  batch_first_capture_ns.store(first_capture_ns, std::memory_order_relaxed);
  batch_last_capture_ns.store(last_capture_ns, std::memory_order_relaxed);
  if (selection.state == InputState::kSafeDefault &&
      selection.reason == SelectionReason::kEndOfStream) {
    safe_default_end_of_stream_count.fetch_add(1U, std::memory_order_relaxed);
  }
  compute_count.fetch_add(1U, std::memory_order_release);
  cv.notify_all();
}

void RuntimeWindowChannelSourceOperator::setup(OperatorSpec& spec) {
  spec.output(anchor, "anchor").max_emits_per_compute(3U);
  spec.output(windowed, "windowed").max_emits_per_compute(5U);
}

[[nodiscard]] Contract RuntimeWindowChannelSourceOperator::contract() const {
  Contract result;
  const bool terminal_drain =
      RuntimeWindowChannelProbeState::script.load(std::memory_order_relaxed) ==
      RuntimeWindowChannelScript::kEndOfStreamDrain;
  result.trigger(
      OnClock{.period = terminal_drain ? std::chrono::milliseconds{1} : std::chrono::seconds{30}});
  if (terminal_drain) {
    result.complete_after(ComputeCalls{3U});
  }
  return result;
}

[[nodiscard]] expected<void, Error> RuntimeWindowChannelSourceOperator::compute(
    ExecutionContext& context) {
  const TimePoint activation_time = context.activation_time();
  if (!activation_time.valid() || activation_time.clock_id != ClockId{1U}) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  const auto emit_captured =
      [](Output<std::uint32_t>& output, std::uint32_t value, std::int64_t capture_timestamp_ns) {
        return output.emit(Sample<std::uint32_t>{
            value,
            SampleMetadata{.capture_timestamp_ns = capture_timestamp_ns,
                           .source_clock_id = ClockId{1U}},
        });
      };
  const std::int64_t base = activation_time.timestamp_ns;
  if (RuntimeWindowChannelProbeState::script.load(std::memory_order_relaxed) ==
      RuntimeWindowChannelScript::kEndOfStreamDrain) {
    if (next_terminal_anchor_ > 30U) {
      return make_unexpected(Error{ErrorCode::kProtocolError});
    }
    const std::uint32_t value = next_terminal_anchor_;
    auto emitted = emit_captured(anchor, value, base + value);
    if (emitted) {
      next_terminal_anchor_ += 10U;
    }
    return emitted;
  }
  auto emitted = emit_captured(windowed, 5U, base + 5);
  if (!emitted) {
    return emitted;
  }
  emitted = emit_captured(anchor, 10U, base + 10);
  if (!emitted) {
    return emitted;
  }
  emitted = emit_captured(windowed, 10U, base + 10);
  if (!emitted) {
    return emitted;
  }
  emitted = emit_captured(windowed, 15U, base + 15);
  if (!emitted) {
    return emitted;
  }
  emitted = emit_captured(anchor, 20U, base + 20);
  if (!emitted) {
    return emitted;
  }
  emitted = emit_captured(windowed, 20U, base + 20);
  if (!emitted || RuntimeWindowChannelProbeState::script.load(std::memory_order_relaxed) ==
                      RuntimeWindowChannelScript::kTimeoutClose) {
    return emitted;
  }
  return emit_captured(windowed, 25U, base + 25);
}

void RuntimeWindowChannelRelayOperator::setup(OperatorSpec& spec) {
  spec.input(input, "input").queue_depth(2U);
  spec.output(anchor, "anchor").max_emits_per_compute(1U);
  spec.output(windowed, "windowed").max_emits_per_compute(1U);
}

[[nodiscard]] Contract RuntimeWindowChannelRelayOperator::contract() const {
  Contract result;
  result.trigger(OnEach{input});
  return result;
}

[[nodiscard]] expected<void, Error> RuntimeWindowChannelRelayOperator::compute(ExecutionContext&) {
  auto sample = input.receive();
  if (!sample) {
    return make_unexpected(std::move(sample.error()));
  }
  return anchor.emit(std::move(*sample));
}

void RuntimeWindowChannelSinkOperator::setup(OperatorSpec& spec) {
  spec.input(anchor, "anchor").queue_depth(8U);
  spec.input(windowed, "windowed").queue_depth(16U);
  spec.output(output, "output").max_emits_per_compute(1U);
}

[[nodiscard]] Contract RuntimeWindowChannelSinkOperator::contract() const {
  Contract result;
  const bool terminal_drain =
      RuntimeWindowChannelProbeState::script.load(std::memory_order_relaxed) ==
      RuntimeWindowChannelScript::kEndOfStreamDrain;
  result
      .trigger(OnWindow{
          .anchor = anchor,
          .max_wait = terminal_drain ? std::chrono::seconds{30} : std::chrono::milliseconds{5},
          .max_pending_windows = 2U})
      .input(anchor, Exact{})
      .input(windowed, BracketedWindow{.lookbehind = 1U, .lookahead = 1U, .max_samples = 5U})
      .on_missing(windowed, SafeDefault{});
  return result;
}

[[nodiscard]] expected<void, Error> RuntimeWindowChannelSinkOperator::compute(
    ExecutionContext& context) {
  auto anchor_sample = anchor.receive();
  if (!anchor_sample) {
    return make_unexpected(std::move(anchor_sample.error()));
  }
  const InputSelection selection = windowed.selection();
  std::size_t selected_size = 0U;
  std::int64_t first_capture_ns = 0;
  std::int64_t last_capture_ns = 0;
  if (selection.state == InputState::kSelected) {
    auto selected = windowed.selected_batch();
    if (!selected) {
      return make_unexpected(std::move(selected.error()));
    }
    selected_size = selected->size();
    if (selected_size != 0U) {
      first_capture_ns = (*selected)[0U].metadata.capture_timestamp_ns;
      last_capture_ns = (*selected)[selected_size - 1U].metadata.capture_timestamp_ns;
    }
  }
  RuntimeWindowChannelProbeState::record(selection,
                                         *anchor_sample,
                                         context.activation_time(),
                                         context.tag(),
                                         context.tag_time(),
                                         selected_size,
                                         first_capture_ns,
                                         last_capture_ns);
  if (RuntimeWindowChannelProbeState::script.load(std::memory_order_relaxed) !=
      RuntimeWindowChannelScript::kEndOfStreamDrain) {
    return {};
  }
  return output.emit(anchor_sample->data);
}

void RuntimeWindowChannelDrainOperator::setup(OperatorSpec& spec) {
  spec.input(input, "input").queue_depth(2U);
}

[[nodiscard]] Contract RuntimeWindowChannelDrainOperator::contract() const {
  Contract result;
  result.trigger(OnEach{input});
  return result;
}

[[nodiscard]] expected<void, Error> RuntimeWindowChannelDrainOperator::compute(ExecutionContext&) {
  auto sample = input.receive();
  if (!sample) {
    return make_unexpected(std::move(sample.error()));
  }
  RuntimeWindowChannelProbeState::downstream_count.fetch_add(1U, std::memory_order_release);
  RuntimeWindowChannelProbeState::cv.notify_all();
  return {};
}

void RuntimeInputTriggerReplayOperator::setup(OperatorSpec& spec) {
  spec.input(first, "first").queue_depth(8U);
  spec.input(second, "second").queue_depth(8U);
}

[[nodiscard]] Contract RuntimeInputTriggerReplayOperator::contract() const {
  Contract result;
  switch (kind.load(std::memory_order_relaxed)) {
    case RuntimeInputTriggerReplayKind::kAny:
      result.trigger(OnAnyInput{});
      break;
    case RuntimeInputTriggerReplayKind::kEach:
      result.trigger(OnEach{first});
      break;
    case RuntimeInputTriggerReplayKind::kAligned:
      result
          .trigger(OnAligned{.ports = ports(first, second), .window = std::chrono::milliseconds{1}})
          .input(first, Exact{})
          .input(second, Exact{});
      break;
    case RuntimeInputTriggerReplayKind::kWindow:
      result
          .trigger(OnWindow{
              .anchor = first, .max_wait = std::chrono::milliseconds{2}, .max_pending_windows = 2U})
          .input(first, Exact{})
          .input(second, BracketedWindow{.lookbehind = 1U, .lookahead = 1U, .max_samples = 3U})
          .on_missing(second, Skip{});
      break;
  }
  return result;
}

[[nodiscard]] expected<void, Error> RuntimeInputTriggerReplayOperator::compute(ExecutionContext&) {
  return {};
}

void RuntimeInputContractPredicateOperator::setup(OperatorSpec& spec) {
  spec.input(driver, "driver").queue_depth(8U);
  spec.input(additional, "additional").queue_depth(8U);
  spec.input(compute_only, "compute_only").queue_depth(8U);
}

[[nodiscard]] Contract RuntimeInputContractPredicateOperator::contract() const {
  const RuntimeInputContractReplayMode current = mode.load(std::memory_order_relaxed);
  auto evaluator =
      predicate("runtime-input-predicate",
                predicate_inputs(additional),
                [](const PredicateEvaluationContext&,
                   const PredicateInputView<std::uint32_t>&,
                   const PredicateInputView<std::uint32_t>&) noexcept { return true; });
  Contract result;
  result
      .trigger(OnPredicate{
          .evaluate_on = OnInputUpdate{driver},
          .predicate = std::move(evaluator),
          .firing = current == RuntimeInputContractReplayMode::kPredicateFiringMismatch
                        ? PredicateFiring::kOnEveryTrueEvaluation
                        : PredicateFiring::kOnFalseToTrue,
      })
      .input(driver, Exact{})
      .input(additional, LatestWithin{.max_age = std::chrono::milliseconds{2}})
      .on_missing(additional, SafeDefault{});
  if (current == RuntimeInputContractReplayMode::kSelectorMismatch) {
    result.input(compute_only, Exact{});
  } else {
    result.input(compute_only, DrainAll{});
  }
  result
      .deadline(current == RuntimeInputContractReplayMode::kDeadlineMismatch
                    ? std::chrono::milliseconds{4}
                    : std::chrono::milliseconds{3})
      .on_violation(OnViolation::kEmitWitness);
  return result;
}

[[nodiscard]] expected<void, Error> RuntimeInputContractPredicateOperator::compute(
    ExecutionContext&) {
  return {};
}

RuntimeClockInputReplayOperator::RuntimeClockInputReplayOperator(RuntimeClockInputReplayKind kind,
                                                                 std::uint32_t replay_ordinal)
    : kind_(kind), replay_ordinal_(replay_ordinal) {}

void RuntimeClockInputReplayOperator::setup(OperatorSpec& spec) {
  spec.input(predicate_input, "predicate_input").queue_depth(8U);
  spec.input(compute_only, "compute_only").queue_depth(8U);
}

[[nodiscard]] Contract RuntimeClockInputReplayOperator::contract() const {
  const auto period = mismatch_ordinal.load(std::memory_order_relaxed) == replay_ordinal_
                          ? std::chrono::milliseconds{2}
                          : std::chrono::milliseconds{1};
  Contract result;
  if (kind_ == RuntimeClockInputReplayKind::kClock) {
    result.trigger(OnClock{.period = period});
  } else {
    auto evaluator =
        predicate("runtime-clock-input-predicate",
                  predicate_inputs(predicate_input),
                  [](const PredicateEvaluationContext&,
                     const PredicateInputView<std::uint32_t>&) noexcept { return true; });
    result.trigger(OnPredicate{
        .evaluate_on = OnClock{.period = period},
        .predicate = std::move(evaluator),
        .firing = PredicateFiring::kOnEveryTrueEvaluation,
    });
  }
  result.input(predicate_input, Exact{})
      .on_missing(predicate_input, Skip{})
      .input(compute_only, DrainAll{})
      .deadline(std::chrono::milliseconds{3})
      .on_violation(OnViolation::kEmitWitness);
  return result;
}

[[nodiscard]] expected<void, Error> RuntimeClockInputReplayOperator::compute(ExecutionContext&) {
  return {};
}

void RuntimeInputPredicateChannelState::reset() {
  std::lock_guard lock(mutex);
  evaluator_count.store(0U, std::memory_order_relaxed);
  compute_count.store(0U, std::memory_order_relaxed);
  source_count.store(0U, std::memory_order_relaxed);
  evaluator_driver_value.store(0U, std::memory_order_relaxed);
  evaluator_additional_value.store(0U, std::memory_order_relaxed);
  evaluator_driver_sequence.store(0U, std::memory_order_relaxed);
  evaluator_additional_sequence.store(0U, std::memory_order_relaxed);
  compute_driver_value.store(0U, std::memory_order_relaxed);
  compute_additional_value.store(0U, std::memory_order_relaxed);
  compute_driver_sequence.store(0U, std::memory_order_relaxed);
  compute_additional_sequence.store(0U, std::memory_order_relaxed);
  evaluation_observations.clear();
  compute_observations.clear();
}

[[nodiscard]] bool RuntimeInputPredicateChannelState::wait_for_source() {
  std::unique_lock lock(mutex);
  return cv.wait_for(lock, std::chrono::seconds{5}, [] {
    return source_count.load(std::memory_order_acquire) != 0U;
  });
}

[[nodiscard]] bool RuntimeInputPredicateChannelState::wait_for_compute(std::size_t expected) {
  std::unique_lock lock(mutex);
  return cv.wait_for(lock, std::chrono::seconds{5}, [expected] {
    return compute_count.load(std::memory_order_acquire) >= expected;
  });
}

[[nodiscard]] bool RuntimeInputPredicateChannelState::wait_for_evaluator(std::size_t expected) {
  std::unique_lock lock(mutex);
  return cv.wait_for(lock, std::chrono::seconds{5}, [expected] {
    return evaluator_count.load(std::memory_order_acquire) >= expected;
  });
}

void RuntimeInputPredicateChannelState::record_source() {
  source_count.fetch_add(1U, std::memory_order_release);
  cv.notify_all();
}

void RuntimeInputPredicateChannelState::record_evaluator(const Sample<std::uint32_t>& driver,
                                                         const Sample<std::uint32_t>& additional,
                                                         const PredicateEvaluationContext& context,
                                                         std::uint32_t candidate_state_count,
                                                         const expected<bool, Error>& result) {
  std::lock_guard lock(mutex);
  evaluator_driver_value.store(driver.data, std::memory_order_relaxed);
  evaluator_additional_value.store(additional.data, std::memory_order_relaxed);
  evaluator_driver_sequence.store(driver.metadata.sequence, std::memory_order_relaxed);
  evaluator_additional_sequence.store(additional.metadata.sequence, std::memory_order_relaxed);
  const auto selections = context.selections();
  evaluation_observations.push_back(RuntimeInputPredicateEvaluationObservation{
      .reference_time = context.reference_time(),
      .tag = context.tag(),
      .tag_time = context.tag_time(),
      .driver_selection = selections[0].selection,
      .additional_selection = selections[1].selection,
      .driver_metadata = driver.metadata,
      .additional_metadata = additional.metadata,
      .driver_value = driver.data,
      .additional_value = additional.data,
      .candidate_state_count = candidate_state_count,
      .value = result && *result,
      .failed = !result,
  });
  evaluator_count.fetch_add(1U, std::memory_order_release);
  cv.notify_all();
}

void RuntimeInputPredicateChannelState::record_compute(const Sample<std::uint32_t>& driver,
                                                       const Sample<std::uint32_t>* additional,
                                                       ExecutionContext& context,
                                                       InputSelection driver_selection,
                                                       InputSelection additional_selection,
                                                       InputSelection compute_only_selection) {
  std::lock_guard lock(mutex);
  compute_driver_value.store(driver.data, std::memory_order_relaxed);
  compute_additional_value.store(additional == nullptr ? 0U : additional->data,
                                 std::memory_order_relaxed);
  compute_driver_sequence.store(driver.metadata.sequence, std::memory_order_relaxed);
  compute_additional_sequence.store(additional == nullptr ? 0U : additional->metadata.sequence,
                                    std::memory_order_relaxed);
  compute_observations.push_back(RuntimeInputPredicateComputeObservation{
      .activation_time = context.activation_time(),
      .tag = context.tag(),
      .tag_time = context.tag_time(),
      .driver_selection = driver_selection,
      .additional_selection = additional_selection,
      .compute_only_selection = compute_only_selection,
      .driver_metadata = driver.metadata,
      .additional_metadata = additional == nullptr ? SampleMetadata{} : additional->metadata,
      .driver_value = driver.data,
      .additional_value = additional == nullptr ? 0U : additional->data,
      .has_additional = additional != nullptr,
  });
  compute_count.fetch_add(1U, std::memory_order_release);
  cv.notify_all();
}

[[nodiscard]] std::vector<RuntimeInputPredicateEvaluationObservation>
RuntimeInputPredicateChannelState::evaluation_snapshot() {
  std::lock_guard lock(mutex);
  return evaluation_observations;
}

[[nodiscard]] std::vector<RuntimeInputPredicateComputeObservation>
RuntimeInputPredicateChannelState::compute_snapshot() {
  std::lock_guard lock(mutex);
  return compute_observations;
}

RuntimeInputPredicateChannelSourceOperator::RuntimeInputPredicateChannelSourceOperator(
    RuntimeInputPredicateMissing missing)
    : missing_(missing) {}

void RuntimeInputPredicateChannelSourceOperator::setup(OperatorSpec& spec) {
  // Declare non-driver services first so the process-local pump can populate
  // predicate and compute-only history before observing each driver witness.
  spec.output(additional, "additional").max_emits_per_compute(3U);
  spec.output(compute_only, "compute_only").max_emits_per_compute(3U);
  spec.output(driver, "driver").max_emits_per_compute(3U);
}

[[nodiscard]] Contract RuntimeInputPredicateChannelSourceOperator::contract() const {
  Contract result;
  result.trigger(OnClock{.period = std::chrono::seconds{30}});
  return result;
}

[[nodiscard]] expected<void, Error> RuntimeInputPredicateChannelSourceOperator::compute(
    ExecutionContext& context) {
  if (emitted_) {
    return {};
  }
  emitted_ = true;
  const std::int64_t base = context.activation_time().timestamp_ns;
  const auto emit_captured =
      [](Output<std::uint32_t>& output, std::uint32_t value, std::int64_t capture_ns) {
        return output.emit(Sample<std::uint32_t>{
            value,
            SampleMetadata{.capture_timestamp_ns = capture_ns,
                           .source_clock_id = ClockId{1U},
                           .trace_id = 1000U + value,
                           .frame_id = 2000U + value},
        });
      };
  const std::uint32_t additional_count =
      missing_ == RuntimeInputPredicateMissing::kFresh
          ? 3U
          : (missing_ == RuntimeInputPredicateMissing::kReuseLast ? 1U : 0U);
  for (std::uint32_t index = 1U; index <= additional_count; ++index) {
    auto emitted = emit_captured(additional, index * 10U, base + index * 10U);
    if (!emitted) {
      return emitted;
    }
    emitted = emit_captured(compute_only, index * 100U, base + index * 10U);
    if (!emitted) {
      return emitted;
    }
  }
  for (std::uint32_t index = 1U; index <= 3U; ++index) {
    auto emitted = emit_captured(driver, index, base + index * 10U);
    if (!emitted) {
      return emitted;
    }
  }
  RuntimeInputPredicateChannelState::record_source();
  return {};
}

RuntimeInputPredicateChannelSinkOperator::RuntimeInputPredicateChannelSinkOperator(
    RuntimeInputPredicatePattern pattern, PredicateFiring firing, bool configured,
    RuntimeInputPredicateMissing missing)
    : pattern_(pattern), firing_(firing), configured_(configured), missing_(missing) {}

void RuntimeInputPredicateChannelSinkOperator::setup(OperatorSpec& spec) {
  spec.input(driver, "driver").queue_depth(8U);
  spec.input(additional, "additional").queue_depth(8U);
  spec.input(compute_only, "compute_only").queue_depth(8U);
}

[[nodiscard]] Contract RuntimeInputPredicateChannelSinkOperator::contract() const {
  const RuntimeInputPredicatePattern pattern = pattern_;
  const auto build_contract = [this](PredicateBinding evaluator) {
    Contract result;
    result
        .trigger(OnPredicate{
            .evaluate_on = OnInputUpdate{driver},
            .predicate = std::move(evaluator),
            .firing = firing_,
        })
        .input(driver, Exact{})
        .input(additional, Exact{})
        .input(compute_only, DrainAll{});
    switch (missing_) {
      case RuntimeInputPredicateMissing::kFresh:
      case RuntimeInputPredicateMissing::kSkip:
        result.on_missing(additional, Skip{});
        break;
      case RuntimeInputPredicateMissing::kReuseLast:
        result.on_missing(additional, ReuseLast{.max_age = std::chrono::seconds{1}});
        break;
      case RuntimeInputPredicateMissing::kSafeDefault:
        result.on_missing(additional, SafeDefault{});
        break;
    }
    return result;
  };
  if (configured_) {
    auto evaluator = predicate<RuntimeInputPredicateTestState>(
        "runtime-input-channel-predicate",
        predicate_inputs(additional),
        RuntimeInputPredicateTestConfig{.minimum_driver_value = 2U},
        [pattern](const PredicateEvaluationContext& context,
                  const PredicateInputView<std::uint32_t>& driver_view,
                  const PredicateInputView<std::uint32_t>& additional_view,
                  RuntimeInputPredicateTestState& state,
                  const RuntimeInputPredicateTestConfig& config) noexcept -> expected<bool, Error> {
          ++state.evaluation_count;
          const auto result = runtime_input_predicate_result(pattern,
                                                             state.evaluation_count,
                                                             driver_view.sample().data,
                                                             config.minimum_driver_value);
          RuntimeInputPredicateChannelState::record_evaluator(driver_view.sample(),
                                                              additional_view.sample(),
                                                              context,
                                                              state.evaluation_count,
                                                              result);
          return result;
        });
    return build_contract(std::move(evaluator));
  }
  auto evaluator = predicate<RuntimeInputPredicateTestState>(
      "runtime-input-channel-predicate",
      predicate_inputs(additional),
      [pattern](const PredicateEvaluationContext& context,
                const PredicateInputView<std::uint32_t>& driver_view,
                const PredicateInputView<std::uint32_t>& additional_view,
                RuntimeInputPredicateTestState& state) noexcept -> expected<bool, Error> {
        ++state.evaluation_count;
        const auto result = runtime_input_predicate_result(
            pattern, state.evaluation_count, driver_view.sample().data, 0U);
        RuntimeInputPredicateChannelState::record_evaluator(driver_view.sample(),
                                                            additional_view.sample(),
                                                            context,
                                                            state.evaluation_count,
                                                            result);
        return result;
      });
  return build_contract(std::move(evaluator));
}

[[nodiscard]] expected<void, Error> RuntimeInputPredicateChannelSinkOperator::compute(
    ExecutionContext& context) {
  const InputSelection driver_selection = driver.selection();
  const InputSelection additional_selection = additional.selection();
  const InputSelection compute_only_selection = compute_only.selection();
  auto selected_driver = driver.receive();
  if (!selected_driver) {
    return make_unexpected(std::move(selected_driver.error()));
  }
  std::optional<Sample<std::uint32_t>> selected_additional;
  if (additional_selection.state == InputState::kSelected ||
      additional_selection.state == InputState::kReused) {
    auto received = additional.receive();
    if (!received) {
      return make_unexpected(std::move(received.error()));
    }
    selected_additional.emplace(std::move(*received));
  }
  if (compute_only_selection.state != InputState::kNotSelected) {
    auto drained = compute_only.selected_batch();
    if (!drained) {
      return make_unexpected(std::move(drained.error()));
    }
  }
  RuntimeInputPredicateChannelState::record_compute(
      *selected_driver,
      selected_additional ? &*selected_additional : nullptr,
      context,
      driver_selection,
      additional_selection,
      compute_only_selection);
  return {};
}

std::mutex RuntimeComputeCallChannelProbeState::mutex{};
std::condition_variable RuntimeComputeCallChannelProbeState::cv{};
std::atomic<std::size_t> RuntimeComputeCallChannelProbeState::source_count{};
std::atomic<std::size_t> RuntimeComputeCallChannelProbeState::transform_count{};
std::atomic<std::size_t> RuntimeComputeCallChannelProbeState::sink_compute_count{};
std::atomic<std::size_t> RuntimeComputeCallChannelProbeState::sink_count{};
std::atomic<std::uint32_t> RuntimeComputeCallChannelProbeState::first_value{};
std::atomic<std::uint32_t> RuntimeComputeCallChannelProbeState::second_value{};
std::atomic_bool RuntimeComputeCallChannelProbeState::restart_accepted{};
std::atomic<ErrorCode> RuntimeComputeCallChannelProbeState::restart_error{};
std::atomic<std::uint64_t>
    RuntimeComputeCallChannelProbeState::successful_compute_calls_before_restart{};
std::atomic<std::size_t> RuntimeComputeCallChannelProbeState::restart_forward_start_count{};
std::atomic<std::size_t> RuntimeComputeCallChannelProbeState::restore_old_start_count{};
bool RuntimeComputeCallChannelProbeState::hold_source{};
bool RuntimeComputeCallChannelProbeState::source_entered{};
bool RuntimeComputeCallChannelProbeState::release_source{};

void RuntimeComputeCallChannelProbeState::reset(const bool hold_first_compute) noexcept {
  const std::lock_guard lock(mutex);
  source_count.store(0U, std::memory_order_relaxed);
  transform_count.store(0U, std::memory_order_relaxed);
  sink_compute_count.store(0U, std::memory_order_relaxed);
  sink_count.store(0U, std::memory_order_relaxed);
  first_value.store(0U, std::memory_order_relaxed);
  second_value.store(0U, std::memory_order_relaxed);
  restart_accepted.store(false, std::memory_order_relaxed);
  restart_error.store(ErrorCode::kSuccess, std::memory_order_relaxed);
  successful_compute_calls_before_restart.store(0U, std::memory_order_relaxed);
  restart_forward_start_count.store(0U, std::memory_order_relaxed);
  restore_old_start_count.store(0U, std::memory_order_relaxed);
  hold_source = hold_first_compute;
  source_entered = false;
  release_source = false;
  RuntimeProbeState::runtime_clock_ns.store(0, std::memory_order_relaxed);
}

[[nodiscard]] std::size_t RuntimeComputeCallChannelProbeState::enter_source_compute() {
  const std::size_t call = source_count.fetch_add(1U, std::memory_order_relaxed) + 1U;
  std::unique_lock lock(mutex);
  source_entered = true;
  cv.notify_all();
  if (hold_source) {
    cv.wait_for(lock, std::chrono::seconds{5}, [] { return release_source; });
  }
  return call;
}

[[nodiscard]] bool RuntimeComputeCallChannelProbeState::wait_until_source_compute() {
  std::unique_lock lock(mutex);
  return cv.wait_for(lock, std::chrono::seconds{5}, [] { return source_entered; });
}

void RuntimeComputeCallChannelProbeState::release_source_compute() {
  {
    const std::lock_guard lock(mutex);
    release_source = true;
  }
  cv.notify_all();
}

void RuntimeComputeCallChannelProbeState::record_sink(const std::uint32_t value) noexcept {
  const std::size_t index = sink_count.fetch_add(1U, std::memory_order_release);
  if (index == 0U) {
    first_value.store(value, std::memory_order_relaxed);
  } else if (index == 1U) {
    second_value.store(value, std::memory_order_relaxed);
  }
}

RuntimeComputeCallSourceOperator::RuntimeComputeCallSourceOperator(
    const std::uint64_t limit, const RuntimeComputeCallAction action,
    const std::uint64_t restart_after_successful_compute_calls) noexcept
    : limit_(limit),
      action_(action),
      restart_after_successful_compute_calls_(restart_after_successful_compute_calls) {}

void RuntimeComputeCallSourceOperator::setup(OperatorSpec& spec) {
  spec.output(output, "output").max_emits_per_compute(1U);
  if (action_ == RuntimeComputeCallAction::kRestartAfterComputeCalls ||
      action_ == RuntimeComputeCallAction::kRestartRollbackAfterFirst) {
    spec.lifecycle().stage(LifecycleStage::kStart, &RuntimeComputeCallSourceOperator::start_stage);
  }
}

[[nodiscard]] Contract RuntimeComputeCallSourceOperator::contract() const {
  Contract result;
  result.trigger(OnClock{.period = std::chrono::milliseconds{1}})
      .complete_after(ComputeCalls{.limit = limit_});
  return result;
}

[[nodiscard]] expected<void, Error> RuntimeComputeCallSourceOperator::compute(
    ExecutionContext& context) {
  const std::size_t call = RuntimeComputeCallChannelProbeState::enter_source_compute();
  if (action_ == RuntimeComputeCallAction::kFailOnSecond && call == 2U) {
    return make_unexpected(Error{ErrorCode::kFailure});
  }
  auto emitted = output.emit(static_cast<std::uint32_t>(76U + call));
  if (!emitted) {
    return make_unexpected(std::move(emitted).error());
  }
  if (action_ == RuntimeComputeCallAction::kFailPublicationOnFirst && call == 1U) {
    return output.emit(999U);
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  if (action_ == RuntimeComputeCallAction::kThrowOnFirst && call == 1U) {
    throw std::runtime_error{"intentional compute-call completion failure"};
  }
#endif
  const bool action_due = action_ == RuntimeComputeCallAction::kRestartAfterComputeCalls
                              ? call == restart_after_successful_compute_calls_
                              : call == 1U;
  if (action_due) {
    switch (action_) {
      case RuntimeComputeCallAction::kRequestStopAfterFirst:
        context.request_stop();
        break;
      case RuntimeComputeCallAction::kRequestCompletionAfterFirst:
        context.request_completion();
        break;
      case RuntimeComputeCallAction::kRestartAfterFirst:
      case RuntimeComputeCallAction::kRestartAfterComputeCalls:
      case RuntimeComputeCallAction::kRestartRollbackAfterFirst: {
        auto receipt = context.request_restart();
        if (receipt) {
          RuntimeComputeCallChannelProbeState::restart_accepted.store(true,
                                                                      std::memory_order_release);
        } else {
          RuntimeComputeCallChannelProbeState::restart_error.store(receipt.error().code,
                                                                   std::memory_order_relaxed);
        }
        break;
      }
      case RuntimeComputeCallAction::kNone:
      case RuntimeComputeCallAction::kFailOnSecond:
      case RuntimeComputeCallAction::kFailPublicationOnFirst:
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      case RuntimeComputeCallAction::kThrowOnFirst:
#endif
        break;
    }
  }
  RuntimeProbeState::runtime_clock_ns.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::milliseconds{1}).count(),
      std::memory_order_relaxed);
  return {};
}

LifecycleStatus RuntimeComputeCallSourceOperator::start_stage(LifecycleContext& context) noexcept {
  if (context.transition_phase() == LifecycleTransitionPhase::kRestartForward) {
    RuntimeComputeCallChannelProbeState::successful_compute_calls_before_restart.store(
        RuntimeComputeCallChannelProbeState::source_count.load(std::memory_order_acquire),
        std::memory_order_release);
    RuntimeComputeCallChannelProbeState::restart_forward_start_count.fetch_add(
        1U, std::memory_order_relaxed);
    if (action_ == RuntimeComputeCallAction::kRestartRollbackAfterFirst) {
      return LifecycleStatus::kFatalFailure;
    }
  } else if (context.transition_phase() == LifecycleTransitionPhase::kRestoreOld) {
    RuntimeComputeCallChannelProbeState::restore_old_start_count.fetch_add(
        1U, std::memory_order_relaxed);
  }
  return LifecycleStatus::kOk;
}

RuntimeDormantComputeCallSourceOperator::RuntimeDormantComputeCallSourceOperator(
    const std::uint64_t limit) noexcept
    : limit_(limit) {}

void RuntimeDormantComputeCallSourceOperator::setup(OperatorSpec& spec) {
  spec.output(output, "output").max_emits_per_compute(1U);
  spec.notification_source(notification_, "notification")
      .capacity(1U)
      .sender_reference_capacity(1U);
  spec.lifecycle().stage(LifecycleStage::kArm, &RuntimeDormantComputeCallSourceOperator::arm);
}

[[nodiscard]] Contract RuntimeDormantComputeCallSourceOperator::contract() const {
  Contract result;
  result.trigger(OnNotified{.event = notification_}).complete_after(ComputeCalls{.limit = limit_});
  return result;
}

[[nodiscard]] expected<void, Error> RuntimeDormantComputeCallSourceOperator::compute(
    ExecutionContext&) {
  RuntimeComputeCallChannelProbeState::source_count.fetch_add(1U, std::memory_order_relaxed);
  return output.emit(1U);
}

LifecycleStatus RuntimeDormantComputeCallSourceOperator::arm(LifecycleContext& context) noexcept {
  auto sender = context.notification_sender(notification_);
  if (!sender) {
    return LifecycleStatus::kFatalFailure;
  }
  sender_ = std::move(*sender);
  return LifecycleStatus::kOk;
}

void RuntimeComputeCallTransformOperator::setup(OperatorSpec& spec) {
  spec.input(input, "input").queue_depth(1U);
  spec.output(output, "output").max_emits_per_compute(1U);
}

[[nodiscard]] Contract RuntimeComputeCallTransformOperator::contract() const {
  Contract result;
  result.trigger(OnEach{input});
  return result;
}

[[nodiscard]] expected<void, Error> RuntimeComputeCallTransformOperator::compute(
    ExecutionContext&) {
  RuntimeComputeCallChannelProbeState::transform_count.fetch_add(1U, std::memory_order_relaxed);
  auto sample = input.receive();
  if (!sample) {
    return make_unexpected(std::move(sample).error());
  }
  return output.emit(sample->data);
}

void RuntimeComputeCallSinkOperator::setup(OperatorSpec& spec) {
  spec.input(input, "input").queue_depth(1U);
}

[[nodiscard]] Contract RuntimeComputeCallSinkOperator::contract() const {
  Contract result;
  result.trigger(OnEach{input});
  return result;
}

[[nodiscard]] expected<void, Error> RuntimeComputeCallSinkOperator::compute(ExecutionContext&) {
  RuntimeComputeCallChannelProbeState::sink_compute_count.fetch_add(1U, std::memory_order_relaxed);
  auto sample = input.receive();
  if (!sample) {
    return make_unexpected(std::move(sample).error());
  }
  RuntimeComputeCallChannelProbeState::record_sink(sample->data);
  return {};
}

[[nodiscard]] expected<bool, Error> runtime_input_predicate_result(
    RuntimeInputPredicatePattern pattern, std::uint32_t evaluation, std::uint32_t driver_value,
    std::uint32_t minimum_driver_value) {
  if (driver_value < minimum_driver_value) {
    return false;
  }
  switch (pattern) {
    case RuntimeInputPredicatePattern::kAlwaysTrue:
      return true;
    case RuntimeInputPredicatePattern::kFalseTrueTrue:
      return evaluation != 1U;
    case RuntimeInputPredicatePattern::kTrueFalseTrue:
      return evaluation != 2U;
    case RuntimeInputPredicatePattern::kTrueThenError:
      if (evaluation == 2U) {
        return make_unexpected(Error{ErrorCode::kFailure, "input predicate failure"});
      }
      return true;
    case RuntimeInputPredicatePattern::kError:
      return make_unexpected(Error{ErrorCode::kFailure, "input predicate failure"});
  }
  return make_unexpected(Error{ErrorCode::kProtocolError});
}

[[nodiscard]] ExecutionPlan make_runtime_test_plan(bool reverse_authoring) {
  return make_runtime_test_plan_with_clock<RuntimeProbeClock>(reverse_authoring);
}

[[nodiscard]] ExecutionPlan make_runtime_stop_dependency_test_plan() {
  Graph graph{"runtime-stop-dependency-test"};
  const auto first = graph.op<RuntimeProbeOperator>("a-first", 1);
  const auto second = graph.op<RuntimeProbeOperator>("z-last", 2);
  graph.partition("first-runtime-spin-worker").add(first);
  graph.partition("second-runtime-spin-worker").add(second);
  graph.lifecycle().before(first, second, LifecycleStage::kStop);
  const ClockKey default_clock = graph.add_clock<RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.deployment
      .bind_thread_placement("first-runtime-spin-worker",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("second-runtime-spin-worker",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return compile(graph, std::move(options));
}

[[nodiscard]] ExecutionPlan make_runtime_param_test_plan() {
  Graph graph{"runtime-param-test"};
  const auto parameterized = graph.op<RuntimeParamProbeOperator>("parameterized");
  graph.partition("runtime-spin-worker").add(parameterized);
  const ClockKey default_clock = graph.add_clock<RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  options.deployment.bind_thread_placement("runtime-spin-worker",
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return compile(graph, std::move(options));
}

[[nodiscard]] ExecutionPlan make_runtime_deadline_param_test_plan() {
  Graph graph{"runtime-deadline-param-test"};
  const auto parameterized = graph.op<RuntimeDeadlineParamProbeOperator>("parameterized-deadline");
  graph.partition("runtime-spin-worker").add(parameterized);
  const ClockKey default_clock = graph.add_clock<RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  options.deployment.bind_thread_placement("runtime-spin-worker",
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return compile(graph, std::move(options));
}

[[nodiscard]] ExecutionPlan make_runtime_two_deadline_param_test_plan() {
  Graph graph{"runtime-two-deadline-param-test"};
  const auto first = graph.op<RuntimeDeadlineParamProbeOperator>("a-parameterized-deadline");
  const auto second = graph.op<RuntimeDeadlineParamProbeOperator>("z-parameterized-deadline");
  graph.partition("first-runtime-spin-worker").add(first);
  graph.partition("second-runtime-spin-worker").add(second);
  const ClockKey default_clock = graph.add_clock<RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  options.deployment
      .bind_thread_placement("first-runtime-spin-worker",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("second-runtime-spin-worker",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return compile(graph, std::move(options));
}

[[nodiscard]] ExecutionPlan make_runtime_role_test_plan() {
  Graph graph{"runtime-role-test"};
  const auto worker = graph.op<RuntimeProbeOperator>("operator", 1, true);
  graph.partition("runtime-worker").add(worker);
  const ClockKey default_clock = graph.add_clock<RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.deployment.bind_thread_placement("runtime-worker",
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return compile(graph, options);
}

[[nodiscard]] ExecutionPlan make_runtime_clock_test_plan() {
  Graph graph{"runtime-clock-test"};
  const auto periodic = graph.op<RuntimeClockProbeOperator>("periodic");
  graph.partition("runtime-clock-worker").add(periodic);
  const ClockKey default_clock = graph.add_clock<RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.deployment.bind_thread_placement("runtime-clock-worker",
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return compile(graph, std::move(options));
}

[[nodiscard]] ExecutionPlan make_runtime_notification_test_plan() {
  // Sleeping wait-set placement exercises the foreign-thread wake path used by production senders.
  Graph graph{"runtime-notification-test"};
  const auto notified = graph.op<RuntimeNotificationProbeOperator>("notified");
  graph.partition("runtime-notification-worker").add(notified);
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-realtime-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.deployment.bind_thread_placement(
      "runtime-notification-worker", ThreadPlacement{.wake_mode = WakeMode::kSleepingWaitset});
  return compile(graph, std::move(options));
}

[[nodiscard]] ExecutionPlan make_runtime_clock_predicate_error_plan() {
  Graph graph{"runtime-clock-predicate-error"};
  const auto predicate = graph.op<RuntimeClockPredicateErrorOperator>("predicate");
  graph.partition("runtime-clock-worker").add(predicate);
  const ClockKey default_clock = graph.add_clock<RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.deployment.bind_thread_placement("runtime-clock-worker",
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return compile(graph, std::move(options));
}

[[nodiscard]] ExecutionPlan make_runtime_clock_predicate_plan(RuntimeClockPredicatePattern pattern,
                                                              PredicateFiring firing,
                                                              bool restart_after_first_compute) {
  Graph graph{"runtime-clock-predicate-success"};
  const auto predicate = graph.op<RuntimeClockPredicateProbeOperator>(
      "predicate", pattern, firing, restart_after_first_compute);
  graph.partition("runtime-clock-worker").add(predicate);
  if (restart_after_first_compute) {
    graph.lifecycle().allow_self_restart(predicate);
  }
  const ClockKey default_clock = graph.add_clock<RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.deployment.bind_thread_placement("runtime-clock-worker",
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  if (restart_after_first_compute) {
    options.lifecycle_transactions = LifecycleTransactionOptions{
        .max_pending_requests = 64U,
        .max_pending_transactions = 64U,
        .max_receipts = 128U,
        .max_fenced_operators = 64U,
        .max_candidate_generations = 64U,
        .max_events = 1024U,
        .max_audit_records = 128U,
        .max_quiesce_records = 256U,
        .max_rollback_records = 256U,
    };
  }
  return compile(graph, std::move(options));
}

[[nodiscard]] ExecutionPlan make_runtime_mixed_source_test_plan() {
  Graph graph{"runtime-mixed-source-test"};
  const auto physical = graph.op<RuntimeProbeOperator>("physical", 1);
  const auto periodic = graph.op<RuntimeClockProbeOperator>("periodic");
  const auto predicate =
      graph.op<RuntimeClockPredicateProbeOperator>("predicate",
                                                   RuntimeClockPredicatePattern::kAlwaysTrue,
                                                   PredicateFiring::kOnEveryTrueEvaluation,
                                                   false);
  graph.partition("runtime-spin-worker").add(physical);
  graph.partition("runtime-clock-worker-periodic").add(periodic);
  graph.partition("runtime-clock-worker-predicate").add(predicate);
  const ClockKey default_clock = graph.add_clock<RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.deployment
      .bind_thread_placement("runtime-spin-worker",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("runtime-clock-worker-periodic",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("runtime-clock-worker-predicate",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return compile(graph, std::move(options));
}

[[nodiscard]] ExecutionPlan make_runtime_input_predicate_replay_plan() {
  RuntimeInputContractPredicateOperator::mode.store(RuntimeInputContractReplayMode::kMatch,
                                                    std::memory_order_relaxed);
  Graph graph{"runtime-input-predicate-replay"};
  auto source = graph.op<RuntimeInputContractSourceOperator>("source");
  auto predicate = graph.op<RuntimeInputContractPredicateOperator>("predicate");
  graph.add_flow(source->driver, predicate->driver, ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(source->additional, predicate->additional, ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(
      source->compute_only, predicate->compute_only, ConnectionOptions{.queue_depth = 8U});
  const ClockKey default_clock = graph.add_clock<RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

[[nodiscard]] ExecutionPlan make_runtime_clock_input_replay_plan(RuntimeClockInputReplayKind kind,
                                                                 Profile profile) {
  RuntimeClockInputReplayOperator::mismatch_ordinal.store(0U, std::memory_order_relaxed);
  Graph graph{"runtime-clock-input-replay"};
  auto source = graph.op<RuntimeInputContractSourceOperator>("source");
  auto sink = graph.op<RuntimeClockInputReplayOperator>("sink", kind);
  graph.partition("runtime-input-source").add(source);
  graph.partition("runtime-clock-input-sink").add(sink);
  graph.add_flow(source->driver, sink->predicate_input, ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(source->compute_only, sink->compute_only, ConnectionOptions{.queue_depth = 8U});
  const ClockKey default_clock = graph.add_clock<RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.profile = profile;
  return compile(graph, std::move(options));
}

[[nodiscard]] ExecutionPlan make_runtime_dual_clock_input_replay_plan() {
  RuntimeClockInputReplayOperator::mismatch_ordinal.store(0U, std::memory_order_relaxed);
  Graph graph{"runtime-dual-clock-input-replay"};
  auto source = graph.op<RuntimeInputContractSourceOperator>("source");
  auto clock = graph.op<RuntimeClockInputReplayOperator>(
      "ordinary-clock", RuntimeClockInputReplayKind::kClock, 1U);
  auto predicate = graph.op<RuntimeClockInputReplayOperator>(
      "predicate-clock", RuntimeClockInputReplayKind::kPredicate, 2U);
  graph.partition("runtime-input-source").add(source);
  graph.partition("runtime-ordinary-clock").add(clock);
  graph.partition("runtime-predicate-clock").add(predicate);
  graph.add_flow(source->driver, clock->predicate_input, ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(source->compute_only, clock->compute_only, ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(source->driver, predicate->predicate_input, ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(
      source->compute_only, predicate->compute_only, ConnectionOptions{.queue_depth = 8U});
  const ClockKey default_clock = graph.add_clock<RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

[[nodiscard]] ExecutionPlan make_runtime_input_predicate_channel_plan(
    RuntimeInputPredicatePattern pattern, PredicateFiring firing, bool configured,
    RuntimeInputPredicateMissing missing) {
  Graph graph{"runtime-input-predicate-channel"};
  auto source = graph.op<RuntimeInputPredicateChannelSourceOperator>("source", missing);
  auto predicate = graph.op<RuntimeInputPredicateChannelSinkOperator>(
      "predicate", pattern, firing, configured, missing);
  graph.add_flow(source->driver, predicate->driver, ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(source->additional, predicate->additional, ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(
      source->compute_only, predicate->compute_only, ConnectionOptions{.queue_depth = 8U});
  const ClockKey default_clock = graph.add_clock<RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

[[nodiscard]] ExecutionPlan make_runtime_compute_call_channel_plan(
    const std::uint64_t limit, const RuntimeComputeCallAction action,
    const std::uint64_t restart_after_successful_compute_calls) {
  Graph graph{unique_runtime_compute_call_graph_name()};
  const auto source = graph.op<RuntimeComputeCallSourceOperator>(
      "source", limit, action, restart_after_successful_compute_calls);
  const auto first = graph.op<RuntimeComputeCallSinkOperator>("first");
  const auto second = graph.op<RuntimeComputeCallSinkOperator>("second");
  graph.partition("runtime-compute-call-source").add(source);
  graph.partition("runtime-compute-call-first").add(first);
  graph.partition("runtime-compute-call-second").add(second);
  graph.add_flow(source->output, first->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(source->output, second->input, ConnectionOptions{.queue_depth = 1U});
  if (action == RuntimeComputeCallAction::kRestartAfterFirst ||
      action == RuntimeComputeCallAction::kRestartAfterComputeCalls ||
      action == RuntimeComputeCallAction::kRestartRollbackAfterFirst) {
    graph.lifecycle().allow_self_restart(source);
  }
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  if (action == RuntimeComputeCallAction::kRestartAfterFirst ||
      action == RuntimeComputeCallAction::kRestartAfterComputeCalls ||
      action == RuntimeComputeCallAction::kRestartRollbackAfterFirst) {
    options.lifecycle_transactions = LifecycleTransactionOptions{
        .max_pending_requests = 64U,
        .max_pending_transactions = 64U,
        .max_receipts = 128U,
        .max_fenced_operators = 64U,
        .max_candidate_generations = 64U,
        .max_events = 1024U,
        .max_audit_records = 128U,
        .max_quiesce_records = 256U,
        .max_rollback_records = 256U,
    };
  }
  const ExecutionPlan compiled = compile(graph, std::move(options));
  return compiled;
}

[[nodiscard]] ExecutionPlan make_runtime_compute_call_chain_plan(const std::uint64_t limit) {
  Graph graph{unique_runtime_compute_call_graph_name()};
  const auto source =
      graph.op<RuntimeComputeCallSourceOperator>("source", limit, RuntimeComputeCallAction::kNone);
  const auto transform = graph.op<RuntimeComputeCallTransformOperator>("transform");
  const auto sink = graph.op<RuntimeComputeCallSinkOperator>("sink");
  graph.partition("runtime-compute-call-chain-source").add(source);
  graph.partition("runtime-compute-call-chain-transform").add(transform);
  graph.partition("runtime-compute-call-chain-sink").add(sink);
  graph.add_flow(source->output, transform->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(transform->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);

  return compile(graph);
}

[[nodiscard]] ExecutionPlan make_runtime_dormant_compute_call_plan(const std::uint64_t limit) {
  Graph graph{"runtime-dormant-compute-call-completion"};
  const auto source = graph.op<RuntimeDormantComputeCallSourceOperator>("source", limit);
  graph.partition("runtime-dormant-compute-call-source").add(source);
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.deployment.bind_thread_placement(
      "runtime-dormant-compute-call-source",
      ThreadPlacement{.wake_mode = WakeMode::kSleepingWaitset});
  return compile(graph, std::move(options));
}

[[nodiscard]] ExecutionPlan make_runtime_input_trigger_replay_plan(
    RuntimeInputTriggerReplayKind kind) {
  RuntimeInputTriggerReplayOperator::kind.store(kind, std::memory_order_relaxed);
  Graph graph{"runtime-input-trigger-replay"};
  auto source = graph.op<RuntimeInputContractSourceOperator>("source");
  auto sink = graph.op<RuntimeInputTriggerReplayOperator>("sink");
  graph.add_flow(source->driver, sink->first, ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(source->additional, sink->second, ConnectionOptions{.queue_depth = 8U});
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

[[nodiscard]] ExecutionPlan make_runtime_aligned_channel_plan(bool complete_manual_cover) {
  Graph graph{"runtime-aligned-channel"};
  auto source = graph.op<RuntimeAlignedChannelSourceOperator>("source");
  auto sink = graph.op<RuntimeAlignedChannelSinkOperator>("sink");
  graph.add_flow(source->first, sink->first, ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(source->second, sink->second, ConnectionOptions{.queue_depth = 8U});
  if (complete_manual_cover) {
    graph.partition("runtime-aligned-channel").include_all();
  }
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

[[nodiscard]] ExecutionPlan make_runtime_hybrid_aligned_channel_plan() {
  Graph graph{"runtime-hybrid-aligned-channel"};
  auto source = graph.op<RuntimeAlignedChannelSourceOperator>("source");
  auto sink = graph.op<RuntimeAlignedChannelSinkOperator>("sink");
  graph.add_flow(source->first, sink->first, ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(source->second, sink->second, ConnectionOptions{.queue_depth = 8U});
  graph.partition("runtime-aligned-anchor").add(source);
  graph.auto_partition_remaining();
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.deployment.bind_thread_placement("runtime-aligned-anchor",
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return compile(graph, std::move(options));
}

[[nodiscard]] ExecutionPlan make_runtime_mixed_channel_unconnected_port_plan() {
  Graph graph{"runtime-mixed-channel-unconnected-port"};
  auto source = graph.op<RuntimeAlignedChannelSourceOperator>("source");
  auto sink = graph.op<RuntimeAlignedChannelSinkOperator>("sink");
  const auto independent = graph.op<RuntimeUnconnectedPortClockOperator>("independent");
  graph.add_flow(source->first, sink->first, ConnectionOptions{.queue_depth = 8U});
  graph.add_flow(source->second, sink->second, ConnectionOptions{.queue_depth = 8U});
  graph.partition("channel-flow").add(source).add(sink);
  graph.partition("independent").add(independent);
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

[[nodiscard]] ExecutionPlan make_runtime_activation_cut_order_plan() {
  Graph graph{"runtime-activation-cut-order"};
  auto source = graph.op<RuntimeActivationCutSourceOperator>("source");
  auto later = graph.op<RuntimeActivationCutSinkOperator>("a-later");
  auto later_copy = graph.op<RuntimeActivationCutSinkOperator>("b-later-copy");
  auto earlier = graph.op<RuntimeActivationCutSinkOperator>("z-earlier");
  auto earlier_copy = graph.op<RuntimeActivationCutSinkOperator>("y-earlier-copy");
  // Edge declaration order deliberately opposes capture-time order. The process-local subscriber
  // scan sees both `later` recipients before both `earlier` recipients, while global activation
  // order must still choose t+5 first.
  graph.add_flow(source->later, later->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(source->later, later_copy->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(source->earlier, earlier->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(source->earlier, earlier_copy->input, ConnectionOptions{.queue_depth = 1U});
  graph.partition("runtime-activation-cut-order").include_all();
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

[[nodiscard]] ExecutionPlan make_runtime_window_channel_plan(bool separate_execution_partitions) {
  Graph graph{"runtime-window-channel"};
  auto source = graph.op<RuntimeWindowChannelSourceOperator>("source");
  auto sink = graph.op<RuntimeWindowChannelSinkOperator>("sink");
  const bool terminal_drain =
      RuntimeWindowChannelProbeState::script.load(std::memory_order_relaxed) ==
      RuntimeWindowChannelScript::kEndOfStreamDrain;
  if (terminal_drain) {
    auto relay = graph.op<RuntimeWindowChannelRelayOperator>("relay");
    auto drain = graph.op<RuntimeWindowChannelDrainOperator>("drain");
    if (separate_execution_partitions) {
      graph.partition("runtime-window-source").add(source);
      graph.partition("runtime-window-relay").add(relay);
      graph.partition("runtime-window-sink").add(sink);
      graph.partition("runtime-window-drain").add(drain);
    }
    graph.add_flow(source->anchor, relay->input, ConnectionOptions{.queue_depth = 2U});
    graph.add_flow(relay->anchor, sink->anchor, ConnectionOptions{.queue_depth = 8U});
    graph.add_flow(relay->windowed, sink->windowed, ConnectionOptions{.queue_depth = 16U});
    graph.add_flow(sink->output, drain->input, ConnectionOptions{.queue_depth = 2U});
  } else {
    if (separate_execution_partitions) {
      graph.partition("runtime-window-source").add(source);
      graph.partition("runtime-window-sink").add(sink);
    }
    graph.add_flow(source->anchor, sink->anchor, ConnectionOptions{.queue_depth = 8U});
    graph.add_flow(source->windowed, sink->windowed, ConnectionOptions{.queue_depth = 16U});
  }
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  return compile(graph);
}

[[nodiscard]] ExecutionPlan make_runtime_multi_lane_test_plan() {
  Graph graph{"runtime-multi-lane-test"};
  const auto first = graph.op<RuntimeProbeOperator>("a-first", 1, false);
  const auto last = graph.op<RuntimeProbeOperator>("z-last", 2, true);
  graph.partition("runtime-spin-worker").add(first);
  graph.partition("runtime-worker").add(last);
  const ClockKey default_clock = graph.add_clock<RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.deployment
      .bind_thread_placement("runtime-spin-worker",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("runtime-worker", ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return compile(graph, options);
}

[[nodiscard]] ExecutionPlan make_runtime_two_param_test_plan() {
  Graph graph{"runtime-two-param-test"};
  const auto first = graph.op<RuntimeParamProbeOperator>("a-parameterized");
  const auto second = graph.op<RuntimeParamProbeOperator>("z-parameterized");
  graph.partition("first-runtime-spin-worker").add(first);
  graph.partition("second-runtime-spin-worker").add(second);
  const ClockKey default_clock = graph.add_clock<RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  options.deployment
      .bind_thread_placement("first-runtime-spin-worker",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("second-runtime-spin-worker",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  return compile(graph, std::move(options));
}

[[nodiscard]] ExecutionPlan make_event_policy_param_test_plan() {
  Graph graph{"event-policy-param-test"};
  (void)graph.op<EventPolicyParamOperator>("parameterized");
  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  return compile(graph, options);
}

[[nodiscard]] ExecutionPlan make_runtime_duration_param_test_plan() {
  Graph graph{"runtime-duration-param-test"};
  (void)graph.op<RuntimeDurationParamProbeOperator>("parameterized");
  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  return compile(graph, options);
}

[[nodiscard]] ExecutionPlan make_runtime_validation_param_test_plan() {
  Graph graph{"runtime-validation-param-test"};
  (void)graph.op<RuntimeValidationParamProbeOperator>("parameterized");
  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  return compile(graph, options);
}

[[nodiscard]] ExecutionPlan make_runtime_float_param_test_plan() {
  Graph graph{"runtime-float-param-test"};
  (void)graph.op<RuntimeFloatParamProbeOperator>("parameterized");
  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  return compile(graph, options);
}

[[nodiscard]] ExecutionPlan make_lifecycle_param_test_plan() {
  Graph graph{"lifecycle-param-test"};
  const auto parameterized = graph.op<LifecycleParamProbeOperator>("parameterized");
  graph.partition("runtime-spin-worker").add(parameterized);
  const ClockKey default_clock = graph.add_clock<RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  options.deployment.bind_thread_placement("runtime-spin-worker",
                                           ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  options.lifecycle_transactions = LifecycleTransactionOptions{
      .max_pending_requests = 4U,
      .max_pending_transactions = 4U,
      .max_receipts = 8U,
      .max_fenced_operators = 16U,
      .max_candidate_generations = 16U,
      .max_events = 64U,
      .max_audit_records = 32U,
      .max_quiesce_records = 32U,
      .max_rollback_records = 64U,
  };
  return compile(graph, options);
}

[[nodiscard]] ExecutionPlan make_mixed_mutability_param_test_plan() {
  Graph graph{"mixed-mutability-param-test"};
  (void)graph.op<MixedMutabilityParamProbeOperator>("parameterized");
  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  options.lifecycle_transactions = LifecycleTransactionOptions{
      .max_pending_requests = 4U,
      .max_pending_transactions = 4U,
      .max_receipts = 8U,
      .max_fenced_operators = 16U,
      .max_candidate_generations = 16U,
      .max_events = 64U,
      .max_audit_records = 32U,
      .max_quiesce_records = 32U,
      .max_rollback_records = 64U,
  };
  return compile(graph, options);
}

[[nodiscard]] ExecutionPlan make_two_lifecycle_param_test_plan() {
  Graph graph{"two-lifecycle-param-test"};
  const auto first = graph.op<LifecycleParamProbeOperator>("a-parameterized", 1);
  const auto second = graph.op<LifecycleParamProbeOperator>("z-parameterized", 2);
  graph.partition("first-runtime-spin-worker").add(first);
  graph.partition("second-runtime-spin-worker").add(second);
  const ClockKey default_clock = graph.add_clock<RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  options.deployment
      .bind_thread_placement("first-runtime-spin-worker",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("second-runtime-spin-worker",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  options.lifecycle_transactions = LifecycleTransactionOptions{
      .max_pending_requests = 16U,
      .max_pending_transactions = 8U,
      .max_receipts = 32U,
      .max_fenced_operators = 16U,
      .max_candidate_generations = 16U,
      .max_events = 256U,
      .max_audit_records = 32U,
      .max_quiesce_records = 32U,
      .max_rollback_records = 64U,
  };
  return compile(graph, options);
}

[[nodiscard]] ExecutionPlan make_three_lifecycle_param_test_plan() {
  Graph graph{"three-lifecycle-param-test"};
  const auto first = graph.op<LifecycleParamProbeOperator>("a-parameterized", 1);
  const auto middle = graph.op<LifecycleParamProbeOperator>("m-parameterized", 2);
  const auto last = graph.op<LifecycleParamProbeOperator>("z-parameterized", 3);
  graph.partition("first-runtime-spin-worker").add(first);
  graph.partition("middle-runtime-spin-worker").add(middle);
  graph.partition("last-runtime-spin-worker").add(last);
  const ClockKey default_clock = graph.add_clock<RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  options.deployment
      .bind_thread_placement("first-runtime-spin-worker",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("middle-runtime-spin-worker",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("last-runtime-spin-worker",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  options.lifecycle_transactions = LifecycleTransactionOptions{
      .max_pending_requests = 16U,
      .max_pending_transactions = 16U,
      .max_receipts = 32U,
      .max_fenced_operators = 16U,
      .max_candidate_generations = 16U,
      .max_events = 256U,
      .max_audit_records = 32U,
      .max_quiesce_records = 32U,
      .max_rollback_records = 64U,
  };
  return compile(graph, options);
}

[[nodiscard]] ExecutionPlan make_lifecycle_param_dependency_test_plan() {
  Graph graph{"lifecycle-param-dependency-test"};
  const auto first = graph.op<LifecycleParamProbeOperator>("a-parameterized", 1);
  const auto second = graph.op<LifecycleParamProbeOperator>("z-parameterized", 2);
  graph.partition("first-runtime-spin-worker").add(first);
  graph.partition("second-runtime-spin-worker").add(second);
  graph.lifecycle().before(first, second, LifecycleStage::kAllocate);
  const ClockKey default_clock = graph.add_clock<RuntimeProbeClock>("runtime-test-clock");
  graph.set_default_clock(default_clock);
  CompileOptions options;
  options.profile = Profile::kStandard;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kEnabledIfAdmitted;
  options.deployment
      .bind_thread_placement("first-runtime-spin-worker",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("second-runtime-spin-worker",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  options.lifecycle_transactions = LifecycleTransactionOptions{
      .max_pending_requests = 8U,
      .max_pending_transactions = 8U,
      .max_receipts = 16U,
      .max_fenced_operators = 16U,
      .max_candidate_generations = 16U,
      .max_events = 128U,
      .max_audit_records = 32U,
      .max_quiesce_records = 32U,
      .max_rollback_records = 64U,
  };
  return compile(graph, options);
}

[[nodiscard]] bool has_runtime_diagnostic(const RunSession& session, std::string_view token) {
  return std::any_of(session.diagnostics().begin(),
                     session.diagnostics().end(),
                     [&](const Diagnostic& diagnostic) { return diagnostic.token == token; });
}

}  // namespace holoscan::test
