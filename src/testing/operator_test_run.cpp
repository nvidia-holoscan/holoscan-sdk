/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "operator_test_run.hpp"

namespace holoscan::testing {
namespace detail {

RunningFixtureState::RunningFixtureState(std::shared_ptr<AttemptGate> gate)
    : lease(std::make_unique<ActiveAttempt>(std::move(gate))) {}

RunningFixtureState::~RunningFixtureState() noexcept {
  close_frame_driver();
  if (session)
    session->request_stop();
  watchdog.request_stop();
  if (watchdog.joinable())
    watchdog.join();
  // Native RAII joins even when collection was never requested or a cold exception escaped.
  // Release the builder lease only after every callback has retired.
  session.reset();
}

void RunningFixtureState::start_watchdog(std::chrono::milliseconds timeout) {
  if (!session)
    throw std::logic_error{"operator test watchdog requires an active session"};
  auto& native = *session;
  watchdog = std::jthread{[this, &native, timeout](const std::stop_token& stop) {
    std::unique_lock lock(watchdog_mutex);
    watchdog_changed.wait_for(lock, stop, timeout, [] { return false; });
    if (!stop.stop_requested() && native.is_running()) {
      watchdog_expired.store(true, std::memory_order_relaxed);
      native.request_stop();
    }
  }};
}

void RunningFixtureState::close_frame_driver() noexcept {
  if (frames) {
    const std::lock_guard lock(frames->mutex);
    frames->driver_closed = true;
  }
}

void RunningFixtureState::complete_cold() noexcept {
  close_frame_driver();
  joined = true;
  lease.reset();
}

void RunningFixtureState::finish() {
  if (joined)
    return;
  if (!session)
    throw std::logic_error{"operator test active attempt has no session"};
  auto& native = *session;
  native.wait();
  close_frame_driver();
  watchdog.request_stop();
  watchdog.join();
  result->watchdog_expired = watchdog_expired.load(std::memory_order_relaxed);
  result->termination = native.termination();
  const auto diagnostics = native.diagnostics();
  try {
    result->diagnostics.insert(result->diagnostics.end(), diagnostics.begin(), diagnostics.end());
  } catch (const std::bad_alloc&) {
    result->diagnostics_copy_failed = true;
  }
  if (captures.size() == 1U && !captures.front().first->failure) {
    result->observed_completion =
        retained_boundary_completed(captures.front().first->retained, native);
  }
  // Destinations were allocated cold. Collection moves complete evidence without payload copies.
  for (auto& [live, detached] : captures) {
    live->transfer_to(*detached);
    live->progress.reset();
  }
  joined = true;
  lease.reset();
}

void capture_recorded(CaptureProgress& progress) noexcept {
  const std::lock_guard lock(progress.mutex);
  ++progress.count;
  progress.changed.notify_all();
}

}  // namespace detail

RunningFixture::RunningFixture(std::unique_ptr<detail::RunningFixtureState> state)
    : state_(std::move(state)) {}
RunningFixture::~RunningFixture() noexcept = default;
RunningFixture::RunningFixture(RunningFixture&&) noexcept = default;

TestResult RunningFixture::finish() {
  if (!state_)
    throw std::logic_error{"operator test running owner was moved from"};
  state_->finish();
  return TestResult{state_->result};
}

expected<void, Error> RunningFixture::wait_admission() const noexcept {
  if (!state_) {
    return {};
  }
  const auto& session = state_->session;
  if (session.has_value()) {
    return session->wait_admission();
  }
  return {};
}

void RunningFixture::request_stop() noexcept {
  if (!state_)
    return;
  auto& session = state_->session;
  if (session)
    session->request_stop();
  if (state_->frames) {
    const std::lock_guard lock(state_->frames->mutex);
    // Do not mark driver_closed: initial Arm distinguishes a live owner from plan replay.
    state_->frames->stop_requested = true;
  }
}

bool RunningFixture::is_running() const noexcept {
  if (!state_)
    return false;
  const auto& session = state_->session;
  return session && session->is_running();
}

RunTermination RunningFixture::termination() const noexcept {
  if (!state_)
    return RunTermination::kFailed;
  const auto& session = state_->session;
  return session ? session->termination() : RunTermination::kFailed;
}

namespace {
Error control_error(ErrorCode code, std::string_view message) noexcept {
  Error result{.code = code};
  try {
    result.message.assign(message);
  } catch (...) {
    // Preserve the native code if optional explanatory text cannot be allocated.
    result.message.clear();
  }
  return result;
}

constexpr std::string_view kRejectedAttemptMessage{
    "operator test attempt rejected before start; call finish() for diagnostics"};

expected<void, Error> release_frame_attempt(
    const detail::RunningFixtureState* state,
    std::optional<std::chrono::steady_clock::time_point> deadline = {}) noexcept {
  if (state == nullptr)
    return make_unexpected(
        control_error(ErrorCode::kInvalidArgument, "operator test running owner was moved from"));
  if (!state->session)
    return make_unexpected(control_error(ErrorCode::kClosed, kRejectedAttemptMessage));
  if (!state->frames)
    return make_unexpected(control_error(ErrorCode::kInvalidArgument,
                                         "release_frame requires a manual grouped-input attempt"));
  if (!state->session->is_running())
    return make_unexpected(control_error(
        ErrorCode::kClosed, "operator test attempt has stopped; call finish() for its result"));
  auto posted = state->frames->post(deadline);
  if (!posted && posted.error().message.empty()) {
    const auto code = posted.error().code;
    if (code == ErrorCode::kTimeout)
      return make_unexpected(control_error(
          code, "operator test frame release timed out before notification admission"));
    if (code == ErrorCode::kClosed)
      return make_unexpected(control_error(code, "manual frame driver is closed"));
    if (code == ErrorCode::kNotReady)
      return make_unexpected(control_error(code, "operator test manual frame source is not ready"));
  }
  return posted;
}
}  // namespace

expected<void, Error> RunningFixture::release_frame() const noexcept {
  return release_frame_attempt(state_.get());
}

expected<void, Error> RunningFixture::release_frame(std::chrono::milliseconds timeout) const {
  const auto now = std::chrono::steady_clock::now();
  const auto available = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::time_point::max() - now);
  if (timeout.count() < 0 || timeout > available) {
    return make_unexpected(
        control_error(ErrorCode::kInvalidArgument,
                      "operator test release timeout must be nonnegative and representable"));
  }
  const auto deadline = now + timeout;
  if (timeout.count() == 0)
    return release_frame();
  auto posted = release_frame_attempt(state_.get(), deadline);
  while (!posted && (posted.error().code == ErrorCode::kNotReady ||
                     posted.error().code == ErrorCode::kBackpressured)) {
    const auto current = std::chrono::steady_clock::now();
    if (current >= deadline)
      return make_unexpected(
          control_error(ErrorCode::kTimeout,
                        "operator test frame release timed out before notification admission"));
    std::this_thread::sleep_until(
        current + std::min(deadline - current,
                           std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                               std::chrono::milliseconds{1})));
    // The final sleep may consume the remaining budget. Never post a retry after the deadline.
    if (std::chrono::steady_clock::now() >= deadline)
      return make_unexpected(
          control_error(ErrorCode::kTimeout,
                        "operator test frame release timed out before notification admission"));
    posted = release_frame_attempt(state_.get(), deadline);
  }
  return posted;
}

expected<OperatorRef, Error> RunningFixture::operator_ref() const noexcept {
  return operator_ref(state_ ? state_->tested_operator : OperatorRef{});
}

expected<OperatorRef, Error> RunningFixture::operator_ref(OperatorRef plan_ref) const noexcept {
  if (!state_)
    return make_unexpected(
        control_error(ErrorCode::kStaleOperatorRef, "operator test running owner was moved from"));
  const auto& session = state_->session;
  if (!session)
    return make_unexpected(control_error(ErrorCode::kStaleOperatorRef, kRejectedAttemptMessage));
  return session->operator_ref(plan_ref);
}

expected<LifecycleControlHandle, Error> RunningFixture::lifecycle_control(
    const LifecycleAccessPolicy& policy) {
  if (!state_)
    return make_unexpected(
        control_error(ErrorCode::kClosed, "operator test running owner was moved from"));
  auto& session = state_->session;
  if (!session)
    return make_unexpected(control_error(ErrorCode::kClosed, kRejectedAttemptMessage));
  return session->lifecycle_control(policy);
}

namespace {
detail::CaptureProgress& require_progress(const detail::RunningFixtureState* state,
                                          std::string_view name) {
  if (state == nullptr)
    throw std::logic_error{"operator test running owner was moved from"};
  const auto found = state->progress.find(name);
  if (found == state->progress.end())
    throw std::invalid_argument{"operator test output '" + std::string(name) + "' has no capture"};
  return *found->second;
}
}  // namespace

std::size_t RunningFixture::output_count(std::string_view name) const {
  auto& progress = require_progress(state_.get(), name);
  const std::lock_guard lock(progress.mutex);
  return progress.count;
}

bool RunningFixture::wait_for_output(std::string_view name, std::size_t minimum,
                                     std::chrono::milliseconds timeout) const {
  auto& progress = require_progress(state_.get(), name);
  const auto now = std::chrono::steady_clock::now();
  const auto available = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::time_point::max() - now);
  if (timeout.count() < 0 || timeout > available)
    throw std::invalid_argument{"operator test observation timeout is not representable"};
  const auto deadline = now + timeout;
  std::unique_lock lock(progress.mutex);
  while (progress.count < minimum) {
    if (!is_running() || std::chrono::steady_clock::now() >= deadline)
      return false;
    // Native termination has no observer callback. Bound the recheck independently of data.
    progress.changed.wait_until(
        lock, std::min(deadline, std::chrono::steady_clock::now() + std::chrono::milliseconds{1}));
  }
  return true;
}

}  // namespace holoscan::testing
