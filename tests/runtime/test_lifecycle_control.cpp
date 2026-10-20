/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <future>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/lifecycle_control.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_catalog.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "common/allocation_counter.hpp"
#include "runtime/lifecycle_control_state.hpp"
#include "runtime/runtime_callback_context.hpp"
#include "runtime/runtime_diagnostic_sink.hpp"
#include "runtime/runtime_test_access.hpp"
#include "support/direct_port_binding.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

class RecordingLifecycleDiagnosticSink final : public detail::RuntimeStructuralDiagnosticSink {
 public:
  explicit RecordingLifecycleDiagnosticSink(bool fail = false) : fail_(fail) {}

  void publish(std::span<const Diagnostic>) override {}

  void publish_lifecycle_receipts(
      std::span<const detail::RuntimeLifecycleReceiptDiagnostic> diagnostics) override {
    if (fail_) {
      throw std::runtime_error("injected lifecycle diagnostic failure");
    }
    const std::lock_guard lock(mutex_);
    records_.assign(diagnostics.begin(), diagnostics.end());
  }

  [[nodiscard]] std::vector<detail::RuntimeLifecycleReceiptDiagnostic> snapshot() const {
    const std::lock_guard lock(mutex_);
    return records_;
  }

 private:
  bool fail_{};
  mutable std::mutex mutex_;
  std::vector<detail::RuntimeLifecycleReceiptDiagnostic> records_;
};

struct BlockingLifecycleWake {
  static void signal(void* state) noexcept {
    auto& wake = *static_cast<BlockingLifecycleWake*>(state);
    std::unique_lock lock(wake.mutex);
    wake.entered = true;
    wake.cv.notify_all();
    wake.cv.wait(lock, [&] { return wake.released; });
  }

  [[nodiscard]] bool wait_until_entered() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [&] { return entered; });
  }

  void release() {
    const std::lock_guard lock(mutex);
    released = true;
    cv.notify_all();
  }

  std::mutex mutex;
  std::condition_variable cv;
  bool entered{};
  bool released{};
};

constexpr std::array kForwardStages{
    LifecycleStage::kConfigure,
    LifecycleStage::kDiscover,
    LifecycleStage::kAllocate,
    LifecycleStage::kWarmUp,
    LifecycleStage::kArm,
    LifecycleStage::kStart,
};

template <typename ContextT>
concept HasLifecycleControlRequest =
    requires(ContextT& context, OperatorRef target) { context.request_restart(target); };

template <typename ContextT>
concept HasLifecycleControlAwait = requires(ContextT& context, LifecycleRestartReceipt receipt) {
  context.await(receipt, std::chrono::nanoseconds{0});
};

static_assert(!HasLifecycleControlRequest<LifecycleContext>);
static_assert(!HasLifecycleControlAwait<LifecycleContext>);
static_assert(sizeof(LifecycleControlHandle) == sizeof(void*));
static_assert(std::is_copy_constructible_v<LifecycleControlHandle>);
static_assert(std::is_nothrow_copy_constructible_v<LifecycleControlHandle>);
static_assert(std::is_copy_assignable_v<LifecycleControlHandle>);
static_assert(std::is_nothrow_copy_assignable_v<LifecycleControlHandle>);
static_assert(std::is_nothrow_move_constructible_v<LifecycleControlHandle>);
static_assert(std::is_nothrow_move_assignable_v<LifecycleControlHandle>);

enum class ProbeAction : std::uint8_t {
  kNone,
  kBlock,
  kRequestSelf,
  kRequestSelfThenFail,
  kRequestCross,
};

enum class StartStopRegistration : std::uint8_t {
  kRequired,
  kOptional,
  kAbsent,
};

enum class ProbeEventKind : std::uint8_t {
  kLifecycle,
  kComputeEntered,
  kRestartSubmitted,
  kComputeReturned,
};

struct ProbeEvent {
  ProbeEventKind kind{ProbeEventKind::kLifecycle};
  std::int32_t identity{};
  LifecycleStage stage{LifecycleStage::kConfigure};
  LifecycleTransitionPhase phase{LifecycleTransitionPhase::kNone};
  ParamGeneration params_generation{};
  Tag tag{};
  detail::RuntimeCallbackIdentity callback_identity{};
};

struct LifecycleFailurePoint {
  std::int32_t identity{};
  LifecycleStage stage{LifecycleStage::kConfigure};
  LifecycleTransitionPhase phase{LifecycleTransitionPhase::kNone};
};

struct LifecycleControlProbeState {
  static void reset() {
    const std::lock_guard lock(mutex);
    events.clear();
    cross_target = {};
    cross_target_ready = false;
    release_compute = false;
    request_finished = false;
    request_accepted = false;
    request_error = ErrorCode::kSuccess;
    request_receipt = {};
    lifecycle_block_enabled = false;
    lifecycle_block_entered = false;
    lifecycle_block_released = false;
    readiness_polled = false;
    lifecycle_failures.clear();
    clock_now_ns.store(0, std::memory_order_relaxed);
  }

  static void record(ProbeEvent event) {
    const std::lock_guard lock(mutex);
    events.push_back(event);
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_event(ProbeEventKind kind, std::int32_t identity) {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [=] {
      return std::any_of(events.begin(), events.end(), [=](const ProbeEvent& event) {
        return event.kind == kind && event.identity == identity;
      });
    });
  }

  [[nodiscard]] static bool wait_for_compute_generation(std::int32_t identity,
                                                        std::uint32_t generation) {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [=] {
      return std::any_of(events.begin(), events.end(), [=](const ProbeEvent& event) {
        return event.kind == ProbeEventKind::kComputeEntered && event.identity == identity &&
               event.tag.generation == generation;
      });
    });
  }

  [[nodiscard]] static bool wait_for_lifecycle(std::int32_t identity, LifecycleStage stage,
                                               LifecycleTransitionPhase phase) {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [=] {
      return std::any_of(events.begin(), events.end(), [=](const ProbeEvent& event) {
        return event.kind == ProbeEventKind::kLifecycle && event.identity == identity &&
               event.stage == stage && event.phase == phase;
      });
    });
  }

  [[nodiscard]] static bool wait_for_request() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return request_finished; });
  }

  static void set_cross_target(OperatorRef target) {
    const std::lock_guard lock(mutex);
    cross_target = target;
    cross_target_ready = true;
    cv.notify_all();
  }

  [[nodiscard]] static OperatorRef wait_for_cross_target() {
    std::unique_lock lock(mutex);
    if (!cv.wait_for(lock, 5s, [] { return cross_target_ready; })) {
      return {};
    }
    return cross_target;
  }

  static void save_request(expected<LifecycleRestartReceipt, Error> result, std::int32_t identity) {
    const std::lock_guard lock(mutex);
    request_finished = true;
    request_accepted = result.has_value();
    if (result) {
      request_receipt = *result;
    } else {
      request_error = result.error().code;
    }
    events.push_back({.kind = ProbeEventKind::kRestartSubmitted, .identity = identity});
    cv.notify_all();
  }

  [[nodiscard]] static LifecycleRestartReceipt receipt() {
    const std::lock_guard lock(mutex);
    return request_receipt;
  }

  [[nodiscard]] static bool accepted() {
    const std::lock_guard lock(mutex);
    return request_accepted;
  }

  [[nodiscard]] static ErrorCode error() {
    const std::lock_guard lock(mutex);
    return request_error;
  }

  static void allow_compute_return() {
    const std::lock_guard lock(mutex);
    release_compute = true;
    cv.notify_all();
  }

  static void publish_readiness_poll() {
    const std::lock_guard lock(mutex);
    readiness_polled = true;
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_readiness_poll() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return readiness_polled; });
  }

  static void wait_until_released() {
    std::unique_lock lock(mutex);
    static_cast<void>(cv.wait_for(lock, 5s, [] { return release_compute; }));
  }

  static void block_lifecycle(std::int32_t identity, LifecycleStage stage,
                              LifecycleTransitionPhase phase) {
    const std::lock_guard lock(mutex);
    lifecycle_block_enabled = true;
    lifecycle_block_identity = identity;
    lifecycle_block_stage = stage;
    lifecycle_block_phase = phase;
    lifecycle_block_entered = false;
    lifecycle_block_released = false;
  }

  [[nodiscard]] static bool wait_for_lifecycle_block() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return lifecycle_block_entered; });
  }

  static void release_lifecycle_block() {
    const std::lock_guard lock(mutex);
    lifecycle_block_released = true;
    cv.notify_all();
  }

  static void wait_if_lifecycle_blocked(std::int32_t identity, LifecycleStage stage,
                                        LifecycleTransitionPhase phase) {
    std::unique_lock lock(mutex);
    if (!lifecycle_block_enabled || lifecycle_block_identity != identity ||
        lifecycle_block_stage != stage || lifecycle_block_phase != phase) {
      return;
    }
    lifecycle_block_entered = true;
    cv.notify_all();
    static_cast<void>(cv.wait_for(lock, 5s, [] { return lifecycle_block_released; }));
  }

  static void fail_lifecycle_once(std::int32_t identity, LifecycleStage stage,
                                  LifecycleTransitionPhase phase) {
    const std::lock_guard lock(mutex);
    lifecycle_failures.push_back({.identity = identity, .stage = stage, .phase = phase});
  }

  [[nodiscard]] static bool consume_lifecycle_failure(std::int32_t identity, LifecycleStage stage,
                                                      LifecycleTransitionPhase phase) {
    const std::lock_guard lock(mutex);
    const auto failure = std::find_if(lifecycle_failures.begin(),
                                      lifecycle_failures.end(),
                                      [&](const LifecycleFailurePoint& candidate) {
                                        return candidate.identity == identity &&
                                               candidate.stage == stage && candidate.phase == phase;
                                      });
    if (failure == lifecycle_failures.end()) {
      return false;
    }
    lifecycle_failures.erase(failure);
    return true;
  }

  [[nodiscard]] static std::vector<ProbeEvent> snapshot() {
    const std::lock_guard lock(mutex);
    return events;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline std::vector<ProbeEvent> events;
  static inline OperatorRef cross_target;
  static inline bool cross_target_ready{};
  static inline bool release_compute{};
  static inline bool request_finished{};
  static inline bool request_accepted{};
  static inline ErrorCode request_error{ErrorCode::kSuccess};
  static inline LifecycleRestartReceipt request_receipt;
  static inline bool lifecycle_block_enabled{};
  static inline std::int32_t lifecycle_block_identity{};
  static inline LifecycleStage lifecycle_block_stage{LifecycleStage::kConfigure};
  static inline LifecycleTransitionPhase lifecycle_block_phase{LifecycleTransitionPhase::kNone};
  static inline bool lifecycle_block_entered{};
  static inline bool lifecycle_block_released{};
  static inline bool readiness_polled{};
  static inline std::vector<LifecycleFailurePoint> lifecycle_failures;
  static inline std::atomic<std::int64_t> clock_now_ns{};
};

class LifecycleControlProbeClock final : public Clock {
 public:
  [[nodiscard]] double time() const override { return timestamp_to_time(timestamp()); }
  [[nodiscard]] std::int64_t timestamp() const override {
    return LifecycleControlProbeState::clock_now_ns.load(std::memory_order_relaxed);
  }
  [[nodiscard]] expected<void, Error> sleep_for(std::int64_t) override { return {}; }
  [[nodiscard]] expected<void, Error> sleep_until(std::int64_t) override { return {}; }
};

class SilentOpenPortBinding final : public detail::PortBinding {
 public:
  explicit SilentOpenPortBinding(const detail::RuntimePortBindingRequest& request) noexcept
      : source_(request.source),
        capacity_(request.capacity),
        source_generation_(request.source_generation) {}

  [[nodiscard]] expected<std::size_t, Error> drain(
      std::span<detail::RuntimeEvent>) noexcept override {
    LifecycleControlProbeState::publish_readiness_poll();
    return 0U;
  }
  [[nodiscard]] std::size_t capacity() const noexcept override { return capacity_; }
  [[nodiscard]] bool closed() const noexcept override { return false; }
  [[nodiscard]] SourceId source() const noexcept override { return source_; }
  [[nodiscard]] expected<std::uint64_t, Error> advance_source_generation() noexcept override {
    if (source_generation_ == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    return ++source_generation_;
  }

 private:
  SourceId source_;
  std::size_t capacity_{};
  std::uint64_t source_generation_{};
};

class SilentOpenPortBindingFactory final : public detail::RuntimePortBindingFactory {
 public:
  [[nodiscard]] expected<std::unique_ptr<detail::PortBinding>, Error> bind(
      const detail::RuntimePortBindingRequest& request,
      TimePoint start_coordinate) noexcept override {
    if (!request.source || request.capacity == 0U || !request.reference_clock.valid() ||
        start_coordinate.clock_id != request.reference_clock) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "RUNTIME_READINESS_BINDING_UNAVAILABLE"});
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      return std::unique_ptr<detail::PortBinding>{new SilentOpenPortBinding{request}};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      return make_unexpected(
          Error{ErrorCode::kResourceExhausted, "RUNTIME_READINESS_BINDING_UNAVAILABLE"});
    }
#endif
  }
};

class LifecycleControlProbeOperator final : public Operator<> {
 public:
  LifecycleControlProbeOperator(
      std::int32_t identity, ProbeAction action,
      StartStopRegistration start_stop_registration = StartStopRegistration::kRequired,
      bool deadline_enabled = false)
      : identity_(identity),
        action_(action),
        start_stop_registration_(start_stop_registration),
        deadline_enabled_(deadline_enabled) {}

  void setup(OperatorSpec& spec) override {
    spec.spin_source(spin_, "activation").capacity(1);
    auto& lifecycle = spec.lifecycle();
    lifecycle.stage(LifecycleStage::kConfigure, &LifecycleControlProbeOperator::configure)
        .stage(LifecycleStage::kDiscover, &LifecycleControlProbeOperator::discover)
        .stage(LifecycleStage::kAllocate, &LifecycleControlProbeOperator::allocate)
        .stage(LifecycleStage::kWarmUp, &LifecycleControlProbeOperator::warm_up)
        .stage(LifecycleStage::kArm, &LifecycleControlProbeOperator::arm)
        .stage(LifecycleStage::kRelease, &LifecycleControlProbeOperator::release);
    if (start_stop_registration_ == StartStopRegistration::kRequired) {
      lifecycle.stage(LifecycleStage::kStart, &LifecycleControlProbeOperator::on_start)
          .stage(LifecycleStage::kStop, &LifecycleControlProbeOperator::on_stop);
    } else if (start_stop_registration_ == StartStopRegistration::kOptional) {
      lifecycle.optional(LifecycleStage::kStart, &LifecycleControlProbeOperator::on_start)
          .optional(LifecycleStage::kStop, &LifecycleControlProbeOperator::on_stop);
    }
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnSpin{.event = spin_});
    if (deadline_enabled_) {
      result.max_start_delay(5ns).on_violation(OnViolation::kEmitWitness);
    }
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    LifecycleControlProbeState::record(
        {.kind = ProbeEventKind::kComputeEntered, .identity = identity_, .tag = context.tag()});
    if (action_ == ProbeAction::kRequestSelf || action_ == ProbeAction::kRequestSelfThenFail) {
      LifecycleControlProbeState::save_request(context.request_restart(), identity_);
    } else if (action_ == ProbeAction::kRequestCross) {
      const OperatorRef target = LifecycleControlProbeState::wait_for_cross_target();
      LifecycleControlProbeState::save_request(context.request_restart(target), identity_);
    }
    if (action_ != ProbeAction::kNone) {
      LifecycleControlProbeState::wait_until_released();
    }
    LifecycleControlProbeState::record(
        {.kind = ProbeEventKind::kComputeReturned, .identity = identity_, .tag = context.tag()});
    if (action_ == ProbeAction::kRequestSelfThenFail) {
      return make_unexpected(
          Error{ErrorCode::kFailure, "compute failure after queued self restart"});
    }
    return {};
  }

  LifecycleStatus checkpoint(LifecycleContext& context, LifecycleStage stage) noexcept {
    LifecycleControlProbeState::record({
        .kind = ProbeEventKind::kLifecycle,
        .identity = identity_,
        .stage = stage,
        .phase = context.transition_phase(),
        .params_generation = context.params_generation(),
        .callback_identity = detail::runtime_callback_identity(),
    });
    LifecycleControlProbeState::wait_if_lifecycle_blocked(
        identity_, stage, context.transition_phase());
    if (LifecycleControlProbeState::consume_lifecycle_failure(
            identity_, stage, context.transition_phase())) {
      return LifecycleStatus::kFatalFailure;
    }
    return LifecycleStatus::kOk;
  }

  LifecycleStatus configure(LifecycleContext& context) noexcept {
    return checkpoint(context, LifecycleStage::kConfigure);
  }
  LifecycleStatus discover(LifecycleContext& context) noexcept {
    return checkpoint(context, LifecycleStage::kDiscover);
  }
  LifecycleStatus allocate(LifecycleContext& context) noexcept {
    return checkpoint(context, LifecycleStage::kAllocate);
  }
  LifecycleStatus warm_up(LifecycleContext& context) noexcept {
    return checkpoint(context, LifecycleStage::kWarmUp);
  }
  LifecycleStatus arm(LifecycleContext& context) noexcept {
    return checkpoint(context, LifecycleStage::kArm);
  }
  LifecycleStatus on_start(LifecycleContext& context) noexcept {
    return checkpoint(context, LifecycleStage::kStart);
  }
  LifecycleStatus on_stop(LifecycleContext& context) noexcept {
    return checkpoint(context, LifecycleStage::kStop);
  }
  LifecycleStatus release(LifecycleContext& context) noexcept {
    return checkpoint(context, LifecycleStage::kRelease);
  }

 private:
  SpinSource spin_;
  std::int32_t identity_{};
  ProbeAction action_{ProbeAction::kNone};
  StartStopRegistration start_stop_registration_{StartStopRegistration::kRequired};
  bool deadline_enabled_{};
};

class LifecycleClockRestartProbeOperator final : public Operator<> {
 public:
  LifecycleClockRestartProbeOperator(std::int32_t identity, ProbeAction action)
      : identity_(identity), action_(action) {}

  void setup(OperatorSpec& spec) override {
    spec.lifecycle()
        .stage(LifecycleStage::kConfigure, &LifecycleClockRestartProbeOperator::configure)
        .stage(LifecycleStage::kDiscover, &LifecycleClockRestartProbeOperator::discover)
        .stage(LifecycleStage::kAllocate, &LifecycleClockRestartProbeOperator::allocate)
        .stage(LifecycleStage::kWarmUp, &LifecycleClockRestartProbeOperator::warm_up)
        .stage(LifecycleStage::kArm, &LifecycleClockRestartProbeOperator::arm)
        .stage(LifecycleStage::kStart, &LifecycleClockRestartProbeOperator::on_start)
        .stage(LifecycleStage::kStop, &LifecycleClockRestartProbeOperator::on_stop)
        .stage(LifecycleStage::kRelease, &LifecycleClockRestartProbeOperator::release);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ns});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    LifecycleControlProbeState::record(
        {.kind = ProbeEventKind::kComputeEntered, .identity = identity_, .tag = context.tag()});
    if (action_ == ProbeAction::kRequestSelf) {
      LifecycleControlProbeState::save_request(context.request_restart(), identity_);
      LifecycleControlProbeState::wait_until_released();
    } else if (action_ == ProbeAction::kRequestCross) {
      const OperatorRef target = LifecycleControlProbeState::wait_for_cross_target();
      LifecycleControlProbeState::save_request(context.request_restart(target), identity_);
      LifecycleControlProbeState::wait_until_released();
    }
    LifecycleControlProbeState::record(
        {.kind = ProbeEventKind::kComputeReturned, .identity = identity_, .tag = context.tag()});
    return {};
  }

  LifecycleStatus checkpoint(LifecycleContext& context, LifecycleStage stage) noexcept {
    LifecycleControlProbeState::record({
        .kind = ProbeEventKind::kLifecycle,
        .identity = identity_,
        .stage = stage,
        .phase = context.transition_phase(),
        .params_generation = context.params_generation(),
        .callback_identity = detail::runtime_callback_identity(),
    });
    if (LifecycleControlProbeState::consume_lifecycle_failure(
            identity_, stage, context.transition_phase())) {
      const auto phase = context.transition_phase();
      context.report_error(phase == LifecycleTransitionPhase::kRestartForward
                               ? ErrorCode::kInvalidArgument
                               : ErrorCode::kFailure,
                           phase == LifecycleTransitionPhase::kRestartForward
                               ? "candidate warm-up failure"
                               : (phase == LifecycleTransitionPhase::kForwardRollback
                                      ? "candidate rollback failure"
                                      : "previous lifecycle restoration failure"));
      return LifecycleStatus::kFatalFailure;
    }
    return LifecycleStatus::kOk;
  }

  LifecycleStatus configure(LifecycleContext& context) noexcept {
    return checkpoint(context, LifecycleStage::kConfigure);
  }
  LifecycleStatus discover(LifecycleContext& context) noexcept {
    return checkpoint(context, LifecycleStage::kDiscover);
  }
  LifecycleStatus allocate(LifecycleContext& context) noexcept {
    return checkpoint(context, LifecycleStage::kAllocate);
  }
  LifecycleStatus warm_up(LifecycleContext& context) noexcept {
    return checkpoint(context, LifecycleStage::kWarmUp);
  }
  LifecycleStatus arm(LifecycleContext& context) noexcept {
    return checkpoint(context, LifecycleStage::kArm);
  }
  LifecycleStatus on_start(LifecycleContext& context) noexcept {
    return checkpoint(context, LifecycleStage::kStart);
  }
  LifecycleStatus on_stop(LifecycleContext& context) noexcept {
    return checkpoint(context, LifecycleStage::kStop);
  }
  LifecycleStatus release(LifecycleContext& context) noexcept {
    return checkpoint(context, LifecycleStage::kRelease);
  }

 private:
  std::int32_t identity_{};
  ProbeAction action_{ProbeAction::kNone};
};

enum class TargetBindingCase : std::uint8_t {
  kSelf,
  kCrossRef,
  kCrossPath,
  kAlreadyRunBound,
  kForeignRun,
  kForeignPlan,
  kUnresolvedRef,
  kUnresolvedPath,
  kTargetNotAdmitted,
  kSelfNotAllowed,
  kCrossNotAllowed,
};

struct TargetBindingObservation {
  TargetBindingCase test_case{TargetBindingCase::kSelf};
  bool success{};
  OperatorRef target{};
  ErrorCode error{ErrorCode::kSuccess};
};

struct LifecycleTargetBindingProbeState {
  static void reset(OperatorRef first, OperatorRef second, OperatorRef third,
                    OperatorPath second_path, OperatorPath missing_path) {
    const std::lock_guard lock(mutex);
    first_target = first;
    second_target = second;
    third_target = third;
    authorized_path = std::move(second_path);
    unresolved_path = std::move(missing_path);
    observations.clear();
    completed_callbacks = 0U;
  }

  static void record(TargetBindingCase test_case, expected<OperatorRef, Error> result) {
    const std::lock_guard lock(mutex);
    observations.push_back({
        .test_case = test_case,
        .success = result.has_value(),
        .target = result ? *result : OperatorRef{},
        .error = result ? ErrorCode::kSuccess : result.error().code,
    });
  }

  static void exercise(std::int32_t identity, LifecycleContext& context) {
    if (identity == 1) {
      record(TargetBindingCase::kSelf, context.restart_target(first_target));
      const auto cross = context.restart_target(second_target);
      record(TargetBindingCase::kCrossRef, cross);
      record(TargetBindingCase::kCrossPath, context.restart_target(authorized_path));
      if (cross) {
        record(TargetBindingCase::kAlreadyRunBound, context.restart_target(*cross));
        OperatorRef foreign_run = *cross;
        ++foreign_run.run_generation;
        record(TargetBindingCase::kForeignRun, context.restart_target(foreign_run));
      }
      OperatorRef foreign_plan = second_target;
      ++foreign_plan.plan_generation;
      record(TargetBindingCase::kForeignPlan, context.restart_target(foreign_plan));
      OperatorRef unresolved = second_target;
      unresolved.id = std::numeric_limits<OperatorInstanceId>::max();
      record(TargetBindingCase::kUnresolvedRef, context.restart_target(unresolved));
      record(TargetBindingCase::kUnresolvedPath, context.restart_target(unresolved_path));
      record(TargetBindingCase::kTargetNotAdmitted, context.restart_target(third_target));
    } else if (identity == 2) {
      record(TargetBindingCase::kSelfNotAllowed, context.restart_target(second_target));
      record(TargetBindingCase::kCrossNotAllowed, context.restart_target(first_target));
    }
    {
      const std::lock_guard lock(mutex);
      ++completed_callbacks;
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait_for_callbacks() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return completed_callbacks == 3U; });
  }

  [[nodiscard]] static std::vector<TargetBindingObservation> snapshot() {
    const std::lock_guard lock(mutex);
    return observations;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline OperatorRef first_target;
  static inline OperatorRef second_target;
  static inline OperatorRef third_target;
  static inline OperatorPath authorized_path;
  static inline OperatorPath unresolved_path;
  static inline std::vector<TargetBindingObservation> observations;
  static inline std::size_t completed_callbacks{};
};

class LifecycleTargetBindingProbeOperator final : public Operator<> {
 public:
  explicit LifecycleTargetBindingProbeOperator(std::int32_t identity) : identity_(identity) {}

  void setup(OperatorSpec& spec) override {
    spec.spin_source(spin_, "activation").capacity(1);
    spec.lifecycle().stage(LifecycleStage::kConfigure,
                           &LifecycleTargetBindingProbeOperator::configure);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnSpin{.event = spin_});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  LifecycleStatus configure(LifecycleContext& context) noexcept {
    LifecycleTargetBindingProbeState::exercise(identity_, context);
    return LifecycleStatus::kOk;
  }

 private:
  SpinSource spin_;
  std::int32_t identity_{};
};

class ComputeReleaseGuard {
 public:
  ComputeReleaseGuard() = default;
  ComputeReleaseGuard(const ComputeReleaseGuard&) = delete;
  ComputeReleaseGuard& operator=(const ComputeReleaseGuard&) = delete;
  ~ComputeReleaseGuard() { LifecycleControlProbeState::allow_compute_return(); }
};

class PreDispatchReleaseGuard {
 public:
  explicit PreDispatchReleaseGuard(std::shared_ptr<detail::RuntimePreDispatchTestControl> control)
      : control_(std::move(control)) {}
  PreDispatchReleaseGuard(const PreDispatchReleaseGuard&) = delete;
  PreDispatchReleaseGuard& operator=(const PreDispatchReleaseGuard&) = delete;
  ~PreDispatchReleaseGuard() {
    if (control_ != nullptr) {
      control_->release();
    }
  }

 private:
  std::shared_ptr<detail::RuntimePreDispatchTestControl> control_;
};

[[nodiscard]] LifecycleTransactionOptions ample_lifecycle_capacity(
    std::uint32_t max_pending_requests = 64U) {
  return LifecycleTransactionOptions{
      .max_pending_requests = max_pending_requests,
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

class AdvisoryRestartProbe final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.lifecycle().stage(LifecycleStage::kConfigure, &AdvisoryRestartProbe::configure);
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    if (requests_++ == 15U) {
      context.request_completion();
      return {};
    }
    auto requested = context.request_restart();
    if (!requested) {
      return make_unexpected(std::move(requested.error()));
    }
    return {};
  }
  LifecycleStatus configure(LifecycleContext& context) noexcept {
    if (context.transition_phase() != LifecycleTransitionPhase::kRestartForward) {
      return LifecycleStatus::kOk;
    }
    context.report_error(ErrorCode::kInvalidArgument, "recoverable restart failure");
    return LifecycleStatus::kFatalFailure;
  }

 private:
  std::size_t requests_{};
};

TEST(LifecycleControl, RecoveredRestartOverflowRemainsASuccessfulAdvisoryReport) {
  Graph graph{"advisory-restart-overflow"};
  const auto probe = graph.op<AdvisoryRestartProbe>("probe");
  graph.lifecycle().allow_self_restart(probe);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  CompileOptions options;
  options.lifecycle_transactions = ample_lifecycle_capacity();
  const auto plan = compile(graph, options);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto session = run_async(plan);
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kRequestedCompleted);
  ASSERT_EQ(session.diagnostics().size(), 15U);
  EXPECT_EQ(session.diagnostics().back().token, "RUNTIME_DIAGNOSTIC_CAPACITY_EXHAUSTED");
  for (const auto& diagnostic : session.diagnostics()) {
    EXPECT_EQ(diagnostic.severity, DiagnosticSeverity::kWarning);
  }
}

[[nodiscard]] LifecycleTransactionOptions exact_active_successor_capacity() {
  return LifecycleTransactionOptions{
      .max_pending_requests = 2U,
      .max_pending_transactions = 2U,
      .max_receipts = 4U,
      .max_fenced_operators = 2U,
      .max_candidate_generations = 2U,
      .max_events = 32U,
      .max_audit_records = 4U,
      .max_quiesce_records = 4U,
      .max_rollback_records = 4U,
  };
}

[[nodiscard]] bool same_capacity(const LifecycleTransactionCapacityValues& left,
                                 const LifecycleTransactionCapacityValues& right) {
  return left.max_pending_requests == right.max_pending_requests &&
         left.max_pending_transactions == right.max_pending_transactions &&
         left.max_receipts == right.max_receipts &&
         left.max_fenced_operators == right.max_fenced_operators &&
         left.max_candidate_generations == right.max_candidate_generations &&
         left.max_events == right.max_events && left.max_audit_records == right.max_audit_records &&
         left.max_quiesce_records == right.max_quiesce_records &&
         left.max_rollback_records == right.max_rollback_records;
}

struct LifecyclePlanOptions {
  ProbeAction first_action{ProbeAction::kBlock};
  bool lifecycle_transactions{true};
  bool allow_cross_request{true};
  std::uint32_t max_pending_requests{64U};
  bool exact_active_successor_capacity{};
  StartStopRegistration start_stop_registration{StartStopRegistration::kRequired};
  bool deadline_second{};
};

[[nodiscard]] ExecutionPlan make_lifecycle_control_plan(LifecyclePlanOptions requested = {}) {
  Graph graph{"lifecycle-control-runtime"};
  const auto first = graph.op<LifecycleControlProbeOperator>(
      "first", 1, requested.first_action, requested.start_stop_registration);
  const auto second = graph.op<LifecycleControlProbeOperator>("second",
                                                              2,
                                                              ProbeAction::kNone,
                                                              requested.start_stop_registration,
                                                              requested.deadline_second);
  graph.partition("first-spin-worker").add(first);
  graph.partition("second-spin-worker").add(second);
  graph.lifecycle().allow_self_restart(first).allow_external_restart(first).allow_external_restart(
      second);
  if (requested.allow_cross_request) {
    graph.lifecycle().allow_restart_request(first, second);
  }
  const ClockKey clock = graph.add_clock<LifecycleControlProbeClock>("control-clock");
  graph.set_default_clock(clock);

  CompileOptions options;
  options.deployment
      .bind_thread_placement("first-spin-worker", ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("second-spin-worker",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  if (requested.lifecycle_transactions) {
    options.lifecycle_transactions = requested.exact_active_successor_capacity
                                         ? exact_active_successor_capacity()
                                         : ample_lifecycle_capacity(requested.max_pending_requests);
  }
  return test::enable_direct_runtime(compile(graph, options));
}

[[nodiscard]] ExecutionPlan make_idle_lifecycle_control_plan() {
  const ExecutionPlan direct = make_lifecycle_control_plan({.first_action = ProbeAction::kNone});
  return detail::RuntimeTestPlanAccess::with_port_binding_factory(
      direct, std::make_shared<SilentOpenPortBindingFactory>());
}

[[nodiscard]] ExecutionPlan make_lifecycle_clock_restart_plan() {
  Graph graph{"lifecycle-clock-restart-runtime"};
  const auto first =
      graph.op<LifecycleClockRestartProbeOperator>("first", 1, ProbeAction::kRequestCross);
  const auto second = graph.op<LifecycleClockRestartProbeOperator>("second", 2, ProbeAction::kNone);
  graph.partition("first-clock-worker").add(first);
  graph.partition("second-clock-worker").add(second);
  graph.lifecycle().allow_external_restart(second).allow_restart_request(first, second);
  const ClockKey clock = graph.add_clock<LifecycleControlProbeClock>("control-clock");
  graph.set_default_clock(clock);

  CompileOptions options;
  options.deployment
      .bind_thread_placement("first-clock-worker",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("second-clock-worker",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  options.lifecycle_transactions = ample_lifecycle_capacity();
  return compile(graph, options);
}

[[nodiscard]] ExecutionPlan make_lifecycle_dependency_closure_restart_plan() {
  Graph graph{"lifecycle-dependency-closure-restart-runtime"};
  const auto requester =
      graph.op<LifecycleClockRestartProbeOperator>("a_requester", 1, ProbeAction::kRequestCross);
  const auto target =
      graph.op<LifecycleClockRestartProbeOperator>("b_target", 2, ProbeAction::kNone);
  const auto unrelated =
      graph.op<LifecycleClockRestartProbeOperator>("c_unrelated", 4, ProbeAction::kNone);
  const auto dependent =
      graph.op<LifecycleClockRestartProbeOperator>("d_dependent", 3, ProbeAction::kNone);
  graph.partition("requester-clock-worker").add(requester);
  graph.partition("target-clock-worker").add(target);
  graph.partition("unrelated-clock-worker").add(unrelated);
  graph.partition("dependent-clock-worker").add(dependent);
  graph.lifecycle()
      .before(target, dependent, LifecycleStage::kStart)
      .allow_external_restart(target)
      .allow_restart_request(requester, target);
  const ClockKey clock = graph.add_clock<LifecycleControlProbeClock>("control-clock");
  graph.set_default_clock(clock);

  CompileOptions options;
  options.deployment
      .bind_thread_placement("requester-clock-worker",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("target-clock-worker",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("unrelated-clock-worker",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("dependent-clock-worker",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  options.lifecycle_transactions = ample_lifecycle_capacity();
  return compile(graph, options);
}

[[nodiscard]] ExecutionPlan make_lifecycle_target_binding_plan() {
  Graph graph{"lifecycle-target-binding-runtime"};
  const auto first = graph.op<LifecycleTargetBindingProbeOperator>("first", 1);
  const auto second = graph.op<LifecycleTargetBindingProbeOperator>("second", 2);
  const auto third = graph.op<LifecycleTargetBindingProbeOperator>("third", 3);
  graph.partition("first-binding-worker").add(first);
  graph.partition("second-binding-worker").add(second);
  graph.partition("third-binding-worker").add(third);
  graph.lifecycle().allow_self_restart(first).allow_external_restart(second).allow_restart_request(
      first, second);
  const ClockKey clock = graph.add_clock<LifecycleControlProbeClock>("binding-clock");
  graph.set_default_clock(clock);

  CompileOptions options;
  options.deployment
      .bind_thread_placement("first-binding-worker",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("second-binding-worker",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll})
      .bind_thread_placement("third-binding-worker",
                             ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
  options.lifecycle_transactions = ample_lifecycle_capacity();
  return test::enable_direct_runtime(compile(graph, options));
}

[[nodiscard]] OperatorRef plan_target(const ExecutionPlan& plan, std::size_t index) {
  return plan.lifecycle_restart_targets()[index].target;
}

[[nodiscard]] bool terminal(LifecycleTransactionState state) noexcept {
  return state == LifecycleTransactionState::kAcceptedNoChange ||
         state == LifecycleTransactionState::kPublished ||
         state == LifecycleTransactionState::kSuperseded ||
         state == LifecycleTransactionState::kCanceled ||
         state == LifecycleTransactionState::kRolledBack ||
         state == LifecycleTransactionState::kFailedStopped;
}

[[nodiscard]] std::size_t first_event_index(std::span<const ProbeEvent> events, ProbeEventKind kind,
                                            std::int32_t identity) {
  const auto found = std::find_if(events.begin(), events.end(), [=](const ProbeEvent& event) {
    return event.kind == kind && event.identity == identity;
  });
  return found == events.end() ? events.size()
                               : static_cast<std::size_t>(std::distance(events.begin(), found));
}

[[nodiscard]] std::size_t first_phase_index(std::span<const ProbeEvent> events,
                                            std::int32_t identity, LifecycleTransitionPhase phase) {
  const auto found = std::find_if(events.begin(), events.end(), [=](const ProbeEvent& event) {
    return event.kind == ProbeEventKind::kLifecycle && event.identity == identity &&
           event.phase == phase;
  });
  return found == events.end() ? events.size()
                               : static_cast<std::size_t>(std::distance(events.begin(), found));
}

[[nodiscard]] std::size_t event_count(std::span<const ProbeEvent> events, std::int32_t identity,
                                      LifecycleStage stage, LifecycleTransitionPhase phase) {
  return static_cast<std::size_t>(
      std::count_if(events.begin(), events.end(), [=](const ProbeEvent& event) {
        return event.kind == ProbeEventKind::kLifecycle && event.identity == identity &&
               event.stage == stage && event.phase == phase;
      }));
}

[[nodiscard]] bool all_lifecycle_generations_are_zero(std::span<const ProbeEvent> events) {
  return std::all_of(events.begin(), events.end(), [](const ProbeEvent& event) {
    return event.kind != ProbeEventKind::kLifecycle || event.params_generation == 0U;
  });
}

[[nodiscard]] std::string diagnostic_tokens(const RunSession& session) {
  std::string result;
  for (const Diagnostic& diagnostic : session.diagnostics()) {
    if (!result.empty()) {
      result += ',';
    }
    result += diagnostic.token;
  }
  return result;
}

void expect_lifecycle_failure(const detail::LifecycleExecutionFailure& failure,
                              detail::LifecycleNodeId node_id, LifecycleStage stage,
                              LifecycleTransitionPhase phase) {
  EXPECT_TRUE(failure.present);
  EXPECT_EQ(failure.node_id, node_id);
  EXPECT_EQ(failure.stage, stage);
  EXPECT_EQ(failure.phase, phase);
  EXPECT_EQ(failure.status, LifecycleStatus::kFatalFailure);
}

void expect_no_lifecycle_failure(const detail::LifecycleExecutionFailure& failure) {
  EXPECT_FALSE(failure.present);
}

void expect_public_lifecycle_cause(const RunSession& session, LifecycleTransactionId transaction,
                                   LifecycleStage stage, LifecycleTransitionPhase phase,
                                   DiagnosticFailureRole role) {
  const auto diagnostics = session.diagnostics();
  const auto found = std::find_if(diagnostics.begin(), diagnostics.end(), [&](const auto& row) {
    return row.detail && row.detail->stage == stage && row.detail->phase == phase;
  });
  ASSERT_NE(found, diagnostics.end());
  const auto& found_detail = found->detail;
  if (!found_detail.has_value()) {
    FAIL() << "Expected diagnostic detail";
  }
  EXPECT_EQ(found_detail->transaction_id, transaction);
  EXPECT_EQ(found_detail->role, role);
  EXPECT_FALSE(found_detail->operator_path.empty());
  EXPECT_EQ(found_detail->error.code, ErrorCode::kLifecycleStageFailed);
  EXPECT_FALSE(found_detail->error.message.empty());
  EXPECT_EQ(found_detail->primary_diagnostic_index, 0U);
}

// REQ: RUNTIME-008
TEST(LifecycleControl, ColdContextBindsOnlyAdmittedAuthorizedTargets) {
  const ExecutionPlan plan = make_lifecycle_target_binding_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.lifecycle_restart_targets().size(), 3U);
  ASSERT_EQ(plan.operator_parameters().size(), 3U);
  const OperatorRef first = plan_target(plan, 0U);
  const OperatorRef second = plan_target(plan, 1U);
  const OperatorRef third = plan_target(plan, 2U);
  auto second_path = OperatorPath::parse(plan.operator_parameters()[1].operator_path);
  auto missing_path = OperatorPath::parse("missing");
  ASSERT_TRUE(second_path);
  ASSERT_TRUE(missing_path);
  // Key setup: three operators with distinct authorization levels; first can self/cross restart,
  // second can only receive external restart, third has no authorization.
  LifecycleTargetBindingProbeState::reset(
      first, second, third, std::move(second_path).value(), std::move(missing_path).value());

  RunSession session = run_async(plan);
  ASSERT_TRUE(LifecycleTargetBindingProbeState::wait_for_callbacks()) << diagnostic_tokens(session);
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted) << diagnostic_tokens(session);

  const std::vector observations = LifecycleTargetBindingProbeState::snapshot();
  const auto find = [&](TargetBindingCase test_case) -> const TargetBindingObservation& {
    const auto found = std::find_if(observations.begin(),
                                    observations.end(),
                                    [&](const auto& item) { return item.test_case == test_case; });
    if (found == observations.end()) {
      ADD_FAILURE() << "missing target-binding observation "
                    << static_cast<std::uint32_t>(test_case);
      static const TargetBindingObservation missing;
      return missing;
    }
    return *found;
  };
  // Key assertion: self/cross-ref/cross-path all succeed and return run-bound refs;
  // every invalid or unauthorized case returns the expected error code.
  for (const TargetBindingCase test_case :
       {TargetBindingCase::kSelf, TargetBindingCase::kCrossRef, TargetBindingCase::kCrossPath}) {
    const auto& observation = find(test_case);
    EXPECT_TRUE(observation.success);
    EXPECT_TRUE(observation.target.run_bound());
    EXPECT_EQ(observation.target.plan_generation, first.plan_generation);
  }
  EXPECT_EQ(find(TargetBindingCase::kSelf).target.id, first.id);
  EXPECT_EQ(find(TargetBindingCase::kCrossRef).target.id, second.id);
  EXPECT_EQ(find(TargetBindingCase::kCrossPath).target, find(TargetBindingCase::kCrossRef).target);

  EXPECT_EQ(find(TargetBindingCase::kAlreadyRunBound).error, ErrorCode::kStaleOperatorRef);
  EXPECT_EQ(find(TargetBindingCase::kForeignRun).error, ErrorCode::kCrossSessionOperatorRef);
  EXPECT_EQ(find(TargetBindingCase::kForeignPlan).error, ErrorCode::kStaleOperatorRef);
  EXPECT_EQ(find(TargetBindingCase::kUnresolvedRef).error, ErrorCode::kStaleOperatorRef);
  EXPECT_EQ(find(TargetBindingCase::kUnresolvedPath).error, ErrorCode::kNotFound);
  EXPECT_EQ(find(TargetBindingCase::kTargetNotAdmitted).error,
            ErrorCode::kLifecycleRestartUnauthorized);
  EXPECT_EQ(find(TargetBindingCase::kSelfNotAllowed).error,
            ErrorCode::kLifecycleRestartUnauthorized);
  EXPECT_EQ(find(TargetBindingCase::kCrossNotAllowed).error,
            ErrorCode::kLifecycleRestartUnauthorized);
}

// REQ: RUNTIME-008
TEST(LifecycleControl, UnsupportedProfilesRejectBeforeRuntimeLifecycleStateOrCallbacksExist) {
  for (const Profile profile : {Profile::kRealTime, Profile::kRegulated}) {
    LifecycleControlProbeState::reset();
    Graph graph{"lifecycle-profile-rejection-runtime"};
    const auto target = graph.op<LifecycleControlProbeOperator>("target", 1, ProbeAction::kNone);
    graph.lifecycle().allow_self_restart(target);
    graph.region("restricted-region").add(target).profile(profile);
    const ClockKey clock = graph.add_clock<LifecycleControlProbeClock>("control-clock");
    graph.set_default_clock(clock);

    CompileOptions options;
    options.lifecycle_transactions = ample_lifecycle_capacity();
    const ExecutionPlan plan = compile(graph, options);
    ASSERT_FALSE(plan.ok());
    EXPECT_TRUE(std::any_of(
        plan.diagnostics().begin(), plan.diagnostics().end(), [](const Diagnostic& diagnostic) {
          return diagnostic.token == "LIFECYCLE_TRANSACTION_UNSUPPORTED_PROFILE";
        }));
    EXPECT_THROW(static_cast<void>(run_async(plan)), CompileError);
    EXPECT_TRUE(LifecycleControlProbeState::snapshot().empty());
  }
}

// REQ: RUNTIME-008
TEST(LifecycleControl, DisabledPlanHasNoActiveControlPathAndOperatorRefIsRunScoped) {
  LifecycleControlProbeState::reset();
  const ExecutionPlan plan = make_lifecycle_control_plan({.lifecycle_transactions = false});
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.lifecycle_restart_targets().size(), 2U);
  const OperatorRef target = plan_target(plan, 0U);

  RunSession session = run_async(plan);
  const ComputeReleaseGuard release_guard;
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 1))
      << diagnostic_tokens(session);
  const auto run_target = session.operator_ref(target);
  ASSERT_TRUE(run_target);
  EXPECT_TRUE(run_target->run_bound());
  EXPECT_EQ(run_target->id, target.id);
  EXPECT_EQ(run_target->plan_generation, target.plan_generation);

  const std::array targets{*run_target};
  constexpr std::array stages{LifecycleStage::kConfigure};
  const auto handle = session.lifecycle_control({
      .allowed_targets = targets,
      .allowed_from_stages = stages,
  });
  ASSERT_FALSE(handle);
  EXPECT_EQ(handle.error().code, ErrorCode::kLifecycleTransactionsDisabled);

  const LifecycleControlHandle inert;
  EXPECT_FALSE(static_cast<bool>(inert));
  EXPECT_EQ(inert.request_restart(*run_target).error().code, ErrorCode::kClosed);

  LifecycleControlProbeState::allow_compute_return();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_EQ(session.operator_ref(target).error().code, ErrorCode::kStaleOperatorRef);
}

// Optional Start/Stop callbacks remain advanced-member callbacks during replay, while an omitted
// optional slot remains a skipped logical checkpoint rather than acquiring legacy requiredness.
// REQ: LIFECYCLE-002
// REQ: RUNTIME-002
TEST(LifecycleControl, ReplaysPresentAndAbsentOptionalStartStopCallbacksExactly) {
  for (const StartStopRegistration registration :
       {StartStopRegistration::kOptional, StartStopRegistration::kAbsent}) {
    SCOPED_TRACE(static_cast<unsigned>(registration));
    LifecycleControlProbeState::reset();
    const ExecutionPlan plan = make_lifecycle_control_plan({
        .first_action = ProbeAction::kNone,
        .lifecycle_transactions = false,
        .start_stop_registration = registration,
    });
    ASSERT_TRUE(plan.ok()) << plan.json();
    LifecycleControlProbeState::reset();

    RunSession session = run_async(plan);
    session.wait();

    ASSERT_EQ(session.termination(), RunTermination::kCompleted) << diagnostic_tokens(session);
    const std::vector events = LifecycleControlProbeState::snapshot();
    const std::size_t expected_callbacks =
        registration == StartStopRegistration::kOptional ? 1U : 0U;
    for (const std::int32_t identity : {1, 2}) {
      EXPECT_EQ(
          event_count(
              events, identity, LifecycleStage::kStart, LifecycleTransitionPhase::kStartupForward),
          expected_callbacks);
      EXPECT_EQ(
          event_count(
              events, identity, LifecycleStage::kStop, LifecycleTransitionPhase::kShutdownTeardown),
          expected_callbacks);
    }
  }
}

// A present optional callback is optional only as a binding: once invoked, its fatal result still
// owns the normal startup-failure and rollback semantics.
// REQ: LIFECYCLE-002
// REQ: RUNTIME-004
TEST(LifecycleControl, OptionalStartFailureRemainsFatalAfterReplay) {
  LifecycleControlProbeState::reset();
  const ExecutionPlan plan = make_lifecycle_control_plan({
      .first_action = ProbeAction::kNone,
      .lifecycle_transactions = false,
      .start_stop_registration = StartStopRegistration::kOptional,
  });
  ASSERT_TRUE(plan.ok()) << plan.json();
  LifecycleControlProbeState::reset();
  LifecycleControlProbeState::fail_lifecycle_once(
      1, LifecycleStage::kStart, LifecycleTransitionPhase::kStartupForward);

  RunSession session = run_async(plan);
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  const std::vector events = LifecycleControlProbeState::snapshot();
  EXPECT_EQ(
      event_count(events, 1, LifecycleStage::kStart, LifecycleTransitionPhase::kStartupForward),
      1U);
}

// REQ: LIFECYCLE-002
// REQ: RUNTIME-002
// REQ: RUNTIME-008
TEST(LifecycleControl, OptionalStartStopCallbacksSurviveSelfRestartReplay) {
  LifecycleControlProbeState::reset();
  const ExecutionPlan plan = make_lifecycle_control_plan({
      .first_action = ProbeAction::kRequestSelf,
      .start_stop_registration = StartStopRegistration::kOptional,
  });
  ASSERT_TRUE(plan.ok()) << plan.json();
  const OperatorRef first = plan_target(plan, 0U);

  RunSession session = run_async(plan);
  const ComputeReleaseGuard release_guard;
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_request()) << diagnostic_tokens(session);
  ASSERT_TRUE(LifecycleControlProbeState::accepted());
  const auto first_run = session.operator_ref(first);
  ASSERT_TRUE(first_run);
  const std::array targets{*first_run};
  auto control = session.lifecycle_control({
      .allowed_targets = targets,
      .allowed_from_stages = kForwardStages,
  });
  ASSERT_TRUE(control);

  const LifecycleRestartReceipt receipt = LifecycleControlProbeState::receipt();
  LifecycleControlProbeState::allow_compute_return();
  const auto completed = control->await(receipt, 5s);
  ASSERT_TRUE(completed) << diagnostic_tokens(session);
  EXPECT_EQ(completed->state, LifecycleTransactionState::kPublished);
  session.wait();

  ASSERT_EQ(session.termination(), RunTermination::kCompleted) << diagnostic_tokens(session);
  const std::vector events = LifecycleControlProbeState::snapshot();
  EXPECT_EQ(
      event_count(events, 1, LifecycleStage::kStop, LifecycleTransitionPhase::kPreRestartTeardown),
      1U);
  EXPECT_EQ(
      event_count(events, 1, LifecycleStage::kStart, LifecycleTransitionPhase::kRestartForward),
      1U);
  EXPECT_EQ(
      event_count(events, 1, LifecycleStage::kStop, LifecycleTransitionPhase::kShutdownTeardown),
      1U);
}

// REQ: RUNTIME-008
// REQ: PARAM-021
TEST(LifecycleControl, ExternalRestartOfIdleTargetRunsBeforeAnyReadyCompute) {
  LifecycleControlProbeState::reset();
  const ExecutionPlan plan = make_idle_lifecycle_control_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const OperatorRef target = plan_target(plan, 0U);

  RunSession session = run_async(plan);
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_readiness_poll());
  const auto run_target = session.operator_ref(target);
  ASSERT_TRUE(run_target);
  const std::array targets{*run_target};
  constexpr std::array stages{LifecycleStage::kConfigure};
  auto control = session.lifecycle_control({
      .allowed_targets = targets,
      .allowed_from_stages = stages,
  });
  ASSERT_TRUE(control);
  const auto receipt = control->request_restart(*run_target);
  ASSERT_TRUE(receipt);

  const auto completed = control->await(*receipt, 5s);
  ASSERT_TRUE(completed) << diagnostic_tokens(session);
  EXPECT_EQ(completed->state, LifecycleTransactionState::kPublished);
  const std::vector events = LifecycleControlProbeState::snapshot();
  EXPECT_NE(first_phase_index(events, 1, LifecycleTransitionPhase::kRestartForward), events.size());
  EXPECT_EQ(first_event_index(events, ProbeEventKind::kComputeEntered, 1), events.size());
  EXPECT_EQ(first_event_index(events, ProbeEventKind::kComputeEntered, 2), events.size());

  session.request_stop();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
}

// REQ: RUNTIME-008
TEST(LifecycleControl, RestartIngressStaysClosedUntilGlobalStartupBoundary) {
  LifecycleControlProbeState::reset();
  LifecycleControlProbeState::block_lifecycle(
      1, LifecycleStage::kStart, LifecycleTransitionPhase::kStartupForward);
  const ExecutionPlan plan = make_lifecycle_control_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const OperatorRef first = plan_target(plan, 0U);

  RunSession session = run_async(plan);
  const ComputeReleaseGuard release_guard;
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_lifecycle_block());
  const auto first_run = session.operator_ref(first);
  ASSERT_TRUE(first_run);
  const std::array targets{*first_run};
  auto control = session.lifecycle_control({
      .allowed_targets = targets,
      .allowed_from_stages = kForwardStages,
  });
  ASSERT_TRUE(control);

  const auto starting = control->request_restart(*first_run);
  ASSERT_FALSE(starting);
  EXPECT_EQ(starting.error().code, ErrorCode::kLifecycleRunState);

  LifecycleControlProbeState::release_lifecycle_block();
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 1));
  const auto accepted = control->request_restart(*first_run);
  ASSERT_TRUE(accepted);
  LifecycleControlProbeState::allow_compute_return();
  const auto completed = control->await(*accepted, 5s);
  ASSERT_TRUE(completed);
  EXPECT_EQ(completed->state, LifecycleTransactionState::kPublished);
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
}

// The lifecycle state can outlive RunSession through a retained public handle. Finalization must
// therefore wait for an accepted submitter to finish dereferencing the run-owned wake state.
// REQ: RUNTIME-008
TEST(LifecycleControl, FinalizationWaitsForAcceptedSubmissionWakeToReturn) {
  const ExecutionPlan plan = make_lifecycle_control_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  constexpr RunGenerationId kRunGeneration = 101U;
  auto state = std::make_shared<detail::LifecycleControlState>(
      detail::ExecutionPlanAccess::retain(plan), kRunGeneration);
  ASSERT_TRUE(state->enabled());

  BlockingLifecycleWake wake;
  state->bind_wake(&wake, &BlockingLifecycleWake::signal);
  ASSERT_TRUE(state->open_ingress());

  OperatorRef target = plan_target(plan, 0U);
  target.run_generation = kRunGeneration;
  const std::array targets{target};
  constexpr std::array stages{LifecycleStage::kConfigure};
  auto control = state->make_handle({
      .allowed_targets = targets,
      .allowed_from_stages = stages,
  });
  ASSERT_TRUE(control);

  auto submitted = std::async(std::launch::async, [&] { return control->request_restart(target); });
  const bool wake_entered = wake.wait_until_entered();
  EXPECT_TRUE(wake_entered);
  if (!wake_entered) {
    wake.release();
    const auto receipt = submitted.get();
    ASSERT_TRUE(receipt);
    return;
  }

  std::promise<void> finalization_started;
  auto started = finalization_started.get_future();
  auto finalized = std::async(std::launch::async, [&] {
    finalization_started.set_value();
    state->finalize_close();
  });
  started.wait();

  EXPECT_EQ(finalized.wait_for(25ms), std::future_status::timeout);
  wake.release();

  const auto receipt = submitted.get();
  ASSERT_TRUE(receipt);
  EXPECT_EQ(finalized.wait_for(5s), std::future_status::ready);
  finalized.get();
}

// REQ: RUNTIME-008
TEST(LifecycleControl, ReceiptDiagnosticSinkFailureCannotChangeLifecycleOrRunOutcome) {
  LifecycleControlProbeState::reset();
  const auto diagnostic_sink = std::make_shared<RecordingLifecycleDiagnosticSink>(true);
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(
      make_lifecycle_control_plan(), diagnostic_sink);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const OperatorRef target = plan_target(plan, 0U);

  RunSession session = run_async(plan);
  const ComputeReleaseGuard release_guard;
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 1));
  const auto run_target = session.operator_ref(target);
  ASSERT_TRUE(run_target);
  const std::array targets{*run_target};
  auto control = session.lifecycle_control({
      .allowed_targets = targets,
      .allowed_from_stages = kForwardStages,
  });
  ASSERT_TRUE(control);
  const auto receipt = control->request_restart(*run_target);
  ASSERT_TRUE(receipt);

  LifecycleControlProbeState::allow_compute_return();
  const auto completed = control->await(*receipt, 5s);
  ASSERT_TRUE(completed);
  EXPECT_EQ(completed->state, LifecycleTransactionState::kPublished);
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
}

// REQ: RUNTIME-008
TEST(LifecycleControl, PolicyCopyIsBoundedAndMapsAllocationFailureWithoutChangingAuthority) {
  LifecycleControlProbeState::reset();
  const ExecutionPlan plan = make_lifecycle_control_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const OperatorRef first = plan_target(plan, 0U);
  const OperatorRef second = plan_target(plan, 1U);

  RunSession session = run_async(plan);
  const ComputeReleaseGuard release_guard;
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 1));
  const auto first_run = session.operator_ref(first);
  const auto second_run = session.operator_ref(second);
  ASSERT_TRUE(first_run);
  ASSERT_TRUE(second_run);
  const std::array targets{*first_run, *second_run};

  expected<LifecycleControlHandle, Error> failed_mint;
  {
    const test::ScopedAllocationFailure fail_first_allocation{0U};
    failed_mint = session.lifecycle_control({
        .allowed_targets = targets,
        .allowed_from_stages = kForwardStages,
    });
  }
  ASSERT_FALSE(failed_mint);
  EXPECT_EQ(failed_mint.error().code, ErrorCode::kResourceExhausted);

  auto control = session.lifecycle_control({
      .allowed_targets = targets,
      .allowed_from_stages = kForwardStages,
  });
  ASSERT_TRUE(control);
  expected<LifecycleControlHandle, Error> failed_restriction;
  {
    const test::ScopedAllocationFailure fail_first_allocation{0U};
    failed_restriction = control->restrict({
        .allowed_targets = targets,
        .allowed_from_stages = kForwardStages,
    });
  }
  ASSERT_FALSE(failed_restriction);
  EXPECT_EQ(failed_restriction.error().code, ErrorCode::kResourceExhausted);

  const std::array oversized_targets{*first_run, *second_run, *first_run};
  const auto oversized_mint = session.lifecycle_control({
      .allowed_targets = oversized_targets,
      .allowed_from_stages = kForwardStages,
  });
  ASSERT_FALSE(oversized_mint);
  EXPECT_EQ(oversized_mint.error().code, ErrorCode::kLifecycleRestartUnauthorized);
  const auto oversized_restriction = control->restrict({
      .allowed_targets = oversized_targets,
      .allowed_from_stages = kForwardStages,
  });
  ASSERT_FALSE(oversized_restriction);
  EXPECT_EQ(oversized_restriction.error().code, ErrorCode::kLifecycleRestartUnauthorized);
  constexpr std::array oversized_stages{
      LifecycleStage::kConfigure,
      LifecycleStage::kDiscover,
      LifecycleStage::kAllocate,
      LifecycleStage::kWarmUp,
      LifecycleStage::kArm,
      LifecycleStage::kStart,
      LifecycleStage::kConfigure,
  };
  const auto oversized_stage_mint = session.lifecycle_control({
      .allowed_targets = targets,
      .allowed_from_stages = oversized_stages,
  });
  ASSERT_FALSE(oversized_stage_mint);
  EXPECT_EQ(oversized_stage_mint.error().code, ErrorCode::kLifecycleRestartUnauthorized);

  // Every failed copy is pre-acceptance and leaves the successfully minted capability usable.
  const auto accepted = control->request_restart(*first_run);
  ASSERT_TRUE(accepted);
  LifecycleControlProbeState::allow_compute_return();
  const auto completed = control->await(*accepted, 5s);
  ASSERT_TRUE(completed);
  EXPECT_EQ(completed->state, LifecycleTransactionState::kPublished);
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
}

// REQ: RUNTIME-008
TEST(LifecycleControl, MintCopiesPolicyAndRestrictionCannotBroadenAuthority) {
  LifecycleControlProbeState::reset();
  const auto diagnostic_sink = std::make_shared<RecordingLifecycleDiagnosticSink>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(
      make_lifecycle_control_plan(), diagnostic_sink);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const OperatorRef first = plan_target(plan, 0U);
  const OperatorRef second = plan_target(plan, 1U);

  RunSession session = run_async(plan);
  const ComputeReleaseGuard release_guard;
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 1))
      << diagnostic_tokens(session);
  const auto first_run = session.operator_ref(first);
  const auto second_run = session.operator_ref(second);
  ASSERT_TRUE(first_run);
  ASSERT_TRUE(second_run);

  std::array targets{*first_run, *second_run};
  std::array stages{LifecycleStage::kConfigure, LifecycleStage::kAllocate};
  auto full = session.lifecycle_control({
      .allowed_targets = targets,
      .allowed_from_stages = stages,
  });
  ASSERT_TRUE(full);
  EXPECT_TRUE(static_cast<bool>(*full));

  LifecycleControlHandle copied = *full;
  LifecycleControlHandle assigned;
  assigned = copied;
  const LifecycleControlHandle moved = std::move(copied);
  // The moved-from invalid state is part of the public handle contract under test.
  EXPECT_FALSE(static_cast<bool>(copied));  // NOLINT(bugprone-use-after-move)
  EXPECT_TRUE(static_cast<bool>(moved));
  EXPECT_TRUE(static_cast<bool>(assigned));

  LifecycleControlHandle allocation_free_copy;
  LifecycleControlHandle allocation_free_move;
  {
    const test::ScopedAllocationFailure fail_first_allocation{0U};
    allocation_free_copy = *full;
    allocation_free_move = std::move(allocation_free_copy);
    allocation_free_copy = allocation_free_move;
  }
  EXPECT_TRUE(static_cast<bool>(allocation_free_copy));
  EXPECT_TRUE(static_cast<bool>(allocation_free_move));

  // The capability owns normalized copies; later caller storage mutations do
  // not alter the admitted authority.
  targets[0] = {};
  stages[0] = LifecycleStage::kStop;
  const std::array restricted_targets{*first_run};
  constexpr std::array restricted_stages{LifecycleStage::kConfigure};
  auto restricted = full->restrict({
      .allowed_targets = restricted_targets,
      .allowed_from_stages = restricted_stages,
  });
  ASSERT_TRUE(restricted);

  const auto accepted = restricted->request_restart(*first_run);
  ASSERT_TRUE(accepted);
  EXPECT_NE(accepted->receipt_id, 0U);
  EXPECT_NE(accepted->acceptance_sequence, 0U);
  EXPECT_EQ(accepted->target, *first_run);
  EXPECT_EQ(accepted->requested_stage, LifecycleStage::kConfigure);

  EXPECT_EQ(restricted->request_restart(*second_run).error().code,
            ErrorCode::kLifecycleRestartUnauthorized);
  EXPECT_EQ(restricted
                ->request_restart(*first_run,
                                  LifecycleRestartOptions{.from_stage = LifecycleStage::kAllocate})
                .error()
                .code,
            ErrorCode::kLifecycleRestartUnauthorized);
  constexpr std::array broadened_stages{LifecycleStage::kConfigure, LifecycleStage::kAllocate};
  EXPECT_EQ(restricted
                ->restrict({
                    .allowed_targets = restricted_targets,
                    .allowed_from_stages = broadened_stages,
                })
                .error()
                .code,
            ErrorCode::kLifecycleRestartUnauthorized);

  EXPECT_EQ(full->request_restart(*first_run,
                                  LifecycleRestartOptions{.from_stage = LifecycleStage::kStop})
                .error()
                .code,
            ErrorCode::kLifecycleInvalidRestartStage);
  EXPECT_EQ(full->request_restart(*first_run,
                                  LifecycleRestartOptions{.from_stage = LifecycleStage::kRelease})
                .error()
                .code,
            ErrorCode::kLifecycleInvalidRestartStage);
  EXPECT_EQ(full->request_restart(first).error().code, ErrorCode::kStaleOperatorRef);
  OperatorRef foreign_run = *first_run;
  ++foreign_run.run_generation;
  EXPECT_EQ(full->request_restart(foreign_run).error().code, ErrorCode::kCrossSessionOperatorRef);
  OperatorRef unknown = *first_run;
  unknown.id = 999U;
  EXPECT_EQ(full->request_restart(unknown).error().code, ErrorCode::kLifecycleRestartUnauthorized);
  const std::array foreign_targets{foreign_run};
  EXPECT_EQ(session
                .lifecycle_control({
                    .allowed_targets = foreign_targets,
                    .allowed_from_stages = restricted_stages,
                })
                .error()
                .code,
            ErrorCode::kCrossSessionOperatorRef);

  const auto queued = full->inspect(*accepted);
  ASSERT_TRUE(queued);
  EXPECT_EQ(queued->state, LifecycleTransactionState::kQueued);
  EXPECT_EQ(queued->effective_stage, LifecycleStage::kConfigure);
  EXPECT_EQ(full->await(*accepted, -1ns).error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(full->await(*accepted, 0ns).error().code, ErrorCode::kTimeout);

  LifecycleControlProbeState::allow_compute_return();
  const auto completed = full->await(*accepted, 5s);
  ASSERT_TRUE(completed);
  EXPECT_EQ(completed->state, LifecycleTransactionState::kPublished);
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);

  const auto tombstone = full->inspect(*accepted);
  ASSERT_TRUE(tombstone);
  EXPECT_TRUE(terminal(tombstone->state));
  EXPECT_EQ(tombstone->receipt.receipt_id, accepted->receipt_id);
  EXPECT_EQ(tombstone->receipt.acceptance_sequence, accepted->acceptance_sequence);
  EXPECT_EQ(tombstone->receipt.target, accepted->target);
  EXPECT_EQ(tombstone->receipt.requested_stage, accepted->requested_stage);
  EXPECT_EQ(full->request_restart(*first_run).error().code, ErrorCode::kClosed);

  const auto receipt_diagnostics = diagnostic_sink->snapshot();
  ASSERT_EQ(receipt_diagnostics.size(), 1U);
  const auto& diagnostic = receipt_diagnostics.front();
  EXPECT_EQ(diagnostic.origin, detail::RuntimeLifecycleRequestOrigin::kExternal);
  EXPECT_FALSE(diagnostic.requester.valid());
  EXPECT_EQ(diagnostic.receipt.receipt_id, accepted->receipt_id);
  EXPECT_EQ(diagnostic.receipt.acceptance_sequence, accepted->acceptance_sequence);
  EXPECT_EQ(diagnostic.receipt.target, accepted->target);
  EXPECT_EQ(diagnostic.receipt.requested_stage, accepted->requested_stage);
  EXPECT_EQ(diagnostic.state, LifecycleTransactionState::kPublished);
  EXPECT_NE(diagnostic.transaction_id, 0U);
  EXPECT_NE(diagnostic.fence_id, 0U);
  EXPECT_EQ(diagnostic.effective_stage, LifecycleStage::kConfigure);
  EXPECT_EQ(diagnostic.previous_epoch, 1U);
  EXPECT_EQ(diagnostic.candidate_epoch, 2U);
  EXPECT_EQ(diagnostic.effective_epoch, 2U);
  EXPECT_EQ(diagnostic.recovery, detail::RuntimeLifecycleRecoveryDisposition::kPublished);
}

// REQ: RUNTIME-008
TEST(LifecycleControl, RetiredHandleKeepsReceiptButReleasesPlanOwnedDiagnosticSink) {
  LifecycleControlProbeState::reset();
  std::weak_ptr<RecordingLifecycleDiagnosticSink> weak_sink;
  std::weak_ptr<const detail::ExecutionPlanState> weak_plan;
  std::weak_ptr<detail::OperatorFactoryBase> weak_operator_factory;
  std::weak_ptr<detail::ClockFactoryBase> weak_clock_factory;
  LifecycleControlHandle retained;
  LifecycleRestartReceipt retained_receipt;

  {
    auto sink = std::make_shared<RecordingLifecycleDiagnosticSink>();
    weak_sink = sink;
    const ExecutionPlan plan =
        detail::RuntimeTestPlanAccess::with_diagnostic_sink(make_lifecycle_control_plan(), sink);
    ASSERT_TRUE(plan.ok()) << plan.json();
    auto plan_state = detail::ExecutionPlanAccess::retain(plan);
    ASSERT_TRUE(plan_state);
    ASSERT_FALSE(plan_state->executable_factories.operators.empty());
    ASSERT_FALSE(plan_state->executable_factories.clocks.empty());
    weak_plan = plan_state;
    weak_operator_factory = plan_state->executable_factories.operators.front().factory;
    weak_clock_factory = plan_state->executable_factories.clocks.front().factory;
    plan_state.reset();
    const OperatorRef target = plan_target(plan, 0U);

    RunSession session = run_async(plan);
    const ComputeReleaseGuard release_guard;
    ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 1));
    const auto run_target = session.operator_ref(target);
    ASSERT_TRUE(run_target);
    const std::array targets{*run_target};
    auto control = session.lifecycle_control({
        .allowed_targets = targets,
        .allowed_from_stages = kForwardStages,
    });
    ASSERT_TRUE(control);
    retained = *control;
    const auto receipt = retained.request_restart(*run_target);
    ASSERT_TRUE(receipt);
    retained_receipt = *receipt;

    LifecycleControlProbeState::allow_compute_return();
    const auto completed = retained.await(retained_receipt, 5s);
    ASSERT_TRUE(completed);
    EXPECT_EQ(completed->state, LifecycleTransactionState::kPublished);
    session.wait();
    EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  }

  EXPECT_TRUE(weak_sink.expired());
  EXPECT_TRUE(weak_plan.expired());
  EXPECT_TRUE(weak_operator_factory.expired());
  EXPECT_TRUE(weak_clock_factory.expired());
  const auto tombstone = retained.inspect(retained_receipt);
  ASSERT_TRUE(tombstone);
  EXPECT_TRUE(terminal(tombstone->state));
  EXPECT_EQ(retained.request_restart(retained_receipt.target).error().code, ErrorCode::kClosed);
}

// REQ: RUNTIME-008
TEST(LifecycleControl, EveryForwardRestartStageExecutesItsExactLifecycleSuffix) {
  for (const LifecycleStage requested_stage : kForwardStages) {
    LifecycleControlProbeState::reset();
    const ExecutionPlan plan = make_lifecycle_control_plan();
    ASSERT_TRUE(plan.ok()) << plan.json();
    const OperatorRef first = plan_target(plan, 0U);

    RunSession session = run_async(plan);
    const ComputeReleaseGuard release_guard;
    ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 1))
        << diagnostic_tokens(session);
    const auto first_run = session.operator_ref(first);
    ASSERT_TRUE(first_run);
    const std::array targets{*first_run};
    auto control = session.lifecycle_control({
        .allowed_targets = targets,
        .allowed_from_stages = kForwardStages,
    });
    ASSERT_TRUE(control);
    const auto receipt = control->request_restart(
        *first_run, LifecycleRestartOptions{.from_stage = requested_stage});
    ASSERT_TRUE(receipt);

    LifecycleControlProbeState::allow_compute_return();
    const auto completed = control->await(*receipt, 5s);
    ASSERT_TRUE(completed);
    EXPECT_EQ(completed->state, LifecycleTransactionState::kPublished);
    EXPECT_EQ(completed->effective_stage, requested_stage);
    session.wait();
    ASSERT_EQ(session.termination(), RunTermination::kCompleted) << diagnostic_tokens(session);

    const std::vector events = LifecycleControlProbeState::snapshot();
    for (const auto& event : events) {
      if (event.kind != ProbeEventKind::kLifecycle) {
        continue;
      }
      EXPECT_EQ(event.callback_identity.context, detail::RuntimeCallbackContext::kLifecycle);
      EXPECT_EQ(event.callback_identity.operator_id,
                static_cast<OperatorInstanceId>(event.identity));
      EXPECT_EQ(event.callback_identity.lifecycle_stage, event.stage);
      EXPECT_EQ(event.callback_identity.lifecycle_phase, event.phase);
    }
    const std::size_t requested_index = static_cast<std::size_t>(requested_stage);
    EXPECT_EQ(event_count(
                  events, 1, LifecycleStage::kStop, LifecycleTransitionPhase::kPreRestartTeardown),
              1U);
    EXPECT_EQ(
        event_count(
            events, 1, LifecycleStage::kRelease, LifecycleTransitionPhase::kPreRestartTeardown),
        requested_index <= static_cast<std::size_t>(LifecycleStage::kAllocate) ? 1U : 0U);
    for (std::size_t stage_index = 0U; stage_index < kForwardStages.size(); ++stage_index) {
      EXPECT_EQ(
          event_count(
              events, 1, kForwardStages[stage_index], LifecycleTransitionPhase::kRestartForward),
          stage_index >= requested_index ? 1U : 0U)
          << "requested stage " << requested_index << ", observed stage " << stage_index;
    }
    EXPECT_TRUE(all_lifecycle_generations_are_zero(events));
  }
}

// REQ: RUNTIME-008
TEST(LifecycleControl, RejectsStaleOrTamperedCapacityEvidenceBeforeCallbacks) {
  using LifecycleMutation = detail::RuntimeTestPlanAccess::LifecycleMutation;
  constexpr std::array mutations{
      LifecycleMutation::kRestartStructuralEventCount,
      LifecycleMutation::kCapacityDerivationReason,
      LifecycleMutation::kRequiredCapacityUnderflow,
      LifecycleMutation::kRequiredCapacityOverflow,
  };

  const ExecutionPlan baseline = make_lifecycle_control_plan({.first_action = ProbeAction::kNone});
  ASSERT_TRUE(baseline.ok()) << baseline.json();
  for (const LifecycleMutation mutation : mutations) {
    SCOPED_TRACE(static_cast<unsigned>(mutation));
    const ExecutionPlan plan = detail::RuntimeTestPlanAccess::mutate_lifecycle(baseline, mutation);
    ASSERT_TRUE(plan.ok()) << plan.json();
    LifecycleControlProbeState::reset();

    RunSession session = run_async(plan);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kFailed);
    EXPECT_NE(diagnostic_tokens(session).find("RUNTIME_LIFECYCLE_PLAN_INVALID"), std::string::npos);
    EXPECT_TRUE(LifecycleControlProbeState::snapshot().empty());
  }
}

// REQ: RUNTIME-008
// REQ: PARAM-021
// REQ: PARAM-023
TEST(LifecycleControl, CapacityRejectionLeavesTheAcceptedTransactionUnchanged) {
  LifecycleControlProbeState::reset();
  const auto diagnostic_sink = std::make_shared<RecordingLifecycleDiagnosticSink>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(
      make_lifecycle_control_plan({.max_pending_requests = 2U}), diagnostic_sink);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const OperatorRef first = plan_target(plan, 0U);

  RunSession session = run_async(plan);
  const ComputeReleaseGuard release_guard;
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 1));
  const auto first_run = session.operator_ref(first);
  ASSERT_TRUE(first_run);
  const std::array targets{*first_run};
  auto control = session.lifecycle_control({
      .allowed_targets = targets,
      .allowed_from_stages = kForwardStages,
  });
  ASSERT_TRUE(control);
  const auto accepted = control->request_restart(*first_run);
  ASSERT_TRUE(accepted);
  const auto successor = control->request_restart(
      *first_run, LifecycleRestartOptions{.from_stage = LifecycleStage::kAllocate});
  ASSERT_TRUE(successor);
  const auto before = control->inspect(*accepted);
  ASSERT_TRUE(before);

  const auto rejected = control->request_restart(
      *first_run, LifecycleRestartOptions{.from_stage = LifecycleStage::kWarmUp});
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, ErrorCode::kLifecycleRestartQueueFull);
  const auto after = control->inspect(*accepted);
  ASSERT_TRUE(after);
  EXPECT_EQ(after->state, before->state);
  EXPECT_EQ(after->transaction_id, before->transaction_id);
  EXPECT_EQ(after->coalesced_into_transaction, before->coalesced_into_transaction);
  EXPECT_EQ(after->related_transaction_id, before->related_transaction_id);
  EXPECT_EQ(after->fence_id, before->fence_id);
  EXPECT_EQ(after->effective_stage, before->effective_stage);
  EXPECT_EQ(after->candidate_generation, before->candidate_generation);

  LifecycleControlProbeState::allow_compute_return();
  const auto completed = control->await(*accepted, 5s);
  ASSERT_TRUE(completed);
  EXPECT_EQ(completed->state, LifecycleTransactionState::kPublished);
  const auto successor_completed = control->await(*successor, 5s);
  ASSERT_TRUE(successor_completed);
  EXPECT_EQ(successor_completed->state, LifecycleTransactionState::kPublished);
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);

  // The rejected request has no receipt or fence and cannot disturb either exact-minimum retained
  // slot. Closing the session must publish both accepted receipts rather than silently losing
  // either one when ingress reports full capacity.
  const auto receipt_diagnostics = diagnostic_sink->snapshot();
  ASSERT_EQ(receipt_diagnostics.size(), 2U);
  EXPECT_EQ(receipt_diagnostics.front().receipt.receipt_id, accepted->receipt_id);
  EXPECT_EQ(receipt_diagnostics.front().receipt.acceptance_sequence, accepted->acceptance_sequence);
  EXPECT_EQ(receipt_diagnostics.front().receipt.target.id, accepted->target.id);
  EXPECT_EQ(receipt_diagnostics.front().receipt.requested_stage, accepted->requested_stage);
  EXPECT_EQ(receipt_diagnostics.front().state, LifecycleTransactionState::kPublished);
  EXPECT_EQ(receipt_diagnostics.front().recovery,
            detail::RuntimeLifecycleRecoveryDisposition::kPublished);
  EXPECT_EQ(receipt_diagnostics.back().receipt.receipt_id, successor->receipt_id);
  EXPECT_EQ(receipt_diagnostics.back().state, LifecycleTransactionState::kPublished);
}

// REQ: RUNTIME-008
// REQ: PARAM-021
TEST(LifecycleControl, CompatiblePreCutoffRequestsCoalesceAndExecuteOnlyOnce) {
  LifecycleControlProbeState::reset();
  const ExecutionPlan plan = make_lifecycle_control_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const OperatorRef first = plan_target(plan, 0U);

  RunSession session = run_async(plan);
  const ComputeReleaseGuard release_guard;
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 1));
  const auto first_run = session.operator_ref(first);
  ASSERT_TRUE(first_run);
  const std::array targets{*first_run};
  auto control = session.lifecycle_control({
      .allowed_targets = targets,
      .allowed_from_stages = kForwardStages,
  });
  ASSERT_TRUE(control);
  const auto first_receipt = control->request_restart(*first_run);
  const auto second_receipt = control->request_restart(*first_run);
  ASSERT_TRUE(first_receipt);
  ASSERT_TRUE(second_receipt);
  const auto first_queued = control->inspect(*first_receipt);
  const auto second_queued = control->inspect(*second_receipt);
  ASSERT_TRUE(first_queued);
  ASSERT_TRUE(second_queued);
  EXPECT_EQ(first_queued->transaction_id, second_queued->transaction_id);
  EXPECT_EQ(second_queued->coalesced_into_transaction, first_queued->transaction_id);
  EXPECT_EQ(first_queued->fence_id, second_queued->fence_id);
  EXPECT_EQ(first_queued->effective_stage, LifecycleStage::kConfigure);
  EXPECT_EQ(second_queued->effective_stage, LifecycleStage::kConfigure);

  LifecycleControlProbeState::allow_compute_return();
  const auto first_completed = control->await(*first_receipt, 5s);
  const auto second_completed = control->await(*second_receipt, 5s);
  ASSERT_TRUE(first_completed);
  ASSERT_TRUE(second_completed);
  EXPECT_EQ(first_completed->state, LifecycleTransactionState::kPublished);
  EXPECT_EQ(second_completed->state, LifecycleTransactionState::kPublished);
  session.wait();

  const std::vector events = LifecycleControlProbeState::snapshot();
  for (const LifecycleStage stage : kForwardStages) {
    EXPECT_EQ(event_count(events, 1, stage, LifecycleTransitionPhase::kRestartForward), 1U);
  }
}

// REQ: RUNTIME-008
// REQ: PARAM-021
// REQ: PARAM-023
// GoogleTest fatal assertions terminate this lifecycle test before optional access, but clang-tidy
// does not model the macro's control flow.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
TEST(LifecycleControl, PostCutoffRequestCreatesOneFenceRetainingSuccessor) {
  LifecycleControlProbeState::reset();
  LifecycleControlProbeState::block_lifecycle(
      1, LifecycleStage::kStop, LifecycleTransitionPhase::kPreRestartTeardown);
  const auto diagnostic_sink = std::make_shared<RecordingLifecycleDiagnosticSink>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(
      make_lifecycle_control_plan({.exact_active_successor_capacity = true}), diagnostic_sink);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto capacity = plan.lifecycle_transaction_capacity();
  ASSERT_TRUE(capacity.has_value());
  EXPECT_TRUE(same_capacity(capacity->configured, capacity->required_minimum));
  for (const auto& closure : plan.lifecycle_restart_closures()) {
    for (const auto& stage : closure.stage_resolutions) {
      EXPECT_EQ(stage.required_structural_event_records, 8U);
    }
  }
  const OperatorRef first = plan_target(plan, 0U);
  const OperatorRef second = plan_target(plan, 1U);

  RunSession session = run_async(plan);
  const ComputeReleaseGuard release_guard;
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 1));
  const auto first_run = session.operator_ref(first);
  const auto second_run = session.operator_ref(second);
  ASSERT_TRUE(first_run);
  ASSERT_TRUE(second_run);
  const std::array targets{*first_run, *second_run};
  auto control = session.lifecycle_control({
      .allowed_targets = targets,
      .allowed_from_stages = kForwardStages,
  });
  ASSERT_TRUE(control);
  const auto predecessor = control->request_restart(
      *first_run, LifecycleRestartOptions{.from_stage = LifecycleStage::kStart});
  ASSERT_TRUE(predecessor);

  LifecycleControlProbeState::allow_compute_return();
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_lifecycle_block());
  const auto predecessor_cutoff = control->inspect(*predecessor);
  ASSERT_TRUE(predecessor_cutoff);
  EXPECT_EQ(predecessor_cutoff->state, LifecycleTransactionState::kRestarting);

  const auto successor = control->request_restart(*first_run);
  ASSERT_TRUE(successor);
  const auto successor_queued = control->inspect(*successor);
  const auto predecessor_linked = control->inspect(*predecessor);
  ASSERT_TRUE(successor_queued);
  ASSERT_TRUE(predecessor_linked);
  EXPECT_NE(successor_queued->transaction_id, predecessor_linked->transaction_id);
  EXPECT_EQ(successor_queued->coalesced_into_transaction, 0U);
  EXPECT_EQ(successor_queued->related_transaction_id, predecessor_linked->transaction_id);
  EXPECT_EQ(predecessor_linked->related_transaction_id, successor_queued->transaction_id);
  EXPECT_NE(successor_queued->fence_id, 0U);
  EXPECT_NE(predecessor_linked->fence_id, 0U);
  EXPECT_EQ(control->request_restart(*second_run).error().code,
            ErrorCode::kLifecycleRestartQueueFull);
  const auto predecessor_after_rejection = control->inspect(*predecessor);
  const auto successor_after_rejection = control->inspect(*successor);
  ASSERT_TRUE(predecessor_after_rejection);
  ASSERT_TRUE(successor_after_rejection);
  EXPECT_EQ(predecessor_after_rejection->state, predecessor_linked->state);
  EXPECT_EQ(predecessor_after_rejection->transaction_id, predecessor_linked->transaction_id);
  EXPECT_EQ(predecessor_after_rejection->fence_id, predecessor_linked->fence_id);
  EXPECT_EQ(predecessor_after_rejection->related_transaction_id,
            predecessor_linked->related_transaction_id);
  EXPECT_EQ(successor_after_rejection->state, successor_queued->state);
  EXPECT_EQ(successor_after_rejection->transaction_id, successor_queued->transaction_id);
  EXPECT_EQ(successor_after_rejection->fence_id, successor_queued->fence_id);
  EXPECT_EQ(successor_after_rejection->related_transaction_id,
            successor_queued->related_transaction_id);

  LifecycleControlProbeState::release_lifecycle_block();
  const auto predecessor_done = control->await(*predecessor, 5s);
  const auto successor_done = control->await(*successor, 5s);
  ASSERT_TRUE(predecessor_done);
  ASSERT_TRUE(successor_done);
  EXPECT_EQ(predecessor_done->state, LifecycleTransactionState::kPublished);
  EXPECT_EQ(successor_done->state, LifecycleTransactionState::kPublished);
  session.wait();
  ASSERT_EQ(session.termination(), RunTermination::kCompleted) << diagnostic_tokens(session);

  const std::vector events = LifecycleControlProbeState::snapshot();
  EXPECT_EQ(
      event_count(events, 1, LifecycleStage::kStart, LifecycleTransitionPhase::kRestartForward),
      2U);
  for (const LifecycleStage stage : {LifecycleStage::kConfigure,
                                     LifecycleStage::kDiscover,
                                     LifecycleStage::kAllocate,
                                     LifecycleStage::kWarmUp,
                                     LifecycleStage::kArm}) {
    EXPECT_EQ(event_count(events, 1, stage, LifecycleTransitionPhase::kRestartForward), 1U);
  }

  const auto receipt_diagnostics = diagnostic_sink->snapshot();
  ASSERT_EQ(receipt_diagnostics.size(), 2U);
  EXPECT_EQ(receipt_diagnostics[0].transaction_id, predecessor_linked->transaction_id);
  EXPECT_EQ(receipt_diagnostics[1].transaction_id, successor_queued->transaction_id);
  EXPECT_EQ(receipt_diagnostics[0].related_transaction_id, receipt_diagnostics[1].transaction_id);
  EXPECT_EQ(receipt_diagnostics[1].related_transaction_id, receipt_diagnostics[0].transaction_id);
  EXPECT_NE(receipt_diagnostics[0].fence_id, receipt_diagnostics[1].fence_id);
  EXPECT_EQ(receipt_diagnostics[0].candidate_epoch, 2U);
  EXPECT_EQ(receipt_diagnostics[0].effective_epoch, 2U);
  EXPECT_EQ(receipt_diagnostics[1].previous_epoch, 2U);
  EXPECT_EQ(receipt_diagnostics[1].candidate_epoch, 3U);
  EXPECT_EQ(receipt_diagnostics[1].effective_epoch, 3U);
}

// REQ: RUNTIME-008
// NOLINTEND(bugprone-unchecked-optional-access)
TEST(LifecycleControl, ReceiptOwnershipRejectsForeignSessionsAndUnknownIdentities) {
  LifecycleControlProbeState::reset();
  const ExecutionPlan first_plan = make_lifecycle_control_plan();
  ASSERT_TRUE(first_plan.ok()) << first_plan.json();
  const OperatorRef first_target = plan_target(first_plan, 0U);
  RunSession first_session = run_async(first_plan);
  const ComputeReleaseGuard first_release_guard;
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 1));
  const auto first_run = first_session.operator_ref(first_target);
  ASSERT_TRUE(first_run);
  const std::array first_targets{*first_run};
  auto first_control = first_session.lifecycle_control({
      .allowed_targets = first_targets,
      .allowed_from_stages = kForwardStages,
  });
  ASSERT_TRUE(first_control);
  const auto first_receipt = first_control->request_restart(*first_run);
  ASSERT_TRUE(first_receipt);
  LifecycleControlProbeState::allow_compute_return();
  ASSERT_TRUE(first_control->await(*first_receipt, 5s));
  first_session.wait();
  const auto retained_tombstone = first_control->inspect(*first_receipt);
  ASSERT_TRUE(retained_tombstone);
  EXPECT_TRUE(terminal(retained_tombstone->state));

  LifecycleControlProbeState::reset();
  const ExecutionPlan second_plan = make_lifecycle_control_plan();
  ASSERT_TRUE(second_plan.ok()) << second_plan.json();
  const OperatorRef second_target = plan_target(second_plan, 0U);
  RunSession second_session = run_async(second_plan);
  const ComputeReleaseGuard second_release_guard;
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 1));
  const auto second_run = second_session.operator_ref(second_target);
  ASSERT_TRUE(second_run);
  const std::array second_targets{*second_run};
  auto second_control = second_session.lifecycle_control({
      .allowed_targets = second_targets,
      .allowed_from_stages = kForwardStages,
  });
  ASSERT_TRUE(second_control);
  EXPECT_EQ(second_control->inspect(*first_receipt).error().code,
            ErrorCode::kCrossSessionOperatorRef);
  EXPECT_EQ(second_control->await(*first_receipt, 0ns).error().code,
            ErrorCode::kCrossSessionOperatorRef);

  const auto second_receipt = second_control->request_restart(*second_run);
  ASSERT_TRUE(second_receipt);
  LifecycleRestartReceipt unknown = *second_receipt;
  unknown.receipt_id += 1000U;
  EXPECT_EQ(second_control->inspect(unknown).error().code, ErrorCode::kCrossSessionOperatorRef);
  LifecycleControlProbeState::allow_compute_return();
  ASSERT_TRUE(second_control->await(*second_receipt, 5s));
  second_session.wait();
}

// REQ: RUNTIME-008
TEST(LifecycleControl, ControlAndWaitIngressFailInsideRuntimeCallbackContexts) {
  LifecycleControlProbeState::reset();
  const ExecutionPlan plan = make_lifecycle_control_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const OperatorRef first = plan_target(plan, 0U);

  RunSession session = run_async(plan);
  const ComputeReleaseGuard release_guard;
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 1));
  const auto first_run = session.operator_ref(first);
  ASSERT_TRUE(first_run);
  const std::array targets{*first_run};
  auto control = session.lifecycle_control({
      .allowed_targets = targets,
      .allowed_from_stages = kForwardStages,
  });
  ASSERT_TRUE(control);
  const auto receipt = control->request_restart(*first_run);
  ASSERT_TRUE(receipt);

  for (const detail::RuntimeCallbackContext context : {
           detail::RuntimeCallbackContext::kCompute,
           detail::RuntimeCallbackContext::kLifecycle,
           detail::RuntimeCallbackContext::kDispatcher,
       }) {
    const detail::RuntimeCallbackContextScope callback_scope{context};
    EXPECT_EQ(control->request_restart(*first_run).error().code,
              ErrorCode::kLifecycleInvalidCallingContext);
    EXPECT_EQ(control->await(*receipt, 0ns).error().code,
              ErrorCode::kLifecycleInvalidCallingContext);
    EXPECT_TRUE(control->inspect(*receipt));
    EXPECT_TRUE(control->restrict({
        .allowed_targets = targets,
        .allowed_from_stages = kForwardStages,
    }));
  }

  LifecycleControlProbeState::allow_compute_return();
  ASSERT_TRUE(control->await(*receipt, 5s));
  session.wait();
}

// REQ: RUNTIME-008
// REQ: PARAM-021
TEST(LifecycleControl, SelfRequestWaitsForCurrentComputeRetirementAndDefaultsToConfigure) {
  LifecycleControlProbeState::reset();
  const auto diagnostic_sink = std::make_shared<RecordingLifecycleDiagnosticSink>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(
      make_lifecycle_control_plan({.first_action = ProbeAction::kRequestSelf}), diagnostic_sink);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const OperatorRef first = plan_target(plan, 0U);

  RunSession session = run_async(plan);
  const ComputeReleaseGuard release_guard;
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_request()) << diagnostic_tokens(session);
  ASSERT_TRUE(LifecycleControlProbeState::accepted());
  const auto first_run = session.operator_ref(first);
  ASSERT_TRUE(first_run);
  const std::array targets{*first_run};
  constexpr std::array stages{LifecycleStage::kConfigure};
  auto control = session.lifecycle_control({
      .allowed_targets = targets,
      .allowed_from_stages = stages,
  });
  ASSERT_TRUE(control);

  const LifecycleRestartReceipt receipt = LifecycleControlProbeState::receipt();
  EXPECT_EQ(receipt.requested_stage, LifecycleStage::kConfigure);
  const auto queued = control->inspect(receipt);
  ASSERT_TRUE(queued);
  EXPECT_EQ(queued->state, LifecycleTransactionState::kQueued);
  const std::vector before_release = LifecycleControlProbeState::snapshot();
  EXPECT_EQ(first_phase_index(before_release, 1, LifecycleTransitionPhase::kRestartForward),
            before_release.size());

  LifecycleControlProbeState::allow_compute_return();
  const auto completed = control->await(receipt, 5s);
  ASSERT_TRUE(completed);
  EXPECT_EQ(completed->state, LifecycleTransactionState::kPublished);
  session.wait();

  const std::vector events = LifecycleControlProbeState::snapshot();
  const std::size_t compute_returned =
      first_event_index(events, ProbeEventKind::kComputeReturned, 1);
  const std::size_t restart_forward =
      first_phase_index(events, 1, LifecycleTransitionPhase::kRestartForward);
  ASSERT_LT(compute_returned, events.size());
  ASSERT_LT(restart_forward, events.size());
  EXPECT_LT(compute_returned, restart_forward);

  const auto receipt_diagnostics = diagnostic_sink->snapshot();
  ASSERT_EQ(receipt_diagnostics.size(), 1U);
  EXPECT_EQ(receipt_diagnostics.front().origin,
            detail::RuntimeLifecycleRequestOrigin::kComputeSelf);
  EXPECT_EQ(receipt_diagnostics.front().requester, *first_run);
  EXPECT_EQ(receipt_diagnostics.front().receipt.target, *first_run);
}

// REQ: RUNTIME-008
// REQ: RUNTIME-054
// REQ: RUNTIME-060
TEST(LifecycleControl, RestartForwardFailureRollsBackAndRestoresThePublishedLifecycle) {
  LifecycleControlProbeState::reset();
  LifecycleControlProbeState::fail_lifecycle_once(
      1, LifecycleStage::kWarmUp, LifecycleTransitionPhase::kRestartForward);
  const auto diagnostic_sink = std::make_shared<RecordingLifecycleDiagnosticSink>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(
      make_lifecycle_control_plan(), diagnostic_sink);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const OperatorRef first = plan_target(plan, 0U);

  RunSession session = run_async(plan);
  const ComputeReleaseGuard release_guard;
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 1));
  const auto first_run = session.operator_ref(first);
  ASSERT_TRUE(first_run);
  const std::array targets{*first_run};
  auto control = session.lifecycle_control({
      .allowed_targets = targets,
      .allowed_from_stages = kForwardStages,
  });
  ASSERT_TRUE(control);
  const auto receipt = control->request_restart(*first_run);
  ASSERT_TRUE(receipt);

  LifecycleControlProbeState::allow_compute_return();
  const auto completed = control->await(*receipt, 5s);
  ASSERT_TRUE(completed);
  EXPECT_EQ(completed->state, LifecycleTransactionState::kRolledBack);
  session.wait();
  ASSERT_EQ(session.termination(), RunTermination::kCompleted) << diagnostic_tokens(session);
  const auto failure = std::find_if(
      session.diagnostics().begin(), session.diagnostics().end(), [](const Diagnostic& diagnostic) {
        return diagnostic.detail && diagnostic.detail->stage == LifecycleStage::kWarmUp &&
               diagnostic.detail->phase == LifecycleTransitionPhase::kRestartForward;
      });
  ASSERT_NE(failure, session.diagnostics().end());
  const auto& failure_detail = failure->detail;
  if (!failure_detail.has_value()) {
    FAIL() << "Expected diagnostic detail";
  }
  EXPECT_EQ(failure->severity, DiagnosticSeverity::kWarning);
  EXPECT_EQ(failure_detail->transaction_id, completed->transaction_id);
  EXPECT_FALSE(failure_detail->operator_path.empty());
  EXPECT_TRUE(failure->source.vertex);

  const std::vector events = LifecycleControlProbeState::snapshot();
  EXPECT_NE(first_phase_index(events, 1, LifecycleTransitionPhase::kStartupForward), events.size());
  EXPECT_NE(first_phase_index(events, 1, LifecycleTransitionPhase::kPreRestartTeardown),
            events.size());
  EXPECT_NE(first_phase_index(events, 1, LifecycleTransitionPhase::kRestartForward), events.size());
  EXPECT_NE(first_phase_index(events, 1, LifecycleTransitionPhase::kForwardRollback),
            events.size());
  EXPECT_NE(first_phase_index(events, 1, LifecycleTransitionPhase::kRestoreOld), events.size());
  EXPECT_NE(first_phase_index(events, 1, LifecycleTransitionPhase::kShutdownTeardown),
            events.size());
  EXPECT_EQ(
      event_count(events, 1, LifecycleStage::kRelease, LifecycleTransitionPhase::kForwardRollback),
      1U);
  for (const LifecycleStage stage : kForwardStages) {
    EXPECT_EQ(event_count(events, 1, stage, LifecycleTransitionPhase::kRestoreOld), 1U);
  }
  EXPECT_TRUE(all_lifecycle_generations_are_zero(events));

  const auto receipt_diagnostics = diagnostic_sink->snapshot();
  ASSERT_EQ(receipt_diagnostics.size(), 1U);
  EXPECT_EQ(receipt_diagnostics.front().state, LifecycleTransactionState::kRolledBack);
  EXPECT_EQ(receipt_diagnostics.front().previous_epoch, 1U);
  EXPECT_EQ(receipt_diagnostics.front().candidate_epoch, 2U);
  EXPECT_EQ(receipt_diagnostics.front().effective_epoch, 1U);
  EXPECT_EQ(receipt_diagnostics.front().recovery,
            detail::RuntimeLifecycleRecoveryDisposition::kRestoredOldEpoch);
  expect_lifecycle_failure(receipt_diagnostics.front().primary_failure,
                           1U,
                           LifecycleStage::kWarmUp,
                           LifecycleTransitionPhase::kRestartForward);
  expect_no_lifecycle_failure(receipt_diagnostics.front().rollback_failure);
  expect_no_lifecycle_failure(receipt_diagnostics.front().restoration_failure);
  expect_no_lifecycle_failure(receipt_diagnostics.front().restoration_rollback_failure);
}

// REQ: RUNTIME-008
TEST(LifecycleControl, RestartRollbackFailurePreservesTheCandidateFailureSeparately) {
  LifecycleControlProbeState::reset();
  LifecycleControlProbeState::fail_lifecycle_once(
      1, LifecycleStage::kWarmUp, LifecycleTransitionPhase::kRestartForward);
  LifecycleControlProbeState::fail_lifecycle_once(
      1, LifecycleStage::kRelease, LifecycleTransitionPhase::kForwardRollback);
  const auto diagnostic_sink = std::make_shared<RecordingLifecycleDiagnosticSink>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(
      make_lifecycle_control_plan(), diagnostic_sink);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const OperatorRef first = plan_target(plan, 0U);

  RunSession session = run_async(plan);
  const ComputeReleaseGuard release_guard;
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 1));
  const auto first_run = session.operator_ref(first);
  ASSERT_TRUE(first_run);
  const std::array targets{*first_run};
  auto control = session.lifecycle_control({
      .allowed_targets = targets,
      .allowed_from_stages = kForwardStages,
  });
  ASSERT_TRUE(control);
  const auto receipt = control->request_restart(*first_run);
  ASSERT_TRUE(receipt);

  LifecycleControlProbeState::allow_compute_return();
  const auto completed = control->await(*receipt, 5s);
  ASSERT_TRUE(completed);
  EXPECT_EQ(completed->state, LifecycleTransactionState::kFailedStopped);
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kFailed);

  const std::vector events = LifecycleControlProbeState::snapshot();
  EXPECT_EQ(first_phase_index(events, 1, LifecycleTransitionPhase::kRestoreOld), events.size());

  const auto receipt_diagnostics = diagnostic_sink->snapshot();
  ASSERT_EQ(receipt_diagnostics.size(), 1U);
  EXPECT_EQ(receipt_diagnostics.front().recovery,
            detail::RuntimeLifecycleRecoveryDisposition::kFailedStopped);
  expect_lifecycle_failure(receipt_diagnostics.front().primary_failure,
                           1U,
                           LifecycleStage::kWarmUp,
                           LifecycleTransitionPhase::kRestartForward);
  expect_lifecycle_failure(receipt_diagnostics.front().rollback_failure,
                           1U,
                           LifecycleStage::kRelease,
                           LifecycleTransitionPhase::kForwardRollback);
  expect_no_lifecycle_failure(receipt_diagnostics.front().restoration_failure);
  expect_no_lifecycle_failure(receipt_diagnostics.front().restoration_rollback_failure);
  expect_public_lifecycle_cause(session,
                                completed->transaction_id,
                                LifecycleStage::kRelease,
                                LifecycleTransitionPhase::kForwardRollback,
                                DiagnosticFailureRole::kRollback);
}

// REQ: RUNTIME-008
TEST(LifecycleControl, RestartRestorationFailureStopsTheRunAndMarksTheReceiptFailed) {
  LifecycleControlProbeState::reset();
  LifecycleControlProbeState::fail_lifecycle_once(
      1, LifecycleStage::kWarmUp, LifecycleTransitionPhase::kRestartForward);
  LifecycleControlProbeState::fail_lifecycle_once(
      1, LifecycleStage::kConfigure, LifecycleTransitionPhase::kRestoreOld);
  const auto diagnostic_sink = std::make_shared<RecordingLifecycleDiagnosticSink>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(
      make_lifecycle_control_plan(), diagnostic_sink);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const OperatorRef first = plan_target(plan, 0U);

  RunSession session = run_async(plan);
  const ComputeReleaseGuard release_guard;
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 1));
  const auto first_run = session.operator_ref(first);
  ASSERT_TRUE(first_run);
  const std::array targets{*first_run};
  auto control = session.lifecycle_control({
      .allowed_targets = targets,
      .allowed_from_stages = kForwardStages,
  });
  ASSERT_TRUE(control);
  const auto receipt = control->request_restart(*first_run);
  ASSERT_TRUE(receipt);

  LifecycleControlProbeState::allow_compute_return();
  const auto completed = control->await(*receipt, 5s);
  ASSERT_TRUE(completed);
  EXPECT_EQ(completed->state, LifecycleTransactionState::kFailedStopped);
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  const auto tokens = diagnostic_tokens(session);
  EXPECT_NE(tokens.find("RUNTIME_LIFECYCLE_CALLBACK_FAILED"), std::string::npos);

  const std::vector events = LifecycleControlProbeState::snapshot();
  EXPECT_NE(first_phase_index(events, 1, LifecycleTransitionPhase::kRestartForward), events.size());
  EXPECT_NE(first_phase_index(events, 1, LifecycleTransitionPhase::kForwardRollback),
            events.size());
  EXPECT_NE(first_phase_index(events, 1, LifecycleTransitionPhase::kRestoreOld), events.size());

  const auto receipt_diagnostics = diagnostic_sink->snapshot();
  ASSERT_EQ(receipt_diagnostics.size(), 1U);
  EXPECT_EQ(receipt_diagnostics.front().state, LifecycleTransactionState::kFailedStopped);
  EXPECT_EQ(receipt_diagnostics.front().effective_epoch, 0U);
  EXPECT_EQ(receipt_diagnostics.front().recovery,
            detail::RuntimeLifecycleRecoveryDisposition::kFailedStopped);
  expect_lifecycle_failure(receipt_diagnostics.front().primary_failure,
                           1U,
                           LifecycleStage::kWarmUp,
                           LifecycleTransitionPhase::kRestartForward);
  expect_no_lifecycle_failure(receipt_diagnostics.front().rollback_failure);
  expect_lifecycle_failure(receipt_diagnostics.front().restoration_failure,
                           1U,
                           LifecycleStage::kConfigure,
                           LifecycleTransitionPhase::kRestoreOld);
  expect_no_lifecycle_failure(receipt_diagnostics.front().restoration_rollback_failure);
  expect_public_lifecycle_cause(session,
                                completed->transaction_id,
                                LifecycleStage::kConfigure,
                                LifecycleTransitionPhase::kRestoreOld,
                                DiagnosticFailureRole::kRestoration);
}

// REQ: RUNTIME-008
TEST(LifecycleControl, RestorationRollbackFailurePreservesEveryFailureStageSeparately) {
  LifecycleControlProbeState::reset();
  LifecycleControlProbeState::fail_lifecycle_once(
      1, LifecycleStage::kConfigure, LifecycleTransitionPhase::kRestartForward);
  LifecycleControlProbeState::fail_lifecycle_once(
      1, LifecycleStage::kWarmUp, LifecycleTransitionPhase::kRestoreOld);
  LifecycleControlProbeState::fail_lifecycle_once(
      1, LifecycleStage::kRelease, LifecycleTransitionPhase::kForwardRollback);
  const auto diagnostic_sink = std::make_shared<RecordingLifecycleDiagnosticSink>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(
      make_lifecycle_control_plan(), diagnostic_sink);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const OperatorRef first = plan_target(plan, 0U);

  RunSession session = run_async(plan);
  const ComputeReleaseGuard release_guard;
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 1));
  const auto first_run = session.operator_ref(first);
  ASSERT_TRUE(first_run);
  const std::array targets{*first_run};
  auto control = session.lifecycle_control({
      .allowed_targets = targets,
      .allowed_from_stages = kForwardStages,
  });
  ASSERT_TRUE(control);
  const auto receipt = control->request_restart(*first_run);
  ASSERT_TRUE(receipt);

  LifecycleControlProbeState::allow_compute_return();
  const auto completed = control->await(*receipt, 5s);
  ASSERT_TRUE(completed);
  EXPECT_EQ(completed->state, LifecycleTransactionState::kFailedStopped);
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kFailed);

  const auto receipt_diagnostics = diagnostic_sink->snapshot();
  ASSERT_EQ(receipt_diagnostics.size(), 1U);
  EXPECT_EQ(receipt_diagnostics.front().recovery,
            detail::RuntimeLifecycleRecoveryDisposition::kFailedStopped);
  expect_lifecycle_failure(receipt_diagnostics.front().primary_failure,
                           1U,
                           LifecycleStage::kConfigure,
                           LifecycleTransitionPhase::kRestartForward);
  expect_no_lifecycle_failure(receipt_diagnostics.front().rollback_failure);
  expect_lifecycle_failure(receipt_diagnostics.front().restoration_failure,
                           1U,
                           LifecycleStage::kWarmUp,
                           LifecycleTransitionPhase::kRestoreOld);
  expect_lifecycle_failure(receipt_diagnostics.front().restoration_rollback_failure,
                           1U,
                           LifecycleStage::kRelease,
                           LifecycleTransitionPhase::kForwardRollback);
  expect_public_lifecycle_cause(session,
                                completed->transaction_id,
                                LifecycleStage::kWarmUp,
                                LifecycleTransitionPhase::kRestoreOld,
                                DiagnosticFailureRole::kRestoration);
  expect_public_lifecycle_cause(session,
                                completed->transaction_id,
                                LifecycleStage::kRelease,
                                LifecycleTransitionPhase::kForwardRollback,
                                DiagnosticFailureRole::kRollback);
}

// REQ: RUNTIME-008
TEST(LifecycleControl, StopDuringRestartCommitsTheExactExecutionOutcome) {
  enum class RestartOutcome : std::uint8_t {
    kPublished,
    kRolledBack,
    kFailedStopped,
  };
  constexpr std::array outcomes{
      RestartOutcome::kPublished,
      RestartOutcome::kRolledBack,
      RestartOutcome::kFailedStopped,
  };

  for (const RestartOutcome outcome : outcomes) {
    SCOPED_TRACE(static_cast<unsigned>(outcome));
    LifecycleControlProbeState::reset();
    if (outcome != RestartOutcome::kPublished) {
      LifecycleControlProbeState::fail_lifecycle_once(
          1, LifecycleStage::kWarmUp, LifecycleTransitionPhase::kRestartForward);
    }
    if (outcome == RestartOutcome::kFailedStopped) {
      LifecycleControlProbeState::fail_lifecycle_once(
          1, LifecycleStage::kRelease, LifecycleTransitionPhase::kForwardRollback);
    }
    LifecycleControlProbeState::block_lifecycle(
        1, LifecycleStage::kWarmUp, LifecycleTransitionPhase::kRestartForward);
    const auto diagnostic_sink = std::make_shared<RecordingLifecycleDiagnosticSink>();
    const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(
        make_lifecycle_control_plan(), diagnostic_sink);
    ASSERT_TRUE(plan.ok()) << plan.json();
    const OperatorRef first = plan_target(plan, 0U);

    RunSession session = run_async(plan);
    const ComputeReleaseGuard release_guard;
    ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 1));
    const auto first_run = session.operator_ref(first);
    ASSERT_TRUE(first_run);
    const std::array targets{*first_run};
    auto control = session.lifecycle_control({
        .allowed_targets = targets,
        .allowed_from_stages = kForwardStages,
    });
    ASSERT_TRUE(control);
    const auto receipt = control->request_restart(*first_run);
    ASSERT_TRUE(receipt);

    LifecycleControlProbeState::allow_compute_return();
    ASSERT_TRUE(LifecycleControlProbeState::wait_for_lifecycle_block());
    const auto restarting = control->inspect(*receipt);
    ASSERT_TRUE(restarting);
    EXPECT_EQ(restarting->state, LifecycleTransactionState::kRestarting);
    session.request_stop();
    LifecycleControlProbeState::release_lifecycle_block();

    const auto completed = control->await(*receipt, 5s);
    ASSERT_TRUE(completed);
    session.wait();
    const auto diagnostics = diagnostic_sink->snapshot();
    ASSERT_EQ(diagnostics.size(), 1U);
    const auto& diagnostic = diagnostics.front();

    switch (outcome) {
      case RestartOutcome::kPublished:
        EXPECT_EQ(completed->state, LifecycleTransactionState::kPublished);
        EXPECT_EQ(session.termination(), RunTermination::kStopped);
        EXPECT_EQ(diagnostic.state, LifecycleTransactionState::kPublished);
        EXPECT_EQ(diagnostic.recovery, detail::RuntimeLifecycleRecoveryDisposition::kPublished);
        EXPECT_EQ(diagnostic.effective_epoch, 2U);
        break;
      case RestartOutcome::kRolledBack:
        EXPECT_EQ(completed->state, LifecycleTransactionState::kRolledBack);
        EXPECT_EQ(session.termination(), RunTermination::kStopped);
        EXPECT_EQ(diagnostic.state, LifecycleTransactionState::kRolledBack);
        EXPECT_EQ(diagnostic.recovery,
                  detail::RuntimeLifecycleRecoveryDisposition::kRestoredOldEpoch);
        EXPECT_EQ(diagnostic.effective_epoch, 1U);
        break;
      case RestartOutcome::kFailedStopped:
        EXPECT_EQ(completed->state, LifecycleTransactionState::kFailedStopped);
        EXPECT_EQ(session.termination(), RunTermination::kFailed);
        EXPECT_EQ(diagnostic.state, LifecycleTransactionState::kFailedStopped);
        EXPECT_EQ(diagnostic.recovery, detail::RuntimeLifecycleRecoveryDisposition::kFailedStopped);
        EXPECT_EQ(diagnostic.effective_epoch, 0U);
        break;
    }
  }
}

// REQ: RUNTIME-008
TEST(LifecycleControl, StopCancelsQueuedRestartAndClosesFutureIngress) {
  LifecycleControlProbeState::reset();
  const auto diagnostic_sink = std::make_shared<RecordingLifecycleDiagnosticSink>();
  const ExecutionPlan plan = detail::RuntimeTestPlanAccess::with_diagnostic_sink(
      make_lifecycle_control_plan(), diagnostic_sink);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const OperatorRef first = plan_target(plan, 0U);

  RunSession session = run_async(plan);
  const ComputeReleaseGuard release_guard;
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 1));
  const auto first_run = session.operator_ref(first);
  ASSERT_TRUE(first_run);
  const std::array targets{*first_run};
  auto control = session.lifecycle_control({
      .allowed_targets = targets,
      .allowed_from_stages = kForwardStages,
  });
  ASSERT_TRUE(control);
  const auto receipt = control->request_restart(*first_run);
  ASSERT_TRUE(receipt);

  session.request_stop();
  const auto queued = control->inspect(*receipt);
  ASSERT_TRUE(queued);
  EXPECT_EQ(queued->state, LifecycleTransactionState::kQueued);
  EXPECT_EQ(control->request_restart(*first_run).error().code, ErrorCode::kClosed);
  LifecycleControlProbeState::allow_compute_return();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  const auto canceled = control->inspect(*receipt);
  ASSERT_TRUE(canceled);
  EXPECT_EQ(canceled->state, LifecycleTransactionState::kCanceled);

  const std::vector events = LifecycleControlProbeState::snapshot();
  EXPECT_EQ(first_phase_index(events, 1, LifecycleTransitionPhase::kRestartForward), events.size());

  const auto receipt_diagnostics = diagnostic_sink->snapshot();
  ASSERT_EQ(receipt_diagnostics.size(), 1U);
  EXPECT_EQ(receipt_diagnostics.front().state, LifecycleTransactionState::kCanceled);
  EXPECT_EQ(receipt_diagnostics.front().effective_epoch, 1U);
  EXPECT_EQ(receipt_diagnostics.front().recovery,
            detail::RuntimeLifecycleRecoveryDisposition::kCanceled);
}

// Both paths hold the first callback while the second Operator's source-time-zero candidate is
// frozen, then advance the deterministic clock beyond its five-nanosecond latest start. Stop and
// restart must cancel that candidate before the invocation gate, so neither path may publish the
// deadline miss that would be mandatory if the stale candidate reached the gate.
// REQ: RUNTIME-008
// REQ: RUNTIME-028
// REQ: RUNTIME-081
TEST(LifecycleControl, StopAndRestartFencesSuppressQueuedCandidateDeadlineGate) {
  {
    LifecycleControlProbeState::reset();
    const ExecutionPlan base = make_lifecycle_control_plan({.deadline_second = true});
    ASSERT_TRUE(base.ok()) << base.json();
    const OperatorRef second = plan_target(base, 1U);
    auto invocation_control = std::make_shared<detail::RuntimePreDispatchTestControl>(second.id);
    const PreDispatchReleaseGuard invocation_release{invocation_control};
    const ExecutionPlan plan =
        detail::RuntimeTestPlanAccess::enable_pre_dispatch_interleaving(base, invocation_control);
    ASSERT_TRUE(plan.ok()) << plan.json();
    ASSERT_EQ(plan.max_start_delays().size(), 1U);

    RunSession session = run_async(plan);
    const ComputeReleaseGuard release_guard;
    ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 1))
        << diagnostic_tokens(session);
    ASSERT_TRUE(invocation_control->wait_until_paused()) << diagnostic_tokens(session);
    LifecycleControlProbeState::clock_now_ns.store(6, std::memory_order_relaxed);
    session.request_stop();
    invocation_control->release();
    LifecycleControlProbeState::allow_compute_return();
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kStopped) << diagnostic_tokens(session);
    const std::vector events = LifecycleControlProbeState::snapshot();
    EXPECT_EQ(first_event_index(events, ProbeEventKind::kComputeEntered, 2), events.size());
    EXPECT_EQ(diagnostic_tokens(session).find("MAX_START_DELAY_MISS"), std::string::npos);
  }

  {
    LifecycleControlProbeState::reset();
    const ExecutionPlan base = make_lifecycle_control_plan({
        .first_action = ProbeAction::kRequestCross,
        .deadline_second = true,
    });
    ASSERT_TRUE(base.ok()) << base.json();
    const OperatorRef second = plan_target(base, 1U);
    auto invocation_control = std::make_shared<detail::RuntimePreDispatchTestControl>(second.id);
    const PreDispatchReleaseGuard invocation_release{invocation_control};
    const ExecutionPlan plan =
        detail::RuntimeTestPlanAccess::enable_pre_dispatch_interleaving(base, invocation_control);
    ASSERT_TRUE(plan.ok()) << plan.json();
    ASSERT_EQ(plan.max_start_delays().size(), 1U);

    RunSession session = run_async(plan);
    const ComputeReleaseGuard release_guard;
    ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 1))
        << diagnostic_tokens(session);
    ASSERT_TRUE(invocation_control->wait_until_paused()) << diagnostic_tokens(session);
    const auto second_run = session.operator_ref(second);
    ASSERT_TRUE(second_run);
    LifecycleControlProbeState::set_cross_target(*second_run);
    ASSERT_TRUE(LifecycleControlProbeState::wait_for_request()) << diagnostic_tokens(session);
    ASSERT_TRUE(LifecycleControlProbeState::accepted());

    const std::array targets{*second_run};
    constexpr std::array stages{LifecycleStage::kConfigure};
    auto control = session.lifecycle_control({
        .allowed_targets = targets,
        .allowed_from_stages = stages,
    });
    ASSERT_TRUE(control);
    const LifecycleRestartReceipt receipt = LifecycleControlProbeState::receipt();

    LifecycleControlProbeState::clock_now_ns.store(6, std::memory_order_relaxed);
    invocation_control->release();
    LifecycleControlProbeState::allow_compute_return();
    const auto completed = control->await(receipt, 5s);
    ASSERT_TRUE(completed) << diagnostic_tokens(session);
    EXPECT_EQ(completed->state, LifecycleTransactionState::kPublished);
    session.wait();

    EXPECT_EQ(session.termination(), RunTermination::kCompleted) << diagnostic_tokens(session);
    const std::vector events = LifecycleControlProbeState::snapshot();
    EXPECT_EQ(first_event_index(events, ProbeEventKind::kComputeEntered, 2), events.size());
    EXPECT_EQ(diagnostic_tokens(session).find("MAX_START_DELAY_MISS"), std::string::npos);
  }
}

// REQ: RUNTIME-054
// REQ: RUNTIME-060
// REQ: LIFECYCLE-013
// REQ: LIFECYCLE-014
TEST(LifecycleControl, SingleOwnerFatalRestartPreservesPrimaryAndRecoveryContext) {
  struct RecoveryCase {
    LifecycleStage stage;
    LifecycleTransitionPhase phase;
    DiagnosticFailureRole role;
    const char* message;
  };
  constexpr std::array recoveries{
      RecoveryCase{LifecycleStage::kRelease,
                   LifecycleTransitionPhase::kForwardRollback,
                   DiagnosticFailureRole::kRollback,
                   "candidate rollback failure"},
      RecoveryCase{LifecycleStage::kConfigure,
                   LifecycleTransitionPhase::kRestoreOld,
                   DiagnosticFailureRole::kRestoration,
                   "previous lifecycle restoration failure"},
  };
  for (const bool cross_target : {false, true}) {
    for (const auto& recovery : recoveries) {
      SCOPED_TRACE(cross_target ? "cross target" : "self target");
      SCOPED_TRACE(recovery.message);
      LifecycleControlProbeState::reset();
      const std::int32_t target_identity = cross_target ? 2 : 1;
      LifecycleControlProbeState::fail_lifecycle_once(
          target_identity, LifecycleStage::kWarmUp, LifecycleTransitionPhase::kRestartForward);
      LifecycleControlProbeState::fail_lifecycle_once(
          target_identity, recovery.stage, recovery.phase);
      Graph graph{"single-owner-fatal-restart"};
      const auto first = graph.op<LifecycleClockRestartProbeOperator>(
          "first", 1, cross_target ? ProbeAction::kRequestCross : ProbeAction::kRequestSelf);
      const auto second =
          graph.op<LifecycleClockRestartProbeOperator>("second", 2, ProbeAction::kNone);
      graph.partition("shared-clock-worker").include_all();
      graph.lifecycle()
          .allow_self_restart(first)
          .allow_external_restart(first)
          .allow_external_restart(second)
          .allow_restart_request(first, second);
      graph.set_default_clock(graph.add_clock<LifecycleControlProbeClock>("control-clock"));
      CompileOptions options;
      options.deployment.bind_thread_placement("shared-clock-worker",
                                               ThreadPlacement{.wake_mode = WakeMode::kBusyPoll});
      options.lifecycle_transactions = ample_lifecycle_capacity();
      const auto sink = std::make_shared<RecordingLifecycleDiagnosticSink>();
      const ExecutionPlan plan =
          detail::RuntimeTestPlanAccess::with_diagnostic_sink(compile(graph, options), sink);
      ASSERT_TRUE(plan.ok()) << plan.json();
      ASSERT_EQ(plan.execution_partitions().size(), 1U);
      ASSERT_EQ(plan.execution_lanes().size(), 1U);
      RunSession session = run_async(plan);
      const ComputeReleaseGuard release_guard;
      ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 1));
      const auto target = session.operator_ref(plan_target(plan, cross_target ? 1U : 0U));
      ASSERT_TRUE(target);
      if (cross_target) {
        LifecycleControlProbeState::set_cross_target(*target);
      }
      ASSERT_TRUE(LifecycleControlProbeState::wait_for_request());
      ASSERT_TRUE(LifecycleControlProbeState::accepted());
      const std::array targets{*target};
      auto control = session.lifecycle_control({
          .allowed_targets = targets,
          .allowed_from_stages = kForwardStages,
      });
      ASSERT_TRUE(control);
      const auto receipt = LifecycleControlProbeState::receipt();
      LifecycleControlProbeState::allow_compute_return();
      const auto completed = control->await(receipt, 5s);
      ASSERT_TRUE(completed);
      EXPECT_EQ(completed->state, LifecycleTransactionState::kFailedStopped);
      EXPECT_NE(completed->transaction_id, 0U);
      session.wait();
      EXPECT_EQ(session.termination(), RunTermination::kFailed);

      // Inspect after shutdown so borrowed executor captures cannot satisfy these checks.
      const auto diagnostics = session.diagnostics();
      const auto primary =
          std::find_if(diagnostics.begin(), diagnostics.end(), [](const auto& row) {
            return row.detail && row.detail->stage == LifecycleStage::kWarmUp &&
                   row.detail->phase == LifecycleTransitionPhase::kRestartForward;
          });
      const auto secondary =
          std::find_if(diagnostics.begin(), diagnostics.end(), [&](const auto& row) {
            return row.detail && row.detail->stage == recovery.stage &&
                   row.detail->phase == recovery.phase;
          });
      ASSERT_NE(primary, diagnostics.end());
      ASSERT_NE(secondary, diagnostics.end());
      EXPECT_NE(primary, secondary);
      for (const auto* record : {&*primary, &*secondary}) {
        const auto& record_detail = record->detail;
        if (!record_detail.has_value()) {
          FAIL() << "Expected diagnostic detail";
        }
        EXPECT_EQ(record->token, "RUNTIME_LIFECYCLE_CALLBACK_FAILED");
        EXPECT_EQ(record_detail->operation, DiagnosticOperation::kLifecycle);
        EXPECT_EQ(record_detail->operator_path, cross_target ? "second" : "first");
        EXPECT_EQ(record_detail->transaction_id, completed->transaction_id);
      }
      const auto& primary_detail = primary->detail;
      if (!primary_detail.has_value()) {
        FAIL() << "Expected diagnostic detail";
      }
      const auto& secondary_detail = secondary->detail;
      if (!secondary_detail.has_value()) {
        FAIL() << "Expected diagnostic detail";
      }
      EXPECT_EQ(primary_detail->error.code, ErrorCode::kInvalidArgument);
      EXPECT_EQ(primary_detail->error.message, "candidate warm-up failure");
      EXPECT_EQ(primary_detail->role, DiagnosticFailureRole::kPrimary);
      EXPECT_FALSE(primary_detail->primary_diagnostic_index);
      EXPECT_EQ(secondary_detail->error.code, ErrorCode::kFailure);
      EXPECT_EQ(secondary_detail->error.message, recovery.message);
      EXPECT_EQ(secondary_detail->role, recovery.role);
      EXPECT_EQ(secondary_detail->primary_diagnostic_index,
                static_cast<std::uint32_t>(std::distance(diagnostics.begin(), primary)));
      const auto receipts = sink->snapshot();
      ASSERT_EQ(receipts.size(), 1U);
      EXPECT_EQ(receipts.front().transaction_id, completed->transaction_id);
      EXPECT_EQ(receipts.front().state, LifecycleTransactionState::kFailedStopped);
      EXPECT_EQ(receipts.front().recovery,
                detail::RuntimeLifecycleRecoveryDisposition::kFailedStopped);
      EXPECT_EQ(receipts.front().effective_epoch, 0U);
    }
  }
}

// REQ: RUNTIME-008
// REQ: RUNTIME-054
// REQ: RUNTIME-060
TEST(LifecycleControl, ComputeFailureCancelsItsQueuedSelfRestartBeforeLifecycleEntry) {
  LifecycleControlProbeState::reset();
  const ExecutionPlan plan =
      make_lifecycle_control_plan({.first_action = ProbeAction::kRequestSelfThenFail});
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 2U);
  ASSERT_EQ(plan.execution_lanes().size(), 2U);
  const OperatorRef first = plan_target(plan, 0U);

  RunSession session = run_async(plan);
  const ComputeReleaseGuard release_guard;
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_request());
  ASSERT_TRUE(LifecycleControlProbeState::accepted());
  const auto first_run = session.operator_ref(first);
  ASSERT_TRUE(first_run);
  const std::array targets{*first_run};
  auto control = session.lifecycle_control({
      .allowed_targets = targets,
      .allowed_from_stages = kForwardStages,
  });
  ASSERT_TRUE(control);
  const LifecycleRestartReceipt receipt = LifecycleControlProbeState::receipt();

  LifecycleControlProbeState::allow_compute_return();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kFailed);
  EXPECT_EQ(diagnostic_tokens(session).find("RUNTIME_EXECUTION_LANE_PLAN_INVALID"),
            std::string::npos);
  const auto diagnostics = session.diagnostics();
  const auto failure = std::find_if(diagnostics.begin(), diagnostics.end(), [](const auto& row) {
    return row.token == "RUNTIME_OPERATOR_CALLBACK_FAILED";
  });
  ASSERT_NE(failure, diagnostics.end());
  const auto& failure_detail = failure->detail;
  if (!failure_detail.has_value()) {
    FAIL() << "Expected diagnostic detail";
  }
  EXPECT_EQ(failure_detail->operation, DiagnosticOperation::kCompute);
  EXPECT_EQ(failure_detail->operator_path, "first");
  EXPECT_EQ(failure_detail->error.code, ErrorCode::kFailure);
  EXPECT_EQ(failure_detail->error.message, "compute failure after queued self restart");
  const auto canceled = control->inspect(receipt);
  ASSERT_TRUE(canceled);
  EXPECT_EQ(canceled->state, LifecycleTransactionState::kCanceled);
  const std::vector events = LifecycleControlProbeState::snapshot();
  EXPECT_EQ(first_phase_index(events, 1, LifecycleTransitionPhase::kRestartForward), events.size());
}

// REQ: RUNTIME-008
// REQ: PARAM-021
TEST(LifecycleControl, AuthorizedCrossTargetRequestFencesIdleTargetsPreRestartCandidate) {
  LifecycleControlProbeState::reset();
  const auto diagnostic_sink = std::make_shared<RecordingLifecycleDiagnosticSink>();
  const ExecutionPlan base = detail::RuntimeTestPlanAccess::with_diagnostic_sink(
      make_lifecycle_control_plan({.first_action = ProbeAction::kRequestCross}), diagnostic_sink);
  ASSERT_TRUE(base.ok()) << base.json();
  const OperatorRef second = plan_target(base, 1U);
  auto invocation_control = std::make_shared<detail::RuntimePreDispatchTestControl>(second.id);
  const PreDispatchReleaseGuard invocation_release{invocation_control};
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::enable_pre_dispatch_interleaving(base, invocation_control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const ComputeReleaseGuard release_guard;
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 1))
      << diagnostic_tokens(session);
  ASSERT_TRUE(invocation_control->wait_until_paused()) << diagnostic_tokens(session);
  const auto second_run = session.operator_ref(second);
  ASSERT_TRUE(second_run);
  LifecycleControlProbeState::set_cross_target(*second_run);
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_request());
  ASSERT_TRUE(LifecycleControlProbeState::accepted());

  const std::array targets{*second_run};
  constexpr std::array stages{LifecycleStage::kConfigure};
  auto control = session.lifecycle_control({
      .allowed_targets = targets,
      .allowed_from_stages = stages,
  });
  ASSERT_TRUE(control);
  const LifecycleRestartReceipt receipt = LifecycleControlProbeState::receipt();
  ASSERT_EQ(receipt.target, *second_run);

  invocation_control->release();
  LifecycleControlProbeState::allow_compute_return();
  const auto completed = control->await(receipt, 5s);
  ASSERT_TRUE(completed);
  EXPECT_EQ(completed->state, LifecycleTransactionState::kPublished);
  session.wait();

  const std::vector events = LifecycleControlProbeState::snapshot();
  const std::size_t first_compute_return =
      first_event_index(events, ProbeEventKind::kComputeReturned, 1);
  const std::size_t target_restart =
      first_phase_index(events, 2, LifecycleTransitionPhase::kRestartForward);
  const std::size_t target_compute = first_event_index(events, ProbeEventKind::kComputeEntered, 2);
  ASSERT_LT(first_compute_return, events.size());
  ASSERT_LT(target_restart, events.size());
  EXPECT_LT(first_compute_return, target_restart);
  EXPECT_EQ(target_compute, events.size());

  const auto receipt_diagnostics = diagnostic_sink->snapshot();
  ASSERT_EQ(receipt_diagnostics.size(), 1U);
  EXPECT_EQ(receipt_diagnostics.front().origin,
            detail::RuntimeLifecycleRequestOrigin::kComputeCrossTarget);
  EXPECT_EQ(receipt_diagnostics.front().requester.id, plan_target(plan, 0U).id);
  EXPECT_EQ(receipt_diagnostics.front().receipt.target, *second_run);
}

// REQ: RUNTIME-004
// REQ: RUNTIME-008
// REQ: RUNTIME-012
TEST(LifecycleControl, CrossTargetRestartFencesFrozenCandidateBeforePublishingNewGeneration) {
  LifecycleControlProbeState::reset();
  const ExecutionPlan base = make_lifecycle_clock_restart_plan();
  ASSERT_TRUE(base.ok()) << base.json();
  const OperatorRef second = plan_target(base, 1U);
  auto invocation_control = std::make_shared<detail::RuntimePreDispatchTestControl>(second.id);
  const PreDispatchReleaseGuard invocation_release{invocation_control};
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::enable_pre_dispatch_interleaving(base, invocation_control);
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const ComputeReleaseGuard release_guard;
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 1));
  ASSERT_TRUE(invocation_control->wait_until_paused()) << diagnostic_tokens(session);
  const auto second_run = session.operator_ref(second);
  ASSERT_TRUE(second_run);
  LifecycleControlProbeState::set_cross_target(*second_run);
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_request());
  ASSERT_TRUE(LifecycleControlProbeState::accepted());

  const std::array targets{*second_run};
  constexpr std::array stages{LifecycleStage::kConfigure};
  auto control = session.lifecycle_control({
      .allowed_targets = targets,
      .allowed_from_stages = stages,
  });
  ASSERT_TRUE(control);
  const LifecycleRestartReceipt receipt = LifecycleControlProbeState::receipt();

  invocation_control->release();
  LifecycleControlProbeState::allow_compute_return();
  const auto completed = control->await(receipt, 5s);
  ASSERT_TRUE(completed);
  EXPECT_EQ(completed->state, LifecycleTransactionState::kPublished);
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 2));
  session.request_stop();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);

  const std::vector events = LifecycleControlProbeState::snapshot();
  std::vector<Tag> target_tags;
  for (const auto& event : events) {
    if (event.kind == ProbeEventKind::kComputeEntered && event.identity == 2) {
      target_tags.push_back(event.tag);
    }
  }
  ASSERT_EQ(target_tags.size(), 1U);
  EXPECT_EQ(target_tags.front(), (Tag{1U, 2U, 0U}));

  const std::size_t requester_return =
      first_event_index(events, ProbeEventKind::kComputeReturned, 1);
  const std::size_t target_restart =
      first_phase_index(events, 2, LifecycleTransitionPhase::kRestartForward);
  const std::size_t target_compute = first_event_index(events, ProbeEventKind::kComputeEntered, 2);
  ASSERT_LT(requester_return, events.size());
  ASSERT_LT(target_restart, events.size());
  ASSERT_LT(target_compute, events.size());
  EXPECT_LT(requester_return, target_restart);
  EXPECT_LT(target_restart, target_compute);
}

// REQ: RUNTIME-004
// REQ: RUNTIME-008
// REQ: RUNTIME-012
// REQ: RUNTIME-021
TEST(LifecycleControl, DependencyClosureRestartFencesEveryMemberAndPreservesUnrelatedGeneration) {
  LifecycleControlProbeState::reset();
  const ExecutionPlan plan = make_lifecycle_dependency_closure_restart_plan();
  ASSERT_TRUE(plan.ok()) << plan.json();
  const OperatorRef target = plan_target(plan, 1U);
  const OperatorRef unrelated = plan_target(plan, 2U);
  const OperatorRef dependent = plan_target(plan, 3U);
  const auto closure = std::find_if(
      plan.lifecycle_restart_closures().begin(),
      plan.lifecycle_restart_closures().end(),
      [&](const LifecycleRestartClosurePlanRecord& record) { return record.target == target; });
  ASSERT_NE(closure, plan.lifecycle_restart_closures().end());
  ASSERT_EQ(closure->closure.size(), 2U);
  EXPECT_NE(std::find(closure->closure.begin(), closure->closure.end(), target),
            closure->closure.end());
  EXPECT_NE(std::find(closure->closure.begin(), closure->closure.end(), dependent),
            closure->closure.end());
  EXPECT_EQ(std::find(closure->closure.begin(), closure->closure.end(), unrelated),
            closure->closure.end());

  RunSession session = run_async(plan);
  const ComputeReleaseGuard release_guard;
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 1))
      << diagnostic_tokens(session);
  const auto target_run = session.operator_ref(target);
  ASSERT_TRUE(target_run);
  LifecycleControlProbeState::set_cross_target(*target_run);
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_request());
  ASSERT_TRUE(LifecycleControlProbeState::accepted());

  const std::array targets{*target_run};
  constexpr std::array stages{LifecycleStage::kConfigure};
  auto control = session.lifecycle_control({
      .allowed_targets = targets,
      .allowed_from_stages = stages,
  });
  ASSERT_TRUE(control);
  const LifecycleRestartReceipt receipt = LifecycleControlProbeState::receipt();

  LifecycleControlProbeState::allow_compute_return();
  const auto completed = control->await(receipt, 5s);
  ASSERT_TRUE(completed);
  EXPECT_EQ(completed->state, LifecycleTransactionState::kPublished);
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_compute_generation(2, 2U));
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_compute_generation(3, 2U));
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 4));
  session.request_stop();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);

  const std::vector events = LifecycleControlProbeState::snapshot();
  std::vector<std::pair<std::int32_t, std::uint32_t>> compute_generations;
  const auto compute_generation_index = [&](std::int32_t identity, std::uint32_t generation) {
    const auto found = std::find_if(events.begin(), events.end(), [=](const ProbeEvent& candidate) {
      return candidate.kind == ProbeEventKind::kComputeEntered && candidate.identity == identity &&
             candidate.tag.generation == generation;
    });
    return found == events.end() ? events.size()
                                 : static_cast<std::size_t>(std::distance(events.begin(), found));
  };
  for (const ProbeEvent& event : events) {
    if (event.kind != ProbeEventKind::kComputeEntered) {
      continue;
    }
    ASSERT_TRUE(event.tag.valid());
    if (event.identity == 2 || event.identity == 3 || event.identity == 4) {
      const std::pair observed{event.identity, event.tag.generation};
      if (std::find(compute_generations.begin(), compute_generations.end(), observed) ==
          compute_generations.end()) {
        compute_generations.push_back(observed);
      }
    }
    if (event.identity == 2 || event.identity == 3) {
      EXPECT_TRUE(event.tag.generation == 1U || event.tag.generation == 2U)
          << "closure member " << event.identity << " computed in unexpected generation "
          << event.tag.generation;
    }
    if (event.identity == 4) {
      EXPECT_EQ(event.tag.generation, 1U);
    }
  }
  // The requester can publish the restart before either independent closure-member lane performs
  // its first activation. Generation one is therefore optional; generation two proves that both
  // closure members crossed the fence, while the unrelated Operator must remain in generation one.
  constexpr std::array required_compute_generations{
      std::pair{2, 2U}, std::pair{3, 2U}, std::pair{4, 1U}};
  for (const auto& required : required_compute_generations) {
    EXPECT_NE(std::find(compute_generations.begin(), compute_generations.end(), required),
              compute_generations.end());
  }
  EXPECT_EQ(std::count_if(compute_generations.begin(),
                          compute_generations.end(),
                          [](const auto& observed) { return observed.first == 4; }),
            1);
  const std::size_t target_restart =
      first_phase_index(events, 2, LifecycleTransitionPhase::kRestartForward);
  const std::size_t dependent_restart =
      first_phase_index(events, 3, LifecycleTransitionPhase::kRestartForward);
  const std::size_t target_generation_two = compute_generation_index(2, 2U);
  const std::size_t dependent_generation_two = compute_generation_index(3, 2U);
  ASSERT_LT(target_restart, events.size());
  ASSERT_LT(dependent_restart, events.size());
  ASSERT_LT(target_generation_two, events.size());
  ASSERT_LT(dependent_generation_two, events.size());
  EXPECT_LT(target_restart, target_generation_two);
  EXPECT_LT(dependent_restart, dependent_generation_two);
  EXPECT_EQ(
      event_count(events, 2, LifecycleStage::kStart, LifecycleTransitionPhase::kRestartForward),
      1U);
  EXPECT_EQ(
      event_count(events, 3, LifecycleStage::kStart, LifecycleTransitionPhase::kRestartForward),
      1U);
  EXPECT_EQ(
      event_count(events, 4, LifecycleStage::kStart, LifecycleTransitionPhase::kRestartForward),
      0U);
}

// REQ: RUNTIME-008
TEST(LifecycleControl, UndeclaredCrossTargetRequestFailsWithoutCreatingReceipt) {
  LifecycleControlProbeState::reset();
  // Key setup: allow_cross_request=false removes the allow_restart_request(first, second)
  // declaration so the cross-target request from first's compute is unauthorized.
  const ExecutionPlan plan = make_lifecycle_control_plan({
      .first_action = ProbeAction::kRequestCross,
      .allow_cross_request = false,
  });
  ASSERT_TRUE(plan.ok()) << plan.json();
  const OperatorRef second = plan_target(plan, 1U);

  RunSession session = run_async(plan);
  const ComputeReleaseGuard release_guard;
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_event(ProbeEventKind::kComputeEntered, 1))
      << diagnostic_tokens(session);
  const auto second_run = session.operator_ref(second);
  ASSERT_TRUE(second_run);
  LifecycleControlProbeState::set_cross_target(*second_run);
  ASSERT_TRUE(LifecycleControlProbeState::wait_for_request());
  // Key assertion: request rejected with kLifecycleRestartUnauthorized and no receipt allocated.
  EXPECT_FALSE(LifecycleControlProbeState::accepted());
  EXPECT_EQ(LifecycleControlProbeState::error(), ErrorCode::kLifecycleRestartUnauthorized);
  EXPECT_EQ(LifecycleControlProbeState::receipt().receipt_id, 0U);

  LifecycleControlProbeState::allow_compute_return();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
}

}  // namespace
}  // namespace holoscan
