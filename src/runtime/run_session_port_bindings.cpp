/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "run_session_port_bindings.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <utility>

namespace holoscan::detail {

// Physical and test-only readiness-source binding construction.

namespace {

/// Retain at most one finite test witness so arbitrary test counts do not exceed source capacity.
class TestDirectPortBinding final : public PortBinding {
 public:
  TestDirectPortBinding(const RuntimePortBindingRequest& request, const TimePoint coordinate,
                        const std::uint32_t witness_count) noexcept
      : source_(request.source),
        port_(request.port),
        coordinate_(coordinate),
        kind_(request.kind),
        source_generation_(request.source_generation),
        next_sequence_(request.initial_source_sequence),
        witness_budget_(witness_count),
        remaining_(witness_count),
        capacity_(request.capacity) {
    publish_next();
  }

  [[nodiscard]] expected<std::size_t, Error> drain(
      const std::span<RuntimeEvent> destination) noexcept override {
    if (!pending_ || destination.empty()) {
      return 0U;
    }
    destination.front() = event_;
    pending_ = false;
    outstanding_sequence_ = event_.source_sequence;
    return 1U;
  }

  [[nodiscard]] SourceId source() const noexcept override { return source_; }

  [[nodiscard]] expected<void, Error> retire(
      const std::uint64_t source_sequence) noexcept override {
    if (outstanding_sequence_ == 0U || source_sequence != outstanding_sequence_) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
    outstanding_sequence_ = 0U;
    if (remaining_ == 0U) {
      closed_ = true;
    } else {
      publish_next();
    }
    return {};
  }

  void cancel() noexcept override {
    pending_ = false;
    outstanding_sequence_ = 0U;
    remaining_ = 0U;
    closed_ = true;
  }

  [[nodiscard]] expected<std::uint64_t, Error> advance_source_generation() noexcept override {
    if (!closed_ || pending_ || outstanding_sequence_ != 0U || remaining_ != 0U) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
    if (source_generation_ == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    ++source_generation_;
    next_sequence_ = 0U;
    remaining_ = witness_budget_;
    closed_ = false;
    publish_next();
    return source_generation_;
  }

  [[nodiscard]] std::size_t capacity() const noexcept override { return capacity_; }
  [[nodiscard]] bool closed() const noexcept override { return closed_; }

 private:
  void publish_next() noexcept {
    ++next_sequence_;
    --remaining_;
    event_ = RuntimeEvent{
        .kind = kind_,
        .source = source_,
        .port = port_,
        .source_decision_time = coordinate_,
        .reference_time = coordinate_,
        .source_generation = source_generation_,
        .source_sequence = next_sequence_,
    };
    pending_ = true;
  }

  SourceId source_{};
  PortId port_{};
  TimePoint coordinate_{};
  RuntimeEventKind kind_{RuntimeEventKind::kSpin};
  std::uint64_t source_generation_{};
  std::uint64_t next_sequence_{};
  std::uint64_t outstanding_sequence_{};
  std::uint32_t witness_budget_{};
  std::uint32_t remaining_{};
  std::size_t capacity_{};
  RuntimeEvent event_{};
  bool pending_{};
  bool closed_{};
};

}  // namespace

expected<std::unique_ptr<ClockTriggerPortBinding>, Error> ClockTriggerPortBinding::create(
    RuntimePortBindingRequest request, ClockRef clock, const TimePoint phase,
    const std::chrono::nanoseconds period, const bool recapture_phase_on_restart) {
  if (request.plan_generation == 0U || request.run_generation == 0U ||
      request.kind != RuntimeEventKind::kClockTick || !request.source || request.port ||
      request.source_clock != clock.id() || request.reference_clock != clock.id() ||
      request.source_to_reference_conversion_id != 0U || request.capacity != 1U ||
      request.source_generation == 0U || request.initial_source_sequence != 0U || !phase.valid() ||
      phase.clock_id != clock.id() || period.count() <= 0) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "RUNTIME_CLOCK_BINDING_UNAVAILABLE"});
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    return std::unique_ptr<ClockTriggerPortBinding>{new ClockTriggerPortBinding{
        request, std::move(clock), phase, period, recapture_phase_on_restart}};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    return make_unexpected(
        Error{ErrorCode::kResourceExhausted, "RUNTIME_CLOCK_BINDING_UNAVAILABLE"});
  }
#endif
}

expected<std::size_t, Error> ClockTriggerPortBinding::drain(
    const std::span<RuntimeEvent> destination) noexcept {
  if (closed_) {
    return 0U;
  }
  if (terminal_error_) {
    return make_unexpected(*terminal_error_);
  }
  if (destination.empty()) {
    return 0U;
  }
  auto now = clock_.now();
  if (!now) {
    return make_unexpected(std::move(now.error()));
  }
  auto due = schedule_.due(now.value());
  if (!due) {
    terminal_error_ = due.error();
    return make_unexpected(std::move(due.error()));
  }
  const auto firing = due.value();
  if (!firing) {
    return 0U;
  }
  const TimePoint scheduled_time = firing->scheduled_time;
  if (source_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
    terminal_error_ = Error{ErrorCode::kResourceExhausted};
    return make_unexpected(*terminal_error_);
  }
  ++source_sequence_;
  destination.front() = RuntimeEvent{
      .kind = RuntimeEventKind::kClockTick,
      .source = request_.source,
      .port = {},
      .source_decision_time = scheduled_time,
      .reference_time = scheduled_time,
      .source_generation = request_.source_generation,
      .source_sequence = source_sequence_,
  };
  if (scheduled_time.timestamp_ns > std::numeric_limits<std::int64_t>::max() - period_.count()) {
    next_wake_.reset();
    terminal_error_ = Error{ErrorCode::kResourceExhausted};
  } else {
    next_wake_ = TimePoint{scheduled_time.timestamp_ns + period_.count(), clock_.id()};
  }
  return 1U;
}

expected<std::optional<TimePoint>, Error> ClockTriggerPortBinding::next_reference() const noexcept {
  auto now = clock_.now();
  if (!now) {
    return make_unexpected(std::move(now).error());
  }
  return schedule_.peek_scheduled(now.value());
}

expected<TimePoint, Error> ClockTriggerPortBinding::restart_phase() const noexcept {
  return recapture_phase_on_restart_ ? clock_.now() : expected<TimePoint, Error>{explicit_phase_};
}

std::uint64_t ClockTriggerPortBinding::publish_restart(const TimePoint phase) noexcept {
  ++request_.source_generation;
  source_sequence_ = 0U;
  schedule_ = CoalescingClockSchedule{phase, period_};
  next_wake_ = phase;
  terminal_error_.reset();
  closed_ = false;
  return request_.source_generation;
}

ClockTriggerPortBinding::ClockTriggerPortBinding(RuntimePortBindingRequest request, ClockRef clock,
                                                 const TimePoint phase,
                                                 const std::chrono::nanoseconds period,
                                                 const bool recapture_phase_on_restart) noexcept
    : request_(request),
      clock_(std::move(clock)),
      explicit_phase_(phase),
      period_(period),
      schedule_(phase, period),
      next_wake_(phase),
      recapture_phase_on_restart_(recapture_phase_on_restart) {}

expected<std::unique_ptr<PortBinding>, Error> TestDirectPortBindingFactory::bind(
    const RuntimePortBindingRequest& request, const TimePoint start_coordinate) noexcept {
  if (witness_count_ == 0U || request.plan_generation == 0U || request.run_generation == 0U ||
      request.kind != RuntimeEventKind::kSpin || !request.source || !request.source_clock.valid() ||
      request.source_clock != request.reference_clock ||
      request.source_to_reference_conversion_id != 0U || request.capacity == 0U ||
      request.source_generation == 0U || start_coordinate.clock_id != request.reference_clock ||
      request.initial_source_sequence >
          std::numeric_limits<std::uint64_t>::max() - witness_count_) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "RUNTIME_READINESS_BINDING_UNAVAILABLE"});
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    return std::unique_ptr<PortBinding>{
        std::make_unique<TestDirectPortBinding>(request, start_coordinate, witness_count_)};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    return make_unexpected(
        Error{ErrorCode::kResourceExhausted, "RUNTIME_READINESS_BINDING_UNAVAILABLE"});
  }
#endif
}

}  // namespace holoscan::detail
