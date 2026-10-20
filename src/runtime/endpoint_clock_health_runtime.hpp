/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include <holoscan/channel/clock_health.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/time/clock_registry.hpp>

#include "../time/clock_registry_internal.hpp"
#include "endpoint_clock_health.hpp"

namespace holoscan::detail {

/// Identify the next generation-qualified clock-health expiry in one session.
struct EndpointClockHealthRuntimeDeadline {
  PortId endpoint{};
  ClockId clock{};
  EndpointClockHealthExpiryTicket ticket{};
};

/// Describe one externally quiesced receiver reset in a coordinator-wide transaction.
struct EndpointClockHealthResetRequest {
  PortId endpoint{};
  EndpointClockHealthUpdateOrigin origin{EndpointClockHealthUpdateOrigin::kAttachmentReset};
  EndpointClockHealthBindingIdentity binding{};
  std::optional<std::uint64_t> cause_sequence;
};

/// Materialize and coordinate the bounded endpoint clock-health receivers for one RunSession.
///
/// Construction is cold and may allocate the exact receiver and scratch arrays. Every operation
/// after construction is externally serialized, nonblocking, and allocation-free. The coordinator
/// reads only plan-admitted independent age clocks. It never emits a readiness event, so a health
/// report or expiry cannot cause an Operator activation by itself.
class EndpointClockHealthRuntimeCoordinator {
 public:
  EndpointClockHealthRuntimeCoordinator(const EndpointClockHealthRuntimeCoordinator&) = delete;
  EndpointClockHealthRuntimeCoordinator& operator=(const EndpointClockHealthRuntimeCoordinator&) =
      delete;
  EndpointClockHealthRuntimeCoordinator(EndpointClockHealthRuntimeCoordinator&&) noexcept = default;
  EndpointClockHealthRuntimeCoordinator& operator=(
      EndpointClockHealthRuntimeCoordinator&&) noexcept = default;
  ~EndpointClockHealthRuntimeCoordinator() = default;

  /// Validate plan/catalog provenance and allocate only monitored endpoint receivers.
  [[nodiscard]] static expected<std::unique_ptr<EndpointClockHealthRuntimeCoordinator>,
                                EndpointClockHealthError>
  create(std::span<const EndpointClockHealthPlanRecord> health_plan,
         std::span<const ClockPlanRecord> clock_plan, PlanGenerationId plan_generation,
         RunGenerationId run_generation, const ClockRegistry& clock_registry,
         std::uint64_t initial_producer_epoch = 1U,
         std::uint64_t initial_clock_epoch = 1U) noexcept;

  /// Return the monitored receiver for one endpoint, or null for an ordinary/unknown endpoint.
  [[nodiscard]] EndpointClockHealthReceiver* receiver(PortId endpoint) noexcept;
  [[nodiscard]] const EndpointClockHealthReceiver* receiver(PortId endpoint) const noexcept;
  [[nodiscard]] std::size_t receiver_count() const noexcept { return slots_.size(); }

  /// Enqueue one public report without reading the age clock.
  [[nodiscard]] expected<void, EndpointClockHealthError> submit_report(
      PortId endpoint, ClockHealthReport report, std::uint64_t producer_epoch,
      std::uint64_t publisher_operation_sequence, std::uint64_t report_sequence) noexcept;

  /// Decode and enqueue one fixed private report without accepting unknown enum values.
  [[nodiscard]] expected<void, EndpointClockHealthError> submit_encoded(
      PortId endpoint, EndpointClockHealthControlRecord record) noexcept;

  /// Apply one endpoint's bounded pending-report budget at its current age-clock coordinate.
  [[nodiscard]] expected<std::size_t, EndpointClockHealthError> dispatch_pending(
      PortId endpoint) noexcept;

  /// Expire every currently due ticket after first reading each active ticket's age clock.
  ///
  /// Clock reads complete before receiver state changes, so one read failure cannot leave a
  /// partially expired session pass. The half-open rule expires a ticket exactly when
  /// `now >= expiry_deadline_ns`.
  [[nodiscard]] expected<std::size_t, EndpointClockHealthError> expire_due() noexcept;

  /// Return the earliest current expiry in one clock domain, ordered by deadline then endpoint.
  ///
  /// Timestamps from different clock domains are intentionally never compared numerically.
  [[nodiscard]] std::optional<EndpointClockHealthRuntimeDeadline> next_expiry(
      ClockId clock) const noexcept;

  /// Reset one monitored receiver after the caller has proven endpoint quiescence.
  [[nodiscard]] expected<void, EndpointClockHealthError> reset(
      PortId endpoint, EndpointClockHealthUpdateOrigin origin,
      EndpointClockHealthBindingIdentity binding,
      std::optional<std::uint64_t> cause_sequence = std::nullopt) noexcept;

  /// Preflight and atomically commit a finite set of externally quiesced receiver resets.
  ///
  /// Every endpoint, binding transition, age-clock read, ingress state, update slot, and
  /// non-wrapping sequence is validated before any receiver is changed. Requests must name
  /// distinct monitored endpoints. Calls are externally serialized with report dispatch.
  [[nodiscard]] expected<void, EndpointClockHealthError> reset_batch(
      std::span<const EndpointClockHealthResetRequest> requests) noexcept;

  /// Record a separately detected independent age-clock fault.
  [[nodiscard]] expected<void, EndpointClockHealthError> record_age_clock_fault(
      PortId endpoint, EndpointHealthAgeClockFault fault,
      std::optional<std::uint64_t> cause_sequence = std::nullopt) noexcept;

  /// Remove the oldest structural update for one monitored endpoint.
  [[nodiscard]] std::optional<EndpointClockHealthUpdateRecord> pop_update(PortId endpoint) noexcept;

 private:
  struct Slot {
    ClockId age_clock{};
    ClockRef age_clock_ref;
    EndpointClockHealthReceiver receiver;
  };

  EndpointClockHealthRuntimeCoordinator() = default;

  [[nodiscard]] Slot* find_slot(PortId endpoint) noexcept;
  [[nodiscard]] const Slot* find_slot(PortId endpoint) const noexcept;
  [[nodiscard]] static expected<std::int64_t, EndpointClockHealthError> read_age_time(
      const Slot& slot) noexcept;

  std::vector<Slot> slots_;
  std::vector<std::int64_t> expiry_time_scratch_;
};

}  // namespace holoscan::detail
