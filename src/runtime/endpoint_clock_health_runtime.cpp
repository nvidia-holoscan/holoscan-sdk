/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "endpoint_clock_health_runtime.hpp"

#include <algorithm>
#include <memory>
#include <new>
#include <tuple>
#include <utility>

namespace holoscan::detail {
namespace {

[[nodiscard]] unexpected<EndpointClockHealthError> coordinator_error(
    ErrorCode code, std::string_view token) noexcept {
  return make_unexpected(EndpointClockHealthError{code, token});
}

[[nodiscard]] const ClockPlanRecord* find_clock(std::span<const ClockPlanRecord> clocks,
                                                ClockId id) noexcept {
  if (!id.valid() || id.value > clocks.size()) {
    return nullptr;
  }
  const ClockPlanRecord& record = clocks[static_cast<std::size_t>(id.value - 1U)];
  return record.id == id ? &record : nullptr;
}

[[nodiscard]] bool ordinary_record_valid(const EndpointClockHealthPlanRecord& record,
                                         PlanGenerationId plan_generation) noexcept {
  return record.endpoint && record.plan_generation == plan_generation &&
         record.capture_conversion_id == 0U && record.conversion_generation == 0U &&
         record.producer_epoch_source == EndpointProducerEpochSource::kNotApplicable &&
         record.capability == ClockHealthCapability::kUnavailable &&
         record.requirement == ClockHealthRequirement::kNotRequired &&
         !record.max_report_age.has_value() && !record.health_age_clock.valid() &&
         record.fixed_report_ingress_capacity == 0U && record.fixed_update_capacity == 0U &&
         record.ingress_capacity_origin == EndpointClockHealthBoundOrigin::kNotApplicable &&
         record.per_dispatch_report_budget == 0U &&
         record.publisher_operation_sequence_id.empty() && record.report_sequence_id.empty() &&
         record.health_update_sequence_id.empty() &&
         record.monitor_path_kind == EndpointClockHealthMonitorPathKind::kNotApplicable &&
         !record.max_monitor_period.has_value() &&
         !record.max_observation_to_call_delay.has_value() &&
         !record.max_report_delivery_delay.has_value() &&
         !record.admitted_jitter_margin.has_value() && record.monitor_evidence_reference.empty() &&
         record.no_report_disposition == EndpointClockHealthDisposition::kNotApplicable &&
         record.degraded_disposition == EndpointClockHealthDisposition::kNotApplicable &&
         record.unhealthy_disposition == EndpointClockHealthDisposition::kNotApplicable &&
         record.stale_disposition == EndpointClockHealthDisposition::kNotApplicable &&
         record.clock_health_update_witness_id.empty() &&
         record.trace_loss_disposition == EndpointClockHealthTraceLossDisposition::kNotApplicable;
}

[[nodiscard]] bool deadline_less(const EndpointClockHealthRuntimeDeadline& left,
                                 const EndpointClockHealthRuntimeDeadline& right) noexcept {
  return std::tuple{left.ticket.expiry_deadline_ns, left.endpoint.value()} <
         std::tuple{right.ticket.expiry_deadline_ns, right.endpoint.value()};
}

}  // namespace

expected<std::unique_ptr<EndpointClockHealthRuntimeCoordinator>, EndpointClockHealthError>
EndpointClockHealthRuntimeCoordinator::create(
    std::span<const EndpointClockHealthPlanRecord> health_plan,
    std::span<const ClockPlanRecord> clock_plan, PlanGenerationId plan_generation,
    RunGenerationId run_generation, const ClockRegistry& clock_registry,
    std::uint64_t initial_producer_epoch, std::uint64_t initial_clock_epoch) noexcept {
  if (plan_generation == 0U || run_generation == 0U || initial_producer_epoch == 0U ||
      initial_clock_epoch == 0U) {
    return coordinator_error(ErrorCode::kInvalidArgument,
                             "ENDPOINT_CLOCK_HEALTH_RUNTIME_PLAN_INVALID");
  }

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    auto result = std::unique_ptr<EndpointClockHealthRuntimeCoordinator>{
        new EndpointClockHealthRuntimeCoordinator{}};
    const std::size_t monitored_count = static_cast<std::size_t>(std::count_if(
        health_plan.begin(), health_plan.end(), [](const EndpointClockHealthPlanRecord& record) {
          return record.capability == ClockHealthCapability::kStatusReports;
        }));
    result->slots_.reserve(monitored_count);
    result->expiry_time_scratch_.resize(monitored_count);

    PortId previous_endpoint{};
    for (const EndpointClockHealthPlanRecord& record : health_plan) {
      if (!record.endpoint ||
          (previous_endpoint && record.endpoint.value() <= previous_endpoint.value())) {
        return coordinator_error(ErrorCode::kInvalidArgument,
                                 "ENDPOINT_CLOCK_HEALTH_RUNTIME_PLAN_INVALID");
      }
      previous_endpoint = record.endpoint;

      if (record.capability != ClockHealthCapability::kStatusReports) {
        if (!ordinary_record_valid(record, plan_generation)) {
          return coordinator_error(ErrorCode::kInvalidArgument,
                                   "ENDPOINT_CLOCK_HEALTH_RUNTIME_PLAN_INVALID");
        }
        continue;
      }
      if (record.plan_generation != plan_generation) {
        return coordinator_error(ErrorCode::kInvalidArgument,
                                 "ENDPOINT_CLOCK_HEALTH_RUNTIME_PLAN_INVALID");
      }
      const ClockPlanRecord* const clock = find_clock(clock_plan, record.health_age_clock);
      if (clock == nullptr || !clock->can_read_now || !clock->steady ||
          !clock->independent_progress || !clock->finite_deadline ||
          clock->capability_evidence_id.empty()) {
        return coordinator_error(ErrorCode::kInvalidArgument,
                                 "ENDPOINT_CLOCK_HEALTH_RUNTIME_PLAN_INVALID");
      }
      auto age_clock = clock_registry.resolve(record.health_age_clock);
      if (!age_clock || !age_clock->capabilities().can_read_now() ||
          !age_clock->capabilities().steady()) {
        return coordinator_error(ErrorCode::kFailure, "RUNTIME_CLOCK_INVENTORY_MISMATCH");
      }
      auto receiver =
          EndpointClockHealthReceiver::create(record,
                                              EndpointClockHealthBindingIdentity{
                                                  .run_generation = run_generation,
                                                  .producer_epoch = initial_producer_epoch,
                                                  .clock_epoch = initial_clock_epoch,
                                              });
      if (!receiver) {
        return make_unexpected(receiver.error());
      }
      result->slots_.push_back(Slot{
          .age_clock = record.health_age_clock,
          .age_clock_ref = std::move(age_clock).value(),
          .receiver = std::move(receiver).value(),
      });
    }
    return result;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    return coordinator_error(ErrorCode::kResourceExhausted,
                             "ENDPOINT_CLOCK_HEALTH_PREALLOCATION_FAILED");
  } catch (...) {
    return coordinator_error(ErrorCode::kFailure, "ENDPOINT_CLOCK_HEALTH_RUNTIME_PLAN_INVALID");
  }
#endif
}

EndpointClockHealthRuntimeCoordinator::Slot* EndpointClockHealthRuntimeCoordinator::find_slot(
    PortId endpoint) noexcept {
  const auto found =
      std::lower_bound(slots_.begin(), slots_.end(), endpoint, [](const Slot& slot, PortId value) {
        return slot.receiver.endpoint().value() < value.value();
      });
  return found != slots_.end() && found->receiver.endpoint() == endpoint ? &*found : nullptr;
}

const EndpointClockHealthRuntimeCoordinator::Slot* EndpointClockHealthRuntimeCoordinator::find_slot(
    PortId endpoint) const noexcept {
  const auto found =
      std::lower_bound(slots_.begin(), slots_.end(), endpoint, [](const Slot& slot, PortId value) {
        return slot.receiver.endpoint().value() < value.value();
      });
  return found != slots_.end() && found->receiver.endpoint() == endpoint ? &*found : nullptr;
}

EndpointClockHealthReceiver* EndpointClockHealthRuntimeCoordinator::receiver(
    PortId endpoint) noexcept {
  Slot* const slot = find_slot(endpoint);
  return slot == nullptr ? nullptr : &slot->receiver;
}

const EndpointClockHealthReceiver* EndpointClockHealthRuntimeCoordinator::receiver(
    PortId endpoint) const noexcept {
  const Slot* const slot = find_slot(endpoint);
  return slot == nullptr ? nullptr : &slot->receiver;
}

expected<std::int64_t, EndpointClockHealthError>
EndpointClockHealthRuntimeCoordinator::read_age_time(const Slot& slot) noexcept {
  auto now = slot.age_clock_ref.now();
  if (!now || now->clock_id != slot.age_clock) {
    return coordinator_error(now ? ErrorCode::kFailure : now.error().code,
                             "RUNTIME_CLOCK_READ_FAILED");
  }
  return now->timestamp_ns;
}

expected<void, EndpointClockHealthError> EndpointClockHealthRuntimeCoordinator::submit_report(
    PortId endpoint, ClockHealthReport report, std::uint64_t producer_epoch,
    std::uint64_t publisher_operation_sequence, std::uint64_t report_sequence) noexcept {
  Slot* const slot = find_slot(endpoint);
  if (slot == nullptr) {
    return coordinator_error(ErrorCode::kNotFound, "ENDPOINT_CLOCK_HEALTH_RUNTIME_PLAN_INVALID");
  }
  return slot->receiver.submit_report(
      report, producer_epoch, publisher_operation_sequence, report_sequence);
}

expected<void, EndpointClockHealthError> EndpointClockHealthRuntimeCoordinator::submit_encoded(
    PortId endpoint, EndpointClockHealthControlRecord record) noexcept {
  Slot* const slot = find_slot(endpoint);
  if (slot == nullptr) {
    return coordinator_error(ErrorCode::kNotFound, "ENDPOINT_CLOCK_HEALTH_RUNTIME_PLAN_INVALID");
  }
  return slot->receiver.submit_encoded(record);
}

expected<std::size_t, EndpointClockHealthError>
EndpointClockHealthRuntimeCoordinator::dispatch_pending(PortId endpoint) noexcept {
  Slot* const slot = find_slot(endpoint);
  if (slot == nullptr) {
    return coordinator_error(ErrorCode::kNotFound, "ENDPOINT_CLOCK_HEALTH_RUNTIME_PLAN_INVALID");
  }
  auto now = read_age_time(*slot);
  if (!now) {
    return make_unexpected(now.error());
  }
  return slot->receiver.dispatch_pending(*now);
}

expected<std::size_t, EndpointClockHealthError>
EndpointClockHealthRuntimeCoordinator::expire_due() noexcept {
  for (std::size_t index = 0U; index < slots_.size(); ++index) {
    if (!slots_[index].receiver.expiry_ticket()) {
      continue;
    }
    auto now = read_age_time(slots_[index]);
    if (!now) {
      return make_unexpected(now.error());
    }
    expiry_time_scratch_[index] = *now;
  }

  std::size_t expired{};
  for (std::size_t index = 0U; index < slots_.size(); ++index) {
    auto ticket = slots_[index].receiver.expiry_ticket();
    if (!ticket) {
      continue;
    }
    auto result = slots_[index].receiver.expire_if_due(*ticket, expiry_time_scratch_[index]);
    if (!result) {
      return make_unexpected(result.error());
    }
    expired += *result ? 1U : 0U;
  }
  return expired;
}

std::optional<EndpointClockHealthRuntimeDeadline>
EndpointClockHealthRuntimeCoordinator::next_expiry(ClockId clock) const noexcept {
  std::optional<EndpointClockHealthRuntimeDeadline> result;
  for (const Slot& slot : slots_) {
    if (slot.age_clock != clock) {
      continue;
    }
    auto ticket = slot.receiver.expiry_ticket();
    if (!ticket) {
      continue;
    }
    const EndpointClockHealthRuntimeDeadline candidate{
        .endpoint = slot.receiver.endpoint(),
        .clock = slot.age_clock,
        .ticket = *ticket,
    };
    if (!result || deadline_less(candidate, *result)) {
      result = candidate;
    }
  }
  return result;
}

expected<void, EndpointClockHealthError> EndpointClockHealthRuntimeCoordinator::reset(
    PortId endpoint, EndpointClockHealthUpdateOrigin origin,
    EndpointClockHealthBindingIdentity binding,
    std::optional<std::uint64_t> cause_sequence) noexcept {
  const EndpointClockHealthResetRequest request{
      .endpoint = endpoint,
      .origin = origin,
      .binding = binding,
      .cause_sequence = cause_sequence,
  };
  return reset_batch(std::span<const EndpointClockHealthResetRequest>{&request, 1U});
}

expected<void, EndpointClockHealthError> EndpointClockHealthRuntimeCoordinator::reset_batch(
    const std::span<const EndpointClockHealthResetRequest> requests) noexcept {
  if (requests.size() > slots_.size()) {
    return coordinator_error(ErrorCode::kInvalidArgument, "ENDPOINT_CLOCK_HEALTH_RESET_INVALID");
  }

  // The caller externally serializes the coordinator. Retain each successful clock read in the
  // construction-time scratch array, then mutate only after the complete finite request set has
  // passed. This prevents an invalid later endpoint from leaving earlier health state reset.
  for (std::size_t index = 0U; index < requests.size(); ++index) {
    const EndpointClockHealthResetRequest& request = requests[index];
    if (!request.endpoint ||
        std::any_of(requests.begin(),
                    requests.begin() + static_cast<std::ptrdiff_t>(index),
                    [&](const auto& prior) { return prior.endpoint == request.endpoint; })) {
      return coordinator_error(ErrorCode::kInvalidArgument, "ENDPOINT_CLOCK_HEALTH_RESET_INVALID");
    }
    Slot* const slot = find_slot(request.endpoint);
    if (slot == nullptr) {
      return coordinator_error(ErrorCode::kNotFound, "ENDPOINT_CLOCK_HEALTH_RUNTIME_PLAN_INVALID");
    }
    auto preflight = slot->receiver.preflight_reset(request.origin, request.binding);
    if (!preflight) {
      return make_unexpected(preflight.error());
    }
    auto now = read_age_time(*slot);
    if (!now) {
      return make_unexpected(now.error());
    }
    expiry_time_scratch_[index] = *now;
  }

  for (std::size_t index = 0U; index < requests.size(); ++index) {
    const EndpointClockHealthResetRequest& request = requests[index];
    // Every endpoint was resolved and validated above, and external serialization keeps that
    // preflight stable until this non-failing commit pass.
    Slot* const slot = find_slot(request.endpoint);
    slot->receiver.commit_reset(
        request.origin, request.binding, expiry_time_scratch_[index], request.cause_sequence);
  }
  return {};
}

expected<void, EndpointClockHealthError>
EndpointClockHealthRuntimeCoordinator::record_age_clock_fault(
    PortId endpoint, EndpointHealthAgeClockFault fault,
    std::optional<std::uint64_t> cause_sequence) noexcept {
  Slot* const slot = find_slot(endpoint);
  if (slot == nullptr) {
    return coordinator_error(ErrorCode::kNotFound, "ENDPOINT_CLOCK_HEALTH_RUNTIME_PLAN_INVALID");
  }
  auto now = read_age_time(*slot);
  if (!now) {
    return make_unexpected(now.error());
  }
  return slot->receiver.record_age_clock_fault(fault, *now, cause_sequence);
}

std::optional<EndpointClockHealthUpdateRecord> EndpointClockHealthRuntimeCoordinator::pop_update(
    PortId endpoint) noexcept {
  Slot* const slot = find_slot(endpoint);
  return slot == nullptr ? std::nullopt : slot->receiver.pop_update();
}

}  // namespace holoscan::detail
