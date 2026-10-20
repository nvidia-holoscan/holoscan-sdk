/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "endpoint_clock_health.hpp"

#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <new>
#include <utility>

namespace holoscan::detail {
namespace {

constexpr std::string_view kPublisherOperationSequenceIdentity{
    "holoscan.clock_health.publisher-operation-sequence.v1"};
constexpr std::string_view kReportSequenceIdentity{"holoscan.clock_health.report-sequence.v1"};
constexpr std::string_view kHealthUpdateSequenceIdentity{
    "holoscan.clock_health.update-sequence.v1"};

[[nodiscard]] unexpected<EndpointClockHealthError> receiver_error(ErrorCode code,
                                                                  std::string_view token) noexcept {
  return make_unexpected(EndpointClockHealthError{code, token});
}

[[nodiscard]] bool valid_binding(EndpointClockHealthBindingIdentity binding) noexcept {
  return binding.run_generation != 0U && binding.producer_epoch != 0U && binding.clock_epoch != 0U;
}

[[nodiscard]] bool valid_runtime_disposition(EndpointClockHealthDisposition disposition) noexcept {
  return disposition == EndpointClockHealthDisposition::kSkip ||
         disposition == EndpointClockHealthDisposition::kDegraded ||
         disposition == EndpointClockHealthDisposition::kFailRun;
}

[[nodiscard]] bool valid_unspecified_disposition(
    ClockHealthUnspecifiedDisposition disposition) noexcept {
  return disposition == ClockHealthUnspecifiedDisposition::kAdmitted ||
         disposition == ClockHealthUnspecifiedDisposition::kQualifiedOnly ||
         disposition == ClockHealthUnspecifiedDisposition::kRejected;
}

[[nodiscard]] bool valid_monitored_plan(const EndpointClockHealthPlanRecord& plan) noexcept {
  if (!plan.endpoint || plan.plan_generation == 0U || !plan.capture_clock.valid() ||
      !plan.health_age_clock.valid() || plan.capability != ClockHealthCapability::kStatusReports ||
      plan.requirement != ClockHealthRequirement::kFreshHealthyReport ||
      !plan.max_report_age.has_value() || plan.max_report_age->count() <= 0 ||
      plan.producer_epoch_source != EndpointProducerEpochSource::kAttachmentGeneration ||
      plan.fixed_report_ingress_capacity == 0U || plan.fixed_update_capacity == 0U ||
      plan.per_dispatch_report_budget == 0U ||
      plan.per_dispatch_report_budget > plan.fixed_report_ingress_capacity ||
      plan.ingress_capacity_origin != EndpointClockHealthBoundOrigin::kPortableMinimum ||
      plan.monitor_path_kind != EndpointClockHealthMonitorPathKind::kPublisherSerialized ||
      !valid_runtime_disposition(plan.no_report_disposition) ||
      !valid_runtime_disposition(plan.degraded_disposition) ||
      !valid_runtime_disposition(plan.unhealthy_disposition) ||
      !valid_runtime_disposition(plan.stale_disposition) ||
      !valid_unspecified_disposition(plan.unspecified_reason_disposition) ||
      plan.trace_loss_disposition != EndpointClockHealthTraceLossDisposition::kFailClosed) {
    return false;
  }

  const bool conversion_absent = plan.capture_conversion_id == 0U;
  if (conversion_absent != (plan.conversion_generation == 0U)) {
    return false;
  }
  if (!conversion_absent && plan.conversion_generation != plan.plan_generation) {
    return false;
  }

  return plan.publisher_operation_sequence_id == kPublisherOperationSequenceIdentity &&
         plan.report_sequence_id == kReportSequenceIdentity &&
         plan.health_update_sequence_id == kHealthUpdateSequenceIdentity &&
         plan.clock_health_update_witness_id == EndpointClockHealthUpdateRecord::kWitnessIdentity;
}

[[nodiscard]] bool valid_assessment_value(std::uint8_t value) noexcept {
  return value <= static_cast<std::uint8_t>(ClockHealthAssessment::kUnhealthy);
}

[[nodiscard]] bool valid_reason_value(std::uint8_t value) noexcept {
  return value <= static_cast<std::uint8_t>(ClockHealthReason::kUnspecified);
}

[[nodiscard]] bool valid_report_pair(ClockHealthAssessment assessment,
                                     ClockHealthReason reason) noexcept {
  const bool healthy = assessment == ClockHealthAssessment::kHealthy;
  const bool no_negative_reason = reason == ClockHealthReason::kNone;
  return healthy == no_negative_reason;
}

[[nodiscard]] EndpointClockHealthAssessmentState assessment_state(
    ClockHealthAssessment assessment) noexcept {
  switch (assessment) {
    case ClockHealthAssessment::kHealthy:
      return EndpointClockHealthAssessmentState::kHealthy;
    case ClockHealthAssessment::kDegraded:
      return EndpointClockHealthAssessmentState::kDegraded;
    case ClockHealthAssessment::kUnhealthy:
      return EndpointClockHealthAssessmentState::kUnhealthy;
  }
  return EndpointClockHealthAssessmentState::kAbsent;
}

[[nodiscard]] bool checked_expiry(std::int64_t accepted_at_ns, std::int64_t max_report_age_ns,
                                  std::int64_t& expiry_deadline_ns) noexcept {
  if (max_report_age_ns <= 0 ||
      accepted_at_ns > std::numeric_limits<std::int64_t>::max() - max_report_age_ns) {
    return false;
  }
  expiry_deadline_ns = accepted_at_ns + max_report_age_ns;
  return true;
}

[[nodiscard]] bool valid_reset_origin(EndpointClockHealthUpdateOrigin origin) noexcept {
  return origin == EndpointClockHealthUpdateOrigin::kAttachmentReset ||
         origin == EndpointClockHealthUpdateOrigin::kProducerEpochReset ||
         origin == EndpointClockHealthUpdateOrigin::kClockEpochReset;
}

}  // namespace

EndpointClockHealthReceiver::EndpointClockHealthReceiver(RuntimePlan plan,
                                                         EndpointClockHealthBindingIdentity binding,
                                                         IngressStorage ingress,
                                                         UpdateStorage updates) noexcept
    : plan_(std::move(plan)), ingress_(std::move(ingress)), updates_(std::move(updates)) {
  state_.binding = binding;
}

expected<EndpointClockHealthReceiver, EndpointClockHealthError> EndpointClockHealthReceiver::create(
    const EndpointClockHealthPlanRecord& plan,
    EndpointClockHealthBindingIdentity binding) noexcept {
  const auto max_report_age = plan.max_report_age;
  if (!max_report_age || !valid_monitored_plan(plan) || !valid_binding(binding)) {
    return receiver_error(ErrorCode::kInvalidArgument,
                          "ENDPOINT_CLOCK_HEALTH_RUNTIME_PLAN_INVALID");
  }

  auto ingress = IngressStorage(
      new (std::nothrow) EndpointClockHealthControlRecord[plan.fixed_report_ingress_capacity]);
  auto updates =
      UpdateStorage(new (std::nothrow) EndpointClockHealthUpdateRecord[plan.fixed_update_capacity]);
  if (!ingress || !updates) {
    return receiver_error(ErrorCode::kResourceExhausted,
                          "ENDPOINT_CLOCK_HEALTH_PREALLOCATION_FAILED");
  }

  RuntimePlan runtime_plan{
      .endpoint = plan.endpoint,
      .plan_generation = plan.plan_generation,
      .capture_clock = plan.capture_clock,
      .capture_conversion_id = plan.capture_conversion_id,
      .conversion_generation = plan.conversion_generation,
      .health_age_clock = plan.health_age_clock,
      .max_report_age_ns = max_report_age.value().count(),
      .report_capacity = plan.fixed_report_ingress_capacity,
      .update_capacity = plan.fixed_update_capacity,
      .per_dispatch_report_budget = plan.per_dispatch_report_budget,
      .no_report_disposition = plan.no_report_disposition,
      .degraded_disposition = plan.degraded_disposition,
      .unhealthy_disposition = plan.unhealthy_disposition,
      .stale_disposition = plan.stale_disposition,
      .unspecified_reason_disposition = plan.unspecified_reason_disposition,
  };
  EndpointClockHealthReceiver receiver{
      std::move(runtime_plan), binding, std::move(ingress), std::move(updates)};

  // Attachment creation is itself an ordered no-report state transition. There is no meaningful
  // evaluation coordinate yet, so zero is not fabricated as a timestamp.
  const EndpointClockHealthState old_state = receiver.state_;
  receiver.state_.health_generation = 1U;
  receiver.state_.health_update_sequence = 1U;
  receiver.push_update(
      EndpointClockHealthUpdateOrigin::kAttachmentReset, old_state, std::nullopt, std::nullopt);
  return receiver;
}

expected<void, EndpointClockHealthError> EndpointClockHealthReceiver::submit_report(
    ClockHealthReport report, std::uint64_t producer_epoch,
    std::uint64_t publisher_operation_sequence, std::uint64_t report_sequence) noexcept {
  return enqueue(
      EndpointClockHealthControlRecord{
          .producer_epoch = producer_epoch,
          .publisher_operation_sequence = publisher_operation_sequence,
          .report_sequence = report_sequence,
          .assessment = static_cast<std::uint8_t>(report.assessment),
          .reason = static_cast<std::uint8_t>(report.reason),
      },
      false);
}

expected<void, EndpointClockHealthError> EndpointClockHealthReceiver::submit_encoded(
    EndpointClockHealthControlRecord record) noexcept {
  return enqueue(record, true);
}

expected<void, EndpointClockHealthError> EndpointClockHealthReceiver::enqueue(
    EndpointClockHealthControlRecord record, bool private_decode) noexcept {
  // Validate every field before touching ingress indexes or sequence high-water marks. An unknown
  // private representation is a protocol failure and is never converted to kUnspecified.
  if (!valid_assessment_value(record.assessment) || !valid_reason_value(record.reason)) {
    return receiver_error(private_decode ? ErrorCode::kProtocolError : ErrorCode::kInvalidArgument,
                          private_decode ? "ENDPOINT_CLOCK_HEALTH_REASON_UNSUPPORTED"
                                         : "ENDPOINT_CLOCK_HEALTH_REPORT_INVALID");
  }

  const auto assessment = static_cast<ClockHealthAssessment>(record.assessment);
  const auto reason = static_cast<ClockHealthReason>(record.reason);
  if (!valid_report_pair(assessment, reason)) {
    return receiver_error(private_decode ? ErrorCode::kProtocolError : ErrorCode::kInvalidArgument,
                          "ENDPOINT_CLOCK_HEALTH_REPORT_INVALID");
  }
  if (reason == ClockHealthReason::kUnspecified &&
      plan_.unspecified_reason_disposition == ClockHealthUnspecifiedDisposition::kRejected) {
    return receiver_error(ErrorCode::kPermissionDenied,
                          "ENDPOINT_CLOCK_HEALTH_UNSPECIFIED_REJECTED");
  }
  if (record.producer_epoch != state_.binding.producer_epoch) {
    return receiver_error(ErrorCode::kStaleEndpoint, "ENDPOINT_CLOCK_HEALTH_GENERATION_STALE");
  }
  if (record.publisher_operation_sequence == 0U || record.report_sequence == 0U ||
      record.publisher_operation_sequence <= enqueued_publisher_operation_sequence_ ||
      record.report_sequence <= enqueued_report_sequence_) {
    return receiver_error(ErrorCode::kProtocolError, "ENDPOINT_CLOCK_HEALTH_SEQUENCE_INVALID");
  }
  if (ingress_size_ == plan_.report_capacity) {
    return receiver_error(ErrorCode::kBackpressured, "ENDPOINT_CLOCK_HEALTH_INGRESS_FULL");
  }

  const std::size_t tail = (ingress_head_ + ingress_size_) % plan_.report_capacity;
  ingress_[tail] = record;
  ++ingress_size_;
  enqueued_publisher_operation_sequence_ = record.publisher_operation_sequence;
  enqueued_report_sequence_ = record.report_sequence;
  return {};
}

expected<std::size_t, EndpointClockHealthError> EndpointClockHealthReceiver::dispatch_pending(
    std::int64_t accepted_at_ns) noexcept {
  if (ingress_size_ == 0U) {
    return 0U;
  }

  const std::size_t available_updates = plan_.update_capacity - update_size_;
  const std::size_t apply_count =
      std::min({ingress_size_, plan_.per_dispatch_report_budget, available_updates});
  if (apply_count == 0U) {
    return receiver_error(ErrorCode::kBackpressured,
                          "ENDPOINT_CLOCK_HEALTH_UPDATE_CAPACITY_EXHAUSTED");
  }

  std::int64_t expiry_deadline_ns{};
  if (!checked_expiry(accepted_at_ns, plan_.max_report_age_ns, expiry_deadline_ns)) {
    return receiver_error(ErrorCode::kProtocolError, "ENDPOINT_CLOCK_HEALTH_DEADLINE_OVERFLOW");
  }
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  if (apply_count > maximum - state_.health_update_sequence ||
      apply_count > maximum - state_.health_generation) {
    return receiver_error(ErrorCode::kResourceExhausted,
                          "ENDPOINT_CLOCK_HEALTH_SEQUENCE_EXHAUSTED");
  }

  for (std::size_t index = 0U; index < apply_count; ++index) {
    const EndpointClockHealthControlRecord record = ingress_[ingress_head_];
    const EndpointClockHealthState old_state = state_;
    state_.assessment = assessment_state(static_cast<ClockHealthAssessment>(record.assessment));
    state_.freshness = EndpointClockHealthFreshness::kFresh;
    state_.reason = static_cast<ClockHealthReason>(record.reason);
    state_.accepted_at_ns = accepted_at_ns;
    state_.expiry_deadline_ns = expiry_deadline_ns;
    state_.publisher_operation_sequence = record.publisher_operation_sequence;
    state_.report_sequence = record.report_sequence;
    ++state_.health_update_sequence;
    ++state_.health_generation;

    ingress_head_ = (ingress_head_ + 1U) % plan_.report_capacity;
    --ingress_size_;
    push_update(
        EndpointClockHealthUpdateOrigin::kReportAccepted, old_state, accepted_at_ns, std::nullopt);
  }
  return apply_count;
}

std::optional<EndpointClockHealthExpiryTicket> EndpointClockHealthReceiver::expiry_ticket()
    const noexcept {
  if (state_.freshness != EndpointClockHealthFreshness::kFresh ||
      !state_.expiry_deadline_ns.has_value()) {
    return std::nullopt;
  }
  return EndpointClockHealthExpiryTicket{
      .health_generation = state_.health_generation,
      .report_sequence = state_.report_sequence,
      .expiry_deadline_ns = *state_.expiry_deadline_ns,
  };
}

expected<bool, EndpointClockHealthError> EndpointClockHealthReceiver::expire_if_due(
    EndpointClockHealthExpiryTicket ticket, std::int64_t evaluation_time_ns) noexcept {
  if (state_.freshness != EndpointClockHealthFreshness::kFresh ||
      ticket.health_generation != state_.health_generation ||
      ticket.report_sequence != state_.report_sequence || !state_.expiry_deadline_ns.has_value() ||
      ticket.expiry_deadline_ns != *state_.expiry_deadline_ns) {
    return false;
  }
  if (evaluation_time_ns < ticket.expiry_deadline_ns) {
    return false;
  }
  if (!update_slot_available()) {
    return receiver_error(ErrorCode::kBackpressured,
                          "ENDPOINT_CLOCK_HEALTH_UPDATE_CAPACITY_EXHAUSTED");
  }
  if (state_.health_update_sequence == std::numeric_limits<std::uint64_t>::max() ||
      state_.health_generation == std::numeric_limits<std::uint64_t>::max()) {
    return receiver_error(ErrorCode::kResourceExhausted,
                          "ENDPOINT_CLOCK_HEALTH_SEQUENCE_EXHAUSTED");
  }

  const EndpointClockHealthState old_state = state_;
  state_.freshness = EndpointClockHealthFreshness::kStale;
  ++state_.health_update_sequence;
  ++state_.health_generation;
  push_update(
      EndpointClockHealthUpdateOrigin::kReportExpired, old_state, evaluation_time_ns, std::nullopt);
  return true;
}

expected<void, EndpointClockHealthError> EndpointClockHealthReceiver::reset(
    EndpointClockHealthUpdateOrigin origin, EndpointClockHealthBindingIdentity binding,
    std::int64_t evaluation_time_ns, std::optional<std::uint64_t> cause_sequence) noexcept {
  auto preflight = preflight_reset(origin, binding);
  if (!preflight) {
    return make_unexpected(preflight.error());
  }
  commit_reset(origin, binding, evaluation_time_ns, cause_sequence);
  return {};
}

expected<void, EndpointClockHealthError> EndpointClockHealthReceiver::preflight_reset(
    EndpointClockHealthUpdateOrigin origin,
    EndpointClockHealthBindingIdentity binding) const noexcept {
  if (!valid_reset_origin(origin) || !valid_binding(binding)) {
    return receiver_error(ErrorCode::kInvalidArgument, "ENDPOINT_CLOCK_HEALTH_RESET_INVALID");
  }
  if (binding.run_generation != state_.binding.run_generation ||
      (origin == EndpointClockHealthUpdateOrigin::kProducerEpochReset &&
       (binding.producer_epoch == state_.binding.producer_epoch ||
        binding.clock_epoch != state_.binding.clock_epoch)) ||
      (origin == EndpointClockHealthUpdateOrigin::kClockEpochReset &&
       (binding.clock_epoch == state_.binding.clock_epoch ||
        binding.producer_epoch != state_.binding.producer_epoch))) {
    return receiver_error(ErrorCode::kInvalidArgument, "ENDPOINT_CLOCK_HEALTH_RESET_INVALID");
  }
  if (ingress_size_ != 0U) {
    return receiver_error(ErrorCode::kNotReady, "ENDPOINT_CLOCK_HEALTH_RESET_REQUIRES_QUIESCENCE");
  }
  if (!update_slot_available()) {
    return receiver_error(ErrorCode::kBackpressured,
                          "ENDPOINT_CLOCK_HEALTH_UPDATE_CAPACITY_EXHAUSTED");
  }
  if (state_.health_update_sequence == std::numeric_limits<std::uint64_t>::max() ||
      state_.health_generation == std::numeric_limits<std::uint64_t>::max()) {
    return receiver_error(ErrorCode::kResourceExhausted,
                          "ENDPOINT_CLOCK_HEALTH_SEQUENCE_EXHAUSTED");
  }
  return {};
}

void EndpointClockHealthReceiver::commit_reset(
    EndpointClockHealthUpdateOrigin origin, EndpointClockHealthBindingIdentity binding,
    std::int64_t evaluation_time_ns, std::optional<std::uint64_t> cause_sequence) noexcept {
  const EndpointClockHealthState old_state = state_;
  state_.binding = binding;
  state_.assessment = EndpointClockHealthAssessmentState::kAbsent;
  state_.freshness = EndpointClockHealthFreshness::kNoReport;
  state_.reason.reset();
  state_.accepted_at_ns.reset();
  state_.expiry_deadline_ns.reset();
  state_.publisher_operation_sequence = 0U;
  state_.report_sequence = 0U;
  ++state_.health_update_sequence;
  ++state_.health_generation;

  if (origin == EndpointClockHealthUpdateOrigin::kAttachmentReset ||
      origin == EndpointClockHealthUpdateOrigin::kProducerEpochReset) {
    enqueued_publisher_operation_sequence_ = 0U;
    enqueued_report_sequence_ = 0U;
  }

  push_update(origin, old_state, evaluation_time_ns, cause_sequence);
}

expected<void, EndpointClockHealthError> EndpointClockHealthReceiver::record_age_clock_fault(
    EndpointHealthAgeClockFault fault, std::int64_t evaluation_time_ns,
    std::optional<std::uint64_t> cause_sequence) noexcept {
  if (fault != EndpointHealthAgeClockFault::kStalled &&
      fault != EndpointHealthAgeClockFault::kRegression) {
    return receiver_error(ErrorCode::kInvalidArgument,
                          "ENDPOINT_CLOCK_HEALTH_AGE_CLOCK_FAULT_INVALID");
  }
  if (!update_slot_available()) {
    return receiver_error(ErrorCode::kBackpressured,
                          "ENDPOINT_CLOCK_HEALTH_UPDATE_CAPACITY_EXHAUSTED");
  }
  if (state_.health_update_sequence == std::numeric_limits<std::uint64_t>::max() ||
      state_.health_generation == std::numeric_limits<std::uint64_t>::max()) {
    return receiver_error(ErrorCode::kResourceExhausted,
                          "ENDPOINT_CLOCK_HEALTH_SEQUENCE_EXHAUSTED");
  }

  const EndpointClockHealthState old_state = state_;
  if (state_.freshness == EndpointClockHealthFreshness::kFresh) {
    state_.freshness = EndpointClockHealthFreshness::kStale;
  }
  ++state_.health_update_sequence;
  ++state_.health_generation;
  push_update(EndpointClockHealthUpdateOrigin::kRuntimeHealthClockFault,
              old_state,
              evaluation_time_ns,
              cause_sequence,
              fault);
  return {};
}

std::optional<EndpointClockHealthUpdateRecord> EndpointClockHealthReceiver::pop_update() noexcept {
  if (update_size_ == 0U) {
    return std::nullopt;
  }
  EndpointClockHealthUpdateRecord result = updates_[update_head_];
  update_head_ = (update_head_ + 1U) % plan_.update_capacity;
  --update_size_;
  return result;
}

bool EndpointClockHealthReceiver::satisfies_requirement() const noexcept {
  return state_.freshness == EndpointClockHealthFreshness::kFresh &&
         state_.assessment == EndpointClockHealthAssessmentState::kHealthy;
}

EndpointClockHealthDisposition EndpointClockHealthReceiver::current_disposition() const noexcept {
  if (state_.freshness == EndpointClockHealthFreshness::kNoReport) {
    return plan_.no_report_disposition;
  }
  if (state_.freshness == EndpointClockHealthFreshness::kStale) {
    return plan_.stale_disposition;
  }
  if (state_.assessment == EndpointClockHealthAssessmentState::kDegraded) {
    return plan_.degraded_disposition;
  }
  if (state_.assessment == EndpointClockHealthAssessmentState::kUnhealthy) {
    return plan_.unhealthy_disposition;
  }
  return EndpointClockHealthDisposition::kNotApplicable;
}

void EndpointClockHealthReceiver::push_update(
    EndpointClockHealthUpdateOrigin origin, const EndpointClockHealthState& old_state,
    std::optional<std::int64_t> evaluation_time_ns, std::optional<std::uint64_t> cause_sequence,
    std::optional<EndpointHealthAgeClockFault> age_clock_fault) noexcept {
  assert(update_slot_available());
  const std::size_t tail = (update_head_ + update_size_) % plan_.update_capacity;
  updates_[tail] = EndpointClockHealthUpdateRecord{
      .endpoint = plan_.endpoint,
      .plan_generation = plan_.plan_generation,
      .capture_clock = plan_.capture_clock,
      .capture_conversion_id = plan_.capture_conversion_id,
      .conversion_generation = plan_.conversion_generation,
      .health_age_clock = plan_.health_age_clock,
      .binding = state_.binding,
      .publisher_operation_sequence = state_.publisher_operation_sequence,
      .report_sequence = state_.report_sequence,
      .health_update_sequence = state_.health_update_sequence,
      .health_generation = state_.health_generation,
      .origin = origin,
      .old_assessment = old_state.assessment,
      .new_assessment = state_.assessment,
      .old_freshness = old_state.freshness,
      .new_freshness = state_.freshness,
      .old_reason = old_state.reason,
      .new_reason = state_.reason,
      .receiver_acceptance_time_ns = state_.accepted_at_ns,
      .expiry_deadline_ns = state_.expiry_deadline_ns,
      .evaluation_time_ns = evaluation_time_ns,
      .age_clock_fault = age_clock_fault,
      .cause_sequence = cause_sequence,
      .disposition = current_disposition(),
  };
  ++update_size_;
}

bool EndpointClockHealthReceiver::update_slot_available() const noexcept {
  return update_size_ < plan_.update_capacity;
}

}  // namespace holoscan::detail
