/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>

#include <holoscan/channel/clock_health.hpp>
#include <holoscan/core/execution_plan.hpp>

#include "runtime/endpoint_clock_health.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;
using detail::EndpointClockHealthAssessmentState;
using detail::EndpointClockHealthBindingIdentity;
using detail::EndpointClockHealthControlRecord;
using detail::EndpointClockHealthFreshness;
using detail::EndpointClockHealthReceiver;
using detail::EndpointClockHealthUpdateOrigin;
using detail::EndpointHealthAgeClockFault;

// Builds a fully-configured EndpointClockHealthPlanRecord with active monitoring: 100 ns max
// age, fail-closed trace-loss, and kFreshHealthyReport requirement. Pass optional overrides
// to exercise specific capacity or unspecified-reason disposition variants.
// Representative test: AllocatesOnlyForACompleteMonitoredPlan.
[[nodiscard]] EndpointClockHealthPlanRecord monitored_plan(
    ClockHealthUnspecifiedDisposition unspecified = ClockHealthUnspecifiedDisposition::kAdmitted,
    std::uint32_t report_capacity = 1U, std::uint32_t update_capacity = 64U,
    std::uint32_t report_budget = 1U) {
  return EndpointClockHealthPlanRecord{
      .endpoint = PortId{11U},
      .plan_generation = 7U,
      .capture_clock = ClockId{3U},
      .capture_conversion_id = 0U,
      .conversion_generation = 0U,
      .producer_epoch_source = EndpointProducerEpochSource::kAttachmentGeneration,
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 100ns,
      .health_age_clock = ClockId{4U},
      .fixed_report_ingress_capacity = report_capacity,
      .fixed_update_capacity = update_capacity,
      .ingress_capacity_origin = EndpointClockHealthBoundOrigin::kPortableMinimum,
      .per_dispatch_report_budget = report_budget,
      .publisher_operation_sequence_id = "holoscan.clock_health.publisher-operation-sequence.v1",
      .report_sequence_id = "holoscan.clock_health.report-sequence.v1",
      .health_update_sequence_id = "holoscan.clock_health.update-sequence.v1",
      .monitor_path_kind = EndpointClockHealthMonitorPathKind::kPublisherSerialized,
      .no_report_disposition = EndpointClockHealthDisposition::kSkip,
      .degraded_disposition = EndpointClockHealthDisposition::kDegraded,
      .unhealthy_disposition = EndpointClockHealthDisposition::kFailRun,
      .stale_disposition = EndpointClockHealthDisposition::kSkip,
      .unspecified_reason_disposition = unspecified,
      .clock_health_update_witness_id = "holoscan.event.clock-health-update.v1",
      .trace_loss_disposition = EndpointClockHealthTraceLossDisposition::kFailClosed,
  };
}

// Stable run/producer/clock epoch triple shared across receiver construction and report
// submissions unless a test explicitly rotates one of the epochs.
constexpr EndpointClockHealthBindingIdentity kBinding{
    .run_generation = 9U,
    .producer_epoch = 5U,
    .clock_epoch = 2U,
};

// Constructs an EndpointClockHealthReceiver from the given plan and asserts creation succeeds.
// Returns by value; callers that need a blank baseline should call drain_updates() first.
// Representative test: AllocatesOnlyForACompleteMonitoredPlan.
[[nodiscard]] EndpointClockHealthReceiver make_receiver(
    const EndpointClockHealthPlanRecord& plan = monitored_plan()) {
  auto created = EndpointClockHealthReceiver::create(plan, kBinding);
  EXPECT_TRUE(created) << (created ? std::string_view{} : created.error().token);
  return std::move(created).value();
}

// Clears the update queue so a test starts with a known empty baseline after construction
// or after a reset-induced kAttachmentReset event.
void drain_updates(EndpointClockHealthReceiver& receiver) {
  while (receiver.pop_update().has_value()) {
  }
}

// Submits a clock-health report, dispatches pending work, and asserts exactly one
// kReportAccepted update is queued with matching sequence and acceptance time.
// Used by AppliesTheCompletePublicAssessmentReasonMatrix and several single-report tests.
//
// GoogleTest fatal assertions terminate this helper before optional access, but clang-tidy does
// not model the macro's control flow.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
void expect_report_accepted(EndpointClockHealthReceiver& receiver, ClockHealthReport report,
                            std::uint64_t sequence, std::int64_t accepted_at) {
  ASSERT_TRUE(receiver.submit_report(report, kBinding.producer_epoch, sequence, sequence));
  const auto applied = receiver.dispatch_pending(accepted_at);
  ASSERT_TRUE(applied);
  EXPECT_EQ(*applied, 1U);
  const auto event = receiver.pop_update();
  ASSERT_TRUE(event.has_value());
  EXPECT_EQ(event->origin, EndpointClockHealthUpdateOrigin::kReportAccepted);
  EXPECT_EQ(event->publisher_operation_sequence, sequence);
  EXPECT_EQ(event->report_sequence, sequence);
  EXPECT_EQ(event->receiver_acceptance_time_ns, accepted_at);
  EXPECT_FALSE(event->cause_sequence.has_value());
}
// NOLINTEND(bugprone-unchecked-optional-access)

// REQ: RUNTIME-009; Q61-T07--T11
// GoogleTest fatal assertions terminate this test before optional access, but clang-tidy does not
// model the macro's control flow.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
TEST(EndpointClockHealthTest, AllocatesOnlyForACompleteMonitoredPlan) {
  // Key setup: non-monitoring plans (kUnavailable), zero-age, mismatched schema, and zero
  // binding are all invalid; only a fully-specified monitored plan with a live binding succeeds.
  auto ordinary = monitored_plan();
  ordinary.capability = ClockHealthCapability::kUnavailable;
  ordinary.requirement = ClockHealthRequirement::kNotRequired;
  ordinary.max_report_age.reset();
  ordinary.fixed_report_ingress_capacity = 0U;
  ordinary.fixed_update_capacity = 0U;
  ordinary.per_dispatch_report_budget = 0U;
  ordinary.producer_epoch_source = EndpointProducerEpochSource::kNotApplicable;
  ordinary.ingress_capacity_origin = EndpointClockHealthBoundOrigin::kNotApplicable;
  ordinary.monitor_path_kind = EndpointClockHealthMonitorPathKind::kNotApplicable;
  ordinary.no_report_disposition = EndpointClockHealthDisposition::kNotApplicable;
  ordinary.degraded_disposition = EndpointClockHealthDisposition::kNotApplicable;
  ordinary.unhealthy_disposition = EndpointClockHealthDisposition::kNotApplicable;
  ordinary.stale_disposition = EndpointClockHealthDisposition::kNotApplicable;
  ordinary.trace_loss_disposition = EndpointClockHealthTraceLossDisposition::kNotApplicable;
  EXPECT_FALSE(EndpointClockHealthReceiver::create(ordinary, kBinding));

  auto zero_age = monitored_plan();
  zero_age.max_report_age = 0ns;
  EXPECT_FALSE(EndpointClockHealthReceiver::create(zero_age, kBinding));

  auto mismatched_schema = monitored_plan();
  mismatched_schema.health_update_sequence_id = "future-schema";
  EXPECT_FALSE(EndpointClockHealthReceiver::create(mismatched_schema, kBinding));

  EXPECT_FALSE(
      EndpointClockHealthReceiver::create(monitored_plan(), EndpointClockHealthBindingIdentity{}));

  auto receiver = make_receiver();
  const auto initial = receiver.pop_update();
  ASSERT_TRUE(initial.has_value());
  // Key assertion: initial update carries kAttachmentReset with kAbsent assessment and
  // kNoReport freshness, and current_disposition resolves to kSkip (no-report policy).
  EXPECT_EQ(initial->origin, EndpointClockHealthUpdateOrigin::kAttachmentReset);
  EXPECT_EQ(initial->endpoint, PortId{11U});
  EXPECT_EQ(initial->plan_generation, 7U);
  EXPECT_EQ(initial->capture_clock, ClockId{3U});
  EXPECT_EQ(initial->health_age_clock, ClockId{4U});
  EXPECT_EQ(initial->binding, kBinding);
  EXPECT_EQ(initial->health_update_sequence, 1U);
  EXPECT_EQ(initial->health_generation, 1U);
  EXPECT_EQ(initial->new_assessment, EndpointClockHealthAssessmentState::kAbsent);
  EXPECT_EQ(initial->new_freshness, EndpointClockHealthFreshness::kNoReport);
  EXPECT_FALSE(initial->evaluation_time_ns.has_value());
  EXPECT_EQ(receiver.current_disposition(), EndpointClockHealthDisposition::kSkip);
}
// NOLINTEND(bugprone-unchecked-optional-access)

// REQ: RUNTIME-009; Q61-T02/T03/T13
TEST(EndpointClockHealthTest, AppliesTheCompletePublicAssessmentReasonMatrix) {
  // Key setup: iterates every ClockHealthReason value, submitting both degraded and unhealthy
  // variants of each to confirm the full reason→disposition mapping is wired correctly.
  auto receiver = make_receiver();
  drain_updates(receiver);

  std::uint64_t sequence = 1U;
  expect_report_accepted(
      receiver, ClockHealthReport::healthy(), sequence, static_cast<std::int64_t>(sequence));
  EXPECT_EQ(receiver.state().assessment, EndpointClockHealthAssessmentState::kHealthy);
  EXPECT_TRUE(receiver.satisfies_requirement());

  for (std::uint8_t value = static_cast<std::uint8_t>(ClockHealthReason::kSynchronizationLockLost);
       value <= static_cast<std::uint8_t>(ClockHealthReason::kUnspecified);
       ++value) {
    const auto reason = static_cast<ClockHealthReason>(value);
    ++sequence;
    expect_report_accepted(receiver,
                           ClockHealthReport::degraded(reason),
                           sequence,
                           static_cast<std::int64_t>(sequence));
    EXPECT_EQ(receiver.state().assessment, EndpointClockHealthAssessmentState::kDegraded);
    EXPECT_EQ(receiver.state().reason, reason);
    EXPECT_EQ(receiver.current_disposition(), EndpointClockHealthDisposition::kDegraded);

    ++sequence;
    expect_report_accepted(receiver,
                           ClockHealthReport::unhealthy(reason),
                           sequence,
                           static_cast<std::int64_t>(sequence));
    EXPECT_EQ(receiver.state().assessment, EndpointClockHealthAssessmentState::kUnhealthy);
    EXPECT_EQ(receiver.state().reason, reason);
    // Key assertion: each unhealthy reason maps to kFailRun; each degraded reason maps to
    // kDegraded — no reason value is silently dropped.
    EXPECT_EQ(receiver.current_disposition(), EndpointClockHealthDisposition::kFailRun);
  }
}

// REQ: RUNTIME-009; Q61-T03/T04
// Deliberately malformed enum representations exercise the fail-closed public boundary.
// NOLINTBEGIN(clang-analyzer-optin.core.EnumCastOutOfRange)
TEST(EndpointClockHealthTest, RejectsInvalidAndUnknownValuesBeforeAnyMutation) {
  // Key setup: healthy report accepted first to establish a known reference state; each
  // subsequent malformed public and encoded submission must leave that state unchanged.
  auto receiver = make_receiver();
  drain_updates(receiver);
  expect_report_accepted(receiver, ClockHealthReport::healthy(), 1U, 10);
  const auto generation = receiver.state().health_generation;
  const auto update_sequence = receiver.state().health_update_sequence;
  const auto deadline = receiver.state().expiry_deadline_ns;

  const auto expect_public_rejection = [&](ClockHealthReport report) {
    const auto result = receiver.submit_report(report, kBinding.producer_epoch, 2U, 2U);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(receiver.state().health_generation, generation);
    EXPECT_EQ(receiver.state().health_update_sequence, update_sequence);
    EXPECT_EQ(receiver.state().expiry_deadline_ns, deadline);
    EXPECT_EQ(receiver.pending_report_count(), 0U);
  };
  expect_public_rejection({.assessment = ClockHealthAssessment::kHealthy,
                           .reason = ClockHealthReason::kSynchronizationLockLost});
  expect_public_rejection(
      {.assessment = ClockHealthAssessment::kDegraded, .reason = ClockHealthReason::kNone});
  expect_public_rejection(
      {.assessment = static_cast<ClockHealthAssessment>(0xFF), .reason = ClockHealthReason::kNone});
  expect_public_rejection({.assessment = ClockHealthAssessment::kUnhealthy,
                           .reason = static_cast<ClockHealthReason>(0xFF)});

  const auto encoded = receiver.submit_encoded(EndpointClockHealthControlRecord{
      .producer_epoch = kBinding.producer_epoch,
      .publisher_operation_sequence = 2U,
      .report_sequence = 2U,
      .assessment = static_cast<std::uint8_t>(ClockHealthAssessment::kUnhealthy),
      .reason = 0xFFU,
  });
  ASSERT_FALSE(encoded);
  EXPECT_EQ(encoded.error().code, ErrorCode::kProtocolError);
  EXPECT_EQ(encoded.error().token, "ENDPOINT_CLOCK_HEALTH_REASON_UNSUPPORTED");
  EXPECT_EQ(receiver.state().health_generation, generation);
  EXPECT_EQ(receiver.pending_update_count(), 0U);

  const auto unknown_assessment = receiver.submit_encoded(EndpointClockHealthControlRecord{
      .producer_epoch = kBinding.producer_epoch,
      .publisher_operation_sequence = 2U,
      .report_sequence = 2U,
      .assessment = 0xFFU,
      .reason = static_cast<std::uint8_t>(ClockHealthReason::kNone),
  });
  ASSERT_FALSE(unknown_assessment);
  EXPECT_EQ(unknown_assessment.error().code, ErrorCode::kProtocolError);
  EXPECT_EQ(unknown_assessment.error().token, "ENDPOINT_CLOCK_HEALTH_REASON_UNSUPPORTED");
  // Key assertion: health_generation is unchanged after every rejection path, confirming
  // no partial mutation occurred.
  EXPECT_EQ(receiver.state().health_generation, generation);
}
// NOLINTEND(clang-analyzer-optin.core.EnumCastOutOfRange)

// REQ: RUNTIME-009; Q61-T06
TEST(EndpointClockHealthTest, EnforcesTheCompiledUnspecifiedDispositionWithoutMutation) {
  // Key setup: kAdmitted and kQualifiedOnly allow kUnspecified reason reports; kRejected
  // must return kPermissionDenied without touching receiver state or pending counts.
  for (const auto disposition : {ClockHealthUnspecifiedDisposition::kAdmitted,
                                 ClockHealthUnspecifiedDisposition::kQualifiedOnly}) {
    auto receiver = make_receiver(monitored_plan(disposition));
    drain_updates(receiver);
    expect_report_accepted(
        receiver, ClockHealthReport::degraded(ClockHealthReason::kUnspecified), 1U, 10);
  }

  auto rejected = make_receiver(monitored_plan(ClockHealthUnspecifiedDisposition::kRejected));
  drain_updates(rejected);
  const auto result =
      rejected.submit_report(ClockHealthReport::unhealthy(ClockHealthReason::kUnspecified),
                             kBinding.producer_epoch,
                             1U,
                             1U);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kPermissionDenied);
  // Key assertion: freshness remains kNoReport and all pending counts stay zero, confirming
  // the rejection left no partial state.
  EXPECT_EQ(rejected.state().freshness, EndpointClockHealthFreshness::kNoReport);
  EXPECT_EQ(rejected.state().health_generation, 1U);
  EXPECT_EQ(rejected.pending_report_count(), 0U);
  EXPECT_EQ(rejected.pending_update_count(), 0U);
}

// REQ: RUNTIME-009; CH9/CH10
TEST(EndpointClockHealthTest, KeepsPublisherIngressBoundedAndReceiverAcceptanceDistinct) {
  // Key setup: report_capacity=1 and update_capacity=1 so ingress and acceptance queues
  // saturate independently; reset is blocked while the ingress queue holds a pending report.
  auto receiver =
      make_receiver(monitored_plan(ClockHealthUnspecifiedDisposition::kAdmitted, 1U, 1U, 1U));

  ASSERT_TRUE(
      receiver.submit_report(ClockHealthReport::healthy(), kBinding.producer_epoch, 1U, 1U));
  EXPECT_EQ(receiver.state().freshness, EndpointClockHealthFreshness::kNoReport);
  EXPECT_EQ(receiver.pending_report_count(), 1U);

  const EndpointClockHealthBindingIdentity next_binding{
      .run_generation = kBinding.run_generation,
      .producer_epoch = kBinding.producer_epoch + 1U,
      .clock_epoch = kBinding.clock_epoch,
  };
  const auto reset_while_pending =
      receiver.reset(EndpointClockHealthUpdateOrigin::kProducerEpochReset, next_binding, 5);
  ASSERT_FALSE(reset_while_pending);
  EXPECT_EQ(reset_while_pending.error().code, ErrorCode::kNotReady);
  EXPECT_EQ(receiver.state().binding, kBinding);
  EXPECT_EQ(receiver.pending_report_count(), 1U);

  const auto full =
      receiver.submit_report(ClockHealthReport::unhealthy(ClockHealthReason::kMonitorFault),
                             kBinding.producer_epoch,
                             2U,
                             2U);
  ASSERT_FALSE(full);
  EXPECT_EQ(full.error().code, ErrorCode::kBackpressured);
  EXPECT_EQ(receiver.pending_report_count(), 1U);
  EXPECT_EQ(receiver.state().report_sequence, 0U);

  const auto blocked = receiver.dispatch_pending(10);
  ASSERT_FALSE(blocked);
  EXPECT_EQ(blocked.error().token, "ENDPOINT_CLOCK_HEALTH_UPDATE_CAPACITY_EXHAUSTED");
  EXPECT_EQ(receiver.state().freshness, EndpointClockHealthFreshness::kNoReport);

  ASSERT_TRUE(receiver.pop_update().has_value());
  const auto applied = receiver.dispatch_pending(10);
  ASSERT_TRUE(applied);
  EXPECT_EQ(*applied, 1U);
  // Key assertion: after draining the update slot, dispatch succeeds, freshness becomes kFresh,
  // and deadline is accepted_at (10) + max_report_age (100 ns) = 110.
  EXPECT_EQ(receiver.state().freshness, EndpointClockHealthFreshness::kFresh);
  EXPECT_EQ(receiver.state().expiry_deadline_ns, 110);
}

// REQ: RUNTIME-009; CH7/CH11/CH13
// GoogleTest fatal assertions terminate these state-transition tests before optional access, but
// clang-tidy does not model the macro's control flow.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
TEST(EndpointClockHealthTest, UsesExactHalfOpenExpiryAndRejectsSupersededTickets) {
  // Key setup: first ticket issued at accepted_at=5 (deadline 105); second report at
  // accepted_at=55 (deadline 155) supersedes the first ticket before it fires.
  auto receiver = make_receiver();
  drain_updates(receiver);
  expect_report_accepted(receiver, ClockHealthReport::healthy(), 1U, 5);
  EXPECT_EQ(receiver.state().health_update_sequence, 2U);
  const auto first_ticket = receiver.expiry_ticket();
  ASSERT_TRUE(first_ticket.has_value());
  EXPECT_EQ(first_ticket->expiry_deadline_ns, 105);

  auto before = receiver.expire_if_due(*first_ticket, 104);
  ASSERT_TRUE(before);
  EXPECT_FALSE(*before);
  EXPECT_EQ(receiver.state().freshness, EndpointClockHealthFreshness::kFresh);

  expect_report_accepted(receiver, ClockHealthReport::healthy(), 2U, 55);
  EXPECT_EQ(receiver.state().health_update_sequence, 3U);
  const auto second_ticket = receiver.expiry_ticket();
  ASSERT_TRUE(second_ticket.has_value());
  EXPECT_EQ(second_ticket->expiry_deadline_ns, 155);

  auto superseded = receiver.expire_if_due(*first_ticket, 105);
  ASSERT_TRUE(superseded);
  EXPECT_FALSE(*superseded);
  EXPECT_EQ(receiver.state().freshness, EndpointClockHealthFreshness::kFresh);

  // Key assertion: expiry fires exactly at the second ticket's deadline (ns=155), half-open
  // [accepted, deadline), and sets freshness to kStale while preserving the prior kHealthy
  // assessment.
  auto at_deadline = receiver.expire_if_due(*second_ticket, 155);
  ASSERT_TRUE(at_deadline);
  EXPECT_TRUE(*at_deadline);
  EXPECT_EQ(receiver.state().freshness, EndpointClockHealthFreshness::kStale);
  EXPECT_EQ(receiver.state().assessment, EndpointClockHealthAssessmentState::kHealthy);
  EXPECT_EQ(receiver.state().reason, ClockHealthReason::kNone);
  EXPECT_EQ(receiver.current_disposition(), EndpointClockHealthDisposition::kSkip);
  const auto expiry_event = receiver.pop_update();
  ASSERT_TRUE(expiry_event.has_value());
  EXPECT_EQ(expiry_event->origin, EndpointClockHealthUpdateOrigin::kReportExpired);
  EXPECT_EQ(expiry_event->health_update_sequence, 4U);
  EXPECT_EQ(expiry_event->health_generation, 4U);
  EXPECT_EQ(expiry_event->evaluation_time_ns, 155);
}

// REQ: RUNTIME-009; Q61-T05
TEST(EndpointClockHealthTest, ResetsEpochsAndNeverPromotesReportReasonsToCausalWitnesses) {
  // Key setup: two reports accepted (kClockStalled, kTimestampRegression) followed by an
  // age-clock fault with cause_sequence=77; then producer epoch advances via reset.
  auto receiver = make_receiver();
  drain_updates(receiver);
  ASSERT_TRUE(receiver.submit_report(ClockHealthReport::degraded(ClockHealthReason::kClockStalled),
                                     kBinding.producer_epoch,
                                     1U,
                                     1U));
  const auto applied = receiver.dispatch_pending(10);
  ASSERT_TRUE(applied);
  EXPECT_EQ(*applied, 1U);
  auto report_event = receiver.pop_update();
  ASSERT_TRUE(report_event.has_value());
  EXPECT_EQ(report_event->origin, EndpointClockHealthUpdateOrigin::kReportAccepted);
  EXPECT_EQ(report_event->new_reason, ClockHealthReason::kClockStalled);
  EXPECT_FALSE(report_event->cause_sequence.has_value());

  ASSERT_TRUE(
      receiver.submit_report(ClockHealthReport::degraded(ClockHealthReason::kTimestampRegression),
                             kBinding.producer_epoch,
                             2U,
                             2U));
  const auto second_applied = receiver.dispatch_pending(15);
  ASSERT_TRUE(second_applied);
  EXPECT_EQ(*second_applied, 1U);
  report_event = receiver.pop_update();
  ASSERT_TRUE(report_event.has_value());
  EXPECT_EQ(report_event->origin, EndpointClockHealthUpdateOrigin::kReportAccepted);
  EXPECT_EQ(report_event->new_reason, ClockHealthReason::kTimestampRegression);
  EXPECT_FALSE(report_event->cause_sequence.has_value());

  ASSERT_TRUE(receiver.record_age_clock_fault(EndpointHealthAgeClockFault::kRegression, 20, 77U));
  EXPECT_EQ(receiver.state().freshness, EndpointClockHealthFreshness::kStale);
  EXPECT_EQ(receiver.state().reason, ClockHealthReason::kTimestampRegression);
  const auto fault_event = receiver.pop_update();
  ASSERT_TRUE(fault_event.has_value());
  EXPECT_EQ(fault_event->origin, EndpointClockHealthUpdateOrigin::kRuntimeHealthClockFault);
  EXPECT_EQ(fault_event->age_clock_fault, EndpointHealthAgeClockFault::kRegression);
  EXPECT_EQ(fault_event->cause_sequence, 77U);
  EXPECT_EQ(fault_event->new_reason, ClockHealthReason::kTimestampRegression);

  const EndpointClockHealthBindingIdentity next_binding{
      .run_generation = kBinding.run_generation,
      .producer_epoch = kBinding.producer_epoch + 1U,
      .clock_epoch = kBinding.clock_epoch,
  };
  ASSERT_TRUE(
      receiver.reset(EndpointClockHealthUpdateOrigin::kProducerEpochReset, next_binding, 30, 88U));
  EXPECT_EQ(receiver.state().freshness, EndpointClockHealthFreshness::kNoReport);
  EXPECT_EQ(receiver.state().assessment, EndpointClockHealthAssessmentState::kAbsent);
  EXPECT_FALSE(receiver.state().reason.has_value());
  EXPECT_FALSE(receiver.state().accepted_at_ns.has_value());

  const auto stale = receiver.submit_encoded(EndpointClockHealthControlRecord{
      .producer_epoch = kBinding.producer_epoch,
      .publisher_operation_sequence = 2U,
      .report_sequence = 2U,
      .assessment = static_cast<std::uint8_t>(ClockHealthAssessment::kHealthy),
      .reason = static_cast<std::uint8_t>(ClockHealthReason::kNone),
  });
  ASSERT_FALSE(stale);
  EXPECT_EQ(stale.error().code, ErrorCode::kStaleEndpoint);
  // Key assertion: encoded record with old producer_epoch is rejected as kStaleEndpoint after
  // the reset; freshness stays kNoReport confirming no mutation occurred.
  EXPECT_EQ(receiver.state().freshness, EndpointClockHealthFreshness::kNoReport);
}
// NOLINTEND(bugprone-unchecked-optional-access)

// REQ: RUNTIME-009; CH9/CH13
TEST(EndpointClockHealthTest, HotOperationsAreNoexceptAndDeadlineOverflowIsStatePreserving) {
  // Key setup: accepted_at = INT64_MAX - 99; deadline = accepted_at + 100 ns overflows int64.
  // Confirms the hot dispatch path is noexcept and rejects overflow without partial mutation.
  static_assert(noexcept(
      std::declval<EndpointClockHealthReceiver&>().submit_report(ClockHealthReport{}, 1U, 1U, 1U)));
  static_assert(
      noexcept(std::declval<EndpointClockHealthReceiver&>().dispatch_pending(std::int64_t{})));
  static_assert(noexcept(std::declval<EndpointClockHealthReceiver&>().expire_if_due(
      detail::EndpointClockHealthExpiryTicket{}, std::int64_t{})));
  static_assert(noexcept(std::declval<EndpointClockHealthReceiver&>().pop_update()));

  auto receiver = make_receiver();
  drain_updates(receiver);
  ASSERT_TRUE(
      receiver.submit_report(ClockHealthReport::healthy(), kBinding.producer_epoch, 1U, 1U));
  const auto result = receiver.dispatch_pending(std::numeric_limits<std::int64_t>::max() - 99);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().token, "ENDPOINT_CLOCK_HEALTH_DEADLINE_OVERFLOW");
  // Key assertion: freshness stays kNoReport and the ingress report remains pending,
  // so the overflow is fully recoverable without data loss.
  EXPECT_EQ(receiver.state().freshness, EndpointClockHealthFreshness::kNoReport);
  EXPECT_EQ(receiver.state().health_generation, 1U);
  EXPECT_EQ(receiver.pending_report_count(), 1U);
  EXPECT_EQ(receiver.pending_update_count(), 0U);
}

}  // namespace
}  // namespace holoscan
