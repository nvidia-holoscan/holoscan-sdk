/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

#include <holoscan/time/manual_clock.hpp>

#include "runtime/endpoint_clock_health_runtime.hpp"
#include "time/clock_registry_internal.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;
using detail::ClockCapabilityOptions;
using detail::ClockRegistry;
using detail::ClockRegistryBuilder;
using detail::EndpointClockHealthAssessmentState;
using detail::EndpointClockHealthBindingIdentity;
using detail::EndpointClockHealthFreshness;
using detail::EndpointClockHealthResetRequest;
using detail::EndpointClockHealthRuntimeCoordinator;
using detail::EndpointClockHealthState;
using detail::EndpointClockHealthUpdateOrigin;

// Shared state bundle for runtime coordinator tests: two ManualClocks (capture and age),
// a ClockRegistry with a 3 ns affine conversion from capture to age, and resolved ClockIds.
// Constructed via make_runtime_health_fixture(); used by every test in this file.
struct RuntimeHealthFixture {
  std::shared_ptr<ManualClock> capture_clock;
  std::shared_ptr<ManualClock> age_clock;
  ClockRegistry registry;
  std::array<ClockPlanRecord, 2U> clocks;
  PlanGenerationId plan_generation{7U};
  ClockId capture_id{};
  ClockId age_id{};
};

// Builds a RuntimeHealthFixture with two independently-controllable ManualClocks registered
// under named keys, a 3 ns affine conversion (age = capture + 3), and a populated ClockRegistry.
// Representative test: MaterializesOnlyMonitoredReceiversAndPlanClocks.
[[nodiscard]] RuntimeHealthFixture make_runtime_health_fixture() {
  auto capture = std::make_shared<ManualClock>(0);
  auto age = std::make_shared<ManualClock>(0);
  const ClockKey capture_key = ClockKey::named("capture");
  const ClockKey age_key = ClockKey::named("health-age");
  ClockRegistryBuilder builder;
  EXPECT_TRUE(builder.add_clock(capture_key,
                                capture,
                                ClockCapabilityOptions{
                                    .can_read_now = true,
                                    .can_wait_until = true,
                                    .steady = true,
                                }));
  EXPECT_TRUE(builder.add_clock(age_key,
                                age,
                                ClockCapabilityOptions{
                                    .can_read_now = true,
                                    .can_wait_until = true,
                                    .steady = true,
                                }));
  EXPECT_TRUE(builder.set_default(age_key));
  EXPECT_TRUE(builder.add_conversion(capture_key, age_key, AffineClockConversion{.offset_ns = 3}));
  auto registry = builder.build();
  EXPECT_TRUE(registry);
  if (!registry) {
    return {};
  }
  const auto capture_ref = registry->resolve(capture_key);
  const auto age_ref = registry->resolve(age_key);
  EXPECT_TRUE(capture_ref);
  EXPECT_TRUE(age_ref);
  if (!capture_ref || !age_ref) {
    return {};
  }

  const std::array clocks{
      ClockPlanRecord{
          .id = capture_ref->id(),
          .graph = GraphId{1U},
          .key = capture_key,
          .shared = true,
          .can_read_now = true,
          .can_wait_until = true,
          .steady = true,
          .capability_evidence_id = "holoscan.test.capture-clock.v1",
      },
      ClockPlanRecord{
          .id = age_ref->id(),
          .graph = GraphId{1U},
          .key = age_key,
          .shared = true,
          .is_default = true,
          .can_read_now = true,
          .can_wait_until = true,
          .steady = true,
          .independent_progress = true,
          .finite_deadline = true,
          .external_driver_admitted = true,
          .capability_evidence_id = "holoscan.test.clock-health-age.v1",
      },
  };
  return RuntimeHealthFixture{
      .capture_clock = std::move(capture),
      .age_clock = std::move(age),
      .registry = std::move(registry).value(),
      .clocks = clocks,
      .capture_id = capture_ref->id(),
      .age_id = age_ref->id(),
  };
}

// Returns an EndpointClockHealthPlanRecord with monitoring disabled (kUnavailable capability,
// kNotRequired requirement). Paired with monitored_plan() to exercise the non-monitored code
// path through the coordinator. Representative test:
// MaterializesOnlyMonitoredReceiversAndPlanClocks.
[[nodiscard]] EndpointClockHealthPlanRecord ordinary_plan(PlanGenerationId generation) {
  return EndpointClockHealthPlanRecord{
      .endpoint = PortId{10U},
      .plan_generation = generation,
      .capability = ClockHealthCapability::kUnavailable,
      .requirement = ClockHealthRequirement::kNotRequired,
  };
}

// Returns a fully-configured EndpointClockHealthPlanRecord wired to the fixture's capture
// and age clocks. Mirrors test_endpoint_clock_health.cpp's monitored_plan() but binds to
// runtime-resolved ClockIds. Representative test: AppliesUnknownHealthyDegradedAndUnhealthyStates.
[[nodiscard]] EndpointClockHealthPlanRecord monitored_plan(const RuntimeHealthFixture& fixture,
                                                           PortId endpoint = PortId{11U}) {
  return EndpointClockHealthPlanRecord{
      .endpoint = endpoint,
      .plan_generation = fixture.plan_generation,
      .capture_clock = fixture.capture_id,
      .producer_epoch_source = EndpointProducerEpochSource::kAttachmentGeneration,
      .capability = ClockHealthCapability::kStatusReports,
      .requirement = ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 100ns,
      .health_age_clock = fixture.age_id,
      .fixed_report_ingress_capacity = 1U,
      .fixed_update_capacity = 16U,
      .ingress_capacity_origin = EndpointClockHealthBoundOrigin::kPortableMinimum,
      .per_dispatch_report_budget = 1U,
      .publisher_operation_sequence_id = "holoscan.clock_health.publisher-operation-sequence.v1",
      .report_sequence_id = "holoscan.clock_health.report-sequence.v1",
      .health_update_sequence_id = "holoscan.clock_health.update-sequence.v1",
      .monitor_path_kind = EndpointClockHealthMonitorPathKind::kPublisherSerialized,
      .no_report_disposition = EndpointClockHealthDisposition::kSkip,
      .degraded_disposition = EndpointClockHealthDisposition::kDegraded,
      .unhealthy_disposition = EndpointClockHealthDisposition::kFailRun,
      .stale_disposition = EndpointClockHealthDisposition::kSkip,
      .unspecified_reason_disposition = ClockHealthUnspecifiedDisposition::kAdmitted,
      .clock_health_update_witness_id = "holoscan.event.clock-health-update.v1",
      .trace_loss_disposition = EndpointClockHealthTraceLossDisposition::kFailClosed,
  };
}

// Creates a coordinator with one ordinary endpoint (PortId 10) and two monitored endpoints
// (PortId 11 and 12). Used by batch-reset tests that need multiple receivers to verify
// atomicity. Representative test: BatchResetPreflightsEveryReceiverBeforeMutation.
[[nodiscard]] std::unique_ptr<EndpointClockHealthRuntimeCoordinator> make_two_endpoint_coordinator(
    RuntimeHealthFixture& fixture) {
  const std::array health_plan{
      ordinary_plan(fixture.plan_generation),
      monitored_plan(fixture, PortId{11U}),
      monitored_plan(fixture, PortId{12U}),
  };
  auto created = EndpointClockHealthRuntimeCoordinator::create(
      health_plan, fixture.clocks, fixture.plan_generation, 9U, fixture.registry, 5U, 2U);
  EXPECT_TRUE(created) << (created ? std::string_view{} : created.error().token);
  return created ? std::move(created).value() : nullptr;
}

// Creates a coordinator with one ordinary endpoint (PortId 10) and one monitored endpoint
// (PortId 11). Covers the common single-endpoint case for most tests.
// Representative test: MaterializesOnlyMonitoredReceiversAndPlanClocks.
[[nodiscard]] std::unique_ptr<EndpointClockHealthRuntimeCoordinator> make_coordinator(
    RuntimeHealthFixture& fixture) {
  const std::array health_plan{
      ordinary_plan(fixture.plan_generation),
      monitored_plan(fixture),
  };
  auto created = EndpointClockHealthRuntimeCoordinator::create(
      health_plan, fixture.clocks, fixture.plan_generation, 9U, fixture.registry, 5U, 2U);
  EXPECT_TRUE(created) << (created ? std::string_view{} : created.error().token);
  return created ? std::move(created).value() : nullptr;
}

// Clears the update queue on PortId 11 so tests begin after the initial kAttachmentReset event.
void drain_updates(EndpointClockHealthRuntimeCoordinator& coordinator) {
  while (coordinator.pop_update(PortId{11U}).has_value()) {
  }
}

// REQ: RUNTIME-009; Q61-T07--T11
TEST(EndpointClockHealthRuntimeTest, MaterializesOnlyMonitoredReceiversAndPlanClocks) {
  // Key setup: coordinator has one ordinary (PortId 10, kUnavailable) and one monitored
  // (PortId 11, kStatusReports) endpoint; only the monitored one allocates a receiver.
  auto fixture = make_runtime_health_fixture();
  auto coordinator = make_coordinator(fixture);
  ASSERT_TRUE(coordinator);

  EXPECT_EQ(coordinator->receiver_count(), 1U);
  EXPECT_EQ(coordinator->receiver(PortId{10U}), nullptr);
  ASSERT_NE(coordinator->receiver(PortId{11U}), nullptr);
  EXPECT_EQ(coordinator->receiver(PortId{11U})->state().freshness,
            EndpointClockHealthFreshness::kNoReport);
  EXPECT_EQ(coordinator->receiver(PortId{11U})->current_disposition(),
            EndpointClockHealthDisposition::kSkip);

  const auto capture = fixture.registry.resolve(fixture.capture_id);
  const auto age = fixture.registry.resolve(fixture.age_id);
  ASSERT_TRUE(capture);
  ASSERT_TRUE(age);
  EXPECT_EQ(capture->id(), fixture.capture_id);
  EXPECT_EQ(age->id(), fixture.age_id);
  // Key assertion: receiver_count==1 (PortId 10 returns nullptr); the age clock's wait_until
  // drives the ManualClock, proving the registry wired the correct clock instance.
  ASSERT_TRUE(fixture.registry.wait_until(fixture.age_id, 7));
  EXPECT_EQ(fixture.age_clock->timestamp(), 7);
}

// REQ: RUNTIME-009; Q61-T02/T03/T13
TEST(EndpointClockHealthRuntimeTest, AppliesUnknownHealthyDegradedAndUnhealthyStates) {
  // Key setup: age_clock advanced manually before each dispatch; three states applied in
  // order: healthy → degraded → unhealthy.
  auto fixture = make_runtime_health_fixture();
  auto coordinator = make_coordinator(fixture);
  ASSERT_TRUE(coordinator);
  drain_updates(*coordinator);

  const auto apply = [&](ClockHealthReport report, std::uint64_t sequence, std::int64_t now_ns) {
    fixture.age_clock->advance_to(now_ns);
    ASSERT_TRUE(coordinator->submit_report(PortId{11U}, report, 5U, sequence, sequence));
    const auto dispatched = coordinator->dispatch_pending(PortId{11U});
    ASSERT_TRUE(dispatched);
    EXPECT_EQ(*dispatched, 1U);
    ASSERT_TRUE(coordinator->pop_update(PortId{11U}).has_value());
  };

  apply(ClockHealthReport::healthy(), 1U, 10);
  EXPECT_EQ(coordinator->receiver(PortId{11U})->state().assessment,
            EndpointClockHealthAssessmentState::kHealthy);
  EXPECT_TRUE(coordinator->receiver(PortId{11U})->satisfies_requirement());

  apply(ClockHealthReport::degraded(ClockHealthReason::kSynchronizationLockLost), 2U, 20);
  EXPECT_EQ(coordinator->receiver(PortId{11U})->current_disposition(),
            EndpointClockHealthDisposition::kDegraded);

  apply(ClockHealthReport::unhealthy(ClockHealthReason::kTimestampSourceFault), 3U, 30);
  // Key assertion: final disposition is kFailRun after the unhealthy submission, confirming
  // the runtime coordinator routes health state through the same disposition map as the
  // pure receiver.
  EXPECT_EQ(coordinator->receiver(PortId{11U})->current_disposition(),
            EndpointClockHealthDisposition::kFailRun);
}

// REQ: RUNTIME-009; Q61-T08/T09
// GoogleTest fatal assertions terminate this test before optional access, but clang-tidy does not
// model the macro's control flow.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
TEST(EndpointClockHealthRuntimeTest, ExpiresAtTheExactHalfOpenDeadline) {
  // Key setup: healthy report submitted at age_ns=10; max_report_age=100 ns so deadline=110.
  // ManualClock allows exact boundary probing without real-time dependency.
  auto fixture = make_runtime_health_fixture();
  auto coordinator = make_coordinator(fixture);
  ASSERT_TRUE(coordinator);
  drain_updates(*coordinator);

  fixture.age_clock->advance_to(10);
  ASSERT_TRUE(coordinator->submit_report(PortId{11U}, ClockHealthReport::healthy(), 5U, 1U, 1U));
  ASSERT_EQ(coordinator->dispatch_pending(PortId{11U}).value(), 1U);
  (void)coordinator->pop_update(PortId{11U});
  const auto deadline = coordinator->next_expiry(fixture.age_id);
  ASSERT_TRUE(deadline.has_value());
  EXPECT_EQ(deadline->endpoint, PortId{11U});
  EXPECT_EQ(deadline->clock, fixture.age_id);
  EXPECT_EQ(deadline->ticket.expiry_deadline_ns, 110);

  fixture.age_clock->advance_to(109);
  ASSERT_EQ(coordinator->expire_due().value(), 0U);
  // Key assertion: expire_due returns 1 exactly at ns=110 and sets freshness to kStale;
  // next_expiry returns nullopt confirming no further ticket is queued.
  fixture.age_clock->advance_to(110);
  ASSERT_EQ(coordinator->expire_due().value(), 1U);
  EXPECT_EQ(coordinator->receiver(PortId{11U})->state().freshness,
            EndpointClockHealthFreshness::kStale);
  const auto update = coordinator->pop_update(PortId{11U});
  ASSERT_TRUE(update.has_value());
  EXPECT_EQ(update->origin, EndpointClockHealthUpdateOrigin::kReportExpired);
  EXPECT_EQ(update->evaluation_time_ns, 110);
  EXPECT_FALSE(coordinator->next_expiry(fixture.age_id).has_value());
}
// NOLINTEND(bugprone-unchecked-optional-access)

// REQ: RUNTIME-009; Q61-T14/T15
TEST(EndpointClockHealthRuntimeTest, ResetsEpochsAndPreservesOverflowFailureWithoutMutation) {
  // Key setup: age_clock at INT64_MAX - 50; deadline = age + 100 ns overflows int64_t.
  // Confirms overflow fails the dispatch without corrupting receiver state.
  auto fixture = make_runtime_health_fixture();
  auto coordinator = make_coordinator(fixture);
  ASSERT_TRUE(coordinator);
  drain_updates(*coordinator);

  fixture.age_clock->advance_to(std::numeric_limits<std::int64_t>::max() - 50);
  ASSERT_TRUE(coordinator->submit_report(PortId{11U}, ClockHealthReport::healthy(), 5U, 1U, 1U));
  const auto overflow = coordinator->dispatch_pending(PortId{11U});
  ASSERT_FALSE(overflow);
  EXPECT_EQ(overflow.error().code, ErrorCode::kProtocolError);
  EXPECT_EQ(overflow.error().token, "ENDPOINT_CLOCK_HEALTH_DEADLINE_OVERFLOW");
  EXPECT_EQ(coordinator->receiver(PortId{11U})->state().freshness,
            EndpointClockHealthFreshness::kNoReport);
  EXPECT_EQ(coordinator->receiver(PortId{11U})->pending_report_count(), 1U);

  // The failed report must be drained before the externally synchronized reset.
  fixture.age_clock->advance_to(1);
  ASSERT_EQ(coordinator->dispatch_pending(PortId{11U}).value(), 1U);
  (void)coordinator->pop_update(PortId{11U});
  ASSERT_TRUE(coordinator->reset(PortId{11U},
                                 EndpointClockHealthUpdateOrigin::kProducerEpochReset,
                                 EndpointClockHealthBindingIdentity{
                                     .run_generation = 9U, .producer_epoch = 6U, .clock_epoch = 2U},
                                 41U));
  EXPECT_EQ(coordinator->receiver(PortId{11U})->state().freshness,
            EndpointClockHealthFreshness::kNoReport);
  EXPECT_EQ(coordinator->receiver(PortId{11U})->state().binding.producer_epoch, 6U);
  (void)coordinator->pop_update(PortId{11U});

  ASSERT_TRUE(
      coordinator->reset(PortId{11U},
                         EndpointClockHealthUpdateOrigin::kClockEpochReset,
                         EndpointClockHealthBindingIdentity{
                             .run_generation = 9U, .producer_epoch = 6U, .clock_epoch = 3U}));
  // Key assertion: clock_epoch advances to 3 after kClockEpochReset, confirming both producer
  // and clock epoch rotation paths are independently exercised.
  EXPECT_EQ(coordinator->receiver(PortId{11U})->state().binding.clock_epoch, 3U);
}

// REQ: RUNTIME-009; Q61-T14/T15
TEST(EndpointClockHealthRuntimeTest, BatchResetPreflightsEveryReceiverBeforeMutation) {
  // Key setup: a pending report on PortId 12 must block the entire batch reset — including
  // PortId 11 which has no pending work — to guarantee atomic all-or-nothing semantics.
  auto fixture = make_runtime_health_fixture();
  auto coordinator = make_two_endpoint_coordinator(fixture);
  ASSERT_TRUE(coordinator);
  while (coordinator->pop_update(PortId{11U}).has_value()) {
  }
  while (coordinator->pop_update(PortId{12U}).has_value()) {
  }

  // A pending report on the later endpoint must reject the complete reset transaction without
  // advancing the earlier endpoint's binding or structural health sequences.
  ASSERT_TRUE(coordinator->submit_report(PortId{12U}, ClockHealthReport::healthy(), 5U, 1U, 1U));
  const EndpointClockHealthState first_before = coordinator->receiver(PortId{11U})->state();
  const EndpointClockHealthState second_before = coordinator->receiver(PortId{12U})->state();
  const std::array requests{
      EndpointClockHealthResetRequest{
          .endpoint = PortId{11U},
          .origin = EndpointClockHealthUpdateOrigin::kProducerEpochReset,
          .binding = {.run_generation = 9U, .producer_epoch = 6U, .clock_epoch = 2U},
      },
      EndpointClockHealthResetRequest{
          .endpoint = PortId{12U},
          .origin = EndpointClockHealthUpdateOrigin::kProducerEpochReset,
          .binding = {.run_generation = 9U, .producer_epoch = 6U, .clock_epoch = 2U},
      },
  };
  auto blocked = coordinator->reset_batch(requests);
  ASSERT_FALSE(blocked);
  EXPECT_EQ(blocked.error().code, ErrorCode::kNotReady);
  EXPECT_EQ(coordinator->receiver(PortId{11U})->state().binding, first_before.binding);
  EXPECT_EQ(coordinator->receiver(PortId{11U})->state().health_update_sequence,
            first_before.health_update_sequence);
  EXPECT_EQ(coordinator->receiver(PortId{11U})->state().health_generation,
            first_before.health_generation);
  EXPECT_EQ(coordinator->receiver(PortId{12U})->state().binding, second_before.binding);
  EXPECT_EQ(coordinator->receiver(PortId{12U})->state().health_update_sequence,
            second_before.health_update_sequence);
  EXPECT_EQ(coordinator->receiver(PortId{12U})->state().health_generation,
            second_before.health_generation);
  EXPECT_EQ(coordinator->receiver(PortId{11U})->pending_update_count(), 0U);
  EXPECT_EQ(coordinator->receiver(PortId{12U})->pending_report_count(), 1U);

  fixture.age_clock->advance_to(1);
  ASSERT_EQ(coordinator->dispatch_pending(PortId{12U}).value(), 1U);
  (void)coordinator->pop_update(PortId{12U});
  const std::uint64_t first_update_sequence =
      coordinator->receiver(PortId{11U})->state().health_update_sequence;
  const std::uint64_t first_generation =
      coordinator->receiver(PortId{11U})->state().health_generation;
  const std::uint64_t second_update_sequence =
      coordinator->receiver(PortId{12U})->state().health_update_sequence;
  const std::uint64_t second_generation =
      coordinator->receiver(PortId{12U})->state().health_generation;
  ASSERT_TRUE(coordinator->reset_batch(requests));

  // Key assertion: after the unblocked batch reset, both endpoints show producer_epoch=6,
  // kAbsent assessment, and kNoReport freshness with no residual reason or timestamp.
  for (const PortId endpoint : {PortId{11U}, PortId{12U}}) {
    const EndpointClockHealthState& state = coordinator->receiver(endpoint)->state();
    EXPECT_EQ(state.binding.producer_epoch, 6U);
    EXPECT_EQ(state.assessment, EndpointClockHealthAssessmentState::kAbsent);
    EXPECT_EQ(state.freshness, EndpointClockHealthFreshness::kNoReport);
    EXPECT_FALSE(state.reason.has_value());
    EXPECT_FALSE(state.accepted_at_ns.has_value());
    EXPECT_FALSE(state.expiry_deadline_ns.has_value());
    EXPECT_EQ(state.publisher_operation_sequence, 0U);
    EXPECT_EQ(state.report_sequence, 0U);
  }
  EXPECT_EQ(coordinator->receiver(PortId{11U})->state().health_update_sequence,
            first_update_sequence + 1U);
  EXPECT_EQ(coordinator->receiver(PortId{11U})->state().health_generation, first_generation + 1U);
  EXPECT_EQ(coordinator->receiver(PortId{12U})->state().health_update_sequence,
            second_update_sequence + 1U);
  EXPECT_EQ(coordinator->receiver(PortId{12U})->state().health_generation, second_generation + 1U);
}

// REQ: RUNTIME-009; Q61-T07/T12
TEST(EndpointClockHealthRuntimeTest, RejectsUnqualifiedAgeClocksWithAPreciseToken) {
  // Key setup: the age clock's independent_progress flag is cleared after fixture construction,
  // simulating a plan that names a clock lacking the required independent-progress capability.
  auto fixture = make_runtime_health_fixture();
  auto clocks = fixture.clocks;
  const auto age = std::find_if(
      clocks.begin(), clocks.end(), [&](const auto& clock) { return clock.id == fixture.age_id; });
  ASSERT_NE(age, clocks.end());
  age->independent_progress = false;

  const std::array health_plan{monitored_plan(fixture)};
  const auto created = EndpointClockHealthRuntimeCoordinator::create(
      health_plan, clocks, fixture.plan_generation, 9U, fixture.registry);
  ASSERT_FALSE(created);
  // Key assertion: creation fails with kInvalidArgument and the precise token
  // ENDPOINT_CLOCK_HEALTH_RUNTIME_PLAN_INVALID, not a generic internal error.
  EXPECT_EQ(created.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(created.error().token, "ENDPOINT_CLOCK_HEALTH_RUNTIME_PLAN_INVALID");
}

}  // namespace
}  // namespace holoscan
