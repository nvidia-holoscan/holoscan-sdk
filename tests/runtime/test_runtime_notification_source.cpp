/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>

#include <gtest/gtest.h>

#include <holoscan/time/manual_clock.hpp>

#include "core/readiness_source_internal.hpp"
#include "core/source_id_internal.hpp"
#include "runtime/runtime_notification_source.hpp"
#include "time/clock_registry_internal.hpp"

namespace holoscan {
namespace {

// Owns the ManualClock generation for the complete test so ClockRef remains live while senders
// post.
struct NotificationRegistryFixture {
  std::shared_ptr<ManualClock> clock;
  detail::ClockRegistry clock_registry;
  ClockRef clock_ref;
  std::uint32_t wakes{};
};

// Builds one readable, waitable, steady clock and returns both its owner and resolved reference.
[[nodiscard]] NotificationRegistryFixture make_fixture() {
  NotificationRegistryFixture fixture;
  fixture.clock = std::make_shared<ManualClock>(17);
  const ClockKey clock_key = ClockKey::named("notification-clock");
  detail::ClockRegistryBuilder builder;
  EXPECT_TRUE(builder.add_clock(clock_key,
                                fixture.clock,
                                detail::ClockCapabilityOptions{
                                    .can_read_now = true,
                                    .can_wait_until = true,
                                    .steady = true,
                                }));
  EXPECT_TRUE(builder.set_default(clock_key));
  auto registry = builder.build();
  EXPECT_TRUE(registry);
  if (!registry) {
    return fixture;
  }
  fixture.clock_registry = std::move(*registry);
  auto clock = fixture.clock_registry.resolve(clock_key);
  EXPECT_TRUE(clock);
  if (clock) {
    fixture.clock_ref = *clock;
  }
  return fixture;
}

// REQ: LIFECYCLE-005
// REQ: RUNTIME-025
TEST(RuntimeNotificationSourceRegistry, PublishesAndRollsBackBoundedSenderGenerations) {
  constexpr VertexId kOwner{1U};
  constexpr PlanGenerationId kPlanGeneration{3U};
  constexpr RunGenerationId kRunGeneration{5U};
  constexpr SourceId kSource = detail::make_source_id(1U);

  // Key setup: one capacity-1 physical source with two admitted sender records permits old and
  // candidate authority to coexist across a restart transaction.
  auto fixture = make_fixture();
  ASSERT_TRUE(fixture.clock_ref.id().valid());
  const detail::RuntimePortBindingRequest request{
      .plan_generation = kPlanGeneration,
      .run_generation = kRunGeneration,
      .kind = detail::RuntimeEventKind::kNotification,
      .source = kSource,
      .source_clock = fixture.clock_ref.id(),
      .reference_clock = fixture.clock_ref.id(),
      .execution_lane = 1U,
      .capacity = 1U,
      .source_generation = 1U,
  };
  const detail::RuntimeNotificationSourceDescriptor descriptor{
      .owner = kOwner,
      .request = request,
      .wake_requirement = ReadinessWakeRequirement::kDeploymentSelected,
      .wake_mode = WakeMode::kSleepingWaitset,
      .sender_reference_capacity = 2U,
      .clock = fixture.clock_ref,
      .wake_state = &fixture.wakes,
      .wake = [](void* opaque) noexcept { ++*static_cast<std::uint32_t*>(opaque); },
  };
  auto registry = detail::RuntimeNotificationSourceRegistry::create(
      std::span<const detail::RuntimeNotificationSourceDescriptor>{&descriptor, 1U});
  ASSERT_TRUE(registry);
  auto binding = (*registry)->bind(request, fixture.clock_ref.at_ns(17));
  ASSERT_TRUE(binding);

  NotificationSource source;
  detail::ReadinessSourceAccess::bind(
      source,
      detail::ReadinessSourceAccess::make_ref<ReadinessSourceKind::kNotification>(kSource,
                                                                                  kPlanGeneration));
  auto first = (*registry)->acquire_sender(
      kOwner, source, LifecycleStage::kArm, LifecycleTransitionPhase::kStartupForward);
  ASSERT_TRUE(first);
  // Key assertion: startup authority is inert before arm_startup(), then publishes one witness.
  const auto before_start = first->post();
  ASSERT_FALSE(before_start);
  EXPECT_EQ(before_start.error().code, ErrorCode::kNotReady);
  ASSERT_TRUE((*registry)->arm_startup());
  ASSERT_TRUE(first->post());

  detail::RuntimeEvent event;
  auto drained = (*binding)->drain(std::span<detail::RuntimeEvent>{&event, 1U});
  ASSERT_TRUE(drained);
  ASSERT_EQ(*drained, 1U);
  ASSERT_TRUE((*binding)->retire(event.source_sequence));

  const std::array owners{kOwner};
  // Key setup: fence ingress, mint the candidate in restart-forward Arm, and advance the physical
  // binding before atomically publishing matching sender generation 2.
  (*registry)->begin_restart(owners);
  const auto fenced_first = first->post();
  ASSERT_FALSE(fenced_first);
  EXPECT_EQ(fenced_first.error().code, ErrorCode::kNotReady);
  auto second = (*registry)->acquire_sender(
      kOwner, source, LifecycleStage::kArm, LifecycleTransitionPhase::kRestartForward);
  ASSERT_TRUE(second);
  const auto unpublished_second = second->post();
  ASSERT_FALSE(unpublished_second);
  EXPECT_EQ(unpublished_second.error().code, ErrorCode::kNotReady);
  auto generation = (*binding)->advance_source_generation();
  ASSERT_TRUE(generation);
  ASSERT_EQ(*generation, 2U);
  ASSERT_TRUE((*registry)->publish_restart(owners));
  const auto stale_first = first->post();
  ASSERT_FALSE(stale_first);
  EXPECT_EQ(stale_first.error().code, ErrorCode::kStaleEndpoint);
  *first = NotificationSender{};
  ASSERT_TRUE(second->post());

  drained = (*binding)->drain(std::span<detail::RuntimeEvent>{&event, 1U});
  ASSERT_TRUE(drained);
  ASSERT_EQ(*drained, 1U);
  ASSERT_TRUE((*binding)->retire(event.source_sequence));

  // Key assertion: rollback reactivates generation 2 and stales the unpublished candidate.
  (*registry)->begin_restart(owners);
  auto rolled_back = (*registry)->acquire_sender(
      kOwner, source, LifecycleStage::kArm, LifecycleTransitionPhase::kRestartForward);
  ASSERT_TRUE(rolled_back);
  (*registry)->rollback_restart(owners);
  ASSERT_TRUE(second->post());
  const auto stale_candidate = rolled_back->post();
  ASSERT_FALSE(stale_candidate);
  EXPECT_EQ(stale_candidate.error().code, ErrorCode::kStaleEndpoint);

  // Key assertion: terminal close wins over both active and already-stale sender states.
  (*registry)->close();
  const auto closed_second = second->post();
  ASSERT_FALSE(closed_second);
  EXPECT_EQ(closed_second.error().code, ErrorCode::kClosed);
  const auto closed_candidate = rolled_back->post();
  ASSERT_FALSE(closed_candidate);
  EXPECT_EQ(closed_candidate.error().code, ErrorCode::kClosed);
}

// REQ: LIFECYCLE-005
// REQ: RUNTIME-025
TEST(RuntimeNotificationSourceRegistry, StartupArmIsAtomicAcrossEverySource) {
  constexpr VertexId kFirstOwner{1U};
  constexpr VertexId kSecondOwner{2U};
  constexpr PlanGenerationId kPlanGeneration{3U};
  constexpr RunGenerationId kRunGeneration{5U};
  constexpr SourceId kFirstSource = detail::make_source_id(1U);
  constexpr SourceId kSecondSource = detail::make_source_id(2U);

  auto fixture = make_fixture();
  ASSERT_TRUE(fixture.clock_ref.id().valid());
  const auto make_request = [&](const SourceId source) {
    return detail::RuntimePortBindingRequest{
        .plan_generation = kPlanGeneration,
        .run_generation = kRunGeneration,
        .kind = detail::RuntimeEventKind::kNotification,
        .source = source,
        .source_clock = fixture.clock_ref.id(),
        .reference_clock = fixture.clock_ref.id(),
        .execution_lane = 1U,
        .capacity = 1U,
        .source_generation = 1U,
    };
  };
  const auto make_descriptor = [&](const VertexId owner,
                                   const detail::RuntimePortBindingRequest& request) {
    return detail::RuntimeNotificationSourceDescriptor{
        .owner = owner,
        .request = request,
        .wake_requirement = ReadinessWakeRequirement::kDeploymentSelected,
        .wake_mode = WakeMode::kSleepingWaitset,
        .sender_reference_capacity = 2U,
        .clock = fixture.clock_ref,
        .wake_state = &fixture.wakes,
        .wake = [](void* opaque) noexcept { ++*static_cast<std::uint32_t*>(opaque); },
    };
  };
  const auto first_request = make_request(kFirstSource);
  const auto second_request = make_request(kSecondSource);
  const std::array descriptors{make_descriptor(kFirstOwner, first_request),
                               make_descriptor(kSecondOwner, second_request)};
  auto registry = detail::RuntimeNotificationSourceRegistry::create(descriptors);
  ASSERT_TRUE(registry);

  // Key setup: bind and acquire the first source while leaving the second source unbound.
  auto first_binding = (*registry)->bind(first_request, fixture.clock_ref.at_ns(17));
  ASSERT_TRUE(first_binding);
  NotificationSource first_source;
  detail::ReadinessSourceAccess::bind(
      first_source,
      detail::ReadinessSourceAccess::make_ref<ReadinessSourceKind::kNotification>(kFirstSource,
                                                                                  kPlanGeneration));
  auto first_sender = (*registry)->acquire_sender(
      kFirstOwner, first_source, LifecycleStage::kArm, LifecycleTransitionPhase::kStartupForward);
  ASSERT_TRUE(first_sender);

  const auto incomplete_arm = (*registry)->arm_startup();
  ASSERT_FALSE(incomplete_arm);
  EXPECT_EQ(incomplete_arm.error().code, ErrorCode::kNotReady);
  const auto first_after_failure = first_sender->post();
  ASSERT_FALSE(first_after_failure);
  // Key assertion: failure on the later source does not activate the earlier sender.
  EXPECT_EQ(first_after_failure.error().code, ErrorCode::kNotReady);

  auto second_binding = (*registry)->bind(second_request, fixture.clock_ref.at_ns(17));
  ASSERT_TRUE(second_binding);
  NotificationSource second_source;
  detail::ReadinessSourceAccess::bind(
      second_source,
      detail::ReadinessSourceAccess::make_ref<ReadinessSourceKind::kNotification>(kSecondSource,
                                                                                  kPlanGeneration));
  auto second_sender = (*registry)->acquire_sender(
      kSecondOwner, second_source, LifecycleStage::kArm, LifecycleTransitionPhase::kStartupForward);
  ASSERT_TRUE(second_sender);
  ASSERT_TRUE((*registry)->arm_startup());
  EXPECT_TRUE(first_sender->post());
  EXPECT_TRUE(second_sender->post());
}

// REQ: LIFECYCLE-005
// REQ: RUNTIME-025
TEST(RuntimeNotificationSourceRegistry, RestartPublicationIsAtomicAcrossEverySource) {
  constexpr VertexId kFirstOwner{1U};
  constexpr VertexId kSecondOwner{2U};
  constexpr PlanGenerationId kPlanGeneration{3U};
  constexpr RunGenerationId kRunGeneration{5U};
  constexpr SourceId kFirstSource = detail::make_source_id(1U);
  constexpr SourceId kSecondSource = detail::make_source_id(2U);

  auto fixture = make_fixture();
  ASSERT_TRUE(fixture.clock_ref.id().valid());
  const auto make_request = [&](const SourceId source) {
    return detail::RuntimePortBindingRequest{
        .plan_generation = kPlanGeneration,
        .run_generation = kRunGeneration,
        .kind = detail::RuntimeEventKind::kNotification,
        .source = source,
        .source_clock = fixture.clock_ref.id(),
        .reference_clock = fixture.clock_ref.id(),
        .execution_lane = 1U,
        .capacity = 1U,
        .source_generation = 1U,
    };
  };
  const auto make_descriptor = [&](const VertexId owner,
                                   const detail::RuntimePortBindingRequest& request) {
    return detail::RuntimeNotificationSourceDescriptor{
        .owner = owner,
        .request = request,
        .wake_requirement = ReadinessWakeRequirement::kDeploymentSelected,
        .wake_mode = WakeMode::kSleepingWaitset,
        .sender_reference_capacity = 2U,
        .clock = fixture.clock_ref,
        .wake_state = &fixture.wakes,
        .wake = [](void* opaque) noexcept { ++*static_cast<std::uint32_t*>(opaque); },
    };
  };
  const auto first_request = make_request(kFirstSource);
  const auto second_request = make_request(kSecondSource);
  const std::array descriptors{make_descriptor(kFirstOwner, first_request),
                               make_descriptor(kSecondOwner, second_request)};
  auto registry = detail::RuntimeNotificationSourceRegistry::create(descriptors);
  ASSERT_TRUE(registry);
  auto first_binding = (*registry)->bind(first_request, fixture.clock_ref.at_ns(17));
  auto second_binding = (*registry)->bind(second_request, fixture.clock_ref.at_ns(17));
  ASSERT_TRUE(first_binding);
  ASSERT_TRUE(second_binding);

  NotificationSource first_source;
  NotificationSource second_source;
  detail::ReadinessSourceAccess::bind(
      first_source,
      detail::ReadinessSourceAccess::make_ref<ReadinessSourceKind::kNotification>(kFirstSource,
                                                                                  kPlanGeneration));
  detail::ReadinessSourceAccess::bind(
      second_source,
      detail::ReadinessSourceAccess::make_ref<ReadinessSourceKind::kNotification>(kSecondSource,
                                                                                  kPlanGeneration));
  auto first_sender = (*registry)->acquire_sender(
      kFirstOwner, first_source, LifecycleStage::kArm, LifecycleTransitionPhase::kStartupForward);
  auto second_sender = (*registry)->acquire_sender(
      kSecondOwner, second_source, LifecycleStage::kArm, LifecycleTransitionPhase::kStartupForward);
  ASSERT_TRUE(first_sender);
  ASSERT_TRUE(second_sender);
  ASSERT_TRUE((*registry)->arm_startup());

  const std::array owners{kFirstOwner, kSecondOwner};
  (*registry)->begin_restart(owners);
  auto first_candidate = (*registry)->acquire_sender(
      kFirstOwner, first_source, LifecycleStage::kArm, LifecycleTransitionPhase::kRestartForward);
  auto second_candidate = (*registry)->acquire_sender(
      kSecondOwner, second_source, LifecycleStage::kArm, LifecycleTransitionPhase::kRestartForward);
  ASSERT_TRUE(first_candidate);
  ASSERT_TRUE(second_candidate);

  // Key setup: only the first physical source advances, making the second preflight fail.
  auto first_generation = (*first_binding)->advance_source_generation();
  ASSERT_TRUE(first_generation);
  ASSERT_EQ(*first_generation, 2U);
  const auto incomplete_restart = (*registry)->publish_restart(owners);
  ASSERT_FALSE(incomplete_restart);
  EXPECT_EQ(incomplete_restart.error().code, ErrorCode::kStaleEndpoint);
  const auto first_after_failure = first_candidate->post();
  const auto second_after_failure = second_candidate->post();
  ASSERT_FALSE(first_after_failure);
  ASSERT_FALSE(second_after_failure);
  // Key assertion: the viable first candidate remains unpublished when the batch preflight fails.
  EXPECT_EQ(first_after_failure.error().code, ErrorCode::kNotReady);
  EXPECT_EQ(second_after_failure.error().code, ErrorCode::kNotReady);

  (*registry)->rollback_restart(owners);
  const auto stale_first_candidate = first_candidate->post();
  const auto stale_second_candidate = second_candidate->post();
  ASSERT_FALSE(stale_first_candidate);
  ASSERT_FALSE(stale_second_candidate);
  EXPECT_EQ(stale_first_candidate.error().code, ErrorCode::kStaleEndpoint);
  EXPECT_EQ(stale_second_candidate.error().code, ErrorCode::kStaleEndpoint);
  EXPECT_TRUE(second_sender->post());
}

// REQ: RUNTIME-025
TEST(RuntimeNotificationSourceRegistry, RejectsStaleBindingIdentityWithoutConsumingTheSource) {
  constexpr VertexId kOwner{1U};
  constexpr PlanGenerationId kPlanGeneration{3U};
  constexpr RunGenerationId kRunGeneration{5U};
  constexpr SourceId kSource = detail::make_source_id(1U);

  // Key setup: retain one exact generation tuple, then probe each stale identity dimension before
  // attempting the valid cold bind.
  auto fixture = make_fixture();
  ASSERT_TRUE(fixture.clock_ref.id().valid());
  const detail::RuntimePortBindingRequest request{
      .plan_generation = kPlanGeneration,
      .run_generation = kRunGeneration,
      .kind = detail::RuntimeEventKind::kNotification,
      .source = kSource,
      .source_clock = fixture.clock_ref.id(),
      .reference_clock = fixture.clock_ref.id(),
      .execution_lane = 1U,
      .capacity = 1U,
      .source_generation = 1U,
  };
  const detail::RuntimeNotificationSourceDescriptor descriptor{
      .owner = kOwner,
      .request = request,
      .wake_requirement = ReadinessWakeRequirement::kDeploymentSelected,
      .wake_mode = WakeMode::kSleepingWaitset,
      .sender_reference_capacity = 1U,
      .clock = fixture.clock_ref,
      .wake_state = &fixture.wakes,
      .wake = [](void* opaque) noexcept { ++*static_cast<std::uint32_t*>(opaque); },
  };
  auto registry = detail::RuntimeNotificationSourceRegistry::create(
      std::span<const detail::RuntimeNotificationSourceDescriptor>{&descriptor, 1U});
  ASSERT_TRUE(registry);

  std::array stale_requests{request, request, request};
  stale_requests[0].plan_generation = kPlanGeneration + 1U;
  stale_requests[1].run_generation = kRunGeneration + 1U;
  stale_requests[2].source_generation = request.source_generation + 1U;
  for (std::size_t index = 0U; index < stale_requests.size(); ++index) {
    SCOPED_TRACE(index);
    const auto rejected = (*registry)->bind(stale_requests[index], fixture.clock_ref.at_ns(17));
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, ErrorCode::kStaleEndpoint);
  }

  // Key assertion: stale requests never consume or cache authority; the exact request still binds
  // once, and the ordinary duplicate-binding outcome remains unchanged afterward.
  ASSERT_TRUE((*registry)->bind(request, fixture.clock_ref.at_ns(17)));
  const auto duplicate = (*registry)->bind(request, fixture.clock_ref.at_ns(17));
  ASSERT_FALSE(duplicate);
  EXPECT_EQ(duplicate.error().code, ErrorCode::kDuplicateName);
}

}  // namespace
}  // namespace holoscan
