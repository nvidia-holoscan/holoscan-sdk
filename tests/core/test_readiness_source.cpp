/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <concepts>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <holoscan/core/activation.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/readiness_source.hpp>
#include <holoscan/core/source_id.hpp>
#include <holoscan/time/clock_registry.hpp>

#include <core/readiness_source_internal.hpp>
#include <core/source_id_internal.hpp>

namespace holoscan {
namespace {

// Concepts used to verify that NotificationSender exposes only the parameterless post()
// and that the payload/timestamp/capacity overloads are intentionally absent.
template <typename Sender>
concept HasPayloadPost = requires(const Sender& sender) { sender.post(std::uint64_t{1}); };

template <typename Sender>
concept HasTimestampPost = requires(const Sender& sender, TimePoint time) { sender.post(time); };

template <typename Sender>
concept HasCapacityMutation = requires(Sender& sender) { sender.capacity(2); };

// Tracks retain/release call counts so tests can verify that NotificationSender
// correctly invokes lifecycle hooks when it is copied and destroyed.
struct SenderLifecycleProbe {
  std::uint32_t retains{};
  std::uint32_t releases{};
};

void retain_sender(void* state) noexcept {
  ++static_cast<SenderLifecycleProbe*>(state)->retains;
}

void release_sender(void* state) noexcept {
  ++static_cast<SenderLifecycleProbe*>(state)->releases;
}

static_assert(!std::copy_constructible<ReadableSource>);
static_assert(!std::move_constructible<ReadableSource>);
static_assert(!std::copy_constructible<NotificationSource>);
static_assert(!std::move_constructible<NotificationSource>);
static_assert(!std::copy_constructible<DoorbellSource>);
static_assert(!std::move_constructible<DoorbellSource>);
static_assert(!std::copy_constructible<SpinSource>);
static_assert(!std::move_constructible<SpinSource>);

static_assert(std::copy_constructible<ReadableSourceRef>);
static_assert(std::copy_constructible<NotificationSourceRef>);
static_assert(std::copy_constructible<DoorbellSourceRef>);
static_assert(std::copy_constructible<SpinSourceRef>);
static_assert(std::convertible_to<const ReadableSource&, ReadableSourceRef>);
static_assert(std::convertible_to<const NotificationSource&, NotificationSourceRef>);
static_assert(std::convertible_to<const DoorbellSource&, DoorbellSourceRef>);
static_assert(std::convertible_to<const SpinSource&, SpinSourceRef>);
static_assert(!std::convertible_to<const ReadableSource&, NotificationSourceRef>);
static_assert(!std::convertible_to<const NotificationSource&, DoorbellSourceRef>);
static_assert(noexcept(static_cast<ReadableSourceRef>(std::declval<const ReadableSource&>())));
static_assert(
    noexcept(static_cast<NotificationSourceRef>(std::declval<const NotificationSource&>())));
static_assert(noexcept(static_cast<DoorbellSourceRef>(std::declval<const DoorbellSource&>())));
static_assert(noexcept(static_cast<SpinSourceRef>(std::declval<const SpinSource&>())));
static_assert(std::is_trivially_copyable_v<ReadableSourceRef>);
static_assert(std::is_trivially_copyable_v<NotificationSourceRef>);
static_assert(ReadableSourceRef::kind() == ReadinessSourceKind::kReadable);
static_assert(NotificationSourceRef::kind() == ReadinessSourceKind::kNotification);
static_assert(DoorbellSourceRef::kind() == ReadinessSourceKind::kDoorbell);
static_assert(SpinSourceRef::kind() == ReadinessSourceKind::kSpin);

static_assert(std::default_initializable<NotificationSender>);
static_assert(std::copy_constructible<NotificationSender>);
static_assert(std::move_constructible<NotificationSender>);
static_assert(std::is_copy_assignable_v<NotificationSender>);
static_assert(std::is_move_assignable_v<NotificationSender>);
static_assert(std::same_as<decltype(std::declval<const NotificationSender&>().post()),
                           expected<void, Error>>);
static_assert(noexcept(std::declval<const NotificationSender&>().post()));
static_assert(!HasPayloadPost<NotificationSender>);
static_assert(!HasTimestampPost<NotificationSender>);
static_assert(!HasCapacityMutation<NotificationSender>);

static_assert(std::same_as<decltype(std::declval<OperatorSpec&>().readable_source(
                               std::declval<ReadableSource&>(), "readable")),
                           SingleWitnessSourceSpecBuilder&>);
static_assert(std::same_as<decltype(std::declval<OperatorSpec&>().notification_source(
                               std::declval<NotificationSource&>(), "notification")),
                           NotificationSourceSpecBuilder&>);
static_assert(std::same_as<decltype(std::declval<OperatorSpec&>().doorbell_source(
                               std::declval<DoorbellSource&>(), "doorbell")),
                           DoorbellSourceSpecBuilder&>);
static_assert(std::same_as<decltype(std::declval<OperatorSpec&>().spin_source(
                               std::declval<SpinSource&>(), "spin")),
                           SingleWitnessSourceSpecBuilder&>);

static_assert(std::same_as<decltype(std::declval<SingleWitnessSourceSpecBuilder&>().capacity()),
                           SingleWitnessSourceSpecBuilder&>);
static_assert(std::same_as<decltype(std::declval<SingleWitnessSourceSpecBuilder&>().clock(
                               std::declval<ClockKey>())),
                           SingleWitnessSourceSpecBuilder&>);
static_assert(std::same_as<decltype(std::declval<NotificationSourceSpecBuilder&>().capacity(4)),
                           NotificationSourceSpecBuilder&>);
static_assert(std::same_as<
              decltype(std::declval<NotificationSourceSpecBuilder&>().sender_reference_capacity(8)),
              NotificationSourceSpecBuilder&>);
static_assert(std::same_as<decltype(std::declval<DoorbellSourceSpecBuilder&>().capacity(4)),
                           DoorbellSourceSpecBuilder&>);

// REQ: OPERATOR-006
// Verifies that default-constructed refs evaluate to false (no bound source) and that
// a default NotificationSender returns kInvalidArgument on post() rather than crashing.
TEST(ReadinessSource, DefaultRefsAndNotificationSenderAreInvalidValues) {
  const ReadableSourceRef readable;
  const NotificationSourceRef notification;
  const DoorbellSourceRef doorbell;
  const SpinSourceRef spin;

  EXPECT_FALSE(static_cast<bool>(readable));
  EXPECT_FALSE(static_cast<bool>(notification));
  EXPECT_FALSE(static_cast<bool>(doorbell));
  EXPECT_FALSE(static_cast<bool>(spin));

  const NotificationSender sender;
  const auto result = sender.post();
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
}

// REQ: OPERATOR-006
// Verifies that retain/release lifecycle hooks fire on copy and destruction even when
// the post callback is null — the lifecycle is independent of the post validity.
TEST(ReadinessSource, SenderRetainsTransferredLifecycleEvenWhenPostCallbackIsInvalid) {
  // Key setup: sender built with retain/release hooks but a null post callback.
  SenderLifecycleProbe probe;
  {
    const auto sender = detail::ReadinessSourceAccess::make_sender(
        &probe, &retain_sender, &release_sender, nullptr);
    const auto result = sender.post();
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);

    {
      const auto copy = sender;  // NOLINT(performance-unnecessary-copy-initialization): copy tested
      // Key assertion: copy increments retains; inner-scope destruction increments releases.
      EXPECT_EQ(probe.retains, 1);
      EXPECT_EQ(probe.releases, 0);
      EXPECT_FALSE(copy.post());
    }
    EXPECT_EQ(probe.releases, 1);
  }
  EXPECT_EQ(probe.releases, 2);
}

// REQ: OPERATOR-006
// Confirms that the OnReadable/OnNotified/OnDoorbell/OnSpin aggregate trigger descriptors
// hold the matching typed ref field rather than a type-erased void pointer.
TEST(ReadinessSource, TriggerDescriptorsContainOnlyTypedLogicalRefs) {
  static_assert(std::is_same_v<decltype(OnReadable::event), ReadableSourceRef>);
  static_assert(std::is_same_v<decltype(OnNotified::event), NotificationSourceRef>);
  static_assert(std::is_same_v<decltype(OnDoorbell::event), DoorbellSourceRef>);
  static_assert(std::is_same_v<decltype(OnSpin::event), SpinSourceRef>);
}

// REQ: OPERATOR-006
// A freshly constructed (unbound) source declaration must convert to an invalid ref;
// binding happens later by the runtime, so operator constructors are safe without it.
TEST(ReadinessSource, UnboundDeclarationsConvertToInvalidTypedRefs) {
  const ReadableSource readable;
  const NotificationSource notification;
  const DoorbellSource doorbell;
  const SpinSource spin;

  const ReadableSourceRef readable_ref = readable;
  const NotificationSourceRef notification_ref = notification;
  const DoorbellSourceRef doorbell_ref = doorbell;
  const SpinSourceRef spin_ref = spin;

  EXPECT_FALSE(static_cast<bool>(readable_ref));
  EXPECT_FALSE(static_cast<bool>(notification_ref));
  EXPECT_FALSE(static_cast<bool>(doorbell_ref));
  EXPECT_FALSE(static_cast<bool>(spin_ref));
}

// REQ: OPERATOR-006
// Exercises the full bind + trigger-descriptor pathway: after the runtime binds a source,
// the typed ref captured in OnReadable/etc. should carry the assigned source ID.
TEST(ReadinessSource, CanonicalTriggerSyntaxPreservesBoundTypedRefs) {
  // Key setup: manually bind four source kinds using internal access helpers, then
  // wrap each in its corresponding trigger descriptor (OnReadable, etc.).
  ReadableSource readable;
  NotificationSource notification;
  DoorbellSource doorbell;
  SpinSource spin;

  detail::ReadinessSourceAccess::bind(
      readable,
      detail::ReadinessSourceAccess::make_ref<ReadinessSourceKind::kReadable>(
          detail::make_source_id(11), 101));
  detail::ReadinessSourceAccess::bind(
      notification,
      detail::ReadinessSourceAccess::make_ref<ReadinessSourceKind::kNotification>(
          detail::make_source_id(12), 101));
  detail::ReadinessSourceAccess::bind(
      doorbell,
      detail::ReadinessSourceAccess::make_ref<ReadinessSourceKind::kDoorbell>(
          detail::make_source_id(13), 101));
  detail::ReadinessSourceAccess::bind(
      spin,
      detail::ReadinessSourceAccess::make_ref<ReadinessSourceKind::kSpin>(
          detail::make_source_id(14), 101));

  const OnReadable on_readable{.event = readable};
  const OnNotified on_notified{.event = notification};
  const OnDoorbell on_doorbell{.event = doorbell};
  const OnSpin on_spin{.event = spin};

  // Key assertion: each trigger descriptor's .event field carries the source ID
  // assigned during binding — proving kind-typed refs are not erased on assignment.
  EXPECT_EQ(on_readable.event.id(), detail::make_source_id(11));
  EXPECT_EQ(on_notified.event.id(), detail::make_source_id(12));
  EXPECT_EQ(on_doorbell.event.id(), detail::make_source_id(13));
  EXPECT_EQ(on_spin.event.id(), detail::make_source_id(14));
}

}  // namespace
}  // namespace holoscan
