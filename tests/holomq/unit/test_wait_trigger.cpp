/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#if defined(HOLOMQ_TRANSPORT_IOX2)

#include <holomq/context_builder.hpp>
#include <holomq/publisher_builder.hpp>
#include <holomq/subscriber_builder.hpp>
#include <holomq/wait_set.hpp>
#include <holomq/wait_strategy.hpp>
#include <holomq/wait_trigger.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include <unistd.h>

namespace {

struct WaitTriggerTestPayload {
  std::uint64_t sequence{0};
};

std::atomic<std::uint64_t> g_topic_counter{0};

[[nodiscard]] std::string make_unique_topic() {
  return "holomq_wait_trigger_test_" + std::to_string(static_cast<long>(::getpid())) + "_" +
         std::to_string(g_topic_counter.fetch_add(1U));
}

[[nodiscard]] auto make_context() {
  return holomq::ContextBuilder().create();
}

[[nodiscard]] auto make_publisher(holomq::Context& context, const std::string& topic,
                                  const bool enable_event_wake) {
  return context.publisher_builder(topic)
      .enable_event_wake(enable_event_wake)
      .payload_type<WaitTriggerTestPayload>()
      .create();
}

[[nodiscard]] auto make_subscriber(holomq::Context& context, const std::string& topic,
                                   const bool enable_event_wake) {
  return context.subscriber_builder(topic)
      .enable_event_wake(enable_event_wake)
      .payload_type<WaitTriggerTestPayload>()
      .create();
}

[[nodiscard]] bool wait_for_subscribers(holomq::Publisher<WaitTriggerTestPayload>& publisher,
                                        const std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (!publisher.update_connections().has_value()) {
      return false;
    }
    if (publisher.has_subscribers()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return publisher.update_connections().has_value() && publisher.has_subscribers();
}

[[nodiscard]] bool publish_one(holomq::Publisher<WaitTriggerTestPayload>& publisher,
                               const std::uint64_t sequence) {
  auto message = publisher.loan();
  if (!message.has_value()) {
    return false;
  }
  message->payload().sequence = sequence;
  const auto sent = publisher.send(std::move(message).value());
  return sent.has_value() && sent.value() > 0;
}

[[nodiscard]] holomq::WaitStrategy kernel_wait(const std::chrono::milliseconds timeout) {
  return holomq::WaitStrategy{
      .mode = holomq::WaitMode::KernelWait,
      .spin_duration = std::chrono::nanoseconds::zero(),
      .timeout = timeout,
  };
}

#define ASSERT_EXPECTED(var, expr)       \
  auto var##_result = (expr);            \
  ASSERT_TRUE(var##_result.has_value()); \
  auto var = std::move(var##_result).value()

}  // namespace

#define REQUIRE_CONTEXT(var) ASSERT_EXPECTED(var, make_context())

// A trigger is kernel attached but is not an event-wake subscriber, so it must not satisfy the
// KernelWait capability gate. Admitting it would produce a kernel wait that no publication can end.
// Topology (event wake disabled on both endpoints):
//   publisher --X no event lane X--> subscriber, plus one trigger
// Key assertion: KernelWait is still rejected with kInvalidArgument.
// REQ: CHAN-066
TEST(WaitTriggerTest, TriggerAloneDoesNotAdmitKernelWait) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(publisher, make_publisher(context, topic, false));
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, false));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  auto waitset_or = context.create_waitset();
  ASSERT_TRUE(waitset_or.has_value());
  const auto attachment = waitset_or->attach(subscriber);
  ASSERT_TRUE(attachment.has_value());
  auto trigger_attachment = waitset_or->attach_trigger();
  ASSERT_TRUE(trigger_attachment.has_value());

  const auto waited = waitset_or->wait(kernel_wait(std::chrono::milliseconds(200)));
  ASSERT_FALSE(waited.has_value());
  EXPECT_EQ(waited.error().code, holomq::ErrorCode::kInvalidArgument);
}

// A trigger occupies an attachment slot but is not a subscriber. attachment_count is documented as
// the number of attached subscribers and is used by callers to size readiness storage.
// Key assertion: attaching a trigger does not change attachment_count.
TEST(WaitTriggerTest, TriggerIsNotCountedAsAnAttachment) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(publisher, make_publisher(context, topic, true));
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, true));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  auto waitset_or = context.create_waitset();
  ASSERT_TRUE(waitset_or.has_value());
  EXPECT_EQ(waitset_or->attachment_count(), 0U);

  const auto attachment = waitset_or->attach(subscriber);
  ASSERT_TRUE(attachment.has_value());
  EXPECT_EQ(waitset_or->attachment_count(), 1U);

  auto trigger_attachment = waitset_or->attach_trigger();
  ASSERT_TRUE(trigger_attachment.has_value());
  EXPECT_EQ(waitset_or->attachment_count(), 1U);
}

// Firing before the wait begins must not be lost. This is the highest-value cancellation case
// because it needs no thread and no interleaving: the descriptor is already readable when the
// kernel wait is entered.
// Topology (event wake enabled, no publication at all):
//   publisher --[nothing]--> subscriber, plus one fired trigger
// Key assertion: the wait returns promptly, reports no ready subscriber, and the latch is set.
TEST(WaitTriggerTest, PreFiredTriggerWakesKernelWaitWithEmptyReadyHandles) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(publisher, make_publisher(context, topic, true));
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, true));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  auto waitset_or = context.create_waitset();
  ASSERT_TRUE(waitset_or.has_value());
  const auto attachment = waitset_or->attach(subscriber);
  ASSERT_TRUE(attachment.has_value());
  auto trigger_attachment = waitset_or->attach_trigger();
  ASSERT_TRUE(trigger_attachment.has_value());

  ASSERT_TRUE(trigger_attachment->trigger()->fire().has_value());
  EXPECT_TRUE(trigger_attachment->trigger()->is_set());

  const auto started = std::chrono::steady_clock::now();
  const auto waited = waitset_or->wait(kernel_wait(std::chrono::seconds(10)));
  const auto elapsed = std::chrono::steady_clock::now() - started;

  ASSERT_TRUE(waited.has_value());
  // The trigger was fired before the wait began, so this must return without blocking at all.
  // The bound is what fails on a delayed wake; the returned value is correct either way.
  EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 500);
  EXPECT_TRUE(waitset_or->ready_handles().empty());
  EXPECT_TRUE(trigger_attachment->trigger()->is_set());
}

// The latch, not the descriptor, carries cancellation state. The descriptor is drained on every
// wake so a level-triggered kernel attachment cannot report readiness forever.
// Key assertion: after the wake is consumed the latch is still set, and a later wait no longer
// wakes from the trigger. This is why callers must check is_set before every block.
TEST(WaitTriggerTest, LatchRemainsSetAfterTheWakeIsDrained) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(publisher, make_publisher(context, topic, true));
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, true));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  auto waitset_or = context.create_waitset();
  ASSERT_TRUE(waitset_or.has_value());
  const auto attachment = waitset_or->attach(subscriber);
  ASSERT_TRUE(attachment.has_value());
  auto trigger_attachment = waitset_or->attach_trigger();
  ASSERT_TRUE(trigger_attachment.has_value());

  ASSERT_TRUE(trigger_attachment->trigger()->fire().has_value());
  const auto first = waitset_or->wait(kernel_wait(std::chrono::seconds(10)));
  ASSERT_TRUE(first.has_value());
  // The trigger contributes no ready handle, so the wake has to be reported on the result axis
  // instead. Without this a handle-dispatch loop would see success with an empty span and spin.
  EXPECT_EQ(first.value(), holomq::WaitResult::cancelled);
  EXPECT_TRUE(waitset_or->ready_handles().empty());

  EXPECT_TRUE(trigger_attachment->trigger()->is_set());

  const auto second = waitset_or->wait(kernel_wait(std::chrono::milliseconds(200)));
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second.value(), holomq::WaitResult::timeout);
  EXPECT_TRUE(trigger_attachment->trigger()->is_set());
}

// A latch is monotonic, so a caller that must wait again after a cancellation has to replace the
// trigger rather than reset it. This is the primitive the Channel layer's wait-context re-arm is
// built on: attaching a replacement and letting move-assignment run the old detach callback must
// remove the stale, still-readable descriptor from the wait set.
// Key assertion: after the swap the wait blocks again and reports timeout, and the replacement
// still cancels. Without the detach the level-triggered stale descriptor would wake every wait.
// REQ: CHAN-142
TEST(WaitTriggerTest, ReattachedTriggerRestoresABlockingWait) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(publisher, make_publisher(context, topic, true));
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, true));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  auto waitset_or = context.create_waitset();
  ASSERT_TRUE(waitset_or.has_value());
  const auto attachment = waitset_or->attach(subscriber);
  ASSERT_TRUE(attachment.has_value());

  auto first_or = waitset_or->attach_trigger();
  ASSERT_TRUE(first_or.has_value());
  holomq::WaitTriggerAttachment trigger_attachment = std::move(first_or).value();

  ASSERT_TRUE(trigger_attachment.trigger()->fire().has_value());
  const auto cancelled = waitset_or->wait(kernel_wait(std::chrono::seconds(10)));
  ASSERT_TRUE(cancelled.has_value());
  EXPECT_EQ(cancelled.value(), holomq::WaitResult::cancelled);
  EXPECT_TRUE(trigger_attachment.trigger()->is_set());

  // Re-arm the old trigger's descriptor before replacing it. Without this the wait above has
  // already drained the eventfd, so the timeout below would hold whether or not the old
  // attachment was detached -- the assertion would have no teeth. Firing again leaves the stale
  // descriptor readable, so if move-assignment fails to detach it, the wait returns immediately
  // instead of timing out.
  ASSERT_TRUE(trigger_attachment.trigger()->fire().has_value());

  // Replace the latched trigger. Move-assignment detaches the old one, which is the only thing
  // that stops its still-readable descriptor from waking every later wait.
  auto replacement_or = waitset_or->attach_trigger();
  ASSERT_TRUE(replacement_or.has_value());
  trigger_attachment = std::move(replacement_or).value();
  EXPECT_FALSE(trigger_attachment.trigger()->is_set());

  // The wait must block again rather than return immediately on the old latch.
  const auto blocked = waitset_or->wait(kernel_wait(std::chrono::milliseconds(200)));
  ASSERT_TRUE(blocked.has_value());
  EXPECT_EQ(blocked.value(), holomq::WaitResult::timeout);

  // The replacement is a working cancellation lane, not an inert one.
  ASSERT_TRUE(trigger_attachment.trigger()->fire().has_value());
  const auto cancelled_again = waitset_or->wait(kernel_wait(std::chrono::seconds(10)));
  ASSERT_TRUE(cancelled_again.has_value());
  EXPECT_EQ(cancelled_again.value(), holomq::WaitResult::cancelled);
  EXPECT_TRUE(trigger_attachment.trigger()->is_set());
}

// fire is idempotent so a stop_callback that runs more than once cannot overflow the counter.
// Key assertion: repeated fires all succeed and the latch stays set.
TEST(WaitTriggerTest, RepeatedFireIsIdempotent) {
  REQUIRE_CONTEXT(context);
  auto waitset_or = context.create_waitset();
  ASSERT_TRUE(waitset_or.has_value());
  auto trigger_attachment = waitset_or->attach_trigger();
  ASSERT_TRUE(trigger_attachment.has_value());

  const auto trigger = trigger_attachment->trigger();
  EXPECT_FALSE(trigger->is_set());
  for (int attempt = 0; attempt < 5; ++attempt) {
    EXPECT_TRUE(trigger->fire().has_value());
    EXPECT_TRUE(trigger->is_set());
  }
}

// fire is the only operation safe to call on a live wait set from another thread.
// Key assertion: a stop requested while the waiter is blocked ends the wait well before its
// deadline. Both interleavings satisfy this, which is what keeps it stable under load: a fire that
// lands before the block leaves the descriptor readable, and one that lands during it wakes epoll.
TEST(WaitTriggerTest, FireFromAnotherThreadWakesABlockedWait) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(publisher, make_publisher(context, topic, true));
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, true));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  auto waitset_or = context.create_waitset();
  ASSERT_TRUE(waitset_or.has_value());
  const auto attachment = waitset_or->attach(subscriber);
  ASSERT_TRUE(attachment.has_value());
  auto trigger_attachment = waitset_or->attach_trigger();
  ASSERT_TRUE(trigger_attachment.has_value());

  const auto trigger = trigger_attachment->trigger();
  std::thread firing_thread([trigger]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    static_cast<void>(trigger->fire());
  });

  const auto started = std::chrono::steady_clock::now();
  const auto waited = waitset_or->wait(kernel_wait(std::chrono::seconds(30)));
  const auto elapsed = std::chrono::steady_clock::now() - started;
  firing_thread.join();

  ASSERT_TRUE(waited.has_value());
  // The fire lands ~50 ms in, so this is a ten-fold margin that still fails on any real delayed
  // wake. A seconds-granularity bound accepted multi-second stalls and proved nothing.
  EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 500);
  EXPECT_TRUE(trigger->is_set());
}

// An attached trigger must not disturb ordinary readiness reporting.
// Topology (event wake enabled):
//   publisher --[sequence 7]--> subscriber, plus one unfired trigger
// Key assertion: the subscriber handle is reported and the trigger contributes no handle.
TEST(WaitTriggerTest, DataWakeStillReportsOnlyTheSubscriberHandle) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(publisher, make_publisher(context, topic, true));
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, true));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  auto waitset_or = context.create_waitset();
  ASSERT_TRUE(waitset_or.has_value());
  const auto attachment = waitset_or->attach(subscriber);
  ASSERT_TRUE(attachment.has_value());
  auto trigger_attachment = waitset_or->attach_trigger();
  ASSERT_TRUE(trigger_attachment.has_value());

  ASSERT_TRUE(publish_one(publisher, 7U));

  const auto waited = waitset_or->wait(kernel_wait(std::chrono::seconds(10)));
  ASSERT_TRUE(waited.has_value());
  ASSERT_EQ(waited.value(), holomq::WaitResult::ready);

  const auto handles = waitset_or->ready_handles();
  ASSERT_EQ(handles.size(), 1U);
  EXPECT_EQ(handles.front().id(), attachment->handle().id());
  EXPECT_FALSE(trigger_attachment->trigger()->is_set());

  const auto sample = subscriber.try_take();
  ASSERT_TRUE(sample.has_value());
  EXPECT_EQ(sample->payload().sequence, 7U);
}

// Data wins over cancellation. A wake that carried both must report readiness, not cancellation,
// or a shutdown racing an in-flight publication would silently drop the sample.
// Topology (event wake enabled):
//   publisher --[sequence 11]--> subscriber, plus one trigger fired before the wait
// Key assertion: WaitResult::ready with the subscriber handle present, even though the latch is
// set; the caller still learns about the cancellation from is_set.
TEST(WaitTriggerTest, DataAndCancellationInOneWakeReportReady) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(publisher, make_publisher(context, topic, true));
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, true));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  auto waitset_or = context.create_waitset();
  ASSERT_TRUE(waitset_or.has_value());
  const auto attachment = waitset_or->attach(subscriber);
  ASSERT_TRUE(attachment.has_value());
  auto trigger_attachment = waitset_or->attach_trigger();
  ASSERT_TRUE(trigger_attachment.has_value());

  ASSERT_TRUE(publish_one(publisher, 11U));
  ASSERT_TRUE(trigger_attachment->trigger()->fire().has_value());

  const auto waited = waitset_or->wait(kernel_wait(std::chrono::seconds(10)));
  ASSERT_TRUE(waited.has_value());
  EXPECT_EQ(waited.value(), holomq::WaitResult::ready);

  const auto handles = waitset_or->ready_handles();
  ASSERT_EQ(handles.size(), 1U);
  EXPECT_EQ(handles.front().id(), attachment->handle().id());

  // The cancellation is not lost, it is just answered on the other axis.
  EXPECT_TRUE(trigger_attachment->trigger()->is_set());

  const auto sample = subscriber.try_take();
  ASSERT_TRUE(sample.has_value());
  EXPECT_EQ(sample->payload().sequence, 11U);
}

// A trigger is reference counted and may outlive its wait set, so a late stop request from an
// already-detached shutdown path cannot dereference freed backend state.
// Key assertion: firing a retained trigger after the wait set is gone is safe and still latches.
TEST(WaitTriggerTest, TriggerSurvivesItsWaitSet) {
  REQUIRE_CONTEXT(context);
  std::shared_ptr<holomq::WaitTrigger> trigger;
  {
    auto waitset_or = context.create_waitset();
    ASSERT_TRUE(waitset_or.has_value());
    auto trigger_attachment = waitset_or->attach_trigger();
    ASSERT_TRUE(trigger_attachment.has_value());
    trigger = trigger_attachment->trigger();
  }
  ASSERT_NE(trigger, nullptr);
  EXPECT_TRUE(trigger->fire().has_value());
  EXPECT_TRUE(trigger->is_set());
}

#endif  // defined(HOLOMQ_TRANSPORT_IOX2)
