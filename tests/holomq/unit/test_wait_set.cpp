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

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

struct WaitSetTestPayload {
  std::uint64_t sequence{0};
};

std::atomic<std::uint64_t> g_topic_counter{0};

[[nodiscard]] std::string make_unique_topic() {
  return "holomq_wait_set_test_" + std::to_string(static_cast<long>(::getpid())) + "_" +
         std::to_string(g_topic_counter.fetch_add(1U));
}

[[nodiscard]] auto make_context() {
  return holomq::ContextBuilder().create();
}

[[nodiscard]] bool wait_for_subscribers(holomq::Publisher<WaitSetTestPayload>& publisher,
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

[[nodiscard]] bool publish_one(holomq::Publisher<WaitSetTestPayload>& publisher,
                               const std::uint64_t sequence) {
  auto message = publisher.loan();
  if (!message.has_value()) {
    return false;
  }
  message->payload().sequence = sequence;
  const auto sent = publisher.send(std::move(message).value());
  return sent.has_value() && sent.value() > 0;
}

[[nodiscard]] auto make_publisher(holomq::Context& context, const std::string& topic,
                                  const bool enable_event_wake) {
  return context.publisher_builder(topic)
      .enable_event_wake(enable_event_wake)
      .payload_type<WaitSetTestPayload>()
      .create();
}

[[nodiscard]] auto make_subscriber(holomq::Context& context, const std::string& topic,
                                   const bool enable_event_wake) {
  return context.subscriber_builder(topic)
      .enable_event_wake(enable_event_wake)
      .payload_type<WaitSetTestPayload>()
      .create();
}

#define ASSERT_EXPECTED(var, expr)       \
  auto var##_result = (expr);            \
  ASSERT_TRUE(var##_result.has_value()); \
  auto var = std::move(var##_result).value()

#define EXPECT_WAIT_RESULT(expr, expected) \
  do {                                     \
    const auto waited = (expr);            \
    ASSERT_TRUE(waited.has_value());       \
    EXPECT_EQ(waited.value(), (expected)); \
  } while (false)

}  // namespace

#define REQUIRE_CONTEXT(var) ASSERT_EXPECTED(var, make_context())

TEST(WaitSetTest, AttachDetachAndIntrospection) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(publisher, make_publisher(context, topic, true));
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, true));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  auto waitset_or = context.create_waitset();
  ASSERT_TRUE(waitset_or.has_value());
  auto waitset = std::move(waitset_or).value();
  EXPECT_GT(waitset.capacity(), 0U);
  EXPECT_EQ(waitset.attachment_count(), 0U);

  {
    auto attachment_or = waitset.attach(subscriber);
    ASSERT_TRUE(attachment_or.has_value());
    const holomq::WaitSetHandle handle = attachment_or->handle();
    EXPECT_NE(handle.id(), 0U);
    EXPECT_EQ(waitset.attachment_count(), 1U);
  }
  EXPECT_EQ(waitset.attachment_count(), 0U);
}

TEST(WaitSetTest, WaitWithoutAttachmentsReturnsInvalidArgument) {
  REQUIRE_CONTEXT(context);
  ASSERT_EXPECTED(waitset, context.create_waitset());

  holomq::WaitStrategy strategy{};
  strategy.mode = holomq::WaitMode::Return;
  const auto waited = waitset.wait(strategy);

  ASSERT_FALSE(waited.has_value());
  EXPECT_EQ(waited.error().code, holomq::ErrorCode::kInvalidArgument);
}

TEST(WaitSetTest, AlreadyAttachedAcrossWaitSets) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(publisher, make_publisher(context, topic, true));
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, true));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  auto waitset_a_or = context.create_waitset();
  auto waitset_b_or = context.create_waitset();
  ASSERT_TRUE(waitset_a_or.has_value());
  ASSERT_TRUE(waitset_b_or.has_value());

  {
    auto attachment_a = waitset_a_or->attach(subscriber);
    ASSERT_TRUE(attachment_a.has_value());

    const auto attachment_b = waitset_b_or->attach(subscriber);
    ASSERT_FALSE(attachment_b.has_value());
    EXPECT_EQ(attachment_b.error().code, holomq::ErrorCode::kInvalidArgument);
  }

  const auto attachment_b_retry = waitset_b_or->attach(subscriber);
  ASSERT_TRUE(attachment_b_retry.has_value());
}

// REQ: CHAN-066
TEST(WaitSetTest, SingleTopicWakeAndTake) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(publisher, make_publisher(context, topic, true));
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, true));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  auto waitset_or = context.create_waitset();
  ASSERT_TRUE(waitset_or.has_value());
  auto attachment = waitset_or->attach(subscriber);
  ASSERT_TRUE(attachment.has_value());

  ASSERT_TRUE(publish_one(publisher, 7U));

  holomq::WaitStrategy block{};
  block.mode = holomq::WaitMode::KernelWait;
  EXPECT_WAIT_RESULT(waitset_or->wait_for_ready(block), holomq::WaitResult::ready);

  const auto handles = waitset_or->ready_handles();
  ASSERT_EQ(handles.size(), 1U);
  EXPECT_EQ(handles.front().id(), attachment->handle().id());

  const auto sample = subscriber.try_take();
  ASSERT_TRUE(sample.has_value());
  EXPECT_EQ(sample->payload().sequence, 7U);
}

// Two publications post two notifications. The wait-set callback consumes one notification for
// the first sample, preserving the second notification so a later wait cannot stall on an already
// queued sample.
// Topology (event wake on both endpoints, default queue depth):
//   publisher --[sequence 1, then sequence 2]--> subscriber
// Key assertion: after the first wait consumes one notification and one sample is taken, a second
// bounded kernel wait still reports readiness for the sample that is already queued.
//
// Regression bound: draining both notifications returned the correct `ready` only after its full
// 500 ms timeout (measured 2026-08-13, against 4-6 ms for every other case in the suite), and an
// unbounded KernelWait never reached the post-wait collect at all and hung outright. The 250 ms
// bound below is the only assertion that separates the fix from that stall; the returned value is
// correct either way.
TEST(WaitSetTest, KernelWaitReportsReadinessForASecondQueuedSample) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(publisher, make_publisher(context, topic, true));
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, true));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  auto waitset_or = context.create_waitset();
  ASSERT_TRUE(waitset_or.has_value());
  auto attachment = waitset_or->attach(subscriber);
  ASSERT_TRUE(attachment.has_value());

  ASSERT_TRUE(publish_one(publisher, 1U));
  ASSERT_TRUE(publish_one(publisher, 2U));

  const holomq::WaitStrategy block{
      .mode = holomq::WaitMode::KernelWait,
      .spin_duration = std::chrono::nanoseconds::zero(),
      .timeout = std::chrono::milliseconds(500),
  };

  EXPECT_WAIT_RESULT(waitset_or->wait_for_ready(block), holomq::WaitResult::ready);
  {
    const auto first = subscriber.try_take();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->payload().sequence, 1U);
  }

  // The queued sample must be observed without paying the kernel timeout. wait_kernel collects
  // readiness after the blocking call returns, so the result is correct either way; the elapsed
  // time is what distinguishes the preserved notification from a full-timeout stall.
  const auto started = std::chrono::steady_clock::now();
  EXPECT_WAIT_RESULT(waitset_or->wait_for_ready(block), holomq::WaitResult::ready);
  const auto elapsed = std::chrono::steady_clock::now() - started;
  EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 250);

  {
    const auto second = subscriber.try_take();
    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(second->payload().sequence, 2U);
  }
}

// A public WaitSet must remain level-safe even if an earlier wait consumed the only notification
// without taking its sample.
// Topology (one queued sample, event wake enabled):
//   publisher --[sequence 1 + wake]--> subscriber queue
//                                          |-- first wait consumes wake only
//                                          `-- second wait observes sequence 1
// Key setup: the first wait blocks before publication, drains the resulting notification, and
// deliberately leaves the sample queued.
// Key assertion: the public pre-block readiness check makes the second wait return immediately;
// the Runtime-only unchecked entry point is not exposed to this caller.
TEST(WaitSetTest, PublicKernelWaitPrechecksAQueuedSampleWithoutANotification) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(publisher, make_publisher(context, topic, true));
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, true));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  auto waitset_or = context.create_waitset();
  ASSERT_TRUE(waitset_or.has_value());
  auto attachment = waitset_or->attach(subscriber);
  ASSERT_TRUE(attachment.has_value());

  const holomq::WaitStrategy block{
      .mode = holomq::WaitMode::KernelWait,
      .spin_duration = std::chrono::nanoseconds::zero(),
      .timeout = std::chrono::milliseconds(500),
  };
  bool first_wait_succeeded = false;
  bool first_wait_ready = false;
  std::thread first_waiter([&] {
    const auto result = waitset_or->wait_for_ready(block);
    first_wait_succeeded = result.has_value();
    first_wait_ready = result.has_value() && *result == holomq::WaitResult::ready;
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  const bool published = publish_one(publisher, 1U);
  first_waiter.join();

  ASSERT_TRUE(published);
  ASSERT_TRUE(first_wait_succeeded);
  ASSERT_TRUE(first_wait_ready);

  const auto started = std::chrono::steady_clock::now();
  EXPECT_WAIT_RESULT(waitset_or->wait_for_ready(block), holomq::WaitResult::ready);
  const auto elapsed = std::chrono::steady_clock::now() - started;
  EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 250);

  const auto sample = subscriber.try_take();
  ASSERT_TRUE(sample.has_value());
  EXPECT_EQ(sample->payload().sequence, 1U);
}

TEST(WaitSetTest, SubscriberFirstEventWakePublishesAndWakes) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscribers = 1U;
  service_options.max_subscriber_queue_depth = 4U;
  service_options.enable_event_wake = true;
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.queue_depth = 4U;

  auto subscriber_or = context.subscriber_builder(topic)
                           .service_options(service_options)
                           .subscriber_options(subscriber_options)
                           .payload_type<WaitSetTestPayload>()
                           .create();
  ASSERT_TRUE(subscriber_or.has_value());
  auto subscriber = std::move(subscriber_or).value();
  auto publisher_or = context.publisher_builder(topic)
                          .service_options(service_options)
                          .payload_type<WaitSetTestPayload>()
                          .create();
  ASSERT_TRUE(publisher_or.has_value());
  auto publisher = std::move(publisher_or).value();
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  ASSERT_EXPECTED(waitset, context.create_waitset());
  auto attachment = waitset.attach(subscriber);
  ASSERT_TRUE(attachment.has_value());

  ASSERT_TRUE(publish_one(publisher, 17U));

  holomq::WaitStrategy strategy{};
  strategy.mode = holomq::WaitMode::KernelWait;
  strategy.timeout = std::chrono::seconds(2);
  EXPECT_WAIT_RESULT(waitset.wait_for_ready(strategy), holomq::WaitResult::ready);

  const auto handles = waitset.ready_handles();
  ASSERT_EQ(handles.size(), 1U);
  EXPECT_EQ(handles.front().id(), attachment->handle().id());

  const auto sample = subscriber.try_take();
  ASSERT_TRUE(sample.has_value());
  EXPECT_EQ(sample->payload().sequence, 17U);
}

TEST(WaitSetTest, MultiTopicReadyHandles) {
  REQUIRE_CONTEXT(context);
  const std::string topic_a = make_unique_topic();
  const std::string topic_b = make_unique_topic();

  ASSERT_EXPECTED(pub_a, make_publisher(context, topic_a, true));
  ASSERT_EXPECTED(pub_b, make_publisher(context, topic_b, true));
  ASSERT_EXPECTED(sub_a, make_subscriber(context, topic_a, true));
  ASSERT_EXPECTED(sub_b, make_subscriber(context, topic_b, true));
  ASSERT_TRUE(wait_for_subscribers(pub_a, std::chrono::seconds(2)));
  ASSERT_TRUE(wait_for_subscribers(pub_b, std::chrono::seconds(2)));

  auto waitset_or = context.create_waitset();
  ASSERT_TRUE(waitset_or.has_value());
  const auto attach_a = waitset_or->attach(sub_a);
  const auto attach_b = waitset_or->attach(sub_b);
  ASSERT_TRUE(attach_a.has_value());
  ASSERT_TRUE(attach_b.has_value());

  ASSERT_TRUE(publish_one(pub_b, 99U));

  holomq::WaitStrategy block{};
  block.mode = holomq::WaitMode::KernelWait;
  EXPECT_WAIT_RESULT(waitset_or->wait_for_ready(block), holomq::WaitResult::ready);

  const auto handles = waitset_or->ready_handles();
  ASSERT_EQ(handles.size(), 1U);
  EXPECT_EQ(handles.front().id(), attach_b->handle().id());
}

TEST(WaitSetTest, CoReadyTopics) {
  REQUIRE_CONTEXT(context);
  const std::string topic_a = make_unique_topic();
  const std::string topic_b = make_unique_topic();

  ASSERT_EXPECTED(pub_a, make_publisher(context, topic_a, true));
  ASSERT_EXPECTED(pub_b, make_publisher(context, topic_b, true));
  ASSERT_EXPECTED(sub_a, make_subscriber(context, topic_a, true));
  ASSERT_EXPECTED(sub_b, make_subscriber(context, topic_b, true));
  ASSERT_TRUE(wait_for_subscribers(pub_a, std::chrono::seconds(2)));
  ASSERT_TRUE(wait_for_subscribers(pub_b, std::chrono::seconds(2)));

  auto waitset_or = context.create_waitset();
  ASSERT_TRUE(waitset_or.has_value());
  const auto attach_a = waitset_or->attach(sub_a);
  const auto attach_b = waitset_or->attach(sub_b);
  ASSERT_TRUE(attach_a.has_value());
  ASSERT_TRUE(attach_b.has_value());

  ASSERT_TRUE(publish_one(pub_a, 1U));
  ASSERT_TRUE(publish_one(pub_b, 2U));

  holomq::WaitStrategy spin{};
  spin.mode = holomq::WaitMode::Return;
  spin.spin_duration = std::chrono::milliseconds(50);
  EXPECT_WAIT_RESULT(waitset_or->wait_for_ready(spin), holomq::WaitResult::ready);
  EXPECT_EQ(waitset_or->ready_handles().size(), 2U);
}

TEST(WaitSetTest, WaitModesSmoke) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(publisher, make_publisher(context, topic, true));
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, true));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  for (const auto mode :
       {holomq::WaitMode::Return, holomq::WaitMode::TimedWait, holomq::WaitMode::KernelWait}) {
    ASSERT_TRUE(publish_one(publisher, static_cast<std::uint64_t>(mode)));

    holomq::WaitStrategy strategy{};
    strategy.mode = mode;
    strategy.spin_duration = std::chrono::milliseconds(20);
    strategy.timeout = std::optional<std::chrono::nanoseconds>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::milliseconds(50)));

    const auto waited = subscriber.wait_and_take(strategy);
    ASSERT_TRUE(waited.has_value()) << "mode=" << static_cast<int>(mode);
    EXPECT_EQ(waited->payload().sequence, static_cast<std::uint64_t>(mode));
  }
}

TEST(WaitSetTest, PollWithoutWakeEnabled) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(publisher, make_publisher(context, topic, false));
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, false));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));
  ASSERT_TRUE(publish_one(publisher, 11U));

  holomq::WaitStrategy spin{};
  spin.mode = holomq::WaitMode::Return;
  spin.spin_duration = std::chrono::milliseconds(200);
  const auto sample = subscriber.wait_and_take(spin);
  ASSERT_TRUE(sample.has_value());
  EXPECT_EQ(sample->payload().sequence, 11U);
}

TEST(WaitSetTest, KernelWaitRequiresEventWakeEnabled) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(publisher, make_publisher(context, topic, false));
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, false));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  holomq::WaitStrategy block{};
  block.mode = holomq::WaitMode::KernelWait;
  const auto blocked = subscriber.wait_and_take(block);
  ASSERT_FALSE(blocked.has_value());
}

// REQ: CHAN-066
TEST(WaitSetTest, WaitSetKernelWaitRequiresEventWakeEnabled) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, false));

  auto waitset_or = context.create_waitset();
  ASSERT_TRUE(waitset_or.has_value());
  const auto attachment = waitset_or->attach(subscriber);
  ASSERT_TRUE(attachment.has_value());

  holomq::WaitStrategy block{};
  block.mode = holomq::WaitMode::KernelWait;
  const auto waited = waitset_or->wait(block);
  ASSERT_FALSE(waited.has_value());
  EXPECT_EQ(waited.error().code, holomq::ErrorCode::kInvalidArgument);
}

TEST(WaitSetTest, TimedWaitWorksWithoutEventWake) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(publisher, make_publisher(context, topic, false));
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, false));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));
  ASSERT_TRUE(publish_one(publisher, 17U));

  holomq::WaitStrategy timed{};
  timed.mode = holomq::WaitMode::TimedWait;
  timed.spin_duration = std::chrono::milliseconds(5);
  timed.timeout = std::chrono::milliseconds(50);
  const auto sample = subscriber.wait_and_take(timed);
  ASSERT_TRUE(sample.has_value());
  EXPECT_EQ(sample->payload().sequence, 17U);
}

// REQ: CHAN-066
TEST(WaitSetTest, PollModeReceivesEvenWhenEventWakeIsEnabled) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(publisher, make_publisher(context, topic, true));
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, true));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  ASSERT_TRUE(publish_one(publisher, 55U));

  holomq::WaitStrategy spin{};
  spin.mode = holomq::WaitMode::Return;
  spin.spin_duration = std::chrono::milliseconds(200);
  const auto sample = subscriber.wait_and_take(spin);
  ASSERT_TRUE(sample.has_value());
  EXPECT_EQ(sample->payload().sequence, 55U);
}

TEST(WaitSetTest, LosslessSlowConsumerWake) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(publisher, make_publisher(context, topic, true));
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, true));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  constexpr std::uint64_t kMessages = 8U;
  std::atomic<bool> consumer_done{false};
  std::atomic<bool> consumer_failed{false};
  std::thread consumer([&]() {
    holomq::WaitStrategy block{};
    block.mode = holomq::WaitMode::KernelWait;
    for (std::uint64_t expected = 0; expected < kMessages; ++expected) {
      const auto sample = subscriber.wait_and_take(block);
      if (!sample.has_value() || sample->payload().sequence != expected) {
        consumer_failed.store(true);
        return;
      }
    }
    consumer_done.store(true);
  });

  for (std::uint64_t sequence = 0; sequence < kMessages; ++sequence) {
    while (!publish_one(publisher, sequence)) {
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  }

  consumer.join();
  EXPECT_FALSE(consumer_failed.load());
  EXPECT_TRUE(consumer_done.load());
}

TEST(WaitSetTest, MovePreservesAttachments) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(publisher, make_publisher(context, topic, true));
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, true));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  auto waitset_or = context.create_waitset();
  ASSERT_TRUE(waitset_or.has_value());
  auto attachment = waitset_or->attach(subscriber);
  ASSERT_TRUE(attachment.has_value());

  holomq::WaitSet moved = std::move(waitset_or).value();

  ASSERT_TRUE(publish_one(publisher, 42U));

  holomq::WaitStrategy block{};
  block.mode = holomq::WaitMode::KernelWait;
  EXPECT_WAIT_RESULT(moved.wait_for_ready(block), holomq::WaitResult::ready);

  const auto handles = moved.ready_handles();
  ASSERT_EQ(handles.size(), 1U);
  EXPECT_EQ(handles.front().id(), attachment->handle().id());

  const auto sample = subscriber.try_take();
  ASSERT_TRUE(sample.has_value());
  EXPECT_EQ(sample->payload().sequence, 42U);
}

TEST(WaitSetTest, SpinTimesOutWithoutSample) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, false));

  auto waitset_or = context.create_waitset();
  ASSERT_TRUE(waitset_or.has_value());
  const auto attachment = waitset_or->attach(subscriber);
  ASSERT_TRUE(attachment.has_value());

  holomq::WaitStrategy spin{};
  spin.mode = holomq::WaitMode::Return;
  spin.spin_duration = std::chrono::milliseconds(5);
  EXPECT_WAIT_RESULT(waitset_or->wait(spin), holomq::WaitResult::timeout);
  EXPECT_TRUE(waitset_or->ready_handles().empty());
}

TEST(WaitSetTest, SubscriberWaitAndTakeTimesOut) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, false));

  holomq::WaitStrategy spin{};
  spin.mode = holomq::WaitMode::Return;
  spin.spin_duration = std::chrono::milliseconds(5);
  EXPECT_FALSE(subscriber.wait_and_take(spin).has_value());
}

TEST(WaitSetTest, MoveAssignSubscriberAfterDetach) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, true));
  ASSERT_EXPECTED(other, make_subscriber(context, topic, true));

  auto waitset_or = context.create_waitset();
  ASSERT_TRUE(waitset_or.has_value());
  {
    const auto attachment = waitset_or->attach(subscriber);
    ASSERT_TRUE(attachment.has_value());
  }

  subscriber = std::move(other);
  SUCCEED();
}

TEST(WaitSetTest, MoveAssignSubscriberWhileAttached) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(publisher, make_publisher(context, topic, true));
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, true));
  ASSERT_EXPECTED(other, make_subscriber(context, topic, true));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  auto waitset_or = context.create_waitset();
  ASSERT_TRUE(waitset_or.has_value());
  const auto attachment = waitset_or->attach(subscriber);
  ASSERT_TRUE(attachment.has_value());

  subscriber = std::move(other);

  ASSERT_TRUE(publish_one(publisher, 11U));

  holomq::WaitStrategy block{};
  block.mode = holomq::WaitMode::KernelWait;
  EXPECT_WAIT_RESULT(waitset_or->wait_for_ready(block), holomq::WaitResult::ready);

  const auto taken = subscriber.try_take();
  ASSERT_TRUE(taken.has_value());
  EXPECT_EQ(taken->payload().sequence, 11U);
}

TEST(WaitSetTest, MoveConstructSubscriberWhileAttached) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(publisher, make_publisher(context, topic, true));
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, true));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  auto waitset_or = context.create_waitset();
  ASSERT_TRUE(waitset_or.has_value());
  const auto attachment = waitset_or->attach(subscriber);
  ASSERT_TRUE(attachment.has_value());

  holomq::Subscriber<WaitSetTestPayload> moved = std::move(subscriber);

  ASSERT_TRUE(publish_one(publisher, 13U));

  holomq::WaitStrategy block{};
  block.mode = holomq::WaitMode::KernelWait;
  EXPECT_WAIT_RESULT(waitset_or->wait_for_ready(block), holomq::WaitResult::ready);

  const auto taken = moved.try_take();
  ASSERT_TRUE(taken.has_value());
  EXPECT_EQ(taken->payload().sequence, 13U);
}

#endif  // defined(HOLOMQ_TRANSPORT_IOX2)
