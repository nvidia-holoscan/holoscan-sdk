/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#if defined(HOLOMQ_TRANSPORT_IOX2)

#include <holomq/context_builder.hpp>
#include <holomq/detail/wait_set_runtime_access.hpp>
#include <holomq/publisher_builder.hpp>
#include <holomq/subscriber_builder.hpp>
#include <holomq/wait_set.hpp>
#include <holomq/wait_strategy.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>

#include <unistd.h>

namespace {

using holomq::detail::WaitSetRuntimeAccess;

struct RuntimeAccessTestPayload {
  std::uint64_t sequence{0};
};

std::atomic<std::uint64_t> g_topic_counter{0};

[[nodiscard]] std::string make_unique_topic() {
  return "holomq_wait_set_runtime_access_test_" + std::to_string(static_cast<long>(::getpid())) +
         "_" + std::to_string(g_topic_counter.fetch_add(1U));
}

[[nodiscard]] auto make_publisher(holomq::Context& context, const std::string& topic,
                                  const bool enable_event_wake) {
  return context.publisher_builder(topic)
      .enable_event_wake(enable_event_wake)
      .payload_type<RuntimeAccessTestPayload>()
      .create();
}

[[nodiscard]] auto make_subscriber(holomq::Context& context, const std::string& topic,
                                   const bool enable_event_wake) {
  return context.subscriber_builder(topic)
      .enable_event_wake(enable_event_wake)
      .payload_type<RuntimeAccessTestPayload>()
      .create();
}

[[nodiscard]] bool wait_for_subscribers(holomq::Publisher<RuntimeAccessTestPayload>& publisher,
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

[[nodiscard]] bool publish_one(holomq::Publisher<RuntimeAccessTestPayload>& publisher,
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

[[nodiscard]] bool reports(const holomq::WaitSet& wait_set, const holomq::WaitSetHandle handle) {
  const auto ready = wait_set.ready_handles();
  return std::find(ready.begin(), ready.end(), handle) != ready.end();
}

#define ASSERT_EXPECTED(var, expr)       \
  auto var##_result = (expr);            \
  ASSERT_TRUE(var##_result.has_value()); \
  auto var = std::move(var##_result).value()

}  // namespace

// A masked subscriber's queued sample must not make a wait report it, or a caller that cannot take
// the sample would wake on it forever. Its notification still ends the wait that drains it.
// Topology (event wake enabled):
//   publisher --[one sample]--> subscriber (masked)
// Key assertion: while masked, the sample's notification ends one wait without the subscriber's
// handle and the next wait times out; once unmasked, or once the mask is removed, the queued
// sample is reported at once.
// REQ: RUNTIME-090
TEST(WaitSetRuntimeAccessTest, MaskedSubscriberIsNotReportedReady) {
  ASSERT_EXPECTED(context, holomq::ContextBuilder().create());
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(publisher, make_publisher(context, topic, true));
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, true));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  ASSERT_EXPECTED(wait_set, context.create_waitset());
  ASSERT_EXPECTED(attachment, wait_set.attach(subscriber));
  const auto mask = std::make_shared<std::atomic<bool>>(true);
  ASSERT_TRUE(
      WaitSetRuntimeAccess::set_readiness_mask(wait_set, attachment.handle(), mask).has_value());
  ASSERT_TRUE(publish_one(publisher, 1U));

  const auto notified = wait_set.wait(kernel_wait(std::chrono::seconds(10)));
  ASSERT_TRUE(notified.has_value());
  EXPECT_EQ(notified.value(), holomq::WaitResult::ready);
  EXPECT_FALSE(reports(wait_set, attachment.handle()));

  const auto masked = wait_set.wait(kernel_wait(std::chrono::milliseconds(200)));
  ASSERT_TRUE(masked.has_value());
  EXPECT_EQ(masked.value(), holomq::WaitResult::timeout);

  mask->store(false);
  const auto unmasked = wait_set.wait(kernel_wait(std::chrono::milliseconds(200)));
  ASSERT_TRUE(unmasked.has_value());
  EXPECT_EQ(unmasked.value(), holomq::WaitResult::ready);
  EXPECT_TRUE(reports(wait_set, attachment.handle()));

  mask->store(true);
  ASSERT_TRUE(
      WaitSetRuntimeAccess::set_readiness_mask(wait_set, attachment.handle(), nullptr).has_value());
  const auto removed = wait_set.wait(kernel_wait(std::chrono::milliseconds(200)));
  ASSERT_TRUE(removed.has_value());
  EXPECT_EQ(removed.value(), holomq::WaitResult::ready);
  EXPECT_TRUE(reports(wait_set, attachment.handle()));
}

// The Runtime pump masks a subscriber whose sample it cannot store, then needs a wake from the
// thread that frees storage. The wake must end a blocked wait without a subscriber handle and,
// unlike a cancellation trigger, leave later waits blocking.
// Key assertion: a wake signaled from another thread ends a long wait promptly as a spurious
// ready, and the following wait times out.
// REQ: RUNTIME-090
TEST(WaitSetRuntimeAccessTest, WakeEndsABlockedWaitAndRearms) {
  ASSERT_EXPECTED(context, holomq::ContextBuilder().create());
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(publisher, make_publisher(context, topic, true));
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, true));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  ASSERT_EXPECTED(wait_set, context.create_waitset());
  ASSERT_EXPECTED(attachment, wait_set.attach(subscriber));
  ASSERT_TRUE(WaitSetRuntimeAccess::attach_wake(wait_set).has_value());

  std::thread waker{[&wait_set] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_TRUE(WaitSetRuntimeAccess::signal_wake(wait_set).has_value());
  }};
  const auto started = std::chrono::steady_clock::now();
  const auto woken = wait_set.wait(kernel_wait(std::chrono::seconds(10)));
  const auto elapsed = std::chrono::steady_clock::now() - started;
  waker.join();
  ASSERT_TRUE(woken.has_value());
  EXPECT_EQ(woken.value(), holomq::WaitResult::ready);
  EXPECT_TRUE(wait_set.ready_handles().empty());
  // The bound is what fails on a lost wake; the returned value is correct either way.
  EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 5000);

  const auto rearmed = wait_set.wait(kernel_wait(std::chrono::milliseconds(200)));
  ASSERT_TRUE(rearmed.has_value());
  EXPECT_EQ(rearmed.value(), holomq::WaitResult::timeout);

  // Posts made before any wait coalesce into one wake.
  ASSERT_TRUE(WaitSetRuntimeAccess::signal_wake(wait_set).has_value());
  ASSERT_TRUE(WaitSetRuntimeAccess::signal_wake(wait_set).has_value());
  const auto coalesced = wait_set.wait(kernel_wait(std::chrono::seconds(10)));
  ASSERT_TRUE(coalesced.has_value());
  EXPECT_EQ(coalesced.value(), holomq::WaitResult::ready);
  const auto drained = wait_set.wait(kernel_wait(std::chrono::milliseconds(200)));
  ASSERT_TRUE(drained.has_value());
  EXPECT_EQ(drained.value(), holomq::WaitResult::timeout);
}

// A masked subscriber whose storage was freed is woken and then unmasked by its owner, which then
// finds every sample queued behind the one it took.
// Key assertion: mask, wake, and unmask compose: the wake arrives without the subscriber's handle,
// and unmasking reports the still-queued sample without another publication.
// REQ: RUNTIME-090
TEST(WaitSetRuntimeAccessTest, MaskedSubscriberIsReportedAgainAfterWakeAndUnmask) {
  ASSERT_EXPECTED(context, holomq::ContextBuilder().create());
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(publisher, make_publisher(context, topic, true));
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, true));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  ASSERT_EXPECTED(wait_set, context.create_waitset());
  ASSERT_EXPECTED(attachment, wait_set.attach(subscriber));
  ASSERT_TRUE(WaitSetRuntimeAccess::attach_wake(wait_set).has_value());
  const auto mask = std::make_shared<std::atomic<bool>>(true);
  ASSERT_TRUE(
      WaitSetRuntimeAccess::set_readiness_mask(wait_set, attachment.handle(), mask).has_value());
  ASSERT_TRUE(publish_one(publisher, 1U));
  // Drain the publication's notification while masked. It must end this wait itself, so that the
  // wake below is the only thing that can end the next one.
  const auto drained = wait_set.wait(kernel_wait(std::chrono::seconds(10)));
  ASSERT_TRUE(drained.has_value());
  ASSERT_EQ(drained.value(), holomq::WaitResult::ready);
  EXPECT_FALSE(reports(wait_set, attachment.handle()));

  ASSERT_TRUE(WaitSetRuntimeAccess::signal_wake(wait_set).has_value());
  const auto woken = wait_set.wait(kernel_wait(std::chrono::seconds(10)));
  ASSERT_TRUE(woken.has_value());
  EXPECT_EQ(woken.value(), holomq::WaitResult::ready);
  EXPECT_FALSE(reports(wait_set, attachment.handle()));

  mask->store(false);
  const auto unmasked = wait_set.wait(kernel_wait(std::chrono::milliseconds(200)));
  ASSERT_TRUE(unmasked.has_value());
  EXPECT_EQ(unmasked.value(), holomq::WaitResult::ready);
  EXPECT_TRUE(reports(wait_set, attachment.handle()));
}

// The wake is not a subscriber. It must neither change the subscriber count that callers use to
// size readiness storage nor satisfy the event-wake capability that KernelWait requires.
// Key assertion: attachment_count is unchanged, a second wake and a mask on the wake are rejected,
// and KernelWait over a subscriber without event wake is still rejected.
// REQ: RUNTIME-090
TEST(WaitSetRuntimeAccessTest, WakeIsNotASubscriber) {
  ASSERT_EXPECTED(context, holomq::ContextBuilder().create());
  const std::string topic = make_unique_topic();
  ASSERT_EXPECTED(publisher, make_publisher(context, topic, false));
  ASSERT_EXPECTED(subscriber, make_subscriber(context, topic, false));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  ASSERT_EXPECTED(wait_set, context.create_waitset());
  const auto unattached = WaitSetRuntimeAccess::signal_wake(wait_set);
  ASSERT_FALSE(unattached.has_value());
  EXPECT_EQ(unattached.error().code, holomq::ErrorCode::kInvalidArgument);

  ASSERT_EXPECTED(attachment, wait_set.attach(subscriber));
  ASSERT_TRUE(WaitSetRuntimeAccess::attach_wake(wait_set).has_value());
  EXPECT_EQ(wait_set.attachment_count(), 1U);

  const auto second = WaitSetRuntimeAccess::attach_wake(wait_set);
  ASSERT_FALSE(second.has_value());
  EXPECT_EQ(second.error().code, holomq::ErrorCode::kInvalidArgument);

  const auto unknown =
      WaitSetRuntimeAccess::set_readiness_mask(wait_set,
                                               holomq::WaitSetHandle{attachment.handle().id() + 1U},
                                               std::make_shared<std::atomic<bool>>(true));
  ASSERT_FALSE(unknown.has_value());
  EXPECT_EQ(unknown.error().code, holomq::ErrorCode::kInvalidArgument);

  const auto waited = wait_set.wait(kernel_wait(std::chrono::milliseconds(200)));
  ASSERT_FALSE(waited.has_value());
  EXPECT_EQ(waited.error().code, holomq::ErrorCode::kInvalidArgument);
}

#endif  // defined(HOLOMQ_TRANSPORT_IOX2)
