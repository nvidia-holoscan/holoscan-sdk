/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/delivery_attribution.hpp>
#include <holomq/local_buffer_descriptor.hpp>
#include <holomq/pubsub_capabilities.hpp>
#include <holomq/pubsub_options.hpp>

#if defined(HOLOMQ_TRANSPORT_IOX2)
#include <holomq/context_builder.hpp>
#include <holomq/publisher_builder.hpp>
#include <holomq/subscriber_builder.hpp>
#include <holomq/wait_set.hpp>
#include <holomq/wait_strategy.hpp>
#endif

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

#if defined(HOLOMQ_TRANSPORT_IOX2)
#include <unistd.h>
#endif

namespace {

struct DescriptorPayload {
  holomq::LocalBufferDescriptor descriptor{};
  std::uint64_t sample_sequence{};
};

#if defined(HOLOMQ_TRANSPORT_IOX2)

std::atomic<std::uint64_t> g_topic_counter{0};

[[nodiscard]] std::string make_unique_topic() {
  return "holomq_local_buffer_descriptor_test_" + std::to_string(static_cast<long>(::getpid())) +
         "_" + std::to_string(g_topic_counter.fetch_add(1U));
}

[[nodiscard]] auto make_context() {
  return holomq::ContextBuilder().create();
}

[[nodiscard]] bool wait_for_subscribers(holomq::Publisher<DescriptorPayload>& publisher,
                                        const std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (publisher.has_subscribers()) {
      return true;
    }
    static_cast<void>(publisher.update_connections());
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return publisher.has_subscribers();
}

struct TestQueueOptions {
  holomq::PubSubServiceOptions service{};
  holomq::PublisherPubSubOptions publisher{};
  holomq::SubscriberPubSubOptions subscriber{};
};

[[nodiscard]] TestQueueOptions drop_oldest_depth_one_options() {
  TestQueueOptions options{};
  options.service.queue_policy = holomq::QueuePolicy::kDropOldest;
  options.service.max_subscriber_queue_depth = 1U;
  options.service.max_subscriber_borrowed_samples = 4U;
  options.subscriber.queue_depth = 1U;
  options.subscriber.max_borrowed_samples = 4U;
  return options;
}

[[nodiscard]] TestQueueOptions drop_oldest_options(const std::uint32_t queue_depth = 3U) {
  TestQueueOptions options{};
  options.service.queue_policy = holomq::QueuePolicy::kDropOldest;
  options.service.max_subscriber_queue_depth = queue_depth;
  options.subscriber.queue_depth = queue_depth;
  return options;
}

[[nodiscard]] holomq::LocalBufferDescriptor make_descriptor(const std::uint64_t sequence) {
  return holomq::LocalBufferDescriptor{
      .abi_version = holomq::kLocalBufferDescriptorAbiVersion,
      .flags = static_cast<std::uint32_t>(sequence & 0xffU),
      .store_id = 0x1000U + sequence,
      .record_index = sequence,
      .generation = 0x2000U + sequence,
      .token = 0x3000U + sequence,
  };
}

[[nodiscard]] int publish_descriptor(holomq::Publisher<DescriptorPayload>& publisher,
                                     const std::uint64_t sequence) {
  auto outbound = publisher.loan();
  if (!outbound.has_value()) {
    return -1;
  }
  outbound->payload().descriptor = make_descriptor(sequence);
  outbound->payload().sample_sequence = sequence;
  auto sent = publisher.send(std::move(outbound).value());
  return sent.has_value() ? sent.value() : -1;
}

template <typename ReleaseCallback>
[[nodiscard]] int publish_descriptor(holomq::Publisher<DescriptorPayload>& publisher,
                                     const std::uint64_t sequence, ReleaseCallback&& on_released) {
  auto outbound = publisher.loan();
  if (!outbound.has_value()) {
    return -1;
  }
  outbound->payload().descriptor = make_descriptor(sequence);
  outbound->payload().sample_sequence = sequence;
  auto sent =
      publisher.send(std::move(outbound).value(), std::forward<ReleaseCallback>(on_released));
  return sent.has_value() ? sent.value() : -1;
}

void drain_samples(holomq::Subscriber<DescriptorPayload>& subscriber) {
  while (subscriber.try_take().has_value()) {
  }
}

template <typename Predicate>
[[nodiscard]] bool wait_until(Predicate&& predicate, const std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return predicate();
}

[[nodiscard]] bool publish_until_recipient_count(holomq::Publisher<DescriptorPayload>& publisher,
                                                 const std::uint64_t sequence,
                                                 const int expected_recipients,
                                                 const std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (publish_descriptor(publisher, sequence) >= expected_recipients) {
      return true;
    }
    static_cast<void>(publisher.update_connections());
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return false;
}

#define ASSERT_OPTIONAL(var, expr)    \
  auto var##_opt = (expr);            \
  ASSERT_TRUE(var##_opt.has_value()); \
  auto var = std::move(var##_opt).value()

#endif  // defined(HOLOMQ_TRANSPORT_IOX2)

}  // namespace

TEST(LocalBufferDescriptorTest, LayoutSupportsTypedPayloadTransport) {
  static_assert(std::is_standard_layout_v<holomq::LocalBufferDescriptor>);
  static_assert(std::is_trivially_copyable_v<holomq::LocalBufferDescriptor>);
  static_assert(std::is_standard_layout_v<DescriptorPayload>);
  static_assert(std::is_trivially_copyable_v<DescriptorPayload>);

  EXPECT_EQ(sizeof(holomq::LocalBufferDescriptor), 40U);
  EXPECT_LE(sizeof(DescriptorPayload), 64U);
}

TEST(LocalBufferDescriptorTest, DefaultDescriptorIsEmpty) {
  constexpr holomq::LocalBufferDescriptor descriptor{};

  static_assert(descriptor.has_supported_abi());
  static_assert(!static_cast<bool>(descriptor));
  EXPECT_EQ(descriptor.abi_version, holomq::kLocalBufferDescriptorAbiVersion);
  EXPECT_EQ(descriptor.flags, 0U);
  EXPECT_EQ(descriptor.store_id, 0U);
  EXPECT_EQ(descriptor.record_index, 0U);
  EXPECT_EQ(descriptor.generation, 0U);
  EXPECT_EQ(descriptor.token, 0U);
}

TEST(LocalBufferDescriptorTest, DescriptorCarriesOpaqueRuntimeIdentity) {
  constexpr holomq::LocalBufferDescriptor descriptor{
      .abi_version = holomq::kLocalBufferDescriptorAbiVersion,
      .flags = 0x10U,
      .store_id = 0x1234U,
      .record_index = 7U,
      .generation = 99U,
      .token = 0xabcdU,
  };

  static_assert(descriptor.has_supported_abi());
  static_assert(static_cast<bool>(descriptor));
  EXPECT_EQ(descriptor.flags, 0x10U);
  EXPECT_EQ(descriptor.store_id, 0x1234U);
  EXPECT_EQ(descriptor.record_index, 7U);
  EXPECT_EQ(descriptor.generation, 99U);
  EXPECT_EQ(descriptor.token, 0xabcdU);
}

TEST(LocalBufferDescriptorTest, UnsupportedAbiIsRejectedByBooleanCheck) {
  constexpr holomq::LocalBufferDescriptor descriptor{
      .abi_version = holomq::kLocalBufferDescriptorAbiVersion + 1U,
      .flags = 0U,
      .store_id = 1U,
      .record_index = 0U,
      .generation = 1U,
      .token = 1U,
  };

  static_assert(!descriptor.has_supported_abi());
  static_assert(!static_cast<bool>(descriptor));
}

TEST(LocalBufferDescriptorTest, CapabilitiesAdvertiseLocalDescriptorTransportOnly) {
  const holomq::PubSubCapabilities capabilities = holomq::default_pubsub_capabilities();

  EXPECT_TRUE(capabilities.local_descriptor_payload_transfer);
  EXPECT_FALSE(capabilities.local_descriptor_resolution);
}

#if defined(HOLOMQ_TRANSPORT_IOX2)

#define REQUIRE_CONTEXT(var) ASSERT_OPTIONAL(var, make_context())

TEST(LocalBufferDescriptorTest, DescriptorPayloadRoundTripsThroughTypedPubSub) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();

  ASSERT_OPTIONAL(publisher,
                  context.publisher_builder(topic).payload_type<DescriptorPayload>().create());
  ASSERT_OPTIONAL(subscriber,
                  context.subscriber_builder(topic).payload_type<DescriptorPayload>().create());
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  auto outbound = publisher.loan();
  ASSERT_TRUE(outbound.has_value());
  outbound->payload().descriptor = holomq::LocalBufferDescriptor{
      .abi_version = holomq::kLocalBufferDescriptorAbiVersion,
      .flags = 0x2U,
      .store_id = 0x101U,
      .record_index = 5U,
      .generation = 8U,
      .token = 0x303U,
  };
  outbound->payload().sample_sequence = 42U;
  const auto sent = publisher.send(std::move(outbound).value());
  ASSERT_TRUE(sent.has_value());
  ASSERT_GT(sent.value(), 0);

  const auto inbound = subscriber.try_take();
  ASSERT_TRUE(inbound.has_value());

  const DescriptorPayload& payload = inbound->payload();
  EXPECT_EQ(payload.sample_sequence, 42U);
  EXPECT_TRUE(static_cast<bool>(payload.descriptor));
  EXPECT_EQ(payload.descriptor.flags, 0x2U);
  EXPECT_EQ(payload.descriptor.store_id, 0x101U);
  EXPECT_EQ(payload.descriptor.record_index, 5U);
  EXPECT_EQ(payload.descriptor.generation, 8U);
  EXPECT_EQ(payload.descriptor.token, 0x303U);
}

TEST(LocalBufferDescriptorTest, DescriptorPayloadDropOldestDepthOneKeepsNewestUnreadSample) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  const TestQueueOptions options = drop_oldest_depth_one_options();

  ASSERT_OPTIONAL(publisher,
                  context.publisher_builder(topic)
                      .service_options(options.service)
                      .publisher_options(options.publisher)
                      .payload_type<DescriptorPayload>()
                      .create());
  ASSERT_OPTIONAL(subscriber,
                  context.subscriber_builder(topic)
                      .service_options(options.service)
                      .subscriber_options(options.subscriber)
                      .payload_type<DescriptorPayload>()
                      .create());
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  ASSERT_GT(publish_descriptor(publisher, 1U), 0);
  ASSERT_GT(publish_descriptor(publisher, 2U), 0);
  ASSERT_GT(publish_descriptor(publisher, 3U), 0);

  const auto has_latest = subscriber.has_samples();
  ASSERT_TRUE(has_latest.has_value());
  EXPECT_TRUE(has_latest.value());
  const auto latest = subscriber.try_take();
  ASSERT_TRUE(latest.has_value());
  EXPECT_EQ(latest->payload().sample_sequence, 3U);
  EXPECT_EQ(latest->payload().descriptor.store_id, 0x1003U);
  EXPECT_EQ(latest->payload().descriptor.record_index, 3U);
  EXPECT_EQ(latest->payload().descriptor.generation, 0x2003U);
  EXPECT_EQ(latest->payload().descriptor.token, 0x3003U);
  const auto has_more = subscriber.has_samples();
  ASSERT_TRUE(has_more.has_value());
  EXPECT_FALSE(has_more.value());
  EXPECT_FALSE(subscriber.try_take().has_value());
}

TEST(LocalBufferDescriptorTest, DescriptorPayloadWakesWaitSet) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();

  ASSERT_OPTIONAL(publisher,
                  context.publisher_builder(topic)
                      .enable_event_wake(true)
                      .payload_type<DescriptorPayload>()
                      .create());
  ASSERT_OPTIONAL(subscriber,
                  context.subscriber_builder(topic)
                      .enable_event_wake(true)
                      .payload_type<DescriptorPayload>()
                      .create());
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  auto waitset_or = context.create_waitset();
  ASSERT_TRUE(waitset_or.has_value());
  auto attachment = waitset_or->attach(subscriber);
  ASSERT_TRUE(attachment.has_value());

  auto outbound = publisher.loan();
  ASSERT_TRUE(outbound.has_value());
  outbound->payload().descriptor = holomq::LocalBufferDescriptor{
      .abi_version = holomq::kLocalBufferDescriptorAbiVersion,
      .flags = 0x4U,
      .store_id = 0x202U,
      .record_index = 9U,
      .generation = 11U,
      .token = 0x404U,
  };
  outbound->payload().sample_sequence = 100U;
  const auto sent = publisher.send(std::move(outbound).value());
  ASSERT_TRUE(sent.has_value());
  ASSERT_GT(sent.value(), 0);

  holomq::WaitStrategy block{};
  block.mode = holomq::WaitMode::KernelWait;
  const auto waited = waitset_or->wait_for_ready(block);
  ASSERT_TRUE(waited.has_value());
  EXPECT_EQ(waited.value(), holomq::WaitResult::ready);

  const auto handles = waitset_or->ready_handles();
  ASSERT_EQ(handles.size(), 1U);
  EXPECT_EQ(handles.front().id(), attachment->handle().id());

  const auto inbound = subscriber.try_take();
  ASSERT_TRUE(inbound.has_value());
  EXPECT_EQ(inbound->payload().sample_sequence, 100U);
  EXPECT_EQ(inbound->payload().descriptor.store_id, 0x202U);
  EXPECT_EQ(inbound->payload().descriptor.record_index, 9U);
  EXPECT_EQ(inbound->payload().descriptor.generation, 11U);
  EXPECT_EQ(inbound->payload().descriptor.token, 0x404U);
}

TEST(LocalBufferDescriptorTest, DescriptorPayloadDropOldestDepthOneWakesWaitSet) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  const TestQueueOptions options = drop_oldest_depth_one_options();

  ASSERT_OPTIONAL(publisher,
                  context.publisher_builder(topic)
                      .enable_event_wake(true)
                      .service_options(options.service)
                      .publisher_options(options.publisher)
                      .payload_type<DescriptorPayload>()
                      .create());
  ASSERT_OPTIONAL(subscriber,
                  context.subscriber_builder(topic)
                      .enable_event_wake(true)
                      .service_options(options.service)
                      .subscriber_options(options.subscriber)
                      .payload_type<DescriptorPayload>()
                      .create());
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  auto waitset_or = context.create_waitset();
  ASSERT_TRUE(waitset_or.has_value());
  auto attachment = waitset_or->attach(subscriber);
  ASSERT_TRUE(attachment.has_value());

  ASSERT_GT(publish_descriptor(publisher, 1U), 0);
  ASSERT_GT(publish_descriptor(publisher, 2U), 0);
  ASSERT_GT(publish_descriptor(publisher, 3U), 0);

  holomq::WaitStrategy block{};
  block.mode = holomq::WaitMode::KernelWait;
  const auto waited = waitset_or->wait_for_ready(block);
  ASSERT_TRUE(waited.has_value());
  EXPECT_EQ(waited.value(), holomq::WaitResult::ready);

  const auto handles = waitset_or->ready_handles();
  ASSERT_EQ(handles.size(), 1U);
  EXPECT_EQ(handles.front().id(), attachment->handle().id());

  const auto latest = subscriber.try_take();
  ASSERT_TRUE(latest.has_value());
  EXPECT_EQ(latest->payload().sample_sequence, 3U);
  EXPECT_EQ(latest->payload().descriptor.store_id, 0x1003U);
  EXPECT_FALSE(subscriber.try_take().has_value());
}

TEST(LocalBufferDescriptorTest, DescriptorPayloadDropOldestDepthOneReleasesSupersededSample) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  const TestQueueOptions options = drop_oldest_depth_one_options();
  std::atomic<std::uint64_t> released_mask{0U};

  ASSERT_OPTIONAL(publisher,
                  context.publisher_builder(topic)
                      .service_options(options.service)
                      .publisher_options(options.publisher)
                      .payload_type<DescriptorPayload>()
                      .create());
  ASSERT_OPTIONAL(subscriber,
                  context.subscriber_builder(topic)
                      .service_options(options.service)
                      .subscriber_options(options.subscriber)
                      .payload_type<DescriptorPayload>()
                      .create());
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  ASSERT_GT(publish_descriptor(
                publisher,
                1U,
                [&released_mask]() { released_mask.fetch_or(0x1U, std::memory_order_relaxed); }),
            0);
  ASSERT_GT(publish_descriptor(
                publisher,
                2U,
                [&released_mask]() { released_mask.fetch_or(0x2U, std::memory_order_relaxed); }),
            0);

  EXPECT_TRUE(wait_until(
      [&released_mask]() { return (released_mask.load(std::memory_order_relaxed) & 0x1U) != 0U; },
      std::chrono::seconds(2)));
  EXPECT_EQ(released_mask.load(std::memory_order_relaxed) & 0x2U, 0U);

  {
    const auto latest = subscriber.try_take();
    ASSERT_TRUE(latest.has_value());
    EXPECT_EQ(latest->payload().sample_sequence, 2U);
  }

  ASSERT_GT(publish_descriptor(publisher, 3U), 0);
  EXPECT_TRUE(wait_until(
      [&released_mask]() { return (released_mask.load(std::memory_order_relaxed) & 0x2U) != 0U; },
      std::chrono::seconds(2)));
  drain_samples(subscriber);
}

TEST(LocalBufferDescriptorTest, DescriptorPayloadDropOldestReportsSlowRecipientReplacement) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  const TestQueueOptions options = drop_oldest_options();
  std::atomic<std::uint64_t> replacement_count{0U};
  holomq::DeliveryRecipientId replaced_receiver{};
  std::uint64_t replaced_sequence = 0U;
  std::uint64_t replaced_token = 0U;
  const void* replaced_payload_ptr = nullptr;
  std::size_t replaced_payload_size = 0U;

  ASSERT_OPTIONAL(
      publisher,
      context.publisher_builder(topic)
          .service_options(options.service)
          .publisher_options(options.publisher)
          .payload_type<DescriptorPayload>()
          .on_sample_replaced([&replacement_count,
                               &replaced_receiver,
                               &replaced_sequence,
                               &replaced_token,
                               &replaced_payload_ptr,
                               &replaced_payload_size](const DescriptorPayload& payload,
                                                       const holomq::SampleReplacedInfo& info) {
            replaced_receiver = info.receiver_id;
            replaced_sequence = payload.sample_sequence;
            replaced_token = payload.descriptor.token;
            replaced_payload_ptr = info.payload_ptr;
            replaced_payload_size = info.payload_size;
            replacement_count.fetch_add(1U, std::memory_order_release);
          })
          .create());
  ASSERT_OPTIONAL(subscriber_fast,
                  context.subscriber_builder(topic)
                      .service_options(options.service)
                      .subscriber_options(options.subscriber)
                      .payload_type<DescriptorPayload>()
                      .create());
  ASSERT_OPTIONAL(subscriber_slow,
                  context.subscriber_builder(topic)
                      .service_options(options.service)
                      .subscriber_options(options.subscriber)
                      .payload_type<DescriptorPayload>()
                      .create());

  ASSERT_TRUE(publish_until_recipient_count(publisher, 0U, 2, std::chrono::seconds(2)));
  drain_samples(subscriber_fast);
  drain_samples(subscriber_slow);

  const holomq::DeliveryRecipientId fast_id = subscriber_fast.recipient_id();
  const holomq::DeliveryRecipientId slow_id = subscriber_slow.recipient_id();
  ASSERT_NE(fast_id, slow_id);

  for (std::uint64_t sequence = 1U; sequence <= 3U; ++sequence) {
    ASSERT_EQ(publish_descriptor(publisher, sequence), 2);
    const auto fast_sample = subscriber_fast.try_take();
    ASSERT_TRUE(fast_sample.has_value());
    EXPECT_EQ(fast_sample->payload().sample_sequence, sequence);
  }

  ASSERT_EQ(publish_descriptor(publisher, 4U), 2);
  EXPECT_TRUE(wait_until(
      [&replacement_count]() { return replacement_count.load(std::memory_order_acquire) == 1U; },
      std::chrono::seconds(2)));
  EXPECT_EQ(replaced_receiver, slow_id);
  EXPECT_NE(replaced_receiver, fast_id);
  EXPECT_EQ(replaced_sequence, 1U);
  EXPECT_EQ(replaced_token, 0x3001U);
  EXPECT_NE(replaced_payload_ptr, nullptr);
  EXPECT_EQ(replaced_payload_size, sizeof(DescriptorPayload));

  for (std::uint64_t expected_sequence = 2U; expected_sequence <= 4U; ++expected_sequence) {
    const auto slow_sample = subscriber_slow.try_take();
    ASSERT_TRUE(slow_sample.has_value());
    EXPECT_EQ(slow_sample->payload().sample_sequence, expected_sequence);
  }
  EXPECT_FALSE(subscriber_slow.try_take().has_value());

  const auto fast_sample = subscriber_fast.try_take();
  ASSERT_TRUE(fast_sample.has_value());
  EXPECT_EQ(fast_sample->payload().sample_sequence, 4U);
  EXPECT_FALSE(subscriber_fast.try_take().has_value());
}

TEST(LocalBufferDescriptorTest,
     DescriptorPayloadDropOldestDepthOneContainsReplacementCallbackExceptions) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  const TestQueueOptions options = drop_oldest_depth_one_options();
  std::atomic<std::uint64_t> replacement_count{0U};

  ASSERT_OPTIONAL(publisher,
                  context.publisher_builder(topic)
                      .service_options(options.service)
                      .publisher_options(options.publisher)
                      .payload_type<DescriptorPayload>()
                      .on_sample_replaced([&replacement_count](const DescriptorPayload&,
                                                               const holomq::SampleReplacedInfo&) {
                        replacement_count.fetch_add(1U, std::memory_order_release);
                        throw std::runtime_error("expected replacement callback test exception");
                      })
                      .create());
  ASSERT_OPTIONAL(subscriber,
                  context.subscriber_builder(topic)
                      .service_options(options.service)
                      .subscriber_options(options.subscriber)
                      .payload_type<DescriptorPayload>()
                      .create());
  ASSERT_TRUE(publish_until_recipient_count(publisher, 0U, 1, std::chrono::seconds(2)));
  drain_samples(subscriber);

  ASSERT_EQ(publish_descriptor(publisher, 1U), 1);
  ASSERT_EQ(publish_descriptor(publisher, 2U), 1);
  EXPECT_TRUE(wait_until(
      [&replacement_count]() { return replacement_count.load(std::memory_order_acquire) == 1U; },
      std::chrono::seconds(2)));

  const auto latest = subscriber.try_take();
  ASSERT_TRUE(latest.has_value());
  EXPECT_EQ(latest->payload().sample_sequence, 2U);
  EXPECT_FALSE(subscriber.try_take().has_value());
}

TEST(LocalBufferDescriptorTest, DescriptorPayloadDropOldestDepthOneFanoutTracksEachSubscriber) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  const TestQueueOptions options = drop_oldest_depth_one_options();

  ASSERT_OPTIONAL(publisher,
                  context.publisher_builder(topic)
                      .service_options(options.service)
                      .publisher_options(options.publisher)
                      .payload_type<DescriptorPayload>()
                      .create());
  ASSERT_OPTIONAL(subscriber_a,
                  context.subscriber_builder(topic)
                      .service_options(options.service)
                      .subscriber_options(options.subscriber)
                      .payload_type<DescriptorPayload>()
                      .create());
  ASSERT_OPTIONAL(subscriber_b,
                  context.subscriber_builder(topic)
                      .service_options(options.service)
                      .subscriber_options(options.subscriber)
                      .payload_type<DescriptorPayload>()
                      .create());

  ASSERT_TRUE(publish_until_recipient_count(publisher, 0U, 2, std::chrono::seconds(2)));
  drain_samples(subscriber_a);
  drain_samples(subscriber_b);

  ASSERT_EQ(publish_descriptor(publisher, 1U), 2);
  const auto first_for_a = subscriber_a.try_take();
  ASSERT_TRUE(first_for_a.has_value());
  EXPECT_EQ(first_for_a->payload().sample_sequence, 1U);

  ASSERT_EQ(publish_descriptor(publisher, 2U), 2);
  ASSERT_EQ(publish_descriptor(publisher, 3U), 2);

  const auto latest_for_a = subscriber_a.try_take();
  ASSERT_TRUE(latest_for_a.has_value());
  EXPECT_EQ(latest_for_a->payload().sample_sequence, 3U);
  EXPECT_EQ(latest_for_a->payload().descriptor.token, 0x3003U);

  const auto latest_for_b = subscriber_b.try_take();
  ASSERT_TRUE(latest_for_b.has_value());
  EXPECT_EQ(latest_for_b->payload().sample_sequence, 3U);
  EXPECT_EQ(latest_for_b->payload().descriptor.token, 0x3003U);

  EXPECT_FALSE(subscriber_a.try_take().has_value());
  EXPECT_FALSE(subscriber_b.try_take().has_value());
}

#endif  // defined(HOLOMQ_TRANSPORT_IOX2)
