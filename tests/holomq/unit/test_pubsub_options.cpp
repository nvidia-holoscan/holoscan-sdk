/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/limits.hpp>
#include <holomq/pubsub_options.hpp>

#if defined(HOLOMQ_TRANSPORT_IOX2)
#include <holomq/context_builder.hpp>
#include <holomq/detail/transport/iox2/unable_to_deliver_evidence.hpp>
#include <holomq/publisher_builder.hpp>
#include <holomq/subscriber_builder.hpp>
#endif

#include <gtest/gtest.h>

#include <limits>
#include <optional>

#if defined(HOLOMQ_TRANSPORT_IOX2)
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#endif

#if defined(HOLOMQ_TRANSPORT_IOX2)
#include <unistd.h>
#endif

TEST(PubSubOptionsTest, ServiceDefaultsMatchBlockingPolicy) {
  static_assert(static_cast<std::uint8_t>(holomq::QueuePolicy::kBlocking) == 0U);
  static_assert(static_cast<std::uint8_t>(holomq::QueuePolicy::kDropOldest) == 1U);

  holomq::PubSubServiceOptions options{};
  EXPECT_EQ(options.queue_policy, holomq::QueuePolicy::kBlocking);
  EXPECT_EQ(options.max_subscribers, 0U);
  EXPECT_EQ(options.max_subscriber_queue_depth, holomq::k_subscriber_queue_depth);
  EXPECT_EQ(options.max_subscriber_borrowed_samples, holomq::k_subscriber_max_borrowed_samples);
  EXPECT_EQ(options.publisher_history_depth, holomq::k_publisher_history_depth);
  EXPECT_TRUE(options.enable_event_wake);
  EXPECT_FALSE(options.uses_safe_overflow());
  EXPECT_EQ(options.effective_max_subscriber_queue_depth(), holomq::k_subscriber_queue_depth);
  EXPECT_EQ(options.effective_max_subscriber_borrowed_samples(), holomq::k_subscriber_queue_depth);
}

// REQ: MQ-SLICE-001
TEST(PubSubOptionsTest, ByteSliceCapacityRejectsUnrepresentableFrame) {
  EXPECT_TRUE(holomq::detail::is_valid_byte_slice_contract(true, 64U, 1U));
  EXPECT_FALSE(holomq::detail::is_valid_byte_slice_contract(
      true, std::numeric_limits<std::uint64_t>::max(), 1U));
  EXPECT_FALSE(holomq::detail::is_valid_byte_slice_contract(
      true, static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()), 1U));
}

// REQ: MQ-SLICE-001
TEST(PubSubOptionsTest, ByteSlicePoolGeometryChecksTheAlignedBoundary) {
  holomq::PubSubServiceOptions service{};
  service.max_subscribers = 1U;
  service.max_subscriber_queue_depth = 1U;
  service.max_subscriber_borrowed_samples = 1U;
  service.publisher_history_depth = 1U;
  // Four samples: one queued, one borrowed, one retained in history, and one publisher loan.
  const std::uint64_t limit = std::numeric_limits<std::ptrdiff_t>::max();
  const std::uint64_t capacity =
      ((limit - 4096U) / 4U - 4096U) / holomq::kByteSliceAlignment * holomq::kByteSliceAlignment;
  service.byte_slice_capacity = capacity;
  EXPECT_TRUE(holomq::detail::is_valid_byte_slice_pool_geometry(service, 1U, 1U));
  service.byte_slice_capacity = capacity - 1U;
  EXPECT_TRUE(holomq::detail::is_valid_byte_slice_pool_geometry(service, 1U, 1U));
  service.byte_slice_capacity = capacity + 1U;
  EXPECT_FALSE(holomq::detail::is_valid_byte_slice_pool_geometry(service, 1U, 1U));
}

// REQ: MQ-SLICE-001
TEST(PubSubOptionsTest, ByteSlicePoolGeometryAccountsForEveryWindowComponent) {
  holomq::PubSubServiceOptions service{};
  service.max_subscribers = 1U;
  service.max_subscriber_queue_depth = 1U;
  service.max_subscriber_borrowed_samples = 0U;
  service.publisher_history_depth = 0U;
  const std::uint64_t limit = std::numeric_limits<std::ptrdiff_t>::max();
  service.byte_slice_capacity =
      ((limit - 4096U) / 3U - 4096U) / holomq::kByteSliceAlignment * holomq::kByteSliceAlignment;
  EXPECT_TRUE(holomq::detail::is_valid_byte_slice_pool_geometry(service, 1U, 1U));
  EXPECT_FALSE(holomq::detail::is_valid_byte_slice_pool_geometry(service, 2U, 1U));
  EXPECT_FALSE(holomq::detail::is_valid_byte_slice_pool_geometry(service, 1U, 2U));
  auto modified = service;
  modified.publisher_history_depth = 1U;
  EXPECT_FALSE(holomq::detail::is_valid_byte_slice_pool_geometry(modified, 1U, 1U));
  modified = service;
  modified.max_subscriber_borrowed_samples = 2U;
  EXPECT_FALSE(holomq::detail::is_valid_byte_slice_pool_geometry(modified, 1U, 1U));
  modified = service;
  modified.max_subscriber_queue_depth = 2U;
  EXPECT_FALSE(holomq::detail::is_valid_byte_slice_pool_geometry(modified, 1U, 1U));
}

// REQ: MQ-SLICE-001
TEST(PubSubOptionsTest, ByteSlicePoolGeometryRejectsCountOverflow) {
  holomq::PubSubServiceOptions service{};
  service.byte_slice_capacity = 64U;
  EXPECT_FALSE(holomq::detail::is_valid_byte_slice_pool_geometry(
      service, 1U, std::numeric_limits<std::uint64_t>::max()));
  EXPECT_FALSE(holomq::detail::is_valid_byte_slice_pool_geometry(
      service, std::numeric_limits<std::uint64_t>::max(), 1U));
  service.max_subscriber_queue_depth = std::numeric_limits<std::uint32_t>::max();
  service.max_subscriber_borrowed_samples = std::numeric_limits<std::uint32_t>::max();
  service.publisher_history_depth = std::numeric_limits<std::uint32_t>::max();
  EXPECT_FALSE(holomq::detail::is_valid_byte_slice_pool_geometry(
      service, 1U, std::numeric_limits<std::uint32_t>::max()));
}

TEST(PubSubOptionsTest, PublisherDefaultsAreEndpointLocal) {
  const holomq::PubSubServiceOptions service_options{};
  holomq::PublisherPubSubOptions options{};
  EXPECT_EQ(options.max_loaned_samples, 0U);
  EXPECT_EQ(options.expected_recipient_count, 0U);
  EXPECT_TRUE(options.enable_sample_release_callbacks);
  EXPECT_EQ(options.effective_max_loaned_samples(service_options),
            holomq::k_subscriber_queue_depth);
  EXPECT_EQ(options.effective_sample_release_callback_capacity(service_options), 136U);
  EXPECT_FALSE(options.replacement_attribution_required);
  EXPECT_EQ(options.pressure_policy, holomq::PublisherPressurePolicy::kFollowService);
  EXPECT_EQ(options.bounded_retry_budget_ns, 0U);
  EXPECT_EQ(options.bounded_retry_cancellation, nullptr);
}

TEST(PubSubOptionsTest, SubscriberDefaultsAreEndpointLocal) {
  holomq::SubscriberPubSubOptions options{};
  EXPECT_EQ(options.queue_depth, holomq::k_subscriber_queue_depth);
  EXPECT_EQ(options.max_borrowed_samples, holomq::k_subscriber_max_borrowed_samples);
  EXPECT_EQ(options.effective_queue_depth(holomq::PubSubServiceOptions{}),
            holomq::k_subscriber_queue_depth);
  EXPECT_EQ(options.effective_max_borrowed_samples(), holomq::k_subscriber_queue_depth);
}

TEST(PubSubOptionsTest, PublisherExplicitMaxLoanedOverridesDerivedDefault) {
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscriber_queue_depth = 8U;
  service_options.publisher_history_depth = 16U;
  holomq::PublisherPubSubOptions options{};
  options.max_loaned_samples = 2U;

  EXPECT_EQ(options.effective_max_loaned_samples(service_options), 2U);
}

TEST(PubSubOptionsTest, PublisherReleaseCallbackCapacityCoversCompleteTransportWindow) {
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscribers = 2U;
  service_options.max_subscriber_queue_depth = 8U;
  service_options.max_subscriber_borrowed_samples = 3U;
  service_options.publisher_history_depth = 4U;
  holomq::PublisherPubSubOptions options{};
  options.max_loaned_samples = 5U;
  options.expected_recipient_count = 1U;

  // 2 subscribers * (8 queued + 3 borrowed) + 4 history + 5 publisher loans.
  EXPECT_EQ(options.effective_sample_release_callback_capacity(service_options), 31U);
}

TEST(PubSubOptionsTest, PublisherReleaseCallbackCapacityRejectsSizeOverflow) {
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscribers = std::numeric_limits<std::uint32_t>::max();
  service_options.max_subscriber_queue_depth = std::numeric_limits<std::uint32_t>::max();
  service_options.max_subscriber_borrowed_samples = std::numeric_limits<std::uint32_t>::max();
  service_options.publisher_history_depth = std::numeric_limits<std::uint32_t>::max();
  holomq::PublisherPubSubOptions options{};
  options.max_loaned_samples = std::numeric_limits<std::uint32_t>::max();

  EXPECT_EQ(options.effective_sample_release_callback_capacity(service_options), std::nullopt);
}

TEST(PubSubOptionsTest, DropOldestSubscriberUsesConfiguredQueueDepth) {
  holomq::PubSubServiceOptions service_options{};
  service_options.queue_policy = holomq::QueuePolicy::kDropOldest;
  holomq::SubscriberPubSubOptions options{};
  options.queue_depth = 8U;

  EXPECT_EQ(options.effective_queue_depth(service_options), 8U);
  EXPECT_EQ(options.effective_max_borrowed_samples(), 8U);
}

#if defined(HOLOMQ_TRANSPORT_IOX2)

namespace {

struct PubSubOptionsTestPayload {
  std::uint64_t sequence{0};
};

std::atomic<std::uint64_t> g_topic_counter{0};

[[nodiscard]] std::string make_unique_topic() {
  return "holomq_pubsub_options_test_" + std::to_string(static_cast<long>(::getpid())) + "_" +
         std::to_string(g_topic_counter.fetch_add(1U));
}

[[nodiscard]] auto make_context() {
  return holomq::ContextBuilder().create();
}

[[nodiscard]] bool wait_for_subscribers(holomq::Publisher<PubSubOptionsTestPayload>& publisher,
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

[[nodiscard]] bool wait_for_subscriber_count(holomq::Publisher<PubSubOptionsTestPayload>& publisher,
                                             const std::uint64_t expected_count,
                                             const std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    static_cast<void>(publisher.update_connections());
    if (publisher.number_of_subscribers() == expected_count) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return publisher.number_of_subscribers() == expected_count;
}

[[nodiscard]] int publish_one(holomq::Publisher<PubSubOptionsTestPayload>& publisher,
                              const std::uint64_t sequence) {
  auto message = publisher.loan();
  if (!message.has_value()) {
    return -1;
  }
  message->payload().sequence = sequence;
  auto sent = publisher.send(std::move(message).value());
  return sent.has_value() ? sent.value() : -1;
}

[[nodiscard]] auto wait_for_sample(holomq::Subscriber<PubSubOptionsTestPayload>& subscriber,
                                   const std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto sample = subscriber.try_take();
    if (sample.has_value() || sample.error().code != holomq::ErrorCode::kNotReady) {
      return sample;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return holomq::expected<holomq::InboundMessage<PubSubOptionsTestPayload>, holomq::Error>(
      holomq::make_unexpected(holomq::Error{holomq::ErrorCode::kTimeout}));
}

void expect_no_sample(holomq::Subscriber<PubSubOptionsTestPayload>& subscriber) {
  const auto sample = subscriber.try_take();
  ASSERT_FALSE(sample.has_value());
  EXPECT_EQ(sample.error().code, holomq::ErrorCode::kNotReady);
}

[[nodiscard]] auto make_publisher(holomq::Context& context, const std::string& topic,
                                  const holomq::PubSubServiceOptions& service_options,
                                  const holomq::PublisherPubSubOptions& options) {
  return context.publisher_builder(topic)
      .service_options(service_options)
      .publisher_options(options)
      .payload_type<PubSubOptionsTestPayload>()
      .create();
}

[[nodiscard]] auto make_subscriber(holomq::Context& context, const std::string& topic,
                                   const holomq::PubSubServiceOptions& service_options,
                                   const holomq::SubscriberPubSubOptions& options) {
  return context.subscriber_builder(topic)
      .service_options(service_options)
      .subscriber_options(options)
      .payload_type<PubSubOptionsTestPayload>()
      .create();
}

[[nodiscard]] auto make_publisher(holomq::Context& context, const std::string& topic,
                                  const holomq::PublisherPubSubOptions& options) {
  return make_publisher(context, topic, holomq::PubSubServiceOptions{}, options);
}

[[nodiscard]] auto make_subscriber(holomq::Context& context, const std::string& topic,
                                   const holomq::SubscriberPubSubOptions& options) {
  return make_subscriber(context, topic, holomq::PubSubServiceOptions{}, options);
}

#define ASSERT_OPTIONAL(var, expr)    \
  auto var##_opt = (expr);            \
  ASSERT_TRUE(var##_opt.has_value()); \
  auto var = std::move(var##_opt).value()

}  // namespace

#define REQUIRE_CONTEXT(var) ASSERT_OPTIONAL(var, make_context())

// Exposes the exact refreshed subscriber inventory needed by topology-aware higher layers without
// changing HoloMQ's capacity-only max_subscribers contract.
TEST(PubSubOptionsTest, PublisherReportsRefreshedSubscriberCount) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  // Key setup: create one publisher whose service admits exactly two subscribers.
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscribers = 2U;
  holomq::PublisherPubSubOptions publisher_options{};
  holomq::SubscriberPubSubOptions subscriber_options{};

  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, service_options, publisher_options));
  // Key assertion: each connection refresh exposes the exact current inventory.
  EXPECT_EQ(publisher.number_of_subscribers(), 0U);
  EXPECT_FALSE(publisher.has_subscribers());

  ASSERT_OPTIONAL(first, make_subscriber(context, topic, service_options, subscriber_options));
  ASSERT_TRUE(wait_for_subscriber_count(publisher, 1U, std::chrono::seconds(2)));
  EXPECT_TRUE(publisher.has_subscribers());

  ASSERT_OPTIONAL(second, make_subscriber(context, topic, service_options, subscriber_options));
  ASSERT_TRUE(wait_for_subscriber_count(publisher, 2U, std::chrono::seconds(2)));
  EXPECT_TRUE(publisher.has_subscribers());
}

// An exact topology contract rejects before backend submission until every planned subscriber is
// connected. No already-connected recipient may observe the rejected sample.
TEST(PubSubOptionsTest, PublisherWaitsForExpectedRecipientCountBeforeSend) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  // Key setup: require three recipients, but initially attach only two.
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscribers = 3U;
  holomq::PublisherPubSubOptions publisher_options{};
  publisher_options.max_loaned_samples = 2U;
  publisher_options.expected_recipient_count = 3U;
  holomq::SubscriberPubSubOptions subscriber_options{};

  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, service_options, publisher_options));
  ASSERT_OPTIONAL(first, make_subscriber(context, topic, service_options, subscriber_options));
  ASSERT_OPTIONAL(second, make_subscriber(context, topic, service_options, subscriber_options));
  ASSERT_TRUE(wait_for_subscriber_count(publisher, 2U, std::chrono::seconds(2)));

  auto incomplete_message = publisher.loan();
  ASSERT_TRUE(incomplete_message);
  incomplete_message->payload().sequence = 1U;
  const auto incomplete = publisher.send(std::move(*incomplete_message), []() noexcept {});
  // Key assertion: incomplete fanout rejects before either attached subscriber observes data.
  ASSERT_FALSE(incomplete);
  EXPECT_EQ(incomplete.error().code, holomq::ErrorCode::kIncompleteFanout);
  expect_no_sample(first);
  expect_no_sample(second);

  ASSERT_OPTIONAL(third, make_subscriber(context, topic, service_options, subscriber_options));
  ASSERT_TRUE(wait_for_subscriber_count(publisher, 3U, std::chrono::seconds(2)));
  auto complete_message = publisher.loan();
  ASSERT_TRUE(complete_message);
  complete_message->payload().sequence = 2U;
  const auto complete = publisher.send(std::move(*complete_message), []() noexcept {});
  // Key assertion: the same publisher reaches all three recipients after final attachment.
  ASSERT_TRUE(complete);
  EXPECT_EQ(*complete, 3);
  const auto first_received = first.try_take();
  const auto second_received = second.try_take();
  const auto third_received = third.try_take();
  ASSERT_TRUE(first_received);
  ASSERT_TRUE(second_received);
  ASSERT_TRUE(third_received);
  EXPECT_EQ(first_received->payload().sequence, 2U);
  EXPECT_EQ(second_received->payload().sequence, 2U);
  EXPECT_EQ(third_received->payload().sequence, 2U);
}

// A capacity-one callback publisher must retire every loan and callback on an incomplete-fanout
// rejection. Repetition makes either a transport-loan or callback-registry leak deterministic.
TEST(PubSubOptionsTest, IncompleteFanoutReusesCapacityOneCallbackPublisher) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscribers = 2U;
  holomq::PublisherPubSubOptions publisher_options{};
  publisher_options.max_loaned_samples = 1U;
  publisher_options.expected_recipient_count = 2U;
  holomq::SubscriberPubSubOptions subscriber_options{};

  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, service_options, publisher_options));
  ASSERT_OPTIONAL(subscriber, make_subscriber(context, topic, service_options, subscriber_options));
  ASSERT_TRUE(wait_for_subscriber_count(publisher, 1U, std::chrono::seconds(2)));

  std::uint32_t released = 0U;
  constexpr std::uint32_t kAttempts = 16U;
  for (std::uint32_t sequence = 1U; sequence <= kAttempts; ++sequence) {
    auto message = publisher.loan();
    ASSERT_TRUE(message) << "loan capacity leaked after attempt " << sequence;
    message->payload().sequence = sequence;
    const auto rejected =
        publisher.send(std::move(*message), [&released]() noexcept { ++released; });
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, holomq::ErrorCode::kIncompleteFanout);
    EXPECT_EQ(released, sequence);
    expect_no_sample(subscriber);
  }
}

// An exact one-recipient contract preserves the historical one-subscriber publication behavior.
TEST(PubSubOptionsTest, ExpectedRecipientCountOnePublishesExactlyOnce) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscribers = 1U;
  holomq::PublisherPubSubOptions publisher_options{};
  publisher_options.max_loaned_samples = 1U;
  publisher_options.expected_recipient_count = 1U;
  holomq::SubscriberPubSubOptions subscriber_options{};

  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, service_options, publisher_options));
  ASSERT_OPTIONAL(subscriber, make_subscriber(context, topic, service_options, subscriber_options));
  ASSERT_TRUE(wait_for_subscriber_count(publisher, 1U, std::chrono::seconds(2)));

  ASSERT_EQ(publish_one(publisher, 1U), 1);
  const auto received = subscriber.try_take();
  ASSERT_TRUE(received);
  EXPECT_EQ(received->payload().sequence, 1U);
  expect_no_sample(subscriber);
}

TEST(PubSubOptionsTest, PublisherReconnectGatePreventsDuplicateFanoutDelivery) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscribers = 2U;
  holomq::PublisherPubSubOptions publisher_options{};
  publisher_options.max_loaned_samples = 2U;
  publisher_options.expected_recipient_count = 2U;
  holomq::SubscriberPubSubOptions subscriber_options{};

  // Key setup: establish both arms, deliver sequence 1, then detach the second subscriber.
  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, service_options, publisher_options));
  ASSERT_OPTIONAL(first, make_subscriber(context, topic, service_options, subscriber_options));
  {
    ASSERT_OPTIONAL(second, make_subscriber(context, topic, service_options, subscriber_options));
    ASSERT_TRUE(wait_for_subscriber_count(publisher, 2U, std::chrono::seconds(2)));
    ASSERT_EQ(publish_one(publisher, 1U), 2);
    const auto first_initial = first.try_take();
    const auto second_initial = second.try_take();
    ASSERT_TRUE(first_initial);
    ASSERT_TRUE(second_initial);
    EXPECT_EQ(first_initial->payload().sequence, 1U);
    EXPECT_EQ(second_initial->payload().sequence, 1U);
  }
  ASSERT_TRUE(wait_for_subscriber_count(publisher, 1U, std::chrono::seconds(2)));

  auto disconnected_message = publisher.loan();
  ASSERT_TRUE(disconnected_message);
  disconnected_message->payload().sequence = 2U;
  const auto disconnected = publisher.send(std::move(*disconnected_message));
  ASSERT_FALSE(disconnected);
  EXPECT_EQ(disconnected.error().code, holomq::ErrorCode::kIncompleteFanout);
  expect_no_sample(first);

  ASSERT_OPTIONAL(replacement,
                  make_subscriber(context, topic, service_options, subscriber_options));
  ASSERT_TRUE(wait_for_subscriber_count(publisher, 2U, std::chrono::seconds(2)));
  ASSERT_EQ(publish_one(publisher, 2U), 2);
  const auto first_reconnected = first.try_take();
  const auto replacement_received = replacement.try_take();

  // Key assertion: the pre-submit rejection remains invisible and the retry reaches each arm once.
  ASSERT_TRUE(first_reconnected);
  ASSERT_TRUE(replacement_received);
  EXPECT_EQ(first_reconnected->payload().sequence, 2U);
  EXPECT_EQ(replacement_received->payload().sequence, 2U);
  expect_no_sample(first);
  expect_no_sample(replacement);
}

void expect_explicit_event_wake_capacity(const bool subscriber_first) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  constexpr std::uint32_t kSubscriberCount = 17U;
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscribers = kSubscriberCount;
  const holomq::PublisherPubSubOptions publisher_options{};
  const holomq::SubscriberPubSubOptions subscriber_options{};
  std::vector<holomq::Subscriber<PubSubOptionsTestPayload>> subscribers;
  subscribers.reserve(kSubscriberCount);

  if (subscriber_first) {
    ASSERT_OPTIONAL(subscriber,
                    make_subscriber(context, topic, service_options, subscriber_options));
    subscribers.push_back(std::move(subscriber));
  }

  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, service_options, publisher_options));
  while (subscribers.size() < kSubscriberCount) {
    ASSERT_OPTIONAL(subscriber,
                    make_subscriber(context, topic, service_options, subscriber_options));
    subscribers.push_back(std::move(subscriber));
  }

  EXPECT_EQ(subscribers.size(), kSubscriberCount);
  EXPECT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));
}

TEST(PubSubOptionsTest, ReplacementAttributionRequiresDropOldestCallback) {
  REQUIRE_CONTEXT(context);

  holomq::PubSubServiceOptions drop_oldest_service{};
  drop_oldest_service.queue_policy = holomq::QueuePolicy::kDropOldest;
  holomq::PublisherPubSubOptions missing_callback{};
  missing_callback.replacement_attribution_required = true;

  const auto publisher_without_callback = context.publisher_builder(make_unique_topic())
                                              .service_options(drop_oldest_service)
                                              .publisher_options(missing_callback)
                                              .payload_type<PubSubOptionsTestPayload>()
                                              .create();
  EXPECT_FALSE(publisher_without_callback.has_value());

  const auto publisher_with_blocking_callback =
      context.publisher_builder(make_unique_topic())
          .payload_type<PubSubOptionsTestPayload>()
          .on_sample_replaced(
              [](const PubSubOptionsTestPayload&, const holomq::SampleReplacedInfo&) {})
          .create();
  EXPECT_FALSE(publisher_with_blocking_callback.has_value());
}

TEST(PubSubOptionsTest, TypedBuildersAcceptTheSharedServiceContract) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscribers = 1U;
  service_options.max_subscriber_queue_depth = 2U;
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.queue_depth = 2U;

  const auto publisher = context.publisher_builder(topic)
                             .payload_type<PubSubOptionsTestPayload>()
                             .service_options(service_options)
                             .create();
  ASSERT_TRUE(publisher.has_value());
  const auto subscriber = context.subscriber_builder(topic)
                              .payload_type<PubSubOptionsTestPayload>()
                              .service_options(service_options)
                              .subscriber_options(subscriber_options)
                              .create();
  EXPECT_TRUE(subscriber.has_value());
}

TEST(PubSubOptionsTest, DisabledReleaseCallbacksPreservePlainSend) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  holomq::PublisherPubSubOptions publisher_options{};
  publisher_options.enable_sample_release_callbacks = false;
  holomq::SubscriberPubSubOptions subscriber_options{};

  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, publisher_options));
  ASSERT_OPTIONAL(subscriber, make_subscriber(context, topic, subscriber_options));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  auto message = publisher.loan();
  ASSERT_TRUE(message.has_value());
  message->payload().sequence = 1U;
  bool callback_invoked = false;
  const auto rejected = publisher.send(std::move(message).value(),
                                       [&callback_invoked]() noexcept { callback_invoked = true; });
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().code, holomq::ErrorCode::kNotImplemented);
  EXPECT_FALSE(callback_invoked);

  const auto sent = publisher.send(std::move(message).value());
  ASSERT_TRUE(sent.has_value());
  EXPECT_EQ(sent.value(), 1);
  const auto received = subscriber.try_take();
  ASSERT_TRUE(received.has_value());
  EXPECT_EQ(received->payload().sequence, 1U);
}

TEST(PubSubOptionsTest, BlockingPreservesQueuedSamples) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscriber_queue_depth = 2U;
  holomq::PublisherPubSubOptions publisher_options{};
  publisher_options.max_loaned_samples = 1U;
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.queue_depth = 2U;

  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, service_options, publisher_options));
  ASSERT_OPTIONAL(subscriber, make_subscriber(context, topic, service_options, subscriber_options));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  ASSERT_GT(publish_one(publisher, 1U), 0);
  ASSERT_GT(publish_one(publisher, 2U), 0);

  {
    const auto first = subscriber.try_take();
    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(first->payload().sequence, 1U);
  }

  const auto second = subscriber.try_take();
  ASSERT_TRUE(second.has_value());
  EXPECT_EQ(second->payload().sequence, 2U);

  EXPECT_FALSE(subscriber.try_take().has_value());
}

// REQ: CHAN-101
TEST(PubSubOptionsTest, BlockingPublisherCanRejectFullQueueWithoutDiscarding) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscriber_queue_depth = 1U;
  service_options.max_subscriber_borrowed_samples = 2U;
  holomq::PublisherPubSubOptions publisher_options{};
  publisher_options.max_loaned_samples = 1U;
  publisher_options.pressure_policy = holomq::PublisherPressurePolicy::kRejectWhenFull;
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.queue_depth = 1U;
  // Keep borrowed-sample accounting out of this queue-capacity regression: iceoryx2 may retire
  // the prior receive loan asynchronously after the C++ handle leaves scope.
  subscriber_options.max_borrowed_samples = 2U;

  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, service_options, publisher_options));
  ASSERT_OPTIONAL(subscriber, make_subscriber(context, topic, service_options, subscriber_options));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));
  ASSERT_GT(publish_one(publisher, 1U), 0);

  std::promise<void> entered_send;
  auto entered = entered_send.get_future();
  auto second =
      std::async(std::launch::async, [&publisher, signal = std::move(entered_send)]() mutable {
        auto message = publisher.loan();
        if (!message) {
          return holomq::expected<int, holomq::Error>(holomq::make_unexpected(message.error()));
        }
        message->payload().sequence = 2U;
        signal.set_value();
        return publisher.send(std::move(*message));
      });
  ASSERT_EQ(entered.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  const auto status = second.wait_for(std::chrono::seconds(1));
  if (status != std::future_status::ready) {
    // Unblock the baseline implementation so the red regression never strands a test worker.
    const auto first = subscriber.try_take();
    ASSERT_TRUE(first.has_value());
    ASSERT_EQ(first->payload().sequence, 1U);
    ASSERT_EQ(second.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    static_cast<void>(second.get());
    FAIL() << "kRejectWhenFull waited for recipient capacity";
  }

  const auto rejected = second.get();
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().code, holomq::ErrorCode::kBackpressured);
  const auto first = subscriber.try_take();
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->payload().sequence, 1U);

  ASSERT_GT(publish_one(publisher, 2U), 0);
  const auto retried = subscriber.try_take();
  ASSERT_TRUE(retried.has_value());
  EXPECT_EQ(retried->payload().sequence, 2U);
}

// Moving a reject-mode publisher must preserve the process-lifetime pressure-handler contract
// while replacing the old backend publisher.
// REQ: CHAN-101
TEST(PubSubOptionsTest, RejectWhenFullPublisherMoveAssignmentPreservesHandlerContract) {
  REQUIRE_CONTEXT(context);
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscriber_queue_depth = 1U;
  service_options.max_subscriber_borrowed_samples = 2U;
  holomq::PublisherPubSubOptions publisher_options{};
  publisher_options.pressure_policy = holomq::PublisherPressurePolicy::kRejectWhenFull;
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.queue_depth = 1U;
  subscriber_options.max_borrowed_samples = 2U;

  const std::string original_topic = make_unique_topic();
  const std::string replacement_topic = make_unique_topic();
  ASSERT_OPTIONAL(original,
                  make_publisher(context, original_topic, service_options, publisher_options));
  ASSERT_OPTIONAL(replacement,
                  make_publisher(context, replacement_topic, service_options, publisher_options));
  ASSERT_OPTIONAL(subscriber,
                  make_subscriber(context, replacement_topic, service_options, subscriber_options));
  ASSERT_TRUE(wait_for_subscribers(replacement, std::chrono::seconds(2)));

  original = std::move(replacement);
  ASSERT_EQ(publish_one(original, 7U), 1);
  const auto received = subscriber.try_take();
  ASSERT_TRUE(received.has_value());
  EXPECT_EQ(received->payload().sequence, 7U);
}

// A full fanout recipient can reject after another recipient accepted the same send. The result
// must therefore remain an error with observable partial delivery, independent of backend
// subscriber iteration order.
// REQ: CHAN-101
TEST(PubSubOptionsTest, RejectWhenFullReportsPartialFanoutAsBackpressure) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscribers = 2U;
  service_options.max_subscriber_queue_depth = 1U;
  service_options.max_subscriber_borrowed_samples = 2U;
  holomq::PublisherPubSubOptions publisher_options{};
  publisher_options.max_loaned_samples = 1U;
  publisher_options.pressure_policy = holomq::PublisherPressurePolicy::kRejectWhenFull;
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.queue_depth = 1U;
  subscriber_options.max_borrowed_samples = 2U;

  // The assertion on the first send is the exact connection-count gate for the test.
  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, service_options, publisher_options));
  ASSERT_OPTIONAL(first_subscriber,
                  make_subscriber(context, topic, service_options, subscriber_options));
  ASSERT_OPTIONAL(second_subscriber,
                  make_subscriber(context, topic, service_options, subscriber_options));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));
  ASSERT_EQ(publish_one(publisher, 1U), 2);

  // Empty exactly one recipient. The other remains full, so the next send has one accepted leg and
  // one rejected leg regardless of which subscriber slot the backend visits first.
  const auto first_received = first_subscriber.try_take();
  ASSERT_TRUE(first_received.has_value());
  ASSERT_EQ(first_received->payload().sequence, 1U);

  auto message = publisher.loan();
  ASSERT_TRUE(message.has_value());
  message->payload().sequence = 2U;
  const auto partial = publisher.send(std::move(*message));
  ASSERT_FALSE(partial.has_value());
  EXPECT_EQ(partial.error().code, holomq::ErrorCode::kBackpressured);
  EXPECT_EQ(partial.error().message,
            holomq::detail::transport::unable_to_deliver_recipient_diagnostic(
                second_subscriber.recipient_id()));

  const auto accepted_leg = first_subscriber.try_take();
  ASSERT_TRUE(accepted_leg.has_value());
  EXPECT_EQ(accepted_leg->payload().sequence, 2U);
  const auto preserved_leg = second_subscriber.try_take();
  ASSERT_TRUE(preserved_leg.has_value());
  EXPECT_EQ(preserved_leg->payload().sequence, 1U);
  expect_no_sample(second_subscriber);
}

// REQ: CHAN-101
TEST(PubSubOptionsTest, BoundedRetrySurvivesPublisherReplacement) {
  REQUIRE_CONTEXT(context);
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscriber_queue_depth = 1U;
  service_options.max_subscriber_borrowed_samples = 2U;
  holomq::PublisherPubSubOptions publisher_options{};
  publisher_options.max_loaned_samples = 1U;
  publisher_options.pressure_policy = holomq::PublisherPressurePolicy::kBoundedRetry;
  publisher_options.bounded_retry_budget_ns =
      static_cast<std::uint64_t>(std::chrono::seconds(1).count()) * 1'000'000'000U;
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.queue_depth = 1U;
  subscriber_options.max_borrowed_samples = 2U;

  // Key setup: replace one public Publisher facade with another, fill its only queue slot, then
  // submit the next sample while the subscriber remains unable to accept it. Releasing the old
  // PublisherImpl exercises backend-before-handler destructor ordering without claiming that the
  // source-private PublisherPort itself is move-assigned.
  ASSERT_OPTIONAL(original,
                  make_publisher(context, make_unique_topic(), service_options, publisher_options));
  const std::string topic = make_unique_topic();
  ASSERT_OPTIONAL(replacement, make_publisher(context, topic, service_options, publisher_options));
  ASSERT_OPTIONAL(subscriber, make_subscriber(context, topic, service_options, subscriber_options));
  ASSERT_TRUE(wait_for_subscribers(replacement, std::chrono::seconds(2)));
  original = std::move(replacement);
  ASSERT_EQ(publish_one(original, 1U), 1);

  std::promise<void> entered_send;
  auto entered = entered_send.get_future();
  auto second =
      std::async(std::launch::async, [&original, signal = std::move(entered_send)]() mutable {
        auto message = original.loan();
        if (!message) {
          return holomq::expected<int, holomq::Error>(holomq::make_unexpected(message.error()));
        }
        message->payload().sequence = 2U;
        signal.set_value();
        return original.send(std::move(*message));
      });
  ASSERT_EQ(entered.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  EXPECT_EQ(second.wait_for(std::chrono::milliseconds(10)), std::future_status::timeout);
  const auto first = subscriber.try_take();
  ASSERT_TRUE(first.has_value());
  EXPECT_EQ(first->payload().sequence, 1U);

  // Key assertion: recipient progress inside the configured budget lets the original send finish
  // successfully, proving both retry behavior and stable handler lifetime across replacement.
  ASSERT_EQ(second.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  const auto delivered = second.get();
  ASSERT_TRUE(delivered.has_value());
  EXPECT_EQ(*delivered, 1);
  const auto received = subscriber.try_take();
  ASSERT_TRUE(received.has_value());
  EXPECT_EQ(received->payload().sequence, 2U);
}

// REQ: CHAN-101
TEST(PubSubOptionsTest, BoundedRetryCompletesFanoutWithoutDuplicateDelivery) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscribers = 2U;
  service_options.max_subscriber_queue_depth = 1U;
  service_options.max_subscriber_borrowed_samples = 2U;
  holomq::PublisherPubSubOptions publisher_options{};
  publisher_options.max_loaned_samples = 1U;
  publisher_options.expected_recipient_count = 2U;
  publisher_options.pressure_policy = holomq::PublisherPressurePolicy::kBoundedRetry;
  publisher_options.bounded_retry_budget_ns =
      static_cast<std::uint64_t>(std::chrono::seconds(1).count()) * 1'000'000'000U;
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.queue_depth = 1U;
  subscriber_options.max_borrowed_samples = 2U;

  // Key setup: free only the first arm after filling both one-slot queues, then publish sequence 2.
  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, service_options, publisher_options));
  ASSERT_OPTIONAL(first, make_subscriber(context, topic, service_options, subscriber_options));
  ASSERT_OPTIONAL(second, make_subscriber(context, topic, service_options, subscriber_options));
  ASSERT_TRUE(wait_for_subscriber_count(publisher, 2U, std::chrono::seconds(2)));
  ASSERT_EQ(publish_one(publisher, 1U), 2);
  const auto first_initial = first.try_take();
  ASSERT_TRUE(first_initial);
  ASSERT_EQ(first_initial->payload().sequence, 1U);

  std::promise<void> entered_send;
  auto entered = entered_send.get_future();
  auto publication =
      std::async(std::launch::async, [&publisher, signal = std::move(entered_send)]() mutable {
        auto message = publisher.loan();
        if (!message) {
          return holomq::expected<int, holomq::Error>(holomq::make_unexpected(message.error()));
        }
        message->payload().sequence = 2U;
        signal.set_value();
        return publisher.send(std::move(*message));
      });
  ASSERT_EQ(entered.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  auto first_second = wait_for_sample(first, std::chrono::milliseconds(100));
  const bool blocked_on_second =
      publication.wait_for(std::chrono::milliseconds(10)) == std::future_status::timeout;
  const auto second_initial = second.try_take();
  ASSERT_TRUE(second_initial);
  ASSERT_EQ(second_initial->payload().sequence, 1U);
  ASSERT_EQ(publication.wait_for(std::chrono::seconds(2)), std::future_status::ready);
  const auto delivered = publication.get();
  auto second_second = wait_for_sample(second, std::chrono::seconds(2));

  // Key assertion: retry completes the original send without redelivering to its accepted arm.
  ASSERT_TRUE(first_second);
  EXPECT_EQ(first_second->payload().sequence, 2U);
  EXPECT_TRUE(blocked_on_second);
  ASSERT_TRUE(delivered);
  EXPECT_EQ(*delivered, 2);
  ASSERT_TRUE(second_second);
  EXPECT_EQ(second_second->payload().sequence, 2U);
  expect_no_sample(first);
  expect_no_sample(second);
}

// A bounded-retry fanout send that times out on one recipient must name that recipient, even when
// another recipient is briefly full and then accepts the same publication within its own budget.
// Each round fills both one-slot queues; the stuck recipient never drains, and the recovering one
// drains after one and a half budgets. Whichever recipient the backend visits second is still full
// when the first one's budget expires, so swapping the roles exercises the brief-pressure case
// regardless of backend iteration order.
// REQ: CHAN-113
TEST(PubSubOptionsTest, BoundedRetryAttributesFailureOnlyToRejectingRecipient) {
  REQUIRE_CONTEXT(context);
  constexpr std::chrono::milliseconds kBudget{200};
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscribers = 2U;
  service_options.max_subscriber_queue_depth = 1U;
  service_options.max_subscriber_borrowed_samples = 2U;
  holomq::PublisherPubSubOptions publisher_options{};
  publisher_options.max_loaned_samples = 1U;
  publisher_options.expected_recipient_count = 2U;
  publisher_options.pressure_policy = holomq::PublisherPressurePolicy::kBoundedRetry;
  publisher_options.bounded_retry_budget_ns =
      static_cast<std::uint64_t>(std::chrono::nanoseconds{kBudget}.count());
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.queue_depth = 1U;
  subscriber_options.max_borrowed_samples = 2U;

  bool recovered_after_pressure = false;
  for (const bool stuck_created_first : {true, false}) {
    SCOPED_TRACE(stuck_created_first ? "stuck recipient created first"
                                     : "stuck recipient created second");
    const std::string topic = make_unique_topic();
    ASSERT_OPTIONAL(publisher, make_publisher(context, topic, service_options, publisher_options));
    ASSERT_OPTIONAL(first, make_subscriber(context, topic, service_options, subscriber_options));
    ASSERT_OPTIONAL(second, make_subscriber(context, topic, service_options, subscriber_options));
    auto& stuck = stuck_created_first ? first : second;
    auto& recovering = stuck_created_first ? second : first;
    ASSERT_TRUE(wait_for_subscriber_count(publisher, 2U, std::chrono::seconds(2)));
    ASSERT_EQ(publish_one(publisher, 1U), 2);

    auto message = publisher.loan();
    ASSERT_TRUE(message);
    message->payload().sequence = 2U;
    const std::chrono::milliseconds drain_delay = kBudget * 3 / 2;
    auto drained = std::async(std::launch::async, [&recovering, drain_delay] {
      std::this_thread::sleep_for(drain_delay);
      return recovering.try_take();
    });
    const auto sent = publisher.send(std::move(*message));
    const auto drained_initial = drained.get();

    ASSERT_FALSE(sent);
    EXPECT_EQ(sent.error().code, holomq::ErrorCode::kBackpressured);
    ASSERT_TRUE(drained_initial);
    EXPECT_EQ(drained_initial->payload().sequence, 1U);
    const auto preserved = stuck.try_take();
    ASSERT_TRUE(preserved);
    EXPECT_EQ(preserved->payload().sequence, 1U);
    expect_no_sample(stuck);
    const std::string stuck_evidence =
        holomq::detail::transport::unable_to_deliver_recipient_diagnostic(stuck.recipient_id());
    const auto recovered = recovering.try_take();
    if (recovered) {
      // Key assertion: a recipient that accepted after brief pressure is never named.
      recovered_after_pressure = true;
      EXPECT_EQ(recovered->payload().sequence, 2U);
      EXPECT_EQ(sent.error().message, stuck_evidence);
    } else {
      // Both recipients rejected the publication; either one is a correct attribution.
      const std::string recovering_evidence =
          holomq::detail::transport::unable_to_deliver_recipient_diagnostic(
              recovering.recipient_id());
      EXPECT_TRUE(sent.error().message == stuck_evidence ||
                  sent.error().message == recovering_evidence)
          << sent.error().message;
    }
  }
  // The recovering recipient accepted in at least one round, so the brief-pressure case ran.
  EXPECT_TRUE(recovered_after_pressure);
}

// A set cancellation flag ends a bounded retry at once: the full recipient is rejected instead of
// being retried for the whole budget, and keeps its queued sample. The send reports the
// cancellation as kInterrupted, with the rejecting recipient's attribution, so the caller can tell
// it from an expired budget.
// REQ: CHAN-158
TEST(PubSubOptionsTest, BoundedRetryCancellationRejectsFullRecipientAtOnce) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  std::atomic<bool> cancelled{false};
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscriber_queue_depth = 1U;
  service_options.max_subscriber_borrowed_samples = 2U;
  holomq::PublisherPubSubOptions publisher_options{};
  publisher_options.max_loaned_samples = 1U;
  publisher_options.pressure_policy = holomq::PublisherPressurePolicy::kBoundedRetry;
  publisher_options.bounded_retry_budget_ns =
      static_cast<std::uint64_t>(std::chrono::nanoseconds{std::chrono::seconds(10)}.count());
  publisher_options.bounded_retry_cancellation = &cancelled;
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.queue_depth = 1U;
  subscriber_options.max_borrowed_samples = 2U;

  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, service_options, publisher_options));
  ASSERT_OPTIONAL(subscriber, make_subscriber(context, topic, service_options, subscriber_options));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));
  ASSERT_EQ(publish_one(publisher, 1U), 1);

  // Key setup: the only queue slot is full, and the flag is set before the next send.
  cancelled.store(true);
  auto message = publisher.loan();
  ASSERT_TRUE(message.has_value());
  message->payload().sequence = 2U;
  const auto started = std::chrono::steady_clock::now();
  const auto rejected = publisher.send(std::move(*message));
  const auto elapsed = std::chrono::steady_clock::now() - started;

  // Key assertion: an immediate cancelled rejection, far inside the ten-second budget.
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().code, holomq::ErrorCode::kInterrupted);
  EXPECT_EQ(
      rejected.error().message,
      holomq::detail::transport::unable_to_deliver_recipient_diagnostic(subscriber.recipient_id()));
  EXPECT_LT(elapsed, std::chrono::seconds(1));
  const auto preserved = subscriber.try_take();
  ASSERT_TRUE(preserved.has_value());
  EXPECT_EQ(preserved->payload().sequence, 1U);
  expect_no_sample(subscriber);
}

// REQ: CHAN-158
TEST(PubSubOptionsTest, RejectsRetryCancellationWithoutBoundedRetry) {
  REQUIRE_CONTEXT(context);
  const std::atomic<bool> cancelled{false};
  holomq::PublisherPubSubOptions publisher_options{};
  publisher_options.bounded_retry_cancellation = &cancelled;

  const auto publisher = make_publisher(
      context, make_unique_topic(), holomq::PubSubServiceOptions{}, publisher_options);
  ASSERT_FALSE(publisher);
  EXPECT_EQ(publisher.error().code, holomq::ErrorCode::kInvalidArgument);
}

// REQ: CHAN-101
// REQ: CHAN-113
TEST(PubSubOptionsTest, BoundedRetryExhaustionReturnsWithoutDiscardingQueuedSample) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscriber_queue_depth = 1U;
  service_options.max_subscriber_borrowed_samples = 2U;
  holomq::PublisherPubSubOptions publisher_options{};
  publisher_options.max_loaned_samples = 1U;
  publisher_options.pressure_policy = holomq::PublisherPressurePolicy::kBoundedRetry;
  publisher_options.bounded_retry_budget_ns = 1'000'000U;
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.queue_depth = 1U;
  subscriber_options.max_borrowed_samples = 2U;

  // Key setup: hold the only subscriber queue slot beyond the one-millisecond retry budget.
  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, service_options, publisher_options));
  ASSERT_OPTIONAL(subscriber, make_subscriber(context, topic, service_options, subscriber_options));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));
  ASSERT_EQ(publish_one(publisher, 1U), 1);

  const auto started = std::chrono::steady_clock::now();
  auto message = publisher.loan();
  ASSERT_TRUE(message.has_value());
  message->payload().sequence = 2U;
  const auto exhausted = publisher.send(std::move(*message));
  const auto elapsed = std::chrono::steady_clock::now() - started;

  // Key assertion: sustained pressure returns a typed bounded failure, leaves the prior sample
  // intact, and never turns the finite budget into an unbounded publisher wait.
  ASSERT_FALSE(exhausted.has_value());
  EXPECT_EQ(exhausted.error().code, holomq::ErrorCode::kBackpressured);
  EXPECT_LT(elapsed, std::chrono::seconds(1));
  const auto preserved = subscriber.try_take();
  ASSERT_TRUE(preserved.has_value());
  EXPECT_EQ(preserved->payload().sequence, 1U);
}

// REQ: CHAN-101
TEST(PubSubOptionsTest, BoundedRetryExhaustionDoesNotAccumulatePendingFanout) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscribers = 2U;
  service_options.max_subscriber_queue_depth = 1U;
  service_options.max_subscriber_borrowed_samples = 2U;
  holomq::PublisherPubSubOptions publisher_options{};
  publisher_options.max_loaned_samples = 1U;
  publisher_options.expected_recipient_count = 2U;
  publisher_options.pressure_policy = holomq::PublisherPressurePolicy::kBoundedRetry;
  publisher_options.bounded_retry_budget_ns = 100'000U;
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.queue_depth = 1U;
  subscriber_options.max_borrowed_samples = 2U;

  // Key setup: fill both bounded recipient queues, then repeatedly exhaust the finite retry
  // budget without allowing either subscriber to make progress.
  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, service_options, publisher_options));
  ASSERT_OPTIONAL(first, make_subscriber(context, topic, service_options, subscriber_options));
  ASSERT_OPTIONAL(second, make_subscriber(context, topic, service_options, subscriber_options));
  ASSERT_TRUE(wait_for_subscriber_count(publisher, 2U, std::chrono::seconds(2)));
  ASSERT_EQ(publish_one(publisher, 1U), 2);
  for (std::uint64_t sequence = 2U; sequence <= 17U; ++sequence) {
    auto message = publisher.loan();
    ASSERT_TRUE(message);
    message->payload().sequence = sequence;
    const auto exhausted = publisher.send(std::move(*message));
    ASSERT_FALSE(exhausted);
    EXPECT_EQ(exhausted.error().code, holomq::ErrorCode::kBackpressured);
  }

  // Key assertion: failed sends did not create a hidden pending queue; each capacity-one arm
  // contains only the original sample, and the publisher recovers after those samples retire.
  const auto first_preserved = first.try_take();
  const auto second_preserved = second.try_take();
  ASSERT_TRUE(first_preserved);
  ASSERT_TRUE(second_preserved);
  EXPECT_EQ(first_preserved->payload().sequence, 1U);
  EXPECT_EQ(second_preserved->payload().sequence, 1U);
  expect_no_sample(first);
  expect_no_sample(second);

  ASSERT_EQ(publish_one(publisher, 18U), 2);
  const auto first_recovered = first.try_take();
  const auto second_recovered = second.try_take();
  ASSERT_TRUE(first_recovered);
  ASSERT_TRUE(second_recovered);
  EXPECT_EQ(first_recovered->payload().sequence, 18U);
  EXPECT_EQ(second_recovered->payload().sequence, 18U);
}

TEST(PubSubOptionsTest, DropOldestKeepsNewestConfiguredNumberOfSamples) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  holomq::PubSubServiceOptions service_options{};
  service_options.queue_policy = holomq::QueuePolicy::kDropOldest;
  service_options.max_subscriber_queue_depth = 3U;
  holomq::PublisherPubSubOptions publisher_options{};
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.queue_depth = 3U;

  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, service_options, publisher_options));
  ASSERT_OPTIONAL(subscriber, make_subscriber(context, topic, service_options, subscriber_options));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  for (std::uint64_t sequence = 1U; sequence <= 5U; ++sequence) {
    ASSERT_GT(publish_one(publisher, sequence), 0);
  }

  for (std::uint64_t expected_sequence = 3U; expected_sequence <= 5U; ++expected_sequence) {
    const auto sample = subscriber.try_take();
    ASSERT_TRUE(sample.has_value());
    EXPECT_EQ(sample->payload().sequence, expected_sequence);
  }
  EXPECT_FALSE(subscriber.try_take().has_value());
}

TEST(PubSubOptionsTest, SubscriberFirstDropOldestUsesConfiguredQueueDepth) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  holomq::PubSubServiceOptions service_options{};
  service_options.queue_policy = holomq::QueuePolicy::kDropOldest;
  service_options.max_subscriber_queue_depth = 3U;
  holomq::PublisherPubSubOptions publisher_options{};
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.queue_depth = 3U;

  ASSERT_OPTIONAL(subscriber, make_subscriber(context, topic, service_options, subscriber_options));
  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, service_options, publisher_options));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  for (std::uint64_t sequence = 1U; sequence <= 4U; ++sequence) {
    ASSERT_GT(publish_one(publisher, sequence), 0);
  }

  for (std::uint64_t expected_sequence = 2U; expected_sequence <= 4U; ++expected_sequence) {
    const auto sample = subscriber.try_take();
    ASSERT_TRUE(sample.has_value());
    EXPECT_EQ(sample->payload().sequence, expected_sequence);
  }
  EXPECT_FALSE(subscriber.try_take().has_value());
}

TEST(PubSubOptionsTest, RejectsZeroQueueDepths) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscriber_queue_depth = 0U;
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.queue_depth = 0U;

  const auto publisher = context.publisher_builder(topic)
                             .service_options(service_options)
                             .payload_type<PubSubOptionsTestPayload>()
                             .create();
  EXPECT_FALSE(publisher.has_value());

  const auto subscriber = context.subscriber_builder(topic)
                              .service_options(service_options)
                              .subscriber_options(subscriber_options)
                              .payload_type<PubSubOptionsTestPayload>()
                              .create();
  EXPECT_FALSE(subscriber.has_value());
}

TEST(PubSubOptionsTest, RejectsExpectedRecipientCountAboveServiceCapacity) {
  REQUIRE_CONTEXT(context);
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscribers = 2U;
  holomq::PublisherPubSubOptions publisher_options{};
  publisher_options.expected_recipient_count = 3U;

  const auto publisher =
      make_publisher(context, make_unique_topic(), service_options, publisher_options);
  ASSERT_FALSE(publisher);
  EXPECT_EQ(publisher.error().code, holomq::ErrorCode::kInvalidArgument);
}

// REQ: CHAN-064
TEST(PubSubOptionsTest, RejectsUnsupportedQueuePolicyBeforePublisherFirstServiceCreation) {
  REQUIRE_CONTEXT(context);
  holomq::PubSubServiceOptions invalid_service{};
  invalid_service.queue_policy = static_cast<holomq::QueuePolicy>(2U);
  const holomq::PubSubServiceOptions valid_service{};
  const holomq::PublisherPubSubOptions publisher_options{};
  const holomq::SubscriberPubSubOptions subscriber_options{};

  const std::string publisher_first_topic = make_unique_topic();
  const auto invalid_publisher =
      make_publisher(context, publisher_first_topic, invalid_service, publisher_options);
  ASSERT_FALSE(invalid_publisher.has_value());
  EXPECT_EQ(invalid_publisher.error().code, holomq::ErrorCode::kInvalidArgument);
  EXPECT_TRUE(make_subscriber(context, publisher_first_topic, valid_service, subscriber_options)
                  .has_value());
}

// REQ: CHAN-064
TEST(PubSubOptionsTest, RejectsUnsupportedQueuePolicyBeforeSubscriberFirstServiceCreation) {
  REQUIRE_CONTEXT(context);
  holomq::PubSubServiceOptions invalid_service{};
  invalid_service.queue_policy = static_cast<holomq::QueuePolicy>(0xFFU);
  const holomq::PubSubServiceOptions valid_service{};
  const holomq::PublisherPubSubOptions publisher_options{};
  const holomq::SubscriberPubSubOptions subscriber_options{};

  const std::string subscriber_first_topic = make_unique_topic();
  const auto invalid_subscriber =
      make_subscriber(context, subscriber_first_topic, invalid_service, subscriber_options);
  ASSERT_FALSE(invalid_subscriber.has_value());
  EXPECT_EQ(invalid_subscriber.error().code, holomq::ErrorCode::kInvalidArgument);
  EXPECT_TRUE(make_publisher(context, subscriber_first_topic, valid_service, publisher_options)
                  .has_value());
}

// REQ: CHAN-101
TEST(PubSubOptionsTest, RejectsUnsupportedPublisherPressurePolicy) {
  REQUIRE_CONTEXT(context);
  holomq::PublisherPubSubOptions publisher_options{};
  publisher_options.pressure_policy = static_cast<holomq::PublisherPressurePolicy>(0xFFU);

  const auto publisher = make_publisher(
      context, make_unique_topic(), holomq::PubSubServiceOptions{}, publisher_options);
  ASSERT_FALSE(publisher.has_value());
  EXPECT_EQ(publisher.error().code, holomq::ErrorCode::kInvalidArgument);
}

// REQ: CHAN-101
TEST(PubSubOptionsTest, RejectWhenFullRequiresBlockingService) {
  REQUIRE_CONTEXT(context);
  holomq::PubSubServiceOptions service_options{};
  service_options.queue_policy = holomq::QueuePolicy::kDropOldest;
  holomq::PublisherPubSubOptions publisher_options{};
  publisher_options.pressure_policy = holomq::PublisherPressurePolicy::kRejectWhenFull;

  const auto publisher =
      make_publisher(context, make_unique_topic(), service_options, publisher_options);
  ASSERT_FALSE(publisher.has_value());
  EXPECT_EQ(publisher.error().code, holomq::ErrorCode::kInvalidArgument);
}

// REQ: CHAN-101
TEST(PubSubOptionsTest, BoundedRetryRequiresBlockingServiceAndExclusivePositiveBudget) {
  REQUIRE_CONTEXT(context);
  holomq::PubSubServiceOptions lossy_service{};
  lossy_service.queue_policy = holomq::QueuePolicy::kDropOldest;
  holomq::PublisherPubSubOptions bounded{};
  bounded.pressure_policy = holomq::PublisherPressurePolicy::kBoundedRetry;
  bounded.bounded_retry_budget_ns = 1U;

  // Key setup: exercise each invalid policy/budget combination before backend creation.
  const auto lossy = make_publisher(context, make_unique_topic(), lossy_service, bounded);
  holomq::PubSubServiceOptions blocking_service{};
  bounded.bounded_retry_budget_ns = 0U;
  const auto zero_budget = make_publisher(context, make_unique_topic(), blocking_service, bounded);
  holomq::PublisherPubSubOptions ignored_budget{};
  ignored_budget.bounded_retry_budget_ns = 1U;
  const auto wrong_policy =
      make_publisher(context, make_unique_topic(), blocking_service, ignored_budget);

  // Key assertion: bounded retry is admitted only for a blocking service with an explicit nonzero
  // budget, and other policies cannot silently ignore a supplied budget.
  EXPECT_FALSE(lossy.has_value());
  EXPECT_EQ(lossy.error().code, holomq::ErrorCode::kInvalidArgument);
  EXPECT_FALSE(zero_budget.has_value());
  EXPECT_EQ(zero_budget.error().code, holomq::ErrorCode::kInvalidArgument);
  EXPECT_FALSE(wrong_policy.has_value());
  EXPECT_EQ(wrong_policy.error().code, holomq::ErrorCode::kInvalidArgument);
}

TEST(PubSubOptionsTest, ZeroBorrowedSampleAllowancesDeriveFromQueueDepth) {
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscriber_borrowed_samples = 0U;
  service_options.max_subscriber_queue_depth = 4U;
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.max_borrowed_samples = 0U;
  subscriber_options.queue_depth = 2U;

  EXPECT_EQ(service_options.effective_max_subscriber_borrowed_samples(), 4U);
  EXPECT_EQ(subscriber_options.effective_max_borrowed_samples(), 2U);
}

TEST(PubSubOptionsTest, AllowsMixedSubscriberDepthsWithinPublisherMaximum) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscriber_queue_depth = 4U;
  holomq::PublisherPubSubOptions publisher_options{};
  holomq::SubscriberPubSubOptions shallow_subscriber_options{};
  shallow_subscriber_options.queue_depth = 1U;
  holomq::SubscriberPubSubOptions deeper_subscriber_options{};
  deeper_subscriber_options.queue_depth = 2U;

  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, service_options, publisher_options));
  ASSERT_OPTIONAL(shallow_subscriber,
                  make_subscriber(context, topic, service_options, shallow_subscriber_options));
  ASSERT_OPTIONAL(deeper_subscriber,
                  make_subscriber(context, topic, service_options, deeper_subscriber_options));
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  ASSERT_EQ(publish_one(publisher, 1U), 2);

  const auto shallow_sample = shallow_subscriber.try_take();
  ASSERT_TRUE(shallow_sample.has_value());
  EXPECT_EQ(shallow_sample->payload().sequence, 1U);
  const auto deeper_sample = deeper_subscriber.try_take();
  ASSERT_TRUE(deeper_sample.has_value());
  EXPECT_EQ(deeper_sample->payload().sequence, 1U);
}

TEST(PubSubOptionsTest, ExplicitSubscriberCapacityLimitsCreatedPorts) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscribers = 1U;
  holomq::PublisherPubSubOptions publisher_options{};
  holomq::SubscriberPubSubOptions subscriber_options{};

  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, service_options, publisher_options));
  ASSERT_OPTIONAL(subscriber, make_subscriber(context, topic, service_options, subscriber_options));
  EXPECT_FALSE(make_subscriber(context, topic, service_options, subscriber_options).has_value());
}

// REQ: CHAN-064
TEST(PubSubOptionsTest, SubscriberFirstCreationUsesExplicitCapacity) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscribers = 1U;
  holomq::PublisherPubSubOptions publisher_options{};
  holomq::SubscriberPubSubOptions subscriber_options{};

  ASSERT_OPTIONAL(subscriber, make_subscriber(context, topic, service_options, subscriber_options));
  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, service_options, publisher_options));
  EXPECT_FALSE(make_subscriber(context, topic, service_options, subscriber_options).has_value());
}

// REQ: CHAN-064
TEST(PubSubOptionsTest, ExplicitCapacityConfiguresEventWakeListenersPublisherFirst) {
  expect_explicit_event_wake_capacity(false);
}

// REQ: CHAN-064
TEST(PubSubOptionsTest, ExplicitCapacityConfiguresEventWakeListenersSubscriberFirst) {
  expect_explicit_event_wake_capacity(true);
}

// REQ: CHAN-064
TEST(PubSubOptionsTest, BothEndpointOrdersPreserveTheSameStaticServiceContract) {
  REQUIRE_CONTEXT(context);
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscribers = 2U;
  service_options.max_subscriber_queue_depth = 3U;
  service_options.max_subscriber_borrowed_samples = 4U;
  service_options.publisher_history_depth = 1U;
  holomq::PublisherPubSubOptions publisher_options{};
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.queue_depth = 2U;
  subscriber_options.max_borrowed_samples = 2U;

  const std::string publisher_first_topic = make_unique_topic();
  ASSERT_OPTIONAL(
      publisher_first,
      make_publisher(context, publisher_first_topic, service_options, publisher_options));
  ASSERT_OPTIONAL(
      subscriber_second,
      make_subscriber(context, publisher_first_topic, service_options, subscriber_options));

  const std::string subscriber_first_topic = make_unique_topic();
  ASSERT_OPTIONAL(
      subscriber_first,
      make_subscriber(context, subscriber_first_topic, service_options, subscriber_options));
  ASSERT_OPTIONAL(
      publisher_second,
      make_publisher(context, subscriber_first_topic, service_options, publisher_options));
}

// REQ: CHAN-064
TEST(PubSubOptionsTest, SubscriberFirstCreationPreservesPublisherHistoryForLateJoiner) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscribers = 2U;
  service_options.max_subscriber_queue_depth = 4U;
  service_options.max_subscriber_borrowed_samples = 4U;
  service_options.publisher_history_depth = 2U;
  const holomq::PublisherPubSubOptions publisher_options{};
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.queue_depth = 4U;
  subscriber_options.max_borrowed_samples = 4U;

  ASSERT_OPTIONAL(service_creator,
                  make_subscriber(context, topic, service_options, subscriber_options));
  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, service_options, publisher_options));
  ASSERT_TRUE(service_creator.has_samples().has_value());
  ASSERT_TRUE(wait_for_subscribers(publisher, std::chrono::seconds(2)));

  ASSERT_EQ(publish_one(publisher, 1U), 1);
  ASSERT_EQ(publish_one(publisher, 2U), 1);
  ASSERT_OPTIONAL(first_for_creator, wait_for_sample(service_creator, std::chrono::seconds(2)));
  ASSERT_OPTIONAL(second_for_creator, wait_for_sample(service_creator, std::chrono::seconds(2)));
  EXPECT_EQ(first_for_creator.payload().sequence, 1U);
  EXPECT_EQ(second_for_creator.payload().sequence, 2U);

  ASSERT_OPTIONAL(late_subscriber,
                  make_subscriber(context, topic, service_options, subscriber_options));
  ASSERT_TRUE(publisher.update_connections().has_value());
  ASSERT_OPTIONAL(first_history, wait_for_sample(late_subscriber, std::chrono::seconds(2)));
  ASSERT_OPTIONAL(second_history, wait_for_sample(late_subscriber, std::chrono::seconds(2)));
  EXPECT_EQ(first_history.payload().sequence, 1U);
  EXPECT_EQ(second_history.payload().sequence, 2U);
}

// REQ: CHAN-064
TEST(PubSubOptionsTest, RejectsEventWakeMismatchRegardlessOfEndpointOrder) {
  REQUIRE_CONTEXT(context);
  const holomq::PubSubServiceOptions event_service{};
  auto polling_service = event_service;
  polling_service.enable_event_wake = false;
  const holomq::PublisherPubSubOptions publisher_options{};
  const holomq::SubscriberPubSubOptions subscriber_options{};

  const std::string publisher_first_topic = make_unique_topic();
  ASSERT_OPTIONAL(publisher,
                  make_publisher(context, publisher_first_topic, event_service, publisher_options));
  const auto mismatched_subscriber =
      make_subscriber(context, publisher_first_topic, polling_service, subscriber_options);
  ASSERT_FALSE(mismatched_subscriber.has_value());
  EXPECT_EQ(mismatched_subscriber.error().code, holomq::ErrorCode::kIncompatible);

  const std::string subscriber_first_topic = make_unique_topic();
  ASSERT_OPTIONAL(
      subscriber,
      make_subscriber(context, subscriber_first_topic, polling_service, subscriber_options));
  const auto mismatched_publisher =
      make_publisher(context, subscriber_first_topic, event_service, publisher_options);
  ASSERT_FALSE(mismatched_publisher.has_value());
  EXPECT_EQ(mismatched_publisher.error().code, holomq::ErrorCode::kIncompatible);
}

TEST(PubSubOptionsTest, RejectsQueuePolicyMismatchRegardlessOfEndpointOrder) {
  REQUIRE_CONTEXT(context);
  const holomq::PubSubServiceOptions blocking_service{};
  holomq::PubSubServiceOptions lossy_service{};
  lossy_service.queue_policy = holomq::QueuePolicy::kDropOldest;
  const holomq::PublisherPubSubOptions publisher_options{};
  const holomq::SubscriberPubSubOptions subscriber_options{};

  const std::string publisher_first_topic = make_unique_topic();
  ASSERT_OPTIONAL(
      publisher,
      make_publisher(context, publisher_first_topic, blocking_service, publisher_options));
  const auto mismatched_subscriber =
      make_subscriber(context, publisher_first_topic, lossy_service, subscriber_options);
  ASSERT_FALSE(mismatched_subscriber.has_value());
  EXPECT_EQ(mismatched_subscriber.error().code, holomq::ErrorCode::kIncompatible);

  const std::string subscriber_first_topic = make_unique_topic();
  ASSERT_OPTIONAL(
      subscriber,
      make_subscriber(context, subscriber_first_topic, blocking_service, subscriber_options));
  const auto mismatched_publisher =
      make_publisher(context, subscriber_first_topic, lossy_service, publisher_options);
  ASSERT_FALSE(mismatched_publisher.has_value());
  EXPECT_EQ(mismatched_publisher.error().code, holomq::ErrorCode::kIncompatible);
}

// REQ: CHAN-064
TEST(PubSubOptionsTest, RejectsDistinctDropOldestDepthContracts) {
  REQUIRE_CONTEXT(context);
  holomq::PubSubServiceOptions established{};
  established.queue_policy = holomq::QueuePolicy::kDropOldest;
  established.max_subscribers = 1U;
  established.max_subscriber_queue_depth = 4U;
  established.max_subscriber_borrowed_samples = 4U;
  const holomq::PublisherPubSubOptions publisher_options{};
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.queue_depth = 1U;
  subscriber_options.max_borrowed_samples = 1U;

  const auto expect_mismatch = [&](const holomq::PubSubServiceOptions& requested) {
    const std::string topic = make_unique_topic();
    auto publisher = make_publisher(context, topic, established, publisher_options);
    ASSERT_TRUE(publisher.has_value());
    auto subscriber = make_subscriber(context, topic, requested, subscriber_options);
    ASSERT_FALSE(subscriber.has_value());
    EXPECT_EQ(subscriber.error().code, holomq::ErrorCode::kIncompatible);
  };

  auto requested = established;
  requested.max_subscriber_queue_depth = 1U;
  expect_mismatch(requested);

  requested = established;
  requested.max_subscriber_queue_depth = 8U;
  expect_mismatch(requested);
}

// REQ: CHAN-064
TEST(PubSubOptionsTest, RejectsInspectableStaticServiceContractMismatch) {
  REQUIRE_CONTEXT(context);
  const holomq::PublisherPubSubOptions publisher_options{};
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.queue_depth = 1U;
  subscriber_options.max_borrowed_samples = 1U;

  const auto expect_mismatch = [&](const holomq::PubSubServiceOptions& established,
                                   const holomq::PubSubServiceOptions& requested) {
    const std::string topic = make_unique_topic();
    auto publisher = make_publisher(context, topic, established, publisher_options);
    ASSERT_TRUE(publisher.has_value());
    auto subscriber = make_subscriber(context, topic, requested, subscriber_options);
    ASSERT_FALSE(subscriber.has_value());
    EXPECT_EQ(subscriber.error().code, holomq::ErrorCode::kIncompatible);
  };

  holomq::PubSubServiceOptions established{};
  established.max_subscribers = 2U;
  established.max_subscriber_queue_depth = 3U;
  established.max_subscriber_borrowed_samples = 4U;
  established.publisher_history_depth = 1U;

  auto requested = established;
  requested.max_subscribers = 3U;
  expect_mismatch(established, requested);
  requested = established;
  requested.max_subscriber_queue_depth = 4U;
  expect_mismatch(established, requested);
  requested = established;
  requested.max_subscriber_borrowed_samples = 5U;
  expect_mismatch(established, requested);
  requested = established;
  requested.publisher_history_depth = 2U;
  expect_mismatch(established, requested);
}

// REQ: CHAN-064
TEST(PubSubOptionsTest, RejectsSubscriberDepthGreaterThanPublisherMaximum) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscriber_queue_depth = 2U;
  holomq::PublisherPubSubOptions publisher_options{};
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.queue_depth = 3U;

  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, service_options, publisher_options));
  const auto subscriber = make_subscriber(context, topic, service_options, subscriber_options);
  EXPECT_FALSE(subscriber.has_value());
}

TEST(PubSubOptionsTest, RejectsBorrowedSamplesGreaterThanPublisherMaximum) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  holomq::PubSubServiceOptions service_options{};
  service_options.max_subscriber_borrowed_samples = 1U;
  holomq::PublisherPubSubOptions publisher_options{};
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.max_borrowed_samples = 2U;

  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, service_options, publisher_options));
  const auto subscriber = make_subscriber(context, topic, service_options, subscriber_options);
  EXPECT_FALSE(subscriber.has_value());
}

#endif  // defined(HOLOMQ_TRANSPORT_IOX2)
