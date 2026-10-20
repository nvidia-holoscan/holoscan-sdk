/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#if defined(HOLOMQ_TRANSPORT_IOX2)

#include <holomq/byte_slice.hpp>
#include <holomq/context_builder.hpp>
#include <holomq/publisher_builder.hpp>
#include <holomq/subscriber_builder.hpp>
#include <holomq/wait_set.hpp>
#include <holomq/wait_strategy.hpp>

#include <gtest/gtest.h>

#include <iox2/config.hpp>
#include <iox2/messaging_pattern.hpp>
#include <iox2/service.hpp>
#include <iox2/service_name.hpp>

#include "../../common/allocation_counter.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

struct FixedPayload {
  std::uint64_t sequence{0};
};

std::atomic<std::uint64_t> g_topic_counter{0};

[[nodiscard]] std::string make_unique_topic() {
  return "holomq_byte_slice_test_" + std::to_string(static_cast<long>(::getpid())) + "_" +
         std::to_string(g_topic_counter.fetch_add(1U));
}

[[nodiscard]] holomq::PubSubServiceOptions slice_service(const std::uint64_t capacity,
                                                         const std::uint32_t profile = 1U) {
  holomq::PubSubServiceOptions options{};
  options.byte_slice_capacity = capacity;
  options.byte_slice_profile = profile;
  return options;
}

[[nodiscard]] auto make_publisher(holomq::Context& context, const std::string& topic,
                                  const holomq::PubSubServiceOptions& service_options,
                                  const holomq::PublisherPubSubOptions& options = {}) {
  return context.publisher_builder(topic)
      .service_options(service_options)
      .publisher_options(options)
      .payload_type<holomq::ByteSlice>()
      .create();
}

[[nodiscard]] auto make_subscriber(holomq::Context& context, const std::string& topic,
                                   const holomq::PubSubServiceOptions& service_options,
                                   const holomq::SubscriberPubSubOptions& options = {}) {
  return context.subscriber_builder(topic)
      .service_options(service_options)
      .subscriber_options(options)
      .payload_type<holomq::ByteSlice>()
      .create();
}

template <typename Payload>
[[nodiscard]] bool wait_for_subscribers(holomq::Publisher<Payload>& publisher) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (std::chrono::steady_clock::now() < deadline) {
    if (publisher.update_connections().has_value() && publisher.has_subscribers()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return false;
}

/// Fill a frame with a pattern that depends on its length and position.
void fill_frame(const std::span<std::byte> frame) {
  for (std::size_t i = 0; i < frame.size(); ++i) {
    frame[i] = static_cast<std::byte>((i * 31U + frame.size()) & 0xFFU);
  }
}

[[nodiscard]] bool frame_matches(const std::span<const std::byte> frame,
                                 const std::size_t expected_size) {
  if (frame.size() != expected_size) {
    return false;
  }
  for (std::size_t i = 0; i < frame.size(); ++i) {
    if (frame[i] != static_cast<std::byte>((i * 31U + frame.size()) & 0xFFU)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] holomq::expected<int, holomq::Error> publish_frame(
    holomq::Publisher<holomq::ByteSlice>& publisher, const std::size_t length) {
  auto message = publisher.loan(length);
  if (!message.has_value()) {
    return holomq::make_unexpected(message.error());
  }
  fill_frame(message->payload());
  return publisher.send(std::move(message).value());
}

[[nodiscard]] auto wait_for_frame(holomq::Subscriber<holomq::ByteSlice>& subscriber) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (true) {
    auto frame = subscriber.try_take();
    if (frame.has_value() || frame.error().code != holomq::ErrorCode::kNotReady ||
        std::chrono::steady_clock::now() >= deadline) {
      return frame;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
}

#define ASSERT_OPTIONAL(var, expr)                                                \
  auto var##_opt = (expr);                                                        \
  ASSERT_TRUE(var##_opt.has_value()) << holomq::error_message(var##_opt.error()); \
  auto var = std::move(var##_opt).value()

}  // namespace

#define REQUIRE_CONTEXT(var) ASSERT_OPTIONAL(var, holomq::ContextBuilder().create())

// A frame of any length up to the capacity arrives with exactly its published bytes, aligned for
// the widest scalar a frame header can hold.
// REQ: MQ-SLICE-001
TEST(ByteSlicePubSubTest, DeliversEveryFrameLengthUpToCapacity) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  constexpr std::uint64_t kCapacity = 64U;
  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, slice_service(kCapacity)));
  ASSERT_OPTIONAL(subscriber, make_subscriber(context, topic, slice_service(kCapacity)));
  ASSERT_TRUE(wait_for_subscribers(publisher));

  for (const std::size_t length : {std::size_t{1}, std::size_t{17}, std::size_t{kCapacity}}) {
    const auto sent = publish_frame(publisher, length);
    ASSERT_TRUE(sent.has_value()) << holomq::error_message(sent.error());
    EXPECT_EQ(sent.value(), 1);
    auto frame = wait_for_frame(subscriber);
    ASSERT_TRUE(frame.has_value()) << holomq::error_message(frame.error());
    EXPECT_TRUE(frame_matches(frame->payload(), length)) << "length " << length;
    EXPECT_EQ(
        reinterpret_cast<std::uintptr_t>(frame->payload().data()) % holomq::kByteSliceAlignment,
        0U);
  }
}

// A zero-length frame is a valid publication; the subscriber sees an empty view.
// REQ: MQ-SLICE-001
TEST(ByteSlicePubSubTest, DeliversAnEmptyFrame) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, slice_service(32U)));
  ASSERT_OPTIONAL(subscriber, make_subscriber(context, topic, slice_service(32U)));
  ASSERT_TRUE(wait_for_subscribers(publisher));

  const auto sent = publish_frame(publisher, 0U);
  ASSERT_TRUE(sent.has_value()) << holomq::error_message(sent.error());
  auto frame = wait_for_frame(subscriber);
  ASSERT_TRUE(frame.has_value()) << holomq::error_message(frame.error());
  EXPECT_TRUE(frame->payload().empty());
}

// The static pool never grows: a loan above the capacity fails without disturbing later loans.
// REQ: MQ-SLICE-001
TEST(ByteSlicePubSubTest, RejectsALoanAboveCapacity) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, slice_service(64U)));

  const auto oversized = publisher.loan(65U);
  ASSERT_FALSE(oversized.has_value());
  EXPECT_EQ(oversized.error().code, holomq::ErrorCode::kInvalidArgument);

  const auto full = publisher.loan(64U);
  ASSERT_TRUE(full.has_value()) << holomq::error_message(full.error());
  EXPECT_EQ(full->payload().size(), 64U);
}

// Services in one process keep independent capacities, including the 32 KiB inline maximum.
// REQ: MQ-SLICE-001
TEST(ByteSlicePubSubTest, KeepsIndependentCapacitiesPerService) {
  REQUIRE_CONTEXT(context);
  constexpr std::uint64_t kCapacities[] = {4U, 4096U, 16384U, 32768U + 64U};
  std::vector<holomq::Publisher<holomq::ByteSlice>> publishers;
  std::vector<holomq::Subscriber<holomq::ByteSlice>> subscribers;
  for (const std::uint64_t capacity : kCapacities) {
    const std::string topic = make_unique_topic();
    ASSERT_OPTIONAL(publisher, make_publisher(context, topic, slice_service(capacity)));
    ASSERT_OPTIONAL(subscriber, make_subscriber(context, topic, slice_service(capacity)));
    publishers.push_back(std::move(publisher));
    subscribers.push_back(std::move(subscriber));
  }
  for (std::size_t i = 0; i < publishers.size(); ++i) {
    ASSERT_TRUE(wait_for_subscribers(publishers[i]));
    const auto sent = publish_frame(publishers[i], kCapacities[i]);
    ASSERT_TRUE(sent.has_value()) << holomq::error_message(sent.error());
    const auto oversized = publishers[i].loan(kCapacities[i] + 1U);
    ASSERT_FALSE(oversized.has_value());
    EXPECT_EQ(oversized.error().code, holomq::ErrorCode::kInvalidArgument);
    auto frame = wait_for_frame(subscribers[i]);
    ASSERT_TRUE(frame.has_value()) << holomq::error_message(frame.error());
    EXPECT_TRUE(frame_matches(frame->payload(), kCapacities[i]));
  }
}

// Capacity and frame profile are part of the service contract: a peer with a different value
// cannot open the service, whichever endpoint created it.
// REQ: MQ-SLICE-001
TEST(ByteSlicePubSubTest, RejectsAPeerWithADifferentCapacityOrProfile) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, slice_service(64U, 1U)));

  const auto wider = make_subscriber(context, topic, slice_service(128U, 1U));
  ASSERT_FALSE(wider.has_value());
  EXPECT_EQ(wider.error().code, holomq::ErrorCode::kIncompatible);

  const auto other_profile = make_subscriber(context, topic, slice_service(64U, 2U));
  ASSERT_FALSE(other_profile.has_value());
  EXPECT_EQ(other_profile.error().code, holomq::ErrorCode::kIncompatible);

  const std::string subscriber_first = make_unique_topic();
  ASSERT_OPTIONAL(subscriber, make_subscriber(context, subscriber_first, slice_service(64U)));
  const auto narrower = make_publisher(context, subscriber_first, slice_service(32U));
  ASSERT_FALSE(narrower.has_value());
  EXPECT_EQ(narrower.error().code, holomq::ErrorCode::kIncompatible);
}

// A fixed-payload service and a byte-slice service of the same name never open each other.
// REQ: MQ-SLICE-001
TEST(ByteSlicePubSubTest, FixedPayloadAndByteSliceServicesDoNotMix) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_OPTIONAL(fixed, context.publisher_builder(topic).payload_type<FixedPayload>().create());
  const auto slice = make_subscriber(context, topic, slice_service(sizeof(FixedPayload)));
  ASSERT_FALSE(slice.has_value());
  EXPECT_EQ(slice.error().code, holomq::ErrorCode::kIncompatible);

  const std::string slice_first = make_unique_topic();
  ASSERT_OPTIONAL(slice_publisher, make_publisher(context, slice_first, slice_service(8U)));
  const auto fixed_subscriber =
      context.subscriber_builder(slice_first).payload_type<FixedPayload>().create();
  ASSERT_FALSE(fixed_subscriber.has_value());
  EXPECT_EQ(fixed_subscriber.error().code, holomq::ErrorCode::kIncompatible);
}

// Capacity is mandatory for a byte-slice service and meaningless for a fixed payload.
// REQ: MQ-SLICE-001
TEST(ByteSlicePubSubTest, ValidatesTheCapacityOption) {
  REQUIRE_CONTEXT(context);
  const auto no_capacity = make_publisher(context, make_unique_topic(), slice_service(0U));
  ASSERT_FALSE(no_capacity.has_value());
  EXPECT_EQ(no_capacity.error().code, holomq::ErrorCode::kInvalidArgument);
  const auto no_capacity_subscriber =
      make_subscriber(context, make_unique_topic(), slice_service(0U));
  ASSERT_FALSE(no_capacity_subscriber.has_value());
  EXPECT_EQ(no_capacity_subscriber.error().code, holomq::ErrorCode::kInvalidArgument);

  const auto fixed_with_capacity = context.publisher_builder(make_unique_topic())
                                       .service_options(slice_service(64U))
                                       .payload_type<FixedPayload>()
                                       .create();
  ASSERT_FALSE(fixed_with_capacity.has_value());
  EXPECT_EQ(fixed_with_capacity.error().code, holomq::ErrorCode::kInvalidArgument);
}

// REQ: MQ-SLICE-001
TEST(ByteSlicePubSubTest, RejectsUnrepresentableGeometryBeforeServiceCreation) {
  REQUIRE_CONTEXT(context);
  std::vector<holomq::PubSubServiceOptions> invalid_services;
  invalid_services.push_back(slice_service(std::numeric_limits<std::uint64_t>::max()));
  invalid_services.push_back(
      slice_service(static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max()) / 4U));
  auto invalid = slice_service(64U);
  invalid.max_subscribers = std::numeric_limits<std::uint32_t>::max();
  invalid_services.push_back(invalid);
  invalid = slice_service(64U);
  invalid.max_subscriber_queue_depth = std::numeric_limits<std::uint32_t>::max();
  invalid.max_subscriber_borrowed_samples = std::numeric_limits<std::uint32_t>::max();
  invalid_services.push_back(invalid);
  invalid = slice_service(64U);
  invalid.publisher_history_depth = std::numeric_limits<std::uint32_t>::max();
  invalid_services.push_back(invalid);

  const auto service_absent = [](const std::string& topic) {
    const auto name = iox2::ServiceName::create(topic.c_str());
    EXPECT_TRUE(name.has_value());
    if (!name.has_value()) {
      return;
    }
    const auto exists = iox2::Service<iox2::ServiceType::Ipc>::does_exist(
        name.value(), iox2::Config::global_config(), iox2::MessagingPattern::PublishSubscribe);
    ASSERT_TRUE(exists.has_value());
    EXPECT_FALSE(exists.value());
  };
  for (const auto& service : invalid_services) {
    for (const bool callbacks : {false, true}) {
      const std::string topic = make_unique_topic();
      holomq::PublisherPubSubOptions options{};
      options.enable_sample_release_callbacks = callbacks;
      const auto publisher = make_publisher(context, topic, service, options);
      ASSERT_FALSE(publisher.has_value());
      EXPECT_EQ(publisher.error().code, holomq::ErrorCode::kInvalidArgument);
      service_absent(topic);
      const auto subscriber = make_subscriber(context, topic, service);
      ASSERT_FALSE(subscriber.has_value());
      EXPECT_EQ(subscriber.error().code, holomq::ErrorCode::kInvalidArgument);
      service_absent(topic);
      // The rejected requests publish no contract: a small valid peer can create this name.
      ASSERT_OPTIONAL(valid, make_publisher(context, topic, slice_service(64U)));
      ASSERT_OPTIONAL(peer, make_subscriber(context, topic, slice_service(64U)));
    }
  }
  for (const bool callbacks : {false, true}) {
    const std::string topic = make_unique_topic();
    holomq::PublisherPubSubOptions options{};
    options.enable_sample_release_callbacks = callbacks;
    options.max_loaned_samples = std::numeric_limits<std::uint32_t>::max();
    const auto publisher = make_publisher(context, topic, slice_service(64U), options);
    ASSERT_FALSE(publisher.has_value());
    EXPECT_EQ(publisher.error().code, holomq::ErrorCode::kInvalidArgument);
    service_absent(topic);
  }
}

// Loaned frames count against the publisher's loan limit like fixed payload samples do.
// REQ: MQ-SLICE-002
TEST(ByteSlicePubSubTest, ReportsBackpressureWhenTheLoanLimitIsReached) {
  REQUIRE_CONTEXT(context);
  holomq::PublisherPubSubOptions options{};
  options.max_loaned_samples = 2U;
  ASSERT_OPTIONAL(publisher,
                  make_publisher(context, make_unique_topic(), slice_service(64U), options));

  auto first = publisher.loan(8U);
  auto second = publisher.loan(64U);
  ASSERT_TRUE(first.has_value());
  ASSERT_TRUE(second.has_value());
  const auto third = publisher.loan(1U);
  ASSERT_FALSE(third.has_value());
  EXPECT_EQ(third.error().code, holomq::ErrorCode::kBackpressured);
}

// The per-send release callback is keyed by the frame and runs once the subscriber releases it.
// REQ: MQ-SLICE-002
TEST(ByteSlicePubSubTest, RunsTheReleaseCallbackForASentFrame) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, slice_service(64U)));
  ASSERT_OPTIONAL(subscriber, make_subscriber(context, topic, slice_service(64U)));
  ASSERT_TRUE(wait_for_subscribers(publisher));

  std::atomic<std::uint32_t> released{0U};
  auto message = publisher.loan(24U);
  ASSERT_TRUE(message.has_value());
  fill_frame(message->payload());
  const auto sent =
      publisher.send(std::move(message).value(), [&released]() noexcept { ++released; });
  ASSERT_TRUE(sent.has_value()) << holomq::error_message(sent.error());
  {
    auto frame = wait_for_frame(subscriber);
    ASSERT_TRUE(frame.has_value());
    EXPECT_TRUE(frame_matches(frame->payload(), 24U));
  }

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (released.load() == 0U && std::chrono::steady_clock::now() < deadline) {
    static_cast<void>(publisher.loan(1U));
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  EXPECT_EQ(released.load(), 1U);
}

// A drop-oldest replacement reports the replaced frame's own bytes, not the service capacity.
// REQ: MQ-SLICE-002
TEST(ByteSlicePubSubTest, ReportsTheReplacedFrameBytes) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  holomq::PubSubServiceOptions service_options = slice_service(64U);
  service_options.queue_policy = holomq::QueuePolicy::kDropOldest;
  service_options.max_subscriber_queue_depth = 1U;

  std::vector<std::size_t> replaced_lengths;
  bool replaced_bytes_match = true;
  ASSERT_OPTIONAL(publisher,
                  context.publisher_builder(topic)
                      .service_options(service_options)
                      .payload_type<holomq::ByteSlice>()
                      .on_sample_replaced([&](const std::span<const std::byte> frame,
                                              const holomq::SampleReplacedInfo& info) {
                        replaced_lengths.push_back(frame.size());
                        replaced_bytes_match = replaced_bytes_match &&
                                               frame_matches(frame, frame.size()) &&
                                               info.payload_size == frame.size();
                      })
                      .create());
  holomq::SubscriberPubSubOptions subscriber_options{};
  subscriber_options.queue_depth = 1U;
  ASSERT_OPTIONAL(subscriber, make_subscriber(context, topic, service_options, subscriber_options));
  ASSERT_TRUE(wait_for_subscribers(publisher));

  ASSERT_TRUE(publish_frame(publisher, 10U).has_value());
  ASSERT_TRUE(publish_frame(publisher, 40U).has_value());
  EXPECT_EQ(replaced_lengths, std::vector<std::size_t>{10U});
  EXPECT_TRUE(replaced_bytes_match);

  auto frame = wait_for_frame(subscriber);
  ASSERT_TRUE(frame.has_value());
  EXPECT_TRUE(frame_matches(frame->payload(), 40U));
}

// A byte-slice subscriber takes part in kernel waits like a fixed-payload subscriber.
// REQ: MQ-SLICE-003
TEST(ByteSlicePubSubTest, WakesAWaitSet) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, slice_service(64U)));
  ASSERT_OPTIONAL(subscriber, make_subscriber(context, topic, slice_service(64U)));
  ASSERT_TRUE(wait_for_subscribers(publisher));

  auto wait_set = context.create_waitset();
  ASSERT_TRUE(wait_set.has_value());
  auto attachment = wait_set->attach(subscriber);
  ASSERT_TRUE(attachment.has_value());
  ASSERT_TRUE(publish_frame(publisher, 33U).has_value());

  holomq::WaitStrategy block{};
  block.mode = holomq::WaitMode::KernelWait;
  block.timeout = std::chrono::seconds(2);
  const auto ready = wait_set->wait_for_ready(block);
  ASSERT_TRUE(ready.has_value());
  EXPECT_EQ(ready.value(), holomq::WaitResult::ready);
  auto frame = subscriber.try_take();
  ASSERT_TRUE(frame.has_value()) << holomq::error_message(frame.error());
  EXPECT_TRUE(frame_matches(frame->payload(), 33U));
}

// After warm-up, loaning, sending, and taking frames of varying length does not allocate.
// REQ: MQ-SLICE-003
TEST(ByteSlicePubSubTest, SteadyStatePublicationDoesNotAllocate) {
  REQUIRE_CONTEXT(context);
  const std::string topic = make_unique_topic();
  constexpr std::uint64_t kCapacity = 4096U;
  ASSERT_OPTIONAL(publisher, make_publisher(context, topic, slice_service(kCapacity)));
  ASSERT_OPTIONAL(subscriber, make_subscriber(context, topic, slice_service(kCapacity)));
  ASSERT_TRUE(wait_for_subscribers(publisher));

  const auto round_trip = [&](const std::size_t length) {
    auto message = publisher.loan(length);
    if (!message.has_value()) {
      return false;
    }
    fill_frame(message->payload());
    const auto sent = publisher.send(std::move(message).value());
    if (!sent.has_value() || sent.value() != 1) {
      return false;
    }
    auto frame = subscriber.try_take();
    return frame.has_value() && frame->payload().size() == length;
  };
  for (std::size_t i = 0; i < 64U; ++i) {
    ASSERT_TRUE(round_trip(1U + (i * 97U) % kCapacity));
  }

  {
    // A direct operator new call cannot be elided, so it proves that counting is active here.
    holoscan::test::ScopedAllocationCounter probe;
    ::operator delete(::operator new(16U));
    ASSERT_EQ(probe.count(), 1U);
  }

  std::size_t allocations = 0U;
  bool delivered = true;
  {
    holoscan::test::ScopedAllocationCounter counter;
    for (std::size_t i = 0; i < 1024U; ++i) {
      delivered = round_trip(1U + (i * 97U) % kCapacity) && delivered;
    }
    allocations = counter.count();
  }
  EXPECT_TRUE(delivered);
  EXPECT_EQ(allocations, 0U);
}

#endif  // defined(HOLOMQ_TRANSPORT_IOX2)
