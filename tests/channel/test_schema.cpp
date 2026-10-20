/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/channel/endpoint_catalog_builder.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/schema/descriptor_flatbuffer_payload.hpp>
// Provides generated owning types and schema-trait specializations.
#include <holoscan/schema/inline_flatbuffer_payload.hpp>
#include <holoscan/schema/sdk_schemas.hpp>  // NOLINT(misc-include-cleaner)
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "../../src/channel/channel_catalog_binding.hpp"
#include "../../src/channel/channel_inline_payload_test_access.hpp"
#include "../../src/channel/generated_schema_adapter.hpp"
#include "../common/allocation_counter.hpp"
#include "../common/schema_id_test_utils.hpp"

#include <flatbuffers/flatbuffer_builder.h>
#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "channel_holomq_test_support.hpp"
#include "channel_test_support.hpp"
#include "custom_control_generated.h"
// Registers the generated CustomControl owning type with schema concepts.
#include "custom_control_schema_traits.hpp"  // NOLINT(misc-include-cleaner)
#include "header_generated.h"
#include "image_generated.h"
#include "imu_generated.h"
#include "string_generated.h"
#include "time_generated.h"

#include "../support/optional_assertions.hpp"
#include "schema_test_types.hpp"

namespace {

using namespace holoscan::channel_schema_test;
using holoscan::channel_test::ChildProcessCleanup;
using holoscan::channel_test::HoloMQTestMemoryKind;
using holoscan::channel_test::make_options;
using holoscan::channel_test::make_unique_channel_name;
using holoscan::channel_test::read_exact;
using holoscan::channel_test::try_acquire_buffer;
using holoscan::channel_test::verify_host_pattern;
using holoscan::channel_test::wait_for_subscribers;
using holoscan::channel_test::wait_until;
using holoscan::channel_test::write_exact;
using holoscan::channel_test::write_host_pattern;

#ifndef HOLOSCAN_CHANNEL_SCHEMA_INTERPROCESS_CHILD_EXE
#define HOLOSCAN_CHANNEL_SCHEMA_INTERPROCESS_CHILD_EXE ""
#endif

template <typename T>
concept HasTransportDescriptor = requires(T value) { value.descriptor; };

template <typename T>
concept HasBinarySchemaAccessor = requires(const T& value) { value.binary_schema(); };

// Test-only InlinePayloadSubscriber that delivers exactly one pre-built byte payload
// with a caller-supplied SampleMetadata, then reports no pending samples. Allows
// injecting crafted envelopes into the typed receive path without a live Channel backend.
// Representative test: CompatibilitySchemaEnvelopeRejectsNonfixedGenerationBeforeTypedExposure
template <std::size_t N>
class OneShotInlinePayloadSubscriber final
    : public holoscan::channel_internal::InlinePayloadSubscriber {
 public:
  OneShotInlinePayloadSubscriber(const std::array<std::byte, N>& payload,
                                 holoscan::SampleMetadata metadata) noexcept
      : payload_(payload), metadata_(metadata) {}

  [[nodiscard]] holoscan::expected<bool, holoscan::Error> has_samples() const override {
    return pending_;
  }

  // This double has no transport, so it has no event lane and no runtime to build a wait context
  // from. Reporting that honestly makes wait_for_readiness fail with kNotSupported rather than
  // pretend to wait.
  [[nodiscard]] bool event_wake_enabled() const noexcept override { return false; }

  [[nodiscard]] void* wait_context_runtime() noexcept override { return nullptr; }

  [[nodiscard]] bool has_publishers() const noexcept override { return true; }

  [[nodiscard]] holoscan::expected<holoscan::channel_internal::InlinePayloadReceiveResult,
                                   holoscan::Error>
  receive(std::span<std::byte> payload) override {
    if (!pending_) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kNotReady});
    }
    if (payload.size() < payload_.size()) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument});
    }
    for (std::size_t index = 0U; index < payload_.size(); ++index) {
      payload[index] = payload_[index];
    }
    pending_ = false;
    return holoscan::channel_internal::InlinePayloadReceiveResult{.metadata = metadata_,
                                                                  .payload_size = payload_.size()};
  }

  [[nodiscard]] holoscan::expected<std::uint64_t, holoscan::Error> attach_wait_set(
      void* wait_set) override {
    if (wait_set == nullptr) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument});
    }
    return 1U;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> detach_wait_set() noexcept override {
    return {};
  }

 private:
  std::array<std::byte, N> payload_{};
  holoscan::SampleMetadata metadata_{};
  bool pending_{true};
};

class LocalRetirementProbeSubscriber final : public holoscan::channel_internal::WaitSetSubscriber {
 public:
  LocalRetirementProbeSubscriber(std::uint32_t& detach_count, std::uint32_t& destruction_count,
                                 const bool fail_detach) noexcept
      : detach_count_(detach_count),
        destruction_count_(destruction_count),
        fail_detach_(fail_detach) {}

  ~LocalRetirementProbeSubscriber() override { ++destruction_count_; }

  [[nodiscard]] holoscan::expected<std::uint64_t, holoscan::Error> attach_wait_set(
      void* wait_set) override {
    if (wait_set == nullptr) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument});
    }
    return 1U;
  }

  [[nodiscard]] holoscan::expected<bool, holoscan::Error> has_samples() const override {
    return false;
  }

  [[nodiscard]] bool event_wake_enabled() const noexcept override { return false; }

  [[nodiscard]] void* wait_context_runtime() noexcept override { return nullptr; }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> detach_wait_set() noexcept override {
    ++detach_count_;
    if (fail_detach_) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kInterrupted, "injected detach context"});
    }
    return {};
  }

 private:
  std::uint32_t& detach_count_;
  std::uint32_t& destruction_count_;
  bool fail_detach_{};
};

static_assert(holoscan::ChannelInlinePayload<StringPayload>);
static_assert(holoscan::ChannelInlinePayload<TimePayload>);
static_assert(holoscan::ChannelInlinePayload<ImuPayload>);
static_assert(holoscan::ChannelInlinePayload<CustomControlPayload>);
static_assert(StringPayload::inline_capacity == 512U);
static_assert(TimePayload::inline_capacity == 512U);
static_assert(ImuPayload::inline_capacity == 512U);
static_assert(CustomControlPayload::inline_capacity == 512U);

template <typename Root>
concept HasInlineFlatBufferPayload = requires {
  typename holoscan::schema::InlineFlatBufferPayload<Root, 512U>;
  typename holoscan::schema::InlineFlatBufferPayloadFor<Root>;
};

static_assert(HasInlineFlatBufferPayload<holoscan::schema::Imu>);
static_assert(!HasInlineFlatBufferPayload<holoscan::schema::Image>);

// The private transaction consumes ownership before cleanup, normalizes a detach failure to an
// allocation-free code, and still runs the existing endpoint destructor exactly once.
// REQ: CHAN-143
TEST(ChannelSchemaPayloadTest, LocalRetirementConsumesOwnerAndContainsDetachFailure) {
  std::uint32_t detach_count = 0U;
  std::uint32_t destruction_count = 0U;
  auto endpoint =
      std::make_unique<LocalRetirementProbeSubscriber>(detach_count, destruction_count, true);

  const auto closed = holoscan::channel_internal::close_local_subscriber(endpoint);

  ASSERT_FALSE(closed);
  EXPECT_EQ(closed.error().code, holoscan::ErrorCode::kInterrupted);
  EXPECT_TRUE(closed.error().message.empty());
  EXPECT_EQ(endpoint, nullptr);
  EXPECT_EQ(detach_count, 1U);
  EXPECT_EQ(destruction_count, 1U);

  const auto repeated = holoscan::channel_internal::close_local_subscriber(endpoint);
  ASSERT_FALSE(repeated);
  EXPECT_EQ(repeated.error().code, holoscan::ErrorCode::kClosed);
  EXPECT_TRUE(repeated.error().message.empty());
  EXPECT_EQ(detach_count, 1U);
  EXPECT_EQ(destruction_count, 1U);
}

// A subscriber with no real attachment inherits the common successful detach. Local retirement
// creates no replacement wait backend and allocates nothing in the steady success case.
// REQ: CHAN-143
TEST(ChannelSchemaPayloadTest, LocalRetirementWithoutAttachmentAllocatesNothing) {
  class NoAttachmentSubscriber final : public holoscan::channel_internal::WaitSetSubscriber {
   public:
    [[nodiscard]] holoscan::expected<std::uint64_t, holoscan::Error> attach_wait_set(
        void*) override {
      return 1U;
    }

    [[nodiscard]] holoscan::expected<bool, holoscan::Error> has_samples() const override {
      return false;
    }

    [[nodiscard]] bool event_wake_enabled() const noexcept override { return false; }

    [[nodiscard]] void* wait_context_runtime() noexcept override { return nullptr; }
  };

  auto endpoint = std::make_unique<NoAttachmentSubscriber>();
  bool close_succeeded = false;
  std::size_t allocation_count = 0U;
  {
    const holoscan::test::ScopedAllocationCounter allocations;
    const auto closed = holoscan::channel_internal::close_local_subscriber(endpoint);
    close_succeeded = static_cast<bool>(closed);
    allocation_count = allocations.count();
  }
  EXPECT_TRUE(close_succeeded);
  EXPECT_EQ(allocation_count, 0U);
  EXPECT_EQ(endpoint, nullptr);
}

// Builds a ChannelEndpointCatalog with a single holomq:// topic for generated-schema
// endpoint tests. Pass kImport for the publishing side, kExport for the subscriber side.
template <typename T>
[[nodiscard]] holoscan::expected<holoscan::ChannelEndpointCatalog, holoscan::Error>
make_catalog_endpoint(const holoscan::TopicDirection direction,
                      const std::string_view endpoint_name) {
  auto builder = holoscan::ChannelEndpointCatalog::builder("channel-schema-test");
  const std::string uri = "holomq://" + std::string(endpoint_name);
  auto added = direction == holoscan::TopicDirection::kImport
                   ? builder.template import_topic<T>(endpoint_name, uri)
                   : builder.template export_topic<T>(endpoint_name, uri);
  if (!added) {
    return holoscan::make_unexpected(std::move(added).error());
  }
  return std::move(builder).build();
}

// Polls until both the subscriber and publisher refresh successfully in two consecutive
// iterations, ensuring catalog-based endpoints are fully attached before the test emits.
// Required because catalog attach negotiation can lag subscriber creation by several ms.
template <typename Publisher, typename Subscriber>
[[nodiscard]] bool wait_for_catalog_attachment(Publisher& publisher, Subscriber& subscriber,
                                               const std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::uint32_t consecutive_ready_refreshes = 0U;
  while (std::chrono::steady_clock::now() < deadline) {
    auto receiver_refreshed = subscriber.has_samples();
    auto publisher_refreshed = publisher.update_connections();
    if (receiver_refreshed && publisher_refreshed && publisher.has_subscribers()) {
      ++consecutive_ready_refreshes;
      if (consecutive_ready_refreshes == 2U) {
        return true;
      }
    } else {
      consecutive_ready_refreshes = 0U;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return false;
}

// Loops publisher.emit(payload) until the emit reports at least expected_count recipients;
// drains all provided subscribers each attempt to prevent queue-full blocking. Used to
// warm up inline-payload channels before the single asserting emit in each test.
template <typename Publisher, typename Payload, typename... Subscribers>
[[nodiscard]] bool wait_for_schema_payload_recipients(Publisher& publisher,
                                                      const std::uint32_t expected_count,
                                                      const Payload& payload,
                                                      const std::chrono::milliseconds timeout,
                                                      Subscribers&... subscribers) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto emitted = publisher.emit(payload);
    ((void)subscribers.receive(), ...);
    if (emitted && emitted.value() >= expected_count) {
      return true;
    }
    static_cast<void>(publisher.update_connections());
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return false;
}

[[nodiscard]] flatbuffers::FlatBufferBuilder make_string_builder(std::string_view value) {
  flatbuffers::FlatBufferBuilder builder;
  const auto data = builder.CreateString(value.data(), value.size());
  holoscan::schema::FinishStringBuffer(builder, holoscan::schema::CreateString(builder, data));
  return builder;
}

[[nodiscard]] flatbuffers::FlatBufferBuilder make_time_builder(const std::int64_t time_ns) {
  flatbuffers::FlatBufferBuilder builder;
  holoscan::schema::FinishTimeBuffer(builder,
                                     holoscan::schema::CreateTime(builder, time_ns, time_ns + 1));
  return builder;
}

[[nodiscard]] flatbuffers::FlatBufferBuilder make_imu_builder(const std::int64_t time_ns,
                                                              std::string_view frame_id) {
  flatbuffers::FlatBufferBuilder builder;
  const auto stamp = holoscan::schema::CreateTime(builder, time_ns, time_ns + 10);
  const auto frame = builder.CreateString(frame_id.data(), frame_id.size());
  const auto header = holoscan::schema::CreateHeader(builder, stamp, frame);
  const holoscan::schema::Quaternionf orientation{0.0F, 0.0F, 0.0F, 1.0F};
  const holoscan::schema::Vector3f angular_velocity{0.1F, 0.2F, 0.3F};
  const holoscan::schema::Vector3f linear_acceleration{9.8F, 0.0F, -0.1F};
  holoscan::schema::FinishImuBuffer(
      builder,
      holoscan::schema::CreateImu(
          builder, header, &orientation, &angular_velocity, &linear_acceleration));
  return builder;
}

[[nodiscard]] flatbuffers::FlatBufferBuilder make_custom_control_builder(
    const std::uint64_t sequence, const float gain, std::string_view label) {
  flatbuffers::FlatBufferBuilder builder;
  const auto label_offset = builder.CreateString(label.data(), label.size());
  holoscan::tests::channel::FinishCustomControlBuffer(
      builder,
      holoscan::tests::channel::CreateCustomControl(builder, sequence, gain, label_offset));
  return builder;
}

// Asserts schema-family ID, FlatBuffer verifier integrity, and string value for a received
// StringPayload; shared by inline-payload round-trip tests to reduce assertion boilerplate.
void verify_string_payload(const StringPayload& payload, std::string_view expected) {
  EXPECT_EQ(payload.schema_family_id,
            holoscan::schema_identity<holoscan::schema::String>().family_id);
  auto valid = payload.validate();
  ASSERT_TRUE(valid) << holoscan::runtime_error_message(valid.error());
  const holoscan::schema::String* parsed = payload.root();
  ASSERT_NE(parsed, nullptr);
  ASSERT_NE(parsed->data(), nullptr);
  EXPECT_EQ(parsed->data()->string_view(), expected);
}

}  // namespace

// Verifies schema trait default capacities drive the convenience aliases for SDK and
// application-defined FlatBuffer payload wrappers.
TEST(ChannelSchemaPayloadTest, PayloadAliasesUseSchemaTraitDefaultCapacities) {
  EXPECT_TRUE(
      (std::is_same_v<StringPayload,
                      holoscan::schema::InlineFlatBufferPayload<holoscan::schema::String, 512>>));
  EXPECT_TRUE((
      std::is_same_v<DescriptorBackedImagePayload,
                     holoscan::schema::DescriptorFlatBufferPayload<holoscan::schema::Image, 512>>));
  EXPECT_TRUE(
      (std::is_same_v<CustomControlPayload,
                      holoscan::schema::
                          InlineFlatBufferPayload<holoscan::tests::channel::CustomControl, 512>>));
  EXPECT_EQ(holoscan::schema::flatbuffer_schema_max_serialized_size_v<
                holoscan::tests::channel::CustomControl>,
            512U);
}

// The catalog frame budget is capacity plus attribution plus the schema envelope, so a schema
// endpoint publishing at its declared capacity is the boundary case for that arithmetic.
// Attachment bounds are covered elsewhere; this exercises a maximum-capacity data transfer.
// REQ: CHAN-152
TEST(ChannelSchemaPayloadTest, MaximumCapacityOwningSchemaTraversesCatalogEndpoints) {
  using Payload = holoscan::schema::StringT;
  constexpr std::size_t kCapacity = holoscan::generated_schema_max_serialized_size_v<Payload>;

  // Find the largest value the package still encodes within its declared bound.
  std::array<std::byte, kCapacity> scratch{};
  const auto encoded_size = [&scratch](const std::size_t length) -> std::size_t {
    Payload probe;
    probe.data = std::string(length, 'x');
    auto encoded = holoscan::schema_package<Payload>().encode(probe, scratch);
    return encoded ? *encoded : 0U;
  };
  std::size_t maximum_length = 0U;
  for (std::size_t length = kCapacity; length > 0U; --length) {
    if (encoded_size(length) != 0U) {
      maximum_length = length;
      break;
    }
  }
  ASSERT_GT(maximum_length, 0U);
  const std::size_t maximum_encoded = encoded_size(maximum_length);

  // Key setup: the value fills the declared capacity up to FlatBuffer alignment, and one more
  // byte cannot be encoded within that bound.
  EXPECT_LE(maximum_encoded, kCapacity);
  EXPECT_GE(maximum_encoded, kCapacity - 8U);
  EXPECT_EQ(encoded_size(maximum_length + 1U), 0U);

  const std::string endpoint_name = make_unique_channel_name();
  auto import_catalog =
      make_catalog_endpoint<Payload>(holoscan::TopicDirection::kImport, endpoint_name);
  auto export_catalog =
      make_catalog_endpoint<Payload>(holoscan::TopicDirection::kExport, endpoint_name);
  ASSERT_TRUE(import_catalog) << holoscan::runtime_error_message(import_catalog.error());
  ASSERT_TRUE(export_catalog) << holoscan::runtime_error_message(export_catalog.error());
  const auto import_endpoint = import_catalog->endpoint(endpoint_name);
  const auto export_endpoint = export_catalog->endpoint(endpoint_name);
  ASSERT_TRUE(import_endpoint);
  ASSERT_TRUE(export_endpoint);

  auto external_runtime = holoscan::ChannelRuntime::create();
  auto graph_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(external_runtime);
  ASSERT_TRUE(graph_runtime);
  auto subscriber = graph_runtime->create_subscriber<Payload>(*export_endpoint);
  auto publisher = external_runtime->create_publisher<Payload>(*import_endpoint);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_catalog_attachment(*publisher, *subscriber, std::chrono::seconds(2)));

  Payload source;
  source.data = std::string(maximum_length, 'x');
  auto emitted = publisher->emit(source);
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());

  // Key assertion: the maximum-capacity payload survives the complete catalog frame intact.
  auto received = subscriber->receive_sample();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->data.data, source.data);
  EXPECT_EQ(received->metadata.byte_size, maximum_encoded);
  EXPECT_EQ(received->metadata.payload_kind, holoscan::PayloadKind::kSerializedBytes);
}

// REQ: CHAN-078
// REQ: CHAN-143
TEST(ChannelSchemaPayloadTest, OwningGeneratedSchemaCloseIsOrderedAndCarriesNoTypedPayload) {
  const std::string endpoint_name = make_unique_channel_name();
  auto import_catalog = make_catalog_endpoint<holoscan::schema::StringT>(
      holoscan::TopicDirection::kImport, endpoint_name);
  auto export_catalog = make_catalog_endpoint<holoscan::schema::StringT>(
      holoscan::TopicDirection::kExport, endpoint_name);
  ASSERT_TRUE(import_catalog) << holoscan::runtime_error_message(import_catalog.error());
  ASSERT_TRUE(export_catalog) << holoscan::runtime_error_message(export_catalog.error());
  const auto import_endpoint = import_catalog->endpoint(endpoint_name);
  const auto export_endpoint = export_catalog->endpoint(endpoint_name);
  ASSERT_TRUE(import_endpoint);
  ASSERT_TRUE(export_endpoint);

  auto external_runtime = holoscan::ChannelRuntime::create();
  auto graph_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(external_runtime);
  ASSERT_TRUE(graph_runtime);
  auto subscriber = graph_runtime->create_subscriber<holoscan::schema::StringT>(*export_endpoint);
  auto publisher = external_runtime->create_publisher<holoscan::schema::StringT>(*import_endpoint);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_catalog_attachment(*publisher, *subscriber, std::chrono::seconds(2)));

  holoscan::schema::StringT source;
  source.data = "ordered generated-schema value";
  auto emitted = publisher->emit(source);
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  auto value = subscriber->receive();
  ASSERT_TRUE(value) << holoscan::runtime_error_message(value.error());
  EXPECT_EQ(value->data, source.data);

  // Consume the data head before publishing EOS. The queue is deliberately bounded and a
  // sequential test must not fill it with both operations before receiving the first one.
  ASSERT_TRUE(publisher->close());
  // The next ordered operation is EOS. It cannot be decoded into an empty generated value,
  // observed as a zero-byte schema message, or exposed as an application Sample.
  auto end_of_stream = subscriber->receive_sample();
  ASSERT_FALSE(end_of_stream);
  EXPECT_EQ(end_of_stream.error().code, holoscan::ErrorCode::kClosed);
  // This typed endpoint deliberately stops at kClosed. RuntimeChannelBinding converts the same
  // private kEndOfStream operation into RuntimeEventKind::kInputEndOfStream, while the generic
  // selector owns SelectionReason::kEndOfStream; making this facade invoke a selector would
  // collapse the Channel/RunSession architecture boundary.
  ASSERT_TRUE(subscriber->close());
  EXPECT_FALSE(subscriber->has_publishers());
  auto post_close = subscriber->receive();
  ASSERT_FALSE(post_close);
  EXPECT_EQ(post_close.error().code, holoscan::ErrorCode::kClosed);
}

// Verifies a self-contained String FlatBuffer can use the native inline typed channel path without
// adding a schema-specific Channel specialization.
// REQ: CHAN-075
// REQ: CHAN-120
TEST(ChannelSchemaPayloadTest, BlockingPublishesAndReceivesStringFlatBuffer) {
  //   publisher --[kBlocking, depth=8]--> subscriber
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  const auto builder = make_string_builder("hello schema channel");
  auto payload = StringPayload::copy_from_builder(builder);
  ASSERT_TRUE(payload) << holoscan::runtime_error_message(payload.error());
  EXPECT_EQ(payload->endpoint_generation, 0U);

  const auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
  const std::string channel_name = make_unique_channel_name();
  auto publisher = runtime->create_publisher<StringPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<StringPayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(3)));

  auto emitted = publisher->emit(payload.value());
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  ASSERT_EQ(emitted.value(), 1U);

  ASSERT_TRUE(subscriber->has_samples().value());
  auto received = subscriber->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->endpoint_generation,
            holoscan::channel_internal::kCompatibilitySchemaEndpointGeneration);
  verify_string_payload(received.value(), "hello schema channel");
}

// Verifies a peer cannot assign arbitrary restart/rebind meaning to the fixed-generation manual
// schema compatibility facade. The public wrapper receive rejects before exposing typed data.
// REQ: CHAN-075
// REQ: CHAN-120
TEST(ChannelSchemaPayloadTest,
     CompatibilitySchemaEnvelopeRejectsNonfixedGenerationBeforeTypedExposure) {
  const auto builder = make_string_builder("wrong compatibility generation");
  // Key setup: endpoint_generation is one past the fixed compatibility sentinel, so the
  //   subscriber's generation check must fire before any typed payload is exposed.
  auto payload = StringPayload::copy_from_builder(
      builder, holoscan::channel_internal::kCompatibilitySchemaEndpointGeneration + 1U);
  ASSERT_TRUE(payload) << holoscan::runtime_error_message(payload.error());
  const holoscan::SampleMetadata metadata{
      .byte_size = static_cast<std::uint32_t>(sizeof(StringPayload)),
      .payload_kind = holoscan::PayloadKind::kInlineTyped,
  };
  const auto bytes = std::bit_cast<std::array<std::byte, sizeof(StringPayload)>>(payload.value());
  auto subscriber =
      holoscan::channel_internal::InlinePayloadTestAccess::make_subscriber<StringPayload>(
          std::make_unique<OneShotInlinePayloadSubscriber<sizeof(StringPayload)>>(bytes, metadata));

  // Key assertion: generation mismatch surfaces as kProtocolError before typed exposure.
  auto received_value = subscriber.receive();
  ASSERT_FALSE(received_value);
  EXPECT_EQ(received_value.error().code, holoscan::ErrorCode::kProtocolError);
}

// Verifies a small self-contained schema payload can cross a real process boundary through the
// production Channel API. This narrows interprocess failures before exercising BufferHandle import.
TEST(ChannelSchemaPayloadTest, BlockingPublishesStringFlatBufferToChildProcess) {
  const char* const child_exe = HOLOSCAN_CHANNEL_SCHEMA_INTERPROCESS_CHILD_EXE;
  ASSERT_NE(child_exe[0], '\0') << "HOLOSCAN_CHANNEL_SCHEMA_INTERPROCESS_CHILD_EXE not configured";

  const auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
  const std::string channel_name = make_unique_channel_name();

  std::array<int, 2> result_pipe{};
  std::array<int, 2> start_pipe{};
  ASSERT_EQ(::pipe(result_pipe.data()), 0);
  ASSERT_EQ(::pipe(start_pipe.data()), 0);
  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    ::close(result_pipe[0]);
    ::close(start_pipe[1]);
    const std::string result_fd = std::to_string(result_pipe[1]);
    const std::string start_fd = std::to_string(start_pipe[0]);
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_SCHEMA_RESULT_FD", result_fd.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_SCHEMA_START_FD", start_fd.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_SCHEMA_NAME", channel_name.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_SCHEMA_MODE", "string", 1));
    ::execl(child_exe, "channel_schema_interprocess_child", static_cast<char*>(nullptr));
    _exit(127);
  }
  ::close(result_pipe[1]);
  ::close(start_pipe[0]);
  ChildProcessCleanup child_cleanup(child, result_pipe[0], start_pipe[1]);

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  auto publisher = runtime->create_publisher<StringPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());

  const auto builder = make_string_builder("ipc-string");
  auto payload = StringPayload::copy_from_builder(builder);
  ASSERT_TRUE(payload) << holoscan::runtime_error_message(payload.error());
  const StringPayload payload_value = std::move(payload).value();

  constexpr std::uint8_t kStartChild = 1U;
  ASSERT_TRUE(write_exact(child_cleanup.start_fd(), &kStartChild, sizeof(kStartChild)));
  child_cleanup.close_start();

  std::uint8_t ready_code = 0xffU;
  ASSERT_TRUE(read_exact(child_cleanup.result_fd(), &ready_code, sizeof(ready_code)));
  if (ready_code != 1U) {
    int status = 0;
    ASSERT_EQ(child_cleanup.wait(&status), child);
    ASSERT_TRUE(WIFEXITED(status));
    FAIL() << "child failed before subscriber-ready handshake; exit=" << WEXITSTATUS(status)
           << " validation code " << static_cast<unsigned>(ready_code);
  }

  std::string last_emit_error;
  bool child_exited_early = false;
  int child_status = 0;
  std::uint8_t early_child_code = 0xffU;
  const bool delivered = wait_until(
      [&publisher,
       &payload_value,
       &last_emit_error,
       child,
       &child_exited_early,
       &child_status,
       &early_child_code,
       &child_cleanup] {
        static_cast<void>(publisher->update_connections());
        auto published = publisher->emit(payload_value);
        if (published && published.value() >= 1U) {
          return true;
        }
        if (!published && published.error().code != holoscan::ErrorCode::kNotReady) {
          last_emit_error = holoscan::runtime_error_message(published.error());
        }
        const pid_t waited = ::waitpid(child, &child_status, WNOHANG);
        if (waited == child) {
          child_exited_early = true;
          static_cast<void>(
              read_exact(child_cleanup.result_fd(), &early_child_code, sizeof(early_child_code)));
          return true;
        }
        return false;
      },
      std::chrono::seconds(5));
  if (child_exited_early) {
    child_cleanup.close_result();
    ASSERT_TRUE(WIFEXITED(child_status));
    FAIL() << "child exited before parent published; exit=" << WEXITSTATUS(child_status)
           << " validation code " << static_cast<unsigned>(early_child_code);
  }
  ASSERT_TRUE(delivered) << "failed to publish String schema payload to child: " << last_emit_error;

  std::uint8_t child_code = 0xffU;
  ASSERT_TRUE(read_exact(child_cleanup.result_fd(), &child_code, sizeof(child_code)));
  child_cleanup.close_result();

  int status = 0;
  ASSERT_EQ(child_cleanup.wait(&status), child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
  EXPECT_EQ(child_code, 0U) << "child validation stage/error code "
                            << static_cast<unsigned>(child_code);
}

// The child reconstructs the schema catalog after exec(), receives the owning native value, then
// observes an explicit ordered EOS. This covers the generated package and catalog envelope rather
// than the older fixed-capacity compatibility wrapper used by the test above.
// REQ: CHAN-075
// REQ: CHAN-078
TEST(ChannelSchemaPayloadTest, OwningGeneratedSchemaAndEOSCrossExecBoundary) {
  const char* const child_exe = HOLOSCAN_CHANNEL_SCHEMA_INTERPROCESS_CHILD_EXE;
  ASSERT_NE(child_exe[0], '\0') << "HOLOSCAN_CHANNEL_SCHEMA_INTERPROCESS_CHILD_EXE not configured";

  const std::string endpoint_name = make_unique_channel_name();
  auto catalog = make_catalog_endpoint<holoscan::schema::StringT>(holoscan::TopicDirection::kImport,
                                                                  endpoint_name);
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());

  std::array<int, 2> result_pipe{};
  std::array<int, 2> start_pipe{};
  ASSERT_EQ(::pipe(result_pipe.data()), 0);
  ASSERT_EQ(::pipe(start_pipe.data()), 0);
  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    ::close(result_pipe[0]);
    ::close(start_pipe[1]);
    const std::string result_fd = std::to_string(result_pipe[1]);
    const std::string start_fd = std::to_string(start_pipe[0]);
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_SCHEMA_RESULT_FD", result_fd.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_SCHEMA_START_FD", start_fd.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_SCHEMA_NAME", endpoint_name.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_SCHEMA_MODE", "owning_string", 1));
    ::execl(child_exe, "channel_schema_interprocess_child", static_cast<char*>(nullptr));
    _exit(127);
  }
  ::close(result_pipe[1]);
  ::close(start_pipe[0]);
  ChildProcessCleanup child_cleanup(child, result_pipe[0], start_pipe[1]);

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  auto publisher = runtime->create_publisher<holoscan::schema::StringT>(*catalog, endpoint_name);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());

  constexpr std::uint8_t kStartChild = 1U;
  ASSERT_TRUE(write_exact(child_cleanup.start_fd(), &kStartChild, sizeof(kStartChild)));
  child_cleanup.close_start();

  std::uint8_t child_code = 0xffU;
  ASSERT_TRUE(read_exact(child_cleanup.result_fd(), &child_code, sizeof(child_code)));
  ASSERT_EQ(child_code, 1U) << "child failed before subscriber-ready handshake";
  ASSERT_TRUE(wait_for_subscribers(*publisher, std::chrono::seconds(2)));

  holoscan::schema::StringT value;
  value.data = "owning schema across exec";
  auto emitted = publisher->emit_sample(holoscan::Sample<holoscan::schema::StringT>{
      std::move(value),
      holoscan::SampleMetadata{
          .trace_id = 0x1234U,
          .frame_id = 0x5678U,
          .flags = holoscan::SampleFlags::kDegraded,
      },
  });
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  ASSERT_EQ(*emitted, 1U);

  ASSERT_TRUE(read_exact(child_cleanup.result_fd(), &child_code, sizeof(child_code)));
  ASSERT_EQ(child_code, 2U) << "child did not validate the owning schema sample";
  auto closed = publisher->close();
  if (!closed) {
    child_cleanup.close_result();
    int status = 0;
    static_cast<void>(child_cleanup.wait(&status));
    FAIL() << "failed to publish generated-schema EOS: "
           << holoscan::runtime_error_message(closed.error());
  }

  ASSERT_TRUE(read_exact(child_cleanup.result_fd(), &child_code, sizeof(child_code)));
  child_cleanup.close_result();
  int status = 0;
  ASSERT_EQ(child_cleanup.wait(&status), child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
  EXPECT_EQ(child_code, 0U) << "child validation stage/error code "
                            << static_cast<unsigned>(child_code);
}
// Verifies depth-one drop-oldest semantics for self-contained schema payloads keep only the newest
// unread Time FlatBuffer for a slow receiver.
TEST(ChannelSchemaPayloadTest, DropOldestDepthOneKeepsNewestTimeFlatBuffer) {
  //   publisher --[kDropOldest, depth=1]--> subscriber (slow; reads newest)
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  auto options = make_options(holoscan::ChannelQueuePolicy::kDropOldest);
  options.subscriber_queue_depth = 1U;
  const std::string channel_name = make_unique_channel_name();
  auto publisher = runtime->create_publisher<TimePayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<TimePayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(3)));

  for (const std::int64_t time_ns : {100, 200, 300}) {
    const auto builder = make_time_builder(time_ns);
    auto payload = TimePayload::copy_from_builder(builder);
    ASSERT_TRUE(payload) << holoscan::runtime_error_message(payload.error());
    auto emitted = publisher->emit(payload.value());
    ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  }

  auto received = subscriber->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->schema_family_id,
            holoscan::schema_identity<holoscan::schema::Time>().family_id);
  auto valid = received->validate();
  ASSERT_TRUE(valid) << holoscan::runtime_error_message(valid.error());
  const holoscan::schema::Time* parsed = received->root();
  ASSERT_NE(parsed, nullptr);
  EXPECT_EQ(parsed->time_ns(), 300);
  EXPECT_EQ(parsed->acqtime_ns(), 301);
}

// Verifies a blocking self-contained Imu FlatBuffer is delivered to every connected subscriber.
TEST(ChannelSchemaPayloadTest, BlockingFanoutPublishesImuFlatBuffer) {
  //   publisher --[kBlocking, recipients=3]--> subscriber_a
  //                                        --> subscriber_b
  //                                        --> subscriber_c
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  const auto builder = make_imu_builder(123456, "imu_link");
  auto payload = ImuPayload::copy_from_builder(builder);
  ASSERT_TRUE(payload) << holoscan::runtime_error_message(payload.error());

  auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking, 3U);
  const std::string channel_name = make_unique_channel_name();
  auto publisher = runtime->create_publisher<ImuPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber_a = runtime->create_subscriber<ImuPayload>(channel_name, options);
  ASSERT_TRUE(subscriber_a) << holoscan::runtime_error_message(subscriber_a.error());
  auto subscriber_b = runtime->create_subscriber<ImuPayload>(channel_name, options);
  ASSERT_TRUE(subscriber_b) << holoscan::runtime_error_message(subscriber_b.error());
  auto subscriber_c = runtime->create_subscriber<ImuPayload>(channel_name, options);
  ASSERT_TRUE(subscriber_c) << holoscan::runtime_error_message(subscriber_c.error());
  ASSERT_TRUE(wait_for_schema_payload_recipients(publisher.value(),
                                                 3U,
                                                 payload.value(),
                                                 std::chrono::seconds(3),
                                                 subscriber_a.value(),
                                                 subscriber_b.value(),
                                                 subscriber_c.value()));

  auto emitted = publisher->emit(payload.value());
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  ASSERT_EQ(emitted.value(), 3U);

  for (auto* subscriber : {&subscriber_a.value(), &subscriber_b.value(), &subscriber_c.value()}) {
    auto received = subscriber->receive();
    ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
    EXPECT_EQ(received->schema_family_id,
              holoscan::schema_identity<holoscan::schema::Imu>().family_id);
    auto valid = received->validate();
    ASSERT_TRUE(valid) << holoscan::runtime_error_message(valid.error());
    const holoscan::schema::Imu* imu = received->root();
    ASSERT_NE(imu, nullptr);
    ASSERT_NE(imu->header(), nullptr);
    ASSERT_NE(imu->header()->frame_id(), nullptr);
    EXPECT_EQ(imu->header()->frame_id()->string_view(), std::string_view("imu_link"));
    ASSERT_NE(imu->angular_velocity(), nullptr);
    EXPECT_FLOAT_EQ(imu->angular_velocity()->y(), 0.2F);
  }
}
// Verifies application-defined FlatBuffer roots can specialize FlatBufferSchemaBinding and use the
// shared inline payload wrapper without adding any Channel specialization.
// REQ: SCHEMA-006
TEST(ChannelSchemaPayloadTest, CustomSchemaTraitPublishesAndReceivesApplicationFlatBuffer) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  const auto builder = make_custom_control_builder(42U, 1.25F, "app_control");
  auto payload = CustomControlPayload::copy_from_builder(builder);
  ASSERT_TRUE(payload) << holoscan::runtime_error_message(payload.error());
  EXPECT_EQ(
      payload->schema_family_id,
      holoscan::schema::flatbuffer_schema_family_id_v<holoscan::tests::channel::CustomControl>);
  EXPECT_EQ(payload->transport_profile_tag, 0U);

  const auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
  const std::string channel_name = make_unique_channel_name();
  auto publisher = runtime->create_publisher<CustomControlPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<CustomControlPayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(3)));

  auto emitted = publisher->emit(payload.value());
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  ASSERT_EQ(emitted.value(), 1U);

  auto received = subscriber->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  auto valid = received->validate();
  ASSERT_TRUE(valid) << holoscan::runtime_error_message(valid.error());
  const auto* parsed = received->root();
  ASSERT_NE(parsed, nullptr);
  EXPECT_EQ(parsed->sequence(), 42U);
  EXPECT_FLOAT_EQ(parsed->gain(), 1.25F);
  ASSERT_NE(parsed->label(), nullptr);
  EXPECT_EQ(parsed->label()->string_view(), std::string_view("app_control"));
}

// Verifies an author-defined FlatBuffer root's hand-pinned identity is actually enforced. The whole
// value of an author-minted schema family ID is exact-match rejection; SDK roots prove
// this for generator-minted identities, and this proves it for the CHAN-031 author-defined path.
// The wrapper validate() gate accepts the payload against its own pinned identity and rejects a
// foreign family id, a nonzero profile tag, and a retired envelope ABI before typed exposure. REQ:
// CHAN-030 REQ: CHAN-031
TEST(ChannelSchemaPayloadTest, CustomSchemaTraitValidateRejectsIdentityMismatch) {
  const auto builder = make_custom_control_builder(7U, 0.5F, "identity");
  auto payload = CustomControlPayload::copy_from_builder(builder);
  ASSERT_TRUE(payload) << holoscan::runtime_error_message(payload.error());

  // The unmodified author payload validates against the identity pinned in its generated binding.
  const auto clean = payload->validate();
  ASSERT_TRUE(clean) << holoscan::runtime_error_message(clean.error());

  // A foreign (SDK String) schema id is rejected.
  auto wrong_id = payload.value();
  wrong_id.schema_family_id = holoscan::schema_identity<holoscan::schema::String>().family_id;
  const auto id_result = wrong_id.validate();
  ASSERT_FALSE(id_result);
  EXPECT_EQ(id_result.error().code, holoscan::ErrorCode::kCodecError);

  // A nonzero reserved transport profile tag is rejected.
  auto wrong_profile = payload.value();
  ++wrong_profile.transport_profile_tag;
  const auto profile_result = wrong_profile.validate();
  ASSERT_FALSE(profile_result);
  EXPECT_EQ(profile_result.error().code, holoscan::ErrorCode::kCodecError);

  // An unsupported envelope ABI is rejected.
  auto wrong_abi = payload.value();
  wrong_abi.envelope_abi_version = holoscan::kSchemaEnvelopeABIVersion + 1U;
  const auto abi_result = wrong_abi.validate();
  ASSERT_FALSE(abi_result);
  EXPECT_EQ(abi_result.error().code, holoscan::ErrorCode::kCodecError);
}

// Verifies an author-defined FlatBuffer root round-trips across a real process boundary. The parent
// publishes a CustomControl inline payload and an independently exec'd child subscribes, validates
// the hand-pinned identity, and reads back the exact fields. SDK roots have cross-process evidence;
// this extends the self-contained-portability claim to the CHAN-031 author-defined path.
// REQ: CHAN-031
TEST(ChannelSchemaPayloadTest, BlockingPublishesCustomSchemaFlatBufferToChildProcess) {
  const char* const child_exe = HOLOSCAN_CHANNEL_SCHEMA_INTERPROCESS_CHILD_EXE;
  ASSERT_NE(child_exe[0], '\0') << "HOLOSCAN_CHANNEL_SCHEMA_INTERPROCESS_CHILD_EXE not configured";

  const auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
  const std::string channel_name = make_unique_channel_name();

  std::array<int, 2> result_pipe{};
  std::array<int, 2> start_pipe{};
  ASSERT_EQ(::pipe(result_pipe.data()), 0);
  ASSERT_EQ(::pipe(start_pipe.data()), 0);
  const pid_t child = ::fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    ::close(result_pipe[0]);
    ::close(start_pipe[1]);
    const std::string result_fd = std::to_string(result_pipe[1]);
    const std::string start_fd = std::to_string(start_pipe[0]);
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_SCHEMA_RESULT_FD", result_fd.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_SCHEMA_START_FD", start_fd.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_SCHEMA_NAME", channel_name.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_SCHEMA_MODE", "custom", 1));
    ::execl(child_exe, "channel_schema_interprocess_child", static_cast<char*>(nullptr));
    _exit(127);
  }
  ::close(result_pipe[1]);
  ::close(start_pipe[0]);
  ChildProcessCleanup child_cleanup(child, result_pipe[0], start_pipe[1]);

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  auto publisher = runtime->create_publisher<CustomControlPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());

  const auto builder = make_custom_control_builder(42U, 1.25F, "ipc-custom");
  auto payload = CustomControlPayload::copy_from_builder(builder);
  ASSERT_TRUE(payload) << holoscan::runtime_error_message(payload.error());
  const CustomControlPayload payload_value = std::move(payload).value();

  constexpr std::uint8_t kStartChild = 1U;
  ASSERT_TRUE(write_exact(child_cleanup.start_fd(), &kStartChild, sizeof(kStartChild)));
  child_cleanup.close_start();

  std::uint8_t ready_code = 0xffU;
  ASSERT_TRUE(read_exact(child_cleanup.result_fd(), &ready_code, sizeof(ready_code)));
  if (ready_code != 1U) {
    int status = 0;
    ASSERT_EQ(child_cleanup.wait(&status), child);
    ASSERT_TRUE(WIFEXITED(status));
    FAIL() << "child failed before subscriber-ready handshake; exit=" << WEXITSTATUS(status)
           << " validation code " << static_cast<unsigned>(ready_code);
  }

  std::string last_emit_error;
  bool child_exited_early = false;
  int child_status = 0;
  std::uint8_t early_child_code = 0xffU;
  const bool delivered = wait_until(
      [&publisher,
       &payload_value,
       &last_emit_error,
       child,
       &child_exited_early,
       &child_status,
       &early_child_code,
       &child_cleanup] {
        static_cast<void>(publisher->update_connections());
        auto published = publisher->emit(payload_value);
        if (published && published.value() >= 1U) {
          return true;
        }
        if (!published && published.error().code != holoscan::ErrorCode::kNotReady) {
          last_emit_error = holoscan::runtime_error_message(published.error());
        }
        const pid_t waited = ::waitpid(child, &child_status, WNOHANG);
        if (waited == child) {
          child_exited_early = true;
          static_cast<void>(
              read_exact(child_cleanup.result_fd(), &early_child_code, sizeof(early_child_code)));
          return true;
        }
        return false;
      },
      std::chrono::seconds(5));
  if (child_exited_early) {
    child_cleanup.close_result();
    ASSERT_TRUE(WIFEXITED(child_status));
    FAIL() << "child exited before parent published; exit=" << WEXITSTATUS(child_status)
           << " validation code " << static_cast<unsigned>(early_child_code);
  }
  ASSERT_TRUE(delivered) << "failed to publish CustomControl schema payload to child: "
                         << last_emit_error;

  std::uint8_t child_code = 0xffU;
  ASSERT_TRUE(read_exact(child_cleanup.result_fd(), &child_code, sizeof(child_code)));
  child_cleanup.close_result();

  int status = 0;
  ASSERT_EQ(child_cleanup.wait(&status), child);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
  EXPECT_EQ(child_code, 0U) << "child validation stage/error code "
                            << static_cast<unsigned>(child_code);
}

// Verifies untrusted bytes are rejected by the FlatBuffer verifier before entering a typed channel.
TEST(ChannelSchemaPayloadTest, InlinePayloadRejectsInvalidFlatBufferBytes) {
  const std::array<std::byte, 8> invalid_bytes{std::byte{0x01},
                                               std::byte{0x02},
                                               std::byte{0x03},
                                               std::byte{0x04},
                                               std::byte{0x05},
                                               std::byte{0x06},
                                               std::byte{0x07},
                                               std::byte{0x08}};

  auto payload = StringPayload::copy_from(
      std::span<const std::byte>(invalid_bytes.data(), invalid_bytes.size()));

  ASSERT_FALSE(payload);
  EXPECT_EQ(payload.error().code, holoscan::ErrorCode::kCodecError);
}

// The four-byte FlatBuffers identifier is an independent type check performed before typed
// exposure. Structurally valid buffers without the identifier or stamped for another root fail.
// REQ: SCHEMA-007
TEST(ChannelSchemaPayloadTest, InlinePayloadRequiresExactFlatBufferFileIdentifier) {
  const auto build_with_identifier = [](const char* identifier) {
    flatbuffers::FlatBufferBuilder builder;
    const auto data = builder.CreateString("identifier check");
    builder.Finish(holoscan::schema::CreateString(builder, data), identifier);
    return builder;
  };

  const auto missing = build_with_identifier(nullptr);
  auto missing_payload = StringPayload::copy_from_builder(missing);
  ASSERT_FALSE(missing_payload);
  EXPECT_EQ(missing_payload.error().code, holoscan::ErrorCode::kCodecError);

  const auto wrong = build_with_identifier(holoscan::schema::TimeIdentifier());
  auto wrong_payload = StringPayload::copy_from_builder(wrong);
  ASSERT_FALSE(wrong_payload);
  EXPECT_EQ(wrong_payload.error().code, holoscan::ErrorCode::kCodecError);
}

// Adding the required identifier consumes serialized capacity. Exercise the exact SDK boundary so
// a future schema/codegen edit cannot silently move a formerly admitted buffer over the limit.
// REQ: SCHEMA-007
TEST(ChannelSchemaPayloadTest, FileIdentifierGrowthRetainsCheckedInlineCapacityBoundary) {
  std::optional<flatbuffers::FlatBufferBuilder> at_capacity;
  std::optional<flatbuffers::FlatBufferBuilder> over_capacity;
  for (std::size_t length = 0U; length <= StringPayload::inline_capacity; ++length) {
    auto builder = make_string_builder(std::string(length, 'x'));
    if (builder.GetSize() == StringPayload::inline_capacity) {
      at_capacity.emplace(std::move(builder));
    } else if (builder.GetSize() > StringPayload::inline_capacity) {
      over_capacity.emplace(std::move(builder));
      break;
    }
  }
  HOLOSCAN_ASSERT_HAS_VALUE(at_capacity);
  HOLOSCAN_ASSERT_HAS_VALUE(over_capacity);

  auto admitted = StringPayload::copy_from_builder(*at_capacity);
  ASSERT_TRUE(admitted) << holoscan::runtime_error_message(admitted.error());
  EXPECT_EQ(admitted->bytes().size(), StringPayload::inline_capacity);

  auto rejected = StringPayload::copy_from_builder(*over_capacity);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, holoscan::ErrorCode::kInvalidArgument);
}

// Verifies fixed-capacity inline schema wrappers reject payloads that cannot fit in the native
// typed transport bucket.
TEST(ChannelSchemaPayloadTest, InlinePayloadRejectsOversizedFlatBuffer) {
  std::array<std::byte, StringPayload::inline_capacity + 1U> oversized_bytes{};

  auto payload = StringPayload::copy_from_trusted_bytes(
      std::span<const std::byte>(oversized_bytes.data(), oversized_bytes.size()));

  ASSERT_FALSE(payload);
  EXPECT_EQ(payload.error().code, holoscan::ErrorCode::kInvalidArgument);
}

// Verifies the schema family field is part of the wrapper validation contract, even though the
// Channel hot path treats the wrapper as an ordinary trivially-copyable typed payload.
TEST(ChannelSchemaPayloadTest, InlinePayloadValidateRejectsSchemaIdentityMismatch) {
  const auto builder = make_string_builder("wrong schema id");
  auto payload = StringPayload::copy_from_builder(builder);
  ASSERT_TRUE(payload) << holoscan::runtime_error_message(payload.error());

  auto corrupted = payload.value();
  corrupted.schema_family_id = holoscan::schema_identity<holoscan::schema::Time>().family_id;
  auto valid = corrupted.validate();

  ASSERT_FALSE(valid);
  EXPECT_EQ(valid.error().code, holoscan::ErrorCode::kCodecError);
}

// Verifies the generated SDK package separates family identity, codec, capacity, and transport
// profile without retaining BFBS bytes.
// REQ: CHAN-075
// REQ: CHAN-120
// REQ: OPERATOR-049
// REQ: SCHEMA-005
TEST(ChannelSchemaPayloadTest, GeneratedPackageContainsOnlyTheRuntimeContract) {
  constexpr holoscan::SchemaId kExpectedStringFamilyId =
      holoscan::test::schema_id_from_hex("3075a4c189f82740f3f2e6efc31ee4bd");
  const auto& package = holoscan::schema_package<holoscan::schema::String>();

  auto valid = holoscan::validate_schema_package(package);
  ASSERT_TRUE(valid) << holoscan::runtime_error_message(valid.error());
  EXPECT_EQ(package.identity().canonical_name, "holoscan.schema.String");
  EXPECT_EQ(package.family_identity(),
            holoscan::schema_family_identity<holoscan::schema::String>());
  EXPECT_EQ(package.identity().family_id, kExpectedStringFamilyId);
  EXPECT_EQ(package.transport_profile().kind,
            holoscan::SchemaTransportProfileKind::kInlineFlatBuffer);
  static_assert(
      !HasBinarySchemaAccessor<holoscan::SchemaPackageDescriptor<holoscan::schema::String>>);
  static_assert(std::is_same_v<decltype(StringPayload::schema_family_id), holoscan::SchemaId>);
  static_assert(sizeof(holoscan::SchemaEnvelopeHeader) == 40U);
}

// Verifies a complete catalog mismatch is rejected at attach even when a valid local package is
// installed. Compact identity never substitutes for the complete identity tuple, and the terminal
// error distinguishes the expected identity from the local identity.
// REQ: CHAN-075
// REQ: CHAN-120
TEST(ChannelSchemaPayloadTest, GeneratedPackageAttachRequiresExactCompleteIdentity) {
  const auto& package = holoscan::schema_package<holoscan::schema::String>();
  auto expected = holoscan::schema_identity<holoscan::schema::String>();
  expected.canonical_name = "holoscan.schema.OtherString";

  auto attached = holoscan::validate_schema_package_for_attach(expected, package);

  ASSERT_FALSE(attached);
  EXPECT_EQ(attached.error().code, holoscan::ErrorCode::kIncompatible);
  EXPECT_NE(attached.error().message.find("expected={"), std::string::npos);
  EXPECT_NE(attached.error().message.find("; local={"), std::string::npos);
}

// Verifies receive checks ABI, stable family ID, endpoint generation, and generated verifier in
// that order before returning a generated root view.
// REQ: CHAN-075
// REQ: CHAN-120
TEST(ChannelSchemaPayloadTest, SchemaEnvelopeRejectsPeerMismatchBeforeTypedExposure) {
  constexpr std::uint64_t kGeneration = 73U;
  const auto& package = holoscan::schema_package<holoscan::schema::String>();
  const auto builder = make_string_builder("verified envelope");
  auto payload = StringPayload::copy_from_builder(builder, kGeneration);
  ASSERT_TRUE(payload) << holoscan::runtime_error_message(payload.error());

  auto parsed = payload->verified_root(package, kGeneration);
  ASSERT_TRUE(parsed) << holoscan::runtime_error_message(parsed.error());
  ASSERT_NE(parsed.value(), nullptr);
  ASSERT_NE(parsed.value()->data(), nullptr);
  EXPECT_EQ(parsed.value()->data()->string_view(), "verified envelope");

  auto wrong_abi = payload.value();
  wrong_abi.envelope_abi_version = holoscan::kSchemaEnvelopeABIVersion + 1U;
  auto abi_result = wrong_abi.verified_root(package, kGeneration);
  ASSERT_FALSE(abi_result);
  EXPECT_EQ(abi_result.error().code, holoscan::ErrorCode::kProtocolError);

  auto wrong_id = payload.value();
  wrong_id.schema_family_id = holoscan::schema_identity<holoscan::schema::Time>().family_id;
  auto id_result = wrong_id.verified_root(package, kGeneration);
  ASSERT_FALSE(id_result);
  EXPECT_EQ(id_result.error().code, holoscan::ErrorCode::kProtocolError);

  auto wrong_profile = payload.value();
  ++wrong_profile.transport_profile_tag;
  auto profile_result = wrong_profile.verified_root(package, kGeneration);
  ASSERT_FALSE(profile_result);
  EXPECT_EQ(profile_result.error().code, holoscan::ErrorCode::kProtocolError);

  auto wrong_generation = payload.value();
  ++wrong_generation.endpoint_generation;
  auto generation_result = wrong_generation.verified_root(package, kGeneration);
  ASSERT_FALSE(generation_result);
  EXPECT_EQ(generation_result.error().code, holoscan::ErrorCode::kProtocolError);

  auto malformed = payload.value();
  malformed.payload[0] = std::byte{0xFF};
  auto malformed_result = malformed.verified_root(package, kGeneration);
  ASSERT_FALSE(malformed_result);
  EXPECT_EQ(malformed_result.error().code, holoscan::ErrorCode::kProtocolError);
}

// Verifies malformed peer frames traverse the production catalog service and receive staging,
// then fail before the public owning generated-schema facade can expose T or Sample<T>.
// REQ: CHAN-075
// REQ: CHAN-120
TEST(ChannelSchemaPayloadTest, CatalogReceiveRejectsCorruptedSchemaFramesBeforeTypedExposure) {
  using holoscan::channel_internal::CatalogChannelBindingAccess;
  using holoscan::channel_internal::CatalogChannelBindingTestAccess;
  using holoscan::channel_internal::CatalogSchemaFrameCorruption;

  constexpr std::array kCorruptions{
      CatalogSchemaFrameCorruption::kMalformedEnvelope,
      CatalogSchemaFrameCorruption::kSchemaId,
      CatalogSchemaFrameCorruption::kPayloadSize,
      CatalogSchemaFrameCorruption::kEndpointGeneration,
      CatalogSchemaFrameCorruption::kPayloadVerifier,
      CatalogSchemaFrameCorruption::kEnvelopeAbi,
  };
  constexpr std::size_t kPayloadCapacity =
      holoscan::generated_schema_max_serialized_size_v<holoscan::schema::StringT>;

  for (const auto corruption : kCorruptions) {
    SCOPED_TRACE(static_cast<std::uint32_t>(corruption));
    const std::string endpoint_name = make_unique_channel_name();
    auto catalog = make_catalog_endpoint<holoscan::schema::StringT>(
        holoscan::TopicDirection::kExport, endpoint_name);
    ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());
    const auto endpoint = catalog->endpoint(endpoint_name);
    ASSERT_TRUE(endpoint) << holoscan::runtime_error_message(endpoint.error());

    auto publisher_runtime = holoscan::ChannelRuntime::create();
    auto subscriber_runtime = holoscan::ChannelRuntime::create();
    ASSERT_TRUE(publisher_runtime) << holoscan::runtime_error_message(publisher_runtime.error());
    ASSERT_TRUE(subscriber_runtime) << holoscan::runtime_error_message(subscriber_runtime.error());
    auto subscriber = subscriber_runtime->create_subscriber<holoscan::schema::StringT>(*endpoint);
    auto publisher = CatalogChannelBindingAccess::create_payload_publisher(
        *publisher_runtime, *endpoint, kPayloadCapacity);
    ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
    ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
    ASSERT_TRUE(wait_for_catalog_attachment(**publisher, *subscriber, std::chrono::seconds(2)));

    holoscan::schema::StringT source;
    source.data = "must not escape malformed peer frame";
    alignas(std::max_align_t) std::array<std::byte, kPayloadCapacity> encoded{};
    auto encoded_size =
        holoscan::schema_package<holoscan::schema::StringT>().encode(source, encoded);
    ASSERT_TRUE(encoded_size) << holoscan::runtime_error_message(encoded_size.error());
    ASSERT_TRUE(
        CatalogChannelBindingTestAccess::corrupt_next_schema_publication(**publisher, corruption));

    const holoscan::SampleMetadata metadata{
        .byte_size = static_cast<std::uint32_t>(*encoded_size),
        .payload_kind = holoscan::PayloadKind::kSerializedBytes,
    };
    auto emitted =
        (*publisher)->emit(std::span<const std::byte>{encoded}.first(*encoded_size), metadata);
    ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
    ASSERT_EQ(*emitted, 1U);

    auto received = subscriber->receive_sample();
    ASSERT_FALSE(received);
    EXPECT_EQ(received.error().code, holoscan::ErrorCode::kProtocolError);
  }
}

// Runtime may borrow the already envelope-validated serialized payload until the next receive.
// Standalone typed subscribers retain their ordinary copy-owning behavior.
// REQ: CHAN-075
// REQ: CHAN-120
TEST(ChannelSchemaPayloadTest, CatalogRuntimeBindingBorrowsStagedSchemaPayloadWithoutCopy) {
  using holoscan::channel_internal::CatalogChannelBindingAccess;

  constexpr std::size_t kPayloadCapacity =
      holoscan::generated_schema_max_serialized_size_v<holoscan::schema::StringT>;
  const std::string endpoint_name = make_unique_channel_name();
  auto catalog = make_catalog_endpoint<holoscan::schema::StringT>(holoscan::TopicDirection::kImport,
                                                                  endpoint_name);
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());
  const auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint) << holoscan::runtime_error_message(endpoint.error());

  auto publisher_runtime = holoscan::ChannelRuntime::create();
  auto subscriber_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(publisher_runtime) << holoscan::runtime_error_message(publisher_runtime.error());
  ASSERT_TRUE(subscriber_runtime) << holoscan::runtime_error_message(subscriber_runtime.error());
  auto publisher = publisher_runtime->create_publisher<holoscan::schema::StringT>(*endpoint);
  auto subscriber = CatalogChannelBindingAccess::create_payload_subscriber(
      *subscriber_runtime, *endpoint, kPayloadCapacity);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_catalog_attachment(*publisher, **subscriber, std::chrono::seconds(2)));

  holoscan::schema::StringT source;
  source.data = "borrowed schema staging";
  auto emitted = publisher->emit(source);
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());

  std::array<std::byte, kPayloadCapacity> destination;
  destination.fill(std::byte{0xA5U});
  holoscan::channel_internal::CatalogReceivedOperation operation;
  ASSERT_TRUE(wait_until(
      [&] {
        auto received =
            CatalogChannelBindingAccess::receive_next(**subscriber, destination, true, true);
        if (!received) {
          return false;
        }
        operation = *received;
        return operation.kind == holoscan::channel_internal::CatalogReceivedOperationKind::kData;
      },
      std::chrono::seconds(2)));
  ASSERT_FALSE(operation.schema_payload.empty());
  EXPECT_TRUE(std::all_of(destination.begin(), destination.end(), [](const std::byte value) {
    return value == std::byte{0xA5U};
  }));
  auto materialized =
      holoscan::schema_package<holoscan::schema::StringT>().materialize(operation.schema_payload);
  ASSERT_TRUE(materialized) << holoscan::runtime_error_message(materialized.error());
  EXPECT_EQ(materialized->data, source.data);
}

// The Runtime-facing virtual subscriber surface preserves the catalog's validated schema staging
// until its receive guard retires, without copying through caller scratch.
// REQ: CHAN-075
TEST(ChannelSchemaPayloadTest, CatalogSubscriberBorrowsValidatedSchemaPayload) {
  using holoscan::channel_internal::CatalogChannelBindingAccess;

  // Key setup: publish one generated schema through a catalog-backed inline endpoint.
  constexpr std::size_t kPayloadCapacity =
      holoscan::generated_schema_max_serialized_size_v<holoscan::schema::StringT>;
  const std::string endpoint_name = make_unique_channel_name();
  auto catalog = make_catalog_endpoint<holoscan::schema::StringT>(holoscan::TopicDirection::kImport,
                                                                  endpoint_name);
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());
  const auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint) << holoscan::runtime_error_message(endpoint.error());

  auto publisher_runtime = holoscan::ChannelRuntime::create();
  auto subscriber_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(publisher_runtime) << holoscan::runtime_error_message(publisher_runtime.error());
  ASSERT_TRUE(subscriber_runtime) << holoscan::runtime_error_message(subscriber_runtime.error());
  auto publisher = publisher_runtime->create_publisher<holoscan::schema::StringT>(*endpoint);
  auto subscriber = CatalogChannelBindingAccess::create_payload_subscriber(
      *subscriber_runtime, *endpoint, kPayloadCapacity);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_catalog_attachment(*publisher, **subscriber, std::chrono::seconds(2)));

  holoscan::schema::StringT source;
  source.data = "virtual borrowed schema staging";
  auto emitted = publisher->emit(source);
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());

  std::optional<holoscan::channel_internal::InlinePayloadReceiveGuard> borrowed;
  ASSERT_TRUE(wait_until(
      [&] {
        auto received = (*subscriber)->receive_borrowed();
        if (!received) {
          return false;
        }
        borrowed.emplace(std::move(*received));
        return true;
      },
      std::chrono::seconds(2)));

  HOLOSCAN_ASSERT_HAS_VALUE(borrowed);

  // Key assertion: the virtual guard keeps validated staging stable and excludes another borrow.
  auto overlapping = (*subscriber)->receive_borrowed();
  ASSERT_FALSE(overlapping);
  EXPECT_EQ(overlapping.error().code, holoscan::ErrorCode::kFailure);

  auto materialized =
      holoscan::schema_package<holoscan::schema::StringT>().materialize(borrowed->payload());
  ASSERT_TRUE(materialized) << holoscan::runtime_error_message(materialized.error());
  EXPECT_EQ(materialized->data, source.data);
}

// Verifies local wrapper construction keeps codec/authoring errors distinct from malformed peer
// protocol errors and does not accept an unsupported wrapper ABI.
// REQ: CHAN-075
// REQ: CHAN-120
TEST(ChannelSchemaPayloadTest, SchemaEnvelopeLocalValidationRejectsUnsupportedAbi) {
  const auto builder = make_string_builder("local schema wrapper");
  auto payload = StringPayload::copy_from_builder(builder, 11U);
  ASSERT_TRUE(payload) << holoscan::runtime_error_message(payload.error());

  payload->envelope_abi_version = holoscan::kSchemaEnvelopeABIVersion + 1U;
  auto valid = payload->validate();

  ASSERT_FALSE(valid);
  EXPECT_EQ(valid.error().code, holoscan::ErrorCode::kCodecError);
}

// Verifies the source-private Channel seam binds package and generation once, then delegates the
// exact per-message ABI/id/generation/verifier protocol without backend vocabulary.
// REQ: CHAN-120
TEST(ChannelSchemaPayloadTest, GeneratedSchemaAdapterBindsExactPackageAndGeneration) {
  constexpr std::uint64_t kGeneration = 91U;
  const auto identity = holoscan::schema_identity<holoscan::schema::String>();
  const auto& package = holoscan::schema_package<holoscan::schema::String>();
  auto adapter =
      holoscan::channel_internal::GeneratedSchemaAdapter<holoscan::schema::String>::create(
          identity, kGeneration, package);
  ASSERT_TRUE(adapter) << holoscan::runtime_error_message(adapter.error());

  const auto builder = make_string_builder("adapter verified");
  auto payload = StringPayload::copy_from_builder(builder, kGeneration);
  ASSERT_TRUE(payload) << holoscan::runtime_error_message(payload.error());
  auto parsed = adapter->decode(payload->envelope_header(), payload->bytes());
  ASSERT_TRUE(parsed) << holoscan::runtime_error_message(parsed.error());
  EXPECT_EQ(parsed.value()->data()->string_view(), "adapter verified");

  auto stale = payload->envelope_header();
  ++stale.endpoint_generation;
  auto rejected = adapter->decode(stale, payload->bytes());
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, holoscan::ErrorCode::kProtocolError);
}

// Verifies warmed-up inline schema payload transport does not allocate on the steady-state
// emit/receive path. Validation is intentionally outside the measured scope.
TEST(ChannelSchemaPayloadTest, InlinePayloadHotPathDoesNotAllocateAfterWarmup) {
  //   publisher --[kBlocking, depth=8]--> subscriber (32 measured emit/receive cycles)
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  const auto builder = make_string_builder("allocation-free schema channel");
  auto payload = StringPayload::copy_from_builder(builder);
  ASSERT_TRUE(payload) << holoscan::runtime_error_message(payload.error());

  const auto options = make_options(holoscan::ChannelQueuePolicy::kBlocking);
  const std::string channel_name = make_unique_channel_name();
  auto publisher = runtime->create_publisher<StringPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<StringPayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_schema_payload_recipients(
      publisher.value(), 1U, payload.value(), std::chrono::seconds(3), subscriber.value()));

  // Key setup: one warmup emit/receive primes all internal caches before the measured loop.
  auto warmup_emit = publisher->emit(payload.value());
  ASSERT_TRUE(warmup_emit) << holoscan::runtime_error_message(warmup_emit.error());
  auto warmup_receive = subscriber->receive();
  ASSERT_TRUE(warmup_receive) << holoscan::runtime_error_message(warmup_receive.error());

  bool all_cycles_succeeded = true;
  std::size_t allocation_count = 0U;
  {
    const holoscan::test::ScopedAllocationCounter allocations;
    for (std::size_t i = 0U; i < 32U; ++i) {
      auto emitted = publisher->emit(payload.value());
      all_cycles_succeeded = all_cycles_succeeded && emitted.has_value() && emitted.value() == 1U;
      auto received = subscriber->receive();
      all_cycles_succeeded = all_cycles_succeeded && received.has_value() &&
                             received->schema_family_id ==
                                 holoscan::schema_identity<holoscan::schema::String>().family_id &&
                             received->bytes().size() == payload->bytes().size();
    }
    allocation_count = allocations.count();
  }

  EXPECT_TRUE(all_cycles_succeeded);
  // Key assertion: zero allocations across 32 inline emit/receive cycles on the hot path.
  EXPECT_EQ(allocation_count, 0U);
}
