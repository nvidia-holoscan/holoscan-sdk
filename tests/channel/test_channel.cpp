/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/pubsub_options.hpp>
#include <holoscan/core/errors.hpp>

#include "../../src/channel/channel_common.hpp"
#include "../../src/channel/channel_inline_payload_test_access.hpp"
#include "../../src/channel/channel_tensor_test_access.hpp"
#include "../common/allocation_counter.hpp"
#include "channel_test_support.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>

#include <sys/wait.h>
#include <unistd.h>

using namespace holoscan::channel_test;  // NOLINT(build/namespaces)

namespace {

// Value 3 is representable by the enum's closed numeric range but intentionally names no HoloMQ
// error, so these tests can exercise fail-closed translation without invalid enum state.
constexpr auto kUnknownHoloMqError =
    static_cast<holomq::ErrorCode>(3);  // NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange)

// General-purpose inline POD for most inline-payload and fanout tests.
// ChannelInlinePayload<ControlPayload> is true; representative test:
//   BlockingPublishesAndReceivesControlPayload.
struct ControlPayload {
  std::uint64_t sequence{};
  std::int32_t value{};
  double timestamp{};
};

// Deleted-default struct; verifies ChannelInlinePayload<DeletedDefaultPayload> is false.
// Used in static_assert concept checks (REQ: CHANNEL-API-002).
struct DeletedDefaultPayload {
  DeletedDefaultPayload() = delete;
  explicit DeletedDefaultPayload(const std::uint64_t initial_value) : value(initial_value) {}

  std::uint64_t value;
};

// Non-noexcept default constructor; verifies ChannelInlinePayload<ThrowingDefaultPayload>
// is false. The Channel contract requires noexcept default construction.
struct ThrowingDefaultPayload {
  ThrowingDefaultPayload() noexcept(false) {}

  std::uint64_t value{};
};

// Test stub for InlinePayloadPublisher that returns a caller-injected phase result.
// Used by PrivatePublicationPhaseMapsToPublicResult to exercise each phase outcome in isolation.
class PhasePublisher final : public holoscan::channel_internal::InlinePayloadPublisher {
 public:
  explicit PhasePublisher(holoscan::channel_internal::InlinePublicationResult result)
      : result_(std::move(result)) {}

  [[nodiscard]] holoscan::channel_internal::InlinePublicationResult emit_phase(
      std::span<const std::byte>, const holoscan::SampleMetadata&) override {
    return result_;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> update_connections() noexcept override {
    return {};
  }

  [[nodiscard]] bool has_subscribers() const noexcept override { return false; }

 private:
  holoscan::channel_internal::InlinePublicationResult result_;
};

// REQ: CHANNEL-API-002
static_assert(holoscan::ChannelInlinePayload<ControlPayload>);
static_assert(!holoscan::PublishedPayload<ControlPayload>);
static_assert(std::is_standard_layout_v<DeletedDefaultPayload>);
static_assert(std::is_trivially_copyable_v<DeletedDefaultPayload>);
static_assert(!holoscan::ChannelInlinePayload<DeletedDefaultPayload>);
static_assert(std::is_standard_layout_v<ThrowingDefaultPayload>);
static_assert(std::is_trivially_copyable_v<ThrowingDefaultPayload>);
static_assert(!holoscan::ChannelInlinePayload<ThrowingDefaultPayload>);

// The adapter owns an explicit fail-closed mapping. Numeric coincidence between the two enums is
// deliberately irrelevant so a HoloMQ addition cannot silently become a new Core outcome.
// REQ: CHAN-066
// REQ: CHAN-076
// REQ: CHAN-110
// REQ: RUNTIME-016
TEST(ChannelErrorMappingTest, MapsEveryNamedHoloMQCodeAndFailsClosedForUnknownValues) {
  constexpr std::array mappings{
      std::pair{holomq::ErrorCode::kSuccess, holoscan::ErrorCode::kSuccess},
      std::pair{holomq::ErrorCode::kFailure, holoscan::ErrorCode::kFailure},
      std::pair{holomq::ErrorCode::kReceiveError, holoscan::ErrorCode::kReceiveError},
      std::pair{holomq::ErrorCode::kInvalidArgument, holoscan::ErrorCode::kInvalidArgument},
      std::pair{holomq::ErrorCode::kNotFound, holoscan::ErrorCode::kNotFound},
      std::pair{holomq::ErrorCode::kNotImplemented, holoscan::ErrorCode::kNotImplemented},
      std::pair{holomq::ErrorCode::kBackpressured, holoscan::ErrorCode::kBackpressured},
      std::pair{holomq::ErrorCode::kNotReady, holoscan::ErrorCode::kNotReady},
      std::pair{holomq::ErrorCode::kStaleEndpoint, holoscan::ErrorCode::kStaleEndpoint},
      std::pair{holomq::ErrorCode::kClosed, holoscan::ErrorCode::kClosed},
      std::pair{holomq::ErrorCode::kTimeout, holoscan::ErrorCode::kTimeout},
      std::pair{holomq::ErrorCode::kResourceExhausted, holoscan::ErrorCode::kResourceExhausted},
      std::pair{holomq::ErrorCode::kPermissionDenied, holoscan::ErrorCode::kPermissionDenied},
      std::pair{holomq::ErrorCode::kInterrupted, holoscan::ErrorCode::kInterrupted},
      std::pair{holomq::ErrorCode::kProtocolError, holoscan::ErrorCode::kProtocolError},
      std::pair{holomq::ErrorCode::kIncompatible, holoscan::ErrorCode::kIncompatible},
      std::pair{holomq::ErrorCode::kIncompleteFanout, holoscan::ErrorCode::kNotReady},
  };

  for (const auto& [source, expected] : mappings) {
    EXPECT_EQ(holoscan::channel_internal::to_holoscan_error_code(source), expected);
  }

  EXPECT_EQ(holoscan::channel_internal::to_holoscan_error_code(kUnknownHoloMqError),
            holoscan::ErrorCode::kFailure);
}

// REQ: CHAN-066
// REQ: CHAN-076
TEST(ChannelErrorMappingTest, RejectsSuccessFailureCarrierAndUnknownValueWithStableInvariants) {
  auto invalid_success = holoscan::channel_internal::from_holomq_error(
      holomq::Error{holomq::ErrorCode::kSuccess, "invalid carrier"});
  EXPECT_EQ(invalid_success.code, holoscan::ErrorCode::kFailure);
  EXPECT_NE(invalid_success.message.find("HOLOMQ_ERROR_SUCCESS_IN_FAILURE"), std::string::npos);

  auto unknown = holoscan::channel_internal::from_holomq_error(
      holomq::Error{kUnknownHoloMqError, "future code"});
  EXPECT_EQ(unknown.code, holoscan::ErrorCode::kFailure);
  EXPECT_NE(unknown.message.find("HOLOMQ_ERROR_UNMAPPED"), std::string::npos);
}

// Retained publisher history can expose a submitted sample to a late subscriber, so graph-owned
// pressure handling must not classify a backend rejection as safely retryable in that mode.
// REQ: CHAN-076
TEST(ChannelPressureTest, GraphRetryBoundaryRequiresZeroPublisherHistory) {
  using holoscan::channel_internal::PublisherPressureOptions;

  // Key setup: select the graph retry policy and compare zero-history with retained-history
  // services.
  constexpr auto pressure = PublisherPressureOptions::reject_when_full();

  // Key assertion: only a zero-history service admits the graph retry boundary.
  EXPECT_TRUE(pressure.enables_graph_retry_boundary(0U));
  EXPECT_FALSE(pressure.enables_graph_retry_boundary(1U));
  EXPECT_FALSE(PublisherPressureOptions::follow_service().enables_graph_retry_boundary(0U));
}

// Once send has been invoked, only the exact frozen fanout proves publication success. A known
// strict subset is reported separately from an unknown transport failure, while malformed counts
// fail as argument/protocol violations.
// REQ: CHAN-067
// REQ: CHAN-076
// REQ: CHAN-109
TEST(ChannelErrorMappingTest, PublicationResultRequiresExactPositiveFanout) {
  struct RecipientCountCase {
    int actual_count;
    std::uint32_t expected_count;
    std::optional<holoscan::ErrorCode> error;
  };
  constexpr std::array count_cases{
      RecipientCountCase{
          .actual_count = -1, .expected_count = 3U, .error = holoscan::ErrorCode::kProtocolError},
      RecipientCountCase{.actual_count = 0,
                         .expected_count = 3U,
                         .error = holoscan::ErrorCode::kPublicationPartial},
      RecipientCountCase{.actual_count = 2,
                         .expected_count = 3U,
                         .error = holoscan::ErrorCode::kPublicationPartial},
      RecipientCountCase{.actual_count = 3, .expected_count = 3U, .error = std::nullopt},
      RecipientCountCase{
          .actual_count = 4, .expected_count = 3U, .error = holoscan::ErrorCode::kProtocolError},
      RecipientCountCase{
          .actual_count = 0, .expected_count = 0U, .error = holoscan::ErrorCode::kInvalidArgument},
  };
  for (const auto& test_case : count_cases) {
    auto mapped = holoscan::channel_internal::from_holomq_publication_result(
        holomq::expected<int, holomq::Error>{test_case.actual_count}, test_case.expected_count);
    EXPECT_EQ(mapped.has_value(), !test_case.error.has_value());
    if (!test_case.error) {
      EXPECT_EQ(mapped.value(), test_case.expected_count);
    } else {
      EXPECT_EQ(mapped.error().code, *test_case.error);
    }
  }

  auto partial = holoscan::channel_internal::from_holomq_publication_result(
      holomq::expected<int, holomq::Error>{2}, 3U);
  ASSERT_FALSE(partial);
  EXPECT_EQ(partial.error().message, "PUBLICATION_PARTIAL delivered=2 expected=3");

  constexpr std::array raw_error_codes{
      holomq::ErrorCode::kSuccess,
      holomq::ErrorCode::kFailure,
      holomq::ErrorCode::kReceiveError,
      holomq::ErrorCode::kInvalidArgument,
      holomq::ErrorCode::kNotFound,
      holomq::ErrorCode::kNotImplemented,
      holomq::ErrorCode::kBackpressured,
      holomq::ErrorCode::kNotReady,
      holomq::ErrorCode::kStaleEndpoint,
      holomq::ErrorCode::kClosed,
      holomq::ErrorCode::kTimeout,
      holomq::ErrorCode::kResourceExhausted,
      holomq::ErrorCode::kPermissionDenied,
      holomq::ErrorCode::kInterrupted,
      holomq::ErrorCode::kProtocolError,
      holomq::ErrorCode::kIncompatible,
      kUnknownHoloMqError,
  };
  for (const holomq::ErrorCode code : raw_error_codes) {
    auto mapped = holoscan::channel_internal::from_holomq_publication_result(
        holomq::make_unexpected(holomq::Error{code, "injected post-submit failure"}), 1U);
    ASSERT_FALSE(mapped);
    EXPECT_EQ(mapped.error().code, holoscan::ErrorCode::kPublicationIndeterminate);
    if (code == holomq::ErrorCode::kSuccess) {
      EXPECT_NE(mapped.error().message.find("HOLOMQ_ERROR_SUCCESS_IN_FAILURE"), std::string::npos);
    }
  }

  auto future_error = holoscan::channel_internal::from_holomq_publication_result(
      holomq::make_unexpected(holomq::Error{kUnknownHoloMqError, "injected future error"}), 1U);
  ASSERT_FALSE(future_error);
  EXPECT_NE(future_error.error().message.find("HOLOMQ_ERROR_UNMAPPED"), std::string::npos);

  auto incomplete_fanout = holoscan::channel_internal::from_holomq_publication_result(
      holomq::make_unexpected(holomq::Error{holomq::ErrorCode::kIncompleteFanout}), 3U);
  ASSERT_FALSE(incomplete_fanout);
  EXPECT_EQ(incomplete_fanout.error().code, holoscan::ErrorCode::kNotReady);
}

// The Runtime-only reject-on-pressure mode preserves exact zero-recipient results as retryable.
// A send error without an exact count remains indeterminate for multiple recipients because
// another receiver may already have observed the sample.
// REQ: CHAN-067
// REQ: CHAN-076
// REQ: CHAN-101
TEST(ChannelErrorMappingTest, GraphPressurePreservesProvableRetryOutcomes) {
  auto retryable = holoscan::channel_internal::from_holomq_publication_result(
      holomq::make_unexpected(holomq::Error{holomq::ErrorCode::kBackpressured}), 1U, true);
  ASSERT_FALSE(retryable);
  EXPECT_EQ(retryable.error().code, holoscan::ErrorCode::kBackpressured);

  auto zero_recipient_fanout = holoscan::channel_internal::from_holomq_publication_result(
      holomq::expected<int, holomq::Error>{0}, 2U, true);
  ASSERT_FALSE(zero_recipient_fanout);
  EXPECT_EQ(zero_recipient_fanout.error().code, holoscan::ErrorCode::kBackpressured);

  auto indeterminate_fanout = holoscan::channel_internal::from_holomq_publication_result(
      holomq::make_unexpected(holomq::Error{holomq::ErrorCode::kBackpressured}), 2U, true);
  ASSERT_FALSE(indeterminate_fanout);
  EXPECT_EQ(indeterminate_fanout.error().code, holoscan::ErrorCode::kPublicationIndeterminate);

  auto ordinary_publisher = holoscan::channel_internal::from_holomq_publication_result(
      holomq::make_unexpected(holomq::Error{holomq::ErrorCode::kBackpressured}), 1U, false);
  ASSERT_FALSE(ordinary_publisher);
  EXPECT_EQ(ordinary_publisher.error().code, holoscan::ErrorCode::kPublicationIndeterminate);
}

// A pressured fanout arm is backend evidence, not a Core policy input. The adapter preserves that
// opaque identity beneath its stable post-submit prefix without parsing it to infer visibility.
// REQ: CHAN-076
// REQ: CHAN-101
TEST(ChannelErrorMappingTest, GraphPressurePreservesOffendingRecipientDiagnostic) {
  constexpr std::string_view kRecipientEvidence{
      "HOLOMQ_BACKPRESSURED recipient=00112233445566778899aabbccddeeff"};

  // Key setup: inject the diagnostic carried by a terminal HoloMQ fanout-pressure failure.
  auto mapped = holoscan::channel_internal::from_holomq_publication_result(
      holomq::make_unexpected(
          holomq::Error{holomq::ErrorCode::kBackpressured, std::string{kRecipientEvidence}}),
      2U,
      true);

  // Key assertion: Core retains the exact offending-arm identity while conservatively classifying
  // a multi-recipient post-submit failure as indeterminate.
  ASSERT_FALSE(mapped);
  EXPECT_EQ(mapped.error().code, holoscan::ErrorCode::kPublicationIndeterminate);
  EXPECT_EQ(mapped.error().message,
            "HOLOMQ_PUBLICATION_POST_SUBMIT: " + std::string{kRecipientEvidence});
}

template <typename Runtime>
concept HasChannelCreateOutput = requires(Runtime& runtime) {
  runtime.template create_output<ControlPayload>(std::string_view{});
};

template <typename Runtime>
concept HasChannelCreateInput = requires(Runtime& runtime) {
  runtime.template create_input<ControlPayload>(std::string_view{});
};

template <typename Runtime>
concept HasStringTensorPublisherFactory = requires(Runtime& runtime) {
  runtime.template create_publisher<holoscan::Tensor>(std::string_view{});
};

template <typename Publisher>
concept HasBareTensorEmit =
    requires(Publisher& publisher, holoscan::Tensor tensor) { publisher.emit(std::move(tensor)); };

template <typename Publisher>
concept HasTensorMemoryOptions =
    requires(const Publisher& publisher) { publisher.tensor_memory_options(); };

template <typename Publisher, typename T>
concept HasSampleEmit = requires(Publisher& publisher, holoscan::Sample<T> sample) {
  publisher.emit_sample(std::move(sample));
};

template <typename Subscriber>
concept HasSampleReceive = requires(Subscriber& subscriber) { subscriber.receive_sample(); };

// REQ: CHANNEL-API-001
static_assert(!HasChannelCreateOutput<holoscan::ChannelRuntime>);
static_assert(!HasChannelCreateInput<holoscan::ChannelRuntime>);
static_assert(!HasStringTensorPublisherFactory<holoscan::ChannelRuntime>);
static_assert(!HasBareTensorEmit<holoscan::ChannelPublisher<holoscan::Tensor>>);
static_assert(!HasTensorMemoryOptions<holoscan::ChannelPublisher<holoscan::Tensor>>);

// REQ: CHANNEL-API-002
static_assert(!HasSampleEmit<holoscan::ChannelPublisher<ControlPayload>, ControlPayload>);
static_assert(!HasSampleReceive<holoscan::ChannelSubscriber<ControlPayload>>);
static_assert(HasSampleEmit<holoscan::ChannelPublisher<std::uint32_t>, std::uint32_t>);
static_assert(HasSampleReceive<holoscan::ChannelSubscriber<std::uint32_t>>);

// REQ: CHANNEL-API-003
static_assert(std::is_move_constructible_v<holoscan::ManualChannelBinding>);
static_assert(!std::is_move_assignable_v<holoscan::ManualChannelBinding>);
static_assert(!std::is_copy_constructible_v<holoscan::ManualChannelBinding>);
static_assert(std::is_move_constructible_v<holoscan::ManualTensorPublisher>);
static_assert(!std::is_move_assignable_v<holoscan::ManualTensorPublisher>);
static_assert(!std::is_copy_constructible_v<holoscan::ManualTensorPublisher>);

#ifndef HOLOSCAN_CHANNEL_INTERPROCESS_CHILD_EXE
#define HOLOSCAN_CHANNEL_INTERPROCESS_CHILD_EXE ""
#endif

using holoscan::channel_test::ChildProcessCleanup;
using holoscan::channel_test::read_exact;
using holoscan::channel_test::write_exact;

// Drains all pending ControlPayload samples from a subscriber during warmup polling.
// Called from wait_for_control_payload_recipients to prevent subscriber queue overflow.
template <typename Subscriber>
void drain_control_payloads(Subscriber& subscriber) {
  while (true) {
    auto received = subscriber.receive();
    if (!received) {
      return;
    }
  }
}

// Polls by emitting warmup ControlPayload samples until all expected recipients are reached.
// Each iteration drains every subscriber to prevent queue backpressure during connect wait.
// Used in fanout tests to ensure all subscribers are connected before the key emit.
template <typename Publisher, typename... Subscribers>
[[nodiscard]] bool wait_for_control_payload_recipients(Publisher& publisher,
                                                       const std::uint32_t expected_recipient_count,
                                                       const std::chrono::milliseconds timeout,
                                                       Subscribers&... subscribers) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::uint64_t sequence = 0U;
  while (std::chrono::steady_clock::now() < deadline) {
    auto emitted =
        publisher.emit(ControlPayload{.sequence = ++sequence, .value = -1, .timestamp = -1.0});
    (drain_control_payloads(subscribers), ...);
    if (emitted && emitted.value() >= expected_recipient_count) {
      return true;
    }
    static_cast<void>(publisher.update_connections());
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
  return false;
}

}  // namespace

// Verifies the descriptor stays a fixed-size POD control-plane record suitable for typed
// transport without dynamic serialization.
// REQ: CHAN-077
TEST(ChannelDescriptorTest, DescriptorIsFixedSizeAndTriviallyCopyable) {
  EXPECT_TRUE(std::is_standard_layout_v<holoscan::ChannelSampleDescriptor>);
  EXPECT_TRUE(std::is_trivially_copyable_v<holoscan::ChannelSampleDescriptor>);
  EXPECT_EQ(holoscan::kChannelSampleDescriptorAbiVersion, 2U);
  EXPECT_EQ(sizeof(holoscan::ChannelSampleDescriptor), 72U);
  EXPECT_EQ(alignof(holoscan::ChannelSampleDescriptor), alignof(std::uint64_t));

  const holoscan::ChannelSampleDescriptor descriptor = make_descriptor(7U);
  EXPECT_TRUE(descriptor.has_supported_abi());
  EXPECT_TRUE(static_cast<bool>(descriptor));
  EXPECT_EQ(descriptor.sequence, 7U);
  EXPECT_NE(descriptor.schema_id.value.front(), descriptor.schema_id.value[8U]);

  auto retired_descriptor = descriptor;
  retired_descriptor.abi_version = 1U;
  EXPECT_FALSE(retired_descriptor.has_supported_abi());
}

// Verifies the production-direction envelope/payload split can be derived losslessly from today's
// fixed-size descriptor while exposing only public metadata to Sample<T> users.
TEST(ChannelDescriptorTest, DescriptorMapsToSampleEnvelopeAndPayloadDescriptor) {
  const holoscan::ChannelSampleDescriptor descriptor = make_descriptor(11U);

  auto maybe_envelope = holoscan::sample_envelope_from_descriptor(descriptor);
  ASSERT_TRUE(maybe_envelope) << holoscan::runtime_error_message(maybe_envelope.error());
  const holoscan::SampleEnvelope& envelope = maybe_envelope.value();
  EXPECT_EQ(envelope.metadata.sequence, descriptor.sequence);
  EXPECT_EQ(envelope.metadata.byte_size, descriptor.byte_size);
  EXPECT_EQ(envelope.metadata.payload_kind, holoscan::PayloadKind::kDescriptorBacked);
  EXPECT_TRUE(envelope.metadata.valid());

  EXPECT_EQ(envelope.payload.payload_kind, descriptor.payload_kind);
  EXPECT_EQ(envelope.payload.flags, descriptor.flags);
  EXPECT_EQ(envelope.payload.schema_id, descriptor.schema_id);
  EXPECT_EQ(envelope.payload.sequence, descriptor.sequence);
  EXPECT_EQ(envelope.payload.byte_size, descriptor.byte_size);
  EXPECT_TRUE(envelope.payload.is_local_buffer());
  EXPECT_EQ(envelope.payload.local.process_id, descriptor.process_id);
  EXPECT_EQ(envelope.payload.local.token, descriptor.sidecar_token);
  EXPECT_EQ(envelope.payload.local.generation, descriptor.generation);
  EXPECT_EQ(envelope.payload.local.expected_recipient_count, descriptor.expected_recipient_count);

  const holoscan::ChannelSampleDescriptor roundtrip =
      holoscan::channel_descriptor_from_payload_descriptor(envelope.payload);
  EXPECT_EQ(roundtrip.magic, descriptor.magic);
  EXPECT_EQ(roundtrip.abi_version, descriptor.abi_version);
  EXPECT_EQ(roundtrip.payload_kind, descriptor.payload_kind);
  EXPECT_EQ(roundtrip.flags, descriptor.flags);
  EXPECT_EQ(roundtrip.expected_recipient_count, descriptor.expected_recipient_count);
  EXPECT_EQ(roundtrip.process_id, descriptor.process_id);
  EXPECT_EQ(roundtrip.sidecar_token, descriptor.sidecar_token);
  EXPECT_EQ(roundtrip.generation, descriptor.generation);
  EXPECT_EQ(roundtrip.schema_id, descriptor.schema_id);
  EXPECT_EQ(roundtrip.sequence, descriptor.sequence);
  EXPECT_EQ(roundtrip.byte_size, descriptor.byte_size);

  const holoscan::PayloadDescriptor buffer_payload{.payload_kind =
                                                       holoscan::ChannelPayloadKind::kBufferHandle};
  EXPECT_FALSE(holoscan::channel_descriptor_from_payload_descriptor(buffer_payload));
}

// Verifies private Channel discriminators are mapped exhaustively rather than integer-cast.
TEST(ChannelDescriptorTest, UnknownPayloadKindDoesNotEnterCoreMetadata) {
  auto descriptor = make_descriptor(12U);
  // This intentionally constructs an invalid discriminator to verify the checked mapping boundary.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  descriptor.payload_kind = static_cast<holoscan::ChannelPayloadKind>(0xffffU);

  auto mapped = holoscan::sample_envelope_from_descriptor(descriptor);
  ASSERT_FALSE(mapped);
  EXPECT_EQ(mapped.error().code, holoscan::ErrorCode::kInvalidArgument);
}

// Descriptor endpoints transport references and therefore reject the inline discriminator before
// a metadata envelope without corresponding inline bytes can cross the endpoint boundary.
// REQ: CHANNEL-API-004
// REQ: CHAN-067
// Topology (raw HoloMQ publisher --> ChannelDescriptorSubscriber):
//   raw_publisher --[kInlinePayload discriminator]--> subscriber
// Key setup: raw HoloMQ publisher bypasses the Channel facade to inject kInlinePayload.
TEST(ChannelDescriptorTest, DescriptorEndpointsRejectInlinePayloadKind) {
  auto mapped = holoscan::channel_internal::descriptor_core_payload_kind(
      holoscan::ChannelPayloadKind::kInlinePayload);
  ASSERT_FALSE(mapped);
  EXPECT_EQ(mapped.error().code, holoscan::ErrorCode::kInvalidArgument);

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string publication_channel_name = make_unique_channel_name();
  auto publisher = runtime->create_descriptor_publisher(publication_channel_name);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());

  const holoscan::PayloadDescriptor inline_payload{
      .payload_kind = holoscan::ChannelPayloadKind::kInlinePayload,
  };
  auto published = publisher->publish(inline_payload);
  ASSERT_FALSE(published);
  EXPECT_EQ(published.error().code, holoscan::ErrorCode::kInvalidArgument);

  auto no_peer =
      publisher->publish(holoscan::payload_descriptor_from_descriptor(make_descriptor(1U)));
  ASSERT_FALSE(no_peer);
  EXPECT_EQ(no_peer.error().code, holoscan::ErrorCode::kNotReady);

  const std::string receipt_channel_name = make_unique_channel_name();
  const holoscan::ChannelOptions channel_options{};
  holomq::Context* const raw_context =
      holoscan::channel_internal::TensorPublisherTestAccess::transport_context(runtime.value());
  ASSERT_NE(raw_context, nullptr);
  const holomq::PubSubServiceOptions raw_service_options{
      .queue_policy = holomq::QueuePolicy::kBlocking,
      .max_subscribers = channel_options.expected_recipient_count,
      .max_subscriber_queue_depth = channel_options.subscriber_queue_depth,
      .max_subscriber_borrowed_samples = channel_options.max_subscriber_borrowed_samples,
      .publisher_history_depth = channel_options.publisher_history_depth,
      .enable_event_wake = false,
  };
  const holomq::PublisherPubSubOptions raw_publisher_options{
      .max_loaned_samples = channel_options.max_loaned_samples,
      .enable_sample_release_callbacks = false,
      .replacement_attribution_required = channel_options.replacement_attribution_required,
  };
  auto raw_publisher = raw_context->publisher_builder(receipt_channel_name)
                           .service_options(raw_service_options)
                           .publisher_options(raw_publisher_options)
                           .payload_type<holoscan::PayloadDescriptor>()
                           .create();
  ASSERT_TRUE(raw_publisher);

  auto subscriber = runtime->create_descriptor_subscriber(receipt_channel_name, channel_options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(raw_publisher.value(), std::chrono::seconds(2)));

  auto payload_message = raw_publisher->loan();
  ASSERT_TRUE(payload_message);
  payload_message->payload() = inline_payload;
  auto payload_sent = raw_publisher->send(std::move(payload_message).value());
  ASSERT_TRUE(payload_sent);
  EXPECT_EQ(payload_sent.value(), 1);

  // Key assertion: both receive_payload_descriptor and receive_envelope return kInvalidArgument.
  auto received_payload = subscriber->receive_payload_descriptor();
  ASSERT_FALSE(received_payload);
  EXPECT_EQ(received_payload.error().code, holoscan::ErrorCode::kInvalidArgument);

  auto envelope_message = raw_publisher->loan();
  ASSERT_TRUE(envelope_message);
  envelope_message->payload() = inline_payload;
  auto envelope_sent = raw_publisher->send(std::move(envelope_message).value());
  ASSERT_TRUE(envelope_sent);
  EXPECT_EQ(envelope_sent.value(), 1);

  auto received_envelope = subscriber->receive_envelope();
  ASSERT_FALSE(received_envelope);
  EXPECT_EQ(received_envelope.error().code, holoscan::ErrorCode::kInvalidArgument);
}

TEST(ChannelDescriptorTest, EnvelopeAdapterRejectsOpaqueBufferWithoutCarrier) {
  holoscan::ChannelSampleDescriptor descriptor{};
  descriptor.magic = holoscan::kChannelSampleDescriptorMagic;
  descriptor.abi_version = holoscan::kChannelSampleDescriptorAbiVersion;
  descriptor.payload_kind = holoscan::ChannelPayloadKind::kBufferHandle;
  descriptor.byte_size = 64U;

  auto envelope = holoscan::sample_envelope_from_descriptor(descriptor);
  ASSERT_FALSE(envelope);
  EXPECT_EQ(envelope.error().code, holoscan::ErrorCode::kProtocolError);
}

// Verifies the production payload descriptor stores same-process and HoloMQ buffer references as
// discriminator-selected union arms instead of carrying both reference forms in every sample.
// REQ: CHAN-077
TEST(ChannelDescriptorTest, PayloadDescriptorUsesUnionStorageForPayloadReferences) {
  EXPECT_TRUE(std::is_standard_layout_v<holoscan::PayloadDescriptor>);
  EXPECT_TRUE(std::is_trivially_copyable_v<holoscan::PayloadDescriptor>);
  EXPECT_EQ(sizeof(holoscan::PayloadDescriptor), 304U);
  EXPECT_EQ(offsetof(holoscan::PayloadDescriptor, local),
            offsetof(holoscan::PayloadDescriptor, buffer));
  EXPECT_LT(sizeof(holoscan::PayloadDescriptor),
            offsetof(holoscan::PayloadDescriptor, local) + sizeof(holoscan::LocalPayloadReference) +
                sizeof(holoscan::BufferHandlePayloadReference));
}

// Verifies descriptor endpoints can be used through the production-direction payload/envelope
// helpers while preserving the same-process descriptor adapter as a local-only compatibility path.
// Topology (blocking, default depth):
//   publisher --[PayloadDescriptor + SampleEnvelope + BufferHandle]--> subscriber
TEST(ChannelRuntimeTest, DescriptorEndpointsPublishPayloadDescriptorAndEnvelope) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  auto publisher = runtime->create_descriptor_publisher(channel_name);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_descriptor_subscriber(channel_name);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  const holoscan::PayloadDescriptor payload =
      holoscan::payload_descriptor_from_descriptor(make_descriptor(12U));
  auto payload_published = publisher->publish(payload);
  ASSERT_TRUE(payload_published) << holoscan::runtime_error_message(payload_published.error());
  EXPECT_EQ(payload_published.value(), 1U);

  auto received_payload = subscriber->receive_payload_descriptor();
  ASSERT_TRUE(received_payload) << holoscan::runtime_error_message(received_payload.error());
  EXPECT_EQ(received_payload->local.token, payload.local.token);
  EXPECT_EQ(received_payload->local.generation, payload.local.generation);
  EXPECT_EQ(received_payload->sequence, payload.sequence);
  EXPECT_EQ(received_payload->byte_size, payload.byte_size);

  auto maybe_envelope = holoscan::sample_envelope_from_descriptor(make_descriptor(13U));
  ASSERT_TRUE(maybe_envelope) << holoscan::runtime_error_message(maybe_envelope.error());
  holoscan::SampleEnvelope envelope = std::move(maybe_envelope).value();
  envelope.metadata.capture_timestamp_ns = 9001;
  auto envelope_published = publisher->publish(envelope);
  ASSERT_TRUE(envelope_published) << holoscan::runtime_error_message(envelope_published.error());
  EXPECT_EQ(envelope_published.value(), 1U);

  auto received_envelope = subscriber->receive_envelope();
  ASSERT_TRUE(received_envelope) << holoscan::runtime_error_message(received_envelope.error());
  EXPECT_EQ(received_envelope->payload.local.token, envelope.payload.local.token);
  EXPECT_EQ(received_envelope->metadata.sequence, envelope.payload.sequence);
  EXPECT_EQ(received_envelope->metadata.byte_size, envelope.payload.byte_size);
  // Key assertion: capture_timestamp_ns is not promoted from payload to core metadata.
  EXPECT_EQ(received_envelope->metadata.capture_timestamp_ns, 0);

  const holoscan::PayloadDescriptor buffer_handle_payload{
      .payload_kind = holoscan::ChannelPayloadKind::kBufferHandle,
      .sequence = 14U,
      .byte_size = 128U,
      .buffer = holoscan::BufferHandlePayloadReference{}};
  // Calling this out-of-line HoloMQ value operation keeps Core's public HoloMQ dependency covered
  // by the ordinary Channel target graph.
  EXPECT_FALSE(buffer_handle_payload.is_buffer_handle());
  auto buffer_handle_published = publisher->publish(buffer_handle_payload);
  ASSERT_TRUE(buffer_handle_published)
      << holoscan::runtime_error_message(buffer_handle_published.error());
  EXPECT_EQ(buffer_handle_published.value(), 1U);

  auto legacy_descriptor = subscriber->receive();
  ASSERT_FALSE(legacy_descriptor);
  EXPECT_EQ(legacy_descriptor.error().code, holoscan::ErrorCode::kNotImplemented);
}

// Verifies the public runtime is created without a backend-shaped option bag.
// REQ: CHAN-072
TEST(ChannelRuntimeTest, CreatesMinimalOptionFreeRuntime) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
}

// Verifies invalid subscriber queue depth is rejected during channel setup.
TEST(ChannelRuntimeTest, RejectsZeroSubscriberQueueDepth) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  holoscan::ChannelOptions options{};
  options.subscriber_queue_depth = 0U;

  auto publisher = runtime->create_descriptor_publisher(make_unique_channel_name(), options);
  ASSERT_FALSE(publisher);
  EXPECT_EQ(publisher.error().code, holoscan::ErrorCode::kInvalidArgument);
}

// Verifies locally inconsistent queue options are rejected before creating transport endpoints.
TEST(ChannelRuntimeTest, RejectsRequestedQueueDepthAboveExplicitMaximum) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  holoscan::ChannelOptions options{};
  options.max_subscriber_queue_depth = 1U;
  options.subscriber_queue_depth = 2U;

  auto publisher = runtime->create_descriptor_publisher(make_unique_channel_name(), options);
  ASSERT_FALSE(publisher);
  EXPECT_EQ(publisher.error().code, holoscan::ErrorCode::kInvalidArgument);
}

// Verifies expected recipient count is bounded by the descriptor claim-mask width.
TEST(ChannelRuntimeTest, RejectsTooManyExpectedRecipients) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  holoscan::ChannelOptions options{};
  options.expected_recipient_count = 65U;

  auto publisher = runtime->create_descriptor_publisher(make_unique_channel_name(), options);
  ASSERT_FALSE(publisher);
  EXPECT_EQ(publisher.error().code, holoscan::ErrorCode::kInvalidArgument);
}

// Verifies Tensor channel fanout cannot exceed MemoryBuffer release-stream tracking capacity.
TEST(ChannelRuntimeTest, RejectsTensorFanoutBeyondMemoryBufferCapacity) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  holoscan::ChannelOptions options{};
  options.expected_recipient_count = holoscan::MemoryBuffer::kDefaultMaxFanoutReleaseStreams + 1U;

  auto publisher = holoscan::channel_internal::TensorPublisherTestAccess::create(
      runtime.value(), make_unique_channel_name(), options);
  ASSERT_FALSE(publisher);
  EXPECT_EQ(publisher.error().code, holoscan::ErrorCode::kInvalidArgument);
}

// Verifies the legacy descriptor adapter still round-trips same-process local descriptors.
// Topology (blocking, default depth):
//   publisher --[ChannelSampleDescriptor]--> subscriber
TEST(ChannelRuntimeTest, BlockingPublishesAndReceivesDescriptor) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  auto publisher = runtime->create_descriptor_publisher(channel_name);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_descriptor_subscriber(channel_name);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  auto published = publisher->publish(make_descriptor(42U));
  ASSERT_TRUE(published) << holoscan::runtime_error_message(published.error());
  EXPECT_EQ(published.value(), 1U);

  auto received = subscriber->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->payload_kind, holoscan::ChannelPayloadKind::kLocalBufferDescriptor);
  EXPECT_EQ(received->process_id, static_cast<std::uint64_t>(::getpid()));
  EXPECT_EQ(received->expected_recipient_count, 1U);
  EXPECT_EQ(received->sidecar_token, 0x102AU);
  EXPECT_EQ(received->generation, 0x202AU);
  EXPECT_EQ(received->schema_id, make_descriptor_schema_id(42U));
  EXPECT_EQ(received->sequence, 42U);
  EXPECT_EQ(received->byte_size, 4096U + 42U);
}

// Verifies descriptor readiness checks do not consume the pending descriptor.
TEST(ChannelRuntimeTest, SubscriberReadinessIsNonConsuming) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  auto publisher = runtime->create_descriptor_publisher(channel_name);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_descriptor_subscriber(channel_name);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  ASSERT_TRUE(publish_descriptor(publisher.value(), 5U));
  EXPECT_TRUE(subscriber->has_samples().value());
  EXPECT_TRUE(subscriber->has_samples().value());

  auto received = subscriber->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->sequence, 5U);
  EXPECT_FALSE(subscriber->has_samples().value());

  auto empty = subscriber->receive();
  ASSERT_FALSE(empty);
  EXPECT_EQ(empty.error().code, holoscan::ErrorCode::kNotReady);
}

// Verifies drop-oldest descriptor delivery retains the newest configured number of samples.
// Topology (kDropOldest, depth=3):
//   publisher --[kDropOldest, depth=3]--> subscriber
// Key assertion: after 5 publishes, only sequences 3-5 remain in the queue.
TEST(ChannelRuntimeTest, DropOldestKeepsNewestQueuedDescriptors) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  holoscan::ChannelOptions options{};
  options.queue_policy = holoscan::ChannelQueuePolicy::kDropOldest;
  options.subscriber_queue_depth = 3U;

  auto publisher = runtime->create_descriptor_publisher(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_descriptor_subscriber(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  for (std::uint64_t sequence = 1U; sequence <= 5U; ++sequence) {
    ASSERT_TRUE(publish_descriptor(publisher.value(), sequence));
  }

  for (std::uint64_t expected_sequence = 3U; expected_sequence <= 5U; ++expected_sequence) {
    auto received = subscriber->receive();
    ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
    EXPECT_EQ(received->sequence, expected_sequence);
    EXPECT_EQ(received->sidecar_token, 0x1000U + expected_sequence);
  }
  EXPECT_FALSE(subscriber->has_samples().value());
}

// Verifies inline POD payload channels use typed transport; empty receive returns kNotReady.
// Topology (blocking, default depth):
//   publisher --[ControlPayload]--> subscriber
TEST(ChannelInlinePayloadTest, BlockingPublishesAndReceivesControlPayload) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  auto publisher = runtime->create_publisher<ControlPayload>(channel_name);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<ControlPayload>(channel_name);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  const ControlPayload expected{.sequence = 17U, .value = -4, .timestamp = 123.5};
  auto emitted = publisher->emit(expected);
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  EXPECT_EQ(emitted.value(), 1U);
  EXPECT_TRUE(subscriber->has_samples().value());

  auto received = subscriber->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->sequence, expected.sequence);
  EXPECT_EQ(received->value, expected.value);
  EXPECT_DOUBLE_EQ(received->timestamp, expected.timestamp);
  EXPECT_FALSE(subscriber->has_samples().value());

  auto empty = subscriber->receive();
  ASSERT_FALSE(empty);
  EXPECT_EQ(empty.error().code, holoscan::ErrorCode::kNotReady);
}

// Blocking inline endpoints retain their original single FIFO lane for both data and EOS. Empty
// payload bytes remain valid data and are never used as the terminal discriminator.
// REQ: CHAN-076
// Topology (blocking, depth=2):
//   publisher --[ControlPayload, then EOS]--> subscriber
TEST(ChannelInlinePayloadTest, BlockingCloseRemainsOnTheDataLane) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  holoscan::ChannelOptions options{};
  options.subscriber_queue_depth = 2U;

  auto publisher = runtime->create_publisher<ControlPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<ControlPayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(*publisher, std::chrono::seconds(2)));

  const ControlPayload expected{.sequence = 5U, .value = 17, .timestamp = 2.5};
  ASSERT_TRUE(publisher->emit(expected));
  ASSERT_TRUE(publisher->close());

  auto received = subscriber->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->sequence, expected.sequence);
  auto end_of_stream = subscriber->receive();
  ASSERT_FALSE(end_of_stream);
  EXPECT_EQ(end_of_stream.error().code, holoscan::ErrorCode::kClosed);
}

// kDropOldest inline endpoints protect EOS on a private capacity-one lane while preserving per-arm
// data-before-terminal observation. Empty payload bytes remain valid data and are never used as
// the terminal discriminator.
// REQ: CHAN-110
// Topology (kDropOldest, depth=1):
//   publisher --[ControlPayload, then EOS]--> subscriber
// Key assertion: after EOS, has_samples is false and has_publishers is false.
TEST(ChannelInlinePayloadTest, ExplicitClosePublishesOrderedEndOfStream) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  holoscan::ChannelOptions options{};
  options.queue_policy = holoscan::ChannelQueuePolicy::kDropOldest;
  options.subscriber_queue_depth = 1U;

  auto publisher = runtime->create_publisher<ControlPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<ControlPayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  const ControlPayload expected{.sequence = 7U, .value = 11, .timestamp = 4.5};
  ASSERT_TRUE(publisher->emit(expected));
  ASSERT_TRUE(publisher->close());

  auto received = subscriber->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->sequence, expected.sequence);
  auto end_of_stream = subscriber->receive();
  ASSERT_FALSE(end_of_stream);
  EXPECT_EQ(end_of_stream.error().code, holoscan::ErrorCode::kClosed);
  auto pending = subscriber->has_samples();
  ASSERT_TRUE(pending);
  EXPECT_FALSE(*pending);
  EXPECT_FALSE(subscriber->has_publishers());

  auto duplicate_close = publisher->close();
  ASSERT_FALSE(duplicate_close);
  EXPECT_EQ(duplicate_close.error().code, holoscan::ErrorCode::kClosed);
}

// A shared fanout service must preserve the same explicit ordered EOS semantics on every arm.
// REQ: CHAN-078
// REQ: CHAN-110
// Topology (kDropOldest, depth=1, fanout=2):
//   publisher --[ControlPayload, then EOS]--> subscriber_a
//                                         --> subscriber_b
TEST(ChannelInlinePayloadTest, ExplicitClosePublishesOrderedEndOfStreamToEveryFanoutArm) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();
  holoscan::ChannelOptions options{};
  options.expected_recipient_count = 2U;
  options.queue_policy = holoscan::ChannelQueuePolicy::kDropOldest;
  options.subscriber_queue_depth = 1U;

  // Key setup: attach both recipients to one shared service before publishing data followed by
  // the explicit close marker.
  auto publisher = runtime->create_publisher<ControlPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber_a = runtime->create_subscriber<ControlPayload>(channel_name, options);
  ASSERT_TRUE(subscriber_a) << holoscan::runtime_error_message(subscriber_a.error());
  auto subscriber_b = runtime->create_subscriber<ControlPayload>(channel_name, options);
  ASSERT_TRUE(subscriber_b) << holoscan::runtime_error_message(subscriber_b.error());
  ASSERT_TRUE(wait_for_control_payload_recipients(
      publisher.value(), 2U, std::chrono::seconds(2), subscriber_a.value(), subscriber_b.value()));

  const ControlPayload expected{.sequence = 41U, .value = 73, .timestamp = 9.5};
  ASSERT_TRUE(publisher->emit(expected));
  ASSERT_TRUE(publisher->close());

  // Key assertion: each arm observes the data before the same terminal state, with no payload
  // manufactured for EOS.
  for (auto* subscriber : {&subscriber_a.value(), &subscriber_b.value()}) {
    auto received = subscriber->receive();
    ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
    EXPECT_EQ(received->sequence, expected.sequence);
    EXPECT_EQ(received->value, expected.value);
    auto end_of_stream = subscriber->receive();
    ASSERT_FALSE(end_of_stream);
    EXPECT_EQ(end_of_stream.error().code, holoscan::ErrorCode::kClosed);
    auto pending = subscriber->has_samples();
    ASSERT_TRUE(pending);
    EXPECT_FALSE(*pending);
  }
}

// A recipient may legitimately miss an earlier partial fanout publication. The protected terminal
// lane drains each recipient's own data lane first instead of requiring a global contiguous
// sequence, so the recipient that accepted data observes it before EOS while the later recipient
// observes EOS directly.
// REQ: CHAN-110
TEST(ChannelInlinePayloadTest, OrderedTerminalDrainsEachFanoutArmIndependently) {
  constexpr std::size_t kBucketSize = 32U;
  using Bucket = holoscan::channel_internal::InlinePayloadBucket<kBucketSize>;
  static_assert(sizeof(ControlPayload) <= kBucketSize);

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  holomq::Context* const context =
      holoscan::channel_internal::TensorPublisherTestAccess::transport_context(runtime.value());
  ASSERT_NE(context, nullptr);
  const std::string channel_name = make_unique_channel_name();
  holoscan::ChannelOptions options{};
  options.expected_recipient_count = 2U;
  options.queue_policy = holoscan::ChannelQueuePolicy::kDropOldest;
  options.subscriber_queue_depth = 1U;

  auto publisher_options = holoscan::channel_internal::to_holomq_publisher_options(options);
  publisher_options.enable_sample_release_callbacks = false;
  publisher_options.expected_recipient_count = 0U;
  auto data_publisher =
      context->publisher_builder(channel_name)
          .service_options(holoscan::channel_internal::to_holomq_service_options(options))
          .publisher_options(publisher_options)
          .payload_type<Bucket>()
          .create();
  ASSERT_TRUE(data_publisher);
  auto terminal_publisher =
      context
          ->publisher_builder(holoscan::channel_internal::ordered_terminal_service_id(channel_name))
          .service_options(holoscan::channel_internal::ordered_terminal_service_options(options))
          .publisher_options(
              holoscan::channel_internal::ordered_terminal_publisher_options(options))
          .payload_type<Bucket>()
          .create();
  ASSERT_TRUE(terminal_publisher);

  auto subscriber_a = runtime->create_subscriber<ControlPayload>(channel_name, options);
  ASSERT_TRUE(subscriber_a) << holoscan::runtime_error_message(subscriber_a.error());
  ASSERT_TRUE(wait_for_subscribers(*data_publisher, std::chrono::seconds(2)));

  const ControlPayload expected{.sequence = 1U, .value = 37, .timestamp = 8.5};
  auto data = data_publisher->loan();
  ASSERT_TRUE(data);
  data->payload().operation = holoscan::channel_internal::InlinePayloadOperation::kData;
  data->payload().metadata = holoscan::SampleMetadata{
      .byte_size = sizeof(ControlPayload),
      .payload_kind = holoscan::PayloadKind::kInlineTyped,
  };
  data->payload().payload_size = sizeof(ControlPayload);
  std::memcpy(data->payload().bytes.data(), &expected, sizeof(expected));
  auto data_sent = data_publisher->send(std::move(*data));
  ASSERT_TRUE(data_sent);
  ASSERT_EQ(*data_sent, 1);

  auto subscriber_b = runtime->create_subscriber<ControlPayload>(channel_name, options);
  ASSERT_TRUE(subscriber_b) << holoscan::runtime_error_message(subscriber_b.error());
  const auto terminal_deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  bool both_terminal_arms_connected = false;
  while (std::chrono::steady_clock::now() < terminal_deadline) {
    auto refreshed_a = subscriber_a->has_samples();
    auto refreshed_b = subscriber_b->has_samples();
    auto refreshed_publisher = terminal_publisher->update_connections();
    ASSERT_TRUE(refreshed_a) << holoscan::runtime_error_message(refreshed_a.error());
    ASSERT_TRUE(refreshed_b) << holoscan::runtime_error_message(refreshed_b.error());
    ASSERT_TRUE(refreshed_publisher);
    if (subscriber_a->has_publishers() && subscriber_b->has_publishers() &&
        terminal_publisher->has_subscribers()) {
      both_terminal_arms_connected = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  ASSERT_TRUE(both_terminal_arms_connected);

  auto terminal = terminal_publisher->loan();
  ASSERT_TRUE(terminal);
  terminal->payload().operation = holoscan::channel_internal::InlinePayloadOperation::kEndOfStream;
  terminal->payload().metadata = holoscan::SampleMetadata{
      .payload_kind = holoscan::PayloadKind::kInlineTyped,
  };
  auto terminal_sent = terminal_publisher->send(std::move(*terminal));
  ASSERT_TRUE(terminal_sent);
  ASSERT_EQ(*terminal_sent, 2);

  auto received = subscriber_a->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->sequence, expected.sequence);
  EXPECT_EQ(received->value, expected.value);
  auto end_of_stream_a = subscriber_a->receive();
  ASSERT_FALSE(end_of_stream_a);
  EXPECT_EQ(end_of_stream_a.error().code, holoscan::ErrorCode::kClosed);

  auto end_of_stream_b = subscriber_b->receive();
  ASSERT_FALSE(end_of_stream_b);
  EXPECT_EQ(end_of_stream_b.error().code, holoscan::ErrorCode::kClosed);
}

// The private terminal identity is activated only for kDropOldest at this low-level physical
// binding seam. Blocking bindings retain the ordinary 255-byte transport-name capacity and one
// wait-set arm; kDropOldest accepts the longest physical data binding whose derived terminal name
// fits and rejects longer or reserved physical names before constructing the public data service.
// REQ: CHAN-110
// REQ: CHAN-132
TEST(ChannelInlinePayloadTest, ProtectedTerminalIdentityIsDropOldestOnlyAndFailsClosed) {
  constexpr std::string_view kSuffix{".hsdk-terminal-v1"};
  constexpr std::size_t kTransportNameCapacity = 255U;

  {
    auto runtime = holoscan::ChannelRuntime::create();
    ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
    const std::string blocking_id(kTransportNameCapacity, 'b');
    const holoscan::ChannelOptions blocking{};
    auto publisher = holoscan::channel_internal::InlinePayloadTestAccess::create_publisher(
        *runtime, blocking_id, sizeof(ControlPayload), blocking);
    ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
    auto subscriber = holoscan::channel_internal::InlinePayloadTestAccess::create_subscriber(
        *runtime, blocking_id, sizeof(ControlPayload), blocking);
    ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
    auto wait_set = holoscan::channel_internal::InlinePayloadTestAccess::create_wait_set(*runtime);
    ASSERT_TRUE(wait_set) << holoscan::runtime_error_message(wait_set.error());
    ASSERT_TRUE((*wait_set)->attach(**subscriber));
    EXPECT_FALSE(holoscan::channel_internal::protected_terminal_wait_handle(**subscriber));
  }

  const std::string longest_data_id(kTransportNameCapacity - kSuffix.size(), 'a');
  {
    auto runtime = holoscan::ChannelRuntime::create();
    ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
    holoscan::ChannelOptions drop_oldest{};
    drop_oldest.queue_policy = holoscan::ChannelQueuePolicy::kDropOldest;
    auto publisher = holoscan::channel_internal::InlinePayloadTestAccess::create_publisher(
        *runtime, longest_data_id, sizeof(ControlPayload), drop_oldest);
    ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
    auto subscriber = holoscan::channel_internal::InlinePayloadTestAccess::create_subscriber(
        *runtime, longest_data_id, sizeof(ControlPayload), drop_oldest);
    ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
    auto wait_set = holoscan::channel_internal::InlinePayloadTestAccess::create_wait_set(*runtime);
    ASSERT_TRUE(wait_set) << holoscan::runtime_error_message(wait_set.error());
    ASSERT_TRUE((*wait_set)->attach(**subscriber));
    EXPECT_TRUE(holoscan::channel_internal::protected_terminal_wait_handle(**subscriber));
  }

  const std::string too_long_data_id(longest_data_id.size() + 1U, 'b');
  {
    auto runtime = holoscan::ChannelRuntime::create();
    ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
    holoscan::ChannelOptions drop_oldest{};
    drop_oldest.queue_policy = holoscan::ChannelQueuePolicy::kDropOldest;
    auto rejected = holoscan::channel_internal::InlinePayloadTestAccess::create_publisher(
        *runtime, too_long_data_id, sizeof(ControlPayload), drop_oldest);
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, holoscan::ErrorCode::kInvalidArgument);

    // The failed kDropOldest request must not leave a data service behind. The same public identity
    // remains available to its pre-existing kBlocking contract.
    const holoscan::ChannelOptions blocking{};
    auto blocking_publisher = holoscan::channel_internal::InlinePayloadTestAccess::create_publisher(
        *runtime, too_long_data_id, sizeof(ControlPayload), blocking);
    EXPECT_TRUE(blocking_publisher) << holoscan::runtime_error_message(blocking_publisher.error());
  }

  {
    auto runtime = holoscan::ChannelRuntime::create();
    ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
    constexpr std::string_view kReservedId{"public.hsdk-terminal-v1"};
    holoscan::ChannelOptions drop_oldest{};
    drop_oldest.queue_policy = holoscan::ChannelQueuePolicy::kDropOldest;
    auto drop_oldest_rejected =
        holoscan::channel_internal::InlinePayloadTestAccess::create_publisher(
            *runtime, kReservedId, sizeof(ControlPayload), drop_oldest);
    ASSERT_FALSE(drop_oldest_rejected);
    EXPECT_EQ(drop_oldest_rejected.error().code, holoscan::ErrorCode::kInvalidArgument);

    const holoscan::ChannelOptions blocking{};
    auto blocking_rejected = holoscan::channel_internal::InlinePayloadTestAccess::create_publisher(
        *runtime, kReservedId, sizeof(ControlPayload), blocking);
    ASSERT_FALSE(blocking_rejected);
    EXPECT_EQ(blocking_rejected.error().code, holoscan::ErrorCode::kInvalidArgument);
  }
}

// A checked no-peer observation before submission is retryable and consumes no publication
// authority. Current HoloMQ cannot prove zero visibility after send, so the backend adapter never
// exposes a raw accepted-zero outcome from this path.
// REQ: CHAN-067
// REQ: CHAN-076
TEST(ChannelInlinePayloadTest, NoRecipientRejectsBeforeSubmission) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  auto publisher = runtime->create_publisher<ControlPayload>(make_unique_channel_name());
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());

  auto emitted = publisher->emit(ControlPayload{.sequence = 1U, .value = 2, .timestamp = 3.0});
  ASSERT_FALSE(emitted);
  EXPECT_EQ(emitted.error().code, holoscan::ErrorCode::kNotReady);
}

// REQ: CHAN-125
TEST(ChannelInlinePayloadTest, SelectsLargeFixedPayloadBucketsThroughThirtyTwoKiB) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  // Key setup: create independent fixed-payload endpoints at each newly supported bucket edge.
  for (const std::size_t payload_size : {8192U, 16384U, 32768U}) {
    auto publisher = holoscan::channel_internal::InlinePayloadTestAccess::create_publisher(
        runtime.value(), make_unique_channel_name(), payload_size, {});
    ASSERT_TRUE(publisher) << "payload_size=" << payload_size << ": "
                           << holoscan::runtime_error_message(publisher.error());
  }
  auto oversized = holoscan::channel_internal::InlinePayloadTestAccess::create_publisher(
      runtime.value(), make_unique_channel_name(), 32769U, {});

  // Key assertion: 8-, 16-, and 32-KiB buckets materialize, while the first larger size fails.
  ASSERT_FALSE(oversized);
  EXPECT_EQ(oversized.error().code, holoscan::ErrorCode::kInvalidArgument);
}

// The private phase result maps pre-submit rejection back to its checked error and converts both
// invalid accepted-zero and uncertain post-submit outcomes into the stable non-retryable category.
// REQ: CHAN-067
// REQ: CHAN-076
// REQ: RUNTIME-019
TEST(ChannelInlinePayloadTest, PrivatePublicationPhaseMapsToPublicResult) {
  const std::array<std::byte, sizeof(ControlPayload)> bytes{};
  const holoscan::SampleMetadata metadata{};
  EXPECT_TRUE(holoscan::channel_internal::terminal_publication_error_consumes_authority(
      holoscan::ErrorCode::kPublicationPartial));
  EXPECT_TRUE(holoscan::channel_internal::terminal_publication_error_consumes_authority(
      holoscan::ErrorCode::kPublicationIndeterminate));
  EXPECT_TRUE(holoscan::channel_internal::terminal_publication_error_consumes_authority(
      holoscan::ErrorCode::kProtocolError));
  EXPECT_TRUE(holoscan::channel_internal::terminal_publication_error_consumes_authority(
      holoscan::ErrorCode::kInvalidArgument));
  EXPECT_FALSE(holoscan::channel_internal::terminal_publication_error_consumes_authority(
      holoscan::ErrorCode::kNotReady));

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  auto transport_publisher = holoscan::channel_internal::InlinePayloadTestAccess::create_publisher(
      runtime.value(), make_unique_channel_name(), sizeof(ControlPayload), {});
  ASSERT_TRUE(transport_publisher) << holoscan::runtime_error_message(transport_publisher.error());
  const std::array<std::byte, 4097U> oversized{};
  const auto oversized_result = transport_publisher.value()->emit_phase(oversized, metadata);
  EXPECT_EQ(oversized_result.phase,
            holoscan::channel_internal::InlinePublicationPhase::kPreSubmitRejected);
  EXPECT_EQ(oversized_result.error.code, holoscan::ErrorCode::kInvalidArgument);

  PhasePublisher rejected({
      .phase = holoscan::channel_internal::InlinePublicationPhase::kPreSubmitRejected,
      .error = holoscan::Error{holoscan::ErrorCode::kBackpressured},
  });
  auto rejected_result = rejected.emit(bytes, metadata);
  ASSERT_FALSE(rejected_result);
  EXPECT_EQ(rejected_result.error().code, holoscan::ErrorCode::kBackpressured);

  PhasePublisher invalid_accepted_zero({
      .phase = holoscan::channel_internal::InlinePublicationPhase::kAccepted,
      .recipient_count = 0U,
  });
  auto invalid_accepted_result = invalid_accepted_zero.emit(bytes, metadata);
  ASSERT_FALSE(invalid_accepted_result);
  EXPECT_EQ(invalid_accepted_result.error().code, holoscan::ErrorCode::kPublicationIndeterminate);

  PhasePublisher accepted({
      .phase = holoscan::channel_internal::InlinePublicationPhase::kAccepted,
      .recipient_count = 1U,
  });
  auto accepted_result = accepted.emit(bytes, metadata);
  ASSERT_TRUE(accepted_result);
  EXPECT_EQ(accepted_result.value(), 1U);

  PhasePublisher uncertain({
      .phase = holoscan::channel_internal::InlinePublicationPhase::kPostSubmitUncertain,
      .error = holoscan::Error{holoscan::ErrorCode::kFailure},
  });
  auto uncertain_result = uncertain.emit(bytes, metadata);
  ASSERT_FALSE(uncertain_result);
  EXPECT_EQ(uncertain_result.error().code, holoscan::ErrorCode::kPublicationIndeterminate);
}

// Runtime attaches event-enabled private subscribers to one opaque Channel wait set and consumes
// checked ready identities without exposing a HoloMQ port or listener handle.
// REQ: CHAN-066
// REQ: RUNTIME-016
// Topology (blocking, event-wake enabled, depth=4):
//   publisher --[event-enabled]--> subscriber (attached to private wait set)
// Key setup: subscriber attached to wait set; the second receive exercises the warmed path.
TEST(ChannelInlinePayloadTest, PrivateWaitSetReportsEventEnabledSubscriberReadiness) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  holoscan::ChannelOptions options{};
  options.enable_event_wake = true;
  options.subscriber_queue_depth = 4U;
  options.max_subscriber_queue_depth = 4U;

  auto subscriber = holoscan::channel_internal::InlinePayloadTestAccess::create_subscriber(
      runtime.value(), channel_name, sizeof(ControlPayload), options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  auto publisher = holoscan::channel_internal::InlinePayloadTestAccess::create_publisher(
      runtime.value(), channel_name, sizeof(ControlPayload), options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_until(
      [&publisher] {
        static_cast<void>(publisher.value()->update_connections());
        return publisher.value()->has_subscribers();
      },
      std::chrono::seconds(2)));

  auto wait_set =
      holoscan::channel_internal::InlinePayloadTestAccess::create_wait_set(runtime.value());
  ASSERT_TRUE(wait_set) << holoscan::runtime_error_message(wait_set.error());
  auto attached = wait_set.value()->attach(*subscriber.value());
  ASSERT_TRUE(attached) << holoscan::runtime_error_message(attached.error());

  auto duplicate = wait_set.value()->attach(*subscriber.value());
  ASSERT_FALSE(duplicate);
  EXPECT_EQ(duplicate.error().code, holoscan::ErrorCode::kInvalidArgument);

  const ControlPayload expected{.sequence = 44U, .value = -8, .timestamp = 3.25};
  const auto payload = std::as_bytes(std::span<const ControlPayload, 1U>(&expected, 1U));
  const bool delivered = wait_until(
      [&publisher, payload] {
        static_cast<void>(publisher.value()->update_connections());
        auto emitted = publisher.value()->emit(payload, {});
        return emitted && emitted.value() == 1U;
      },
      std::chrono::seconds(2));
  ASSERT_TRUE(delivered);

  std::array<std::uint64_t, 1U> ready{};
  const bool woke_with_handle = wait_until(
      [&wait_set, &ready, expected_handle = attached.value()] {
        auto waited = wait_set.value()->wait_for(std::chrono::milliseconds(50));
        if (!waited || !waited.value()) {
          return false;
        }
        auto count = wait_set.value()->ready_handles(ready);
        return count && count.value() == 1U && ready[0] == expected_handle;
      },
      std::chrono::seconds(2));
  ASSERT_TRUE(woke_with_handle);

  auto insufficient = wait_set.value()->ready_handles(std::span<std::uint64_t>{});
  ASSERT_FALSE(insufficient);
  EXPECT_EQ(insufficient.error().code, holoscan::ErrorCode::kResourceExhausted);

  ControlPayload actual{};
  auto received = subscriber.value()->receive(
      std::as_writable_bytes(std::span<ControlPayload, 1U>(&actual, 1U)));
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(actual.sequence, expected.sequence);
  EXPECT_EQ(actual.value, expected.value);
  EXPECT_DOUBLE_EQ(actual.timestamp, expected.timestamp);

  const ControlPayload warmed_expected{.sequence = 45U, .value = 12, .timestamp = 4.5};
  const auto warmed_payload =
      std::as_bytes(std::span<const ControlPayload, 1U>(&warmed_expected, 1U));
  ASSERT_TRUE(publisher.value()->emit(warmed_payload, {}));
  ASSERT_TRUE(wait_until(
      [&subscriber] {
        auto ready = subscriber.value()->has_samples();
        return ready && *ready;
      },
      std::chrono::seconds(2)));

  ControlPayload warmed_actual{};
  // Key assertion: the warmed path (wait_for + ready_handles + receive) performs zero allocations.
  bool warmed_path_succeeded = false;
  std::size_t warmed_allocations = 0U;
  {
    const holoscan::test::ScopedAllocationCounter allocations;
    auto waited = wait_set.value()->wait_for(std::chrono::milliseconds{50});
    auto ready_count = wait_set.value()->ready_handles(ready);
    auto has_samples = subscriber.value()->has_samples();
    auto warmed_received = subscriber.value()->receive(
        std::as_writable_bytes(std::span<ControlPayload, 1U>(&warmed_actual, 1U)));
    warmed_path_succeeded = waited && *waited && ready_count && *ready_count == 1U &&
                            ready[0] == attached.value() && has_samples && *has_samples &&
                            warmed_received;
    warmed_allocations = allocations.count();
  }
  EXPECT_TRUE(warmed_path_succeeded);
  EXPECT_EQ(warmed_actual.sequence, warmed_expected.sequence);
  EXPECT_EQ(warmed_actual.value, warmed_expected.value);
  EXPECT_DOUBLE_EQ(warmed_actual.timestamp, warmed_expected.timestamp);
  EXPECT_EQ(warmed_allocations, 0U);
}

// Verifies a non-schema inline POD payload can cross a real process boundary through the typed
// Channel API without using FlatBuffers or Tensor sidecars.
// Topology (blocking, depth=8, cross-process):
//   parent[publisher] --[ControlPayload over IPC]--> fork[subscriber child process]
// Key setup: parent and child synchronize via pipes; publisher starts before child subscribes.
TEST(ChannelInlinePayloadTest, BlockingPublishesControlPayloadToChildProcess) {
  const char* const child_exe = HOLOSCAN_CHANNEL_INTERPROCESS_CHILD_EXE;
  ASSERT_NE(child_exe[0], '\0') << "HOLOSCAN_CHANNEL_INTERPROCESS_CHILD_EXE not configured";

  holoscan::ChannelOptions options{};
  options.queue_policy = holoscan::ChannelQueuePolicy::kBlocking;
  options.subscriber_queue_depth = 8U;
  options.expected_recipient_count = 1U;
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
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_RESULT_FD", result_fd.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_START_FD", start_fd.c_str(), 1));
    static_cast<void>(::setenv("HOLOSCAN_CHANNEL_NAME", channel_name.c_str(), 1));
    ::execl(child_exe, "channel_interprocess_child", static_cast<char*>(nullptr));
    _exit(127);
  }
  ::close(result_pipe[1]);
  ::close(start_pipe[0]);
  ChildProcessCleanup child_cleanup(child, result_pipe[0], start_pipe[1]);

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  auto publisher = runtime->create_publisher<ControlPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());

  constexpr std::uint8_t kStartChild = 1U;
  ASSERT_TRUE(write_exact(child_cleanup.start_fd(), &kStartChild, sizeof(kStartChild)));
  child_cleanup.close_start();

  // Key setup: parent waits for child's subscriber-ready handshake before emitting payload.
  std::uint8_t ready_code = 0xffU;
  ASSERT_TRUE(read_exact(child_cleanup.result_fd(), &ready_code, sizeof(ready_code)));
  if (ready_code != 1U) {
    int status = 0;
    ASSERT_EQ(child_cleanup.wait(&status), child);
    ASSERT_TRUE(WIFEXITED(status));
    FAIL() << "child failed before subscriber-ready handshake; exit=" << WEXITSTATUS(status)
           << " validation code " << static_cast<unsigned>(ready_code);
  }

  const ControlPayload expected{.sequence = 301U, .value = -33, .timestamp = 99.25};
  std::string last_emit_error;
  bool child_exited_early = false;
  int child_status = 0;
  std::uint8_t early_child_code = 0xffU;
  const bool delivered = wait_until(
      [&publisher,
       &expected,
       &last_emit_error,
       child,
       &child_exited_early,
       &child_status,
       &early_child_code,
       &child_cleanup] {
        static_cast<void>(publisher->update_connections());
        auto published = publisher->emit(expected);
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
  ASSERT_TRUE(delivered) << "failed to publish POD payload to child: " << last_emit_error;

  std::uint8_t child_code = 0xffU;
  ASSERT_TRUE(read_exact(child_cleanup.result_fd(), &child_code, sizeof(child_code)));
  child_cleanup.close_result();

  int status = 0;
  ASSERT_EQ(child_cleanup.wait(&status), child);
  ASSERT_TRUE(WIFEXITED(status));
  // Key assertion: child reports code 0 (success) via pipe after validating the payload.
  EXPECT_EQ(WEXITSTATUS(status), 0);
  EXPECT_EQ(child_code, 0U) << "child validation stage/error code "
                            << static_cast<unsigned>(child_code);
}

// Verifies admitted scalar emit_sample/receive_sample carry Core metadata without changing the
// broader raw Channel-inline value path.
// REQ: CHANNEL-API-002
TEST(ChannelInlinePayloadTest, EmitSamplePublishesScalarMetadata) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  auto publisher = runtime->create_publisher<std::uint64_t>(channel_name);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<std::uint64_t>(channel_name);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  constexpr std::uint64_t expected = 18U;
  const holoscan::Sample<std::uint64_t> sample(expected,
                                               {.sequence = 9001U,
                                                .capture_timestamp_ns = 10'000,
                                                .source_clock_id = holoscan::ClockId{.value = 2U},
                                                .publish_timestamp_ns = 20'000,
                                                .trace_id = 3U,
                                                .frame_id = 4U});

  auto emitted = publisher->emit_sample(sample);
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  EXPECT_EQ(emitted.value(), 1U);

  auto received = subscriber->receive_sample();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->data, expected);
  EXPECT_EQ(received->metadata.sequence, 9001U);
  EXPECT_EQ(received->metadata.capture_timestamp_ns, 10'000);
  EXPECT_EQ(received->metadata.publish_timestamp_ns, 20'000);
  EXPECT_EQ(received->metadata.source_clock_id, holoscan::ClockId{.value = 2U});
  EXPECT_EQ(received->metadata.trace_id, 3U);
  EXPECT_EQ(received->metadata.frame_id, 4U);
  EXPECT_EQ(received->metadata.byte_size, sizeof(std::uint64_t));
  EXPECT_EQ(received->metadata.payload_kind, holoscan::PayloadKind::kInlineTyped);
  EXPECT_TRUE(received->metadata.valid());
}

// A topology-frozen publisher must not submit to an attached subset. Once the final recipient
// attaches, the same public Channel path publishes exactly once to all three.
// REQ: CHAN-067
// Topology (blocking, fanout=3):
//   publisher --[ControlPayload]--> subscriber_a
//                               --> subscriber_b
//                               --> subscriber_c (attached after the rejected attempt)
TEST(ChannelInlinePayloadTest, IncompleteFanoutRejectsBeforeSubmission) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  // Key setup: freeze a three-recipient contract, but initially attach only two subscribers.
  holoscan::ChannelOptions options{};
  options.expected_recipient_count = 3U;
  options.subscriber_queue_depth = 2U;

  auto publisher = runtime->create_publisher<ControlPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber_a = runtime->create_subscriber<ControlPayload>(channel_name, options);
  ASSERT_TRUE(subscriber_a) << holoscan::runtime_error_message(subscriber_a.error());
  auto subscriber_b = runtime->create_subscriber<ControlPayload>(channel_name, options);
  ASSERT_TRUE(subscriber_b) << holoscan::runtime_error_message(subscriber_b.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  const ControlPayload rejected_payload{.sequence = 100U, .value = 1, .timestamp = 100.5};
  const auto rejected = publisher->emit(rejected_payload);
  // Key assertion: incomplete attachment rejects before either connected recipient observes data.
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, holoscan::ErrorCode::kNotReady);
  EXPECT_FALSE(subscriber_a->has_samples().value());
  EXPECT_FALSE(subscriber_b->has_samples().value());

  auto subscriber_c = runtime->create_subscriber<ControlPayload>(channel_name, options);
  ASSERT_TRUE(subscriber_c) << holoscan::runtime_error_message(subscriber_c.error());
  ASSERT_TRUE(wait_for_control_payload_recipients(publisher.value(),
                                                  3U,
                                                  std::chrono::seconds(2),
                                                  subscriber_a.value(),
                                                  subscriber_b.value(),
                                                  subscriber_c.value()));

  const ControlPayload accepted_payload{.sequence = 101U, .value = 2, .timestamp = 101.5};
  const auto accepted = publisher->emit(accepted_payload);
  // Key assertion: after the third attachment, one logical publication reaches all recipients.
  ASSERT_TRUE(accepted) << holoscan::runtime_error_message(accepted.error());
  EXPECT_EQ(*accepted, 3U);
  const auto received_a = subscriber_a->receive();
  const auto received_b = subscriber_b->receive();
  const auto received_c = subscriber_c->receive();
  ASSERT_TRUE(received_a);
  ASSERT_TRUE(received_b);
  ASSERT_TRUE(received_c);
  EXPECT_EQ(received_a->sequence, accepted_payload.sequence);
  EXPECT_EQ(received_b->sequence, accepted_payload.sequence);
  EXPECT_EQ(received_c->sequence, accepted_payload.sequence);
}

// Verifies blocking inline POD fanout delivers the same payload to every connected subscriber.
// Topology (blocking, depth=4, fanout=3):
//   publisher --[ControlPayload]--> subscriber_a
//                               --> subscriber_b
//                               --> subscriber_c
// Key assertion: emit returns fanout count 3; all three subscribers receive the same payload.
// REQ: CHAN-124
TEST(ChannelInlinePayloadTest, BlockingFanoutPublishesToAllControlPayloadSubscribers) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  holoscan::ChannelOptions options{};
  options.expected_recipient_count = 3U;
  options.subscriber_queue_depth = 4U;

  auto publisher = runtime->create_publisher<ControlPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber_a = runtime->create_subscriber<ControlPayload>(channel_name, options);
  ASSERT_TRUE(subscriber_a) << holoscan::runtime_error_message(subscriber_a.error());
  auto subscriber_b = runtime->create_subscriber<ControlPayload>(channel_name, options);
  ASSERT_TRUE(subscriber_b) << holoscan::runtime_error_message(subscriber_b.error());
  auto subscriber_c = runtime->create_subscriber<ControlPayload>(channel_name, options);
  ASSERT_TRUE(subscriber_c) << holoscan::runtime_error_message(subscriber_c.error());
  ASSERT_TRUE(wait_for_control_payload_recipients(publisher.value(),
                                                  3U,
                                                  std::chrono::seconds(2),
                                                  subscriber_a.value(),
                                                  subscriber_b.value(),
                                                  subscriber_c.value()));

  const ControlPayload expected{.sequence = 101U, .value = 42, .timestamp = 101.5};
  auto emitted = publisher->emit(expected);
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  EXPECT_EQ(emitted.value(), 3U);

  auto received_a = subscriber_a->receive();
  auto received_b = subscriber_b->receive();
  auto received_c = subscriber_c->receive();
  ASSERT_TRUE(received_a) << holoscan::runtime_error_message(received_a.error());
  ASSERT_TRUE(received_b) << holoscan::runtime_error_message(received_b.error());
  ASSERT_TRUE(received_c) << holoscan::runtime_error_message(received_c.error());
  EXPECT_EQ(received_a->sequence, expected.sequence);
  EXPECT_EQ(received_b->sequence, expected.sequence);
  EXPECT_EQ(received_c->sequence, expected.sequence);
  EXPECT_EQ(received_a->value, expected.value);
  EXPECT_EQ(received_b->value, expected.value);
  EXPECT_EQ(received_c->value, expected.value);
  EXPECT_DOUBLE_EQ(received_a->timestamp, expected.timestamp);
  EXPECT_DOUBLE_EQ(received_b->timestamp, expected.timestamp);
  EXPECT_DOUBLE_EQ(received_c->timestamp, expected.timestamp);
  EXPECT_FALSE(subscriber_a->has_samples().value());
  EXPECT_FALSE(subscriber_b->has_samples().value());
  EXPECT_FALSE(subscriber_c->has_samples().value());
}

// Verifies subscriber-first and publisher-first endpoint creation share one static service
// contract while local subscriber queue-depth requests remain distinct.
// REQ: CHAN-064
// Topology (blocking, max_depth=4, fanout=2, mixed queue depths):
//   publisher --[depth=1]--> shallow_subscriber
//             --[depth=3]--> deeper_subscriber
// Key setup: shallow_subscriber created first (subscriber-first) to prove both endpoints
//   carry the complete static service contract independently of creation order.
TEST(ChannelInlinePayloadTest, BlockingFanoutSupportsMixedSubscriberQueueDepths) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  holoscan::ChannelOptions publisher_options{};
  publisher_options.expected_recipient_count = 2U;
  publisher_options.max_subscriber_queue_depth = 4U;
  publisher_options.subscriber_queue_depth = 1U;

  holoscan::ChannelOptions shallow_subscriber_options{};
  shallow_subscriber_options.expected_recipient_count = 2U;
  shallow_subscriber_options.max_subscriber_queue_depth = 4U;
  shallow_subscriber_options.subscriber_queue_depth = 1U;

  holoscan::ChannelOptions deeper_subscriber_options{};
  deeper_subscriber_options.expected_recipient_count = 2U;
  deeper_subscriber_options.max_subscriber_queue_depth = 4U;
  deeper_subscriber_options.subscriber_queue_depth = 3U;

  // Create a subscriber first to prove that both endpoint roles carry the complete static service
  // contract rather than relying on publisher-first creation.
  auto shallow_subscriber =
      runtime->create_subscriber<ControlPayload>(channel_name, shallow_subscriber_options);
  ASSERT_TRUE(shallow_subscriber) << holoscan::runtime_error_message(shallow_subscriber.error());
  auto publisher = runtime->create_publisher<ControlPayload>(channel_name, publisher_options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto deeper_subscriber =
      runtime->create_subscriber<ControlPayload>(channel_name, deeper_subscriber_options);
  ASSERT_TRUE(deeper_subscriber) << holoscan::runtime_error_message(deeper_subscriber.error());
  ASSERT_TRUE(wait_for_control_payload_recipients(publisher.value(),
                                                  2U,
                                                  std::chrono::seconds(2),
                                                  shallow_subscriber.value(),
                                                  deeper_subscriber.value()));

  const ControlPayload expected{.sequence = 301U, .value = 73, .timestamp = 301.5};
  auto emitted = publisher->emit(expected);
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  EXPECT_EQ(emitted.value(), 2U);

  auto shallow = shallow_subscriber->receive();
  auto deeper = deeper_subscriber->receive();
  ASSERT_TRUE(shallow) << holoscan::runtime_error_message(shallow.error());
  ASSERT_TRUE(deeper) << holoscan::runtime_error_message(deeper.error());
  EXPECT_EQ(shallow->sequence, expected.sequence);
  EXPECT_EQ(deeper->sequence, expected.sequence);
  EXPECT_EQ(shallow->value, expected.value);
  EXPECT_EQ(deeper->value, expected.value);
}

// Verifies a subscriber cannot request more queue depth than the publisher-side service maximum
// advertised for the channel.
TEST(ChannelInlinePayloadTest, RejectsSubscriberQueueDepthAbovePublisherMaximum) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  holoscan::ChannelOptions publisher_options{};
  publisher_options.max_subscriber_queue_depth = 1U;
  publisher_options.subscriber_queue_depth = 1U;

  auto publisher = runtime->create_publisher<ControlPayload>(channel_name, publisher_options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());

  holoscan::ChannelOptions subscriber_options{};
  subscriber_options.subscriber_queue_depth = 2U;

  auto subscriber = runtime->create_subscriber<ControlPayload>(channel_name, subscriber_options);
  ASSERT_FALSE(subscriber);
  EXPECT_EQ(subscriber.error().code, holoscan::ErrorCode::kIncompatible);
}

// Verifies drop-oldest inline delivery retains the newest configured number of payloads.
// Topology (kDropOldest, depth=3):
//   publisher --[kDropOldest, depth=3]--> subscriber
// Key assertion: after 5 publishes, only sequences 3-5 remain in the queue.
TEST(ChannelInlinePayloadTest, DropOldestKeepsNewestQueuedControlPayloads) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  holoscan::ChannelOptions options{};
  options.queue_policy = holoscan::ChannelQueuePolicy::kDropOldest;
  options.subscriber_queue_depth = 3U;

  auto publisher = runtime->create_publisher<ControlPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<ControlPayload>(channel_name, options);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  for (std::uint64_t sequence = 1U; sequence <= 5U; ++sequence) {
    ASSERT_TRUE(publisher->emit(ControlPayload{.sequence = sequence,
                                               .value = static_cast<std::int32_t>(sequence * 10U),
                                               .timestamp = static_cast<double>(sequence)}));
  }

  for (std::uint64_t expected_sequence = 3U; expected_sequence <= 5U; ++expected_sequence) {
    auto received = subscriber->receive();
    ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
    EXPECT_EQ(received->sequence, expected_sequence);
    EXPECT_EQ(received->value, static_cast<std::int32_t>(expected_sequence * 10U));
  }
  EXPECT_FALSE(subscriber->has_samples().value());
}

// Verifies depth-one drop-oldest POD fanout tracks newest-unread payloads independently per
// subscriber.
// Topology (kDropOldest, depth=1, fanout=3):
//   publisher --[kDropOldest, depth=1]--> fast_subscriber
//             --[kDropOldest, depth=1]--> slow_subscriber_a
//             --[kDropOldest, depth=1]--> slow_subscriber_b
// Key setup: fast_subscriber drains sequence 1 before sequences 2 and 3 are published.
// Key assertion: all three subscribers independently see sequence 3, not sequence 2.
TEST(ChannelInlinePayloadTest,
     DropOldestDepthOneFanoutKeepsNewestUnreadControlPayloadPerSubscriber) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  holoscan::ChannelOptions options{};
  options.expected_recipient_count = 3U;
  options.queue_policy = holoscan::ChannelQueuePolicy::kDropOldest;
  options.subscriber_queue_depth = 1U;

  auto publisher = runtime->create_publisher<ControlPayload>(channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto fast_subscriber = runtime->create_subscriber<ControlPayload>(channel_name, options);
  ASSERT_TRUE(fast_subscriber) << holoscan::runtime_error_message(fast_subscriber.error());
  auto slow_subscriber_a = runtime->create_subscriber<ControlPayload>(channel_name, options);
  ASSERT_TRUE(slow_subscriber_a) << holoscan::runtime_error_message(slow_subscriber_a.error());
  auto slow_subscriber_b = runtime->create_subscriber<ControlPayload>(channel_name, options);
  ASSERT_TRUE(slow_subscriber_b) << holoscan::runtime_error_message(slow_subscriber_b.error());
  ASSERT_TRUE(wait_for_control_payload_recipients(publisher.value(),
                                                  3U,
                                                  std::chrono::seconds(2),
                                                  fast_subscriber.value(),
                                                  slow_subscriber_a.value(),
                                                  slow_subscriber_b.value()));

  ASSERT_TRUE(publisher->emit(ControlPayload{.sequence = 1U, .value = 10, .timestamp = 1.0}));
  auto fast_first = fast_subscriber->receive();
  ASSERT_TRUE(fast_first) << holoscan::runtime_error_message(fast_first.error());
  ASSERT_EQ(fast_first->sequence, 1U);

  ASSERT_TRUE(publisher->emit(ControlPayload{.sequence = 2U, .value = 20, .timestamp = 2.0}));
  ASSERT_TRUE(publisher->emit(ControlPayload{.sequence = 3U, .value = 30, .timestamp = 3.0}));

  auto fast_latest = fast_subscriber->receive();
  auto slow_latest_a = slow_subscriber_a->receive();
  auto slow_latest_b = slow_subscriber_b->receive();
  ASSERT_TRUE(fast_latest) << holoscan::runtime_error_message(fast_latest.error());
  ASSERT_TRUE(slow_latest_a) << holoscan::runtime_error_message(slow_latest_a.error());
  ASSERT_TRUE(slow_latest_b) << holoscan::runtime_error_message(slow_latest_b.error());
  EXPECT_EQ(fast_latest->sequence, 3U);
  EXPECT_EQ(slow_latest_a->sequence, 3U);
  EXPECT_EQ(slow_latest_b->sequence, 3U);
  EXPECT_EQ(fast_latest->value, 30);
  EXPECT_EQ(slow_latest_a->value, 30);
  EXPECT_EQ(slow_latest_b->value, 30);
  EXPECT_FALSE(fast_subscriber->has_samples().value());
  EXPECT_FALSE(slow_subscriber_a->has_samples().value());
  EXPECT_FALSE(slow_subscriber_b->has_samples().value());
}

// Verifies explicit publisher/subscriber nouns preserve inline channel behavior.
TEST(ChannelRuntimeTest, PublisherSubscriberCarryInlinePayload) {
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  const std::string channel_name = make_unique_channel_name();

  auto publisher = runtime->create_publisher<ControlPayload>(channel_name);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<ControlPayload>(channel_name);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_subscribers(publisher.value(), std::chrono::seconds(2)));

  const ControlPayload expected{.sequence = 41U, .value = 7, .timestamp = 99.25};
  auto emitted = publisher->emit(expected);
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  EXPECT_EQ(emitted.value(), 1U);
  EXPECT_TRUE(subscriber->has_samples().value());

  auto received = subscriber->receive();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->sequence, expected.sequence);
  EXPECT_EQ(received->value, expected.value);
  EXPECT_DOUBLE_EQ(received->timestamp, expected.timestamp);
  EXPECT_FALSE(subscriber->has_samples().value());
}
