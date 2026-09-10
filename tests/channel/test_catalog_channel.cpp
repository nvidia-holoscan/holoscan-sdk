/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/channel/channel.hpp>
#include <holoscan/channel/endpoint_catalog_builder.hpp>
// Provides generated owning types and schema-trait specializations.
#include <holoscan/schema/flatbuffer_schema_package.hpp>  // NOLINT(misc-include-cleaner)

#include "../../src/channel/catalog_inline_payload_contract.hpp"
#include "../../src/channel/channel_catalog_binding.hpp"
#include "../../src/channel/endpoint_catalog_state.hpp"
#include "../../src/channel/pod_codec.hpp"
#include "../common/allocation_counter.hpp"
#include "channel_test_support.hpp"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>

using namespace holoscan::channel_test;  // NOLINT(build/namespaces)

namespace holoscan {

// Custom POD payload used to test EOS ordering and canonical fieldwise encoding.
// Representative tests: CustomPodCloseIsOrderedAfterDataAndCarriesNoTypedPayload,
//   CanonicalPodBindingPublishesPreencodedBytes.
struct CatalogEosPod {
  std::uint32_t value{};
};

struct CatalogMaximumInlinePod {
  std::array<std::uint8_t, channel_internal::kMaximumCatalogDataFrameBytes -
                               channel_internal::kCatalogDataAttributionBytes>
      bytes{};
};

struct CatalogOversizedInlinePod {
  std::array<std::uint8_t, channel_internal::kMaximumCatalogDataFrameBytes -
                               channel_internal::kCatalogDataAttributionBytes + 1U>
      bytes{};
};

template <>
struct ChannelTypeIdentity<CatalogEosPod> {
  static constexpr std::string_view name = "nvidia.holoscan.test.CatalogEosPod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<CatalogEosPod> {
  static constexpr auto fields =
      pod_fields(pod_field("value", &CatalogEosPod::value, offsetof(CatalogEosPod, value)));
};

template <>
struct ChannelTypeIdentity<CatalogMaximumInlinePod> {
  static constexpr std::string_view name = "nvidia.holoscan.test.CatalogMaximumInlinePod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<CatalogMaximumInlinePod> {
  static constexpr auto fields = pod_fields(pod_field("bytes", &CatalogMaximumInlinePod::bytes,
                                                      offsetof(CatalogMaximumInlinePod, bytes)));
};

template <>
struct ChannelTypeIdentity<CatalogOversizedInlinePod> {
  static constexpr std::string_view name = "nvidia.holoscan.test.CatalogOversizedInlinePod";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<CatalogOversizedInlinePod> {
  static constexpr auto fields = pod_fields(pod_field("bytes", &CatalogOversizedInlinePod::bytes,
                                                      offsetof(CatalogOversizedInlinePod, bytes)));
};

}  // namespace holoscan

namespace {

using holoscan::channel_internal::CatalogChannelBindingAccess;
using holoscan::channel_internal::CatalogChannelBindingTestAccess;
using holoscan::channel_internal::CatalogScalarCodec;
using holoscan::channel_internal::InlinePayloadPublisher;
using holoscan::channel_internal::InlinePayloadSubscriber;

// Type without a ChannelTypeIdentity specialization; used to verify that
// CatalogChannelScalar<UndescribedPayload> is false (static_asserts below).
struct UndescribedPayload {
  std::uint32_t value{};
};

static_assert(holoscan::CatalogChannelScalar<bool>);
static_assert(holoscan::CatalogChannelScalar<std::uint32_t>);
static_assert(holoscan::CatalogChannelScalar<double>);
static_assert(!holoscan::CatalogChannelScalar<UndescribedPayload>);
static_assert(holoscan::channel_internal::kMaximumCatalogDataFrameBytes ==
              holoscan::channel_internal::kMaximumInlinePayloadBytes);
static_assert(sizeof(holoscan::CatalogMaximumInlinePod) ==
              holoscan::channel_internal::kMaximumCatalogDataFrameBytes -
                  holoscan::channel_internal::kCatalogDataAttributionBytes);
constexpr std::uint64_t kMaximumSchemaCapacity =
    holoscan::channel_internal::kMaximumCatalogDataFrameBytes -
    holoscan::channel_internal::kCatalogDataAttributionBytes - holoscan::kSchemaEnvelopeHeaderSize;

// REQ: CHAN-068
TEST(CatalogChannelTest, ScalarCodecUsesCanonicalLittleEndianBytesAndRejectsInvalidBool) {
  std::uint32_t value = 0x01020304U;
  std::array<std::byte, 4U> encoded{};
  auto encoded_result =
      CatalogScalarCodec::encode(holoscan::ScalarIdentityKind::kUInt32,
                                 std::as_bytes(std::span<std::uint32_t, 1U>{&value, 1U}),
                                 encoded);
  ASSERT_TRUE(encoded_result) << holoscan::runtime_error_message(encoded_result.error());
  EXPECT_EQ(encoded,
            (std::array<std::byte, 4U>{
                std::byte{0x04}, std::byte{0x03}, std::byte{0x02}, std::byte{0x01}}));

  std::uint32_t decoded{};
  auto decoded_result = CatalogScalarCodec::decode(
      holoscan::ScalarIdentityKind::kUInt32,
      encoded,
      std::as_writable_bytes(std::span<std::uint32_t, 1U>{&decoded, 1U}));
  ASSERT_TRUE(decoded_result) << holoscan::runtime_error_message(decoded_result.error());
  EXPECT_EQ(decoded, value);

  bool true_value = true;
  std::array<std::byte, 1U> true_wire{};
  auto true_encoded =
      CatalogScalarCodec::encode(holoscan::ScalarIdentityKind::kBoolean,
                                 std::as_bytes(std::span<bool, 1U>{&true_value, 1U}),
                                 true_wire);
  ASSERT_TRUE(true_encoded) << holoscan::runtime_error_message(true_encoded.error());
  EXPECT_EQ(true_wire, (std::array<std::byte, 1U>{std::byte{0x01}}));
  EXPECT_EQ(CatalogScalarCodec::native_size(holoscan::ScalarIdentityKind::kBoolean), sizeof(bool));
  EXPECT_EQ(CatalogScalarCodec::wire_size(holoscan::ScalarIdentityKind::kBoolean), 1U);

  bool decoded_bool = false;
  auto true_decoded =
      CatalogScalarCodec::decode(holoscan::ScalarIdentityKind::kBoolean,
                                 true_wire,
                                 std::as_writable_bytes(std::span<bool, 1U>{&decoded_bool, 1U}));
  ASSERT_TRUE(true_decoded) << holoscan::runtime_error_message(true_decoded.error());
  EXPECT_TRUE(decoded_bool);

  bool unchanged = true;
  constexpr std::array<std::byte, 1U> invalid_bool{std::byte{0x02}};
  auto invalid =
      CatalogScalarCodec::decode(holoscan::ScalarIdentityKind::kBoolean,
                                 invalid_bool,
                                 std::as_writable_bytes(std::span<bool, 1U>{&unchanged, 1U}));
  ASSERT_FALSE(invalid);
  EXPECT_EQ(invalid.error().code, holoscan::ErrorCode::kProtocolError);
  EXPECT_TRUE(unchanged);
}

// Returns a globally unique endpoint name prefixed with "catalog_" for each test invocation.
[[nodiscard]] std::string unique_endpoint_name() {
  return "catalog_" + make_unique_channel_name();
}

// Builds a single-endpoint catalog for type T in the given direction and options.
// Used by most CatalogChannelTest cases to set up import or export endpoints.
template <typename T>
[[nodiscard]] holoscan::expected<holoscan::ChannelEndpointCatalog, holoscan::Error> build_catalog(
    const holoscan::TopicDirection direction, const std::string_view endpoint_name,
    const holoscan::TopicOptions& options = {}) {
  auto builder = holoscan::ChannelEndpointCatalog::builder("catalog-channel-test");
  const std::string uri = "holomq://" + std::string(endpoint_name);
  auto added = direction == holoscan::TopicDirection::kImport
                   ? builder.template import_topic<T>(endpoint_name, uri, options)
                   : builder.template export_topic<T>(endpoint_name, uri, options);
  if (!added) {
    return holoscan::make_unexpected(std::move(added).error());
  }
  return std::move(builder).build();
}

// REQ: CHAN-137
TEST(CatalogChannelTest, BuilderAcceptsMaximumCatalogFrameAndRejectsOneByteMore) {
  constexpr std::uint64_t kRequiredOversizedFrame =
      sizeof(holoscan::CatalogOversizedInlinePod) +
      holoscan::channel_internal::kCatalogDataAttributionBytes;
  const std::string maximum_name = unique_endpoint_name();
  auto maximum = build_catalog<holoscan::CatalogMaximumInlinePod>(holoscan::TopicDirection::kImport,
                                                                  maximum_name);
  ASSERT_TRUE(maximum) << holoscan::runtime_error_message(maximum.error());

  const std::string oversized_name = unique_endpoint_name();
  auto oversized = build_catalog<holoscan::CatalogOversizedInlinePod>(
      holoscan::TopicDirection::kImport, oversized_name);
  ASSERT_FALSE(oversized);
  EXPECT_EQ(oversized.error().code, holoscan::ErrorCode::kResourceExhausted);
  EXPECT_NE(oversized.error().message.find("ENDPOINT_CATALOG_INLINE_CAPACITY_EXCEEDED"),
            std::string::npos);
  EXPECT_NE(
      oversized.error().message.find("required_frame=" + std::to_string(kRequiredOversizedFrame)),
      std::string::npos);
  EXPECT_NE(oversized.error().message.find(
                "maximum_frame=" +
                std::to_string(holoscan::channel_internal::kMaximumCatalogDataFrameBytes)),
            std::string::npos);
}

// REQ: CHAN-075
// REQ: CHAN-120
// REQ: CHAN-137
TEST(CatalogChannelTest, SchemaCapacityRoundTripsAndBoundsMetadataAndTypedAttachment) {
  using Payload = holoscan::schema::StringT;
  constexpr std::uint64_t kCapacity{holoscan::generated_schema_max_serialized_size_v<Payload>};
  const std::string endpoint_name = unique_endpoint_name();
  auto catalog = build_catalog<Payload>(holoscan::TopicDirection::kImport, endpoint_name);
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());
  const auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);
  const auto schema_capacity = endpoint->schema_max_serialized_size();
  if (!schema_capacity.has_value()) {
    ADD_FAILURE() << "schema endpoint omitted its compiled serialized-capacity bound";
    return;
  }
  EXPECT_EQ(*schema_capacity, kCapacity);
  EXPECT_NE(catalog->json().find("\"max_serialized_size\":" + std::to_string(kCapacity)),
            std::string::npos);

  auto parsed = holoscan::ChannelEndpointCatalog::parse_json(catalog->json());
  ASSERT_TRUE(parsed) << holoscan::runtime_error_message(parsed.error());
  EXPECT_EQ(parsed->json(), catalog->json());
  EXPECT_EQ(parsed->digest(), catalog->digest());
  const auto parsed_endpoint = parsed->endpoint(endpoint_name);
  ASSERT_TRUE(parsed_endpoint);
  EXPECT_EQ(parsed_endpoint->schema_max_serialized_size(), kCapacity);

  const std::string capacity_field = ",\"max_serialized_size\":" + std::to_string(kCapacity);
  const auto expect_invalid_capacity = [&](const std::string& malformed) {
    const auto invalid = holoscan::ChannelEndpointCatalog::parse_json(malformed);
    ASSERT_FALSE(invalid);
    EXPECT_EQ(invalid.error().code, holoscan::ErrorCode::kInvalidArgument);
    EXPECT_EQ(invalid.error().message, "ENDPOINT_PAYLOAD_IDENTITY_INVALID");
  };
  std::string missing_capacity = catalog->json();
  const std::size_t capacity_offset = missing_capacity.find(capacity_field);
  ASSERT_NE(capacity_offset, std::string::npos);
  missing_capacity.erase(capacity_offset, capacity_field.size());
  expect_invalid_capacity(std::move(missing_capacity));

  std::string zero_capacity = catalog->json();
  zero_capacity.replace(
      zero_capacity.find(capacity_field), capacity_field.size(), ",\"max_serialized_size\":0");
  expect_invalid_capacity(std::move(zero_capacity));

  std::string overflowing_capacity = catalog->json();
  overflowing_capacity.replace(overflowing_capacity.find(capacity_field),
                               capacity_field.size(),
                               ",\"max_serialized_size\":4294967296");
  expect_invalid_capacity(std::move(overflowing_capacity));

  holoscan::SampleMetadata metadata{
      .sequence = 1U,
      .byte_size = 1U,
      .payload_kind = holoscan::PayloadKind::kSerializedBytes,
  };
  EXPECT_TRUE(CatalogChannelBindingTestAccess::validate_data_metadata(*parsed_endpoint, metadata));
  metadata.byte_size = static_cast<std::uint32_t>(kCapacity);
  EXPECT_TRUE(CatalogChannelBindingTestAccess::validate_data_metadata(*parsed_endpoint, metadata));
  metadata.byte_size = static_cast<std::uint32_t>(kCapacity + 1U);
  const auto oversized =
      CatalogChannelBindingTestAccess::validate_data_metadata(*parsed_endpoint, metadata);
  ASSERT_FALSE(oversized);
  EXPECT_EQ(oversized.error().code, holoscan::ErrorCode::kProtocolError);

  const auto* state = holoscan::detail::ChannelEndpointCatalogAccess::state(*catalog);
  ASSERT_NE(state, nullptr);
  const auto normalize_schema_capacity = [&](const std::uint64_t capacity) {
    auto endpoints = state->endpoints;
    endpoints.front().serialized_payload_capacity_bytes = capacity;
    return holoscan::detail::ChannelEndpointCatalogAccess::normalize(
        state->scope, std::move(endpoints), state->clocks, state->clock_conversions);
  };
  auto maximum_schema = normalize_schema_capacity(kMaximumSchemaCapacity);
  ASSERT_TRUE(maximum_schema) << holoscan::runtime_error_message(maximum_schema.error());
  constexpr std::uint64_t kOversizedSchemaCapacity = kMaximumSchemaCapacity + 1U;
  constexpr std::uint64_t kSchemaFramingOverhead =
      holoscan::channel_internal::kCatalogDataAttributionBytes +
      holoscan::kSchemaEnvelopeHeaderSize;
  constexpr std::uint64_t kRequiredOversizedSchemaFrame =
      kOversizedSchemaCapacity + kSchemaFramingOverhead;
  auto oversized_schema = normalize_schema_capacity(kOversizedSchemaCapacity);
  ASSERT_FALSE(oversized_schema);
  EXPECT_EQ(oversized_schema.error().code, holoscan::ErrorCode::kResourceExhausted);
  EXPECT_NE(oversized_schema.error().message.find("application_capacity=" +
                                                  std::to_string(kOversizedSchemaCapacity)),
            std::string::npos);
  EXPECT_NE(oversized_schema.error().message.find("framing_overhead=" +
                                                  std::to_string(kSchemaFramingOverhead)),
            std::string::npos);
  EXPECT_NE(oversized_schema.error().message.find("required_frame=" +
                                                  std::to_string(kRequiredOversizedSchemaFrame)),
            std::string::npos);
  EXPECT_NE(oversized_schema.error().message.find(
                "maximum_frame=" +
                std::to_string(holoscan::channel_internal::kMaximumCatalogDataFrameBytes)),
            std::string::npos);

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());

  auto larger_state = std::make_shared<holoscan::detail::ChannelEndpointCatalogState>(*state);
  ASSERT_EQ(larger_state->endpoints.size(), 1U);
  ++larger_state->endpoints.front().serialized_payload_capacity_bytes;
  ++larger_state->endpoints.front().service_static.serialized_payload_capacity_bytes;
  const auto larger_catalog =
      holoscan::detail::ChannelEndpointCatalogAccess::make(std::move(larger_state));
  const auto larger_endpoint = larger_catalog.endpoint(endpoint_name);
  ASSERT_TRUE(larger_endpoint);
  const auto accepted = runtime->create_publisher<Payload>(*larger_endpoint);
  ASSERT_TRUE(accepted) << holoscan::runtime_error_message(accepted.error());

  auto smaller_state = std::make_shared<holoscan::detail::ChannelEndpointCatalogState>(*state);
  --smaller_state->endpoints.front().serialized_payload_capacity_bytes;
  --smaller_state->endpoints.front().service_static.serialized_payload_capacity_bytes;
  const auto smaller_catalog =
      holoscan::detail::ChannelEndpointCatalogAccess::make(std::move(smaller_state));
  const auto smaller_endpoint = smaller_catalog.endpoint(endpoint_name);
  ASSERT_TRUE(smaller_endpoint);
  const auto rejected = runtime->create_publisher<Payload>(*smaller_endpoint);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, holoscan::ErrorCode::kIncompatible);
  EXPECT_EQ(rejected.error().message, "ENDPOINT_PAYLOAD_CAPACITY_INCOMPATIBLE");

  const std::string export_name = unique_endpoint_name();
  auto export_catalog = build_catalog<Payload>(holoscan::TopicDirection::kExport, export_name);
  ASSERT_TRUE(export_catalog) << holoscan::runtime_error_message(export_catalog.error());
  const auto* export_state = holoscan::detail::ChannelEndpointCatalogAccess::state(*export_catalog);
  ASSERT_NE(export_state, nullptr);

  auto smaller_export_state =
      std::make_shared<holoscan::detail::ChannelEndpointCatalogState>(*export_state);
  --smaller_export_state->endpoints.front().serialized_payload_capacity_bytes;
  --smaller_export_state->endpoints.front().service_static.serialized_payload_capacity_bytes;
  const auto smaller_export_catalog =
      holoscan::detail::ChannelEndpointCatalogAccess::make(std::move(smaller_export_state));
  const auto smaller_export_endpoint = smaller_export_catalog.endpoint(export_name);
  ASSERT_TRUE(smaller_export_endpoint);
  const auto accepted_reader = runtime->create_subscriber<Payload>(*smaller_export_endpoint);
  ASSERT_TRUE(accepted_reader) << holoscan::runtime_error_message(accepted_reader.error());

  auto larger_export_state =
      std::make_shared<holoscan::detail::ChannelEndpointCatalogState>(*export_state);
  ++larger_export_state->endpoints.front().serialized_payload_capacity_bytes;
  ++larger_export_state->endpoints.front().service_static.serialized_payload_capacity_bytes;
  const auto larger_export_catalog =
      holoscan::detail::ChannelEndpointCatalogAccess::make(std::move(larger_export_state));
  const auto larger_export_endpoint = larger_export_catalog.endpoint(export_name);
  ASSERT_TRUE(larger_export_endpoint);
  const auto rejected_reader = runtime->create_subscriber<Payload>(*larger_export_endpoint);
  ASSERT_FALSE(rejected_reader);
  EXPECT_EQ(rejected_reader.error().code, holoscan::ErrorCode::kIncompatible);
  EXPECT_EQ(rejected_reader.error().message, "ENDPOINT_PAYLOAD_CAPACITY_INCOMPATIBLE");
}

// REQ: CHAN-068
// REQ: RUNTIME-017
TEST(CatalogChannelTest, CanonicalInboundMetadataValidatorRejectsEveryMalformedCoordinate) {
  const std::string uncaptured_name = unique_endpoint_name();
  auto uncaptured_catalog =
      build_catalog<std::uint32_t>(holoscan::TopicDirection::kImport, uncaptured_name);
  ASSERT_TRUE(uncaptured_catalog) << holoscan::runtime_error_message(uncaptured_catalog.error());
  const auto uncaptured_endpoint = uncaptured_catalog->endpoint(uncaptured_name);
  ASSERT_TRUE(uncaptured_endpoint);

  holoscan::SampleMetadata canonical{
      .sequence = 1U,
      .byte_size = sizeof(std::uint32_t),
      .payload_kind = holoscan::PayloadKind::kInlineTyped,
  };
  ASSERT_TRUE(
      CatalogChannelBindingTestAccess::validate_data_metadata(*uncaptured_endpoint, canonical));
  canonical.flags = holoscan::SampleFlags::kInvalid | holoscan::SampleFlags::kDegraded;
  ASSERT_TRUE(
      CatalogChannelBindingTestAccess::validate_data_metadata(*uncaptured_endpoint, canonical));
  canonical.flags = holoscan::SampleFlags::kNone;
  const auto expect_protocol_error = [&](const holoscan::ChannelEndpointCatalog::EndpointView view,
                                         const holoscan::SampleMetadata& metadata) {
    auto rejected = CatalogChannelBindingTestAccess::validate_data_metadata(view, metadata);
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, holoscan::ErrorCode::kProtocolError);
  };

  auto malformed = canonical;
  malformed.sequence = 0U;
  expect_protocol_error(*uncaptured_endpoint, malformed);
  malformed = canonical;
  malformed.version.major = 2U;
  expect_protocol_error(*uncaptured_endpoint, malformed);
  malformed = canonical;
  malformed.version.minor = 1U;
  expect_protocol_error(*uncaptured_endpoint, malformed);
  malformed = canonical;
  // The validator owns the closed two-bit wire vocabulary and must fail closed on future bits.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  malformed.flags = static_cast<holoscan::SampleFlags>(4U);
  expect_protocol_error(*uncaptured_endpoint, malformed);
  malformed = canonical;
  malformed.byte_size = sizeof(std::uint16_t);
  expect_protocol_error(*uncaptured_endpoint, malformed);
  malformed = canonical;
  malformed.payload_kind = holoscan::PayloadKind::kSerializedBytes;
  expect_protocol_error(*uncaptured_endpoint, malformed);
  malformed = canonical;
  malformed.publish_timestamp_ns = 1;
  expect_protocol_error(*uncaptured_endpoint, malformed);
  malformed = canonical;
  malformed.publish_clock_id = holoscan::ClockId{1U};
  expect_protocol_error(*uncaptured_endpoint, malformed);
  malformed = canonical;
  malformed.capture_timestamp_ns = 1;
  expect_protocol_error(*uncaptured_endpoint, malformed);
  malformed = canonical;
  malformed.source_clock_id = holoscan::ClockId{1U};
  expect_protocol_error(*uncaptured_endpoint, malformed);

  const holoscan::TopicOptions captured_options{
      .capture_time =
          holoscan::RequireCaptureTime{.clock = holoscan::ClockKey::named("capture-clock")},
  };
  const std::string captured_name = unique_endpoint_name();
  auto captured_catalog = build_catalog<std::uint32_t>(
      holoscan::TopicDirection::kImport, captured_name, captured_options);
  ASSERT_TRUE(captured_catalog) << holoscan::runtime_error_message(captured_catalog.error());
  const auto captured_endpoint = captured_catalog->endpoint(captured_name);
  ASSERT_TRUE(captured_endpoint);
  const holoscan::ClockId capture_clock =
      holoscan::detail::ChannelEndpointCatalogAccess::capture_clock_id(*captured_endpoint);
  ASSERT_TRUE(capture_clock.valid());

  canonical.source_clock_id = capture_clock;
  ASSERT_TRUE(
      CatalogChannelBindingTestAccess::validate_data_metadata(*captured_endpoint, canonical));
  malformed = canonical;
  malformed.source_clock_id = holoscan::ClockId{capture_clock.value + 1U};
  expect_protocol_error(*captured_endpoint, malformed);
}

// Performs one type-erased raw receive and casts the bytes to Sample<T>.
// Used when the low-level binding subscriber is driven directly, bypassing typed facades.
template <typename T>
[[nodiscard]] holoscan::expected<holoscan::Sample<T>, holoscan::Error> receive_erased(
    InlinePayloadSubscriber& subscriber) {
  std::array<std::byte, sizeof(T)> bytes{};
  auto received = subscriber.receive(bytes);
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }
  if (received->payload_size != sizeof(T)) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kProtocolError});
  }
  return holoscan::Sample<T>{std::bit_cast<T>(bytes), received->metadata};
}

// Polls receive_erased<T> with backoff until a non-kNotReady result or the timeout expires.
// Necessary because each catalog receive applies at most one operation per call.
template <typename T>
[[nodiscard]] holoscan::expected<holoscan::Sample<T>, holoscan::Error> receive_erased_until(
    InlinePayloadSubscriber& subscriber, const std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto received = receive_erased<T>(subscriber);
    if (received || received.error().code != holoscan::ErrorCode::kNotReady) {
      return received;
    }
    // A monitored catalog receive applies one operation per call. Retry after a health-only
    // operation so these data assertions do not bypass or collapse the shared publisher order.
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kTimeout});
}

// Type-erased emit helper for an InlinePayloadPublisher used from catalog binding tests.
template <typename T>
[[nodiscard]] holoscan::expected<std::uint32_t, holoscan::Error> emit_erased(
    InlinePayloadPublisher& publisher, const T& value,
    const holoscan::SampleMetadata& metadata = {}) {
  return publisher.emit(std::as_bytes(std::span<const T, 1U>{&value, 1U}), metadata);
}

// Polls until publisher and subscriber confirm connection via two consecutive ready observations.
// Two observations avoid acting on an advisory snapshot that predates publisher construction.
template <typename Publisher, typename Subscriber>
[[nodiscard]] bool wait_for_catalog_subscribers(Publisher& publisher, Subscriber& subscriber,
                                                const std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::uint32_t consecutive_ready_refreshes = 0U;
  while (std::chrono::steady_clock::now() < deadline) {
    auto receiver_refreshed = subscriber.has_samples();
    auto refreshed = publisher.update_connections();
    if (receiver_refreshed && refreshed && publisher.has_subscribers()) {
      ++consecutive_ready_refreshes;
      // HoloMQ's advisory snapshot may observe a subscriber that predates publisher construction
      // before the first send-recipient snapshot settles. Two explicit refresh observations avoid
      // using the advisory Boolean as proof of a later publication result.
      if (consecutive_ready_refreshes == 2U) {
        return true;
      }
    } else {
      consecutive_ready_refreshes = 0U;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  return false;
}

// Symmetric counterpart to wait_for_catalog_subscribers for export-direction tests where
// the subscriber must confirm a connected publisher before the test proceeds.
template <typename Publisher, typename Subscriber>
[[nodiscard]] bool wait_for_catalog_publishers(Publisher& publisher, Subscriber& subscriber,
                                               const std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  std::uint32_t consecutive_ready_refreshes = 0U;
  while (std::chrono::steady_clock::now() < deadline) {
    auto receiver_refreshed = subscriber.has_samples();
    auto sender_refreshed = publisher.update_connections();
    if (receiver_refreshed && sender_refreshed && subscriber.has_publishers()) {
      ++consecutive_ready_refreshes;
      if (consecutive_ready_refreshes == 2U) {
        return true;
      }
    } else {
      consecutive_ready_refreshes = 0U;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  return false;
}

// REQ: CHAN-137
TEST(CatalogChannelTest, MaximumCatalogInlinePodTraversesPublicAndGraphEndpoints) {
  using Payload = holoscan::CatalogMaximumInlinePod;
  const std::string endpoint_name = unique_endpoint_name();
  auto catalog = build_catalog<Payload>(holoscan::TopicDirection::kImport, endpoint_name);
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());

  auto external_runtime = holoscan::ChannelRuntime::create();
  auto graph_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(external_runtime);
  ASSERT_TRUE(graph_runtime);

  auto publisher = external_runtime->create_publisher<Payload>(*catalog, endpoint_name);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);
  auto subscriber =
      CatalogChannelBindingAccess::create_subscriber<Payload>(*graph_runtime, *endpoint);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_catalog_subscribers(*publisher, **subscriber, std::chrono::seconds{2}));

  Payload payload{};
  payload.bytes.front() = 0x3CU;
  payload.bytes[payload.bytes.size() / 2U] = 0xA5U;
  payload.bytes.back() = 0x7EU;
  auto emitted = publisher->emit(payload);
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  auto received = receive_erased<Payload>(**subscriber);
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_TRUE(received->data.bytes == payload.bytes);
  EXPECT_EQ(received->metadata.byte_size, sizeof(Payload));

  const std::string export_name = unique_endpoint_name();
  auto export_catalog = build_catalog<Payload>(holoscan::TopicDirection::kExport, export_name);
  ASSERT_TRUE(export_catalog) << holoscan::runtime_error_message(export_catalog.error());
  auto export_endpoint = export_catalog->endpoint(export_name);
  ASSERT_TRUE(export_endpoint);
  auto public_subscriber = external_runtime->create_subscriber<Payload>(*export_endpoint);
  ASSERT_TRUE(public_subscriber) << holoscan::runtime_error_message(public_subscriber.error());
  auto graph_publisher =
      CatalogChannelBindingAccess::create_publisher<Payload>(*graph_runtime, *export_endpoint);
  ASSERT_TRUE(graph_publisher) << holoscan::runtime_error_message(graph_publisher.error());
  ASSERT_TRUE(
      wait_for_catalog_subscribers(**graph_publisher, *public_subscriber, std::chrono::seconds{2}));

  auto export_emitted = emit_erased(**graph_publisher, payload);
  ASSERT_TRUE(export_emitted) << holoscan::runtime_error_message(export_emitted.error());
  auto export_received = public_subscriber->receive_sample();
  ASSERT_TRUE(export_received) << holoscan::runtime_error_message(export_received.error());
  EXPECT_TRUE(export_received->data.bytes == payload.bytes);

  payload.bytes.front() = 0x42U;
  bool warmed_path_succeeded = false;
  std::size_t warmed_allocations = 0U;
  {
    const holoscan::test::ScopedAllocationCounter allocations;
    auto warmed_emitted = emit_erased(**graph_publisher, payload);
    auto warmed_received = public_subscriber->receive_sample();
    warmed_path_succeeded =
        warmed_emitted && warmed_received && warmed_received->data.bytes == payload.bytes;
    warmed_allocations = allocations.count();
  }
  EXPECT_TRUE(warmed_path_succeeded);
  EXPECT_EQ(warmed_allocations, 0U);
}

// REQ: CHAN-068
// REQ: CHAN-070
// Topology (kImport, blocking, default depth):
//   external_runtime[publisher] --[stable-name]--> graph_runtime[subscriber]
TEST(CatalogChannelTest, ImportPublisherFirstAttachesByStableNameAndPreservesOrdinarySequence) {
  const std::string endpoint_name = unique_endpoint_name();
  auto catalog = build_catalog<std::uint32_t>(holoscan::TopicDirection::kImport, endpoint_name);
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());

  auto external_runtime = holoscan::ChannelRuntime::create();
  auto graph_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(external_runtime);
  ASSERT_TRUE(graph_runtime);

  auto publisher = external_runtime->create_publisher<std::uint32_t>(*catalog, endpoint_name);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);
  auto subscriber =
      CatalogChannelBindingAccess::create_subscriber<std::uint32_t>(*graph_runtime, *endpoint);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_catalog_subscribers(*publisher, **subscriber, std::chrono::seconds{2}));

  // Key setup: ordinary endpoints have no health-control authority; the rejected health call
  // must not consume the shared operation sequence.
  auto health = publisher->report_clock_health(holoscan::ClockHealthReport::healthy());
  ASSERT_FALSE(health);
  EXPECT_EQ(health.error().code, holoscan::ErrorCode::kInvalidArgument);
  auto emitted = publisher->emit(42U, holoscan::EmitOptions{.frame_id = 7U});
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  // Key assertion: first data sample after the failed health call arrives at sequence one.
  EXPECT_EQ(*emitted, 1U);
  auto received = receive_erased<std::uint32_t>(**subscriber);
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->data, 42U);
  EXPECT_EQ(received->metadata.sequence, 1U);
  EXPECT_EQ(received->metadata.frame_id, 7U);
  EXPECT_EQ(received->metadata.byte_size, sizeof(std::uint32_t));
}

// REQ: CHAN-067
// REQ: CHAN-068
// Topology (kImport, blocking):
//   external_runtime[publisher] --x (no peer) then --> graph_runtime[subscriber]
TEST(CatalogChannelTest, NoPeerRejectionPreservesSequenceForLaterBlockingAttachment) {
  const std::string endpoint_name = unique_endpoint_name();
  auto catalog = build_catalog<std::uint32_t>(holoscan::TopicDirection::kImport, endpoint_name);
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());

  auto external_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(external_runtime);
  auto publisher = external_runtime->create_publisher<std::uint32_t>(*catalog, endpoint_name);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto refreshed = publisher->update_connections();
  ASSERT_TRUE(refreshed) << holoscan::runtime_error_message(refreshed.error());
  ASSERT_FALSE(publisher->has_subscribers());

  // Key setup: emit with no subscriber present is retryable and does not consume sequence.
  auto unobserved = publisher->emit(11U);
  ASSERT_FALSE(unobserved);
  EXPECT_EQ(unobserved.error().code, holoscan::ErrorCode::kNotReady);

  auto graph_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(graph_runtime);
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);
  auto subscriber =
      CatalogChannelBindingAccess::create_subscriber<std::uint32_t>(*graph_runtime, *endpoint);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_catalog_subscribers(*publisher, **subscriber, std::chrono::seconds{2}));

  auto observed = publisher->emit(22U);
  ASSERT_TRUE(observed) << holoscan::runtime_error_message(observed.error());
  // Key assertion: first successful emit after subscriber connects arrives at sequence one.
  EXPECT_EQ(*observed, 1U);
  auto received = receive_erased<std::uint32_t>(**subscriber);
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->data, 22U);
  EXPECT_EQ(received->metadata.sequence, 1U);
}

// REQ: CHAN-086
// Topology (kImport, blocking, subscriber-first):
//   external_runtime[publisher] --> graph_runtime[subscriber]  (subscriber created first)
TEST(CatalogChannelTest, ImportSubscriberFirstCreatesTheSameServiceStaticContract) {
  const std::string endpoint_name = unique_endpoint_name();
  auto catalog = build_catalog<std::uint32_t>(holoscan::TopicDirection::kImport, endpoint_name);
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());

  auto graph_runtime = holoscan::ChannelRuntime::create();
  auto external_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(graph_runtime);
  ASSERT_TRUE(external_runtime);

  // Subscriber-first startup: the graph-side subscriber creates the service.
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);
  auto subscriber =
      CatalogChannelBindingAccess::create_subscriber<std::uint32_t>(*graph_runtime, *endpoint);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());

  // The later publisher must verify the identical complete service-static
  // contract rather than finding a first-peer-shaped smaller service.
  auto publisher = external_runtime->create_publisher<std::uint32_t>(*catalog, endpoint_name);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_catalog_subscribers(*publisher, **subscriber, std::chrono::seconds{2}));

  auto emitted = publisher->emit(51U);
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  auto received = receive_erased<std::uint32_t>(**subscriber);
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->data, 51U);
  EXPECT_EQ(received->metadata.sequence, 1U);
}

// REQ: CHAN-068
// Topology (kExport, blocking):
//   graph_runtime[publisher] --> external_runtime[subscriber]  (subscriber created first)
TEST(CatalogChannelTest, ExportSubscriberFirstAttachesByViewAndPreservesMetadata) {
  const std::string endpoint_name = unique_endpoint_name();
  auto catalog = build_catalog<std::uint32_t>(holoscan::TopicDirection::kExport, endpoint_name);
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);

  auto external_runtime = holoscan::ChannelRuntime::create();
  auto graph_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(external_runtime);
  ASSERT_TRUE(graph_runtime);

  auto subscriber = external_runtime->create_subscriber<std::uint32_t>(*endpoint);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  auto publisher =
      CatalogChannelBindingAccess::create_publisher<std::uint32_t>(*graph_runtime, *endpoint);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_catalog_subscribers(**publisher, *subscriber, std::chrono::seconds{2}));

  const holoscan::SampleMetadata metadata{.trace_id = 11U, .frame_id = 19U};
  auto emitted = emit_erased(**publisher, std::uint32_t{73U}, metadata);
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  EXPECT_EQ(*emitted, 1U);
  auto received = subscriber->receive_sample();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->data, 73U);
  EXPECT_EQ(received->metadata.sequence, 1U);
  EXPECT_EQ(received->metadata.trace_id, 11U);
  EXPECT_EQ(received->metadata.frame_id, 19U);
}

// REQ: CHAN-068
// REQ: CHAN-110
// REQ: EXTERNAL-008
// Topology (kExport, kDropOldest, depth=1, borrowed=1):
//   graph_runtime[publisher] --[kDropOldest, depth=1]--> external_runtime[subscriber]
// REQ: CHAN-129
// REQ: CHAN-132
// REQ: EXTERNAL-009
TEST(CatalogChannelTest, DropOldestDepthOneCatalogBindingKeepsNewestUnreadScalar) {
  constexpr std::uint32_t kSubscriberDepth = 1U;
  constexpr std::uint32_t kSubscriberBorrowed = 1U;
  constexpr std::size_t kLogicalServiceNameCapacity = 255U;
  const std::string endpoint_name = unique_endpoint_name();
  holoscan::detail::ChannelEndpointRecord endpoint_record{
      .name = endpoint_name,
      .uri = "holomq://" + endpoint_name,
      .direction = holoscan::TopicDirection::kExport,
      .name_origin = holoscan::EndpointNameOrigin::kExplicit,
      .service = std::string(kLogicalServiceNameCapacity, 's'),
      .queue_policy = holoscan::QueuePolicy::kDropOldest,
      .service_max_subscriber_depth = kSubscriberDepth,
      .service_max_subscriber_borrowed = kSubscriberBorrowed,
      .payload_identity =
          holoscan::PayloadIdentityRecord{
              .kind = holoscan::PayloadIdentityKind::kScalar,
              .payload_kind = holoscan::PayloadKind::kInlineTyped,
              .scalar =
                  holoscan::ScalarIdentityRecord{.kind = holoscan::ScalarIdentityKind::kUInt32}},
      .bounds = holoscan::EndpointBounds{.subscriber =
                                             holoscan::SubscriberEndpointBounds{
                                                 .queue_depth = kSubscriberDepth,
                                                 .max_borrowed_samples = kSubscriberBorrowed}},
  };
  auto catalog = holoscan::detail::ChannelEndpointCatalogAccess::normalize(
      "catalog-channel-drop-oldest-test", {std::move(endpoint_record)});
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);
  EXPECT_EQ(endpoint->service().size(), kLogicalServiceNameCapacity);
  const std::string binding =
      holoscan::detail::ChannelEndpointCatalogAccess::backend_service_binding(*endpoint);
  EXPECT_EQ(binding.size(), 30U);
  EXPECT_TRUE(binding.starts_with("hsc5_local_v1_"));

  auto graph_runtime = holoscan::ChannelRuntime::create();
  auto external_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(graph_runtime);
  ASSERT_TRUE(external_runtime);
  auto external_subscriber = external_runtime->create_subscriber<std::uint32_t>(*endpoint);
  ASSERT_TRUE(external_subscriber) << holoscan::runtime_error_message(external_subscriber.error());
  auto graph_publisher =
      CatalogChannelBindingAccess::create_publisher<std::uint32_t>(*graph_runtime, *endpoint);
  ASSERT_TRUE(graph_publisher) << holoscan::runtime_error_message(graph_publisher.error());
  ASSERT_TRUE(wait_for_catalog_subscribers(
      **graph_publisher, *external_subscriber, std::chrono::seconds{2}));

  // Key setup: attach through the short physical binding derived from the maximum-length logical
  // service, then publish two samples without a receive so drop-oldest replaces the first.
  ASSERT_TRUE(emit_erased(**graph_publisher, std::uint32_t{17U}));
  ASSERT_TRUE(emit_erased(**graph_publisher, std::uint32_t{29U}));

  auto received = external_subscriber->receive_sample();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  // Key assertion: receive returns the newest (second) sample with sequence 2.
  EXPECT_EQ(received->data, 29U);
  EXPECT_EQ(received->metadata.sequence, 2U);
  auto empty = external_subscriber->receive_sample();
  ASSERT_FALSE(empty);
  EXPECT_EQ(empty.error().code, holoscan::ErrorCode::kNotReady);

  ASSERT_TRUE(emit_erased(**graph_publisher, std::uint32_t{41U}));
  ASSERT_TRUE(emit_erased(**graph_publisher, std::uint32_t{53U}));
  // Key assertion: the protected terminal derived from the physical binding does not replace the
  // retained newest sample even though the logical queue has only one data slot.
  ASSERT_TRUE((*graph_publisher)->close_stream());
  auto replaced_again = external_subscriber->receive_sample();
  ASSERT_TRUE(replaced_again) << holoscan::runtime_error_message(replaced_again.error());
  EXPECT_EQ(replaced_again->data, 53U);
  EXPECT_EQ(replaced_again->metadata.sequence, 4U);
  auto end_of_stream = external_subscriber->receive_sample();
  ASSERT_FALSE(end_of_stream);
  EXPECT_EQ(end_of_stream.error().code, holoscan::ErrorCode::kClosed);
}

// REQ: CHAN-068
// REQ: CHAN-069
// Topology (kImport, kDropOldest, depth=1, clock_health enabled):
//   external_runtime[publisher] --[kDropOldest + health]--> graph_runtime[subscriber]
// Publication order: H1 (healthy, observed), H2 (degraded), S1 (replaced), S2 (retained).
// REQ: CHAN-130
TEST(CatalogChannelTest, DropOldestDepthOneDataAppliesContiguousHealthBeforeNewestUnreadScalar) {
  using namespace std::chrono_literals;

  constexpr std::uint32_t kServiceDepth = 1U;
  const std::string endpoint_name = unique_endpoint_name();
  holoscan::detail::ChannelEndpointRecord endpoint_record{
      .name = endpoint_name,
      .uri = "holomq://" + endpoint_name,
      .direction = holoscan::TopicDirection::kImport,
      .name_origin = holoscan::EndpointNameOrigin::kExplicit,
      .queue_policy = holoscan::QueuePolicy::kDropOldest,
      .service_max_subscriber_depth = kServiceDepth,
      .service_max_subscriber_borrowed = 1U,
      .payload_identity =
          holoscan::PayloadIdentityRecord{
              .kind = holoscan::PayloadIdentityKind::kScalar,
              .payload_kind = holoscan::PayloadKind::kInlineTyped,
              .scalar =
                  holoscan::ScalarIdentityRecord{.kind = holoscan::ScalarIdentityKind::kUInt32}},
      .clock_health =
          holoscan::ClockHealthOptions{
              .capability = holoscan::ClockHealthCapability::kStatusReports,
              .requirement = holoscan::ClockHealthRequirement::kFreshHealthyReport,
              .max_report_age = 5s,
          },
      .bounds = holoscan::EndpointBounds{.publisher = holoscan::PublisherEndpointBounds{}},
  };
  auto catalog = holoscan::detail::ChannelEndpointCatalogAccess::normalize(
      "catalog-channel-drop-oldest-health-test", {std::move(endpoint_record)});
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);

  auto graph_runtime = holoscan::ChannelRuntime::create();
  auto external_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(graph_runtime);
  ASSERT_TRUE(external_runtime);
  auto graph_subscriber =
      CatalogChannelBindingAccess::create_subscriber<std::uint32_t>(*graph_runtime, *endpoint);
  ASSERT_TRUE(graph_subscriber) << holoscan::runtime_error_message(graph_subscriber.error());
  auto external_publisher = external_runtime->create_publisher<std::uint32_t>(*endpoint);
  ASSERT_TRUE(external_publisher) << holoscan::runtime_error_message(external_publisher.error());
  ASSERT_TRUE(wait_for_catalog_subscribers(
      *external_publisher, **graph_subscriber, std::chrono::seconds{2}));

  std::array<std::byte, sizeof(std::uint32_t)> payload{};
  auto receive_next_until =
      [&]() -> std::optional<holoscan::channel_internal::CatalogReceivedOperation> {
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (std::chrono::steady_clock::now() < deadline) {
      auto operation = CatalogChannelBindingAccess::receive_next(**graph_subscriber, payload);
      if (!operation) {
        return std::nullopt;
      }
      if (operation->kind != holoscan::channel_internal::CatalogReceivedOperationKind::kNone) {
        return *operation;
      }
      std::this_thread::sleep_for(1ms);
    }
    return std::nullopt;
  };

  // Key setup: the lossless health lane has the same depth-one bound, so observe H1 before
  // staging H2 and the drop-oldest replacement.
  ASSERT_TRUE(
      external_publisher->report_clock_health(holoscan::ClockHealthReport::healthy()));  // H1
  const auto first_health = receive_next_until();
  ASSERT_TRUE(first_health.has_value());
  const auto first_health_value =
      first_health.value_or(holoscan::channel_internal::CatalogReceivedOperation{});
  ASSERT_EQ(first_health_value.kind,
            holoscan::channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(first_health_value.clock_health.publisher_operation_sequence, 1U);
  EXPECT_EQ(first_health_value.clock_health.report_sequence, 1U);

  ASSERT_TRUE(external_publisher->report_clock_health(holoscan::ClockHealthReport::degraded(
      holoscan::ClockHealthReason::kSynchronizationLockLost)));  // H2
  ASSERT_TRUE(external_publisher->emit(17U));                    // S1, replaced
  ASSERT_TRUE(external_publisher->emit(29U));                    // S2, retained

  const auto second_health = receive_next_until();
  ASSERT_TRUE(second_health.has_value());
  const auto second_health_value =
      second_health.value_or(holoscan::channel_internal::CatalogReceivedOperation{});
  ASSERT_EQ(second_health_value.kind,
            holoscan::channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(second_health_value.clock_health.publisher_operation_sequence, 2U);
  EXPECT_EQ(second_health_value.clock_health.report_sequence, 2U);
  EXPECT_EQ(second_health_value.clock_health.assessment,
            holoscan::ClockHealthAssessment::kDegraded);

  // Key assertion: H2 still precedes the newest retained data sample (S2).
  const auto newest_data = receive_next_until();
  ASSERT_TRUE(newest_data.has_value());
  const auto newest_data_value =
      newest_data.value_or(holoscan::channel_internal::CatalogReceivedOperation{});
  ASSERT_EQ(newest_data_value.kind,
            holoscan::channel_internal::CatalogReceivedOperationKind::kData);
  EXPECT_EQ(newest_data_value.metadata.sequence, 4U);
  EXPECT_EQ(std::bit_cast<std::uint32_t>(payload), 29U);
}

// REQ: CHAN-069
// Topology (kImport, kDropOldest, depth=1, clock_health enabled):
//   external_runtime[publisher] --[kDropOldest + delayed health]--> graph_runtime[subscriber]
// Publication order: S1 (replaced), H1, S2 (retained); health observation is then delayed.
TEST(CatalogChannelTest,
     DropOldestDepthOneDataWaitsForDelayedHealthBeforeAttributingReplacementGap) {
  using namespace std::chrono_literals;

  constexpr std::uint32_t kServiceDepth = 1U;
  const std::string endpoint_name = unique_endpoint_name();
  holoscan::detail::ChannelEndpointRecord endpoint_record{
      .name = endpoint_name,
      .uri = "holomq://" + endpoint_name,
      .direction = holoscan::TopicDirection::kImport,
      .name_origin = holoscan::EndpointNameOrigin::kExplicit,
      .queue_policy = holoscan::QueuePolicy::kDropOldest,
      .service_max_subscriber_depth = kServiceDepth,
      .service_max_subscriber_borrowed = 1U,
      .payload_identity =
          holoscan::PayloadIdentityRecord{
              .kind = holoscan::PayloadIdentityKind::kScalar,
              .payload_kind = holoscan::PayloadKind::kInlineTyped,
              .scalar =
                  holoscan::ScalarIdentityRecord{.kind = holoscan::ScalarIdentityKind::kUInt32}},
      .clock_health =
          holoscan::ClockHealthOptions{
              .capability = holoscan::ClockHealthCapability::kStatusReports,
              .requirement = holoscan::ClockHealthRequirement::kFreshHealthyReport,
              .max_report_age = 5s,
          },
      .bounds = holoscan::EndpointBounds{.publisher = holoscan::PublisherEndpointBounds{}},
  };
  auto catalog = holoscan::detail::ChannelEndpointCatalogAccess::normalize(
      "catalog-channel-delayed-health-test", {std::move(endpoint_record)});
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);

  auto graph_runtime = holoscan::ChannelRuntime::create();
  auto external_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(graph_runtime);
  ASSERT_TRUE(external_runtime);
  auto graph_subscriber =
      CatalogChannelBindingAccess::create_subscriber<std::uint32_t>(*graph_runtime, *endpoint);
  ASSERT_TRUE(graph_subscriber) << holoscan::runtime_error_message(graph_subscriber.error());
  auto external_publisher = external_runtime->create_publisher<std::uint32_t>(*endpoint);
  ASSERT_TRUE(external_publisher) << holoscan::runtime_error_message(external_publisher.error());
  ASSERT_TRUE(wait_for_catalog_subscribers(
      *external_publisher, **graph_subscriber, std::chrono::seconds{2}));

  ASSERT_TRUE(external_publisher->emit(17U));  // S1, replaced
  ASSERT_TRUE(
      external_publisher->report_clock_health(holoscan::ClockHealthReport::healthy()));  // H1
  ASSERT_TRUE(external_publisher->emit(29U));  // S2, retained with health watermark one

  const auto data_deadline = std::chrono::steady_clock::now() + 2s;
  bool data_ready = false;
  while (std::chrono::steady_clock::now() < data_deadline) {
    auto observed = (*graph_subscriber)->has_samples();
    ASSERT_TRUE(observed) << holoscan::runtime_error_message(observed.error());
    if (*observed) {
      data_ready = true;
      break;
    }
    std::this_thread::sleep_for(1ms);
  }
  ASSERT_TRUE(data_ready);

  // Key setup: health observation is artificially delayed to block data delivery.
  ASSERT_TRUE(CatalogChannelBindingTestAccess::delay_health_observation(**graph_subscriber, 1U));
  std::array<std::byte, sizeof(std::uint32_t)> payload{};
  auto blocked = CatalogChannelBindingAccess::receive_next(**graph_subscriber, payload);
  ASSERT_TRUE(blocked);
  EXPECT_EQ(blocked->kind, holoscan::channel_internal::CatalogReceivedOperationKind::kNone);
  auto before_health = CatalogChannelBindingAccess::clock_health_state(**graph_subscriber);
  ASSERT_TRUE(before_health);
  EXPECT_FALSE(before_health->has_report);

  auto health = CatalogChannelBindingAccess::receive_next(**graph_subscriber, payload);
  ASSERT_TRUE(health);
  ASSERT_EQ(health->kind, holoscan::channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(health->clock_health.publisher_operation_sequence, 2U);
  EXPECT_EQ(health->clock_health.report_sequence, 1U);

  // Key assertion: data is retained until health is surfaced; sequence reflects both ops.
  auto newest_data = CatalogChannelBindingAccess::receive_next(**graph_subscriber, payload);
  ASSERT_TRUE(newest_data);
  ASSERT_EQ(newest_data->kind, holoscan::channel_internal::CatalogReceivedOperationKind::kData);
  EXPECT_EQ(newest_data->metadata.sequence, 3U);
  EXPECT_EQ(std::bit_cast<std::uint32_t>(payload), 29U);
}

// REQ: CHAN-087
// Topology (kImport, kDropOldest(1), two publisher incarnations):
//   old_runtime[publisher] --> graph_runtime[receiver]  (retires)
//   new_runtime[publisher] --> graph_runtime[receiver]
// Key timing: new incarnation sends health before old-incarnation data head is fenced.
TEST(CatalogChannelTest, UnlatchedMismatchedHealthIsFencedBeforeApplication) {
  using namespace std::chrono_literals;

  constexpr std::uint32_t kServiceDepth = 1U;
  const std::string endpoint_name = unique_endpoint_name();
  holoscan::detail::ChannelEndpointRecord endpoint_record{
      .name = endpoint_name,
      .uri = "holomq://" + endpoint_name,
      .direction = holoscan::TopicDirection::kImport,
      .name_origin = holoscan::EndpointNameOrigin::kExplicit,
      .queue_policy = holoscan::QueuePolicy::kDropOldest,
      .service_max_subscriber_depth = kServiceDepth,
      .service_max_subscriber_borrowed = 1U,
      .payload_identity =
          holoscan::PayloadIdentityRecord{
              .kind = holoscan::PayloadIdentityKind::kScalar,
              .payload_kind = holoscan::PayloadKind::kInlineTyped,
              .scalar =
                  holoscan::ScalarIdentityRecord{.kind = holoscan::ScalarIdentityKind::kUInt32}},
      .clock_health =
          holoscan::ClockHealthOptions{
              .capability = holoscan::ClockHealthCapability::kStatusReports,
              .requirement = holoscan::ClockHealthRequirement::kFreshHealthyReport,
              .max_report_age = 5s,
          },
      .bounds = holoscan::EndpointBounds{.publisher = holoscan::PublisherEndpointBounds{}},
  };
  auto catalog = holoscan::detail::ChannelEndpointCatalogAccess::normalize(
      "catalog-channel-incarnation-health-test", {std::move(endpoint_record)});
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);

  auto graph_runtime = holoscan::ChannelRuntime::create();
  auto old_runtime = holoscan::ChannelRuntime::create();
  auto new_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(graph_runtime);
  ASSERT_TRUE(old_runtime);
  ASSERT_TRUE(new_runtime);
  auto receiver =
      CatalogChannelBindingAccess::create_subscriber<std::uint32_t>(*graph_runtime, *endpoint);
  auto created_old_publisher = old_runtime->create_publisher<std::uint32_t>(*endpoint);
  ASSERT_TRUE(receiver) << holoscan::runtime_error_message(receiver.error());
  ASSERT_TRUE(created_old_publisher)
      << holoscan::runtime_error_message(created_old_publisher.error());
  std::optional<holoscan::ChannelPublisher<std::uint32_t>> old_publisher{
      std::move(*created_old_publisher)};
  ASSERT_TRUE(wait_for_catalog_subscribers(*old_publisher, **receiver, 2s));

  // Key setup: old incarnation leaves drop-oldest data in flight; new publisher sends health first.
  // Depth-one drop-oldest retains only the second old-incarnation data operation (sequence two).
  ASSERT_TRUE(old_publisher->emit(17U));
  ASSERT_TRUE(old_publisher->emit(29U));
  old_publisher.reset();

  auto new_publisher = new_runtime->create_publisher<std::uint32_t>(*endpoint);
  ASSERT_TRUE(new_publisher) << holoscan::runtime_error_message(new_publisher.error());
  ASSERT_TRUE(wait_for_catalog_subscribers(*new_publisher, **receiver, 2s));
  ASSERT_TRUE(new_publisher->report_clock_health(holoscan::ClockHealthReport::healthy()));

  // Key assertion: old-incarnation data is fenced before new-incarnation health can latch.
  std::array<std::byte, sizeof(std::uint32_t)> payload{};
  auto fenced = CatalogChannelBindingAccess::receive_next(**receiver, payload);
  ASSERT_TRUE(fenced) << holoscan::runtime_error_message(fenced.error());
  EXPECT_EQ(fenced->kind,
            holoscan::channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged);
  auto before_health = CatalogChannelBindingAccess::clock_health_state(**receiver);
  ASSERT_TRUE(before_health);
  EXPECT_FALSE(before_health->has_report);
  auto before_latch = CatalogChannelBindingAccess::latched_producer_generation(**receiver);
  ASSERT_TRUE(before_latch);
  EXPECT_EQ(*before_latch, 0U);

  // The earlier new-incarnation health head remains staged and can latch only after the
  // conflicting old-incarnation data head has been fenced.
  auto health = CatalogChannelBindingAccess::receive_next(**receiver, payload);
  ASSERT_TRUE(health) << holoscan::runtime_error_message(health.error());
  ASSERT_EQ(health->kind, holoscan::channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(health->clock_health.publisher_operation_sequence, 1U);
  EXPECT_TRUE(health->clock_health.has_report);
  auto latched = CatalogChannelBindingAccess::latched_producer_generation(**receiver);
  ASSERT_TRUE(latched);
  EXPECT_EQ(*latched, health->clock_health.producer_epoch);
}

// REQ: CHAN-068
// Topology (kImport, blocking, capture_time required):
//   external_runtime[publisher] --[capture timestamp]--> graph_runtime[subscriber]
TEST(CatalogChannelTest, ScalarCaptureHelperUsesTheCatalogClockAndPreservesMetadata) {
  const holoscan::TopicOptions options{
      .capture_time =
          holoscan::RequireCaptureTime{.clock = holoscan::ClockKey::named("sensor-capture-clock")},
  };
  const std::string endpoint_name = unique_endpoint_name();
  auto catalog =
      build_catalog<std::uint32_t>(holoscan::TopicDirection::kImport, endpoint_name, options);
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);

  auto external_runtime = holoscan::ChannelRuntime::create();
  auto graph_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(external_runtime);
  ASSERT_TRUE(graph_runtime);
  auto subscriber =
      CatalogChannelBindingAccess::create_subscriber<std::uint32_t>(*graph_runtime, *endpoint);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  auto publisher = external_runtime->create_publisher<std::uint32_t>(*endpoint);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_catalog_subscribers(*publisher, **subscriber, std::chrono::seconds{2}));

  auto capture = publisher->make_capture_time(1234567);
  ASSERT_TRUE(capture) << holoscan::runtime_error_message(capture.error());
  EXPECT_TRUE(capture->clock_id.valid());
  constexpr std::uint32_t expected = 9U;
  auto emitted = publisher->emit_captured(
      expected, capture->timestamp_ns, holoscan::EmitOptions{.frame_id = 99U});
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  auto received = receive_erased<std::uint32_t>(**subscriber);
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->data, expected);
  EXPECT_EQ(received->metadata.capture_timestamp_ns, 1234567);
  EXPECT_EQ(received->metadata.source_clock_id, capture->clock_id);
  EXPECT_EQ(received->metadata.frame_id, 99U);
}

// REQ: CHAN-069
// REQ: EXTERNAL-003
// Topology (kImport, blocking, capture_time + clock_health):
//   external_runtime[publisher] --[H+S interleaved]--> graph_runtime[receiver]
// Operation order: H1(seq=1), S1(seq=2), S2(seq=3), H2-unhealthy(seq=4), H3-healthy(seq=5),
//   S3(seq=6).
TEST(CatalogChannelTest, ClockHealthAndDataShareOneContiguousOperationSequence) {
  using namespace std::chrono_literals;

  const holoscan::TopicOptions options{
      .capture_time =
          holoscan::RequireCaptureTime{.clock = holoscan::ClockKey::named("capture-clock")},
      .clock_health =
          holoscan::ClockHealthOptions{
              .capability = holoscan::ClockHealthCapability::kStatusReports,
              .requirement = holoscan::ClockHealthRequirement::kFreshHealthyReport,
              .max_report_age = 5ms,
          },
  };
  const std::string endpoint_name = unique_endpoint_name();
  auto catalog =
      build_catalog<std::uint32_t>(holoscan::TopicDirection::kImport, endpoint_name, options);
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);

  auto external_runtime = holoscan::ChannelRuntime::create();
  auto graph_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(external_runtime);
  ASSERT_TRUE(graph_runtime);
  auto receiver =
      CatalogChannelBindingAccess::create_subscriber<std::uint32_t>(*graph_runtime, *endpoint);
  ASSERT_TRUE(receiver) << holoscan::runtime_error_message(receiver.error());
  auto publisher = external_runtime->create_publisher<std::uint32_t>(*endpoint);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_catalog_subscribers(*publisher, **receiver, std::chrono::seconds{2}));

  ASSERT_TRUE(publisher->report_clock_health(holoscan::ClockHealthReport::healthy()));  // H1
  ASSERT_TRUE(publisher->emit_captured(1U, 101));                                       // S1
  auto sample_one = receive_erased_until<std::uint32_t>(**receiver, 2s);
  ASSERT_TRUE(sample_one);
  EXPECT_EQ(sample_one->metadata.sequence, 2U);
  ASSERT_TRUE(publisher->emit_captured(2U, 102));  // S2
  auto sample_two = receive_erased<std::uint32_t>(**receiver);
  ASSERT_TRUE(sample_two);
  EXPECT_EQ(sample_two->metadata.sequence, 3U);

  // Key setup: unsafe H2→H3 transition without data between them; both must surface before S3.
  // The private health lane is intentionally bounded to one report, so H2 is applied before H3
  // rather than relaxing the ingress bound for the test.
  std::array<std::byte, sizeof(std::uint32_t)> payload{};
  std::optional<holoscan::Error> receive_error;
  auto receive_next_until = [&](const std::chrono::milliseconds timeout)
      -> std::optional<holoscan::channel_internal::CatalogReceivedOperation> {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
      auto operation = CatalogChannelBindingAccess::receive_next(**receiver, payload);
      if (!operation) {
        receive_error = std::move(operation).error();
        return std::nullopt;
      }
      if (operation->kind == holoscan::channel_internal::CatalogReceivedOperationKind::kNone) {
        std::this_thread::sleep_for(1ms);
        continue;
      }
      return *operation;
    }
    return std::nullopt;
  };

  ASSERT_TRUE(publisher->report_clock_health(holoscan::ClockHealthReport::unhealthy(
      holoscan::ClockHealthReason::kSynchronizationLockLost)));  // H2
  const auto unhealthy = receive_next_until(2s);
  ASSERT_FALSE(receive_error.has_value());
  ASSERT_TRUE(unhealthy.has_value());
  const auto unhealthy_value =
      unhealthy.value_or(holoscan::channel_internal::CatalogReceivedOperation{});
  EXPECT_EQ(unhealthy_value.kind,
            holoscan::channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(unhealthy_value.clock_health.publisher_operation_sequence, 4U);
  EXPECT_EQ(unhealthy_value.clock_health.report_sequence, 2U);
  EXPECT_EQ(unhealthy_value.clock_health.assessment, holoscan::ClockHealthAssessment::kUnhealthy);

  ASSERT_TRUE(publisher->report_clock_health(holoscan::ClockHealthReport::healthy()));  // H3
  const auto recovered = receive_next_until(2s);
  ASSERT_FALSE(receive_error.has_value());
  ASSERT_TRUE(recovered.has_value());
  const auto recovered_value =
      recovered.value_or(holoscan::channel_internal::CatalogReceivedOperation{});
  EXPECT_EQ(recovered_value.kind,
            holoscan::channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(recovered_value.clock_health.publisher_operation_sequence, 5U);
  EXPECT_EQ(recovered_value.clock_health.report_sequence, 3U);
  EXPECT_EQ(recovered_value.clock_health.assessment, holoscan::ClockHealthAssessment::kHealthy);

  // Key assertion: S3 arrives at publisher_operation_sequence 6 with no gaps.
  ASSERT_TRUE(publisher->emit_captured(3U, 103));  // S3
  const auto sample_three = receive_next_until(2s);
  ASSERT_FALSE(receive_error.has_value());
  ASSERT_TRUE(sample_three.has_value());
  const auto sample_three_value =
      sample_three.value_or(holoscan::channel_internal::CatalogReceivedOperation{});
  EXPECT_EQ(sample_three_value.kind,
            holoscan::channel_internal::CatalogReceivedOperationKind::kData);
  EXPECT_EQ(sample_three_value.metadata.sequence, 6U);
  EXPECT_EQ(std::bit_cast<std::uint32_t>(payload), 3U);

  auto state = CatalogChannelBindingAccess::clock_health_state(**receiver);
  ASSERT_TRUE(state);
  EXPECT_TRUE(state->has_report);
  // The wire producer field carries the publisher attachment's incarnation identity token (not
  // an ordinal). It must be nonzero and exactly the incarnation the receiver latched.
  auto latched_generation = CatalogChannelBindingAccess::latched_producer_generation(**receiver);
  ASSERT_TRUE(latched_generation);
  EXPECT_NE(state->producer_epoch, 0U);
  EXPECT_EQ(state->producer_epoch, *latched_generation);
  EXPECT_EQ(state->publisher_operation_sequence, 5U);
  EXPECT_EQ(state->report_sequence, 3U);
  EXPECT_EQ(state->assessment, holoscan::ClockHealthAssessment::kHealthy);
  EXPECT_EQ(state->reason, holoscan::ClockHealthReason::kNone);
}

// REQ: CHAN-078
// REQ: CHAN-110
// REQ: EXTERNAL-003
// REQ: EXTERNAL-008
// Topology (kImport, blocking, capture_time + clock_health):
//   external_runtime[publisher] --[S1, H1, EOS]--> graph_runtime[receiver]
// Health and EOS use separate private queues but share one publisher operation sequence.
// REQ: EXTERNAL-009
TEST(CatalogChannelTest, ExplicitEndOfStreamIsOrderedAndConsumesPublisherAuthorityOnce) {
  using namespace std::chrono_literals;

  const holoscan::TopicOptions options{
      .capture_time =
          holoscan::RequireCaptureTime{.clock = holoscan::ClockKey::named("capture-clock")},
      .clock_health =
          holoscan::ClockHealthOptions{
              .capability = holoscan::ClockHealthCapability::kStatusReports,
              .requirement = holoscan::ClockHealthRequirement::kFreshHealthyReport,
              .max_report_age = 5ms,
          },
  };
  const std::string endpoint_name = unique_endpoint_name();
  auto catalog =
      build_catalog<std::uint32_t>(holoscan::TopicDirection::kImport, endpoint_name, options);
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);

  auto external_runtime = holoscan::ChannelRuntime::create();
  auto graph_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(external_runtime);
  ASSERT_TRUE(graph_runtime);
  auto receiver =
      CatalogChannelBindingAccess::create_subscriber<std::uint32_t>(*graph_runtime, *endpoint);
  ASSERT_TRUE(receiver) << holoscan::runtime_error_message(receiver.error());
  auto publisher = external_runtime->create_publisher<std::uint32_t>(*endpoint);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_catalog_subscribers(*publisher, **receiver, 2s));

  ASSERT_TRUE(publisher->emit_captured(41U, 101));  // S1
  auto sample = receive_erased_until<std::uint32_t>(**receiver, 2s);
  ASSERT_TRUE(sample) << holoscan::runtime_error_message(sample.error());
  EXPECT_EQ(sample->data, 41U);
  EXPECT_EQ(sample->metadata.sequence, 1U);

  // Key setup: H1 and EOS are staged simultaneously after S1 is consumed; H1 must precede EOS.
  ASSERT_TRUE(publisher->report_clock_health(holoscan::ClockHealthReport::healthy()));  // H1
  ASSERT_TRUE(publisher->close());                                                      // EOS

  std::array<std::byte, sizeof(std::uint32_t)> payload{};
  auto health = CatalogChannelBindingAccess::receive_next(**receiver, payload);
  ASSERT_TRUE(health) << holoscan::runtime_error_message(health.error());
  ASSERT_EQ(health->kind, holoscan::channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(health->clock_health.publisher_operation_sequence, 2U);
  EXPECT_EQ(health->clock_health.report_sequence, 1U);

  auto end_of_stream = CatalogChannelBindingAccess::receive_next(**receiver, payload);
  ASSERT_TRUE(end_of_stream) << holoscan::runtime_error_message(end_of_stream.error());
  ASSERT_EQ(end_of_stream->kind,
            holoscan::channel_internal::CatalogReceivedOperationKind::kEndOfStream);
  EXPECT_EQ(end_of_stream->metadata.sequence, 3U);
  EXPECT_EQ(end_of_stream->metadata.byte_size, 0U);

  // Key assertion: second close() and any post-close emit return kClosed.
  auto repeated = publisher->close();
  ASSERT_FALSE(repeated);
  EXPECT_EQ(repeated.error().code, holoscan::ErrorCode::kClosed);
  auto emitted_after_close = publisher->emit_captured(42U, 102);
  ASSERT_FALSE(emitted_after_close);
  EXPECT_EQ(emitted_after_close.error().code, holoscan::ErrorCode::kClosed);
  auto drained = CatalogChannelBindingAccess::receive_next(**receiver, payload);
  ASSERT_TRUE(drained) << holoscan::runtime_error_message(drained.error());
  EXPECT_EQ(drained->kind, holoscan::channel_internal::CatalogReceivedOperationKind::kNone);
}

// REQ: CHAN-087
// Topology (kImport, blocking, clock_health, two subscriber generations):
//   external_runtime[old_publisher] --> graph_runtime[first_receiver]  (reset)
//                                   --> graph_runtime[replacement_receiver]
//   external_runtime[new_publisher] --> graph_runtime[replacement_receiver]
TEST(CatalogChannelTest, ReplacementSubscriberDropsForeignDataAndEosBeforeLatchingNewPeer) {
  using namespace std::chrono_literals;

  const std::string endpoint_name = unique_endpoint_name();
  holoscan::TopicOptions options;
  options.clock_health = holoscan::ClockHealthOptions{
      .capability = holoscan::ClockHealthCapability::kStatusReports,
      .requirement = holoscan::ClockHealthRequirement::kFreshHealthyReport,
      .max_report_age = 5s,
  };
  auto catalog =
      build_catalog<std::uint32_t>(holoscan::TopicDirection::kImport, endpoint_name, options);
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);
  auto external_runtime = holoscan::ChannelRuntime::create();
  auto graph_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(external_runtime);
  ASSERT_TRUE(graph_runtime);

  auto first_receiver =
      CatalogChannelBindingAccess::create_subscriber<std::uint32_t>(*graph_runtime, *endpoint);
  auto created_old_publisher = external_runtime->create_publisher<std::uint32_t>(*endpoint);
  ASSERT_TRUE(first_receiver) << holoscan::runtime_error_message(first_receiver.error());
  ASSERT_TRUE(created_old_publisher)
      << holoscan::runtime_error_message(created_old_publisher.error());
  std::optional<holoscan::ChannelPublisher<std::uint32_t>> old_publisher{
      std::move(*created_old_publisher)};
  ASSERT_TRUE(wait_for_catalog_subscribers(*old_publisher, **first_receiver, 2s));

  ASSERT_TRUE(old_publisher->emit(11U));
  auto initial = receive_erased_until<std::uint32_t>(**first_receiver, 2s);
  ASSERT_TRUE(initial) << holoscan::runtime_error_message(initial.error());
  EXPECT_EQ(initial->data, 11U);
  auto initial_generation =
      CatalogChannelBindingAccess::latched_producer_generation(**first_receiver);
  ASSERT_TRUE(initial_generation);
  const std::uint64_t old_generation = *initial_generation;
  ASSERT_NE(old_generation, 0U);

  first_receiver->reset();
  auto replacement_receiver =
      CatalogChannelBindingAccess::create_subscriber<std::uint32_t>(*graph_runtime, *endpoint);
  ASSERT_TRUE(replacement_receiver)
      << holoscan::runtime_error_message(replacement_receiver.error());
  // Key setup: replacement receiver supersedes old generation to reject stale operations.
  ASSERT_TRUE(CatalogChannelBindingAccess::supersede_producer_generation(**replacement_receiver,
                                                                         old_generation));
  ASSERT_TRUE(wait_for_catalog_subscribers(*old_publisher, **replacement_receiver, 2s));

  std::array<std::byte, sizeof(std::uint32_t)> payload{};
  auto health_before = CatalogChannelBindingAccess::clock_health_state(**replacement_receiver);
  ASSERT_TRUE(health_before);
  EXPECT_FALSE(health_before->has_report);
  ASSERT_TRUE(old_publisher->report_clock_health(holoscan::ClockHealthReport::healthy()));
  auto stale_health = CatalogChannelBindingAccess::receive_next(**replacement_receiver, payload);
  ASSERT_TRUE(stale_health) << holoscan::runtime_error_message(stale_health.error());
  EXPECT_EQ(stale_health->kind,
            holoscan::channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged);
  auto health_after = CatalogChannelBindingAccess::clock_health_state(**replacement_receiver);
  ASSERT_TRUE(health_after);
  EXPECT_EQ(health_after->has_report, health_before->has_report);
  EXPECT_EQ(health_after->producer_epoch, health_before->producer_epoch);
  EXPECT_EQ(health_after->publisher_operation_sequence,
            health_before->publisher_operation_sequence);
  EXPECT_EQ(health_after->report_sequence, health_before->report_sequence);
  ASSERT_TRUE(old_publisher->emit(22U));
  auto stale_data = CatalogChannelBindingAccess::receive_next(**replacement_receiver, payload);
  ASSERT_TRUE(stale_data) << holoscan::runtime_error_message(stale_data.error());
  EXPECT_EQ(stale_data->kind,
            holoscan::channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged);
  EXPECT_EQ(std::bit_cast<std::uint32_t>(payload), 0U);
  ASSERT_TRUE(old_publisher->close());
  auto stale_eos = CatalogChannelBindingAccess::receive_next(**replacement_receiver, payload);
  ASSERT_TRUE(stale_eos) << holoscan::runtime_error_message(stale_eos.error());
  EXPECT_EQ(stale_eos->kind,
            holoscan::channel_internal::CatalogReceivedOperationKind::kPeerIncarnationChanged);
  // Key assertion: stale health, data, and EOS all return kPeerIncarnationChanged; not latched.
  auto still_unlatched =
      CatalogChannelBindingAccess::latched_producer_generation(**replacement_receiver);
  ASSERT_TRUE(still_unlatched);
  EXPECT_EQ(*still_unlatched, 0U);

  old_publisher.reset();
  auto new_publisher = external_runtime->create_publisher<std::uint32_t>(*endpoint);
  ASSERT_TRUE(new_publisher) << holoscan::runtime_error_message(new_publisher.error());
  ASSERT_TRUE(wait_for_catalog_subscribers(*new_publisher, **replacement_receiver, 2s));
  ASSERT_TRUE(new_publisher->emit(33U));
  auto current = receive_erased_until<std::uint32_t>(**replacement_receiver, 2s);
  ASSERT_TRUE(current) << holoscan::runtime_error_message(current.error());
  EXPECT_EQ(current->data, 33U);
  EXPECT_EQ(current->metadata.sequence, 1U);
  auto current_generation =
      CatalogChannelBindingAccess::latched_producer_generation(**replacement_receiver);
  ASSERT_TRUE(current_generation);
  EXPECT_NE(*current_generation, old_generation);
}

// REQ: CHAN-078
// REQ: EXTERNAL-003
// Topology (kImport, blocking):
//   external_runtime[publisher] --> graph_runtime[receiver]
// Key timing: publisher calls close() before any subscriber exists (kNotReady); later
//   publisher destructs without an explicit close, so no EOS marker appears.
TEST(CatalogChannelTest, UnobservedCloseIsRetryableAndDisconnectDoesNotImplyEndOfStream) {
  using namespace std::chrono_literals;

  const std::string endpoint_name = unique_endpoint_name();
  auto catalog = build_catalog<std::uint32_t>(holoscan::TopicDirection::kImport, endpoint_name);
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);
  auto external_runtime = holoscan::ChannelRuntime::create();
  auto graph_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(external_runtime);
  ASSERT_TRUE(graph_runtime);
  auto created_publisher = external_runtime->create_publisher<std::uint32_t>(*endpoint);
  ASSERT_TRUE(created_publisher) << holoscan::runtime_error_message(created_publisher.error());
  std::optional<holoscan::ChannelPublisher<std::uint32_t>> publisher{std::move(*created_publisher)};

  // Key setup: close() before subscriber exists is retryable; it consumes no sequence authority.
  auto unobserved = publisher->close();
  ASSERT_FALSE(unobserved);
  EXPECT_EQ(unobserved.error().code, holoscan::ErrorCode::kNotReady);
  auto receiver =
      CatalogChannelBindingAccess::create_subscriber<std::uint32_t>(*graph_runtime, *endpoint);
  ASSERT_TRUE(receiver) << holoscan::runtime_error_message(receiver.error());
  ASSERT_TRUE(wait_for_catalog_subscribers(*publisher, **receiver, 2s));
  auto emitted = publisher->emit(73U);
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  EXPECT_EQ(*emitted, 1U);
  auto received = receive_erased_until<std::uint32_t>(**receiver, 2s);
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->data, 73U);
  EXPECT_EQ(received->metadata.sequence, 1U);
  publisher.reset();

  // Key assertion: publisher destructs without close(); no EOS marker appears in receive loop.
  std::array<std::byte, sizeof(std::uint32_t)> payload{};
  const auto deadline = std::chrono::steady_clock::now() + 250ms;
  while (std::chrono::steady_clock::now() < deadline) {
    auto operation = CatalogChannelBindingAccess::receive_next(**receiver, payload);
    ASSERT_TRUE(operation) << holoscan::runtime_error_message(operation.error());
    EXPECT_EQ(operation->kind, holoscan::channel_internal::CatalogReceivedOperationKind::kNone);
    std::this_thread::sleep_for(1ms);
  }
}

// REQ: CHAN-078
// REQ: EXTERNAL-003
// Topology (import + export catalog, blocking, CatalogEosPod):
//   external_runtime[publisher via import] --> graph_runtime[subscriber via export]
// Key constraint: data must be consumed before EOS to avoid blocking a depth-1 blocking queue.
TEST(CatalogChannelTest, CustomPodCloseIsOrderedAfterDataAndCarriesNoTypedPayload) {
  using namespace std::chrono_literals;

  const std::string endpoint_name = unique_endpoint_name();
  auto import_catalog =
      build_catalog<holoscan::CatalogEosPod>(holoscan::TopicDirection::kImport, endpoint_name);
  auto export_catalog =
      build_catalog<holoscan::CatalogEosPod>(holoscan::TopicDirection::kExport, endpoint_name);
  ASSERT_TRUE(import_catalog) << holoscan::runtime_error_message(import_catalog.error());
  ASSERT_TRUE(export_catalog) << holoscan::runtime_error_message(export_catalog.error());
  auto import_endpoint = import_catalog->endpoint(endpoint_name);
  auto export_endpoint = export_catalog->endpoint(endpoint_name);
  ASSERT_TRUE(import_endpoint);
  ASSERT_TRUE(export_endpoint);

  auto external_runtime = holoscan::ChannelRuntime::create();
  auto graph_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(external_runtime);
  ASSERT_TRUE(graph_runtime);
  auto subscriber = graph_runtime->create_subscriber<holoscan::CatalogEosPod>(*export_endpoint);
  auto publisher = external_runtime->create_publisher<holoscan::CatalogEosPod>(*import_endpoint);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_catalog_subscribers(*publisher, *subscriber, 2s));

  ASSERT_TRUE(publisher->emit(holoscan::CatalogEosPod{.value = 91U}));
  auto sample = subscriber->receive_sample();
  ASSERT_TRUE(sample) << holoscan::runtime_error_message(sample.error());
  EXPECT_EQ(sample->data.value, 91U);
  EXPECT_EQ(sample->metadata.sequence, 1U);

  // Consume the data head before publishing EOS (depth-1 blocking queue).
  ASSERT_TRUE(publisher->close());
  // Key assertion: receive_sample after EOS returns kClosed, not a default-value sample.
  auto end_of_stream = subscriber->receive_sample();
  ASSERT_FALSE(end_of_stream);
  EXPECT_EQ(end_of_stream.error().code, holoscan::ErrorCode::kClosed);
}

// Runtime fieldwise-encodes a custom POD once for all local and external recipients. The
// setup-bound catalog call consumes those canonical bytes without repeating the field walk.
// REQ: CHAN-074
// Topology (kExport, blocking, CatalogEosPod, internal binding path):
//   graph_runtime[InlinePayloadPublisher + canonical bytes] --> external_runtime[subscriber]
TEST(CatalogChannelTest, CanonicalPodBindingPublishesPreencodedBytes) {
  using namespace std::chrono_literals;

  const std::string endpoint_name = unique_endpoint_name();
  auto catalog =
      build_catalog<holoscan::CatalogEosPod>(holoscan::TopicDirection::kExport, endpoint_name);
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint) << holoscan::runtime_error_message(endpoint.error());

  auto graph_runtime = holoscan::ChannelRuntime::create();
  auto external_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(graph_runtime);
  ASSERT_TRUE(external_runtime);
  auto publisher = CatalogChannelBindingAccess::create_payload_publisher(
      *graph_runtime, *endpoint, sizeof(holoscan::CatalogEosPod));
  auto subscriber = external_runtime->create_subscriber<holoscan::CatalogEosPod>(*endpoint);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(wait_for_catalog_subscribers(**publisher, *subscriber, 2s));

  const holoscan::CatalogEosPod value{.value = 0x01020304U};
  const auto identity =
      holoscan::channel_internal::catalog_payload_identity<holoscan::CatalogEosPod>();
  if (!identity.pod) {
    ADD_FAILURE() << "catalog POD identity omitted its fieldwise codec contract";
    return;
  }
  const auto& pod_identity = *identity.pod;
  std::array<std::byte, sizeof(value)> canonical{};
  ASSERT_TRUE(holoscan::channel_internal::PodFieldwiseCodec::encode(
      pod_identity,
      std::as_bytes(std::span<const holoscan::CatalogEosPod, 1U>{&value, 1U}),
      canonical));
  const auto binding = CatalogChannelBindingAccess::bind_canonical_pod_publisher(**publisher);
  ASSERT_TRUE(binding);
  const holoscan::SampleMetadata metadata{
      .trace_id = 0xABCDU,
      .byte_size = sizeof(value),
      .payload_kind = holoscan::PayloadKind::kInlineTyped,
  };
  const auto published = binding.emit_phase(binding.state, canonical, metadata);
  ASSERT_EQ(published.phase, holoscan::channel_internal::InlinePublicationPhase::kAccepted)
      << holoscan::runtime_error_message(published.error);

  auto received = subscriber->receive_sample();
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->data.value, value.value);
  EXPECT_EQ(received->metadata.trace_id, metadata.trace_id);
  EXPECT_EQ(received->metadata.byte_size, sizeof(value));
}

// REQ: CHAN-069
// Topology (kImport, blocking, capture_time + clock_health):
//   external_runtime[publisher] --[H1, H2 back-to-back]--> graph_runtime[receiver]
// Each ordinary receive applies at most one health operation so both are observed in sequence.
TEST(CatalogChannelTest, OrdinaryReceiveAppliesAtMostOneHealthOperationPerCall) {
  using namespace std::chrono_literals;

  const holoscan::TopicOptions options{
      .capture_time =
          holoscan::RequireCaptureTime{.clock = holoscan::ClockKey::named("capture-clock")},
      .clock_health =
          holoscan::ClockHealthOptions{
              .capability = holoscan::ClockHealthCapability::kStatusReports,
              .requirement = holoscan::ClockHealthRequirement::kFreshHealthyReport,
              .max_report_age = 5s,
          },
  };
  const std::string endpoint_name = unique_endpoint_name();
  auto catalog =
      build_catalog<std::uint32_t>(holoscan::TopicDirection::kImport, endpoint_name, options);
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);

  auto external_runtime = holoscan::ChannelRuntime::create();
  auto graph_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(external_runtime);
  ASSERT_TRUE(graph_runtime);
  auto receiver =
      CatalogChannelBindingAccess::create_subscriber<std::uint32_t>(*graph_runtime, *endpoint);
  ASSERT_TRUE(receiver) << holoscan::runtime_error_message(receiver.error());
  auto publisher = external_runtime->create_publisher<std::uint32_t>(*endpoint);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_catalog_subscribers(*publisher, **receiver, 2s));

  std::array<std::byte, sizeof(std::uint32_t)> payload{};
  auto apply_one_health_operation = [&](const std::uint64_t report_sequence)
      -> std::optional<holoscan::channel_internal::CatalogClockHealthState> {
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (std::chrono::steady_clock::now() < deadline) {
      auto received = (*receiver)->receive(payload);
      if (received || received.error().code != holoscan::ErrorCode::kNotReady) {
        return std::nullopt;
      }
      auto state = CatalogChannelBindingAccess::clock_health_state(**receiver);
      if (!state) {
        return std::nullopt;
      }
      if (state->report_sequence == report_sequence) {
        return *state;
      }
      std::this_thread::sleep_for(1ms);
    }
    return std::nullopt;
  };

  ASSERT_TRUE(publisher->report_clock_health(holoscan::ClockHealthReport::healthy()));
  const auto first_state = apply_one_health_operation(1U);
  ASSERT_TRUE(first_state.has_value());
  const auto first_state_value =
      first_state.value_or(holoscan::channel_internal::CatalogClockHealthState{});
  EXPECT_EQ(first_state_value.report_sequence, 1U);

  ASSERT_TRUE(publisher->report_clock_health(holoscan::ClockHealthReport::degraded(
      holoscan::ClockHealthReason::kSynchronizationLockLost)));
  const auto second_state = apply_one_health_operation(2U);
  ASSERT_TRUE(second_state.has_value());
  const auto second_state_value =
      second_state.value_or(holoscan::channel_internal::CatalogClockHealthState{});
  EXPECT_EQ(second_state_value.report_sequence, 2U);
  EXPECT_EQ(second_state_value.assessment, holoscan::ClockHealthAssessment::kDegraded);
}

// REQ: CHAN-069
// REQ: RUNTIME-017
// Topology (kImport, blocking, capture_time + clock_health):
//   external_runtime[publisher] --[H1, S1 pending, H2]--> graph_runtime[receiver]
// The receiver may refuse to accept data (accept_data=false); H2 must wait behind S1.
TEST(CatalogChannelTest, RetainsContiguousDataUntilRuntimeAcceptsItsBoundedClaim) {
  using namespace std::chrono_literals;

  const holoscan::TopicOptions options{
      .capture_time =
          holoscan::RequireCaptureTime{.clock = holoscan::ClockKey::named("capture-clock")},
      .clock_health =
          holoscan::ClockHealthOptions{
              .capability = holoscan::ClockHealthCapability::kStatusReports,
              .requirement = holoscan::ClockHealthRequirement::kFreshHealthyReport,
              .max_report_age = 5s,
          },
  };
  const std::string endpoint_name = unique_endpoint_name();
  auto catalog =
      build_catalog<std::uint32_t>(holoscan::TopicDirection::kImport, endpoint_name, options);
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);

  auto external_runtime = holoscan::ChannelRuntime::create();
  auto graph_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(external_runtime);
  ASSERT_TRUE(graph_runtime);
  auto receiver =
      CatalogChannelBindingAccess::create_subscriber<std::uint32_t>(*graph_runtime, *endpoint);
  ASSERT_TRUE(receiver) << holoscan::runtime_error_message(receiver.error());
  auto publisher = external_runtime->create_publisher<std::uint32_t>(*endpoint);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_catalog_subscribers(*publisher, **receiver, 2s));

  std::array<std::byte, sizeof(std::uint32_t)> payload;
  payload.fill(std::byte{0xa5});
  std::optional<holoscan::Error> receive_error;
  auto receive_next_until = [&](const bool accept_data)
      -> std::optional<holoscan::channel_internal::CatalogReceivedOperation> {
    const auto deadline = std::chrono::steady_clock::now() + 2s;
    while (std::chrono::steady_clock::now() < deadline) {
      auto operation = CatalogChannelBindingAccess::receive_next(**receiver, payload, accept_data);
      if (!operation) {
        receive_error = std::move(operation).error();
        return std::nullopt;
      }
      if (operation->kind == holoscan::channel_internal::CatalogReceivedOperationKind::kNone) {
        std::this_thread::sleep_for(1ms);
        continue;
      }
      return *operation;
    }
    return std::nullopt;
  };

  ASSERT_TRUE(publisher->report_clock_health(holoscan::ClockHealthReport::healthy()));  // H1
  ASSERT_TRUE(publisher->emit_captured(0x01020304U, 101));                              // S1

  const auto health = receive_next_until(false);
  ASSERT_FALSE(receive_error.has_value());
  ASSERT_TRUE(health.has_value());
  const auto health_value = health.value_or(holoscan::channel_internal::CatalogReceivedOperation{});
  ASSERT_EQ(health_value.kind,
            holoscan::channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(health_value.clock_health.publisher_operation_sequence, 1U);
  EXPECT_EQ(health_value.clock_health.report_sequence, 1U);

  const auto pending = receive_next_until(false);
  ASSERT_FALSE(receive_error.has_value());
  ASSERT_TRUE(pending.has_value());
  const auto pending_value =
      pending.value_or(holoscan::channel_internal::CatalogReceivedOperation{});
  EXPECT_EQ(pending_value.kind,
            holoscan::channel_internal::CatalogReceivedOperationKind::kDataPending);
  EXPECT_EQ(payload,
            (std::array<std::byte, sizeof(std::uint32_t)>{
                std::byte{0xa5}, std::byte{0xa5}, std::byte{0xa5}, std::byte{0xa5}}));

  // Key setup: H2 is visible on the private health service but must remain behind S1 until
  // the Runtime accepts S1's typed-storage claim; repeated refusal must not advance sequence.
  ASSERT_TRUE(publisher->report_clock_health(holoscan::ClockHealthReport::unhealthy(
      holoscan::ClockHealthReason::kSynchronizationLockLost)));  // H2
  const auto still_pending = receive_next_until(false);
  ASSERT_FALSE(receive_error.has_value());
  ASSERT_TRUE(still_pending.has_value());
  const auto still_pending_value =
      still_pending.value_or(holoscan::channel_internal::CatalogReceivedOperation{});
  EXPECT_EQ(still_pending_value.kind,
            holoscan::channel_internal::CatalogReceivedOperationKind::kDataPending);
  auto retained_health = CatalogChannelBindingAccess::clock_health_state(**receiver);
  ASSERT_TRUE(retained_health);
  EXPECT_EQ(retained_health->publisher_operation_sequence, 1U);
  EXPECT_EQ(retained_health->report_sequence, 1U);

  // Key assertion: once accept_data=true, S1 is delivered and H2 follows at a later sequence.
  const auto accepted = receive_next_until(true);
  ASSERT_FALSE(receive_error.has_value());
  ASSERT_TRUE(accepted.has_value());
  const auto accepted_value =
      accepted.value_or(holoscan::channel_internal::CatalogReceivedOperation{});
  ASSERT_EQ(accepted_value.kind, holoscan::channel_internal::CatalogReceivedOperationKind::kData);
  EXPECT_EQ(accepted_value.metadata.sequence, 2U);
  EXPECT_EQ(accepted_value.metadata.capture_timestamp_ns, 101);
  EXPECT_EQ(std::bit_cast<std::uint32_t>(payload), 0x01020304U);

  const auto later_health = receive_next_until(false);
  ASSERT_FALSE(receive_error.has_value());
  ASSERT_TRUE(later_health.has_value());
  const auto later_health_value =
      later_health.value_or(holoscan::channel_internal::CatalogReceivedOperation{});
  ASSERT_EQ(later_health_value.kind,
            holoscan::channel_internal::CatalogReceivedOperationKind::kClockHealth);
  EXPECT_EQ(later_health_value.clock_health.publisher_operation_sequence, 3U);
  EXPECT_EQ(later_health_value.clock_health.report_sequence, 2U);
  EXPECT_EQ(later_health_value.clock_health.assessment,
            holoscan::ClockHealthAssessment::kUnhealthy);

  auto drained = CatalogChannelBindingAccess::receive_next(**receiver, payload, false);
  ASSERT_TRUE(drained);
  EXPECT_EQ(drained->kind, holoscan::channel_internal::CatalogReceivedOperationKind::kNone);
}

// REQ: CHAN-069
// Topology (kImport, blocking, clock_health, max_report_age=1ms):
//   external_runtime[publisher] --[10 health+data pairs]--> graph_runtime[receiver]
// Iterates all 10 ClockHealthReason values; each pair increments publisher_operation_sequence
// by 2. An invalid (kHealthy + kMonitorFault) pair must not consume sequence authority.
TEST(CatalogChannelTest, AcceptsExactlyTheTenPortableHealthReasonsWithoutSequenceGaps) {
  using namespace std::chrono_literals;

  const holoscan::TopicOptions options{
      .clock_health =
          holoscan::ClockHealthOptions{
              .capability = holoscan::ClockHealthCapability::kStatusReports,
              .requirement = holoscan::ClockHealthRequirement::kFreshHealthyReport,
              .max_report_age = 1ms,
          },
  };
  const std::string endpoint_name = unique_endpoint_name();
  auto catalog =
      build_catalog<std::uint32_t>(holoscan::TopicDirection::kImport, endpoint_name, options);
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);

  auto external_runtime = holoscan::ChannelRuntime::create();
  auto graph_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(external_runtime);
  ASSERT_TRUE(graph_runtime);
  auto receiver =
      CatalogChannelBindingAccess::create_subscriber<std::uint32_t>(*graph_runtime, *endpoint);
  ASSERT_TRUE(receiver);
  auto publisher = external_runtime->create_publisher<std::uint32_t>(*endpoint);
  ASSERT_TRUE(publisher);
  ASSERT_TRUE(wait_for_catalog_subscribers(*publisher, **receiver, std::chrono::seconds{2}));

  constexpr std::array<holoscan::ClockHealthReason, 10U> kReasons{
      holoscan::ClockHealthReason::kNone,
      holoscan::ClockHealthReason::kSynchronizationLockLost,
      holoscan::ClockHealthReason::kPhaseErrorLimitExceeded,
      holoscan::ClockHealthReason::kRateErrorLimitExceeded,
      holoscan::ClockHealthReason::kCalibrationExpired,
      holoscan::ClockHealthReason::kClockStalled,
      holoscan::ClockHealthReason::kTimestampRegression,
      holoscan::ClockHealthReason::kTimestampSourceFault,
      holoscan::ClockHealthReason::kMonitorFault,
      holoscan::ClockHealthReason::kUnspecified,
  };
  for (std::size_t index = 0U; index < kReasons.size(); ++index) {
    SCOPED_TRACE(index);
    const auto report = index == 0U ? holoscan::ClockHealthReport::healthy()
                                    : holoscan::ClockHealthReport::degraded(kReasons[index]);
    ASSERT_TRUE(publisher->report_clock_health(report));
    ASSERT_TRUE(publisher->emit(static_cast<std::uint32_t>(index)));
    auto sample = receive_erased_until<std::uint32_t>(**receiver, 2s);
    ASSERT_TRUE(sample);
    EXPECT_EQ(sample->metadata.sequence, 2U * (index + 1U));
    auto state = CatalogChannelBindingAccess::clock_health_state(**receiver);
    ASSERT_TRUE(state);
    EXPECT_EQ(state->publisher_operation_sequence, 2U * index + 1U);
    EXPECT_EQ(state->report_sequence, index + 1U);
    EXPECT_EQ(state->reason, kReasons[index]);
  }

  const holoscan::ClockHealthReport invalid{
      .assessment = holoscan::ClockHealthAssessment::kHealthy,
      .reason = holoscan::ClockHealthReason::kMonitorFault,
  };
  auto rejected = publisher->report_clock_health(invalid);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, holoscan::ErrorCode::kInvalidArgument);
  ASSERT_TRUE(publisher->emit(99U));
  auto after_rejection = receive_erased<std::uint32_t>(**receiver);
  ASSERT_TRUE(after_rejection);
  EXPECT_EQ(after_rejection->metadata.sequence, 21U);
}

// REQ: CHAN-080
// Topology (kImport, blocking, clock_health):
//   external_runtime[publisher] --[invalid reports then 3 valid H+S pairs]-->
//     graph_runtime[receiver]
// Invalid native enum values (out-of-range assessment, reason, or mismatched pair) must all
// be rejected before consuming any publisher operation or report sequence authority.
TEST(CatalogChannelTest, DefaultHealthReportsAreSafeAndInvalidNativeValuesDoNotMutateSequence) {
  using namespace std::chrono_literals;

  constexpr holoscan::ClockHealthReport kBraceDefault{};
  constexpr holoscan::ClockHealthReport kNamedHealthy = holoscan::ClockHealthReport::healthy();
  static_assert(kBraceDefault.assessment == holoscan::ClockHealthAssessment::kHealthy);
  static_assert(kBraceDefault.reason == holoscan::ClockHealthReason::kNone);
  static_assert(kNamedHealthy.assessment == kBraceDefault.assessment);
  static_assert(kNamedHealthy.reason == kBraceDefault.reason);
  const holoscan::ClockHealthReport plain_default;
  EXPECT_EQ(plain_default.assessment, holoscan::ClockHealthAssessment::kHealthy);
  EXPECT_EQ(plain_default.reason, holoscan::ClockHealthReason::kNone);

  const holoscan::TopicOptions options{
      .clock_health =
          holoscan::ClockHealthOptions{
              .capability = holoscan::ClockHealthCapability::kStatusReports,
              .requirement = holoscan::ClockHealthRequirement::kFreshHealthyReport,
              .max_report_age = 1ms,
          },
  };
  const std::string endpoint_name = unique_endpoint_name();
  auto catalog =
      build_catalog<std::uint32_t>(holoscan::TopicDirection::kImport, endpoint_name, options);
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);

  auto external_runtime = holoscan::ChannelRuntime::create();
  auto graph_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(external_runtime);
  ASSERT_TRUE(graph_runtime);
  auto receiver =
      CatalogChannelBindingAccess::create_subscriber<std::uint32_t>(*graph_runtime, *endpoint);
  ASSERT_TRUE(receiver);
  auto publisher = external_runtime->create_publisher<std::uint32_t>(*endpoint);
  ASSERT_TRUE(publisher);
  ASSERT_TRUE(wait_for_catalog_subscribers(*publisher, **receiver, 2s));

  // Key setup: three invalid reports submitted; none must consume sequence authority.
  const holoscan::ClockHealthReport invalid_assessment{
      // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
      .assessment = static_cast<holoscan::ClockHealthAssessment>(0xffU),
      .reason = holoscan::ClockHealthReason::kNone,
  };
  const holoscan::ClockHealthReport invalid_reason{
      .assessment = holoscan::ClockHealthAssessment::kUnhealthy,
      // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
      .reason = static_cast<holoscan::ClockHealthReason>(0xffU),
  };
  const holoscan::ClockHealthReport invalid_pair{
      .assessment = holoscan::ClockHealthAssessment::kHealthy,
      .reason = holoscan::ClockHealthReason::kMonitorFault,
  };
  for (const auto report : {invalid_assessment, invalid_reason, invalid_pair}) {
    auto rejected = publisher->report_clock_health(report);
    ASSERT_FALSE(rejected);
    EXPECT_EQ(rejected.error().code, holoscan::ErrorCode::kInvalidArgument);
  }

  // Key assertion: publisher_operation_sequence for valid reports starts at 1, not 4.
  const std::array valid_reports{kBraceDefault, plain_default, kNamedHealthy};
  for (std::size_t index = 0U; index < valid_reports.size(); ++index) {
    SCOPED_TRACE(index);
    ASSERT_TRUE(publisher->report_clock_health(valid_reports[index]));
    ASSERT_TRUE(publisher->emit(static_cast<std::uint32_t>(index)));
    auto sample = receive_erased_until<std::uint32_t>(**receiver, 2s);
    ASSERT_TRUE(sample);
    EXPECT_EQ(sample->metadata.sequence, 2U * (index + 1U));
    auto state = CatalogChannelBindingAccess::clock_health_state(**receiver);
    ASSERT_TRUE(state);
    EXPECT_EQ(state->publisher_operation_sequence, 2U * index + 1U);
    EXPECT_EQ(state->report_sequence, index + 1U);
    EXPECT_EQ(state->assessment, holoscan::ClockHealthAssessment::kHealthy);
    EXPECT_EQ(state->reason, holoscan::ClockHealthReason::kNone);
  }
}

// REQ: CHAN-068
// REQ: EXTERNAL-001
// Verifies four distinct catalog pre-creation rejections for a kImport endpoint:
// wrong role, wrong type, zero-timeout, and excessive-timeout (>1s max).
// REQ: CHAN-137
TEST(CatalogChannelTest, RejectsDirectionTypeAndAttachBoundBeforeEndpointCreation) {
  using namespace std::chrono_literals;

  const std::string endpoint_name = unique_endpoint_name();
  auto catalog = build_catalog<std::uint32_t>(holoscan::TopicDirection::kImport, endpoint_name);
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime);

  auto wrong_role = runtime->create_subscriber<std::uint32_t>(*endpoint);
  ASSERT_FALSE(wrong_role);
  EXPECT_EQ(wrong_role.error().code, holoscan::ErrorCode::kInvalidArgument);
  EXPECT_EQ(wrong_role.error().message, "ENDPOINT_CLIENT_ROLE_MISMATCH");

  auto wrong_type = runtime->create_publisher<std::uint64_t>(*endpoint);
  ASSERT_FALSE(wrong_type);
  EXPECT_EQ(wrong_type.error().code, holoscan::ErrorCode::kIncompatible);
  EXPECT_EQ(wrong_type.error().message, "ENDPOINT_TYPE_MISMATCH");

  auto zero_timeout = runtime->create_publisher<std::uint32_t>(
      *endpoint, holoscan::ChannelAttachOptions{.timeout = 0ns});
  ASSERT_FALSE(zero_timeout);
  EXPECT_EQ(zero_timeout.error().code, holoscan::ErrorCode::kInvalidArgument);
  EXPECT_EQ(zero_timeout.error().message, "ENDPOINT_ATTACH_BOUND_INVALID");

  auto excessive_timeout = runtime->create_publisher<std::uint32_t>(
      *endpoint, holoscan::ChannelAttachOptions{.timeout = 3s});
  ASSERT_FALSE(excessive_timeout);
  EXPECT_EQ(excessive_timeout.error().code, holoscan::ErrorCode::kInvalidArgument);
  EXPECT_EQ(excessive_timeout.error().message, "ENDPOINT_ATTACH_BOUND_INVALID");
}

// REQ: CHAN-068
// REQ: EXTERNAL-001
// Verifies that kOptional peer policy with timeout=1ns returns immediately when no peer
// is present (import publisher and export subscriber, both with no counterpart).
TEST(CatalogChannelTest, OptionalAttachReturnsWithoutWaitingForAnAbsentPeer) {
  using namespace std::chrono_literals;

  const holoscan::TopicOptions options{
      .attach =
          holoscan::TopicAttachPolicy{
              .peer = holoscan::PeerRequirement::kOptional,
              .max_wait = 1s,
          },
  };

  const std::string import_name = unique_endpoint_name();
  auto import_catalog =
      build_catalog<std::uint32_t>(holoscan::TopicDirection::kImport, import_name, options);
  ASSERT_TRUE(import_catalog) << holoscan::runtime_error_message(import_catalog.error());
  auto import_endpoint = import_catalog->endpoint(import_name);
  ASSERT_TRUE(import_endpoint);

  const std::string export_name = unique_endpoint_name();
  auto export_catalog =
      build_catalog<std::uint32_t>(holoscan::TopicDirection::kExport, export_name, options);
  ASSERT_TRUE(export_catalog) << holoscan::runtime_error_message(export_catalog.error());
  auto export_endpoint = export_catalog->endpoint(export_name);
  ASSERT_TRUE(export_endpoint);

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime);
  auto publisher = runtime->create_publisher<std::uint32_t>(
      *import_endpoint, holoscan::ChannelAttachOptions{.timeout = 1ns});
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  auto subscriber = runtime->create_subscriber<std::uint32_t>(
      *export_endpoint, holoscan::ChannelAttachOptions{.timeout = 1ns});
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
}

// REQ: CHAN-068
// REQ: EXTERNAL-001
// Verifies kRequired peer policy honors the caller's shorter ChannelAttachOptions timeout
// (20ms) rather than the catalog's max_wait (1s); overall wall time must stay well under 1s.
TEST(CatalogChannelTest, RequiredPublisherUsesTheCallerShortenedPeerWaitBound) {
  using namespace std::chrono_literals;

  const holoscan::TopicOptions options{
      .attach =
          holoscan::TopicAttachPolicy{
              .peer = holoscan::PeerRequirement::kRequired,
              .max_wait = 1s,
          },
  };
  const std::string endpoint_name = unique_endpoint_name();
  auto catalog =
      build_catalog<std::uint32_t>(holoscan::TopicDirection::kImport, endpoint_name, options);
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime);

  const auto started = std::chrono::steady_clock::now();
  auto publisher = runtime->create_publisher<std::uint32_t>(
      *endpoint, holoscan::ChannelAttachOptions{.timeout = 20ms});
  const auto elapsed = std::chrono::steady_clock::now() - started;
  ASSERT_FALSE(publisher);
  EXPECT_EQ(publisher.error().code, holoscan::ErrorCode::kTimeout);
  EXPECT_EQ(publisher.error().message, "ENDPOINT_REQUIRED_PEER_TIMEOUT");
  EXPECT_LT(elapsed, 500ms);
}

// REQ: CHAN-068
// kExport subscriber with kRequired peer and caller timeout=5ms; no publisher exists.
TEST(CatalogChannelTest, RequiredSubscriberTimesOutWhenItsPublisherIsAbsent) {
  using namespace std::chrono_literals;

  const holoscan::TopicOptions options{
      .attach =
          holoscan::TopicAttachPolicy{
              .peer = holoscan::PeerRequirement::kRequired,
              .max_wait = 50ms,
          },
  };
  const std::string endpoint_name = unique_endpoint_name();
  auto catalog =
      build_catalog<std::uint32_t>(holoscan::TopicDirection::kExport, endpoint_name, options);
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime);

  auto subscriber = runtime->create_subscriber<std::uint32_t>(
      *endpoint, holoscan::ChannelAttachOptions{.timeout = 5ms});
  ASSERT_FALSE(subscriber);
  EXPECT_EQ(subscriber.error().code, holoscan::ErrorCode::kTimeout);
  EXPECT_EQ(subscriber.error().message, "ENDPOINT_REQUIRED_PEER_TIMEOUT");
}

// REQ: CHAN-068
// REQ: EXTERNAL-001
// Topology (import + export, kRequired peer attach, peer pre-created):
//   import_graph_runtime[subscriber] <-- import_client_runtime[publisher, timeout=250ms]
//   export_graph_runtime[publisher]  --> export_client_runtime[subscriber, timeout=250ms]
// Graph-side endpoints are created first so each client-side attach finds its peer.
TEST(CatalogChannelTest, RequiredPublisherAndSubscriberAttachWhenTheirPeerAlreadyExists) {
  using namespace std::chrono_literals;

  const holoscan::TopicOptions options{
      .attach =
          holoscan::TopicAttachPolicy{
              .peer = holoscan::PeerRequirement::kRequired,
              .max_wait = 500ms,
          },
  };

  const std::string import_name = unique_endpoint_name();
  auto import_catalog =
      build_catalog<std::uint32_t>(holoscan::TopicDirection::kImport, import_name, options);
  ASSERT_TRUE(import_catalog) << holoscan::runtime_error_message(import_catalog.error());
  auto import_endpoint = import_catalog->endpoint(import_name);
  ASSERT_TRUE(import_endpoint);
  auto import_graph_runtime = holoscan::ChannelRuntime::create();
  auto import_client_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(import_graph_runtime);
  ASSERT_TRUE(import_client_runtime);
  auto graph_subscriber = CatalogChannelBindingAccess::create_subscriber<std::uint32_t>(
      *import_graph_runtime, *import_endpoint);
  ASSERT_TRUE(graph_subscriber) << holoscan::runtime_error_message(graph_subscriber.error());
  auto client_publisher = import_client_runtime->create_publisher<std::uint32_t>(
      *import_endpoint, holoscan::ChannelAttachOptions{.timeout = 250ms});
  ASSERT_TRUE(client_publisher) << holoscan::runtime_error_message(client_publisher.error());

  const std::string export_name = unique_endpoint_name();
  auto export_catalog =
      build_catalog<std::uint32_t>(holoscan::TopicDirection::kExport, export_name, options);
  ASSERT_TRUE(export_catalog) << holoscan::runtime_error_message(export_catalog.error());
  auto export_endpoint = export_catalog->endpoint(export_name);
  ASSERT_TRUE(export_endpoint);
  auto export_graph_runtime = holoscan::ChannelRuntime::create();
  auto export_client_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(export_graph_runtime);
  ASSERT_TRUE(export_client_runtime);
  auto graph_publisher = CatalogChannelBindingAccess::create_publisher<std::uint32_t>(
      *export_graph_runtime, *export_endpoint);
  ASSERT_TRUE(graph_publisher) << holoscan::runtime_error_message(graph_publisher.error());
  auto client_subscriber = export_client_runtime->create_subscriber<std::uint32_t>(
      *export_endpoint, holoscan::ChannelAttachOptions{.timeout = 250ms});
  ASSERT_TRUE(client_subscriber) << holoscan::runtime_error_message(client_subscriber.error());
}

// REQ: CHAN-068
// Verifies that all ChannelPublisher and ChannelSubscriber operations return kClosed after
// the handle has been moved from. The active moved-to handle remains fully operational.
TEST(CatalogChannelTest, MovedFromCatalogClientsReportClosed) {
  const std::string endpoint_name = unique_endpoint_name();
  auto catalog = build_catalog<std::uint32_t>(holoscan::TopicDirection::kImport, endpoint_name);
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime);
  auto publisher = runtime->create_publisher<std::uint32_t>(*endpoint);
  ASSERT_TRUE(publisher);

  auto active = std::move(*publisher);
  auto emitted = publisher->emit(1U);
  ASSERT_FALSE(emitted);
  EXPECT_EQ(emitted.error().code, holoscan::ErrorCode::kClosed);
  emitted = publisher->emit_captured(
      1U,
      1,
      {.capture_time =
           holoscan::TimePoint{.timestamp_ns = 1, .clock_id = holoscan::ClockId{.value = 1U}}});
  ASSERT_FALSE(emitted);
  EXPECT_EQ(emitted.error().code, holoscan::ErrorCode::kClosed);
  auto refreshed = publisher->update_connections();
  ASSERT_FALSE(refreshed);
  EXPECT_EQ(refreshed.error().code, holoscan::ErrorCode::kClosed);
  auto capture = publisher->make_capture_time(1);
  ASSERT_FALSE(capture);
  EXPECT_EQ(capture.error().code, holoscan::ErrorCode::kClosed);
  auto health = publisher->report_clock_health(holoscan::ClockHealthReport::healthy());
  ASSERT_FALSE(health);
  EXPECT_EQ(health.error().code, holoscan::ErrorCode::kClosed);
  EXPECT_FALSE(publisher->has_subscribers());
  EXPECT_TRUE(active.update_connections());

  const std::string export_name = unique_endpoint_name();
  auto export_catalog =
      build_catalog<std::uint32_t>(holoscan::TopicDirection::kExport, export_name);
  ASSERT_TRUE(export_catalog) << holoscan::runtime_error_message(export_catalog.error());
  auto export_endpoint = export_catalog->endpoint(export_name);
  ASSERT_TRUE(export_endpoint);
  auto graph_runtime = holoscan::ChannelRuntime::create();
  auto client_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(graph_runtime);
  ASSERT_TRUE(client_runtime);
  auto graph_publisher = CatalogChannelBindingAccess::create_publisher<std::uint32_t>(
      *graph_runtime, *export_endpoint);
  ASSERT_TRUE(graph_publisher) << holoscan::runtime_error_message(graph_publisher.error());
  auto subscriber = client_runtime->create_subscriber<std::uint32_t>(*export_endpoint);
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  ASSERT_TRUE(
      wait_for_catalog_publishers(**graph_publisher, *subscriber, std::chrono::milliseconds{500}));
  EXPECT_TRUE(subscriber->has_publishers());

  auto active_subscriber = std::move(*subscriber);
  EXPECT_FALSE(subscriber->has_publishers());
  auto moved_from_ready = subscriber->has_samples();
  ASSERT_FALSE(moved_from_ready);
  EXPECT_EQ(moved_from_ready.error().code, holoscan::ErrorCode::kClosed);
  EXPECT_TRUE(active_subscriber.has_publishers());
}

}  // namespace

// A monitored catalog endpoint applies at most one private operation per receive, so a clock
// health report published ahead of the data makes the first receive report kNotReady even though
// the endpoint is genuinely ready. This is the readiness race the wait API must absorb: a wake and
// a receive are not atomic, and readiness is only a hint. The shared retry loop must deliver the
// data sample from a single call rather than surfacing the intermediate kNotReady.
//
// Note this exercises the loop through the erased binding subscriber rather than the typed facade.
// Clock health is import-direction only, while ChannelRuntime::create_subscriber<T> attaches a
// catalog endpoint as kExport, so no typed ChannelSubscriber<T> can ever carry clock health.
// Topology (import catalog, clock health, event wake forced true by the catalog):
//   publisher --[health report, then 17U]--> erased catalog subscriber
// Key assertion: one call returns the data sample; never kNotReady, never kTimeout.
// REQ: CHAN-142
TEST(CatalogChannelTest, CatalogWaitRetriesAfterHealthOnlyOperation) {
  using namespace std::chrono_literals;

  const holoscan::TopicOptions options{
      .clock_health =
          holoscan::ClockHealthOptions{
              .capability = holoscan::ClockHealthCapability::kStatusReports,
              .requirement = holoscan::ClockHealthRequirement::kFreshHealthyReport,
              .max_report_age = 5s,
          },
  };
  const std::string endpoint_name = unique_endpoint_name();
  auto catalog =
      build_catalog<std::uint32_t>(holoscan::TopicDirection::kImport, endpoint_name, options);
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);

  auto external_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << holoscan::runtime_error_message(external_runtime.error());
  auto graph_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(graph_runtime) << holoscan::runtime_error_message(graph_runtime.error());

  auto receiver =
      CatalogChannelBindingAccess::create_subscriber<std::uint32_t>(*graph_runtime, *endpoint);
  ASSERT_TRUE(receiver) << holoscan::runtime_error_message(receiver.error());
  auto publisher = external_runtime->create_publisher<std::uint32_t>(*endpoint);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_catalog_subscribers(*publisher, **receiver, 2s));

  // Health first, then data, on one shared operation sequence. A bare receive would burn the
  // health operation and report kNotReady, which is why receive_erased_until has to poll.
  ASSERT_TRUE(publisher->report_clock_health(holoscan::ClockHealthReport::healthy()));
  auto emitted = publisher->emit(17U);
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());

  auto received = holoscan::channel_internal::wait_and_receive_next(
      **receiver, std::chrono::steady_clock::now() + 5s, nullptr, [&] {
        return receive_erased<std::uint32_t>(**receiver);
      });
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  EXPECT_EQ(received->data, 17U);
}

// A readiness hint must not buy an unbounded retry. The health-then-data sequence makes
// has_samples() true while the first receive still reports kNotReady, so the retry loop takes its
// readiness branch. That branch must still honour the caller's deadline: given one already
// expired, the call has to give up rather than spend another receive attempt on it.
// Topology (catalog import, clock health): publisher --[health report, then 17U]--> erased
// subscriber Key assertion: an expired deadline yields kTimeout, not the sample a further retry
// would find. REQ: CHAN-142
TEST(CatalogChannelTest, CatalogWaitHonoursAnExpiredDeadlineOnTheReadinessPath) {
  using namespace std::chrono_literals;

  const holoscan::TopicOptions options{
      .clock_health =
          holoscan::ClockHealthOptions{
              .capability = holoscan::ClockHealthCapability::kStatusReports,
              .requirement = holoscan::ClockHealthRequirement::kFreshHealthyReport,
              .max_report_age = 5s,
          },
  };
  const std::string endpoint_name = unique_endpoint_name();
  auto catalog =
      build_catalog<std::uint32_t>(holoscan::TopicDirection::kImport, endpoint_name, options);
  ASSERT_TRUE(catalog) << holoscan::runtime_error_message(catalog.error());
  auto endpoint = catalog->endpoint(endpoint_name);
  ASSERT_TRUE(endpoint);

  auto external_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << holoscan::runtime_error_message(external_runtime.error());
  auto graph_runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(graph_runtime) << holoscan::runtime_error_message(graph_runtime.error());

  auto receiver =
      CatalogChannelBindingAccess::create_subscriber<std::uint32_t>(*graph_runtime, *endpoint);
  ASSERT_TRUE(receiver) << holoscan::runtime_error_message(receiver.error());
  auto publisher = external_runtime->create_publisher<std::uint32_t>(*endpoint);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  ASSERT_TRUE(wait_for_catalog_subscribers(*publisher, **receiver, 2s));

  ASSERT_TRUE(publisher->report_clock_health(holoscan::ClockHealthReport::healthy()));
  auto emitted = publisher->emit(17U);
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());

  // A probe, not a wait. Both publications above are already asserted to have reached a recipient
  // -- emit reports kPublicationIndeterminate at zero recipients and report_clock_health demands
  // exactly one -- and the transport delivers into the subscriber's queue inline, so readiness is
  // true by the time emit returns. Pin it anyway: this is the precondition the branch under test
  // needs, and without it the call below takes the ordinary expired-deadline exit and proves
  // nothing while still passing.
  auto queued = (**receiver).has_samples();
  ASSERT_TRUE(queued) << holoscan::runtime_error_message(queued.error());
  ASSERT_TRUE(*queued) << "the readiness branch needs the queued work visible before the call";

  // Already expired. Without a deadline test on the readiness branch the loop would consume the
  // health operation, take the hint, retry, and deliver 17U from a call that had no time left.
  const auto expired = std::chrono::steady_clock::now() - 1s;
  auto received = holoscan::channel_internal::wait_and_receive_next(
      **receiver, expired, nullptr, [&] { return receive_erased<std::uint32_t>(**receiver); });

  ASSERT_FALSE(received) << "an expired deadline must bound the retry loop";
  EXPECT_EQ(received.error().code, holoscan::ErrorCode::kTimeout);

  // Witness which exit ran. The readiness branch and the ordinary expired-deadline exit both
  // return a bare kTimeout, so the code above cannot tell them apart; only the readiness branch
  // consumes the health operation and leaves the sample behind.
  auto health_state = CatalogChannelBindingAccess::clock_health_state(**receiver);
  ASSERT_TRUE(health_state) << holoscan::runtime_error_message(health_state.error());
  EXPECT_TRUE(health_state->has_report) << "the retry loop never applied the health operation";
  auto staged = receive_erased_until<std::uint32_t>(**receiver, 2s);
  ASSERT_TRUE(staged) << holoscan::runtime_error_message(staged.error());
  EXPECT_EQ(staged->data, 17U) << "the expired call must not have consumed the sample";
}
