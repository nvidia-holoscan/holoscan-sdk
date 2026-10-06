/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "channel_catalog_binding.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/schema/schema_package.hpp>

#include "catalog_attach_wait.hpp"
#include "channel_common.hpp"
#include "channel_inline_payload.hpp"
#include "channel_readiness_handle.hpp"
#include "channel_runtime_impl.hpp"
#include "endpoint_catalog_state.hpp"
#include "pod_codec.hpp"
#include "process_identity.hpp"
#include "schema_envelope_codec.hpp"

namespace holoscan {
namespace channel_internal {
namespace {

// Version 3 grows the data-attribution frame by one 8-byte producer-generation field
// (magic@0, version@4, kind@6, health_report_sequence@8, producer_generation@16). This is a
// coordinated pre-GA ABI break: version-2 frames are rejected outright with kProtocolError and
// no negotiation or compatibility fallback exists.
constexpr std::size_t kHealthControlBytes = 40U;
constexpr std::uint32_t kDataAttributionMagic = 0x44434835U;  // "DCH5"
constexpr std::uint16_t kDataAttributionVersion = 3U;
constexpr std::uint32_t kHealthControlMagic = 0x48434835U;  // "HCH5"
// Version 2 repurposes the epoch field at offset 8 as the publisher's actual producer
// generation (previously hardcoded to the initial attachment-local epoch). Same coordinated
// break: version-1 frames are rejected with kProtocolError.
constexpr std::uint16_t kHealthControlVersion = 2U;
constexpr std::uint16_t kHealthControlKind = 1U;

enum class DataAttributionKind : std::uint16_t {
  kData = 1U,
  kEndOfStream = 2U,
};

struct DataAttribution {
  DataAttributionKind kind{DataAttributionKind::kData};
  std::uint64_t health_report_sequence{};
  std::uint64_t producer_generation{};
};

[[nodiscard]] Error catalog_error(ErrorCode code, const char* token) {
  return Error{code, token};
}

[[nodiscard]] bool closed_clock_health_enum(const ClockHealthCapability value) noexcept {
  return value == ClockHealthCapability::kUnavailable ||
         value == ClockHealthCapability::kStatusReports;
}

[[nodiscard]] bool closed_clock_health_enum(const ClockHealthRequirement value) noexcept {
  return value == ClockHealthRequirement::kNotRequired ||
         value == ClockHealthRequirement::kFreshHealthyReport;
}

[[nodiscard]] bool valid_clock_health_contract(const ClockHealthOptions& options) noexcept {
  if (!closed_clock_health_enum(options.capability) ||
      !closed_clock_health_enum(options.requirement)) {
    return false;
  }
  if (options.capability == ClockHealthCapability::kUnavailable &&
      options.requirement == ClockHealthRequirement::kNotRequired) {
    return !options.max_report_age.has_value();
  }
  return options.capability == ClockHealthCapability::kStatusReports &&
         options.requirement == ClockHealthRequirement::kFreshHealthyReport &&
         options.max_report_age.has_value() && options.max_report_age->count() > 0;
}

[[nodiscard]] bool valid_health_report(const ClockHealthReport report) noexcept {
  const auto assessment = static_cast<std::uint8_t>(report.assessment);
  const auto reason = static_cast<std::uint8_t>(report.reason);
  if (assessment > static_cast<std::uint8_t>(ClockHealthAssessment::kUnhealthy) ||
      reason > static_cast<std::uint8_t>(ClockHealthReason::kUnspecified)) {
    return false;
  }
  return report.assessment == ClockHealthAssessment::kHealthy
             ? report.reason == ClockHealthReason::kNone
             : report.reason != ClockHealthReason::kNone;
}

[[nodiscard]] constexpr std::size_t catalog_wire_payload_size(
    const PayloadIdentityRecord& identity, const std::size_t payload_capacity = 0U) noexcept {
  const auto contract = catalog_inline_payload_contract(identity, payload_capacity);
  return contract ? static_cast<std::size_t>(contract->wire_payload_capacity_bytes) : 0U;
}

[[nodiscard]] constexpr std::size_t catalog_native_payload_size(
    const PayloadIdentityRecord& identity, const std::size_t payload_capacity = 0U) noexcept {
  const auto contract = catalog_inline_payload_contract(identity, payload_capacity);
  return contract ? static_cast<std::size_t>(contract->native_payload_capacity_bytes) : 0U;
}

[[nodiscard]] expected<void, Error> validate_payload_attachment(
    const detail::ResolvedServiceStaticRequirements& service,
    const PayloadIdentityRecord& attaching_identity) {
  const PayloadIdentityRecord& frozen_identity = service.payload_identity;
  if (frozen_identity.kind != attaching_identity.kind) {
    return make_unexpected(catalog_error(ErrorCode::kIncompatible, "ENDPOINT_TYPE_MISMATCH"));
  }
  if (frozen_identity.kind == PayloadIdentityKind::kSchema) {
    if (frozen_identity.payload_kind != PayloadKind::kSerializedBytes ||
        attaching_identity.payload_kind != PayloadKind::kSerializedBytes ||
        !frozen_identity.schema || !attaching_identity.schema || !service.schema_identity ||
        frozen_identity.scalar || attaching_identity.scalar || frozen_identity.pod ||
        attaching_identity.pod ||
        !schema_identity_equal(*frozen_identity.schema, *service.schema_identity) ||
        !schema_identity_equal(*frozen_identity.schema, *attaching_identity.schema)) {
      return make_unexpected(catalog_error(ErrorCode::kIncompatible, "ENDPOINT_TYPE_MISMATCH"));
    }
    return {};
  }
  if (frozen_identity.payload_kind != PayloadKind::kInlineTyped ||
      attaching_identity.payload_kind != PayloadKind::kInlineTyped) {
    return make_unexpected(catalog_error(ErrorCode::kIncompatible, "ENDPOINT_TYPE_MISMATCH"));
  }
  if (frozen_identity.kind == PayloadIdentityKind::kScalar && frozen_identity.scalar &&
      attaching_identity.scalar &&
      frozen_identity.scalar->kind == attaching_identity.scalar->kind && !frozen_identity.pod &&
      !attaching_identity.pod && !frozen_identity.schema && !attaching_identity.schema) {
    return {};
  }
  if (frozen_identity.kind != PayloadIdentityKind::kPod || !frozen_identity.pod ||
      !attaching_identity.pod || !service.pod_identity || frozen_identity.scalar ||
      attaching_identity.scalar || frozen_identity.schema || attaching_identity.schema) {
    return make_unexpected(catalog_error(ErrorCode::kIncompatible, "ENDPOINT_TYPE_MISMATCH"));
  }

  auto attaching_evidence = derive_pod_contract_evidence(*attaching_identity.pod);
  if (!attaching_evidence) {
    return make_unexpected(std::move(attaching_evidence).error());
  }
  const PodContractEvidence frozen_evidence{
      .identity_digest = service.pod_identity->digest_bytes,
      .key = service.pod_identity->key,
      .layout_digest = service.pod_identity->layout_digest_bytes,
  };
  return validate_pod_attachment_contract(
      *frozen_identity.pod, frozen_evidence, *attaching_identity.pod, *attaching_evidence);
}

[[nodiscard]] expected<void, Error> encode_catalog_payload(const PayloadIdentityRecord& identity,
                                                           std::span<const std::byte> native,
                                                           std::span<std::byte> wire) {
  if (identity.kind == PayloadIdentityKind::kScalar && identity.scalar) {
    return CatalogScalarCodec::encode(identity.scalar->kind, native, wire);
  }
  if (identity.kind == PayloadIdentityKind::kPod && identity.pod) {
    // Catalog attachment validates and freezes POD geometry before constructing the endpoint.
    return PodFieldwiseCodec::encode_prevalidated(*identity.pod, native, wire);
  }
  return make_unexpected(catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_TYPE_MISMATCH"));
}

[[nodiscard]] expected<void, Error> decode_catalog_payload(const PayloadIdentityRecord& identity,
                                                           std::span<const std::byte> wire,
                                                           std::span<std::byte> native) {
  if (identity.kind == PayloadIdentityKind::kScalar && identity.scalar) {
    return CatalogScalarCodec::decode(identity.scalar->kind, wire, native);
  }
  if (identity.kind == PayloadIdentityKind::kPod && identity.pod) {
    // Catalog attachment validates and freezes POD geometry before constructing the endpoint.
    return PodFieldwiseCodec::decode_prevalidated(*identity.pod, wire, native);
  }
  return make_unexpected(catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_TYPE_MISMATCH"));
}

template <typename Integer>
void store_little(std::span<std::byte> destination, std::size_t offset, Integer value) noexcept {
  static_assert(std::is_unsigned_v<Integer>);
  for (std::size_t index = 0U; index < sizeof(Integer); ++index) {
    destination[offset + index] =
        static_cast<std::byte>((value >> (index * 8U)) & static_cast<Integer>(0xffU));
  }
}

template <typename Integer>
[[nodiscard]] Integer load_little(std::span<const std::byte> source, std::size_t offset) noexcept {
  static_assert(std::is_unsigned_v<Integer>);
  Integer result{};
  for (std::size_t index = 0U; index < sizeof(Integer); ++index) {
    result |= static_cast<Integer>(std::to_integer<std::uint8_t>(source[offset + index]))
              << (index * 8U);
  }
  return result;
}

void corrupt_schema_frame(std::span<std::byte> frame,
                          const CatalogSchemaFrameCorruption corruption) noexcept {
  constexpr std::size_t kEnvelopeOffset = kCatalogDataAttributionBytes;
  constexpr std::size_t kSchemaIdOffset = kEnvelopeOffset + 8U;
  constexpr std::size_t kPayloadSizeOffset = kEnvelopeOffset + 24U;
  constexpr std::size_t kEndpointGenerationOffset = kEnvelopeOffset + 32U;
  constexpr std::size_t kPayloadOffset = kEnvelopeOffset + kSchemaEnvelopeHeaderSize;

  switch (corruption) {
    case CatalogSchemaFrameCorruption::kMalformedEnvelope:
      store_little<std::uint32_t>(frame, kEnvelopeOffset + 4U, 0U);
      break;
    case CatalogSchemaFrameCorruption::kSchemaId:
      frame[kSchemaIdOffset] ^= std::byte{0xffU};
      break;
    case CatalogSchemaFrameCorruption::kPayloadSize:
      store_little<std::uint32_t>(frame, kPayloadSizeOffset, 0U);
      break;
    case CatalogSchemaFrameCorruption::kEndpointGeneration:
      store_little<std::uint64_t>(
          frame, kEndpointGenerationOffset, kInitialSchemaEndpointGeneration + 1U);
      break;
    case CatalogSchemaFrameCorruption::kPayloadVerifier:
      std::fill(frame.begin() + kPayloadOffset, frame.end(), std::byte{0xffU});
      break;
    case CatalogSchemaFrameCorruption::kEnvelopeAbi:
      store_little<std::uint32_t>(frame, kEnvelopeOffset, 0U);
      break;
  }
}

void encode_data_attribution(std::span<std::byte> wire, const DataAttributionKind kind,
                             const std::uint64_t health_report_sequence,
                             const std::uint64_t producer_generation) noexcept {
  store_little<std::uint32_t>(wire, 0U, kDataAttributionMagic);
  store_little<std::uint16_t>(wire, 4U, kDataAttributionVersion);
  store_little<std::uint16_t>(wire, 6U, static_cast<std::uint16_t>(kind));
  store_little<std::uint64_t>(wire, 8U, health_report_sequence);
  store_little<std::uint64_t>(wire, 16U, producer_generation);
}

[[nodiscard]] expected<DataAttribution, Error> decode_data_attribution(
    std::span<const std::byte> wire) {
  // Reject any non-current frame version outright. There is deliberately no negotiation or
  // compatibility fallback for the superseded version-2 frame (pre-GA coordinated break).
  if (wire.size() < kCatalogDataAttributionBytes ||
      load_little<std::uint32_t>(wire, 0U) != kDataAttributionMagic ||
      load_little<std::uint16_t>(wire, 4U) != kDataAttributionVersion) {
    return make_unexpected(
        catalog_error(ErrorCode::kProtocolError, "ENDPOINT_DATA_ATTRIBUTION_INVALID"));
  }
  const auto kind = static_cast<DataAttributionKind>(load_little<std::uint16_t>(wire, 6U));
  if (kind != DataAttributionKind::kData && kind != DataAttributionKind::kEndOfStream) {
    return make_unexpected(
        catalog_error(ErrorCode::kProtocolError, "ENDPOINT_DATA_ATTRIBUTION_KIND_UNSUPPORTED"));
  }
  const std::uint64_t producer_generation = load_little<std::uint64_t>(wire, 16U);
  if (producer_generation == 0U) {
    return make_unexpected(
        catalog_error(ErrorCode::kProtocolError, "ENDPOINT_DATA_ATTRIBUTION_INVALID"));
  }
  return DataAttribution{
      .kind = kind,
      .health_report_sequence = load_little<std::uint64_t>(wire, 8U),
      .producer_generation = producer_generation,
  };
}

[[nodiscard]] expected<void, Error> validate_received_data_metadata(
    const SampleMetadata& metadata, const PayloadIdentityRecord& identity,
    const std::size_t payload_capacity, const ClockId capture_clock_id,
    const std::uint64_t catalog_clock_count, const DataAttributionKind kind) {
  if (metadata.sequence == 0U) {
    return make_unexpected(
        catalog_error(ErrorCode::kProtocolError, "ENDPOINT_OPERATION_SEQUENCE_INVALID"));
  }
  // EOS is an ordered control operation, not a sample captured in the endpoint clock domain.
  // Requiring the data capture clock here would either fabricate a timestamp or make a valid
  // close indistinguishable from malformed data on every capture-time-required endpoint.
  const bool capture_coordinate_valid =
      kind == DataAttributionKind::kEndOfStream
          ? !metadata.source_clock_id.valid() && metadata.capture_timestamp_ns == 0
      : capture_clock_id.valid()
          ? metadata.source_clock_id == capture_clock_id
          : !metadata.source_clock_id.valid() && metadata.capture_timestamp_ns == 0;
  const bool schema_payload = identity.kind == PayloadIdentityKind::kSchema && identity.schema;
  const bool byte_size_valid =
      kind == DataAttributionKind::kEndOfStream ? metadata.byte_size == 0U
      : schema_payload ? metadata.byte_size != 0U && metadata.byte_size <= payload_capacity
                       : metadata.byte_size == catalog_wire_payload_size(identity);
  const PayloadKind expected_kind = schema_payload && kind == DataAttributionKind::kData
                                        ? PayloadKind::kSerializedBytes
                                        : PayloadKind::kInlineTyped;
  // A control close carries no publication instant of its own, so it stays canonically absent.
  const bool publish_coordinate_valid =
      kind == DataAttributionKind::kEndOfStream
          ? !metadata.publish_clock_id.valid() && metadata.publish_timestamp_ns == 0
          : publication_coordinate_declared(metadata, catalog_clock_count);
  if (metadata.version.major != 1U || metadata.version.minor != 0U ||
      (static_cast<std::uint32_t>(metadata.flags) & ~0x3U) != 0U || !byte_size_valid ||
      metadata.payload_kind != expected_kind || !publish_coordinate_valid ||
      !capture_coordinate_valid) {
    return make_unexpected(
        catalog_error(ErrorCode::kProtocolError, "ENDPOINT_SAMPLE_METADATA_INVALID"));
  }
  return {};
}

[[nodiscard]] std::array<std::byte, kHealthControlBytes> encode_health_control(
    std::uint64_t producer_generation, std::uint64_t operation_sequence,
    std::uint64_t report_sequence, ClockHealthReport report) noexcept {
  std::array<std::byte, kHealthControlBytes> wire{};
  store_little<std::uint32_t>(wire, 0U, kHealthControlMagic);
  store_little<std::uint16_t>(wire, 4U, kHealthControlVersion);
  store_little<std::uint16_t>(wire, 6U, kHealthControlKind);
  store_little<std::uint64_t>(wire, 8U, producer_generation);
  store_little<std::uint64_t>(wire, 16U, operation_sequence);
  store_little<std::uint64_t>(wire, 24U, report_sequence);
  wire[32U] = static_cast<std::byte>(static_cast<std::uint8_t>(report.assessment));
  wire[33U] = static_cast<std::byte>(static_cast<std::uint8_t>(report.reason));
  return wire;
}

[[nodiscard]] expected<CatalogClockHealthState, Error> decode_health_control(
    std::span<const std::byte> wire) {
  if (wire.size() != kHealthControlBytes ||
      load_little<std::uint32_t>(wire, 0U) != kHealthControlMagic ||
      load_little<std::uint16_t>(wire, 4U) != kHealthControlVersion ||
      load_little<std::uint16_t>(wire, 6U) != kHealthControlKind) {
    return make_unexpected(
        catalog_error(ErrorCode::kProtocolError, "ENDPOINT_CLOCK_HEALTH_CONTROL_INVALID"));
  }
  const auto assessment =
      static_cast<ClockHealthAssessment>(std::to_integer<std::uint8_t>(wire[32U]));
  const auto reason = static_cast<ClockHealthReason>(std::to_integer<std::uint8_t>(wire[33U]));
  const ClockHealthReport report{.assessment = assessment, .reason = reason};
  if (static_cast<std::uint8_t>(reason) >
      static_cast<std::uint8_t>(ClockHealthReason::kUnspecified)) {
    return make_unexpected(
        catalog_error(ErrorCode::kProtocolError, "ENDPOINT_CLOCK_HEALTH_REASON_UNSUPPORTED"));
  }
  const std::uint64_t producer_generation = load_little<std::uint64_t>(wire, 8U);
  if (!valid_health_report(report) || producer_generation == 0U) {
    return make_unexpected(
        catalog_error(ErrorCode::kProtocolError, "ENDPOINT_CLOCK_HEALTH_CONTROL_INVALID"));
  }
  return CatalogClockHealthState{
      .has_report = true,
      .assessment = assessment,
      .reason = reason,
      .producer_epoch = producer_generation,
      .publisher_operation_sequence = load_little<std::uint64_t>(wire, 16U),
      .report_sequence = load_little<std::uint64_t>(wire, 24U),
  };
}

[[nodiscard]] ChannelQueuePolicy channel_queue_policy(const QueuePolicy policy) noexcept {
  switch (policy) {
    case QueuePolicy::kBlocking:
      return ChannelQueuePolicy::kBlocking;
    case QueuePolicy::kDropOldest:
      return ChannelQueuePolicy::kDropOldest;
    default:
      return ChannelQueuePolicy::kBlocking;
  }
}

[[nodiscard]] bool supported_queue_policy(const QueuePolicy policy) noexcept {
  return policy == QueuePolicy::kBlocking || policy == QueuePolicy::kDropOldest;
}

[[nodiscard]] ChannelOptions channel_options(const detail::ChannelEndpointRecord& endpoint,
                                             bool publisher) noexcept {
  ChannelOptions options{
      .queue_policy = channel_queue_policy(endpoint.service_static.queue_policy),
      .max_subscriber_queue_depth = endpoint.service_static.max_subscriber_depth,
      .subscriber_queue_depth = endpoint.subscriber_request
                                    ? endpoint.subscriber_request->queue_depth
                                    : endpoint.service_static.max_subscriber_depth,
      .max_subscriber_borrowed_samples = endpoint.service_static.max_subscriber_borrowed,
      .max_borrowed_samples = endpoint.subscriber_request
                                  ? endpoint.subscriber_request->max_borrowed_samples
                                  : endpoint.service_static.max_subscriber_borrowed,
      .publisher_history_depth = endpoint.service_static.publisher_history_depth,
      .max_loaned_samples =
          endpoint.publisher_request && endpoint.publisher_request->max_loaned_samples
              ? *endpoint.publisher_request->max_loaned_samples
              : 1U,
      .expected_recipient_count = 1U,
      .enable_event_wake = endpoint.service_static.event_wake,
      .replacement_attribution_required =
          publisher && endpoint.service_static.queue_policy != QueuePolicy::kBlocking,
  };
  if (publisher) {
    options.subscriber_queue_depth =
        std::min(options.subscriber_queue_depth, options.max_subscriber_queue_depth);
  }
  return options;
}

[[nodiscard]] ChannelOptions health_channel_options(ChannelOptions data_options) noexcept {
  // CHAN-069 gives health its own bounded service so every accepted report remains contiguous,
  // even when the data service admits replacement. Preserve all finite service geometry while
  // removing the data lane's lossy policy and publisher-only replacement callback requirement.
  data_options.queue_policy = ChannelQueuePolicy::kBlocking;
  data_options.replacement_attribution_required = false;
  return data_options;
}

[[nodiscard]] expected<CatalogInlinePayloadContract, Error> validate_attach(
    const detail::ChannelEndpointRecord& endpoint, const PayloadIdentityRecord& payload_identity,
    std::size_t payload_size, ChannelAttachOptions options, TopicDirection expected_direction,
    const bool attaching_publisher) {
  if (endpoint.direction != expected_direction) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_CLIENT_ROLE_MISMATCH"));
  }
  auto payload_compatible = validate_payload_attachment(endpoint.service_static, payload_identity);
  if (!payload_compatible) {
    return make_unexpected(std::move(payload_compatible).error());
  }
  const auto service_contract =
      catalog_inline_payload_contract(endpoint.service_static.payload_identity,
                                      endpoint.service_static.schema_writer_max_serialized_bytes);
  const auto attaching_contract = catalog_inline_payload_contract(payload_identity, payload_size);
  if (!service_contract || service_contract->selected_bucket_bytes == 0U || !attaching_contract ||
      attaching_contract->native_payload_capacity_bytes != payload_size) {
    return make_unexpected(catalog_error(ErrorCode::kIncompatible, "ENDPOINT_TYPE_MISMATCH"));
  }
  if (endpoint.service_static.required_data_frame_bytes != service_contract->required_frame_bytes ||
      endpoint.service_static.selected_inline_bucket_bytes !=
          service_contract->selected_bucket_bytes ||
      endpoint.service_static.bucket_usable_serialized_bytes !=
          service_contract->bucket_usable_native_payload_bytes) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_SERVICE_CONTRACT_INVALID"));
  }
  const bool capacity_admitted = attaching_publisher
                                     ? attaching_contract->native_payload_capacity_bytes <=
                                           service_contract->native_payload_capacity_bytes
                                     : attaching_contract->native_payload_capacity_bytes >=
                                           service_contract->native_payload_capacity_bytes;
  if (!capacity_admitted) {
    return make_unexpected(
        catalog_error(ErrorCode::kIncompatible, "ENDPOINT_PAYLOAD_CAPACITY_INCOMPATIBLE"));
  }
  if (!supported_queue_policy(endpoint.service_static.queue_policy) ||
      endpoint.service_static.publisher_history_depth != 0U ||
      endpoint.service_static.max_subscriber_depth == 0U ||
      endpoint.service_static.max_subscriber_borrowed == 0U || endpoint.service_static.id.empty()) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_SERVICE_CONTRACT_INVALID"));
  }
  if ((endpoint.attach.peer != PeerRequirement::kOptional &&
       endpoint.attach.peer != PeerRequirement::kRequired) ||
      !endpoint.attach.max_wait || endpoint.attach.max_wait->count() <= 0 ||
      (options.timeout &&
       (options.timeout->count() <= 0 || *options.timeout > *endpoint.attach.max_wait))) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_ATTACH_BOUND_INVALID"));
  }
  const ClockHealthOptions health =
      endpoint.publisher_request ? endpoint.publisher_request->clock_health : ClockHealthOptions{};
  if (!valid_clock_health_contract(health) ||
      (health.capability == ClockHealthCapability::kStatusReports &&
       (endpoint.direction != TopicDirection::kImport || !endpoint.publisher_request))) {
    return make_unexpected(
        catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_CLOCK_HEALTH_CONTRACT_INVALID"));
  }
  return *service_contract;
}

[[nodiscard]] expected<void, Error> wait_for_required_subscribers(
    const detail::ChannelEndpointRecord& endpoint, const ChannelAttachOptions& options,
    InlinePayloadPublisher& data, InlinePayloadPublisher* health) noexcept {
  return wait_for_catalog_required_peer(endpoint, options, [&]() -> expected<bool, Error> {
    if (health != nullptr) {
      auto refreshed = health->update_connections();
      if (!refreshed) {
        return make_unexpected(std::move(refreshed).error());
      }
    }
    auto refreshed = data.update_connections();
    if (!refreshed) {
      return make_unexpected(std::move(refreshed).error());
    }
    return data.has_subscribers() && (health == nullptr || health->has_subscribers());
  });
}

[[nodiscard]] expected<void, Error> wait_for_required_publishers(
    const detail::ChannelEndpointRecord& endpoint, const ChannelAttachOptions& options,
    InlinePayloadSubscriber& data, InlinePayloadSubscriber* health) noexcept {
  return wait_for_catalog_required_peer(endpoint, options, [&]() -> expected<bool, Error> {
    auto data_ready = data.has_samples();
    if (!data_ready) {
      return make_unexpected(std::move(data_ready).error());
    }
    if (health != nullptr) {
      auto health_ready = health->has_samples();
      if (!health_ready) {
        return make_unexpected(std::move(health_ready).error());
      }
    }
    // Retained data or terminal control proves a short-lived publisher existed even if it retired
    // before this refresh observed its peer count. Readiness is non-consuming, so attachment can
    // succeed without stealing the queued record from the caller.
    return *data_ready ||
           (data.has_publishers() && (health == nullptr || health->has_publishers()));
  });
}

class CatalogInlinePayloadPublisher final : public InlinePayloadPublisher {
 public:
  CatalogInlinePayloadPublisher(std::unique_ptr<InlinePayloadPublisher> data,
                                std::unique_ptr<InlinePayloadPublisher> health,
                                PayloadIdentityRecord payload_identity, std::size_t payload_size,
                                const std::size_t service_frame_size, ClockId capture_clock_id,
                                std::uint64_t catalog_clock_count, ClockHealthOptions clock_health,
                                const bool protected_terminal_required,
                                const bool graph_owned_publication)
      : data_(std::move(data)),
        health_(std::move(health)),
        payload_identity_(std::move(payload_identity)),
        native_payload_size_(payload_size),
        wire_payload_size_(catalog_wire_payload_size(payload_identity_, payload_size)),
        framed_payload_size_(service_frame_size),
        encoded_(service_frame_size),
        capture_clock_id_(capture_clock_id),
        catalog_clock_count_(catalog_clock_count),
        clock_health_(clock_health),
        protected_terminal_required_(protected_terminal_required),
        graph_owned_publication_(graph_owned_publication) {}

  [[nodiscard]] InlinePublicationResult emit_phase(
      std::span<const std::byte> payload, const SampleMetadata& supplied_metadata) override {
    return emit_phase_impl(payload, supplied_metadata, false);
  }

  [[nodiscard]] InlinePublicationResult emit_canonical_pod_phase(
      std::span<const std::byte> payload, const SampleMetadata& supplied_metadata) {
    return emit_phase_impl(payload, supplied_metadata, true);
  }

  [[nodiscard]] bool supports_canonical_pod_publication() const noexcept {
    return payload_identity_.kind == PayloadIdentityKind::kPod && payload_identity_.pod;
  }

 private:
  // Publication admission, framing, fault injection, and authority consumption form one ordered
  // transaction. Keeping the checks together makes the pre-submit versus indeterminate boundary
  // explicit and prevents a future early return from consuming the operation sequence.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] InlinePublicationResult emit_phase_impl(std::span<const std::byte> payload,
                                                        const SampleMetadata& supplied_metadata,
                                                        const bool canonical_pod) {
    if (stream_closed_) {
      return {.error = Error{ErrorCode::kClosed}};
    }
    const bool schema_payload =
        payload_identity_.kind == PayloadIdentityKind::kSchema && payload_identity_.schema;
    const bool custom_pod =
        payload_identity_.kind == PayloadIdentityKind::kPod && payload_identity_.pod;
    if ((canonical_pod && !custom_pod) ||
        (!schema_payload &&
         payload.size() != (canonical_pod ? wire_payload_size_ : native_payload_size_)) ||
        (schema_payload && (payload.empty() || payload.size() > native_payload_size_ ||
                            supplied_metadata.byte_size != payload.size() ||
                            supplied_metadata.payload_kind != PayloadKind::kSerializedBytes))) {
      return {.error = catalog_error(ErrorCode::kInvalidArgument, "ENDPOINT_TYPE_MISMATCH")};
    }
    if (operation_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
      return {.error = Error{ErrorCode::kResourceExhausted}};
    }
    if (supplied_metadata.version.major != 1U || supplied_metadata.version.minor != 0U ||
        (static_cast<std::uint32_t>(supplied_metadata.flags) & ~0x3U) != 0U) {
      return {.error = Error{ErrorCode::kInvalidArgument}};
    }
    if (capture_clock_id_.valid()) {
      if (supplied_metadata.source_clock_id != capture_clock_id_) {
        return {.error = Error{ErrorCode::kInvalidArgument}};
      }
    } else if (supplied_metadata.source_clock_id.valid() ||
               supplied_metadata.capture_timestamp_ns != 0) {
      return {.error = Error{ErrorCode::kInvalidArgument}};
    }

    // Every submitted byte is overwritten below. The cold endpoint-owned scratch avoids a large
    // stack frame and leaves its unused tail untouched on variable-size schema publications.
    const std::span<std::byte> encoded{encoded_};
    encode_data_attribution(
        encoded, DataAttributionKind::kData, report_sequence_, producer_generation_);
    std::size_t submitted_frame_size = framed_payload_size_;
    if (schema_payload) {
      auto envelope =
          encode_schema_envelope(std::span<std::byte>{encoded}.subspan(kCatalogDataAttributionBytes,
                                                                       kSchemaEnvelopeHeaderSize),
                                 *payload_identity_.schema,
                                 payload.size());
      if (!envelope) {
        return {.error = std::move(envelope).error()};
      }
      std::copy(payload.begin(),
                payload.end(),
                encoded.begin() + kCatalogDataAttributionBytes + kSchemaEnvelopeHeaderSize);
      submitted_frame_size =
          kCatalogDataAttributionBytes + kSchemaEnvelopeHeaderSize + payload.size();
    } else if (canonical_pod) {
      std::copy(payload.begin(), payload.end(), encoded.begin() + kCatalogDataAttributionBytes);
    } else {
      auto encoded_result = encode_catalog_payload(
          payload_identity_,
          payload,
          std::span<std::byte>{encoded}.subspan(kCatalogDataAttributionBytes, wire_payload_size_));
      if (!encoded_result) {
        return {.error = std::move(encoded_result).error()};
      }
    }
    if (next_schema_corruption_) {
      // Keep fault injection below the catalog encoder and above the backend submit boundary so
      // the conformance test exercises the same transport, staging, envelope decoder, verifier,
      // and typed materialization path as a malformed peer publication.
      corrupt_schema_frame(std::span<std::byte>{encoded}.first(submitted_frame_size),
                           *next_schema_corruption_);
      next_schema_corruption_.reset();
    }
    SampleMetadata metadata = supplied_metadata;
    const std::uint64_t proposed_sequence = operation_sequence_ + 1U;
    metadata.sequence = proposed_sequence;
    if (graph_owned_publication_) {
      // Runtime has already translated the publication coordinate into this catalog's clock
      // namespace, so forward it instead of erasing it. An identity outside the declared clock
      // range cannot be resolved by any peer reading this catalog, and shipping one would hand a
      // subscriber a number it must silently ignore.
      if (!publication_coordinate_declared(metadata, catalog_clock_count_)) {
        return {.error =
                    catalog_error(ErrorCode::kProtocolError, "ENDPOINT_PUBLISH_CLOCK_INVALID")};
      }
    } else {
      // This endpoint publishes into the Graph, so the client on this side holds no stamp authority
      // and its coordinate is erased rather than rejected. Runtime refuses any populated
      // publication coordinate on an import, so forwarding a supplied pair would turn a formerly
      // accepted client publication into a terminal protocol failure.
      metadata.publish_timestamp_ns = 0;
      metadata.publish_clock_id = {};
    }
    metadata.byte_size = schema_payload ? payload.size() : wire_payload_size_;
    metadata.payload_kind =
        schema_payload ? PayloadKind::kSerializedBytes : PayloadKind::kInlineTyped;
    InlinePublicationResult result = data_->emit_phase(
        std::span<const std::byte>{encoded}.first(submitted_frame_size), metadata);
    // A standalone catalog client may publish before its optional peer attaches. HoloMQ proves
    // that an accepted zero-recipient send was observed by nobody, so preserve that public result
    // without consuming the next catalog operation identity. Graph-owned outputs still validate
    // the accepted count against frozen fanout in Runtime's OutputState.
    if (result.phase == InlinePublicationPhase::kAccepted && result.recipient_count == 0U) {
      return result;
    }
    if (result.phase == InlinePublicationPhase::kAccepted && result.recipient_count != 1U) {
      result.phase = InlinePublicationPhase::kPostSubmitUncertain;
      result.error =
          result.recipient_count < 1U
              ? make_publication_partial_error(result.recipient_count, 1U)
              : Error{ErrorCode::kProtocolError, "CATALOG_PUBLICATION_RECIPIENT_COUNT_EXCESS"};
    }
    if (result.phase == InlinePublicationPhase::kAccepted ||
        result.phase == InlinePublicationPhase::kPostSubmitUncertain) {
      operation_sequence_ = proposed_sequence;
    }
    return result;
  }

 public:
  [[nodiscard]] expected<TimePoint, Error> make_capture_time(
      std::int64_t timestamp_ns) const noexcept override {
    if (!capture_clock_id_.valid()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    return TimePoint{.timestamp_ns = timestamp_ns, .clock_id = capture_clock_id_};
  }

  [[nodiscard]] expected<void, Error> report_clock_health(ClockHealthReport report) override {
    if (stream_closed_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    if (clock_health_.capability != ClockHealthCapability::kStatusReports || !health_ ||
        !valid_health_report(report)) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (operation_sequence_ == std::numeric_limits<std::uint64_t>::max() ||
        report_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    if (auto refreshed = update_connections(); !refreshed) {
      return make_unexpected(std::move(refreshed).error());
    }
    const std::uint64_t proposed_operation = operation_sequence_ + 1U;
    const std::uint64_t proposed_report = report_sequence_ + 1U;
    const auto wire =
        encode_health_control(producer_generation_, proposed_operation, proposed_report, report);
    const SampleMetadata metadata{
        .sequence = proposed_operation,
        .byte_size = wire.size(),
        .payload_kind = PayloadKind::kInlineTyped,
    };
    InlinePublicationResult result = health_->emit_phase(wire, metadata);
    if (result.phase == InlinePublicationPhase::kAccepted && result.recipient_count != 1U) {
      result.phase = InlinePublicationPhase::kPostSubmitUncertain;
      result.error =
          result.recipient_count < 1U
              ? make_publication_partial_error(result.recipient_count, 1U)
              : Error{ErrorCode::kProtocolError, "CATALOG_PUBLICATION_RECIPIENT_COUNT_EXCESS"};
    }
    if (result.phase == InlinePublicationPhase::kAccepted ||
        result.phase == InlinePublicationPhase::kPostSubmitUncertain) {
      operation_sequence_ = proposed_operation;
      report_sequence_ = proposed_report;
    }
    if (result.phase == InlinePublicationPhase::kAccepted) {
      return {};
    }
    if (result.phase == InlinePublicationPhase::kPostSubmitUncertain) {
      // The backend may have exposed this operation, so the publisher cannot safely attribute a
      // later report, data sample, or EOS to the same generation. Consume producer authority just
      // as an indeterminate close does and require the owner to create a new endpoint generation.
      stream_closed_ = true;
      connections_refreshed_ = false;
      if (publication_error_consumes_authority(result.error.code) ||
          result.error.code == ErrorCode::kProtocolError ||
          result.error.code == ErrorCode::kInvalidArgument) {
        return make_unexpected(std::move(result.error));
      }
      return make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
    }
    return make_unexpected(std::move(result.error));
  }

  [[nodiscard]] expected<void, Error> close_stream() override {
    if (stream_closed_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    if (operation_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    const std::span<std::byte> encoded{encoded_};
    // The fixed-width transport stages the complete configured frame before it can inspect the
    // EOS attribution. Initialize only that frame, not the unused maximum-size buffer tail.
    std::fill_n(encoded.begin(), framed_payload_size_, std::byte{});
    encode_data_attribution(
        encoded, DataAttributionKind::kEndOfStream, report_sequence_, producer_generation_);
    const std::uint64_t proposed_sequence = operation_sequence_ + 1U;
    const SampleMetadata metadata{
        .sequence = proposed_sequence,
        .byte_size = 0U,
        .payload_kind = PayloadKind::kInlineTyped,
    };
    const auto terminal_payload = std::span<const std::byte>{encoded}.first(framed_payload_size_);
    InlinePublicationResult result =
        protected_terminal_required_ ? emit_protected_terminal(*data_, terminal_payload, metadata)
                                     : data_->emit_phase(terminal_payload, metadata);
    if (result.phase == InlinePublicationPhase::kAccepted && result.recipient_count != 1U) {
      result.phase = InlinePublicationPhase::kPostSubmitUncertain;
      result.error =
          result.recipient_count < 1U
              ? make_publication_partial_error(result.recipient_count, 1U)
              : Error{ErrorCode::kProtocolError, "CATALOG_PUBLICATION_RECIPIENT_COUNT_EXCESS"};
    }
    if (result.phase == InlinePublicationPhase::kAccepted ||
        result.phase == InlinePublicationPhase::kPostSubmitUncertain) {
      operation_sequence_ = proposed_sequence;
      stream_closed_ = true;
      connections_refreshed_ = false;
    }
    if (result.phase == InlinePublicationPhase::kAccepted) {
      return {};
    }
    if (result.phase == InlinePublicationPhase::kPostSubmitUncertain) {
      if (publication_error_consumes_authority(result.error.code) ||
          result.error.code == ErrorCode::kProtocolError ||
          result.error.code == ErrorCode::kInvalidArgument) {
        return make_unexpected(std::move(result.error));
      }
      return make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
    }
    return make_unexpected(std::move(result.error));
  }

  [[nodiscard]] expected<void, Error> update_connections() noexcept override {
    if (health_) {
      auto health_updated = health_->update_connections();
      if (!health_updated) {
        return health_updated;
      }
    }
    auto data_updated = data_->update_connections();
    if (!data_updated) {
      return data_updated;
    }
    connections_refreshed_ = true;
    return {};
  }

  [[nodiscard]] bool has_subscribers() const noexcept override {
    return !stream_closed_ && connections_refreshed_ && data_->has_subscribers() &&
           (!health_ || health_->has_subscribers());
  }

  [[nodiscard]] expected<void, Error> corrupt_next_schema_publication(
      const CatalogSchemaFrameCorruption corruption) noexcept {
    if (payload_identity_.kind != PayloadIdentityKind::kSchema || !payload_identity_.schema ||
        next_schema_corruption_) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    switch (corruption) {
      case CatalogSchemaFrameCorruption::kMalformedEnvelope:
      case CatalogSchemaFrameCorruption::kSchemaId:
      case CatalogSchemaFrameCorruption::kPayloadSize:
      case CatalogSchemaFrameCorruption::kEndpointGeneration:
      case CatalogSchemaFrameCorruption::kPayloadVerifier:
      case CatalogSchemaFrameCorruption::kEnvelopeAbi:
        next_schema_corruption_ = corruption;
        return {};
    }
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }

  [[nodiscard]] std::uint64_t producer_generation() const noexcept { return producer_generation_; }

  [[nodiscard]] expected<void, Error> override_producer_generation(
      const std::uint64_t generation) noexcept {
    // Source-tree-only seam: a conformance test may impersonate one exact prior incarnation to
    // prove the receiver-side generation fence. Refuse after any published operation so the seam
    // can never split one live incarnation across two tokens.
    if (generation == 0U || operation_sequence_ != 0U || stream_closed_) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    producer_generation_ = generation;
    return {};
  }

 private:
  std::unique_ptr<InlinePayloadPublisher> data_;
  std::unique_ptr<InlinePayloadPublisher> health_;
  PayloadIdentityRecord payload_identity_;
  std::size_t native_payload_size_{};
  std::size_t wire_payload_size_{};
  std::size_t framed_payload_size_{};
  std::vector<std::byte> encoded_;
  ClockId capture_clock_id_{};
  std::uint64_t catalog_clock_count_{};
  ClockHealthOptions clock_health_{};
  bool protected_terminal_required_{};
  // True only on an export endpoint, where Runtime is the publisher and owns publication time. An
  // import endpoint is published by a client with no stamp authority, so it forwards no coordinate.
  bool graph_owned_publication_{};
  // Incarnation identity token stamped on every data, EOS, and health frame of this attachment.
  // It is not a total order; receivers fence on ANY change (see next_producer_generation()).
  std::uint64_t producer_generation_{mint_producer_generation()};
  std::uint64_t operation_sequence_{};
  std::uint64_t report_sequence_{};
  bool connections_refreshed_{};
  bool stream_closed_{};
  std::optional<CatalogSchemaFrameCorruption> next_schema_corruption_;
};

class CatalogInlinePayloadSubscriber final : public InlinePayloadSubscriber,
                                             public ProtectedTerminalWaitSetSubscriber,
                                             public StagedDataWaitSetSubscriber {
 public:
  CatalogInlinePayloadSubscriber(std::unique_ptr<InlinePayloadSubscriber> data,
                                 std::unique_ptr<InlinePayloadSubscriber> health,
                                 PayloadIdentityRecord payload_identity, std::size_t payload_size,
                                 const std::size_t service_frame_size, ClockId capture_clock_id,
                                 std::uint64_t catalog_clock_count,
                                 const bool replacement_gaps_allowed)
      : data_(std::move(data)),
        health_(std::move(health)),
        payload_identity_(std::move(payload_identity)),
        native_payload_size_(payload_size),
        wire_payload_size_(catalog_wire_payload_size(payload_identity_, payload_size)),
        framed_payload_size_(service_frame_size),
        capture_clock_id_(capture_clock_id),
        catalog_clock_count_(catalog_clock_count),
        replacement_gaps_allowed_(replacement_gaps_allowed),
        pending_wire_(service_frame_size) {}

  [[nodiscard]] expected<bool, Error> has_samples() const override {
    if (stream_closed_) {
      return false;
    }
    if (pending_data_) {
      return true;
    }
    if (health_) {
      // Iceoryx2 refreshes receiver-side publisher mappings through its readiness query. Health is
      // not a data trigger, but it must participate in that cold connection refresh so a
      // subscriber-first monitored endpoint can accept the publisher's first control operation.
      auto health_ready = health_->has_samples();
      if (!health_ready) {
        return make_unexpected(std::move(health_ready).error());
      }
    }
    return data_->has_samples();
  }

  [[nodiscard]] bool has_publishers() const noexcept override {
    return !stream_closed_ && data_->has_publishers() && (!health_ || health_->has_publishers());
  }

  [[nodiscard]] expected<InlinePayloadReceiveResult, Error> receive(
      std::span<std::byte> payload) override {
    if (stream_closed_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    // Ordinary receives remain bounded even when a producer continuously reports clock health.
    // Apply at most one admissible private operation per call; callers observe health-only work as
    // not-ready and may retry without letting control traffic monopolize their execution thread.
    auto operation = receive_next(payload, true, false);
    if (!operation) {
      return make_unexpected(std::move(operation).error());
    }
    if (operation->kind == CatalogReceivedOperationKind::kData) {
      return InlinePayloadReceiveResult{.metadata = operation->metadata,
                                        .payload_size = operation->payload_size};
    }
    if (operation->kind == CatalogReceivedOperationKind::kEndOfStream) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    if (operation->kind == CatalogReceivedOperationKind::kPeerIncarnationChanged) {
      return make_unexpected(
          catalog_error(ErrorCode::kStaleEndpoint, "ENDPOINT_PEER_INCARNATION_CHANGED"));
    }
    return make_unexpected(Error{ErrorCode::kNotReady});
  }

  [[nodiscard]] expected<InlinePayloadReceiveGuard, Error> receive_borrowed() override {
    return receive_borrowed_if_data_accepted(true);
  }

  [[nodiscard]] expected<InlinePayloadReceiveGuard, Error> receive_borrowed_if_data_accepted(
      const bool accept_data) override {
    if (borrowed_schema_payload_) {
      return make_unexpected(
          catalog_error(ErrorCode::kFailure, "ENDPOINT_PAYLOAD_LOAN_ALREADY_ACTIVE"));
    }
    if (stream_closed_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    if (payload_identity_.kind != PayloadIdentityKind::kSchema || !payload_identity_.schema) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    auto operation = receive_next(std::span<std::byte>{}, accept_data, true);
    if (!operation) {
      return make_unexpected(std::move(operation).error());
    }
    if (operation->kind == CatalogReceivedOperationKind::kData) {
      if (operation->schema_payload.empty()) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      borrowed_schema_payload_ = true;
      return InlinePayloadReceiveGuard{*this, operation->schema_payload, operation->metadata};
    }
    if (operation->kind == CatalogReceivedOperationKind::kEndOfStream) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    if (operation->kind == CatalogReceivedOperationKind::kDataPending) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
    if (operation->kind == CatalogReceivedOperationKind::kPeerIncarnationChanged) {
      return make_unexpected(
          catalog_error(ErrorCode::kStaleEndpoint, "ENDPOINT_PEER_INCARNATION_CHANGED"));
    }
    return make_unexpected(Error{ErrorCode::kNotReady});
  }

  void release_payload_loan() noexcept override { borrowed_schema_payload_ = false; }

  [[nodiscard]] expected<std::uint64_t, Error> attach_wait_set(void* wait_set) override {
    return data_->attach_wait_set(wait_set);
  }

  // Waiting is delegated wholesale to the inner data binding: it owns the transport lanes, so it
  // owns both the event-wake capability and the runtime the wait context is built from. Readiness
  // is not delegated; this composite's own has_samples additionally reports staged data that the
  // transport lane has already surrendered.
  [[nodiscard]] bool event_wake_enabled() const noexcept override {
    return data_->event_wake_enabled();
  }

  [[nodiscard]] void* wait_context_runtime() noexcept override {
    return data_->wait_context_runtime();
  }

  [[nodiscard]] bool wait_set_attached() const noexcept override {
    return data_->wait_set_attached() || (health_ && health_->wait_set_attached());
  }

  [[nodiscard]] bool readiness_handle_attached() const noexcept override {
    return data_->readiness_handle_attached() || (health_ && health_->readiness_handle_attached());
  }

  [[nodiscard]] expected<ChannelReadinessHandle, Error> readiness() override {
    auto claim = claim_readiness_owner();
    if (!claim) {
      return make_unexpected(std::move(claim).error());
    }
    if (stream_closed_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    auto ready = has_samples();
    if (!ready) {
      return make_unexpected(std::move(ready).error());
    }

    std::vector<ChannelReadinessHandle> handles;
    handles.reserve(health_ ? 2U : 1U);
    auto data = data_->readiness();
    if (!data) {
      return make_unexpected(std::move(data).error());
    }
    handles.push_back(std::move(*data));
    if (health_) {
      auto health = health_->readiness();
      if (!health) {
        return make_unexpected(std::move(health).error());
      }
      handles.push_back(std::move(*health));
    }
    return ChannelReadinessHandleAccess::create({}, std::move(handles), *ready);
  }

  [[nodiscard]] std::optional<std::uint64_t> protected_terminal_wait_handle()
      const noexcept override {
    return channel_internal::protected_terminal_wait_handle(*data_);
  }

  [[nodiscard]] std::optional<StagedDataState> staged_data_state() const noexcept override {
    if (pending_data_ && pending_data_attribution_.kind == DataAttributionKind::kData &&
        pending_metadata_.sequence != 0U &&
        !foreign_producer_generation(pending_data_attribution_.producer_generation)) {
      return StagedDataState{.sequence = pending_metadata_.sequence,
                             .capture_time = pending_metadata_.capture_time()};
    }
    return channel_internal::staged_data_state(*data_);
  }

  [[nodiscard]] std::optional<std::uint64_t> staged_terminal_sequence() const noexcept override {
    return channel_internal::staged_terminal_sequence(*data_);
  }

  [[nodiscard]] expected<void, Error> detach_wait_set() noexcept override {
    return data_->detach_wait_set();
  }

  [[nodiscard]] CatalogClockHealthState clock_health_state() const noexcept {
    return health_state_;
  }

  [[nodiscard]] expected<void, Error> delay_health_observation(
      const std::uint32_t attempts) noexcept {
    if (!health_ || attempts == 0U) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    delayed_health_observations_ = attempts;
    return {};
  }

  [[nodiscard]] expected<bool, Error> has_pending_operation() const noexcept {
    if (stream_closed_) {
      return false;
    }
    if (pending_data_ || pending_health_) {
      return true;
    }
    auto data_ready = data_->has_samples();
    if (!data_ready) {
      return make_unexpected(std::move(data_ready).error());
    }
    if (*data_ready || !health_) {
      return *data_ready;
    }
    return health_->has_samples();
  }

  [[nodiscard]] std::uint64_t latched_producer_generation() const noexcept {
    return latched_producer_generation_;
  }

  [[nodiscard]] expected<void, Error> supersede_producer_generation(
      const std::uint64_t generation) noexcept {
    // Only a freshly materialized replacement endpoint may record the incarnation it supersedes.
    // Post-cut frames still stamped with that token are then fenced even before any operation of
    // the replacement incarnation has been latched.
    if (generation == 0U || latched_producer_generation_ != 0U || pending_data_ ||
        pending_health_ || expected_operation_sequence_ != 1U || stream_closed_) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    superseded_producer_generation_ = generation;
    return {};
  }

  [[nodiscard]] expected<CatalogImportQuiescence, Error> import_quiescence() noexcept {
    // Stage the bounded per-lane heads (one data, one health) so pending traffic can be
    // attributed to a producer incarnation. Staging deliberately ignores an accepted EOS: a
    // foreign frame arriving after close is exactly what the rebind gate must observe.
    auto staged_data = stage_data();
    if (!staged_data) {
      return make_unexpected(std::move(staged_data).error());
    }
    auto staged_health = stage_health();
    if (!staged_health) {
      return make_unexpected(std::move(staged_health).error());
    }
    CatalogImportQuiescence result{
        .latched_producer_generation = latched_producer_generation_,
    };
    if (pending_data_) {
      const bool foreign =
          foreign_producer_generation(pending_data_attribution_.producer_generation);
      result.pending_foreign = result.pending_foreign || foreign;
      result.pending_current = result.pending_current || !foreign;
    }
    if (pending_health_) {
      const bool foreign = foreign_producer_generation(pending_health_state_.producer_epoch);
      result.pending_foreign = result.pending_foreign || foreign;
      result.pending_current = result.pending_current || !foreign;
    }
    return result;
  }

  // Receive staging, operation-sequence arbitration, health attribution, and typed decode are a
  // single bounded state-machine step. The helpers retain one bounded head per lane without
  // advancing transport before Runtime accepts the payload.
  [[nodiscard]] expected<CatalogReceivedOperation, Error> receive_next(
      std::span<std::byte> payload, const bool accept_data,
      const bool borrow_schema_payload) noexcept {
    if (borrowed_schema_payload_) {
      return make_unexpected(
          catalog_error(ErrorCode::kFailure, "ENDPOINT_PAYLOAD_LOAN_ALREADY_ACTIVE"));
    }
    if (stream_closed_) {
      return CatalogReceivedOperation{};
    }
    if ((!borrow_schema_payload && payload.size() != native_payload_size_) ||
        (borrow_schema_payload && !payload.empty() && payload.size() != native_payload_size_)) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto staged_data = stage_data();
    if (!staged_data) {
      return make_unexpected(std::move(staged_data).error());
    }
    auto staged_health = stage_health();
    if (!staged_health) {
      return make_unexpected(std::move(staged_health).error());
    }
    if (!pending_data_ && !pending_health_) {
      return CatalogReceivedOperation{};
    }
    if (operation_sequence_exhausted_) {
      return make_unexpected(
          catalog_error(ErrorCode::kProtocolError, "ENDPOINT_OPERATION_SEQUENCE_INVALID"));
    }

    auto fenced = fence_pending_incarnations();
    if (!fenced) {
      return make_unexpected(std::move(fenced).error());
    }
    if (*fenced) {
      return CatalogReceivedOperation{
          .kind = CatalogReceivedOperationKind::kPeerIncarnationChanged,
      };
    }

    auto lane = next_pending_lane();
    if (!lane) {
      return make_unexpected(std::move(lane).error());
    }
    if (*lane == PendingLane::kNone) {
      return CatalogReceivedOperation{};
    }
    if (*lane == PendingLane::kHealth) {
      return accept_pending_health();
    }
    return accept_pending_data(payload, accept_data, borrow_schema_payload);
  }

 private:
  enum class PendingLane : std::uint8_t {
    kNone,
    kHealth,
    kData,
  };

  [[nodiscard]] expected<bool, Error> fence_pending_incarnations() noexcept {
    // Before the first producer incarnation is latched, data and health heads from different
    // incarnations must not be treated as one operation sequence. Fence the later head before
    // applying either operation; the earlier retained head is validated and latches its producer
    // on a subsequent call. Equal sequence numbers are ambiguous and remain a protocol error.
    if (latched_producer_generation_ == 0U && pending_data_ && pending_health_ &&
        pending_data_attribution_.producer_generation != pending_health_state_.producer_epoch) {
      const std::uint64_t data_sequence = pending_metadata_.sequence;
      const std::uint64_t health_sequence = pending_health_state_.publisher_operation_sequence;
      if (data_sequence == health_sequence) {
        return make_unexpected(
            catalog_error(ErrorCode::kProtocolError, "ENDPOINT_OPERATION_SEQUENCE_INVALID"));
      }
      if (data_sequence < health_sequence) {
        pending_health_ = false;
      } else {
        pending_data_ = false;
      }
      return true;
    }

    // Classify each staged head before sequence arithmetic because operation sequences from
    // distinct producer incarnations are incomparable. Foreign EOS is dropped rather than
    // terminalizing the accepted stream.
    const bool foreign_data =
        pending_data_ && foreign_producer_generation(pending_data_attribution_.producer_generation);
    const bool foreign_health =
        pending_health_ && foreign_producer_generation(pending_health_state_.producer_epoch);
    if (!foreign_data && !foreign_health) {
      return false;
    }

    // A replacement attachment owns a fresh operation-sequence domain. Drop only the bounded
    // staged heads from another incarnation and mutate no accepted receiver state.
    pending_data_ = pending_data_ && !foreign_data;
    pending_health_ = pending_health_ && !foreign_health;
    return true;
  }

  [[nodiscard]] expected<PendingLane, Error> next_pending_lane() noexcept {
    const std::uint64_t data_sequence =
        pending_data_ ? pending_metadata_.sequence : std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t health_sequence = pending_health_
                                              ? pending_health_state_.publisher_operation_sequence
                                              : std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t next_sequence = std::min(data_sequence, health_sequence);
    auto sequence_aligned =
        align_expected_operation_sequence(next_sequence, data_sequence, health_sequence);
    if (!sequence_aligned) {
      return make_unexpected(std::move(sequence_aligned).error());
    }
    if (!*sequence_aligned) {
      return PendingLane::kNone;
    }
    return health_sequence == expected_operation_sequence_ ? PendingLane::kHealth
                                                           : PendingLane::kData;
  }

  [[nodiscard]] expected<CatalogReceivedOperation, Error> accept_pending_health() noexcept {
    const std::uint64_t health_sequence = pending_health_state_.publisher_operation_sequence;
    const bool report_sequence_contiguous =
        health_state_.report_sequence != std::numeric_limits<std::uint64_t>::max() &&
        pending_health_state_.report_sequence == health_state_.report_sequence + 1U;
    if (!report_sequence_contiguous || pending_health_metadata_.sequence != health_sequence ||
        pending_health_metadata_.byte_size != kHealthControlBytes ||
        pending_health_metadata_.payload_kind != PayloadKind::kInlineTyped) {
      return make_unexpected(
          catalog_error(ErrorCode::kProtocolError, "ENDPOINT_OPERATION_SEQUENCE_INVALID"));
    }
    latched_producer_generation_ = pending_health_state_.producer_epoch;
    health_state_ = pending_health_state_;
    pending_health_ = false;
    advance_expected_operation_sequence();
    return CatalogReceivedOperation{
        .kind = CatalogReceivedOperationKind::kClockHealth,
        .clock_health = health_state_,
    };
  }

  [[nodiscard]] expected<CatalogReceivedOperation, Error> accept_pending_data(
      std::span<std::byte> payload, const bool accept_data,
      const bool borrow_schema_payload) noexcept {
    if (pending_data_attribution_.health_report_sequence != health_state_.report_sequence) {
      return make_unexpected(
          catalog_error(ErrorCode::kProtocolError, "ENDPOINT_OPERATION_SEQUENCE_INVALID"));
    }

    if (pending_data_attribution_.kind == DataAttributionKind::kEndOfStream) {
      const SampleMetadata metadata = pending_metadata_;
      latched_producer_generation_ = pending_data_attribution_.producer_generation;
      pending_data_ = false;
      stream_closed_ = true;
      advance_expected_operation_sequence();
      return CatalogReceivedOperation{
          .kind = CatalogReceivedOperationKind::kEndOfStream,
          .metadata = metadata,
      };
    }

    // If every Runtime retention slot is protected, keep this staged head without decoding or
    // advancing the publisher operation sequence. An earlier health head has already been
    // applied, while a later health head remains ordered behind this data operation.
    if (!accept_data) {
      return CatalogReceivedOperation{
          .kind = CatalogReceivedOperationKind::kDataPending,
      };
    }

    const bool schema_payload =
        payload_identity_.kind == PayloadIdentityKind::kSchema && payload_identity_.schema;
    std::span<const std::byte> borrowed_schema_payload;
    std::size_t decoded_payload_size = 0U;
    if (schema_payload) {
      auto payload_size = decode_schema_envelope(
          std::span<const std::byte>{pending_wire_}.subspan(kCatalogDataAttributionBytes),
          *payload_identity_.schema,
          native_payload_size_);
      if (!payload_size || *payload_size != pending_metadata_.byte_size) {
        return make_unexpected(payload_size ? catalog_error(ErrorCode::kProtocolError,
                                                            "ENDPOINT_SCHEMA_PAYLOAD_SIZE_INVALID")
                                            : std::move(payload_size).error());
      }
      // pending_wire_ is endpoint-owned and reused, so its tail still holds the previous sample.
      // Admit only serialized bytes this staging operation actually received.
      if (kCatalogDataAttributionBytes + kSchemaEnvelopeHeaderSize + *payload_size >
          pending_wire_size_) {
        return make_unexpected(
            catalog_error(ErrorCode::kProtocolError, "ENDPOINT_SCHEMA_PAYLOAD_SIZE_INVALID"));
      }
      const auto serialized = std::span<const std::byte>{pending_wire_}.subspan(
          kCatalogDataAttributionBytes + kSchemaEnvelopeHeaderSize, *payload_size);
      if (borrow_schema_payload) {
        borrowed_schema_payload = serialized;
      } else {
        std::copy(serialized.begin(), serialized.end(), payload.begin());
        decoded_payload_size = *payload_size;
      }
    } else {
      auto decoded = decode_catalog_payload(payload_identity_,
                                            std::span<const std::byte>{pending_wire_}.subspan(
                                                kCatalogDataAttributionBytes, wire_payload_size_),
                                            payload);
      if (!decoded) {
        return make_unexpected(std::move(decoded).error());
      }
      decoded_payload_size = native_payload_size_;
    }

    SampleMetadata metadata = pending_metadata_;
    // Return Core metadata in native payload terms, never a transport representation width.
    metadata.byte_size = schema_payload ? pending_metadata_.byte_size : native_payload_size_;
    latched_producer_generation_ = pending_data_attribution_.producer_generation;
    pending_data_ = false;
    advance_expected_operation_sequence();
    return CatalogReceivedOperation{
        .kind = CatalogReceivedOperationKind::kData,
        .metadata = metadata,
        .payload_size = decoded_payload_size,
        .schema_payload = borrowed_schema_payload,
    };
  }

  [[nodiscard]] expected<bool, Error> align_expected_operation_sequence(
      const std::uint64_t next_sequence, const std::uint64_t data_sequence,
      const std::uint64_t health_sequence) noexcept {
    if (next_sequence < expected_operation_sequence_ ||
        (pending_data_ && pending_health_ && data_sequence == health_sequence)) {
      return make_unexpected(
          catalog_error(ErrorCode::kProtocolError, "ENDPOINT_OPERATION_SEQUENCE_INVALID"));
    }
    if (next_sequence == expected_operation_sequence_) {
      return true;
    }

    // A subscriber that attaches after this producer already published data can first observe its
    // ordered EOS. Admit that terminal operation as the attachment boundary instead of waiting for
    // data that was published before this receiver existed. Keep the contiguous fast path above
    // unchanged, and do not grant the same gap to a late ordinary data operation.
    if (latched_producer_generation_ == 0U && pending_data_ && !pending_health_ &&
        pending_data_attribution_.kind == DataAttributionKind::kEndOfStream) {
      health_state_.report_sequence = pending_data_attribution_.health_report_sequence;
      expected_operation_sequence_ = next_sequence;
      return true;
    }

    // Lossless data and health services may become visible at different times. Retain one head
    // from each service and wait for the missing predecessor instead of letting a later operation
    // overtake it. A replacement queue may legitimately erase that predecessor before the
    // subscriber observes it, so advance only that queue family to its oldest observable head.
    if (!replacement_gaps_allowed_) {
      return false;
    }
    if (health_sequence == next_sequence) {
      // The health lane is lossless and FIFO. Its next contiguous report proves that every skipped
      // publisher operation before this health record was replaced by the lossy data queue.
      if (health_state_.report_sequence == std::numeric_limits<std::uint64_t>::max() ||
          pending_health_state_.report_sequence != health_state_.report_sequence + 1U) {
        return make_unexpected(
            catalog_error(ErrorCode::kProtocolError, "ENDPOINT_OPERATION_SEQUENCE_INVALID"));
      }
    } else if (pending_data_attribution_.health_report_sequence > health_state_.report_sequence) {
      // A later data head records how many health reports the publisher had accepted before it.
      // Wait for the separate lossless lane instead of misclassifying delayed health as replaced
      // data. Staging remains bounded to one head from each service.
      return false;
    } else if (pending_data_attribution_.health_report_sequence < health_state_.report_sequence) {
      return make_unexpected(
          catalog_error(ErrorCode::kProtocolError, "ENDPOINT_OPERATION_SEQUENCE_INVALID"));
    }
    expected_operation_sequence_ = next_sequence;
    return true;
  }

  [[nodiscard]] bool foreign_producer_generation(const std::uint64_t generation) const noexcept {
    // Hot-path fence: two integer compares. The token is an identity, never an order; any
    // difference from the latched incarnation (or equality with the superseded one recorded by a
    // runtime rebind) marks the frame as another producer incarnation.
    return generation == superseded_producer_generation_ ||
           (latched_producer_generation_ != 0U && generation != latched_producer_generation_);
  }

  [[nodiscard]] expected<void, Error> stage_data() noexcept {
    if (pending_data_) {
      return {};
    }
    auto ready = data_->has_samples();
    if (!ready) {
      return make_unexpected(std::move(ready).error());
    }
    if (!*ready) {
      return {};
    }
    auto received = data_->receive(std::span<std::byte>{pending_wire_}.first(framed_payload_size_));
    if (!received) {
      return make_unexpected(std::move(received).error());
    }
    // A schema publication submits only its encoded prefix, so the staged frame is variable. Any
    // frame too short to carry attribution cannot be decoded at all, and every later read of
    // pending_wire_ is bounded by this length so the reused tail is never mistaken for payload.
    if (received->payload_size < kCatalogDataAttributionBytes ||
        received->payload_size > framed_payload_size_) {
      return make_unexpected(
          catalog_error(ErrorCode::kProtocolError, "ENDPOINT_DATA_FRAME_SIZE_INVALID"));
    }
    auto attribution = decode_data_attribution(
        std::span<const std::byte>{pending_wire_}.first(framed_payload_size_));
    if (!attribution) {
      return make_unexpected(std::move(attribution).error());
    }
    // A non-schema endpoint publishes its complete fixed frame; a short one is malformed.
    const bool schema_payload =
        payload_identity_.kind == PayloadIdentityKind::kSchema && payload_identity_.schema;
    if (!schema_payload && attribution->kind == DataAttributionKind::kData &&
        received->payload_size != framed_payload_size_) {
      return make_unexpected(
          catalog_error(ErrorCode::kProtocolError, "ENDPOINT_DATA_FRAME_SIZE_INVALID"));
    }
    auto metadata_valid = validate_received_data_metadata(received->metadata,
                                                          payload_identity_,
                                                          native_payload_size_,
                                                          capture_clock_id_,
                                                          catalog_clock_count_,
                                                          attribution->kind);
    if (!metadata_valid) {
      return make_unexpected(std::move(metadata_valid).error());
    }
    pending_metadata_ = received->metadata;
    pending_wire_size_ = received->payload_size;
    pending_data_attribution_ = *attribution;
    pending_data_ = true;
    return {};
  }

  [[nodiscard]] expected<void, Error> stage_health() noexcept {
    if (!health_ || pending_health_) {
      return {};
    }
    if (delayed_health_observations_ != 0U) {
      --delayed_health_observations_;
      return {};
    }
    auto ready = health_->has_samples();
    if (!ready) {
      return make_unexpected(std::move(ready).error());
    }
    if (!*ready) {
      return {};
    }
    std::array<std::byte, kHealthControlBytes> wire{};
    auto received = health_->receive(wire);
    if (!received) {
      return make_unexpected(std::move(received).error());
    }
    if (received->payload_size != kHealthControlBytes) {
      return make_unexpected(
          catalog_error(ErrorCode::kProtocolError, "ENDPOINT_HEALTH_CONTROL_INVALID"));
    }
    auto report = decode_health_control(wire);
    if (!report) {
      return make_unexpected(std::move(report).error());
    }
    pending_health_metadata_ = received->metadata;
    pending_health_state_ = *report;
    pending_health_ = true;
    return {};
  }

  void advance_expected_operation_sequence() noexcept {
    if (expected_operation_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
      operation_sequence_exhausted_ = true;
    } else {
      ++expected_operation_sequence_;
    }
  }

  std::unique_ptr<InlinePayloadSubscriber> data_;
  std::unique_ptr<InlinePayloadSubscriber> health_;
  PayloadIdentityRecord payload_identity_;
  std::size_t native_payload_size_{};
  std::size_t wire_payload_size_{};
  std::size_t framed_payload_size_{};
  ClockId capture_clock_id_{};
  std::uint64_t catalog_clock_count_{};
  bool replacement_gaps_allowed_{};
  std::vector<std::byte> pending_wire_;
  /// Bytes the staged frame actually wrote into pending_wire_; the remainder is a previous sample.
  std::size_t pending_wire_size_{};
  SampleMetadata pending_metadata_{};
  DataAttribution pending_data_attribution_{};
  bool pending_data_{};
  SampleMetadata pending_health_metadata_{};
  CatalogClockHealthState pending_health_state_{};
  bool pending_health_{};
  std::uint32_t delayed_health_observations_{};
  std::uint64_t expected_operation_sequence_{1U};
  // Producer incarnation latched from the first accepted operation; zero until then. The
  // superseded token is seeded by a runtime rebind so stale post-cut frames are fenced too.
  std::uint64_t latched_producer_generation_{};
  std::uint64_t superseded_producer_generation_{};
  bool operation_sequence_exhausted_{};
  bool stream_closed_{};
  bool borrowed_schema_payload_{};
  CatalogClockHealthState health_state_{};
};

}  // namespace

expected<CatalogClockHealthState, Error> CatalogChannelBindingAccess::clock_health_state(
    const InlinePayloadSubscriber& subscriber) noexcept {
  const auto* catalog = dynamic_cast<const CatalogInlinePayloadSubscriber*>(&subscriber);
  if (catalog == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return catalog->clock_health_state();
}

CatalogCanonicalPodPublisherBinding CatalogChannelBindingAccess::bind_canonical_pod_publisher(
    InlinePayloadPublisher& publisher) noexcept {
  auto* catalog = dynamic_cast<CatalogInlinePayloadPublisher*>(&publisher);
  if (catalog == nullptr || !catalog->supports_canonical_pod_publication()) {
    return {};
  }
  return CatalogCanonicalPodPublisherBinding{
      .state = catalog,
      .emit_phase =
          [](void* state,
             const std::span<const std::byte>
                 payload,
             const SampleMetadata& metadata) {
            return static_cast<CatalogInlinePayloadPublisher*>(state)->emit_canonical_pod_phase(
                payload, metadata);
          },
  };
}

expected<void, Error> CatalogChannelBindingTestAccess::corrupt_next_schema_publication(
    InlinePayloadPublisher& publisher, const CatalogSchemaFrameCorruption corruption) noexcept {
  auto* catalog = dynamic_cast<CatalogInlinePayloadPublisher*>(&publisher);
  if (catalog == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return catalog->corrupt_next_schema_publication(corruption);
}

expected<std::uint64_t, Error> CatalogChannelBindingTestAccess::producer_generation(
    const InlinePayloadPublisher& publisher) noexcept {
  const auto* catalog = dynamic_cast<const CatalogInlinePayloadPublisher*>(&publisher);
  if (catalog == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return catalog->producer_generation();
}

expected<void, Error> CatalogChannelBindingTestAccess::override_producer_generation(
    InlinePayloadPublisher& publisher, const std::uint64_t generation) noexcept {
  auto* catalog = dynamic_cast<CatalogInlinePayloadPublisher*>(&publisher);
  if (catalog == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return catalog->override_producer_generation(generation);
}

expected<void, Error> CatalogChannelBindingTestAccess::delay_health_observation(
    InlinePayloadSubscriber& subscriber, const std::uint32_t attempts) noexcept {
  auto* catalog = dynamic_cast<CatalogInlinePayloadSubscriber*>(&subscriber);
  if (catalog == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return catalog->delay_health_observation(attempts);
}

expected<void, Error> CatalogChannelBindingTestAccess::validate_data_metadata(
    const ChannelEndpointCatalog::EndpointView endpoint, const SampleMetadata& metadata) noexcept {
  const detail::ChannelEndpointRecord* const record =
      detail::ChannelEndpointCatalogAccess::record(endpoint);
  if (record == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const std::uint64_t capacity = record->service_static.schema_writer_max_serialized_bytes;
  if (capacity > std::numeric_limits<std::size_t>::max()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const std::size_t payload_capacity = static_cast<std::size_t>(capacity);
  if (channel_internal::catalog_wire_payload_size(record->service_static.payload_identity,
                                                  payload_capacity) == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return validate_received_data_metadata(
      metadata,
      record->service_static.payload_identity,
      catalog_native_payload_size(record->service_static.payload_identity, payload_capacity),
      detail::ChannelEndpointCatalogAccess::capture_clock_id(endpoint),
      detail::ChannelEndpointCatalogAccess::clock_count(endpoint),
      DataAttributionKind::kData);
}

expected<std::unique_ptr<InlinePayloadSubscriber>, Error>
CatalogChannelBindingAccess::create_payload_subscriber(
    ChannelRuntime& runtime, ChannelEndpointCatalog::EndpointView endpoint,
    std::size_t payload_size, ChannelAttachOptions options) {
  const detail::ChannelEndpointRecord* record =
      detail::ChannelEndpointCatalogAccess::record(endpoint);
  if (record == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return runtime.create_catalog_inline_payload_subscriber(endpoint,
                                                          record->service_static.payload_identity,
                                                          payload_size,
                                                          std::move(options),
                                                          TopicDirection::kImport,
                                                          false);
}

expected<std::unique_ptr<InlinePayloadPublisher>, Error>
CatalogChannelBindingAccess::create_payload_publisher(ChannelRuntime& runtime,
                                                      ChannelEndpointCatalog::EndpointView endpoint,
                                                      std::size_t payload_size,
                                                      ChannelAttachOptions options) {
  return create_payload_publisher(
      runtime, endpoint, payload_size, std::move(options), PublisherPressureOptions{});
}

expected<std::unique_ptr<InlinePayloadPublisher>, Error>
CatalogChannelBindingAccess::create_payload_publisher(ChannelRuntime& runtime,
                                                      ChannelEndpointCatalog::EndpointView endpoint,
                                                      std::size_t payload_size,
                                                      ChannelAttachOptions options,
                                                      const PublisherPressureOptions& pressure) {
  const detail::ChannelEndpointRecord* record =
      detail::ChannelEndpointCatalogAccess::record(endpoint);
  if (record == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return runtime.create_catalog_inline_payload_publisher(endpoint,
                                                         record->service_static.payload_identity,
                                                         payload_size,
                                                         std::move(options),
                                                         TopicDirection::kExport,
                                                         false,
                                                         pressure);
}

expected<CatalogReceivedOperation, Error> CatalogChannelBindingAccess::receive_next(
    InlinePayloadSubscriber& subscriber, std::span<std::byte> payload, const bool accept_data,
    const bool borrow_schema_payload) noexcept {
  auto* catalog = dynamic_cast<CatalogInlinePayloadSubscriber*>(&subscriber);
  if (catalog == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return catalog->receive_next(payload, accept_data, borrow_schema_payload);
}

expected<bool, Error> CatalogChannelBindingAccess::has_pending_operation(
    const InlinePayloadSubscriber& subscriber) noexcept {
  const auto* catalog = dynamic_cast<const CatalogInlinePayloadSubscriber*>(&subscriber);
  if (catalog == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return catalog->has_pending_operation();
}

expected<std::uint64_t, Error> CatalogChannelBindingAccess::latched_producer_generation(
    const InlinePayloadSubscriber& subscriber) noexcept {
  const auto* catalog = dynamic_cast<const CatalogInlinePayloadSubscriber*>(&subscriber);
  if (catalog == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return catalog->latched_producer_generation();
}

expected<void, Error> CatalogChannelBindingAccess::supersede_producer_generation(
    InlinePayloadSubscriber& subscriber, const std::uint64_t generation) noexcept {
  auto* catalog = dynamic_cast<CatalogInlinePayloadSubscriber*>(&subscriber);
  if (catalog == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return catalog->supersede_producer_generation(generation);
}

expected<CatalogImportQuiescence, Error> CatalogChannelBindingAccess::import_quiescence(
    InlinePayloadSubscriber& subscriber) noexcept {
  auto* catalog = dynamic_cast<CatalogInlinePayloadSubscriber*>(&subscriber);
  if (catalog == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return catalog->import_quiescence();
}

}  // namespace channel_internal

expected<std::unique_ptr<channel_internal::InlinePayloadPublisher>, Error>
ChannelRuntime::create_catalog_inline_payload_publisher(
    ChannelEndpointCatalog::EndpointView endpoint, PayloadIdentityRecord payload_identity,
    const std::size_t payload_size, ChannelAttachOptions options,
    const TopicDirection publisher_direction, const bool enforce_peer_requirement) {
  return create_catalog_inline_payload_publisher(endpoint,
                                                 std::move(payload_identity),
                                                 payload_size,
                                                 std::move(options),
                                                 publisher_direction,
                                                 enforce_peer_requirement,
                                                 channel_internal::PublisherPressureOptions{});
}

expected<std::unique_ptr<channel_internal::InlinePayloadPublisher>, Error>
ChannelRuntime::create_catalog_inline_payload_publisher(
    ChannelEndpointCatalog::EndpointView endpoint, PayloadIdentityRecord payload_identity,
    std::size_t payload_size, ChannelAttachOptions options, TopicDirection publisher_direction,
    const bool enforce_peer_requirement,
    const channel_internal::PublisherPressureOptions& pressure) {
  if (!impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  const detail::ChannelEndpointRecord* endpoint_record =
      detail::ChannelEndpointCatalogAccess::record(endpoint);
  if (endpoint_record == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto service_contract = channel_internal::validate_attach(
      *endpoint_record, payload_identity, payload_size, options, publisher_direction, true);
  if (!service_contract) {
    return make_unexpected(std::move(service_contract).error());
  }
  const std::string binding =
      detail::ChannelEndpointCatalogAccess::backend_service_binding(endpoint);
  const std::size_t framed_payload_size =
      static_cast<std::size_t>(service_contract->required_frame_bytes);
  const std::size_t selected_bucket_bytes =
      static_cast<std::size_t>(service_contract->selected_bucket_bytes);
  const ChannelOptions endpoint_options = channel_internal::channel_options(*endpoint_record, true);
  const channel_internal::PublisherPressureOptions data_pressure =
      pressure.for_queue(endpoint_options.queue_policy);
  auto data =
      channel_internal::create_inline_payload_publisher(impl_->context_,
                                                        binding,
                                                        selected_bucket_bytes,
                                                        endpoint_options,
                                                        impl_->context_.pubsub_capabilities(),
                                                        data_pressure);
  if (!data) {
    return make_unexpected(std::move(data).error());
  }

  const ClockHealthOptions health_options = endpoint_record->publisher_request
                                                ? endpoint_record->publisher_request->clock_health
                                                : ClockHealthOptions{};
  std::unique_ptr<channel_internal::InlinePayloadPublisher> health;
  if (health_options.capability == ClockHealthCapability::kStatusReports) {
    const ChannelOptions health_endpoint_options =
        channel_internal::health_channel_options(endpoint_options);
    const channel_internal::PublisherPressureOptions health_pressure =
        pressure.for_queue(health_endpoint_options.queue_policy);
    auto created =
        channel_internal::create_inline_payload_publisher(impl_->context_,
                                                          binding + ".clock_health.v1",
                                                          channel_internal::kHealthControlBytes,
                                                          health_endpoint_options,
                                                          impl_->context_.pubsub_capabilities(),
                                                          health_pressure);
    if (!created) {
      return make_unexpected(std::move(created).error());
    }
    health = std::move(created).value();
  }
  if (enforce_peer_requirement) {
    auto attached = channel_internal::wait_for_required_subscribers(
        *endpoint_record, options, **data, health.get());
    if (!attached) {
      return make_unexpected(std::move(attached).error());
    }
  }
  const ClockId capture_clock = detail::ChannelEndpointCatalogAccess::capture_clock_id(endpoint);
  return std::make_unique<channel_internal::CatalogInlinePayloadPublisher>(
      std::move(data).value(),
      std::move(health),
      std::move(payload_identity),
      payload_size,
      framed_payload_size,
      capture_clock,
      detail::ChannelEndpointCatalogAccess::clock_count(endpoint),
      health_options,
      channel_internal::replaces_oldest_unread_sample(endpoint_options.queue_policy),
      endpoint_record->direction == TopicDirection::kExport);
}

expected<std::unique_ptr<channel_internal::InlinePayloadSubscriber>, Error>
ChannelRuntime::create_catalog_inline_payload_subscriber(
    ChannelEndpointCatalog::EndpointView endpoint, PayloadIdentityRecord payload_identity,
    std::size_t payload_size, ChannelAttachOptions options, TopicDirection subscriber_direction,
    const bool enforce_peer_requirement) {
  if (!impl_) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  const detail::ChannelEndpointRecord* endpoint_record =
      detail::ChannelEndpointCatalogAccess::record(endpoint);
  if (endpoint_record == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto service_contract = channel_internal::validate_attach(
      *endpoint_record, payload_identity, payload_size, options, subscriber_direction, false);
  if (!service_contract) {
    return make_unexpected(std::move(service_contract).error());
  }
  const std::string binding =
      detail::ChannelEndpointCatalogAccess::backend_service_binding(endpoint);
  const std::size_t framed_payload_size =
      static_cast<std::size_t>(service_contract->required_frame_bytes);
  const std::size_t selected_bucket_bytes =
      static_cast<std::size_t>(service_contract->selected_bucket_bytes);
  const ChannelOptions endpoint_options =
      channel_internal::channel_options(*endpoint_record, false);
  auto data =
      channel_internal::create_inline_payload_subscriber(impl_->context_,
                                                         binding,
                                                         selected_bucket_bytes,
                                                         endpoint_options,
                                                         impl_->context_.pubsub_capabilities());
  if (!data) {
    return make_unexpected(std::move(data).error());
  }

  const ClockHealthOptions health_options = endpoint_record->publisher_request
                                                ? endpoint_record->publisher_request->clock_health
                                                : ClockHealthOptions{};
  std::unique_ptr<channel_internal::InlinePayloadSubscriber> health;
  if (health_options.capability == ClockHealthCapability::kStatusReports) {
    const ChannelOptions health_endpoint_options =
        channel_internal::health_channel_options(endpoint_options);
    auto created =
        channel_internal::create_inline_payload_subscriber(impl_->context_,
                                                           binding + ".clock_health.v1",
                                                           channel_internal::kHealthControlBytes,
                                                           health_endpoint_options,
                                                           impl_->context_.pubsub_capabilities());
    if (!created) {
      return make_unexpected(std::move(created).error());
    }
    health = std::move(created).value();
  }
  if (enforce_peer_requirement) {
    auto attached = channel_internal::wait_for_required_publishers(
        *endpoint_record, options, **data, health.get());
    if (!attached) {
      return make_unexpected(std::move(attached).error());
    }
  }
  return std::make_unique<channel_internal::CatalogInlinePayloadSubscriber>(
      std::move(data).value(),
      std::move(health),
      std::move(payload_identity),
      payload_size,
      framed_payload_size,
      detail::ChannelEndpointCatalogAccess::capture_clock_id(endpoint),
      detail::ChannelEndpointCatalogAccess::clock_count(endpoint),
      channel_internal::replaces_oldest_unread_sample(endpoint_options.queue_policy));
}

}  // namespace holoscan
