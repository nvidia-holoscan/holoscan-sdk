/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "channel_payload_limits.hpp"
#include "channel_publisher_pressure.hpp"
#include "protected_terminal_channel.hpp"

#include <holoscan/holoscan_core_export.h>
#include <holoscan/channel/channel.hpp>

#include <holomq/errors.hpp>
#include <holomq/expected.hpp>
#include <holomq/pubsub_capabilities.hpp>
#include <holomq/pubsub_options.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

namespace holoscan::channel_internal {

using TransportCapabilities = holomq::PubSubCapabilities;

/// @brief Translate one successful or independent HoloMQ code by named meaning.
/// @param code Independent HoloMQ-domain code.
/// @return Corresponding Core code, or fail-closed `kFailure` for an unknown future value.
/// @note Failure carriers must use \ref from_holomq_error so an invalid `kSuccess` carrier and an
/// unknown future value retain their stable structural invariant token.
[[nodiscard]] HOLOSCAN_CORE_EXPORT ErrorCode
to_holoscan_error_code(holomq::ErrorCode code) noexcept;

[[nodiscard]] HOLOSCAN_CORE_EXPORT Error from_holomq_error(holomq::Error error) noexcept;

[[nodiscard]] HOLOSCAN_CORE_EXPORT Error from_holomq_error(holomq::Error error,
                                                           std::string_view context) noexcept;

template <typename T>
[[nodiscard]] expected<T, Error> make_unexpected_nothrow(ErrorCode code,
                                                         const char* message) noexcept {
  try {
    return make_unexpected(Error{code, message});
  } catch (...) {
    return make_unexpected(Error{code});
  }
}

[[nodiscard]] expected<void, Error> from_holomq_result(
    holomq::expected<void, holomq::Error> result) noexcept;

[[nodiscard]] expected<bool, Error> from_holomq_result(
    holomq::expected<bool, holomq::Error> result) noexcept;

/// @brief Map a HoloMQ send result around its explicit submission boundary.
/// @param result Raw send result. HoloMQ's distinct incomplete-fanout result proves rejection
/// before backend submission; every other send error is treated as post-submit.
/// @param expected_recipient_count Exact compiler/setup-frozen fanout.
/// @return Exact recipient count only on a complete match, retryable not-ready for proven
/// incomplete attachment, non-retryable partial with exact counts for a known strict subset,
/// protocol/argument failure for invalid counts, or non-retryable indeterminate for another send
/// error.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<std::uint32_t, Error> from_holomq_publication_result(
    holomq::expected<int, holomq::Error> result, std::uint32_t expected_recipient_count) noexcept;

/// @brief Map a HoloMQ send under the graph-owned bounded-pressure boundary.
/// @param result Raw send result.
/// @param expected_recipient_count Exact compiler/setup-frozen fanout.
/// @param graph_retry_boundary `true` only for a zero-history graph publisher with an explicit
/// reject or bounded-retry pressure policy.
/// @return Retryable pressure for a proven single-recipient pre-submit rejection or an exact
/// zero-recipient result; otherwise the ordinary completed-send classification.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<std::uint32_t, Error> from_holomq_publication_result(
    holomq::expected<int, holomq::Error> result, std::uint32_t expected_recipient_count,
    bool graph_retry_boundary) noexcept;

[[nodiscard]] expected<void, Error> validate_queue_options(const ChannelOptions& options);

[[nodiscard]] expected<void, Error> validate_channel_options(const ChannelOptions& options);

[[nodiscard]] expected<void, Error> validate_tensor_channel_options(const ChannelOptions& options);

[[nodiscard]] expected<void, Error> validate_inline_payload_options(const ChannelOptions& options);

[[nodiscard]] holomq::QueuePolicy to_holomq_queue_policy(ChannelQueuePolicy policy) noexcept;

[[nodiscard]] bool replaces_oldest_unread_sample(ChannelQueuePolicy policy) noexcept;

[[nodiscard]] std::uint32_t effective_max_subscriber_queue_depth(
    const ChannelOptions& options) noexcept;

[[nodiscard]] expected<std::size_t, Error> tensor_sidecar_record_capacity(
    const ChannelOptions& options);

[[nodiscard]] holomq::PubSubServiceOptions to_holomq_service_options(
    const ChannelOptions& options) noexcept;

[[nodiscard]] holomq::PublisherPubSubOptions to_holomq_publisher_options(
    const ChannelOptions& options,
    PublisherPressureOptions pressure = PublisherPressureOptions::follow_service()) noexcept;

[[nodiscard]] holomq::SubscriberPubSubOptions to_holomq_subscriber_options(
    const ChannelOptions& options) noexcept;

[[nodiscard]] bool supports_queue_policy(const TransportCapabilities& capabilities,
                                         ChannelQueuePolicy policy) noexcept;

[[nodiscard]] expected<void, Error> validate_capabilities(const TransportCapabilities& capabilities,
                                                          const ChannelOptions& options);

[[nodiscard]] expected<void, Error> validate_inline_payload_capabilities(
    const TransportCapabilities& capabilities, const ChannelOptions& options);

// Descriptor endpoints carry references only. Keep the checked Channel-to-Core mapping in one
// operation so both publication and receipt reject an inline discriminator before exposing a
// descriptor with no corresponding inline bytes.
[[nodiscard]] inline expected<PayloadKind, Error> descriptor_core_payload_kind(
    const ChannelPayloadKind payload_kind) {
  if (payload_kind == ChannelPayloadKind::kInlinePayload) {
    return make_unexpected(Error{
        ErrorCode::kInvalidArgument,
        "descriptor endpoints require a descriptor-backed payload kind",
    });
  }
  return core_payload_kind(payload_kind);
}

enum class InlinePayloadOperation : std::uint8_t {
  kData = 0U,
  kEndOfStream = 1U,
};

template <std::size_t Size>
struct InlinePayloadBucket {
  InlinePayloadOperation operation{InlinePayloadOperation::kData};
  SampleMetadata metadata{};
  std::uint32_t payload_size{};
  alignas(std::max_align_t) std::array<std::byte, Size> bytes{};
};

/// @brief Carry descriptor-schema bytes plus catalog incarnation attribution.
template <std::size_t Size>
struct TensorSchemaPayloadBucket {
  InlinePayloadOperation operation{InlinePayloadOperation::kData};
  SampleMetadata metadata{};
  std::uint64_t producer_generation{};
  std::uint64_t health_report_sequence{};
  std::uint32_t payload_size{};
  alignas(std::max_align_t) std::array<std::byte, Size> bytes{};
};

/// @brief Return whether a sample's publication coordinate names a clock the catalog declares.
///
/// Catalog clock identities are dense and one-based, so a range check is exactly a membership test
/// against the declared clock array. That is the property a peer depends on: an identity inside the
/// range resolves to a clock record, and one outside it cannot. Every catalog payload family shares
/// this rule so a subscriber's guarantee does not depend on which family carried the sample.
///
/// The check deliberately does not pin which declared clock a publication must use. The catalog
/// records no designated publication domain, so requiring one specific identity here would encode
/// an agreement the format does not carry.
///
/// @param metadata Sample metadata under validation.
/// @param catalog_clock_count Number of clocks the owning catalog declares.
/// @return \c true when the coordinate is canonically absent, or names a declared clock.
[[nodiscard]] constexpr bool publication_coordinate_declared(
    const SampleMetadata& metadata, const std::uint64_t catalog_clock_count) noexcept {
  if (!metadata.publish_clock_id.valid()) {
    return metadata.publish_timestamp_ns == 0;
  }
  return metadata.publish_clock_id.value <= catalog_clock_count;
}

/// @brief Return whether a public service id occupies the private terminal namespace.
[[nodiscard]] bool is_ordered_terminal_service_id(std::string_view service_id) noexcept;

/// @brief Deterministically derive the private protected-terminal service identity.
/// @param data_service_id Compiler/setup-frozen logical data service identity.
/// @return Source-private service identity, or an empty string when the public id collides with the
/// reserved suffix or leaves insufficient transport-name capacity.
[[nodiscard]] std::string ordered_terminal_service_id(std::string_view data_service_id);

/// @brief Return the capacity-one blocking service contract for ordered terminal operations.
/// @param options Frozen logical endpoint options whose recipient count and event-wake choice are
/// inherited.
/// @return Private service options with one protected record per recipient.
[[nodiscard]] holomq::PubSubServiceOptions ordered_terminal_service_options(
    const ChannelOptions& options) noexcept;

/// @brief Return the capacity-one publisher contract for ordered terminal operations.
/// @param options Frozen logical endpoint options whose exact fanout is inherited.
/// @return Private publisher options with an exact-recipient pre-submit gate.
[[nodiscard]] holomq::PublisherPubSubOptions ordered_terminal_publisher_options(
    const ChannelOptions& options) noexcept;

/// @brief Return the capacity-one subscriber contract for ordered terminal operations.
/// @return Private subscriber options with one queue and borrow slot.
[[nodiscard]] holomq::SubscriberPubSubOptions ordered_terminal_subscriber_options() noexcept;

static_assert(std::is_standard_layout_v<InlinePayloadBucket<8U>>);
static_assert(std::is_trivially_copyable_v<InlinePayloadBucket<8U>>);
static_assert(std::is_standard_layout_v<TensorSchemaPayloadBucket<8U>>);
static_assert(std::is_trivially_copyable_v<TensorSchemaPayloadBucket<8U>>);

}  // namespace holoscan::channel_internal
