/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/holoscan_core_export.h>
#include <holoscan/channel/endpoint_concepts.hpp>
#include <holoscan/channel/endpoint_interfaces.hpp>
#include <holoscan/channel/types.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/memory/tensor_storage.hpp>

#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace holoscan {

namespace channel_internal {

struct InlinePayloadTestAccess;

template <typename T>
inline constexpr std::size_t kSchemaEndpointScratchCapacity = [] {
  if constexpr (CatalogChannelSchema<T>) {
    return generated_schema_max_serialized_size_v<T>;
  }
  return std::size_t{1U};
}();

template <CatalogChannelSchema T>
[[nodiscard]] expected<T, Error> materialize_catalog_schema(
    std::span<const std::byte> payload) noexcept {
  auto value = schema_package<T>().materialize(payload);
  if (!value && value.error().code == ErrorCode::kCodecError) {
    Error error = std::move(value).error();
    error.code = ErrorCode::kProtocolError;
    return make_unexpected(std::move(error));
  }
  return value;
}

template <ChannelSchemaEnvelopePayload T>
[[nodiscard]] expected<void, Error> prepare_compatibility_schema_payload(T& payload) {
  payload.endpoint_generation = kCompatibilitySchemaEndpointGeneration;
  return payload.validate();
}

template <ChannelSchemaEnvelopePayload T>
[[nodiscard]] expected<void, Error> validate_received_compatibility_schema_payload(
    const T& payload) {
  auto valid = payload.validate();
  if (!valid || payload.endpoint_generation != kCompatibilitySchemaEndpointGeneration) {
    return make_unexpected(
        Error{ErrorCode::kProtocolError, "received compatibility schema envelope is invalid"});
  }
  return {};
}

}  // namespace channel_internal

/// @brief Publish typed values and public metadata through one Channel endpoint.
///
/// The move-only facade owns its endpoint but borrows the parent \ref ChannelRuntime, which must
/// outlive it. Catalog attachment is available only when \c T satisfies
/// \ref CatalogChannelPayload. The catalog binding owns scalar/raw-POD wire conversion and the
/// exact generated-schema envelope. An owning generated-schema facade serializes into endpoint-
/// owned bounded scratch before publication. The manual name-based compatibility factory remains
/// available only when \c T satisfies \ref ChannelInlinePayload and retains its host-object path.
/// Calls on one facade are not internally serialized. Moving transfers the endpoint; the source
/// becomes closed, its checked operations return \ref ErrorCode::kClosed, and
/// \ref has_subscribers returns false.
///
/// @tparam T Manual inline payload or admitted scalar, raw-POD, or owning generated-schema type.
template <typename T>
  requires(ChannelInlinePayload<T> || CatalogChannelPayload<T>)
class ChannelPublisher<T> {
 public:
  ChannelPublisher(ChannelPublisher&& other) noexcept = default;
  ChannelPublisher& operator=(ChannelPublisher&& other) noexcept = default;
  ~ChannelPublisher() = default;

  ChannelPublisher(const ChannelPublisher&) = delete;
  ChannelPublisher& operator=(const ChannelPublisher&) = delete;

  /// @brief Emit an inline payload value and optional semantic metadata.
  /// @param value Value copied or moved into the by-value argument, then encoded for publication.
  /// @param options Optional capture time, frame, trace, and validity flags.
  /// @return Recipient count after accepted publication, or a checked closed, metadata, pressure,
  /// resource, or publication error.
  /// @note A catalog endpoint that requires capture time rejects an absent or foreign-domain
  /// \ref EmitOptions::capture_time. A terminal post-submit result returns either \ref
  /// ErrorCode::kPublicationPartial with exact counts or \ref ErrorCode::kPublicationIndeterminate
  /// with unknown visibility; neither may be retried as the same sample.
  [[nodiscard]] expected<std::uint32_t, Error> emit(T value, EmitOptions options = {}) {
    if (!publisher_) {
      return make_unexpected(
          channel_internal::make_channel_error(ErrorCode::kClosed, "channel publisher is closed"));
    }
    SampleMetadata metadata{};
    if (options.capture_time) {
      metadata.capture_timestamp_ns = options.capture_time->timestamp_ns;
      metadata.source_clock_id = options.capture_time->clock_id;
    }
    metadata.frame_id = options.frame_id.value_or(0U);
    metadata.trace_id = options.trace ? options.trace->trace_id : 0U;
    metadata.flags = options.flags;
    if constexpr (CatalogChannelSchema<T>) {
      auto encoded = schema_package<T>().encode(value, schema_scratch_);
      if (!encoded) {
        return make_unexpected(std::move(encoded).error());
      }
      metadata.byte_size = static_cast<std::uint32_t>(*encoded);
      metadata.payload_kind = PayloadKind::kSerializedBytes;
      return publisher_->emit(std::span<const std::byte>(schema_scratch_).first(*encoded),
                              metadata);
    } else {
      if constexpr (ChannelSchemaEnvelopePayload<T>) {
        auto valid = channel_internal::prepare_compatibility_schema_payload(value);
        if (!valid) {
          return make_unexpected(std::move(valid).error());
        }
      }
      metadata.byte_size = sizeof(T);
      metadata.payload_kind = PayloadKind::kInlineTyped;
      const auto bytes = std::as_bytes(std::span<const T, 1U>(&value, 1U));
      return publisher_->emit(bytes, metadata);
    }
  }

  /// @brief Emit a value with a capture coordinate in the bound catalog clock.
  /// @param value Value copied or moved into the by-value argument, then encoded for publication.
  /// @param capture_timestamp_ns Exact caller-supplied capture coordinate.
  /// @param options Remaining optional metadata; an existing capture time is invalid.
  /// @return Recipient count, or a checked capture, closed, pressure, or publication error.
  /// @note This operation labels \p capture_timestamp_ns with the catalog-bound capture clock; it
  /// does not read a clock, perform synchronization, or substitute publication time.
  [[nodiscard]] expected<std::uint32_t, Error> emit_captured(T value,
                                                             std::int64_t capture_timestamp_ns,
                                                             EmitOptions options = {}) {
    if (!publisher_) {
      return make_unexpected(
          channel_internal::make_channel_error(ErrorCode::kClosed, "channel publisher is closed"));
    }
    if (options.capture_time) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto capture_time = make_capture_time(capture_timestamp_ns);
    if (!capture_time) {
      return make_unexpected(std::move(capture_time).error());
    }
    options.capture_time = *capture_time;
    return emit(std::move(value), std::move(options));
  }

  /// @brief Label one coordinate with this catalog publisher's capture clock.
  /// @param timestamp_ns Exact caller-supplied nanosecond coordinate.
  /// @return Bound time point, or a checked error when closed or not catalog-authorized.
  [[nodiscard]] expected<TimePoint, Error> make_capture_time(
      std::int64_t timestamp_ns) const noexcept {
    if (!publisher_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    return publisher_->make_capture_time(timestamp_ns);
  }

  /// @brief Emit an inline payload sample and its public metadata.
  ///
  /// @tparam U Core-admitted published payload, constrained to this endpoint's payload type.
  /// @param sample Value and metadata copied or moved into the by-value argument; byte size and
  /// payload kind are normalized before publication.
  /// @return Recipient count after accepted publication, or a checked closed, metadata, pressure,
  /// resource, or publication error.
  /// @note A post-submit result that cannot prove its delivery outcome returns
  /// \ref ErrorCode::kPublicationIndeterminate or \ref ErrorCode::kPublicationPartial and must not
  /// be retried as the same sample.
  template <PublishedPayload U = T>
    requires std::same_as<U, T>
  [[nodiscard]] expected<std::uint32_t, Error> emit_sample(Sample<U> sample) {
    if (!publisher_) {
      return make_unexpected(
          channel_internal::make_channel_error(ErrorCode::kClosed, "channel publisher is closed"));
    }

    SampleMetadata metadata = sample.metadata;
    if constexpr (CatalogChannelSchema<T>) {
      auto encoded = schema_package<T>().encode(sample.data, schema_scratch_);
      if (!encoded) {
        return make_unexpected(std::move(encoded).error());
      }
      metadata.byte_size = static_cast<std::uint32_t>(*encoded);
      metadata.payload_kind = PayloadKind::kSerializedBytes;
      return publisher_->emit(std::span<const std::byte>(schema_scratch_).first(*encoded),
                              metadata);
    } else {
      if constexpr (ChannelSchemaEnvelopePayload<T>) {
        auto valid = channel_internal::prepare_compatibility_schema_payload(sample.data);
        if (!valid) {
          return make_unexpected(std::move(valid).error());
        }
      }
      metadata.byte_size = sizeof(T);
      metadata.payload_kind = PayloadKind::kInlineTyped;
      const auto bytes = std::as_bytes(std::span<const U, 1U>(&sample.data, 1U));
      return publisher_->emit(bytes, metadata);
    }
  }

  /// @brief Publish an explicit end-of-stream operation and close this publisher generation.
  /// @return Success after the close was accepted by every frozen recipient.
  /// @retval ErrorCode::kNotReady The close was rejected before submission; the publisher remains
  /// usable.
  /// @retval ErrorCode::kClosed This facade is moved from or already closed.
  /// @retval ErrorCode::kPublicationIndeterminate Submission began but its terminal outcome is
  /// unknown; this facade becomes closed and the operation must not be retried.
  /// @retval ErrorCode::kPublicationPartial Submission reached a known strict subset; this facade
  /// becomes closed and the operation must not be retried.
  /// @retval ErrorCode::kProtocolError The backend reported a malformed recipient outcome; this
  /// facade becomes closed and the operation must not be retried.
  /// @retval ErrorCode::kInvalidArgument The backend reported an impossible recipient count; this
  /// facade becomes closed and the operation must not be retried.
  /// @note This is the only operation that publishes EOS. Destruction, disconnect, silence, and
  /// process exit do not publish or imply EOS.
  /// @note With a lossy replacement policy, EOS uses a bounded protected terminal lane and a
  /// recipient drains its data lane before observing EOS. New data may replace stale data, but EOS
  /// cannot replace the final unread data sample accepted by that recipient.
  /// QueuePolicy::kBlocking keeps data and EOS on its existing lossless FIFO lane.
  [[nodiscard]] expected<void, Error> close() {
    if (!publisher_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    auto closed = publisher_->close_stream();
    if (closed || publication_error_consumes_authority(closed.error().code) ||
        closed.error().code == ErrorCode::kProtocolError ||
        closed.error().code == ErrorCode::kInvalidArgument) {
      publisher_.reset();
    }
    return closed;
  }

  /// @brief Submit one publisher-bound clock-health control report.
  /// @param report Assessment and portable primary reason.
  /// @return Success after local bounded transport acceptance; this is not receiver
  /// acknowledgment and does not prove that Runtime accepted or applied the report.
  /// @retval ErrorCode::kClosed This facade is moved from or otherwise closed.
  /// @retval ErrorCode::kInvalidArgument The endpoint has no admitted status-report capability or
  /// the assessment/reason combination is invalid.
  /// @retval ErrorCode::kNotReady The checked health lane has no connected subscriber.
  /// @retval ErrorCode::kResourceExhausted A bounded sequence or transport resource is exhausted.
  /// @retval ErrorCode::kPublicationIndeterminate Submission began but its terminal outcome is
  /// unknown; the committed operation identity is consumed and the caller must not retry the same
  /// logical report.
  /// @retval ErrorCode::kPublicationPartial Submission reached a known strict subset; the
  /// committed operation identity is consumed and the caller must not retry the same report.
  /// @note A failure that occurs before backend submission leaves operation/report sequence and
  /// receiver freshness state unchanged. Calls on one publisher are externally serialized with
  /// every data publication.
  [[nodiscard]] expected<void, Error> report_clock_health(ClockHealthReport report) {
    if (!publisher_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    return publisher_->report_clock_health(report);
  }

  /// @brief Refresh backend connection state.
  /// @return Success, or an error when closed or when the backend refresh fails.
  /// @note This operation refreshes only the advisory snapshot returned by
  /// \ref has_subscribers; it does not wait for a peer or guarantee the next publication.
  [[nodiscard]] expected<void, Error> update_connections() noexcept {
    if (!publisher_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    return publisher_->update_connections();
  }

  /// @brief Return whether at least one subscriber was observed by the latest refresh.
  /// @return True when a subscriber was observed; false when closed or disconnected.
  /// @note This advisory snapshot is not proof that the next publication has a recipient.
  [[nodiscard]] bool has_subscribers() const noexcept {
    return publisher_ && publisher_->has_subscribers();
  }

 private:
  friend class ChannelRuntime;
  friend struct channel_internal::InlinePayloadTestAccess;

  explicit ChannelPublisher(
      std::unique_ptr<channel_internal::InlinePayloadPublisher> publisher) noexcept
      : publisher_(std::move(publisher)) {}

  std::unique_ptr<channel_internal::InlinePayloadPublisher> publisher_;
  alignas(std::max_align_t)
      std::array<std::byte, channel_internal::kSchemaEndpointScratchCapacity<T>> schema_scratch_{};
};

/// @brief Consume typed values and public metadata from one Channel endpoint.
///
/// The move-only facade owns its endpoint but borrows the parent \ref ChannelRuntime, which must
/// outlive it. Catalog attachment is available only when \c T satisfies
/// \ref CatalogChannelPayload. The catalog binding decodes canonical scalar/raw-POD fields and
/// verifies the generated-schema envelope. An owning generated-schema facade receives serialized
/// bytes into endpoint-owned bounded scratch, invokes the exact package verifier, and materializes
/// \c T only after successful verification. The manual name-based compatibility factory remains
/// available only when \c T satisfies \ref ChannelInlinePayload. Receive operations consume one
/// queued sample and calls on one facade are not internally serialized. Moving transfers the
/// endpoint; the source becomes closed, its checked operations return \ref ErrorCode::kClosed, and
/// \ref has_publishers returns false.
///
/// @tparam T Manual inline payload or admitted scalar, raw-POD, or owning generated-schema type.
template <typename T>
  requires(ChannelInlinePayload<T> || CatalogChannelPayload<T>)
class ChannelSubscriber<T> {
 public:
  ChannelSubscriber(ChannelSubscriber&& other) noexcept = default;
  ChannelSubscriber& operator=(ChannelSubscriber&& other) noexcept = default;
  ~ChannelSubscriber() = default;

  ChannelSubscriber(const ChannelSubscriber&) = delete;
  ChannelSubscriber& operator=(const ChannelSubscriber&) = delete;

  /// @brief Query pending inline samples without consuming one.
  /// @return True or false readiness, or an error when closed or when the backend query fails.
  /// @note Readiness is a snapshot; another endpoint or operation may change it before
  /// \ref receive or \ref receive_sample.
  [[nodiscard]] expected<bool, Error> has_samples() const {
    if (!subscriber_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    return subscriber_->has_samples();
  }

  /// @brief Return whether the endpoint currently observes at least one publisher.
  /// @return True when a publisher is connected; false when closed or disconnected.
  /// @note This cold-path advisory observation is not receive readiness and does not prove that a
  /// sample will become available or that a publication will be delivered. It is intended for
  /// optional-peer discovery and diagnostics; callers that require an attachment deadline must
  /// impose their own bounded wait. Graph startup must not wait on this observation for an
  /// optional peer.
  [[nodiscard]] bool has_publishers() const noexcept {
    return subscriber_ && subscriber_->has_publishers();
  }

  /// @brief Receive and consume the next typed payload value.
  /// @return Owning value on success, \ref ErrorCode::kClosed or
  /// \ref ErrorCode::kNotReady when unavailable, or a checked receive/protocol error.
  /// @note Scalar/raw-POD paths bit-cast only after complete decode. Generated-schema paths verify
  /// bounded serialized bytes and then return a new owning native object.
  /// @note A catalog-attached monitored endpoint applies at most one private operation per call.
  /// A clock-health-only operation is applied in sequence and returns \ref ErrorCode::kNotReady;
  /// retry to receive a later data operation without allowing control traffic to create an
  /// unbounded call.
  [[nodiscard]] expected<T, Error> receive() {
    if (!subscriber_) {
      return make_unexpected(
          channel_internal::make_channel_error(ErrorCode::kClosed, "channel subscriber is closed"));
    }
    if constexpr (CatalogChannelSchema<T>) {
      auto received = subscriber_->receive(schema_scratch_);
      if (!received) {
        return make_unexpected(std::move(received).error());
      }
      if (received->byte_size == 0U || received->byte_size > schema_scratch_.size()) {
        return make_unexpected(Error{ErrorCode::kProtocolError,
                                     "received schema byte size exceeds its package bound"});
      }
      return channel_internal::materialize_catalog_schema<T>(
          std::span<const std::byte>(schema_scratch_).first(received->byte_size));
    } else {
      std::array<std::byte, sizeof(T)> bytes{};
      auto received = subscriber_->receive(bytes);
      if (!received) {
        return make_unexpected(std::move(received).error());
      }
      T value = std::bit_cast<T>(bytes);
      if constexpr (ChannelSchemaEnvelopePayload<T>) {
        auto valid = channel_internal::validate_received_compatibility_schema_payload(value);
        if (!valid) {
          return make_unexpected(std::move(valid).error());
        }
      }
      return value;
    }
  }

  /// @brief Receive and consume the next typed payload and its public metadata.
  ///
  /// @tparam U Core-admitted published payload, constrained to this endpoint's payload type.
  /// @return The sample on success, or \ref ErrorCode::kClosed,
  /// \ref ErrorCode::kNotReady, or a receive failure.
  /// @note Scalar/raw-POD paths bit-cast only after complete decode. Generated-schema paths verify
  /// bounded serialized bytes and then return a new owning native object.
  /// @note A catalog-attached monitored endpoint applies at most one private operation per call.
  /// A clock-health-only operation is applied in sequence and returns \ref ErrorCode::kNotReady.
  template <PublishedPayload U = T>
    requires std::same_as<U, T>
  [[nodiscard]] expected<Sample<U>, Error> receive_sample() {
    if (!subscriber_) {
      return make_unexpected(
          channel_internal::make_channel_error(ErrorCode::kClosed, "channel subscriber is closed"));
    }
    if constexpr (CatalogChannelSchema<T>) {
      auto received = subscriber_->receive(schema_scratch_);
      if (!received) {
        return make_unexpected(std::move(received).error());
      }
      if (received->byte_size == 0U || received->byte_size > schema_scratch_.size()) {
        return make_unexpected(Error{ErrorCode::kProtocolError,
                                     "received schema byte size exceeds its package bound"});
      }
      auto value = channel_internal::materialize_catalog_schema<T>(
          std::span<const std::byte>(schema_scratch_).first(received->byte_size));
      if (!value) {
        return make_unexpected(std::move(value).error());
      }
      return Sample<U>(std::move(*value), *received);
    } else {
      std::array<std::byte, sizeof(U)> bytes{};
      auto received = subscriber_->receive(bytes);
      if (!received) {
        return make_unexpected(std::move(received).error());
      }
      U value = std::bit_cast<U>(bytes);
      if constexpr (ChannelSchemaEnvelopePayload<U>) {
        auto valid = channel_internal::validate_received_compatibility_schema_payload(value);
        if (!valid) {
          return make_unexpected(std::move(valid).error());
        }
      }
      return Sample<U>(std::move(value), received.value());
    }
  }

 private:
  friend class ChannelRuntime;
  friend struct channel_internal::InlinePayloadTestAccess;

  explicit ChannelSubscriber(
      std::unique_ptr<channel_internal::InlinePayloadSubscriber> subscriber) noexcept
      : subscriber_(std::move(subscriber)) {}

  std::unique_ptr<channel_internal::InlinePayloadSubscriber> subscriber_;
  alignas(std::max_align_t)
      std::array<std::byte, channel_internal::kSchemaEndpointScratchCapacity<T>> schema_scratch_{};
};

}  // namespace holoscan
