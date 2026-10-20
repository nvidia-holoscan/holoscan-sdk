/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/holoscan_core_export.h>
#include <holoscan/channel/clock_health.hpp>
#include <holoscan/channel/endpoint_concepts.hpp>
#include <holoscan/channel/endpoint_interfaces.hpp>
#include <holoscan/channel/readiness_handle.hpp>
#include <holoscan/channel/types.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/payload_kind.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/time/time_point.hpp>

#include <array>
#include <bit>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stop_token>
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

template <typename T>
inline constexpr std::size_t kSubscriberEndpointScratchCapacity = [] {
  if constexpr (CatalogChannelSchema<T>) {
    return generated_schema_max_serialized_size_v<T>;
  }
  return sizeof(T);
}();

/// @brief Admit one received frame whose whole width is bit-cast into persistent endpoint scratch.
/// @tparam T Typed payload reconstructed from the complete scratch array.
/// @param received Erased receive result carrying both the written length and the producer claim.
/// @return Success, or ErrorCode::kProtocolError when the frame does not cover every scratch byte.
/// @note The written length and the producer's claimed width are independent. Requiring both to
/// equal `sizeof(T)` is what keeps a short frame from combining with bytes an earlier sample left
/// in the reused scratch. A peer may claim the full width while submitting fewer bytes.
template <typename T>
[[nodiscard]] expected<void, Error> admit_typed_frame(
    const InlinePayloadReceiveResult& received) noexcept {
  if (received.payload_size != sizeof(T) || received.metadata.byte_size != sizeof(T)) {
    return make_unexpected(
        Error{ErrorCode::kProtocolError, "received payload byte size does not match its type"});
  }
  return {};
}

/// @brief Admit one received serialized-schema frame and return its verified byte count.
/// @param received Erased receive result carrying both the written length and the producer claim.
/// @param scratch_capacity Bytes available in the caller's persistent receive scratch.
/// @return Serialized length safe to materialize, or ErrorCode::kProtocolError.
/// @note Materialization must never read past the bytes this operation wrote, so the written
/// length has to agree with the claimed length before either bounds the decoded span.
[[nodiscard]] inline expected<std::size_t, Error> admitted_schema_frame(
    const InlinePayloadReceiveResult& received, const std::size_t scratch_capacity) noexcept {
  if (received.metadata.byte_size == 0U || received.metadata.byte_size > scratch_capacity ||
      received.payload_size != received.metadata.byte_size) {
    return make_unexpected(
        Error{ErrorCode::kProtocolError, "received schema byte size exceeds its package bound"});
  }
  return expected<std::size_t, Error>{received.payload_size};
}

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
/// owned bounded scratch before publication. The name-based compatibility factory accepts owning
/// generated schemas through that same bounded codec and retains the host-object path for types
/// satisfying \ref ChannelInlinePayload.
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
/// \c T only after successful verification. The name-based compatibility factory accepts owning
/// generated schemas through that same bounded codec and retains the host-object path for types
/// satisfying \ref ChannelInlinePayload. Receive operations consume one queued sample and calls on
/// one facade are not internally serialized. Moving transfers the endpoint; the source becomes
/// closed, its checked operations return \ref ErrorCode::kClosed, and \ref has_publishers returns
/// false.
///
/// @tparam T Manual inline payload or admitted scalar, raw-POD, or owning generated-schema type.
template <typename T>
  requires(ChannelInlinePayload<T> || CatalogChannelPayload<T>)
class ChannelSubscriber<T> {
 public:
  ChannelSubscriber(ChannelSubscriber&& other) noexcept = default;
  ChannelSubscriber& operator=(ChannelSubscriber&& other) noexcept {
    if (this != &other) {
      static_cast<void>(channel_internal::close_local_subscriber(subscriber_));
      subscriber_ = std::move(other.subscriber_);
      receive_scratch_ = std::move(other.receive_scratch_);
    }
    return *this;
  }
  ~ChannelSubscriber() { static_cast<void>(channel_internal::close_local_subscriber(subscriber_)); }

  ChannelSubscriber(const ChannelSubscriber&) = delete;
  ChannelSubscriber& operator=(const ChannelSubscriber&) = delete;

  /// @brief Retire this process's local subscriber endpoint.
  /// @return Success on the first live close, or a code-only cleanup error. Repeated close and
  /// close on a moved-from facade return @ref ErrorCode::kClosed.
  /// @note This externally serialized, consuming operation publishes no producer end-of-stream
  /// record and supplies no graph-completion evidence. Cancel and join any active waiter before
  /// closing. Already returned owning values retain their ordinary independent lifetime; unread
  /// local data may be discarded. A cleanup failure still consumes the endpoint; later checked
  /// operations return @ref ErrorCode::kClosed.
  [[nodiscard]] expected<void, Error> close() noexcept {
    return channel_internal::close_local_subscriber(subscriber_);
  }

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

  /// @brief Claim an externally pollable readiness descriptor for this subscriber.
  /// @return A move-only handle, or a checked closed, capability, or ownership error.
  /// @retval ErrorCode::kClosed This facade is locally closed, moved from, or already observed
  /// producer end of stream.
  /// @retval ErrorCode::kNotSupported Native descriptor readiness is unavailable on this platform,
  /// or this endpoint was created without event wake.
  /// @retval ErrorCode::kInvalidArgument A wait is in flight, the Runtime readiness pump owns the
  /// subscriber, or another readiness handle is live.
  /// @note Readability means this endpoint may have a sample. Call
  /// \ref ChannelReadinessHandle::acknowledge first, then drain \ref receive or
  /// \ref receive_sample until \ref ErrorCode::kNotReady. A wake arriving during that drain stays
  /// pending. Producer end of stream and a changed producer incarnation also make the descriptor
  /// readable so the next receive reports \ref ErrorCode::kClosed or
  /// \ref ErrorCode::kStaleEndpoint.
  /// @note The handle claims the subscriber's single wait-owner slot. Blocking wait operations
  /// return \ref ErrorCode::kInvalidArgument until the handle is destroyed.
  [[nodiscard]] expected<ChannelReadinessHandle, Error> readiness() {
    if (!subscriber_ || stream_closed_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    return subscriber_->readiness();
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

  /// @brief Refresh the backend mappings consulted by \ref has_publishers.
  /// @return Success, or a checked closed or backend refresh error.
  /// @note This cold operation does not consume a sample. The publisher-presence value remains an
  /// advisory snapshot and does not prove that a following receive will succeed.
  /// @note The backend's readiness query also refreshes receiver-side publisher mappings. This
  /// method deliberately calls `has_samples()` for that side effect and discards its snapshot.
  [[nodiscard]] expected<void, Error> refresh_connections() {
    if (!subscriber_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    if (stream_closed_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    auto refreshed = subscriber_->has_samples();
    if (!refreshed) {
      return make_unexpected(std::move(refreshed).error());
    }
    return {};
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
  /// @note Observing \ref ErrorCode::kClosed here latches end of stream for this facade, so a
  /// later \ref wait_for_sample or \ref wait_and_receive_sample reports \ref ErrorCode::kClosed
  /// rather than waiting out its deadline. The result of this call is unaffected.
  [[nodiscard]] expected<T, Error> receive() {
    if (!subscriber_) {
      return make_unexpected(
          channel_internal::make_channel_error(ErrorCode::kClosed, "channel subscriber is closed"));
    }
    if constexpr (CatalogChannelSchema<T>) {
      auto received = subscriber_->receive(receive_scratch_);
      if (!received) {
        // Latch here and not only in wait_and_receive_sample. The transport surfaces end of
        // stream exactly once and then reads as an empty, never-ready endpoint, so a caller that
        // consumed it through a non-waiting path would leave a later wait with nothing to
        // observe: it would block for its whole deadline and report kTimeout for a stream that
        // has already finished. Set-only, so this call's own result is unchanged.
        if (received.error().code == ErrorCode::kClosed) {
          stream_closed_ = true;
        }
        return make_unexpected(std::move(received).error());
      }
      auto admitted = channel_internal::admitted_schema_frame(*received, receive_scratch_.size());
      if (!admitted) {
        return make_unexpected(std::move(admitted).error());
      }
      return channel_internal::materialize_catalog_schema<T>(
          std::span<const std::byte>(receive_scratch_).first(*admitted));
    } else {
      if constexpr (ChannelSchemaEnvelopePayload<T>) {
        receive_scratch_.fill(std::byte{});
      }
      auto received = subscriber_->receive(receive_scratch_);
      if (!received) {
        // Same one-shot end-of-stream latch as above.
        if (received.error().code == ErrorCode::kClosed) {
          stream_closed_ = true;
        }
        return make_unexpected(std::move(received).error());
      }
      auto admitted = channel_internal::admit_typed_frame<T>(*received);
      if (!admitted) {
        return make_unexpected(std::move(admitted).error());
      }
      T value = std::bit_cast<T>(receive_scratch_);
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
  /// @note Observing \ref ErrorCode::kClosed here latches end of stream for this facade, so a
  /// later \ref wait_for_sample or \ref wait_and_receive_sample reports \ref ErrorCode::kClosed
  /// rather than waiting out its deadline. The result of this call is unaffected.
  template <PublishedPayload U = T>
    requires std::same_as<U, T>
  // The generated-schema and POD paths intentionally share one terminal/error-latching facade.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<Sample<U>, Error> receive_sample() {
    if (!subscriber_) {
      return make_unexpected(
          channel_internal::make_channel_error(ErrorCode::kClosed, "channel subscriber is closed"));
    }
    if constexpr (CatalogChannelSchema<T>) {
      auto received = subscriber_->receive(receive_scratch_);
      if (!received) {
        // Same one-shot end-of-stream latch as above.
        if (received.error().code == ErrorCode::kClosed) {
          stream_closed_ = true;
        }
        return make_unexpected(std::move(received).error());
      }
      auto admitted = channel_internal::admitted_schema_frame(*received, receive_scratch_.size());
      if (!admitted) {
        return make_unexpected(std::move(admitted).error());
      }
      auto value = channel_internal::materialize_catalog_schema<T>(
          std::span<const std::byte>(receive_scratch_).first(*admitted));
      if (!value) {
        return make_unexpected(std::move(value).error());
      }
      return Sample<U>(std::move(*value), received->metadata);
    } else {
      if constexpr (ChannelSchemaEnvelopePayload<U>) {
        receive_scratch_.fill(std::byte{});
      }
      auto received = subscriber_->receive(receive_scratch_);
      if (!received) {
        // Same one-shot end-of-stream latch as above.
        if (received.error().code == ErrorCode::kClosed) {
          stream_closed_ = true;
        }
        return make_unexpected(std::move(received).error());
      }
      auto admitted = channel_internal::admit_typed_frame<U>(*received);
      if (!admitted) {
        return make_unexpected(std::move(admitted).error());
      }
      U value = std::bit_cast<U>(receive_scratch_);
      if constexpr (ChannelSchemaEnvelopePayload<U>) {
        auto valid = channel_internal::validate_received_compatibility_schema_payload(value);
        if (!valid) {
          return make_unexpected(std::move(valid).error());
        }
      }
      return Sample<U>(std::move(value), received->metadata);
    }
  }

  /// @brief Block until a sample is available, then receive and consume it.
  ///
  /// A receive is attempted before any wait is built, so an already queued sample costs no wait
  /// context and no kernel wait. \ref ErrorCode::kNotReady never escapes: readiness is a hint, so
  /// an attempt that loses its race waits again against the same absolute \p deadline.
  ///
  /// @tparam U Core-admitted published payload, constrained to this endpoint's payload type.
  /// @param deadline Steady-clock coordinate after which the wait reports a timeout.
  /// @return The sample on success, or the first checked error.
  /// @retval ErrorCode::kTimeout \p deadline elapsed. Retryable; endpoint state is unchanged.
  /// @retval ErrorCode::kClosed The producer published end of stream, this facade was locally
  /// closed, or this facade was moved from. Each means no later sample arrives through this
  /// facade, and it latches the state so every later call reports it.
  /// @retval ErrorCode::kStaleEndpoint The producer incarnation changed; rebuild the subscriber.
  /// @retval ErrorCode::kNotSupported This endpoint was created with
  /// \c ChannelOptions::enable_event_wake disabled, so it has no kernel event lane. Reported only
  /// when a wait is actually required, and before any wait context is built. This operation
  /// deliberately does not fall back to polling.
  /// @retval ErrorCode::kInvalidArgument This subscriber's readiness is already owned by another
  /// wait set, such as the Runtime readiness pump, or another wait on this subscriber is already
  /// in flight.
  /// @retval ErrorCode::kFailure An inline payload loan is still active. Reported by the first
  /// receive attempt, before any wait.
  /// @note \ref ErrorCode::kProtocolError, \ref ErrorCode::kCodecError,
  /// \ref ErrorCode::kReceiveError, and \ref ErrorCode::kResourceExhausted pass through
  /// unchanged; they are never reclassified.
  /// @note There is no duration overload by design. An absolute deadline cannot silently restart
  /// its budget on a spurious wake; callers pass \c std::chrono::steady_clock::now() plus a bound.
  /// @note Readiness is a hint, so a subscriber that keeps reporting it while every receive loses
  /// its race retries at full speed until \p deadline. The absolute deadline is the only bound on
  /// that loop; there is no backoff. It is not reachable by an idle endpoint -- something has to be
  /// producing readiness that no receive can consume, such as a competing consumer on the same
  /// endpoint -- but a caller that shares a subscriber should size the deadline accordingly.
  /// @warning One subscriber supports one waiting thread and calls on one facade are not
  /// internally serialized. A second concurrent waiter is rejected with
  /// \ref ErrorCode::kInvalidArgument rather than sharing the first one's wait context. The facade
  /// must outlive an in-flight wait and must not be moved while one is in flight.
  template <PublishedPayload U = T>
    requires std::same_as<U, T>
  [[nodiscard]] expected<Sample<U>, Error> wait_and_receive_sample(
      const std::chrono::steady_clock::time_point deadline) {
    return wait_and_receive_sample<U>(deadline, std::stop_token{});
  }

  /// @brief Block until a sample is available or a stop is requested, then receive and consume it.
  ///
  /// @tparam U Core-admitted published payload, constrained to this endpoint's payload type.
  /// @param deadline Steady-clock coordinate after which the wait reports a timeout.
  /// @param token Cancellation observer for this call; a default-constructed token never cancels.
  /// @return The sample on success, or the first checked error.
  /// @retval ErrorCode::kInterrupted \p token requested a stop. Caller-local and non-terminal.
  /// @note Every other retval of the single-argument overload applies unchanged.
  /// @note Cancellation is scoped to one call. A stop that fires ends this wait and leaves the
  /// subscriber usable; the next wait arms a fresh cancellation trigger.
  template <PublishedPayload U = T>
    requires std::same_as<U, T>
  [[nodiscard]] expected<Sample<U>, Error> wait_and_receive_sample(
      const std::chrono::steady_clock::time_point deadline, const std::stop_token& token) {
    if (!subscriber_) {
      return make_unexpected(
          channel_internal::make_channel_error(ErrorCode::kClosed, "channel subscriber is closed"));
    }
    if (stream_closed_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    auto received = channel_internal::wait_and_receive_next(
        *subscriber_, deadline, channel_internal::borrowed_stop_token(token), [this] {
          return receive_sample<U>();
        });
    // A manually created inline endpoint reports end of stream exactly once and then reads as an
    // empty, never-ready endpoint, which would turn every later call into a full-deadline
    // kTimeout. Latch it here so kClosed stays terminal for this facade on both endpoint kinds.
    if (!received && received.error().code == ErrorCode::kClosed) {
      stream_closed_ = true;
    }
    return received;
  }

  /// @brief Block until this endpoint may have a sample, without consuming one.
  ///
  /// @param deadline Steady-clock coordinate after which the wait reports a timeout.
  /// @param token Cancellation observer for this call; a default-constructed token never cancels.
  /// @return Empty success when a sample may be available, or a checked wait error.
  /// @retval ErrorCode::kTimeout \p deadline elapsed with no readiness.
  /// @retval ErrorCode::kInterrupted \p token requested a stop.
  /// @retval ErrorCode::kClosed This facade was locally closed or moved from, or a previous
  /// \ref receive, \ref receive_sample, or \ref wait_and_receive_sample observed end of stream.
  /// @retval ErrorCode::kNotSupported This endpoint was created with
  /// \c ChannelOptions::enable_event_wake disabled and has no queued sample.
  /// @retval ErrorCode::kInvalidArgument This subscriber's readiness is already owned by another
  /// wait set, or another wait on this subscriber is already in flight.
  /// @note Success is a hint, exactly like \ref has_samples. A following \ref receive or
  /// \ref receive_sample may still report \ref ErrorCode::kNotReady.
  /// @note An already ready endpoint returns success without consulting the event lane, so a
  /// poll-only endpoint that already holds a sample is not refused with
  /// \ref ErrorCode::kNotSupported.
  [[nodiscard]] expected<void, Error> wait_for_sample(
      const std::chrono::steady_clock::time_point deadline, const std::stop_token& token = {}) {
    if (!subscriber_) {
      return make_unexpected(
          channel_internal::make_channel_error(ErrorCode::kClosed, "channel subscriber is closed"));
    }
    if (stream_closed_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    return channel_internal::wait_for_sample_readiness(
        *subscriber_, deadline, channel_internal::borrowed_stop_token(token));
  }

 private:
  friend class ChannelRuntime;
  friend struct channel_internal::InlinePayloadTestAccess;

  explicit ChannelSubscriber(
      std::unique_ptr<channel_internal::InlinePayloadSubscriber> subscriber) noexcept
      : subscriber_(std::move(subscriber)) {}

  std::unique_ptr<channel_internal::InlinePayloadSubscriber> subscriber_;
  /// Latched once a wait observed end of stream, so kClosed stays terminal for the waiting API
  /// even on an endpoint whose transport reports it only once.
  bool stream_closed_{false};
  alignas(std::max_align_t) std::array<
      std::byte, channel_internal::kSubscriberEndpointScratchCapacity<T>> receive_scratch_{};
};

}  // namespace holoscan
