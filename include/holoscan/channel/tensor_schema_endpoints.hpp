/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/holoscan_core_export.h>
#include <holoscan/channel/clock_health.hpp>
#include <holoscan/channel/endpoint_concepts.hpp>
#include <holoscan/channel/endpoint_interfaces.hpp>
#include <holoscan/channel/types.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/time/time_point.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stop_token>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace holoscan {
namespace detail {
class RuntimeChannelBindingAccess;
}  // namespace detail

/** @brief Validated FlatBuffer metadata for a resolved Tensor sample.
 * @tparam Payload Descriptor-backed FlatBuffer payload type.
 * @details Transport-only descriptor fields remain internal; use
 * ChannelResolvedTensorSample::payload for the public runtime payload reference. */
template <ChannelDescriptorFlatBufferPayload Payload>
struct ChannelTensorSchemaMetadata {
  using root_type = typename Payload::root_type;
  static constexpr std::size_t kMetadataCapacity = Payload::metadata_capacity;

  /// Complete compact schema identity verified on the received wrapper.
  SchemaId schema_family_id{};
  /// Reserved transport-profile tag verified on the received wrapper.
  std::uint32_t transport_profile_tag{};
  /// Number of valid serialized metadata bytes.
  std::uint32_t metadata_size{};
  /// Endpoint generation carried by the received schema wrapper.
  std::uint64_t endpoint_generation{};
  /// Fixed-capacity storage for the verified serialized metadata.
  std::array<std::byte, kMetadataCapacity> metadata{};

  /** @brief Return serialized FlatBuffer metadata bytes.
   * @return A bounded view of the validated metadata bytes. */
  [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
    return {metadata.data(),
            metadata_size <= kMetadataCapacity ? metadata_size : kMetadataCapacity};
  }

  /** @brief Return the FlatBuffer root view.
   * @return Root pointer, or nullptr when metadata is empty. */
  [[nodiscard]] const root_type* root() const noexcept {
    return Payload::root_from_metadata(bytes());
  }
};

/** @brief Resolved metadata, Tensor, and public payload reference.
 * @tparam Payload Descriptor-backed FlatBuffer payload type. */
template <ChannelDescriptorFlatBufferPayload Payload>
struct ChannelResolvedTensorSample {
  using payload_type = Payload;
  using root_type = typename Payload::root_type;
  using metadata_type = ChannelTensorSchemaMetadata<Payload>;

  /// Validated schema metadata with transport-only descriptor fields removed.
  metadata_type metadata{};
  /// Transport-neutral sample metadata carried by the Runtime envelope.
  SampleMetadata sample_metadata{};
  /// Tensor materialized from a local sidecar or imported opaque external-buffer reference.
  Tensor tensor{};
  /// Runtime payload reference for the materialized Tensor.
  PayloadDescriptor payload{};

  /// Return a view over the serialized FlatBuffer metadata bytes.
  [[nodiscard]] std::span<const std::byte> metadata_bytes() const noexcept {
    return metadata.bytes();
  }

  /// Return the FlatBuffer root view into metadata, or nullptr if metadata is empty.
  ///
  /// @warning The returned pointer is valid only while this sample object remains alive and
  /// unmodified. The metadata bytes were verifier-checked before receive() returned.
  [[nodiscard]] const root_type* root() const noexcept { return metadata.root(); }
};

/** @brief Resolved metadata and canonically indexed Tensor fields.
 * @tparam Payload Descriptor-backed FlatBuffer payload type. */
template <ChannelDescriptorFlatBufferPayload Payload>
struct ChannelResolvedTensorBundle {
  using payload_type = Payload;
  using root_type = typename Payload::root_type;
  using metadata_type = ChannelTensorSchemaMetadata<Payload>;
  static constexpr std::size_t kTensorCount = Payload::max_tensors;

  /// Validated schema metadata with transport-only descriptor fields removed.
  metadata_type metadata{};
  /// Transport-neutral sample metadata carried by the Runtime envelope.
  SampleMetadata sample_metadata{};
  /// Tensors materialized in canonical generated-field order.
  std::array<Tensor, kTensorCount> tensors{};
  /// Runtime payload references corresponding to the materialized Tensors.
  std::array<PayloadDescriptor, kTensorCount> payloads{};

  /// Return a view over the serialized FlatBuffer metadata bytes.
  [[nodiscard]] std::span<const std::byte> metadata_bytes() const noexcept {
    return metadata.bytes();
  }

  /// Return the FlatBuffer root view into metadata, or nullptr if metadata is empty.
  [[nodiscard]] const root_type* root() const noexcept { return metadata.root(); }
};

namespace channel_internal {

/// @brief Match the Runtime wire envelope used for descriptor-backed generated schemas.
/// @tparam Payload Fixed descriptor-backed schema payload.
template <ChannelDescriptorFlatBufferPayload Payload>
struct TensorSchemaRuntimeEnvelope {
  SampleMetadata metadata{};
  Payload payload{};
};

}  // namespace channel_internal

/** @brief Publisher for schema metadata and local or interprocess Tensor payloads.
 * @tparam Payload Descriptor-backed FlatBuffer payload type.
 * @details The caller must serialize every publication call on one publisher instance, including
 * `emit*`, `close()`, and `report_clock_health()`. Transport release callbacks are synchronized
 * internally. */
template <ChannelDescriptorFlatBufferPayload Payload>
class ChannelTensorSchemaPublisher {
 public:
  ChannelTensorSchemaPublisher(ChannelTensorSchemaPublisher&& other) noexcept = default;
  ChannelTensorSchemaPublisher& operator=(ChannelTensorSchemaPublisher&& other) noexcept = default;
  ~ChannelTensorSchemaPublisher() = default;

  ChannelTensorSchemaPublisher(const ChannelTensorSchemaPublisher&) = delete;
  ChannelTensorSchemaPublisher& operator=(const ChannelTensorSchemaPublisher&) = delete;

  /** @brief Emit schema metadata and retain a Tensor through the configured transport path.
   * @param metadata Serialized FlatBuffer metadata.
   * @param tensor Tensor retained until the private transport releases the submitted sample.
   * @return Recipient count or ErrorCode::kClosed, kCodecError, or a reserve/publish failure. */
  [[nodiscard]] expected<std::uint32_t, Error> emit(
      std::span<const std::byte> metadata,
      Tensor tensor,  // NOLINT(performance-unnecessary-value-param)
      EmitOptions options = {})
    requires(Payload::max_tensors == 1U)
  {  // NOLINT(whitespace/braces)
    const std::array tensors{std::move(tensor)};
    return emit(metadata, std::span<const Tensor>{tensors}, std::move(options));
  }

  /** @brief Emit schema metadata and retain every canonical Tensor field.
   * @param metadata Serialized FlatBuffer metadata.
   * @param tensors Exact generated-field-order Tensor inventory.
   * @return Recipient count or ErrorCode::kClosed, kCodecError, or a reserve/publish failure. */
  [[nodiscard]] expected<std::uint32_t, Error> emit(std::span<const std::byte> metadata,
                                                    std::span<const Tensor> tensors,
                                                    EmitOptions options = {}) {
    SampleMetadata sample_metadata{};
    if (options.capture_time) {
      sample_metadata.capture_timestamp_ns = options.capture_time->timestamp_ns;
      sample_metadata.source_clock_id = options.capture_time->clock_id;
    }
    sample_metadata.frame_id = options.frame_id.value_or(0U);
    sample_metadata.trace_id = options.trace ? options.trace->trace_id : 0U;
    sample_metadata.flags = options.flags;
    return emit_sample(metadata, tensors, sample_metadata);
  }

  /** @brief Emit one Tensor with a capture coordinate in the bound catalog clock.
   * @param metadata Serialized FlatBuffer metadata.
   * @param tensor Tensor retained until the private transport releases the sample.
   * @param capture_timestamp_ns Exact caller-supplied capture coordinate.
   * @param options Remaining optional metadata; an existing capture time is invalid.
   * @return Recipient count or a checked capture, reserve, or publication error. */
  [[nodiscard]] expected<std::uint32_t, Error> emit_captured(
      std::span<const std::byte> metadata,
      Tensor tensor,  // NOLINT(performance-unnecessary-value-param)
      std::int64_t capture_timestamp_ns, EmitOptions options = {})
    requires(Payload::max_tensors == 1U)
  {  // NOLINT(whitespace/braces)
    const std::array tensors{std::move(tensor)};
    return emit_captured(
        metadata, std::span<const Tensor>{tensors}, capture_timestamp_ns, std::move(options));
  }

  /** @brief Emit canonical Tensor fields with a capture coordinate in the bound catalog clock.
   * @param metadata Serialized FlatBuffer metadata.
   * @param tensors Exact generated-field-order Tensor inventory.
   * @param capture_timestamp_ns Exact caller-supplied capture coordinate.
   * @param options Remaining optional metadata; an existing capture time is invalid.
   * @return Recipient count or a checked capture, reserve, or publication error. */
  [[nodiscard]] expected<std::uint32_t, Error> emit_captured(std::span<const std::byte> metadata,
                                                             std::span<const Tensor> tensors,
                                                             std::int64_t capture_timestamp_ns,
                                                             EmitOptions options = {}) {
    if (options.capture_time) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto capture_time = make_capture_time(capture_timestamp_ns);
    if (!capture_time) {
      return make_unexpected(std::move(capture_time).error());
    }
    options.capture_time = *capture_time;
    return emit(metadata, tensors, std::move(options));
  }

  /** @brief Label one coordinate with this catalog publisher's capture clock.
   * @param timestamp_ns Exact caller-supplied nanosecond coordinate.
   * @return Bound time point, or a checked error when closed or not catalog-authorized. */
  [[nodiscard]] expected<TimePoint, Error> make_capture_time(
      std::int64_t timestamp_ns) const noexcept {
    if (!publisher_ || closed_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    return publisher_->make_capture_time(timestamp_ns);
  }

  /// @brief Publish one endpoint-bound clock-health report.
  /// @param report Valid payload-independent assessment and reason.
  /// @return Success after local bounded acceptance, or a checked contract/publication error.
  [[nodiscard]] expected<void, Error> report_clock_health(ClockHealthReport report) {
    if (!publisher_ || closed_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    return publisher_->report_clock_health(report);
  }

  /// @brief Publish an explicit ordered end-of-stream operation and consume this generation.
  /// @return Success after the frozen recipient accepts EOS, or a checked publication error.
  [[nodiscard]] expected<void, Error> close() {
    if (!publisher_ || closed_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    if (!runtime_envelope_) {
      return make_unexpected(Error{ErrorCode::kNotSupported});
    }
    auto proposed_sequence = publisher_->next_operation_sequence();
    if (!proposed_sequence) {
      if (proposed_sequence.error().code == ErrorCode::kClosed) {
        closed_ = true;
      }
      return make_unexpected(std::move(proposed_sequence).error());
    }
    const SampleMetadata metadata{
        .sequence = *proposed_sequence,
        .byte_size = 0U,
        .payload_kind = PayloadKind::kDescriptorBacked,
    };
    auto result = publisher_->close_stream(metadata);
    if (result.phase == channel_internal::InlinePublicationPhase::kAccepted ||
        result.phase == channel_internal::InlinePublicationPhase::kPostSubmitUncertain) {
      closed_ = true;
    }
    if (result.phase != channel_internal::InlinePublicationPhase::kAccepted) {
      return make_unexpected(std::move(result.error));
    }
    return {};
  }

  /** @brief Emit one Tensor with explicit public sample metadata.
   * @param metadata Serialized FlatBuffer metadata.
   * @param tensor Tensor retained until the private transport releases the sample.
   * @param sample_metadata Caller semantic metadata; transport-owned fields are normalized.
   * @return Recipient count or a checked metadata, reserve, or publication error. */
  [[nodiscard]] expected<std::uint32_t, Error> emit_sample(
      std::span<const std::byte> metadata,
      Tensor tensor,  // NOLINT(performance-unnecessary-value-param)
      SampleMetadata sample_metadata)
    requires(Payload::max_tensors == 1U)
  {  // NOLINT(whitespace/braces)
    const std::array tensors{std::move(tensor)};
    return emit_sample(metadata, std::span<const Tensor>{tensors}, sample_metadata);
  }

  /** @brief Emit canonical Tensor fields with explicit public sample metadata.
   * @param metadata Serialized FlatBuffer metadata.
   * @param tensors Exact generated-field-order Tensor inventory.
   * @param sample_metadata Caller semantic metadata; transport-owned fields are normalized.
   * @return Recipient count or a checked metadata, reserve, or publication error. */
  [[nodiscard]] expected<std::uint32_t, Error> emit_sample(std::span<const std::byte> metadata,
                                                           std::span<const Tensor> tensors,
                                                           SampleMetadata sample_metadata) {
    if (!publisher_ || closed_) {
      return make_unexpected(channel_internal::make_channel_error(
          ErrorCode::kClosed, "Tensor schema channel publisher is closed"));
    }

    if (tensors.size() != Payload::max_tensors) {
      return make_unexpected(channel_internal::make_channel_error(
          ErrorCode::kInvalidArgument,
          "Tensor schema publication does not match the generated field count"));
    }

    auto aggregate_byte_size = std::uint64_t{};
    for (const auto& tensor : tensors) {
      const std::int64_t byte_size = tensor.nbytes();
      if (byte_size < 0 || static_cast<std::uint64_t>(byte_size) >
                               std::numeric_limits<std::uint64_t>::max() - aggregate_byte_size) {
        return make_unexpected(channel_internal::make_channel_error(
            ErrorCode::kInvalidArgument, "Tensor schema payload byte size is invalid"));
      }
      aggregate_byte_size += static_cast<std::uint64_t>(byte_size);
    }

    auto verified = Payload::verify_metadata(metadata);
    if (!verified) {
      return make_unexpected(std::move(verified).error());
    }
    auto metadata_valid = publisher_->validate_sample_metadata(sample_metadata);
    if (!metadata_valid) {
      return make_unexpected(std::move(metadata_valid).error());
    }

    for (std::size_t index = 0U; index < tensors.size(); ++index) {
      auto decoded = Payload::tensor_view_metadata_at_from_verified(metadata, index);
      if (!decoded || !tensor_matches(tensors[index], *decoded)) {
        return make_unexpected(channel_internal::make_channel_error(
            ErrorCode::kInvalidArgument,
            "Tensor schema metadata does not match the submitted Tensor"));
      }
    }

    auto reservation = publisher_->reserve_tensors(tensors);
    if (!reservation) {
      return make_unexpected(std::move(reservation).error());
    }
    auto reservation_value = std::move(reservation).value();

    std::array<PayloadDescriptor, Payload::max_tensors> descriptors{};
    for (std::size_t index = 0U; index < descriptors.size(); ++index) {
      auto* field = reservation_value.field_at(index);
      if (field == nullptr) {
        publisher_->release_tensor(std::move(reservation_value));
        return make_unexpected(channel_internal::make_channel_error(
            ErrorCode::kProtocolError, "Tensor schema reservation is missing a field"));
      }
      field->descriptor.schema_id = Payload::expected_schema_family_id;
      descriptors[index] = field->descriptor;
    }
    auto payload = Payload::copy_from_trusted_metadata(
        metadata, descriptors, channel_internal::kCompatibilitySchemaEndpointGeneration);
    if (!payload) {
      publisher_->release_tensor(std::move(reservation_value));
      return make_unexpected(std::move(payload).error());
    }

    return publish_prepared_payload(
        std::move(*payload), std::move(reservation_value), aggregate_byte_size, sample_metadata);
  }

  /** @brief Emit an already constructed descriptor-backed payload.
   * @param payload Validated schema payload containing an existing payload descriptor.
   * @return Recipient count or ErrorCode::kClosed, kCodecError, or a publish failure.
   * @note This low-level path supports runtime-owned materialization flows that already minted a
   * descriptor, such as an interprocess opaque external-buffer descriptor. Use emit(metadata,
   * tensor) for an ordinary Tensor so the runtime manages its same-process sidecar record.
   * @warning An opaque external-buffer payload does not transfer an owning Tensor lease through
   * this overload. Runtime code using this low-level path must retain the referenced storage until
   * the private transport releases its submitted sample. Prefer emit(metadata, tensor) when a
   * Tensor is available. */
  [[nodiscard]] expected<std::uint32_t, Error> emit_payload(const Payload& payload,
                                                            SampleMetadata sample_metadata = {})
    requires(Payload::max_tensors == 1U)
  {  // NOLINT(whitespace/braces)
    if (!publisher_ || closed_) {
      return make_unexpected(channel_internal::make_channel_error(
          ErrorCode::kClosed, "Tensor schema channel publisher is closed"));
    }

    Payload payload_value = payload;
    payload_value.endpoint_generation = channel_internal::kCompatibilitySchemaEndpointGeneration;
    auto valid = payload_value.validate();
    if (!valid) {
      return make_unexpected(std::move(valid).error());
    }
    auto metadata_valid = publisher_->validate_sample_metadata(sample_metadata);
    if (!metadata_valid) {
      return make_unexpected(std::move(metadata_valid).error());
    }
    for (std::size_t index = 0U; index < Payload::max_tensors; ++index) {
      const PayloadDescriptor* descriptor = payload_value.descriptor_at(index);
      auto decoded = Payload::tensor_view_metadata_at_from_verified(payload_value, index);
      if (descriptor == nullptr || !decoded) {
        return make_unexpected(descriptor == nullptr ? channel_internal::make_channel_error(
                                                           ErrorCode::kProtocolError,
                                                           "Tensor schema descriptor is missing")
                                                     : std::move(decoded).error());
      }
      auto contract_valid = publisher_->validate_field_descriptor(
          static_cast<std::uint32_t>(index), *decoded, *descriptor);
      if (!contract_valid) {
        return make_unexpected(std::move(contract_valid).error());
      }
    }
    channel_internal::TensorSchemaPayloadReservation reservation{
        .field_count = static_cast<std::uint32_t>(Payload::max_tensors),
    };
    for (std::size_t index = 0U; index < Payload::max_tensors; ++index) {
      const PayloadDescriptor* descriptor = payload_value.descriptor_at(index);
      auto* field = reservation.field_at(index);
      if (descriptor == nullptr || field == nullptr) {
        return make_unexpected(channel_internal::make_channel_error(
            ErrorCode::kProtocolError, "Tensor schema descriptor is missing"));
      }
      field->descriptor = *descriptor;
    }
    std::uint64_t aggregate_byte_size = 0U;
    for (std::size_t index = 0U; index < Payload::max_tensors; ++index) {
      const PayloadDescriptor* descriptor = payload_value.descriptor_at(index);
      if (descriptor == nullptr ||
          descriptor->byte_size > std::numeric_limits<std::uint64_t>::max() - aggregate_byte_size) {
        return make_unexpected(channel_internal::make_channel_error(
            ErrorCode::kInvalidArgument, "Tensor schema descriptor byte size is invalid"));
      }
      aggregate_byte_size += descriptor->byte_size;
    }
    return publish_prepared_payload(
        std::move(payload_value), std::move(reservation), aggregate_byte_size, sample_metadata);
  }

  /** @brief Refresh backend connection state.
   * @return Empty success or Error when closed or when the backend refresh fails. */
  [[nodiscard]] expected<void, Error> update_connections() noexcept {
    if (!publisher_ || closed_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    return publisher_->update_connections();
  }

  /** @brief Return whether at least one subscriber is connected.
   * @return True when connected; false otherwise. */
  [[nodiscard]] bool has_subscribers() const noexcept {
    return !closed_ && publisher_ && publisher_->has_subscribers();
  }

  /** @brief Return graph-derived Tensor allocation options.
   * @return MemoryBuffer options or Error when closed or unavailable. */
  [[nodiscard]] expected<MemoryBuffer::Options, Error> tensor_memory_options() const {
    if (!publisher_ || closed_) {
      return make_unexpected(channel_internal::make_channel_error(
          ErrorCode::kClosed, "Tensor schema channel publisher is closed"));
    }
    return publisher_->tensor_memory_options();
  }

 private:
  friend class ChannelRuntime;
  friend class detail::RuntimeChannelBindingAccess;

  explicit ChannelTensorSchemaPublisher(
      std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher> publisher,
      const bool runtime_envelope = false) noexcept
      : publisher_(std::move(publisher)), runtime_envelope_(runtime_envelope) {}

  [[nodiscard]] expected<std::uint32_t, Error> publish_prepared_payload(
      Payload payload,
      channel_internal::TensorSchemaPayloadReservation
          reservation,  // NOLINT(performance-unnecessary-value-param)
      const std::uint64_t aggregate_byte_size, SampleMetadata metadata) {
    if (!runtime_envelope_) {
      const auto bytes = std::as_bytes(std::span<const Payload, 1U>(&payload, 1U));
      // A manual Tensor schema channel publishes the payload aggregate itself as one inline typed
      // frame, so declare that frame exactly like ChannelPublisher<T>::emit. A standalone client
      // subscribing with the plain typed facade needs the declared size to reject a short
      // same-bucket frame before bit-casting its endpoint scratch. Semantic metadata stays absent
      // on this path because the manual subscriber resolves descriptors, not transport metadata.
      SampleMetadata frame_metadata{};
      frame_metadata.byte_size = bytes.size();
      frame_metadata.payload_kind = PayloadKind::kInlineTyped;
      return publisher_->publish_payload(bytes, std::move(reservation), frame_metadata);
    }
    auto proposed_sequence = publisher_->next_operation_sequence();
    if (!proposed_sequence) {
      publisher_->release_tensor(std::move(reservation));
      return make_unexpected(std::move(proposed_sequence).error());
    }
    // Publication time is framework-owned, and every route here is a caller publication: Runtime's
    // descriptor-schema export bypasses this facade and calls publish_encoded_payload with the
    // coordinate it stamped. A caller pair forwarded from here would name a clock the caller has no
    // authority over, and a graph import rejects any populated coordinate outright.
    metadata.publish_timestamp_ns = 0;
    metadata.publish_clock_id = {};
    metadata.sequence = *proposed_sequence;
    metadata.byte_size = aggregate_byte_size;
    metadata.payload_kind = PayloadKind::kDescriptorBacked;
    using Envelope = channel_internal::TensorSchemaRuntimeEnvelope<Payload>;
    Envelope envelope{
        .metadata = metadata,
        .payload = std::move(payload),
    };
    const auto bytes = std::as_bytes(std::span<const Envelope, 1U>(&envelope, 1U));
    return publisher_->publish_payload(bytes, std::move(reservation), metadata);
  }

  std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher> publisher_;
  bool runtime_envelope_{};
  bool closed_{};
};

/** @brief Subscriber for schema metadata plus resolved local or interprocess Tensor data.
 * @tparam Payload Descriptor-backed FlatBuffer payload type. */
template <ChannelDescriptorFlatBufferPayload Payload>
class ChannelTensorSchemaSubscriber {
 public:
  ChannelTensorSchemaSubscriber(ChannelTensorSchemaSubscriber&& other) noexcept = default;
  ChannelTensorSchemaSubscriber& operator=(ChannelTensorSchemaSubscriber&& other) noexcept {
    if (this != &other) {
      static_cast<void>(channel_internal::close_local_subscriber(subscriber_));
      subscriber_ = std::move(other.subscriber_);
      last_sequence_ = std::exchange(other.last_sequence_, 0U);
      runtime_envelope_ = std::exchange(other.runtime_envelope_, false);
    }
    return *this;
  }
  ~ChannelTensorSchemaSubscriber() {
    static_cast<void>(channel_internal::close_local_subscriber(subscriber_));
  }

  ChannelTensorSchemaSubscriber(const ChannelTensorSchemaSubscriber&) = delete;
  ChannelTensorSchemaSubscriber& operator=(const ChannelTensorSchemaSubscriber&) = delete;

  /** @brief Retire this process's local descriptor-Tensor subscriber endpoint.
   * @return Success on the first live close, or a code-only cleanup error. Repeated close and
   * close on a moved-from facade return @ref ErrorCode::kClosed.
   * @note This externally serialized, consuming operation publishes no producer end-of-stream
   * record and supplies no graph-completion evidence. Cancel and join any active waiter before
   * closing. Already returned Tensor bundles retain their ordinary owning lifetime; unread local
   * data may be discarded. A cleanup failure still consumes the endpoint; later checked
   * operations return @ref ErrorCode::kClosed. */
  [[nodiscard]] expected<void, Error> close() noexcept {
    return channel_internal::close_local_subscriber(subscriber_);
  }

  /** @brief Query pending metadata descriptors without consuming one.
   * @return Readiness or Error when closed or the backend query fails. */
  [[nodiscard]] expected<bool, Error> has_samples() const {
    if (!subscriber_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    return subscriber_->has_samples();
  }

  /** @brief Return whether at least one publisher is connected.
   * @return True when connected; false when closed or no publisher has been discovered. */
  [[nodiscard]] bool has_publishers() const noexcept {
    return subscriber_ && subscriber_->has_publishers();
  }

  /** @brief Receive metadata and materialize its referenced Tensor.
   * @return Resolved sample or ErrorCode::kClosed, kCodecError, or a receive, sidecar-resolution,
   * buffer-import, or Tensor-materialization failure. */
  [[nodiscard]] expected<ChannelResolvedTensorSample<Payload>, Error> receive()
    requires(Payload::max_tensors == 1U)
  {  // NOLINT(whitespace/braces)
    auto bundle = receive_tensors();
    if (!bundle) {
      return make_unexpected(std::move(bundle).error());
    }
    return ChannelResolvedTensorSample<Payload>{.metadata = std::move(bundle->metadata),
                                                .sample_metadata = bundle->sample_metadata,
                                                .tensor = std::move(bundle->tensors[0]),
                                                .payload = bundle->payloads[0]};
  }

  /** @brief Receive metadata and materialize every canonical Tensor field.
   * @return Resolved bundle or ErrorCode::kClosed, kCodecError, or a receive/materialization
   * failure. */
  // Receive validates the envelope and every field before publishing an owning bundle. Keep the
  // protocol checks adjacent so a partially verified field inventory cannot escape.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] expected<ChannelResolvedTensorBundle<Payload>, Error> receive_tensors() {
    if (!subscriber_) {
      return make_unexpected(channel_internal::make_channel_error(
          ErrorCode::kClosed, "Tensor schema channel subscriber is closed"));
    }

    using Envelope = channel_internal::TensorSchemaRuntimeEnvelope<Payload>;
    Payload payload{};
    Envelope envelope{};
    std::span<std::byte> bytes;
    if (runtime_envelope_) {
      bytes = std::as_writable_bytes(std::span<Envelope, 1U>(&envelope, 1U));
    } else {
      bytes = std::as_writable_bytes(std::span<Payload, 1U>(&payload, 1U));
    }
    auto received = subscriber_->receive_payload(bytes);
    if (!received) {
      return make_unexpected(std::move(received).error());
    }
    if (runtime_envelope_) {
      payload = std::move(envelope.payload);
    }

    const auto release_local_claims_from = [&](const std::size_t first) noexcept {
      for (std::size_t index = first; index < Payload::max_tensors; ++index) {
        const PayloadDescriptor* descriptor = payload.descriptor_at(index);
        if (descriptor != nullptr) {
          release_local_descriptor_claim(*descriptor);
        }
      }
    };
    if (runtime_envelope_) {
      if (envelope.metadata.version.major != 1U || envelope.metadata.version.minor != 0U ||
          envelope.metadata.sequence == 0U || envelope.metadata.sequence <= last_sequence_ ||
          envelope.metadata.payload_kind != PayloadKind::kDescriptorBacked) {
        release_local_claims_from(0U);
        return make_unexpected(
            Error{ErrorCode::kProtocolError, "Tensor schema Runtime envelope metadata is invalid"});
      }
    }
    auto metadata_status = validate_metadata(payload);
    if (!metadata_status) {
      release_local_claims_from(0U);
      return make_unexpected(std::move(metadata_status).error());
    }
    if (runtime_envelope_) {
      std::uint64_t aggregate_byte_size = 0U;
      for (std::size_t index = 0U; index < Payload::max_tensors; ++index) {
        const PayloadDescriptor* descriptor = payload.descriptor_at(index);
        if (descriptor == nullptr ||
            descriptor->byte_size >
                std::numeric_limits<std::uint64_t>::max() - aggregate_byte_size) {
          release_local_claims_from(0U);
          return make_unexpected(
              Error{ErrorCode::kProtocolError, "Tensor schema descriptor byte size is invalid"});
        }
        aggregate_byte_size += descriptor->byte_size;
      }
      if (envelope.metadata.byte_size != aggregate_byte_size) {
        release_local_claims_from(0U);
        return make_unexpected(
            Error{ErrorCode::kProtocolError, "Tensor schema Runtime payload byte size is invalid"});
      }
    }
    // Decode every representation before typed exposure so same-process descriptors also reject
    // malformed metadata and embedded Tensor bytes. Buffer-handle import additionally consumes the
    // decoded view below.
    std::array<TensorViewMetadata, Payload::max_tensors> tensor_metadata{};
    for (std::size_t index = 0U; index < Payload::max_tensors; ++index) {
      if (payload.descriptor_at(index) == nullptr) {
        release_local_claims_from(0U);
        return make_unexpected(
            Error{ErrorCode::kProtocolError, "Tensor schema descriptor is missing"});
      }
      auto decoded = Payload::tensor_view_metadata_at_from_verified(payload, index);
      if (!decoded) {
        release_local_claims_from(0U);
        return make_unexpected(std::move(decoded).error());
      }
      auto contract_valid = subscriber_->validate_field_descriptor(
          static_cast<std::uint32_t>(index), *decoded, *payload.descriptor_at(index));
      if (!contract_valid) {
        release_local_claims_from(0U);
        return make_unexpected(std::move(contract_valid).error());
      }
      tensor_metadata[index] = *decoded;
    }

    ChannelResolvedTensorBundle<Payload> result{
        .metadata =
            ChannelTensorSchemaMetadata<Payload>{
                .schema_family_id = payload.schema_family_id,
                .transport_profile_tag = payload.transport_profile_tag,
                .metadata_size = payload.metadata_size,
                .endpoint_generation = payload.endpoint_generation,
                .metadata = payload.metadata},
        .sample_metadata = runtime_envelope_ ? envelope.metadata : SampleMetadata{},
    };
    for (std::size_t index = 0U; index < Payload::max_tensors; ++index) {
      const PayloadDescriptor& descriptor = *payload.descriptor_at(index);
      expected<Tensor, Error> tensor =
          descriptor.payload_kind == ChannelPayloadKind::kBufferHandle
              ? subscriber_->import_buffer_handle_tensor(
                    descriptor, tensor_metadata[index], static_cast<std::uint32_t>(index))
              : subscriber_->acquire_tensor(descriptor);
      if (!tensor) {
        release_local_claims_from(index);
        return make_unexpected(std::move(tensor).error());
      }
      result.tensors[index] = std::move(*tensor);
      result.payloads[index] = descriptor;
    }
    if (runtime_envelope_) {
      last_sequence_ = envelope.metadata.sequence;
    }
    return result;
  }

  /// @brief Block until a sample is available, then receive it and materialize its Tensor.
  ///
  /// A receive is attempted before any wait is built, so an already queued sample costs no wait
  /// context and no kernel wait. \ref ErrorCode::kNotReady never escapes: readiness is a hint, so
  /// an attempt that loses its race waits again against the same absolute \p deadline.
  ///
  /// @param deadline Steady-clock coordinate after which the wait reports a timeout.
  /// @return The resolved sample on success, or the first checked error.
  /// @retval ErrorCode::kTimeout \p deadline elapsed. Retryable; endpoint state is unchanged.
  /// @retval ErrorCode::kClosed The producer published end of stream through
  /// \ref ChannelTensorSchemaPublisher::close, or this facade was locally closed or moved from.
  /// Each means no later sample arrives through this facade. Only a catalog-attached or
  /// runtime-envelope publisher can publish end of stream; \ref ChannelTensorSchemaPublisher::close
  /// on a name-based publisher reports \ref ErrorCode::kNotSupported, so a name-based stream never
  /// reaches the remote end-of-stream outcome.
  /// @retval ErrorCode::kStaleEndpoint The producer was replaced by a new incarnation; rebuild the
  /// subscriber. Reported only by a catalog-attached subscriber, which is the only form that
  /// carries a producer generation on the wire. A name-based subscriber cannot tell a replaced
  /// producer from an idle one and reports \ref ErrorCode::kTimeout instead.
  /// @retval ErrorCode::kNotSupported This endpoint was created with
  /// \c ChannelOptions::enable_event_wake disabled, so it has no kernel event lane. Reported only
  /// when a wait is actually required, and before any wait context is built. This operation
  /// deliberately does not fall back to polling.
  /// @retval ErrorCode::kInvalidArgument This subscriber's readiness is already owned by another
  /// wait set, such as the Runtime readiness pump, or another wait on this subscriber is already
  /// in flight.
  /// @retval ErrorCode::kFailure This endpoint still holds an active transport payload loan.
  /// Reported by the first receive attempt, before any wait.
  /// @note \ref ErrorCode::kProtocolError, \ref ErrorCode::kCodecError,
  /// \ref ErrorCode::kReceiveError, and \ref ErrorCode::kResourceExhausted pass through unchanged.
  /// @note There is no duration overload by design. An absolute deadline cannot silently restart
  /// its budget on a spurious wake.
  /// @note Readiness is a hint, so a subscriber that keeps reporting it while every receive loses
  /// its race retries at full speed until \p deadline. The absolute deadline is the only bound on
  /// that loop; there is no backoff. It is not reachable by an idle endpoint -- something has to be
  /// producing readiness that no receive can consume, such as a competing consumer on the same
  /// endpoint -- but a caller that shares a subscriber should size the deadline accordingly.
  /// @warning One subscriber supports one waiting thread and calls are not internally serialized.
  /// A second concurrent waiter is rejected with \ref ErrorCode::kInvalidArgument rather than
  /// sharing the first one's wait context. The facade must outlive an in-flight wait and must not
  /// be moved while one is in flight.
  [[nodiscard]] expected<ChannelResolvedTensorSample<Payload>, Error> wait_and_receive_sample(
      const std::chrono::steady_clock::time_point deadline) {
    return wait_and_receive_sample(deadline, std::stop_token{});
  }

  /// @brief Block until a sample is available or a stop is requested, then receive it.
  /// @param deadline Steady-clock coordinate after which the wait reports a timeout.
  /// @param token Cancellation observer for this call; a default-constructed token never cancels.
  /// @return The resolved sample on success, or the first checked error.
  /// @retval ErrorCode::kInterrupted \p token requested a stop. Caller-local and non-terminal.
  /// @note Every other retval of the single-argument overload applies unchanged.
  /// @note Cancellation is scoped to one call. A stop that fires ends this wait and leaves the
  /// subscriber usable; the next wait arms a fresh cancellation trigger.
  [[nodiscard]] expected<ChannelResolvedTensorSample<Payload>, Error> wait_and_receive_sample(
      const std::chrono::steady_clock::time_point deadline, const std::stop_token& token) {
    if (!subscriber_) {
      return make_unexpected(channel_internal::make_channel_error(
          ErrorCode::kClosed, "Tensor schema channel subscriber is closed"));
    }
    return channel_internal::wait_and_receive_next(
        *subscriber_, deadline, channel_internal::borrowed_stop_token(token), [this] {
          return receive();
        });
  }

  /// @brief Block until this endpoint may have a sample, without consuming one.
  /// @param deadline Steady-clock coordinate after which the wait reports a timeout.
  /// @param token Cancellation observer for this call; a default-constructed token never cancels.
  /// @return Empty success when a sample may be available, or a checked wait error.
  /// @retval ErrorCode::kTimeout \p deadline elapsed with no readiness.
  /// @retval ErrorCode::kInterrupted \p token requested a stop.
  /// @retval ErrorCode::kClosed This facade was locally closed or moved from.
  /// @retval ErrorCode::kNotSupported This endpoint was created with
  /// \c ChannelOptions::enable_event_wake disabled and has no queued sample.
  /// @retval ErrorCode::kInvalidArgument This subscriber's readiness is already owned by another
  /// wait set, or another wait on this subscriber is already in flight.
  /// @note Success is a hint, exactly like \ref has_samples. A following \ref receive may still
  /// report \ref ErrorCode::kNotReady.
  [[nodiscard]] expected<void, Error> wait_for_sample(
      const std::chrono::steady_clock::time_point deadline, const std::stop_token& token = {}) {
    if (!subscriber_) {
      return make_unexpected(channel_internal::make_channel_error(
          ErrorCode::kClosed, "Tensor schema channel subscriber is closed"));
    }
    return channel_internal::wait_for_sample_readiness(
        *subscriber_, deadline, channel_internal::borrowed_stop_token(token));
  }

  /// @brief Claim an externally pollable readiness descriptor for this subscriber.
  /// @return A move-only handle, or a checked closed, capability, or ownership error.
  /// @retval ErrorCode::kClosed This facade is locally closed or moved from.
  /// @retval ErrorCode::kNotSupported Native descriptor readiness is unavailable on this platform,
  /// or this endpoint was created without event wake.
  /// @retval ErrorCode::kInvalidArgument A wait is in flight, the Runtime readiness pump owns the
  /// subscriber, or another readiness handle is live.
  /// @note Readability is a hint. Call \ref ChannelReadinessHandle::acknowledge first, then drain
  /// \ref receive until \ref ErrorCode::kNotReady. A wake arriving during that drain stays pending.
  /// Producer end of stream and producer-incarnation changes are also readable terminal/control
  /// events when the endpoint contract carries those operations.
  /// @note Blocking wait operations return \ref ErrorCode::kInvalidArgument while the handle is
  /// live.
  [[nodiscard]] expected<ChannelReadinessHandle, Error> readiness() {
    if (!subscriber_) {
      return make_unexpected(channel_internal::make_channel_error(
          ErrorCode::kClosed, "Tensor schema channel subscriber is closed"));
    }
    return subscriber_->readiness();
  }

 private:
  friend class ChannelRuntime;
  friend class detail::RuntimeChannelBindingAccess;

  explicit ChannelTensorSchemaSubscriber(
      std::unique_ptr<channel_internal::TensorSchemaPayloadSubscriber> subscriber,
      const bool runtime_envelope = false) noexcept
      : subscriber_(std::move(subscriber)), runtime_envelope_(runtime_envelope) {}

  /** @brief Validate schema identity and serialized metadata.
   * @param payload Received descriptor-backed payload.
   * @return Empty success or ErrorCode::kCodecError when identity, size, or FlatBuffer validation
   * fails. */
  [[nodiscard]] static expected<void, Error> validate_metadata(const Payload& payload) {
    auto valid = payload.validate();
    if (!valid) {
      return make_unexpected(std::move(valid).error());
    }
    if (payload.endpoint_generation != channel_internal::kCompatibilitySchemaEndpointGeneration) {
      return make_unexpected(
          Error{ErrorCode::kProtocolError, "Tensor schema endpoint generation is invalid"});
    }
    return {};
  }

  void release_local_descriptor_claim(const PayloadDescriptor& payload_descriptor) noexcept {
    if (payload_descriptor.payload_kind != ChannelPayloadKind::kLocalBufferDescriptor) {
      return;
    }
    subscriber_->release_recipient_claim(payload_descriptor);
  }

  std::unique_ptr<channel_internal::TensorSchemaPayloadSubscriber> subscriber_;
  std::uint64_t last_sequence_{};
  bool runtime_envelope_{};
};

}  // namespace holoscan
