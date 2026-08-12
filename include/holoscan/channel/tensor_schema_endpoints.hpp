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
#include <holoscan/memory/tensor_storage.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace holoscan {
/** @brief Validated FlatBuffer metadata for a resolved Tensor sample.
 * @tparam Payload Descriptor-backed FlatBuffer payload type.
 * @details Transport-only descriptor fields remain internal; use
 * ChannelResolvedTensorSample::payload for the public runtime payload reference. */
template <ChannelDescriptorFlatBufferPayload Payload>
struct ChannelTensorSchemaMetadata {
  using root_type = typename Payload::root_type;
  static constexpr std::size_t kMetadataCapacity = Payload::metadata_capacity;

  /// Complete compact schema identity verified on the received wrapper.
  SchemaId schema_id{};
  /// Exact semantic schema version verified on the received wrapper.
  std::uint32_t schema_version{};
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
/** @brief Publisher for schema metadata and local or interprocess Tensor payloads.
 * @tparam Payload Descriptor-backed FlatBuffer payload type. */
template <ChannelDescriptorFlatBufferPayload Payload>
class ChannelTensorSchemaPublisher {
 public:
  static_assert(Payload::max_tensors == 1U,
                "the direct Channel Tensor schema publisher currently admits one Tensor");
  ChannelTensorSchemaPublisher(ChannelTensorSchemaPublisher&& other) noexcept = default;
  ChannelTensorSchemaPublisher& operator=(ChannelTensorSchemaPublisher&& other) noexcept = default;
  ~ChannelTensorSchemaPublisher() = default;

  ChannelTensorSchemaPublisher(const ChannelTensorSchemaPublisher&) = delete;
  ChannelTensorSchemaPublisher& operator=(const ChannelTensorSchemaPublisher&) = delete;

  /** @brief Emit schema metadata and retain a Tensor through the configured transport path.
   * @param metadata Serialized FlatBuffer metadata.
   * @param tensor Tensor retained until the private transport releases the submitted sample.
   * @return Recipient count or ErrorCode::kClosed, kCodecError, or a reserve/publish failure. */
  [[nodiscard]] expected<std::uint32_t, Error> emit(std::span<const std::byte> metadata,
                                                    Tensor tensor) {
    if (!publisher_) {
      return make_unexpected(channel_internal::make_channel_error(
          ErrorCode::kClosed, "Tensor schema channel publisher is closed"));
    }

    auto verified = Payload::verify_metadata(metadata);
    if (!verified) {
      return make_unexpected(std::move(verified).error());
    }

    const std::array tensors{std::move(tensor)};
    auto reservation = publisher_->reserve_tensors(tensors);
    if (!reservation) {
      return make_unexpected(std::move(reservation).error());
    }
    auto reservation_value = std::move(reservation).value();

    auto* field = reservation_value.field_at(0U);
    if (field == nullptr) {
      publisher_->release_tensor(std::move(reservation_value));
      return make_unexpected(channel_internal::make_channel_error(
          ErrorCode::kProtocolError, "Tensor schema reservation is missing field zero"));
    }
    field->descriptor.schema_id = Payload::expected_schema_id;
    auto payload = Payload::copy_from_trusted_metadata(
        metadata, field->descriptor, channel_internal::kCompatibilitySchemaEndpointGeneration);
    if (!payload) {
      publisher_->release_tensor(std::move(reservation_value));
      return make_unexpected(std::move(payload).error());
    }

    Payload payload_value = std::move(payload).value();
    const auto bytes = std::as_bytes(std::span<const Payload, 1U>(&payload_value, 1U));
    return publisher_->publish_payload(bytes, std::move(reservation_value));
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
  [[nodiscard]] expected<std::uint32_t, Error> emit_payload(const Payload& payload) {
    if (!publisher_) {
      return make_unexpected(channel_internal::make_channel_error(
          ErrorCode::kClosed, "Tensor schema channel publisher is closed"));
    }

    Payload payload_value = payload;
    payload_value.endpoint_generation = channel_internal::kCompatibilitySchemaEndpointGeneration;
    auto valid = payload_value.validate();
    if (!valid) {
      return make_unexpected(std::move(valid).error());
    }
    const auto bytes = std::as_bytes(std::span<const Payload, 1U>(&payload_value, 1U));
    const PayloadDescriptor* descriptor = payload_value.descriptor_at(0U);
    if (descriptor == nullptr) {
      return make_unexpected(channel_internal::make_channel_error(
          ErrorCode::kProtocolError, "Tensor schema descriptor is missing"));
    }
    channel_internal::TensorSchemaPayloadReservation reservation{.field_count = 1U};
    reservation.first_field.descriptor = *descriptor;
    return publisher_->publish_payload(bytes, std::move(reservation));
  }

  /** @brief Refresh backend connection state.
   * @return Empty success or Error when closed or when the backend refresh fails. */
  [[nodiscard]] expected<void, Error> update_connections() noexcept {
    if (!publisher_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    return publisher_->update_connections();
  }

  /** @brief Return whether at least one subscriber is connected.
   * @return True when connected; false otherwise. */
  [[nodiscard]] bool has_subscribers() const noexcept {
    return publisher_ && publisher_->has_subscribers();
  }

  /** @brief Return graph-derived Tensor allocation options.
   * @return MemoryBuffer options or Error when closed or unavailable. */
  [[nodiscard]] expected<MemoryBuffer::Options, Error> tensor_memory_options() const {
    if (!publisher_) {
      return make_unexpected(channel_internal::make_channel_error(
          ErrorCode::kClosed, "Tensor schema channel publisher is closed"));
    }
    return publisher_->tensor_memory_options();
  }

 private:
  friend class ChannelRuntime;

  explicit ChannelTensorSchemaPublisher(
      std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher> publisher) noexcept
      : publisher_(std::move(publisher)) {}

  std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher> publisher_;
};

/** @brief Subscriber for schema metadata plus resolved local or interprocess Tensor data.
 * @tparam Payload Descriptor-backed FlatBuffer payload type. */
template <ChannelDescriptorFlatBufferPayload Payload>
class ChannelTensorSchemaSubscriber {
 public:
  static_assert(Payload::max_tensors == 1U,
                "the direct Channel Tensor schema subscriber currently admits one Tensor");
  ChannelTensorSchemaSubscriber(ChannelTensorSchemaSubscriber&& other) noexcept = default;
  ChannelTensorSchemaSubscriber& operator=(ChannelTensorSchemaSubscriber&& other) noexcept =
      default;
  ~ChannelTensorSchemaSubscriber() = default;

  ChannelTensorSchemaSubscriber(const ChannelTensorSchemaSubscriber&) = delete;
  ChannelTensorSchemaSubscriber& operator=(const ChannelTensorSchemaSubscriber&) = delete;

  /** @brief Query pending metadata descriptors without consuming one.
   * @return Readiness or Error when closed or the backend query fails. */
  [[nodiscard]] expected<bool, Error> has_samples() const {
    if (!subscriber_) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    return subscriber_->has_samples();
  }

  /** @brief Receive metadata and materialize its referenced Tensor.
   * @return Resolved sample or ErrorCode::kClosed, kCodecError, or a receive, sidecar-resolution,
   * buffer-import, or Tensor-materialization failure. */
  [[nodiscard]] expected<ChannelResolvedTensorSample<Payload>, Error> receive() {
    if (!subscriber_) {
      return make_unexpected(channel_internal::make_channel_error(
          ErrorCode::kClosed, "Tensor schema channel subscriber is closed"));
    }

    Payload payload{};
    auto bytes = std::as_writable_bytes(std::span<Payload, 1U>(&payload, 1U));
    auto received = subscriber_->receive_payload(bytes);
    if (!received) {
      return make_unexpected(std::move(received).error());
    }

    auto metadata_status = validate_metadata(payload);
    const PayloadDescriptor* descriptor = payload.descriptor_at(0U);
    if (!metadata_status) {
      if (descriptor != nullptr) {
        release_local_descriptor_claim(*descriptor);
      }
      return make_unexpected(std::move(metadata_status).error());
    }
    if (descriptor == nullptr) {
      return make_unexpected(
          Error{ErrorCode::kProtocolError, "Tensor schema descriptor is missing"});
    }
    // Decode every representation before typed exposure so same-process descriptors also reject
    // malformed metadata and embedded Tensor bytes. Buffer-handle import additionally consumes the
    // decoded view below.
    auto tensor_metadata = Payload::tensor_view_metadata_from_verified(payload);
    if (!tensor_metadata) {
      release_local_descriptor_claim(*descriptor);
      return make_unexpected(std::move(tensor_metadata).error());
    }
    ChannelTensorSchemaMetadata<Payload> metadata{
        .schema_id = payload.schema_id,
        .schema_version = payload.schema_version,
        .metadata_size = payload.metadata_size,
        .endpoint_generation = payload.endpoint_generation,
        .metadata = payload.metadata};
    if (descriptor->payload_kind == ChannelPayloadKind::kBufferHandle) {
      // Buffer-handle payloads have no local sidecar recipient claim. The receive guard releases
      // the active transport loan when this function returns, including on import failure.
      auto tensor =
          subscriber_->import_buffer_handle_tensor(*descriptor, tensor_metadata.value(), 0U);
      if (!tensor) {
        return make_unexpected(std::move(tensor).error());
      }
      return ChannelResolvedTensorSample<Payload>{.metadata = std::move(metadata),
                                                  .tensor = std::move(tensor).value(),
                                                  .payload = *descriptor};
    }

    auto tensor = subscriber_->acquire_tensor(*descriptor);
    if (!tensor) {
      return make_unexpected(std::move(tensor).error());
    }
    return ChannelResolvedTensorSample<Payload>{.metadata = std::move(metadata),
                                                .tensor = std::move(tensor).value(),
                                                .payload = *descriptor};
  }

 private:
  friend class ChannelRuntime;

  explicit ChannelTensorSchemaSubscriber(
      std::unique_ptr<channel_internal::TensorSchemaPayloadSubscriber> subscriber) noexcept
      : subscriber_(std::move(subscriber)) {}

  /** @brief Validate schema identity and serialized metadata.
   * @param payload Received descriptor-backed payload.
   * @return Empty success or ErrorCode::kCodecError when identity, size, or FlatBuffer validation
   * fails. */
  [[nodiscard]] static expected<void, Error> validate_metadata(const Payload& payload) {
    auto valid = payload.validate();
    if (!valid ||
        payload.endpoint_generation != channel_internal::kCompatibilitySchemaEndpointGeneration) {
      return make_unexpected(Error{ErrorCode::kProtocolError, "Tensor schema envelope is invalid"});
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
};

}  // namespace holoscan
