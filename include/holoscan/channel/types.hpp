/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/schema_identity.hpp>
#include <holoscan/memory/memory_storage_type.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/schema/flatbuffer_schema_traits.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace holoscan::channel_internal {

struct OpaquePayloadReferenceAccess;

}  // namespace holoscan::channel_internal

namespace holoscan {

/// @brief Supply caller-local bounds for one catalog endpoint attachment.
///
/// The timeout may only shorten the positive finite bound admitted by the catalog. It never
/// changes peer requirement, payload identity, service capacity, queue policy, or capability.
struct ChannelAttachOptions {
  /// Optional positive caller-local maximum for create-or-verify and required-peer waiting.
  ///
  /// Absence uses the catalog maximum. A present duration must not exceed that maximum; zero,
  /// negative, or longer values are rejected without creating a partial endpoint.
  std::optional<std::chrono::nanoseconds> timeout;
};

namespace schema {

template <OrdinaryFlatBufferSchemaRoot Root, std::size_t Capacity>
struct InlineFlatBufferPayload;

template <typename Root, std::size_t MetadataCapacity, std::size_t MaxTensors = 1U>
struct DescriptorFlatBufferPayload;

}  // namespace schema

/// Magic value for \ref ChannelSampleDescriptor payloads ("HSC5").
inline constexpr std::uint32_t kChannelSampleDescriptorMagic = 0x48534335U;

/// ABI version for \ref ChannelSampleDescriptor.
inline constexpr std::uint16_t kChannelSampleDescriptorAbiVersion = 2U;

/// Payload representation carried by a channel control descriptor.
enum class ChannelPayloadKind : std::uint16_t {
  /// The complete payload is embedded in the typed transport sample.
  kInlinePayload = 1U,
  /// The transport sample carries a token resolved by a same-process sidecar store.
  kLocalBufferDescriptor = 2U,
  /// The transport sample carries an opaque external-buffer reference.
  kBufferHandle = 3U,
};

/// Queue behavior requested for the channel's transport-control plane.
enum class ChannelQueuePolicy : std::uint8_t {
  /// Preserve lossless delivery by applying backpressure when subscriber queues are full.
  kBlocking = 0U,
  /// Keep the newest configured number of unread samples by replacing the oldest when full.
  kDropOldest = 1U,
};

/// @brief Select the Tensor payload representation for lower transport endpoints.
enum class ChannelTensorTransferMode : std::uint8_t {
  /// Keep Tensor handles in one process-local sidecar and transport address-space-local
  /// descriptors.
  kSameProcessSidecar,
  /// Transport opaque buffer references and Tensor view metadata, including across processes.
  kBufferHandle,
};

/** @brief Describe a graph-selected exportable Tensor pool for an interface output.
 * @details This setup-time value contains no backend types. Graph finalization resolves these
 * values before endpoint creation for use by the output-owned allocation path. */
struct InterfaceTensorPoolPlan {
  /// Storage kind for every slot in the pool.
  MemoryStorageType storage_type{MemoryStorageType::kDevice};
  /// CUDA device ordinal for device storage, or `-1` for host and system storage.
  std::int32_t cuda_device_ordinal{0};
  /// Maximum number of logical Tensor bytes that one slot can hold.
  std::uint64_t max_buffer_bytes{0U};
  /// Number of fixed-size slots owned by the output interface.
  std::uint32_t slot_count{0U};
  /// Fanout capacity resolved for the output interface.
  std::uint32_t fanout_capacity{1U};
  /// Maximum borrowed samples supported for each recipient.
  std::uint32_t max_borrowed_samples_per_recipient{1U};
};

/** @brief Setup-time options for the channel transport-control plane.
 * @details Queue policy, service maxima, publisher history, recipient capacity, and event-wake
 * configuration form one service-static contract and must match on publisher and subscriber
 * creation. Requested subscriber queue and borrow limits are subscriber-local; the publisher loan
 * limit and replacement-attribution request are publisher-local. Remaining fields configure
 * Tensor transfer, pool planning, and sidecar recipients. */
struct ChannelOptions {
  /// Service-static queue behavior for full subscriber queues.
  ChannelQueuePolicy queue_policy{ChannelQueuePolicy::kBlocking};
  /// Maximum per-subscriber queue depth supported by the publisher service.
  ///
  /// Zero derives this value from \ref subscriber_queue_depth. Graph finalization should set this
  /// to the maximum queue depth requested by subscribers on this logical channel.
  std::uint32_t max_subscriber_queue_depth{0U};
  /// Subscriber-local requested queue depth. Must be greater than zero and no greater than the
  /// service maximum.
  std::uint32_t subscriber_queue_depth{8U};
  /// Maximum samples each subscriber may borrow concurrently from the publisher service.
  ///
  /// Zero asks the transport to derive this value from the configured publisher service queue
  /// depth.
  std::uint32_t max_subscriber_borrowed_samples{0U};
  /// Requested borrowed-sample allowance for this subscriber.
  ///
  /// Zero asks the transport to derive this value from \ref subscriber_queue_depth.
  std::uint32_t max_borrowed_samples{0U};
  /// Service-static publisher history retained for late-joining subscribers. Zero disables
  /// history.
  std::uint32_t publisher_history_depth{0U};
  /// Maximum unsent samples a publisher may loan concurrently.
  ///
  /// Zero asks the transport to derive its compatibility default.
  std::uint32_t max_loaned_samples{0U};
  /// Service-static recipient capacity and expected same-process recipients for typed sidecars.
  std::uint32_t expected_recipient_count{1U};
  /// Enable the service-static event wake integration used by wait sets.
  bool enable_event_wake{false};
  /// Require per-recipient replacement attribution for descriptor-backed lossy delivery.
  ///
  /// Runtime may satisfy this through a setup-time publisher replacement callback rather than a
  /// per-send callback registry.
  bool replacement_attribution_required{false};
  /// Direct Tensor transport representation.
  ///
  /// Non-Tensor channels ignore this field. Opaque external-reference mode requires emitted
  /// Tensors to use endpoint-owned exportable storage; arbitrary external Tensor storage must be
  /// copied into the admitted export pool before it can cross a process boundary.
  ChannelTensorTransferMode tensor_transfer_mode{ChannelTensorTransferMode::kSameProcessSidecar};
  /// Graph-selected exportable Tensor pool plan for an interprocess output.
  ///
  /// This must be present only with \ref ChannelTensorTransferMode::kBufferHandle. Endpoint setup
  /// validates that its resolved fanout and borrowed-sample capacities agree with this option set.
  std::optional<InterfaceTensorPoolPlan> tensor_pool_plan{};
};

/// Fixed-size descriptor carried through the channel transport-control plane.
///
/// This descriptor is identity and metadata, not the buffer itself. For same-process tensor/buffer
/// channels, Holoscan Runtime resolves \ref sidecar_token against a runtime-owned sidecar store.
/// The private transport carries only this POD value and does not resolve the token.
struct ChannelSampleDescriptor {
  /// Descriptor magic. Receivers reject descriptors with a different value.
  std::uint32_t magic{kChannelSampleDescriptorMagic};
  /// Descriptor ABI version.
  std::uint16_t abi_version{kChannelSampleDescriptorAbiVersion};
  /// Payload representation selected by graph finalization.
  ChannelPayloadKind payload_kind{ChannelPayloadKind::kLocalBufferDescriptor};
  /// Runtime-defined flags. The private transport does not interpret these bits.
  std::uint32_t flags{};
  /// Number of recipients expected to observe this descriptor.
  std::uint32_t expected_recipient_count{};
  /// Origin process id used as a safety check for same-process descriptors.
  std::uint64_t process_id{};
  /// Runtime-owned sidecar token. Zero is reserved for an empty descriptor.
  std::uint64_t sidecar_token{};
  /// ABA-protection generation for the sidecar record.
  std::uint64_t generation{};
  /// Complete compact schema identity for schema-backed payloads.
  SchemaId schema_id{};
  /// Producer sequence number.
  std::uint64_t sequence{};
  /// Logical payload byte size.
  std::uint64_t byte_size{};

  /// Return whether the descriptor uses the ABI compiled into this Core build.
  [[nodiscard]] constexpr bool has_supported_abi() const noexcept {
    return magic == kChannelSampleDescriptorMagic &&
           abi_version == kChannelSampleDescriptorAbiVersion;
  }

  /// Return whether the descriptor is non-empty and ABI-compatible.
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return has_supported_abi() && sidecar_token != 0U;
  }
};

static_assert(std::is_standard_layout_v<ChannelSampleDescriptor>);
static_assert(std::is_trivially_copyable_v<ChannelSampleDescriptor>);
static_assert(sizeof(ChannelSampleDescriptor) == 72U);
static_assert(alignof(ChannelSampleDescriptor) == 8U);
static_assert(offsetof(ChannelSampleDescriptor, magic) == 0U);
static_assert(offsetof(ChannelSampleDescriptor, abi_version) == 4U);
static_assert(offsetof(ChannelSampleDescriptor, payload_kind) == 6U);
static_assert(offsetof(ChannelSampleDescriptor, flags) == 8U);
static_assert(offsetof(ChannelSampleDescriptor, expected_recipient_count) == 12U);
static_assert(offsetof(ChannelSampleDescriptor, process_id) == 16U);
static_assert(offsetof(ChannelSampleDescriptor, sidecar_token) == 24U);
static_assert(offsetof(ChannelSampleDescriptor, generation) == 32U);
static_assert(offsetof(ChannelSampleDescriptor, schema_id) == 40U);
static_assert(offsetof(ChannelSampleDescriptor, sequence) == 56U);
static_assert(offsetof(ChannelSampleDescriptor, byte_size) == 64U);

/// Map a private Channel representation to the Core whole-sample semantic kind.
///
/// @param kind Private Channel descriptor representation.
/// @return The corresponding Core kind, or an error for an unknown private discriminator.
/// @note The mapping is exhaustive and field based. The two enums are not integer-compatible wire
/// layouts and must never be converted with a cast.
[[nodiscard]] inline expected<PayloadKind, Error> core_payload_kind(
    const ChannelPayloadKind kind) noexcept {
  switch (kind) {
    case ChannelPayloadKind::kInlinePayload:
      return PayloadKind::kInlineTyped;
    case ChannelPayloadKind::kLocalBufferDescriptor:
    case ChannelPayloadKind::kBufferHandle:
      return PayloadKind::kDescriptorBacked;
    default:
      // This adapter is noexcept and can run on descriptor hot paths. Preserve the typed outcome
      // without allocating diagnostic text; callers may add context at a non-noexcept boundary.
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
}

/// Return the Core metadata subset represented by \p descriptor.
///
/// @param descriptor Private Channel transport descriptor.
/// @return Transport-neutral Core metadata, or an error for an unknown payload-kind discriminator.
[[nodiscard]] inline expected<SampleMetadata, Error> sample_metadata_from_descriptor(
    const ChannelSampleDescriptor& descriptor) noexcept {
  auto payload_kind = core_payload_kind(descriptor.payload_kind);
  if (!payload_kind) {
    return make_unexpected(std::move(payload_kind).error());
  }
  return SampleMetadata{
      .sequence = descriptor.sequence,
      .byte_size = descriptor.byte_size,
      .payload_kind = payload_kind.value(),
      .flags = descriptor.has_supported_abi() ? SampleFlags::kNone : SampleFlags::kInvalid};
}

/// Same-process payload reference carried by the current descriptor transport.
///
/// This is a Channel-layer reference into a runtime-owned sidecar store. It is not a durable or
/// cross-process identifier, and should not be exposed beyond runtime/debug surfaces.
struct LocalPayloadReference {
  /// Origin process id used to reject process-local handles from a different process.
  std::uint64_t process_id{};
  /// Runtime-owned sidecar token. Zero means no local payload reference.
  std::uint64_t token{};
  /// ABA-protection generation for the sidecar record.
  std::uint64_t generation{};
  /// Number of recipients expected to observe this payload reference.
  std::uint32_t expected_recipient_count{};
};

static_assert(std::is_standard_layout_v<LocalPayloadReference>);
static_assert(std::is_trivially_copyable_v<LocalPayloadReference>);

/// @brief Carry one bounded, versioned, transport-opaque external-buffer reference.
///
/// This Holoscan-owned value is the supported Channel boundary for descriptor-backed payloads.
/// Application code can inspect only its envelope version and occupied size; concrete transport,
/// file-descriptor, CUDA-handle, and backend-layout details remain source-private. The reference is
/// copied as an immutable value and is meaningful only to the runtime generation that materializes
/// it. It grants no ownership, import, or mutation authority by itself.
class BufferHandlePayloadReference {
 public:
  /// ABI version of the Holoscan opaque-reference envelope.
  static constexpr std::uint16_t kAbiVersion = 1U;
  /// Maximum number of source-private carrier bytes in one reference.
  static constexpr std::size_t kPayloadCapacity = 256U;

  /// Create an empty reference using the current envelope ABI.
  constexpr BufferHandlePayloadReference() noexcept = default;

  /// Return whether the envelope fields use the supported ABI and size bound.
  [[nodiscard]] constexpr bool has_supported_abi() const noexcept {
    return magic_ == kMagic && abi_version_ == kAbiVersion && payload_size_ <= kPayloadCapacity;
  }

  /// Return whether this reference contains a nonempty, structurally supported private carrier.
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return has_supported_abi() && payload_size_ != 0U;
  }

  /// Return the opaque-reference envelope ABI version.
  [[nodiscard]] constexpr std::uint16_t abi_version() const noexcept { return abi_version_; }

  /// Return the occupied source-private carrier size in bytes.
  [[nodiscard]] constexpr std::uint16_t payload_size() const noexcept { return payload_size_; }

 private:
  friend struct channel_internal::OpaquePayloadReferenceAccess;

  static constexpr std::uint32_t kMagic = 0x48534F52U;  // "HSOR"

  std::uint32_t magic_{kMagic};
  std::uint16_t abi_version_{kAbiVersion};
  std::uint16_t payload_size_{};
  alignas(std::uint64_t) std::array<std::byte, kPayloadCapacity> payload_{};
};

static_assert(std::is_standard_layout_v<BufferHandlePayloadReference>);
static_assert(std::is_trivially_copyable_v<BufferHandlePayloadReference>);
static_assert(sizeof(BufferHandlePayloadReference) == 264U);
static_assert(alignof(BufferHandlePayloadReference) == alignof(std::uint64_t));

/// Runtime payload reference associated with a sample envelope.
///
/// This is the production-direction shape that separates payload identity from temporal/framing
/// metadata. Today it is losslessly derived from the 72-byte same-process
/// `ChannelSampleDescriptor`. Cross-process Tensor paths carry a transport-opaque external-buffer
/// reference without changing `Sample<T>`. The local and opaque references share storage;
/// `payload_kind` is the active-arm discriminator.
struct PayloadDescriptor {
  /// Payload representation selected by graph finalization.
  ChannelPayloadKind payload_kind{ChannelPayloadKind::kInlinePayload};
  /// Runtime-defined payload flags.
  std::uint32_t flags{};
  /// Complete compact schema identity for schema-backed payloads.
  SchemaId schema_id{};
  /// Producer sequence number.
  std::uint64_t sequence{};
  /// Logical payload byte size.
  std::uint64_t byte_size{};
  union {
    /// Same-process sidecar reference, valid when payload_kind is `kLocalBufferDescriptor`.
    LocalPayloadReference local{};
    /// Cross-process opaque buffer reference, valid when payload_kind is `kBufferHandle`.
    BufferHandlePayloadReference buffer;
  };

  /// Return whether this payload references a same-process sidecar record.
  [[nodiscard]] constexpr bool is_local_buffer() const noexcept {
    return payload_kind == ChannelPayloadKind::kLocalBufferDescriptor && local.token != 0U;
  }

  /// @brief Return whether this payload has a structurally valid opaque buffer reference.
  /// @return True when the buffer-reference union arm is active and nonempty.
  [[nodiscard]] constexpr bool is_buffer_handle() const noexcept {
    return payload_kind == ChannelPayloadKind::kBufferHandle && static_cast<bool>(buffer);
  }
};

static_assert(std::is_standard_layout_v<PayloadDescriptor>);
static_assert(std::is_trivially_copyable_v<PayloadDescriptor>);
static_assert(sizeof(PayloadDescriptor) == 304U);

/// Public sample envelope: temporal/framing metadata plus a payload reference.
///
/// Application-facing `Sample<T>` APIs expose only `metadata`. Runtime code that materializes
/// descriptor-backed buffers can also use this envelope to keep temporal metadata distinct from
/// the payload reference used for lifetime/materialization.
struct SampleEnvelope {
  /// Public temporal/framing metadata.
  SampleMetadata metadata{};
  /// Runtime payload reference used to materialize descriptor-backed data.
  PayloadDescriptor payload{};
};

static_assert(std::is_standard_layout_v<SampleEnvelope>);
static_assert(std::is_trivially_copyable_v<SampleEnvelope>);

/// Return the payload-reference subset represented by \p descriptor.
[[nodiscard]] constexpr PayloadDescriptor payload_descriptor_from_descriptor(
    const ChannelSampleDescriptor& descriptor) noexcept {
  if (descriptor.payload_kind == ChannelPayloadKind::kBufferHandle) {
    // The same-process descriptor has no external-reference carrier. Preserve the discriminator
    // and common metadata while activating an empty opaque-reference arm, so callers cannot read
    // an inactive union member merely because an untrusted descriptor selected this kind.
    return PayloadDescriptor{.payload_kind = descriptor.payload_kind,
                             .flags = descriptor.flags,
                             .schema_id = descriptor.schema_id,
                             .sequence = descriptor.sequence,
                             .byte_size = descriptor.byte_size,
                             .buffer = BufferHandlePayloadReference{}};
  }
  return PayloadDescriptor{.payload_kind = descriptor.payload_kind,
                           .flags = descriptor.flags,
                           .schema_id = descriptor.schema_id,
                           .sequence = descriptor.sequence,
                           .byte_size = descriptor.byte_size,
                           .local = LocalPayloadReference{
                               .process_id = descriptor.process_id,
                               .token = descriptor.sidecar_token,
                               .generation = descriptor.generation,
                               .expected_recipient_count = descriptor.expected_recipient_count}};
}

/// Return the public envelope represented by \p descriptor.
[[nodiscard]] inline expected<SampleEnvelope, Error> sample_envelope_from_descriptor(
    const ChannelSampleDescriptor& descriptor) noexcept {
  if (descriptor.payload_kind == ChannelPayloadKind::kBufferHandle) {
    return make_unexpected(Error{
        ErrorCode::kProtocolError,
        "same-process channel descriptor cannot carry an opaque buffer reference",
    });
  }
  auto metadata = sample_metadata_from_descriptor(descriptor);
  if (!metadata) {
    return make_unexpected(std::move(metadata).error());
  }
  return SampleEnvelope{.metadata = metadata.value(),
                        .payload = payload_descriptor_from_descriptor(descriptor)};
}

/// Return the same-process 72-byte descriptor representation for \p payload.
///
/// This adapter exists while same-process sidecar internals still use
/// \ref ChannelSampleDescriptor. Temporal fields live in `SampleEnvelope::metadata` and are
/// therefore not represented in this descriptor.
///
/// @warning This conversion cannot carry `ChannelPayloadKind::kBufferHandle` payloads because the
/// same-process descriptor has no opaque external-reference field. Use \ref PayloadDescriptor
/// directly for cross-process descriptor-backed references. Non-local payloads return an empty
/// descriptor without inspecting the inactive local-reference union arm.
[[nodiscard]] constexpr ChannelSampleDescriptor channel_descriptor_from_payload_descriptor(
    const PayloadDescriptor& payload) noexcept {
  if (payload.payload_kind != ChannelPayloadKind::kLocalBufferDescriptor) {
    return {};
  }
  return ChannelSampleDescriptor{.magic = kChannelSampleDescriptorMagic,
                                 .abi_version = kChannelSampleDescriptorAbiVersion,
                                 .payload_kind = payload.payload_kind,
                                 .flags = payload.flags,
                                 .expected_recipient_count = payload.local.expected_recipient_count,
                                 .process_id = payload.local.process_id,
                                 .sidecar_token = payload.local.token,
                                 .generation = payload.local.generation,
                                 .schema_id = payload.schema_id,
                                 .sequence = payload.sequence,
                                 .byte_size = payload.byte_size};
}

}  // namespace holoscan
