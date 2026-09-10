/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "catalog_inline_payload_contract.hpp"
#include "catalog_received_operation.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <utility>

#include <holoscan/channel/runtime.hpp>

namespace holoscan::channel_internal {

/// @brief Canonicalize catalog scalar values at the transport attachment boundary.
///
/// Public scalar facades pass native object bytes to their erased endpoints. This source-private
/// codec is the single owner of conversion between that native representation and the catalog's
/// fixed-width little-endian wire representation. In particular, the native width of \c bool is
/// \c sizeof(bool), while its only admitted wire representation is one byte with value zero or one.
class CatalogScalarCodec {
 public:
  /// @brief Return the fixed catalog wire width for one scalar identity.
  /// @param kind Exact admitted scalar identity.
  /// @return Width in bytes, or zero for the unsupported local-enum identity.
  [[nodiscard]] static constexpr std::size_t wire_size(const ScalarIdentityKind kind) noexcept {
    return catalog_scalar_wire_size(kind);
  }

  /// @brief Return the native object width accepted for one scalar identity.
  /// @param kind Exact admitted scalar identity.
  /// @return Native width in bytes, or zero for the unsupported local-enum identity.
  [[nodiscard]] static constexpr std::size_t native_size(const ScalarIdentityKind kind) noexcept {
    return catalog_scalar_native_size(kind);
  }

  /// @brief Encode one native scalar into the canonical catalog representation.
  /// @param kind Exact admitted scalar identity.
  /// @param native Native object bytes with exactly \ref native_size bytes.
  /// @param wire Destination with exactly \ref wire_size bytes.
  /// @return Success, or \ref ErrorCode::kInvalidArgument without modifying an invalid request.
  [[nodiscard]] static expected<void, Error> encode(const ScalarIdentityKind kind,
                                                    std::span<const std::byte> native,
                                                    std::span<std::byte> wire) noexcept {
    return encode_for_native_endian<std::endian::native>(kind, native, wire);
  }

  /// @brief Encode bytes represented in one compile-time native byte order.
  /// @tparam NativeEndian Byte order represented by \p native.
  /// @param kind Exact admitted scalar identity.
  /// @param native Native-order object bytes.
  /// @param wire Canonical little-endian destination.
  /// @return Success or a checked size mismatch.
  /// @note Production uses only `std::endian::native`; the explicit template is a source-private
  /// conformance seam for a host byte order unavailable to the test runner.
  template <std::endian NativeEndian>
  [[nodiscard]] static expected<void, Error> encode_for_native_endian(
      const ScalarIdentityKind kind, std::span<const std::byte> native,
      std::span<std::byte> wire) noexcept {
    static_assert(NativeEndian == std::endian::little || NativeEndian == std::endian::big);
    if (!valid_sizes(kind, native.size(), wire.size())) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument, "ENDPOINT_TYPE_MISMATCH"});
    }
    if (kind == ScalarIdentityKind::kBoolean) {
      bool value{};
      std::memcpy(&value, native.data(), sizeof(value));
      wire.front() = static_cast<std::byte>(value ? 1U : 0U);
      return {};
    }
    copy_native_to_little<NativeEndian>(native, wire);
    return {};
  }

  /// @brief Decode one canonical catalog scalar into native object storage.
  /// @param kind Exact admitted scalar identity.
  /// @param wire Canonical bytes with exactly \ref wire_size bytes.
  /// @param native Destination with exactly \ref native_size bytes.
  /// @return Success, \ref ErrorCode::kInvalidArgument for a size mismatch, or
  /// \ref ErrorCode::kProtocolError for a noncanonical Boolean.
  /// @note A failed decode does not modify \p native.
  [[nodiscard]] static expected<void, Error> decode(const ScalarIdentityKind kind,
                                                    std::span<const std::byte> wire,
                                                    std::span<std::byte> native) noexcept {
    return decode_for_native_endian<std::endian::native>(kind, wire, native);
  }

  /// @brief Decode canonical bytes into one compile-time native byte order.
  /// @tparam NativeEndian Byte order to represent in \p native.
  /// @param kind Exact admitted scalar identity.
  /// @param wire Canonical little-endian bytes.
  /// @param native Native-order destination.
  /// @return Success or a checked size/noncanonical-Boolean error.
  /// @note Production uses only `std::endian::native`; the explicit template is a source-private
  /// conformance seam for a host byte order unavailable to the test runner.
  template <std::endian NativeEndian>
  [[nodiscard]] static expected<void, Error> decode_for_native_endian(
      const ScalarIdentityKind kind, std::span<const std::byte> wire,
      std::span<std::byte> native) noexcept {
    static_assert(NativeEndian == std::endian::little || NativeEndian == std::endian::big);
    if (!valid_sizes(kind, native.size(), wire.size())) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument, "ENDPOINT_TYPE_MISMATCH"});
    }
    if (kind == ScalarIdentityKind::kBoolean) {
      if (wire.front() != std::byte{0U} && wire.front() != std::byte{1U}) {
        return make_unexpected(Error{ErrorCode::kProtocolError, "ENDPOINT_TYPE_MISMATCH"});
      }
      const bool value = wire.front() == std::byte{1U};
      std::memcpy(native.data(), &value, sizeof(value));
      return {};
    }
    copy_little_to_native<NativeEndian>(wire, native);
    return {};
  }

 private:
  [[nodiscard]] static constexpr bool valid_sizes(const ScalarIdentityKind kind,
                                                  const std::size_t native,
                                                  const std::size_t wire) noexcept {
    return wire_size(kind) != 0U && native == native_size(kind) && wire == wire_size(kind);
  }

  template <std::endian NativeEndian>
  static void copy_native_to_little(std::span<const std::byte> source,
                                    std::span<std::byte> destination) noexcept {
    if constexpr (NativeEndian == std::endian::little) {
      std::copy(source.begin(), source.end(), destination.begin());
    } else {
      std::reverse_copy(source.begin(), source.end(), destination.begin());
    }
  }

  template <std::endian NativeEndian>
  static void copy_little_to_native(std::span<const std::byte> source,
                                    std::span<std::byte> destination) noexcept {
    if constexpr (NativeEndian == std::endian::little) {
      std::copy(source.begin(), source.end(), destination.begin());
    } else {
      std::reverse_copy(source.begin(), source.end(), destination.begin());
    }
  }
};

/// @brief Publish already-canonical custom-POD bytes through one catalog endpoint.
///
/// Runtime binds this table once during cold setup after independently validating the complete POD
/// identity. It avoids a second fieldwise encode for external and mixed-fanout Graph outputs.
struct CatalogCanonicalPodPublisherBinding {
  void* state{};  ///< Non-owning catalog publisher state retained by the endpoint owner.
  InlinePublicationResult (*emit_phase)(void*, std::span<const std::byte>,
                                        const SampleMetadata&) = nullptr;  ///< Canonical call.

  /// @brief Return whether this binding contains a complete callable table.
  explicit operator bool() const noexcept { return state != nullptr && emit_phase != nullptr; }
};

/// @brief Select one source-tree-only corruption for the next generated-schema publication.
enum class CatalogSchemaFrameCorruption : std::uint8_t {
  kMalformedEnvelope = 0,   ///< Replace the envelope header-size field with an invalid value.
  kSchemaId = 1,            ///< Replace the first compact schema-id byte.
  kSchemaVersion = 2,       ///< Replace the exact semantic schema version.
  kEndpointGeneration = 3,  ///< Replace the endpoint generation bound at attachment.
  kPayloadVerifier = 4,     ///< Preserve the envelope but corrupt the serialized payload.
};

/// @brief Reach catalog Channel endpoint materialization without exposing backend handles.
///
/// This source-private accessor is the seam for the later graph Runtime. It deliberately returns
/// the same erased data endpoints used by standalone facades and exposes only transport-neutral
/// applied health evidence.
class CatalogChannelBindingAccess {
 public:
  /// @brief Create the graph-side publisher for one catalog export endpoint.
  template <CatalogChannelValue T>
    requires PublishedPayload<T>
  [[nodiscard]] static expected<std::unique_ptr<InlinePayloadPublisher>, Error> create_publisher(
      ChannelRuntime& runtime, ChannelEndpointCatalog::EndpointView endpoint,
      ChannelAttachOptions options = {}) {
    return runtime.create_catalog_inline_payload_publisher(endpoint,
                                                           catalog_payload_identity<T>(),
                                                           sizeof(T),
                                                           std::move(options),
                                                           TopicDirection::kExport,
                                                           false);
  }

  /// @brief Create one type-erased scalar or custom-POD publisher from exact catalog identity.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<std::unique_ptr<InlinePayloadPublisher>, Error>
  create_payload_publisher(ChannelRuntime& runtime, ChannelEndpointCatalog::EndpointView endpoint,
                           std::size_t payload_size, ChannelAttachOptions options = {});

  /// @brief Create a type-erased catalog publisher with graph-owned pressure handling.
  /// @param runtime Runtime that owns the publisher transport context.
  /// @param endpoint Exact catalog export endpoint to attach.
  /// @param payload_size Canonical inline payload size in bytes.
  /// @param options Attachment options applied at endpoint creation.
  /// @param pressure Graph-owned publication pressure policy.
  /// @return Attached publisher, or a checked setup error.
  /// @throws std::bad_alloc if required setup storage cannot be allocated.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<std::unique_ptr<InlinePayloadPublisher>, Error>
  create_payload_publisher(ChannelRuntime& runtime, ChannelEndpointCatalog::EndpointView endpoint,
                           std::size_t payload_size, ChannelAttachOptions options,
                           const PublisherPressureOptions& pressure);

  /// @brief Bind the canonical custom-POD publication path once during Runtime setup.
  /// @param publisher Catalog payload publisher returned by this accessor.
  /// @return Callable binding for an exact custom-POD endpoint, or an empty binding otherwise.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static CatalogCanonicalPodPublisherBinding
  bind_canonical_pod_publisher(InlinePayloadPublisher& publisher) noexcept;

  /// @brief Create the graph-side subscriber for one catalog import endpoint.
  template <CatalogChannelValue T>
    requires PublishedPayload<T>
  [[nodiscard]] static expected<std::unique_ptr<InlinePayloadSubscriber>, Error> create_subscriber(
      ChannelRuntime& runtime, ChannelEndpointCatalog::EndpointView endpoint,
      ChannelAttachOptions options = {}) {
    return runtime.create_catalog_inline_payload_subscriber(endpoint,
                                                            catalog_payload_identity<T>(),
                                                            sizeof(T),
                                                            std::move(options),
                                                            TopicDirection::kImport,
                                                            false);
  }

  /// @brief Create one type-erased scalar or custom-POD subscriber from exact catalog identity.
  /// @param runtime Runtime that owns the transport context.
  /// @param endpoint Exact immutable import endpoint.
  /// @param payload_size Native value width expected by the replay declaration.
  /// @param options Caller-local attach options that may only shorten the catalog bound.
  /// @return Catalog subscriber with private data/health ordering, or a checked attach error.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<std::unique_ptr<InlinePayloadSubscriber>,
                                                     Error>
  create_payload_subscriber(ChannelRuntime& runtime, ChannelEndpointCatalog::EndpointView endpoint,
                            std::size_t payload_size, ChannelAttachOptions options = {});

  /// @brief Consume at most one next publisher operation.
  /// @param subscriber Catalog subscriber returned by this accessor.
  /// @param payload Destination for a decoded data operation.
  /// @param accept_data Whether the caller can retain a contiguous data operation now.
  /// @param borrow_schema_payload Return schema bytes borrowed from bounded staging instead of
  ///     copying them into `payload`. Scalar and POD data always decode into `payload`.
  /// @return Health, data, retained pending data, or no-current-operation without exposing private
  /// wire bytes. Lossless queues require the exact contiguous operation; replacement queues may
  /// advance the shared operation sequence to their oldest observable data operation after an
  /// overwritten unread predecessor. The separate health service and its report sequence remain
  /// lossless and contiguous.
  /// @note A health-only operation never appears through ordinary \c receive() and carries no
  /// readiness semantics. The caller must serialize this function with all other receives.
  /// @note A borrowed schema span remains valid only until the next receive on this subscriber.
  /// @note When \p accept_data is false, an earlier contiguous health operation is still applied,
  /// but a contiguous data operation remains in the subscriber's bounded staging slot and is
  /// returned as \ref CatalogReceivedOperationKind::kDataPending without advancing the operation
  /// sequence or modifying \p payload.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<CatalogReceivedOperation, Error> receive_next(
      InlinePayloadSubscriber& subscriber, std::span<std::byte> payload, bool accept_data = true,
      bool borrow_schema_payload = false) noexcept;

  /// @brief Report whether either private catalog stream has an unconsumed operation.
  /// @param subscriber Catalog subscriber returned by this accessor.
  /// @return Whether data or clock-health control evidence is pending without consuming it.
  /// @note This cold quiescence probe includes the subscriber's bounded staging slots and the
  /// underlying private services. It does not advance the combined operation sequence.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<bool, Error> has_pending_operation(
      const InlinePayloadSubscriber& subscriber) noexcept;

  /// @brief Return receiver-applied private health evidence without consuming application data.
  /// @param subscriber Catalog subscriber returned by this accessor.
  /// @return Current bounded state, or ErrorCode::kInvalidArgument for another endpoint kind.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<CatalogClockHealthState, Error>
  clock_health_state(const InlinePayloadSubscriber& subscriber) noexcept;

  /// @brief Return the producer-incarnation token latched from the first accepted operation.
  /// @param subscriber Catalog subscriber returned by this accessor.
  /// @return Latched token, zero before any accepted operation, or ErrorCode::kInvalidArgument
  /// for another endpoint kind.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<std::uint64_t, Error>
  latched_producer_generation(const InlinePayloadSubscriber& subscriber) noexcept;

  /// @brief Record the incarnation token a runtime rebind replaced on a fresh subscriber.
  /// @param subscriber Freshly created catalog subscriber with no accepted or staged operation.
  /// @param generation Nonzero token of the detached predecessor incarnation.
  /// @return Success, or ErrorCode::kInvalidArgument for another endpoint kind, a zero token, or
  /// a subscriber that already latched, staged, or closed.
  /// @note Frames still stamped with the superseded token are then fenced as
  /// \ref CatalogReceivedOperationKind::kPeerIncarnationChanged even before the replacement
  /// incarnation latches.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<void, Error> supersede_producer_generation(
      InlinePayloadSubscriber& subscriber, std::uint64_t generation) noexcept;

  /// @brief Attribute pending private traffic to producer incarnations for the rebind gate.
  /// @param subscriber Catalog subscriber returned by this accessor.
  /// @return Bounded classification snapshot; staging may consume transport queue heads into the
  /// subscriber's retained per-lane staging slots but never delivers payload bytes or advances
  /// the combined operation sequence.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<CatalogImportQuiescence, Error>
  import_quiescence(InlinePayloadSubscriber& subscriber) noexcept;
};

/// @brief Reach deterministic source-tree-only seams for catalog binding conformance tests.
///
/// This class is declared only in a source-private header and is not installed. It can apply one
/// bounded post-encoding schema fault or delay receiver observation without exposing backend
/// handles, allowing adversarial tests to exercise the production receive and ordering paths.
class CatalogChannelBindingTestAccess {
 public:
  /// @brief Corrupt one generated-schema frame after normal publisher encoding.
  /// @param publisher Catalog publisher returned by \ref CatalogChannelBindingAccess.
  /// @param corruption Exact one-shot corruption to apply before backend submission.
  /// @return Success, or \ref ErrorCode::kInvalidArgument for another endpoint kind, an unknown
  /// corruption selector, or an already armed publisher.
  /// @note This source-tree-only seam preserves the normal catalog service, queue, attribution,
  /// metadata, and receive path. It exposes no raw transport handle or public injection API.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<void, Error> corrupt_next_schema_publication(
      InlinePayloadPublisher& publisher, CatalogSchemaFrameCorruption corruption) noexcept;

  /// @brief Return the producer-incarnation token of one catalog publisher attachment.
  /// @param publisher Catalog publisher returned by \ref CatalogChannelBindingAccess.
  /// @return Nonzero identity token, or \ref ErrorCode::kInvalidArgument for another kind.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<std::uint64_t, Error> producer_generation(
      const InlinePayloadPublisher& publisher) noexcept;

  /// @brief Impersonate one exact prior producer incarnation before any publication.
  /// @param publisher Catalog publisher returned by \ref CatalogChannelBindingAccess that has not
  /// published any operation yet.
  /// @param generation Nonzero incarnation token to stamp on every subsequent frame.
  /// @return Success, or \ref ErrorCode::kInvalidArgument for another endpoint kind, a zero
  /// token, or a publisher that already published or closed.
  /// @note This source-tree-only seam lets conformance tests prove the receiver generation fence
  /// against stale-incarnation frames without exposing any raw transport handle.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<void, Error> override_producer_generation(
      InlinePayloadPublisher& publisher, std::uint64_t generation) noexcept;

  /// @brief Delay health-lane staging for a bounded number of receive attempts.
  /// @param subscriber Catalog subscriber returned by \ref CatalogChannelBindingAccess.
  /// @param attempts Positive number of subsequent receives that shall not stage health.
  /// @return Success, or \ref ErrorCode::kInvalidArgument for another endpoint kind or zero.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<void, Error> delay_health_observation(
      InlinePayloadSubscriber& subscriber, std::uint32_t attempts) noexcept;

  /// @brief Validate one untrusted data-lane metadata record against an endpoint contract.
  /// @param endpoint Immutable catalog endpoint that scopes payload and clock identities.
  /// @param metadata Metadata observed from the private transport service.
  /// @return Success only for the exact canonical catalog representation, or a protocol error.
  /// @note This source-tree-only seam invokes the same validator used before subscriber staging;
  /// it neither consumes transport nor changes pending operation state.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<void, Error> validate_data_metadata(
      ChannelEndpointCatalog::EndpointView endpoint, const SampleMetadata& metadata) noexcept;
};

}  // namespace holoscan::channel_internal
