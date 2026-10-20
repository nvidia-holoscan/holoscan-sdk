/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/holomq_export.h>
#include <holomq/pubsub_options.hpp>

#include <cstdint>

namespace holomq {

/// @brief Buffer-reference mode a typed pub/sub payload can represent.
///
/// These modes describe what the payload *carries*. HoloMQ transports the payload bytes; Holoscan
/// Runtime decides which mode a Channel uses at graph finalization.
enum class BufferTransferMode : std::uint8_t {
  /// The complete application value is stored directly in the typed pub/sub payload.
  kInlinePayload,
  /// The payload carries a @ref LocalBufferDescriptor resolved by a runtime-owned sidecar store.
  kLocalDescriptor,
  /// The payload carries a HoloMQ \ref BufferHandle resolved through VMM/fd-pass or host SHM.
  kBufferHandle,
};

/// @brief Per-subscriber delivery identity exposed by the pub/sub backend.
enum class DeliveryAttribution : std::uint8_t {
  /// The backend does not report which recipients accepted, skipped, or dropped a sample.
  kNone,
  /// Publish reports an aggregate delivered-recipient count only.
  kDeliveredCountOnly,
  /// Publish/drop reporting identifies individual recipients.
  kPerRecipient,
};

/// @brief Setup-time capabilities of HoloMQ typed pub/sub.
///
/// This is intentionally conservative and public: Holoscan Runtime can inspect it during graph
/// finalization and reject channel plans that require unsupported queue, fanout, readiness, or
/// buffer-transfer behavior. It is not used on the steady-state message path.
struct PubSubCapabilities {
  /// Typed standard-layout/trivially-copyable payloads are supported.
  bool typed_payloads{true};
  /// One publisher can have multiple subscribers.
  bool fanout{true};
  /// Subscriber readiness can be queried without consuming a sample.
  bool non_consuming_ready{true};
  /// A wait set can multiplex subscriber readiness.
  bool waitset{true};
  /// Kernel/event-backed waits are supported when endpoints are created with event wake enabled.
  bool kernel_wait{true};
  /// Lossless/backpressured queue policy is supported.
  bool blocking_queue_policy{true};
  /// Bounded drop-oldest queue policy is supported.
  bool drop_oldest_queue_policy{true};
  /// The payload can store complete inline values.
  bool inline_payload_transfer{true};
  /// The payload can carry local descriptors without invoking HoloMQ buffer import/export.
  bool local_descriptor_payload_transfer{true};
  /// The payload can carry @ref BufferHandle values for cross-process buffer import/export.
  bool buffer_handle_payload_transfer{true};
  /// HoloMQ itself resolves same-process local descriptors into buffer lifetimes.
  bool local_descriptor_resolution{false};
  /// HoloMQ resolves \ref BufferHandle values through its cross-process buffer plane.
  bool buffer_handle_resolution{true};
  /// Granularity of delivery/drop attribution.
  DeliveryAttribution delivery_attribution{DeliveryAttribution::kDeliveredCountOnly};
};

/// @brief Return whether endpoints can be created using a queue policy.
/// @param capabilities Capability set to inspect.
/// @param policy Queue policy requested by the endpoint builder.
/// @return `true` when @p capabilities advertises support for @p policy.
[[nodiscard]] constexpr bool supports_queue_policy(const PubSubCapabilities& capabilities,
                                                   const QueuePolicy policy) noexcept {
  switch (policy) {
    case QueuePolicy::kBlocking:
      return capabilities.blocking_queue_policy;
    case QueuePolicy::kDropOldest:
      return capabilities.drop_oldest_queue_policy;
    default:
      return false;
  }
}

/// @brief Return whether typed pub/sub can carry payloads using a buffer transfer mode.
/// @param capabilities Capability set to inspect.
/// @param mode Buffer transfer mode requested by Runtime.
/// @return `true` when the typed pub/sub sample can transport @p mode.
///
/// This answers whether the typed pub/sub sample can *transport* that reference kind. It does not
/// imply that HoloMQ resolves the reference into a buffer lifetime.
[[nodiscard]] constexpr bool supports_buffer_transfer_mode(const PubSubCapabilities& capabilities,
                                                           const BufferTransferMode mode) noexcept {
  switch (mode) {
    case BufferTransferMode::kInlinePayload:
      return capabilities.inline_payload_transfer;
    case BufferTransferMode::kLocalDescriptor:
      return capabilities.local_descriptor_payload_transfer;
    case BufferTransferMode::kBufferHandle:
      return capabilities.buffer_handle_payload_transfer;
    default:
      return false;
  }
}

/// @brief Return whether HoloMQ itself resolves payload references using a transfer mode.
/// @param capabilities Capability set to inspect.
/// @param mode Buffer transfer mode requested by Runtime.
/// @return `true` when HoloMQ resolves payload references for @p mode.
///
/// Same-process descriptors are transported but intentionally resolved by Holoscan Runtime, not by
/// HoloMQ.
[[nodiscard]] constexpr bool resolves_buffer_transfer_mode(const PubSubCapabilities& capabilities,
                                                           const BufferTransferMode mode) noexcept {
  switch (mode) {
    case BufferTransferMode::kInlinePayload:
      return capabilities.inline_payload_transfer;
    case BufferTransferMode::kLocalDescriptor:
      return capabilities.local_descriptor_resolution;
    case BufferTransferMode::kBufferHandle:
      return capabilities.buffer_handle_resolution;
    default:
      return false;
  }
}

/// @brief Return compile-time HoloMQ typed pub/sub capabilities for the selected backend.
/// @return Capability flags for the backend selected when HoloMQ was compiled.
[[nodiscard]] HOLOMQ_EXPORT PubSubCapabilities default_pubsub_capabilities() noexcept;

}  // namespace holomq
