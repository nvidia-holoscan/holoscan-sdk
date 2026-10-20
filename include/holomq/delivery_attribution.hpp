/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/byte_slice.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>

namespace holomq {

/// @brief Backend-independent identity for one pub/sub recipient.
///
/// For the iceoryx2 backend this is the 16-byte unique subscriber port id. HoloMQ treats it as an
/// opaque token so Channel/runtime code can map replacement/drop notifications back to the
/// subscriber it created without including backend headers.
struct DeliveryRecipientId {
  static constexpr std::size_t kSize = 16U;

  std::array<std::uint8_t, kSize> bytes{};
};

/// @brief Return whether two recipient ids refer to the same backend recipient.
/// @param lhs First recipient id to compare.
/// @param rhs Second recipient id to compare.
/// @return `true` when the opaque byte values match.
[[nodiscard]] constexpr bool operator==(const DeliveryRecipientId& lhs,
                                        const DeliveryRecipientId& rhs) noexcept {
  return lhs.bytes == rhs.bytes;
}

/// @brief Return whether two recipient ids refer to different backend recipients.
/// @param lhs First recipient id to compare.
/// @param rhs Second recipient id to compare.
/// @return `true` when the opaque byte values differ.
[[nodiscard]] constexpr bool operator!=(const DeliveryRecipientId& lhs,
                                        const DeliveryRecipientId& rhs) noexcept {
  return !(lhs == rhs);
}

/// @brief Metadata for a sample replaced for one recipient by a drop-oldest queue policy.
struct SampleReplacedInfo {
  /// @brief Recipient whose unread queue entry was replaced.
  DeliveryRecipientId receiver_id{};
  /// @brief Read-only pointer to the replaced payload, valid only during the replacement callback.
  const void* payload_ptr{};
  /// @brief Number of bytes available at @ref payload_ptr.
  std::size_t payload_size{};
};

/// @brief Setup-time callback invoked when a drop-oldest queue replaces a sample for one recipient.
/// @tparam Payload Application payload type stored in the typed pub/sub sample.
///
/// The callback receives `const Payload&`, or the replaced frame's bytes for \ref ByteSlice. That
/// view and @ref SampleReplacedInfo::payload_ptr are valid only during the callback. Use this path
/// for descriptor-backed payloads that can be cleaned up directly from the replaced payload value,
/// avoiding per-send callback registration. A replacement event describes a recipient queue
/// overwrite; the transport may later report the same physical sample through the sample-release
/// path when its shared-memory slot is finally released.
template <typename Payload>
using SampleReplacedCallback =
    std::function<void(ConstPayloadView<Payload>, const SampleReplacedInfo&)>;

}  // namespace holomq
