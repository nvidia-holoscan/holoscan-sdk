/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <span>
#include <type_traits>

#include <holoscan/channel/types.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan::channel_internal {

namespace payload_descriptor_wire_detail {

template <typename T>
[[nodiscard]] bool write_payload_descriptor_field(const std::span<std::byte> destination,
                                                  const std::size_t offset,
                                                  const T& value) noexcept {
  static_assert(std::is_trivially_copyable_v<T>);
  if (offset > destination.size() || sizeof(T) > destination.size() - offset) {
    return false;
  }
  std::memcpy(destination.data() + offset, &value, sizeof(T));
  return true;
}

template <typename T>
[[nodiscard]] bool read_payload_descriptor_field(const std::span<const std::byte> source,
                                                 const std::size_t offset, T& value) noexcept {
  static_assert(std::is_trivially_copyable_v<T>);
  if (offset > source.size() || sizeof(T) > source.size() - offset) {
    return false;
  }
  std::memcpy(&value, source.data() + offset, sizeof(T));
  return true;
}

}  // namespace payload_descriptor_wire_detail

[[nodiscard]] inline expected<void, Error> encode_payload_descriptor_wire(
    const PayloadDescriptor& descriptor, const std::span<std::byte> destination) noexcept {
  if (destination.size() != sizeof(PayloadDescriptor)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  std::ranges::fill(destination, std::byte{});
  const auto payload_kind =
      static_cast<std::underlying_type_t<ChannelPayloadKind>>(descriptor.payload_kind);
  bool encoded = payload_descriptor_wire_detail::write_payload_descriptor_field(
                     destination, offsetof(PayloadDescriptor, payload_kind), payload_kind) &&
                 payload_descriptor_wire_detail::write_payload_descriptor_field(
                     destination, offsetof(PayloadDescriptor, flags), descriptor.flags) &&
                 payload_descriptor_wire_detail::write_payload_descriptor_field(
                     destination, offsetof(PayloadDescriptor, schema_id), descriptor.schema_id) &&
                 payload_descriptor_wire_detail::write_payload_descriptor_field(
                     destination, offsetof(PayloadDescriptor, sequence), descriptor.sequence) &&
                 payload_descriptor_wire_detail::write_payload_descriptor_field(
                     destination, offsetof(PayloadDescriptor, byte_size), descriptor.byte_size);
  constexpr std::size_t kReferenceOffset = offsetof(PayloadDescriptor, local);
  switch (descriptor.payload_kind) {
    case ChannelPayloadKind::kInlinePayload:
      break;
    case ChannelPayloadKind::kLocalBufferDescriptor:
      encoded = encoded &&
                payload_descriptor_wire_detail::write_payload_descriptor_field(
                    destination,
                    kReferenceOffset + offsetof(LocalPayloadReference, process_id),
                    descriptor.local.process_id) &&
                payload_descriptor_wire_detail::write_payload_descriptor_field(
                    destination,
                    kReferenceOffset + offsetof(LocalPayloadReference, token),
                    descriptor.local.token) &&
                payload_descriptor_wire_detail::write_payload_descriptor_field(
                    destination,
                    kReferenceOffset + offsetof(LocalPayloadReference, generation),
                    descriptor.local.generation) &&
                payload_descriptor_wire_detail::write_payload_descriptor_field(
                    destination,
                    kReferenceOffset + offsetof(LocalPayloadReference, expected_recipient_count),
                    descriptor.local.expected_recipient_count);
      break;
    case ChannelPayloadKind::kBufferHandle:
      encoded = encoded && payload_descriptor_wire_detail::write_payload_descriptor_field(
                               destination, kReferenceOffset, descriptor.buffer);
      break;
    default:
      return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  if (!encoded) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  return {};
}

[[nodiscard]] inline expected<PayloadDescriptor, Error> decode_payload_descriptor_wire(
    const std::span<const std::byte> source) noexcept {
  if (source.size() != sizeof(PayloadDescriptor)) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  PayloadDescriptor descriptor{};
  std::underlying_type_t<ChannelPayloadKind> payload_kind = 0U;
  const bool decoded = payload_descriptor_wire_detail::read_payload_descriptor_field(
                           source, offsetof(PayloadDescriptor, payload_kind), payload_kind) &&
                       payload_descriptor_wire_detail::read_payload_descriptor_field(
                           source, offsetof(PayloadDescriptor, flags), descriptor.flags) &&
                       payload_descriptor_wire_detail::read_payload_descriptor_field(
                           source, offsetof(PayloadDescriptor, schema_id), descriptor.schema_id) &&
                       payload_descriptor_wire_detail::read_payload_descriptor_field(
                           source, offsetof(PayloadDescriptor, sequence), descriptor.sequence) &&
                       payload_descriptor_wire_detail::read_payload_descriptor_field(
                           source, offsetof(PayloadDescriptor, byte_size), descriptor.byte_size);
  if (!decoded) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  switch (payload_kind) {
    case static_cast<std::underlying_type_t<ChannelPayloadKind>>(
        ChannelPayloadKind::kInlinePayload):
      descriptor.payload_kind = ChannelPayloadKind::kInlinePayload;
      break;
    case static_cast<std::underlying_type_t<ChannelPayloadKind>>(
        ChannelPayloadKind::kLocalBufferDescriptor):
      descriptor.payload_kind = ChannelPayloadKind::kLocalBufferDescriptor;
      if (!payload_descriptor_wire_detail::read_payload_descriptor_field(
              source,
              offsetof(PayloadDescriptor, local) + offsetof(LocalPayloadReference, process_id),
              descriptor.local.process_id) ||
          !payload_descriptor_wire_detail::read_payload_descriptor_field(
              source,
              offsetof(PayloadDescriptor, local) + offsetof(LocalPayloadReference, token),
              descriptor.local.token) ||
          !payload_descriptor_wire_detail::read_payload_descriptor_field(
              source,
              offsetof(PayloadDescriptor, local) + offsetof(LocalPayloadReference, generation),
              descriptor.local.generation) ||
          !payload_descriptor_wire_detail::read_payload_descriptor_field(
              source,
              offsetof(PayloadDescriptor, local) +
                  offsetof(LocalPayloadReference, expected_recipient_count),
              descriptor.local.expected_recipient_count)) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      break;
    case static_cast<std::underlying_type_t<ChannelPayloadKind>>(ChannelPayloadKind::kBufferHandle):
      descriptor.payload_kind = ChannelPayloadKind::kBufferHandle;
      if (!payload_descriptor_wire_detail::read_payload_descriptor_field(
              source, offsetof(PayloadDescriptor, buffer), descriptor.buffer)) {
        return make_unexpected(Error{ErrorCode::kProtocolError});
      }
      break;
    default:
      return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  return descriptor;
}

}  // namespace holoscan::channel_internal
