/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/payload_identity.hpp>
#include <holoscan/schema/schema_package.hpp>

namespace holoscan::channel_internal {

inline constexpr std::uint64_t kInitialSchemaEndpointGeneration = 1U;

namespace schema_envelope_detail {

template <typename Integer>
void store_little(std::span<std::byte> destination, std::size_t offset, Integer value) noexcept {
  static_assert(std::is_unsigned_v<Integer>);
  for (std::size_t index = 0U; index < sizeof(Integer); ++index) {
    destination[offset + index] =
        static_cast<std::byte>((value >> (index * 8U)) & static_cast<Integer>(0xffU));
  }
}

template <typename Integer>
[[nodiscard]] Integer load_little(std::span<const std::byte> source, std::size_t offset) noexcept {
  static_assert(std::is_unsigned_v<Integer>);
  Integer result{};
  for (std::size_t index = 0U; index < sizeof(Integer); ++index) {
    result |= static_cast<Integer>(std::to_integer<std::uint8_t>(source[offset + index]))
              << (index * 8U);
  }
  return result;
}

}  // namespace schema_envelope_detail

[[nodiscard]] inline expected<void, Error> encode_schema_envelope(
    std::span<std::byte> wire, const SchemaIdentityRecord& identity, const std::size_t payload_size,
    const std::uint64_t endpoint_generation = kInitialSchemaEndpointGeneration) noexcept {
  if (wire.size() < kSchemaEnvelopeHeaderSize || identity.compatibility_epoch == 0U ||
      endpoint_generation == 0U || payload_size == 0U ||
      payload_size > std::numeric_limits<std::uint32_t>::max()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  schema_envelope_detail::store_little<std::uint32_t>(wire, 0U, kSchemaEnvelopeABIVersion);
  schema_envelope_detail::store_little<std::uint32_t>(wire, 4U, kSchemaEnvelopeHeaderSize);
  std::copy(identity.family_id.value.begin(), identity.family_id.value.end(), wire.begin() + 8U);
  schema_envelope_detail::store_little<std::uint32_t>(
      wire, 24U, static_cast<std::uint32_t>(payload_size));
  schema_envelope_detail::store_little<std::uint32_t>(wire, 28U, 0U);
  schema_envelope_detail::store_little<std::uint64_t>(wire, 32U, endpoint_generation);
  return {};
}

[[nodiscard]] inline expected<std::size_t, Error> decode_schema_envelope(
    std::span<const std::byte> wire, const SchemaIdentityRecord& expected,
    const std::size_t payload_capacity,
    const std::uint64_t expected_generation = kInitialSchemaEndpointGeneration) noexcept {
  if (wire.size() < kSchemaEnvelopeHeaderSize || expected.compatibility_epoch == 0U ||
      expected_generation == 0U ||
      schema_envelope_detail::load_little<std::uint32_t>(wire, 0U) != kSchemaEnvelopeABIVersion ||
      schema_envelope_detail::load_little<std::uint32_t>(wire, 4U) != kSchemaEnvelopeHeaderSize ||
      !std::equal(
          expected.family_id.value.begin(), expected.family_id.value.end(), wire.begin() + 8U) ||
      schema_envelope_detail::load_little<std::uint32_t>(wire, 28U) != 0U ||
      schema_envelope_detail::load_little<std::uint64_t>(wire, 32U) != expected_generation) {
    return make_unexpected(Error{ErrorCode::kProtocolError, "ENDPOINT_SCHEMA_ENVELOPE_INVALID"});
  }
  const std::size_t payload_size = schema_envelope_detail::load_little<std::uint32_t>(wire, 24U);
  if (payload_size == 0U || payload_size > payload_capacity ||
      payload_size > wire.size() - kSchemaEnvelopeHeaderSize) {
    return make_unexpected(
        Error{ErrorCode::kProtocolError, "ENDPOINT_SCHEMA_PAYLOAD_SIZE_INVALID"});
  }
  return payload_size;
}

}  // namespace holoscan::channel_internal
