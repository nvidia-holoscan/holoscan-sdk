/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/byte_slice.hpp>

#include <iox2/bb/slice.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

namespace holomq::detail::transport {

/// iceoryx2 payload type of a HoloMQ payload: the payload itself, or a \c u8 slice for
/// \ref holomq::ByteSlice. A \c u8 slice keeps the service type name portable to Rust peers.
template <typename Payload>
using IoxPayload =
    std::conditional_t<holomq::kIsByteSlice<Payload>, iox2::bb::Slice<std::uint8_t>, Payload>;

/// Return a mutable byte view of a loaned iceoryx2 slice.
[[nodiscard]] inline std::span<std::byte> as_bytes(
    iox2::bb::MutableSlice<std::uint8_t> slice) noexcept {
  return {reinterpret_cast<std::byte*>(slice.data()), slice.number_of_elements()};
}

/// Return a read-only byte view of a received iceoryx2 slice.
[[nodiscard]] inline std::span<const std::byte> as_bytes(
    const iox2::bb::ImmutableSlice<std::uint8_t> slice) noexcept {
  return {reinterpret_cast<const std::byte*>(slice.data()), slice.number_of_elements()};
}

}  // namespace holomq::detail::transport
