/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <span>
#include <type_traits>

namespace holomq {

/// @brief Payload tag that selects a bounded byte-slice pub/sub service.
///
/// A byte-slice service carries frames of runtime length up to the capacity fixed by
/// \ref PubSubServiceOptions::byte_slice_capacity. Every publisher preallocates its sample pool for
/// that capacity, so a shorter frame still occupies a full-capacity slot. \c Publisher<ByteSlice>
/// loans a frame of an explicit length, and received frames expose exactly the published bytes.
///
/// Frame bytes start at an address aligned to \ref kByteSliceAlignment.
struct ByteSlice final {};

/// Alignment of the first byte of every byte-slice frame.
inline constexpr std::size_t kByteSliceAlignment = alignof(std::max_align_t);

/// @brief Return whether \p Payload is the byte-slice tag.
template <typename Payload>
inline constexpr bool kIsByteSlice = std::is_same_v<Payload, ByteSlice>;

/// @brief Mutable view of a loaned payload: \c Payload& or the loaned frame bytes.
template <typename Payload>
using MutablePayloadView =
    std::conditional_t<kIsByteSlice<Payload>, std::span<std::byte>, Payload&>;

/// @brief Read-only view of a payload: \c const Payload& or the frame bytes.
template <typename Payload>
using ConstPayloadView =
    std::conditional_t<kIsByteSlice<Payload>, std::span<const std::byte>, const Payload&>;

}  // namespace holomq
