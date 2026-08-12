/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/buffer_handle.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace holomq_main {

/// Compile-time frame dimensions for the \c holomq_main demo (\c width × \c height × 3 lanes).
inline constexpr std::uint32_t k_frame_width = 640U;
inline constexpr std::uint32_t k_frame_height = 480U;

/// Demo pub/sub payload: sequence metadata, informational frame size, and an exported \ref
/// holomq::BufferHandle.
///
/// Subscribers import GPU-accessible memory via \c buffer_handle only; \c width and \c height are
/// not validated against the handle in the demo (set by the publisher for logging / future layout
/// checks).
struct FramePayload {
  std::uint64_t sequence{};
  std::uint32_t width{k_frame_width};
  std::uint32_t height{k_frame_height};
  holomq::BufferHandle buffer_handle{};
};

static_assert(std::is_standard_layout_v<FramePayload>);
static_assert(std::is_trivially_copyable_v<FramePayload>);

/// Byte size of one external \ref holomq::BufferPool slot for the demo frame (device or host;
/// \c k_frame_width × \c k_frame_height × 3 lanes × \c sizeof(std::uint64_t)).
inline constexpr std::size_t k_frame_buffer_bytes = static_cast<std::size_t>(k_frame_width) *
                                                    static_cast<std::size_t>(k_frame_height) * 3U *
                                                    sizeof(std::uint64_t);

}  // namespace holomq_main
