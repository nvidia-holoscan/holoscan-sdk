/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <type_traits>

namespace holomq::fd_pass {

/// Wire format is native-endian (same host, AF_UNIX).
constexpr std::uint32_t k_magic = 0x484F4950U;  // "HOIP"

/// Bumped when the \ref Request wire layout changes (must match on client and server).
constexpr std::uint32_t k_version = 1U;

/// Wire status codes for \ref Response::status.
enum class Status : std::uint32_t {
  ok = 0U,
  bad_magic = 1U,
  bad_version = 2U,
  internal_error = 3U,
  pool_unavailable = 4U,
  pool_mismatch = 5U,
};

/// Client-to-server message on the AF_UNIX FD-pass stream (native-endian; see file comment).
/// A single request type: obtain the publisher VMM pool FD matching \c pool_cookie.
struct Request {
  Request() noexcept = default;

  constexpr explicit Request(std::uint64_t pool_cookie) noexcept : pool_cookie(pool_cookie) {}

  /// Must equal \ref k_magic; otherwise the server replies with \ref Status::bad_magic and does not
  /// dispatch.
  std::uint32_t magic{k_magic};
  /// Protocol revision; must equal \ref k_version or the server replies with \ref
  /// Status::bad_version.
  std::uint32_t version{k_version};
  /// Opaque pool id (\ref detail::device::BufferHandleData::pool_cookie or \ref
  /// detail::host::BufferHandleData::gen_pool_cookie).
  std::uint64_t pool_cookie{};
};

/// Fixed-size body sent with \c sendmsg / \c recvmsg; optional \c SCM_RIGHTS carries the passed FD
/// when \c status is \ref Status::ok.
struct Response {
  /// \ref Status as \c uint32_t wire value.
  std::uint32_t status{};
};

static_assert(std::is_standard_layout_v<Request>);
static_assert(std::is_trivially_copyable_v<Request>);
static_assert(sizeof(Request) == 16U);
static_assert(std::is_standard_layout_v<Response>);
static_assert(std::is_trivially_copyable_v<Response>);
static_assert(sizeof(Response) == 4U);

}  // namespace holomq::fd_pass
