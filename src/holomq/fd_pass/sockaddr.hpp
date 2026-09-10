/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <string>

#include <sys/socket.h>
#include <sys/un.h>

#ifndef HOLOMQ_FD_PASS_USE_ABSTRACT
#define HOLOMQ_FD_PASS_USE_ABSTRACT 0
#endif

namespace holomq::fd_pass {

/// \c socket_path_field_max() is the char-array capacity in serializable handle payloads
/// (\ref detail::device::BufferHandleData::fd_pass_socket_path_max / \ref
/// detail::host::BufferHandleData::fd_pass_socket_path_max).
/// \c endpoint_max_length() is the maximum endpoint string length accepted by \ref fill_sockaddr_un
/// (PATH or ABSTRACT encoding). It is never greater than \c socket_path_field_max() - 1U.

/// Fixed byte capacity of wire-format \c fd_pass_socket_path fields in handle payloads.
[[nodiscard]] constexpr std::size_t socket_path_field_max() noexcept {
  return sizeof(::sockaddr_un::sun_path);
}

/// Maximum length of the \c std::string endpoint passed to \ref fill_sockaddr_un (bytes).
[[nodiscard]] constexpr std::size_t endpoint_max_length() noexcept {
#if HOLOMQ_FD_PASS_USE_ABSTRACT
  // Leading NUL marks abstract; name follows in sun_path[1..].
  return sizeof(::sockaddr_un::sun_path) - 2U;
#else
  return sizeof(::sockaddr_un::sun_path) - 1U;
#endif
}

/// Filesystem pathname mode uses \c unlink(2) before \c bind(2) when reusing a path; abstract mode
/// does not.
[[nodiscard]] constexpr bool endpoint_uses_filesystem_path() noexcept {
#if HOLOMQ_FD_PASS_USE_ABSTRACT
  return false;
#else
  return true;
#endif
}

/// Fills \p addr for \c bind(2) / \c connect(2). Returns \c false if \p endpoint is empty or too
/// long.
[[nodiscard]] bool fill_sockaddr_un(const std::string& endpoint, ::sockaddr_un* addr,
                                    ::socklen_t* out_len) noexcept;

static_assert(endpoint_max_length() <= socket_path_field_max() - 1U);

}  // namespace holomq::fd_pass
