/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "fd_pass/sockaddr.hpp"

#include <cstddef>
#include <cstring>
#include <string>

namespace holomq::fd_pass {

bool fill_sockaddr_un(const std::string& endpoint, ::sockaddr_un* addr,
                      ::socklen_t* out_len) noexcept {
  if (endpoint.empty() || addr == nullptr || out_len == nullptr) {
    return false;
  }
  if (endpoint.size() > endpoint_max_length()) {
    return false;
  }

  std::memset(addr, 0, sizeof(*addr));
  addr->sun_family = AF_UNIX;

#if HOLOMQ_FD_PASS_USE_ABSTRACT
  addr->sun_path[0] = '\0';
  std::memcpy(addr->sun_path + 1, endpoint.data(), endpoint.size());
  *out_len = static_cast<::socklen_t>(offsetof(::sockaddr_un, sun_path) + 1U + endpoint.size());
#else
  std::memcpy(addr->sun_path, endpoint.c_str(), endpoint.size() + 1U);
  *out_len = static_cast<::socklen_t>(offsetof(::sockaddr_un, sun_path) + endpoint.size() + 1U);
#endif
  return true;
}

}  // namespace holomq::fd_pass
