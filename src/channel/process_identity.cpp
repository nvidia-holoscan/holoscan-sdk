/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "process_identity.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace holoscan::channel_internal {

std::uint64_t current_process_identity() noexcept {
#if defined(_WIN32)
  return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
  return static_cast<std::uint64_t>(::getpid());
#endif
}

std::uint64_t mint_producer_generation() noexcept {
  static std::atomic<std::uint64_t> counter{0U};
  const auto system_now =
      static_cast<std::uint64_t>(std::chrono::system_clock::now().time_since_epoch().count());
  const auto steady_now =
      static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
  const auto process_address =
      static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(&counter));
  std::uint64_t token =
      system_now ^ (steady_now + 0x9E3779B97F4A7C15ULL) ^
      (current_process_identity() * 0xBF58476D1CE4E5B9ULL) ^ process_address ^
      ((counter.fetch_add(1U, std::memory_order_relaxed) + 1U) * 0x94D049BB133111EBULL);
  token = (token ^ (token >> 30U)) * 0xBF58476D1CE4E5B9ULL;
  token = (token ^ (token >> 27U)) * 0x94D049BB133111EBULL;
  token ^= token >> 31U;
  return token == 0U ? 1U : token;
}

}  // namespace holoscan::channel_internal
