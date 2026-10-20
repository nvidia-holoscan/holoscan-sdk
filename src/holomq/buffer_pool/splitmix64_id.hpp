/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/// \file splitmix64_id.hpp
/// Lock-free opaque id generation for holomq wire and cache keys.
///
/// Holomq assigns per-object ids that must be unique for the object's lifetime and must not
/// repeat in predictable order on the wire (for example \c shm_ref_count_id, \c pool_cookie).
/// These ids are **not** secrets; the goal is practical uniqueness and spread across the numeric
/// range, not cryptographic randomness.
///
/// ## Algorithm
///
/// Implements the **SplitMix64** finalizer (Steele & Vigna): a process-local atomic counter is
/// advanced, then mixed with xor/shift/multiply steps so successive calls do not produce \c 1,
/// \c 2, \c 3, …. The mixer constants below are the standard SplitMix64 values:
///
/// | Constant | Role |
/// |----------|------|
/// | \c 0x9E3779B97F4A7C15 | Golden-ratio increment; advances hidden state |
/// | \c 0xBF58476D1CE4E5B9, \c 0x94D049BB133111EB | Avalanche multipliers in the finalizer |
/// | Shifts 30 / 27 / 31 | Fold high bits back in after each multiply |
///
/// Hidden state is seeded once per process from \c getpid() and the state object address so
/// different processes (and runs) start in different parts of the sequence.
///
/// ## Why not \c std::random or \c /dev/urandom?
///
/// Call sites (\ref ShmRefCountMapping::create, \ref VmmExportedPool::create) are \c noexcept and
/// may run on hot paths. SplitMix64 is lock-free, allocation-free, and needs no syscall. Entropy
/// sources would require failure handling or would break \c noexcept.
///
/// ## Callers
///
/// - \c shm_ref_count_id (32-bit wire field; truncate \ref next_splitmix64_nonzero_id)
/// - \c pool_cookie / \c gen_pool_cookie (64-bit wire field; use full value)

#include <atomic>
#include <cstdint>

#include <unistd.h>

namespace holomq {
namespace detail {

/// Returns the next mixed 64-bit id. Never returns \c 0 (reserved as invalid on holomq wires).
[[nodiscard]] inline std::uint64_t next_splitmix64_nonzero_id() noexcept {
  static std::atomic<std::uint64_t> state{[]() noexcept -> std::uint64_t {
    const std::uint64_t seed = (static_cast<std::uint64_t>(::getpid()) << 32U) ^
                               static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(&state));
    return seed != 0U ? seed : 0x9E3779B97F4A7C15ULL;
  }()};

  std::uint64_t x = state.fetch_add(0x9E3779B97F4A7C15ULL, std::memory_order_relaxed);
  x ^= x >> 30U;
  x *= 0xBF58476D1CE4E5B9ULL;
  x ^= x >> 27U;
  x *= 0x94D049BB133111EBULL;
  x ^= x >> 31U;
  return x == 0U ? 1U : x;
}

}  // namespace detail
}  // namespace holomq
