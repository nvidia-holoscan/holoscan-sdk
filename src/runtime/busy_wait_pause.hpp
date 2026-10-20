/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>

#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
#include <intrin.h>
#endif

namespace holoscan::detail {

/// Pause the CPU briefly inside a spin without entering the kernel or yielding the CPU.
inline void busy_wait_pause() noexcept {
#if defined(__i386__) || defined(__x86_64__)
  __builtin_ia32_pause();
#elif defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
  _mm_pause();
#elif defined(__arm__) || defined(__aarch64__)
  __asm__ __volatile__("yield");
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

}  // namespace holoscan::detail
