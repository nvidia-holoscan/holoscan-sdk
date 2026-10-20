/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// LD_PRELOAD shim that counts sched_yield() calls per thread for the control-loop benchmark.
//
// CONTROL_LOOP_YIELD_SHIM=pass (default) forwards every call; =noop returns 0 without yielding,
// which is the A/B the robotics team used to measure the cost of yielding in timed waits. Always
// compare noop with pass, never with no shim. The benchmark reads the counts through
// control_loop_yield_shim_snapshot() when the shim is loaded.

#include <dlfcn.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

constexpr std::size_t kSlots = 256U;

struct Slot {
  std::atomic<std::uint64_t> tid{};
  std::atomic<std::uint64_t> count{};
};

std::array<Slot, kSlots> g_slots;
__thread Slot* t_slot = nullptr;
__thread bool t_slot_claimed = false;

using YieldFunction = int (*)();

bool noop_mode() {
  static const bool noop = [] {
    const char* mode = std::getenv("CONTROL_LOOP_YIELD_SHIM");
    return mode != nullptr && std::strcmp(mode, "noop") == 0;
  }();
  return noop;
}

YieldFunction real_yield() {
  static const auto function = reinterpret_cast<YieldFunction>(dlsym(RTLD_NEXT, "sched_yield"));
  return function;
}

// Claim one slot per thread; threads beyond the table are still forwarded, just not counted.
Slot* thread_slot() {
  if (!t_slot_claimed) {
    t_slot_claimed = true;
    const auto tid = static_cast<std::uint64_t>(syscall(SYS_gettid));
    for (Slot& slot : g_slots) {
      std::uint64_t expected = 0U;
      if (slot.tid.compare_exchange_strong(expected, tid, std::memory_order_acq_rel)) {
        t_slot = &slot;
        break;
      }
    }
  }
  return t_slot;
}

}  // namespace

extern "C" {

__attribute__((visibility("default"))) int sched_yield() {
  if (Slot* slot = thread_slot()) {
    slot->count.fetch_add(1U, std::memory_order_relaxed);
  }
  if (noop_mode()) {
    return 0;
  }
  const YieldFunction function = real_yield();
  return function != nullptr ? function() : static_cast<int>(syscall(SYS_sched_yield));
}

// Copy up to capacity (tid, count) pairs of threads that have yielded; return the pair count.
__attribute__((visibility("default"))) std::size_t control_loop_yield_shim_snapshot(
    std::uint64_t* tids, std::uint64_t* counts, std::size_t capacity) {
  std::size_t written = 0U;
  for (const Slot& slot : g_slots) {
    const std::uint64_t tid = slot.tid.load(std::memory_order_acquire);
    if (tid == 0U || written == capacity) {
      continue;
    }
    tids[written] = tid;
    counts[written] = slot.count.load(std::memory_order_relaxed);
    ++written;
  }
  return written;
}

// Report whether calls are forwarded (0) or dropped (1).
__attribute__((visibility("default"))) int control_loop_yield_shim_noop() {
  return noop_mode() ? 1 : 0;
}

}  // extern "C"
