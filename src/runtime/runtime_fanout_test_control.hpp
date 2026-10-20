/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <optional>

#include <holoscan/core/port_types.hpp>

namespace holoscan::detail {

enum class RuntimeFanoutObservationPoint : std::uint8_t {
  kAdmissionCompleted,
  kReadyHintStarted,
  kReadyHintReturned,
};

/// Private, attempt-owned attribution. Callbacks must not block, allocate, or call Runtime. They
/// run after the attributed receive has released its locks, and report the sequence that receive
/// admitted. They may run concurrently on different threads (concurrent admission runs without the
/// channel scan lock). The observer and its context must outlive the joined session.
struct RuntimeFanoutTestControl {
  // Asserts concurrent emit-time admission, which Runtime uses whenever delivery inside emit()
  // serves a service; materialization rejects it when none is served. Concurrent admission:
  // - Admission relies on each recipient's transport mutex instead of the channel scan lock.
  // - A retirement retries only the blocked heads of the subscribers it released, also without
  //   that lock; flags are read under each subscriber's transport mutex.
  // - Admission and post-compute cuts skip heads known to be blocked. Complete retirement cuts,
  //   terminal cuts, and pump scans keep the lock and try every head.
  bool concurrent_admission{};
  // Turn off the default publication credit check, for A/B runs and for tests of the HoloMQ
  // partial and indeterminate outcomes it otherwise prevents.
  bool disable_publication_credits{};
  // Leave a pump-served subscriber whose staged transport head is blocked visible to the pump's
  // wait, without the readiness mask or the release wake, for A/B runs.
  bool disable_blocked_head_wake{};
  // Replace the pump's spin before it hides a blocked head, for tests and tuning runs.
  std::optional<std::chrono::nanoseconds> blocked_head_spin;
  void* context{};
  void (*observe)(void*, PortId, std::uint64_t, RuntimeFanoutObservationPoint) noexcept {};
};

}  // namespace holoscan::detail
