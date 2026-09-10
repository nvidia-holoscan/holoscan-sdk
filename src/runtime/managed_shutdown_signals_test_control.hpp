/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <cstdint>

namespace holoscan::detail {

/// Source-tree-only one-shot fault points for the managed-signal transaction.
enum class RuntimeManagedSignalTestPoint : std::uint8_t {
  kNone = 0,
  kReserveDomainGeneration,
  kCensusBeforeMask,
  kSnapshotMask,
  kBlockMask,
  kCensusAfterMask,
  kQueryDisposition,
  kMaskBlockedBeforeSignalFd,
  kCreateSignalFd,
  kCreateNotification,
  kStartBroker,
  kOpenBrokerTask,
  kReserveBindingGeneration,
  kAcquireBinding,
  kNotifyController,
  kDrainNotification,
  kConsumeFixedState,
  kPoll,
  kBoundWaitForActivity,
  kUnboundWaitForActivity,
  kReadSignal,
  kFinalDrain,
  kCloseSignalFd,
  kCloseNotification,
  kRestoreMask,
  kTerminalRelease,
  kDelegateSignal,
  kJoinBroker,
  kWaitBrokerRetirement,
  kCloseDomain,
  kWorkerStart,
  kRegistryLockedBeforeBinding,
  kBindingPublishedBeforeAttach,
  kCandidatePublishedBeforeWake,
  kCandidateReadyBeforeFinalDrain,
  kBeforeControlFdClose,
  kBrokerReadyForDirectedSignal,
};

using RuntimeManagedSignalTestCheckpointHook = void (*)(RuntimeManagedSignalTestPoint,
                                                        void*) noexcept;

/// Inject one bounded OS-style failure without entering any production plan identity.
class RuntimeManagedSignalTestControl final {
 public:
  RuntimeManagedSignalTestControl(const RuntimeManagedSignalTestPoint point,
                                  const int os_error) noexcept
      : first_point_(point), first_os_error_(os_error) {}

  RuntimeManagedSignalTestControl(const RuntimeManagedSignalTestPoint point, const int os_error,
                                  const std::uint32_t fail_on_hit) noexcept
      : first_point_(point), first_os_error_(os_error), first_fail_on_hit_(fail_on_hit) {}

  RuntimeManagedSignalTestControl(const RuntimeManagedSignalTestPoint first_point,
                                  const int first_os_error,
                                  const RuntimeManagedSignalTestPoint second_point,
                                  const int second_os_error) noexcept
      : first_point_(first_point),
        first_os_error_(first_os_error),
        second_point_(second_point),
        second_os_error_(second_os_error) {}

  RuntimeManagedSignalTestControl(const RuntimeManagedSignalTestPoint fault_point,
                                  const int os_error,
                                  const RuntimeManagedSignalTestPoint checkpoint_point,
                                  RuntimeManagedSignalTestCheckpointHook hook,
                                  void* hook_state) noexcept
      : first_point_(fault_point),
        first_os_error_(os_error),
        checkpoint_point_(checkpoint_point),
        checkpoint_hook_(hook),
        checkpoint_state_(hook_state) {}

  RuntimeManagedSignalTestControl(const RuntimeManagedSignalTestPoint fault_point,
                                  const int os_error, const std::uint32_t fail_on_hit,
                                  const RuntimeManagedSignalTestPoint checkpoint_point,
                                  RuntimeManagedSignalTestCheckpointHook hook,
                                  void* hook_state) noexcept
      : first_point_(fault_point),
        first_os_error_(os_error),
        first_fail_on_hit_(fail_on_hit),
        checkpoint_point_(checkpoint_point),
        checkpoint_hook_(hook),
        checkpoint_state_(hook_state) {}

  RuntimeManagedSignalTestControl(const RuntimeManagedSignalTestPoint point,
                                  RuntimeManagedSignalTestCheckpointHook hook,
                                  void* hook_state) noexcept
      : checkpoint_point_(point), checkpoint_hook_(hook), checkpoint_state_(hook_state) {}

  RuntimeManagedSignalTestControl(const RuntimeManagedSignalTestPoint first_point,
                                  RuntimeManagedSignalTestCheckpointHook first_hook,
                                  void* first_state,
                                  const RuntimeManagedSignalTestPoint second_point,
                                  RuntimeManagedSignalTestCheckpointHook second_hook,
                                  void* second_state) noexcept
      : checkpoint_point_(first_point),
        checkpoint_hook_(first_hook),
        checkpoint_state_(first_state),
        second_checkpoint_point_(second_point),
        second_checkpoint_hook_(second_hook),
        second_checkpoint_state_(second_state) {}

  [[nodiscard]] int consume(const RuntimeManagedSignalTestPoint point) noexcept {
    if (point == first_point_) {
      const std::uint32_t visit = first_visits_.fetch_add(1U, std::memory_order_relaxed) + 1U;
      hits_.fetch_add(1U, std::memory_order_relaxed);
      if (visit == first_fail_on_hit_) {
        return first_os_error_;
      }
    }
    if (point == second_point_) {
      const std::uint32_t visit = second_visits_.fetch_add(1U, std::memory_order_relaxed) + 1U;
      hits_.fetch_add(1U, std::memory_order_relaxed);
      if (visit == second_fail_on_hit_) {
        return second_os_error_;
      }
    }
    return 0;
  }

  [[nodiscard]] std::uint32_t hits() const noexcept {
    return hits_.load(std::memory_order_relaxed);
  }

  void checkpoint(const RuntimeManagedSignalTestPoint point) noexcept {
    if (point == checkpoint_point_ && checkpoint_hook_ != nullptr) {
      bool expected{};
      if (checkpoint_consumed_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        hits_.fetch_add(1U, std::memory_order_relaxed);
        checkpoint_hook_(point, checkpoint_state_);
      }
    }
    if (point == second_checkpoint_point_ && second_checkpoint_hook_ != nullptr) {
      bool expected{};
      if (second_checkpoint_consumed_.compare_exchange_strong(
              expected, true, std::memory_order_acq_rel)) {
        hits_.fetch_add(1U, std::memory_order_relaxed);
        second_checkpoint_hook_(point, second_checkpoint_state_);
      }
    }
  }

  void seed_next_generation(const std::uint64_t generation) noexcept {
    next_generation_seed_.store(generation, std::memory_order_release);
  }

  [[nodiscard]] std::uint64_t next_generation_seed() const noexcept {
    return next_generation_seed_.load(std::memory_order_acquire);
  }

 private:
  RuntimeManagedSignalTestPoint first_point_{RuntimeManagedSignalTestPoint::kNone};
  int first_os_error_{};
  std::uint32_t first_fail_on_hit_{1U};
  std::atomic<std::uint32_t> first_visits_{};
  RuntimeManagedSignalTestPoint second_point_{RuntimeManagedSignalTestPoint::kNone};
  int second_os_error_{};
  std::uint32_t second_fail_on_hit_{1U};
  std::atomic<std::uint32_t> second_visits_{};
  RuntimeManagedSignalTestPoint checkpoint_point_{RuntimeManagedSignalTestPoint::kNone};
  std::atomic_bool checkpoint_consumed_{};
  RuntimeManagedSignalTestPoint second_checkpoint_point_{RuntimeManagedSignalTestPoint::kNone};
  std::atomic_bool second_checkpoint_consumed_{};
  std::atomic<std::uint32_t> hits_{};
  std::atomic<std::uint64_t> next_generation_seed_{};
  RuntimeManagedSignalTestCheckpointHook checkpoint_hook_{};
  void* checkpoint_state_{};
  RuntimeManagedSignalTestCheckpointHook second_checkpoint_hook_{};
  void* second_checkpoint_state_{};
};

}  // namespace holoscan::detail
