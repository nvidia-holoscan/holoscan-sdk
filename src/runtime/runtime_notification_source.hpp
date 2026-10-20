/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <span>

#include <holoscan/core/expected.hpp>
#include <holoscan/core/readiness_source.hpp>
#include <holoscan/threading/thread_placement.hpp>
#include <holoscan/time/clock_registry.hpp>

#include "port_binding.hpp"

namespace holoscan::detail {

/// @brief Describe one process-local notification source before lifecycle entry.
///
/// The registry copies the descriptor during cold construction. The clock and wake callback state
/// are borrowed and must remain valid until the registry is closed and destroyed.
struct RuntimeNotificationSourceDescriptor {
  /// Exact operator authorized to acquire sender authority.
  VertexId owner{};
  /// Frozen physical source request.
  RuntimePortBindingRequest request;
  /// Compiler-admitted wake requirement.
  ReadinessWakeRequirement wake_requirement{ReadinessWakeRequirement::kDeploymentSelected};
  WakeMode wake_mode{WakeMode::kSleepingWaitset};  ///< Physical waiting mode.
  std::uint32_t sender_reference_capacity{};       ///< Finite sender-authority record count.
  ClockRef clock;                   ///< Borrowed retained clock used to timestamp posts.
  void* wake_state{};               ///< Optional borrowed state passed to \ref wake.
  void (*wake)(void*) noexcept {};  ///< Nonblocking controller-wake callback.
};

/// @brief Own bounded notification readiness and sender tombstones for one run.
///
/// The registry owns physical notification sources and finite sender records. It borrows the clocks
/// and wake callback state supplied by its descriptors. The owning run must call \ref close before
/// those borrowed objects retire.
class RuntimeNotificationSourceRegistry {
 public:
  RuntimeNotificationSourceRegistry(const RuntimeNotificationSourceRegistry&) = delete;
  RuntimeNotificationSourceRegistry& operator=(const RuntimeNotificationSourceRegistry&) = delete;
  RuntimeNotificationSourceRegistry(RuntimeNotificationSourceRegistry&&) = delete;
  RuntimeNotificationSourceRegistry& operator=(RuntimeNotificationSourceRegistry&&) = delete;
  ~RuntimeNotificationSourceRegistry();

  /// @brief Construct a closed-to-publication registry from frozen source descriptors.
  /// @param descriptors Complete nonempty source inventory copied by the registry.
  /// @return A registry, or a typed validation, capacity, or physical-source setup error.
  /// @throws std::length_error if descriptor storage exceeds the implementation maximum.
  [[nodiscard]] static expected<std::unique_ptr<RuntimeNotificationSourceRegistry>, Error> create(
      std::span<const RuntimeNotificationSourceDescriptor> descriptors);

  /// @brief Bind one prevalidated physical source before its owner enters the Arm stage.
  /// @param request Exact plan-bound request identifying the source and initial generation.
  /// @param start_coordinate Initial source-clock coordinate for physical readiness.
  /// @return An owning binding, or a typed closed, lookup, or physical-binding error.
  [[nodiscard]] expected<std::unique_ptr<PortBinding>, Error> bind(
      const RuntimePortBindingRequest& request, TimePoint start_coordinate) noexcept;

  /// @brief Acquire one finite sender authority during an admitted Arm callback.
  /// @param owner Exact operator requesting the sender.
  /// @param source Declared notification source whose identity must match the frozen descriptor.
  /// @param stage Current lifecycle stage, which must be \ref LifecycleStage::kArm.
  /// @param phase Startup- or restart-forward transition that selects the sender generation.
  /// @return A sender, or a typed lifecycle, identity, state, or capacity error.
  [[nodiscard]] expected<NotificationSender, Error> acquire_sender(
      VertexId owner, NotificationSource& source, LifecycleStage stage,
      LifecycleTransitionPhase phase) noexcept;

  /// @brief Atomically activate all startup-generation senders after the global Start barrier.
  /// @return Success, or a typed closed, duplicate-arm, or incomplete-binding error. Failure leaves
  /// every startup sender inactive.
  [[nodiscard]] expected<void, Error> arm_startup() noexcept;

  /// @brief Fence current-generation sender ingress for selected restart owners.
  /// @param owners Complete restart closure whose matching sources become not-ready.
  void begin_restart(std::span<const VertexId> owners) noexcept;

  /// @brief Atomically publish replacement controls and stale prior-generation senders.
  /// @param owners Complete restart closure previously passed to \ref begin_restart.
  /// @return Success, or a typed closed, generation, or physical-control error. Failure leaves all
  /// selected sources fenced for a subsequent \ref rollback_restart.
  [[nodiscard]] expected<void, Error> publish_restart(std::span<const VertexId> owners) noexcept;

  /// @brief Restore old-generation senders and stale unpublished replacements.
  /// @param owners Complete restart closure to unwind after a failed restart fence or publication.
  void rollback_restart(std::span<const VertexId> owners) noexcept;

  /// @brief Close every sender and physical source before borrowed wake state retires.
  ///
  /// This operation is idempotent. It does not destroy borrowed descriptor state.
  void close() noexcept;

 private:
  struct Impl;
  explicit RuntimeNotificationSourceRegistry(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::detail
