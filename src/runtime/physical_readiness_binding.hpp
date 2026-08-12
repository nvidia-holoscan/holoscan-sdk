/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/threading/thread_placement.hpp>

#include "port_binding.hpp"

namespace holoscan::detail {

struct PhysicalReadinessState;

/// Identify source-tree checkpoints in the physical readiness publication protocol.
enum class PhysicalReadinessPublicationCheckpoint : std::uint8_t {
  kPublisherEntered,
  kPositionClaimed,
  kCreditReserved,
  kReadyPublished,
};

/// Describe one cold, transport-neutral physical readiness binding.
///
/// The descriptor copies immutable plan facts and borrows only optional poll
/// and wake callbacks. It owns no file descriptor, CUDA object, HoloMQ object,
/// wait set, or other backend resource. The embedding adapter must keep
/// callback state alive until the binding has been closed and destroyed.
struct PhysicalReadinessBindingDescriptor {
  RuntimePortBindingRequest request;
  ReadinessWakeRequirement wake_requirement{ReadinessWakeRequirement::kDeploymentSelected};
  WakeMode wake_mode{WakeMode::kSleepingWaitset};
  std::uint32_t execution_lane{};

  /// Whether this source requires an admitted monitored-health binding.
  bool monitored_health_required{};
  /// Whether the immutable plan supplied that monitored-health binding.
  bool monitored_health_bound{};
  /// Frozen action for absent/stale monitored evidence when monitoring is required.
  EndpointClockHealthDisposition monitored_health_disposition{
      EndpointClockHealthDisposition::kNotApplicable};

  /// Borrowed polling state used only by readable or spin adapters.
  void* poll_state{};
  /// Observe the current readable level or spin predicate without blocking.
  expected<bool, Error> (*poll_asserted)(void*) noexcept {};

  /// Borrowed wake state notified after accepted ingress, close, or fatal failure.
  void* wake_state{};
  /// Wake the Runtime controller without transferring backend ownership.
  void (*wake)(void*) noexcept {};

  /// Source-tree-only state for deterministic publication interleaving tests.
  void* test_publish_state{};
  /// Source-tree-only publication checkpoint callback; production descriptors leave this null.
  void (*test_publish_checkpoint)(void*, PhysicalReadinessPublicationCheckpoint) noexcept {};
};

/// Report one doorbell observation without exposing a device handle.
struct PhysicalDoorbellAcceptance {
  bool witness_accepted{};
  bool coalesced{};
  std::uint64_t previous_token{};
  std::uint64_t accepted_token{};
  std::uint64_t missed_count{};
};

/// Control one current-generation physical readiness adapter.
///
/// This private value owns only bounded Core ingress state. It may be handed to
/// backend glue, but no method blocks, allocates, invokes user code, or grants
/// ownership of a backend object. Controls are generation-scoped; a normal
/// close and successful restart invalidation remain distinct.
class PhysicalReadinessControl {
 public:
  PhysicalReadinessControl() noexcept = default;

  [[nodiscard]] expected<void, Error> observe_readable(bool readable,
                                                       TimePoint observed_at) const noexcept;
  [[nodiscard]] expected<void, Error> post_notification(TimePoint accepted_at) const noexcept;
  [[nodiscard]] expected<PhysicalDoorbellAcceptance, Error> ring_doorbell(
      std::uint64_t token, TimePoint accepted_at) const noexcept;
  [[nodiscard]] expected<void, Error> observe_spin(bool asserted,
                                                   TimePoint observed_at) const noexcept;

  /// Poll one admitted readable/spin predicate on its compiler-selected lane.
  [[nodiscard]] expected<void, Error> poll_once(TimePoint observed_at) const noexcept;

  /// Return one accepted-through-retirement credit after Dispatcher retirement.
  [[nodiscard]] expected<void, Error> retire(std::uint64_t source_sequence) const noexcept;

  /// Close normal acceptance and cancel queued, not-yet-drained witnesses.
  void close() const noexcept;
  /// Invalidate this generation after a successful lifecycle restart.
  void invalidate_stale() const noexcept;
  /// Close the source with one typed fatal backend error and wake Runtime.
  void fail(ErrorCode error) const noexcept;

  [[nodiscard]] explicit operator bool() const noexcept { return state_ != nullptr; }
  [[nodiscard]] std::uint64_t generation() const noexcept { return source_generation_; }
  [[nodiscard]] std::uint32_t execution_lane() const noexcept;
  [[nodiscard]] WakeMode wake_mode() const noexcept;
  [[nodiscard]] std::uint64_t wake_epoch() const noexcept;
  [[nodiscard]] std::size_t outstanding() const noexcept;

 private:
  PhysicalReadinessControl(std::shared_ptr<PhysicalReadinessState> state,
                           PlanGenerationId plan_generation, RunGenerationId run_generation,
                           std::uint64_t source_generation) noexcept;

  [[nodiscard]] expected<std::shared_ptr<PhysicalReadinessState>, Error> checked_state(
      RuntimeEventKind expected_kind) const noexcept;

  std::shared_ptr<PhysicalReadinessState> state_;
  PlanGenerationId plan_generation_{};
  RunGenerationId run_generation_{};
  std::uint64_t source_generation_{};

  friend class PhysicalReadinessBindingFactory;
};

/// Bind fixed immutable-plan readiness descriptors to bounded Core adapters.
///
/// Creation validates every descriptor, duplicate source, monitored-health
/// requirement, and busy-poll placement before allocating receiver state.
/// `bind()` then requires an exact request match and permits one binding per
/// descriptor. Missing or mismatched descriptors fail closed.
class PhysicalReadinessBindingFactory final : public RuntimePortBindingFactory {
 public:
  PhysicalReadinessBindingFactory(const PhysicalReadinessBindingFactory&) = delete;
  PhysicalReadinessBindingFactory& operator=(const PhysicalReadinessBindingFactory&) = delete;
  PhysicalReadinessBindingFactory(PhysicalReadinessBindingFactory&&) = delete;
  PhysicalReadinessBindingFactory& operator=(PhysicalReadinessBindingFactory&&) = delete;
  ~PhysicalReadinessBindingFactory() override;

  [[nodiscard]] static expected<std::shared_ptr<PhysicalReadinessBindingFactory>, Error> create(
      std::span<const PhysicalReadinessBindingDescriptor> descriptors);

  [[nodiscard]] expected<std::unique_ptr<PortBinding>, Error> bind(
      const RuntimePortBindingRequest& request, TimePoint start_coordinate) noexcept override;

  /// Resolve the control only after the exact plan, run, and source generation is bound.
  [[nodiscard]] expected<PhysicalReadinessControl, Error> control(
      SourceId source, PlanGenerationId plan_generation, RunGenerationId run_generation,
      std::uint64_t source_generation) const noexcept;

  /// Close every bound generation without touching backend ownership.
  void close_all() noexcept;

 private:
  struct Slot;
  // Slots are frozen at compile admission and retain stable addresses for the complete run.
  using SlotStorage = std::unique_ptr<Slot[]>;  // NOLINT(cppcoreguidelines-avoid-c-arrays)

  explicit PhysicalReadinessBindingFactory(SlotStorage slots, std::size_t slot_count) noexcept;

  SlotStorage slots_;
  std::size_t slot_count_{};
};

}  // namespace holoscan::detail
