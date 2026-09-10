/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>
#include <cstdint>
#include <utility>

#include <holoscan/core/lifecycle.hpp>

namespace holoscan::sensor_io {

/// @brief Constrain a stage body to the non-throwing status-returning shape Core requires.
///
/// Lifecycle callbacks are \c noexcept and return \ref LifecycleStatus. Requiring the same shape
/// of the guarded body keeps the guard from being the place where an exception escapes.
///
/// @tparam Fn Candidate stage body.
template <typename Fn>
concept StageBody = requires(Fn body) {
  { body() } noexcept -> std::same_as<LifecycleStatus>;
};  // NOLINT(readability/braces): cpplint misparses a concept requirement-body terminator.

/// @brief Withhold a sensor's teardown bodies from resources its forward bodies never took.
///
/// Core decides whether to invoke an operator's teardown stages from a completion ledger, and that
/// ledger does not distinguish a stage that did work from one that had none. \c callback_succeeded
/// admits both \c kOk and \c kSkipped, \c run_forward_stage records a completion on that basis, and
/// \c run_planned_teardown then gates \c kRelease on a recorded \c kAllocate and \c kStop on a
/// recorded \c kArm or \c kStart (\c public/src/core/lifecycle_kernel.cpp). An operator whose
/// \c kAllocate reserved nothing and correctly said so is therefore still asked to release. For
/// most operators that is harmless. For one holding mapped device buffers, the release body runs
/// against handles its allocate body never filled in, or ones a previous epoch already returned.
///
/// The operator cannot consult that ledger to defend itself. \ref LifecycleContext offers
/// cancellation, parameters, the parameter generation, the transition phase, clock resolution, and
/// restart-target binding, and nothing that reports what an earlier stage of the same operator
/// returned.
///
/// This object keeps that one missing fact where the operator can reach it: which of its own
/// forward bodies actually took something. Only \ref LifecycleStatus::kOk claims ownership, because
/// a body that reports \c kSkipped states that it created no work. It states nothing about what an
/// earlier epoch took, so a forward body that reports anything other than \c kOk leaves the record
/// as it found it: acquiring and relinquishing are separate events, and only the stages listed
/// below relinquish. \c kStop is then withheld from a device that was never armed, and \c kRelease
/// from buffers that were never reserved. A withheld body is *not invoked* and reports \ref
/// LifecycleStatus::kSkipped, which Core treats as a distinct success rather than a failure, and
/// which \ref guard_skipped distinguishes afterwards from a body that ran and chose to skip its own
/// work.
///
/// Forward bodies are never withheld. Core's forward loop returns on the first stage that fails and
/// rolls back without entering any later stage, so an operator's \c kArm body is never invoked
/// after its own \c kAllocate body failed, and a gate protecting against that ordering would guard
/// a sequence the runtime does not produce. It would, however, withhold real work: the lifecycle
/// guide's recommended shape is to register a stage only where the operator genuinely has work,
/// commonly just a matched \c kAllocate and \c kRelease pair, and gating each forward stage on a
/// record written by an earlier one silently withholds every body of an operator that registered
/// sparsely. Core would read the resulting run as a success in which the sensor never started.
///
/// Ownership is cleared by the stage that hands the resource back, not by the stage that stops
/// using it:
///
/// - A successful \c kStop clears only the start record. An \c kArm body's acquisitions outlive a
///   stop, so the shutdown \c kStop that follows a restart's \c kStop must still reach its body.
///   The same contract binds the bodies this object guards: an \c kArm body must not establish
///   anything its \c kStop body tears down, or Core's retained \c kArm checkpoint outlives the
///   state it stands for and the minimal restart suffix restores nothing.
/// - A successful \c kRelease clears the arm and start records as well as its own. Released
///   resources cannot still be armed or running, whatever an earlier \c kStop reported, so a
///   release that follows a failed stop cannot leave a freed device describing itself as armed.
///
/// \c kRelease is gated on ownership alone and not on the device having stopped cleanly. Teardown
/// must remain reachable after teardown fails, or a failed \c kStop would strand the buffers it was
/// trying to quiesce: the one stage that can hand them back would be permanently unreachable, and
/// the failure that most needs reclamation would be the failure that prevents it. A release body
/// that needs to behave differently when the device never stopped can inspect \ref armed and
/// \ref started, which still hold their pre-release values while it runs.
///
/// A \c kConfigure body resets the record first. Core reaches \c kConfigure only through a suffix
/// that already ran \c kRelease, so this is a backstop for an operator that registers no release
/// body rather than the ordinary path for clearing ownership.
///
/// The object is not thread-safe and does not need to be: Core quiesces an operator's callbacks
/// against each other. It allocates nothing and performs no I/O.
class StageGuards {
 public:
  /// @brief Run one stage body and record what it took, withholding teardown that owns nothing.
  /// @tparam kStage Lifecycle stage owning \p body.
  /// @tparam Fn Non-throwing body returning \ref LifecycleStatus.
  /// @param body Stage work to attempt.
  /// @return The body's own status, or \ref LifecycleStatus::kSkipped when \p kStage is a teardown
  /// stage with nothing recorded to tear down, in which case \p body was never invoked.
  /// @note Only \c kStop and \c kRelease are ever withheld. A forward body always runs, because
  /// Core does not enter a forward stage whose predecessor failed.
  /// @note A withheld body is recorded and can be inspected afterwards with \ref guard_skipped,
  /// which distinguishes it from a body that ran and chose to skip its own work.
  template <LifecycleStage kStage, StageBody Fn>
  [[nodiscard]] LifecycleStatus run(Fn&& body) noexcept {
    if constexpr (kStage == LifecycleStage::kConfigure) {
      // Backstop reset, so an operator that registers no kRelease body does not carry a previous
      // epoch's ownership into a restart that re-enters here.
      *this = StageGuards{};
      return std::forward<Fn>(body)();
    } else if constexpr (kStage == LifecycleStage::kDiscover || kStage == LifecycleStage::kWarmUp) {
      // Neither stage is permitted to acquire anything a teardown body would hand back, so there
      // is nothing to record and nothing Core's ledger is coarse about.
      return std::forward<Fn>(body)();
    } else if constexpr (kStage == LifecycleStage::kAllocate) {
      const LifecycleStatus status = std::forward<Fn>(body)();
      if (status == LifecycleStatus::kOk) {
        allocated_ = true;
      }
      return status;
    } else if constexpr (kStage == LifecycleStage::kArm) {
      const LifecycleStatus status = std::forward<Fn>(body)();
      if (status == LifecycleStatus::kOk) {
        armed_ = true;
      }
      return status;
    } else if constexpr (kStage == LifecycleStage::kStart) {
      const LifecycleStatus status = std::forward<Fn>(body)();
      if (status == LifecycleStatus::kOk) {
        started_ = true;
      }
      return status;
    } else if constexpr (kStage == LifecycleStage::kStop) {
      // Core keeps an advanced kStop eligible after a completed kArm, so a sensor that armed but
      // never started still owes its device a quiesce.
      if (!armed_ && !started_) {
        return guard_skip(kStage);
      }
      const LifecycleStatus status = std::forward<Fn>(body)();
      if (succeeded(status)) {
        // Only the start record. An kArm body's acquisitions outlive a stop, so the shutdown kStop
        // that follows a restart's kStop must still reach its body. The arm record is invalidated
        // by kRelease, or by a later kArm body that runs, and not by a stop.
        started_ = false;
      }
      return status;
    } else {
      static_assert(kStage == LifecycleStage::kRelease, "unhandled lifecycle stage");
      // Deliberately gated on ownership alone, and not additionally on the device having been
      // quiesced. A failed kStop leaves armed_ and started_ set, and withholding kRelease on that
      // basis would strand the buffers kAllocate reserved: the one stage that can hand them back
      // would be permanently unreachable, and the failure that most needs reclamation would be the
      // failure that prevents it. A body that must know whether it is tearing down a device that
      // never stopped cleanly can ask, because the flags below are cleared only after it runs.
      if (!allocated_) {
        return guard_skip(kStage);
      }
      const LifecycleStatus status = std::forward<Fn>(body)();
      if (succeeded(status)) {
        allocated_ = false;
        // Released resources cannot still be armed or running, whatever kStop reported earlier.
        // Leaving these set after a failed kStop would let a later kStop run against freed buffers.
        armed_ = false;
        started_ = false;
      }
      return status;
    }
  }

  /// @brief Return whether a \c kAllocate body reported \c kOk and therefore owns resources.
  /// @note This is what gates \c kRelease. A \c kAllocate body that returned \c kSkipped reserved
  /// nothing, so this stays \c false even though Core records the checkpoint as complete.
  [[nodiscard]] bool allocated() const noexcept { return allocated_; }

  /// @brief Return whether an \c kArm body reported \c kOk and therefore owns device state.
  /// @note This is what gates \c kStop, together with \ref started. An \c kArm body that returned
  /// \c kSkipped armed nothing, so this stays \c false.
  /// @note A successful \c kStop does not clear this, because an arm body's acquisitions outlive a
  /// stop. Only \c kRelease or a \c kConfigure reset invalidates it. A re-entered \c kArm body does
  /// not: the restart suffix that re-enters \c kArm runs no \c kRelease, so a re-arm that reports
  /// \c kSkipped has performed no relinquishment and the previous epoch's acquisition is still
  /// outstanding. Clearing the record there would withhold the shutdown \c kStop that owes that
  /// device its quiesce.
  [[nodiscard]] bool armed() const noexcept { return armed_; }

  /// @brief Return whether a \c kStart body reported \c kOk and therefore owns running state.
  /// @note This is what gates \c kStop, together with \ref armed. A \c kStart body that returned
  /// \c kSkipped started nothing, so this stays \c false.
  /// @note Cleared by a successful \c kStop, by a successful \c kRelease, or by a \c kConfigure
  /// reset, and by nothing else. A re-entered \c kStart body does not clear it for the same reason
  /// a re-entered \c kArm body does not: reporting \c kSkipped is not a relinquishment. Core
  /// happens to run the \c kStop that does relinquish this before it re-enters \c kStart, so the
  /// record is already clear by then and the two readings agree on every sequence the runtime
  /// produces; the guard does not depend on that ordering to stay correct.
  [[nodiscard]] bool started() const noexcept { return started_; }

  /// @brief Return whether a stage body was withheld because its precondition was unmet.
  /// @param stage Stage to query.
  /// @return \c true when \ref run declined to invoke that stage's body since the last reset, and
  /// \c false for any value outside the closed stage vocabulary.
  /// @note This reports a guard decision, not the body's own \c kSkipped return.
  /// @note A value no stage names cannot have been withheld, so it answers \c false rather than
  /// shifting by it. Nothing in the vocabulary is ever withheld silently as a result: the widest
  /// valid stage still fits the record.
  [[nodiscard]] bool guard_skipped(LifecycleStage stage) const noexcept {
    if (!is_valid_lifecycle_stage(stage)) {
      return false;
    }
    return (guard_skips_ & stage_bit(stage)) != 0U;
  }

  /// @brief Return whether any stage body was withheld since the last reset.
  [[nodiscard]] bool any_guard_skipped() const noexcept { return guard_skips_ != 0U; }

 private:
  static constexpr bool succeeded(LifecycleStatus status) noexcept {
    return status == LifecycleStatus::kOk || status == LifecycleStatus::kSkipped;
  }

  // Callers must have established that the stage names a real one. Every stage in the vocabulary
  // shifts within a uint8_t, so the record needs no wider type; a value outside it would shift by
  // more than the promoted operand's width, which is why the public query rejects one first.
  static constexpr std::uint8_t stage_bit(LifecycleStage stage) noexcept {
    return static_cast<std::uint8_t>(1U << static_cast<std::uint8_t>(stage));
  }

  constexpr LifecycleStatus guard_skip(LifecycleStage stage) noexcept {
    guard_skips_ = static_cast<std::uint8_t>(guard_skips_ | stage_bit(stage));
    return LifecycleStatus::kSkipped;
  }

  bool allocated_{};
  bool armed_{};
  bool started_{};
  std::uint8_t guard_skips_{};
};

}  // namespace holoscan::sensor_io
