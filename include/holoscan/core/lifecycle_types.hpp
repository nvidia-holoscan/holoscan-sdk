/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <string_view>
#include <type_traits>

namespace holoscan {

/// @brief Identify one global operator lifecycle barrier.
///
/// The first six values are forward stages. \c kStop and \c kRelease are
/// teardown stages and are never valid restart \c from_stage values.
enum class LifecycleStage : std::uint8_t {
  kConfigure = 0,  ///< Validate immutable configuration and derive cold intent.
  kDiscover = 1,   ///< Discover external capabilities without publishing readiness.
  kAllocate = 2,   ///< Reserve admitted finite runtime resources.
  kWarmUp = 3,     ///< Prepare resources without making the operator activatable.
  kArm = 4,        ///< Bind current-generation external and readiness capabilities.
  kStart = 5,      ///< Publish the operator as ready for steady-state activation.
  kStop = 6,       ///< Close activation acceptance and quiesce steady-state work.
  kRelease = 7,    ///< Release resources owned by the active lifecycle epoch.
};

/// @brief Return whether a value names one lifecycle stage in the closed vocabulary.
/// @param stage Candidate lifecycle stage.
/// @return \c true only for \c kConfigure through \c kRelease.
[[nodiscard]] constexpr bool is_valid_lifecycle_stage(LifecycleStage stage) noexcept {
  return static_cast<std::uint8_t>(stage) <= static_cast<std::uint8_t>(LifecycleStage::kRelease);
}

/// @brief Report the bounded disposition of one lifecycle callback.
enum class LifecycleStatus : std::uint8_t {
  kOk = 0,                ///< Stage completed successfully.
  kSkipped = 1,           ///< Optional stage had no work for this operator.
  kRetryableFailure = 2,  ///< Stage failed with an explicitly retryable cold-path cause.
  kFatalFailure = 3,      ///< Stage failed and cannot continue the run.
};

/// @brief Identify which transition phase owns a lifecycle callback.
enum class LifecycleTransitionPhase : std::uint8_t {
  kNone = 0,                ///< Event or operation outside a lifecycle callback phase.
  kStartupForward = 1,      ///< Initial forward stages using generation zero.
  kPreRestartTeardown = 2,  ///< Teardown of the old active epoch before restart.
  kRestartForward = 3,      ///< Forward suffix using candidate or retained current state.
  kForwardRollback = 4,     ///< Unwind successful forward work after a primary failure.
  kRestoreOld = 5,          ///< Restore or re-enter the old published lifecycle state.
  kShutdownTeardown = 6,    ///< Final teardown of the active published epoch.
};

/// @brief Return whether a lifecycle stage is a valid forward restart entry.
/// @param stage Candidate lifecycle stage.
/// @return \c true only for \c kConfigure through \c kStart.
[[nodiscard]] constexpr bool is_lifecycle_forward_stage(LifecycleStage stage) noexcept {
  return static_cast<std::uint8_t>(stage) <= static_cast<std::uint8_t>(LifecycleStage::kStart);
}

/// @brief Name one completed ordering fact in a lifecycle-stage DAG.
///
/// This value carries only the stable-name component of an ordering fact.
/// The surrounding lifecycle-builder operation supplies the graph-instance
/// scope and lifecycle stage. The key grants no typed object access,
/// ownership, borrowing, or lifetime authority on its own.
struct LifecycleDependencyKey {
  std::string_view stable_name{};  ///< Qualified name copied before the builder call returns.
};

/// @brief Represent a registered fixed-size audit reason; zero is unspecified.
using ReasonId = std::uint64_t;

/// @brief Supply a registered bounded reason for one restart request.
struct LifecycleRestartReason {
  ReasonId stable_id{};  ///< Registered reason identity; zero means unspecified.
};

/// @brief Select the requested first forward lifecycle stage of a restart.
struct LifecycleRestartOptions {
  /// Requested earliest invalidated and first forward stage. Only
  /// \c kConfigure through \c kStart are valid request values.
  LifecycleStage from_stage{LifecycleStage::kConfigure};
  LifecycleRestartReason reason{};  ///< Registered audit reason with no hot-path text.
};

/// @brief Identify one lifecycle transaction within a run session.
using LifecycleTransactionId = std::uint64_t;

/// @brief Identify one callback declaration slot within an Operator lifecycle contract.
///
/// baseline uses the unique stage slot as semantic identity. It is pointer-free and portable, but
/// it is not a compiler-specific member-pointer encoding or a standalone binary fingerprint.
using LifecycleMethodId = std::uint8_t;

/// @brief Identify one published lifecycle epoch for an Operator.
using LifecycleEpoch = std::uint64_t;

/// @brief Identify one atomically installed activation fence.
using LifecycleFenceId = std::uint64_t;

/// @brief Report one bounded lifecycle transaction disposition.
enum class LifecycleTransactionState : std::uint8_t {
  kRejected = 0,          ///< Validation rejected before acceptance; no fence or receipt exists.
  kAcceptedNoChange = 1,  ///< Equivalent parameter-only candidate terminated without execution.
  kQueued = 2,            ///< Accepted, fenced, and still open to compatible coalescing.
  kQuiescing = 3,         ///< Coalescing cutoff passed while affected work retires.
  kRestarting = 4,        ///< Teardown or forward lifecycle callbacks are executing.
  kPublished = 5,         ///< Successful \c kStart published the new lifecycle epoch.
  kSuperseded = 6,        ///< A later complete candidate replaced this submission before publish.
  kCanceled = 7,          ///< Stop or activation failure won before publication.
  kRolledBack = 8,        ///< Forward failure restored the prior published epoch.
  kFailedStopped = 9,     ///< Recovery could not restore a coherent active epoch.
};

/// @brief Record the bounded stage decision for one accepted lifecycle transaction.
struct LifecycleStageDecision {
  LifecycleTransactionId transaction{};  ///< Nonzero run-local transaction identity.
  LifecycleStage requested{LifecycleStage::kConfigure};  ///< Earliest stage requested by ingress.
  LifecycleStage effective{LifecycleStage::kConfigure};  ///< Earliest compiler/executor stage used.
  ReasonId widening_reason{};  ///< Stable reason when \ref effective is earlier than requested.
};

static_assert(std::is_standard_layout_v<LifecycleRestartOptions>);
static_assert(std::is_trivially_copyable_v<LifecycleRestartOptions>);
static_assert(std::is_standard_layout_v<LifecycleStageDecision>);
static_assert(std::is_trivially_copyable_v<LifecycleStageDecision>);

}  // namespace holoscan
