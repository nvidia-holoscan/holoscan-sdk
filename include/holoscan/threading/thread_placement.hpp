/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan {

/**
 * @brief Inclusive CPU core range used when building a CPUAffinity.
 */
struct CPUCoreRange {
  uint32_t first = 0;
  uint32_t last = 0;
};

/**
 * @brief Canonical CPU affinity request for a runtime thread.
 *
 * A default-constructed CPUAffinity has no cores and is invalid when present inside a
 * ThreadPlacement. That lets ThreadPlacement distinguish "no affinity requested" from "an affinity
 * request was provided but contained no CPUs."
 */
struct CPUAffinity {
  std::vector<uint32_t> cores;

  /**
   * @brief Construct an empty CPU affinity request.
   *
   * @return CPUAffinity with no CPU core IDs.
   * @throws None.
   */
  CPUAffinity() noexcept = default;

  /**
   * @brief Construct a CPU affinity request from explicit CPU core IDs.
   *
   * @param core_ids CPU core IDs to normalize into sorted, unique order.
   * @return CPUAffinity containing the normalized CPU core IDs.
   * @throws None.
   */
  HOLOSCAN_CORE_EXPORT explicit CPUAffinity(std::vector<uint32_t> core_ids);

  /**
   * @brief Construct a CPU affinity request from an initializer list of CPU core IDs.
   *
   * @param core_ids CPU core IDs to normalize into sorted, unique order.
   * @return CPUAffinity containing the normalized CPU core IDs.
   * @throws std::bad_alloc if storing or normalizing the core list requires allocation and
   * allocation fails.
   */
  HOLOSCAN_CORE_EXPORT CPUAffinity(std::initializer_list<uint32_t> core_ids);

  /**
   * @brief Build a CPU affinity request from inclusive CPU core ranges.
   *
   * @param ranges Inclusive CPU core ranges to expand and normalize.
   * @return expected<CPUAffinity, Error> containing CPUAffinity on success or Error on invalid
   * range ordering or excessive range expansion.
   * @throws None. Validation failures are returned in the expected result.
   */
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<CPUAffinity, Error> from_ranges(
      std::span<const CPUCoreRange> ranges);

  /**
   * @brief Build a CPU affinity request from initializer-list CPU core ranges.
   *
   * @param ranges Inclusive CPU core ranges to expand and normalize.
   * @return expected<CPUAffinity, Error> containing CPUAffinity on success or Error on invalid
   * range ordering or excessive range expansion.
   * @throws None. Validation failures are returned in the expected result.
   */
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<CPUAffinity, Error> from_ranges(
      std::initializer_list<CPUCoreRange> ranges);

  /**
   * @brief Build a CPU affinity request from explicit core IDs plus inclusive ranges.
   *
   * @param core_ids CPU core IDs to include in the affinity request.
   * @param ranges Inclusive CPU core ranges to expand and include in the affinity request.
   * @return expected<CPUAffinity, Error> containing CPUAffinity on success or Error on invalid
   * range ordering or excessive range expansion.
   * @throws None. Validation failures are returned in the expected result.
   */
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<CPUAffinity, Error> from_cores_and_ranges(
      std::span<const uint32_t> core_ids, std::span<const CPUCoreRange> ranges);

  /**
   * @brief Return whether this CPUAffinity contains no CPU core IDs.
   *
   * @return bool true when the affinity has no CPU core IDs.
   * @throws None.
   */
  [[nodiscard]] HOLOSCAN_CORE_EXPORT bool empty() const noexcept;

  /**
   * @brief Normalize CPU core IDs in place into sorted, unique order.
   *
   * @throws None.
   */
  HOLOSCAN_CORE_EXPORT void normalize();

  /**
   * @brief Validate that this CPUAffinity contains at least one CPU core ID.
   *
   * @return expected<void, Error> containing void on success or Error when the affinity is empty.
   * @throws None. Validation failures are returned in the expected result.
   */
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> validate() const;
};

/**
 * @brief Linux scheduling policy requested for a runtime thread.
 */
enum class SchedulingPolicy {
  kOther,
  kFIFO,
  kRoundRobin,
  kDeadline,
};

/// @brief Select how one Runtime lane waits for readiness.
enum class WakeMode : std::uint8_t {
  kSleepingWaitset = 0,  ///< Block in the admitted wait-set path when no work is ready.
  kBusyPoll = 1,         ///< Poll continuously on an explicitly admitted dedicated lane.
};

/// @brief Request deployment evidence that a lane's CPUs are isolated from unmanaged IRQ load.
enum class IRQIsolation : std::uint8_t {
  kNotRequested = 0,     ///< Do not require or prefer IRQ-isolation evidence.
  kPreferIsolated = 1,   ///< Record an explicit fallback when isolation cannot be proven.
  kRequireIsolated = 2,  ///< Reject admission when isolation cannot be proven.
};

/// @brief Select the admission outcome when a placement probe cannot prove its request.
enum class ProbeFailurePolicy : std::uint8_t {
  kWarnAndRecordFallback = 0,  ///< Preserve an explicit bounded fallback record.
  kFailAdmission = 1,          ///< Reject before Runtime creates the lane.
};

/**
 * @brief SCHED_DEADLINE parameters, expressed in nanoseconds.
 */
struct DeadlineScheduling {
  uint64_t runtime_ns = 0;
  uint64_t deadline_ns = 0;
  uint64_t period_ns = 0;

  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> validate() const;
};

/**
 * @brief Scheduling request for a runtime thread.
 */
struct ThreadScheduling {
  SchedulingPolicy policy = SchedulingPolicy::kOther;
  std::optional<int> priority;
  std::optional<DeadlineScheduling> deadline;

  /**
   * @brief Validate the SchedulingPolicy value, optional priority range, and optional
   * DeadlineScheduling fields.
   *
   * @return expected<void, Error> containing void on success or Error describing the invalid
   * SchedulingPolicy, priority, or DeadlineScheduling configuration.
   * @throws None. Validation failures are returned in the expected result.
   */
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> validate() const;
};

/// @brief Describe one complete deployment-time thread placement request.
///
/// This owning value contains only requested policy. Compilation validates the
/// value and may use a read-only platform probe to freeze separate requested and
/// accepted plan records. Constructing or validating this value never changes
/// the calling thread. The value is not thread-safe for concurrent mutation;
/// immutable copies may be read concurrently.
struct ThreadPlacement {
  std::optional<CPUAffinity> affinity;  ///< Requested CPU set; absence leaves affinity unchanged.
  ThreadScheduling scheduling;          ///< Requested scheduler policy and policy-specific fields.
  WakeMode wake_mode{WakeMode::kSleepingWaitset};           ///< Runtime readiness-wait strategy.
  IRQIsolation irq_isolation{IRQIsolation::kNotRequested};  ///< Requested isolation evidence.
  ProbeFailurePolicy probe_failure_policy{
      ProbeFailurePolicy::kWarnAndRecordFallback};  ///< Missing-proof admission behavior.

  /// @brief Validate every closed placement field without probing or applying it.
  ///
  /// @return An empty result on success, or \ref ErrorCode::kInvalidArgument
  /// for an empty present affinity, invalid scheduling tuple, or unknown closed
  /// enum value.
  /// @throws Nothing. Allocation failures are not possible in this validation.
  /// @note This method performs no OS query and does not mutate thread state.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> validate() const;
};

}  // namespace holoscan
