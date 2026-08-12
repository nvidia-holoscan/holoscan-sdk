/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <type_traits>

namespace holoscan {

/// @brief Describe whether an endpoint can publish clock-health status reports.
enum class ClockHealthCapability : std::uint8_t {
  kUnavailable = 0,    ///< The endpoint exposes no clock-health report operation.
  kStatusReports = 1,  ///< The endpoint can publish ordered bounded status reports.
};

/// @brief Describe whether fresh healthy clock evidence gates nominal use.
enum class ClockHealthRequirement : std::uint8_t {
  kNotRequired = 0,         ///< Clock health does not gate the endpoint.
  kFreshHealthyReport = 1,  ///< A fresh healthy report is required by admitted policy.
};

/// @brief Classify one caller-supplied clock-health assessment.
enum class ClockHealthAssessment : std::uint8_t {
  kHealthy = 0,    ///< The monitored clock satisfies its admitted health contract.
  kDegraded = 1,   ///< The clock remains usable only under an admitted degraded policy.
  kUnhealthy = 2,  ///< The clock does not satisfy nominal use.
};

/// @brief Identify the portable primary reason for a clock-health assessment.
enum class ClockHealthReason : std::uint8_t {
  kNone = 0,                     ///< No negative condition; required for a healthy assessment.
  kSynchronizationLockLost = 1,  ///< External synchronization is no longer locked.
  kPhaseErrorLimitExceeded = 2,  ///< Observed phase error exceeded the admitted finite limit.
  kRateErrorLimitExceeded = 3,   ///< Observed clock-rate error exceeded the admitted finite limit.
  kCalibrationExpired = 4,       ///< The evidence supporting calibration is no longer fresh.
  kClockStalled = 5,             ///< The monitored clock stopped advancing when progress was due.
  kTimestampRegression = 6,      ///< A timestamp regressed within the monitored clock domain.
  kTimestampSourceFault = 7,     ///< The underlying timestamp source reported a fault.
  kMonitorFault = 8,             ///< The health monitor itself could not produce reliable evidence.
  kUnspecified = 9,              ///< A negative condition exists but has no more specific reason.
};

/// @brief Carry one bounded endpoint-local clock-health report.
struct ClockHealthReport {
  ClockHealthAssessment assessment{
      ClockHealthAssessment::kHealthy};  ///< Current assessment; defaults to healthy.
  ClockHealthReason reason{
      ClockHealthReason::kNone};  ///< Primary reason; \c kNone is valid only with healthy.

  /// @brief Create a healthy report with no negative reason.
  /// @return Valid healthy report.
  [[nodiscard]] static constexpr ClockHealthReport healthy() noexcept { return {}; }

  /// @brief Create a degraded report.
  /// @param reason Non-`kNone` primary reason validated at submission.
  /// @return Degraded report for later endpoint validation.
  [[nodiscard]] static constexpr ClockHealthReport degraded(ClockHealthReason reason) noexcept {
    return {.assessment = ClockHealthAssessment::kDegraded, .reason = reason};
  }

  /// @brief Create an unhealthy report.
  /// @param reason Non-`kNone` primary reason validated at submission.
  /// @return Unhealthy report for later endpoint validation.
  [[nodiscard]] static constexpr ClockHealthReport unhealthy(ClockHealthReason reason) noexcept {
    return {.assessment = ClockHealthAssessment::kUnhealthy, .reason = reason};
  }
};

/// @brief Carry the complete cold clock-health capability/requirement contract.
struct ClockHealthOptions {
  ClockHealthCapability capability{
      ClockHealthCapability::kUnavailable};  ///< Endpoint reporting capability; unavailable by
                                             ///< default.
  ClockHealthRequirement requirement{
      ClockHealthRequirement::kNotRequired};  ///< Whether fresh healthy evidence gates use.
  std::optional<std::chrono::nanoseconds>
      max_report_age;  ///< Positive maximum report age in nanoseconds; required only for monitored
                       ///< mode.
};

static_assert(std::is_standard_layout_v<ClockHealthReport>);
static_assert(std::is_trivially_copyable_v<ClockHealthReport>);

}  // namespace holoscan
