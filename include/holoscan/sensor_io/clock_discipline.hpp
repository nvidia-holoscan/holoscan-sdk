/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>
#include <cstdint>
#include <limits>
#include <optional>

#include <holoscan/core/lifecycle.hpp>
#include <holoscan/time/clock_registry.hpp>
#include <holoscan/time/time_point.hpp>

namespace holoscan::sensor_io {

/// @brief Constrain a timestamp source to a non-throwing reader that can report unavailability.
/// @tparam Fn Candidate timestamp source.
template <typename Fn>
concept TimestampSource = requires(Fn source) {
  { source() } noexcept -> std::same_as<std::optional<std::int64_t>>;
};  // NOLINT(readability/braces): cpplint misparses a concept requirement-body terminator.

/// @brief Carry the measured relationship between a device's clock and a reference clock.
struct ClockOffsetEstimate {
  /// Value added to a device timestamp to reach the reference.
  std::int64_t offset_ns{};
  /// Conservative bound on the offset's measurement error: the ceiling of half the tightest
  /// observed bracket. This bounds the measurement only. It does not bound a projection made later,
  /// because relative clock drift accrues after the measurement and this estimate carries no rate
  /// term.
  std::int64_t uncertainty_ns{};
  /// Paired reads that contributed to the estimate.
  std::uint32_t samples{};
};

/// @brief Project device capture timestamps into a declared reference clock domain.
///
/// A capture device timestamps samples on its own clock, and that clock is not the one the rest of
/// the graph reasons about. Handing the raw device value to Core as a capture time is worse than
/// handing it nothing: the number is plausible, it is monotonic, and it is wrong by whatever the
/// two clocks happen to differ by, so every downstream alignment, window, and latency measurement
/// is quietly biased by that offset. Discarding the device timestamp instead and stamping arrival
/// time is the other common shortcut, and it replaces a fixed bias with jitter that tracks
/// scheduling delay.
///
/// This object measures the offset once and then applies it arithmetically. Each measurement round
/// brackets one device read between two reference reads and keeps the round with the tightest
/// bracket, which is the sample least contaminated by scheduling delay. The ceiling of half that
/// bracket is reported as the estimate's uncertainty, and \ref prime refuses to settle when even
/// the tightest round is wider than the caller's admitted maximum. Refusing to publish a capture
/// time the operator cannot justify is therefore the default, not a follow-up check that each
/// operator has to remember to write.
///
/// The split between measuring and applying is the whole design, so it is stated here rather than
/// left to be inferred. Measurement is cold: \ref bind, \ref bind_identity, \ref prime, and \ref
/// unbind read clocks and mutate state, and belong in a lifecycle body or in the single first
/// activation that owns the object. One \ref prime takes two reference reads and one device read
/// per round, so the default eight rounds cost 24 clock reads once for the epoch rather than once
/// per sample. Application is hot: \ref project, \ref settled, \ref estimate, and \ref domain are
/// the only members intended for \c compute. They read already-measured state and perform bounded
/// checked integer arithmetic, with no clock read, allocation, lock, or system call. Nothing here
/// re-measures on its own, so no sample ever pays for bracketing.
///
/// Priming belongs in a \c kWarmUp body and cannot currently live there. It needs the graph clock,
/// and the only cold route to that is \ref LifecycleContext::resolve_clock, whose resolver the
/// runtime never installs: the lifecycle executor fills in the notification and restart resolvers
/// of its internal callback state and leaves the clock resolver null, so every cold resolution
/// fails with \c ErrorCode::kLifecycleInvalidCallingContext. Until Core installs that resolver, an
/// operator has to take the clock from \c ExecutionContext::clock() and prime from its first
/// \c compute, guarding the call with \ref settled so later activations skip it.
/// That placement is a workaround for the Core gap and not a preference: it costs one \ref settled
/// check on every activation and the bracketed rounds on the first one. Move priming to \c kWarmUp
/// once cold clock resolution works.
///
/// A device that already timestamps in the reference domain needs no measurement, and \ref
/// bind_identity says so explicitly. That is a different statement from an unmeasured offset of
/// zero, and keeping the two distinguishable is the point: \ref project reports absence until the
/// relationship is actually known, and Core treats an absent capture time as absent rather than
/// substituting arrival time for it.
///
/// What this class does not do is discipline a clock. It is a one-time constant-offset calibration:
/// \ref project adds an offset and nothing else, with no rate term, no drift bound, no re-sampling
/// policy, and no expiry. That is correct only while the two clocks' relative rate error stays
/// negligible for the whole resource epoch, and the error admitted by violating that assumption
/// grows without bound in elapsed time, as the relative rate error multiplied by the time since the
/// measurement. An uncorrected crystal pair at 20 ppm, an ordinary commodity oscillator
/// specification, therefore accrues 20 microseconds of projection error per second, 72 milliseconds
/// per hour, and 1.7 seconds per day. The figure to compare that against is the reported
/// uncertainty, not zero: a healthy eight-round prime brackets well under a microsecond, and
/// 20 ppm consumes a microsecond of error in 50 milliseconds, so \ref
/// ClockOffsetEstimate::uncertainty_ns describes the measurement and says nothing about a
/// projection made a minute later. Devices disciplined to the reference's own source satisfy the
/// assumption by construction, whether through a shared PTP grandmaster or, as with V4L2, a device
/// that stamps the host's own \c CLOCK_MONOTONIC. A device running on an independent oscillator
/// does not, and needs a rate-estimating affine model this class deliberately does not provide.
///
/// The estimator also never measures rate scaling, and cannot detect it. \c RealtimeClock accepts
/// an \c initial_time_scale other than 1.0, which makes the true device-to-reference relationship
/// affine with a slope this class assumes to be exactly one; the resulting error is the scale's
/// distance from one multiplied by elapsed time, and no bracketed offset measurement can
/// distinguish it from a correct projection. The slope is at least fixed for the run, because
/// \c RealtimeClock is immutable after construction and runtime re-scaling is not supported. Do not
/// use this class against a scaled reference clock.
///
/// The object is not thread-safe. Priming performs the clock reads its callers supply; projection
/// performs arithmetic only, and allocates nothing.
class ClockDiscipline {
 public:
  /// @brief Measurement rounds \ref prime attempts when the caller names no other budget.
  static constexpr std::uint32_t kDefaultPrimeRounds = 8;

  /// @brief Admit any measured uncertainty, leaving the acceptance decision entirely to the caller.
  static constexpr std::int64_t kUnboundedUncertaintyNs = std::numeric_limits<std::int64_t>::max();

  /// @brief Widest uncertainty \ref prime admits when the caller names no other bound.
  ///
  /// One millisecond is where this estimate stops being worth having. The shortcut it replaces is
  /// stamping arrival time, whose error is dequeue and scheduling jitter of roughly that order for
  /// a typical capture pipeline, so an offset whose own uncertainty is wider buys nothing over the
  /// shortcut it displaced. The figure is also three orders of magnitude above what a healthy prime
  /// measures, because two vDSO clock reads bracket a device read in tens to hundreds of
  /// nanoseconds and the tightest of eight rounds normally lands well under a microsecond. The
  /// default therefore rejects only a cold path that was preempted in every one of its rounds,
  /// which is a real signal rather than something to publish through. Name a tighter figure when a
  /// consumer's alignment budget demands one, or \ref kUnboundedUncertaintyNs to admit anything.
  static constexpr std::int64_t kDefaultMaxUncertaintyNs = 1'000'000;

  /// @brief Bind the reference domain that projected timestamps will carry.
  /// @param reference_domain Clock identity resolved for the current lifecycle generation.
  /// @note Binding alone does not make the offset known; \ref project reports absence until the
  /// discipline is primed.
  void bind(ClockId reference_domain) noexcept {
    domain_ = reference_domain;
    estimate_ = {};
    settled_ = false;
  }

  /// @brief Bind the reference domain from a clock resolved through \ref LifecycleContext.
  /// @param reference Clock reference resolved for the current lifecycle generation.
  void bind(const ClockRef& reference) noexcept { bind(reference.id()); }

  /// @brief Declare that the device already timestamps in the reference domain.
  /// @param reference_domain Clock identity the device's own timestamps belong to.
  /// @note This is an assertion about the hardware, not a measurement. Use it only when the device
  /// is disciplined to the same source as the reference clock, such as a sensor and host that share
  /// a PTP grandmaster.
  /// @note An unset \ref ClockId leaves the discipline unsettled. The assertion this call makes is
  /// that the device stamps in a particular domain, so it says nothing at all when no domain is
  /// named, and reporting settled for one would let \ref project label a timestamp with a clock the
  /// caller never resolved. \ref prime already refuses the same input for the same reason.
  void bind_identity(ClockId reference_domain) noexcept {
    domain_ = reference_domain;
    estimate_ = {};
    settled_ = reference_domain != ClockId{};
  }

  /// @brief Measure the device-to-reference offset from bracketed paired clock reads.
  /// @tparam DeviceNow Non-throwing reader for the device's own clock.
  /// @tparam ReferenceNow Non-throwing reader for the bound reference clock.
  /// @param device_now Reader returning the device's current timestamp in nanoseconds.
  /// @param reference_now Reader returning the reference clock's current timestamp in nanoseconds.
  /// @param rounds Paired reads to attempt; the tightest bracket wins. This doubles as the retry
  /// budget for scheduling noise, since one preempted round only has to be beaten by another, and
  /// the budget is bounded, explicit, and spent entirely within this call without sleeping.
  /// @param max_uncertainty_ns Widest uncertainty the caller is willing to publish through. A bound
  /// of zero admits only a round whose two reference reads returned the same value.
  /// @return \ref LifecycleStatus::kOk once an offset is known, \ref LifecycleStatus::kSkipped when
  /// the discipline is already settled, \ref LifecycleStatus::kRetryableFailure when a round
  /// produced a usable pair but the tightest bracket exceeded \p max_uncertainty_ns, or \ref
  /// LifecycleStatus::kFatalFailure when no domain is bound or no round produced a usable pair.
  /// @note A rejected or failed prime leaves the discipline unsettled, so \ref project keeps
  /// reporting absence and the caller may prime again against the same binding. Core classifies
  /// \ref LifecycleStatus::kRetryableFailure but does not itself retry a stage, so an operator that
  /// wants another attempt must either widen \p rounds or call \ref prime again from a later cold
  /// call or activation.
  /// @note This belongs in a \c kWarmUp body. See the class description for why the missing Core
  /// clock resolver currently forces it into the first \c compute instead.
  template <TimestampSource DeviceNow, TimestampSource ReferenceNow>
  [[nodiscard]] LifecycleStatus prime(
      DeviceNow&& device_now, ReferenceNow&& reference_now,
      std::uint32_t rounds = kDefaultPrimeRounds,
      std::int64_t max_uncertainty_ns = kDefaultMaxUncertaintyNs) noexcept {
    if (domain_ == ClockId{}) {
      return LifecycleStatus::kFatalFailure;
    }
    if (settled_) {
      return LifecycleStatus::kSkipped;
    }

    ClockOffsetEstimate best{.uncertainty_ns = std::numeric_limits<std::int64_t>::max()};
    for (std::uint32_t round = 0; round < rounds; ++round) {
      const std::optional<std::int64_t> before = reference_now();
      const std::optional<std::int64_t> device = device_now();
      const std::optional<std::int64_t> after = reference_now();
      if (!before.has_value() || !device.has_value() || !after.has_value()) {
        continue;
      }
      const std::optional<std::int64_t> bracket = checked_sub(*after, *before);
      if (!bracket.has_value() || *bracket < 0) {
        continue;
      }
      // The midpoint of the bracket is the best available estimate of the reference coordinate at
      // the instant the device was read.
      const std::optional<std::int64_t> span = checked_add(*before, *bracket / 2);
      if (!span.has_value()) {
        continue;
      }
      const std::optional<std::int64_t> offset = checked_sub(*span, *device);
      if (!offset.has_value()) {
        continue;
      }
      // The midpoint above truncates toward `before`, which leaves `after` exactly
      // ceil(bracket / 2) away, so the ceiling is the true bound and not a pessimism. An
      // uncertainty that under-reports by even 1ns is worse than no figure at all, because a
      // consumer comparing two capture times against it can conclude they are ordered when they are
      // not. The bracket is known non-negative by the guard above, so this form is exactly the
      // ceiling and cannot overflow, unlike `(bracket + 1) / 2`, which wraps at the widest bracket
      // and truncates the wrong way for a negative operand.
      const std::int64_t uncertainty = *bracket - *bracket / 2;
      ++best.samples;
      if (uncertainty < best.uncertainty_ns) {
        best.offset_ns = *offset;
        best.uncertainty_ns = uncertainty;
      }
    }

    if (best.samples == 0) {
      return LifecycleStatus::kFatalFailure;
    }
    if (best.uncertainty_ns > max_uncertainty_ns) {
      // Every round landed across a scheduling event. The clocks themselves are not implicated, so
      // this is worth another cold attempt rather than a fatal startup failure, and leaving the
      // discipline unsettled keeps the rejected offset out of every projection in the meantime.
      return LifecycleStatus::kRetryableFailure;
    }
    estimate_ = best;
    settled_ = true;
    return LifecycleStatus::kOk;
  }

  /// @brief Project a device timestamp into the bound reference domain.
  /// @param device_ns Device-reported capture timestamp in nanoseconds.
  /// @return Labelled reference-domain time point, or absence when the discipline is unbound, not
  /// yet primed, or the projection would overflow.
  /// @note The result is shaped for \c EmitOptions::capture_time, so an unprojectable timestamp
  /// propagates as canonical absence instead of a fabricated coordinate.
  /// @note This applies the measured offset and nothing else. It assumes unity relative rate
  /// between the two clocks, never expires, and carries no drift term, so its error grows with the
  /// time elapsed since \ref prime whenever that assumption is violated. See the class description
  /// for the magnitude that assumption admits.
  [[nodiscard]] std::optional<TimePoint> project(std::int64_t device_ns) const noexcept {
    if (!settled_ || domain_ == ClockId{}) {
      return std::nullopt;
    }
    const std::optional<std::int64_t> projected = checked_add(device_ns, estimate_.offset_ns);
    if (!projected.has_value()) {
      return std::nullopt;
    }
    return TimePoint{.timestamp_ns = *projected, .clock_id = domain_};
  }

  /// @brief Return whether the device-to-reference relationship is known.
  [[nodiscard]] bool settled() const noexcept { return settled_; }

  /// @brief Return the measured offset and its uncertainty.
  [[nodiscard]] ClockOffsetEstimate estimate() const noexcept { return estimate_; }

  /// @brief Return the bound reference clock identity, or zero when unbound.
  [[nodiscard]] ClockId domain() const noexcept { return domain_; }

  /// @brief Discard the binding and the measured offset.
  /// @note Call this when the device stream is released, so a later epoch re-measures instead of
  /// projecting through an offset that belonged to the previous one.
  void unbind() noexcept { *this = ClockDiscipline{}; }

 private:
  static constexpr std::optional<std::int64_t> checked_add(std::int64_t lhs,
                                                           std::int64_t rhs) noexcept {
    constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
    constexpr std::int64_t kMin = std::numeric_limits<std::int64_t>::min();
    if ((rhs > 0 && lhs > kMax - rhs) || (rhs < 0 && lhs < kMin - rhs)) {
      return std::nullopt;
    }
    return lhs + rhs;
  }

  static constexpr std::optional<std::int64_t> checked_sub(std::int64_t lhs,
                                                           std::int64_t rhs) noexcept {
    constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
    constexpr std::int64_t kMin = std::numeric_limits<std::int64_t>::min();
    if ((rhs < 0 && lhs > kMax + rhs) || (rhs > 0 && lhs < kMin + rhs)) {
      return std::nullopt;
    }
    return lhs - rhs;
  }

  ClockId domain_{};
  ClockOffsetEstimate estimate_{};
  bool settled_{};
};

}  // namespace holoscan::sensor_io
