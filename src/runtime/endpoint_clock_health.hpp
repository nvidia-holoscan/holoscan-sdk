/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>

#include <holoscan/channel/clock_health.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan::detail {

class EndpointClockHealthRuntimeCoordinator;

/// Carry a bounded, allocation-free runtime failure from the private clock-health subsystem.
struct EndpointClockHealthError {
  ErrorCode code{ErrorCode::kFailure};
  std::string_view token;
};

/// Bind mutable receiver state to one admitted RunSession attachment.
struct EndpointClockHealthBindingIdentity {
  std::uint64_t run_generation{};
  std::uint64_t producer_epoch{};
  std::uint64_t clock_epoch{};

  friend constexpr bool operator==(EndpointClockHealthBindingIdentity lhs,
                                   EndpointClockHealthBindingIdentity rhs) noexcept = default;
};

/// Carry the fixed private control fields received from one publisher binding.
struct EndpointClockHealthControlRecord {
  std::uint64_t producer_epoch{};
  std::uint64_t publisher_operation_sequence{};
  std::uint64_t report_sequence{};
  std::uint8_t assessment{};
  std::uint8_t reason{};
};

/// Preserve the caller-supplied assessment independently from Runtime freshness.
enum class EndpointClockHealthAssessmentState : std::uint8_t {
  kAbsent = 0,
  kHealthy = 1,
  kDegraded = 2,
  kUnhealthy = 3,
};

/// Preserve whether receiver-accepted evidence is absent, current, or expired.
enum class EndpointClockHealthFreshness : std::uint8_t {
  kNoReport = 0,
  kFresh = 1,
  kStale = 2,
};

/// Identify why one stable CLOCK_HEALTH_UPDATE record was produced.
enum class EndpointClockHealthUpdateOrigin : std::uint8_t {
  kReportAccepted = 0,
  kReportExpired = 1,
  kAttachmentReset = 2,
  kProducerEpochReset = 3,
  kClockEpochReset = 4,
  kRuntimeHealthClockFault = 5,
};

/// Identify a separately detected fault in the independent health-age clock.
enum class EndpointHealthAgeClockFault : std::uint8_t {
  kStalled = 0,
  kRegression = 1,
};

/// Snapshot the complete bounded mutable state for eligibility and evidence consumers.
struct EndpointClockHealthState {
  EndpointClockHealthBindingIdentity binding{};
  EndpointClockHealthAssessmentState assessment{EndpointClockHealthAssessmentState::kAbsent};
  EndpointClockHealthFreshness freshness{EndpointClockHealthFreshness::kNoReport};
  std::optional<ClockHealthReason> reason;
  std::optional<std::int64_t> accepted_at_ns;
  std::optional<std::int64_t> expiry_deadline_ns;
  std::uint64_t publisher_operation_sequence{};
  std::uint64_t report_sequence{};
  std::uint64_t health_update_sequence{};
  std::uint64_t health_generation{};
};

/// Capture the report generation targeted by one independently scheduled expiry check.
struct EndpointClockHealthExpiryTicket {
  std::uint64_t health_generation{};
  std::uint64_t report_sequence{};
  std::int64_t expiry_deadline_ns{};
};

/// Carry one fixed-shape structural CLOCK_HEALTH_UPDATE event.
struct EndpointClockHealthUpdateRecord {
  static constexpr std::string_view kWitnessIdentity{"holoscan.event.clock-health-update.v1"};

  PortId endpoint{};
  PlanGenerationId plan_generation{};
  ClockId capture_clock{};
  std::uint64_t capture_conversion_id{};
  PlanGenerationId conversion_generation{};
  ClockId health_age_clock{};
  EndpointClockHealthBindingIdentity binding{};
  std::uint64_t publisher_operation_sequence{};
  std::uint64_t report_sequence{};
  std::uint64_t health_update_sequence{};
  std::uint64_t health_generation{};
  EndpointClockHealthUpdateOrigin origin{EndpointClockHealthUpdateOrigin::kAttachmentReset};
  EndpointClockHealthAssessmentState old_assessment{EndpointClockHealthAssessmentState::kAbsent};
  EndpointClockHealthAssessmentState new_assessment{EndpointClockHealthAssessmentState::kAbsent};
  EndpointClockHealthFreshness old_freshness{EndpointClockHealthFreshness::kNoReport};
  EndpointClockHealthFreshness new_freshness{EndpointClockHealthFreshness::kNoReport};
  std::optional<ClockHealthReason> old_reason;
  std::optional<ClockHealthReason> new_reason;
  std::optional<std::int64_t> receiver_acceptance_time_ns;
  std::optional<std::int64_t> expiry_deadline_ns;
  std::optional<std::int64_t> evaluation_time_ns;
  std::optional<EndpointHealthAgeClockFault> age_clock_fault;
  std::optional<std::uint64_t> cause_sequence;
  EndpointClockHealthDisposition disposition{EndpointClockHealthDisposition::kNotApplicable};
};

/// Own one endpoint's preallocated report ingress, receiver state, and update evidence.
///
/// Construction is cold and may allocate the two fixed-capacity arrays. Every operation after
/// construction is nonblocking, allocation-free, and externally serialized by the Dispatcher.
/// The class owns no thread and performs no clock reads; callers supply admitted age-clock values
/// and preserve Dispatcher TagTime order around report, expiry, reset, and fault operations.
class EndpointClockHealthReceiver {
 public:
  EndpointClockHealthReceiver(const EndpointClockHealthReceiver&) = delete;
  EndpointClockHealthReceiver& operator=(const EndpointClockHealthReceiver&) = delete;
  EndpointClockHealthReceiver(EndpointClockHealthReceiver&&) noexcept = default;
  EndpointClockHealthReceiver& operator=(EndpointClockHealthReceiver&&) noexcept = default;
  ~EndpointClockHealthReceiver() = default;

  /// Validate one monitored plan record and preallocate its exact finite storage.
  [[nodiscard]] static expected<EndpointClockHealthReceiver, EndpointClockHealthError> create(
      const EndpointClockHealthPlanRecord& plan,
      EndpointClockHealthBindingIdentity binding) noexcept;

  /// Enqueue one public report after fieldwise and generation validation.
  [[nodiscard]] expected<void, EndpointClockHealthError> submit_report(
      ClockHealthReport report, std::uint64_t producer_epoch,
      std::uint64_t publisher_operation_sequence, std::uint64_t report_sequence) noexcept;

  /// Decode and enqueue one fixed private record without mapping unknown values to kUnspecified.
  [[nodiscard]] expected<void, EndpointClockHealthError> submit_encoded(
      EndpointClockHealthControlRecord record) noexcept;

  /// Apply at most the compiled per-pass report budget at one admitted age-clock coordinate.
  [[nodiscard]] expected<std::size_t, EndpointClockHealthError> dispatch_pending(
      std::int64_t accepted_at_ns) noexcept;

  /// Return a generation-qualified ticket for the current fresh report.
  [[nodiscard]] std::optional<EndpointClockHealthExpiryTicket> expiry_ticket() const noexcept;

  /// Expire the ticket at the exact half-open deadline if it still targets current state.
  [[nodiscard]] expected<bool, EndpointClockHealthError> expire_if_due(
      EndpointClockHealthExpiryTicket ticket, std::int64_t evaluation_time_ns) noexcept;

  /// Reset state after externally proven quiescence for an attachment, producer, or clock epoch.
  [[nodiscard]] expected<void, EndpointClockHealthError> reset(
      EndpointClockHealthUpdateOrigin origin, EndpointClockHealthBindingIdentity binding,
      std::int64_t evaluation_time_ns,
      std::optional<std::uint64_t> cause_sequence = std::nullopt) noexcept;

  /// Apply fail-safe freshness and emit one update for a separately detected age-clock fault.
  [[nodiscard]] expected<void, EndpointClockHealthError> record_age_clock_fault(
      EndpointHealthAgeClockFault fault, std::int64_t evaluation_time_ns,
      std::optional<std::uint64_t> cause_sequence = std::nullopt) noexcept;

  /// Remove the oldest structural update for the trace/replay consumer.
  [[nodiscard]] std::optional<EndpointClockHealthUpdateRecord> pop_update() noexcept;

  /// Return the current immutable snapshot.
  [[nodiscard]] const EndpointClockHealthState& state() const noexcept { return state_; }

  /// Return whether current evidence exactly satisfies fresh_healthy_report.
  [[nodiscard]] bool satisfies_requirement() const noexcept;

  /// Return the plan-frozen disposition for the current state.
  [[nodiscard]] EndpointClockHealthDisposition current_disposition() const noexcept;

  /// Return the plan-local endpoint guarded by this receiver.
  [[nodiscard]] PortId endpoint() const noexcept { return plan_.endpoint; }

  [[nodiscard]] std::size_t pending_report_count() const noexcept { return ingress_size_; }
  [[nodiscard]] std::size_t pending_update_count() const noexcept { return update_size_; }

 private:
  // These plan-sized blocks are allocated once during cold construction and never grow. Dynamic
  // extent is required because the admitted capacity is known only after compilation.
  using IngressStorage =  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
      std::unique_ptr<EndpointClockHealthControlRecord[]>;
  using UpdateStorage =  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
      std::unique_ptr<EndpointClockHealthUpdateRecord[]>;

  friend class EndpointClockHealthRuntimeCoordinator;

  [[nodiscard]] expected<void, EndpointClockHealthError> preflight_reset(
      EndpointClockHealthUpdateOrigin origin,
      EndpointClockHealthBindingIdentity binding) const noexcept;
  void commit_reset(EndpointClockHealthUpdateOrigin origin,
                    EndpointClockHealthBindingIdentity binding, std::int64_t evaluation_time_ns,
                    std::optional<std::uint64_t> cause_sequence) noexcept;

  struct RuntimePlan {
    PortId endpoint{};
    PlanGenerationId plan_generation{};
    ClockId capture_clock{};
    std::uint64_t capture_conversion_id{};
    PlanGenerationId conversion_generation{};
    ClockId health_age_clock{};
    std::int64_t max_report_age_ns{};
    std::size_t report_capacity{};
    std::size_t update_capacity{};
    std::size_t per_dispatch_report_budget{};
    EndpointClockHealthDisposition no_report_disposition{
        EndpointClockHealthDisposition::kNotApplicable};
    EndpointClockHealthDisposition degraded_disposition{
        EndpointClockHealthDisposition::kNotApplicable};
    EndpointClockHealthDisposition unhealthy_disposition{
        EndpointClockHealthDisposition::kNotApplicable};
    EndpointClockHealthDisposition stale_disposition{
        EndpointClockHealthDisposition::kNotApplicable};
    ClockHealthUnspecifiedDisposition unspecified_reason_disposition{
        ClockHealthUnspecifiedDisposition::kAdmitted};
  };

  EndpointClockHealthReceiver(RuntimePlan plan, EndpointClockHealthBindingIdentity binding,
                              IngressStorage ingress, UpdateStorage updates) noexcept;

  [[nodiscard]] expected<void, EndpointClockHealthError> enqueue(
      EndpointClockHealthControlRecord record, bool private_decode) noexcept;
  void push_update(
      EndpointClockHealthUpdateOrigin origin, const EndpointClockHealthState& old_state,
      std::optional<std::int64_t> evaluation_time_ns, std::optional<std::uint64_t> cause_sequence,
      std::optional<EndpointHealthAgeClockFault> age_clock_fault = std::nullopt) noexcept;
  [[nodiscard]] bool update_slot_available() const noexcept;

  RuntimePlan plan_;
  IngressStorage ingress_;
  UpdateStorage updates_;
  EndpointClockHealthState state_;
  std::size_t ingress_head_{};
  std::size_t ingress_size_{};
  std::size_t update_head_{};
  std::size_t update_size_{};
  std::uint64_t enqueued_publisher_operation_sequence_{};
  std::uint64_t enqueued_report_sequence_{};
};

}  // namespace holoscan::detail
