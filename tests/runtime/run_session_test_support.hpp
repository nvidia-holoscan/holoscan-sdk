/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/channel/pod_type.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/operator_ref.hpp>
#include <holoscan/core/run_session.hpp>

#include "runtime/execution_lane.hpp"
#include "runtime/runtime_diagnostic_sink.hpp"

// Diagnostic and terminal-summary recorders shared by the RunSession test files.
namespace holoscan {

struct RuntimeCompletionFencePayload {
  std::uint32_t value{};
};

template <>
struct ChannelTypeIdentity<RuntimeCompletionFencePayload> {
  static constexpr std::string_view name = "holoscan.test.RuntimeCompletionFencePayload";
  static constexpr std::uint32_t version = 1U;
};

template <>
struct ChannelPodLayout<RuntimeCompletionFencePayload> {
  static constexpr auto fields =
      pod_fields(pod_field("value", &RuntimeCompletionFencePayload::value,
                           offsetof(RuntimeCompletionFencePayload, value)));
};

namespace test::run_session {

// Captures every published Diagnostic sequence, runtime detail sequence, and lane
// startup evidence in memory for post-run assertion. When fail_=true it throws on
// publish to verify that sink exceptions cannot alter run outcome.
// Used by: BlockingAndAsyncRunsPublishTheExactSameTerminalDiagnostics, and others.
class RecordingDiagnosticSink final : public detail::RuntimeStructuralDiagnosticSink {
 public:
  explicit RecordingDiagnosticSink(bool fail = false) : fail_(fail) {}

  void publish(std::span<const Diagnostic> diagnostics) override {
    if (discard_for_injected_failure()) {
      return;
    }
    const std::lock_guard lock(mutex_);
    sequences_.emplace_back(diagnostics.begin(), diagnostics.end());
  }

  void publish_runtime_details(std::span<const detail::RuntimeDiagnosticDetail> details) override {
    if (discard_for_injected_failure()) {
      return;
    }
    const std::lock_guard lock(mutex_);
    detail_sequences_.emplace_back(details.begin(), details.end());
  }

  void publish_lifecycle_receipts(
      std::span<const detail::RuntimeLifecycleReceiptDiagnostic> receipts) override {
    if (discard_for_injected_failure()) {
      return;
    }
    const std::lock_guard lock(mutex_);
    lifecycle_receipt_sequences_.emplace_back(receipts.begin(), receipts.end());
  }

  void publish_lane_startup_evidence(
      std::span<const detail::ExecutionLaneStartupEvidence> evidence) override {
    if (discard_for_injected_failure()) {
      return;
    }
    const std::lock_guard lock(mutex_);
    lane_sequences_.emplace_back(evidence.begin(), evidence.end());
  }

  void publish_operator_completion_progress(
      std::span<const OperatorCompletionProgressRecord> progress) override {
    if (discard_for_injected_failure()) {
      return;
    }
    const std::lock_guard lock(mutex_);
    completion_progress_sequences_.emplace_back(progress.begin(), progress.end());
  }

  [[nodiscard]] std::vector<std::vector<Diagnostic>> snapshot() const {
    const std::lock_guard lock(mutex_);
    return sequences_;
  }

  [[nodiscard]] std::vector<std::vector<detail::RuntimeDiagnosticDetail>> snapshot_details() const {
    const std::lock_guard lock(mutex_);
    return detail_sequences_;
  }

  [[nodiscard]] std::vector<std::vector<detail::ExecutionLaneStartupEvidence>>
  snapshot_lane_evidence() const {
    const std::lock_guard lock(mutex_);
    return lane_sequences_;
  }

  [[nodiscard]] std::vector<std::vector<detail::RuntimeLifecycleReceiptDiagnostic>>
  snapshot_lifecycle_receipts() const {
    const std::lock_guard lock(mutex_);
    return lifecycle_receipt_sequences_;
  }

  [[nodiscard]] std::vector<std::vector<OperatorCompletionProgressRecord>>
  snapshot_operator_completion_progress() const {
    const std::lock_guard lock(mutex_);
    return completion_progress_sequences_;
  }

 private:
  [[nodiscard]] bool discard_for_injected_failure() const {
    if (!fail_) {
      return false;
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    throw std::runtime_error{"injected diagnostic sink failure"};
#else
    return true;
#endif
  }

  bool fail_{};
  mutable std::mutex mutex_;
  std::vector<std::vector<Diagnostic>> sequences_;
  std::vector<std::vector<detail::RuntimeDiagnosticDetail>> detail_sequences_;
  std::vector<std::vector<detail::RuntimeLifecycleReceiptDiagnostic>> lifecycle_receipt_sequences_;
  std::vector<std::vector<detail::ExecutionLaneStartupEvidence>> lane_sequences_;
  std::vector<std::vector<OperatorCompletionProgressRecord>> completion_progress_sequences_;
};

struct TerminalSummarySnapshot {
  RunGenerationId run_generation{};
  detail::RuntimeTerminalCategory category{detail::RuntimeTerminalCategory::kFailed};
  std::size_t diagnostic_count{};
  DiagnosticId first_diagnostic{};
  DiagnosticSource first_source{};
  std::string first_token;
  std::size_t structured_publication_count{};
};

class RecordingTerminalSummarySink final : public detail::RuntimeTerminalSummarySink {
 public:
  explicit RecordingTerminalSummarySink(
      std::shared_ptr<RecordingDiagnosticSink> structured_sink = {}, bool fail = false)
      : structured_sink_(std::move(structured_sink)), fail_(fail) {}

  void publish(const detail::RuntimeTerminalSummary& summary) override {
    const std::size_t structured_publication_count =
        structured_sink_ != nullptr ? structured_sink_->snapshot().size() : 0U;
    {
      const std::lock_guard lock(mutex_);
      snapshots_.push_back(TerminalSummarySnapshot{
          .run_generation = summary.run_generation,
          .category = summary.category,
          .diagnostic_count = summary.diagnostic_count,
          .first_diagnostic = summary.first_diagnostic,
          .first_source = summary.first_source,
          .first_token = std::string{summary.first_token},
          .structured_publication_count = structured_publication_count,
      });
    }
    if (fail_) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      throw std::runtime_error{"injected terminal summary sink failure"};
#endif
    }
  }

  [[nodiscard]] std::vector<TerminalSummarySnapshot> snapshot() const {
    const std::lock_guard lock(mutex_);
    return snapshots_;
  }

 private:
  std::shared_ptr<RecordingDiagnosticSink> structured_sink_;
  bool fail_{};
  mutable std::mutex mutex_;
  std::vector<TerminalSummarySnapshot> snapshots_;
};

void expect_diagnostic_sequences_equal(std::span<const Diagnostic> expected,
                                       std::span<const Diagnostic> actual);

void expect_terminal_compute_call_progress(const RunSession& session, const ExecutionPlan& plan,
                                           std::uint64_t configured_limit,
                                           std::uint64_t successful_compute_calls,
                                           bool limit_reached);

}  // namespace test::run_session
}  // namespace holoscan
