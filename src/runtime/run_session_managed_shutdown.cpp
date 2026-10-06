/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "run_session_internal.hpp"

#include "runtime_cuda_graph_region_resources.hpp"

#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace holoscan::detail {
namespace {

[[nodiscard]] int termination_result(const RunSession& session) noexcept {
  const RunTermination outcome = session.termination();
  return outcome == RunTermination::kCompleted || outcome == RunTermination::kRequestedCompleted
             ? 0
             : 1;
}

[[nodiscard]] constexpr ::holoscan::DiagnosticSeverity managed_signal_severity(
    const std::string_view token) noexcept {
  return token == "RUNTIME_MANAGED_SIGNAL_RECEIVED" || token == "RUNTIME_MANAGED_SIGNAL_ESCALATED"
             ? ::holoscan::DiagnosticSeverity::kWarning
             : ::holoscan::DiagnosticSeverity::kError;
}

template <typename Append>
void for_each_managed_signal_admission_diagnostic(const ManagedSignalAdmissionFailure& failure,
                                                  Append append) {
  RuntimeDiagnosticDetail detail{
      .present = true,
      .managed_signal = failure.detail,
      .operation = DiagnosticOperation::kRuntime,
  };
  detail.capture.emplace();
  detail.capture->append(failure.error.code, failure.error.message);
  append(failure.token, detail, managed_signal_severity(failure.token));
  if (!failure.cleanup_present) {
    return;
  }
  RuntimeDiagnosticDetail cleanup_detail{
      .present = true,
      .managed_signal = failure.cleanup_detail,
      .operation = DiagnosticOperation::kRuntime,
      .role = DiagnosticFailureRole::kTeardown,
  };
  cleanup_detail.capture.emplace();
  cleanup_detail.capture->append(failure.cleanup_error.code, failure.cleanup_error.message);
  append(failure.cleanup_token, cleanup_detail, managed_signal_severity(failure.cleanup_token));
}

void publish_managed_signal_admission_failure(
    const std::shared_ptr<const ExecutionPlanState>& plan,
    const ManagedSignalAdmissionFailure& failure) noexcept {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    const DiagnosticSource source{
        .graph_generation = plan != nullptr ? plan->graph.generation() : GraphGeneration{},
    };
    RuntimeDiagnosticBuffer diagnostics{diagnostic_id("RUNTIME_DIAGNOSTIC_CAPACITY_EXHAUSTED"),
                                        source};
    for_each_managed_signal_admission_diagnostic(
        failure,
        [&](const std::string_view token,
            const RuntimeDiagnosticDetail& detail,
            const ::holoscan::DiagnosticSeverity severity) {
          diagnostics.append(diagnostic_id(token), token, source, {}, &detail, {}, {}, severity);
        });
    if (plan != nullptr && plan->runtime_diagnostic_sink != nullptr) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        plan->runtime_diagnostic_sink->publish(diagnostics.diagnostics());
        plan->runtime_diagnostic_sink->publish_runtime_details(diagnostics.details());
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (...) {
        // A private observer cannot change fail-closed managed-domain admission.
        static_cast<void>(0);
      }
#endif
    }
    publish_runtime_terminal_summary(
        RunGenerationId{},
        diagnostics.diagnostics(),
        plan != nullptr ? plan->executable_factories.terminal_summary_sink.get() : nullptr,
        source);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    // Admission remains fail closed even when bounded structural evidence cannot be allocated.
    static_cast<void>(0);
  }
#endif
}

[[nodiscard]] RunSession managed_signal_admission_failure_session(
    const std::shared_ptr<const ExecutionPlanState>& retained,
    const ManagedSignalAdmissionFailure& failure) noexcept {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    auto state = std::make_unique<RunSessionState>(
        retained, failure, RunSessionState::ManagedSignalAdmissionFailureTag{});
    return make_run_session(state.release());
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    publish_managed_signal_admission_failure(retained, failure);
    RunSessionState* const startup_failure = RunSessionState::shared_startup_failure();
    startup_failure->publish_startup_failure(retained);
    return make_run_session(startup_failure);
  }
#endif
}

[[nodiscard]] int run_caller_managed(const ExecutionPlan& plan) {
  RunSession session = run_async(plan);
  session.wait();
  return termination_result(session);
}

#if defined(__linux__)
[[nodiscard]] int run_with_default_managed_signals(const ExecutionPlan& plan) {
  plan.require_no_errors();
  const auto retained = ExecutionPlanAccess::retain(plan);
  auto prepared = prepare_default_managed_signal_binding(
      retained->executable_factories.test_managed_signal_control);
  if (!prepared) {
    publish_managed_signal_admission_failure(retained, prepared.error());
    return 1;
  }
  auto controller = std::move(prepared).value().controller;
  RunSession session =
      start_run_session_retained(retained, std::nullopt, controller->terminal_participant());
  if (controller->caller_drives()) {
    controller->drive();
  }
  session.wait();
  return termination_result(session);
}
#endif

[[nodiscard]] RunSession run_async_with_managed_signals(const ExecutionPlan& plan,
                                                        ManagedShutdownSignals& signals) {
  plan.require_no_errors();
  const auto retained = ExecutionPlanAccess::retain(plan);
  auto prepared = prepare_explicit_managed_signal_binding(
      signals, retained->executable_factories.test_managed_signal_control);
  if (!prepared) {
    return managed_signal_admission_failure_session(retained, prepared.error());
  }
  auto controller = std::move(prepared).value().controller;
  return start_run_session_retained(retained, std::nullopt, controller->terminal_participant());
}

}  // namespace

RunSessionState::RunSessionState(std::shared_ptr<const ExecutionPlanState> plan,
                                 const ManagedSignalAdmissionFailure& failure,
                                 ManagedSignalAdmissionFailureTag)
    : plan_(std::move(plan)),
      joined_(true),
      diagnostic_buffer_(
          diagnostic_id("RUNTIME_DIAGNOSTIC_CAPACITY_EXHAUSTED"),
          DiagnosticSource{.graph_generation = plan_ != nullptr ? plan_->graph.generation() : 0U}) {
  for_each_managed_signal_admission_diagnostic(failure,
                                               [&](const std::string_view token,
                                                   const RuntimeDiagnosticDetail& detail,
                                                   const ::holoscan::DiagnosticSeverity severity) {
                                                 append_diagnostic(
                                                     token, {}, {}, &detail, {}, {}, severity);
                                               });
  finish(RunTermination::kFailed);
}

RunSessionState::RunSessionState(SharedStartupFailureTag)
    : joined_(true),
      shared_terminal_(true),
      diagnostic_buffer_(diagnostic_id("RUNTIME_DIAGNOSTIC_CAPACITY_EXHAUSTED"), {}) {
  append_diagnostic("RUNTIME_STARTUP_FAILED");
  {
    const std::lock_guard lock(join_mutex_);
    termination_.store(RunTermination::kFailed, std::memory_order_release);
  }
  join_cv_.notify_all();
}

void RunSessionState::attach_terminal_participant(
    std::shared_ptr<RuntimeTerminalParticipant> participant) noexcept {
  terminal_participant_ = std::move(participant);
  if (terminal_participant_ == nullptr) {
    return;
  }
  terminal_participant_->attach(RuntimeTerminalParticipantControl{
      .state = this,
      .request_stop =
          [](void* state) noexcept { static_cast<RunSessionState*>(state)->request_stop(); },
      .append_managed_diagnostic =
          [](void* state,
             const std::string_view token,
             const RuntimeManagedSignalDiagnostic& detail,
             const bool event) noexcept {
            static_cast<RunSessionState*>(state)->append_managed_signal_diagnostic(
                token, detail, event);
          },
      .finish_start_failure =
          [](void* state) noexcept {
            static_cast<RunSessionState*>(state)->finish_staged_start_failure();
          },
  });
}

void RunSessionState::fail_worker_start() noexcept {
  append_diagnostic("RUNTIME_WORKER_START_FAILED");
  retire_parameter_control();
  if (terminal_participant_ != nullptr) {
    terminal_participant_->stage_start_failure();
  } else {
    finish(RunTermination::kFailed);
  }
}

void RunSessionState::append_managed_signal_diagnostic(
    const std::string_view token, const RuntimeManagedSignalDiagnostic& managed_detail,
    const bool publish_event) noexcept {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    const RuntimeDiagnosticDetail detail{
        .present = true,
        .managed_signal = managed_detail,
    };
    const auto severity = managed_signal_severity(token);
    append_diagnostic(token, {}, {}, &detail, {}, {}, severity);
    if (!publish_event || plan_ == nullptr || plan_->runtime_diagnostic_sink == nullptr) {
      return;
    }
    const Diagnostic diagnostic{
        .id = diagnostic_id(token),
        .severity = severity,
        .token = std::string{token},
        .source = DiagnosticSource{.graph_generation = plan_->graph.generation()},
    };
    plan_->runtime_diagnostic_sink->publish_managed_signal_event(diagnostic, managed_detail);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    // Structural signal evidence is best effort and cannot change signal or run authority.
    static_cast<void>(0);
  }
#endif
}

}  // namespace holoscan::detail

namespace holoscan {

RunSession run_async(const ExecutionPlan& plan, ManagedShutdownSignals& signals) {
  return detail::run_async_with_managed_signals(plan, signals);
}

int run(const ExecutionPlan& plan) {
  return run(plan, RunSignalPolicy::kManaged);
}

int run(const ExecutionPlan& plan, const RunSignalPolicy signal_policy) {
  switch (signal_policy) {
    case RunSignalPolicy::kManaged:
#if defined(__linux__)
      return detail::run_with_default_managed_signals(plan);
#else
      return detail::run_caller_managed(plan);
#endif
    case RunSignalPolicy::kCallerManaged:
      return detail::run_caller_managed(plan);
  }
  return 1;
}

}  // namespace holoscan
