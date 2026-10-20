/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "run_session_internal.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>

#include <holoscan/core/run.hpp>

#include "compiler/execution_plan_lookup.hpp"
#include "core/canonical_text_encoding.hpp"
#include "core/diagnostic_reporting.hpp"
#include "managed_shutdown_signals_test_control.hpp"
#include "runtime_cuda_graph_diagnostics.hpp"
#include "runtime_cuda_graph_region_resources.hpp"
#include "runtime_termination.hpp"
#include "runtime_test_controls.hpp"
#include "runtime_work_inventory.hpp"
#include "runtime_worker.hpp"

namespace holoscan {
namespace detail {

// Run-generation ownership, public controls, diagnostics, and terminal resource lifetime.

namespace {

const void*& run_owned_session_storage() noexcept {
  static thread_local const void* owner = nullptr;
  return owner;
}

}  // namespace

const void* current_run_owned_session() noexcept {
  return run_owned_session_storage();
}

void set_run_owned_session(const void* owner) noexcept {
  run_owned_session_storage() = owner;
}

namespace {

[[nodiscard]] std::shared_ptr<const ExecutionPlanState> validate_run_plan(
    std::shared_ptr<const ExecutionPlanState> plan) {
  if (plan == nullptr ||
      plan->operator_parameters.size() > std::numeric_limits<OperatorInstanceId>::max()) {
    throw RuntimeError{Error{ErrorCode::kIncompatible}};
  }
  return plan;
}

inline constexpr std::array<std::string_view, 18> kRuntimeFallbackTokens{{
    "RUNTIME_PORT_BINDING_UNAVAILABLE",
    "RUNTIME_EXECUTION_LANE_PLAN_INVALID",
    "RUNTIME_LANE_STARTUP_FAILED",
    "RUNTIME_LANE_JOIN_FAILED",
    "RUNTIME_OPERATOR_REPLAY_MISMATCH",
    "READY_POLICY_PLAN_INVALID",
    "RUNTIME_READINESS_BINDING_UNAVAILABLE",
    "RUNTIME_CLOCK_BINDING_UNAVAILABLE",
    "RUNTIME_STARTUP_FAILED",
    "RUNTIME_PARAMETER_CONTROL_BINDING_FAILED",
    "RUNTIME_LIFECYCLE_PLAN_INVALID",
    "RUNTIME_MANAGED_SIGNAL_RECEIVED",
    "RUNTIME_MANAGED_SIGNAL_OWNER_BUSY",
    "RUNTIME_MANAGED_SIGNAL_DOMAIN_REQUIRED",
    "RUNTIME_MANAGED_SIGNAL_INSTALL_FAILED",
    "RUNTIME_MANAGED_SIGNAL_WAIT_FAILED",
    "RUNTIME_MANAGED_SIGNAL_RESTORE_FAILED",
    "RUNTIME_MANAGED_SIGNAL_ESCALATED",
}};

consteval bool runtime_fallback_tokens_are_registered() {
  for (const std::string_view token : kRuntimeFallbackTokens) {
    if (!diagnostic_id(token)) {
      return false;
    }
  }
  return true;
}

static_assert(runtime_fallback_tokens_are_registered(),
              "every Runtime fallback token must be registered");

[[nodiscard]] RunGenerationId reserve_run_generation() {
  static std::atomic<RunGenerationId> next_generation{1U};
  RunGenerationId generation = next_generation.load(std::memory_order_relaxed);
  while (generation != std::numeric_limits<RunGenerationId>::max()) {
    if (next_generation.compare_exchange_weak(
            generation, generation + 1U, std::memory_order_relaxed)) {
      return generation;
    }
  }
  throw RuntimeError{Error{ErrorCode::kResourceExhausted, "RUNTIME_RUN_GENERATION_EXHAUSTED"}};
}

[[nodiscard]] PortId requested_diagnostic_port(PortId explicit_port,
                                               const RuntimeDiagnosticDetail* detail) noexcept {
  if (detail != nullptr && detail->tensor_failure.port) {
    return detail->tensor_failure.port;
  }
  return explicit_port;
}

[[nodiscard]] RuntimeTerminalSummary make_terminal_summary(
    RunGenerationId run_generation, std::span<const Diagnostic> diagnostics,
    DiagnosticSource fallback_source = {}) noexcept {
  RuntimeTerminalSummary summary{
      .run_generation = run_generation,
      .category = RuntimeTerminalCategory::kFailed,
      .diagnostic_count = diagnostics.size(),
      .first_source = fallback_source,
      .first_token = "none",
  };
  if (!diagnostics.empty()) {
    summary.first_diagnostic = diagnostics.front().id;
    summary.first_source = diagnostics.front().source;
    if (summary.first_source.graph_generation == 0U) {
      summary.first_source.graph_generation = fallback_source.graph_generation;
    }
    if (!summary.first_source.graph) {
      summary.first_source.graph = fallback_source.graph;
    }
    if (!summary.first_source.vertex) {
      summary.first_source.vertex = fallback_source.vertex;
    }
    if (!summary.first_source.port) {
      summary.first_source.port = fallback_source.port;
    }
    if (summary.first_diagnostic &&
        diagnostic_id(diagnostics.front().token) == summary.first_diagnostic) {
      summary.first_token = diagnostics.front().token;
    }
  }
  return summary;
}

void publish_terminal_summary(RunGenerationId run_generation,
                              std::span<const Diagnostic> diagnostics,
                              RuntimeTerminalSummarySink* sink,
                              DiagnosticSource fallback_source = {}) noexcept {
  const RuntimeTerminalSummary summary =
      make_terminal_summary(run_generation, diagnostics, fallback_source);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    if (sink != nullptr) {
      sink->publish(summary);
      return;
    }
    HOLOSCAN_LOG_AUTOMATIC_ERROR(
        "RunSession failed: run_generation={} category=failed diagnostics={}\n{}",
        summary.run_generation,
        summary.diagnostic_count,
        diagnostics.empty() ? "No diagnostic detail retained."
                            : format_diagnostics(diagnostics, ExecutionGraphView{}));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    // Supplemental observability cannot weaken an already-committed terminal outcome.
    static_cast<void>(0);
  }
#endif
}

enum class ProcessGroupResolutionDisposition : std::uint8_t {
  kAccepted,
  kSelectionRequired,
  kInvalid,
};

struct ProcessGroupResolution {
  ProcessGroupId process_group{};
  ProcessGroupResolutionDisposition disposition{ProcessGroupResolutionDisposition::kInvalid};
};

[[nodiscard]] ProcessGroupResolution resolve_process_group(
    const ExecutionPlanState& plan, std::optional<ProcessGroupId> requested) noexcept {
  ProcessGroupId sole_group{};
  bool multiple_groups = false;
  bool requested_present = false;
  for (const auto& placement : plan.vertex_process_placements) {
    if (!placement.process_group) {
      return {};
    }
    requested_present =
        requested_present || (requested.has_value() && placement.process_group == *requested);
    if (!sole_group) {
      sole_group = placement.process_group;
    } else if (sole_group != placement.process_group) {
      multiple_groups = true;
    }
  }
  if (!has_structural_runtime_topology(plan)) {
    if (requested.has_value()) {
      return {};
    }
    return ProcessGroupResolution{
        .disposition = ProcessGroupResolutionDisposition::kAccepted,
    };
  }
  if (!sole_group || (requested.has_value() && (!*requested || !requested_present))) {
    return {};
  }
  if (requested.has_value()) {
    return ProcessGroupResolution{
        .process_group = *requested,
        .disposition = ProcessGroupResolutionDisposition::kAccepted,
    };
  }
  if (multiple_groups) {
    return ProcessGroupResolution{
        .disposition = ProcessGroupResolutionDisposition::kSelectionRequired,
    };
  }
  return ProcessGroupResolution{
      .process_group = sole_group,
      .disposition = ProcessGroupResolutionDisposition::kAccepted,
  };
}

[[nodiscard]] bool parameter_field_visible(const ParamFieldDescriptor& field,
                                           const ParamAccessPolicy& policy, bool update) noexcept {
  if ((field.visibility == ParamVisibility::kInternal && !policy.include_internal) ||
      (field.sensitivity == ParamSensitivity::kSensitive && !policy.include_sensitive)) {
    return false;
  }
  if (policy.origin != ParamControlOrigin::kExternalAdapter) {
    return true;
  }
  return update ? field.external_access == ParamExternalAccess::kTuning
                : field.external_access != ParamExternalAccess::kNone;
}

[[nodiscard]] bool parameter_scope_allows(ParamScopeFilter scope, ParamAddress address) noexcept {
  if (scope == ParamScopeFilter::run()) {
    return true;
  }
  return scope.operator_instance == address.operator_instance &&
         (scope.field == 0U || scope.field == address.field);
}

[[nodiscard]] expected<ParamReadResult, Error> parameter_control_read(
    void* coordinator, const ParamAccessPolicy& policy, ParamAddress address, ParamReadView view) {
  if (!has_param_permission(policy.permissions, ParamPermission::kRead)) {
    return make_unexpected(Error{ErrorCode::kPermissionDenied});
  }
  if (!parameter_scope_allows(policy.allowed_scope, address)) {
    return make_unexpected(Error{ErrorCode::kPermissionDenied});
  }
  auto& updates = *static_cast<ParameterUpdateCoordinator*>(coordinator);
  const ParamFieldDescriptor* const field = updates.field_descriptor(address);
  if (field != nullptr && !parameter_field_visible(*field, policy, false)) {
    return make_unexpected(Error{ErrorCode::kPermissionDenied});
  }
  return updates.read(address, view);
}

[[nodiscard]] const ParameterCatalogView& parameter_control_catalog(void* coordinator) noexcept {
  return static_cast<ParameterUpdateCoordinator*>(coordinator)->catalog();
}

[[nodiscard]] std::uint32_t parameter_control_max_event_cursors(void* coordinator) noexcept {
  return static_cast<ParameterUpdateCoordinator*>(coordinator)->max_event_cursors();
}

[[nodiscard]] ParamBatchUpdateResult parameter_permission_rejection(
    std::size_t patch_count, std::size_t failed_patch) noexcept {
  ParamBatchUpdateResult result{
      .status = ParamUpdateStatus::kRejected,
      .committed = false,
  };
  if (patch_count <= ParamBatchUpdateResult::kNoFailedPatch) {
    result.patch_count = static_cast<std::uint32_t>(patch_count);
  }
  if (failed_patch <= ParamBatchUpdateResult::kNoFailedPatch) {
    result.failed_patch = static_cast<std::uint32_t>(failed_patch);
  }
  return result;
}

[[nodiscard]] ParamBatchUpdateResult parameter_control_submit(void* coordinator,
                                                              const ParamAccessPolicy& policy,
                                                              std::span<const ParamPatch> patches,
                                                              ParamUpdateOptions options) {
  if (!has_param_permission(policy.permissions, ParamPermission::kUpdate)) {
    return parameter_permission_rejection(patches.size(), 0U);
  }
  auto& updates = *static_cast<ParameterUpdateCoordinator*>(coordinator);
  for (std::size_t index = 0U; index < patches.size(); ++index) {
    if (!parameter_scope_allows(policy.allowed_scope, patches[index].target)) {
      return parameter_permission_rejection(patches.size(), index);
    }
    const ParamFieldDescriptor* const field = updates.field_descriptor(patches[index].target);
    if (field != nullptr && !parameter_field_visible(*field, policy, true)) {
      return parameter_permission_rejection(patches.size(), index);
    }
  }
  return updates.submit(patches, options);
}

[[nodiscard]] expected<ParamUpdateResult, Error> parameter_control_await(
    void* coordinator, const ParamAccessPolicy& policy, ParamUpdateSequence sequence,
    std::chrono::nanoseconds timeout) {
  if (!has_param_permission(policy.permissions, ParamPermission::kAwait)) {
    return make_unexpected(Error{ErrorCode::kPermissionDenied});
  }
  return static_cast<ParameterUpdateCoordinator*>(coordinator)
      ->await_authorized(sequence, timeout, policy);
}

[[nodiscard]] expected<ParamSnapshot, Error> parameter_control_capture(
    void* coordinator, const ParamAccessPolicy& policy, ParamCatalogDigest source_catalog_digest,
    ParamCaptureOptions options) {
  auto& updates = *static_cast<ParameterUpdateCoordinator*>(coordinator);
  if (!has_param_permission(policy.permissions, ParamPermission::kRead)) {
    return updates.capture(policy, options);
  }
  return updates.capture(policy, source_catalog_digest, options);
}

[[nodiscard]] expected<std::uint32_t, Error> parameter_control_register_event_cursor(
    void* coordinator, const ParamAccessPolicy& policy) noexcept {
  return static_cast<ParameterUpdateCoordinator*>(coordinator)->register_event_cursor(policy);
}

void parameter_control_unregister_event_cursor(void* coordinator, std::uint32_t cursor) noexcept {
  static_cast<ParameterUpdateCoordinator*>(coordinator)->unregister_event_cursor(cursor);
}

[[nodiscard]] expected<std::size_t, Error> parameter_control_drain_events(
    void* coordinator, std::uint32_t cursor, std::span<ParamEvent> output) noexcept {
  return static_cast<ParameterUpdateCoordinator*>(coordinator)->drain_events(cursor, output);
}

void parameter_control_begin_detach(void* coordinator) noexcept {
  static_cast<ParameterUpdateCoordinator*>(coordinator)->begin_detach();
}

constexpr ParamControlOperations kParameterControlOperations{
    .catalog = &parameter_control_catalog,
    .max_event_cursors = &parameter_control_max_event_cursors,
    .read = &parameter_control_read,
    .submit = &parameter_control_submit,
    .await = &parameter_control_await,
    .capture = &parameter_control_capture,
    .register_event_cursor = &parameter_control_register_event_cursor,
    .unregister_event_cursor = &parameter_control_unregister_event_cursor,
    .drain_events = &parameter_control_drain_events,
    .begin_detach = &parameter_control_begin_detach,
};

}  // namespace

[[nodiscard]] std::string_view registered_token_or(std::string_view candidate,
                                                   std::string_view fallback) noexcept {
  const std::string_view token = candidate.substr(0U, candidate.find(':'));
  return !token.empty() && diagnostic_id(token) ? token : fallback;
}

void publish_runtime_terminal_summary(RunGenerationId run_generation,
                                      std::span<const Diagnostic> diagnostics,
                                      RuntimeTerminalSummarySink* sink,
                                      DiagnosticSource fallback_source) noexcept {
  publish_terminal_summary(run_generation, diagnostics, sink, fallback_source);
}

// A failed CUDA quiesce leaves native execution state uncertain. Preserve every dependent owner
// for the remainder of the process so neither Operator state nor Tensor backing can be reclaimed
// underneath potentially active device work. The device claim remains quarantined as well, so
// recovery requires process restart. Nodes are intentionally never destroyed at process exit.
struct QuarantinedRunExecutionOwners {
  std::unique_ptr<GraphExec> graph;
  std::unique_ptr<RuntimeTensorResources> tensors;
  std::shared_ptr<RuntimeChannelSession> channels;
  std::unique_ptr<RuntimeCUDAGraphRegionResources> cuda_graph;
  std::unique_ptr<RuntimeCudaExecutionResources> cuda;
  QuarantinedRunExecutionOwners* next{};
};

void quarantine_run_execution_owners(std::unique_ptr<GraphExec> graph,
                                     std::unique_ptr<RuntimeTensorResources> tensors,
                                     std::shared_ptr<RuntimeChannelSession> channels,
                                     std::unique_ptr<RuntimeCUDAGraphRegionResources> cuda_graph,
                                     std::unique_ptr<RuntimeCudaExecutionResources> cuda) noexcept {
  if (cuda_graph != nullptr) {
    cuda_graph->detach_session_state();
  }
  auto* node = new (std::nothrow) QuarantinedRunExecutionOwners{.graph = std::move(graph),
                                                                .tensors = std::move(tensors),
                                                                .channels = std::move(channels),
                                                                .cuda_graph = std::move(cuda_graph),
                                                                .cuda = std::move(cuda)};
  if (node == nullptr) {
    // Allocation failure cannot make destruction safe. Deliberately relinquish ownership while
    // retaining the quarantined native claim instead of running unsafe destructors.
    // NOLINTBEGIN(bugprone-unused-return-value,clang-analyzer-cplusplus.NewDeleteLeaks)
    static_cast<void>(graph.release());
    static_cast<void>(tensors.release());
    if (channels != nullptr) {
      // A shared owner cannot relinquish its reference in place. Leak a moved copy; if even that
      // allocation fails, termination is safer than unmapping storage beneath uncertain CUDA work.
      auto* leaked_channels =
          new (std::nothrow) std::shared_ptr<RuntimeChannelSession>(std::move(channels));
      if (leaked_channels == nullptr) {
        std::terminate();
      }
      static_cast<void>(leaked_channels);
    }
    static_cast<void>(cuda_graph.release());
    static_cast<void>(cuda.release());
    // NOLINTEND(bugprone-unused-return-value,clang-analyzer-cplusplus.NewDeleteLeaks)
    return;
  }
  static std::mutex quarantine_mutex;
  static QuarantinedRunExecutionOwners* quarantine_head{};
  const std::lock_guard lock(quarantine_mutex);
  node->next = quarantine_head;
  quarantine_head = node;
}

RunSessionState::RunSessionState(std::shared_ptr<const ExecutionPlanState> plan,
                                 RunGenerationId run_generation, ProcessGroupId process_group)
    : plan_(validate_run_plan(std::move(plan))),
      run_generation_(run_generation),
      process_group_(process_group),
      phase_counters_(plan_->executable_factories.phase_counters.get()),
      compute_lane_invoke_(phase_counters_ != nullptr ? &invoke_compute_lane<true>
                                                      : &invoke_compute_lane<false>),
      compute_lane_with_completion_invoke_(select_compute_lane_with_completion_invoke(
          phase_counters_ != nullptr,
          plan_->executable_factories.test_operator_completion_final_entry_control != nullptr)),
      diagnostic_buffer_(diagnostic_id("RUNTIME_DIAGNOSTIC_CAPACITY_EXHAUSTED"),
                         DiagnosticSource{.graph_generation = plan_->graph.generation()}) {
  if (plan_ == nullptr ||
      plan_->operator_parameters.size() > std::numeric_limits<OperatorInstanceId>::max()) {
    throw RuntimeError{Error{ErrorCode::kIncompatible}};
  }
  initialize_local_operator_completion_progress();
}

RunSessionState::RunSessionState(std::shared_ptr<const ExecutionPlanState> plan,
                                 RunGenerationId run_generation, ProcessGroupSelectionFailureTag)
    : plan_(std::move(plan)),
      run_generation_(run_generation),
      joined_(true),
      diagnostic_buffer_(
          diagnostic_id("RUNTIME_DIAGNOSTIC_CAPACITY_EXHAUSTED"),
          DiagnosticSource{.graph_generation = plan_ != nullptr ? plan_->graph.generation() : 0U}) {
  if (plan_ != nullptr) {
    // The default-zero process_group_ is the deliberate unselected sentinel. Inventory
    // validation still runs, but worker_owns_vertex() materializes no local progress cells.
    initialize_local_operator_completion_progress();
  }
  append_diagnostic("RUNTIME_PROCESS_GROUP_SELECTION_REQUIRED");
  finish(RunTermination::kFailed);
}

RunSessionState::~RunSessionState() {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    retire_parameter_control();
    if (cuda_graph_quiesce_failure_) {
      if (cuda_execution_resources_ != nullptr) {
        cuda_execution_resources_->quarantine_after_external_failure();
      }
      quarantine_run_execution_owners(std::move(graph_exec_),
                                      std::move(tensor_resources_),
                                      std::move(channel_session_),
                                      std::move(cuda_graph_region_resources_),
                                      std::move(cuda_execution_resources_));
      return;
    }
    const bool graph_quiesced = cuda_graph_region_resources_ == nullptr ||
                                cuda_graph_region_resources_->quiesce().has_value();
    const bool streams_quiesced =
        cuda_execution_resources_ == nullptr || cuda_execution_resources_->quiesce().has_value();
    if (!graph_quiesced || !streams_quiesced) {
      if (!graph_quiesced && cuda_execution_resources_ != nullptr) {
        cuda_execution_resources_->quarantine_after_external_failure();
      }
      quarantine_run_execution_owners(std::move(graph_exec_),
                                      std::move(tensor_resources_),
                                      std::move(channel_session_),
                                      std::move(cuda_graph_region_resources_),
                                      std::move(cuda_execution_resources_));
      return;
    }
    if (cuda_graph_region_resources_ != nullptr &&
        !cuda_graph_region_resources_->shutdown().has_value()) {
      if (cuda_execution_resources_ != nullptr) {
        cuda_execution_resources_->quarantine_after_external_failure();
      }
      quarantine_run_execution_owners(std::move(graph_exec_),
                                      std::move(tensor_resources_),
                                      std::move(channel_session_),
                                      std::move(cuda_graph_region_resources_),
                                      std::move(cuda_execution_resources_));
      return;
    }
    if (tensor_resources_ != nullptr) {
      tensor_resources_->close();
      tensor_resources_->detach_ports();
    }
    if (cuda_graph_region_resources_ != nullptr) {
      cuda_graph_region_resources_->detach_session_state();
    }
    // Channel bindings borrow reconstructed Operator ports, while Operators may retain
    // unpublished loans. After CUDA quiescence, destroy the Channel session before its borrowed
    // ports, then destroy Operators while the fixed Tensor slot state still exists.
    channel_session_.reset();
    retained_parameter_updates_.reset();
    cuda_graph_region_resources_.reset();
    graph_exec_.reset();
    tensor_resources_.reset();
    cuda_execution_resources_.reset();
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    // A destructor cannot surface a cold-path failure. Retain the complete native owner closure
    // rather than terminating or destroying storage whose CUDA completion is now uncertain.
    quarantine_run_execution_owners(std::move(graph_exec_),
                                    std::move(tensor_resources_),
                                    std::move(channel_session_),
                                    std::move(cuda_graph_region_resources_),
                                    std::move(cuda_execution_resources_));
  }
#endif
}

[[nodiscard]] RunSessionState* RunSessionState::shared_startup_failure() noexcept {
  static RunSessionState state{SharedStartupFailureTag{}};
  return &state;
}

void RunSessionState::release_owner() noexcept {
  if (!shared_terminal_) {
    if (lifecycle_control_ != nullptr) {
      // The worker normally finalizes this state. A worker-construction
      // failure has no worker cleanup path, so strict RAII performs the same
      // idempotent finalization after wait() proves no run-owned thread can
      // still publish or consume lifecycle records.
      lifecycle_control_->finalize_close();
    }
    delete this;
  }
}

void RunSessionState::publish_startup_failure(
    const std::shared_ptr<const ExecutionPlanState>& attempted_plan,
    RunGenerationId attempted_run_generation) const noexcept {
  RuntimeTerminalSummarySink* terminal_summary_sink =
      attempted_plan != nullptr ? attempted_plan->executable_factories.terminal_summary_sink.get()
                                : nullptr;
  if (attempted_plan != nullptr && attempted_plan->runtime_diagnostic_sink != nullptr) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      attempted_plan->runtime_diagnostic_sink->publish(diagnostics());
      attempted_plan->runtime_diagnostic_sink->publish_runtime_details(
          diagnostic_buffer_.details());
      attempted_plan->runtime_diagnostic_sink->publish_operator_completion_progress({});
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      // Diagnostic-only hooks cannot turn an already-terminal startup failure
      // back into a synchronous exception.
      static_cast<void>(0);
    }
#endif
  }
  publish_terminal_summary(
      attempted_run_generation,
      diagnostics(),
      terminal_summary_sink,
      DiagnosticSource{
          .graph_generation =
              attempted_plan != nullptr ? attempted_plan->graph.generation() : GraphGeneration{},
      });
}

void RunSessionState::start() noexcept {
  if (plan_ != nullptr && plan_->executable_factories.test_managed_signal_control != nullptr &&
      plan_->executable_factories.test_managed_signal_control->consume(
          RuntimeManagedSignalTestPoint::kWorkerStart) != 0) {
    fail_worker_start();
    return;
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    // Keep ordinary sessions on the original entry point. Thread-usage accounting is selected
    // once during cold startup and adds no scope or sampling work to the uninstrumented loop.
    if (phase_counters_ != nullptr) {
      worker_ = std::thread([this] { worker_main_with_thread_usage(); });
    } else {
      worker_ = std::thread([this] { worker_main(); });
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    fail_worker_start();
  }
#endif
}

[[nodiscard]] bool RunSessionState::is_running() const noexcept {
  return termination_.load(std::memory_order_acquire) == RunTermination::kRunning;
}

[[nodiscard]] RunTermination RunSessionState::termination() const noexcept {
  return termination_.load(std::memory_order_acquire);
}

void RunSessionState::request_stop() noexcept {
  if (!shared_terminal_ && publish_termination_request(detail::TerminationRequest::kStop)) {
    request_shutdown();
  }
}

void RunSessionState::request_termination(void* opaque,
                                          detail::TerminationRequest request) noexcept {
  static_cast<RunSessionState*>(opaque)->request_callback_termination(request);
}

[[nodiscard]] bool RunSessionState::publish_termination_request(
    detail::TerminationRequest request) noexcept {
  std::uint64_t observed = termination_state_.load(std::memory_order_acquire);
  for (;;) {
    if ((observed & kTerminationFrozen) != 0U) {
      return false;
    }
    std::uint64_t desired = observed | kCallbackAdmissionClosed;
    switch (request) {
      case detail::TerminationRequest::kCompletion:
        if ((observed & (kRequestedCompletion | kRequestedStop)) != 0U) {
          return false;
        }
        desired |= kRequestedCompletion;
        break;
      case detail::TerminationRequest::kStop:
        if ((observed & kRequestedStop) != 0U) {
          return false;
        }
        desired |= kRequestedStop;
        break;
      case detail::TerminationRequest::kNone:
        return false;
    }
    if (termination_state_.compare_exchange_weak(
            observed, desired, std::memory_order_acq_rel, std::memory_order_acquire)) {
      if (request == detail::TerminationRequest::kStop) {
        // Admission is closed, so a consumer with a full queue may never run again: let blocked
        // publications abort instead of waiting out the graph blocking budget.
        publication_stop_latch_.store(true, std::memory_order_release);
      }
      return true;
    }
  }
}

void RunSessionState::request_callback_termination(detail::TerminationRequest request) noexcept {
  if (shared_terminal_ || !publish_termination_request(request)) {
    return;
  }
  request_shutdown();
}

void RunSessionState::request_shutdown() noexcept {
  if (shared_terminal_) {
    return;
  }
  if (lifecycle_control_ != nullptr) {
    lifecycle_control_->request_stop_ingress();
  }
  stop_requested_.store(true, std::memory_order_release);
  // Every accepted close or completion-to-stop promotion wakes all partition owners. Repeated
  // requests are rejected by the atomic publication above and therefore do not add wake work.
  signal_controller_wake(RuntimePhaseCounter::kStopSignals);
  // Pair this latch and target through the sequentially consistent order so an accepted request
  // racing startup cannot miss the coordinator publication in attach_parameter_control(). The
  // published coordinator moves from GraphExec into retained_parameter_updates_ at shutdown and
  // therefore remains alive for the RunSessionState lifetime. request_close_ingress() is one
  // lock-free atomic store, so callback termination never waits for parameter control or capture.
  parameter_ingress_close_requested_.store(true, std::memory_order_seq_cst);
  if (ParameterUpdateCoordinator* const updates =
          parameter_ingress_close_target_.load(std::memory_order_seq_cst);
      updates != nullptr) {
    updates->request_close_ingress();
  }
}

void RunSessionState::wait() {
  if (runtime_callback_context_restricted()) {
    runtime_log_and_terminate_noexcept([] {
      HOLOSCAN_LOG_ERROR(
          "RUNTIME_SELF_JOIN_FORBIDDEN: RunSession wait attempted from a Runtime "
          "callback context");
    });
  }
  if (shared_terminal_) {
    return;
  }
  std::unique_lock lock(join_mutex_);
  for (;;) {
    if (joined_) {
      return;
    }
    // A completed-but-unjoined std::thread retains its old ID, which the OS
    // may already have reused for a new waiter. The worker-lifetime token
    // identifies the actual run-owned thread without that reuse ambiguity.
    if (current_run_owned_session() == this) {
      runtime_log_and_terminate_noexcept([] {
        HOLOSCAN_LOG_ERROR(
            "RUNTIME_SELF_JOIN_FORBIDDEN: RunSession wait attempted from its run-owned thread");
      });
    }
    if (!joining_) {
      joining_ = true;
      break;
    }

    if (plan_->executable_factories.test_wait_control) {
      plan_->executable_factories.test_wait_control->peer_waiter_entered();
    }
    // A failed physical join returns ownership to one of the waiting callers.
    // Waiting only for `joined_` would strand every peer when std::thread::join
    // reports a system error to the first join owner.
    join_cv_.wait(lock, [this] { return joined_ || !joining_; });
  }
  lock.unlock();
  if (plan_->executable_factories.test_wait_control) {
    plan_->executable_factories.test_wait_control->join_owner_selected();
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    if (worker_.joinable()) {
      worker_.join();
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    lock.lock();
    joining_ = false;
    lock.unlock();
    join_cv_.notify_all();
    throw;
  }
#endif
  if (terminal_participant_ != nullptr) {
    terminal_participant_->wait_for_retirement();
  }
  lock.lock();
  joined_ = true;
  joining_ = false;
  lock.unlock();
  join_cv_.notify_all();
}

bool RunSessionState::wait_for(std::chrono::nanoseconds timeout) {
  if (const auto admission = wait_admission(); !admission) {
    runtime_log_and_terminate_noexcept([] {
      HOLOSCAN_LOG_ERROR(
          "RUNTIME_SELF_JOIN_FORBIDDEN: RunSession bounded wait attempted from a Runtime-owned "
          "thread");
    });
  }
  if (shared_terminal_) {
    return true;
  }
  std::unique_lock lock(join_mutex_);
  const auto terminal = [this] {
    return joined_ || termination_.load(std::memory_order_acquire) != RunTermination::kRunning;
  };
  if (timeout <= std::chrono::nanoseconds::zero()) {
    return terminal();
  }
  if (plan_->executable_factories.test_wait_control) {
    plan_->executable_factories.test_wait_control->bounded_waiter_entered();
  }
  return join_cv_.wait_for(lock, timeout, terminal);
}

expected<void, Error> RunSessionState::wait_admission() const noexcept {
  if (runtime_callback_context_restricted()) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "RUNTIME_SELF_JOIN_FORBIDDEN callback_context"});
  }
  if (shared_terminal_) {
    return {};
  }
  const std::lock_guard lock(join_mutex_);
  if (!joined_ && current_run_owned_session() == this) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "RUNTIME_SELF_JOIN_FORBIDDEN run_owned_thread"});
  }
  return {};
}

[[nodiscard]] std::span<const Diagnostic> RunSessionState::diagnostics() const noexcept {
  return diagnostic_buffer_.diagnostics();
}

std::string RunSessionState::cuda_graph_region_failure_evidence(
    const CUDAGraphRuntimeRegionIdentity& identity, const Error& error) const noexcept {
  std::string evidence;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    if (identity.region) {
      evidence = "region_id=" + std::to_string(identity.region.value);
    }
    if (plan_ != nullptr) {
      const auto* capture =
          find_unique_cuda_graph_capture_plan_by_region(*plan_, identity.graph, identity.region);
      if (capture != nullptr && !capture->region_name.empty()) {
        if (!evidence.empty()) {
          evidence += ',';
        }
        evidence += "region_name=" + percent_encode_uri_component(capture->region_name);
      }
    }
    if (!evidence.empty()) {
      evidence += ',';
    }
    evidence += "error=" + runtime_error_message(error);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    // Region text is optional. Structural graph and fused-operator identity remain authoritative.
    evidence.clear();
  }
#endif
  return evidence;
}

void RunSessionState::append_compute_failure_diagnostic(const RuntimeOperatorInstance& instance,
                                                        const RuntimeDiagnosticDetail& detail,
                                                        const Error& error) noexcept {
  if (instance.fused_cuda_graph_operator == nullptr) {
    append_diagnostic(
        "RUNTIME_OPERATOR_CALLBACK_FAILED", instance.vertex, "operator_compute", &detail);
    return;
  }
  CUDAGraphRuntimeRegionIdentity identity;
  if (plan_ != nullptr) {
    const auto* region = find_unique_cuda_graph_execution_region_by_member(*plan_, instance.vertex);
    if (region != nullptr) {
      identity = {.graph = region->graph,
                  .region = region->region,
                  .fused_operator = region->fused_operator};
    }
  }
  append_diagnostic(kCUDAGraphRuntimeExecutionFailed,
                    instance.vertex,
                    "cuda_graph_execution",
                    &detail,
                    {},
                    cuda_graph_region_failure_evidence(identity, error),
                    ::holoscan::DiagnosticSeverity::kError,
                    identity.graph);
}

void RunSessionState::append_cuda_graph_quiesce_failure(
    const Error& error, RuntimeCudaExecutionDiagnostic diagnostic) noexcept {
  diagnostic.plan_generation = plan_ != nullptr ? plan_->plan_generation : PlanGenerationId{};
  diagnostic.run_generation = run_generation_;
  diagnostic.lifecycle_stage = LifecycleStage::kStop;
  const CUDAGraphRuntimeRegionIdentity identity{.graph = diagnostic.graph,
                                                .region = diagnostic.region,
                                                .fused_operator = diagnostic.fused_operator};
  RuntimeDiagnosticDetail detail{
      .present = true, .error = Error{error.code}, .cuda_execution = std::move(diagnostic)};
  if (plan_ != nullptr && plan_->executable_factories.test_cuda_execution_control != nullptr) {
    plan_->executable_factories.test_cuda_execution_control->notify_failure_observer(
        RuntimeCudaExecutionObserverPoint::kFailureDiagnostic);
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    detail.error.message = error.message;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    // Optional text must not prevent cleanup or terminal failed-session publication.
    detail.error.message.clear();
  }
#endif
  append_diagnostic(kCUDAGraphRuntimeQuiesceFailed,
                    identity.fused_operator,
                    "cuda_graph_quiesce",
                    &detail,
                    {},
                    cuda_graph_region_failure_evidence(identity, error),
                    ::holoscan::DiagnosticSeverity::kError,
                    identity.graph);
}

/// Invoke one steady-state task directly on the already verified inline owner.
///
/// The controller and sole partition runner are the same thread in this
/// specialization. Re-entering ExecutionLaneSet would only reacquire the
/// lane's stop/serialization locks around a callback that cannot migrate.
[[nodiscard]] expected<void, Error> RunSessionState::invoke_inline_owner_task(
    const ExecutionLaneTask& task) noexcept {
  if (task.invoke == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "RUNTIME_LANE_DISPATCH_INVALID"});
  }
  const RuntimeCallbackContextScope callback_scope{task.callback_identity, task.callback_context};
  count_phase(RuntimePhaseCounter::kLaneInlineOwnerInvocations);
  count_phase(RuntimePhaseCounter::kLaneCallbacksEntered);
  struct CallbackExit {
    RunSessionState* session;
    ~CallbackExit() { session->count_phase(RuntimePhaseCounter::kLaneCallbacksExited); }
  } const callback_exit{this};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    return task.invoke(task.context);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::exception& exception) {
    return make_unexpected(Error{ErrorCode::kFailure, exception.what()});
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kFailure, "RUNTIME_INLINE_LANE_CALLBACK_FAILED"});
  }
#endif
}

void RunSessionState::signal_controller_wake(RuntimePhaseCounter counter) noexcept {
  if (phase_counters_ != nullptr) {
    phase_counters_->increment(counter);
    phase_counters_->trace(
        RuntimePhaseTracePoint::kControllerWakeSignaled, kNoRuntimePhaseReceiver, {}, 0U);
  }
  if (partition_wakes_.empty()) {
    readiness_wake_.signal();
    if (local_partition_runner_count_ == 1U) {
      count_phase(RuntimePhaseCounter::kPartitionRunnerWakeSignals);
    }
  } else {
    for (const auto& wake : partition_wakes_) {
      if (wake != nullptr) {
        wake->signal();
        count_phase(RuntimePhaseCounter::kPartitionRunnerWakeSignals);
      }
    }
  }
}

void RunSessionState::signal_channel_owner_progress(VertexId owner) noexcept {
  RuntimeWakeState* const wake = partition_wake_by_vertex_.find(owner);
  if (wake == nullptr) {
    // Materialization proves every process-local endpoint owner is local. Preserve liveness
    // defensively if corrupted private state reaches this noexcept callback.
    signal_controller_wake(RuntimePhaseCounter::kWakeSignals);
    return;
  }
  count_phase(RuntimePhaseCounter::kWakeSignals);
  trace_phase(RuntimePhaseTracePoint::kControllerWakeSignaled, kNoRuntimePhaseReceiver, {}, 0U);
  const RuntimeWakeState* const current = current_partition_owner_wake();
  if (wake == current) {
    // This thread is the destination's owner and rescans before it parks: no token is needed.
    current_partition_owner_rescan() = true;
    if (phase_counters_ != nullptr) {
      phase_counters_->increment(wake->accounting_partition, RuntimePartitionCounter::kSignals);
      phase_counters_->increment(wake->accounting_partition, RuntimePartitionCounter::kSelfSignals);
    }
    return;
  }
  if (phase_counters_ != nullptr) {
    phase_counters_->increment(wake->accounting_partition, RuntimePartitionCounter::kSignals);
    if (current == nullptr) {
      phase_counters_->increment(wake->accounting_partition,
                                 RuntimePartitionCounter::kForeignSignals);
    }
  }
  bool kernel_wake = false;
  if (wake->signal_coalesced(&kernel_wake)) {
    count_phase(RuntimePhaseCounter::kPartitionRunnerWakeSignals);
    if (phase_counters_ != nullptr) {
      phase_counters_->increment(wake->accounting_partition,
                                 RuntimePartitionCounter::kSignalsPublished);
      if (kernel_wake) {
        phase_counters_->increment(wake->accounting_partition,
                                   RuntimePartitionCounter::kKernelWakes);
      }
    }
  }
}

[[nodiscard]] expected<OperatorRef, Error> RunSessionState::operator_ref(
    OperatorRef plan_ref) const noexcept {
  if (!is_running() || plan_ == nullptr || !plan_ref.plan_bound() ||
      plan_ref.plan_generation != plan_->plan_generation) {
    return make_unexpected(Error{ErrorCode::kStaleOperatorRef});
  }
  const bool present =
      std::any_of(plan_->executable_factories.operators.begin(),
                  plan_->executable_factories.operators.end(),
                  [&](const auto& record) {
                    return record.operator_id == plan_ref.id &&
                           worker_owns_vertex(*plan_, process_group_, record.vertex);
                  });
  if (!present) {
    return make_unexpected(Error{ErrorCode::kStaleOperatorRef});
  }
  plan_ref.run_generation = run_generation_;
  return plan_ref;
}

[[nodiscard]] expected<LifecycleControlHandle, Error> RunSessionState::lifecycle_control(
    const LifecycleAccessPolicy& policy) {
  if (lifecycle_control_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return lifecycle_control_->make_handle(policy);
}

[[nodiscard]] expected<ParamControlHandle, Error> RunSessionState::parameter_control(
    const ParamAccessPolicy& policy) {
  if (general_parameter_control_forbidden()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  ParamControlLink* retained_link = nullptr;
  {
    const std::lock_guard lock(parameter_mutex_);
    if (parameter_control_link_ == nullptr) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    auto retained = ParamControlAccess::retain_for_handle(parameter_control_link_, policy);
    if (!retained) {
      return make_unexpected(retained.error());
    }
    retained_link = *retained;
  }
  // Catalog preparation may wait for startup attachment and allocate its
  // owning projection. It must run after releasing parameter_mutex_ so the
  // worker can attach or retire the shared link.
  return ParamControlAccess::make_handle_from_retained(retained_link, policy);
}

[[nodiscard]] expected<ParamStopFenceResult, Error>
RunSessionState::close_parameter_update_ingress() {
  if (general_parameter_control_forbidden()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  parameter_ingress_close_requested_.store(true, std::memory_order_release);
  std::unique_lock lock(parameter_mutex_);
  parameter_cv_.wait(
      lock, [this] { return parameter_updates_ != nullptr || parameter_control_link_ == nullptr; });
  if (parameter_updates_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  parameter_updates_->request_close_ingress();
  return parameter_updates_->close_ingress();
}

[[nodiscard]] expected<ParamSnapshot, Error> RunSessionState::capture_params(
    ParamCaptureOptions options) {
  if (plan_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  if (general_parameter_control_forbidden()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  std::unique_lock lock(parameter_mutex_);
  if (const auto& control = plan_->executable_factories.test_termination_admission_control;
      control != nullptr) {
    control->parameter_mutex_held();
  }
  parameter_cv_.wait(
      lock, [this] { return parameter_updates_ != nullptr || parameter_control_link_ == nullptr; });
  if (parameter_updates_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return parameter_updates_->capture(
      ParamAccessPolicy{
          .origin = ParamControlOrigin::kLocalApplication,
          .permissions = ParamPermission::kRead | ParamPermission::kCapture,
      },
      options);
}

[[nodiscard]] expected<RateGenerationSnapshot, Error> RunSessionState::capture_rate_generations() {
  if (general_parameter_control_forbidden()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  RuntimeRateState* rates = nullptr;
  {
    std::unique_lock lock(rate_snapshot_mutex_);
    rate_snapshot_cv_.wait(lock, [this] {
      return rate_snapshot_state_ != nullptr ||
             termination_.load(std::memory_order_acquire) != RunTermination::kRunning;
    });
    rates = rate_snapshot_state_;
  }
  if (rates == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return rates->capture_snapshot();
}

[[nodiscard]] expected<void, Error> RunSessionState::attach_parameter_control(
    ParameterUpdateCoordinator& updates) {
  const std::lock_guard lock(parameter_mutex_);
  if (parameter_control_link_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  parameter_updates_ = &updates;
  auto attached = ParamControlAccess::attach(
      parameter_control_link_, parameter_updates_, &kParameterControlOperations);
  if (!attached) {
    parameter_updates_ = nullptr;
    parameter_cv_.notify_all();
    return attached;
  }
  parameter_ingress_close_target_.store(parameter_updates_, std::memory_order_seq_cst);
  if (parameter_ingress_close_requested_.load(std::memory_order_seq_cst)) {
    parameter_updates_->request_close_ingress();
  }
  parameter_cv_.notify_all();
  return {};
}

void RunSessionState::attach_rate_snapshot_state(RuntimeRateState& rates) noexcept {
  const std::lock_guard lock(rate_snapshot_mutex_);
  rate_snapshot_state_ = &rates;
  rate_snapshot_cv_.notify_all();
}

void RunSessionState::terminalize_parameter_updates_at_run_end() noexcept {
  parameter_ingress_close_requested_.store(true, std::memory_order_release);
  const std::lock_guard lock(parameter_mutex_);
  if (parameter_updates_ != nullptr) {
    parameter_updates_->terminalize_unpublished_at_run_end();
  }
}

void RunSessionState::retire_parameter_control() noexcept {
  ParamControlLink* link = nullptr;
  {
    const std::lock_guard lock(parameter_mutex_);
    if (parameter_updates_ != nullptr) {
      parameter_updates_->request_close_ingress();
    }
    parameter_updates_ = nullptr;
    link = std::exchange(parameter_control_link_, nullptr);
    parameter_cv_.notify_all();
  }
  ParamControlAccess::detach_and_release(link);
}

void RunSessionState::append_diagnostic(std::string_view token, VertexId vertex,
                                        std::string_view service,
                                        const RuntimeDiagnosticDetail* detail,
                                        std::string_view required, std::string_view provided,
                                        ::holoscan::DiagnosticSeverity severity, GraphId graph,
                                        PortId explicit_port) noexcept {
  const std::lock_guard diagnostic_lock(diagnostic_mutex_);
  const auto graph_view = plan_ != nullptr ? plan_->graph : GraphView{};
  GraphId graph_id = graph;
  if (!graph_id && detail != nullptr) {
    graph_id = detail->component_graph;
  }
  if (!graph_id) {
    for (const auto& candidate : graph_view.vertices()) {
      if (candidate.id == vertex) {
        graph_id = candidate.graph;
        break;
      }
    }
  }
  if (!graph_id) {
    for (const auto& candidate : graph_view.graphs()) {
      if (!candidate.parent) {
        graph_id = candidate.id;
        break;
      }
    }
  }
  PortId effective_port = explicit_port;
  std::string_view port_name;
  const PortId requested_port = requested_diagnostic_port(explicit_port, detail);
  if (plan_ != nullptr && requested_port) {
    for (const auto& candidate : plan_->ports) {
      if (candidate.id == requested_port && candidate.owner == vertex) {
        effective_port = candidate.id;
        port_name = candidate.local_name;
        break;
      }
    }
  }
  diagnostic_buffer_.append(
      diagnostic_id(token),
      token,
      DiagnosticSource{
          .graph_generation = plan_ != nullptr ? plan_->graph.generation() : 0U,
          .graph = graph_id,
          .vertex = vertex,
          .port = effective_port,
      },
      service,
      detail,
      required,
      provided,
      severity,
      graph_view,
      port_name);
}

// Construct the bounded terminal fallback before application code can inject
// or encounter a cold allocation failure. Every wrapper that observes it is
// immutable and release_owner() deliberately leaves this process-lifetime
// tombstone intact.
RunSessionState* const kStartupFailureSessionState = RunSessionState::shared_startup_failure();

}  // namespace detail

namespace detail {

class RunSessionAccess {
 public:
  [[nodiscard]] static RunSession make(RunSessionState* state) noexcept {
    return RunSession{state};
  }
};

RunSession make_run_session(RunSessionState* state) noexcept {
  return RunSessionAccess::make(state);
}

[[nodiscard]] RunSession start_run_session_retained(
    const std::shared_ptr<const ExecutionPlanState>& retained,
    const std::optional<ProcessGroupId> process_group,
    const std::shared_ptr<RuntimeTerminalParticipant>& terminal_participant,
    const std::shared_ptr<RuntimeFixtureObservation>& observation) {
  const ProcessGroupResolution resolution = resolve_process_group(*retained, process_group);
  RunGenerationId attempted_run_generation{};
  std::unique_ptr<RunSessionState> state;
  [[maybe_unused]] bool participant_attached = false;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    if (resolution.disposition == ProcessGroupResolutionDisposition::kSelectionRequired) {
      if (terminal_participant != nullptr) {
        terminal_participant->abort_before_session();
      }
      attempted_run_generation = reserve_run_generation();
      auto state = std::make_unique<RunSessionState>(
          retained, attempted_run_generation, RunSessionState::ProcessGroupSelectionFailureTag{});
      return make_run_session(state.release());
    }
    if (resolution.disposition == ProcessGroupResolutionDisposition::kInvalid) {
      if (terminal_participant != nullptr) {
        terminal_participant->abort_before_session();
      }
      kStartupFailureSessionState->publish_startup_failure(retained);
      return make_run_session(kStartupFailureSessionState);
    }
    attempted_run_generation = reserve_run_generation();
    state = std::make_unique<RunSessionState>(
        retained, attempted_run_generation, resolution.process_group);
    state->attach_fixture_observation(observation);
    state->initialize_after_progress_materialization();
    // Attach last: a persistent signal domain may call request_stop() from its broker as soon as
    // the participant is attached, and that reads the lifecycle state initialized above.
    state->attach_terminal_participant(terminal_participant);
    participant_attached = true;
    state->start();
    return make_run_session(state.release());
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    if (!participant_attached && terminal_participant != nullptr) {
      terminal_participant->abort_before_session();
    }
    if (state != nullptr) {
      state->fail_startup();
      return make_run_session(state.release());
    }
    kStartupFailureSessionState->publish_startup_failure(retained, attempted_run_generation);
    return make_run_session(kStartupFailureSessionState);
  }
#endif
}

void RunSessionState::attach_fixture_observation(
    std::shared_ptr<RuntimeFixtureObservation> observation) {
  if (observation != nullptr && !observation->bind(plan_->plan_generation, run_generation_)) {
    throw std::invalid_argument{"operator test observation must belong to one fresh run"};
  }
  fixture_observation_ = std::move(observation);
}

void RunSessionState::observe_input_ingestion(VertexId owner,
                                              const FrozenRuntimeCandidate& candidate,
                                              const TemporalSampleRecord& sample) noexcept {
  const std::array inputs{
      FixtureObservedInput{.input = candidate.witness.port,
                           .selection = {InputState::kSelected, SelectionReason::kNone},
                           .sample = sample,
                           .has_sample = true}};
  static_cast<void>(fixture_observation_->append(
      FixtureObservationRecord{.kind = FixtureObservationKind::kInputIngested,
                               .owner = owner,
                               .witness = candidate.witness,
                               .tag = candidate.candidate.tag},
      inputs,
      {}));
}

RunSession start_operator_test_run(const ExecutionPlan& plan,
                                   const std::shared_ptr<RuntimeFixtureObservation>& observation) {
  plan.require_no_errors();
  const auto retained = ExecutionPlanAccess::retain(plan);
  if (observation == nullptr || retained->execution_partitions.size() != 1U ||
      !retained->closed_island_activations.empty() ||
      !retained->cuda_graph_execution_regions.empty()) {
    throw std::invalid_argument{
        "controlled operator tests require one ordinary, non-composite execution partition"};
  }
  return start_run_session_retained(retained, {}, {}, observation);
}

[[nodiscard]] RunSession start_run_session(
    const ExecutionPlan& plan, const std::optional<ProcessGroupId> process_group,
    const std::shared_ptr<RuntimeTerminalParticipant>& terminal_participant = {}) {
  plan.require_no_errors();
  return start_run_session_retained(
      ExecutionPlanAccess::retain(plan), process_group, terminal_participant);
}

RunSession run_async_worker(const ExecutionPlan& plan, ProcessGroupId process_group) {
  return start_run_session(plan, process_group);
}

[[nodiscard]] std::span<const OperatorCompletionProgressRecord>
RunSessionState::operator_completion_progress() const noexcept {
  return termination_.load(std::memory_order_acquire) == RunTermination::kRunning
             ? std::span<const OperatorCompletionProgressRecord>{}
             : std::span<const OperatorCompletionProgressRecord>{operator_completion_progress_};
}

}  // namespace detail

RunSession::RunSession(detail::RunSessionState* state) noexcept : state_(state) {}

RunSession::RunSession(RunSession&& other) noexcept
    : state_(std::exchange(other.state_, nullptr)) {}

RunSession::~RunSession() noexcept {
  if (state_ != nullptr) {
    state_->request_stop();
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      state_->wait();
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      std::terminate();
    }
#endif
    state_->release_owner();
  }
}

bool RunSession::is_running() const noexcept {
  return state_ != nullptr && state_->is_running();
}

RunTermination RunSession::termination() const noexcept {
  return state_ != nullptr ? state_->termination() : RunTermination::kStopped;
}

void RunSession::request_stop() noexcept {
  if (state_ != nullptr) {
    state_->request_stop();
  }
}

void RunSession::wait() {
  if (state_ != nullptr) {
    state_->wait();
  }
}

bool RunSession::wait_for(std::chrono::nanoseconds timeout) {
  return state_ == nullptr || state_->wait_for(timeout);
}

expected<void, Error> RunSession::wait_admission() const noexcept {
  return state_ != nullptr ? state_->wait_admission() : expected<void, Error>{};
}

std::span<const Diagnostic> RunSession::diagnostics() const noexcept {
  return state_ != nullptr ? state_->diagnostics() : std::span<const Diagnostic>{};
}

std::span<const OperatorCompletionProgressRecord> RunSession::operator_completion_progress()
    const noexcept {
  return state_ != nullptr ? state_->operator_completion_progress()
                           : std::span<const OperatorCompletionProgressRecord>{};
}

expected<OperatorRef, Error> RunSession::operator_ref(OperatorRef plan_ref) const noexcept {
  if (state_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kStaleOperatorRef});
  }
  return state_->operator_ref(plan_ref);
}

expected<LifecycleControlHandle, Error> RunSession::lifecycle_control(
    const LifecycleAccessPolicy& policy) {
  if (state_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return state_->lifecycle_control(policy);
}

expected<ParamControlHandle, Error> RunSession::parameter_control(const ParamAccessPolicy& policy) {
  if (state_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return state_->parameter_control(policy);
}

expected<ParamStopFenceResult, Error> RunSession::close_parameter_update_ingress() {
  if (state_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return state_->close_parameter_update_ingress();
}

expected<ParamSnapshot, Error> RunSession::capture_params(ParamCaptureOptions options) {
  if (state_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return state_->capture_params(options);
}

expected<RateGenerationSnapshot, Error> RunSession::capture_rate_generations() {
  if (state_ == nullptr) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  return state_->capture_rate_generations();
}

RunSession run_async(const ExecutionPlan& plan) {
  return detail::start_run_session(plan, std::nullopt);
}

}  // namespace holoscan
