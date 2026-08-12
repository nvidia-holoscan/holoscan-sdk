/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/run.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <holoscan/logger/logger.hpp>

#include "../compiler/execution_plan_state.hpp"
#include "../core/diagnostic_registry.hpp"
#include "../core/execution_context_internal.hpp"
#include "../core/operator_runtime_access.hpp"
#include "../core/operator_spec_compiler_access.hpp"
#include "../core/source_id_internal.hpp"
#include "../core/temporal_contract_internal.hpp"
#include "dispatcher.hpp"
#include "endpoint_clock_health_runtime.hpp"
#include "execution_lane.hpp"
#include "graph_exec.hpp"
#include "lifecycle_control_state.hpp"
#include "parameter_control_state.hpp"
#include "predicate_evaluator.hpp"
#include "runtime_callback_context.hpp"
#include "runtime_channel_binding.hpp"
#include "runtime_diagnostic_sink.hpp"
#include "runtime_direct_composite.hpp"
#include "runtime_notification_source.hpp"
#include "runtime_phase_counters.hpp"
#include "runtime_support_thread_control.hpp"
#include "runtime_tensor_resources.hpp"
#include "runtime_termination.hpp"
#include "runtime_test_access.hpp"
#include "runtime_worker.hpp"
#include "source_wait.hpp"

namespace holoscan {
namespace detail {
namespace {

std::atomic<RunGenerationId> g_next_run_generation{1U};
constinit thread_local const void* g_run_owned_session = nullptr;

inline constexpr std::array<std::string_view, 11> kRuntimeFallbackTokens{{
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

[[nodiscard]] std::string_view registered_token_or(std::string_view candidate,
                                                   std::string_view fallback) noexcept {
  return !candidate.empty() && diagnostic_id(candidate) ? candidate : fallback;
}

[[nodiscard]] constexpr std::string_view channel_terminal_token(
    const std::optional<StructuralDiagnosticId> diagnostic) noexcept {
  if (!diagnostic) {
    return "RUNTIME_READINESS_BINDING_UNAVAILABLE";
  }
  if (const auto* descriptor = find_structural_diagnostic(*diagnostic); descriptor != nullptr) {
    return descriptor->token;
  }
  // A new Channel producer must register its typed structural identity instead of silently
  // degrading to a generic readiness failure.
  return "COMPILER_DIAGNOSTIC_TOKEN_UNREGISTERED";
}

[[nodiscard]] std::string replay_digest_text(ParamSchemaHash digest) {
  constexpr std::string_view kHexDigits{"0123456789abcdef"};
  std::string text{"sha256:"};
  text.reserve(7U + digest.value.size() * 2U);
  for (const std::byte byte : digest.value) {
    const auto value = static_cast<std::uint8_t>(byte);
    text.push_back(kHexDigits[value >> 4U]);
    text.push_back(kHexDigits[value & 0x0FU]);
  }
  return text;
}

class RunOwnedThreadScope final {
 public:
  explicit RunOwnedThreadScope(const void* owner) noexcept : previous_owner_(g_run_owned_session) {
    g_run_owned_session = owner;
  }

  ~RunOwnedThreadScope() noexcept { g_run_owned_session = previous_owner_; }

 private:
  const void* previous_owner_;
};

class ScopedComputeCudaDevice final {
 public:
  explicit ScopedComputeCudaDevice(const DeviceId requested) noexcept {
    if (!requested.valid()) {
      active_ = true;
      restored_ = true;
      return;
    }
    if (cudaGetDevice(&previous_) != cudaSuccess) {
      return;
    }
    if (previous_ != requested.value) {
      if (cudaSetDevice(requested.value) != cudaSuccess) {
        return;
      }
      changed_ = true;
    }
    active_ = true;
  }

  ScopedComputeCudaDevice(const ScopedComputeCudaDevice&) = delete;
  ScopedComputeCudaDevice& operator=(const ScopedComputeCudaDevice&) = delete;

  ~ScopedComputeCudaDevice() {
    if (!restored_ && changed_) {
      static_cast<void>(cudaSetDevice(previous_));
    }
  }

  [[nodiscard]] bool active() const noexcept { return active_; }

  /// Restore the lane's prior ambient device after user code, including when the requested device
  /// already matched on entry but the callback changed it. Host-only callbacks take the invalid
  /// DeviceId fast path and make no CUDA calls.
  [[nodiscard]] bool restore_after_callback() noexcept {
    if (restored_) {
      return true;
    }
    int current{};
    if (cudaGetDevice(&current) != cudaSuccess) {
      return false;
    }
    if (current != previous_) {
      changed_ = true;
      if (cudaSetDevice(previous_) != cudaSuccess) {
        return false;
      }
    }
    restored_ = true;
    changed_ = false;
    return true;
  }

 private:
  int previous_{};
  bool changed_{};
  bool active_{};
  bool restored_{};
};

[[nodiscard]] RunGenerationId reserve_run_generation() {
  RunGenerationId generation = g_next_run_generation.load(std::memory_order_relaxed);
  while (generation != std::numeric_limits<RunGenerationId>::max()) {
    if (g_next_run_generation.compare_exchange_weak(
            generation, generation + 1U, std::memory_order_relaxed)) {
      return generation;
    }
  }
  throw RuntimeError{Error{ErrorCode::kResourceExhausted, "RUNTIME_RUN_GENERATION_EXHAUSTED"}};
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
    HOLOSCAN_LOG_ERROR(
        "RunSession failed: run_generation={} category=failed diagnostics={} first_diagnostic={} "
        "first_token={} graph_generation={} graph={} vertex={} port={}",
        summary.run_generation,
        summary.diagnostic_count,
        summary.first_diagnostic.value,
        summary.first_token,
        summary.first_source.graph_generation,
        summary.first_source.graph.value,
        summary.first_source.vertex.value,
        summary.first_source.port.value());
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
  if (plan.graph.vertices().empty()) {
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

[[nodiscard]] const ReadinessPlanRecord* readiness_for(const ExecutionPlanState& plan,
                                                       VertexId vertex) noexcept {
  const auto found = std::find_if(plan.readiness.begin(),
                                  plan.readiness.end(),
                                  [&](const auto& row) { return row.owner == vertex; });
  return found == plan.readiness.end() ? nullptr : &*found;
}

[[nodiscard]] VertexId readiness_owner_for_source(const ExecutionPlanState& plan,
                                                  SourceId source) noexcept {
  const auto found = std::ranges::find(plan.readiness, source, &ReadinessPlanRecord::source_id);
  return found != plan.readiness.end() ? found->owner : VertexId{};
}

[[nodiscard]] RuntimeEventKind runtime_event_kind(ReadinessSourceKind kind) noexcept {
  switch (kind) {
    case ReadinessSourceKind::kReadable:
      return RuntimeEventKind::kReadable;
    case ReadinessSourceKind::kNotification:
      return RuntimeEventKind::kNotification;
    case ReadinessSourceKind::kDoorbell:
      return RuntimeEventKind::kDoorbell;
    case ReadinessSourceKind::kSpin:
      return RuntimeEventKind::kSpin;
  }
  return RuntimeEventKind::kSourceError;
}

[[nodiscard]] const TemporalTriggerPlanRecord* temporal_trigger_for(
    const ExecutionPlanState& plan, OperatorInstanceId owner) noexcept {
  const auto found = std::find_if(plan.temporal_triggers.begin(),
                                  plan.temporal_triggers.end(),
                                  [&](const auto& row) { return row.owner.id == owner; });
  return found == plan.temporal_triggers.end() ? nullptr : &*found;
}

[[nodiscard]] const ClockTriggerPlanRecord* clock_trigger_for(const ExecutionPlanState& plan,
                                                              OperatorRef owner) noexcept {
  const auto found = std::find_if(plan.clock_triggers.begin(),
                                  plan.clock_triggers.end(),
                                  [&](const auto& row) { return row.owner == owner; });
  return found == plan.clock_triggers.end() ? nullptr : &*found;
}

[[nodiscard]] const PredicateTriggerPlanRecord* predicate_trigger_for(
    const ExecutionPlanState& plan, OperatorRef owner) noexcept {
  const auto found = std::find_if(plan.predicate_triggers.begin(),
                                  plan.predicate_triggers.end(),
                                  [&](const auto& row) { return row.owner == owner; });
  return found == plan.predicate_triggers.end() ? nullptr : &*found;
}

[[nodiscard]] const PredicateCallablePlanRecord* predicate_callable_for(
    const ExecutionPlanState& plan, std::string_view predicate_id) noexcept {
  const auto found = std::find_if(
      plan.predicate_callables.begin(), plan.predicate_callables.end(), [&](const auto& row) {
        return row.predicate_id == predicate_id;
      });
  return found == plan.predicate_callables.end() ? nullptr : &*found;
}

[[nodiscard]] expected<std::optional<std::uint32_t>, Error> bind_runtime_derived_term(
    const RuntimeOperatorInstance& instance, ParameterDerivedTermKind kind, PortId input,
    std::int64_t initial_value_ns) noexcept {
  std::optional<std::uint32_t> matched;
  bool found{};
  for (std::size_t index = 0U; index < instance.parameter_derived_terms.size(); ++index) {
    const ParameterDerivedTermPlanRecord& term = instance.parameter_derived_terms[index];
    if (term.kind != kind || term.input != input) {
      continue;
    }
    if (found || index > std::numeric_limits<std::uint32_t>::max() ||
        term.owner.id != instance.operator_id || term.owner.run_generation != 0U ||
        term.initial_value_ns != initial_value_ns) {
      return make_unexpected(Error{ErrorCode::kIncompatible});
    }
    found = true;
    switch (term.update_disposition) {
      case ParameterDerivedTermUpdateDisposition::kRuntimeAtomic:
        if (!instance.runtime_parameter_updates_enabled ||
            term.mutability != ParamMutability::kRuntime) {
          return make_unexpected(Error{ErrorCode::kIncompatible});
        }
        matched = static_cast<std::uint32_t>(index);
        break;
      case ParameterDerivedTermUpdateDisposition::kFrozenCompileTime:
      case ParameterDerivedTermUpdateDisposition::kRecompileRequired:
        break;
    }
  }
  return matched;
}

[[nodiscard]] bool input_trigger_kind(TemporalRecordTriggerKind kind) noexcept {
  return kind == TemporalRecordTriggerKind::kAnyInput || kind == TemporalRecordTriggerKind::kEach ||
         kind == TemporalRecordTriggerKind::kAligned || kind == TemporalRecordTriggerKind::kWindow;
}

class ClockTriggerPortBinding final : public PortBinding {
 public:
  [[nodiscard]] static expected<std::unique_ptr<ClockTriggerPortBinding>, Error> create(
      RuntimePortBindingRequest request, ClockRef clock, TimePoint phase,
      std::chrono::nanoseconds period, bool recapture_phase_on_restart) {
    if (request.plan_generation == 0U || request.run_generation == 0U ||
        request.kind != RuntimeEventKind::kClockTick || !request.source || request.port ||
        request.source_clock != clock.id() || request.reference_clock != clock.id() ||
        request.source_to_reference_conversion_id != 0U || request.capacity != 1U ||
        request.source_generation == 0U || request.initial_source_sequence != 0U ||
        !phase.valid() || phase.clock_id != clock.id() || period.count() <= 0) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "RUNTIME_CLOCK_BINDING_UNAVAILABLE"});
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      return std::unique_ptr<ClockTriggerPortBinding>{new ClockTriggerPortBinding{
          request, std::move(clock), phase, period, recapture_phase_on_restart}};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      return make_unexpected(
          Error{ErrorCode::kResourceExhausted, "RUNTIME_CLOCK_BINDING_UNAVAILABLE"});
    }
#endif
  }

  [[nodiscard]] expected<std::size_t, Error> drain(
      std::span<RuntimeEvent> destination) noexcept override {
    if (terminal_error_) {
      return make_unexpected(*terminal_error_);
    }
    if (destination.empty()) {
      return 0U;
    }
    auto now = clock_.now();
    if (!now) {
      return make_unexpected(std::move(now.error()));
    }
    auto due = schedule_.due(now.value());
    if (!due) {
      terminal_error_ = due.error();
      return make_unexpected(std::move(due.error()));
    }
    const auto firing = due.value();
    if (!firing) {
      return 0U;
    }
    const TimePoint scheduled_time = firing->scheduled_time;
    if (source_sequence_ == std::numeric_limits<std::uint64_t>::max()) {
      terminal_error_ = Error{ErrorCode::kResourceExhausted};
      return make_unexpected(*terminal_error_);
    }
    ++source_sequence_;
    destination.front() = RuntimeEvent{
        .kind = RuntimeEventKind::kClockTick,
        .source = request_.source,
        .port = {},
        .source_decision_time = scheduled_time,
        .reference_time = scheduled_time,
        .source_generation = request_.source_generation,
        .source_sequence = source_sequence_,
    };
    if (scheduled_time.timestamp_ns > std::numeric_limits<std::int64_t>::max() - period_.count()) {
      next_wake_.reset();
      terminal_error_ = Error{ErrorCode::kResourceExhausted};
    } else {
      next_wake_ = TimePoint{scheduled_time.timestamp_ns + period_.count(), clock_.id()};
    }
    return 1U;
  }

  [[nodiscard]] std::size_t capacity() const noexcept override { return 1U; }
  [[nodiscard]] bool closed() const noexcept override { return false; }
  [[nodiscard]] SourceId source() const noexcept override { return request_.source; }
  [[nodiscard]] std::optional<TimePoint> next_wake() const noexcept override { return next_wake_; }

  [[nodiscard]] expected<TimePoint, Error> restart_phase() const noexcept {
    return recapture_phase_on_restart_ ? clock_.now() : expected<TimePoint, Error>{explicit_phase_};
  }

  [[nodiscard]] std::uint64_t publish_restart(TimePoint phase) noexcept {
    ++request_.source_generation;
    source_sequence_ = 0U;
    schedule_ = CoalescingClockSchedule{phase, period_};
    next_wake_ = phase;
    terminal_error_.reset();
    return request_.source_generation;
  }

 private:
  ClockTriggerPortBinding(RuntimePortBindingRequest request, ClockRef clock, TimePoint phase,
                          std::chrono::nanoseconds period, bool recapture_phase_on_restart) noexcept
      : request_(request),
        clock_(std::move(clock)),
        explicit_phase_(phase),
        period_(period),
        schedule_(phase, period),
        next_wake_(phase),
        recapture_phase_on_restart_(recapture_phase_on_restart) {}

  RuntimePortBindingRequest request_;
  ClockRef clock_;
  TimePoint explicit_phase_;
  std::chrono::nanoseconds period_{};
  CoalescingClockSchedule schedule_;
  std::optional<TimePoint> next_wake_;
  std::optional<Error> terminal_error_;
  std::uint64_t source_sequence_{};
  bool recapture_phase_on_restart_{};
};

/// Source-tree-only finite Spin witness used by RuntimeTestPlanAccess.
///
/// The positive witness count lives only in the process-local executable inventory: it is not
/// serialized, hashed, installed, or accepted from a production plan artifact. Keep at most one
/// witness resident so an arbitrary finite test count cannot exceed the plan-admitted source
/// capacity. Retirement publishes the next sequence, preserving the ordinary Runtime credit path.
class TestDirectPortBinding final : public PortBinding {
 public:
  TestDirectPortBinding(const RuntimePortBindingRequest& request, TimePoint coordinate,
                        std::uint32_t witness_count) noexcept
      : source_(request.source),
        port_(request.port),
        coordinate_(coordinate),
        kind_(request.kind),
        source_generation_(request.source_generation),
        next_sequence_(request.initial_source_sequence),
        remaining_(witness_count),
        capacity_(request.capacity) {
    publish_next();
  }

  [[nodiscard]] expected<std::size_t, Error> drain(
      std::span<RuntimeEvent> destination) noexcept override {
    if (!pending_ || destination.empty()) {
      return 0U;
    }
    destination.front() = event_;
    pending_ = false;
    outstanding_sequence_ = event_.source_sequence;
    return 1U;
  }

  [[nodiscard]] SourceId source() const noexcept override { return source_; }

  [[nodiscard]] expected<void, Error> retire(std::uint64_t source_sequence) noexcept override {
    if (outstanding_sequence_ == 0U || source_sequence != outstanding_sequence_) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
    outstanding_sequence_ = 0U;
    if (remaining_ == 0U) {
      closed_ = true;
    } else {
      publish_next();
    }
    return {};
  }

  void cancel() noexcept override {
    pending_ = false;
    outstanding_sequence_ = 0U;
    remaining_ = 0U;
    closed_ = true;
  }

  [[nodiscard]] expected<std::uint64_t, Error> advance_source_generation() noexcept override {
    if (!closed_ || pending_ || outstanding_sequence_ != 0U || remaining_ != 0U) {
      return make_unexpected(Error{ErrorCode::kBackpressured});
    }
    if (source_generation_ == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    ++source_generation_;
    next_sequence_ = 0U;
    return source_generation_;
  }

  [[nodiscard]] std::size_t capacity() const noexcept override { return capacity_; }
  [[nodiscard]] bool closed() const noexcept override { return closed_; }

 private:
  void publish_next() noexcept {
    ++next_sequence_;
    --remaining_;
    event_ = RuntimeEvent{
        .kind = kind_,
        .source = source_,
        .port = port_,
        .source_decision_time = coordinate_,
        .reference_time = coordinate_,
        .source_generation = source_generation_,
        .source_sequence = next_sequence_,
    };
    pending_ = true;
  }

  SourceId source_{};
  PortId port_{};
  TimePoint coordinate_{};
  RuntimeEventKind kind_{RuntimeEventKind::kSpin};
  std::uint64_t source_generation_{};
  std::uint64_t next_sequence_{};
  std::uint64_t outstanding_sequence_{};
  std::uint32_t remaining_{};
  std::size_t capacity_{};
  RuntimeEvent event_{};
  bool pending_{};
  bool closed_{};
};

class TestDirectPortBindingFactory final : public RuntimePortBindingFactory {
 public:
  explicit TestDirectPortBindingFactory(std::uint32_t witness_count) noexcept
      : witness_count_(witness_count) {}

  [[nodiscard]] expected<std::unique_ptr<PortBinding>, Error> bind(
      const RuntimePortBindingRequest& request, TimePoint start_coordinate) noexcept override {
    if (witness_count_ == 0U || request.plan_generation == 0U || request.run_generation == 0U ||
        request.kind != RuntimeEventKind::kSpin || !request.source ||
        !request.source_clock.valid() || request.source_clock != request.reference_clock ||
        request.source_to_reference_conversion_id != 0U || request.capacity == 0U ||
        request.source_generation == 0U || start_coordinate.clock_id != request.reference_clock ||
        request.initial_source_sequence >
            std::numeric_limits<std::uint64_t>::max() - witness_count_) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "RUNTIME_READINESS_BINDING_UNAVAILABLE"});
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      return std::unique_ptr<PortBinding>{
          std::make_unique<TestDirectPortBinding>(request, start_coordinate, witness_count_)};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      return make_unexpected(
          Error{ErrorCode::kResourceExhausted, "RUNTIME_READINESS_BINDING_UNAVAILABLE"});
    }
#endif
  }

 private:
  std::uint32_t witness_count_{};
};

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

struct ComputeParameterIngress {
  ParameterUpdateCoordinator* coordinator{};
  OperatorRef self{};
};

[[nodiscard]] ParamSubmitResult submit_compute_parameter_update(void* state,
                                                                const ParamInlinePatch* patches,
                                                                std::size_t patch_count) noexcept {
  auto* const ingress = static_cast<ComputeParameterIngress*>(state);
  if (ingress == nullptr || ingress->coordinator == nullptr || patches == nullptr ||
      patch_count == 0U) {
    return ParamSubmitResult{
        .status = ParamUpdateStatus::kRejected,
        .state = ParamUpdateState::kRejected,
    };
  }
  for (std::size_t index = 0U; index < patch_count; ++index) {
    if (patches[index].target.operator_instance != ingress->self) {
      return ParamSubmitResult{
          .status = ParamUpdateStatus::kUnsupportedAtomicScope,
          .state = ParamUpdateState::kRejected,
      };
    }
  }
  return ingress->coordinator->try_submit_from_compute(patches, patch_count);
}

void publish_pending_parameter_updates(ParameterUpdateCoordinator& coordinator, OperatorRef owner) {
  const ParamUpdateSequence accepted_cutoff = coordinator.publication_frontier();
  static_cast<void>(coordinator.publish_next_at_callback_boundary(owner, accepted_cutoff));
}

class ActivationParameterBinding final {
 public:
  ActivationParameterBinding(ParameterUpdateCoordinator& coordinator,
                             const PinnedParameterHeader& static_header) noexcept
      : coordinator_(&coordinator), header_(&static_header) {}
  ActivationParameterBinding(const ActivationParameterBinding&) = delete;
  ActivationParameterBinding& operator=(const ActivationParameterBinding&) = delete;
  ~ActivationParameterBinding() { release(); }

  void bind(ParameterGenerationPin pin) noexcept {
    pin_.emplace(std::move(pin));
    header_ = pin_->header();
  }

  void release() noexcept {
    if (pin_.has_value()) {
      coordinator_->release(*pin_);
      pin_.reset();
    }
  }

  [[nodiscard]] const PinnedParameterHeader* header() const noexcept { return header_; }
  [[nodiscard]] const ParameterGenerationPin* pin() const noexcept {
    return pin_.has_value() ? &*pin_ : nullptr;
  }

 private:
  ParameterUpdateCoordinator* coordinator_{};
  const PinnedParameterHeader* header_{};
  std::optional<ParameterGenerationPin> pin_;
};

}  // namespace

class RunSessionState final {
  enum class ComputeAdmissionDisposition : std::uint8_t {
    kPending = 0,
    kAdmitted = 1,
    kDeadlineReadFailed = 2,
    kDeadlineEvaluationFailed = 3,
    kDeadlineFailRun = 4,
    kParameterObservationFailed = 5,
    kCudaDeviceUnavailable = 6,
  };

  struct ComputeLaneInvocation {
    RuntimeOperatorInstance* instance{};
    ExecutionContextState* context{};
    Dispatcher* dispatcher{};
    const DispatcherSelection* selection{};
    const RuntimeDeadlineBindingRecord* deadline{};
    ClockRef deadline_clock{};
    TimePoint source_decision_time{};
    TimePoint reference_time{};
    std::int64_t deadline_budget_ns{};
    ParameterUpdateCoordinator* parameter_updates{};
    OperatorRef self{};
    const ParameterGenerationPin* parameter_generation{};
    std::uint64_t source_sequence{};
    DeviceId cuda_device{};
    RuntimePhaseCounters* phase_counters{};
    RuntimeChannelSession* channels{};
    VertexId channel_owner{};
    std::uint32_t execution_lane{};
    ClockRef direct_continuation_clock{};
    std::optional<DeadlineGateResult> deadline_gate;
    std::optional<RuntimeChannelDirectContinuationReceipt> direct_continuation_receipt;
    Error admission_error{ErrorCode::kSuccess};
    Error direct_continuation_error{ErrorCode::kSuccess};
    ComputeAdmissionDisposition admission{ComputeAdmissionDisposition::kPending};
    bool completed_selection{};
    bool direct_continuation_enabled{};
    bool direct_continuation_clock_failed{};
  };

  template <bool TraceCallback>
  [[nodiscard]] static expected<void, Error> invoke_compute_lane(void* state) noexcept {
    auto& task = *static_cast<ComputeLaneInvocation*>(state);
    // Device selection is callback admission, not callback work. Complete it before the final
    // deadline sample and parameter-generation observation so a failed selection cannot publish
    // callback evidence or hide selection latency.
    ScopedComputeCudaDevice selected_device{task.cuda_device};
    if (!selected_device.active()) {
      task.admission = ComputeAdmissionDisposition::kCudaDeviceUnavailable;
      return {};
    }
    // The accepted lane and process-local dispatch token are both held here. Sample the final
    // policy coordinate before observation commit and user compute so lane handoff and every
    // earlier same- or cross-Operator delay remain measurable.
    if (task.deadline != nullptr) {
      auto dispatch_time = task.deadline_clock.now();
      if (!dispatch_time) {
        task.admission = ComputeAdmissionDisposition::kDeadlineReadFailed;
        return {};
      }
      auto gate = evaluate_runtime_deadline(*task.deadline,
                                            task.source_decision_time,
                                            task.reference_time,
                                            *dispatch_time,
                                            std::chrono::nanoseconds{task.deadline_budget_ns});
      if (!gate) {
        task.admission_error = std::move(gate.error());
        task.admission = ComputeAdmissionDisposition::kDeadlineEvaluationFailed;
        return {};
      }
      task.deadline_gate.emplace(*gate);
      task.context->activation = merge_deadline_activation_status(task.context->activation, *gate);
      if (gate->disposition == DeadlineGateDisposition::kFailRun) {
        task.admission = ComputeAdmissionDisposition::kDeadlineFailRun;
        return {};
      }
    }
    if (task.parameter_generation != nullptr) {
      auto observed = task.parameter_updates->commit_activation_observation(
          task.self, *task.parameter_generation);
      if (!observed) {
        task.admission = ComputeAdmissionDisposition::kParameterObservationFailed;
        return {};
      }
    }
    task.admission = ComputeAdmissionDisposition::kAdmitted;
    expected<void, Error> result;
    if constexpr (TraceCallback) {
      task.phase_counters->trace(RuntimePhaseTracePoint::kComputeCallbackEntered,
                                 task.selection->receiver_index,
                                 task.selection->ready.stable_source_id,
                                 task.source_sequence);
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      const ParamGeneralControlContextScope general_control_scope;
      result = OperatorRuntimeAccess::invoke(*task.instance->op, *task.context);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      result = make_unexpected(Error{ErrorCode::kFailure});
    }
#endif
    if constexpr (TraceCallback) {
      task.phase_counters->trace(RuntimePhaseTracePoint::kComputeCallbackReturned,
                                 task.selection->receiver_index,
                                 task.selection->ready.stable_source_id,
                                 task.source_sequence);
    }
    if (!selected_device.restore_after_callback()) {
      result = make_unexpected(
          Error{ErrorCode::kNotSupported, "RUNTIME_COMPUTE_CUDA_DEVICE_RESTORE_FAILED"});
    }
    task.completed_selection = task.dispatcher->complete(*task.selection).has_value();
    if (result && task.completed_selection && task.direct_continuation_enabled &&
        task.channels != nullptr && task.direct_continuation_clock.id().valid()) {
      auto observed_at = task.direct_continuation_clock.now();
      if (!observed_at) {
        task.direct_continuation_error = std::move(observed_at.error());
        task.direct_continuation_clock_failed = true;
      } else {
        auto receipt = task.channels->poll_direct_continuation(
            task.channel_owner, task.execution_lane, *observed_at);
        if (!receipt) {
          task.direct_continuation_error = std::move(receipt.error());
        } else {
          task.direct_continuation_receipt = std::move(*receipt);
        }
      }
    }
    return result;
  }

 public:
  struct SharedStartupFailureTag {};
  struct ProcessGroupSelectionFailureTag {};

  RunSessionState(std::shared_ptr<const ExecutionPlanState> plan, RunGenerationId run_generation,
                  ProcessGroupId process_group)
      : plan_(std::move(plan)),
        run_generation_(run_generation),
        process_group_(process_group),
        lifecycle_control_(
            std::make_shared<LifecycleControlState>(plan_, run_generation_, process_group_)),
        phase_counters_(plan_->executable_factories.phase_counters.get()),
        compute_lane_invoke_(phase_counters_ != nullptr ? &invoke_compute_lane<true>
                                                        : &invoke_compute_lane<false>),
        diagnostic_buffer_(diagnostic_id("RUNTIME_DIAGNOSTIC_CAPACITY_EXHAUSTED"),
                           DiagnosticSource{.graph_generation = plan_->graph.generation()}) {
    if (plan_ == nullptr ||
        plan_->operator_parameters.size() > std::numeric_limits<OperatorInstanceId>::max()) {
      throw RuntimeError{Error{ErrorCode::kIncompatible}};
    }
    local_partition_runner_count_ =
        static_cast<std::size_t>(std::count_if(plan_->partition_runners.begin(),
                                               plan_->partition_runners.end(),
                                               [&](const PartitionRunnerPlanRecord& runner) {
                                                 return runner.process_group == process_group_;
                                               }));
    if (local_partition_runner_count_ > 1U) {
      partition_wakes_.reserve(local_partition_runner_count_);
      for (std::size_t index = 0U; index < local_partition_runner_count_; ++index) {
        partition_wakes_.push_back(std::make_unique<RuntimeWakeState>());
      }
    }
    lifecycle_control_->bind_wake(this, [](void* state) noexcept {
      auto* const session = static_cast<RunSessionState*>(state);
      session->signal_controller_wake(RuntimePhaseCounter::kWakeSignals);
    });
    std::vector<ParamAddress> parameter_fields;
    parameter_fields.reserve(plan_->parameter_fields.size());
    for (const ParameterFieldPlanRecord& field : plan_->parameter_fields) {
      parameter_fields.push_back(ParamAddress{
          .operator_instance =
              OperatorRef{
                  .id = field.owner.id,
                  .plan_generation = plan_->plan_generation,
                  .run_generation = run_generation_,
              },
          .field = field.field,
      });
    }
    auto link = ParamControlAccess::create_unattached_link(
        plan_->plan_generation,
        run_generation_,
        static_cast<OperatorInstanceId>(plan_->operator_parameters.size()),
        parameter_fields);
    if (!link) {
      throw RuntimeError{std::move(link.error())};
    }
    parameter_control_link_ = std::move(link).value();
  }

  RunSessionState(std::shared_ptr<const ExecutionPlanState> plan, RunGenerationId run_generation,
                  ProcessGroupSelectionFailureTag)
      : plan_(std::move(plan)),
        run_generation_(run_generation),
        joined_(true),
        diagnostic_buffer_(
            diagnostic_id("RUNTIME_DIAGNOSTIC_CAPACITY_EXHAUSTED"),
            DiagnosticSource{.graph_generation =
                                 plan_ != nullptr ? plan_->graph.generation() : 0U}) {
    append_diagnostic("RUNTIME_PROCESS_GROUP_SELECTION_REQUIRED");
    finish(RunTermination::kFailed);
  }

  explicit RunSessionState(SharedStartupFailureTag)
      : joined_(true),
        shared_terminal_(true),
        diagnostic_buffer_(diagnostic_id("RUNTIME_DIAGNOSTIC_CAPACITY_EXHAUSTED"), {}) {
    append_diagnostic("RUNTIME_STARTUP_FAILED");
    termination_.store(RunTermination::kFailed, std::memory_order_release);
  }

  RunSessionState(const RunSessionState&) = delete;
  RunSessionState& operator=(const RunSessionState&) = delete;
  ~RunSessionState() {
    retire_parameter_control();
    if (tensor_resources_ != nullptr) {
      tensor_resources_->close();
      tensor_resources_->detach_ports();
    }
    // Reconstructed Operators may retain unpublished loans. Destroy them while the fixed slot
    // state still exists so loan abandonment cannot call into a dead pool.
    graph_exec_.reset();
    tensor_resources_.reset();
  }

  [[nodiscard]] static RunSessionState* shared_startup_failure() noexcept {
    static RunSessionState state{SharedStartupFailureTag{}};
    return &state;
  }

  void release_owner() noexcept {
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

  void publish_startup_failure(const std::shared_ptr<const ExecutionPlanState>& attempted_plan,
                               RunGenerationId attempted_run_generation = {}) const noexcept {
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

  void start() noexcept {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      worker_ = std::thread([this] { worker_main(); });
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      append_diagnostic("RUNTIME_WORKER_START_FAILED");
      retire_parameter_control();
      finish(RunTermination::kFailed);
    }
#endif
  }

  [[nodiscard]] bool is_running() const noexcept {
    return termination_.load(std::memory_order_acquire) == RunTermination::kRunning;
  }

  [[nodiscard]] RunTermination termination() const noexcept {
    return termination_.load(std::memory_order_acquire);
  }

  void request_stop() noexcept {
    if (shared_terminal_) {
      return;
    }
    if (lifecycle_control_ != nullptr) {
      lifecycle_control_->request_stop_ingress();
    }
    const bool stop_was_requested = stop_requested_.exchange(true, std::memory_order_acq_rel);
    if (!stop_was_requested) {
      signal_controller_wake(RuntimePhaseCounter::kStopSignals);
    }
    parameter_ingress_close_requested_.store(true, std::memory_order_release);
    // The coordinator is owned by GraphExec, so the session mutex is the
    // bounded lifetime claim for this cold control-path operation. Retirement
    // clears the pointer under the same mutex before GraphExec can be destroyed.
    const std::lock_guard lock(parameter_mutex_);
    if (parameter_updates_ != nullptr) {
      parameter_updates_->request_close_ingress();
    }
  }

  void wait() {
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
      if (g_run_owned_session == this) {
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
    lock.lock();
    joined_ = true;
    joining_ = false;
    lock.unlock();
    join_cv_.notify_all();
  }

  [[nodiscard]] std::span<const Diagnostic> diagnostics() const noexcept {
    return diagnostic_buffer_.diagnostics();
  }

  void count_phase(RuntimePhaseCounter counter) noexcept {
    if (phase_counters_ != nullptr) {
      phase_counters_->increment(counter);
    }
  }

  /// Invoke one steady-state task directly on the already verified inline owner.
  ///
  /// The controller and sole partition runner are the same thread in this
  /// specialization. Re-entering ExecutionLaneSet would only reacquire the
  /// lane's stop/serialization locks around a callback that cannot migrate.
  [[nodiscard]] expected<void, Error> invoke_inline_owner_task(ExecutionLaneTask task) noexcept {
    if (task.invoke == nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument, "RUNTIME_LANE_DISPATCH_INVALID"});
    }
    RuntimeCallbackIdentity identity = task.callback_identity;
    if (identity.context == RuntimeCallbackContext::kNone) {
      identity.context = task.callback_context;
    }
    const RuntimeCallbackContextScope callback_scope{identity};
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

  void signal_controller_wake(RuntimePhaseCounter counter) noexcept {
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

  void signal_channel_owner_progress(VertexId owner) noexcept {
    RuntimeWakeState* const wake = partition_wake_by_vertex_.find(owner);
    if (wake == nullptr) {
      // Materialization proves every process-local endpoint owner is local. Preserve liveness
      // defensively if corrupted private state reaches this noexcept callback.
      signal_controller_wake(RuntimePhaseCounter::kWakeSignals);
      return;
    }
    count_phase(RuntimePhaseCounter::kWakeSignals);
    trace_phase(RuntimePhaseTracePoint::kControllerWakeSignaled, kNoRuntimePhaseReceiver, {}, 0U);
    if (wake->signal_coalesced()) {
      count_phase(RuntimePhaseCounter::kPartitionRunnerWakeSignals);
    }
  }

  void trace_phase(RuntimePhaseTracePoint point, std::size_t receiver_index, SourceId source,
                   std::uint64_t source_sequence) noexcept {
    if (phase_counters_ != nullptr) {
      phase_counters_->trace(point, receiver_index, source, source_sequence);
    }
  }

  [[nodiscard]] expected<OperatorRef, Error> operator_ref(OperatorRef plan_ref) const noexcept {
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

  [[nodiscard]] expected<LifecycleControlHandle, Error> lifecycle_control(
      const LifecycleAccessPolicy& policy) {
    if (lifecycle_control_ == nullptr) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    return lifecycle_control_->make_handle(policy);
  }

  [[nodiscard]] expected<ParamControlHandle, Error> parameter_control(
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

  [[nodiscard]] expected<ParamStopFenceResult, Error> close_parameter_update_ingress() {
    if (general_parameter_control_forbidden()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    parameter_ingress_close_requested_.store(true, std::memory_order_release);
    std::unique_lock lock(parameter_mutex_);
    parameter_cv_.wait(lock, [this] {
      return parameter_updates_ != nullptr || parameter_control_link_ == nullptr;
    });
    if (parameter_updates_ == nullptr) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    parameter_updates_->request_close_ingress();
    return parameter_updates_->close_ingress();
  }

  [[nodiscard]] expected<ParamSnapshot, Error> capture_params(ParamCaptureOptions options) {
    if (general_parameter_control_forbidden()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    std::unique_lock lock(parameter_mutex_);
    parameter_cv_.wait(lock, [this] {
      return parameter_updates_ != nullptr || parameter_control_link_ == nullptr;
    });
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

 private:
  [[nodiscard]] expected<void, Error> attach_parameter_control(
      ParameterUpdateCoordinator& updates) {
    const std::lock_guard lock(parameter_mutex_);
    if (parameter_control_link_ == nullptr) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    parameter_updates_ = &updates;
    if (parameter_ingress_close_requested_.load(std::memory_order_acquire)) {
      parameter_updates_->request_close_ingress();
    }
    auto attached = ParamControlAccess::attach(
        parameter_control_link_, parameter_updates_, &kParameterControlOperations);
    if (!attached) {
      parameter_updates_ = nullptr;
      parameter_cv_.notify_all();
      return attached;
    }
    parameter_cv_.notify_all();
    return {};
  }

  void terminalize_parameter_updates_at_run_end() noexcept {
    parameter_ingress_close_requested_.store(true, std::memory_order_release);
    const std::lock_guard lock(parameter_mutex_);
    if (parameter_updates_ != nullptr) {
      parameter_updates_->terminalize_unpublished_at_run_end();
    }
  }

  void retire_parameter_control() noexcept {
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

  void append_diagnostic(
      std::string_view token, VertexId vertex = {}, std::string_view service = {},
      const RuntimeDiagnosticDetail* detail = nullptr, std::string_view required = {},
      std::string_view provided = {},
      ::holoscan::DiagnosticSeverity severity = ::holoscan::DiagnosticSeverity::kError) noexcept {
    const std::lock_guard diagnostic_lock(diagnostic_mutex_);
    diagnostic_buffer_.append(
        diagnostic_id(token),
        token,
        DiagnosticSource{
            .graph_generation = plan_ != nullptr ? plan_->graph.generation() : 0U,
            .vertex = vertex,
        },
        service,
        detail,
        required,
        provided,
        severity);
  }

  void worker_main() noexcept {
    const RunOwnedThreadScope run_owned_thread{this};
    RunTermination outcome{RunTermination::kFailed};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      outcome = worker_main_impl();
      // Once the worker has passed every possible callback publication boundary, accepted
      // candidates cannot remain pending. Terminalize them before publishing the run outcome so
      // waiters always observe a final state at run completion.
      terminalize_parameter_updates_at_run_end();
      if (graph_exec_ == nullptr) {
        retire_parameter_control();
      }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      append_diagnostic("RUNTIME_STARTUP_FAILED");
      terminalize_parameter_updates_at_run_end();
      if (graph_exec_ == nullptr) {
        retire_parameter_control();
      }
    }
#endif
    finish(outcome);
  }

  struct ReadyRunState {
    struct InputSelectorSlot {
      VertexId owner{};
      PortId input{};
      RuntimeInputSelectionState selector;
      std::optional<std::uint32_t> derived_term_index;
      std::uint64_t arrival_watermark{};
      std::uint64_t producer_epoch{1U};
      std::string_view predicate_id;
      std::uint32_t predicate_ordinal{};
      bool predicate_input{};
      bool aligned_member{};
      bool end_of_stream_received{};
      std::vector<TemporalSampleRecord> channel_payload_claims;
      std::vector<TemporalSampleRecord> channel_payload_claim_scratch;
      std::vector<TemporalSampleRecord> channel_payload_retire_scratch;
    };

    struct OperatorParameterPolicy {
      std::vector<std::uint32_t> selector_indices;
      std::optional<std::uint32_t> deadline_term_index;
    };

    struct AlignedSelectorSlot {
      VertexId owner{};
      RuntimeAlignedSelectionState selector;
    };

    struct ClockBindingRequest {
      RuntimePortBindingRequest source;
      ClockTriggerPlanRecord schedule;
      std::size_t receiver_index{};
    };

    SchedulerPlanRecord dispatcher_scheduler;
    std::unique_ptr<Dispatcher> dispatcher;
    std::unique_ptr<RuntimeCandidateCoordinator> coordinator;
    std::shared_ptr<EndpointClockHealthRuntimeCoordinator> clock_health;
    std::shared_ptr<ExecutionLaneSet> lanes;
    std::shared_ptr<RuntimeChannelSession> channels;
    std::shared_ptr<RuntimeNotificationSourceRegistry> notifications;
    std::vector<RuntimeChannelPortReplayRecord> channel_replay_ports;
    std::vector<std::uint8_t> channel_activation_owners;
    std::vector<std::uint8_t> channel_input_owners;
    std::vector<RuntimePortBindingRequest> binding_requests;
    std::vector<std::pair<VertexId, RuntimePortBindingRequest>> channel_binding_requests;
    std::vector<ClockBindingRequest> clock_binding_requests;
    std::size_t expected_binding_count{};
    std::vector<std::unique_ptr<PortBinding>> notification_bindings;
    std::vector<std::unique_ptr<PortBinding>> bindings;
    std::vector<ClockTriggerPortBinding*> clock_bindings;
    std::vector<DispatcherCandidate> candidates;
    std::vector<FrozenRuntimeCandidate> frozen_candidates;
    std::vector<InputSelectorSlot> input_selectors;
    std::vector<OperatorParameterPolicy> parameter_policies;
    std::vector<AlignedSelectorSlot> aligned_selectors;
    std::vector<ActivationSelectionState> activation_selections;
    std::vector<RuntimeInputSelectionState::Transaction> selector_transactions;
    std::vector<RuntimeAlignedSelectionState::Transaction> aligned_transactions;
    std::vector<InputSelection> input_selection_scratch;
    std::vector<TemporalSampleRecord> selected_sample_scratch;
    std::vector<RuntimeChannelActivationSelection> channel_selection_scratch;
    std::vector<PredicateFrozenInput> predicate_input_scratch;
    std::vector<std::optional<RuntimePredicateEvaluatorEngine>> predicates;
    std::vector<PredicateReceiver> predicate_receivers;
    std::vector<std::optional<RuntimeDeadlineBindingRecord>> deadline_bindings;
    std::vector<std::size_t> restart_receiver_scratch;
    std::vector<VertexId> restart_member_vertex_scratch;
    std::vector<VertexId> restart_vertex_scratch;
    std::vector<VertexId> restart_aggregate_member_vertex_scratch;
    std::vector<VertexId> restart_aggregate_vertex_scratch;
    std::vector<RuntimeEvent> restart_event_scratch;
    std::vector<std::optional<TimePoint>> restart_phase_scratch;
    std::optional<RuntimeChannelDirectContinuationReceipt> direct_continuation_receipt;
    std::vector<RuntimeDirectCompositeBinding> direct_composites_by_receiver;
    std::vector<std::size_t> direct_composite_owner_by_receiver;
    std::vector<std::size_t> direct_chain_successor_by_receiver;
    std::vector<PortBinding*> direct_chain_binding_by_producer;
    SourceId direct_chain_root_source{};
    ClockId direct_chain_root_clock{};
    ClockTriggerPortBinding* direct_chain_root_binding{};
    std::size_t direct_chain_root_receiver{std::numeric_limits<std::size_t>::max()};
    std::optional<TimePoint> direct_chain_next_root_wake;
    std::vector<std::uint8_t> owned_receivers;
    RuntimeWakeState* wake{};
    const PartitionRunnerPlanRecord* runner{};
    std::optional<std::uint32_t> inline_execution_lane;
    bool inline_partition_execution{};
    bool partition_owner_execution{};
    bool close_shared_notifications{true};

    [[nodiscard]] bool uses_inline_invoke() const noexcept {
      return inline_partition_execution || partition_owner_execution;
    }
  };

  [[nodiscard]] static bool owns_receiver(const ReadyRunState& state,
                                          std::size_t receiver) noexcept {
    return state.owned_receivers.empty() ||
           (receiver < state.owned_receivers.size() && state.owned_receivers[receiver] != 0U);
  }

  [[nodiscard]] static bool owns_vertex(const ReadyRunState& state, const GraphExec& exec,
                                        VertexId vertex) noexcept {
    const auto found = std::find_if(
        exec.operators().begin(),
        exec.operators().end(),
        [&](const RuntimeOperatorInstance& instance) { return instance.vertex == vertex; });
    return found != exec.operators().end() &&
           owns_receiver(state,
                         static_cast<std::size_t>(std::distance(exec.operators().begin(), found)));
  }

  enum class ClosedIslandActivationDisposition : std::uint8_t {
    kCompleted = 0,
    kNoFire,
    kStopped,
    kRestarted,
    kFailed,
  };

  struct ClosedIslandActivationResult {
    RuntimeCallbackIdentity callback_identity{};
    std::optional<RuntimeChannelDirectContinuationReceipt> direct_continuation_receipt;
    Error error{ErrorCode::kSuccess};
    ClosedIslandActivationDisposition disposition{ClosedIslandActivationDisposition::kFailed};
    bool callback_attempted{};
    bool callback_completed{};
    bool activation_retired{};
    bool activation_cut_completed{};
    bool direct_continuation_clock_failed{};
  };

  [[nodiscard]] bool initialize_channel_session(GraphExec& exec, ReadyRunState& state) {
    const bool channel_plan_present = !plan_->process_local_channel_services.empty() ||
                                      !plan_->channel_services.empty() ||
                                      !plan_->cross_process_channel_services.empty();
    if (!channel_plan_present) {
      return true;
    }

    state.channel_replay_ports.reserve(plan_->ports.size());
    for (const RuntimeOperatorInstance& instance : exec.operators()) {
      if (instance.replay_spec == nullptr) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
        return false;
      }
      for (std::size_t index = 0U;
           index < OperatorSpecCompilerAccess::port_count(*instance.replay_spec);
           ++index) {
        const auto replay = OperatorSpecCompilerAccess::port(*instance.replay_spec, index);
        const auto planned = std::find_if(
            plan_->ports.begin(), plan_->ports.end(), [&](const PortPlanRecord& candidate) {
              return candidate.owner == instance.vertex &&
                     candidate.owner_kind == PortOwnerKind::kOperator &&
                     candidate.direction == replay.direction && candidate.local_name == replay.name;
            });
        if (planned == plan_->ports.end() || replay.declaration == nullptr ||
            replay.runtime_type_ops == nullptr) {
          append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
          return false;
        }
        state.channel_replay_ports.push_back(RuntimeChannelPortReplayRecord{
            .port = planned->id,
            .owner = instance.vertex,
            .direction = replay.direction,
            .declaration = const_cast<void*>(replay.declaration),
            .type_ops = replay.runtime_type_ops,
        });
      }
    }
    auto channels = RuntimeChannelSession::create(
        *plan_,
        state.channel_replay_ports,
        exec.process_group(),
        RuntimeChannelReadinessWake{
            .state = this,
            .signal =
                [](void* opaque) noexcept {
                  auto* const session = static_cast<RunSessionState*>(opaque);
                  session->signal_controller_wake(RuntimePhaseCounter::kWakeSignals);
                },
            .signal_owner_progress =
                local_partition_runner_count_ > 1U
                    ? +[](void* opaque, VertexId owner) noexcept {
                        static_cast<RunSessionState*>(opaque)->signal_channel_owner_progress(owner);
                      }
                    : nullptr,
        });
    if (!channels) {
      append_diagnostic(
          registered_token_or(channels.error().message, "RUNTIME_PORT_BINDING_UNAVAILABLE"));
      return false;
    }
    state.channel_activation_owners.assign(exec.operators().size(), 0U);
    state.channel_input_owners.assign(exec.operators().size(), 0U);
    for (std::size_t receiver = 0U; receiver < exec.operators().size(); ++receiver) {
      state.channel_activation_owners[receiver] = static_cast<std::uint8_t>(
          (*channels)->has_materialized_activation_state(exec.operators()[receiver].vertex));
      state.channel_input_owners[receiver] = static_cast<std::uint8_t>(
          (*channels)->has_materialized_input_state(exec.operators()[receiver].vertex));
    }
    state.channels = std::move(*channels);
    return true;
  }

  // Materialize each selector together with its parameter-derived term and retained-claim bounds;
  // partial initialization cannot be exposed to dispatch.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] bool initialize_input_selectors(GraphExec& exec, ReadyRunState& state) {
    state.input_selectors.reserve(plan_->input_selections.size());
    std::size_t max_selected = 0U;
    std::size_t local_selection_count = 0U;
    for (const InputSelectionPlanRecord& selection : plan_->input_selections) {
      const auto port = std::find_if(
          plan_->graph.ports().begin(), plan_->graph.ports().end(), [&](const PortRecord& record) {
            return record.id.value == selection.input.value() &&
                   record.direction == PortDirection::kInput;
          });
      if (port == plan_->graph.ports().end()) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
        return false;
      }
      if (!owns_vertex(state, exec, port->owner)) {
        continue;
      }
      if (!worker_owns_vertex(*plan_, exec.process_group(), port->owner)) {
        continue;
      }
      if (max_selected > std::numeric_limits<std::size_t>::max() - selection.max_selected_samples) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
        return false;
      }
      max_selected += selection.max_selected_samples;
      ++local_selection_count;
      const auto bracketed = std::find_if(
          plan_->bracketed_windows.begin(),
          plan_->bracketed_windows.end(),
          [&](const BracketedWindowPlanRecord& record) { return record.input == selection.input; });
      const BracketedWindowPlanRecord* bracketed_plan =
          bracketed == plan_->bracketed_windows.end() ? nullptr : &*bracketed;
      std::uint32_t max_pending_windows = 0U;
      if (bracketed_plan != nullptr) {
        const auto window = std::find_if(plan_->window_triggers.begin(),
                                         plan_->window_triggers.end(),
                                         [&](const WindowTriggerPlanRecord& record) {
                                           return record.owner == bracketed_plan->owner;
                                         });
        if (window == plan_->window_triggers.end()) {
          append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", port->owner);
          return false;
        }
        max_pending_windows = window->max_pending_windows;
      }
      auto created =
          RuntimeInputSelectionState::create(selection, bracketed_plan, max_pending_windows);
      if (!created) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", port->owner);
        return false;
      }
      const auto predicate_input = std::find_if(
          plan_->predicate_inputs.begin(),
          plan_->predicate_inputs.end(),
          [&](const PredicateInputPlanRecord& input) { return input.input == selection.input; });
      const std::size_t payload_claim_capacity = created->payload_claim_capacity();
      if (payload_claim_capacity == 0U) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", port->owner);
        return false;
      }
      ReadyRunState::InputSelectorSlot slot{
          .owner = port->owner,
          .input = selection.input,
          .selector = std::move(created).value(),
          .predicate_id = predicate_input == plan_->predicate_inputs.end()
                              ? std::string_view{}
                              : std::string_view{predicate_input->predicate_id},
          .predicate_ordinal = predicate_input == plan_->predicate_inputs.end()
                                   ? 0U
                                   : predicate_input->argument_ordinal,
          .predicate_input = predicate_input != plan_->predicate_inputs.end(),
          .aligned_member = false,
      };
      slot.channel_payload_claims.reserve(payload_claim_capacity);
      slot.channel_payload_claim_scratch.resize(payload_claim_capacity);
      slot.channel_payload_retire_scratch.resize(payload_claim_capacity);
      state.input_selectors.push_back(std::move(slot));
    }
    std::sort(state.input_selectors.begin(),
              state.input_selectors.end(),
              [](const auto& left, const auto& right) {
                if (left.owner != right.owner) {
                  return left.owner.value < right.owner.value;
                }
                if (left.predicate_input != right.predicate_input) {
                  return left.predicate_input;
                }
                if (left.predicate_input && left.predicate_id != right.predicate_id) {
                  return left.predicate_id < right.predicate_id;
                }
                if (left.predicate_input && left.predicate_ordinal != right.predicate_ordinal) {
                  return left.predicate_ordinal < right.predicate_ordinal;
                }
                return left.input.value() < right.input.value();
              });
    const std::size_t trigger_pin_capacity = static_cast<std::size_t>(std::count_if(
        plan_->temporal_triggers.begin(), plan_->temporal_triggers.end(), [&](const auto& trigger) {
          const bool local = std::any_of(
              exec.operators().begin(), exec.operators().end(), [&](const auto& instance) {
                return instance.operator_id == trigger.owner.id;
              });
          return local && (trigger.kind == TemporalRecordTriggerKind::kAnyInput ||
                           trigger.kind == TemporalRecordTriggerKind::kEach);
        }));
    if (local_selection_count > std::numeric_limits<std::size_t>::max() - trigger_pin_capacity ||
        max_selected > std::numeric_limits<std::size_t>::max() - trigger_pin_capacity) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
      return false;
    }
    const std::size_t selection_capacity = local_selection_count + trigger_pin_capacity;
    state.selector_transactions.reserve(local_selection_count);
    state.input_selection_scratch.reserve(selection_capacity);
    state.selected_sample_scratch.reserve(max_selected + trigger_pin_capacity);
    state.channel_selection_scratch.reserve(selection_capacity);
    state.predicate_input_scratch.reserve(local_selection_count);
    return true;
  }

  [[nodiscard]] bool initialize_aligned_selectors(GraphExec& exec, ReadyRunState& state) {
    state.aligned_selectors.reserve(plan_->aligned_triggers.size());
    state.aligned_transactions.reserve(plan_->aligned_triggers.size());
    for (const AlignedTriggerPlanRecord& aligned : plan_->aligned_triggers) {
      const auto instance = std::find_if(
          exec.operators().begin(),
          exec.operators().end(),
          [&](const RuntimeOperatorInstance& row) { return row.operator_id == aligned.owner.id; });
      if (instance == exec.operators().end() ||
          !owns_receiver(
              state, static_cast<std::size_t>(std::distance(exec.operators().begin(), instance)))) {
        continue;
      }
      auto created = RuntimeAlignedSelectionState::create(aligned, plan_->input_selections);
      if (!created) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance->vertex);
        return false;
      }
      state.aligned_selectors.push_back(ReadyRunState::AlignedSelectorSlot{
          .owner = instance->vertex,
          .selector = std::move(*created),
      });
      for (const AlignedMemberPlanRecord& member : aligned.members) {
        const auto input =
            std::find_if(state.input_selectors.begin(),
                         state.input_selectors.end(),
                         [&](const ReadyRunState::InputSelectorSlot& slot) {
                           return slot.owner == instance->vertex && slot.input == member.input;
                         });
        if (input == state.input_selectors.end() || input->predicate_input ||
            input->aligned_member) {
          append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance->vertex);
          return false;
        }
        input->aligned_member = true;
        const std::size_t payload_claim_capacity =
            state.aligned_selectors.back().selector.payload_claim_capacity(member.input);
        if (payload_claim_capacity == 0U) {
          append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance->vertex);
          return false;
        }
        input->channel_payload_claims.reserve(payload_claim_capacity);
        input->channel_payload_claim_scratch.resize(payload_claim_capacity);
        input->channel_payload_retire_scratch.resize(payload_claim_capacity);
      }
    }
    return true;
  }

  // Runner materialization validates and commits the complete per-partition runtime state.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] bool materialize_partition_runner_states(const GraphExec& exec,
                                                         std::vector<ReadyRunState>& states) {
    std::vector<const PartitionRunnerPlanRecord*> local_runners;
    local_runners.reserve(plan_->partition_runners.size());
    for (const PartitionRunnerPlanRecord& runner : plan_->partition_runners) {
      if (runner.process_group != exec.process_group()) {
        continue;
      }
      local_runners.push_back(&runner);
    }
    if (local_runners.empty()) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
      return false;
    }

    std::sort(local_runners.begin(),
              local_runners.end(),
              [](const PartitionRunnerPlanRecord* lhs, const PartitionRunnerPlanRecord* rhs) {
                return lhs->ordinal < rhs->ordinal;
              });
    if (local_runners.size() != local_partition_runner_count_ ||
        (local_runners.size() == 1U && !partition_wakes_.empty()) ||
        (local_runners.size() > 1U && partition_wakes_.size() != local_runners.size())) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
      return false;
    }
    partition_wake_by_vertex_.clear();
    if (local_runners.size() > 1U) {
      partition_wake_by_vertex_.reset(plan_->graph.vertices().size() + 1U);
    }
    std::vector<std::uint8_t> covered(exec.operators().size(), 0U);
    std::vector<std::uint8_t> lanes(plan_->execution_lanes.size(), 0U);
    states.clear();
    states.reserve(local_runners.size());

    for (std::size_t runner_index = 0U; runner_index < local_runners.size(); ++runner_index) {
      const PartitionRunnerPlanRecord* const runner = local_runners[runner_index];
      if (runner == nullptr || runner->execution_lane >= lanes.size() ||
          lanes[runner->execution_lane] != 0U || runner->operators.empty()) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
        return false;
      }
      lanes[runner->execution_lane] = 1U;
      states.emplace_back();
      ReadyRunState& state = states.back();
      state.runner = runner;
      state.owned_receivers.assign(exec.operators().size(), 0U);
      if (local_runners.size() == 1U) {
        state.wake = &readiness_wake_;
        state.inline_execution_lane = runner->execution_lane;
        state.inline_partition_execution = true;
      } else {
        state.wake = partition_wakes_[runner_index].get();
        state.partition_owner_execution = true;
        state.close_shared_notifications = false;
      }
      for (const PartitionRunnerOperatorPlanRecord& member : runner->operators) {
        const auto instance = std::find_if(exec.operators().begin(),
                                           exec.operators().end(),
                                           [&](const RuntimeOperatorInstance& candidate) {
                                             return candidate.vertex == member.vertex &&
                                                    candidate.operator_id == member.operator_id;
                                           });
        if (instance == exec.operators().end() || instance->partition != runner->partition ||
            instance->execution_lane != runner->execution_lane) {
          append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", member.vertex);
          return false;
        }
        const std::size_t receiver =
            static_cast<std::size_t>(std::distance(exec.operators().begin(), instance));
        if (covered[receiver] != 0U) {
          append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", member.vertex);
          return false;
        }
        covered[receiver] = 1U;
        state.owned_receivers[receiver] = 1U;
        if (!partition_wake_by_vertex_.empty()) {
          if (!partition_wake_by_vertex_.bind(member.vertex, state.wake)) {
            append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", member.vertex);
            return false;
          }
        }
      }
    }
    if (std::ranges::find(covered, std::uint8_t{0U}) != covered.end()) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
      return false;
    }

    if (plan_->inline_partition_execution) {
      if (states.size() != 1U) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
        return false;
      }
      ReadyRunState& state = states.front();
      const PartitionRunnerPlanRecord* const sole_runner = state.runner;
      const InlinePartitionExecutionPlanRecord& record = *plan_->inline_partition_execution;
      const bool owns_record =
          !exec.process_group() || exec.process_group() == record.process_group;
      if (owns_record) {
        const bool exact_operator_cover =
            exec.operators().size() == record.operators.size() &&
            std::all_of(exec.operators().begin(),
                        exec.operators().end(),
                        [&](const RuntimeOperatorInstance& instance) {
                          return instance.partition == record.partition &&
                                 instance.execution_lane == record.execution_lane &&
                                 std::count(record.operators.begin(),
                                            record.operators.end(),
                                            instance.vertex) == 1;
                        });
        if (!exact_operator_cover) {
          append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
          return false;
        }
        if (sole_runner != nullptr && (record.partition != sole_runner->partition ||
                                       record.execution_lane != sole_runner->execution_lane ||
                                       record.process_group != sole_runner->process_group)) {
          append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
          return false;
        }
        state.inline_execution_lane = record.execution_lane;
        state.inline_partition_execution = true;
      }
    }
    return true;
  }

  [[nodiscard]] bool initialize_lanes(const GraphExec& exec, ReadyRunState& state) {
    std::vector<ExecutionLanePlanRecord> worker_lanes;
    worker_lanes.reserve(exec.operators().size());
    for (const RuntimeOperatorInstance& instance : exec.operators()) {
      if (instance.execution_lane >= plan_->execution_lanes.size() ||
          plan_->execution_lanes[instance.execution_lane].execution_lane !=
              instance.execution_lane) {
        append_diagnostic("RUNTIME_EXECUTION_LANE_PLAN_INVALID", instance.vertex);
        return false;
      }
      worker_lanes.push_back(plan_->execution_lanes[instance.execution_lane]);
    }
    std::sort(worker_lanes.begin(),
              worker_lanes.end(),
              [](const ExecutionLanePlanRecord& lhs, const ExecutionLanePlanRecord& rhs) {
                return lhs.execution_lane < rhs.execution_lane;
              });
    worker_lanes.erase(
        std::unique(worker_lanes.begin(),
                    worker_lanes.end(),
                    [](const ExecutionLanePlanRecord& lhs, const ExecutionLanePlanRecord& rhs) {
                      return lhs.execution_lane == rhs.execution_lane;
                    }),
        worker_lanes.end());
    if (worker_lanes.empty()) {
      append_diagnostic("RUNTIME_EXECUTION_LANE_PLAN_INVALID");
      return false;
    }

    SchedulerPlanRecord local_scheduler = plan_->scheduler_plan;
    // Preserve global lane identities as the topology upper bound while projecting the selected
    // process group's materialized callback owners onto their exact local scheduler geometry.
    local_scheduler.worker_count = static_cast<std::uint32_t>(worker_lanes.size());
    local_scheduler.max_concurrent_activations = local_scheduler.worker_count;
    local_scheduler.activation_overlap =
        local_scheduler.worker_count > 1U && !state.inline_execution_lane;
    local_scheduler.intra_path_parallelism = 1U;
    auto lanes = ExecutionLaneSet::create(local_scheduler,
                                          worker_lanes,
                                          plan_->executable_factories.lane_placement_adapter,
                                          plan_->executable_factories.phase_counters.get(),
                                          state.inline_execution_lane);
    if (!lanes) {
      append_diagnostic(
          registered_token_or(lanes.error().message, "RUNTIME_EXECUTION_LANE_PLAN_INVALID"));
      return false;
    }
    state.lanes = std::move(lanes).value();
    auto started = state.lanes->start();
    lane_startup_evidence_ = state.lanes->take_startup_evidence();
    if (!started) {
      append_diagnostic(registered_token_or(
          execution_lane_startup_diagnostic_token(started.error()), "RUNTIME_LANE_STARTUP_FAILED"));
      return false;
    }
    return true;
  }

  [[nodiscard]] bool stop_lanes(ReadyRunState& state) noexcept {
    if (!state.lanes) {
      return true;
    }
    state.lanes->request_stop();
    const auto joined = state.lanes->join();
    if (!joined) {
      append_diagnostic(registered_token_or(joined.error().message, "RUNTIME_LANE_JOIN_FAILED"));
      return false;
    }
    return true;
  }

  void teardown_startup_runtime(ReadyRunState& state) noexcept {
    if (state.notifications != nullptr) {
      state.notifications->close();
    }
    static_cast<void>(stop_channels(state));
    static_cast<void>(stop_lanes(state));
  }

  [[nodiscard]] expected<std::optional<RuntimePredicateEvaluatorEngine>, Error>
  initialize_predicate(const RuntimeOperatorInstance& instance,
                       const TemporalTriggerPlanRecord& trigger, PredicateReceiver& receiver) {
    if (trigger.kind != TemporalRecordTriggerKind::kPredicate) {
      receiver = {};
      return std::optional<RuntimePredicateEvaluatorEngine>{};
    }
    const PredicateTriggerPlanRecord* const predicate =
        predicate_trigger_for(*plan_, trigger.owner);
    if (predicate == nullptr) {
      return make_unexpected(Error{ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"});
    }
    std::vector<PredicateInputPlanRecord> inputs;
    inputs.reserve(plan_->predicate_inputs.size());
    for (const auto& input : plan_->predicate_inputs) {
      if (input.predicate_id == predicate->id) {
        inputs.push_back(input);
      }
    }
    std::sort(inputs.begin(), inputs.end(), [](const auto& left, const auto& right) {
      return left.argument_ordinal < right.argument_ordinal;
    });
    const PredicateCallablePlanRecord* const callable =
        predicate_callable_for(*plan_, predicate->id);
    if (callable == nullptr) {
      return make_unexpected(Error{ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"});
    }

    const auto compute_input_count = static_cast<std::size_t>(std::count_if(
        plan_->input_selections.begin(), plan_->input_selections.end(), [&](const auto& selection) {
          const auto port = std::find_if(plan_->ports.begin(),
                                         plan_->ports.end(),
                                         [&](const auto& p) { return p.id == selection.input; });
          if (port == plan_->ports.end() || port->owner != instance.vertex) {
            return false;
          }
          return std::none_of(inputs.begin(), inputs.end(), [&](const auto& input) {
            return input.input == selection.input;
          });
        }));

    expected<TemporalContractSnapshot, Error> normalized =
        make_unexpected(Error{ErrorCode::kFailure});
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      normalized = ContractTestAccess::normalize(instance.op->contract(), 0U);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"});
    }
#endif
    if (!normalized || normalized->trigger.kind != TemporalTriggerKind::kPredicate ||
        normalized->trigger.predicate_clock_source !=
            (predicate->source == PredicateEvaluationSourceKind::kClock)) {
      return make_unexpected(Error{ErrorCode::kFailure, "RUNTIME_OPERATOR_REPLAY_MISMATCH"});
    }
    auto storage = ContractTestAccess::clone_predicate_storage(normalized->trigger.predicate);
    if (!storage) {
      return make_unexpected(std::move(storage.error()));
    }
    auto engine = RuntimePredicateEvaluatorEngine::create(
        *predicate, inputs, compute_input_count, *callable, std::move(*storage));
    if (!engine) {
      return make_unexpected(std::move(engine.error()));
    }
    engine->reset(PredicateResetReason::kColdRunStart, instance.tags->generation());
    receiver =
        PredicateReceiver{.address = instance.op.get(),
                          .local_type_token = normalized->trigger.predicate_receiver_type_token};
    return std::optional<RuntimePredicateEvaluatorEngine>{std::move(*engine)};
  }

  // Freeze one coherent readiness inventory before any source can publish into the Dispatcher.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] bool initialize_ready_state(GraphExec& exec, ReadyRunState& state) {
    if (state.clock_health == nullptr) {
      auto health = EndpointClockHealthRuntimeCoordinator::create(plan_->endpoint_clock_health,
                                                                  plan_->clocks,
                                                                  plan_->plan_generation,
                                                                  run_generation_,
                                                                  exec.clock_registry());
      if (!health) {
        append_diagnostic(health.error().token.empty()
                              ? std::string_view{"ENDPOINT_CLOCK_HEALTH_RUNTIME_PLAN_INVALID"}
                              : health.error().token);
        return false;
      }
      state.clock_health = std::move(health).value();
      if (state.channels != nullptr) {
        auto bound = state.channels->bind_clock_health(*state.clock_health, clock_health_mutex_);
        if (!bound) {
          append_diagnostic("ENDPOINT_CLOCK_HEALTH_RUNTIME_PLAN_INVALID");
          return false;
        }
      }
    }
    if (!initialize_input_selectors(exec, state)) {
      return false;
    }
    if (!initialize_aligned_selectors(exec, state)) {
      return false;
    }

    std::size_t max_outstanding = 0U;
    const std::size_t operator_count = exec.operators().size();
    if (operator_count > std::numeric_limits<std::uint32_t>::max() ||
        operator_count > std::numeric_limits<std::size_t>::max() / 2U) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
      return false;
    }
    std::uint32_t next_auxiliary_source = 1U;
    const auto advance_auxiliary_source = [&](const SourceId source) {
      if (next_auxiliary_source == 0U) {
        return;
      }
      const std::uint32_t value = source_id_value(source);
      if (value == std::numeric_limits<std::uint32_t>::max()) {
        next_auxiliary_source = 0U;
        return;
      }
      next_auxiliary_source = std::max(next_auxiliary_source, value + 1U);
    };
    for (const ReadinessPlanRecord& readiness : plan_->readiness) {
      advance_auxiliary_source(readiness.source_id);
    }
    for (const TemporalTriggerPlanRecord& trigger : plan_->temporal_triggers) {
      advance_auxiliary_source(trigger.readiness_source);
    }

    std::vector<RuntimeSourceBindingRecord> sources;
    std::vector<TagAllocator*> tag_allocators;
    sources.reserve(operator_count * 2U);
    tag_allocators.reserve(operator_count);
    state.binding_requests.reserve(operator_count);
    state.channel_binding_requests.reserve(operator_count);
    state.clock_binding_requests.reserve(operator_count);
    state.bindings.reserve(operator_count * 2U);
    state.clock_bindings.reserve(operator_count);
    state.activation_selections.reserve(operator_count);
    state.predicates.reserve(operator_count);
    state.predicate_receivers.reserve(operator_count);
    state.deadline_bindings.reserve(operator_count);
    state.parameter_policies.reserve(operator_count);
    state.restart_receiver_scratch.reserve(operator_count);
    state.restart_member_vertex_scratch.reserve(operator_count);
    state.restart_vertex_scratch.reserve(operator_count);
    state.restart_aggregate_member_vertex_scratch.reserve(operator_count);
    state.restart_aggregate_vertex_scratch.reserve(operator_count);
    state.restart_phase_scratch.resize(operator_count);
    for (std::size_t index = 0U; index < operator_count; ++index) {
      const RuntimeOperatorInstance& instance = exec.operators()[index];
      const bool receiver_owned = owns_receiver(state, index);
      const TemporalTriggerPlanRecord* const trigger =
          temporal_trigger_for(*plan_, instance.operator_id);
      if (trigger == nullptr || !trigger->readiness_source ||
          trigger->owner.plan_generation != plan_->plan_generation) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
        return false;
      }
      const PredicateTriggerPlanRecord* const predicate =
          trigger->kind == TemporalRecordTriggerKind::kPredicate
              ? predicate_trigger_for(*plan_, trigger->owner)
              : nullptr;
      const bool clock_source =
          trigger->kind == TemporalRecordTriggerKind::kClock ||
          (predicate != nullptr && predicate->source == PredicateEvaluationSourceKind::kClock);
      const bool input_source = input_trigger_kind(trigger->kind) ||
                                (predicate != nullptr &&
                                 predicate->source == PredicateEvaluationSourceKind::kInputUpdate);
      // Every non-input trigger owns activation authority independently from connected Channel
      // inputs. Materialize one auxiliary arrival source for those inputs so their bounded
      // temporal state advances without allowing an arrival to activate the Operator.
      const bool has_channel_side_inputs = !input_source &&
                                           index < state.channel_input_owners.size() &&
                                           state.channel_input_owners[index] != 0U;
      const bool direct_source = !clock_source && !input_source;
      const ReadinessPlanRecord* const readiness =
          direct_source ? readiness_for(*plan_, instance.vertex) : nullptr;
      const ClockTriggerPlanRecord* const clock =
          clock_source ? clock_trigger_for(*plan_, trigger->owner) : nullptr;
      if ((!clock_source && !input_source && readiness == nullptr) ||
          (clock_source && clock == nullptr) ||
          ((input_source || has_channel_side_inputs) && state.channels == nullptr)) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
        return false;
      }
      expected<std::uint32_t, Error> channel_capacity = 0U;
      if (input_source || has_channel_side_inputs) {
        channel_capacity = state.channels->readiness_capacity(instance.vertex);
        if (!channel_capacity) {
          append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
          return false;
        }
      }
      const std::uint32_t source_capacity = clock_source   ? 1U
                                            : input_source ? *channel_capacity
                                                           : readiness->accepted_capacity;
      const ClockId reference_clock =
          direct_source ? readiness->resolved_clock : trigger->reference_clock;
      const std::uint64_t source_generation =
          direct_source ? readiness->initial_source_generation : 1U;
      const std::uint64_t source_sequence = direct_source ? readiness->initial_source_sequence : 0U;
      const RuntimeEventKind event_kind = clock_source   ? RuntimeEventKind::kClockTick
                                          : input_source ? RuntimeEventKind::kInputArrival
                                                         : runtime_event_kind(readiness->kind);
      // A production Channel readiness source is owner-multiplexed: polling drains every input
      // arrival in deterministic order so non-trigger samples can populate temporal selectors
      // before the trigger witness is considered. Preserve the concrete event port on each
      // RuntimeEvent, but keep the frozen source port open so the coordinator accepts the complete
      // owner-local prefix. Explicitly port-scoped lower-level bindings still retain exact-port
      // validation in RuntimeCandidateCoordinator.
      const PortId source_port{};
      if ((direct_source &&
           (readiness == nullptr || readiness->source_id != trigger->readiness_source ||
            !readiness->referenced || !readiness->materialized)) ||
          (clock_source &&
           (clock == nullptr || clock->clock != reference_clock || clock->period_ns <= 0)) ||
          !reference_clock.valid() || source_capacity == 0U ||
          (receiver_owned &&
           max_outstanding > std::numeric_limits<std::size_t>::max() - source_capacity)) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
        return false;
      }
      auto deadline = bind_runtime_deadline(plan_->deadlines,
                                            *trigger,
                                            plan_->clock_conversions,
                                            plan_->plan_generation,
                                            trigger->kind == TemporalRecordTriggerKind::kPredicate
                                                ? predicate_trigger_for(*plan_, trigger->owner)
                                                : nullptr);
      if (!deadline) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
        return false;
      }
      const std::optional<RuntimeDeadlineBindingRecord> deadline_binding =
          std::move(deadline).value();
      const RuntimeDeadlineBindingRecord deadline_values =
          deadline_binding.value_or(RuntimeDeadlineBindingRecord{});
      const ClockId source_clock =
          deadline_binding ? deadline_values.source_clock : reference_clock;
      const std::uint64_t conversion_id = deadline_binding ? deadline_values.conversion_id : 0U;
      if (receiver_owned) {
        max_outstanding += source_capacity;
        sources.push_back(RuntimeSourceBindingRecord{
            .candidate_kind = event_kind,
            .source = trigger->readiness_source,
            .port = source_port,
            .source_clock = source_clock,
            .reference_clock = reference_clock,
            .source_to_reference_conversion_id = conversion_id,
            .source_order = static_cast<std::uint32_t>(index),
            .capacity = source_capacity,
            .source_generation = source_generation,
            .tag_generation = instance.tags->generation(),
            .receiver_index = index,
            .partition = instance.partition,
            .execution_lane = instance.execution_lane,
            .vertex = instance.vertex,
            .plan_vertex_order = static_cast<std::uint32_t>(index),
            .stable_candidate_tie =
                direct_source ? readiness->declaration_ordinal : static_cast<std::uint64_t>(index),
            .deadline = deadline_binding,
        });
      }
      const RuntimePortBindingRequest binding_request{
          .plan_generation = plan_->plan_generation,
          .run_generation = run_generation_,
          .kind = event_kind,
          .source = trigger->readiness_source,
          .port = source_port,
          .source_clock = source_clock,
          .reference_clock = reference_clock,
          .source_to_reference_conversion_id = conversion_id,
          .execution_lane = instance.execution_lane,
          .capacity = source_capacity,
          .source_generation = source_generation,
          .initial_source_sequence = source_sequence,
      };
      if (receiver_owned) {
        if (clock_source) {
          state.clock_binding_requests.push_back(ReadyRunState::ClockBindingRequest{
              .source = binding_request, .schedule = *clock, .receiver_index = index});
        } else if (input_source) {
          state.channel_binding_requests.emplace_back(instance.vertex, binding_request);
        } else {
          state.binding_requests.push_back(binding_request);
        }
      }
      if (has_channel_side_inputs) {
        const std::size_t auxiliary_order = operator_count + index;
        if (next_auxiliary_source == 0U ||
            auxiliary_order > std::numeric_limits<std::uint32_t>::max() ||
            (receiver_owned &&
             max_outstanding > std::numeric_limits<std::size_t>::max() - *channel_capacity)) {
          append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
          return false;
        }
        const SourceId auxiliary_source = make_source_id(next_auxiliary_source);
        next_auxiliary_source = next_auxiliary_source == std::numeric_limits<std::uint32_t>::max()
                                    ? 0U
                                    : next_auxiliary_source + 1U;
        if (receiver_owned) {
          max_outstanding += *channel_capacity;
          sources.push_back(RuntimeSourceBindingRecord{
              .candidate_kind = RuntimeEventKind::kInputArrival,
              .source = auxiliary_source,
              .port = {},
              .source_clock = reference_clock,
              .reference_clock = reference_clock,
              .source_to_reference_conversion_id = 0U,
              .source_order = static_cast<std::uint32_t>(auxiliary_order),
              .capacity = *channel_capacity,
              .source_generation = 1U,
              .tag_generation = instance.tags->generation(),
              .receiver_index = index,
              .partition = instance.partition,
              .execution_lane = instance.execution_lane,
              .vertex = instance.vertex,
              .plan_vertex_order = static_cast<std::uint32_t>(index),
              .stable_candidate_tie = static_cast<std::uint64_t>(auxiliary_order),
          });
          // Independent clock/readiness activation authority remains on the primary source. This
          // private, plan-derived Channel source only admits connected samples into bounded
          // selectors; its arrival candidates are retired before Dispatcher eligibility.
          state.channel_binding_requests.emplace_back(instance.vertex,
                                                      RuntimePortBindingRequest{
                                                          .plan_generation = plan_->plan_generation,
                                                          .run_generation = run_generation_,
                                                          .kind = RuntimeEventKind::kInputArrival,
                                                          .source = auxiliary_source,
                                                          .port = {},
                                                          .source_clock = reference_clock,
                                                          .reference_clock = reference_clock,
                                                          .source_to_reference_conversion_id = 0U,
                                                          .execution_lane = instance.execution_lane,
                                                          .capacity = *channel_capacity,
                                                          .source_generation = 1U,
                                                          .initial_source_sequence = 0U,
                                                      });
        }
      }
      tag_allocators.push_back(instance.tags.get());

      PredicateReceiver predicate_receiver;
      auto predicate_engine = initialize_predicate(instance, *trigger, predicate_receiver);
      if (!predicate_engine) {
        append_diagnostic(registered_token_or(predicate_engine.error().message,
                                              "RUNTIME_OPERATOR_REPLAY_MISMATCH"),
                          instance.vertex);
        return false;
      }
      state.predicates.push_back(std::move(*predicate_engine));
      state.predicate_receivers.push_back(predicate_receiver);
      state.deadline_bindings.push_back(deadline_binding);

      std::size_t max_inputs = 0U;
      std::size_t max_selected = 0U;
      ReadyRunState::OperatorParameterPolicy parameter_policy;
      for (std::size_t selector_index = 0U; selector_index < state.input_selectors.size();
           ++selector_index) {
        auto& selector = state.input_selectors[selector_index];
        if (selector.owner != instance.vertex) {
          continue;
        }
        if (selector_index > std::numeric_limits<std::uint32_t>::max()) {
          append_diagnostic("RUNTIME_PARAMETER_BINDING_UNAVAILABLE", instance.vertex);
          return false;
        }
        if (selector.selector.plan().selector == InputSelectorKind::kLatestWithin) {
          auto term = bind_runtime_derived_term(instance,
                                                ParameterDerivedTermKind::kLatestWithinMaxAge,
                                                selector.input,
                                                selector.selector.plan().max_age_ns);
          if (!term) {
            append_diagnostic("RUNTIME_PARAMETER_BINDING_UNAVAILABLE", instance.vertex);
            return false;
          }
          // Preserve the exact cold binding for EA1-012's production selector transaction.
          selector.derived_term_index = *term;
        }
        parameter_policy.selector_indices.push_back(static_cast<std::uint32_t>(selector_index));
        ++max_inputs;
        if (max_selected > std::numeric_limits<std::size_t>::max() -
                               selector.selector.plan().max_selected_samples) {
          append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
          return false;
        }
        max_selected += selector.selector.plan().max_selected_samples;
      }
      for (std::size_t term_index = 0U; term_index < instance.parameter_derived_terms.size();
           ++term_index) {
        const auto& term = instance.parameter_derived_terms[term_index];
        if (term.kind != ParameterDerivedTermKind::kLatestWithinMaxAge ||
            term.update_disposition != ParameterDerivedTermUpdateDisposition::kRuntimeAtomic) {
          continue;
        }
        const bool has_consumer = term_index <= std::numeric_limits<std::uint32_t>::max() &&
                                  std::any_of(state.input_selectors.begin(),
                                              state.input_selectors.end(),
                                              [&](const auto& selector) {
                                                return selector.owner == instance.vertex &&
                                                       selector.derived_term_index ==
                                                           static_cast<std::uint32_t>(term_index);
                                              });
        if (!has_consumer) {
          append_diagnostic("RUNTIME_PARAMETER_BINDING_UNAVAILABLE", instance.vertex);
          return false;
        }
      }
      if (deadline_binding.has_value()) {
        auto term = bind_runtime_derived_term(instance,
                                              ParameterDerivedTermKind::kContractDeadline,
                                              {},
                                              deadline_values.deadline->budget_ns);
        if (!term) {
          append_diagnostic("RUNTIME_PARAMETER_BINDING_UNAVAILABLE", instance.vertex);
          return false;
        }
        parameter_policy.deadline_term_index = *term;
      }
      if (trigger != nullptr && (trigger->kind == TemporalRecordTriggerKind::kAnyInput ||
                                 trigger->kind == TemporalRecordTriggerKind::kEach)) {
        if (max_inputs == std::numeric_limits<std::size_t>::max() ||
            max_selected == std::numeric_limits<std::size_t>::max()) {
          append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
          return false;
        }
        ++max_inputs;
        ++max_selected;
      }
      auto activation = ActivationSelectionState::create(max_inputs, max_selected);
      if (!activation) {
        HOLOSCAN_LOG_ERROR(
            "Runtime activation-selection storage failed with {} for vertex {} ({} inputs, {} "
            "selected samples)",
            error_code_name(activation.error().code),
            instance.vertex.value,
            max_inputs,
            max_selected);
        const RuntimeDiagnosticDetail detail{.present = true, .error = activation.error()};
        append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE", instance.vertex, {}, &detail);
        return false;
      }
      state.activation_selections.push_back(std::move(activation).value());
      state.parameter_policies.push_back(std::move(parameter_policy));
    }
    state.dispatcher_scheduler = plan_->scheduler_plan;
    if (state.runner != nullptr) {
      state.dispatcher_scheduler.max_concurrent_activations = 1U;
      state.dispatcher_scheduler.activation_overlap = false;
      state.dispatcher_scheduler.worker_count = 1U;
      state.dispatcher_scheduler.intra_path_parallelism = 1U;
    }
    auto dispatcher = Dispatcher::create(state.dispatcher_scheduler, max_outstanding);
    if (!dispatcher) {
      append_diagnostic(
          registered_token_or(dispatcher.error().message, "READY_POLICY_PLAN_INVALID"));
      return false;
    }
    state.dispatcher = std::make_unique<Dispatcher>(std::move(dispatcher).value());
    auto coordinator =
        RuntimeCandidateCoordinator::create_with_tags(sources, tag_allocators, max_outstanding);
    if (!coordinator) {
      HOLOSCAN_LOG_ERROR("Runtime candidate coordinator construction failed with {}",
                         error_code_name(coordinator.error().code));
      const RuntimeDiagnosticDetail detail{.present = true, .error = coordinator.error()};
      append_diagnostic(
          registered_token_or(coordinator.error().message, "RUNTIME_READINESS_BINDING_UNAVAILABLE"),
          {},
          {},
          &detail);
      return false;
    }
    state.coordinator =
        std::make_unique<RuntimeCandidateCoordinator>(std::move(coordinator).value());
    state.expected_binding_count = sources.size();
    state.candidates.reserve(max_outstanding);
    state.frozen_candidates.reserve(max_outstanding);
    state.restart_event_scratch.resize(max_outstanding);
    return true;
  }

  // Bind production notifications before Arm so lifecycle callbacks can acquire sender authority.
  // The sender registry keeps posts not-ready until the complete Start barrier succeeds.
  [[nodiscard]] bool bind_production_notifications(GraphExec& exec, ReadyRunState& state) {
    if (plan_->executable_factories.test_direct_activation_count != 0U ||
        plan_->executable_factories.port_binding_factory != nullptr) {
      return true;
    }
    std::vector<RuntimeNotificationSourceDescriptor> descriptors;
    descriptors.reserve(state.binding_requests.size());
    for (const auto& request : state.binding_requests) {
      if (request.kind != RuntimeEventKind::kNotification) {
        continue;
      }
      const auto readiness = std::find_if(
          plan_->readiness.begin(), plan_->readiness.end(), [&](const ReadinessPlanRecord& row) {
            return row.source_id == request.source && row.owner && row.referenced &&
                   row.materialized;
          });
      const auto lane = std::find_if(plan_->execution_lanes.begin(),
                                     plan_->execution_lanes.end(),
                                     [&](const ExecutionLanePlanRecord& row) {
                                       return row.execution_lane == request.execution_lane;
                                     });
      auto clock = exec.resolve_clock(request.reference_clock);
      if (readiness == plan_->readiness.end() ||
          readiness->kind != ReadinessSourceKind::kNotification ||
          readiness->accepted_capacity != request.capacity ||
          readiness->accepted_sender_reference_capacity == 0U ||
          lane == plan_->execution_lanes.end() || !clock) {
        append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
        return false;
      }
      descriptors.push_back(RuntimeNotificationSourceDescriptor{
          .owner = readiness->owner,
          .request = request,
          .wake_requirement = readiness->wake_requirement,
          .wake_mode = lane->accepted_placement.wake_mode,
          .sender_reference_capacity = readiness->accepted_sender_reference_capacity,
          .clock = *clock,
          .wake_state = this,
          .wake =
              [](void* opaque) noexcept {
                auto* const session = static_cast<RunSessionState*>(opaque);
                session->signal_controller_wake(RuntimePhaseCounter::kWakeSignals);
              },
      });
    }
    if (descriptors.empty()) {
      return true;
    }
    auto registry = RuntimeNotificationSourceRegistry::create(descriptors);
    if (!registry) {
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
      return false;
    }
    state.notification_bindings.reserve(descriptors.size());
    for (const auto& descriptor : descriptors) {
      auto start_coordinate = descriptor.clock.now();
      auto binding = start_coordinate ? (*registry)->bind(descriptor.request, *start_coordinate)
                                      : expected<std::unique_ptr<PortBinding>, Error>{
                                            make_unexpected(start_coordinate.error())};
      if (!binding || *binding == nullptr || (*binding)->capacity() < descriptor.request.capacity) {
        append_diagnostic(start_coordinate ? "RUNTIME_READINESS_BINDING_UNAVAILABLE"
                                           : "RUNTIME_CLOCK_READ_FAILED");
        return false;
      }
      state.notification_bindings.push_back(std::move(*binding));
    }
    state.notifications = std::move(*registry);
    return true;
  }

  // Shared notification binding canonicalizes all partition requests in one ownership pass.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] bool bind_shared_production_notifications(GraphExec& exec,
                                                          std::span<ReadyRunState> states) {
    if (states.size() == 1U) {
      return bind_production_notifications(exec, states.front());
    }
    if (plan_->executable_factories.test_direct_activation_count != 0U ||
        plan_->executable_factories.port_binding_factory != nullptr) {
      return true;
    }
    struct PendingDescriptor {
      std::size_t state_index{};
      RuntimeNotificationSourceDescriptor descriptor;
    };
    std::vector<PendingDescriptor> pending;
    for (std::size_t state_index = 0U; state_index < states.size(); ++state_index) {
      const ReadyRunState& state = states[state_index];
      for (const auto& request : state.binding_requests) {
        if (request.kind != RuntimeEventKind::kNotification) {
          continue;
        }
        const auto readiness = std::find_if(
            plan_->readiness.begin(), plan_->readiness.end(), [&](const ReadinessPlanRecord& row) {
              return row.source_id == request.source && row.owner && row.referenced &&
                     row.materialized;
            });
        const auto lane = std::find_if(plan_->execution_lanes.begin(),
                                       plan_->execution_lanes.end(),
                                       [&](const ExecutionLanePlanRecord& row) {
                                         return row.execution_lane == request.execution_lane;
                                       });
        auto clock = exec.resolve_clock(request.reference_clock);
        if (readiness == plan_->readiness.end() ||
            readiness->kind != ReadinessSourceKind::kNotification ||
            readiness->accepted_capacity != request.capacity ||
            readiness->accepted_sender_reference_capacity == 0U ||
            lane == plan_->execution_lanes.end() || !clock) {
          append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
          return false;
        }
        pending.push_back(PendingDescriptor{
            .state_index = state_index,
            .descriptor =
                RuntimeNotificationSourceDescriptor{
                    .owner = readiness->owner,
                    .request = request,
                    .wake_requirement = readiness->wake_requirement,
                    .wake_mode = lane->accepted_placement.wake_mode,
                    .sender_reference_capacity = readiness->accepted_sender_reference_capacity,
                    .clock = *clock,
                    .wake_state = this,
                    .wake =
                        [](void* opaque) noexcept {
                          auto* const session = static_cast<RunSessionState*>(opaque);
                          session->signal_controller_wake(RuntimePhaseCounter::kWakeSignals);
                        },
                },
        });
      }
    }
    if (pending.empty()) {
      return true;
    }
    std::sort(pending.begin(),
              pending.end(),
              [](const PendingDescriptor& lhs, const PendingDescriptor& rhs) {
                return source_id_value(lhs.descriptor.request.source) <
                       source_id_value(rhs.descriptor.request.source);
              });
    std::vector<RuntimeNotificationSourceDescriptor> descriptors;
    descriptors.reserve(pending.size());
    for (const PendingDescriptor& row : pending) {
      descriptors.push_back(row.descriptor);
    }
    auto registry = RuntimeNotificationSourceRegistry::create(descriptors);
    if (!registry) {
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
      return false;
    }
    for (std::size_t index = 0U; index < pending.size(); ++index) {
      const RuntimeNotificationSourceDescriptor& descriptor = descriptors[index];
      auto start_coordinate = descriptor.clock.now();
      auto binding = start_coordinate ? (*registry)->bind(descriptor.request, *start_coordinate)
                                      : expected<std::unique_ptr<PortBinding>, Error>{
                                            make_unexpected(start_coordinate.error())};
      if (!binding || *binding == nullptr || (*binding)->capacity() < descriptor.request.capacity) {
        append_diagnostic(start_coordinate ? "RUNTIME_READINESS_BINDING_UNAVAILABLE"
                                           : "RUNTIME_CLOCK_READ_FAILED");
        return false;
      }
      states[pending[index].state_index].notification_bindings.push_back(std::move(*binding));
    }
    const std::shared_ptr<RuntimeNotificationSourceRegistry> shared_registry = std::move(*registry);
    for (ReadyRunState& state : states) {
      state.notifications = shared_registry;
    }
    return true;
  }

  // Bind every remaining physical, Channel, and clock source before opening lifecycle ingress;
  // partial binding is a startup failure and must not publish a runnable session.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] bool bind_ready_sources(GraphExec& exec, ReadyRunState& state,
                                        bool manage_channel_session = true) {
    const std::uint32_t direct_witness_count =
        plan_->executable_factories.test_direct_activation_count;
    const std::shared_ptr<RuntimePortBindingFactory>& configured_factory =
        plan_->executable_factories.port_binding_factory;
    if (direct_witness_count != 0U && configured_factory != nullptr) {
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
      return false;
    }
    std::optional<TestDirectPortBindingFactory> direct_factory;
    RuntimePortBindingFactory* factory = configured_factory.get();
    if (direct_witness_count != 0U) {
      direct_factory.emplace(direct_witness_count);
      factory = &*direct_factory;
    }
    const bool has_unbound_direct_request = std::any_of(
        state.binding_requests.begin(),
        state.binding_requests.end(),
        [&](const RuntimePortBindingRequest& request) {
          return request.kind != RuntimeEventKind::kNotification || state.notifications == nullptr;
        });
    if ((has_unbound_direct_request && factory == nullptr) ||
        state.binding_requests.size() + state.channel_binding_requests.size() +
                state.clock_binding_requests.size() !=
            state.expected_binding_count) {
      HOLOSCAN_LOG_ERROR(
          "Runtime readiness binding inventory is inconsistent: {} direct, {} Channel, {} clock "
          "requests for {} frozen sources across {} operators",
          state.binding_requests.size(),
          state.channel_binding_requests.size(),
          state.clock_binding_requests.size(),
          state.expected_binding_count,
          exec.operators().size());
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
      return false;
    }
    state.bindings = std::move(state.notification_bindings);
    state.clock_bindings.assign(exec.operators().size(), nullptr);
    for (const auto& request : state.binding_requests) {
      if (request.kind == RuntimeEventKind::kNotification && state.notifications != nullptr) {
        continue;
      }
      auto clock = exec.resolve_clock(request.reference_clock);
      auto start_coordinate =
          clock ? clock->now() : expected<TimePoint, Error>{make_unexpected(clock.error())};
      if (!start_coordinate) {
        append_diagnostic("RUNTIME_CLOCK_READ_FAILED");
        return false;
      }
      if (request.reference_clock != start_coordinate->clock_id) {
        append_diagnostic("RUNTIME_CLOCK_BINDING_UNAVAILABLE");
        return false;
      }
      auto binding = factory->bind(request, *start_coordinate);
      if (!binding || binding.value() == nullptr ||
          binding.value()->capacity() < request.capacity) {
        const std::string_view candidate =
            !binding ? std::string_view{binding.error().message} : std::string_view{};
        append_diagnostic(registered_token_or(candidate, "RUNTIME_READINESS_BINDING_UNAVAILABLE"),
                          readiness_owner_for_source(*plan_, request.source));
        return false;
      }
      state.bindings.push_back(std::move(binding).value());
    }
    for (const auto& [owner, request] : state.channel_binding_requests) {
      auto binding = state.channels->create_readiness_binding(owner, request);
      if (!binding || binding.value() == nullptr ||
          binding.value()->capacity() < request.capacity) {
        const Error error = !binding
                                ? binding.error()
                                : Error{binding.value() == nullptr ? ErrorCode::kNotFound
                                                                   : ErrorCode::kResourceExhausted};
        HOLOSCAN_LOG_ERROR(
            "Channel readiness binding failed with {} for vertex {} (required capacity {})",
            error_code_name(error.code),
            owner.value,
            request.capacity);
        const RuntimeDiagnosticDetail detail{.present = true, .error = error};
        append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE", owner, {}, &detail);
        return false;
      }
      state.bindings.push_back(std::move(*binding));
    }
    for (const auto& request : state.clock_binding_requests) {
      auto clock = exec.resolve_clock(request.schedule.clock);
      if (!clock) {
        append_diagnostic("RUNTIME_CLOCK_BINDING_UNAVAILABLE");
        return false;
      }
      TimePoint phase;
      if (request.schedule.phase_origin == ClockPhaseOrigin::kCapturedAtSuccessfulStart) {
        auto captured = clock->now();
        if (!captured) {
          append_diagnostic("RUNTIME_CLOCK_READ_FAILED");
          return false;
        }
        phase = *captured;
      } else if (request.schedule.phase_origin == ClockPhaseOrigin::kExplicitAbsolute &&
                 request.schedule.explicit_phase) {
        auto converted =
            exec.clock_registry().convert(*request.schedule.explicit_phase, request.schedule.clock);
        if (!converted) {
          append_diagnostic("RUNTIME_CLOCK_BINDING_UNAVAILABLE");
          return false;
        }
        phase = *converted;
      } else {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
        return false;
      }
      auto binding = ClockTriggerPortBinding::create(
          request.source,
          *clock,
          phase,
          std::chrono::nanoseconds{request.schedule.period_ns},
          request.schedule.phase_origin == ClockPhaseOrigin::kCapturedAtSuccessfulStart);
      if (!binding) {
        append_diagnostic(
            registered_token_or(binding.error().message, "RUNTIME_CLOCK_BINDING_UNAVAILABLE"));
        return false;
      }
      state.clock_bindings[request.receiver_index] = binding->get();
      state.bindings.push_back(std::move(*binding));
    }
    if (manage_channel_session && state.channels != nullptr) {
      auto refreshed = state.channels->refresh_connections();
      if (!refreshed) {
        HOLOSCAN_LOG_ERROR("Channel connection refresh failed with {}",
                           error_code_name(refreshed.error().code));
        const RuntimeDiagnosticDetail detail{.present = true, .error = refreshed.error()};
        append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE", {}, {}, &detail);
        return false;
      }
      auto armed = state.channels->arm();
      if (!armed) {
        HOLOSCAN_LOG_ERROR("Channel readiness arm failed with {}",
                           error_code_name(armed.error().code));
        const RuntimeDiagnosticDetail detail{.present = true, .error = armed.error()};
        append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE", {}, {}, &detail);
        return false;
      }
    }
    return initialize_direct_composites(exec, state);
  }

  [[nodiscard]] bool arm_shared_channels(ReadyRunState& state) {
    if (state.channels == nullptr) {
      return true;
    }
    auto refreshed = state.channels->refresh_connections();
    if (!refreshed) {
      HOLOSCAN_LOG_ERROR("Channel connection refresh failed with {}",
                         error_code_name(refreshed.error().code));
      const RuntimeDiagnosticDetail detail{.present = true, .error = refreshed.error()};
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE", {}, {}, &detail);
      return false;
    }
    auto armed = state.channels->arm();
    if (!armed) {
      HOLOSCAN_LOG_ERROR("Channel readiness arm failed with {}",
                         error_code_name(armed.error().code));
      const RuntimeDiagnosticDetail detail{.present = true, .error = armed.error()};
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE", {}, {}, &detail);
      return false;
    }
    return true;
  }

  [[nodiscard]] PortBinding* physical_binding_for(ReadyRunState& state, SourceId source) noexcept {
    const auto found =
        std::find_if(state.bindings.begin(), state.bindings.end(), [&](const auto& binding) {
          return binding->source() && binding->source() == source;
        });
    return found == state.bindings.end() ? nullptr : found->get();
  }

  // Direct-chain admission derives one all-or-nothing topology and binding contract.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] bool initialize_direct_chain(GraphExec& exec, ReadyRunState& state) noexcept {
    constexpr std::size_t kNoReceiver = std::numeric_limits<std::size_t>::max();
    state.direct_chain_successor_by_receiver.assign(exec.operators().size(), kNoReceiver);
    state.direct_chain_binding_by_producer.assign(exec.operators().size(), nullptr);
    state.direct_chain_root_source = {};
    state.direct_chain_root_clock = {};
    state.direct_chain_root_binding = nullptr;
    state.direct_chain_root_receiver = kNoReceiver;
    state.direct_chain_next_root_wake.reset();
    if (state.runner == nullptr || state.runner->operators.size() < 2U ||
        !state.runner->incoming_boundary_edges.empty() ||
        !state.runner->outgoing_boundary_edges.empty() ||
        state.runner->internal_edges.size() + 1U != state.runner->operators.size() ||
        std::ranges::count_if(state.runner->operators,
                              [](const auto& member) { return member.topology_root; }) != 1 ||
        !state.runner->operators.front().topology_root ||
        state.runner->operators.front().trigger_kind != TemporalRecordTriggerKind::kClock ||
        !state.runner->operators.front().readiness_source) {
      return true;
    }
    // Preserve the already-proven two-Operator closed-island transaction. Besides being the
    // tighter execution shape, that transaction owns exact no-emission, failure, stop, and
    // generation-boundary semantics. The causal chain specializes longer compiler-proven chains;
    // installing both would create competing successor TagTime authorities.
    if (state.runner->operators.size() == 2U &&
        std::ranges::any_of(plan_->closed_island_activations,
                            [&](const ClosedIslandActivationPlanRecord& record) {
                              return record.process_group == exec.process_group() &&
                                     record.source == state.runner->operators[0].vertex &&
                                     record.successor == state.runner->operators[1].vertex &&
                                     record.source_partition == state.runner->partition &&
                                     record.successor_partition == state.runner->partition;
                            })) {
      return true;
    }
    const TemporalTriggerPlanRecord* const root_trigger =
        temporal_trigger_for(*plan_, state.runner->operators.front().operator_id);
    if (root_trigger == nullptr ||
        root_trigger->readiness_source != state.runner->operators.front().readiness_source ||
        !root_trigger->trigger_clock.valid() ||
        root_trigger->trigger_clock != root_trigger->reference_clock) {
      return true;
    }
    const auto root_instance =
        std::find_if(exec.operators().begin(), exec.operators().end(), [&](const auto& instance) {
          return instance.vertex == state.runner->operators.front().vertex &&
                 instance.operator_id == state.runner->operators.front().operator_id;
        });
    if (root_instance == exec.operators().end()) {
      return false;
    }
    const std::size_t root_receiver =
        static_cast<std::size_t>(std::distance(exec.operators().begin(), root_instance));
    if (root_receiver >= state.clock_bindings.size() ||
        state.clock_bindings[root_receiver] == nullptr ||
        state.clock_bindings[root_receiver]->source() != root_trigger->readiness_source) {
      return false;
    }
    for (std::size_t index = 1U; index < state.runner->operators.size(); ++index) {
      if (state.runner->operators[index].trigger_kind != TemporalRecordTriggerKind::kEach) {
        return true;
      }
    }

    std::vector<std::size_t> successor_by_receiver(exec.operators().size(), kNoReceiver);
    std::vector<PortBinding*> binding_by_producer(exec.operators().size(), nullptr);
    for (std::size_t index = 0U; index + 1U < state.runner->operators.size(); ++index) {
      const auto& producer = state.runner->operators[index];
      const auto& successor = state.runner->operators[index + 1U];
      const auto producer_instance =
          std::find_if(exec.operators().begin(), exec.operators().end(), [&](const auto& instance) {
            return instance.vertex == producer.vertex &&
                   instance.operator_id == producer.operator_id;
          });
      const auto successor_instance =
          std::find_if(exec.operators().begin(), exec.operators().end(), [&](const auto& instance) {
            return instance.vertex == successor.vertex &&
                   instance.operator_id == successor.operator_id;
          });
      if (producer_instance == exec.operators().end() ||
          successor_instance == exec.operators().end()) {
        return false;
      }
      const auto edge_count =
          std::ranges::count_if(state.runner->internal_edges, [&](EdgeId edge_id) {
            const auto edge = std::find_if(
                plan_->payload_edges.begin(),
                plan_->payload_edges.end(),
                [&](const PayloadEdgePlanRecord& candidate) { return candidate.edge == edge_id; });
            if (edge == plan_->payload_edges.end()) {
              return false;
            }
            const auto source = std::find_if(
                plan_->ports.begin(), plan_->ports.end(), [&](const PortPlanRecord& port) {
                  return port.id == edge->source;
                });
            const auto target = std::find_if(
                plan_->ports.begin(), plan_->ports.end(), [&](const PortPlanRecord& port) {
                  return port.id == edge->target;
                });
            return source != plan_->ports.end() && target != plan_->ports.end() &&
                   source->owner == producer.vertex && target->owner == successor.vertex;
          });
      if (edge_count != 1) {
        return true;
      }
      const std::size_t producer_receiver =
          static_cast<std::size_t>(std::distance(exec.operators().begin(), producer_instance));
      const std::size_t successor_receiver =
          static_cast<std::size_t>(std::distance(exec.operators().begin(), successor_instance));
      if (!owns_receiver(state, producer_receiver) || !owns_receiver(state, successor_receiver) ||
          successor_by_receiver[producer_receiver] != kNoReceiver) {
        return false;
      }
      PortBinding* const successor_binding =
          physical_binding_for(state, successor.readiness_source);
      if (successor_binding == nullptr || !successor.readiness_source ||
          successor_binding->source() != successor.readiness_source) {
        return false;
      }
      successor_by_receiver[producer_receiver] = successor_receiver;
      binding_by_producer[producer_receiver] = successor_binding;
    }
    state.direct_chain_successor_by_receiver = std::move(successor_by_receiver);
    state.direct_chain_binding_by_producer = std::move(binding_by_producer);
    state.direct_chain_root_source = state.runner->operators.front().readiness_source;
    state.direct_chain_root_clock = root_trigger->reference_clock;
    state.direct_chain_root_binding = state.clock_bindings[root_receiver];
    state.direct_chain_root_receiver = root_receiver;
    return true;
  }

  // Composite initialization preserves conservative fallback across every candidate island.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] bool initialize_direct_composites(GraphExec& exec, ReadyRunState& state) noexcept {
    state.direct_composites_by_receiver.assign(exec.operators().size(), {});
    state.direct_composite_owner_by_receiver.assign(exec.operators().size(),
                                                    std::numeric_limits<std::size_t>::max());
    if (!initialize_direct_chain(exec, state)) {
      return false;
    }
    // A strict chain owns every one of its causal edges. Do not also install a closed-island
    // composite for that same runner; its independent successor freeze would create a second
    // TagTime authority. Matching two-Operator closed islands were deliberately retained above
    // and therefore leave direct_chain_root_source empty.
    if (state.direct_chain_root_source) {
      return true;
    }
    if (!plan_->inline_partition_execution && !state.inline_partition_execution) {
      return true;
    }
    for (const ClosedIslandActivationPlanRecord& record : plan_->closed_island_activations) {
      if (record.process_group != exec.process_group()) {
        continue;
      }
      const auto producer =
          std::find_if(exec.operators().begin(), exec.operators().end(), [&](const auto& instance) {
            return instance.vertex == record.source;
          });
      const auto successor =
          std::find_if(exec.operators().begin(), exec.operators().end(), [&](const auto& instance) {
            return instance.vertex == record.successor;
          });
      if (producer == exec.operators().end() || successor == exec.operators().end() ||
          producer->partition != record.source_partition ||
          successor->partition != record.successor_partition ||
          producer->execution_lane != record.execution_lane ||
          successor->execution_lane != record.execution_lane) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", record.source);
        return false;
      }
      const std::size_t producer_receiver =
          static_cast<std::size_t>(std::distance(exec.operators().begin(), producer));
      const std::size_t successor_receiver =
          static_cast<std::size_t>(std::distance(exec.operators().begin(), successor));
      if (!owns_receiver(state, producer_receiver)) {
        continue;
      }
      if (!owns_receiver(state, successor_receiver)) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", record.source);
        return false;
      }
      const auto edge = std::find_if(plan_->payload_edges.begin(),
                                     plan_->payload_edges.end(),
                                     [&](const PayloadEdgePlanRecord& candidate) {
                                       return candidate.edge == record.edge &&
                                              candidate.source == record.source_output &&
                                              candidate.target == record.successor_input;
                                     });
      const TemporalTriggerPlanRecord* const source_trigger =
          temporal_trigger_for(*plan_, producer->operator_id);
      bool readiness_replayed = false;
      if (source_trigger != nullptr &&
          source_trigger->readiness_source == record.readiness_source) {
        if (record.readiness_root == ClosedIslandReadinessRootKind::kClock) {
          const auto request =
              std::find_if(state.clock_binding_requests.begin(),
                           state.clock_binding_requests.end(),
                           [&](const ReadyRunState::ClockBindingRequest& candidate) {
                             return candidate.receiver_index == producer_receiver &&
                                    candidate.source.kind == RuntimeEventKind::kClockTick &&
                                    candidate.source.source == record.readiness_source &&
                                    candidate.source.source_clock == record.readiness_clock &&
                                    candidate.source.reference_clock == record.readiness_clock &&
                                    candidate.schedule.clock == record.readiness_clock;
                           });
          readiness_replayed = request != state.clock_binding_requests.end();
        } else {
          const ReadinessPlanRecord* const readiness = readiness_for(*plan_, record.source);
          const bool notification =
              readiness != nullptr && readiness->kind == ReadinessSourceKind::kNotification;
          const auto request =
              std::find_if(state.binding_requests.begin(),
                           state.binding_requests.end(),
                           [&](const RuntimePortBindingRequest& candidate) {
                             return readiness != nullptr &&
                                    candidate.kind == runtime_event_kind(readiness->kind) &&
                                    candidate.source == record.readiness_source &&
                                    candidate.source_clock == record.readiness_clock &&
                                    candidate.reference_clock == record.readiness_clock &&
                                    candidate.execution_lane == record.execution_lane;
                           });
          readiness_replayed =
              readiness != nullptr && readiness->source_id == record.readiness_source &&
              readiness->resolved_clock == record.readiness_clock &&
              notification ==
                  (record.readiness_root == ClosedIslandReadinessRootKind::kNotification) &&
              request != state.binding_requests.end();
        }
      }
      if (edge == plan_->payload_edges.end() || !readiness_replayed) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", record.source);
        return false;
      }
      if (producer_receiver >= state.direct_composites_by_receiver.size() ||
          successor_receiver >= state.direct_composite_owner_by_receiver.size() ||
          state.direct_composites_by_receiver[producer_receiver].valid() ||
          state.direct_composite_owner_by_receiver[producer_receiver] !=
              std::numeric_limits<std::size_t>::max() ||
          state.direct_composite_owner_by_receiver[successor_receiver] !=
              std::numeric_limits<std::size_t>::max()) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", record.source);
        return false;
      }
      const auto request =
          std::find_if(state.channel_binding_requests.begin(),
                       state.channel_binding_requests.end(),
                       [&](const auto& candidate) {
                         return candidate.first == record.successor &&
                                candidate.second.kind == RuntimeEventKind::kInputArrival &&
                                candidate.second.execution_lane == record.execution_lane &&
                                !candidate.second.port &&
                                candidate.second.source == record.successor_readiness_source;
                       });
      if (request == state.channel_binding_requests.end()) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", record.successor);
        return false;
      }
      PortBinding* const successor_binding = physical_binding_for(state, request->second.source);
      if (successor_binding == nullptr || successor_binding->poll_target().has_value()) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", record.successor);
        return false;
      }
      state.direct_composites_by_receiver[producer_receiver] = RuntimeDirectCompositeBinding{
          .producer_receiver = producer_receiver,
          .successor_receiver = successor_receiver,
          .producer = record.source,
          .successor = record.successor,
          .source_output = record.source_output,
          .successor_input = record.successor_input,
          .readiness_source = record.successor_readiness_source,
          .successor_binding = successor_binding,
          .execution_lane = record.execution_lane,
      };
      state.direct_composite_owner_by_receiver[producer_receiver] = producer_receiver;
      state.direct_composite_owner_by_receiver[successor_receiver] = producer_receiver;
    }
    return true;
  }

  [[nodiscard]] expected<void, Error> retire_frozen_source(
      ReadyRunState& state, FrozenRuntimeCandidate& candidate) noexcept {
    if (candidate.source_retired) {
      return make_unexpected(Error{ErrorCode::kStaleEndpoint});
    }
    auto retired = state.coordinator->retire(candidate);
    if (!retired) {
      return make_unexpected(std::move(retired.error()));
    }
    // Once coordinator credit is returned, retrying this candidate would be a
    // double retirement even if the binding reports a terminal failure.
    candidate.source_retired = true;
    PortBinding* const binding = physical_binding_for(state, candidate.witness.source);
    return binding == nullptr ? expected<void, Error>{}
                              : binding->retire(candidate.witness.source_sequence);
  }

  [[nodiscard]] bool cancel_ready_sources(ReadyRunState& state) noexcept {
    bool retired_all = true;
    if (state.close_shared_notifications && state.notifications != nullptr) {
      state.notifications->close();
    }
    // Every pending witness is frozen before dispatch begins. Retire that
    // complete frozen prefix exactly once, then let coordinator cancellation
    // release any witnesses left pending by a freeze failure.
    for (auto& candidate : state.frozen_candidates) {
      if (!candidate.source_retired && !retire_frozen_source(state, candidate)) {
        retired_all = false;
      }
    }
    state.coordinator->cancel_all();
    for (auto& binding : state.bindings) {
      binding->cancel();
    }
    return retired_all;
  }

  [[nodiscard]] bool stop_channels(ReadyRunState& state) noexcept {
    if (state.channels == nullptr) {
      return true;
    }
    state.channels->cancel();
    const auto joined = state.channels->join();
    if (!joined) {
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
      return false;
    }
    return true;
  }

  // Validate every member's pending-window state before publishing a timeout witness for the
  // operator, preserving one coherent closure and clock coordinate.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] bool enqueue_due_window_timeouts(GraphExec& exec, ReadyRunState& state) noexcept {
    for (const WindowTriggerPlanRecord& window : plan_->window_triggers) {
      const auto instance = std::find_if(
          exec.operators().begin(), exec.operators().end(), [&](const auto& candidate) {
            return candidate.operator_id == window.owner.id &&
                   window.owner.plan_generation == plan_->plan_generation;
          });
      if (instance == exec.operators().end()) {
        continue;
      }
      const std::size_t receiver =
          static_cast<std::size_t>(std::distance(exec.operators().begin(), instance));
      if (!owns_receiver(state, receiver)) {
        continue;
      }
      if (window.bracketed_inputs.empty() || window.max_wait_ns <= 0 ||
          !window.wait_clock.valid()) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
        return false;
      }

      std::optional<std::size_t> pending_count;
      std::optional<TimePoint> oldest_reference;
      for (const PortId input : window.bracketed_inputs) {
        const auto selector = std::find_if(
            state.input_selectors.begin(), state.input_selectors.end(), [&](const auto& slot) {
              return slot.owner == instance->vertex && slot.input == input;
            });
        if (selector == state.input_selectors.end()) {
          append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance->vertex);
          return false;
        }
        const std::size_t count = selector->selector.pending_window_count();
        const std::optional<TimePoint> reference =
            selector->selector.oldest_pending_window_reference();
        if ((count == 0U) != !reference || (pending_count && *pending_count != count) ||
            (oldest_reference && reference && *oldest_reference != *reference)) {
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance->vertex);
          return false;
        }
        pending_count = count;
        if (reference) {
          oldest_reference = *reference;
        }
      }
      if (!pending_count || *pending_count == 0U) {
        continue;
      }
      if (!oldest_reference || oldest_reference->clock_id != window.wait_clock ||
          oldest_reference->timestamp_ns >
              std::numeric_limits<std::int64_t>::max() - window.max_wait_ns) {
        append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance->vertex);
        return false;
      }
      auto clock = exec.resolve_clock(window.wait_clock);
      auto now = clock ? clock->now() : expected<TimePoint, Error>{make_unexpected(clock.error())};
      if (!now || now->clock_id != window.wait_clock) {
        append_diagnostic("RUNTIME_CLOCK_READ_FAILED", instance->vertex);
        return false;
      }
      const std::int64_t deadline_ns = oldest_reference->timestamp_ns + window.max_wait_ns;
      if (now->timestamp_ns < deadline_ns) {
        continue;
      }
      const TemporalTriggerPlanRecord* const trigger =
          temporal_trigger_for(*plan_, instance->operator_id);
      PortBinding* const binding =
          trigger == nullptr ? nullptr : physical_binding_for(state, trigger->readiness_source);
      if (trigger == nullptr || trigger->kind != TemporalRecordTriggerKind::kWindow ||
          trigger->trigger_input != window.anchor || binding == nullptr) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance->vertex);
        return false;
      }
      auto enqueued = binding->enqueue_window_timeout(window.anchor, *now);
      if (!enqueued && enqueued.error().code != ErrorCode::kBackpressured) {
        append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance->vertex);
        return false;
      }
      if (enqueued && plan_->executable_factories.test_window_timeout_control != nullptr) {
        plan_->executable_factories.test_window_timeout_control->evaluated(
            state.runner != nullptr ? state.runner->partition : ExecutionPartitionId{},
            instance->vertex);
      }
    }
    return true;
  }

  [[nodiscard]] bool arrival_trigger_pin_matches(
      const RuntimeOperatorInstance& instance,
      const FrozenRuntimeCandidate& candidate) const noexcept {
    if (candidate.witness.kind != RuntimeEventKind::kInputArrival || !candidate.witness.port ||
        candidate.witness.source_sequence == 0U) {
      return false;
    }
    const TemporalTriggerPlanRecord* const trigger =
        temporal_trigger_for(*plan_, instance.operator_id);
    if (trigger == nullptr || trigger->owner.plan_generation != plan_->plan_generation) {
      return false;
    }
    if (trigger->kind == TemporalRecordTriggerKind::kEach) {
      return trigger->trigger_input == candidate.witness.port;
    }
    if (trigger->kind != TemporalRecordTriggerKind::kAnyInput) {
      return false;
    }
    // OnAnyInput is defined over every connected input, including ports without an explicit
    // temporal selector. Those ports intentionally do not appear in declared_inputs, so validate
    // the witness against the frozen physical input inventory instead of the selector inventory.
    return std::any_of(plan_->ports.begin(), plan_->ports.end(), [&](const PortPlanRecord& port) {
      return port.id == candidate.witness.port && port.owner == instance.vertex &&
             port.owner_kind == PortOwnerKind::kOperator && port.direction == PortDirection::kInput;
    });
  }

  [[nodiscard]] bool arrival_trigger_pin_matches(
      const GraphExec& exec, const FrozenRuntimeCandidate& candidate) const noexcept {
    return candidate.candidate.receiver_index < exec.operators().size() &&
           arrival_trigger_pin_matches(exec.operators()[candidate.candidate.receiver_index],
                                       candidate);
  }

  [[nodiscard]] expected<void, Error> synchronize_temporal_payload_claims(
      ReadyRunState& state, ReadyRunState::InputSelectorSlot& slot) noexcept {
    if (state.channels == nullptr) {
      slot.channel_payload_claims.clear();
      return {};
    }
    auto claims = [&]() -> expected<std::size_t, Error> {
      if (!slot.aligned_member) {
        return slot.selector.copy_payload_claims(slot.channel_payload_claim_scratch);
      }
      const auto aligned = std::find_if(state.aligned_selectors.begin(),
                                        state.aligned_selectors.end(),
                                        [&](const auto& row) { return row.owner == slot.owner; });
      return aligned == state.aligned_selectors.end()
                 ? expected<std::size_t, Error>{make_unexpected(Error{ErrorCode::kInvalidArgument})}
                 : aligned->selector.copy_payload_claims(slot.input,
                                                         slot.channel_payload_claim_scratch);
    }();
    if (!claims || *claims > slot.channel_payload_claim_scratch.size()) {
      return make_unexpected(claims ? Error{ErrorCode::kProtocolError} : std::move(claims).error());
    }

    const auto less = [](const TemporalSampleRecord& left,
                         const TemporalSampleRecord& right) noexcept {
      return std::tie(left.producer_epoch,
                      left.arrival_sequence,
                      left.source_sequence,
                      left.capture_time.clock_id.value,
                      left.capture_time.timestamp_ns) < std::tie(right.producer_epoch,
                                                                 right.arrival_sequence,
                                                                 right.source_sequence,
                                                                 right.capture_time.clock_id.value,
                                                                 right.capture_time.timestamp_ns);
    };
    auto scratch_begin = slot.channel_payload_claim_scratch.begin();
    auto scratch_end = scratch_begin + static_cast<std::ptrdiff_t>(*claims);
    std::sort(scratch_begin, scratch_end, less);
    scratch_end = std::unique(scratch_begin, scratch_end);

    std::size_t retired_count = 0U;
    for (const TemporalSampleRecord& prior : slot.channel_payload_claims) {
      if (!std::binary_search(scratch_begin, scratch_end, prior, less)) {
        if (retired_count == slot.channel_payload_retire_scratch.size()) {
          return make_unexpected(Error{ErrorCode::kProtocolError});
        }
        slot.channel_payload_retire_scratch[retired_count++] = prior;
      }
    }
    if (retired_count != 0U) {
      // Channel validates and retires the complete delta atomically. Supplying one preallocated
      // batch prevents a malformed later record from releasing any earlier selector claim.
      auto released = state.channels->retire_samples(
          slot.input,
          std::span<const TemporalSampleRecord>{slot.channel_payload_retire_scratch}.first(
              retired_count));
      if (!released) {
        return make_unexpected(std::move(released).error());
      }
    }
    slot.channel_payload_claims.clear();
    slot.channel_payload_claims.insert(
        slot.channel_payload_claims.end(), scratch_begin, scratch_end);
    return {};
  }

  [[nodiscard]] bool synchronize_temporal_payload_claims(ReadyRunState& state,
                                                         VertexId owner) noexcept {
    for (auto& slot : state.input_selectors) {
      if (slot.owner != owner) {
        continue;
      }
      if (!synchronize_temporal_payload_claims(state, slot)) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool synchronize_temporal_payload_claims(ReadyRunState& state) noexcept {
    for (auto& slot : state.input_selectors) {
      if (!synchronize_temporal_payload_claims(state, slot)) {
        return false;
      }
    }
    return true;
  }

  // Channel ingestion atomically reconciles producer epochs, temporal selectors, bracket anchors,
  // predicate state, and payload claims for one frozen witness.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] bool ingest_channel_candidate(GraphExec& exec, ReadyRunState& state,
                                              const FrozenRuntimeCandidate& candidate) {
    if (candidate.witness.kind == RuntimeEventKind::kInputEndOfStream) {
      if (!candidate.witness.port || !candidate.witness.reference_time.valid() ||
          candidate.candidate.receiver_index >= exec.operators().size()) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
        return false;
      }
      const RuntimeOperatorInstance& instance =
          exec.operators()[candidate.candidate.receiver_index];
      bool closed = false;
      for (auto& selector : state.input_selectors) {
        if (selector.owner != instance.vertex || selector.input != candidate.witness.port) {
          continue;
        }
        if (selector.aligned_member) {
          const auto aligned =
              std::find_if(state.aligned_selectors.begin(),
                           state.aligned_selectors.end(),
                           [&](const auto& slot) { return slot.owner == selector.owner; });
          if (aligned == state.aligned_selectors.end() ||
              !aligned->selector.close_end_of_stream(selector.input,
                                                     candidate.witness.reference_time)) {
            append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", selector.owner);
            return false;
          }
        } else if (!selector.selector.close_end_of_stream()) {
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", selector.owner);
          return false;
        }
        selector.end_of_stream_received = true;
        closed = true;
      }
      if (closed && !synchronize_temporal_payload_claims(state, instance.vertex)) {
        append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
        return false;
      }
      return true;
    }
    if (candidate.witness.kind == RuntimeEventKind::kWindowTimeout) {
      if (!candidate.witness.port ||
          candidate.candidate.receiver_index >= exec.operators().size()) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
        return false;
      }
      const RuntimeOperatorInstance& instance =
          exec.operators()[candidate.candidate.receiver_index];
      const auto window =
          std::find_if(plan_->window_triggers.begin(),
                       plan_->window_triggers.end(),
                       [&](const WindowTriggerPlanRecord& record) {
                         return record.owner.id == instance.operator_id &&
                                record.owner.plan_generation == plan_->plan_generation &&
                                record.anchor == candidate.witness.port;
                       });
      if (window == plan_->window_triggers.end()) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
        return false;
      }
      std::size_t closed_count = 0U;
      for (auto& selector : state.input_selectors) {
        if (selector.owner != instance.vertex ||
            std::find(window->bracketed_inputs.begin(),
                      window->bracketed_inputs.end(),
                      selector.input) == window->bracketed_inputs.end()) {
          continue;
        }
        if (!selector.selector.close_oldest(SelectionReason::kWindowTimeout)) {
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
          return false;
        }
        ++closed_count;
      }
      if (closed_count != window->bracketed_inputs.size()) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
        return false;
      }
      if (!synchronize_temporal_payload_claims(state, instance.vertex)) {
        append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
        return false;
      }
      return true;
    }
    if (candidate.witness.kind != RuntimeEventKind::kInputArrival) {
      return true;
    }
    if (state.channels == nullptr || !candidate.witness.port ||
        candidate.witness.source_sequence == 0U) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
      return false;
    }
    auto sample =
        state.channels->temporal_sample(candidate.witness.port, candidate.witness.source_sequence);
    if (!sample) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
      return false;
    }
    bool consumed = false;
    for (auto& selector : state.input_selectors) {
      if (selector.input == candidate.witness.port) {
        if (selector.arrival_watermark != 0U && selector.producer_epoch != sample->producer_epoch) {
          if (selector.aligned_member) {
            const auto aligned =
                std::find_if(state.aligned_selectors.begin(),
                             state.aligned_selectors.end(),
                             [&](const auto& slot) { return slot.owner == selector.owner; });
            if (aligned == state.aligned_selectors.end() || !aligned->selector.reset()) {
              append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", selector.owner);
              return false;
            }
            for (auto& member : state.input_selectors) {
              if (member.owner == selector.owner && member.aligned_member) {
                member.arrival_watermark = 0U;
                member.end_of_stream_received = false;
              }
            }
          } else if (!selector.selector.reset()) {
            append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", selector.owner);
            return false;
          } else {
            selector.end_of_stream_received = false;
          }
          const auto instance = std::find_if(
              exec.operators().begin(), exec.operators().end(), [&](const auto& operator_instance) {
                return operator_instance.vertex == selector.owner;
              });
          if (instance == exec.operators().end()) {
            append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", selector.owner);
            return false;
          }
          const std::size_t receiver =
              static_cast<std::size_t>(std::distance(exec.operators().begin(), instance));
          if (state.predicates[receiver]) {
            state.predicates[receiver]->reset(PredicateResetReason::kProducerEpoch,
                                              instance->tags->generation());
          }
        }
        const auto pushed = [&]() -> expected<void, Error> {
          if (!selector.aligned_member) {
            return selector.selector.push(*sample);
          }
          const auto aligned =
              std::find_if(state.aligned_selectors.begin(),
                           state.aligned_selectors.end(),
                           [&](const auto& slot) { return slot.owner == selector.owner; });
          return aligned == state.aligned_selectors.end()
                     ? expected<void, Error>{make_unexpected(Error{ErrorCode::kInvalidArgument})}
                     : aligned->selector.push(selector.input, *sample);
        }();
        if (!pushed) {
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", selector.owner);
          return false;
        }
        selector.arrival_watermark = sample->arrival_sequence;
        selector.producer_epoch = sample->producer_epoch;
        consumed = true;
        if (!synchronize_temporal_payload_claims(state, selector.owner)) {
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", selector.owner);
          return false;
        }
      }
      const auto bracketed = std::find_if(
          plan_->bracketed_windows.begin(), plan_->bracketed_windows.end(), [&](const auto& row) {
            return row.input == selector.input && row.anchor == candidate.witness.port;
          });
      if (bracketed != plan_->bracketed_windows.end()) {
        consumed = true;
        // EOS terminates this bracketed input generation. Later anchors remain valid for their
        // own selectors, but must not create windows that have no remaining EOS event to close.
        if (!selector.end_of_stream_received) {
          auto advanced = selector.selector.push_anchor(*sample);
          if (!advanced) {
            append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", selector.owner);
            return false;
          }
          if (!synchronize_temporal_payload_claims(state, selector.owner)) {
            append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", selector.owner);
            return false;
          }
        }
      }
    }
    if (!consumed && !arrival_trigger_pin_matches(exec, candidate)) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
      return false;
    }
    const bool payload_retained = std::any_of(
        state.input_selectors.begin(), state.input_selectors.end(), [&](const auto& selector) {
          return selector.input == candidate.witness.port &&
                 std::find(selector.channel_payload_claims.begin(),
                           selector.channel_payload_claims.end(),
                           *sample) != selector.channel_payload_claims.end();
        });
    if (!payload_retained) {
      // Anchor-only arrivals and direct OnEach/OnAny trigger pins retain only metadata in the
      // selector layer. Release their temporal claim now; the unretired readiness witness keeps
      // direct-pin storage alive through prepare/compute, then permits bounded reclamation.
      const std::array unretained{*sample};
      auto released = state.channels->retire_samples(candidate.witness.port, unretained);
      if (!released) {
        append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE");
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool channel_candidate_is_eligible(
      const GraphExec& exec, const ReadyRunState& state,
      const FrozenRuntimeCandidate& candidate) const noexcept {
    if (candidate.candidate.receiver_index >= exec.operators().size()) {
      return false;
    }
    const RuntimeOperatorInstance& instance = exec.operators()[candidate.candidate.receiver_index];
    const TemporalTriggerPlanRecord* const trigger =
        temporal_trigger_for(*plan_, instance.operator_id);
    if (trigger == nullptr) {
      return false;
    }
    if (candidate.witness.kind == RuntimeEventKind::kWindowTimeout) {
      return trigger->kind == TemporalRecordTriggerKind::kWindow &&
             candidate.witness.port == trigger->trigger_input;
    }
    const auto input_witness_can_activate = [&]() noexcept {
      if (trigger->kind == TemporalRecordTriggerKind::kEach) {
        return candidate.witness.port == trigger->trigger_input;
      }
      if (trigger->kind == TemporalRecordTriggerKind::kPredicate) {
        const PredicateTriggerPlanRecord* const predicate =
            predicate_trigger_for(*plan_, trigger->owner);
        return predicate != nullptr &&
               predicate->source == PredicateEvaluationSourceKind::kInputUpdate &&
               candidate.witness.port == predicate->driver;
      }
      return trigger->kind == TemporalRecordTriggerKind::kAnyInput ||
             trigger->kind == TemporalRecordTriggerKind::kAligned ||
             trigger->kind == TemporalRecordTriggerKind::kWindow;
    };
    if (candidate.witness.kind == RuntimeEventKind::kInputEndOfStream) {
      const bool witness_matches = std::any_of(
          state.input_selectors.begin(), state.input_selectors.end(), [&](const auto& selector) {
            return selector.owner == instance.vertex && selector.input == candidate.witness.port;
          });
      return input_witness_can_activate() && witness_matches &&
             std::all_of(state.input_selectors.begin(),
                         state.input_selectors.end(),
                         [&](const auto& selector) {
                           return selector.owner != instance.vertex ||
                                  selector.end_of_stream_received;
                         });
    }
    if (candidate.witness.kind != RuntimeEventKind::kInputArrival) {
      return true;
    }
    // Only input-backed trigger families may turn an arrival or EOS witness into an activation.
    // Clock and readiness-driven Operators still ingest side-input state and retire its readiness
    // credit here, but the independent source remains their sole activation authority. Window
    // successors remain eligible because an input witness can close the frozen bracket.
    return input_witness_can_activate();
  }

  // Form the bounded activation snapshot, health disposition, and retirement obligations in one
  // pass so dispatch never observes a partially frozen candidate set.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] bool freeze_ready_candidates(
      GraphExec& exec, ReadyRunState& state,
      std::optional<std::uint32_t> current_execution_lane = std::nullopt) {
    if (exec.operators().empty()) {
      return true;
    }
    if (std::any_of(state.frozen_candidates.begin(),
                    state.frozen_candidates.end(),
                    [](const auto& candidate) { return !candidate.source_retired; })) {
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
      static_cast<void>(cancel_ready_sources(state));
      return false;
    }
    state.candidates.clear();
    state.frozen_candidates.clear();
    expected<std::size_t, EndpointClockHealthError> expired{0U};
    if (state.clock_health->receiver_count() != 0U) {
      const std::lock_guard clock_health_lock(clock_health_mutex_);
      expired = state.clock_health->expire_due();
    }
    if (!expired) {
      append_diagnostic(expired.error().token.empty()
                            ? std::string_view{"RUNTIME_CLOCK_READ_FAILED"}
                            : expired.error().token);
      return false;
    }
    if (state.channels != nullptr) {
      auto channel_status = state.channels->status();
      if (!channel_status) {
        const RuntimeDiagnosticDetail detail{.present = true, .error = channel_status.error()};
        append_diagnostic(channel_terminal_token(state.channels->terminal_diagnostic()),
                          state.channels->terminal_owner(),
                          {},
                          &detail);
        return false;
      }
    }
    std::size_t ingested_count = 0U;
    for (auto& binding : state.bindings) {
      // The coordinator bound is the sum of all source-local bounds. Once it is full, every
      // source already has its admitted prefix represented, so polling another binding can only
      // expose a successor that must remain behind that prefix. Freeze the deterministic minimum
      // first instead of treating the expected bounded condition as a Runtime failure.
      if (state.coordinator->outstanding_count() == state.coordinator->max_outstanding()) {
        break;
      }
      // A different source may leave room in the global bound while this binding's complete
      // source-local prefix is already pending. Do not poll or drain a successor until that older
      // witness retires: drain is destructive, and rejecting the removed successor as
      // backpressured would lose its credit and fail an otherwise valid mixed-source run.
      const auto source_credit = state.coordinator->source_credit_remaining(binding->source());
      if (!source_credit) {
        append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
        static_cast<void>(cancel_ready_sources(state));
        return false;
      }
      if (*source_credit == 0U) {
        continue;
      }
      bool direct_continuation_poll_elided = false;
      if (state.direct_continuation_receipt &&
          state.direct_continuation_receipt->readiness_source == binding->source()) {
        direct_continuation_poll_elided = state.channels != nullptr &&
                                          state.channels->validate_direct_continuation_receipt(
                                              *state.direct_continuation_receipt) &&
                                          !binding->poll_target().has_value();
        state.direct_continuation_receipt.reset();
        if (direct_continuation_poll_elided) {
          count_phase(RuntimePhaseCounter::kDirectContinuationReceiptsAccepted);
          count_phase(RuntimePhaseCounter::kDirectContinuationPollElisions);
        }
      }
      const std::optional<PortBindingPollTarget> target =
          direct_continuation_poll_elided ? std::nullopt : binding->poll_target();
      if (target) {
        auto clock = exec.resolve_clock(target->reference_clock);
        auto observed_at =
            clock ? clock->now() : expected<TimePoint, Error>{make_unexpected(clock.error())};
        if (!observed_at || observed_at->clock_id != target->reference_clock) {
          append_diagnostic("RUNTIME_CLOCK_READ_FAILED");
          static_cast<void>(cancel_ready_sources(state));
          return false;
        }
        struct PollInvocation {
          PortBinding* binding{};
          TimePoint observed_at{};
        } invocation{.binding = binding.get(), .observed_at = *observed_at};
        if (!current_execution_lane) {
          count_phase(RuntimePhaseCounter::kReadinessPollDispatches);
        }
        expected<void, Error> polled;
        if (current_execution_lane) {
          if (*current_execution_lane != target->execution_lane) {
            append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
            static_cast<void>(cancel_ready_sources(state));
            return false;
          }
          polled = binding->poll(*observed_at);
        } else {
          polled = state.lanes->dispatch(
              target->execution_lane,
              ExecutionLaneTask{
                  .context = &invocation,
                  .invoke = [](void* state) noexcept -> expected<void, Error> {
                    auto& task = *static_cast<PollInvocation*>(state);
                    return task.binding->poll(task.observed_at);
                  },
                  .callback_context = RuntimeCallbackContext::kDispatcher,
              });
        }
        if (!polled) {
          const auto terminal_diagnostic =
              state.channels != nullptr ? state.channels->terminal_diagnostic() : std::nullopt;
          VertexId vertex{};
          if (terminal_diagnostic) {
            vertex = state.channels->terminal_owner();
          } else {
            const auto receiver = state.coordinator->source_receiver(binding->source());
            vertex = receiver && *receiver < exec.operators().size()
                         ? exec.operators()[*receiver].vertex
                         : VertexId{};
          }
          const RuntimeDiagnosticDetail detail{
              .present = true, .error = polled.error(), .source = binding->source()};
          append_diagnostic(channel_terminal_token(terminal_diagnostic), vertex, {}, &detail);
          static_cast<void>(cancel_ready_sources(state));
          return false;
        }
        if (!current_execution_lane) {
          trace_phase(RuntimePhaseTracePoint::kReadinessPollReturned,
                      kNoRuntimePhaseReceiver,
                      binding->source(),
                      0U);
        }
      }
      auto ingested = state.coordinator->ingest(*binding, binding->source(), *source_credit);
      if (!ingested || ingested_count > std::numeric_limits<std::size_t>::max() - *ingested) {
        append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
        static_cast<void>(cancel_ready_sources(state));
        return false;
      }
      ingested_count += *ingested;
    }
    static_cast<void>(ingested_count);
    bool freeze_failed = false;
    const auto freeze_one_eligible = [&]() -> bool {
      while (state.coordinator->pending_count() != 0U) {
        auto candidate = state.coordinator->freeze_next();
        if (!candidate) {
          HOLOSCAN_LOG_ERROR("Runtime candidate freeze failed with {}: pending={} outstanding={}",
                             error_code_name(candidate.error().code),
                             state.coordinator->pending_count(),
                             state.coordinator->outstanding_count());
          const RuntimeDiagnosticDetail detail{.present = true, .error = candidate.error()};
          append_diagnostic(candidate.error().code == ErrorCode::kResourceExhausted
                                ? std::string_view{"RUNTIME_TAG_CAPACITY_EXHAUSTED"}
                                : std::string_view{"RUNTIME_READINESS_BINDING_UNAVAILABLE"},
                            {},
                            "candidate_freeze",
                            &detail);
          freeze_failed = true;
          return false;
        }
        if (!ingest_channel_candidate(exec, state, *candidate)) {
          freeze_failed = true;
          return false;
        }
        if (!channel_candidate_is_eligible(exec, state, *candidate)) {
          auto retired = retire_frozen_source(state, *candidate);
          if (!retired) {
            append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
            freeze_failed = true;
            return false;
          }
          continue;
        }
        // Serial-baseline dispatch exposes exactly the selector state established by this
        // witness. Leaving later accepted events in the coordinator prevents a successor or
        // timeout from closing a window before the earlier candidate has committed or rolled back.
        state.candidates.push_back(candidate->candidate);
        state.frozen_candidates.push_back(std::move(*candidate));
        count_phase(RuntimePhaseCounter::kCandidatesFrozen);
        trace_phase(RuntimePhaseTracePoint::kCandidateFrozen,
                    state.frozen_candidates.back().candidate.receiver_index,
                    state.frozen_candidates.back().witness.source,
                    state.frozen_candidates.back().witness.source_sequence);
        return true;
      }
      return false;
    };

    bool candidate_ready = freeze_one_eligible();
    if (!candidate_ready && !freeze_failed) {
      if (!enqueue_due_window_timeouts(exec, state)) {
        freeze_failed = true;
      } else {
        for (auto& binding : state.bindings) {
          if (state.coordinator->outstanding_count() == state.coordinator->max_outstanding()) {
            break;
          }
          const auto source_credit = state.coordinator->source_credit_remaining(binding->source());
          if (!source_credit) {
            append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
            freeze_failed = true;
            break;
          }
          if (*source_credit == 0U) {
            continue;
          }
          auto ingested = state.coordinator->ingest(*binding, binding->source(), *source_credit);
          if (!ingested) {
            append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
            freeze_failed = true;
            break;
          }
        }
        candidate_ready = !freeze_failed && freeze_one_eligible();
      }
    }
    if (freeze_failed) {
      static_cast<void>(cancel_ready_sources(state));
      return false;
    }
    if (!candidate_ready) {
      return true;
    }
    const RuntimeCallbackContextScope callback_scope{RuntimeCallbackContext::kDispatcher};
    auto frozen = state.dispatcher->freeze(state.candidates);
    if (!frozen) {
      static_cast<void>(cancel_ready_sources(state));
      append_diagnostic("READY_POLICY_PLAN_INVALID");
      return false;
    }
    // Health-suppressed candidates reserve their receiver-local Tag before
    // eligibility is known, but never enter compute. Release source credit now
    // so a later direct-runtime freeze cannot inherit a phantom outstanding
    // event; the Tag itself remains consumed.
    for (auto& candidate : state.frozen_candidates) {
      if (candidate.candidate.clock_health_disposition == EndpointClockHealthDisposition::kSkip &&
          !retire_frozen_source(state, candidate)) {
        static_cast<void>(cancel_ready_sources(state));
        append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
        return false;
      }
    }
    return true;
  }

  enum class CausalFrontierDisposition : std::uint8_t {
    kAccepted,
    kHandled,
    kFallback,
    kFailed,
  };

  // Causal continuation validation and fallback form one ordered frontier transition.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] CausalFrontierDisposition freeze_causal_successor(
      GraphExec& exec, ReadyRunState& state, std::size_t producer_receiver,
      TagTime parent_tag_time) noexcept {
    constexpr std::size_t kNoReceiver = std::numeric_limits<std::size_t>::max();
    if (!state.direct_continuation_receipt ||
        producer_receiver >= state.direct_chain_successor_by_receiver.size() ||
        state.direct_chain_successor_by_receiver[producer_receiver] == kNoReceiver) {
      return CausalFrontierDisposition::kFallback;
    }
    count_phase(RuntimePhaseCounter::kPartitionCausalFrontierAttempts);
    const auto fallback = [&]() noexcept {
      count_phase(RuntimePhaseCounter::kPartitionCausalFrontierFallbacks);
      return CausalFrontierDisposition::kFallback;
    };
    if (stop_requested_.load(std::memory_order_acquire) || lifecycle_control_->has_pending() ||
        state.channels == nullptr || state.dispatcher->active_count() != 0U ||
        state.coordinator->pending_count() != 0U || state.coordinator->outstanding_count() != 0U ||
        !state.direct_chain_root_source || !state.direct_chain_root_clock.valid() ||
        state.direct_chain_root_binding == nullptr || !parent_tag_time.valid()) {
      return fallback();
    }

    const RuntimeChannelDirectContinuationReceipt receipt = *state.direct_continuation_receipt;
    const std::size_t successor_receiver =
        state.direct_chain_successor_by_receiver[producer_receiver];
    PortBinding* const successor_binding =
        producer_receiver < state.direct_chain_binding_by_producer.size()
            ? state.direct_chain_binding_by_producer[producer_receiver]
            : nullptr;
    const auto receipt_receiver = state.coordinator->source_receiver(receipt.readiness_source);
    if (successor_receiver >= exec.operators().size() || !receipt_receiver ||
        *receipt_receiver != successor_receiver ||
        receipt.destination != exec.operators()[successor_receiver].vertex ||
        successor_binding == nullptr || successor_binding->source() != receipt.readiness_source ||
        successor_binding->poll_target().has_value() ||
        !state.channels->validate_direct_continuation_receipt(receipt)) {
      return fallback();
    }

    // The compiler-proven chain gives the receipt-proven successor the next
    // microstep at its parent's logical coordinate. Confirm that the root's
    // next scheduled coordinate is strictly later before ingesting the
    // receipt; otherwise leave the receipt intact for the ordinary
    // deterministic READY-set path.
    if (producer_receiver == state.direct_chain_root_receiver ||
        !state.direct_chain_next_root_wake) {
      state.direct_chain_next_root_wake = state.direct_chain_root_binding->next_wake();
    }
    if (!state.direct_chain_next_root_wake) {
      return fallback();
    }
    if (state.direct_chain_next_root_wake->clock_id != state.direct_chain_root_clock ||
        state.direct_chain_next_root_wake->timestamp_ns <= parent_tag_time.logical_ns) {
      return fallback();
    }

    // The cold chain proof binds the sole successor source. Keep the common
    // health/terminal prelude, then ingest that exact binding instead of
    // rescanning every partition source and validating the receipt twice.
    if (std::ranges::any_of(state.frozen_candidates,
                            [](const auto& candidate) { return !candidate.source_retired; })) {
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
      static_cast<void>(cancel_ready_sources(state));
      return CausalFrontierDisposition::kFailed;
    }
    state.candidates.clear();
    state.frozen_candidates.clear();
    expected<std::size_t, EndpointClockHealthError> expired{0U};
    if (state.clock_health->receiver_count() != 0U) {
      const std::lock_guard clock_health_lock(clock_health_mutex_);
      expired = state.clock_health->expire_due();
    }
    if (!expired) {
      append_diagnostic(expired.error().token.empty()
                            ? std::string_view{"RUNTIME_CLOCK_READ_FAILED"}
                            : expired.error().token);
      return CausalFrontierDisposition::kFailed;
    }
    auto channel_status = state.channels->status();
    if (!channel_status) {
      const RuntimeDiagnosticDetail detail{.present = true, .error = channel_status.error()};
      append_diagnostic(channel_terminal_token(state.channels->terminal_diagnostic()),
                        state.channels->terminal_owner(),
                        {},
                        &detail);
      return CausalFrontierDisposition::kFailed;
    }
    const auto source_credit = state.coordinator->source_credit_remaining(receipt.readiness_source);
    if (!source_credit || *source_credit == 0U) {
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE",
                        exec.operators()[successor_receiver].vertex,
                        "partition_causal_frontier_credit");
      static_cast<void>(cancel_ready_sources(state));
      return CausalFrontierDisposition::kFailed;
    }
    // Consume the receipt before freeze_causal_next() because that operation assumes exclusive
    // authority over the sole pending event. Ingress is destructive: any failure below is terminal,
    // and cancel_ready_sources() retires coordinator credit rather than attempting an ordinary
    // poll that cannot recover the already-accepted continuation witness.
    state.direct_continuation_receipt.reset();
    count_phase(RuntimePhaseCounter::kDirectContinuationReceiptsAccepted);
    count_phase(RuntimePhaseCounter::kDirectContinuationPollElisions);
    const auto ingested =
        state.coordinator->ingest(*successor_binding, receipt.readiness_source, *source_credit);
    if (!ingested || *ingested != 1U) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH",
                        exec.operators()[successor_receiver].vertex,
                        "partition_causal_frontier_ingress");
      static_cast<void>(cancel_ready_sources(state));
      return CausalFrontierDisposition::kFailed;
    }
    const bool force_causal_freeze_failure =
        plan_->executable_factories.test_direct_continuation_control != nullptr &&
        plan_->executable_factories.test_direct_continuation_control
            ->consume_causal_freeze_failure();
    auto candidate = state.coordinator->freeze_causal_next(
        force_causal_freeze_failure ? SourceId{} : receipt.readiness_source, parent_tag_time);
    if (!candidate) {
      const RuntimeDiagnosticDetail detail{.present = true, .error = candidate.error()};
      append_diagnostic(candidate.error().code == ErrorCode::kResourceExhausted
                            ? std::string_view{"RUNTIME_TAG_CAPACITY_EXHAUSTED"}
                            : std::string_view{"RUNTIME_READINESS_BINDING_UNAVAILABLE"},
                        exec.operators()[successor_receiver].vertex,
                        "partition_causal_frontier_freeze",
                        &detail);
      const std::size_t pending_before_cancel = state.coordinator->pending_count();
      const std::size_t outstanding_before_cancel = state.coordinator->outstanding_count();
      const bool retired_all = cancel_ready_sources(state);
      if (force_causal_freeze_failure) {
        plan_->executable_factories.test_direct_continuation_control
            ->record_causal_freeze_failure_cleanup(pending_before_cancel,
                                                   outstanding_before_cancel,
                                                   state.coordinator->pending_count(),
                                                   state.coordinator->outstanding_count(),
                                                   state.direct_continuation_receipt.has_value(),
                                                   retired_all);
      }
      return CausalFrontierDisposition::kFailed;
    }
    if (!ingest_channel_candidate(exec, state, *candidate)) {
      static_cast<void>(cancel_ready_sources(state));
      return CausalFrontierDisposition::kFailed;
    }
    if (!channel_candidate_is_eligible(exec, state, *candidate)) {
      if (!retire_frozen_source(state, *candidate)) {
        append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
        static_cast<void>(cancel_ready_sources(state));
        return CausalFrontierDisposition::kFailed;
      }
      return CausalFrontierDisposition::kHandled;
    }
    state.candidates.push_back(candidate->candidate);
    state.frozen_candidates.push_back(std::move(*candidate));
    count_phase(RuntimePhaseCounter::kCandidatesFrozen);
    trace_phase(RuntimePhaseTracePoint::kCandidateFrozen,
                state.frozen_candidates.back().candidate.receiver_index,
                state.frozen_candidates.back().witness.source,
                state.frozen_candidates.back().witness.source_sequence);
    {
      const RuntimeCallbackContextScope callback_scope{RuntimeCallbackContext::kDispatcher};
      auto frozen = state.dispatcher->freeze(state.candidates);
      if (!frozen) {
        static_cast<void>(cancel_ready_sources(state));
        append_diagnostic("READY_POLICY_PLAN_INVALID");
        return CausalFrontierDisposition::kFailed;
      }
    }
    for (auto& frozen : state.frozen_candidates) {
      if (frozen.candidate.clock_health_disposition == EndpointClockHealthDisposition::kSkip &&
          !retire_frozen_source(state, frozen)) {
        static_cast<void>(cancel_ready_sources(state));
        append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
        return CausalFrontierDisposition::kFailed;
      }
    }
    if (state.dispatcher->active_count() == 0U) {
      if (state.direct_continuation_receipt || state.coordinator->pending_count() != 0U ||
          state.coordinator->outstanding_count() != 0U) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH",
                          exec.operators()[successor_receiver].vertex,
                          "partition_causal_frontier_suppression");
        static_cast<void>(cancel_ready_sources(state));
        return CausalFrontierDisposition::kFailed;
      }
      return CausalFrontierDisposition::kHandled;
    }
    if (state.frozen_candidates.size() != 1U ||
        state.frozen_candidates.front().candidate.receiver_index != successor_receiver ||
        state.frozen_candidates.front().witness.source != receipt.readiness_source) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH",
                        exec.operators()[successor_receiver].vertex,
                        "partition_causal_frontier");
      static_cast<void>(cancel_ready_sources(state));
      return CausalFrontierDisposition::kFailed;
    }
    count_phase(RuntimePhaseCounter::kPartitionCausalFrontierAccepted);
    return CausalFrontierDisposition::kAccepted;
  }

  // Freeze the exact lifecycle-member and Channel-dependent receiver closure, including saved
  // clock phases, before any restart fence mutates dispatcher or transport state.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] bool prepare_runtime_restart_scope(
      GraphExec& exec, ReadyRunState& state, std::span<const OperatorInstanceId> members,
      std::span<const VertexId> member_vertices) noexcept {
    if (members.empty() || members.size() != member_vertices.size()) {
      return false;
    }
    state.restart_receiver_scratch.clear();
    state.restart_member_vertex_scratch.clear();
    state.restart_vertex_scratch.clear();
    const auto append_affected = [&](std::size_t receiver, VertexId vertex, bool lifecycle_member) {
      if (receiver >= exec.operators().size() || !vertex ||
          std::find(state.restart_receiver_scratch.begin(),
                    state.restart_receiver_scratch.end(),
                    receiver) != state.restart_receiver_scratch.end()) {
        return false;
      }
      state.restart_receiver_scratch.push_back(receiver);
      state.restart_vertex_scratch.push_back(vertex);
      if (lifecycle_member) {
        state.restart_member_vertex_scratch.push_back(vertex);
      }
      return true;
    };
    for (const OperatorInstanceId member : members) {
      const auto instance = std::find_if(exec.operators().begin(),
                                         exec.operators().end(),
                                         [&](const RuntimeOperatorInstance& candidate) {
                                           return candidate.operator_id == member;
                                         });
      if (instance == exec.operators().end()) {
        return false;
      }
      const std::size_t receiver =
          static_cast<std::size_t>(std::distance(exec.operators().begin(), instance));
      if (!owns_receiver(state, receiver)) {
        continue;
      }
      if (!append_affected(receiver, instance->vertex, true)) {
        return false;
      }
    }
    for (const auto& subscriber : plan_->process_local_channel_subscribers) {
      const auto source =
          std::find_if(plan_->ports.begin(), plan_->ports.end(), [&](const PortPlanRecord& port) {
            return port.id == subscriber.source && port.direction == PortDirection::kOutput;
          });
      const auto input =
          std::find_if(plan_->ports.begin(), plan_->ports.end(), [&](const PortPlanRecord& port) {
            return port.id == subscriber.target && port.direction == PortDirection::kInput;
          });
      if (source == plan_->ports.end() || input == plan_->ports.end()) {
        return false;
      }
      if (std::find(member_vertices.begin(), member_vertices.end(), source->owner) ==
              member_vertices.end() ||
          std::find(state.restart_vertex_scratch.begin(),
                    state.restart_vertex_scratch.end(),
                    input->owner) != state.restart_vertex_scratch.end()) {
        continue;
      }
      const auto affected = std::find_if(exec.operators().begin(),
                                         exec.operators().end(),
                                         [&](const RuntimeOperatorInstance& candidate) {
                                           return candidate.vertex == input->owner;
                                         });
      if (affected == exec.operators().end()) {
        return false;
      }
      const std::size_t receiver =
          static_cast<std::size_t>(std::distance(exec.operators().begin(), affected));
      if (!owns_receiver(state, receiver)) {
        continue;
      }
      if (!append_affected(receiver, affected->vertex, false)) {
        return false;
      }
    }
    for (const std::size_t receiver : state.restart_receiver_scratch) {
      ClockTriggerPortBinding* const clock = state.clock_bindings[receiver];
      state.restart_phase_scratch[receiver].reset();
      if (clock == nullptr) {
        continue;
      }
      auto phase = clock->restart_phase();
      if (!phase) {
        return false;
      }
      state.restart_phase_scratch[receiver] = *phase;
    }
    return true;
  }

  [[nodiscard]] bool prepare_runtime_restart_aggregate(
      GraphExec& exec, std::span<ReadyRunState> states,
      std::span<const OperatorInstanceId> members) noexcept {
    if (states.empty() || members.empty()) {
      return false;
    }
    ReadyRunState& aggregate = states.front();
    aggregate.restart_aggregate_member_vertex_scratch.clear();
    aggregate.restart_aggregate_vertex_scratch.clear();
    for (const OperatorInstanceId member : members) {
      const auto instance = std::find_if(exec.operators().begin(),
                                         exec.operators().end(),
                                         [&](const RuntimeOperatorInstance& candidate) {
                                           return candidate.operator_id == member;
                                         });
      if (instance == exec.operators().end() ||
          std::find(aggregate.restart_aggregate_member_vertex_scratch.begin(),
                    aggregate.restart_aggregate_member_vertex_scratch.end(),
                    instance->vertex) != aggregate.restart_aggregate_member_vertex_scratch.end()) {
        return false;
      }
      aggregate.restart_aggregate_member_vertex_scratch.push_back(instance->vertex);
    }
    for (ReadyRunState& state : states) {
      if (!prepare_runtime_restart_scope(
              exec, state, members, aggregate.restart_aggregate_member_vertex_scratch)) {
        return false;
      }
      for (const VertexId vertex : state.restart_vertex_scratch) {
        if (std::find(aggregate.restart_aggregate_vertex_scratch.begin(),
                      aggregate.restart_aggregate_vertex_scratch.end(),
                      vertex) == aggregate.restart_aggregate_vertex_scratch.end()) {
          aggregate.restart_aggregate_vertex_scratch.push_back(vertex);
        }
      }
    }
    return !aggregate.restart_aggregate_vertex_scratch.empty();
  }

  [[nodiscard]] expected<bool, Error> restart_scope_uses_channels(
      const ReadyRunState& aggregate) const noexcept {
    return aggregate.channels == nullptr ? expected<bool, Error>{false}
                                         : aggregate.channels->restart_affects(
                                               aggregate.restart_aggregate_member_vertex_scratch,
                                               aggregate.restart_aggregate_vertex_scratch);
  }

  [[nodiscard]] bool fence_runtime_restart_state(ReadyRunState& state) noexcept {
    for (const std::size_t receiver : state.restart_receiver_scratch) {
      auto canceled = state.dispatcher->cancel_receiver(receiver);
      if (!canceled) {
        return false;
      }
      std::size_t retired = 0U;
      for (auto& candidate : state.frozen_candidates) {
        if (candidate.candidate.receiver_index != receiver || candidate.source_retired) {
          continue;
        }
        if (!retire_frozen_source(state, candidate)) {
          return false;
        }
        ++retired;
      }
      // Every active Dispatcher slot has exactly one coordinator witness. A mismatch means the
      // restart fence could not establish an exact-once retirement boundary.
      if (retired != *canceled) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool fence_runtime_restart(GraphExec& exec, std::span<ReadyRunState> states,
                                           std::span<const OperatorInstanceId> members,
                                           LifecycleStage from_stage,
                                           bool& channels_quiesced) noexcept {
    channels_quiesced = false;
    if (!prepare_runtime_restart_aggregate(exec, states, members)) {
      return false;
    }
    ReadyRunState& aggregate = states.front();
    if (aggregate.notifications != nullptr) {
      aggregate.notifications->begin_restart(aggregate.restart_aggregate_vertex_scratch);
    }
    const auto fail_restart_fence = [&]() noexcept {
      if (aggregate.notifications != nullptr) {
        aggregate.notifications->rollback_restart(aggregate.restart_aggregate_vertex_scratch);
      }
      return false;
    };
    const auto uses_channels = restart_scope_uses_channels(aggregate);
    if (!uses_channels) {
      return fail_restart_fence();
    }
    if (*uses_channels) {
      aggregate.channels->cancel();
      if (!aggregate.channels->join()) {
        return fail_restart_fence();
      }
      channels_quiesced = true;
    }
    for (ReadyRunState& state : states) {
      if (!fence_runtime_restart_state(state)) {
        return fail_restart_fence();
      }
    }
    if (tensor_resources_ != nullptr) {
      auto fenced = tensor_resources_->begin_restart(members, from_stage);
      if (!fenced) {
        return fail_restart_fence();
      }
    }
    return true;
  }

  // Retire the coordinator and binding prefixes for the complete prepared restart scope before any
  // source generation is advanced.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] bool retire_restart_witnesses(ReadyRunState& state) noexcept {
    for (const std::size_t receiver : state.restart_receiver_scratch) {
      auto canceled = state.coordinator->cancel_receiver(receiver, state.restart_event_scratch);
      if (!canceled) {
        return false;
      }
      for (std::size_t index = 0U; index < *canceled; ++index) {
        const RuntimeEvent& event = state.restart_event_scratch[index];
        PortBinding* const binding = physical_binding_for(state, event.source);
        if (binding != nullptr && !binding->retire(event.source_sequence)) {
          return false;
        }
      }
    }

    // Joined Channel ingress cannot append while the remaining binding-ring prefix is retired.
    // Resetting each affected source generation after this loop therefore cannot strand an
    // un-ingested old-generation witness.
    for (auto& binding : state.bindings) {
      const SourceId source = binding->source();
      if (!source) {
        continue;
      }
      const auto receiver = state.coordinator->source_receiver(source);
      if (!receiver || std::find(state.restart_receiver_scratch.begin(),
                                 state.restart_receiver_scratch.end(),
                                 *receiver) == state.restart_receiver_scratch.end()) {
        continue;
      }
      RuntimeEvent event;
      for (;;) {
        auto drained = binding->drain({&event, 1U});
        if (!drained || *drained > 1U) {
          return false;
        }
        if (*drained == 0U) {
          break;
        }
        if (!binding->retire(event.source_sequence)) {
          return false;
        }
      }
    }
    return true;
  }

  // Publish Channel, selector, readiness-source, clock, and predicate generations as one ordered
  // restart transaction after the exact affected scope has been fenced.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] bool publish_runtime_restart_state(GraphExec& exec, ReadyRunState& state) noexcept {
    for (auto& selector : state.input_selectors) {
      if (std::find(state.restart_vertex_scratch.begin(),
                    state.restart_vertex_scratch.end(),
                    selector.owner) != state.restart_vertex_scratch.end()) {
        if (!selector.selector.reset()) {
          return false;
        }
        selector.end_of_stream_received = false;
      }
    }
    for (auto& aligned : state.aligned_selectors) {
      if (std::find(state.restart_vertex_scratch.begin(),
                    state.restart_vertex_scratch.end(),
                    aligned.owner) != state.restart_vertex_scratch.end() &&
          !aligned.selector.reset()) {
        return false;
      }
    }
    // Channel restart retires the affected typed slots before the metadata
    // selectors reset. Drop the mirrored claim sets without calling back into
    // the now-new Channel generation.
    for (auto& selector : state.input_selectors) {
      if (std::find(state.restart_vertex_scratch.begin(),
                    state.restart_vertex_scratch.end(),
                    selector.owner) != state.restart_vertex_scratch.end()) {
        selector.channel_payload_claims.clear();
      }
    }

    for (const std::size_t receiver : state.restart_receiver_scratch) {
      if (!state.coordinator->publish_restart(receiver)) {
        return false;
      }
      std::size_t advanced_sources = 0U;
      for (auto& binding : state.bindings) {
        const SourceId source = binding->source();
        if (!source) {
          continue;
        }
        const auto source_receiver = state.coordinator->source_receiver(source);
        if (!source_receiver || source_receiver.value() != receiver) {
          continue;
        }
        const auto expected_generation = state.coordinator->source_generation(source);
        if (!expected_generation) {
          return false;
        }
        ClockTriggerPortBinding* const clock = state.clock_bindings[receiver];
        const auto& restart_phase = state.restart_phase_scratch[receiver];
        if (binding.get() == clock && !restart_phase) {
          return false;
        }
        const expected<std::uint64_t, Error> advanced_generation =
            binding.get() == clock ? expected<std::uint64_t, Error>{clock->publish_restart(
                                         restart_phase.value_or(TimePoint{}))}
                                   : binding->advance_source_generation();
        if (!advanced_generation || *advanced_generation != *expected_generation) {
          return false;
        }
        ++advanced_sources;
      }
      if (advanced_sources == 0U) {
        return false;
      }
      if (state.predicates[receiver]) {
        const bool lifecycle_member = std::find(state.restart_member_vertex_scratch.begin(),
                                                state.restart_member_vertex_scratch.end(),
                                                exec.operators()[receiver].vertex) !=
                                      state.restart_member_vertex_scratch.end();
        state.predicates[receiver]->reset(lifecycle_member ? PredicateResetReason::kOperatorRestart
                                                           : PredicateResetReason::kProducerEpoch,
                                          exec.operators()[receiver].tags->generation());
      }
    }
    return true;
  }

  [[nodiscard]] bool publish_runtime_restart(GraphExec& exec, std::span<ReadyRunState> states,
                                             OperatorRef target, bool channels_quiesced) noexcept {
    if (states.empty()) {
      return false;
    }
    ReadyRunState& aggregate = states.front();
    const auto target_instance =
        std::find_if(exec.operators().begin(), exec.operators().end(), [&](const auto& candidate) {
          return candidate.operator_id == target.id;
        });
    if (target_instance == exec.operators().end() ||
        std::find(aggregate.restart_aggregate_member_vertex_scratch.begin(),
                  aggregate.restart_aggregate_member_vertex_scratch.end(),
                  target_instance->vertex) ==
            aggregate.restart_aggregate_member_vertex_scratch.end()) {
      return false;
    }
    for (ReadyRunState& state : states) {
      if (!retire_restart_witnesses(state)) {
        return false;
      }
    }
    if (tensor_resources_ != nullptr) {
      auto preflight = tensor_resources_->preflight_restart_publication();
      if (!preflight) {
        return false;
      }
    }
    if (channels_quiesced) {
      // Channel commit is the irreversible transport-discard boundary. Every failure after this
      // point is fail-stopped by process_pending_restarts(); no runner may resume an activation
      // against a partially published generation, and rollback must not resurrect discarded
      // ingress.
      auto committed =
          aggregate.channels->commit_restart(aggregate.restart_aggregate_member_vertex_scratch,
                                             aggregate.restart_aggregate_vertex_scratch);
      // `restart_scope_uses_channels()` proved that this lifecycle incident owns at least one
      // frozen Channel binding. A missing affected Channel state is therefore an inconsistent
      // session, not an ignorable empty scope. Fail before publishing selector, readiness, or
      // producer generations so old samples can never be relabeled as post-restart state.
      if (!committed) {
        return false;
      }
    }
    for (ReadyRunState& state : states) {
      if (!publish_runtime_restart_state(exec, state)) {
        return false;
      }
    }
    if (channels_quiesced && !aggregate.channels->arm()) {
      return false;
    }
    // Direct Tensor pools are the final publication participant. Every earlier Channel,
    // selector, readiness, and coordinator transition can still fail; publishing a replacement
    // pool before them would consume the old pool generation even when the restart is ultimately
    // rejected. Tensor publication is preflighted by begin_restart() and cannot expose user work
    // until this safe point returns.
    if (tensor_resources_ != nullptr) {
      auto published = tensor_resources_->publish_restart();
      if (!published) {
        return false;
      }
    }
    if (aggregate.notifications != nullptr &&
        !aggregate.notifications->publish_restart(aggregate.restart_aggregate_vertex_scratch)) {
      return false;
    }
    return true;
  }

  [[nodiscard]] bool rollback_runtime_restart(std::span<ReadyRunState> states,
                                              bool channels_quiesced) noexcept {
    if (states.empty()) {
      return false;
    }
    ReadyRunState& aggregate = states.front();
    const bool tensor_rolled_back =
        tensor_resources_ == nullptr || tensor_resources_->rollback_restart().has_value();
    const bool channels_rearmed = !channels_quiesced || aggregate.channels->arm().has_value();
    if (aggregate.notifications != nullptr) {
      aggregate.notifications->rollback_restart(aggregate.restart_aggregate_vertex_scratch);
    }
    return tensor_rolled_back && channels_rearmed;
  }

  // Restart processing is one rollback-capable transaction across all runtime subsystems.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] bool process_pending_restarts(GraphExec& exec,
                                              std::span<ReadyRunState> states) noexcept {
    if (states.empty()) {
      return false;
    }
    const RuntimeCallbackContextScope control_scope{RuntimeCallbackContext::kDispatcher};
    struct RestartPublicationState {
      RunSessionState* session{};
      GraphExec* exec{};
      std::span<ReadyRunState> ready_states;
    } publication{.session = this, .exec = &exec, .ready_states = states};
    return lifecycle_control_->process_pending(
        [](void* state, const LifecycleRestartExecutionRequest& request) noexcept {
          auto& publication = *static_cast<RestartPublicationState*>(state);
          const auto transition_candidates = [&](LifecycleTransactionState lifecycle_state) {
            for (const auto& candidate : request.parameter_candidates) {
              if (!publication.exec->parameter_updates().transition_lifecycle_candidate(
                      candidate.owner,
                      candidate.generation,
                      request.transaction,
                      lifecycle_state)) {
                return false;
              }
            }
            return true;
          };
          if (!transition_candidates(LifecycleTransactionState::kQuiescing) ||
              !transition_candidates(LifecycleTransactionState::kRestarting)) {
            static_cast<void>(transition_candidates(LifecycleTransactionState::kFailedStopped));
            return LifecycleRestartExecutionOutcome{};
          }
          bool channels_quiesced = false;
          if (!publication.session->fence_runtime_restart(*publication.exec,
                                                          publication.ready_states,
                                                          request.members,
                                                          request.stage,
                                                          channels_quiesced)) {
            static_cast<void>(transition_candidates(LifecycleTransactionState::kFailedStopped));
            return LifecycleRestartExecutionOutcome{};
          }
          LifecycleRestartExecutionOutcome outcome = publication.exec->lifecycle_restart(request);
          // A Parameter-origin restart shares the Parameter terminal fence:
          // do not publish its candidate or source generation after stop.
          // Plain lifecycle work instead preserves the exact outcome already
          // produced by the executor, as required by the lifecycle contract.
          const bool parameter_stop_fence =
              !request.parameter_candidates.empty() &&
              publication.session->stop_requested_.load(std::memory_order_acquire);
          const bool restart_publication_failed =
              !parameter_stop_fence &&
              ((outcome.disposition == LifecycleRestartExecutionDisposition::kPublished &&
                !publication.session->publish_runtime_restart(*publication.exec,
                                                              publication.ready_states,
                                                              request.target,
                                                              channels_quiesced)) ||
               (outcome.disposition == LifecycleRestartExecutionDisposition::kRolledBack &&
                !publication.session->rollback_runtime_restart(publication.ready_states,
                                                               channels_quiesced)));
          if (parameter_stop_fence || restart_publication_failed) {
            outcome.disposition = LifecycleRestartExecutionDisposition::kFailedStopped;
          }
          const LifecycleTransactionState terminal_state = [&] {
            switch (outcome.disposition) {
              case LifecycleRestartExecutionDisposition::kPublished:
                return LifecycleTransactionState::kPublished;
              case LifecycleRestartExecutionDisposition::kRolledBack:
                return LifecycleTransactionState::kRolledBack;
              case LifecycleRestartExecutionDisposition::kFailedStopped:
                return LifecycleTransactionState::kFailedStopped;
            }
            return LifecycleTransactionState::kFailedStopped;
          }();
          if (!transition_candidates(terminal_state)) {
            outcome.disposition = LifecycleRestartExecutionDisposition::kFailedStopped;
            static_cast<void>(transition_candidates(LifecycleTransactionState::kFailedStopped));
          }
          return outcome;
        },
        &publication,
        stop_requested_);
  }

  [[nodiscard]] bool process_pending_restarts(GraphExec& exec, ReadyRunState& state) noexcept {
    return process_pending_restarts(exec, std::span<ReadyRunState>{&state, 1U});
  }

  enum class SelectionPreparation : std::uint8_t {
    kReady,
    kNoFire,
    kFailed,
  };

  struct PredicateComputeSelectionContext {
    RunSessionState* session{};
    ReadyRunState* ready{};
    const RuntimeOperatorInstance* instance{};
    const PredicateTriggerPlanRecord* predicate{};
    const FrozenRuntimeCandidate* candidate{};
    TimePoint* semantic_reference{};
  };

  void clear_input_selection_scratch(ReadyRunState& state, const FrozenRuntimeCandidate& candidate,
                                     TimePoint& semantic_reference) noexcept {
    state.selector_transactions.clear();
    state.aligned_transactions.clear();
    state.input_selection_scratch.clear();
    state.selected_sample_scratch.clear();
    state.channel_selection_scratch.clear();
    state.predicate_input_scratch.clear();
    semantic_reference = candidate.witness.reference_time;
  }

  void append_frozen_input_selection(
      ReadyRunState& state, ReadyRunState::InputSelectorSlot& slot,
      RuntimeInputSelectionState::Transaction& transaction) noexcept {
    const InputSelection selection = transaction.selection();
    const auto selected = transaction.selected();
    state.input_selection_scratch.push_back(selection);
    state.selected_sample_scratch.insert(
        state.selected_sample_scratch.end(), selected.begin(), selected.end());
    const InputSelectionPlanRecord& plan = slot.selector.plan();
    state.channel_selection_scratch.push_back(RuntimeChannelActivationSelection{
        .port = slot.input,
        .selection = selection,
        .samples = selected,
        .batch = plan.cardinality == InputSelectionCardinality::kSampleBatch,
        .destructive = plan.batch_selection_mode != BatchSelectionMode::kNonDestructiveSnapshot,
    });
  }

  [[nodiscard]] expected<void, Error> append_input_selection(
      ReadyRunState& state, ReadyRunState::InputSelectorSlot& slot, TimePoint reference_time,
      TimePoint& semantic_reference) noexcept {
    auto transaction =
        slot.selector.begin(reference_time, slot.arrival_watermark, slot.producer_epoch);
    if (!transaction) {
      return make_unexpected(std::move(transaction.error()));
    }
    const TimePoint selector_reference = transaction->semantic_reference_time();
    if (selector_reference.valid()) {
      semantic_reference = selector_reference;
    }
    state.selector_transactions.push_back(std::move(*transaction));
    append_frozen_input_selection(state, slot, state.selector_transactions.back());
    return {};
  }

  // Form aligned, bracketed-window, and ordinary selections in authoring order while retaining one
  // semantic reference and one transaction set for the activation.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] SelectionPreparation prepare_nonpredicate_input_selections(
      ReadyRunState& state, const RuntimeOperatorInstance& instance,
      const FrozenRuntimeCandidate& candidate, TimePoint& semantic_reference) noexcept {
    if (arrival_trigger_pin_matches(instance, candidate) &&
        std::none_of(
            state.input_selectors.begin(), state.input_selectors.end(), [&](const auto& slot) {
              return slot.owner == instance.vertex && slot.input == candidate.witness.port;
            })) {
      if (state.channels == nullptr || candidate.witness.source_sequence == 0U) {
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
        return SelectionPreparation::kFailed;
      }
      auto sample = state.channels->temporal_sample(candidate.witness.port,
                                                    candidate.witness.source_sequence);
      if (!sample) {
        append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
        return SelectionPreparation::kFailed;
      }
      state.input_selection_scratch.push_back(
          InputSelection{InputState::kSelected, SelectionReason::kNone});
      state.selected_sample_scratch.push_back(*sample);
      state.channel_selection_scratch.push_back(RuntimeChannelActivationSelection{
          .port = candidate.witness.port,
          .selection = state.input_selection_scratch.back(),
          .samples =
              std::span<const TemporalSampleRecord>{&state.selected_sample_scratch.back(), 1U},
          .batch = false,
          .destructive = true,
      });
    }
    const auto aligned = std::find_if(
        state.aligned_selectors.begin(), state.aligned_selectors.end(), [&](const auto& slot) {
          return slot.owner == instance.vertex;
        });
    if (aligned != state.aligned_selectors.end()) {
      AlignedCandidateDisposition disposition{AlignedCandidateDisposition::kPending};
      while (true) {
        auto transaction = aligned->selector.begin();
        if (!transaction) {
          return transaction.error().code == ErrorCode::kNotReady ? SelectionPreparation::kNoFire
                                                                  : SelectionPreparation::kFailed;
        }
        if (transaction->disposition() != AlignedCandidateDisposition::kRetireMisaligned) {
          disposition = transaction->disposition();
          state.aligned_transactions.push_back(std::move(*transaction));
          break;
        }
        // A window miss deterministically retires every minimum head. Continue on the same
        // readiness candidate so a backlog of incompatible heads cannot stall until unrelated
        // future input arrives.
        if (!transaction->finish(false)) {
          return SelectionPreparation::kFailed;
        }
        if (!synchronize_temporal_payload_claims(state, instance.vertex)) {
          return SelectionPreparation::kFailed;
        }
        continue;
      }
      auto& frozen = state.aligned_transactions.back();
      if (frozen.reference_time().valid()) {
        semantic_reference = frozen.reference_time();
      }
      if (disposition == AlignedCandidateDisposition::kNoFire ||
          disposition == AlignedCandidateDisposition::kPending) {
        return SelectionPreparation::kNoFire;
      }

      const auto inputs = aligned->selector.member_inputs();
      const auto selections = frozen.selections();
      const auto selected = frozen.selected();
      if (inputs.size() != selections.size()) {
        return SelectionPreparation::kFailed;
      }
      std::size_t selected_index = 0U;
      for (std::size_t index = 0U; index < inputs.size(); ++index) {
        const InputSelection selection = selections[index];
        const bool has_sample =
            selection.state == InputState::kSelected || selection.state == InputState::kReused;
        if (has_sample && selected_index >= selected.size()) {
          return SelectionPreparation::kFailed;
        }
        const std::span<const TemporalSampleRecord> member_samples =
            has_sample ? selected.subspan(selected_index++, 1U)
                       : std::span<const TemporalSampleRecord>{};
        state.input_selection_scratch.push_back(selection);
        state.selected_sample_scratch.insert(
            state.selected_sample_scratch.end(), member_samples.begin(), member_samples.end());
        state.channel_selection_scratch.push_back(RuntimeChannelActivationSelection{
            .port = inputs[index],
            .selection = selection,
            .samples = member_samples,
            .batch = false,
            .destructive = selection.state == InputState::kSelected,
        });
      }
      if (selected_index != selected.size()) {
        return SelectionPreparation::kFailed;
      }
    }
    // OnWindow closes at a successor or timeout witness that is later than its semantic anchor.
    // Freeze the bracket first to recover that anchor, then evaluate every peer selector at the
    // anchor while retaining the close witness as activation/TagTime. Materialize selections in
    // authored input order even when the bracketed input is not declared first.
    auto bracketed = state.input_selectors.end();
    for (auto slot = state.input_selectors.begin(); slot != state.input_selectors.end(); ++slot) {
      if (slot->owner != instance.vertex || slot->aligned_member ||
          slot->selector.plan().selector != InputSelectorKind::kBracketedWindow) {
        continue;
      }
      if (bracketed != state.input_selectors.end()) {
        append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
        return SelectionPreparation::kFailed;
      }
      bracketed = slot;
    }

    std::optional<std::size_t> bracketed_transaction_index;
    if (bracketed != state.input_selectors.end()) {
      auto transaction = bracketed->selector.begin(candidate.witness.reference_time,
                                                   bracketed->arrival_watermark,
                                                   bracketed->producer_epoch);
      if (!transaction) {
        if (transaction.error().code == ErrorCode::kNotReady) {
          return SelectionPreparation::kNoFire;
        }
        append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
        return SelectionPreparation::kFailed;
      }
      const TimePoint anchor = transaction->semantic_reference_time();
      if (!anchor.valid() || anchor.clock_id != candidate.witness.reference_time.clock_id ||
          anchor.timestamp_ns > candidate.witness.reference_time.timestamp_ns) {
        append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
        return SelectionPreparation::kFailed;
      }
      semantic_reference = anchor;
      state.selector_transactions.push_back(std::move(*transaction));
      bracketed_transaction_index = state.selector_transactions.size() - 1U;
    }

    for (auto& slot : state.input_selectors) {
      if (slot.owner != instance.vertex || slot.aligned_member) {
        continue;
      }
      if (&slot == (bracketed == state.input_selectors.end() ? nullptr : &*bracketed)) {
        const std::size_t transaction_index =
            bracketed_transaction_index.value_or(state.selector_transactions.size());
        if (transaction_index >= state.selector_transactions.size()) {
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
          return SelectionPreparation::kFailed;
        }
        append_frozen_input_selection(state, slot, state.selector_transactions[transaction_index]);
        continue;
      }
      auto appended = append_input_selection(state,
                                             slot,
                                             bracketed_transaction_index.has_value()
                                                 ? semantic_reference
                                                 : candidate.witness.reference_time,
                                             semantic_reference);
      if (!appended) {
        if (appended.error().code == ErrorCode::kNotReady) {
          return SelectionPreparation::kNoFire;
        }
        append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
        return SelectionPreparation::kFailed;
      }
    }

    return aggregate_missing_inputs(state.input_selection_scratch) ==
                   MissingAggregateDisposition::kNoFire
               ? SelectionPreparation::kNoFire
               : SelectionPreparation::kReady;
  }

  [[nodiscard]] expected<void, Error> prepare_predicate_input_selections(
      ReadyRunState& state, const RuntimeOperatorInstance& instance,
      const PredicateTriggerPlanRecord& predicate, const FrozenRuntimeCandidate& candidate,
      TimePoint& semantic_reference) noexcept {
    std::uint32_t expected_ordinal = 0U;
    for (auto& slot : state.input_selectors) {
      if (slot.owner != instance.vertex || !slot.predicate_input ||
          slot.predicate_id != predicate.id) {
        continue;
      }
      if (slot.predicate_ordinal != expected_ordinal) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      auto appended =
          append_input_selection(state, slot, candidate.witness.reference_time, semantic_reference);
      if (!appended) {
        return appended;
      }
      ++expected_ordinal;
    }
    return {};
  }

  [[nodiscard]] expected<std::span<const InputSelection>, Error>
  prepare_predicate_compute_selections(PredicateComputeSelectionContext& context,
                                       TimePoint reference_time) noexcept {
    if (context.session != this || context.ready == nullptr || context.instance == nullptr ||
        context.predicate == nullptr || context.candidate == nullptr ||
        context.semantic_reference == nullptr ||
        reference_time != context.candidate->witness.reference_time) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    ReadyRunState& state = *context.ready;
    const std::size_t selection_begin = state.input_selection_scratch.size();
    for (auto& slot : state.input_selectors) {
      if (slot.owner != context.instance->vertex ||
          (slot.predicate_input && slot.predicate_id == context.predicate->id)) {
        continue;
      }
      auto appended = append_input_selection(
          state, slot, context.candidate->witness.reference_time, *context.semantic_reference);
      if (!appended) {
        return make_unexpected(std::move(appended.error()));
      }
    }
    // Compute-only selections are not visible to the predicate callable. Keep their Channel
    // payload claims unprepared until the evaluator admits the callback so a successful no-fire
    // can commit its predicate inputs while rolling this suffix back independently.
    return std::span<const InputSelection>{state.input_selection_scratch}.subspan(selection_begin);
  }

  [[nodiscard]] static expected<std::span<const InputSelection>, Error>
  select_predicate_compute_inputs(void* opaque, TimePoint reference_time) noexcept {
    if (opaque == nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto& context = *static_cast<PredicateComputeSelectionContext*>(opaque);
    return context.session->prepare_predicate_compute_selections(context, reference_time);
  }

  [[nodiscard]] bool finish_input_selection_prefix(ReadyRunState& state,
                                                   std::size_t accepted_selector_count,
                                                   bool aligned_accepted) noexcept {
    if (accepted_selector_count > state.selector_transactions.size()) {
      return false;
    }
    bool finished = true;
    for (std::size_t index = 0U; index < state.selector_transactions.size(); ++index) {
      if (!state.selector_transactions[index].finish(index < accepted_selector_count)) {
        finished = false;
      }
    }
    state.selector_transactions.clear();
    for (auto& transaction : state.aligned_transactions) {
      if (!transaction.finish(aligned_accepted)) {
        finished = false;
      }
    }
    state.aligned_transactions.clear();
    if (!finished) {
      // A failed selector commit is terminal for this dispatch. Retaining the previous mirrored
      // claim superset is fail-closed; synchronizing here could otherwise retire payloads after a
      // partially committed transaction set has already reported failure.
      return false;
    }
    return synchronize_temporal_payload_claims(state);
  }

  [[nodiscard]] bool finish_input_selections(ReadyRunState& state, bool accepted) noexcept {
    return finish_input_selection_prefix(
        state, accepted ? state.selector_transactions.size() : 0U, accepted);
  }

  [[nodiscard]] bool retire_activation_transaction(
      GraphExec& exec, ReadyRunState& state, RuntimeOperatorInstance& instance,
      const DispatcherSelection& selection, FrozenRuntimeCandidate& frozen,
      std::uint64_t activation_generation, bool channel_activation_prepared,
      ActivationParameterBinding& parameter_binding, bool callback_succeeded,
      bool completed_selection) noexcept {
    const bool retired_source =
        completed_selection && retire_frozen_source(state, frozen).has_value();
    const bool finished_selectors = finish_input_selections(state, callback_succeeded);
    const bool retired_channel =
        !channel_activation_prepared ||
        state.channels
            ->retire_activation(instance.vertex, activation_generation, callback_succeeded)
            .has_value();
    const bool retired_activation =
        state.activation_selections[selection.receiver_index].retire(selection.tag).has_value();
    parameter_binding.release();
    if (callback_succeeded && instance.runtime_parameter_updates_enabled) {
      static_cast<void>(exec.parameter_updates().drain_compute_ingress());
      if (!stop_requested_.load(std::memory_order_acquire)) {
        publish_pending_parameter_updates(exec.parameter_updates(),
                                          OperatorRef{
                                              .id = instance.operator_id,
                                              .plan_generation = plan_->plan_generation,
                                              .run_generation = run_generation_,
                                          });
      }
    }
    if (!completed_selection || !retired_source || !finished_selectors || !retired_channel ||
        !retired_activation) {
      return false;
    }
    if (!callback_succeeded) {
      return true;
    }
    trace_phase(RuntimePhaseTracePoint::kActivationRetirementCompleted,
                selection.receiver_index,
                frozen.witness.source,
                frozen.witness.source_sequence);
    if (state.channels != nullptr) {
      auto cut = state.channels->flush_process_local_activation_cut();
      if (!cut) {
        const RuntimeDiagnosticDetail detail{
            .present = true,
            .error = cut.error(),
            .operator_id = instance.operator_id,
            .source = frozen.witness.source,
            .source_sequence = frozen.witness.source_sequence,
            .tag = frozen.candidate.tag,
            .tag_time = frozen.candidate.ready.tag_time,
            .reference_time = frozen.witness.reference_time,
            .state_generation = frozen.candidate.tag.generation,
        };
        append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE",
                          instance.vertex,
                          "post_compute_channel_cut",
                          &detail);
        return false;
      }
    }
    trace_phase(RuntimePhaseTracePoint::kActivationCutCompleted,
                selection.receiver_index,
                frozen.witness.source,
                frozen.witness.source_sequence);
    count_phase(RuntimePhaseCounter::kActivationsCompleted);
    return true;
  }

  // Execute the exact callback transaction admitted by ClosedIslandActivationPlanRecord.
  //
  // This is deliberately narrower than dispatch_ready(): it accepts no predicate, deadline,
  // mutable Parameter, lifecycle, or Tensor state. Both the ordinary one-dispatch-per-callback
  // control and the two-callback composite call this function, so changing lane-task geometry
  // cannot silently change selection, Channel preparation, callback context, or retirement.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  void run_closed_island_activation_transaction(GraphExec& exec, ReadyRunState& state,
                                                const RuntimeDirectCompositeBinding& binding,
                                                const DispatcherSelection& selection,
                                                FrozenRuntimeCandidate& frozen,
                                                bool arm_direct_continuation,
                                                bool successor_boundary,
                                                ClosedIslandActivationResult& result) noexcept {
    result = {};
    if (selection.receiver_index >= exec.operators().size() ||
        (selection.receiver_index != binding.producer_receiver &&
         selection.receiver_index != binding.successor_receiver) ||
        selection.receiver_index >= state.parameter_policies.size() ||
        selection.receiver_index >= state.predicates.size() ||
        selection.receiver_index >= state.deadline_bindings.size() || state.channels == nullptr) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
      static_cast<void>(cancel_ready_sources(state));
      result.error = Error{ErrorCode::kProtocolError, "RUNTIME_OPERATOR_REPLAY_MISMATCH"};
      return;
    }
    // Closed-island derivation rejects every plan with lifecycle transaction capacity. Reaching
    // an enabled controller here therefore means the frozen plan/runtime structure disagrees; it
    // is not a pending-restart observation.
    if (lifecycle_control_->enabled()) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH");
      static_cast<void>(cancel_ready_sources(state));
      result.error = Error{ErrorCode::kProtocolError, "RUNTIME_OPERATOR_REPLAY_MISMATCH"};
      return;
    }

    RuntimeOperatorInstance& instance = exec.operators()[selection.receiver_index];
    const bool source_member = selection.receiver_index == binding.producer_receiver;
    if (instance.execution_lane != binding.execution_lane ||
        instance.vertex != (source_member ? binding.producer : binding.successor) ||
        instance.runtime_parameter_updates_enabled ||
        instance.lifecycle_parameter_updates_enabled ||
        state.predicates[selection.receiver_index].has_value() ||
        state.deadline_bindings[selection.receiver_index].has_value() ||
        state.parameter_policies[selection.receiver_index].deadline_term_index.has_value() ||
        std::any_of(state.parameter_policies[selection.receiver_index].selector_indices.begin(),
                    state.parameter_policies[selection.receiver_index].selector_indices.end(),
                    [&](std::uint32_t index) {
                      return index >= state.input_selectors.size() ||
                             state.input_selectors[index].derived_term_index.has_value();
                    })) {
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
      static_cast<void>(cancel_ready_sources(state));
      result.error = Error{ErrorCode::kProtocolError, "RUNTIME_OPERATOR_REPLAY_MISMATCH"};
      return;
    }

    ActivationParameterBinding parameters{exec.parameter_updates(),
                                          instance.static_parameter_header};
    TimePoint semantic_reference{};
    clear_input_selection_scratch(state, frozen, semantic_reference);
    const std::uint64_t activation_generation =
        state.activation_selections[selection.receiver_index].generation();
    const SelectionPreparation prepared =
        prepare_nonpredicate_input_selections(state, instance, frozen, semantic_reference);
    if (prepared == SelectionPreparation::kFailed) {
      static_cast<void>(finish_input_selections(state, false));
      append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
      static_cast<void>(cancel_ready_sources(state));
      result.error = Error{ErrorCode::kFailure, "RUNTIME_PORT_BINDING_UNAVAILABLE"};
      return;
    }

    const bool channel_activation_required =
        selection.receiver_index < state.channel_activation_owners.size() &&
        state.channel_activation_owners[selection.receiver_index] != 0U;
    bool channel_activation_prepared = false;
    if (prepared == SelectionPreparation::kReady && channel_activation_required) {
      auto channel_prepared = state.channels->prepare_activation(
          instance.vertex, activation_generation, state.channel_selection_scratch);
      if (!channel_prepared) {
        static_cast<void>(finish_input_selections(state, false));
        append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
        static_cast<void>(cancel_ready_sources(state));
        result.error = std::move(channel_prepared.error());
        return;
      }
      auto tensor_access = state.channels->prepare_tensor_access(
          instance.vertex, activation_generation, instance.compute_stream);
      if (!tensor_access) {
        static_cast<void>(finish_input_selections(state, false));
        static_cast<void>(
            state.channels->retire_activation(instance.vertex, activation_generation, false));
        append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
        static_cast<void>(cancel_ready_sources(state));
        result.error = std::move(tensor_access.error());
        return;
      }
      channel_activation_prepared = true;
    }

    if (prepared == SelectionPreparation::kNoFire) {
      const bool selectors_aborted = finish_input_selections(state, false);
      const bool channel_retired =
          !channel_activation_prepared ||
          state.channels->retire_activation(instance.vertex, activation_generation, false)
              .has_value();
      const bool dispatcher_completed = state.dispatcher->complete(selection).has_value();
      const bool source_retired =
          dispatcher_completed && retire_frozen_source(state, frozen).has_value();
      parameters.release();
      if (!selectors_aborted || !channel_retired || !dispatcher_completed || !source_retired) {
        append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE", instance.vertex);
        static_cast<void>(cancel_ready_sources(state));
        result.error = Error{ErrorCode::kFailure, "RUNTIME_READINESS_BINDING_UNAVAILABLE"};
        return;
      }
      result.activation_retired = true;
      result.disposition = ClosedIslandActivationDisposition::kNoFire;
      return;
    }

    auto activation = state.activation_selections[selection.receiver_index].begin(
        frozen, state.input_selection_scratch, state.selected_sample_scratch, semantic_reference);
    if (!activation) {
      static_cast<void>(finish_input_selections(state, false));
      if (channel_activation_prepared) {
        static_cast<void>(
            state.channels->retire_activation(instance.vertex, activation_generation, false));
      }
      append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
      static_cast<void>(cancel_ready_sources(state));
      result.error = std::move(activation.error());
      return;
    }

    const auto retire_without_callback = [&]() noexcept {
      const bool dispatcher_completed = state.dispatcher->complete(selection).has_value();
      const bool source_retired =
          dispatcher_completed && retire_frozen_source(state, frozen).has_value();
      const bool selectors_aborted = finish_input_selections(state, false);
      const bool channel_retired =
          !channel_activation_prepared ||
          state.channels->retire_activation(instance.vertex, activation_generation, false)
              .has_value();
      const bool activation_retired =
          state.activation_selections[selection.receiver_index].retire(selection.tag).has_value();
      parameters.release();
      return dispatcher_completed && source_retired && selectors_aborted && channel_retired &&
             activation_retired;
    };

    if (successor_boundary) {
      const auto& test_control = plan_->executable_factories.test_closed_island_control;
      if (test_control != nullptr) {
        test_control->successor_boundary();
      }
    }
    if (stop_requested_.load(std::memory_order_acquire) || lifecycle_control_->enabled()) {
      const bool stop_observed = stop_requested_.load(std::memory_order_acquire);
      if (!retire_without_callback()) {
        append_diagnostic("RUNTIME_LIFECYCLE_CALLBACK_FAILED", instance.vertex);
        static_cast<void>(cancel_ready_sources(state));
        result.error = Error{ErrorCode::kFailure, "RUNTIME_LIFECYCLE_CALLBACK_FAILED"};
        return;
      }
      result.activation_retired = true;
      result.disposition = stop_observed ? ClosedIslandActivationDisposition::kStopped
                                         : ClosedIslandActivationDisposition::kRestarted;
      return;
    }

    const OperatorRef self{
        .id = instance.operator_id,
        .plan_generation = plan_->plan_generation,
        .run_generation = run_generation_,
    };
    ExecutionContextState context;
    context.tag = selection.tag;
    context.tag_time = selection.ready.tag_time;
    context.activation_time = frozen.witness.reference_time;
    context.clock = exec.default_clock();
    context.cuda_stream = instance.compute_stream;
    context.activation = ActivationStatus{ActivationReason::kNominal, OnViolation::kEmitWitness};
    context.cancellation_source = &stop_requested_;
    context.pinned_params = parameters.header();
    context.self = self;
    ComputeParameterIngress compute_parameter_ingress{
        .coordinator = &exec.parameter_updates(),
        .self = self,
    };
    context.param_control_state = &compute_parameter_ingress;
    context.submit_param_update = &submit_compute_parameter_update;

    ComputeLaneInvocation invocation{
        .instance = &instance,
        .context = &context,
        .dispatcher = state.dispatcher.get(),
        .selection = &selection,
        .parameter_updates = &exec.parameter_updates(),
        .self = self,
        .source_sequence = frozen.witness.source_sequence,
        .cuda_device = instance.compute_device,
        .phase_counters = phase_counters_,
    };
    if (arm_direct_continuation &&
        !plan_->executable_factories.test_disable_direct_continuation_elision) {
      const auto target =
          state.channels->arm_direct_continuation(instance.vertex, instance.execution_lane);
      if (target) {
        const auto source_credit =
            state.coordinator->source_credit_remaining(target->readiness_source);
        if (!source_credit) {
          static_cast<void>(retire_without_callback());
          append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE", instance.vertex);
          static_cast<void>(cancel_ready_sources(state));
          result.error = std::move(source_credit.error());
          return;
        }
        // The causal receipt may replace a readiness poll only while the successor owns its full
        // source-credit budget. Any outstanding credit could represent an earlier publication,
        // so relaxing this equality would destroy the single-pending-event authority required by
        // freeze_causal_next().
        if (*source_credit == target->readiness_capacity) {
          auto clock = exec.resolve_clock(target->reference_clock);
          if (!clock) {
            static_cast<void>(retire_without_callback());
            append_diagnostic("RUNTIME_CLOCK_BINDING_UNAVAILABLE", instance.vertex);
            static_cast<void>(cancel_ready_sources(state));
            result.error = std::move(clock.error());
            return;
          }
          invocation.channels = state.channels.get();
          invocation.channel_owner = instance.vertex;
          invocation.execution_lane = instance.execution_lane;
          invocation.direct_continuation_clock = std::move(*clock);
          invocation.direct_continuation_enabled = true;
        }
      }
    }

    result.callback_identity = RuntimeCallbackIdentity{
        .context = RuntimeCallbackContext::kCompute,
        .operator_id = instance.operator_id,
        .source = frozen.witness.source,
        .source_sequence = frozen.witness.source_sequence,
        .tag = frozen.candidate.tag,
        .tag_time = frozen.candidate.ready.tag_time,
        .reference_time = frozen.witness.reference_time,
        .state_generation = frozen.candidate.tag.generation,
    };
    count_phase(RuntimePhaseCounter::kComputeDispatches);
    trace_phase(RuntimePhaseTracePoint::kComputeDispatchStarted,
                selection.receiver_index,
                frozen.witness.source,
                frozen.witness.source_sequence);
    expected<void, Error> computed;
    {
      const RuntimeCallbackContextScope callback_scope{result.callback_identity};
      computed = compute_lane_invoke_(&invocation);
    }
    trace_phase(RuntimePhaseTracePoint::kComputeDispatchReturned,
                selection.receiver_index,
                frozen.witness.source,
                frozen.witness.source_sequence);
    if (invocation.admission != ComputeAdmissionDisposition::kAdmitted) {
      static_cast<void>(retire_without_callback());
      const std::string_view token =
          invocation.admission == ComputeAdmissionDisposition::kCudaDeviceUnavailable
              ? std::string_view{"RUNTIME_COMPUTE_CUDA_DEVICE_UNAVAILABLE"}
              : std::string_view{"RUNTIME_OPERATOR_REPLAY_MISMATCH"};
      append_diagnostic(token, instance.vertex);
      static_cast<void>(cancel_ready_sources(state));
      result.error = Error{ErrorCode::kFailure, std::string{token}};
      return;
    }

    result.callback_attempted = true;
    result.callback_completed = computed.has_value();
    result.error = computed ? Error{ErrorCode::kSuccess} : computed.error();
    result.direct_continuation_receipt = std::move(invocation.direct_continuation_receipt);
    result.direct_continuation_clock_failed = invocation.direct_continuation_clock_failed;
    if (!retire_activation_transaction(exec,
                                       state,
                                       instance,
                                       selection,
                                       frozen,
                                       activation_generation,
                                       channel_activation_prepared,
                                       parameters,
                                       computed.has_value(),
                                       invocation.completed_selection)) {
      append_diagnostic("RUNTIME_LIFECYCLE_CALLBACK_FAILED", instance.vertex);
      static_cast<void>(cancel_ready_sources(state));
      result.error = Error{ErrorCode::kFailure, "RUNTIME_LIFECYCLE_CALLBACK_FAILED"};
      return;
    }
    result.activation_retired = true;
    result.activation_cut_completed = computed.has_value();
    if (!computed) {
      const RuntimeDiagnosticDetail detail{
          .present = true,
          .error = computed.error(),
          .operator_id = instance.operator_id,
          .source = frozen.witness.source,
          .source_sequence = frozen.witness.source_sequence,
          .tag = frozen.candidate.tag,
          .tag_time = frozen.candidate.ready.tag_time,
          .reference_time = frozen.witness.reference_time,
          .state_generation = frozen.candidate.tag.generation,
      };
      append_diagnostic(
          "RUNTIME_OPERATOR_CALLBACK_FAILED", instance.vertex, "operator_compute", &detail);
      static_cast<void>(cancel_ready_sources(state));
      return;
    }
    if (invocation.direct_continuation_error.code != ErrorCode::kSuccess) {
      const RuntimeDiagnosticDetail detail{
          .present = true,
          .error = invocation.direct_continuation_error,
          .operator_id = instance.operator_id,
          .source = frozen.witness.source,
          .source_sequence = frozen.witness.source_sequence,
          .tag = frozen.candidate.tag,
          .tag_time = frozen.candidate.ready.tag_time,
          .reference_time = frozen.witness.reference_time,
          .state_generation = frozen.candidate.tag.generation,
      };
      append_diagnostic(invocation.direct_continuation_clock_failed
                            ? std::string_view{"RUNTIME_CLOCK_READ_FAILED"}
                            : std::string_view{"RUNTIME_READINESS_BINDING_UNAVAILABLE"},
                        instance.vertex,
                        "direct_continuation",
                        &detail);
      static_cast<void>(cancel_ready_sources(state));
      result.error = std::move(invocation.direct_continuation_error);
      return;
    }
    result.disposition = ClosedIslandActivationDisposition::kCompleted;
  }

  // Direct-composite execution preserves activation, CUDA, and fallback state in one loop.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  void run_direct_composite_lane(GraphExec& exec, ReadyRunState& state,
                                 const RuntimeDirectCompositeBinding& binding,
                                 const DispatcherSelection& source_selection,
                                 FrozenRuntimeCandidate& source_candidate,
                                 RuntimeDirectCompositeCompletionReport& report) noexcept {
    count_phase(RuntimePhaseCounter::kDirectCompositeAttempts);
    trace_phase(RuntimePhaseTracePoint::kDirectCompositeStarted,
                source_selection.receiver_index,
                source_candidate.witness.source,
                source_candidate.witness.source_sequence);
    const auto& test_control = plan_->executable_factories.test_closed_island_control;
    if (test_control != nullptr) {
      test_control->transaction_started();
    }
    struct TransactionHookGuard {
      const std::shared_ptr<RuntimeClosedIslandTestControl>* control{};
      const RuntimeDirectCompositeCompletionReport* report{};
      ~TransactionHookGuard() {
        if (control != nullptr && *control != nullptr && report != nullptr) {
          (*control)->transaction_finished(*report);
        }
      }
    } const transaction_hook{.control = &test_control, .report = &report};

    const auto account = [&](std::size_t slot,
                             const ClosedIslandActivationResult& transaction) noexcept {
      if (transaction.callback_attempted) {
        report.callback_identities[slot] = transaction.callback_identity;
        report.callback_errors[slot] = transaction.error.code;
        ++report.callbacks_attempted;
        ++report.phase_deltas[static_cast<std::size_t>(
            RuntimeDirectCompositePhaseDelta::kCallbacksAttempted)];
        count_phase(RuntimePhaseCounter::kDirectCompositeCallbacksAttempted);
      }
      if (transaction.callback_completed) {
        ++report.callbacks_completed;
        ++report.phase_deltas[static_cast<std::size_t>(
            RuntimeDirectCompositePhaseDelta::kCallbacksCompleted)];
        count_phase(RuntimePhaseCounter::kDirectCompositeCallbacksCompleted);
      }
      if (transaction.activation_retired) {
        ++report.phase_deltas[static_cast<std::size_t>(
            RuntimeDirectCompositePhaseDelta::kActivationsRetired)];
        if (slot == 0U) {
          report.source_retired = true;
        } else {
          report.successor_retired = true;
        }
      }
      if (transaction.activation_cut_completed) {
        ++report.phase_deltas[static_cast<std::size_t>(
            RuntimeDirectCompositePhaseDelta::kActivationCuts)];
      }
      if (report.first_error.code == ErrorCode::kSuccess &&
          transaction.error.code != ErrorCode::kSuccess) {
        report.first_error = transaction.error;
      }
      report.stop_observed = report.stop_observed ||
                             transaction.disposition == ClosedIslandActivationDisposition::kStopped;
      report.restart_observed =
          report.restart_observed ||
          transaction.disposition == ClosedIslandActivationDisposition::kRestarted;
    };
    const auto fail = [&]() noexcept {
      report.continuation = RuntimeDirectCompositeContinuationDisposition::kFailed;
      report.resume = RuntimeDirectCompositeResumeDisposition::kTerminalFailure;
      count_phase(RuntimePhaseCounter::kDirectCompositeFailures);
    };
    const auto fallback = [&]() noexcept {
      report.continuation = RuntimeDirectCompositeContinuationDisposition::kRejected;
      report.resume = RuntimeDirectCompositeResumeDisposition::kControllerResumesScan;
      count_phase(RuntimePhaseCounter::kDirectCompositeFallbacks);
    };

    ClosedIslandActivationResult source_result;
    run_closed_island_activation_transaction(
        exec, state, binding, source_selection, source_candidate, true, false, source_result);
    account(0U, source_result);
    if (source_result.disposition == ClosedIslandActivationDisposition::kFailed) {
      fail();
      return;
    }
    if (source_result.disposition != ClosedIslandActivationDisposition::kCompleted ||
        !source_result.direct_continuation_receipt) {
      fallback();
      return;
    }

    RuntimeChannelDirectContinuationReceipt receipt = *source_result.direct_continuation_receipt;
    report.phase_deltas[static_cast<std::size_t>(
        RuntimeDirectCompositePhaseDelta::kContinuationReceipts)] = 1U;
    if (test_control != nullptr) {
      test_control->before_successor();
      if (test_control->corrupt_generation()) {
        receipt.source_generation =
            receipt.source_generation == std::numeric_limits<std::uint64_t>::max()
                ? 0U
                : receipt.source_generation + 1U;
      }
    }
    if (stop_requested_.load(std::memory_order_acquire)) {
      report.stop_observed = true;
      fallback();
      return;
    }
    if (state.channels == nullptr ||
        !state.channels->validate_direct_continuation_receipt(receipt) ||
        receipt.destination != binding.successor ||
        receipt.destination_input != binding.successor_input ||
        receipt.source_output != binding.source_output ||
        receipt.readiness_source != binding.readiness_source ||
        state.dispatcher->active_count() != 0U) {
      fallback();
      return;
    }

    // Re-run the ordinary whole-round ingress and global ordering operation while this lane owns
    // the controller transaction. The compiler admits only a complete two-Operator process group,
    // so every poll target is this lane and no nested lane dispatch is possible. A source
    // re-notification that races the publication therefore competes with the successor under the
    // same coordinator ordering as the ordinary path instead of being bypassed by the receipt.
    state.direct_continuation_receipt = receipt;
    if (!freeze_ready_candidates(exec, state, binding.execution_lane)) {
      report.first_error = Error{ErrorCode::kFailure, "RUNTIME_READINESS_BINDING_UNAVAILABLE"};
      fail();
      return;
    }
    if (state.frozen_candidates.empty()) {
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE", binding.successor);
      static_cast<void>(cancel_ready_sources(state));
      report.first_error = Error{ErrorCode::kProtocolError};
      fail();
      return;
    }
    FrozenRuntimeCandidate& successor_frozen = state.frozen_candidates.front();
    if (successor_frozen.candidate.receiver_index != binding.successor_receiver ||
        successor_frozen.witness.source != binding.readiness_source ||
        successor_frozen.witness.port != binding.successor_input) {
      fallback();
      return;
    }
    auto successor_selection = state.dispatcher->select();
    if (!successor_selection || successor_selection->receiver_index != binding.successor_receiver) {
      append_diagnostic("READY_POLICY_PLAN_INVALID", binding.successor);
      static_cast<void>(cancel_ready_sources(state));
      report.first_error =
          successor_selection ? Error{ErrorCode::kProtocolError} : successor_selection.error();
      fail();
      return;
    }
    count_phase(RuntimePhaseCounter::kCandidatesSelected);
    trace_phase(RuntimePhaseTracePoint::kDirectCompositeSuccessorEntered,
                binding.successor_receiver,
                successor_frozen.witness.source,
                successor_frozen.witness.source_sequence);

    ClosedIslandActivationResult successor_result;
    run_closed_island_activation_transaction(exec,
                                             state,
                                             binding,
                                             *successor_selection,
                                             successor_frozen,
                                             false,
                                             true,
                                             successor_result);
    account(1U, successor_result);
    if (successor_result.disposition == ClosedIslandActivationDisposition::kFailed) {
      fail();
      return;
    }
    if (successor_result.disposition != ClosedIslandActivationDisposition::kCompleted) {
      fallback();
      return;
    }

    report.continuation = RuntimeDirectCompositeContinuationDisposition::kAccepted;
    report.resume = RuntimeDirectCompositeResumeDisposition::kCompositeCompleted;
    report.activation_cut_completed = true;
    count_phase(RuntimePhaseCounter::kDirectCompositeCompletions);
    trace_phase(RuntimePhaseTracePoint::kDirectCompositeReturned,
                binding.successor_receiver,
                successor_frozen.witness.source,
                successor_frozen.witness.source_sequence);
  }
  // Keep selection, policy gating, callback invocation, and exact credit retirement adjacent so a
  // terminal path cannot skip one part of the activation transaction.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] RunTermination dispatch_ready(GraphExec& exec, ReadyRunState& state) noexcept {
    while (state.dispatcher->active_count() != 0U) {
      if (stop_requested_.load(std::memory_order_acquire)) {
        state.dispatcher->cancel_all();
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kStopped;
      }
      if (state.partition_owner_execution) {
        // The source-tree barrier holds one already-frozen candidate before Dispatcher::select()
        // marks an activation in progress. That lets lifecycle tests place an accepted restart or
        // stop on the exact admission boundary without changing production plan identity.
        if (plan_->executable_factories.test_pre_dispatch_control != nullptr &&
            state.runner != nullptr && state.runner->operators.size() == 1U) {
          plan_->executable_factories.test_pre_dispatch_control->candidate_frozen(
              state.runner->operators.front().operator_id);
          if (stop_requested_.load(std::memory_order_acquire)) {
            state.dispatcher->cancel_all();
            static_cast<void>(cancel_ready_sources(state));
            return RunTermination::kStopped;
          }
        }
        if (lifecycle_control_->has_pending()) {
          // No activation is in progress at this boundary. Preserve the frozen prefix until the
          // complete owner round retires; the controller-side restart fence cancels affected
          // candidates, while unrelated candidates resume on the same stable owner afterward.
          return RunTermination::kCompleted;
        }
      }
      if (!state.partition_owner_execution && !process_pending_restarts(exec, state)) {
        append_diagnostic("RUNTIME_LIFECYCLE_CALLBACK_FAILED");
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      }
      if (state.dispatcher->active_count() == 0U) {
        break;
      }
      const RuntimeCallbackContextScope dispatcher_scope{RuntimeCallbackContext::kDispatcher};
      auto selected = state.dispatcher->select();
      if (!selected || selected->receiver_index >= exec.operators().size()) {
        append_diagnostic("READY_POLICY_PLAN_INVALID");
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      }
      count_phase(RuntimePhaseCounter::kCandidatesSelected);
      const scheduling::ReadyCandidateView candidate = selected->ready;
      const auto frozen =
          std::find_if(state.frozen_candidates.begin(),
                       state.frozen_candidates.end(),
                       [&](const FrozenRuntimeCandidate& value) {
                         // Tag sequence/generation is receiver-local. Match the
                         // receiver as well so two Operators' first candidates do
                         // not alias merely because both carry Tag{1, 1, 0}.
                         return value.candidate.receiver_index == selected->receiver_index &&
                                value.candidate.tag == selected->tag;
                       });
      if (frozen == state.frozen_candidates.end()) {
        append_diagnostic("READY_POLICY_PLAN_INVALID");
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      }
      RuntimeOperatorInstance& instance = exec.operators()[selected->receiver_index];
      if (selected->receiver_index >= state.parameter_policies.size()) {
        append_diagnostic("RUNTIME_PARAMETER_BINDING_UNAVAILABLE", instance.vertex);
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      }
      RuntimeDirectCompositeBinding* closed_island = nullptr;
      if (selected->receiver_index < state.direct_composite_owner_by_receiver.size()) {
        const std::size_t owner =
            state.direct_composite_owner_by_receiver[selected->receiver_index];
        if (owner != std::numeric_limits<std::size_t>::max()) {
          if (owner >= state.direct_composites_by_receiver.size() ||
              !state.direct_composites_by_receiver[owner].valid()) {
            append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
            static_cast<void>(cancel_ready_sources(state));
            return RunTermination::kFailed;
          }
          closed_island = &state.direct_composites_by_receiver[owner];
        }
      }
      if (closed_island != nullptr) {
        const bool source_member = selected->receiver_index == closed_island->producer_receiver;
        const bool composite_enabled =
            source_member && !plan_->executable_factories.test_disable_closed_island_composite &&
            !plan_->executable_factories.test_disable_direct_continuation_elision;
        RuntimeDirectCompositeCompletionReport composite_report;
        ClosedIslandActivationResult activation_result;
        struct ClosedIslandLaneTask {
          RunSessionState* session{};
          GraphExec* exec{};
          ReadyRunState* ready{};
          RuntimeDirectCompositeBinding* binding{};
          const DispatcherSelection* selection{};
          FrozenRuntimeCandidate* candidate{};
          ClosedIslandActivationResult* activation{};
          RuntimeDirectCompositeCompletionReport* composite{};
          bool source_member{};
        } task{
            .session = this,
            .exec = &exec,
            .ready = &state,
            .binding = closed_island,
            .selection = &*selected,
            .candidate = &*frozen,
            .activation = &activation_result,
            .composite = composite_enabled ? &composite_report : nullptr,
            .source_member = source_member,
        };
        const auto invoke = [](void* opaque) noexcept -> expected<void, Error> {
          auto& lane_task = *static_cast<ClosedIslandLaneTask*>(opaque);
          if (lane_task.composite != nullptr) {
            lane_task.session->run_direct_composite_lane(*lane_task.exec,
                                                         *lane_task.ready,
                                                         *lane_task.binding,
                                                         *lane_task.selection,
                                                         *lane_task.candidate,
                                                         *lane_task.composite);
          } else {
            lane_task.session->run_closed_island_activation_transaction(*lane_task.exec,
                                                                        *lane_task.ready,
                                                                        *lane_task.binding,
                                                                        *lane_task.selection,
                                                                        *lane_task.candidate,
                                                                        lane_task.source_member,
                                                                        false,
                                                                        *lane_task.activation);
          }
          return {};
        };
        const RuntimeCallbackIdentity identity{
            .context = RuntimeCallbackContext::kCompute,
            .operator_id = instance.operator_id,
            .source = frozen->witness.source,
            .source_sequence = frozen->witness.source_sequence,
            .tag = frozen->candidate.tag,
            .tag_time = frozen->candidate.ready.tag_time,
            .reference_time = frozen->witness.reference_time,
            .state_generation = frozen->candidate.tag.generation,
        };
        const ExecutionLaneTask lane_task =
            composite_enabled
                ? ExecutionLaneTask{
                      .context = &task,
                      .invoke = invoke,
                      .callback_context = RuntimeCallbackContext::kDispatcher,
                  }
                : ExecutionLaneTask{
                      .context = &task,
                      .invoke = invoke,
                      .callback_identity = identity,
                  };
        const expected<void, Error> dispatched =
            state.uses_inline_invoke() ? invoke_inline_owner_task(lane_task)
                                       : state.lanes->dispatch(instance.execution_lane, lane_task);
        if (!dispatched) {
          append_diagnostic("RUNTIME_OPERATOR_CALLBACK_FAILED", instance.vertex);
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        }
        if (composite_enabled) {
          if (composite_report.resume ==
              RuntimeDirectCompositeResumeDisposition::kTerminalFailure) {
            return RunTermination::kFailed;
          }
          if (composite_report.stop_observed) {
            return RunTermination::kStopped;
          }
          if (composite_report.resume ==
                  RuntimeDirectCompositeResumeDisposition::kCompositeCompleted &&
              !composite_report.complete()) {
            // A claimed completion whose bounded report is internally inconsistent cannot be
            // resumed safely; fail the run instead of exposing partially retired callbacks.
            append_diagnostic("RUNTIME_LIFECYCLE_CALLBACK_FAILED", instance.vertex);
            static_cast<void>(cancel_ready_sources(state));
            return RunTermination::kFailed;
          }
          continue;
        }
        if (activation_result.disposition == ClosedIslandActivationDisposition::kFailed) {
          return RunTermination::kFailed;
        }
        if (activation_result.disposition == ClosedIslandActivationDisposition::kStopped) {
          return RunTermination::kStopped;
        }
        if (activation_result.disposition == ClosedIslandActivationDisposition::kRestarted) {
          append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        }
        if (activation_result.direct_continuation_receipt &&
            state.channels->validate_direct_continuation_receipt(
                *activation_result.direct_continuation_receipt)) {
          if (state.direct_continuation_receipt) {
            append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE",
                              instance.vertex,
                              "direct_continuation_receipt_overflow");
            static_cast<void>(cancel_ready_sources(state));
            return RunTermination::kFailed;
          }
          state.direct_continuation_receipt =
              std::move(activation_result.direct_continuation_receipt);
          if (plan_->executable_factories.test_direct_continuation_control != nullptr) {
            plan_->executable_factories.test_direct_continuation_control->receipt_staged();
          }
        }
        if (!state.partition_owner_execution && lifecycle_control_->enabled() &&
            !process_pending_restarts(exec, state)) {
          append_diagnostic("RUNTIME_LIFECYCLE_CALLBACK_FAILED", instance.vertex);
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        }
        const CausalFrontierDisposition frontier = freeze_causal_successor(
            exec, state, selected->receiver_index, selected->ready.tag_time);
        if (frontier == CausalFrontierDisposition::kFailed) {
          return RunTermination::kFailed;
        }
        continue;
      }
      const OperatorRef self{
          .id = instance.operator_id,
          .plan_generation = plan_->plan_generation,
          .run_generation = run_generation_,
      };
      const bool active_parameter_updates = instance.runtime_parameter_updates_enabled ||
                                            instance.lifecycle_parameter_updates_enabled;
      ActivationParameterBinding parameter_binding{exec.parameter_updates(),
                                                   instance.static_parameter_header};
      if (active_parameter_updates) {
        if (instance.runtime_parameter_updates_enabled) {
          publish_pending_parameter_updates(exec.parameter_updates(), self);
        }
        auto current = exec.parameter_updates().pin_published(self);
        if (!current) {
          append_diagnostic("RUNTIME_PARAMETER_BINDING_UNAVAILABLE", instance.vertex);
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        }
        parameter_binding.bind(std::move(current).value());
      }
      const auto& parameter_policy = state.parameter_policies[selected->receiver_index];
      for (const std::uint32_t selector_index : parameter_policy.selector_indices) {
        if (selector_index >= state.input_selectors.size()) {
          append_diagnostic("RUNTIME_PARAMETER_BINDING_UNAVAILABLE", instance.vertex);
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        }
        auto& selector = state.input_selectors[selector_index];
        if (!selector.derived_term_index.has_value()) {
          continue;
        }
        const ParameterGenerationPin* const generation = parameter_binding.pin();
        const auto max_age =
            generation == nullptr
                ? std::optional<std::int64_t>{}
                : generation->duration_term_at(*selector.derived_term_index,
                                               ParameterDerivedTermKind::kLatestWithinMaxAge,
                                               selector.input);
        if (!max_age ||
            !selector.selector.set_latest_within_max_age(std::chrono::nanoseconds{*max_age})) {
          append_diagnostic("RUNTIME_PARAMETER_BINDING_UNAVAILABLE", instance.vertex);
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        }
      }
      TimePoint semantic_reference{};
      clear_input_selection_scratch(state, *frozen, semantic_reference);
      const std::uint64_t activation_generation =
          state.activation_selections[selected->receiver_index].generation();
      const bool channel_activation_required =
          state.channels != nullptr &&
          selected->receiver_index < state.channel_activation_owners.size() &&
          state.channel_activation_owners[selected->receiver_index] != 0U;
      auto& predicate = state.predicates[selected->receiver_index];
      const PredicateTriggerPlanRecord* const predicate_trigger =
          predicate ? predicate_trigger_for(*plan_,
                                            OperatorRef{.id = instance.operator_id,
                                                        .plan_generation = plan_->plan_generation})
                    : nullptr;
      SelectionPreparation selection_preparation = SelectionPreparation::kReady;
      if (predicate_trigger == nullptr) {
        selection_preparation =
            prepare_nonpredicate_input_selections(state, instance, *frozen, semantic_reference);
      } else {
        auto predicate_inputs = prepare_predicate_input_selections(
            state, instance, *predicate_trigger, *frozen, semantic_reference);
        if (!predicate_inputs) {
          selection_preparation = predicate_inputs.error().code == ErrorCode::kNotReady
                                      ? SelectionPreparation::kNoFire
                                      : SelectionPreparation::kFailed;
        }
      }
      const std::size_t predicate_selector_transaction_count = state.selector_transactions.size();
      const std::size_t predicate_channel_selection_count = state.channel_selection_scratch.size();
      if (selection_preparation == SelectionPreparation::kFailed) {
        static_cast<void>(finish_input_selections(state, false));
        append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      }
      bool channel_activation_prepared = false;
      if (selection_preparation == SelectionPreparation::kReady && channel_activation_required) {
        auto prepared = state.channels->prepare_activation(
            instance.vertex, activation_generation, state.channel_selection_scratch);
        if (!prepared) {
          static_cast<void>(finish_input_selections(state, false));
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        }
        auto tensor_access = state.channels->prepare_tensor_access(
            instance.vertex, activation_generation, instance.compute_stream);
        if (!tensor_access) {
          static_cast<void>(finish_input_selections(state, false));
          static_cast<void>(
              state.channels->retire_activation(instance.vertex, activation_generation, false));
          append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        }
        channel_activation_prepared = true;
      }
      if (selection_preparation == SelectionPreparation::kNoFire) {
        const bool selectors_aborted = finish_input_selections(state, false);
        const bool channel_retired =
            !channel_activation_prepared ||
            state.channels->retire_activation(instance.vertex, activation_generation, false)
                .has_value();
        if (!state.dispatcher->complete(*selected) || !retire_frozen_source(state, *frozen) ||
            !selectors_aborted || !channel_retired) {
          append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE", instance.vertex);
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        }
        continue;
      }
      ActivationStatus predicate_status{ActivationReason::kNominal, OnViolation::kEmitWitness};
      if (predicate) {
        if (predicate_trigger == nullptr) {
          append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
          static_cast<void>(finish_input_selections(state, false));
          if (channel_activation_prepared) {
            static_cast<void>(
                state.channels->retire_activation(instance.vertex, activation_generation, false));
          }
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        }
        state.predicate_input_scratch.clear();
        const std::size_t predicate_input_count = state.input_selection_scratch.size();
        for (std::size_t index = 0U; index < predicate_input_count; ++index) {
          const auto& channel_selection = state.channel_selection_scratch[index];
          const InputSelection input_selection = state.input_selection_scratch[index];
          const bool has_value = input_selection.state == InputState::kSelected ||
                                 input_selection.state == InputState::kReused;
          const void* typed_view = nullptr;
          if (has_value) {
            if (state.channels == nullptr) {
              append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
              static_cast<void>(finish_input_selections(state, false));
              static_cast<void>(cancel_ready_sources(state));
              return RunTermination::kFailed;
            }
            auto view =
                state.channels->predicate_view(channel_selection.port, activation_generation);
            if (!view) {
              append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
              static_cast<void>(finish_input_selections(state, false));
              static_cast<void>(
                  state.channels->retire_activation(instance.vertex, activation_generation, false));
              static_cast<void>(cancel_ready_sources(state));
              return RunTermination::kFailed;
            }
            typed_view = *view;
          }
          state.predicate_input_scratch.push_back(PredicateFrozenInput{
              .port = channel_selection.port,
              .selection = input_selection,
              .typed_view = typed_view,
          });
        }
        const std::size_t compute_input_count = static_cast<std::size_t>(std::count_if(
            state.input_selectors.begin(), state.input_selectors.end(), [&](const auto& slot) {
              return slot.owner == instance.vertex &&
                     !(slot.predicate_input && slot.predicate_id == predicate_trigger->id);
            }));
        PredicateComputeSelectionContext compute_context{
            .session = this,
            .ready = &state,
            .instance = &instance,
            .predicate = predicate_trigger,
            .candidate = &*frozen,
            .semantic_reference = &semantic_reference,
        };
        auto decision = predicate->evaluate(
            PredicateFrozenSource{
                .source = frozen->witness.source,
                .source_sequence = frozen->witness.source_sequence,
                .driver_port = predicate_trigger->driver,
                .reference_time = frozen->witness.reference_time,
                .tag = frozen->candidate.tag,
                .tag_time = frozen->candidate.ready.tag_time,
            },
            state.predicate_input_scratch,
            PredicateComputeSelectionRequest{
                .planned_input_count = compute_input_count,
                .state = &compute_context,
                .select = &select_predicate_compute_inputs,
            },
            state.predicate_receivers[selected->receiver_index]);
        if (!decision) {
          const PredicateTriggerPlanRecord* const trigger =
              predicate_trigger_for(*plan_,
                                    OperatorRef{
                                        .id = instance.operator_id,
                                        .plan_generation = plan_->plan_generation,
                                    });
          const RuntimeDiagnosticDetail detail{
              .present = true,
              .error = decision.error(),
              .predicate_id = trigger != nullptr ? trigger->id : std::string{},
              .operator_id = instance.operator_id,
              .source = frozen->witness.source,
              .source_sequence = frozen->witness.source_sequence,
              .tag = frozen->candidate.tag,
              .tag_time = frozen->candidate.ready.tag_time,
              .reference_time = frozen->witness.reference_time,
              .state_generation = trigger != nullptr ? trigger->state_generation : 0U,
          };
          append_diagnostic(
              "RUNTIME_OPERATOR_CALLBACK_FAILED", instance.vertex, "predicate_runtime", &detail);
          static_cast<void>(finish_input_selections(state, false));
          if (channel_activation_prepared) {
            static_cast<void>(
                state.channels->retire_activation(instance.vertex, activation_generation, false));
          }
          static_cast<void>(state.dispatcher->complete(*selected));
          static_cast<void>(retire_frozen_source(state, *frozen));
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        }
        if (decision->disposition == PredicateCandidateDisposition::kEvaluatorError) {
          const RuntimeDiagnosticDetail detail{
              .present = true,
              .error = decision->evaluator_error,
              .predicate_id = std::string(decision->predicate_id),
              .operator_id = decision->operator_id,
              .source = decision->source.source,
              .source_sequence = decision->source.source_sequence,
              .tag = decision->source.tag,
              .tag_time = decision->source.tag_time,
              .reference_time = decision->source.reference_time,
              .state_generation = decision->state_generation,
          };
          append_diagnostic(
              "RUNTIME_OPERATOR_CALLBACK_FAILED", instance.vertex, "predicate_evaluator", &detail);
          static_cast<void>(finish_input_selections(state, false));
          if (channel_activation_prepared) {
            static_cast<void>(
                state.channels->retire_activation(instance.vertex, activation_generation, false));
          }
          static_cast<void>(state.dispatcher->complete(*selected));
          static_cast<void>(retire_frozen_source(state, *frozen));
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        }
        if (decision->disposition == PredicateCandidateDisposition::kNoFire) {
          // A successful evaluation commits remembered predicate truth, so its exact input
          // snapshot must retire with it. Any compute-only suffix remains uncommitted: a Skip or
          // other compute-input miss suppresses the callback without consuming data that the
          // callback never observed. Missing predicate inputs and evaluator failures keep the
          // existing all-or-nothing rollback path.
          const bool predicate_inputs_accepted = decision->evaluator_called;
          const bool dispatcher_completed = state.dispatcher->complete(*selected).has_value();
          const bool source_retired =
              dispatcher_completed && retire_frozen_source(state, *frozen).has_value();
          const bool commit_predicate_inputs = predicate_inputs_accepted && source_retired;
          const bool selectors_finished =
              commit_predicate_inputs ? finish_input_selection_prefix(
                                            state, predicate_selector_transaction_count, false)
                                      : finish_input_selections(state, false);
          const bool channel_retired =
              !channel_activation_prepared ||
              state.channels
                  ->retire_activation(
                      instance.vertex, activation_generation, commit_predicate_inputs)
                  .has_value();
          if (!dispatcher_completed || !source_retired || !selectors_finished || !channel_retired) {
            append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE", instance.vertex);
            static_cast<void>(cancel_ready_sources(state));
            return RunTermination::kFailed;
          }
          continue;
        }
        if (channel_activation_required &&
            state.channel_selection_scratch.size() > predicate_channel_selection_count) {
          auto prepared = state.channels->prepare_activation(
              instance.vertex,
              activation_generation,
              std::span<const RuntimeChannelActivationSelection>{state.channel_selection_scratch}
                  .subspan(predicate_channel_selection_count));
          if (!prepared) {
            static_cast<void>(finish_input_selections(state, false));
            if (channel_activation_prepared) {
              static_cast<void>(
                  state.channels->retire_activation(instance.vertex, activation_generation, false));
            }
            append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
            static_cast<void>(cancel_ready_sources(state));
            return RunTermination::kFailed;
          }
          auto tensor_access = state.channels->prepare_tensor_access(
              instance.vertex, activation_generation, instance.compute_stream);
          if (!tensor_access) {
            static_cast<void>(finish_input_selections(state, false));
            static_cast<void>(
                state.channels->retire_activation(instance.vertex, activation_generation, false));
            append_diagnostic("RUNTIME_PORT_BINDING_UNAVAILABLE", instance.vertex);
            static_cast<void>(cancel_ready_sources(state));
            return RunTermination::kFailed;
          }
        }
        if (decision->disposition == PredicateCandidateDisposition::kDegradedCompute) {
          predicate_status =
              ActivationStatus{ActivationReason::kInvariantViolated, OnViolation::kSafeDefault};
        }
      }
      auto activation =
          state.activation_selections[selected->receiver_index].begin(*frozen,
                                                                      state.input_selection_scratch,
                                                                      state.selected_sample_scratch,
                                                                      semantic_reference);
      if (!activation) {
        static_cast<void>(finish_input_selections(state, false));
        if (channel_activation_prepared) {
          static_cast<void>(
              state.channels->retire_activation(instance.vertex, activation_generation, false));
        }
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_MISMATCH", instance.vertex);
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      }
      const auto abort_selection = [&]() noexcept {
        const bool selectors_aborted = finish_input_selections(state, false);
        const bool channel_retired =
            !channel_activation_prepared ||
            state.channels->retire_activation(instance.vertex, activation_generation, false)
                .has_value();
        const bool activation_retired =
            state.activation_selections[selected->receiver_index].retire(selected->tag).has_value();
        return selectors_aborted && channel_retired && activation_retired;
      };
      ExecutionContextState context;
      context.tag = selected->tag;
      context.tag_time = candidate.tag_time;
      context.activation_time = frozen->witness.reference_time;
      context.clock = exec.default_clock();
      context.cuda_stream = instance.compute_stream;
      context.cancellation_source = &stop_requested_;
      context.pinned_params = parameter_binding.header();
      context.activation = predicate_status;
      context.self = self;
      ComputeParameterIngress compute_parameter_ingress{
          .coordinator = &exec.parameter_updates(),
          .self = self,
      };
      context.param_control_state = &compute_parameter_ingress;
      context.submit_param_update = &submit_compute_parameter_update;

      const auto& deadline = state.deadline_bindings[selected->receiver_index];
      std::int64_t deadline_budget_ns{};
      std::optional<ClockRef> deadline_clock;
      if (deadline) {
        deadline_budget_ns = deadline->deadline->budget_ns;
        if (parameter_policy.deadline_term_index.has_value()) {
          const ParameterGenerationPin* const generation = parameter_binding.pin();
          const auto pinned_budget =
              generation == nullptr
                  ? std::optional<std::int64_t>{}
                  : generation->duration_term_at(*parameter_policy.deadline_term_index,
                                                 ParameterDerivedTermKind::kContractDeadline);
          if (!pinned_budget) {
            static_cast<void>(abort_selection());
            append_diagnostic("RUNTIME_PARAMETER_BINDING_UNAVAILABLE", instance.vertex);
            static_cast<void>(cancel_ready_sources(state));
            return RunTermination::kFailed;
          }
          deadline_budget_ns = *pinned_budget;
        }
        auto dispatch_clock = exec.resolve_clock(deadline->reference_clock);
        if (!dispatch_clock) {
          static_cast<void>(abort_selection());
          append_diagnostic("RUNTIME_CLOCK_BINDING_UNAVAILABLE", instance.vertex);
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        }
        deadline_clock.emplace(std::move(*dispatch_clock));
      }
      struct ComputeRestartBinding {
        LifecycleControlState* control{};
        OperatorRef requester{};
      } restart_binding{
          .control = lifecycle_control_.get(),
          .requester = context.self,
      };
      if (lifecycle_control_->enabled()) {
        context.restart_state = &restart_binding;
        context.request_restart = [](void* state,
                                     OperatorRef target,
                                     LifecycleRestartOptions options) noexcept {
          auto& binding = *static_cast<ComputeRestartBinding*>(state);
          return binding.control->submit(target.id == binding.requester.id
                                             ? LifecycleRuntimeRequestOrigin::kComputeSelf
                                             : LifecycleRuntimeRequestOrigin::kComputeCrossTarget,
                                         binding.requester,
                                         target,
                                         options,
                                         nullptr);
        };
      }

      ComputeLaneInvocation invocation{
          .instance = &instance,
          .context = &context,
          .dispatcher = state.dispatcher.get(),
          .selection = &*selected,
          .deadline = deadline ? &*deadline : nullptr,
          .deadline_clock = deadline_clock.value_or(ClockRef{}),
          .source_decision_time = frozen->witness.source_decision_time,
          .reference_time = frozen->witness.reference_time,
          .deadline_budget_ns = deadline_budget_ns,
          .parameter_updates = &exec.parameter_updates(),
          .self = self,
          .parameter_generation = parameter_binding.pin(),
          .source_sequence = frozen->witness.source_sequence,
          .cuda_device = instance.compute_device,
          .phase_counters = phase_counters_,
      };
      if (state.channels != nullptr && !state.direct_continuation_receipt &&
          !lifecycle_control_->enabled() &&
          !plan_->executable_factories.test_disable_direct_continuation_elision) {
        const auto target =
            state.channels->arm_direct_continuation(instance.vertex, instance.execution_lane);
        if (target) {
          const auto source_credit =
              state.coordinator->source_credit_remaining(target->readiness_source);
          if (!source_credit) {
            static_cast<void>(abort_selection());
            append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE", instance.vertex);
            static_cast<void>(cancel_ready_sources(state));
            return RunTermination::kFailed;
          }
          // Full source credit proves that this receipt is the successor's sole pending
          // publication. A partially occupied binding must retain the ordinary globally ordered
          // readiness scan.
          if (*source_credit == target->readiness_capacity) {
            auto clock = exec.resolve_clock(target->reference_clock);
            if (!clock) {
              static_cast<void>(abort_selection());
              append_diagnostic("RUNTIME_CLOCK_BINDING_UNAVAILABLE", instance.vertex);
              static_cast<void>(cancel_ready_sources(state));
              return RunTermination::kFailed;
            }
            invocation.channels = state.channels.get();
            invocation.channel_owner = instance.vertex;
            invocation.execution_lane = instance.execution_lane;
            invocation.direct_continuation_clock = std::move(*clock);
            invocation.direct_continuation_enabled = true;
          }
        }
      }
      count_phase(RuntimePhaseCounter::kComputeDispatches);
      trace_phase(RuntimePhaseTracePoint::kComputeDispatchStarted,
                  selected->receiver_index,
                  frozen->witness.source,
                  frozen->witness.source_sequence);
      const RuntimeCallbackIdentity source_identity{
          .context = RuntimeCallbackContext::kCompute,
          .operator_id = instance.operator_id,
          .source = frozen->witness.source,
          .source_sequence = frozen->witness.source_sequence,
          .tag = frozen->candidate.tag,
          .tag_time = frozen->candidate.ready.tag_time,
          .reference_time = frozen->witness.reference_time,
          .state_generation = frozen->candidate.tag.generation,
      };
      const ExecutionLaneTask lane_task{
          .context = &invocation,
          .invoke = compute_lane_invoke_,
          .callback_identity = source_identity,
      };
      const expected<void, Error> computed =
          state.uses_inline_invoke() ? invoke_inline_owner_task(lane_task)
                                     : state.lanes->dispatch(instance.execution_lane, lane_task);
      trace_phase(RuntimePhaseTracePoint::kComputeDispatchReturned,
                  selected->receiver_index,
                  source_identity.source,
                  source_identity.source_sequence);
      if (invocation.deadline_gate &&
          invocation.deadline_gate->disposition != DeadlineGateDisposition::kInvokeNominal) {
        const RuntimeDiagnosticDetail detail{
            .present = true,
            .operator_id = instance.operator_id,
            .source = frozen->witness.source,
            .source_sequence = frozen->witness.source_sequence,
            .tag = frozen->candidate.tag,
            .tag_time = frozen->candidate.ready.tag_time,
            .reference_time = frozen->witness.reference_time,
            .state_generation = frozen->candidate.tag.generation,
            .deadline_miss =
                RuntimeDeadlineMissDiagnostic{
                    .present = true,
                    .source_coordinate = deadline->deadline->source_coordinate,
                    .source_decision_time = frozen->witness.source_decision_time,
                    .reference_time = frozen->witness.reference_time,
                    .budget_ns = deadline_budget_ns,
                    .latest_start = invocation.deadline_gate->latest_start,
                    .dispatch_start = invocation.deadline_gate->dispatch_start,
                    .checked_lateness_ns = invocation.deadline_gate->checked_lateness_ns,
                    .action = deadline->deadline->action,
                    .callback_count = deadline->deadline->callback_count_if_late,
                },
        };
        append_diagnostic("DEADLINE_MISS",
                          instance.vertex,
                          "latest_start",
                          &detail,
                          {},
                          {},
                          ::holoscan::DiagnosticSeverity::kWarning);
      }
      switch (invocation.admission) {
        case ComputeAdmissionDisposition::kDeadlineReadFailed:
          static_cast<void>(abort_selection());
          append_diagnostic("RUNTIME_CLOCK_READ_FAILED", instance.vertex);
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        case ComputeAdmissionDisposition::kDeadlineEvaluationFailed: {
          static_cast<void>(abort_selection());
          const RuntimeDiagnosticDetail detail{
              .present = true,
              .error = std::move(invocation.admission_error),
              .operator_id = instance.operator_id,
              .source = frozen->witness.source,
              .source_sequence = frozen->witness.source_sequence,
              .tag = frozen->candidate.tag,
              .tag_time = frozen->candidate.ready.tag_time,
              .reference_time = frozen->witness.reference_time,
              .state_generation = frozen->candidate.tag.generation,
          };
          append_diagnostic(
              "RUNTIME_DEADLINE_EVALUATION_FAILED", instance.vertex, "latest_start", &detail);
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        }
        case ComputeAdmissionDisposition::kDeadlineFailRun:
          static_cast<void>(abort_selection());
          static_cast<void>(state.dispatcher->complete(*selected));
          static_cast<void>(retire_frozen_source(state, *frozen));
          append_diagnostic("RUNTIME_DEADLINE_FAIL_RUN", instance.vertex, "latest_start");
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        case ComputeAdmissionDisposition::kParameterObservationFailed:
          static_cast<void>(abort_selection());
          append_diagnostic("RUNTIME_PARAMETER_BINDING_UNAVAILABLE", instance.vertex);
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        case ComputeAdmissionDisposition::kCudaDeviceUnavailable:
          static_cast<void>(abort_selection());
          append_diagnostic("RUNTIME_COMPUTE_CUDA_DEVICE_UNAVAILABLE", instance.vertex);
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        case ComputeAdmissionDisposition::kPending:
        case ComputeAdmissionDisposition::kAdmitted:
          break;
      }
      const bool activation_retired = retire_activation_transaction(exec,
                                                                    state,
                                                                    instance,
                                                                    *selected,
                                                                    *frozen,
                                                                    activation_generation,
                                                                    channel_activation_prepared,
                                                                    parameter_binding,
                                                                    computed.has_value(),
                                                                    invocation.completed_selection);
      if (!computed) {
        const RuntimeDiagnosticDetail detail{
            .present = true,
            .error = computed.error(),
            .operator_id = instance.operator_id,
            .source = frozen->witness.source,
            .source_sequence = frozen->witness.source_sequence,
            .tag = frozen->candidate.tag,
            .tag_time = frozen->candidate.ready.tag_time,
            .reference_time = frozen->witness.reference_time,
            .state_generation = frozen->candidate.tag.generation,
        };
        append_diagnostic(
            "RUNTIME_OPERATOR_CALLBACK_FAILED", instance.vertex, "operator_compute", &detail);
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      }
      if (!activation_retired) {
        append_diagnostic("RUNTIME_LIFECYCLE_CALLBACK_FAILED", instance.vertex);
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      }
      if (invocation.direct_continuation_error.code != ErrorCode::kSuccess) {
        const RuntimeDiagnosticDetail detail{
            .present = true,
            .error = std::move(invocation.direct_continuation_error),
            .operator_id = instance.operator_id,
            .source = frozen->witness.source,
            .source_sequence = frozen->witness.source_sequence,
            .tag = frozen->candidate.tag,
            .tag_time = frozen->candidate.ready.tag_time,
            .reference_time = frozen->witness.reference_time,
            .state_generation = frozen->candidate.tag.generation,
        };
        append_diagnostic(invocation.direct_continuation_clock_failed
                              ? std::string_view{"RUNTIME_CLOCK_READ_FAILED"}
                              : std::string_view{"RUNTIME_READINESS_BINDING_UNAVAILABLE"},
                          instance.vertex,
                          "direct_continuation",
                          &detail);
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      }
      if (!state.partition_owner_execution && lifecycle_control_->enabled() &&
          !process_pending_restarts(exec, state)) {
        append_diagnostic("RUNTIME_LIFECYCLE_CALLBACK_FAILED", instance.vertex);
        static_cast<void>(cancel_ready_sources(state));
        return RunTermination::kFailed;
      }
      if (invocation.direct_continuation_receipt &&
          state.channels->validate_direct_continuation_receipt(
              *invocation.direct_continuation_receipt)) {
        if (state.direct_continuation_receipt) {
          append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE",
                            instance.vertex,
                            "direct_continuation_receipt_overflow");
          static_cast<void>(cancel_ready_sources(state));
          return RunTermination::kFailed;
        }
        state.direct_continuation_receipt = std::move(invocation.direct_continuation_receipt);
        if (plan_->executable_factories.test_direct_continuation_control != nullptr) {
          plan_->executable_factories.test_direct_continuation_control->receipt_staged();
        }
      }
      const CausalFrontierDisposition frontier =
          freeze_causal_successor(exec, state, selected->receiver_index, selected->ready.tag_time);
      if (frontier == CausalFrontierDisposition::kFailed) {
        return RunTermination::kFailed;
      }
    }
    return RunTermination::kCompleted;
  }

  [[nodiscard]] bool all_sources_closed(const ReadyRunState& state) const noexcept {
    return std::all_of(state.bindings.begin(), state.bindings.end(), [](const auto& binding) {
      return binding->closed();
    });
  }

  [[nodiscard]] bool wait_for_source_progress(GraphExec& exec,
                                              const ReadyRunState& state) noexcept {
    using namespace std::chrono_literals;
    std::chrono::nanoseconds wait_duration = 1ms;
    for (const auto& binding : state.bindings) {
      const std::optional<PortBindingPollTarget> target = binding->poll_target();
      if (target && target->busy_poll) {
        // One controller loop performs one bounded poll. Returning here keeps
        // stop checks cooperative without parking the admitted busy-poll lane.
        return true;
      }
      const std::optional<TimePoint> wake = binding->next_wake();
      if (!wake) {
        continue;
      }
      auto clock = exec.resolve_clock(wake->clock_id);
      auto now = clock ? clock->now() : expected<TimePoint, Error>{make_unexpected(clock.error())};
      if (!now) {
        append_diagnostic("RUNTIME_CLOCK_READ_FAILED");
        return false;
      }
      if (now->timestamp_ns >= wake->timestamp_ns) {
        return true;
      }
      wait_duration =
          bounded_source_wait_duration(now->timestamp_ns, wake->timestamp_ns, wait_duration);
    }
    if (stop_requested_.load(std::memory_order_acquire)) {
      return true;
    }
    RuntimeWakeState& wake = state.wake != nullptr ? *state.wake : readiness_wake_;
    if (phase_counters_ == nullptr) {
      static_cast<void>(wake.wait_for(wait_duration));
      return true;
    }
    phase_counters_->increment(RuntimePhaseCounter::kWaitCalls);
    phase_counters_->trace(
        RuntimePhaseTracePoint::kControllerParkStarted, kNoRuntimePhaseReceiver, {}, 0U);
    if (wake.wait_for(wait_duration)) {
      phase_counters_->increment(RuntimePhaseCounter::kWakeReturns);
      phase_counters_->trace(
          RuntimePhaseTracePoint::kControllerWakeReturned, kNoRuntimePhaseReceiver, {}, 0U);
    } else {
      phase_counters_->increment(RuntimePhaseCounter::kTimeoutReturns);
      phase_counters_->trace(
          RuntimePhaseTracePoint::kControllerTimeoutReturned, kNoRuntimePhaseReceiver, {}, 0U);
    }
    return true;
  }

  struct PartitionOwnerGroup {
    std::atomic_bool failed{false};
  };

  struct PartitionOwnerContext {
    RunSessionState* session{};
    GraphExec* exec{};
    ReadyRunState* ready{};
    PartitionOwnerGroup* group{};
    RunTermination outcome{RunTermination::kCompleted};
  };

  // The owner loop serializes lifecycle, readiness, activation, and termination transitions.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] RunTermination run_partition_owner(PartitionOwnerContext& context) noexcept {
    GraphExec& exec = *context.exec;
    ReadyRunState& ready = *context.ready;
    while (!context.group->failed.load(std::memory_order_acquire)) {
      if (stop_requested_.load(std::memory_order_acquire)) {
        return RunTermination::kStopped;
      }
      if (lifecycle_control_->has_pending()) {
        return RunTermination::kCompleted;
      }
      if (ready.dispatcher->active_count() != 0U) {
        const RunTermination dispatched = dispatch_ready(exec, ready);
        if (dispatched != RunTermination::kCompleted) {
          if (dispatched == RunTermination::kFailed) {
            context.group->failed.store(true, std::memory_order_release);
            signal_controller_wake(RuntimePhaseCounter::kWakeSignals);
          }
          return dispatched;
        }
        if (lifecycle_control_->has_pending()) {
          return RunTermination::kCompleted;
        }
        count_phase(RuntimePhaseCounter::kPartitionTailDispatchAttempts);
        count_phase((ready.partition_owner_execution || ready.direct_continuation_receipt)
                        ? RuntimePhaseCounter::kPartitionTailDispatchAccepted
                        : RuntimePhaseCounter::kPartitionTailDispatchFallbacks);
        continue;
      }
      RuntimeWakeState& wake = ready.wake != nullptr ? *ready.wake : readiness_wake_;
      static_cast<void>(wake.drain());
      count_phase(RuntimePhaseCounter::kReadinessScans);
      count_phase(RuntimePhaseCounter::kPartitionRunnerSweeps);
      if (!freeze_ready_candidates(exec, ready, ready.runner->execution_lane)) {
        context.group->failed.store(true, std::memory_order_release);
        signal_controller_wake(RuntimePhaseCounter::kWakeSignals);
        return RunTermination::kFailed;
      }
      if (lifecycle_control_->has_pending()) {
        return RunTermination::kCompleted;
      }
      if (ready.dispatcher->active_count() != 0U) {
        const RunTermination dispatched = dispatch_ready(exec, ready);
        if (dispatched != RunTermination::kCompleted) {
          if (dispatched == RunTermination::kFailed) {
            context.group->failed.store(true, std::memory_order_release);
            signal_controller_wake(RuntimePhaseCounter::kWakeSignals);
          }
          return dispatched;
        }
        if (lifecycle_control_->has_pending()) {
          return RunTermination::kCompleted;
        }
        count_phase(RuntimePhaseCounter::kPartitionTailDispatchAttempts);
        count_phase((ready.partition_owner_execution || ready.direct_continuation_receipt)
                        ? RuntimePhaseCounter::kPartitionTailDispatchAccepted
                        : RuntimePhaseCounter::kPartitionTailDispatchFallbacks);
        continue;
      }
      if (all_sources_closed(ready)) {
        return RunTermination::kCompleted;
      }
      count_phase(RuntimePhaseCounter::kNoProgressScans);
      if (!wait_for_source_progress(exec, ready)) {
        context.group->failed.store(true, std::memory_order_release);
        signal_controller_wake(RuntimePhaseCounter::kWakeSignals);
        return RunTermination::kFailed;
      }
    }
    return RunTermination::kFailed;
  }

  [[nodiscard]] static expected<void, Error> invoke_partition_owner(void* opaque) noexcept {
    if (opaque == nullptr) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "RUNTIME_PARTITION_RUNNER_CONTEXT_INVALID"});
    }
    auto& context = *static_cast<PartitionOwnerContext*>(opaque);
    if (context.session == nullptr || context.exec == nullptr || context.ready == nullptr ||
        context.group == nullptr || context.ready->runner == nullptr) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "RUNTIME_PARTITION_RUNNER_CONTEXT_INVALID"});
    }
    context.outcome = context.session->run_partition_owner(context);
    return context.outcome == RunTermination::kFailed
               ? expected<void, Error>{make_unexpected(
                     Error{ErrorCode::kFailure, "RUNTIME_PARTITION_RUNNER_FAILED"})}
               : expected<void, Error>{};
  }

  // Keep startup, dispatch, and terminal cleanup in one transaction so ReadyRunState ownership and
  // shutdown ordering remain visible at the call site.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  [[nodiscard]] RunTermination worker_main_impl() {
    if (plan_->executable_factories.support_thread_control != nullptr) {
      auto entered = plan_->executable_factories.support_thread_control->enter(
          RuntimeSupportThreadRole::kController);
      if (!entered) {
        HOLOSCAN_LOG_ERROR("Runtime controller thread setup failed with {}: {}",
                           error_code_name(entered.error().code),
                           entered.error().message);
        append_diagnostic("RUNTIME_STARTUP_FAILED");
        return RunTermination::kFailed;
      }
    }
    RuntimeReplayMismatchDetail replay_mismatch;
    auto reconstructed =
        GraphExec::reconstruct(plan_, run_generation_, process_group_, &replay_mismatch);
    if (!reconstructed) {
      const std::string_view token =
          registered_token_or(reconstructed.error().message, "RUNTIME_STARTUP_FAILED");
      if (replay_mismatch.present) {
        const std::string expected = replay_digest_text(replay_mismatch.expected);
        const std::string observed = replay_digest_text(replay_mismatch.observed);
        const RuntimeDiagnosticDetail detail{
            .present = true,
            .error = reconstructed.error(),
            .operator_id = replay_mismatch.operator_id,
            .replay_mismatch = replay_mismatch,
        };
        append_diagnostic("RUNTIME_OPERATOR_REPLAY_FINGERPRINT_MISMATCH",
                          replay_mismatch.vertex,
                          {},
                          &detail,
                          expected,
                          observed);
      } else {
        append_diagnostic(token);
      }
      return RunTermination::kFailed;
    }
    graph_exec_ = std::move(reconstructed).value();
    if (plan_->graph.vertices().empty()) {
      retire_parameter_control();
      return RunTermination::kCompleted;
    }
    if (auto bound = graph_exec_->parameter_updates().bind_lifecycle_submitter(
            lifecycle_control_->parameter_submit_binding());
        !bound) {
      append_diagnostic("RUNTIME_PARAMETER_CONTROL_BINDING_FAILED");
      retire_parameter_control();
      return RunTermination::kFailed;
    }
    if (auto attached = attach_parameter_control(graph_exec_->parameter_updates()); !attached) {
      append_diagnostic(registered_token_or(attached.error().message,
                                            "RUNTIME_PARAMETER_CONTROL_BINDING_FAILED"));
      retire_parameter_control();
      return RunTermination::kFailed;
    }
    GraphExec& exec = *graph_exec_;
    std::vector<ReadyRunState> ready_states;
    if (!materialize_partition_runner_states(exec, ready_states)) {
      return RunTermination::kFailed;
    }
    ReadyRunState& shared_ready = ready_states.front();
    if (!initialize_channel_session(exec, shared_ready) || !initialize_lanes(exec, shared_ready)) {
      teardown_startup_runtime(shared_ready);
      return RunTermination::kFailed;
    }
    for (std::size_t index = 1U; index < ready_states.size(); ++index) {
      ReadyRunState& ready = ready_states[index];
      ready.channels = shared_ready.channels;
      ready.lanes = shared_ready.lanes;
      ready.channel_activation_owners = shared_ready.channel_activation_owners;
      ready.channel_input_owners = shared_ready.channel_input_owners;
    }
    auto tensor_resources = RuntimeTensorResources::create(
        *plan_, exec.operators(), run_generation_, exec.process_group());
    if (!tensor_resources) {
      HOLOSCAN_LOG_ERROR("Runtime Tensor resource materialization failed: {}",
                         runtime_error_message(tensor_resources.error()));
      append_diagnostic("PROFILE_MATERIALIZATION_FAILED");
      teardown_startup_runtime(shared_ready);
      return RunTermination::kFailed;
    }
    tensor_resources_ = std::move(*tensor_resources);
    for (std::size_t index = 0U; index < ready_states.size(); ++index) {
      ReadyRunState& ready = ready_states[index];
      if (index != 0U) {
        ready.clock_health = shared_ready.clock_health;
      }
      if (!initialize_ready_state(exec, ready)) {
        teardown_startup_runtime(shared_ready);
        return RunTermination::kFailed;
      }
    }
    if (!bind_shared_production_notifications(exec, ready_states)) {
      teardown_startup_runtime(shared_ready);
      return RunTermination::kFailed;
    }
    if (auto lifecycle = exec.initialize_lifecycle(run_generation_,
                                                   &stop_requested_,
                                                   shared_ready.lanes.get(),
                                                   shared_ready.notifications.get());
        !lifecycle) {
      append_diagnostic(
          registered_token_or(lifecycle.error().message, "RUNTIME_LIFECYCLE_PLAN_INVALID"));
      teardown_startup_runtime(shared_ready);
      return RunTermination::kFailed;
    }

    RunTermination outcome = RunTermination::kCompleted;
    if (stop_requested_.load(std::memory_order_acquire)) {
      outcome = RunTermination::kStopped;
    } else if (!exec.lifecycle_startup()) {
      append_diagnostic(exec.lifecycle_failure_token());
      outcome = RunTermination::kFailed;
    }
    if (outcome == RunTermination::kCompleted && shared_ready.notifications != nullptr &&
        !shared_ready.notifications->arm_startup()) {
      append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
      outcome = RunTermination::kFailed;
    }
    if (outcome == RunTermination::kCompleted) {
      if (ready_states.size() == 1U) {
        if (!bind_ready_sources(exec, shared_ready)) {
          outcome = RunTermination::kFailed;
        }
      } else {
        for (ReadyRunState& ready : ready_states) {
          if (!bind_ready_sources(exec, ready, false)) {
            outcome = RunTermination::kFailed;
            break;
          }
        }
        if (outcome == RunTermination::kCompleted && !arm_shared_channels(shared_ready)) {
          outcome = RunTermination::kFailed;
        }
      }
    }
    if (outcome == RunTermination::kCompleted && !lifecycle_control_->open_ingress()) {
      if (stop_requested_.load(std::memory_order_acquire)) {
        outcome = RunTermination::kStopped;
      } else {
        append_diagnostic("RUNTIME_STARTUP_FAILED");
        outcome = RunTermination::kFailed;
      }
    }
    if (outcome == RunTermination::kCompleted && shared_ready.runner != nullptr) {
      count_phase(RuntimePhaseCounter::kPartitionControlBarriers);
    }
    if (outcome == RunTermination::kCompleted && ready_states.size() == 1U) {
      ReadyRunState& ready = shared_ready;
      while (outcome == RunTermination::kCompleted) {
        if (stop_requested_.load(std::memory_order_acquire)) {
          outcome = RunTermination::kStopped;
          break;
        }
        if (!process_pending_restarts(exec, ready)) {
          append_diagnostic("RUNTIME_LIFECYCLE_CALLBACK_FAILED");
          outcome = RunTermination::kFailed;
          break;
        }
        // Consume signals that preceded this readiness scan. A concurrent signal remains as a
        // durable token, so the timed wait below cannot miss the scan-to-wait transition.
        RuntimeWakeState& wake = ready.wake != nullptr ? *ready.wake : readiness_wake_;
        static_cast<void>(wake.drain());
        count_phase(RuntimePhaseCounter::kReadinessScans);
        count_phase(RuntimePhaseCounter::kPartitionRunnerSweeps);
        if (!freeze_ready_candidates(exec, ready, ready.inline_execution_lane)) {
          outcome = RunTermination::kFailed;
          break;
        }
        if (ready.dispatcher->active_count() != 0U) {
          outcome = dispatch_ready(exec, ready);
          if (outcome != RunTermination::kCompleted) {
            break;
          }
          count_phase(RuntimePhaseCounter::kPartitionTailDispatchAttempts);
          count_phase(ready.direct_continuation_receipt
                          ? RuntimePhaseCounter::kPartitionTailDispatchAccepted
                          : RuntimePhaseCounter::kPartitionTailDispatchFallbacks);
          continue;
        }
        if (all_sources_closed(ready)) {
          break;
        }
        count_phase(RuntimePhaseCounter::kNoProgressScans);
        if (!wait_for_source_progress(exec, ready)) {
          outcome = RunTermination::kFailed;
        }
      }
    } else if (outcome == RunTermination::kCompleted) {
      PartitionOwnerGroup group;
      std::vector<PartitionOwnerContext> contexts;
      std::vector<ExecutionLaneOwnerTask> tasks;
      contexts.reserve(ready_states.size());
      tasks.reserve(ready_states.size());
      for (ReadyRunState& ready : ready_states) {
        contexts.push_back(PartitionOwnerContext{
            .session = this,
            .exec = &exec,
            .ready = &ready,
            .group = &group,
        });
      }
      for (PartitionOwnerContext& context : contexts) {
        tasks.push_back(ExecutionLaneOwnerTask{
            .execution_lane = context.ready->runner->execution_lane,
            .context = &context,
            .invoke = &invoke_partition_owner,
        });
      }
      while (outcome == RunTermination::kCompleted) {
        group.failed.store(false, std::memory_order_release);
        for (PartitionOwnerContext& context : contexts) {
          context.outcome = RunTermination::kCompleted;
        }
        auto launched = shared_ready.lanes->launch_owner_tasks(tasks);
        auto waited = launched ? shared_ready.lanes->wait_owner_tasks()
                               : expected<void, Error>{make_unexpected(launched.error())};
        if (!waited || group.failed.load(std::memory_order_acquire)) {
          if (!waited) {
            append_diagnostic(
                registered_token_or(waited.error().message, "RUNTIME_EXECUTION_LANE_PLAN_INVALID"));
          }
          outcome = RunTermination::kFailed;
          break;
        }
        if (stop_requested_.load(std::memory_order_acquire) ||
            std::ranges::any_of(contexts, [](const PartitionOwnerContext& context) {
              return context.outcome == RunTermination::kStopped;
            })) {
          outcome = RunTermination::kStopped;
          break;
        }
        if (!lifecycle_control_->has_pending()) {
          break;
        }
        count_phase(RuntimePhaseCounter::kPartitionControlBarriers);
        if (!process_pending_restarts(exec, ready_states)) {
          append_diagnostic("RUNTIME_LIFECYCLE_CALLBACK_FAILED");
          outcome = RunTermination::kFailed;
          break;
        }
      }
    }
    for (ReadyRunState& ready : ready_states) {
      if (!cancel_ready_sources(ready)) {
        append_diagnostic("RUNTIME_READINESS_BINDING_UNAVAILABLE");
        outcome = RunTermination::kFailed;
      }
    }
    if (ready_states.size() > 1U && shared_ready.notifications != nullptr) {
      shared_ready.notifications->close();
    }
    if (!stop_channels(shared_ready)) {
      outcome = RunTermination::kFailed;
    }
    if (tensor_resources_ != nullptr) {
      tensor_resources_->close();
    }
    if (!exec.lifecycle_shutdown()) {
      append_diagnostic(exec.lifecycle_failure_token());
      outcome = RunTermination::kFailed;
    }
    if (!stop_lanes(shared_ready)) {
      outcome = RunTermination::kFailed;
    }
    if (outcome == RunTermination::kCompleted && stop_requested_.load(std::memory_order_acquire)) {
      outcome = RunTermination::kStopped;
    }
    return outcome;
  }

  void finish(RunTermination outcome) noexcept {
    if (lifecycle_control_ != nullptr) {
      // worker_main() reaches finish only after worker_main_impl() has unwound
      // every callback, readiness binding, and lane owner. The same idempotent
      // finalizer closes a worker-construction failure before its terminal
      // session can mint a still-starting lifecycle capability.
      lifecycle_control_->finalize_close();
    }
    if (plan_ != nullptr && plan_->runtime_diagnostic_sink != nullptr) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        plan_->runtime_diagnostic_sink->publish(diagnostics());
        plan_->runtime_diagnostic_sink->publish_runtime_details(diagnostic_buffer_.details());
        plan_->runtime_diagnostic_sink->publish_lane_startup_evidence(lane_startup_evidence_);
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (...) {
        // Observability is subordinate to execution. A private sink cannot
        // replace or weaken the already-determined cleanup and run outcome.
        static_cast<void>(0);
      }
#endif
    }
    termination_.store(outcome, std::memory_order_release);
    if (outcome == RunTermination::kFailed) {
      publish_terminal_summary(
          run_generation_,
          diagnostics(),
          plan_ != nullptr ? plan_->executable_factories.terminal_summary_sink.get() : nullptr);
    }
  }

  std::shared_ptr<const ExecutionPlanState> plan_;
  RunGenerationId run_generation_{};
  ProcessGroupId process_group_{};
  std::shared_ptr<LifecycleControlState> lifecycle_control_;
  std::unique_ptr<GraphExec> graph_exec_;
  std::unique_ptr<RuntimeTensorResources> tensor_resources_;
  std::atomic_bool stop_requested_{false};
  std::atomic<RunTermination> termination_{RunTermination::kRunning};
  std::thread worker_;
  RuntimeWakeState readiness_wake_;
  std::vector<std::unique_ptr<RuntimeWakeState>> partition_wakes_;
  // Cold-built dense routing avoids waking unrelated partition owners for one direct-retained
  // edge. Index zero remains invalid because VertexId is one-based.
  RuntimePartitionWakeRouter partition_wake_by_vertex_;
  std::size_t local_partition_runner_count_{};
  RuntimePhaseCounters* phase_counters_{};
  expected<void, Error> (*compute_lane_invoke_)(void*) = &invoke_compute_lane<false>;
  mutable std::mutex diagnostic_mutex_;
  mutable std::mutex clock_health_mutex_;
  mutable std::mutex join_mutex_;
  std::condition_variable join_cv_;
  bool joining_{};
  bool joined_{};
  bool shared_terminal_{};
  RuntimeDiagnosticBuffer diagnostic_buffer_;
  std::vector<ExecutionLaneStartupEvidence> lane_startup_evidence_;
  mutable std::mutex parameter_mutex_;
  std::condition_variable parameter_cv_;
  ParamControlLink* parameter_control_link_{};
  ParameterUpdateCoordinator* parameter_updates_{};
  std::atomic_bool parameter_ingress_close_requested_{};
};

// Construct the bounded terminal fallback before application code can inject
// or encounter a cold allocation failure. Every wrapper that observes it is
// immutable and release_owner() deliberately leaves this process-lifetime
// tombstone intact.
RunSessionState* const kStartupFailureSessionState = RunSessionState::shared_startup_failure();

}  // namespace detail

namespace detail {

/// Construct the public owner without making a free-function declaration in
/// run_session.hpp. The sole installed run_async declaration remains exported
/// from run.hpp, which avoids conflicting DLL linkage on MSVC.
class RunSessionAccess {
 public:
  [[nodiscard]] static RunSession make(RunSessionState* state) noexcept {
    return RunSession{state};
  }
};

[[nodiscard]] RunSession start_run_session(const ExecutionPlan& plan,
                                           std::optional<ProcessGroupId> process_group) {
  plan.require_no_errors();
  const auto retained = ExecutionPlanAccess::retain(plan);
  const ProcessGroupResolution resolution = resolve_process_group(*retained, process_group);
  RunGenerationId attempted_run_generation{};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    if (resolution.disposition == ProcessGroupResolutionDisposition::kSelectionRequired) {
      attempted_run_generation = reserve_run_generation();
      auto state = std::make_unique<RunSessionState>(
          retained, attempted_run_generation, RunSessionState::ProcessGroupSelectionFailureTag{});
      return RunSessionAccess::make(state.release());
    }
    if (resolution.disposition == ProcessGroupResolutionDisposition::kInvalid) {
      kStartupFailureSessionState->publish_startup_failure(retained);
      return RunSessionAccess::make(kStartupFailureSessionState);
    }
    attempted_run_generation = reserve_run_generation();
    auto state = std::make_unique<RunSessionState>(
        retained, attempted_run_generation, resolution.process_group);
    state->start();
    return RunSessionAccess::make(state.release());
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (...) {
    kStartupFailureSessionState->publish_startup_failure(retained, attempted_run_generation);
    return RunSessionAccess::make(kStartupFailureSessionState);
  }
#endif
}

RunSession run_async_worker(const ExecutionPlan& plan, ProcessGroupId process_group) {
  return start_run_session(plan, process_group);
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

std::span<const Diagnostic> RunSession::diagnostics() const noexcept {
  return state_ != nullptr ? state_->diagnostics() : std::span<const Diagnostic>{};
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

RunSession run_async(const ExecutionPlan& plan) {
  return detail::start_run_session(plan, std::nullopt);
}

int run(const ExecutionPlan& plan) {
  RunSession session = run_async(plan);
  session.wait();
  return session.termination() == RunTermination::kCompleted ? 0 : 1;
}

}  // namespace holoscan
