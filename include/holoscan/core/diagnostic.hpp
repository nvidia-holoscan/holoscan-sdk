/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/port.hpp>

namespace holoscan {

/// @brief Bound free-form diagnostic detail copied from one runtime failure.
inline constexpr std::size_t kDiagnosticMessageCapacity = 4096U;

class ExecutionPlan;

class HOLOSCAN_CORE_EXPORT ExecutionGraphView;

/// @brief Identify one stable Core structural diagnostic definition.
///
/// Values are append-only within the Core diagnostic registry. Zero denotes an
/// unspecified identity and never aliases a registered diagnostic.
struct DiagnosticId {
  std::uint32_t value{};  ///< Registry-assigned numeric identity; zero is unspecified.

  /// @brief Return whether this identity is registered and nonzero.
  /// @return \c true when \ref value is nonzero.
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value != 0U; }

  friend constexpr bool operator==(DiagnosticId, DiagnosticId) noexcept = default;
};

/// @brief Classify one structural diagnostic's compile-admission or runtime effect.
enum class DiagnosticSeverity : std::uint8_t {
  kWarning = 0,  ///< Preserve evidence without rejecting admission or failing a run.
  kError = 1,    ///< Reject compile admission or fail a run while retaining evidence.
};

/// @brief Identify the bounded plan location associated with one diagnostic.
///
/// Every identity is interpreted only within the enclosing plan or the
/// RunSession that retains that plan. Zero fields are absent. The record
/// contains no authoring address, backend handle, or arbitrary source text.
struct DiagnosticSource {
  GraphGeneration graph_generation{};  ///< Frozen source Graph generation, or zero.
  GraphId graph{};                     ///< Canonical Graph scope, or zero.
  VertexId vertex{};                   ///< Canonical logical vertex, or zero.
  PortId port{};                       ///< Dense plan-local port, or invalid.
};

/// @brief Identify the operation that produced supplemental failure evidence.
enum class DiagnosticOperation : std::uint8_t {
  kUnknown = 0,  ///< No operation was captured.
  kCompose,      ///< Graph composition or connection validation.
  kConstruct,    ///< Operator or service construction.
  kSetup,        ///< Operator setup declaration.
  kContract,     ///< Operator contract replay.
  kLifecycle,    ///< A lifecycle callback or barrier.
  kCompute,      ///< An Operator compute callback.
  kRuntime,      ///< A Runtime service operation.
  kCompile,      ///< Compile admission or physical lowering.
};

/// @brief Identify the relationship of a failure to the initiating transition.
enum class DiagnosticFailureRole : std::uint8_t {
  kPrimary = 0,  ///< Initiating failure.
  kRollback,     ///< Failure while unwinding a failed forward transition.
  kRestoration,  ///< Failure while restoring the previous generation.
  kTeardown,     ///< Failure during shutdown or pre-restart teardown.
};

/// @brief Describe one incompatible field without parsing a rendered message.
struct DiagnosticMismatch {
  std::string field;     ///< Constraint name, including an axis or schema field when applicable.
  std::string provided;  ///< Producer or candidate value.
  std::string required;  ///< Consumer or required value.
};

/// @brief Own bounded supplemental evidence for one diagnostic.
/// @note Supplemental detail is not part of canonical execution-plan JSON or its digest.
/// Core capture retains at most 4096 bytes of cause text and four cause entries; truncation is
/// explicit. Messages must not contain secrets. Rendering escapes terminal control characters.
struct DiagnosticDetail {
  DiagnosticOperation operation{DiagnosticOperation::kUnknown};  ///< Failing operation.
  /// Original code/message. Empty kSuccess denotes context without an underlying Error value.
  Error error{ErrorCode::kFailure};
  std::optional<LifecycleStage> stage;  ///< Lifecycle stage when applicable.
  LifecycleTransitionPhase phase{LifecycleTransitionPhase::kNone};  ///< Transition phase.
  LifecycleTransactionId
      transaction_id{};  ///< Owning restart transaction, or zero outside restart.
  DiagnosticFailureRole role{DiagnosticFailureRole::kPrimary};  ///< Primary or cleanup failure.
  std::string operator_path;                   ///< Owning qualified operator or component path.
  std::string port_name;                       ///< Local port name when applicable.
  DiagnosticSource related_source{};           ///< Other endpoint, interpreted in the same plan.
  std::string related_operator_path;           ///< Qualified other endpoint owner.
  std::string related_port_name;               ///< Local other endpoint port.
  std::vector<DiagnosticMismatch> mismatches;  ///< Bounded incompatible field evidence.
  std::vector<Error> causes;  ///< Nested causes, ordered outermost first after the primary error.
  std::optional<std::uint32_t> primary_diagnostic_index;  ///< Earlier zero-based report record.
  bool truncated{};  ///< Some supplemental evidence could not be retained.
};

/// @brief Report one stable bounded compile or runtime result.
///
/// The token is a registry spelling, not free-form log prose. Core producers
/// bound it before publishing immutable evidence. Compile structural records are
/// plan-owned and serialized deterministically; runtime records are bounded
/// and owned by one RunSession. Supplemental detail is owning, non-semantic evidence.
struct Diagnostic {
  DiagnosticId id{};                                        ///< Stable registry identity.
  DiagnosticSeverity severity{DiagnosticSeverity::kError};  ///< Admission or run-outcome effect.
  std::string token;                                        ///< Owned stable registry token.
  DiagnosticSource source{};                                ///< Optional bounded plan-local source.
  std::string service;       ///< Stable service identity when the diagnostic is catalog-scoped.
  std::string endpoint;      ///< Stable endpoint identity when the diagnostic is endpoint-scoped.
  std::string json_pointer;  ///< RFC 6901 location in the normalized diagnostic document.
  std::string required;      ///< Canonical required-side summary when applicable.
  std::string provided;      ///< Canonical candidate-side summary when applicable.
  std::optional<DiagnosticId> cause_id;    ///< Earlier report record that caused this result.
  std::optional<DiagnosticDetail> detail;  ///< Owning supplemental cause and location evidence.
};

/// @brief Format one structural diagnostic for human-readable output.
/// @param diagnostic Structured diagnostic to render.
/// @param graph Immutable compiled graph from the diagnostic's plan, used to resolve primary
/// and related vertices and ports. A default-constructed view retains numeric identities and
/// available owned names without graph lookup.
/// @return One human-readable line without a trailing newline.
/// @note Source vertex IDs are canonical plan IDs, so an authoring GraphView is not accepted.
/// Related endpoint names prefer owned detail over compiled names. An owned primary port name
/// matching its compiled name appears once; when both names are nonempty and differ, the owned
/// name uses captured_port_name.
/// @note Preserve and inspect the structured fields when program logic depends on diagnostics.
/// The rendered text is not a stable machine-readable format.
[[nodiscard]] HOLOSCAN_CORE_EXPORT std::string format_diagnostic(const Diagnostic& diagnostic,
                                                                 const ExecutionGraphView& graph);

/// @brief Format an ordered diagnostic report with a bounded output size.
/// @param diagnostics Borrowed diagnostic records. Errors render before warnings, preserving
/// publication order within each group. Each rendered record includes its original zero-based
/// diagnostic_index so causal references into the borrowed span remain traceable.
/// @param graph Immutable compiled graph used to resolve structural locations.
/// @return Human-readable report capped at 16384 bytes, with explicit truncation when necessary.
[[nodiscard]] HOLOSCAN_CORE_EXPORT std::string format_diagnostics(
    std::span<const Diagnostic> diagnostics, const ExecutionGraphView& graph);

/// @brief Log a plan's retained warnings and errors on explicit caller request.
/// @param plan Plan to report, including rejected plans and accepted plans with warnings.
/// @note This synchronous best-effort helper never throws, changes admission, or installs
/// future reporting. It ignores automatic-report policy, including its environment override.
/// Normal logger filtering applies. Empty diagnostics produce no output. Repeated calls may
/// repeat output. Call outside the plan.ok() failure branch to include accepted-plan warnings.
HOLOSCAN_CORE_EXPORT void log_diagnostics(const ExecutionPlan& plan) noexcept;

/// @brief Log supplied diagnostics at ERROR and WARN through the SDK logger.
/// @param diagnostics Borrowed records, read only for the duration of this call.
/// @param graph Compiled graph from the records' plan, or an empty view for numeric fallbacks.
/// @note Errors precede warnings; original indices and causal references are preserved. One
/// 16384-byte text budget covers both severities, excluding logger-owned prefixes. Filtered
/// severities skip formatting. Rendering and logging failures are contained. This helper does
/// not wait, drain records, subscribe, deduplicate, or initialize automatic-report policy.
/// Wait for a retained RunSession before calling when complete terminal diagnostics are needed.
HOLOSCAN_CORE_EXPORT void log_diagnostics(std::span<const Diagnostic> diagnostics,
                                          const ExecutionGraphView& graph) noexcept;

/// @brief Set the process-wide preference for existing automatic Core warning/error reports.
/// @param enabled Whether future automatic reports are enabled; the initial preference is true.
/// @note Calling with true at startup is normally unnecessary. This setting never prints saved
/// diagnostics, replays a report, or makes compile() dump its returned diagnostics. False does
/// not suppress explicit log_diagnostics() calls, user logging, or structured diagnostic sinks.
/// The first policy access snapshots HOLOSCAN_AUTOMATIC_DIAGNOSTIC_LOGGING: case-insensitive
/// ON/OFF overrides this preference; unset/empty does not. Invalid values generate at most one
/// bounded normal warning. Later environment changes are ignored. Access is thread-safe and never
/// throws. Fatal-invariant reports before process termination remain independent of this policy.
HOLOSCAN_CORE_EXPORT void set_automatic_diagnostic_logging_enabled(bool enabled) noexcept;

/// @brief Return the effective automatic-report policy before normal logger filtering.
/// @return Environment override when valid, otherwise the current API preference (initially true).
/// @note Explicit helpers are independent. Automatic reports sample the policy at each attempt;
/// changing it can affect later failure reports from already-running sessions. Access never throws.
[[nodiscard]] HOLOSCAN_CORE_EXPORT bool automatic_diagnostic_logging_enabled() noexcept;

}  // namespace holoscan
