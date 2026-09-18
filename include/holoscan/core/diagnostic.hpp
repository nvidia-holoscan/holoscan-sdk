/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/port.hpp>

namespace holoscan {

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

}  // namespace holoscan
