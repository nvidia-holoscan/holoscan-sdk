/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <type_traits>

namespace holoscan {

/// @brief Represent a fixed-size trace identity; zero is invalid or absent.
using TraceId = std::uint64_t;

/// @brief Represent a fixed-size span identity; zero is invalid or absent.
using SpanId = std::uint64_t;

/// @brief Carry fixed-size distributed-trace identity for one logical operation.
///
/// A context owns no tracer or runtime state. It may be copied into bounded
/// records and does not grant span creation, logging, or export authority.
struct TraceContext {
  TraceId trace_id{};       ///< Nonzero trace identity, or zero when absent.
  SpanId span_id{};         ///< Current span identity, or zero when absent.
  SpanId parent_span_id{};  ///< Parent span identity, or zero for a root.

  /// @brief Return whether the context contains a trace identity.
  /// @return \c true when \ref trace_id is nonzero.
  [[nodiscard]] constexpr bool valid() const noexcept { return trace_id != 0; }

  /// @brief Construct a root trace context from a trace identity.
  /// @param id Trace identity supplied by the caller or admitted runtime.
  /// @return Root context with no current or parent span identity.
  [[nodiscard]] static constexpr TraceContext root(TraceId id) noexcept {
    return TraceContext{.trace_id = id};
  }

  /// @brief Compare trace contexts by all fixed-size identity fields.
  /// @param lhs Left trace context.
  /// @param rhs Right trace context.
  /// @return \c true when every trace and span identity matches.
  friend constexpr bool operator==(TraceContext lhs, TraceContext rhs) noexcept = default;
};

/// @brief Select the Runtime action for a final operator deadline violation.
///
/// The action does not itself publish output. The first two values enter the
/// operator callback exactly once with an \ref ActivationStatus; \c kFailRun
/// terminates the candidate without exposing an activation context.
enum class OnViolation : std::uint8_t {
  kSafeDefault = 0,  ///< Invoke the operator-authored degraded path once.
  kEmitWitness = 1,  ///< Invoke once and record deadline evidence; operator code may emit.
  kFailRun = 2,      ///< Do not invoke compute; begin the fatal run path.
};

/// @brief Describe the realized value state of one declared input for an activation.
enum class InputState : std::uint8_t {
  kSelected = 0,     ///< A fresh selector result is available.
  kReused = 1,       ///< A bounded prior value was selected by \c ReuseLast.
  kSafeDefault = 2,  ///< Missing policy selected the operator-authored safe branch.
  kAbsent = 3,       ///< Missing-policy evaluation completed without a value.
  kNotSelected = 4,  ///< Live declared input was excluded from this activation.
};

/// @brief Explain the frozen temporal or missing-policy result for one input.
///
/// These values describe selection decisions. Input-access guard failures such
/// as an unbound, stale, closed, wrong-cardinality, or backend-materialization
/// error remain typed \c ErrorCode values and are not folded into this enum.
enum class SelectionReason : std::uint8_t {
  kNone = 0,                 ///< No missing or temporal failure affected selection.
  kNeverSeen = 1,            ///< No eligible sample has ever been observed.
  kTooStale = 2,             ///< Available history is older than the admitted bound.
  kInsufficientHistory = 3,  ///< Required bracketing or history is not available.
  kWindowTimeout = 4,        ///< A bounded window closed before a complete selection.
  kWindowSampleLimit = 5,    ///< A finite selected-sample bound prevented completion.
  kWindowPendingLimit = 6,   ///< A finite pending-window bound prevented admission.
  kEndOfStream = 7,          ///< The input closed without another eligible value.
  kNotConnected = 8,         ///< Explicit SafeDefault selected for an admitted unwired Exact input.
  kAlignmentWindowMiss = 9,  ///< An impossible aligned FIFO head was retired.
  kNotSelectedForActivation = 10,  ///< Live input was not a member of this activation.
  kCaptureTimeAbsent = 11,         ///< Event-time selection found canonical absent capture time.
  kClockIncomparable = 12,         ///< Source and reference clock domains are not comparable.
  kConversionUnavailable = 13,     ///< A required admitted direct clock conversion is unavailable.
};

/// @brief Carry the complete frozen selection outcome for one input.
///
/// \c kNotSelected must pair with \c kNotSelectedForActivation. Runtime and
/// synthetic test bindings validate all state/reason combinations before the
/// value becomes visible to operator code.
struct InputSelection {
  InputState state{InputState::kAbsent};  ///< Frozen realized input state.
  SelectionReason reason{
      SelectionReason::kNeverSeen};  ///< Exact selection or missing-policy reason.
};

/// @brief Explain why an admitted operator callback is executing.
enum class ActivationReason : std::uint8_t {
  kNominal = 0,            ///< Candidate entered before its final latest-start deadline.
  kDeadlineExceeded = 1,   ///< Candidate entered under its explicit violation action.
  kInvariantViolated = 2,  ///< Runtime admitted a bounded invariant-recovery callback path.
};

/// @brief Expose the operator-level status already frozen for one callback.
///
/// Reading this value has no witness, logging, scheduling, or publication side
/// effect. Input-specific temporal reasons remain in \ref InputSelection.
struct ActivationStatus {
  ActivationReason reason{ActivationReason::kNominal};  ///< Frozen activation reason.
  OnViolation policy{OnViolation::kEmitWitness};        ///< Compiled violation action.
};

static_assert(std::is_standard_layout_v<TraceContext>);
static_assert(std::is_trivially_copyable_v<TraceContext>);
static_assert(std::is_standard_layout_v<InputSelection>);
static_assert(std::is_trivially_copyable_v<InputSelection>);
static_assert(std::is_standard_layout_v<ActivationStatus>);
static_assert(std::is_trivially_copyable_v<ActivationStatus>);

}  // namespace holoscan
