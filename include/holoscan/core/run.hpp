/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <new>
#include <system_error>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/run_session.hpp>

namespace holoscan {

class ManagedShutdownSignals;

/// @brief Select whether a blocking run uses HSDK-managed process signals.
///
/// This policy affects only the blocking `run(plan, policy)` overload. It does not change
/// `run_async(plan)`, bind an existing session, or transfer process-signal ownership.
enum class RunSignalPolicy : std::uint8_t {
  /// On Linux, adopt or safely establish the sole managed SIGINT/SIGTERM domain. Admission fails
  /// before session creation when neither route is safe. On non-Linux, use ordinary execution.
  /// The stable underlying value is zero.
  kManaged = 0U,
  /// Bypass all managed-signal ownership, census, mask, descriptor, and diagnostic work and leave
  /// process-signal policy entirely to the caller. The stable underlying value is one.
  kCallerManaged = 1U
};

/// @brief Attempt to execute one accepted plan synchronously through the session runtime.
/// @param plan Immutable accepted plan to execute.
/// @return Zero for natural or explicitly requested successful completion; nonzero for stopped,
/// failed, or rejected managed-signal admission.
/// A compiler-accepted empty Graph is a valid zero-work plan and returns zero.
/// An Operator-requested completion returns zero even though it does not guarantee EOS, source
/// closure, downstream drain, or queued-sample delivery.
/// On Linux, a first managed SIGINT or SIGTERM requests stopped execution and therefore returns
/// nonzero after cleanup.
/// @throws CompileError when \p plan is a default-constructed empty plan handle
/// or compile admission rejected it.
/// @throws std::system_error if the platform thread join operation fails.
/// @note This is the blocking wrapper over the exact path used by `run_async()`.
/// On Linux it is equivalent to `run(plan, RunSignalPolicy::kManaged)`. A compatible early
/// `ManagedShutdownSignals` domain is adopted when present; otherwise transient admission requires
/// both process-task census reads to observe exactly one live task. Census failure, change, or any
/// additional live task rejects admission before session creation. Adopting an explicit domain
/// consumes its sole binding attempt; after retirement, close it and create a fresh domain before
/// another managed run. Only one process-wide managed binding can be active.
/// An accepted plan whose required runtime bindings are unavailable terminates
/// with failure before invoking user callbacks. Cold run-generation or
/// session-storage exhaustion likewise returns nonzero through the terminal
/// failed-session path rather than escaping as a startup exception.
/// @warning This call blocks through cooperative callback retirement, cleanup, and worker join.
/// No whole-run or forced-cancellation time bound is provided.
[[nodiscard]] HOLOSCAN_CORE_EXPORT int run(const ExecutionPlan& plan);

/// @brief Execute one accepted plan synchronously with an explicit signal policy.
/// @param plan Immutable accepted plan to execute.
/// @param signal_policy Managed Linux admission or an explicit caller-managed opt-out.
/// @return Zero for natural or requested successful completion; nonzero for stopped, failed, or
/// failed managed-domain admission.
/// @throws CompileError when \p plan is a default-constructed empty plan handle or compile
/// admission rejected it.
/// @throws std::system_error if the platform thread join operation fails.
/// @note `kCallerManaged` performs no HSDK-managed signal ownership, census, mask, descriptor, or
/// diagnostic work.
/// The application disposition is authoritative and may terminate the process without HSDK
/// cleanup. On non-Linux platforms both policies use the ordinary blocking execution path.
/// Concurrent managed calls compete for one process-wide binding; a losing call returns nonzero
/// before compute admission. This function may allocate and start Runtime worker threads.
/// An unsupported enum value returns nonzero before plan, session, or managed-signal admission and
/// produces no managed-signal diagnostic or process-signal side effect.
/// @warning This call can wait indefinitely for uncooperative application callbacks or external
/// work; selecting managed signals does not add forced cancellation.
[[nodiscard]] HOLOSCAN_CORE_EXPORT int run(const ExecutionPlan& plan,
                                           RunSignalPolicy signal_policy);

/// @brief Start one accepted-plan attempt through a strict asynchronous session owner.
/// @param plan Immutable accepted plan whose private executable state is retained.
/// @return Move-only session owning the run generation and all started work.
/// A compiler-accepted empty Graph produces a terminal \c kCompleted zero-work session.
/// @throws CompileError when \p plan is a default-constructed empty plan handle
/// or compile admission rejected it.
/// @note The source Graph and caller's plan wrapper may be destroyed after return.
/// A platform worker-start failure is reported by a terminal failed session.
/// Missing runtime bindings, run-generation exhaustion, and cold session-storage
/// exhaustion likewise produce a terminal failed session before any user callback
/// rather than escaping synchronously or selecting weaker execution semantics.
/// Calling this overload for a plan spanning multiple process groups produces
/// a terminal failed session with `RUNTIME_PROCESS_GROUP_SELECTION_REQUIRED`;
/// this public API does not launch or select per-group workers.
/// Process signals remain entirely caller-managed; this overload never binds an HSDK-managed
/// stop target. The call may allocate and start a worker thread before returning, but it does not
/// wait for the run to finish.
[[nodiscard]] HOLOSCAN_CORE_EXPORT RunSession run_async(const ExecutionPlan& plan);

/// @brief Start one session atomically bound to an early managed signal domain.
/// @param plan Immutable accepted plan whose private executable state is retained.
/// @param signals Open creator-thread domain borrowed through this run attempt. It must outlive the
/// returned session, its binding, and every participant thread.
/// @return A strict session owner; binding admission failure is a terminal failed session.
/// @throws CompileError when \p plan is a default-constructed empty plan handle or compile
/// admission rejected it.
/// @note Call this overload only on the domain's creator thread. Binding is serialized and occurs
/// before the session starts; once reserved, it consumes the domain's sole binding attempt.
/// Wrong-thread, closed, spent, competing-owner, or otherwise incompatible binding admission
/// returns an already-terminal failed session with bounded diagnostics. Plain `run_async(plan)`
/// remains caller-managed. This call may allocate and start a worker thread before returning, but
/// it does not wait for the run to finish.
/// @warning `signals` is borrowed and `close()` reports `kNotReady` while this binding is active.
/// Managed shutdown remains cooperative and provides no whole-run or forced-cancellation bound.
[[nodiscard]] HOLOSCAN_CORE_EXPORT RunSession run_async(const ExecutionPlan& plan,
                                                        ManagedShutdownSignals& signals);

}  // namespace holoscan
