/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <new>
#include <system_error>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/run_session.hpp>

namespace holoscan {

/// @brief Attempt to execute one accepted plan synchronously through the session runtime.
/// @param plan Immutable accepted plan to execute.
/// @return Zero for clean completion; nonzero for stopped or failed execution.
/// A compiler-accepted empty Graph is a valid zero-work plan and returns zero.
/// @throws CompileError when \p plan is a default-constructed empty plan handle
/// or compile admission rejected it.
/// @throws std::system_error if the platform thread join operation fails.
/// @note This is the blocking wrapper over the exact path used by `run_async()`.
/// An accepted plan whose required runtime bindings are unavailable terminates
/// with failure before invoking user callbacks. Cold run-generation or
/// session-storage exhaustion likewise returns nonzero through the terminal
/// failed-session path rather than escaping as a startup exception.
[[nodiscard]] HOLOSCAN_CORE_EXPORT int run(const ExecutionPlan& plan);

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
[[nodiscard]] HOLOSCAN_CORE_EXPORT RunSession run_async(const ExecutionPlan& plan);

}  // namespace holoscan
