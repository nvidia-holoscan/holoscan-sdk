/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// INTERNAL HEADER — not installed, not part of the public HoloInfer surface.
//
// Runtime half of the InferenceOp fallback ladder. The start()-time rungs live in the public,
// pure header <holoscan/operators/inference/fallback_ladder.hpp>; the rungs below can only be
// detected while CUDA work is being issued (e.g. a failed cuCtxPushCurrent() meaning
// green-context isolation is not actually in effect, or a CUDA graph capture invalidated during
// a real compute() tick).
//
// Everything here is designed so the steady state costs nothing: the only per-tick check is a
// single relaxed atomic load (see RuntimeFallbackFlags), and per-model "already degraded" state
// is a plain bool vector consulted only after an error has already been returned.

#pragma once

#include <atomic>
#include <string>

#include <holoscan/operators/inference/fallback_ladder.hpp>

namespace holoscan {
namespace inference {

namespace fallback = holoscan::ops::inference_eligibility;
using FallbackReason = fallback::FallbackReason;

/**
 * @brief Process-wide runtime fallback signals raised deep inside a backend (where there is no
 * back-pointer to the owning ManagerInfer) and consumed by the dispatch loops.
 *
 * Currently only: TrtInfer's GreenContextDispatchGuard raises
 * `green_context_push_failed` when cuCtxPushCurrent() fails, and
 * ManagerInfer::run_core_inference_par_fast() reacts by dropping its lock-free
 * green-context-isolated assumption and re-arming the capture-serialization mutex. The flag is
 * process-wide rather than per-instance because the failure is a property of the CUDA context
 * state shared by every InferenceOp in the process; reacting conservatively across instances is
 * the safe direction. Read cost is one relaxed atomic load per tick, and only for instances that
 * believed they were green-context isolated in the first place.
 */
struct RuntimeFallbackFlags {
  std::atomic<bool> green_context_push_failed{false};
};

/// @brief Accessor for the single process-wide RuntimeFallbackFlags instance.
inline RuntimeFallbackFlags& runtime_fallback_flags() {
  static RuntimeFallbackFlags flags;
  return flags;
}

/// @brief Emits a runtime fallback log line in the standard ladder format.
inline void emit_runtime_fallback(FallbackReason reason, fallback::DispatchKind kind,
                                  fallback::CaptureProtectionOutcome from_protection,
                                  fallback::CaptureProtectionOutcome to_protection,
                                  const std::string& reason_detail, const std::string& fix) {
  fallback::emit_fallback_step(fallback::make_runtime_fallback_step(
      reason, kind, from_protection, to_protection, reason_detail, fix));
}

}  // namespace inference
}  // namespace holoscan
