/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */
#pragma once

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace holoscan::ops::inference_eligibility {

/**
 * @brief Returns true when any tensor produced by some model (present as a
 * value in `output_map`) is consumed as an input by some other model
 * (present as a value in `input_map`). Chained models cannot use the
 * multi-model fast paths — kSeqFast iterates in std::map key order, and
 * kParFast fans out with no cross-model waits.
 */
inline bool has_chained_models(const std::map<std::string, std::vector<std::string>>& input_map,
                               const std::map<std::string, std::vector<std::string>>& output_map) {
  std::set<std::string> produced;
  for (const auto& [_, outs] : output_map) {
    for (const auto& t : outs) {
      produced.insert(t);
    }
  }
  for (const auto& [_, ins] : input_map) {
    for (const auto& t : ins) {
      if (produced.count(t) != 0U) {
        return true;
      }
    }
  }
  return false;
}

/**
 * @brief Seven-state dispatch enum. Selected automatically at InferenceOp::start() from
 * the operator's existing configuration (backend, model count, parallel_inference,
 * dynamic_input_dims). Fast paths are enabled ONLY when every model uses the "trt" backend
 * and no static blocker (see select_dispatch_kind()) is present.
 *   STATIC-shape fast paths:
 *     - kFast       : single-model, static shape
 *     - kSeqFast    : multi-model, static shape, sequential
 *     - kParFast    : multi-model, static shape, parallel
 *   DYNAMIC-shape fast paths (chosen when dynamic_input_dims is true):
 *     - kDynFast    : single-model, dynamic shape
 *     - kDynSeqFast : multi-model, dynamic shape, sequential
 *     - kDynParFast : multi-model, dynamic shape, parallel
 *   Fallback:
 *     - kStandard   : canonical execute_inference (handles non-TRT backends and
 *                     any static blocker)
 */
enum class DispatchKind {
  kFast,
  kSeqFast,
  kParFast,
  kDynFast,
  kDynSeqFast,
  kDynParFast,
  kStandard
};

/// @brief Result of select_dispatch_kind(): the chosen DispatchKind plus, for the
/// single-model fast paths (kFast / kDynFast), the name of the cached model.
struct DispatchSelection {
  DispatchKind kind = DispatchKind::kStandard;
  std::string fast_single_model_name;
};

/**
 * @brief Pure, unit-testable dispatch-kind selector.
 *
 * Static blockers (any one forces kStandard):
 *  - `!input_on_cuda`: the fast paths (do_inference_on_stream) do not stage host<->device
 *    input buffers.
 *  - chained models: any tensor produced by one model (output_map value) consumed as
 *    another model's input (input_map value) — see has_chained_models().
 *
 * @param model_path_map Model name -> model path.
 * @param input_map Model name -> input tensor names.
 * @param output_map Model name -> output tensor names.
 * @param backend Scalar backend string (used when backend_map is empty).
 * @param backend_map Optional per-model backend override map.
 * @param input_on_cuda Whether input tensors are device-resident.
 * @param dynamic_input_dims Whether input dimensions may change between compute() calls.
 * @param parallel_inference Whether multi-model configurations should run in parallel.
 * @return The selected DispatchKind and, for single-model fast paths, the cached model name.
 */
inline DispatchSelection select_dispatch_kind(
    const std::map<std::string, std::string>& model_path_map,
    const std::map<std::string, std::vector<std::string>>& input_map,
    const std::map<std::string, std::vector<std::string>>& output_map, const std::string& backend,
    const std::map<std::string, std::string>& backend_map, bool input_on_cuda,
    bool dynamic_input_dims, bool parallel_inference) {
  DispatchSelection sel;

  const bool needs_host_input = !input_on_cuda;
  const bool has_chained = has_chained_models(input_map, output_map);
  const bool static_blockers = needs_host_input || has_chained;

  bool all_trt = false;
  if (!backend_map.empty()) {
    all_trt = true;
    for (const auto& [_, b] : backend_map) {
      if (b != "trt") {
        all_trt = false;
        break;
      }
    }
  } else {
    all_trt = (backend == "trt");
  }

  const size_t n_models = model_path_map.size();
  const bool eligible = all_trt && !static_blockers && n_models >= 1;

  if (!eligible) {
    sel.kind = DispatchKind::kStandard;
  } else if (n_models == 1) {
    sel.fast_single_model_name = model_path_map.begin()->first;
    sel.kind = dynamic_input_dims ? DispatchKind::kDynFast : DispatchKind::kFast;
  } else if (parallel_inference) {
    sel.kind = dynamic_input_dims ? DispatchKind::kDynParFast : DispatchKind::kParFast;
  } else {
    sel.kind = dynamic_input_dims ? DispatchKind::kDynSeqFast : DispatchKind::kSeqFast;
  }

  return sel;
}

/**
 * @brief Demotes kParFast/kDynParFast to kSeqFast/kDynSeqFast when the configured
 * `cuda_stream_pool` cannot supply `n_models` distinct CUDA streams (pool absent, or a
 * bounded `max_size` smaller than `n_models`). All other DispatchKind values pass through
 * unchanged.
 *
 * @param kind The DispatchKind selected by select_dispatch_kind().
 * @param pool_present Whether InferenceOpParams::cuda_stream_pool is non-null.
 * @param pool_max_size The pool's configured max_size (0 means unlimited).
 * @param n_models Number of configured models.
 * @return `kind`, or the sequential-fast-path demotion of `kind` when the pool is insufficient.
 */
inline DispatchKind demote_if_stream_pool_insufficient(DispatchKind kind, bool pool_present,
                                                       uint32_t pool_max_size, size_t n_models) {
  if (kind != DispatchKind::kParFast && kind != DispatchKind::kDynParFast) {
    return kind;
  }
  const bool insufficient = !pool_present || (pool_max_size != 0 && pool_max_size < n_models);
  if (!insufficient) {
    return kind;
  }
  return kind == DispatchKind::kDynParFast ? DispatchKind::kDynSeqFast : DispatchKind::kSeqFast;
}

/**
 * @brief Outcome of decide_capture_protection() for kParFast/kDynParFast + CUDA Graphs.
 *
 * CUDA-Graph capture under TRT's per-model-stream fast paths is process-context scoped:
 * concurrent stream capture from multiple InferenceOp instances (multiple ManagerInfer /
 * TrtInfer contexts) sharing the same CUDA context can invalidate an in-progress capture on
 * another stream. There are two durable ways to make this safe, and one combination that is
 * NOT durable on its own:
 *   - kGreenContextIsolated : a CudaGreenContextPool partitions SMs into separate CUDA
 *     contexts, so concurrent capture on different partitions cannot cross-invalidate.
 *   - kMutexProtected       : runtime mutex-based serialization of concurrent CUDA-Graph
 *     capture. (Internal implementation note: for static shapes (kParFast), capture is first
 *     serialized single-threaded at setup time via an internal warm-up pass; ManagerInfer's
 *     capture_serialization_mutex_ is the durable runtime safety net, held for the entire
 *     per-model dispatch loop — not just the host-side enqueue call — so it cannot be released
 *     before TRT's synchronous capture-end/instantiate sequence completes.)
 *   - kRefuse               : kDynParFast has no static shapes for the setup-time warm-up path;
 *     without a CudaGreenContextPool there is no durable protection. Return kRefuse as a
 *     signal for the fallback ladder (fallback_ladder.hpp) — NOT an actual refusal to start.
 *     The ladder always degrades past kRefuse to a runnable configuration (drop CUDA graphs,
 *     demote to a sequential fast path, or fall through to kStandard).
 */
enum class CaptureProtectionOutcome {
  kNotApplicable,
  kGreenContextIsolated,
  kMutexProtected,
  kRefuse
};

/// @brief Result of decide_capture_protection()
struct CaptureProtectionDecision {
  CaptureProtectionOutcome outcome = CaptureProtectionOutcome::kNotApplicable;
  std::string message;
};

/**
 * @brief Pure, unit-testable CUDA-Graph capture-protection decision for kParFast/kDynParFast.
 *
 * @param kind Final DispatchKind (after demote_if_stream_pool_insufficient()).
 * @param enable_cuda_graphs Whether TensorRT CUDA-Graph capture is enabled.
 * @param green_context_pool_present Whether InferenceOpParams::cuda_green_context_pool is
 *   non-null.
 * @param green_context_num_partitions Number of materialized partitions in the pool (only
 *   used for the log message; ignored when the pool is absent).
 * @return The chosen protection outcome and its associated message.
 */
inline CaptureProtectionDecision decide_capture_protection(DispatchKind kind,
                                                           bool enable_cuda_graphs,
                                                           bool green_context_pool_present,
                                                           uint32_t green_context_num_partitions) {
  CaptureProtectionDecision decision;

  const bool needs_protection =
      (kind == DispatchKind::kParFast || kind == DispatchKind::kDynParFast) && enable_cuda_graphs;
  if (!needs_protection) {
    return decision;
  }

  if (green_context_pool_present) {
    decision.outcome = CaptureProtectionOutcome::kGreenContextIsolated;
    decision.message = "Concurrent CUDA-Graph capture protected by CudaGreenContextPool (" +
                       std::to_string(green_context_num_partitions) + " partitions).";
    return decision;
  }

  if (kind == DispatchKind::kParFast) {
    decision.outcome = CaptureProtectionOutcome::kMutexProtected;
    decision.message =
        "Concurrent CUDA-Graph capture protected by runtime mutex-based serialization "
        "(static shapes).";
    return decision;
  }

  // kDynParFast: dynamic shapes have no setup-time warm-up path, and no green context was
  // supplied. There is no durable protection available. Return kRefuse as a
  // signal for the fallback ladder to degrade further (dropping CUDA graphs or demoting the
  // dispatch kind); no exception is raised and start() still completes with a runnable
  // configuration.
  decision.outcome = CaptureProtectionOutcome::kRefuse;
  decision.message =
      "kDynParFast with enable_cuda_graphs=true requires either static input shapes (for "
      "mutex-based capture serialization) OR a CudaGreenContextPool for context isolation. "
      "Neither is supplied. Set enable_cuda_graphs=false, make dims static, or provide a "
      "CudaGreenContextPool.";
  return decision;
}

}  // namespace holoscan::ops::inference_eligibility
