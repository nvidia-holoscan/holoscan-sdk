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
#include <mutex>
#include <string>
#include <vector>

#include <holoscan/logger/logger.hpp>
#include <holoscan/operators/inference/fast_path_eligibility.hpp>

namespace holoscan::ops::inference_eligibility {

/**
 * @brief Every known way a requested InferenceOp dispatch configuration can fail to be
 * honored. Each value maps 1:1 to a documented rung of the fallback ladder (see
 * `modules/inference/README.md`, section "Fallback ladder"). `reason_key()` returns a stable,
 * greppable identifier for each value, used in log messages and the documentation table, so an
 * operator log line can be grepped straight back to the README entry that explains it.
 *
 * Some reasons are resolvable statically at InferenceOp::start()
 * (`resolve_dispatch_with_fallback()`); others can only be detected at setup/dispatch time
 * inside HoloInfer and are handled by the runtime fallback state in
 * `modules/inference/src/manager/fallback_runtime.hpp`.
 */
enum class FallbackReason : uint8_t {
  kNone = 0,
  /// kDynParFast + CUDA graphs, but no CudaGreenContextPool to isolate concurrent capture.
  kDynParFastGraphsNoGreenContext,
  /// The CudaStreamPool cannot supply one distinct stream per model.
  kStreamPoolInsufficient,
  /// A CudaGreenContextPool is present but has fewer partitions than models.
  kGreenContextPartitionsInsufficient,
  /// cuCtxPushCurrent() failed at runtime, so green-context isolation is not actually in
  /// effect for this dispatch.
  kGreenContextPushFailed,
  /// The setup-time CUDA-graph warm-up capture pass failed for a model.
  kCaptureWarmupFailed,
  /// A CUDA-graph capture was invalidated (or otherwise failed) during a real compute() tick.
  kRuntimeCaptureInvalidated,
  /// CUDA graphs were requested but the supplied stream is blocking / the legacy stream, so
  /// the TensorRT backend swapped in an internally-created non-blocking stream.
  kBlockingStreamSwapped,
  /// Chained models (one model consumes another model's output tensor).
  kChainedModels,
  /// Host-resident inputs (`input_on_cuda=false`).
  kHostInputs,
  /// At least one model uses a non-TensorRT backend.
  kNonTrtBackend,
  /// `parallel_inference=true` was requested but a sequential dispatch kind was selected.
  kParallelRequestedSerialSelected,
};

/// @brief Log severity a fallback step is emitted at. Informational degradations (the config is
/// simply not eligible for a fast path) are kInfo; anything the user probably wants to act on is
/// kWarn.
enum class FallbackSeverity : uint8_t { kInfo, kWarn };

/// @brief Stable, greppable key for a FallbackReason. Used verbatim in the emitted log line
/// ("fallback triggered — <REASON_KEY>") and as the anchor in the README ladder table.
inline const char* reason_key(FallbackReason reason) {
  switch (reason) {
    case FallbackReason::kDynParFastGraphsNoGreenContext:
      return "A_DYNPARFAST_GRAPHS_NO_GREEN_CONTEXT";
    case FallbackReason::kStreamPoolInsufficient:
      return "B_STREAM_POOL_INSUFFICIENT";
    case FallbackReason::kGreenContextPartitionsInsufficient:
      return "C_GREEN_CONTEXT_PARTITIONS_INSUFFICIENT";
    case FallbackReason::kGreenContextPushFailed:
      return "D_GREEN_CONTEXT_PUSH_FAILED";
    case FallbackReason::kCaptureWarmupFailed:
      return "E_CAPTURE_WARMUP_FAILED";
    case FallbackReason::kRuntimeCaptureInvalidated:
      return "F_RUNTIME_CAPTURE_INVALIDATED";
    case FallbackReason::kBlockingStreamSwapped:
      return "G_BLOCKING_STREAM_SWAPPED";
    case FallbackReason::kChainedModels:
      return "H_CHAINED_MODELS";
    case FallbackReason::kHostInputs:
      return "I_HOST_INPUTS";
    case FallbackReason::kNonTrtBackend:
      return "J_NON_TRT_BACKEND";
    case FallbackReason::kParallelRequestedSerialSelected:
      return "K_PARALLEL_REQUESTED_SERIAL_SELECTED";
    case FallbackReason::kNone:
    default:
      return "NONE";
  }
}

/// @brief Default severity for a reason: most rungs warn; informational-only rungs (blocking
/// stream swap, host inputs, non-TRT backend) inform.
inline FallbackSeverity default_severity(FallbackReason reason) {
  switch (reason) {
    case FallbackReason::kBlockingStreamSwapped:
    case FallbackReason::kHostInputs:
    case FallbackReason::kNonTrtBackend:
      return FallbackSeverity::kInfo;
    default:
      return FallbackSeverity::kWarn;
  }
}

/// @brief Human-readable DispatchKind name, matching the enum spelling used in the docs.
inline const char* to_string(DispatchKind kind) {
  switch (kind) {
    case DispatchKind::kFast:
      return "kFast";
    case DispatchKind::kSeqFast:
      return "kSeqFast";
    case DispatchKind::kParFast:
      return "kParFast";
    case DispatchKind::kDynFast:
      return "kDynFast";
    case DispatchKind::kDynSeqFast:
      return "kDynSeqFast";
    case DispatchKind::kDynParFast:
      return "kDynParFast";
    case DispatchKind::kStandard:
    default:
      return "kStandard";
  }
}

/// @brief Human-readable capture-protection name.
inline const char* to_string(CaptureProtectionOutcome outcome) {
  switch (outcome) {
    case CaptureProtectionOutcome::kGreenContextIsolated:
      return "green-context-isolation";
    case CaptureProtectionOutcome::kMutexProtected:
      return "mutex-based";
    case CaptureProtectionOutcome::kRefuse:
      return "refuse-to-start";
    case CaptureProtectionOutcome::kNotApplicable:
    default:
      return "no-capture-protection";
  }
}

/// @brief One rung of the ladder: the primary configuration that was rejected, the safe
/// configuration substituted for it, why, and (when one exists) the user-side fix.
struct FallbackStep {
  FallbackReason reason = FallbackReason::kNone;
  DispatchKind from = DispatchKind::kStandard;
  DispatchKind to = DispatchKind::kStandard;
  CaptureProtectionOutcome from_protection = CaptureProtectionOutcome::kNotApplicable;
  CaptureProtectionOutcome to_protection = CaptureProtectionOutcome::kNotApplicable;
  FallbackSeverity severity = FallbackSeverity::kWarn;
  /// One-line explanation of why the primary configuration could not be honored.
  std::string reason_detail;
  /// Suggested user-side fix; empty means "none" (nothing the user can change).
  std::string fix;
};

/// @brief The full outcome of running the ladder: what the config asked for, what will actually
/// run, and every degradation step in between (in the order they were applied).
struct FallbackDecision {
  /// The DispatchKind the configuration would have gotten if nothing had blocked it.
  DispatchKind primary_choice = DispatchKind::kStandard;
  /// The DispatchKind that will actually be used.
  DispatchKind final_choice = DispatchKind::kStandard;
  /// Capture protection implied by `primary_choice` + the requested CUDA-graph setting.
  CaptureProtectionOutcome primary_protection = CaptureProtectionOutcome::kNotApplicable;
  /// Capture protection that will actually be in effect. Never kRefuse — the ladder always
  /// terminates in a runnable configuration.
  CaptureProtectionOutcome final_protection = CaptureProtectionOutcome::kNotApplicable;
  /// Whether CUDA graphs remain enabled after the ladder ran.
  bool enable_cuda_graphs = true;
  /// Cached model name for the single-model fast paths (kFast / kDynFast).
  std::string fast_single_model_name;
  /// Every degradation applied, in order.
  std::vector<FallbackStep> chain;
  /// One-line summary suitable for a single INFO log at the end of start().
  std::string user_message;

  /// @brief True when `reason` appears anywhere in the chain.
  [[nodiscard]] bool has(FallbackReason reason) const {
    for (const auto& step : chain) {
      if (step.reason == reason) {
        return true;
      }
    }
    return false;
  }
};

/// @brief Everything `resolve_dispatch_with_fallback()` needs from InferenceOpParams plus the
/// resolved resource capabilities (stream-pool size, green-context partition count).
struct FallbackContext {
  std::map<std::string, std::string> model_path_map;
  std::map<std::string, std::vector<std::string>> input_map;
  std::map<std::string, std::vector<std::string>> output_map;
  std::string backend{"trt"};
  std::map<std::string, std::string> backend_map;
  bool input_on_cuda = true;
  bool dynamic_input_dims = false;
  bool parallel_inference = true;
  bool enable_cuda_graphs = true;
  bool stream_pool_present = false;
  /// CudaStreamPool::capacity(); 0 means unbounded.
  uint32_t stream_pool_max_size = 0;
  bool green_context_pool_present = false;
  uint32_t green_context_num_partitions = 0;
};

namespace detail {

/// @brief Return true when `kind` is a parallel multi-model fast-path variant.
inline bool is_parallel_kind(DispatchKind kind) {
  return kind == DispatchKind::kParFast || kind == DispatchKind::kDynParFast;
}

/// @brief Join `items` into a comma-separated string, used for compact log formatting.
inline std::string join(const std::vector<std::string>& items) {
  std::string out;
  for (size_t i = 0; i < items.size(); ++i) {
    if (i != 0) {
      out += ", ";
    }
    out += items[i];
  }
  return out;
}

/// Model names whose backend is not "trt".
inline std::vector<std::string> non_trt_models(const std::map<std::string, std::string>& paths,
                                               const std::string& backend,
                                               const std::map<std::string, std::string>& map) {
  std::vector<std::string> offenders;
  if (!map.empty()) {
    for (const auto& [name, b] : map) {
      if (b != "trt") {
        offenders.push_back(name + "=" + b);
      }
    }
    return offenders;
  }
  if (backend != "trt") {
    for (const auto& [name, _] : paths) {
      offenders.push_back(name + "=" + backend);
    }
  }
  return offenders;
}

/// "producer -> consumer(tensor)" descriptions of every chained pair.
inline std::vector<std::string> chained_pairs(
    const std::map<std::string, std::vector<std::string>>& input_map,
    const std::map<std::string, std::vector<std::string>>& output_map) {
  std::map<std::string, std::string> producer_of;
  for (const auto& [model, outs] : output_map) {
    for (const auto& t : outs) {
      producer_of[t] = model;
    }
  }
  std::vector<std::string> pairs;
  for (const auto& [model, ins] : input_map) {
    for (const auto& t : ins) {
      auto it = producer_of.find(t);
      if (it != producer_of.end()) {
        pairs.push_back(it->second + " -> " + model + " (" + t + ")");
      }
    }
  }
  return pairs;
}

}  // namespace detail

/**
 * @brief Runs the full fallback ladder for one InferenceOp configuration.
 *
 * This is the single entry point InferenceOp::start() uses instead of calling
 * select_dispatch_kind() / demote_if_stream_pool_insufficient() / decide_capture_protection()
 * itself: it wraps those three (their logic is not duplicated here) and adds the degradation
 * ladder plus the diagnostics needed to explain each degradation to the user.
 *
 * Guarantees:
 *  - The returned configuration is always runnable: `final_protection` is never kRefuse, and a
 *    configuration that decide_capture_protection() would refuse is degraded instead of
 *    throwing.
 *  - Every degradation is recorded in `chain` in the order it was applied, so the caller can
 *    emit one log block per rung (see emit_fallback_decision()).
 *
 * Rung order (see the README table): the static eligibility blockers (non-TRT backend, host
 * inputs, chained models) -> stream pool insufficiency -> kDynParFast + graphs with no green
 * context -> too few green-context partitions -> a summary warning that parallelism was
 * requested but not obtained.
 *
 * @param ctx Operator configuration plus resolved resource capabilities.
 * @return The final safe dispatch configuration and the full degradation chain.
 */
inline FallbackDecision resolve_dispatch_with_fallback(const FallbackContext& ctx) {
  FallbackDecision decision;
  decision.enable_cuda_graphs = ctx.enable_cuda_graphs;

  const size_t n_models = ctx.model_path_map.size();

  // What the config actually resolves to today.
  const auto selection = select_dispatch_kind(ctx.model_path_map,
                                              ctx.input_map,
                                              ctx.output_map,
                                              ctx.backend,
                                              ctx.backend_map,
                                              ctx.input_on_cuda,
                                              ctx.dynamic_input_dims,
                                              ctx.parallel_inference);
  decision.fast_single_model_name = selection.fast_single_model_name;

  // What it WOULD have resolved to with every static blocker removed (all-TRT, device inputs, no
  // chaining) — this is the "requested" side of the log line for cases H/I/J.
  const auto ideal = select_dispatch_kind(ctx.model_path_map,
                                          {},
                                          {},
                                          "trt",
                                          {},
                                          /*input_on_cuda=*/true,
                                          ctx.dynamic_input_dims,
                                          ctx.parallel_inference)
                         .kind;
  decision.primary_choice = ideal;
  decision.primary_protection = decide_capture_protection(ideal,
                                                          ctx.enable_cuda_graphs,
                                                          ctx.green_context_pool_present,
                                                          ctx.green_context_num_partitions)
                                    .outcome;

  DispatchKind current = selection.kind;
  // Green-context isolation is "available" until proven that the pool is too small for it.
  bool green_isolation_available = ctx.green_context_pool_present;

  if (selection.kind == DispatchKind::kStandard) {
    // Static eligibility blockers: non-TRT backend, host inputs, chained models. All are
    // reported (a config can trip several), and all land on kStandard.
    const auto offenders = detail::non_trt_models(ctx.model_path_map, ctx.backend, ctx.backend_map);
    if (!offenders.empty()) {
      FallbackStep step;
      step.reason = FallbackReason::kNonTrtBackend;
      step.from = ideal;
      step.to = DispatchKind::kStandard;
      step.from_protection = decision.primary_protection;
      step.severity = default_severity(step.reason);
      step.reason_detail = "fast paths require all-TRT — found: [" + detail::join(offenders) + "]";
      step.fix =
          "use backend=\"trt\" (or an all-\"trt\" backend_map) for every model to enable the fast "
          "dispatch paths";
      decision.chain.push_back(step);
    }
    if (!ctx.input_on_cuda) {
      FallbackStep step;
      step.reason = FallbackReason::kHostInputs;
      step.from = ideal;
      step.to = DispatchKind::kStandard;
      step.from_protection = decision.primary_protection;
      step.severity = default_severity(step.reason);
      step.reason_detail =
          "fast paths disabled — host inputs require kStandard staging (the fast paths dispatch "
          "straight to TensorRT on the caller's stream and never stage host<->device buffers)";
      step.fix = "set input_on_cuda=true and feed device-resident tensors";
      decision.chain.push_back(step);
    }
    const auto pairs = detail::chained_pairs(ctx.input_map, ctx.output_map);
    if (!pairs.empty()) {
      FallbackStep step;
      step.reason = FallbackReason::kChainedModels;
      step.from = ideal;
      step.to = DispatchKind::kStandard;
      step.from_protection = decision.primary_protection;
      step.severity = default_severity(step.reason);
      step.reason_detail = std::string(to_string(ideal)) +
                           " rejected — chained models detected: [" + detail::join(pairs) + "]";
      step.fix =
          "if the models are actually independent, split them into separate InferenceOps; the "
          "fast paths cannot order a producer before its consumer";
      decision.chain.push_back(step);
    }
    current = DispatchKind::kStandard;
  } else {
    // CudaStreamPool cannot supply one distinct stream per model.
    if (detail::is_parallel_kind(current)) {
      const auto demoted = demote_if_stream_pool_insufficient(
          current, ctx.stream_pool_present, ctx.stream_pool_max_size, n_models);
      if (demoted != current) {
        FallbackStep step;
        step.reason = FallbackReason::kStreamPoolInsufficient;
        step.from = current;
        step.to = demoted;
        step.from_protection = decide_capture_protection(current,
                                                         ctx.enable_cuda_graphs,
                                                         green_isolation_available,
                                                         ctx.green_context_num_partitions)
                                   .outcome;
        step.to_protection = CaptureProtectionOutcome::kNotApplicable;
        step.severity = default_severity(step.reason);
        step.reason_detail =
            ctx.stream_pool_present
                ? "cuda_stream_pool max_size (" + std::to_string(ctx.stream_pool_max_size) +
                      ") is smaller than the model count (" + std::to_string(n_models) +
                      "); inline parallel dispatch needs one distinct stream per model"
                : "no cuda_stream_pool was configured, so no per-model streams can be leased for "
                  "inline parallel dispatch";
        step.fix =
            ctx.stream_pool_present
                ? "increase cuda_stream_pool max_size to at least " + std::to_string(n_models)
                : "configure a cuda_stream_pool with max_size >= " + std::to_string(n_models);
        decision.chain.push_back(step);
        current = demoted;
      }
    }

    // kDynParFast + CUDA graphs with no green-context pool. Dynamic shapes have no setup-time
    // warm-up path, so there is no durable protection for concurrent capture; drop to the
    // single-stream dynamic sequential path (which keeps graphs — capture there is serialized on
    // one stream). If graphs still fail at runtime, they are disabled per model
    // (F_RUNTIME_CAPTURE_INVALIDATED), and the remaining rung is kStandard without graphs.
    if (current == DispatchKind::kDynParFast && ctx.enable_cuda_graphs &&
        !green_isolation_available) {
      FallbackStep step;
      step.reason = FallbackReason::kDynParFastGraphsNoGreenContext;
      step.from = DispatchKind::kDynParFast;
      step.to = DispatchKind::kDynSeqFast;
      step.from_protection = CaptureProtectionOutcome::kRefuse;
      step.to_protection = CaptureProtectionOutcome::kNotApplicable;
      step.severity = default_severity(step.reason);
      step.reason_detail =
          "kDynParFast with enable_cuda_graphs=true has no durable concurrent-capture protection "
          "without a CudaGreenContextPool (dynamic shapes have no setup-time warm-up path); using "
          "the single-stream dynamic sequential path with graphs instead of refusing to start. If "
          "capture still fails at runtime, graphs are disabled per model "
          "(F_RUNTIME_CAPTURE_INVALIDATED) and the last rung is kStandard without graphs";
      step.fix =
          "supply a cuda_green_context_pool (>= one partition per model), or set "
          "enable_cuda_graphs=false, or make the input dimensions static";
      decision.chain.push_back(step);
      current = DispatchKind::kDynSeqFast;
    }

    // Green-context pool present but too few partitions to give every model its own.
    if (detail::is_parallel_kind(current) && ctx.enable_cuda_graphs &&
        ctx.green_context_pool_present &&
        ctx.green_context_num_partitions < static_cast<uint32_t>(n_models)) {
      FallbackStep step;
      step.reason = FallbackReason::kGreenContextPartitionsInsufficient;
      step.from = current;
      step.from_protection = CaptureProtectionOutcome::kGreenContextIsolated;
      step.severity = default_severity(step.reason);
      const std::string detail_prefix =
          "cuda_green_context_pool has " + std::to_string(ctx.green_context_num_partitions) +
          " partition(s) for " + std::to_string(n_models) +
          " models, so models share a CUcontext and green-context isolation cannot protect "
          "concurrent CUDA-Graph capture";
      green_isolation_available = false;
      if (current == DispatchKind::kParFast) {
        // Static shapes: runtime mutex-based capture serialization is durable here.
        step.to = DispatchKind::kParFast;
        step.to_protection = CaptureProtectionOutcome::kMutexProtected;
        step.reason_detail =
            detail_prefix + "; falling back to runtime mutex-based capture serialization";
        step.fix = "increase cuda_green_context_pool num_partitions to at least " +
                   std::to_string(n_models) + " to restore lock-free green-context isolation";
      } else {
        // kDynParFast: dynamic shapes have no setup-time warm-up path, so mutex-based
        // serialization is not durable here either — go sequential.
        step.to = DispatchKind::kDynSeqFast;
        step.to_protection = CaptureProtectionOutcome::kNotApplicable;
        step.reason_detail = detail_prefix +
                             "; dynamic shapes have no setup-time warm-up path, so mutex-based "
                             "capture serialization is not available either — using the "
                             "single-stream sequential path";
        step.fix = "increase cuda_green_context_pool num_partitions to at least " +
                   std::to_string(n_models) + ", or set enable_cuda_graphs=false";
        current = DispatchKind::kDynSeqFast;
      }
      decision.chain.push_back(step);
    }
  }

  // Final protection for whatever survived the ladder.
  auto protection = decide_capture_protection(current,
                                              decision.enable_cuda_graphs,
                                              green_isolation_available,
                                              ctx.green_context_num_partitions);
  if (protection.outcome == CaptureProtectionOutcome::kRefuse) {
    // Defensive: the ladder above is expected to have removed every refusable combination. If a
    // future policy change reintroduces one, degrade rather than refuse.
    FallbackStep step;
    step.reason = FallbackReason::kDynParFastGraphsNoGreenContext;
    step.from = current;
    step.to = DispatchKind::kStandard;
    step.from_protection = CaptureProtectionOutcome::kRefuse;
    step.to_protection = CaptureProtectionOutcome::kNotApplicable;
    step.severity = FallbackSeverity::kWarn;
    step.reason_detail = "no durable CUDA-Graph capture protection is available for " +
                         std::string(to_string(current)) +
                         "; disabling CUDA graphs and using the canonical dispatch path";
    step.fix = "supply a cuda_green_context_pool or set enable_cuda_graphs=false";
    decision.chain.push_back(step);
    current = DispatchKind::kStandard;
    decision.enable_cuda_graphs = false;
    protection = decide_capture_protection(
        current, false, green_isolation_available, ctx.green_context_num_partitions);
  }

  // Parallelism was requested but not obtained. Emitted last so its `reason` can name the rung
  // that actually caused it.
  if (ctx.parallel_inference && n_models >= 2 && !detail::is_parallel_kind(current)) {
    std::string cause;
    if (!decision.chain.empty()) {
      cause = std::string("of ") + reason_key(decision.chain.front().reason) + " (" +
              decision.chain.front().reason_detail + ")";
    } else {
      cause =
          "the fast-path eligibility gate did not select a parallel dispatch kind for this "
          "configuration";
    }
    FallbackStep step;
    step.reason = FallbackReason::kParallelRequestedSerialSelected;
    step.from = ctx.dynamic_input_dims ? DispatchKind::kDynParFast : DispatchKind::kParFast;
    step.to = current;
    step.from_protection = decision.primary_protection;
    step.to_protection = protection.outcome;
    step.severity = default_severity(step.reason);
    step.reason_detail = std::string("parallel_inference=true honored -> ") + to_string(current) +
                         " selected because " + cause;
    step.fix =
        "address the cause above, or set parallel_inference=false to make the sequential "
        "dispatch explicit";
    decision.chain.push_back(step);
  }

  decision.final_choice = current;
  decision.final_protection = protection.outcome;

  std::string summary =
      std::string("InferenceOp dispatch ladder: requested ") + to_string(decision.primary_choice) +
      " + " + to_string(decision.primary_protection) + ", using " +
      to_string(decision.final_choice) + " + " + to_string(decision.final_protection);
  if (!decision.chain.empty()) {
    std::vector<std::string> keys;
    keys.reserve(decision.chain.size());
    for (const auto& step : decision.chain) {
      keys.emplace_back(reason_key(step.reason));
    }
    summary += " after [" + detail::join(keys) + "]";
  }
  decision.user_message = summary;
  return decision;
}

/**
 * @brief Formats one ladder rung in the standard InferenceOp fallback layout:
 *
 * ```
 * InferenceOp/HoloInfer: fallback triggered — <REASON_KEY>
 *     requested: <primary_dispatch> + <protection>
 *     using:     <fallback_dispatch> + <fallback_protection>
 *     reason:    <one-line explanation>
 *     fix:       <user-side suggestion, or "none">
 * ```
 *
 * @param step The rung to format.
 * @return The formatted, multi-line message (no trailing newline).
 */
inline std::string format_fallback_message(const FallbackStep& step) {
  std::string message = "InferenceOp/HoloInfer: fallback triggered — ";
  message += reason_key(step.reason);
  message += "\n    requested: ";
  message += to_string(step.from);
  message += " + ";
  message += to_string(step.from_protection);
  message += "\n    using:     ";
  message += to_string(step.to);
  message += " + ";
  message += to_string(step.to_protection);
  message += "\n    reason:    ";
  message += step.reason_detail;
  message += "\n    fix:       ";
  message += step.fix.empty() ? "none" : step.fix;
  return message;
}

/**
 * @brief Bounded, process-wide record of every fallback message emitted through
 * emit_fallback_step().
 *
 * The Holoscan logger has no sink-hooking API, so this recorder is what makes the fallback
 * messages observable to tests (and to anyone debugging a pipeline programmatically). It stores
 * the exact string handed to HOLOSCAN_LOG_WARN/HOLOSCAN_LOG_INFO. Only touched when a fallback
 * actually fires — never on the steady-state dispatch path.
 */
class FallbackLogRecorder {
 public:
  /// Maximum number of retained messages; the oldest are dropped past this.
  static constexpr size_t kCapacity = 128;

  static void record(const std::string& message) {
    std::lock_guard<std::mutex> lock(mutex());
    auto& records = storage();
    if (records.size() >= kCapacity) {
      records.erase(records.begin());
    }
    records.push_back(message);
  }

  /// @brief Snapshot of the retained messages, oldest first.
  static std::vector<std::string> snapshot() {
    std::lock_guard<std::mutex> lock(mutex());
    return storage();
  }

  /// @brief Drops all retained messages (tests call this in SetUp()).
  static void clear() {
    std::lock_guard<std::mutex> lock(mutex());
    storage().clear();
  }

  /// @brief True when any retained message reports `reason`.
  static bool contains(FallbackReason reason) {
    const std::string needle = std::string("fallback triggered — ") + reason_key(reason);
    std::lock_guard<std::mutex> lock(mutex());
    for (const auto& record : storage()) {
      if (record.find(needle) != std::string::npos) {
        return true;
      }
    }
    return false;
  }

 private:
  // Function-local statistics (rather than inline variables) so every translation unit and shared
  // library in the process shares exactly one instance.
  /// @brief Return the process-wide record storage vector.
  static std::vector<std::string>& storage() {
    static std::vector<std::string> records;
    return records;
  }
  /// @brief Return the mutex guarding concurrent access to `storage()`.
  static std::mutex& mutex() {
    static std::mutex m;
    return m;
  }
};

/// @brief Logs one ladder rung at its severity and records it for diagnostics/tests.
inline void emit_fallback_step(const FallbackStep& step) {
  const std::string message = format_fallback_message(step);
  FallbackLogRecorder::record(message);
  if (step.severity == FallbackSeverity::kWarn) {
    HOLOSCAN_LOG_WARN("{}", message);
  } else {
    HOLOSCAN_LOG_INFO("{}", message);
  }
}

/// @brief Logs every rung of a resolved decision, in order, followed by the one-line summary.
inline void emit_fallback_decision(const FallbackDecision& decision) {
  for (const auto& step : decision.chain) {
    emit_fallback_step(step);
  }
  HOLOSCAN_LOG_INFO("{}", decision.user_message);
}

/**
 * @brief Convenience builder for the runtime (non-start()) fallback reasons, which have no
 * dispatch-kind change to report — only a protection/feature downgrade.
 *
 * @param reason The runtime fallback reason.
 * @param kind The dispatch kind in effect (reported on both sides of the message).
 * @param from_protection Protection that was expected.
 * @param to_protection Protection actually in effect.
 * @param reason_detail One-line explanation.
 * @param fix Suggested user-side fix ("none" when empty).
 * @return The populated FallbackStep.
 */
inline FallbackStep make_runtime_fallback_step(FallbackReason reason, DispatchKind kind,
                                               CaptureProtectionOutcome from_protection,
                                               CaptureProtectionOutcome to_protection,
                                               const std::string& reason_detail,
                                               const std::string& fix) {
  FallbackStep step;
  step.reason = reason;
  step.from = kind;
  step.to = kind;
  step.from_protection = from_protection;
  step.to_protection = to_protection;
  step.severity = default_severity(reason);
  step.reason_detail = reason_detail;
  step.fix = fix;
  return step;
}

}  // namespace holoscan::ops::inference_eligibility
