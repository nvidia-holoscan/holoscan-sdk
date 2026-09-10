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

#include "infer_manager.hpp"
#include "tensor_lookup.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "manager/fallback_runtime.hpp"

namespace holoscan {
namespace inference {

/// @brief Check green context isolation lost.
void ManagerInfer::check_green_context_isolation_lost() {
  if (!green_context_isolated_) {
    return;
  }
  if (!runtime_fallback_flags().green_context_push_failed.load(std::memory_order_relaxed)) {
    return;
  }
  // A backend could not push its green-context partition, so concurrent capture is no
  // longer protected by context isolation. Fall back to the prewarm + capture-serialization
  // mutex net for the rest of this instance's life, and re-arm the mutex window.
  green_context_isolated_ = false;
  par_fast_capture_ticks_ = 0;
  HOLOSCAN_LOG_WARN(
      "InferenceOp: green-context isolation was reported lost by a backend dispatch. "
      "Concurrent-capture protection has been demoted from green-context isolation to the "
      "runtime mutex net for the remaining life of this operator instance.");
  emit_runtime_fallback(
      FallbackReason::kGreenContextPushFailed,
      fallback::DispatchKind::kParFast,
      fallback::CaptureProtectionOutcome::kGreenContextIsolated,
      fallback::CaptureProtectionOutcome::kMutexProtected,
      "green-context isolation was reported lost by a backend dispatch; this ManagerInfer is "
      "re-engaging capture_serialization_mutex_ for its CUDA-Graph capture window",
      "verify the cuda_green_context_pool outlives every InferenceOp that uses it");
}

/// @brief Dispatch on stream with graph fallback.
InferStatus ManagerInfer::dispatch_on_stream_with_graph_fallback(
    size_t model_index, const std::vector<std::shared_ptr<DataBuffer>>& indata,
    std::vector<std::shared_ptr<DataBuffer>>& outdata, cudaStream_t stream) {
  auto* ctx = all_ctx_[model_index];
  InferStatus status = ctx->do_inference_on_stream(indata, outdata, stream);

  if (status.get_code() != holoinfer_code::H_ERROR) {
    return status;
  }

  // The dispatch failed while CUDA-Graph capture was still enabled for this model. The
  // overwhelmingly common cause is a capture invalidated by concurrent capture in the same CUDA
  // context. Disable graphs for THIS model only, warn once, and retry the tick without graphs.
  const bool already_degraded =
      model_index < runtime_graph_disabled_.size() && runtime_graph_disabled_[model_index] != 0;
  if (already_degraded || !ctx->cuda_graphs_enabled()) {
    return status;
  }

  ctx->disable_cuda_graphs();
  if (model_index < runtime_graph_disabled_.size()) {
    runtime_graph_disabled_[model_index] = 1;
  }
  emit_runtime_fallback(
      FallbackReason::kRuntimeCaptureInvalidated,
      fallback::DispatchKind::kParFast,
      fallback::CaptureProtectionOutcome::kMutexProtected,
      fallback::CaptureProtectionOutcome::kNotApplicable,
      "model '" + all_model_names_[model_index] +
          "': dispatch failed while CUDA-Graph capture was active (" + status.get_message() +
          "); CUDA Graphs are now disabled for this model and the tick is retried without them",
      "supply a cuda_green_context_pool with one partition per concurrently-capturing "
      "InferenceOp, or set enable_cuda_graphs=false for this model's operator");

  return ctx->do_inference_on_stream(indata, outdata, stream);
}

InferStatus ManagerInfer::run_core_inference_fast(const DataMap& input_data,
                                                  const DataMap& output_data,
                                                  cudaStream_t cuda_stream) {
  if (!fast_param_ || !fast_ctx_) {
    return InferStatus(holoinfer_code::H_ERROR, "InferenceOp fast path: pointers not cached");
  }

  // Lazy build of per-tensor DataBuffer pointer vectors.
  if (!fast_data_cached_) {
    const auto& in_tensors = fast_param_->get_input_tensor_names();
    const auto& out_tensors = fast_param_->get_output_tensor_names();
    if (in_tensors.empty() || out_tensors.empty()) {
      return InferStatus(holoinfer_code::H_ERROR, "InferenceOp fast path: empty tensor names");
    }
    fast_indata_.clear();
    fast_indata_.reserve(in_tensors.size());
    fast_outdata_.clear();
    fast_outdata_.reserve(out_tensors.size());
    for (const auto& t : in_tensors) {
      auto it = input_data.find(t);
      if (it == input_data.end()) {
        it = output_data.find(t);
        if (it == output_data.end()) {
          return InferStatus(holoinfer_code::H_ERROR, "InferenceOp fast path: missing input " + t);
        }
      }
      fast_indata_.push_back(it->second);
    }
    for (const auto& t : out_tensors) {
      auto it = output_data.find(t);
      if (it == output_data.end()) {
        return InferStatus(holoinfer_code::H_ERROR, "InferenceOp fast path: missing output " + t);
      }
      fast_outdata_.push_back(it->second);
    }
    fast_data_cached_ = true;
  }

  check_cuda(cudaEventRecord(cuda_event_, cuda_stream));
  cudaEvent_t cuda_event_inference = nullptr;
  auto i_status =
      fast_ctx_->do_inference(fast_indata_, fast_outdata_, cuda_event_, &cuda_event_inference);
  if (i_status.get_code() == holoinfer_code::H_ERROR) {
    i_status.display_message();
    return InferStatus(holoinfer_code::H_ERROR, "InferenceOp fast path: inference failed");
  }
  if (cuda_event_inference) {
    check_cuda(cudaStreamWaitEvent(cuda_stream, cuda_event_inference));
  }

  return InferStatus();
}

// Builds the multi-model I/O DataBuffer pointer cache. Pure tensor-name lookups — issues no
// CUDA work, so it is safe to invoke regardless of which stream (if any) the caller will
// subsequently dispatch inference on. Shared by run_core_inference_seq_fast and
// run_core_inference_par_fast so the latter never has to fall back to seq-fast's
// stream-issuing dispatch loop just to warm the cache (see run_core_inference_par_fast).
InferStatus ManagerInfer::build_seq_fast_cache(const DataMap& input_data,
                                               const DataMap& output_data) {
  if (seq_fast_data_cached_) {
    return InferStatus();
  }
  const size_t N = all_params_.size();
  all_indata_.assign(N, {});
  all_outdata_.assign(N, {});
  for (size_t m = 0; m < N; ++m) {
    const auto& in_tensors = all_params_[m]->get_input_tensor_names();
    const auto& out_tensors = all_params_[m]->get_output_tensor_names();
    if (in_tensors.empty() || out_tensors.empty()) {
      return InferStatus(
          holoinfer_code::H_ERROR,
          "InferenceOp fast-path cache: empty tensor names for " + all_model_names_[m]);
    }
    all_indata_[m].reserve(in_tensors.size());
    all_outdata_[m].reserve(out_tensors.size());
    for (const auto& t : in_tensors) {
      const std::shared_ptr<DataBuffer>* buf = nullptr;
      if (!lookup_tensor_buffer(input_data, output_data, t, &buf)) {
        return InferStatus(holoinfer_code::H_ERROR,
                           "InferenceOp fast-path cache: missing input tensor " + t);
      }
      all_indata_[m].push_back(*buf);
    }
    for (const auto& t : out_tensors) {
      auto it = output_data.find(t);
      if (it == output_data.end()) {
        return InferStatus(holoinfer_code::H_ERROR,
                           "InferenceOp fast-path cache: missing output tensor " + t);
      }
      all_outdata_[m].push_back(it->second);
    }
  }
  seq_fast_data_cached_ = true;
  return InferStatus();
}

// Multi-model sequential fast-path dispatch.
// Walks the cached all_ctx_[m] / all_params_[m] flat vectors with no map lookups,
// using do_inference_on_stream so each model's enqueueV3 lands on the caller's
// stream (no per-instance event handshake).
/// @brief Run core inference seq fast.
InferStatus ManagerInfer::run_core_inference_seq_fast(const DataMap& input_data,
                                                      const DataMap& output_data,
                                                      cudaStream_t cuda_stream) {
  if (all_params_.empty() || all_ctx_.empty()) {
    return InferStatus(holoinfer_code::H_ERROR,
                       "InferenceOp seq-fast: pointers not cached (eligibility blocked?)");
  }
  auto cache_status = build_seq_fast_cache(input_data, output_data);
  if (cache_status.get_code() == holoinfer_code::H_ERROR) {
    return cache_status;
  }
  for (size_t m = 0; m < all_ctx_.size(); ++m) {
    auto i_status =
        dispatch_on_stream_with_graph_fallback(m, all_indata_[m], all_outdata_[m], cuda_stream);
    if (i_status.get_code() == holoinfer_code::H_ERROR) {
      i_status.display_message();
      return InferStatus(holoinfer_code::H_ERROR,
                         "InferenceOp seq-fast: inference failed for " + all_model_names_[m]);
    }
  }
  return InferStatus();
}

// Multi-model parallel fast-path: inline enqueueV3 on
// per-model CUDA streams; outer stream waits on all per-model completion events.
// Returns error if par_streams_ is empty (caller should fall back to execute_inference).
/// @brief Run core inference par fast.
InferStatus ManagerInfer::run_core_inference_par_fast(const DataMap& input_data,
                                                      const DataMap& output_data,
                                                      cudaStream_t cuda_stream) {
  if (par_streams_.empty() || all_params_.empty()) {
    return InferStatus(holoinfer_code::H_ERROR,
                       "InferenceOp par-fast: streams not initialized (demoted)");
  }

  auto cache_status = build_seq_fast_cache(input_data, output_data);
  if (cache_status.get_code() == holoinfer_code::H_ERROR) {
    return cache_status;
  }

  // Capture-window safety net (see capture_serialization_mutex_ in infer_manager.hpp).
  check_green_context_isolation_lost();
  constexpr int kCaptureProtectionTicks = 3;
  std::unique_lock<std::mutex> capture_lock;
  if (!green_context_isolated_ && par_fast_capture_ticks_ < kCaptureProtectionTicks) {
    capture_lock = std::unique_lock<std::mutex>(capture_serialization_mutex_);
    ++par_fast_capture_ticks_;
  }

  check_cuda(cudaEventRecord(par_outer_event_, cuda_stream));
  for (size_t m = 0; m < all_ctx_.size(); ++m) {
    check_cuda(cudaStreamWaitEvent(par_streams_[m], par_outer_event_, 0));
    auto i_status =
        dispatch_on_stream_with_graph_fallback(m, all_indata_[m], all_outdata_[m], par_streams_[m]);
    if (i_status.get_code() == holoinfer_code::H_ERROR) {
      i_status.display_message();
      return InferStatus(holoinfer_code::H_ERROR,
                         "InferenceOp par-fast: inference failed for " + all_model_names_[m]);
    }
    check_cuda(cudaEventRecord(par_done_events_[m], par_streams_[m]));
  }
  for (size_t m = 0; m < par_done_events_.size(); ++m) {
    check_cuda(cudaStreamWaitEvent(cuda_stream, par_done_events_[m], 0));
  }
  return InferStatus();
}

// Indexed worker — race-free per-worker I/O vector
// construction. Used by execute_inference's work_queue fallback when fast_track
// is on but par-fast isn't applicable (mixed backend, pool exhausted, etc.).
InferStatus ManagerInfer::run_core_inference_indexed(size_t model_index, const DataMap& input_data,
                                                     const DataMap& output_data,
                                                     cudaStream_t cuda_stream) {
  if (model_index >= all_ctx_.size()) {
    return InferStatus(holoinfer_code::H_ERROR, "InferenceOp indexed worker: model index OOB");
  }
  std::vector<std::shared_ptr<DataBuffer>> indata, outdata;
  const auto& in_tensors = all_params_[model_index]->get_input_tensor_names();
  const auto& out_tensors = all_params_[model_index]->get_output_tensor_names();
  indata.reserve(in_tensors.size());
  outdata.reserve(out_tensors.size());
  for (const auto& t : in_tensors) {
    const std::shared_ptr<DataBuffer>* buf = nullptr;
    if (!lookup_tensor_buffer(input_data, output_data, t, &buf)) {
      return InferStatus(holoinfer_code::H_ERROR, "InferenceOp indexed worker: missing input " + t);
    }
    indata.push_back(*buf);
  }
  for (const auto& t : out_tensors) {
    auto it = output_data.find(t);
    if (it == output_data.end()) {
      return InferStatus(holoinfer_code::H_ERROR,
                         "InferenceOp indexed worker: missing output " + t);
    }
    outdata.push_back(it->second);
  }
  check_cuda(cudaEventRecord(cuda_event_, cuda_stream));
  cudaEvent_t evt = nullptr;
  auto i = all_ctx_[model_index]->do_inference(indata, outdata, cuda_event_, &evt);
  if (i.get_code() == holoinfer_code::H_ERROR) {
    i.display_message();
    return InferStatus(
        holoinfer_code::H_ERROR,
        "InferenceOp indexed worker: inference failed for " + all_model_names_[model_index]);
  }
  if (evt)
    check_cuda(cudaStreamWaitEvent(cuda_stream, evt));
  return InferStatus();
}

// Par-fast CUDA-graph prewarm. Allocates scratch I/O,
// runs synthetic forward passes on every per-model parallel stream, frees scratch.
InferStatus ManagerInfer::prewarm_par_fast(int passes) {
  if (par_streams_.empty() || all_ctx_.empty())
    return InferStatus();
  std::vector<DataMap> tmp_maps(all_ctx_.size());
  std::vector<std::vector<std::shared_ptr<DataBuffer>>> tmp_in(all_ctx_.size());
  std::vector<std::vector<std::shared_ptr<DataBuffer>>> tmp_out(all_ctx_.size());
  for (size_t m = 0; m < all_ctx_.size(); ++m) {
    const auto& in_names = all_params_[m]->get_input_tensor_names();
    const auto& out_names = all_params_[m]->get_output_tensor_names();
    auto in_dims = all_ctx_[m]->get_input_dims();
    auto out_dims = all_ctx_[m]->get_output_dims();
    auto in_dt = all_ctx_[m]->get_input_datatype();
    auto out_dt = all_ctx_[m]->get_output_datatype();
    for (size_t i = 0; i < in_names.size(); ++i) {
      auto dims = in_dims[i];
      for (auto& d : dims)
        if (d < 1)
          d = 1;
      const std::string key = "_prewarm_in_" + std::to_string(m) + "_" + in_names[i];
      auto a = allocate_buffers(tmp_maps[m], dims, in_dt[i], key, true, device_gpu_dt_);
      if (a.get_code() != holoinfer_code::H_SUCCESS) {
        return InferStatus(holoinfer_code::H_ERROR,
                           "prewarm_par_fast: input buffer alloc failed for " + in_names[i]);
      }
      tmp_in[m].push_back(tmp_maps[m].at(key));
    }
    for (size_t i = 0; i < out_names.size(); ++i) {
      auto dims = out_dims[i];
      for (auto& d : dims)
        if (d < 1)
          d = 1;
      const std::string key = "_prewarm_out_" + std::to_string(m) + "_" + out_names[i];
      auto a = allocate_buffers(tmp_maps[m], dims, out_dt[i], key, true, device_gpu_dt_);
      if (a.get_code() != holoinfer_code::H_SUCCESS) {
        return InferStatus(holoinfer_code::H_ERROR,
                           "prewarm_par_fast: output buffer alloc failed for " + out_names[i]);
      }
      tmp_out[m].push_back(tmp_maps[m].at(key));
    }
  }
  for (int p = 0; p < passes; ++p) {
    for (size_t m = 0; m < all_ctx_.size(); ++m) {
      InferStatus s = all_ctx_[m]->do_inference_on_stream(tmp_in[m], tmp_out[m], par_streams_[m]);
      if (s.get_code() == holoinfer_code::H_ERROR) {
        s.display_message();
        // The setup-time CUDA-graph prewarm failed for THIS model (e.g.
        // cudaErrorStreamCaptureUnsupported). Rather than tearing down par-fast for every model,
        // disable graphs for this model only and keep prewarming the rest. The model still runs
        // — it just enqueues directly instead of replaying a captured graph.
        if (all_ctx_[m]->cuda_graphs_enabled()) {
          all_ctx_[m]->disable_cuda_graphs();
          if (m < runtime_graph_disabled_.size()) {
            runtime_graph_disabled_[m] = 1;
          }
          emit_runtime_fallback(
              FallbackReason::kCaptureWarmupFailed,
              fallback::DispatchKind::kParFast,
              fallback::CaptureProtectionOutcome::kMutexProtected,
              fallback::CaptureProtectionOutcome::kNotApplicable,
              "model '" + all_model_names_[m] + "': setup-time CUDA-Graph warm-up pass failed (" +
                  s.get_message() +
                  "); CUDA Graphs are disabled for this model and warm-up continues for the "
                  "remaining models",
              "check that this model's engine supports CUDA stream capture, or set "
              "enable_cuda_graphs=false for this operator");
          continue;
        }
        // Graphs were already off for this model: a plain dispatch failure, not a capture
        // problem. Report it to the caller as before.
        return InferStatus(holoinfer_code::H_ERROR, "prewarm_par_fast: inference failed");
      }
    }
    for (size_t m = 0; m < par_streams_.size(); ++m) {
      check_cuda(cudaStreamSynchronize(par_streams_[m]));
    }
  }
  return InferStatus();
}

// Single-model dynamic-shape fast dispatch.
/// @brief Run core inference dyn fast.
InferStatus ManagerInfer::run_core_inference_dyn_fast(std::shared_ptr<InferenceSpecs>& specs,
                                                      cudaStream_t cuda_stream) {
  if (!fast_param_ || !fast_ctx_) {
    return InferStatus(holoinfer_code::H_ERROR, "InferenceOp dyn-fast: pointers not cached");
  }
  const auto& in_tensors = fast_param_->get_input_tensor_names();
  const auto& out_tensors = fast_param_->get_output_tensor_names();

  // TRT binding update for dynamic input shape.
  if (!fast_ctx_->set_dynamic_input_dimension(in_tensors, specs->dims_per_tensor_)) {
    return InferStatus(holoinfer_code::H_ERROR,
                       "InferenceOp dyn-fast: set_dynamic_input_dimension failed");
  }

  // Resolve I/O per compute() call (cannot cache under dynamic dims — shape may have changed).
  std::vector<std::shared_ptr<DataBuffer>> indata, outdata;
  indata.reserve(in_tensors.size());
  outdata.reserve(out_tensors.size());
  for (const auto& t : in_tensors) {
    const std::shared_ptr<DataBuffer>* buf = nullptr;
    if (!lookup_tensor_buffer(specs->data_per_tensor_, specs->output_per_model_, t, &buf)) {
      return InferStatus(holoinfer_code::H_ERROR, "InferenceOp dyn-fast: missing input " + t);
    }
    indata.push_back(*buf);
  }
  for (const auto& t : out_tensors) {
    auto it = specs->output_per_model_.find(t);
    if (it == specs->output_per_model_.end()) {
      return InferStatus(holoinfer_code::H_ERROR, "InferenceOp dyn-fast: missing output " + t);
    }
    outdata.push_back(it->second);
  }

  // Dispatch — no execution_plan walk, no activation/temporal scans.
  check_cuda(cudaEventRecord(cuda_event_, cuda_stream));
  cudaEvent_t evt = nullptr;
  auto s = fast_ctx_->do_inference(indata, outdata, cuda_event_, &evt);
  if (s.get_code() == holoinfer_code::H_ERROR) {
    s.display_message();
    return InferStatus(holoinfer_code::H_ERROR, "InferenceOp dyn-fast: inference failed");
  }
  if (evt) {
    check_cuda(cudaStreamWaitEvent(cuda_stream, evt));
  }

  // Refresh output dims — REQUIRED for dynamic dims.
  models_output_dims_[fast_model_name_] = fast_ctx_->get_output_dims();
  return InferStatus();
}

// Multi-model sequential dynamic-shape dispatch.
/// @brief Run core inference dyn seq fast.
InferStatus ManagerInfer::run_core_inference_dyn_seq_fast(std::shared_ptr<InferenceSpecs>& specs,
                                                          cudaStream_t cuda_stream) {
  if (all_ctx_.empty() || all_params_.empty()) {
    return InferStatus(holoinfer_code::H_ERROR, "InferenceOp dyn-seq-fast: pointers not cached");
  }
  for (size_t m = 0; m < all_ctx_.size(); ++m) {
    const auto& in_tensors = all_params_[m]->get_input_tensor_names();
    const auto& out_tensors = all_params_[m]->get_output_tensor_names();

    if (!all_ctx_[m]->set_dynamic_input_dimension(in_tensors, specs->dims_per_tensor_)) {
      return InferStatus(holoinfer_code::H_ERROR,
                         "InferenceOp dyn-seq-fast: set_dynamic_input_dimension failed for " +
                             all_model_names_[m]);
    }

    std::vector<std::shared_ptr<DataBuffer>> indata, outdata;
    indata.reserve(in_tensors.size());
    outdata.reserve(out_tensors.size());
    for (const auto& t : in_tensors) {
      const std::shared_ptr<DataBuffer>* buf = nullptr;
      if (!lookup_tensor_buffer(specs->data_per_tensor_, specs->output_per_model_, t, &buf)) {
        return InferStatus(holoinfer_code::H_ERROR, "InferenceOp dyn-seq-fast: missing input " + t);
      }
      indata.push_back(*buf);
    }
    for (const auto& t : out_tensors) {
      auto it = specs->output_per_model_.find(t);
      if (it == specs->output_per_model_.end()) {
        return InferStatus(holoinfer_code::H_ERROR,
                           "InferenceOp dyn-seq-fast: missing output " + t);
      }
      outdata.push_back(it->second);
    }

    auto s = dispatch_on_stream_with_graph_fallback(m, indata, outdata, cuda_stream);
    if (s.get_code() == holoinfer_code::H_ERROR) {
      s.display_message();
      return InferStatus(holoinfer_code::H_ERROR,
                         "InferenceOp dyn-seq-fast: inference failed for " + all_model_names_[m]);
    }
  }
  // Refresh output dims for all models (required for dynamic).
  for (size_t m = 0; m < all_ctx_.size(); ++m) {
    models_output_dims_[all_model_names_[m]] = all_ctx_[m]->get_output_dims();
  }
  return InferStatus();
}

// Multi-model parallel dynamic-shape dispatch.
// Reuses per-model streams + events from par_streams_/par_done_events_ for the
// same inline fanout pattern as run_core_inference_par_fast — no work_queue.
// No CUDA Graphs (would be invalidated by shape changes).
/// @brief Run core inference dyn par fast.
InferStatus ManagerInfer::run_core_inference_dyn_par_fast(std::shared_ptr<InferenceSpecs>& specs,
                                                          cudaStream_t cuda_stream) {
  if (par_streams_.empty() || all_ctx_.empty()) {
    return InferStatus(holoinfer_code::H_ERROR,
                       "InferenceOp dyn-par-fast: streams not initialized");
  }
  check_cuda(cudaEventRecord(par_outer_event_, cuda_stream));
  for (size_t m = 0; m < all_ctx_.size(); ++m) {
    const auto& in_tensors = all_params_[m]->get_input_tensor_names();
    const auto& out_tensors = all_params_[m]->get_output_tensor_names();

    if (!all_ctx_[m]->set_dynamic_input_dimension(in_tensors, specs->dims_per_tensor_)) {
      return InferStatus(holoinfer_code::H_ERROR,
                         "InferenceOp dyn-par-fast: set_dynamic_input_dimension failed for " +
                             all_model_names_[m]);
    }

    std::vector<std::shared_ptr<DataBuffer>> indata, outdata;
    indata.reserve(in_tensors.size());
    outdata.reserve(out_tensors.size());
    for (const auto& t : in_tensors) {
      const std::shared_ptr<DataBuffer>* buf = nullptr;
      if (!lookup_tensor_buffer(specs->data_per_tensor_, specs->output_per_model_, t, &buf)) {
        return InferStatus(holoinfer_code::H_ERROR, "InferenceOp dyn-par-fast: missing input " + t);
      }
      indata.push_back(*buf);
    }
    for (const auto& t : out_tensors) {
      auto it = specs->output_per_model_.find(t);
      if (it == specs->output_per_model_.end()) {
        return InferStatus(holoinfer_code::H_ERROR,
                           "InferenceOp dyn-par-fast: missing output " + t);
      }
      outdata.push_back(it->second);
    }

    check_cuda(cudaStreamWaitEvent(par_streams_[m], par_outer_event_, 0));
    auto s = dispatch_on_stream_with_graph_fallback(m, indata, outdata, par_streams_[m]);
    if (s.get_code() == holoinfer_code::H_ERROR) {
      s.display_message();
      return InferStatus(holoinfer_code::H_ERROR,
                         "InferenceOp dyn-par-fast: inference failed for " + all_model_names_[m]);
    }
    check_cuda(cudaEventRecord(par_done_events_[m], par_streams_[m]));
  }
  for (size_t m = 0; m < par_done_events_.size(); ++m) {
    check_cuda(cudaStreamWaitEvent(cuda_stream, par_done_events_[m], 0));
  }
  for (size_t m = 0; m < all_ctx_.size(); ++m) {
    models_output_dims_[all_model_names_[m]] = all_ctx_[m]->get_output_dims();
  }
  return InferStatus();
}

}  // namespace inference
}  // namespace holoscan
