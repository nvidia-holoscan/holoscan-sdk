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
#include <string>
#include <utility>
#include <vector>
namespace holoscan {
namespace inference {

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

// Multi-model sequential fast-path dispatch.
// Walks the cached all_ctx_[m] / all_params_[m] flat vectors with no map lookups,
// using do_inference_on_stream so each model's enqueueV3 lands on the caller's
// stream (no per-instance event handshake).
InferStatus ManagerInfer::run_core_inference_seq_fast(const DataMap& input_data,
                                                      const DataMap& output_data,
                                                      cudaStream_t cuda_stream) {
  if (all_params_.empty() || all_ctx_.empty()) {
    return InferStatus(holoinfer_code::H_ERROR,
                       "InferenceOp seq-fast: pointers not cached (eligibility blocked?)");
  }
  if (!seq_fast_data_cached_) {
    const size_t N = all_params_.size();
    all_indata_.assign(N, {});
    all_outdata_.assign(N, {});
    for (size_t m = 0; m < N; ++m) {
      const auto& in_tensors = all_params_[m]->get_input_tensor_names();
      const auto& out_tensors = all_params_[m]->get_output_tensor_names();
      if (in_tensors.empty() || out_tensors.empty()) {
        return InferStatus(holoinfer_code::H_ERROR,
                           "InferenceOp seq-fast: empty tensor names for " + all_model_names_[m]);
      }
      all_indata_[m].reserve(in_tensors.size());
      all_outdata_[m].reserve(out_tensors.size());
      for (const auto& t : in_tensors) {
        const std::shared_ptr<DataBuffer>* buf = nullptr;
        if (!lookup_tensor_buffer(input_data, output_data, t, &buf)) {
          return InferStatus(holoinfer_code::H_ERROR,
                             "InferenceOp seq-fast: missing input tensor " + t);
        }
        all_indata_[m].push_back(*buf);
      }
      for (const auto& t : out_tensors) {
        auto it = output_data.find(t);
        if (it == output_data.end()) {
          return InferStatus(holoinfer_code::H_ERROR,
                             "InferenceOp seq-fast: missing output tensor " + t);
        }
        all_outdata_[m].push_back(it->second);
      }
    }
    seq_fast_data_cached_ = true;
  }
  for (size_t m = 0; m < all_ctx_.size(); ++m) {
    auto i_status =
        all_ctx_[m]->do_inference_on_stream(all_indata_[m], all_outdata_[m], cuda_stream);
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
InferStatus ManagerInfer::run_core_inference_par_fast(const DataMap& input_data,
                                                      const DataMap& output_data,
                                                      cudaStream_t cuda_stream) {
  if (par_streams_.empty() || all_params_.empty()) {
    return InferStatus(holoinfer_code::H_ERROR,
                       "InferenceOp par-fast: streams not initialized (demoted)");
  }
  if (!seq_fast_data_cached_) {
    // Reuse the seq-fast cache build logic.
    auto s = run_core_inference_seq_fast(input_data, output_data, cuda_stream);
    // Discard the seq-fast invocation's result; the next loop will redo it on per-model
    // streams. The cache is now populated so subsequent compute() calls skip the rebuild. Note:
    // this means the FIRST compute() call effectively runs sequentially; subsequent compute()
    // calls run in parallel.
    seq_fast_data_cached_ = true;
    if (s.get_code() == holoinfer_code::H_ERROR)
      return s;
    return InferStatus();
  }
  check_cuda(cudaEventRecord(par_outer_event_, cuda_stream));
  for (size_t m = 0; m < all_ctx_.size(); ++m) {
    check_cuda(cudaStreamWaitEvent(par_streams_[m], par_outer_event_, 0));
    auto i_status =
        all_ctx_[m]->do_inference_on_stream(all_indata_[m], all_outdata_[m], par_streams_[m]);
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
      auto s = all_ctx_[m]->do_inference_on_stream(tmp_in[m], tmp_out[m], par_streams_[m]);
      if (s.get_code() == holoinfer_code::H_ERROR) {
        s.display_message();
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

    auto s = all_ctx_[m]->do_inference_on_stream(indata, outdata, cuda_stream);
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
    auto s = all_ctx_[m]->do_inference_on_stream(indata, outdata, par_streams_[m]);
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
