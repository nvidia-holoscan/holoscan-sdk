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

#include "manager/fallback_runtime.hpp"

#include <sys/sysinfo.h>

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace holoscan {
namespace inference {

InferStatus ManagerInfer::set_inference_params(std::shared_ptr<InferenceSpecs>& inference_specs) {
  InferStatus status = InferStatus(holoinfer_code::H_ERROR);

  auto multi_model_map = inference_specs->get_path_map();
  auto dla_core_map = inference_specs->get_dla_core_map();
  const auto& backend_type = inference_specs->backend_type_;
  auto backend_map = inference_specs->get_backend_map();
  auto trt_opt_profile = inference_specs->trt_opt_profile_;
  cuda_buffer_in_ = inference_specs->cuda_buffer_in_;
  cuda_buffer_out_ = inference_specs->cuda_buffer_out_;
  green_context_isolated_ = inference_specs->green_context_isolated_;

  if (multi_model_map.size() <= 0) {
    status.set_message("Inference manager, Empty model map for setup");
    return status;
  }

  if (backend_type.length() != 0 && backend_map.size() != 0) {
    status.set_message(
        "Inference manager, Backend type for all models and Backend map for each models specified. "
        "Either Backend type or Backend map is allowed.");
    return status;
  }

  if (backend_type.length() != 0) {
    if (supported_backend_.find(backend_type) == supported_backend_.end()) {
      status.set_message("Inference manager, " + backend_type +
                         " does not exist in inference toolkit");
      return status;
    }
  } else {
    if (backend_map.size() != 0) {
      for (auto const& [_, backend_] : backend_map) {
        if (supported_backend_.find(backend_) == supported_backend_.end()) {
          status.set_message("Inference manager, " + backend_ +
                             " does not exist in inference toolkit");
          return status;
        }
      }
    } else {
      status.set_message("Inference manager, neither backend or backend map specified");
      return status;
    }
  }

  // Single-GPU only: all inference happens on the default data-transfer GPU (device_gpu_dt_ = 0).
  check_cuda(cudaSetDevice(device_gpu_dt_));

  try {
    // create inference contexts and memory allocations for each model
    for (auto& [model_name, model_path] : multi_model_map) {
      if (infer_param_.find(model_name) != infer_param_.end()) {
        status.set_message("Duplicate entry in settings for " + model_name);
        return status;
      }

      infer_param_.insert({model_name, std::make_unique<Params>()});

      infer_param_.at(model_name)->set_cuda_flag(inference_specs->oncuda_);
      infer_param_.at(model_name)->set_instance_name(model_name);
      infer_param_.at(model_name)->set_model_path(model_path);

      if (inference_specs->output_map_.find(model_name) == inference_specs->output_map_.end()) {
        status.set_message("Inference Map not found for " + model_name);
        return status;
      }

      const auto device_id = device_gpu_dt_;
      infer_param_.at(model_name)->set_device_id(device_id);

      auto dla_core = inference_specs->dla_core_;
      if (dla_core_map.find(model_name) != dla_core_map.end()) {
        dla_core = std::stoi(dla_core_map.at(model_name));
        HOLOSCAN_LOG_INFO("DLA core: {} for Model: {}", dla_core, model_name);
      }

      // Get input and output tensor maps of the model from inference_specs
      auto out_tensor_names = inference_specs->output_map_.at(model_name);
      auto in_tensor_names = inference_specs->input_map_.at(model_name);

      // assign the input and output tensor names to the infer_param object
      infer_param_.at(model_name)->set_tensor_names(in_tensor_names, true);
      infer_param_.at(model_name)->set_tensor_names(out_tensor_names, false);

      check_cuda(cudaSetDevice(device_id));

      auto current_backend = holoinfer_backend::h_trt;
      if (backend_type.length() != 0) {
        current_backend = supported_backend_.at(backend_type);
      }

      if (backend_map.size() != 0) {
        if (backend_map.find(model_name) == backend_map.end()) {
          status.set_message("ERROR: Backend not found for model " + model_name);
          return status;
        }
        const auto& backend_ = backend_map.at(model_name);
        current_backend = supported_backend_.at(backend_);
      }

      switch (current_backend) {
        case holoinfer_backend::h_trt: {
          if (inference_specs->is_engine_path_) {
            if (inference_specs->use_fp16_) {
              HOLOSCAN_LOG_WARN("Engine files are the input, fp16 check/conversion is ignored");
            }
            if (dla_core > -1) {
              HOLOSCAN_LOG_WARN("Engine files are the input, DLA core is ignored");
            }
          }
          if (!inference_specs->oncuda_) {
            status.set_message("ERROR: TRT backend supports inference on GPU only");
            return status;
          }

          // check here for valid entry
          if (trt_opt_profile.find(model_name) == trt_opt_profile.end()) {
            trt_opt_profile[model_name] = {};
          }
          auto current_trt_opt_profile = trt_opt_profile.at(model_name);

          // The multi-model fast paths (kSeqFast/kParFast/kDynSeqFast/kDynParFast) dispatch
          // exclusively through TrtInfer::do_inference_on_stream(), which enqueues on the
          // *caller-supplied* stream (either the operator's own cuda_stream for seq-fast, or a
          // dedicated par_streams_[m] for par-fast) and never touches the context's own
          // internally-cached cuda_stream_. In that mode requesting a stream from the pool here
          // would be pure waste and would also inflate the pool's call count beyond one
          // allocation per model (see the par_streams_ provisioning below, which is the sole
          // legitimate per-model pool consumer for fast_multi_model_). Only pass the allocator
          // through when this context may still fall back to the event-based do_inference() path
          // (single-model kFast/kDynFast, or kStandard's work_queue-indexed worker), which does
          // rely on cuda_stream_.
          const auto& allocate_cuda_stream_for_ctx = inference_specs->fast_multi_model_
                                                         ? std::function<cudaStream_t(int32_t)>{}
                                                         : inference_specs->allocate_cuda_stream_;

          holo_infer_context_.insert(
              {model_name,
               std::make_unique<TrtInfer>(model_path,
                                          model_name,
                                          current_trt_opt_profile,
                                          device_id,
                                          device_gpu_dt_,
                                          inference_specs->use_fp16_,
                                          inference_specs->use_cuda_graphs_,
                                          dla_core,
                                          inference_specs->dla_gpu_fallback_,
                                          inference_specs->is_engine_path_,
                                          cuda_buffer_in_,
                                          cuda_buffer_out_,
                                          allocate_cuda_stream_for_ctx,
                                          inference_specs->build_cuda_context_,
                                          inference_specs->build_sm_count_,
                                          inference_specs->max_batch_size_,
                                          inference_specs->max_memory_)});

          if (inference_specs->gpu_resident_inference_) {
            try {
              holo_infer_context_.at(model_name)
                  ->init_gr_inference(inference_specs->gpu_resident_input_,
                                      inference_specs->gpu_resident_output_);
            } catch (const std::runtime_error& e) {
              status.set_message("ERROR: " + std::string(e.what()));
              return status;
            }
          }
          break;
        }
        default: {
          status.set_message("ERROR: Backend not supported");
          return status;
        }
      }

      auto output_node_size = holo_infer_context_.at(model_name)->get_output_dims().size();
      auto input_node_size = holo_infer_context_.at(model_name)->get_input_dims().size();

      if (out_tensor_names.size() != output_node_size) {
        HOLOSCAN_LOG_ERROR("Size mismatch. Out tensor names: {}, Output node size: {}",
                           out_tensor_names.size(),
                           output_node_size);
        status.set_message("Output tensor size not equal to output nodes in model.");
        return status;
      }

      if (in_tensor_names.size() != input_node_size) {
        status.set_message("Input tensor size not equal to input nodes in model.");
        return status;
      }

      for (unsigned int d = 0; d < out_tensor_names.size(); d++) {
        std::vector<int64_t> dims = holo_infer_context_.at(model_name)->get_output_dims()[d];
        for (int td = 0; td < dims.size(); td++) {
          if (dims[td] < 1) {
            dims[td] = 1;
          }
        }
        auto datatype = holo_infer_context_.at(model_name)->get_output_datatype()[d];
        if (datatype == holoinfer_datatype::h_Unsupported) {
          status.set_message("Unsupported datatype for tensor" + out_tensor_names[d]);
          return status;
        }

        auto astatus = allocate_buffers(inference_specs->output_per_model_,
                                        dims,
                                        datatype,
                                        out_tensor_names[d],
                                        true /* allocate_cuda */,
                                        device_id);
        if (astatus.get_code() != holoinfer_code::H_SUCCESS) {
          astatus.display_message();
          status.set_message("Allocation failed for output tensor: " + out_tensor_names[d]);
          return status;
        }

        HOLOSCAN_LOG_INFO("HoloInfer buffer created for {}", out_tensor_names[d]);
      }

      models_input_dims_.insert({model_name, holo_infer_context_.at(model_name)->get_input_dims()});
      models_output_dims_.insert(
          {model_name, holo_infer_context_.at(model_name)->get_output_dims()});
    }

    check_cuda(cudaEventCreateWithFlags(&cuda_event_, cudaEventDisableTiming));

    if (inference_specs->parallel_processing_) {
      // create the work queue for parallel processing, limit the worker count the available core
      // count
      work_queue_ = std::make_unique<WorkQueue>(
          std::min(infer_param_.size(), static_cast<size_t>(get_nprocs())));
    }

    // Cache the single-model raw pointers used by run_core_inference_fast.
    if (infer_param_.size() == 1) {
      fast_model_name_ = infer_param_.begin()->first;
      fast_param_ = infer_param_.at(fast_model_name_).get();
      fast_ctx_ = holo_infer_context_.at(fast_model_name_).get();
    }

    // default-path optimization
    // ALWAYS populate the flat per-model pointer vectors, regardless of fast_multi_model_.
    // These pointers reference model-state objects that are immutable after
    // set_inference_params, so caching them is safe even for dynamic_input_dims pipelines.
    // This makes the work_queue parallel branch's run_core_inference_indexed reachable
    // even when fast_multi_model is off.
    all_params_.reserve(infer_param_.size());
    all_ctx_.reserve(infer_param_.size());
    all_model_names_.reserve(infer_param_.size());
    for (auto& [name, param_uptr] : infer_param_) {
      all_params_.push_back(param_uptr.get());
      all_ctx_.push_back(holo_infer_context_.at(name).get());
      all_model_names_.push_back(name);
    }
    // Per-model fallback-ladder state (cases E/F): "CUDA graphs were disabled for this model".
    runtime_graph_disabled_.assign(all_ctx_.size(), 0);

    // Set up the streamlined dispatch caches and per-model parallel
    // streams ONLY when:
    //   - the caller has explicitly opted in (fast_multi_model_ == true), AND
    //   - none of the canonical operator's advanced features are in use:
    //       per-model DLA, GR inference, dynamic input dims. These features have their own
    //       bookkeeping that the fast paths bypass; turning them on together would silently
    //       skip required work.
    fast_multi_model_ = inference_specs->fast_multi_model_;

    // Static-mode blockers — these force kStandard even when fast_multi_model is on.
    // DLA (dla_core / dla_core_map) is NOT a blocker: DLA is a build-time device
    // assignment; per-compute() call dispatch (enqueueV3) is identical for GPU- and DLA-bound
    // execution contexts. The DLA engine is built at set_inference_params; runtime
    // routing is transparent.
    const bool static_blockers = inference_specs->gpu_resident_inference_;

    // dynamic_input_dims no longer blocks par-fast stream allocation —
    // the DynParFast path uses the same per-model streams but skips the prewarm and
    // pre-built I/O cache. Still blocked for the static cache (seq_fast_data_cached_),
    // which the dyn paths don't use.
    if (fast_multi_model_ && !static_blockers) {
      // Per-model parallel streams + events
      bool all_trt = true;
      if (!inference_specs->backend_type_.empty()) {
        all_trt = (inference_specs->backend_type_ == "trt");
      } else {
        for (const auto& [m, b] : inference_specs->backend_map_) {
          if (b != "trt") {
            all_trt = false;
            break;
          }
        }
      }
      const int par_fast_max_n = []() {
        const char* env = std::getenv("HOLOINFER_PAR_FAST_MAX_N");
        return env ? std::atoi(env) : std::numeric_limits<int>::max();
      }();
      if (inference_specs->parallel_processing_ && infer_param_.size() >= 2 && all_trt &&
          static_cast<int>(infer_param_.size()) <= par_fast_max_n &&
          inference_specs->allocate_cuda_stream_) {
        std::vector<cudaStream_t> tentative(infer_param_.size(), nullptr);
        bool ok = true;
        for (size_t m = 0; m < infer_param_.size() && ok; ++m) {
          try {
            tentative[m] = inference_specs->allocate_cuda_stream_(device_gpu_dt_);
          } catch (const std::exception&) {
            ok = false;
            break;
          }
          if (!tentative[m])
            ok = false;
        }
        if (ok) {
          std::set<cudaStream_t> uniq(tentative.begin(), tentative.end());
          if (uniq.size() != tentative.size())
            ok = false;
        }
        if (ok) {
          par_streams_ = std::move(tentative);
          par_done_events_.assign(par_streams_.size(), nullptr);
          for (auto& e : par_done_events_) {
            check_cuda(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
          }
          check_cuda(cudaEventCreateWithFlags(&par_outer_event_, cudaEventDisableTiming));

          // Runtime green-context binding: when a CudaGreenContextPool was supplied
          // (InferenceSpecs::green_context_partition_contexts_ non-empty), bind each model's
          // TrtInfer instance (all_ctx_[m]) to partition m % N's CUcontext BEFORE prewarm, so even
          // the setup-time synthetic passes exercise the same per-partition dispatch path as
          // steady-state compute() calls. Without this, decide_capture_protection() reports
          // kGreenContextIsolated but every model still enqueues into whatever context happens to
          // be current — safety would come only from capture_serialization_mutex_, defeating the
          // purpose of supplying a pool.
          const auto& partition_contexts = inference_specs->green_context_partition_contexts_;
          if (!partition_contexts.empty()) {
            for (size_t m = 0; m < all_ctx_.size(); ++m) {
              const size_t partition_idx = m % partition_contexts.size();
              CUcontext ctx = partition_contexts[partition_idx];
              if (ctx == nullptr) {
                HOLOSCAN_LOG_WARN(
                    "InferenceOp/HoloInfer: green context partition {} has no resolved "
                    "CUcontext; model '{}' will dispatch in the default/shared context.",
                    partition_idx,
                    all_model_names_[m]);
                continue;
              }
              all_ctx_[m]->bind_runtime_green_context(ctx, static_cast<uint32_t>(partition_idx));
              HOLOSCAN_LOG_INFO(
                  "InferenceOp/HoloInfer: green-context bound: partition {} / model {}",
                  partition_idx,
                  all_model_names_[m]);
            }
          }

          // Prewarm CUDA-graph captures during setup — STATIC SHAPES ONLY, and only when CUDA
          // Graphs are actually enabled. Under dynamic dims, the captured graphs would be
          // invalidated on the first shape change anyway; when enable_cuda_graphs is off there is
          // no graph capture to warm (do_inference_on_stream's capture branch is gated on
          // enable_cuda_graphs_), so prewarming would just be wasted synthetic passes.
          if (!inference_specs->dynamic_input_dims_ && inference_specs->use_cuda_graphs_) {
            auto pw = prewarm_par_fast(/*passes=*/3);
            if (pw.get_code() == holoinfer_code::H_ERROR) {
              pw.display_message();
              // Tear down on prewarm failure → demote to work_queue path.
              if (par_outer_event_) {
                cudaEventDestroy(par_outer_event_);
                par_outer_event_ = nullptr;
              }
              for (auto& e : par_done_events_)
                if (e)
                  cudaEventDestroy(e);
              par_done_events_.clear();
              par_streams_.clear();
            }
          }
        } else {
          // Stream-pool insufficiency, detected here rather than at InferenceOp::start(): the pool
          // advertised enough capacity but could not actually hand out N distinct streams (it was
          // shared with other operators, or it handed back a duplicate). par_streams_ stays empty,
          // which makes the par-fast dispatch entry point fall back to the canonical path.
          emit_runtime_fallback(
              FallbackReason::kStreamPoolInsufficient,
              fallback::DispatchKind::kParFast,
              fallback::CaptureProtectionOutcome::kMutexProtected,
              fallback::CaptureProtectionOutcome::kNotApplicable,
              "the executor's CudaStreamPool could not supply " +
                  std::to_string(infer_param_.size()) +
                  " distinct CUDA streams at setup time; inline parallel dispatch is unavailable "
                  "and this operator falls back to the canonical dispatch path",
              "increase cuda_stream_pool max_size to at least " +
                  std::to_string(infer_param_.size()) +
                  ", or give this operator its own cuda_stream_pool");
        }
      }
    }
  } catch (const std::runtime_error& rt) {
    raise_error("Inference Manager", "Setting Inference parameters: " + std::string(rt.what()));
  } catch (...) {
    raise_error("Inference Manager", "Setting Inference parameters: unknown exception occurred.");
  }

  parallel_processing_ = inference_specs->parallel_processing_;

  HOLOSCAN_LOG_INFO("Building execution plan...");
  auto plan_status = build_execution_plan(
      inference_specs->input_map_, inference_specs->output_map_, execution_plan_);

  if (plan_status.get_code() == holoinfer_code::H_ERROR) {
    HOLOSCAN_LOG_ERROR("Execution plan build failed: {}", plan_status.get_message());
    return plan_status;
  }

  return InferStatus();
}

}  // namespace inference
}  // namespace holoscan
