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

#include <memory>
#include <mutex>
#include <string>
#include <utility>
namespace holoscan {
namespace inference {

// File-local mutex used by InferContext to serialize modifications
// to the global g_managers map. Previously in an anonymous namespace at the
// top of infer_manager.cpp; moved here because InferContext methods are the
// only consumers.
namespace {
std::mutex g_managers_mutex;
}  // namespace

InferStatus InferContext::execute_inference(std::shared_ptr<InferenceSpecs>& inference_specs,
                                            cudaStream_t cuda_stream) {
  InferStatus status = InferStatus();

  // Fast path: if the manager pointer is already cached for this context (set up
  // during set_inference_params), use it directly and skip the global mutex + map lookup
  // entirely.
  if (cached_manager_) {
    try {
      if (inference_specs->gpu_resident_inference_) {
        status = cached_manager_->execute_gr_inference(inference_specs, cuda_stream);
      } else {
        status = cached_manager_->execute_inference(inference_specs, cuda_stream);
      }
    } catch (const std::exception& e) {
      status.set_code(holoinfer_code::H_ERROR);
      status.set_message(std::string("Inference manager, Error in inference execution: ") +
                         e.what());
    }
    return status;
  }

  // Slow path (only used before set_inference_params() populates the cache, or if
  // the manager was reset).
  std::lock_guard<std::mutex> lock(g_managers_mutex);
  if (g_managers.find(unique_id_) == g_managers.end()) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message("Inference manager, Error: Inference manager not created or is not set up.");
    return status;
  }

  try {
    g_manager = g_managers.at(unique_id_);

    if (inference_specs->gpu_resident_inference_) {
      status = g_manager->execute_gr_inference(inference_specs, cuda_stream);
    } else {
      status = g_manager->execute_inference(inference_specs, cuda_stream);
    }
  } catch (const std::exception& e) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message(std::string("Inference manager, Error in inference execution: ") + e.what());
    return status;
  }

  return status;
}

// Streamlined dispatch for the fast path.
//
// This entry point bypasses InferContext::execute_inference() and ManagerInfer::execute_inference()
// entirely, going straight to ManagerInfer::run_core_inference() for the named model. The only
// bookkeeping retained here is the activation/temporal/dynamic-dims guard — and all three are
// validated as "off" by InferenceOp::start() before fast_path_enabled_ flips to true. The
// frame_counter_ in ManagerInfer is intentionally not incremented (it is used only by
// temporal_map, which the fast path excludes).
InferStatus InferContext::execute_inference_fast(std::shared_ptr<InferenceSpecs>& inference_specs,
                                                 cudaStream_t cuda_stream,
                                                 const std::string& model_name) {
  InferStatus status;
  if (!cached_manager_) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message("execute_inference_fast called before set_inference_params");
    return status;
  }
  try {
    // Prefer the cached-pointer fast path when available. Falls back to
    // the string-keyed run_core_inference path if the cached pointers haven't been set
    // (multi-model setup) or the requested model_name doesn't match the cached one.
    if (cached_manager_->has_fast_pointers() && cached_manager_->fast_model_name() == model_name) {
      status = cached_manager_->run_core_inference_fast(
          inference_specs->data_per_tensor_, inference_specs->output_per_model_, cuda_stream);
    } else {
      status = cached_manager_->run_core_inference(model_name,
                                                   inference_specs->data_per_tensor_,
                                                   inference_specs->output_per_model_,
                                                   cuda_stream);
    }
  } catch (const std::exception& e) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message(std::string("execute_inference_fast: ") + e.what());
  }
  return status;
}

// Multi-model sequential fast-path entry. Falls back to execute_inference
// if the manager's seq-fast cache isn't populated.
InferStatus InferContext::execute_inference_seq_fast(std::shared_ptr<InferenceSpecs>& specs,
                                                     cudaStream_t cuda_stream) {
  InferStatus status;
  if (!cached_manager_) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message("execute_inference_seq_fast called before set_inference_params");
    return status;
  }
  try {
    if (cached_manager_->has_seq_fast_pointers()) {
      status = cached_manager_->run_core_inference_seq_fast(
          specs->data_per_tensor_, specs->output_per_model_, cuda_stream);
    } else {
      status = cached_manager_->execute_inference(specs, cuda_stream);
    }
  } catch (const std::exception& e) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message(std::string("execute_inference_seq_fast: ") + e.what());
  }
  return status;
}

// Multi-model parallel fast-path entry. Falls back to execute_inference
// if the manager's par-fast streams aren't allocated.
InferStatus InferContext::execute_inference_par_fast(std::shared_ptr<InferenceSpecs>& specs,
                                                     cudaStream_t cuda_stream) {
  InferStatus status;
  if (!cached_manager_) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message("execute_inference_par_fast called before set_inference_params");
    return status;
  }
  try {
    if (cached_manager_->has_par_fast_pointers()) {
      status = cached_manager_->run_core_inference_par_fast(
          specs->data_per_tensor_, specs->output_per_model_, cuda_stream);
    } else {
      status = cached_manager_->execute_inference(specs, cuda_stream);
    }
  } catch (const std::exception& e) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message(std::string("execute_inference_par_fast: ") + e.what());
  }
  return status;
}

// InferContext entry points for the three dynamic-shape fast paths.
// Each routes into the matching ManagerInfer::run_core_inference_dyn_* method.
InferStatus InferContext::execute_inference_dyn_fast(std::shared_ptr<InferenceSpecs>& specs,
                                                     cudaStream_t cuda_stream,
                                                     const std::string& model_name) {
  InferStatus status;
  if (!cached_manager_) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message("execute_inference_dyn_fast called before set_inference_params");
    return status;
  }
  try {
    if (cached_manager_->has_fast_pointers() && cached_manager_->fast_model_name() == model_name) {
      status = cached_manager_->run_core_inference_dyn_fast(specs, cuda_stream);
    } else {
      // Cached pointers absent for this model — defer to canonical path.
      status = cached_manager_->execute_inference(specs, cuda_stream);
    }
  } catch (const std::exception& e) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message(std::string("execute_inference_dyn_fast: ") + e.what());
  }
  return status;
}

InferStatus InferContext::execute_inference_dyn_seq_fast(std::shared_ptr<InferenceSpecs>& specs,
                                                         cudaStream_t cuda_stream) {
  InferStatus status;
  if (!cached_manager_) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message("execute_inference_dyn_seq_fast called before set_inference_params");
    return status;
  }
  try {
    if (cached_manager_->has_seq_fast_pointers()) {
      status = cached_manager_->run_core_inference_dyn_seq_fast(specs, cuda_stream);
    } else {
      status = cached_manager_->execute_inference(specs, cuda_stream);
    }
  } catch (const std::exception& e) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message(std::string("execute_inference_dyn_seq_fast: ") + e.what());
  }
  return status;
}

InferStatus InferContext::execute_inference_dyn_par_fast(std::shared_ptr<InferenceSpecs>& specs,
                                                         cudaStream_t cuda_stream) {
  InferStatus status;
  if (!cached_manager_) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message("execute_inference_dyn_par_fast called before set_inference_params");
    return status;
  }
  try {
    if (cached_manager_->has_par_fast_pointers()) {
      status = cached_manager_->run_core_inference_dyn_par_fast(specs, cuda_stream);
    } else {
      status = cached_manager_->execute_inference(specs, cuda_stream);
    }
  } catch (const std::exception& e) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message(std::string("execute_inference_dyn_par_fast: ") + e.what());
  }
  return status;
}

InferStatus InferContext::set_inference_params(std::shared_ptr<InferenceSpecs>& inference_specs) {
  std::lock_guard<std::mutex> lock(g_managers_mutex);
  InferStatus status = InferStatus();

  try {
    auto multi_model_map = inference_specs->get_path_map();

    if (multi_model_map.size() == 0) {
      status.set_code(holoinfer_code::H_ERROR);
      status.set_message("Inference manager, Error: Multi modal map cannot be empty in setup.");
      return status;
    }

    std::string unique_id_name("");
    for (auto& [model_name, _] : multi_model_map) {
      unique_id_name += model_name + "_[]_";
    }

    if (g_managers.find(unique_id_name) != g_managers.end()) {
      status.set_code(holoinfer_code::H_ERROR);
      status.set_message(
          "Inference manager, Error: A manager with the same unique ID already exists.");
      HOLOSCAN_LOG_ERROR(
          "Inference manager setup error: model keywords are repeated in multiple instances of "
          "inference. All model instances must have unique keyword in the configuration file.");
      return status;
    }

    HOLOSCAN_LOG_INFO("Inference context ID: {}", unique_id_name);
    // Configure this context's manager while holding the map lock, then publish it only after
    // setup succeeds. No shared placeholder is visible to another concurrently starting context.
    auto manager = std::make_shared<ManagerInfer>();
    status = manager->set_inference_params(inference_specs);

    if (status.get_code() == holoinfer_code::H_SUCCESS) {
      unique_id_ = std::move(unique_id_name);
      cached_manager_ = manager;
      g_managers.emplace(unique_id_, std::move(manager));
    }
  } catch (const std::exception& e) {
    status.set_code(holoinfer_code::H_ERROR);
    status.set_message(std::string("Inference manager, Error in inference setup: ") + e.what());
    return status;
  }

  return status;
}

InferContext::InferContext() = default;

InferContext::~InferContext() {
  std::lock_guard<std::mutex> lock(g_managers_mutex);
  // Clear the cached pointer BEFORE we erase from g_managers, so that any other
  // thread that might still hold a reference to this InferContext sees the cache invalidate
  // first and falls back to the slow path (which acquires the lock and observes the missing
  // entry).
  cached_manager_.reset();
  if (g_managers.find(unique_id_) != g_managers.end()) {
    g_manager = g_managers.at(unique_id_);
    g_manager.reset();
    g_managers.erase(unique_id_);
  }
}

// Use the cached manager pointer if available so we can skip the global mutex.
// In typical Holoscan operator usage, set_inference_params() has run before any of these
// queries, so cached_manager_ is always populated. The slow fallback path is kept for safety
// in case a caller queries dimensions before setup completes — but that should not happen on
// the hot per-compute() call path of InferenceOp.
DimType InferContext::get_output_dimensions() const {
  if (cached_manager_) {
    return cached_manager_->get_output_dimensions();
  }
  std::lock_guard<std::mutex> lock(g_managers_mutex);
  g_manager = g_managers.at(unique_id_);
  return g_manager->get_output_dimensions();
}

DimType InferContext::get_input_dimensions() const {
  if (cached_manager_) {
    return cached_manager_->get_input_dimensions();
  }
  std::lock_guard<std::mutex> lock(g_managers_mutex);
  g_manager = g_managers.at(unique_id_);
  return g_manager->get_input_dimensions();
}

}  // namespace inference
}  // namespace holoscan
