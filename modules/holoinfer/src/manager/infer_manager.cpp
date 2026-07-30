/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// infer_manager.cpp is the main entry point for the inference manager. It has been modularized
// into five sibling files under this directory:
//
//   * infer_manager.cpp             — constructor, destructor, cleanup (this file)
//   * infer_manager_setup.cpp       — ManagerInfer::set_inference_params (backend
//                                     plugin loading, per-model context creation,
//                                     multi-GPU / DLA / temporal / activation setup,
//                                     fast-path cache population).
//   * infer_manager_dispatch.cpp    — canonical dispatch paths: run_core_inference,
//                                     run_core_gr_inference, execute_inference,
//                                     execute_gr_inference.
//   * infer_manager_fast_paths.cpp  — opt-in fast paths: run_core_inference_fast,
//                                     _seq_fast, _par_fast, _indexed, prewarm_par_fast,
//                                     and the dynamic-shape variants
//                                     _dyn_fast / _dyn_seq_fast / _dyn_par_fast.
//   * infer_context.cpp             — public InferContext facade: set_inference_params,
//                                     execute_inference / _fast / _seq_fast / _par_fast /
//                                     _dyn_fast / _dyn_seq_fast / _dyn_par_fast, and
//                                     the get_*_dimensions accessors.
//
// The class layout in infer_manager.hpp is unchanged; only the .cpp is split.

#include "infer_manager.hpp"

// dlclose used by cleanup() when releasing backend plugin handles.
#include <dlfcn.h>

namespace holoscan {
namespace inference {

ManagerInfer::ManagerInfer() {}

void ManagerInfer::cleanup() {
  for (auto& [_, context] : holo_infer_context_) {
    if (context) {
      context->cleanup();
    }
  }
  holo_infer_context_.clear();
  infer_param_.clear();

  for (auto* handle : backend_plugin_handles_) {
    dlclose(handle);
  }
  backend_plugin_handles_.clear();

  if (cuda_event_) {
    cudaEventDestroy(cuda_event_);
    cuda_event_ = nullptr;
  }
  // Release par-fast events. Per-model streams are executor-managed
  // via allocate_cuda_stream_, so we don't destroy them — the executor's pool owns
  // them. par_outer_event_ and par_done_events_ are created here and must be freed.
  if (par_outer_event_) {
    cudaEventDestroy(par_outer_event_);
    par_outer_event_ = nullptr;
  }
  for (auto& e : par_done_events_)
    if (e)
      cudaEventDestroy(e);
  par_done_events_.clear();
  par_streams_.clear();
  all_params_.clear();
  all_ctx_.clear();
  all_model_names_.clear();
  all_indata_.clear();
  all_outdata_.clear();
  seq_fast_data_cached_ = false;
  fast_multi_model_ = false;
}

ManagerInfer::~ManagerInfer() {
  cleanup();
}

}  // namespace inference
}  // namespace holoscan
