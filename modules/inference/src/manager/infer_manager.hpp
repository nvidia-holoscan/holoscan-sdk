/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <holoinfer/holoinfer.hpp>
#include <holoinfer/holoinfer_buffer.hpp>
#include <holoinfer/holoinfer_constants.hpp>
#include <holoinfer/holoinfer_utils.hpp>
#include <infer/infer.hpp>
#include <utils/work_queue.hpp>

#include <infer/trt/core.hpp>
#include <params/infer_param.hpp>

namespace holoscan {
namespace inference {
/**
 * @brief Manager class for inference
 */
class ManagerInfer {
 public:
  /**
   * @brief Default Constructor
   */
  ManagerInfer();

  /**
   * @brief Destructor
   */
  ~ManagerInfer();

  /**
   * @brief Create inference settings and memory
   *
   * @param inference_specs specifications for inference
   *
   * @return InferStatus with appropriate code and message
   */
  InferStatus set_inference_params(std::shared_ptr<InferenceSpecs>& inference_specs);

  /**
   * @brief Prepares and launches single/multiple inference
   *
   * The provided CUDA stream is used to prepare the input data and will be used to operate on the
   * output data, any execution of CUDA work should be in sync with this stream.
   *
   * @param inference_specs specifications for inference
   * @param cuda_stream CUDA stream
   *
   * @return InferStatus with appropriate code and message
   */
  InferStatus execute_inference(std::shared_ptr<InferenceSpecs>& inference_specs,
                                cudaStream_t cuda_stream);

  /**
   * @brief Executes GPU Resident inference for a particular model and generates inferred data
   *
   * @param inference_specs specifications for inference
   * @param cuda_stream CUDA stream
   *
   * @return InferStatus with appropriate code and message
   */
  InferStatus execute_gr_inference(std::shared_ptr<InferenceSpecs>& inference_specs,
                                   cudaStream_t cuda_stream);

  /**
   * @brief Runs the GPU Resident inference for a particular model
   *
   * @param model_name Input model to do the inference on
   * @param in_buffer Input buffer
   * @param out_buffer Output buffer
   * @param cuda_stream CUDA stream
   * @return InferStatus with appropriate code and message
   */
  InferStatus run_core_gr_inference(const std::string& model_name, void* in_buffer,
                                    void* out_buffer, cudaStream_t cuda_stream);

  /**
   * @brief Executes Core inference for a particular model and generates inferred data
   * The provided CUDA stream is used to prepare the input data and will be used to operate on the
   * output data, any execution of CUDA work should be in sync with this stream.
   *
   * @param model_name Input model to do the inference on
   * @param permodel_preprocess_data Input DataMap with model name as key and DataBuffer as value
   * @param permodel_output_data Output DataMap with tensor name as key and DataBuffer as value
   * @param cuda_stream CUDA stream
   *
   * @return InferStatus with appropriate code and message
   */
  InferStatus run_core_inference(const std::string& model_name,
                                 const DataMap& permodel_preprocess_data,
                                 const DataMap& permodel_output_data, cudaStream_t cuda_stream);
  /**
   * @brief Cleans up internal context per model
   *
   */
  void cleanup();

  /**
   * @brief Get input dimension per model
   *
   * @return Map with model name as key and dimension as value
   */
  DimType get_input_dimensions() const;

  /**
   * @brief Get output dimension per tensor
   *
   * @return Map with tensor name as key and dimension as value
   */
  DimType get_output_dimensions() const;

  /**
   * @brief Streamlined single-model dispatch for canonical InferenceOp.
   *
   * @param input_data Input data map
   * @param output_data Output data map
   * @param cuda_stream CUDA stream
   * @return InferStatus with appropriate code and message
   */
  InferStatus run_core_inference_fast(const DataMap& input_data, const DataMap& output_data,
                                      cudaStream_t cuda_stream);

  /// True after set_inference_params has populated the single-model fast-path cached
  /// pointers. Consumed by InferContext::execute_inference_fast / _dyn_fast to decide
  /// whether the fast dispatch or the map-keyed fallback is used.
  bool has_fast_pointers() const noexcept { return fast_param_ != nullptr; }

  /// Name of the single model cached for the fast path (set in set_inference_params when
  /// path_map.size() == 1). Returns an empty string if the fast pointers are not populated.
  const std::string& fast_model_name() const noexcept { return fast_model_name_; }

  /**
   * @brief Streamlined multi-model SEQUENTIAL dispatch using cached all_params_/all_ctx_ pointers
   * and pre-built I/O DataBuffer vectors.
   *
   * @param input_data Input data map
   * @param output_data Output data map
   * @param cuda_stream CUDA stream
   * @return InferStatus with appropriate code and message
   */
  InferStatus run_core_inference_seq_fast(const DataMap& input_data, const DataMap& output_data,
                                          cudaStream_t cuda_stream);

  /**
   * @brief Streamlined multi-model PARALLEL dispatch using per-model CUDA streams + completion
   * events; bypasses work_queue. Falls back if par_streams_ is empty.
   *
   * @param input_data Input data map
   * @param output_data Output data map
   * @param cuda_stream CUDA stream
   * @return InferStatus with appropriate code and message
   */
  InferStatus run_core_inference_par_fast(const DataMap& input_data, const DataMap& output_data,
                                          cudaStream_t cuda_stream);

  /// True after set_inference_params has populated the multi-model fast-path pointer cache.
  bool has_seq_fast_pointers() const noexcept { return !all_params_.empty(); }

  /// True after set_inference_params has set up the per-model parallel streams + events.
  bool has_par_fast_pointers() const noexcept { return !par_streams_.empty(); }

  /**
   * @brief Dynamic-shape single-model dispatch.
   *
   * @param specs Inference specs
   * @param cuda_stream CUDA stream
   * @return InferStatus with appropriate code and message
   */
  InferStatus run_core_inference_dyn_fast(std::shared_ptr<InferenceSpecs>& specs,
                                          cudaStream_t cuda_stream);

  /**
   * @brief Dynamic-shape multi-model sequential dispatch.
   *
   * @param specs Inference specs
   * @param cuda_stream CUDA stream
   * @return InferStatus with appropriate code and message
   */
  InferStatus run_core_inference_dyn_seq_fast(std::shared_ptr<InferenceSpecs>& specs,
                                              cudaStream_t cuda_stream);

  /**
   * @brief Dynamic-shape multi-model parallel dispatch.
   *
   * @param specs Inference specs
   * @param cuda_stream CUDA stream
   * @return InferStatus with appropriate code and message
   */
  InferStatus run_core_inference_dyn_par_fast(std::shared_ptr<InferenceSpecs>& specs,
                                              cudaStream_t cuda_stream);

 private:
  /**
   * @brief Indexed worker for the work_queue parallel fallback (used when not all backends are
   * TRT). Builds LOCAL per-worker I/O vectors so concurrent invocations don't race on
   * shared state.
   *
   * @param model_index Model index
   * @param input_data Input data map
   * @param output_data Output data map
   * @param cuda_stream CUDA stream
   * @return InferStatus with appropriate code and message
   */
  InferStatus run_core_inference_indexed(size_t model_index, const DataMap& input_data,
                                         const DataMap& output_data, cudaStream_t cuda_stream);

  /**
   * @brief Synthetic forward passes on every per-model parallel stream so CUDA-graph capture
   * happens during setup, not in the first compute() call.
   *
   * @param passes Number of passes
   * @return InferStatus with appropriate code and message
   */
  InferStatus prewarm_par_fast(int passes = 3);

  /**
   * @brief Lazily builds the multi-model I/O DataBuffer pointer cache (all_indata_/all_outdata_)
   * used by both run_core_inference_seq_fast and run_core_inference_par_fast. Pure tensor-name
   * lookups only — issues no CUDA work, so it is safe to call regardless of which CUDA stream
   * (if any) the caller will subsequently dispatch on. Idempotent: a no-op once
   * seq_fast_data_cached_ is true.
   *
   * @param input_data Input data map
   * @param output_data Output data map
   * @return InferStatus with appropriate code and message
   */
  InferStatus build_seq_fast_cache(const DataMap& input_data, const DataMap& output_data);

  /**
   * @brief Dispatches one model on `stream` with the fallback ladder's runtime CUDA-Graph
   *
   * @param model_index Index into all_ctx_/all_params_/all_model_names_.
   * @param indata Input DataBuffer vector for this model.
   * @param outdata Output DataBuffer vector for this model.
   * @param stream CUDA stream to enqueue on.
   * @return InferStatus with appropriate code and message.
   */
  InferStatus dispatch_on_stream_with_graph_fallback(
      size_t model_index, const std::vector<std::shared_ptr<DataBuffer>>& indata,
      std::vector<std::shared_ptr<DataBuffer>>& outdata, cudaStream_t stream);

  /**
   * @brief Reacts to a failed cuCtxPushCurrent() reported by a backend (see
   * manager/fallback_runtime.hpp) by dropping this instance's green-context-isolated assumption
   * and re-arming the capture-serialization mutex window. One relaxed atomic load per tick, and
   * only while this instance still believes it is isolated.
   */
  void check_green_context_isolation_lost();

 private:
  /// Flag to infer models in parallel. Defaults to False
  bool parallel_processing_ = false;

  /// Flag to demonstrate if input data buffer is on cuda
  bool cuda_buffer_in_ = false;

  /// Flag to demonstrate if output data buffer will be on cuda
  bool cuda_buffer_out_ = false;

  /// Map storing parameters per model
  std::map<std::string, std::unique_ptr<Params>> infer_param_;

  /// Map storing Inference context per model
  std::map<std::string, std::unique_ptr<InferBase>> holo_infer_context_;

  /// Plugin handles backing dlopened inference contexts. Closed after contexts are destroyed.
  std::vector<void*> backend_plugin_handles_;

  // @brief Pre-resolved raw pointers for the single-model fast path.
  Params* fast_param_ = nullptr;
  InferBase* fast_ctx_ = nullptr;
  std::string fast_model_name_;

  // @brief Pre-resolved per-tensor in/out DataBuffer pointers for the single-model fast path.
  // Built lazily on first run_core_inference_fast call.
  std::vector<std::shared_ptr<DataBuffer>> fast_indata_;
  std::vector<std::shared_ptr<DataBuffer>> fast_outdata_;
  bool fast_data_cached_ = false;

  // @brief Multi-model fast-path pointer / vector cache]
  // Parallel arrays indexed by model position (insertion order from path_map).
  // Populated in set_inference_params() when fast_track_ is true AND the config is
  // eligible (no dla map, no GR inference, no dynamic dims).
  std::vector<Params*> all_params_;
  std::vector<InferBase*> all_ctx_;
  std::vector<std::string> all_model_names_;
  std::vector<std::vector<std::shared_ptr<DataBuffer>>> all_indata_;
  std::vector<std::vector<std::shared_ptr<DataBuffer>>> all_outdata_;
  bool seq_fast_data_cached_ = false;

  // @brief Inline parallel TRT dispatch — per-model streams + events]
  // Populated when fast_track_ AND parallel_processing_ AND all backends are TRT AND the
  // executor's allocate_cuda_stream callback can supply N distinct streams. par_streams_
  // empty means par-fast is unavailable and dispatcher will demote to work_queue.
  std::vector<cudaStream_t> par_streams_;
  std::vector<cudaEvent_t> par_done_events_;
  cudaEvent_t par_outer_event_ = nullptr;

  // @brief Mirror of InferenceSpecs::fast_multi_model_
  bool fast_multi_model_ = false;

  // @brief Mirror of InferenceSpecs::green_context_isolated_, cached at set_inference_params()
  // time since run_core_inference_par_fast() does not receive the InferenceSpecs. True exactly
  // when InferenceOp::start()'s decide_capture_protection() chose kGreenContextIsolated: every
  // multi-model fast-path TrtInfer instance was bound (see infer_manager_setup.cpp) to a
  // distinct CudaGreenContextPool partition's CUcontext, so concurrent CUDA-Graph capture on
  // different partitions cannot cross-invalidate by CUDA-context-isolation semantics. When
  // true, capture_serialization_mutex_ is redundant and is skipped in
  // run_core_inference_par_fast() to keep the hot path lock-free.
  bool green_context_isolated_ = false;

  // @brief Process-wide (class-static, shared by every ManagerInfer instance) mutex that
  // serializes the CUDA-Graph capture window for run_core_inference_par_fast(). kParFast's
  // primary protection is the setup-time prewarm (prewarm_par_fast(), single-threaded, so
  // capture naturally happens serialized before any concurrent compute() call). This mutex is
  // a runtime safety net for the case where TRT's stream-capture-begin/enqueueV3/
  // capture-end/instantiate sequence still runs during one of the first few
  // run_core_inference_par_fast() calls (see infer_manager_fast_paths.cpp): it is held for the
  // ENTIRE per-model dispatch loop of those calls, across ALL ManagerInfer instances in the
  // process (hence static, not per-instance) — never released mid-capture the way a
  // compute()-scoped mutex would be. A compute()-scoped mutex is not durable in the general case
  // because TRT's capture window can extend past the point where such a mutex is released,
  // leaving a partially-captured graph vulnerable to concurrent stream-capture invalidation.
  static std::mutex capture_serialization_mutex_;

  // @brief Number of run_core_inference_par_fast() calls made on THIS instance so far, used to
  // bound the capture_serialization_mutex_ lock window to the first kCaptureProtectionTicks
  // calls (matching prewarm_par_fast's default pass count). After that window, prewarm (or the
  // already-captured/updated graph) is assumed to have completed and the mutex is skipped to
  // keep the steady-state hot path lock-free.
  int par_fast_capture_ticks_ = 0;

  // @brief Per-model "CUDA graphs were disabled by the fallback ladder" cache, indexed like
  // all_ctx_.
  std::vector<uint8_t> runtime_graph_disabled_;

  /// Map storing input dimension per model
  DimType models_input_dims_;

  /// Frame counter into the inference engine
  unsigned int frame_counter_ = 0;

  /// Data transfer GPU. Always 0 (default GPU); multi-GPU is not supported.
  int device_gpu_dt_ = 0;

  /// CUDA event on data transfer GPU, used to synchronize inference execution with data transfer.
  cudaEvent_t cuda_event_ = nullptr;

  /// Map storing inferred output dimension per tensor
  DimType models_output_dims_;

  /// Work queue use for parallel processing
  std::unique_ptr<WorkQueue> work_queue_;

  /// Execution plan for the inference
  std::vector<std::vector<std::string>> execution_plan_;

  /// Map storing Backends supported with holoinfer mapping
  // NOLINTNEXTLINE(cert-err58-cpp)
  inline static std::map<std::string, holoinfer_backend> supported_backend_{
      {"trt", holoinfer_backend::h_trt}};
};

}  // namespace inference
}  // namespace holoscan
