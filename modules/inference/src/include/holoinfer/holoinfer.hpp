/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda_runtime.h>

#include <memory>
#include <string>

#include "holoinfer_buffer.hpp"

namespace holoscan {
namespace inference {

// Forward-declared ManagerInfer class
class ManagerInfer;

/**
 * Inference Context class
 */
class HOLOINFER_EXPORT InferContext {
 public:
  InferContext();
  ~InferContext();
  /**
   * Set Inference parameters
   *
   * @param inference_specs   Pointer to inference specifications
   *
   * @return InferStatus with appropriate holoinfer_code and message.
   */
  InferStatus set_inference_params(std::shared_ptr<InferenceSpecs>& inference_specs);

  /**
   * Executes the inference
   * Toolkit supports one input per model, in float32 type.
   * The provided CUDA stream is used to prepare the input data and will be used to operate on the
   * output data, any execution of CUDA work should be in sync with this stream.
   *
   * @param inference_specs   Pointer to inference specifications
   * @param cuda_stream CUDA stream
   *
   * @return InferStatus with appropriate holoinfer_code and message.
   */
  InferStatus execute_inference(std::shared_ptr<InferenceSpecs>& inference_specs,
                                cudaStream_t cuda_stream = 0);

  /**
   * Streamlined inference dispatch for the single-model / static-shape / non-parallel
   * configuration.
   *
   * @param inference_specs    The same specs object used by execute_inference().
   * @param cuda_stream        Stream to record/wait on for cross-stream sync.
   * @param model_name         Model key into infer_param_ / input_map_.
   * @return InferStatus.
   */
  InferStatus execute_inference_fast(std::shared_ptr<InferenceSpecs>& inference_specs,
                                     cudaStream_t cuda_stream, const std::string& model_name);

  /**
   * Multi-model SEQUENTIAL fast-path dispatch.
   *
   * @param inference_specs    The same specs object used by execute_inference().
   * @param cuda_stream        Stream to record/wait on for cross-stream sync.
   * @return InferStatus.
   */
  InferStatus execute_inference_seq_fast(std::shared_ptr<InferenceSpecs>& inference_specs,
                                         cudaStream_t cuda_stream);

  /**
   * Multi-model PARALLEL fast-path dispatch.
   *
   * @param inference_specs    The same specs object used by execute_inference().
   * @param cuda_stream        Stream to record/wait on for cross-stream sync.
   * @return InferStatus.
   */
  InferStatus execute_inference_par_fast(std::shared_ptr<InferenceSpecs>& inference_specs,
                                         cudaStream_t cuda_stream);

  /**
   * Streamlined single-model dispatch for dynamic-shape pipelines.
   *
   * @param inference_specs    The same specs object used by execute_inference().
   * @param cuda_stream        Stream to record/wait on for cross-stream sync.
   * @param model_name         Model key into infer_param_ / input_map_.
   * @return InferStatus.
   */
  InferStatus execute_inference_dyn_fast(std::shared_ptr<InferenceSpecs>& inference_specs,
                                         cudaStream_t cuda_stream, const std::string& model_name);

  /**
   * Streamlined multi-model sequential dispatch for dynamic-shape pipelines.
   *
   * @param inference_specs    The same specs object used by execute_inference().
   * @param cuda_stream        Stream to record/wait on for cross-stream sync.
   * @return InferStatus.
   */
  InferStatus execute_inference_dyn_seq_fast(std::shared_ptr<InferenceSpecs>& inference_specs,
                                             cudaStream_t cuda_stream);

  /**
   * Streamlined multi-model parallel dispatch for dynamic-shape pipelines.
   *
   * @param inference_specs    The same specs object used by execute_inference().
   * @param cuda_stream        Stream to record/wait on for cross-stream sync.
   * @return InferStatus.
   */
  InferStatus execute_inference_dyn_par_fast(std::shared_ptr<InferenceSpecs>& inference_specs,
                                             cudaStream_t cuda_stream);

  /**
   * Gets output dimension per model
   *
   * @return Map of model as key mapped to the output dimension (of inferred data)
   */
  DimType get_output_dimensions() const;

  /**
   * Gets input dimension per model
   *
   * @return Map of model as key mapped to the input dimensions
   */
  DimType get_input_dimensions() const;

 private:
  std::string unique_id_;

  // @brief Cached ManagerInfer pointer.
  std::shared_ptr<ManagerInfer> cached_manager_;
};

}  // namespace inference
}  // namespace holoscan
