/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef _HOLOSCAN_INFER_CORE_H
#define _HOLOSCAN_INFER_CORE_H

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <holoinfer_buffer.hpp>

namespace holoscan {
namespace inference {
/**
 * @brief Base Inference Class
 */
class InferBase {
 public:
  /**
   * @brief Default destructor
   * */
  virtual ~InferBase() = default;

  /**
   * @brief Does the Core inference
   * The provided CUDA data event is used to prepare the input data any execution of CUDA work
   * should be in sync with this event. If the inference is using CUDA it should record a CUDA
   * event and pass it back in `cuda_event_inference`.
   *
   * @param input_data Input DataBuffer
   * @param output_buffer Output DataBuffer, is populated with inferred results
   * @param cuda_event_data CUDA event recorded after data transfer
   * @param cuda_event_inference CUDA event recorded after inference
   * @return InferStatus
   * */
  virtual InferStatus do_inference(const std::vector<std::shared_ptr<DataBuffer>>& input_data,
                                   std::vector<std::shared_ptr<DataBuffer>>& output_buffer,
                                   cudaEvent_t cuda_event_data, cudaEvent_t* cuda_event_inference) {
    return InferStatus();
  }

  /**
   * @brief Stream-Shared Sequential Dispatch
   * @param input_data Input DataBuffer(s)
   * @param output_buffer Output DataBuffer(s) (populated in-place)
   * @param stream The CUDA stream to enqueue inference work on.
   * @return InferStatus
   */
  virtual InferStatus do_inference_on_stream(
      const std::vector<std::shared_ptr<DataBuffer>>& input_data,
      std::vector<std::shared_ptr<DataBuffer>>& output_buffer, cudaStream_t stream) {
    (void)stream;  // Default: delegate to event-based path. Subclasses override.
    cudaEvent_t inference_event = nullptr;
    cudaEvent_t data_event = nullptr;
    cudaEventCreateWithFlags(&data_event, cudaEventDisableTiming);
    cudaEventRecord(data_event, stream);
    auto rc = do_inference(input_data, output_buffer, data_event, &inference_event);
    if (inference_event)
      cudaStreamWaitEvent(stream, inference_event);
    cudaEventDestroy(data_event);
    return rc;
  }

  /**
   * @brief Does the GPU Resident inference
   * @param input_buffer Input buffer on GPU
   * @param output_buffer Output buffer
   * @param cuda_stream CUDA stream
   */
  virtual void do_gr_inference(void* input_buffer, void* output_buffer, cudaStream_t cuda_stream) {
    return;
  }

  /**
   * @brief Initializes the GPU Resident inference
   * @param input_buffer Input buffer on GPU
   * @param output_buffer Output buffer
   */
  virtual void init_gr_inference(void* input_buffer, void* output_buffer) { return; }

  /**
   * @brief Updates the dimensions per tensor in case of dynamic inputs.
   * Using the input Holoscan tensors and their dimension mapping, the internal input size vector is
   * updated
   *
   * @param input_tensors Vector of input Holoscan tensor names
   * @param dims_per_tensor Map storing the dimensions as values and Holoscan tensor names as keys.
   * @return true if the dynamic input dimensions were successfully updated, false otherwise
   */
  virtual bool set_dynamic_input_dimension(
      const std::vector<std::string>& input_tensors,
      const std::map<std::string, std::vector<int>>& dims_per_tensor) {
    return true;
  }

  /**
   * @brief Get input data dimensions to the model
   * @return Vector of values as dimension
   * */
  virtual std::vector<std::vector<int64_t>> get_input_dims() const { return {}; }

  /**
   * @brief Get output data dimensions from the model
   * @return Vector of output dimensions. Each dimension is a vector of int64_t corresponding to
   *         the shape of the output tensor.
   * */
  virtual std::vector<std::vector<int64_t>> get_output_dims() const { return {}; }

  /**
   * @brief Get input data types from the model
   * @return Vector of input dimensions. Each dimension is a vector of int64_t corresponding to
   *         the shape of the input tensor.
   * */
  virtual std::vector<holoinfer_datatype> get_input_datatype() const { return {}; }

  /**
   * @brief Get output data types from the model
   * @return Vector of values as datatype per output tensor
   * */
  virtual std::vector<holoinfer_datatype> get_output_datatype() const { return {}; }

  virtual void cleanup() {}
};

}  // namespace inference
}  // namespace holoscan
#endif
