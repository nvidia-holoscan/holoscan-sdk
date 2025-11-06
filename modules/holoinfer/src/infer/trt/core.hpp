/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>

#include <holoinfer_constants.hpp>
#include <infer/infer.hpp>
#include "utils.hpp"

namespace holoscan {
namespace inference {
/**
 * Class to execute TensorRT based inference
 * */
class TrtInfer : public InferBase {
 public:
  /**
   * @brief Constructor
   */
  TrtInfer(const std::string& model_path, const std::string& model_name,
           const std::vector<std::string>& trt_opt_profile, int device_id, int device_id_dt,
           bool enable_fp16, bool enable_cuda_graphs, int32_t dla_core, bool dla_gpu_fallback,
           bool is_engine_path, bool cuda_buf_in, bool cuda_buf_out,
           std::function<cudaStream_t(int32_t device_id)> allocate_cuda_stream);

  /**
   * @brief Destructor
   */
  ~TrtInfer();

  /**
   * @brief Does the Core inference with TRT backend
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
  InferStatus do_inference(const std::vector<std::shared_ptr<DataBuffer>>& input_data,
                           std::vector<std::shared_ptr<DataBuffer>>& output_buffer,
                           cudaEvent_t cuda_event_data, cudaEvent_t* cuda_event_inference);

  /**
   * @brief Get input data dimensions to the model
   * @return Vector of values as dimension
   * */
  std::vector<std::vector<int64_t>> get_input_dims() const;

  /**
   * @brief Updates the dimensions per tensor in case of dynamic inputs.
   * Using the input Holoscan tensors and their dimension mapping, the internal input size vector is
   * updated
   *
   * @param input_tensors Vector of input Holoscan tensor names
   * @param dims_per_tensor Map storing the dimensions as values and Holoscan tensor names as keys.
   * @return true if the dynamic input dimensions were successfully updated, false otherwise
   */
  bool set_dynamic_input_dimension(const std::vector<std::string>& input_tensors,
                                   const std::map<std::string, std::vector<int>>& dims_per_tensor);

  /**
   * @brief Get output data dimensions from the model
   * @return Vector of input dimensions. Each dimension is a vector of int64_t corresponding to
   *         the shape of the input tensor.
   * */
  std::vector<std::vector<int64_t>> get_output_dims() const;

  /**
   * @brief Get input data types from the model
   * @return Vector of input dimensions. Each dimension is a vector of int64_t corresponding to
   *         the shape of the input tensor.
   * */
  std::vector<holoinfer_datatype> get_input_datatype() const;

  /**
   * @brief Get output data types from the model
   * @return Vector of values as datatype per output tensor
   * */
  std::vector<holoinfer_datatype> get_output_datatype() const;

  void cleanup() {}

 private:
  /// @brief Path to onnx model file
  std::string model_path_;

  /// @brief A Unique identifier for the model
  std::string model_name_;

  /// @brief Dimensions of input buffer
  std::vector<std::vector<int64_t>> input_dims_;

  /// @brief Dimensions of output buffer
  std::vector<std::vector<int64_t>> output_dims_;

  /// @brief Vector of input data types
  std::vector<holoinfer_datatype> in_data_types_;

  /// @brief Vector of output data types
  std::vector<holoinfer_datatype> out_data_types_;

  /// @brief Vector of trt optimization profile
  std::vector<std::string> trt_opt_profile_;

  /// @brief Use FP16 in TRT engine file generation
  bool enable_fp16_;

  ///  @brief Use CUDA Graphs
  bool enable_cuda_graphs_;

  /// @brief The DLA core index to execute the engine on, starts at 0. Set to -1 to disable DLA.
  int32_t dla_core_;

  /// @brief If DLA is enabled, use the GPU if a layer cannot be executed on DLA. If the fallback is
  /// disabled, engine creation will fail if a layer cannot executed on DLA.
  bool dla_gpu_fallback_;

  /// @brief Flag showing if input buffer is on cuda
  bool cuda_buf_in_;

  /// @brief Flag showing if output buffer will be on cuda
  bool cuda_buf_out_;

  /// @brief Flag showing if input path map is path to engine file
  bool is_engine_path_;

  /// @brief Function to allocate a CUDA stream (optional)
  std::function<cudaStream_t(int32_t device_id)> allocate_cuda_stream_;

  /**
   * @brief Parameter initialization
   */
  bool initialize_parameters();

  /**
   * @brief Load and prepare the (trt converted) network for inference
   */
  bool load_engine();

  /// @brief Pointer to TRT cuda engine
  std::unique_ptr<nvinfer1::ICudaEngine> engine_ = nullptr;

  /// @brief Pointer to TRT execution context
  std::unique_ptr<nvinfer1::IExecutionContext> context_ = nullptr;

  /// @brief Options to generate TRT engine file from onnx
  NetworkOptions network_options_;

  /// @brief GPU ID to do inference on
  int device_id_;
  /// @brief Logger object
  Logger logger_;

  /// @brief Pointer to input DataBuffer. Holds both host and device buffers as specified
  std::shared_ptr<DataBuffer> input_buffer_;

  /// @brief Pointer to output DataBuffer. Holds both host and device buffers as specified
  std::shared_ptr<DataBuffer> output_buffer_;

  /// @brief Generated engine file path. The extension is unique per GPU model
  std::string engine_path_;

  /// Cuda stream
  cudaStream_t cuda_stream_ = nullptr;
  /// CUDA event for device
  cudaEvent_t cuda_event_ = nullptr;

  /// This is set when the model is executed the first time, used for CUDA graph logic
  bool first_phase_ = true;
  /// CUDA graph instance
  cudaGraphExec_t cuda_graph_instance_ = nullptr;

  /// @brief Inference runtime
  std::unique_ptr<nvinfer1::IRuntime> infer_runtime_;
};

}  // namespace inference
}  // namespace holoscan
