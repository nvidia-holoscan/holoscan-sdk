/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_UTILS_HOLOINFER_UTILS_HPP
#define HOLOSCAN_UTILS_HOLOINFER_UTILS_HPP

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/core/io_context.hpp>
#include <holoscan/utils/cuda_stream_handler.hpp>

#include <holoinfer_buffer.hpp>

#include "gxf/core/entity.hpp"
#include "gxf/core/expected.hpp"

namespace HoloInfer = holoscan::inference;

namespace holoscan::utils {

/**
 * Persistent cache for output message and tensor allocations used by the cached variant of
 * transmit_data_per_model. Holds a reusable GXF entity and per-tensor metadata to minimise
 * per-frame overhead. Three cases are handled each frame:
 *   1. First use or buffer too small: full reshape (reallocate + update shape).
 *   2. Incoming fits in existing allocation but dims changed: wrapMemory to update shape
 *      metadata only — no free/alloc of the underlying buffer.
 *   3. Same dims as last frame: fast path, no tensor mutation at all.
 */
struct TensorTransmitCache {
  /// Persistent output entity. Invalid (falsy) until the first call to transmit_data_per_model.
  nvidia::gxf::Expected<nvidia::gxf::Entity> out_message{
      nvidia::gxf::Unexpected{GXF_UNINITIALIZED_VALUE}};
  /// Maximum element count that has been allocated for each output tensor.
  std::map<std::string, size_t> allocated_sizes;
  /// Dimension vector from the most recent frame for each output tensor, used to detect shape
  /// changes that require a wrapMemory call even when the element count has not grown.
  std::map<std::string, std::vector<int64_t>> last_dims;
};

/**
 * Buffer wrapping a GXF tensor
 */
class GxfTensorBuffer : public HoloInfer::Buffer {
 public:
  /**
   * @brief Constructor
   *
   * @param entity GXF entity holding the tensor
   * @param tensor GXF tensor
   */
  explicit GxfTensorBuffer(const holoscan::gxf::Entity& entity,
                           const nvidia::gxf::Handle<nvidia::gxf::Tensor>& tensor);
  GxfTensorBuffer() = delete;

  /// Buffer class virtual members implemented by this class
  ///@{
  void* data() override;
  size_t size() const override;
  size_t get_bytes() const override;
  void resize(size_t number_of_elements) override;
  ///@}

 private:
  holoscan::gxf::Entity entity_;
  nvidia::gxf::Handle<nvidia::gxf::Tensor> tensor_;
};

// shim around HoloInfer::report_error to return gxf_result_t enum instead of int
gxf_result_t report_error(const std::string& module, const std::string& submodule);

/**
 * Extracts data from GXF Receivers.
 *
 * This version of get_data_per_model uses the legacy CudaStreamHandler utility. Use the variant
 * without a cuda_stream_handler argument to use the built-in CudaObjectHandler instead.
 *
 * @param op_input Input context
 * @param in_tensors Input tensor names
 * @param data_per_input_tensor Map is updated with output tensor name as key mapped to data
 * buffer
 * @param dims_per_tensor Map is updated with tensor name as key mapped to dimension of input tensor
 * @param cuda_buffer_out Flag defining the location of output memory (Device or Host)
 * @param module Module that called for data extraction
 * @param context GXF execution context
 * @param cuda_stream_handler Cuda stream handler
 * @return GXF result code
 */
gxf_result_t get_data_per_model(InputContext& op_input, const std::vector<std::string>& in_tensors,
                                HoloInfer::DataMap& data_per_input_tensor,
                                std::map<std::string, std::vector<int>>& dims_per_tensor,
                                bool cuda_buffer_out, const std::string& module,
                                gxf_context_t& context, CudaStreamHandler& cuda_stream_handler);

/**
 * Extracts data from GXF Receivers.
 *
 * @param op_input Input context
 * @param in_tensors Input tensor names
 * @param data_per_input_tensor Map is updated with output tensor name as key mapped to data
 * buffer
 * @param dims_per_tensor Map is updated with tensor name as key mapped to dimension of input tensor
 * @param cuda_buffer_out Flag defining the location of output memory (Device or Host)
 * @param module Module that called for data extraction
 * @param cuda_stream_out Any stream used from the input port will be stored here.
 * @return GXF result code
 */
gxf_result_t get_data_per_model(InputContext& op_input, const std::vector<std::string>& in_tensors,
                                HoloInfer::DataMap& data_per_input_tensor,
                                std::map<std::string, std::vector<int>>& dims_per_tensor,
                                bool cuda_buffer_out, const std::string& module,
                                cudaStream_t& cuda_stream_out);

/**
 * Transmits multiple buffers via GXF Transmitters.
 *
 * This version of transmit_data_per_model uses the legacy CudaStreamHandler utility. Use the
 * variant without a cuda_stream_handler argument to use the built-in CudaObjectHandler instead.
 *
 * @param cont GXF context for transmission
 * @param model_to_tensor_map Map of model name as key, mapped to a vector of tensor names
 * @param input_data_map Map of tensor name as key, mapped to the data buffer as a vector
 * @param op_output Output context. Assume that the output port's name is "transmitter".
 * @param out_tensors Output tensor names
 * @param tensor_out_dims_map Map is updated with model name as key mapped to dimension of
 *                          output tensor as a vector
 * @param cuda_buffer_in Flag to demonstrate if memory storage of input buffers is on CUDA
 * @param cuda_buffer_out Flag to demonstrate if memory storage of output message is on CUDA
 * @param allocator_ GXF Memory allocator
 * @param module Module that called for data transmission
 * @param cuda_stream_handler Cuda stream handler
 * @return GXF result code
 */
gxf_result_t transmit_data_per_model(gxf_context_t& cont,
                                     const HoloInfer::MultiMappings& model_to_tensor_map,
                                     HoloInfer::DataMap& input_data_map, OutputContext& op_output,
                                     std::vector<std::string>& out_tensors,
                                     HoloInfer::DimType& tensor_out_dims_map, bool cuda_buffer_in,
                                     bool cuda_buffer_out,
                                     const nvidia::gxf::Handle<nvidia::gxf::Allocator>& allocator_,
                                     const std::string& module,
                                     CudaStreamHandler& cuda_stream_handler);

/**
 * Transmits multiple buffers via GXF Transmitters.
 *
 * @param cont GXF context for transmission
 * @param model_to_tensor_map Map of model name as key, mapped to a vector of tensor names
 * @param input_data_map Map of tensor name as key, mapped to the data buffer as a vector
 * @param op_output Output context. Assume that the output port's name is "transmitter".
 * @param out_tensors Output tensor names
 * @param tensor_out_dims_map Map is updated with model name as key mapped to dimension of
 *                          output tensor as a vector
 * @param cuda_buffer_in Flag to demonstrate if memory storage of input buffers is on CUDA
 * @param cuda_buffer_out Flag to demonstrate if memory storage of output message is on CUDA
 * @param allocator_ GXF Memory allocator
 * @param module Module that called for data transmission
 * @param cstream The CUDA stream to use for async memory copies.
 * @return GXF result code
 */
gxf_result_t transmit_data_per_model(gxf_context_t& cont,
                                     const HoloInfer::MultiMappings& model_to_tensor_map,
                                     HoloInfer::DataMap& input_data_map, OutputContext& op_output,
                                     std::vector<std::string>& out_tensors,
                                     HoloInfer::DimType& tensor_out_dims_map, bool cuda_buffer_in,
                                     bool cuda_buffer_out,
                                     const nvidia::gxf::Handle<nvidia::gxf::Allocator>& allocator_,
                                     const std::string& module, const cudaStream_t& cstream);

/**
 * Transmits multiple buffers via GXF Transmitters with persistent message caching.
 *
 * Unlike the other overloads, this variant allocates the output GXF entity and inserts tensor
 * components only once (on the first call). On subsequent calls the pre-allocated tensor buffers
 * are reused and only the data is updated via memcpy / cudaMemcpyAsync. A tensor is reshaped
 * (and its backing memory reallocated) only when the incoming buffer size exceeds the current
 * allocation capacity. The persistent entity is emitted by reference-counted copy each frame
 * rather than being moved, so the cache remains valid for the next call.
 *
 * @param cont GXF context for transmission
 * @param model_to_tensor_map Map of model name as key, mapped to a vector of tensor names
 * @param input_data_map Map of tensor name as key, mapped to the data buffer
 * @param op_output Output context. Assumes the output port name is "transmitter".
 * @param out_tensors Output tensor names
 * @param tensor_out_dims_map Map with model name as key mapped to output tensor dimensions
 * @param cuda_buffer_in Whether input buffers reside on the GPU
 * @param cuda_buffer_out Whether the output message should reside on the GPU
 * @param allocator_ GXF Memory allocator
 * @param module Module name used in error reporting
 * @param cstream CUDA stream to use for async memory copies
 * @param cache Persistent cache holding the reusable output entity and per-tensor allocation info
 * @return GXF result code
 */
gxf_result_t transmit_data_per_model(gxf_context_t& cont,
                                     const HoloInfer::MultiMappings& model_to_tensor_map,
                                     HoloInfer::DataMap& input_data_map, OutputContext& op_output,
                                     std::vector<std::string>& out_tensors,
                                     HoloInfer::DimType& tensor_out_dims_map, bool cuda_buffer_in,
                                     bool cuda_buffer_out,
                                     const nvidia::gxf::Handle<nvidia::gxf::Allocator>& allocator_,
                                     const std::string& module, const cudaStream_t& cstream,
                                     TensorTransmitCache& cache);

/**
 * Setting up activation for each model by activation_map and ActivationSpec
 *
 * @param inference_specs: Inference spec that are needed to set the it's activation_map.
 * @param activation_map: The predefined activation_map comes from the InferenceOp's
 * `activation_map` parameter.
 * @param activation_specs: Activation specs received from the input port
 * `model_activation_specs` of InferenceOp.
 * @param module Module that called for setting up activation specifications
 */
gxf_result_t set_activation_per_model(
    std::shared_ptr<HoloInfer::InferenceSpecs>& inference_specs,
    const HoloInfer::Mappings& activation_map,
    const std::vector<HoloInfer::ActivationSpec>& activation_specs, const std::string& module);
}  // namespace holoscan::utils

#endif /* HOLOSCAN_UTILS_HOLOINFER_UTILS_HPP */
