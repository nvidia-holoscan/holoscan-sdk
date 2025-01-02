/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_UTILS_HOLOINFER_HPP
#define HOLOSCAN_UTILS_HOLOINFER_HPP

#include <map>
#include <string>
#include <vector>

#include "holoscan/core/io_context.hpp"
#include "holoscan/utils/cuda_stream_handler.hpp"

#include <holoinfer_buffer.hpp>

namespace HoloInfer = holoscan::inference;

namespace holoscan::utils {

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
 * @param context GXF context for transmission
 * @param model_to_tensor_map Map of model name as key, mapped to a vector of tensor names
 * @param input_data_map Map of tensor name as key, mapped to the data buffer as a vector
 * @param op_output Output context
 * @param out_tensors Output tensor names
 * @param data_per_model Map is updated with output tensor name as key mapped to data buffer
 * @param tensor_out_dims_map Map is updated with model name as key mapped to dimension of
 *                          output tensor as a vector
 * @param cuda_buffer_in Flag to demonstrate if memory storage of input buffers is on CUDA
 * @param cuda_buffer_out Flag to demonstrate if memory storage of output message is on CUDA
 * @param allocator GXF Memory allocator
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
 * @param context GXF context for transmission
 * @param model_to_tensor_map Map of model name as key, mapped to a vector of tensor names
 * @param input_data_map Map of tensor name as key, mapped to the data buffer as a vector
 * @param op_output Output context
 * @param out_tensors Output tensor names
 * @param data_per_model Map is updated with output tensor name as key mapped to data buffer
 * @param tensor_out_dims_map Map is updated with model name as key mapped to dimension of
 *                          output tensor as a vector
 * @param cuda_buffer_in Flag to demonstrate if memory storage of input buffers is on CUDA
 * @param cuda_buffer_out Flag to demonstrate if memory storage of output message is on CUDA
 * @param allocator GXF Memory allocator
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

}  // namespace holoscan::utils

#endif /* HOLOSCAN_UTILS_HOLOINFER_HPP */
