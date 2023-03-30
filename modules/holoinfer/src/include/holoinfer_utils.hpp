/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef _HOLOSCAN_INFER_UTILS_API_H
#define _HOLOSCAN_INFER_UTILS_API_H

#include <sys/utsname.h>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "gxf/core/entity.hpp"
#include "gxf/core/gxf.h"
#include "gxf/core/parameter.hpp"
#include "gxf/cuda/cuda_stream.hpp"
#include "gxf/cuda/cuda_stream_id.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/clock.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/parameter_parser_std.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/timestamp.hpp"
#include "gxf/std/transmitter.hpp"

#include "holoinfer_buffer.hpp"
#include "holoinfer_constants.hpp"

namespace holoscan {
namespace inference {

void timer_init(TimePoint& _t);

using GXFTransmitters = std::vector<nvidia::gxf::Handle<nvidia::gxf::Transmitter>>;
using GXFReceivers = std::vector<nvidia::gxf::Handle<nvidia::gxf::Receiver>>;

/**
 * Extracts data from multi GXF Receivers.
 *
 * @param receivers             Vector of GXF Receivers
 * @param in_tensors            Input tensor names
 * @param data_per_tensor       Map is updated with output tensor name as key mapped to data buffer
 * @param dims_per_tensor       Map is updated with tensor name as key mapped to dimension of
 *                              input tensor
 * @param cuda_buffer_out       Flag defining the location of output memory (Device or Host)
 * @param module                Module that called for data extraction
 *
 * @returns GXF result code
 */
gxf_result_t _HOLOSCAN_EXTERNAL_API_ multiai_get_data_per_model(
    const GXFReceivers& receivers, const std::vector<std::string>& in_tensors,
    DataMap& data_per_tensor, std::map<std::string, std::vector<int>>& dims_per_tensor,
    bool cuda_buffer_out, const std::string& module);

/**
 * Transmits multiple buffers via multi GXF Transmitters.
 *
 * @param context           GXF context for transmission
 * @param model_map         Map of model name as key, mapped to a tensor name
 * @param data_map          Map of tensor name as key, mapped to the data buffer as a vector of
 *                          float32 type
 * @param transmitters      Vector of GXF Transmitters
 * @param out_tensors       Output tensor names
 * @param data_per_model    Map is updated with output tensor name as key mapped to data buffer
 * @param out_dims_map      Map is updated with model name as key mapped to dimension of
 *                          output tensor as a vector
 * @param in_on_cuda        Flag to demonstrate if memory storage of input buffers is on CUDA
 * @param out_on_cuda       Flag to demonstrate if memory storage of output message is on CUDA
 * @param element_type      Data type of input buffers (float32 only)
 * @param module            Module that called for data transmission
 * @param allocator         GXF Memory allocator
 *
 * @returns GXF result code
 */
gxf_result_t _HOLOSCAN_EXTERNAL_API_ multiai_transmit_data_per_model(
    gxf_context_t& context, const Mappings& model_map, DataMap& data_map,
    const GXFTransmitters& transmitters, const std::vector<std::string>& out_tensors,
    DimType& out_dims_map, bool in_on_cuda, bool out_on_cuda,
    const nvidia::gxf::PrimitiveType& element_type, const std::string& module,
    const nvidia::gxf::Handle<nvidia::gxf::Allocator>& allocator);

/**
 * @brief Maps data per tensor to data per model
 * @param model_data_mapping Model to tensor mapping
 * @param data_per_model Map to be populated with model as key name and value as DataBuffer
 * @param data_per_input_tensor Map with key as tensor name and value as DataBuffer
 */
InferStatus map_data_to_model_from_tensor(const MultiMappings& model_data_mapping,
                                          DataMap& data_per_model, DataMap& data_per_input_tensor);

/**
 * Reports error with module, submodule and message
 *
 * @param module    Module of error occurrence
 * @param submodule Submodule/Function of error occurrence with the error message (as string)
 *
 * @returns GXF Error code: GXF_FAILURE
 */
gxf_result_t _HOLOSCAN_EXTERNAL_API_ report_error(const std::string& module,
                                                  const std::string& submodule);

/**
 * Raise error with module, submodule and message
 *
 * @param module    Module of error occurrence
 * @param submodule Submodule/Function of error occurrence with the error message (as string)
 */
void _HOLOSCAN_EXTERNAL_API_ raise_error(const std::string& module, const std::string& submodule);

/**
 * @brief Checks for correctness of inference parameters from configuration.
 * @param model_path_map Map with model name as key, path to model as value
 * @param pre_processor_map Map of model name as key, mapped to vector of tensor names
 * @param inference_map Map with model name as key, output tensor name as value
 * @param in_tensor_names Input tensor names
 * @param out_tensor_names Output tensor names
 * @return InferStatus with appropriate code and message
 */
InferStatus multiai_inference_validity_check(const Mappings& model_path_map,
                                             const MultiMappings& pre_processor_map,
                                             const Mappings& inference_map,
                                             const std::vector<std::string>& in_tensor_names,
                                             const std::vector<std::string>& out_tensor_names);

/**
 * @brief Checks for correctness of processing parameters from configuration.
 * @param processed_map Map with input tensor name as key, output tensor name as value
 * @param in_tensor_names Input tensor names
 * @param out_tensor_names Output tensor names
 */
InferStatus multiai_processor_validity_check(const Mappings& processed_map,
                                             const std::vector<std::string>& in_tensor_names,
                                             const std::vector<std::string>& out_tensor_names);

/**
 * @brief Checks if the processor is arm based
 */
bool is_platform_aarch64();

}  // namespace inference
}  // namespace holoscan
#endif
