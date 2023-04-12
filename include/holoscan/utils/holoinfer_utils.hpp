/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoinfer_buffer.hpp>

namespace HoloInfer = holoscan::inference;

namespace holoscan::utils {

gxf_result_t multiai_get_data_per_model(InputContext& op_input,
                                        const std::vector<std::string>& in_tensors,
                                        HoloInfer::DataMap& data_per_input_tensor,
                                        std::map<std::string, std::vector<int>>& dims_per_tensor,
                                        bool cuda_buffer_out, const std::string& module);

gxf_result_t multiai_transmit_data_per_model(
    gxf_context_t& cont, const HoloInfer::Mappings& model_to_tensor_map,
    HoloInfer::DataMap& input_data_map, OutputContext& op_output,
    const std::vector<std::string>& out_tensors, HoloInfer::DimType& tensor_out_dims_map,
    bool cuda_buffer_in, bool cuda_buffer_out, const nvidia::gxf::PrimitiveType& element_type,
    const nvidia::gxf::Handle<nvidia::gxf::Allocator>& allocator_, const std::string& module);

}  // namespace holoscan::utils

#endif /* HOLOSCAN_UTILS_HOLOINFER_HPP */
