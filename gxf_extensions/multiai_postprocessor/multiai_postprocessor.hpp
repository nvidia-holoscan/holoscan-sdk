/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef NVIDIA_GXF_EXTENSIONS_MULTIAI_POSTPROCESSOR_HPP_
#define NVIDIA_GXF_EXTENSIONS_MULTIAI_POSTPROCESSOR_HPP_

#include <algorithm>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
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

#include <holoinfer.hpp>
#include <holoinfer_utils.hpp>

namespace HoloInfer = holoscan::inference;

namespace nvidia {
namespace holoscan {
namespace multiai {
/**
 * @brief Generic Multi AI postprocessor codelet to perform multiple operations on multiple tensors.
 */
class MultiAIPostprocessor : public gxf::Codelet {
 public:
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;

 private:
  /// Map with key as tensor name and value as vector of supported operations.
  ///  Supported operations: "max_per_channel_scaled"
  gxf::Parameter<HoloInfer::MultiMappings> process_operations_;

  /// Map with key as input tensor name and value as processed tensor name
  gxf::Parameter<HoloInfer::Mappings> processed_map_;

  /// Vector of input tensor names
  gxf::Parameter<std::vector<std::string> > in_tensor_names_;

  /// Vector of output tensor names
  gxf::Parameter<std::vector<std::string> > out_tensor_names_;

  /// Memory allocator
  gxf::Parameter<gxf::Handle<gxf::Allocator> > allocator_;

  /// Vector of input receivers. Multiple receivers supported.
  gxf::Parameter<HoloInfer::GXFReceivers> receivers_;

  /// Output transmitter. Single transmitter supported.
  gxf::Parameter<HoloInfer::GXFTransmitters> transmitter_;

  /// Flag showing if input buffers are on CUDA. Default is False.
  ///  Supported value: False
  gxf::Parameter<bool> input_on_cuda_;

  /// Flag showing if output buffers are on CUDA. Default is False.
  ///  Supported value: False
  gxf::Parameter<bool> output_on_cuda_;

  /// Flag showing if data transmission on CUDA. Default is False.
  ///  Supported value: False
  gxf::Parameter<bool> transmit_on_cuda_;

  /// Pointer to Data Processor context.
  std::unique_ptr<HoloInfer::ProcessorContext> holoscan_postprocess_context_;

  /// Map holding data per input tensor.
  HoloInfer::DataMap data_per_tensor_;

  /// Map holding dimensions per model. Key is model name and value is a vector with
  /// dimensions.
  std::map<std::string, std::vector<int> > dims_per_tensor_;

  /// Codelet Identifier, used in reporting.
  const std::string module_{"Multi AI Postprocessor Codelet"};
};
}  // namespace multiai
}  // namespace holoscan
}  // namespace nvidia

#endif  // NVIDIA_GXF_EXTENSIONS_MULTIAI_POSTPROCESSOR_HPP_
