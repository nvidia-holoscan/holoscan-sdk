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
#ifndef NVIDIA_GXF_EXTENSIONS_MULTIAI_INFERENCE_HPP_
#define NVIDIA_GXF_EXTENSIONS_MULTIAI_INFERENCE_HPP_

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
 * @brief Generic Multi AI inference codelet to perform multi model inference.
 */
class MultiAIInference : public gxf::Codelet {
 public:
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;

 private:
  /// Map with key as model name and value as model file path
  gxf::Parameter<HoloInfer::Mappings> model_path_map_;

  /// Map with key as model name and value as vector of input tensor names
  gxf::Parameter<HoloInfer::MultiMappings> pre_processor_map_;

  /// Map with key as model name and value as inferred tensor name
  gxf::Parameter<HoloInfer::Mappings> inference_map_;

  /// Flag to show if input model path mapping is for cached trt engine files
  gxf::Parameter<bool> is_engine_path_;

  /// Input tensor names
  gxf::Parameter<std::vector<std::string> > in_tensor_names_;

  /// Output tensor names
  gxf::Parameter<std::vector<std::string> > out_tensor_names_;

  /// Memory allocator
  gxf::Parameter<gxf::Handle<gxf::Allocator> > allocator_;

  /// Flag to enable inference on CPU (only supported by onnxruntime).
  /// Default is False.
  gxf::Parameter<bool> infer_on_cpu_;

  /// Flag to enable parallel inference. Default is True.
  gxf::Parameter<bool> parallel_inference_;

  /// Flag showing if trt engine file conversion will use FP16. Default is False.
  gxf::Parameter<bool> enable_fp16_;

  /// Backend to do inference on. Supported values: "trt", "onnxrt".
  gxf::Parameter<std::string> backend_;

  /// Vector of input receivers. Multiple receivers supported.
  gxf::Parameter<HoloInfer::GXFReceivers> receivers_;

  /// Output transmitter. Single transmitter supported.
  gxf::Parameter<HoloInfer::GXFTransmitters> transmitter_;

  /// Flag showing if input buffers are on CUDA. Default is True.
  gxf::Parameter<bool> input_on_cuda_;

  /// Flag showing if output buffers are on CUDA. Default is True.
  gxf::Parameter<bool> output_on_cuda_;

  /// Flag showing if data transmission is on CUDA. Default is True.
  gxf::Parameter<bool> transmit_on_cuda_;

  /// Pointer to inference context.
  std::unique_ptr<HoloInfer::InferContext> holoscan_infer_context_;

  /// Pointer to multi ai inference specifications
  std::shared_ptr<HoloInfer::MultiAISpecs> multiai_specs_;

  /// Data type of message to be transmitted. Supported value: float32
  gxf::PrimitiveType data_type_ = gxf::PrimitiveType::kFloat32;

  /// Map holding dimensions per model. Key is model name and value is a vector with
  /// dimensions.
  std::map<std::string, std::vector<int> > dims_per_tensor_;

  /// Codelet Identifier, used in reporting.
  const std::string module_{"Multi AI Inference Codelet"};
};

}  // namespace multiai
}  // namespace holoscan
}  // namespace nvidia

#endif  // NVIDIA_GXF_EXTENSIONS_MULTIAI_INFERENCE_HPP_
