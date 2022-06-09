/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_SEGMENTATION_POSTPROCESSOR_HPP_
#define NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_SEGMENTATION_POSTPROCESSOR_HPP_

#include <string>

#include "gxf/std/codelet.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"

#include "segmentation_postprocessor.cu.hpp"

namespace nvidia {
namespace holoscan {
namespace segmentation_postprocessor {

/// @brief Segmentation model postproceessing Codelet converting inference output to class index.
///
/// This Codelet performs segmentation model postprocessing in CUDA.
/// It takes in the output of inference, either with the final softmax layer (multiclass) or sigmoid
/// (2-class), and emits a Tensor<uint8_t> that contains the highest probability class index.
/// The class index can then be consumed downstream for visualization or other purposes.
/// The inference output currently supported are either HWC or NCHW.
class Postprocessor : public gxf::Codelet {
 public:
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;

 private:
  NetworkOutputType network_output_type_value_;
  DataFormat data_format_value_;

  gxf::Parameter<gxf::Handle<gxf::Receiver>> in_;
  gxf::Parameter<std::string> in_tensor_name_;
  gxf::Parameter<std::string> network_output_type_;
  gxf::Parameter<std::string> data_format_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> out_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> allocator_;
};
}  // namespace segmentation_postprocessor
}  // namespace holoscan
}  // namespace nvidia

#endif  // NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_SEGMENTATION_POSTPROCESSOR_HPP_
