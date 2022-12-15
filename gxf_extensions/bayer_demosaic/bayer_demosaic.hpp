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

#ifndef NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_BAYER_DEMOSAIC_HPP_
#define NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_BAYER_DEMOSAIC_HPP_



#include <npp.h>

#include <string>

#include "gxf/core/entity.hpp"
#include "gxf/core/handle.hpp"
#include "gxf/multimedia/video.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/transmitter.hpp"
#include "gxf/std/memory_buffer.hpp"

namespace nvidia::holoscan {

class BayerDemosaic : public gxf::Codelet {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t initialize() override;
  gxf_result_t deinitialize() override;

  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;

 private:
  gxf::Parameter<gxf::Handle<gxf::Receiver>> receiver_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> transmitter_;
  gxf::Parameter<std::string> in_tensor_name_;
  gxf::Parameter<std::string> out_tensor_name_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> pool_;
  gxf::Parameter<gxf::Handle<gxf::CudaStreamPool>> cuda_stream_pool_;
  gxf::Parameter<int> bayer_interp_mode_;
  gxf::Parameter<int> bayer_grid_pos_;
  gxf::Parameter<bool> generate_alpha_;
  gxf::Parameter<int> alpha_value_;

  gxf::Handle<gxf::CudaStream> cuda_stream_;
  NppStreamContext npp_stream_ctx_{};

  NppiInterpolationMode npp_bayer_interp_mode_;
  NppiBayerGridPosition npp_bayer_grid_pos_;

  gxf::MemoryBuffer device_scratch_buffer_;
};

}  // namespace nvidia::holoscan

#endif  // NVIDIA_CLARA_HOLOSCAN_GXF_EXTENSIONS_BAYER_DEMOSAIC_HPP_
