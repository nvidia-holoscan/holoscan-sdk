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

#ifndef TENSOR_INTEROP_CPP_SEND_TENSOR_GXF_HPP
#define TENSOR_INTEROP_CPP_SEND_TENSOR_GXF_HPP

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/parameter_parser_std.hpp"
#include "gxf/std/tensor.hpp"
#include "gxf/std/transmitter.hpp"

#ifndef CUDA_TRY
#define CUDA_TRY(stmt)                                                                     \
  ({                                                                                       \
    cudaError_t _holoscan_cuda_err = stmt;                                                 \
    if (cudaSuccess != _holoscan_cuda_err) {                                               \
      GXF_LOG_ERROR("CUDA Runtime call %s in line %d of file %s failed with '%s' (%d).\n", \
                    #stmt,                                                                 \
                    __LINE__,                                                              \
                    __FILE__,                                                              \
                    cudaGetErrorString(_holoscan_cuda_err),                                \
                    _holoscan_cuda_err);                                                   \
    }                                                                                      \
    _holoscan_cuda_err;                                                                    \
  })
#endif

namespace nvidia {
namespace gxf {
namespace test {

class SendTensor : public Codelet {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override {
    gxf::Expected<void> result;
    result &= registrar->parameter(signal_, "signal");
    result &= registrar->parameter(pool_, "pool", "Pool", "Allocator instance for output tensors.");
    return gxf::ToResultCode(result);
  }
  gxf_result_t start() override { return GXF_SUCCESS; }
  gxf_result_t tick() override {
    constexpr int rows = 4;
    constexpr int cols = 4;
    constexpr int out_channels = 3;
    constexpr gxf::PrimitiveType element_type = gxf::PrimitiveType::kUnsigned8;
    const gxf::Shape tensor_shape{rows, cols, out_channels};

    gxf::Expected<gxf::Entity> out_message = CreateTensorMap(
        context(),
        pool_,
        {{"tensor",
          gxf::MemoryStorageType::kDevice,
          tensor_shape,
          gxf::PrimitiveType::kUnsigned8,
          0,
          gxf::ComputeTrivialStrides(tensor_shape, gxf::PrimitiveTypeSize(element_type))}});

    const auto maybe_output_tensor = out_message.value().get<gxf::Tensor>("tensor");

    if (!maybe_output_tensor) {
      GXF_LOG_ERROR("Failed to access output tensor with name `tensor`");
      return gxf::ToResultCode(maybe_output_tensor);
    }

    void* output_data_ptr = maybe_output_tensor.value()->pointer();
    CUDA_TRY(cudaMemset(output_data_ptr, value_, tensor_shape.size() *
                                                 gxf::PrimitiveTypeSize(element_type)));

    value_ = (value_ + 1) % 255;

    const auto result = signal_->publish(out_message.value());
    return gxf::ToResultCode(result);
  }
  gxf_result_t stop() override { return GXF_SUCCESS; }

 private:
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> signal_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> pool_;
  int value_ = 1;
};

}  // namespace test
}  // namespace gxf
}  // namespace nvidia

#endif /* TENSOR_INTEROP_CPP_SEND_TENSOR_GXF_HPP */
