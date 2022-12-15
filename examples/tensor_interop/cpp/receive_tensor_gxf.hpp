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

#ifndef TENSOR_INTEROP_CPP_RECEIVE_TENSOR_GXF_HPP
#define TENSOR_INTEROP_CPP_RECEIVE_TENSOR_GXF_HPP

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/parameter_parser_std.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/tensor.hpp"

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

class ReceiveTensor : public Codelet {
 public:
  gxf_result_t registerInterface(Registrar* registrar) override {
    gxf::Expected<void> result;
    result &= registrar->parameter(signal_, "signal");
    return gxf::ToResultCode(result);
  }
  gxf_result_t start() override { return GXF_SUCCESS; }
  gxf_result_t tick() override {
    const auto in_message = signal_->receive();
    if (!in_message || in_message.value().is_null()) { return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE; }

    const auto maybe_in_tensor = in_message.value().get<gxf::Tensor>("tensor");
    if (!maybe_in_tensor) {
      GXF_LOG_ERROR("Failed to access in tensor with name `tensor`");
      return gxf::ToResultCode(maybe_in_tensor);
    }
    void* in_data_ptr = maybe_in_tensor.value()->pointer();

    size_t data_size = maybe_in_tensor->get()->bytes_size();
    std::vector<uint8_t> in_data(data_size);

    CUDA_TRY(cudaMemcpy(in_data.data(), in_data_ptr, data_size, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < data_size; i++) { std::cout << static_cast<int>(in_data[i]) << " "; }
    std::cout << std::endl;

    return GXF_SUCCESS;
  }
  gxf_result_t stop() override { return GXF_SUCCESS; }

 private:
  gxf::Parameter<gxf::Handle<gxf::Receiver>> signal_;
};

}  // namespace test
}  // namespace gxf
}  // namespace nvidia

#endif /* TENSOR_INTEROP_CPP_RECEIVE_TENSOR_GXF_HPP */
