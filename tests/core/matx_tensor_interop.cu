/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <matx.h>

#include <iostream>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>

namespace {

const char* dl_device_type_name(int device_type) {
  switch (device_type) {
    case kDLCUDA:
      return "kDLCUDA";
    case kDLCUDAHost:
      return "kDLCUDAHost";
    case kDLCUDAManaged:
      return "kDLCUDAManaged";
    default:
      return "unknown";
  }
}

}  // namespace

TEST(MatXInterop, MatxTensorToHoloscanTensorViaDlpack) {
  // Create a MatX tensor on the GPU and populate it.
  auto matx_tensor = matx::make_tensor<float>({10});
  matx_tensor.SetVals({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

  // Export to DLPack and construct a holoscan::Tensor (zero-copy).
  holoscan::Tensor tensor(matx_tensor.ToDlPack());

  EXPECT_EQ(tensor.ndim(), 1);
  const auto shape = tensor.shape();
  ASSERT_EQ(shape.size(), 1UL);
  EXPECT_EQ(shape[0], 10);

  // Validate DLPack metadata is consistent with CUDA-accessible MatX tensor memory.
  const auto device = tensor.device();
  const auto device_type = device.device_type;
  cudaDeviceProp device_prop;
  auto result = cudaGetDeviceProperties(&device_prop, device.device_id);
  ASSERT_EQ(result, cudaSuccess);
  const bool is_integrated_gpu = device_prop.integrated != 0;
  std::cout << "Detected DLPack device type: " << dl_device_type_name(device_type) << " ("
            << static_cast<int>(device_type) << "), CUDA device " << device.device_id
            << ", integrated=" << is_integrated_gpu << std::endl;

  if (is_integrated_gpu) {
    EXPECT_TRUE(device_type == kDLCUDAHost || device_type == kDLCUDAManaged)
        << "Expected CUDA host or managed DLPack memory on integrated GPU, got "
        << dl_device_type_name(device_type) << " (" << static_cast<int>(device_type) << ")";
  } else {
    EXPECT_EQ(device_type, kDLCUDA)
        << "Expected CUDA device DLPack memory on discrete GPU, got "
        << dl_device_type_name(device_type) << " (" << static_cast<int>(device_type) << ")";
  }
  EXPECT_EQ(tensor.dtype().code, kDLFloat);
  EXPECT_EQ(tensor.dtype().bits, 32);
  EXPECT_EQ(tensor.itemsize(), 4);
  EXPECT_EQ(tensor.size(), 10);
  EXPECT_EQ(tensor.nbytes(), 40);

  // Validate data can be accessed (and values are as expected).
  std::vector<float> host(10);
  cudaMemcpyKind copy_kind;
  switch (device_type) {
    case kDLCUDAHost:
      copy_kind = cudaMemcpyHostToHost;
      break;
    case kDLCUDAManaged:
      copy_kind = cudaMemcpyDefault;
      break;
    case kDLCUDA:
      copy_kind = cudaMemcpyDeviceToHost;
      break;
    default:
      FAIL() << "Unexpected DLPack device type for memory copy: " << static_cast<int>(device_type);
      return;
  }
  result = cudaMemcpy(host.data(), tensor.data(), host.size() * sizeof(float), copy_kind);
  ASSERT_EQ(result, cudaSuccess);

  for (int i = 0; i < 10; ++i) {
    EXPECT_FLOAT_EQ(host[i], static_cast<float>(i + 1));
  }
}
