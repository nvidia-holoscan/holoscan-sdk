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

#include <vector>

#include <holoscan/core/domain/tensor.hpp>

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

  // Validate DLPack metadata is consistent with the MatX tensor.
  EXPECT_EQ(tensor.device().device_type, kDLCUDA);
  EXPECT_EQ(tensor.dtype().code, kDLFloat);
  EXPECT_EQ(tensor.dtype().bits, 32);
  EXPECT_EQ(tensor.itemsize(), 4);
  EXPECT_EQ(tensor.size(), 10);
  EXPECT_EQ(tensor.nbytes(), 40);

  // Validate data can be accessed (and values are as expected).
  std::vector<float> host(10);
  auto result =
      cudaMemcpy(host.data(), tensor.data(), host.size() * sizeof(float), cudaMemcpyDeviceToHost);
  ASSERT_EQ(result, cudaSuccess);

  for (int i = 0; i < 10; ++i) {
    EXPECT_FLOAT_EQ(host[i], static_cast<float>(i + 1));
  }
}
