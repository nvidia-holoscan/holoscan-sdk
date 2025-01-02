/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "data_processor.hpp"

#include <unistd.h>

#include <cub/device/device_reduce.cuh>

namespace holoscan {
namespace inference {

/**
 * This class implements an iterator which skips `step` elements between each iteration.
 */
class step_iterator : public std::iterator<std::random_access_iterator_tag, float> {
  pointer cur_;
  size_t step_;

 public:
  explicit step_iterator(pointer cur, size_t step) : cur_(cur), step_(step) {}
  template <typename Distance>
  __host__ __device__ __forceinline__ reference operator[](Distance offset) const {
    return cur_[offset * step_];
  }
};

/**
 * CUDA kernel normalizing the coordinates stored in the `key` member.
 *
 * @param rows
 * @param cols
 * @param channels
 * @param d_argmax
 * @param out
 */
static __global__ void normalize(size_t rows, size_t cols, size_t channels,
                                 cub::KeyValuePair<int, float>* d_argmax, float* out) {
  const uint index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index > channels) { return; }

  const int src_index = d_argmax[index].key;
  int row = src_index / cols;
  int col = src_index - (row * cols);
  out[index * 2 + 0] = (float)row / (float)rows;
  out[index * 2 + 1] = (float)col / (float)cols;
}

void DataProcessor::max_per_channel_scaled_cuda(size_t rows, size_t cols, size_t channels,
                                                const float* indata, float* outdata,
                                                cudaStream_t cuda_stream) {
  /// @todo This algorithm needs temporary storage, currently data processors are just functions
  /// without state. This should be an object with state so we can avoid re-allocating the temporary
  /// storage at each invocation.

  // Allocate result storage
  cub::KeyValuePair<int, float>* d_argmax = nullptr;
  check_cuda(
      cudaMallocAsync(&d_argmax, sizeof(cub::KeyValuePair<int, float>) * channels, cuda_stream));

  // get temp storage size
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, indata, d_argmax, rows * cols);

  // Allocate temporary storage
  check_cuda(cudaMallocAsync(&d_temp_storage, temp_storage_bytes, cuda_stream));

  for (size_t channel = 0; channel < channels; ++channel) {
    step_iterator iterator((float*)(indata + channel), channels);
    cub::DeviceReduce::ArgMax(
        d_temp_storage, temp_storage_bytes, iterator, &d_argmax[channel], rows * cols, cuda_stream);
  }

  check_cuda(cudaFreeAsync(d_temp_storage, cuda_stream));

  dim3 block(32, 1, 1);
  dim3 grid((channels + block.x - 1) / block.x, 1, 1);
  normalize<<<grid, block, 0, cuda_stream>>>(rows, cols, channels, d_argmax, outdata);
  check_cuda(cudaPeekAtLastError());

  check_cuda(cudaFreeAsync(d_argmax, cuda_stream));
}

}  // namespace inference
}  // namespace holoscan
