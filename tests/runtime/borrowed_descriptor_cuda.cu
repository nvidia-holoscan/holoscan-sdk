/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>
#include <cstdint>

namespace holoscan::test {
namespace {
__global__ void borrowed_image_kernel(const std::uint64_t* input, std::uint64_t* output,
                                      std::uint64_t* observed, std::uint64_t increment) {
  *output = *input + increment;
  *observed = *output;
}
}  // namespace

cudaError_t launch_borrowed_image_kernel(const std::uint64_t* input, std::uint64_t* output,
                                         std::uint64_t* observed, std::uint64_t increment,
                                         cudaStream_t stream) {
  borrowed_image_kernel<<<1, 1, 0, stream>>>(input, output, observed, increment);
  return cudaGetLastError();
}
}  // namespace holoscan::test
