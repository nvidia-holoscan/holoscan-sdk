/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// matrix multiplication
__global__ void matrix_multiply(float* A, float* B, float* C, int N) {
  // Calculate the row and column indices for the current thread
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // Initialize the sum for the current element of C
  float sum = 0.0f;

  // Perform the matrix multiplication for the current element
  if (row < N && col < N) {
    for (int i = 0; i < N; ++i) {
      sum += A[row * N + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
  }
}

void asyncLaunchMatrixMultiplyKernel(float* A, float* B, float* C, int N, cudaStream_t stream) {
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
  matrix_multiply<<<numBlocks, threadsPerBlock, 0, stream>>>(A, B, C, N);
}
