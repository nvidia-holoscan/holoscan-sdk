/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
