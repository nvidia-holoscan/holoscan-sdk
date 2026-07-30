/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// NOLINTBEGIN(build/include_what_you_use)

extern "C" __global__ void customKernel2(const unsigned char* input, unsigned char* output,
                                         int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int sobel_x[3][3] = {// Sobel X kernel
                       {-1, 0, 1},
                       {-2, 0, 2},
                       {-1, 0, 1}};

  int sobel_y[3][3] = {// Sobel Y kernel
                       {-1, -2, -1},
                       {0, 0, 0},
                       {1, 2, 1}};

  int Gx = 0, Gy = 0;

  // Compute Sobel filter in 3x3 neighborhood
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      int neighbor_x = min(max(x + i, 0), width - 1);
      int neighbor_y = min(max(y + j, 0), height - 1);

      int idx = (neighbor_y * width + neighbor_x);
      unsigned char gray = input[idx];

      Gx += gray * sobel_x[j + 1][i + 1];
      Gy += gray * sobel_y[j + 1][i + 1];
    }
  }

  // Compute gradient magnitude
  int magnitude = min((int)sqrtf(Gx * Gx + Gy * Gy), 255);

  int outIdx = y * width + x;
  output[outIdx] = magnitude;
}

// NOLINTEND(build/include_what_you_use)
