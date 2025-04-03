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

extern "C" __global__ void customKernel2(const unsigned char* input, unsigned char* output,
                                         int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int channels = 3;  // RGB

  if (x >= width || y >= height) return;

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

      int idx = (neighbor_y * width + neighbor_x) * channels;
      unsigned char gray =
          (input[idx] + input[idx + 1] + input[idx + 2]) / 3;  // Convert to grayscale

      Gx += gray * sobel_x[j + 1][i + 1];
      Gy += gray * sobel_y[j + 1][i + 1];
    }
  }

  // Compute gradient magnitude
  int magnitude = min((int)sqrtf(Gx * Gx + Gy * Gy), 255);

  int outIdx = (y * width + x) * channels;
  output[outIdx] = output[outIdx + 1] = output[outIdx + 2] = magnitude;  // Set grayscale output
}
