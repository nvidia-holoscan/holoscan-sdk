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

#ifndef HOLOVIZ_SRC_CUDA_CONVERT_HPP
#define HOLOVIZ_SRC_CUDA_CONVERT_HPP

#include <cuda.h>

#include <cstddef>
#include <cstdint>

namespace holoscan::viz {

/**
 * Convert an image form R8G8B8 (24 bit) to R8G8B8A8 (32 bit).
 *
 * @param width, height size of the image
 * @param src           source image data pointer
 * @param src_pitch     source image line pitch
 * @param dst           destination image data pointer
 * @param stream        stream to use for this operation
 */
void ConvertR8G8B8ToR8G8B8A8(uint32_t width, uint32_t height, CUdeviceptr src, size_t src_pitch,
                             CUarray dst, CUstream stream);

/**
 * Convert an image form B8G8R8A8 (32 bit) to R8G8B8A8 (32 bit).
 *
 * @param width, height size of the image
 * @param src           source image data pointer
 * @param src_pitch     source image line pitch
 * @param dst           destination image data pointer
 * @param dst_pitch     destination image line pitch
 * @param stream        stream to use for this operation
 */
void ConvertB8G8R8A8ToR8G8B8A8(uint32_t width, uint32_t height, CUdeviceptr src, size_t src_pitch,
                               CUdeviceptr dst, size_t dst_pitch, CUstream stream);

}  // namespace holoscan::viz

#endif /* HOLOVIZ_SRC_CUDA_CONVERT_HPP */
