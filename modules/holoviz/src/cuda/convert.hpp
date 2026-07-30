/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
 * @param alpha         alpha value
 */
void ConvertR8G8B8ToR8G8B8A8(uint32_t width, uint32_t height, CUdeviceptr src, size_t src_pitch,
                             CUarray dst, CUstream stream, uint8_t alpha);

/**
 * Convert an image form R16G16B16 (48 bit) to R16G16B16A16 (64 bit).
 *
 * @param width, height size of the image
 * @param src           source image data pointer
 * @param src_pitch     source image line pitch
 * @param dst           destination image data pointer
 * @param stream        stream to use for this operation
 * @param alpha         alpha value
 */
void ConvertR16G16B16ToR16G16B16A16(uint32_t width, uint32_t height, CUdeviceptr src,
                                    size_t src_pitch, CUarray dst, CUstream stream, uint16_t alpha);

/**
 * Convert an image form R32G32B32 (96 bit) to R32G32B32A32 (128 bit).
 *
 * @param width, height size of the image
 * @param src           source image data pointer
 * @param src_pitch     source image line pitch
 * @param dst           destination image data pointer
 * @param stream        stream to use for this operation
 * @param alpha         alpha value
 */
void ConvertR32G32B32ToR32G32B32A32(uint32_t width, uint32_t height, CUdeviceptr src,
                                    size_t src_pitch, CUarray dst, CUstream stream, uint32_t alpha);

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
