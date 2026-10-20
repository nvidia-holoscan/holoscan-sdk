/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "blur.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace {

// All four channels are blurred as separate planes; see the GaussianBlur documentation for
// why alpha is filtered like the colors.
constexpr uint32_t kChannels = 4;

void check_cuda(cudaError_t result, const char* what) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorString(result));
  }
}

void check_cufft(cufftResult result, const char* what) {
  if (result != CUFFT_SUCCESS) {
    throw std::runtime_error(std::string(what) + " failed: cufftResult " + std::to_string(result));
  }
}

/// Unpack the interleaved RGBA8 texels into four planar float channels on the FFT grid,
/// edge-replicating outside the source image.
__global__ void unpack_kernel(const unsigned char* src, size_t src_pitch, uint32_t src_w,
                              uint32_t src_h, float* planes) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if ((x >= kBlurGridWidth) || (y >= kBlurGridHeight)) {
    return;
  }
  const uint32_t sx = min(x, src_w - 1);
  const uint32_t sy = min(y, src_h - 1);
  const uchar4 texel =
      *reinterpret_cast<const uchar4*>(src + size_t(sy) * src_pitch + size_t(sx) * sizeof(uchar4));
  const size_t plane_size = size_t(kBlurGridWidth) * kBlurGridHeight;
  const size_t index = size_t(y) * kBlurGridWidth + x;
  planes[index] = float(texel.x);
  planes[plane_size + index] = float(texel.y);
  planes[2 * plane_size + index] = float(texel.z);
  planes[3 * plane_size + index] = float(texel.w);
}

/// The Fourier transform of a spatial Gaussian with standard deviation sigma is
/// exp(-2 pi^2 sigma^2 |f|^2); multiplying the spectrum by it convolves the image with the
/// Gaussian (with periodic boundaries). The 1/(width*height) normalization of the cuFFT
/// forward+inverse round trip is folded into the same factor.
__global__ void gaussian_kernel(cufftComplex* spectra, uint32_t spectrum_width, float sigma,
                                float norm) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  const uint32_t channel = blockIdx.z;
  if ((x >= spectrum_width) || (y >= kBlurGridHeight)) {
    return;
  }
  // the R2C layout holds only non-negative x frequencies; y wraps around to negative ones
  const float fx = float(x) / float(kBlurGridWidth);
  const int signed_y = (y <= kBlurGridHeight / 2) ? int(y) : int(y) - int(kBlurGridHeight);
  const float fy = float(signed_y) / float(kBlurGridHeight);
  constexpr float kPi = 3.14159265358979323846F;
  const float gauss = expf(-2.0F * kPi * kPi * sigma * sigma * (fx * fx + fy * fy)) * norm;
  const size_t index = (size_t(channel) * kBlurGridHeight + y) * spectrum_width + x;
  spectra[index].x *= gauss;
  spectra[index].y *= gauss;
}

/// Pack the four planar float channels back into interleaved RGBA8 texels.
__global__ void pack_kernel(const float* planes, uchar4* dst) {
  const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if ((x >= kBlurGridWidth) || (y >= kBlurGridHeight)) {
    return;
  }
  const size_t plane_size = size_t(kBlurGridWidth) * kBlurGridHeight;
  const size_t index = size_t(y) * kBlurGridWidth + x;
  const auto to_byte = [](float value) {
    return (unsigned char)(fminf(fmaxf(value + 0.5F, 0.0F), 255.0F));
  };
  dst[index] = make_uchar4(to_byte(planes[index]),
                           to_byte(planes[plane_size + index]),
                           to_byte(planes[2 * plane_size + index]),
                           to_byte(planes[3 * plane_size + index]));
}

}  // namespace

CufftGaussianBlur::CufftGaussianBlur(uint32_t width, uint32_t height, float sigma)
    : width_(width), height_(height), sigma_(sigma) {
  if ((width == 0U) || (height == 0U)) {
    throw std::runtime_error("source image dimensions must be greater than zero");
  }
  if (width > kBlurGridWidth || height > kBlurGridHeight) {
    throw std::runtime_error(
        "source image " + std::to_string(width) + "x" + std::to_string(height) +
        " exceeds the FFT grid " + std::to_string(kBlurGridWidth) + "x" +
        std::to_string(kBlurGridHeight) + "; enlarge kBlurGridWidth/kBlurGridHeight in blur.hpp");
  }

  const size_t plane_size = size_t(kBlurGridWidth) * kBlurGridHeight;
  const uint32_t spectrum_width = kBlurGridWidth / 2 + 1;
  const size_t spectrum_size = size_t(spectrum_width) * kBlurGridHeight;

  void* pointer = nullptr;
  check_cuda(cudaMalloc(&pointer, kChannels * plane_size * sizeof(float)), "cudaMalloc planes");
  planes_ = reinterpret_cast<CUdeviceptr>(pointer);
  check_cuda(cudaMalloc(&pointer, kChannels * spectrum_size * sizeof(cufftComplex)),
             "cudaMalloc spectra");
  spectra_ = reinterpret_cast<CUdeviceptr>(pointer);
  check_cuda(cudaMalloc(&pointer, plane_size * sizeof(uchar4)), "cudaMalloc output");
  output_ = reinterpret_cast<CUdeviceptr>(pointer);

  int extents[2] = {int(kBlurGridHeight), int(kBlurGridWidth)};
  int real_embed[2] = {int(kBlurGridHeight), int(kBlurGridWidth)};
  int complex_embed[2] = {int(kBlurGridHeight), int(spectrum_width)};
  check_cufft(cufftPlanMany(&forward_plan_,
                            2,
                            extents,
                            real_embed,
                            1,
                            int(plane_size),
                            complex_embed,
                            1,
                            int(spectrum_size),
                            CUFFT_R2C,
                            kChannels),
              "cufftPlanMany forward");
  check_cufft(cufftPlanMany(&inverse_plan_,
                            2,
                            extents,
                            complex_embed,
                            1,
                            int(spectrum_size),
                            real_embed,
                            1,
                            int(plane_size),
                            CUFFT_C2R,
                            kChannels),
              "cufftPlanMany inverse");
}

CufftGaussianBlur::~CufftGaussianBlur() {
  cufftDestroy(forward_plan_);
  cufftDestroy(inverse_plan_);
  cudaFree(reinterpret_cast<void*>(planes_));
  cudaFree(reinterpret_cast<void*>(spectra_));
  cudaFree(reinterpret_cast<void*>(output_));
}

void CufftGaussianBlur::blur_async(CUdeviceptr src, size_t src_pitch, CUstream stream) {
  const dim3 block(16, 16);
  const dim3 grid((kBlurGridWidth + block.x - 1) / block.x,
                  (kBlurGridHeight + block.y - 1) / block.y);

  unpack_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const unsigned char*>(src),
                                            src_pitch,
                                            width_,
                                            height_,
                                            reinterpret_cast<float*>(planes_));
  check_cuda(cudaGetLastError(), "unpack_kernel launch");

  check_cufft(cufftSetStream(forward_plan_, stream), "cufftSetStream forward");
  check_cufft(cufftExecR2C(forward_plan_,
                           reinterpret_cast<cufftReal*>(planes_),
                           reinterpret_cast<cufftComplex*>(spectra_)),
              "cufftExecR2C");

  const uint32_t spectrum_width = kBlurGridWidth / 2 + 1;
  const dim3 spectrum_grid((spectrum_width + block.x - 1) / block.x,
                           (kBlurGridHeight + block.y - 1) / block.y,
                           kChannels);
  gaussian_kernel<<<spectrum_grid, block, 0, stream>>>(
      reinterpret_cast<cufftComplex*>(spectra_),
      spectrum_width,
      sigma_,
      1.0F / (float(kBlurGridWidth) * float(kBlurGridHeight)));
  check_cuda(cudaGetLastError(), "gaussian_kernel launch");

  check_cufft(cufftSetStream(inverse_plan_, stream), "cufftSetStream inverse");
  check_cufft(cufftExecC2R(inverse_plan_,
                           reinterpret_cast<cufftComplex*>(spectra_),
                           reinterpret_cast<cufftReal*>(planes_)),
              "cufftExecC2R");

  pack_kernel<<<grid, block, 0, stream>>>(reinterpret_cast<const float*>(planes_),
                                          reinterpret_cast<uchar4*>(output_));
  check_cuda(cudaGetLastError(), "pack_kernel launch");
}
