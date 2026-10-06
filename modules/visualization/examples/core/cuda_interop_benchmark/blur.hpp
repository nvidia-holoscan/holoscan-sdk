/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda.h>
#include <cufft.h>

/// FFT grid the blur processes, sized for the bundled 4001x3337 nv_logo.png (4096 and
/// 3375 = 15^3 are FFT-friendly sizes covering the image). Sources smaller than the grid
/// are padded by edge replication, which also softens the circular-convolution
/// wrap-around that a plain crop would exhibit at the image borders.
inline constexpr uint32_t kBlurGridWidth = 4096;
inline constexpr uint32_t kBlurGridHeight = 3375;

/// Gaussian-blurs an RGBA8 image on the GPU with FFT convolution: forward 2D FFT, Gaussian
/// transfer function applied to the spectrum (the Fourier transform of a spatial Gaussian
/// with standard deviation sigma is exp(-2 pi^2 sigma^2 |f|^2)), inverse FFT. All working
/// buffers live in device memory, separate from the source image, so no data leaves VRAM.
/// Alpha is filtered like the colors on purpose: the image layer alpha-blends, and
/// compositing blurred RGB through the original sharp alpha would re-mask the result with
/// hard edges.
class GaussianBlur {
 public:
  virtual ~GaussianBlur() = default;

  /// Short human-readable implementation name for benchmark output.
  virtual const char* label() const = 0;

  /// Enqueue the blur of `src` (pitched RGBA8 device memory of the size given at construction)
  /// into output() on `stream`. Asynchronous; ordered by the stream.
  virtual void blur_async(CUdeviceptr src, size_t src_pitch, CUstream stream) = 0;

  /// The blurred RGBA8 image in device memory, valid once the stream reaches the end of the
  /// work enqueued by blur_async(). Rows are output_pitch() bytes apart; only the
  /// width x height region given at construction is meaningful.
  virtual CUdeviceptr output() const = 0;

  /// Row pitch in bytes of output()
  size_t output_pitch() const { return static_cast<size_t>(kBlurGridWidth) * 4; }
};

/// cuFFT implementation: kernels unpack the texels into four planar float channels and
/// pack them back, and cuFFT runs batched real 2D transforms (R2C, Gaussian multiply on
/// the half-spectrum, C2R) with the spectrum round-tripping through global memory.
class CufftGaussianBlur final : public GaussianBlur {
 public:
  /// @param width, height source image size in pixels; must not exceed the FFT grid
  ///                      (throws std::runtime_error otherwise)
  /// @param sigma         Gaussian standard deviation in pixels
  CufftGaussianBlur(uint32_t width, uint32_t height, float sigma);
  ~CufftGaussianBlur() override;

  CufftGaussianBlur(const CufftGaussianBlur&) = delete;
  CufftGaussianBlur& operator=(const CufftGaussianBlur&) = delete;

  const char* label() const override { return "cuFFT unfused"; }
  void blur_async(CUdeviceptr src, size_t src_pitch, CUstream stream) override;
  CUdeviceptr output() const override { return output_; }

 private:
  uint32_t width_;
  uint32_t height_;
  float sigma_;
  cufftHandle forward_plan_{};
  cufftHandle inverse_plan_{};
  CUdeviceptr planes_{};   ///< scratch: 4 planar float channels, one FFT grid each
  CUdeviceptr spectra_{};  ///< scratch: 4 spectra, (kBlurGridWidth/2+1)*kBlurGridHeight each
  CUdeviceptr output_{};   ///< blurred RGBA8 image on the FFT grid, tightly packed rows
};
