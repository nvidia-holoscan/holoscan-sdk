/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda/stream.hpp"

#include <holomq/logger.hpp>

#include <cuda.h>

#include <utility>

namespace holomq {
namespace cuda {

Stream::Stream(const unsigned int flags) noexcept {
  CUstream stream{};
  if (cuStreamCreate(&stream, flags) != CUDA_SUCCESS) {
    holomq::log_literal(holomq::LogLevel::Error,
                        "cuStreamCreate failed during holomq::cuda::Stream construction");
    return;
  }
  stream_ = stream;
}

Stream::Stream(Stream&& other) noexcept : stream_(other.stream_) {
  other.stream_ = nullptr;
}

Stream& Stream::operator=(Stream&& other) noexcept {
  if (this != &other) {
    drain_and_destroy();
    stream_ = other.stream_;
    other.stream_ = nullptr;
  }
  return *this;
}

void Stream::drain_and_destroy() noexcept {
  if (stream_ == nullptr) {
    return;
  }
  const CUresult sync_rc = cuStreamSynchronize(stream_);
  if (sync_rc != CUDA_SUCCESS) {
    holomq::log_literal(holomq::LogLevel::Warn,
                        "cuStreamSynchronize failed during holomq::cuda::Stream teardown; "
                        "destroying stream anyway");
  }
  const CUresult destroy_rc = cuStreamDestroy(stream_);
  if (destroy_rc != CUDA_SUCCESS) {
    holomq::log_literal(holomq::LogLevel::Warn,
                        "cuStreamDestroy failed during holomq::cuda::Stream teardown");
  }
  stream_ = nullptr;
}

Stream::~Stream() noexcept {
  drain_and_destroy();
}

}  // namespace cuda
}  // namespace holomq
