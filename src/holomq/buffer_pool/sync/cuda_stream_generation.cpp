/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer_pool/sync/cuda_stream_generation.hpp"

namespace holomq {

bool stream_write_generation(CUstream_st* const stream, const CUdeviceptr address,
                             const std::uint64_t value) noexcept {
  if (stream == nullptr || address == 0U) {
    return false;
  }
  return cuStreamWriteValue64(reinterpret_cast<CUstream>(stream),
                              address,
                              value,
                              static_cast<unsigned long long>(CU_STREAM_WRITE_VALUE_DEFAULT)) ==
         CUDA_SUCCESS;
}

bool stream_wait_generation(CUstream_st* const stream, const CUdeviceptr address,
                            const std::uint64_t value) noexcept {
  if (stream == nullptr || address == 0U) {
    return false;
  }
  return cuStreamWaitValue64(reinterpret_cast<CUstream>(stream),
                             address,
                             value,
                             static_cast<unsigned long long>(CU_STREAM_WAIT_VALUE_GEQ)) ==
         CUDA_SUCCESS;
}

}  // namespace holomq
