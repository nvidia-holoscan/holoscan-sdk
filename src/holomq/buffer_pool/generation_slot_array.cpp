/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer_pool/generation_slot_array.hpp"

#include "buffer_pool/sync/cuda_stream_generation.hpp"

#include <cuda.h>

namespace holomq {

bool GenerationSlotArray::Slot::write(CUstream_st* const stream,
                                      const std::uint64_t value) const noexcept {
  const auto address = reinterpret_cast<std::uintptr_t>(device_address());
  if (address == 0U) {
    return false;
  }
  return stream_write_generation(stream, static_cast<CUdeviceptr>(address), value);
}

bool GenerationSlotArray::Slot::wait(CUstream_st* const stream,
                                     const std::uint64_t generation) const noexcept {
  const auto address = reinterpret_cast<std::uintptr_t>(device_address());
  if (address == 0U) {
    return false;
  }
  return stream_wait_generation(stream, static_cast<CUdeviceptr>(address), generation);
}

}  // namespace holomq
