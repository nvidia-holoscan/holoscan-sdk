/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer_pool/buffer_impl.hpp"

namespace holomq {

std::uint64_t Buffer::Impl::record_ready(CUstream_st* const stream) noexcept {
  if (stream == nullptr || !generation_slots_) {
    return 0U;
  }
  const std::uint64_t generation = next_export_generation();
  if (generation == 0U || !generation_slots_[index_].write(stream, generation)) {
    return 0U;
  }
  lease_export_generation_ = generation;
  return generation;
}

bool Buffer::Impl::wait_ready(CUstream_st* const stream, const std::uint64_t generation) noexcept {
  if (stream == nullptr || generation == 0U || !generation_slots_) {
    return false;
  }
  return generation_slots_[index_].wait(stream, generation);
}

}  // namespace holomq
