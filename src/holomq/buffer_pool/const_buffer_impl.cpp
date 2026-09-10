/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer_pool/const_buffer_impl.hpp"

namespace holomq {

bool ConstBuffer::Impl::wait_ready(CUstream_st* const stream,
                                   const std::uint64_t generation) noexcept {
  if (stream == nullptr || generation == 0U || !generation_slots_) {
    return false;
  }
  return generation_slots_[index_].wait(stream, generation);
}

}  // namespace holomq
