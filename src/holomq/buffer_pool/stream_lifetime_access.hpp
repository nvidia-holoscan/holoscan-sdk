/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/holomq_export.h>
#include <holomq/buffer.hpp>
#include <holomq/const_buffer.hpp>
#include <memory>
#include "../cuda/stream_lifetime.hpp"

namespace holomq {

// Source-tree-only seam: the installed Buffer/ConstBuffer API and opaque layouts stay stable.
// Core and HoloMQ share header-defined atomic wait/notify on these lifetime records. Cross-DSO
// wakeups rely on libstdc++'s shared waiter pool, pinned in ci/abi/holomq.exports.baseline.txt.
class BufferStreamAccess {
 public:
  [[nodiscard]] HOLOMQ_EXPORT static bool register_consumer(
      Buffer& buffer, CUstream_st* stream,
      const std::shared_ptr<const cuda::StreamLifetime>& lifetime) noexcept;
  [[nodiscard]] HOLOMQ_EXPORT static bool register_consumer(
      ConstBuffer& buffer, CUstream_st* stream,
      const std::shared_ptr<const cuda::StreamLifetime>& lifetime) noexcept;
  [[nodiscard]] HOLOMQ_EXPORT static bool wait(
      ConstBuffer& buffer, CUstream_st* stream,
      const std::shared_ptr<const cuda::StreamLifetime>& lifetime) noexcept;
  [[nodiscard]] HOLOMQ_EXPORT static expected<BufferHandle, Error> export_handle(
      Buffer& buffer, CUstream_st* stream,
      const std::shared_ptr<const cuda::StreamLifetime>& lifetime) noexcept;
};

}  // namespace holomq
