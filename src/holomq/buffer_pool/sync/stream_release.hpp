/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/holomq_export.h>
#include <memory>
#include <mutex>
#include <vector>

#include "../../cuda/stream_lifetime.hpp"
#include "../ref_count/shm_ref_count.hpp"

namespace holomq {

// Reserve release metadata at admission. The group holds one SHM reference until every
// consumer completes; no extra SHM references or allocations are needed at last release.
class BufferStreamRelease {
 public:
  struct Group;
  struct Owners {
    std::shared_ptr<void> payload;
    std::shared_ptr<void> generation;
  };
  struct Entry;
  // Wrap CUDA backing once at cache admission. All aliases share preallocated last-owner
  // cleanup; arbitrary impl/cache destruction must never call CUDA on a host-function thread.
  [[nodiscard]] static bool defer_cuda_owners(Owners& owners, int device) noexcept;
  BufferStreamRelease() noexcept;
  ~BufferStreamRelease();
  BufferStreamRelease(BufferStreamRelease&&) noexcept;
  BufferStreamRelease& operator=(BufferStreamRelease&&) noexcept;
  [[nodiscard]] bool add(const ShmRefCountMappingPtr& mapping, std::uint32_t slot,
                         CUstream_st* stream,
                         const std::shared_ptr<const cuda::StreamLifetime>& lifetime = {},
                         const Owners& owners = {}) noexcept;
  [[nodiscard]] bool contains(CUstream_st* stream) const noexcept;
  [[nodiscard]] bool empty() const noexcept;
  void mark_ready(CUstream_st* stream) noexcept;
  void release() noexcept;
  void prepare_for_reuse() noexcept;

 private:
  mutable std::mutex mutex_;
  std::shared_ptr<Group> group_;
  std::vector<std::unique_ptr<Entry>> entries_;
};

// Endpoint stream retirement also runs on the worker: last Tensor release may be a CUDA
// host function. This owner has preallocated teardown metadata before exposing its handle.
class EndpointStream {
 public:
  /// Create a stream on the current device's retained primary context, or the first visible
  /// device when no context is current. Restore the caller's context and reserve lifetime and
  /// teardown metadata before exposing the handle.
  [[nodiscard]] HOLOMQ_EXPORT static std::unique_ptr<EndpointStream> create() noexcept;
  /// Enqueue native retirement without CUDA calls, waits, or allocation on this thread.
  HOLOMQ_EXPORT ~EndpointStream();
  /// Drain queued worker tasks on a normal host thread. Unresolved Closing nodes and GPU
  /// callbacks are not waited for; this must not be called from a CUDA host function.
  HOLOMQ_EXPORT static void drain_retirements() noexcept;
  /// Return the borrowed stream handle while this endpoint owner remains alive.
  [[nodiscard]] HOLOMQ_EXPORT CUstream_st* get() const noexcept;
  /// Return stable identity independently of the endpoint owner and its native handle.
  [[nodiscard]] HOLOMQ_EXPORT const std::shared_ptr<const cuda::StreamLifetime>& lifetime()
      const noexcept;

 private:
  explicit EndpointStream(std::unique_ptr<BufferStreamRelease::Entry> entry) noexcept;
  std::unique_ptr<BufferStreamRelease::Entry> entry_;
};

}  // namespace holomq
