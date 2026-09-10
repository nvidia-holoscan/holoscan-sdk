/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// QueryPool: GPU timestamp queries for profiling submitted work. A pool is a
// fixed array of timestamp slots written by command lists (see
// CommandList::write_timestamp). Readback is deliberately a three-step
// contract — reset + write + resolve recorded on the GPU timeline, then
// results_ns() on the host — because not every planned backend can read query
// results straight from the pool (D3D12 requires an in-command-list resolve
// into a readback buffer), so the portable API bakes that step in. Backends
// where a step is unnecessary implement it as a recording no-op.

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <span>

#include <holoviz/holoviz_export.h>

#include "common.hpp"

namespace holoviz::rhi {

struct QueryPoolDesc {
  uint32_t timestamp_count = 0;
  const char* debug_name = nullptr;
};

/// Where in the GPU pipeline a timestamp samples the clock. Deliberately not
/// the full backend pipeline-stage vocabulary: elapsed time is defined as the
/// queue-execution span between the two write points, and backends without
/// stage granularity (D3D12) treat both values identically.
enum class TimestampLocation : uint8_t {
  kStart,  ///< Sample as soon as prior commands have started draining.
  kEnd,    ///< Sample once all prior commands have completed.
};

namespace detail {
class QueryPoolImpl;
}

/// A fixed-size pool of GPU timestamp slots. Creation fails if the device's
/// graphics queue cannot timestamp (a hardware capability, universally present
/// on the target GPUs).
///
/// Slot lifecycle, all recorded into a command list: reset_queries() the range,
/// write_timestamp() each slot, resolve_queries() the range. Once the submit
/// carrying the resolve has completed on the device timeline, results_ns()
/// returns the values; the caller is responsible for that ordering. Reading
/// early is a contract violation: backends that can detect it fail with
/// kBackpressured (Vulkan), backends that resolve through a readback buffer
/// return stale data.
class QueryPool {
 public:
  HOLOVIZ_EXPORT explicit QueryPool(std::unique_ptr<detail::QueryPoolImpl> impl);
  HOLOVIZ_EXPORT QueryPool();
  HOLOVIZ_EXPORT QueryPool(QueryPool&&) noexcept;
  HOLOVIZ_EXPORT QueryPool& operator=(QueryPool&&) noexcept;
  QueryPool(const QueryPool&) = delete;
  QueryPool& operator=(const QueryPool&) = delete;
  HOLOVIZ_EXPORT ~QueryPool();

  explicit operator bool() const { return impl_ != nullptr; }

  /// Release the underlying backend resources now instead of at destruction.
  /// \p timeout bounds how long to wait for the object to stop being in use:
  /// zero (the default) fails immediately with ErrorCode::kBackpressured if
  /// the object is in use; a non-zero timeout waits up to that duration before
  /// returning ErrorCode::kBackpressured. Only CommandList currently tracks
  /// in-use state; other objects destroy immediately. On success (and on any
  /// error other than kBackpressured) the handle becomes empty. Safe to call
  /// on an empty handle (no-op success). The destructor calls this
  /// automatically and logs any error.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> destroy(
      std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero());

  [[nodiscard]] HOLOVIZ_EXPORT expected<uint32_t> timestamp_count() const;

  /// Read resolved timestamps [first, first + count) into \p out_ns, in
  /// nanoseconds (backend tick-to-ns conversion already applied). Values are
  /// points on the GPU clock, not durations — only the difference between two
  /// timestamps from the same device is meaningful.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> results_ns(uint32_t first, uint32_t count,
                                                         std::span<uint64_t> out_ns);

  /// Vulkan: answers kVkQueryPool.
  [[nodiscard]] HOLOVIZ_EXPORT expected<NativeObject> native_object(NativeObjectType type) const;

  /// Backend-internal access; not part of the stable API surface.
  /// Precondition: the handle is non-empty.
  detail::QueryPoolImpl& impl() { return *impl_; }
  const detail::QueryPoolImpl& impl() const { return *impl_; }

 private:
  std::unique_ptr<detail::QueryPoolImpl> impl_;
};

}  // namespace holoviz::rhi
