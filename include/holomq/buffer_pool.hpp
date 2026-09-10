/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/holomq_export.h>
#include <holomq/buffer.hpp>
#include <holomq/buffer_memory_type.hpp>
#include <holomq/errors.hpp>
#include <holomq/expected.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

namespace holomq {

/// Owns a fixed-size buffer pool (device VMM or host SHM backing).
///
/// Obtain via \ref Context::create_device_buffer_pool, \ref Context::create_host_buffer_pool,
/// \ref Context::create_host_pinned_buffer_pool, or \ref BufferPool::create. Factory failure
/// returns an error; there is no default-constructed empty pool. Move-only (not copyable).
class BufferPool {
 public:
  /// Construct a pool via \p ImplType::try_create; failures are logged by the implementation.
  template <typename ImplType, typename... Args>
  [[nodiscard]] static auto create(Args&&... args) noexcept -> expected<BufferPool, Error> {
    auto impl_or = ImplType::try_create(std::forward<Args>(args)...);
    if (!impl_or.has_value()) {
      return holomq::forward_error(impl_or);
    }
    return BufferPool(std::move(impl_or).value());
  }

  BufferPool(const BufferPool&) = delete;
  BufferPool& operator=(const BufferPool&) = delete;
  HOLOMQ_EXPORT BufferPool(BufferPool&&) noexcept;
  HOLOMQ_EXPORT BufferPool& operator=(BufferPool&&) noexcept;
  HOLOMQ_EXPORT ~BufferPool();

  /// Backing memory type for slots in this pool.
  [[nodiscard]] HOLOMQ_EXPORT BufferMemoryType memory_type() const noexcept;

  /// CUDA device ordinal when \ref memory_type is \ref BufferMemoryType::Device; empty for host
  /// pools.
  [[nodiscard]] HOLOMQ_EXPORT std::optional<int32_t> cuda_device_ordinal() const noexcept;

  /// User-visible payload bytes per slot (excludes per-slot internal sync metadata and stride
  /// padding).
  [[nodiscard]] HOLOMQ_EXPORT std::size_t buffer_bytes() const noexcept;

  /// Fixed number of slots in the pool for the lifetime of this object.
  [[nodiscard]] HOLOMQ_EXPORT std::uint32_t buffer_count() const noexcept;

  /// \c true when this handle owns a valid pool implementation; \c false after move-from.
  [[nodiscard]] HOLOMQ_EXPORT explicit operator bool() const noexcept;

  /// Take the next free slot (CAS ref-count 0→1). \ref ErrorCode::kBackpressured when exhausted.
  [[nodiscard]] HOLOMQ_EXPORT auto acquire() noexcept -> expected<Buffer, Error>;

  struct Impl;

 private:
  HOLOMQ_EXPORT explicit BufferPool(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;
};

}  // namespace holomq
