/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/holomq_export.h>
#include <holomq/buffer_handle.hpp>
#include <holomq/buffer_memory_type.hpp>
#include <holomq/errors.hpp>
#include <holomq/expected.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

struct CUstream_st;

namespace holomq {

class DeviceBufferPoolImpl;
class HostBufferPoolImpl;

/// Mutable pool-owner lease of one fixed-size slot from \ref BufferPool::acquire (device or host
/// pools).
///
/// Move-only: each lease has a single owner; copying is not allowed. Default-constructed and
/// moved-from handles are empty (\c explicit operator bool() is \c false; \c memory_type() is \c
/// BufferMemoryType::None).
class Buffer {
 public:
  using MemoryType = BufferMemoryType;

  HOLOMQ_EXPORT Buffer() noexcept;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  HOLOMQ_EXPORT Buffer(Buffer&&) noexcept;
  HOLOMQ_EXPORT Buffer& operator=(Buffer&&) noexcept;
  HOLOMQ_EXPORT ~Buffer();

  [[nodiscard]] HOLOMQ_EXPORT MemoryType memory_type() const noexcept;
  /// CUDA device ordinal when \ref memory_type is \ref BufferMemoryType::Device; empty otherwise.
  [[nodiscard]] HOLOMQ_EXPORT std::optional<int32_t> cuda_device_ordinal() const noexcept;
  [[nodiscard]] HOLOMQ_EXPORT std::size_t size_bytes() const noexcept;
  [[nodiscard]] HOLOMQ_EXPORT explicit operator bool() const noexcept;

  /// Device-accessible address of this slot (see \ref size_bytes).
  [[nodiscard]] HOLOMQ_EXPORT void* ptr() noexcept;
  [[nodiscard]] HOLOMQ_EXPORT const void* ptr() const noexcept;

  [[nodiscard]] HOLOMQ_EXPORT expected<BufferHandle, Error> export_handle(
      CUstream_st* stream) noexcept;

  /// Return a \ref BufferHandle for this lease without enqueuing a ready-generation set or
  /// deferring ref-count release on a stream.
  ///
  /// @warning Subscribers must not use GPU memory for cross-process visibility until they wait
  /// when the returned handle carries an export fence from a prior \ref enqueue_ready_set. When
  /// the handle carries no export fence, the application must coordinate ordering externally.
  ///
  /// @warning Without a prior \ref enqueue_ready_set, the pool slot may be reused while GPU work
  /// on this lease is still in flight.
  [[nodiscard]] HOLOMQ_EXPORT expected<BufferHandle, Error> export_handle_unsynchronized() noexcept;

  /// Enqueue a ready-generation set on \p stream after prior ops on that stream and defer SHM
  /// ref-count release until \p stream completes prior work. Does not block the caller. Returns
  /// \c false on invalid \p stream or enqueue failure. Idempotent after the first success on this
  /// lease. When \p stream is being captured, the capture records the set and stays valid, and
  /// subscribers wait until the captured graph runs; launch that graph once, before releasing this
  /// lease.
  [[nodiscard]] HOLOMQ_EXPORT bool enqueue_ready_set(CUstream_st* stream) noexcept;

  /// @brief Defer this owner lease's release until prior work on \p stream completes.
  /// @param stream CUDA stream that orders the final use of this owner lease.
  /// @return \c true when release deferral is established; otherwise \c false.
  /// @warning This does not record readiness in the exported handle. Consumers must wait through
  /// another valid readiness contract before accessing this buffer.
  [[nodiscard]] HOLOMQ_EXPORT bool defer_release_on_stream(CUstream_st* stream) noexcept;

  /// @brief Prepare this cold-bound owner lease for another export generation.
  /// @return `true` after all imported views have retired and the export state was reset;
  /// `false` while another process still retains the slot or this lease is empty.
  /// @warning The caller must also finish every local read or write ordered against the previous
  /// generation before reusing the payload bytes.
  [[nodiscard]] HOLOMQ_EXPORT bool prepare_for_reuse() noexcept;

  struct Impl;

 private:
  HOLOMQ_EXPORT explicit Buffer(std::unique_ptr<Impl> impl) noexcept;

  friend class DeviceBufferPoolImpl;
  friend class HostBufferPoolImpl;
  friend class BufferStreamAccess;

  std::unique_ptr<Impl> impl_;
};

}  // namespace holomq
