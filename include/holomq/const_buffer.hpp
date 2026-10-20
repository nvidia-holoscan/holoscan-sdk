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

class DeviceConstBufferImpl;
class HostConstBufferImpl;
class HostBufferImport;

/// Read-only subscriber import view from \ref Context::import_buffer or
/// \ref Context::import_buffer_unsynchronized.
///
/// Move-only: each import lease has a single owner; copying is not allowed. Default-constructed and
/// moved-from handles are empty (\c explicit operator bool() is \c false; \c memory_type() is \c
/// BufferMemoryType::None).
///
/// \c CUdeviceptr is not implicitly convertible to \c void*; import views use
/// \c reinterpret_cast&lt;const void*&gt;(static_cast&lt;std::uintptr_t&gt;(device_base)).
class ConstBuffer {
 public:
  using MemoryType = BufferMemoryType;

  HOLOMQ_EXPORT ConstBuffer() noexcept;
  ConstBuffer(const ConstBuffer&) = delete;
  ConstBuffer& operator=(const ConstBuffer&) = delete;
  HOLOMQ_EXPORT ConstBuffer(ConstBuffer&&) noexcept;
  HOLOMQ_EXPORT ConstBuffer& operator=(ConstBuffer&&) noexcept;
  HOLOMQ_EXPORT ~ConstBuffer();

  [[nodiscard]] HOLOMQ_EXPORT MemoryType memory_type() const noexcept;
  /// CUDA device ordinal when \ref memory_type is \ref BufferMemoryType::Device; empty otherwise.
  [[nodiscard]] HOLOMQ_EXPORT std::optional<int32_t> cuda_device_ordinal() const noexcept;
  [[nodiscard]] HOLOMQ_EXPORT std::size_t size_bytes() const noexcept;
  [[nodiscard]] HOLOMQ_EXPORT explicit operator bool() const noexcept;

  /// Device-accessible payload address (see \ref size_bytes).
  [[nodiscard]] HOLOMQ_EXPORT const void* ptr() const noexcept;

  /// Return whether this imported lease carries a producer CUDA readiness fence.
  ///
  /// A fence-free active lease was exported without asynchronous producer work and needs no CUDA
  /// wait before host access. A fenced lease still requires \ref enqueue_ready_wait or equivalent
  /// synchronization; this query does not report fence completion and does not mutate release
  /// ordering. Empty and released leases return \c false.
  [[nodiscard]] HOLOMQ_EXPORT bool has_producer_readiness_fence() const noexcept;

  /// Enqueue a ready-generation wait on \p stream until the publisher's export-time GPU work is
  /// visible, and arm deferred ref release on success. Does not block the caller. No-op success
  /// when the handle carries no export fence. Outside stream capture, idempotent on each registered
  /// stream. Register every consumer stream before GPU use; release waits for all of them.
  /// Caller-owned streams must remain alive until this lease is released. When \p stream is being
  /// captured, the capture records its own wait and stays valid. That wait orders only the captured
  /// graph, so a later call outside that capture enqueues the wait again. Launch the captured graph
  /// before releasing this lease, since graph work that runs after release can read a reused slot.
  /// Returns \c false when \p stream is invalid and a wait is required.
  [[nodiscard]] HOLOMQ_EXPORT bool enqueue_ready_wait(CUstream_st* stream) noexcept;

  /// Re-publish the imported handle without enqueuing a new ready-generation set. Deferred ref
  /// release stays ordered after every stream registered by \ref import_buffer or
  /// \ref enqueue_ready_wait.
  [[nodiscard]] HOLOMQ_EXPORT expected<BufferHandle, Error> export_handle() noexcept;

  /// @brief Release the active imported lease while retaining reusable pool mappings.
  /// @note A later @ref Context::rebind_buffer_unsynchronized call may reactivate this control for
  /// another slot in the same producer pool without allocating a new implementation.
  HOLOMQ_EXPORT void release() noexcept;

  struct Impl;

 private:
  HOLOMQ_EXPORT explicit ConstBuffer(std::unique_ptr<Impl> impl) noexcept;

  friend class DeviceConstBufferImpl;
  friend class HostConstBufferImpl;
  friend class HostBufferImport;
  friend class Context;
  friend class BufferStreamAccess;

  std::unique_ptr<Impl> impl_;
};

}  // namespace holomq
