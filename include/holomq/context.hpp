/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/holomq_export.h>
#include <holomq/buffer_pool.hpp>
#include <holomq/const_buffer.hpp>
#include <holomq/device_buffer_allocator.hpp>
#include <holomq/errors.hpp>
#include <holomq/expected.hpp>
#include <holomq/host_buffer_allocator.hpp>
#include <holomq/publisher_builder.hpp>
#include <holomq/pubsub_capabilities.hpp>
#include <holomq/subscriber_builder.hpp>
#include <holomq/wait_set.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

struct CUstream_st;

namespace holomq {

class HostBufferImport;

class ContextBuilder;
class ContextImpl;

/// Owns a shared transport session (\ref ContextImpl). Buffer pools hold \c
/// std::shared_ptr<ContextImpl> so session state outlives this shell when pools remain.
class Context {
  friend class ContextBuilder;

 public:
  Context(Context&&) noexcept = default;
  ~Context() = default;
  Context& operator=(Context&&) noexcept = default;
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  /// When \c true, the internal FD-pass server is running. Required for \ref
  /// create_device_buffer_pool,
  /// \ref create_host_buffer_pool, and split sync-pool \ref create_host_pinned_buffer_pool.
  /// Exception: devicemap interleaved pinned host (\c CU_MEMHOSTREGISTER_DEVICEMAP) does not
  /// require publisher FD-pass.
  [[nodiscard]] HOLOMQ_EXPORT bool vmm_fd_export_enabled() const noexcept;

  /// Resolved FD-pass listen path (non-empty; embedded in buffer descriptors). A server listens on
  /// this path only when \ref vmm_fd_export_enabled is \c true.
  [[nodiscard]] HOLOMQ_EXPORT const std::string& fd_pass_socket_path() const noexcept;

  /// @brief Return setup-time typed pub/sub capabilities for this context's compiled backend.
  /// @return Capability flags for the backend selected when HoloMQ was compiled.
  [[nodiscard]] HOLOMQ_EXPORT PubSubCapabilities pubsub_capabilities() const noexcept;

  /// Adopt the handle and, when it carries an export fence, enqueue a wait on \p stream until the
  /// publisher's GPU work is visible. \ref ErrorCode::kInvalidArgument when \p stream is null while
  /// a wait is required.
  [[nodiscard]] HOLOMQ_EXPORT auto import_buffer(const BufferHandle& handle,
                                                 CUstream_st* stream) noexcept
      -> expected<ConstBuffer, Error>;

  /// Import a buffer handle without waiting for the publisher's export-time GPU work.
  ///
  /// @warning Using \ref ConstBuffer::ptr for GPU work before
  /// \ref ConstBuffer::enqueue_ready_wait is undefined for cross-process visibility when the
  /// handle carries an export fence.
  [[nodiscard]] HOLOMQ_EXPORT auto import_buffer_unsynchronized(const BufferHandle& handle) noexcept
      -> expected<ConstBuffer, Error>;

  /// @brief Import or retarget one reusable buffer control within its original producer pool.
  /// @param handle Buffer handle to import; a rebound handle must retain the original memory type,
  /// pool identity, geometry, and device while it may select another slot.
  /// @param destination Inactive reusable destination. Its first import may resolve mappings;
  /// later imports reuse them and acquire a fresh shared-memory lease.
  /// @return Success or a typed invalid-handle, active-destination, or import error.
  /// @warning Accessing the imported pointer before enqueueing any required ready wait is undefined
  /// for cross-process visibility.
  [[nodiscard]] HOLOMQ_EXPORT auto rebind_buffer_unsynchronized(const BufferHandle& handle,
                                                                ConstBuffer& destination) noexcept
      -> expected<void, Error>;

  /// @brief Import or reacquire one exact host-buffer slot without waiting for producer work.
  /// @param handle Host buffer handle to import. Reacquisition must identify the same pool slot
  /// as the first successful import into \p destination.
  /// @param destination Inactive reusable destination. Its first import may resolve mappings;
  /// later imports reuse those mappings and acquire a fresh shared-memory lease.
  /// @return Success or a typed invalid-handle/import error.
  /// @warning Using \ref HostBufferImport::ptr for GPU work before enqueueing any required ready
  /// wait is undefined for cross-process visibility.
  [[nodiscard]] HOLOMQ_EXPORT auto import_host_buffer_unsynchronized(
      const BufferHandle& handle, HostBufferImport& destination) noexcept -> expected<void, Error>;

  /// @brief Import or retarget one reusable control within its original host-buffer pool.
  /// @param handle Host buffer handle to import. After the first successful import, later handles
  /// must identify the same pool and geometry but may select a different slot.
  /// @param destination Inactive reusable destination whose retained pool mappings are reused.
  /// @return Success or a typed invalid-handle/import error.
  /// @warning Using @ref HostBufferImport::ptr for GPU work before enqueueing any required ready
  /// wait is undefined for cross-process visibility.
  [[nodiscard]] HOLOMQ_EXPORT auto rebind_host_buffer_unsynchronized(
      const BufferHandle& handle, HostBufferImport& destination) noexcept -> expected<void, Error>;

  /// @brief Import a host-buffer handle into its preallocated pool-slot destination.
  /// @param handle Host buffer handle whose opaque slot index selects one entry in \p destinations.
  /// @param destinations Pool-sized reusable destination table. The selected entry must be inactive
  /// and must either be unused or already bound to the same exact producer pool slot.
  /// @return The selected destination index on success, or a typed invalid-handle, capacity, or
  /// import error.
  /// @warning Using the selected \ref HostBufferImport::ptr for GPU work before enqueueing any
  /// required ready wait is undefined for cross-process visibility.
  [[nodiscard]] HOLOMQ_EXPORT auto import_host_buffer_unsynchronized(
      const BufferHandle& handle, std::span<HostBufferImport> destinations) noexcept
      -> expected<std::size_t, Error>;

  /// Create a device \ref BufferPool with \p buffer_bytes per slot and \p buffer_count slots.
  /// Requires \ref vmm_fd_export_enabled (\ref ContextBuilder::with_vmm_fd_export on the
  /// publisher). Empty \p allocator selects the built-in CUDA VMM allocator.
  [[nodiscard]] HOLOMQ_EXPORT auto create_device_buffer_pool(
      std::size_t buffer_bytes, std::uint32_t buffer_count,
      DeviceBufferAllocator allocator = {}) const noexcept -> expected<BufferPool, Error>;

  /// Create a pageable host \ref BufferPool (payload SHM + separate sync VMM pool).
  /// Requires \ref vmm_fd_export_enabled (\ref ContextBuilder::with_vmm_fd_export on the
  /// publisher). Empty \p allocator selects the built-in SHM allocator.
  [[nodiscard]] HOLOMQ_EXPORT auto create_host_buffer_pool(
      std::size_t buffer_bytes, std::uint32_t buffer_count,
      HostBufferAllocator allocator = {}) const noexcept -> expected<BufferPool, Error>;

  /// Create a host-pinned \ref BufferPool; \p cuda_host_register_flags is passed to \c
  /// cuMemHostRegister (including \c 0 for CUDA default register). Requires \ref
  /// vmm_fd_export_enabled when
  /// \c CU_MEMHOSTREGISTER_DEVICEMAP is clear (separate sync-pool layout). Devicemap interleaved
  /// (\c DEVICEMAP set) does not require publisher FD-pass. Empty \p allocator selects the built-in
  /// SHM allocator.
  [[nodiscard]] HOLOMQ_EXPORT auto create_host_pinned_buffer_pool(
      std::size_t buffer_bytes, std::uint32_t buffer_count, unsigned int cuda_host_register_flags,
      HostBufferAllocator allocator = {}) const noexcept -> expected<BufferPool, Error>;

  /// Starts building a publish–subscribe publisher for \p service_name: call
  /// \c payload_type&lt;Payload&gt;() then \c create().
  [[nodiscard]] HOLOMQ_EXPORT PublisherServiceBuilder publisher_builder(std::string service_name);

  /// Starts building a subscriber for \p service_name: call \c payload_type&lt;Payload&gt;() then
  /// \c create().
  [[nodiscard]] HOLOMQ_EXPORT SubscriberServiceBuilder subscriber_builder(std::string service_name);

  [[nodiscard]] HOLOMQ_EXPORT auto create_waitset() -> expected<WaitSet, Error>;

 private:
  HOLOMQ_EXPORT explicit Context(std::shared_ptr<ContextImpl> impl) noexcept;

  std::shared_ptr<ContextImpl> impl_;
};

}  // namespace holomq
