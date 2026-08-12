/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/holomq_export.h>
#include <holomq/buffer_handle.hpp>
#include <holomq/buffer_memory_type.hpp>
#include <holomq/const_buffer.hpp>
#include <holomq/errors.hpp>
#include <holomq/expected.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>

struct CUstream_st;

namespace holomq {

class Context;

/// @brief Reusable read-only lease over one imported host-buffer pool.
///
/// The first import resolves and retains the backing mappings for one host pool slot. After
/// \ref release, exact-slot import can reacquire a later generation of that slot, while the
/// same-pool rebind API can select another slot without repeating mapping-cache lookup or
/// allocating another buffer implementation. Each active interval owns an independent
/// shared-memory ref-count lease.
///
/// This object is move-only and serialized: callers must not import, release, or access one
/// instance concurrently. The object and its owning Context must outlive every active access.
class HostBufferImport {
 public:
  HOLOMQ_EXPORT HostBufferImport() noexcept;
  HostBufferImport(const HostBufferImport&) = delete;
  HostBufferImport& operator=(const HostBufferImport&) = delete;
  HOLOMQ_EXPORT HostBufferImport(HostBufferImport&& other) noexcept;
  HOLOMQ_EXPORT HostBufferImport& operator=(HostBufferImport&& other) noexcept;
  HOLOMQ_EXPORT ~HostBufferImport();

  /// @brief Return whether this object currently owns an imported slot lease.
  [[nodiscard]] HOLOMQ_EXPORT explicit operator bool() const noexcept;

  /// @brief Return the active imported buffer memory type, or `None` when inactive.
  [[nodiscard]] HOLOMQ_EXPORT BufferMemoryType memory_type() const noexcept;

  /// @brief Return the active imported buffer size, or zero when inactive.
  [[nodiscard]] HOLOMQ_EXPORT std::size_t size_bytes() const noexcept;

  /// @brief Return the active read-only payload pointer, or null when inactive.
  [[nodiscard]] HOLOMQ_EXPORT const void* ptr() const noexcept;

  /// @brief Return the CUDA device ordinal when applicable.
  [[nodiscard]] HOLOMQ_EXPORT std::optional<std::int32_t> cuda_device_ordinal() const noexcept;

  /// @brief Enqueue the active ready-generation wait and arm deferred release.
  /// @param stream Consumer stream on which to enqueue the wait.
  /// @return `true` on success; `false` when inactive or the wait cannot be enqueued.
  [[nodiscard]] HOLOMQ_EXPORT bool enqueue_ready_wait(CUstream_st* stream) noexcept;

  /// @brief Re-export the active imported handle.
  /// @return The handle or a typed error when inactive or export fails.
  [[nodiscard]] HOLOMQ_EXPORT expected<BufferHandle, Error> export_handle() noexcept;

  /// @brief Release the active shared-memory lease while retaining reusable mappings.
  HOLOMQ_EXPORT void release() noexcept;

 private:
  friend class Context;

  ConstBuffer buffer_;
  bool active_{false};
};

}  // namespace holomq
