/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Buffer + BufferSlice. There is deliberately no per-buffer usage flag: every
// buffer is created usable as vertex/index/storage/indirect/copy source and
// destination (usage bits are near-free on the target hardware), so callers
// pick a size and share mode and nothing else. BufferSlice is the universal
// currency — binding, vertex/index setup, and copies all take slices, which
// is what makes the one-big-buffer + suballocate pattern natural.
//
// Ownership model (all HoloViz objects): the public class is a move-only value
// handle owning a heap-allocated backend implementation. Exactly one live
// handle corresponds to each backend object; destroying the handle destroys
// the backend object. Resources can also be released early and in a
// controlled order via destroy(), which returns any error from the underlying
// free calls; the destructor is the RAII fallback that calls destroy() and
// logs failures. The detail::*Impl classes are the pure-virtual plug-in
// points backends derive from — not part of the user-facing API; their
// definitions live in src/detail/.

#pragma once

#include <chrono>
#include <memory>

#include <cuda.h>

#include <holoviz/holoviz_export.h>

#include "common.hpp"

namespace holoviz {

struct BufferDesc {
  size_t size_bytes = 0;
  ShareMode share = ShareMode::kPrivate;
  const char* debug_name = nullptr;
};

class Buffer;

namespace detail {
class BufferImpl;
}  // namespace detail

/// A non-owning byte range of a Buffer. Cheap to copy and pass by value; the
/// referenced Buffer must stay alive for as long as any slice of it is in use.
/// Slices reference the backend object, so they stay valid across handle moves.
struct BufferSlice {
  const detail::BufferImpl* buffer = nullptr;
  size_t offset_bytes = 0;
  size_t size_bytes = 0;

  inline BufferSlice(const Buffer& buffer, size_t offset_bytes, size_t size_bytes);

  BufferSlice(const detail::BufferImpl& buffer, size_t offset_bytes, size_t size_bytes)
      : buffer(&buffer), offset_bytes(offset_bytes), size_bytes(size_bytes) {}

  /// A slice's whole is itself — infallible identity, provided so generic code can
  /// take either a Buffer (whose whole() is fallible) or a BufferSlice uniformly.
  [[nodiscard]] BufferSlice whole() const { return *this; }

  /// Return a sub-slice; offsets are relative to this slice, not the buffer.
  expected<BufferSlice> slice(size_t offset_bytes, size_t size_bytes) const {
    if (buffer == nullptr) {
      return make_unexpected(Error{.code = ErrorCode::kInvalidState,
                                   .message = "BufferSlice::slice on an empty slice"});
    }
    if (offset_bytes > this->size_bytes || size_bytes > this->size_bytes - offset_bytes) {
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument, .message = "BufferSlice::slice out of range"});
    }
    return BufferSlice(*buffer, this->offset_bytes + offset_bytes, size_bytes);
  }
};

/// Device-local buffer memory. Note the actual allocation may be larger than
/// the requested size (allocator granularity)
class Buffer {
 public:
  HOLOVIZ_EXPORT explicit Buffer(std::unique_ptr<detail::BufferImpl> impl);
  HOLOVIZ_EXPORT Buffer();
  HOLOVIZ_EXPORT Buffer(Buffer&&) noexcept;
  HOLOVIZ_EXPORT Buffer& operator=(Buffer&&) noexcept;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  HOLOVIZ_EXPORT ~Buffer();

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

  [[nodiscard]] HOLOVIZ_EXPORT expected<BufferSlice> whole() const;
  [[nodiscard]] HOLOVIZ_EXPORT expected<BufferSlice> slice(size_t offset_bytes,
                                                           size_t size_bytes) const;

  [[nodiscard]] HOLOVIZ_EXPORT expected<CUdeviceptr> cuda_device_ptr() const;

  /// Vulkan: answers kVkDevice, kVkBuffer, kVmaAllocation.
  [[nodiscard]] HOLOVIZ_EXPORT expected<NativeObject> native_object(NativeObjectType type) const;

  /// Backend-internal access; not part of the stable API surface.
  /// Precondition: the handle is non-empty.
  detail::BufferImpl& impl() { return *impl_; }
  const detail::BufferImpl& impl() const { return *impl_; }

  /// Backend-internal; not part of the stable API surface. Abandon the handle WITHOUT
  /// destroying the underlying resource: unretired GPU/CUDA work may still reference it,
  /// so a deliberate leak beats a use-after-free. Allocation-free and no-throw by
  /// construction — safe inside terminal error paths. The caller must poison the owning
  /// device (DeviceImpl::report_leaked_resources()) so it never destroys the leaked
  /// child's native parents.
  void leak() noexcept { [[maybe_unused]] auto* abandoned = impl_.release(); }

 private:
  std::unique_ptr<detail::BufferImpl> impl_;
};

inline BufferSlice::BufferSlice(const Buffer& buffer, size_t offset_bytes, size_t size_bytes)
    : buffer(buffer ? &buffer.impl() : nullptr),
      offset_bytes(offset_bytes),
      size_bytes(size_bytes) {}

}  // namespace holoviz
