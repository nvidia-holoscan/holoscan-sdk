/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "scoped_file_descriptor.hpp"

#include <cstdint>
#include <functional>
#include <mutex>
#include <unordered_map>

namespace holomq {

/// Thread-safe publisher-side fd-pass export table: \p pool_cookie → dup of the exported VMM POSIX
/// fd.
///
/// \p pool_cookie must match \c detail::device::BufferHandleData::pool_cookie or \c
/// detail::host::BufferHandleData::gen_pool_cookie on the wire. The FD-pass server calls \ref
/// dup_exported_fd from its worker thread; pools register on create and unregister on destroy from
/// application threads.
class VmmPoolExportRegistry {
 public:
  /// Registers \p dup_fd for \p pool_cookie (replaces any prior entry). Ignores an empty \p dup_fd.
  void register_vmm_fd_export(std::uint64_t pool_cookie,
                              std::function<ScopedFileDescriptor()> dup_fd);
  void unregister_vmm_fd_export(std::uint64_t pool_cookie);

  /// Looks up \p pool_cookie and returns a dup of the export fd (empty on miss or callback
  /// failure).
  [[nodiscard]] ScopedFileDescriptor dup_exported_fd(std::uint64_t pool_cookie);

 private:
  std::mutex mutex_;
  std::unordered_map<std::uint64_t, std::function<ScopedFileDescriptor()>> fd_exports_;
};

}  // namespace holomq
