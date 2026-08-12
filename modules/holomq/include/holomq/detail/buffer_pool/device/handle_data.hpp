/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <sys/un.h>

namespace holomq::detail::device {

/// Serializable device-pool handle payload (fd_pass + VMM geometry + ref-count SHM identity).
///
/// \c pool_cookie is an opaque per-pool id assigned at VMM pool creation (SplitMix64; see
/// \c buffer_pool/splitmix64_id.hpp). It keys fd_pass import and the subscriber
/// \ref VmmPoolImportCache.
/// \c owner_pid and \c shm_ref_count_id identify the POSIX SHM ref-count segment (with \ref
/// BufferHandle slot index for the slot lease).
/// \c slot_stride_bytes is the full slot stride (GPU generation word, user payload, and alignment
/// padding); \c buffer_bytes is the user-visible payload byte length per slot; \c pool_total_bytes
/// is the full exportable mapping size for \c cuMemMap after fd_pass import (subscribers map at
/// slot index × \c slot_stride_bytes).
///
/// \c fd_pass_socket_path is the NUL-terminated fd_pass endpoint: a filesystem path (when holomq
/// is built with
/// \c HOLOMQ_FD_PASS_SOCKET=PATH) or a Linux abstract socket name without a leading NUL (CMake
/// default on Linux:
/// \c HOLOMQ_FD_PASS_SOCKET=ABSTRACT). Publisher and subscribers must agree on the string and on
/// the build mode.
///
struct BufferHandleData {
  /// \c sizeof(\c sockaddr_un::sun_path) for this compile target (typically 108 on Linux).
  static constexpr std::size_t fd_pass_socket_path_max = sizeof(::sockaddr_un::sun_path);

  std::uint64_t pool_cookie{};
  /// Creator process id; with \c shm_ref_count_id derives ref-count SHM name.
  std::uint32_t owner_pid{};
  /// Per-pool ref-count SHM segment id assigned at \ref ShmRefCountMapping::create.
  std::uint32_t shm_ref_count_id{};
  /// Byte stride between consecutive pool slots (generation word, user payload, and alignment
  /// padding).
  std::size_t slot_stride_bytes{};
  /// User-visible payload bytes per slot (excludes GPU generation word and stride padding).
  std::size_t buffer_bytes{};
  /// Full exportable VMM mapping size (\c DeviceBufferPoolMemory::size() from the pool allocator);
  /// used after import for \c cuMemMap.
  std::size_t pool_total_bytes{};
  /// CUDA device ordinal the exported VMM physically resides on (the publisher's device at pool
  /// creation). Subscribers must use this — not the importing thread's current context — for the
  /// VMM location id (\c cuMemGetAllocationGranularity / \c cuMemSetAccess) and for the imported
  /// buffer's reported device, so cross-process import is correct on a nonzero device. \c -1 marks
  /// an unset/invalid ordinal (rejected by \ref is_valid_device_buffer_wire).
  std::int32_t cuda_device_ordinal{-1};
  char fd_pass_socket_path[fd_pass_socket_path_max]{};
};

static_assert(std::is_standard_layout_v<BufferHandleData>);
static_assert(std::is_trivially_copyable_v<BufferHandleData>);

}  // namespace holomq::detail::device
