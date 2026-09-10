/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <sys/un.h>

namespace holomq::detail::host {

/// Serializable host pool handle payload (POSIX SHM payload identity + optional split-gen-pool
/// import).
///
/// \c owner_pid with \c shm_pool_id derives the payload SHM attach name; \c owner_pid with \c
/// shm_ref_count_id identifies the ref-count segment (with \ref BufferHandle slot index for the
/// slot lease).
///
/// \c payload_pinned selects whether remotes call \c cuMemHostRegister on attach. When set, \c
/// host_register_flags (including \c 0 for CUDA default register) is passed to \c
/// cuMemHostRegister.
///
/// Layout when pinned is selected by \c host_register_flags:
/// - **Devicemap interleaved** (\c CU_MEMHOSTREGISTER_DEVICEMAP): generation words live in the
/// payload SHM
///   mapping; \c slot_stride_bytes includes the GPU generation word. Split-gen-pool / fd-pass
///   fields are unused.
/// - **Split generation VMM** (\c CU_MEMHOSTREGISTER_DEVICEMAP clear): payload SHM is payload-only
///   (\c slot_stride_bytes == \c buffer_bytes); generation words live in a separate publisher VMM
///   pool imported via \c gen_pool_* and \c fd_pass_socket_path.
///
/// When \c payload_pinned is clear, payload SHM is pageable (split-gen VMM only; mmap only on
/// attach).
/// \c host_register_flags must be \c 0; non-zero flags with pageable payload are rejected at handle
/// validation.
///
struct BufferHandleData {
  /// \c sizeof(\c sockaddr_un::sun_path) for this compile target (typically 108 on Linux).
  static constexpr std::size_t fd_pass_socket_path_max = sizeof(::sockaddr_un::sun_path);

  /// Creator process id; with \c shm_pool_id / \c shm_ref_count_id derives POSIX SHM attach names.
  std::uint32_t owner_pid{};
  /// Per-pool payload SHM segment id assigned at \ref HostBufferPoolMemory create.
  std::uint32_t shm_pool_id{};
  /// Per-pool ref-count SHM segment id assigned at \ref ShmRefCountMapping::create.
  std::uint32_t shm_ref_count_id{};
  /// **Devicemap interleaved:** stride covers generation word + user payload + padding.
  /// **Split generation VMM:** payload-only stride (\c == buffer_bytes).
  std::size_t slot_stride_bytes{};
  /// User-visible payload bytes per slot (excludes GPU generation word and stride padding).
  std::size_t buffer_bytes{};
  /// Payload SHM region size for attach validation (\c HostBufferPoolMemory::size()).
  std::size_t pool_total_bytes{};
  /// Non-zero when payload SHM is registered with CUDA on attach; zero for pageable payload.
  std::uint8_t payload_pinned{};
  /// \c cuMemHostRegister flags when \c payload_pinned is set (\c 0 is CUDA default register).
  unsigned int host_register_flags{};

  // Split generation VMM pool (when \c CU_MEMHOSTREGISTER_DEVICEMAP is clear).

  /// Opaque generation VMM pool id for fd-pass import (per-pool id from \ref VmmExportedPool;
  /// \c buffer_pool/splitmix64_id.hpp).
  std::uint64_t gen_pool_cookie{};
  /// Byte stride between generation words in the gen pool (typically \c
  /// k_slot_gpu_generation_bytes).
  std::size_t gen_slot_stride_bytes{};
  /// Full exportable gen-pool mapping size for import validation / \c cuMemMap.
  std::size_t gen_pool_total_bytes{};
  /// NUL-terminated fd_pass endpoint (same max and semantics as \ref
  /// holomq::detail::device::BufferHandleData).
  char fd_pass_socket_path[fd_pass_socket_path_max]{};
};

static_assert(std::is_standard_layout_v<BufferHandleData>);
static_assert(std::is_trivially_copyable_v<BufferHandleData>);

}  // namespace holomq::detail::host
