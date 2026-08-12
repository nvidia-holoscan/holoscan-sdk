/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cuda/mem.hpp"
#include "cuda/mem_address_reservation.hpp"
#include "cuda/mem_mapping.hpp"
#include "scoped_file_descriptor.hpp"

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <shared_mutex>
#include <unordered_map>

namespace holomq {

namespace detail {

/// Resolve the CUDA device ordinal that owns an imported VMM allocation's reported location.
///
/// A locally created allocation reports \c CU_MEM_LOCATION_TYPE_DEVICE, where \c location.id is
/// already the device ordinal. An allocation observed through \c cuMemImportFromShareableHandle
/// is instead described by its physical backing, so on coherent/UMA platforms it reports
/// \c CU_MEM_LOCATION_TYPE_HOST_NUMA with \c location.id set to a host NUMA node id. Measured on
/// GB10 (DGX Spark), where CPU and GPU share one LPDDR5X pool exposed as a single NUMA node, a
/// pool created as \c DEVICE / id 0 reads back as \c HOST_NUMA / id 0. Map that node back to a
/// device through \c CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID rather than trusting \c location.id as an
/// ordinal.
///
/// \param location         Location reported for the imported allocation.
/// \param expected_ordinal Publisher's wire ordinal, or \c -1 when unknown. When nonnegative the
///                         mapping is confirmed against that exact device; otherwise the node
///                         must resolve to exactly one device.
/// \return The owning device ordinal, or \c std::nullopt when the location cannot be attributed.
///
/// Exposed for unit testing; not part of the installed HoloMQ API.
[[nodiscard]] std::optional<std::int32_t> device_ordinal_for_location(
    const CUmemLocation& location, std::int32_t expected_ordinal) noexcept;

}  // namespace detail

/// Neutral fd-pass import descriptor for a full VMM pool (device payload or host split generation
/// pool).
struct VmmPoolImportDescriptor {
  std::uint64_t pool_cookie{};
  std::size_t slot_stride_bytes{};
  std::size_t pool_total_bytes{};
  const char* fd_pass_socket_path{nullptr};
  /// Expected CUDA device ordinal for the exported VMM allocation. A nonnegative value from the
  /// publisher's wire is checked against the imported allocation's actual properties. \c -1
  /// accepts the allocation's reported device (the host split-generation pool path).
  std::int32_t cuda_device_ordinal{-1};
};

/// Imported pool mapping summary returned by \ref VmmPoolImportCache::import_pool_base (no CUDA
/// ownership).
struct VmmPoolImportBase {
  CUdeviceptr base{};
  std::size_t pool_total_bytes{0U};
  std::size_t slot_stride_bytes{0U};
  /// Device ordinal reported by \c cuMemGetAllocationPropertiesFromHandle for this import.
  std::int32_t cuda_device_ordinal{-1};
};

/// Subscriber-side import path for device \ref BufferPool and host split-generation pools: imports
/// publisher VMM over fd-pass (once per \c pool_cookie), maps with read-only device access (\c
/// CU_MEM_ACCESS_FLAGS_PROT_READ). One instance per \ref Context, shared across subscribers.
///
/// The application must have initialized the CUDA driver (\c cuInit) and have a valid \c CUcontext
/// current on the calling thread before \ref import_pool_base. The importer queries the imported
/// handle's allocation properties once and attributes the reported location to a CUDA device: a
/// \c CU_MEM_LOCATION_TYPE_DEVICE location supplies the ordinal directly, while a
/// \c CU_MEM_LOCATION_TYPE_HOST_NUMA location (reported on coherent/UMA platforms such as GB10)
/// is mapped back through \c CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID. A nonnegative expected ordinal
/// that does not match the attributed device is rejected. Granularity and access always name the
/// owning device rather than the imported allocation's physical-backing location. Import teardown
/// (\c cuMemUnmap / \c cuMemRelease / \c cuMemAddressFree) runs when \ref VmmPoolImportCache is
/// destroyed.
///
/// \p descriptor.fd_pass_socket_path must be non-empty when import is required.
///
/// Concurrent cache hits use shared access. The first import for a pool is serialized so only one
/// caller performs FD transfer and CUDA mapping.
///
/// Cache eviction for imported pools is unspecified today (entries persist until cache
/// destruction). See
/// \c design-notes/vmm-pool-import-cache-entry-eviction.md.
class VmmPoolImportCache {
 public:
  VmmPoolImportCache() = default;

  VmmPoolImportCache(const VmmPoolImportCache&) = delete;
  VmmPoolImportCache& operator=(const VmmPoolImportCache&) = delete;
  VmmPoolImportCache(VmmPoolImportCache&& other) noexcept;
  VmmPoolImportCache& operator=(VmmPoolImportCache&& other) noexcept;

  /// Import (or reuse cached import for) the full pool described by \p descriptor and return its
  /// mapped base.
  [[nodiscard]] auto import_pool_base(const VmmPoolImportDescriptor& descriptor) noexcept
      -> std::optional<VmmPoolImportBase>;

 private:
  struct VmmPoolImportEntry {
    VmmPoolImportEntry() = default;
    VmmPoolImportEntry(const VmmPoolImportEntry&) = delete;
    VmmPoolImportEntry& operator=(const VmmPoolImportEntry&) = delete;
    VmmPoolImportEntry(VmmPoolImportEntry&&) noexcept = default;
    VmmPoolImportEntry& operator=(VmmPoolImportEntry&&) noexcept = default;

    static auto import(ScopedFileDescriptor import_fd, std::size_t pool_total_bytes,
                       std::size_t slot_stride_bytes, std::int32_t cuda_device_ordinal) noexcept
        -> std::optional<VmmPoolImportEntry>;

    [[nodiscard]] CUdeviceptr base() const noexcept { return address_reservation_.get(); }

    [[nodiscard]] std::size_t pool_total_bytes() const noexcept { return pool_total_bytes_; }

    [[nodiscard]] std::size_t slot_stride_bytes() const noexcept { return slot_stride_bytes_; }

    [[nodiscard]] std::int32_t cuda_device_ordinal() const noexcept { return cuda_device_ordinal_; }

    /// Member order matches device pool backing: mapping unmapped first, then \c cuMemRelease, then
    /// \c cuMemAddressFree (see \c CudaVmmBufferPoolMemory in \c
    /// default_device_buffer_allocator.cpp).
    cuda::MemAddressReservation address_reservation_;
    cuda::Mem mem_;
    cuda::MemMapping mapping_;
    std::size_t pool_total_bytes_{};
    std::size_t slot_stride_bytes_{};
    std::int32_t cuda_device_ordinal_{-1};
  };

  [[nodiscard]] static auto cached_pool_base(const VmmPoolImportEntry& entry,
                                             const VmmPoolImportDescriptor& descriptor) noexcept
      -> std::optional<VmmPoolImportBase>;

  std::shared_mutex entries_mutex_;
  std::unordered_map<std::uint64_t, VmmPoolImportEntry> entries_;
};

}  // namespace holomq
