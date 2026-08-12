/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "vmm_pool_import_cache.hpp"

#include "fd_pass/client.hpp"
#include "fd_pass/protocol.hpp"
#include "fd_pass/sockaddr.hpp"

#include <holomq/logger.hpp>

#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <utility>

namespace holomq {

namespace {

/// Read \c CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID for one device.
///
/// \return The host NUMA node the device attaches to, or \c std::nullopt when the driver does
///         not report the attribute.
[[nodiscard]] std::optional<int> device_host_numa_id(const CUdevice device) noexcept {
  int host_numa_id = -1;
  if (cuDeviceGetAttribute(&host_numa_id, CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device) !=
      CUDA_SUCCESS) {
    return std::nullopt;
  }
  return host_numa_id;
}

}  // namespace

namespace detail {

std::optional<std::int32_t> device_ordinal_for_location(
    const CUmemLocation& location, const std::int32_t expected_ordinal) noexcept {
  if (location.type == CU_MEM_LOCATION_TYPE_DEVICE) {
    if (location.id < 0 || (expected_ordinal >= 0 && location.id != expected_ordinal)) {
      return std::nullopt;
    }
    return static_cast<std::int32_t>(location.id);
  }
  if (location.type != CU_MEM_LOCATION_TYPE_HOST_NUMA || location.id < 0) {
    return std::nullopt;
  }

  // HoloMQ creates VMM backing with CU_MEM_LOCATION_TYPE_DEVICE. On coherent platforms, an imported
  // handle for that backing can nevertheless report its physical location as HOST_NUMA. Device-pool
  // descriptors carry the publisher's ordinal, so that path confirms the reported host NUMA node
  // against the exact device and remains unambiguous at any device count.
  //
  // Host split-generation descriptors do not carry the generation pool's device ordinal. That path
  // therefore supports a HOST_NUMA report only when the node resolves to exactly one CUDA device.
  // Several devices can report the same CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID on a multi-GPU coherent
  // host; such an ambiguous import fails closed rather than guessing an ordinal. Explicitly
  // HOST_NUMA-backed pools are outside the current HoloMQ export contract. Supporting either case
  // requires plumbing the generation pool's device ordinal through the host wire format.
  if (expected_ordinal >= 0) {
    CUdevice device{};
    if (cuDeviceGet(&device, expected_ordinal) != CUDA_SUCCESS) {
      return std::nullopt;
    }
    const std::optional<int> host_numa_id = device_host_numa_id(device);
    if (!host_numa_id.has_value() || *host_numa_id != location.id) {
      return std::nullopt;
    }
    return expected_ordinal;
  }

  // Without a wire ordinal the node must resolve to exactly one device.
  int device_count = 0;
  if (cuDeviceGetCount(&device_count) != CUDA_SUCCESS) {
    return std::nullopt;
  }
  std::optional<std::int32_t> resolved;
  for (int ordinal = 0; ordinal < device_count; ++ordinal) {
    CUdevice device{};
    if (cuDeviceGet(&device, ordinal) != CUDA_SUCCESS) {
      continue;
    }
    const std::optional<int> host_numa_id = device_host_numa_id(device);
    if (!host_numa_id.has_value() || *host_numa_id != location.id) {
      continue;
    }
    if (resolved.has_value()) {
      return std::nullopt;  // ambiguous: more than one device on this host NUMA node
    }
    resolved = static_cast<std::int32_t>(ordinal);
  }
  return resolved;
}

}  // namespace detail

VmmPoolImportCache::VmmPoolImportCache(VmmPoolImportCache&& other) noexcept {
  const std::unique_lock lock(other.entries_mutex_);
  entries_ = std::move(other.entries_);
}

VmmPoolImportCache& VmmPoolImportCache::operator=(VmmPoolImportCache&& other) noexcept {
  if (this != &other) {
    const std::scoped_lock lock(entries_mutex_, other.entries_mutex_);
    entries_ = std::move(other.entries_);
  }
  return *this;
}

auto VmmPoolImportCache::VmmPoolImportEntry::import(ScopedFileDescriptor import_fd,
                                                    const std::size_t pool_total_bytes,
                                                    const std::size_t slot_stride_bytes,
                                                    const std::int32_t cuda_device_ordinal) noexcept
    -> std::optional<VmmPoolImportEntry> {
  VmmPoolImportEntry entry;
  entry.pool_total_bytes_ = pool_total_bytes;
  entry.slot_stride_bytes_ = slot_stride_bytes;
  entry.mem_ = cuda::Mem(std::move(import_fd));
  if (!entry.mem_) {
    HOLOMQ_LOG_ERROR("CUDA VMM import failed: cuMemImportFromShareableHandle failed");
    return std::nullopt;
  }

  CUmemAllocationProp prop{};
  if (cuMemGetAllocationPropertiesFromHandle(&prop, entry.mem_.get()) != CUDA_SUCCESS) {
    HOLOMQ_LOG_ERROR("CUDA VMM import failed: cuMemGetAllocationPropertiesFromHandle failed");
    return std::nullopt;
  }
  // A successful cuMemImportFromShareableHandle above establishes the POSIX-FD handle type.
  // cuMemGetAllocationPropertiesFromHandle does not round-trip creation properties for an
  // imported handle: requestedHandleTypes is cleared, and location describes the allocation's
  // physical backing rather than the requested device. Validate the allocation type here and
  // attribute the reported location to a device below.
  if (prop.type != CU_MEM_ALLOCATION_TYPE_PINNED) {
    HOLOMQ_LOG_ERROR(
        "CUDA VMM import failed: imported allocation properties are unsupported "
        "(type={}, location_type={}, location_id={})",
        static_cast<int>(prop.type),
        static_cast<int>(prop.location.type),
        prop.location.id);
    return std::nullopt;
  }

  const std::optional<std::int32_t> resolved_ordinal =
      detail::device_ordinal_for_location(prop.location, cuda_device_ordinal);
  if (!resolved_ordinal.has_value()) {
    HOLOMQ_LOG_ERROR(
        "CUDA VMM import failed: imported allocation location is not attributable to a CUDA "
        "device (type={}, location_type={}, location_id={}, expected_ordinal={})",
        static_cast<int>(prop.type),
        static_cast<int>(prop.location.type),
        prop.location.id,
        cuda_device_ordinal);
    return std::nullopt;
  }
  entry.cuda_device_ordinal_ = *resolved_ordinal;
  if (cuda_device_ordinal >= 0 && cuda_device_ordinal != entry.cuda_device_ordinal_) {
    HOLOMQ_LOG_ERROR(
        "CUDA VMM import failed: CUDA device ordinal mismatch (expected={}, resolved={})",
        cuda_device_ordinal,
        entry.cuda_device_ordinal_);
    return std::nullopt;
  }

  // Granularity and access must describe the owning device, not the imported allocation's
  // physical-backing location, which is CU_MEM_LOCATION_TYPE_HOST_NUMA on coherent platforms.
  CUmemAllocationProp device_prop{};
  device_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  device_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  device_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  device_prop.location.id = entry.cuda_device_ordinal_;

  std::size_t granularity = 0;
  if (cuMemGetAllocationGranularity(
          &granularity, &device_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS) {
    HOLOMQ_LOG_ERROR("CUDA VMM import failed: cuMemGetAllocationGranularity failed");
    return std::nullopt;
  }

  entry.address_reservation_ = cuda::MemAddressReservation(pool_total_bytes, granularity);
  if (!entry.address_reservation_) {
    HOLOMQ_LOG_ERROR("CUDA VMM import failed: cuMemAddressReserve failed");
    return std::nullopt;
  }

  entry.mapping_ =
      cuda::MemMapping(entry.address_reservation_.get(), pool_total_bytes, entry.mem_.get());
  if (!entry.mapping_) {
    HOLOMQ_LOG_ERROR("CUDA VMM import failed: cuMemMap failed");
    return std::nullopt;
  }

  // Subscriber import: read-only VA mapping (pairs with \ref ConstBuffer). The publisher keeps a
  // separate PROT_READWRITE mapping in its process; device writes through this import mapping are
  // rejected by the driver.
  // cuMemSetAccess ignores location.id and grants to the host when the descriptor names
  // CU_MEM_LOCATION_TYPE_HOST_NUMA, so the grant must always name the owning device. Copying the
  // imported allocation's reported location here would leave the GPU without access on coherent
  // platforms.
  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = entry.cuda_device_ordinal_;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
  if (cuMemSetAccess(entry.address_reservation_.get(), pool_total_bytes, &access, 1U) !=
      CUDA_SUCCESS) {
    HOLOMQ_LOG_ERROR("CUDA VMM import failed: cuMemSetAccess failed");
    return std::nullopt;
  }

  return entry;
}

auto VmmPoolImportCache::cached_pool_base(const VmmPoolImportEntry& entry,
                                          const VmmPoolImportDescriptor& descriptor) noexcept
    -> std::optional<VmmPoolImportBase> {
  if (entry.pool_total_bytes() != descriptor.pool_total_bytes) {
    HOLOMQ_LOG_ERROR("CUDA VMM import failed: pool total bytes mismatch");
    return std::nullopt;
  }
  if (entry.slot_stride_bytes() != descriptor.slot_stride_bytes) {
    HOLOMQ_LOG_ERROR("CUDA VMM import failed: slot stride bytes mismatch");
    return std::nullopt;
  }
  if (descriptor.cuda_device_ordinal >= 0 &&
      entry.cuda_device_ordinal() != descriptor.cuda_device_ordinal) {
    HOLOMQ_LOG_ERROR("CUDA VMM import failed: cached CUDA device ordinal mismatch");
    return std::nullopt;
  }
  return VmmPoolImportBase{
      entry.base(),
      entry.pool_total_bytes(),
      entry.slot_stride_bytes(),
      entry.cuda_device_ordinal(),
  };
}

auto VmmPoolImportCache::import_pool_base(const VmmPoolImportDescriptor& descriptor) noexcept
    -> std::optional<VmmPoolImportBase> {
  if (descriptor.pool_total_bytes == 0U || descriptor.slot_stride_bytes == 0U ||
      descriptor.pool_cookie == 0U) {
    HOLOMQ_LOG_ERROR("CUDA VMM import failed: invalid descriptor");
    return std::nullopt;
  }

  {
    const std::shared_lock lock(entries_mutex_);
    const auto it = entries_.find(descriptor.pool_cookie);
    if (it != entries_.end()) {
      return cached_pool_base(it->second, descriptor);
    }
  }

  const std::unique_lock lock(entries_mutex_);
  const auto existing = entries_.find(descriptor.pool_cookie);
  if (existing != entries_.end()) {
    return cached_pool_base(existing->second, descriptor);
  }

  std::string socket_path;
  if (descriptor.fd_pass_socket_path != nullptr) {
    const std::size_t n =
        ::strnlen(descriptor.fd_pass_socket_path, fd_pass::socket_path_field_max());
    if (n != 0U) {
      socket_path.assign(descriptor.fd_pass_socket_path, n);
    }
  }
  if (socket_path.empty()) {
    HOLOMQ_LOG_ERROR("CUDA VMM import failed: fd_pass socket path empty");
    return std::nullopt;
  }

  const fd_pass::Request req{descriptor.pool_cookie};
  auto fd_client_or = fd_pass::Client::create(socket_path);
  if (!fd_client_or.has_value()) {
    HOLOMQ_LOG_ERROR("CUDA VMM import failed: fd_pass client create failed");
    return std::nullopt;
  }
  auto fd_or = std::move(fd_client_or).value().send_request(req);
  if (!fd_or.has_value()) {
    HOLOMQ_LOG_ERROR("CUDA VMM import failed: fd_pass request failed");
    return std::nullopt;
  }
  auto entry_or = VmmPoolImportCache::VmmPoolImportEntry::import(std::move(fd_or).value(),
                                                                 descriptor.pool_total_bytes,
                                                                 descriptor.slot_stride_bytes,
                                                                 descriptor.cuda_device_ordinal);
  if (!entry_or.has_value()) {
    return std::nullopt;
  }
  VmmPoolImportEntry entry = std::move(entry_or).value();
  VmmPoolImportBase result{
      entry.base(),
      entry.pool_total_bytes(),
      entry.slot_stride_bytes(),
      entry.cuda_device_ordinal(),
  };
  entries_.emplace(descriptor.pool_cookie, std::move(entry));
  return result;
}

}  // namespace holomq
