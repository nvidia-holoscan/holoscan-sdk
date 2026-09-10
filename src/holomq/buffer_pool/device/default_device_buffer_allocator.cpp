/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/device_buffer_allocator.hpp>

#include "cuda/mem.hpp"
#include "cuda/mem_address_reservation.hpp"
#include "cuda/mem_mapping.hpp"

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

namespace holomq {
namespace {

[[nodiscard]] std::optional<CUmemAllocationProp> vmm_allocation_prop_for_current_device() noexcept {
  CUdevice device{};
  if (cuCtxGetDevice(&device) != CUDA_SUCCESS) {
    return std::nullopt;
  }

  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = static_cast<int>(device);
  return prop;
}

class CudaVmmBufferPoolMemory final : public DeviceBufferPoolMemory {
 public:
  CudaVmmBufferPoolMemory(cuda::MemAddressReservation reservation, cuda::Mem mem,
                          cuda::MemMapping mapping, std::size_t size_bytes) noexcept
      : reservation_(std::move(reservation)),
        mem_(std::move(mem)),
        mapping_(std::move(mapping)),
        size_bytes_(size_bytes) {}

  [[nodiscard]] void* ptr() noexcept override {
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(reservation_.get()));
  }

  [[nodiscard]] const void* ptr() const noexcept override {
    return reinterpret_cast<const void*>(static_cast<std::uintptr_t>(reservation_.get()));
  }

  [[nodiscard]] std::size_t size() const noexcept override { return size_bytes_; }

 private:
  cuda::MemAddressReservation reservation_;
  cuda::Mem mem_;
  cuda::MemMapping mapping_;
  std::size_t size_bytes_{0};
};

[[nodiscard]] std::unique_ptr<DeviceBufferPoolMemory> allocate_cuda_vmm_mapping(
    const std::size_t number_of_bytes) {
  if (number_of_bytes == 0U) {
    return nullptr;
  }

  const auto prop_or = vmm_allocation_prop_for_current_device();
  if (!prop_or.has_value()) {
    return nullptr;
  }
  const CUmemAllocationProp& prop = prop_or.value();

  const std::size_t granularity = recommended_device_vmm_granularity();
  if (granularity == 0U) {
    return nullptr;
  }

  cuda::MemAddressReservation reservation(number_of_bytes, granularity);
  if (!reservation) {
    return nullptr;
  }

  cuda::Mem mem(number_of_bytes, &prop, 0ULL);
  if (!mem) {
    return nullptr;
  }

  const CUdeviceptr region_base = reservation.get();
  cuda::MemMapping mapping(region_base, number_of_bytes, mem.get());
  if (!mapping) {
    return nullptr;
  }

  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = prop.location.id;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  if (cuMemSetAccess(region_base, number_of_bytes, &access, 1U) != CUDA_SUCCESS) {
    return nullptr;
  }

  return std::make_unique<CudaVmmBufferPoolMemory>(
      std::move(reservation), std::move(mem), std::move(mapping), number_of_bytes);
}

}  // namespace

std::size_t recommended_device_vmm_granularity() noexcept {
  const auto prop_or = vmm_allocation_prop_for_current_device();
  if (!prop_or.has_value()) {
    return 0U;
  }

  std::size_t granularity = 0;
  if (cuMemGetAllocationGranularity(
          &granularity, &*prop_or, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS) {
    return 0U;
  }
  return granularity;
}

bool device_supports_vmm_posix_fd_export() noexcept {
  CUdevice device{};
  if (cuCtxGetDevice(&device) != CUDA_SUCCESS) {
    return false;
  }

  int posix_fd_supported = 0;
  if (cuDeviceGetAttribute(&posix_fd_supported,
                           CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED,
                           device) != CUDA_SUCCESS) {
    return false;
  }
  return posix_fd_supported != 0;
}

bool device_vmm_allocation_is_exportable(const CUmemGenericAllocationHandle handle) noexcept {
  CUmemAllocationProp prop{};
  if (cuMemGetAllocationPropertiesFromHandle(&prop, handle) != CUDA_SUCCESS) {
    return false;
  }
  if (prop.type != CU_MEM_ALLOCATION_TYPE_PINNED) {
    return false;
  }
  if (prop.location.type != CU_MEM_LOCATION_TYPE_DEVICE) {
    return false;
  }
  if ((prop.requestedHandleTypes & CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) == 0U) {
    return false;
  }
  return true;
}

DeviceBufferAllocator default_device_buffer_allocator() {
  return allocate_cuda_vmm_mapping;
}

}  // namespace holomq
