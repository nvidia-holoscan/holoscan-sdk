/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace holomq {

/// Non-owning descriptor for one element type within a strided slot array.
///
/// \ref element_address does not bounds-check \p index against a slot count; callers must keep
/// \p index within the mapped pool region.
struct SlotRegion {
  void* base{nullptr};
  std::size_t stride{0U};
  std::size_t element_size{0U};
  std::size_t slot_offset{0U};

  [[nodiscard]] void* element_address(const std::uint32_t index) const noexcept {
    const auto base_address = reinterpret_cast<std::uintptr_t>(base);
    if (base_address == 0U || stride == 0U) {
      return nullptr;
    }
    if (index != 0U &&
        stride > (std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(index))) {
      return nullptr;
    }
    const std::uintptr_t slot = base_address + static_cast<std::size_t>(index) * stride;
    if (slot_offset > (std::numeric_limits<std::uintptr_t>::max() - slot)) {
      return nullptr;
    }
    return reinterpret_cast<void*>(slot + slot_offset);
  }

  [[nodiscard]] explicit operator bool() const noexcept {
    return base != nullptr && stride != 0U && element_size != 0U;
  }
};

/// Payload and generation slot-array descriptors; bases may be shared (interleaved) or independent
/// (split).
struct SlotRegions {
  SlotRegion payload{};
  SlotRegion generation{};
};

inline constexpr std::size_t k_slot_gpu_generation_bytes = sizeof(std::uint64_t);

/// Interleaved layout: one base, `[ uint64_t gpu_generation ][ user payload ][ stride padding ]`
/// per slot.
///
/// Caller must ensure \p slot_stride_bytes >= \ref k_slot_gpu_generation_bytes + \p buffer_bytes.
[[nodiscard]] inline SlotRegions make_interleaved_slot_regions(
    void* base, const std::size_t slot_stride_bytes, const std::size_t buffer_bytes) noexcept {
  SlotRegions regions{};
  regions.generation = SlotRegion{base, slot_stride_bytes, k_slot_gpu_generation_bytes, 0U};
  regions.payload = SlotRegion{base, slot_stride_bytes, buffer_bytes, k_slot_gpu_generation_bytes};
  return regions;
}

/// Split layout: independent payload and generation bases/strides (e.g. host split generation VMM).
///
/// Caller must size strides so each slot fits its element (\p buffer_bytes payload,
/// \ref k_slot_gpu_generation_bytes generation).
[[nodiscard]] inline SlotRegions make_split_slot_regions(void* payload_base,
                                                         const std::size_t payload_stride_bytes,
                                                         void* generation_base,
                                                         const std::size_t generation_stride_bytes,
                                                         const std::size_t buffer_bytes) noexcept {
  SlotRegions regions{};
  regions.payload = SlotRegion{payload_base, payload_stride_bytes, buffer_bytes, 0U};
  regions.generation =
      SlotRegion{generation_base, generation_stride_bytes, k_slot_gpu_generation_bytes, 0U};
  return regions;
}

/// Host devicemap interleaved SHM layout with generation words bound for CUDA stream memops.
///
/// Builds the same slot geometry as \ref make_interleaved_slot_regions (generation at slot start,
/// payload immediately after). \c payload keeps the host \c shm_base; only \c generation.base is
/// replaced with the device pointer from \c cuMemHostGetDevicePointer for the slot-0 generation
/// word. Stride and offsets are unchanged so \c SlotRegion::element_address can address later slots
/// via device-base arithmetic.
///
/// Requires the backing memory to have been registered with \c CU_MEMHOSTREGISTER_DEVICEMAP.
/// Returns
/// \c std::nullopt if device-pointer resolution fails.
[[nodiscard]] inline std::optional<SlotRegions> make_host_devicemap_interleaved_slot_regions(
    void* shm_base, const std::size_t slot_stride_bytes, const std::size_t buffer_bytes) noexcept {
  SlotRegions regions = make_interleaved_slot_regions(shm_base, slot_stride_bytes, buffer_bytes);

  // regions.generation.base is the host address of slot 0's generation word (start of shm_base).
  // cuMemHostGetDevicePointer requires CU_MEMHOSTREGISTER_DEVICEMAP on the backing allocation.
  CUdeviceptr dev_gen0{};
  if (cuMemHostGetDevicePointer(&dev_gen0, regions.generation.base, 0) != CUDA_SUCCESS) {
    return std::nullopt;
  }

  // Store the device VA in generation.base only; stride and slot_offset are unchanged so
  // element_address(index) reaches slot N via dev_gen0 + index * slot_stride_bytes.
  regions.generation.base = reinterpret_cast<void*>(static_cast<std::uintptr_t>(dev_gen0));
  return regions;
}

}  // namespace holomq
