/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer_pool/generation_slot_array.hpp"
#include "buffer_pool/payload_slot_array.hpp"
#include "buffer_pool/slot_region.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace {

constexpr std::size_t kBufferBytes = 256U;
constexpr std::size_t kSlotStride = 512U;
constexpr std::uint32_t kSlotIndex = 2U;

[[nodiscard]] std::uintptr_t expected_generation_address(const std::uintptr_t pool_base,
                                                         const std::uint32_t index) noexcept {
  return pool_base + static_cast<std::size_t>(index) * kSlotStride;
}

[[nodiscard]] std::uintptr_t expected_payload_address(const std::uintptr_t pool_base,
                                                      const std::uint32_t index) noexcept {
  return expected_generation_address(pool_base, index) + holomq::k_slot_gpu_generation_bytes;
}

}  // namespace

TEST(SlotRegionTest, OwnerInterleavedBindingMatchesExpectedAddresses) {
  std::vector<std::byte> backing(kSlotStride * 4U);
  const auto pool_base = reinterpret_cast<std::uintptr_t>(backing.data());

  const holomq::SlotRegions regions =
      holomq::make_interleaved_slot_regions(backing.data(), kSlotStride, kBufferBytes);
  const holomq::PayloadSlotArray payload_slots(regions.payload);
  const holomq::GenerationSlotArray generation_slots(regions.generation);

  ASSERT_EQ(reinterpret_cast<std::uintptr_t>(generation_slots[kSlotIndex].device_address()),
            expected_generation_address(pool_base, kSlotIndex));
  ASSERT_EQ(reinterpret_cast<std::uintptr_t>(payload_slots[kSlotIndex].ptr()),
            expected_payload_address(pool_base, kSlotIndex));
  EXPECT_EQ(payload_slots[kSlotIndex].size_bytes(), kBufferBytes);
}

TEST(SlotRegionTest, SubscriberSlotBaseBindingUsesIndexZero) {
  std::vector<std::byte> slot(kSlotStride);
  const auto slot_base = reinterpret_cast<std::uintptr_t>(slot.data());

  const holomq::SlotRegions regions =
      holomq::make_interleaved_slot_regions(slot.data(), kSlotStride, kBufferBytes);
  const holomq::PayloadSlotArray payload_slots(regions.payload);
  const holomq::GenerationSlotArray generation_slots(regions.generation);

  ASSERT_EQ(reinterpret_cast<std::uintptr_t>(generation_slots[0U].device_address()), slot_base);
  ASSERT_EQ(reinterpret_cast<std::uintptr_t>(payload_slots[0U].ptr()),
            slot_base + holomq::k_slot_gpu_generation_bytes);
}

TEST(SlotRegionTest, SplitFactoryMatchesManualRegions) {
  std::vector<std::byte> payload_backing(kSlotStride * 2U);
  std::vector<std::byte> generation_backing(kSlotStride * 2U);

  const holomq::SlotRegions regions = holomq::make_split_slot_regions(
      payload_backing.data(), kSlotStride, generation_backing.data(), kSlotStride, kBufferBytes);
  const holomq::PayloadSlotArray payload_slots(regions.payload);
  const holomq::GenerationSlotArray generation_slots(regions.generation);

  const auto payload_base = reinterpret_cast<std::uintptr_t>(payload_backing.data());
  const auto generation_base = reinterpret_cast<std::uintptr_t>(generation_backing.data());

  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(payload_slots[1U].ptr()), payload_base + kSlotStride);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(generation_slots[1U].device_address()),
            generation_base + kSlotStride);
}

TEST(SlotRegionTest, SplitRegionsUseIndependentBases) {
  std::vector<std::byte> payload_backing(kSlotStride * 2U);
  std::vector<std::byte> generation_backing(kSlotStride * 2U);

  const holomq::SlotRegion payload_region{
      payload_backing.data(),
      kSlotStride,
      kBufferBytes,
      0U,
  };
  const holomq::SlotRegion generation_region{
      generation_backing.data(),
      kSlotStride,
      holomq::k_slot_gpu_generation_bytes,
      0U,
  };

  const holomq::PayloadSlotArray payload_slots(payload_region);
  const holomq::GenerationSlotArray generation_slots(generation_region);

  const auto payload_base = reinterpret_cast<std::uintptr_t>(payload_backing.data());
  const auto generation_base = reinterpret_cast<std::uintptr_t>(generation_backing.data());

  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(payload_slots[1U].ptr()), payload_base + kSlotStride);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(generation_slots[1U].device_address()),
            generation_base + kSlotStride);
}
