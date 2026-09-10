// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <type_traits>

#include <holoviz/geometry_topology.hpp>
#include <holoviz/image_color_space.hpp>
#include <holoviz/image_format.hpp>
#include <holoviz/present_mode.hpp>
#include <holoviz/rhi/common.hpp>
#include <holoviz/rhi/pipeline.hpp>

#include <gtest/gtest.h>

namespace holoviz {

// PresentMode is the one enum genuinely shared between the layer API and the RHI: a single
// definition, aliased into holoviz::rhi (same pattern as ErrorCode).
static_assert(std::is_same_v<PresentMode, rhi::PresentMode>);
static_assert(std::is_same_v<std::underlying_type_t<PresentMode>, uint8_t>);

// The remaining pairs are deliberately distinct types: different concepts (source-image vs
// backend format, source-data interpretation vs swapchain surface color space, semantic
// overlay shapes vs GPU primitives).
static_assert(!std::is_same_v<ImageFormat, rhi::Format>);
static_assert(!std::is_same_v<ImageColorSpace, rhi::ColorSpace>);
static_assert(!std::is_same_v<GeometryTopology, rhi::PrimitiveTopology>);

// Public enums carry explicit underlying types.
static_assert(std::is_same_v<std::underlying_type_t<ImageFormat>, uint32_t>);
static_assert(std::is_same_v<std::underlying_type_t<ImageColorSpace>, uint8_t>);
static_assert(std::is_same_v<std::underlying_type_t<GeometryTopology>, uint8_t>);
static_assert(std::is_same_v<std::underlying_type_t<YUVModelConversion>, uint8_t>);
static_assert(std::is_same_v<std::underlying_type_t<YUVRange>, uint8_t>);

TEST(EnumSharingTest, PresentModeAliasSpellsTheSameValues) {
  EXPECT_EQ(rhi::PresentMode::kFIFO, PresentMode::kFIFO);
  EXPECT_EQ(rhi::PresentMode::kImmediate, PresentMode::kImmediate);
}

}  // namespace holoviz
