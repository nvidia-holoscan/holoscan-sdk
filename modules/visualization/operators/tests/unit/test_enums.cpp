// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <type_traits>

#include <holoscan/visualization/holoviz_op.hpp>
#include <holoviz/common.hpp>
#include <holoviz/pipeline.hpp>

#include "common/holoviz_conversions.hpp"

#include "holoviz_op/types.hpp"

#include <gtest/gtest.h>

namespace holoscan::visualization {

// Every public enum is a distinct type from its HoloViz counterpart so the installed operator
// headers carry no HoloViz header. PresentMode maps 1:1 (checked below); the remaining pairs are
// different concepts (source-image vs backend format, source-data interpretation vs swapchain
// surface color space, semantic overlay shapes vs GPU primitives).
static_assert(!std::is_same_v<PresentMode, holoviz::PresentMode>);
static_assert(std::is_same_v<std::underlying_type_t<PresentMode>, uint8_t>);
static_assert(!std::is_same_v<ImageFormat, holoviz::Format>);
static_assert(!std::is_same_v<GeometryTopology, holoviz::PrimitiveTopology>);

// Public enums carry explicit underlying types.
static_assert(std::is_same_v<std::underlying_type_t<ImageFormat>, uint32_t>);
static_assert(std::is_same_v<std::underlying_type_t<GeometryTopology>, uint8_t>);
static_assert(std::is_same_v<std::underlying_type_t<YUVModelConversion>, uint8_t>);
static_assert(std::is_same_v<std::underlying_type_t<YUVRange>, uint8_t>);

// The layer backends share HoloViz core's error vocabulary: the bridge header
// (holoviz_op/types.hpp)
// must resolve the unqualified names to the holoviz types inside this namespace.
static_assert(std::is_same_v<Error, holoviz::Error>);
static_assert(std::is_same_v<ErrorCode, holoviz::ErrorCode>);
static_assert(std::is_same_v<expected<int>, holoviz::expected<int>>);

TEST(LayerEnumTest, PresentModeRoundTripsThroughHoloviz) {
  constexpr std::array<PresentMode, 6> kModes = {{PresentMode::kAuto,
                                                  PresentMode::kFIFO,
                                                  PresentMode::kFIFOLatestReady,
                                                  PresentMode::kMailbox,
                                                  PresentMode::kImmediate,
                                                  PresentMode::kFIFORelaxed}};
  for (const PresentMode mode : kModes) {
    EXPECT_EQ(common::to_present_mode(common::to_holoviz_present_mode(mode)), mode);
  }
  EXPECT_EQ(common::to_holoviz_present_mode(PresentMode::kFIFOLatestReady),
            holoviz::PresentMode::kFIFOLatestReady);
  EXPECT_EQ(common::to_present_mode(holoviz::PresentMode::kImmediate), PresentMode::kImmediate);
  EXPECT_EQ(common::to_holoviz_present_mode(PresentMode::kFIFORelaxed),
            holoviz::PresentMode::kFIFORelaxed);
}

// PresentMode values feed PresentModeSet bit positions and are append-only by contract; pin the
// numbering on both sides so an accidental reorder fails loudly.
static_assert(static_cast<uint8_t>(PresentMode::kFIFORelaxed) == 5);
static_assert(static_cast<uint8_t>(holoviz::PresentMode::kFIFORelaxed) == 5);

TEST(LayerEnumTest, ErrorVocabularyIsHoloviz) {
  const expected<int> failed = make_unexpected(Error{.code = ErrorCode::kInvalidArgument});
  ASSERT_FALSE(failed.has_value());
  EXPECT_EQ(failed.error().code, holoviz::ErrorCode::kInvalidArgument);
}

}  // namespace holoscan::visualization
