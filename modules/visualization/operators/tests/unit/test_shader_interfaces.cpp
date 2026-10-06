// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Every layer shader's interface header and generated header, included into one translation
// unit. Each interface header opens its own per-shader namespace, so the three PushConstants are
// distinct C++ types; a second header reusing a namespace would fail to compile here (and would
// otherwise violate the one-definition rule across the layer translation units).

#include <gtest/gtest.h>

#include <type_traits>

#include "holoviz_op/backends/holoviz_geometry_interface.hpp"
#include "holoviz_op/backends/holoviz_image_interface.hpp"
#include "holoviz_op/backends/holoviz_lut_interface.hpp"

#include "backends/holoviz_geometry.slang.hpp"
#include "backends/holoviz_image.slang.hpp"
#include "backends/holoviz_lut.slang.hpp"

namespace holoscan::visualization::backends {
namespace {

static_assert(
    !std::is_same_v<detail::holoviz_geometry::PushConstants, detail::holoviz_image::PushConstants>);
static_assert(
    !std::is_same_v<detail::holoviz_image::PushConstants, detail::holoviz_lut::PushConstants>);
static_assert(
    !std::is_same_v<detail::holoviz_lut::PushConstants, detail::holoviz_geometry::PushConstants>);

// The generated headers assert every field offset; this pins the whole-block sizes side by side.
// The geometry shader has one entry-point pair per primitive, all declared with the same block.
TEST(ShaderInterfaces, PushConstantSizesMatchReflection) {
  namespace geometry = detail::holoviz_geometry;
  EXPECT_EQ(sizeof(geometry::PushConstants), geometry::line_list::kPushConstantSize);
  EXPECT_EQ(sizeof(geometry::PushConstants), geometry::line_strip::kPushConstantSize);
  EXPECT_EQ(sizeof(geometry::PushConstants), geometry::cross::kPushConstantSize);
  EXPECT_EQ(sizeof(geometry::PushConstants), geometry::rect::kPushConstantSize);
  EXPECT_EQ(sizeof(geometry::PushConstants), geometry::oval::kPushConstantSize);
  EXPECT_EQ(sizeof(detail::holoviz_image::PushConstants), detail::holoviz_image::kPushConstantSize);
  EXPECT_EQ(sizeof(detail::holoviz_lut::PushConstants), detail::holoviz_lut::kPushConstantSize);
}

}  // namespace
}  // namespace holoscan::visualization::backends
