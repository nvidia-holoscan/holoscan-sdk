// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <concepts>
#include <cstdint>
#include <limits>
#include <string_view>

#include <gtest/gtest.h>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/vision/bayer_demosaic_op.hpp>
#include <holoscan/vision/convert_to_op.hpp>
#include <holoscan/vision/copy_make_border_op.hpp>
#include <holoscan/vision/custom_crop_op.hpp>
#include <holoscan/vision/non_maximum_suppression_op.hpp>
#include <holoscan/vision/normalize_op.hpp>
#include <holoscan/vision/reformat_op.hpp>
#include <holoscan/vision/remap_op.hpp>
#include <holoscan/vision/resize_op.hpp>
#include <holoscan/vision/vision_types.hpp>

namespace holoscan::vision {
namespace {

// Default construction lets generic authoring and future language bindings instantiate Vision ops.
static_assert(std::default_initializable<ResizeOp>);
static_assert(!std::constructible_from<ResizeOp, std::int32_t, std::int32_t, std::uint32_t>);
static_assert(std::default_initializable<CustomCropOp>);
static_assert(!std::constructible_from<CustomCropOp, std::int32_t, std::int32_t, std::uint32_t>);
static_assert(std::default_initializable<CopyMakeBorderOp>);
static_assert(
    !std::constructible_from<CopyMakeBorderOp, std::int32_t, std::int32_t, std::uint32_t>);
static_assert(std::default_initializable<RemapOp>);
static_assert(!std::constructible_from<RemapOp, std::int32_t, std::int32_t, std::uint32_t>);
static_assert(std::default_initializable<NormalizeOp>);
static_assert(!std::constructible_from<NormalizeOp, std::int32_t, std::int32_t, std::uint32_t>);
static_assert(std::default_initializable<ConvertToOp>);
static_assert(!std::constructible_from<ConvertToOp, std::int32_t, std::int32_t, std::uint32_t>);
static_assert(std::default_initializable<ReformatOp>);
static_assert(!std::constructible_from<ReformatOp, std::int32_t, std::int32_t, std::uint32_t>);
static_assert(std::default_initializable<NonMaximumSuppressionOp>);
static_assert(!std::constructible_from<NonMaximumSuppressionOp, std::int32_t, std::uint32_t>);
static_assert(std::default_initializable<BayerDemosaicOp>);
static_assert(!std::constructible_from<BayerDemosaicOp, std::int32_t, std::uint32_t>);

// REQ: VISION-003
TEST(VisionCapacityAPI, ImageOperatorsUseImagePorts) {
  EXPECT_TRUE((std::same_as<decltype(ResizeOp::input), Input<schema::ImageT>>));
  EXPECT_TRUE((std::same_as<decltype(ResizeOp::output), Output<schema::ImageT>>));
  EXPECT_TRUE((std::same_as<decltype(CustomCropOp::input), Input<schema::ImageT>>));
  EXPECT_TRUE((std::same_as<decltype(CustomCropOp::output), Output<schema::ImageT>>));
  EXPECT_TRUE((std::same_as<decltype(CopyMakeBorderOp::input), Input<schema::ImageT>>));
  EXPECT_TRUE((std::same_as<decltype(CopyMakeBorderOp::output), Output<schema::ImageT>>));
  EXPECT_TRUE((std::same_as<decltype(RemapOp::input), Input<schema::ImageT>>));
  EXPECT_TRUE((std::same_as<decltype(RemapOp::output), Output<schema::ImageT>>));
  EXPECT_TRUE((std::same_as<decltype(NormalizeOp::input), Input<schema::ImageT>>));
  EXPECT_TRUE((std::same_as<decltype(NormalizeOp::output), Output<schema::ImageT>>));
  EXPECT_TRUE((std::same_as<decltype(ConvertToOp::input), Input<schema::ImageT>>));
  EXPECT_TRUE((std::same_as<decltype(ConvertToOp::output), Output<schema::ImageT>>));
  EXPECT_TRUE((std::same_as<decltype(ReformatOp::input), Input<schema::ImageT>>));
  EXPECT_TRUE((std::same_as<decltype(ReformatOp::output), Output<schema::ImageT>>));
  EXPECT_TRUE((std::same_as<decltype(RemapOp::map), Input<Tensor>>));
  EXPECT_TRUE((std::same_as<decltype(NonMaximumSuppressionOp::boxes), Input<Tensor>>));
  EXPECT_TRUE((std::same_as<decltype(BayerDemosaicOp::input), Input<schema::ImageT>>));
  EXPECT_TRUE((std::same_as<decltype(BayerDemosaicOp::output), Output<schema::ImageT>>));
}

// REQ: VISION-001
// REQ: VISION-011
TEST(VisionCapacityAPI, StructuralParametersRejectInvalidBounds) {
  // Key setup: instantiate each operator through the same default-construction path used by generic
  // authoring, then override one structural field at a time.
  //
  //   resize    nms    border
  //
  // The disconnected vertices exercise independent parameter schemas in one graph.
  Graph graph{"invalid-vision-capacity-parameters"};
  auto resize = graph.op<ResizeOp>("resize");
  auto nms = graph.op<NonMaximumSuppressionOp>("nms");
  auto border = graph.op<CopyMakeBorderOp>("border");

  // Key assertion: parameter validation rejects invalid structural bounds before plan compilation.
  EXPECT_THROW(resize.set_params(ResizeOpParams{.width = 0}), RuntimeError);
  EXPECT_THROW(resize.set_params(ResizeOpParams{.height = -1}), RuntimeError);
  EXPECT_THROW(resize.set_params(ResizeOpParams{.max_batch_size = 0U}), RuntimeError);
  EXPECT_THROW(resize.set_params(ResizeOpParams{.max_batch_size = kMaximumBatchSize + 1U}),
               RuntimeError);
  EXPECT_THROW(nms.set_params(NonMaximumSuppressionOpParams{.max_boxes = 0}), RuntimeError);
  EXPECT_THROW(nms.set_params(NonMaximumSuppressionOpParams{.max_batch_size = 0U}), RuntimeError);
  EXPECT_THROW(
      nms.set_params(NonMaximumSuppressionOpParams{.max_batch_size = kMaximumBatchSize + 1U}),
      RuntimeError);

  // Key assertion: projected byte-span failures retain their typed error codes.
  try {
    resize.set_params(ResizeOpParams{.width = 32768, .height = 32768});
    FAIL() << "expected oversized ResizeOp capacity rejection";
  } catch (const RuntimeError& error) {
    EXPECT_EQ(error.code(), ErrorCode::kNotSupported);
    EXPECT_NE(std::string_view{error.what()}.find("GRAPH_OPERATOR_SETUP_RUNTIME_ERROR"),
              std::string_view::npos);
  }
  try {
    nms.set_params(NonMaximumSuppressionOpParams{
        .max_boxes = std::numeric_limits<std::int32_t>::max() / 8 + 1});
    FAIL() << "expected oversized NonMaximumSuppressionOp boxes rejection";
  } catch (const RuntimeError& error) {
    EXPECT_EQ(error.code(), ErrorCode::kNotSupported);
    EXPECT_NE(std::string_view{error.what()}.find("GRAPH_OPERATOR_SETUP_RUNTIME_ERROR"),
              std::string_view::npos);
  }
  EXPECT_THROW(border.set_params(
                   CopyMakeBorderOpParams{.max_width = 2, .max_height = 1, .left = 1, .right = 1}),
               RuntimeError);
}

}  // namespace
}  // namespace holoscan::vision
