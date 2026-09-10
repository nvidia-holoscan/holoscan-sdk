// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <concepts>
#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/graph.hpp>
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

// Catch a return to positional image-pool construction, which can transpose width and height.
static_assert(std::constructible_from<ResizeOp, VisionImageCapacity>);
static_assert(!std::default_initializable<ResizeOp>);
static_assert(!std::constructible_from<ResizeOp, std::int32_t, std::int32_t, std::uint32_t>);

// Catch a return to positional crop-pool construction, which can transpose width and height.
static_assert(std::constructible_from<CustomCropOp, VisionImageCapacity>);
static_assert(!std::default_initializable<CustomCropOp>);
static_assert(!std::constructible_from<CustomCropOp, std::int32_t, std::int32_t, std::uint32_t>);

// Catch a return to positional border-pool construction, which can transpose width and height.
static_assert(std::constructible_from<CopyMakeBorderOp, VisionImageCapacity>);
static_assert(!std::default_initializable<CopyMakeBorderOp>);
static_assert(
    !std::constructible_from<CopyMakeBorderOp, std::int32_t, std::int32_t, std::uint32_t>);

// Catch a return to positional remap-pool construction, which can transpose width and height.
static_assert(std::constructible_from<RemapOp, VisionImageCapacity>);
static_assert(!std::default_initializable<RemapOp>);
static_assert(!std::constructible_from<RemapOp, std::int32_t, std::int32_t, std::uint32_t>);

// Catch a return to positional normalization-pool construction, which obscures input admission.
static_assert(std::constructible_from<NormalizeOp, VisionImageCapacity>);
static_assert(!std::default_initializable<NormalizeOp>);
static_assert(!std::constructible_from<NormalizeOp, std::int32_t, std::int32_t, std::uint32_t>);

// Catch a return to positional conversion-pool construction, which obscures input admission.
static_assert(std::constructible_from<ConvertToOp, VisionImageCapacity>);
static_assert(!std::default_initializable<ConvertToOp>);
static_assert(!std::constructible_from<ConvertToOp, std::int32_t, std::int32_t, std::uint32_t>);

// Catch a return to positional reformat-pool construction, which obscures input admission.
static_assert(std::constructible_from<ReformatOp, VisionImageCapacity>);
static_assert(!std::default_initializable<ReformatOp>);
static_assert(!std::constructible_from<ReformatOp, std::int32_t, std::int32_t, std::uint32_t>);

// Catch a return to positional NMS-pool construction, which can swap box and batch limits.
static_assert(std::constructible_from<NonMaximumSuppressionOp, NMSCapacity>);
static_assert(!std::default_initializable<NonMaximumSuppressionOp>);
static_assert(!std::constructible_from<NonMaximumSuppressionOp, std::int32_t, std::uint32_t>);

TEST(VisionCapacityAPI, ResizeRejectsZeroMaximumWidthBeforePoolMaterialization) {
  Graph graph{"zero-image-width-capacity"};
  EXPECT_THROW(
      (void)graph.op<ResizeOp>(
          "resize", VisionImageCapacity{.max_width = 0, .max_height = 1, .max_batch_size = 1U}),
      RuntimeError);
}

TEST(VisionCapacityAPI, ResizeRejectsNegativeMaximumHeightBeforePoolMaterialization) {
  Graph graph{"negative-image-height-capacity"};
  EXPECT_THROW(
      (void)graph.op<ResizeOp>(
          "resize", VisionImageCapacity{.max_width = 1, .max_height = -1, .max_batch_size = 1U}),
      RuntimeError);
}

TEST(VisionCapacityAPI, ResizeRejectsZeroMaximumBatchBeforePoolMaterialization) {
  Graph graph{"zero-image-batch-capacity"};
  EXPECT_THROW(
      (void)graph.op<ResizeOp>(
          "resize", VisionImageCapacity{.max_width = 1, .max_height = 1, .max_batch_size = 0U}),
      RuntimeError);
}

TEST(VisionCapacityAPI, NMSRejectsZeroMaximumBoxesBeforePoolMaterialization) {
  Graph graph{"zero-nms-box-capacity"};
  EXPECT_THROW((void)graph.op<NonMaximumSuppressionOp>(
                   "nms", NMSCapacity{.max_boxes = 0, .max_batch_size = 1U}),
               RuntimeError);
}

TEST(VisionCapacityAPI, NMSRejectsZeroMaximumBatchBeforePoolMaterialization) {
  Graph graph{"zero-nms-batch-capacity"};
  EXPECT_THROW((void)graph.op<NonMaximumSuppressionOp>(
                   "nms", NMSCapacity{.max_boxes = 1, .max_batch_size = 0U}),
               RuntimeError);
}

TEST(VisionCapacityAPI, ResizeRejectsBatchAboveVisionKernelLimitBeforePoolMaterialization) {
  Graph graph{"oversized-image-batch-capacity"};
  EXPECT_THROW((void)graph.op<ResizeOp>(
                   "resize",
                   VisionImageCapacity{
                       .max_width = 1, .max_height = 1, .max_batch_size = kMaximumBatchSize + 1U}),
               RuntimeError);
}

TEST(VisionCapacityAPI, NMSRejectsBatchAboveVisionKernelLimitBeforePoolMaterialization) {
  Graph graph{"oversized-nms-batch-capacity"};
  EXPECT_THROW((void)graph.op<NonMaximumSuppressionOp>(
                   "nms", NMSCapacity{.max_boxes = 1, .max_batch_size = kMaximumBatchSize + 1U}),
               RuntimeError);
}

TEST(VisionCapacityAPI, NMSRejectsCapacityWhoseBoxesExceedCvCudaAddressing) {
  Graph graph{"oversized-nms-box-address-span"};
  constexpr std::int32_t kFirstOversizedBoxCount = std::numeric_limits<std::int32_t>::max() / 8 + 1;
  EXPECT_THROW((void)graph.op<NonMaximumSuppressionOp>(
                   "nms", NMSCapacity{.max_boxes = kFirstOversizedBoxCount, .max_batch_size = 1U}),
               RuntimeError);
}

}  // namespace
}  // namespace holoscan::vision
