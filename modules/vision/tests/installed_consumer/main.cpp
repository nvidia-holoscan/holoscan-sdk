// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <concepts>

#include <holoscan/vision/convert_to_op.hpp>
#include <holoscan/vision/copy_make_border_op.hpp>
#include <holoscan/vision/custom_crop_op.hpp>
#include <holoscan/vision/non_maximum_suppression_op.hpp>
#include <holoscan/vision/normalize_op.hpp>
#include <holoscan/vision/reformat_op.hpp>
#include <holoscan/vision/remap_op.hpp>
#include <holoscan/vision/resize_op.hpp>
#include <holoscan/vision/vision_types.hpp>

static_assert(std::derived_from<holoscan::vision::ResizeOp,
                                holoscan::Operator<holoscan::vision::ResizeOpParams>>);
static_assert(std::derived_from<holoscan::vision::CustomCropOp,
                                holoscan::Operator<holoscan::vision::CustomCropOpParams>>);
static_assert(std::derived_from<holoscan::vision::CopyMakeBorderOp,
                                holoscan::Operator<holoscan::vision::CopyMakeBorderOpParams>>);
static_assert(std::derived_from<holoscan::vision::RemapOp,
                                holoscan::Operator<holoscan::vision::RemapOpParams>>);
static_assert(std::derived_from<holoscan::vision::NormalizeOp,
                                holoscan::Operator<holoscan::vision::NormalizeOpParams>>);
static_assert(std::derived_from<holoscan::vision::ConvertToOp,
                                holoscan::Operator<holoscan::vision::ConvertToOpParams>>);
static_assert(std::derived_from<holoscan::vision::ReformatOp,
                                holoscan::Operator<holoscan::vision::ReformatOpParams>>);
static_assert(
    std::derived_from<holoscan::vision::NonMaximumSuppressionOp,
                      holoscan::Operator<holoscan::vision::NonMaximumSuppressionOpParams>>);

int main() {
  constexpr holoscan::vision::VisionImageCapacity kImageCapacity{
      .max_width = 1, .max_height = 1, .max_batch_size = 1U};
  holoscan::vision::ResizeOp resize{kImageCapacity};
  holoscan::vision::CustomCropOp crop{kImageCapacity};
  holoscan::vision::CopyMakeBorderOp border{kImageCapacity};
  holoscan::vision::RemapOp remap{kImageCapacity};
  holoscan::vision::NormalizeOp normalize{kImageCapacity};
  holoscan::vision::ConvertToOp convert{kImageCapacity};
  holoscan::vision::ReformatOp reformat{kImageCapacity};
  holoscan::vision::NonMaximumSuppressionOp nms{
      holoscan::vision::NMSCapacity{.max_boxes = 1, .max_batch_size = 1U}};
  const bool ports_are_available =
      resize.input.name() == "input" && crop.output.name() == "output" &&
      border.input.name() == "input" && remap.map.name() == "map" &&
      normalize.output.name() == "output" && convert.input.name() == "input" &&
      reformat.output.name() == "output" && nms.boxes.name() == "boxes" &&
      nms.mask.name() == "mask";
  return ports_are_available ? 0 : 1;
}
