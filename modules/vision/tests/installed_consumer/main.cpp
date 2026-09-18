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
  holoscan::vision::ResizeOp resize;
  holoscan::vision::CustomCropOp crop;
  holoscan::vision::CopyMakeBorderOp border;
  holoscan::vision::RemapOp remap;
  holoscan::vision::NormalizeOp normalize;
  holoscan::vision::ConvertToOp convert;
  holoscan::vision::ReformatOp reformat;
  holoscan::vision::NonMaximumSuppressionOp nms;
  const bool ports_are_available =
      resize.input.name() == "input" && crop.output.name() == "output" &&
      border.input.name() == "input" && remap.map.name() == "map" &&
      normalize.output.name() == "output" && convert.input.name() == "input" &&
      reformat.output.name() == "output" && nms.boxes.name() == "boxes" &&
      nms.mask.name() == "mask";
  return ports_are_available ? 0 : 1;
}
