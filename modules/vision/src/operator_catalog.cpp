// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <holoscan/vision/operator_catalog.hpp>

#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/vision/bayer_demosaic_op.hpp>
#include <holoscan/vision/convert_to_op.hpp>
#include <holoscan/vision/copy_make_border_op.hpp>
#include <holoscan/vision/custom_crop_op.hpp>
#include <holoscan/vision/non_maximum_suppression_op.hpp>
#include <holoscan/vision/normalize_op.hpp>
#include <holoscan/vision/reformat_op.hpp>
#include <holoscan/vision/remap_op.hpp>
#include <holoscan/vision/resize_op.hpp>

namespace holoscan::vision {

HOLOSCAN_REGISTER_OPERATOR(BayerDemosaicOp, "holoscan.vision.BayerDemosaicOp");
HOLOSCAN_REGISTER_OPERATOR(ConvertToOp, "holoscan.vision.ConvertToOp");
HOLOSCAN_REGISTER_OPERATOR(CopyMakeBorderOp, "holoscan.vision.CopyMakeBorderOp");
HOLOSCAN_REGISTER_OPERATOR(CustomCropOp, "holoscan.vision.CustomCropOp");
HOLOSCAN_REGISTER_OPERATOR(NonMaximumSuppressionOp, "holoscan.vision.NonMaximumSuppressionOp");
HOLOSCAN_REGISTER_OPERATOR(NormalizeOp, "holoscan.vision.NormalizeOp");
HOLOSCAN_REGISTER_OPERATOR(ReformatOp, "holoscan.vision.ReformatOp");
HOLOSCAN_REGISTER_OPERATOR(RemapOp, "holoscan.vision.RemapOp");
HOLOSCAN_REGISTER_OPERATOR(ResizeOp, "holoscan.vision.ResizeOp");

expected<void, Error> register_vision_operator_catalog() {
  return authoring::authoring_registry().registration_status();
}

}  // namespace holoscan::vision
