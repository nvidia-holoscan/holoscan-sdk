/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoinfer/holoinfer.hpp>
#include <holoscan/operators/inference/inference.hpp>
#include <holoscan/utils/holoinfer_utils.hpp>

int inference_header_smoke() {
  const auto dtype = holoscan::utils::holoinfer_dtype_to_dlpack(
      holoscan::inference::holoinfer_datatype::h_Float32);
  const bool valid_dtype = dtype.code == kDLFloat && dtype.bits == 32U;
  return sizeof(holoscan::ops::InferenceOp) > 0U && valid_dtype ? 0 : 1;
}
