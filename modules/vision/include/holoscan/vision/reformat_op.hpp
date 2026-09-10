// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

#include <holoscan/vision/holoscan_vision_export.h>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/vision/vision_types.hpp>

namespace holoscan::vision {

/// @brief Configure layout conversion for \ref ReformatOp.
struct ReformatOpParams {
  ReformatMode mode{ReformatMode::kInterleavedToPlanar};  ///< Input-to-output layout conversion.
};

HOLOSCAN_PARAM_SCHEMA(ReformatOpParams,
                      holoscan::param_schema("com.nvidia.holoscan.vision.ReformatOpParams", 1),
                      holoscan::field<&ReformatOpParams::mode>("mode")
                          .allowed_values({ReformatMode::kInterleavedToPlanar,
                                           ReformatMode::kPlanarToInterleaved})
                          .description("Input-to-output image layout conversion."));

/// @brief Convert CUDA image tensors between interleaved and planar layouts.
///
/// Interleaved HWC/NHWC input becomes CHW/NCHW output, or the inverse when
/// configured for planar-to-interleaved conversion. The operator preserves
/// image dimensions, dtype, and sample metadata while reordering storage.
/// Inputs may have one through four channels and use U8, S8, U16, S16, S32,
/// F32, or F64 elements. Interleaved input channels and pixels must be tightly
/// packed. Planar input pixels must be tightly packed. Row, channel, and batch
/// pitch may include padding where the layout defines those dimensions. Output
/// storage is contiguous.
class HOLOSCAN_VISION_EXPORT ReformatOp final : public Operator<ReformatOpParams> {
 public:
  /// @brief Construct an operator with fixed output-pool capacity.
  /// @param capacity Maximum admitted image width, image height, and batch size.
  explicit ReformatOp(VisionImageCapacity capacity);

  /// @brief Destroy the operator and its retained CV-CUDA resource after runtime quiescence.
  ~ReformatOp() override;

  /// @brief Declare tensor ports and bounded output allocation.
  /// @param spec Fresh operator declaration surface.
  void setup(OperatorSpec& spec) override;

  /// @brief Trigger once for each selected input tensor.
  /// @return The operator's cold temporal contract.
  [[nodiscard]] Contract contract() const override;

  /// @brief Create the CV-CUDA resource for a running session.
  void start() override;

  /// @brief End a session while retaining the CV-CUDA resource through runtime quiescence.
  void stop() override;

  /// @brief Reformat one image tensor and publish a plan-owned output tensor.
  /// @param context Callback-lifetime runtime services and pinned parameters.
  /// @return Success after asynchronous submission and publication, or a typed error.
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<Tensor> input{"input"};     ///< Input interleaved or planar CUDA image tensor.
  Output<Tensor> output{"output"};  ///< Output tensor in the configured target layout.

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  VisionImageCapacity capacity_;
};

static_assert(param_schema_availability<ReformatOpParams>() == ParamSchemaAvailability::kDeclared,
              "ReformatOpParams schema must be declared in its owning namespace");

}  // namespace holoscan::vision
