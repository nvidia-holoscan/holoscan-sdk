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

/// @brief Configure the crop rectangle of \ref CustomCropOp.
struct CustomCropOpParams {
  std::int32_t x{};        ///< Nonnegative left coordinate in pixels.
  std::int32_t y{};        ///< Nonnegative top coordinate in pixels.
  std::int32_t width{1};   ///< Positive crop width in pixels.
  std::int32_t height{1};  ///< Positive crop height in pixels.
};

HOLOSCAN_PARAM_SCHEMA(CustomCropOpParams,
                      holoscan::param_schema("com.nvidia.holoscan.vision.CustomCropOpParams", 1),
                      holoscan::field<&CustomCropOpParams::x>("x")
                          .range(0, INT32_MAX)
                          .description("Left crop coordinate in pixels."),
                      holoscan::field<&CustomCropOpParams::y>("y")
                          .range(0, INT32_MAX)
                          .description("Top crop coordinate in pixels."),
                      holoscan::field<&CustomCropOpParams::width>("width")
                          .range(1, INT32_MAX)
                          .description("Crop width in pixels."),
                      holoscan::field<&CustomCropOpParams::height>("height")
                          .range(1, INT32_MAX)
                          .description("Crop height in pixels."));

/// @brief Crop interleaved HWC or NHWC CUDA tensors with CV-CUDA.
///
/// The operator wraps input and output storage as non-owning NVCV tensors and
/// submits work on the runtime CUDA stream. It performs no staging copy or host
/// synchronization. The output preserves input rank, batch size, channel count,
/// dtype, and sample metadata while replacing the spatial dimensions. The crop
/// rectangle must be fully contained in the input. Inputs may have one through
/// four channels and use U8, S8, U16, S16, S32, F32, or F64 elements. Input
/// channels and pixels must be tightly packed; row and batch pitch may include
/// padding. Output storage is contiguous.
class HOLOSCAN_VISION_EXPORT CustomCropOp final : public Operator<CustomCropOpParams> {
 public:
  /// @brief Construct an operator with fixed output-pool capacity.
  /// @param capacity Maximum crop width, crop height, and admitted batch size.
  ///
  /// The capacity is graph structure and cannot be changed by a parameter
  /// override. Each field must be positive and every bound
  /// \ref CustomCropOpParams geometry must fit within it.
  explicit CustomCropOp(VisionImageCapacity capacity);

  /// @brief Destroy the operator and its retained CV-CUDA resource after runtime quiescence.
  ~CustomCropOp() override;

  /// @brief Declare tensor ports and bounded output allocation.
  /// @param spec Fresh operator declaration surface.
  void setup(OperatorSpec& spec) override;

  /// @brief Trigger once for each selected input tensor.
  /// @return The operator's cold temporal contract.
  [[nodiscard]] Contract contract() const override;

  /// @brief Create the CV-CUDA crop resource for a running session.
  void start() override;

  /// @brief End a session while retaining the CV-CUDA resource through runtime quiescence.
  void stop() override;

  /// @brief Crop one input tensor and publish a plan-owned output tensor.
  /// @param context Callback-lifetime runtime services and pinned parameters.
  /// @return Success after asynchronous submission and publication, or a typed error.
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<Tensor> input{"input"};     ///< Input HWC or NHWC CUDA tensor.
  Output<Tensor> output{"output"};  ///< Cropped HWC or NHWC CUDA tensor.

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  VisionImageCapacity capacity_;
};

static_assert(param_schema_availability<CustomCropOpParams>() == ParamSchemaAvailability::kDeclared,
              "CustomCropOpParams schema must be declared in its owning namespace");

}  // namespace holoscan::vision
