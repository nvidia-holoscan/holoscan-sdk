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

/// @brief Configure numeric conversion for \ref ConvertToOp.
struct ConvertToOpParams {
  VisionElementType output_type{VisionElementType::kF32};  ///< Scalar output element type.
  double alpha{1.0};                              ///< Multiplier applied before conversion.
  double beta{};                                  ///< Offset applied before conversion.
  RoundingMode rounding{RoundingMode::kNearest};  ///< Integral-output rounding mode.
};

HOLOSCAN_PARAM_SCHEMA(
    ConvertToOpParams, holoscan::param_schema("com.nvidia.holoscan.vision.ConvertToOpParams", 1),
    holoscan::field<&ConvertToOpParams::output_type>("output_type")
        .allowed_values({VisionElementType::kU8,
                         VisionElementType::kS8,
                         VisionElementType::kU16,
                         VisionElementType::kS16,
                         VisionElementType::kS32,
                         VisionElementType::kF32,
                         VisionElementType::kF64})
        .description("Scalar output element type."),
    holoscan::field<&ConvertToOpParams::alpha>("alpha")
        .validator<&detail::finite_double>("com.nvidia.holoscan.vision.finite_f64")
        .description("Multiplier applied before saturating conversion."),
    holoscan::field<&ConvertToOpParams::beta>("beta")
        .validator<&detail::finite_double>("com.nvidia.holoscan.vision.finite_f64")
        .description("Offset applied before saturating conversion."),
    holoscan::field<&ConvertToOpParams::rounding>("rounding")
        .allowed_values({RoundingMode::kNearest, RoundingMode::kTruncate})
        .description("Rounding mode used for integral outputs."));

/// @brief Convert interleaved HWC or NHWC CUDA tensor elements with CV-CUDA.
///
/// Each output is the saturating conversion of `alpha * input + beta`. The
/// output preserves input shape, layout, and sample metadata while replacing
/// its scalar element type. Inputs may have one through four channels. Accepted
/// input and output element types are U8, S8, U16, S16, S32, F32, and F64.
/// Input channels and pixels must be tightly packed; row and batch pitch may
/// include padding. Output storage is contiguous.
class HOLOSCAN_VISION_EXPORT ConvertToOp final : public Operator<ConvertToOpParams> {
 public:
  /// @brief Construct an operator with fixed output-pool capacity.
  /// @param capacity Maximum admitted image width, image height, and batch size.
  explicit ConvertToOp(VisionImageCapacity capacity);

  /// @brief Destroy the operator and its retained CV-CUDA resource after runtime quiescence.
  ~ConvertToOp() override;

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

  /// @brief Convert one image tensor and publish a plan-owned output tensor.
  /// @param context Callback-lifetime runtime services and pinned parameters.
  /// @return Success after asynchronous submission and publication, or a typed error.
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<Tensor> input{"input"};     ///< Input HWC or NHWC CUDA tensor.
  Output<Tensor> output{"output"};  ///< Converted HWC or NHWC CUDA tensor.

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  VisionImageCapacity capacity_;
};

static_assert(param_schema_availability<ConvertToOpParams>() == ParamSchemaAvailability::kDeclared,
              "ConvertToOpParams schema must be declared in its owning namespace");

}  // namespace holoscan::vision
