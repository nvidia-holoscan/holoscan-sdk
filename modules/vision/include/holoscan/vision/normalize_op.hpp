// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

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

/// @brief Configure channel normalization for \ref NormalizeOp.
struct NormalizeOpParams {
  std::vector<float> base{0.F};   ///< One or one-per-channel value subtracted from input.
  std::vector<float> scale{1.F};  ///< One or one-per-channel multiplier or standard deviation.
  float global_scale{1.F};        ///< Additional multiplier applied after channel scale.
  float shift{};                  ///< Bias added to the normalized value.
  float epsilon{};                ///< Nonnegative variance regularizer for standard deviation.
  bool scale_is_stddev{};         ///< Interpret scale as standard deviation when true.
};

HOLOSCAN_PARAM_SCHEMA(
    NormalizeOpParams, holoscan::param_schema("com.nvidia.holoscan.vision.NormalizeOpParams", 1),
    holoscan::field<&NormalizeOpParams::base>("base")
        .max_elements(4U)
        .validator<&detail::nonempty_finite_float_vector>(
            "com.nvidia.holoscan.vision.nonempty_finite_f32_vector")
        .description("Subtracted channel values: one value or one value per image channel."),
    holoscan::field<&NormalizeOpParams::scale>("scale")
        .max_elements(4U)
        .validator<&detail::nonempty_finite_float_vector>(
            "com.nvidia.holoscan.vision.nonempty_finite_f32_vector")
        .description("Channel scales or standard deviations: one or one per image channel."),
    holoscan::field<&NormalizeOpParams::global_scale>("global_scale")
        .validator<&detail::finite_float>("com.nvidia.holoscan.vision.finite_f32")
        .description("Additional scale applied to every output value."),
    holoscan::field<&NormalizeOpParams::shift>("shift")
        .validator<&detail::finite_float>("com.nvidia.holoscan.vision.finite_f32")
        .description("Bias added to every output value."),
    holoscan::field<&NormalizeOpParams::epsilon>("epsilon")
        .range(0.F, std::numeric_limits<float>::max())
        .description("Nonnegative variance regularizer used with standard-deviation scale."),
    holoscan::field<&NormalizeOpParams::scale_is_stddev>("scale_is_stddev")
        .description("Interpret scale values as standard deviations."));

/// @brief Normalize interleaved HWC or NHWC CUDA tensors with CV-CUDA.
///
/// With ordinary scale values, each output is `(input - base) * scale *
/// global_scale + shift`. With standard-deviation scale values, CV-CUDA uses
/// `1 / sqrt(scale * scale + epsilon)` as the channel multiplier. A zero
/// standard-deviation scale therefore requires a positive epsilon; otherwise,
/// the operator reports \ref ErrorCode::kInvalidArgument.
/// The output preserves input shape, dtype, and sample metadata. Inputs may have one,
/// three, or four channels and use U8, S8, U16, S16, S32, or F32 elements.
/// To normalize U8 input into F32 output, place \ref ConvertToOp configured for
/// \ref VisionElementType::kF32 before this operator.
/// Input channels and pixels must be tightly packed; row and batch pitch may
/// include padding. Output storage is contiguous.
class HOLOSCAN_VISION_EXPORT NormalizeOp final : public Operator<NormalizeOpParams> {
 public:
  /// @brief Construct an operator with fixed output-pool capacity.
  /// @param capacity Maximum admitted image width, image height, and batch size.
  explicit NormalizeOp(VisionImageCapacity capacity);

  /// @brief Destroy the operator and its retained CV-CUDA resource after runtime quiescence.
  ~NormalizeOp() override;

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

  /// @brief Normalize one image tensor and publish a plan-owned output tensor.
  /// @param context Callback-lifetime runtime services and pinned parameters.
  /// @return Success after asynchronous submission and publication, or a typed error.
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<Tensor> input{"input"};     ///< Input HWC or NHWC CUDA tensor.
  Output<Tensor> output{"output"};  ///< Normalized HWC or NHWC CUDA tensor.

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  VisionImageCapacity capacity_;
};

static_assert(param_schema_availability<NormalizeOpParams>() == ParamSchemaAvailability::kDeclared,
              "NormalizeOpParams schema must be declared in its owning namespace");

}  // namespace holoscan::vision
