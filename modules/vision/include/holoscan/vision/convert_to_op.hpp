// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

#include <holoscan/vision/holoscan_vision_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
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
    ConvertToOpParams, holoscan::param_schema("holoscan.vision.ConvertToOpParams", 2),
    holoscan::field<&ConvertToOpParams::output_type>("output_type")
        .enum_values({holoscan::param_enum_value("uint8", VisionElementType::kU8),
                      holoscan::param_enum_value("int8", VisionElementType::kS8),
                      holoscan::param_enum_value("uint16", VisionElementType::kU16),
                      holoscan::param_enum_value("int16", VisionElementType::kS16),
                      holoscan::param_enum_value("int32", VisionElementType::kS32),
                      holoscan::param_enum_value("float32", VisionElementType::kF32),
                      holoscan::param_enum_value("float64", VisionElementType::kF64)})
        .description("Scalar output element type."),
    holoscan::field<&ConvertToOpParams::alpha>("alpha")
        .validator<&detail::finite_double>("holoscan.vision.validator.finite-f64.v1")
        .description("Multiplier applied before saturating conversion."),
    holoscan::field<&ConvertToOpParams::beta>("beta")
        .validator<&detail::finite_double>("holoscan.vision.validator.finite-f64.v1")
        .description("Offset applied before saturating conversion."),
    holoscan::field<&ConvertToOpParams::rounding>("rounding")
        .enum_values({holoscan::param_enum_value("nearest", RoundingMode::kNearest),
                      holoscan::param_enum_value("truncate", RoundingMode::kTruncate)})
        .description("Rounding mode used for integral outputs."));

/// @brief Convert interleaved HWC or NHWC CUDA tensor elements with CV-CUDA.
///
/// Each output is the saturating conversion of `alpha * input + beta`. The
/// output preserves input shape, layout, and sample metadata while replacing
/// its scalar element type. Inputs may have one through four channels. Accepted
/// input and output element types are U8, S8, U16, S16, S32, F32, and F64.
/// Input channels and pixels must be tightly packed; row and batch pitch may
/// include padding. Output storage is contiguous.
///
/// Ports carry schema::ImageT with a scalar CUDA tensor in its data field.
/// Image dimensions, encoding, and supplied planes must match tensor geometry,
/// dtype, channels, layout, and strides. Compatible CUSTOM encodings are
/// supported; packed and chroma-subsampled encodings are rejected.
/// Output planes are rebuilt from actual tensor strides; for a batch they
/// describe the first image, repeated by the tensor batch stride. Image header,
/// exposure, and gain are preserved. Publication attaches the committed output
/// loan through emit_tensor() without copying pixel storage.
///
/// Image geometry and ROI offsets are preserved. Dtype changes select a
/// corresponding named image encoding when available, or CUSTOM otherwise.
/// Unscaled lossless casts retain source significant_bits, expanding zero to
/// the input container width when the dtype changes. Lossy casts or nonidentity
/// alpha/beta set significant_bits to schema::kUnknownImageSignificantBits.
/// A dtype change or nonidentity alpha/beta clears color_space: numeric conversion
/// does not establish a new code range or valid-bit alignment. An identity
/// conversion preserves these fields.
class HOLOSCAN_VISION_EXPORT ConvertToOp final : public Operator<ConvertToOpParams> {
 public:
  /// @brief Construct an operator whose output capacity follows its input and output dtype.
  ConvertToOp();

  /// @brief Destroy the operator and its retained CV-CUDA resource after runtime quiescence.
  ~ConvertToOp() override;

  /// @brief Declare image ports and bounded output allocation.
  /// @param spec Fresh operator declaration surface.
  void setup(OperatorSpec& spec) override;

  /// @brief Trigger once for each selected input image.
  /// @return The operator's cold temporal contract.
  [[nodiscard]] Contract contract() const override;

  /// @brief Create the CV-CUDA resource for a running session.
  void start() override;

  /// @brief End a session while retaining the CV-CUDA resource through runtime quiescence.
  void stop() override;

  /// @brief Convert one image tensor and publish a plan-owned output image.
  /// @param context Callback-lifetime runtime services and pinned parameters.
  /// @return Success after asynchronous submission and publication, or a typed error.
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<schema::ImageT> input{"input"};     ///< Image carrying an HWC or NHWC CUDA tensor.
  Output<schema::ImageT> output{"output"};  ///< Converted image with an HWC or NHWC CUDA tensor.

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

static_assert(param_schema_availability<ConvertToOpParams>() == ParamSchemaAvailability::kDeclared,
              "ConvertToOpParams schema must be declared in its owning namespace");

}  // namespace holoscan::vision
