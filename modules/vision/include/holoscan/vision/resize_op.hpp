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

/// @brief Configure the output geometry and interpolation of \ref ResizeOp.
struct ResizeOpParams {
  std::int32_t width{1};             ///< Positive output width in pixels.
  std::int32_t height{1};            ///< Positive output height in pixels.
  std::uint32_t max_batch_size{1U};  ///< Maximum admitted image batch size.
  InterpolationMode interpolation{InterpolationMode::kLinear};  ///< Resize algorithm.
};

HOLOSCAN_PARAM_SCHEMA(
    ResizeOpParams, holoscan::param_schema("com.nvidia.holoscan.vision.ResizeOpParams", 2),
    holoscan::field<&ResizeOpParams::width>("width")
        .recompile_required()
        .range(1, INT32_MAX)
        .description("Output image width in pixels."),
    holoscan::field<&ResizeOpParams::height>("height")
        .recompile_required()
        .range(1, INT32_MAX)
        .description("Output image height in pixels."),
    holoscan::field<&ResizeOpParams::max_batch_size>("max_batch_size")
        .recompile_required()
        .required_on_load()
        .range(1U, kMaximumBatchSize)
        .description("Maximum image batch size admitted by the operator."),
    holoscan::field<&ResizeOpParams::interpolation>("interpolation")
        .enum_values({holoscan::param_enum_value("nearest", InterpolationMode::kNearest),
                      holoscan::param_enum_value("linear", InterpolationMode::kLinear),
                      holoscan::param_enum_value("cubic", InterpolationMode::kCubic),
                      holoscan::param_enum_value("area", InterpolationMode::kArea)})
        .description("Interpolation algorithm."));

/// @brief Resize interleaved HWC or NHWC CUDA images with CV-CUDA.
///
/// The operator wraps input and output storage as non-owning NVCV tensors and
/// submits work on the runtime CUDA stream. It performs no staging copy or host
/// synchronization. The output preserves input rank, batch size, channel count,
/// dtype, and sample metadata while replacing the spatial dimensions. Inputs
/// may have one, three, or four channels and use U8, U16, S16, or F32 elements.
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
/// Changed dimensions reset both ROI offsets and replace Bayer encoding with
/// CUSTOM. Nearest-neighbor preserves significant_bits; other interpolation
/// sets it to schema::kUnknownImageSignificantBits. Unchanged dimensions preserve
/// these metadata claims. Color space is preserved.
class HOLOSCAN_VISION_EXPORT ResizeOp final : public Operator<ResizeOpParams> {
 public:
  /// @brief Construct an operator whose output capacity is derived from its Parameters.
  ResizeOp();

  /// @brief Destroy the operator and its retained CV-CUDA resource after runtime quiescence.
  ~ResizeOp() override;

  /// @brief Declare image ports and bounded output allocation.
  /// @param spec Fresh operator declaration surface.
  void setup(OperatorSpec& spec) override;

  /// @brief Trigger once for each selected input image.
  /// @return The operator's cold temporal contract.
  [[nodiscard]] Contract contract() const override;

  /// @brief Create the CV-CUDA resize resource for a running session.
  void start() override;

  /// @brief End a session while retaining the CV-CUDA resource through runtime quiescence.
  void stop() override;

  /// @brief Resize one input image and publish a plan-owned output image.
  /// @param context Callback-lifetime runtime services and pinned parameters.
  /// @return Success after asynchronous submission and publication, or a typed error.
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<schema::ImageT> input{"input"};     ///< Image carrying an HWC or NHWC CUDA tensor.
  Output<schema::ImageT> output{"output"};  ///< Resized image with an HWC or NHWC CUDA tensor.

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

static_assert(param_schema_availability<ResizeOpParams>() == ParamSchemaAvailability::kDeclared,
              "ResizeOpParams schema must be declared in its owning namespace");

}  // namespace holoscan::vision
