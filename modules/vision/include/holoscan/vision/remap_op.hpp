// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

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

/// @brief Configure coordinate mapping for \ref RemapOp.
struct RemapOpParams {
  std::int32_t width{1};             ///< Positive output width in pixels.
  std::int32_t height{1};            ///< Positive output height in pixels.
  std::uint32_t max_batch_size{1U};  ///< Maximum admitted image batch size.
  InterpolationMode input_interpolation{InterpolationMode::kLinear};  ///< Image sampling mode.
  InterpolationMode map_interpolation{InterpolationMode::kLinear};    ///< Map sampling mode.
  RemapMapValueMode map_value_mode{RemapMapValueMode::kAbsolute};     ///< Map encoding.
  bool align_corners{};  ///< Whether normalized coordinates align corner pixel centers.
  BorderMode border_mode{BorderMode::kConstant};  ///< Image border extrapolation mode.
  std::vector<float> border_value{0.F};           ///< One or one-per-channel fill values.
  RemapMapSynchronizationMode map_synchronization{
      RemapMapSynchronizationMode::kLatest};  ///< Image-to-map synchronization policy.
  std::chrono::nanoseconds map_max_age{
      std::chrono::nanoseconds::max()};  ///< Maximum age of a selected or reused map.
};

HOLOSCAN_PARAM_SCHEMA(
    RemapOpParams, holoscan::param_schema("holoscan.vision.RemapOpParams", 2),
    holoscan::field<&RemapOpParams::width>("width")
        .recompile_required()
        .range(1, INT32_MAX)
        .description("Output image width in pixels."),
    holoscan::field<&RemapOpParams::height>("height")
        .recompile_required()
        .range(1, INT32_MAX)
        .description("Output image height in pixels."),
    holoscan::field<&RemapOpParams::max_batch_size>("max_batch_size")
        .recompile_required()
        .required_on_load()
        .range(1U, kMaximumBatchSize)
        .description("Maximum image batch size admitted by the operator."),
    holoscan::field<&RemapOpParams::input_interpolation>("input_interpolation")
        .enum_values({holoscan::param_enum_value("nearest", InterpolationMode::kNearest),
                      holoscan::param_enum_value("linear", InterpolationMode::kLinear),
                      holoscan::param_enum_value("cubic", InterpolationMode::kCubic)})
        .description("Interpolation used to sample the input image."),
    holoscan::field<&RemapOpParams::map_interpolation>("map_interpolation")
        .enum_values({holoscan::param_enum_value("nearest", InterpolationMode::kNearest),
                      holoscan::param_enum_value("linear", InterpolationMode::kLinear),
                      holoscan::param_enum_value("cubic", InterpolationMode::kCubic)})
        .description("Interpolation used to sample the coordinate map."),
    holoscan::field<&RemapOpParams::map_value_mode>("map_value_mode")
        .enum_values({holoscan::param_enum_value("absolute", RemapMapValueMode::kAbsolute),
                      holoscan::param_enum_value("absolute_normalized",
                                                 RemapMapValueMode::kAbsoluteNormalized),
                      holoscan::param_enum_value("relative_normalized",
                                                 RemapMapValueMode::kRelativeNormalized)})
        .description("Coordinate-map value encoding."),
    holoscan::field<&RemapOpParams::align_corners>("align_corners")
        .description("Align normalized coordinates to corner pixel centers."),
    holoscan::field<&RemapOpParams::border_mode>("border_mode")
        .enum_values({holoscan::param_enum_value("constant", BorderMode::kConstant),
                      holoscan::param_enum_value("replicate", BorderMode::kReplicate),
                      holoscan::param_enum_value("reflect", BorderMode::kReflect),
                      holoscan::param_enum_value("wrap", BorderMode::kWrap),
                      holoscan::param_enum_value("reflect_101", BorderMode::kReflect101)})
        .description("Input-image border extrapolation mode."),
    holoscan::field<&RemapOpParams::border_value>("border_value")
        .max_elements(4U)
        .validator<&detail::nonempty_finite_float_vector>(
            "holoscan.vision.validator.nonempty-finite-f32-vector.v1")
        .description("Constant fill value: one value or one value per image channel."),
    holoscan::field<&RemapOpParams::map_synchronization>("map_synchronization")
        .enum_values({holoscan::param_enum_value("latest", RemapMapSynchronizationMode::kLatest),
                      holoscan::param_enum_value("exact", RemapMapSynchronizationMode::kExact)})
        .description("Image-to-map capture-time synchronization policy."),
    holoscan::field<&RemapOpParams::map_max_age>("map_max_age")
        .range(std::chrono::nanoseconds{1}, std::chrono::nanoseconds::max())
        .description("Maximum capture-time age of a map selected or reused for an image."));

/// @brief Remap interleaved HWC or NHWC CUDA images with a coordinate tensor.
///
/// The map must be an HWC or NHWC F32 tensor with two interleaved coordinate
/// channels. Its batch size must be one or match the image batch. Its spatial
/// dimensions may differ from both input and output because CV-CUDA samples it
/// over the destination grid. The output preserves image rank, batch size,
/// channel count, dtype, and sample metadata.
/// Images may be U8 with one, three, or four channels, or F32 with one channel.
/// Input channels and pixels must be tightly packed; row and batch pitch may
/// include padding. Output storage is contiguous.
///
/// By default, every image triggers an activation using the newest causal map
/// no older than \ref RemapOpParams::map_max_age. A previously selected map is
/// reused for later images within the same age bound. Select
/// \ref RemapMapSynchronizationMode::kExact to require image and map samples
/// with identical capture times instead.
/// In either mode, the map is configuration: output capture time, frame identity,
/// and trace identity come from the image. Output flags are the union of image
/// and selected map flags, including when a retained map is reused.
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
/// Both ROI offsets reset to zero and Bayer encoding becomes CUSTOM even for
/// an identity coordinate map. Color space is preserved. significant_bits is
/// preserved only for nearest-neighbor image sampling with zero constant
/// borders or sample-copy border modes; otherwise it is set to
/// schema::kUnknownImageSignificantBits.
class HOLOSCAN_VISION_EXPORT RemapOp final : public Operator<RemapOpParams> {
 public:
  /// @brief Construct an operator whose output capacity is derived from its Parameters.
  RemapOp();

  /// @brief Destroy the operator and its retained CV-CUDA resource after runtime quiescence.
  ~RemapOp() override;

  /// @brief Declare image ports and bounded output allocation.
  /// @param spec Fresh operator declaration surface.
  void setup(OperatorSpec& spec) override;

  /// @brief Trigger on each image with a recent map, or on an exactly aligned pair.
  /// @return The operator's cold temporal contract.
  [[nodiscard]] Contract contract() const override;

  /// @brief Create the CV-CUDA resource for a running session.
  void start() override;

  /// @brief End a session while retaining the CV-CUDA resource through runtime quiescence.
  void stop() override;

  /// @brief Remap one image and publish a plan-owned output image.
  /// @param context Callback-lifetime runtime services and pinned parameters.
  /// @return Success after asynchronous submission and publication, or a typed error.
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<schema::ImageT> input{"input"};     ///< Image carrying an HWC or NHWC CUDA tensor.
  Input<Tensor> map{"map"};                 ///< HWC or NHWC F32 two-channel coordinate tensor.
  Output<schema::ImageT> output{"output"};  ///< Remapped image with an HWC or NHWC CUDA tensor.

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

static_assert(param_schema_availability<RemapOpParams>() == ParamSchemaAvailability::kDeclared,
              "RemapOpParams schema must be declared in its owning namespace");

}  // namespace holoscan::vision
