// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
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

/// @brief Configure coordinate mapping for \ref RemapOp.
struct RemapOpParams {
  std::int32_t width{1};   ///< Positive output width in pixels.
  std::int32_t height{1};  ///< Positive output height in pixels.
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
    RemapOpParams, holoscan::param_schema("com.nvidia.holoscan.vision.RemapOpParams", 1),
    holoscan::field<&RemapOpParams::width>("width")
        .range(1, INT32_MAX)
        .description("Output image width in pixels."),
    holoscan::field<&RemapOpParams::height>("height")
        .range(1, INT32_MAX)
        .description("Output image height in pixels."),
    holoscan::field<&RemapOpParams::input_interpolation>("input_interpolation")
        .allowed_values(
            {InterpolationMode::kNearest, InterpolationMode::kLinear, InterpolationMode::kCubic})
        .description("Interpolation used to sample the input image."),
    holoscan::field<&RemapOpParams::map_interpolation>("map_interpolation")
        .allowed_values(
            {InterpolationMode::kNearest, InterpolationMode::kLinear, InterpolationMode::kCubic})
        .description("Interpolation used to sample the coordinate map."),
    holoscan::field<&RemapOpParams::map_value_mode>("map_value_mode")
        .allowed_values({RemapMapValueMode::kAbsolute,
                         RemapMapValueMode::kAbsoluteNormalized,
                         RemapMapValueMode::kRelativeNormalized})
        .description("Coordinate-map value encoding."),
    holoscan::field<&RemapOpParams::align_corners>("align_corners")
        .description("Align normalized coordinates to corner pixel centers."),
    holoscan::field<&RemapOpParams::border_mode>("border_mode")
        .allowed_values({BorderMode::kConstant,
                         BorderMode::kReplicate,
                         BorderMode::kReflect,
                         BorderMode::kWrap,
                         BorderMode::kReflect101})
        .description("Input-image border extrapolation mode."),
    holoscan::field<&RemapOpParams::border_value>("border_value")
        .max_elements(4U)
        .validator<&detail::nonempty_finite_float_vector>(
            "com.nvidia.holoscan.vision.nonempty_finite_f32_vector")
        .description("Constant fill value: one value or one value per image channel."),
    holoscan::field<&RemapOpParams::map_synchronization>("map_synchronization")
        .allowed_values({RemapMapSynchronizationMode::kLatest, RemapMapSynchronizationMode::kExact})
        .description("Image-to-map capture-time synchronization policy."),
    holoscan::field<&RemapOpParams::map_max_age>("map_max_age")
        .range(std::chrono::nanoseconds{1}, std::chrono::nanoseconds::max())
        .description("Maximum capture-time age of a map selected or reused for an image."));

/// @brief Remap interleaved HWC or NHWC CUDA tensors with a coordinate tensor.
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
/// \ref RemapMapSynchronizationMode::kExact to require image and map tensors
/// with identical capture times instead.
class HOLOSCAN_VISION_EXPORT RemapOp final : public Operator<RemapOpParams> {
 public:
  /// @brief Construct an operator with fixed output-pool capacity.
  /// @param capacity Maximum remapped output width, output height, and admitted batch size.
  explicit RemapOp(VisionImageCapacity capacity);

  /// @brief Destroy the operator and its retained CV-CUDA resource after runtime quiescence.
  ~RemapOp() override;

  /// @brief Declare tensor ports and bounded output allocation.
  /// @param spec Fresh operator declaration surface.
  void setup(OperatorSpec& spec) override;

  /// @brief Trigger on each image with a recent map, or on an exactly aligned pair.
  /// @return The operator's cold temporal contract.
  [[nodiscard]] Contract contract() const override;

  /// @brief Create the CV-CUDA resource for a running session.
  void start() override;

  /// @brief End a session while retaining the CV-CUDA resource through runtime quiescence.
  void stop() override;

  /// @brief Remap one image and publish a plan-owned output tensor.
  /// @param context Callback-lifetime runtime services and pinned parameters.
  /// @return Success after asynchronous submission and publication, or a typed error.
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<Tensor> input{"input"};     ///< Input HWC or NHWC CUDA image tensor.
  Input<Tensor> map{"map"};         ///< HWC or NHWC F32 two-channel coordinate tensor.
  Output<Tensor> output{"output"};  ///< Remapped HWC or NHWC CUDA image tensor.

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  VisionImageCapacity capacity_;
};

static_assert(param_schema_availability<RemapOpParams>() == ParamSchemaAvailability::kDeclared,
              "RemapOpParams schema must be declared in its owning namespace");

}  // namespace holoscan::vision
