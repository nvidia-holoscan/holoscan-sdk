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

/// @brief Configure the output geometry and interpolation of \ref ResizeOp.
struct ResizeOpParams {
  std::int32_t width{1};             ///< Positive output width in pixels.
  std::int32_t height{1};            ///< Positive output height in pixels.
  std::uint32_t max_batch_size{1U};  ///< Maximum admitted image batch size.
  InterpolationMode interpolation{InterpolationMode::kLinear};  ///< Resize algorithm.
};

HOLOSCAN_PARAM_SCHEMA(ResizeOpParams,
                      holoscan::param_schema("com.nvidia.holoscan.vision.ResizeOpParams", 2),
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
                          .allowed_values({InterpolationMode::kNearest,
                                           InterpolationMode::kLinear,
                                           InterpolationMode::kCubic,
                                           InterpolationMode::kArea})
                          .description("Interpolation algorithm."));

/// @brief Resize interleaved HWC or NHWC CUDA tensors with CV-CUDA.
///
/// The operator wraps input and output storage as non-owning NVCV tensors and
/// submits work on the runtime CUDA stream. It performs no staging copy or host
/// synchronization. The output preserves input rank, batch size, channel count,
/// dtype, and sample metadata while replacing the spatial dimensions. Inputs
/// may have one, three, or four channels and use U8, U16, S16, or F32 elements.
/// Input channels and pixels must be tightly packed; row and batch pitch may
/// include padding. Output storage is contiguous.
class HOLOSCAN_VISION_EXPORT ResizeOp final : public Operator<ResizeOpParams> {
 public:
  /// @brief Construct an operator whose output capacity is derived from its Parameters.
  ResizeOp();

  /// @brief Destroy the operator and its retained CV-CUDA resource after runtime quiescence.
  ~ResizeOp() override;

  /// @brief Declare tensor ports and bounded output allocation.
  /// @param spec Fresh operator declaration surface.
  void setup(OperatorSpec& spec) override;

  /// @brief Trigger once for each selected input tensor.
  /// @return The operator's cold temporal contract.
  [[nodiscard]] Contract contract() const override;

  /// @brief Create the CV-CUDA resize resource for a running session.
  void start() override;

  /// @brief End a session while retaining the CV-CUDA resource through runtime quiescence.
  void stop() override;

  /// @brief Resize one input tensor and publish a plan-owned output tensor.
  /// @param context Callback-lifetime runtime services and pinned parameters.
  /// @return Success after asynchronous submission and publication, or a typed error.
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<Tensor> input{"input"};     ///< Input HWC or NHWC CUDA tensor.
  Output<Tensor> output{"output"};  ///< Resized HWC or NHWC CUDA tensor.

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

static_assert(param_schema_availability<ResizeOpParams>() == ParamSchemaAvailability::kDeclared,
              "ResizeOpParams schema must be declared in its owning namespace");

}  // namespace holoscan::vision
