// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

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

/// @brief Configure image borders for \ref CopyMakeBorderOp.
struct CopyMakeBorderOpParams {
  std::int32_t max_width{1};                      ///< Maximum bordered output width in pixels.
  std::int32_t max_height{1};                     ///< Maximum bordered output height in pixels.
  std::uint32_t max_batch_size{1U};               ///< Maximum admitted image batch size.
  std::int32_t top{};                             ///< Nonnegative top border width in pixels.
  std::int32_t bottom{};                          ///< Nonnegative bottom border width in pixels.
  std::int32_t left{};                            ///< Nonnegative left border width in pixels.
  std::int32_t right{};                           ///< Nonnegative right border width in pixels.
  BorderMode border_mode{BorderMode::kConstant};  ///< Border extrapolation mode.
  std::vector<float> border_value{0.F};           ///< One or one-per-channel fill values.
};

HOLOSCAN_PARAM_SCHEMA(
    CopyMakeBorderOpParams,
    holoscan::param_schema("com.nvidia.holoscan.vision.CopyMakeBorderOpParams", 2),
    holoscan::field<&CopyMakeBorderOpParams::max_width>("max_width")
        .recompile_required()
        .required_on_load()
        .range(1, INT32_MAX)
        .description("Maximum bordered output width in pixels."),
    holoscan::field<&CopyMakeBorderOpParams::max_height>("max_height")
        .recompile_required()
        .required_on_load()
        .range(1, INT32_MAX)
        .description("Maximum bordered output height in pixels."),
    holoscan::field<&CopyMakeBorderOpParams::max_batch_size>("max_batch_size")
        .recompile_required()
        .required_on_load()
        .range(1U, kMaximumBatchSize)
        .description("Maximum image batch size admitted by the operator."),
    holoscan::field<&CopyMakeBorderOpParams::top>("top")
        .range(0, INT32_MAX)
        .description("Top border width in pixels."),
    holoscan::field<&CopyMakeBorderOpParams::bottom>("bottom")
        .range(0, INT32_MAX)
        .description("Bottom border width in pixels."),
    holoscan::field<&CopyMakeBorderOpParams::left>("left")
        .range(0, INT32_MAX)
        .description("Left border width in pixels."),
    holoscan::field<&CopyMakeBorderOpParams::right>("right")
        .range(0, INT32_MAX)
        .description("Right border width in pixels."),
    holoscan::field<&CopyMakeBorderOpParams::border_mode>("border_mode")
        .allowed_values({BorderMode::kConstant,
                         BorderMode::kReplicate,
                         BorderMode::kReflect,
                         BorderMode::kWrap,
                         BorderMode::kReflect101})
        .description("Border extrapolation mode."),
    holoscan::field<&CopyMakeBorderOpParams::border_value>("border_value")
        .max_elements(4U)
        .validator<&detail::nonempty_finite_float_vector>(
            "com.nvidia.holoscan.vision.nonempty_finite_f32_vector")
        .description("Constant fill value: one value or one value per image channel."));

/// @brief Add borders to interleaved HWC or NHWC CUDA tensors with CV-CUDA.
///
/// The output preserves input rank, batch size, channel count, dtype, and sample
/// metadata while increasing each spatial dimension by the configured borders.
/// U8 inputs may have one through four channels. U16, S16, and F32 inputs may
/// have one, three, or four channels. Input channels and pixels must be tightly
/// packed; row and batch pitch may include padding. Output storage is
/// contiguous.
class HOLOSCAN_VISION_EXPORT CopyMakeBorderOp final : public Operator<CopyMakeBorderOpParams> {
 public:
  /// @brief Construct an operator whose output capacity is derived from its Parameters.
  CopyMakeBorderOp();

  /// @brief Destroy the operator and its retained CV-CUDA resource after runtime quiescence.
  ~CopyMakeBorderOp() override;

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

  /// @brief Add borders and publish a plan-owned output tensor.
  /// @param context Callback-lifetime runtime services and pinned parameters.
  /// @return Success after asynchronous submission and publication, or a typed error.
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<Tensor> input{"input"};     ///< Input HWC or NHWC CUDA tensor.
  Output<Tensor> output{"output"};  ///< Bordered HWC or NHWC CUDA tensor.

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

static_assert(param_schema_availability<CopyMakeBorderOpParams>() ==
                  ParamSchemaAvailability::kDeclared,
              "CopyMakeBorderOpParams schema must be declared in its owning namespace");

}  // namespace holoscan::vision
