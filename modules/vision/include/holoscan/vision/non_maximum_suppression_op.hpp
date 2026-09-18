// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>
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

/// @brief Configure score and overlap thresholds for \ref NonMaximumSuppressionOp.
struct NonMaximumSuppressionOpParams {
  std::int32_t max_boxes{1};         ///< Maximum boxes admitted per batch item.
  std::uint32_t max_batch_size{1U};  ///< Maximum admitted detection batch size.
  float score_threshold{};           ///< Inclusive minimum score in the range [0, 1].
  float iou_threshold{0.5F};         ///< Maximum permitted overlap in the range (0, 1].
};

HOLOSCAN_PARAM_SCHEMA(
    NonMaximumSuppressionOpParams,
    holoscan::param_schema("com.nvidia.holoscan.vision.NonMaximumSuppressionOpParams", 2),
    holoscan::field<&NonMaximumSuppressionOpParams::max_boxes>("max_boxes")
        .recompile_required()
        .required_on_load()
        .range(1, INT32_MAX)
        .description("Maximum boxes admitted per batch item."),
    holoscan::field<&NonMaximumSuppressionOpParams::max_batch_size>("max_batch_size")
        .recompile_required()
        .required_on_load()
        .range(1U, kMaximumBatchSize)
        .description("Maximum detection batch size admitted by the operator."),
    holoscan::field<&NonMaximumSuppressionOpParams::score_threshold>("score_threshold")
        .range(0.F, 1.F)
        .description("Inclusive minimum score for retaining a bounding box."),
    holoscan::field<&NonMaximumSuppressionOpParams::iou_threshold>("iou_threshold")
        .range(std::numeric_limits<float>::denorm_min(), 1.F)
        .description("Maximum permitted Intersection-over-Union overlap."));

/// @brief Select non-overlapping bounding boxes with CV-CUDA.
///
/// The boxes input is an S16 tensor shaped `[N, W, 4]`, with each box encoded as
/// `(x, y, width, height)`. Scores are an F32 `[N, W]` tensor. Both inputs may
/// use row pitch, but coordinates and scores must be tightly packed within each
/// row. The boxes base address and batch pitch must be 8-byte aligned; the
/// scores base address and batch pitch must be 4-byte aligned. The output is a
/// contiguous U8 `[N, W]` selection mask and preserves boxes metadata. The
/// boxes, scores, and output tensors use the same CUDA device.
class HOLOSCAN_VISION_EXPORT NonMaximumSuppressionOp final
    : public Operator<NonMaximumSuppressionOpParams> {
 public:
  /// @brief Construct an operator whose output capacity is derived from its Parameters.
  NonMaximumSuppressionOp();

  /// @brief Destroy the operator and its retained CV-CUDA resource after runtime quiescence.
  ~NonMaximumSuppressionOp() override;

  /// @brief Declare tensor ports and bounded output allocation.
  /// @param spec Fresh operator declaration surface.
  void setup(OperatorSpec& spec) override;

  /// @brief Trigger on exactly capture-time-aligned boxes and scores tensors.
  /// @return The operator's cold temporal contract.
  [[nodiscard]] Contract contract() const override;

  /// @brief Create the CV-CUDA resource for a running session.
  void start() override;

  /// @brief End a session while retaining the CV-CUDA resource through runtime quiescence.
  void stop() override;

  /// @brief Suppress overlapping boxes and publish a plan-owned mask tensor.
  /// @param context Callback-lifetime runtime services and pinned parameters.
  /// @return Success after asynchronous submission and publication, or a typed error.
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;

  Input<Tensor> boxes{"boxes"};    ///< S16 `[N, W, 4]` CUDA bounding boxes.
  Input<Tensor> scores{"scores"};  ///< F32 `[N, W]` CUDA confidence scores.
  Output<Tensor> mask{"mask"};     ///< U8 `[N, W]` CUDA selection mask.

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

static_assert(param_schema_availability<NonMaximumSuppressionOpParams>() ==
                  ParamSchemaAvailability::kDeclared,
              "NonMaximumSuppressionOpParams schema must be declared in its owning namespace");

}  // namespace holoscan::vision
