// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include <cvcuda/OpNonMaximumSuppression.hpp>
#include <nvcv/Exception.hpp>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/vision/non_maximum_suppression_op.hpp>

#include "detail/cvcuda_tensor.hpp"

namespace holoscan::vision {
namespace {

constexpr DLDataType kS16{kDLInt, 16U, 1U};
constexpr DLDataType kF32{kDLFloat, 32U, 1U};
constexpr DLDataType kU8{kDLUInt, 8U, 1U};
constexpr std::int64_t kMaxCvCudaByteSpan = std::numeric_limits<std::int32_t>::max();
constexpr std::uint64_t kBoxBytesPerDetection = 4U * sizeof(std::int16_t);

[[nodiscard]] expected<std::uint64_t, Error> nms_output_byte_span(
    const NonMaximumSuppressionOpParams& params) noexcept {
  if (params.max_boxes <= 0 || params.max_batch_size == 0U ||
      params.max_batch_size > kMaximumBatchSize) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  const auto boxes = static_cast<std::uint64_t>(params.max_boxes);
  if (boxes > std::numeric_limits<std::uint64_t>::max() / params.max_batch_size) {
    return make_unexpected(Error{ErrorCode::kArithmeticOverflow});
  }
  const std::uint64_t mask_byte_span = boxes * params.max_batch_size;
  if (mask_byte_span > static_cast<std::uint64_t>(kMaxCvCudaByteSpan) / kBoxBytesPerDetection) {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }
  return mask_byte_span;
}

[[nodiscard]] auto failure(ErrorCode code, std::string message) {
  return make_unexpected(Error{code, std::move(message)});
}

[[nodiscard]] bool same_dtype(DLDataType lhs, DLDataType rhs) noexcept {
  return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}

[[nodiscard]] expected<void, Error> validate_common_tensor(const Tensor& tensor,
                                                           std::string_view name) {
  if (tensor.data() == nullptr) {
    return failure(ErrorCode::kInvalidArgument,
                   "NonMaximumSuppressionOp " + std::string(name) + " tensor is empty");
  }
  if (tensor.device().device_type != kDLCUDA || tensor.device().device_id < 0) {
    return failure(ErrorCode::kInvalidArgument,
                   "NonMaximumSuppressionOp requires CUDA device tensors");
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_pitched_rows(
    const void* data, std::span<const std::int64_t> shape,
    std::span<const std::int64_t> byte_strides, std::int64_t element_stride,
    std::size_t access_alignment, std::string_view name) {
  if (shape.size() != 2U || byte_strides.size() != 2U) {
    return failure(ErrorCode::kInvalidArgument,
                   "NonMaximumSuppressionOp " + std::string(name) + " tensor rank is invalid");
  }
  if (reinterpret_cast<std::uintptr_t>(data) % access_alignment != 0U ||
      byte_strides[0] % static_cast<std::int64_t>(access_alignment) != 0) {
    return failure(
        ErrorCode::kInvalidArgument,
        "NonMaximumSuppressionOp " + std::string(name) + " tensor access alignment is invalid");
  }
  if (byte_strides[1] != element_stride || byte_strides[0] <= 0 ||
      byte_strides[0] > kMaxCvCudaByteSpan) {
    return failure(
        ErrorCode::kInvalidArgument,
        "NonMaximumSuppressionOp " + std::string(name) + " tensor has invalid byte strides");
  }
  if (shape[1] > kMaxCvCudaByteSpan / element_stride ||
      byte_strides[0] < shape[1] * element_stride ||
      shape[0] > kMaxCvCudaByteSpan / byte_strides[0]) {
    return failure(ErrorCode::kInvalidArgument,
                   "NonMaximumSuppressionOp " + std::string(name) + " tensor byte span is invalid");
  }
  return {};
}

[[nodiscard]] expected<detail::FixedTensorByteStrides, Error> validate_boxes(
    const Tensor& tensor, std::int32_t max_boxes, std::uint32_t max_batch_size) {
  auto common = validate_common_tensor(tensor, "boxes");
  if (!common) {
    return make_unexpected(std::move(common).error());
  }
  const auto shape = tensor.shape_span();
  if (!same_dtype(tensor.dtype(), kS16) || shape.size() != 3U || shape[0] <= 0 || shape[1] <= 0 ||
      shape[2] != 4 || shape[0] > static_cast<std::int64_t>(max_batch_size) ||
      shape[1] > max_boxes) {
    return failure(ErrorCode::kInvalidArgument,
                   "NonMaximumSuppressionOp boxes must be S16 [N, W, 4] within capacity");
  }
  auto byte_strides = detail::fixed_tensor_byte_strides(tensor);
  if (!byte_strides) {
    return make_unexpected(std::move(byte_strides).error());
  }
  if (byte_strides->span().size() != 3U || byte_strides->span()[2] != 2) {
    return failure(ErrorCode::kInvalidArgument,
                   "NonMaximumSuppressionOp boxes must contain tightly packed S16 coordinates");
  }
  auto valid_rows = validate_pitched_rows(
      tensor.data(), shape.first<2U>(), byte_strides->span().first<2U>(), 8, 8U, "boxes");
  if (!valid_rows) {
    return make_unexpected(std::move(valid_rows).error());
  }
  return *byte_strides;
}

[[nodiscard]] expected<detail::FixedTensorByteStrides, Error> validate_scores(const Tensor& tensor,
                                                                              std::int64_t batch,
                                                                              std::int64_t boxes) {
  auto common = validate_common_tensor(tensor, "scores");
  if (!common) {
    return make_unexpected(std::move(common).error());
  }
  const auto shape = tensor.shape_span();
  if (!same_dtype(tensor.dtype(), kF32) || shape.size() != 2U || shape[0] != batch ||
      shape[1] != boxes) {
    return failure(ErrorCode::kInvalidArgument,
                   "NonMaximumSuppressionOp scores must be F32 [N, W] matching boxes");
  }
  auto byte_strides = detail::fixed_tensor_byte_strides(tensor);
  if (!byte_strides) {
    return make_unexpected(std::move(byte_strides).error());
  }
  auto valid_rows =
      validate_pitched_rows(tensor.data(), shape, byte_strides->span(), 4, 4U, "scores");
  if (!valid_rows) {
    return make_unexpected(std::move(valid_rows).error());
  }
  return *byte_strides;
}

}  // namespace

class NonMaximumSuppressionOp::Impl {
 public:
  std::unique_ptr<cvcuda::NonMaximumSuppression> operation;
};

NonMaximumSuppressionOp::NonMaximumSuppressionOp() : impl_(std::make_unique<Impl>()) {}

NonMaximumSuppressionOp::~NonMaximumSuppressionOp() = default;

void NonMaximumSuppressionOp::setup(OperatorSpec& spec) {
  const auto projected_capacity = nms_output_byte_span(params());
  if (!projected_capacity) {
    throw RuntimeError(projected_capacity.error().code,
                       "NonMaximumSuppressionOp capacity exceeds CV-CUDA's 32-bit tensor limit");
  }
  const auto capacity = parameter_projected_byte_span<&nms_output_byte_span>(
      "holoscan.vision.NonMaximumSuppressionOp.mask-max-byte-span.v1",
      param_ref<&NonMaximumSuppressionOpParams::max_boxes>(),
      param_ref<&NonMaximumSuppressionOpParams::max_batch_size>());

  spec.input(boxes, "boxes")
      .queue_depth(1U)
      .expects_tensor(TensorInputSpec{
          .representation = {.memory_kind = MemoryKind::kCudaDevice, .dtype = kS16, .rank = 3U},
      });
  spec.input(scores, "scores")
      .queue_depth(1U)
      .expects_tensor(TensorInputSpec{
          .representation = {.memory_kind = MemoryKind::kCudaDevice, .dtype = kF32, .rank = 2U},
      });
  spec.output(mask, "mask")
      .produces_tensor(
          TensorOutputSpec{
              .representation = {.memory_kind = MemoryKind::kCudaDevice,
                                 .dtype = kU8,
                                 .rank = 2U,
                                 .strides = TensorStridePolicy::kContiguousOnly},
              .bounds = TensorBounds{.max_rank = 2U},
              .storage = TensorOutputStorage::kRuntimePool,
          },
          capacity);
}

Contract NonMaximumSuppressionOp::contract() const {
  Contract result;
  result
      .trigger(OnAligned{.ports = ports(boxes, scores), .window = std::chrono::nanoseconds::zero()})
      .input(boxes, Exact{})
      .input(scores, Exact{});
  return result;
}

void NonMaximumSuppressionOp::start() {
  if (impl_->operation) {
    return;
  }
  try {
    impl_->operation = std::make_unique<cvcuda::NonMaximumSuppression>();
  } catch (const nvcv::Exception& exception) {
    throw RuntimeError(ErrorCode::kFailure,
                       std::string("NonMaximumSuppressionOp failed to create CV-CUDA resource: ") +
                           exception.what());
  }
}

void NonMaximumSuppressionOp::stop() {
  // Retain the handle until terminal runtime CUDA quiescence completes.
}

expected<void, Error> NonMaximumSuppressionOp::compute(ExecutionContext& context) {
  const NonMaximumSuppressionOpParams& configuration =
      context.params<NonMaximumSuppressionOpParams>();
  if (!impl_->operation) {
    return failure(ErrorCode::kNotReady,
                   "NonMaximumSuppressionOp CV-CUDA resource is not initialized");
  }
  const cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return failure(ErrorCode::kNotSupported,
                   "NonMaximumSuppressionOp requires a runtime CUDA stream");
  }

  auto boxes_sample = boxes.receive();
  if (!boxes_sample) {
    return make_unexpected(std::move(boxes_sample).error());
  }
  auto scores_sample = scores.receive();
  if (!scores_sample) {
    return make_unexpected(std::move(scores_sample).error());
  }
  auto boxes_strides =
      validate_boxes(boxes_sample->data, configuration.max_boxes, configuration.max_batch_size);
  if (!boxes_strides) {
    return make_unexpected(std::move(boxes_strides).error());
  }
  const auto boxes_shape = boxes_sample->data.shape_span();
  auto scores_strides = validate_scores(scores_sample->data, boxes_shape[0], boxes_shape[1]);
  if (!scores_strides) {
    return make_unexpected(std::move(scores_strides).error());
  }
  if (scores_sample->data.device().device_id != boxes_sample->data.device().device_id) {
    return failure(ErrorCode::kIncompatible,
                   "NonMaximumSuppressionOp boxes and scores must use the same CUDA device");
  }

  const std::array<std::int64_t, 2U> output_shape{boxes_shape[0], boxes_shape[1]};
  auto output_strides = detail::contiguous_byte_strides(output_shape, kU8);
  if (!output_strides) {
    return make_unexpected(std::move(output_strides).error());
  }
  const std::span<const std::int64_t> strides{output_strides->data(), output_shape.size()};
  auto loan = mask.allocate_tensor(TensorLoanRequest{
      .shape = output_shape,
      .dtype = kU8,
      .byte_strides = TensorByteStridesSpan{strides},
  });
  if (!loan) {
    return make_unexpected(std::move(loan).error());
  }
  if (loan->device_id().value != boxes_sample->data.device().device_id) {
    return failure(
        ErrorCode::kIncompatible,
        "NonMaximumSuppressionOp input and output tensors must use the same CUDA device");
  }
  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return make_unexpected(std::move(writer).error());
  }

  auto nvcv_boxes = detail::wrap_nvcv_tensor(boxes_sample->data.data(),
                                             boxes_shape,
                                             boxes_strides->span(),
                                             boxes_sample->data.dtype(),
                                             "NWC");
  if (!nvcv_boxes) {
    return make_unexpected(std::move(nvcv_boxes).error());
  }
  auto nvcv_scores = detail::wrap_nvcv_tensor(scores_sample->data.data(),
                                              scores_sample->data.shape_span(),
                                              scores_strides->span(),
                                              scores_sample->data.dtype(),
                                              "NW");
  if (!nvcv_scores) {
    return make_unexpected(std::move(nvcv_scores).error());
  }
  auto nvcv_mask = detail::wrap_nvcv_tensor(writer->data(), output_shape, strides, kU8, "NW");
  if (!nvcv_mask) {
    return make_unexpected(std::move(nvcv_mask).error());
  }

  try {
    (*impl_->operation)(stream,
                        *nvcv_boxes,
                        *nvcv_mask,
                        *nvcv_scores,
                        configuration.score_threshold,
                        configuration.iou_threshold);
  } catch (const nvcv::Exception& exception) {
    return failure(
        ErrorCode::kFailure,
        std::string("NonMaximumSuppressionOp CV-CUDA submission failed: ") + exception.what());
  }

  auto committed = std::move(*writer).commit();
  if (!committed) {
    return make_unexpected(std::move(committed).error());
  }
  return mask.emit(std::move(*loan), detail::forward_metadata(boxes_sample->metadata));
}

}  // namespace holoscan::vision
