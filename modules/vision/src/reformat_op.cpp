// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>

#include <cvcuda/OpReformat.hpp>
#include <nvcv/Exception.hpp>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/vision/reformat_op.hpp>

#include "detail/cvcuda_tensor.hpp"

namespace holoscan::vision {
namespace {

constexpr TensorRepresentation kCudaTensorRepresentation{
    .memory_kind = MemoryKind::kCudaDevice,
};

[[nodiscard]] auto failure(ErrorCode code, std::string message) {
  return make_unexpected(Error{code, std::move(message)});
}

}  // namespace

class ReformatOp::Impl {
 public:
  std::unique_ptr<cvcuda::Reformat> operation;
};

ReformatOp::ReformatOp() : impl_(std::make_unique<Impl>()) {}

ReformatOp::~ReformatOp() = default;

void ReformatOp::setup(OperatorSpec& spec) {
  spec.input(input, "input")
      .queue_depth(1U)
      .expects_tensor(TensorInputSpec{
          .representation = kCudaTensorRepresentation,
          .bounds = TensorBounds{.max_rank = detail::ImageTensorView::kMaxRank},
      });
  spec.output(output, "output")
      .produces_tensor(
          TensorOutputSpec{
              .representation = {.memory_kind = MemoryKind::kCudaDevice,
                                 .strides = TensorStridePolicy::kContiguousOnly},
              .bounds = TensorBounds{.max_rank = detail::ImageTensorView::kMaxRank},
              .storage = TensorOutputStorage::kRuntimePool,
          },
          input_max_byte_span(input));
}

Contract ReformatOp::contract() const {
  Contract result;
  result.trigger(OnEach{input});
  return result;
}

void ReformatOp::start() {
  if (impl_->operation) {
    return;
  }
  try {
    impl_->operation = std::make_unique<cvcuda::Reformat>();
  } catch (const nvcv::Exception& exception) {
    throw RuntimeError(
        ErrorCode::kFailure,
        std::string("ReformatOp failed to create CV-CUDA resource: ") + exception.what());
  }
}

void ReformatOp::stop() {
  // Retain the handle until terminal runtime CUDA quiescence completes.
}

expected<void, Error> ReformatOp::compute(ExecutionContext& context) {
  const ReformatOpParams& configuration = context.params<ReformatOpParams>();
  if (!impl_->operation) {
    return failure(ErrorCode::kNotReady, "ReformatOp CV-CUDA resource is not initialized");
  }
  const cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return failure(ErrorCode::kNotSupported, "ReformatOp requires a runtime CUDA stream");
  }

  auto sample = input.receive();
  if (!sample) {
    return make_unexpected(std::move(sample).error());
  }
  const bool interleaved_to_planar = configuration.mode == ReformatMode::kInterleavedToPlanar;
  const detail::ImageTensorLayout input_layout = interleaved_to_planar
                                                     ? detail::ImageTensorLayout::kInterleaved
                                                     : detail::ImageTensorLayout::kPlanar;
  const detail::ImageTensorLayout output_layout = interleaved_to_planar
                                                      ? detail::ImageTensorLayout::kPlanar
                                                      : detail::ImageTensorLayout::kInterleaved;
  auto input_view = detail::describe_image_tensor(
      sample->data, detail::TensorOperation::kReformat, kMaximumBatchSize, input_layout);
  if (!input_view) {
    return make_unexpected(std::move(input_view).error());
  }

  std::array<std::int64_t, detail::ImageTensorView::kMaxRank> output_shape{};
  if (input_view->rank == 3U) {
    output_shape =
        interleaved_to_planar
            ? std::array<std::int64_t, detail::ImageTensorView::kMaxRank>{input_view->channels,
                                                                          input_view->height,
                                                                          input_view->width,
                                                                          0}
            : std::array<std::int64_t, detail::ImageTensorView::kMaxRank>{
                  input_view->height, input_view->width, input_view->channels, 0};
  } else {
    output_shape =
        interleaved_to_planar
            ? std::array<std::int64_t, detail::ImageTensorView::kMaxRank>{input_view->batch,
                                                                          input_view->channels,
                                                                          input_view->height,
                                                                          input_view->width}
            : std::array<std::int64_t, detail::ImageTensorView::kMaxRank>{
                  input_view->batch, input_view->height, input_view->width, input_view->channels};
  }
  const std::span<const std::int64_t> shape{output_shape.data(), input_view->rank};
  auto output_strides = detail::contiguous_byte_strides(shape, input_view->dtype);
  if (!output_strides) {
    return make_unexpected(std::move(output_strides).error());
  }
  const std::span<const std::int64_t> strides{output_strides->data(), input_view->rank};
  auto loan = output.allocate_tensor(TensorLoanRequest{
      .shape = shape,
      .dtype = input_view->dtype,
      .byte_strides = TensorByteStridesSpan{strides},
  });
  if (!loan) {
    return make_unexpected(std::move(loan).error());
  }
  if (loan->device_id().value != input_view->device.device_id) {
    return failure(ErrorCode::kIncompatible,
                   "ReformatOp input and output tensors must use the same CUDA device");
  }
  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return make_unexpected(std::move(writer).error());
  }

  auto output_view = detail::describe_image_tensor(writer->data(),
                                                   shape,
                                                   strides,
                                                   input_view->dtype,
                                                   DLDevice{kDLCUDA, loan->device_id().value},
                                                   detail::TensorOperation::kReformat,
                                                   kMaximumBatchSize,
                                                   output_layout);
  if (!output_view) {
    return make_unexpected(std::move(output_view).error());
  }
  auto nvcv_input = detail::wrap_nvcv_tensor(*input_view);
  if (!nvcv_input) {
    return make_unexpected(std::move(nvcv_input).error());
  }
  auto nvcv_output = detail::wrap_nvcv_tensor(*output_view);
  if (!nvcv_output) {
    return make_unexpected(std::move(nvcv_output).error());
  }

  try {
    (*impl_->operation)(stream, *nvcv_input, *nvcv_output);
  } catch (const nvcv::Exception& exception) {
    return failure(ErrorCode::kFailure,
                   std::string("ReformatOp CV-CUDA submission failed: ") + exception.what());
  }

  auto committed = std::move(*writer).commit();
  if (!committed) {
    return make_unexpected(std::move(committed).error());
  }
  return output.emit(std::move(*loan), detail::forward_metadata(sample->metadata));
}

}  // namespace holoscan::vision
