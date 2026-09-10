// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>

#include <cvcuda/OpResize.hpp>
#include <nvcv/Exception.hpp>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/vision/resize_op.hpp>

#include "detail/cvcuda_tensor.hpp"

namespace holoscan::vision {
namespace {

constexpr TensorRepresentation kCudaTensorRepresentation{
    .memory_kind = MemoryKind::kCudaDevice,
};

[[nodiscard]] auto failure(ErrorCode code, std::string message) {
  return make_unexpected(Error{code, std::move(message)});
}

[[nodiscard]] NVCVInterpolationType to_nvcv_interpolation(InterpolationMode interpolation) {
  switch (interpolation) {
    case InterpolationMode::kNearest:
      return NVCV_INTERP_NEAREST;
    case InterpolationMode::kLinear:
      return NVCV_INTERP_LINEAR;
    case InterpolationMode::kArea:
      return NVCV_INTERP_AREA;
    case InterpolationMode::kCubic:
      return NVCV_INTERP_CUBIC;
  }
  return NVCV_INTERP_LINEAR;
}

}  // namespace

class ResizeOp::Impl {
 public:
  std::unique_ptr<cvcuda::Resize> operation;
};

ResizeOp::ResizeOp(VisionImageCapacity capacity)
    : impl_(std::make_unique<Impl>()), capacity_(capacity) {}

ResizeOp::~ResizeOp() = default;

void ResizeOp::setup(OperatorSpec& spec) {
  const ResizeOpParams& configuration = params();
  if (configuration.width > capacity_.max_width || configuration.height > capacity_.max_height) {
    throw RuntimeError(ErrorCode::kInvalidArgument,
                       "ResizeOp parameters exceed the authored output-pool capacity");
  }
  auto byte_span = detail::maximum_output_byte_span(
      capacity_.max_width, capacity_.max_height, capacity_.max_batch_size, sizeof(float));
  if (!byte_span) {
    throw RuntimeError(byte_span.error());
  }

  spec.input(input, "input")
      .queue_depth(1U)
      .expects_tensor(TensorInputSpec{
          .representation = kCudaTensorRepresentation,
          .bounds = TensorBounds{.max_rank = detail::ImageTensorView::kMaxRank},
      });
  spec.output(output, "output")
      .produces_tensor(TensorOutputSpec{
          .representation = {.memory_kind = MemoryKind::kCudaDevice,
                             .strides = TensorStridePolicy::kContiguousOnly},
          .bounds =
              TensorBounds{
                  .max_rank = detail::ImageTensorView::kMaxRank,
                  .max_byte_span = *byte_span,
              },
          .storage = TensorOutputStorage::kRuntimePool,
      });
}

Contract ResizeOp::contract() const {
  Contract result;
  result.trigger(OnEach{input});
  return result;
}

void ResizeOp::start() {
  if (impl_->operation) {
    return;
  }
  try {
    impl_->operation = std::make_unique<cvcuda::Resize>();
  } catch (const nvcv::Exception& exception) {
    throw RuntimeError(
        ErrorCode::kFailure,
        std::string("ResizeOp failed to create CV-CUDA resource: ") + exception.what());
  }
}

void ResizeOp::stop() {
  // Retain the handle until terminal runtime CUDA quiescence completes.
}

expected<void, Error> ResizeOp::compute(ExecutionContext& context) {
  const ResizeOpParams& configuration = context.params<ResizeOpParams>();
  auto valid_width = detail::validate_authored_capacity(
      "ResizeOp", "width", configuration.width, capacity_.max_width);
  if (!valid_width) {
    return make_unexpected(std::move(valid_width).error());
  }
  auto valid_height = detail::validate_authored_capacity(
      "ResizeOp", "height", configuration.height, capacity_.max_height);
  if (!valid_height) {
    return make_unexpected(std::move(valid_height).error());
  }
  if (!impl_->operation) {
    return failure(ErrorCode::kNotReady, "ResizeOp CV-CUDA resource is not initialized");
  }
  const cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return failure(ErrorCode::kNotSupported, "ResizeOp requires a runtime CUDA stream");
  }

  auto sample = input.receive();
  if (!sample) {
    return make_unexpected(std::move(sample).error());
  }
  auto input_view = detail::describe_image_tensor(
      sample->data, detail::TensorOperation::kResize, capacity_.max_batch_size);
  if (!input_view) {
    return make_unexpected(std::move(input_view).error());
  }

  std::array<std::int64_t, detail::ImageTensorView::kMaxRank> output_shape = input_view->shape;
  const std::size_t height_index = input_view->rank == 3U ? 0U : 1U;
  output_shape[height_index] = configuration.height;
  output_shape[height_index + 1U] = configuration.width;
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
                   "ResizeOp input and output tensors must use the same CUDA device");
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
                                                   detail::TensorOperation::kResize,
                                                   capacity_.max_batch_size);
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
    (*impl_->operation)(
        stream, *nvcv_input, *nvcv_output, to_nvcv_interpolation(configuration.interpolation));
  } catch (const nvcv::Exception& exception) {
    return failure(ErrorCode::kFailure,
                   std::string("ResizeOp CV-CUDA submission failed: ") + exception.what());
  }

  auto committed = std::move(*writer).commit();
  if (!committed) {
    return make_unexpected(std::move(committed).error());
  }
  return output.emit(std::move(*loan), detail::forward_metadata(sample->metadata));
}

}  // namespace holoscan::vision
