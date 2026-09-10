// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>

#include <cvcuda/OpCustomCrop.hpp>
#include <nvcv/Exception.hpp>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/vision/custom_crop_op.hpp>

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

class CustomCropOp::Impl {
 public:
  std::unique_ptr<cvcuda::CustomCrop> operation;
};

CustomCropOp::CustomCropOp(VisionImageCapacity capacity)
    : impl_(std::make_unique<Impl>()), capacity_(capacity) {}

CustomCropOp::~CustomCropOp() = default;

void CustomCropOp::setup(OperatorSpec& spec) {
  const CustomCropOpParams& configuration = params();
  if (configuration.width > capacity_.max_width || configuration.height > capacity_.max_height) {
    throw RuntimeError(ErrorCode::kInvalidArgument,
                       "CustomCropOp parameters exceed the authored output-pool capacity");
  }
  auto byte_span = detail::maximum_output_byte_span(
      capacity_.max_width, capacity_.max_height, capacity_.max_batch_size, sizeof(double));
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

Contract CustomCropOp::contract() const {
  Contract result;
  result.trigger(OnEach{input});
  return result;
}

void CustomCropOp::start() {
  if (impl_->operation) {
    return;
  }
  try {
    impl_->operation = std::make_unique<cvcuda::CustomCrop>();
  } catch (const nvcv::Exception& exception) {
    throw RuntimeError(
        ErrorCode::kFailure,
        std::string("CustomCropOp failed to create CV-CUDA resource: ") + exception.what());
  }
}

void CustomCropOp::stop() {
  // Retain the handle until terminal runtime CUDA quiescence completes.
}

expected<void, Error> CustomCropOp::compute(ExecutionContext& context) {
  const CustomCropOpParams& configuration = context.params<CustomCropOpParams>();
  auto valid_width = detail::validate_authored_capacity(
      "CustomCropOp", "width", configuration.width, capacity_.max_width);
  if (!valid_width) {
    return make_unexpected(std::move(valid_width).error());
  }
  auto valid_height = detail::validate_authored_capacity(
      "CustomCropOp", "height", configuration.height, capacity_.max_height);
  if (!valid_height) {
    return make_unexpected(std::move(valid_height).error());
  }
  if (!impl_->operation) {
    return failure(ErrorCode::kNotReady, "CustomCropOp CV-CUDA resource is not initialized");
  }
  const cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return failure(ErrorCode::kNotSupported, "CustomCropOp requires a runtime CUDA stream");
  }

  auto sample = input.receive();
  if (!sample) {
    return make_unexpected(std::move(sample).error());
  }
  auto input_view = detail::describe_image_tensor(
      sample->data, detail::TensorOperation::kCustomCrop, capacity_.max_batch_size);
  if (!input_view) {
    return make_unexpected(std::move(input_view).error());
  }
  auto valid_crop = detail::validate_crop_rectangle(input_view->width,
                                                    input_view->height,
                                                    configuration.x,
                                                    configuration.y,
                                                    configuration.width,
                                                    configuration.height);
  if (!valid_crop) {
    return make_unexpected(std::move(valid_crop).error());
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
                   "CustomCropOp input and output tensors must use the same CUDA device");
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
                                                   detail::TensorOperation::kCustomCrop,
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
        stream,
        *nvcv_input,
        *nvcv_output,
        NVCVRectI{configuration.x, configuration.y, configuration.width, configuration.height});
  } catch (const nvcv::Exception& exception) {
    return failure(ErrorCode::kFailure,
                   std::string("CustomCropOp CV-CUDA submission failed: ") + exception.what());
  }

  auto committed = std::move(*writer).commit();
  if (!committed) {
    return make_unexpected(std::move(committed).error());
  }
  return output.emit(std::move(*loan), detail::forward_metadata(sample->metadata));
}

}  // namespace holoscan::vision
