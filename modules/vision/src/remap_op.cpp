// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <cvcuda/OpRemap.hpp>
#include <nvcv/Exception.hpp>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/vision/remap_op.hpp>

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
    case InterpolationMode::kCubic:
      return NVCV_INTERP_CUBIC;
    case InterpolationMode::kArea:
      break;
  }
  return NVCV_INTERP_LINEAR;
}

[[nodiscard]] NVCVRemapMapValueType to_nvcv_map_value_mode(RemapMapValueMode mode) {
  switch (mode) {
    case RemapMapValueMode::kAbsolute:
      return NVCV_REMAP_ABSOLUTE;
    case RemapMapValueMode::kAbsoluteNormalized:
      return NVCV_REMAP_ABSOLUTE_NORMALIZED;
    case RemapMapValueMode::kRelativeNormalized:
      return NVCV_REMAP_RELATIVE_NORMALIZED;
  }
  return NVCV_REMAP_ABSOLUTE;
}

[[nodiscard]] NVCVBorderType to_nvcv_border(BorderMode mode) {
  switch (mode) {
    case BorderMode::kConstant:
      return NVCV_BORDER_CONSTANT;
    case BorderMode::kReplicate:
      return NVCV_BORDER_REPLICATE;
    case BorderMode::kReflect:
      return NVCV_BORDER_REFLECT;
    case BorderMode::kWrap:
      return NVCV_BORDER_WRAP;
    case BorderMode::kReflect101:
      return NVCV_BORDER_REFLECT101;
  }
  return NVCV_BORDER_CONSTANT;
}

[[nodiscard]] expected<float4, Error> to_border_value(const std::vector<float>& values,
                                                      std::int64_t channels) {
  if (values.empty() ||
      (values.size() != 1U && values.size() != static_cast<std::size_t>(channels))) {
    return failure(ErrorCode::kInvalidArgument,
                   "RemapOp border_value must contain one value or one per channel");
  }
  if (values.size() == 1U) {
    return float4{values[0], values[0], values[0], values[0]};
  }
  return float4{values[0],
                values.size() > 1U ? values[1] : 0.F,
                values.size() > 2U ? values[2] : 0.F,
                values.size() > 3U ? values[3] : 0.F};
}

}  // namespace

class RemapOp::Impl {
 public:
  std::unique_ptr<cvcuda::Remap> operation;
};

RemapOp::RemapOp(VisionImageCapacity capacity)
    : impl_(std::make_unique<Impl>()), capacity_(capacity) {}

RemapOp::~RemapOp() = default;

void RemapOp::setup(OperatorSpec& spec) {
  const RemapOpParams& configuration = params();
  if (configuration.width > capacity_.max_width || configuration.height > capacity_.max_height) {
    throw RuntimeError(ErrorCode::kInvalidArgument,
                       "RemapOp parameters exceed the authored output-pool capacity");
  }
  if (configuration.border_value.empty()) {
    throw RuntimeError(ErrorCode::kInvalidArgument, "RemapOp border_value must not be empty");
  }
  auto byte_span = detail::maximum_output_byte_span(
      capacity_.max_width, capacity_.max_height, capacity_.max_batch_size, sizeof(std::uint8_t));
  if (!byte_span) {
    throw RuntimeError(byte_span.error());
  }

  spec.input(input, "input")
      .queue_depth(1U)
      .expects_tensor(TensorInputSpec{
          .representation = kCudaTensorRepresentation,
          .bounds = TensorBounds{.max_rank = detail::ImageTensorView::kMaxRank},
      });
  // A retained map must leave a second slot available for a newer publication. Queue depth is
  // authored port shape, so it cannot vary with map_synchronization.
  spec.input(map, "map")
      .queue_depth(2U)
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

Contract RemapOp::contract() const {
  Contract result;
  const RemapOpParams& configuration = params();
  if (configuration.map_synchronization == RemapMapSynchronizationMode::kLatest) {
    result.trigger(OnEach{input})
        .input(map, LatestWithin{.max_age = configuration.map_max_age})
        .on_missing(map, ReuseLast{.max_age = configuration.map_max_age});
  } else {
    result
        .trigger(OnAligned{.ports = ports(input, map), .window = std::chrono::nanoseconds::zero()})
        .input(input, Exact{})
        .input(map, Exact{});
  }
  return result;
}

void RemapOp::start() {
  if (impl_->operation) {
    return;
  }
  try {
    impl_->operation = std::make_unique<cvcuda::Remap>();
  } catch (const nvcv::Exception& exception) {
    throw RuntimeError(
        ErrorCode::kFailure,
        std::string("RemapOp failed to create CV-CUDA resource: ") + exception.what());
  }
}

void RemapOp::stop() {
  // Retain the handle until terminal runtime CUDA quiescence completes.
}

expected<void, Error> RemapOp::compute(ExecutionContext& context) {
  const RemapOpParams& configuration = context.params<RemapOpParams>();
  auto valid_width = detail::validate_authored_capacity(
      "RemapOp", "width", configuration.width, capacity_.max_width);
  if (!valid_width) {
    return make_unexpected(std::move(valid_width).error());
  }
  auto valid_height = detail::validate_authored_capacity(
      "RemapOp", "height", configuration.height, capacity_.max_height);
  if (!valid_height) {
    return make_unexpected(std::move(valid_height).error());
  }
  if (!impl_->operation) {
    return failure(ErrorCode::kNotReady, "RemapOp CV-CUDA resource is not initialized");
  }
  const cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return failure(ErrorCode::kNotSupported, "RemapOp requires a runtime CUDA stream");
  }

  auto input_sample = input.receive();
  if (!input_sample) {
    return make_unexpected(std::move(input_sample).error());
  }
  auto map_sample = map.receive();
  if (!map_sample) {
    return make_unexpected(std::move(map_sample).error());
  }
  auto input_view = detail::describe_image_tensor(
      input_sample->data, detail::TensorOperation::kRemap, capacity_.max_batch_size);
  if (!input_view) {
    return make_unexpected(std::move(input_view).error());
  }
  auto map_view = detail::describe_image_tensor(
      map_sample->data, detail::TensorOperation::kRemapMap, capacity_.max_batch_size);
  if (!map_view) {
    return make_unexpected(std::move(map_view).error());
  }
  if (map_view->device.device_id != input_view->device.device_id) {
    return failure(ErrorCode::kIncompatible,
                   "RemapOp image and map tensors must use the same CUDA device");
  }
  if (map_view->batch != 1 && map_view->batch != input_view->batch) {
    return failure(ErrorCode::kIncompatible,
                   "RemapOp map batch size must be one or match the image batch size");
  }
  auto border_value = to_border_value(configuration.border_value, input_view->channels);
  if (!border_value) {
    return make_unexpected(std::move(border_value).error());
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
                   "RemapOp input and output tensors must use the same CUDA device");
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
                                                   detail::TensorOperation::kRemap,
                                                   capacity_.max_batch_size);
  if (!output_view) {
    return make_unexpected(std::move(output_view).error());
  }
  auto nvcv_input = detail::wrap_nvcv_tensor(*input_view);
  if (!nvcv_input) {
    return make_unexpected(std::move(nvcv_input).error());
  }
  auto nvcv_map = detail::wrap_nvcv_tensor(*map_view);
  if (!nvcv_map) {
    return make_unexpected(std::move(nvcv_map).error());
  }
  auto nvcv_output = detail::wrap_nvcv_tensor(*output_view);
  if (!nvcv_output) {
    return make_unexpected(std::move(nvcv_output).error());
  }

  try {
    (*impl_->operation)(stream,
                        *nvcv_input,
                        *nvcv_output,
                        *nvcv_map,
                        to_nvcv_interpolation(configuration.input_interpolation),
                        to_nvcv_interpolation(configuration.map_interpolation),
                        to_nvcv_map_value_mode(configuration.map_value_mode),
                        configuration.align_corners,
                        to_nvcv_border(configuration.border_mode),
                        *border_value);
  } catch (const nvcv::Exception& exception) {
    return failure(ErrorCode::kFailure,
                   std::string("RemapOp CV-CUDA submission failed: ") + exception.what());
  }

  auto committed = std::move(*writer).commit();
  if (!committed) {
    return make_unexpected(std::move(committed).error());
  }
  return output.emit(std::move(*loan), detail::forward_metadata(input_sample->metadata));
}

}  // namespace holoscan::vision
