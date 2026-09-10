// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cvcuda/OpNormalize.hpp>
#include <nvcv/Exception.hpp>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/vision/normalize_op.hpp>

#include "detail/cvcuda_tensor.hpp"

namespace holoscan::vision {
namespace {

constexpr TensorRepresentation kCudaTensorRepresentation{
    .memory_kind = MemoryKind::kCudaDevice,
};

struct ChannelValues {
  float4 values{};
  std::int32_t channels{};
};

[[nodiscard]] auto failure(ErrorCode code, std::string message) {
  return make_unexpected(Error{code, std::move(message)});
}

[[nodiscard]] expected<void, Error> validate_standard_deviation_scale(
    const NormalizeOpParams& configuration) {
  if (!configuration.scale_is_stddev || configuration.epsilon != 0.F) {
    return {};
  }
  for (const float value : configuration.scale) {
    if (value == 0.F) {
      return failure(
          ErrorCode::kInvalidArgument,
          "NormalizeOp scale values must be nonzero when scale_is_stddev is true and epsilon is 0");
    }
  }
  return {};
}

[[nodiscard]] expected<ChannelValues, Error> to_channel_values(const std::vector<float>& values,
                                                               std::int64_t image_channels,
                                                               std::string_view name) {
  if (values.empty() ||
      (values.size() != 1U && values.size() != static_cast<std::size_t>(image_channels))) {
    return failure(
        ErrorCode::kInvalidArgument,
        "NormalizeOp " + std::string(name) + " must contain one value or one per channel");
  }
  return ChannelValues{.values = float4{values[0],
                                        values.size() > 1U ? values[1] : 0.F,
                                        values.size() > 2U ? values[2] : 0.F,
                                        values.size() > 3U ? values[3] : 0.F},
                       .channels = static_cast<std::int32_t>(values.size())};
}

}  // namespace

class NormalizeOp::Impl {
 public:
  std::unique_ptr<cvcuda::Normalize> operation;
};

NormalizeOp::NormalizeOp(VisionImageCapacity capacity)
    : impl_(std::make_unique<Impl>()), capacity_(capacity) {}

NormalizeOp::~NormalizeOp() = default;

void NormalizeOp::setup(OperatorSpec& spec) {
  const NormalizeOpParams& configuration = params();
  if (configuration.base.empty() || configuration.scale.empty()) {
    throw RuntimeError(ErrorCode::kInvalidArgument, "NormalizeOp base and scale must not be empty");
  }
  auto valid_scale = validate_standard_deviation_scale(configuration);
  if (!valid_scale) {
    throw RuntimeError(valid_scale.error());
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

Contract NormalizeOp::contract() const {
  Contract result;
  result.trigger(OnEach{input});
  return result;
}

void NormalizeOp::start() {
  if (impl_->operation) {
    return;
  }
  try {
    impl_->operation = std::make_unique<cvcuda::Normalize>();
  } catch (const nvcv::Exception& exception) {
    throw RuntimeError(
        ErrorCode::kFailure,
        std::string("NormalizeOp failed to create CV-CUDA resource: ") + exception.what());
  }
}

void NormalizeOp::stop() {
  // Retain the handle until terminal runtime CUDA quiescence completes.
}

expected<void, Error> NormalizeOp::compute(ExecutionContext& context) {
  const NormalizeOpParams& configuration = context.params<NormalizeOpParams>();
  if (!impl_->operation) {
    return failure(ErrorCode::kNotReady, "NormalizeOp CV-CUDA resource is not initialized");
  }
  const cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return failure(ErrorCode::kNotSupported, "NormalizeOp requires a runtime CUDA stream");
  }

  auto sample = input.receive();
  if (!sample) {
    return make_unexpected(std::move(sample).error());
  }
  auto input_view = detail::describe_image_tensor(
      sample->data, detail::TensorOperation::kNormalize, capacity_.max_batch_size);
  if (!input_view) {
    return make_unexpected(std::move(input_view).error());
  }
  if (input_view->width > capacity_.max_width || input_view->height > capacity_.max_height) {
    return failure(ErrorCode::kInvalidArgument,
                   "NormalizeOp output exceeds the authored output-pool capacity");
  }
  auto base = to_channel_values(configuration.base, input_view->channels, "base");
  if (!base) {
    return make_unexpected(std::move(base).error());
  }
  auto scale = to_channel_values(configuration.scale, input_view->channels, "scale");
  if (!scale) {
    return make_unexpected(std::move(scale).error());
  }
  auto valid_scale = validate_standard_deviation_scale(configuration);
  if (!valid_scale) {
    return make_unexpected(std::move(valid_scale).error());
  }

  const std::span<const std::int64_t> shape{input_view->shape.data(), input_view->rank};
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
                   "NormalizeOp input and output tensors must use the same CUDA device");
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
                                                   detail::TensorOperation::kNormalize,
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

  const std::uint32_t flags = configuration.scale_is_stddev ? CVCUDA_NORMALIZE_SCALE_IS_STDDEV : 0U;
  try {
    (*impl_->operation)(stream,
                        *nvcv_input,
                        base->values,
                        scale->values,
                        base->channels,
                        scale->channels,
                        *nvcv_output,
                        configuration.global_scale,
                        configuration.shift,
                        configuration.epsilon,
                        flags);
  } catch (const nvcv::Exception& exception) {
    return failure(ErrorCode::kFailure,
                   std::string("NormalizeOp CV-CUDA submission failed: ") + exception.what());
  }

  auto committed = std::move(*writer).commit();
  if (!committed) {
    return make_unexpected(std::move(committed).error());
  }
  return output.emit(std::move(*loan), detail::forward_metadata(sample->metadata));
}

}  // namespace holoscan::vision
