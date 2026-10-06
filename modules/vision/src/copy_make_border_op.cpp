// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <cvcuda/OpCopyMakeBorder.hpp>
#include <nvcv/Exception.hpp>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/image_descriptor_scratch.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/vision/copy_make_border_op.hpp>

#include "detail/cvcuda_tensor.hpp"
#include "detail/image_payload.hpp"

namespace holoscan::vision {
namespace {

constexpr TensorRepresentation kCudaTensorRepresentation{
    .memory_kind = MemoryKind::kCudaDevice,
};

[[nodiscard]] expected<std::uint64_t, Error> border_output_byte_span(
    const CopyMakeBorderOpParams& params) noexcept {
  return detail::projected_image_output_byte_span(
      params.max_width, params.max_height, params.max_batch_size, sizeof(float));
}

[[nodiscard]] auto failure(ErrorCode code, std::string message) {
  return make_unexpected(Error{code, std::move(message)});
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
                   "CopyMakeBorderOp border_value must contain one value or one per channel");
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

class CopyMakeBorderOp::Impl {
 public:
  std::optional<sensor_io::ImageDescriptorScratch> image_metadata;
  std::unique_ptr<cvcuda::CopyMakeBorder> operation;
};

CopyMakeBorderOp::CopyMakeBorderOp() : impl_(std::make_unique<Impl>()) {}

CopyMakeBorderOp::~CopyMakeBorderOp() = default;

void CopyMakeBorderOp::setup(OperatorSpec& spec) {
  const CopyMakeBorderOpParams& configuration = params();
  const std::int64_t minimum_width =
      1LL + static_cast<std::int64_t>(configuration.left) + configuration.right;
  const std::int64_t minimum_height =
      1LL + static_cast<std::int64_t>(configuration.top) + configuration.bottom;
  if (minimum_width > configuration.max_width || minimum_height > configuration.max_height) {
    throw RuntimeError(ErrorCode::kInvalidArgument,
                       "CopyMakeBorderOp parameters exceed the authored output-pool capacity");
  }
  if (configuration.border_value.empty()) {
    throw RuntimeError(ErrorCode::kInvalidArgument,
                       "CopyMakeBorderOp border_value must not be empty");
  }
  const auto projected_capacity = border_output_byte_span(configuration);
  if (!projected_capacity) {
    throw RuntimeError(projected_capacity.error().code,
                       "CopyMakeBorderOp output capacity exceeds CV-CUDA's 32-bit tensor limit");
  }
  const auto capacity = parameter_projected_byte_span<&border_output_byte_span>(
      "holoscan.vision.CopyMakeBorderOp.output-max-byte-span.v1",
      param_ref<&CopyMakeBorderOpParams::max_width>(),
      param_ref<&CopyMakeBorderOpParams::max_height>(),
      param_ref<&CopyMakeBorderOpParams::max_batch_size>());

  spec.input(input, "input")
      .queue_depth(1U)
      .tensor_field(schema::ImageTensorFields::data)
      .expects_tensor(TensorInputSpec{
          .representation = kCudaTensorRepresentation,
          .bounds = TensorBounds{.max_rank = detail::ImageTensorView::kMaxRank},
      });
  spec.output(output, "output")
      .tensor_field(schema::ImageTensorFields::data)
      .produces_tensor(
          TensorOutputSpec{
              .representation = {.memory_kind = MemoryKind::kCudaDevice,
                                 .strides = TensorStridePolicy::kContiguousOnly},
              .bounds = TensorBounds{.max_rank = detail::ImageTensorView::kMaxRank},
              .storage = TensorOutputStorage::kRuntimePool,
          },
          capacity);
}

Contract CopyMakeBorderOp::contract() const {
  Contract result;
  result.trigger(OnEach{input});
  return result;
}

void CopyMakeBorderOp::start() {
  if (!impl_->image_metadata) {
    auto created = sensor_io::ImageDescriptorScratch::create();
    if (!created) {
      throw RuntimeError(created.error());
    }
    impl_->image_metadata.emplace(std::move(*created));
  }
  if (impl_->operation) {
    return;
  }
  auto prepared = detail::prepare_nvcv_tensor_handles(2);
  if (!prepared) {
    throw RuntimeError(prepared.error().code,
                       "CopyMakeBorderOp failed to prepare NVCV Tensor handles");
  }
  try {
    impl_->operation = std::make_unique<cvcuda::CopyMakeBorder>();
  } catch (const nvcv::Exception& exception) {
    throw RuntimeError(
        ErrorCode::kFailure,
        std::string("CopyMakeBorderOp failed to create CV-CUDA resource: ") + exception.what());
  }
}

void CopyMakeBorderOp::stop() {
  // Retain the handle until terminal runtime CUDA quiescence completes.
}

expected<void, Error> CopyMakeBorderOp::compute(ExecutionContext& context) {
  const CopyMakeBorderOpParams& configuration = context.params<CopyMakeBorderOpParams>();
  if (!impl_->operation || !impl_->image_metadata) {
    return failure(ErrorCode::kNotReady, "CopyMakeBorderOp resources are not initialized");
  }
  const cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return failure(ErrorCode::kNotSupported, "CopyMakeBorderOp requires a runtime CUDA stream");
  }

  auto sample = input.selected_sample();
  if (!sample) {
    return make_unexpected(std::move(sample).error());
  }
  auto input_view = detail::describe_image_payload(
      sample->data(), detail::TensorOperation::kCopyMakeBorder, configuration.max_batch_size);
  if (!input_view) {
    return make_unexpected(std::move(input_view).error());
  }
  auto metadata_copied = impl_->image_metadata->copy_metadata_from(sample->data());
  if (!metadata_copied) {
    return make_unexpected(std::move(metadata_copied).error());
  }
  auto& image = impl_->image_metadata->image();
  auto border_value = to_border_value(configuration.border_value, input_view->channels);
  if (!border_value) {
    return make_unexpected(std::move(border_value).error());
  }

  const std::int64_t output_width = input_view->width + configuration.left + configuration.right;
  const std::int64_t output_height = input_view->height + configuration.top + configuration.bottom;
  if (output_width > configuration.max_width || output_height > configuration.max_height) {
    return failure(ErrorCode::kInvalidArgument,
                   "CopyMakeBorderOp output exceeds the authored output-pool capacity");
  }
  std::array<std::int64_t, detail::ImageTensorView::kMaxRank> output_shape = input_view->shape;
  const std::size_t height_index = input_view->rank == 3U ? 0U : 1U;
  output_shape[height_index] = output_height;
  output_shape[height_index + 1U] = output_width;
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
                   "CopyMakeBorderOp input and output tensors must use the same CUDA device");
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
                                                   detail::TensorOperation::kCopyMakeBorder,
                                                   configuration.max_batch_size);
  if (!output_view) {
    return make_unexpected(std::move(output_view).error());
  }
  auto metadata_updated = detail::update_image_metadata(
      image,
      *input_view,
      *output_view,
      detail::ImageMetadataTransform{
          .geometry = detail::ImageGeometryChange::kPad,
          .x = configuration.left,
          .y = configuration.top,
          .preserves_precision = configuration.border_mode != BorderMode::kConstant ||
                                 detail::zero_border(configuration.border_value)});
  if (!metadata_updated) {
    return make_unexpected(std::move(metadata_updated).error());
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
    (*impl_->operation)(stream,
                        *nvcv_input,
                        *nvcv_output,
                        configuration.top,
                        configuration.left,
                        to_nvcv_border(configuration.border_mode),
                        *border_value);
  } catch (const nvcv::Exception& exception) {
    return failure(ErrorCode::kFailure,
                   std::string("CopyMakeBorderOp CV-CUDA submission failed: ") + exception.what());
  }

  auto committed = std::move(*writer).commit();
  if (!committed) {
    return make_unexpected(std::move(committed).error());
  }
  return output.emit_tensor(std::move(*loan), image, sensor_io::derive_from(sample->metadata()));
}

}  // namespace holoscan::vision
