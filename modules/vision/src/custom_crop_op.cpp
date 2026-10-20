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

#include <cvcuda/OpCustomCrop.hpp>
#include <nvcv/Exception.hpp>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/image_descriptor_scratch.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/vision/custom_crop_op.hpp>

#include "detail/cvcuda_tensor.hpp"
#include "detail/image_payload.hpp"

namespace holoscan::vision {
namespace {

constexpr TensorRepresentation kCudaTensorRepresentation{
    .memory_kind = MemoryKind::kCudaDevice,
};

[[nodiscard]] expected<std::uint64_t, Error> crop_output_byte_span(
    const CustomCropOpParams& params) noexcept {
  return detail::projected_image_output_byte_span(
      params.width, params.height, params.max_batch_size, sizeof(double));
}

[[nodiscard]] auto failure(ErrorCode code, std::string message) {
  return make_unexpected(Error{code, std::move(message)});
}

}  // namespace

class CustomCropOp::Impl {
 public:
  std::optional<sensor_io::ImageDescriptorScratch> image_metadata;
  std::unique_ptr<cvcuda::CustomCrop> operation;
};

CustomCropOp::CustomCropOp() : impl_(std::make_unique<Impl>()) {}

CustomCropOp::~CustomCropOp() = default;

void CustomCropOp::setup(OperatorSpec& spec) {
  const auto projected_capacity = crop_output_byte_span(params());
  if (!projected_capacity) {
    throw RuntimeError(projected_capacity.error().code,
                       "CustomCropOp output capacity exceeds CV-CUDA's 32-bit tensor limit");
  }
  const auto capacity = parameter_projected_byte_span<&crop_output_byte_span>(
      "holoscan.vision.CustomCropOp.output-max-byte-span.v1",
      param_ref<&CustomCropOpParams::width>(),
      param_ref<&CustomCropOpParams::height>(),
      param_ref<&CustomCropOpParams::max_batch_size>());

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

Contract CustomCropOp::contract() const {
  Contract result;
  result.trigger(OnEach{input});
  return result;
}

void CustomCropOp::start() {
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
    throw RuntimeError(prepared.error().code, "CustomCropOp failed to prepare NVCV Tensor handles");
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
  auto& prepared_metadata = impl_->image_metadata;
  if (!impl_->operation || !prepared_metadata.has_value()) {
    return failure(ErrorCode::kNotReady, "CustomCropOp resources are not initialized");
  }
  auto& image_metadata = *prepared_metadata;
  auto* const stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return failure(ErrorCode::kNotSupported, "CustomCropOp requires a runtime CUDA stream");
  }

  auto sample = input.selected_sample();
  if (!sample) {
    return make_unexpected(std::move(sample).error());
  }
  auto input_view = detail::describe_image_payload(
      sample->data(), detail::TensorOperation::kCustomCrop, configuration.max_batch_size);
  if (!input_view) {
    return make_unexpected(std::move(input_view).error());
  }
  auto metadata_copied = image_metadata.copy_metadata_from(sample->data());
  if (!metadata_copied) {
    return make_unexpected(std::move(metadata_copied).error());
  }
  auto& image = image_metadata.image();
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
                                                   configuration.max_batch_size);
  if (!output_view) {
    return make_unexpected(std::move(output_view).error());
  }
  auto metadata_updated = detail::update_image_metadata(
      image,
      *input_view,
      *output_view,
      detail::ImageMetadataTransform{.geometry = detail::ImageGeometryChange::kCrop,
                                     .x = configuration.x,
                                     .y = configuration.y});
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
  return output.emit_tensor(std::move(*loan), image, sensor_io::derive_from(sample->metadata()));
}

}  // namespace holoscan::vision
