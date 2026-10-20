// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>

#include <cvcuda/OpConvertTo.hpp>
#include <nvcv/Exception.hpp>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/image_descriptor_scratch.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/vision/convert_to_op.hpp>

#include "detail/cvcuda_tensor.hpp"
#include "detail/image_payload.hpp"

namespace holoscan::vision {
namespace {

constexpr TensorRepresentation kCudaTensorRepresentation{
    .memory_kind = MemoryKind::kCudaDevice,
};

[[nodiscard]] expected<std::uint64_t, Error> convert_output_element_size(
    const ConvertToOpParams& params) noexcept {
  switch (params.output_type) {
    case VisionElementType::kU8:
    case VisionElementType::kS8:
      return sizeof(std::uint8_t);
    case VisionElementType::kU16:
    case VisionElementType::kS16:
      return sizeof(std::uint16_t);
    case VisionElementType::kS32:
    case VisionElementType::kF32:
      return sizeof(std::uint32_t);
    case VisionElementType::kF64:
      return sizeof(std::uint64_t);
  }
  return make_unexpected(Error{ErrorCode::kInvalidArgument});
}

[[nodiscard]] auto failure(ErrorCode code, std::string message) {
  return make_unexpected(Error{code, std::move(message)});
}

[[nodiscard]] DLDataType to_dl_data_type(VisionElementType element_type) {
  switch (element_type) {
    case VisionElementType::kU8:
      return DLDataType{kDLUInt, 8U, 1U};
    case VisionElementType::kS8:
      return DLDataType{kDLInt, 8U, 1U};
    case VisionElementType::kU16:
      return DLDataType{kDLUInt, 16U, 1U};
    case VisionElementType::kS16:
      return DLDataType{kDLInt, 16U, 1U};
    case VisionElementType::kS32:
      return DLDataType{kDLInt, 32U, 1U};
    case VisionElementType::kF32:
      return DLDataType{kDLFloat, 32U, 1U};
    case VisionElementType::kF64:
      return DLDataType{kDLFloat, 64U, 1U};
  }
  return DLDataType{kDLFloat, 32U, 1U};
}

[[nodiscard]] NVCVRoundMode to_nvcv_rounding(RoundingMode rounding) {
  switch (rounding) {
    case RoundingMode::kNearest:
      return NVCV_ROUND_NEAREST;
    case RoundingMode::kTruncate:
      return NVCV_ROUND_TRUNCATE;
  }
  return NVCV_ROUND_NEAREST;
}

}  // namespace

class ConvertToOp::Impl {
 public:
  std::optional<sensor_io::ImageDescriptorScratch> image_metadata;
  std::unique_ptr<cvcuda::ConvertTo> operation;
};

ConvertToOp::ConvertToOp() : impl_(std::make_unique<Impl>()) {}

ConvertToOp::~ConvertToOp() = default;

void ConvertToOp::setup(OperatorSpec& spec) {
  const auto output_element_size = parameter_projected_byte_span<&convert_output_element_size>(
      "holoscan.vision.ConvertToOp.output-element-size.v1",
      param_ref<&ConvertToOpParams::output_type>());

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
          // The producer bound is measured in bytes. Multiplying by output element size is a safe
          // dtype ratio because every admitted input element occupies at least one byte.
          input_max_byte_span(input).multiplied_by(output_element_size));
}

Contract ConvertToOp::contract() const {
  Contract result;
  result.trigger(OnEach{input});
  return result;
}

void ConvertToOp::start() {
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
    throw RuntimeError(prepared.error().code, "ConvertToOp failed to prepare NVCV Tensor handles");
  }
  try {
    impl_->operation = std::make_unique<cvcuda::ConvertTo>();
  } catch (const nvcv::Exception& exception) {
    throw RuntimeError(
        ErrorCode::kFailure,
        std::string("ConvertToOp failed to create CV-CUDA resource: ") + exception.what());
  }
}

void ConvertToOp::stop() {
  // Retain the handle until terminal runtime CUDA quiescence completes.
}

expected<void, Error> ConvertToOp::compute(ExecutionContext& context) {
  const ConvertToOpParams& configuration = context.params<ConvertToOpParams>();
  auto& prepared_metadata = impl_->image_metadata;
  if (!impl_->operation || !prepared_metadata.has_value()) {
    return failure(ErrorCode::kNotReady, "ConvertToOp resources are not initialized");
  }
  auto& image_metadata = *prepared_metadata;
  auto* const stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return failure(ErrorCode::kNotSupported, "ConvertToOp requires a runtime CUDA stream");
  }

  auto sample = input.selected_sample();
  if (!sample) {
    return make_unexpected(std::move(sample).error());
  }
  auto input_view = detail::describe_image_payload(
      sample->data(), detail::TensorOperation::kConvertTo, kMaximumBatchSize);
  if (!input_view) {
    return make_unexpected(std::move(input_view).error());
  }
  auto metadata_copied = image_metadata.copy_metadata_from(sample->data());
  if (!metadata_copied) {
    return make_unexpected(std::move(metadata_copied).error());
  }
  auto& image = image_metadata.image();

  const DLDataType output_dtype = to_dl_data_type(configuration.output_type);
  const std::span<const std::int64_t> shape{input_view->shape.data(), input_view->rank};
  auto output_strides = detail::contiguous_byte_strides(shape, output_dtype);
  if (!output_strides) {
    return make_unexpected(std::move(output_strides).error());
  }
  const std::span<const std::int64_t> strides{output_strides->data(), input_view->rank};
  auto loan = output.allocate_tensor(TensorLoanRequest{
      .shape = shape,
      .dtype = output_dtype,
      .byte_strides = TensorByteStridesSpan{strides},
  });
  if (!loan) {
    return make_unexpected(std::move(loan).error());
  }
  if (loan->device_id().value != input_view->device.device_id) {
    return failure(ErrorCode::kIncompatible,
                   "ConvertToOp input and output tensors must use the same CUDA device");
  }
  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return make_unexpected(std::move(writer).error());
  }

  auto output_view = detail::describe_image_tensor(writer->data(),
                                                   shape,
                                                   strides,
                                                   output_dtype,
                                                   DLDevice{kDLCUDA, loan->device_id().value},
                                                   detail::TensorOperation::kConvertTo,
                                                   kMaximumBatchSize);
  if (!output_view) {
    return make_unexpected(std::move(output_view).error());
  }
  auto metadata_updated = detail::update_image_metadata(
      image,
      *input_view,
      *output_view,
      detail::ImageMetadataTransform{
          .geometry = detail::ImageGeometryChange::kPreserve,
          .values_changed = configuration.alpha != 1.0 || configuration.beta != 0.0});
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
                        configuration.alpha,
                        configuration.beta,
                        to_nvcv_rounding(configuration.rounding));
  } catch (const nvcv::Exception& exception) {
    return failure(ErrorCode::kFailure,
                   std::string("ConvertToOp CV-CUDA submission failed: ") + exception.what());
  }

  auto committed = std::move(*writer).commit();
  if (!committed) {
    return make_unexpected(std::move(committed).error());
  }
  return output.emit_tensor(std::move(*loan), image, sensor_io::derive_from(sample->metadata()));
}

}  // namespace holoscan::vision
