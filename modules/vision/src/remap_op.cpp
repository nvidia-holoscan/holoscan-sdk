// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <cvcuda/OpRemap.hpp>
#include <nvcv/Exception.hpp>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/image_descriptor_scratch.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/vision/remap_op.hpp>

#include "detail/cvcuda_tensor.hpp"
#include "detail/image_payload.hpp"

namespace holoscan::vision {
namespace {

constexpr TensorRepresentation kCudaTensorRepresentation{
    .memory_kind = MemoryKind::kCudaDevice,
};

[[nodiscard]] expected<std::uint64_t, Error> remap_output_byte_span(
    const RemapOpParams& params) noexcept {
  return detail::projected_image_output_byte_span(
      params.width, params.height, params.max_batch_size, sizeof(std::uint8_t));
}

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
  std::optional<sensor_io::ImageDescriptorScratch> image_metadata;
  std::unique_ptr<cvcuda::Remap> operation;
};

RemapOp::RemapOp() : impl_(std::make_unique<Impl>()) {}

RemapOp::~RemapOp() = default;

void RemapOp::setup(OperatorSpec& spec) {
  const RemapOpParams& configuration = params();
  if (configuration.border_value.empty()) {
    throw RuntimeError(ErrorCode::kInvalidArgument, "RemapOp border_value must not be empty");
  }
  const auto projected_capacity = remap_output_byte_span(configuration);
  if (!projected_capacity) {
    throw RuntimeError(projected_capacity.error().code,
                       "RemapOp output capacity exceeds CV-CUDA's 32-bit tensor limit");
  }
  const auto capacity = parameter_projected_byte_span<&remap_output_byte_span>(
      "holoscan.vision.RemapOp.output-max-byte-span.v1",
      param_ref<&RemapOpParams::width>(),
      param_ref<&RemapOpParams::height>(),
      param_ref<&RemapOpParams::max_batch_size>());

  spec.input(input, "input")
      .queue_depth(1U)
      .tensor_field(schema::ImageTensorFields::data)
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
  auto prepared = detail::prepare_nvcv_tensor_handles(3);
  if (!prepared) {
    throw RuntimeError(prepared.error().code, "RemapOp failed to prepare NVCV Tensor handles");
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

// Keep retained-map selection, image borrowing, asynchronous submission, and publication in
// order so provenance and map lifetime remain visible through every early return.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void, Error> RemapOp::compute(ExecutionContext& context) {
  const RemapOpParams& configuration = context.params<RemapOpParams>();
  auto& prepared_metadata = impl_->image_metadata;
  if (!impl_->operation || !prepared_metadata.has_value()) {
    return failure(ErrorCode::kNotReady, "RemapOp resources are not initialized");
  }
  auto& image_metadata = *prepared_metadata;
  auto* const stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return failure(ErrorCode::kNotSupported, "RemapOp requires a runtime CUDA stream");
  }

  auto input_sample = input.selected_sample();
  if (!input_sample) {
    return make_unexpected(std::move(input_sample).error());
  }
  auto map_sample = map.receive();
  if (!map_sample) {
    return make_unexpected(std::move(map_sample).error());
  }
  auto input_view = detail::describe_image_payload(
      input_sample->data(), detail::TensorOperation::kRemap, configuration.max_batch_size);
  if (!input_view) {
    return make_unexpected(std::move(input_view).error());
  }
  auto metadata_copied = image_metadata.copy_metadata_from(input_sample->data());
  if (!metadata_copied) {
    return make_unexpected(std::move(metadata_copied).error());
  }
  auto& image = image_metadata.image();
  auto map_view = detail::describe_image_tensor(
      map_sample->data, detail::TensorOperation::kRemapMap, configuration.max_batch_size);
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
                                                   configuration.max_batch_size);
  if (!output_view) {
    return make_unexpected(std::move(output_view).error());
  }
  auto metadata_updated = detail::update_image_metadata(
      image,
      *input_view,
      *output_view,
      detail::ImageMetadataTransform{
          .geometry = detail::ImageGeometryChange::kWarp,
          .preserves_precision = configuration.input_interpolation == InterpolationMode::kNearest &&
                                 (configuration.border_mode != BorderMode::kConstant ||
                                  detail::zero_border(configuration.border_value))});
  if (!metadata_updated) {
    return make_unexpected(std::move(metadata_updated).error());
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
  auto emit_options = sensor_io::derive_from(input_sample->metadata());
  // The map is configuration, but an invalid or degraded map also demotes the output image.
  emit_options.flags |= map_sample->metadata.flags;
  return output.emit_tensor(std::move(*loan), image, emit_options);
}

}  // namespace holoscan::vision
