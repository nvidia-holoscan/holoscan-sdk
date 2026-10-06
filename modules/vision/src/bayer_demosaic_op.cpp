// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>

#include <npp.h>

#include <dlpack/dlpack.h>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/image_descriptor_scratch.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/vision/bayer_demosaic_op.hpp>

#include "detail/cvcuda_tensor.hpp"
#include "detail/image_payload.hpp"

namespace holoscan::vision {
namespace {

constexpr TensorRepresentation kCudaTensorRepresentation{
    .memory_kind = MemoryKind::kCudaDevice,
};

[[nodiscard]] auto failure(ErrorCode code, std::string message) {
  return make_unexpected(Error{code, std::move(message)});
}

// NPP's own documentation requires every field of the "application managed stream context"
// contract to be populated; a partially-populated context (e.g. hStream alone) is undefined
// behavior for kernels that read the device attributes, even though it happens to work today.
[[nodiscard]] expected<NppStreamContext, Error> build_npp_stream_context(cudaStream_t stream) {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return failure(ErrorCode::kFailure, "BayerDemosaicOp failed to query the active CUDA device");
  }
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
    return failure(ErrorCode::kFailure, "BayerDemosaicOp failed to query CUDA device properties");
  }
  int compute_major = 0;
  int compute_minor = 0;
  unsigned int stream_flags = 0;
  if (cudaDeviceGetAttribute(&compute_major, cudaDevAttrComputeCapabilityMajor, device) !=
          cudaSuccess ||
      cudaDeviceGetAttribute(&compute_minor, cudaDevAttrComputeCapabilityMinor, device) !=
          cudaSuccess ||
      cudaStreamGetFlags(stream, &stream_flags) != cudaSuccess) {
    return failure(ErrorCode::kFailure, "BayerDemosaicOp failed to query CUDA stream attributes");
  }
  return NppStreamContext{
      .hStream = stream,
      .nCudaDeviceId = device,
      .nMultiProcessorCount = properties.multiProcessorCount,
      .nMaxThreadsPerMultiProcessor = properties.maxThreadsPerMultiProcessor,
      .nMaxThreadsPerBlock = properties.maxThreadsPerBlock,
      .nSharedMemPerBlock = properties.sharedMemPerBlock,
      .nCudaDevAttrComputeCapabilityMajor = compute_major,
      .nCudaDevAttrComputeCapabilityMinor = compute_minor,
      .nStreamFlags = stream_flags,
  };
}

}  // namespace

class BayerDemosaicOp::Impl {
 public:
  std::optional<sensor_io::ImageDescriptorScratch> image_metadata;
};

BayerDemosaicOp::BayerDemosaicOp() : impl_(std::make_unique<Impl>()) {}

BayerDemosaicOp::~BayerDemosaicOp() = default;

void BayerDemosaicOp::setup(OperatorSpec& spec) {
  spec.input(input, "input")
      .queue_depth(1U)
      .tensor_field(schema::ImageTensorFields::data)
      .expects_tensor(TensorInputSpec{
          .representation = kCudaTensorRepresentation,
          .bounds = TensorBounds{.max_rank = 3U},
      });
  spec.output(output, "output")
      .tensor_field(schema::ImageTensorFields::data)
      .produces_tensor(
          TensorOutputSpec{
              .representation = {.memory_kind = MemoryKind::kCudaDevice,
                                 .strides = TensorStridePolicy::kContiguousOnly},
              .bounds = TensorBounds{.max_rank = 3U},
              .storage = TensorOutputStorage::kRuntimePool,
          },
          // Output has up to 4 channels (RGBA) versus the single-channel Bayer input; setup()
          // runs before params() is available, so bound for the worst case (generate_alpha=true)
          // regardless of the configured value.
          input_max_byte_span(input).multiplied_by(tensor_byte_span(4)));
}

Contract BayerDemosaicOp::contract() const {
  Contract result;
  result.trigger(OnEach{input});
  return result;
}

void BayerDemosaicOp::start() {
  if (!impl_->image_metadata) {
    auto created = sensor_io::ImageDescriptorScratch::create();
    if (!created) {
      throw RuntimeError(created.error());
    }
    impl_->image_metadata.emplace(std::move(*created));
  }
}

void BayerDemosaicOp::stop() {
  // Host-side metadata scratch only; no CUDA resource to retain through quiescence.
}

expected<void, Error> BayerDemosaicOp::compute(ExecutionContext& context) {
  const BayerDemosaicOpParams& configuration = context.params<BayerDemosaicOpParams>();
  if (!impl_->image_metadata) {
    return failure(ErrorCode::kNotReady, "BayerDemosaicOp resources are not initialized");
  }
  const cudaStream_t stream = context.cuda_stream().get();
  if (stream == nullptr) {
    return failure(ErrorCode::kNotSupported, "BayerDemosaicOp requires a runtime CUDA stream");
  }

  auto sample = input.selected_sample();
  if (!sample) {
    return make_unexpected(std::move(sample).error());
  }
  auto input_view = detail::describe_image_payload(
      sample->data(), detail::TensorOperation::kBayerDemosaic, /*max_batch_size=*/1U);
  if (!input_view) {
    return make_unexpected(std::move(input_view).error());
  }
  auto metadata_copied = impl_->image_metadata->copy_metadata_from(sample->data());
  if (!metadata_copied) {
    return make_unexpected(std::move(metadata_copied).error());
  }
  auto& image = impl_->image_metadata->image();
  constexpr std::array kNamedPhases{
      std::pair{schema::ImageEncoding_BAYER_RGGB, BayerGridPosition::kRGGB},
      std::pair{schema::ImageEncoding_BAYER_BGGR, BayerGridPosition::kBGGR},
      std::pair{schema::ImageEncoding_BAYER_GRBG, BayerGridPosition::kGRBG},
      std::pair{schema::ImageEncoding_BAYER_GBRG, BayerGridPosition::kGBRG},
  };
  for (const auto& [encoding, phase] : kNamedPhases) {
    if (image.encoding == encoding && configuration.bayer_grid_pos != phase) {
      return failure(ErrorCode::kInvalidArgument,
                     "BayerDemosaicOp bayer_grid_pos must match the named Bayer image encoding");
    }
  }
  if (input_view->rank != 3U || input_view->channels != 1) {
    return failure(ErrorCode::kInvalidArgument,
                   "BayerDemosaicOp expects a single-channel (HWC, C=1) Bayer mosaic image");
  }
  if (input_view->height % 2 != 0 || input_view->width % 2 != 0) {
    return failure(ErrorCode::kInvalidArgument,
                   "BayerDemosaicOp requires even width and height; got " +
                       std::to_string(input_view->width) + "x" +
                       std::to_string(input_view->height));
  }
  // describe_image_payload() above already restricted dtype to unsigned 8-bit or 16-bit for
  // kBayerDemosaic (see dtype_supported() in detail/cvcuda_tensor.cpp), the two bit depths NPP
  // can be dispatched for and schema::ImageT can describe the output in (RGB8/RGBA8 or
  // RGB16/RGBA16).
  const bool is_16bit = input_view->dtype.bits == 16U;
  if (configuration.generate_alpha && !is_16bit && configuration.alpha_value > 255U) {
    return failure(ErrorCode::kInvalidArgument,
                   "BayerDemosaicOp alpha_value must be within 0-255 for 8-bit RGBA output");
  }
  if (input_view->byte_strides[1] != static_cast<std::int64_t>(input_view->dtype.bits / 8U)) {
    return failure(ErrorCode::kInvalidArgument, "BayerDemosaicOp requires packed C1 input samples");
  }
  if (input_view->byte_strides[0] > std::numeric_limits<int>::max()) {
    return failure(ErrorCode::kInvalidArgument, "BayerDemosaicOp: input row stride is too large");
  }
  const int in_step = static_cast<int>(input_view->byte_strides[0]);

  const std::int32_t out_channels = configuration.generate_alpha ? 4 : 3;
  const std::array<std::int64_t, 3> out_shape{input_view->height, input_view->width, out_channels};
  auto out_strides = detail::contiguous_byte_strides(out_shape, input_view->dtype);
  if (!out_strides) {
    return make_unexpected(std::move(out_strides).error());
  }
  const std::span<const std::int64_t> out_shape_span{out_shape.data(), out_shape.size()};
  const std::span<const std::int64_t> out_strides_span{out_strides->data(), out_shape.size()};

  auto loan = output.allocate_tensor(TensorLoanRequest{
      .shape = out_shape_span,
      .dtype = input_view->dtype,
      .byte_strides = TensorByteStridesSpan{out_strides_span},
  });
  if (!loan) {
    return make_unexpected(std::move(loan).error());
  }
  if (loan->device_id().value != input_view->device.device_id) {
    return failure(ErrorCode::kIncompatible,
                   "BayerDemosaicOp input and output tensors must use the same CUDA device");
  }
  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return make_unexpected(std::move(writer).error());
  }
  const int out_step = static_cast<int>((*out_strides)[0]);

  auto output_view = detail::describe_image_tensor(writer->data(),
                                                   out_shape_span,
                                                   out_strides_span,
                                                   input_view->dtype,
                                                   DLDevice{kDLCUDA, loan->device_id().value},
                                                   detail::TensorOperation::kBayerDemosaic,
                                                   /*max_batch_size=*/1U);
  if (!output_view) {
    return make_unexpected(std::move(output_view).error());
  }

  // update_image_metadata() has no operation-specific rule for a channel-count-changing
  // transform like demosaic; set the resulting encoding first so the generic geometry-preserving
  // path (significant_bits/color_space reset, ROI left alone) validates against it.
  image.encoding = is_16bit ? (configuration.generate_alpha ? schema::ImageEncoding_RGBA16
                                                            : schema::ImageEncoding_RGB16)
                            : (configuration.generate_alpha ? schema::ImageEncoding_RGBA8
                                                            : schema::ImageEncoding_RGB8);
  auto metadata_updated =
      detail::update_image_metadata(image,
                                    *input_view,
                                    *output_view,
                                    detail::ImageMetadataTransform{
                                        .geometry = detail::ImageGeometryChange::kPreserve,
                                        .values_changed = true,
                                        .preserves_precision = false,
                                    });
  if (!metadata_updated) {
    return make_unexpected(std::move(metadata_updated).error());
  }

  auto npp_stream_ctx = build_npp_stream_context(stream);
  if (!npp_stream_ctx) {
    return make_unexpected(std::move(npp_stream_ctx).error());
  }

  // NPP's Bayer demosaic kernels only support NPPI_INTER_UNDEFINED -- any other interpolation
  // mode is rejected by NPP itself, so this is not exposed as a configurable parameter.
  const NppiSize roi_size{static_cast<int>(input_view->width),
                          static_cast<int>(input_view->height)};
  const NppiRect roi_rect{
      0, 0, static_cast<int>(input_view->width), static_cast<int>(input_view->height)};
  const auto npp_bayer_grid_pos = static_cast<NppiBayerGridPosition>(configuration.bayer_grid_pos);

  // alpha_value is an unscaled widening cast into the output element width, matching how the
  // rest of this module (and schema::ImageT's own component conversions) treat integer storage:
  // 8-bit 255 becomes 16-bit 255, not a rescaled 65535, so a caller wanting full-scale 16-bit
  // alpha must say so explicitly.
  NppStatus npp_status = NPP_ERROR;
  if (is_16bit) {
    const auto* const in_ptr = static_cast<const Npp16u*>(input_view->data);
    auto* const out_ptr = static_cast<Npp16u*>(writer->data());
    npp_status = configuration.generate_alpha
                     ? nppiCFAToRGBA_16u_C1AC4R_Ctx(in_ptr,
                                                    in_step,
                                                    roi_size,
                                                    roi_rect,
                                                    out_ptr,
                                                    out_step,
                                                    npp_bayer_grid_pos,
                                                    NPPI_INTER_UNDEFINED,
                                                    static_cast<Npp16u>(configuration.alpha_value),
                                                    *npp_stream_ctx)
                     : nppiCFAToRGB_16u_C1C3R_Ctx(in_ptr,
                                                  in_step,
                                                  roi_size,
                                                  roi_rect,
                                                  out_ptr,
                                                  out_step,
                                                  npp_bayer_grid_pos,
                                                  NPPI_INTER_UNDEFINED,
                                                  *npp_stream_ctx);
  } else {
    const auto* const in_ptr = static_cast<const Npp8u*>(input_view->data);
    auto* const out_ptr = static_cast<Npp8u*>(writer->data());
    npp_status = configuration.generate_alpha
                     ? nppiCFAToRGBA_8u_C1AC4R_Ctx(in_ptr,
                                                   in_step,
                                                   roi_size,
                                                   roi_rect,
                                                   out_ptr,
                                                   out_step,
                                                   npp_bayer_grid_pos,
                                                   NPPI_INTER_UNDEFINED,
                                                   static_cast<Npp8u>(configuration.alpha_value),
                                                   *npp_stream_ctx)
                     : nppiCFAToRGB_8u_C1C3R_Ctx(in_ptr,
                                                 in_step,
                                                 roi_size,
                                                 roi_rect,
                                                 out_ptr,
                                                 out_step,
                                                 npp_bayer_grid_pos,
                                                 NPPI_INTER_UNDEFINED,
                                                 *npp_stream_ctx);
  }
  if (npp_status != NPP_SUCCESS) {
    return failure(ErrorCode::kFailure,
                   "BayerDemosaicOp NPP demosaic failed with status " +
                       std::to_string(static_cast<int>(npp_status)));
  }

  auto committed = std::move(*writer).commit();
  if (!committed) {
    return make_unexpected(std::move(committed).error());
  }
  return output.emit_tensor(std::move(*loan), image, sensor_io::derive_from(sample->metadata()));
}

}  // namespace holoscan::vision
