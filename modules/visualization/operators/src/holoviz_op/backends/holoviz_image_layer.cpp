// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "holoviz_op/backends/holoviz_image_layer.hpp"

#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <holoviz/adapter.hpp>
#include "holoviz_op/types.hpp"

#include "cuda/cu_errors.hpp"
#include "holoviz_op/backends/holoviz_image_interface.hpp"
#include "holoviz_op/backends/image_layout.hpp"
#include "holoviz_op/backends/layer_blend.hpp"
#include "holoviz_op/backends/rgb_to_rgba.hpp"

#include "backends/holoviz_image.slang.hpp"  // generated from holoviz_image.slang

namespace holoscan::visualization::backends {

namespace {

namespace shader = detail::holoviz_image;

// The push block and its flag bits come from holoviz_image_interface.hpp, the header the shader
// compiles from too; the generated holoviz_image.slang.hpp asserts the block's layout.

expected<ImageLayout> validate_image_frame(const ImageLayerDesc& desc,
                                           const ImageFrameData& frame) {
  if (frame.width == 0 || frame.height == 0 || frame.width > desc.max_width ||
      frame.height > desc.max_height) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message =
                                     "[holoviz] ImageFrameData size is zero or exceeds the layer's "
                                     "declared max extent — a mis-declaration."});
  }
  const auto layout = image_layout(desc.format, frame.width, frame.height);
  if (!layout || frame.plane_count != layout->plane_count) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message = "[holoviz] Invalid image extent or plane count."});
  }
  for (uint32_t plane = 0; plane < layout->plane_count; ++plane) {
    const auto& input = frame.planes[plane];
    const size_t pitch =
        input.row_pitch_bytes != 0 ? input.row_pitch_bytes : layout->row_bytes[plane];
    const auto span = image_plane_span(layout->row_bytes[plane], layout->rows[plane], pitch);
    if (input.device_ptr == 0 || !span ||
        input.device_ptr > std::numeric_limits<CUdeviceptr>::max() - (*span - 1)) {
      return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                   .message = "[holoviz] Invalid image plane pointer or pitch."});
    }
  }

  return *layout;
}

}  // namespace

// Keep format selection and dependent per-slot resource creation together for failure cleanup.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<HolovizImageLayerState> allocate_holoviz_image_layer(holoviz::Device& device,
                                                              const ImageLayerDesc& desc,
                                                              uint32_t frames_in_flight) {
  const auto layout = image_layout(desc.format, desc.max_width, desc.max_height);
  if (!layout) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message = "[holoviz] Invalid image format or maximum extent (YUV "
                                            "requires even subsampled dimensions)."});
  }

  HolovizImageLayerState state;
  state.desc = desc;

  state.slot_bytes = layout->bytes;
  holoviz::Format texture_format = holoviz::Format::kR8G8B8A8Unorm;
  switch (desc.format) {
    case ImageFormat::kR8G8B8A8Unorm:
      texture_format = holoviz::Format::kR8G8B8A8Unorm;
      break;
    case ImageFormat::kR8Unorm:
      texture_format = holoviz::Format::kR8Unorm;
      break;
    case ImageFormat::kR16Unorm:
      // Sampled as UNORM: 0..65535 lands as 0..1, replicated to gray by the shader flag.
      texture_format = holoviz::Format::kR16Unorm;
      break;
    case ImageFormat::kR16G16B16A16Unorm:
      texture_format = holoviz::Format::kR16G16B16A16Unorm;
      break;
    case ImageFormat::kNV12:
      texture_format = holoviz::Format::kR8Unorm;
      break;
    case ImageFormat::kYUYV422:
      break;
    case ImageFormat::kR8G8B8Unorm:
      if (desc.max_width > std::numeric_limits<size_t>::max() / 4 / desc.max_height) {
        return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                     .message = "[holoviz] RGBA image allocation overflows."});
      }
      state.slot_bytes = static_cast<size_t>(desc.max_width) * desc.max_height * 4;
      break;
    case ImageFormat::kB8G8R8A8Unorm:
      // The slot texture stays RGBA8: exported images need a CUDA array format and BGRA has
      // none (cu_enum_conversions.cpp). The upload is a byte blit, so BGRA bytes land unchanged
      // and the fragment shader swaps R and B via kImageFlagSwizzleBGRA.
      texture_format = holoviz::Format::kR8G8B8A8Unorm;
      break;
    case ImageFormat::kUnknown:
    case ImageFormat::kR8G8B8A8Srgb:
    case ImageFormat::kB8G8R8A8Srgb:
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = "[holoviz] Invalid image layer format; sRGB sources use an "
                           "8-bit RGBA/BGRA UNORM format with ColorSpace::kSRGBNonlinear."});
    default:
      return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                   .message = "[holoviz] Unknown image layer format."});
  }

  // Native sRGB sampling decodes RGB BEFORE filtering. CUDA copies the encoded bytes
  // unchanged into the RGBA8 array; BGRA sources retain the existing shader swizzle.
  // Alpha is never transfer-encoded. Output encoding is independent, at the attachment.
  if (desc.color_space == ColorSpace::kSRGBNonlinear) {
    if (desc.format != ImageFormat::kR8G8B8A8Unorm && desc.format != ImageFormat::kB8G8R8A8Unorm) {
      return make_unexpected(Error{.code = ErrorCode::kNotImplemented,
                                   .message = "[holoviz] sRGB input decoding currently requires "
                                              "8-bit RGBA or BGRA; supply linear data otherwise."});
    }
    texture_format = holoviz::Format::kR8G8B8A8Srgb;
  }
  if (desc.color_space != ColorSpace::kPassThrough &&
      desc.color_space != ColorSpace::kSRGBNonlinear &&
      desc.color_space != ColorSpace::kBT709Linear) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message = "[holoviz] Unknown ImageLayerDesc::color_space."});
  }
  if (is_yuv(desc.format)) {
    if (!desc.yuv_model || !desc.yuv_range ||
        (*desc.yuv_model != YUVModelConversion::kBT601 &&
         *desc.yuv_model != YUVModelConversion::kBT709 &&
         *desc.yuv_model != YUVModelConversion::kBT2020) ||
        (*desc.yuv_range != YUVRange::kLimited && *desc.yuv_range != YUVRange::kFull)) {
      return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                   .message = "[holoviz] YUV requires a valid model and range."});
    }
  } else if (desc.yuv_model || desc.yuv_range) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] Non-YUV images cannot specify YUV conversion."});
  }

  VIZ_UNWRAP(holoviz::CapabilityReport, caps, device.capabilities());
  const auto supported = caps.format_capabilities(texture_format);
  if (!holoviz::has_flag(supported, holoviz::FormatCapability::kSampled) ||
      !holoviz::has_flag(supported, holoviz::FormatCapability::kCudaImportableImage)) {
    return make_unexpected(Error{.code = ErrorCode::kNotImplemented,
                                 .message = "[holoviz] Image layer format cannot be sampled "
                                            "with CUDA interop on this device."});
  }

  state.slots.reserve(frames_in_flight);
  for (uint32_t i = 0; i < frames_in_flight; ++i) {
    HolovizImageLayerState::Slot slot;
    auto texture = device.create_image(holoviz::ImageDesc{
        .extent = {desc.format == ImageFormat::kYUYV422 ? desc.max_width / 2 : desc.max_width,
                   desc.max_height},
        .format = texture_format,
        .usage = holoviz::ImageUsage::kSampled,
        .share = holoviz::ShareMode::kExported,
        .debug_name = "holoviz render backend image layer slot",
        .cuda_surface_load_store = desc.format == ImageFormat::kR8G8B8Unorm,
    });
    if (!texture) {
      return forward_error(texture);
    }
    slot.texture = std::move(*texture);
    VIZ_UNWRAP(CUarray, level_array, slot.texture.cuda_mipmapped_array_get_level(0));
    slot.array = level_array;
    if (desc.format == ImageFormat::kR8G8B8Unorm) {
      CUDA_RESOURCE_DESC resource{};
      resource.resType = CU_RESOURCE_TYPE_ARRAY;
      resource.res.array.hArray = slot.array;
      VIZ_CU_FORWARD_ERROR(cuSurfObjectCreate(&slot.surface.handle, &resource));
    }
    if (desc.format == ImageFormat::kNV12) {
      VIZ_UNWRAP(holoviz::Image,
                 chroma,
                 device.create_image(holoviz::ImageDesc{
                     .extent = {desc.max_width / 2, desc.max_height / 2},
                     .format = holoviz::Format::kR8G8Unorm,
                     .usage = holoviz::ImageUsage::kSampled,
                     .share = holoviz::ShareMode::kExported,
                     .debug_name = "holoviz render backend chroma slot",
                 }));
      slot.chroma_texture = std::move(chroma);
      VIZ_UNWRAP(CUarray, chroma_array, slot.chroma_texture.cuda_mipmapped_array_get_level(0));
      slot.chroma_array = chroma_array;
    }
    state.slots.push_back(std::move(slot));
  }

  // The producer-ordering event (recorded on the producer's stream, waited on the backend
  // stream); DISABLE_TIMING keeps it a pure dependency edge.
  VIZ_CU_FORWARD_ERROR(cuEventCreate(&state.producer_ready, CU_EVENT_DISABLE_TIMING));

  return state;
}

expected<void> build_holoviz_image_layer(holoviz::Device& device, HolovizImageLayerState& state,
                                         uint32_t frames_in_flight) {
  holoviz::ShaderBlob blob = shader::make_shader_blob();
  if (!blob.attributes.empty()) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Image shader unexpectedly reflected vertex attributes; it "
                         "must stay vertex-buffer-less."});
  }

  auto pipeline = device.create_pipeline(holoviz::PipelineDesc{
      .shader_blob = std::move(blob),
      .topology = holoviz::PrimitiveTopology::kTriangleList,
      .blend = kStraightAlphaLayerBlend,
      .debug_name = "holoviz render backend image layer",
  });
  if (!pipeline) {
    return forward_error(pipeline);
  }
  state.pipeline = std::move(*pipeline);

  state.per_slot_descriptors.reserve(frames_in_flight);
  for (uint32_t slot = 0; slot < frames_in_flight; ++slot) {
    VIZ_UNWRAP(holoviz::Descriptor, descriptor, state.pipeline.create_descriptor());
    VIZ_FORWARD_ERROR(descriptor.bind_image(state.slots[slot].texture, "sourceTexture"));
    VIZ_FORWARD_ERROR(descriptor.bind_image(state.desc.format == ImageFormat::kNV12
                                                ? state.slots[slot].chroma_texture
                                                : state.slots[slot].texture,
                                            "chromaTexture"));
    VIZ_FORWARD_ERROR(
        descriptor.bind_sampler(device,
                                holoviz::SamplerDesc{},  // linear, clamp-to-edge defaults
                                "sourceSampler"));
    state.per_slot_descriptors.push_back(std::move(descriptor));
  }
  return {};
}

expected<void> warm_holoviz_image_layer(HolovizImageLayerState& state, holoviz::CommandList& cmd) {
  return cmd.bind_pipeline(state.pipeline);
}

// Keep producer ordering, upload, and draw submission in GPU execution order.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void> record_holoviz_image_layer(HolovizImageLayerState& state, holoviz::CommandList& cmd,
                                          uint32_t slot, holoviz::SubmitSyncToken slot_last_use,
                                          const ImageFrameData& frame, CUstream stream,
                                          holoviz::ExternalSemaphore& semaphore,
                                          std::vector<holoviz::ExternalSyncToken>& waits) {
  const auto layout = validate_image_frame(state.desc, frame);
  if (!layout) {
    return make_unexpected(layout.error());
  }

  // Order: producer stream -> backend stream -> (after the slot's last GPU read) copy.
  if (frame.stream != nullptr && frame.stream != stream) {
    VIZ_CU_FORWARD_ERROR(cuEventRecord(state.producer_ready, frame.stream));
    VIZ_CU_FORWARD_ERROR(cuStreamWaitEvent(stream, state.producer_ready, 0));
  }
  VIZ_FORWARD_ERROR(semaphore.wait(slot_last_use));

  if (state.desc.format == ImageFormat::kR8G8B8Unorm) {
    const auto error =
        launch_rgb_to_rgba(frame.planes[0].device_ptr,
                           (frame.planes[0].row_pitch_bytes) != 0 ? frame.planes[0].row_pitch_bytes
                                                                  : layout->row_bytes[0],
                           frame.width,
                           frame.height,
                           state.slots[slot].surface.handle,
                           stream);
    if (error != cudaSuccess) {
      return make_unexpected(Error{
          .code = ErrorCode::kFailure,
          .message =
              std::string("[holoviz] RGB expansion launch failed: ") + cudaGetErrorString(error)});
    }
  } else {
    for (uint32_t plane = 0; plane < layout->plane_count; ++plane) {
      const auto& input = frame.planes[plane];
      const CUDA_MEMCPY2D copy{
          .srcXInBytes = 0,
          .srcY = 0,
          .srcMemoryType = CU_MEMORYTYPE_DEVICE,
          .srcDevice = input.device_ptr,
          .srcPitch = input.row_pitch_bytes != 0 ? input.row_pitch_bytes : layout->row_bytes[plane],
          .dstXInBytes = 0,
          .dstY = 0,
          .dstMemoryType = CU_MEMORYTYPE_ARRAY,
          .dstArray = plane == 0 ? state.slots[slot].array : state.slots[slot].chroma_array,
          .WidthInBytes = layout->row_bytes[plane],
          .Height = layout->rows[plane],
      };
      VIZ_CU_FORWARD_ERROR(cuMemcpy2DAsync(&copy, stream));
    }
  }
  VIZ_UNWRAP(holoviz::ExternalSyncToken, upload_token, semaphore.signal());
  waits.push_back(upload_token);

  VIZ_FORWARD_ERROR(cmd.bind_pipeline(state.pipeline));
  VIZ_FORWARD_ERROR(cmd.bind_descriptor(state.per_slot_descriptors[slot]));

  const ViewTransform view = state.desc.view.value_or(ViewTransform{});
  shader::PushConstants push{
      shader::float2(view.offset_x, view.offset_y),
      shader::float2(view.scale_x, view.scale_y),
      shader::float2(static_cast<float>(frame.width) / static_cast<float>(state.desc.max_width),
                     static_cast<float>(frame.height) / static_cast<float>(state.desc.max_height)),
      shader::float2(
          (static_cast<float>(frame.width) - 0.5F) / static_cast<float>(state.desc.max_width),
          (static_cast<float>(frame.height) - 0.5F) / static_cast<float>(state.desc.max_height)),
      ((state.desc.format == ImageFormat::kR8Unorm || state.desc.format == ImageFormat::kR16Unorm)
           ? shader::kImageFlagGrayscaleReplicate
           : 0U) |
          (state.desc.format == ImageFormat::kB8G8R8A8Unorm ? shader::kImageFlagSwizzleBGRA : 0U) |
          (state.desc.format == ImageFormat::kNV12 ? shader::kImageFlagNV12 : 0U) |
          (state.desc.format == ImageFormat::kYUYV422 ? shader::kImageFlagYUYV422 : 0U),
      0U,
  };
  push.frame_size[0] = static_cast<float>(frame.width);
  push.frame_size[1] = static_cast<float>(frame.height);
  push.max_size[0] = static_cast<float>(state.desc.max_width);
  push.max_size[1] = static_cast<float>(state.desc.max_height);
  if (is_yuv(state.desc.format)) {
    // allocate_holoviz_image_layer() admits YUV layers only with a valid model and range.
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const YUVModelConversion yuv_model = *state.desc.yuv_model;
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    const YUVRange yuv_range = *state.desc.yuv_range;
    float kr = 0.299F;
    float kb = 0.114F;
    if (yuv_model == YUVModelConversion::kBT709) {
      kr = 0.2126F;
      kb = 0.0722F;
    } else if (yuv_model == YUVModelConversion::kBT2020) {
      kr = 0.2627F;
      kb = 0.0593F;
    }
    const float kg = 1.0F - kr - kb;
    push.yuv_coefficients[0] = 2.0F * (1.0F - kr);
    push.yuv_coefficients[1] = 2.0F * (1.0F - kb);
    push.yuv_coefficients[2] = -2.0F * kb * (1.0F - kb) / kg;
    push.yuv_coefficients[3] = -2.0F * kr * (1.0F - kr) / kg;
    const bool limited = yuv_range == YUVRange::kLimited;
    push.yuv_range[0] = limited ? 16.0F / 255.0F : 0.0F;
    push.yuv_range[1] = limited ? 255.0F / 219.0F : 1.0F;
    push.yuv_range[2] = 128.0F / 255.0F;
    push.yuv_range[3] = limited ? 255.0F / 224.0F : 1.0F;
  }
  VIZ_FORWARD_ERROR(cmd.set_push_data(
      state.pipeline,
      std::span<const std::byte>(reinterpret_cast<const std::byte*>(&push), sizeof(push))));

  return cmd.draw(shader::kImageQuadVertices);
}

// Keep best-effort release of each dependent slot resource together, preserving the first error.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void> destroy_holoviz_image_layer(HolovizImageLayerState& state,
                                           std::chrono::nanoseconds timeout) {
  expected<void> result;
  for (holoviz::Descriptor& descriptor : state.per_slot_descriptors) {
    if (const expected<void> r = descriptor.destroy(timeout); !r && result) {
      result = r;
    }
  }
  state.per_slot_descriptors.clear();
  if (state.pipeline) {
    if (const expected<void> r = state.pipeline.destroy(timeout); !r && result) {
      result = r;
    }
  }
  for (HolovizImageLayerState::Slot& slot : state.slots) {
    if (slot.surface.handle != 0) {
      const auto error = cuSurfObjectDestroy(slot.surface.handle);
      if (error != CUDA_SUCCESS) {
        if (result) {
          result = make_unexpected(Error{.code = ErrorCode::kFailure,
                                         .message = "[holoviz] CUDA surface destruction failed."});
        }
      } else {
        slot.surface.handle = 0;
      }
    }
    if (slot.chroma_texture) {
      if (const expected<void> r = slot.chroma_texture.destroy(timeout); !r && result) {
        result = r;
      }
    }
    if (slot.texture) {
      if (const expected<void> r = slot.texture.destroy(timeout); !r && result) {
        result = r;
      }
    }
  }
  state.slots.clear();
  if (state.producer_ready != nullptr) {
    cuEventDestroy(state.producer_ready);
    state.producer_ready = nullptr;
  }
  return result;
}

}  // namespace holoscan::visualization::backends
