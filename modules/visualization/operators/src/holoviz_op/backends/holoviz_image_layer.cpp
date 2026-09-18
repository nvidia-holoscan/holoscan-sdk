// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "holoviz_op/backends/holoviz_image_layer.hpp"

#include <utility>
#include <vector>

#include <holoviz/slang_compiler.hpp>
#include "holoviz_op/types.hpp"

#include "cuda/cu_errors.hpp"
#include "holoviz_op/backends/holoviz_image_shader.hpp"
#include "holoviz_op/backends/layer_blend.hpp"

namespace holoscan::visualization::backends {

namespace {

using detail::kHolovizImageFlagGrayscaleReplicate;
using detail::kHolovizImageFlagSwizzleBGRA;
using detail::kHolovizImagePushBytes;
using detail::kHolovizImageQuadVertices;

struct ImagePush {
  float view_offset[2];
  float view_scale[2];
  float uv_scale[2];  // valid region (frame size / declared max)
  float uv_max[2];    // (frame extent - 0.5) / max — see holoviz_image_shader.cpp
  uint32_t flags;
  uint32_t pad;
};
static_assert(sizeof(ImagePush) == kHolovizImagePushBytes);

}  // namespace

expected<HolovizImageLayerState> allocate_holoviz_image_layer(holoviz::Device& device,
                                                              const ImageLayerDesc& desc,
                                                              uint32_t frames_in_flight) {
  if (desc.max_width == 0 || desc.max_height == 0) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] ImageLayerSpec needs nonzero max_width/max_height."});
  }

  HolovizImageLayerState state;
  state.desc = desc;

  holoviz::Format texture_format = holoviz::Format::kR8G8B8A8Unorm;
  switch (desc.format) {
    case ImageFormat::kR8G8B8A8Unorm:
      state.bytes_per_pixel = 4;
      texture_format = holoviz::Format::kR8G8B8A8Unorm;
      break;
    case ImageFormat::kR8Unorm:
      state.bytes_per_pixel = 1;
      texture_format = holoviz::Format::kR8Unorm;
      break;
    case ImageFormat::kR16Unorm:
      // Sampled as UNORM: 0..65535 lands as 0..1, replicated to gray by the shader flag.
      state.bytes_per_pixel = 2;
      texture_format = holoviz::Format::kR16Unorm;
      break;
    case ImageFormat::kR16G16B16A16Unorm:
      state.bytes_per_pixel = 8;
      texture_format = holoviz::Format::kR16G16B16A16Unorm;
      break;
    case ImageFormat::kNV12:
    case ImageFormat::kYUYV422:
      return make_unexpected(
          Error{.code = ErrorCode::kNotImplemented,
                .message = "[holoviz] YUV image formats (kNV12/kYUYV422) are deferred to a "
                           "later version; see holoviz_image_layer.hpp for the planned lowering."});
    case ImageFormat::kR8G8B8Unorm:
      return make_unexpected(
          Error{.code = ErrorCode::kNotImplemented,
                .message = "[holoviz] kR8G8B8Unorm needs the buffer-fetch decode path, which "
                           "is not implemented yet; supply kR8G8B8A8Unorm."});
    case ImageFormat::kB8G8R8A8Unorm:
      // The slot texture stays RGBA8: exported images need a CUDA array format and BGRA has
      // none (cu_enum_conversions.cpp). The upload is a byte blit, so BGRA bytes land unchanged
      // and the fragment shader swaps R and B via kHolovizImageFlagSwizzleBGRA.
      state.bytes_per_pixel = 4;
      texture_format = holoviz::Format::kR8G8B8A8Unorm;
      break;
    case ImageFormat::kUnknown:
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = "[holoviz] kUnknown is not a valid image layer format."});
  }

  // Reject rather than silently ignore declaration fields this backend does not act on
  // yet: accepting them would render with silently wrong color.
  if (desc.color_space != ColorSpace::kPassThrough) {
    return make_unexpected(
        Error{.code = ErrorCode::kNotImplemented,
              .message = "[holoviz] ImageLayerDesc::color_space conversion is not implemented "
                         "yet; declare ColorSpace::kPassThrough."});
  }
  if (desc.yuv_model.has_value() || desc.yuv_range.has_value()) {
    // Unreachable with a YUV format (those return kNotImplemented above); per the
    // holoviz_op.hpp contract these must be nullopt for non-YUV formats.
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] ImageLayerDesc::yuv_model/yuv_range must be nullopt for "
                         "non-YUV formats."});
  }

  state.slots.reserve(frames_in_flight);
  for (uint32_t i = 0; i < frames_in_flight; ++i) {
    HolovizImageLayerState::Slot slot;
    auto texture = device.create_image(holoviz::ImageDesc{
        .extent = {desc.max_width, desc.max_height},
        .format = texture_format,
        .usage = holoviz::ImageUsage::kSampled,
        .share = holoviz::ShareMode::kExported,
        .debug_name = "holoviz render backend image layer slot",
    });
    if (!texture) {
      return forward_error(texture);
    }
    slot.texture = std::move(*texture);
    VIZ_UNWRAP(CUarray, level_array, slot.texture.cuda_mipmapped_array_get_level(0));
    slot.array = level_array;
    state.slots.push_back(std::move(slot));
  }

  // The producer-ordering event (recorded on the producer's stream, waited on the backend
  // stream); DISABLE_TIMING keeps it a pure dependency edge.
  VIZ_CU_FORWARD_ERROR(cuEventCreate(&state.producer_ready, CU_EVENT_DISABLE_TIMING));

  return state;
}

expected<void> build_holoviz_image_layer(holoviz::Device& device, HolovizImageLayerState& state,
                                         uint32_t frames_in_flight) {
  VIZ_UNWRAP(holoviz::ShaderBlob,
             blob,
             holoviz::compile_slang_shader(holoviz::Backend::kVulkan,
                                           detail::kHolovizImageVs,
                                           detail::kHolovizImageFs,
                                           detail::kHolovizImageSlangSource));
  if (blob.push_constant_size != kHolovizImagePushBytes || !blob.attributes.empty()) {
    return make_unexpected(
        Error{.code = ErrorCode::kFailure,
              .message = "[holoviz] Image shader reflection disagrees with ImagePush / the "
                         "vertex-buffer-less contract."});
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

expected<void> record_holoviz_image_layer(HolovizImageLayerState& state, holoviz::CommandList& cmd,
                                          uint32_t slot, holoviz::SubmitSyncToken slot_last_use,
                                          const ImageFrameData& frame, CUstream stream,
                                          holoviz::ExternalSemaphore& semaphore,
                                          std::vector<holoviz::ExternalSyncToken>& waits) {
  if (frame.width == 0 || frame.height == 0 || frame.width > state.desc.max_width ||
      frame.height > state.desc.max_height) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message =
                                     "[holoviz] ImageFrameData size is zero or exceeds the layer's "
                                     "declared max extent — a mis-declaration."});
  }
  if (frame.plane_count != 1 || frame.planes[0].device_ptr == 0) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] This image format takes exactly one plane of device "
                         "memory."});
  }

  // Order: producer stream -> backend stream -> (after the slot's last GPU read) copy.
  if (frame.stream != nullptr && frame.stream != stream) {
    VIZ_CU_FORWARD_ERROR(cuEventRecord(state.producer_ready, frame.stream));
    VIZ_CU_FORWARD_ERROR(cuStreamWaitEvent(stream, state.producer_ready, 0));
  }
  VIZ_FORWARD_ERROR(semaphore.wait(slot_last_use));

  const size_t width_bytes = static_cast<size_t>(frame.width) * state.bytes_per_pixel;
  const CUDA_MEMCPY2D copy{
      .srcXInBytes = 0,
      .srcY = 0,
      .srcMemoryType = CU_MEMORYTYPE_DEVICE,
      .srcDevice = frame.planes[0].device_ptr,
      .srcPitch =
          frame.planes[0].row_pitch_bytes != 0 ? frame.planes[0].row_pitch_bytes : width_bytes,
      .dstXInBytes = 0,
      .dstY = 0,
      .dstMemoryType = CU_MEMORYTYPE_ARRAY,
      .dstArray = state.slots[slot].array,
      .WidthInBytes = width_bytes,
      .Height = frame.height,
  };
  VIZ_CU_FORWARD_ERROR(cuMemcpy2DAsync(&copy, stream));
  VIZ_UNWRAP(holoviz::ExternalSyncToken, upload_token, semaphore.signal());
  waits.push_back(upload_token);

  VIZ_FORWARD_ERROR(cmd.bind_pipeline(state.pipeline));
  VIZ_FORWARD_ERROR(cmd.bind_descriptor(state.per_slot_descriptors[slot]));

  const ViewTransform view = state.desc.view.value_or(ViewTransform{});
  const ImagePush push{
      {view.offset_x, view.offset_y},
      {view.scale_x, view.scale_y},
      {static_cast<float>(frame.width) / static_cast<float>(state.desc.max_width),
       static_cast<float>(frame.height) / static_cast<float>(state.desc.max_height)},
      {(static_cast<float>(frame.width) - 0.5F) / static_cast<float>(state.desc.max_width),
       (static_cast<float>(frame.height) - 0.5F) / static_cast<float>(state.desc.max_height)},
      ((state.desc.format == ImageFormat::kR8Unorm || state.desc.format == ImageFormat::kR16Unorm)
           ? kHolovizImageFlagGrayscaleReplicate
           : 0U) |
          (state.desc.format == ImageFormat::kB8G8R8A8Unorm ? kHolovizImageFlagSwizzleBGRA : 0U),
      0U,
  };
  VIZ_FORWARD_ERROR(cmd.set_push_data(
      state.pipeline,
      std::span<const std::byte>(reinterpret_cast<const std::byte*>(&push), sizeof(push))));

  return cmd.draw(kHolovizImageQuadVertices);
}

expected<void> destroy_holoviz_image_layer(HolovizImageLayerState& state,
                                           std::chrono::nanoseconds timeout) {
  expected<void> result;
  for (holoviz::Descriptor& descriptor : state.per_slot_descriptors) {
    if (expected<void> r = descriptor.destroy(timeout); !r && result) {
      result = r;
    }
  }
  state.per_slot_descriptors.clear();
  if (state.pipeline) {
    if (expected<void> r = state.pipeline.destroy(timeout); !r && result) {
      result = r;
    }
  }
  for (HolovizImageLayerState::Slot& slot : state.slots) {
    if (slot.texture) {
      if (expected<void> r = slot.texture.destroy(timeout); !r && result) {
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
