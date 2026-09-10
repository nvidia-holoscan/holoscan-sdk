// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "backends/rhi_lut_layer.hpp"

#include <algorithm>
#include <cstring>
#include <utility>
#include <vector>

#include <holoviz/rhi/adapter.hpp>
#include <holoviz/rhi/slang_compiler.hpp>

#include "backends/layer_blend.hpp"
#include "backends/rhi_lut_shader.hpp"
#include "holoviz/layer_desc.hpp"
#include "rhi/cuda/cu_errors.hpp"

namespace holoviz::backends {

namespace {

using detail::kRhiLutPushBytes;
using detail::kRhiLutQuadVertices;

struct LutPush {
  float view_offset[2];
  float view_scale[2];
  float uv_scale[2];  // valid region (frame size / declared max)
  float uv_max[2];    // (frame extent - 0.5) / max — mirrors rhi_image_shader.cpp
  uint32_t lut_size;
  uint32_t pad;
};
static_assert(sizeof(LutPush) == kRhiLutPushBytes);

}  // namespace

expected<RhiLUTLayerState> allocate_rhi_lut_layer(rhi::Device& device, const LUTLayerDesc& desc,
                                                  uint32_t frames_in_flight) {
  if (desc.lut_size == 0 || desc.lut_size > 256 || desc.max_width == 0 || desc.max_height == 0) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] LutLayerSpec needs lut_size in [1, 256] (a kR8Unorm mask "
                         "cannot index further) and nonzero max_width/max_height."});
  }

  RhiLUTLayerState state;
  state.desc = desc;

  state.slots.reserve(frames_in_flight);
  for (uint32_t i = 0; i < frames_in_flight; ++i) {
    RhiLUTLayerState::Slot slot;
    auto mask = device.create_image(rhi::ImageDesc{
        .extent = {desc.max_width, desc.max_height},
        .format = rhi::Format::kR8Unorm,
        .usage = rhi::ImageUsage::kSampled,
        .share = rhi::ShareMode::kExported,
        .debug_name = "rhi render backend lut mask slot",
    });
    if (!mask) {
      return forward_error(mask);
    }
    slot.mask = std::move(*mask);
    VIZ_UNWRAP(CUarray, level_array, slot.mask.cuda_mipmapped_array_get_level(0));
    slot.array = level_array;
    state.slots.push_back(std::move(slot));
  }

  VIZ_UNWRAP(rhi::CapabilityReport, caps, device.capabilities());
  const size_t palette_bytes = static_cast<size_t>(desc.lut_size) * 4 * sizeof(float);
  const size_t alignment = std::max<size_t>(caps.storage_buffer_alignment, sizeof(float));
  state.palette_stride_bytes = (palette_bytes + alignment - 1) / alignment * alignment;

  auto pool = device.create_buffer(rhi::BufferDesc{
      .size_bytes = state.palette_stride_bytes * frames_in_flight,
      .share = rhi::ShareMode::kExported,
      .debug_name = "rhi render backend lut palette pool",
  });
  if (!pool) {
    return forward_error(pool);
  }
  state.palette_pool = std::move(*pool);
  VIZ_UNWRAP(CUdeviceptr, palette_ptr, state.palette_pool.cuda_device_ptr());
  state.palette_ptr = palette_ptr;

  // One pinned staging region per frame slot (aligned stride, matching the device pool's
  // indexing); see the header for why staging must be per-slot.
  void* staging = nullptr;
  VIZ_CU_FORWARD_ERROR(cuMemHostAlloc(&staging, state.palette_stride_bytes * frames_in_flight, 0));
  state.palette_staging = static_cast<float*>(staging);
  state.palette_slot_floats = state.palette_stride_bytes / sizeof(float);
  state.palette_host.assign(static_cast<size_t>(desc.lut_size) * 4, 0.0F);

  VIZ_CU_FORWARD_ERROR(cuEventCreate(&state.producer_ready, CU_EVENT_DISABLE_TIMING));

  return state;
}

expected<void> build_rhi_lut_layer(rhi::Device& device, RhiLUTLayerState& state,
                                   uint32_t frames_in_flight) {
  VIZ_UNWRAP(
      rhi::ShaderBlob,
      blob,
      rhi::compile_slang_shader(
          rhi::Backend::kVulkan, detail::kRhiLutVs, detail::kRhiLutFs, detail::kRhiLutSlangSource));
  if (blob.push_constant_size != kRhiLutPushBytes || !blob.attributes.empty()) {
    return make_unexpected(Error{.code = ErrorCode::kFailure,
                                 .message =
                                     "[holoviz] LUT shader reflection disagrees with LutPush / the "
                                     "vertex-buffer-less contract."});
  }

  auto pipeline = device.create_pipeline(rhi::PipelineDesc{
      .shader_blob = std::move(blob),
      .topology = rhi::PrimitiveTopology::kTriangleList,
      .blend = kStraightAlphaLayerBlend,
      .debug_name = "rhi render backend lut layer",
  });
  if (!pipeline) {
    return forward_error(pipeline);
  }
  state.pipeline = std::move(*pipeline);

  state.per_slot_descriptors.reserve(frames_in_flight);
  for (uint32_t slot = 0; slot < frames_in_flight; ++slot) {
    VIZ_UNWRAP(rhi::Descriptor, descriptor, state.pipeline.create_descriptor());
    VIZ_FORWARD_ERROR(descriptor.bind_image(state.slots[slot].mask, "maskTexture"));
    // Nearest sampling is load-bearing: mask indices must never interpolate.
    VIZ_FORWARD_ERROR(descriptor.bind_sampler(device,
                                              rhi::SamplerDesc{
                                                  .min_filter = rhi::FilterMode::kNearest,
                                                  .mag_filter = rhi::FilterMode::kNearest,
                                              },
                                              "maskSampler"));
    const auto palette_slice =
        state.palette_pool.slice(slot * state.palette_stride_bytes, state.palette_stride_bytes);
    if (!palette_slice) {
      return forward_error(palette_slice);
    }
    VIZ_FORWARD_ERROR(descriptor.bind_buffer(*palette_slice, "lutTable"));
    state.per_slot_descriptors.push_back(std::move(descriptor));
  }
  return {};
}

expected<void> warm_rhi_lut_layer(RhiLUTLayerState& state, rhi::CommandList& cmd) {
  return cmd.bind_pipeline(state.pipeline);
}

expected<void> record_rhi_lut_layer(RhiLUTLayerState& state, rhi::CommandList& cmd, uint32_t slot,
                                    rhi::SubmitSyncToken slot_last_use, const LutFrameData& frame,
                                    CUstream stream, rhi::ExternalSemaphore& semaphore,
                                    std::vector<rhi::ExternalSyncToken>& waits) {
  if (frame.width == 0 || frame.height == 0 || frame.width > state.desc.max_width ||
      frame.height > state.desc.max_height || frame.mask.device_ptr == 0) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] LutFrameData mask is missing or exceeds the layer's "
                         "declared max extent — a mis-declaration."});
  }
  if (!frame.lut.empty() && frame.lut.size() != state.palette_host.size()) {
    return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                 .message =
                                     "[holoviz] LutFrameData::lut must supply exactly lut_size * 4 "
                                     "floats (or be empty to keep the previous palette)."});
  }
  if (frame.lut.empty() && !state.palette_ever_set) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidState,
              .message = "[holoviz] The first LUT frame must supply a palette."});
  }

  // The host copy is authoritative; re-staging it into the current slot each frame is what
  // makes "empty lut = keep previous" exact across frame slots.
  if (!frame.lut.empty()) {
    std::memcpy(state.palette_host.data(), frame.lut.data(), frame.lut.size() * sizeof(float));
    state.palette_ever_set = true;
  }
  float* slot_staging =
      state.palette_staging + static_cast<size_t>(slot) * state.palette_slot_floats;
  std::memcpy(slot_staging, state.palette_host.data(), state.palette_host.size() * sizeof(float));

  if (frame.stream != nullptr && frame.stream != stream) {
    VIZ_CU_FORWARD_ERROR(cuEventRecord(state.producer_ready, frame.stream));
    VIZ_CU_FORWARD_ERROR(cuStreamWaitEvent(stream, state.producer_ready, 0));
  }
  // Orders the device-side copies after the GPU's last read of this slot. Does NOT
  // protect staging reuse (the host already passed this value in begin_frame()'s
  // reset()); staging safety comes from the per-slot staging regions.
  VIZ_FORWARD_ERROR(semaphore.wait(slot_last_use));

  const CUDA_MEMCPY2D mask_copy{
      .srcXInBytes = 0,
      .srcY = 0,
      .srcMemoryType = CU_MEMORYTYPE_DEVICE,
      .srcDevice = frame.mask.device_ptr,
      .srcPitch = frame.mask.row_pitch_bytes != 0 ? frame.mask.row_pitch_bytes : frame.width,
      .dstXInBytes = 0,
      .dstY = 0,
      .dstMemoryType = CU_MEMORYTYPE_ARRAY,
      .dstArray = state.slots[slot].array,
      .WidthInBytes = frame.width,
      .Height = frame.height,
  };
  VIZ_CU_FORWARD_ERROR(cuMemcpy2DAsync(&mask_copy, stream));
  VIZ_CU_FORWARD_ERROR(cuMemcpyHtoDAsync(state.palette_ptr + slot * state.palette_stride_bytes,
                                         slot_staging,
                                         state.palette_host.size() * sizeof(float),
                                         stream));
  VIZ_UNWRAP(rhi::ExternalSyncToken, upload_token, semaphore.signal());
  waits.push_back(upload_token);

  VIZ_FORWARD_ERROR(cmd.bind_pipeline(state.pipeline));
  VIZ_FORWARD_ERROR(cmd.bind_descriptor(state.per_slot_descriptors[slot]));

  const LutPush push{
      {0.0F, 0.0F},
      {1.0F, 1.0F},
      {static_cast<float>(frame.width) / static_cast<float>(state.desc.max_width),
       static_cast<float>(frame.height) / static_cast<float>(state.desc.max_height)},
      {(static_cast<float>(frame.width) - 0.5F) / static_cast<float>(state.desc.max_width),
       (static_cast<float>(frame.height) - 0.5F) / static_cast<float>(state.desc.max_height)},
      state.desc.lut_size,
      0U,
  };
  VIZ_FORWARD_ERROR(cmd.set_push_data(
      state.pipeline,
      std::span<const std::byte>(reinterpret_cast<const std::byte*>(&push), sizeof(push))));

  return cmd.draw(kRhiLutQuadVertices);
}

expected<void> destroy_rhi_lut_layer(RhiLUTLayerState& state, std::chrono::nanoseconds timeout) {
  expected<void> result;
  for (rhi::Descriptor& descriptor : state.per_slot_descriptors) {
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
  for (RhiLUTLayerState::Slot& slot : state.slots) {
    if (slot.mask) {
      if (expected<void> r = slot.mask.destroy(timeout); !r && result) {
        result = r;
      }
    }
  }
  state.slots.clear();
  if (state.palette_pool) {
    if (expected<void> r = state.palette_pool.destroy(timeout); !r && result) {
      result = r;
    }
  }
  if (state.palette_staging != nullptr) {
    cuMemFreeHost(state.palette_staging);
    state.palette_staging = nullptr;
  }
  if (state.producer_ready != nullptr) {
    cuEventDestroy(state.producer_ready);
    state.producer_ready = nullptr;
  }
  return result;
}

}  // namespace holoviz::backends
