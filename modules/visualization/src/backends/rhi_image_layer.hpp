// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The image layer of RhiRenderBackend: GPU-resident pixel frames (CUdeviceptr + pitch)
// async-copied into a per-slot exported texture and drawn as one view-transformed quad.
//
// Initial format scope: kR8G8B8A8Unorm and kR8Unorm (sampled directly; R8 replicates to
// gray). Deferred, failing allocate() with kNotImplemented until their stages land:
//  - kNv12 / kYuyv422 — planned lowering: Y plane (kR8Unorm image) + UV plane (a future
//    rhi::Format::kR8G8Unorm image) with a model/range conversion matrix, and the packed
//    422 trick (RGBA8 image of width/2) respectively.
//  - kR8G8B8Unorm — no 3-byte texel format in rhi::Format; planned as an exported storage
//    buffer with byte-fetch decoding in the fragment shader.

#pragma once

#include <chrono>
#include <vector>

#include <cuda.h>

#include <holoviz/expected.hpp>
#include <holoviz/frame_data.hpp>
#include <holoviz/layer_desc.hpp>
#include <holoviz/rhi/command_list.hpp>
#include <holoviz/rhi/common.hpp>
#include <holoviz/rhi/descriptor.hpp>
#include <holoviz/rhi/device.hpp>
#include <holoviz/rhi/image.hpp>
#include <holoviz/rhi/pipeline.hpp>

namespace holoviz::backends {

struct RhiImageLayerState {
  ImageLayerDesc desc;
  uint32_t bytes_per_pixel = 0;

  struct Slot {
    rhi::Image texture;       ///< kExported | kSampled, declared max extent
    CUarray array = nullptr;  ///< mip 0 of the texture, for the per-frame copy
  };
  std::vector<Slot> slots;

  /// Orders the backend's copy after the producer's stream (recorded on the producer,
  /// waited on the backend stream); created once, reused every frame.
  CUevent producer_ready = nullptr;

  // build()-time:
  rhi::Pipeline pipeline;
  std::vector<rhi::Descriptor> per_slot_descriptors;
};

[[nodiscard]] expected<RhiImageLayerState> allocate_rhi_image_layer(rhi::Device& device,
                                                                    const ImageLayerDesc& desc,
                                                                    uint32_t frames_in_flight);

[[nodiscard]] expected<void> build_rhi_image_layer(rhi::Device& device, RhiImageLayerState& state,
                                                   uint32_t frames_in_flight);

[[nodiscard]] expected<void> warm_rhi_image_layer(RhiImageLayerState& state, rhi::CommandList& cmd);

/// Record one frame: order the copy after the producer stream (if any) and the slot's last
/// GPU use, land the pixels, draw the quad. The caller keeps the device's CUDA context
/// current and collects `waits` into the frame's submit.
[[nodiscard]] expected<void> record_rhi_image_layer(RhiImageLayerState& state,
                                                    rhi::CommandList& cmd, uint32_t slot,
                                                    rhi::SubmitSyncToken slot_last_use,
                                                    const ImageFrameData& frame, CUstream stream,
                                                    rhi::ExternalSemaphore& semaphore,
                                                    std::vector<rhi::ExternalSyncToken>& waits);

[[nodiscard]] expected<void> destroy_rhi_image_layer(RhiImageLayerState& state,
                                                     std::chrono::nanoseconds timeout);

}  // namespace holoviz::backends
