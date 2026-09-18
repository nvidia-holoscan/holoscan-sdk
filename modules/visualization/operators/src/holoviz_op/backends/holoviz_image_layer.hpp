// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The image layer of HolovizRenderBackend: GPU-resident pixel frames (CUdeviceptr + pitch)
// async-copied into a per-slot exported texture and drawn as one view-transformed quad.
//
// Supported formats: kR8G8B8A8Unorm, kB8G8R8A8Unorm, kR8Unorm, kR16Unorm, and
// kR16G16B16A16Unorm, each sampled from a slot texture of the matching holoviz::Format
// (BGRA reuses RGBA8 and swizzles in the shader; R8/R16 replicate to gray).
// Deferred, failing allocate() with kNotImplemented until their stages land:
//  - kNv12 / kYuyv422 — planned lowering: Y plane (kR8Unorm image) + UV plane (a future
//    holoviz::Format::kR8G8Unorm image) with a model/range conversion matrix, and the packed
//    422 trick (RGBA8 image of width/2) respectively.
//  - kR8G8B8Unorm — no 3-byte texel format in holoviz::Format; planned as an exported storage
//    buffer with byte-fetch decoding in the fragment shader.

#pragma once

#include <chrono>
#include <vector>

#include <cuda.h>

#include <holoscan/visualization/holoviz_op.hpp>
#include <holoviz/command_list.hpp>
#include <holoviz/common.hpp>
#include <holoviz/descriptor.hpp>
#include <holoviz/device.hpp>
#include <holoviz/expected.hpp>
#include <holoviz/image.hpp>
#include <holoviz/pipeline.hpp>
#include "holoviz_op/backends/frame_data.hpp"
#include "holoviz_op/types.hpp"

namespace holoscan::visualization::backends {

struct HolovizImageLayerState {
  ImageLayerDesc desc;
  uint32_t bytes_per_pixel = 0;

  struct Slot {
    holoviz::Image texture;   ///< kExported | kSampled, declared max extent
    CUarray array = nullptr;  ///< mip 0 of the texture, for the per-frame copy
  };
  std::vector<Slot> slots;

  /// Orders the backend's copy after the producer's stream (recorded on the producer,
  /// waited on the backend stream); created once, reused every frame.
  CUevent producer_ready = nullptr;

  // build()-time:
  holoviz::Pipeline pipeline;
  std::vector<holoviz::Descriptor> per_slot_descriptors;
};

[[nodiscard]] expected<HolovizImageLayerState> allocate_holoviz_image_layer(
    holoviz::Device& device, const ImageLayerDesc& desc, uint32_t frames_in_flight);

[[nodiscard]] expected<void> build_holoviz_image_layer(holoviz::Device& device,
                                                       HolovizImageLayerState& state,
                                                       uint32_t frames_in_flight);

[[nodiscard]] expected<void> warm_holoviz_image_layer(HolovizImageLayerState& state,
                                                      holoviz::CommandList& cmd);

/// Record one frame: order the copy after the producer stream (if any) and the slot's last
/// GPU use, land the pixels, draw the quad. The caller keeps the device's CUDA context
/// current and collects `waits` into the frame's submit.
[[nodiscard]] expected<void> record_holoviz_image_layer(
    HolovizImageLayerState& state, holoviz::CommandList& cmd, uint32_t slot,
    holoviz::SubmitSyncToken slot_last_use, const ImageFrameData& frame, CUstream stream,
    holoviz::ExternalSemaphore& semaphore, std::vector<holoviz::ExternalSyncToken>& waits);

[[nodiscard]] expected<void> destroy_holoviz_image_layer(HolovizImageLayerState& state,
                                                         std::chrono::nanoseconds timeout);

}  // namespace holoscan::visualization::backends
