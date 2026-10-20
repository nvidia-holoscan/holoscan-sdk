// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The image layer of HolovizRenderBackend: GPU-resident pixel frames (CUdeviceptr + pitch)
// async-copied into a per-slot exported texture and drawn as one view-transformed quad.
//
// Supported formats: kR8G8B8A8Unorm, kB8G8R8A8Unorm, kR8Unorm, kR16Unorm, kR16G16B16A16Unorm
// (sampled from a slot texture of the matching holoviz::Format; BGRA reuses RGBA8 and swizzles
// in the shader; R8/R16 replicate to gray), plus the source-only formats:
//  - kR8G8B8Unorm — expanded to an RGBA8 slot texture by a CUDA kernel through a surface object.
//  - kNv12 — Y plane (kR8Unorm) + interleaved chroma plane (kR8G8Unorm), converted in the shader
//    with the declared model/range.
//  - kYuyv422 — packed pairs uploaded as an RGBA8 texture of width/2 and decoded in the shader.
// Encoded sRGB RGBA8/BGRA8 sources use an RGBA8 sRGB texture for native decode before
// filtering. Source decoding is independent of the sink's native output encoding.

#pragma once

#include <chrono>
#include <utility>
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
  size_t slot_bytes = 0;

  struct Surface {
    CUsurfObject handle = 0;
    Surface() = default;
    Surface(Surface&& other) noexcept : handle(std::exchange(other.handle, 0)) {}
    Surface(const Surface&) = delete;
    Surface& operator=(const Surface&) = delete;
    ~Surface() {
      if (handle != 0) {
        cuSurfObjectDestroy(handle);
      }
    }
  };

  struct Slot {
    holoviz::Image texture;   ///< kExported | kSampled, declared max extent
    CUarray array = nullptr;  ///< mip 0 of the texture, for the per-frame copy
    holoviz::Image chroma_texture;
    CUarray chroma_array = nullptr;
    Surface surface;  // destroyed before the texture, including allocation rollback
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
