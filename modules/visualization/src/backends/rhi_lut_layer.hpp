// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The LUT layer of RhiRenderBackend: a kR8Unorm segmentation-mask frame colorized through
// an RGBA float palette. The mask travels like an image-layer frame (async copy into a
// per-slot exported texture); the palette is kept authoritative on the host (LutFrameData
// with an empty `lut` span keeps the previous palette) and re-staged into the current
// slot's palette buffer each drawn frame — a few KB, and it makes "keep previous" exact
// across frame slots with no cross-slot coherence bookkeeping.

#pragma once

#include <chrono>
#include <cstddef>
#include <vector>

#include <cuda.h>

#include <holoviz/expected.hpp>
#include <holoviz/frame_data.hpp>
#include <holoviz/layer_desc.hpp>
#include <holoviz/rhi/buffer.hpp>
#include <holoviz/rhi/command_list.hpp>
#include <holoviz/rhi/common.hpp>
#include <holoviz/rhi/descriptor.hpp>
#include <holoviz/rhi/device.hpp>
#include <holoviz/rhi/image.hpp>
#include <holoviz/rhi/pipeline.hpp>

namespace holoviz::backends {

struct RhiLUTLayerState {
  LUTLayerDesc desc;

  struct Slot {
    rhi::Image mask;          ///< kR8Unorm, kExported | kSampled, declared max extent
    CUarray array = nullptr;  ///< mip 0, for the per-frame mask copy
  };
  std::vector<Slot> slots;

  rhi::Buffer palette_pool;  ///< kExported; lut_size * 4 floats per slot
  CUdeviceptr palette_ptr = 0;
  size_t palette_stride_bytes = 0;
  /// Pinned staging, one palette region per frame slot (palette_stride_bytes each):
  /// slot N-1's async HtoD may still be pending (behind a producer-stream event or a
  /// backlogged stream) while frame N stages, so the regions must not be shared. Reuse
  /// of slot s's region is fenced by begin_frame()'s command-list reset().
  float* palette_staging = nullptr;
  size_t palette_slot_floats = 0;   ///< palette_stride_bytes / sizeof(float): per-slot stride
  std::vector<float> palette_host;  ///< authoritative palette (lut_size * 4)
  bool palette_ever_set = false;

  CUevent producer_ready = nullptr;

  // build()-time:
  rhi::Pipeline pipeline;
  std::vector<rhi::Descriptor> per_slot_descriptors;
};

[[nodiscard]] expected<RhiLUTLayerState> allocate_rhi_lut_layer(rhi::Device& device,
                                                                const LUTLayerDesc& desc,
                                                                uint32_t frames_in_flight);

[[nodiscard]] expected<void> build_rhi_lut_layer(rhi::Device& device, RhiLUTLayerState& state,
                                                 uint32_t frames_in_flight);

[[nodiscard]] expected<void> warm_rhi_lut_layer(RhiLUTLayerState& state, rhi::CommandList& cmd);

[[nodiscard]] expected<void> record_rhi_lut_layer(RhiLUTLayerState& state, rhi::CommandList& cmd,
                                                  uint32_t slot, rhi::SubmitSyncToken slot_last_use,
                                                  const LutFrameData& frame, CUstream stream,
                                                  rhi::ExternalSemaphore& semaphore,
                                                  std::vector<rhi::ExternalSyncToken>& waits);

[[nodiscard]] expected<void> destroy_rhi_lut_layer(RhiLUTLayerState& state,
                                                   std::chrono::nanoseconds timeout);

}  // namespace holoviz::backends
