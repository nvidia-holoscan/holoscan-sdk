// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The geometry layer of RhiRenderBackend: 2D overlay primitives (lines, strips, crosses,
// rectangle and oval outlines) with GPU-side expansion from a storage-buffer parameter
// pool. allocate() sizes an exported pool slot per frame-in-flight from the declared
// max_primitives plus a pinned staging block; per frame, the batches' floats are memcpy'd
// into staging, one async HtoD copy lands them in the slot, and each batch is one draw
// with a 32-byte push block — no per-frame allocation anywhere.

#pragma once

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
#include <holoviz/rhi/pipeline.hpp>

namespace holoviz::backends {

struct RhiGeometryLayerState {
  GeometryLayerDesc desc;

  // allocate()-time memory: one pool region and one pinned staging region per frame
  // slot. Staging must be per-slot: with F frames in flight, slot N-1's async HtoD copy
  // may still be pending (behind a backlogged stream) when frame N stages — only the
  // slot recycled by begin_frame()'s command-list reset() is proven done with its
  // staging region.
  size_t slot_float_capacity = 0;
  size_t slot_stride_bytes = 0;
  rhi::Buffer pool;  ///< ShareMode::kExported; slot_stride_bytes * frames_in_flight
  CUdeviceptr pool_ptr = 0;
  float* staging = nullptr;        ///< pinned host memory, slot_stride_bytes * frames_in_flight
  size_t staging_slot_floats = 0;  ///< slot_stride_bytes / sizeof(float): per-slot stride

  // build()-time pipelines: one per declared topology, each with a per-slot descriptor
  // bound to that slot's pool slice (descriptors are pre-baked; per frame they are only
  // bound).
  struct TopologyPipeline {
    GeometryTopology topology = GeometryTopology::kLineList;
    rhi::Pipeline pipeline;
    std::vector<rhi::Descriptor> per_slot;
  };
  std::vector<TopologyPipeline> pipelines;

  size_t primitive_high_water = 0;
};

/// Parameter floats one primitive of `topology` occupies in the pool/frame data.
[[nodiscard]] uint32_t rhi_geometry_param_floats(GeometryTopology topology);

[[nodiscard]] expected<RhiGeometryLayerState> allocate_rhi_geometry_layer(
    rhi::Device& device, const GeometryLayerDesc& desc, uint32_t frames_in_flight);

/// Compile the expansion pipelines for the declared topologies and pre-bake the per-slot
/// descriptors. Pipelines compile their pass-specific variant lazily on first bind; the
/// backend's build() warms them inside its warm render scope via warm_rhi_geometry_layer().
[[nodiscard]] expected<void> build_rhi_geometry_layer(rhi::Device& device,
                                                      RhiGeometryLayerState& state,
                                                      uint32_t frames_in_flight);

/// Bind every pipeline once inside the caller's open (warm) render scope, driving the lazy
/// per-pass compilation at build() time.
[[nodiscard]] expected<void> warm_rhi_geometry_layer(RhiGeometryLayerState& state,
                                                     rhi::CommandList& cmd);

/// Record one frame's batches: stage the floats, one async HtoD copy on `stream` ordered
/// after `slot_last_use` via `semaphore`, then one draw per batch. Appends the upload's
/// completion token to `waits`. The caller pushes/keeps the device's CUDA context current.
[[nodiscard]] expected<void> record_rhi_geometry_layer(
    RhiGeometryLayerState& state, rhi::CommandList& cmd, rhi::Extent2D extent, uint32_t slot,
    rhi::SubmitSyncToken slot_last_use, const GeometryFrameData& frame, CUstream stream,
    rhi::ExternalSemaphore& semaphore, std::vector<rhi::ExternalSyncToken>& waits);

[[nodiscard]] expected<void> destroy_rhi_geometry_layer(RhiGeometryLayerState& state,
                                                        std::chrono::nanoseconds timeout);

}  // namespace holoviz::backends
