// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The geometry layer of HolovizRenderBackend: 2D overlay primitives (lines, strips, crosses,
// rectangle and oval outlines) with GPU-side expansion from a storage-buffer parameter
// pool. allocate() sizes an exported pool slot per frame-in-flight from the declared
// max_primitives plus a pinned staging block; per frame, the batches' floats are memcpy'd
// into staging, one async HtoD copy lands them in the slot, and each batch is one draw
// with a 32-byte push block — no per-frame allocation anywhere.

#pragma once

#include <cstddef>
#include <vector>

#include <cuda.h>

#include <holoscan/visualization/holoviz_op.hpp>
#include <holoviz/buffer.hpp>
#include <holoviz/command_list.hpp>
#include <holoviz/common.hpp>
#include <holoviz/descriptor.hpp>
#include <holoviz/device.hpp>
#include <holoviz/expected.hpp>
#include <holoviz/pipeline.hpp>
#include "holoviz_op/backends/frame_data.hpp"
#include "holoviz_op/types.hpp"

namespace holoscan::visualization::backends {

struct HolovizGeometryLayerState {
  GeometryLayerDesc desc;

  // allocate()-time memory: one pool region and one pinned staging region per frame
  // slot. Staging must be per-slot: with F frames in flight, slot N-1's async HtoD copy
  // may still be pending (behind a backlogged stream) when frame N stages — only the
  // slot recycled by begin_frame()'s command-list reset() is proven done with its
  // staging region.
  size_t slot_float_capacity = 0;
  size_t slot_stride_bytes = 0;
  holoviz::Buffer pool;  ///< ShareMode::kExported; slot_stride_bytes * frames_in_flight
  CUdeviceptr pool_ptr = 0;
  float* staging = nullptr;        ///< pinned host memory, slot_stride_bytes * frames_in_flight
  size_t staging_slot_floats = 0;  ///< slot_stride_bytes / sizeof(float): per-slot stride

  // build()-time pipelines: one per declared topology, each with a per-slot descriptor
  // bound to that slot's pool slice (descriptors are pre-baked; per frame they are only
  // bound).
  struct TopologyPipeline {
    GeometryTopology topology = GeometryTopology::kLineList;
    holoviz::Pipeline pipeline;
    std::vector<holoviz::Descriptor> per_slot;
  };
  std::vector<TopologyPipeline> pipelines;

  size_t primitive_high_water = 0;
};

/// Parameter floats one primitive of `topology` occupies in the pool/frame data.
[[nodiscard]] uint32_t holoviz_geometry_param_floats(GeometryTopology topology);

[[nodiscard]] expected<HolovizGeometryLayerState> allocate_holoviz_geometry_layer(
    holoviz::Device& device, const GeometryLayerDesc& desc, uint32_t frames_in_flight);

/// Compile the expansion pipelines for the declared topologies and pre-bake the per-slot
/// descriptors. Pipelines compile their pass-specific variant lazily on first bind; the
/// backend's build() warms them inside its warm render scope via warm_holoviz_geometry_layer().
[[nodiscard]] expected<void> build_holoviz_geometry_layer(holoviz::Device& device,
                                                          HolovizGeometryLayerState& state,
                                                          uint32_t frames_in_flight);

/// Bind every pipeline once inside the caller's open (warm) render scope, driving the lazy
/// per-pass compilation at build() time.
[[nodiscard]] expected<void> warm_holoviz_geometry_layer(HolovizGeometryLayerState& state,
                                                         holoviz::CommandList& cmd);

/// Record one frame's batches: stage the floats, one async HtoD copy on `stream` ordered
/// after `slot_last_use` via `semaphore`, then one draw per batch. Appends the upload's
/// completion token to `waits`. The caller pushes/keeps the device's CUDA context current.
[[nodiscard]] expected<void> record_holoviz_geometry_layer(
    HolovizGeometryLayerState& state, holoviz::CommandList& cmd, holoviz::Extent2D extent,
    uint32_t slot, holoviz::SubmitSyncToken slot_last_use, const GeometryFrameData& frame,
    CUstream stream, holoviz::ExternalSemaphore& semaphore,
    std::vector<holoviz::ExternalSyncToken>& waits);

[[nodiscard]] expected<void> destroy_holoviz_geometry_layer(HolovizGeometryLayerState& state,
                                                            std::chrono::nanoseconds timeout);

}  // namespace holoscan::visualization::backends
