// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "holoviz_op/backends/holoviz_geometry_layer.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

#include <holoviz/adapter.hpp>
#include "holoviz_op/types.hpp"

#include "cuda/cu_errors.hpp"
#include "holoviz_op/backends/holoviz_geometry_interface.hpp"
#include "holoviz_op/backends/layer_blend.hpp"

#include "backends/holoviz_geometry.slang.hpp"  // generated from holoviz_geometry.slang

namespace holoscan::visualization::backends {

namespace {

namespace shader = detail::holoviz_geometry;

// The push block, the vertex count each primitive expands to and the parameter-buffer strides
// come from holoviz_geometry_interface.hpp, the header the shader compiles from too. One Slang
// source, one vertex entry point per expansion family: the generated header asserts every pair's
// reflected block against the same shader::PushConstants.

holoviz::ShaderBlob shader_blob_for(GeometryTopology topology) {
  switch (topology) {
    case GeometryTopology::kLineList:
      return shader::line_list::make_shader_blob();
    case GeometryTopology::kLineStrip:
      return shader::line_strip::make_shader_blob();
    case GeometryTopology::kCrossList:
      return shader::cross::make_shader_blob();
    case GeometryTopology::kRectangleList:
      return shader::rect::make_shader_blob();
    case GeometryTopology::kOvalList:
      return shader::oval::make_shader_blob();
  }
  return shader::line_list::make_shader_blob();
}

// Every topology lowers to triangles: lines/strips/crosses expand to quads in the vertex
// shader so GeometryBatch::line_width is honored (native GPU lines are fixed-width 1 and
// wide lines are an optional Vulkan feature).
holoviz::PrimitiveTopology gpu_topology_for(GeometryTopology topology) {
  (void)topology;
  return holoviz::PrimitiveTopology::kTriangleList;
}

}  // namespace

uint32_t holoviz_geometry_param_floats(GeometryTopology topology) {
  switch (topology) {
    case GeometryTopology::kLineList:
      return shader::kGeometryLineListFloats;
    case GeometryTopology::kLineStrip:
      return shader::kGeometryLineStripFloats;  // per point (primitive_count = points - 1)
    case GeometryTopology::kCrossList:
      return shader::kGeometryCrossFloats;
    case GeometryTopology::kRectangleList:
      return shader::kGeometryRectFloats;
    case GeometryTopology::kOvalList:
      return shader::kGeometryOvalFloats;
  }
  return shader::kGeometryLineListFloats;
}

expected<HolovizGeometryLayerState> allocate_holoviz_geometry_layer(holoviz::Device& device,
                                                                    const GeometryLayerDesc& desc,
                                                                    uint32_t frames_in_flight) {
  if (desc.max_primitives == 0 || desc.topologies.empty()) {
    return make_unexpected(
        Error{.code = ErrorCode::kInvalidArgument,
              .message = "[holoviz] GeometryLayerDesc needs a nonzero max_primitives and at "
                         "least one declared topology."});
  }

  HolovizGeometryLayerState state;
  state.desc = desc;

  // Batches of any declared topology can fill the frame, so the slot must hold the
  // worst-case packing: max_primitives of the most float-hungry declared topology. Line
  // strips are charged at 4 floats per primitive: a batch of s segments stages 2s + 2
  // floats (the trailing point), and 2s + 2 <= 4s for s >= 1, so this bounds ANY mix of
  // strip batches — including max_primitives one-segment strips — within the primitive
  // budget.
  size_t worst_floats_per_frame = 0;
  for (const GeometryTopology topology : desc.topologies) {
    size_t floats =
        static_cast<size_t>(holoviz_geometry_param_floats(topology)) * desc.max_primitives;
    if (topology == GeometryTopology::kLineStrip) {
      floats = static_cast<size_t>(4) * desc.max_primitives;
    }
    worst_floats_per_frame = std::max(worst_floats_per_frame, floats);
  }
  state.slot_float_capacity = worst_floats_per_frame;

  VIZ_UNWRAP(holoviz::CapabilityReport, caps, device.capabilities());
  const size_t alignment = std::max<size_t>(caps.storage_buffer_alignment, sizeof(float));
  const size_t raw_bytes = worst_floats_per_frame * sizeof(float);
  state.slot_stride_bytes = (raw_bytes + alignment - 1) / alignment * alignment;

  VIZ_UNWRAP(holoviz::Buffer,
             pool,
             device.create_buffer(holoviz::BufferDesc{
                 .size_bytes = state.slot_stride_bytes * frames_in_flight,
                 .share = holoviz::ShareMode::kExported,
                 .debug_name = "holoviz render backend geometry pool",
             }));
  state.pool = std::move(pool);
  VIZ_UNWRAP(CUdeviceptr, pool_ptr, state.pool.cuda_device_ptr());
  state.pool_ptr = pool_ptr;

  // Pinned staging so the per-frame HtoD copy is a true async DMA. One region per frame
  // slot (see the header): reuse of slot s's region is fenced by begin_frame()'s
  // command-list reset(), which proves slot s's previous submit — and therefore the
  // upload it waited on — completed.
  VIZ_UNWRAP(CUcontext, context, device.cuda_context());
  VIZ_CU_FORWARD_ERROR(cuCtxPushCurrent(context));
  void* staging = nullptr;
  const CUresult alloc_result =
      cuMemHostAlloc(&staging, state.slot_stride_bytes * frames_in_flight, 0);
  CUcontext popped = nullptr;
  cuCtxPopCurrent(&popped);
  if (alloc_result != CUDA_SUCCESS) {
    return make_unexpected(Error{.code = ErrorCode::kCudaError,
                                 .message = "[holoviz] cuMemHostAlloc failed for the geometry "
                                            "staging block."});
  }
  state.staging = static_cast<float*>(staging);
  state.staging_slot_floats = state.slot_stride_bytes / sizeof(float);

  return state;
}

expected<void> build_holoviz_geometry_layer(holoviz::Device& device,
                                            HolovizGeometryLayerState& state,
                                            uint32_t frames_in_flight) {
  state.pipelines.reserve(state.desc.topologies.size());
  for (const GeometryTopology topology : state.desc.topologies) {
    holoviz::ShaderBlob blob = shader_blob_for(topology);
    if (!blob.attributes.empty()) {
      return make_unexpected(Error{.code = ErrorCode::kFailure,
                                   .message =
                                       "[holoviz] Geometry shader unexpectedly reflected vertex "
                                       "attributes; it must stay vertex-buffer-less."});
    }

    HolovizGeometryLayerState::TopologyPipeline entry;
    entry.topology = topology;
    auto pipeline = device.create_pipeline(holoviz::PipelineDesc{
        .shader_blob = std::move(blob),
        .topology = gpu_topology_for(topology),
        .blend = kStraightAlphaLayerBlend,
        .debug_name = "holoviz render backend geometry",
    });
    if (!pipeline) {
      return forward_error(pipeline);
    }
    entry.pipeline = std::move(*pipeline);

    entry.per_slot.reserve(frames_in_flight);
    for (uint32_t slot = 0; slot < frames_in_flight; ++slot) {
      const auto slot_slice =
          state.pool.slice(slot * state.slot_stride_bytes, state.slot_stride_bytes);
      if (!slot_slice) {
        return forward_error(slot_slice);
      }
      VIZ_UNWRAP(holoviz::Descriptor, descriptor, entry.pipeline.create_descriptor());
      VIZ_FORWARD_ERROR(descriptor.bind_buffer(*slot_slice, "geometryParams"));
      entry.per_slot.push_back(std::move(descriptor));
    }
    state.pipelines.push_back(std::move(entry));
  }
  return {};
}

expected<void> warm_holoviz_geometry_layer(HolovizGeometryLayerState& state,
                                           holoviz::CommandList& cmd) {
  for (HolovizGeometryLayerState::TopologyPipeline& entry : state.pipelines) {
    VIZ_FORWARD_ERROR(cmd.bind_pipeline(entry.pipeline));
  }
  return {};
}

// Keep batch admission, staged upload, synchronization, and ordered draws in one frame transaction.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
expected<void> record_holoviz_geometry_layer(HolovizGeometryLayerState& state,
                                             holoviz::CommandList& cmd, holoviz::Extent2D extent,
                                             uint32_t slot, holoviz::SubmitSyncToken slot_last_use,
                                             const GeometryFrameData& frame, CUstream stream,
                                             holoviz::ExternalSemaphore& semaphore,
                                             std::vector<holoviz::ExternalSyncToken>& waits) {
  // Validate the declaration bounds and stage all batches' floats, recording each batch's
  // base offset within the slot.
  size_t primitive_count = 0;
  size_t float_offset = 0;
  for (const GeometryBatch& batch : frame.batches) {
    const uint32_t per_prim = holoviz_geometry_param_floats(batch.topology);
    size_t batch_floats = static_cast<size_t>(per_prim) * batch.primitive_count;
    if (batch.topology == GeometryTopology::kLineStrip) {
      batch_floats += 2;  // the strip's trailing point
    }
    if (batch.coords.size() < batch_floats) {
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = "[holoviz] GeometryBatch::coords is shorter than primitive_count "
                           "requires for its topology."});
    }
    primitive_count += batch.primitive_count;
    if (primitive_count > state.desc.max_primitives ||
        float_offset + batch_floats > state.slot_float_capacity) {
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = "[holoviz] Frame geometry exceeds the layer's declared "
                           "max_primitives — a mis-declaration, not an expected code path."});
    }
    std::memcpy(
        state.staging + static_cast<size_t>(slot) * state.staging_slot_floats + float_offset,
        batch.coords.data(),
        batch_floats * sizeof(float));
    float_offset += batch_floats;
  }
  state.primitive_high_water = std::max(state.primitive_high_water, primitive_count);

  if (float_offset == 0) {
    return {};  // nothing to draw this frame
  }

  // Order the device-side copy after the GPU's last read of this pool slot, land it, and
  // hand the completion token to the frame's submit. Note this stream-side wait does NOT
  // protect staging reuse (its timeline value already passed on the host by the time
  // record() runs — begin_frame()'s reset() waited it); staging safety comes from the
  // per-slot staging regions.
  VIZ_FORWARD_ERROR(semaphore.wait(slot_last_use));
  VIZ_CU_FORWARD_ERROR(
      cuMemcpyHtoDAsync(state.pool_ptr + slot * state.slot_stride_bytes,
                        state.staging + static_cast<size_t>(slot) * state.staging_slot_floats,
                        float_offset * sizeof(float),
                        stream));
  VIZ_UNWRAP(holoviz::ExternalSyncToken, upload_token, semaphore.signal());
  waits.push_back(upload_token);

  // One draw per batch.
  uint32_t base_float = 0;
  for (const GeometryBatch& batch : frame.batches) {
    HolovizGeometryLayerState::TopologyPipeline* pipeline = nullptr;
    for (HolovizGeometryLayerState::TopologyPipeline& entry : state.pipelines) {
      if (entry.topology == batch.topology) {
        pipeline = &entry;
        break;
      }
    }
    if (pipeline == nullptr) {
      return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                                   .message =
                                       "[holoviz] GeometryBatch uses a topology the layer did not "
                                       "declare in GeometryLayerSpec::primitives."});
    }

    VIZ_FORWARD_ERROR(cmd.bind_pipeline(pipeline->pipeline));
    VIZ_FORWARD_ERROR(cmd.bind_descriptor(pipeline->per_slot[slot]));

    const shader::PushConstants push{
        shader::float4(batch.color[0], batch.color[1], batch.color[2], batch.color[3]),
        shader::float2(static_cast<float>(extent.width), static_cast<float>(extent.height)),
        batch.line_width > 0.0F ? batch.line_width : state.desc.default_line_width,
        base_float,
    };
    VIZ_FORWARD_ERROR(cmd.set_push_data(
        pipeline->pipeline,
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(&push), sizeof(push))));

    // All topologies draw instanced: one instance per primitive (per segment for strips).
    const uint32_t count = batch.primitive_count;
    switch (batch.topology) {
      case GeometryTopology::kLineList:
      case GeometryTopology::kLineStrip:
        VIZ_FORWARD_ERROR(cmd.draw(shader::kGeometryLineVertices, count));
        break;
      case GeometryTopology::kCrossList:
        VIZ_FORWARD_ERROR(cmd.draw(shader::kGeometryCrossVertices, count));
        break;
      case GeometryTopology::kRectangleList:
        VIZ_FORWARD_ERROR(cmd.draw(shader::kGeometryRectVertices, count));
        break;
      case GeometryTopology::kOvalList:
        VIZ_FORWARD_ERROR(cmd.draw(shader::kGeometryOvalVertices, count));
        break;
    }

    uint32_t batch_floats = holoviz_geometry_param_floats(batch.topology) * count;
    if (batch.topology == GeometryTopology::kLineStrip) {
      batch_floats += 2;
    }
    base_float += batch_floats;
  }

  return {};
}

expected<void> destroy_holoviz_geometry_layer(HolovizGeometryLayerState& state,
                                              std::chrono::nanoseconds timeout) {
  expected<void> result;
  for (HolovizGeometryLayerState::TopologyPipeline& entry : state.pipelines) {
    for (holoviz::Descriptor& descriptor : entry.per_slot) {
      if (const expected<void> r = descriptor.destroy(timeout); !r && result) {
        result = r;
      }
    }
    if (entry.pipeline) {
      if (const expected<void> r = entry.pipeline.destroy(timeout); !r && result) {
        result = r;
      }
    }
  }
  state.pipelines.clear();
  if (state.staging != nullptr) {
    cuMemFreeHost(state.staging);
    state.staging = nullptr;
  }
  if (state.pool) {
    if (const expected<void> r = state.pool.destroy(timeout); !r && result) {
      result = r;
    }
  }
  return result;
}

}  // namespace holoscan::visualization::backends
