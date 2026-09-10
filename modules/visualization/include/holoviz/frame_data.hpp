// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include <cuda.h>

#include <holoviz/geometry_topology.hpp>

namespace holoviz {

/// The concrete runtime frame-handle types LayerFrame::frame_data points at, one per layer
/// kind in LayerSpec. The RenderBackend seam only moves these (render_backend.hpp); the
/// concrete backend interprets them against the layer's declared spec.
///
/// Lifetime contract:
/// - Host memory (spans, string_views — GeometryBatch::coords, TextRun::utf8_text,
///   LutFrameData::lut, ...) is consumed during record() and must stay valid until
///   RenderBackend::submit() returns for the frame that carries it.
/// - Device memory (ImagePlane::device_ptr, LutFrameData::mask) is read by asynchronous
///   copies that are ordered before the frame's GPU work but may still be executing when
///   submit() returns (e.g. behind a busy producer stream). It must stay valid until the
///   frame's slot recycles — i.e. until the next begin_frame() RETURNS SUCCESSFULLY for the
///   same frame_index — or until abort_frame() returns successfully for this frame, or
///   destroy() returns successfully, whichever comes first. An error return from any of
///   these (e.g. begin_frame()'s kBackpressured timeout, or a failed abort_frame() or
///   destroy()) proves nothing about GPU quiescence and does NOT end the retention duty:
///   the caller must keep the buffers valid and regains recycle permission only at the next
///   successful return of one of the three calls for that slot (retry begin_frame(), abort
///   the open frame, or fall back to destroy()). If destroy() itself fails at terminal
///   teardown, the device may still be reading the buffers; the caller must treat them as
///   in use for the remainder of the process (deliberately not freeing them is the only
///   sound option). Callers cycling frame_index round-robin can equivalently hold each
///   frame's producer buffers for max_frames_in_flight frames (a small ring of the last F
///   input messages) — but only while every quiescing call succeeds; after any failure the
///   ring shortcut is void and holding reverts to the explicit rule above.

/// One plane of GPU-resident pixel data.
struct ImagePlane {
  CUdeviceptr device_ptr = 0;
  size_t row_pitch_bytes = 0;
};

/// Frame data for an ImageLayerSpec layer. `width`/`height` are this frame's actual size and
/// must not exceed the layer's declared max_width/max_height.
///
/// `planes` is sized for the two-plane YUV formats (kNv12: [Y, UV]); single-plane formats use
/// planes[0] with plane_count == 1. YUV formats are not implemented yet — declaring one fails
/// at allocate() — but the shape is fixed now so adding them breaks no signatures.
struct ImageFrameData {
  uint32_t width = 0;
  uint32_t height = 0;
  ImagePlane planes[2] = {};
  uint32_t plane_count = 1;
  /// Stream the producer wrote `planes` on; the backend orders its consumption after the
  /// work queued there. nullptr = the data is already visible (synchronized upstream).
  CUstream stream = nullptr;
};

/// One homogeneous run of primitives for a GeometryLayerSpec layer. `coords` packing
/// (normalized [0,1] framebuffer coordinates, origin top-left), per primitive:
///   kLineList:      x0,y0,x1,y1
///   kLineStrip:     x,y per point (primitive_count = point count - 1)
///   kCrossList:     cx,cy,size
///   kRectangleList: x0,y0,x1,y1 (two corners)
///   kOvalList:      cx,cy,rx,ry
struct GeometryBatch {
  GeometryTopology topology = GeometryTopology::kLineList;
  std::span<const float> coords;  ///< host memory; consumed during record()
  uint32_t primitive_count = 0;
  float color[4] = {1.0F, 1.0F, 1.0F, 1.0F};
  /// Pixels; 0 = the layer's GeometryLayerSpec::default_line_width.
  float line_width = 0.0F;
};

/// Frame data for a GeometryLayerSpec layer. The summed primitive count across batches must
/// not exceed the layer's declared max_primitives.
struct GeometryFrameData {
  std::span<const GeometryBatch> batches;
};

/// One positioned run of text for a TextLayerSpec layer.
struct TextRun {
  float x = 0.0F;  ///< normalized [0,1], top-left anchor
  float y = 0.0F;
  float size_px = 16.0F;
  uint32_t color = 0xFFFFFFFFU;  ///< packed 0xAABBGGRR (text::TextDesc convention)
  std::string_view utf8_text;    ///< host memory; consumed during record()
};

/// Frame data for a TextLayerSpec layer. The summed character count across runs must not
/// exceed the layer's declared max_glyphs.
struct TextFrameData {
  std::span<const TextRun> runs;
};

/// Frame data for a LutLayerSpec layer: a kR8Unorm index mask plus an optional palette
/// update.
struct LutFrameData {
  uint32_t width = 0;
  uint32_t height = 0;
  ImagePlane mask;
  CUstream stream = nullptr;
  /// lut_size * 4 RGBA floats; empty = keep the previously supplied palette.
  std::span<const float> lut;
};

}  // namespace holoviz
