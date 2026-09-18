// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// HolovizOp's private implementation: the runtime state that exists between start() and stop()
// and the frame loop that drives it. Kept out of <holoscan/visualization/holoviz_op.hpp> so the
// installed header depends on neither the backend seam nor HoloViz.

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda.h>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/visualization/holoviz_op.hpp>
#include <holoviz/holoviz.hpp>

#include "holoviz_op/backends/frame_data.hpp"
#include "holoviz_op/backends/handles.hpp"
#include "holoviz_op/backends/present_backend.hpp"
#include "holoviz_op/backends/render_backend.hpp"
#include "holoviz_op/types.hpp"

namespace holoscan::visualization {

/// Compile-time upper bound on the swapchain image count HolovizOp supports. The render slot
/// ring (RenderConfig::max_frames_in_flight) and the per-image staging buffer ring are both
/// sized to this bound once in start(), not to the swapchain's initial actual image count — a
/// present-mode-driven recreation can grow the image count later (kImageCountChanged), and a
/// ring sized only to the initial count would then be indexed out of bounds. 8 comfortably
/// covers the present modes and adapters this operator targets (observed swapchains request at
/// most a handful of images); start() rejects an initial count above this bound, and compute()
/// rejects a later grown count above it the same way, rather than attempt a live ring rebuild.
inline constexpr std::uint32_t kMaxSwapchainImages = 8;

struct HolovizOp::Impl {
  explicit Impl(HolovizOp& owner) : op(owner) {}

  /// The implementation behind \p op, for tests that compile the operator sources directly
  /// (operators/tests/unit) and inspect runtime state the public API deliberately hides.
  [[nodiscard]] static Impl& of(HolovizOp& op) noexcept { return *op.impl_; }

  /// Host-side cache of a layer's most recently received image sample.
  ///
  /// build_render_list() re-uploads this on activations where the layer's input is missing (see
  /// its usage for why: kLatestWithin selects a sample at most once, so a producer slower than
  /// the display clock legitimately has no new sample on most activations — without holding the
  /// last frame here the layer would flicker between real content and nothing every such tick).
  /// Empty (bytes.empty() == true) until the first real sample for that layer arrives.
  struct ImageLayerCache {
    std::vector<std::byte> bytes;
    std::uint32_t width{0};
    std::uint32_t height{0};
    std::size_t pitch{0};
  };

  // ---- lifecycle (called from the HolovizOp overrides) --------------------------

  void start();
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context);
  void stop();

  // ---- helpers ------------------------------------------------------------------

  /// Per-frame heap storage for geometry batches and image planes, and the render list built
  /// over it. Must outlive render_->record() and render_->submit() for the same frame.
  struct FrameDataStorage {
    // One entry per geometry layer (indexed by layer order, not port index).
    std::vector<std::vector<backends::GeometryBatch>> geometry_batches;
    std::vector<backends::GeometryFrameData> geometry_frames;
    std::vector<backends::ImageFrameData> image_frames;
    // Geometry spans are borrowed through record(); direct CUDA image pointers are borrowed
    // until the render frame slot recycles. Fixed arrays avoid another per-activation
    // allocation.
    std::array<holoscan::Tensor, kMaxLayerPorts> record_tensors;
    std::array<holoscan::Tensor, kMaxLayerPorts> asynchronous_image_tensors;
  };
  struct BuildResult {
    backends::RenderList list;
    FrameDataStorage storage;
  };

  /// Hand every direct CUDA image Tensor in @p storage whose backing supports it to stream-ordered
  /// release on @p stream (record() has queued the final source read there); the rest stay behind
  /// for the frame-slot retention fallback.
  static void register_stream_ordered_image_release(FrameDataStorage& storage, CUstream stream);

  /// Build the render list from the non-missing input ports for swapchain image @p image_index.
  /// Hard failures (copy/upload) propagate so compute() can recover the open frame; soft failures
  /// (bad shape, unsupported format) log and skip the affected layer.
  [[nodiscard]] holoscan::expected<BuildResult, holoscan::Error> build_render_list(
      holoscan::ExecutionContext& context, std::uint32_t image_index);

  /// Return the device staging buffer for layer @p index's host-resident image data.
  ///
  /// One buffer exists per swapchain image (see start()'s Step 5) so that a new frame's upload
  /// never overwrites a buffer the GPU may still be sampling for a previous, still-in-flight
  /// frame — the swapchain image fence alone does not protect this side-channel buffer.
  ///
  /// @param index       Zero-based layer index; must be less than kMaxLayerPorts.
  /// @param image_index Swapchain image index for the frame currently being built (see
  ///                     PresentBackend::acquire()'s AcquiredImage::image_index).
  /// @return A CUDA device pointer sized to that layer's declared max_width*max_height*4 bytes,
  ///         valid only between start() and stop(); zero for non-ImageLayerDesc layers.
  [[nodiscard]] CUdeviceptr image_staging_devptr(std::size_t index,
                                                 std::uint32_t image_index) const noexcept {
    if (index >= kMaxLayerPorts || image_index >= image_staging_devptrs_.size()) {
      return 0;
    }
    return image_staging_devptrs_[image_index][index];
  }

  /// @param index Zero-based layer index; must be less than kMaxLayerPorts.
  [[nodiscard]] ImageLayerCache& image_layer_cache(std::size_t index) noexcept {
    return image_layer_caches_[index];
  }

  /// Adopt a post-recreation SwapchainInfo reported via kImageCountChanged.
  ///
  /// The render slot ring and staging ring are both pre-sized to kMaxSwapchainImages in start()
  /// (see its doc comment), so a grown count within that bound needs no ring rebuild — this just
  /// updates swapchain_info_. A grown count exceeding the bound cannot be served by either ring;
  /// unlike kBackpressured/kOutOfDate, that is not safely retryable, so this reports false rather
  /// than let the caller treat it as transient.
  /// @return true if `info` was adopted; false if `info.image_count` exceeds kMaxSwapchainImages.
  [[nodiscard]] bool adopt_swapchain_info_change(const backends::SwapchainInfo& info) noexcept;

  // ---- state --------------------------------------------------------------------

  /// The owning operator: the declared layer stack and sink, and the port bank.
  HolovizOp& op;

  // Runtime backend handles — all valid only between start() and stop().
  // holoviz::Window and holoviz::Device expose operator bool() (impl_ != nullptr) so bare
  // value members suffice; stop() guards each step with the bool predicate.
  holoviz::Window window_;
  backends::SurfaceHandle surface_handle_;
  holoviz::Device device_;
  CUstream image_upload_stream_{nullptr};
  std::unique_ptr<backends::RenderBackend> render_;
  std::unique_ptr<backends::PresentBackend> present_;
  backends::SwapchainInfo swapchain_info_{};
  backends::PresentId present_id_{0};
  // Set to true if stop() returns early because render_->destroy() could not prove quiescence.
  // Prevents restart (start() throws) and signals the operator destructor to clear
  // retained_image_tensors_ before member destructors release GPU-backing memory.
  bool terminal_failed_{false};

  // CUDA ordinal device_ was created against (start()), used to reject an activation stream from
  // a different device at the top of compute() — see the doc comment there.
  int cuda_device_ordinal_{-1};

  // Device staging buffers for ImageLayerDesc layers: one array of kMaxLayerPorts buffers per
  // swapchain image (outer index == swapchain image_index), so uploads for frame N+1 cannot
  // race a still-in-flight GPU read of frame N's buffer. Host-resident image tensors are
  // uploaded here in compute() before being handed to the render backend, which requires a
  // real CUDA device pointer (ImagePlane::device_ptr). Zero for layers that are not
  // ImageLayerDesc, or before start() runs. Sized to kMaxSwapchainImages in start().
  std::vector<std::array<CUdeviceptr, kMaxLayerPorts>> image_staging_devptrs_;

  // Direct CUDA image inputs whose backing cannot register image_upload_stream_ for stream-ordered
  // release remain here until the next successful begin_frame() proves the previous reads have
  // completed. Native Holoscan and HoloMQ-backed tensors normally take the stream-aware fast path;
  // producer-owned DLPack tensors use this bounded fallback.
  std::array<std::array<holoscan::Tensor, kMaxLayerPorts>, kMaxSwapchainImages>
      retained_image_tensors_;

  // Last-received-sample cache per layer, used to hold the last frame on activations where a
  // layer's LatestWithin selector has nothing new (see image_layer_cache()'s doc comment).
  std::array<ImageLayerCache, kMaxLayerPorts> image_layer_caches_;
};

}  // namespace holoscan::visualization
