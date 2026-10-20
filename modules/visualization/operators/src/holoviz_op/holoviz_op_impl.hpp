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
#include <optional>
#include <vector>

#include <cuda.h>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/visualization/holoviz_op.hpp>
#include <holoviz/holoviz.hpp>

#include "holoviz_op/backends/cuda_present_backend.hpp"
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

/// Why build_render_list() dropped an image layer's sample, or kAccepted. Kept per declared
/// layer in Impl::last_image_reject_reason_: the operator logs a rejection only when the reason
/// changes, and the admission tests read the reason back instead of scraping log text.
enum class ImageRejectReason : std::uint8_t {
  kAccepted = 0,
  kNoData,                  ///< ImageT::data is null
  kEncodingMismatch,        ///< encoding maps to an ImageFormat other than the layer's
  kUnmappedEncoding,        ///< encoding has no ImageFormat this operator renders
  kColorSpaceIncompatible,  ///< labelled color_space disagrees with the layer's declaration
  kPlaneLayoutsOnCustom,    ///< CUSTOM frame declares planes the single-plane path never consults
  kPlaneLayoutsInvalid,     ///< multi-planar frame whose plane records are missing or unusable
  kGeometryMismatch,        ///< declared width/height disagree with the Tensor or exceed the maxima
  kUnsupportedMemory,       ///< backing Tensor lives on a device kind this path cannot read
};

struct HolovizOp::Impl {
  explicit Impl(HolovizOp& owner) : op(owner) {}

  /// The implementation behind \p op, for tests that compile the operator sources directly
  /// (operators/tests/unit) and inspect runtime state the public API deliberately hides.
  [[nodiscard]] static Impl& of(HolovizOp& op) noexcept { return *op.impl_; }

  /// Host-side copy of a layer's most recently admitted host-resident image sample.
  ///
  /// build_render_list() stages every admitted host sample here before the H2D upload, replacing
  /// the previous contents atomically so a failed copy leaves the last good extent intact. It is
  /// not read on activations where the layer is missing: holding the last frame across
  /// activations is the temporal contract's job (ReuseLast, see HolovizOp::contract()). The
  /// image-admission test uses the cached extent as its oracle. Empty (bytes.empty() == true)
  /// until the first admitted sample for that layer.
  struct ImageLayerCache {
    std::vector<std::byte> bytes;  ///< host repack; empty on the direct device-pointer path
    std::uint32_t width{0};
    std::uint32_t height{0};
    std::size_t pitch{0};
    /// Planes of the last frame handed to the backend: device pointers into the staging buffer
    /// (host path) or the producer's CUDA tensor (direct path). Gives tests an oracle for the
    /// direct path, which never touches `bytes`.
    backends::ImageFrameData last_frame{};
  };

  // ---- layer input ports ---------------------------------------------------------

  /// Input port for image slot \p slot (< \ref kMaxImageLayerPorts). \p slot is a position in
  /// the image-kind pool, not a declared layer index — see HolovizOp's constructor. These ports
  /// live here (not on HolovizOp) to keep the port bank itself an implementation detail; the
  /// public HolovizOp::image_layer_input()/tensor_layer_input() accessors (declared-layer-index
  /// based, for graph wiring) forward to these through HolovizOp's pool_slot_for_layer_ map.
  /// Note this does not keep sensor_io private: the public accessors' own return types still
  /// require schema::ImageT to be a complete type in holoviz_op.hpp, so holoscan::sensor_io is a
  /// PUBLIC dependency of holoscan_operators_holoviz regardless of where the ports live — see
  /// the ImageT port alignment design doc's Decision 2 addendum (second correction).
  [[nodiscard]] const holoscan::Input<holoscan::schema::ImageT>& image_layer_input(
      std::size_t slot) const noexcept {
    switch (slot) {
      case 0:
        return image_layer_input_0_;
      case 1:
        return image_layer_input_1_;
      case 2:
        return image_layer_input_2_;
      case 3:
        return image_layer_input_3_;
      case 4:
        return image_layer_input_4_;
      case 5:
        return image_layer_input_5_;
      case 6:
        return image_layer_input_6_;
      case 7:
        return image_layer_input_7_;
      default:
        std::terminate();  // slot >= kMaxImageLayerPorts
    }
  }
  [[nodiscard]] holoscan::Input<holoscan::schema::ImageT>& image_layer_input(
      std::size_t slot) noexcept {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    return const_cast<holoscan::Input<holoscan::schema::ImageT>&>(
        static_cast<const Impl*>(this)->image_layer_input(slot));
  }

  /// Input port for tensor slot \p slot (< \ref kMaxTensorLayerPorts). \p slot is a position in
  /// the tensor-kind pool (geometry/text/LUT), not a declared layer index — see HolovizOp's
  /// constructor.
  [[nodiscard]] const holoscan::Input<holoscan::Tensor>& tensor_layer_input(
      std::size_t slot) const noexcept {
    switch (slot) {
      case 0:
        return tensor_layer_input_0_;
      case 1:
        return tensor_layer_input_1_;
      case 2:
        return tensor_layer_input_2_;
      case 3:
        return tensor_layer_input_3_;
      case 4:
        return tensor_layer_input_4_;
      case 5:
        return tensor_layer_input_5_;
      case 6:
        return tensor_layer_input_6_;
      case 7:
        return tensor_layer_input_7_;
      default:
        std::terminate();  // slot >= kMaxTensorLayerPorts
    }
  }
  [[nodiscard]] holoscan::Input<holoscan::Tensor>& tensor_layer_input(std::size_t slot) noexcept {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    return const_cast<holoscan::Input<holoscan::Tensor>&>(
        static_cast<const Impl*>(this)->tensor_layer_input(slot));
  }

  // ---- lifecycle (called from the HolovizOp overrides) --------------------------

  void start();
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context);
  void stop();

  // ---- helpers ------------------------------------------------------------------

  /// Per-frame heap storage for geometry batches, text runs and image planes, and the render
  /// list built over it. Must outlive render_->record() and render_->submit() for the same frame.
  struct FrameDataStorage {
    // One entry per geometry layer (indexed by layer order, not port index).
    std::vector<std::vector<backends::GeometryBatch>> geometry_batches;
    std::vector<backends::GeometryFrameData> geometry_frames;
    // One entry per text layer with runs this frame; the runs' string_views borrow the sample
    // Tensor kept in record_tensors.
    std::vector<std::vector<backends::TextRun>> text_runs;
    std::vector<backends::TextFrameData> text_frames;
    std::vector<backends::ImageFrameData> image_frames;
    // Geometry spans and text bytes are borrowed through record(); direct CUDA image pointers
    // are borrowed until the render frame slot recycles. Fixed arrays avoid another
    // per-activation allocation.
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
  /// @return A CUDA device pointer sized for the tightly packed maximum source image,
  ///         including both NV12 planes. RGB8 uses three bytes per pixel before expansion.
  ///         Valid only between start() and stop(); zero for non-ImageLayerDesc layers.
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

  /// Wire semantics a CudaSinkDesc frame carries on the output port for a sink format: the
  /// schema::ImageEncoding a consumer reads back as the format, and the color space (the two sRGB
  /// formats are stamped ColorSpace_SRGB; every other admitted format makes no claim). The inverse
  /// of image_format_for_encoding() in holoviz_op_impl.cpp, kept here rather than in
  /// common/holoviz_conversions.* so the sensor_io dependency stays confined to the operator.
  struct CudaSinkWire {
    schema::ImageEncoding encoding;
    schema::ColorSpace color_space;
  };
  /// Element layout of a CudaSinkDesc frame's data Tensor for a sink format: the format's native
  /// element type, and `{H, W}` (rank 2, one channel) or `{H, W, channels}` (rank 3).
  struct CudaSinkElement {
    DLDataType dtype;
    std::uint32_t rank;
    std::int64_t channels;
  };
  /// nullopt for a format the CUDA sink does not publish (planar/YUV, RGB8, kUnknown). Both
  /// tables admit exactly the same formats; setup() and start() gate on them.
  [[nodiscard]] static std::optional<CudaSinkWire> cuda_sink_wire(ImageFormat format) noexcept;
  [[nodiscard]] static std::optional<CudaSinkElement> cuda_sink_element(
      ImageFormat format) noexcept;

  /// CudaSinkDesc: loan this frame's output Tensor from the port's Runtime pool, open it for a
  /// stream-ordered write on the activation stream, and register its storage as the present
  /// backend's resolve target for @p image_index. On error the caller abandons whatever was
  /// placed in @p loan / @p guard (guard first, then loan).
  [[nodiscard]] holoscan::expected<void, holoscan::Error> prepare_cuda_output(
      const CudaSinkDesc& sink, std::uint32_t image_index, holoscan::ExecutionContext& context,
      std::optional<holoscan::TensorOutputLoan>& loan,
      std::optional<holoscan::TensorOutputWriteGuard>& guard);

  /// Park a CUDA-sink output whose resolve is already enqueued (see RetainedOutput).
  void retain_output(std::uint32_t image_index,
                     std::optional<holoscan::TensorOutputWriteGuard>& guard,
                     std::optional<holoscan::TensorOutputLoan>& loan);
  /// Release every retained output whose image's ready event has completed.
  void release_completed_retained_outputs();

  // ---- state --------------------------------------------------------------------

  /// The owning operator: the declared layer stack and sink, and the port bank.
  HolovizOp& op;

  // Input<T> is non-movable, hence fixed banks rather than arrays (matching the named-member
  // port bank style used upstream). Both banks hold kMaxLayerPorts slots so any kind mix of a
  // legal stack fits. Slot assignment (which declared layer binds to which slot) is computed in
  // HolovizOp's constructor and stored in op.pool_slot_for_layer_; ports beyond the declared
  // pool usage stay unbound.
  holoscan::Input<holoscan::schema::ImageT> image_layer_input_0_;
  holoscan::Input<holoscan::schema::ImageT> image_layer_input_1_;
  holoscan::Input<holoscan::schema::ImageT> image_layer_input_2_;
  holoscan::Input<holoscan::schema::ImageT> image_layer_input_3_;
  holoscan::Input<holoscan::schema::ImageT> image_layer_input_4_;
  holoscan::Input<holoscan::schema::ImageT> image_layer_input_5_;
  holoscan::Input<holoscan::schema::ImageT> image_layer_input_6_;
  holoscan::Input<holoscan::schema::ImageT> image_layer_input_7_;
  holoscan::Input<holoscan::Tensor> tensor_layer_input_0_;
  holoscan::Input<holoscan::Tensor> tensor_layer_input_1_;
  holoscan::Input<holoscan::Tensor> tensor_layer_input_2_;
  holoscan::Input<holoscan::Tensor> tensor_layer_input_3_;
  holoscan::Input<holoscan::Tensor> tensor_layer_input_4_;
  holoscan::Input<holoscan::Tensor> tensor_layer_input_5_;
  holoscan::Input<holoscan::Tensor> tensor_layer_input_6_;
  holoscan::Input<holoscan::Tensor> tensor_layer_input_7_;

  // Wire-semantics diagnostics per declared layer (indexed by declared layer index). The
  // rejection reason is logged only when it changes, so a producer that never matches its layer
  // costs one log line rather than one per activation; the UNSPECIFIED-color-space note is
  // emitted once per layer.
  std::array<ImageRejectReason, kMaxLayerPorts> last_image_reject_reason_{};
  std::array<bool, kMaxLayerPorts> unspecified_color_space_noted_{};

  // Runtime backend handles — all valid only between start() and stop().
  // holoviz::Window and holoviz::Device expose operator bool() (impl_ != nullptr) so bare
  // value members suffice; stop() guards each step with the bool predicate.
  holoviz::Window window_;
  backends::SurfaceHandle surface_handle_;
  holoviz::Device device_;
  CUstream image_upload_stream_{nullptr};
  std::unique_ptr<backends::RenderBackend> render_;
  std::unique_ptr<backends::PresentBackend> present_;
  // Typed view of present_ for a CudaSinkDesc sink (nullptr for the display sinks). Never
  // outlives present_: stop() clears it before present_.reset().
  backends::CudaPresentBackend* cuda_present_{nullptr};
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

  /// A CUDA-sink output whose resolve was enqueued on the present backend's stream but whose
  /// frame then failed before commit()/emit(). Releasing the loan there would let the pool
  /// record readiness on the activation stream only — which the resolve is not ordered before —
  /// and recycle the allocation under the in-flight copy. The loan is parked here instead until
  /// the image's ready event proves the copy complete (release_completed_retained_outputs()),
  /// or until stop() drained the backend.
  struct RetainedOutput {
    std::uint32_t image_index = 0;
    std::optional<holoscan::TensorOutputWriteGuard> guard;
    std::optional<holoscan::TensorOutputLoan> loan;
  };
  // unique_ptr: the guard and loan are move-constructible only, and the vector is compacted.
  std::vector<std::unique_ptr<RetainedOutput>> retained_outputs_;

  // Last-received-sample cache per layer, used to hold the last frame on activations where a
  // layer's LatestWithin selector has nothing new (see image_layer_cache()'s doc comment).
  std::array<ImageLayerCache, kMaxLayerPorts> image_layer_caches_;
};

}  // namespace holoscan::visualization
