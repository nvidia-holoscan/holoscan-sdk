// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <holoscan/visualization/holoviz_op.hpp>
#include <holoviz/expected.hpp>
#include "holoviz_op/backends/handles.hpp"
#include "holoviz_op/types.hpp"

namespace holoscan::visualization::backends {

/// Backend-agnostic configuration passed to \c RenderBackend::allocate().
///
/// \p max_frames_in_flight is required for sizing the per-frame command/sync ring.
/// The actual swapchain image count is determined later by
/// \c PresentBackend::create_swapchain(), which may clamp the final count; the renderer
/// must tolerate \c actual_image_count <= \p max_frames_in_flight. Set this to the maximum
/// image count you would ever request (e.g. 3 for windowed, 2 for exclusive display) so
/// \c allocate() can size all per-frame resources once without reallocation.
struct RenderConfig {
  bool headless = false;
  uint32_t max_frames_in_flight = 3;  ///< upper bound for per-frame ring sizing
  uint32_t headless_width = 0;
  uint32_t headless_height = 0;
  /// Upper bound on begin_frame()'s slot-recycle wait (the frame path's one bounded block);
  /// exceeding it returns kBackpressured. A kBackpressured return leaves the slot
  /// un-recycled: its previous frame's FrameData device memory must stay valid (see the
  /// lifetime contract in frame_data.hpp).
  uint64_t backpressure_timeout_ns = 100'000'000;
  /// RGBA color every frame's render target is cleared to before layer 0 composites
  /// (straight alpha; also used for the pipeline-priming pass in build()).
  float clear_color[4] = {0.0F, 0.0F, 0.0F, 1.0F};
  /// Headless only: use an RGBA8 sRGB target instead of RGBA8 UNORM. Clear/layer RGB values
  /// are linear with this option; readback contains sRGB bytes. Alpha remains linear.
  bool headless_framebuffer_srgb = false;
};

// The set of graphics pipelines to pre-compile is enumerated by the backend from the declared
// LayerStack (given to allocate()) and the surface's output format (given to build()); it is
// not described at this seam. Pipeline caching is likewise a backend-internal concern.

// FrameCommands and SyncToken are defined in "holoviz_op/backends/handles.hpp"
// (the shared opaque-handle header, so PresentBackend can depend on the handles without
// pulling in all of this header).

/// One declared layer's live data for the current frame. `layer_index` indexes into the
/// LayerStack passed to allocate(); `frame_data` is an opaque pointer to the runtime frame
/// handle read from that layer's input port this cycle (e.g. a GPU image or a GPU-resident
/// geometry batch). The concrete runtime frame-handle types are specified separately; this
/// seam only needs to move them, not interpret them.
struct LayerFrame {
  uint32_t layer_index = 0;
  const void* frame_data = nullptr;
};

/// Per-frame layer data to record, indexed by declaration order.
struct RenderList {
  std::vector<LayerFrame> layers;
};

/// Compositing order applied during \c record(). Index 0 is recorded first (backmost);
/// higher indices composite over it. Kept separate from \c RenderList so re-ordering never
/// requires new layer data.
///
/// \p order contains indices into the \c LayerStack / \c RenderList in composite order.
/// An empty \p order means "use declaration order unchanged."
/// Every entry must be in [0, LayerStack::layers().size()) and unique. Backends reject
/// out-of-range and duplicate indices with kInvalidArgument.
struct CompositeSpec {
  std::vector<uint32_t>
      order;  ///< layer indices in desired composite order; empty = declaration order
};

/// Context passed to \c submit(). The \p frame_index here must equal the \p frame_index
/// passed to the corresponding \c begin_frame() call for this frame — it identifies which
/// ring slot's resources to release/fence. Mismatching the two is a precondition violation.
struct SubmitContext {
  uint32_t frame_index = 0;
};

/// Static backend identity, for logging/diagnostics only.
struct BackendInfo {
  std::string name;
};

/// Allocation/usage figures, for debug/diagnostic use only (not a control surface).
/// Units are in the field names: the two pool figures are static allocation sizes fixed
/// at allocate() time (in bytes); the glyph figure is a genuine per-frame high-water mark
/// (in glyphs — text draws through push constants, so there is no text byte pool).
struct AllocationStats {
  std::size_t texture_pool_bytes = 0;
  std::size_t geometry_pool_bytes = 0;
  std::size_t text_glyph_high_water = 0;
};

/// \c RenderBackend is the ownership seam for rasterization: everything above it (the
/// HoloViz operator, layer declarations, compositing order) is backend-agnostic; everything
/// below it is an implementation detail of the concrete backend (native Vulkan, CUDA-native,
/// ...). Nothing above this seam may depend on a concrete backend type.
///
/// Lifecycle:
/// - \c allocate() is called once at operator \c initialize() time: the device is available,
///   the display surface is not required yet. All GPU resources implied by the declared
///   \c LayerStack must be allocated here; no GPU allocation is permitted after this call.
/// - \c build() is called once at operator \c start() time, after the display surface exists:
///   its format (\p output_format) is needed for render-pass-compatible pipeline objects. It
///   pre-compiles every pipeline the declared layer set can need. Pipeline caching is a
///   backend-internal concern, not part of this seam.
/// - \c begin_frame() / \c record() / \c submit() are called every frame. None may perform
///   GPU allocation, pipeline compilation, or a blocking CPU sync call — with one
///   documented exception: the first \c begin_frame() after a swapchain recreation
///   re-creates the per-target framebuffers (see \c register_targets()); every other frame
///   is allocation-free.
/// - On any per-frame error (a failed \c record() or \c submit()), \c abort_frame() before
///   reusing the slot is always sound. The one narrower alternative is record()'s
///   documented prevalidation rejections, which leave the frame open (see \c record()).
class RenderBackend {
 public:
  virtual ~RenderBackend() = default;

  /// One-time GPU resource allocation for the full declared layer set.
  ///
  /// Argument-validation failures (rejected before anything is created) leave the backend
  /// untouched and retryable. A failure past that point is terminal for the backend
  /// instance: the partially created state is released, and every later call — including
  /// another allocate() — fails with \c kInvalidState. Create a fresh backend to retry
  /// with a corrected layer set.
  [[nodiscard]] virtual expected<void> allocate(const LayerStack& layers,
                                                const RenderConfig& config) = 0;

  /// Pre-compile the pipelines implied by the declared LayerStack, targeting `output_format`
  /// (the surface's format, for render-pass-compatible pipeline objects). Pipeline caching is
  /// a backend-internal concern and is not exposed at this seam.
  [[nodiscard]] virtual expected<void> build(ImageFormat output_format) = 0;

  /// Tear down the swapchain and release the surface reference. Blocks (bounded by
  /// `timeout`) until the device is done with the swapchain images. The destructor calls
  /// this and logs failures. On failure (e.g. the timeout expired) the device may still
  /// reference the swapchain images and any in-flight frame's FrameData device memory, so
  /// the caller's retention duty continues (see frame_data.hpp): retry with a longer
  /// timeout, or treat that memory as unreclaimable for the rest of the process.
  [[nodiscard]] virtual expected<void> destroy(std::chrono::nanoseconds timeout) = 0;

  /// Open this frame's pre-allocated command-recording context. No allocation.
  ///
  /// Backends may bound how many \c FrameCommands handles the caller retains per slot (the
  /// HoloViz backend: the current and the immediately previous issuance). With both still
  /// retained, begin_frame() fails with \c kBackpressured and the slot untouched — it
  /// never allocates a replacement handle; release the older handle and retry.
  ///
  /// \p acquired carries the swapchain image identity and acquire dependency returned by
  /// \c PresentBackend::acquire(): the backend uses \p acquired.image_index to select the
  /// correct framebuffer and must wait on \p acquired.image_ready before writing any pixel.
  /// \p frame_index is the frame-in-flight slot (independent of the swapchain image index)
  /// and selects which per-frame descriptor pool, command buffer, etc. to use.
  ///
  /// Returning successfully also guarantees the slot's previous frame — including its
  /// asynchronous reads of that frame's FrameData device memory — has completed; the
  /// caller may recycle that frame's producer buffers now (see frame_data.hpp). An error
  /// return (notably kBackpressured, when the slot-recycle wait exceeds
  /// RenderConfig::backpressure_timeout_ns) implies no such quiescence: the previous frame
  /// may still be reading its FrameData device memory, so the caller's retention duty
  /// continues until a later successful begin_frame() on this frame_index, a successful
  /// abort_frame(), or a successful destroy().
  [[nodiscard]] virtual expected<FrameCommands> begin_frame(const AcquireToken& acquired,
                                                            uint32_t frame_index) = 0;

  /// Pre-create per-target rendering state (e.g. framebuffers) for every current
  /// swapchain image, so begin_frame() honors its no-allocation contract from the first
  /// frame. Call once after build(), with the token from \c PresentBackend::targets().
  /// After a swapchain recreation the backend re-creates that state on the first
  /// begin_frame() of the new generation — the one documented allocation exception, on
  /// the recreation frame only (the recreation itself already allocated). Default: no-op
  /// success, for headless configurations and backends that cache nothing per target.
  [[nodiscard]] virtual expected<void> register_targets(const SyncToken& /*targets*/) { return {}; }

  /// Record draw commands for every layer in `layers`, in the order given by `composite`.
  /// Implementations must not allocate; they read only from pools sized in allocate().
  ///
  /// Error recovery: a \c kInvalidArgument failure is a prevalidation rejection —
  /// implementations guarantee the rejected layer recorded and enqueued nothing, so the
  /// frame stays open with every earlier layer's recording intact; the caller may record
  /// corrected data for that layer, submit without it, or abort. Any other record()
  /// failure may leave partial per-layer state behind, and \c abort_frame() is the only
  /// sound recovery from it.
  [[nodiscard]] virtual expected<void> record(FrameCommands& cmds, const RenderList& layers,
                                              const CompositeSpec& composite) = 0;

  /// Submit the recorded commands. Returns a token the caller hands to
  /// PresentBackend::present() to sequence presentation after rendering completes.
  /// submit() does not wait for the frame's asynchronous FrameData reads; see
  /// frame_data.hpp for the device-memory lifetime rule.
  [[nodiscard]] virtual expected<SyncToken> submit(const FrameCommands& cmds,
                                                   const SubmitContext& ctx) = 0;

  /// Abandon a frame opened by begin_frame() without submitting it: discard the
  /// recorded commands, complete or cancel the frame's staged uploads (after this
  /// returns successfully, no FrameData memory from this frame is referenced), and
  /// reopen the slot so the next begin_frame() on it succeeds. This is the universally
  /// sound recovery from a failed record() or submit() — and the only one for submit()
  /// failures and non-prevalidation record() failures (record()'s kInvalidArgument
  /// rejections alone leave the frame open; see record()); may block briefly (bounded like
  /// begin_frame()'s recycle wait). Calling it on a frame that is not open (never
  /// begun, already submitted/aborted, or a stale handle from an earlier issuance of the
  /// slot — a later begin_frame() superseded it, and the live issuance is unaffected) is
  /// a no-op success, so error-path callers can invoke it unconditionally; a handle not
  /// minted by this backend still fails with kInvalidArgument. On an error return the
  /// frame's staged uploads may
  /// still be in flight and its FrameData memory may still be referenced; the caller's
  /// retention duty continues (see frame_data.hpp) — retry abort_frame(), or fall back
  /// to destroy().
  ///
  /// Aborting does not release the swapchain image: pass the frame's image_index to
  /// PresentBackend::present() with a null render_done (an "empty present") to hand
  /// the image back, or let the next swapchain recreation reclaim it.
  [[nodiscard]] virtual expected<void> abort_frame(FrameCommands& cmds) = 0;

  /// \return static backend identity, for logging/diagnostics.
  [[nodiscard]] virtual expected<BackendInfo> info() const = 0;

  /// \return current pool high-water marks, for debug/diagnostic use only.
  [[nodiscard]] virtual expected<AllocationStats> stats() const = 0;
};

}  // namespace holoscan::visualization::backends
