/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Renderer: a HoloViz renderer for Dear ImGui draw data. ImGui tessellates on
// the CPU; the backend copies the resulting vertex/index lists into exported
// buffers with CUDA (cuMemcpyHtoDAsync on the desc stream) — HoloViz has no
// transfer queues, so per-frame data moves through CUDA — and records the
// draws into the caller's already-begun render scope. ImGui types appear here
// only as forward declarations: including this header does not require the
// ImGui headers. Callers that build their own UI through render_ui() include
// <holoviz/imgui/ui.hpp>, which pulls in the bundled ImGui API.

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include <cuda.h>

#include <holoviz/holoviz_export.h>

#include <holoviz/command_list.hpp>
#include <holoviz/common.hpp>
#include <holoviz/expected.hpp>
#include <holoviz/image.hpp>  // Image, SamplerDesc (register_image)

struct ImDrawData;
struct ImFontAtlas;
struct ImGuiContext;

namespace holoviz {
class Device;
class Window;
}  // namespace holoviz

namespace holoviz::imgui {

struct RendererDesc {
  /// Stream all per-frame host-to-device copies and semaphore operations run on. The stream
  /// must belong to the device's CUDA context, and that context must be current on the calling
  /// thread for every per-frame call (the same contract HoloViz's exported resources follow).
  /// Must be non-null — creation fails with kInvalidArgument otherwise; the CUDA legacy
  /// default stream is not supported.
  CUstream stream = nullptr;
  /// Vertex/index buffer ring depth. Must cover the maximum number of frames whose draws can be
  /// in flight at once; swapchain image count + 1 is a safe choice. Callers keep one
  /// SubmitSyncToken per slot for begin_frame (see begin_frame()).
  uint32_t frames_in_flight = 4;
  /// Initial per-slot capacities, in elements. Buffers grow geometrically on overflow — a cold
  /// path that recreates exported memory and host-waits, so size generously.
  size_t initial_vertex_capacity = 4096;
  size_t initial_index_capacity = 8192;
  /// Optional TTF font loaded into the atlas at creation; nullptr keeps the ImGui default font.
  const char* ttf_path = nullptr;
  /// Pixel size \p ttf_path is rasterized into the atlas at; ignored without a ttf_path.
  float ttf_size_px = 32.0F;
  /// Optional caller-populated font atlas shared with the created context; the caller keeps
  /// ownership and must keep it alive. nullptr gives the context its own atlas.
  ImFontAtlas* font_atlas = nullptr;
  /// Optional label used for the backend's pipeline and error messages; nullptr selects a
  /// default.
  const char* debug_name = nullptr;
};

/// What a frame's upload produced. Add the token to the SubmitDesc that carries the command
/// list the draws were recorded into (SubmitDesc::wait_external_sync_tokens).
struct FrameSync {
  ExternalSyncToken upload_done;  ///< Signaled when the frame's CUDA copies complete.
};

namespace detail {
class RendererImpl;
}

/// Renders Dear ImGui draw data through HoloViz. Per-frame flow:
///
///   backend.begin_frame(frame, slot_tokens[frame % frames_in_flight]);  // gate slot reuse
///   ... build ImGui draw data ...
///   auto sync = backend.upload_draw_data(dd);      // CUDA copies + semaphore signal
///   ... caller: begin()/begin_render() ...
///   backend.record_draws(cl, framebuffer_extent);  // inside the caller's render scope
///   ... caller: submit with sync.upload_done in wait_external_sync_tokens ...
///   slot_tokens[frame % frames_in_flight] = submit_token;  // the slot's next reuse gate
///
/// or, for arbitrary debug UI, replace the middle two steps with render_ui(). The Device must
/// outlive the backend and stay at a stable address; destroy the backend before the device.
class Renderer {
 public:
  HOLOVIZ_EXPORT explicit Renderer(std::unique_ptr<detail::RendererImpl> impl);
  HOLOVIZ_EXPORT Renderer();
  HOLOVIZ_EXPORT Renderer(Renderer&&) noexcept;
  HOLOVIZ_EXPORT Renderer& operator=(Renderer&&) noexcept;
  Renderer(const Renderer&) = delete;
  Renderer& operator=(const Renderer&) = delete;
  HOLOVIZ_EXPORT ~Renderer();

  explicit operator bool() const { return impl_ != nullptr; }

  /// Release the underlying resources now instead of at destruction. The GPU must be done with
  /// the backend's draws (e.g. Device::wait_idle() or a wait on the last consuming submit)
  /// before calling this; buffer destruction does not wait for in-flight work. \p timeout
  /// bounds the wait for the backend's own pending CUDA copies to retire: if they do not, the
  /// call returns ErrorCode::kBackpressured with the backend fully intact — drain the stream
  /// (or pass a longer timeout) and retry. On success, and on any error other than
  /// kBackpressured, the handle becomes empty. A terminal backend (one whose stream-work
  /// retirement could not be proved) deliberately leaks its CUDA-referenced resources instead
  /// of freeing them under possibly-live work and has poisoned the owning Device, whose
  /// teardown then leaks its native state; that leak is reported as an error while the handle
  /// still empties. Safe to call on an empty handle (no-op success). The destructor calls this
  /// automatically, escalates a backpressured result to the terminal leak (it cannot leave the
  /// resources alive), and logs any error.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> destroy(
      std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero());

  /// Return the ImGui context this backend created. Callers that build UI directly wrap their
  /// ImGui calls in ImGui::SetCurrentContext(ctx) — or use render_ui(), which does it for them.
  [[nodiscard]] HOLOVIZ_EXPORT expected<ImGuiContext*> context() const;

  /// Start a frame in buffer slot `renderer_slot % frames_in_flight` and make CUDA wait
  /// (stream-side, the host never blocks) until the GPU is done with the submit
  /// \p slot_last_use identifies. The token must come from the submit that last consumed THIS
  /// renderer's draws for the same slot: keep a SubmitSyncToken array sized frames_in_flight,
  /// pass the frame counter as \p renderer_slot and the array element it selects as the token
  /// (a default token on the first lap), and store each submit's token back into that element.
  /// Do not key the tokens by swapchain image — with image_count != frames_in_flight the wait
  /// would target another slot's consumer and the slot could be rewritten while the GPU still
  /// reads it. On failure no state changes; the call is retryable with the same arguments.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> begin_frame(uint32_t renderer_slot,
                                                          SubmitSyncToken slot_last_use);

  /// Abandon a begun frame without uploading: clear the begin_frame() state so the next
  /// begin_frame() starts clean after a mid-frame error. The slot rotation and stream-side
  /// wait begin_frame() performed are harmless to leave behind. Idempotent — success when
  /// no frame is begun.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> cancel_frame();

  /// Copy \p draw_data's vertex/index lists into the current slot's exported buffers
  /// (cuMemcpyHtoDAsync on the desc stream, growing the buffers if needed) and enqueue the
  /// semaphore signal. The data is staged internally, so \p draw_data may be invalidated after
  /// this returns. ImDrawList user callbacks are not executed (logged and skipped).
  /// The full display transform is honored: vertex positions are interpreted in the draw
  /// data's display space (DisplayPos/DisplaySize) and clip rectangles are rebased to
  /// DisplayPos and scaled by FramebufferScale into framebuffer pixels. A non-empty frame
  /// with non-positive DisplaySize or FramebufferScale returns kInvalidArgument.
  [[nodiscard]] HOLOVIZ_EXPORT expected<FrameSync> upload_draw_data(const ImDrawData* draw_data);

  /// Record the draws for the data passed to the immediately preceding upload_draw_data().
  /// Must be called inside the caller's begin_render/end_render scope. Sets viewport, scissor,
  /// pipeline, bindings, and push data — re-set any caller state afterwards. The viewport
  /// spans all of \p framebuffer_extent and the vertex projection maps the uploaded frame's
  /// display rect (DisplayPos/DisplaySize) onto it, so \p framebuffer_extent should equal
  /// DisplaySize * FramebufferScale for pixel-exact output.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> record_draws(CommandList& command_list,
                                                           Extent2D framebuffer_extent);

  /// Debug UI-callback mode: build arbitrary ImGui UI. Makes the backend's context current,
  /// sets the display size, runs NewFrame(), invokes \p build (which calls ImGui functions
  /// freely — see <holoviz/imgui/ui.hpp>), then Render() and upload_draw_data(). Replaces
  /// the upload_draw_data() step: still call begin_frame() before and record_draws() after.
  /// Correctness over efficiency — this is a debug path. Widgets only react to input when
  /// forward_input() is wired; without it the UI still renders (output-only overlays).
  [[nodiscard]] HOLOVIZ_EXPORT expected<FrameSync> render_ui(Extent2D framebuffer_extent,
                                                             const std::function<void()>& build);

  /// Register \p image for use with ImGui::Image()/ImDrawList::AddImage(): pre-bakes one
  /// descriptor sampling it and returns the value to pass as ImTextureID. Setup-path only —
  /// this allocates a descriptor pool; do not call it in the frame loop. The image is
  /// borrowed: keep it alive until unregister_image() (plus completion of the last submit
  /// that sampled it) or until the backend is destroyed. Draws referencing an unregistered
  /// id are skipped with a warning, never a frame error.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void*> register_image(const Image& image,
                                                              const SamplerDesc& sampler = {});

  /// Remove a registration made by register_image(). The GPU must be done with draws that
  /// sampled the image (same contract as destroy()). kNotFound for an unknown id;
  /// kInvalidArgument for the backend-owned font-atlas id.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> unregister_image(
      void* texture_id, std::chrono::nanoseconds timeout = std::chrono::nanoseconds::zero());

  /// Event-driven input: install GLFW callbacks on \p window that feed mouse position,
  /// buttons, scroll, keyboard, character input, and focus into this backend's ImGui
  /// context as the events arrive (during poll_window_events(), main thread). One
  /// backend per window: a second install (or a window whose input is already claimed)
  /// fails with kInvalidState. The registration follows the underlying window, not the
  /// handle object: moving \p window into another Window while callbacks are
  /// installed is safe. The window must outlive the backend or
  /// uninstall_input_callbacks() must be called first; destroy() uninstalls automatically.
  /// Only implemented for WindowSystem::kGlfw; others return kNotImplemented.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> install_input_callbacks(Window& window);

  /// Remove callbacks installed by install_input_callbacks(). No-op success when none are
  /// installed, so teardown paths can call it unconditionally.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> uninstall_input_callbacks();

  /// DEPRECATED polling fallback — prefer install_input_callbacks(): polling misses scroll
  /// and keyboard entirely and loses press+release pairs that happen between polls. Pushes
  /// \p window's current mouse state (position and the three primary buttons) into the
  /// ImGui context; call once per frame after poll_window_events(). Returns
  /// kInvalidState while callbacks are installed (double-feeding would corrupt the edge
  /// tracking). Only implemented for WindowSystem::kGlfw; others return kNotImplemented.
  [[nodiscard]] HOLOVIZ_EXPORT expected<void> forward_input(const Window& window);

  /// Renderer-internal access; not part of the stable API surface.
  /// Precondition: the handle is non-empty.
  detail::RendererImpl& impl() { return *impl_; }
  const detail::RendererImpl& impl() const { return *impl_; }

 private:
  std::unique_ptr<detail::RendererImpl> impl_;
};

/// Create an ImGui backend on \p device: creates an ImGui context, rasterizes the font atlas
/// and uploads it once (Device::upload_to_image), compiles the built-in Slang shader, and
/// creates the pipeline, descriptor, exported buffer ring, and ExternalSemaphore(desc.stream).
/// Returns kInvalidArgument — before any resource is created — for a null desc.stream, a zero
/// desc.frames_in_flight, or a ttf_path combined with a font_atlas.
/// The device must outlive the backend and stay at a stable address.
[[nodiscard]] HOLOVIZ_EXPORT expected<Renderer> create_renderer(Device& device,
                                                                const RendererDesc& desc);

}  // namespace holoviz::imgui
