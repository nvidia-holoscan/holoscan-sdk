/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Concrete implementation behind the Renderer handle. A single class, not
// a backend-plug-point like the core detail interfaces: HoloViz already
// abstracts the graphics backends, so one ImGui renderer covers them all.
// This header includes the real ImGui API and is deliberately not installed.
// It lives outside detail/ so unit tests may include it as a white-box seam
// (same pattern as the other module-private src/ headers).

#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include <cuda.h>

#include <holoviz/imgui/vendor/imgui.h>

#include <holoviz/buffer.hpp>
#include <holoviz/command_list.hpp>
#include <holoviz/common.hpp>
#include <holoviz/descriptor.hpp>
#include <holoviz/device.hpp>
#include <holoviz/expected.hpp>
#include <holoviz/external_semaphore.hpp>
#include <holoviz/image.hpp>
#include <holoviz/imgui/renderer.hpp>
#include <holoviz/pipeline.hpp>

struct GLFWwindow;

namespace holoviz::detail {
class WindowImpl;
}  // namespace holoviz::detail

namespace holoviz::imgui::detail {

class RendererImpl {
 public:
  static expected<std::unique_ptr<RendererImpl>> create(Device& device, const RendererDesc& desc);

  RendererImpl() = default;
  RendererImpl(const RendererImpl&) = delete;
  RendererImpl& operator=(const RendererImpl&) = delete;
  RendererImpl(RendererImpl&&) = delete;
  RendererImpl& operator=(RendererImpl&&) = delete;
  ~RendererImpl();

  expected<void> destroy(std::chrono::nanoseconds timeout);

  expected<ImGuiContext*> context() const;

  expected<void> begin_frame(uint32_t renderer_slot, SubmitSyncToken slot_last_use);
  expected<void> cancel_frame();

  expected<FrameSync> upload_draw_data(const ImDrawData* draw_data);
  expected<void> record_draws(CommandList& command_list, Extent2D framebuffer_extent);
  expected<FrameSync> render_ui(Extent2D framebuffer_extent, const std::function<void()>& build);
  expected<void> forward_input(const Window& window);
  expected<void*> register_image(const Image& image, const SamplerDesc& sampler);
  expected<void> unregister_image(void* texture_id, std::chrono::nanoseconds timeout);
  expected<void> install_input_callbacks(Window& window);
  expected<void> uninstall_input_callbacks();

  /// Make this backend's context current, set the display size and delta time, and run
  /// ImGui::NewFrame(). Internal building block shared by render_ui() and the TextRenderer
  /// implementation.
  expected<void> new_frame(Extent2D framebuffer_extent);

 private:
  struct Slot {
    Buffer vertex_buffer;
    Buffer index_buffer;
    CUdeviceptr vertex_ptr = 0;
    CUdeviceptr index_ptr = 0;
    size_t vertex_capacity_bytes = 0;
    size_t index_capacity_bytes = 0;
    // Most recent begin_frame token for this slot; the growth path host-waits on it before
    // destroying the outgrown buffers.
    SubmitSyncToken reuse_token;
    // Pinned staging for this slot, laid out [vertex region | index region] with the index
    // region at offset vertex_capacity_bytes. Pinned + async HtoD means the copy may still
    // be reading when the slot comes around again — staging_free (recorded after the copy
    // pair) is host-synced before the block is rewritten, freed, or grown. begin_frame's
    // semaphore wait is stream-side only and does NOT protect this block.
    std::byte* staging = nullptr;
    CUevent staging_free = nullptr;
  };

  struct DrawCommand {
    Rect2D scissor;  // clamped to non-negative origin; intersected with the framebuffer
                     // at record time
    uint32_t index_count = 0;
    uint32_t first_index = 0;
    int32_t vertex_offset = 0;
    uint64_t texture_id = 0;  // ImDrawCmd::TextureId; 0 falls back to the font atlas
  };

  expected<void> ensure_slot_capacity(Slot& slot, size_t vertex_bytes, size_t index_bytes);

  /// Record staging_free over the stream work just enqueued for \p slot (copies and/or the
  /// semaphore signal). If the record fails, drain the stream instead so the work retires
  /// now (slot coherent, error retryable); if the drain also fails, quarantine the backend.
  expected<void> refresh_staging_fence(Slot& slot);

  /// Enter the terminal retirement-unknown state: retirement is a property of the shared
  /// stream, so the whole backend goes terminal — every entry point rejects new frames, all
  /// CUDA-referenced resources (slot buffers, pinned staging, the semaphore import) are
  /// frozen until destroy() leaks them, and the owning device is poisoned
  /// (DeviceImpl::report_leaked_resources()) so it never destroys the leaked children's
  /// native parents. Allocation-free and no-throw: callers build diagnostics only after
  /// this returns, so a failed allocation can never unwind a live resource.
  void quarantine_backend() noexcept;

  Device* device_ = nullptr;
  CUstream stream_ = nullptr;
  ImGuiContext* context_ = nullptr;
  Image font_image_;
  Pipeline pipeline_;
  /// One pre-baked descriptor per drawable texture, keyed by opaque integer id (the value
  /// callers pass as ImTextureID). Ids are integers, never pointers to movable handles.
  /// kFontTextureId is minted internally at create(); register_image() mints the rest.
  static constexpr uint64_t kFontTextureId = 1;
  std::unordered_map<uint64_t, Descriptor> textures_;
  uint64_t next_texture_id_ = kFontTextureId + 1;
  ExternalSemaphore semaphore_;
  // Terminal state: stream work was enqueued but neither the fence record nor a stream
  // drain (nor, at teardown, a bounded fence wait) could prove it retired. The condition
  // is stream-wide, so it poisons the whole backend, not one slot — see
  // quarantine_backend().
  bool stream_retirement_unknown_ = false;
  std::vector<Slot> slots_;
  uint32_t current_slot_ = 0;
  bool frame_begun_ = false;

  std::vector<DrawCommand> draw_commands_;
  size_t frame_vertex_bytes_ = 0;
  size_t frame_index_bytes_ = 0;
  // Display transform captured with the uploaded frame: vertex positions stay in ImGui
  // display coordinates (record_draws derives the clip-space projection from these), while
  // scissors are pre-scaled to framebuffer pixels at upload time via FramebufferScale.
  ImVec2 frame_display_pos_{0.0F, 0.0F};
  ImVec2 frame_display_size_{0.0F, 0.0F};

  std::chrono::steady_clock::time_point last_new_frame_time_{};
  std::array<bool, 3> mouse_down_prev_ = {{false, false, false}};

  // Callback-based input (install_input_callbacks): the registered window and its GLFW
  // handle, cached for window->framebuffer coordinate scaling inside the handlers.
  // Stored as the heap-stable WindowImpl, not the public Window wrapper: the wrapper
  // is movable, and callbacks must remain removable after the caller moves the handle.
  holoviz::detail::WindowImpl* input_window_impl_ = nullptr;
  GLFWwindow* input_glfw_window_ = nullptr;

  // GLFW->ImGui trampolines; `user` is the RendererImpl*. Public-ish (static, exposed
  // via this internal header) so tests can drive them without a real window system.
 public:
  static void on_cursor_pos(void* user, double x, double y);
  static void on_mouse_button(void* user, int button, int action, int mods);
  static void on_scroll(void* user, double dx, double dy);
  static void on_key(void* user, int key, int scancode, int action, int mods);
  static void on_char(void* user, unsigned int codepoint);
  static void on_focus(void* user, int focused);

  // Pinned-staging fault-injection seam (test_imgui_renderer_fault_injection.cpp reassigns
  // these to failing fakes); production code never reassigns them.
  CUresult (*host_alloc_fn_)(void**, size_t, unsigned int) = &cuMemHostAlloc;
  CUresult (*host_free_fn_)(void*) = &cuMemFreeHost;
  CUresult (*host_memcpy_fn_)(CUdeviceptr, const void*, size_t, CUstream) = &cuMemcpyHtoDAsync;
  CUresult (*event_record_fn_)(CUevent, CUstream) = &cuEventRecord;
  CUresult (*event_query_fn_)(CUevent) = &cuEventQuery;
  CUresult (*event_sync_fn_)(CUevent) = &cuEventSynchronize;
  CUresult (*stream_sync_fn_)(CUstream) = &cuStreamSynchronize;
};

}  // namespace holoviz::imgui::detail
