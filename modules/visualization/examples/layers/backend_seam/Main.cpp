// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The RenderBackend/PresentBackend seam driven end to end: this is the frame loop the
// HolovizOp operator will run, and the living documentation of the acquire -> render ->
// present handoff between the two backends. Unlike the other examples it never touches the
// RHI on the frame path — everything flows through the two seam interfaces:
//
//   window -> make_surface_handle -> PresentBackend::create_swapchain
//   RenderBackend::allocate(declared layers) -> build(swapchain_info's target)
//   per frame: acquire -> begin_frame(image_index, image_ready) -> record -> submit
//              -> present(render_done, present_id) [-> wait_for_present]
//
// Run with --frames N to exit after N frames (used for automated smoke testing).

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoviz/backends/rhi_render_backend.hpp>
#include <holoviz/backends/wsi_present_backend.hpp>
#include <holoviz/frame_data.hpp>
#include <holoviz/layer_desc.hpp>
#include <holoviz/present_backend.hpp>
#include <holoviz/render_backend.hpp>
#include <holoviz/rhi.hpp>

namespace {

/// Exit-on-error unwrap for example code (the library itself never aborts).
template <typename T>
T unwrap(holoviz::expected<T> result, const char* what) {
  if (!result) {
    std::fprintf(stderr, "%s failed: %s\n", what, result.error().message.c_str());
    std::exit(1);
  }
  return std::move(*result);
}

void unwrap_void(holoviz::expected<void> result, const char* what) {
  if (!result) {
    std::fprintf(stderr, "%s failed: %s\n", what, result.error().message.c_str());
    std::exit(1);
  }
}

}  // namespace

int main(int argc, char** argv) {
  using namespace holoviz;
  namespace rhi = holoviz::rhi;

  uint64_t max_frames = 0;  // 0 = run until the window closes
  for (int i = 1; i < argc; ++i) {
    if (std::string_view(argv[i]) == "--frames" && i + 1 < argc) {
      max_frames = std::strtoull(argv[++i], nullptr, 10);
    }
  }

  // --- Setup: window, surface handle, device, the two backends. ---
  auto window = unwrap(
      rhi::create_window({.title = "HoloViz backend demo", .size = {800, 600}, .resizable = true}),
      "create_window");
  auto surface_handle = unwrap(backends::make_surface_handle(window), "make_surface_handle");
  auto device = unwrap(rhi::create_device_for_surface(
                           {}, *std::static_pointer_cast<rhi::Surface>(surface_handle.impl)),
                       "create_device_for_surface");

  auto present = unwrap(backends::create_wsi_present_backend(device), "create_wsi_present_backend");
  auto render =
      unwrap(backends::create_rhi_render_backend(device, {}), "create_rhi_render_backend");

  // --- Declare the scene once: a geometry overlay under a text layer. ---
  LayerStack layers;
  layers.add_geometry_layer("overlay",
                            GeometryLayerDesc{{GeometryTopology::kRectangleList,
                                               GeometryTopology::kOvalList,
                                               GeometryTopology::kCrossList},
                                              64,
                                              3.0F});
  layers.add_text_layer("labels", TextLayerDesc{.max_glyphs = 512});

  // allocate() before the swapchain exists (the operator's initialize() phase)...
  RenderConfig render_config;
  render_config.max_frames_in_flight = 4;
  unwrap_void(render->allocate(layers, render_config), "RenderBackend::allocate");

  // ...then the display path, then build() against the swapchain's format (start() phase).
  auto swapchain_result = present->create_swapchain(SwapchainDesc{
      .surface = surface_handle, .preferred_mode = PresentMode::kAuto, .image_count = 3});
  if (!swapchain_result) {
    std::fprintf(stderr, "create_swapchain failed: %s\n", swapchain_result.error().message.c_str());
    return 1;
  }

  SwapchainInfo swapchain_info =
      unwrap(present->swapchain_info(), "PresentBackend::swapchain_info");
  std::printf("swapchain: %ux%u, %u images, mode %u, format %u\n",
              swapchain_info.extent.width,
              swapchain_info.extent.height,
              swapchain_info.image_count,
              static_cast<unsigned>(swapchain_info.active_mode),
              static_cast<unsigned>(swapchain_info.format));
  if (render_config.max_frames_in_flight < swapchain_info.image_count) {
    std::fprintf(stderr, "max_frames_in_flight must cover the swapchain image count\n");
    return 1;
  }
  unwrap_void(render->build(swapchain_info.format), "RenderBackend::build");
  // Pre-create the per-target framebuffers on the setup path, so the frame loop's
  // begin_frame() allocates nothing. Extent-only recreations re-warm automatically on the
  // first frame of the new generation; kImageCountChanged re-registers below.
  unwrap_void(render->register_targets(unwrap(present->targets(), "PresentBackend::targets")),
              "RenderBackend::register_targets");

  const PresentInfo present_info = present->info();
  std::printf("present backend '%s': present_id=%d present_wait=%d\n",
              present_info.name.c_str(),
              present_info.has_present_id,
              present_info.has_present_wait);

  // --- The frame loop the operator will run. ---
  PresentId present_id = 0;
  uint64_t frame_counter = 0;
  char fps_text[64] = "warming up...";
  auto fps_window_start = std::chrono::steady_clock::now();
  uint64_t fps_window_frames = 0;

  // Window resizes are handled inside the backends (the window is resizable, so the seam's
  // acquire()/present() recreate the swapchain automatically). Only kImageCountChanged
  // needs frame-loop code: the render backend's slot ring must still cover the new count.
  const auto handle_image_count_change = [&]() -> bool {
    const auto info = present->swapchain_info();
    if (!info || info->image_count > render_config.max_frames_in_flight) {
      return false;
    }
    swapchain_info = *info;
    // Eager re-warm at the rebuild moment (the begin_frame() re-warm would cover it one
    // frame later; doing it here keeps even that frame allocation-free).
    const auto targets = present->targets();
    if (targets.has_value()) {
      unwrap_void(render->register_targets(*targets), "RenderBackend::register_targets");
    }
    std::printf("swapchain image count changed: %u images\n", swapchain_info.image_count);
    return true;
  };

  while (max_frames == 0 || frame_counter < max_frames) {
    rhi::poll_window_events();
    if (unwrap(window.close_requested(), "close_requested")) {
      break;
    }

    auto acquired = present->acquire(std::chrono::milliseconds(100));
    if (!acquired) {
      if (acquired.error().code == ErrorCode::kBackpressured) {
        continue;  // display busy or window minimized; retry
      }
      if (acquired.error().code == ErrorCode::kOutOfDate) {
        continue;  // a fast resize outran the bounded recreation attempts; retry
      }
      if (acquired.error().code == ErrorCode::kImageCountChanged) {
        if (!handle_image_count_change()) {
          std::fprintf(stderr, "swapchain image count exceeds frames_in_flight\n");
          return 1;
        }
        continue;
      }
      std::fprintf(stderr, "acquire failed: %s\n", acquired.error().message.c_str());
      return 1;
    }

    auto cmds = unwrap(render->begin_frame(acquired.value(), acquired->image_index),
                       "RenderBackend::begin_frame");

    char counter_text[64];
    std::snprintf(counter_text,
                  sizeof(counter_text),
                  "frame %llu",
                  static_cast<unsigned long long>(frame_counter));
    const TextRun runs[] = {
        {0.05F, 0.08F, 32.0F, 0xFFFFFFFFU, "HoloViz backend seam"},
        {0.05F, 0.20F, 24.0F, 0xFF40C0FFU, counter_text},
        {0.05F, 0.30F, 24.0F, 0xFF80FF80U, fps_text},
    };
    const TextFrameData text{runs};

    // A "detection" box orbiting the center, with a cross on its target.
    const float phase = static_cast<float>(frame_counter % 240) / 240.0F * 6.2831853F;
    const float cx = 0.5F + 0.25F * std::cos(phase);
    const float cy = 0.55F + 0.25F * std::sin(phase);
    const float rect_coords[] = {cx - 0.08F, cy - 0.08F, cx + 0.08F, cy + 0.08F};
    const float oval_coords[] = {0.5F, 0.55F, 0.30F, 0.30F};
    const float cross_coords[] = {cx, cy, 0.04F};
    const GeometryBatch batches[] = {
        {GeometryTopology::kOvalList, oval_coords, 1, {0.3F, 0.5F, 1.0F, 1.0F}, 2.0F},
        {GeometryTopology::kRectangleList, rect_coords, 1, {0.2F, 1.0F, 0.4F, 1.0F}, 3.0F},
        {GeometryTopology::kCrossList, cross_coords, 1, {1.0F, 0.4F, 0.3F, 1.0F}},
    };
    const GeometryFrameData geometry{batches};

    RenderList list;
    list.layers.push_back(LayerFrame{0, &geometry});
    list.layers.push_back(LayerFrame{1, &text});
    unwrap_void(render->record(cmds, list, CompositeSpec{}), "RenderBackend::record");

    auto render_done =
        unwrap(render->submit(cmds, SubmitContext{acquired->image_index}), "RenderBackend::submit");

    auto presented = present->present(PresentArgs{.image_index = acquired->image_index,
                                                  .acquire_issuance = acquired->issuance,
                                                  .render_done = render_done,
                                                  .present_id = ++present_id});
    if (!presented) {
      if (presented.error().code == ErrorCode::kFrameDropped) {
        continue;  // a resize consumed the frame before display; re-acquire at the new size
      }
      if (presented.error().code == ErrorCode::kImageCountChanged) {
        if (!handle_image_count_change()) {
          std::fprintf(stderr, "swapchain image count exceeds frames_in_flight\n");
          return 1;
        }
        continue;  // that frame was dropped too
      }
      std::fprintf(stderr, "present failed: %s\n", presented.error().message.c_str());
      return 1;
    }

    // Exercise the wait path once per second's worth of frames: wait for the frame that
    // was presented a full swapchain depth ago (never the one just queued). Only where the
    // device confirms display completion — without has_present_wait the backend reports
    // kNotImplemented rather than dressing a render-done wait up as kDisplayed.
    if (present_info.has_present_wait && present_id > swapchain_info.image_count &&
        (present_id % 60) == 0) {
      auto wait_result = present->wait_for_present(present_id - swapchain_info.image_count,
                                                   std::chrono::seconds(1));
      if (!wait_result) {
        // kNotFound: the id's outcome is no longer tracked (an image-count-changing
        // recreation happened in between) — benign here, waits across a recreation are
        // unsupported. A dropped frame reports success with WaitResult::kDropped.
        if (wait_result.error().code != ErrorCode::kNotFound) {
          std::fprintf(
              stderr, "wait_for_present failed: %s\n", wait_result.error().message.c_str());
          return 1;
        }
      }
    }

    ++frame_counter;
    ++fps_window_frames;
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration<double>(now - fps_window_start).count();
    if (elapsed >= 1.0) {
      std::snprintf(fps_text, sizeof(fps_text), "%.1f fps", fps_window_frames / elapsed);
      fps_window_start = now;
      fps_window_frames = 0;
    }
  }

  std::printf("rendered %llu frames\n", static_cast<unsigned long long>(frame_counter));

  // Teardown order: backends (children of the device) before the device, surface handle
  // before the window.
  unwrap_void(present->destroy(std::chrono::seconds(5)), "PresentBackend::destroy");
  render.reset();
  present.reset();
  surface_handle.impl.reset();
  unwrap_void(device.destroy(std::chrono::seconds(5)), "Device::destroy");
  unwrap_void(window.destroy(), "Window::destroy");

  return 0;
}
