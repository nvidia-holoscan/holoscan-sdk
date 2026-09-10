/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// ImGui demo: per-frame animated text through TextRenderer (FPS counter,
// centered title, orbiting label) plus a Dear ImGui debug window via
// ImGuiBackend::render_ui with input forwarding. Both renderers rebuild their
// content every frame; the geometry reaches the GPU exclusively through CUDA
// (cuMemcpyHtoDAsync into exported buffers), synchronized with the render via
// external-semaphore tokens. For the minimal text-only path, see
// examples/text/minimal.

// example code

#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda.h>

#include <holoviz/rhi.hpp>
#include <holoviz/text.hpp>
#include <holoviz/text/imgui_ui.hpp>

#define CHECK_ERROR(expr)                                           \
  {                                                                 \
    auto result = (expr);                                           \
    if (!result) {                                                  \
      throw std::runtime_error("Error: " + result.error().message); \
    }                                                               \
  }

#define CHECK_CUDA_ERROR(expr)                                           \
  {                                                                      \
    CUresult result = (expr);                                            \
    if (result != CUDA_SUCCESS) {                                        \
      throw std::runtime_error("CUDA Error: " + std::to_string(result)); \
    }                                                                    \
  }

namespace rhi = holoviz::rhi;
namespace text = holoviz::text;

int main() {
  // Initialize CUDA and bind the primary context before creating any exported resources.
  CHECK_CUDA_ERROR(cuInit(0));
  CUdevice cu_dev{};
  CHECK_CUDA_ERROR(cuDeviceGet(&cu_dev, 0));
  CUcontext cu_ctx{};
  CHECK_CUDA_ERROR(cuDevicePrimaryCtxRetain(&cu_ctx, cu_dev));
  CHECK_CUDA_ERROR(cuCtxSetCurrent(cu_ctx));
  CUstream cuda_stream{};
  CHECK_CUDA_ERROR(cuStreamCreate(&cuda_stream, CU_STREAM_NON_BLOCKING));

  rhi::WindowDesc window_desc{
      .title = "Holoviz ImGui Demo",
      .size = {800, 600},
  };
  auto window = rhi::create_window(window_desc).value();
  auto surface = window.create_surface().value();

  rhi::DeviceDesc device_desc{
      .cuda_device = cu_dev,
      .cuda_context = cu_ctx,
  };
  auto device = rhi::create_device_for_surface(device_desc, surface).value();

  rhi::SwapchainDesc swapchain_desc{
      .surface = surface,
      .present_mode = rhi::PresentMode::kFIFO,
      .image_count = 3,
  };
  auto swapchain = device.create_swapchain(swapchain_desc).value();
  const uint32_t swapchain_image_count = swapchain.image_count().value();

  rhi::RenderPassDesc render_pass_desc{.render_targets = {{
                                           .format = swapchain.format().value(),
                                       }}};
  auto render_pass = device.create_render_pass(render_pass_desc).value();

  // Animated per-frame text through the convenience API.
  auto text_renderer_result =
      text::create_text_renderer(device,
                                 {
                                     .stream = cuda_stream,
                                     .frames_in_flight = swapchain_image_count + 1,
                                 });
  if (!text_renderer_result) {
    throw std::runtime_error("Error: " + text_renderer_result.error().message);
  }
  auto text_renderer = std::move(*text_renderer_result);

  // Arbitrary debug UI through the callback mode (see <holoviz/text/imgui_ui.hpp>).
  auto ui_backend = text::create_imgui_backend(device,
                                               {
                                                   .stream = cuda_stream,
                                                   .frames_in_flight = swapchain_image_count + 1,
                                               })
                        .value();

  std::vector<rhi::CommandList> command_lists;
  for (uint32_t i = 0; i < swapchain_image_count; ++i) {
    command_lists.push_back(
        device.create_command_list({.type = rhi::CommandListType::kTransient}).value());
  }
  std::vector<rhi::SubmitSyncToken> sync_tokens(swapchain_image_count, rhi::SubmitSyncToken{});
  // Renderer-slot reuse tokens, one per buffer slot (frames_in_flight, NOT per swapchain
  // image), keyed by the frame counter. One shared array serves both renderers: they use the
  // same ring depth and their draws are consumed by the same submit. See
  // ImGuiBackend::begin_frame for the contract.
  const uint32_t renderer_frames_in_flight = swapchain_image_count + 1;
  std::vector<rhi::SubmitSyncToken> slot_tokens(renderer_frames_in_flight, rhi::SubmitSyncToken{});
  uint32_t frame_counter = 0;

  float orbit_speed = 1.0F;
  float orbit_angle = 0.0F;
  bool show_orbit = true;
  float fps = 0.0F;
  auto last_frame_time = std::chrono::steady_clock::now();

  // Event-driven input: installed once; events flow during rhi::poll_window_events().
  CHECK_ERROR(ui_backend.install_input_callbacks(window));

  while (!window.close_requested().value()) {
    rhi::poll_window_events();

    const auto now = std::chrono::steady_clock::now();
    const float delta_seconds = std::chrono::duration<float>(now - last_frame_time).count();
    last_frame_time = now;
    if (delta_seconds > 0.0F) {
      fps = 0.95F * fps + 0.05F / delta_seconds;  // rolling average
    }
    orbit_angle += orbit_speed * delta_seconds;

    auto acquired = swapchain.acquire(std::chrono::milliseconds(100)).value();
    auto& command_list = command_lists[acquired.image_index];
    const uint32_t renderer_slot = frame_counter % renderer_frames_in_flight;
    const rhi::SubmitSyncToken slot_last_use = slot_tokens[renderer_slot];

    const rhi::Extent2D extent = swapchain.extent().value();
    const float center_x = static_cast<float>(extent.width) / 2.0F;
    const float center_y = static_cast<float>(extent.height) / 2.0F;

    // Per-frame text: everything below is rebuilt from scratch each frame.
    CHECK_ERROR(text_renderer.begin_frame(extent, renderer_slot, slot_last_use));
    char fps_label[64];
    std::snprintf(fps_label, sizeof(fps_label), "%.1f fps", fps);
    CHECK_ERROR(text_renderer.add_text({.position = {20.0F, 20.0F}, .size_px = 20.0F}, fps_label));
    CHECK_ERROR(text_renderer.add_text(
        {
            .position = {center_x, center_y},
            .size_px = 40.0F,
            .color = 0xFF00D4FFu,  // amber (packed 0xAABBGGRR)
            .anchor = text::Anchor::kMiddleCenter,
        },
        "Holoviz Text"));
    if (show_orbit) {
      // Color cycles through hues; position orbits the title.
      const float hue = orbit_angle * 0.3F;
      const auto channel = [hue](float phase) {
        return static_cast<uint32_t>((0.5F + 0.5F * std::sin(hue + phase)) * 255.0F);
      };
      const uint32_t color =
          0xFF000000u | (channel(4.19F) << 16) | (channel(2.09F) << 8) | channel(0.0F);
      CHECK_ERROR(text_renderer.add_text(
          {
              .position = {center_x + 200.0F * std::cos(orbit_angle),
                           center_y + 150.0F * std::sin(orbit_angle)},
              .size_px = 24.0F,
              .color = color,
              .anchor = text::Anchor::kMiddleCenter,
          },
          "updated every frame"));
    }
    const rhi::ExternalSyncToken text_token = text_renderer.end_frame().value();

    // Debug UI: plain ImGui calls inside the callback.
    CHECK_ERROR(ui_backend.begin_frame(renderer_slot, slot_last_use));
    const text::ImGuiFrameSync ui_sync =
        ui_backend
            .render_ui(extent,
                       [&] {
                         ImGui::Begin("Debug");
                         ImGui::Text("%.1f fps", fps);
                         ImGui::Checkbox("orbit", &show_orbit);
                         ImGui::SliderFloat("speed", &orbit_speed, 0.0F, 5.0F);
                         ImGui::End();
                       })
            .value();

    CHECK_ERROR(command_list.reset(std::chrono::milliseconds(100)));
    auto current_image = swapchain.image(acquired.image_index).value();
    CHECK_ERROR(render_pass.set_render_image(current_image, 0, {0.05F, 0.07F, 0.12F, 1.0F}));

    const rhi::Rect2D render_area{0, 0, extent.width, extent.height};
    CHECK_ERROR(command_list.begin());
    CHECK_ERROR(command_list.begin_render(render_pass, render_area));
    CHECK_ERROR(text_renderer.record(command_list, extent));
    CHECK_ERROR(ui_backend.record_draws(command_list, extent));
    CHECK_ERROR(command_list.end_render());
    CHECK_ERROR(command_list.end());

    const rhi::ExternalSyncToken upload_tokens[] = {text_token, ui_sync.upload_done};
    rhi::SubmitDesc submit_desc{
        .wait_submit_sync_tokens = {&sync_tokens[acquired.image_index], 1},
        .wait_external_sync_tokens = upload_tokens,
        .command_lists = {&command_list, 1},
        .acquire_gates = {&acquired.acquire_gate, 1},
        .present_gates = {&acquired.present_gate, 1},
    };
    sync_tokens[acquired.image_index] = device.submit(submit_desc).value();
    // The same submit consumed both renderers' draws for this slot; gate its reuse on it.
    slot_tokens[renderer_slot] = sync_tokens[acquired.image_index];
    ++frame_counter;

    // Read image_index before the move: function-argument evaluation order is
    // unsequenced, so reading through `acquired` in the same call is fragile.
    const uint32_t present_image_index = acquired.image_index;
    CHECK_ERROR(swapchain.present(std::move(acquired), sync_tokens[present_image_index]));
  }

  // Drain the GPU, then destroy children before the device.
  CHECK_ERROR(device.wait_idle());
  CHECK_ERROR(text_renderer.destroy());
  CHECK_ERROR(ui_backend.destroy());

  CHECK_CUDA_ERROR(cuStreamSynchronize(cuda_stream));
  CHECK_CUDA_ERROR(cuStreamDestroy(cuda_stream));

  return EXIT_SUCCESS;
}
