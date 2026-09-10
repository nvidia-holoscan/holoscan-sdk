/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Direct-display example: enumerate the displays connected to each adapter
// (with ownership: compositor-owned vs available for direct latch-on), then —
// with --display — acquire one for exclusive scanout and render a color sweep
// to it through the regular swapchain machinery. No window is created.
//
//   holoviz_rhi_direct_display --list [--all-modes]
//   holoviz_rhi_direct_display --display <name substring>
//       [--width W] [--height H] [--hz R] [--frames N]

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
// example code

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoviz/rhi.hpp>

#define CHECK_ERROR(expr)                                           \
  {                                                                 \
    auto result = (expr);                                           \
    if (!result) {                                                  \
      throw std::runtime_error("Error: " + result.error().message); \
    }                                                               \
  }

namespace rhi = holoviz::rhi;

namespace {

const char* ownership_name(rhi::DisplayOwnership ownership) {
  switch (ownership) {
    case rhi::DisplayOwnership::kCompositor:
      return "compositor-owned";
    case rhi::DisplayOwnership::kAvailable:
      return "available for direct display";
    default:
      return "unknown";
  }
}

void print_mode(const rhi::DisplayMode& mode) {
  std::cout << mode.resolution.width << "x" << mode.resolution.height << " "
            << static_cast<float>(mode.refresh_rate_mhz) / 1000.F << " Hz";
}

/// Larger resolution first (width, then height), then faster refresh — the
/// same preference order create_surface uses for a don't-care request.
bool mode_less(const rhi::DisplayMode& a, const rhi::DisplayMode& b) {
  if (a.resolution.width != b.resolution.width) {
    return a.resolution.width < b.resolution.width;
  }
  if (a.resolution.height != b.resolution.height) {
    return a.resolution.height < b.resolution.height;
  }
  return a.refresh_rate_mhz < b.refresh_rate_mhz;
}

void list_displays(bool all_modes) {
  auto adapters = rhi::enumerate_adapters().value();
  for (const rhi::Adapter& adapter : adapters) {
    std::cout << "Adapter: " << adapter.name.data() << "\n";

    auto displays = rhi::enumerate_displays(adapter);
    if (!displays) {
      std::cout << "  (" << displays.error().message << ")\n";
      continue;
    }
    if (displays->empty()) {
      std::cout << "  No displays connected.\n";
      continue;
    }
    for (const rhi::Display& display : displays.value()) {
      // ASCII only: the Windows console decodes with the OEM codepage by default.
      std::cout << "  Display: \"" << display.name.data() << "\" - "
                << ownership_name(display.ownership);
      if (display.current_mode.resolution.width > 0) {
        std::cout << ", current mode ";
        print_mode(display.current_mode);
      }
      std::cout << "\n";
      auto modes = display.modes();
      if (!modes || modes->empty()) {
        continue;
      }
      if (all_modes) {
        std::cout << "    Modes:";
        for (const rhi::DisplayMode& mode : modes.value()) {
          std::cout << " ";
          print_mode(mode);
          std::cout << ";";
        }
      } else {
        std::cout << "    Best mode: ";
        print_mode(*std::max_element(modes->begin(), modes->end(), mode_less));
        std::cout << " (" << modes->size() << " modes total, --all-modes to list)";
      }
      std::cout << "\n";
    }
  }
}

void render_to_display(const std::string& name_filter, const rhi::DisplayMode& desired_mode,
                       uint32_t frame_count, rhi::PresentMode present_mode,
                       uint32_t image_count_request) {
  auto adapters = rhi::enumerate_adapters().value();

  rhi::Display selected{};  // Display is a plain value type; copy the first match.
  bool found = false;
  for (const rhi::Adapter& adapter : adapters) {
    auto adapter_displays = rhi::enumerate_displays(adapter);
    if (!adapter_displays) {
      continue;
    }
    for (const rhi::Display& display : adapter_displays.value()) {
      if (name_filter.empty() ||
          std::string(display.name.data()).find(name_filter) != std::string::npos) {
        selected = display;
        found = true;
        break;
      }
    }
    if (found) {
      break;
    }
  }
  if (!found) {
    throw std::runtime_error("No display matching \"" + name_filter +
                             "\" found; run with --list to see the connected displays.");
  }

  std::cout << "Latching onto display \"" << selected.name.data() << "\" ("
            << ownership_name(selected.ownership) << ")\n";

  auto surface_result = selected.create_surface({.mode = desired_mode});
  if (!surface_result) {
    throw std::runtime_error(surface_result.error().message);
  }
  auto surface = std::move(surface_result).value();
  const rhi::Extent2D extent = surface.extent().value();
  std::cout << "Display surface: " << extent.width << "x" << extent.height << "\n";

  auto device = rhi::create_device_for_surface({}, surface).value();
  std::cout << "Using device: " << device.adapter().value().name.data() << "\n";
  std::cout << "exclusive_display capability: "
            << (device.capabilities().value().exclusive_display ? "yes" : "no") << "\n";

  const rhi::SurfaceCapabilities surface_caps = device.surface_capabilities(surface).value();
  std::cout << "Present modes:";
  for (auto mode : {rhi::PresentMode::kFIFO,
                    rhi::PresentMode::kFIFOLatestReady,
                    rhi::PresentMode::kMailbox,
                    rhi::PresentMode::kImmediate}) {
    if (surface_caps.present_modes.contains(mode)) {
      std::cout << " " << static_cast<int>(mode) << ";";
    }
  }
  std::cout << " (image count [" << surface_caps.min_image_count << ", "
            << surface_caps.max_image_count << "])\n";

  rhi::SwapchainDesc swapchain_desc{
      .surface = surface,
      .present_mode = present_mode,
      .image_count = image_count_request,  // default 2: exclusive low-latency display
      .debug_name = "direct_display",
  };
  auto swapchain = device.create_swapchain(swapchain_desc).value();
  const uint32_t image_count = swapchain.image_count().value();

  rhi::RenderPassDesc render_pass_desc{.render_targets = {{.format = swapchain.format().value()}}};
  auto render_pass = device.create_render_pass(render_pass_desc).value();

  std::vector<rhi::CommandList> command_lists;
  std::vector<rhi::SubmitSyncToken> sync_tokens(image_count, rhi::SubmitSyncToken{});
  for (uint32_t i = 0; i < image_count; ++i) {
    command_lists.push_back(
        device.create_command_list({.type = rhi::CommandListType::kTransient}).value());
  }

  // A fixed-mode display surface never resizes: no per-frame surface tracking,
  // and any kOutOfDate would be a genuine display loss.
  bool measure_vblank =
      device.capabilities().value().present_completion && frame_count >= image_count;
  std::vector<double> present_done_seconds;
  const auto loop_start = std::chrono::steady_clock::now();
  auto window_start = loop_start;
  uint32_t window_frames = 0;
  for (uint32_t frame = 0; frame < frame_count; ++frame) {
    auto acquire_result = swapchain.acquire(std::chrono::milliseconds(500));
    if (!acquire_result) {
      if (acquire_result.error().code == rhi::ErrorCode::kBackpressured) {
        continue;
      }
      throw std::runtime_error("Error: " + acquire_result.error().message);
    }
    auto acquired = std::move(acquire_result).value();

    auto& command_list = command_lists[acquired.image_index];
    CHECK_ERROR(command_list.reset(std::chrono::milliseconds(500)));

    // Slow color sweep so scanout is obviously live.
    const float phase = static_cast<float>(frame) * 0.02F;
    const rhi::Color clear{0.5F + 0.5F * std::sin(phase),
                           0.5F + 0.5F * std::sin(phase + 2.094F),
                           0.5F + 0.5F * std::sin(phase + 4.189F),
                           1.0F};

    auto current_image = swapchain.image(acquired.image_index).value();
    CHECK_ERROR(render_pass.set_render_image(current_image, 0, clear));

    const rhi::Rect2D render_area{0, 0, extent.width, extent.height};
    CHECK_ERROR(command_list.begin());
    CHECK_ERROR(command_list.begin_render(render_pass, render_area));
    CHECK_ERROR(command_list.end_render());
    CHECK_ERROR(command_list.end());

    rhi::SubmitDesc submit_desc{
        .wait_submit_sync_tokens = {&sync_tokens[acquired.image_index], 1},
        .command_lists = {&command_list, 1},
        .acquire_gates = {&acquired.acquire_gate, 1},
        .present_gates = {&acquired.present_gate, 1},
    };
    const uint32_t image_index = acquired.image_index;
    sync_tokens[image_index] = device.submit(submit_desc).value();

    CHECK_ERROR(swapchain.present(
        std::move(acquired), sync_tokens[image_index], measure_vblank ? frame + 1 : 0));

    // First frames only: with the queue still full, the spacing of display-done
    // completions is the true scanout period (vblank cadence), independent of
    // how fast the loop itself is paced.
    if (measure_vblank && frame < image_count) {
      if (auto waited = swapchain.wait_for_present(frame + 1, std::chrono::seconds(2)); !waited) {
        std::cout << "wait_for_present unavailable on this swapchain (" << waited.error().message
                  << ")\n";
        measure_vblank = false;
      } else {
        present_done_seconds.push_back(
            std::chrono::duration<double>(std::chrono::steady_clock::now() - loop_start).count());
        if (frame + 1 == image_count) {
          std::cout << "Display-done spacing (ms):";
          for (size_t i = 1; i < present_done_seconds.size(); ++i) {
            std::cout << " " << (present_done_seconds[i] - present_done_seconds[i - 1]) * 1000.0;
          }
          std::cout << "\n";
        }
      }
    }

    ++window_frames;
    if (const auto now = std::chrono::steady_clock::now();
        now - window_start >= std::chrono::seconds(1)) {
      const double seconds = std::chrono::duration<double>(now - window_start).count();
      std::cout << "  " << static_cast<double>(window_frames) / seconds << " fps\n";
      window_start = now;
      window_frames = 0;
    }
  }
  const double total_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - loop_start).count();

  CHECK_ERROR(device.wait_idle());
  std::cout << "Rendered " << frame_count << " frames in " << total_seconds << " s ("
            << static_cast<double>(frame_count) / total_seconds
            << " fps average); releasing the display.\n";
  // The swapchain must go before the surface it was created on.
  CHECK_ERROR(swapchain.destroy(std::chrono::seconds(5)));
  CHECK_ERROR(surface.destroy());
}

}  // namespace

int main(int argc, char** argv) {
  bool list = false;
  bool all_modes = false;
  std::string display_filter;
  bool have_display = false;
  rhi::DisplayMode desired_mode{};
  uint32_t frame_count = 600;
  rhi::PresentMode present_mode = rhi::PresentMode::kFIFOLatestReady;
  uint32_t image_count_request = 2;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto next_value = [&]() -> const char* {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + arg);
      }
      return argv[++i];
    };
    if (arg == "--list" || arg == "-l") {
      list = true;
    } else if (arg == "--all-modes") {
      all_modes = true;
    } else if (arg == "--display" || arg == "-d") {
      display_filter = next_value();
      have_display = true;
    } else if (arg == "--width") {
      desired_mode.resolution.width = std::stoul(next_value());
    } else if (arg == "--height") {
      desired_mode.resolution.height = std::stoul(next_value());
    } else if (arg == "--hz") {
      desired_mode.refresh_rate_mhz = static_cast<uint32_t>(std::stof(next_value()) * 1000.F);
    } else if (arg == "--frames") {
      frame_count = std::stoul(next_value());
    } else if (arg == "--immediate") {
      present_mode = rhi::PresentMode::kImmediate;
    } else if (arg == "--fifo") {
      present_mode = rhi::PresentMode::kFIFO;
    } else if (arg == "--images") {
      image_count_request = std::stoul(next_value());
    } else {
      std::cerr << "Usage: " << argv[0]
                << " [--list] [--all-modes] [--display <name substring> [--width W] [--height H]"
                   " [--hz R] [--frames N]]\n";
      return arg == "--help" || arg == "-h" ? EXIT_SUCCESS : EXIT_FAILURE;
    }
  }

  try {
    if (!have_display || list) {
      list_displays(all_modes);
    }
    if (have_display) {
      render_to_display(
          display_filter, desired_mode, frame_count, present_mode, image_count_request);
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

// NOLINTEND(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
