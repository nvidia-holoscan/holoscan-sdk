/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// gray_scale_example — a 16-bit grayscale image displayed natively by HolovizOp.
//
//   GrayScaleSource --(schema::ImageT, MONO16, uint16 {H, W})--> HolovizOp (image layer kR16Unorm)
//
// A CUDA kernel renders a smooth 16-bit ramp with a sweeping bar; HolovizOp uploads the two-byte
// samples as an R16_UNORM texture and replicates the single channel to gray. The source, the
// transport and the upload stay 16-bit; only the window's BGRA8 swapchain quantizes the final
// displayed color to 8 bits per channel.

#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

#include "gray_scale_source.hpp"

namespace {

constexpr std::uint32_t kWidth = 640;
constexpr std::uint32_t kHeight = 480;

// Where HolovizOp presents: a window, an undecorated fullscreen window, or exclusive scanout to a
// display no compositor owns (Linux DRM only in this release).
enum class OutputMode : std::uint8_t { kWindow, kFullscreen, kDisplay };

struct Arguments {
  std::uint64_t seconds{0};  // --seconds: frames = seconds * fps (0: run until Ctrl-C)
  int device{0};             // CUDA device ordinal for the tensor pool
  OutputMode output{OutputMode::kWindow};
  // --fullscreen: window-system monitor name (empty = primary). --display: case-sensitive
  // substring of the DRM connector name, e.g. "DP-2" (empty = the first display no compositor
  // owns). The two namespaces differ: `holoviz_direct_display --list` prints the DRM names.
  std::string display_name;
  // --fullscreen / --display. 0 means: fullscreen keeps the monitor's current rate; exclusive
  // display picks the fastest refresh at the selected (largest) resolution.
  std::uint32_t refresh_hz{0};
  std::uint32_t fps{60};  // source rate (display follows it); keep inside the VRR range

  /// Compute calls after which the source completes; zero leaves it unbounded.
  [[nodiscard]] std::uint64_t frame_limit() const { return seconds * fps; }
};

// Frame period for a rate in Hz.
[[nodiscard]] constexpr std::chrono::nanoseconds period_for(std::uint32_t fps) {
  return std::chrono::nanoseconds{1'000'000'000LL / fps};
}

/// Parse the whole of @p text as a nonnegative decimal integer of type T. Rejects empty input,
/// trailing text, a sign, and values outside T's range.
template <typename T>
[[nodiscard]] bool parse_nonnegative(std::string_view text, T& value) noexcept {
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc() && end == text.data() + text.size() && value >= T{0};
}

/// Upper bound of --fps; --seconds is bounded so that `seconds * kMaxFps` cannot overflow.
constexpr std::uint32_t kMaxFps = 1000U;
constexpr std::uint64_t kMaxSeconds = std::numeric_limits<std::uint64_t>::max() / kMaxFps;

void print_usage() {
  std::cerr << "usage: gray_scale_example [--seconds N] [--device D] [--fps N]"
               " [--fullscreen [MONITOR] | --display [NAME]] [--hz HZ]\n";
}

[[nodiscard]] bool parse_arguments(int argc, char** argv, Arguments& out) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg{argv[i]};
    const bool has_value = i + 1 < argc;
    if (arg == "--seconds" && has_value) {
      const std::string_view text{argv[++i]};
      std::uint64_t value{};
      if (!parse_nonnegative(text, value) || value > kMaxSeconds) {
        std::cerr << "invalid --seconds value: '" << text << "' (expected a nonnegative integer)\n";
        print_usage();
        return false;
      }
      out.seconds = value;
    } else if (arg == "--device" && has_value) {
      const std::string_view text{argv[++i]};
      int value{};
      if (!parse_nonnegative(text, value)) {
        std::cerr << "invalid --device value: '" << text
                  << "' (expected a nonnegative CUDA device ordinal)\n";
        print_usage();
        return false;
      }
      out.device = value;
    } else if (arg == "--fullscreen" || arg == "--display") {
      if (out.output != OutputMode::kWindow) {
        std::cerr << "--fullscreen and --display are mutually exclusive and accepted once\n";
        return false;
      }
      out.output = arg == "--display" ? OutputMode::kDisplay : OutputMode::kFullscreen;
      // An optional monitor/display name follows unless the next token is another option.
      if (has_value && std::string_view{argv[i + 1]}.substr(0, 2) != "--") {
        out.display_name = argv[++i];
      }
    } else if (arg == "--hz" && has_value) {
      const std::string_view text{argv[++i]};
      std::uint32_t value{};
      if (!parse_nonnegative(text, value)) {
        std::cerr << "invalid --hz value: '" << text << "' (expected a nonnegative refresh rate)\n";
        print_usage();
        return false;
      }
      out.refresh_hz = value;
    } else if (arg == "--fps" && has_value) {
      const std::string_view text{argv[++i]};
      std::uint32_t value{};
      if (!parse_nonnegative(text, value) || value == 0 || value > kMaxFps) {
        std::cerr << "invalid --fps value: '" << text << "' (expected 1..1000)\n";
        print_usage();
        return false;
      }
      out.fps = value;
    } else {
      print_usage();
      return false;
    }
  }
  if (out.refresh_hz != 0 && out.output == OutputMode::kWindow) {
    std::cerr << "--hz requires --fullscreen or --display\n";
    return false;
  }
  return true;
}

// Exclusive display: width/height 0 select the display's largest resolution and refresh_hz 0 the
// fastest refresh at that resolution (see ExclusiveDisplaySinkDesc), so a modeset may occur even
// with the defaults; there is no window, so the run ends with Ctrl-C or --seconds.
// A fullscreen window takes the monitor's current video mode (width/height 0) and closes on ESC,
// since an undecorated window has no close button. It also pins plain FIFO presentation: the
// default kAuto prefers FIFO_LATEST_READY when the driver offers it, which the NVIDIA X11 driver
// flips with G-Sync as well but reports as "VSYNC OFF" in its overlay; FIFO is the unambiguous
// vsync path for a G-Sync demonstration.
[[nodiscard]] holoscan::visualization::SinkDesc make_sink(const Arguments& arguments) {
  if (arguments.output == OutputMode::kDisplay) {
    return holoscan::visualization::ExclusiveDisplaySinkDesc{.display_name = arguments.display_name,
                                                             .refresh_hz = arguments.refresh_hz};
  }
  const bool fullscreen = arguments.output == OutputMode::kFullscreen;
  return holoscan::visualization::WindowSinkDesc{
      .title = "gray_scale_example (16-bit)",
      .width = fullscreen ? 0U : kWidth,
      .height = fullscreen ? 0U : kHeight,
      .present_mode = fullscreen ? holoscan::visualization::PresentMode::kFIFO
                                 : holoscan::visualization::PresentMode::kAuto,
      .fullscreen = fullscreen,
      .display_name = arguments.display_name,
      .refresh_hz = arguments.refresh_hz,
      .close_on_escape = fullscreen};
}

}  // namespace

int main(int argc, char** argv) {
  Arguments arguments;
  if (!parse_arguments(argc, argv, arguments)) {
    return 1;
  }

#if !defined(_WIN32)
  // The NVIDIA ICD reads __GL_SYNC_DISPLAY_DEVICE once, when Vulkan first loads in the process,
  // so it has to be exported before the graph runs; HolovizOp only logs a hint. main() is still
  // single-threaded here, so setenv() is safe. An explicit setting wins (overwrite = 0).
  if (arguments.output == OutputMode::kFullscreen && !arguments.display_name.empty()) {
    setenv("__GL_SYNC_DISPLAY_DEVICE", arguments.display_name.c_str(), 0);
  }
#endif

  holoscan::Graph graph{"gray-scale-example"};

  // The source sets the frame rate (--fps); HolovizOp presents each frame as it arrives (its
  // default LayerTrigger on layer 0), so the display follows the source with no clock in between.
  // For G-Sync/VRR the rate has to sit inside the panel's variable-refresh range and below its
  // maximum: a 60 Hz default on a 60..144 Hz panel is at the floor, where jitter makes the driver
  // frame-double instead of following the presents.
  const std::chrono::nanoseconds period = period_for(arguments.fps);
  const auto source = graph.op<gray_scale::GrayScaleSource>(
      "source", kWidth, kHeight, period, arguments.frame_limit());

  // The layer format tells HolovizOp the tensor holds one uint16 per pixel; the operator checks
  // the incoming dtype and channel count against it and drops mismatching samples.
  holoscan::visualization::LayerStack stack;
  stack.add_image_layer("gray",
                        holoscan::visualization::ImageLayerDesc{
                            .format = holoscan::visualization::ImageFormat::kR16Unorm,
                            .max_width = kWidth,
                            .max_height = kHeight});
  const auto viz = graph.op<holoscan::visualization::HolovizOp>(
      "holoviz", std::move(stack), make_sink(arguments));
  // Depth-1 "latest frame" mailbox: never backpressure the source while the window starts up.
  graph.add_flow(source->frame,
                 viz->image_layer_input(0),
                 holoscan::ConnectionOptions{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                             .queue_depth = 1U});

  graph.partition("display").add(viz);
  graph.auto_partition_remaining();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // CUDA tensor pools must be placed explicitly; compile rejects an implicit device 0.
  holoscan::CompileOptions options;
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{.operator_path = "source",
                                            .output_port = "frame",
                                            .device = holoscan::DeviceId{arguments.device}});

  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  // Blocking run. On Linux the runtime owns SIGINT/SIGTERM for the duration: the first one
  // requests a cooperative stop and run() returns nonzero. With --seconds the source completes
  // after seconds * fps frames (complete_after in its contract), HolovizOp drains, and run()
  // returns zero. Failures are summarized on the log by the runtime.
  if (const int result = holoscan::run(plan); result != 0) {
    std::cerr << "gray_scale_example: run ended with status " << result << '\n';
    return 2;
  }
  std::cout << "gray_scale_example complete\n";
  return 0;
}
