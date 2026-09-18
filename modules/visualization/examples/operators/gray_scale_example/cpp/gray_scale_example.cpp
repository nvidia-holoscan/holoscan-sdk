/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// gray_scale_example — a 16-bit grayscale image displayed natively by HolovizOp.
//
//   GrayScaleSource --(uint16 {H, W} Tensor)--> HolovizOp (image layer kR16Unorm)
//
// A CUDA kernel renders a smooth 16-bit ramp with a sweeping bar; HolovizOp uploads the two-byte
// samples as an R16_UNORM texture and replicates the single channel to gray. The source, the
// transport and the upload stay 16-bit; only the window's BGRA8 swapchain quantizes the final
// displayed color to 8 bits per channel.

#include <atomic>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
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

using namespace std::chrono_literals;

constexpr std::uint32_t kWidth = 640;
constexpr std::uint32_t kHeight = 480;

struct Arguments {
  std::chrono::seconds run_for{0};  // zero: run until Ctrl-C
  int device{0};                    // CUDA device ordinal for the tensor pool
  bool fullscreen{false};           // undecorated monitor-sized window instead of a decorated one
  std::string display_name;         // --fullscreen monitor (window-system name); empty = primary
  std::uint32_t refresh_hz{0};      // --fullscreen video-mode refresh rate; 0 keeps the current
  std::uint32_t fps{60};            // source and display rate; keep inside the panel's VRR range
};

// Frame period for a rate in Hz.
[[nodiscard]] constexpr std::chrono::nanoseconds period_for(std::uint32_t fps) {
  return std::chrono::nanoseconds{1'000'000'000LL / fps};
}

std::atomic<bool> g_stop_requested{false};

void on_signal(int) {
  g_stop_requested.store(true, std::memory_order_relaxed);
}

/// Parse the whole of @p text as a nonnegative decimal integer of type T. Rejects empty input,
/// trailing text, a sign, and values outside T's range.
template <typename T>
[[nodiscard]] bool parse_nonnegative(std::string_view text, T& value) noexcept {
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  return error == std::errc() && end == text.data() + text.size() && value >= T{0};
}

/// True when `steady_clock::now() + seconds{value}` is representable, so the run deadline computed
/// in main() cannot overflow.
[[nodiscard]] bool deadline_is_representable(std::int64_t value) {
  const auto headroom =
      std::chrono::steady_clock::time_point::max() - std::chrono::steady_clock::now();
  return value <= std::chrono::duration_cast<std::chrono::seconds>(headroom).count();
}

void print_usage() {
  std::cerr << "usage: gray_scale_example [--seconds N] [--device D] [--fps N]"
               " [--fullscreen [MONITOR] [--hz HZ]]\n";
}

[[nodiscard]] bool parse_arguments(int argc, char** argv, Arguments& out) {
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg{argv[i]};
    const bool has_value = i + 1 < argc;
    if (arg == "--seconds" && has_value) {
      const std::string_view text{argv[++i]};
      std::int64_t value{};
      if (!parse_nonnegative(text, value) || !deadline_is_representable(value)) {
        std::cerr << "invalid --seconds value: '" << text << "' (expected a nonnegative integer)\n";
        print_usage();
        return false;
      }
      out.run_for = std::chrono::seconds{value};
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
    } else if (arg == "--fullscreen") {
      out.fullscreen = true;
      // An optional monitor name follows unless the next token is another option.
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
      if (!parse_nonnegative(text, value) || value == 0 || value > 1000) {
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
  if (out.refresh_hz != 0 && !out.fullscreen) {
    std::cerr << "--hz requires --fullscreen\n";
    return false;
  }
  return true;
}

// A fullscreen window takes the monitor's current video mode (width/height 0) and closes on ESC,
// since an undecorated window has no close button. It also pins plain FIFO presentation: the
// default kAuto prefers FIFO_LATEST_READY when the driver offers it, which the NVIDIA X11 driver
// flips with G-Sync as well but reports as "VSYNC OFF" in its overlay; FIFO is the unambiguous
// vsync path for a G-Sync demonstration.
[[nodiscard]] holoscan::visualization::SinkDesc make_sink(const Arguments& arguments) {
  return holoscan::visualization::WindowSinkDesc{
      .title = "gray_scale_example (16-bit)",
      .width = arguments.fullscreen ? 0U : kWidth,
      .height = arguments.fullscreen ? 0U : kHeight,
      .present_mode = arguments.fullscreen ? holoscan::visualization::PresentMode::kFIFO
                                           : holoscan::visualization::PresentMode::kAuto,
      .fullscreen = arguments.fullscreen,
      .display_name = arguments.display_name,
      .refresh_hz = arguments.refresh_hz,
      .close_on_escape = arguments.fullscreen};
}

}  // namespace

int main(int argc, char** argv) {
  Arguments arguments;
  if (!parse_arguments(argc, argv, arguments)) {
    return 1;
  }
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

#if !defined(_WIN32)
  // The NVIDIA ICD reads __GL_SYNC_DISPLAY_DEVICE once, when Vulkan first loads in the process,
  // so it has to be exported before the graph runs; HolovizOp only logs a hint. main() is still
  // single-threaded here, so setenv() is safe. An explicit setting wins (overwrite = 0).
  if (arguments.fullscreen && !arguments.display_name.empty()) {
    setenv("__GL_SYNC_DISPLAY_DEVICE", arguments.display_name.c_str(), 0);
  }
#endif

  holoscan::Graph graph{"gray-scale-example"};

  // Source and display share one period (--fps). For G-Sync/VRR the rate has to sit inside the
  // panel's variable-refresh range and below its maximum: a 60 Hz default on a 60..144 Hz panel is
  // at the floor, where jitter makes the driver frame-double instead of following the presents.
  const std::chrono::nanoseconds period = period_for(arguments.fps);
  const auto source = graph.op<gray_scale::GrayScaleSource>("source", kWidth, kHeight, period);

  // The layer format tells HolovizOp the tensor holds one uint16 per pixel; the operator checks
  // the incoming dtype and channel count against it and drops mismatching samples.
  holoscan::visualization::LayerStack stack;
  stack.add_image_layer("gray",
                        holoscan::visualization::ImageLayerDesc{
                            .format = holoscan::visualization::ImageFormat::kR16Unorm,
                            .max_width = kWidth,
                            .max_height = kHeight});
  const auto viz = graph.op<holoscan::visualization::HolovizOp>(
      "holoviz", std::move(stack), make_sink(arguments), period, std::chrono::nanoseconds{100ms});
  // Depth-1 "latest frame" mailbox: never backpressure the source while the window starts up.
  graph.add_flow(source->frame,
                 viz->layer_input(0),
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

  holoscan::RunSession session = holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + arguments.run_for;
  while (session.termination() == holoscan::RunTermination::kRunning) {
    if (g_stop_requested.load(std::memory_order_relaxed)) {
      break;
    }
    if (arguments.run_for.count() > 0 && std::chrono::steady_clock::now() >= deadline) {
      break;
    }
    std::this_thread::sleep_for(50ms);
  }
  session.request_stop();
  session.wait();

  for (const auto& diagnostic : session.diagnostics()) {
    std::cerr << "diagnostic: " << diagnostic.token << " vertex=" << diagnostic.source.vertex.value
              << '\n';
  }
  if (session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    std::cerr << "gray_scale_example failed: termination="
              << static_cast<int>(session.termination())
              << " diagnostics=" << session.diagnostics().size() << '\n';
    return 2;
  }
  std::cout << "gray_scale_example complete\n";
  return 0;
}
