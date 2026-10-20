/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// cuda_sink_example — two HolovizOps in one graph, chained through the CUDA sink.
//
//   PatternSource --(ImageT RGBA8, uint8 {H, W, 4} CUDA)--> HolovizOp "render" (image + geometry
//                 --(float32 rectangles) ---------------->    layers, CudaSinkDesc)
//                                                    |  "output": ImageT RGBA8,
//                                                    |  uint8 {H, W, 4} CUDA
//                                                    v
//                                              HolovizOp "display" (image layer -> window /
//                                                    fullscreen / exclusive display)
//
// The first HolovizOp composites a kernel-generated image with a rectangle overlay and, instead of
// presenting, publishes the finished frame as a schema::ImageT whose data Tensor lives in CUDA
// memory. The second HolovizOp consumes that ImageT directly as an image layer (no copy, no host
// round trip) and puts it on screen through whichever sink `--display` / `--fullscreen` select,
// exactly as gray_scale_example does.

#include <atomic>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
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
#include <holoviz/adapter.hpp>

#include "pattern_source.hpp"

namespace {

using namespace std::chrono_literals;
namespace viz = holoscan::visualization;

constexpr std::uint32_t kWidth = 640;
constexpr std::uint32_t kHeight = 480;

// Where the second HolovizOp presents: a window, an undecorated fullscreen window, or exclusive
// scanout to a display no compositor owns (Linux DRM only in this release).
enum class OutputMode : std::uint8_t { kWindow, kFullscreen, kDisplay };

struct Arguments {
  std::chrono::seconds run_for{0};  // zero: run until Ctrl-C
  // CUDA device ordinal for every tensor pool. Unset: the device of the HoloViz adapter the
  // "render" HolovizOp will pick (see holoviz_cuda_device()); set: must be that same device.
  std::optional<int> device;
  OutputMode output{OutputMode::kWindow};
  // --fullscreen: window-system monitor name (empty = primary). --display: case-sensitive
  // substring of the DRM connector name, e.g. "DP-2" (empty = the first display no compositor
  // owns). The two namespaces differ: `holoviz_direct_display --list` prints the DRM names.
  std::string display_name;
  // --fullscreen / --display. 0 means: fullscreen keeps the monitor's current rate; exclusive
  // display picks the fastest refresh at the selected (largest) resolution.
  std::uint32_t refresh_hz{0};
  std::uint32_t fps{60};  // source rate; both HolovizOps follow it
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
  std::cerr << "usage: cuda_sink_example [--seconds N] [--device D] [--fps N]"
               " [--fullscreen [MONITOR] | --display [NAME]] [--hz HZ]\n";
}

/// The CUDA device the CUDA-sink HolovizOp renders on: it has no surface to pick an adapter
/// from, so it takes the first HoloViz adapter backed by a CUDA device. Its output pool must live
/// on that device (compute() rejects an activation stream from any other), so the example
/// derives the pool placement from the same rule instead of trusting --device blindly.
[[nodiscard]] std::optional<int> holoviz_cuda_device() {
  auto adapters = holoviz::enumerate_adapters();
  if (!adapters) {
    return std::nullopt;
  }
  for (const holoviz::Adapter& adapter : *adapters) {
    if (adapter.cuda_device_index >= 0) {
      return adapter.cuda_device_index;
    }
  }
  return std::nullopt;
}

// Keep the command-line option dispatch together so accepted syntax and rejection paths can be
// compared.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
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
  if (out.refresh_hz != 0 && out.output == OutputMode::kWindow) {
    std::cerr << "--hz requires --fullscreen or --display\n";
    return false;
  }
  return true;
}

// The display-side sink, selected like gray_scale_example does. Exclusive display: width/height 0
// select the display's largest resolution and refresh_hz 0 the fastest refresh at that resolution
// (see ExclusiveDisplaySinkDesc), so a modeset may occur even with the defaults; there is no
// window, so the run ends with Ctrl-C or --seconds. A fullscreen window takes the monitor's
// current video mode (width/height 0), closes on ESC, and pins plain FIFO presentation.
[[nodiscard]] viz::SinkDesc make_display_sink(const Arguments& arguments) {
  if (arguments.output == OutputMode::kDisplay) {
    return viz::ExclusiveDisplaySinkDesc{.display_name = arguments.display_name,
                                         .refresh_hz = arguments.refresh_hz};
  }
  const bool fullscreen = arguments.output == OutputMode::kFullscreen;
  return viz::WindowSinkDesc{
      .title = "cuda_sink_example",
      .width = fullscreen ? 0U : kWidth,
      .height = fullscreen ? 0U : kHeight,
      .present_mode = fullscreen ? viz::PresentMode::kFIFO : viz::PresentMode::kAuto,
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
  // The NVIDIA ICD reads __GL_SYNC_DISPLAY_DEVICE once, when the Vulkan instance is first
  // created in the process — which the adapter enumeration right below does — so it has to be
  // exported before that; HolovizOp only logs a hint. main() is still single-threaded here, so
  // setenv() is safe. An explicit setting wins (overwrite = 0).
  if (arguments.output == OutputMode::kFullscreen && !arguments.display_name.empty()) {
    setenv("__GL_SYNC_DISPLAY_DEVICE", arguments.display_name.c_str(), 0);
  }
#endif

  // Argument syntax is checked above without touching the GPU; the device choice needs the
  // adapter list. A --device that is not the render operator's device would only fail later, at
  // compute(), with a cross-device error — reject it here instead.
  const std::optional<int> render_device = holoviz_cuda_device();
  if (!render_device) {
    std::cerr << "cuda_sink_example: no HoloViz adapter backed by a CUDA device is available\n";
    return 1;
  }
  if (arguments.device && *arguments.device != *render_device) {
    std::cerr << "invalid --device value: '" << *arguments.device
              << "' (the render HolovizOp uses CUDA device " << *render_device
              << ", the first HoloViz adapter backed by a CUDA device; the tensor pools must be "
                 "placed there)\n";
    return 1;
  }
  const int device = *render_device;
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  holoscan::Graph graph{"cuda-sink-example"};

  // The source sets the frame rate (--fps). Both HolovizOps render one frame per arriving image
  // sample (their default LayerTrigger on layer 0), so the whole chain follows the source.
  const auto source =
      graph.op<cuda_sink::PatternSource>("source", kWidth, kHeight, period_for(arguments.fps));

  // First HolovizOp: composite the kernel image (layer 0, the trigger) with the rectangle overlay
  // (layer 1, freshest sample within max_input_age) and capture the result into CUDA memory.
  viz::LayerStack render_stack;
  render_stack.add_image_layer(
      "image",
      viz::ImageLayerDesc{
          .format = viz::ImageFormat::kR8G8B8A8Unorm, .max_width = kWidth, .max_height = kHeight});
  render_stack.add_geometry_layer(
      "overlay",
      viz::GeometryLayerDesc{.topologies = {viz::GeometryTopology::kRectangleList},
                             .max_primitives = cuda_sink::kOverlayRectangles,
                             .default_line_width = 3.0F});
  const auto render = graph.op<viz::HolovizOp>(
      "render",
      std::move(render_stack),
      viz::CudaSinkDesc{
          .width = kWidth, .height = kHeight, .format = viz::ImageFormat::kR8G8B8A8Unorm});

  // Second HolovizOp: the captured frame is an ordinary image layer. Its ImageT is RGBA8-encoded
  // (the sink format), matching this layer's declared format, and its Tensor arrives on the
  // device, so the operator hands the pointer to the renderer without any copy.
  viz::LayerStack display_stack;
  display_stack.add_image_layer(
      "captured",
      viz::ImageLayerDesc{
          .format = viz::ImageFormat::kR8G8B8A8Unorm, .max_width = kWidth, .max_height = kHeight});
  const auto display =
      graph.op<viz::HolovizOp>("display", std::move(display_stack), make_display_sink(arguments));

  // Depth-1 "latest sample" mailboxes everywhere: a stalled window or a slow stage costs at most
  // one stale frame and never backpressures the source.
  const holoscan::ConnectionOptions mailbox{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                            .queue_depth = 1U};
  graph.add_flow(source->frame, render->image_layer_input(0), mailbox);
  graph.add_flow(source->overlay, render->tensor_layer_input(1), mailbox);
  graph.add_flow(render->output(), display->image_layer_input(0), mailbox);

  // Each HolovizOp owns a GPU frame loop; give both their own partition (thread).
  graph.partition("render").add(render);
  graph.partition("display").add(display);
  graph.auto_partition_remaining();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // CUDA tensor pools must be placed explicitly; compile rejects an implicit device 0. Both live
  // on the render operator's device (see holoviz_cuda_device()).
  holoscan::CompileOptions options;
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "source", .output_port = "frame", .device = holoscan::DeviceId{device}});
  options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "render", .output_port = "output", .device = holoscan::DeviceId{device}});

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
    std::cerr << "cuda_sink_example failed: termination=" << static_cast<int>(session.termination())
              << " diagnostics=" << session.diagnostics().size() << '\n';
    return 2;
  }
  std::cout << "cuda_sink_example complete\n";
  return 0;
}
