/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Minimal text demo: the smallest per-frame text loop. Draws an FPS counter
// and a centered label through TextRenderer only — no ImGui headers anywhere
// in this file. By default the text geometry reaches the GPU through CUDA
// (cuMemcpyHtoDAsync into exported buffers) and the returned sync token
// orders the render behind the upload; with --monospace the push-constant
// monospace renderer is used instead (zero per-frame uploads, no-op token) —
// the rest of the loop is identical, demonstrating the renderers are drop-in
// interchangeable — and --fillrect additionally rasterizes each glyph as one
// triangle via VK_NV_fill_rectangle (requires the extension). For arbitrary
// debug UI (widgets, input), see examples/text/imgui.
//
// --benchmark runs every available renderer back to back over an identical
// per-frame text workload (imgui, monospace triangles, and monospace fill
// rect when the extension exists) and prints a timing breakdown per renderer
// (mean/std/p95/p99/max), measured like examples/core/cuda_interop_benchmark: GPU
// render spans via timestamp queries, upload spans via CUDA events on the
// shared stream, text build cost and frame-to-frame period on the host clock.

// example code

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda.h>

#include <holoviz/holoviz.hpp>
#include <holoviz/text.hpp>

#include "cuda_event_ring.hpp"
#include "measured_query_tracker.hpp"

#define CHECK_ERROR(expr)                                           \
  {                                                                 \
    auto result = (expr);                                           \
    if (!result) {                                                  \
      throw std::runtime_error("Error: " + result.error().message); \
    }                                                               \
  }

#define CHECK_CUDA_ERROR(expr)                                           \
  {                                                                      \
    const CUresult result = (expr);                                      \
    if (result != CUDA_SUCCESS) {                                        \
      throw std::runtime_error("CUDA Error: " + std::to_string(result)); \
    }                                                                    \
  }

namespace text = holoviz::text;

namespace {

constexpr uint32_t kWarmupFrames = 240;
constexpr std::chrono::milliseconds kMeasureDuration{5000};

struct Stats {
  double mean = 0.0;
  double std_dev = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
  double max = 0.0;
};

Stats compute_stats(const std::vector<double>& samples) {
  Stats s;
  if (samples.empty()) {
    return s;
  }
  for (const double sample : samples) {
    s.mean += sample;
    s.max = std::max(s.max, sample);
  }
  s.mean /= static_cast<double>(samples.size());
  if (samples.size() > 1) {
    double variance = 0.0;
    for (const double sample : samples) {
      variance += (sample - s.mean) * (sample - s.mean);
    }
    s.std_dev = std::sqrt(variance / static_cast<double>(samples.size() - 1));
  }
  std::vector<double> sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  const auto percentile = [&sorted](double p) {
    const size_t rank = static_cast<size_t>(std::ceil(p * static_cast<double>(sorted.size())));
    return sorted[std::min(rank == 0 ? 0 : rank - 1, sorted.size() - 1)];
  };
  s.p95 = percentile(0.95);
  s.p99 = percentile(0.99);
  return s;
}

struct BenchmarkResult {
  std::string label;
  double fps = 0.0;
  size_t frames = 0;
  Stats cpu_build;    // begin_frame..end_frame on the host (tessellation/packing + enqueue)
  Stats cuda_upload;  // stream-side span around end_frame (HtoD copies; ~0 for monospace)
  Stats gpu_render;   // timestamp-query span around the render pass
  Stats frame;        // host frame-to-frame period (jitter lives in std/p99/max)
};

/// The identical per-frame workload both renderers draw: an FPS-style header,
/// a centered title, and a HUD block of lines. Every line embeds the frame
/// index so the text differs each frame and nothing can be cached.
void add_benchmark_text(text::TextRenderer& renderer, holoviz::Extent2D extent,
                        uint64_t frame_index) {
  std::array<char, 96> line{};
  std::snprintf(
      line.data(), sizeof(line), "frame %09llu", static_cast<unsigned long long>(frame_index));
  CHECK_ERROR(renderer.add_text({.position = {20.0F, 20.0F}, .size_px = 20.0F}, line.data()));
  CHECK_ERROR(renderer.add_text(
      {
          .position = {static_cast<float>(extent.width) / 2.0F, 48.0F},
          .size_px = 32.0F,
          .color = 0xFF00D4FFu,
          .anchor = text::Anchor::kTopCenter,
      },
      "Holoviz Text Benchmark"));
  for (int i = 0; i < 12; ++i) {
    std::snprintf(line.data(),
                  sizeof(line),
                  "hud line %02d | frame %09llu | value % 8.3f | status OK 0123456789",
                  i,
                  static_cast<unsigned long long>(frame_index),
                  static_cast<double>((frame_index + static_cast<uint64_t>(i) * 37) % 10000) / 7.0);
    CHECK_ERROR(renderer.add_text(
        {.position = {20.0F, 100.0F + static_cast<float>(i) * 24.0F}, .size_px = 18.0F},
        line.data()));
  }
}

/// Run one renderer through warmup + a timed measurement window, instrumented
/// like examples/core/cuda_interop_benchmark.
// Keep the example resource lifecycle and frame synchronization visible in execution order.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
BenchmarkResult run_benchmark_pass(
    const std::string& label,
    const std::function<holoviz::expected<text::TextRenderer>()>& create_renderer,
    holoviz::Window& window, holoviz::Device& device, holoviz::Swapchain& swapchain,
    holoviz::RenderPass& render_pass, holoviz::QueryPool& query_pool,
    std::vector<holoviz::CommandList>& command_lists,
    std::vector<holoviz::SubmitSyncToken>& sync_tokens, CUstream cuda_stream) {
  auto renderer_result = create_renderer();
  if (!renderer_result) {
    throw std::runtime_error("Error: " + renderer_result.error().message);
  }
  auto renderer = std::move(*renderer_result);

  const uint32_t swapchain_image_count = swapchain.image_count().value();

  // CUDA event ring bracketing each frame's end_frame() stream work; depth is
  // one past the swapchain depth so a slot's previous pair has always been
  // recorded before it is recycled. The ring enforces collect-before-record,
  // so a measured frame never host-syncs on its own upload.
  holoviz::examples::CudaEventSpanRing upload_ring(swapchain_image_count + 1);

  // Per-image GPU-query ownership: an image's resolved range is only sampled
  // when the submit that wrote it ran inside the measured window, so the first
  // measured acquire of each image skips its warmup-owned range and the
  // end-of-run drain below flushes the measured tail the loop leaves pending.
  holoviz::examples::MeasuredQueryTracker query_tracker(swapchain_image_count);

  // Renderer-slot reuse tokens, one per buffer slot (the renderer's frames_in_flight, NOT per
  // swapchain image), keyed by the frame counter — see imgui::Renderer::begin_frame.
  const uint32_t renderer_frames_in_flight = swapchain_image_count + 1;
  std::vector<holoviz::SubmitSyncToken> slot_tokens(renderer_frames_in_flight,
                                                    holoviz::SubmitSyncToken{});

  std::vector<double> cpu_build_ms;
  std::vector<double> cuda_upload_ms;
  std::vector<double> gpu_render_ms;
  std::vector<double> frame_ms;
  bool collecting = false;
  uint64_t frame_index = 0;
  auto last_frame_end = std::chrono::steady_clock::now();

  const auto render_frame = [&]() {
    holoviz::poll_window_events();
    if (window.close_requested().value()) {
      throw std::runtime_error("window closed during benchmark");
    }

    auto acquired = swapchain.acquire(std::chrono::milliseconds(100)).value();
    auto& command_list = command_lists[acquired.image_index];
    const holoviz::Extent2D extent = swapchain.extent().value();

    // reset() blocks until this slot's previous submit completed, which is
    // also the submit that resolved its query range — safe to read now.
    CHECK_ERROR(command_list.reset(std::chrono::milliseconds(100)));
    if (query_tracker.should_collect(acquired.image_index)) {
      std::array<uint64_t, 2> span_ns = {};
      CHECK_ERROR(query_pool.results_ns(2 * acquired.image_index, 2, span_ns));
      gpu_render_ms.push_back(static_cast<double>(span_ns[1] - span_ns[0]) / 1.0e6);
      query_tracker.on_collect(acquired.image_index);
    }

    // Collect the span still held by this frame's ring slot — a full ring lap
    // old, so the sync is ~zero in steady state — BEFORE the slot is
    // re-recorded below (the ring throws if the order is ever inverted).
    if (collecting) {
      if (const auto upload_ms = upload_ring.collect(frame_index)) {
        cuda_upload_ms.push_back(*upload_ms);
      }
    }

    // Text build + upload, both instrumented: host clock around the whole
    // build, CUDA events around the stream work end_frame() enqueues (the
    // HtoD copies for the ImGui path; nothing for the monospace path).
    const auto build_start = std::chrono::steady_clock::now();
    const uint32_t renderer_slot = static_cast<uint32_t>(frame_index % renderer_frames_in_flight);
    CHECK_ERROR(renderer.begin_frame(extent, renderer_slot, slot_tokens[renderer_slot]));
    add_benchmark_text(renderer, extent, frame_index);
    upload_ring.record_start(frame_index, cuda_stream);
    const holoviz::ExternalSyncToken text_token = renderer.end_frame().value();
    upload_ring.record_end(frame_index, cuda_stream);
    if (collecting) {
      cpu_build_ms.push_back(
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - build_start)
              .count());
    }

    auto current_image = swapchain.image(acquired.image_index).value();
    CHECK_ERROR(render_pass.set_render_image(current_image, 0, {0.05F, 0.07F, 0.12F, 1.0F}));

    const holoviz::Rect2D render_area{0, 0, extent.width, extent.height};
    CHECK_ERROR(command_list.begin());
    CHECK_ERROR(command_list.reset_queries(query_pool, 2 * acquired.image_index, 2));
    CHECK_ERROR(command_list.write_timestamp(
        query_pool, 2 * acquired.image_index, holoviz::TimestampLocation::kStart));
    CHECK_ERROR(command_list.begin_render(render_pass, render_area));
    CHECK_ERROR(renderer.record(command_list, extent));
    CHECK_ERROR(command_list.end_render());
    CHECK_ERROR(command_list.write_timestamp(
        query_pool, 2 * acquired.image_index + 1, holoviz::TimestampLocation::kEnd));
    CHECK_ERROR(command_list.resolve_queries(query_pool, 2 * acquired.image_index, 2));
    CHECK_ERROR(command_list.end());

    const holoviz::SubmitDesc submit_desc{
        .wait_submit_sync_tokens = {&sync_tokens[acquired.image_index], 1},
        .wait_external_sync_tokens = {&text_token, 1},
        .command_lists = {&command_list, 1},
        .acquire_gates = {&acquired.acquire_gate, 1},
        .present_gates = {&acquired.present_gate, 1},
    };
    sync_tokens[acquired.image_index] = device.submit(submit_desc).value();
    // This submit consumed the renderer slot's draws; gate the slot's reuse on it.
    slot_tokens[renderer_slot] = sync_tokens[acquired.image_index];
    // This submit rewrote the image's query range; tag its measurement ownership.
    query_tracker.on_submit(acquired.image_index, collecting);
    // Read image_index before the move: function-argument evaluation order is
    // unsequenced, so reading through `acquired` in the same call is fragile.
    const uint32_t present_image_index = acquired.image_index;
    CHECK_ERROR(swapchain.present(std::move(acquired), sync_tokens[present_image_index]));

    const auto now = std::chrono::steady_clock::now();
    if (collecting) {
      frame_ms.push_back(std::chrono::duration<double, std::milli>(now - last_frame_end).count());
    }
    last_frame_end = now;
    ++frame_index;
  };

  // Warmup absorbs first-use pipeline compilation and fills the swapchain's
  // frames-in-flight slots before timing starts.
  for (uint32_t i = 0; i < kWarmupFrames; ++i) {
    render_frame();
  }

  // Forget the warmup frames' recorded spans so the first measured collects
  // return nothing instead of warmup samples; the frame counter stays
  // monotonic (a reset would desync the ring's slot-ownership checks).
  upload_ring.reset();
  collecting = true;
  last_frame_end = std::chrono::steady_clock::now();
  const auto measure_start = last_frame_end;
  std::chrono::milliseconds elapsed{};
  do {
    render_frame();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - measure_start);
  } while (elapsed < kMeasureDuration);

  // Drain before destroying this pass's renderer and the event ring (whose
  // destructor runs when this function returns).
  CHECK_ERROR(device.wait_idle());
  CHECK_CUDA_ERROR(cuStreamSynchronize(cuda_stream));
  // Flush the last ring lap of measured frames: their spans are recorded but
  // were never a full lap old inside the loop. collect(frame_index + i) visits
  // each slot exactly once, holding frames [frame_index - depth, frame_index).
  for (uint32_t i = 0; i < upload_ring.depth(); ++i) {
    if (const auto upload_ms = upload_ring.collect(frame_index + i)) {
      cuda_upload_ms.push_back(*upload_ms);
    }
  }
  // Likewise flush each image's final measured query: the loop only reads an
  // image's range on its NEXT acquire, so the last measured submit per image
  // is still pending here. wait_idle() above proved every range resolved.
  for (uint32_t image = 0; image < swapchain_image_count; ++image) {
    if (query_tracker.should_collect(image)) {
      std::array<uint64_t, 2> span_ns = {};
      CHECK_ERROR(query_pool.results_ns(2 * image, 2, span_ns));
      gpu_render_ms.push_back(static_cast<double>(span_ns[1] - span_ns[0]) / 1.0e6);
      query_tracker.on_collect(image);
    }
  }
  CHECK_ERROR(renderer.destroy());

  BenchmarkResult result;
  result.label = label;
  result.frames = frame_ms.size();
  result.fps =
      static_cast<double>(frame_ms.size()) / (static_cast<double>(elapsed.count()) / 1000.0);
  result.cpu_build = compute_stats(cpu_build_ms);
  result.cuda_upload = compute_stats(cuda_upload_ms);
  result.gpu_render = compute_stats(gpu_render_ms);
  result.frame = compute_stats(frame_ms);
  return result;
}

std::string fmt_ms(double ms) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3) << ms;
  return oss.str();
}

void print_result_table(const BenchmarkResult& result) {
  std::cout << "\n=== " << result.label << " ===\n";
  std::cout << result.frames << " frames measured, " << std::fixed << std::setprecision(1)
            << result.fps << " fps\n";

  const std::vector<std::pair<std::string, const Stats*>> columns = {
      {"CPU Build", &result.cpu_build},
      {"CUDA Upload", &result.cuda_upload},
      {"GPU Render", &result.gpu_render},
      {"Frame", &result.frame},
  };

  constexpr size_t kRowLabelWidth = 8;
  constexpr size_t kColWidth = 11;
  const auto print_hline = [&]() {
    std::cout << "+" << std::string(kRowLabelWidth + 2, '-');
    for (size_t i = 0; i < columns.size(); ++i) {
      std::cout << "+" << std::string(kColWidth + 2, '-');
    }
    std::cout << "+\n";
  };
  const auto print_row = [&](const std::string& row_label,
                             const std::function<std::string(const Stats&)>& cell) {
    std::cout << "| " << std::left << std::setw(static_cast<int>(kRowLabelWidth)) << row_label
              << " ";
    for (const auto& [label, stats] : columns) {
      std::cout << "| " << std::left << std::setw(static_cast<int>(kColWidth))
                << (stats != nullptr ? cell(*stats) : label) << " ";
    }
    std::cout << "|\n";
  };

  print_hline();
  std::cout << "| " << std::left << std::setw(static_cast<int>(kRowLabelWidth)) << "" << " ";
  for (const auto& [label, stats] : columns) {
    std::cout << "| " << std::left << std::setw(static_cast<int>(kColWidth)) << label << " ";
  }
  std::cout << "|\n";
  print_hline();
  print_row("Mean(ms)", [](const Stats& s) { return fmt_ms(s.mean); });
  print_row("Std(ms)", [](const Stats& s) { return fmt_ms(s.std_dev); });
  print_row("P95(ms)", [](const Stats& s) { return fmt_ms(s.p95); });
  print_row("P99(ms)", [](const Stats& s) { return fmt_ms(s.p99); });
  print_row("Max(ms)", [](const Stats& s) { return fmt_ms(s.max); });
  print_hline();
}

void print_comparison(const BenchmarkResult& from, const BenchmarkResult& to) {
  const auto delta = [](double from_ms, double to_ms) {
    std::ostringstream oss;
    oss << fmt_ms(from_ms) << " -> " << fmt_ms(to_ms) << " ms";
    if (from_ms > 0.0) {
      oss << " (" << std::showpos << std::fixed << std::setprecision(1)
          << (to_ms - from_ms) / from_ms * 100.0 << std::noshowpos << "%)";
    }
    return oss.str();
  };

  std::cout << "\n=== comparison: " << from.label << " -> " << to.label << " ===\n";
  std::cout << "Frame mean:   " << delta(from.frame.mean, to.frame.mean) << "\n";
  std::cout << "Frame p95:    " << delta(from.frame.p95, to.frame.p95) << "\n";
  std::cout << "Frame p99:    " << delta(from.frame.p99, to.frame.p99) << "\n";
  std::cout << "Frame max:    " << delta(from.frame.max, to.frame.max) << "\n";
  std::cout << "Frame jitter: " << delta(from.frame.std_dev, to.frame.std_dev) << "  (std dev)\n";
  std::cout << "CPU build:    " << delta(from.cpu_build.mean, to.cpu_build.mean) << "\n";
  std::cout << "CUDA upload:  " << delta(from.cuda_upload.mean, to.cuda_upload.mean) << "\n";
  std::cout << "GPU render:   " << delta(from.gpu_render.mean, to.gpu_render.mean) << "\n";
}

}  // namespace

// Keep the example resource lifecycle and frame synchronization visible in execution order.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
int main(int argc, char** argv) {
  const auto print_usage = [&](std::ostream& out) {
    out << "Usage: " << argv[0] << " [mode]\n"
        << "Modes (mutually exclusive; the default renders with the ImGui CUDA-upload "
           "renderer):\n"
        << "  --monospace  render with the push-constant monospace renderer (triangles)\n"
        << "  --fillrect   render with the monospace renderer via VK_NV_fill_rectangle\n"
        << "  --benchmark  run every available renderer over an identical workload and\n"
        << "               print a timing comparison\n"
        << "Options:\n"
        << "  -h, --help   print this message and exit\n";
  };

  if (argc > 2) {
    std::cerr << "Expected at most one argument, got " << (argc - 1) << "\n";
    print_usage(std::cerr);
    return EXIT_FAILURE;
  }
  std::string_view mode;  // empty = the default ImGui demo
  if (argc == 2) {
    const std::string_view arg(argv[1]);
    if (arg == "--help" || arg == "-h") {
      print_usage(std::cout);
      return EXIT_SUCCESS;
    }
    if (arg != "--monospace" && arg != "--fillrect" && arg != "--benchmark") {
      std::cerr << "Unknown argument: " << arg << "\n";
      print_usage(std::cerr);
      return EXIT_FAILURE;
    }
    mode = arg;
  }
  const bool benchmark = mode == "--benchmark";
  const bool fillrect = mode == "--fillrect";  // fill-rect is a monospace rasterization mode
  const bool monospace = fillrect || mode == "--monospace";

  // Initialize CUDA and bind the primary context before creating any exported resources.
  CHECK_CUDA_ERROR(cuInit(0));
  CUdevice cu_dev{};
  CHECK_CUDA_ERROR(cuDeviceGet(&cu_dev, 0));
  CUcontext cu_ctx{};
  CHECK_CUDA_ERROR(cuDevicePrimaryCtxRetain(&cu_ctx, cu_dev));
  CHECK_CUDA_ERROR(cuCtxSetCurrent(cu_ctx));
  CUstream cuda_stream{};
  CHECK_CUDA_ERROR(cuStreamCreate(&cuda_stream, CU_STREAM_NON_BLOCKING));

  const std::string title = benchmark   ? "Holoviz Text Benchmark"
                            : fillrect  ? "Holoviz Text Demo (monospace fill-rect)"
                            : monospace ? "Holoviz Text Demo (monospace)"
                                        : "Holoviz Text Demo";
  auto window = holoviz::create_window({.title = title, .size = {800, 600}}).value();
  auto surface = window.create_surface().value();
  auto device =
      holoviz::create_device_for_surface({.cuda_device = cu_dev, .cuda_context = cu_ctx}, surface)
          .value();

  // The benchmark needs the fastest present mode the surface supports; a
  // vsync-locked mode would cap both renderers at the display refresh rate
  // and mask their differences.
  auto swapchain = [&]() {
    // Prefer the latest-frame modes (both tear-free, both drop stale queued frames); only
    // FIFO is guaranteed by the spec, so it is the final fallback. In benchmark mode a
    // vsync-locked FIFO caps both renderers at the display refresh rate and masks their
    // differences, hence the note.
    auto latest_ready = device.create_swapchain({
        .surface = surface,
        .present_mode = holoviz::PresentMode::kFIFOLatestReady,
        .image_count = 3,
    });
    if (latest_ready) {
      std::cout << "Present mode: fifo latest ready\n";
      return std::move(*latest_ready);
    }
    auto mailbox = device.create_swapchain({
        .surface = surface,
        .present_mode = holoviz::PresentMode::kMailbox,
        .image_count = 3,
    });
    if (mailbox) {
      std::cout << "Present mode: mailbox\n";
      return std::move(*mailbox);
    }
    std::cout << "Present mode: fifo (latest-ready and mailbox unavailable"
              << (benchmark ? "; expect vsync-capped numbers" : "") << ")\n";
    return device
        .create_swapchain({
            .surface = surface,
            .present_mode = holoviz::PresentMode::kFIFO,
            .image_count = 3,
        })
        .value();
  }();
  const uint32_t swapchain_image_count = swapchain.image_count().value();

  const holoviz::RenderPassDesc render_pass_desc{.render_targets = {{
                                                     .format = swapchain.format().value(),
                                                 }}};
  auto render_pass = device.create_render_pass(render_pass_desc).value();

  std::vector<holoviz::CommandList> command_lists;
  for (uint32_t i = 0; i < swapchain_image_count; ++i) {
    command_lists.push_back(
        device.create_command_list({.type = holoviz::CommandListType::kTransient}).value());
  }
  std::vector<holoviz::SubmitSyncToken> sync_tokens(swapchain_image_count,
                                                    holoviz::SubmitSyncToken{});
  // Renderer-slot reuse tokens, one per buffer slot (frames_in_flight, NOT per swapchain
  // image), keyed by the frame counter — see imgui::Renderer::begin_frame. The benchmark path
  // keeps its own per-pass array inside run_benchmark_pass.
  const uint32_t renderer_frames_in_flight = swapchain_image_count + 1;
  std::vector<holoviz::SubmitSyncToken> slot_tokens(renderer_frames_in_flight,
                                                    holoviz::SubmitSyncToken{});
  uint32_t frame_counter = 0;

  if (benchmark) {
    auto query_pool = device
                          .create_query_pool({.timestamp_count = 2 * swapchain_image_count,
                                              .debug_name = "text benchmark gpu timer"})
                          .value();

    using RendererFactory = std::function<holoviz::expected<text::TextRenderer>()>;
    std::vector<std::pair<std::string, RendererFactory>> passes;
    passes.emplace_back("imgui (CUDA upload)", [&]() {
      return text::create_text_renderer(device,
                                        {
                                            .stream = cuda_stream,
                                            .frames_in_flight = swapchain_image_count + 1,
                                        });
    });
    passes.emplace_back("monospace (triangles)", [&]() {
      return text::create_monospace_text_renderer(
          device, {.fill_mode = text::MonospaceFillMode::kTriangles});
    });
    if (device.capabilities().value().fill_rectangle) {
      passes.emplace_back("monospace (fill rect)", [&]() {
        return text::create_monospace_text_renderer(
            device, {.fill_mode = text::MonospaceFillMode::kFillRectangle});
      });
    } else {
      std::cout << "Skipping the fill-rect pass: VK_NV_fill_rectangle unavailable\n";
    }

    std::vector<BenchmarkResult> results;
    for (const auto& [label, factory] : passes) {
      std::cout << "Running: " << label << "...\n";
      results.push_back(run_benchmark_pass(label,
                                           factory,
                                           window,
                                           device,
                                           swapchain,
                                           render_pass,
                                           query_pool,
                                           command_lists,
                                           sync_tokens,
                                           cuda_stream));
    }

    for (const BenchmarkResult& result : results) {
      print_result_table(result);
    }
    // Pairwise comparisons: imgui -> monospace triangles, then triangles -> fill rect.
    for (size_t i = 1; i < results.size(); ++i) {
      print_comparison(results[i - 1], results[i]);
    }

    CHECK_ERROR(device.wait_idle());
    CHECK_CUDA_ERROR(cuStreamSynchronize(cuda_stream));
    CHECK_CUDA_ERROR(cuStreamDestroy(cuda_stream));
    return EXIT_SUCCESS;
  }

  auto text_renderer_result =
      monospace ? text::create_monospace_text_renderer(
                      device,
                      {.fill_mode = fillrect ? text::MonospaceFillMode::kFillRectangle
                                             : text::MonospaceFillMode::kTriangles})
                : text::create_text_renderer(device,
                                             {
                                                 .stream = cuda_stream,
                                                 .frames_in_flight = swapchain_image_count + 1,
                                             });
  if (!text_renderer_result) {
    throw std::runtime_error("Error: " + text_renderer_result.error().message);
  }
  auto text_renderer = std::move(*text_renderer_result);

  float fps = 0.0F;
  auto last_frame_time = std::chrono::steady_clock::now();

  while (!window.close_requested().value()) {
    holoviz::poll_window_events();

    const auto now = std::chrono::steady_clock::now();
    const float delta_seconds = std::chrono::duration<float>(now - last_frame_time).count();
    last_frame_time = now;
    if (delta_seconds > 0.0F) {
      fps = 0.95F * fps + 0.05F / delta_seconds;  // rolling average
    }

    auto acquired = swapchain.acquire(std::chrono::milliseconds(100)).value();
    auto& command_list = command_lists[acquired.image_index];
    const holoviz::Extent2D extent = swapchain.extent().value();

    // Rebuild the frame's text and start the async CUDA upload.
    const uint32_t renderer_slot = frame_counter % renderer_frames_in_flight;
    CHECK_ERROR(text_renderer.begin_frame(extent, renderer_slot, slot_tokens[renderer_slot]));
    std::array<char, 64> fps_label{};
    std::snprintf(fps_label.data(), sizeof(fps_label), "%.1f fps", fps);
    CHECK_ERROR(
        text_renderer.add_text({.position = {20.0F, 20.0F}, .size_px = 20.0F}, fps_label.data()));
    CHECK_ERROR(text_renderer.add_text(
        {
            .position = {static_cast<float>(extent.width) / 2.0F,
                         static_cast<float>(extent.height) / 2.0F},
            .size_px = 40.0F,
            .color = 0xFF00D4FFu,  // amber (packed 0xAABBGGRR)
            .anchor = text::Anchor::kMiddleCenter,
        },
        "Hello, Holoviz!"));
    const holoviz::ExternalSyncToken text_token = text_renderer.end_frame().value();

    CHECK_ERROR(command_list.reset(std::chrono::milliseconds(100)));
    auto current_image = swapchain.image(acquired.image_index).value();
    CHECK_ERROR(render_pass.set_render_image(current_image, 0, {0.05F, 0.07F, 0.12F, 1.0F}));

    const holoviz::Rect2D render_area{0, 0, extent.width, extent.height};
    CHECK_ERROR(command_list.begin());
    CHECK_ERROR(command_list.begin_render(render_pass, render_area));
    CHECK_ERROR(text_renderer.record(command_list, extent));
    CHECK_ERROR(command_list.end_render());
    CHECK_ERROR(command_list.end());

    const holoviz::SubmitDesc submit_desc{
        .wait_submit_sync_tokens = {&sync_tokens[acquired.image_index], 1},
        .wait_external_sync_tokens = {&text_token, 1},
        .command_lists = {&command_list, 1},
        .acquire_gates = {&acquired.acquire_gate, 1},
        .present_gates = {&acquired.present_gate, 1},
    };
    sync_tokens[acquired.image_index] = device.submit(submit_desc).value();
    // This submit consumed the renderer slot's draws; gate the slot's reuse on it.
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

  CHECK_CUDA_ERROR(cuStreamSynchronize(cuda_stream));
  CHECK_CUDA_ERROR(cuStreamDestroy(cuda_stream));

  return EXIT_SUCCESS;
}
