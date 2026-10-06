/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Sub-frame visualizer: display a full video frame that arrives as horizontal bands
// ("sub-frames"), each band shown as soon as its content lands — the pattern of a camera
// sensor streaming slices of a frame, here simulated with a CUDA kernel. This recreates
// what a holoviz 4.x sub-frame visualizer did with RenderFlags, on the 5.x API:
//
//   4.x                                      5.x
//   ---------------------------------------  --------------------------------------------
//   DONT_CLEAR_COLOR / DONT_CLEAR_DEPTH      band pass with RenderTargetDesc::load =
//                                            LoadOp::kLoad (preserves the presentable
//                                            image outside and inside the render area)
//   DONT_SWAP_BUFFERS                        presentation is an explicit call: one
//                                            device.submit() per band on the same acquired
//                                            image, swapchain.present() after the last band
//   LayerAddView(0, off/H, 1, h/H)           begin_render(band_pass, band_rect) +
//                                            set_viewport(full) + set_scissor(band_rect)
//   ImageCudaDevice(...)                     CUDA writes into a ShareMode::kExported image,
//                                            each band's submit waits that band's
//                                            ExternalSemaphore signal
//   FPO path (never swap, exclusive)         --shared-continuous: the presentation engine
//                                            reads the single shared image continuously, so
//                                            after one initial present each band streams to
//                                            the display as its submit executes; the loop is
//                                            paced by first-pixel-out display events
//
// Because every frame renders all bands before the single present, the acquired image is
// fully overwritten each frame — no stale content from image_count presents ago can reach
// the display, which is what makes present-per-frame sub-frame rendering correct without
// the per-image convergence tracking a present-per-fragment renderer would need.
//
// README.md next to this file documents the options, the measurement, and the output.
//
//   holoviz_sub_frame [--bands N] [--frames N]              # windowed
//   holoviz_sub_frame --fullscreen [--monitor <name>]       # borderless fullscreen
//       [--width W] [--height H] [--hz R] [--fps N]
//   holoviz_sub_frame --display <name substring>            # exclusive scanout
//       [--width W] [--height H] [--hz R]
//       [--immediate | --fifo | --shared-demand | --shared-continuous]
//   ... [--no-latency] [--verify-timing]
//
// Use `holoviz_direct_display --list` to enumerate display names.
//
// Latency measurement: when the driver reports present timing (CapabilityReport::
// present_timing), the last band's command list writes two GPU timestamps (TOP_OF_PIPE
// before its render = "last band arrived", BOTTOM_OF_PIPE after it = "frame complete"),
// every present is tagged with an id, and the driver's first-pixel-out time for that id
// (Swapchain::past_present_timings(), device clock) is subtracted: frame-complete -> first
// pixel out with no host clock involved. Reported per second next to the frame-time line.

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
// example code

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <holoviz/holoviz.hpp>

#include "shader.slang.hpp"      // generated from shader.slang at build time
#include "shader_interface.hpp"  // PushConstants, shared with shader.slang

#include <cuda.h>

#include "pattern.hpp"

namespace shader = holoviz::examples::sub_frame;

#define CHECK_ERROR(expr)                                                                   \
  {                                                                                         \
    auto result = (expr);                                                                   \
    if (!result) {                                                                          \
      throw std::runtime_error(std::string("Error in `") + #expr + "` (code " +             \
                               std::to_string(static_cast<uint32_t>(result.error().code)) + \
                               "): " + result.error().message);                             \
    }                                                                                       \
  }

// Unwrap a holoviz::expected value, throwing a descriptive error (including the failing
// expression, error code, and message) instead of crashing via bad_expected_access.
template <typename Expected>
auto unwrap(Expected&& result, const char* what) {
  if (!result) {
    throw std::runtime_error(std::string("Error in `") + what + "` (code " +
                             std::to_string(static_cast<uint32_t>(result.error().code)) +
                             "): " + result.error().message);
  }
  return std::move(result).value();
}

#define UNWRAP(expr) unwrap((expr), #expr)

#define CHECK_CUDA_ERROR(expr)                                           \
  {                                                                      \
    CUresult result = (expr);                                            \
    if (result != CUDA_SUCCESS) {                                        \
      throw std::runtime_error("CUDA Error: " + std::to_string(result)); \
    }                                                                    \
  }

namespace {

/// Size of the simulated sensor frame the bands assemble into (the sampled source image;
/// the swapchain stretches it to the window/display extent).
constexpr uint32_t kFrameWidth = 800;
constexpr uint32_t kFrameHeight = 600;

struct Options {
  std::optional<int> exit_code;

  uint32_t bands = 8;
  uint32_t frames = 0;  ///< 0 = until the window closes (windowed) / 600 (display)
  bool have_display = false;
  std::string display_filter;
  bool fullscreen = false;   ///< borderless fullscreen window (mutually exclusive with --display)
  std::string monitor_name;  ///< GLFW monitor for --fullscreen; empty = primary
  holoviz::DisplayMode desired_mode{};
  double fps = 0.0;  ///< --fps: present-rate cap in a window (0 = unpaced); see the loop
  std::optional<holoviz::PresentMode> present_mode;
  bool no_latency = false;     ///< --no-latency: skip the frame-complete -> first-pixel-out stats
  bool verify_timing = false;  ///< --verify-timing: host cross-checks (serializes the loop)
  double fps_cap = 0.0;        ///< --fps-cap: sleep after each present to hold this rate (0 = off)
};

void print_usage(std::ostream& out, const char* program) {
  out << "Usage: " << program << " [options]\n"
      << "Options:\n"
      << "  -h, --help             print this message and exit\n"
      << "  --bands N              number of sub-frame bands per frame (default 8)\n"
      << "  --frames N             stop after N frames (default: windowed runs until the\n"
      << "                         window closes, display modes run 600 frames)\n"
      << "  --display <substring>  render to this display via exclusive scanout instead of\n"
      << "                         a window (holoviz_direct_display --list shows names)\n"
      << "  --fullscreen           borderless fullscreen window on the primary monitor (ESC or\n"
      << "                         close to exit); --width/--height/--hz pick the video mode\n"
      << "  --monitor <name>       GLFW monitor name for --fullscreen (default: primary). On\n"
      << "                         Linux also sets __GL_SYNC_DISPLAY_DEVICE=<name> so the\n"
      << "                         NVIDIA X11 driver paces presents on that monitor's vblank\n"
      << "                         instead of the primary's (an existing value is kept)\n"
      << "  --width W --height H   desired resolution (with --display or --fullscreen)\n"
      << "  --hz R                 desired refresh rate (with --display or --fullscreen)\n"
      << "  --fps N                cap the present rate at N per second in a window (default:\n"
      << "                         unpaced, i.e. the refresh rate under FIFO). Holding presents\n"
      << "                         below the panel's maximum refresh is needed for G-Sync/VRR\n"
      << "                         in a fullscreen window: --fullscreen --fifo --fps 80 (see\n"
      << "                         README.md for the other requirements)\n"
      << "  --immediate | --fifo   present mode (default: FIFO windowed, FIFO-latest-ready\n"
      << "                         on a display)\n"
      << "  --shared-demand | --shared-continuous   shared presentable image (with\n"
      << "                         --display): bands stream to the display as they render\n"
      << "  --fps-cap R            sleep after each present to hold R frames per second. With\n"
      << "                         --fifo this is the VRR test for a fullscreen window: the\n"
      << "                         first-pixel-out period equals 1/R under G-Sync and snaps to\n"
      << "                         multiples of the refresh period at fixed refresh\n"
      << "  --no-latency           disable the frame-complete -> first-pixel-out latency stats\n"
      << "  --verify-timing        cross-check the latency stats against wait_for_present and\n"
      << "                         host/GPU clock calibration (serializes the loop)\n";
}

Options parse_args(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg{argv[i]};
    const auto next_value = [&]() -> const char* {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + std::string(arg));
      }
      return argv[++i];
    };
    if (arg == "--help" || arg == "-h") {
      print_usage(std::cout, argv[0]);
      options.exit_code = EXIT_SUCCESS;
      return options;
    }
    if (arg == "--bands") {
      options.bands = std::stoul(next_value());
    } else if (arg == "--frames") {
      options.frames = std::stoul(next_value());
    } else if (arg == "--display" || arg == "-d") {
      options.display_filter = next_value();
      options.have_display = true;
    } else if (arg == "--fullscreen") {
      options.fullscreen = true;
    } else if (arg == "--monitor") {
      options.monitor_name = next_value();
    } else if (arg == "--width") {
      options.desired_mode.resolution.width = std::stoul(next_value());
    } else if (arg == "--height") {
      options.desired_mode.resolution.height = std::stoul(next_value());
    } else if (arg == "--hz") {
      options.desired_mode.refresh_rate_mhz =
          static_cast<uint32_t>(std::stof(next_value()) * 1000.F);
    } else if (arg == "--fps") {
      options.fps = std::stod(next_value());
    } else if (arg == "--immediate") {
      options.present_mode = holoviz::PresentMode::kImmediate;
    } else if (arg == "--fifo") {
      options.present_mode = holoviz::PresentMode::kFIFO;
    } else if (arg == "--shared-demand") {
      options.present_mode = holoviz::PresentMode::kSharedDemandRefresh;
    } else if (arg == "--shared-continuous") {
      options.present_mode = holoviz::PresentMode::kSharedContinuousRefresh;
    } else if (arg == "--fps-cap") {
      options.fps_cap = std::stod(next_value());
      if (options.fps_cap <= 0.0) {
        throw std::runtime_error("--fps-cap expects a positive rate");
      }
    } else if (arg == "--no-latency") {
      options.no_latency = true;
    } else if (arg == "--verify-timing") {
      options.verify_timing = true;
    } else if (!arg.empty() && arg.front() == '-') {
      std::cerr << "Unknown option: " << arg << "\n";
      print_usage(std::cerr, argv[0]);
      options.exit_code = EXIT_FAILURE;
      return options;
    } else {
      print_usage(std::cerr, argv[0]);
      options.exit_code = EXIT_FAILURE;
      return options;
    }
  }

  if (options.bands == 0 || options.bands > kFrameHeight) {
    throw std::runtime_error("--bands must be between 1 and " + std::to_string(kFrameHeight));
  }
  if (options.fullscreen && options.have_display) {
    throw std::runtime_error("--fullscreen and --display are mutually exclusive");
  }
  if (options.fps < 0.0 || !std::isfinite(options.fps)) {
    throw std::runtime_error("--fps must be a non-negative number");
  }
  if (options.fps > 0.0 && options.have_display) {
    throw std::runtime_error(
        "--fps applies to windows only; the --display loop paces on display events "
        "(use --hz-sweep in holoviz_direct_display to vary the present rate there)");
  }

  return options;
}

holoviz::Pipeline make_pipeline(holoviz::Device& device) {
  // Compiled to SPIR-V at build time from shader.slang (see CMakeLists.txt).
  const holoviz::ShaderBlob shader_blob = shader::make_shader_blob();

  holoviz::PipelineDesc pipeline_desc{
      .shader_blob = shader_blob,
      .topology = holoviz::PrimitiveTopology::kTriangleList,
      .raster =
          holoviz::RasterState{
              .cull = holoviz::CullMode::kNone,
          },
      .depth =
          holoviz::DepthState{
              .test = false,
              .write = false,
              .compare = holoviz::CompareOp::kAlways,
          },
  };
  return UNWRAP(device.create_pipeline(pipeline_desc));
}

/// Rows [y0, y1) of a height split into equal bands (the last band absorbs the remainder).
uint32_t band_offset(uint32_t band, uint32_t band_count, uint32_t height) {
  return static_cast<uint32_t>(static_cast<uint64_t>(band) * height / band_count);
}

/// avg/min/max accumulator over millisecond intervals.
/// Human-readable form of the engine's fixed/variable refresh claim.
const char* describe_refresh_mode(holoviz::RefreshMode mode) {
  switch (mode) {
    case holoviz::RefreshMode::kFixed:
      return "fixed refresh";
    case holoviz::RefreshMode::kVariable:
      return "driver reports variable refresh (period above is the minimum; verify with "
             "holoviz_direct_display's VRR probe)";
    case holoviz::RefreshMode::kUnknown:
      return "refresh mode not reported";
  }
  return "refresh mode not reported";
}

struct IntervalStats {
  uint32_t count = 0;
  double sum_ms = 0.0;
  double min_ms = std::numeric_limits<double>::infinity();
  double max_ms = 0.0;

  void add(double interval_ms) {
    ++count;
    sum_ms += interval_ms;
    min_ms = std::min(min_ms, interval_ms);
    max_ms = std::max(max_ms, interval_ms);
  }

  double average_ms() const { return count == 0 ? 0.0 : sum_ms / count; }

  /// Frame-time line: CPU-side frame-to-frame intervals measured after the present (or
  /// display-event pacing) of each frame, so they reflect the delivered frame cadence.
  void print_fps(std::ostream& out, const char* label) const {
    if (count == 0) {
      return;
    }
    out << label << std::fixed << std::setprecision(1) << 1000.0 / average_ms() << " fps"
        << std::setprecision(2) << " | frame time avg " << average_ms() << " ms, min " << min_ms
        << " ms, max " << max_ms << " ms (" << count << " frames)\n";
  }

  /// "avg X ms, min Y ms, max Z ms" (no trailing newline).
  void print_range(std::ostream& out) const {
    out << std::fixed << std::setprecision(2) << "avg " << average_ms() << " ms, min " << min_ms
        << " ms, max " << max_ms << " ms";
  }
};

/// Frame-complete -> first-pixel-out latency measurement. Two GPU timestamps bracket the
/// last band's render (see the file header); the driver's present-timing sample for the
/// frame's present id supplies the first-pixel-out time on the same device clock. The two
/// halves arrive asynchronously (the sample 1-3 refreshes after the present, the GPU stamps
/// one swapchain lap later when the image's command list is reset), so they are joined by
/// present id in `records`.
struct LatencyTracker {
  struct Record {
    std::optional<uint64_t> gpu_start_ns;  ///< last band's TOP_OF_PIPE: data arrival (upper bound)
    std::optional<uint64_t> gpu_end_ns;    ///< last band's BOTTOM_OF_PIPE: frame complete
    std::optional<uint64_t> first_pixel_out_ns;
    std::optional<std::chrono::steady_clock::time_point> present_wait_returned;  // --verify-timing
  };
  struct Counters {
    uint32_t used = 0;
    uint32_t dropped = 0;       ///< present() reported kFrameDropped
    uint32_t incomplete = 0;    ///< sample without first-pixel-out (never displayed)
    uint32_t no_sample = 0;     ///< present skipped by the engine; no sample ever came
    uint32_t recreated = 0;     ///< records lost to a swapchain recreation
    uint32_t anomalies = 0;     ///< first pixel out before render completion (clock trouble)
    uint32_t gpu_readback = 0;  ///< query results unavailable
    bool any() const {
      return dropped + incomplete + no_sample + recreated + anomalies + gpu_readback > 0;
    }
  };
  struct Stats {
    IntervalStats render_done_to_fpo;
    IntervalStats arrival_to_fpo;
    IntervalStats fpo_period;
    IntervalStats verify_wait_minus_fpo;  // --verify-timing
    Counters counters;
  };

  bool enabled = false;
  bool verify = false;
  const char* fpo_label = "FPO";  ///< "flip" for immediate-mode windows (not vblank-aligned)
  holoviz::QueryPool query_pool;
  uint32_t image_count = 0;
  std::vector<uint64_t> slot_present_id;  ///< image_index -> id whose stamps its slots hold
  std::map<uint64_t, Record> records;
  uint64_t refresh_ns = 0;
  uint64_t newest_sampled_id = 0;
  std::optional<uint64_t> previous_fpo_ns;
  std::optional<uint64_t> previous_fpo_id;
  std::optional<holoviz::CalibratedTimestamps> calibration;
  uint64_t max_calibration_deviation_ns = 0;
  Stats window;
  Stats total;
  bool warmed_up = false;

  uint32_t slot(uint32_t image_index) const { return 2 * image_index; }

  /// Read the GPU stamps the slot pair of \p image_index holds from the previous lap. Called
  /// right after the last band's command list reset(), which blocked on that lap's submit.
  void collect_slot(uint32_t image_index) {
    if (!enabled) {
      return;
    }
    const uint64_t id = slot_present_id[image_index];
    slot_present_id[image_index] = 0;
    if (id == 0) {
      return;
    }
    std::array<uint64_t, 2> stamps{};
    const auto read = query_pool.results_ns(slot(image_index), 2, stamps);
    auto it = records.find(id);
    if (!read) {
      ++window.counters.gpu_readback;
      if (it != records.end()) {
        records.erase(it);
      }
      return;
    }
    if (it == records.end()) {
      return;  // dropped or pruned frame: stamps without a present
    }
    it->second.gpu_start_ns = stamps[0];
    it->second.gpu_end_ns = stamps[1];
    try_finalize(it);
  }

  /// Record the GPU timestamps and their resolve into the last band's command list.
  void record_start(holoviz::CommandList& command_list, uint32_t image_index) {
    if (!enabled) {
      return;
    }
    CHECK_ERROR(command_list.reset_queries(query_pool, slot(image_index), 2));
    // TOP_OF_PIPE right after the submit's semaphore waits (all-commands stage): the earliest
    // point the GPU can have observed the last band's CUDA signal. Upper bound on arrival.
    CHECK_ERROR(command_list.write_timestamp(
        query_pool, slot(image_index), holoviz::TimestampLocation::kStart));
  }
  void record_end(holoviz::CommandList& command_list, uint32_t image_index) {
    if (!enabled) {
      return;
    }
    CHECK_ERROR(command_list.write_timestamp(
        query_pool, slot(image_index) + 1, holoviz::TimestampLocation::kEnd));
    CHECK_ERROR(command_list.resolve_queries(query_pool, slot(image_index), 2));
  }

  void on_presented(uint32_t image_index, uint64_t present_id) {
    if (!enabled) {
      return;
    }
    slot_present_id[image_index] = present_id;
    records[present_id] = {};
  }
  void on_dropped() {
    if (enabled) {
      ++window.counters.dropped;
    }
  }
  void on_recreated(holoviz::Swapchain& swapchain) {
    if (!enabled) {
      return;
    }
    window.counters.recreated += static_cast<uint32_t>(records.size());
    records.clear();
    std::fill(slot_present_id.begin(), slot_present_id.end(), 0);
    previous_fpo_ns.reset();
    previous_fpo_id.reset();
    refresh_ns = swapchain.refresh_duration_ns().value_or(refresh_ns);
  }

  /// Pull every completed timing sample and join it with its record.
  void drain(holoviz::Swapchain& swapchain) {
    if (!enabled) {
      return;
    }
    if (refresh_ns == 0) {
      refresh_ns = swapchain.refresh_duration_ns().value_or(0);
    }
    std::array<holoviz::PresentTimingSample, 16> samples{};
    uint32_t drained = 0;
    do {
      auto result = swapchain.past_present_timings(samples);
      if (!result) {
        std::cout << "Latency stats: past_present_timings failed (" << result.error().message
                  << "); disabling\n";
        enabled = false;
        return;
      }
      drained = *result;
      for (uint32_t i = 0; i < drained; ++i) {
        const holoviz::PresentTimingSample& sample = samples[i];
        if (sample.time_domain != holoviz::TimeDomain::kDevice) {
          std::cout << "Latency stats: present timing is not on the device clock here; "
                       "disabling\n";
          enabled = false;
          return;
        }
        newest_sampled_id = std::max(newest_sampled_id, sample.present_id);
        auto it = records.find(sample.present_id);
        if (!sample.complete || !sample.first_pixel_out_ns) {
          ++window.counters.incomplete;
          if (it != records.end()) {
            records.erase(it);
          }
          continue;
        }
        if (previous_fpo_ns && previous_fpo_id && *previous_fpo_id + 1 == sample.present_id) {
          add_interval(&Stats::fpo_period, ns_to_ms(*sample.first_pixel_out_ns - *previous_fpo_ns));
        }
        previous_fpo_ns = sample.first_pixel_out_ns;
        previous_fpo_id = sample.present_id;
        if (it == records.end()) {
          ++window.counters.no_sample;  // record already pruned
          continue;
        }
        it->second.first_pixel_out_ns = sample.first_pixel_out_ns;
        try_finalize(it);
      }
    } while (drained == samples.size());

    // A present the engine skipped (FIFO-latest-ready / mailbox) never yields a sample: prune
    // records the sample stream has moved well past.
    const uint64_t horizon = 2ULL * image_count + 8;
    for (auto it = records.begin(); it != records.end();) {
      if (it->first + horizon < newest_sampled_id) {
        ++window.counters.no_sample;
        it = records.erase(it);
      } else {
        ++it;
      }
    }
  }

  static double ns_to_ms(uint64_t ns) { return static_cast<double>(ns) / 1e6; }

  void add_interval(IntervalStats Stats::* member, double ms) {
    (window.*member).add(ms);
    if (warmed_up) {
      (total.*member).add(ms);
    }
  }

  void try_finalize(std::map<uint64_t, Record>::iterator it) {
    Record& record = it->second;
    if (!record.gpu_end_ns || !record.first_pixel_out_ns) {
      return;
    }
    if (*record.first_pixel_out_ns < *record.gpu_end_ns ||
        (record.gpu_start_ns && *record.gpu_start_ns > *record.gpu_end_ns)) {
      ++window.counters.anomalies;
      records.erase(it);
      return;
    }
    add_interval(&Stats::render_done_to_fpo,
                 ns_to_ms(*record.first_pixel_out_ns - *record.gpu_end_ns));
    if (record.gpu_start_ns) {
      add_interval(&Stats::arrival_to_fpo,
                   ns_to_ms(*record.first_pixel_out_ns - *record.gpu_start_ns));
    }
    if (verify && calibration && record.present_wait_returned) {
      const auto fpo_host_ns = calibration->to_host_ns(*record.first_pixel_out_ns);
      const auto wait_ns = static_cast<uint64_t>(record.present_wait_returned->time_since_epoch() /
                                                 std::chrono::nanoseconds(1));
      add_interval(&Stats::verify_wait_minus_fpo,
                   (static_cast<double>(wait_ns) - static_cast<double>(fpo_host_ns)) / 1e6);
    }
    ++window.counters.used;
    if (warmed_up) {
      ++total.counters.used;
    }
    records.erase(it);
  }

  void recalibrate(holoviz::Device& device) {
    if (!enabled || !verify) {
      return;
    }
    if (auto calibrated = device.calibrated_timestamps()) {
      calibration = *calibrated;
      max_calibration_deviation_ns =
          std::max(max_calibration_deviation_ns, calibrated->max_deviation_ns);
    }
  }

  void print_stats(std::ostream& out, const Stats& stats, const char* indent) const {
    if (stats.render_done_to_fpo.count == 0) {
      out << indent << "latency: no complete samples\n";
    } else {
      out << indent << "render-done->" << fpo_label << " ";
      stats.render_done_to_fpo.print_range(out);
      out << " (" << stats.render_done_to_fpo.count << " samples)\n";
      out << indent << "band-arrival->" << fpo_label << " ";
      stats.arrival_to_fpo.print_range(out);
      out << std::fixed << std::setprecision(2) << " | last-pixel-out +" << ns_to_ms(refresh_ns)
          << " ms\n";
      if (stats.fpo_period.count > 0) {
        out << indent << fpo_label << " period ";
        stats.fpo_period.print_range(out);
        out << "\n";
      }
    }
    if (verify && stats.verify_wait_minus_fpo.count > 0) {
      out << indent << "verify: wait_for_present - " << fpo_label << " ";
      stats.verify_wait_minus_fpo.print_range(out);
      out << std::fixed << std::setprecision(1) << " | calibration deviation <= "
          << static_cast<double>(max_calibration_deviation_ns) / 1e3 << " us\n";
    }
    if (stats.counters.any()) {
      const Counters& c = stats.counters;
      out << indent << "lost: " << c.dropped << " dropped, " << c.incomplete << " incomplete, "
          << c.no_sample << " no-sample, " << c.recreated << " recreated, " << c.anomalies
          << " anomalies, " << c.gpu_readback << " gpu-readback\n";
    }
  }

  /// End of a one-second window: print it (unless \p out is null), fold its counters into
  /// the total, and reset.
  void end_window(std::ostream* out, const char* indent) {
    if (!enabled) {
      return;
    }
    if (out != nullptr) {
      print_stats(*out, window, indent);
    }
    if (warmed_up) {
      Counters& t = total.counters;
      const Counters& w = window.counters;
      t.dropped += w.dropped;
      t.incomplete += w.incomplete;
      t.no_sample += w.no_sample;
      t.recreated += w.recreated;
      t.anomalies += w.anomalies;
      t.gpu_readback += w.gpu_readback;
    }
    window = {};
  }
};

int run(int argc, char** argv) {
  const Options options = parse_args(argc, argv);
  if (options.exit_code) {
    return *options.exit_code;
  }

  const bool shared_mode = options.present_mode == holoviz::PresentMode::kSharedDemandRefresh ||
                           options.present_mode == holoviz::PresentMode::kSharedContinuousRefresh;
  const bool continuous_mode =
      options.present_mode == holoviz::PresentMode::kSharedContinuousRefresh;
  if (shared_mode && !options.have_display) {
    throw std::runtime_error("Shared present modes require --display (exclusive scanout)");
  }
  const holoviz::PresentMode present_mode = options.present_mode.value_or(
      options.have_display ? holoviz::PresentMode::kFIFOLatestReady : holoviz::PresentMode::kFIFO);
  const uint32_t frame_count = options.frames > 0     ? options.frames
                               : options.have_display ? 600
                                                      : 0;  // 0 = until window close
  const uint32_t bands = options.bands;

  std::cout << argv[0] << " - Holoviz sub-frame visualizer (" << bands << " bands)\n";

#ifndef _WIN32
  if (options.fullscreen && !options.monitor_name.empty()) {
    // The NVIDIA X11 driver paces FIFO presents on the primary display's vblank unless
    // __GL_SYNC_DISPLAY_DEVICE names the output to sync to; a fullscreen window on a
    // secondary monitor would otherwise run at the primary's refresh rate. The driver reads
    // it when the Vulkan ICD loads, so it must be set before the device is created. GLFW's
    // X11 monitor names are the RandR output names the driver expects.
    constexpr const char* kSyncDisplayEnv = "__GL_SYNC_DISPLAY_DEVICE";
    if (const char* existing = std::getenv(kSyncDisplayEnv); existing != nullptr) {
      std::cout << kSyncDisplayEnv << " already set to \"" << existing << "\", keeping it\n";
    } else if (setenv(kSyncDisplayEnv, options.monitor_name.c_str(), 0) == 0) {
      std::cout << "Set " << kSyncDisplayEnv << "=" << options.monitor_name << "\n";
    }
  }
#endif

  CHECK_CUDA_ERROR(cuInit(0));

  std::optional<holoviz::Window> window;
  std::optional<holoviz::Display> display;
  if (options.have_display) {
    display = UNWRAP(holoviz::get_display(options.display_filter));
    std::cout << "Latching onto display \"" << display->name.data() << "\"\n";
  } else {
    window = UNWRAP(holoviz::create_window({
        .title = "Holoviz Sub-Frame Visualizer",
        // Fullscreen: {0, 0} keeps the monitor's current video mode unless --width/--height
        // were given; windowed: the frame size.
        .size = options.fullscreen ? options.desired_mode.resolution
                                   : holoviz::Extent2D{kFrameWidth, kFrameHeight},
        .resizable = !options.fullscreen,
        .fullscreen = options.fullscreen,
        .display_name = options.monitor_name,
        .refresh_rate_mhz = options.fullscreen ? options.desired_mode.refresh_rate_mhz : 0U,
        .close_on_escape = options.fullscreen,  // an undecorated window has no close button
    }));
    if (options.fullscreen) {
      const holoviz::Extent2D extent = UNWRAP(window->extent());
      // The window system warns and falls back to the primary monitor for unknown names.
      const std::string placement = options.monitor_name.empty()
                                        ? " on the primary monitor"
                                        : ", requested monitor \"" + options.monitor_name + "\"";
      std::cout << "Fullscreen window " << extent.width << "x" << extent.height << placement
                << "\n";
    }
  }
  holoviz::Surface surface = options.have_display
                                 ? UNWRAP(display->create_surface({.mode = options.desired_mode}))
                                 : UNWRAP(window->create_surface());

  // Let the module pick the adapter that presents to this surface, then do all CUDA work on
  // that adapter's CUDA device: on a multi-GPU host the display is not necessarily on CUDA
  // device 0, and CUDA-exported images must live on the GPU that scans them out.
  auto device = UNWRAP(holoviz::create_device_for_surface({}, surface));
  std::cout << "Using device: " << UNWRAP(device.adapter()).name.data() << "\n";
  const CUdevice cu_dev = UNWRAP(device.cuda_device());
  CUcontext cu_ctx{};
  CHECK_CUDA_ERROR(cuDevicePrimaryCtxRetain(&cu_ctx, cu_dev));
  CHECK_CUDA_ERROR(cuCtxSetCurrent(cu_ctx));
  CUstream cuda_stream{};
  CHECK_CUDA_ERROR(cuStreamCreate(&cuda_stream, CU_STREAM_NON_BLOCKING));

  // Latency stats need per-frame present ids (none in the shared modes) and the driver's
  // present timing; a missing capability disables them with one status line, never an error.
  const holoviz::CapabilityReport capabilities = UNWRAP(device.capabilities());
  LatencyTracker latency;
  std::string latency_status;
  if (shared_mode) {
    latency_status = "n/a in shared present modes (no per-frame present ids)";
  } else if (options.no_latency) {
    latency_status = "disabled (--no-latency)";
  } else if (!capabilities.present_timing) {
    latency_status = "unavailable on this driver (no present timing); latency stats disabled";
  } else {
    latency.enabled = true;
  }
  latency.verify = options.verify_timing && capabilities.calibrated_timestamps &&
                   capabilities.present_completion;
  if (options.verify_timing && !latency.verify) {
    std::cout << "--verify-timing needs calibrated timestamps and present-wait; skipped\n";
  }

  holoviz::SwapchainDesc swapchain_desc{
      .surface = surface,
      .present_mode = present_mode,
      // Shared modes use the single shared image; a display gets the low-latency depth.
      .image_count = shared_mode ? 1U : (options.have_display ? 2U : 3U),
      .enable_present_timing = latency.enabled,
  };
  auto swapchain_result = device.create_swapchain(swapchain_desc);
  if (!swapchain_result && latency.enabled &&
      swapchain_result.error().code == holoviz::ErrorCode::kInvalidArgument) {
    latency.enabled = false;
    latency_status = "unavailable on this surface; latency stats disabled";
    swapchain_desc.enable_present_timing = false;
    swapchain_result = device.create_swapchain(swapchain_desc);
  }
  auto swapchain = unwrap(std::move(swapchain_result), "device.create_swapchain(swapchain_desc)");
  uint32_t image_count = UNWRAP(swapchain.image_count());
  const holoviz::Format swapchain_format = UNWRAP(swapchain.format());
  holoviz::RefreshMode reported_refresh_mode = holoviz::RefreshMode::kUnknown;

  if (latency.enabled) {
    latency.image_count = image_count;
    latency.slot_present_id.assign(image_count, 0);
    latency.query_pool = UNWRAP(device.create_query_pool(
        {.timestamp_count = 2 * image_count, .debug_name = "sub-frame latency"}));
    latency.refresh_ns = swapchain.refresh_duration_ns().value_or(0);
    latency.recalibrate(device);
    if (present_mode == holoviz::PresentMode::kImmediate && !options.have_display) {
      // On a composited window an immediate present's "first pixel out" is the flip, not a
      // vblank-aligned scanout; direct display (or FIFO) gives real scanout timing.
      latency.fpo_label = "flip";
    }
    const holoviz::SurfaceCapabilities surface_caps = UNWRAP(device.surface_capabilities(surface));
    std::cout << "Present timing: device clock, stages"
              << (surface_caps.present_stages.contains(holoviz::PresentStage::kQueueOperationsEnd)
                      ? " queue-ops-end"
                      : "")
              << (surface_caps.present_stages.contains(holoviz::PresentStage::kFirstPixelOut)
                      ? " first-pixel-out"
                      : "")
              << (surface_caps.present_stages.contains(holoviz::PresentStage::kFirstPixelVisible)
                      ? " first-pixel-visible"
                      : "");
    if (latency.refresh_ns > 0) {
      std::cout << std::fixed << std::setprecision(3) << "; refresh "
                << LatencyTracker::ns_to_ms(latency.refresh_ns) << " ms (" << std::setprecision(1)
                << 1e9 / static_cast<double>(latency.refresh_ns) << " Hz)";
    } else {
      std::cout << "; refresh not reported yet";
    }
    // The engine's own fixed/variable claim (VK_EXT_present_timing refreshInterval). It can
    // change once frames are on the display, so the loop re-reads it every second and prints
    // transitions; holoviz_direct_display's probe verifies the claim against the vblank
    // counter, which this example does not do.
    reported_refresh_mode = swapchain.refresh_mode().value_or(holoviz::RefreshMode::kUnknown);
    std::cout << "; " << describe_refresh_mode(reported_refresh_mode);
    std::cout << "\n";
    if (latency.fpo_label[0] == 'f' && latency.fpo_label[1] == 'l') {
      std::cout << "Note: immediate mode on a window reports flip time, not vblank scanout "
                   "(use --fifo or --display)\n";
    }
  } else {
    std::cout << "Latency stats: " << latency_status << "\n";
  }

  // On an exclusive display the loop paces itself on first-pixel-out display events in every
  // mode — real display timing or nothing, no silent fallback (the sub-frame contract).
  // Shared and immediate presents never block on the display, and on current NVIDIA drivers
  // FIFO / FIFO-latest-ready presents to a direct-display (VRR) surface do not block either
  // (present-wait returns within 0.1 ms), so an
  // unpaced loop would present thousands of frames per refresh. Windowed modes pace through
  // acquire/present.
  const bool pace_on_display_events = options.have_display;
  // --fps: host-side pacing for the window paths. Under FIFO an unpaced window presents
  // once per refresh, so on a VRR panel the refresh has nothing to follow and the driver
  // stays at the mode's maximum ("NORMAL" on the G-Sync indicator). Holding the present rate
  // below that maximum is one requirement for variable refresh in a fullscreen window (the
  // other, on NVIDIA X11, is that the window flips, which needs the panel to be the only
  // display on the X screen; see README.md); `FPO period` then tracks the present period
  // instead of quantizing to the refresh.
  const bool pace_on_host = !pace_on_display_events && options.fps > 0.0;
  const auto host_frame_period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(pace_on_host ? 1.0 / options.fps : 0.0));
  auto next_present_deadline = std::chrono::steady_clock::now();
  if (pace_on_host) {
    std::cout << std::fixed << std::setprecision(1) << "Present pacing: " << options.fps
              << " fps (host sleep)\n";
  }

  // The full frame the bands assemble into, sampled by the band draws and written by CUDA
  // band by band through the exported mipmapped array.
  auto image = UNWRAP(device.create_image({
      .extent = {kFrameWidth, kFrameHeight},
      .format = holoviz::Format::kR8G8B8A8Unorm,
      .usage = holoviz::ImageUsage::kSampled,
      .share = holoviz::ShareMode::kExported,
  }));
  CUarray image_level0{};
  CHECK_CUDA_ERROR(
      cuMipmappedArrayGetLevel(&image_level0, UNWRAP(image.cuda_mipmapped_array()), 0));

  // The band-sized staging buffer stands in for the per-sub-frame tensor the bands arrive
  // in; sized for the largest band (the last one absorbs the division remainder).
  const uint32_t max_band_height = kFrameHeight - band_offset(bands - 1, bands, kFrameHeight);
  const size_t band_pitch = static_cast<size_t>(kFrameWidth) * 4;
  CUdeviceptr band_buffer{};
  CHECK_CUDA_ERROR(cuMemAlloc(&band_buffer, band_pitch * max_band_height));

  // Full-clear pass for each swapchain image's first use: it moves the image into its
  // presentation home layout, which the kLoad band pass below loads from.
  auto init_pass = UNWRAP(device.create_render_pass(
      {.render_targets = {{.format = swapchain_format, .shared_present = shared_mode}}}));
  // The band pass — the 4.x DONT_CLEAR_COLOR|DONT_CLEAR_DEPTH analogue: LoadOp::kLoad
  // keeps the presentable image's existing contents, so each pass adds one band and the
  // bands accumulate. (Depth has no equivalent knob in 5.0; this example needs none.)
  auto band_pass = UNWRAP(device.create_render_pass({.render_targets = {{
                                                         .load = holoviz::LoadOp::kLoad,
                                                         .format = swapchain_format,
                                                         .shared_present = shared_mode,
                                                     }}}));

  auto pipeline = make_pipeline(device);
  auto descriptor = UNWRAP(pipeline.create_descriptor());
  CHECK_ERROR(descriptor.bind_image(image, "frameTexture"));
  CHECK_ERROR(descriptor.bind_sampler(device, {}, "frameSampler"));

  auto external_semaphore = UNWRAP(device.create_external_semaphore(cuda_stream));

  // One transient command list per (swapchain image, band): each band is recorded and
  // submitted the moment its content is ready, like the 4.x operator's per-tensor
  // Begin()/End() pair.
  std::vector<holoviz::CommandList> command_lists;
  command_lists.reserve(static_cast<size_t>(image_count) * bands);
  for (uint32_t i = 0; i < image_count * bands; ++i) {
    command_lists.push_back(
        UNWRAP(device.create_command_list({.type = holoviz::CommandListType::kTransient})));
  }

  uint64_t generation = UNWRAP(swapchain.generation());
  std::vector<bool> image_initialized(image_count, false);
  holoviz::SubmitSyncToken last_frame_token{};
  bool presentation_started = false;

  // Frame pacing statistics: a one-second window is printed as it fills, and the run is
  // summarized on exit. The first window is warm-up (swapchain images filling, pipeline and
  // mode settling) and is excluded from the summary.
  IntervalStats window_stats;
  IntervalStats total_stats;
  bool warmed_up = false;
  std::optional<std::chrono::steady_clock::time_point> last_frame_end;
  auto window_start = std::chrono::steady_clock::now();
  // Present ids are strictly increasing per swapchain and decoupled from `frame`: only a
  // present that entered the sequence (PresentConsumed::id) advances the counter.
  uint64_t next_present_id = 1;

  // A recreation (window resize) changed the number of swapchain images, reported by acquire()
  // or present() as kImageCountChanged: rebuild every per-image resource for the new count.
  // The device is idled first so the old command lists and query slots are no longer in use.
  auto rebuild_per_image_resources = [&]() {
    const uint32_t new_image_count = UNWRAP(swapchain.image_count());
    std::cout << "Swapchain image count changed: " << image_count << " -> " << new_image_count
              << "; rebuilding per-image resources\n";
    CHECK_ERROR(device.wait_idle());
    for (auto& list : command_lists) {
      CHECK_ERROR(list.destroy());
    }
    command_lists.clear();
    command_lists.reserve(static_cast<size_t>(new_image_count) * bands);
    for (uint32_t i = 0; i < new_image_count * bands; ++i) {
      command_lists.push_back(
          UNWRAP(device.create_command_list({.type = holoviz::CommandListType::kTransient})));
    }
    image_count = new_image_count;
    image_initialized.assign(image_count, false);
    generation = UNWRAP(swapchain.generation());
    if (latency.enabled) {
      CHECK_ERROR(latency.query_pool.destroy());
      latency.query_pool = UNWRAP(device.create_query_pool(
          {.timestamp_count = 2 * image_count, .debug_name = "sub-frame latency"}));
      latency.image_count = image_count;
      latency.slot_present_id.assign(image_count, 0);
      latency.on_recreated(swapchain);
    }
  };

  uint32_t frame = 0;
  while ((frame_count == 0 || frame < frame_count) &&
         !(window && UNWRAP(window->close_requested()))) {
    if (window) {
      holoviz::poll_window_events();
    }

    // In continuous shared mode the presentation engine reads the shared image on its own
    // once started: no per-frame acquire/present at all — bands stream straight to the
    // display as their submits execute, the 4.x "never swap buffers" FPO path.
    std::optional<holoviz::AcquireSyncToken> acquired;
    if (!(continuous_mode && presentation_started)) {
      auto acquire_result = swapchain.acquire(std::chrono::milliseconds(100));
      if (!acquire_result) {
        if (acquire_result.error().code == holoviz::ErrorCode::kBackpressured ||
            acquire_result.error().code == holoviz::ErrorCode::kOutOfDate) {
          continue;
        }
        if (acquire_result.error().code == holoviz::ErrorCode::kImageCountChanged) {
          rebuild_per_image_resources();
          continue;
        }
        throw std::runtime_error(
            "Error in `swapchain.acquire()` (code " +
            std::to_string(static_cast<uint32_t>(acquire_result.error().code)) +
            "): " + acquire_result.error().message);
      }
      acquired = std::move(acquire_result).value();
    }
    const uint32_t image_index = acquired ? acquired->image_index : 0;

    // A window resize recreated the swapchain in place: the images are new, so their
    // first-use clears run again.
    if (const uint64_t current_generation = UNWRAP(swapchain.generation());
        current_generation != generation) {
      generation = current_generation;
      image_initialized.assign(image_count, false);
      latency.on_recreated(swapchain);
    }

    const holoviz::Image& current_image = UNWRAP(swapchain.image(image_index));
    CHECK_ERROR(band_pass.set_render_image(current_image, 0, {0.0F, 0.0F, 0.0F, 1.0F}));
    const holoviz::Extent2D extent = UNWRAP(swapchain.extent());
    const holoviz::Rect2D full_area{0, 0, extent.width, extent.height};

    // Texture reuse hazard: the previous frame's draws (possibly still in flight on
    // another swapchain image) sample `image` — hold the CUDA stream until they finish
    // before overwriting any band. Within a frame each band's draw reads only the rows its
    // own upload wrote (the shader clamps its sampling to them, see shader_interface.hpp),
    // so the per-band upload/draw pairs need no ordering against each other.
    CHECK_ERROR(external_semaphore.wait(last_frame_token));

    // Shared-mode acquires after the first mint pre-consumed gates (there is no
    // presentation-engine handshake to synchronize with) — they must not go to submit().
    const bool use_gates = acquired && !UNWRAP(acquired->acquire_gate.consumed());

    holoviz::SubmitSyncToken previous_band_token = last_frame_token;
    for (uint32_t band = 0; band < bands; ++band) {
      // --- Producer: the sub-frame "tensor" arrives (CUDA pattern into the band buffer,
      // then into the band's rows of the full-frame image). ---
      const uint32_t y0 = band_offset(band, bands, kFrameHeight);
      const uint32_t band_height = band_offset(band + 1, bands, kFrameHeight) - y0;
      launch_band_pattern(
          band_buffer, band_pitch, kFrameWidth, band_height, y0, kFrameHeight, frame, cuda_stream);
      CUDA_MEMCPY2D copy_desc{};
      copy_desc.srcMemoryType = CU_MEMORYTYPE_DEVICE;
      copy_desc.srcDevice = band_buffer;
      copy_desc.srcPitch = band_pitch;
      copy_desc.dstMemoryType = CU_MEMORYTYPE_ARRAY;
      copy_desc.dstArray = image_level0;
      copy_desc.dstY = y0;
      copy_desc.WidthInBytes = band_pitch;
      copy_desc.Height = band_height;
      CHECK_CUDA_ERROR(cuMemcpy2DAsync(&copy_desc, cuda_stream));
      auto external_sync_token = UNWRAP(external_semaphore.signal());

      // --- Display side: render and submit this band the moment its content is ready. ---
      auto& command_list = command_lists[static_cast<size_t>(image_index) * bands + band];
      const bool last_band = band == bands - 1;
      CHECK_ERROR(command_list.reset(std::chrono::milliseconds(100)));
      if (last_band) {
        // reset() blocked on this list's previous submit, which resolved the previous lap's
        // timestamps for this image: read them before re-recording the slots.
        latency.collect_slot(image_index);
      }
      CHECK_ERROR(command_list.begin());
      if (last_band) {
        latency.record_start(command_list, image_index);
      }
      if (band == 0 && !image_initialized[image_index]) {
        CHECK_ERROR(init_pass.set_render_image(current_image, 0, {0.0F, 0.0F, 0.0F, 1.0F}));
        CHECK_ERROR(command_list.begin_render(init_pass, full_area));
        CHECK_ERROR(command_list.end_render());
        image_initialized[image_index] = true;
      }
      // The band's slice of the presentable image, the 4.x LayerAddView analogue: the
      // render area scopes the pass, the full-extent viewport keeps the quad's UVs
      // full-frame, and the scissor clips the draw to the band. The scissor bounds only the
      // written pixels; the push constants bound the *read* rows to this band's source rows
      // (texel centres), so linear filtering at the band edges cannot pull in a neighbouring
      // band whose upload this frame has not landed yet.
      //
      // Integer partitioning leaves some output bands empty once the framebuffer has fewer
      // rows than --bands (a resize, or a tall band count): Vulkan forbids a zero render
      // area (VUID-VkRenderPassBeginInfo-None-08997), so such a band records no pass and
      // no draw. Everything else in the iteration — the upload wait, the first-use clear,
      // the timestamps, the gate choreography and the submit — still happens so the frame
      // stays correctly ordered and presented.
      const uint32_t window_y0 = band_offset(band, bands, extent.height);
      const uint32_t window_height = band_offset(band + 1, bands, extent.height) - window_y0;
      if (window_height != 0) {
        const holoviz::Rect2D band_rect{
            0, static_cast<int32_t>(window_y0), extent.width, window_height};
        const shader::PushConstants push_constants{
            .uv_min_y = (static_cast<float>(y0) + 0.5F) / static_cast<float>(kFrameHeight),
            .uv_max_y =
                (static_cast<float>(y0 + band_height) - 0.5F) / static_cast<float>(kFrameHeight),
        };
        CHECK_ERROR(command_list.begin_render(band_pass, band_rect));
        CHECK_ERROR(command_list.set_viewport(full_area));
        CHECK_ERROR(command_list.set_scissor(band_rect));
        CHECK_ERROR(command_list.bind_pipeline(pipeline));
        CHECK_ERROR(command_list.bind_descriptor(descriptor));
        CHECK_ERROR(command_list.set_push_data(
            pipeline,
            std::span{reinterpret_cast<const std::byte*>(&push_constants),
                      sizeof(push_constants)}));
        CHECK_ERROR(command_list.draw(3));
        CHECK_ERROR(command_list.end_render());
      }
      if (last_band) {
        latency.record_end(command_list, image_index);
      }
      CHECK_ERROR(command_list.end());

      holoviz::SubmitDesc submit_desc{
          .wait_submit_sync_tokens = {&previous_band_token, 1},
          .wait_external_sync_tokens = {&external_sync_token, 1},
          .command_lists = {&command_list, 1},
      };
      // The 4.x flag choreography: the first band consumes the acquire gate, the last
      // band arms the present gate, the bands in between carry neither (each gate is
      // consumed exactly once; the later bands are ordered behind the first by the
      // device timeline, so gating only the first is safe).
      if (use_gates && band == 0) {
        submit_desc.acquire_gates = {&acquired->acquire_gate, 1};
      }
      if (use_gates && band == bands - 1) {
        submit_desc.present_gates = {&acquired->present_gate, 1};
      }
      previous_band_token = UNWRAP(device.submit(submit_desc));
    }
    last_frame_token = previous_band_token;

    if (acquired) {
      if (shared_mode) {
        // A shared-mode present is a refresh request showing whatever is in the image at
        // that moment and (after the first frame) waits no semaphore — wait for render
        // completion first so the refresh shows the finished frame.
        CHECK_ERROR(device.wait_on_submit(last_frame_token, std::chrono::seconds(1)));
      }
      const uint64_t present_id = latency.enabled ? next_present_id : 0;
      holoviz::PresentConsumed consumed{};
      auto present_result =
          swapchain.present(std::move(*acquired), last_frame_token, present_id, &consumed);
      if (consumed.id) {
        ++next_present_id;
      }
      if (!present_result) {
        const holoviz::ErrorCode present_code = present_result.error().code;
        if (present_code == holoviz::ErrorCode::kImageCountChanged) {
          // The recreation that consumed this frame also changed the image count: the token is
          // spent, so rebuild for the new count and continue like a dropped frame below.
          rebuild_per_image_resources();
        }
        if (present_code == holoviz::ErrorCode::kFrameDropped ||
            present_code == holoviz::ErrorCode::kImageCountChanged) {
          // A resize consumed this frame before display; the swapchain already recreated
          // itself — just render the next frame at the new size (not a delivered frame, so
          // restart the interval measurement).
          latency.on_dropped();
          last_frame_end.reset();
          ++frame;
          continue;
        }
        throw std::runtime_error(
            "Error in `swapchain.present()` (code " +
            std::to_string(static_cast<uint32_t>(present_result.error().code)) +
            "): " + present_result.error().message);
      }
      presentation_started = true;
      if (present_id != 0) {
        latency.on_presented(image_index, present_id);
        if (latency.verify) {
          if (swapchain.wait_for_present(present_id, std::chrono::seconds(2))) {
            latency.records[present_id].present_wait_returned = std::chrono::steady_clock::now();
          }
        }
      }
    }
    latency.drain(swapchain);

    if (pace_on_display_events) {
      CHECK_ERROR(device.wait_for_display_event(
          *display, holoviz::DisplayEventType::kFirstPixelOut, std::chrono::milliseconds(100)));
    } else if (pace_on_host) {
      // Sleep to the next deadline, then advance it by one period; a late frame resets the
      // deadline to now rather than accumulating debt (no catch-up bursts).
      std::this_thread::sleep_until(next_present_deadline);
      next_present_deadline =
          std::max(next_present_deadline, std::chrono::steady_clock::now()) + host_frame_period;
    }
    if (options.fps_cap > 0.0) {
      // Hold the present rate below the refresh rate so the first-pixel-out period shows
      // whether the display follows the presents (VRR) or snaps to its own refresh.
      std::this_thread::sleep_for(std::chrono::duration<double>(1.0 / options.fps_cap));
    }

    const auto now = std::chrono::steady_clock::now();
    if (last_frame_end) {
      const double frame_ms =
          std::chrono::duration<double, std::milli>(now - *last_frame_end).count();
      window_stats.add(frame_ms);
      if (warmed_up) {
        total_stats.add(frame_ms);
      }
    }
    last_frame_end = now;
    if (now - window_start >= std::chrono::seconds(1)) {
      window_stats.print_fps(std::cout, warmed_up ? "  " : "  (warm-up) ");
      latency.recalibrate(device);
      latency.end_window(&std::cout, "    ");
      if (latency.enabled) {
        const auto mode = swapchain.refresh_mode().value_or(holoviz::RefreshMode::kUnknown);
        if (mode != reported_refresh_mode) {
          reported_refresh_mode = mode;
          std::cout << "    refresh mode (driver) now: " << describe_refresh_mode(mode) << "\n";
        }
      }
      window_stats = {};
      window_start = now;
      warmed_up = true;
      latency.warmed_up = true;
    }
    ++frame;
  }

  CHECK_ERROR(device.wait_idle());
  if (latency.enabled) {
    // Recover the last lap's stamps (the pool is idle after wait_idle) and the last few
    // timing samples before tearing down.
    if (capabilities.present_completion && next_present_id > 1) {
      (void)swapchain.wait_for_present(next_present_id - 1, std::chrono::milliseconds(500));
    }
    for (uint32_t image_index = 0; image_index < image_count; ++image_index) {
      latency.collect_slot(image_index);
    }
    latency.drain(swapchain);
    latency.window.counters.no_sample += static_cast<uint32_t>(latency.records.size());
  }
  CHECK_CUDA_ERROR(cuStreamSynchronize(cuda_stream));
  CHECK_CUDA_ERROR(cuStreamDestroy(cuda_stream));
  CHECK_CUDA_ERROR(cuMemFree(band_buffer));

  if (options.have_display) {
    // The swapchain must go before the display surface it was created on.
    CHECK_ERROR(swapchain.destroy(std::chrono::seconds(5)));
    CHECK_ERROR(surface.destroy());
  }

  std::cout << "Rendered " << frame << " frames of " << bands << " bands.\n";
  if (warmed_up) {
    total_stats.print_fps(std::cout, "Overall (after warm-up): ");
    if (latency.enabled) {
      latency.end_window(nullptr, "");  // fold the unfinished window into the totals
      std::cout << "Overall latency (after warm-up):\n";
      latency.print_stats(std::cout, latency.total, "  ");
    }
  } else {
    window_stats.print_fps(std::cout, "Overall (run shorter than the warm-up window): ");
    if (latency.enabled) {
      std::cout << "Latency (run shorter than the warm-up window):\n";
      latency.print_stats(std::cout, latency.window, "  ");
    }
  }
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }
}

// NOLINTEND(cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
