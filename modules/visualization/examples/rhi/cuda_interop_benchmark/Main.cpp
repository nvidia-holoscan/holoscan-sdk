/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// example code
//
// Reference benchmark for the RHI CUDA interop path. Every frame the CUDA
// side Gaussian-blurs the source image with FFT convolution (forward 2D FFT,
// Gaussian transfer function applied to the spectrum, inverse FFT) into a
// scratch buffer — all data stays in VRAM — and the blurred frame is copied
// device-to-device into an exported RHI image (the equivalent of the 4.x
// ImageCudaDevice() path), then sampled onto a full-screen quad and
// presented. Frames per second are measured over a fixed wall-clock window
// and printed on exit. The blur is implemented with cuFFT (batched real 2D
// transforms; see blur.hpp).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "holoviz/logger.hpp"
#include "holoviz/rhi/adapter.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <holoviz/rhi.hpp>

#include <cuda.h>

#include "../example_utils.hpp"
#include "blur.hpp"
#include "cuda_event_ring.hpp"

#define RHI_ERR(expr)                                               \
  {                                                                 \
    auto result = (expr);                                           \
    if (!result) {                                                  \
      throw std::runtime_error("Error: " + result.error().message); \
    }                                                               \
  }

#define CU_ERR(expr)                                                     \
  {                                                                      \
    CUresult result = (expr);                                            \
    if (result != CUDA_SUCCESS) {                                        \
      throw std::runtime_error("CUDA Error: " + std::to_string(result)); \
    }                                                                    \
  }

namespace rhi = holoviz::rhi;

namespace {

struct Context {
  rhi::Window window;
  rhi::Surface surface;
  rhi::Device device;
};

constexpr uint32_t kWarmupFrames = 300;
constexpr std::chrono::milliseconds kMeasureDuration{5000};

/// Standard deviation in pixels of the Gaussian the CUDA side applies to the source image
constexpr float kGaussianSigma = 25.0F;

/// Load the benchmark source image, forced to 4-channel RGBA. The blurs process it at full
/// size: both implementations run on the fixed FFT grid (see blur.hpp) and pad the source
/// up to that grid by edge replication.
std::vector<uint8_t> load_image(uint32_t& width, uint32_t& height, const char* res_path) {
  const std::string image_path = std::string(res_path) + "/nv_logo.png";

  int stb_width = 0, stb_height = 0, channels = 0;
  unsigned char* data =
      stbi_load(image_path.c_str(), &stb_width, &stb_height, &channels, STBI_rgb_alpha);
  if (!data) {
    throw std::runtime_error("Failed to load image " + image_path + ": " +
                             std::string(stbi_failure_reason()));
  }

  width = static_cast<uint32_t>(stb_width);
  height = static_cast<uint32_t>(stb_height);

  std::vector<uint8_t> pixels(data, data + static_cast<size_t>(width) * height * 4);

  stbi_image_free(data);

  return pixels;
}

/// Copy the image pixels into linear CUDA device memory; this is the
/// benchmark's per-frame copy source, matching the 4.x demo's DEVICE mode.
CUdeviceptr make_source_device_memory(const std::vector<uint8_t>& pixels, uint32_t width,
                                      uint32_t height, size_t& pitch) {
  CUdeviceptr device_ptr{};
  CU_ERR(cuMemAllocPitch(&device_ptr, &pitch, static_cast<size_t>(width) * 4, height, 4));

  CUDA_MEMCPY2D copy_desc{};
  copy_desc.srcMemoryType = CU_MEMORYTYPE_HOST;
  copy_desc.srcHost = pixels.data();
  copy_desc.srcPitch = static_cast<size_t>(width) * 4;
  copy_desc.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  copy_desc.dstDevice = device_ptr;
  copy_desc.dstPitch = pitch;
  copy_desc.WidthInBytes = static_cast<size_t>(width) * 4;
  copy_desc.Height = height;
  CU_ERR(cuMemcpy2D(&copy_desc));

  return device_ptr;
}

rhi::Buffer make_vertex_buffer(rhi::Device& device) {
  // Full-screen quad, interleaved position.xy + uv.xy per vertex. UV (0,0)
  // maps to NDC (-1,-1) (top-left in Vulkan), keeping the image upright.
  static const float vertices[16] = {
      -1.0f,
      -1.0f,
      0.0f,
      0.0f,  // top left
      1.0f,
      -1.0f,
      1.0f,
      0.0f,  // top right
      1.0f,
      1.0f,
      1.0f,
      1.0f,  // bottom right
      -1.0f,
      1.0f,
      0.0f,
      1.0f,  // bottom left
  };

  rhi::BufferDesc buffer_desc{
      .size_bytes = sizeof(vertices),
  };
  auto buffer = device.create_buffer(buffer_desc).value();
  auto upload_span = std::span{reinterpret_cast<const std::byte*>(vertices), sizeof(vertices)};
  RHI_ERR(device.upload_to_buffer(buffer.whole().value(), upload_span));
  return buffer;
}

rhi::Buffer make_index_buffer(rhi::Device& device) {
  static const uint16_t indices[6] = {0, 1, 2, 0, 2, 3};
  rhi::BufferDesc buffer_desc{
      .size_bytes = sizeof(indices),
  };
  auto buffer = device.create_buffer(buffer_desc).value();
  auto upload_span = std::span{reinterpret_cast<const std::byte*>(indices), sizeof(indices)};
  RHI_ERR(device.upload_to_buffer(buffer.whole().value(), upload_span));
  return buffer;
}

/// Create the exported image that CUDA copies into each frame.
rhi::Image make_interop_image(const rhi::Device& device, uint32_t width, uint32_t height) {
  rhi::ImageDesc image_desc{
      .extent = {width, height},
      .format = rhi::Format::kR8G8B8A8Unorm,
      .usage = rhi::ImageUsage::kSampled,
      .share = rhi::ShareMode::kExported,
      .debug_name = "benchmark source image",
  };
  return device.create_image(image_desc).value();
}

rhi::Pipeline make_pipeline(const rhi::Device& device, const char* res_path) {
  const std::string shader_path = std::string(res_path) + "/shader.slang";

  std::ifstream shader_file(shader_path);
  if (!shader_file.is_open()) {
    throw std::runtime_error("Failed to open shader file: " + shader_path);
  }
  std::string shader_source((std::istreambuf_iterator<char>(shader_file)),
                            std::istreambuf_iterator<char>());

  auto shader_blob =
      rhi::compile_slang_shader(
          device.backend().value(), "vertex_main", "fragment_main", shader_source.c_str())
          .value();

  rhi::PipelineDesc pipeline_desc{
      .shader_blob = shader_blob,
      .topology = rhi::PrimitiveTopology::kTriangleList,
      .raster =
          rhi::RasterState{
              .cull = rhi::CullMode::kNone,
          },
      .depth =
          rhi::DepthState{
              .test = false,
              .write = false,
              .compare = rhi::CompareOp::kAlways,
          },
      .blend =
          rhi::BlendState{
              .enable = false,
          },
  };

  return device.create_pipeline(pipeline_desc).value();
}

/// Create the swapchain with the fastest present mode the surface supports;
/// vsync-locked modes would cap the benchmark at the display refresh rate.
rhi::Swapchain make_swapchain(const rhi::Device& device, const rhi::Surface& surface) {
  static const std::pair<rhi::PresentMode, const char*> candidates[] = {
      {rhi::PresentMode::kFIFOLatestReady, "fifo_latest_ready"},
      // {rhi::PresentMode::kImmediate, "immediate"},
      {rhi::PresentMode::kMailbox, "mailbox"},
      {rhi::PresentMode::kFIFO, "fifo"},
  };

  for (const auto& [mode, name] : candidates) {
    rhi::SwapchainDesc swapchain_desc{
        .surface = surface,
        .present_mode = mode,
        .image_count = 8,  // more than enough to keep the CUDA stream busy
    };
    auto swapchain = device.create_swapchain(swapchain_desc);
    if (swapchain) {
      std::cout << "Present mode: " << name << "\n";
      return std::move(swapchain).value();
    }
  }
  throw std::runtime_error("Failed to create a swapchain with any present mode");
}

}  // namespace

int run(int argc, char** argv) {
  const auto args = holoviz::examples::parse_resource_path_args(
      argc, argv, "holoviz_rhi_cuda_interop_benchmark_assets");
  if (args.exit_code) {
    return *args.exit_code;
  }
  const std::string& res_path = args.res_path;

  std::cout << argv[0] << " - Holoviz RHI CUDA Interop Benchmark\n";
  std::cout << "Resource path: " << res_path << "\n";

  holoviz::set_log_level(holoviz::LogLevel::kWarn);

  Context context{};

  // context.device.cuda_device();

  uint32_t image_width = 0, image_height = 0;
  const std::vector<uint8_t> pixels = load_image(image_width, image_height, res_path.c_str());
  const size_t frame_bytes = static_cast<size_t>(image_width) * image_height * 4;

  std::cout << "Source image: " << image_width << "x" << image_height << " R8G8B8A8_UNORM ("
            << frame_bytes / (1024.0 * 1024.0) << " MiB per frame)\n";
  std::cout << "FFT Gaussian blur: sigma " << kGaussianSigma << " pixels, " << kBlurGridWidth << "x"
            << kBlurGridHeight << " FFT grid\n";

  // Size the window to the image aspect ratio, like the 4.x demo
  const uint32_t display_width = 1024;
  const uint32_t display_height = static_cast<uint32_t>(static_cast<float>(image_height) /
                                                        static_cast<float>(image_width) * 1024.0f);

  rhi::WindowDesc window_desc{
      .title = "Holoviz Benchmark Reference",
      .size = {display_width, display_height},
  };
  context.window = rhi::create_window(window_desc).value();
  context.surface = context.window.create_surface().value();

  rhi::DeviceDesc device_desc{
      .backend = rhi::Backend::kAuto,
      .policy = rhi::DevicePolicy::kDynamic,
      .cuda_context_mode = rhi::CudaContextMode::kCigPreferred,  // Works on Windows, falls back to
                                                                 // kExternal on Linux
  };

  context.device = rhi::create_device_for_surface(device_desc, context.surface).value();

  auto& device = context.device;

  std::cout << "Using device: " << device.adapter().value().name.data() << "\n";
  std::cout << "  CigMode: " << device.cuda_is_cig().value() << "\n";

  auto swapchain = make_swapchain(device, context.surface);
  const uint32_t swapchain_image_count = swapchain.image_count().value();
  const rhi::Extent2D swapchain_extent = swapchain.extent().value();

  // Can't upload image data without a CUDA stream, which the device owns here, so we do this after
  // the device is created.
  size_t source_pitch = 0;
  CUdeviceptr source_device_mem =
      make_source_device_memory(pixels, image_width, image_height, source_pitch);

  // The per-frame CUDA workload: FFT-convolution Gaussian blur of the source image into its
  // own device-memory scratch buffers (the source is only ever read).
  CufftGaussianBlur blur(image_width, image_height, kGaussianSigma);

  auto vertex_buffer = make_vertex_buffer(device);
  auto index_buffer = make_index_buffer(device);
  auto pipeline = make_pipeline(device, res_path.c_str());

  rhi::RenderPassDesc render_pass_desc{.render_targets = {{
                                           .format = swapchain.format().value(),
                                       }}};
  auto render_pass = device.create_render_pass(render_pass_desc).value();

  // Two timestamp slots per swapchain image bracket that image's command
  // list, measuring the Vulkan render span. The submit waits on the CUDA
  // timeline before the list executes, so the CUDA interop copy is excluded.
  auto query_pool = device
                        .create_query_pool({.timestamp_count = 2 * swapchain_image_count,
                                            .debug_name = "frame gpu timer"})
                        .value();

  std::vector<rhi::Image> interop_images;
  std::vector<CUarray> interop_arrays;
  std::vector<rhi::Descriptor> descriptors;

  std::vector<rhi::CommandList> command_lists;
  for (uint32_t i = 0; i < swapchain_image_count; ++i) {
    command_lists.push_back(device.create_command_list({}).value());
    interop_images.push_back(make_interop_image(device, image_width, image_height));
    interop_arrays.push_back(interop_images[i].cuda_mipmapped_array_get_level(0).value());
    descriptors.push_back(pipeline.create_descriptor().value());

    RHI_ERR(descriptors[i].bind_image(interop_images[i], "image_texture"));
    RHI_ERR(descriptors[i].bind_sampler(device, {}, "image_sampler"));

    auto current_image = swapchain.image(i).value();
    RHI_ERR(render_pass.set_render_image(current_image, 0, {0.0F, 0.0F, 0.0F, 1.0F}));

    rhi::Rect2D render_area{0, 0, swapchain_extent.width, swapchain_extent.height};

    auto command_list = &command_lists[i];

    RHI_ERR(command_list->begin());
    RHI_ERR(command_list->reset_queries(query_pool, 2 * i, 2));
    RHI_ERR(command_list->write_timestamp(query_pool, 2 * i, rhi::TimestampLocation::kStart));
    RHI_ERR(command_list->begin_render(render_pass, render_area));
    RHI_ERR(command_list->set_viewport(render_area));
    RHI_ERR(command_list->set_scissor(render_area));
    RHI_ERR(command_list->bind_pipeline(pipeline));
    RHI_ERR(command_list->bind_vertex_buffer(0, vertex_buffer.whole().value()));
    RHI_ERR(command_list->bind_index_buffer(index_buffer.whole().value(), rhi::IndexType::kUint16));
    RHI_ERR(command_list->bind_descriptor(descriptors[i]));
    RHI_ERR(command_list->draw_indexed(6, 1, 0, 0, 0));
    RHI_ERR(command_list->end_render());
    RHI_ERR(command_list->write_timestamp(query_pool, 2 * i + 1, rhi::TimestampLocation::kEnd));
    RHI_ERR(command_list->resolve_queries(query_pool, 2 * i, 2));
    RHI_ERR(command_list->end());
  }

  CUstream cuda_stream{};
  CU_ERR(cuStreamCreate(&cuda_stream, CU_STREAM_NON_BLOCKING));

  auto external_semaphore = device.create_external_semaphore(cuda_stream).value();

  std::vector<rhi::SubmitSyncToken> sync_tokens(swapchain_image_count, rhi::SubmitSyncToken{});
  std::vector<double> gpu_frame_ms;
  bool collect_gpu_samples = false;

  // CUDA event pairs bracketing each frame's Vulkan span as the CUDA stream
  // sees it: "start" is the last stream op before the semaphore signal hands
  // the frame to Vulkan, "end" is the first stream op after the semaphore
  // wait returns control to CUDA on a later frame. Ring depth is one past
  // the swapchain depth so a slot's previous pair has always been recorded
  // before it is recycled; the rings enforce collect-before-record so a
  // measured frame never host-syncs on its own stream work.
  const uint32_t cuda_timer_depth = swapchain_image_count + 1;

  holoviz::examples::CudaEventSpanRing frame_ring(cuda_timer_depth);
  holoviz::examples::CudaEventSpanRing vulkan_span_ring(cuda_timer_depth);
  holoviz::examples::CudaEventSpanRing cuda_blur_ring(cuda_timer_depth);
  holoviz::examples::CudaEventSpanRing cuda_memcpy_ring(cuda_timer_depth);
  uint64_t frame_index = 0;
  std::vector<double> frame_ms;
  std::vector<double> cuda_span_ms;
  std::vector<double> cuda_blur_ms;
  std::vector<double> cuda_memcpy_ms;

  rhi::SubmitSyncToken sync_token{};

  const auto render_frame = [&](GaussianBlur& blur) {
    rhi::poll_window_events();

    auto acquired = swapchain.acquire(std::chrono::milliseconds(100)).value();

    auto& command_list = command_lists[acquired.image_index];
    auto& acquire_gate = acquired.acquire_gate;
    auto& present_gate = acquired.present_gate;

    // Collect this slot's GPU render span from its previous submit before the
    // resubmit below recycles the query slots. acquire() only orders the GPU
    // (via the acquire gate) — the host can run a full swapchain depth ahead
    // of the slot's last submit, so gate the read on the device timeline. The
    // wait is ~zero in steady state: that submit is image_count frames old.
    if (collect_gpu_samples && sync_tokens[acquired.image_index].timeline_value != 0) {
      RHI_ERR(device.wait_on_submit(sync_tokens[acquired.image_index],
                                    std::chrono::nanoseconds(std::chrono::milliseconds(100))));
      uint64_t span_ns[2] = {};
      RHI_ERR(query_pool.results_ns(2 * acquired.image_index, 2, span_ns));
      gpu_frame_ms.push_back(static_cast<double>(span_ns[1] - span_ns[0]) / 1.0e6);
    }

    // GPU-side wait: the copy must not start until all prior frames have
    // finished sampling the image
    RHI_ERR(external_semaphore.wait(sync_token));

    // CUDA-side timer "end": the semaphore wait above is where the CUDA
    // stream regains control from Vulkan, closing the span opened by the
    // previous frame's "start" event.
    if (frame_index > 0) {
      vulkan_span_ring.record_end(frame_index - 1, cuda_stream);
      frame_ring.record_end(frame_index - 1, cuda_stream);
    }

    // Collect the spans still held by this frame's ring slots — a full ring
    // lap old, so the sync is ~zero in steady state — before the slots are
    // re-recorded below (the rings throw if the order is ever inverted).
    if (collect_gpu_samples) {
      if (const auto frame_ms_val = frame_ring.collect(frame_index)) {
        frame_ms.push_back(*frame_ms_val);
      }
      if (const auto span_ms = vulkan_span_ring.collect(frame_index)) {
        cuda_span_ms.push_back(*span_ms);
      }
      if (const auto blur_ms = cuda_blur_ring.collect(frame_index)) {
        cuda_blur_ms.push_back(*blur_ms);
      }
      if (const auto memcpy_ms = cuda_memcpy_ring.collect(frame_index)) {
        cuda_memcpy_ms.push_back(*memcpy_ms);
      }
    }

    frame_ring.record_start(frame_index, cuda_stream);

    // The benchmarked operation: FFT-convolution Gaussian blur of the source
    // image into the scratch output buffer (everything stays in VRAM) ...
    cuda_blur_ring.record_start(frame_index, cuda_stream);
    blur.blur_async(source_device_mem, source_pitch, cuda_stream);
    cuda_blur_ring.record_end(frame_index, cuda_stream);

    // ... followed by a device-to-device copy of the blurred frame into the
    // exported image
    CUDA_MEMCPY2D copy_desc{};
    copy_desc.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy_desc.srcDevice = blur.output();
    copy_desc.srcPitch = blur.output_pitch();
    copy_desc.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy_desc.dstArray = interop_arrays[acquired.image_index];
    copy_desc.WidthInBytes = static_cast<size_t>(image_width) * 4;
    copy_desc.Height = image_height;
    cuda_memcpy_ring.record_start(frame_index, cuda_stream);
    CU_ERR(cuMemcpy2DAsync(&copy_desc, cuda_stream));
    cuda_memcpy_ring.record_end(frame_index, cuda_stream);

    // CUDA-side timer "start": last stream op before the semaphore signal
    // hands the frame off to Vulkan
    vulkan_span_ring.record_start(frame_index, cuda_stream);
    ++frame_index;

    auto external_sync_token = external_semaphore.signal().value();

    rhi::SubmitDesc submit_desc{
        .wait_submit_sync_tokens = {&sync_tokens[acquired.image_index], 1},
        .wait_external_sync_tokens = {&external_sync_token, 1},
        .command_lists = {&command_list, 1},
        .acquire_gates = {&acquire_gate, 1},
        .present_gates = {&present_gate, 1},
    };

    sync_tokens[acquired.image_index] = device.submit(submit_desc).value();
    sync_token = sync_tokens[acquired.image_index];

    RHI_ERR(swapchain.present(std::move(acquired), sync_token));
  };

  try {
    std::cout << "\n=== " << blur.label() << " ===\n";

    // Warm up: absorbs first-use pipeline compilation, fills the swapchain's
    // frames-in-flight slots before timing starts, and on the second pass
    // drains all frames of the previous implementation
    collect_gpu_samples = false;
    bool should_close = false;
    for (uint32_t i = 0; i < kWarmupFrames && !should_close; ++i) {
      auto close_query = context.window.close_requested();
      if (!close_query) {
        std::cerr << "Error querying window close request status" << std::endl;
        return -1;
      }
      should_close = close_query.value();
      render_frame(blur);
    }

    gpu_frame_ms.clear();
    frame_ms.clear();
    cuda_span_ms.clear();
    cuda_blur_ms.clear();
    cuda_memcpy_ms.clear();
    // Forget the warmup frames' recorded spans so the first measured collects
    // return nothing instead of warmup samples; the frame counter stays
    // monotonic (a reset would desync the rings' slot-ownership checks).
    frame_ring.reset();
    vulkan_span_ring.reset();
    cuda_blur_ring.reset();
    cuda_memcpy_ring.reset();
    collect_gpu_samples = true;

    const auto start = std::chrono::steady_clock::now();
    uint32_t iterations = 0;
    std::chrono::milliseconds elapsed{};
    do {
      render_frame(blur);
      ++iterations;
      elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start);
      auto close_query = context.window.close_requested();
      if (!close_query) {
        std::cerr << "Error querying window close request status" << std::endl;
        return -1;
      }
      should_close = close_query.value();
    } while (elapsed < kMeasureDuration && !should_close);

    // Flush the last ring lap of measured frames: their spans are recorded but
    // were never a full lap old inside the loop. collect(frame_index + i)
    // visits each slot exactly once (collect host-syncs the end event itself);
    // still-incomplete pairs — the final frame's Vulkan span, whose end never
    // got recorded — come back empty and are correctly excluded.
    for (uint32_t i = 0; i < cuda_timer_depth; ++i) {
      if (const auto frame_ms_val = frame_ring.collect(frame_index + i)) {
        frame_ms.push_back(*frame_ms_val);
      }
      if (const auto span_ms = vulkan_span_ring.collect(frame_index + i)) {
        cuda_span_ms.push_back(*span_ms);
      }
      if (const auto blur_ms = cuda_blur_ring.collect(frame_index + i)) {
        cuda_blur_ms.push_back(*blur_ms);
      }
      if (const auto memcpy_ms = cuda_memcpy_ring.collect(frame_index + i)) {
        cuda_memcpy_ms.push_back(*memcpy_ms);
      }
    }

    const float seconds = static_cast<float>(elapsed.count()) / 1000.0f;
    const float fps = static_cast<float>(iterations) / seconds;
    const double throughput_gb =
        (static_cast<double>(iterations) * static_cast<double>(frame_bytes)) / seconds / 1.0e9;

    struct Stats {
      double mean = 0.0;
      double std_dev = 0.0;
      double max = 0.0;
      double p95 = 0.0;
      double p99 = 0.0;
    };
    const auto compute_stats = [](const std::vector<double>& samples) {
      Stats s;
      for (const double sample : samples) {
        s.mean += sample;
        s.max = std::max(s.max, sample);
      }
      if (!samples.empty()) {
        s.mean /= static_cast<double>(samples.size());
      }
      if (samples.size() > 1) {
        double variance = 0.0;
        for (const double sample : samples) {
          variance += (sample - s.mean) * (sample - s.mean);
        }
        s.std_dev = std::sqrt(variance / static_cast<double>(samples.size() - 1));
      }
      if (!samples.empty()) {
        std::vector<double> sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        const auto percentile = [&sorted](double p) {
          const size_t rank =
              static_cast<size_t>(std::ceil(p * static_cast<double>(sorted.size())));
          return sorted[std::min(rank == 0 ? 0 : rank - 1, sorted.size() - 1)];
        };
        s.p95 = percentile(0.95);
        s.p99 = percentile(0.99);
      }
      return s;
    };

    const Stats gpu_stats = compute_stats(gpu_frame_ms);
    const Stats frame_stats = compute_stats(frame_ms);
    const Stats cuda_span_stats = compute_stats(cuda_span_ms);
    const Stats cuda_blur_stats = compute_stats(cuda_blur_ms);
    const Stats cuda_memcpy_stats = compute_stats(cuda_memcpy_ms);

    std::cout << "DEVICE R8G8B8A8_UNORM  " << fps << " fps  " << 1000.0 / fps << " ms/frame\n";
    std::cout << "Delivered image throughput: " << throughput_gb << " GB/s\n\n";

    const auto fmt_ms = [](double ms) -> std::string {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(3) << ms;
      return oss.str();
    };

    struct Col {
      std::string label;
      std::string mean_str;
      std::string std_str;
      std::string p95_str;
      std::string p99_str;
      std::string max_str;
    };
    const auto make_col = [&](const std::string& label, const Stats& s) {
      return Col{
          label, fmt_ms(s.mean), fmt_ms(s.std_dev), fmt_ms(s.p95), fmt_ms(s.p99), fmt_ms(s.max)};
    };
    const std::vector<Col> cols = {
        {"Frame Time (1000 / FPS)", fmt_ms(1000.0 / fps), "-", "-", "-", "-"},
        make_col("Frame Time", frame_stats),
        make_col("Blur", cuda_blur_stats),
        make_col("Memcpy", cuda_memcpy_stats),
        make_col("CUDA->Vulkan", cuda_span_stats),
        make_col("GPU Render", gpu_stats),
    };

    std::vector<size_t> col_widths;
    col_widths.reserve(cols.size());
    for (const auto& c : cols) {
      size_t w = c.label.size();
      w = std::max(w, c.mean_str.size());
      w = std::max(w, c.std_str.size());
      w = std::max(w, c.p95_str.size());
      w = std::max(w, c.p99_str.size());
      w = std::max(w, c.max_str.size());
      col_widths.push_back(w);
    }

    constexpr size_t kRowLabelWidth = 8;  // "Mean(ms)" / "Std(ms)"

    const auto print_hline = [&]() {
      std::cout << "+" << std::string(kRowLabelWidth + 2, '-');
      for (const size_t w : col_widths) {
        std::cout << "+" << std::string(w + 2, '-');
      }
      std::cout << "+\n";
    };

    const auto print_row = [&](const std::string& row_label,
                               const std::vector<std::string>& cells) {
      std::cout << "| " << std::left << std::setw(static_cast<int>(kRowLabelWidth)) << row_label
                << " ";
      for (size_t i = 0; i < cells.size(); ++i) {
        std::cout << "| " << std::left << std::setw(static_cast<int>(col_widths[i])) << cells[i]
                  << " ";
      }
      std::cout << "|\n";
    };

    std::vector<std::string> header_cells, mean_cells, std_cells, p95_cells, p99_cells, max_cells;
    for (const auto& c : cols) {
      header_cells.push_back(c.label);
      mean_cells.push_back(c.mean_str);
      std_cells.push_back(c.std_str);
      p95_cells.push_back(c.p95_str);
      p99_cells.push_back(c.p99_str);
      max_cells.push_back(c.max_str);
    }

    print_hline();
    print_row("", header_cells);
    print_hline();
    print_row("Std(ms)", std_cells);
    print_row("Mean(ms)", mean_cells);
    print_row("P95(ms)", p95_cells);
    print_row("P99(ms)", p99_cells);
    print_row("Max(ms)", max_cells);
    print_hline();

    std::cout << "(" << cuda_blur_ms.size() << " CUDA frames, " << gpu_frame_ms.size()
              << " GPU frames)\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }

  RHI_ERR(device.wait_idle());
  CU_ERR(cuStreamSynchronize(cuda_stream));
  // The event rings destroy their events when they go out of scope below.
  CU_ERR(cuMemFree(source_device_mem));

  return EXIT_SUCCESS;
}

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
