/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Demonstrates HoloViz-to-CUDA interop: render to a Vulkan image, then access the
// result as a CUDA array.  No display is required; the final frame is written
// to render_to_cuda_output.bmp so it can be inspected offline.

#include <cuda.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoviz/holoviz.hpp>

#define CHECK_CUDA_ERROR(expr)                                           \
  {                                                                      \
    const CUresult result = (expr);                                      \
    if (result != CUDA_SUCCESS) {                                        \
      throw std::runtime_error("CUDA Error: " + std::to_string(result)); \
    }                                                                    \
  }

namespace {

constexpr uint32_t kImageWidth = 640;
constexpr uint32_t kImageHeight = 480;
constexpr uint32_t kFrameCount = 60;
constexpr const char* kOutputPath = "render_to_cuda_output.bmp";

struct Context {
  holoviz::Adapter adapter_;
  holoviz::Device device_;
  holoviz::Image render_target_;
  holoviz::RenderPass render_pass_;
  holoviz::CommandList command_list_;
  holoviz::ExternalSemaphore external_semaphore_;
  holoviz::SubmitSyncToken sync_token_;
  CUstream cuda_stream_{};
  CUarray mapped_array_{};

  holoviz::expected<void> init() {
    VIZ_UNWRAP(std::vector<holoviz::Adapter>, adapters, holoviz::enumerate_adapters());
    adapter_ = adapters[0];

    VIZ_UNWRAP_NODECL(holoviz::Device, device_, adapter_.create_device({}));

    const holoviz::ImageDesc image_desc{
        .extent = {kImageWidth, kImageHeight},
        .format = holoviz::Format::kR8G8B8A8Unorm,
        .usage = holoviz::ImageUsage::kRenderTarget,
        .share = holoviz::ShareMode::kExported,
    };
    VIZ_UNWRAP_NODECL(holoviz::Image, render_target_, device_.create_image(image_desc));

    const holoviz::RenderPassDesc render_pass_desc{
        .render_targets = {{.format = image_desc.format}}};
    VIZ_UNWRAP_NODECL(
        holoviz::RenderPass, render_pass_, device_.create_render_pass(render_pass_desc));

    CHECK_CUDA_ERROR(cuStreamCreate(&cuda_stream_, CU_STREAM_NON_BLOCKING));

    VIZ_UNWRAP_NODECL(holoviz::ExternalSemaphore,
                      external_semaphore_,
                      device_.create_external_semaphore(cuda_stream_));
    VIZ_UNWRAP_NODECL(holoviz::CommandList,
                      command_list_,
                      device_.create_command_list({.type = holoviz::CommandListType::kTransient}));
    VIZ_UNWRAP_NODECL(CUarray, mapped_array_, render_target_.cuda_mipmapped_array_get_level(0));

    return {};
  }

  holoviz::expected<void> render(void* frame_out_ptr, float angle) {
    auto external_sync_token = external_semaphore_.signal().value();

    VIZ_FORWARD_ERROR(render_pass_.set_render_image(
        render_target_,
        0,
        {std::sin(angle) / 2.0F + 0.5F, 0.0F, std::cos(angle) / 2.0F + 0.5F, 1.0F}));

    VIZ_FORWARD_ERROR(command_list_.reset(std::chrono::milliseconds(100)));
    VIZ_FORWARD_ERROR(command_list_.begin());
    VIZ_FORWARD_ERROR(command_list_.begin_render(render_pass_, {0, 0, kImageWidth, kImageHeight}));
    VIZ_FORWARD_ERROR(command_list_.end_render());
    VIZ_FORWARD_ERROR(command_list_.end());

    const holoviz::SubmitDesc submit_desc{
        .wait_submit_sync_tokens = {&sync_token_, 1},
        .wait_external_sync_tokens = {&external_sync_token, 1},
        .command_lists = {&command_list_, 1},
    };
    sync_token_ = device_.submit(submit_desc).value();

    VIZ_FORWARD_ERROR(external_semaphore_.wait(sync_token_));

    CUDA_MEMCPY2D copy_desc{};
    copy_desc.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy_desc.srcArray = mapped_array_;
    copy_desc.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy_desc.dstHost = frame_out_ptr;
    copy_desc.WidthInBytes = kImageWidth * sizeof(uint32_t);
    copy_desc.Height = kImageHeight;
    CHECK_CUDA_ERROR(cuMemcpy2DAsync(&copy_desc, cuda_stream_));
    CHECK_CUDA_ERROR(cuStreamSynchronize(cuda_stream_));

    return {};
  }

  ~Context() {
    auto result = device_.wait_idle();
    if (!result) {
      std::cerr << "Error waiting for device idle: " << result.error().message << "\n";
    }

    auto log_destroy_error = [](const char* what, const holoviz::expected<void>& r) {
      if (!r) {
        std::cerr << "Error destroying " << what << ": " << r.error().message << "\n";
      }
    };
    log_destroy_error("command list", command_list_.destroy());
    log_destroy_error("external semaphore", external_semaphore_.destroy());
    log_destroy_error("render pass", render_pass_.destroy());
    log_destroy_error("render target", render_target_.destroy());

    if (cuda_stream_ != nullptr) {
      cuStreamDestroy(cuda_stream_);
    }

    log_destroy_error("device", device_.destroy());
  }
};

void write_bmp(const char* path, const uint32_t* pixels, uint32_t width, uint32_t height) {
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error(std::string("cannot open ") + path);
  }

  const uint32_t row_size = (width * 3 + 3) & ~3U;  // rows are padded to 4 bytes
  const uint32_t pixel_data_size = row_size * height;
  const uint32_t header_size = 14 + 40;  // file header + BITMAPINFOHEADER
  const uint32_t file_size = header_size + pixel_data_size;

  auto put16 = [&f](uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
  auto put32 = [&f](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };

  // BITMAPFILEHEADER
  f.write("BM", 2);
  put32(file_size);
  put32(0);            // reserved
  put32(header_size);  // pixel data offset

  // BITMAPINFOHEADER
  put32(40);
  put32(width);
  put32(height);  // positive height: rows are stored bottom-up
  put16(1);       // planes
  put16(24);      // bits per pixel
  put32(0);       // BI_RGB, no compression
  put32(pixel_data_size);
  put32(0);  // x pixels per meter
  put32(0);  // y pixels per meter
  put32(0);  // colors used
  put32(0);  // important colors

  const std::array<char, 3> padding = {};
  for (uint32_t y = 0; y < height; ++y) {
    const uint32_t* row = pixels + size_t(height - 1 - y) * width;
    for (uint32_t x = 0; x < width; ++x) {
      const uint32_t px = row[x];
      const std::array<char, 3> bgr = {{
          static_cast<char>((px >> 16) & 0xFF),
          static_cast<char>((px >> 8) & 0xFF),
          static_cast<char>(px & 0xFF),
      }};
      f.write(bgr.data(), 3);
    }
    f.write(padding.data(), row_size - width * 3);
  }
}

}  // namespace

int main(int /*argc*/, char** /*argv*/) {
  try {
    Context context;
    auto init_result = context.init();
    if (!init_result) {
      std::cerr << "Failed to initialize context: " << init_result.error().message << "\n";
      return EXIT_FAILURE;
    }

    std::vector<uint32_t> image(size_t(kImageWidth) * kImageHeight);

    float angle = 0.0F;
    for (uint32_t frame = 0; frame < kFrameCount; ++frame) {
      angle -= 0.01F;
      auto render_result = context.render(image.data(), angle);
      if (!render_result) {
        std::cerr << "Failed to render frame " << frame << ": " << render_result.error().message
                  << "\n";
        return EXIT_FAILURE;
      }
    }

    write_bmp(kOutputPath, image.data(), kImageWidth, kImageHeight);
    std::cout << "Rendered " << kFrameCount << " frames. Output: " << kOutputPath << "\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
