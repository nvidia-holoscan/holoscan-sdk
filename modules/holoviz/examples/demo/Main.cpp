/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>
#include <getopt.h>
#include <imgui.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <holoviz/holoviz.hpp>

namespace viz = holoscan::viz;

enum class Source { HOST, DEVICE, ARRAY };
static const char* source_items[]{"Host", "Device", "Array"};

static const char* format_items[]{"R8_UINT", "R8G8B8_UNORM", "R8G8B8A8_UNORM"};
static viz::ImageFormat formats[]{
    viz::ImageFormat::R8_UINT, viz::ImageFormat::R8G8B8_UNORM, viz::ImageFormat::R8G8B8A8_UNORM};

// define the '<<' operators to get a nice output
#define CASE(VALUE)            \
  case VALUE:                  \
    os << std::string(#VALUE); \
    break;

std::ostream& operator<<(std::ostream& os, const Source& source) {
  switch (source) {
    CASE(Source::HOST)
    CASE(Source::DEVICE)
    CASE(Source::ARRAY)
    default:
      os.setstate(std::ios_base::failbit);
  }
  return os;
}

#undef CASE

// options
bool benchmark_mode = false;
bool headless_mode = false;

bool show_ui = true;

bool show_image_layer = true;
Source current_source = Source::DEVICE;
uint32_t current_format_index = 2;
float image_layer_opacity = 1.f;
int image_layer_priority = 0;

bool show_geometry_layer = true;
float geometry_layer_opacity = 1.f;
int geometry_layer_priority = 1;

bool show_geometry_3d_layer = true;
float geometry_3d_layer_opacity = 1.f;
int geometry_3d_layer_priority = 2;

uint32_t width = 1920;
uint32_t height = 1080;

// timing
std::chrono::steady_clock::time_point start;
std::chrono::milliseconds elapsed;
uint32_t iterations = 0;
float fps = 0.f;

// memory
std::unique_ptr<uint8_t> host_mem_r8;
std::unique_ptr<uint8_t> host_mem_r8g8b8;
std::unique_ptr<uint8_t> host_mem_r8g8b8a8;

std::vector<uint32_t> palette;

// cuda
CUcontext cuda_context = nullptr;
CUdeviceptr cu_device_mem_r8 = 0;
CUdeviceptr cu_device_mem_r8g8b8 = 0;
CUdeviceptr cu_device_mem_r8g8b8a8 = 0;

void tick() {
  if (start.time_since_epoch().count() == 0) {
    start = std::chrono::steady_clock::now();
    iterations = 0;
  }

  viz::Begin();
  if (show_ui) {
    // UI
    viz::BeginImGuiLayer();

    viz::LayerPriority(11);

    ImGui::Begin("Options", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    ImGui::Checkbox("Image layer", &show_image_layer);
    if (show_image_layer) {
      ImGui::Combo("Source",
                   reinterpret_cast<int*>(&current_source),
                   source_items,
                   IM_ARRAYSIZE(source_items));
      ImGui::Combo("Format",
                   reinterpret_cast<int*>(&current_format_index),
                   format_items,
                   IM_ARRAYSIZE(format_items));
      ImGui::SliderFloat("Opacity##image", &image_layer_opacity, 0.f, 1.f);
      ImGui::SliderInt("Priority##image", &image_layer_priority, -10, 10);

      // color picker for first item of LUT
      if (formats[current_format_index] == viz::ImageFormat::R8_UINT) {
        static int color_index = 0;
        ImGui::SliderInt("LUT index", &color_index, 0, palette.size() - 1);

        uint32_t& item = palette[color_index];
        float color[]{(item & 0xFF) / 255.f,
                      ((item >> 8) & 0xFF) / 255.f,
                      ((item >> 16) & 0xFF) / 255.f,
                      ((item >> 24) & 0xFF) / 255.f};
        ImGui::ColorEdit4("##color", color, ImGuiColorEditFlags_DefaultOptions_);
        item = static_cast<uint32_t>((color[0] * 255.f) + 0.5f) +
               (static_cast<uint32_t>((color[1] * 255.f) + 0.5f) << 8) +
               (static_cast<uint32_t>((color[2] * 255.f) + 0.5f) << 16) +
               (static_cast<uint32_t>((color[3] * 255.f) + 0.5f) << 24);
      }
    }

    ImGui::Separator();
    ImGui::Checkbox("Geometry layer", &show_geometry_layer);
    if (show_geometry_layer) {
      ImGui::SliderFloat("Opacity##geom", &geometry_layer_opacity, 0.f, 1.f);
      ImGui::SliderInt("Priority##geom", &geometry_layer_priority, -10, 10);
    }

    ImGui::Separator();
    ImGui::Checkbox("3D Geometry layer", &show_geometry_3d_layer);
    if (show_geometry_3d_layer) {
      ImGui::SliderFloat("Opacity##geom3d", &geometry_3d_layer_opacity, 0.f, 1.f);
      ImGui::SliderInt("Priority##geom3d", &geometry_3d_layer_priority, -10, 10);
    }

    ImGui::Text("%.1f frames/s", fps);

    ImGui::End();

    viz::EndLayer();
  }

  if (show_image_layer) {
    viz::BeginImageLayer();
    viz::LayerOpacity(image_layer_opacity);
    viz::LayerPriority(image_layer_priority);

    if ((formats[current_format_index] == viz::ImageFormat::R8G8B8_UNORM) ||
        (formats[current_format_index] == viz::ImageFormat::R8G8B8A8_UNORM)) {
      // Color image

      // host memory
      switch (current_source) {
        case Source::HOST: {
          const void* data = nullptr;
          switch (formats[current_format_index]) {
            case viz::ImageFormat::R8G8B8_UNORM:
              data = host_mem_r8g8b8.get();
              break;
            case viz::ImageFormat::R8G8B8A8_UNORM:
              data = host_mem_r8g8b8a8.get();
              break;
            default:
              throw std::runtime_error("Unhandled image format");
          }
          viz::ImageHost(width, height, formats[current_format_index], data);
          break;
        }
        case Source::DEVICE: {
          CUdeviceptr device_ptr = 0;
          switch (formats[current_format_index]) {
            case viz::ImageFormat::R8G8B8_UNORM:
              device_ptr = cu_device_mem_r8g8b8;
              break;
            case viz::ImageFormat::R8G8B8A8_UNORM:
              device_ptr = cu_device_mem_r8g8b8a8;
              break;
            default:
              throw std::runtime_error("Unhandled image format");
          }
          viz::ImageCudaDevice(width, height, formats[current_format_index], device_ptr);
          break;
        }
      }
    } else {
      // Image with LUT
      viz::LUT(palette.size(),
               viz::ImageFormat::R8G8B8A8_UNORM,
               palette.size() * sizeof(uint32_t),
               palette.data());

      if (current_source == Source::DEVICE) {
        viz::ImageCudaDevice(width, height, formats[current_format_index], cu_device_mem_r8);
      } else {
        viz::ImageHost(width, height, formats[current_format_index], host_mem_r8.get());
      }
    }

    viz::EndLayer();
  }

  if (show_geometry_layer) {
    viz::BeginGeometryLayer();
    viz::LayerOpacity(geometry_layer_opacity);
    viz::LayerPriority(geometry_layer_priority);

    const float text_size = 0.05f;

    viz::Color(1.f, 0.f, 0.f, 1.f);
    viz::Text(0.65f, 0.05f, text_size, "POINT_LIST");
    {
      const float data[]{0.9f, 0.1f, 0.95f, 0.05f};
      viz::PointSize(5.f);
      viz::Primitive(viz::PrimitiveTopology::POINT_LIST, 2, sizeof(data) / sizeof(data[0]), data);
    }

    viz::Color(0.f, 1.f, 0.f, 1.f);
    viz::Text(0.65f, 0.2f, text_size, "LINE_LIST");
    {
      const float data[]{0.9f, 0.25f, 0.95f, 0.2f, 0.92f, 0.27f, 0.93f, 0.23f};
      viz::LineWidth(2.f);
      viz::Primitive(viz::PrimitiveTopology::LINE_LIST, 2, sizeof(data) / sizeof(data[0]), data);
    }

    viz::Color(1.f, 1.f, 0.f, 1.f);
    viz::Text(0.65f, 0.35f, text_size, "LINE_STRIP");
    {
      const float data[]{0.9f, 0.35f, 0.95f, 0.3f, 0.97f, 0.37f, 0.93f, 0.35f};
      viz::LineWidth(1.f);
      viz::Primitive(viz::PrimitiveTopology::LINE_STRIP, 3, sizeof(data) / sizeof(data[0]), data);
    }

    viz::Color(0.f, 0.f, 1.f, 1.f);
    viz::Text(0.65f, 0.5f, text_size, "TRIANGLE_LIST");
    {
      const float data[]{
          0.9f, 0.45f, 0.92f, 0.45f, 0.91f, 0.5f, 0.95f, 0.45f, 0.95f, 0.55f, 0.975f, 0.50f};
      viz::Primitive(
          viz::PrimitiveTopology::TRIANGLE_LIST, 2, sizeof(data) / sizeof(data[0]), data);
    }

    viz::Color(1.f, 0.f, 1.f, 1.f);
    viz::Text(0.65f, 0.65f, text_size, "CROSS_LIST");
    {
      const float data[]{0.9f, 0.7f, 0.08f, 0.95f, 0.65f, 0.05f};
      viz::Primitive(viz::PrimitiveTopology::CROSS_LIST, 2, sizeof(data) / sizeof(data[0]), data);
    }

    viz::Color(0.f, 1.f, 1.f, 1.f);
    viz::Text(0.65f, 0.8f, text_size, "RECTANGLE_LIST");
    {
      const float data[]{0.9f, 0.75f, 0.98f, 0.85f, 0.95f, 0.8f, 0.97f, 0.83f};
      viz::Primitive(
          viz::PrimitiveTopology::RECTANGLE_LIST, 2, sizeof(data) / sizeof(data[0]), data);
    }

    viz::Color(1.f, 1.f, 1.f, 1.f);
    viz::Text(0.65f, 0.95f, text_size, "OVAL_LIST");
    {
      const float data[]{0.9f, 0.95f, 0.1f, 0.1f, 0.95f, 0.975f, 0.05f, 0.1f};
      viz::LineWidth(3.f);
      viz::Primitive(viz::PrimitiveTopology::OVAL_LIST, 2, sizeof(data) / sizeof(data[0]), data);
    }

    viz::EndLayer();
  }

  if (show_geometry_3d_layer) {
    viz::BeginGeometryLayer();
    viz::LayerOpacity(geometry_3d_layer_opacity);
    viz::LayerPriority(geometry_3d_layer_priority);

    {
      const float x_min = -0.25f;
      const float x_max = 0.25f;
      const float y_min = -0.25f;
      const float y_max = 0.25f;
      const float z_min = -0.25f;
      const float z_max = 0.25f;
      const float data[]{
          x_min, y_min, z_min, x_max, y_min, z_min, x_min, y_max, z_min, x_max, y_max, z_min,
          x_min, y_min, z_max, x_max, y_min, z_max, x_min, y_max, z_max, x_max, y_max, z_max,
          x_min, y_max, z_min, x_min, y_min, z_min, x_max, y_max, z_min, x_max, y_min, z_min,
          x_min, y_max, z_max, x_min, y_min, z_max, x_max, y_max, z_max, x_max, y_min, z_max,
          x_min, y_max, z_min, x_min, y_max, z_max, x_min, y_min, z_min, x_min, y_min, z_max,
          x_max, y_max, z_min, x_max, y_max, z_max, x_max, y_min, z_min, x_max, y_min, z_max};
      viz::Color(0.75f, 0.f, 0.25f, 1.f);
      viz::LineWidth(4.f);
      viz::Primitive(
          viz::PrimitiveTopology::LINE_LIST_3D, 12, sizeof(data) / sizeof(data[0]), data);

      viz::Color(0.f, 1.f, 0.f, 1.f);
      viz::PointSize(6.f);
      viz::Primitive(
          viz::PrimitiveTopology::POINT_LIST_3D, 24, sizeof(data) / sizeof(data[0]), data);
    }
    {
      const float data[]{-0.125f, -0.125f, 0.f, 0.125f, -0.125f, 0.f, 0.125f, 0.125f, 0.f};

      viz::Color(0.f, 0.f, 1.f, 1.f);
      viz::Primitive(
          viz::PrimitiveTopology::TRIANGLE_LIST_3D, 1, sizeof(data) / sizeof(data[0]), data);
    }

    viz::EndLayer();
  }

  viz::End();

  // timing
  ++iterations;
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                                  start);
  if (!benchmark_mode && (elapsed.count() > 1000)) {
    fps = static_cast<float>(iterations) / (static_cast<float>(elapsed.count()) / 1000.f);
    start = std::chrono::steady_clock::now();
    iterations = 0;
  }
}

void initCuda() {
  if (cuInit(0) != CUDA_SUCCESS) { throw std::runtime_error("cuInit failed."); }

  if (cuDevicePrimaryCtxRetain(&cuda_context, 0) != CUDA_SUCCESS) {
    throw std::runtime_error("cuDevicePrimaryCtxRetain failed.");
  }

  if (cuCtxPushCurrent(cuda_context) != CUDA_SUCCESS) {
    throw std::runtime_error("cuDevicePrimaryCtxRetain failed.");
  }
}

void cleanupCuda() {
  if (cu_device_mem_r8) {
    if (cuMemFree(cu_device_mem_r8) != CUDA_SUCCESS) {
      throw std::runtime_error("cuMemFree failed.");
    }
  }
  if (cu_device_mem_r8g8b8) {
    if (cuMemFree(cu_device_mem_r8g8b8) != CUDA_SUCCESS) {
      throw std::runtime_error("cuMemFree failed.");
    }
  }
  if (cu_device_mem_r8g8b8a8) {
    if (cuMemFree(cu_device_mem_r8g8b8a8) != CUDA_SUCCESS) {
      throw std::runtime_error("cuMemFree failed.");
    }
  }
  if (cuda_context) {
    if (cuCtxPopCurrent(&cuda_context) != CUDA_SUCCESS) {
      throw std::runtime_error("cuCtxPopCurrent failed.");
    }
    cuda_context = nullptr;

    if (cuDevicePrimaryCtxRelease(0) != CUDA_SUCCESS) {
      throw std::runtime_error("cuDevicePrimaryCtxRelease failed.");
    }
  }
}

void loadImage() {
  int components;

  stbi_uc* const image_data = stbi_load("nv_logo.png",
                                        reinterpret_cast<int*>(&width),
                                        reinterpret_cast<int*>(&height),
                                        &components,
                                        0);
  if (!image_data) { throw std::runtime_error("Loading image failed."); }

  // allocate and set host memory
  host_mem_r8.reset(new uint8_t[width * height]);
  host_mem_r8g8b8.reset(new uint8_t[width * height * 3]);
  host_mem_r8g8b8a8.reset(new uint8_t[width * height * 4]);

  uint8_t const* src = image_data;

  uint8_t* dst_r8 = host_mem_r8.get();
  uint8_t* dst_r8g8b8a8 = host_mem_r8g8b8a8.get();
  uint8_t* dst_r8g8b8 = host_mem_r8g8b8.get();
  for (uint32_t i = 0; i < width * height; ++i) {
    dst_r8g8b8[0] = src[0];
    dst_r8g8b8[1] = src[1];
    dst_r8g8b8[2] = src[2];
    dst_r8g8b8 += 3;

    dst_r8g8b8a8[0] = src[0];
    dst_r8g8b8a8[1] = src[1];
    dst_r8g8b8a8[2] = src[2];
    dst_r8g8b8a8[3] = (components == 4) ? src[3] : 0xFF;
    const uint32_t pixel = *reinterpret_cast<uint32_t*>(dst_r8g8b8a8);
    dst_r8g8b8a8 += 4;

    std::vector<uint32_t>::iterator it = std::find(palette.begin(), palette.end(), pixel);
    if (it == palette.end()) {
      palette.push_back(pixel);
      it = --palette.end();
    }
    dst_r8[0] = std::distance(palette.begin(), it);
    dst_r8 += 1;

    src += components;
  }

  stbi_image_free(image_data);

  // allocate and set device memory
  if (cuMemAlloc(&cu_device_mem_r8, width * height) != CUDA_SUCCESS) {
    throw std::runtime_error("cuMemAlloc failed.");
  }
  if (cuMemcpyHtoD(cu_device_mem_r8, host_mem_r8.get(), width * height) != CUDA_SUCCESS) {
    throw std::runtime_error("cuMemcpyHtoD failed.");
  }
  if (cuMemAlloc(&cu_device_mem_r8g8b8, width * height * 3) != CUDA_SUCCESS) {
    throw std::runtime_error("cuMemAlloc failed.");
  }
  if (cuMemcpyHtoD(cu_device_mem_r8g8b8, host_mem_r8g8b8.get(), width * height * 3) !=
      CUDA_SUCCESS) {
    throw std::runtime_error("cuMemcpyHtoD failed.");
  }
  if (cuMemAlloc(&cu_device_mem_r8g8b8a8, width * height * 4) != CUDA_SUCCESS) {
    throw std::runtime_error("cuMemAlloc failed.");
  }
  if (cuMemcpyHtoD(cu_device_mem_r8g8b8a8, host_mem_r8g8b8a8.get(), width * height * 4) !=
      CUDA_SUCCESS) {
    throw std::runtime_error("cuMemcpyHtoD failed.");
  }
}

int main(int argc, char** argv) {
  struct option long_options[] = {{"help", no_argument, 0, 'h'},
                                  {"bench", no_argument, 0, 'b'},
                                  {"headless", no_argument, 0, 'l'},
                                  {"fullscreen", no_argument, 0, 'f'},
                                  {"exclusive_display", no_argument, 0, 'e'},
                                  {"display", required_argument, 0, 'd'},
                                  {"present_mode", required_argument, 0, 'd'},
                                  {0, 0, 0, 0}};
  bool fullscreen = false;
  bool exclusive_display = false;
  std::string display_name;
  viz::PresentMode present_mode = viz::PresentMode::AUTO;
  // parse options
  while (true) {
    int option_index = 0;
    const int c = getopt_long(argc, argv, "hblfed:p:", long_options, &option_index);

    if (c == -1) { break; }

    const std::string argument(optarg ? optarg : "");
    switch (c) {
      case 'h':
        std::cout << "Usage: " << argv[0] << " [options]" << std::endl
                  << "Options:" << std::endl
                  << "  -h, --help               display this information" << std::endl
                  << "  -b, --bench              benchmark mode" << std::endl
                  << "  -l, --headless           headless mode" << std::endl
                  << "  -f, --fullscreen         fullscreen mode" << std::endl
                  << "  -e, --exclusive_display  exclusive display mode" << std::endl
                  << "  -d, --display NAME       name of the display to use in exclusive display "
                     "or fullscreen mode (either EDID, `xrandr` or `hwinfo --monitor` name)"
                  << "  -p, --present_mode MODE  determines how the rendered result will be "
                     "presented on the screen (`auto`, `fifo`, `immediate` or `mailbox`)"
                  << std::endl;
        return EXIT_SUCCESS;

      case 'b':
        benchmark_mode = true;
        show_ui = false;
        show_geometry_layer = false;
        show_geometry_3d_layer = false;
        break;
      case 'l':
        headless_mode = true;
        show_ui = false;
        break;
      case 'f':
        fullscreen = true;
        break;
      case 'e':
        exclusive_display = true;
        break;
      case 'd':
        display_name = argument;
        break;
      case 'p': {
        std::string lower_case_argument;
        std::transform(argument.begin(),
                       argument.end(),
                       std::back_inserter(lower_case_argument),
                       [](unsigned char c) { return std::tolower(c); });
        if (lower_case_argument == "auto") {
          present_mode = viz::PresentMode::AUTO;
        } else if (lower_case_argument == "fifo") {
          present_mode = viz::PresentMode::FIFO;
        } else if (lower_case_argument == "immediate") {
          present_mode = viz::PresentMode::IMMEDIATE;
        } else if (lower_case_argument == "mailbox") {
          present_mode = viz::PresentMode::MAILBOX;
        } else {
          throw std::runtime_error("Unhandled present mode");
        }
      } break;
      default:
        throw std::runtime_error("Unhandled option");
    }
  }

  try {
    initCuda();
    loadImage();

    // set the Cuda stream to be used by Holoviz
    viz::SetCudaStream(CU_STREAM_PER_THREAD);
    viz::SetPresentMode(present_mode);

    uint32_t display_width, display_height;

    // setup the window
    if (exclusive_display) {
      display_width = width;
      display_height = height;
      viz::Init(display_name.c_str(), width, height);
    } else {
      viz::InitFlags flags = viz::InitFlags::NONE;

      if (headless_mode) { flags = viz::InitFlags::HEADLESS; }
      if (fullscreen) { flags = viz::InitFlags::FULLSCREEN; }

      display_width = 1024;
      display_height = uint32_t(static_cast<float>(height) / static_cast<float>(width) * 1024.f);
      viz::Init(display_width,
                display_height,
                "Holoviz Example",
                flags,
                display_name.empty() ? nullptr : display_name.c_str());
    }

    if (benchmark_mode) {
      for (auto source : {Source::DEVICE, Source::HOST}) {
        current_source = source;
        for (auto format_index : {2, 1, 0}) {
          current_format_index = format_index;
          start = std::chrono::steady_clock::time_point();
          do { tick(); } while (elapsed.count() < 2000);
          std::cout << current_source << " " << format_items[current_format_index] << " "
                    << float(iterations) / (float(elapsed.count()) / 1000.f) << " fps" << std::endl;
        }
      }
    } else if (headless_mode) {
      tick();

      // allocate a cuda buffer to hold the framebuffer data
      const size_t data_size = display_width * display_height * 4 * sizeof(uint8_t);
      CUdeviceptr read_buffer;
      if (cuMemAlloc(&read_buffer, data_size) != CUDA_SUCCESS) {
        throw std::runtime_error("cuMemAlloc failed.");
      }

      // read back the framebuffer
      viz::ReadFramebuffer(
          viz::ImageFormat::R8G8B8A8_UNORM, display_width, display_height, data_size, read_buffer);

      std::vector<uint8_t> data(data_size);
      if (cuMemcpyDtoHAsync(data.data(), read_buffer, data_size, CU_STREAM_PER_THREAD) !=
          CUDA_SUCCESS) {
        throw std::runtime_error("cuMemcpyDtoHAsync failed.");
      }
      if (cuStreamSynchronize(CU_STREAM_PER_THREAD) != CUDA_SUCCESS) {
        throw std::runtime_error("cuStreamSynchronize failed.");
      }

      // write to a file
      const char* filename = "framebuffer.png";
      std::cout << "Writing image to " << filename << "." << std::endl;
      stbi_write_png("framebuffer.png", display_width, display_height, 4, data.data(), 0);

      if (cuMemFree(read_buffer) != CUDA_SUCCESS) { throw std::runtime_error("cuMemFree failed."); }
    } else {
      while (!viz::WindowShouldClose()) {
        if (!viz::WindowIsMinimized()) { tick(); }
      }
    }

    viz::Shutdown();

    cleanupCuda();
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
