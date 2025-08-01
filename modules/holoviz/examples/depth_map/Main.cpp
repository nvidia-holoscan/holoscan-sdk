/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <misc/cpp/imgui_stdlib.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <holoviz/holoviz.hpp>

namespace viz = holoscan::viz;

uint32_t map_width = 0;
uint32_t map_height = 0;
uint32_t frame_index = 0;

std::vector<uint32_t> palette{0xFF000000, 0xFF7F7F7F, 0xFFFFFFFF};

// UI state
enum class SourceMode { GENERATED, FILES };
static const char* source_mode_items[]{"Generated", "Files"};
SourceMode current_source_mode = SourceMode::GENERATED;

char depth_dir[256];
char color_dir[256];

enum class RenderMode { POINTS, LINES, TRIANGLES, DEPTH };
static const char* render_mode_items[]{"Points", "Lines", "Triangles", "Depth"};
RenderMode current_render_mode = RenderMode::TRIANGLES;

static const char* format_items[]{"R8_UNORM", "D32_SFLOAT"};
static viz::ImageFormat formats[]{viz::ImageFormat::R8_UNORM, viz::ImageFormat::D32_SFLOAT};
uint32_t current_format_index = 0;

bool show_ui = true;
bool unlimited_fps = false;
float fps = 15.F;
int color_index = 0;
float line_width = 1.F;
float point_size = 1.F;

// cuda
CUcontext cuda_context = nullptr;
std::vector<CUdeviceptr> depth_mems;
std::vector<CUdeviceptr> color_mems;

void loadImage(const std::string& filename, CUdeviceptr* cu_device_mem, int* width, int* height,
               int* components, bool convert_to_intensity) {
  const void* image_data = nullptr;
  stbi_uc* const file_image_data =
      stbi_load(filename.c_str(), width, height, components, convert_to_intensity ? 0 : 4);
  if (!file_image_data) {
    throw std::runtime_error("Loading image failed.");
  }

  std::vector<uint8_t> tmp_image_data;
  if ((*components != 1) && (convert_to_intensity)) {
    tmp_image_data.reserve(*width * *height);
    for (int index = 0; index < *width * *height; ++index) {
      const uint8_t* src = &file_image_data[index * *components];
      tmp_image_data.push_back(static_cast<uint8_t>(0.2126F * static_cast<float>(src[0]) +
                                                    0.7152F * static_cast<float>(src[1]) +
                                                    0.0722F * static_cast<float>(src[2]) + 0.5F));
    }
    image_data = tmp_image_data.data();
    *components = 1;
  } else {
    image_data = file_image_data;
    *components = 4;
  }

  if (cuMemAlloc(cu_device_mem, *width * *height * *components) != CUDA_SUCCESS) {
    throw std::runtime_error("cuMemAlloc failed.");
  }
  if (cuMemcpyHtoD(*cu_device_mem, image_data, *width * *height * *components) != CUDA_SUCCESS) {
    throw std::runtime_error("cuMemcpyHtoD failed.");
  }

  stbi_image_free(file_image_data);
}

void freeSourceData() {
  for (auto&& depth_mem : depth_mems) {
    if (cuMemFree(depth_mem) != CUDA_SUCCESS) {
      throw std::runtime_error("cuMemFree failed.");
    }
  }
  depth_mems.clear();
  for (auto&& color_mem : color_mems) {
    if (cuMemFree(color_mem) != CUDA_SUCCESS) {
      throw std::runtime_error("cuMemFree failed.");
    }
  }
  color_mems.clear();
}

void loadSourceData(const char* depth_dir, const char* color_dir) {
  std::vector<std::string> depth_files;
  for (auto const& dir_entry : std::filesystem::directory_iterator{depth_dir}) {
    depth_files.push_back(dir_entry.path());
  }
  std::sort(depth_files.begin(), depth_files.end());

  bool first = true;
  CUdeviceptr cu_device_mem;
  int cur_width, cur_height, cur_components;

  for (auto&& file_name : depth_files) {
    std::cout << "\rReading depth image " << file_name << std::flush;
    loadImage(file_name, &cu_device_mem, &cur_width, &cur_height, &cur_components, true);
    if (first) {
      map_width = cur_width;
      map_height = cur_height;
      first = false;
    } else if ((cur_width != map_width) || (cur_height != map_height) || (cur_components != 1)) {
      throw std::runtime_error("Inconsistent depth image sequence");
    }
    depth_mems.push_back(cu_device_mem);
  }
  std::cout << std::endl;

  std::vector<std::string> color_files;
  for (auto const& dir_entry : std::filesystem::directory_iterator{color_dir}) {
    color_files.push_back(dir_entry.path());
  }
  std::sort(color_files.begin(), color_files.end());

  for (auto&& file_name : color_files) {
    std::cout << "\rReading color image " << file_name << std::flush;
    loadImage(file_name, &cu_device_mem, &cur_width, &cur_height, &cur_components, false);
    if (first) {
      map_width = cur_width;
      map_height = cur_height;
      first = false;
    } else if ((cur_width != map_width) || (cur_height != map_height) || (cur_components != 4)) {
      throw std::runtime_error("Inconsistent color image sequence");
    }
    color_mems.push_back(cu_device_mem);
  }
  std::cout << std::endl;
}

void generateSourceData() {
  const uint32_t images = 100;
  CUdeviceptr cu_device_mem;

  if (map_width == 0) {
    map_width = 512;
  }
  if (map_height == 0) {
    map_height = 512;
  }

  size_t element_size;
  std::function<void(void*, size_t, float)> write_depth;
  switch (formats[current_format_index]) {
    case viz::ImageFormat::R8_UNORM:
      element_size = sizeof(uint8_t);
      write_depth = [](void* data, size_t index, float value) {
        reinterpret_cast<uint8_t*>(data)[index] = value * 63.F;
      };
      break;
    case viz::ImageFormat::D32_SFLOAT:
      element_size = sizeof(float);
      write_depth = [](void* data, size_t index, float value) {
        reinterpret_cast<float*>(data)[index] = value / 4.F;
      };
      break;
    default:
      throw std::runtime_error("Unhandled format");
  }

  if (depth_mems.empty()) {
    for (uint32_t index = 0; index < images; ++index) {
      if (cuMemAlloc(&cu_device_mem, map_width * map_height * element_size) != CUDA_SUCCESS) {
        throw std::runtime_error("cuMemAlloc failed.");
      }
      depth_mems.push_back(cu_device_mem);
    }
  }
  if (color_mems.empty()) {
    for (uint32_t index = 0; index < images; ++index) {
      if (cuMemAlloc(&cu_device_mem, map_width * map_height * sizeof(uint32_t)) != CUDA_SUCCESS) {
        throw std::runtime_error("cuMemAlloc failed.");
      }
      color_mems.push_back(cu_device_mem);
    }
  }

  auto depth_data = std::make_unique<uint8_t[]>(map_width * map_height * element_size);
  std::vector<uint32_t> color_data(map_width * map_height);
  for (uint32_t index = 0; index < images; ++index) {
    const float offset = float(index) / float(images);

    for (uint32_t y = 0; y < map_height; ++y) {
      for (uint32_t x = 0; x < map_width; ++x) {
        const float depth = (std::sin((float(x) / float(map_width)) * 3.14F * 4.F) *
                                 std::cos((float(y) / float(map_height)) * 3.14F * 3.F) +
                             1.F) *
                            offset;

        write_depth(depth_data.get(), y * map_width + x, depth);
        const uint8_t color = depth * 63.F;
        color_data[y * map_width + x] = color | ((color << (8 + (x & 1))) & 0xFF00) |
                                        ((color << (16 + (y & 1) * 2)) & 0xFF0000) | 0xFF204060;
      }
    }

    if (cuMemcpyHtoD(depth_mems[index], depth_data.get(), map_width * map_height * element_size) !=
        CUDA_SUCCESS) {
      throw std::runtime_error("cuMemcpyHtoD failed.");
    }

    if (cuMemcpyHtoD(color_mems[index], color_data.data(), color_data.size() * sizeof(uint32_t)) !=
        CUDA_SUCCESS) {
      throw std::runtime_error("cuMemcpyHtoD failed.");
    }
  }
}

void tick() {
  viz::Begin();

  bool regenerate = depth_mems.empty();
  if (show_ui) {
    // UI
    viz::BeginImGuiLayer();

    viz::LayerPriority(11);

    ImGui::Begin("Options", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

    regenerate |= ImGui::Combo("Source Mode",
                               reinterpret_cast<int*>(&current_source_mode),
                               source_mode_items,
                               IM_ARRAYSIZE(source_mode_items));

    if (current_source_mode == SourceMode::FILES) {
      ImGui::InputText("Depth Dir", depth_dir, sizeof(depth_dir));
      ImGui::InputText("Color Dir", color_dir, sizeof(color_dir));
      regenerate |= ImGui::Button("Load");
    } else if (current_source_mode == SourceMode::GENERATED) {
      regenerate |= ImGui::Combo("Format",
                                 reinterpret_cast<int*>(&current_format_index),
                                 format_items,
                                 IM_ARRAYSIZE(format_items));
    }

    ImGui::Separator();
    ImGui::Combo("Render Mode",
                 reinterpret_cast<int*>(&current_render_mode),
                 render_mode_items,
                 IM_ARRAYSIZE(render_mode_items));

    switch (current_render_mode) {
      case RenderMode::DEPTH:
        if (formats[current_format_index] == viz::ImageFormat::R8_UNORM) {
          if (ImGui::Button("+")) {
            palette.push_back(static_cast<uint32_t>(std::rand()) | 0xFF000000);
          }
          ImGui::SameLine();
          if (ImGui::Button("-") && (palette.size() > 1)) {
            palette.pop_back();
          }
          ImGui::SliderInt("LUT index", &color_index, 0, palette.size() - 1);

          {
            uint32_t& item = palette[color_index];
            float color[]{(item & 0xFF) / 255.F,
                          ((item >> 8) & 0xFF) / 255.F,
                          ((item >> 16) & 0xFF) / 255.F,
                          ((item >> 24) & 0xFF) / 255.F};
            ImGui::ColorEdit4("##color", color, ImGuiColorEditFlags_DefaultOptions_);
            item = static_cast<uint32_t>((color[0] * 255.F) + 0.5F) +
                   (static_cast<uint32_t>((color[1] * 255.F) + 0.5F) << 8) +
                   (static_cast<uint32_t>((color[2] * 255.F) + 0.5F) << 16) +
                   (static_cast<uint32_t>((color[3] * 255.F) + 0.5F) << 24);
          }
        }
        break;
      case RenderMode::LINES:
        ImGui::SliderFloat("Line width", &line_width, 1.F, 20.F);
        break;
      case RenderMode::POINTS:
        ImGui::SliderFloat("Point size", &point_size, 1.F, 20.F);
        break;
      case RenderMode::TRIANGLES:
        // No specific UI controls needed for triangles mode
        break;
    }

    ImGui::End();

    viz::EndLayer();
  }

  // regenerate source data if needed
  if (regenerate) {
    frame_index = 0;
    freeSourceData();

    switch (current_source_mode) {
      case SourceMode::FILES:
        loadSourceData(depth_dir, color_dir);
        break;
      case SourceMode::GENERATED:
        generateSourceData();
        break;
    }
  }

  if (!depth_mems.empty()) {
    switch (current_render_mode) {
      case RenderMode::DEPTH:
        viz::BeginImageLayer();

        switch (formats[current_format_index]) {
          case viz::ImageFormat::R8_UNORM:
            // Image with LUT
            viz::LUT(palette.size(),
                     viz::ImageFormat::R8G8B8A8_UNORM,
                     palette.size() * sizeof(uint32_t),
                     palette.data(),
                     true);

            viz::ImageCudaDevice(
                map_width, map_height, viz::ImageFormat::R8_UNORM, depth_mems[frame_index]);
            break;
          case viz::ImageFormat::D32_SFLOAT:
            // draw as intensity image
            viz::ImageComponentMapping(viz::ComponentSwizzle::R,
                                       viz::ComponentSwizzle::R,
                                       viz::ComponentSwizzle::R,
                                       viz::ComponentSwizzle::ONE);
            viz::ImageCudaDevice(
                map_width, map_height, viz::ImageFormat::R32_SFLOAT, depth_mems[frame_index]);
            break;
          default:
            throw std::runtime_error("Unhandled format");
        }

        viz::EndLayer();
        break;
      case RenderMode::POINTS:
      case RenderMode::LINES:
      case RenderMode::TRIANGLES:
        viz::BeginGeometryLayer();

        viz::DepthMapRenderMode render_mode;
        switch (current_render_mode) {
          case RenderMode::POINTS:
            render_mode = viz::DepthMapRenderMode::POINTS;
            viz::PointSize(point_size);
            break;
          case RenderMode::LINES:
            render_mode = viz::DepthMapRenderMode::LINES;
            viz::LineWidth(line_width);
            break;
          case RenderMode::TRIANGLES:
            render_mode = viz::DepthMapRenderMode::TRIANGLES;
            break;
          default:
            throw std::runtime_error("Unhandled mode.");
        }
        viz::DepthMap(render_mode,
                      map_width,
                      map_height,
                      formats[current_format_index],
                      depth_mems[frame_index],
                      viz::ImageFormat::R8G8B8A8_UNORM,
                      color_mems.size() > frame_index ? color_mems[frame_index] : 0);

        viz::EndLayer();

        break;
    }
    ++frame_index;
    if (frame_index >= depth_mems.size()) {
      frame_index = 0;
    }
  }

  viz::End();
}

void initCuda() {
  if (cuInit(0) != CUDA_SUCCESS) {
    throw std::runtime_error("cuInit failed.");
  }

  if (cuDevicePrimaryCtxRetain(&cuda_context, 0) != CUDA_SUCCESS) {
    throw std::runtime_error("cuDevicePrimaryCtxRetain failed.");
  }

  if (cuCtxPushCurrent(cuda_context) != CUDA_SUCCESS) {
    throw std::runtime_error("cuDevicePrimaryCtxRetain failed.");
  }
}

void cleanupCuda() {
  freeSourceData();
  if (cuda_context) {
    if (cuCtxPopCurrent(&cuda_context) != CUDA_SUCCESS) {
      throw std::runtime_error("cuDevicePrimaryCtxRetain failed.");
    }
    cuda_context = nullptr;

    if (cuDevicePrimaryCtxRelease(0) != CUDA_SUCCESS) {
      throw std::runtime_error("cuDevicePrimaryCtxRelease failed.");
    }
  }
}

int main(int argc, char** argv) {
  try {
    bool benchmark_mode = false;
    bool headless_mode = false;

    struct option long_options[] = {{"help", no_argument, 0, 'h'},
                                    {"depth_dir", required_argument, 0, 'd'},
                                    {"color_dir", required_argument, 0, 'c'},
                                    {"bench", no_argument, 0, 'b'},
                                    {"headless", no_argument, 0, 'l'},
                                    {0, 0, 0, 0}};

    // parse options
    while (true) {
      int option_index = 0;
      const int c =
          getopt_long(argc, argv, "hd:c:bl", static_cast<option*>(long_options), &option_index);

      if (c == -1) {
        break;
      }

      const std::string argument(optarg ? optarg : "");
      switch (c) {
        case 'h':
          std::cout << "Usage: " << argv[0] << " [options]" << std::endl
                    << "Options:" << std::endl
                    << "  -d DIR, --depth_dir DIR  directory to load depth images from" << std::endl
                    << "  -c DIR, --color_dir DIR  directory to load color images from" << std::endl
                    << "  -b, --bench              benchmark mode" << std::endl
                    << "  -l, --headless           headless mode" << std::endl
                    << "  -h, --help               display this information" << std::endl
                    << std::endl;
          return EXIT_SUCCESS;
        case 'b':
          benchmark_mode = true;
          show_ui = false;
          break;
        case 'l':
          headless_mode = true;
          break;
        case 'd':
          if (argument.length() >= sizeof(depth_dir)) {
            throw std::runtime_error("Depth directory path is too long (max " +
                                     std::to_string(sizeof(depth_dir) - 1) + " characters)");
          }
          snprintf(depth_dir, sizeof(depth_dir), "%s", argument.c_str());
          break;
        case 'c':
          if (argument.length() >= sizeof(color_dir)) {
            throw std::runtime_error("Color directory path is too long (max " +
                                     std::to_string(sizeof(color_dir) - 1) + " characters)");
          }
          snprintf(color_dir, sizeof(color_dir), "%s", argument.c_str());
          break;
        default:
          throw std::runtime_error("Unhandled option ");
      }
    }

    initCuda();

    viz::InitFlags flags = viz::InitFlags::NONE;

    if (headless_mode) {
      flags = viz::InitFlags::HEADLESS;
    }

    viz::Init(1024, 768, "Holoviz Depth Render Example", flags);

    if (benchmark_mode) {
      for (auto size : {64, 512}) {
        map_width = map_height = size;
        for (auto format_index : {0, 1}) {
          current_format_index = format_index;
          freeSourceData();

          for (auto render_mode :
               {RenderMode::DEPTH, RenderMode::POINTS, RenderMode::LINES, RenderMode::TRIANGLES}) {
            current_render_mode = render_mode;

            std::chrono::milliseconds elapsed;
            uint32_t iterations = 0;
            const auto start = std::chrono::steady_clock::now();
            do {
              tick();
              elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - start);
              ++iterations;
            } while (elapsed.count() < 2000);

            std::cout << size << " " << format_items[current_format_index] << " "
                      << render_mode_items[int(current_render_mode)] << " "
                      << float(iterations) / (float(elapsed.count()) / 1000.F) << " fps"
                      << std::endl;
          }
        }
      }
    } else {
      while (!viz::WindowShouldClose()) {
        if (!viz::WindowIsMinimized()) {
          tick();
          if (!unlimited_fps) {
            std::this_thread::sleep_for(std::chrono::duration<float, std::milli>(1000.F / fps));
          }
        }
      }
    }

    viz::Shutdown();

    cleanupCuda();
  } catch (std::exception& e) {
    std::cerr << e.what();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
