/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <holoviz/holoviz.hpp>

namespace viz = holoscan::viz;

uint32_t width;
uint32_t height;
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

bool unlimited_fps = false;
float fps = 15.f;
int color_index = 0;
float line_width = 1.f;
float point_size = 1.f;

// cuda
CUcontext cuda_context = nullptr;
std::vector<CUdeviceptr> depth_mems;
std::vector<CUdeviceptr> color_mems;

void loadImage(const std::string& filename, CUdeviceptr* cu_device_mem, int* width, int* height,
               int* components, bool convert_to_intensity) {
  const unsigned char* image_data =
      stbi_load(filename.c_str(), width, height, components, convert_to_intensity ? 0 : 4);
  if (!image_data) { throw std::runtime_error("Loading image failed."); }

  std::vector<uint8_t> tmp_image_data;
  if ((*components != 1) && (convert_to_intensity)) {
    tmp_image_data.reserve(*width * *height);
    for (int index = 0; index < *width * *height; ++index) {
      const uint8_t* src = &image_data[index * *components];
      tmp_image_data.push_back(static_cast<uint8_t>(0.2126f * static_cast<float>(src[0]) +
                                                    0.7152f * static_cast<float>(src[1]) +
                                                    0.0722f * static_cast<float>(src[2]) + 0.5f));
    }
    image_data = tmp_image_data.data();
    *components = 1;
  } else {
    *components = 4;
  }

  if (cuMemAlloc(cu_device_mem, *width * *height * *components) != CUDA_SUCCESS) {
    throw std::runtime_error("cuMemAlloc failed.");
  }
  if (cuMemcpyHtoD(*cu_device_mem, image_data, *width * *height * *components) != CUDA_SUCCESS) {
    throw std::runtime_error("cuMemcpyHtoD failed.");
  }
}

void freeSourceData() {
  for (auto&& depth_mem : depth_mems) {
    if (cuMemFree(depth_mem) != CUDA_SUCCESS) { throw std::runtime_error("cuMemFree failed."); }
  }
  depth_mems.clear();
  for (auto&& color_mem : color_mems) {
    if (cuMemFree(color_mem) != CUDA_SUCCESS) { throw std::runtime_error("cuMemFree failed."); }
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
      width = cur_width;
      height = cur_height;
      first = false;
    } else if ((cur_width != width) || (cur_height != height) || (cur_components != 1)) {
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
      width = cur_width;
      height = cur_height;
      first = false;
    } else if ((cur_width != width) || (cur_height != height) || (cur_components != 4)) {
      throw std::runtime_error("Inconsistent color image sequence");
    }
    color_mems.push_back(cu_device_mem);
  }
  std::cout << std::endl;
}

void generateSourceData(uint32_t frame_index) {
  const uint32_t images = 100;
  CUdeviceptr cu_device_mem;

  width = 64;
  height = 64;

  if (depth_mems.empty()) {
    for (uint32_t index = 0; index < images; ++index) {
      if (cuMemAlloc(&cu_device_mem, width * height * sizeof(uint8_t)) != CUDA_SUCCESS) {
        throw std::runtime_error("cuMemAlloc failed.");
      }
      depth_mems.push_back(cu_device_mem);
    }
  }
  if (color_mems.empty()) {
    for (uint32_t index = 0; index < images; ++index) {
      if (cuMemAlloc(&cu_device_mem, width * height * sizeof(uint32_t)) != CUDA_SUCCESS) {
        throw std::runtime_error("cuMemAlloc failed.");
      }
      color_mems.push_back(cu_device_mem);
    }
  }

  std::vector<uint8_t> depth_data(width * height);
  std::vector<uint32_t> color_data(width * height);
  for (uint32_t index = 0; index < images; ++index) {
    const float offset = float(frame_index) / float(images);

    for (uint32_t y = 0; y < height; ++y) {
      for (uint32_t x = 0; x < width; ++x) {
        const uint8_t depth = (std::sin((float(x) / float(width)) * 3.14f * 4.f) *
                                   std::cos((float(y) / float(height)) * 3.14f * 3.f) +
                               1.f) * offset *
                              63.f;

        depth_data[y * width + x] = depth;
        color_data[y * width + x] = depth | ((depth << (8 + (x & 1))) & 0xFF00) |
                                    ((depth << (16 + (y & 1) * 2)) & 0xFF0000) | 0xFF204060;
      }
    }

    if (cuMemcpyHtoD(depth_mems[index], depth_data.data(), depth_data.size() * sizeof(uint8_t)) !=
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

  // UI
  viz::BeginImGuiLayer();

  viz::LayerPriority(11);

  ImGui::Begin("Options", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

  bool regenerate = depth_mems.empty();
  regenerate |= ImGui::Combo("Source Mode",
                             reinterpret_cast<int*>(&current_source_mode),
                             source_mode_items,
                             IM_ARRAYSIZE(source_mode_items));

  if (current_source_mode == SourceMode::FILES) {
    ImGui::InputText("Depth Dir", depth_dir, sizeof(depth_dir));
    ImGui::InputText("Color Dir", color_dir, sizeof(color_dir));
    regenerate |= ImGui::Button("Load");
  }

  ImGui::Separator();
  ImGui::Combo("Render Mode",
               reinterpret_cast<int*>(&current_render_mode),
               render_mode_items,
               IM_ARRAYSIZE(render_mode_items));

  switch (current_render_mode) {
    case RenderMode::DEPTH:
      if (ImGui::Button("+")) {
        palette.push_back(static_cast<uint32_t>(std::rand()) | 0xFF000000);
      }
      ImGui::SameLine();
      if (ImGui::Button("-") && (palette.size() > 1)) { palette.pop_back(); }
      ImGui::SliderInt("LUT index", &color_index, 0, palette.size() - 1);

      {
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
      break;
    case RenderMode::LINES:
      ImGui::SliderFloat("Line width", &line_width, 1.f, 20.f);
      break;
    case RenderMode::POINTS:
      ImGui::SliderFloat("Point size", &point_size, 1.f, 20.f);
      break;
  }

  ImGui::End();

  viz::EndLayer();

  // regenerate source data if needed
  if (regenerate) {
    frame_index = 0;
    freeSourceData();

    switch (current_source_mode) {
      case SourceMode::FILES:
        loadSourceData(depth_dir, color_dir);
        break;
      case SourceMode::GENERATED:
        generateSourceData(frame_index);
        break;
    }
  } else if (current_source_mode == SourceMode::GENERATED) {
    generateSourceData(frame_index);
  }

  if (!depth_mems.empty()) {
    switch (current_render_mode) {
      case RenderMode::DEPTH:
        viz::BeginImageLayer();

        // Image with LUT
        viz::LUT(palette.size(),
                 viz::ImageFormat::R8G8B8A8_UNORM,
                 palette.size() * sizeof(uint32_t),
                 palette.data(),
                 true);

        viz::ImageCudaDevice(width, height, viz::ImageFormat::R8_UNORM, depth_mems[frame_index]);

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
                      width,
                      height,
                      viz::ImageFormat::R8_UNORM,
                      depth_mems[frame_index],
                      viz::ImageFormat::R8G8B8A8_UNORM,
                      color_mems.size() > frame_index ? color_mems[frame_index] : 0);

        viz::EndLayer();

        break;
    }
    ++frame_index;
    if (frame_index >= depth_mems.size()) { frame_index = 0; }
  }

  viz::End();
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
  struct option long_options[] = {{"help", no_argument, 0, 'h'},
                                  {"depth_dir", required_argument, 0, 'd'},
                                  {"color_dir", required_argument, 0, 'c'},
                                  {0, 0, 0, 0}};

  // parse options
  while (true) {
    int option_index = 0;
    const int c = getopt_long(argc, argv, "hd:c:", long_options, &option_index);

    if (c == -1) { break; }

    const std::string argument(optarg ? optarg : "");
    switch (c) {
      case 'h':
        std::cout << "Usage: " << argv[0] << " [options]" << std::endl
                  << "Options:" << std::endl
                  << "  -d DIR, --depth_dir DIR  directory to load depth images from" << std::endl
                  << "  -c DIR, --color_dir DIR  directory to load color images from" << std::endl
                  << "  -h, --help               display this information" << std::endl
                  << std::endl;
        return EXIT_SUCCESS;
      case 'd':
        std::strncpy(depth_dir, argument.c_str(), sizeof(depth_dir));
        break;
      case 'c':
        std::strncpy(color_dir, argument.c_str(), sizeof(color_dir));
        break;
      default:
        throw std::runtime_error("Unhandled option ");
    }
  }

  initCuda();
  // If using ImGui, create a context and pass it to Holoviz, do this before calling viz::Init().
  ImGui::CreateContext();
  viz::ImGuiSetCurrentContext(ImGui::GetCurrentContext());

  viz::Init(1024, 768, "Holoviz Depth Render Example");

  while (!viz::WindowShouldClose()) {
    if (!viz::WindowIsMinimized()) {
      tick();
      if (!unlimited_fps) {
        std::this_thread::sleep_for(std::chrono::duration<float, std::milli>(1000.f / fps));
      }
    }
  }

  viz::Shutdown();

  cleanupCuda();

  return EXIT_SUCCESS;
}
