/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
// example code

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "holoviz/image.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <holoviz/holoviz.hpp>

#include <cuda.h>

#include "../example_utils.hpp"

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

holoviz::Buffer make_vertex_buffer(holoviz::Device& device) {
  static const float positions[16] = {0.5f,
                                      -0.5f,
                                      1.0f,
                                      0.0f,
                                      -0.5f,
                                      0.5f,
                                      0.0f,
                                      1.0f,
                                      -0.5f,
                                      -0.5f,
                                      0.0f,
                                      0.0f,
                                      0.5f,
                                      0.5f,
                                      1.0f,
                                      1.0f};

  holoviz::BufferDesc buffer_desc{
      .size_bytes = sizeof(positions),
      .share = holoviz::ShareMode::kExported,  // Mark the buffer as exported for external access
  };

  auto buffer = UNWRAP(device.create_buffer(buffer_desc));

  CHECK_CUDA_ERROR(cuMemcpyHtoD(UNWRAP(buffer.cuda_device_ptr()), positions, sizeof(positions)));

  return buffer;
}

holoviz::Buffer make_index_buffer(holoviz::Device& device) {
  static const uint16_t indices[6] = {0, 1, 2, 0, 3, 1};
  holoviz::BufferDesc buffer_desc{
      .size_bytes = sizeof(indices),
  };
  auto buffer = UNWRAP(device.create_buffer(buffer_desc));
  auto upload_span = std::span{reinterpret_cast<const std::byte*>(indices), sizeof(indices)};
  CHECK_ERROR(device.upload_to_buffer(UNWRAP(buffer.whole()), upload_span));
  return buffer;
}

void load_image(std::vector<uint8_t>& dst_array, int& width, int& height, int& channels,
                const char* res_path) {
  const std::string image_path = std::string(res_path) + "/nv_logo.png";

  unsigned char* data = stbi_load(image_path.c_str(), &width, &height, &channels, 0);
  if (!data) {
    throw std::runtime_error("Failed to load image " + image_path + ": " +
                             std::string(stbi_failure_reason()));
  }

  if (channels != 4)
    throw std::runtime_error("Image " + image_path + " must have 4 channels (RGBA).");

  const uint32_t row_pitch = width * channels;
  const size_t image_data_size = static_cast<size_t>(height) * row_pitch;

  dst_array.resize(static_cast<size_t>(height) * static_cast<size_t>(width) * 4);
  uint8_t* dst_ptr = dst_array.data();

  for (uint32_t y = 0; y < height; ++y) {
    // explicitly bound the source row offset to the loaded buffer (satisfies cpp:S3519)
    const size_t row_offset = static_cast<size_t>(y) * row_pitch;
    if (row_offset + row_pitch > image_data_size) {
      break;
    }
    uint8_t const* src = &data[row_offset];
    for (uint32_t x = 0; x < width; ++x) {
      const uint8_t r = src[0];
      const uint8_t g = src[1];
      const uint8_t b = src[2];

      dst_ptr[0] = r;
      dst_ptr[1] = g;
      dst_ptr[2] = b;
      dst_ptr[3] = (channels == 4) ? src[3] : 0xFF;
      dst_ptr += 4;

      src += channels;
    }
  }

  stbi_image_free(data);
}

holoviz::Image make_image(holoviz::Device& device, const char* res_path) {
  int width, height, channels;
  std::vector<uint8_t> image_data;
  load_image(image_data, width, height, channels, res_path);

  std::cout << "Loaded image: " << width << "x" << height << " with " << channels << " channels."
            << std::endl;

  holoviz::ImageDesc image_desc{
      .extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)},
      .format = holoviz::Format::kR8G8B8A8Unorm,
      .usage = holoviz::ImageUsage::kSampled,
      .share = holoviz::ShareMode::kExported,
  };

  auto image = UNWRAP(device.create_image(image_desc));

  std::cout << "Created image of size " << width << "x" << height << "." << std::endl;

  CUarray base_level{};
  CHECK_CUDA_ERROR(cuMipmappedArrayGetLevel(&base_level, UNWRAP(image.cuda_mipmapped_array()), 0));

  CUDA_MEMCPY2D copy_desc{};
  copy_desc.srcMemoryType = CU_MEMORYTYPE_HOST;
  copy_desc.srcHost = image_data.data();
  copy_desc.srcPitch = static_cast<size_t>(width) * 4;
  copy_desc.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  copy_desc.dstArray = base_level;
  copy_desc.WidthInBytes = static_cast<size_t>(width) * 4;
  copy_desc.Height = static_cast<size_t>(height);

  CHECK_CUDA_ERROR(cuMemcpy2D(&copy_desc));

  return image;
}

holoviz::RenderPass make_render_pass(holoviz::Device& device, holoviz::Format format) {
  holoviz::RenderPassDesc render_pass_desc{.render_targets = {{
                                               .format = format,
                                           }}};

  return UNWRAP(device.create_render_pass(render_pass_desc));
}

holoviz::Pipeline make_pipeline(holoviz::Device& device, const char* res_path) {
  const std::string shader_path = std::string(res_path) + "/shader.slang";

  std::ifstream shader_file(shader_path);
  if (!shader_file.is_open()) {
    throw std::runtime_error("Failed to open shader file: " + shader_path);
  }

  shader_file.seekg(0, std::ios::end);
  size_t shader_size = shader_file.tellg();
  shader_file.seekg(0, std::ios::beg);

  std::string shader_source(shader_size, '\0');
  shader_file.read(&shader_source[0], shader_size);

  shader_file.close();

  auto shader_blob = UNWRAP(holoviz::compile_slang_shader(
      UNWRAP(device.backend()), "vertex_main", "fragment_main", shader_source.c_str()));

  // print shader attributes and bindings
  for (auto attribute : shader_blob.attributes) {
    std::cout << "Attribute: name=" << attribute.name << ", location=" << attribute.location
              << ", format=" << static_cast<int>(attribute.format)
              << ", byte_size=" << attribute.byte_size << std::endl;
  }

  for (auto binding : shader_blob.bindings) {
    std::cout << "Binding: name=" << binding.name << ", index=" << binding.index
              << ", kind=" << static_cast<int>(binding.kind) << ", count=" << binding.count
              << std::endl;
  }

  std::cout << "Push constant size: " << shader_blob.push_constant_size << std::endl;

  holoviz::PipelineDesc pipeline_desc{
      .shader_blob = shader_blob,
      .topology = holoviz::PrimitiveTopology::kTriangleList,
      .raster =
          holoviz::RasterState{
              .cull = holoviz::CullMode::kBack,
          },
      .depth =
          holoviz::DepthState{
              .test = false,
              .write = false,
              .compare = holoviz::CompareOp::kAlways,
          },
      .blend =
          holoviz::BlendState{
              .enable = true,
          },
  };

  return UNWRAP(device.create_pipeline(pipeline_desc));
}

int run(int argc, char** argv) {
  const auto args =
      holoviz::examples::parse_resource_path_args(argc, argv, "holoviz_textured_quad_assets");
  if (args.exit_code) {
    return *args.exit_code;
  }
  const std::string& res_path = args.res_path;

  std::cout << argv[0] << " - HoloViz Textured Quad\n";
  std::cout << "Resource path: " << res_path << "\n";

  // Initialize CUDA and bind the primary context for HoloViz device before importing any resources
  // into CUDA
  CHECK_CUDA_ERROR(cuInit(0));
  CUdevice cu_dev{};
  CHECK_CUDA_ERROR(cuDeviceGet(&cu_dev, 0));
  CUcontext cu_ctx{};
  CHECK_CUDA_ERROR(cuDevicePrimaryCtxRetain(&cu_ctx, cu_dev));
  CHECK_CUDA_ERROR(cuCtxSetCurrent(cu_ctx));
  CUstream cuda_stream{};
  CHECK_CUDA_ERROR(cuStreamCreate(&cuda_stream, CU_STREAM_NON_BLOCKING));

  holoviz::WindowDesc window_desc{
      .title = "Holoviz Example",
      .size = {800, 600},
      .resizable = true,
  };

  auto window = UNWRAP(holoviz::create_window(window_desc));
  auto surface = UNWRAP(window.create_surface());

  holoviz::DeviceDesc device_desc{
      .cuda_device = cu_dev,
      .cuda_context = cu_ctx,
  };

  auto device = UNWRAP(holoviz::create_device_for_surface(device_desc, surface));

  const holoviz::Adapter adapter = UNWRAP(device.adapter());
  std::cout << "Using device: " << adapter.name.data() << std::endl;
  std::cout << "Backend: " << static_cast<int>(UNWRAP(device.backend())) << std::endl;
  std::cout << "CUDA device index: " << adapter.cuda_device_index << std::endl;
  std::cout << "Device UUID: ";
  for (auto byte : adapter.uuid) {
    std::cout << std::hex << static_cast<int>(byte);
  }
  std::cout << std::dec << std::endl;

  holoviz::SwapchainDesc swapchain_desc{
      .surface = surface,
      .present_mode = holoviz::PresentMode::kFIFO,
      .image_count = 3,
  };

  auto swapchain = UNWRAP(device.create_swapchain(swapchain_desc));

  auto vertex_buffer = make_vertex_buffer(device);
  auto index_buffer = make_index_buffer(device);
  auto image = make_image(device, res_path.c_str());
  auto render_pass = make_render_pass(device, UNWRAP(swapchain.format()));
  auto pipeline = make_pipeline(device, res_path.c_str());

  const uint32_t swapchain_image_count = UNWRAP(swapchain.image_count());

  holoviz::BufferDesc rotation_buffer_desc{
      .size_bytes = swapchain_image_count * sizeof(float),  // Size for a single float (angle)
      .share = holoviz::ShareMode::kExported,
  };
  auto rotation_buffer = UNWRAP(device.create_buffer(rotation_buffer_desc));

  auto descriptor = UNWRAP(pipeline.create_descriptor());

  // we can bind the image and sampler to the descriptor using the names from the shader
  CHECK_ERROR(descriptor.bind_image(image, "myTexture"));
  CHECK_ERROR(descriptor.bind_sampler(device, {}, "mySampler"));
  CHECK_ERROR(descriptor.bind_buffer(UNWRAP(rotation_buffer.whole()), "rotation"));

  // Async cuda memsets on a dedicated stream.

  // CHECK_ERROR(device.cuda_set_convergence_stream(cuda_stream));

  std::vector<holoviz::SubmitSyncToken> sync_tokens(
      swapchain_image_count, holoviz::SubmitSyncToken{});  // Timeline values for each frame

  const CUdeviceptr rotation_device_ptr = UNWRAP(rotation_buffer.cuda_device_ptr());

  std::vector<holoviz::CommandList> command_lists;
  for (size_t i = 0; i < swapchain_image_count; ++i) {
    command_lists.push_back(
        UNWRAP(device.create_command_list({.type = holoviz::CommandListType::kTransient})));
  }

  float angle = 0.0f;

  auto external_semaphore = UNWRAP(device.create_external_semaphore(cuda_stream));

  // Window resizes are handled inside the swapchain (the window is resizable, so acquire()
  // tracks the surface and recreates in place — handle, format, and image indices all
  // survive, keeping everything created above valid). Only two conditions surface here:
  // kBackpressured (display busy or window minimized — retry) and kImageCountChanged (a
  // driver returned a different image count across a recreation; this example sizes its
  // per-image resources — sync_tokens, command_lists, rotation_buffer — once, so it treats
  // that as fatal where a resilient app would rebuild them).
  while (!UNWRAP(window.close_requested())) {
    holoviz::poll_window_events();

    angle -= 0.0025f;

    auto acquire_result = swapchain.acquire(std::chrono::milliseconds(100));
    if (!acquire_result) {
      if (acquire_result.error().code == holoviz::ErrorCode::kBackpressured ||
          acquire_result.error().code == holoviz::ErrorCode::kOutOfDate) {
        continue;
      }
      throw std::runtime_error("Error in `swapchain.acquire()` (code " +
                               std::to_string(static_cast<uint32_t>(acquire_result.error().code)) +
                               "): " + acquire_result.error().message);
    }
    auto acquired = std::move(acquire_result).value();

    auto& command_list = command_lists[acquired.image_index];
    auto& acquire_gate = acquired.acquire_gate;
    auto& present_gate = acquired.present_gate;

    CHECK_ERROR(external_semaphore.wait(sync_tokens[acquired.image_index]));

    uint32_t bits;
    std::memcpy(&bits, &angle, sizeof(bits));
    CHECK_CUDA_ERROR(cuMemsetD32Async(
        rotation_device_ptr + acquired.image_index * sizeof(float), bits, 1, cuda_stream));

    auto external_sync_token = UNWRAP(external_semaphore.signal());

    CHECK_ERROR(command_list.reset(std::chrono::milliseconds(100)));

    auto current_image = UNWRAP(swapchain.image(acquired.image_index));

    CHECK_ERROR(render_pass.set_render_image(current_image, 0, {0.3F, 0.5F, 0.8F, 1.0F}));

    const holoviz::Extent2D swapchain_extent = UNWRAP(swapchain.extent());
    holoviz::Rect2D render_area{0, 0, swapchain_extent.width, swapchain_extent.height};

    CHECK_ERROR(command_list.begin());
    CHECK_ERROR(command_list.begin_render(render_pass, render_area));

    CHECK_ERROR(command_list.set_viewport(render_area));
    CHECK_ERROR(command_list.set_scissor(render_area));

    CHECK_ERROR(command_list.bind_pipeline(pipeline));
    CHECK_ERROR(command_list.bind_vertex_buffer(0, UNWRAP(vertex_buffer.whole())));
    CHECK_ERROR(
        command_list.bind_index_buffer(UNWRAP(index_buffer.whole()), holoviz::IndexType::kUint16));
    CHECK_ERROR(command_list.bind_descriptor(descriptor));

    uint32_t rotation_index = static_cast<uint32_t>(acquired.image_index);

    auto push_constant_span =
        std::span{reinterpret_cast<const std::byte*>(&rotation_index), sizeof(rotation_index)};
    CHECK_ERROR(command_list.set_push_data(pipeline, push_constant_span));

    CHECK_ERROR(command_list.draw_indexed(6, 1, 0, 0, 0));

    CHECK_ERROR(command_list.end_render());
    CHECK_ERROR(command_list.end());

    holoviz::SubmitDesc submit_desc{
        // .timeline_wait_value = timeline_values[acquired.image_index],
        // .cuda_timeline_wait_value = signal_value,
        .wait_submit_sync_tokens = {&sync_tokens[acquired.image_index], 1},
        .wait_external_sync_tokens = {&external_sync_token, 1},
        .command_lists = {&command_list, 1},
        .acquire_gates = {&acquire_gate, 1},
        .present_gates = {&present_gate, 1},
    };

    // CUDA will be able to wait on this timeline value through the exported semaphore
    const uint32_t image_index = acquired.image_index;
    sync_tokens[image_index] = UNWRAP(device.submit(submit_desc));

    // std::cout << "Submitted frame " << acquired.frame_index << " for image ptr " <<
    // swapchain.image(acquired.image_index).value() << " with timeline value " <<
    // timeline_value << std::endl;

    auto present_result = swapchain.present(std::move(acquired), sync_tokens[image_index]);
    if (!present_result) {
      if (present_result.error().code == holoviz::ErrorCode::kFrameDropped) {
        // A resize consumed this frame before display; the swapchain already recreated
        // itself — just render the next frame at the new size.
        continue;
      }
      throw std::runtime_error("Error in `swapchain.present()` (code " +
                               std::to_string(static_cast<uint32_t>(present_result.error().code)) +
                               "): " + present_result.error().message);
    }
  }

  // Wait for the device to finish all operations before exiting
  CHECK_ERROR(device.wait_idle());

  CHECK_CUDA_ERROR(cuStreamSynchronize(cuda_stream));
  CHECK_CUDA_ERROR(cuStreamDestroy(cuda_stream));

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

// NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index,cppcoreguidelines-pro-bounds-array-to-pointer-decay,hicpp-no-array-decay)
