/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>

#include "holoviz_viewer.hpp"
#include <imgui.h>


namespace viz = clara::holoviz;

namespace nvidia::holoscan {

gxf_result_t HolovizViewer::registerInterface(nvidia::gxf::Registrar* registrar) {
  nvidia::gxf::Expected<void> result;
  result &= registrar->parameter(
    receiver_, "receiver", "Entity receiver",
    "Receiver channel to log");
  result &= registrar->parameter(
    input_image_name_, "input_image_name", "Name of image tensor",
    "Name of image in input message to display", std::string(""));
  result &= registrar->parameter(
    window_title_, "window_title", "Window title",
    "Title on window canvas", std::string("Holoviz Viewer"));
  result &= registrar->parameter(
    display_name_, "display_name", "Display name",
    "Name of Display as shown with xrandr", std::string("DP-0"));
  result &= registrar->parameter(
    width_, "width", "Width",
    "Width of the stream", kDefaultWidth);
  result &= registrar->parameter(
    height_, "height", "Height",
    "Height of the stream", kDefaultHeight);
  result &= registrar->parameter(
    framerate_, "framerate", "Framerate",
    "Framerate of the stream.", kDefaultFramerate);
  result &= registrar->parameter(
    use_exclusive_display_, "use_exclusive_display", "Use exclusive display",
    "Enable exclusive display", kDefaultExclusiveDisplay);
  return nvidia::gxf::ToResultCode(result);
}

gxf_result_t HolovizViewer::initialize() {
  is_initialized_ = false;

  return GXF_SUCCESS;
}

gxf_result_t HolovizViewer::deinitialize() {
  return GXF_SUCCESS;
}

gxf_result_t HolovizViewer::start() {
  try
  {
    ImGui::CreateContext();
    viz::ImGuiSetCurrentContext(ImGui::GetCurrentContext());
  }
  catch(const std::exception& e)
  {
    GXF_LOG_ERROR(e.what());
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

gxf_result_t HolovizViewer::stop() {
  try
  {
    viz::Shutdown();
  }
  catch(const std::exception& e)
  {
    GXF_LOG_ERROR(e.what());
    return GXF_FAILURE;
  }
  return GXF_SUCCESS;
}

gxf_result_t HolovizViewer::tick() {
  try
  {
    // Process input message
    const auto in_message = receiver_->receive();
    if (!in_message || in_message.value().is_null()) {
      return GXF_CONTRACT_MESSAGE_NOT_AVAILABLE;
    }

    // get tensor attached to message by the name defined in the parameters
    auto maybe_input_image = in_message.value().get<gxf::Tensor>(input_image_name_.get().c_str());
    if (!maybe_input_image) {
      GXF_LOG_ERROR("Failed to retrieve input with name %s", input_image_name_.get().c_str());
      return GXF_FAILURE;
    }

    auto input_image = maybe_input_image.value();

    // get input image metadata
    auto image_shape = input_image->shape();
    auto width = image_shape.dimension(1);
    auto height = image_shape.dimension(0);
    auto channels = image_shape.dimension(2);
    auto element_type = input_image->element_type();
    auto element_size = gxf::PrimitiveTypeSize(element_type);

    if (element_type != gxf::PrimitiveType::kUnsigned8) {
      GXF_LOG_ERROR("Unexpected bytes in element representation %d", element_size);
      return GXF_FAILURE;
    }

    if (!is_initialized_) {
      // set the expected number of channels
      switch (channels) {
        case 3:
          image_format_ = viz::ImageFormat::R8G8B8_UNORM;
          break;
        case 4:
          image_format_ = viz::ImageFormat::R8G8B8A8_UNORM;
          break;
        default:
          GXF_LOG_ERROR("Unexpected number of channels %d", channels);
          return GXF_FAILURE;
      }

      GXF_LOG_INFO("Initializing window with %u x %u image", width_.get(), height_.get());
      GXF_LOG_INFO("Number of channels %d", channels);
      GXF_LOG_INFO("Element type %d (size %d)", element_type, element_size);

      if (use_exclusive_display_) {
        viz::Init(display_name_.get().c_str(), width_, height_, framerate_);
      } else {
        viz::Init(width_, height_, window_title_.get().c_str(),
          viz::InitFlags::FULLSCREEN);
      }

      // set the visualization component as initialized
      is_initialized_ = true;
    }

    // get pointer to tensor buffer
    auto maybe_buffer_ptr = input_image->data<uint8_t>();
    if (!maybe_buffer_ptr) {
      GXF_LOG_ERROR("Could not get pointer to input data");
      return GXF_FAILURE;
    }

    const auto buffer_ptr = maybe_buffer_ptr.value();

    if (viz::WindowIsMinimized()) {
      return GXF_SUCCESS;
    }

    if (viz::WindowShouldClose()) {
      GXF_LOG_INFO("Window closed");
      return GXF_FAILURE;
    }

    viz::Begin();

    // Image
    viz::BeginImageLayer();

    if (input_image->storage_type() == gxf::MemoryStorageType::kDevice) {
      // if it's the device convert to `CUDeviceptr`
      const auto cu_buffer_ptr = reinterpret_cast<CUdeviceptr>(buffer_ptr);
      viz::ImageCudaDevice(width, height, image_format_, cu_buffer_ptr);
    } else {
      // convert to void * if using the system/host
      const auto host_buffer_ptr = reinterpret_cast<void *>(buffer_ptr);
      viz::ImageHost(width, height, image_format_, host_buffer_ptr);
    }
    viz::EndLayer();

    viz::End();

    return GXF_SUCCESS;
  }
  catch(const std::exception& e)
  {
    GXF_LOG_ERROR(e.what());
    return GXF_FAILURE;
  }

}

}
