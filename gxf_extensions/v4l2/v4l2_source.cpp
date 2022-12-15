/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "v4l2_source.hpp"

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <string>
#include <utility>

#include "gxf/core/handle.hpp"
#include "gxf/multimedia/video.hpp"

namespace nvidia {
namespace holoscan {

static constexpr char kDefaultDevice[] = "/dev/video0";
static constexpr uint32_t kDefaultWidth = 640;
static constexpr uint32_t kDefaultHeight = 480;
static constexpr uint32_t kDefaultNumBuffers = 2;

gxf_result_t V4L2Source::registerInterface(gxf::Registrar* registrar) {
  gxf::Expected<void> result;
  result &= registrar->parameter(signal_, "signal", "Output", "Output channel");
  result &= registrar->parameter(allocator_, "allocator", "Allocator", "Output Allocator");
  result &= registrar->parameter(device_, "device", "VideoDevice", "Path to the V4L2 device",
                                 std::string(kDefaultDevice));
  result &=
      registrar->parameter(width_, "width", "Width", "Width of the V4L2 image", kDefaultWidth);
  result &=
      registrar->parameter(height_, "height", "Height", "Height of the V4L2 image", kDefaultHeight);
  result &= registrar->parameter(num_buffers_, "numBuffers", "NumBuffers",
                                 "Number of V4L2 buffers to use", kDefaultNumBuffers);
  return gxf::ToResultCode(result);
}

gxf_result_t V4L2Source::start() {
  // Open the device
  fd_ = open(device_.get().c_str(), O_RDWR);
  if (fd_ < 0) {
    perror("Failed to open device, OPEN");
    return GXF_FAILURE;
  }

  // Ask the device if it can capture frames
  v4l2_capability capability;
  if (ioctl(fd_, VIDIOC_QUERYCAP, &capability) < 0) {
    perror("Failed to get device capabilities, VIDIOC_QUERYCAP");
    return GXF_FAILURE;
  }

  // Get and check the device capabilities
  v4l2_capability caps;
  if (ioctl(fd_, VIDIOC_QUERYCAP, &caps) < 0) {
    GXF_LOG_ERROR("%s is not a v4l2 device.", device_);
    return GXF_FAILURE;
  }
  if (!(caps.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
    GXF_LOG_ERROR("%s is not a video capture device.", device_);
    return GXF_FAILURE;
  }
  if (!(caps.capabilities & V4L2_CAP_STREAMING)) {
    GXF_LOG_ERROR("%s does not support streaming I/O.", device_);
    return GXF_FAILURE;
  }

  // Set image format
  v4l2_format fmt;
  uint32_t pixel_format = V4L2_PIX_FMT_YUYV;
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = width_.get();
  fmt.fmt.pix.height = height_.get();
  fmt.fmt.pix.pixelformat = pixel_format;
  fmt.fmt.pix.field = V4L2_FIELD_NONE;
  if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
    GXF_LOG_ERROR("Failed to set the image format on %s (%dx%d, format = %d)", device_,
                  width_.get(), height_.get(), pixel_format);
    return GXF_FAILURE;
  }
  if (fmt.fmt.pix.width != width_.get() || fmt.fmt.pix.height != height_.get() ||
      fmt.fmt.pix.pixelformat != pixel_format) {
    GXF_LOG_ERROR("Format not supported by %s", device_);
    return GXF_FAILURE;
  }

  // Request buffers from the device
  v4l2_requestbuffers req = {0};
  req.count = num_buffers_.get();
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
    GXF_LOG_ERROR("Could not request buffer from device, VIDIOC_REQBUFS");
    return GXF_FAILURE;
  }

  // Retrieve and map the buffers.
  for (size_t i = 0; i < req.count; i++) {
    v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
      GXF_LOG_ERROR("Failed to query buffer from %s", device_);
      return GXF_FAILURE;
    }

    void* ptr = mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
    if (!ptr) {
      GXF_LOG_ERROR("Failed to map buffer provided by %s", device_);
      return GXF_FAILURE;
    }

    buffers_.push_back(Buffer(ptr, buf.length));
  }

  // Queue all buffers.
  for (size_t i = 0; i < buffers_.size(); i++) {
    v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
      GXF_LOG_ERROR("Failed to queue buffer %d on %s", i, device_);
      return GXF_FAILURE;
    }
  }

  // Start streaming.
  v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
    GXF_LOG_ERROR("Could not start streaming, VIDIOC_STREAMON");
    return GXF_FAILURE;
  }

  return GXF_SUCCESS;
}

gxf_result_t V4L2Source::stop() {
  gxf_result_t result = GXF_SUCCESS;

  // Stop streaming.
  v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (ioctl(fd_, VIDIOC_STREAMOFF, &type) < 0) {
    GXF_LOG_ERROR("Could not end streaming, VIDIOC_STREAMOFF");
    result = GXF_FAILURE;
  }

  // Unmap the buffers.
  for (const auto& buffer : buffers_) {
    if (munmap(buffer.ptr, buffer.length) < 0) {
      GXF_LOG_ERROR("Failed to unmap buffer from %s", device_);
      result = GXF_FAILURE;
    }
  }
  buffers_.clear();

  close(fd_);
  fd_ = -1;

  return result;
}

gxf_result_t V4L2Source::tick() {
  auto message = gxf::Entity::New(context());
  if (!message) {
    GXF_LOG_ERROR("Failed to allocate message");
    return GXF_FAILURE;
  }

  auto rgba_buf = message.value().add<gxf::VideoBuffer>();
  if (!rgba_buf) {
    GXF_LOG_ERROR("Failed to allocate RGBA buffer");
    return GXF_FAILURE;
  }

  // Dequeue the next available buffer.
  v4l2_buffer v4l2_buf = {0};
  v4l2_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  v4l2_buf.memory = V4L2_MEMORY_MMAP;
  if (ioctl(fd_, VIDIOC_DQBUF, &v4l2_buf) < 0) {
    GXF_LOG_ERROR("Failed to dequeue buffer from %s", device_);
    return GXF_FAILURE;
  }
  Buffer& buf = buffers_[v4l2_buf.index];

  // Allocate and convert to an RGBA output buffer.
  rgba_buf.value()->resize<gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA>(
      width_, height_, gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR,
      gxf::MemoryStorageType::kHost, allocator_);
  if (!rgba_buf.value()->pointer()) {
    GXF_LOG_ERROR("Failed to allocate RGBA buffer.");
    return GXF_FAILURE;
  }
  YUYVToRGBA(buf.ptr, rgba_buf.value()->pointer(), width_, height_);

  // Return (queue) the buffer.
  if (ioctl(fd_, VIDIOC_QBUF, &v4l2_buf) < 0) {
    GXF_LOG_ERROR("Failed to queue buffer %d on %s", v4l2_buf.index, device_);
    return GXF_FAILURE;
  }

  const auto result = signal_->publish(std::move(message.value()));

  return gxf::ToResultCode(message);
}

void V4L2Source::YUYVToRGBA(const void* yuyv, void* rgba, size_t width, size_t height) {
  auto r_convert = [](int y, int cr) {
    double r = y + (1.4065 * (cr - 128));
    return static_cast<unsigned int>(std::max(0, std::min(255, static_cast<int>(r))));
  };
  auto g_convert = [](int y, int cb, int cr) {
    double g = y - (0.3455 * (cb - 128)) - (0.7169 * (cr - 128));
    return static_cast<unsigned int>(std::max(0, std::min(255, static_cast<int>(g))));
  };
  auto b_convert = [](int y, int cb) {
    double b = y + (1.7790 * (cb - 128));
    return static_cast<unsigned int>(std::max(0, std::min(255, static_cast<int>(b))));
  };

  const unsigned char* yuyv_buf = static_cast<const unsigned char*>(yuyv);
  unsigned char* rgba_buf = static_cast<unsigned char*>(rgba);

  for (unsigned int i = 0, j = 0; i < width * height * 4; i += 8, j += 4) {
    int cb = yuyv_buf[j + 1];
    int cr = yuyv_buf[j + 3];

    // First pixel
    int y = yuyv_buf[j];
    rgba_buf[i] = r_convert(y, cr);
    rgba_buf[i + 1] = g_convert(y, cb, cr);
    rgba_buf[i + 2] = b_convert(y, cb);
    rgba_buf[i + 3] = 1;

    // Second pixel
    y = yuyv_buf[j + 2];
    rgba_buf[i + 4] = r_convert(y, cr);
    rgba_buf[i + 5] = g_convert(y, cb, cr);
    rgba_buf[i + 6] = b_convert(y, cb);
    rgba_buf[i + 7] = 1;
  }
}

V4L2Source::Buffer::Buffer(void* _ptr, size_t _length) : ptr(_ptr), length(_length) {}

}  // namespace holoscan
}  // namespace nvidia
