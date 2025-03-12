/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/operators/v4l2_video_capture/v4l2_video_capture.hpp"

#include <errno.h>
#include <fcntl.h>
#include <jpeglib.h>
#include <libv4l2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <list>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <magic_enum.hpp>

#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/utils/cuda_macros.hpp"

#define POSIX_CALL(stmt)                                                             \
  ({                                                                                 \
    int _result;                                                                     \
    do { _result = stmt; } while ((_result == -1) && (errno == EINTR));              \
    if (_result == -1) {                                                             \
      throw std::runtime_error(                                                      \
          fmt::format("Call `{}` in line {} of file {} failed with '{}' (errno {})", \
                      #stmt,                                                         \
                      __LINE__,                                                      \
                      __FILE__,                                                      \
                      strerror(errno),                                               \
                      errno));                                                       \
    }                                                                                \
    _result;                                                                         \
  })

/// convert FOURCC value to string
#define FOURCC2STRING(val)              \
  fmt::format("{}{}{}{}",               \
              char(val & 0x7F),         \
              char((val >> 8) & 0x7F),  \
              char((val >> 16) & 0x7F), \
              char((val >> 24) & 0x7F))

namespace holoscan::ops {

/**
 * Holds a nvidia::gxf::VideoFormat and the corresponding function to create the default color plane
 * description
 **/
class GxfFormat {
 public:
  explicit GxfFormat(nvidia::gxf::VideoFormat format) : format_(format) {}
  GxfFormat() = delete;

  virtual ~GxfFormat() = default;

  const nvidia::gxf::VideoFormat format_;
  virtual std::vector<nvidia::gxf::ColorPlane> get_default_color_planes(
      [[maybe_unused]] uint32_t width, [[maybe_unused]] uint32_t height,
      [[maybe_unused]] bool stride_align) const {
    throw std::runtime_error("Unsupported");
  }
  virtual std::tuple<nvidia::gxf::Shape, nvidia::gxf::PrimitiveType> get_shape_and_type(
      [[maybe_unused]] uint32_t width, [[maybe_unused]] uint32_t height) const {
    throw std::runtime_error("Unsupported");
  }
};

/**
 * Template used to get a pointer to the default color plane description for a format.
 */
template <nvidia::gxf::VideoFormat FORMAT>
class GxfFormatTemplate : public GxfFormat {
 public:
  GxfFormatTemplate() : GxfFormat(FORMAT) {}

  std::vector<nvidia::gxf::ColorPlane> get_default_color_planes(uint32_t width, uint32_t height,
                                                                bool stride_align) const override {
    return const_cast<nvidia::gxf::VideoFormatSize<FORMAT>*>(&video_format_size_)
        ->getDefaultColorPlanes(width, height, stride_align);
  }

 private:
  nvidia::gxf::VideoFormatSize<FORMAT> video_format_size_;
};

class GxfFormatTensor : public GxfFormat {
 public:
  explicit GxfFormatTensor(uint32_t v4l2_pixel_format)
      : GxfFormat(nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_CUSTOM),
        v4l2_pixel_format_(v4l2_pixel_format) {}
  GxfFormatTensor() = delete;

  std::tuple<nvidia::gxf::Shape, nvidia::gxf::PrimitiveType> get_shape_and_type(
      uint32_t width, uint32_t height) const override {
    switch (v4l2_pixel_format_) {
      case V4L2_PIX_FMT_YUYV:
        return {{int(height), int(width), 2}, nvidia::gxf::PrimitiveType::kUnsigned8};
      default:
        throw std::runtime_error(
            fmt::format("Unhandled pixel format {}", FOURCC2STRING(v4l2_pixel_format_)));
    }
  }

 private:
  const uint32_t v4l2_pixel_format_;
};

/**
 * Map from a V4L2 format to a nvidia::gxf::VideoFormat
 */
static const V4L2VideoCaptureOp::FormatList v4l2_to_gxf_format{
    {V4L2_PIX_FMT_RGBA32,
     std::shared_ptr<GxfFormat>(
         new GxfFormatTemplate<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA>())},
    {V4L2_PIX_FMT_BGRA32,
     std::shared_ptr<GxfFormat>(
         new GxfFormatTemplate<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_BGRA>())},
    {V4L2_PIX_FMT_ARGB32,
     std::shared_ptr<GxfFormat>(
         new GxfFormatTemplate<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_ARGB>())},
    // caution, unless all other formats where the name matches the memory, the ABGR format is BGRA
    // in memory
    {V4L2_PIX_FMT_ABGR32,
     std::shared_ptr<GxfFormat>(
         new GxfFormatTemplate<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_BGRA>())},
    {V4L2_PIX_FMT_RGBX32,
     std::shared_ptr<GxfFormat>(
         new GxfFormatTemplate<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBX>())},
    {V4L2_PIX_FMT_BGRX32,
     std::shared_ptr<GxfFormat>(
         new GxfFormatTemplate<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_BGRX>())},
    {V4L2_PIX_FMT_XRGB32,
     std::shared_ptr<GxfFormat>(
         new GxfFormatTemplate<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_XRGB>())},
    // caution, unless all other formats where the name matches the memory, the XBGR format is BGRX
    // in memory
    {V4L2_PIX_FMT_XBGR32,
     std::shared_ptr<GxfFormat>(
         new GxfFormatTemplate<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_BGRX>())},
    {V4L2_PIX_FMT_RGB24,
     std::shared_ptr<GxfFormat>(
         new GxfFormatTemplate<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGB>())},
    {V4L2_PIX_FMT_BGR24,
     std::shared_ptr<GxfFormat>(
         new GxfFormatTemplate<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_BGR>())},
    {V4L2_PIX_FMT_GREY,
     std::shared_ptr<GxfFormat>(
         new GxfFormatTemplate<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY>())},
    {V4L2_PIX_FMT_Y16,
     std::shared_ptr<GxfFormat>(
         new GxfFormatTemplate<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_GRAY16>())},
    {V4L2_PIX_FMT_NV12,
     std::shared_ptr<GxfFormat>(
         new GxfFormatTemplate<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV12>())},
    {V4L2_PIX_FMT_YUV420,
     std::shared_ptr<GxfFormat>(
         new GxfFormatTemplate<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_YUV420>())},
    {V4L2_PIX_FMT_NV24,
     std::shared_ptr<GxfFormat>(
         new GxfFormatTemplate<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_NV24>())},
    // these V4L2 formats have no equivalent GXF video buffer format, output as tensor instead
    {V4L2_PIX_FMT_YUYV, std::shared_ptr<GxfFormat>(new GxfFormatTensor(V4L2_PIX_FMT_YUYV))},
    {V4L2_PIX_FMT_MJPEG, nullptr},
};

static void YUYVToRGBA(const void* yuyv, void* rgba, size_t width, size_t height) {
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
    rgba_buf[i + 3] = 255;

    // Second pixel
    y = yuyv_buf[j + 2];
    rgba_buf[i + 4] = r_convert(y, cr);
    rgba_buf[i + 5] = g_convert(y, cb, cr);
    rgba_buf[i + 6] = b_convert(y, cb);
    rgba_buf[i + 7] = 255;
  }
}

// Support for RGB24 format (`RGB3` in 4CC code)
// For every pixel, add alpha channel to get RGBA
static void RGB24ToRGBA(const void* rgb3, void* rgba, size_t width, size_t height) {
  const unsigned char* rgb3_buf = static_cast<const unsigned char*>(rgb3);
  unsigned char* rgba_buf = static_cast<unsigned char*>(rgba);

  for (unsigned int i = 0, j = 0; i < width * height * 3; i += 3, j += 4) {
    rgba_buf[j] = rgb3_buf[i];
    rgba_buf[j + 1] = rgb3_buf[i + 1];
    rgba_buf[j + 2] = rgb3_buf[i + 2];
    rgba_buf[j + 3] = 255;
  }
}

// Support for MJPEG format
// Each frame is a JPEG image so use libjpeg to decompress the image and modify it to
// add alpha channel
static void MJPEGToRGBA(const void* mjpg, void* rgba, size_t width, size_t height) {
  jpeg_decompress_struct cinfo;
  jpeg_error_mgr jerr;
  // Size of image is width * height * 3 (RGB)
  unsigned long jpg_size = width * height * 3;
  int row_stride;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);

  const unsigned char* src_buf =
      const_cast<unsigned char*>(static_cast<const unsigned char*>(mjpg));
  unsigned char* dest_buf = static_cast<unsigned char*>(rgba);
  jpeg_mem_src(&cinfo, src_buf, jpg_size);
  int rc = jpeg_read_header(&cinfo, TRUE);

  if (rc != 1) { throw std::runtime_error("Failed to read jpeg header"); }

  jpeg_start_decompress(&cinfo);

  // Each row has width * 4 pixels (RGBA)
  row_stride = width * 4;

  while (cinfo.output_scanline < cinfo.output_height) {
    unsigned char* buffer_array[1];
    buffer_array[0] = dest_buf + (cinfo.output_scanline) * row_stride;
    // Decompress jpeg image and write it to buffer_arary
    jpeg_read_scanlines(&cinfo, buffer_array, 1);
    unsigned char* buf = buffer_array[0];
    // Modify image to add alpha channel with values set to 255
    // start from the end so we don't overwrite existing values
    for (int i = (int)width * 3 - 1, j = row_stride - 1; i > 0; i -= 3, j -= 4) {
      buf[j] = 255;
      buf[j - 1] = buf[i];
      buf[j - 2] = buf[i - 1];
      buf[j - 3] = buf[i - 2];
    }
  }
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
}

void V4L2VideoCaptureOp::setup(OperatorSpec& spec) {
  spec.output<std::shared_ptr<holoscan::gxf::Entity>>("signal");

  static constexpr char kDefaultDevice[] = "/dev/video0";
  static constexpr char kDefaultPixelFormat[] = "auto";
  static constexpr bool kDefaultPassThrough = false;
  static constexpr uint32_t kDefaultWidth = 0;
  static constexpr uint32_t kDefaultHeight = 0;
  static constexpr float kDefaultFrameRate = 0.f;
  static constexpr uint32_t kDefaultNumBuffers = 4;

  spec.param(allocator_,
             "allocator",
             "Allocator",
             "Deprecated. Memory allocator to use for the output if `pass_through` is `false`.");

  spec.param(
      device_, "device", "VideoDevice", "Path to the V4L2 device", std::string(kDefaultDevice));
  spec.param(width_, "width", "Width", "Width of the V4L2 image", kDefaultWidth);
  spec.param(height_, "height", "Height", "Height of the V4L2 image", kDefaultHeight);
  spec.param(frame_rate_, "frame_rate", "Frame rate", "Capture frame rate", kDefaultFrameRate);
  spec.param(num_buffers_,
             "numBuffers",
             "NumBuffers",
             "Number of V4L2 buffers to use",
             kDefaultNumBuffers);
  spec.param(pixel_format_,
             "pixel_format",
             "Pixel Format",
             "Pixel format of capture stream (little endian four character code (fourcc))",
             std::string(kDefaultPixelFormat));
  spec.param(
      pass_through_,
      "pass_through",
      "Pass Through",
      "Deprecated, use `FormatConverterOp` to convert the output buffer to the desired format. If "
      "set, pass through the input buffer to the output unmodified, else convert to "
      "RGBA32.",
      kDefaultPassThrough);
  spec.param(exposure_time_,
             "exposure_time",
             "Exposure Time",
             "Exposure time of the camera sensor in multiples of 100 μs (e.g. setting "
             "exposure_time to 100 is 10 ms). See V4L2_CID_EXPOSURE_ABSOLUTE.",
             ParameterFlag::kOptional);
  spec.param(gain_,
             "gain",
             "Gain",
             "Gain of the camera sensor. See V4L2_CID_GAIN.",
             ParameterFlag::kOptional);
}

void V4L2VideoCaptureOp::initialize() {
  Operator::initialize();
}

void V4L2VideoCaptureOp::start() {
  v4l2_initialize();
  v4l2_get_format();
  v4l2_check_formats();
  v4l2_set_format();
  v4l2_set_camera_settings();
  v4l2_request_buffers();
  v4l2_start();
}

void V4L2VideoCaptureOp::compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
                                 ExecutionContext& context) {
  // Read buffer.
  v4l2_buffer buf{};
  v4l2_read_buffer(buf);

  // Create tensor or video buffer
  auto out_message = nvidia::gxf::Entity::New(context.context());
  if (!out_message) { throw std::runtime_error("Failed to allocate video output; terminating."); }

  if (pass_through_ && ((v4l2_to_gxf_format_ == v4l2_to_gxf_format.end()) ||
                        (v4l2_to_gxf_format_->second->format_ ==
                         nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_CUSTOM))) {
    // if pass through is enabled and this is a custom format (might be compressed
    // (e.g. MJPEG, H264) or YUYV (common for USB cameras)), output as Tensor
    nvidia::gxf::Expected<nvidia::gxf::Handle<nvidia::gxf::Tensor>> tensor =
        out_message.value().add<nvidia::gxf::Tensor>();
    if (!tensor) { throw std::runtime_error("Failed to allocate tensor; terminating."); }

    nvidia::gxf::Shape shape;
    nvidia::gxf::PrimitiveType element_type;
    if ((format_desc_.flags & V4L2_FMT_FLAG_COMPRESSED) ||
        (v4l2_to_gxf_format_ == v4l2_to_gxf_format.end())) {
      // otuput compressed or unknown formats as a simple memory blob
      shape = {int(buf.length)};
      element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
    } else {
      std::tie(shape, element_type) = v4l2_to_gxf_format_->second->get_shape_and_type(
          format_.fmt.pix.width, format_.fmt.pix.height);
    }
    tensor.value()->wrapMemory(shape,
                               element_type,
                               nvidia::gxf::PrimitiveTypeSize(element_type),
                               nvidia::gxf::Unexpected{GXF_UNINITIALIZED_VALUE},
                               memory_storage_type_,
                               buffers_[buf.index].ptr,
                               [buffer = buf, fd = fd_](void*) mutable {
                                 POSIX_CALL(ioctl(fd, VIDIOC_QBUF, &buffer));
                                 return nvidia::gxf::Success;
                               });
  } else {
    nvidia::gxf::Expected<nvidia::gxf::Handle<nvidia::gxf::VideoBuffer>> video_buffer =
        out_message.value().add<nvidia::gxf::VideoBuffer>();
    if (!video_buffer) {
      throw std::runtime_error("Failed to allocate video buffer; terminating.");
    }

    if (converter_) {
      if (!allocator_.get()) {
        throw std::runtime_error("An allocator is required when converting to RGBA.");
      }
      // Get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
      auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
                                                                           allocator_->gxf_cid());
      video_buffer.value()->resize<nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA>(
          format_.fmt.pix.width,
          format_.fmt.pix.height,
          nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR,
          nvidia::gxf::MemoryStorageType::kHost,
          allocator.value(),
          false);
      if (!video_buffer.value()->pointer()) {
        throw std::runtime_error("Failed to allocate output buffer.");
      }

      // Convert to RGBA output buffer
      (*converter_)(buffers_[buf.index].ptr,
                    video_buffer.value()->pointer(),
                    format_.fmt.pix.width,
                    format_.fmt.pix.height);

      // Return (queue) the buffer.
      POSIX_CALL(ioctl(fd_, VIDIOC_QBUF, &buf));
    } else {
      // Wrap memory into output buffer
      nvidia::gxf::VideoBufferInfo video_buffer_info{};
      video_buffer_info.width = format_.fmt.pix.width;
      video_buffer_info.height = format_.fmt.pix.height;
      video_buffer_info.color_format = v4l2_to_gxf_format_->second->format_;
      video_buffer_info.color_planes = v4l2_to_gxf_format_->second->get_default_color_planes(
          video_buffer_info.width, video_buffer_info.height, false /*stride_align*/);
      video_buffer_info.surface_layout =
          nvidia::gxf::SurfaceLayout::GXF_SURFACE_LAYOUT_PITCH_LINEAR;

      video_buffer.value()->wrapMemory(video_buffer_info,
                                       buf.length,
                                       memory_storage_type_,
                                       buffers_[buf.index].ptr,
                                       [buffer = buf, fd = fd_](void*) mutable {
                                         POSIX_CALL(ioctl(fd, VIDIOC_QBUF, &buffer));
                                         return nvidia::gxf::Success;
                                       });
    }
  }

  // add metadata if enabled
  if (is_metadata_enabled()) {
    auto meta = metadata();
    meta->set("V4L2_pixel_format", FOURCC2STRING(format_desc_.pixelformat));
    meta->set(
        "V4L2_ycbcr_encoding",
        std::string(magic_enum::enum_name((enum v4l2_ycbcr_encoding)(format_.fmt.pix.ycbcr_enc))));
    meta->set(
        "V4L2_quantization",
        std::string(magic_enum::enum_name((enum v4l2_quantization)(format_.fmt.pix.quantization))));
  }

  auto result = gxf::Entity(std::move(out_message.value()));
  op_output.emit(result, "signal");
}

void V4L2VideoCaptureOp::stop() {
  // stream off
  enum v4l2_buf_type buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  POSIX_CALL(ioctl(fd_, VIDIOC_STREAMOFF, &buf_type));

  // free buffers
  for (uint32_t i = 0; i < buffers_.size(); ++i) {
    switch (memory_storage_type_) {
      case nvidia::gxf::MemoryStorageType::kHost:
        HOLOSCAN_CUDA_CALL(cudaFreeHost(buffers_[i].ptr));
        break;
      case nvidia::gxf::MemoryStorageType::kDevice:
        HOLOSCAN_CUDA_CALL(cudaFree(buffers_[i].ptr));
        break;
      case nvidia::gxf::MemoryStorageType::kSystem:
        POSIX_CALL(munmap(buffers_[i].ptr, buffers_[i].length));
        break;
      default:
        throw std::runtime_error("Unhandled memory type");
    }
  }
  buffers_.clear();

  // close FD
  POSIX_CALL(v4l2_close(fd_));
  fd_ = -1;
}

void V4L2VideoCaptureOp::v4l2_initialize() {
  // Initialise V4L2 device
  fd_ = v4l2_open(device_.get().c_str(), O_RDWR | O_NONBLOCK);
  if (fd_ < 0) {
    throw std::runtime_error(
        "Failed to open device! Possible permission issue with accessing the device.");
  }

  v4l2_capability caps{};
  POSIX_CALL(ioctl(fd_, VIDIOC_QUERYCAP, &caps));
  if (!(caps.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
    throw std::runtime_error("No V4l2 Video capture node");
  }
  if (!(caps.capabilities & V4L2_CAP_STREAMING)) {
    throw std::runtime_error("Does not support streaming i/o");
  }
}

void V4L2VideoCaptureOp::v4l2_request_buffers() {
  v4l2_requestbuffers req{};

  // call with a count set to 0 to query capabilities
  req.count = 0;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;

  POSIX_CALL(ioctl(fd_, VIDIOC_REQBUFS, &req));

  capture_memory_method_ = V4L2_MEMORY_MMAP;
  memory_storage_type_ = nvidia::gxf::MemoryStorageType::kSystem;
  if (converter_) {
    // the converters use CPU memory, therefore capture to `kSystem` memory (which is cached CPU
    // memory)
    HOLOSCAN_LOG_WARN(
        "Deprecation warning. Converting the input stream from {} to RGBA using a CPU based "
        "converter is deprecated. Please set `pass_through` to `true` and use the "
        "`FormatConverterOp` to convert the image data.",
        FOURCC2STRING(format_desc_.pixelformat));
  } else if (req.capabilities & V4L2_BUF_CAP_SUPPORTS_USERPTR) {
    capture_memory_method_ = V4L2_MEMORY_USERPTR;

    // select memory type, start with pinned memory which has good write performance for the V4L2
    // driver and can be access by CUDA (at PCIe performance)
    memory_storage_type_ = nvidia::gxf::MemoryStorageType::kHost;

    // when running on L4T and managed memory is supported, then output is managed memory
    // which has the best combination of CPU and GPU access performance
    if (std::filesystem::exists("/etc/nv_tegra_release")) {
      const int device = 0;
      int managed_memory = 0;
      HOLOSCAN_CUDA_CALL(cudaDeviceGetAttribute(&managed_memory, cudaDevAttrManagedMemory, device));
      // Workaround: frames are not updated when running bare metal on IGX with dGPU so enabled
      // this on iGPU only for now
      int integrated = 0;
      HOLOSCAN_CUDA_CALL(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device));
      if (managed_memory && integrated) {
          memory_storage_type_ = nvidia::gxf::MemoryStorageType::kDevice;
      }
    }
  }

  // Request V4L2 buffers
  req.count = num_buffers_.get();
  req.memory = capture_memory_method_;
  POSIX_CALL(ioctl(fd_, VIDIOC_REQBUFS, &req));

  buffers_.resize(req.count);

  for (uint32_t i = 0; i < req.count; ++i) {
    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = req.memory;
    buf.index = i;

    POSIX_CALL(ioctl(fd_, VIDIOC_QUERYBUF, &buf));

    buffers_[i].length = buf.length;
    switch (memory_storage_type_) {
      case nvidia::gxf::MemoryStorageType::kHost:
        HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMallocHost(&buffers_[i].ptr, buf.length),
                                       "Failed to allocate CUDA host memory");
        break;
      case nvidia::gxf::MemoryStorageType::kDevice:
        HOLOSCAN_CUDA_CALL_THROW_ERROR(
            cudaMallocManaged(&buffers_[i].ptr, buf.length, cudaMemAttachGlobal),
            "Failed to allocate CUDA malloced memory");
        break;
      case nvidia::gxf::MemoryStorageType::kSystem:
        buffers_[i].ptr =
            mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
        if (MAP_FAILED == buffers_[i].ptr) { throw std::runtime_error("MMAP failed"); }
        break;
      default:
        throw std::runtime_error("Unhandled memory type");
    }
  }
}

void V4L2VideoCaptureOp::v4l2_check_formats() {
  // Check user-given pixel format (also OK if "auto")
  if (pixel_format_.get().length() != 4) {
    throw std::runtime_error(
        fmt::format("User-given pixel format should have V4L2 FourCC string format, got {}",
                    pixel_format_.get()));
  }

  // get all the supported formats
  std::list<v4l2_fmtdesc> v4l2_formats;
  v4l2_fmtdesc fmtdesc{};
  fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  while (ioctl(fd_, VIDIOC_ENUM_FMT, &fmtdesc) == 0) {
    v4l2_formats.push_back(fmtdesc);
    fmtdesc.index++;
  }

  // Before release 35 the HDMI IN capture driver reports `V4L2_PIX_FMT_ABGR32` (BGRA32) although it
  // outputs `V4L2_PIX_FMT_RGBA32` (RGBA32). Fix this by reporting `V4L2_PIX_FMT_RGBA32` instead of
  // `V4L2_PIX_FMT_ABGR32`.
  auto patched_format = v4l2_formats.end();
  v4l2_capability caps{};
  POSIX_CALL(ioctl(fd_, VIDIOC_QUERYCAP, &caps));
  const std::string busInfo = reinterpret_cast<const char*>(caps.bus_info);
  if (busInfo.find("tegra-capture-vi") != std::string::npos) {
    std::ifstream tegra_release_file("/etc/nv_tegra_release");
    if (tegra_release_file.is_open()) {
      // file format is '# R??', where `??` is the version
      tegra_release_file.ignore(3, 'R');
      uint32_t version;
      tegra_release_file >> version;
      if (!tegra_release_file.eof() && !tegra_release_file.bad()) {
        if (version < 35) {
          auto it = std::find_if(
              v4l2_formats.begin(), v4l2_formats.end(), [](const v4l2_fmtdesc& format_desc) {
                return format_desc.pixelformat == V4L2_PIX_FMT_ABGR32;
              });
          if (it != v4l2_formats.end()) {
            HOLOSCAN_LOG_INFO(
                "Detected HDMI IN source on L4T release < 35, replacing pixel format "
                "`V4L2_PIX_FMT_ABGR32` with `V4L2_PIX_FMT_RGBA32`.");
            patched_format = it;
            it->pixelformat = V4L2_PIX_FMT_RGBA32;
            std::strncpy(reinterpret_cast<char*>(it->description),
                         "32-bit RGBA 8-8-8-8",
                         sizeof(v4l2_fmtdesc::description));
          }
        }
      }
    }
  }

  if (pixel_format_.get() == "auto") {
    FormatList::const_iterator it = v4l2_to_gxf_format.begin();
    while (it != v4l2_to_gxf_format.end()) {
      auto v4l_format = std::find_if(v4l2_formats.cbegin(),
                                     v4l2_formats.cend(),
                                     [pixel_format = it->first](const v4l2_fmtdesc& format_desc) {
                                       return pixel_format == format_desc.pixelformat;
                                     });
      if (v4l_format != v4l2_formats.end()) {
        format_desc_ = *v4l_format;
        format_.fmt.pix.pixelformat = format_desc_.pixelformat;

        if (v4l_format == patched_format) {
          // if we patched this format use the original reported format to be passed to IOCTLs
          format_.fmt.pix.pixelformat = V4L2_PIX_FMT_ABGR32;
        }
        break;
      }
      ++it;
    }

    if (it == v4l2_to_gxf_format.end()) {
      std::vector<std::string> supported_formats;
      for (auto&& format : v4l2_to_gxf_format) {
        supported_formats.push_back(FOURCC2STRING(format.first));
      }
      throw std::runtime_error(
          fmt::format("Automatic setting of pixel format failed: device does not support any of {}",
                      fmt::join(supported_formats, ", ")));
    }
    v4l2_to_gxf_format_ = it;
  } else {
    // Convert user given format to FourCC
    const uint32_t pixel_format = v4l2_fourcc(pixel_format_.get()[0],
                                              pixel_format_.get()[1],
                                              pixel_format_.get()[2],
                                              pixel_format_.get()[3]);

    // Check if the device supports the requested pixel format
    auto v4l_format = std::find_if(v4l2_formats.cbegin(),
                                   v4l2_formats.cend(),
                                   [pixel_format](const v4l2_fmtdesc& format_desc) {
                                     return pixel_format == format_desc.pixelformat;
                                   });
    if (v4l_format == v4l2_formats.end()) {
      throw std::runtime_error(
          fmt::format("Device does not support '{}' pixel format", pixel_format_.get()));
    }

    format_desc_ = *v4l_format;
    format_.fmt.pix.pixelformat = format_desc_.pixelformat;

    if (v4l_format == patched_format) {
      // if we patched this format use the original reported format to be passed into IOCTLs
      format_.fmt.pix.pixelformat = V4L2_PIX_FMT_ABGR32;
    }

    // Update format with valid user-given format
    v4l2_to_gxf_format_ =
        std::find_if(v4l2_to_gxf_format.cbegin(),
                     v4l2_to_gxf_format.cend(),
                     [pixel_format = format_desc_.pixelformat](const FormatListItem& item) {
                       return item.first == pixel_format;
                     });
  }

  if (!pass_through_) {
    if (format_desc_.pixelformat == V4L2_PIX_FMT_YUYV) {
      converter_ = &YUYVToRGBA;
    } else if (format_desc_.pixelformat == V4L2_PIX_FMT_RGB24) {
      converter_ = &RGB24ToRGBA;
    } else if (format_desc_.pixelformat == V4L2_PIX_FMT_MJPEG) {
      converter_ = &MJPEGToRGBA;
    } else if (format_desc_.pixelformat != V4L2_PIX_FMT_RGBA32) {
      throw std::runtime_error(
          fmt::format("Unsupported pixel format {}", FOURCC2STRING(format_desc_.pixelformat)));
    }
  }

  if ((width_.get() > 0) || (height_.get() > 0)) {
    // Check if the device supports the requested width and height
    v4l2_frmsizeenum frmsize{};
    frmsize.pixel_format = format_.fmt.pix.pixelformat;
    int supported_formats = 0;
    while (ioctl(fd_, VIDIOC_ENUM_FRAMESIZES, &frmsize) == 0) {
      supported_formats = 0;
      if (frmsize.type != V4L2_FRMSIZE_TYPE_DISCRETE) {
        throw std::runtime_error("Non-discrete frame sizes not supported");
      }
      if ((width_.get() == 0) || (frmsize.discrete.width == width_.get())) {
        supported_formats += 1;
      }
      if ((height_.get() == 0) || (frmsize.discrete.height == height_.get())) {
        supported_formats += 1;
      }
      if (supported_formats == 2) { break; }
      frmsize.index++;
    }
    if (supported_formats != 2) {
      throw std::runtime_error(
          fmt::format("Device does not support '{}x{}'", width_.get(), height_.get()));
    }
    // Update format with valid user-given format
    if (width_.get() > 0) {
      format_.fmt.pix.width = width_.get();
    } else {
      format_.fmt.pix.width = frmsize.discrete.width;
    }
    if (height_.get() > 0) {
      format_.fmt.pix.height = height_.get();
    } else {
      format_.fmt.pix.height = frmsize.discrete.height;
    }
  }

  v4l2_frmivalenum frmival{};

  frmival.pixel_format = format_.fmt.pix.pixelformat;
  frmival.width = format_.fmt.pix.width;
  frmival.height = format_.fmt.pix.height;

  bool found = false;
  v4l2_frmivalenum best_match{};

  while (ioctl(fd_, VIDIOC_ENUM_FRAMEINTERVALS, &frmival) == 0) {
    if (frmival.type != V4L2_FRMIVAL_TYPE_DISCRETE) {
      throw std::runtime_error("Non-discrete frame intervals not supported");
    }

    // if setting the frame rate is not supported, take the first reported
    if (!supports_frame_rate_) {
      frame_rate_denominator_use_ = frmival.discrete.denominator;
      frame_rate_numerator_use_ = frmival.discrete.numerator;
      break;
    }

    // if the user requests a frame rate, check if the current frame interval is a better match
    if (frame_rate_.get() != 0.f) {
      if (!found || (std::abs(frame_rate_.get() - float(frmival.discrete.denominator) /
                                                      float(frmival.discrete.numerator)) <
                     std::abs(frame_rate_.get() - float(best_match.discrete.denominator) /
                                                      float(best_match.discrete.numerator)))) {
        best_match = frmival;
        found = true;
      }
    }

    ++frmival.index;
  }
  if (frame_rate_.get() != 0.f) {
    if (!found) {
      // this would only happen if the device is not supporting any frame interval
      throw std::runtime_error(
          fmt::format("Device does not support frame rate '{}'", frame_rate_.get()));
    }

    frame_rate_denominator_use_ = best_match.discrete.denominator;
    frame_rate_numerator_use_ = best_match.discrete.numerator;

    if (frame_rate_.get() !=
        float(best_match.discrete.denominator) / float(best_match.discrete.numerator)) {
      HOLOSCAN_LOG_INFO(
          "Device does not support exact frame rate '{}', using nearest frame rate '{}' instead",
          frame_rate_.get(),
          float(frame_rate_denominator_use_) / float(frame_rate_numerator_use_));
    }
  }

  HOLOSCAN_LOG_INFO("Using V4L2 format {} ({}), {}x{}, {} fps",
                    FOURCC2STRING(format_desc_.pixelformat),
                    reinterpret_cast<char*>(format_desc_.description),
                    format_.fmt.pix.width,
                    format_.fmt.pix.height,
                    float(frame_rate_denominator_use_) / float(frame_rate_numerator_use_));
}

void V4L2VideoCaptureOp::v4l2_get_format() {
  // Get the current format
  format_.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  POSIX_CALL(ioctl(fd_, VIDIOC_G_FMT, &format_));

  v4l2_streamparm streamparm{};
  streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  // VIDIOC_G_PARM might not be supported
  if (ioctl(fd_, VIDIOC_G_PARM, &streamparm) == 0) {
    supports_frame_rate_ =
        ((streamparm.parm.capture.capability & V4L2_CAP_TIMEPERFRAME) != 0);  // codespell-ignore
    if (supports_frame_rate_) {
      frame_rate_denominator_use_ =
          streamparm.parm.capture.timeperframe.denominator;  // codespell-ignore
      frame_rate_numerator_use_ =
          streamparm.parm.capture.timeperframe.numerator;  // codespell-ignore
    } else if (frame_rate_.get() > 0) {
      throw std::runtime_error("The device does not support setting the frame rate.");
    }
  }
}

void V4L2VideoCaptureOp::v4l2_set_format() {
  // Set format
  POSIX_CALL(ioctl(fd_, VIDIOC_S_FMT, &format_));

  if (supports_frame_rate_) {
    // Get the current stream parameter
    v4l2_streamparm streamparm{};
    streamparm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    POSIX_CALL(ioctl(fd_, VIDIOC_G_PARM, &streamparm));

    streamparm.parm.capture.timeperframe.denominator =  // codespell-ignore
        frame_rate_denominator_use_;
    streamparm.parm.capture.timeperframe.numerator = frame_rate_numerator_use_;  // codespell-ignore

    // Set the stream parameters
    POSIX_CALL(ioctl(fd_, VIDIOC_S_PARM, &streamparm));
  }
}

bool V4L2VideoCaptureOp::v4l2_camera_supports_control(int cid, const char* control_name) {
  v4l2_queryctrl queryctrl;

  memset(&queryctrl, 0, sizeof(queryctrl));
  queryctrl.id = cid;

  if (ioctl(fd_, VIDIOC_QUERYCTRL, &queryctrl) == -1) {
    // EINVAL indicates that the control is not supported
    if (errno == EINVAL) {
      return false;
    } else {
      throw std::runtime_error(fmt::format("Querying {} failed", control_name));
    }
  }

  return true;
}

// Set the device's control settings if supported
void V4L2VideoCaptureOp::v4l2_set_camera_control(v4l2_control control, const char* control_name,
                                                 bool warn) {
  HOLOSCAN_LOG_DEBUG("Setting {} to {}", control_name, control.value);
  if (v4l2_camera_supports_control(control.id, control_name)) {
    POSIX_CALL(ioctl(fd_, VIDIOC_S_CTRL, &control));
  } else {
    HOLOSCAN_LOG_CALL(
        warn ? LogLevel::WARN : LogLevel::DEBUG, "Device does not support {}", control_name);
  }
}

void V4L2VideoCaptureOp::v4l2_set_camera_settings() {
  v4l2_capability caps{};

  // To check if the dev is v4l2loopback
  POSIX_CALL(ioctl(fd_, VIDIOC_QUERYCAP, &caps));
  std::string busInfo = reinterpret_cast<const char*>(caps.bus_info);
  if (busInfo.find("v4l2loopback") != std::string::npos) {
    // Return before setting the camera parameters as loopback option
    // does not have camera settings to run.
    HOLOSCAN_LOG_DEBUG("Found a v4l2loopback device");
    return;
  }

  v4l2_control control{};

  // Set Exposure
  if (exposure_time_.try_get().has_value()) {
    // Manual exposure: try EXPOSURE_SHUTTER_PRIORITY first (manual exposure, auto iris)
    control.id = V4L2_CID_EXPOSURE_AUTO;
    control.value = V4L2_EXPOSURE_SHUTTER_PRIORITY;
    try {
      v4l2_set_camera_control(control, "V4L2_CID_EXPOSURE_AUTO", false);
    } catch (std::exception& e) {
      // If fails, try setting to full manual mode
      control.value = V4L2_EXPOSURE_MANUAL;
      v4l2_set_camera_control(control, "V4L2_CID_EXPOSURE_AUTO", true);
    }
    // Then set the value
    control.id = V4L2_CID_EXPOSURE_ABSOLUTE;
    control.value = exposure_time_;
    v4l2_set_camera_control(control, "V4L2_CID_EXPOSURE_ABSOLUTE", true);
  } else {
    // Auto exposure: try fully auto first (auto exposure, auto iris)
    control.id = V4L2_CID_EXPOSURE_AUTO;
    control.value = V4L2_EXPOSURE_AUTO;
    try {
      v4l2_set_camera_control(control, "V4L2_CID_EXPOSURE_AUTO", false);
    } catch (std::exception& e) {
      // If fails, try setting to APERTURE_PRIORITY (auto exposure, manual iris)
      control.value = V4L2_EXPOSURE_APERTURE_PRIORITY;
      v4l2_set_camera_control(control, "V4L2_CID_EXPOSURE_AUTO", false);
    }
  }

  // Set Gain
  if (gain_.try_get().has_value()) {
    // Manual: turn auto gain off
    control.id = V4L2_CID_AUTOGAIN;
    control.value = 0;
    v4l2_set_camera_control(control, "V4L2_CID_AUTOGAIN", false);

    // Then set value
    control.id = V4L2_CID_GAIN;
    control.value = gain_;
    v4l2_set_camera_control(control, "V4L2_CID_GAIN", true);
  } else {
    // Auto gain
    control.id = V4L2_CID_AUTOGAIN;
    control.value = 1;
    v4l2_set_camera_control(control, "V4L2_CID_AUTOGAIN", false);
  }
}

void V4L2VideoCaptureOp::v4l2_start() {
  // Start streaming on V4L2 device
  // queue capture plane into device
  for (uint32_t i = 0; i < buffers_.size(); i++) {
    v4l2_buffer buf{};

    buf.index = i;
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = capture_memory_method_;
    if (buf.memory == V4L2_MEMORY_USERPTR) {
      buf.m.userptr = reinterpret_cast<unsigned long>(buffers_[i].ptr);
      buf.length = buffers_[i].length;
    }

    POSIX_CALL(ioctl(fd_, VIDIOC_QBUF, &buf));
  }

  enum v4l2_buf_type buf_type;
  buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  POSIX_CALL(ioctl(fd_, VIDIOC_STREAMON, &buf_type));
}

void V4L2VideoCaptureOp::v4l2_read_buffer(v4l2_buffer& buf) {
  fd_set fds;
  FD_ZERO(&fds);
  FD_SET(fd_, &fds);

  timeval tv;
  tv.tv_sec = 15;
  tv.tv_usec = 0;

  int result;
  do {
    if (POSIX_CALL(select(fd_ + 1, &fds, NULL, NULL, &tv)) == 0) {
      throw std::runtime_error("Querying file descriptor timed out");
    }

    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = capture_memory_method_;
    do { result = ioctl(fd_, VIDIOC_DQBUF, &buf); } while ((result == -1) && (errno == EINTR));
    if (result == -1) {
      if (errno == EAGAIN) { continue; }
      throw std::runtime_error(
          fmt::format("VIDIOC_DQBUF failed with `{}` (errno {})", strerror(errno), errno));
    }

    if (buf.index >= buffers_.size()) {
      throw std::runtime_error(
          fmt::format("Buf index is {} more than the queue size {}", buf.index, buffers_.size()));
    }
  } while (result != 0);
}

}  // namespace holoscan::ops
