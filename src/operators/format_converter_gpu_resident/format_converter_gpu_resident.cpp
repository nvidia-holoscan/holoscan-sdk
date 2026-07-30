/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "holoscan/operators/format_converter_gpu_resident/format_converter_gpu_resident.hpp"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/logger/logger.hpp"
#include "holoscan/operators/format_converter/format_converter_common.hpp"
#include "holoscan/utils/cuda_macros.hpp"

namespace holoscan::ops {

namespace {

// I420 (YUV420 planar) layout with no stride padding; even width/height. Matches the default
// layout assumptions used by the GPU-resident operator when no explicit plane metadata is passed.
std::vector<FormatConverterPlaneLayout> yuv420I420PlanesLayout(int32_t rows, int32_t cols) {
  const uint32_t width_even = static_cast<uint32_t>(cols) + (static_cast<uint32_t>(cols) & 1u);
  const uint32_t height_even = static_cast<uint32_t>(rows) + (static_cast<uint32_t>(rows) & 1u);
  const uint32_t y_stride = width_even;
  const uint32_t uv_stride = y_stride / 2;
  const uint32_t half_h = height_even / 2;

  std::vector<FormatConverterPlaneLayout> planes(3);
  planes[0].stride_bytes = static_cast<int32_t>(y_stride);
  planes[0].byte_offset = 0;
  planes[0].plane_bytes = static_cast<uint64_t>(y_stride) * height_even;

  uint64_t next_offset = planes[0].plane_bytes;
  planes[1].stride_bytes = static_cast<int32_t>(uv_stride);
  if (next_offset > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::runtime_error(
        "FormatConverterGpuResidentOp: I420 plane layout offset overflow (frame too large).");
  }
  planes[1].byte_offset = static_cast<uint32_t>(next_offset);
  planes[1].plane_bytes = static_cast<uint64_t>(uv_stride) * half_h;

  next_offset += planes[1].plane_bytes;
  planes[2].stride_bytes = static_cast<int32_t>(uv_stride);
  if (next_offset > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::runtime_error(
        "FormatConverterGpuResidentOp: I420 plane layout offset overflow (frame too large).");
  }
  planes[2].byte_offset = static_cast<uint32_t>(next_offset);
  planes[2].plane_bytes = static_cast<uint64_t>(uv_stride) * half_h;

  return planes;
}

std::vector<FormatConverterPlaneLayout> nv12PlanesTightPacked(int32_t rows, int32_t cols) {
  std::vector<FormatConverterPlaneLayout> planes(2);
  planes[0].stride_bytes = cols;
  planes[0].byte_offset = 0;
  planes[0].plane_bytes = static_cast<uint64_t>(rows) * static_cast<uint64_t>(cols);

  planes[1].stride_bytes = cols;
  if (planes[0].plane_bytes > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::runtime_error(
        "FormatConverterGpuResidentOp: NV12 plane layout offset overflow (frame too large).");
  }
  planes[1].byte_offset = static_cast<uint32_t>(planes[0].plane_bytes);
  planes[1].plane_bytes = static_cast<uint64_t>(rows / 2) * static_cast<uint64_t>(cols);
  return planes;
}

std::vector<FormatConverterPlaneLayout> packedPlaneLayout(int32_t rows, int32_t cols,
                                                          int16_t channels, uint32_t element_size) {
  std::vector<FormatConverterPlaneLayout> planes(1);
  planes[0].stride_bytes = cols * channels * static_cast<int32_t>(element_size);
  planes[0].byte_offset = 0;
  planes[0].plane_bytes =
      static_cast<uint64_t>(rows) * static_cast<uint64_t>(planes[0].stride_bytes);
  return planes;
}

size_t planeLayoutTotalBytes(const std::vector<FormatConverterPlaneLayout>& planes) {
  uint64_t total_bytes = 0;
  for (const auto& plane : planes) {
    const uint64_t plane_end = static_cast<uint64_t>(plane.byte_offset) + plane.plane_bytes;
    if (plane_end > total_bytes) {
      total_bytes = plane_end;
    }
  }
  return static_cast<size_t>(total_bytes);
}

bool isPlanarYuvInput(FormatDType dt) {
  return dt == FormatDType::kYUV420 || dt == FormatDType::kNV12BT601Full ||
         dt == FormatDType::kNV12BT709HDTV || dt == FormatDType::kNV12BT709CSC;
}

// Convert a validated byte offset into a plane base pointer.
const uint8_t* addByteOffset(const void* base, uint32_t byte_offset) {
  return static_cast<const uint8_t*>(base) + byte_offset;
}

// Non-const overload for output plane pointers passed to NPP.
uint8_t* addByteOffset(void* base, uint32_t byte_offset) {
  return static_cast<uint8_t*>(base) + byte_offset;
}

// These helpers expect a fixed plane count so layout mistakes fail early.
void validatePlaneLayoutCount(const std::vector<FormatConverterPlaneLayout>& layout,
                              size_t expected_plane_count, const char* layout_name) {
  if (layout.size() != expected_plane_count) {
    throw std::runtime_error(fmt::format(
        "{}: expected {} planes, got {}", layout_name, expected_plane_count, layout.size()));
  }
}

struct I420InputView {
  const uint8_t* ptrs[3];
  int32_t steps[3];
};

struct I420OutputView {
  uint8_t* ptrs[3];
  int32_t steps[3];
};

struct NV12InputView {
  const uint8_t* ptrs[2];
  int32_t step = 0;
};

// Build the Y/U/V pointer and step arrays that NPP expects for I420 input.
I420InputView makeI420InputView(const void* base,
                                const std::vector<FormatConverterPlaneLayout>& layout,
                                const char* layout_name) {
  validatePlaneLayoutCount(layout, 3, layout_name);
  return {{
              addByteOffset(base, layout[0].byte_offset),
              addByteOffset(base, layout[1].byte_offset),
              addByteOffset(base, layout[2].byte_offset),
          },
          {layout[0].stride_bytes, layout[1].stride_bytes, layout[2].stride_bytes}};
}

// Build writable Y/U/V plane views for RGB->I420 output conversion.
I420OutputView makeI420OutputView(void* base, const std::vector<FormatConverterPlaneLayout>& layout,
                                  const char* layout_name) {
  validatePlaneLayoutCount(layout, 3, layout_name);
  return {{
              addByteOffset(base, layout[0].byte_offset),
              addByteOffset(base, layout[1].byte_offset),
              addByteOffset(base, layout[2].byte_offset),
          },
          {layout[0].stride_bytes, layout[1].stride_bytes, layout[2].stride_bytes}};
}

// NV12 uses a shared step for Y and UV in the NPP APIs, so validate that once here.
NV12InputView makeNV12InputView(const void* base,
                                const std::vector<FormatConverterPlaneLayout>& layout,
                                const char* layout_name) {
  validatePlaneLayoutCount(layout, 2, layout_name);
  if (layout[0].stride_bytes != layout[1].stride_bytes) {
    throw std::runtime_error(
        fmt::format("{}: NV12 requires matching Y and UV strides, got {} and {}",
                    layout_name,
                    layout[0].stride_bytes,
                    layout[1].stride_bytes));
  }
  return {{addByteOffset(base, layout[0].byte_offset), addByteOffset(base, layout[1].byte_offset)},
          layout[0].stride_bytes};
}

std::vector<FormatConverterPlaneLayout> applyPlaneLayoutOverrides(
    const std::vector<FormatConverterPlaneLayout>& default_layout,
    const std::vector<int32_t>& stride_overrides, const std::vector<uint32_t>& offset_overrides,
    const char* layout_name) {
  const size_t plane_count = default_layout.size();
  if (!stride_overrides.empty() && stride_overrides.size() != plane_count) {
    throw std::runtime_error(fmt::format("{}: expected {} stride values but got {}",
                                         layout_name,
                                         plane_count,
                                         stride_overrides.size()));
  }
  if (!offset_overrides.empty() && offset_overrides.size() != plane_count) {
    throw std::runtime_error(fmt::format("{}: expected {} offset values but got {}",
                                         layout_name,
                                         plane_count,
                                         offset_overrides.size()));
  }

  auto layout = default_layout;
  for (size_t i = 0; i < plane_count; ++i) {
    if (default_layout[i].stride_bytes <= 0 ||
        (default_layout[i].plane_bytes % static_cast<uint64_t>(default_layout[i].stride_bytes)) !=
            0) {
      throw std::runtime_error(
          fmt::format("{}: invalid default plane layout for plane {}", layout_name, i));
    }
    const uint64_t plane_rows =
        default_layout[i].plane_bytes / static_cast<uint64_t>(default_layout[i].stride_bytes);
    if (!stride_overrides.empty()) {
      if (stride_overrides[i] < default_layout[i].stride_bytes) {
        throw std::runtime_error(fmt::format("{}: stride for plane {} must be >= {} bytes, got {}",
                                             layout_name,
                                             i,
                                             default_layout[i].stride_bytes,
                                             stride_overrides[i]));
      }
      layout[i].stride_bytes = stride_overrides[i];
      layout[i].plane_bytes = plane_rows * static_cast<uint64_t>(layout[i].stride_bytes);
    }
    if (!offset_overrides.empty()) {
      layout[i].byte_offset = offset_overrides[i];
    }
  }

  if (layout[0].byte_offset != 0) {
    throw std::runtime_error(
        fmt::format("{}: plane 0 offset must be 0, got {}", layout_name, layout[0].byte_offset));
  }

  uint64_t previous_plane_end = 0;
  for (size_t i = 0; i < plane_count; ++i) {
    const uint64_t plane_offset = static_cast<uint64_t>(layout[i].byte_offset);
    if (plane_offset < previous_plane_end) {
      throw std::runtime_error(fmt::format("{}: plane {} offset {} overlaps previous plane end {}",
                                           layout_name,
                                           i,
                                           plane_offset,
                                           previous_plane_end));
    }
    previous_plane_end = plane_offset + layout[i].plane_bytes;
  }

  return layout;
}

std::vector<FormatConverterPlaneLayout> defaultInputPlaneLayout(FormatDType dtype, int32_t rows,
                                                                int32_t cols, int16_t channels,
                                                                uint32_t element_size) {
  if (dtype == FormatDType::kYUV420) {
    return yuv420I420PlanesLayout(rows, cols);
  }
  if (isPlanarYuvInput(dtype)) {
    return nv12PlanesTightPacked(rows, cols);
  }
  return packedPlaneLayout(rows, cols, channels, element_size);
}

std::vector<FormatConverterPlaneLayout> defaultOutputPlaneLayout(FormatConversionType conversion,
                                                                 int32_t rows, int32_t cols,
                                                                 int16_t channels,
                                                                 uint32_t element_size) {
  if (conversion == FormatConversionType::kRGB888ToYUV420) {
    return yuv420I420PlanesLayout(rows, cols);
  }
  return packedPlaneLayout(rows, cols, channels, element_size);
}

}  // namespace

FormatConverterGpuResidentOp::~FormatConverterGpuResidentOp() {
  if (resize_buffer_dev_) {
    HOLOSCAN_CUDA_CALL_ERR_MSG(cudaFree(resize_buffer_dev_), "cudaFree resize buffer");
    resize_buffer_dev_ = nullptr;
    resize_buffer_bytes_ = 0;
  }
  if (channel_buffer_dev_) {
    HOLOSCAN_CUDA_CALL_ERR_MSG(cudaFree(channel_buffer_dev_), "cudaFree channel buffer");
    channel_buffer_dev_ = nullptr;
    channel_buffer_bytes_ = 0;
  }
}

void FormatConverterGpuResidentOp::setup(OperatorSpec& spec) {
  // For GPU-resident operators we use device_input/device_output with fixed buffer sizes.
  // Total bytes depend on width, height, the chosen in_dtype/out_dtype pair, optional resize,
  // and optional per-plane stride/offset overrides (packed RGB vs I420 vs NV12, etc.).
  //
  // Sizes must be known at setup time for GPU-resident execution; buffer byte counts are
  // finalized in initialize() after dtype validation and plane layout computation. Port sizes
  // here are placeholders (0).

  spec.device_input("in", 0);
  spec.device_output("out", 0);

  spec.param(width_, "width", "Image width", "Source frame width in pixels.", 0);
  spec.param(height_, "height", "Image height", "Source frame height in pixels.", 0);
  spec.param(in_dtype_str_,
             "in_dtype",
             "Input data type",
             "Source dtype string (see FormatConverterOp).",
             std::string(""));
  spec.param(out_dtype_str_, "out_dtype", "Output data type", "Destination dtype string.");
  spec.param(scale_min_, "scale_min", "Scale min", "Minimum value of the scale.", 0.F);
  spec.param(scale_max_, "scale_max", "Scale max", "Maximum value of the scale.", 1.F);
  spec.param(alpha_value_,
             "alpha_value",
             "Alpha value",
             "Alpha for RGB888->RGBA8888.",
             static_cast<uint8_t>(255));
  spec.param(resize_width_, "resize_width", "Resize width", "0 = no resize.", 0);
  spec.param(resize_height_, "resize_height", "Resize height", "0 = no resize.", 0);
  spec.param(resize_mode_, "resize_mode", "Resize mode", "NPP interpolation; 0 uses cubic.", 0);
  spec.param(in_plane_strides_bytes_,
             "in_plane_strides_bytes",
             "Input plane strides",
             "Optional fixed input plane strides in bytes. Empty uses the default packed/I420/NV12 "
             "layout for the selected in_dtype.",
             std::vector<int32_t>{});
  spec.param(in_plane_offsets_bytes_,
             "in_plane_offsets_bytes",
             "Input plane offsets",
             "Optional fixed input plane offsets in bytes. Empty uses the default packed/I420/NV12 "
             "layout for the selected in_dtype. Plane 0 offset must remain 0.",
             std::vector<uint32_t>{});
  spec.param(out_plane_strides_bytes_,
             "out_plane_strides_bytes",
             "Output plane strides",
             "Optional fixed output plane strides in bytes. Empty uses the default tightly packed "
             "or I420 output layout for the selected conversion.",
             std::vector<int32_t>{});
  spec.param(out_plane_offsets_bytes_,
             "out_plane_offsets_bytes",
             "Output plane offsets",
             "Optional fixed output plane offsets in bytes. Empty uses the default tightly packed "
             "or I420 output layout for the selected conversion. Plane 0 offset must remain 0.",
             std::vector<uint32_t>{});
  spec.param(out_channel_order_,
             "out_channel_order",
             "Output channel order",
             "Channel permutation (same as FormatConverterOp).",
             std::vector<int>{});
}

void FormatConverterGpuResidentOp::initialize() {
  GPUResidentOperator::initialize();

#if CUDART_VERSION >= 13000
  int device = 0;
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaGetDevice(&device), "Failed to get CUDA device");
  cudaDeviceProp prop{};
  HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaGetDeviceProperties(&prop, device),
                                 "Failed to get CUDA device properties");
  npp_stream_ctx_.nCudaDeviceId = device;
  npp_stream_ctx_.nMultiProcessorCount = prop.multiProcessorCount;
  npp_stream_ctx_.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
  npp_stream_ctx_.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
  npp_stream_ctx_.nSharedMemPerBlock = prop.sharedMemPerBlock;
  npp_stream_ctx_.nCudaDevAttrComputeCapabilityMajor = prop.major;
  npp_stream_ctx_.nCudaDevAttrComputeCapabilityMinor = prop.minor;
#else
  const auto nppStatus = nppGetStreamContext(&npp_stream_ctx_);
  if (NPP_SUCCESS != nppStatus) {
    throw std::runtime_error("Failed to get NPP CUDA stream context");
  }
#endif

  src_cols_ = width_.get();
  src_rows_ = height_.get();
  if (src_cols_ <= 0 || src_rows_ <= 0) {
    throw std::runtime_error(
        fmt::format("FormatConverterGpuResidentOp: width ({}) and "
                    "height ({}) must be positive",
                    src_cols_,
                    src_rows_));
  }

  out_dtype_ = toFormatDType(out_dtype_str_.get());
  if (out_dtype_ == FormatDType::kUnknown) {
    throw std::runtime_error(fmt::format("Unsupported output dtype: {}", out_dtype_str_.get()));
  }
  out_element_size_ = elementSizeFromFormatDType(out_dtype_);
  if (out_element_size_ == 0) {
    throw std::runtime_error(fmt::format("Unsupported output dtype: {}", out_dtype_str_.get()));
  }

  if (in_dtype_str_.get().empty()) {
    throw std::runtime_error("FormatConverterGpuResidentOp: in_dtype must be set explicitly.");
  }
  in_dtype_ = toFormatDType(in_dtype_str_.get());
  if (in_dtype_ == FormatDType::kUnknown) {
    throw std::runtime_error(fmt::format("Unsupported input dtype: {}", in_dtype_str_.get()));
  }
  format_conversion_type_ =
      getFormatConversionType(normalizeFormatDType(in_dtype_), normalizeFormatDType(out_dtype_));
  in_element_size_ = elementSizeFromFormatDType(in_dtype_);
  if (in_element_size_ == 0) {
    throw std::runtime_error(fmt::format("Unsupported input dtype: {}", in_dtype_str_.get()));
  }

  if (format_conversion_type_ == FormatConversionType::kUnknown) {
    throw std::runtime_error(
        fmt::format("Unsupported conversion: {} -> {}", in_dtype_str_.get(), out_dtype_str_.get()));
  }

  const int32_t rw = resize_width_.get();
  const int32_t rh = resize_height_.get();
  const bool do_resize = (rw > 0 && rh > 0);
  if ((rw > 0) != (rh > 0)) {
    throw std::runtime_error(
        "FormatConverterGpuResidentOp: resize_width and resize_height must both be positive or "
        "both zero.");
  }
  if (do_resize && isPlanarYuvInput(in_dtype_)) {
    throw std::runtime_error(
        "FormatConverterGpuResidentOp: resize is not supported for YUV420/NV12 inputs.");
  }
  if (do_resize && in_dtype_ == FormatDType::kYUYV) {
    throw std::runtime_error(
        "FormatConverterGpuResidentOp: resize is not supported for YUYV input (use 3/4-channel "
        "packed RGB/RGBA).");
  }
  if (do_resize && in_dtype_ == FormatDType::kUYVY) {
    throw std::runtime_error(
        "FormatConverterGpuResidentOp: resize is not supported for UYVY input (use 3/4-channel "
        "packed RGB/RGBA).");
  }
  if (do_resize && in_dtype_ == FormatDType::kFloat32) {
    throw std::runtime_error(
        "FormatConverterGpuResidentOp: resize is not supported for float32 input tensors.");
  }
  if (do_resize &&
      (in_dtype_ == FormatDType::kRGB161616 || in_dtype_ == FormatDType::kRGBA16161616 ||
       in_dtype_ == FormatDType::kUnsigned16)) {
    throw std::runtime_error(
        "FormatConverterGpuResidentOp: resize is not supported for 16-bit input tensors.");
  }

  switch (resize_mode_.get()) {
    case 0:
      npp_resize_mode_ = NPPI_INTER_CUBIC;
      break;
    case 1:
    case 2:
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 16:
    case 17:
    case static_cast<int32_t>(0x8000000):
      npp_resize_mode_ = static_cast<NppiInterpolationMode>(resize_mode_.get());
      break;
    default:
      throw std::runtime_error(fmt::format("Unsupported resize mode: {}", resize_mode_.get()));
  }

  if (isPlanarYuvInput(in_dtype_)) {
    if ((src_rows_ % 2) != 0 || (src_cols_ % 2) != 0) {
      throw std::runtime_error(
          "FormatConverterGpuResidentOp: YUV420/NV12 require even width and height.");
    }
  }

  // Packed 4:2:2 chroma subsampling shares U/V between adjacent pixel pairs; require even width.
  if ((in_dtype_ == FormatDType::kYUYV || in_dtype_ == FormatDType::kUYVY) &&
      (src_cols_ % 2) != 0) {
    throw std::runtime_error(
        fmt::format("FormatConverterGpuResidentOp: {} input requires an even width, got {}.",
                    in_dtype_str_.get(),
                    src_cols_));
  }

  work_rows_ = do_resize ? rh : src_rows_;
  work_cols_ = do_resize ? rw : src_cols_;

  if (isPlanarYuvInput(in_dtype_)) {
    in_channels_ = 0;
  } else if (in_dtype_ == FormatDType::kYUYV || in_dtype_ == FormatDType::kUYVY) {
    in_channels_ = 2;
  } else if (in_dtype_ == FormatDType::kRGB888 || in_dtype_ == FormatDType::kUnsigned8 ||
             in_dtype_ == FormatDType::kFloat32 || in_dtype_ == FormatDType::kRGB161616) {
    in_channels_ = 3;
  } else if (in_dtype_ == FormatDType::kRGBA8888 || in_dtype_ == FormatDType::kRGBA16161616) {
    in_channels_ = 4;
  } else {
    throw std::runtime_error(
        "FormatConverterGpuResidentOp: unsupported input dtype "
        "for channel layout.");
  }

  out_channels_ = in_channels_;
  switch (format_conversion_type_) {
    case FormatConversionType::kRGB888ToRGBA8888:
    case FormatConversionType::kYUV420ToRGBA8888:
      out_channels_ = 4;
      break;
    case FormatConversionType::kRGBA16161616ToRGB888:
    case FormatConversionType::kRGB161616ToRGB888:
    case FormatConversionType::kRGBA8888ToRGB888:
    case FormatConversionType::kNV12BT601FullToRGB888:
    case FormatConversionType::kNV12BT709CSCToRGB888:
    case FormatConversionType::kNV12BT709HDTVToRGB888:
    case FormatConversionType::kYUV420ToRGB888:
    case FormatConversionType::kRGBA8888ToFloat32:
    case FormatConversionType::kYUYVToRGB888:
    case FormatConversionType::kUYVYToRGB888:
      out_channels_ = 3;
      break;
    default:
      break;
  }

  if (!isPlanarYuvInput(in_dtype_)) {
    verifyFormatDTypeChannelsOrThrow(in_dtype_, in_channels_);
  }
  if (format_conversion_type_ != FormatConversionType::kRGB888ToYUV420) {
    verifyFormatDTypeChannelsOrThrow(out_dtype_, out_channels_);
  }
  if (format_conversion_type_ == FormatConversionType::kNone) {
    if (in_element_size_ != out_element_size_ || in_channels_ != out_channels_) {
      throw std::runtime_error(
          fmt::format("FormatConverterGpuResidentOp: kNone conversion requires matching layout, "
                      "but in({} ch x {} bytes) != out({} ch x {} bytes)",
                      in_channels_,
                      in_element_size_,
                      out_channels_,
                      out_element_size_));
    }
  }

  input_plane_layout_ = applyPlaneLayoutOverrides(
      defaultInputPlaneLayout(in_dtype_, src_rows_, src_cols_, in_channels_, in_element_size_),
      in_plane_strides_bytes_.get(),
      in_plane_offsets_bytes_.get(),
      "FormatConverterGpuResidentOp input plane layout");
  output_plane_layout_ = applyPlaneLayoutOverrides(
      defaultOutputPlaneLayout(
          format_conversion_type_, work_rows_, work_cols_, out_channels_, out_element_size_),
      out_plane_strides_bytes_.get(),
      out_plane_offsets_bytes_.get(),
      "FormatConverterGpuResidentOp output plane layout");

  const uint32_t src_elem = in_element_size_;
  const size_t in_bytes = planeLayoutTotalBytes(input_plane_layout_);
  const size_t out_bytes = planeLayoutTotalBytes(output_plane_layout_);

  if (do_resize) {
    resize_buffer_bytes_ = static_cast<size_t>(work_rows_) * static_cast<size_t>(work_cols_) *
                           static_cast<size_t>(in_channels_) * static_cast<size_t>(src_elem);
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&resize_buffer_dev_, resize_buffer_bytes_),
                                   "cudaMalloc resize buffer");
  }

  if (format_conversion_type_ == FormatConversionType::kRGBA16161616ToRGB888) {
    channel_buffer_bytes_ = static_cast<size_t>(work_rows_) * static_cast<size_t>(work_cols_) * 4u;
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&channel_buffer_dev_, channel_buffer_bytes_),
                                   "cudaMalloc channel scratch buffer");
  } else if (format_conversion_type_ == FormatConversionType::kRGBA8888ToFloat32) {
    channel_buffer_bytes_ = static_cast<size_t>(work_rows_) * static_cast<size_t>(work_cols_) * 3u;
    HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMalloc(&channel_buffer_dev_, channel_buffer_bytes_),
                                   "cudaMalloc channel scratch buffer");
  }

  // Register device ports with the GPU-resident executor using computed packed/planar sizes.
  auto& ospec = *this->spec();
  ospec.device_input("in", in_bytes);
  ospec.device_output("out", out_bytes);

  HOLOSCAN_LOG_INFO(
      "FormatConverterGpuResidentOp: {}x{} -> work {}x{}, conversion={}, in_bytes={}, out_bytes={}",
      src_cols_,
      src_rows_,
      work_cols_,
      work_rows_,
      static_cast<int>(format_conversion_type_),
      in_bytes,
      out_bytes);
}

void FormatConverterGpuResidentOp::compute([[maybe_unused]] InputContext& op_input,
                                           [[maybe_unused]] OutputContext& op_output,
                                           [[maybe_unused]] ExecutionContext& context) {
  // In GPU-resident mode:
  // - This function is called once from the CPU during graph capture.
  // - CUDA work launched here (NPP resize, NPP format conversion, device memcpy) is captured
  //   into a CUDA graph; later iterations replay the graph on the GPU.
  //
  // Avoid host-side branching on per-frame payload data, host-device synchronization, and any
  // work that cannot be captured. Paths here depend only on configuration fixed in initialize().

  void* in_ptr = device_memory("in");
  void* out_ptr = device_memory("out");
  if (!in_ptr || !out_ptr) {
    throw std::runtime_error("FormatConverterGpuResidentOp::compute: device memory not set.");
  }

  auto stream_ptr = cuda_stream();
  if (!stream_ptr) {
    throw std::runtime_error("FormatConverterGpuResidentOp::compute: CUDA stream not available.");
  }
  npp_stream_ctx_.hStream = *stream_ptr;

  const void* convert_in = in_ptr;
  std::vector<FormatConverterPlaneLayout> input_plane_layout = input_plane_layout_;

  if (resize_buffer_dev_) {
    const NppiSize src_size{static_cast<int>(src_cols_), static_cast<int>(src_rows_)};
    const NppiRect src_roi{0, 0, static_cast<int>(src_cols_), static_cast<int>(src_rows_)};
    const NppiSize dst_size{static_cast<int>(work_cols_), static_cast<int>(work_rows_)};
    const NppiRect dst_roi{0, 0, static_cast<int>(work_cols_), static_cast<int>(work_rows_)};
    const int32_t src_step = input_plane_layout_[0].stride_bytes;
    const int32_t dst_step = work_cols_ * in_channels_ * static_cast<int32_t>(in_element_size_);

    NppStatus st = NPP_SUCCESS;
    bool resize_dispatched = false;
    switch (in_channels_) {
      case 3:
        if (in_element_size_ == sizeof(uint8_t)) {
          st = nppiResize_8u_C3R_Ctx(static_cast<const Npp8u*>(in_ptr),
                                     src_step,
                                     src_size,
                                     src_roi,
                                     static_cast<Npp8u*>(resize_buffer_dev_),
                                     dst_step,
                                     dst_size,
                                     dst_roi,
                                     npp_resize_mode_,
                                     npp_stream_ctx_);
          resize_dispatched = true;
        }
        break;
      case 4:
        if (in_element_size_ == sizeof(uint8_t)) {
          st = nppiResize_8u_C4R_Ctx(static_cast<const Npp8u*>(in_ptr),
                                     src_step,
                                     src_size,
                                     src_roi,
                                     static_cast<Npp8u*>(resize_buffer_dev_),
                                     dst_step,
                                     dst_size,
                                     dst_roi,
                                     npp_resize_mode_,
                                     npp_stream_ctx_);
          resize_dispatched = true;
        }
        break;
      default:
        break;
    }
    if (!resize_dispatched) {
      throw std::runtime_error(
          "FormatConverterGpuResidentOp: resize is only supported for 8-bit 3-channel or "
          "4-channel packed input.");
    }
    if (st != NPP_SUCCESS) {
      throw std::runtime_error(
          fmt::format("FormatConverterGpuResidentOp: resize failed (NPP error code: {})",
                      static_cast<int>(st)));
    }
    convert_in = resize_buffer_dev_;
    input_plane_layout = packedPlaneLayout(work_rows_, work_cols_, in_channels_, in_element_size_);
  }

  launchConvert(convert_in, input_plane_layout, out_ptr, output_plane_layout_);
}

// Dispatches the fixed conversion chosen in initialize() via NPP (or D2D copy for kNone).
void FormatConverterGpuResidentOp::launchConvert(
    const void* in_tensor_data, const std::vector<FormatConverterPlaneLayout>& input_plane_layout,
    void* out_tensor_data, const std::vector<FormatConverterPlaneLayout>& output_plane_layout) {
  const int32_t rows = work_rows_;
  const int32_t columns = work_cols_;
  const int16_t out_ch = out_channels_;

  const uint32_t dst_typesize = out_element_size_;

  const int32_t src_step = input_plane_layout[0].stride_bytes;
  const int32_t dst_step = output_plane_layout[0].stride_bytes;

  const auto& out_channel_order = out_channel_order_.get();

  NppStatus status = NPP_ERROR;
  const NppiSize roi = {static_cast<int>(columns), static_cast<int>(rows)};

  switch (format_conversion_type_) {
    case FormatConversionType::kNone: {
      const auto* in_tensor_ptr = static_cast<const uint8_t*>(in_tensor_data);
      auto* out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);
      const size_t width_bytes =
          static_cast<size_t>(columns) * static_cast<size_t>(out_ch) * dst_typesize;
      HOLOSCAN_CUDA_CALL_THROW_ERROR(cudaMemcpy2DAsync(out_tensor_ptr,
                                                       dst_step,
                                                       in_tensor_ptr,
                                                       src_step,
                                                       width_bytes,
                                                       rows,
                                                       cudaMemcpyDeviceToDevice,
                                                       npp_stream_ctx_.hStream),
                                     "cudaMemcpy2DAsync");
      status = NPP_SUCCESS;
      break;
    }
    case FormatConversionType::kUnsigned8ToFloat32: {
      const auto* in_tensor_ptr = static_cast<const uint8_t*>(in_tensor_data);
      auto* out_tensor_ptr = static_cast<float*>(out_tensor_data);
      status = nppiScale_8u32f_C3R_Ctx(in_tensor_ptr,
                                       src_step,
                                       out_tensor_ptr,
                                       dst_step,
                                       roi,
                                       scale_min_.get(),
                                       scale_max_.get(),
                                       npp_stream_ctx_);
      break;
    }
    case FormatConversionType::kFloat32ToUnsigned8: {
      const auto* in_tensor_ptr = static_cast<const float*>(in_tensor_data);
      auto* out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);
      status = nppiScale_32f8u_C3R_Ctx(in_tensor_ptr,
                                       src_step,
                                       out_tensor_ptr,
                                       dst_step,
                                       roi,
                                       scale_min_.get(),
                                       scale_max_.get(),
                                       npp_stream_ctx_);
      break;
    }
    case FormatConversionType::kRGB888ToRGBA8888: {
      const auto* in_tensor_ptr = static_cast<const uint8_t*>(in_tensor_data);
      auto* out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);
      int dst_order[4]{0, 1, 2, 3};
      if (!out_channel_order.empty()) {
        if (out_channel_order.size() != 4) {
          throw std::runtime_error("Invalid channel order for RGBA8888.");
        }
        for (int i = 0; i < 4; i++) {
          dst_order[i] = out_channel_order[i];
        }
      }
      status = nppiSwapChannels_8u_C3C4R_Ctx(in_tensor_ptr,
                                             src_step,
                                             out_tensor_ptr,
                                             dst_step,
                                             roi,
                                             dst_order,
                                             alpha_value_.get(),
                                             npp_stream_ctx_);
      break;
    }
    case FormatConversionType::kRGBA8888ToRGB888: {
      const auto* in_tensor_ptr = static_cast<const uint8_t*>(in_tensor_data);
      auto* out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);
      int dst_order[3]{0, 1, 2};
      if (!out_channel_order.empty()) {
        if (out_channel_order.size() != 3) {
          throw std::runtime_error("Invalid channel order for RGB888.");
        }
        for (int i = 0; i < 3; i++) {
          dst_order[i] = out_channel_order[i];
        }
      }
      status = nppiSwapChannels_8u_C4C3R_Ctx(
          in_tensor_ptr, src_step, out_tensor_ptr, dst_step, roi, dst_order, npp_stream_ctx_);
      break;
    }
    case FormatConversionType::kRGB161616ToRGB888: {
      const auto* in_tensor_ptr = static_cast<const uint16_t*>(in_tensor_data);
      auto* out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);
      status = nppiScale_16u8u_C3R_Ctx(in_tensor_ptr,
                                       src_step,
                                       out_tensor_ptr,
                                       dst_step,
                                       roi,
                                       NPP_ALG_HINT_NONE,
                                       npp_stream_ctx_);
      break;
    }
    case FormatConversionType::kRGBA16161616ToRGB888: {
      const auto* in_tensor_ptr = static_cast<const uint16_t*>(in_tensor_data);
      auto* out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);
      if (!channel_buffer_dev_) {
        throw std::runtime_error("channel buffer not allocated");
      }
      const int32_t channel_buffer_step = columns * 4 * static_cast<int32_t>(dst_typesize);
      int dst_order[3]{0, 1, 2};
      if (!out_channel_order.empty()) {
        if (out_channel_order.size() != 3) {
          throw std::runtime_error("Invalid channel order for RGB888");
        }
        for (int i = 0; i < 3; i++) {
          dst_order[i] = out_channel_order[i];
        }
      }
      status = nppiScale_16u8u_C4R_Ctx(in_tensor_ptr,
                                       src_step,
                                       static_cast<Npp8u*>(channel_buffer_dev_),
                                       channel_buffer_step,
                                       roi,
                                       NPP_ALG_HINT_NONE,
                                       npp_stream_ctx_);
      if (status == NPP_SUCCESS) {
        status = nppiSwapChannels_8u_C4C3R_Ctx(static_cast<Npp8u*>(channel_buffer_dev_),
                                               channel_buffer_step,
                                               out_tensor_ptr,
                                               dst_step,
                                               roi,
                                               dst_order,
                                               npp_stream_ctx_);
      }
      break;
    }
    case FormatConversionType::kRGBA8888ToFloat32: {
      const auto* in_tensor_ptr = static_cast<const uint8_t*>(in_tensor_data);
      auto* out_tensor_ptr = static_cast<float*>(out_tensor_data);
      if (!channel_buffer_dev_) {
        throw std::runtime_error("channel buffer not allocated");
      }
      int dst_order[3]{0, 1, 2};
      if (!out_channel_order.empty()) {
        if (out_channel_order.size() != 3) {
          throw std::runtime_error("Invalid channel order for RGB888");
        }
        for (int i = 0; i < 3; i++) {
          dst_order[i] = out_channel_order[i];
        }
      }
      const int32_t channel_buffer_step = columns * 3 * static_cast<int32_t>(sizeof(uint8_t));
      status = nppiSwapChannels_8u_C4C3R_Ctx(in_tensor_ptr,
                                             src_step,
                                             static_cast<Npp8u*>(channel_buffer_dev_),
                                             channel_buffer_step,
                                             roi,
                                             dst_order,
                                             npp_stream_ctx_);
      if (status == NPP_SUCCESS) {
        status = nppiScale_8u32f_C3R_Ctx(static_cast<const Npp8u*>(channel_buffer_dev_),
                                         channel_buffer_step,
                                         out_tensor_ptr,
                                         dst_step,
                                         roi,
                                         scale_min_.get(),
                                         scale_max_.get(),
                                         npp_stream_ctx_);
      }
      break;
    }
    case FormatConversionType::kRGB888ToYUV420: {
      const auto* in_tensor_ptr = static_cast<const uint8_t*>(in_tensor_data);
      // Resolve the configured output layout once, then hand NPP the ready-made plane arrays.
      auto out_i420 = makeI420OutputView(
          out_tensor_data, output_plane_layout, "FormatConverterGpuResidentOp output I420 layout");
      status = nppiRGBToYUV420_8u_C3P3R_Ctx(
          in_tensor_ptr, src_step, out_i420.ptrs, out_i420.steps, roi, npp_stream_ctx_);
      break;
    }
    case FormatConversionType::kYUV420ToRGBA8888: {
      // Convert the fixed Y/U/V layout into the pointer/step arrays the NPP call needs.
      auto in_i420 = makeI420InputView(
          in_tensor_data, input_plane_layout, "FormatConverterGpuResidentOp input I420 layout");
      auto* out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);
      status = nppiYUV420ToRGB_8u_P3AC4R_Ctx(
          in_i420.ptrs, in_i420.steps, out_tensor_ptr, dst_step, roi, npp_stream_ctx_);
      break;
    }
    case FormatConversionType::kYUV420ToRGB888: {
      // Same I420 input view as above; only the destination format differs.
      auto in_i420 = makeI420InputView(
          in_tensor_data, input_plane_layout, "FormatConverterGpuResidentOp input I420 layout");
      auto* out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);
      status = nppiYUV420ToRGB_8u_P3C3R_Ctx(
          in_i420.ptrs, in_i420.steps, out_tensor_ptr, dst_step, roi, npp_stream_ctx_);
      break;
    }
    case FormatConversionType::kNV12BT709HDTVToRGB888: {
      // Centralize NV12 plane/step validation before calling the color-space-specific kernel.
      const auto in_nv12 = makeNV12InputView(
          in_tensor_data, input_plane_layout, "FormatConverterGpuResidentOp input NV12 layout");
      auto* out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);
      status = nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx(
          in_nv12.ptrs, in_nv12.step, out_tensor_ptr, dst_step, roi, npp_stream_ctx_);
      break;
    }
    case FormatConversionType::kNV12BT709CSCToRGB888: {
      // Reuse the same validated NV12 view; only the NPP conversion entry point changes.
      const auto in_nv12 = makeNV12InputView(
          in_tensor_data, input_plane_layout, "FormatConverterGpuResidentOp input NV12 layout");
      auto* out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);
      status = nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx(
          in_nv12.ptrs, in_nv12.step, out_tensor_ptr, dst_step, roi, npp_stream_ctx_);
      break;
    }
    case FormatConversionType::kNV12BT601FullToRGB888: {
      // Reuse the same validated NV12 view; only the NPP conversion entry point changes.
      const auto in_nv12 = makeNV12InputView(
          in_tensor_data, input_plane_layout, "FormatConverterGpuResidentOp input NV12 layout");
      auto* out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);
      status = nppiNV12ToRGB_8u_P2C3R_Ctx(
          in_nv12.ptrs, in_nv12.step, out_tensor_ptr, dst_step, roi, npp_stream_ctx_);
      break;
    }
    case FormatConversionType::kYUYVToRGB888: {
      const auto* in_tensor_ptr = static_cast<const uint8_t*>(in_tensor_data);
      const int32_t in_step = input_plane_layout[0].stride_bytes;
      auto* out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);
      status = nppiYUV422ToRGB_8u_C2C3R_Ctx(
          in_tensor_ptr, in_step, out_tensor_ptr, dst_step, roi, npp_stream_ctx_);
      break;
    }
    case FormatConversionType::kUYVYToRGB888: {
      const auto* in_tensor_ptr = static_cast<const uint8_t*>(in_tensor_data);
      const int32_t in_step = input_plane_layout[0].stride_bytes;
      auto* out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);
      status = nppiCbYCr422ToRGB_8u_C2C3R_Ctx(
          in_tensor_ptr, in_step, out_tensor_ptr, dst_step, roi, npp_stream_ctx_);
      break;
    }
    default:
      throw std::runtime_error(fmt::format("Unsupported format conversion in GPU-resident op: {}",
                                           static_cast<int>(format_conversion_type_)));
  }

  if (status != NPP_SUCCESS) {
    throw std::runtime_error(
        fmt::format("FormatConverterGpuResidentOp: conversion failed (NPP error code: {})",
                    static_cast<int>(status)));
  }

  switch (format_conversion_type_) {
    case FormatConversionType::kNone:
    case FormatConversionType::kUnsigned8ToFloat32:
    case FormatConversionType::kFloat32ToUnsigned8: {
      if (!out_channel_order.empty()) {
        switch (out_ch) {
          case 3: {
            int dst_order[3]{0, 1, 2};
            if (out_channel_order.size() != 3) {
              throw std::runtime_error(
                  fmt::format("Invalid channel order for {}", out_dtype_str_.get()));
            }
            for (int i = 0; i < 3; i++) {
              dst_order[i] = out_channel_order[i];
            }
            switch (out_element_size_) {
              case sizeof(uint8_t): {
                auto* out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);
                status = nppiSwapChannels_8u_C3IR_Ctx(
                    out_tensor_ptr, dst_step, roi, dst_order, npp_stream_ctx_);
                break;
              }
              case sizeof(float): {
                auto* out_tensor_ptr = static_cast<float*>(out_tensor_data);
                status = nppiSwapChannels_32f_C3IR_Ctx(
                    out_tensor_ptr, dst_step, roi, dst_order, npp_stream_ctx_);
                break;
              }
              default:
                throw std::runtime_error(fmt::format(
                    "Unsupported output dtype for channel reorder: {}", out_dtype_str_.get()));
            }
            break;
          }
          case 4: {
            int dst_order[4]{0, 1, 2, 3};
            if (out_channel_order.size() != 4) {
              throw std::runtime_error(
                  fmt::format("Invalid channel order for {}", out_dtype_str_.get()));
            }
            for (int i = 0; i < 4; i++) {
              dst_order[i] = out_channel_order[i];
            }
            switch (out_element_size_) {
              case sizeof(uint8_t): {
                auto* out_tensor_ptr = static_cast<uint8_t*>(out_tensor_data);
                status = nppiSwapChannels_8u_C4IR_Ctx(
                    out_tensor_ptr, dst_step, roi, dst_order, npp_stream_ctx_);
                break;
              }
              case sizeof(float): {
                auto* out_tensor_ptr = static_cast<float*>(out_tensor_data);
                status = nppiSwapChannels_32f_C4IR_Ctx(
                    out_tensor_ptr, dst_step, roi, dst_order, npp_stream_ctx_);
                break;
              }
              default:
                throw std::runtime_error(fmt::format(
                    "Unsupported output dtype for channel reorder: {}", out_dtype_str_.get()));
            }
            break;
          }
          default:
            HOLOSCAN_LOG_ERROR(
                "FormatConverterGpuResidentOp: out_channel_order is only supported for 3- or "
                "4-channel packed outputs; ignoring (out_ch={})",
                static_cast<int>(out_ch));
            break;
        }
        if (status != NPP_SUCCESS) {
          throw std::runtime_error("Failed to reorder output channels");
        }
      }
      break;
    }
    default:
      break;
  }
}

}  // namespace holoscan::ops
