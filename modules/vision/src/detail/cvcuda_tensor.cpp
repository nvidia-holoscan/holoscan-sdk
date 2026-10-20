// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <limits>
#include <new>
#include <string>
#include <utility>

#include <cvcuda/cuda_tools/TypeTraits.hpp>
#include <nvcv/DataType.hpp>
#include <nvcv/Exception.hpp>
#include <nvcv/TensorData.hpp>
#include <nvcv/TensorShape.hpp>

#include <holoscan/vision/vision_types.hpp>

#include "detail/cvcuda_tensor.hpp"

namespace holoscan::vision::detail {
namespace {

[[nodiscard]] auto invalid_argument(std::string message) {
  return make_unexpected(Error{ErrorCode::kInvalidArgument, std::move(message)});
}

[[nodiscard]] expected<std::size_t, Error> element_size(DLDataType dtype) {
  if (dtype.lanes != 1U || dtype.bits == 0U || dtype.bits % 8U != 0U) {
    return invalid_argument("Vision tensors require a scalar byte-addressable dtype");
  }
  return static_cast<std::size_t>(dtype.bits / 8U);
}

[[nodiscard]] expected<nvcv::DataType, Error> nvcv_dtype(DLDataType dtype) {
  if (dtype.lanes != 1U) {
    return invalid_argument("Vision tensors require a single-lane dtype");
  }
  if (dtype.code == kDLUInt && dtype.bits == 8U) {
    return nvcv::TYPE_U8;
  }
  if (dtype.code == kDLInt && dtype.bits == 8U) {
    return nvcv::TYPE_S8;
  }
  if (dtype.code == kDLUInt && dtype.bits == 16U) {
    return nvcv::TYPE_U16;
  }
  if (dtype.code == kDLInt && dtype.bits == 16U) {
    return nvcv::TYPE_S16;
  }
  if (dtype.code == kDLUInt && dtype.bits == 32U) {
    return nvcv::TYPE_U32;
  }
  if (dtype.code == kDLInt && dtype.bits == 32U) {
    return nvcv::TYPE_S32;
  }
  if (dtype.code == kDLFloat && dtype.bits == 32U) {
    return nvcv::TYPE_F32;
  }
  if (dtype.code == kDLFloat && dtype.bits == 64U) {
    return nvcv::TYPE_F64;
  }
  return invalid_argument("Vision tensor dtype is not supported by CV-CUDA");
}

template <typename Element>
[[nodiscard]] std::size_t interleaved_access_alignment(std::int64_t channels) noexcept {
  switch (channels) {
    case 1:
      return alignof(nvcv::cuda::MakeType<Element, 1>);
    case 2:
      return alignof(nvcv::cuda::MakeType<Element, 2>);
    case 3:
      return alignof(nvcv::cuda::MakeType<Element, 3>);
    case 4:
      return alignof(nvcv::cuda::MakeType<Element, 4>);
    default:
      return 0U;
  }
}

template <typename Element>
[[nodiscard]] std::size_t access_alignment(std::int64_t channels,
                                           ImageTensorLayout layout) noexcept {
  return layout == ImageTensorLayout::kPlanar ? alignof(Element)
                                              : interleaved_access_alignment<Element>(channels);
}

[[nodiscard]] expected<std::size_t, Error> cvcuda_access_alignment(DLDataType dtype,
                                                                   std::int64_t channels,
                                                                   ImageTensorLayout layout) {
  if (dtype.code == kDLUInt && dtype.bits == 8U) {
    return access_alignment<unsigned char>(channels, layout);
  }
  if (dtype.code == kDLInt && dtype.bits == 8U) {
    return access_alignment<signed char>(channels, layout);
  }
  if (dtype.code == kDLUInt && dtype.bits == 16U) {
    return access_alignment<unsigned short>(channels, layout);
  }
  if (dtype.code == kDLInt && dtype.bits == 16U) {
    return access_alignment<short>(channels, layout);
  }
  if (dtype.code == kDLUInt && dtype.bits == 32U) {
    return access_alignment<unsigned int>(channels, layout);
  }
  if (dtype.code == kDLInt && dtype.bits == 32U) {
    return access_alignment<int>(channels, layout);
  }
  if (dtype.code == kDLFloat && dtype.bits == 32U) {
    return access_alignment<float>(channels, layout);
  }
  if (dtype.code == kDLFloat && dtype.bits == 64U) {
    return access_alignment<double>(channels, layout);
  }
  return invalid_argument("Vision tensor access type is not supported by CV-CUDA");
}

[[nodiscard]] expected<void, Error> validate_access_alignment(
    const void* data, std::span<const std::int64_t> byte_strides, std::size_t height_index,
    std::size_t channel_index, bool is_planar, std::size_t alignment) {
  if (alignment == 0U || reinterpret_cast<std::uintptr_t>(data) % alignment != 0U) {
    return invalid_argument("Vision tensor data address does not satisfy CV-CUDA access alignment");
  }
  if (byte_strides[height_index] % static_cast<std::int64_t>(alignment) != 0) {
    return invalid_argument("Vision tensor row stride does not satisfy CV-CUDA access alignment");
  }
  if (is_planar && byte_strides[channel_index] % static_cast<std::int64_t>(alignment) != 0) {
    return invalid_argument(
        "Vision tensor channel stride does not satisfy CV-CUDA access alignment");
  }
  if (byte_strides.size() == 4U && byte_strides[0] % static_cast<std::int64_t>(alignment) != 0) {
    return invalid_argument("Vision tensor batch stride does not satisfy CV-CUDA access alignment");
  }
  return {};
}

[[nodiscard]] bool dtype_supported(DLDataType dtype, TensorOperation operation) {
  if (dtype.lanes != 1U) {
    return false;
  }
  const bool is_u8 = dtype.code == kDLUInt && dtype.bits == 8U;
  const bool is_s8 = dtype.code == kDLInt && dtype.bits == 8U;
  const bool is_u16 = dtype.code == kDLUInt && dtype.bits == 16U;
  const bool is_s16 = dtype.code == kDLInt && dtype.bits == 16U;
  const bool is_s32 = dtype.code == kDLInt && dtype.bits == 32U;
  const bool is_f32 = dtype.code == kDLFloat && dtype.bits == 32U;
  const bool is_f64 = dtype.code == kDLFloat && dtype.bits == 64U;

  switch (operation) {
    case TensorOperation::kResize:
    case TensorOperation::kCopyMakeBorder:
      return is_u8 || is_u16 || is_s16 || is_f32;
    case TensorOperation::kCustomCrop:
      return is_u8 || is_s8 || is_u16 || is_s16 || is_s32 || is_f32 || is_f64;
    case TensorOperation::kRemap:
      return is_u8 || is_f32;
    case TensorOperation::kRemapMap:
      return is_f32;
    case TensorOperation::kNormalize:
      return is_u8 || is_s8 || is_u16 || is_s16 || is_s32 || is_f32;
    case TensorOperation::kConvertTo:
    case TensorOperation::kReformat:
      return is_u8 || is_s8 || is_u16 || is_s16 || is_s32 || is_f32 || is_f64;
    case TensorOperation::kBayerDemosaic:
      // NPP's Bayer kernels (and schema::ImageT's RGB8/RGBA8, RGB16/RGBA16 encodings) support
      // exactly these two bit depths.
      return is_u8 || is_u16;
  }
  return false;
}

[[nodiscard]] bool layout_supported(TensorOperation operation, ImageTensorLayout layout) {
  return layout == ImageTensorLayout::kInterleaved ||
         (layout == ImageTensorLayout::kPlanar && operation == TensorOperation::kReformat);
}

[[nodiscard]] bool channels_supported(DLDataType dtype, TensorOperation operation,
                                      ImageTensorLayout layout, std::int64_t channels) {
  switch (operation) {
    case TensorOperation::kResize:
    case TensorOperation::kNormalize:
      return channels == 1 || channels == 3 || channels == 4;
    case TensorOperation::kCustomCrop:
      return channels == 1 || channels == 3 || channels == 4 ||
             (layout == ImageTensorLayout::kInterleaved && channels == 2);
    case TensorOperation::kCopyMakeBorder:
      return channels == 1 || channels == 3 || channels == 4 ||
             (layout == ImageTensorLayout::kInterleaved && channels == 2 && dtype.code == kDLUInt &&
              dtype.bits == 8U);
    case TensorOperation::kRemap:
      return (dtype.code == kDLUInt && dtype.bits == 8U &&
              (channels == 1 || channels == 3 || channels == 4)) ||
             (dtype.code == kDLFloat && dtype.bits == 32U && channels == 1);
    case TensorOperation::kRemapMap:
      return channels == 2;
    case TensorOperation::kConvertTo:
    case TensorOperation::kReformat:
      return channels >= 1 && channels <= 4;
    case TensorOperation::kBayerDemosaic:
      // Single-channel Bayer mosaic input; three (RGB) or four (RGBA) channel output.
      return channels == 1 || channels == 3 || channels == 4;
  }
  return false;
}

[[nodiscard]] expected<std::int64_t, Error> checked_stride_product(std::int64_t lhs,
                                                                   std::int64_t rhs) {
  if (lhs <= 0 || rhs <= 0 || lhs > std::numeric_limits<std::int64_t>::max() / rhs) {
    return invalid_argument("Vision tensor stride is invalid or overflows");
  }
  return lhs * rhs;
}

[[nodiscard]] constexpr bool is_non_positive(std::int64_t value) noexcept {
  return value <= 0;
}

constexpr std::int64_t kMaxCvCudaTensorByteSpan = std::numeric_limits<std::int32_t>::max();

[[nodiscard]] constexpr bool exceeds_cvcuda_byte_span(std::int64_t value) noexcept {
  return value > kMaxCvCudaTensorByteSpan;
}

struct ImageAxes {
  std::size_t height{};
  std::size_t width{};
  std::size_t channel{};
};

// Axis positions for interleaved (HWC/NHWC) or planar (CHW/NCHW) storage of the given rank.
[[nodiscard]] constexpr ImageAxes image_axes(ImageTensorLayout layout, std::size_t rank) noexcept {
  return layout == ImageTensorLayout::kPlanar
             ? ImageAxes{.height = rank - 2U, .width = rank - 1U, .channel = rank - 3U}
             : ImageAxes{.height = rank - 3U, .width = rank - 2U, .channel = rank - 1U};
}

[[nodiscard]] expected<void, Error> validate_image_tensor_outer_strides(
    const ImageTensorView& view) {
  const std::span<const std::int64_t> byte_strides{view.byte_strides.data(), view.rank};
  const bool is_planar = view.image_layout == ImageTensorLayout::kPlanar;
  const ImageAxes axes = image_axes(view.image_layout, view.rank);
  if (is_planar) {
    auto minimum_channel_stride = checked_stride_product(view.height, byte_strides[axes.height]);
    if (!minimum_channel_stride) {
      return make_unexpected(std::move(minimum_channel_stride).error());
    }
    if (byte_strides[axes.channel] < *minimum_channel_stride) {
      return invalid_argument("Vision tensor channel stride overlaps adjacent planes");
    }
  }
  if (view.rank == 4U) {
    auto minimum_batch_stride =
        checked_stride_product(is_planar ? view.channels : view.height,
                               byte_strides[is_planar ? axes.channel : axes.height]);
    if (!minimum_batch_stride) {
      return make_unexpected(std::move(minimum_batch_stride).error());
    }
    if (byte_strides[0] < *minimum_batch_stride) {
      return invalid_argument("Vision tensor batch stride overlaps adjacent images");
    }
  }
  return {};
}

[[nodiscard]] expected<void, Error> validate_image_tensor_storage(const ImageTensorView& view,
                                                                  std::size_t dtype_size) {
  const std::span<const std::int64_t> byte_strides{view.byte_strides.data(), view.rank};
  const bool is_planar = view.image_layout == ImageTensorLayout::kPlanar;
  const ImageAxes axes = image_axes(view.image_layout, view.rank);
  auto alignment = cvcuda_access_alignment(view.dtype, view.channels, view.image_layout);
  if (!alignment) {
    return make_unexpected(std::move(alignment).error());
  }
  auto valid_alignment = validate_access_alignment(
      view.data, byte_strides, axes.height, axes.channel, is_planar, *alignment);
  if (!valid_alignment) {
    return make_unexpected(std::move(valid_alignment).error());
  }

  const auto element_stride = static_cast<std::int64_t>(dtype_size);
  const auto column_stride = is_planar ? expected<std::int64_t, Error>{element_stride}
                                       : checked_stride_product(view.channels, element_stride);
  if (!column_stride) {
    return make_unexpected(std::move(column_stride).error());
  }
  if ((!is_planar && byte_strides[axes.channel] != element_stride) ||
      byte_strides[axes.width] != *column_stride) {
    return invalid_argument(is_planar ? "Vision tensor pixels must be tightly packed"
                                      : "Vision tensor pixels must be tightly interleaved");
  }
  auto minimum_row_stride = checked_stride_product(view.width, *column_stride);
  if (!minimum_row_stride) {
    return make_unexpected(std::move(minimum_row_stride).error());
  }
  if (byte_strides[axes.height] < *minimum_row_stride) {
    return invalid_argument("Vision tensor row stride overlaps adjacent rows");
  }
  auto valid_outer_strides = validate_image_tensor_outer_strides(view);
  if (!valid_outer_strides) {
    return make_unexpected(std::move(valid_outer_strides).error());
  }
  const std::int64_t outer_extent =
      view.rank == 3U ? (is_planar ? view.channels : view.height) : view.batch;
  auto tensor_byte_span = checked_stride_product(byte_strides[0], outer_extent);
  if (!tensor_byte_span || *tensor_byte_span > kMaxCvCudaTensorByteSpan) {
    return invalid_argument("Vision tensor byte span exceeds CV-CUDA's 32-bit limit");
  }
  return {};
}

}  // namespace

expected<ImageTensorView, Error> describe_image_tensor(
    const void* data, std::span<const std::int64_t> shape,
    std::span<const std::int64_t> byte_strides, DLDataType dtype, DLDevice device,
    TensorOperation operation, std::uint32_t max_batch_size, ImageTensorLayout image_layout) {
  if (data == nullptr) {
    return invalid_argument("Vision tensor data pointer is null");
  }
  if (device.device_type != kDLCUDA || device.device_id < 0) {
    return invalid_argument("Vision operators require CUDA device tensors");
  }
  if (shape.size() != 3U && shape.size() != 4U) {
    return invalid_argument("Vision image operators accept only rank-3 or rank-4 tensors");
  }
  if (byte_strides.size() != shape.size()) {
    return invalid_argument("Vision tensor byte strides must match the tensor rank");
  }
  if (max_batch_size == 0U || max_batch_size > kMaximumBatchSize) {
    return invalid_argument("Vision operator max_batch_size is outside the supported range");
  }
  if (!dtype_supported(dtype, operation)) {
    return invalid_argument("Vision tensor dtype is not supported by this operator");
  }
  if (!layout_supported(operation, image_layout)) {
    return invalid_argument("Vision tensor layout is not supported by this operator");
  }
  auto dtype_size = element_size(dtype);
  if (!dtype_size) {
    return make_unexpected(std::move(dtype_size).error());
  }

  ImageTensorView view{
      .data = data,
      .dtype = dtype,
      .device = device,
      .rank = static_cast<std::uint32_t>(shape.size()),
      .image_layout = image_layout,
  };
  std::ranges::copy(shape, view.shape.begin());
  std::ranges::copy(byte_strides, view.byte_strides.begin());

  const ImageAxes axes = image_axes(image_layout, shape.size());
  view.batch = shape.size() == 3U ? 1 : shape[0];
  view.height = shape[axes.height];
  view.width = shape[axes.width];
  view.channels = shape[axes.channel];

  if (std::ranges::any_of(shape, is_non_positive)) {
    return invalid_argument("Vision tensor dimensions must be positive");
  }
  if (view.batch > static_cast<std::int64_t>(max_batch_size)) {
    return invalid_argument("Vision tensor batch exceeds max_batch_size");
  }
  if (!channels_supported(dtype, operation, image_layout, view.channels)) {
    return invalid_argument("Vision tensor channel count is not supported by this operator");
  }
  if (std::ranges::any_of(byte_strides, is_non_positive)) {
    return invalid_argument("Vision tensor byte strides must be positive");
  }
  if (std::ranges::any_of(byte_strides, exceeds_cvcuda_byte_span)) {
    return invalid_argument("Vision tensor byte strides exceed CV-CUDA's 32-bit limit");
  }

  auto valid_storage = validate_image_tensor_storage(view, *dtype_size);
  if (!valid_storage) {
    return make_unexpected(std::move(valid_storage).error());
  }
  return view;
}

expected<ImageTensorView, Error> describe_image_tensor(const Tensor& tensor,
                                                       TensorOperation operation,
                                                       std::uint32_t max_batch_size,
                                                       ImageTensorLayout image_layout) {
  if (tensor.data() == nullptr) {
    return invalid_argument("Vision input tensor is empty");
  }
  auto byte_strides = fixed_tensor_byte_strides(tensor);
  if (!byte_strides) {
    return make_unexpected(std::move(byte_strides).error());
  }
  return describe_image_tensor(tensor.data(),
                               tensor.shape_span(),
                               byte_strides->span(),
                               tensor.dtype(),
                               tensor.device(),
                               operation,
                               max_batch_size,
                               image_layout);
}

expected<FixedTensorByteStrides, Error> fixed_tensor_byte_strides(const Tensor& tensor) {
  const std::span<const std::int64_t> shape = tensor.shape_span();
  if (shape.empty() || shape.size() > ImageTensorView::kMaxRank) {
    return invalid_argument("Vision tensor rank cannot produce fixed byte strides");
  }

  FixedTensorByteStrides result{.rank = static_cast<std::uint32_t>(shape.size())};
  const std::span<const std::int64_t> element_strides = tensor.dlpack_strides_span();
  if (element_strides.empty()) {
    auto contiguous = contiguous_byte_strides(shape, tensor.dtype());
    if (!contiguous) {
      return make_unexpected(std::move(contiguous).error());
    }
    result.values = *contiguous;
    return result;
  }
  if (element_strides.size() != shape.size()) {
    return invalid_argument("Vision tensor DLPack strides must match the tensor rank");
  }

  auto dtype_size = element_size(tensor.dtype());
  if (!dtype_size ||
      *dtype_size > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    return invalid_argument("Vision tensor dtype size cannot produce byte strides");
  }
  const std::int64_t byte_size = static_cast<std::int64_t>(*dtype_size);
  for (std::size_t index = 0U; index < element_strides.size(); ++index) {
    auto stride = checked_stride_product(element_strides[index], byte_size);
    if (!stride) {
      return make_unexpected(std::move(stride).error());
    }
    result.values[index] = *stride;
  }
  return result;
}

expected<std::array<std::int64_t, ImageTensorView::kMaxRank>, Error> contiguous_byte_strides(
    std::span<const std::int64_t> shape, DLDataType dtype) {
  if (shape.empty() || shape.size() > ImageTensorView::kMaxRank ||
      std::ranges::any_of(shape, is_non_positive)) {
    return invalid_argument("Vision tensor shape cannot produce contiguous strides");
  }
  auto dtype_size = element_size(dtype);
  if (!dtype_size ||
      *dtype_size > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
    return invalid_argument("Vision tensor dtype size cannot produce contiguous strides");
  }

  std::array<std::int64_t, ImageTensorView::kMaxRank> strides{};
  strides[shape.size() - 1U] = static_cast<std::int64_t>(*dtype_size);
  for (std::size_t index = shape.size() - 1U; index > 0U; --index) {
    auto stride = checked_stride_product(strides[index], shape[index]);
    if (!stride) {
      return make_unexpected(std::move(stride).error());
    }
    strides[index - 1U] = *stride;
  }
  return strides;
}

expected<std::uint64_t, Error> projected_image_output_byte_span(
    std::int32_t width, std::int32_t height, std::uint32_t max_batch_size,
    std::size_t max_element_size) noexcept {
  if (width <= 0 || height <= 0 || max_batch_size == 0U || max_batch_size > kMaximumBatchSize ||
      max_element_size == 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  std::uint64_t span = static_cast<std::uint64_t>(width);
  for (const std::uint64_t factor : {static_cast<std::uint64_t>(height),
                                     static_cast<std::uint64_t>(max_batch_size),
                                     std::uint64_t{4U},
                                     static_cast<std::uint64_t>(max_element_size)}) {
    if (span > std::numeric_limits<std::uint64_t>::max() / factor) {
      return make_unexpected(Error{ErrorCode::kArithmeticOverflow});
    }
    span *= factor;
  }
  if (span > static_cast<std::uint64_t>(kMaxCvCudaTensorByteSpan)) {
    return make_unexpected(Error{ErrorCode::kNotSupported});
  }
  return span;
}

expected<void, Error> validate_crop_rectangle(std::int64_t input_width, std::int64_t input_height,
                                              std::int32_t x, std::int32_t y, std::int32_t width,
                                              std::int32_t height) {
  if (input_width <= 0 || input_height <= 0 || x < 0 || y < 0 || width <= 0 || height <= 0) {
    return invalid_argument("CustomCropOp crop rectangle dimensions are invalid");
  }
  const std::int64_t crop_right = static_cast<std::int64_t>(x) + width;
  const std::int64_t crop_bottom = static_cast<std::int64_t>(y) + height;
  if (crop_right > input_width || crop_bottom > input_height) {
    return invalid_argument("CustomCropOp crop rectangle exceeds the input tensor bounds");
  }
  return {};
}

expected<nvcv::Tensor, Error> wrap_nvcv_tensor(const ImageTensorView& view) {
  return wrap_nvcv_tensor(view.data,
                          std::span<const std::int64_t>(view.shape.data(), view.rank),
                          std::span<const std::int64_t>(view.byte_strides.data(), view.rank),
                          view.dtype,
                          view.layout());
}

expected<nvcv::Tensor, Error> wrap_nvcv_tensor(const void* data,
                                               std::span<const std::int64_t> shape,
                                               std::span<const std::int64_t> byte_strides,
                                               DLDataType dtype, const char* layout) {
  if (data == nullptr) {
    return invalid_argument("Vision tensor data pointer is null");
  }
  if (shape.empty() || shape.size() > NVCV_TENSOR_MAX_RANK) {
    return invalid_argument("Vision tensor rank is not supported by NVCV");
  }
  if (byte_strides.size() != shape.size()) {
    return invalid_argument("Vision tensor byte strides must match the tensor rank");
  }
  if (layout == nullptr || layout[0] == '\0') {
    return invalid_argument("Vision tensor layout must not be empty");
  }
  if (std::ranges::any_of(shape, is_non_positive) ||
      std::ranges::any_of(byte_strides, is_non_positive)) {
    return invalid_argument("Vision tensor dimensions and byte strides must be positive");
  }
  if (std::ranges::any_of(byte_strides, exceeds_cvcuda_byte_span)) {
    return invalid_argument("Vision tensor byte strides exceed CV-CUDA's 32-bit limit");
  }

  auto nvcv_data_type = nvcv_dtype(dtype);
  if (!nvcv_data_type) {
    return make_unexpected(std::move(nvcv_data_type).error());
  }
  try {
    nvcv::TensorDataStridedCuda::Buffer buffer{};
    std::ranges::copy(byte_strides, buffer.strides);
    buffer.basePtr = reinterpret_cast<NVCVByte*>(const_cast<void*>(data));
    const nvcv::TensorShape tensor_shape(
        shape.data(), static_cast<std::int32_t>(shape.size()), layout);
    // CV-CUDA 0.18.x copies this descriptor synchronously and the wrapper has no cleanup
    // callback, so it neither owns nor releases the Holoscan backing allocation. Revalidate this
    // assumption, plus asynchronous metadata/workspace retention, for every new CV-CUDA operator
    // and supported CV-CUDA version before relying on callback-lifetime wrapper destruction.
    return nvcv::TensorWrapData(nvcv::TensorDataStridedCuda(tensor_shape, *nvcv_data_type, buffer));
  } catch (const nvcv::Exception& exception) {
    return make_unexpected(Error{
        ErrorCode::kFailure, std::string("CV-CUDA tensor wrapping failed: ") + exception.what()});
  }
}

expected<void, Error> prepare_nvcv_tensor_handles(std::size_t count) noexcept {
  if (count == 0 || count > 3) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  try {
    // NVCV 0.18 wraps/validates metadata without dereferencing the base address or querying CUDA.
    // The placeholder is never submitted to a GPU operation; destroy every wrapper before it
    // leaves scope. Revalidate when changing HOLOSCAN_VISION_CVCUDA_VERSION_MIN/MAX in
    // public/modules/vision/cmake/CVCUDAVersion.cmake.
    std::byte placeholder{};
    constexpr std::array<std::int64_t, 3> kShape{1, 1, 1};
    nvcv::TensorDataStridedCuda::Buffer buffer{};
    buffer.basePtr = reinterpret_cast<NVCVByte*>(&placeholder);
    std::fill_n(buffer.strides, 3, 1);
    const nvcv::TensorDataStridedCuda data{
        nvcv::TensorShape{kShape.data(), 3, "HWC"}, nvcv::TYPE_U8, buffer};
    std::array<nvcv::Tensor, 3> handles{};
    for (std::size_t index = 0; index != count; ++index) {
      handles[index] = nvcv::TensorWrapData(data);
    }
    return {};
  } catch (const nvcv::Exception& exception) {
    return make_unexpected(Error{exception.code() == nvcv::Status::ERROR_OUT_OF_MEMORY
                                     ? ErrorCode::kResourceExhausted
                                     : ErrorCode::kFailure});
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kFailure});
  }
}

}  // namespace holoscan::vision::detail
