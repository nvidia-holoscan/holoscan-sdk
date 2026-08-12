/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "holoscan/schema/image_metadata.hpp"

#include "schema/image_metadata_internal.hpp"

#include <flatbuffers/flatbuffer_builder.h>
#include <flatbuffers/verifier.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <utility>
#include <vector>

#include "core/domain/tensor_layout_validation.hpp"
#include "header_generated.h"
#include "holoscan/schema/flatbuffer_schema_package.hpp"
#include "holoscan/schema/tensor_materialization.hpp"
#include "tensor_generated.h"
#include "time_generated.h"

namespace holoscan::schema {

namespace {

const std::uint8_t* metadata_bytes(std::span<const std::byte> metadata) {
  return reinterpret_cast<const std::uint8_t*>(metadata.data());
}

expected<void, Error> validate_options(const ImageMetadataOptions& options) {
  if (options.height <= 0 || options.width <= 0) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Image metadata height and width must be > 0"});
  }
  return {};
}

expected<void, Error> validate_image_encoding(const ImageEncoding encoding) {
  if (encoding < ImageEncoding_MIN || encoding > ImageEncoding_MAX) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Image metadata encoding is invalid"});
  }
  return {};
}

[[nodiscard]] expected<std::uint64_t, Error> dtype_item_size_bytes(const ::DLDataType& dtype) {
  if (dtype.bits == 0 || dtype.lanes == 0) {
    return make_unexpected(
        Error{ErrorCode::kCodecError, "Image Tensor metadata dtype bits/lanes must be nonzero"});
  }
  const std::uint64_t bits = dtype.bits;
  const std::uint64_t lanes = dtype.lanes;
  if (bits > (std::numeric_limits<std::uint64_t>::max() - 7U) / lanes) {
    return make_unexpected(
        Error{ErrorCode::kCodecError, "Image Tensor metadata item size overflows uint64"});
  }
  return (bits * lanes + 7U) / 8U;
}

[[nodiscard]] expected<std::uint64_t, Error> infer_byte_size(const ImageTensorMetadata& metadata,
                                                             const std::uint64_t item_size_bytes) {
  std::uint64_t element_count = 1U;
  for (std::uint32_t i = 0; i < metadata.rank; ++i) {
    if (metadata.shape[i] < 0 ||
        (metadata.has_explicit_strides && metadata.dlpack_strides[i] < 0)) {
      return make_unexpected(
          Error{ErrorCode::kCodecError, "Image Tensor metadata shape/strides contain negative"});
    }
    const auto dimension = static_cast<std::uint64_t>(metadata.shape[i]);
    if (dimension != 0U && element_count > std::numeric_limits<std::uint64_t>::max() / dimension) {
      return make_unexpected(
          Error{ErrorCode::kCodecError, "Image Tensor metadata element count overflows uint64"});
    }
    element_count *= dimension;
  }
  if (item_size_bytes != 0U &&
      element_count > std::numeric_limits<std::uint64_t>::max() / item_size_bytes) {
    return make_unexpected(
        Error{ErrorCode::kCodecError, "Image Tensor metadata byte size overflows uint64"});
  }
  return element_count * item_size_bytes;
}

[[nodiscard]] flatbuffers::Offset<Header> create_header_metadata(
    flatbuffers::FlatBufferBuilder& builder, const HeaderT* header) {
  if (header == nullptr) {
    return 0;
  }
  const auto stamp = header->stamp
                         ? CreateTime(builder, header->stamp->time_ns, header->stamp->acqtime_ns)
                         : flatbuffers::Offset<Time>{};
  const auto frame_id = header->frame_id.empty() ? flatbuffers::Offset<flatbuffers::String>{}
                                                 : builder.CreateString(header->frame_id);
  return CreateHeader(builder, stamp, frame_id);
}

}  // namespace

namespace detail {

expected<std::size_t, Error> encode_image_metadata(
    const ImageT& value, const std::span<std::byte> destination) noexcept {
  if (!value.data) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "Image metadata Tensor is missing"});
  }
  return encode_image_metadata(value, *value.data, destination);
}

expected<std::size_t, Error> encode_image_metadata(
    const ImageT& value, const holoscan::Tensor& tensor,
    const std::span<std::byte> destination) noexcept {
  if (destination.empty() || reinterpret_cast<std::uintptr_t>(destination.data()) %
                                     alignof(flatbuffers::largest_scalar_t) !=
                                 0U) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Image metadata storage is empty or unaligned"});
  }
  if (value.height <= 0 || value.width <= 0) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Image metadata height and width must be > 0"});
  }
  auto encoding_status = validate_image_encoding(value.encoding);
  if (!encoding_status) {
    return make_unexpected(std::move(encoding_status).error());
  }
  try {
    BoundedSpanAllocator allocator{destination};
    flatbuffers::FlatBufferBuilder builder{destination.size(), &allocator, false};
    const auto tensor_metadata = create_tensor_metadata(builder, tensor);
    if (!tensor_metadata) {
      return make_unexpected(std::move(tensor_metadata).error());
    }
    const auto header = create_header_metadata(builder, value.header.get());
    const auto image =
        CreateImage(builder, header, value.height, value.width, value.encoding, *tensor_metadata);
    FinishImageBuffer(builder, image);
    const auto size = builder.GetSize();
    if (size == 0U || size > destination.size()) {
      return make_unexpected(
          Error{ErrorCode::kResourceExhausted, "Image metadata exceeds bounded storage"});
    }
    std::memmove(destination.data(), builder.GetBufferPointer(), size);
    return size;
  } catch (const std::bad_alloc&) {
    return make_unexpected(
        Error{ErrorCode::kResourceExhausted, "Image metadata exceeds bounded storage"});
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kCodecError, "Image metadata encoding failed"});
  }
}

}  // namespace detail

expected<std::vector<std::byte>, Error> build_image_metadata(const holoscan::Tensor& tensor,
                                                             const ImageMetadataOptions& options) {
  auto valid = validate_options(options);
  if (!valid) {
    return make_unexpected(valid.error());
  }
  auto encoding_status = validate_image_encoding(options.encoding);
  if (!encoding_status) {
    return make_unexpected(std::move(encoding_status).error());
  }

  ImageT image{};
  image.height = options.height;
  image.width = options.width;
  image.encoding = options.encoding;
  image.header = std::make_shared<HeaderT>();
  image.header->stamp = std::make_shared<TimeT>();
  image.header->stamp->time_ns = options.time_ns;
  image.header->stamp->acqtime_ns = options.acqtime_ns;
  image.header->frame_id = options.frame_id;
  image.data = std::make_shared<holoscan::Tensor>(tensor);

  alignas(std::max_align_t)
      std::array<std::byte, flatbuffer_default_descriptor_metadata_capacity_v<Image>>
          storage{};
  auto encoded = detail::encode_image_metadata(image, storage);
  if (!encoded) {
    return make_unexpected(std::move(encoded).error());
  }
  return std::vector<std::byte>(storage.begin(), storage.begin() + encoded.value());
}

expected<ImageMetadataView, Error> parse_image_metadata(std::span<const std::byte> metadata) {
  if (metadata.empty()) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Image metadata FlatBuffer is empty"});
  }

  flatbuffers::Verifier verifier(metadata_bytes(metadata), metadata.size());
  if (!VerifyImageBuffer(verifier)) {
    return make_unexpected(
        Error{ErrorCode::kCodecError, "Image metadata FlatBuffer verification failed"});
  }

  const Image* image = GetImage(metadata_bytes(metadata));
  if (image == nullptr) {
    return make_unexpected(Error{ErrorCode::kCodecError, "Image metadata root is null"});
  }
  return ImageMetadataView{image};
}

expected<ImageTensorMetadata, Error> image_tensor_metadata(const ImageMetadataView& view) {
  if (view.image == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument, "Image metadata view is null"});
  }
  const Tensor* tensor = view.image->data();
  if (tensor == nullptr) {
    return make_unexpected(Error{ErrorCode::kCodecError, "Image metadata Tensor is missing"});
  }
  const auto* shape = tensor->shape();
  if (shape == nullptr) {
    return make_unexpected(Error{ErrorCode::kCodecError, "Image metadata Tensor shape is missing"});
  }
  const auto* dtype = tensor->dtype();
  if (dtype == nullptr) {
    return make_unexpected(Error{ErrorCode::kCodecError, "Image metadata Tensor dtype is missing"});
  }
  const auto* device = tensor->device();
  if (device == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kCodecError, "Image metadata Tensor device is missing"});
  }

  const std::uint32_t rank = tensor->ndim();
  if (rank != shape->size()) {
    return make_unexpected(
        Error{ErrorCode::kCodecError, "Image metadata Tensor ndim does not match shape size"});
  }
  if (rank > ImageTensorMetadata::kMaxRank) {
    return make_unexpected(
        Error{ErrorCode::kCodecError, "Image metadata Tensor rank exceeds descriptor limit"});
  }

  ImageTensorMetadata metadata{};
  metadata.rank = rank;
  metadata.dtype = ::DLDataType{.code = static_cast<std::uint8_t>(dtype->code()),
                                .bits = dtype->bits(),
                                .lanes = dtype->lanes()};
  metadata.device = ::DLDevice{.device_type = static_cast<::DLDeviceType>(device->device_type()),
                               .device_id = device->device_id()};
  if (!holoscan::detail::valid_tensor_dtype(metadata.dtype)) {
    return make_unexpected(Error{ErrorCode::kCodecError, "Image metadata Tensor dtype is invalid"});
  }
  switch (metadata.device.device_type) {
    case kDLCPU:
    case kDLCUDA:
    case kDLCUDAHost:
    case kDLCUDAManaged:
      break;
    default:
      return make_unexpected(
          Error{ErrorCode::kCodecError, "Image metadata Tensor device is unsupported"});
  }
  if (metadata.device.device_id < 0) {
    return make_unexpected(
        Error{ErrorCode::kCodecError, "Image metadata Tensor device id is negative"});
  }

  for (std::uint32_t i = 0; i < rank; ++i) {
    metadata.shape[i] = shape->Get(i);
  }

  const auto* strides = tensor->strides();
  if (strides != nullptr) {
    if (strides->size() != rank) {
      return make_unexpected(
          Error{ErrorCode::kCodecError, "Image metadata Tensor strides size does not match rank"});
    }
    metadata.has_explicit_strides = true;
    for (std::uint32_t i = 0; i < rank; ++i) {
      metadata.dlpack_strides[i] = strides->Get(i);
    }
  }

  auto item_size_bytes = dtype_item_size_bytes(metadata.dtype);
  if (!item_size_bytes) {
    return make_unexpected(std::move(item_size_bytes).error());
  }
  auto byte_size = infer_byte_size(metadata, item_size_bytes.value());
  if (!byte_size) {
    return make_unexpected(std::move(byte_size).error());
  }
  const auto* data = tensor->data();
  if (data != nullptr && data->size() > 0U) {
    return make_unexpected(
        Error{ErrorCode::kCodecError, "descriptor-backed Image metadata contains Tensor bytes"});
  }
  metadata.byte_size = byte_size.value();
  return metadata;
}

}  // namespace holoscan::schema
