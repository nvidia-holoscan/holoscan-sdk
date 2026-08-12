/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/schema/flatbuffer_schema_package.hpp>

#include "schema/image_metadata_internal.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <span>
#include <utility>

#include <holoscan/schema/image_metadata.hpp>

namespace holoscan::detail {
namespace {

using ImagePayload = schema::DescriptorFlatBufferPayload<
    schema::Image, schema::flatbuffer_default_descriptor_metadata_capacity_v<schema::Image>>;

[[nodiscard]] expected<void, Error> prepare_image_sample(void* storage) noexcept {
  if (storage == nullptr ||
      reinterpret_cast<std::uintptr_t>(storage) % alignof(Sample<schema::ImageT>) != 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  try {
    schema::ImageT image{};
    image.header = std::make_shared<schema::HeaderT>();
    image.header->stamp = std::make_shared<schema::TimeT>();
    image.header->frame_id.reserve(ImagePayload::metadata_capacity);
    image.data = std::make_shared<Tensor>();
    std::construct_at(
        static_cast<Sample<schema::ImageT>*>(storage), std::move(image), SampleMetadata{});
    return {};
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kFailure});
  }
}

void reset_image_sample(void* storage) noexcept {
  auto& sample = *static_cast<Sample<schema::ImageT>*>(storage);
  sample.metadata = {};
  sample.data.height = 0;
  sample.data.width = 0;
  sample.data.encoding = schema::ImageEncoding_CUSTOM;
  if (sample.data.header) {
    sample.data.header->frame_id.clear();
    if (sample.data.header->stamp) {
      sample.data.header->stamp->time_ns = 0;
      sample.data.header->stamp->acqtime_ns = 0;
    }
  }
  if (sample.data.data) {
    *sample.data.data = Tensor{};
  }
}

[[nodiscard]] expected<std::size_t, Error> encode_image_metadata(
    const schema::ImageT& value, const schema::DescriptorTensorReferences tensors,
    const std::span<std::byte> destination) noexcept {
  const Tensor* tensor = tensors.at(0U);
  if (tensors.size() != 1U || tensor == nullptr) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  return schema::detail::encode_image_metadata(value, *tensor, destination);
}

[[nodiscard]] expected<void, Error> assign_validated_image_sample(
    void* storage, const schema::Image& image, const std::span<Tensor> tensors,
    const SampleMetadata metadata) noexcept {
  if (storage == nullptr || tensors.size() != 1U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  auto& sample = *static_cast<Sample<schema::ImageT>*>(storage);
  if (!sample.data.header || !sample.data.header->stamp || !sample.data.data) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  const schema::Header* header = image.header();
  const schema::Time* stamp = header == nullptr ? nullptr : header->stamp();
  const auto* frame_id = header == nullptr ? nullptr : header->frame_id();
  if (frame_id != nullptr && frame_id->size() > ImagePayload::metadata_capacity) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  sample.data.height = image.height();
  sample.data.width = image.width();
  sample.data.encoding = image.encoding();
  sample.data.header->stamp->time_ns = stamp == nullptr ? 0 : stamp->time_ns();
  sample.data.header->stamp->acqtime_ns = stamp == nullptr ? 0 : stamp->acqtime_ns();
  if (frame_id == nullptr) {
    sample.data.header->frame_id.clear();
  } else {
    sample.data.header->frame_id.assign(frame_id->c_str(), frame_id->size());
  }
  *sample.data.data = std::move(tensors[0]);
  sample.metadata = metadata;
  return {};
}

struct PrepareImageSampleCallback {
  [[nodiscard]] expected<void, Error> operator()(void* storage) const noexcept {
    return prepare_image_sample(storage);
  }
};

struct ResetImageSampleCallback {
  void operator()(void* storage) const noexcept { reset_image_sample(storage); }
};

struct EncodeImageMetadataCallback {
  [[nodiscard]] expected<std::size_t, Error> operator()(
      const schema::ImageT& value, schema::DescriptorTensorReferences tensors,
      std::span<std::byte> destination) const noexcept {
    return encode_image_metadata(value, tensors, destination);
  }
};

struct AssignValidatedImageSampleCallback {
  [[nodiscard]] expected<void, Error> operator()(void* storage, const schema::Image& image,
                                                 std::span<Tensor> tensors,
                                                 SampleMetadata metadata) const noexcept {
    return assign_validated_image_sample(storage, image, tensors, metadata);
  }
};

}  // namespace

template <>
const RuntimePortTypeOps* runtime_port_type_ops<schema::ImageT>() noexcept {
  static const RuntimePortTypeOps ops = schema::make_descriptor_tensor_port_type_ops<
      schema::ImageT,
      schema::Image,
      schema::flatbuffer_default_descriptor_metadata_capacity_v<schema::Image>>(
      PrepareImageSampleCallback{},
      ResetImageSampleCallback{},
      EncodeImageMetadataCallback{},
      AssignValidatedImageSampleCallback{});
  return &ops;
}

}  // namespace holoscan::detail
