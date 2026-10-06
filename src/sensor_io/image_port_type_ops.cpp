/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/schema/flatbuffer_schema_package.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "sensor_io/image_metadata_internal.hpp"
#include "sensor_io/image_metadata_storage.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <span>
#include <utility>

#include <holoscan/sensor_io/image_metadata.hpp>

namespace holoscan::detail {
namespace {

using ImagePayload = schema::DescriptorFlatBufferPayload<
    schema::Image, schema::flatbuffer_descriptor_metadata_capacity_v<schema::Image>>;

// plane_layouts arrives from the wire and lands in a std::vector, so its capacity is reserved
// during prepare and the bound comes from schema::kMaximumImagePlanes rather than a second constant
// here. Sharing it with the authoring path is what makes a layout that builds one that decodes.
using schema::kMaximumImagePlanes;

[[nodiscard]] expected<void, Error> prepare_image_sample(void* storage) noexcept {
  if (storage == nullptr ||
      reinterpret_cast<std::uintptr_t>(storage) % alignof(Sample<schema::ImageT>) != 0U) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  try {
    schema::ImageT image{};
    sensor_io::detail::prepare_image_metadata(image);
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
  sample.data.color_space = schema::ColorSpace_UNSPECIFIED;
  sample.data.significant_bits = 0U;
  sample.data.roi_offset_x = 0;
  sample.data.roi_offset_y = 0;
  // clear() keeps the capacity reserved during prepare, so the next decode still does not allocate.
  sample.data.plane_layouts.clear();
  sample.data.exposure_time_ns = 0;
  sample.data.gain = 0.0F;
  if (sample.data.header) {
    sample.data.header->frame_id.clear();
    sample.data.header->capture_timestamp_ns = 0;
    sample.data.header->device_sequence = 0U;
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
  // FlatBuffers verification establishes that these bytes are structurally an Image. It says
  // nothing about whether the values mean anything, so an external producer that never ran the
  // authoring encoder can present a negative dimension, an out-of-range enumerator, a negative
  // exposure, a NaN gain, or a plane layout contradicting its own encoding, and the sample slot
  // would carry it to a consumer looking exactly like a measurement. The verdict comes from the
  // same function the authoring path uses, against a view of this root rather than a copy of it,
  // and it is reduced to a bare code here because this path can neither allocate a diagnostic
  // string nor throw while building one.
  //
  // Taken before the first write below, so a rejected message leaves the slot exactly as the
  // previous reset left it rather than half updated. The plane-count bound is part of this verdict,
  // which is what lets the copy loop below stay inside the capacity prepare reserved.
  //
  // The Tensor is the one this message resolved to and the one the sample is about to own, so the
  // plane extents are bounded against the memory a consumer of this sample will actually address.
  // Bounding them here rather than trusting the producer is the point: a peer that computed its
  // offsets against a different allocation than the one it sent cannot be detected any later.
  if (schema::detail::validate_image_facts(schema::detail::image_facts(image, tensors[0])) !=
      schema::detail::ImageFactViolation::kNone) {
    return make_unexpected(Error{ErrorCode::kProtocolError});
  }
  sample.data.height = image.height();
  sample.data.width = image.width();
  sample.data.encoding = image.encoding();
  sample.data.color_space = image.color_space();
  sample.data.significant_bits = image.significant_bits();
  sample.data.roi_offset_x = image.roi_offset_x();
  sample.data.roi_offset_y = image.roi_offset_y();
  sample.data.exposure_time_ns = image.exposure_time_ns();
  sample.data.gain = image.gain();
  // Copied element by element rather than through assign(): a FlatBuffers vector of structs
  // iterates as pointers into the wire buffer, so the elements have to be dereferenced into the
  // sample rather than assigned from the range.
  const auto* plane_layouts = image.plane_layouts();
  sample.data.plane_layouts.clear();
  if (plane_layouts != nullptr) {
    for (flatbuffers::uoffset_t index = 0U; index < plane_layouts->size(); ++index) {
      sample.data.plane_layouts.push_back(*plane_layouts->Get(index));
    }
  }
  sample.data.header->capture_timestamp_ns = header == nullptr ? 0 : header->capture_timestamp_ns();
  sample.data.header->device_sequence = header == nullptr ? 0U : header->device_sequence();
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
      schema::flatbuffer_descriptor_metadata_capacity_v<schema::Image>>(
      PrepareImageSampleCallback{},
      ResetImageSampleCallback{},
      EncodeImageMetadataCallback{},
      AssignValidatedImageSampleCallback{});
  return &ops;
}

}  // namespace holoscan::detail
