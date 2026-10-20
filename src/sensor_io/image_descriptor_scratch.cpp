/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/sensor_io/image_descriptor_scratch.hpp>

#include <new>
#include <utility>

#include "sensor_io/image_metadata_storage.hpp"

namespace holoscan::sensor_io {

expected<ImageDescriptorScratch, Error> ImageDescriptorScratch::create() noexcept {
  try {
    ImageDescriptorScratch scratch;
    detail::prepare_image_metadata(scratch.image_);
    scratch.header_ = scratch.image_.header;
    scratch.stamp_ = scratch.header_->stamp;
    scratch.reset();
    return scratch;
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  } catch (...) {
    return make_unexpected(Error{ErrorCode::kFailure});
  }
}

expected<void, Error> ImageDescriptorScratch::copy_metadata_from(
    const schema::ImageT& source) noexcept {
  // Check protocol bounds as well as actual capacity; an allocator may reserve extra capacity.
  if (source.plane_layouts.size() > schema::kMaximumImagePlanes ||
      source.plane_layouts.size() > image_.plane_layouts.capacity() ||
      (source.header && (source.header->frame_id.size() >
                             schema::flatbuffer_descriptor_metadata_capacity_v<schema::Image> ||
                         source.header->frame_id.size() > header_->frame_id.capacity()))) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  if (&source == &image_) {
    return {};
  }
  detail::copy_image_scalar_metadata(image_, source);
  image_.plane_layouts.assign(source.plane_layouts.begin(), source.plane_layouts.end());
  if (source.header) {
    header_->frame_id.assign(source.header->frame_id);
    header_->capture_timestamp_ns = source.header->capture_timestamp_ns;
    header_->device_sequence = source.header->device_sequence;
    if (source.header->stamp) {
      stamp_->time_ns = source.header->stamp->time_ns;
      stamp_->acqtime_ns = source.header->stamp->acqtime_ns;
      header_->stamp = stamp_;
    } else {
      header_->stamp.reset();
    }
    image_.header = header_;
  } else {
    image_.header.reset();
  }
  return {};
}

void ImageDescriptorScratch::reset() noexcept {
  detail::copy_image_scalar_metadata(image_, schema::ImageT{});
  image_.plane_layouts.clear();
  header_->frame_id.clear();
  header_->capture_timestamp_ns = 0;
  header_->device_sequence = 0;
  stamp_->time_ns = 0;
  stamp_->acqtime_ns = 0;
  header_->stamp.reset();
  image_.header.reset();
}

}  // namespace holoscan::sensor_io
