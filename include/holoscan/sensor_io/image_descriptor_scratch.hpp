/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

#include <holoscan/holoscan_sensor_io_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

#include "image_generated.h"

namespace holoscan::sensor_io {

/// @brief Prepare reusable, metadata-only Image storage for callback-local edits.
///
/// Creation reserves the same plane and frame-ID bounds as prepared Runtime Image receive slots.
/// Copying metadata preserves optional Header/Time presence, never copies a Tensor, and allocates
/// nothing. Container bounds do not guarantee the complete descriptor fits the wire-size budget;
/// publication still validates metadata and the separately supplied output Tensor.
/// @warning Do not replace the prepared Header/Time owners, populate image().data, or grow a string
/// or vector beyond its reserved capacity. Do not assign an owning ImageT into image(). Copies of
/// metadata into this storage do not authorize mutation of a borrowed input descriptor.
class HOLOSCAN_SENSOR_IO_EXPORT ImageDescriptorScratch {
 public:
  /// @brief Allocate bounded metadata storage during cold initialization.
  /// @return Prepared scratch, or a typed allocation/setup error.
  [[nodiscard]] static expected<ImageDescriptorScratch, Error> create() noexcept;

  /// @brief Reject copying prepared storage.
  ImageDescriptorScratch(const ImageDescriptorScratch&) = delete;
  /// @brief Reject copy assignment of prepared storage.
  ImageDescriptorScratch& operator=(const ImageDescriptorScratch&) = delete;
  /// @brief Transfer prepared storage; the source may only be destroyed or reassigned.
  ImageDescriptorScratch(ImageDescriptorScratch&&) noexcept = default;
  /// @brief Transfer prepared storage, releasing this object's previous metadata storage.
  ImageDescriptorScratch& operator=(ImageDescriptorScratch&&) noexcept = default;
  /// @brief Release prepared metadata storage.
  ~ImageDescriptorScratch() = default;

  /// @brief Copy every non-Tensor field without retaining source ownership or allocating.
  /// @param source Descriptor whose optional metadata presence and values are preserved.
  /// @return Success, or kResourceExhausted before mutation if a source container exceeds a bound.
  /// @note Missing source.data is permitted. The resulting image().data is always null.
  [[nodiscard]] expected<void, Error> copy_metadata_from(const schema::ImageT& source) noexcept;

  /// @brief Clear metadata and optional presence while retaining all prepared capacity.
  void reset() noexcept;

  /// @brief Return prepared metadata for bounded edits and publication with a separate Tensor.
  /// @return Mutable metadata descriptor; observe the class storage-preservation preconditions.
  [[nodiscard]] schema::ImageT& image() noexcept { return image_; }
  /// @brief Return prepared metadata for inspection or publication with a separate Tensor.
  /// @return Immutable metadata descriptor.
  [[nodiscard]] const schema::ImageT& image() const noexcept { return image_; }

 private:
  ImageDescriptorScratch() = default;

  schema::ImageT image_;
  std::shared_ptr<schema::HeaderT> header_;
  std::shared_ptr<schema::TimeT> stamp_;
};

}  // namespace holoscan::sensor_io
