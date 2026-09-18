/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"
#include "holoscan/holoscan_sensor_io_export.h"
#include "holoscan/schema/descriptor_flatbuffer_payload.hpp"

#include "image_generated.h"

namespace holoscan::schema {

/// @brief Most planes an Image may describe, on either the authoring or the receive path.
///
/// The most subdivided encoding the schema defines is three-plane I420, so four leaves room for one
/// more without letting a malformed or hostile producer size a heap allocation on the receive path.
/// Authoring and decoding share this bound so a layout that builds cannot fail to decode.
constexpr std::size_t kMaximumImagePlanes = 4U;

/** @brief Options encoded into serialized Image metadata. */
struct ImageMetadataOptions {
  /// Image height in pixels.
  int32_t height = 0;
  /// Image width in pixels.
  int32_t width = 0;
  /// Image pixel encoding.
  ImageEncoding encoding = ImageEncoding_CUSTOM;
  /// Instant the measurement was taken, in nanoseconds; zero when the producer has none.
  int64_t capture_timestamp_ns = 0;
  /// Producer-defined coordinate frame name.
  std::string frame_id;
  /// Monotonic per-source sample number as the device reported it; zero when it has none. Distinct
  /// from the runtime's publication counter in \c SampleMetadata::sequence, which cannot show a
  /// sensor-side drop.
  uint64_t device_sequence = 0;
  /// Color primaries and transfer function the samples are encoded against.
  ColorSpace color_space = ColorSpace_UNSPECIFIED;
  /// Bits of real precision per component; zero when it equals the container width.
  uint8_t significant_bits = 0;
  /// Column of the first pixel within the full sensor frame.
  int32_t roi_offset_x = 0;
  /// Row of the first pixel within the full sensor frame.
  int32_t roi_offset_y = 0;
  /// Offset, row stride, and extent of each plane, in the plane order the encoding defines.
  /// Required, with one entry per plane, for every multi-planar encoding; empty for single-plane
  /// encodings, where the Tensor strides are the one authority for the one plane. A row stride
  /// alone cannot locate a plane inside the single Tensor an Image carries, which is why each entry
  /// also states where the plane begins and how far it runs, and why omitting the vector for a
  /// multi-planar encoding is rejected rather than read as deferral to those strides. All planes
  /// live inside that one Tensor allocation; separately allocated planes are out of scope. That is
  /// enforced rather than assumed: a plane whose offset plus extent reaches past the Tensor's
  /// addressable span is rejected here and on the receive path alike.
  std::vector<ImagePlaneLayout> plane_layouts;
  /// Sensor integration time in nanoseconds; zero when not reported.
  int64_t exposure_time_ns = 0;
  /// Linear sensor gain applied to the frame; zero when not reported. One combined figure rather
  /// than an analog stage awaiting a digital sibling, which is why it is not named analog_gain.
  float gain = 0.0F;
};

/** @brief Non-owning view of parsed Image metadata.
 * @warning The image pointer remains valid only while the source metadata bytes remain alive and
 * unmodified. */
struct ImageMetadataView {
  /// Parsed Image root pointing into caller-owned metadata storage.
  const Image* image = nullptr;
};

/** @brief Tensor view information extracted from serialized Image metadata. */
struct ImageTensorMetadata {
  /// Maximum Image Tensor rank represented inline.
  static constexpr std::size_t kMaxRank = 8;

  /// Tensor rank.
  std::uint32_t rank = 0;
  /// DLPack element type.
  ::DLDataType dtype{};
  /// DLPack device type and ordinal.
  ::DLDevice device{};
  /// Whether dlpack_strides contains explicit element strides.
  bool has_explicit_strides = false;
  /// Logical Tensor byte size.
  std::uint64_t byte_size = 0;
  /// Tensor shape; only entries below rank are valid.
  std::array<int64_t, kMaxRank> shape{};
  /// DLPack element strides; only entries below rank are valid when has_explicit_strides is true.
  std::array<int64_t, kMaxRank> dlpack_strides{};
};

/// @brief Serialize bounded descriptor-backed Image metadata for a Tensor.
/// @param tensor Tensor whose shape, type, device, and strides are encoded.
/// @param options Image dimensions, encoding, timestamps, and frame identifier.
/// @return Serialized FlatBuffer bytes on success, or Error for incompatible Tensor metadata.
/// @note Tensor data bytes are never serialized by this helper, including while a
/// \ref WithTensorMaterialization or \ref WithTensorGather scope is active. The returned metadata
/// is bounded by \c flatbuffer_descriptor_metadata_capacity_v<Image>; an oversized frame
/// identifier or future metadata returns \c ErrorCode::kResourceExhausted.
/// @note Returns \c ErrorCode::kInvalidArgument for a descriptor the schema gives no meaning to: a
/// non-positive height or width, an \c encoding or \c color_space outside its enumeration, a
/// negative \c roi_offset_x, \c roi_offset_y, or \c exposure_time_ns, a \c gain that is
/// negative or not a real number, or \c plane_layouts disagreeing with the encoding. The same
/// checks run on the receive path against a decoded root, so a value rejected here is also one a
/// peer cannot deliver.
/// @note \c plane_layouts is required for every multi-planar encoding, with exactly one entry per
/// plane; an empty vector is rejected for those encodings rather than read as deferral to the
/// Tensor strides, which cannot supply an independent base per plane. A single-plane encoding must
/// leave it empty, while \c ImageEncoding_CUSTOM names no plane count and so accepts any the
/// producer defines up to \ref kMaximumImagePlanes.
/// @note Every declared plane is checked for internal consistency regardless of encoding: a zero
/// \c row_stride_bytes describes no rows, an \c offset_bytes plus \c size_bytes that overflows
/// describes no range, and two planes claiming the same bytes cannot both be right. Where the
/// encoding also fixes each plane's dimensions and bytes per sample -- the multi-planar encodings
/// do, applying the ceiling rule \c image.fbs states for a subsampled odd extent -- the
/// \c row_stride_bytes must cover one packed row and the \c size_bytes must cover the rows
/// \c height requires at that stride.
/// @note What the descriptor says it occupies is bounded against what the Tensor can address. That
/// span is measured from the Tensor's first logical element and accounts for its strides, so a
/// padded backing is bounded by what it reaches rather than by the product of its extents. A
/// declared plane whose \c offset_bytes plus \c size_bytes passes the end of that span is rejected,
/// and so is a single-plane encoding whose \c width, \c height, and byte organization require more
/// than the span holds. A Bayer encoding reaches that bound through the element type it defers its
/// container to. The receive path applies the same verdict, so a descriptor accepted here is one a
/// peer can deliver.
/// @note A known encoding may not contradict the Tensor carrying it. Where a single-plane encoding
/// names its own element type, the Tensor must carry either that type or an opaque \c uint8 byte
/// buffer, so a \c float32 Tensor labelled \c MONO16 is rejected rather than left for a consumer to
/// resolve. \c significant_bits is bounded by the container the encoding gives a component rather
/// than by the Tensor element type, which is what lets P010 report the 10 bits it is defined to
/// carry while being delivered as a flat byte buffer. \c ImageEncoding_CUSTOM states no
/// organization and keeps the weaker bound of the widest component any element type provides.
/// @note The Tensor element must be scalar: channels belong in the Tensor shape, and every encoding
/// except \c ImageEncoding_CUSTOM already fixes the component organization of a sample, so a
/// vectorized element states it a second time with nothing to relate the two. A four-lane \c uint8
/// element labelled \c RGB8 leaves both the frame width and the meaning of those lanes undecided,
/// so a lane count other than one is rejected for every encoding but \c CUSTOM. Byte accounting is
/// lane-aware either way: a vectorized element counts at its full width in the frame and plane
/// bounds. Interleaved channels are expressed as a trailing shape axis, as \c (height, width, 4)
/// for
/// \c RGBA8, which is the form these bounds are derived against.
[[nodiscard]] HOLOSCAN_SENSOR_IO_EXPORT expected<std::vector<std::byte>, Error>
build_image_metadata(const holoscan::Tensor& tensor, const ImageMetadataOptions& options);

/// @brief Parse serialized Image metadata without copying its storage.
/// @param metadata Serialized FlatBuffer metadata bytes.
/// @return Non-owning Image view on success, or Error for empty or invalid metadata.
[[nodiscard]] HOLOSCAN_SENSOR_IO_EXPORT expected<ImageMetadataView, Error> parse_image_metadata(
    std::span<const std::byte> metadata);

/// @brief Extract descriptor-backed Tensor metadata from a parsed Image view.
/// @param view Parsed Image metadata view.
/// @return Tensor metadata on success, or Error for missing, invalid, overflowing, or embedded-data
/// fields.
/// @note This helper accepts metadata-only Image buffers. A nonempty Tensor data vector is rejected
/// so descriptor-backed paths cannot silently carry a second materialized payload representation.
[[nodiscard]] HOLOSCAN_SENSOR_IO_EXPORT expected<ImageTensorMetadata, Error> image_tensor_metadata(
    const ImageMetadataView& view);

}  // namespace holoscan::schema
