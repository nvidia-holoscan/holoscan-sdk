/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "holoscan/sensor_io/image_metadata.hpp"

#include "sensor_io/image_metadata_internal.hpp"

#include <flatbuffers/flatbuffer_builder.h>
#include <flatbuffers/verifier.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/domain/tensor_layout_validation.hpp"
#include "header_generated.h"
#include "holoscan/schema/flatbuffer_schema_package.hpp"
#include "holoscan/schema/tensor_materialization.hpp"
#include "holoscan/sensor_io/sensor_schema_package.hpp"
#include "tensor_generated.h"
#include "time_generated.h"

namespace holoscan::schema {

namespace {

const std::uint8_t* metadata_bytes(std::span<const std::byte> metadata) {
  return reinterpret_cast<const std::uint8_t*>(metadata.data());
}

/// @brief Planes an encoding stores, or absence when only the producer knows the layout.
///
/// Every case is listed rather than defaulted so that adding an encoding to the schema fails to
/// compile here instead of silently inheriting the single-plane answer.
[[nodiscard]] std::optional<std::size_t> encoding_plane_count(
    const ImageEncoding encoding) noexcept {
  switch (encoding) {
    case ImageEncoding_CUSTOM:
      // A custom encoding names no layout, so any plane vector is the producer's to define.
      return std::nullopt;
    case ImageEncoding_R8_G8_B8:
    case ImageEncoding_B8_G8_R8:
    case ImageEncoding_I420:
      return 3U;
    case ImageEncoding_NV12:
    case ImageEncoding_P010:
      return 2U;
    case ImageEncoding_RGB8:
    case ImageEncoding_RGBA8:
    case ImageEncoding_BGR8:
    case ImageEncoding_BGRA8:
    case ImageEncoding_MONO8:
    case ImageEncoding_MONO16:
    case ImageEncoding_MONO32:
    case ImageEncoding_MONO32F:
    case ImageEncoding_YUYV:
    case ImageEncoding_UYVY:
    case ImageEncoding_RGB32F:
    case ImageEncoding_RGBA32F:
    case ImageEncoding_YUV32F:
    case ImageEncoding_RGBA1010102:
    case ImageEncoding_BAYER_RGGB:
    case ImageEncoding_BAYER_BGGR:
    case ImageEncoding_BAYER_GRBG:
    case ImageEncoding_BAYER_GBRG:
      return 1U;
  }
  return std::nullopt;
}

/// @brief Packed row width and active row count of one plane, where the encoding fixes both.
struct ImagePlaneGeometry {
  std::uint64_t row_bytes = 0U;  ///< Bytes one active row occupies without row padding.
  std::uint64_t rows = 0U;       ///< Active rows the plane stores.
};

/// @brief Samples a plane holds along a dimension subsampled by two, rounded up.
///
/// image.fbs fixes the direction rather than leaving it to the producer: an odd extent rounds up,
/// so the final luma column or row keeps a chroma group of its own instead of losing the samples
/// that describe it. Rounding down is the cheaper rule and the wrong one, because the group it
/// discards is the only description of that edge. The caller has already rejected a non-positive
/// extent, so the widening to unsigned here cannot produce a wrapped value.
[[nodiscard]] constexpr std::uint64_t subsampled_extent(const std::int32_t extent) {
  return (static_cast<std::uint64_t>(extent) + 1U) / 2U;
}

/// @brief Geometry of one plane, or absence when the encoding does not determine it.
///
/// Only the multi-planar encodings appear, because they are the only encodings that carry plane
/// layouts at all and this function exists to size those records. A single-plane encoding must
/// leave the vector empty and a custom one is the producer's to define, so neither has a plane
/// whose row width could be asked for here.
///
/// Absence from this table is not a statement that the encoding fixes nothing. Several single-plane
/// entries do fix their bytes per pixel, and \ref encoding_packed_row_bytes derives a whole-frame
/// bound from them. The entries that fix nothing derivable are \c ImageEncoding_CUSTOM, which names
/// no sample organization, and the MONO and Bayer entries, which defer their sample width to the
/// Tensor element type.
///
/// Every case is listed for the same reason \ref encoding_plane_count lists them: a new encoding
/// should fail to compile here rather than silently acquire the unchecked answer.
[[nodiscard]] std::optional<ImagePlaneGeometry> encoding_plane_geometry(
    const ImageEncoding encoding, const std::size_t plane, const std::int32_t width,
    const std::int32_t height) noexcept {
  const auto full_width = static_cast<std::uint64_t>(width);
  const auto full_height = static_cast<std::uint64_t>(height);
  switch (encoding) {
    case ImageEncoding_R8_G8_B8:
    case ImageEncoding_B8_G8_R8:
      // Three co-sited 8-bit planes, each carrying the whole frame.
      return ImagePlaneGeometry{full_width, full_height};
    case ImageEncoding_I420:
      // Luma at full extent, then a U and a V plane each subsampled by two in both dimensions.
      return plane == 0U ? ImagePlaneGeometry{full_width, full_height}
                         : ImagePlaneGeometry{subsampled_extent(width), subsampled_extent(height)};
    case ImageEncoding_NV12:
      // As I420, except the two chroma planes are interleaved into one that holds a U byte and a V
      // byte per chroma column.
      return plane == 0U
                 ? ImagePlaneGeometry{full_width, full_height}
                 : ImagePlaneGeometry{2U * subsampled_extent(width), subsampled_extent(height)};
    case ImageEncoding_P010:
      // NV12's plane structure with every sample in a 16-bit container.
      return plane == 0U
                 ? ImagePlaneGeometry{2U * full_width, full_height}
                 : ImagePlaneGeometry{4U * subsampled_extent(width), subsampled_extent(height)};
    case ImageEncoding_CUSTOM:
    case ImageEncoding_RGB8:
    case ImageEncoding_RGBA8:
    case ImageEncoding_BGR8:
    case ImageEncoding_BGRA8:
    case ImageEncoding_MONO8:
    case ImageEncoding_MONO16:
    case ImageEncoding_MONO32:
    case ImageEncoding_MONO32F:
    case ImageEncoding_YUYV:
    case ImageEncoding_UYVY:
    case ImageEncoding_RGB32F:
    case ImageEncoding_RGBA32F:
    case ImageEncoding_YUV32F:
    case ImageEncoding_RGBA1010102:
    case ImageEncoding_BAYER_RGGB:
    case ImageEncoding_BAYER_BGGR:
    case ImageEncoding_BAYER_GRBG:
    case ImageEncoding_BAYER_GBRG:
      return std::nullopt;
  }
  return std::nullopt;
}

/// @brief Decide whether the declared planes describe memory, and the memory the encoding implies.
///
/// The first group of checks is independent of the encoding, so it applies to
/// \c ImageEncoding_CUSTOM too: a producer-defined layout is still a claim about bytes, and none of
/// these states is a claim at all. A stride of zero advances through no rows. An offset plus an
/// extent that wraps names no range. Two planes covering the same bytes cannot both own them, and a
/// consumer told to write one of them would corrupt the other.
///
/// The second group holds only where the encoding fixes the plane count, and the third only where
/// it also fixes each plane's dimensions and bytes per sample. There the stride has to reach the
/// end of one packed row and the extent has to reach the last row that \c height requires at the
/// declared stride, because a producer that understated either has published a plane a consumer
/// cannot read without leaving it.
///
/// Containment against the Tensor is deliberately not here. It holds for every encoding, including
/// the ones this function answers early for, so it belongs to the caller rather than to a branch
/// this function can return past. See \ref validate_plane_layouts.
[[nodiscard]] detail::ImageFactViolation validate_plane_layout_geometry(
    const detail::ImageFacts& facts) noexcept {
  using Violation = detail::ImageFactViolation;
  const std::span<const ImagePlaneLayout> layouts = facts.plane_layouts;
  if (layouts.size() > kMaximumImagePlanes) {
    return Violation::kPlaneCountExceedsBound;
  }
  for (const ImagePlaneLayout& layout : layouts) {
    if (layout.row_stride_bytes() == 0U) {
      return Violation::kPlaneRowStrideZero;
    }
    if (layout.size_bytes() == 0U) {
      return Violation::kPlaneSizeZero;
    }
    if (layout.offset_bytes() > std::numeric_limits<std::uint64_t>::max() - layout.size_bytes()) {
      return Violation::kPlaneExtentOverflow;
    }
  }
  // Quadratic over at most kMaximumImagePlanes entries, which keeps the check independent of plane
  // order rather than requiring the producer to declare planes in ascending offset.
  for (std::size_t outer = 0U; outer < layouts.size(); ++outer) {
    for (std::size_t inner = outer + 1U; inner < layouts.size(); ++inner) {
      const ImagePlaneLayout& left = layouts[outer];
      const ImagePlaneLayout& right = layouts[inner];
      if (left.offset_bytes() < right.offset_bytes() + right.size_bytes() &&
          right.offset_bytes() < left.offset_bytes() + left.size_bytes()) {
        return Violation::kPlaneOverlap;
      }
    }
  }
  // Absence means a producer-defined layout, which no plane count can contradict.
  const std::optional<std::size_t> planes = encoding_plane_count(facts.encoding);
  if (!planes) {
    return Violation::kNone;
  }
  if (*planes == 1U) {
    return layouts.empty() ? Violation::kNone : Violation::kPlaneLayoutsForbidden;
  }
  // Required, not merely constrained. Omitting the vector used to mean "the Tensor strides are
  // authoritative instead", which cannot be true of a multi-planar image: one Tensor has one base
  // address, so a deferred layout left the plane offsets stated nowhere while appearing to state
  // them somewhere.
  if (layouts.empty()) {
    return Violation::kPlaneLayoutsMissing;
  }
  if (layouts.size() != *planes) {
    return Violation::kPlaneCountMismatch;
  }
  for (std::size_t plane = 0U; plane < layouts.size(); ++plane) {
    const std::optional<ImagePlaneGeometry> geometry =
        encoding_plane_geometry(facts.encoding, plane, facts.width, facts.height);
    if (!geometry) {
      continue;
    }
    if (layouts[plane].row_stride_bytes() < geometry->row_bytes) {
      return Violation::kPlaneRowStrideTooSmall;
    }
    // The minimum an extent can be and still hold every row: the stride carries the reader to the
    // last row, which then needs only its own packed width. A producer is free to reserve more, and
    // a trailing-padded plane declaring rows * stride satisfies this too. The product cannot wrap:
    // rows is bounded by the int32 height, the stride by uint32, and the row width by four times
    // the int32 width, so the sum stays well inside the uint64 the extent is expressed in.
    const std::uint64_t required_bytes =
        (geometry->rows - 1U) * layouts[plane].row_stride_bytes() + geometry->row_bytes;
    if (layouts[plane].size_bytes() < required_bytes) {
      return Violation::kPlaneSizeTooSmall;
    }
  }
  return Violation::kNone;
}

/// @brief Decide whether the declared planes describe memory the Image can actually address.
///
/// Geometry first, so a plane that is both undersized for its encoding and outside the Tensor is
/// reported as the former: that names the arithmetic the producer got wrong, while containment only
/// says the result did not fit. Every geometry check is a statement about the layout alone, and
/// this is the only one that relates it to the memory it claims to describe.
///
/// Containment then applies to every declared plane, \c ImageEncoding_CUSTOM and the single-plane
/// encodings included, which is why it is here and not inside the function above: that function
/// answers early for both, and a bound placed after those branches would silently not apply to
/// them. \c image.fbs states containment in the single allocation backing \c Image.data as a
/// restriction on the format rather than as advice to consumers, and a producer-defined layout is
/// not exempt from it. A plane reaching past that allocation is not a layout a careful reader can
/// work around; it is a range the Image cannot address, since one Tensor supplies one base and no
/// offset from it names a second allocation.
///
/// The bound is the Tensor's addressable extent and not its logical size, because those differ for
/// exactly the layouts plane records exist to describe. A producer that pads its rows has a backing
/// wider than the product of its extents, and bounding against the logical size would reject the
/// aligned multi-planar images the encodings advertise. \ref ImageFacts::tensor_byte_span carries
/// the stride-aware extent for that reason, and a Tensor whose extent is zero or whose layout
/// cannot be represented bounds every plane to nothing: a plane claiming bytes inside a backing
/// that has none, or one whose span cannot be established, is unprovable either way.
[[nodiscard]] detail::ImageFactViolation validate_plane_layouts(
    const detail::ImageFacts& facts) noexcept {
  using Violation = detail::ImageFactViolation;
  const Violation geometry = validate_plane_layout_geometry(facts);
  if (geometry != Violation::kNone) {
    return geometry;
  }
  for (const ImagePlaneLayout& layout : facts.plane_layouts) {
    // Safe to add: validate_plane_layout_geometry rejected the wrap before returning kNone.
    if (layout.offset_bytes() + layout.size_bytes() > facts.tensor_byte_span) {
      return Violation::kPlaneOutsideTensor;
    }
  }
  return Violation::kNone;
}

/// @brief Bytes one Tensor element occupies, or zero when the element type states no width.
///
/// The lane count is part of the element and not a separate axis, so a vectorized element is as
/// wide as its lanes make it. Neither factor can overflow the product: both are byte- and
/// short-width fields on \c DLDataType.
[[nodiscard]] constexpr std::uint64_t element_byte_width(
    const ::DLDataType& element_type) noexcept {
  const std::uint64_t bits = element_type.bits;
  const std::uint64_t lanes = element_type.lanes;
  if (bits == 0U || lanes == 0U) {
    return 0U;
  }
  return (bits * lanes + 7U) / 8U;
}

/// @brief DLPack element type an encoding stores one component in.
struct ImageSampleType {
  std::uint8_t code = 0U;  ///< DLPack type code, \c kDLUInt or \c kDLFloat.
  std::uint8_t bits = 0U;  ///< Width of the container holding one component.
};

/// @brief Element type a single-plane encoding fixes, or absence when it fixes none.
///
/// This is the type a consumer indexes the frame with, which is why only the single-plane
/// encodings appear. \c image.fbs makes the Tensor shape and strides the single authority for the
/// one plane of such an encoding, so its element type is a statement about the same samples the
/// encoding describes and the two can therefore disagree.
///
/// The multi-planar encodings are absent because for them the Tensor is a byte-addressed backing
/// that \c plane_layouts carves up, not a typed view of the samples. P010 is the case that settles
/// it: its samples are 16-bit, its carriage is routinely a flat byte buffer, and requiring the
/// element type to be the sample container would reject the schema's own definition of the
/// encoding. The Bayer entries are absent because the schema defers their width to the element
/// type, so no element type can contradict them, and \c ImageEncoding_CUSTOM because it states no
/// organization for one to contradict.
///
/// Every case is listed for the same reason \ref encoding_plane_count lists them: a new encoding
/// should fail to compile here rather than silently acquire the unchecked answer.
[[nodiscard]] std::optional<ImageSampleType> encoding_sample_type(
    const ImageEncoding encoding) noexcept {
  switch (encoding) {
    case ImageEncoding_RGB8:
    case ImageEncoding_RGBA8:
    case ImageEncoding_BGR8:
    case ImageEncoding_BGRA8:
    case ImageEncoding_MONO8:
    case ImageEncoding_YUYV:
    case ImageEncoding_UYVY:
      return ImageSampleType{kDLUInt, 8U};
    case ImageEncoding_MONO16:
      return ImageSampleType{kDLUInt, 16U};
    case ImageEncoding_MONO32:
      return ImageSampleType{kDLUInt, 32U};
    // Four components packed into one 32-bit word, so the word is the addressable container even
    // though no single component fills it.
    case ImageEncoding_RGBA1010102:
      return ImageSampleType{kDLUInt, 32U};
    case ImageEncoding_MONO32F:
    case ImageEncoding_RGB32F:
    case ImageEncoding_RGBA32F:
    case ImageEncoding_YUV32F:
      return ImageSampleType{kDLFloat, 32U};
    case ImageEncoding_CUSTOM:
    case ImageEncoding_BAYER_RGGB:
    case ImageEncoding_BAYER_BGGR:
    case ImageEncoding_BAYER_GRBG:
    case ImageEncoding_BAYER_GBRG:
    case ImageEncoding_R8_G8_B8:
    case ImageEncoding_B8_G8_R8:
    case ImageEncoding_NV12:
    case ImageEncoding_I420:
    case ImageEncoding_P010:
      return std::nullopt;
  }
  return std::nullopt;
}

/// @brief Bits of real precision one component of an encoding can hold, where the encoding says.
///
/// The bound \c significant_bits is measured against. This is the encoding's own container width
/// rather than the Tensor element type, which is the distinction P010 turns on: it carries 10-bit
/// samples in 16-bit words through a byte buffer, so bounding by the element type would cap it at
/// 8 and bounding by the widest type any element provides would not bound it at all.
///
/// The multi-planar encodings appear here even though \ref encoding_sample_type omits them,
/// because a component width is a property of the encoding whether or not the Tensor is a typed
/// view of it. Only the entries that genuinely state no width are absent: the Bayer entries, which
/// resolve theirs from the element type, and \c ImageEncoding_CUSTOM.
[[nodiscard]] std::optional<std::uint8_t> encoding_component_bits(
    const ImageEncoding encoding) noexcept {
  switch (encoding) {
    case ImageEncoding_RGB8:
    case ImageEncoding_RGBA8:
    case ImageEncoding_BGR8:
    case ImageEncoding_BGRA8:
    case ImageEncoding_MONO8:
    case ImageEncoding_YUYV:
    case ImageEncoding_UYVY:
    case ImageEncoding_R8_G8_B8:
    case ImageEncoding_B8_G8_R8:
    case ImageEncoding_NV12:
    case ImageEncoding_I420:
      return 8U;
    case ImageEncoding_MONO16:
    // The 16-bit container the encoding's 10-bit samples sit in the high bits of. A producer
    // reporting 10 here is the case the field exists for, and one reporting 16 is claiming the
    // container is full rather than claiming something impossible.
    case ImageEncoding_P010:
      return 16U;
    case ImageEncoding_MONO32:
    case ImageEncoding_MONO32F:
    case ImageEncoding_RGB32F:
    case ImageEncoding_RGBA32F:
    case ImageEncoding_YUV32F:
      return 32U;
    // The word is 32 bits wide, but no colour component in it is: the widest holds 10 and the
    // alpha holds 2. The bound is per component, so it is 10 and not the width of the word.
    case ImageEncoding_RGBA1010102:
      return 10U;
    case ImageEncoding_CUSTOM:
    case ImageEncoding_BAYER_RGGB:
    case ImageEncoding_BAYER_BGGR:
    case ImageEncoding_BAYER_GRBG:
    case ImageEncoding_BAYER_GBRG:
      return std::nullopt;
  }
  return std::nullopt;
}

/// @brief Whether an encoding takes its sample width from the Tensor element type.
[[nodiscard]] constexpr bool encoding_defers_sample_width(const ImageEncoding encoding) noexcept {
  switch (encoding) {
    case ImageEncoding_BAYER_RGGB:
    case ImageEncoding_BAYER_BGGR:
    case ImageEncoding_BAYER_GRBG:
    case ImageEncoding_BAYER_GBRG:
      return true;
    default:
      return false;
  }
}

/// @brief Bytes one packed row of a single-plane encoding occupies, or absence when not derivable.
///
/// Only the single-plane encodings that fix their own byte organization appear. Those are the ones
/// whose row width follows from the width alone, so a descriptor that names one is making a
/// checkable claim about how much memory the frame needs.
///
/// Two groups are deliberately absent, and each for a reason the schema states rather than for
/// convenience. The multi-planar encodings carry \c plane_layouts, which already bound every byte
/// they claim against the Tensor, so deriving a second figure here would duplicate that verdict and
/// disagree with it the first time a producer pads between planes. \c ImageEncoding_CUSTOM supplies
/// no sample organization at all.
///
/// The Bayer entries are derivable only once their sample width is resolved, which is what
/// \p deferred_element_bytes carries. The schema defers their container to the Tensor element type,
/// so the width is a fact about this Image rather than about the encoding, and a zero here means
/// the Tensor did not supply one and the frame stays unbounded.
///
/// Every case is listed for the same reason \ref encoding_plane_geometry lists them: a new encoding
/// should fail to compile here rather than silently acquire the unchecked answer.
[[nodiscard]] std::optional<std::uint64_t> encoding_packed_row_bytes(
    const ImageEncoding encoding, const std::int32_t width,
    const std::uint64_t deferred_element_bytes) noexcept {
  const auto full_width = static_cast<std::uint64_t>(width);
  switch (encoding) {
    case ImageEncoding_CUSTOM:
    case ImageEncoding_R8_G8_B8:
    case ImageEncoding_B8_G8_R8:
    case ImageEncoding_NV12:
    case ImageEncoding_I420:
    case ImageEncoding_P010:
      return std::nullopt;
    case ImageEncoding_BAYER_RGGB:
    case ImageEncoding_BAYER_BGGR:
    case ImageEncoding_BAYER_GRBG:
    case ImageEncoding_BAYER_GBRG:
      // One mosaic sample per pixel, in whatever container the Tensor element type established.
      return deferred_element_bytes == 0U ? std::optional<std::uint64_t>{}
                                          : deferred_element_bytes * full_width;
    case ImageEncoding_MONO8:
      return full_width;
    case ImageEncoding_MONO16:
      return 2U * full_width;
    case ImageEncoding_MONO32:
    case ImageEncoding_MONO32F:
      return 4U * full_width;
    case ImageEncoding_RGB8:
    case ImageEncoding_BGR8:
      return 3U * full_width;
    case ImageEncoding_RGBA8:
    case ImageEncoding_BGRA8:
    case ImageEncoding_RGBA1010102:
      return 4U * full_width;
    case ImageEncoding_YUYV:
    case ImageEncoding_UYVY:
      // Packed 4:2:2: one four-byte group per two horizontally adjacent pixels, and an odd width
      // keeps a group of its own for the final column rather than dropping its chroma.
      return 4U * subsampled_extent(width);
    case ImageEncoding_RGB32F:
    case ImageEncoding_YUV32F:
      return 12U * full_width;
    case ImageEncoding_RGBA32F:
      return 16U * full_width;
  }
  return std::nullopt;
}

/// @brief Decide whether the Tensor can address the frame the encoding and extent describe.
///
/// A single-plane encoding declares no plane records, so nothing in \ref validate_plane_layouts
/// compares it to the Tensor and \c width, \c height, and the backing would otherwise be
/// independent facts. That is not a gap a consumer can close: it is told the encoding fixes the
/// byte organization, so it computes addresses from the geometry and reaches past a backing too
/// small to hold it. Requiring the Tensor to cover one packed frame is the weakest bound that
/// closes it, and it is a bound on the minimum: a padded backing addresses more than a packed frame
/// needs and stays admissible.
///
/// A Bayer encoding reaches the same bound through the element type it defers its container to,
/// which is a fact this Image carries even though the encoding does not. That is the schema's own
/// reading of the deferral rather than a stronger rule imposed on it: the width is stated, just
/// somewhere other than in the encoding.
[[nodiscard]] detail::ImageFactViolation validate_encoded_frame_extent(
    const detail::ImageFacts& facts) noexcept {
  using Violation = detail::ImageFactViolation;
  const std::optional<std::uint64_t> row_bytes = encoding_packed_row_bytes(
      facts.encoding, facts.width, element_byte_width(facts.element_type));
  if (!row_bytes.has_value()) {
    return Violation::kNone;
  }
  const auto full_height = static_cast<std::uint64_t>(facts.height);
  // A frame whose byte count does not fit a uint64 exceeds every possible backing, so the overflow
  // and the oversize case carry the same verdict rather than the overflow escaping as a wrap.
  if (*row_bytes != 0U && full_height > std::numeric_limits<std::uint64_t>::max() / *row_bytes) {
    return Violation::kFrameOutsideTensor;
  }
  if (*row_bytes * full_height > facts.tensor_byte_span) {
    return Violation::kFrameOutsideTensor;
  }
  return Violation::kNone;
}

/// @brief View a wire plane-layout vector without copying it.
///
/// A FlatBuffers vector of structs stores its elements inline, contiguously, and aligned to the
/// struct's own alignment, which is what the generated \c Vector::Get reconstructs each element
/// pointer from. Viewing the same bytes as a span is therefore the same access the generated
/// accessor performs, and it is what keeps the receive path free of a copy it has no capacity for.
[[nodiscard]] std::span<const ImagePlaneLayout> wire_plane_layouts(
    const flatbuffers::Vector<const ImagePlaneLayout*>* layouts) noexcept {
  if (layouts == nullptr || layouts->size() == 0U) {
    return {};
  }
  return {reinterpret_cast<const ImagePlaneLayout*>(layouts->Data()), layouts->size()};
}

/// @brief Render one violation as the diagnostic a producer can act on.
///
/// Only the authoring path calls this. The receive path deliberately discards the distinction: it
/// runs per frame and cannot afford to build a string, so it reports a bare protocol error.
[[nodiscard]] std::string_view image_fact_violation_message(
    const detail::ImageFactViolation violation) {
  using Violation = detail::ImageFactViolation;
  switch (violation) {
    case Violation::kNone:
      return "Image metadata is valid";
    case Violation::kDimensions:
      return "Image metadata height and width must be > 0";
    case Violation::kEncoding:
      return "Image metadata encoding is invalid";
    case Violation::kColorSpace:
      return "Image metadata color space is invalid";
    case Violation::kRoiOffset:
      return "Image metadata ROI offsets must be >= 0";
    case Violation::kExposureTime:
      return "Image metadata exposure time must be >= 0";
    case Violation::kGain:
      return "Image metadata gain must be finite and >= 0";
    case Violation::kSignificantBits:
      return "Image metadata significant bits exceed the encoding's component width";
    case Violation::kElementType:
      return "Image metadata Tensor element type contradicts the encoding";
    case Violation::kElementLanes:
      return "Image metadata Tensor element must carry one lane unless the encoding is CUSTOM";
    case Violation::kPlaneCountExceedsBound:
      return "Image metadata declares more planes than supported";
    case Violation::kPlaneRowStrideZero:
      return "Image metadata plane row stride must be > 0";
    case Violation::kPlaneSizeZero:
      return "Image metadata plane size must be > 0";
    case Violation::kPlaneExtentOverflow:
      return "Image metadata plane extent overflows uint64";
    case Violation::kPlaneOutsideTensor:
      return "Image metadata plane extent reaches past the Tensor backing the Image";
    case Violation::kFrameOutsideTensor:
      return "Image metadata frame requires more bytes than the Tensor backing the Image holds";
    case Violation::kPlaneOverlap:
      return "Image metadata planes overlap";
    case Violation::kPlaneLayoutsForbidden:
      return "Image metadata plane layouts must be empty for a single-plane encoding";
    case Violation::kPlaneLayoutsMissing:
      return "Image metadata plane layouts are required for a multi-planar encoding";
    case Violation::kPlaneCountMismatch:
      return "Image metadata plane layouts must have one entry per plane of the encoding";
    case Violation::kPlaneRowStrideTooSmall:
      return "Image metadata plane row stride is shorter than the encoded row width";
    case Violation::kPlaneSizeTooSmall:
      return "Image metadata plane size is shorter than the encoded plane extent";
  }
  return "Image metadata is invalid";
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
  // stamp is deprecated in favour of the inline timestamps, but a producer migrating from the
  // nested form may still populate it, so forward whatever the caller set rather than dropping it.
  const auto stamp = header->stamp
                         ? CreateTime(builder, header->stamp->time_ns, header->stamp->acqtime_ns)
                         : flatbuffers::Offset<Time>{};
  const auto frame_id = header->frame_id.empty() ? flatbuffers::Offset<flatbuffers::String>{}
                                                 : builder.CreateString(header->frame_id);
  return CreateHeader(
      builder, stamp, frame_id, header->capture_timestamp_ns, header->device_sequence);
}

/// Widest component width any element type carries, and so the widest real precision a component
/// can hold. A 64-bit float or integer component is the widest DLPack scalar; a complex value is
/// wider than this in total but not in either of its two components, which is what this bounds.
///
/// The bound of last resort, reached only by an encoding that states no component organization and
/// resolves none from its Tensor. Every encoding that states one is measured against that instead.
constexpr std::uint8_t kMaximumComponentBits = 64U;

/// @brief Report the addressable byte extent of a Tensor's backing, from its first logical element.
///
/// The stride-aware extent, not the product of the extents, because a padded plane layout is
/// exactly the case where the two differ and the padded case is the one plane records exist for.
/// Core's layout validator already computes it, having rejected negative extents and strides,
/// overflow, and ambiguously interleaved axes on the way, so a failure here means the Tensor does
/// not describe a range and zero is the honest answer rather than a fallback.
[[nodiscard]] std::uint64_t tensor_byte_span(const holoscan::Tensor& tensor) noexcept {
  // The null-safe gate for the unguarded dtype() below: itemsize() reports zero for a Tensor
  // carrying no DLPack context, which is the only state in which reading dtype() would not be safe.
  if (tensor.itemsize() == 0U) {
    return 0U;
  }
  const std::span<const std::int64_t> strides = tensor.dlpack_strides_span();
  const auto layout = holoscan::detail::validate_tensor_layout(
      tensor.shape_span(),
      tensor.dtype(),
      strides.empty() ? std::nullopt : std::optional<std::span<const std::int64_t>>{strides},
      holoscan::detail::TensorStrideUnit::kElements);
  return layout ? layout->byte_span : 0U;
}

/// @brief Report the element type of a Tensor's backing, or a zero-bit type when it has none.
///
/// Guarded the same way \ref tensor_byte_span is and for the same reason: \c itemsize() reports
/// zero for a Tensor carrying no DLPack context, which is the only state in which reading
/// \c dtype() would not be safe.
[[nodiscard]] ::DLDataType tensor_element_type(const holoscan::Tensor& tensor) noexcept {
  if (tensor.itemsize() == 0U) {
    return ::DLDataType{};
  }
  return tensor.dtype();
}

/// @brief Decide whether a vectorized Tensor element can carry an encoding that states
/// organization.
///
/// A lane count groups several values into one element. Every encoding except \c CUSTOM already
/// fixes the component organization of a sample, so a vectorized element states that organization a
/// second time and the descriptor carries nothing that relates the two: for a four-lane \c uint8
/// element labelled \c RGB8, nothing decides whether the frame is \c width or four times \c width
/// samples across, nor whether those lanes are the encoding's own components. This is the same
/// fault
/// \ref validate_element_type rejects, reached through the lane count instead of the type.
///
/// The format is written against the convention that channels live in the Tensor shape and the
/// element stays scalar, so requiring one lane costs nothing that is produced today. It is also the
/// direction that stays open: admitting lane grouping later widens what is accepted, while
/// admitting it now and restricting it afterwards would reject descriptors this codec had already
/// published.
///
/// \c CUSTOM states no organization for a lane count to contradict, so it keeps the latitude it
/// already has over plane count and frame extent. Bytes are not what is at stake either way --
/// \ref element_byte_width counts a vectorized element at its full width, so the frame and plane
/// bounds are lane-aware regardless of this verdict.
[[nodiscard]] detail::ImageFactViolation validate_element_lanes(
    const detail::ImageFacts& facts) noexcept {
  using Violation = detail::ImageFactViolation;
  if (facts.encoding == ImageEncoding_CUSTOM || facts.element_type.bits == 0U) {
    return Violation::kNone;
  }
  return facts.element_type.lanes == 1U ? Violation::kNone : Violation::kElementLanes;
}

/// @brief Decide whether the Tensor element type can be the samples the encoding describes.
///
/// Two carriages are admissible for an encoding that fixes its element type, and they are the two
/// a producer actually has. The first is the typed one, where the element is the encoding's own
/// component and a consumer indexes the frame directly. The second is opaque byte carriage, where
/// the frame is a flat \c uint8 buffer and the consumer casts: the Tensor then claims nothing about
/// the samples beyond how many bytes there are, so it cannot contradict the encoding.
///
/// Anything else is a descriptor that states the sample organization twice and states it
/// differently. A \c float32 Tensor labelled \c MONO16 is the case: one of the two is wrong, the
/// pair does not say which, and a consumer trusting either reads the frame the other way.
///
/// An element type of no width is not judged here. It means the Image carries no usable backing at
/// all, which \ref tensor_byte_span reports as an extent of zero and the frame and plane bounds
/// reject on their own terms; naming it a type contradiction would describe the wrong fault.
[[nodiscard]] detail::ImageFactViolation validate_element_type(
    const detail::ImageFacts& facts) noexcept {
  using Violation = detail::ImageFactViolation;
  const std::optional<ImageSampleType> sample_type = encoding_sample_type(facts.encoding);
  if (!sample_type.has_value() || facts.element_type.bits == 0U) {
    return Violation::kNone;
  }
  if (facts.element_type.code == sample_type->code &&
      facts.element_type.bits == sample_type->bits) {
    return Violation::kNone;
  }
  const bool opaque_byte_carriage =
      facts.element_type.code == kDLUInt && facts.element_type.bits == 8U;
  return opaque_byte_carriage ? Violation::kNone : Violation::kElementType;
}

/// @brief Decide whether the reported precision fits the component that has to hold it.
///
/// \c significant_bits is precision within a container, so the container is what bounds it. Where
/// the encoding names one that is the bound, and it is the encoding's container rather than the
/// Tensor element type: P010 carries 10-bit samples in 16-bit words through a byte buffer, so
/// measuring against the element type would cap the encoding at 8 and reject its own definition.
///
/// A Bayer encoding has no container of its own and takes the Tensor's, which is the same deferral
/// the frame bound reads. \c ImageEncoding_CUSTOM names neither, so it keeps the only bound that
/// holds without knowing the organization: no component of any element type is wider than
/// \ref kMaximumComponentBits, so a figure above it describes no sample anywhere.
[[nodiscard]] detail::ImageFactViolation validate_significant_bits(
    const detail::ImageFacts& facts) noexcept {
  using Violation = detail::ImageFactViolation;
  std::optional<std::uint8_t> component_bits = encoding_component_bits(facts.encoding);
  if (!component_bits.has_value() && encoding_defers_sample_width(facts.encoding) &&
      facts.element_type.bits != 0U) {
    component_bits = facts.element_type.bits;
  }
  const std::uint8_t bound = component_bits.value_or(kMaximumComponentBits);
  return facts.significant_bits > bound ? Violation::kSignificantBits : Violation::kNone;
}

}  // namespace

namespace detail {

ImageFacts image_facts(const ImageT& value, const holoscan::Tensor& tensor) noexcept {
  return ImageFacts{
      .height = value.height,
      .width = value.width,
      .encoding = value.encoding,
      .color_space = value.color_space,
      .significant_bits = value.significant_bits,
      .roi_offset_x = value.roi_offset_x,
      .roi_offset_y = value.roi_offset_y,
      .exposure_time_ns = value.exposure_time_ns,
      .gain = value.gain,
      .plane_layouts = std::span<const ImagePlaneLayout>{value.plane_layouts},
      .tensor_byte_span = tensor_byte_span(tensor),
      .element_type = tensor_element_type(tensor),
  };
}

ImageFacts image_facts(const Image& image, const holoscan::Tensor& tensor) noexcept {
  return ImageFacts{
      .height = image.height(),
      .width = image.width(),
      .encoding = image.encoding(),
      .color_space = image.color_space(),
      .significant_bits = image.significant_bits(),
      .roi_offset_x = image.roi_offset_x(),
      .roi_offset_y = image.roi_offset_y(),
      .exposure_time_ns = image.exposure_time_ns(),
      .gain = image.gain(),
      .plane_layouts = wire_plane_layouts(image.plane_layouts()),
      .tensor_byte_span = tensor_byte_span(tensor),
      .element_type = tensor_element_type(tensor),
  };
}

ImageFactViolation validate_image_facts(const ImageFacts& facts) noexcept {
  // Dimensions and encoding come first because the plane checks below are stated in terms of them:
  // a plane's row width is a function of the width and the encoding's subsampling, and neither is
  // meaningful for a non-positive extent or an enumerator the schema does not define.
  if (facts.height <= 0 || facts.width <= 0) {
    return ImageFactViolation::kDimensions;
  }
  if (facts.encoding < ImageEncoding_MIN || facts.encoding > ImageEncoding_MAX) {
    return ImageFactViolation::kEncoding;
  }
  if (facts.color_space < ColorSpace_MIN || facts.color_space > ColorSpace_MAX) {
    return ImageFactViolation::kColorSpace;
  }
  if (facts.roi_offset_x < 0 || facts.roi_offset_y < 0) {
    return ImageFactViolation::kRoiOffset;
  }
  if (facts.exposure_time_ns < 0) {
    return ImageFactViolation::kExposureTime;
  }
  // Ordered so that a NaN gain, which compares false against everything, is rejected too.
  if (!std::isfinite(facts.gain) || facts.gain < 0.0F) {
    return ImageFactViolation::kGain;
  }
  // Before the precision check, because the bound that check applies to a deferred encoding is
  // read off the element type and there is no sense measuring against a type the encoding has
  // already contradicted.
  const ImageFactViolation element_lanes = validate_element_lanes(facts);
  if (element_lanes != ImageFactViolation::kNone) {
    return element_lanes;
  }
  const ImageFactViolation element_type = validate_element_type(facts);
  if (element_type != ImageFactViolation::kNone) {
    return element_type;
  }
  const ImageFactViolation significant_bits = validate_significant_bits(facts);
  if (significant_bits != ImageFactViolation::kNone) {
    return significant_bits;
  }
  // Planes first: where records exist they are the more specific claim, and reporting the frame
  // bound ahead of a malformed plane would name the consequence instead of the cause.
  const ImageFactViolation planes = validate_plane_layouts(facts);
  if (planes != ImageFactViolation::kNone) {
    return planes;
  }
  return validate_encoded_frame_extent(facts);
}

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
  // Both producers reach the wire through here, so taking the verdict at this one point covers
  // build_image_metadata as well as a caller assembling an ImageT directly. The receive path takes
  // the same verdict from the same function against the decoded root, which is what keeps a value
  // this rejects from arriving at a consumer through a peer that never ran this code.
  const ImageFactViolation violation = validate_image_facts(image_facts(value, tensor));
  if (violation != ImageFactViolation::kNone) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, std::string{image_fact_violation_message(violation)}});
  }
  try {
    BoundedSpanAllocator allocator{destination};
    flatbuffers::FlatBufferBuilder builder{destination.size(), &allocator, false};
    const auto tensor_metadata = create_tensor_metadata(builder, tensor);
    if (!tensor_metadata) {
      return make_unexpected(std::move(tensor_metadata).error());
    }
    const auto header = create_header_metadata(builder, value.header.get());
    const auto plane_layouts =
        value.plane_layouts.empty()
            ? flatbuffers::Offset<flatbuffers::Vector<const ImagePlaneLayout*>>{}
            : builder.CreateVectorOfStructs(value.plane_layouts);
    const auto image = CreateImage(builder,
                                   header,
                                   value.height,
                                   value.width,
                                   value.encoding,
                                   *tensor_metadata,
                                   value.color_space,
                                   value.significant_bits,
                                   value.roi_offset_x,
                                   value.roi_offset_y,
                                   plane_layouts,
                                   value.exposure_time_ns,
                                   value.gain);
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
  // No pre-check of the options here on purpose. Every field this helper copies is validated by
  // detail::encode_image_metadata below, and screening a subset of them twice is how the two copies
  // start to disagree.
  ImageT image{};
  image.height = options.height;
  image.width = options.width;
  image.encoding = options.encoding;
  image.header = std::make_shared<HeaderT>();
  image.header->capture_timestamp_ns = options.capture_timestamp_ns;
  image.header->frame_id = options.frame_id;
  image.header->device_sequence = options.device_sequence;
  image.data = std::make_shared<holoscan::Tensor>(tensor);
  image.color_space = options.color_space;
  image.significant_bits = options.significant_bits;
  image.roi_offset_x = options.roi_offset_x;
  image.roi_offset_y = options.roi_offset_y;
  image.plane_layouts = options.plane_layouts;
  image.exposure_time_ns = options.exposure_time_ns;
  image.gain = options.gain;

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
  // The same requirement \c detail::encode_image_metadata places on its destination, checked here
  // because this is the only place a buffer this function did not produce can be caught. Verifying
  // establishes that the offsets inside the buffer are internally aligned; it takes the base
  // address as given, so an unaligned base leaves every 64-bit field in a verified Image unaligned.
  // On x86 that reads correctly and hides the mistake. On aarch64 the same access can fault, which
  // turns a caller passing a span into unaligned storage into a crash on one architecture only.
  if (reinterpret_cast<std::uintptr_t>(metadata.data()) % alignof(flatbuffers::largest_scalar_t) !=
      0U) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Image metadata FlatBuffer storage is unaligned"});
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
