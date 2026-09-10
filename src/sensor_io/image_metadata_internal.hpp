/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"

#include "image_generated.h"

namespace holoscan::schema {

namespace detail {

/// @brief Image descriptor facts whose meaning does not depend on where the Image came from.
///
/// The two paths that need these checks hold different representations of the same message: the
/// authoring path an owning \ref ImageT it is about to encode, the receive path a verifier-valid
/// \ref Image root it is about to copy into a sample slot. The fields below are the ones whose
/// meaning is identical in both, so collecting them into one borrowed record lets a single verdict
/// function serve both paths. The alternative -- a producer-side check beside a receive-side check
/// -- agrees only until one of the two is edited.
///
/// \c plane_layouts is a borrowed view. For a wire root it aliases the FlatBuffer directly, which
/// is what keeps the receive path free of a copy it has no capacity to make.
struct ImageFacts {
  std::int32_t height = 0;                          ///< Image height in pixels.
  std::int32_t width = 0;                           ///< Image width in pixels.
  ImageEncoding encoding = ImageEncoding_CUSTOM;    ///< Sample organization of the data.
  ColorSpace color_space = ColorSpace_UNSPECIFIED;  ///< Primaries and transfer function.
  std::uint8_t significant_bits = 0;                ///< Bits of real precision per component.
  std::int32_t roi_offset_x = 0;                    ///< Column of the first pixel.
  std::int32_t roi_offset_y = 0;                    ///< Row of the first pixel.
  std::int64_t exposure_time_ns = 0;                ///< Sensor integration time in nanoseconds.
  float gain = 0.0F;                                ///< Linear sensor gain.
  std::span<const ImagePlaneLayout> plane_layouts;  ///< Borrowed per-plane layout records.
  /// Addressable byte extent of the Tensor backing the Image, measured from its first logical
  /// element, or zero when the backing has none or its layout cannot be represented.
  std::uint64_t tensor_byte_span = 0;
  /// Element type of the Tensor backing the Image, or a zero-bit type when the backing carries no
  /// DLPack context. The encodings that fix their sample organization are checked against it, and
  /// the ones that defer their sample width resolve it from here.
  ::DLDataType element_type{};
};

/// @brief Reason an Image descriptor states something the schema gives no meaning to.
///
/// A verdict rather than an \ref Error because the two callers can afford different diagnostics.
/// Authoring runs once per configured producer and renders the reason as a message; the receive
/// path runs per frame and must neither allocate nor format, so it maps every reason to one
/// protocol error code. Naming the reasons here is what lets both do that from the same decision.
enum class ImageFactViolation : std::uint8_t {
  kNone = 0,                ///< The descriptor is a description of something.
  kDimensions,              ///< A non-positive height or width.
  kEncoding,                ///< An encoding outside the enumeration.
  kColorSpace,              ///< A color space outside the enumeration.
  kRoiOffset,               ///< A pixel origin left of or above the sensor.
  kExposureTime,            ///< A negative integration time.
  kGain,                    ///< A gain that is negative or not a real number.
  kSignificantBits,         ///< More bits of precision than the encoding's component holds.
  kElementType,             ///< A Tensor element type the encoding does not admit.
  kElementLanes,            ///< A vectorized element where the encoding states the organization.
  kPlaneCountExceedsBound,  ///< More planes than the shared inline bound admits.
  kPlaneRowStrideZero,      ///< A plane whose rows do not advance.
  kPlaneSizeZero,           ///< A plane that reserves no bytes.
  kPlaneExtentOverflow,     ///< A plane whose offset plus extent wraps.
  kPlaneOutsideTensor,      ///< A plane whose extent leaves the Tensor backing the Image.
  kFrameOutsideTensor,      ///< A frame whose encoded extent leaves the Tensor backing the Image.
  kPlaneOverlap,            ///< Two planes claiming the same bytes.
  kPlaneLayoutsForbidden,   ///< Layouts declared for a single-plane encoding.
  kPlaneLayoutsMissing,     ///< No layouts declared for a multi-planar encoding.
  kPlaneCountMismatch,      ///< A layout count other than the encoding's plane count.
  kPlaneRowStrideTooSmall,  ///< A stride shorter than the plane's packed row.
  kPlaneSizeTooSmall,       ///< An extent short of the rows the geometry requires.
};

/// @brief Collect the encoding-independent facts of an owning Image value.
/// @param value Owning Image value whose plane-layout vector is borrowed, not copied.
/// @param tensor Tensor that will back the encoded Image, which is not always \c value.data.
/// @return Borrowed facts valid only while \p value is alive and unmodified.
[[nodiscard]] ImageFacts image_facts(const ImageT& value, const holoscan::Tensor& tensor) noexcept;

/// @brief Collect the encoding-independent facts of a verifier-valid Image root.
/// @param image Verified Image root whose plane-layout vector is viewed in place.
/// @param tensor Materialized Tensor this root's descriptor resolved to.
/// @return Borrowed facts valid only while the root's backing bytes are alive and unmodified.
[[nodiscard]] ImageFacts image_facts(const Image& image, const holoscan::Tensor& tensor) noexcept;

/// @brief Decide whether an Image descriptor states anything the schema gives meaning to.
///
/// Checks the signed and enumerated scalars against the ranges the schema defines, and each
/// declared plane against its own internal consistency, the geometry its encoding implies, and the
/// addressable extent of the Tensor the Image carries. \c image.fbs makes containment in that one
/// allocation a restriction rather than an incidental property, so a plane reaching past it is a
/// descriptor the schema gives no meaning to and not merely one a consumer should be careful with.
///
/// Where the encoding states a sample organization, the descriptor is also checked against it: the
/// Tensor element type has to be one the encoding admits, the reported precision has to fit the
/// component holding it, and a single-plane frame has to fit the Tensor backing it. A known
/// encoding that stated its organization twice and stated it differently is refused here rather
/// than left for whichever of the two claims a consumer happens to believe.
///
/// @param facts Borrowed descriptor facts from either path.
/// @return \c kNone when the descriptor is admissible, otherwise the first violation found.
[[nodiscard]] ImageFactViolation validate_image_facts(const ImageFacts& facts) noexcept;

/**
 * Encode descriptor-backed Image metadata without serializing embedded Tensor bytes.
 *
 * The caller supplies the complete bounded FlatBuffers arena and final destination. The encoded
 * Image preserves its optional Header, Time, frame identifier, dimensions, encoding, and Tensor
 * layout metadata, but deliberately omits the Tensor `data` vector regardless of ambient Tensor
 * materialization or gather scopes.
 *
 * @param value Owning Image value containing the Tensor whose storage is carried separately.
 * @param destination Bounded storage aligned for the FlatBuffers largest scalar type.
 * @return Number of serialized bytes at the beginning of `destination`, or an error without a
 *         partial usable payload.
 */
[[nodiscard]] expected<std::size_t, Error> encode_image_metadata(
    const ImageT& value, std::span<std::byte> destination) noexcept;

/// @brief Encode Image metadata using separately owned authoritative Tensor storage.
/// @param value Owning Image value supplying all non-Tensor metadata fields.
/// @param tensor Provider-backed Tensor whose layout replaces `value.data`.
/// @param destination Bounded storage aligned for the FlatBuffers largest scalar type.
/// @return Number of serialized bytes at the beginning of `destination`, or an error without a
/// partial usable payload.
[[nodiscard]] expected<std::size_t, Error> encode_image_metadata(
    const ImageT& value, const holoscan::Tensor& tensor, std::span<std::byte> destination) noexcept;

}  // namespace detail
}  // namespace holoscan::schema
