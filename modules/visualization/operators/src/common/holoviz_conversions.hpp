// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string_view>

#include <holoscan/visualization/image_format.hpp>
#include <holoscan/visualization/sink_desc.hpp>
#include <holoviz/common.hpp>
#include <holoviz/errors.hpp>
#include <holoviz/expected.hpp>

/// \file
/// Translations between the public holoscan::visualization vocabulary and HoloViz's.
/// Shared by every visualization operator built on HoloViz; deliberately independent of any
/// one operator's private headers so a new operator can include it directly.

namespace holoscan::visualization::common {

using holoviz::Error;
using holoviz::ErrorCode;
using holoviz::expected;
using holoviz::make_unexpected;

/// Conversions between the public layer-API format vocabulary (ImageFormat) and HoloViz's
/// (holoviz::Format). The mapping is intentionally partial in both directions: multi-planar and
/// packed YUV ImageFormats lower to one or more HoloViz images inside a RenderBackend
/// implementation (see the holoviz::Format doc in <holoviz/common.hpp>), and most HoloViz
/// formats (float render targets, depth, vertex-attribute formats) have no public spelling.

/// Map a public ImageFormat to its byte-preserving HoloViz format (for output targets).
/// Source conversion is handled separately by the image layer. Fails with kInvalidArgument
/// for kUnknown and formats requiring upload conversion or shader decoding
/// (kR8G8B8Unorm, kNV12, kYUYV422).
expected<holoviz::Format> to_holoviz_format(ImageFormat format);

/// Map a HoloViz format (e.g. a swapchain's actual format) to its public ImageFormat. Fails
/// with kFailure for HoloViz formats that have no public spelling.
expected<ImageFormat> to_image_format(holoviz::Format format);

/// Enumerator name of \p format for diagnostics (e.g. "kR16Unorm"); empty for out-of-range values.
std::string_view image_format_name(ImageFormat format);

/// Bytes per pixel of a packed single-plane \p format (the row pitch of a tightly packed row is
/// `width * bytes_per_pixel`). Zero for kUnknown and for the multi-plane / subsampled YUV kinds
/// (kNV12, kYUYV422), which have no single per-pixel byte count.
uint32_t bytes_per_pixel(ImageFormat format);

/// The public PresentMode and HoloViz's are distinct enums with the same five enumerators, so
/// the public API carries no HoloViz header. Both mappings are total; a new enumerator on either
/// side fails to compile here (-Wswitch, no default).
holoviz::PresentMode to_holoviz_present_mode(PresentMode mode);
PresentMode to_present_mode(holoviz::PresentMode mode);

}  // namespace holoscan::visualization::common
