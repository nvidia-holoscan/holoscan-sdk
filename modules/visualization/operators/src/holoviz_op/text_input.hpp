// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

#include "holoviz_op/backends/frame_data.hpp"

namespace holoscan::visualization {

/// What decode_text_records() admitted from one sample.
struct TextInput {
  std::size_t glyph_count = 0;  ///< Summed UTF-8 byte length of the appended runs.
};

/// Decode one text layer sample — a host uint8 `{N, 16 + L}` tensor of fixed-width text records
/// (see TextLayerDesc) — into backends::TextRun entries appended to \p runs. Rows with empty
/// text are skipped. The runs' string_views borrow the tensor's bytes, so the caller must keep
/// the Tensor alive for as long as the runs are consumed (through RenderBackend::record()).
///
/// Every rejection is kInvalidArgument with a message naming the offending property; the
/// caller decides whether that is a soft (drop the layer for this activation) or hard failure.
/// On rejection \p runs is left as it was.
inline holoscan::expected<TextInput, holoscan::Error> decode_text_records(
    const holoscan::Tensor& tensor, const TextLayerDesc& desc,
    std::vector<backends::TextRun>& runs) {
  const auto invalid = [](const char* message) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument, message});
  };
  const DLDataType dtype = tensor.dtype();
  if (dtype.code != kDLUInt || dtype.bits != 8U || dtype.lanes != 1U) {
    return invalid("text records must be a uint8 tensor");
  }
  const auto shape = tensor.shape();
  if (shape.size() != 2U) {
    return invalid("text records must be rank 2: {runs, 16 + text bytes}");
  }
  if (shape[0] < 0 || shape[1] < static_cast<std::int64_t>(kTextRunHeaderBytes)) {
    return invalid("each text record row needs at least the 16-byte TextRunHeader");
  }
  const auto row_count = static_cast<std::size_t>(shape[0]);
  const auto row_bytes = static_cast<std::size_t>(shape[1]);
  const auto strides = tensor.strides();
  if (!strides.empty()) {
    // Rows must be packed: the row stride is the row width and bytes within a row are
    // contiguous. As in image_input.hpp, a stride on an extent-1 axis never reaches an address
    // and is ignored.
    const bool rows_packed = row_count <= 1U || strides[0] == static_cast<std::int64_t>(row_bytes);
    const bool bytes_packed = row_bytes <= 1U || strides[1] == 1;
    if (strides.size() != 2U || !rows_packed || !bytes_packed) {
      return invalid("text records must be packed rows (strides {16 + L, 1})");
    }
  }
  if (row_count == 0U) {
    return TextInput{};
  }
  const auto* bytes = static_cast<const std::uint8_t*>(tensor.data());
  if (bytes == nullptr) {
    return invalid("text records tensor has no data pointer");
  }

  std::vector<backends::TextRun> decoded;
  decoded.reserve(row_count);
  std::size_t glyph_count = 0;
  const std::size_t max_text_bytes = row_bytes - kTextRunHeaderBytes;
  for (std::size_t row = 0; row < row_count; ++row) {
    const std::uint8_t* record = bytes + row * row_bytes;
    const char* text = reinterpret_cast<const char*>(record + kTextRunHeaderBytes);
    const std::size_t text_bytes = ::strnlen(text, max_text_bytes);
    if (text_bytes == 0U) {
      continue;  // blank row: a producer sized N to its maximum and left this one unused
    }
    TextRunHeader header{};
    std::memcpy(&header, record, sizeof(header));
    // Only non-finite positions are rejected: the unit (pixels or normalized, per
    // TextLayerDesc::normalized_coordinates) is applied by the backend, and a run outside the
    // framebuffer is legitimate (e.g. text sliding off-screen) and clipped when drawn.
    if (!std::isfinite(header.x) || !std::isfinite(header.y)) {
      return invalid("text record position must be finite");
    }
    if (!(header.size_px > 0.0f) || !std::isfinite(header.size_px)) {
      return invalid("text record size_px must be positive and finite");
    }
    glyph_count += text_bytes;
    if (glyph_count > desc.max_glyphs) {
      return invalid("text bytes exceed the layer's declared max_glyphs");
    }
    decoded.push_back(backends::TextRun{
        .x = header.x,
        .y = header.y,
        .size_px = header.size_px,
        .color = header.color,
        .utf8_text = std::string_view{text, text_bytes},
    });
  }
  runs.insert(runs.end(), decoded.begin(), decoded.end());
  return TextInput{.glyph_count = glyph_count};
}

}  // namespace holoscan::visualization
