// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The text layer's input decoder (text_input.hpp): fixed-width uint8 {N, 16 + L} records become
// backends::TextRun entries whose string_views borrow the tensor. Headless — no GPU, no window.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "holoviz_op/text_input.hpp"

namespace holoscan::visualization {
namespace {

constexpr std::size_t kTextBytes = 12;  // L
constexpr std::int64_t kRowBytes = static_cast<std::int64_t>(kTextRunHeaderBytes + kTextBytes);
constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

/// One producer-side record: header + NUL-padded text, exactly as it sits in a row.
struct Record {
  TextRunHeader header{};
  std::string text;
};

/// Rows of records in a host byte buffer imported as a Tensor. The Tensor snapshots
/// shape/strides and borrows the backing until it is destroyed.
class RecordTensor {
 public:
  RecordTensor(const std::vector<Record>& records, std::size_t text_bytes = kTextBytes,
               std::vector<std::int64_t> shape = {}, std::vector<std::int64_t> strides = {},
               DLDataType dtype = {kDLUInt, 8, 1}) {
    const std::size_t row_bytes = kTextRunHeaderBytes + text_bytes;
    bytes_.assign(records.size() * row_bytes + 64, 0);  // slack for stride experiments
    for (std::size_t i = 0; i < records.size(); ++i) {
      auto* row = bytes_.data() + i * row_bytes;
      std::memcpy(row, &records[i].header, sizeof(TextRunHeader));
      const auto n = std::min(records[i].text.size(), text_bytes);
      // TextRun stores length-delimited bytes; the payload is not a C string.
      // NOLINTNEXTLINE(bugprone-not-null-terminated-result)
      std::memcpy(row + kTextRunHeaderBytes, records[i].text.data(), n);
    }
    shape_ = shape.empty() ? std::vector<std::int64_t>{static_cast<std::int64_t>(records.size()),
                                                       static_cast<std::int64_t>(row_bytes)}
                           : std::move(shape);
    strides_ = std::move(strides);
    managed_.dl_tensor = {bytes_.data(),
                          {kDLCPU, 0},
                          static_cast<int32_t>(shape_.size()),
                          dtype,
                          shape_.data(),
                          strides_.empty() ? nullptr : strides_.data(),
                          0};
    auto tensor = make_tensor_from_dlpack(&managed_);
    if (!tensor) {
      ADD_FAILURE() << tensor.error().message;
    } else {
      tensor_ = std::move(*tensor);
    }
  }

  [[nodiscard]] const holoscan::Tensor& tensor() const { return tensor_; }

 private:
  std::vector<std::uint8_t> bytes_;
  std::vector<std::int64_t> shape_;
  std::vector<std::int64_t> strides_;
  DLManagedTensor managed_{};
  holoscan::Tensor tensor_;
};

holoscan::expected<TextInput, holoscan::Error> decode(const RecordTensor& records,
                                                      std::vector<backends::TextRun>& runs,
                                                      std::uint32_t max_glyphs = 64,
                                                      bool normalized_coordinates = false) {
  return decode_text_records(
      records.tensor(),
      TextLayerDesc{.max_glyphs = max_glyphs, .normalized_coordinates = normalized_coordinates},
      runs);
}

TEST(TextInput, DecodesEveryFieldAndBorrowsTheTensorBytes) {
  const RecordTensor records{{
      {{.x = 0.1f, .y = 0.2f, .size_px = 24.0f, .color = 0xFF00D4FFu}, "hello"},
      {{.x = 0.5f, .y = 0.75f, .size_px = 12.5f, .color = 0x80FF0000u}, "world!"},
  }};
  std::vector<backends::TextRun> runs;
  const auto input = decode(records, runs);
  ASSERT_TRUE(input) << input.error().message;
  EXPECT_EQ(input->glyph_count, 11U);
  ASSERT_EQ(runs.size(), 2U);
  EXPECT_EQ(runs[0].x, 0.1f);
  EXPECT_EQ(runs[0].y, 0.2f);
  EXPECT_EQ(runs[0].size_px, 24.0f);
  EXPECT_EQ(runs[0].color, 0xFF00D4FFu);
  EXPECT_EQ(runs[0].utf8_text, "hello");
  EXPECT_EQ(runs[1].color, 0x80FF0000u);
  EXPECT_EQ(runs[1].utf8_text, "world!");
  // Zero-copy: the view points into the tensor's own row, right after the header.
  const auto* base = static_cast<const std::uint8_t*>(records.tensor().data());
  EXPECT_EQ(reinterpret_cast<const std::uint8_t*>(runs[0].utf8_text.data()),
            base + kTextRunHeaderBytes);
  EXPECT_EQ(reinterpret_cast<const std::uint8_t*>(runs[1].utf8_text.data()),
            base + kRowBytes + kTextRunHeaderBytes);
}

TEST(TextInput, TextEndsAtTheFirstNulOrTheRowEnd) {
  std::string with_nul = "ab";
  with_nul.push_back('\0');
  with_nul += "zz";  // bytes after the terminator are padding, not text
  const RecordTensor records{{
      {{}, with_nul},
      {{}, "twelve chars"},  // exactly L bytes, no NUL anywhere in the row
  }};
  std::vector<backends::TextRun> runs;
  const auto input = decode(records, runs);
  ASSERT_TRUE(input) << input.error().message;
  ASSERT_EQ(runs.size(), 2U);
  EXPECT_EQ(runs[0].utf8_text, "ab");
  EXPECT_EQ(runs[1].utf8_text, "twelve chars");
  EXPECT_EQ(input->glyph_count, 14U);
}

TEST(TextInput, SkipsBlankRowsAndAcceptsAnEmptySample) {
  const RecordTensor records{{{{}, ""}, {{.x = 0.5f}, "kept"}, {{}, ""}}};
  std::vector<backends::TextRun> runs;
  const auto input = decode(records, runs);
  ASSERT_TRUE(input) << input.error().message;
  ASSERT_EQ(runs.size(), 1U);
  EXPECT_EQ(runs[0].utf8_text, "kept");
  EXPECT_EQ(runs[0].x, 0.5f);

  // N == 0: a legitimate "nothing this frame".
  const RecordTensor none{{}, kTextBytes, {0, kRowBytes}};
  std::vector<backends::TextRun> no_runs;
  const auto empty = decode(none, no_runs);
  ASSERT_TRUE(empty) << empty.error().message;
  EXPECT_EQ(empty->glyph_count, 0U);
  EXPECT_TRUE(no_runs.empty());
}

TEST(TextInput, AppendsToExistingRunsAndLeavesThemOnRejection) {
  std::vector<backends::TextRun> runs{backends::TextRun{.utf8_text = "existing"}};
  const RecordTensor good{{{{}, "new"}}};
  ASSERT_TRUE(decode(good, runs));
  ASSERT_EQ(runs.size(), 2U);
  EXPECT_EQ(runs[1].utf8_text, "new");

  const RecordTensor bad{{{{}, "ok"}, {{.size_px = 0.0f}, "bad"}}};
  EXPECT_FALSE(decode(bad, runs));
  EXPECT_EQ(runs.size(), 2U);  // nothing from the rejected sample leaked in
}

TEST(TextInput, RejectsTheWrongTensorShapeOrType) {
  const std::vector<Record> one{{{}, "x"}};
  std::vector<backends::TextRun> runs;
  EXPECT_FALSE(decode(RecordTensor{one, kTextBytes, {kRowBytes}}, runs)) << "rank 1";
  EXPECT_FALSE(decode(RecordTensor{one, kTextBytes, {1, kRowBytes, 1}}, runs)) << "rank 3";
  EXPECT_FALSE(decode(RecordTensor{one, kTextBytes, {}, {}, {kDLUInt, 16, 1}}, runs)) << "u16";
  EXPECT_FALSE(decode(RecordTensor{one, kTextBytes, {}, {}, {kDLInt, 8, 1}}, runs)) << "int8";
  EXPECT_FALSE(decode(RecordTensor{one, kTextBytes, {}, {}, {kDLFloat, 32, 1}}, runs)) << "f32";
  EXPECT_FALSE(decode(RecordTensor{one, kTextBytes, {}, {}, {kDLUInt, 8, 2}}, runs)) << "lanes";
  EXPECT_FALSE(decode(RecordTensor{one, kTextBytes, {1, 15}}, runs)) << "row shorter than header";
  EXPECT_TRUE(runs.empty());
}

TEST(TextInput, RequiresPackedRowsButIgnoresSingletonStrides) {
  const std::vector<Record> two{{{}, "a"}, {{}, "b"}};
  std::vector<backends::TextRun> runs;
  EXPECT_TRUE(decode(RecordTensor{two, kTextBytes, {}, {kRowBytes, 1}}, runs));
  EXPECT_FALSE(decode(RecordTensor{two, kTextBytes, {}, {kRowBytes + 4, 1}}, runs)) << "padded";
  EXPECT_FALSE(decode(RecordTensor{two, kTextBytes, {}, {kRowBytes, 2}}, runs)) << "strided";
  // DLPack carries no stride count: a strides array shorter than ndim is malformed input the
  // importer cannot detect, not a Tensor the decoder can be asked to reject.
  // One row: the row stride can never reach a second address.
  const std::vector<Record> one{{{}, "a"}};
  EXPECT_TRUE(decode(RecordTensor{one, kTextBytes, {}, {0, 1}}, runs));
  EXPECT_TRUE(decode(RecordTensor{one, kTextBytes, {}, {999, 1}}, runs));
}

TEST(TextInput, RejectsBadPositionsAndSizes) {
  std::vector<backends::TextRun> runs;
  EXPECT_FALSE(decode(RecordTensor{{{{.size_px = 0.0f}, "x"}}}, runs)) << "size 0";
  EXPECT_FALSE(decode(RecordTensor{{{{.size_px = -1.0f}, "x"}}}, runs)) << "size < 0";
  EXPECT_FALSE(decode(RecordTensor{{{{.size_px = kNaN}, "x"}}}, runs)) << "size NaN";
  EXPECT_FALSE(decode(RecordTensor{{{{.size_px = INFINITY}, "x"}}}, runs)) << "size inf";
  EXPECT_FALSE(decode(RecordTensor{{{{.x = kNaN}, "x"}}}, runs)) << "x NaN";
  EXPECT_FALSE(decode(RecordTensor{{{{.y = -INFINITY}, "x"}}}, runs)) << "y inf";
  // A blank row is never inspected, so garbage in its header is fine.
  EXPECT_TRUE(decode(RecordTensor{{{{.x = kNaN, .size_px = 0.0f}, ""}}}, runs));
  EXPECT_TRUE(runs.empty());
}

// The decoder is unit-agnostic: positions (pixels by default, fractions of the framebuffer with
// TextLayerDesc::normalized_coordinates) pass through unchanged and are not bounded to the
// framebuffer or to [0,1], so a producer may slide a run off-screen and the renderer clips it.
TEST(TextInput, AdmitsPositionsOutsideTheFramebufferInEitherUnit) {
  std::vector<backends::TextRun> runs;
  const RecordTensor records{{
      {{.x = 5.0f, .y = 0.5f}, "right"},
      {{.x = -0.25f, .y = 2.0f}, "below-left"},
      {{.x = 0.0f, .y = 1.0f}, "edge"},
      {{.x = 3840.0f, .y = -12.0f}, "pixels"},
  }};
  ASSERT_TRUE(decode(records, runs));
  ASSERT_TRUE(decode(records, runs, /*max_glyphs=*/64, /*normalized_coordinates=*/true));
  ASSERT_EQ(runs.size(), 8U);
  for (const std::size_t base : {0U, 4U}) {
    EXPECT_EQ(runs[base + 0].x, 5.0f);
    EXPECT_EQ(runs[base + 0].y, 0.5f);
    EXPECT_EQ(runs[base + 1].x, -0.25f);
    EXPECT_EQ(runs[base + 1].y, 2.0f);
    EXPECT_EQ(runs[base + 2].x, 0.0f);
    EXPECT_EQ(runs[base + 2].y, 1.0f);
    EXPECT_EQ(runs[base + 3].x, 3840.0f);
    EXPECT_EQ(runs[base + 3].y, -12.0f);
  }
}

TEST(TextInput, EnforcesTheGlyphBudgetInUtf8Bytes) {
  std::vector<backends::TextRun> runs;
  const RecordTensor ten{{{{}, "12345"}, {{}, "67890"}}};
  EXPECT_TRUE(decode(ten, runs, 10));
  EXPECT_FALSE(decode(ten, runs, 9));
  // "été" is three characters but five UTF-8 bytes: the budget counts bytes, matching the
  // backend's glyph accounting.
  const RecordTensor accented{{{{}, "\xC3\xA9t\xC3\xA9"}}};
  EXPECT_FALSE(decode(accented, runs, 4));
  const auto input = decode(accented, runs, 5);
  ASSERT_TRUE(input) << input.error().message;
  EXPECT_EQ(input->glyph_count, 5U);
}

}  // namespace
}  // namespace holoscan::visualization
