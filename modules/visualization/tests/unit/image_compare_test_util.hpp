// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Shared helpers for image-comparing render tests (factored out of
// test_text_renderer_compare.cpp so the render-backend tests reuse the same comparator,
// BMP dumps, and device/renderer scaffolding). Header-only; test targets only.

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoviz/rhi.hpp>
#include <holoviz/text.hpp>

#include <gtest/gtest.h>

namespace holoviz::test_util {

/// Create a device on the first adapter, or an empty handle if this machine
/// cannot (no Vulkan adapter, no matching CUDA device, ...).
inline rhi::Device try_create_device() {
  auto adapters = rhi::enumerate_adapters();
  if (!adapters || adapters->empty()) {
    return {};
  }
  auto device = adapters->front().create_device({});
  if (!device) {
    return {};
  }
  return std::move(*device);
}

struct Draw {
  text::TextDesc desc;
  std::string_view utf8_text;
};

struct ImageDiff {
  double differing_fraction = 0.0;
  int max_channel_delta = 0;
  size_t ink_a = 0;
  size_t ink_b = 0;
  // Foreground-sensitive metrics: on a mostly-clear render target, normalizing by the whole
  // image lets equal-ink but displaced/disjoint foregrounds pass a small whole-image gate.
  // These normalize by the ink itself, so a missing/translated/garbled glyph region moves
  // them sharply even when the total ink balances out.
  size_t ink_union = 0;                   // pixels inked in either image
  size_t ink_intersection = 0;            // pixels inked in both images
  double ink_iou = 0.0;                   // intersection over union; 0 when both images are blank
  double differing_over_ink_union = 0.0;  // differing pixels / max(ink_union, 1)
};

/// Any RGB channel above this counts as "ink" (glyph coverage on the black clear).
inline constexpr uint8_t kInkThreshold = 16;

/// Compare two RGBA8 images: a pixel differs when any RGB channel delta exceeds
/// channel_tol (alpha ignored). Also counts ink pixels per image so a
/// blank-vs-blank comparison cannot pass, and ink overlap (see ImageDiff) so
/// equal-ink but displaced foregrounds cannot either.
inline ImageDiff compare_images(std::span<const uint8_t> a, std::span<const uint8_t> b,
                                uint8_t channel_tol) {
  ImageDiff diff;
  size_t differing = 0;
  const size_t pixel_count = a.size() / 4;
  for (size_t i = 0; i < pixel_count * 4; i += 4) {
    int max_delta = 0;
    bool ink_a = false;
    bool ink_b = false;
    for (size_t c = 0; c < 3; ++c) {
      max_delta = std::max(max_delta, std::abs(int{a[i + c]} - int{b[i + c]}));
      ink_a |= a[i + c] > kInkThreshold;
      ink_b |= b[i + c] > kInkThreshold;
    }
    differing += max_delta > channel_tol ? 1 : 0;
    diff.max_channel_delta = std::max(diff.max_channel_delta, max_delta);
    diff.ink_a += ink_a ? 1 : 0;
    diff.ink_b += ink_b ? 1 : 0;
    diff.ink_union += (ink_a || ink_b) ? 1 : 0;
    diff.ink_intersection += (ink_a && ink_b) ? 1 : 0;
  }
  diff.differing_fraction = static_cast<double>(differing) / static_cast<double>(pixel_count);
  diff.ink_iou = diff.ink_union != 0 ? static_cast<double>(diff.ink_intersection) /
                                           static_cast<double>(diff.ink_union)
                                     : 0.0;
  diff.differing_over_ink_union =
      static_cast<double>(differing) / static_cast<double>(std::max<size_t>(diff.ink_union, 1));
  return diff;
}

/// 24-bit uncompressed BMP: viewable directly in Windows Explorer, VS Code,
/// and every image viewer, unlike PPM.
inline void write_bmp(const std::filesystem::path& path, std::span<const uint8_t> rgba,
                      uint32_t width, uint32_t height) {
  std::ofstream f(path, std::ios::binary);
  ASSERT_TRUE(f) << "cannot open " << path;
  const uint32_t row_bytes = (width * 3 + 3) & ~3u;  // rows pad to 4 bytes
  const uint32_t pixel_bytes = row_bytes * height;
  const uint32_t header_bytes = 14 + 40;
  const auto put32 = [&f](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
  const auto put16 = [&f](uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
  f.write("BM", 2);
  put32(header_bytes + pixel_bytes);
  put32(0);
  put32(header_bytes);
  put32(40);  // BITMAPINFOHEADER
  put32(width);
  put32(height);
  put16(1);   // planes
  put16(24);  // bits per pixel
  put32(0);   // BI_RGB
  put32(pixel_bytes);
  put32(2835);  // 72 dpi
  put32(2835);
  put32(0);
  put32(0);
  const char pad[3] = {};
  for (uint32_t y = height; y-- > 0;) {  // BMP rows are bottom-up
    for (uint32_t x = 0; x < width; ++x) {
      const size_t i = (static_cast<size_t>(y) * width + x) * 4;
      const char bgr[3] = {static_cast<char>(rgba[i + 2]),
                           static_cast<char>(rgba[i + 1]),
                           static_cast<char>(rgba[i])};
      f.write(bgr, 3);
    }
    f.write(pad, row_bytes - width * 3);
  }
}

/// Both renders plus a per-pixel diff (max RGB channel delta, amplified 8x
/// and clamped so near-threshold differences are visible), written next to
/// the test binary; returns the message listing the paths.
inline std::string dump_artifacts(std::string_view tag, std::span<const uint8_t> a,
                                  std::span<const uint8_t> b, rhi::Extent2D extent) {
  // Callers with no reference image pass an empty (or mismatched) `b`: dump only `a`,
  // there is nothing to diff against.
  const bool have_b = b.size() == a.size() && !b.empty();
  std::vector<uint8_t> diff;
  if (have_b) {
    diff.assign(a.size(), 0xFF);
    for (size_t i = 0; i + 3 < a.size(); i += 4) {
      int max_delta = 0;
      for (size_t c = 0; c < 3; ++c) {
        max_delta = std::max(max_delta, std::abs(int{a[i + c]} - int{b[i + c]}));
      }
      diff[i] = diff[i + 1] = diff[i + 2] = static_cast<uint8_t>(std::min(max_delta * 8, 255));
    }
  }
  std::string message = "dumped comparison images:";
  const std::span<const uint8_t> images[] = {a, b, diff};
  const char* suffixes[] = {"a", "b", "diff"};
  for (size_t i = 0; i < (have_b ? 3u : 1u); ++i) {
    const auto path = std::filesystem::absolute(
        std::filesystem::path("text_compare_" + std::string(tag) + "_" + suffixes[i] + ".bmp"));
    write_bmp(path, images[i], extent.width, extent.height);
    message += "\n  " + path.string();
  }
  return message;
}

/// Print the comparison metrics to stdout in both the pass and fail case;
/// the images can agree so closely that the diff dump is fully black.
inline void log_diff(std::string_view tag, const ImageDiff& diff) {
  std::cout << "[compare] " << tag << ": differing_fraction=" << diff.differing_fraction
            << " max_channel_delta=" << diff.max_channel_delta << " ink_a=" << diff.ink_a
            << " ink_b=" << diff.ink_b << " ink_iou=" << diff.ink_iou
            << " differing_over_ink_union=" << diff.differing_over_ink_union << "\n";
}

/// Set HOLOVIZ_TEXT_COMPARE_DUMP=1 to write the comparison images even when
/// the test passes.
inline bool dump_requested() {
  const char* value = std::getenv("HOLOVIZ_TEXT_COMPARE_DUMP");
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

/// Dump on failure (appended to the failure output) or on request.
inline void maybe_dump(std::string_view tag, std::span<const uint8_t> a, std::span<const uint8_t> b,
                       rhi::Extent2D extent, bool failed) {
  if (!failed && !dump_requested()) {
    return;
  }
  const std::string message = dump_artifacts(tag, a, b, extent);
  if (failed) {
    ADD_FAILURE() << message;
  } else {
    std::cout << message << "\n";
  }
}

/// Render the draws into a fresh offscreen target with the given renderer and
/// download the RGBA8 pixels. Works for both text implementations: the monospace
/// renderer returns a default (no-op) end_frame token, which submit() skips;
/// the ImGui renderer returns a real CUDA-signaled token.
inline void render_with(rhi::Device& device, text::TextRenderer& renderer, rhi::Extent2D extent,
                        std::span<const Draw> draws, std::vector<uint8_t>& out_pixels) {
  auto target = device.create_image({
      .extent = extent,
      .format = rhi::Format::kR8G8B8A8Unorm,
      .usage = rhi::ImageUsage::kRenderTarget,
      .debug_name = "text compare target",
  });
  ASSERT_TRUE(target.has_value()) << target.error().message;

  auto render_pass =
      device.create_render_pass({.render_targets = {{.format = rhi::Format::kR8G8B8A8Unorm}}});
  ASSERT_TRUE(render_pass.has_value()) << render_pass.error().message;
  ASSERT_TRUE(render_pass->set_render_image(*target, 0, {0.0F, 0.0F, 0.0F, 1.0F}).has_value());

  ASSERT_TRUE(renderer.begin_frame(extent, 0, {}).has_value());
  for (const Draw& draw : draws) {
    ASSERT_TRUE(renderer.add_text(draw.desc, draw.utf8_text).has_value());
  }
  const auto token = renderer.end_frame();
  ASSERT_TRUE(token.has_value());

  auto command_list = device.create_command_list({.type = rhi::CommandListType::kTransient});
  ASSERT_TRUE(command_list.has_value()) << command_list.error().message;
  ASSERT_TRUE(command_list->begin().has_value());
  ASSERT_TRUE(
      command_list->begin_render(*render_pass, {0, 0, extent.width, extent.height}).has_value());
  ASSERT_TRUE(renderer.record(*command_list, extent).has_value());
  ASSERT_TRUE(command_list->end_render().has_value());
  ASSERT_TRUE(command_list->end().has_value());

  const rhi::SubmitDesc submit_desc{
      .wait_external_sync_tokens = {&token.value(), 1},
      .command_lists = {&command_list.value(), 1},
  };
  const auto submit_token = device.submit(submit_desc);
  ASSERT_TRUE(submit_token.has_value()) << submit_token.error().message;
  ASSERT_TRUE(device.wait_on_submit(*submit_token, std::chrono::seconds(5)).has_value());

  const size_t row_pitch = static_cast<size_t>(extent.width) * 4;
  out_pixels.resize(row_pitch * extent.height);
  ASSERT_TRUE(
      device
          .download_from_image(*target, 0, std::as_writable_bytes(std::span(out_pixels)), row_pitch)
          .has_value());

  EXPECT_TRUE(device.wait_idle().has_value());
  EXPECT_TRUE(command_list->destroy().has_value());
  EXPECT_TRUE(render_pass->destroy().has_value());
  EXPECT_TRUE(target->destroy().has_value());
}

}  // namespace holoviz::test_util
