/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file bit_depth_quantization_warning_test.cpp
 *
 * End-to-end test for the bit-depth quantization warning emitted by
 * `Vulkan::Impl::draw_texture()`.
 *
 * When an image (texture) whose representative bit depth is higher than the framebuffer
 * surface's bit depth is rendered, Holoviz quantizes the colors on write-out and warns the
 * user once per distinct (surface format, image format) pair. This test drives the public
 * Holoviz API on a headless window to exercise the real Vulkan render path:
 *
 *   - Positive case: an 8-bit surface with a 10-bit image MUST warn.
 *   - Null cases: an image whose bit depth is equal to or smaller than the surface's MUST
 *     NOT warn.
 *
 * The warning is delivered through `HOLOSCAN_LOG_WARN`, which writes to stderr, so the test
 * captures stderr around the rendered frame and asserts on the presence/absence of the
 * warning's distinctive substring.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include <holoviz/holoviz.hpp>

#include "test_fixture.hpp"

namespace viz = holoscan::viz;

namespace {

/**
 * One end-to-end scenario: render an image of `image_format` into a headless framebuffer
 * whose surface uses `surface_format`, and declare whether the quantization warning is
 * expected.
 */
struct QuantizationScenario {
  viz::SurfaceFormat surface_format;
  viz::ImageFormat image_format;
  bool expect_warning;
  const char* label;
};

std::ostream& operator<<(std::ostream& os, const QuantizationScenario& scenario) {
  return os << scenario.label;
}

// Distinctive substring of the warning emitted in `Vulkan::Impl::draw_texture()`. Matching on
// this phrase (rather than the whole message) keeps the test robust to minor wording changes
// while remaining unique to the bit-depth quantization warning.
constexpr const char* kQuantizationWarning = "quantized when written to the surface";

}  // namespace

/**
 * Headless fixture that selects the surface format *before* `viz::Init()`, as
 * `viz::SetSurfaceFormat()` requires.
 *
 * Each parameterized instance gets its own Holoviz context (Init in SetUp, Shutdown in
 * TearDown), so the per-context "warn once per format pair" deduplication state cannot leak
 * between cases.
 */
class BitDepthQuantizationWarning : public TestHeadless,
                                    public testing::WithParamInterface<QuantizationScenario> {
 protected:
  void SetUp() override {
    // The framebuffer/surface format must be chosen before the window and Vulkan device are
    // created.
    ASSERT_NO_THROW(viz::SetSurfaceFormat(GetParam().surface_format));
    // Initializes Holoviz headless at the fixture's default size.
    TestHeadless::SetUp();
  }
};

TEST_P(BitDepthQuantizationWarning, WarnsOnlyWhenImageExceedsSurfaceBitDepth) {
  const QuantizationScenario& scenario = GetParam();

  // The source image format must be uploadable on this device; skip rather than fail if not.
  uint32_t image_format_count = 0;
  ASSERT_NO_THROW(viz::GetImageFormats(&image_format_count, nullptr));
  std::vector<viz::ImageFormat> supported_image_formats(image_format_count);
  ASSERT_NO_THROW(viz::GetImageFormats(&image_format_count, supported_image_formats.data()));
  if (std::find(supported_image_formats.begin(),
                supported_image_formats.end(),
                scenario.image_format) == supported_image_formats.end()) {
    GTEST_SKIP() << "Image format " << static_cast<int>(scenario.image_format)
                 << " is not supported on this device";
  }

  // Every format under test is 4 bytes per pixel (RGBA8 or a 32-bit packed 10-bit format). The
  // pixel contents are irrelevant: the warning depends only on the format's bit depth, so
  // zero-filled data is sufficient to drive the render path.
  const std::vector<uint8_t> image_data(static_cast<size_t>(width_) * height_ * 4, 0);

  // Render a single frame, capturing everything Holoviz logs to stderr during the draw. The
  // warning (if any) is emitted from draw_texture() during End().
  testing::internal::CaptureStderr();
  EXPECT_NO_THROW(viz::Begin());
  EXPECT_NO_THROW(viz::BeginImageLayer());
  EXPECT_NO_THROW(viz::ImageHost(width_, height_, scenario.image_format, image_data.data()));
  EXPECT_NO_THROW(viz::EndLayer());
  EXPECT_NO_THROW(viz::End());
  const std::string log_output = testing::internal::GetCapturedStderr();

  const bool warned = log_output.find(kQuantizationWarning) != std::string::npos;

  if (scenario.expect_warning) {
    EXPECT_TRUE(warned)
        << "Expected a bit-depth quantization warning (image deeper than surface) but none was "
           "logged.\n=== captured stderr ===\n"
        << log_output;
  } else {
    EXPECT_FALSE(warned)
        << "Did not expect a quantization warning (image not deeper than surface) but one was "
           "logged.\n=== captured stderr ===\n"
        << log_output;
  }
}

INSTANTIATE_TEST_SUITE_P(
    , BitDepthQuantizationWarning,
    testing::Values(
        // 8-bit surface, 10-bit image: colors are quantized on write-out -> warn.
        QuantizationScenario{{viz::ImageFormat::R8G8B8A8_UNORM, viz::ColorSpace::PASS_THROUGH},
                             viz::ImageFormat::A2B10G10R10_UNORM_PACK32,
                             /*expect_warning=*/true,
                             "8bit_surface_10bit_image_warns"},
        QuantizationScenario{{viz::ImageFormat::R8G8B8A8_SRGB, viz::ColorSpace::PASS_THROUGH},
                             viz::ImageFormat::A2B10G10R10_UNORM_PACK32,
                             /*expect_warning=*/true,
                             "8bit_surface_10bit_image_warns_srgb"},
        // Null case (equal depth): 8-bit surface, 8-bit image -> no quantization, no warning.
        QuantizationScenario{{viz::ImageFormat::R8G8B8A8_UNORM, viz::ColorSpace::PASS_THROUGH},
                             viz::ImageFormat::R8G8B8A8_SRGB,
                             /*expect_warning=*/false,
                             "8bit_surface_8bit_image_no_warn"},
        // Null case (shallower image): 10-bit surface, 8-bit image -> no quantization, no
        // warning.
        QuantizationScenario{
            {viz::ImageFormat::A2B10G10R10_UNORM_PACK32, viz::ColorSpace::PASS_THROUGH},
            viz::ImageFormat::R8G8B8A8_UNORM,
            /*expect_warning=*/false,
            "10bit_surface_8bit_image_no_warn"}),
    [](const testing::TestParamInfo<QuantizationScenario>& info) { return info.param.label; });
