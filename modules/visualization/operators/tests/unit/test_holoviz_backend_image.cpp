// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Image layer tests for HolovizRenderBackend: a CUDA-resident RGBA frame must survive the
// upload + draw byte-exactly (1:1 mapping, nearest-equivalent sampling), producer-stream
// ordering must hold without any caller synchronization, and slot reuse must show the
// newest frame only.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda.h>

#include <holoscan/visualization/holoviz_op.hpp>
#include <holoviz/holoviz.hpp>
#include "holoviz_op/backends/frame_data.hpp"
#include "holoviz_op/backends/holoviz_render_backend.hpp"
#include "holoviz_op/backends/render_backend.hpp"

#include "image_compare_test_util.hpp"

// Module-private headers (src/ is on this target's include path).
#include "holoviz_op/backends/backend_tokens.hpp"

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"

namespace holoscan::visualization::backends {

namespace test_util = holoviz::test_util;
namespace {

using test_util::maybe_dump;
using test_util::try_create_device;

constexpr uint32_t kSize = 128;  // target and declared image max: 1:1 texel-to-pixel

class HolovizBackendImageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_ = try_create_device();
    if (!device_) {
      HOLOVIZ_ENV_SKIP() << "no usable rendering device/adapter.";
    }
    const auto context = device_.cuda_context();
    ASSERT_TRUE(context.has_value());
    if (cuCtxSetCurrent(*context) != CUDA_SUCCESS ||
        cuStreamCreate(&producer_stream_, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
      HOLOVIZ_ENV_SKIP() << "cannot create a CUDA stream on the device context";
    }
    // Match the backend default: imported-array copies need default flags on some drivers.
    ASSERT_EQ(cuStreamCreate(&upload_stream_, CU_STREAM_DEFAULT), CUDA_SUCCESS);
    ASSERT_EQ(cuEventCreate(&rgb_reads_done_, CU_EVENT_DISABLE_TIMING), CUDA_SUCCESS);
  }

  // Keep the fixture retirement and cleanup sequence together so failed assertions retain their
  // context.
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  void TearDown() override {
    // Fatal assertions must not strand the callback or free memory still read by uploads.
    release_producer();
    if (producer_stream_ != nullptr) {
      EXPECT_EQ(cuStreamSynchronize(producer_stream_), CUDA_SUCCESS);
    }
    if (upload_stream_ != nullptr) {
      EXPECT_EQ(cuStreamSynchronize(upload_stream_), CUDA_SUCCESS);
    }
    backend_.reset();
    if (rgb_reads_done_ != nullptr) {
      EXPECT_EQ(cuEventDestroy(rgb_reads_done_), CUDA_SUCCESS);
    }
    if (upload_stream_ != nullptr) {
      EXPECT_EQ(cuStreamDestroy(upload_stream_), CUDA_SUCCESS);
    }
    if (source_ != 0) {
      cuMemFree(source_);
    }
    if (producer_stream_ != nullptr) {
      cuStreamSynchronize(producer_stream_);
      cuStreamDestroy(producer_stream_);
    }
    if (device_) {
      ASSERT_TRUE(device_.destroy(std::chrono::seconds(5)).has_value());
    }
  }

  void make_backend(ImageLayerDesc spec, bool srgb = false, float clear_rgb = 0.F,
                    float clear_alpha = 1.F) {
    auto backend = create_holoviz_render_backend(device_, {.stream = upload_stream_});
    ASSERT_TRUE(backend.has_value());
    backend_ = std::move(*backend);

    LayerStack bundle;
    bundle.add_image_layer("video", std::move(spec));

    RenderConfig config;
    config.headless = true;
    config.headless_width = kSize;
    config.headless_height = kSize;
    config.max_frames_in_flight = 2;
    config.headless_framebuffer_srgb = srgb;
    config.clear_color[0] = config.clear_color[1] = config.clear_color[2] = clear_rgb;
    config.clear_color[3] = clear_alpha;
    ASSERT_TRUE(backend_->allocate(bundle, config).has_value());
    ASSERT_TRUE(backend_->build(ImageFormat::kB8G8R8A8Unorm).has_value());
    auto* concrete = static_cast<HolovizRenderBackend*>(backend_.get());
    EXPECT_EQ(concrete->headless_target(0)->get().format().value(),
              srgb ? holoviz::Format::kR8G8B8A8Srgb : holoviz::Format::kR8G8B8A8Unorm);
  }

  /// Upload `rgba` to device memory synchronously (the "already visible" producer case).
  void make_source(const std::vector<uint8_t>& bytes) {
    if (source_ != 0) {
      cuMemFree(source_);
      source_ = 0;
    }
    ASSERT_EQ(cuMemAlloc(&source_, bytes.size()), CUDA_SUCCESS);
    ASSERT_EQ(cuMemcpyHtoD(source_, bytes.data(), bytes.size()), CUDA_SUCCESS);
  }

  void render(const ImageFrameData& image, std::vector<uint8_t>& out_pixels) {
    RenderList list;
    list.layers.push_back(LayerFrame{0, &image});
    render(list, out_pixels);
  }

  void render(const RenderList& list, std::vector<uint8_t>& out_pixels) {
    auto cmds = backend_->begin_frame(AcquireToken{}, 0);
    ASSERT_TRUE(cmds.has_value()) << cmds.error().message;
    ASSERT_TRUE(backend_->record(*cmds, list, CompositeSpec{}).has_value());
    auto token = backend_->submit(*cmds, SubmitContext{0});
    ASSERT_TRUE(token.has_value()) << token.error().message;
    // Use the public device wait so this readback harness also works with a shared renderer.
    ASSERT_TRUE(device_.wait_idle().has_value());

    auto* concrete = static_cast<HolovizRenderBackend*>(backend_.get());
    out_pixels.resize(static_cast<size_t>(kSize) * kSize * 4);
    ASSERT_TRUE(device_
                    .download_from_image(concrete->headless_target(0)->get(),
                                         0,
                                         std::as_writable_bytes(std::span(out_pixels)),
                                         static_cast<size_t>(kSize) * 4)
                    .has_value());
  }

  static std::vector<uint8_t> gradient_rgba() {
    std::vector<uint8_t> bytes(static_cast<size_t>(kSize) * kSize * 4);
    for (uint32_t y = 0; y < kSize; ++y) {
      for (uint32_t x = 0; x < kSize; ++x) {
        const size_t i = (static_cast<size_t>(y) * kSize + x) * 4;
        bytes[i + 0] = static_cast<uint8_t>(x * 2);
        bytes[i + 1] = static_cast<uint8_t>(y * 2);
        bytes[i + 2] = static_cast<uint8_t>((x + y));
        bytes[i + 3] = 255;
      }
    }
    return bytes;
  }

  static void CUDA_CB hold_producer(void* data) {
    auto& fixture = *static_cast<HolovizBackendImageTest*>(data);
    std::unique_lock lock(fixture.producer_gate_mutex_);
    fixture.producer_entered_ = true;
    fixture.producer_gate_cv_.notify_all();
    fixture.producer_gate_cv_.wait(lock, [&] { return fixture.producer_released_; });
  }

  void release_producer() {
    {
      const std::lock_guard lock(producer_gate_mutex_);
      producer_released_ = true;
    }
    producer_gate_cv_.notify_all();
  }

  std::mutex producer_gate_mutex_;
  std::condition_variable producer_gate_cv_;
  bool producer_entered_ = false;
  bool producer_released_ = false;
  CUstream upload_stream_ = nullptr;
  CUevent rgb_reads_done_ = nullptr;
  holoviz::Device device_;
  std::unique_ptr<RenderBackend> backend_;
  CUstream producer_stream_ = nullptr;
  CUdeviceptr source_ = 0;
};

TEST_F(HolovizBackendImageTest, RgbaFrameRoundTripsByteExact) {
  make_backend(ImageLayerDesc{ImageFormat::kR8G8B8A8Unorm, kSize, kSize});

  const std::vector<uint8_t> expected = gradient_rgba();
  make_source(expected);

  const ImageFrameData image{kSize, kSize, {{source_, kSize * 4}}, 1, nullptr};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render(image, pixels));

  size_t mismatches = 0;
  for (size_t i = 0; i < expected.size() && mismatches < 8; ++i) {
    if (pixels[i] != expected[i]) {
      ++mismatches;
      ADD_FAILURE() << "byte " << i << ": got " << int{pixels[i]} << " want " << int{expected[i]};
    }
  }
  EXPECT_EQ(mismatches, 0u);
  maybe_dump("image_rgba", pixels, expected, holoviz::Extent2D{kSize, kSize}, HasFailure());
}

// A BGRA source lands byte-for-byte in the RGBA8 slot texture and the shader swaps R and B, so
// the RGBA8 headless readback must be the source with bytes 0 and 2 exchanged per pixel.
TEST_F(HolovizBackendImageTest, BGRAFrameRoundTripsWithChannelSwap) {
  make_backend(ImageLayerDesc{ImageFormat::kB8G8R8A8Unorm, kSize, kSize});

  const std::vector<uint8_t> source = gradient_rgba();
  make_source(source);

  std::vector<uint8_t> expected = source;
  for (size_t i = 0; i < expected.size(); i += 4) {
    std::swap(expected[i + 0], expected[i + 2]);
  }

  const ImageFrameData image{kSize, kSize, {{source_, kSize * 4}}, 1, nullptr};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render(image, pixels));

  size_t mismatches = 0;
  for (size_t i = 0; i < expected.size() && mismatches < 8; ++i) {
    if (pixels[i] != expected[i]) {
      ++mismatches;
      ADD_FAILURE() << "byte " << i << ": got " << int{pixels[i]} << " want " << int{expected[i]};
    }
  }
  EXPECT_EQ(mismatches, 0u);
  maybe_dump("image_bgra", pixels, expected, holoviz::Extent2D{kSize, kSize}, HasFailure());
}

// A kR8Unorm layer is a grayscale source: the single channel must land in R, G and B alike
// with opaque alpha (kImageFlagGrayscaleReplicate), not as red-only.
TEST_F(HolovizBackendImageTest, R8GrayscaleReplicatesToRgb) {
  make_backend(ImageLayerDesc{ImageFormat::kR8Unorm, kSize, kSize});

  std::vector<uint8_t> source(static_cast<size_t>(kSize) * kSize);
  for (uint32_t y = 0; y < kSize; ++y) {
    for (uint32_t x = 0; x < kSize; ++x) {
      source[static_cast<size_t>(y) * kSize + x] = static_cast<uint8_t>(x + y);
    }
  }
  make_source(source);

  const ImageFrameData image{kSize, kSize, {{source_, kSize}}, 1, nullptr};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render(image, pixels));

  size_t mismatches = 0;
  for (size_t p = 0; p < source.size() && mismatches < 8; ++p) {
    const uint8_t want = source[p];
    for (size_t c = 0; c < 3; ++c) {
      if (pixels[p * 4 + c] != want) {
        ++mismatches;
        ADD_FAILURE() << "pixel " << p << " channel " << c << ": got " << int{pixels[p * 4 + c]}
                      << " want " << int{want};
      }
    }
    if (pixels[p * 4 + 3] != 255) {
      ++mismatches;
      ADD_FAILURE() << "pixel " << p << " alpha " << int{pixels[p * 4 + 3]} << " want 255";
    }
  }
  EXPECT_EQ(mismatches, 0u);
}

// A kR16Unorm layer uploads two bytes per pixel and samples as UNORM16, so a value v shows as
// round(v / 257) gray. Values of the form hi * 257 are exact; a column holding 0x00FF proves
// the full 16 bits are honored (an 8-bit misread of the low byte would show white, UNORM16
// shows ~1/255).
TEST_F(HolovizBackendImageTest, R16GrayscaleUploadsFullPrecision) {
  make_backend(ImageLayerDesc{ImageFormat::kR16Unorm, kSize, kSize});

  constexpr uint32_t kLowByteColumn = 5;
  std::vector<uint16_t> source16(static_cast<size_t>(kSize) * kSize);
  for (uint32_t y = 0; y < kSize; ++y) {
    for (uint32_t x = 0; x < kSize; ++x) {
      const uint16_t hi = static_cast<uint16_t>((x + y) & 0xFFU);
      source16[static_cast<size_t>(y) * kSize + x] =
          x == kLowByteColumn ? uint16_t{0x00FF} : static_cast<uint16_t>(hi * 257U);
    }
  }
  std::vector<uint8_t> source_bytes(source16.size() * sizeof(uint16_t));
  std::memcpy(source_bytes.data(), source16.data(), source_bytes.size());
  make_source(source_bytes);

  const ImageFrameData image{kSize, kSize, {{source_, kSize * 2}}, 1, nullptr};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render(image, pixels));

  size_t mismatches = 0;
  for (size_t p = 0; p < source16.size() && mismatches < 8; ++p) {
    const uint32_t x = static_cast<uint32_t>(p % kSize);
    const int want = x == kLowByteColumn ? 1 : static_cast<int>(source16[p] / 257U);
    for (size_t c = 0; c < 3; ++c) {
      const int got = pixels[p * 4 + c];
      if (got < want - 1 || got > want + 1) {
        ++mismatches;
        ADD_FAILURE() << "pixel " << p << " channel " << c << ": got " << got << " want " << want;
      }
    }
    if (pixels[p * 4 + 3] != 255) {
      ++mismatches;
      ADD_FAILURE() << "pixel " << p << " alpha " << int{pixels[p * 4 + 3]} << " want 255";
    }
  }
  EXPECT_EQ(mismatches, 0u);
}

TEST_F(HolovizBackendImageTest, LinearR16EncodesSRGBIncludingDarkSegment) {
  make_backend(ImageLayerDesc{.format = ImageFormat::kR16Unorm,
                              .max_width = kSize,
                              .max_height = kSize,
                              .color_space = ColorSpace::kBT709Linear},
               true);
  // Black, the linear toe, middle gray, half intensity, and white. Expected bytes are
  // sRGB display codes, not the source's linear intensity (18% gray must be about 118).
  constexpr std::array<uint16_t, 5> kSource{0, 128, 11796, 32768, 65535};
  constexpr std::array<int, 5> kExpected{0, 6, 118, 188, 255};
  std::vector<uint16_t> source16(static_cast<size_t>(kSize) * kSize);
  for (size_t p = 0; p < source16.size(); ++p) {
    source16[p] = kSource[(p % kSize) / 26];
  }
  std::vector<uint8_t> bytes(source16.size() * sizeof(uint16_t));
  std::memcpy(bytes.data(), source16.data(), bytes.size());
  make_source(bytes);
  const ImageFrameData image{kSize, kSize, {{source_, kSize * 2}}, 1, nullptr};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render(image, pixels));
  for (size_t stripe = 0; stripe < kSource.size(); ++stripe) {
    const size_t p = (kSize / 2 * kSize + stripe * 26 + 8) * 4;
    for (size_t channel = 0; channel < 3; ++channel) {
      EXPECT_NEAR(pixels[p + channel], kExpected[stripe], 1) << "stripe " << stripe;
    }
    EXPECT_EQ(pixels[p + 3], 255);
  }
}

TEST_F(HolovizBackendImageTest, EncodedSRGBIsNotEncodedTwice) {
  make_backend(ImageLayerDesc{.format = ImageFormat::kR8G8B8A8Unorm,
                              .max_width = kSize,
                              .max_height = kSize,
                              .color_space = ColorSpace::kSRGBNonlinear},
               true);
  std::vector<uint8_t> bytes(static_cast<size_t>(kSize) * kSize * 4, 118);
  for (size_t i = 3; i < bytes.size(); i += 4) {
    bytes[i] = 255;
  }
  make_source(bytes);
  const ImageFrameData image{kSize, kSize, {{source_, kSize * 4}}, 1, nullptr};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render(image, pixels));
  for (size_t i = 0; i < bytes.size(); ++i) {
    EXPECT_NEAR(pixels[i], bytes[i], 1);
  }
}

TEST_F(HolovizBackendImageTest, LinearRGBBlendsBeforeEncodingAndPreservesAlpha) {
  make_backend(ImageLayerDesc{.format = ImageFormat::kR16G16B16A16Unorm,
                              .max_width = kSize,
                              .max_height = kSize,
                              .color_space = ColorSpace::kBT709Linear},
               true,
               0.F,
               0.F);
  std::vector<uint16_t> rgba(static_cast<size_t>(kSize) * kSize * 4);
  for (size_t i = 0; i < rgba.size(); i += 4) {
    rgba[i] = 11796;
    rgba[i + 1] = 32768;
    rgba[i + 2] = 65535;
    rgba[i + 3] = 32768;
  }
  std::vector<uint8_t> bytes(rgba.size() * sizeof(uint16_t));
  std::memcpy(bytes.data(), rgba.data(), bytes.size());
  make_source(bytes);
  const ImageFrameData image{kSize, kSize, {{source_, kSize * 8}}, 1, nullptr};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render(image, pixels));
  // Blend in linear light, THEN encode (.09, .25, .5). Alpha stays linear (.5), not sRGB.
  const size_t p = (kSize / 2 * kSize + kSize / 2) * 4;
  EXPECT_NEAR(pixels[p], 85, 1);
  EXPECT_NEAR(pixels[p + 1], 137, 1);
  EXPECT_NEAR(pixels[p + 2], 188, 1);
  EXPECT_NEAR(pixels[p + 3], 128, 1);
}

TEST_F(HolovizBackendImageTest, SRGBSamplingDecodesBeforeLinearFiltering) {
  make_backend(ImageLayerDesc{.format = ImageFormat::kR8G8B8A8Unorm,
                              .max_width = 2,
                              .max_height = 1,
                              .color_space = ColorSpace::kSRGBNonlinear},
               true);
  make_source({0, 0, 0, 255, 255, 255, 255, 255});
  const ImageFrameData image{2, 1, {{source_, 8}}, 1, nullptr};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render(image, pixels));
  // At x=63, bilinear sampling gives 0.4921875 linear white. Decoding AFTER filtering
  // would instead produce about 126 after the framebuffer re-encodes, rather than 186.
  const size_t p = (kSize / 2 * kSize + 63) * 4;
  for (size_t c = 0; c < 3; ++c) {
    EXPECT_NEAR(pixels[p + c], 186, 1);
  }
  EXPECT_EQ(pixels[p + 3], 255);
}

TEST_F(HolovizBackendImageTest, EncodedBGRAOverNonblackDestinationBlendsInLinearLight) {
  make_backend(ImageLayerDesc{.format = ImageFormat::kB8G8R8A8Unorm,
                              .max_width = 1,
                              .max_height = 1,
                              .color_space = ColorSpace::kSRGBNonlinear},
               true,
               0.25F,
               0.25F);
  make_source({255, 188, 118, 128});
  const ImageFrameData image{1, 1, {{source_, 4}}, 1, nullptr};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render(image, pixels));
  // Source RGB is approximately (.18, .5, 1), destination is .25. Both must be
  // linearized before blending; BGRA swizzling must not move or decode alpha.
  const size_t p = (kSize / 2 * kSize + kSize / 2) * 4;
  EXPECT_NEAR(pixels[p], 128, 1);
  EXPECT_NEAR(pixels[p + 1], 165, 1);
  EXPECT_NEAR(pixels[p + 2], 207, 1);
  EXPECT_NEAR(pixels[p + 3], 160, 1);
}

TEST_F(HolovizBackendImageTest, LinearColorSpaceDoesNotEncodeWithoutSRGBFramebuffer) {
  make_backend(ImageLayerDesc{.format = ImageFormat::kR8G8B8A8Unorm,
                              .max_width = 1,
                              .max_height = 1,
                              .color_space = ColorSpace::kBT709Linear});
  make_source({46, 128, 255, 255});
  const ImageFrameData image{1, 1, {{source_, 4}}, 1, nullptr};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render(image, pixels));
  EXPECT_EQ(pixels[0], 46);
  EXPECT_EQ(pixels[1], 128);
  EXPECT_EQ(pixels[2], 255);
}

TEST_F(HolovizBackendImageTest, MixedLinearAndEncodedLayersBlendInLinearLight) {
  auto backend = create_holoviz_render_backend(device_, {});
  ASSERT_TRUE(backend.has_value());
  backend_ = std::move(*backend);
  LayerStack layers;
  layers.add_image_layer("linear",
                         ImageLayerDesc{.format = ImageFormat::kR8G8B8A8Unorm,
                                        .max_width = 1,
                                        .max_height = 1,
                                        .color_space = ColorSpace::kBT709Linear});
  layers.add_image_layer("encoded",
                         ImageLayerDesc{.format = ImageFormat::kR8G8B8A8Unorm,
                                        .max_width = 1,
                                        .max_height = 1,
                                        .color_space = ColorSpace::kSRGBNonlinear});
  RenderConfig config{.headless = true,
                      .max_frames_in_flight = 2,
                      .headless_width = kSize,
                      .headless_height = kSize};
  config.headless_framebuffer_srgb = true;
  config.clear_color[3] = 0.F;
  ASSERT_TRUE(backend_->allocate(layers, config).has_value());
  ASSERT_TRUE(backend_->build(ImageFormat::kR8G8B8A8Srgb).has_value());
  make_source({128, 128, 128, 128});
  const ImageFrameData image{1, 1, {{source_, 4}}, 1, nullptr};
  const RenderList list{.layers = {{0, &image}, {1, &image}}};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render(list, pixels));
  // Half-alpha linear .502 followed by half-alpha decoded sRGB .216 -> .234 linear.
  for (size_t c = 0; c < 3; ++c) {
    EXPECT_NEAR(pixels[c], 133, 1);
  }
  EXPECT_NEAR(pixels[3], 192, 1);
}

TEST_F(HolovizBackendImageTest, UnsupportedSRGBSourceFormatsFailDuringSetup) {
  for (auto format :
       {ImageFormat::kR8Unorm, ImageFormat::kR16Unorm, ImageFormat::kR16G16B16A16Unorm}) {
    auto backend = create_holoviz_render_backend(device_, {});
    ASSERT_TRUE(backend.has_value());
    LayerStack layers;
    layers.add_image_layer("video",
                           ImageLayerDesc{.format = format,
                                          .max_width = 1,
                                          .max_height = 1,
                                          .color_space = ColorSpace::kSRGBNonlinear});
    const auto result = (*backend)->allocate(layers, RenderConfig{});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ErrorCode::kNotImplemented);
  }
}

// A kR16G16B16A16Unorm layer uploads eight bytes per pixel; each channel's high byte must
// arrive in the RGBA8 target in source channel order (values hi * 257 are exact in UNORM).
TEST_F(HolovizBackendImageTest, R16G16B16A16FrameRoundTripsHighBytes) {
  make_backend(ImageLayerDesc{ImageFormat::kR16G16B16A16Unorm, kSize, kSize});

  const std::vector<uint8_t> expected = gradient_rgba();
  std::vector<uint16_t> source16(expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    source16[i] = static_cast<uint16_t>(expected[i] * 257U);
  }
  std::vector<uint8_t> source_bytes(source16.size() * sizeof(uint16_t));
  std::memcpy(source_bytes.data(), source16.data(), source_bytes.size());
  make_source(source_bytes);

  const ImageFrameData image{kSize, kSize, {{source_, kSize * 8}}, 1, nullptr};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render(image, pixels));

  size_t mismatches = 0;
  for (size_t i = 0; i < expected.size() && mismatches < 8; ++i) {
    if (pixels[i] != expected[i]) {
      ++mismatches;
      ADD_FAILURE() << "byte " << i << ": got " << int{pixels[i]} << " want " << int{expected[i]};
    }
  }
  EXPECT_EQ(mismatches, 0u);
  maybe_dump("image_rgba16", pixels, expected, holoviz::Extent2D{kSize, kSize}, HasFailure());
}

TEST_F(HolovizBackendImageTest, AbortAfterEnqueuedUploadsReleasesFrameDataImmediately) {
  make_backend(ImageLayerDesc{ImageFormat::kR8G8B8A8Unorm, kSize, kSize});
  make_source(gradient_rgba());

  // Hold the producer stream back so the frame's image copy is enqueued but pending
  // when the frame is aborted.
  const CUhostFn sleep_fn = [](void*) {
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
  };
  ASSERT_EQ(cuLaunchHostFunc(producer_stream_, sleep_fn, nullptr), CUDA_SUCCESS);

  auto cmds = backend_->begin_frame(AcquireToken{}, 0);
  ASSERT_TRUE(cmds.has_value());
  const ImageFrameData pending{kSize, kSize, {{source_, kSize * 4}}, 1, producer_stream_};
  RenderList list;
  list.layers.push_back(LayerFrame{0, &pending});
  ASSERT_TRUE(backend_->record(*cmds, list, CompositeSpec{}).has_value());

  // abort_frame() must not release the frame until the enqueued reads of source_ are done.
  // The wait is bounded (RenderConfig::backpressure_timeout_ns, 100 ms default, shorter
  // than the 150 ms gate): a kBackpressured result leaves the frame open and abort
  // retryable, and the retry succeeds once the producer stream drains.
  const auto abort_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (true) {
    const auto aborted = backend_->abort_frame(*cmds);
    if (aborted.has_value()) {
      break;
    }
    ASSERT_EQ(aborted.error().code, ErrorCode::kBackpressured) << aborted.error().message;
    ASSERT_LT(std::chrono::steady_clock::now(), abort_deadline)
        << "abort_frame never succeeded after the producer stream drained";
  }

  // ... making it legal (per the frame_data.hpp contract) to overwrite the producer
  // buffer immediately. Render a clean frame from the overwritten buffer: it must show
  // the new content, and nothing may have read the buffer mid-overwrite.
  ASSERT_EQ(cuMemsetD8(source_, 0xFF, static_cast<size_t>(kSize) * kSize * 4), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamSynchronize(nullptr), CUDA_SUCCESS);

  const ImageFrameData white{kSize, kSize, {{source_, kSize * 4}}, 1, nullptr};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render(white, pixels));
  const size_t center_i = (static_cast<size_t>(kSize / 2) * kSize + kSize / 2) * 4;
  EXPECT_EQ(pixels[center_i + 0], 0xFF);
  EXPECT_EQ(pixels[center_i + 1], 0xFF);
  EXPECT_EQ(pixels[center_i + 2], 0xFF);
}

TEST_F(HolovizBackendImageTest, RGBKernelAbortWaitsForProducerAndAllowsSourceReuse) {
  ASSERT_NO_FATAL_FAILURE(make_backend(ImageLayerDesc{ImageFormat::kR8G8B8Unorm, kSize, kSize}));
  ASSERT_NO_FATAL_FAILURE(make_source(gradient_rgba()));

  // Warm CUDA's runtime and kernel setup before holding a stream: first-use initialization
  // may synchronize the context and otherwise wait on the test's own producer gate.
  const ImageFrameData initial{kSize, kSize, {{source_, kSize * 3}}, 1, nullptr};
  std::vector<uint8_t> warmup_pixels;
  ASSERT_NO_FATAL_FAILURE(render(initial, warmup_pixels));

  // Observe entry before recording work, and keep the producer held through the first abort.
  ASSERT_EQ(cuLaunchHostFunc(producer_stream_, &hold_producer, this), CUDA_SUCCESS);
  {
    std::unique_lock lock(producer_gate_mutex_);
    ASSERT_TRUE(producer_gate_cv_.wait_for(
        lock, std::chrono::seconds(5), [&] { return producer_entered_; }));
  }

  auto cmds = backend_->begin_frame(AcquireToken{}, 0);
  ASSERT_TRUE(cmds.has_value());
  const ImageFrameData pending{kSize, kSize, {{source_, kSize * 3}}, 1, producer_stream_};
  RenderList list;
  list.layers.push_back(LayerFrame{0, &pending});
  ASSERT_TRUE(backend_->record(*cmds, list, CompositeSpec{}).has_value());

  ASSERT_EQ(cuEventRecord(rgb_reads_done_, upload_stream_), CUDA_SUCCESS);
  ASSERT_EQ(cuEventQuery(rgb_reads_done_), CUDA_ERROR_NOT_READY);

  const auto held_abort_start = std::chrono::steady_clock::now();
  const auto held_abort = backend_->abort_frame(*cmds);
  ASSERT_FALSE(held_abort.has_value()) << "abort succeeded while RGB source reads were blocked";
  EXPECT_EQ(held_abort.error().code, ErrorCode::kBackpressured);
  EXPECT_LT(std::chrono::steady_clock::now() - held_abort_start, std::chrono::seconds(5));
  EXPECT_EQ(cuEventQuery(rgb_reads_done_), CUDA_ERROR_NOT_READY);
  release_producer();

  const auto abort_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (true) {
    const auto aborted = backend_->abort_frame(*cmds);
    if (aborted.has_value()) {
      // Observe completion at the quiescing return, before anything else can drain the stream.
      ASSERT_EQ(cuEventQuery(rgb_reads_done_), CUDA_SUCCESS);
      break;
    }
    ASSERT_EQ(aborted.error().code, ErrorCode::kBackpressured) << aborted.error().message;
    ASSERT_LT(std::chrono::steady_clock::now(), abort_deadline)
        << "abort_frame never succeeded after the producer stream drained";
  }

  // ... making it legal (per the frame_data.hpp contract) to overwrite the producer
  // buffer immediately. Render a clean frame from the overwritten buffer: it must show
  // the new content, and nothing may have read the buffer mid-overwrite.
  ASSERT_EQ(cuMemsetD8(source_, 0xFF, static_cast<size_t>(kSize) * kSize * 3), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamSynchronize(nullptr), CUDA_SUCCESS);

  const ImageFrameData white{kSize, kSize, {{source_, kSize * 3}}, 1, nullptr};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render(white, pixels));
  const size_t center_i = (static_cast<size_t>(kSize / 2) * kSize + kSize / 2) * 4;
  EXPECT_EQ(pixels[center_i + 0], 0xFF);
  EXPECT_EQ(pixels[center_i + 1], 0xFF);
  EXPECT_EQ(pixels[center_i + 2], 0xFF);
}

TEST_F(HolovizBackendImageTest, ProducerStreamOrderingNeedsNoCallerSync) {
  make_backend(ImageLayerDesc{ImageFormat::kR8G8B8A8Unorm, kSize, kSize});

  // Queue a slow-ish producer write on its own stream and hand the frame over WITHOUT
  // synchronizing: the backend must order its copy after the producer via the event edge.
  // Texels are opaque (alpha 0xFF, the D32 pattern's high byte) so straight-alpha
  // blending over the clear leaves the color bytes untouched for the exact compare.
  const size_t source_bytes = static_cast<size_t>(kSize) * kSize * 4;
  const size_t source_texels = static_cast<size_t>(kSize) * kSize;
  ASSERT_EQ(cuMemAlloc(&source_, source_bytes), CUDA_SUCCESS);
  ASSERT_EQ(cuMemsetD32Async(source_, 0xFF111111u, source_texels, producer_stream_), CUDA_SUCCESS);
  // A pile of dependent writes to keep the producer stream busy.
  for (int i = 0; i < 64; ++i) {
    ASSERT_EQ(cuMemsetD32Async(source_, 0xFF111111u, source_texels, producer_stream_),
              CUDA_SUCCESS);
  }
  ASSERT_EQ(cuMemsetD32Async(source_, 0xFF777777u, source_texels, producer_stream_), CUDA_SUCCESS);

  const ImageFrameData image{kSize, kSize, {{source_, kSize * 4}}, 1, producer_stream_};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render(image, pixels));

  for (size_t i = 0; i < pixels.size(); i += 4) {
    ASSERT_EQ(pixels[i], 0x77) << "pixel " << i / 4 << " raced the producer stream";
  }
}

TEST_F(HolovizBackendImageTest, ViewTransformPlacesTheImageInItsQuadrant) {
  make_backend(ImageLayerDesc{.format = ImageFormat::kR8G8B8A8Unorm,
                              .max_width = kSize,
                              .max_height = kSize,
                              .view = ViewTransform{0.5F, 0.5F, 0.5F, 0.5F}});

  const std::vector<uint8_t> solid(static_cast<size_t>(kSize) * kSize * 4, 0xFF);
  make_source(solid);
  const ImageFrameData image{kSize, kSize, {{source_, kSize * 4}}, 1, nullptr};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render(image, pixels));

  // Bottom-right quadrant is white; everything else stays the black clear.
  size_t wrong = 0;
  for (uint32_t y = 0; y < kSize && wrong < 8; ++y) {
    for (uint32_t x = 0; x < kSize; ++x) {
      const size_t i = (static_cast<size_t>(y) * kSize + x) * 4;
      const bool in_quadrant = x >= kSize / 2 + 1 && y >= kSize / 2 + 1;
      const bool near_edge =
          (x >= kSize / 2 - 1 && x <= kSize / 2 + 1) || (y >= kSize / 2 - 1 && y <= kSize / 2 + 1);
      if (near_edge) {
        continue;  // rasterization boundary
      }
      const bool white = pixels[i] > 200;
      if (white != in_quadrant) {
        ++wrong;
        ADD_FAILURE() << "pixel (" << x << "," << y << ") white=" << white;
      }
    }
  }
  EXPECT_EQ(wrong, 0u);
}

TEST_F(HolovizBackendImageTest, SlotReuseShowsOnlyTheNewestFrame) {
  make_backend(ImageLayerDesc{ImageFormat::kR8G8B8A8Unorm, kSize, kSize});

  // Opaque alpha: this test is about slot reuse, not blending.
  std::vector<uint8_t> first(static_cast<size_t>(kSize) * kSize * 4, 0x20);
  std::vector<uint8_t> second(static_cast<size_t>(kSize) * kSize * 4, 0xE0);
  for (size_t i = 3; i < first.size(); i += 4) {
    first[i] = 0xFF;
    second[i] = 0xFF;
  }

  make_source(first);
  const ImageFrameData first_frame{kSize, kSize, {{source_, kSize * 4}}, 1, nullptr};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render(first_frame, pixels));

  make_source(second);
  const ImageFrameData second_frame{kSize, kSize, {{source_, kSize * 4}}, 1, nullptr};
  ASSERT_NO_FATAL_FAILURE(render(second_frame, pixels));

  for (size_t i = 0; i < pixels.size(); i += 4) {
    ASSERT_EQ(pixels[i], 0xE0) << "pixel " << i / 4 << " shows a stale frame";
  }
}

TEST_F(HolovizBackendImageTest, YUVRequiresExplicitConversion) {
  for (const ImageFormat format : {ImageFormat::kNV12, ImageFormat::kYUYV422}) {
    auto backend = create_holoviz_render_backend(device_, {.stream = upload_stream_});
    ASSERT_TRUE(backend.has_value());
    LayerStack bundle;
    bundle.add_image_layer("video", ImageLayerDesc{format, kSize, kSize});
    RenderConfig config;
    config.headless = true;
    config.headless_width = kSize;
    config.headless_height = kSize;
    const auto allocated = (*backend)->allocate(bundle, config);
    ASSERT_FALSE(allocated.has_value());
    EXPECT_EQ(allocated.error().code, ErrorCode::kInvalidArgument);
  }
}

TEST_F(HolovizBackendImageTest, RGBExpandsPaddedUnalignedRowsToOpaqueRGBA) {
  make_backend(ImageLayerDesc{ImageFormat::kR8G8B8Unorm, kSize, kSize});
  const auto expected = gradient_rgba();
  constexpr size_t pitch = kSize * 3 + 7;
  std::vector<uint8_t> rgb(1 + (kSize - 1) * pitch + kSize * 3, 0xAD);
  for (size_t y = 0; y < kSize; ++y) {
    for (size_t x = 0; x < kSize; ++x) {
      std::memcpy(rgb.data() + 1 + y * pitch + x * 3, expected.data() + (y * kSize + x) * 4, 3);
    }
  }
  make_source(rgb);
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(
      render({kSize, kSize, {{source_ + 1, pitch}}, 1, producer_stream_}, pixels));
  EXPECT_EQ(pixels, expected);
  const auto stats = backend_->stats();
  ASSERT_TRUE(stats.has_value());
  // Two exported RGBA slots plus two headless RGBA targets.
  EXPECT_EQ(stats->texture_pool_bytes, size_t{kSize} * kSize * 4 * 4);
}

TEST_F(HolovizBackendImageTest, RGBSmallFrameOverwritesAndClampsEveryEdge) {
  make_backend(ImageLayerDesc{ImageFormat::kR8G8B8Unorm, kSize, kSize});
  make_source(std::vector<uint8_t>(size_t{kSize} * kSize * 3, 255));
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render({kSize, kSize, {{source_, kSize * 3}}, 1, nullptr}, pixels));
  for (const auto extent : {holoviz::Extent2D{35, 19}, holoviz::Extent2D{1, 1}}) {
    std::vector<uint8_t> rgb(size_t{extent.width} * extent.height * 3);
    for (size_t i = 0; i < rgb.size(); i += 3) {
      rgb[i] = 37;
      rgb[i + 1] = 91;
      rgb[i + 2] = 173;
    }
    make_source(rgb);
    ASSERT_NO_FATAL_FAILURE(
        render({extent.width, extent.height, {{source_, extent.width * 3}}, 1, nullptr}, pixels));
    for (size_t i = 0; i < pixels.size(); i += 4) {
      ASSERT_EQ(pixels[i], 37);
      ASSERT_EQ(pixels[i + 1], 91);
      ASSERT_EQ(pixels[i + 2], 173);
      ASSERT_EQ(pixels[i + 3], 255);
    }
  }
}

class YUVImageTest
    : public HolovizBackendImageTest,
      public ::testing::WithParamInterface<std::tuple<ImageFormat, YUVModelConversion, YUVRange>> {
};

TEST_P(YUVImageTest, PitchedPlanesAndLumaParityMatchCPUReference) {
  const auto [format, model, range] = GetParam();
  make_backend(ImageLayerDesc{format, kSize, kSize, ColorSpace::kPassThrough, model, range});
  const bool nv12 = format == ImageFormat::kNV12;
  constexpr size_t y_pitch = kSize + 16;
  constexpr size_t uv_pitch = kSize + 32;
  constexpr size_t packed_pitch = kSize * 2 + 8;
  const size_t uv_offset = y_pitch * kSize + 11;
  std::vector<uint8_t> bytes(nv12 ? uv_offset + uv_pitch * (kSize / 2) : packed_pitch * kSize,
                             0xAD);
  // Asymmetric chroma detects U/V reversal; alternating luma detects packed-pixel parity.
  const auto u = [](size_t x, size_t y) { return static_cast<uint8_t>(64 + x / 2 + y / 3); };
  const auto v = [](size_t x, size_t y) { return static_cast<uint8_t>(192 - x / 3 + y / 4); };
  for (size_t y = 0; y < kSize; ++y) {
    for (size_t x = 0; x < kSize; x += 2) {
      if (nv12) {
        bytes[y * y_pitch + x] = 73;
        bytes[y * y_pitch + x + 1] = 149;
        if (y % 2 == 0) {
          bytes[uv_offset + y / 2 * uv_pitch + x] = u(x / 2, y / 2);
          bytes[uv_offset + y / 2 * uv_pitch + x + 1] = v(x / 2, y / 2);
        }
      } else {
        const size_t offset = y * packed_pitch + x * 2;
        bytes[offset] = 73;
        bytes[offset + 1] = u(x / 2, y);
        bytes[offset + 2] = 149;
        bytes[offset + 3] = v(x / 2, y);
      }
    }
  }
  make_source(bytes);
  ImageFrameData frame{
      kSize, kSize, {{source_, nv12 ? y_pitch : packed_pitch}}, nv12 ? 2U : 1U, producer_stream_};
  if (nv12) {
    frame.planes[1] = {source_ + uv_offset, uv_pitch};
  }
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render(frame, pixels));
  // Independent tabulated normalized YCbCr-to-RGB coefficients.
  const std::array<double, 4> matrix =
      model == YUVModelConversion::kBT601
          ? std::array<double, 4>{1.402, 1.772, -0.344136, -0.714136}
      : model == YUVModelConversion::kBT709
          ? std::array<double, 4>{1.5748, 1.8556, -0.187324, -0.468124}
          : std::array<double, 4>{1.4746, 1.8814, -0.164553, -0.571353};
  const bool limited = range == YUVRange::kLimited;
  const auto chroma = [nv12](auto sample, size_t x, size_t y) {
    const double cx =
        std::clamp((static_cast<double>(x) + 0.5) / 2.0 - 0.5, 0.0, kSize / 2.0 - 1.0);
    const double cy =
        nv12 ? std::clamp((static_cast<double>(y) + 0.5) / 2.0 - 0.5, 0.0, kSize / 2.0 - 1.0)
             : static_cast<double>(y);
    const auto x0 = static_cast<size_t>(cx);
    const auto y0 = static_cast<size_t>(cy);
    const auto x1 = std::min(x0 + 1, size_t{kSize / 2 - 1});
    const auto y1 = std::min(y0 + 1, size_t{nv12 ? kSize / 2 - 1 : kSize - 1});
    return std::lerp(std::lerp(static_cast<double>(sample(x0, y0)),
                               static_cast<double>(sample(x1, y0)),
                               cx - static_cast<double>(x0)),
                     std::lerp(static_cast<double>(sample(x0, y1)),
                               static_cast<double>(sample(x1, y1)),
                               cx - static_cast<double>(x0)),
                     cy - static_cast<double>(y0));
  };
  for (size_t y = 0; y < kSize; ++y) {
    for (size_t x = 0; x < kSize; ++x) {
      const double cb = (chroma(u, x, y) - 128.0) / (limited ? 224.0 : 255.0);
      const double cr = (chroma(v, x, y) - 128.0) / (limited ? 224.0 : 255.0);
      const double luma =
          (((x % 2) != 0 ? 149.0 : 73.0) - (limited ? 16.0 : 0.0)) / (limited ? 219.0 : 255.0);
      const std::array<double, 3> rgb{
          luma + matrix[0] * cr, luma + matrix[2] * cb + matrix[3] * cr, luma + matrix[1] * cb};
      const size_t offset = (y * kSize + x) * 4;
      for (size_t c = 0; c < 3; ++c) {
        ASSERT_NEAR(pixels[offset + c], std::round(std::clamp(rgb[c], 0.0, 1.0) * 255), 1);
      }
      ASSERT_EQ(pixels[offset + 3], 255);
    }
  }
}

TEST_P(YUVImageTest, RejectsInvalidPlanesBeforeUpload) {
  const auto [format, model, range] = GetParam();
  make_backend(ImageLayerDesc{format, kSize, kSize, ColorSpace::kPassThrough, model, range});
  make_source(std::vector<uint8_t>(size_t{kSize} * kSize * 2, 128));
  const bool nv12 = format == ImageFormat::kNV12;
  const ImageFrameData valid{
      kSize,
      kSize,
      {{{source_, nv12 ? kSize : kSize * 2}, {source_ + kSize * kSize, kSize}}},
      nv12 ? 2U : 1U,
      nullptr};
  for (int failure = 0; failure < 4; ++failure) {
    auto frame = valid;
    if (failure == 0) {
      frame.plane_count = nv12 ? 1 : 2;
    }
    if (failure == 1) {
      frame.planes[nv12 ? 1 : 0].device_ptr = 0;
    }
    if (failure == 2) {
      frame.planes[nv12 ? 1 : 0].row_pitch_bytes = 1;
    }
    if (failure == 3) {
      frame.width -= 1;
    }
    auto cmd = backend_->begin_frame(AcquireToken{}, 0);
    ASSERT_TRUE(cmd);
    RenderList list;
    list.layers.push_back({0, &frame});
    const auto recorded = backend_->record(*cmd, list, {});
    ASSERT_FALSE(recorded);
    EXPECT_EQ(recorded.error().code, ErrorCode::kInvalidArgument);
    ASSERT_TRUE(backend_->abort_frame(*cmd));
  }
}

// Linear sampling of a frame smaller than the declared max must not blend in stale chroma from
// beyond the frame edge. The primer's chroma has to be far from neutral: with 128 everywhere a
// leak would still decode to gray and the black/white checks below could never fail.
TEST_P(YUVImageTest, SmallFrameNeutralBlackAndWhiteDoNotBleedStaleChroma) {
  const auto [format, model, range] = GetParam();
  make_backend(ImageLayerDesc{format, kSize, kSize, ColorSpace::kPassThrough, model, range});
  const bool nv12 = format == ImageFormat::kNV12;
  const bool limited = range == YUVRange::kLimited;
  std::vector<uint8_t> pixels;

  // Build an extent x extent frame with the given luma everywhere and (u, v) in every chroma pair.
  const auto make_frame = [&](uint32_t extent, uint8_t luma, uint8_t u, uint8_t v) {
    std::vector<uint8_t> bytes(size_t{extent} * extent * (nv12 ? 3 : 4) / 2);
    if (nv12) {
      std::fill_n(bytes.begin(), size_t{extent} * extent, luma);
      for (size_t i = size_t{extent} * extent; i < bytes.size(); i += 2) {
        bytes[i] = u;
        bytes[i + 1] = v;
      }
    } else {
      for (size_t i = 0; i < bytes.size(); i += 4) {  // Y0 U Y1 V
        bytes[i] = luma;
        bytes[i + 1] = u;
        bytes[i + 2] = luma;
        bytes[i + 3] = v;
      }
    }
    make_source(bytes);
    return ImageFrameData{
        extent,
        extent,
        {{{source_, nv12 ? extent : extent * 2}, {source_ + extent * extent, extent}}},
        nv12 ? 2U : 1U,
        nullptr};
  };

  // Prime slot 0's full texture with saturated chroma so any texel sampled from beyond a smaller
  // frame's edge is strongly tinted rather than gray.
  ASSERT_NO_FATAL_FAILURE(render(make_frame(kSize, limited ? 126 : 128, 255, 0), pixels));

  for (const uint32_t extent : {18U, 2U}) {
    for (const bool white : {false, true}) {
      const uint8_t luma = white ? (limited ? 235 : 255) : (limited ? 16 : 0);
      ASSERT_NO_FATAL_FAILURE(render(make_frame(extent, luma, 128, 128), pixels));
      for (size_t i = 0; i < pixels.size(); i += 4) {
        ASSERT_EQ(pixels[i], white ? 255 : 0) << "extent " << extent << " pixel " << i / 4;
        ASSERT_EQ(pixels[i + 1], white ? 255 : 0) << "extent " << extent << " pixel " << i / 4;
        ASSERT_EQ(pixels[i + 2], white ? 255 : 0) << "extent " << extent << " pixel " << i / 4;
        ASSERT_EQ(pixels[i + 3], 255);
      }
      maybe_dump(
          "image_yuv_chroma_bleed", pixels, {}, holoviz::Extent2D{kSize, kSize}, HasFailure());
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    AllConversions, YUVImageTest,
    ::testing::Combine(::testing::Values(ImageFormat::kNV12, ImageFormat::kYUYV422),
                       ::testing::Values(YUVModelConversion::kBT601, YUVModelConversion::kBT709,
                                         YUVModelConversion::kBT2020),
                       ::testing::Values(YUVRange::kLimited, YUVRange::kFull)));

// FF38: linear sampling of a frame smaller than the declared max must never blend in
// stale texels beyond the frame edge. Prime the slot texture with white via a full-size
// frame, then render a smaller all-red frame into the same slot: with the half-texel
// UV mapping every output pixel samples only valid (red) texels; pre-fix the right/bottom
// edges blended red with the stale white.
TEST_F(HolovizBackendImageTest, SmallFrameEdgesDoNotBleedStaleTexels) {
  make_backend(ImageLayerDesc{ImageFormat::kR8G8B8A8Unorm, kSize, kSize});

  const std::vector<uint8_t> white(static_cast<size_t>(kSize) * kSize * 4, 0xFF);
  make_source(white);
  const ImageFrameData full{kSize, kSize, {{source_, kSize * 4}}, 1, nullptr};
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render(full, pixels));

  constexpr uint32_t kSmall = kSize / 2;
  std::vector<uint8_t> red(static_cast<size_t>(kSmall) * kSmall * 4);
  for (size_t i = 0; i < red.size(); i += 4) {
    red[i + 0] = 0xFF;
    red[i + 1] = 0x00;
    red[i + 2] = 0x00;
    red[i + 3] = 0xFF;
  }
  make_source(red);
  const ImageFrameData small{kSmall, kSmall, {{source_, kSmall * 4}}, 1, nullptr};
  ASSERT_NO_FATAL_FAILURE(render(small, pixels));

  // Probe the extreme output pixels (the ones whose UVs sit closest to the valid-region
  // boundary). Any green/blue contamination means a stale (white) texel was sampled.
  const auto probe = [&](uint32_t x, uint32_t y) {
    const size_t i = (static_cast<size_t>(y) * kSize + x) * 4;
    EXPECT_EQ(pixels[i + 0], 0xFF) << "at " << x << "," << y;
    EXPECT_EQ(pixels[i + 1], 0x00) << "at " << x << "," << y;
    EXPECT_EQ(pixels[i + 2], 0x00) << "at " << x << "," << y;
  };
  probe(kSize - 1, kSize / 2);  // right edge
  probe(kSize / 2, kSize - 1);  // bottom edge
  probe(kSize - 1, kSize - 1);  // corner
  probe(0, 0);                  // top-left stays valid too
  maybe_dump("image_edge_bleed", pixels, {}, holoviz::Extent2D{kSize, kSize}, HasFailure());
}

// FF42: a duplicate layer index in the composite order is rejected up front — recording
// the same CUDA-upload layer twice in one frame would double-copy into a pending slot.
TEST_F(HolovizBackendImageTest, DuplicateCompositeIndexIsRejected) {
  make_backend(ImageLayerDesc{ImageFormat::kR8G8B8A8Unorm, kSize, kSize});
  make_source(gradient_rgba());
  const ImageFrameData image{kSize, kSize, {{source_, kSize * 4}}, 1, nullptr};

  auto cmds = backend_->begin_frame(AcquireToken{}, 0);
  ASSERT_TRUE(cmds.has_value()) << cmds.error().message;
  RenderList list;
  list.layers.push_back(LayerFrame{0, &image});
  CompositeSpec composite;
  composite.order = {0, 0};
  const auto recorded = backend_->record(*cmds, list, composite);
  ASSERT_FALSE(recorded.has_value());
  EXPECT_EQ(recorded.error().code, ErrorCode::kInvalidArgument);
  // Recover the slot per the documented error-path contract.
  ASSERT_TRUE(backend_->abort_frame(*cmds).has_value());
}

}  // namespace
}  // namespace holoscan::visualization::backends
