// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// LUT layer tests for HolovizRenderBackend: known index bands must colorize to exact palette
// entries, and an empty per-frame lut span must keep the previously supplied palette even
// across frame-slot cycling.

#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include <cuda.h>

#include <holoscan/visualization/holoviz_op.hpp>
#include <holoviz/holoviz.hpp>
#include "holoviz_op/backends/frame_data.hpp"
#include "holoviz_op/backends/holoviz_render_backend.hpp"
#include "holoviz_op/backends/render_backend.hpp"

#include "holoviz_op/backends/handles.hpp"
#include "image_compare_test_util.hpp"

// Module-private headers (src/ is on this target's include path).
#include "holoviz_op/backends/backend_tokens.hpp"

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"

namespace holoscan::visualization::backends {

namespace test_util = holoviz::test_util;
namespace {

using test_util::try_create_device;

constexpr uint32_t kSize = 64;
constexpr uint32_t kLutSize = 4;

// 4 horizontal bands of indices 0..3.
std::vector<uint8_t> band_mask() {
  std::vector<uint8_t> mask(static_cast<size_t>(kSize) * kSize);
  for (uint32_t y = 0; y < kSize; ++y) {
    for (uint32_t x = 0; x < kSize; ++x) {
      mask[static_cast<size_t>(y) * kSize + x] = static_cast<uint8_t>(y / (kSize / kLutSize));
    }
  }
  return mask;
}

// Opaque, saturated, distinct.
constexpr float kPalette[kLutSize * 4] = {
    1.0F,
    0.0F,
    0.0F,
    1.0F,  // 0: red
    0.0F,
    1.0F,
    0.0F,
    1.0F,  // 1: green
    0.0F,
    0.0F,
    1.0F,
    1.0F,  // 2: blue
    1.0F,
    1.0F,
    0.0F,
    1.0F,  // 3: yellow
};

class HolovizBackendLutTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_ = try_create_device();
    if (!device_) {
      HOLOVIZ_ENV_SKIP() << "no usable rendering device/adapter.";
    }
    const auto context = device_.cuda_context();
    ASSERT_TRUE(context.has_value());
    ASSERT_EQ(cuCtxSetCurrent(*context), CUDA_SUCCESS);

    auto backend = create_holoviz_render_backend(device_, {});
    ASSERT_TRUE(backend.has_value());
    backend_ = std::move(*backend);

    LayerStack bundle;
    bundle.add_lut_layer(
        "mask", LUTLayerDesc{.lut_size = kLutSize, .max_width = kSize, .max_height = kSize});

    RenderConfig config;
    config.headless = true;
    config.headless_width = kSize;
    config.headless_height = kSize;
    config.max_frames_in_flight = 2;
    ASSERT_TRUE(backend_->allocate(bundle, config).has_value());
    ASSERT_TRUE(backend_->build(ImageFormat::kB8G8R8A8Unorm).has_value());

    const std::vector<uint8_t> mask = band_mask();
    ASSERT_EQ(cuMemAlloc(&mask_ptr_, mask.size()), CUDA_SUCCESS);
    ASSERT_EQ(cuMemcpyHtoD(mask_ptr_, mask.data(), mask.size()), CUDA_SUCCESS);
  }

  void TearDown() override {
    backend_.reset();
    if (mask_ptr_ != 0) {
      cuMemFree(mask_ptr_);
    }
    if (device_) {
      ASSERT_TRUE(device_.destroy(std::chrono::seconds(5)).has_value());
    }
  }

  void render(std::span<const float> lut, std::vector<uint8_t>& out_pixels) {
    auto cmds = backend_->begin_frame(AcquireToken{}, 0);
    ASSERT_TRUE(cmds.has_value()) << cmds.error().message;
    const LutFrameData frame{kSize, kSize, {mask_ptr_, kSize}, nullptr, lut};
    RenderList list;
    list.layers.push_back(LayerFrame{0, &frame});
    ASSERT_TRUE(backend_->record(*cmds, list, CompositeSpec{}).has_value());
    auto token = backend_->submit(*cmds, SubmitContext{0});
    ASSERT_TRUE(token.has_value());
    ASSERT_TRUE(device_
                    .wait_on_submit(submit_payload(*token, device_identity(device_)).payload->token,
                                    std::chrono::seconds(5))
                    .has_value());

    auto* concrete = static_cast<HolovizRenderBackend*>(backend_.get());
    out_pixels.resize(static_cast<size_t>(kSize) * kSize * 4);
    ASSERT_TRUE(device_
                    .download_from_image(concrete->headless_target(0)->get(),
                                         0,
                                         std::as_writable_bytes(std::span(out_pixels)),
                                         static_cast<size_t>(kSize) * 4)
                    .has_value());
  }

  static void expect_bands(const std::vector<uint8_t>& pixels,
                           const float (&palette)[kLutSize * 4]) {
    size_t wrong = 0;
    for (uint32_t y = 0; y < kSize && wrong < 8; ++y) {
      const uint32_t band = y / (kSize / kLutSize);
      for (uint32_t x = 0; x < kSize; ++x) {
        const size_t i = (static_cast<size_t>(y) * kSize + x) * 4;
        for (uint32_t c = 0; c < 3; ++c) {
          const int expected = static_cast<int>(palette[band * 4 + c] * 255.0F + 0.5F);
          if (std::abs(int{pixels[i + c]} - expected) > 1) {
            ++wrong;
            ADD_FAILURE() << "pixel (" << x << "," << y << ") channel " << c << ": got "
                          << int{pixels[i + c]} << " want " << expected;
            break;
          }
        }
      }
    }
    EXPECT_EQ(wrong, 0u);
  }

  holoviz::Device device_;
  std::unique_ptr<RenderBackend> backend_;
  CUdeviceptr mask_ptr_ = 0;
};

TEST_F(HolovizBackendLutTest, IndexBandsColorizeToExactPaletteEntries) {
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render(kPalette, pixels));
  expect_bands(pixels, kPalette);
}

TEST_F(HolovizBackendLutTest, EmptyLutKeepsThePreviousPaletteAcrossSlots) {
  std::vector<uint8_t> pixels;
  ASSERT_NO_FATAL_FAILURE(render(kPalette, pixels));

  // Several palette-less frames (cycling both frame slots): the original palette must hold.
  for (int i = 0; i < 3; ++i) {
    ASSERT_NO_FATAL_FAILURE(render({}, pixels));
  }
  expect_bands(pixels, kPalette);

  // A new palette replaces it.
  constexpr float kInverted[kLutSize * 4] = {
      1.0F,
      1.0F,
      0.0F,
      1.0F,
      0.0F,
      0.0F,
      1.0F,
      1.0F,
      0.0F,
      1.0F,
      0.0F,
      1.0F,
      1.0F,
      0.0F,
      0.0F,
      1.0F,
  };
  ASSERT_NO_FATAL_FAILURE(render(kInverted, pixels));
  expect_bands(pixels, kInverted);
}

TEST_F(HolovizBackendLutTest, FirstFrameWithoutAPaletteIsRejected) {
  auto cmds = backend_->begin_frame(AcquireToken{}, 0);
  ASSERT_TRUE(cmds.has_value());
  const LutFrameData frame{kSize, kSize, {mask_ptr_, kSize}, nullptr, {}};
  RenderList list;
  list.layers.push_back(LayerFrame{0, &frame});
  const auto recorded = backend_->record(*cmds, list, CompositeSpec{});
  ASSERT_FALSE(recorded.has_value());
  EXPECT_EQ(recorded.error().code, ErrorCode::kInvalidState);
  // kInvalidState is not a prevalidation rejection, so abort_frame() is the only sound
  // recovery here (record() contract).
  ASSERT_TRUE(backend_->abort_frame(*cmds).has_value());
}

TEST_F(HolovizBackendLutTest, BackloggedProducerStreamDoesNotCorruptEarlierPalette) {
  // Deterministic reproduction of the shared-staging race: frame 0's palette HtoD copy is
  // queued on the backend stream behind a producer-stream event that a host function holds
  // back for ~150 ms; frame 1 stages its (different) palette immediately afterwards. With
  // per-slot staging, frame 0 must still render its own palette; with shared staging,
  // frame 1's memcpy overwrites the block before frame 0's copy executes.
  CUstream producer = nullptr;
  ASSERT_EQ(cuStreamCreate(&producer, CU_STREAM_DEFAULT), CUDA_SUCCESS);
  const CUhostFn sleep_fn = [](void*) {
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
  };
  ASSERT_EQ(cuLaunchHostFunc(producer, sleep_fn, nullptr), CUDA_SUCCESS);

  constexpr float kOtherPalette[kLutSize * 4] = {
      1.0F,
      0.0F,
      1.0F,
      1.0F,  // 0: magenta
      0.0F,
      1.0F,
      1.0F,
      1.0F,  // 1: cyan
      1.0F,
      1.0F,
      1.0F,
      1.0F,  // 2: white
      0.0F,
      0.0F,
      0.0F,
      1.0F,  // 3: black
  };

  SyncToken tokens[2];
  const std::span<const float> palettes[2] = {kPalette, kOtherPalette};
  const CUstream streams[2] = {producer, nullptr};
  for (uint32_t slot = 0; slot < 2; ++slot) {
    auto cmds = backend_->begin_frame(AcquireToken{}, slot);
    ASSERT_TRUE(cmds.has_value()) << cmds.error().message;
    const LutFrameData frame{kSize, kSize, {mask_ptr_, kSize}, streams[slot], palettes[slot]};
    RenderList list;
    list.layers.push_back(LayerFrame{0, &frame});
    ASSERT_TRUE(backend_->record(*cmds, list, CompositeSpec{}).has_value());
    auto token = backend_->submit(*cmds, SubmitContext{slot});
    ASSERT_TRUE(token.has_value()) << token.error().message;
    tokens[slot] = *token;
  }

  auto* concrete = static_cast<HolovizRenderBackend*>(backend_.get());
  ASSERT_TRUE(
      device_
          .wait_on_submit(submit_payload(tokens[0], device_identity(device_)).payload->token,
                          std::chrono::seconds(5))
          .has_value());
  ASSERT_TRUE(
      device_
          .wait_on_submit(submit_payload(tokens[1], device_identity(device_)).payload->token,
                          std::chrono::seconds(5))
          .has_value());

  std::vector<uint8_t> pixels(static_cast<size_t>(kSize) * kSize * 4);
  ASSERT_TRUE(device_
                  .download_from_image(concrete->headless_target(0)->get(),
                                       0,
                                       std::as_writable_bytes(std::span(pixels)),
                                       static_cast<size_t>(kSize) * 4)
                  .has_value());
  expect_bands(pixels, kPalette);

  ASSERT_TRUE(device_
                  .download_from_image(concrete->headless_target(1)->get(),
                                       0,
                                       std::as_writable_bytes(std::span(pixels)),
                                       static_cast<size_t>(kSize) * 4)
                  .has_value());
  expect_bands(pixels, kOtherPalette);

  ASSERT_EQ(cuStreamSynchronize(producer), CUDA_SUCCESS);
  ASSERT_EQ(cuStreamDestroy(producer), CUDA_SUCCESS);
}

}  // namespace
}  // namespace holoscan::visualization::backends
