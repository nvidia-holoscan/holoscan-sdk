// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Image layer tests for RhiRenderBackend: a CUDA-resident RGBA frame must survive the
// upload + draw byte-exactly (1:1 mapping, nearest-equivalent sampling), producer-stream
// ordering must hold without any caller synchronization, and slot reuse must show the
// newest frame only.

#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include <cuda.h>

#include <holoviz/backends/rhi_render_backend.hpp>
#include <holoviz/frame_data.hpp>
#include <holoviz/layer_desc.hpp>
#include <holoviz/render_backend.hpp>
#include <holoviz/rhi.hpp>

#include "holoviz/image_format.hpp"
#include "image_compare_test_util.hpp"

// Module-private headers (src/ is on this target's include path).
#include "backends/backend_tokens.hpp"
#include "backends/rhi_render_backend.hpp"

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"

namespace holoviz::backends {
namespace {

using test_util::maybe_dump;
using test_util::try_create_device;

constexpr uint32_t kSize = 128;  // target and declared image max: 1:1 texel-to-pixel

class RhiBackendImageTest : public ::testing::Test {
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
  }

  void TearDown() override {
    backend_.reset();
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

  void make_backend(ImageLayerDesc spec) {
    auto backend = create_rhi_render_backend(device_, {});
    ASSERT_TRUE(backend.has_value());
    backend_ = std::move(*backend);

    LayerStack bundle;
    bundle.add_image_layer("video", std::move(spec));

    RenderConfig config;
    config.headless = true;
    config.headless_width = kSize;
    config.headless_height = kSize;
    config.max_frames_in_flight = 2;
    ASSERT_TRUE(backend_->allocate(bundle, config).has_value());
    ASSERT_TRUE(backend_->build(ImageFormat::kB8G8R8A8Unorm).has_value());
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
    auto cmds = backend_->begin_frame(AcquireToken{}, 0);
    ASSERT_TRUE(cmds.has_value()) << cmds.error().message;
    RenderList list;
    list.layers.push_back(LayerFrame{0, &image});
    ASSERT_TRUE(backend_->record(*cmds, list, CompositeSpec{}).has_value());
    auto token = backend_->submit(*cmds, SubmitContext{0});
    ASSERT_TRUE(token.has_value()) << token.error().message;
    ASSERT_TRUE(device_
                    .wait_on_submit(submit_payload(*token, device_identity(device_)).payload->token,
                                    std::chrono::seconds(5))
                    .has_value());

    auto* concrete = static_cast<RhiRenderBackend*>(backend_.get());
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

  rhi::Device device_;
  std::unique_ptr<RenderBackend> backend_;
  CUstream producer_stream_ = nullptr;
  CUdeviceptr source_ = 0;
};

TEST_F(RhiBackendImageTest, RgbaFrameRoundTripsByteExact) {
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
  maybe_dump("image_rgba", pixels, expected, rhi::Extent2D{kSize, kSize}, HasFailure());
}

TEST_F(RhiBackendImageTest, AbortAfterEnqueuedUploadsReleasesFrameDataImmediately) {
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

TEST_F(RhiBackendImageTest, ProducerStreamOrderingNeedsNoCallerSync) {
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

TEST_F(RhiBackendImageTest, ViewTransformPlacesTheImageInItsQuadrant) {
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

TEST_F(RhiBackendImageTest, SlotReuseShowsOnlyTheNewestFrame) {
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

TEST_F(RhiBackendImageTest, DeferredFormatsFailAllocateCleanly) {
  for (const ImageFormat format :
       {ImageFormat::kNV12, ImageFormat::kYUYV422, ImageFormat::kR8G8B8Unorm}) {
    auto backend = create_rhi_render_backend(device_, {});
    ASSERT_TRUE(backend.has_value());

    LayerStack bundle;
    bundle.add_image_layer("video", ImageLayerDesc{format, kSize, kSize});
    RenderConfig config;
    config.headless = true;
    config.headless_width = kSize;
    config.headless_height = kSize;
    const auto allocated = (*backend)->allocate(bundle, config);
    ASSERT_FALSE(allocated.has_value());
    EXPECT_EQ(allocated.error().code, ErrorCode::kNotImplemented);
  }
}

// FF38: linear sampling of a frame smaller than the declared max must never blend in
// stale texels beyond the frame edge. Prime the slot texture with white via a full-size
// frame, then render a smaller all-red frame into the same slot: with the half-texel
// UV mapping every output pixel samples only valid (red) texels; pre-fix the right/bottom
// edges blended red with the stale white.
TEST_F(RhiBackendImageTest, SmallFrameEdgesDoNotBleedStaleTexels) {
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
  maybe_dump("image_edge_bleed", pixels, {}, rhi::Extent2D{kSize, kSize}, HasFailure());
}

// FF42: a duplicate layer index in the composite order is rejected up front — recording
// the same CUDA-upload layer twice in one frame would double-copy into a pending slot.
TEST_F(RhiBackendImageTest, DuplicateCompositeIndexIsRejected) {
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
}  // namespace holoviz::backends
