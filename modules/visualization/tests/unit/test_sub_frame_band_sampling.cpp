/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Regression for the sub_frame example's band draws (examples/core/sub_frame): a band's draw
// must read only the source rows that band's CUDA upload wrote. The scissor bounds the
// destination pixels, but with a linear sampler a full-frame UV at a band edge blends in the
// neighbouring band's rows — which may still hold the previous frame while CUDA is (or is not
// yet) uploading them. The example's real shader is compiled into this test; the band pipeline
// is reproduced offscreen (swapchain images cannot be read back) with the next band's upload
// deliberately parked behind a host-gated CUDA stream, at nonintegral source→output scales.

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <span>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda.h>

#include <holoviz/holoviz.hpp>

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"
#include "shader.slang.hpp"      // generated from the example's shader.slang at build time
#include "shader_interface.hpp"  // the example's PushConstants

namespace holoviz {
namespace {

namespace shader = holoviz::examples::sub_frame;

// The example's simulated sensor frame size.
constexpr uint32_t kSourceWidth = 800;
constexpr uint32_t kSourceHeight = 600;

/// Rows [y0, y1) of a height split into equal bands (the example's band_offset()).
uint32_t band_offset(uint32_t band, uint32_t band_count, uint32_t height) {
  return static_cast<uint32_t>(static_cast<uint64_t>(band) * height / band_count);
}

/// Host function parked on the CUDA stream: holds everything queued behind it until the test
/// flips the flag — the deterministic "next band still uploading" state.
void block_stream_until_released(void* user_data) {
  const auto* released = static_cast<std::atomic<bool>*>(user_data);
  while (!released->load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
}

struct Rgba {
  uint8_t r, g, b, a;
  uint32_t packed() const {
    return static_cast<uint32_t>(r) | (static_cast<uint32_t>(g) << 8) |
           (static_cast<uint32_t>(b) << 16) | (static_cast<uint32_t>(a) << 24);
  }
};

/// Solid, row-constant colour keyed by (frame, band): a stale row is unmistakable after
/// filtering, unlike the example's smoothly varying pattern. The frame lives in red; the band
/// index is spread over green (band / 64) and blue (band % 64) in steps of 4, so every band below
/// 4096 has a distinct colour and neighbouring codes differ by more than the readback tolerance.
Rgba band_color(uint32_t frame, uint32_t band) {
  const auto g = static_cast<uint8_t>(4 * (band / 64));
  const auto b = static_cast<uint8_t>(4 * (band % 64));
  return Rgba{static_cast<uint8_t>(frame == 0 ? 220 : 20), g, b, 255};
}

struct Case {
  uint32_t output_height;
  uint32_t bands;
  uint32_t band;
};

class SubFrameBandSamplingTest : public ::testing::TestWithParam<Case> {
 protected:
  void SetUp() override {
    auto adapters = enumerate_adapters();
    if (!adapters || adapters->empty()) {
      HOLOVIZ_ENV_SKIP() << "no Vulkan adapter";
    }
    auto device = adapters->front().create_device({});
    if (!device) {
      HOLOVIZ_ENV_SKIP() << "no device: " << device.error().message;
    }
    device_ = std::move(*device);

    auto cuda_context = device_.cuda_context();
    if (!cuda_context) {
      HOLOVIZ_ENV_SKIP() << "no CUDA context: " << cuda_context.error().message;
    }
    ASSERT_EQ(cuCtxPushCurrent(*cuda_context), CUDA_SUCCESS);
    context_pushed_ = true;
    ASSERT_EQ(cuStreamCreate(&cuda_stream_, CU_STREAM_NON_BLOCKING), CUDA_SUCCESS);
  }

  void TearDown() override {
    release_stream();
    if (device_) {
      EXPECT_TRUE(device_.wait_idle().has_value());
    }
    if (external_semaphore_) {
      EXPECT_TRUE(external_semaphore_.destroy(std::chrono::seconds(5)).has_value());
    }
    if (band_buffer_ != 0) {
      EXPECT_EQ(cuMemFree(band_buffer_), CUDA_SUCCESS);
    }
    if (cuda_stream_ != nullptr) {
      EXPECT_EQ(cuStreamDestroy(cuda_stream_), CUDA_SUCCESS);
    }
    if (context_pushed_) {
      CUcontext popped = nullptr;
      EXPECT_EQ(cuCtxPopCurrent(&popped), CUDA_SUCCESS);
    }
  }

  void release_stream() {
    stream_released_.store(true);
    if (cuda_stream_ != nullptr) {
      EXPECT_EQ(cuStreamSynchronize(cuda_stream_), CUDA_SUCCESS);
    }
  }

  /// Queue the CUDA upload of one band's rows of \p frame's colour (the example's memcpy
  /// into the exported array), on cuda_stream_.
  void upload_band(CUarray level0, uint32_t frame, uint32_t bands, uint32_t band) {
    const uint32_t y0 = band_offset(band, bands, kSourceHeight);
    const uint32_t band_height = band_offset(band + 1, bands, kSourceHeight) - y0;
    ASSERT_EQ(cuMemsetD32Async(band_buffer_,
                               band_color(frame, band).packed(),
                               static_cast<size_t>(kSourceWidth) * band_height,
                               cuda_stream_),
              CUDA_SUCCESS);
    CUDA_MEMCPY2D copy_desc{};
    copy_desc.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy_desc.srcDevice = band_buffer_;
    copy_desc.srcPitch = static_cast<size_t>(kSourceWidth) * 4;
    copy_desc.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy_desc.dstArray = level0;
    copy_desc.dstY = y0;
    copy_desc.WidthInBytes = static_cast<size_t>(kSourceWidth) * 4;
    copy_desc.Height = band_height;
    ASSERT_EQ(cuMemcpy2DAsync(&copy_desc, cuda_stream_), CUDA_SUCCESS);
  }

  Device device_;
  bool context_pushed_ = false;
  CUstream cuda_stream_ = nullptr;
  CUdeviceptr band_buffer_ = 0;
  ExternalSemaphore external_semaphore_;
  std::atomic<bool> stream_released_{false};
};

TEST_P(SubFrameBandSamplingTest, BandDrawReadsOnlyItsOwnUploadedRows) {
  const Case c = GetParam();
  const Extent2D output{kSourceWidth, c.output_height};

  // --- The example's resources: exported source image, sampler, pipeline, offscreen target
  // standing in for the swapchain image. ---
  auto image = device_.create_image({
      .extent = {kSourceWidth, kSourceHeight},
      .format = Format::kR8G8B8A8Unorm,
      .usage = ImageUsage::kSampled,
      .share = ShareMode::kExported,
  });
  ASSERT_TRUE(image.has_value()) << image.error().message;
  auto level0 = image->cuda_mipmapped_array_get_level(0);
  ASSERT_TRUE(level0.has_value()) << level0.error().message;
  ASSERT_EQ(cuMemAlloc(&band_buffer_, static_cast<size_t>(kSourceWidth) * 4 * kSourceHeight),
            CUDA_SUCCESS);

  auto target = device_.create_image({
      .extent = output,
      .format = Format::kR8G8B8A8Unorm,
      .usage = ImageUsage::kRenderTarget,
  });
  ASSERT_TRUE(target.has_value()) << target.error().message;
  auto render_pass =
      device_.create_render_pass({.render_targets = {{.format = Format::kR8G8B8A8Unorm}}});
  ASSERT_TRUE(render_pass.has_value()) << render_pass.error().message;
  ASSERT_TRUE(render_pass->set_render_image(*target, 0, {0.0F, 0.0F, 0.0F, 1.0F}).has_value());

  auto pipeline = device_.create_pipeline({
      .shader_blob = shader::make_shader_blob(),
      .topology = PrimitiveTopology::kTriangleList,
      .raster = RasterState{.cull = CullMode::kNone},
      .depth = DepthState{.test = false, .write = false, .compare = CompareOp::kAlways},
  });
  ASSERT_TRUE(pipeline.has_value()) << pipeline.error().message;
  auto descriptor = pipeline->create_descriptor();
  ASSERT_TRUE(descriptor.has_value()) << descriptor.error().message;
  ASSERT_TRUE(descriptor->bind_image(*image, "frameTexture").has_value());
  ASSERT_TRUE(descriptor->bind_sampler(device_, {}, "frameSampler").has_value());  // linear

  auto external_semaphore = device_.create_external_semaphore(cuda_stream_);
  ASSERT_TRUE(external_semaphore.has_value()) << external_semaphore.error().message;
  external_semaphore_ = std::move(*external_semaphore);

  // --- Frame 0: every band lands, so the whole texture holds frame-0 colours. ---
  for (uint32_t band = 0; band < c.bands; ++band) {
    upload_band(*level0, 0, c.bands, band);
  }
  ASSERT_EQ(cuStreamSynchronize(cuda_stream_), CUDA_SUCCESS);

  // --- Frame 1: only the band under test lands; every other band's upload is parked behind
  // the host gate, so its rows still hold frame 0 while this band is drawn. ---
  upload_band(*level0, 1, c.bands, c.band);
  auto external_token = external_semaphore_.signal();
  ASSERT_TRUE(external_token.has_value()) << external_token.error().message;
  ASSERT_EQ(cuLaunchHostFunc(cuda_stream_, block_stream_until_released, &stream_released_),
            CUDA_SUCCESS);
  for (uint32_t band = 0; band < c.bands; ++band) {
    if (band != c.band) {
      upload_band(*level0, 1, c.bands, band);
    }
  }

  // The example's band draw: full-extent viewport (full-frame UVs), scissor to the band's
  // output rows, and the band's source-row bounds as push constants.
  const uint32_t y0 = band_offset(c.band, c.bands, kSourceHeight);
  const uint32_t band_height = band_offset(c.band + 1, c.bands, kSourceHeight) - y0;
  const uint32_t window_y0 = band_offset(c.band, c.bands, output.height);
  const uint32_t window_height = band_offset(c.band + 1, c.bands, output.height) - window_y0;
  const Rect2D full_area{0, 0, output.width, output.height};
  const Rect2D band_rect{0, static_cast<int32_t>(window_y0), output.width, window_height};
  const shader::PushConstants push_constants{
      .uv_min_y = (static_cast<float>(y0) + 0.5F) / static_cast<float>(kSourceHeight),
      .uv_max_y = (static_cast<float>(y0 + band_height) - 0.5F) / static_cast<float>(kSourceHeight),
  };

  auto command_list = device_.create_command_list({.type = CommandListType::kTransient});
  ASSERT_TRUE(command_list.has_value()) << command_list.error().message;
  ASSERT_TRUE(command_list->begin().has_value());
  ASSERT_TRUE(command_list->begin_render(*render_pass, full_area).has_value());
  ASSERT_TRUE(command_list->set_viewport(full_area).has_value());
  ASSERT_TRUE(command_list->set_scissor(band_rect).has_value());
  ASSERT_TRUE(command_list->bind_pipeline(*pipeline).has_value());
  ASSERT_TRUE(command_list->bind_descriptor(*descriptor).has_value());
  ASSERT_TRUE(command_list
                  ->set_push_data(*pipeline,
                                  std::span{reinterpret_cast<const std::byte*>(&push_constants),
                                            sizeof(push_constants)})
                  .has_value());
  ASSERT_TRUE(command_list->draw(3).has_value());
  ASSERT_TRUE(command_list->end_render().has_value());
  ASSERT_TRUE(command_list->end().has_value());

  const SubmitDesc submit_desc{
      .wait_external_sync_tokens = {&*external_token, 1},
      .command_lists = {&*command_list, 1},
  };
  const auto submitted = device_.submit(submit_desc);
  ASSERT_TRUE(submitted.has_value()) << submitted.error().message;
  ASSERT_TRUE(device_.wait_on_submit(*submitted, std::chrono::seconds(10)).has_value());

  const size_t row_pitch = static_cast<size_t>(output.width) * 4;
  std::vector<uint8_t> pixels(row_pitch * output.height);
  ASSERT_TRUE(
      device_.download_from_image(*target, 0, std::as_writable_bytes(std::span(pixels)), row_pitch)
          .has_value());

  // --- Every pixel of the band's output rows is this band's frame-1 colour: in particular
  // the first and last rows, whose full-frame UV straddles a neighbouring (frame-0) row. ---
  const Rgba expected = band_color(1, c.band);
  auto check_row = [&](uint32_t row) {
    const uint8_t* p = pixels.data() + static_cast<size_t>(row) * row_pitch;
    for (uint32_t x = 0; x < output.width; x += 7) {
      const uint8_t* px = p + static_cast<size_t>(x) * 4;
      const bool ok = std::abs(int{px[0]} - int{expected.r}) <= 1 &&
                      std::abs(int{px[1]} - int{expected.g}) <= 1 &&
                      std::abs(int{px[2]} - int{expected.b}) <= 1;
      ASSERT_TRUE(ok) << "row " << row << " x " << x << ": got (" << int{px[0]} << ", "
                      << int{px[1]} << ", " << int{px[2]} << ") expected (" << int{expected.r}
                      << ", " << int{expected.g} << ", " << int{expected.b}
                      << ") — the band draw sampled a neighbouring band's stale rows";
    }
  };
  for (uint32_t row = window_y0; row < window_y0 + window_height; ++row) {
    check_row(row);
    if (HasFatalFailure()) {
      break;
    }
  }
  // And the scissor kept the draw out of the other rows: every channel of the neighbouring rows
  // is the (black) clear colour. Red is what tells any band colour apart from the clear, so a
  // full-viewport overdraw cannot pass this by matching green alone.
  auto expect_clear_row = [&](uint32_t row) {
    const uint8_t* p = pixels.data() + static_cast<size_t>(row) * row_pitch;
    for (uint32_t x = 0; x < output.width; x += 7) {
      EXPECT_EQ(p[x * 4 + 0], 0U) << "row " << row << " x " << x << " red: band draw leaked";
      EXPECT_EQ(p[x * 4 + 1], 0U) << "row " << row << " x " << x << " green";
      EXPECT_EQ(p[x * 4 + 2], 0U) << "row " << row << " x " << x << " blue";
    }
  };
  if (window_y0 > 0) {
    expect_clear_row(window_y0 - 1);
  }
  if (window_y0 + window_height < output.height) {
    expect_clear_row(window_y0 + window_height);
  }

  release_stream();
  EXPECT_TRUE(device_.wait_idle().has_value());
  EXPECT_TRUE(command_list->destroy().has_value());
  EXPECT_TRUE(descriptor->destroy().has_value());
  EXPECT_TRUE(pipeline->destroy().has_value());
  EXPECT_TRUE(render_pass->destroy().has_value());
  EXPECT_TRUE(target->destroy().has_value());
  EXPECT_TRUE(image->destroy().has_value());
}

/// The example's band loop at output heights below, equal to and above the band count: a
/// band that owns no output rows records no draw (a zero render area / scissor is invalid
/// Vulkan), and the non-empty bands still cover every output row with their own content.
/// \c Case::band is unused here.
class SubFrameBandPartitionTest : public SubFrameBandSamplingTest {};

TEST_P(SubFrameBandPartitionTest, EmptyBandsAreSkippedAndTheRestCoverEveryRow) {
  const Case c = GetParam();
  const Extent2D output{kSourceWidth, c.output_height};

  auto image = device_.create_image({
      .extent = {kSourceWidth, kSourceHeight},
      .format = Format::kR8G8B8A8Unorm,
      .usage = ImageUsage::kSampled,
      .share = ShareMode::kExported,
  });
  ASSERT_TRUE(image.has_value()) << image.error().message;
  auto level0 = image->cuda_mipmapped_array_get_level(0);
  ASSERT_TRUE(level0.has_value()) << level0.error().message;
  ASSERT_EQ(cuMemAlloc(&band_buffer_, static_cast<size_t>(kSourceWidth) * 4 * kSourceHeight),
            CUDA_SUCCESS);
  for (uint32_t band = 0; band < c.bands; ++band) {
    upload_band(*level0, 0, c.bands, band);
  }
  ASSERT_EQ(cuStreamSynchronize(cuda_stream_), CUDA_SUCCESS);

  auto target = device_.create_image({
      .extent = output,
      .format = Format::kR8G8B8A8Unorm,
      .usage = ImageUsage::kRenderTarget,
  });
  ASSERT_TRUE(target.has_value()) << target.error().message;
  auto render_pass =
      device_.create_render_pass({.render_targets = {{.format = Format::kR8G8B8A8Unorm}}});
  ASSERT_TRUE(render_pass.has_value()) << render_pass.error().message;
  ASSERT_TRUE(render_pass->set_render_image(*target, 0, {0.0F, 0.0F, 0.0F, 1.0F}).has_value());
  auto pipeline = device_.create_pipeline({
      .shader_blob = shader::make_shader_blob(),
      .topology = PrimitiveTopology::kTriangleList,
      .raster = RasterState{.cull = CullMode::kNone},
      .depth = DepthState{.test = false, .write = false, .compare = CompareOp::kAlways},
  });
  ASSERT_TRUE(pipeline.has_value()) << pipeline.error().message;
  auto descriptor = pipeline->create_descriptor();
  ASSERT_TRUE(descriptor.has_value()) << descriptor.error().message;
  ASSERT_TRUE(descriptor->bind_image(*image, "frameTexture").has_value());
  ASSERT_TRUE(descriptor->bind_sampler(device_, {}, "frameSampler").has_value());

  // The example's per-band decision, over every band of one frame. (An offscreen target
  // cannot take the example's per-band kLoad pass, so all bands draw inside one pass; the
  // decision under test — record nothing for an empty band — is the same.)
  const Rect2D full_area{0, 0, output.width, output.height};
  auto command_list = device_.create_command_list({.type = CommandListType::kTransient});
  ASSERT_TRUE(command_list.has_value()) << command_list.error().message;
  ASSERT_TRUE(command_list->begin().has_value());
  ASSERT_TRUE(command_list->begin_render(*render_pass, full_area).has_value());
  ASSERT_TRUE(command_list->set_viewport(full_area).has_value());
  ASSERT_TRUE(command_list->bind_pipeline(*pipeline).has_value());
  ASSERT_TRUE(command_list->bind_descriptor(*descriptor).has_value());
  uint32_t skipped = 0;
  std::vector<uint32_t> owner(output.height, UINT32_MAX);
  for (uint32_t band = 0; band < c.bands; ++band) {
    const uint32_t y0 = band_offset(band, c.bands, kSourceHeight);
    const uint32_t band_height = band_offset(band + 1, c.bands, kSourceHeight) - y0;
    const uint32_t window_y0 = band_offset(band, c.bands, output.height);
    const uint32_t window_height = band_offset(band + 1, c.bands, output.height) - window_y0;
    if (window_height == 0) {
      ++skipped;
      continue;
    }
    for (uint32_t row = window_y0; row < window_y0 + window_height; ++row) {
      owner[row] = band;
    }
    const Rect2D band_rect{0, static_cast<int32_t>(window_y0), output.width, window_height};
    const shader::PushConstants push_constants{
        .uv_min_y = (static_cast<float>(y0) + 0.5F) / static_cast<float>(kSourceHeight),
        .uv_max_y =
            (static_cast<float>(y0 + band_height) - 0.5F) / static_cast<float>(kSourceHeight),
    };
    ASSERT_TRUE(command_list->set_scissor(band_rect).has_value());
    ASSERT_TRUE(command_list
                    ->set_push_data(*pipeline,
                                    std::span{reinterpret_cast<const std::byte*>(&push_constants),
                                              sizeof(push_constants)})
                    .has_value());
    ASSERT_TRUE(command_list->draw(3).has_value());
  }
  ASSERT_TRUE(command_list->end_render().has_value());
  ASSERT_TRUE(command_list->end().has_value());

  // Integer partitioning: with fewer rows than bands exactly (bands - rows) bands are empty
  // (every other band owns one row); with at least as many rows none is.
  const uint32_t expected_skipped = c.bands > output.height ? c.bands - output.height : 0;
  EXPECT_EQ(skipped, expected_skipped);
  if (c.bands > output.height) {
    EXPECT_EQ(band_offset(1, c.bands, output.height), 0U) << "the first band should be empty";
  }

  const auto submitted = device_.submit({.command_lists = {&*command_list, 1}});
  ASSERT_TRUE(submitted.has_value()) << submitted.error().message;
  ASSERT_TRUE(device_.wait_on_submit(*submitted, std::chrono::seconds(10)).has_value());

  const size_t row_pitch = static_cast<size_t>(output.width) * 4;
  std::vector<uint8_t> pixels(row_pitch * output.height);
  ASSERT_TRUE(
      device_.download_from_image(*target, 0, std::as_writable_bytes(std::span(pixels)), row_pitch)
          .has_value());

  // The check below can only tell bands apart if their expected colours are distinct under the
  // +-1 per-channel tolerance: assert that for every non-empty band before relying on it.
  {
    std::set<uint32_t> colors;
    uint32_t nonempty = 0;
    for (uint32_t band = 0; band < c.bands; ++band) {
      if (band_offset(band + 1, c.bands, output.height) ==
          band_offset(band, c.bands, output.height)) {
        continue;
      }
      ++nonempty;
      const Rgba color = band_color(0, band);
      // Codes step by 4 per channel, so equality of the packed value is the collision test.
      colors.insert(color.packed());
    }
    ASSERT_EQ(colors.size(), nonempty) << "expected band colours collide";
  }

  // Every output row was drawn by the band that owns it: no clear-colour gaps where an empty
  // band was skipped, and no band spilled over its rows.
  for (uint32_t row = 0; row < output.height; ++row) {
    ASSERT_NE(owner[row], UINT32_MAX) << "row " << row << " belongs to no band";
    const Rgba expected = band_color(0, owner[row]);
    const uint8_t* px = pixels.data() + static_cast<size_t>(row) * row_pitch;
    const bool ok = std::abs(int{px[0]} - int{expected.r}) <= 1 &&
                    std::abs(int{px[1]} - int{expected.g}) <= 1 &&
                    std::abs(int{px[2]} - int{expected.b}) <= 1;
    ASSERT_TRUE(ok) << "row " << row << " (band " << owner[row] << "): got (" << int{px[0]} << ", "
                    << int{px[1]} << ", " << int{px[2]} << ") expected (" << int{expected.r} << ", "
                    << int{expected.g} << ", " << int{expected.b} << ")";
  }

  EXPECT_TRUE(device_.wait_idle().has_value());
  EXPECT_TRUE(command_list->destroy().has_value());
  EXPECT_TRUE(descriptor->destroy().has_value());
  EXPECT_TRUE(pipeline->destroy().has_value());
  EXPECT_TRUE(render_pass->destroy().has_value());
  EXPECT_TRUE(target->destroy().has_value());
  EXPECT_TRUE(image->destroy().has_value());
}

// Output rows below / equal to / above the band count, plus the reviewer's --bands 600 window
// shrunk to 300 rows (band 0 empty, the case the example's first-use clear must survive).
INSTANTIATE_TEST_SUITE_P(Partition, SubFrameBandPartitionTest,
                         ::testing::Values(Case{5, 8, 0}, Case{8, 8, 0}, Case{13, 8, 0},
                                           Case{300, 600, 0}),
                         [](const ::testing::TestParamInfo<Case>& info) {
                           return "out" + std::to_string(info.param.output_height) + "_bands" +
                                  std::to_string(info.param.bands);
                         });

// 600 source rows onto 1080 and 700 output rows: nonintegral scales, so band edges fall
// between source texels. First, middle and last bands cover both the upper (uv_min_y) and
// lower (uv_max_y) clamp edges.
INSTANTIATE_TEST_SUITE_P(Scales, SubFrameBandSamplingTest,
                         ::testing::Values(Case{1080, 8, 0}, Case{1080, 8, 3}, Case{1080, 8, 7},
                                           Case{700, 7, 0}, Case{700, 7, 6}),
                         [](const ::testing::TestParamInfo<Case>& info) {
                           return "out" + std::to_string(info.param.output_height) + "_bands" +
                                  std::to_string(info.param.bands) + "_band" +
                                  std::to_string(info.param.band);
                         });

}  // namespace
}  // namespace holoviz
