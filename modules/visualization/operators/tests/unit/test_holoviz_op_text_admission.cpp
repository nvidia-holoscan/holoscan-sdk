// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// HolovizOp text-layer record admission, exercised through the production operator path: real
// holoscan::Tensor samples of fixed-width text records are bound to the layer ports with the
// private synthetic fixture and HolovizOp::compute() runs build_render_list() against a live
// window/swapchain. The oracle is the render backend's glyph high-water mark
// (AllocationStats::text_glyph_high_water, reached through the operator-private impl header):
// an admitted sample raises it to the sample's byte count, a rejected sample leaves it untouched
// while compute() still succeeds (per-layer soft drop, never a frame failure). The decoder's own
// rules are covered headless in test_text_input.cpp; the glyph rendering in
// test_holoviz_backend_text.cpp.
//
// Needs a window system and a presentable adapter (HolovizOp::start() supports WindowSinkDesc
// only); the whole suite skips otherwise, like the windowed WSI present backend tests.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include <holoscan/core/errors.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/visualization/holoviz_op.hpp>
#include <holoviz/holoviz.hpp>

#include <gtest/gtest.h>

#include "holoviz_op/backends/window_surface.hpp"
#include "holoviz_op/holoviz_op_impl.hpp"
#include "holoviz_test_skip.hpp"
#include "operator_test_fixture.hpp"

namespace holoscan::visualization {
namespace {

constexpr std::uint32_t kMaxDim = 32;  // declared max_width/max_height of the peer image layer
constexpr std::uint32_t kMaxGlyphs = 24;
constexpr std::size_t kTextBytes = 16;  // L
constexpr std::int64_t kRowBytes = static_cast<std::int64_t>(kTextRunHeaderBytes + kTextBytes);
constexpr DLDataType kU8{kDLUInt, 8U, 1U};

/// One producer-side record: header + NUL-padded text.
struct Record {
  TextRunHeader header{};
  std::string text;
};

class HolovizOpTextAdmissionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Same environment probe as the windowed WSI present backend tests: no window system or no
    // presentable adapter is a skip, not a failure. HolovizOp::start() would throw instead.
    auto window = holoviz::create_window(
        {.title = "holoviz op text admission probe", .size = {64, 64}, .resizable = false});
    if (!window) {
      HOLOVIZ_ENV_SKIP() << "no window system available: " << window.error().message;
    }
    auto surface = backends::make_window_surface_handle(*window);
    ASSERT_TRUE(surface.has_value()) << surface.error().message;
    auto device = holoviz::create_device_for_surface(
        {}, *std::static_pointer_cast<holoviz::Surface>(surface->impl));
    if (!device) {
      surface->impl.reset();
      (void)window->destroy();
      HOLOVIZ_ENV_SKIP() << "no presentable adapter: " << device.error().message;
    }
    ASSERT_TRUE(device->destroy(std::chrono::seconds(5)).has_value());
    surface->impl.reset();
    ASSERT_TRUE(window->destroy().has_value());

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
      HOLOVIZ_ENV_SKIP() << "no CUDA device";
    }
    device_count_ = device_count;

    // Layer 0 = text under test, layer 1 = RGBA8 peer image whose extent proves a frame was
    // built. The peer is the trigger so the text port is a freshness-sampled side layer, as in
    // the common video-plus-labels composition.
    LayerStack stack;
    stack.add_text_layer("under_test", TextLayerDesc{.max_glyphs = kMaxGlyphs});
    stack.add_image_layer("peer", ImageLayerDesc{ImageFormat::kR8G8B8A8Unorm, kMaxDim, kMaxDim});
    op_ = std::make_unique<HolovizOp>(
        std::move(stack),
        WindowSinkDesc{.title = "holoviz op text admission", .width = 160, .height = 120},
        LayerTrigger{1});
    fixture_ = std::make_unique<holoscan::test::SyntheticOperatorFixture<HolovizOp>>(*op_);
    ASSERT_TRUE(fixture_->declare_ports().has_value());
    try {
      op_->start();
    } catch (const holoscan::RuntimeError& error) {
      FAIL() << "HolovizOp::start() failed: " << error.what();
    }
  }

  void TearDown() override {
    // The fixture's binding references the operator's ports: retire it before the operator.
    if (op_) {
      op_->stop();
    }
    fixture_.reset();
    op_.reset();
    if (stream_ != nullptr) {
      (void)cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
  }

  /// Pinned host Tensor (DLPack kDLCUDAHost) of the given shape.
  static Tensor make_host_tensor(DLDataType dtype, const std::vector<std::int64_t>& shape) {
    static const auto allocator = std::make_shared<UnboundedAllocator>();
    auto tensor = make_tensor(shape, dtype, std::nullopt, MemoryStorageType::kHost, allocator);
    if (!tensor) {
      ADD_FAILURE() << "make_tensor failed: " << tensor.error().message;
      return Tensor{};
    }
    return std::move(*tensor);
  }

  /// A {records, kRowBytes} uint8 sample with the records packed as TextLayerDesc specifies.
  static Tensor make_records(const std::vector<Record>& records,
                             std::int64_t row_bytes = kRowBytes) {
    Tensor tensor = make_host_tensor(kU8, {static_cast<std::int64_t>(records.size()), row_bytes});
    // The test owns this freshly allocated storage; Tensor only hands out a read view.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto* bytes = static_cast<std::uint8_t*>(const_cast<void*>(tensor.data()));
    if (bytes == nullptr) {
      ADD_FAILURE() << "records tensor has no storage";
      return tensor;
    }
    std::memset(bytes, 0, records.size() * static_cast<std::size_t>(row_bytes));
    for (std::size_t i = 0; i < records.size(); ++i) {
      auto* row = bytes + i * static_cast<std::size_t>(row_bytes);
      const auto header_bytes = std::min<std::size_t>(sizeof(TextRunHeader), row_bytes);
      std::memcpy(row, &records[i].header, header_bytes);
      if (static_cast<std::size_t>(row_bytes) > kTextRunHeaderBytes) {
        const auto n = std::min(records[i].text.size(),
                                static_cast<std::size_t>(row_bytes) - kTextRunHeaderBytes);
        // TextRun stores length-delimited bytes; the payload is not a C string.
        // NOLINTNEXTLINE(bugprone-not-null-terminated-result)
        std::memcpy(row + kTextRunHeaderBytes, records[i].text.data(), n);
      }
    }
    return tensor;
  }

  /// A valid RGBA8 peer sample (a wire-correct schema::ImageT around a host tensor) whose extent
  /// differs on every call, so the peer cache proves that this activation's frame really was
  /// built (acquire()/begin_frame() may transiently skip a frame with kBackpressured, which
  /// compute() reports as success).
  schema::ImageT next_peer_sample(std::int64_t& height, std::int64_t& width) {
    ++peer_sequence_;
    height = 4 + static_cast<std::int64_t>(peer_sequence_ % 8U);
    width = 8 + static_cast<std::int64_t>(peer_sequence_ % 16U);
    schema::ImageT image;
    image.encoding = schema::ImageEncoding_RGBA8;
    image.color_space = schema::ColorSpace_UNSPECIFIED;
    image.height = static_cast<std::int32_t>(height);
    image.width = static_cast<std::int32_t>(width);
    image.data = std::make_shared<Tensor>(make_host_tensor(kU8, {height, width, 4}));
    return image;
  }

  /// Bind `sample` to the text layer and a fresh peer to the image layer, then run compute()
  /// until the frame with this peer extent has been built. Ports are pooled by kind: the text
  /// layer (declared first) is tensor slot 0 and the image layer is image slot 0.
  void activate(const Tensor& sample) {
    std::int64_t peer_height = 0;
    std::int64_t peer_width = 0;
    const schema::ImageT peer = next_peer_sample(peer_height, peer_width);

    for (int attempt = 0; attempt < 200; ++attempt) {
      ASSERT_TRUE(fixture_->bind_value(HolovizOp::Impl::of(*op_).tensor_layer_input(0), sample)
                      .has_value());
      ASSERT_TRUE(
          fixture_->bind_value(HolovizOp::Impl::of(*op_).image_layer_input(0), peer).has_value());
      auto result = fixture_->invoke_once(
          holoscan::detail::SyntheticActivationOptions{.cuda_stream = activation_stream()});
      if (!result && result.error().code == holoscan::ErrorCode::kIncompatible &&
          result.error().message.find("cross-device") != std::string::npos &&
          select_next_device()) {
        continue;  // multi-GPU host: HoloViz picked a different device than the current one
      }
      ASSERT_TRUE(result.has_value()) << result.error().message;
      const HolovizOp::Impl::ImageLayerCache& peer_cache =
          HolovizOp::Impl::of(*op_).image_layer_cache(1);
      if (peer_cache.height == static_cast<std::uint32_t>(peer_height) &&
          peer_cache.width == static_cast<std::uint32_t>(peer_width)) {
        return;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    FAIL() << "no frame was built for peer extent " << peer_width << "x" << peer_height;
  }

  /// The render backend's text glyph high-water mark: raised by record() for every admitted run.
  std::size_t glyph_high_water() {
    const auto stats = HolovizOp::Impl::of(*op_).render_->stats();
    if (!stats) {
      ADD_FAILURE() << "render backend stats unavailable: " << stats.error().message;
      return 0;
    }
    return stats->text_glyph_high_water;
  }

 private:
  cudaStream_t activation_stream() {
    if (stream_ == nullptr) {
      if (cudaSetDevice(device_ordinal_) != cudaSuccess ||
          cudaStreamCreate(&stream_) != cudaSuccess) {
        ADD_FAILURE() << "cannot create a CUDA stream on device " << device_ordinal_;
        stream_ = nullptr;
      }
    }
    return stream_;
  }

  bool select_next_device() {
    if (device_ordinal_ + 1 >= device_count_) {
      return false;
    }
    ++device_ordinal_;
    if (stream_ != nullptr) {
      (void)cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
    return true;
  }

  std::unique_ptr<HolovizOp> op_;
  std::unique_ptr<holoscan::test::SyntheticOperatorFixture<HolovizOp>> fixture_;
  cudaStream_t stream_ = nullptr;
  int device_count_ = 0;
  int device_ordinal_ = 0;
  std::uint32_t peer_sequence_ = 0;
};

TEST_F(HolovizOpTextAdmissionTest, AdmittedRecordsReachTheRendererAsGlyphs) {
  ASSERT_EQ(glyph_high_water(), 0U);

  // Two runs, 5 + 6 bytes; a blank third row is not a run.
  ASSERT_NO_FATAL_FAILURE(activate(make_records({
      {{.x = 0.1f, .y = 0.1f, .size_px = 20.0f, .color = 0xFF00D4FFu}, "hello"},
      {{.x = 0.1f, .y = 0.5f, .size_px = 12.0f}, "world!"},
      {{}, ""},
  })));
  EXPECT_EQ(glyph_high_water(), 11U);

  // A larger sample within the budget raises the mark to its own byte count.
  ASSERT_NO_FATAL_FAILURE(activate(make_records({{{}, "0123456789abcdef"}, {{}, "01234567"}})));
  EXPECT_EQ(glyph_high_water(), 24U);
}

// A rejected sample drops only the text layer for that activation: compute() succeeds, the peer
// image layer still renders (the frame was built), and the mark is untouched. The next valid
// sample resumes the layer.
TEST_F(HolovizOpTextAdmissionTest, RejectedRecordsDropOnlyThisActivationOfTheLayer) {
  ASSERT_NO_FATAL_FAILURE(activate(make_records({{{}, "abc"}})));
  ASSERT_EQ(glyph_high_water(), 3U);

  // Over the declared max_glyphs (25 bytes > 24).
  ASSERT_NO_FATAL_FAILURE(activate(make_records({{{}, "0123456789abcdef"}, {{}, "012345678"}})));
  EXPECT_EQ(glyph_high_water(), 3U);
  // A non-positive glyph size.
  ASSERT_NO_FATAL_FAILURE(activate(make_records({{{.size_px = 0.0f}, "0123456789abcdef"}})));
  EXPECT_EQ(glyph_high_water(), 3U);
  // Rows shorter than the 16-byte header (still a host uint8 rank-2 tensor, so the port's
  // typed contract admits it; the decoder does not).
  ASSERT_NO_FATAL_FAILURE(activate(make_records({{{}, "0123456789abcdef"}}, 8)));
  EXPECT_EQ(glyph_high_water(), 3U);
  // A sample of blank rows draws nothing and is not an error.
  ASSERT_NO_FATAL_FAILURE(activate(make_records({{{}, ""}, {{}, ""}})));
  EXPECT_EQ(glyph_high_water(), 3U);

  ASSERT_NO_FATAL_FAILURE(activate(make_records({{{}, "0123456789"}})));
  EXPECT_EQ(glyph_high_water(), 10U);
}

}  // namespace
}  // namespace holoscan::visualization
