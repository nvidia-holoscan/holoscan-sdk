// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// HolovizOp image-layer Tensor admission, exercised through the production operator path: real
// holoscan::Tensor samples are bound to the layer ports with the private synthetic fixture and
// HolovizOp::compute() runs build_render_list() against a live window/swapchain. The oracle is
// HolovizOp::Impl::image_layer_cache() (reached through the operator-private impl header): a
// host-resident sample reaches the cache only after the dtype / rank / channel-count check admits
// it, so an accepted sample changes the cached extent and a rejected sample leaves it untouched
// while compute() still succeeds (per-layer soft drop, never a frame failure). The backend round
// trips in test_holoviz_backend_image.cpp cover the upload itself.
//
// Needs a window system and a presentable adapter (HolovizOp::start() supports WindowSinkDesc
// only); the whole suite skips otherwise, like the windowed WSI present backend tests.

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include <holoscan/core/errors.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/visualization/holoviz_op.hpp>
#include <holoviz/holoviz.hpp>

#include <gtest/gtest.h>

#include "holoviz_op/backends/wsi_present_backend.hpp"
#include "holoviz_op/holoviz_op_impl.hpp"
#include "holoviz_test_skip.hpp"
#include "operator_test_fixture.hpp"

namespace holoscan::visualization {
namespace {

constexpr std::uint32_t kMaxDim = 32;  // declared max_width/max_height of every test layer
constexpr std::int64_t kHeight = 8;
constexpr std::int64_t kWidth = 16;

constexpr DLDataType kU8{kDLUInt, 8U, 1U};
constexpr DLDataType kU16{kDLUInt, 16U, 1U};
constexpr DLDataType kU16x2{kDLUInt, 16U, 2U};
constexpr DLDataType kS16{kDLInt, 16U, 1U};
constexpr DLDataType kF32{kDLFloat, 32U, 1U};

struct AdmissionCase {
  std::string_view name;
  ImageFormat format;
  DLDataType dtype;
  std::vector<std::int64_t> shape;  // {H, W[, C]} or a deliberately wrong rank
  bool accepted;
};

// gtest prints the case name on failure instead of the raw bytes of the parameter.
void PrintTo(const AdmissionCase& c, std::ostream* os) {
  *os << c.name;
}

/// Accepted-shape table from <holoscan/visualization/holoviz_op.hpp> (ImageLayerDesc) plus
/// representative rejections. Every entry runs through HolovizOp::compute(), not the backend
/// directly.
const std::vector<AdmissionCase>& admission_cases() {
  static const std::vector<AdmissionCase> cases{
      // kR16Unorm: natural uint16 rank 2 / rank 3 single channel, and the packed byte alias.
      {"R16_u16_HW", ImageFormat::kR16Unorm, kU16, {kHeight, kWidth}, true},
      {"R16_u16_HW1", ImageFormat::kR16Unorm, kU16, {kHeight, kWidth, 1}, true},
      {"R16_u8_HW2_packed", ImageFormat::kR16Unorm, kU8, {kHeight, kWidth, 2}, true},
      // kR16G16B16A16Unorm: natural uint16 HW4 and the packed byte alias.
      {"RGBA16_u16_HW4", ImageFormat::kR16G16B16A16Unorm, kU16, {kHeight, kWidth, 4}, true},
      {"RGBA16_u8_HW8_packed", ImageFormat::kR16G16B16A16Unorm, kU8, {kHeight, kWidth, 8}, true},
      // Existing 8-bit users must keep working under the new guard.
      {"R8_u8_HW", ImageFormat::kR8Unorm, kU8, {kHeight, kWidth}, true},
      {"R8_u8_HW1", ImageFormat::kR8Unorm, kU8, {kHeight, kWidth, 1}, true},
      {"RGBA8_u8_HW4", ImageFormat::kR8G8B8A8Unorm, kU8, {kHeight, kWidth, 4}, true},
      {"BGRA8_u8_HW4", ImageFormat::kB8G8R8A8Unorm, kU8, {kHeight, kWidth, 4}, true},
      // Rejected: signed / float element types.
      {"R16_s16_HW_rejected", ImageFormat::kR16Unorm, kS16, {kHeight, kWidth}, false},
      {"R16_f32_HW_rejected", ImageFormat::kR16Unorm, kF32, {kHeight, kWidth}, false},
      {"RGBA16_f32_HW4_rejected",
       ImageFormat::kR16G16B16A16Unorm,
       kF32,
       {kHeight, kWidth, 4},
       false},
      // Rejected: vectorized element (lanes != 1).
      {"R16_u16x2_HW_rejected", ImageFormat::kR16Unorm, kU16x2, {kHeight, kWidth}, false},
      // Rejected: wrong rank.
      {"R16_u16_rank4_rejected", ImageFormat::kR16Unorm, kU16, {1, kHeight, kWidth, 1}, false},
      {"R16_u16_rank1_rejected", ImageFormat::kR16Unorm, kU16, {kWidth}, false},
      // Rejected: wrong channel count for the natural type or the packed alias.
      {"RGBA16_u16_HW3_rejected",
       ImageFormat::kR16G16B16A16Unorm,
       kU16,
       {kHeight, kWidth, 3},
       false},
      {"R16_u16_HW2_rejected", ImageFormat::kR16Unorm, kU16, {kHeight, kWidth, 2}, false},
      {"R16_u8_HW4_rejected", ImageFormat::kR16Unorm, kU8, {kHeight, kWidth, 4}, false},
      {"RGBA8_u8_HW2_rejected", ImageFormat::kR8G8B8A8Unorm, kU8, {kHeight, kWidth, 2}, false},
      // Rejected: the original silent-garble case (uint16 on an 8-bit layer).
      {"R8_u16_HW_rejected", ImageFormat::kR8Unorm, kU16, {kHeight, kWidth}, false},
  };
  return cases;
}

std::string case_name(const ::testing::TestParamInfo<AdmissionCase>& info) {
  return std::string{info.param.name};
}

class HolovizOpImageAdmissionFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    // Same environment probe as the windowed WSI present backend tests: no window system or no
    // presentable adapter is a skip, not a failure. HolovizOp::start() would throw instead.
    auto window = holoviz::create_window(
        {.title = "holoviz op admission probe", .size = {64, 64}, .resizable = false});
    if (!window) {
      HOLOVIZ_ENV_SKIP() << "no window system available: " << window.error().message;
    }
    auto surface = backends::make_surface_handle(*window);
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

  /// Build a two-image-layer HolovizOp (layer 0 = format under test, layer 1 = RGBA8 peer),
  /// declare its ports through the synthetic fixture and bring the window/backends up.
  void make_operator(ImageFormat format) {
    LayerStack stack;
    stack.add_image_layer("under_test", ImageLayerDesc{format, kMaxDim, kMaxDim});
    stack.add_image_layer("peer", ImageLayerDesc{ImageFormat::kR8G8B8A8Unorm, kMaxDim, kMaxDim});
    op_ = std::make_unique<HolovizOp>(
        std::move(stack),
        WindowSinkDesc{.title = "holoviz op admission", .width = 160, .height = 120});
    fixture_ = std::make_unique<holoscan::test::SyntheticOperatorFixture<HolovizOp>>(*op_);
    ASSERT_TRUE(fixture_->declare_ports().has_value());
    try {
      op_->start();
    } catch (const holoscan::RuntimeError& error) {
      FAIL() << "HolovizOp::start() failed: " << error.what();
    }
  }

  /// Host-resident Tensor of the requested dtype and shape (contents irrelevant to admission).
  static Tensor make_host_tensor(DLDataType dtype, const std::vector<std::int64_t>& shape) {
    static const auto allocator = std::make_shared<UnboundedAllocator>();
    auto tensor = make_tensor(shape, dtype, std::nullopt, MemoryStorageType::kHost, allocator);
    if (!tensor) {
      ADD_FAILURE() << "make_tensor failed: " << tensor.error().message;
      return Tensor{};
    }
    return std::move(*tensor);
  }

  /// A valid RGBA8 peer sample whose extent differs on every call, so the peer cache proves that
  /// this activation's frame really was built (acquire()/begin_frame() may transiently skip a
  /// frame with kBackpressured, which compute() reports as success).
  Tensor next_peer_tensor(std::int64_t& height, std::int64_t& width) {
    ++peer_sequence_;
    height = 4 + static_cast<std::int64_t>(peer_sequence_ % 8U);
    width = 8 + static_cast<std::int64_t>(peer_sequence_ % 16U);
    return make_host_tensor(kU8, {height, width, 4});
  }

  /// Bind `sample` to layer 0 and a fresh peer to layer 1, then run compute() until the frame
  /// with this peer extent has been built. Fails if no frame is built within the retry budget.
  void activate(const Tensor& sample) {
    std::int64_t peer_height = 0;
    std::int64_t peer_width = 0;
    const Tensor peer = next_peer_tensor(peer_height, peer_width);

    for (int attempt = 0; attempt < 200; ++attempt) {
      ASSERT_TRUE(fixture_->bind_value(op_->layer_input(0), sample).has_value());
      ASSERT_TRUE(fixture_->bind_value(op_->layer_input(1), peer).has_value());
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

  const HolovizOp::Impl::ImageLayerCache& cache_under_test() const {
    return HolovizOp::Impl::of(*op_).image_layer_cache(0);
  }
  const HolovizOp::Impl::ImageLayerCache& peer_cache() const {
    return HolovizOp::Impl::of(*op_).image_layer_cache(1);
  }

 private:
  /// Stream on the currently selected CUDA device; HolovizOp::compute() derives the activation
  /// device from it and rejects a device other than HoloViz's.
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

class HolovizOpImageAdmissionTest : public HolovizOpImageAdmissionFixture,
                                    public ::testing::WithParamInterface<AdmissionCase> {};

TEST_P(HolovizOpImageAdmissionTest, TensorIsAdmittedOnlyWhenItMatchesTheDeclaredFormat) {
  const AdmissionCase& c = GetParam();
  ASSERT_NO_FATAL_FAILURE(make_operator(c.format));
  ASSERT_TRUE(cache_under_test().bytes.empty());

  const Tensor sample = make_host_tensor(c.dtype, c.shape);
  ASSERT_NE(sample.data(), nullptr);
  ASSERT_NO_FATAL_FAILURE(activate(sample));

  const HolovizOp::Impl::ImageLayerCache& cache = cache_under_test();
  if (c.accepted) {
    EXPECT_FALSE(cache.bytes.empty()) << "admitted sample never reached the layer";
    EXPECT_EQ(cache.height, static_cast<std::uint32_t>(c.shape[0]));
    EXPECT_EQ(cache.width, static_cast<std::uint32_t>(c.shape[1]));
  } else {
    EXPECT_TRUE(cache.bytes.empty()) << "mismatching sample was admitted";
    EXPECT_EQ(cache.width, 0U);
    EXPECT_EQ(cache.height, 0U);
  }
}

INSTANTIATE_TEST_SUITE_P(Formats, HolovizOpImageAdmissionTest,
                         ::testing::ValuesIn(admission_cases()), case_name);

// A rejected sample drops only its own layer for that activation: the peer layer still renders,
// the next valid sample resumes the layer, a later rejection keeps the last admitted extent
// (the layer is omitted, not failed), and the packed byte alias resumes it again.
TEST_F(HolovizOpImageAdmissionFixture, RejectedSampleDoesNotPoisonTheLayerOrItsPeers) {
  ASSERT_NO_FATAL_FAILURE(make_operator(ImageFormat::kR16Unorm));

  ASSERT_NO_FATAL_FAILURE(activate(make_host_tensor(kS16, {kHeight, kWidth})));
  EXPECT_TRUE(cache_under_test().bytes.empty());
  EXPECT_FALSE(peer_cache().bytes.empty()) << "peer layer must render alongside a rejection";

  ASSERT_NO_FATAL_FAILURE(activate(make_host_tensor(kU16, {kHeight, kWidth})));
  EXPECT_EQ(cache_under_test().height, static_cast<std::uint32_t>(kHeight));
  EXPECT_EQ(cache_under_test().width, static_cast<std::uint32_t>(kWidth));

  ASSERT_NO_FATAL_FAILURE(activate(make_host_tensor(kF32, {kHeight + 4, kWidth + 4})));
  EXPECT_EQ(cache_under_test().height, static_cast<std::uint32_t>(kHeight))
      << "rejected sample must not replace the last admitted extent";
  EXPECT_EQ(cache_under_test().width, static_cast<std::uint32_t>(kWidth));

  ASSERT_NO_FATAL_FAILURE(activate(make_host_tensor(kU8, {kHeight + 4, kWidth + 4, 2})));
  EXPECT_EQ(cache_under_test().height, static_cast<std::uint32_t>(kHeight + 4));
  EXPECT_EQ(cache_under_test().width, static_cast<std::uint32_t>(kWidth + 4));
}

}  // namespace
}  // namespace holoscan::visualization
