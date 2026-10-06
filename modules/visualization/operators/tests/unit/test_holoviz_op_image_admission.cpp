// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// HolovizOp image-layer schema::ImageT admission, exercised through the production operator
// path: real schema::ImageT samples (wrapping a holoscan::Tensor) are bound to the image-kind
// layer ports with the private synthetic fixture and HolovizOp::compute() runs
// build_render_list() against a live window/swapchain. The oracle is
// HolovizOp::Impl::image_layer_cache() (reached through the operator-private impl header): a
// host-resident sample reaches the cache only after the wire encoding/color_space check against
// the declared ImageLayerDesc (Decision 3 in the ImageT port alignment design doc) and the
// dtype / rank / channel-count check both admit it, so an accepted sample changes the cached
// extent and a rejected sample leaves it untouched while compute() still succeeds (per-layer
// soft drop, never a frame failure). The backend round trips in test_holoviz_backend_image.cpp
// cover the upload itself.
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
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/visualization/holoviz_op.hpp>
#include <holoviz/holoviz.hpp>

#include <gtest/gtest.h>

#include "holoviz_op/backends/image_layout.hpp"
#include "holoviz_op/backends/window_surface.hpp"
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
  bool omit_plane_layouts = false;  // NV12 only: send the frame without its plane records
};

/// The two NV12 plane records for a Y plane at offset 0 followed by the interleaved UV plane,
/// both at \p pitch bytes per row; sized exactly like image_plane_span (the last row owns no
/// padding), which is also the storage the data-path cases allocate.
std::vector<schema::ImagePlaneLayout> nv12_plane_layouts(std::size_t width, std::size_t height,
                                                         std::size_t pitch) {
  const auto luma_bytes = static_cast<std::uint64_t>(height) * pitch;
  const auto chroma_bytes = static_cast<std::uint64_t>(height / 2 - 1) * pitch + width;
  return {schema::ImagePlaneLayout{0U, static_cast<std::uint32_t>(pitch), luma_bytes},
          schema::ImagePlaneLayout{luma_bytes, static_cast<std::uint32_t>(pitch), chroma_bytes}};
}

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
      {"RGB8_u8_HW3", ImageFormat::kR8G8B8Unorm, kU8, {kHeight, kWidth, 3}, true},
      {"NV12_u8_storage", ImageFormat::kNV12, kU8, {kHeight * 3 / 2, kWidth}, true},
      {"YUYV_u8_HW2", ImageFormat::kYUYV422, kU8, {kHeight, kWidth, 2}, true},
      {"RGB8_u8_HW4_rejected", ImageFormat::kR8G8B8Unorm, kU8, {kHeight, kWidth, 4}, false},
      {"NV12_odd_width_rejected", ImageFormat::kNV12, kU8, {kHeight * 3 / 2, kWidth - 1}, false},
      // NV12 geometry comes from ImageT::width/height and plane_layouts, not the Tensor shape, so
      // a frame without its plane records has no planes to render from.
      {"NV12_missing_plane_layouts_rejected",
       ImageFormat::kNV12,
       kU8,
       {kHeight * 3 / 2, kWidth},
       false,
       true},
      {"YUYV_odd_width_rejected", ImageFormat::kYUYV422, kU8, {kHeight, kWidth - 1, 2}, false},
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
  void make_operator(ImageFormat format, ColorSpace color_space = ColorSpace::kPassThrough) {
    LayerStack stack;
    ImageLayerDesc desc{format, kMaxDim, kMaxDim};
    desc.color_space = color_space;
    if (format == ImageFormat::kNV12 || format == ImageFormat::kYUYV422) {
      desc.yuv_model = YUVModelConversion::kBT709;
      desc.yuv_range = YUVRange::kLimited;
    }
    stack.add_image_layer("under_test", desc);
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

  /// Tensor of the requested residency, dtype, shape, and optional byte strides. kHost is CUDA
  /// pinned memory (DLPack kDLCUDAHost); kSystem is plain pageable memory (kDLCPU).
  static Tensor make_tensor_of(MemoryStorageType storage, DLDataType dtype,
                               const std::vector<std::int64_t>& shape,
                               const std::optional<TensorByteStrides>& strides = std::nullopt) {
    static const auto allocator = std::make_shared<UnboundedAllocator>();
    auto tensor = make_tensor(shape, dtype, strides, storage, allocator);
    if (!tensor) {
      ADD_FAILURE() << "make_tensor failed: " << tensor.error().message;
      return Tensor{};
    }
    return std::move(*tensor);
  }

  /// Pinned host Tensor (contents irrelevant to admission).
  static Tensor make_host_tensor(DLDataType dtype, const std::vector<std::int64_t>& shape) {
    return make_tensor_of(MemoryStorageType::kHost, dtype, shape);
  }

  /// Copy `bytes` into the tensor's storage on whichever side it lives.
  static void fill_tensor(Tensor& tensor, const std::vector<std::uint8_t>& bytes) {
    // The test owns this freshly allocated storage; Tensor only hands out a read view.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    auto* destination = const_cast<void*>(tensor.data());
    ASSERT_NE(destination, nullptr);
    const auto device_type = tensor.device().device_type;
    if (device_type == kDLCUDA || device_type == kDLCUDAManaged) {
      ASSERT_EQ(cudaMemcpy(destination, bytes.data(), bytes.size(), cudaMemcpyHostToDevice),
                cudaSuccess);
    } else {
      std::memcpy(destination, bytes.data(), bytes.size());
    }
  }

  /// The schema::ImageEncoding a declared ImageFormat maps to on the wire, matching
  /// image_format_for_encoding()'s inverse in holoviz_op_impl.cpp. Only formats this test suite
  /// exercises need an entry.
  static schema::ImageEncoding encoding_for_format(ImageFormat format) {
    switch (format) {
      case ImageFormat::kR8Unorm:
        return schema::ImageEncoding_MONO8;
      case ImageFormat::kR8G8B8A8Unorm:
        return schema::ImageEncoding_RGBA8;
      case ImageFormat::kB8G8R8A8Unorm:
        return schema::ImageEncoding_BGRA8;
      case ImageFormat::kR16Unorm:
        return schema::ImageEncoding_MONO16;
      case ImageFormat::kR8G8B8Unorm:
        return schema::ImageEncoding_RGB8;
      case ImageFormat::kNV12:
        return schema::ImageEncoding_NV12;
      case ImageFormat::kYUYV422:
        return schema::ImageEncoding_YUYV;
      case ImageFormat::kR16G16B16A16Unorm:
        return schema::ImageEncoding_RGBA16;
      default:
        ADD_FAILURE() << "encoding_for_format: no test mapping for this ImageFormat";
        return schema::ImageEncoding_CUSTOM;
    }
  }

  /// Wrap `tensor` in a wire-correct ImageT for `format` (the layer's declared format), unless
  /// `encoding_override`/`color_space_override`/`width_height_override` are given to
  /// deliberately construct a mismatched frame. By default `image.width`/`image.height` are
  /// derived from `tensor.shape()` so every case not explicitly testing the width/height check
  /// (added for the ImageT-geometry-vs-Tensor-extent bug) continues to exercise the check it was
  /// written for instead of tripping over an always-zero descriptor extent. `plane_layouts` is
  /// what an NV12 frame must carry (see nv12_plane_layouts()).
  static schema::ImageT make_image_sample(
      ImageFormat format, Tensor tensor,
      std::optional<schema::ImageEncoding> encoding_override = std::nullopt,
      std::optional<schema::ColorSpace> color_space_override = std::nullopt,
      std::optional<std::pair<std::int32_t, std::int32_t>> width_height_override = std::nullopt,
      std::vector<schema::ImagePlaneLayout> plane_layouts = {}) {
    schema::ImageT image;
    image.encoding = encoding_override.value_or(encoding_for_format(format));
    image.color_space = color_space_override.value_or(schema::ColorSpace_UNSPECIFIED);
    if (width_height_override) {
      image.width = width_height_override->first;
      image.height = width_height_override->second;
    } else {
      const auto shape = tensor.shape();
      if (shape.size() >= 2U) {
        image.height = static_cast<std::int32_t>(shape[0]);
        image.width = static_cast<std::int32_t>(shape[1]);
      }
    }
    image.plane_layouts = std::move(plane_layouts);
    image.data = std::make_shared<Tensor>(std::move(tensor));
    return image;
  }

  static constexpr std::pair<std::int32_t, std::int32_t> kDeclaredExtent{
      static_cast<std::int32_t>(kWidth), static_cast<std::int32_t>(kHeight)};

  /// A valid RGBA8 peer sample whose extent differs on every call, so the peer cache proves that
  /// this activation's frame really was built (acquire()/begin_frame() may transiently skip a
  /// frame with kBackpressured, which compute() reports as success).
  schema::ImageT next_peer_sample(std::int64_t& height, std::int64_t& width) {
    ++peer_sequence_;
    height = 4 + static_cast<std::int64_t>(peer_sequence_ % 8U);
    width = 8 + static_cast<std::int64_t>(peer_sequence_ % 16U);
    return make_image_sample(ImageFormat::kR8G8B8A8Unorm,
                             make_host_tensor(kU8, {height, width, 4}));
  }

  /// Bind `sample` to layer 0 and a fresh peer to layer 1, then run compute() until the frame
  /// with this peer extent has been built. Fails if no frame is built within the retry budget.
  void activate(const schema::ImageT& sample) {
    std::int64_t peer_height = 0;
    std::int64_t peer_width = 0;
    const schema::ImageT peer = next_peer_sample(peer_height, peer_width);

    for (int attempt = 0; attempt < 200; ++attempt) {
      ASSERT_TRUE(
          fixture_->bind_value(HolovizOp::Impl::of(*op_).image_layer_input(0), sample).has_value());
      ASSERT_TRUE(
          fixture_->bind_value(HolovizOp::Impl::of(*op_).image_layer_input(1), peer).has_value());
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
  /// Why layer 0's most recent sample was dropped (kAccepted after an admitted sample).
  ImageRejectReason reason_under_test() const {
    return HolovizOp::Impl::of(*op_).last_image_reject_reason_[0];
  }

  /// Stream on the currently selected CUDA device; HolovizOp::compute() derives the activation
  /// device from it and rejects a device other than HoloViz's. Protected (not private) so tests
  /// exercising a single-image-layer HolovizOp (no peer, so activate() doesn't fit) can drive
  /// their own retry loop with the same stream logic.
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

  Tensor tensor = make_host_tensor(c.dtype, c.shape);
  ASSERT_NE(tensor.data(), nullptr);
  // Every case declares the same logical extent; NV12's Tensor shape is a storage layout
  // (plane rows by row bytes), so it cannot be derived from the shape here, and NV12 frames
  // locate their planes through plane_layouts (packed here, at the case's own width).
  const bool nv12 = c.format == ImageFormat::kNV12;
  const auto width = static_cast<std::size_t>(c.shape.size() >= 2U ? c.shape[1] : kWidth);
  const schema::ImageT sample = make_image_sample(
      c.format,
      std::move(tensor),
      std::nullopt,
      std::nullopt,
      std::make_pair(static_cast<std::int32_t>(width), static_cast<std::int32_t>(kHeight)),
      nv12 && !c.omit_plane_layouts ? nv12_plane_layouts(width, kHeight, width)
                                    : std::vector<schema::ImagePlaneLayout>{});
  ASSERT_NO_FATAL_FAILURE(activate(sample));

  const HolovizOp::Impl::ImageLayerCache& cache = cache_under_test();
  if (c.accepted) {
    EXPECT_FALSE(cache.bytes.empty()) << "admitted sample never reached the layer";
    const auto visible_height = c.format == ImageFormat::kNV12 ? c.shape[0] / 3 * 2 : c.shape[0];
    EXPECT_EQ(cache.height, static_cast<std::uint32_t>(visible_height));
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

  ASSERT_NO_FATAL_FAILURE(activate(
      make_image_sample(ImageFormat::kR16Unorm, make_host_tensor(kS16, {kHeight, kWidth}))));
  EXPECT_TRUE(cache_under_test().bytes.empty());
  EXPECT_FALSE(peer_cache().bytes.empty()) << "peer layer must render alongside a rejection";

  ASSERT_NO_FATAL_FAILURE(activate(
      make_image_sample(ImageFormat::kR16Unorm, make_host_tensor(kU16, {kHeight, kWidth}))));
  EXPECT_EQ(cache_under_test().height, static_cast<std::uint32_t>(kHeight));
  EXPECT_EQ(cache_under_test().width, static_cast<std::uint32_t>(kWidth));

  ASSERT_NO_FATAL_FAILURE(activate(make_image_sample(
      ImageFormat::kR16Unorm, make_host_tensor(kF32, {kHeight + 4, kWidth + 4}))));
  EXPECT_EQ(cache_under_test().height, static_cast<std::uint32_t>(kHeight))
      << "rejected sample must not replace the last admitted extent";
  EXPECT_EQ(cache_under_test().width, static_cast<std::uint32_t>(kWidth));

  ASSERT_NO_FATAL_FAILURE(activate(make_image_sample(
      ImageFormat::kR16Unorm, make_host_tensor(kU8, {kHeight + 4, kWidth + 4, 2}))));
  EXPECT_EQ(cache_under_test().height, static_cast<std::uint32_t>(kHeight + 4));
  EXPECT_EQ(cache_under_test().width, static_cast<std::uint32_t>(kWidth + 4));
}

// ---- data path: padded / host / pinned / CUDA YUV samples with known pixel values -------------

enum class Residency { kHost, kPinned, kCuda };

struct DataPathCase {
  std::string_view name;
  ImageFormat format;
  Residency residency;
  bool padded;
};

void PrintTo(const DataPathCase& c, std::ostream* os) {
  *os << c.name;
}

const std::vector<DataPathCase>& data_path_cases() {
  static const std::vector<DataPathCase> cases{
      {"NV12_host_packed", ImageFormat::kNV12, Residency::kHost, false},
      {"NV12_host_padded", ImageFormat::kNV12, Residency::kHost, true},
      {"NV12_pinned_packed", ImageFormat::kNV12, Residency::kPinned, false},
      {"NV12_pinned_padded", ImageFormat::kNV12, Residency::kPinned, true},
      {"NV12_cuda_packed", ImageFormat::kNV12, Residency::kCuda, false},
      {"NV12_cuda_padded", ImageFormat::kNV12, Residency::kCuda, true},
      {"YUYV_host_packed", ImageFormat::kYUYV422, Residency::kHost, false},
      {"YUYV_host_padded", ImageFormat::kYUYV422, Residency::kHost, true},
      {"YUYV_pinned_packed", ImageFormat::kYUYV422, Residency::kPinned, false},
      {"YUYV_pinned_padded", ImageFormat::kYUYV422, Residency::kPinned, true},
      {"YUYV_cuda_packed", ImageFormat::kYUYV422, Residency::kCuda, false},
      {"YUYV_cuda_padded", ImageFormat::kYUYV422, Residency::kCuda, true},
  };
  return cases;
}

std::string data_path_case_name(const ::testing::TestParamInfo<DataPathCase>& info) {
  return std::string{info.param.name};
}

MemoryStorageType storage_type(Residency residency) {
  switch (residency) {
    case Residency::kHost:
      return MemoryStorageType::kSystem;
    case Residency::kPinned:
      return MemoryStorageType::kHost;
    case Residency::kCuda:
      return MemoryStorageType::kDevice;
  }
  return MemoryStorageType::kSystem;
}

constexpr std::size_t kPaddingBytes = 16;
constexpr std::uint8_t kPaddingFill = 0xEE;  // never produced by known_byte()

/// Deterministic, position-unique sample byte so a wrong pointer, pitch, or plane offset changes
/// the comparison. Stays clear of kPaddingFill.
constexpr std::uint8_t known_byte(std::size_t row, std::size_t col) {
  const auto value = static_cast<std::uint8_t>((row * 7 + col * 3 + 1) & 0xFF);
  return value == kPaddingFill ? std::uint8_t{0x11} : value;
}

/// Producer-side storage: `storage_rows` rows of `row_bytes` active bytes at `pitch`, padding
/// filled with kPaddingFill. Sized like image_plane_span (the last row owns no padding).
std::vector<std::uint8_t> known_storage(std::size_t storage_rows, std::size_t row_bytes,
                                        std::size_t pitch) {
  std::vector<std::uint8_t> bytes((storage_rows - 1) * pitch + row_bytes, kPaddingFill);
  for (std::size_t row = 0; row < storage_rows; ++row) {
    for (std::size_t col = 0; col < row_bytes; ++col) {
      bytes[row * pitch + col] = known_byte(row, col);
    }
  }
  return bytes;
}

/// The same sample tightly packed, as the backend must see it.
std::vector<std::uint8_t> known_tight(std::size_t storage_rows, std::size_t row_bytes) {
  return known_storage(storage_rows, row_bytes, row_bytes);
}

class HolovizOpImageDataPathTest : public HolovizOpImageAdmissionFixture,
                                   public ::testing::WithParamInterface<DataPathCase> {};

// Every residency and pitch must deliver the producer's exact bytes to the backend: the host and
// pinned paths through the repacked cache and staging upload, the CUDA path through the direct
// device pointer (with the NV12 chroma plane at height * pitch). ImageLayerCache::last_frame is
// the oracle for all three, read back plane by plane with the frame's own pitch.
TEST_P(HolovizOpImageDataPathTest, DeliversKnownPixelsThroughEveryResidencyAndPitch) {
  const DataPathCase& c = GetParam();
  ASSERT_NO_FATAL_FAILURE(make_operator(c.format));
  ASSERT_TRUE(cache_under_test().bytes.empty());

  const bool nv12 = c.format == ImageFormat::kNV12;
  const auto layout = backends::image_layout(
      c.format, static_cast<std::uint32_t>(kWidth), static_cast<std::uint32_t>(kHeight));
  ASSERT_TRUE(layout.has_value());
  const std::size_t row_bytes = layout->row_bytes[0];
  const std::size_t storage_rows = layout->rows[0] + (nv12 ? layout->rows[1] : 0);
  const std::size_t pitch = c.padded ? row_bytes + kPaddingBytes : row_bytes;

  const std::vector<std::int64_t> shape =
      nv12 ? std::vector<std::int64_t>{static_cast<std::int64_t>(storage_rows), kWidth}
           : std::vector<std::int64_t>{kHeight, kWidth, 2};
  std::optional<TensorByteStrides> strides;
  if (c.padded) {
    const auto pitch_i = static_cast<std::int64_t>(pitch);
    strides = nv12 ? TensorByteStrides{pitch_i, 1} : TensorByteStrides{pitch_i, 2, 1};
  }

  // The CUDA sample lands on the current device; on a multi-GPU host where HoloViz picks another
  // adapter, compute() reports the cross-device mismatch rather than reading the wrong memory.
  Tensor sample = make_tensor_of(storage_type(c.residency), kU8, shape, strides);
  ASSERT_NE(sample.data(), nullptr);
  ASSERT_NO_FATAL_FAILURE(fill_tensor(sample, known_storage(storage_rows, row_bytes, pitch)));
  // The declared extent comes from the case, not from the Tensor's shape: for NV12/YUYV the
  // backing Tensor's shape is a storage layout (plane rows by row bytes), not the image's
  // width and height. NV12 also carries the plane records at the producer's pitch.
  ASSERT_NO_FATAL_FAILURE(
      activate(make_image_sample(c.format,
                                 std::move(sample),
                                 std::nullopt,
                                 std::nullopt,
                                 kDeclaredExtent,
                                 nv12 ? nv12_plane_layouts(kWidth, kHeight, pitch)
                                      : std::vector<schema::ImagePlaneLayout>{})));
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(reason_under_test(), ImageRejectReason::kAccepted);

  const HolovizOp::Impl::ImageLayerCache& cache = cache_under_test();
  EXPECT_EQ(cache.width, static_cast<std::uint32_t>(kWidth));
  EXPECT_EQ(cache.height, static_cast<std::uint32_t>(kHeight));
  EXPECT_EQ(cache.pitch, c.residency == Residency::kCuda ? pitch : row_bytes);

  const std::vector<std::uint8_t> expected = known_tight(storage_rows, row_bytes);
  if (c.residency == Residency::kCuda) {
    EXPECT_TRUE(cache.bytes.empty()) << "direct device-pointer path must not repack on the host";
  } else {
    ASSERT_EQ(cache.bytes.size(), expected.size());
    EXPECT_TRUE(std::equal(
        cache.bytes.begin(),
        cache.bytes.end(),
        expected.begin(),
        [](std::byte a, std::uint8_t b) { return std::to_integer<std::uint8_t>(a) == b; }))
        << "host repack must drop the row padding and keep every active byte";
  }

  const backends::ImageFrameData& frame = cache.last_frame;
  EXPECT_EQ(frame.width, static_cast<std::uint32_t>(kWidth));
  EXPECT_EQ(frame.height, static_cast<std::uint32_t>(kHeight));
  ASSERT_EQ(frame.plane_count, layout->plane_count);
  for (std::uint32_t plane = 0; plane < frame.plane_count; ++plane) {
    ASSERT_NE(frame.planes[plane].device_ptr, 0U) << "plane " << plane;
    const std::size_t plane_row_bytes = layout->row_bytes[plane];
    const std::size_t plane_rows = layout->rows[plane];
    std::vector<std::uint8_t> actual(plane_row_bytes * plane_rows);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    const auto* source = reinterpret_cast<const void*>(frame.planes[plane].device_ptr);
    ASSERT_EQ(cudaMemcpy2D(actual.data(),
                           plane_row_bytes,
                           source,
                           frame.planes[plane].row_pitch_bytes,
                           plane_row_bytes,
                           plane_rows,
                           cudaMemcpyDeviceToHost),
              cudaSuccess)
        << "plane " << plane;
    const auto* expected_plane = expected.data() + layout->offsets[plane];
    EXPECT_TRUE(std::equal(actual.begin(), actual.end(), expected_plane))
        << "plane " << plane << " bytes at the backend differ from the producer's sample";
  }
}

INSTANTIATE_TEST_SUITE_P(Residency, HolovizOpImageDataPathTest,
                         ::testing::ValuesIn(data_path_cases()), data_path_case_name);

// The wire ImageT's encoding/color_space is validated against the layer's configured
// ImageLayerDesc before the tensor-shape checks even run: a mismatch is rejected the same way a
// bad tensor shape is (per-layer soft drop, peer still renders), per the ImageT port alignment
// design doc's rejection matrix (Decision 3).
TEST_F(HolovizOpImageAdmissionFixture, WireEncodingMismatchIsRejectedIndependentlyOfTensorShape) {
  ASSERT_NO_FATAL_FAILURE(make_operator(ImageFormat::kR8G8B8A8Unorm));

  // Tensor shape/dtype matches kR8G8B8A8Unorm exactly, but the wire encoding claims MONO8.
  ASSERT_NO_FATAL_FAILURE(activate(make_image_sample(ImageFormat::kR8G8B8A8Unorm,
                                                     make_host_tensor(kU8, {kHeight, kWidth, 4}),
                                                     schema::ImageEncoding_MONO8)));
  EXPECT_TRUE(cache_under_test().bytes.empty())
      << "encoding mismatch must be rejected even when the tensor shape matches";
  EXPECT_EQ(reason_under_test(), ImageRejectReason::kEncodingMismatch);

  // An encoding this operator has no ImageFormat for is its own reason.
  ASSERT_NO_FATAL_FAILURE(activate(make_image_sample(ImageFormat::kR8G8B8A8Unorm,
                                                     make_host_tensor(kU8, {kHeight, kWidth, 4}),
                                                     schema::ImageEncoding_I420)));
  EXPECT_TRUE(cache_under_test().bytes.empty());
  EXPECT_EQ(reason_under_test(), ImageRejectReason::kUnmappedEncoding);

  // An sRGB-labelled frame in a kPassThrough layer would be sampled as linear -- a silently
  // wrong image -- so it is rejected rather than rendered.
  ASSERT_NO_FATAL_FAILURE(activate(make_image_sample(ImageFormat::kR8G8B8A8Unorm,
                                                     make_host_tensor(kU8, {kHeight, kWidth, 4}),
                                                     std::nullopt,
                                                     schema::ColorSpace_SRGB)));
  EXPECT_TRUE(cache_under_test().bytes.empty())
      << "SRGB wire color_space into a kPassThrough layer must be rejected";
  EXPECT_EQ(reason_under_test(), ImageRejectReason::kColorSpaceIncompatible);

  // A correctly-matching frame resumes the layer.
  ASSERT_NO_FATAL_FAILURE(activate(
      make_image_sample(ImageFormat::kR8G8B8A8Unorm, make_host_tensor(kU8, {kHeight, kWidth, 4}))));
  EXPECT_FALSE(cache_under_test().bytes.empty());
  EXPECT_EQ(reason_under_test(), ImageRejectReason::kAccepted);
}

// The wire color_space is checked against what the layer declares. UNSPECIFIED is the absence of
// a claim and is accepted into any layer (the declared color space is the authority); a label
// must agree with the layer.
TEST_F(HolovizOpImageAdmissionFixture, LabelledColorSpaceMustAgreeWithTheLayer) {
  ASSERT_NO_FATAL_FAILURE(make_operator(ImageFormat::kR8G8B8A8Unorm, ColorSpace::kSRGBNonlinear));

  ASSERT_NO_FATAL_FAILURE(activate(make_image_sample(ImageFormat::kR8G8B8A8Unorm,
                                                     make_host_tensor(kU8, {kHeight, kWidth, 4}),
                                                     std::nullopt,
                                                     schema::ColorSpace_SRGB)));
  EXPECT_FALSE(cache_under_test().bytes.empty()) << "SRGB frame into an sRGB layer is admitted";
  EXPECT_EQ(reason_under_test(), ImageRejectReason::kAccepted);

  ASSERT_NO_FATAL_FAILURE(activate(make_image_sample(
      ImageFormat::kR8G8B8A8Unorm, make_host_tensor(kU8, {kHeight + 2, kWidth, 4}))));
  EXPECT_EQ(cache_under_test().height, static_cast<std::uint32_t>(kHeight + 2))
      << "UNSPECIFIED frame into an sRGB layer is admitted";
  EXPECT_EQ(reason_under_test(), ImageRejectReason::kAccepted);

  // REC709 names a YUV matrix, not a transfer function: on an RGB layer it agrees with nothing.
  ASSERT_NO_FATAL_FAILURE(activate(make_image_sample(ImageFormat::kR8G8B8A8Unorm,
                                                     make_host_tensor(kU8, {kHeight, kWidth, 4}),
                                                     std::nullopt,
                                                     schema::ColorSpace_REC709)));
  EXPECT_EQ(cache_under_test().height, static_cast<std::uint32_t>(kHeight + 2));
  EXPECT_EQ(reason_under_test(), ImageRejectReason::kColorSpaceIncompatible);

  // HDR transfer functions have no counterpart in this operator.
  ASSERT_NO_FATAL_FAILURE(activate(make_image_sample(ImageFormat::kR8G8B8A8Unorm,
                                                     make_host_tensor(kU8, {kHeight, kWidth, 4}),
                                                     std::nullopt,
                                                     schema::ColorSpace_REC2020_PQ)));
  EXPECT_EQ(reason_under_test(), ImageRejectReason::kColorSpaceIncompatible);
}

// On a YUV layer a REC601/709/2020 label is a second statement of the matrix the layer declares
// through yuv_model; the two must agree.
TEST_F(HolovizOpImageAdmissionFixture, YuvColorSpaceLabelMustMatchTheDeclaredModel) {
  ASSERT_NO_FATAL_FAILURE(make_operator(ImageFormat::kNV12));  // yuv_model = kBT709

  const auto nv12_sample = [](schema::ColorSpace color_space) {
    return make_image_sample(ImageFormat::kNV12,
                             make_host_tensor(kU8, {kHeight * 3 / 2, kWidth}),
                             std::nullopt,
                             color_space,
                             kDeclaredExtent,
                             nv12_plane_layouts(kWidth, kHeight, kWidth));
  };
  ASSERT_NO_FATAL_FAILURE(activate(nv12_sample(schema::ColorSpace_REC709)));
  EXPECT_FALSE(cache_under_test().bytes.empty()) << "REC709 agrees with yuv_model kBT709";
  EXPECT_EQ(reason_under_test(), ImageRejectReason::kAccepted);

  ASSERT_NO_FATAL_FAILURE(activate(nv12_sample(schema::ColorSpace_REC601)));
  EXPECT_EQ(reason_under_test(), ImageRejectReason::kColorSpaceIncompatible)
      << "REC601 contradicts yuv_model kBT709";

  ASSERT_NO_FATAL_FAILURE(activate(nv12_sample(schema::ColorSpace_UNSPECIFIED)));
  EXPECT_EQ(reason_under_test(), ImageRejectReason::kAccepted);
}

// NV12 planes live wherever plane_layouts say they do: the chroma plane need not follow the luma
// plane, and the backing Tensor can be a flat byte buffer. The host repack must gather both
// planes into the tight staging layout.
TEST_F(HolovizOpImageAdmissionFixture, Nv12PlaneLayoutsLocateNonAdjacentChromaPlane) {
  ASSERT_NO_FATAL_FAILURE(make_operator(ImageFormat::kNV12));

  constexpr std::size_t kWidthBytes = static_cast<std::size_t>(kWidth);
  constexpr std::size_t kLumaRows = static_cast<std::size_t>(kHeight);
  constexpr std::size_t kChromaRows = kLumaRows / 2;
  constexpr std::size_t kGap = 64;  // bytes between the luma and chroma planes
  constexpr std::size_t kChromaOffset = kLumaRows * kWidthBytes + kGap;
  std::vector<std::uint8_t> bytes(kChromaOffset + kChromaRows * kWidthBytes, kPaddingFill);
  for (std::size_t row = 0; row < kLumaRows; ++row) {
    for (std::size_t col = 0; col < kWidthBytes; ++col) {
      bytes[row * kWidthBytes + col] = known_byte(row, col);
    }
  }
  for (std::size_t row = 0; row < kChromaRows; ++row) {
    for (std::size_t col = 0; col < kWidthBytes; ++col) {
      bytes[kChromaOffset + row * kWidthBytes + col] = known_byte(kLumaRows + row, col);
    }
  }
  Tensor flat = make_host_tensor(kU8, {static_cast<std::int64_t>(bytes.size())});
  ASSERT_NO_FATAL_FAILURE(fill_tensor(flat, bytes));

  const std::vector<schema::ImagePlaneLayout> planes{
      schema::ImagePlaneLayout{
          0U, static_cast<std::uint32_t>(kWidthBytes), kLumaRows * kWidthBytes},
      schema::ImagePlaneLayout{
          kChromaOffset, static_cast<std::uint32_t>(kWidthBytes), kChromaRows * kWidthBytes}};
  ASSERT_NO_FATAL_FAILURE(activate(make_image_sample(
      ImageFormat::kNV12, std::move(flat), std::nullopt, std::nullopt, kDeclaredExtent, planes)));
  EXPECT_EQ(reason_under_test(), ImageRejectReason::kAccepted);

  const HolovizOp::Impl::ImageLayerCache& cache = cache_under_test();
  EXPECT_EQ(cache.width, static_cast<std::uint32_t>(kWidth));
  EXPECT_EQ(cache.height, static_cast<std::uint32_t>(kHeight));
  const std::vector<std::uint8_t> expected = known_tight(kLumaRows + kChromaRows, kWidthBytes);
  ASSERT_EQ(cache.bytes.size(), expected.size());
  EXPECT_TRUE(
      std::equal(cache.bytes.begin(),
                 cache.bytes.end(),
                 expected.begin(),
                 [](std::byte a, std::uint8_t b) { return std::to_integer<std::uint8_t>(a) == b; }))
      << "both planes must be gathered from their own offsets into the tight layout";
}

// Plane records that do not fit the encoding's geometry or the Tensor are rejected before any
// byte is read: the synthetic fixture bypasses Core's wire validation, so the operator's own
// containment check is what stands between a bad record and an out-of-bounds read.
TEST_F(HolovizOpImageAdmissionFixture, Nv12PlaneRecordsMustFitTheTensor) {
  ASSERT_NO_FATAL_FAILURE(make_operator(ImageFormat::kNV12));

  const auto nv12_with = [](std::vector<schema::ImagePlaneLayout> planes) {
    return make_image_sample(ImageFormat::kNV12,
                             make_host_tensor(kU8, {kHeight * 3 / 2, kWidth}),
                             std::nullopt,
                             std::nullopt,
                             kDeclaredExtent,
                             std::move(planes));
  };
  // Chroma plane placed past the end of the Tensor.
  auto past_the_end = nv12_plane_layouts(kWidth, kHeight, kWidth);
  past_the_end[1] = schema::ImagePlaneLayout{past_the_end[1].offset_bytes() + 1U,
                                             past_the_end[1].row_stride_bytes(),
                                             past_the_end[1].size_bytes()};
  ASSERT_NO_FATAL_FAILURE(activate(nv12_with(std::move(past_the_end))));
  EXPECT_TRUE(cache_under_test().bytes.empty());
  EXPECT_EQ(reason_under_test(), ImageRejectReason::kPlaneLayoutsInvalid);

  // Row stride shorter than a packed row.
  auto narrow = nv12_plane_layouts(kWidth, kHeight, kWidth);
  narrow[0] =
      schema::ImagePlaneLayout{0U, static_cast<std::uint32_t>(kWidth - 1), narrow[0].size_bytes()};
  ASSERT_NO_FATAL_FAILURE(activate(nv12_with(std::move(narrow))));
  EXPECT_EQ(reason_under_test(), ImageRejectReason::kPlaneLayoutsInvalid);

  // One record for a two-plane encoding.
  ASSERT_NO_FATAL_FAILURE(activate(nv12_with({nv12_plane_layouts(kWidth, kHeight, kWidth)[0]})));
  EXPECT_EQ(reason_under_test(), ImageRejectReason::kPlaneLayoutsInvalid);

  ASSERT_NO_FATAL_FAILURE(activate(nv12_with(nv12_plane_layouts(kWidth, kHeight, kWidth))));
  EXPECT_FALSE(cache_under_test().bytes.empty());
  EXPECT_EQ(reason_under_test(), ImageRejectReason::kAccepted);
}

// The schema permits ImageT::data's backing Tensor to be larger than the declared image (row
// padding lives in the Tensor's own strides, not a wider shape) -- but this operator has no
// mechanism to address a sub-region smaller than the Tensor's HWC extent, so a frame whose
// declared width/height disagree with the Tensor's shape must be rejected rather than rendering
// the Tensor's full extent (pixels outside the declared image, with no diagnostic).
TEST_F(HolovizOpImageAdmissionFixture, DeclaredExtentMismatchWithBackingTensorIsRejected) {
  ASSERT_NO_FATAL_FAILURE(make_operator(ImageFormat::kR8G8B8A8Unorm));

  // Declares 8x4 but is backed by a 16x8 Tensor -- schema-valid (the Tensor may be larger than
  // the declared image), but this operator cannot honor it.
  ASSERT_NO_FATAL_FAILURE(
      activate(make_image_sample(ImageFormat::kR8G8B8A8Unorm,
                                 make_host_tensor(kU8, {8, 16, 4}),
                                 std::nullopt,
                                 std::nullopt,
                                 std::make_pair(std::int32_t{8}, std::int32_t{4}))));
  EXPECT_TRUE(cache_under_test().bytes.empty())
      << "declared width/height disagreeing with the backing Tensor's shape must be rejected";
  EXPECT_EQ(reason_under_test(), ImageRejectReason::kGeometryMismatch);

  // A frame whose declared extent actually matches its Tensor resumes the layer.
  ASSERT_NO_FATAL_FAILURE(activate(
      make_image_sample(ImageFormat::kR8G8B8A8Unorm, make_host_tensor(kU8, {kHeight, kWidth, 4}))));
  EXPECT_FALSE(cache_under_test().bytes.empty());
}

// The named 16-bit RGBA encoding is admitted into a kR16G16B16A16Unorm layer for both the natural
// uint16 layout and its packed-byte alias, and it is checked like any other named encoding.
TEST_F(HolovizOpImageAdmissionFixture, NamedRgba16EncodingIsAdmitted) {
  ASSERT_NO_FATAL_FAILURE(make_operator(ImageFormat::kR16G16B16A16Unorm));

  ASSERT_NO_FATAL_FAILURE(activate(make_image_sample(ImageFormat::kR16G16B16A16Unorm,
                                                     make_host_tensor(kU16, {kHeight, kWidth, 4}),
                                                     schema::ImageEncoding_RGBA16)));
  EXPECT_FALSE(cache_under_test().bytes.empty());
  EXPECT_EQ(reason_under_test(), ImageRejectReason::kAccepted);

  ASSERT_NO_FATAL_FAILURE(activate(make_image_sample(ImageFormat::kR16G16B16A16Unorm,
                                                     make_host_tensor(kU8, {kHeight, kWidth, 8}),
                                                     schema::ImageEncoding_RGBA16)));
  EXPECT_EQ(reason_under_test(), ImageRejectReason::kAccepted);

  // RGBA16 into an 8-bit RGBA layer is an encoding mismatch, not a layout problem.
  ASSERT_NO_FATAL_FAILURE(make_operator(ImageFormat::kR8G8B8A8Unorm));
  ASSERT_NO_FATAL_FAILURE(activate(make_image_sample(ImageFormat::kR8G8B8A8Unorm,
                                                     make_host_tensor(kU8, {kHeight, kWidth, 4}),
                                                     schema::ImageEncoding_RGBA16)));
  EXPECT_TRUE(cache_under_test().bytes.empty());
  EXPECT_EQ(reason_under_test(), ImageRejectReason::kEncodingMismatch);
}

// ImageEncoding_CUSTOM defers to the tensor-level shape/dtype check instead of the encoding
// cross-check (see check_image_semantics()'s doc comment) -- but a CUSTOM frame with non-empty
// plane_layouts is asserting a multi-plane organization this operator's single-plane HWC upload
// path never consults, and must still be rejected rather than silently read as the wrong plane.
TEST_F(HolovizOpImageAdmissionFixture, CustomEncodingWithPlaneLayoutsIsRejected) {
  ASSERT_NO_FATAL_FAILURE(make_operator(ImageFormat::kR16G16B16A16Unorm));

  schema::ImageT sample = make_image_sample(ImageFormat::kR16G16B16A16Unorm,
                                            make_host_tensor(kU16, {kHeight, kWidth, 4}),
                                            schema::ImageEncoding_CUSTOM);
  sample.plane_layouts.push_back(
      schema::ImagePlaneLayout{0U,
                               static_cast<std::uint32_t>(kWidth) * 8U,
                               static_cast<std::uint64_t>(kHeight) * kWidth * 8U});
  ASSERT_NO_FATAL_FAILURE(activate(sample));
  EXPECT_TRUE(cache_under_test().bytes.empty())
      << "CUSTOM frame with non-empty plane_layouts must be rejected";
  EXPECT_EQ(reason_under_test(), ImageRejectReason::kPlaneLayoutsOnCustom);

  // A CUSTOM frame with no plane_layouts (a producer that does not classify its layout) defers to
  // the dtype/shape check and resumes the layer.
  ASSERT_NO_FATAL_FAILURE(activate(make_image_sample(ImageFormat::kR16G16B16A16Unorm,
                                                     make_host_tensor(kU16, {kHeight, kWidth, 4}),
                                                     schema::ImageEncoding_CUSTOM)));
  EXPECT_FALSE(cache_under_test().bytes.empty());
  EXPECT_EQ(reason_under_test(), ImageRejectReason::kAccepted);
}

// Layer-budget and mixed-order compile coverage lives in test_holoviz_op_contract.cpp (no
// window needed there); this fixture covers the runtime side of slot assignment.

// Slot assignment partitions by kind, not by declaration order (Decision 1 in the ImageT port
// alignment design doc): a geometry layer declared before an image layer must not force a
// reorder, nor land on an image-typed physical port slot.
TEST_F(HolovizOpImageAdmissionFixture, SlotAssignmentIsByKindNotDeclarationOrder) {
  LayerStack stack;
  stack.add_geometry_layer("detections",
                           GeometryLayerDesc{{GeometryTopology::kRectangleList}, 256, 1.0f});
  stack.add_image_layer("video", ImageLayerDesc{ImageFormat::kR8G8B8A8Unorm, kMaxDim, kMaxDim});

  op_ = std::make_unique<HolovizOp>(
      std::move(stack),
      WindowSinkDesc{.title = "slot assignment order", .width = 160, .height = 120});
  fixture_ = std::make_unique<holoscan::test::SyntheticOperatorFixture<HolovizOp>>(*op_);
  ASSERT_TRUE(fixture_->declare_ports().has_value());
  try {
    op_->start();
  } catch (const holoscan::RuntimeError& error) {
    FAIL() << "HolovizOp::start() failed: " << error.what();
  }

  // The image layer (declared second, layer index 1) must bind to image_layer_input(0) -- the
  // first image-pool slot -- because it is the first *image-kind* layer declared, independent
  // of its position in the overall LayerStack.
  const schema::ImageT sample =
      make_image_sample(ImageFormat::kR8G8B8A8Unorm, make_host_tensor(kU8, {kHeight, kWidth, 4}));
  ASSERT_TRUE(
      fixture_->bind_value(HolovizOp::Impl::of(*op_).image_layer_input(0), sample).has_value());

  for (int attempt = 0; attempt < 200; ++attempt) {
    auto result = fixture_->invoke_once(
        holoscan::detail::SyntheticActivationOptions{.cuda_stream = activation_stream()});
    if (!result && result.error().code == holoscan::ErrorCode::kIncompatible &&
        result.error().message.find("cross-device") != std::string::npos && select_next_device()) {
      continue;  // multi-GPU host: HoloViz picked a different device than the current one
    }
    ASSERT_TRUE(result.has_value()) << result.error().message;
    // image_layer_cache() is indexed by declared layer index (1 = the image layer here), not by
    // physical port slot -- see holoviz_op_impl.hpp's doc comment.
    if (!HolovizOp::Impl::of(*op_).image_layer_cache(1).bytes.empty()) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  FAIL() << "image layer declared after a geometry layer never admitted its sample";
}

}  // namespace
}  // namespace holoscan::visualization
