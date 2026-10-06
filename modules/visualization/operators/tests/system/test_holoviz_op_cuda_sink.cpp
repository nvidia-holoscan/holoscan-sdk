// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// HolovizOp through its CUDA sink, driven by a real Graph + RunSession:
//  - an empty layer stack publishes clear-color frames on the "output" port as schema::ImageT
//    carrying the declared geometry, wire encoding, color space, and a CUDA Tensor in the
//    format's native element type (RGBA8, MONO16, sRGB-stamped RGBA8, and both BGRA8 kinds);
//  - a host RGBA8 gradient layer survives the capture byte-for-byte, and channel-swapped through
//    a BGRA8 sink;
//  - a second HolovizOp consumes the first one's output as a CUDA image layer (the zero-copy
//    chain the cuda_sink_example builds) and re-captures it unchanged;
//  - under the shared process-device stream policy (one compute stream, logical pool
//    retirement), a consumer whose asynchronous read of a frame is still queued when the pool
//    recycles that frame's allocation still reads its own frame.
// Skips (exit 77) when no CUDA device is available.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <cuda.h>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/schema_identity.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/visualization/holoviz_op.hpp>
#include <holoviz/adapter.hpp>

#include "graph_run_test_util.hpp"

namespace {

using namespace std::chrono_literals;
namespace viz = holoscan::visualization;
using viz::test_util::expect_clean_stop;
using viz::test_util::run_until_done;

constexpr std::uint32_t kWidth = 64;
constexpr std::uint32_t kHeight = 48;
constexpr std::uint32_t kBytesPerPixel = 4;
constexpr std::size_t kTargetFrames = 5;
// RenderConfig's default clear color (opaque black), as RGBA8 bytes.
constexpr std::array<std::uint8_t, 4> kClearRgba{0, 0, 0, 255};

/// One CudaSinkDesc format and the ImageT contract HolovizOp publishes for it (the table in the
/// CudaSinkDesc documentation): wire encoding and color space, the data Tensor's element type
/// and rank, and the packed pixel size the checker copies back.
struct SinkCase {
  viz::ImageFormat format;
  holoscan::schema::ImageEncoding encoding;
  holoscan::schema::ColorSpace color_space;
  DLDataType dtype;
  std::uint32_t rank;
  std::uint32_t bytes_per_pixel;
  const char* name;
};

constexpr std::array<SinkCase, 5> kSinkCases{{
    {viz::ImageFormat::kR8G8B8A8Unorm,
     holoscan::schema::ImageEncoding_RGBA8,
     holoscan::schema::ColorSpace_UNSPECIFIED,
     DLDataType{kDLUInt, 8U, 1U},
     3U,
     4U,
     "Rgba8"},
    {viz::ImageFormat::kR16Unorm,
     holoscan::schema::ImageEncoding_MONO16,
     holoscan::schema::ColorSpace_UNSPECIFIED,
     DLDataType{kDLUInt, 16U, 1U},
     2U,
     2U,
     "Mono16"},
    {viz::ImageFormat::kR8G8B8A8Srgb,
     holoscan::schema::ImageEncoding_RGBA8,
     holoscan::schema::ColorSpace_SRGB,
     DLDataType{kDLUInt, 8U, 1U},
     3U,
     4U,
     "Rgba8Srgb"},
    // The BGRA kinds render into an RGBA target and are channel-swapped in the resolve; they
    // must start and publish under the same contract as their RGBA siblings.
    {viz::ImageFormat::kB8G8R8A8Unorm,
     holoscan::schema::ImageEncoding_BGRA8,
     holoscan::schema::ColorSpace_UNSPECIFIED,
     DLDataType{kDLUInt, 8U, 1U},
     3U,
     4U,
     "Bgra8"},
    {viz::ImageFormat::kB8G8R8A8Srgb,
     holoscan::schema::ImageEncoding_BGRA8,
     holoscan::schema::ColorSpace_SRGB,
     DLDataType{kDLUInt, 8U, 1U},
     3U,
     4U,
     "Bgra8Srgb"},
}};

bool is_bgra(viz::ImageFormat format) {
  return format == viz::ImageFormat::kB8G8R8A8Unorm || format == viz::ImageFormat::kB8G8R8A8Srgb;
}

const SinkCase& sink_case(viz::ImageFormat format) {
  for (const SinkCase& c : kSinkCases) {
    if (c.format == format) {
      return c;
    }
  }
  std::terminate();
}

/// The clear color as the bytes of one packed pixel in \p format: opaque black is all-zero
/// except an 8-bit alpha of 255; a single-channel format has no alpha.
std::vector<std::uint8_t> clear_pixel_bytes(const SinkCase& c) {
  std::vector<std::uint8_t> bytes(c.bytes_per_pixel, 0U);
  if (c.rank == 3U) {
    bytes[c.bytes_per_pixel - 1U] = 255U;
  }
  return bytes;
}

const holoscan::ConnectionOptions kMailbox{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                           .queue_depth = 1U};

/// Byte-wise comparison with a one-count tolerance for the float -> UNORM8 rounding.
bool matches(const std::uint8_t* pixel, const std::array<std::uint8_t, 4>& expected) {
  for (std::size_t c = 0; c < 4; ++c) {
    const int diff = static_cast<int>(pixel[c]) - static_cast<int>(expected[c]);
    if (diff < -1 || diff > 1) {
      return false;
    }
  }
  return true;
}

/// The CUDA ordinal to place the output pools on, or -1 without a usable device: the same
/// selection HolovizOp's CUDA sink makes in start() — the first HoloViz adapter backed by a CUDA
/// device — so the pools land on the operator's device on multi-GPU hosts too (the operator
/// rejects a mismatched activation device otherwise).
int holoviz_cuda_device() {
  auto adapters = holoviz::enumerate_adapters();
  if (!adapters) {
    return -1;
  }
  for (const holoviz::Adapter& adapter : *adapters) {
    if (adapter.cuda_device_index >= 0) {
      return adapter.cuda_device_index;
    }
  }
  return -1;
}

/// Host RGBA8 gradient: R = 4x, G = 5y, B = R ^ G, A = 255 (kWidth x kHeight keeps all < 256).
/// The layer is uploaded as RGBA8; a BGRA sink publishes it with bytes 0 and 2 exchanged.
std::uint8_t gradient_byte(std::size_t x, std::size_t y, std::size_t c) {
  switch (c) {
    case 0:
      return static_cast<std::uint8_t>(x * 4);
    case 1:
      return static_cast<std::uint8_t>(y * 5);
    case 2:
      return static_cast<std::uint8_t>((x * 4) ^ (y * 5));
    default:
      return 255;
  }
}

struct FrameStats {
  std::atomic<std::size_t> frames{0};
  std::atomic<std::size_t> metadata_mismatches{0};  ///< ImageT wire fields off the contract
  std::atomic<std::size_t> shape_mismatches{0};     ///< data Tensor dtype/shape/residency
  std::atomic<std::size_t> pixel_mismatches{0};

  void reset() {
    frames = 0;
    metadata_mismatches = 0;
    shape_mismatches = 0;
    pixel_mismatches = 0;
  }
};

// Graph reconstructs operators from copied constructor arguments and rejects pointer/callable
// captures, so the checkers reach their statistics through this file-static instance.
FrameStats g_stats;

/// What a checker expects in every frame.
enum class Expectation : std::uint8_t { kClearColor, kGradient };

/// Sink operator for the captured frames: validates the ImageT's wire fields against the sink
/// format's contract, the data Tensor's dtype/shape/residency, and, once the first non-clear
/// frame arrived for a gradient expectation, every pixel; asks for a cooperative stop after
/// enough frames. The gradient expectation is 8-bit four-channel only (RGBA8 or BGRA8).
class FrameChecker final : public holoscan::Operator<> {
 public:
  explicit FrameChecker(Expectation expectation,
                        viz::ImageFormat format = viz::ImageFormat::kR8G8B8A8Unorm)
      : expectation_(expectation), format_(format) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(frames, "frames").queue_depth(4U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{frames});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto image = frames.receive_data();
    if (!image) {
      return holoscan::make_unexpected(std::move(image).error());
    }
    const SinkCase& expected = sink_case(format_);
    // The wire fields are the contract a downstream HolovizOp validates against its own layer:
    // geometry, encoding, and color space must follow the sink's declared format.
    const bool metadata_ok = image->width == static_cast<std::int32_t>(kWidth) &&
                             image->height == static_cast<std::int32_t>(kHeight) &&
                             image->encoding == expected.encoding &&
                             image->color_space == expected.color_space &&
                             image->plane_layouts.empty();
    if (!metadata_ok) {
      g_stats.metadata_mismatches.fetch_add(1, std::memory_order_relaxed);
    }
    if (!image->data) {
      g_stats.shape_mismatches.fetch_add(1, std::memory_order_relaxed);
      return {};
    }
    const holoscan::Tensor& tensor = *image->data;
    const std::vector<std::int64_t> shape = tensor.shape();
    const DLDataType dtype = tensor.dtype();
    // Native element type, {H, W} for one channel and {H, W, 4} otherwise.
    const bool shape_ok = shape.size() == expected.rank && shape[0] == kHeight &&
                          shape[1] == kWidth && (expected.rank == 2U || shape[2] == 4) &&
                          dtype.code == expected.dtype.code && dtype.bits == expected.dtype.bits &&
                          dtype.lanes == 1U && tensor.device().device_type == kDLCUDA;
    if (!shape_ok) {
      g_stats.shape_mismatches.fetch_add(1, std::memory_order_relaxed);
      return {};
    }
    const std::size_t bpp = expected.bytes_per_pixel;
    std::vector<std::uint8_t> host(static_cast<std::size_t>(kWidth) * kHeight * bpp);
    auto copied = tensor.copy_to_host(host.data(), host.size(), context.cuda_stream());
    if (!copied) {
      return holoscan::make_unexpected(std::move(copied).error());
    }
    std::size_t mismatches = 0;
    if (expectation_ == Expectation::kClearColor) {
      const std::vector<std::uint8_t> clear = clear_pixel_bytes(expected);
      for (const std::size_t pixel : {std::size_t{0},
                                      static_cast<std::size_t>(kWidth) * (kHeight / 2) + kWidth / 2,
                                      static_cast<std::size_t>(kWidth) * kHeight - 1}) {
        if (!std::equal(clear.begin(), clear.end(), &host[pixel * bpp])) {
          ++mismatches;
        }
      }
    } else {
      // The gradient is black at (0, 0) like the clear colour; probe the centre pixel instead
      // to tell "the layer has not reached the operator yet" from a real mismatch.
      const std::size_t probe =
          (static_cast<std::size_t>(kWidth) * (kHeight / 2) + kWidth / 2) * kBytesPerPixel;
      if (matches(&host[probe], kClearRgba) && !seen_layer_) {
        return {};
      }
      seen_layer_ = true;
      const bool swapped = is_bgra(format_);
      for (std::size_t y = 0; y < kHeight; ++y) {
        for (std::size_t x = 0; x < kWidth; ++x) {
          for (std::size_t c = 0; c < kBytesPerPixel; ++c) {
            const int got = host[(y * kWidth + x) * kBytesPerPixel + c];
            const std::size_t source_channel = swapped && c != 3 ? 2 - c : c;
            const int want = gradient_byte(x, y, source_channel);
            if (got < want - 1 || got > want + 1) {
              ++mismatches;
            }
          }
        }
      }
    }
    g_stats.pixel_mismatches.fetch_add(mismatches, std::memory_order_relaxed);
    if (g_stats.frames.fetch_add(1, std::memory_order_relaxed) + 1 >= kTargetFrames) {
      context.request_stop();
    }
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> frames;

 private:
  Expectation expectation_;
  viz::ImageFormat format_;
  bool seen_layer_ = false;
};

/// Consumer for the shared-stream recycling test: takes a blocking host copy of each frame as
/// ground truth, then queues a host-side delay and an asynchronous device copy of the same frame
/// on its activation stream and returns — releasing the Tensor while that read is still queued.
/// Under same-stream readiness elision the pool recycles the allocation logically (no GPU
/// event), so the producer's next resolve into it must be ordered after this queued read; the
/// delayed copy is compared with the ground truth on the following activation.
class DelayedReadChecker final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(frames, "frames").queue_depth(4U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{frames});
    return result;
  }

  void start() override {
    CUdevice device = 0;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&device, holoviz_cuda_device()) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&context_, device) != CUDA_SUCCESS) {
      throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure,
                                   "DelayedReadChecker: CUDA context unavailable"};
    }
    CUcontext previous = nullptr;
    (void)cuCtxGetCurrent(&previous);
    if (cuCtxSetCurrent(context_) != CUDA_SUCCESS) {
      throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure,
                                   "DelayedReadChecker: cuCtxSetCurrent failed"};
    }
    for (Pending& pending : pending_) {
      if (cuMemAlloc(&pending.scratch, kFrameBytes) != CUDA_SUCCESS ||
          cuEventCreate(&pending.done, CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS) {
        throw holoscan::RuntimeError{holoscan::ErrorCode::kFailure,
                                     "DelayedReadChecker: scratch allocation failed"};
      }
      pending.truth.resize(kFrameBytes);
    }
    (void)cuCtxSetCurrent(previous);
  }

  void stop() override {
    if (context_ == nullptr) {
      return;
    }
    (void)cuCtxSetCurrent(context_);
    for (Pending& pending : pending_) {
      if (pending.armed) {
        check(pending);
      }
      if (pending.scratch != 0) {
        (void)cuMemFree(pending.scratch);
        pending.scratch = 0;
      }
      if (pending.done != nullptr) {
        (void)cuEventDestroy(pending.done);
        pending.done = nullptr;
      }
    }
    CUdevice device = 0;
    if (cuCtxGetDevice(&device) == CUDA_SUCCESS) {
      (void)cuDevicePrimaryCtxRelease(device);
    }
    context_ = nullptr;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto image = frames.receive_data();
    if (!image) {
      return holoscan::make_unexpected(std::move(image).error());
    }
    if (!image->data || image->data->nbytes() < static_cast<std::int64_t>(kFrameBytes) ||
        image->data->device().device_type != kDLCUDA) {
      g_stats.shape_mismatches.fetch_add(1, std::memory_order_relaxed);
      return {};
    }
    const std::shared_ptr<holoscan::Tensor> tensor = image->data;
    Pending& pending = pending_[next_];
    next_ = (next_ + 1) % pending_.size();
    if (pending.armed) {
      check(pending);  // the read queued two frames ago has long completed
    }
    auto copied = tensor->copy_to_host(pending.truth.data(), kFrameBytes, context.cuda_stream());
    if (!copied) {
      return holoscan::make_unexpected(std::move(copied).error());
    }
    const CUstream stream = context.cuda_stream().get();
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    const auto source = static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(tensor->data()));
    if (cuLaunchHostFunc(stream, &DelayedReadChecker::sleep_a_while, nullptr) != CUDA_SUCCESS ||
        cuMemcpyDtoDAsync(pending.scratch, source, kFrameBytes, stream) != CUDA_SUCCESS ||
        cuEventRecord(pending.done, stream) != CUDA_SUCCESS) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "DelayedReadChecker: enqueue failed"});
    }
    pending.armed = true;
    if (g_stats.frames.fetch_add(1, std::memory_order_relaxed) + 1 >= kRaceFrames) {
      context.request_stop();
    }
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> frames;

 private:
  static constexpr std::size_t kFrameBytes =
      static_cast<std::size_t>(kWidth) * kHeight * kBytesPerPixel;
  /// Enough frames for the pool to recycle every allocation several times over.
  static constexpr std::size_t kRaceFrames = 12;

  struct Pending {
    CUdeviceptr scratch = 0;
    CUevent done = nullptr;
    std::vector<std::uint8_t> truth;
    bool armed = false;
  };

  static void CUDA_CB sleep_a_while(void* /*user*/) {
    // Widens the window in which the recycled allocation is exposed to the next resolve: long
    // enough for the producer to cycle through every pool slot (source period 8 ms) while this
    // frame's delayed read is still queued.
    std::this_thread::sleep_for(80ms);
  }

  /// Compare the delayed copy with the ground truth taken when the frame arrived.
  void check(Pending& pending) {
    pending.armed = false;
    std::vector<std::uint8_t> delayed(kFrameBytes);
    if (cuEventSynchronize(pending.done) != CUDA_SUCCESS ||
        cuMemcpyDtoH(delayed.data(), pending.scratch, kFrameBytes) != CUDA_SUCCESS) {
      g_stats.pixel_mismatches.fetch_add(1, std::memory_order_relaxed);
      return;
    }
    if (delayed != pending.truth) {
      g_stats.pixel_mismatches.fetch_add(1, std::memory_order_relaxed);
    }
  }

  CUcontext context_ = nullptr;
  std::array<Pending, 2> pending_{};
  std::size_t next_ = 0;
};

class GradientSource final : public holoscan::Operator<> {
 public:
  /// \p animate shifts the gradient by one step per frame so consecutive frames differ.
  explicit GradientSource(bool animate = false) : animate_(animate) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(frame, "frame")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = DLDataType{kDLUInt, 8U, 1U},
                               .rank = 3U},
            .bounds = holoscan::tensor_bounds(static_cast<std::uint64_t>(kWidth) * kHeight *
                                              kBytesPerPixel),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        })
        .payload_options(holoscan::PayloadOptions{
            .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 8ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const std::array<std::int64_t, 3> shape{kHeight, kWidth, kBytesPerPixel};
    auto loan = frame.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = shape, .dtype = DLDataType{kDLUInt, 8U, 1U}});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto committed = loan->write_host_and_commit([&](const holoscan::TensorOutputWriteGuard& writer)
                                                     -> holoscan::expected<void, holoscan::Error> {
      auto* dst = static_cast<std::uint8_t*>(writer.data());
      if (dst == nullptr ||
          writer.byte_size() < static_cast<std::size_t>(kWidth) * kHeight * kBytesPerPixel) {
        return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument});
      }
      const std::uint8_t shift = animate_ ? static_cast<std::uint8_t>(frame_index_ * 7U) : 0U;
      for (std::size_t y = 0; y < kHeight; ++y) {
        for (std::size_t x = 0; x < kWidth; ++x) {
          for (std::size_t c = 0; c < kBytesPerPixel; ++c) {
            dst[(y * kWidth + x) * kBytesPerPixel + c] =
                c < 3 ? static_cast<std::uint8_t>(gradient_byte(x, y, c) + shift)
                      : gradient_byte(x, y, c);
          }
        }
      }
      return {};
    });
    if (!committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    ++frame_index_;
    // emit_tensor() takes the Tensor from the loan; only the wire fields are set here.
    holoscan::schema::ImageT descriptor{};
    descriptor.height = static_cast<std::int32_t>(kHeight);
    descriptor.width = static_cast<std::int32_t>(kWidth);
    descriptor.encoding = holoscan::schema::ImageEncoding_RGBA8;
    descriptor.color_space = holoscan::schema::ColorSpace_UNSPECIFIED;
    return frame.emit_tensor(std::move(*loan),
                             descriptor,
                             holoscan::EmitOptions{.capture_time = context.activation_time()});
  }

  holoscan::Output<holoscan::schema::ImageT> frame;

 private:
  bool animate_ = false;
  std::size_t frame_index_ = 0;
};

viz::CudaSinkDesc cuda_sink(viz::ImageFormat format = viz::ImageFormat::kR8G8B8A8Unorm) {
  return viz::CudaSinkDesc{.width = kWidth, .height = kHeight, .format = format};
}

viz::LayerStack rgba_layer(const char* port) {
  viz::LayerStack stack;
  stack.add_image_layer(port,
                        viz::ImageLayerDesc{viz::ImageFormat::kR8G8B8A8Unorm, kWidth, kHeight});
  return stack;
}

void bind_output_device(holoscan::CompileOptions& options, const char* operator_path,
                        int cuda_device) {
  options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{.operator_path = operator_path,
                                            .output_port = "output",
                                            .device = holoscan::DeviceId{cuda_device}});
}

void run_and_check(holoscan::Graph& graph, holoscan::CompileOptions options) {
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();

  holoscan::RunSession session = holoscan::run_async(plan);
  run_until_done(session, 15s);
  expect_clean_stop(session);
  EXPECT_GE(g_stats.frames.load(), kTargetFrames);
  EXPECT_EQ(g_stats.metadata_mismatches.load(), 0U);
  EXPECT_EQ(g_stats.shape_mismatches.load(), 0U);
  EXPECT_EQ(g_stats.pixel_mismatches.load(), 0U);
}

/// One CudaSinkDesc format per instance; the checker validates the ImageT contract for it.
class HolovizOpCudaSinkFormat : public ::testing::TestWithParam<SinkCase> {};

// Every admitted format kind publishes its clear color under the documented contract: RGBA8 as
// uint8 {H, W, 4}, MONO16 as uint16 {H, W}, and the sRGB kind as RGBA8 stamped ColorSpace_SRGB.
TEST_P(HolovizOpCudaSinkFormat, PublishesClearColorFramesOnTheOutputPort) {
  const int cuda_device = holoviz_cuda_device();
  if (cuda_device < 0) {
    GTEST_SKIP() << "[holoviz-skip] no CUDA device";
  }
  g_stats.reset();
  const SinkCase& c = GetParam();

  holoscan::Graph graph{"holoviz-cuda-sink"};
  const auto viz_op = graph.op<viz::HolovizOp>(
      "holoviz", viz::LayerStack{}, cuda_sink(c.format), viz::ClockTrigger{.period = 4ms});
  const auto checker = graph.op<FrameChecker>("checker", Expectation::kClearColor, c.format);
  graph.add_flow(viz_op->output(), checker->frames, kMailbox);
  graph.partition("render").add(viz_op);
  graph.auto_partition_remaining();

  holoscan::CompileOptions options;
  bind_output_device(options, "holoviz", cuda_device);
  run_and_check(graph, std::move(options));
}

INSTANTIATE_TEST_SUITE_P(Formats, HolovizOpCudaSinkFormat, ::testing::ValuesIn(kSinkCases),
                         [](const ::testing::TestParamInfo<SinkCase>& info) {
                           return std::string{info.param.name};
                         });

/// The sink formats a gradient layer is captured through: RGBA8 byte-for-byte, BGRA8 with the
/// red and blue bytes exchanged by the resolve's channel swap.
class HolovizOpCudaSinkGradient : public ::testing::TestWithParam<viz::ImageFormat> {};

// A non-uniform image layer survives the capture: the exported render target and the pitch-linear
// resolve preserve pixel positions and row pitch, and the channel order follows the sink format.
TEST_P(HolovizOpCudaSinkGradient, PreservesLayerPixels) {
  const int cuda_device = holoviz_cuda_device();
  if (cuda_device < 0) {
    GTEST_SKIP() << "[holoviz-skip] no CUDA device";
  }
  g_stats.reset();
  const viz::ImageFormat format = GetParam();

  holoscan::Graph graph{"holoviz-cuda-sink-gradient"};
  const auto source = graph.op<GradientSource>("gradient");
  const auto viz_op = graph.op<viz::HolovizOp>("holoviz",
                                               rgba_layer("gradient"),
                                               cuda_sink(format),
                                               viz::LayerTrigger{},
                                               std::chrono::nanoseconds{500ms});
  const auto checker = graph.op<FrameChecker>("checker", Expectation::kGradient, format);
  graph.add_flow(source->frame, viz_op->image_layer_input(0), kMailbox);
  graph.add_flow(viz_op->output(), checker->frames, kMailbox);
  graph.partition("render").add(viz_op);
  graph.auto_partition_remaining();

  holoscan::CompileOptions options;
  bind_output_device(options, "holoviz", cuda_device);
  run_and_check(graph, std::move(options));
}

INSTANTIATE_TEST_SUITE_P(Formats, HolovizOpCudaSinkGradient,
                         ::testing::Values(viz::ImageFormat::kR8G8B8A8Unorm,
                                           viz::ImageFormat::kB8G8R8A8Unorm),
                         [](const ::testing::TestParamInfo<viz::ImageFormat>& info) {
                           return info.param == viz::ImageFormat::kB8G8R8A8Unorm ? "Bgra8"
                                                                                 : "Rgba8";
                         });

// Two chained HolovizOps: the first captures the gradient, the second consumes that ImageT's
// CUDA tensor as its own image layer (the direct device-pointer path, no copy; the wire RGBA8
// encoding matches the layer's declared format) and captures it again. The
// checker sees the gradient unchanged after two render passes.
TEST(HolovizOpCudaSink, ChainsIntoASecondHolovizOp) {
  const int cuda_device = holoviz_cuda_device();
  if (cuda_device < 0) {
    GTEST_SKIP() << "[holoviz-skip] no CUDA device";
  }
  g_stats.reset();

  holoscan::Graph graph{"holoviz-cuda-sink-chain"};
  const auto source = graph.op<GradientSource>("gradient");
  const auto first = graph.op<viz::HolovizOp>("first",
                                              rgba_layer("gradient"),
                                              cuda_sink(),
                                              viz::LayerTrigger{},
                                              std::chrono::nanoseconds{500ms});
  const auto second = graph.op<viz::HolovizOp>("second",
                                               rgba_layer("captured"),
                                               cuda_sink(),
                                               viz::LayerTrigger{},
                                               std::chrono::nanoseconds{500ms});
  const auto checker = graph.op<FrameChecker>("checker", Expectation::kGradient);
  graph.add_flow(source->frame, first->image_layer_input(0), kMailbox);
  graph.add_flow(first->output(), second->image_layer_input(0), kMailbox);
  graph.add_flow(second->output(), checker->frames, kMailbox);
  graph.partition("first").add(first);
  graph.partition("second").add(second);
  graph.auto_partition_remaining();

  holoscan::CompileOptions options;
  bind_output_device(options, "first", cuda_device);
  bind_output_device(options, "second", cuda_device);
  run_and_check(graph, std::move(options));
}

// Same-stream pool recycling: with one shared compute stream for the whole process, the pool
// retires a released frame logically (no GPU event) and hands the allocation back to HolovizOp
// while the consumer's asynchronous read of it is still queued on that stream (the delay is long
// enough to cycle the whole pool). Every delayed read must still see its own (changing) frame.
// End-to-end coverage of the policy path; the backend-level guarantee that the resolve waits for
// the writer stream is pinned by CudaPresentBackendTest.ResolveWaitsForTheWriterStream (here, a
// layer's upload on the activation stream orders the render after the queued read as well).
TEST(HolovizOpCudaSink, SharedStreamConsumerReadsItsOwnFrame) {
  const int cuda_device = holoviz_cuda_device();
  if (cuda_device < 0) {
    GTEST_SKIP() << "[holoviz-skip] no CUDA device";
  }
  g_stats.reset();

  holoscan::Graph graph{"holoviz-cuda-sink-shared-stream"};
  const auto source = graph.op<GradientSource>("gradient", /*animate=*/true);
  const auto viz_op = graph.op<viz::HolovizOp>("holoviz",
                                               rgba_layer("gradient"),
                                               cuda_sink(),
                                               viz::LayerTrigger{},
                                               std::chrono::nanoseconds{500ms});
  const auto checker = graph.op<DelayedReadChecker>("checker");
  graph.add_flow(source->frame, viz_op->image_layer_input(0), kMailbox);
  graph.add_flow(viz_op->output(), checker->frames, kMailbox);
  graph.partition("render").add(viz_op);
  graph.auto_partition_remaining();

  holoscan::CompileOptions options;
  options.cuda_execution.compute_stream_policy =
      holoscan::CudaComputeStreamPolicy::kSharedProcessDevice;
  bind_output_device(options, "holoviz", cuda_device);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();

  holoscan::RunSession session = holoscan::run_async(plan);
  run_until_done(session, 20s);
  expect_clean_stop(session);
  EXPECT_GE(g_stats.frames.load(), 12U);
  EXPECT_EQ(g_stats.shape_mismatches.load(), 0U);
  EXPECT_EQ(g_stats.pixel_mismatches.load(), 0U);
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return viz::test_util::skip_aware_exit_code(RUN_ALL_TESTS());
}
