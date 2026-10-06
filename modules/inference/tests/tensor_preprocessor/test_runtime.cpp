/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Runtime tests for TensorPreprocessorOp::compute() using a real Fragment: builds a
// Feeder -> TensorPreprocessorOp -> Sink graph, runs a few activations, and asserts the sink
// received a tensor with the expected shape/dtype. Requires a CUDA device.

#include <gtest/gtest.h>

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include <dlpack/dlpack.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/operators/tensor_preprocessor/tensor_preprocessor.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace holoscan::ops {
namespace {

using namespace std::chrono_literals;

constexpr std::chrono::nanoseconds kFeedPeriod = 500us;
constexpr std::chrono::seconds kRunTimeout = 30s;
constexpr std::size_t kTicks = 4U;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

// Emits a single interleaved-HWC uint8 tensor per tick, filled with `fill_value` on device.
class Uint8HwcFeeder final : public holoscan::Operator<> {
 public:
  Uint8HwcFeeder(std::string port_name, std::vector<int64_t> shape, std::uint8_t fill_value,
                 std::size_t ticks)
      : port_name_(std::move(port_name)),
        shape_(std::move(shape)),
        fill_value_(fill_value),
        ticks_(ticks) {}

  void setup(holoscan::OperatorSpec& spec) override {
    const std::uint32_t rank = static_cast<std::uint32_t>(shape_.size());
    std::size_t elements = 1U;
    for (const int64_t dim : shape_) {
      elements *= static_cast<std::size_t>(dim);
    }
    spec.output(output_, port_name_)
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = holoscan::MemoryKind::kCudaDevice,
                    .dtype = kUInt8,
                    .rank = rank,
                },
            .bounds = holoscan::TensorBounds{.max_rank = rank, .max_byte_span = elements},
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = kFeedPeriod});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (sent_ >= ticks_) {
      return {};
    }
    const cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kNotSupported});
    }
    auto loan =
        output_.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape_, .dtype = kUInt8});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    if (cudaMemsetAsync(writer->data(), fill_value_, writer->byte_size(), stream) != cudaSuccess) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    // Stamp a synthetic frame_id (kSyntheticFrameIdBase + tick) so the test can assert the
    // preprocessor propagates it through, along with the capture_time from the clock.
    const std::uint64_t frame_id = kSyntheticFrameIdBase + sent_;
    ++sent_;
    return output_.emit(
        std::move(*loan),
        holoscan::EmitOptions{.capture_time = context.activation_time(), .frame_id = frame_id});
  }

  static constexpr std::uint64_t kSyntheticFrameIdBase = 1000U;

  [[nodiscard]] holoscan::Output<holoscan::Tensor>* output_port() const {
    return const_cast<holoscan::Output<holoscan::Tensor>*>(&output_);
  }

 private:
  std::string port_name_;
  std::vector<int64_t> shape_;
  std::uint8_t fill_value_{};
  std::size_t ticks_{};
  std::size_t sent_{};
  holoscan::Output<holoscan::Tensor> output_;
};

// Observation captured by the sink: shape + dtype of the last received tensor, its capture-time
// clock domain, and the last frame_id (so tests can assert metadata propagated through the
// preprocessor).
struct SinkObservation {
  static void reset() {
    const std::lock_guard lock(mutex);
    ticks = 0U;
    last_shape.clear();
    last_dtype = DLDataType{};
    last_source_clock_id = holoscan::ClockId{};
    last_capture_timestamp_ns = 0;
    last_frame_id = 0U;
  }

  static void record(const std::vector<int64_t>& shape, const DLDataType& dtype,
                     const holoscan::SampleMetadata& metadata) {
    {
      const std::lock_guard lock(mutex);
      ++ticks;
      last_shape = shape;
      last_dtype = dtype;
      last_source_clock_id = metadata.source_clock_id;
      last_capture_timestamp_ns = metadata.capture_timestamp_ns;
      last_frame_id = metadata.frame_id;
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for(std::size_t expected) {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, kRunTimeout, [expected] { return ticks >= expected; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::size_t ticks{};
  static inline std::vector<int64_t> last_shape;
  static inline DLDataType last_dtype{};
  static inline holoscan::ClockId last_source_clock_id{};
  static inline std::int64_t last_capture_timestamp_ns{};
  static inline std::uint64_t last_frame_id{};
};

class TensorRecorder final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(4U)
        .expects_tensor(holoscan::TensorInputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = holoscan::MemoryKind::kCudaDevice,
                },
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    const holoscan::Tensor& tensor = sample->data;
    SinkObservation::record(tensor.shape(), tensor.dtype(), sample->metadata);
    return {};
  }

  holoscan::Input<holoscan::Tensor> input;
};

// Runs Feeder -> TensorPreprocessorOp -> Sink with the given params + input shape/fill and returns
// after `kTicks` sink activations. Populates `err_message` on failure; returns true on success.
[[nodiscard]] bool run_pipeline(TensorPreprocessorOpParams params,
                                const std::vector<int64_t>& in_shape, std::uint8_t fill_value,
                                std::string& err_message) {
  SinkObservation::reset();
  try {
    holoscan::Graph graph{"tensor-preprocessor-runtime-test"};

    const auto pre = graph.op<TensorPreprocessorOp>("pre", params);
    const auto feeder =
        graph.op<Uint8HwcFeeder>("feeder", params.input_port_name, in_shape, fill_value, kTicks);
    const auto sink = graph.op<TensorRecorder>("sink");

    graph.add_flow(
        *feeder->output_port(), *pre->input_port(), holoscan::ConnectionOptions{.queue_depth = 4U});
    graph.add_flow(
        *pre->output_port(), sink->input, holoscan::ConnectionOptions{.queue_depth = 4U});

    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

    holoscan::CompileOptions options;
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "feeder",
        .output_port = params.input_port_name,
        .device = holoscan::DeviceId{0},
    });
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = "pre",
        .output_port = params.output_port_name,
        .device = holoscan::DeviceId{0},
    });

    const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
    if (!plan.ok()) {
      err_message = "compile failed: " + std::string(plan.json());
      return false;
    }

    holoscan::RunSession session = holoscan::run_async(plan);
    const bool complete = SinkObservation::wait_for(kTicks);
    session.request_stop();
    session.wait();
    if (!complete) {
      err_message = "sink did not observe expected ticks within timeout";
      return false;
    }
    return true;
  } catch (const std::exception& exception) {
    err_message = std::string("exception: ") + exception.what();
    return false;
  }
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// RGB24 -> NCHW fp32 (default normalization)
// ---------------------------------------------------------------------------------------------

TEST(TensorPreprocessorRuntime, Rgb24_ToNchwFp32_ProducesExpectedShape) {
  TensorPreprocessorOpParams params;
  params.input_port_name = "in";
  params.output_port_name = "out";
  params.input_pix_fmt = "rgb24";
  params.output_width = 128;
  params.output_height = 128;
  params.output_pix_fmt = "rgb_fp32";
  params.output_layout = "nchw";
  params.prepend_batch_dim = true;

  std::string err;
  ASSERT_TRUE(run_pipeline(params, {240, 320, 3}, 128U, err)) << err;

  const std::lock_guard lock(SinkObservation::mutex);
  EXPECT_EQ(SinkObservation::ticks, kTicks);
  const std::vector<int64_t> expected{1, 3, 128, 128};
  EXPECT_EQ(SinkObservation::last_shape, expected);
  EXPECT_EQ(SinkObservation::last_dtype.code, kDLFloat);
  EXPECT_EQ(SinkObservation::last_dtype.bits, 32U);
}

// ---------------------------------------------------------------------------------------------
// RGB24 -> NHWC uint8 (no dtype conversion, no plane split)
// ---------------------------------------------------------------------------------------------

TEST(TensorPreprocessorRuntime, Rgb24_ToNhwcU8_ProducesExpectedShape) {
  TensorPreprocessorOpParams params;
  params.output_width = 64;
  params.output_height = 64;
  params.output_pix_fmt = "rgb_u8";
  params.output_layout = "nhwc";
  params.prepend_batch_dim = false;

  std::string err;
  ASSERT_TRUE(run_pipeline(params, {96, 96, 3}, 200U, err)) << err;

  const std::lock_guard lock(SinkObservation::mutex);
  EXPECT_EQ(SinkObservation::ticks, kTicks);
  const std::vector<int64_t> expected{64, 64, 3};
  EXPECT_EQ(SinkObservation::last_shape, expected);
  EXPECT_EQ(SinkObservation::last_dtype.code, kDLUInt);
  EXPECT_EQ(SinkObservation::last_dtype.bits, 8U);
}

// ---------------------------------------------------------------------------------------------
// YUYV -> RGB -> NCHW fp32 (exercises the YUV color-conversion path)
// ---------------------------------------------------------------------------------------------

TEST(TensorPreprocessorRuntime, Yuyv_ToNchwFp32_ProducesExpectedShape) {
  TensorPreprocessorOpParams params;
  params.input_pix_fmt = "yuyv";
  params.output_width = 128;
  params.output_height = 128;
  params.output_pix_fmt = "rgb_fp32";
  params.output_layout = "nchw";
  params.prepend_batch_dim = true;

  std::string err;
  // YUYV: 2 channels, even width. Fill 128 -> Y=128, Cb/Cr=128 -> mid-gray RGB.
  ASSERT_TRUE(run_pipeline(params, {240, 320, 2}, 128U, err)) << err;

  const std::lock_guard lock(SinkObservation::mutex);
  EXPECT_EQ(SinkObservation::ticks, kTicks);
  const std::vector<int64_t> expected{1, 3, 128, 128};
  EXPECT_EQ(SinkObservation::last_shape, expected);
  EXPECT_EQ(SinkObservation::last_dtype.code, kDLFloat);
}

// ---------------------------------------------------------------------------------------------
// V4L2-style rank-2 YUYV: input shape [H, 2 * W] uint8 (each row is byte-packed).
// ---------------------------------------------------------------------------------------------

TEST(TensorPreprocessorRuntime, Yuyv_Rank2_V4l2Layout_ProducesExpectedShape) {
  TensorPreprocessorOpParams params;
  params.input_pix_fmt = "yuyv";
  params.output_width = 64;
  params.output_height = 64;
  params.output_pix_fmt = "rgb_u8";
  params.output_layout = "nhwc";
  params.prepend_batch_dim = false;

  std::string err;
  // 240 rows × 640 bytes/row = W_bytes/2 = 320 pixels wide.
  ASSERT_TRUE(run_pipeline(params, {240, 640}, 128U, err)) << err;

  const std::lock_guard lock(SinkObservation::mutex);
  EXPECT_EQ(SinkObservation::ticks, kTicks);
  const std::vector<int64_t> expected{64, 64, 3};
  EXPECT_EQ(SinkObservation::last_shape, expected);
}

// ---------------------------------------------------------------------------------------------
// Flat bare-Tensor NV12: rank-1 byte buffer + input_planes-carried Y/CbCr geometry
// (mirrors what an Argus-style capture operator publishes for a NV12 frame).
// ---------------------------------------------------------------------------------------------

TEST(TensorPreprocessorRuntime, Nv12_Flat_InputPlanes_ProducesExpectedShape) {
  constexpr int32_t kW = 320;
  constexpr int32_t kH = 240;
  constexpr std::uint64_t kYBytes = static_cast<std::uint64_t>(kW) * kH;
  constexpr std::uint64_t kUvBytes = kYBytes / 2U;

  TensorPreprocessorOpParams params;
  params.input_pix_fmt = "nv12";
  params.input_width = kW;
  params.input_height = kH;
  params.input_planes = {
      {.offset_bytes = 0, .row_stride_bytes = kW},
      {.offset_bytes = kYBytes, .row_stride_bytes = kW},
  };
  params.output_width = 128;
  params.output_height = 128;
  params.output_pix_fmt = "rgb_u8";
  params.output_layout = "nhwc";
  params.prepend_batch_dim = false;

  std::string err;
  // Uniform fill 128 = Y=128, Cb=Cr=128 -> neutral RGB(≈128, 128, 128).
  ASSERT_TRUE(run_pipeline(params, {static_cast<int64_t>(kYBytes + kUvBytes)}, 128U, err)) << err;

  const std::lock_guard lock(SinkObservation::mutex);
  EXPECT_EQ(SinkObservation::ticks, kTicks);
  const std::vector<int64_t> expected{128, 128, 3};
  EXPECT_EQ(SinkObservation::last_shape, expected);
}

// ---------------------------------------------------------------------------------------------
// Metadata propagation: capture_time (clock domain) and frame_id survive through the operator
// ---------------------------------------------------------------------------------------------

TEST(TensorPreprocessorRuntime, PropagatesCaptureTimeAndFrameIdFromUpstream) {
  TensorPreprocessorOpParams params;
  params.output_width = 128;
  params.output_height = 128;
  params.output_pix_fmt = "rgb_fp32";
  params.output_layout = "nchw";
  params.prepend_batch_dim = true;

  std::string err;
  ASSERT_TRUE(run_pipeline(params, {240, 320, 3}, 128U, err)) << err;

  const std::lock_guard lock(SinkObservation::mutex);
  ASSERT_EQ(SinkObservation::ticks, kTicks);
  // Feeder emits capture_time = context.activation_time(), which carries a valid runtime
  // clock domain. If the preprocessor forwards the incoming metadata (instead of stamping
  // its own activation time), the sink observes a nonzero clock_id and timestamp.
  EXPECT_TRUE(SinkObservation::last_source_clock_id.valid());
  EXPECT_GT(SinkObservation::last_capture_timestamp_ns, 0);
  // Feeder stamps frame_id = kSyntheticFrameIdBase + tick; the last-emitted tick is kTicks-1.
  EXPECT_EQ(SinkObservation::last_frame_id, Uint8HwcFeeder::kSyntheticFrameIdBase + (kTicks - 1U));
}

}  // namespace holoscan::ops
