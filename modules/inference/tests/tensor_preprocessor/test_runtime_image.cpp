/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Runtime tests for TensorPreprocessorOp::compute() via the schema::ImageT input path:
//   ImageFeeder -> TensorPreprocessorOp (input_kind=image) -> TensorRecorder
//
// Exercises NV12 and I420 ImageT payloads with BT.601 and BT.709 colorimetry, plus a
// reference-value assertion that a uniform-gray input becomes near-constant RGB(128,128,128).
// Requires a CUDA device.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
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
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace holoscan::ops::test_image {
namespace {

using namespace std::chrono_literals;

constexpr std::chrono::nanoseconds kFeedPeriod = 1ms;
constexpr std::chrono::seconds kRunTimeout = 30s;
constexpr std::size_t kTicks = 3U;
constexpr DLDataType kUInt8{kDLUInt, 8U, 1U};

// Fixture describing one synthetic ImageT frame.
struct ImageFrame {
  schema::ImageEncoding encoding{schema::ImageEncoding_CUSTOM};
  schema::ColorSpace color_space{schema::ColorSpace_UNSPECIFIED};
  int32_t width{};
  int32_t height{};
  std::vector<std::uint8_t> host_bytes;                 // Raw byte layout of the frame.
  std::vector<schema::ImagePlaneLayout> plane_layouts;  // Empty for single-plane encodings.
};

// Build a uniform NV12 frame: Y plane filled with `y_value`, CbCr plane filled with `chroma_value`.
[[nodiscard]] ImageFrame uniform_nv12(int32_t width, int32_t height, std::uint8_t y_value,
                                      std::uint8_t chroma_value, schema::ColorSpace cs) {
  const std::size_t y_bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  const std::size_t uv_bytes = y_bytes / 2U;  // 4:2:0 subsampling.
  ImageFrame frame;
  frame.encoding = schema::ImageEncoding_NV12;
  frame.color_space = cs;
  frame.width = width;
  frame.height = height;
  frame.host_bytes.resize(y_bytes + uv_bytes);
  std::fill_n(frame.host_bytes.begin(), y_bytes, y_value);
  std::fill_n(frame.host_bytes.begin() + y_bytes, uv_bytes, chroma_value);
  frame.plane_layouts.push_back(
      schema::ImagePlaneLayout{0U, static_cast<std::uint32_t>(width), y_bytes});
  frame.plane_layouts.push_back(
      schema::ImagePlaneLayout{y_bytes, static_cast<std::uint32_t>(width), uv_bytes});
  return frame;
}

// Build a uniform I420 frame: Y filled with `y_value`, Cb filled with `cb`, Cr filled with `cr`.
[[nodiscard]] ImageFrame uniform_i420(int32_t width, int32_t height, std::uint8_t y_value,
                                      std::uint8_t cb, std::uint8_t cr) {
  const std::size_t y_bytes = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  const std::size_t chroma_plane_bytes = y_bytes / 4U;
  ImageFrame frame;
  frame.encoding = schema::ImageEncoding_I420;
  frame.color_space = schema::ColorSpace_REC601;
  frame.width = width;
  frame.height = height;
  frame.host_bytes.resize(y_bytes + 2U * chroma_plane_bytes);
  auto* base = frame.host_bytes.data();
  std::fill_n(base, y_bytes, y_value);
  std::fill_n(base + y_bytes, chroma_plane_bytes, cb);
  std::fill_n(base + y_bytes + chroma_plane_bytes, chroma_plane_bytes, cr);
  frame.plane_layouts.push_back(
      schema::ImagePlaneLayout{0U, static_cast<std::uint32_t>(width), y_bytes});
  frame.plane_layouts.push_back(
      schema::ImagePlaneLayout{y_bytes, static_cast<std::uint32_t>(width / 2), chroma_plane_bytes});
  frame.plane_layouts.push_back(schema::ImagePlaneLayout{
      y_bytes + chroma_plane_bytes, static_cast<std::uint32_t>(width / 2), chroma_plane_bytes});
  return frame;
}

// Emits `kTicks` copies of a fixed ImageT frame onto its Output<schema::ImageT> port.
class ImageFeeder final : public holoscan::Operator<> {
 public:
  ImageFeeder(std::string port_name, ImageFrame frame, std::size_t ticks)
      : port_name_(std::move(port_name)), frame_(std::move(frame)), ticks_(ticks) {}

  void setup(holoscan::OperatorSpec& spec) override {
    const std::size_t max_bytes = frame_.host_bytes.size();
    spec.output(output_, port_name_)
        .tensor_field(holoscan::schema::ImageTensorFields::data)
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation =
                holoscan::TensorRepresentation{
                    .memory_kind = holoscan::MemoryKind::kCudaDevice,
                    .dtype = kUInt8,
                },
            .bounds = holoscan::TensorBounds{.max_rank = 3U, .max_byte_span = max_bytes},
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
    // RGB8 emits rank-3 [H, W, 3] so a downstream `Input<Tensor>` projection can decode
    // the tensor without needing ImageT metadata for geometry. NV12/I420 have no such rank-3
    // layout and emit a flat byte buffer whose planes are found via ImageT.plane_layouts.
    std::vector<int64_t> shape;
    if (frame_.encoding == holoscan::schema::ImageEncoding_RGB8) {
      shape = {frame_.height, frame_.width, 3};
    } else {
      shape = {static_cast<int64_t>(frame_.host_bytes.size())};
    }
    auto loan =
        output_.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    if (cudaMemcpyAsync(writer->data(),
                        frame_.host_bytes.data(),
                        frame_.host_bytes.size(),
                        cudaMemcpyHostToDevice,
                        stream) != cudaSuccess) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }

    holoscan::schema::ImageT descriptor;
    descriptor.width = frame_.width;
    descriptor.height = frame_.height;
    descriptor.encoding = frame_.encoding;
    descriptor.color_space = frame_.color_space;
    descriptor.plane_layouts = frame_.plane_layouts;
    ++sent_;
    return output_.emit_tensor(std::move(*loan), descriptor);
  }

  [[nodiscard]] holoscan::Output<holoscan::schema::ImageT>* output_port() const {
    return const_cast<holoscan::Output<holoscan::schema::ImageT>*>(&output_);
  }

 private:
  std::string port_name_;
  ImageFrame frame_;
  std::size_t ticks_{};
  std::size_t sent_{};
  holoscan::Output<holoscan::schema::ImageT> output_;
};

struct Observation {
  static void reset() {
    const std::lock_guard lock(mutex);
    ticks = 0U;
    last_shape.clear();
    last_dtype = DLDataType{};
    last_host_bytes.clear();
  }

  static void record(const std::vector<int64_t>& shape, const DLDataType& dtype,
                     std::vector<std::uint8_t> bytes) {
    {
      const std::lock_guard lock(mutex);
      ++ticks;
      last_shape = shape;
      last_dtype = dtype;
      last_host_bytes = std::move(bytes);
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
  static inline std::vector<std::uint8_t> last_host_bytes;
};

class Recorder final : public holoscan::Operator<> {
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
      holoscan::ExecutionContext& context) override {
    auto sample = input.receive_data();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    const holoscan::Tensor& tensor = *sample;
    // Copy device bytes to host so reference-value tests can inspect them.
    std::vector<std::uint8_t> host_bytes;
    std::size_t byte_size = 1U;
    for (const int64_t dim : tensor.shape()) {
      byte_size *= static_cast<std::size_t>(dim);
    }
    byte_size *= (tensor.dtype().bits + 7U) / 8U;
    host_bytes.resize(byte_size);
    const cudaStream_t stream = context.cuda_stream().get();
    if (cudaMemcpyAsync(
            host_bytes.data(), tensor.data(), byte_size, cudaMemcpyDeviceToHost, stream) !=
            cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
    }
    Observation::record(tensor.shape(), tensor.dtype(), std::move(host_bytes));
    return {};
  }

  holoscan::Input<holoscan::Tensor> input;
};

// Runs ImageFeeder -> TensorPreprocessorOp -> Recorder. When `params.input_kind == "tensor"`,
// the connection uses projection: the ImageT emitter's Tensor field is narrowed onto the
// operator's `.accepts_projected_tensor()`-annotated `Input<Tensor>` port. Otherwise the
// `Input<schema::ImageT>` port is used directly.
[[nodiscard]] bool run_image_pipeline(TensorPreprocessorOpParams params, ImageFrame frame,
                                      std::string& err_message) {
  Observation::reset();
  const bool use_projection = params.input_kind == "tensor";
  try {
    holoscan::Graph graph{"tensor-preprocessor-image-test"};
    const auto pre = graph.op<TensorPreprocessorOp>("pre", params);
    const auto feeder = graph.op<ImageFeeder>("feeder", params.input_port_name, frame, kTicks);
    const auto sink = graph.op<Recorder>("sink");

    if (use_projection) {
      // Projection: the emitter's Output<schema::ImageT>'s Tensor field is narrowed onto the
      // operator's `.accepts_projected_tensor()`-annotated `Input<Tensor>` port.
      graph.add_flow(*feeder->output_port(),
                     *pre->input_port(),
                     holoscan::ConnectionOptions{.queue_depth = 4U});
    } else {
      graph.add_flow(*feeder->output_port(),
                     *pre->image_input_port(),
                     holoscan::ConnectionOptions{.queue_depth = 4U});
    }
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
    const bool complete = Observation::wait_for(kTicks);
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
// NV12 via ImageT, BT.601 (default NPP dispatcher)
// ---------------------------------------------------------------------------------------------

TEST(TensorPreprocessorImageRuntime, Nv12_Bt601_ProducesExpectedShape) {
  TensorPreprocessorOpParams params;
  params.input_kind = "image";
  params.output_width = 128;
  params.output_height = 128;
  params.output_pix_fmt = "rgb_u8";
  params.output_layout = "nhwc";
  params.prepend_batch_dim = false;
  params.yuv_matrix = "bt601";

  std::string err;
  ASSERT_TRUE(run_image_pipeline(
      params, uniform_nv12(320, 240, 128U, 128U, schema::ColorSpace_REC601), err))
      << err;

  const std::lock_guard lock(Observation::mutex);
  EXPECT_EQ(Observation::ticks, kTicks);
  const std::vector<int64_t> expected_shape{128, 128, 3};
  EXPECT_EQ(Observation::last_shape, expected_shape);
  EXPECT_EQ(Observation::last_dtype.code, kDLUInt);
  EXPECT_EQ(Observation::last_dtype.bits, 8U);
}

// ---------------------------------------------------------------------------------------------
// NV12 via ImageT, BT.709 (nppiNV12ToRGB_709CSC_... variant)
// ---------------------------------------------------------------------------------------------

TEST(TensorPreprocessorImageRuntime, Nv12_Bt709_ProducesExpectedShape) {
  TensorPreprocessorOpParams params;
  params.input_kind = "image";
  params.output_width = 128;
  params.output_height = 128;
  params.output_pix_fmt = "rgb_u8";
  params.output_layout = "nhwc";
  params.prepend_batch_dim = false;
  // yuv_matrix is not set explicitly; the ImageT color_space REC709 field selects BT.709 for
  // this frame regardless of the operator's yaml default.

  std::string err;
  ASSERT_TRUE(run_image_pipeline(
      params, uniform_nv12(320, 240, 128U, 128U, schema::ColorSpace_REC709), err))
      << err;

  const std::lock_guard lock(Observation::mutex);
  EXPECT_EQ(Observation::ticks, kTicks);
  const std::vector<int64_t> expected_shape{128, 128, 3};
  EXPECT_EQ(Observation::last_shape, expected_shape);
}

// ---------------------------------------------------------------------------------------------
// I420 via ImageT, BT.601
// ---------------------------------------------------------------------------------------------

TEST(TensorPreprocessorImageRuntime, I420_Bt601_ProducesExpectedShape) {
  TensorPreprocessorOpParams params;
  params.input_kind = "image";
  params.output_width = 128;
  params.output_height = 128;
  params.output_pix_fmt = "rgb_u8";
  params.output_layout = "nhwc";
  params.prepend_batch_dim = false;

  std::string err;
  ASSERT_TRUE(run_image_pipeline(params, uniform_i420(320, 240, 128U, 128U, 128U), err)) << err;

  const std::lock_guard lock(Observation::mutex);
  EXPECT_EQ(Observation::ticks, kTicks);
  const std::vector<int64_t> expected_shape{128, 128, 3};
  EXPECT_EQ(Observation::last_shape, expected_shape);
}

// ---------------------------------------------------------------------------------------------
// Reference value: uniform Y=128, Cb=Cr=128 -> RGB(128, 128, 128) within tolerance.
// The BT.601 / BT.709 CSC both map neutral chroma at mid-luma to gray, so the assertion is
// matrix-agnostic. Tolerance covers rounding + bilinear-resize interpolation noise.
// ---------------------------------------------------------------------------------------------

TEST(TensorPreprocessorImageRuntime, Nv12_UniformGray_ReferenceValues) {
  TensorPreprocessorOpParams params;
  params.input_kind = "image";
  params.output_width = 32;
  params.output_height = 32;
  params.output_pix_fmt = "rgb_u8";
  params.output_layout = "nhwc";
  params.prepend_batch_dim = false;

  std::string err;
  ASSERT_TRUE(
      run_image_pipeline(params, uniform_nv12(64, 64, 128U, 128U, schema::ColorSpace_REC601), err))
      << err;

  const std::lock_guard lock(Observation::mutex);
  ASSERT_FALSE(Observation::last_host_bytes.empty());
  // Sample the center pixel to skip any interpolation edges. NHWC layout with 3 bytes/pixel.
  constexpr std::size_t center_row = 16;
  constexpr std::size_t center_col = 16;
  constexpr std::size_t idx = (center_row * 32U + center_col) * 3U;
  const std::uint8_t r = Observation::last_host_bytes[idx];
  const std::uint8_t g = Observation::last_host_bytes[idx + 1U];
  const std::uint8_t b = Observation::last_host_bytes[idx + 2U];
  // Uniform gray input should decode to gray output within 4 LSB.
  EXPECT_NEAR(static_cast<int>(r), 128, 4);
  EXPECT_NEAR(static_cast<int>(g), 128, 4);
  EXPECT_NEAR(static_cast<int>(b), 128, 4);
}

// ---------------------------------------------------------------------------------------------
// Projection: an upstream Output<schema::ImageT> connects to Input<Tensor>.accepts_projected_
// tensor() and the operator processes the projected Tensor field as RGB24.
// ---------------------------------------------------------------------------------------------

namespace {

// Build a uniform interleaved-RGB8 frame (rank-3 [H, W, 3]-shaped bytes). Encoding is RGB8
// so the projected Tensor is interpreted as HWC RGB.
[[nodiscard]] ImageFrame uniform_rgb8(int32_t width, int32_t height, std::uint8_t r, std::uint8_t g,
                                      std::uint8_t b) {
  ImageFrame frame;
  frame.encoding = schema::ImageEncoding_RGB8;
  frame.color_space = schema::ColorSpace_SRGB;
  frame.width = width;
  frame.height = height;
  const std::size_t pixels = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  frame.host_bytes.resize(pixels * 3U);
  for (std::size_t i = 0; i < pixels; ++i) {
    frame.host_bytes[i * 3U + 0U] = r;
    frame.host_bytes[i * 3U + 1U] = g;
    frame.host_bytes[i * 3U + 2U] = b;
  }
  return frame;
}

}  // namespace

TEST(TensorPreprocessorImageRuntime, Rgb8_Projection_ProducesExpectedShape) {
  TensorPreprocessorOpParams params;
  params.input_kind = "tensor";  // Selects the projection path with .accepts_projected_tensor().
  params.input_pix_fmt = "rgb24";
  params.output_width = 64;
  params.output_height = 64;
  params.output_pix_fmt = "rgb_u8";
  params.output_layout = "nhwc";
  params.prepend_batch_dim = false;

  std::string err;
  ASSERT_TRUE(run_image_pipeline(params, uniform_rgb8(96, 96, 200U, 100U, 50U), err)) << err;

  const std::lock_guard lock(Observation::mutex);
  EXPECT_EQ(Observation::ticks, kTicks);
  const std::vector<int64_t> expected_shape{64, 64, 3};
  EXPECT_EQ(Observation::last_shape, expected_shape);
  ASSERT_FALSE(Observation::last_host_bytes.empty());
  // Center pixel should retain the input color within a small tolerance (bilinear noise).
  constexpr std::size_t idx = (32U * 64U + 32U) * 3U;
  EXPECT_NEAR(static_cast<int>(Observation::last_host_bytes[idx + 0U]), 200, 4);
  EXPECT_NEAR(static_cast<int>(Observation::last_host_bytes[idx + 1U]), 100, 4);
  EXPECT_NEAR(static_cast<int>(Observation::last_host_bytes[idx + 2U]), 50, 4);
}

}  // namespace holoscan::ops::test_image
