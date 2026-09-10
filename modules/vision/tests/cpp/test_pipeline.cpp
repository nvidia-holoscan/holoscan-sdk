// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <span>
#include <utility>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <holoscan/core/activation.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/vision/convert_to_op.hpp>
#include <holoscan/vision/copy_make_border_op.hpp>
#include <holoscan/vision/custom_crop_op.hpp>
#include <holoscan/vision/non_maximum_suppression_op.hpp>
#include <holoscan/vision/normalize_op.hpp>
#include <holoscan/vision/reformat_op.hpp>
#include <holoscan/vision/remap_op.hpp>
#include <holoscan/vision/resize_op.hpp>

namespace holoscan::vision {
namespace {

using namespace std::chrono_literals;

constexpr DLDataType kU8{kDLUInt, 8U, 1U};
constexpr std::array<std::int64_t, 3U> kInputShape{4, 4, 1};
constexpr std::array<std::uint8_t, 16U> kInputValues{
    0U,
    1U,
    2U,
    3U,
    4U,
    5U,
    6U,
    7U,
    8U,
    9U,
    10U,
    11U,
    12U,
    13U,
    14U,
    15U,
};
constexpr std::array<std::int64_t, 3U> kTwoChannelInputShape{4, 4, 2};
constexpr std::array<std::uint8_t, 32U> kTwoChannelInputValues{
    0U, 100U, 1U, 101U, 2U,  102U, 3U,  103U, 4U,  104U, 5U,  105U, 6U,  106U, 7U,  107U,
    8U, 108U, 9U, 109U, 10U, 110U, 11U, 111U, 12U, 112U, 13U, 113U, 14U, 114U, 15U, 115U,
};
constexpr std::uint64_t kFrameId = 42U;
constexpr TraceId kTraceId = 314U;
constexpr TensorRepresentation kInputTensorRepresentation{
    .memory_kind = MemoryKind::kCudaDevice,
    .dtype = kU8,
    .rank = 3U,
};
constexpr TensorRepresentation kCudaTensorRepresentation{
    .memory_kind = MemoryKind::kCudaDevice,
};
constexpr TensorRepresentation kColorImageTensorRepresentation{
    .memory_kind = MemoryKind::kCudaDevice,
    .dtype = kU8,
    .rank = 3U,
};

enum class ImageFixture {
  kSingleChannel,
  kTwoChannel,
};

struct PipelineObservation {
  static void reset() {
    const std::lock_guard lock(mutex);
    values.fill(0U);
    shape.fill(0);
    dtype = {};
    metadata = {};
    failure.reset();
    complete = false;
  }

  static void fail(ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      failure = code;
    }
    ready.notify_all();
  }

  static void record(std::span<const std::uint8_t> result,
                     std::span<const std::int64_t, 3U> result_shape, DLDataType result_dtype,
                     SampleMetadata result_metadata) {
    {
      const std::lock_guard lock(mutex);
      std::ranges::copy(result, values.begin());
      std::ranges::copy(result_shape, shape.begin());
      dtype = result_dtype;
      metadata = result_metadata;
      complete = true;
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 5s, [] { return complete || failure.has_value(); });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::array<std::uint8_t, 8U> values{};
  static inline std::array<std::int64_t, 3U> shape{};
  static inline DLDataType dtype{};
  static inline SampleMetadata metadata{};
  static inline std::optional<ErrorCode> failure;
  static inline bool complete{};
};

[[nodiscard]] expected<void, Error> pipeline_failure(ErrorCode code) {
  PipelineObservation::fail(code);
  return make_unexpected(Error{code});
}

class ImageSource final : public Operator<> {
 public:
  explicit ImageSource(ImageFixture fixture = ImageFixture::kSingleChannel) : fixture_(fixture) {}

  void setup(OperatorSpec& spec) override {
    const std::size_t maximum_byte_span =
        fixture_ == ImageFixture::kTwoChannel ? kTwoChannelInputValues.size() : kInputValues.size();
    spec.output(output, "output")
        .produces_tensor(TensorOutputSpec{
            .representation = kInputTensorRepresentation,
            .bounds =
                TensorBounds{
                    .max_rank = 3U,
                    .max_byte_span = maximum_byte_span,
                },
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (emitted_) {
      return {};
    }
    emitted_ = true;
    const cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return pipeline_failure(ErrorCode::kNotSupported);
    }
    const std::span<const std::int64_t> shape =
        fixture_ == ImageFixture::kTwoChannel ? kTwoChannelInputShape : kInputShape;
    const std::span<const std::uint8_t> values =
        fixture_ == ImageFixture::kTwoChannel
            ? std::span<const std::uint8_t>{kTwoChannelInputValues}
            : std::span<const std::uint8_t>{kInputValues};
    auto loan = output.allocate_tensor(TensorLoanRequest{.shape = shape, .dtype = kU8});
    if (!loan) {
      return pipeline_failure(loan.error().code);
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return pipeline_failure(writer.error().code);
    }
    if (writer->byte_size() != values.size() ||
        cudaMemcpyAsync(
            writer->data(), values.data(), values.size(), cudaMemcpyHostToDevice, stream) !=
            cudaSuccess) {
      return pipeline_failure(ErrorCode::kFailure);
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return pipeline_failure(committed.error().code);
    }
    auto emitted = output.emit(std::move(*loan),
                               EmitOptions{.capture_time = context.activation_time(),
                                           .frame_id = kFrameId,
                                           .trace = TraceContext::root(kTraceId),
                                           .flags = SampleFlags::kDegraded});
    if (!emitted) {
      return pipeline_failure(emitted.error().code);
    }
    return {};
  }

  Output<Tensor> output{"output"};

 private:
  ImageFixture fixture_;
  bool emitted_{};
};

class ImageSink final : public Operator<> {
 public:
  explicit ImageSink(ImageFixture fixture = ImageFixture::kSingleChannel) : fixture_(fixture) {}

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{.representation = kCudaTensorRepresentation});
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return pipeline_failure(sample.error().code);
    }
    const std::span<const std::int64_t> received_shape = sample->data.shape_span();
    const std::int64_t expected_channels = fixture_ == ImageFixture::kTwoChannel ? 2 : 1;
    const std::size_t expected_byte_size = static_cast<std::size_t>(4 * expected_channels);
    if (received_shape.size() != 3U || received_shape[0] != 2 || received_shape[1] != 2 ||
        received_shape[2] != expected_channels || sample->data.nbytes() != expected_byte_size ||
        sample->data.device().device_type != kDLCUDA) {
      return pipeline_failure(ErrorCode::kInvalidArgument);
    }

    std::array<std::uint8_t, 8U> host{};
    auto copied = sample->data.copy_to_host_async(
        host.data(), expected_byte_size, context.cuda_stream().get());
    if (!copied) {
      return pipeline_failure(copied.error().code);
    }
    if (cudaStreamSynchronize(context.cuda_stream().get()) != cudaSuccess) {
      return pipeline_failure(ErrorCode::kFailure);
    }
    const std::span<const std::int64_t, 3U> fixed_shape{received_shape.data(), 3U};
    PipelineObservation::record(std::span<const std::uint8_t>{host}.first(expected_byte_size),
                                fixed_shape,
                                sample->data.dtype(),
                                sample->metadata);
    return {};
  }

  Input<Tensor> input{"input"};

 private:
  ImageFixture fixture_;
};

TEST(VisionPipeline, ResizeUsesPlanOwnedCudaTensorAndForwardsMetadata) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0 ||
      cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    GTEST_SKIP() << "No usable CUDA device";
  }

  PipelineObservation::reset();
  Graph graph{"vision-resize-test"};
  const auto source = graph.op<ImageSource>("source");
  auto resize = graph.op<ResizeOp>(
      "resize", VisionImageCapacity{.max_width = 2, .max_height = 2, .max_batch_size = 1U});
  resize.set_params(
      ResizeOpParams{.width = 2, .height = 2, .interpolation = InterpolationMode::kNearest});
  const auto sink = graph.op<ImageSink>("sink");
  graph.add_flow(source->output, resize->input);
  graph.add_flow(resize->output, sink->input);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-clock"));

  CompileOptions options;
  options.deployment
      .bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = "source", .output_port = "output", .device = DeviceId{0}})
      .bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = "resize", .output_port = "output", .device = DeviceId{0}});
  const ExecutionPlan plan = compile(graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_allocations().size(), 2U);

  RunSession session = run_async(plan);
  const bool observed = PipelineObservation::wait();
  session.request_stop();
  session.wait();

  EXPECT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  const std::lock_guard lock(PipelineObservation::mutex);
  EXPECT_FALSE(PipelineObservation::failure.has_value());
  EXPECT_TRUE(PipelineObservation::complete);
  EXPECT_EQ(PipelineObservation::shape, (std::array<std::int64_t, 3U>{2, 2, 1}));
  EXPECT_EQ(PipelineObservation::values,
            (std::array<std::uint8_t, 8U>{5U, 7U, 13U, 15U, 0U, 0U, 0U, 0U}));
  EXPECT_EQ(PipelineObservation::dtype.code, kU8.code);
  EXPECT_EQ(PipelineObservation::dtype.bits, kU8.bits);
  EXPECT_EQ(PipelineObservation::dtype.lanes, kU8.lanes);
  EXPECT_TRUE(PipelineObservation::metadata.source_clock_id.valid());
  EXPECT_EQ(PipelineObservation::metadata.frame_id, kFrameId);
  EXPECT_EQ(PipelineObservation::metadata.trace_id, kTraceId);
  EXPECT_TRUE(PipelineObservation::metadata.degraded());
}

TEST(VisionPipeline, ChainsCubicResizeAndCustomCropWithoutStaging) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0 ||
      cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    GTEST_SKIP() << "No usable CUDA device";
  }

  PipelineObservation::reset();
  Graph graph{"vision-chain-test"};
  const auto source = graph.op<ImageSource>("source");
  auto resize = graph.op<ResizeOp>(
      "resize", VisionImageCapacity{.max_width = 4, .max_height = 4, .max_batch_size = 1U});
  resize.set_params(
      ResizeOpParams{.width = 4, .height = 4, .interpolation = InterpolationMode::kCubic});
  auto crop = graph.op<CustomCropOp>(
      "crop", VisionImageCapacity{.max_width = 2, .max_height = 2, .max_batch_size = 1U});
  crop.set_params(CustomCropOpParams{.x = 1, .y = 1, .width = 2, .height = 2});
  const auto sink = graph.op<ImageSink>("sink");
  graph.add_flow(source->output, resize->input);
  graph.add_flow(resize->output, crop->input);
  graph.add_flow(crop->output, sink->input);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-clock"));

  CompileOptions options;
  options.deployment
      .bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = "source", .output_port = "output", .device = DeviceId{0}})
      .bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = "resize", .output_port = "output", .device = DeviceId{0}})
      .bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = "crop", .output_port = "output", .device = DeviceId{0}});
  const ExecutionPlan plan = compile(graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_allocations().size(), 3U);

  RunSession session = run_async(plan);
  const bool observed = PipelineObservation::wait();
  session.request_stop();
  session.wait();

  EXPECT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  const std::lock_guard lock(PipelineObservation::mutex);
  EXPECT_FALSE(PipelineObservation::failure.has_value());
  EXPECT_TRUE(PipelineObservation::complete);
  EXPECT_EQ(PipelineObservation::shape, (std::array<std::int64_t, 3U>{2, 2, 1}));
  EXPECT_EQ(PipelineObservation::values,
            (std::array<std::uint8_t, 8U>{5U, 6U, 9U, 10U, 0U, 0U, 0U, 0U}));
  EXPECT_EQ(PipelineObservation::dtype.code, kU8.code);
  EXPECT_EQ(PipelineObservation::dtype.bits, kU8.bits);
  EXPECT_EQ(PipelineObservation::dtype.lanes, kU8.lanes);
  EXPECT_TRUE(PipelineObservation::metadata.source_clock_id.valid());
  EXPECT_EQ(PipelineObservation::metadata.frame_id, kFrameId);
  EXPECT_EQ(PipelineObservation::metadata.trace_id, kTraceId);
  EXPECT_TRUE(PipelineObservation::metadata.degraded());
}

TEST(VisionPipeline, CustomCropProcessesTwoChannelInput) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0 ||
      cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    GTEST_SKIP() << "No usable CUDA device";
  }

  PipelineObservation::reset();
  Graph graph{"vision-two-channel-crop-test"};
  const auto source = graph.op<ImageSource>("source", ImageFixture::kTwoChannel);
  auto crop = graph.op<CustomCropOp>(
      "crop", VisionImageCapacity{.max_width = 2, .max_height = 2, .max_batch_size = 1U});
  crop.set_params(CustomCropOpParams{.x = 1, .y = 1, .width = 2, .height = 2});
  const auto sink = graph.op<ImageSink>("sink", ImageFixture::kTwoChannel);
  graph.add_flow(source->output, crop->input);
  graph.add_flow(crop->output, sink->input);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-clock"));

  CompileOptions options;
  options.deployment
      .bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = "source", .output_port = "output", .device = DeviceId{0}})
      .bind_tensor_output_device(TensorOutputDevicePlacement{
          .operator_path = "crop", .output_port = "output", .device = DeviceId{0}});
  const ExecutionPlan plan = compile(graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_allocations().size(), 2U);

  RunSession session = run_async(plan);
  const bool observed = PipelineObservation::wait();
  session.request_stop();
  session.wait();

  EXPECT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  const std::lock_guard lock(PipelineObservation::mutex);
  EXPECT_FALSE(PipelineObservation::failure.has_value());
  EXPECT_TRUE(PipelineObservation::complete);
  EXPECT_EQ(PipelineObservation::shape, (std::array<std::int64_t, 3U>{2, 2, 2}));
  EXPECT_EQ(PipelineObservation::values,
            (std::array<std::uint8_t, 8U>{5U, 105U, 6U, 106U, 9U, 109U, 10U, 110U}));
  EXPECT_EQ(PipelineObservation::dtype.code, kU8.code);
  EXPECT_EQ(PipelineObservation::dtype.bits, kU8.bits);
  EXPECT_EQ(PipelineObservation::dtype.lanes, kU8.lanes);
  EXPECT_TRUE(PipelineObservation::metadata.source_clock_id.valid());
  EXPECT_EQ(PipelineObservation::metadata.frame_id, kFrameId);
  EXPECT_EQ(PipelineObservation::metadata.trace_id, kTraceId);
  EXPECT_TRUE(PipelineObservation::metadata.degraded());
}

constexpr DLDataType kS16{kDLInt, 16U, 1U};
constexpr DLDataType kF32{kDLFloat, 32U, 1U};
constexpr std::array<std::int64_t, 3U> kColorImageShape{4, 4, 3};
constexpr std::array<std::uint8_t, 48U> kColorImageValues{
    0U,  20U, 40U, 1U,  21U, 41U, 2U,  22U, 42U, 3U,  23U, 43U, 4U,  24U, 44U, 5U,
    25U, 45U, 6U,  26U, 46U, 7U,  27U, 47U, 8U,  28U, 48U, 9U,  29U, 49U, 10U, 30U,
    50U, 11U, 31U, 51U, 12U, 32U, 52U, 13U, 33U, 53U, 14U, 34U, 54U, 15U, 35U, 55U,
};
constexpr std::array<std::int64_t, 3U> kMapShape{2, 2, 2};
constexpr std::array<float, 8U> kMapValues{1.F, 1.F, 2.F, 1.F, 1.F, 2.F, 2.F, 2.F};
constexpr std::array<float, 8U> kReplacementMapValues{0.F, 0.F, 3.F, 0.F, 0.F, 3.F, 3.F, 3.F};
constexpr std::array<std::uint8_t, 12U> kInitialRemapValues{
    5U, 25U, 45U, 6U, 26U, 46U, 9U, 29U, 49U, 10U, 30U, 50U};
constexpr std::array<std::uint8_t, 12U> kReplacementRemapValues{
    0U, 20U, 40U, 3U, 23U, 43U, 12U, 32U, 52U, 15U, 35U, 55U};
constexpr std::array<std::int64_t, 3U> kBoxesShape{1, 3, 4};
constexpr std::array<std::int16_t, 12U> kBoxesValues{0, 0, 10, 10, 0, 0, 10, 10, 20, 20, 5, 5};
constexpr std::array<std::int64_t, 2U> kScoresShape{1, 3};
constexpr std::array<float, 3U> kScoresValues{0.9F, 0.8F, 0.7F};
constexpr std::array<float, 12U> kExpectedPlanarValues{
    5.F, 6.F, 9.F, 10.F, 15.F, 16.F, 19.F, 20.F, 25.F, 26.F, 29.F, 30.F};
constexpr std::array<std::uint8_t, 3U> kExpectedMask{1U, 0U, 1U};
constexpr std::size_t kStaticMapImageCount{3U};
constexpr TensorRepresentation kMapTensorRepresentation{
    .memory_kind = MemoryKind::kCudaDevice,
    .dtype = kF32,
    .rank = 3U,
};
constexpr TensorRepresentation kBoxesTensorRepresentation{
    .memory_kind = MemoryKind::kCudaDevice,
    .dtype = kS16,
    .rank = 3U,
};
constexpr TensorRepresentation kScoresTensorRepresentation{
    .memory_kind = MemoryKind::kCudaDevice,
    .dtype = kF32,
    .rank = 2U,
};

struct FullPipelineObservation {
  static void reset() {
    const std::lock_guard lock(mutex);
    image_values.fill(0.F);
    image_shape.fill(0);
    image_dtype = {};
    image_metadata = {};
    mask_values.fill(0U);
    mask_shape.fill(0);
    mask_dtype = {};
    mask_metadata = {};
    failure.reset();
    image_count = 0U;
    image_values_match = true;
    mask_complete = false;
  }

  static void fail(ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      failure = code;
    }
    ready.notify_all();
  }

  static void record_image(std::span<const float, 12U> values,
                           std::span<const std::int64_t, 3U> shape, DLDataType dtype,
                           SampleMetadata metadata) {
    {
      const std::lock_guard lock(mutex);
      std::ranges::copy(values, image_values.begin());
      std::ranges::copy(shape, image_shape.begin());
      image_dtype = dtype;
      image_metadata = metadata;
      image_values_match = image_values_match && std::ranges::equal(values, kExpectedPlanarValues);
      ++image_count;
    }
    ready.notify_all();
  }

  static void record_mask(std::span<const std::uint8_t, 3U> values,
                          std::span<const std::int64_t, 2U> shape, DLDataType dtype,
                          SampleMetadata metadata) {
    {
      const std::lock_guard lock(mutex);
      std::ranges::copy(values, mask_values.begin());
      std::ranges::copy(shape, mask_shape.begin());
      mask_dtype = dtype;
      mask_metadata = metadata;
      mask_complete = true;
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 5s, [] {
      return (image_count == kStaticMapImageCount && mask_complete) || failure.has_value();
    });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::array<float, 12U> image_values{};
  static inline std::array<std::int64_t, 3U> image_shape{};
  static inline DLDataType image_dtype{};
  static inline SampleMetadata image_metadata{};
  static inline std::array<std::uint8_t, 3U> mask_values{};
  static inline std::array<std::int64_t, 2U> mask_shape{};
  static inline DLDataType mask_dtype{};
  static inline SampleMetadata mask_metadata{};
  static inline std::optional<ErrorCode> failure;
  static inline std::size_t image_count{};
  static inline bool image_values_match{true};
  static inline bool mask_complete{};
};

[[nodiscard]] expected<void, Error> full_pipeline_failure(ErrorCode code) {
  FullPipelineObservation::fail(code);
  return make_unexpected(Error{code});
}

class FullPipelineSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(image, "image")
        .produces_tensor(TensorOutputSpec{
            .representation = kColorImageTensorRepresentation,
            .bounds =
                TensorBounds{
                    .max_rank = 3U,
                    .max_byte_span = kColorImageValues.size(),
                },
            .storage = TensorOutputStorage::kRuntimePool,
        });
    spec.output(map, "map")
        .produces_tensor(TensorOutputSpec{
            .representation = kMapTensorRepresentation,
            .bounds =
                TensorBounds{
                    .max_rank = 3U,
                    .max_byte_span = sizeof(kMapValues),
                },
            .storage = TensorOutputStorage::kRuntimePool,
        });
    spec.output(boxes, "boxes")
        .produces_tensor(TensorOutputSpec{
            .representation = kBoxesTensorRepresentation,
            .bounds =
                TensorBounds{
                    .max_rank = 3U,
                    .max_byte_span = sizeof(kBoxesValues),
                },
            .storage = TensorOutputStorage::kRuntimePool,
        });
    spec.output(scores, "scores")
        .produces_tensor(TensorOutputSpec{
            .representation = kScoresTensorRepresentation,
            .bounds =
                TensorBounds{
                    .max_rank = 2U,
                    .max_byte_span = sizeof(kScoresValues),
                },
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    const std::size_t activation_index = activation_count_++;
    if (activation_index >= kStaticMapImageCount) {
      return {};
    }
    const cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return full_pipeline_failure(ErrorCode::kNotSupported);
    }
    const auto capture_time = context.activation_time();
    if (auto result = emit_tensor(
            image, kColorImageShape, kColorImageValues, kU8, capture_time, context.cuda_stream());
        !result) {
      return result;
    }
    if (activation_index != 0U) {
      return {};
    }
    if (auto result =
            emit_tensor(map, kMapShape, kMapValues, kF32, capture_time, context.cuda_stream());
        !result) {
      return result;
    }
    if (auto result = emit_tensor(
            boxes, kBoxesShape, kBoxesValues, kS16, capture_time, context.cuda_stream());
        !result) {
      return result;
    }
    return emit_tensor(
        scores, kScoresShape, kScoresValues, kF32, capture_time, context.cuda_stream());
  }

  Output<Tensor> image{"image"};
  Output<Tensor> map{"map"};
  Output<Tensor> boxes{"boxes"};
  Output<Tensor> scores{"scores"};

 private:
  template <typename T, std::size_t Rank, std::size_t Size>
  [[nodiscard]] static expected<void, Error> emit_tensor(
      Output<Tensor>& output, const std::array<std::int64_t, Rank>& shape,
      const std::array<T, Size>& values, DLDataType dtype, TimePoint capture_time,
      ::cuda::stream_ref stream) {
    auto loan = output.allocate_tensor(TensorLoanRequest{.shape = shape, .dtype = dtype});
    if (!loan) {
      return full_pipeline_failure(loan.error().code);
    }
    auto writer = loan->write(stream);
    if (!writer) {
      return full_pipeline_failure(writer.error().code);
    }
    if (writer->byte_size() != sizeof(values) ||
        cudaMemcpyAsync(
            writer->data(), values.data(), sizeof(values), cudaMemcpyHostToDevice, stream.get()) !=
            cudaSuccess) {
      return full_pipeline_failure(ErrorCode::kFailure);
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return full_pipeline_failure(committed.error().code);
    }
    auto emitted = output.emit(std::move(*loan),
                               EmitOptions{.capture_time = capture_time,
                                           .frame_id = kFrameId,
                                           .trace = TraceContext::root(kTraceId),
                                           .flags = SampleFlags::kDegraded});
    if (!emitted) {
      return full_pipeline_failure(emitted.error().code);
    }
    return {};
  }

  std::size_t activation_count_{};
};

TEST(RemapOp, ExactMapSynchronizationCompilesToAlignedExactSelections) {
  constexpr std::int32_t kMaxOutputWidth = 3;
  constexpr std::int32_t kMaxOutputHeight = 5;
  constexpr std::uint32_t kMaxBatchSize = 2U;
  Graph graph{"vision-remap-exact-contract-test"};
  const auto source = graph.op<FullPipelineSource>("source");
  auto remap = graph.op<RemapOp>("remap",
                                 VisionImageCapacity{.max_width = kMaxOutputWidth,
                                                     .max_height = kMaxOutputHeight,
                                                     .max_batch_size = kMaxBatchSize});
  remap.set_params(RemapOpParams{
      .width = 2,
      .height = 2,
      .input_interpolation = InterpolationMode::kNearest,
      .map_interpolation = InterpolationMode::kNearest,
      .map_value_mode = RemapMapValueMode::kAbsolute,
      .align_corners = false,
      .border_mode = BorderMode::kConstant,
      .border_value = {0.F},
      .map_synchronization = RemapMapSynchronizationMode::kExact,
      .map_max_age = 1s,
  });
  graph.add_flow(source->image, remap->input);
  graph.add_flow(source->map, remap->map);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-clock"));

  CompileOptions options;
  for (const auto& [operator_path, output_port] :
       std::array<std::pair<const char*, const char*>, 5U>{{
           {"source", "image"},
           {"source", "map"},
           {"source", "boxes"},
           {"source", "scores"},
           {"remap", "output"},
       }}) {
    options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
        .operator_path = operator_path, .output_port = output_port, .device = DeviceId{0}});
  }
  const ExecutionPlan plan = compile(graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto remap_vertex =
      std::ranges::find(plan.graph().vertices(), "remap", &VertexRecord::name);
  ASSERT_NE(remap_vertex, plan.graph().vertices().end());
  const auto remap_output = std::ranges::find_if(plan.ports(), [&](const PortPlanRecord& port) {
    return port.owner == remap_vertex->id && port.direction == PortDirection::kOutput &&
           port.local_name == "output";
  });
  ASSERT_NE(remap_output, plan.ports().end());
  const auto remap_allocation = std::ranges::find(
      plan.tensor_allocations(), remap_output->id, &TensorAllocationPlanRecord::output);
  ASSERT_NE(remap_allocation, plan.tensor_allocations().end());
  EXPECT_EQ(remap_allocation->bounds.max_byte_span,
            4U * static_cast<std::size_t>(kMaxOutputWidth) *
                static_cast<std::size_t>(kMaxOutputHeight) * kMaxBatchSize);
  ASSERT_EQ(plan.aligned_triggers().size(), 1U);
  const AlignedTriggerPlanRecord& aligned = plan.aligned_triggers().front();
  EXPECT_EQ(aligned.inclusive_window_ns, 0);
  ASSERT_EQ(aligned.members.size(), 2U);
  ASSERT_EQ(plan.input_selections().size(), 2U);
  for (const InputSelectionPlanRecord& selection : plan.input_selections()) {
    EXPECT_EQ(selection.selector, InputSelectorKind::kExact);
    EXPECT_EQ(selection.missing, TemporalRecordMissingKind::kUnspecified);
  }
}

struct RemapRefreshObservation {
  static void reset() {
    const std::lock_guard lock(mutex);
    failure.reset();
    image_count = 0U;
    saw_initial_map = false;
    saw_replacement_map = false;
    outputs_valid = true;
  }

  static void fail(ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      failure = code;
    }
    ready.notify_all();
  }

  static void record(std::span<const std::uint8_t, 12U> values) {
    {
      const std::lock_guard lock(mutex);
      const bool initial = std::ranges::equal(values, kInitialRemapValues);
      const bool replacement = std::ranges::equal(values, kReplacementRemapValues);
      saw_initial_map = saw_initial_map || initial;
      saw_replacement_map = saw_replacement_map || replacement;
      outputs_valid = outputs_valid && (initial || replacement);
      ++image_count;
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 5s, [] { return saw_replacement_map || failure.has_value(); });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::optional<ErrorCode> failure;
  static inline std::size_t image_count{};
  static inline bool saw_initial_map{};
  static inline bool saw_replacement_map{};
  static inline bool outputs_valid{true};
};

[[nodiscard]] expected<void, Error> remap_refresh_failure(ErrorCode code) {
  RemapRefreshObservation::fail(code);
  return make_unexpected(Error{code});
}

class RemapRefreshSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(image, "image")
        .produces_tensor(TensorOutputSpec{
            .representation = kColorImageTensorRepresentation,
            .bounds =
                TensorBounds{
                    .max_rank = 3U,
                    .max_byte_span = kColorImageValues.size(),
                },
            .storage = TensorOutputStorage::kRuntimePool,
        });
    spec.output(map, "map")
        .produces_tensor(TensorOutputSpec{
            .representation = kMapTensorRepresentation,
            .bounds =
                TensorBounds{
                    .max_rank = 3U,
                    .max_byte_span = sizeof(kMapValues),
                },
            .storage = TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 10ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    constexpr std::size_t kImageCount = 4U;
    const std::size_t activation_index = activation_count_++;
    if (activation_index >= kImageCount) {
      return {};
    }
    if (context.cuda_stream().get() == nullptr) {
      return remap_refresh_failure(ErrorCode::kNotSupported);
    }
    const TimePoint capture_time = context.activation_time();
    if (activation_index == 0U) {
      if (auto result =
              emit_tensor(map, kMapShape, kMapValues, kF32, capture_time, context.cuda_stream());
          !result) {
        return result;
      }
    } else if (activation_index == 2U) {
      if (auto result = emit_tensor(
              map, kMapShape, kReplacementMapValues, kF32, capture_time, context.cuda_stream());
          !result) {
        return result;
      }
    }
    return emit_tensor(
        image, kColorImageShape, kColorImageValues, kU8, capture_time, context.cuda_stream());
  }

  Output<Tensor> image{"image"};
  Output<Tensor> map{"map"};

 private:
  template <typename T, std::size_t Rank, std::size_t Size>
  [[nodiscard]] static expected<void, Error> emit_tensor(
      Output<Tensor>& output, const std::array<std::int64_t, Rank>& shape,
      const std::array<T, Size>& values, DLDataType dtype, TimePoint capture_time,
      ::cuda::stream_ref stream) {
    auto loan = output.allocate_tensor(TensorLoanRequest{.shape = shape, .dtype = dtype});
    if (!loan) {
      return remap_refresh_failure(loan.error().code);
    }
    auto writer = loan->write(stream);
    if (!writer) {
      return remap_refresh_failure(writer.error().code);
    }
    if (writer->byte_size() != sizeof(values) ||
        cudaMemcpyAsync(
            writer->data(), values.data(), sizeof(values), cudaMemcpyHostToDevice, stream.get()) !=
            cudaSuccess) {
      return remap_refresh_failure(ErrorCode::kFailure);
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return remap_refresh_failure(committed.error().code);
    }
    auto emitted = output.emit(std::move(*loan), EmitOptions{.capture_time = capture_time});
    if (!emitted) {
      return remap_refresh_failure(emitted.error().code);
    }
    return {};
  }

  std::size_t activation_count_{};
};

class RemapRefreshSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{.representation = kCudaTensorRepresentation});
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return remap_refresh_failure(sample.error().code);
    }
    const auto shape = sample->data.shape_span();
    if (shape.size() != 3U || shape[0] != 2 || shape[1] != 2 || shape[2] != 3 ||
        sample->data.nbytes() != kInitialRemapValues.size() ||
        sample->data.device().device_type != kDLCUDA) {
      return remap_refresh_failure(ErrorCode::kInvalidArgument);
    }
    std::array<std::uint8_t, 12U> host{};
    auto copied =
        sample->data.copy_to_host_async(host.data(), host.size(), context.cuda_stream().get());
    if (!copied) {
      return remap_refresh_failure(copied.error().code);
    }
    if (cudaStreamSynchronize(context.cuda_stream().get()) != cudaSuccess) {
      return remap_refresh_failure(ErrorCode::kFailure);
    }
    RemapRefreshObservation::record(host);
    return {};
  }

  Input<Tensor> input{"input"};
};

TEST(RemapOp, LatestModeAdmitsANewerMapWhileReusingThePreviousMap) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0 ||
      cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    GTEST_SKIP() << "No usable CUDA device";
  }

  RemapRefreshObservation::reset();
  Graph graph{"vision-remap-refresh-test"};
  const auto source = graph.op<RemapRefreshSource>("source");
  auto remap = graph.op<RemapOp>(
      "remap", VisionImageCapacity{.max_width = 2, .max_height = 2, .max_batch_size = 1U});
  remap.set_params(RemapOpParams{
      .width = 2,
      .height = 2,
      .input_interpolation = InterpolationMode::kNearest,
      .map_interpolation = InterpolationMode::kNearest,
      .map_value_mode = RemapMapValueMode::kAbsolute,
      .align_corners = false,
      .border_mode = BorderMode::kConstant,
      .border_value = {0.F},
      .map_max_age = 1s,
  });
  const auto sink = graph.op<RemapRefreshSink>("sink");
  graph.add_flow(source->image, remap->input);
  graph.add_flow(source->map, remap->map);
  graph.add_flow(remap->output, sink->input);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-clock"));

  CompileOptions options;
  for (const auto& [operator_path, output_port] :
       std::array<std::pair<const char*, const char*>, 3U>{{
           {"source", "image"},
           {"source", "map"},
           {"remap", "output"},
       }}) {
    options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
        .operator_path = operator_path, .output_port = output_port, .device = DeviceId{0}});
  }
  const ExecutionPlan plan = compile(graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();

  RunSession session = run_async(plan);
  const bool observed = RemapRefreshObservation::wait();
  session.request_stop();
  session.wait();

  EXPECT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  const std::lock_guard lock(RemapRefreshObservation::mutex);
  EXPECT_FALSE(RemapRefreshObservation::failure.has_value());
  EXPECT_GE(RemapRefreshObservation::image_count, 3U);
  EXPECT_TRUE(RemapRefreshObservation::saw_initial_map);
  EXPECT_TRUE(RemapRefreshObservation::saw_replacement_map);
  EXPECT_TRUE(RemapRefreshObservation::outputs_valid);
}

class PlanarImageSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{.representation = kCudaTensorRepresentation});
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return full_pipeline_failure(sample.error().code);
    }
    const auto shape = sample->data.shape_span();
    if (shape.size() != 3U || shape[0] != 3 || shape[1] != 2 || shape[2] != 2 ||
        sample->data.nbytes() != sizeof(kExpectedPlanarValues) ||
        sample->data.device().device_type != kDLCUDA) {
      return full_pipeline_failure(ErrorCode::kInvalidArgument);
    }
    std::array<float, 12U> host{};
    auto copied =
        sample->data.copy_to_host_async(host.data(), sizeof(host), context.cuda_stream().get());
    if (!copied) {
      return full_pipeline_failure(copied.error().code);
    }
    if (cudaStreamSynchronize(context.cuda_stream().get()) != cudaSuccess) {
      return full_pipeline_failure(ErrorCode::kFailure);
    }
    FullPipelineObservation::record_image(host,
                                          std::span<const std::int64_t, 3U>{shape.data(), 3U},
                                          sample->data.dtype(),
                                          sample->metadata);
    return {};
  }

  Input<Tensor> input{"input"};
};

class MaskSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorInputSpec{.representation = kCudaTensorRepresentation});
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return full_pipeline_failure(sample.error().code);
    }
    const auto shape = sample->data.shape_span();
    if (shape.size() != 2U || shape[0] != 1 || shape[1] != 3 ||
        sample->data.nbytes() != kExpectedMask.size() ||
        sample->data.device().device_type != kDLCUDA) {
      return full_pipeline_failure(ErrorCode::kInvalidArgument);
    }
    std::array<std::uint8_t, 3U> host{};
    auto copied =
        sample->data.copy_to_host_async(host.data(), host.size(), context.cuda_stream().get());
    if (!copied) {
      return full_pipeline_failure(copied.error().code);
    }
    if (cudaStreamSynchronize(context.cuda_stream().get()) != cudaSuccess) {
      return full_pipeline_failure(ErrorCode::kFailure);
    }
    FullPipelineObservation::record_mask(host,
                                         std::span<const std::int64_t, 2U>{shape.data(), 2U},
                                         sample->data.dtype(),
                                         sample->metadata);
    return {};
  }

  Input<Tensor> input{"input"};
};

TEST(VisionPipeline, ComposesAllCvCudaOperatorsAcrossImageAndDetectionBranches) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0 ||
      cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    GTEST_SKIP() << "No usable CUDA device";
  }

  FullPipelineObservation::reset();
  Graph graph{"vision-full-pipeline-test"};
  const auto source = graph.op<FullPipelineSource>("source");
  auto resize = graph.op<ResizeOp>(
      "resize", VisionImageCapacity{.max_width = 4, .max_height = 4, .max_batch_size = 1U});
  resize.set_params(
      ResizeOpParams{.width = 4, .height = 4, .interpolation = InterpolationMode::kNearest});
  auto crop = graph.op<CustomCropOp>(
      "crop", VisionImageCapacity{.max_width = 2, .max_height = 2, .max_batch_size = 1U});
  crop.set_params(CustomCropOpParams{.x = 1, .y = 1, .width = 2, .height = 2});
  auto border = graph.op<CopyMakeBorderOp>(
      "border", VisionImageCapacity{.max_width = 4, .max_height = 4, .max_batch_size = 1U});
  border.set_params(CopyMakeBorderOpParams{.top = 1,
                                           .bottom = 1,
                                           .left = 1,
                                           .right = 1,
                                           .border_mode = BorderMode::kConstant,
                                           .border_value = {0.F}});
  auto remap = graph.op<RemapOp>(
      "remap", VisionImageCapacity{.max_width = 2, .max_height = 2, .max_batch_size = 1U});
  remap.set_params(RemapOpParams{.width = 2,
                                 .height = 2,
                                 .input_interpolation = InterpolationMode::kNearest,
                                 .map_interpolation = InterpolationMode::kNearest,
                                 .map_value_mode = RemapMapValueMode::kAbsolute,
                                 .align_corners = false,
                                 .border_mode = BorderMode::kConstant,
                                 .border_value = {0.F},
                                 .map_max_age = 1s});
  auto convert = graph.op<ConvertToOp>(
      "convert", VisionImageCapacity{.max_width = 2, .max_height = 2, .max_batch_size = 1U});
  convert.set_params(ConvertToOpParams{.output_type = VisionElementType::kF32,
                                       .alpha = 1.0,
                                       .beta = 0.0,
                                       .rounding = RoundingMode::kNearest});
  auto normalize = graph.op<NormalizeOp>(
      "normalize", VisionImageCapacity{.max_width = 2, .max_height = 2, .max_batch_size = 1U});
  normalize.set_params(NormalizeOpParams{.base = {0.F, 10.F, 20.F},
                                         .scale = {1.F},
                                         .global_scale = 1.F,
                                         .shift = 0.F,
                                         .epsilon = 0.F,
                                         .scale_is_stddev = false});
  auto reformat = graph.op<ReformatOp>(
      "reformat", VisionImageCapacity{.max_width = 2, .max_height = 2, .max_batch_size = 1U});
  reformat.set_params(ReformatOpParams{.mode = ReformatMode::kInterleavedToPlanar});
  auto nms =
      graph.op<NonMaximumSuppressionOp>("nms", NMSCapacity{.max_boxes = 3, .max_batch_size = 1U});
  nms.set_params(NonMaximumSuppressionOpParams{.score_threshold = 0.F, .iou_threshold = 0.5F});
  const auto image_sink = graph.op<PlanarImageSink>("image-sink");
  const auto mask_sink = graph.op<MaskSink>("mask-sink");

  graph.add_flow(source->image, resize->input);
  graph.add_flow(resize->output, crop->input);
  graph.add_flow(crop->output, border->input);
  graph.add_flow(border->output, remap->input);
  graph.add_flow(source->map, remap->map);
  graph.add_flow(remap->output, convert->input);
  graph.add_flow(convert->output, normalize->input);
  graph.add_flow(normalize->output, reformat->input);
  graph.add_flow(reformat->output, image_sink->input);
  graph.add_flow(source->boxes, nms->boxes);
  graph.add_flow(source->scores, nms->scores);
  graph.add_flow(nms->mask, mask_sink->input);
  graph.set_default_clock(graph.add_clock<RealtimeClock>("runtime-clock"));
  graph.partition("application").include_all();

  CompileOptions options;
  for (const auto& [operator_path, output_port] :
       std::array<std::pair<const char*, const char*>, 12U>{{
           {"source", "image"},
           {"source", "map"},
           {"source", "boxes"},
           {"source", "scores"},
           {"resize", "output"},
           {"crop", "output"},
           {"border", "output"},
           {"remap", "output"},
           {"convert", "output"},
           {"normalize", "output"},
           {"reformat", "output"},
           {"nms", "mask"},
       }}) {
    options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
        .operator_path = operator_path, .output_port = output_port, .device = DeviceId{0}});
  }
  const ExecutionPlan plan = compile(graph, std::move(options));
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_allocations().size(), 12U);
  ASSERT_EQ(plan.aligned_triggers().size(), 1U);
  const auto latest_map = std::find_if(
      plan.input_selections().begin(), plan.input_selections().end(), [](const auto& selection) {
        return selection.selector == InputSelectorKind::kLatestWithin;
      });
  ASSERT_NE(latest_map, plan.input_selections().end());
  EXPECT_EQ(latest_map->max_age_ns,
            std::chrono::duration_cast<std::chrono::nanoseconds>(1s).count());
  EXPECT_EQ(latest_map->missing, TemporalRecordMissingKind::kReuseLast);
  EXPECT_EQ(latest_map->reuse_max_age_ns,
            std::chrono::duration_cast<std::chrono::nanoseconds>(1s).count());

  RunSession session = run_async(plan);
  const bool observed = FullPipelineObservation::wait();
  session.request_stop();
  session.wait();

  EXPECT_TRUE(observed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty());
  const std::lock_guard lock(FullPipelineObservation::mutex);
  EXPECT_FALSE(FullPipelineObservation::failure.has_value());
  EXPECT_EQ(FullPipelineObservation::image_count, kStaticMapImageCount);
  EXPECT_TRUE(FullPipelineObservation::image_values_match);
  EXPECT_TRUE(FullPipelineObservation::mask_complete);
  EXPECT_EQ(FullPipelineObservation::image_shape, (std::array<std::int64_t, 3U>{3, 2, 2}));
  EXPECT_EQ(FullPipelineObservation::image_values, kExpectedPlanarValues);
  EXPECT_EQ(FullPipelineObservation::image_dtype.code, kF32.code);
  EXPECT_EQ(FullPipelineObservation::image_dtype.bits, kF32.bits);
  EXPECT_EQ(FullPipelineObservation::image_dtype.lanes, kF32.lanes);
  EXPECT_EQ(FullPipelineObservation::mask_shape, (std::array<std::int64_t, 2U>{1, 3}));
  EXPECT_EQ(FullPipelineObservation::mask_values, kExpectedMask);
  EXPECT_EQ(FullPipelineObservation::mask_dtype.code, kU8.code);
  EXPECT_EQ(FullPipelineObservation::mask_dtype.bits, kU8.bits);
  EXPECT_EQ(FullPipelineObservation::mask_dtype.lanes, kU8.lanes);
  for (const SampleMetadata* metadata :
       {&FullPipelineObservation::image_metadata, &FullPipelineObservation::mask_metadata}) {
    EXPECT_TRUE(metadata->source_clock_id.valid());
    EXPECT_EQ(metadata->frame_id, kFrameId);
    EXPECT_EQ(metadata->trace_id, kTraceId);
    EXPECT_TRUE(metadata->degraded());
  }
}

}  // namespace
}  // namespace holoscan::vision
