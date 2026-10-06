// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <utility>

#include <cuda_runtime_api.h>

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
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/vision/convert_to_op.hpp>
#include <holoscan/vision/copy_make_border_op.hpp>
#include <holoscan/vision/custom_crop_op.hpp>
#include <holoscan/vision/non_maximum_suppression_op.hpp>
#include <holoscan/vision/normalize_op.hpp>
#include <holoscan/vision/reformat_op.hpp>
#include <holoscan/vision/remap_op.hpp>
#include <holoscan/vision/resize_op.hpp>

namespace {

using namespace std::chrono_literals;

constexpr int kSkipped = 77;
constexpr DLDataType kU8{kDLUInt, 8U, 1U};
constexpr DLDataType kS16{kDLInt, 16U, 1U};
constexpr DLDataType kF32{kDLFloat, 32U, 1U};
constexpr std::size_t kFrameCount = 3U;
constexpr std::array<std::int64_t, 3U> kImageShape{4, 4, 3};
constexpr std::array<std::uint8_t, 48U> kImageValues{
    0U,  20U, 40U, 1U,  21U, 41U, 2U,  22U, 42U, 3U,  23U, 43U, 4U,  24U, 44U, 5U,
    25U, 45U, 6U,  26U, 46U, 7U,  27U, 47U, 8U,  28U, 48U, 9U,  29U, 49U, 10U, 30U,
    50U, 11U, 31U, 51U, 12U, 32U, 52U, 13U, 33U, 53U, 14U, 34U, 54U, 15U, 35U, 55U,
};
constexpr std::array<std::int64_t, 3U> kMapShape{2, 2, 2};
constexpr std::array<float, 8U> kMapValues{1.F, 1.F, 2.F, 1.F, 1.F, 2.F, 2.F, 2.F};
constexpr std::array<std::int64_t, 3U> kBoxesShape{1, 3, 4};
constexpr std::array<std::int16_t, 12U> kBoxesValues{0, 0, 10, 10, 0, 0, 10, 10, 20, 20, 5, 5};
constexpr std::array<std::int64_t, 2U> kScoresShape{1, 3};
constexpr std::array<float, 3U> kScoresValues{0.9F, 0.8F, 0.7F};
constexpr std::array<float, 12U> kExpectedImage{
    5.F, 6.F, 9.F, 10.F, 15.F, 16.F, 19.F, 20.F, 25.F, 26.F, 29.F, 30.F};
constexpr std::array<std::uint8_t, 3U> kExpectedMask{1U, 0U, 1U};

[[nodiscard]] constexpr std::array<std::uint8_t, 48U> image_values_for(std::size_t frame_index) {
  auto values = kImageValues;
  for (auto& value : values) {
    value = static_cast<std::uint8_t>(value + frame_index);
  }
  return values;
}

[[nodiscard]] constexpr std::array<std::int16_t, 12U> boxes_values_for(std::size_t frame_index) {
  auto values = kBoxesValues;
  for (std::size_t index : {0U, 1U, 4U, 5U, 8U, 9U}) {
    values[index] = static_cast<std::int16_t>(values[index] + frame_index);
  }
  return values;
}

[[nodiscard]] constexpr std::array<float, 3U> scores_values_for(std::size_t frame_index) {
  auto values = kScoresValues;
  for (auto& value : values) {
    value -= static_cast<float>(frame_index) * 0.01F;
  }
  return values;
}

[[nodiscard]] constexpr std::array<float, 12U> expected_image_for(std::size_t frame_index) {
  auto values = kExpectedImage;
  for (auto& value : values) {
    value += static_cast<float>(frame_index);
  }
  return values;
}

constexpr std::array<std::array<std::uint8_t, 48U>, kFrameCount> kFrameImageValues{
    image_values_for(0U), image_values_for(1U), image_values_for(2U)};
constexpr std::array<std::array<std::int16_t, 12U>, kFrameCount> kFrameBoxesValues{
    boxes_values_for(0U), boxes_values_for(1U), boxes_values_for(2U)};
constexpr std::array<std::array<float, 3U>, kFrameCount> kFrameScoresValues{
    scores_values_for(0U), scores_values_for(1U), scores_values_for(2U)};
constexpr holoscan::TensorRepresentation kCudaTensorRepresentation{
    .memory_kind = holoscan::MemoryKind::kCudaDevice,
};
constexpr holoscan::TensorRepresentation kImageTensorRepresentation{
    .memory_kind = holoscan::MemoryKind::kCudaDevice,
    .dtype = kU8,
    .rank = 3U,
};
constexpr holoscan::TensorRepresentation kMapTensorRepresentation{
    .memory_kind = holoscan::MemoryKind::kCudaDevice,
    .dtype = kF32,
    .rank = 3U,
};
constexpr holoscan::TensorRepresentation kBoxesTensorRepresentation{
    .memory_kind = holoscan::MemoryKind::kCudaDevice,
    .dtype = kS16,
    .rank = 3U,
};
constexpr holoscan::TensorRepresentation kScoresTensorRepresentation{
    .memory_kind = holoscan::MemoryKind::kCudaDevice,
    .dtype = kF32,
    .rank = 2U,
};

[[nodiscard]] bool same_dtype(DLDataType lhs, DLDataType rhs) noexcept {
  return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}

struct Result {
  struct Frame {
    std::array<float, 12U> image{};
    std::array<std::uint8_t, 3U> mask{};
    holoscan::SampleMetadata image_metadata{};
    holoscan::SampleMetadata mask_metadata{};
    bool image_received{};
    bool mask_received{};
  };

  static void reset() {
    {
      const std::lock_guard lock(mutex);
      frames() = {};
      error.reset();
    }
    ready.notify_all();
  }

  static void fail(holoscan::ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      error = code;
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool record_image(std::span<const float, 12U> output,
                                         holoscan::SampleMetadata metadata) {
    bool recorded = false;
    {
      const std::lock_guard lock(mutex);
      if (metadata.frame_id == 0U || metadata.frame_id > kFrameCount || !metadata.valid() ||
          !metadata.source_clock_id.valid()) {
        error = holoscan::ErrorCode::kInvalidArgument;
      } else {
        Frame& frame = frames()[metadata.frame_id - 1U];
        if (frame.image_received ||
            !std::ranges::equal(output, expected_image_for(metadata.frame_id - 1U))) {
          error = holoscan::ErrorCode::kInvalidArgument;
        } else {
          std::ranges::copy(output, frame.image.begin());
          frame.image_metadata = metadata;
          frame.image_received = true;
          recorded = true;
        }
      }
    }
    ready.notify_all();
    return recorded;
  }

  [[nodiscard]] static bool record_mask(std::span<const std::uint8_t, 3U> output,
                                        holoscan::SampleMetadata metadata) {
    bool recorded = false;
    {
      const std::lock_guard lock(mutex);
      if (metadata.frame_id == 0U || metadata.frame_id > kFrameCount || !metadata.valid() ||
          !metadata.source_clock_id.valid()) {
        error = holoscan::ErrorCode::kInvalidArgument;
      } else {
        Frame& frame = frames()[metadata.frame_id - 1U];
        if (frame.mask_received || !std::ranges::equal(output, kExpectedMask)) {
          error = holoscan::ErrorCode::kInvalidArgument;
        } else {
          std::ranges::copy(output, frame.mask.begin());
          frame.mask_metadata = metadata;
          frame.mask_received = true;
          recorded = true;
        }
      }
    }
    ready.notify_all();
    return recorded;
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return ready.wait_for(lock, 5s, [] { return complete() || error.has_value(); });
  }

  [[nodiscard]] static bool valid() {
    const std::lock_guard lock(mutex);
    if (error.has_value() || !complete()) {
      return false;
    }
    std::int64_t previous_capture_timestamp{};
    for (std::size_t frame_index = 0U; frame_index < frames().size(); ++frame_index) {
      const Frame& frame = frames()[frame_index];
      const auto expected_frame_id = static_cast<std::uint64_t>(frame_index + 1U);
      if (frame.image != expected_image_for(frame_index) || frame.mask != kExpectedMask ||
          frame.image_metadata.frame_id != expected_frame_id ||
          frame.mask_metadata.frame_id != expected_frame_id ||
          frame.image_metadata.capture_timestamp_ns != frame.mask_metadata.capture_timestamp_ns ||
          frame.image_metadata.source_clock_id != frame.mask_metadata.source_clock_id ||
          (frame_index != 0U &&
           frame.image_metadata.capture_timestamp_ns <= previous_capture_timestamp)) {
        return false;
      }
      previous_capture_timestamp = frame.image_metadata.capture_timestamp_ns;
    }
    return true;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::optional<holoscan::ErrorCode> error;

 private:
  [[nodiscard]] static std::array<Frame, kFrameCount>& frames() {
    static std::array<Frame, kFrameCount> result{};
    return result;
  }

  [[nodiscard]] static bool complete() {
    return std::ranges::all_of(
        frames(), [](const Frame& frame) { return frame.image_received && frame.mask_received; });
  }
};

[[nodiscard]] holoscan::expected<void, holoscan::Error> fail(holoscan::ErrorCode code) {
  Result::fail(code);
  return holoscan::make_unexpected(holoscan::Error{code});
}

class PipelineSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(image, "image")
        .tensor_field(holoscan::schema::ImageTensorFields::data)
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = kImageTensorRepresentation,
            .bounds =
                holoscan::TensorBounds{
                    .max_rank = 3U,
                    .max_byte_span = kImageValues.size(),
                },
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        });
    spec.output(map, "map")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = kMapTensorRepresentation,
            .bounds =
                holoscan::TensorBounds{
                    .max_rank = 3U,
                    .max_byte_span = sizeof(kMapValues),
                },
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        });
    spec.output(boxes, "boxes")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = kBoxesTensorRepresentation,
            .bounds =
                holoscan::TensorBounds{
                    .max_rank = 3U,
                    .max_byte_span = sizeof(kBoxesValues),
                },
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        });
    spec.output(scores, "scores")
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = kScoresTensorRepresentation,
            .bounds =
                holoscan::TensorBounds{
                    .max_rank = 2U,
                    .max_byte_span = sizeof(kScoresValues),
                },
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    if (activation_count_ == kFrameCount) {
      return {};
    }
    const auto stream = context.cuda_stream();
    if (stream.get() == nullptr) {
      return fail(holoscan::ErrorCode::kNotSupported);
    }
    const holoscan::TimePoint capture_time = context.activation_time();
    const std::size_t frame_index = activation_count_++;
    const std::uint64_t frame_id = static_cast<std::uint64_t>(frame_index + 1U);
    if (frame_index == 0U) {
      if (auto result =
              emit_tensor(map, kMapShape, kMapValues, kF32, capture_time, frame_id, stream);
          !result) {
        return result;
      }
    }
    if (auto result = emit_tensor(image,
                                  kImageShape,
                                  kFrameImageValues[frame_index],
                                  kU8,
                                  capture_time,
                                  frame_id,
                                  stream);
        !result) {
      return result;
    }
    if (auto result = emit_tensor(boxes,
                                  kBoxesShape,
                                  kFrameBoxesValues[frame_index],
                                  kS16,
                                  capture_time,
                                  frame_id,
                                  stream);
        !result) {
      return result;
    }
    return emit_tensor(scores,
                       kScoresShape,
                       kFrameScoresValues[frame_index],
                       kF32,
                       capture_time,
                       frame_id,
                       stream);
  }

  holoscan::Output<holoscan::schema::ImageT> image{"image"};
  holoscan::Output<holoscan::Tensor> map{"map"};
  holoscan::Output<holoscan::Tensor> boxes{"boxes"};
  holoscan::Output<holoscan::Tensor> scores{"scores"};

 private:
  template <typename Published, typename T, std::size_t Rank, std::size_t Size>
  [[nodiscard]] static holoscan::expected<void, holoscan::Error> emit_tensor(
      holoscan::Output<Published>& output, const std::array<std::int64_t, Rank>& shape,
      const std::array<T, Size>& values, DLDataType dtype, holoscan::TimePoint capture_time,
      std::uint64_t frame_id, ::cuda::stream_ref stream) {
    auto loan = output.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = dtype});
    if (!loan) {
      return fail(loan.error().code);
    }
    auto writer = loan->write(stream);
    if (!writer) {
      return fail(writer.error().code);
    }
    if (writer->byte_size() != sizeof(values) ||
        cudaMemcpyAsync(
            writer->data(), values.data(), sizeof(values), cudaMemcpyHostToDevice, stream.get()) !=
            cudaSuccess) {
      return fail(holoscan::ErrorCode::kFailure);
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return fail(committed.error().code);
    }
    const holoscan::EmitOptions metadata{.capture_time = capture_time, .frame_id = frame_id};
    auto emitted = [&]() -> holoscan::expected<void, holoscan::Error> {
      if constexpr (std::same_as<Published, holoscan::schema::ImageT>) {
        holoscan::schema::ImageT descriptor;
        descriptor.width = 4;
        descriptor.height = 4;
        descriptor.encoding = holoscan::schema::ImageEncoding_RGB8;
        descriptor.color_space = holoscan::schema::ColorSpace_SRGB;
        descriptor.significant_bits = 7;
        descriptor.roi_offset_x = 10;
        descriptor.roi_offset_y = 20;
        descriptor.exposure_time_ns = 123;
        descriptor.gain = 2.F;
        descriptor.header = std::make_shared<holoscan::schema::HeaderT>();
        descriptor.header->capture_timestamp_ns = 321;
        descriptor.header->device_sequence = frame_id;
        descriptor.header->frame_id = "camera";
        return output.emit_tensor(std::move(*loan), descriptor, metadata);
      } else {
        return output.emit(std::move(*loan), metadata);
      }
    }();
    if (!emitted) {
      return fail(emitted.error().code);
    }
    return {};
  }

  std::size_t activation_count_{};
};

class ImageSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .tensor_field(holoscan::schema::ImageTensorFields::data)
        .expects_tensor(holoscan::TensorInputSpec{.representation = kCudaTensorRepresentation});
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = input.selected_sample();
    if (!sample) {
      return fail(sample.error().code);
    }
    const auto& image = sample->data();
    if (!image.data || image.width != 2 || image.height != 2 ||
        image.encoding != holoscan::schema::ImageEncoding_CUSTOM ||
        image.color_space != holoscan::schema::ColorSpace_UNSPECIFIED ||
        image.significant_bits != holoscan::schema::kUnknownImageSignificantBits ||
        image.roi_offset_x != 0 || image.roi_offset_y != 0 || image.exposure_time_ns != 123 ||
        image.gain != 2.F || !image.header || image.header->capture_timestamp_ns != 321 ||
        image.header->frame_id != "camera" ||
        image.header->device_sequence != sample->metadata().frame_id ||
        image.plane_layouts.size() != 3U) {
      return fail(holoscan::ErrorCode::kInvalidArgument);
    }
    for (std::size_t i = 0; i < image.plane_layouts.size(); ++i) {
      const auto& plane = image.plane_layouts[i];
      if (plane.offset_bytes() != i * 16U || plane.row_stride_bytes() != 8U ||
          plane.size_bytes() != 16U) {
        return fail(holoscan::ErrorCode::kInvalidArgument);
      }
    }
    const auto shape = image.data->shape_span();
    if (shape.size() != 3U || shape[0] != 3 || shape[1] != 2 || shape[2] != 2 ||
        std::cmp_not_equal(image.data->nbytes(), sizeof(kExpectedImage)) ||
        !same_dtype(image.data->dtype(), kF32)) {
      return fail(holoscan::ErrorCode::kInvalidArgument);
    }
    std::array<float, 12U> host{};
    auto copied =
        image.data->copy_to_host_async(host.data(), sizeof(host), context.cuda_stream().get());
    if (!copied) {
      return fail(copied.error().code);
    }
    if (cudaStreamSynchronize(context.cuda_stream().get()) != cudaSuccess) {
      return fail(holoscan::ErrorCode::kFailure);
    }
    if (!Result::record_image(host, sample->metadata())) {
      return fail(holoscan::ErrorCode::kInvalidArgument);
    }
    return {};
  }

  holoscan::Input<holoscan::schema::ImageT> input{"input"};
};

class MaskSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(holoscan::TensorInputSpec{.representation = kCudaTensorRepresentation});
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto sample = input.receive();
    if (!sample) {
      return fail(sample.error().code);
    }
    const auto shape = sample->data.shape_span();
    if (shape.size() != 2U || shape[0] != 1 || shape[1] != 3 ||
        std::cmp_not_equal(sample->data.nbytes(), kExpectedMask.size()) ||
        !same_dtype(sample->data.dtype(), kU8)) {
      return fail(holoscan::ErrorCode::kInvalidArgument);
    }
    std::array<std::uint8_t, 3U> host{};
    auto copied =
        sample->data.copy_to_host_async(host.data(), host.size(), context.cuda_stream().get());
    if (!copied) {
      return fail(copied.error().code);
    }
    if (cudaStreamSynchronize(context.cuda_stream().get()) != cudaSuccess) {
      return fail(holoscan::ErrorCode::kFailure);
    }
    if (!Result::record_mask(host, sample->metadata)) {
      return fail(holoscan::ErrorCode::kInvalidArgument);
    }
    return {};
  }

  holoscan::Input<holoscan::Tensor> input{"input"};
};

[[nodiscard]] holoscan::ExecutionPlan compile_pipeline() {
  holoscan::Graph graph{"vision-pipeline"};
  const auto source = graph.op<PipelineSource>("source");
  auto resize = graph.op<holoscan::vision::ResizeOp>("resize");
  resize.set_params(holoscan::vision::ResizeOpParams{
      .width = 4,
      .height = 4,
      .max_batch_size = 1U,
      .interpolation = holoscan::vision::InterpolationMode::kNearest,
  });
  auto crop = graph.op<holoscan::vision::CustomCropOp>("crop");
  crop.set_params(holoscan::vision::CustomCropOpParams{
      .x = 1, .y = 1, .width = 2, .height = 2, .max_batch_size = 1U});
  auto border = graph.op<holoscan::vision::CopyMakeBorderOp>("border");
  border.set_params(holoscan::vision::CopyMakeBorderOpParams{
      .max_width = 4,
      .max_height = 4,
      .max_batch_size = 1U,
      .top = 1,
      .bottom = 1,
      .left = 1,
      .right = 1,
      .border_mode = holoscan::vision::BorderMode::kConstant,
      .border_value = {0.F},
  });
  auto remap = graph.op<holoscan::vision::RemapOp>("remap");
  remap.set_params(holoscan::vision::RemapOpParams{
      .width = 2,
      .height = 2,
      .max_batch_size = 1U,
      .input_interpolation = holoscan::vision::InterpolationMode::kNearest,
      .map_interpolation = holoscan::vision::InterpolationMode::kNearest,
      .map_value_mode = holoscan::vision::RemapMapValueMode::kAbsolute,
      .align_corners = false,
      .border_mode = holoscan::vision::BorderMode::kConstant,
      .border_value = {0.F},
      .map_synchronization = holoscan::vision::RemapMapSynchronizationMode::kLatest,
  });
  auto convert = graph.op<holoscan::vision::ConvertToOp>("convert");
  convert.set_params(holoscan::vision::ConvertToOpParams{
      .output_type = holoscan::vision::VisionElementType::kF32,
      .alpha = 1.0,
      .beta = 0.0,
      .rounding = holoscan::vision::RoundingMode::kNearest,
  });
  auto normalize = graph.op<holoscan::vision::NormalizeOp>("normalize");
  normalize.set_params(holoscan::vision::NormalizeOpParams{
      .base = {0.F, 10.F, 20.F},
      .scale = {1.F},
      .global_scale = 1.F,
      .shift = 0.F,
      .epsilon = 0.F,
      .scale_is_stddev = false,
  });
  auto reformat = graph.op<holoscan::vision::ReformatOp>("reformat");
  reformat.set_params(holoscan::vision::ReformatOpParams{
      .mode = holoscan::vision::ReformatMode::kInterleavedToPlanar,
  });
  auto nms = graph.op<holoscan::vision::NonMaximumSuppressionOp>("nms");
  nms.set_params(holoscan::vision::NonMaximumSuppressionOpParams{
      .max_boxes = 3, .max_batch_size = 1U, .score_threshold = 0.F, .iou_threshold = 0.5F});
  const auto image_sink = graph.op<ImageSink>("image-sink");
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
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));
  graph.partition("application").include_all();

  holoscan::CompileOptions options;
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
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
        .operator_path = operator_path,
        .output_port = output_port,
        .device = holoscan::DeviceId{0},
    });
  }
  return holoscan::compile(graph, std::move(options));
}

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0 ||
      cudaSetDevice(0) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
    static_cast<void>(cudaGetLastError());
    std::cout << "vision_pipeline skipped: no usable CUDA device\n";
    return kSkipped;
  }

  Result::reset();
  const holoscan::ExecutionPlan plan = compile_pipeline();
  if (!plan.ok() || plan.tensor_allocations().size() != 12U) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  holoscan::RunSession session = holoscan::run_async(plan);
  const bool observed = Result::wait();
  session.request_stop();
  session.wait();

  if (!observed || !Result::valid() ||
      session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty()) {
    return 2;
  }
  std::cout << "vision_pipeline: image and detection branches passed\n";
  return 0;
}
