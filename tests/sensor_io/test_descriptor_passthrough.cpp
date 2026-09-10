/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Pins the metadata-forwarding idiom for descriptor-backed schema payloads, which is the pattern a
// chain of sensor operators lives or dies by: an operator receives a typed sample, replaces the
// Tensor, and must hand the payload metadata and the sample provenance downstream without
// enumerating either.
//
// Two independent records travel with every sample and they forward by different means:
//
//   * The payload descriptor (Image height, encoding, exposure, header, ...) forwards by handing
//     the received descriptor straight back to emit_tensor, whose contract states that the
//     descriptor's embedded Tensor member is ignored in favour of the loan. The operator therefore
//     names no field, and a schema that grows a field tomorrow needs no change here.
//   * The runtime envelope (SampleMetadata: capture time, clock domain, frame identity, trace)
//     is *not* part of the payload and does not ride along with it. It is rebuilt from EmitOptions
//     on every publication, so an operator that emits with default options silently publishes a
//     sample with no capture time at all. sensor_io::derive_from carries it across.
//
// The third scenario below deliberately drops provenance to prove that the loss is real and silent,
// because that failure surfaces several operators later as an unexplained alignment or windowing
// failure rather than as an error at the operator that caused it.

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
// Provides generated owning types and schema-trait specializations.
#include <holoscan/schema/flatbuffer_schema_package.hpp>  // NOLINT(misc-include-cleaner)
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "image_generated.h"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

/// @brief Select what the middle operator does with the metadata it receives.
enum class TransformMode : std::uint8_t {
  /// Hand the received descriptor back verbatim. The Frame<TensorA, M> -> Frame<TensorB, M> case.
  kForwardVerbatim,
  /// Change exactly one descriptor field and forward every other field untouched.
  kAdjustOneField,
  /// Emit with default options to demonstrate the provenance loss derive_from exists to prevent.
  kDropsProvenance,
};

constexpr std::array<std::int64_t, 1U> kShape{8};
constexpr std::array<std::uint8_t, 8U> kSourceBytes{0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U};
constexpr std::array<std::uint8_t, 8U> kTransformedBytes{
    255U, 254U, 253U, 252U, 251U, 250U, 249U, 248U};

// Payload descriptor values. Deliberately dense rather than physically coherent: the point is that
// every field survives an operator that mentions none of them.
constexpr std::string_view kCoordinateFrame{"camera_left_optical"};
constexpr std::int64_t kHeaderCaptureTimestampNs = 1'234'567'890;
constexpr std::uint64_t kDeviceSequence = 4'096U;
constexpr std::int32_t kRoiOffsetX = 16;
constexpr std::int32_t kSourceRoiOffsetY = 480;
constexpr std::int32_t kAdjustedRoiOffsetY = 960;
constexpr std::uint8_t kSignificantBits = 10U;
constexpr std::int64_t kExposureTimeNs = 8'000'000;
constexpr float kGain = 2.5F;
// Two non-overlapping planes: the first runs to byte 4, the second begins there and ends on the
// Tensor's last byte. Sized to the eight-byte Tensor the source operator loans, because a plane
// reaching past the allocation backing the Image is rejected before it can be forwarded.
const std::vector<schema::ImagePlaneLayout> kPlaneLayouts{
    schema::ImagePlaneLayout{0U, 4U, 4U},
    schema::ImagePlaneLayout{4U, 4U, 4U},
};

// Runtime envelope values, carried by EmitOptions rather than by the payload.
constexpr ClockId kCaptureClock{1U};
constexpr std::int64_t kCaptureTimestampNs = 987'654'321;
constexpr std::uint64_t kFrameId = 77U;
constexpr std::uint64_t kTraceId = 0xC0FFEEU;

/// @brief Author one Image descriptor with every field set.
[[nodiscard]] schema::ImageT make_source_descriptor() {
  schema::ImageT value{};
  value.header = std::make_shared<schema::HeaderT>();
  value.header->frame_id = std::string{kCoordinateFrame};
  value.header->capture_timestamp_ns = kHeaderCaptureTimestampNs;
  value.header->device_sequence = kDeviceSequence;
  value.height = 1;
  value.width = 8;
  value.encoding = schema::ImageEncoding_CUSTOM;
  value.color_space = schema::ColorSpace_REC2020_PQ;
  value.significant_bits = kSignificantBits;
  value.roi_offset_x = kRoiOffsetX;
  value.roi_offset_y = kSourceRoiOffsetY;
  value.plane_layouts = kPlaneLayouts;
  value.exposure_time_ns = kExposureTimeNs;
  value.gain = kGain;
  return value;
}

/// @brief Every authored descriptor field, copied out so a failure names the field that broke.
///
/// The fields are copied rather than the descriptor retained, because the Tensor a descriptor
/// references belongs to the sample and must not outlive it.
struct DescriptorFacts {
  bool has_header{};
  std::string frame_id;
  std::int64_t header_capture_timestamp_ns{};
  std::uint64_t device_sequence{};
  std::int32_t height{};
  std::int32_t width{};
  schema::ImageEncoding encoding{};
  schema::ColorSpace color_space{};
  std::uint8_t significant_bits{};
  std::int32_t roi_offset_x{};
  std::int32_t roi_offset_y{};
  std::vector<schema::ImagePlaneLayout> plane_layouts;
  std::int64_t exposure_time_ns{};
  float gain{};
};

/// @brief Copy the descriptor facts out of an observed sample.
[[nodiscard]] DescriptorFacts facts_of(const schema::ImageT& value) {
  DescriptorFacts facts{};
  facts.has_header = value.header != nullptr;
  if (facts.has_header) {
    facts.frame_id = value.header->frame_id;
    facts.header_capture_timestamp_ns = value.header->capture_timestamp_ns;
    facts.device_sequence = value.header->device_sequence;
  }
  facts.height = value.height;
  facts.width = value.width;
  facts.encoding = value.encoding;
  facts.color_space = value.color_space;
  facts.significant_bits = value.significant_bits;
  facts.roi_offset_x = value.roi_offset_x;
  facts.roi_offset_y = value.roi_offset_y;
  facts.plane_layouts = value.plane_layouts;
  facts.exposure_time_ns = value.exposure_time_ns;
  facts.gain = value.gain;
  return facts;
}

/// @brief What the sink saw, recorded field by field so a failure names the field that broke.
struct Observation {
  DescriptorFacts descriptor{};
  bool tensor_is_transform_output{};
  bool bytes_transformed{};
  std::int64_t capture_timestamp_ns{};
  ClockId source_clock_id{};
  std::uint64_t frame_id{};
  std::uint64_t trace_id{};
};

/// @brief Assert that every descriptor field survived the transform.
/// @param facts Descriptor facts as observed by the sink.
/// @param expected_roi_offset_y The one field the adjusting transform is allowed to change.
void expect_descriptor_survived(const DescriptorFacts& facts, std::int32_t expected_roi_offset_y) {
  EXPECT_TRUE(facts.has_header);
  EXPECT_EQ(facts.frame_id, kCoordinateFrame);
  EXPECT_EQ(facts.header_capture_timestamp_ns, kHeaderCaptureTimestampNs);
  EXPECT_EQ(facts.device_sequence, kDeviceSequence);
  EXPECT_EQ(facts.height, 1);
  EXPECT_EQ(facts.width, 8);
  EXPECT_EQ(facts.encoding, schema::ImageEncoding_CUSTOM);
  EXPECT_EQ(facts.color_space, schema::ColorSpace_REC2020_PQ);
  EXPECT_EQ(facts.significant_bits, kSignificantBits);
  EXPECT_EQ(facts.roi_offset_x, kRoiOffsetX);
  EXPECT_EQ(facts.roi_offset_y, expected_roi_offset_y);
  // Compared field by field because flatc generates no equality for a struct.
  ASSERT_EQ(facts.plane_layouts.size(), kPlaneLayouts.size());
  for (std::size_t plane = 0U; plane < kPlaneLayouts.size(); ++plane) {
    EXPECT_EQ(facts.plane_layouts[plane].offset_bytes(), kPlaneLayouts[plane].offset_bytes());
    EXPECT_EQ(facts.plane_layouts[plane].row_stride_bytes(),
              kPlaneLayouts[plane].row_stride_bytes());
    EXPECT_EQ(facts.plane_layouts[plane].size_bytes(), kPlaneLayouts[plane].size_bytes());
  }
  EXPECT_EQ(facts.exposure_time_ns, kExposureTimeNs);
  EXPECT_EQ(facts.gain, kGain);
}

/// @brief Cross-thread probe shared by the three operators of one scenario.
template <TransformMode kMode>
struct Probe {
  static void reset() {
    const std::lock_guard lock(mutex);
    transform_output_pointer = nullptr;
    observation = Observation{};
    observed = false;
    error.reset();
  }

  static void record_transform_output(const void* pointer) {
    const std::lock_guard lock(mutex);
    transform_output_pointer = pointer;
  }

  static void record_sink(const Sample<schema::ImageT>& sample) {
    const std::lock_guard lock(mutex);
    const Tensor* const tensor = sample.data.data.get();
    const std::uint8_t* const bytes = tensor == nullptr ? nullptr : tensor->data_as<std::uint8_t>();
    observation.descriptor = facts_of(sample.data);
    observation.tensor_is_transform_output =
        tensor != nullptr && tensor->data() == transform_output_pointer && tensor->nbytes() == 8 &&
        std::ranges::equal(tensor->shape_span(), kShape);
    observation.bytes_transformed =
        bytes != nullptr &&
        std::ranges::equal(std::span<const std::uint8_t>{bytes, kTransformedBytes.size()},
                           kTransformedBytes);
    observation.capture_timestamp_ns = sample.metadata.capture_timestamp_ns;
    observation.source_clock_id = sample.metadata.source_clock_id;
    observation.frame_id = sample.metadata.frame_id;
    observation.trace_id = sample.metadata.trace_id;
    observed = true;
    cv.notify_all();
  }

  static void fail(ErrorCode code) {
    {
      const std::lock_guard lock(mutex);
      error = code;
      observed = true;
    }
    cv.notify_all();
  }

  [[nodiscard]] static bool wait() {
    std::unique_lock lock(mutex);
    return cv.wait_for(lock, 5s, [] { return observed; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable cv;
  static inline const void* transform_output_pointer{};
  static inline Observation observation{};
  static inline bool observed{};
  static inline std::optional<ErrorCode> error;
};

/// @brief Declare the host Tensor layout and descriptor payload shared by both schema ports.
void declare_image_output(OperatorSpec& spec, Output<schema::ImageT>& port) {
  spec.output(port, "output")
      .max_emits_per_compute(1U)
      .produces_tensor(TensorOutputSpec{
          .representation = {.memory_kind = MemoryKind::kHost,
                             .dtype = DLDataType{kDLUInt, 8U, 1U},
                             .rank = 1U},
          .bounds = tensor_bounds(8U),
          .storage = TensorOutputStorage::kRuntimePool,
          // Transfer policy belongs to the Tensor contract. Declaring it in payload_options as
          // well is rejected as GRAPH_TENSOR_TRANSFER_SOURCE_CONFLICT, because the two would be
          // independent sources of truth for the same staging bound.
          .transfer =
              TensorTransferPolicy{
                  .foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded,
                  .max_staging_byte_span = 8U,
              },
      })
      .payload_options(PayloadOptions{
          .schema = schema_identity<schema::ImageT>(),
      });
}

/// @brief Declare the matching host Tensor input layout.
void declare_image_input(OperatorSpec& spec, Input<schema::ImageT>& port) {
  spec.input(port, "input")
      .queue_depth(1U)
      .expects_tensor(TensorInputSpec{
          .representation = {.memory_kind = MemoryKind::kHost,
                             .dtype = DLDataType{kDLUInt, 8U, 1U},
                             .rank = 1U},
      });
}

/// @brief Publish one fully populated Image with full provenance.
template <TransformMode kMode>
class SourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { declare_image_output(spec, output); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 1ms});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    if (emitted_) {
      return {};
    }
    emitted_ = true;

    auto loan = output.allocate_tensor(TensorLoanRequest{
        .shape = kShape,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
    });
    if (!loan) {
      return fail(loan.error().code);
    }
    const auto written = loan->write_host_and_commit([](const TensorOutputWriteGuard& writer) {
      std::ranges::copy(kSourceBytes, writer.data_as<std::uint8_t>());
    });
    if (!written) {
      return fail(written.error().code);
    }
    const auto emitted = output.emit_tensor(std::move(*loan),
                                            make_source_descriptor(),
                                            EmitOptions{
                                                .capture_time =
                                                    TimePoint{
                                                        .timestamp_ns = kCaptureTimestampNs,
                                                        .clock_id = kCaptureClock,
                                                    },
                                                .frame_id = kFrameId,
                                                .trace = TraceContext::root(kTraceId),
                                            });
    if (!emitted) {
      return fail(emitted.error().code);
    }
    return {};
  }

  Output<schema::ImageT> output;

 private:
  [[nodiscard]] static expected<void, Error> fail(ErrorCode code) {
    Probe<kMode>::fail(code);
    return make_unexpected(Error{code});
  }

  bool emitted_{};
};

/// @brief Replace the Tensor and forward the metadata, which is the behaviour under test.
template <TransformMode kMode>
class TransformOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    declare_image_input(spec, input);
    declare_image_output(spec, output);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      Probe<kMode>::fail(sample.error().code);
      return make_unexpected(std::move(sample).error());
    }
    const Tensor* const incoming = sample->data.data.get();
    if (incoming == nullptr || incoming->nbytes() != kSourceBytes.size()) {
      return fail(ErrorCode::kInvalidArgument);
    }

    auto loan = output.allocate_tensor(TensorLoanRequest{
        .shape = kShape,
        .dtype = DLDataType{kDLUInt, 8U, 1U},
    });
    if (!loan) {
      return fail(loan.error().code);
    }
    const std::span<const std::uint8_t> incoming_bytes{incoming->data_as<std::uint8_t>(),
                                                       kSourceBytes.size()};
    const auto written =
        loan->write_host_and_commit([incoming_bytes](const TensorOutputWriteGuard& writer) {
          Probe<kMode>::record_transform_output(writer.data());
          std::ranges::transform(
              incoming_bytes, writer.data_as<std::uint8_t>(), [](std::uint8_t byte) {
                return static_cast<std::uint8_t>(std::numeric_limits<std::uint8_t>::max() - byte);
              });
        });
    if (!written) {
      return fail(written.error().code);
    }

    const auto emitted = publish(std::move(*loan), *sample);
    if (!emitted) {
      return fail(emitted.error().code);
    }
    return {};
  }

  Input<schema::ImageT> input;
  Output<schema::ImageT> output;

 private:
  /// @brief Publish the replacement Tensor under the received metadata.
  [[nodiscard]] expected<void, Error> publish(TensorOutputLoan loan,
                                              const Sample<schema::ImageT>& sample) {
    if constexpr (kMode == TransformMode::kForwardVerbatim) {
      // The received descriptor goes straight back out. emit_tensor ignores its embedded Tensor
      // member and takes the loan's Tensor as authoritative, so replacing the pixels costs nothing
      // in metadata handling: no field is read, named, or copied by this operator.
      return output.emit_tensor(std::move(loan), sample.data, sensor_io::derive_from(sample));
    } else if constexpr (kMode == TransformMode::kAdjustOneField) {
      // Change one field and forward the rest. Copying the descriptor moves scalars, one shared_ptr
      // refcount for the header, and a two-element pitch vector; nothing proportional to the frame.
      // Only a scalar directly on Image is adjusted here on purpose: the copied header is *shared*
      // with the received sample, so an operator that needs to change a nested table field must
      // deep-copy that table rather than write through the aliased pointer.
      schema::ImageT descriptor = sample.data;
      descriptor.roi_offset_y = kAdjustedRoiOffsetY;
      return output.emit_tensor(std::move(loan), descriptor, sensor_io::derive_from(sample));
    } else {
      // The mistake this file exists to make visible. The payload metadata still forwards, because
      // it rides in the descriptor, but the envelope is rebuilt from these empty options and the
      // capture time, clock domain, frame identity, and trace are gone without a diagnostic.
      return output.emit_tensor(std::move(loan), sample.data, EmitOptions{});
    }
  }

  [[nodiscard]] static expected<void, Error> fail(ErrorCode code) {
    Probe<kMode>::fail(code);
    return make_unexpected(Error{code});
  }
};

/// @brief Record the descriptor, Tensor, and envelope that survived the transform.
template <TransformMode kMode>
class SinkOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { declare_image_input(spec, input); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      Probe<kMode>::fail(sample.error().code);
      return make_unexpected(std::move(sample).error());
    }
    Probe<kMode>::record_sink(*sample);
    return {};
  }

  Input<schema::ImageT> input;
};

// Graph topology:
//   source --[descriptor-backed ImageT]--> transform --[descriptor-backed ImageT]--> sink
template <TransformMode kMode>
[[nodiscard]] ExecutionPlan make_passthrough_plan() {
  Graph graph{"sensor-io-descriptor-passthrough"};
  auto source = graph.op<SourceOperator<kMode>>("source");
  auto transform = graph.op<TransformOperator<kMode>>("transform");
  auto sink = graph.op<SinkOperator<kMode>>("sink");
  graph.add_flow(source->output, transform->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(transform->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.set_default_clock(graph.add_clock<RealtimeClock>("sensor-io-passthrough-clock"));
  return compile(graph);
}

/// @brief Run one scenario to completion and return what the sink saw.
template <TransformMode kMode>
[[nodiscard]] Observation run_scenario() {
  Probe<kMode>::reset();
  const ExecutionPlan plan = make_passthrough_plan<kMode>();
  EXPECT_TRUE(plan.ok()) << plan.json();
  if (!plan.ok()) {
    return Observation{};
  }

  RunSession session = run_async(plan);
  const bool observed = Probe<kMode>::wait();
  session.request_stop();
  session.wait();

  EXPECT_TRUE(observed) << (session.diagnostics().empty() ? "no runtime diagnostic"
                                                          : session.diagnostics().front().token);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);

  const std::lock_guard lock(Probe<kMode>::mutex);
  EXPECT_FALSE(Probe<kMode>::error.has_value());
  return Probe<kMode>::observation;
}

// An operator that replaces the payload forwards every descriptor field by handing the received
// descriptor back to emit_tensor, and forwards the envelope with derive_from. It names no field of
// either, so neither forwarding path has to change when a schema grows a field.
// REQ: SENSOR-IO-007
TEST(SensorIoDescriptorPassthrough, ForwardsEveryMetadataFieldWithoutNamingOne) {
  const Observation observation = run_scenario<TransformMode::kForwardVerbatim>();

  expect_descriptor_survived(observation.descriptor, kSourceRoiOffsetY);
  EXPECT_TRUE(observation.tensor_is_transform_output);
  EXPECT_TRUE(observation.bytes_transformed);
  EXPECT_EQ(observation.capture_timestamp_ns, kCaptureTimestampNs);
  EXPECT_EQ(observation.source_clock_id, kCaptureClock);
  EXPECT_EQ(observation.frame_id, kFrameId);
  EXPECT_EQ(observation.trace_id, kTraceId);
}

// The partial-update case: one descriptor field changes and every other field forwards untouched,
// still without the operator enumerating what it is forwarding.
// REQ: SENSOR-IO-007
TEST(SensorIoDescriptorPassthrough, ForwardsUnmodifiedFieldsWhileAdjustingOne) {
  const Observation observation = run_scenario<TransformMode::kAdjustOneField>();

  expect_descriptor_survived(observation.descriptor, kAdjustedRoiOffsetY);
  EXPECT_TRUE(observation.tensor_is_transform_output);
  EXPECT_TRUE(observation.bytes_transformed);
  EXPECT_EQ(observation.capture_timestamp_ns, kCaptureTimestampNs);
  EXPECT_EQ(observation.source_clock_id, kCaptureClock);
  EXPECT_EQ(observation.frame_id, kFrameId);
  EXPECT_EQ(observation.trace_id, kTraceId);
}

// Emitting with default options is what derive_from exists to prevent. The payload metadata still
// arrives, which is exactly why the loss is easy to miss: the descriptor looks complete while the
// envelope that every temporal selector reads has been silently emptied.
// REQ: SENSOR-IO-007
TEST(SensorIoDescriptorPassthrough, DefaultEmitOptionsSilentlyDropProvenance) {
  const Observation observation = run_scenario<TransformMode::kDropsProvenance>();

  expect_descriptor_survived(observation.descriptor, kSourceRoiOffsetY);
  EXPECT_TRUE(observation.tensor_is_transform_output);
  EXPECT_TRUE(observation.bytes_transformed);
  EXPECT_EQ(observation.capture_timestamp_ns, 0);
  EXPECT_FALSE(observation.source_clock_id.valid());
  EXPECT_EQ(observation.frame_id, 0U);
  EXPECT_EQ(observation.trace_id, 0U);
}

}  // namespace
}  // namespace holoscan
