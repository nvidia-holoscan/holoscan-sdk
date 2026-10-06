// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// HolovizOp's temporal contract as the compiler lowers it: a LayerTrigger becomes an OnEach
// trigger on the chosen layer's port with no selector on that port, every other layer is sampled
// by freshness (LatestWithin, ReuseLast for images), and a ClockTrigger keeps the fixed-period
// OnClock form. Compile-only — the operator is never started, so no window, adapter, or GPU is
// needed — plus the construction-time validation of the trigger and freshness parameters.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/temporal_plan.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

namespace holoscan::visualization {
namespace {

using namespace std::chrono_literals;

constexpr std::uint32_t kDim = 64;
constexpr std::chrono::nanoseconds kMaxInputAge = 100ms;

/// Clock-driven host Tensor source shaped for one HolovizOp layer kind; compute() never runs.
/// Feeds an image-kind layer, whose port is Input<schema::ImageT>. The contract test only cares
/// that an edge exists and is admitted, so the descriptor carries CUSTOM encoding: it states no
/// organization and is exempt from HolovizOp's encoding check against the declared format.
class ImageLayerSource final : public holoscan::Operator<> {
 public:
  ImageLayerSource(DLDataType dtype, std::uint32_t rank, std::uint64_t max_bytes)
      : dtype_(dtype), rank_(rank), max_bytes_(max_bytes) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(frame, "frame")
        .max_emits_per_compute(1U)
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = dtype_,
                               .rank = rank_},
            .bounds = holoscan::tensor_bounds(max_bytes_),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        })
        .payload_options(holoscan::PayloadOptions{
            .schema = holoscan::schema_identity<holoscan::schema::ImageT>(),
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 16ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Output<holoscan::schema::ImageT> frame;

 private:
  DLDataType dtype_;
  std::uint32_t rank_;
  std::uint64_t max_bytes_;
};

class LayerSource final : public holoscan::Operator<> {
 public:
  LayerSource(DLDataType dtype, std::uint32_t rank, std::uint64_t max_bytes)
      : dtype_(dtype), rank_(rank), max_bytes_(max_bytes) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(frame, "frame")
        .max_emits_per_compute(1U)
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = dtype_,
                               .rank = rank_},
            .bounds = holoscan::tensor_bounds(max_bytes_),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = 16ms});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Output<holoscan::Tensor> frame;

 private:
  DLDataType dtype_;
  std::uint32_t rank_;
  std::uint64_t max_bytes_;
};

/// Two-layer stack: "video" (image, layer 0) over "overlay" (geometry, layer 1).
[[nodiscard]] LayerStack two_layer_stack() {
  LayerStack stack;
  stack.add_image_layer(
      "video",
      ImageLayerDesc{.format = ImageFormat::kR8G8B8A8Unorm, .max_width = kDim, .max_height = kDim});
  stack.add_geometry_layer(
      "overlay",
      GeometryLayerDesc{.topologies = {GeometryTopology::kRectangleList}, .max_primitives = 16});
  return stack;
}

/// Two-layer stack: "video" (image, layer 0) under "labels" (text, layer 1).
[[nodiscard]] LayerStack text_layer_stack() {
  LayerStack stack;
  stack.add_image_layer(
      "video",
      ImageLayerDesc{.format = ImageFormat::kR8G8B8A8Unorm, .max_width = kDim, .max_height = kDim});
  stack.add_text_layer("labels", TextLayerDesc{.max_glyphs = 64});
  return stack;
}

/// Compile a source -> HolovizOp graph for the given trigger, mirroring gray_scale_example's
/// composition (explicit display partition, drop-oldest depth-1 edges, RealtimeClock default).
[[nodiscard]] holoscan::ExecutionPlan compile_two_layer_graph(FrameTrigger trigger) {
  holoscan::Graph graph{"holoviz-op-contract"};
  const auto video = graph.op<ImageLayerSource>(
      "video_src", DLDataType{kDLUInt, 8U, 1U}, 3U, std::uint64_t{kDim} * kDim * 4U);
  const auto overlay = graph.op<LayerSource>(
      "overlay_src", DLDataType{kDLFloat, 32U, 1U}, 1U, std::uint64_t{16U * 4U * sizeof(float)});
  const auto viz = graph.op<HolovizOp>(
      "holoviz", two_layer_stack(), WindowSinkDesc{.title = "contract"}, trigger, kMaxInputAge);
  const holoscan::ConnectionOptions mailbox{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                            .queue_depth = 1U};
  graph.add_flow(video->frame, viz->image_layer_input(0), mailbox);
  graph.add_flow(overlay->frame, viz->tensor_layer_input(1), mailbox);
  graph.partition("display").add(viz);
  graph.auto_partition_remaining();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  return holoscan::compile(graph);
}

/// Four layers in an order that separates declared index from physical slot: geometry (0),
/// image (1), geometry (2), image (3). The second image layer is the second image-kind slot even
/// though it is the fourth declared layer.
[[nodiscard]] LayerStack mixed_order_stack() {
  LayerStack stack;
  stack.add_geometry_layer(
      "boxes",
      GeometryLayerDesc{.topologies = {GeometryTopology::kRectangleList}, .max_primitives = 16});
  stack.add_image_layer(
      "video",
      ImageLayerDesc{.format = ImageFormat::kR8G8B8A8Unorm, .max_width = kDim, .max_height = kDim});
  stack.add_geometry_layer(
      "marks",
      GeometryLayerDesc{.topologies = {GeometryTopology::kCrossList}, .max_primitives = 8});
  stack.add_image_layer(
      "inset",
      ImageLayerDesc{.format = ImageFormat::kR8Unorm, .max_width = kDim, .max_height = kDim});
  return stack;
}

/// Compile the mixed-order stack with every edge wired through the public declared-index
/// accessors, the way an application does it.
[[nodiscard]] holoscan::ExecutionPlan compile_mixed_order_graph(FrameTrigger trigger) {
  holoscan::Graph graph{"holoviz-op-mixed-order"};
  const auto boxes = graph.op<LayerSource>(
      "boxes_src", DLDataType{kDLFloat, 32U, 1U}, 1U, std::uint64_t{16U * 4U * sizeof(float)});
  const auto video = graph.op<ImageLayerSource>(
      "video_src", DLDataType{kDLUInt, 8U, 1U}, 3U, std::uint64_t{kDim} * kDim * 4U);
  const auto marks = graph.op<LayerSource>(
      "marks_src", DLDataType{kDLFloat, 32U, 1U}, 1U, std::uint64_t{8U * 3U * sizeof(float)});
  const auto inset = graph.op<ImageLayerSource>(
      "inset_src", DLDataType{kDLUInt, 8U, 1U}, 2U, std::uint64_t{kDim} * kDim);
  const auto viz = graph.op<HolovizOp>(
      "holoviz", mixed_order_stack(), WindowSinkDesc{.title = "mixed"}, trigger, kMaxInputAge);
  const holoscan::ConnectionOptions mailbox{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                            .queue_depth = 1U};
  graph.add_flow(boxes->frame, viz->tensor_layer_input(0), mailbox);
  graph.add_flow(video->frame, viz->image_layer_input(1), mailbox);
  graph.add_flow(marks->frame, viz->tensor_layer_input(2), mailbox);
  graph.add_flow(inset->frame, viz->image_layer_input(3), mailbox);
  graph.partition("display").add(viz);
  graph.auto_partition_remaining();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  return holoscan::compile(graph);
}

/// A stack of \p count layers of one kind, every port wired through the matching public accessor.
[[nodiscard]] holoscan::ExecutionPlan compile_uniform_graph(std::size_t count, bool image_kind) {
  holoscan::Graph graph{"holoviz-op-uniform"};
  LayerStack stack;
  for (std::size_t i = 0; i < count; ++i) {
    const std::string name = "layer" + std::to_string(i);
    if (image_kind) {
      stack.add_image_layer(
          name,
          ImageLayerDesc{
              .format = ImageFormat::kR8G8B8A8Unorm, .max_width = kDim, .max_height = kDim});
    } else {
      stack.add_geometry_layer(
          name,
          GeometryLayerDesc{.topologies = {GeometryTopology::kLineList}, .max_primitives = 4});
    }
  }
  const auto viz = graph.op<HolovizOp>(
      "holoviz", std::move(stack), WindowSinkDesc{.title = "uniform"}, ClockTrigger{20ms});
  for (std::size_t i = 0; i < count; ++i) {
    const std::string name = "src" + std::to_string(i);
    if (image_kind) {
      const auto source = graph.op<ImageLayerSource>(
          name, DLDataType{kDLUInt, 8U, 1U}, 3U, std::uint64_t{kDim} * kDim * 4U);
      graph.add_flow(source->frame, viz->image_layer_input(i));
    } else {
      const auto source = graph.op<LayerSource>(
          name, DLDataType{kDLFloat, 32U, 1U}, 1U, std::uint64_t{4U * 4U * sizeof(float)});
      graph.add_flow(source->frame, viz->tensor_layer_input(i));
    }
  }
  graph.partition("display").add(viz);
  graph.auto_partition_remaining();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  return holoscan::compile(graph);
}

/// Same graph shape over text_layer_stack(): the text source emits uint8 {N, 16 + L} records.
[[nodiscard]] holoscan::ExecutionPlan compile_text_layer_graph(FrameTrigger trigger) {
  holoscan::Graph graph{"holoviz-op-text-contract"};
  const auto video = graph.op<ImageLayerSource>(
      "video_src", DLDataType{kDLUInt, 8U, 1U}, 3U, std::uint64_t{kDim} * kDim * 4U);
  const auto labels = graph.op<LayerSource>(
      "labels_src", DLDataType{kDLUInt, 8U, 1U}, 2U, std::uint64_t{4U * (16U + 32U)});
  const auto viz = graph.op<HolovizOp>(
      "holoviz", text_layer_stack(), WindowSinkDesc{.title = "contract"}, trigger, kMaxInputAge);
  const holoscan::ConnectionOptions mailbox{.queue_policy = holoscan::QueuePolicy::kDropOldest,
                                            .queue_depth = 1U};
  graph.add_flow(video->frame, viz->image_layer_input(0), mailbox);
  graph.add_flow(labels->frame, viz->tensor_layer_input(1), mailbox);
  graph.partition("display").add(viz);
  graph.auto_partition_remaining();
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  return holoscan::compile(graph);
}

/// Plan-local identity of the input port named \p name (layer port names are unique here).
[[nodiscard]] holoscan::PortId input_port(const holoscan::ExecutionPlan& plan,
                                          std::string_view name) {
  const auto ports = plan.ports();
  const auto it = std::find_if(ports.begin(), ports.end(), [&](const auto& port) {
    return port.direction == holoscan::PortDirection::kInput && port.local_name == name;
  });
  return it == ports.end() ? holoscan::PortId{} : it->id;
}

[[nodiscard]] std::optional<holoscan::TemporalTriggerPlanRecord> trigger_declaring(
    const holoscan::ExecutionPlan& plan, holoscan::PortId input) {
  for (const auto& record : plan.temporal_triggers()) {
    if (record.trigger_input == input ||
        std::find(record.declared_inputs.begin(), record.declared_inputs.end(), input) !=
            record.declared_inputs.end()) {
      return record;
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<holoscan::InputSelectionPlanRecord> selection_for(
    const holoscan::ExecutionPlan& plan, holoscan::PortId input) {
  for (const auto& record : plan.input_selections()) {
    if (record.input == input) {
      return record;
    }
  }
  return std::nullopt;
}

TEST(HolovizOpContract, LayerTriggerLowersToOnEachOnTheTriggerLayer) {
  const auto plan = compile_two_layer_graph(LayerTrigger{0});
  ASSERT_TRUE(plan.ok()) << plan.json();

  const auto video = input_port(plan, "video");
  const auto overlay = input_port(plan, "overlay");
  ASSERT_TRUE(video);
  ASSERT_TRUE(overlay);

  const auto trigger = trigger_declaring(plan, video);
  ASSERT_TRUE(trigger.has_value());
  EXPECT_EQ(trigger->kind, holoscan::TemporalRecordTriggerKind::kEach);
  EXPECT_EQ(trigger->trigger_input, video);
  EXPECT_EQ(trigger->period_ns, 0);
  EXPECT_NE(std::find(trigger->presence_guaranteed_inputs.begin(),
                      trigger->presence_guaranteed_inputs.end(),
                      video),
            trigger->presence_guaranteed_inputs.end());

  // The OnEach driver carries no selector; the side layer is sampled by freshness and skipped
  // (SafeDefault) when nothing fresh exists, because it is a geometry layer.
  EXPECT_FALSE(selection_for(plan, video).has_value());
  const auto side = selection_for(plan, overlay);
  ASSERT_TRUE(side.has_value());
  EXPECT_EQ(side->selector, holoscan::InputSelectorKind::kLatestWithin);
  EXPECT_EQ(side->max_age_ns, kMaxInputAge.count());
  EXPECT_EQ(side->missing, holoscan::TemporalRecordMissingKind::kSafeDefault);
}

TEST(HolovizOpContract, LayerTriggerMayNameAnyDeclaredLayer) {
  const auto plan = compile_two_layer_graph(LayerTrigger{1});
  ASSERT_TRUE(plan.ok()) << plan.json();

  const auto video = input_port(plan, "video");
  const auto overlay = input_port(plan, "overlay");
  ASSERT_TRUE(video);
  ASSERT_TRUE(overlay);
  const auto trigger = trigger_declaring(plan, overlay);
  ASSERT_TRUE(trigger.has_value());
  EXPECT_EQ(trigger->kind, holoscan::TemporalRecordTriggerKind::kEach);
  EXPECT_EQ(trigger->trigger_input, overlay);

  // The image layer is now a side input: freshest within the bound, holding its last sample for
  // the same bound when nothing new arrived.
  EXPECT_FALSE(selection_for(plan, overlay).has_value());
  const auto side = selection_for(plan, video);
  ASSERT_TRUE(side.has_value());
  EXPECT_EQ(side->selector, holoscan::InputSelectorKind::kLatestWithin);
  EXPECT_EQ(side->max_age_ns, kMaxInputAge.count());
  EXPECT_EQ(side->missing, holoscan::TemporalRecordMissingKind::kReuseLast);
  EXPECT_EQ(side->reuse_max_age_ns, kMaxInputAge.count());
}

TEST(HolovizOpContract, ClockTriggerLowersToOnClockWithEveryLayerSelected) {
  const auto plan = compile_two_layer_graph(ClockTrigger{20ms});
  ASSERT_TRUE(plan.ok()) << plan.json();

  const auto video = input_port(plan, "video");
  const auto overlay = input_port(plan, "overlay");
  ASSERT_TRUE(video);
  ASSERT_TRUE(overlay);
  const auto trigger = trigger_declaring(plan, video);
  ASSERT_TRUE(trigger.has_value());
  EXPECT_EQ(trigger->kind, holoscan::TemporalRecordTriggerKind::kClock);
  EXPECT_FALSE(trigger->trigger_input);
  EXPECT_EQ(trigger->period_ns, std::chrono::nanoseconds{20ms}.count());

  const auto image = selection_for(plan, video);
  ASSERT_TRUE(image.has_value());
  EXPECT_EQ(image->selector, holoscan::InputSelectorKind::kLatestWithin);
  EXPECT_EQ(image->missing, holoscan::TemporalRecordMissingKind::kReuseLast);
  const auto geometry = selection_for(plan, overlay);
  ASSERT_TRUE(geometry.has_value());
  EXPECT_EQ(geometry->selector, holoscan::InputSelectorKind::kLatestWithin);
  EXPECT_EQ(geometry->missing, holoscan::TemporalRecordMissingKind::kSafeDefault);
}

// The declared-index accessors resolve through the kind pools: a geometry layer declared first
// must not shift the image layers' slots, and a LayerTrigger on an image layer must lower to
// OnEach on that layer's Input<schema::ImageT> port.
TEST(HolovizOpContract, MixedOrderStackWiresThroughPublicAccessors) {
  const auto plan = compile_mixed_order_graph(LayerTrigger{1});
  ASSERT_TRUE(plan.ok()) << plan.json();

  const auto boxes = input_port(plan, "boxes");
  const auto video = input_port(plan, "video");
  const auto marks = input_port(plan, "marks");
  const auto inset = input_port(plan, "inset");
  ASSERT_TRUE(boxes);
  ASSERT_TRUE(video);
  ASSERT_TRUE(marks);
  ASSERT_TRUE(inset);

  const auto trigger = trigger_declaring(plan, video);
  ASSERT_TRUE(trigger.has_value());
  EXPECT_EQ(trigger->kind, holoscan::TemporalRecordTriggerKind::kEach);
  EXPECT_EQ(trigger->trigger_input, video);
  EXPECT_FALSE(selection_for(plan, video).has_value());

  const auto other_image = selection_for(plan, inset);
  ASSERT_TRUE(other_image.has_value());
  EXPECT_EQ(other_image->selector, holoscan::InputSelectorKind::kLatestWithin);
  EXPECT_EQ(other_image->missing, holoscan::TemporalRecordMissingKind::kReuseLast);
  for (const auto geometry_port : {boxes, marks}) {
    const auto geometry = selection_for(plan, geometry_port);
    ASSERT_TRUE(geometry.has_value());
    EXPECT_EQ(geometry->selector, holoscan::InputSelectorKind::kLatestWithin);
    EXPECT_EQ(geometry->missing, holoscan::TemporalRecordMissingKind::kSafeDefault);
  }
}

// Any kind mix of up to kMaxLayerPorts layers is legal: eight image layers and eight geometry
// layers both compile with every port bound through the public accessors.
TEST(HolovizOpContract, EveryLegalPartitionOfTheLayerBudgetCompiles) {
  const auto all_images = compile_uniform_graph(kMaxLayerPorts, true);
  EXPECT_TRUE(all_images.ok()) << all_images.json();
  const auto all_geometry = compile_uniform_graph(kMaxLayerPorts, false);
  EXPECT_TRUE(all_geometry.ok()) << all_geometry.json();
}

// A text layer is a side layer like geometry: freshest within the bound, skipped (SafeDefault,
// no hold-last) when nothing fresh exists — and it may itself be the LayerTrigger. The port's
// typed contract (host uint8 rank 2) must accept a matching producer at compile time.
TEST(HolovizOpContract, TextLayerIsSampledByFreshnessAndMayBeTheTrigger) {
  const auto plan = compile_text_layer_graph(LayerTrigger{0});
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto labels = input_port(plan, "labels");
  ASSERT_TRUE(labels);
  const auto side = selection_for(plan, labels);
  ASSERT_TRUE(side.has_value());
  EXPECT_EQ(side->selector, holoscan::InputSelectorKind::kLatestWithin);
  EXPECT_EQ(side->max_age_ns, kMaxInputAge.count());
  EXPECT_EQ(side->missing, holoscan::TemporalRecordMissingKind::kSafeDefault);

  const auto text_driven = compile_text_layer_graph(LayerTrigger{1});
  ASSERT_TRUE(text_driven.ok()) << text_driven.json();
  const auto trigger_port = input_port(text_driven, "labels");
  ASSERT_TRUE(trigger_port);
  const auto trigger = trigger_declaring(text_driven, trigger_port);
  ASSERT_TRUE(trigger.has_value());
  EXPECT_EQ(trigger->kind, holoscan::TemporalRecordTriggerKind::kEach);
  EXPECT_EQ(trigger->trigger_input, trigger_port);
  EXPECT_FALSE(selection_for(text_driven, trigger_port).has_value());
}

TEST(HolovizOpContract, EmptyStackCompilesWithClockTrigger) {
  holoscan::Graph graph{"holoviz-op-empty"};
  (void)graph.op<HolovizOp>(
      "holoviz", LayerStack{}, WindowSinkDesc{.title = "empty"}, ClockTrigger{});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  const auto plan = holoscan::compile(graph);
  EXPECT_TRUE(plan.ok()) << plan.json();
}

TEST(HolovizOpContract, TriggerAccessorReturnsTheConstructedTrigger) {
  const HolovizOp layer_op{two_layer_stack(), WindowSinkDesc{.title = "t"}, LayerTrigger{1}};
  ASSERT_TRUE(std::holds_alternative<LayerTrigger>(layer_op.trigger()));
  EXPECT_EQ(std::get<LayerTrigger>(layer_op.trigger()).layer, 1U);

  const HolovizOp clock_op{LayerStack{}, WindowSinkDesc{.title = "t"}, ClockTrigger{5ms}};
  ASSERT_TRUE(std::holds_alternative<ClockTrigger>(clock_op.trigger()));
  EXPECT_EQ(std::get<ClockTrigger>(clock_op.trigger()).period, 5ms);
}

#if GTEST_HAS_DEATH_TEST
TEST(HolovizOpContractDeathTest, RejectsInvalidTriggerAndFreshnessAtConstruction) {
  // A LayerTrigger must name a declared layer; the default LayerTrigger{0} on an empty stack is
  // the same mistake.
  EXPECT_DEATH((HolovizOp{two_layer_stack(), WindowSinkDesc{.title = "t"}, LayerTrigger{2}}),
               "LayerTrigger layer 2 is out of range");
  EXPECT_DEATH((HolovizOp{LayerStack{}, WindowSinkDesc{.title = "t"}}),
               "LayerTrigger layer 0 is out of range");
  // Temporal parameters must be positive.
  EXPECT_DEATH((HolovizOp{two_layer_stack(), WindowSinkDesc{.title = "t"}, ClockTrigger{0ns}}),
               "ClockTrigger period must be > 0 ns");
  EXPECT_DEATH((HolovizOp{two_layer_stack(), WindowSinkDesc{.title = "t"}, LayerTrigger{0}, 0ns}),
               "max_input_age must be > 0 ns");
  // The aggregate layer budget is the only capacity limit; a ninth layer of any kind trips it.
  LayerStack nine_layers;
  for (std::size_t i = 0; i <= kMaxLayerPorts; ++i) {
    nine_layers.add_image_layer(
        "layer" + std::to_string(i),
        ImageLayerDesc{.format = ImageFormat::kR8Unorm, .max_width = kDim, .max_height = kDim});
  }
  EXPECT_DEATH((HolovizOp{std::move(nine_layers), WindowSinkDesc{.title = "t"}, ClockTrigger{}}),
               "exceeds kMaxLayerPorts");
}
#endif

}  // namespace
}  // namespace holoscan::visualization

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
