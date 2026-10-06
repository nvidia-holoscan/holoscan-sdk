/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <typeinfo>
#include <utility>
#include <vector>

#include <holoscan/authoring/parameter_materializer.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/testing/operator_test_harness.hpp>
#include "provider.hpp"

// REQ: OPERATOR-073
bool tensor_consumer() {
  using namespace holoscan;
  const auto& row = test_tensor_operator_provider();
  const std::array assignments{
      authoring::ParameterAssignment{"factor", authoring::ParameterValue::from(std::int64_t{2})}};
  auto materialized =
      authoring::materialize_parameters(*row.param_type, row.param_schema, assignments);
  if (!materialized)
    return false;
  auto params = std::make_shared<authoring::MaterializedParameters>(std::move(*materialized));
  testing::NativeOperatorTest h{[&row, params](Graph& graph) {
    auto op = row.add_to_graph(graph, "operator", nullptr);
    row.set_params(op, *params);
    return op;
  }};
  const std::array<std::int64_t, 1> shape{1};
  auto input = make_tensor(shape,
                           DLDataType{kDLInt, 32, 1},
                           std::nullopt,
                           MemoryStorageType::kSystem,
                           std::make_shared<UnboundedAllocator>());
  if (!input)
    return false;
  auto writable = input->mutable_data();
  if (!writable)
    return false;
  *static_cast<std::int32_t*>(*writable) = 21;
  h.stage_input("input", test_tensor_input(*input));
  h.stage_input("input", testing::staged_values<Tensor>({std::move(*input)}));
  h.declare_capture("output", test_tensor_capture());
  h.input_queue_depth("input", 4U);
  h.input_flow("input", ConnectionOptions{.queue_depth = 4U});
  h.output_flow("output", ConnectionOptions{.queue_depth = 4U});
  h.configure_graph([](testing::NativeTestGraphView& view) {
    view.graph().configure(view.input_flow<Tensor>("input")).max_inflight_samples(4U);
    view.graph().configure(view.output_flow<Tensor>("output")).max_inflight_samples(4U);
  });
  const auto result = h.run({});
  if (!result.ok()) {
    std::cerr << result.describe_failure() << '\n';
    return false;
  }
  const auto snapshots = result.snapshots<Tensor>("output");
  if (snapshots.size() != 2U)
    return false;
  const auto& snapshot = snapshots.front();
  if (snapshot.bytes().size() != sizeof(std::int32_t))
    return false;
  std::int32_t value{};
  std::memcpy(&value, snapshot.bytes().data(), sizeof(value));
  return value == 42;
}

// REQ: OPERATOR-058
bool clock_consumer() {
  using namespace holoscan;
  using namespace std::chrono_literals;
  const auto& row = test_clock_operator_provider();
  testing::NativeOperatorTest h{
      [&row](Graph& graph) { return row.add_to_graph(graph, "operator", nullptr); }};
  h.stage_input(
      "input",
      testing::staged_samples<std::int32_t>({testing::sample_at(std::int32_t{42}, 900ms)}));
  h.declare_capture("output", testing::value_capture<std::int32_t>(4U));
  const auto result = h.run_controlled_clock(testing::ControlledClockRun{.tick = 1000ms});
  if (!result.ok()) {
    std::cerr << result.describe_failure() << '\n';
    return false;
  }
  return result.termination() == RunTermination::kStopped &&
         result.clock_outcome() == testing::ClockTestOutcome::kRetired &&
         result.clock_selection("input").state == InputState::kSelected &&
         result.value<std::int32_t>("output") == 42;
}

namespace {
template <typename Exception, typename Action>
bool rejects_named(Action&& action, std::string_view name) {
  try {
    std::forward<Action>(action)();
  } catch (const Exception& error) {
    return std::string_view{error.what()}.find(name) != std::string_view::npos;
  }
  return false;
}

template <holoscan::PublishedPayload T>
void qualify_helper_ports(holoscan::testing::NativeOperatorTest& fixture) {
  fixture.configure_graph([](holoscan::testing::NativeTestGraphView& view) {
    static_cast<void>(view.source_output<T>("input"));
    static_cast<void>(view.capture_input<T>("output"));
    view.graph().configure(view.input_flow<T>("input")).max_inflight_samples(4U);
    view.graph().configure(view.output_flow<T>("output")).max_inflight_samples(4U);
  });
}

bool native_type_witnesses() {
  const bool same_types = test_pod_native_type() == typeid(InstalledTestPod) &&
                          test_schema_native_type() == typeid(holoscan::schema::StringT) &&
                          test_packet_native_type() == typeid(holoscan::schema::TensorPacketT) &&
                          test_image_native_type() == typeid(holoscan::schema::ImageT);
  if (!same_types)
    return false;
#if TEST_OPERATOR_PROVIDER_SHARED && defined(__ELF__)
  // Hidden provider and consumer each emit their own native RTTI objects. Value equality above
  // must hold despite these distinct addresses, exercising the actual cross-module ABI case.
  if (&test_pod_native_type() == &typeid(InstalledTestPod) ||
      &test_schema_native_type() == &typeid(holoscan::schema::StringT) ||
      &test_packet_native_type() == &typeid(holoscan::schema::TensorPacketT) ||
      &test_image_native_type() == &typeid(holoscan::schema::ImageT)) {
    std::cerr << "hidden shared provider did not retain independent native RTTI witnesses\n";
    return false;
  }
#endif
  return true;
}

// REQ: OPERATOR-064
bool pod_consumer(bool provider_first, bool grouped) {
  using namespace holoscan;
  const auto& row = test_pod_operator_provider();
  testing::NativeOperatorTest h{
      [&row](Graph& graph) { return row.add_to_graph(graph, "operator", nullptr); }};
  const auto stage = [&](bool from_provider, std::uint32_t value) {
    auto input = from_provider ? test_pod_input({.value = value})
                               : testing::staged_values<InstalledTestPod>({{.value = value}});
    if (grouped) {
      h.push_frame(testing::TestFrame{.fields = {{.name = "input", .value = std::move(input)}}});
    } else {
      h.stage_input("input", std::move(input));
    }
  };
  stage(provider_first, 11U);
  const auto wrong = [&] {
    auto input = testing::staged_values<InstalledOtherPod>({{.value = 99U}});
    if (grouped) {
      h.push_frame(testing::TestFrame{.fields = {{.name = "input", .value = std::move(input)}}});
    } else {
      h.stage_input("input", std::move(input));
    }
  };
  // Both POD declarations have the same semantic name/version/size/alignment/field layout.
  if (!rejects_named<RuntimeError>(wrong, "input"))
    return false;
  stage(!provider_first, 22U);
  h.declare_capture(
      "output", provider_first ? test_pod_capture() : testing::value_capture<InstalledTestPod>(4U));
  qualify_helper_ports<InstalledTestPod>(h);
  const auto result = h.run();
  if (!result.ok()) {
    std::cerr << result.describe_failure() << '\n';
    return false;
  }
  const auto values = result.values<InstalledTestPod>("output");
  if (values.size() != 2U || values[0].value != 11U || values[1].value != 22U)
    return false;
  return rejects_named<std::invalid_argument>(
      [&] { static_cast<void>(result.samples<InstalledOtherPod>("output")); }, "output");
}

// REQ: OPERATOR-064
bool schema_consumer(bool provider_first) {
  using namespace holoscan;
  using Native = schema::StringT;
  const auto& row = test_schema_operator_provider();
  testing::NativeOperatorTest h{
      [&row](Graph& graph) { return row.add_to_graph(graph, "operator", nullptr); }};
  Native first;
  first.data = "first owning schema";
  Native second;
  second.data = "second owning schema";
  auto first_input =
      provider_first ? test_schema_input(first) : testing::staged_values<Native>({first});
  auto second_input =
      provider_first ? testing::staged_values<Native>({second}) : test_schema_input(second);
  first.data = "changed after public factory";
  second.data = "changed after public factory";
  h.stage_input("input", std::move(first_input));
  h.stage_input("input", std::move(second_input));
  h.declare_capture("output",
                    provider_first ? test_schema_capture() : testing::value_capture<Native>(4U));
  qualify_helper_ports<Native>(h);
  const auto result = h.run({.timeout = std::chrono::seconds{5}});
  if (!result.ok()) {
    std::cerr << result.describe_failure() << '\n';
    return false;
  }
  const auto values = result.values<Native>("output");
  if (values.size() != 2U || values[0].data != "first owning schema" ||
      values[1].data != "second owning schema")
    return false;
  Native next;
  next.data = "later run";
  h.stage_input("input", test_schema_input(std::move(next)));
  const auto later = h.run();
  return later.ok() && later.value<Native>("output").data == "later run" &&
         result.values<Native>("output")[0].data == "first owning schema";
}

// REQ: OPERATOR-057
bool packet_consumer(bool provider_adapters) {
  using namespace holoscan;
  using Native = schema::TensorPacketT;
  static_assert(testing::TensorSnapshotPayload<Native>);
  static_assert(std::same_as<testing::Snapshot<Native>, testing::SchemaSnapshot<Native>>);
  const auto& row = test_packet_operator_provider();
  testing::NativeOperatorTest h{
      [&row](Graph& graph) { return row.add_to_graph(graph, "operator", nullptr); }};
  const std::array<std::int64_t, 1> shape{1};
  auto tensor = make_tensor(shape,
                            DLDataType{kDLInt, 32, 1},
                            std::nullopt,
                            MemoryStorageType::kSystem,
                            std::make_shared<UnboundedAllocator>());
  if (!tensor)
    return false;
  auto writable = tensor->mutable_data();
  if (!writable)
    return false;
  *static_cast<std::int32_t*>(*writable) = 73;
  Native packet;
  packet.sequence = 19U;
  packet.source = "provider packet";
  packet.data = std::make_shared<Tensor>(std::move(*tensor));
  auto input =
      provider_adapters ? test_packet_input(packet) : testing::staged_values<Native>({packet});
  packet.source = "changed after staging";
  h.stage_input("input", std::move(input));
  h.declare_capture(
      "output",
      provider_adapters ? test_packet_capture() : testing::snapshot_capture<Native>(4U, 16U));
  qualify_helper_ports<Native>(h);
  const auto result = h.run();
  if (!result.ok()) {
    std::cerr << result.describe_failure() << '\n';
    return false;
  }
  const testing::Snapshot<Native> snapshot = result.snapshot<Native>("output");
  const auto metadata = snapshot.metadata();
  if (metadata.sequence != 19U || metadata.source != "provider packet" || metadata.data)
    return false;
  constexpr TensorFieldSelector<Native> data_field{
      DescriptorTensorSchemaNativeTraits<Native>::tensor_field_ids[0U], 0U};
  if (!snapshot.has_tensor(data_field))
    return false;
  const auto& bytes = snapshot.tensor(data_field).bytes();
  if (bytes.size() != sizeof(std::int32_t))
    return false;
  std::int32_t observed{};
  std::memcpy(&observed, bytes.data(), sizeof(observed));
  return observed == 73 &&
         rejects_named<std::invalid_argument>(
             [&] { static_cast<void>(result.snapshots<Tensor>("output")); }, "output");
}

// REQ: OPERATOR-066
// REQ: OPERATOR-067
// REQ: OPERATOR-068
// REQ: OPERATOR-069
bool managed_consumer() {
  using namespace holoscan;
  using namespace std::chrono_literals;
  const auto& row = test_operator_provider();
  testing::NativeOperatorTest h{
      [&row](Graph& graph) { return row.add_to_graph(graph, "operator", nullptr); }};
  h.declare_capture("output", testing::value_capture<std::int32_t>(4U));
  h.configure_graph(test_allow_restart);
  const auto execute = [&] {
    h.push_frame(testing::frame_at(0ms, testing::input("input", std::int32_t{21})));
    h.push_frame(testing::frame_at(1ms, testing::input("input", std::int32_t{5})));
    auto running = h.run_async({.frames = testing::FrameRelease::kManual});
    if (!running.release_frame(5s) || !running.wait_for_output("output", 1U, 5s))
      return false;
    const auto target = running.operator_ref();
    if (!target)
      return false;
    const std::array targets{*target};
    constexpr std::array stages{LifecycleStage::kStart};
    auto control =
        running.lifecycle_control({.allowed_targets = targets, .allowed_from_stages = stages});
    if (!control)
      return false;
    const auto request = control->request_restart(*target, {.from_stage = LifecycleStage::kStart});
    if (!request)
      return false;
    const auto terminal = control->await(*request, 5s);
    if (!terminal || terminal->state != LifecycleTransactionState::kPublished ||
        terminal->effective_stage != LifecycleStage::kStart)
      return false;
    if (!running.release_frame(5s) || !running.wait_for_output("output", 2U, 5s))
      return false;
    const auto result = running.finish();
    return result.ok() && result.values<std::int32_t>("output") == std::vector<std::int32_t>{21, 5};
  };
  // Rebuilding creates a fresh native run; restart above preserves the existing run identity.
  return execute() && execute();
}

// REQ: OPERATOR-064
// REQ: OPERATOR-073
bool image_consumer(bool provider_adapters) {
  using namespace holoscan;
  using Native = schema::ImageT;
  const std::array<std::int64_t, 3> shape{1, 2, 3};
  const std::array<std::byte, 6> pixels{
      std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}, std::byte{5}, std::byte{6}};
  auto source = make_tensor(shape,
                            DLDataType{kDLUInt, 8, 1},
                            std::nullopt,
                            MemoryStorageType::kSystem,
                            std::make_shared<UnboundedAllocator>());
  if (!source)
    return false;
  auto writable = source->mutable_data();
  if (!writable)
    return false;
  std::memcpy(*writable, pixels.data(), pixels.size());
  Native image;
  image.height = 1;
  image.width = 2;
  image.encoding = schema::ImageEncoding_RGB8;
  image.color_space = schema::ColorSpace_SRGB;
  image.significant_bits = 8U;
  image.exposure_time_ns = 2000;
  image.gain = 1.5F;
  image.header = std::make_shared<schema::HeaderT>();
  image.header->frame_id = "installed-camera";
  image.header->capture_timestamp_ns = 123456;
  image.header->device_sequence = 19U;
  image.data = std::make_shared<Tensor>(*source);
  std::optional<testing::Snapshot<Native>> saved;
  {
    const auto& row = test_image_operator_provider();
    testing::NativeOperatorTest h{
        [&row](Graph& graph) { return row.add_to_graph(graph, "operator", nullptr); }};
    // The false branch instantiates the ordinary public factories in this executable.
    h.stage_input(
        "input",
        provider_adapters ? test_image_input(image) : testing::staged_values<Native>({image}));
    h.declare_capture(
        "output",
        provider_adapters ? test_image_capture() : testing::snapshot_capture<Native>(4U, 6U));
    qualify_helper_ports<Native>(h);
    image.header->frame_id = "changed after staging";
    image.header->capture_timestamp_ns = 999;
    image.header->device_sequence = 99U;
    const auto result = h.run();
    if (!result.ok()) {
      std::cerr << result.describe_failure() << '\n';
      return false;
    }
    // In the provider branch, the typed read accesses provider-created capture storage.
    saved.emplace(result.snapshot<Native>("output"));
    if (!rejects_named<std::invalid_argument>(
            [&] { static_cast<void>(result.snapshots<schema::TensorPacketT>("output")); },
            "output"))
      return false;
  }
  // The copied snapshot survives both fixture/result destruction and mutation of the old input.
  writable = source->mutable_data();
  if (!writable)
    return false;
  std::memset(*writable, 0, pixels.size());
  auto metadata = saved->metadata();
  if (metadata.data || metadata.height != 1 || metadata.width != 2 ||
      metadata.encoding != schema::ImageEncoding_RGB8 ||
      metadata.color_space != schema::ColorSpace_SRGB || metadata.significant_bits != 8U ||
      metadata.exposure_time_ns != 2000 || metadata.gain != 1.5F || !metadata.header ||
      metadata.header.get() == image.header.get() ||
      metadata.header->frame_id != "installed-camera" ||
      metadata.header->capture_timestamp_ns != 123456 || metadata.header->device_sequence != 19U)
    return false;
  metadata.header->frame_id = "changed returned metadata";
  metadata.header->capture_timestamp_ns = 0;
  const auto again = saved->metadata();
  if (!again.header || again.header.get() == metadata.header.get() ||
      again.header->frame_id != "installed-camera" || again.header->capture_timestamp_ns != 123456)
    return false;
  if (!saved->has_tensor(schema::ImageTensorFields::data))
    return false;
  const auto& snapshot = saved->tensor(schema::ImageTensorFields::data);
  return snapshot.shape() == std::vector<std::int64_t>{shape.begin(), shape.end()} &&
         snapshot.bytes() == std::vector<std::byte>{pixels.begin(), pixels.end()} &&
         snapshot.dtype().code == kDLUInt && snapshot.dtype().bits == 8U &&
         snapshot.dtype().lanes == 1U && snapshot.source_device().device_type == kDLCPU;
}
}  // namespace

// REQ: OPERATOR-064
int main(int argc, char** argv) {
  if (argc == 3 && std::string_view(argv[1]) == "--case") {
    try {
      if (!native_type_witnesses())
        return 6;
      const std::string_view name{argv[2]};
      if (name == "pod-append")
        return pod_consumer(false, false) && pod_consumer(true, false) ? 0 : 7;
      if (name == "pod-frames")
        return pod_consumer(false, true) && pod_consumer(true, true) ? 0 : 7;
      if (name == "schema-values")
        return schema_consumer(false) && schema_consumer(true) ? 0 : 8;
      if (name == "packet-snapshot")
        return packet_consumer(false) && packet_consumer(true) ? 0 : 9;
      if (name == "image-snapshot")
        return image_consumer(false) && image_consumer(true) ? 0 : 12;
      std::cerr << "unknown installed qualification case: " << name << '\n';
      return 11;
    } catch (const std::exception& error) {
      std::cerr << "native provider qualification failed: " << error.what() << '\n';
      return 10;
    }
  }

  using namespace holoscan;
  using namespace holoscan::authoring;
  const auto& row = test_operator_provider();
  const std::array assignments{
      ParameterAssignment{"factor", ParameterValue::from(std::int64_t{2})}};
  auto materialized = materialize_parameters(*row.param_type, row.param_schema, assignments);
  if (!materialized)
    return 1;
  auto params = std::make_shared<MaterializedParameters>(std::move(*materialized));
  testing::NativeOperatorTest h{[&row, params](Graph& graph) {
    auto op = row.add_to_graph(graph, "operator", nullptr);
    row.set_params(op, *params);
    return op;
  }};
  h.stage_input("input", testing::staged_values<std::int32_t>({21}));
  h.declare_capture("output", testing::value_capture<std::int32_t>(4U));
  h.input_queue_depth("input", 4U);
  std::string topic = "external-consumer-owned-input";
  h.input_flow("input", ConnectionOptions{.queue_depth = 4U, .topic_name = topic});
  topic.assign("changed-after-staging");
  h.output_flow("output", ConnectionOptions{.queue_depth = 4U});
  h.configure_graph([&row](testing::NativeTestGraphView& view) {
    if (!row.port(view.tested_operator(), "input") || !row.port(view.tested_operator(), "output")) {
      throw std::runtime_error{"registered port lookup failed"};
    }
    view.graph().configure(view.input_flow<std::int32_t>("input")).queue_depth(4U);
    view.graph().configure(view.output_flow<std::int32_t>("output")).queue_depth(4U);
    auto sources = view.graph().partition("sources");
    view.add_sources_to(sources);
    auto tested = view.graph().partition("operator");
    row.partition_add(tested, view.tested_operator());
    auto collectors = view.graph().partition("collectors");
    view.add_collectors_to(collectors);
  });
  CompileOptions options;
  options.deployment.bind_thread_placement("operator", ThreadPlacement{});
  h.set_compile_options(options);
  const auto result = h.run();
  if (!result.ok()) {
    std::cerr << result.describe_failure() << '\n';
    return 2;
  }
  if (result.value<std::int32_t>("output") != 42)
    return 3;
  if (!tensor_consumer())
    return 4;
  if (!clock_consumer())
    return 5;
  try {
    if (!managed_consumer())
      return 13;
    if (!native_type_witnesses())
      return 6;
    for (const bool provider_first : {false, true}) {
      if (!pod_consumer(provider_first, false) || !pod_consumer(provider_first, true))
        return 7;
      if (!schema_consumer(provider_first))
        return 8;
      if (!packet_consumer(provider_first))
        return 9;
      if (!image_consumer(provider_first))
        return 12;
    }
  } catch (const std::exception& error) {
    std::cerr << "native provider qualification failed: " << error.what() << '\n';
    return 10;
  }
  std::cout << "installed registered operator result: 42; native provider mode="
            << (TEST_OPERATOR_PROVIDER_SHARED ? "shared" : "static") << '\n';
}
