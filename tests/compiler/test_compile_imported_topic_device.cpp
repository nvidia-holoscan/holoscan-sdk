/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include <memory>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/schema/flatbuffer_schema_package.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "compile_test_graph.hpp"

// GTest fatal assertions establish optional preconditions that clang-tidy does not model.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
namespace holoscan {

// Two descriptor-backed Tensor fields with independent contracts, modeled on the multi-field
// fixture in test_compile_tensor_byte_span_derivation.cpp.
struct ImportedMultiTensorSchema {
  std::shared_ptr<Tensor> image;
  std::shared_ptr<Tensor> calibration;
};

template <>
consteval SchemaIdentity schema_identity<ImportedMultiTensorSchema>() {
  return schema_identity<schema::ImageT>();
}

template <>
struct DescriptorTensorSchemaNativeTraits<ImportedMultiTensorSchema> {
  using root_type = schema::Image;
  static constexpr std::size_t metadata_capacity =
      schema::flatbuffer_descriptor_metadata_capacity_v<root_type>;
  static constexpr std::size_t tensor_count = 2U;
  static constexpr std::array<std::uint16_t, 2U> tensor_field_ids{1U, 7U};

  [[nodiscard]] static const Tensor* tensor_at(const ImportedMultiTensorSchema& value,
                                               std::size_t index) noexcept {
    return index == 0U ? value.image.get() : (index == 1U ? value.calibration.get() : nullptr);
  }
};

namespace detail {

template <>
const RuntimePortTypeOps* runtime_port_type_ops<ImportedMultiTensorSchema>() noexcept {
  static const RuntimePortTypeOps ops = schema::make_descriptor_tensor_port_type_ops<
      ImportedMultiTensorSchema,
      schema::Image,
      schema::flatbuffer_descriptor_metadata_capacity_v<schema::Image>>(
      [](void*) noexcept -> expected<void, Error> {
        return make_unexpected(Error{ErrorCode::kNotImplemented});
      },
      [](void*) noexcept {},
      [](const ImportedMultiTensorSchema&,
         schema::DescriptorTensorReferences,
         std::span<std::byte>) noexcept -> expected<std::size_t, Error> {
        return make_unexpected(Error{ErrorCode::kNotImplemented});
      },
      [](void*, const schema::Image&, std::span<Tensor>, SampleMetadata) noexcept
          -> expected<void, Error> { return make_unexpected(Error{ErrorCode::kNotImplemented}); });
  return &ops;
}

}  // namespace detail

namespace {

constexpr std::string_view kEndpoint = "proto_image";
constexpr DLDataType kDtype{kDLUInt, 8U, 1U};
constexpr std::array<std::int64_t, 2U> kImageShape{1, 8};
constexpr TensorFieldSelector<schema::ImageT> kImageData{4U, 0U};

class DeviceTensorSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<Tensor> input;
};

template <MemoryKind kMemory>
class DeviceImageSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .tensor_field(kImageData)
        .expects_tensor(
            TensorInputSpec{.representation = {.memory_kind = kMemory,
                                               .dtype = kDtype,
                                               .rank = 2U,
                                               .strides = TensorStridePolicy::kContiguousOnly}});
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<schema::ImageT> input;
};

[[nodiscard]] TensorBoundaryContract device_contract(
    MemoryKind memory,
    InterfaceTensorEmitPolicy policy = InterfaceTensorEmitPolicy::kCopyIfNeeded) {
  return TensorBoundaryContract{
      .representation = {.memory_kind = memory,
                         .dtype = kDtype,
                         .rank = 1U,
                         .strides = TensorStridePolicy::kContiguousOnly},
      .bounds = tensor_bounds(8U),
      .transfer = TensorTransferPolicy{.foreign_emit_policy = policy},
  };
}

[[nodiscard]] TopicOptions import_options(std::string_view endpoint_name) {
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
  return options;
}

// Compose one imported Tensor topic into a sink:
//
//   [holomq://proto_image] --Tensor--> sink
void add_tensor_import(
    Graph& graph, MemoryKind memory, std::string_view endpoint_name = kEndpoint,
    InterfaceTensorEmitPolicy policy = InterfaceTensorEmitPolicy::kCopyIfNeeded) {
  test::add_compile_test_default_clock(graph);
  TopicOptions options = import_options(endpoint_name);
  options.tensor = device_contract(memory, policy);
  const auto imported =
      graph.import_topic<Tensor>("holomq://" + std::string(endpoint_name), std::move(options));
  const auto sink = graph.op<DeviceTensorSink>("sink");
  graph.add_flow(imported.output(), sink->input);
}

// Compose one imported descriptor-schema topic whose Image data field has the given memory kind:
//
//   [holomq://proto_image] --ImageT--> sink
void add_image_import(Graph& graph, MemoryKind memory, std::string_view endpoint_name = kEndpoint,
                      InterfaceTensorEmitPolicy policy = InterfaceTensorEmitPolicy::kCopyIfNeeded) {
  test::add_compile_test_default_clock(graph);
  DescriptorTensorTopicContract<schema::ImageT> contract;
  contract.field(kImageData,
                 TensorBoundaryContract{
                     .representation = {.memory_kind = memory,
                                        .dtype = kDtype,
                                        .rank = 2U,
                                        .strides = TensorStridePolicy::kContiguousOnly},
                     .bounds = tensor_bounds(kImageShape),
                     .transfer = TensorTransferPolicy{.foreign_emit_policy = policy},
                 });
  const auto imported = graph.import_topic<schema::ImageT>(
      "holomq://" + std::string(endpoint_name), import_options(endpoint_name), std::move(contract));
  if (memory == MemoryKind::kPinnedHost) {
    graph.add_flow(imported.output(),
                   graph.op<DeviceImageSink<MemoryKind::kPinnedHost>>("sink")->input);
  } else {
    graph.add_flow(imported.output(),
                   graph.op<DeviceImageSink<MemoryKind::kCudaDevice>>("sink")->input);
  }
}

[[nodiscard]] CompileOptions bind_import(std::string_view endpoint_name, DeviceId device) {
  CompileOptions options;
  options.deployment.bind_imported_topic_device(
      ImportedTopicDevicePlacement{.endpoint_name = endpoint_name, .device = device});
  return options;
}

constexpr TensorFieldSelector<ImportedMultiTensorSchema> kMultiImage{1U, 0U};
constexpr TensorFieldSelector<ImportedMultiTensorSchema> kMultiCalibration{7U, 1U};

template <MemoryKind kFirst, MemoryKind kSecond>
class MultiFieldSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    auto& builder = spec.input(input, "input").queue_depth(1U);
    builder.tensor_field(kMultiImage)
        .expects_tensor(
            TensorInputSpec{.representation = {.memory_kind = kFirst,
                                               .dtype = kDtype,
                                               .rank = 1U,
                                               .strides = TensorStridePolicy::kContiguousOnly}});
    builder.tensor_field(kMultiCalibration)
        .expects_tensor(
            TensorInputSpec{.representation = {.memory_kind = kSecond,
                                               .dtype = kDtype,
                                               .rank = 1U,
                                               .strides = TensorStridePolicy::kContiguousOnly}});
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<ImportedMultiTensorSchema> input;
};

// Compose one imported two-field descriptor topic with independent per-field storage and policy:
//
//   [holomq://proto_image] --{field 1, field 7}--> sink
template <MemoryKind kFirst, MemoryKind kSecond>
void add_multi_field_import(Graph& graph, InterfaceTensorEmitPolicy first_policy,
                            InterfaceTensorEmitPolicy second_policy) {
  test::add_compile_test_default_clock(graph);
  DescriptorTensorTopicContract<ImportedMultiTensorSchema> contract;
  contract.field(kMultiImage, device_contract(kFirst, first_policy));
  contract.field(kMultiCalibration, device_contract(kSecond, second_policy));
  const auto imported = graph.import_topic<ImportedMultiTensorSchema>(
      "holomq://" + std::string(kEndpoint), import_options(kEndpoint), std::move(contract));
  graph.add_flow(imported.output(), graph.op<MultiFieldSink<kFirst, kSecond>>("sink")->input);
}

[[nodiscard]] const Diagnostic* find_diagnostic(const ExecutionPlan& plan, std::string_view token) {
  const auto found = std::ranges::find(plan.diagnostics(), token, &Diagnostic::token);
  return found == plan.diagnostics().end() ? nullptr : &*found;
}

// Expect one PHYSICAL_TENSOR_DEVICE_BINDING_INVALID diagnostic carrying the given reason.
void expect_rejected_with_reason(
    const ExecutionPlan& plan, std::string_view reason,
    std::string_view pointer = "/deployment/imported_topic_devices/0") {
  ASSERT_FALSE(plan.ok());
  const Diagnostic* const diagnostic =
      find_diagnostic(plan, "PHYSICAL_TENSOR_DEVICE_BINDING_INVALID");
  ASSERT_NE(diagnostic, nullptr) << plan.json();
  EXPECT_EQ(diagnostic->json_pointer, pointer);
  EXPECT_NE(diagnostic->provided.find("reason=" + std::string(reason)), std::string::npos)
      << diagnostic->provided;
}

// REQ: COMPILE-052
TEST(CompileImportedTopicDevice, RawCudaTensorImportCompilesWithEndpointBinding) {
  Graph graph{"imported-raw-cuda"};
  add_tensor_import(graph, MemoryKind::kCudaDevice);

  // Key setup: without a binding the imported device topic names its endpoint and the new API.
  const ExecutionPlan unbound = compile(graph);
  ASSERT_FALSE(unbound.ok());
  const Diagnostic* const missing = find_diagnostic(unbound, "IMPLICIT_DEVICE_ZERO_REJECTED");
  ASSERT_NE(missing, nullptr) << unbound.json();
  EXPECT_NE(missing->provided.find("imported_topic=proto_image"), std::string::npos)
      << missing->provided;
  EXPECT_EQ(missing->provided.find("<invalid>"), std::string::npos) << missing->provided;
  EXPECT_NE(missing->required.find("bind_imported_topic_device"), std::string::npos)
      << missing->required;
  EXPECT_NE(missing->required.find("\"proto_image\""), std::string::npos) << missing->required;

  // Key assertion: the endpoint binding resolves the topic's own port and reaches its pool.
  const ExecutionPlan plan = compile(graph, bind_import(kEndpoint, DeviceId{0}));
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto pools = plan.tensor_interface_pools();
  ASSERT_EQ(pools.size(), 1U);
  EXPECT_EQ(pools.front().memory_kind, MemoryKind::kCudaDevice);
  EXPECT_EQ(pools.front().device, DeviceId{0});
  // Key assertion: the consuming Operator is CUDA-active on the bound device, so the runtime
  // can order its reads of the imported device storage.
  ASSERT_EQ(plan.operator_cuda_stream_bindings().size(), 1U);
  EXPECT_EQ(plan.operator_cuda_stream_bindings().front().device, DeviceId{0});
}

// REQ: COMPILE-052
TEST(CompileImportedTopicDevice, PinnedHostTensorImportCompilesWithEndpointBinding) {
  Graph graph{"imported-raw-pinned"};
  add_tensor_import(graph, MemoryKind::kPinnedHost);

  const ExecutionPlan unbound = compile(graph);
  EXPECT_FALSE(unbound.ok());

  // Key assertion: pinned-host imports use the same explicit endpoint device binding.
  const ExecutionPlan plan = compile(graph, bind_import(kEndpoint, DeviceId{0}));
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto pools = plan.tensor_interface_pools();
  ASSERT_EQ(pools.size(), 1U);
  EXPECT_EQ(pools.front().memory_kind, MemoryKind::kPinnedHost);
  EXPECT_EQ(pools.front().device, DeviceId{0});
  // Key assertion: the consuming Operator is CUDA-active on the bound device, so the runtime
  // can order its reads of the imported device storage.
  ASSERT_EQ(plan.operator_cuda_stream_bindings().size(), 1U);
  EXPECT_EQ(plan.operator_cuda_stream_bindings().front().device, DeviceId{0});
}

// REQ: COMPILE-052
TEST(CompileImportedTopicDevice, DescriptorSchemaCudaFieldCompilesWithEndpointBinding) {
  Graph graph{"imported-image-cuda"};
  add_image_import(graph, MemoryKind::kCudaDevice);

  const ExecutionPlan unbound = compile(graph);
  ASSERT_FALSE(unbound.ok());
  EXPECT_TRUE(test::has_diagnostic(unbound, "IMPLICIT_DEVICE_ZERO_REJECTED")) << unbound.json();

  // Key assertion: the schema field pool records the bound device.
  const ExecutionPlan plan = compile(graph, bind_import(kEndpoint, DeviceId{0}));
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto pools = plan.tensor_interface_pools();
  ASSERT_EQ(pools.size(), 1U);
  EXPECT_EQ(pools.front().field_id, std::uint16_t{4});
  EXPECT_EQ(pools.front().memory_kind, MemoryKind::kCudaDevice);
  EXPECT_EQ(pools.front().device, DeviceId{0});
  // Key assertion: the consuming Operator is CUDA-active on the bound device, so the runtime
  // can order its reads of the imported device storage.
  ASSERT_EQ(plan.operator_cuda_stream_bindings().size(), 1U);
  EXPECT_EQ(plan.operator_cuda_stream_bindings().front().device, DeviceId{0});
}

// REQ: COMPILE-052
// REQ: CHAN-173
TEST(CompileImportedTopicDevice, DescriptorSchemaPinnedHostFieldCompilesWithEndpointBinding) {
  Graph graph{"imported-image-pinned"};
  add_image_import(graph, MemoryKind::kPinnedHost);

  EXPECT_FALSE(compile(graph).ok());
  const ExecutionPlan plan = compile(graph, bind_import(kEndpoint, DeviceId{0}));
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto pools = plan.tensor_interface_pools();
  ASSERT_EQ(pools.size(), 1U);
  EXPECT_EQ(pools.front().memory_kind, MemoryKind::kPinnedHost);
  EXPECT_EQ(pools.front().device, DeviceId{0});
  // Key assertion: the external publisher may write the pinned-host field on a CUDA stream, so
  // the consuming Operator gets a stream on the bound device to wait for that write, as it would
  // for a raw pinned-host Tensor.
  ASSERT_EQ(plan.operator_cuda_stream_bindings().size(), 1U);
  EXPECT_EQ(plan.operator_cuda_stream_bindings().front().device, DeviceId{0});
}

// REQ: COMPILE-052
TEST(CompileImportedTopicDevice, UnnamedImportIsBoundByUri) {
  Graph graph{"imported-uri-bound"};
  test::add_compile_test_default_clock(graph);
  TopicOptions options;
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
  options.tensor = device_contract(MemoryKind::kCudaDevice);
  const auto imported = graph.import_topic<Tensor>("holomq://uri_only", std::move(options));
  const auto sink = graph.op<DeviceTensorSink>("sink");
  graph.add_flow(imported.output(), sink->input);

  // Key assertion: with no endpoint name the topic URI is the binding key.
  const ExecutionPlan plan = compile(graph, bind_import("holomq://uri_only", DeviceId{0}));
  EXPECT_TRUE(plan.ok()) << plan.json();
}

// REQ: COMPILE-052
TEST(CompileImportedTopicDevice, RejectsUnknownEndpoint) {
  Graph graph{"imported-unknown"};
  add_tensor_import(graph, MemoryKind::kCudaDevice);

  const ExecutionPlan plan = compile(graph, bind_import("other_endpoint", DeviceId{0}));

  // Key assertion: an unmatched binding is rejected instead of being ignored.
  expect_rejected_with_reason(plan, "unknown_imported_topic");
}

// REQ: COMPILE-052
TEST(CompileImportedTopicDevice, RejectsAmbiguousImportedEndpoint) {
  Graph graph{"imported-ambiguous"};
  test::add_compile_test_default_clock(graph);
  // Two unnamed imports of one URI share the URI as their binding key.
  for (const std::string_view sink_name : {"first_sink", "second_sink"}) {
    TopicOptions options;
    options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
    options.tensor = device_contract(MemoryKind::kCudaDevice);
    const auto imported = graph.import_topic<Tensor>("holomq://shared_uri", std::move(options));
    const auto sink = graph.op<DeviceTensorSink>(sink_name);
    graph.add_flow(imported.output(), sink->input);
  }

  const ExecutionPlan plan = compile(graph, bind_import("holomq://shared_uri", DeviceId{0}));

  // Key assertion: a key matching more than one imported topic is rejected, not guessed.
  expect_rejected_with_reason(plan, "ambiguous_imported_topic");
}

// REQ: COMPILE-052
TEST(CompileImportedTopicDevice, RejectsExportedTopic) {
  Graph graph{"imported-exported-name"};
  add_tensor_import(graph, MemoryKind::kCudaDevice);
  TopicOptions options;
  options.endpoint_name = "exported_endpoint";
  options.bounds.subscriber =
      SubscriberEndpointBounds{.queue_depth = 1U, .max_borrowed_samples = 1U};
  const auto exported =
      graph.export_topic<std::uint32_t>("holomq://exported_endpoint", std::move(options));
  const auto producer = graph.op<test::CompileTestOperator>("producer");
  graph.add_flow(producer->output, exported.input());

  const ExecutionPlan plan = compile(graph, bind_import("exported_endpoint", DeviceId{0}));

  // Key assertion: only imported topics are selectable by this API.
  ASSERT_FALSE(plan.ok());
  const Diagnostic* const diagnostic =
      find_diagnostic(plan, "PHYSICAL_TENSOR_DEVICE_BINDING_INVALID");
  ASSERT_NE(diagnostic, nullptr) << plan.json();
  EXPECT_NE(diagnostic->provided.find("reason=exported_topic"), std::string::npos)
      << diagnostic->provided;
}

// REQ: COMPILE-052
TEST(CompileImportedTopicDevice, RejectsNonTensorPayload) {
  Graph graph{"imported-scalar"};
  test::add_compile_test_default_clock(graph);
  TopicOptions options;
  options.endpoint_name = kEndpoint;
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
  const auto imported =
      graph.import_topic<std::uint32_t>("holomq://proto_image", std::move(options));
  const auto sink = graph.op<test::CompileTestOperator>("sink");
  graph.add_flow(imported.output(), sink->input);

  const ExecutionPlan plan = compile(graph, bind_import(kEndpoint, DeviceId{0}));

  // Key assertion: a scalar topic has no Tensor storage to place.
  expect_rejected_with_reason(plan, "non_tensor_output");
}

// REQ: COMPILE-052
TEST(CompileImportedTopicDevice, RejectsInvalidDevice) {
  Graph graph{"imported-invalid-device"};
  add_tensor_import(graph, MemoryKind::kCudaDevice);

  const ExecutionPlan plan = compile(graph, bind_import(kEndpoint, DeviceId{}));

  // Key assertion: an unresolved device is rejected rather than selecting device zero.
  expect_rejected_with_reason(plan, "invalid_device");
}

// REQ: COMPILE-052
TEST(CompileImportedTopicDevice, RejectsDuplicateBinding) {
  Graph graph{"imported-duplicate"};
  add_tensor_import(graph, MemoryKind::kCudaDevice);
  CompileOptions options = bind_import(kEndpoint, DeviceId{0});
  options.deployment.bind_imported_topic_device(
      {.endpoint_name = kEndpoint, .device = DeviceId{1}});

  const ExecutionPlan plan = compile(graph, std::move(options));

  // Key assertion: the second binding for the same topic is rejected at its authored index.
  expect_rejected_with_reason(plan, "duplicate_output", "/deployment/imported_topic_devices/1");
}

// An imported CUDA Tensor with a require-exportable policy creates no staging pool, so it has no
// planned device or consumer stream and could never deliver a sample. Topic lowering rejects it
// whether or not a binding exists, before any device is requested, and reports one diagnostic.
void expect_unsupported_cuda_transfer(const ExecutionPlan& plan, std::string_view pointer,
                                      bool other_field_needs_binding = false) {
  ASSERT_FALSE(plan.ok());
  const Diagnostic* const diagnostic =
      find_diagnostic(plan, "PHYSICAL_TENSOR_LOGICAL_REQUIREMENTS_INVALID");
  ASSERT_NE(diagnostic, nullptr) << plan.json();
  EXPECT_EQ(diagnostic->json_pointer, pointer);
  EXPECT_NE(diagnostic->required.find("kCopyIfNeeded"), std::string::npos) << diagnostic->required;
  // An unbound copied pinned-host field next to the rejected field legitimately needs a binding.
  if (!other_field_needs_binding) {
    EXPECT_FALSE(test::has_diagnostic(plan, "IMPLICIT_DEVICE_ZERO_REJECTED")) << plan.json();
  }
  EXPECT_FALSE(test::has_diagnostic(plan, "PHYSICAL_TENSOR_DEVICE_BINDING_UNUSED")) << plan.json();
  EXPECT_FALSE(test::has_diagnostic(plan, "PHYSICAL_TENSOR_DEVICE_BINDING_INVALID")) << plan.json();
}

// REQ: COMPILE-052
TEST(CompileImportedTopicDevice, RejectsCudaImportWithoutCopyCapableTransfer) {
  constexpr auto kExportable = InterfaceTensorEmitPolicy::kRequireExportable;
  for (const bool bound : {false, true}) {
    SCOPED_TRACE(bound ? "bound" : "unbound");
    const auto compile_with = [&](Graph& graph) {
      return bound ? compile(graph, bind_import(kEndpoint, DeviceId{0})) : compile(graph);
    };

    Graph raw_cuda{"imported-raw-cuda-exportable"};
    add_tensor_import(raw_cuda, MemoryKind::kCudaDevice, kEndpoint, kExportable);
    expect_unsupported_cuda_transfer(compile_with(raw_cuda), "/external_topics/1/tensor");

    Graph raw_managed{"imported-raw-managed-exportable"};
    add_tensor_import(raw_managed, MemoryKind::kCudaManaged, kEndpoint, kExportable);
    expect_unsupported_cuda_transfer(compile_with(raw_managed), "/external_topics/1/tensor");

    // Key assertion: a default-policy ImageT field no longer compiles into a streamless consumer.
    Graph image_cuda{"imported-image-cuda-exportable"};
    add_image_import(image_cuda, MemoryKind::kCudaDevice, kEndpoint, kExportable);
    expect_unsupported_cuda_transfer(compile_with(image_cuda),
                                     "/external_topics/1/tensor_fields/4");
  }
}

// REQ: COMPILE-052
TEST(CompileImportedTopicDevice, RejectsBoundPinnedHostImportWithoutCopyCapableTransfer) {
  Graph graph{"imported-raw-pinned-exportable"};
  add_tensor_import(
      graph, MemoryKind::kPinnedHost, kEndpoint, InterfaceTensorEmitPolicy::kRequireExportable);

  const ExecutionPlan plan = compile(graph, bind_import(kEndpoint, DeviceId{0}));

  // Key assertion: a pinned-host binding without a staging pool is rejected, not left unused.
  expect_rejected_with_reason(plan, "requires_copy_capable_transfer");
  EXPECT_FALSE(test::has_diagnostic(plan, "PHYSICAL_TENSOR_DEVICE_BINDING_UNUSED")) << plan.json();
}

// A copied pinned-host field creates a pool, but a pinned-host descriptor pool gives its consumer
// no CUDA stream, so it cannot stand in for an unstaged CUDA field in the same topic.
//
// REQ: COMPILE-052
TEST(CompileImportedTopicDevice, RejectsUnstagedCudaFieldNextToCopiedPinnedHostField) {
  constexpr auto kCopy = InterfaceTensorEmitPolicy::kCopyIfNeeded;
  constexpr auto kExportable = InterfaceTensorEmitPolicy::kRequireExportable;
  for (const bool bound : {false, true}) {
    SCOPED_TRACE(bound ? "bound" : "unbound");
    const auto compile_with = [&](Graph& graph) {
      return bound ? compile(graph, bind_import(kEndpoint, DeviceId{0})) : compile(graph);
    };

    Graph pinned_first{"imported-multi-pinned-then-cuda"};
    add_multi_field_import<MemoryKind::kPinnedHost, MemoryKind::kCudaDevice>(
        pinned_first, kCopy, kExportable);
    expect_unsupported_cuda_transfer(
        compile_with(pinned_first), "/external_topics/1/tensor_fields/7", !bound);

    Graph cuda_first{"imported-multi-cuda-then-pinned"};
    add_multi_field_import<MemoryKind::kCudaDevice, MemoryKind::kPinnedHost>(
        cuda_first, kExportable, kCopy);
    expect_unsupported_cuda_transfer(
        compile_with(cuda_first), "/external_topics/1/tensor_fields/1", !bound);
  }
}

// REQ: COMPILE-052
TEST(CompileImportedTopicDevice, MultiFieldCopyCapableImportCompilesWithEndpointBinding) {
  Graph graph{"imported-multi-copy-capable"};
  add_multi_field_import<MemoryKind::kCudaDevice, MemoryKind::kPinnedHost>(
      graph, InterfaceTensorEmitPolicy::kCopyIfNeeded, InterfaceTensorEmitPolicy::kCopyIfNeeded);

  const ExecutionPlan plan = compile(graph, bind_import(kEndpoint, DeviceId{0}));

  // Key assertion: independently copied CUDA and pinned-host fields stay admitted, and the CUDA
  // field's pool gives the consumer its stream.
  ASSERT_TRUE(plan.ok()) << plan.json();
  EXPECT_EQ(plan.tensor_interface_pools().size(), 2U);
  ASSERT_EQ(plan.operator_cuda_stream_bindings().size(), 1U);
  EXPECT_EQ(plan.operator_cuda_stream_bindings().front().device, DeviceId{0});
}

// REQ: COMPILE-052
TEST(CompileImportedTopicDevice, ManagedImportCompilesWithEndpointBinding) {
  Graph graph{"imported-raw-managed"};
  add_tensor_import(graph, MemoryKind::kCudaManaged);

  // Key setup: a copy-capable managed import needs a device just like a CUDA-device import.
  const ExecutionPlan unbound = compile(graph);
  ASSERT_FALSE(unbound.ok());
  const Diagnostic* const missing = find_diagnostic(unbound, "IMPLICIT_DEVICE_ZERO_REJECTED");
  ASSERT_NE(missing, nullptr) << unbound.json();
  EXPECT_NE(missing->required.find("bind_imported_topic_device"), std::string::npos)
      << missing->required;

  const ExecutionPlan plan = compile(graph, bind_import(kEndpoint, DeviceId{0}));
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_interface_pools().size(), 1U);
  EXPECT_EQ(plan.tensor_interface_pools().front().memory_kind, MemoryKind::kCudaManaged);
  ASSERT_EQ(plan.operator_cuda_stream_bindings().size(), 1U);
}

// REQ: COMPILE-052
TEST(CompileImportedTopicDevice, RejectsBindingOnHostTopic) {
  Graph graph{"imported-host-bound"};
  add_tensor_import(graph, MemoryKind::kHost);

  const ExecutionPlan plan = compile(graph, bind_import(kEndpoint, DeviceId{0}));

  // Key assertion: host storage keeps rejecting device bindings, imported or not.
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PHYSICAL_TENSOR_DEVICE_BINDING_UNEXPECTED"))
      << plan.json();
}

// REQ: COMPILE-052
TEST(CompileImportedTopicDevice, BoundDevicePartitionsPlanIdentity) {
  Graph graph{"imported-identity"};
  add_tensor_import(graph, MemoryKind::kCudaDevice);

  const ExecutionPlan device0 = compile(graph, bind_import(kEndpoint, DeviceId{0}));
  const ExecutionPlan device0_again = compile(graph, bind_import(kEndpoint, DeviceId{0}));
  const ExecutionPlan device1 = compile(graph, bind_import(kEndpoint, DeviceId{1}));

  // Key assertion: the deployment hash includes the new binding and is deterministic.
  ASSERT_TRUE(device0.ok()) << device0.json();
  ASSERT_TRUE(device1.ok()) << device1.json();
  EXPECT_EQ(device0.digest(), device0_again.digest());
  EXPECT_NE(device0.digest(), device1.digest());
}

}  // namespace
}  // namespace holoscan
// NOLINTEND(bugprone-unchecked-optional-access)
