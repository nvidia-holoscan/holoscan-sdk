/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/authoring/register_payload.hpp>
#include <holoscan/authoring/registry.hpp>
#include <holoscan/channel/runtime.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/schema/authoring_registry.hpp>
#include <holoscan/schema/sdk_schemas.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <new>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

#include <unistd.h>

#include "../common/allocation_counter.hpp"

namespace holoscan {

using RegistrySchema = schema::StringT;

// Distinct C++ declarations with one complete raw-POD identity exercise registry dispatch by
// semantic contract rather than by either declaration's template specialization.
// Representative test: AuthoringPayloadRegistry.PreservesDistinctCompatiblePodDeclarationTypes.
struct RegistryPodA {
  std::uint32_t value{};
};

struct RegistryPodB {
  std::uint32_t value{};
};

template <>
struct ChannelTypeIdentity<RegistryPodA> {
  static constexpr std::string_view name = "nvidia.holoscan.test.RegistryPod";
  static constexpr std::uint32_t version = 1;
};

template <>
struct ChannelTypeIdentity<RegistryPodB> : ChannelTypeIdentity<RegistryPodA> {};

template <>
struct ChannelPodLayout<RegistryPodA> {
  static constexpr auto fields =
      pod_fields(pod_field("value", &RegistryPodA::value, offsetof(RegistryPodA, value)));
};

template <>
struct ChannelPodLayout<RegistryPodB> {
  static constexpr auto fields =
      pod_fields(pod_field("value", &RegistryPodB::value, offsetof(RegistryPodB, value)));
};

}  // namespace holoscan

// A hand-written thunk models a raw erased caller without admitting enum registration.
// Representative test: AuthoringPayloadRegistry.RejectsLocalEnumOpaqueResolution.
namespace holoscan::authoring::detail {
template <>
struct PayloadRegistrationThunks<RegistryPodB> {
  static const void* resolve(const OpaquePortRef& port, const PayloadIdentityRecord& identity,
                             PortDirection direction) {
    return port.resolve(identity, direction);
  }
};
}  // namespace holoscan::authoring::detail

namespace {

// Equal-width enums deliberately have indistinguishable pointer-free identities.
// Representative test: AuthoringPayloadRegistry.RejectsLocalEnumOpaqueResolution.
enum class RegistryEnumA : std::uint32_t { kValue = 1 };
enum class RegistryEnumB : std::uint32_t { kValue = 1 };

// Unevaluated calls distinguish constraint rejection from errors inside a thunk body.
// Representative test: AuthoringPayloadRegistry.ExcludesLocalEnumsAtRegistrationConstraints.
template <typename T>
constexpr bool kPayloadFactoryAvailable =
    requires { holoscan::authoring::detail::make_registered_payload<T>("payload"); };

template <typename T>
constexpr bool kAutomaticPayloadRegistrationAvailable =
    requires { holoscan::authoring::detail::register_payload_automatically<T>("payload"); };

// Provides typed output declarations for
// AuthoringPayloadRegistry.AuthorsFlowInterfacesTopicsAndConfiguratorsThroughRows.
template <typename T>
class PayloadSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { (void)spec.output(output); }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Output<T> output{"output"};
};

// Provides matching and mismatched typed inputs for
// AuthoringPayloadRegistry.RejectsIdentityMismatchBeforeTypedDowncast.
template <typename T>
class PayloadSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { (void)spec.input(input); }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Input<T> input{"input"};
};

using UIntSource = PayloadSource<std::uint32_t>;
using UIntSink = PayloadSink<std::uint32_t>;
using IntSource = PayloadSource<std::int32_t>;
using IntSink = PayloadSink<std::int32_t>;
using PodSourceA = PayloadSource<holoscan::RegistryPodA>;
using PodSourceB = PayloadSource<holoscan::RegistryPodB>;
using PodSinkB = PayloadSink<holoscan::RegistryPodB>;
using TensorPacketSource = PayloadSource<holoscan::schema::TensorPacketT>;
using TensorSink = PayloadSink<holoscan::Tensor>;

constexpr std::string_view kUIntSourceName{"holoscan.test.UIntSource"};
constexpr std::string_view kUIntSinkName{"holoscan.test.UIntSink"};
constexpr std::string_view kIntSourceName{"holoscan.test.IntSource"};
constexpr std::string_view kIntSinkName{"holoscan.test.IntSink"};
constexpr std::string_view kPodSourceAName{"holoscan.test.PodSourceA"};
constexpr std::string_view kPodSourceBName{"holoscan.test.PodSourceB"};
constexpr std::string_view kPodSinkBName{"holoscan.test.PodSinkB"};

HOLOSCAN_REGISTER_OPERATOR(UIntSource, kUIntSourceName);
HOLOSCAN_REGISTER_OPERATOR(UIntSink, kUIntSinkName);
HOLOSCAN_REGISTER_OPERATOR(IntSource, kIntSourceName);
HOLOSCAN_REGISTER_OPERATOR(IntSink, kIntSinkName);
HOLOSCAN_REGISTER_OPERATOR(PodSourceA, kPodSourceAName);
HOLOSCAN_REGISTER_OPERATOR(PodSourceB, kPodSourceBName);
HOLOSCAN_REGISTER_OPERATOR(PodSinkB, kPodSinkBName);

// Checks that a registered payload retains flow and interface authoring operations.
// Representative test: AuthoringPayloadRegistry.PreservesStablePayloadRegistrationFamilies.
//   source.output --+--> sink.input <-- public_input (interface alias)
//                   +--> public_output (interface alias)
template <typename T>
void verify_registered_payload_family(holoscan::authoring::AuthoringRegistry& registry,
                                      std::string_view name) {
  // Key setup: Author one flow and two interface aliases using the registered payload row.
  const auto row = registry.payload_type(name);
  ASSERT_TRUE(row);
  const auto source_row =
      holoscan::authoring::detail::make_registered_operator<PayloadSource<T>>("source");
  const auto sink_row =
      holoscan::authoring::detail::make_registered_operator<PayloadSink<T>>("sink");
  holoscan::Graph graph{"stable_payload"};
  const auto source = source_row.add_to_graph(graph, "source", nullptr);
  const auto sink = sink_row.add_to_graph(graph, "sink", nullptr);
  const auto output = source_row.port(source, "output");
  const auto input = sink_row.port(sink, "input");
  const auto input_ports = row->get().make_interface_ports("public_input");
  const auto output_ports = row->get().make_interface_ports("public_output");
  row->get().add_flow(graph, output, input, {});
  row->get().export_input(graph, input_ports.input(), input);
  row->get().export_output(graph, output_ports.output(), output);
  // Key assertion: The row creates one edge and both interface bindings.
  EXPECT_EQ(graph.view().edges().size(), 1U);
  EXPECT_EQ(graph.view().interface_bindings().size(), 2U);
}

}  // namespace

// REQ: AUTHOR-012
TEST(AuthoringPayloadRegistry, ExcludesLocalEnumsAtRegistrationConstraints) {
  // Key setup: Probe registration constraints for two local enums and four stable payload families.
  // Key assertion: Local enums are excluded; scalar, Tensor, schema, and declared POD remain
  // accepted.
  EXPECT_FALSE(kPayloadFactoryAvailable<RegistryEnumA>);
  EXPECT_FALSE(kPayloadFactoryAvailable<RegistryEnumB>);
  EXPECT_FALSE(kAutomaticPayloadRegistrationAvailable<RegistryEnumA>);
  EXPECT_FALSE(kAutomaticPayloadRegistrationAvailable<RegistryEnumB>);
  EXPECT_TRUE(kPayloadFactoryAvailable<std::uint32_t>);
  EXPECT_TRUE(kAutomaticPayloadRegistrationAvailable<std::uint32_t>);
  EXPECT_TRUE(kPayloadFactoryAvailable<holoscan::Tensor>);
  EXPECT_TRUE(kAutomaticPayloadRegistrationAvailable<holoscan::Tensor>);
  EXPECT_TRUE(kPayloadFactoryAvailable<holoscan::RegistrySchema>);
  EXPECT_TRUE(kAutomaticPayloadRegistrationAvailable<holoscan::RegistrySchema>);
  EXPECT_TRUE(kPayloadFactoryAvailable<holoscan::RegistryPodA>);
  EXPECT_TRUE(kAutomaticPayloadRegistrationAvailable<holoscan::RegistryPodA>);
}

// REQ: AUTHOR-012
//   source.output --+--> sink.input <-- public_input (interface alias)
//                   +--> public_output (interface alias)
TEST(AuthoringPayloadRegistry, PreservesStablePayloadRegistrationFamilies) {
  // Key setup: Register schema and POD rows alongside built-in scalar and Tensor rows.
  // Key assertion: Each family retains flow and interface authoring through the shared helper.
  holoscan::authoring::AuthoringRegistry registry;
  ASSERT_TRUE(registry.register_payload(
      holoscan::authoring::detail::make_registered_payload<holoscan::RegistrySchema>("schema")));
  ASSERT_TRUE(registry.register_payload(
      holoscan::authoring::detail::make_registered_payload<holoscan::RegistryPodA>("pod")));

  verify_registered_payload_family<std::uint32_t>(registry, "holoscan.uint32");
  verify_registered_payload_family<holoscan::Tensor>(registry, "holoscan.Tensor");
  verify_registered_payload_family<holoscan::RegistrySchema>(registry, "schema");
  verify_registered_payload_family<holoscan::RegistryPodA>(registry, "pod");
}

// REQ: GRAPH-043
TEST(AuthoringPayloadRegistry, NamesProjectionPortFieldAndAvailableFieldsOnFailure) {
  // Key setup: Project an unknown field through the registered descriptor-payload thunk.
  const auto payload =
      holoscan::authoring::detail::make_registered_payload<holoscan::schema::TensorPacketT>(
          "holoscan.schema.TensorPacket");
  const auto source_row = holoscan::authoring::detail::make_registered_operator<TensorPacketSource>(
      "holoscan.test.TensorPacketSource");
  const auto sink_row =
      holoscan::authoring::detail::make_registered_operator<TensorSink>("holoscan.test.TensorSink");
  holoscan::Graph graph{"registered-projection-error"};
  const auto source = source_row.add_to_graph(graph, "source", nullptr);
  const auto sink = sink_row.add_to_graph(graph, "sink", nullptr);

  // Key assertion: The failure identifies the declaration, typo, and canonical field inventory.
  try {
    (void)payload.add_tensor_projection(
        graph, source_row.port(source, "output"), sink_row.port(sink, "input"), "missing", {});
    FAIL() << "unknown projected Tensor field unexpectedly succeeded";
  } catch (const holoscan::RuntimeError& error) {
    EXPECT_EQ(error.code(), holoscan::ErrorCode::kInvalidArgument);
    EXPECT_EQ(holoscan::make_error(error).message,
              "GRAPH_PROJECTED_TENSOR_FIELD_UNKNOWN port=source.output field_name=missing "
              "known_fields=data");
  }
  EXPECT_TRUE(graph.view().edges().empty());
}

// REQ: SCHEMA-PY-004
TEST(AuthoringPayloadRegistry, PublishesImmutableIdentityMatchedPythonFacade) {
  // Key setup: Register the native schema row first, as an owning component does before Python
  // imports its generated Stable-ABI facade.
  holoscan::authoring::AuthoringRegistry registry;
  auto native = holoscan::authoring::detail::make_registered_payload<holoscan::RegistrySchema>(
      "holoscan.test.SchemaFacade");
  ASSERT_FALSE(native.generated_schema_facade);
  ASSERT_TRUE(registry.register_payload(native));
  const auto before = registry.payload_type(native.name);
  ASSERT_TRUE(before);

  std::atomic_bool reader_started{};
  std::atomic_bool stop_reader{};
  std::atomic_bool observed_mutation{};
  std::thread reader([&] {
    reader_started.store(true, std::memory_order_release);
    while (!stop_reader.load(std::memory_order_acquire)) {
      const auto& retained = before->get();
      if (retained.generated_schema_facade || retained.name != native.name ||
          !holoscan::authoring::payload_identities_equal(retained.identity, native.identity)) {
        observed_mutation.store(true, std::memory_order_relaxed);
      }
    }
  });
  while (!reader_started.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  // Mark the otherwise complete test row as facade-capable; generated code obtains this bit from
  // HasPythonSchemaNativeTraits<T> and supplies the corresponding conversion callbacks.
  auto facade = native;
  facade.generated_schema_facade = true;
  const auto installed = registry.install_payload_facade(facade);
  stop_reader.store(true, std::memory_order_release);
  reader.join();
  ASSERT_TRUE(installed) << installed.error().message;
  const auto after = registry.payload_type(native.name);
  ASSERT_TRUE(after);

  // Key assertion: Readers retain an immutable native generation while later lookups acquire the
  // facade generation; repeated installation is idempotent and identity takeover still rejects.
  EXPECT_FALSE(observed_mutation.load(std::memory_order_relaxed));
  EXPECT_NE(&before->get(), &after->get());
  EXPECT_FALSE(before->get().generated_schema_facade);
  EXPECT_TRUE(after->get().generated_schema_facade);
  EXPECT_TRUE(registry.install_payload_facade(facade));
  EXPECT_TRUE(registry.register_payload(native));
  EXPECT_EQ(&registry.payload_type(native.name)->get(), &after->get());
  auto conflict = facade;
  conflict.identity = holoscan::detail::operator_spec_payload_identity<holoscan::schema::TimeT>();
  const auto rejected = registry.install_payload_facade(conflict);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code, holoscan::ErrorCode::kDuplicateName);
  EXPECT_TRUE(
      holoscan::authoring::payload_identities_equal(after->get().identity, native.identity));
}

// REQ: AUTHOR-012
TEST(AuthoringPayloadRegistry, RejectsRowsWithoutForeignSpecBindingThunks) {
  // Key setup: Remove each foreign-Operator binding callback from an otherwise complete row.
  holoscan::authoring::AuthoringRegistry registry;
  const auto stable = registry.payload_type("holoscan.uint32");
  ASSERT_TRUE(stable);

  auto missing_input = stable->get();
  missing_input.name = "holoscan.test.MissingInputBinding";
  missing_input.bind_spec_input = nullptr;
  auto missing_output = stable->get();
  missing_output.name = "holoscan.test.MissingOutputBinding";
  missing_output.bind_spec_output = nullptr;

  // Key assertion: Incomplete foreign-host rows fail before registry publication.
  for (const auto& row : {missing_input, missing_output}) {
    const auto result = registry.register_payload(row);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, holoscan::ErrorCode::kInvalidArgument);
    EXPECT_NE(result.error().message.find("AUTHORING_REGISTRY_ROW_INVALID kind=payload"),
              std::string::npos);
    EXPECT_FALSE(registry.payload_type(row.name));
  }
}

// REQ: AUTHOR-012
TEST(AuthoringPayloadRegistry, RejectsHandBuiltLocalEnumRowsWithoutInsertion) {
  // Key setup: Give otherwise complete rows two same-width local-enum identities.
  // Key assertion: Both rows are rejected with corrective guidance and leave the registry
  // unchanged.
  holoscan::authoring::AuthoringRegistry registry;
  const auto before = registry.payloads().size();
  const auto stable = registry.payload_type("holoscan.uint32");
  ASSERT_TRUE(stable);
  const auto enum_a = holoscan::detail::operator_spec_payload_identity<RegistryEnumA>();
  const auto enum_b = holoscan::detail::operator_spec_payload_identity<RegistryEnumB>();
  ASSERT_TRUE(holoscan::authoring::payload_identities_equal(enum_a, enum_b));

  for (const auto& identity : {enum_a, enum_b}) {
    // Retain every mandatory thunk: rejection must be about capability, not missing functions.
    auto row = stable->get();
    row.name = "holoscan.test.LocalEnum";
    row.identity = identity;
    const auto result = registry.register_payload(row);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, holoscan::ErrorCode::kInvalidArgument);
    EXPECT_NE(result.error().message.find("AUTHORING_REGISTRY_ROW_INVALID"), std::string::npos);
    EXPECT_NE(result.error().message.find("process-local"), std::string::npos);
    EXPECT_NE(result.error().message.find("typed C++"), std::string::npos);
    EXPECT_FALSE(registry.payload_type(row.name));
    EXPECT_FALSE(registry.payload_type(identity));
    EXPECT_EQ(registry.payloads().size(), before);
  }
  EXPECT_TRUE(registry.registration_status());
  EXPECT_EQ(&registry.payload_type("holoscan.uint32")->get(), &stable->get());
}

// REQ: AUTHOR-012
TEST(AuthoringPayloadRegistry, LocalEnumGuidanceFailurePreservesRejectionCode) {
  // Key setup: Exhaust each allocation made while rejecting a hand-built local-enum row.
  // Key assertion: Every failure preserves kInvalidArgument and inserts no row.
  holoscan::authoring::AuthoringRegistry registry;
  const auto stable = registry.payload_type("holoscan.uint32");
  ASSERT_TRUE(stable);
  auto row = stable->get();
  row.name = "holoscan.test.ExhaustedLocalEnum";
  row.identity = holoscan::detail::operator_spec_payload_identity<RegistryEnumA>();
  std::size_t allocations{};
  {
    const holoscan::test::ScopedAllocationCounter counter;
    (void)registry.register_payload(row);
    allocations = counter.count();
  }
  ASSERT_GT(allocations, 0U);
  for (std::size_t checkpoint = 0U; checkpoint < allocations; ++checkpoint) {
    holoscan::expected<void, holoscan::Error> result;
    {
      const holoscan::test::ScopedAllocationExhaustion exhausted{checkpoint};
      result = registry.register_payload(row);
    }
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, holoscan::ErrorCode::kInvalidArgument);
    EXPECT_FALSE(registry.payload_type(row.name));
    if (!result.error().message.empty()) {
      EXPECT_NE(result.error().message.find("AUTHORING_REGISTRY_ROW_INVALID"), std::string::npos);
    }
  }
}

// REQ: AUTHOR-012
TEST(AuthoringPayloadRegistry, RetainsAutomaticLocalEnumRowRejection) {
  // Key setup: Submit a hand-built local-enum row through automatic registration.
  // Key assertion: The registry retains its rejection status and guidance without inserting the
  // row.
  holoscan::authoring::AuthoringRegistry registry;
  const auto before = registry.payloads().size();
  const auto stable = registry.payload_type("holoscan.uint32");
  ASSERT_TRUE(stable);
  auto row = stable->get();
  row.name = "holoscan.test.AutomaticLocalEnum";
  row.identity = holoscan::detail::operator_spec_payload_identity<RegistryEnumA>();

  EXPECT_FALSE(registry.register_payload_automatically(row));
  EXPECT_FALSE(registry.payload_type(row.name));
  EXPECT_EQ(registry.payloads().size(), before);
  const auto status = registry.registration_status();
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().code, holoscan::ErrorCode::kInvalidArgument);
  EXPECT_NE(status.error().message.find("AUTHORING_REGISTRY_ROW_INVALID"), std::string::npos);
  EXPECT_NE(status.error().message.find("process-local"), std::string::npos);
  EXPECT_NE(status.error().message.find("typed C++"), std::string::npos);
}

// REQ: AUTHOR-012
//   source.output    sink.input    interface.{input,output}
//   (unconnected; every erased resolution attempt is rejected)
TEST(AuthoringPayloadRegistry, RejectsLocalEnumOpaqueResolution) {
  // Key setup: Resolve local-enum operator and interface ports through the erased API.
  // Key assertion: Even matching enum identities are rejected without changing the graph.
  using Source = PayloadSource<RegistryEnumA>;
  using Sink = PayloadSink<RegistryEnumA>;
  using Probe = holoscan::authoring::detail::PayloadRegistrationThunks<holoscan::RegistryPodB>;
  const auto source_row =
      holoscan::authoring::detail::make_registered_operator<Source>("holoscan.test.EnumSource");
  const auto sink_row =
      holoscan::authoring::detail::make_registered_operator<Sink>("holoscan.test.EnumSink");
  holoscan::Graph graph{"local_enum_opaque"};
  const auto source = source_row.add_to_graph(graph, "source", nullptr);
  const auto sink = sink_row.add_to_graph(graph, "sink", nullptr);
  const auto output = source_row.port(source, "output");
  const auto input = sink_row.port(sink, "input");
  const auto ports = holoscan::authoring::InterfacePortPair::make<RegistryEnumA>("interface");
  const auto before = graph.view().identity();
  const auto enum_a = holoscan::detail::operator_spec_payload_identity<RegistryEnumA>();
  const auto enum_b = holoscan::detail::operator_spec_payload_identity<RegistryEnumB>();
  ASSERT_TRUE(holoscan::authoring::payload_identities_equal(enum_a, enum_b));

  // Both operator-backed and held ports must reject even equal-width or same-enum requests.
  for (const auto& identity : {enum_a, enum_b}) {
    const auto expect_rejected = [&](const auto& port, holoscan::PortDirection direction) {
      try {
        (void)Probe::resolve(port, identity, direction);
        FAIL() << "local enum escaped erased resolution";
      } catch (const holoscan::RuntimeError& error) {
        EXPECT_EQ(error.code(), holoscan::ErrorCode::kIncompatible);
        EXPECT_NE(std::string_view(error.what()).find("GRAPH_PORT_TYPE_MISMATCH"),
                  std::string_view::npos);
        EXPECT_NE(std::string_view(error.what()).find("process-local"), std::string_view::npos);
        EXPECT_NE(std::string_view(error.what()).find("typed C++"), std::string_view::npos);
      }
    };
    expect_rejected(output, holoscan::PortDirection::kOutput);
    expect_rejected(input, holoscan::PortDirection::kInput);
    expect_rejected(ports.output(), holoscan::PortDirection::kOutput);
    expect_rejected(ports.input(), holoscan::PortDirection::kInput);
  }
  EXPECT_EQ(graph.view().identity(), before);
  EXPECT_TRUE(graph.view().edges().empty());
  EXPECT_TRUE(graph.view().interface_bindings().empty());
}

// REQ: AUTHOR-012
//   source.output -X-> sink.input <--X- interface.input
//   source.output -X-> interface.output
//   (all attempted connections/aliases are rejected)
TEST(AuthoringPayloadRegistry, RejectsLocalEnumPortsThroughScalarThunks) {
  // Key setup: Use a scalar payload row with source and sink ports of distinct local-enum types.
  // Key assertion: Flow and interface operations reject the ports without changing the graph.
  using Source = PayloadSource<RegistryEnumA>;
  using Sink = PayloadSink<RegistryEnumB>;
  const auto source_row =
      holoscan::authoring::detail::make_registered_operator<Source>("holoscan.test.EnumSource");
  const auto sink_row =
      holoscan::authoring::detail::make_registered_operator<Sink>("holoscan.test.EnumSink");
  holoscan::Graph graph{"local_enum_scalar_thunk"};
  const auto source = source_row.add_to_graph(graph, "source", nullptr);
  const auto sink = sink_row.add_to_graph(graph, "sink", nullptr);
  const auto output = source_row.port(source, "output");
  const auto input = sink_row.port(sink, "input");
  const auto row = holoscan::authoring::detail::make_registered_payload<std::uint32_t>("scalar");
  const auto ports = row.make_interface_ports("interface");
  const auto before = graph.view().identity();

  EXPECT_THROW(row.add_flow(graph, output, input, {}), holoscan::RuntimeError);
  EXPECT_THROW(row.export_input(graph, ports.input(), input), holoscan::RuntimeError);
  EXPECT_THROW(row.export_output(graph, ports.output(), output), holoscan::RuntimeError);
  EXPECT_EQ(graph.view().identity(), before);
  EXPECT_TRUE(graph.view().edges().empty());
  EXPECT_TRUE(graph.view().interface_bindings().empty());
}

// REQ: AUTHOR-012
// REQ: GRAPH-036
//   source.output --+--> sink.input <-- public_input (interface alias)
//                   +--> public_output (interface alias)
TEST(AuthoringPayloadRegistry, PreservesTypedLocalEnumFlowsAndInterfaces) {
  // Key setup: Connect and export ports of the same local enum through the typed C++ API.
  // Key assertion: Typed authoring retains one flow and two interface bindings.
  holoscan::Graph graph{"typed_local_enum"};
  auto source = graph.op<PayloadSource<RegistryEnumA>>("source");
  auto sink = graph.op<PayloadSink<RegistryEnumA>>("sink");
  const holoscan::Input<RegistryEnumA> public_input{"public_input"};
  const holoscan::Output<RegistryEnumA> public_output{"public_output"};

  graph.add_flow(source->output, sink->input);
  graph.export_input(public_input, sink->input);
  graph.export_output(public_output, source->output);
  EXPECT_EQ(graph.view().edges().size(), 1U);
  EXPECT_EQ(graph.view().interface_bindings().size(), 2U);
}

// REQ: AUTHOR-021
TEST(AuthoringPayloadRegistry, InstallsTheClosedBuiltInCatalog) {
  // Key setup: Initialize the SDK schema provider and enumerate the process-global catalog.
  ASSERT_TRUE(holoscan::schema::register_sdk_schema_payloads());
  ASSERT_TRUE(holoscan::schema::register_sdk_schema_payloads());
  const auto rows = holoscan::authoring::authoring_registry().payloads();
  constexpr std::array expected_names{
      "holoscan.Tensor",
      "holoscan.boolean",
      "holoscan.float32",
      "holoscan.float64",
      "holoscan.int16",
      "holoscan.int32",
      "holoscan.int64",
      "holoscan.int8",
      "holoscan.schema.Header",
      "holoscan.schema.String",
      "holoscan.schema.TensorPacket",
      "holoscan.schema.Time",
      "holoscan.uint16",
      "holoscan.uint32",
      "holoscan.uint64",
      "holoscan.uint8",
  };
  ASSERT_EQ(rows.size(), expected_names.size());

  // Key assertion: The catalog is closed over every supported built-in payload identity.
  for (const std::string_view name : expected_names) {
    EXPECT_NE(
        std::find_if(
            rows.begin(), rows.end(), [name](const auto& row) { return row.get().name == name; }),
        rows.end());
  }
}

// REQ: AUTHOR-003
TEST(AuthoringPayloadRegistry, RetainsFailureBeforeAutomaticRowConstructionCompletes) {
  // Key setup: Fail the first allocation while constructing a valid raw-POD registration row.
  constexpr std::string_view kName{"holoscan.test.AutomaticRegistryPod"};
  auto& registry = holoscan::authoring::authoring_registry();
  ASSERT_FALSE(registry.payload_type(kName));
  bool registered = true;
  {
    const holoscan::test::ScopedAllocationFailure failure{0U};
    registered =
        holoscan::authoring::detail::register_payload_automatically<holoscan::RegistryPodA>(kName);
  }

  // Key assertion: The row stays absent while process registration status retains the failure.
  EXPECT_FALSE(registered);
  EXPECT_FALSE(registry.payload_type(kName));
  const auto status = registry.registration_status();
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().code, holoscan::ErrorCode::kFailure);
}

// REQ: PY-CHANNEL-001
TEST(AuthoringPayloadRegistry, ChannelThunksOwnTypedScalarEndpointsAndPreserveErrors) {
  // Topology (name-based, registry-only): opaque publisher --[uint32]--> opaque subscriber
  // Key setup: Create and operate both endpoints without naming ChannelPublisher<T> or
  // ChannelSubscriber<T> at the call site.
  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  auto& registry = holoscan::authoring::authoring_registry();
  const auto uint32_row = registry.payload_type("holoscan.uint32");
  const auto int32_row = registry.payload_type("holoscan.int32");
  ASSERT_TRUE(uint32_row && int32_row);

  holoscan::ChannelOptions options{};
  options.enable_event_wake = true;
  const std::string channel_name =
      "authoring_registry_channel_" + std::to_string(static_cast<long>(::getpid()));
  auto publisher =
      uint32_row->get().channel->create_name_publisher(*runtime, channel_name, options);
  auto subscriber =
      uint32_row->get().channel->create_name_subscriber(*runtime, channel_name, options);
  ASSERT_TRUE(publisher) << holoscan::runtime_error_message(publisher.error());
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());

  const auto peer_deadline = std::chrono::steady_clock::now() + std::chrono::seconds{2};
  while (!uint32_row->get().channel->has_subscribers(*publisher) &&
         std::chrono::steady_clock::now() < peer_deadline) {
    ASSERT_TRUE(uint32_row->get().channel->update_connections(*publisher));
    ASSERT_TRUE(uint32_row->get().channel->refresh_connections(*subscriber));
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  ASSERT_TRUE(uint32_row->get().channel->has_subscribers(*publisher));

  auto readiness = uint32_row->get().channel->subscriber_readiness(*subscriber);
  ASSERT_TRUE(readiness) << holoscan::runtime_error_message(readiness.error());
  EXPECT_GE(readiness->descriptor(), 0);
  const auto wait_while_attached = uint32_row->get().channel->wait(
      *subscriber, std::chrono::steady_clock::now() + std::chrono::seconds{30}, {});
  ASSERT_FALSE(wait_while_attached);
  EXPECT_EQ(wait_while_attached.error().code, holoscan::ErrorCode::kInvalidArgument);
  readiness->detach();

  std::stop_source cancelled;  // NOLINT(misc-const-correctness): request_stop() mutates the source.
  cancelled.request_stop();
  const auto interrupted =
      uint32_row->get().channel->wait(*subscriber,
                                      std::chrono::steady_clock::now() + std::chrono::seconds{30},
                                      cancelled.get_token());
  ASSERT_FALSE(interrupted);
  EXPECT_EQ(interrupted.error().code, holoscan::ErrorCode::kInterrupted);

  const auto mismatch = int32_row->get().channel->emit_scalar(
      *publisher, holoscan::ParamScalarValue::from(std::int32_t{19}), {});
  ASSERT_FALSE(mismatch);
  EXPECT_EQ(mismatch.error().code, holoscan::ErrorCode::kInvalidArgument);
  EXPECT_EQ(mismatch.error().message, "AUTHORING_CHANNEL_PUBLISHER_TYPE_MISMATCH");

  const auto emitted = uint32_row->get().channel->emit_scalar(
      *publisher, holoscan::ParamScalarValue::from(std::uint32_t{37U}), {});
  ASSERT_TRUE(emitted) << holoscan::runtime_error_message(emitted.error());
  EXPECT_EQ(*emitted, 1U);
  auto received = uint32_row->get().channel->wait_and_receive_scalar(
      *subscriber, std::chrono::steady_clock::now() + std::chrono::seconds{2}, {});
  ASSERT_TRUE(received) << holoscan::runtime_error_message(received.error());
  std::uint32_t value{};
  ASSERT_TRUE(received->data.try_as(value));
  EXPECT_EQ(value, 37U);
  EXPECT_TRUE(received->has_data);

  ASSERT_TRUE(uint32_row->get().channel->publisher_close(*publisher));
  const auto closed = uint32_row->get().channel->wait_and_receive_scalar(
      *subscriber, std::chrono::steady_clock::now() + std::chrono::seconds{2}, {});
  ASSERT_FALSE(closed);
  EXPECT_EQ(closed.error().code, holoscan::ErrorCode::kClosed);
  ASSERT_TRUE(uint32_row->get().channel->subscriber_close(*subscriber));
}

// REQ: PY-CHANNEL-001
TEST(AuthoringPayloadRegistry, RejectsRowsMissingMandatoryChannelThunks) {
  holoscan::authoring::AuthoringRegistry registry;
  auto row = holoscan::authoring::detail::make_registered_payload<std::uint32_t>(
      "holoscan.test.incomplete-channel-row");
  auto incomplete_channel = *row.channel;
  incomplete_channel.wait = nullptr;
  row.channel = &incomplete_channel;

  const auto registered = registry.register_payload(std::move(row));

  ASSERT_FALSE(registered);
  EXPECT_EQ(registered.error().code, holoscan::ErrorCode::kInvalidArgument);
  EXPECT_NE(registered.error().message.find("AUTHORING_REGISTRY_ROW_INVALID"), std::string::npos);
}

// REQ: PY-CHANNEL-007
TEST(AuthoringPayloadRegistry, TensorChannelRowRequiresCatalogForItsBoundedPublisher) {
  // Key setup: Resolve the built-in Tensor row without naming its endpoint type at the call site.
  auto& registry = holoscan::authoring::authoring_registry();
  const auto tensor = registry.payload_type("holoscan.Tensor");
  ASSERT_TRUE(tensor);
  EXPECT_NE(tensor->get().channel->loan_tensor, nullptr);
  EXPECT_NE(tensor->get().channel->emit_tensor_loan, nullptr);
  EXPECT_NE(tensor->get().channel->emit_tensor_copy, nullptr);
  EXPECT_NE(tensor->get().channel->try_receive_tensor, nullptr);
  EXPECT_NE(tensor->get().channel->wait_and_receive_tensor, nullptr);

  auto runtime = holoscan::ChannelRuntime::create();
  ASSERT_TRUE(runtime) << holoscan::runtime_error_message(runtime.error());
  holoscan::ChannelOptions options{};
  options.enable_event_wake = true;
  const std::string channel_name =
      "authoring_registry_tensor_channel_" + std::to_string(static_cast<long>(::getpid()));
  auto publisher = tensor->get().channel->create_name_publisher(*runtime, channel_name, options);
  auto subscriber = tensor->get().channel->create_name_subscriber(*runtime, channel_name, options);

  // Key assertion: A subscriber needs no allocation policy, while a publisher cannot invent the
  // compiler-owned bounded Tensor contract that only an endpoint catalog carries.
  ASSERT_FALSE(publisher);
  EXPECT_EQ(publisher.error().code, holoscan::ErrorCode::kNotSupported);
  EXPECT_EQ(publisher.error().message, "CHANNEL_NAME_TENSOR_PUBLISHER_REQUIRES_CATALOG");
  ASSERT_TRUE(subscriber) << holoscan::runtime_error_message(subscriber.error());
  const auto empty = tensor->get().channel->try_receive_tensor(*subscriber);
  ASSERT_FALSE(empty);
  EXPECT_EQ(empty.error().code, holoscan::ErrorCode::kNotReady);
  EXPECT_TRUE(tensor->get().channel->subscriber_close(*subscriber));
}

// REQ: AUTHOR-021
// REQ: PY-CHANNEL-008
TEST(AuthoringPayloadRegistry, DescriptorTopicRowsEnforceTypedDispatch) {
  // Key setup: Resolve one descriptor payload and one scalar payload through erased registry rows.
  auto& registry = holoscan::authoring::authoring_registry();
  ASSERT_TRUE(holoscan::schema::register_sdk_schema_payloads());
  const auto descriptor = registry.payload_type("holoscan.schema.TensorPacket");
  const auto scalar = registry.payload_type("holoscan.uint32");
  ASSERT_TRUE(descriptor && scalar);
  const holoscan::TensorBoundaryContract contract{
      .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                         .dtype = DLDataType{kDLUInt, 8U, 1U},
                         .rank = 1U},
      .bounds = {.max_rank = 1U, .max_byte_span = 64U},
  };
  const std::array<std::pair<std::string_view, holoscan::TensorBoundaryContract>, 1U> fields{{
      {"data", contract},
  }};
  const auto expect_token = [](const auto& operation, std::string_view token) {
    try {
      operation();
      FAIL() << "incompatible descriptor dispatch unexpectedly succeeded";
    } catch (const holoscan::RuntimeError& error) {
      EXPECT_NE(std::string_view(error.what()).find(token), std::string_view::npos);
    }
  };

  holoscan::Graph graph{"descriptor_payload_rows"};
  const auto imported = descriptor->get().import_descriptor_topic(
      graph, "holomq://descriptor", holoscan::TopicOptions{}, fields);

  // Key assertion: The matching descriptor path authors successfully and every crossed path fails.
  EXPECT_TRUE(imported.port());
  expect_token(
      [&] {
        (void)descriptor->get().import_topic(
            graph, "holomq://descriptor-direct", holoscan::TopicOptions{});
      },
      "AUTHORING_DESCRIPTOR_TENSOR_CONTRACT_REQUIRED");
  expect_token(
      [&] {
        (void)descriptor->get().export_topic(
            graph, "holomq://descriptor-direct-export", holoscan::TopicOptions{});
      },
      "AUTHORING_DESCRIPTOR_TENSOR_CONTRACT_REQUIRED");
  expect_token(
      [&] {
        (void)scalar->get().import_descriptor_topic(
            graph, "holomq://scalar-descriptor", holoscan::TopicOptions{}, fields);
      },
      "AUTHORING_DESCRIPTOR_TENSOR_PAYLOAD_REQUIRED");
  expect_token(
      [&] {
        (void)scalar->get().export_descriptor_topic(
            graph, "holomq://scalar-descriptor-export", holoscan::TopicOptions{}, fields);
      },
      "AUTHORING_DESCRIPTOR_TENSOR_PAYLOAD_REQUIRED");
}

// REQ: AUTHOR-021
TEST(AuthoringPayloadRegistry, AuthorsFlowInterfacesTopicsAndConfiguratorsThroughRows) {
  // Graph:
  //
  // public_input --> sink    source --> public_output
  //                              |
  //                              +-------> sink
  //
  // Key setup: Author a flow, two interface bindings, and two topics exclusively through rows.
  auto& registry = holoscan::authoring::authoring_registry();
  const auto source_row = registry.operator_type(kUIntSourceName);
  const auto sink_row = registry.operator_type(kUIntSinkName);
  const auto source_description = registry.describe_operator(kUIntSourceName);
  const auto sink_description = registry.describe_operator(kUIntSinkName);
  ASSERT_TRUE(source_row);
  ASSERT_TRUE(sink_row);
  ASSERT_TRUE(source_description);
  ASSERT_TRUE(sink_description);
  ASSERT_EQ(source_description->ports.size(), 1U);
  const auto payload = registry.payload_type(source_description->ports.front().payload_identity);
  ASSERT_TRUE(payload) << payload.error().message;

  holoscan::Graph graph{"payload_rows"};
  const auto source = source_row->get().add_to_graph(graph, "source", nullptr);
  const auto sink = sink_row->get().add_to_graph(graph, "sink", nullptr);
  const auto output = source_row->get().port(source, "output");
  const auto input = sink_row->get().port(sink, "input");
  auto flow = payload->get().add_flow(graph, output, input, {});
  EXPECT_TRUE(flow);
  payload->get().configure_input(graph, input).queue_depth(3U).diagnostic_alias("registry-input");
  payload->get()
      .configure_output(graph, output)
      .max_emits_per_compute(2U)
      .late_joiner_history_depth(4U);
  flow.queue_policy(holoscan::QueuePolicy::kDropOldest).max_inflight_samples(5U);

  auto input_interface = payload->get().make_interface_ports("public_input");
  auto output_interface = payload->get().make_interface_ports("public_output");
  payload->get().export_input(graph, input_interface.input(), input);
  payload->get().export_output(graph, output_interface.output(), output);

  const auto imported =
      payload->get().import_topic(graph, "holomq://registry/import", holoscan::TopicOptions{});
  const auto exported =
      payload->get().export_topic(graph, "holomq://registry/export", holoscan::TopicOptions{});
  EXPECT_TRUE(imported.port());
  EXPECT_TRUE(exported.port());

  const holoscan::GraphView view = graph.view();

  // Key assertion: Row-based authoring preserves flow, interface, topic, and override records.
  EXPECT_EQ(view.edges().size(), 1U);
  EXPECT_EQ(view.interface_bindings().size(), 2U);
  EXPECT_EQ(view.external_topics().size(), 2U);
  ASSERT_EQ(view.edges().front().graph_override.queue_policy, holoscan::QueuePolicy::kDropOldest);
  ASSERT_EQ(view.edges().front().graph_override.max_inflight_samples, 5U);
  const auto configured_input =
      std::find_if(view.ports().begin(), view.ports().end(), [](const auto& port) {
        return port.name == "input" && port.owner_kind == holoscan::PortOwnerKind::kOperator;
      });
  const auto configured_output =
      std::find_if(view.ports().begin(), view.ports().end(), [](const auto& port) {
        return port.name == "output" && port.owner_kind == holoscan::PortOwnerKind::kOperator;
      });
  ASSERT_NE(configured_input, view.ports().end());
  ASSERT_NE(configured_output, view.ports().end());
  EXPECT_EQ(configured_input->graph_override.queue_depth, 3U);
  EXPECT_EQ(configured_input->graph_override.diagnostic_alias, "registry-input");
  EXPECT_EQ(configured_output->graph_override.max_emits_per_compute, 2U);
  EXPECT_EQ(configured_output->graph_override.late_joiner_history_depth, 4U);
}

// REQ: AUTHOR-001
TEST(AuthoringPayloadRegistry, PreparesOpaqueTopicStorageBeforeGraphMutation) {
  // Key setup: Fail each allocation made by the erased import and export paths in turn.
  const auto payload = holoscan::authoring::authoring_registry().payload_type("holoscan.uint32");
  ASSERT_TRUE(payload);
  const auto allocation_count = [&](bool imported) {
    holoscan::Graph graph{imported ? "measure_import" : "measure_export"};
    const holoscan::test::ScopedAllocationCounter counter;
    if (imported) {
      (void)payload->get().import_topic(graph, "holomq://registry/measure-import", {});
    } else {
      (void)payload->get().export_topic(graph, "holomq://registry/measure-export", {});
    }
    return counter.count();
  };

  const auto verify_atomic = [&](bool imported, std::size_t allocations) {
    for (std::size_t checkpoint = 0U; checkpoint < allocations; ++checkpoint) {
      holoscan::Graph graph{imported ? "fail_import" : "fail_export"};
      bool failed = false;
      bool unexpected_exception = false;
      {
        const holoscan::test::ScopedAllocationFailure failure{checkpoint};
        try {
          if (imported) {
            (void)payload->get().import_topic(graph, "holomq://registry/fail-import", {});
          } else {
            (void)payload->get().export_topic(graph, "holomq://registry/fail-export", {});
          }
        } catch (const std::bad_alloc&) {
          failed = true;
        } catch (const std::exception&) {
          unexpected_exception = true;
        } catch (...) {
          unexpected_exception = true;
        }
      }
      if (unexpected_exception) {
        ADD_FAILURE() << "unexpected " << (imported ? "import" : "export")
                      << " exception at checkpoint=" << checkpoint;
        continue;
      }
      if (failed) {
        EXPECT_TRUE(graph.view().external_topics().empty()) << "checkpoint=" << checkpoint;
        if (imported) {
          (void)payload->get().import_topic(graph, "holomq://registry/retry-import", {});
        } else {
          (void)payload->get().export_topic(graph, "holomq://registry/retry-export", {});
        }
        EXPECT_EQ(graph.view().external_topics().size(), 1U) << "checkpoint=" << checkpoint;
      }
    }
  };

  // Key assertion: Every injected allocation failure leaves no committed boundary, and the same
  // Graph remains usable.
  verify_atomic(true, allocation_count(true));
  verify_atomic(false, allocation_count(false));
}

// REQ: AUTHOR-001
TEST(AuthoringPayloadRegistry, RejectsIdentityMismatchBeforeTypedDowncast) {
  // Attempted Graph:
  //
  // source(uint32) --X--> sink(int32)
  //
  // Key setup: Select the uint32 thunk, but provide an int32 destination port reference.
  auto& registry = holoscan::authoring::authoring_registry();
  const auto source_row = registry.operator_type(kUIntSourceName);
  const auto sink_row = registry.operator_type(kIntSinkName);
  const auto source_description = registry.describe_operator(kUIntSourceName);
  const auto sink_description = registry.describe_operator(kIntSinkName);
  ASSERT_TRUE(source_row && sink_row && source_description && sink_description);
  ASSERT_EQ(source_description->ports.size(), 1U);
  const auto payload = registry.payload_type(source_description->ports.front().payload_identity);
  ASSERT_TRUE(payload);

  holoscan::Graph graph{"payload_mismatch"};
  const auto source = source_row->get().add_to_graph(graph, "source", nullptr);
  const auto sink = sink_row->get().add_to_graph(graph, "sink", nullptr);

  // Key assertion: Identity validation rejects the edge before either declaration is downcast.
  try {
    payload->get().add_flow(
        graph, source_row->get().port(source, "output"), sink_row->get().port(sink, "input"), {});
    FAIL() << "mismatched payload identities unexpectedly reached typed authoring";
  } catch (const holoscan::RuntimeError& error) {
    EXPECT_EQ(error.code(), holoscan::ErrorCode::kIncompatible);
    EXPECT_NE(std::string_view(error.what()).find("GRAPH_PORT_TYPE_MISMATCH"),
              std::string_view::npos);
  }
}

// REQ: AUTHOR-001
TEST(AuthoringPayloadRegistry, PreservesDistinctCompatiblePodDeclarationTypes) {
  // Graph:
  //
  // public_input(A) --> sink(B)    source(B) --> public_output(A)
  //                         ^
  //                         |
  //                    source(A)
  //
  // Key setup: Use an A registry row with distinct A/B declarations sharing one POD contract.
  auto& global_registry = holoscan::authoring::authoring_registry();
  const auto source_a_row = global_registry.operator_type(kPodSourceAName);
  const auto source_b_row = global_registry.operator_type(kPodSourceBName);
  const auto sink_b_row = global_registry.operator_type(kPodSinkBName);
  ASSERT_TRUE(source_a_row && source_b_row && sink_b_row);
  holoscan::authoring::AuthoringRegistry registry;
  ASSERT_TRUE(registry.register_payload(
      holoscan::authoring::detail::make_registered_payload<holoscan::RegistryPodA>(
          "holoscan.test.RegistryPod")));
  const auto payload = registry.payload_type("holoscan.test.RegistryPod");
  ASSERT_TRUE(payload);

  holoscan::Graph graph{"compatible_pod_rows"};
  const auto source_a = source_a_row->get().add_to_graph(graph, "source_a", nullptr);
  const auto source_b = source_b_row->get().add_to_graph(graph, "source_b", nullptr);
  const auto sink_b = sink_b_row->get().add_to_graph(graph, "sink_b", nullptr);
  payload->get().add_flow(graph,
                          source_a_row->get().port(source_a, "output"),
                          sink_b_row->get().port(sink_b, "input"),
                          {});
  auto input_interface = payload->get().make_interface_ports("public_input");
  auto output_interface = payload->get().make_interface_ports("public_output");
  payload->get().export_input(
      graph, input_interface.input(), sink_b_row->get().port(sink_b, "input"));
  payload->get().export_output(
      graph, output_interface.output(), source_b_row->get().port(source_b, "output"));

  // Key assertion: The base-declaration gateway preserves all three compatible operations.
  const auto view = graph.view();
  EXPECT_EQ(view.edges().size(), 1U);
  EXPECT_EQ(view.interface_bindings().size(), 2U);
}

// REQ: AUTHOR-001
TEST(AuthoringPayloadRegistry, RejectsExportIdentityMismatchBeforeTypedDowncast) {
  // Attempted Graph interfaces:
  //
  // public_input(uint32) --X--> sink(int32)
  // source(int32) --X--> public_output(uint32)
  //
  // Key setup: Pair uint32 public interfaces with int32 internal Operator ports.
  auto& registry = holoscan::authoring::authoring_registry();
  const auto source_row = registry.operator_type(kIntSourceName);
  const auto sink_row = registry.operator_type(kIntSinkName);
  const auto payload = registry.payload_type("holoscan.uint32");
  ASSERT_TRUE(source_row && sink_row && payload);

  holoscan::Graph graph{"payload_export_mismatch"};
  const auto source = source_row->get().add_to_graph(graph, "source", nullptr);
  const auto sink = sink_row->get().add_to_graph(graph, "sink", nullptr);
  auto input_interface = payload->get().make_interface_ports("public_input");
  auto output_interface = payload->get().make_interface_ports("public_output");

  const auto expect_mismatch = [](const auto& operation) {
    try {
      operation();
      FAIL() << "mismatched export port unexpectedly reached typed authoring";
    } catch (const holoscan::RuntimeError& error) {
      EXPECT_EQ(error.code(), holoscan::ErrorCode::kIncompatible);
      EXPECT_NE(std::string_view(error.what()).find("GRAPH_PORT_TYPE_MISMATCH"),
                std::string_view::npos);
    }
  };

  // Key assertion: Both export directions validate identity before any typed downcast.
  expect_mismatch([&] {
    payload->get().export_input(
        graph, input_interface.input(), sink_row->get().port(sink, "input"));
  });
  expect_mismatch([&] {
    payload->get().export_output(
        graph, output_interface.output(), source_row->get().port(source, "output"));
  });
}

// REQ: AUTHOR-001
TEST(AuthoringClockRegistry, InstallsAndAuthorsEverySdkClock) {
  // Key setup: Instantiate every SDK clock row in one otherwise empty graph.
  //
  //   RealtimeClock    ManualClock    SyntheticClock
  const auto rows = holoscan::authoring::authoring_registry().clocks();
  ASSERT_EQ(rows.size(), 3U);
  holoscan::Graph graph{"clock_rows"};
  for (const auto& row : rows) {
    const std::string_view qualified = row.get().name;
    const std::size_t separator = qualified.rfind('.');
    ASSERT_NE(separator, std::string_view::npos);
    const auto key = row.get().add_to_graph(graph, qualified.substr(separator + 1U));
    EXPECT_FALSE(key.is_default());
  }

  // Key assertion: Every catalog row contributes exactly one non-default clock record.
  EXPECT_EQ(graph.view().clocks().size(), rows.size());
}
