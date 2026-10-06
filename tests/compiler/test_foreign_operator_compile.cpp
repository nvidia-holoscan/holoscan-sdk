/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/holoscan.hpp>

#include "compile_test_graph.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/process_placement_internal.hpp"

namespace holoscan {

// Supplies the fixed process-local identity used by opaque-object policy tests.
// Representative test: ForeignOperatorCompileTest.AppliesTheOpaqueObjectPolicyPerEdge.
struct PyObjectPayloadPolicyProbe {
  std::uint32_t value{};
};

template <>
consteval SchemaIdentity schema_identity<PyObjectPayloadPolicyProbe>() {
  return make_schema_identity(make_schema_family_identity("holoscan.PyObjectPayload"),
                              make_flatbuffer_file_identifier("PYOB"));
}

namespace detail {

template <>
const RuntimePortTypeOps* runtime_port_type_ops<PyObjectPayloadPolicyProbe>() noexcept {
  static const RuntimePortTypeOps ops{
      .payload_size = sizeof(PyObjectPayloadPolicyProbe),
      .payload_alignment = alignof(PyObjectPayloadPolicyProbe),
      .wire_capacity = 0U,
      .sample_size = sizeof(Sample<PyObjectPayloadPolicyProbe>),
      .sample_alignment = alignof(Sample<PyObjectPayloadPolicyProbe>),
      .predicate_view_size = sizeof(PredicateInputView<PyObjectPayloadPolicyProbe>),
      .predicate_view_alignment = alignof(PredicateInputView<PyObjectPayloadPolicyProbe>),
      .codec_family = RuntimePayloadCodecFamily::kProcessLocalObject,
      .payload_kind = PayloadKind::kInlineTyped,
  };
  return &ops;
}

}  // namespace detail

namespace {

// Supplies a schema-identity-only edge used by
// ForeignOperatorCompileTest.AppliesTheOpaqueObjectPolicyPerEdge.
class PyObjectPolicySource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Output<PyObjectPayloadPolicyProbe> output;
};

// Consumes the policy-probe edge in
// ForeignOperatorCompileTest.AppliesTheOpaqueObjectPolicyPerEdge.
class PyObjectPolicySink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<PyObjectPayloadPolicyProbe> input;
};

// Supplies the first serialized endpoint for
// ForeignOperatorCompileTest.SplitsAConnectedDomainToPreserveAcyclicPartitioning.
class SerializationDomainSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }
  [[nodiscard]] OperatorHostMetadata host_metadata() const noexcept override {
    return {.serialization_domain = "python.gil"};
  }

  Output<std::uint32_t> output;
};

// Provides an ordinary native fan-out leg between serialized endpoints in
// ForeignOperatorCompileTest.SplitsAConnectedDomainToPreserveAcyclicPartitioning.
class SerializationDomainRelay final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.output(output, "output");
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
  Output<std::uint32_t> output;
};

// Supplies the terminal serialized endpoint for
// ForeignOperatorCompileTest.SplitsAConnectedDomainToPreserveAcyclicPartitioning.
class SerializationDomainSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(left, "left");
    spec.input(right, "right");
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }
  [[nodiscard]] OperatorHostMetadata host_metadata() const noexcept override {
    return {.serialization_domain = "python.gil"};
  }

  Input<std::uint32_t> left;
  Input<std::uint32_t> right;
};

// Returns deliberately malformed host metadata for
// ForeignOperatorCompileTest.RejectsInvalidHostMetadataBeforeGraphPublication.
class InvalidHostMetadataOperator final : public Operator<> {
 public:
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }
  [[nodiscard]] OperatorHostMetadata host_metadata() const noexcept override {
    // Deliberately forge an invalid closed-enum value to exercise compiler validation.
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    return {.kind = static_cast<OperatorHostKind>(255U)};
  }
};

[[nodiscard]] ForeignOperatorResult successful_compute(void*, ExecutionContext*) noexcept {
  return ForeignOperatorResult::success();
}

[[nodiscard]] ForeignOperatorCallbacks stateless_callbacks() {
  return ForeignOperatorCallbacks{ForeignOperatorCallbackTable{.compute_cb = &successful_compute}};
}

// Declares one optional input and one ordinary output through the erased payload row. Used by
// ForeignOperatorCompileTest.OptionalInputThunkPreservesPermissionAndOutputBinding.
struct OptionalPortSetupProbe {
  const authoring::RegisteredPayloadType* payload{};

  static void destroy(void* context) noexcept {
    delete static_cast<OptionalPortSetupProbe*>(context);
  }

  static ForeignOperatorResult setup(void* context, ForeignOperatorSpec* spec) noexcept {
    const auto& probe = *static_cast<OptionalPortSetupProbe*>(context);
    const auto input = spec->input(*probe.payload, "input", 4U, true);
    if (!input) {
      return ForeignOperatorResult{.code = input.error().code};
    }
    const auto output = spec->output(*probe.payload, "output", 2U);
    if (!output) {
      return ForeignOperatorResult{.code = output.error().code};
    }
    return ForeignOperatorResult::success();
  }
};

[[nodiscard]] ForeignOperatorCallbacks optional_port_callbacks() {
  auto payload = authoring::authoring_registry().payload_type("holoscan.uint32");
  if (!payload) {
    throw RuntimeError{payload.error()};
  }
  // ForeignOperatorCallbacks takes ownership of the context and invokes the paired destroy thunk;
  // the analyzer does not model that erased ownership transfer.
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  return ForeignOperatorCallbacks{ForeignOperatorCallbackTable{
      .context = new OptionalPortSetupProbe{&payload->get()},
      .destroy = &OptionalPortSetupProbe::destroy,
      .setup_cb = &OptionalPortSetupProbe::setup,
      .compute_cb = &successful_compute,
  }};
}

[[nodiscard]] bool has_diagnostic(const ExecutionPlan& plan, std::string_view token) {
  return std::any_of(plan.diagnostics().begin(), plan.diagnostics().end(), [&](const auto& item) {
    return item.token == token;
  });
}

[[nodiscard]] ExecutionPlan compile_policy_probe(OpaqueObjectPayloadPolicy policy) {
  Graph graph{"pyobject-policy"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<PyObjectPolicySource>("source");
  const auto sink = graph.op<PyObjectPolicySink>("sink");
  graph.add_flow(source->output, sink->input);
  CompileOptions options;
  options.opaque_object_payloads = policy;
  return compile(graph, std::move(options));
}

// REQ: PY-OP-001
TEST(ForeignOperatorCompileTest, OptionalInputThunkPreservesPermissionAndOutputBinding) {
  //   [unwired] --> foreign.input   foreign.output --> [unwired]
  //
  // Key setup: declare an optional input and ordinary output through the same erased payload row.
  Graph graph{"foreign-optional-input-thunk"};
  static_cast<void>(graph.op<ForeignOperatorCore>("foreign", optional_port_callbacks()));
  const auto view = graph.view();
  const auto ports = view.ports();
  const auto input = std::find_if(
      ports.begin(), ports.end(), [](const auto& port) { return port.name == "input"; });
  const auto output = std::find_if(
      ports.begin(), ports.end(), [](const auto& port) { return port.name == "output"; });

  // Key assertion: the explicit permission reaches only the input declaration; ordinary bounds
  // and the output path remain unchanged.
  ASSERT_NE(input, ports.end());
  EXPECT_TRUE(input->optional);
  EXPECT_EQ(input->setup.queue_depth, 4U);
  ASSERT_NE(output, ports.end());
  EXPECT_FALSE(output->optional);
  EXPECT_EQ(output->setup.max_emits_per_compute, 2U);
}

// REQ: COMPILE-092
// REQ: COMPILE-093
TEST(ForeignOperatorCompileTest, CoPartitionsOneSerializationDomainByDefault) {
  //   first   second   third
  //
  // The disconnected Operators share one serialization domain.
  // Key setup: three disconnected foreign Operators request one common serialization domain.
  Graph graph{"serialization-domain-default"};
  const auto callbacks = stateless_callbacks();
  for (const std::string_view name : {"first", "second", "third"}) {
    static_cast<void>(graph.op<ForeignOperatorCore>(
        name, callbacks, ForeignOperatorOptions{.serialization_domain = "python.gil"}));
  }

  const ExecutionPlan plan = compile(graph);

  // Key assertion: automatic partitioning treats the domain as one co-partition constraint.
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 1U);
  ASSERT_EQ(plan.operator_hosts().size(), 3U);
  EXPECT_TRUE(
      std::all_of(plan.operator_hosts().begin(), plan.operator_hosts().end(), [](const auto& host) {
        return host.kind == OperatorHostKind::kForeign && host.serialization_domain == "python.gil";
      }));
  EXPECT_EQ(plan.execution_partitions().front().operator_paths,
            (std::vector<std::string>{"first", "second", "third"}));
  ASSERT_FALSE(plan.partition_decisions().empty());
  EXPECT_TRUE(std::all_of(plan.partition_decisions().begin(),
                          plan.partition_decisions().end(),
                          [](const auto& decision) {
                            return decision.decision == PartitionDecisionKind::kSerializationDomain;
                          }));
}

// REQ: COMPILE-093
TEST(ForeignOperatorCompileTest, ManualSplitOverridesDomainWithAnAdvisory) {
  //   [first]   [second]   [third]
  //
  // Authored partitions deliberately split one serialization domain.
  // Key setup: place each foreign Operator in a distinct authored partition.
  Graph graph{"serialization-domain-manual-split"};
  const auto callbacks = stateless_callbacks();
  const auto first = graph.op<ForeignOperatorCore>(
      "first", callbacks, ForeignOperatorOptions{.serialization_domain = "python.gil"});
  const auto second = graph.op<ForeignOperatorCore>(
      "second", callbacks, ForeignOperatorOptions{.serialization_domain = "python.gil"});
  const auto third = graph.op<ForeignOperatorCore>(
      "third", callbacks, ForeignOperatorOptions{.serialization_domain = "python.gil"});
  graph.partition("first-partition").add(first);
  graph.partition("second-partition").add(second);
  graph.partition("third-partition").add(third);

  const ExecutionPlan plan = compile(graph);

  // Key assertion: authored placement wins without rejection and reports the exact domain split.
  ASSERT_TRUE(plan.ok()) << plan.json();
  EXPECT_EQ(plan.execution_partitions().size(), 3U);
  EXPECT_EQ(std::count_if(plan.diagnostics().begin(),
                          plan.diagnostics().end(),
                          [](const auto& item) {
                            return item.token == "OPERATOR_SERIALIZATION_DOMAIN_SPLIT_ADVISORY";
                          }),
            1);
  ASSERT_TRUE(has_diagnostic(plan, "OPERATOR_SERIALIZATION_DOMAIN_SPLIT_ADVISORY"));
  const auto advisory =
      std::find_if(plan.diagnostics().begin(), plan.diagnostics().end(), [](const auto& item) {
        return item.token == "OPERATOR_SERIALIZATION_DOMAIN_SPLIT_ADVISORY";
      });
  ASSERT_NE(advisory, plan.diagnostics().end());
  EXPECT_EQ(advisory->provided, "python.gil");
}

// REQ: COMPILE-093
TEST(ForeignOperatorCompileTest, SplitsAConnectedDomainToPreserveAcyclicPartitioning) {
  //   source[d] --> left --\
  //       \------> right ---> sink[d]
  //
  // Key setup: reconnect one serialization domain around two native fan-out legs.
  Graph graph{"serialization-domain-connected-split"};
  const auto source = graph.op<SerializationDomainSource>("source");
  const auto left = graph.op<SerializationDomainRelay>("left");
  const auto right = graph.op<SerializationDomainRelay>("right");
  const auto sink = graph.op<SerializationDomainSink>("sink");
  test::add_compile_test_default_clock(graph);
  graph.add_flow(source->output, left->input);
  graph.add_flow(source->output, right->input);
  graph.add_flow(left->output, sink->left);
  graph.add_flow(right->output, sink->right);

  const ExecutionPlan plan = compile(graph);

  // Key assertion: compile preserves an acyclic cover and reports one forced domain split.
  ASSERT_TRUE(plan.ok()) << plan.json();
  EXPECT_EQ(plan.execution_partitions().size(), 4U);
  EXPECT_EQ(std::count_if(plan.diagnostics().begin(),
                          plan.diagnostics().end(),
                          [](const auto& item) {
                            return item.token == "OPERATOR_SERIALIZATION_DOMAIN_SPLIT_ADVISORY";
                          }),
            1);
}

// REQ: COMPILE-093
TEST(ForeignOperatorCompileTest, EmptyDomainOptsOutOfCoPartitioning) {
  //   first   second
  //
  // The disconnected Operators have no common serialization domain.
  // Key setup: compose two ordinary independent foreign source Operators.
  Graph graph{"serialization-domain-opt-out"};
  const auto callbacks = stateless_callbacks();
  static_cast<void>(graph.op<ForeignOperatorCore>("first", callbacks));
  static_cast<void>(graph.op<ForeignOperatorCore>("second", callbacks));

  const ExecutionPlan plan = compile(graph);

  // Key assertion: absent domain metadata leaves independent source Operators in ordinary
  // automatic partitions.
  ASSERT_TRUE(plan.ok()) << plan.json();
  EXPECT_EQ(plan.execution_partitions().size(), 2U);
  EXPECT_FALSE(has_diagnostic(plan, "OPERATOR_SERIALIZATION_DOMAIN_SPLIT_ADVISORY"));
}

// REQ: COMPILE-094
TEST(ForeignOperatorCompileTest, AppliesTheOpaqueObjectPolicyPerEdge) {
  //   source --PyObjectPayload--> sink
  //
  // Key setup: compile the same opaque-object edge under each policy value.
  const ExecutionPlan allowed = compile_policy_probe(OpaqueObjectPayloadPolicy::kAllow);
  const ExecutionPlan warned = compile_policy_probe(OpaqueObjectPayloadPolicy::kWarn);
  const ExecutionPlan rejected = compile_policy_probe(OpaqueObjectPayloadPolicy::kReject);

  // Key assertion: the tri-state knob has no default noise, one advisory, or one admission error.
  ASSERT_TRUE(allowed.ok()) << allowed.json();
  EXPECT_FALSE(has_diagnostic(allowed, "PYOBJECT_PAYLOAD_ADVISORY"));
  ASSERT_TRUE(warned.ok()) << warned.json();
  EXPECT_TRUE(has_diagnostic(warned, "PYOBJECT_PAYLOAD_ADVISORY"));
  EXPECT_FALSE(rejected.ok());
  EXPECT_TRUE(has_diagnostic(rejected, "PYOBJECT_PAYLOAD_REJECTED"));
}

// REQ: PY-OP-006
TEST(ForeignOperatorCompileTest, RejectsOpaqueObjectEdgesAcrossAProcessBoundary) {
  // Key setup: Place the source and sink from one PyObject edge in separate launch groups.
  //
  //   source[process-a] --PyObjectPayload--> sink[process-b]
  Graph graph{"pyobject-cross-process"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<PyObjectPolicySource>("source");
  const auto sink = graph.op<PyObjectPolicySink>("sink");
  graph.add_flow(source->output, sink->input);
  constexpr std::array source_selector{detail::GraphObjectSelector{.graph_path = "source"}};
  constexpr std::array sink_selector{detail::GraphObjectSelector{.graph_path = "sink"}};
  detail::ProcessPlacementBinding placement;
  placement
      .bind_process_role("source-process",
                         detail::ProcessPlacement{
                             .members = source_selector,
                             .launch_group = "source-process",
                         })
      .bind_process_role("sink-process",
                         detail::ProcessPlacement{
                             .members = sink_selector,
                             .launch_group = "sink-process",
                         });

  const ExecutionPlan plan =
      detail::compile_with_process_placement(graph, CompileOptions{}, std::move(placement));

  // Key assertion: No wire codec exists, so physical lowering rejects the process boundary.
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(has_diagnostic(plan, "PROCESS_LOCAL_CHANNEL_PLAN_INVALID")) << plan.json();
}

// REQ: COMPILE-094
TEST(ForeignOperatorCompileTest, RejectsAnInvalidOpaqueObjectPolicy) {
  // Key setup: forge a value outside the closed public policy enumeration.
  CompileOptions options;
  // Deliberately forge an invalid closed-enum value to exercise compiler validation.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  options.opaque_object_payloads = static_cast<OpaqueObjectPayloadPolicy>(255U);
  Graph graph{"invalid-opaque-object-policy"};

  const ExecutionPlan plan = compile(graph, std::move(options));

  // Key assertion: invalid policy input fails even when the Graph has no opaque-object edge.
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(has_diagnostic(plan, "OPAQUE_OBJECT_PAYLOAD_POLICY_INVALID"));
}

// REQ: COMPILE-093
TEST(ForeignOperatorCompileTest, RejectsInvalidSerializationDomainsAtAuthoring) {
  const auto callbacks = stateless_callbacks();

  // Key assertion: empty-present and non-portable domain spellings carry the stable authoring
  // token.
  for (const std::string& domain : {std::string{}, std::string{"python/domain"}}) {
    try {
      static_cast<void>(
          ForeignOperatorCore(callbacks, ForeignOperatorOptions{.serialization_domain = domain}));
      FAIL() << "expected invalid serialization domain rejection";
    } catch (const RuntimeError& error) {
      EXPECT_EQ(error.code(), ErrorCode::kInvalidArgument);
      EXPECT_NE(
          std::string_view{error.what()}.find("FOREIGN_OPERATOR_SERIALIZATION_DOMAIN_INVALID"),
          std::string_view::npos);
    }
  }
}

// REQ: COMPILE-092
TEST(ForeignOperatorCompileTest, RejectsInvalidHostMetadataBeforeGraphPublication) {
  Graph graph{"invalid-host-metadata"};

  // Key assertion: an invalid host-kind value fails closed with the stable Graph token.
  try {
    static_cast<void>(graph.op<InvalidHostMetadataOperator>("invalid"));
    FAIL() << "expected invalid host metadata rejection";
  } catch (const RuntimeError& error) {
    EXPECT_EQ(error.code(), ErrorCode::kInvalidArgument);
    EXPECT_NE(std::string_view{error.what()}.find("GRAPH_OPERATOR_HOST_METADATA_INVALID"),
              std::string_view::npos);
  }
}

}  // namespace
}  // namespace holoscan
