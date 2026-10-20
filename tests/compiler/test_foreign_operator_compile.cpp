/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
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
#include "compiler/execution_plan_state.hpp"
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

// Adds `source` with an opaque-object output fanned out to `left` and `right`, used by the
// ForeignOperatorCompileTest opaque-object fanout tests.
void add_opaque_object_fanout(Graph& graph, const QueuePolicy right_policy) {
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<PyObjectPolicySource>("source");
  const auto left = graph.op<PyObjectPolicySink>("left");
  const auto right = graph.op<PyObjectPolicySink>("right");
  graph.add_flow(source->output, left->input);
  graph.add_flow(source->output, right->input, ConnectionOptions{.queue_policy = right_policy});
}

// Returns the candidate-side summary of every process-local channel rejection, sorted.
[[nodiscard]] std::vector<std::string> process_local_rejections(const ExecutionPlan& plan) {
  std::vector<std::string> provided;
  for (const auto& item : plan.diagnostics()) {
    if (item.token == "PROCESS_LOCAL_CHANNEL_PLAN_INVALID") {
      EXPECT_EQ(item.required, "direct-retained process-local delivery for a PyObjectPayload edge");
      provided.push_back(item.provided);
    }
  }
  std::ranges::sort(provided);
  return provided;
}

constexpr std::string_view kOpaqueObjectFamilyNeedsHoloMQ{
    "a fanout family that must use HoloMQ, such as one with a member in another process, a "
    "blocking member whose Operator can be restarted, or a member whose Operators another HoloMQ "
    "edge connects"};

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

// REQ: COMPILE-102
TEST(ForeignOperatorCompileTest, AppliesTheSerializationDomainSplitPolicyPerDomain) {
  //   [first]   [second]          or          [first second]
  //
  // Key setup: compile one authored domain split, and one unsplit domain, under each policy.
  const auto compile_with_policy = [](const bool split,
                                      const SerializationDomainSplitPolicy policy) {
    Graph graph{split ? "serialization-domain-split-policy"
                      : "serialization-domain-unsplit-policy"};
    const auto callbacks = stateless_callbacks();
    const auto first = graph.op<ForeignOperatorCore>(
        "first", callbacks, ForeignOperatorOptions{.serialization_domain = "python.gil"});
    const auto second = graph.op<ForeignOperatorCore>(
        "second", callbacks, ForeignOperatorOptions{.serialization_domain = "python.gil"});
    if (split) {
      graph.partition("first-partition").add(first);
      graph.partition("second-partition").add(second);
    }
    CompileOptions options;
    options.serialization_domain_splits = policy;
    return compile(graph, std::move(options));
  };
  const auto split_diagnostics = [](const ExecutionPlan& plan) {
    std::vector<std::pair<std::string, DiagnosticSeverity>> found;
    for (const auto& item : plan.diagnostics()) {
      if (item.token.starts_with("OPERATOR_SERIALIZATION_DOMAIN_SPLIT_")) {
        EXPECT_EQ(item.provided, "python.gil");
        found.emplace_back(item.token, item.severity);
      }
    }
    return found;
  };
  const ExecutionPlan warned = compile_with_policy(true, SerializationDomainSplitPolicy::kWarn);
  const ExecutionPlan allowed = compile_with_policy(true, SerializationDomainSplitPolicy::kAllow);
  const ExecutionPlan rejected = compile_with_policy(true, SerializationDomainSplitPolicy::kReject);
  const ExecutionPlan unsplit = compile_with_policy(false, SerializationDomainSplitPolicy::kReject);

  // Key assertion: warn is the default and reports the split once; allow keeps the same cover
  // without a diagnostic; reject fails admission once per split domain and never without a split.
  EXPECT_EQ(CompileOptions{}.serialization_domain_splits, SerializationDomainSplitPolicy::kWarn);
  ASSERT_TRUE(warned.ok()) << warned.json();
  EXPECT_EQ(split_diagnostics(warned),
            (std::vector<std::pair<std::string, DiagnosticSeverity>>{
                {"OPERATOR_SERIALIZATION_DOMAIN_SPLIT_ADVISORY", DiagnosticSeverity::kWarning}}));
  ASSERT_TRUE(allowed.ok()) << allowed.json();
  EXPECT_EQ(allowed.execution_partitions().size(), 2U);
  EXPECT_TRUE(split_diagnostics(allowed).empty());
  EXPECT_FALSE(rejected.ok());
  EXPECT_EQ(split_diagnostics(rejected),
            (std::vector<std::pair<std::string, DiagnosticSeverity>>{
                {"OPERATOR_SERIALIZATION_DOMAIN_SPLIT_REJECTED", DiagnosticSeverity::kError}}));
  ASSERT_TRUE(unsplit.ok()) << unsplit.json();
  EXPECT_EQ(unsplit.execution_partitions().size(), 1U);
  EXPECT_TRUE(split_diagnostics(unsplit).empty());
}

// REQ: COMPILE-102
TEST(ForeignOperatorCompileTest, RejectsAnInvalidSerializationDomainSplitPolicy) {
  // Key setup: forge a value outside the closed public policy enumeration.
  CompileOptions options;
  // Deliberately forge an invalid closed-enum value to exercise compiler validation.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  options.serialization_domain_splits = static_cast<SerializationDomainSplitPolicy>(255U);
  Graph graph{"invalid-serialization-domain-split-policy"};

  const ExecutionPlan plan = compile(graph, std::move(options));

  // Key assertion: invalid policy input fails even when the Graph has no serialization domain.
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(has_diagnostic(plan, "SERIALIZATION_DOMAIN_SPLIT_POLICY_INVALID"));
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
// REQ: PY-OP-013
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
  EXPECT_EQ(process_local_rejections(plan), std::vector<std::string>{"an edge to another process"});
}

// REQ: PY-OP-013
// REQ: CHAN-171
TEST(ForeignOperatorCompileTest, DeliversAnOpaqueObjectFanoutFamilyByDirectRetention) {
  //   source --PyObjectPayload--+--> left
  //                             +--> right [right_policy]
  for (const QueuePolicy right_policy : {QueuePolicy::kBlocking, QueuePolicy::kDropOldest}) {
    SCOPED_TRACE(right_policy == QueuePolicy::kBlocking ? "blocking" : "mixed policy");
    // Key setup: fan one opaque-object output out to two recipients.
    Graph graph{"pyobject-fanout"};
    add_opaque_object_fanout(graph, right_policy);

    const ExecutionPlan plan = compile(graph);

    // Key assertion: every service of the family is direct, one per queue policy, with a
    // subscriber per recipient.
    ASSERT_TRUE(plan.ok()) << plan.json();
    const auto* state = detail::ExecutionPlanAccess::state(plan);
    ASSERT_NE(state, nullptr);
    ASSERT_EQ(state->process_local_channel_services.size(),
              right_policy == QueuePolicy::kBlocking ? 1U : 2U);
    std::uint32_t subscribers = 0U;
    for (const auto& service : state->process_local_channel_services) {
      EXPECT_EQ(service.delivery_kind, detail::ProcessLocalChannelDeliveryKind::kDirectRetained);
      subscribers += service.max_subscribers;
    }
    EXPECT_EQ(subscribers, 2U);
  }
}

// REQ: PY-OP-013
// REQ: CHAN-171
TEST(ForeignOperatorCompileTest, NamesWhatKeepsAnOpaqueObjectEdgeOffDirectRetention) {
  // Key setup: place one fanout recipient in another launch group, and compile an all-local
  // family and a fanout-one edge under the private HoloMQ controls.
  //
  //   source[process-a] --PyObjectPayload--+--> left[process-a]
  //                                        +--> right[process-b]
  Graph split{"pyobject-fanout-cross-process"};
  add_opaque_object_fanout(split, QueuePolicy::kDropOldest);
  constexpr std::array local_selectors{detail::GraphObjectSelector{.graph_path = "source"},
                                       detail::GraphObjectSelector{.graph_path = "left"}};
  constexpr std::array remote_selector{detail::GraphObjectSelector{.graph_path = "right"}};
  detail::ProcessPlacementBinding placement;
  placement
      .bind_process_role("local-process",
                         detail::ProcessPlacement{
                             .members = local_selectors,
                             .launch_group = "local-process",
                         })
      .bind_process_role("remote-process",
                         detail::ProcessPlacement{
                             .members = remote_selector,
                             .launch_group = "remote-process",
                         });
  const ExecutionPlan cross_process =
      detail::compile_with_process_placement(split, CompileOptions{}, std::move(placement));
  Graph fanout{"pyobject-fanout-forced-holomq"};
  add_opaque_object_fanout(fanout, QueuePolicy::kDropOldest);
  const ExecutionPlan forced_fanout =
      detail::compile_with_process_local_holomq_fanout(fanout, CompileOptions{});
  Graph single{"pyobject-forced-holomq"};
  test::add_compile_test_default_clock(single);
  const auto source = single.op<PyObjectPolicySource>("source");
  const auto sink = single.op<PyObjectPolicySink>("sink");
  single.add_flow(source->output, sink->input);
  const ExecutionPlan forced_single = detail::compile_with_process_local_holomq(single, {});

  // Key assertion: each rejection names what kept its edge off direct retention. Every edge of a
  // family reports the same source port, so a family-wide obstacle appears once.
  EXPECT_FALSE(cross_process.ok());
  EXPECT_EQ(process_local_rejections(cross_process),
            (std::vector<std::string>{std::string{kOpaqueObjectFamilyNeedsHoloMQ},
                                      "an edge to another process"}))
      << cross_process.json();
  EXPECT_FALSE(forced_fanout.ok());
  EXPECT_EQ(process_local_rejections(forced_fanout),
            std::vector<std::string>{std::string{kOpaqueObjectFamilyNeedsHoloMQ}})
      << forced_fanout.json();
  EXPECT_FALSE(forced_single.ok());
  EXPECT_EQ(process_local_rejections(forced_single),
            std::vector<std::string>{"forced process-local HoloMQ delivery"})
      << forced_single.json();
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
