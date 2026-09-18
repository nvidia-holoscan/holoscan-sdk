/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/authoring/parameter_materializer.hpp>
#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/authoring/registry.hpp>
#if defined(HOLOSCAN_AUTHORING_TEST_PARAMS)
#include <holoscan/config/document.hpp>
#include <holoscan/params/config.hpp>
#include <holoscan/params/register_operator.hpp>
#endif
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "../common/allocation_counter.hpp"

namespace {

// Changes the prototype generation so lazy port lookup can detect reconstructed declarations.
// Representative test: AuthoringThunkLifetime.ResolvesFreshDeclarationAfterParameterReconstruction.
struct RebuildParams {
  std::uint32_t generation{1U};
};

HOLOSCAN_PARAM_SCHEMA(RebuildParams,
                      holoscan::param_schema("com.nvidia.holoscan.test.RebuildParams", 1),
                      holoscan::field<&RebuildParams::generation>("generation").range(1U, 16U));

// Declares the output whose address changes when RebuildParams reconstructs the prototype.
// Representative test: AuthoringThunkLifetime.ResolvesFreshDeclarationAfterParameterReconstruction.
class RebuildOperator final : public holoscan::Operator<RebuildParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { (void)spec.output(output); }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Output<std::uint32_t> output{"output"};
};

constexpr std::string_view kRebuildOperatorName{"holoscan.test.RebuildOperator"};

HOLOSCAN_REGISTER_OPERATOR(RebuildOperator, kRebuildOperatorName);

// Throws once so registered construction can prove owned causes and atomic retry behavior.
// Representative test: AuthoringThunkParameters.PreservesConstructorFailureContextBeforeInsertion.
class FailingConstructionOperator final : public holoscan::Operator<RebuildParams> {
 public:
  FailingConstructionOperator() {
    if (std::exchange(fail_next_construction, false)) {
      try {
        throw std::runtime_error{"registered constructor inner cause"};
      } catch (...) {
        std::throw_with_nested(holoscan::RuntimeError{
            holoscan::Error{holoscan::ErrorCode::kIncompatible, "registered constructor failed"}});
      }
    }
  }

  void setup(holoscan::OperatorSpec&) override {}

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  inline static bool fail_next_construction{false};
};

constexpr std::string_view kFailingConstructionOperatorName{
    "holoscan.test.FailingConstructionOperator"};

HOLOSCAN_REGISTER_OPERATOR(FailingConstructionOperator, kFailingConstructionOperatorName);

// Changes its port shape when configured to exercise atomic registered-operator insertion.
// Representative test: AuthoringThunkParameters.FailedInitialParametersRollBackInsertion.
struct ShapeChangingParams {
  bool extra_output{false};
};

HOLOSCAN_PARAM_SCHEMA(ShapeChangingParams,
                      holoscan::param_schema("com.nvidia.holoscan.test.ShapeChangingParams", 1),
                      holoscan::field<&ShapeChangingParams::extra_output>("extra_output"));

// Adds a parameter-dependent output that cannot replace the default registered declaration.
// Representative test: AuthoringThunkParameters.FailedInitialParametersRollBackInsertion.
class ShapeChangingOperator final : public holoscan::Operator<ShapeChangingParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    (void)spec.output(output);
    if (params().extra_output) {
      (void)spec.output(extra_output);
    }
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Output<std::uint32_t> output{"output"};
  holoscan::Output<std::uint32_t> extra_output{"extra_output"};
};

constexpr std::string_view kShapeChangingOperatorName{"holoscan.test.ShapeChangingOperator"};

HOLOSCAN_REGISTER_OPERATOR(ShapeChangingOperator, kShapeChangingOperatorName);

// Supplies a non-scalar field that exercises a generated typed configuration-binding thunk.
// Representative test: AuthoringConfigBinding.CapturesTypedRoutesForNonScalarFields.
struct ConfigRouteParams {
  std::int32_t threshold{1};
  std::vector<std::string> labels{"baseline"};
};

HOLOSCAN_PARAM_SCHEMA(ConfigRouteParams,
                      holoscan::param_schema("com.nvidia.holoscan.test.ConfigRouteParams", 1),
                      holoscan::field<&ConfigRouteParams::threshold>("threshold"),
                      holoscan::field<&ConfigRouteParams::labels>("labels"));

// Owns the parameter aggregate populated through a registry configuration-binding row.
// Representative test: AuthoringConfigBinding.CapturesTypedRoutesForNonScalarFields.
class ConfigRouteOperator final : public holoscan::Operator<ConfigRouteParams> {
 public:
  void setup(holoscan::OperatorSpec&) override {}

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }
};

constexpr std::string_view kConfigRouteOperatorName{"holoscan.test.ConfigRouteOperator"};

#if defined(HOLOSCAN_AUTHORING_TEST_PARAMS)
HOLOSCAN_REGISTER_CONFIGURABLE_OPERATOR(ConfigRouteOperator, kConfigRouteOperatorName);
#else
HOLOSCAN_REGISTER_OPERATOR(ConfigRouteOperator, kConfigRouteOperatorName);
#endif

// Sets the finite source configuration used by typed-versus-row plan parity checks.
// Representative test: AuthoringThunkParity.ProducesByteIdenticalPlanThroughRowsOnly.
struct PipelineSourceParams {
  std::uint32_t limit{4U};
};

HOLOSCAN_PARAM_SCHEMA(PipelineSourceParams,
                      holoscan::param_schema("com.nvidia.holoscan.test.PipelineSourceParams", 1),
                      holoscan::field<&PipelineSourceParams::limit>("limit").range(1U, 16U));

// Provides the clock-triggered head of the three-operator parity pipeline.
// Representative test: AuthoringThunkParity.PreservesMissingDefaultClockDiagnostic.
class PipelineSource final : public holoscan::Operator<PipelineSourceParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { (void)spec.output(output); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{10}});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Output<std::uint32_t> output{"output"};
};

// Controls the middle-stage configuration used by typed-versus-row plan parity checks.
// Representative test: AuthoringThunkParity.ProducesByteIdenticalPlanThroughRowsOnly.
struct PipelineTransformParams {
  std::uint32_t multiplier{2U};
};

HOLOSCAN_PARAM_SCHEMA(PipelineTransformParams,
                      holoscan::param_schema("com.nvidia.holoscan.test.PipelineTransformParams", 1),
                      holoscan::field<&PipelineTransformParams::multiplier>("multiplier")
                          .allowed_values({1U, 2U, 4U}));

// Provides the input-triggered middle stage of the parity pipeline.
// Representative test: AuthoringThunkParity.ProducesByteIdenticalPlanThroughRowsOnly.
class PipelineTransform final : public holoscan::Operator<PipelineTransformParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    (void)spec.input(input).queue_depth(4U);
    (void)spec.output(output);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Input<std::uint32_t> input{"input"};
  holoscan::Output<std::uint32_t> output{"output"};
};

// Terminates the parity pipeline with one input-triggered consumer.
// Representative test: AuthoringThunkParity.ProducesByteIdenticalPlanThroughRowsOnly.
class PipelineSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { (void)spec.input(input).queue_depth(4U); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Input<std::uint32_t> input{"input"};
};

// Supplies identical input/output ports for duplicate, fan-in, and cycle diagnostics.
// Representative test: AuthoringThunkParity.PreservesDuplicateCycleAndFaninErrors.
class PipelinePass final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    (void)spec.input(input);
    (void)spec.output(output);
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Input<std::uint32_t> input{"input"};
  holoscan::Output<std::uint32_t> output{"output"};
};

constexpr std::string_view kPipelineSourceName{"holoscan.test.PipelineSource"};
constexpr std::string_view kPipelineTransformName{"holoscan.test.PipelineTransform"};
constexpr std::string_view kPipelineSinkName{"holoscan.test.PipelineSink"};
constexpr std::string_view kPipelinePassName{"holoscan.test.PipelinePass"};

HOLOSCAN_REGISTER_OPERATOR(PipelineSource, kPipelineSourceName);
HOLOSCAN_REGISTER_OPERATOR(PipelineTransform, kPipelineTransformName);
HOLOSCAN_REGISTER_OPERATOR(PipelineSink, kPipelineSinkName);
HOLOSCAN_REGISTER_OPERATOR(PipelinePass, kPipelinePassName);

[[nodiscard]] const holoscan::authoring::RegisteredOperatorType& operator_row(
    std::string_view name) {
  const auto found = holoscan::authoring::authoring_registry().operator_type(name);
  if (!found) {
    throw holoscan::RuntimeError{found.error()};
  }
  return found->get();
}

[[nodiscard]] holoscan::authoring::MaterializedParameters materialize(
    const holoscan::authoring::RegisteredOperatorType& row,
    std::span<const holoscan::authoring::ParameterAssignment> assignments) {
  auto value =
      holoscan::authoring::materialize_parameters(*row.param_type, row.param_schema, assignments);
  if (!value) {
    throw holoscan::RuntimeError{value.error()};
  }
  return std::move(*value);
}

void add_registered_flow(holoscan::Graph& graph,
                         const holoscan::authoring::RegisteredOperatorType& source_row,
                         const holoscan::authoring::OpaqueOperatorRef& source,
                         std::string_view source_port,
                         const holoscan::authoring::RegisteredOperatorType& target_row,
                         const holoscan::authoring::OpaqueOperatorRef& target,
                         std::string_view target_port) {
  const auto source_description =
      holoscan::authoring::authoring_registry().describe_operator(source_row.name);
  const auto target_description =
      holoscan::authoring::authoring_registry().describe_operator(target_row.name);
  if (!source_description || !target_description) {
    throw holoscan::RuntimeError{!source_description ? source_description.error()
                                                     : target_description.error()};
  }
  const auto find_port = [](const auto& description, std::string_view name) {
    return std::find_if(description.ports.begin(),
                        description.ports.end(),
                        [name](const auto& port) { return port.name == name; });
  };
  const auto output = find_port(*source_description, source_port);
  const auto input = find_port(*target_description, target_port);
  if (output == source_description->ports.end() || input == target_description->ports.end()) {
    throw holoscan::RuntimeError{
        holoscan::Error{holoscan::ErrorCode::kNotFound, "AUTHORING_OPERATOR_PORT_UNKNOWN"}};
  }
  const auto payload =
      holoscan::authoring::authoring_registry().payload_type(output->payload_identity);
  if (!payload) {
    throw holoscan::RuntimeError{payload.error()};
  }
  payload->get().add_flow(graph,
                          source_row.port(source, source_port),
                          target_row.port(target, target_port),
                          holoscan::ConnectionOptions{.queue_depth = 4U});
}

void author_template_pipeline(holoscan::Graph& graph, bool add_clock) {
  auto source = graph.op<PipelineSource>("source");
  source.set_params(PipelineSourceParams{.limit = 8U});
  auto transform = graph.op<PipelineTransform>("transform");
  transform.set_params(PipelineTransformParams{.multiplier = 4U});
  const auto sink = graph.op<PipelineSink>("sink");
  (void)graph.add_flow(
      source->output, transform->input, holoscan::ConnectionOptions{.queue_depth = 4U});
  (void)graph.add_flow(
      transform->output, sink->input, holoscan::ConnectionOptions{.queue_depth = 4U});
  auto region = graph.region("pipeline");
  (void)region.add(source).add(transform).add(sink);
  auto partition = graph.partition("serial");
  (void)partition.add(source).add(transform).add(sink);
  graph.lifecycle().before(source, transform, holoscan::LifecycleStage::kStart);
  if (add_clock) {
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  }
}

void author_registered_pipeline(holoscan::Graph& graph, bool add_clock) {
  const auto& source_row = operator_row(kPipelineSourceName);
  const auto& transform_row = operator_row(kPipelineTransformName);
  const auto& sink_row = operator_row(kPipelineSinkName);
  const std::array source_assignments{holoscan::authoring::ParameterAssignment{
      "limit", holoscan::authoring::ParameterValue::from(std::uint64_t{8U})}};
  const std::array transform_assignments{holoscan::authoring::ParameterAssignment{
      "multiplier", holoscan::authoring::ParameterValue::from(std::uint64_t{4U})}};
  auto source_params = materialize(source_row, source_assignments);
  auto transform_params = materialize(transform_row, transform_assignments);
  const auto source = source_row.add_to_graph(graph, "source", &source_params);
  const auto transform = transform_row.add_to_graph(graph, "transform", &transform_params);
  const auto sink = sink_row.add_to_graph(graph, "sink", nullptr);
  add_registered_flow(graph, source_row, source, "output", transform_row, transform, "input");
  add_registered_flow(graph, transform_row, transform, "output", sink_row, sink, "input");
  auto region = graph.region("pipeline");
  source_row.region_add(region, source);
  transform_row.region_add(region, transform);
  sink_row.region_add(region, sink);
  auto partition = graph.partition("serial");
  source_row.partition_add(partition, source);
  transform_row.partition_add(partition, transform);
  sink_row.partition_add(partition, sink);
  auto lifecycle = graph.lifecycle();
  source_row.lifecycle_before(lifecycle, source, transform, holoscan::LifecycleStage::kStart);
  if (add_clock) {
    const auto clock =
        holoscan::authoring::authoring_registry().clock_type("holoscan.RealtimeClock");
    if (!clock) {
      throw holoscan::RuntimeError{clock.error()};
    }
    graph.set_default_clock(clock->get().add_to_graph(graph, "clock"));
  }
}

[[nodiscard]] std::pair<holoscan::ErrorCode, std::string> capture_error(const auto& operation) {
  try {
    operation();
  } catch (const holoscan::RuntimeError& error) {
    return {error.code(), error.what()};
  }
  return {holoscan::ErrorCode::kSuccess, {}};
}

[[nodiscard]] std::vector<std::string> diagnostic_tokens(const holoscan::ExecutionPlan& plan) {
  std::vector<std::string> result;
  for (const auto& diagnostic : plan.diagnostics()) {
    result.push_back(diagnostic.token);
  }
  return result;
}

#if defined(HOLOSCAN_AUTHORING_TEST_PARAMS)
[[nodiscard]] holoscan::config::Document vector_config_document() {
  auto label = holoscan::config::ConfigValue::try_string("configured");
  if (!label) {
    throw std::runtime_error("failed to construct test string value");
  }
  const std::array label_values{std::move(*label)};
  auto labels = holoscan::config::ConfigValue::try_array(label_values);
  if (!labels) {
    throw std::runtime_error("failed to construct test array value");
  }
  const std::array managed_entries{
      holoscan::config::ConfigObjectEntry{.key = "labels", .value = *labels}};
  auto managed = holoscan::config::ConfigValue::try_object(managed_entries);
  if (!managed) {
    throw std::runtime_error("failed to construct test parameter object");
  }
  const std::array root_entries{
      holoscan::config::ConfigObjectEntry{.key = "managed", .value = *managed}};
  auto root = holoscan::config::ConfigValue::try_object(root_entries);
  if (!root) {
    throw std::runtime_error("failed to construct test root object");
  }
  auto document = holoscan::config::Document::try_from_value(std::move(*root));
  if (!document) {
    throw std::runtime_error("failed to construct test document");
  }
  return std::move(*document);
}
#endif

}  // namespace

// REQ: AUTHOR-011
#if defined(HOLOSCAN_AUTHORING_TEST_PARAMS)
TEST(AuthoringConfigBinding, CapturesTypedRoutesForNonScalarFields) {
  // Graph:
  //
  // configured
  //
  // Key setup: Bind the registered Operator's vector field to a nested configuration object.
  const auto& row = operator_row(kConfigRouteOperatorName);
  ASSERT_NE(row.bind_config, nullptr);
  holoscan::Graph graph{"registry-config-binding"};
  (void)row.add_to_graph(graph, "configured", nullptr);
  auto path = holoscan::config::Path::try_parse("/managed");
  ASSERT_TRUE(path.has_value()) << path.error().message();
  auto binding = row.bind_config(
      graph, "configured", std::move(*path), holoscan::params::ParamBindingMode::kPatch);
  const holoscan::config::Document document = vector_config_document();

  auto attached = holoscan::params::set_params(graph, document, {std::move(binding)});

  // Key assertion: The generated thunk preserves the typed route through compilation.
  ASSERT_TRUE(attached.has_value()) << attached.error().token();
  ASSERT_EQ(attached->config_fields.size(), 1U);
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  EXPECT_TRUE(plan.ok()) << plan.json();
}
#endif

// REQ: AUTHOR-011
TEST(AuthoringConfigBinding, LeavesPlainRegistrationUnsupported) {
  // Key setup: Inspect a row registered through the Core-only macro.
  const auto& row = operator_row(kRebuildOperatorName);

  // Key assertion: Config support is explicit and independent of target link configuration.
  EXPECT_EQ(row.bind_config, nullptr);
}

// REQ: AUTHOR-001
TEST(AuthoringThunkParameters, RejectsMismatchedMaterializationBeforeTypedDowncast) {
  // Key setup: Materialize source parameters, then pass them to the transform row.
  const auto& source_row = operator_row(kPipelineSourceName);
  const auto& transform_row = operator_row(kPipelineTransformName);
  const std::array assignments{holoscan::authoring::ParameterAssignment{
      "limit", holoscan::authoring::ParameterValue::from(std::uint32_t{8U})}};
  auto source_params = materialize(source_row, assignments);
  holoscan::Graph graph{"parameter-mismatch"};

  const auto mismatch =
      capture_error([&] { (void)transform_row.add_to_graph(graph, "transform", &source_params); });

  // Key assertion: Type identity is checked before a wrong aggregate is read or the Graph mutates.
  EXPECT_EQ(mismatch.first, holoscan::ErrorCode::kInvalidArgument);
  EXPECT_NE(mismatch.second.find("AUTHORING_PARAMETER_TYPE_MISMATCH"), std::string::npos);
  EXPECT_TRUE(graph.view().vertices().empty());

  const auto transform = transform_row.add_to_graph(graph, "transform", nullptr);
  const auto replacement_mismatch =
      capture_error([&] { transform_row.set_params(transform, source_params); });

  // Key assertion: Replacement applies the same check before reconstructing the prototype.
  EXPECT_EQ(replacement_mismatch.first, holoscan::ErrorCode::kInvalidArgument);
  EXPECT_NE(replacement_mismatch.second.find("AUTHORING_PARAMETER_TYPE_MISMATCH"),
            std::string::npos);
}

// REQ: GRAPH-039
TEST(AuthoringThunkParameters, PreservesConstructorFailureContextBeforeInsertion) {
  // Key setup: Exercise both registered construction routes with an owning nested cause.
  const auto& row = operator_row(kFailingConstructionOperatorName);
  const std::array assignments{holoscan::authoring::ParameterAssignment{
      "generation", holoscan::authoring::ParameterValue::from(std::uint32_t{2U})}};
  auto params = materialize(row, assignments);
  for (const bool with_initial_params : {false, true}) {
    SCOPED_TRACE(with_initial_params);
    holoscan::Graph graph{"atomic-constructor-failure"};
    const auto before = graph.view();
    const auto* initial_params = with_initial_params ? &params : nullptr;
    FailingConstructionOperator::fail_next_construction = true;

    const auto failed =
        capture_error([&] { (void)row.add_to_graph(graph, "unstable", initial_params); });

    // Key assertion: Copied evidence survives unwinding and the rejected Graph stays unchanged.
    EXPECT_EQ(failed.first, holoscan::ErrorCode::kIncompatible);
    EXPECT_NE(failed.second.find("GRAPH_OPERATOR_CONSTRUCTION_FAILED"), std::string::npos);
    EXPECT_NE(failed.second.find("registered constructor failed"), std::string::npos);
    EXPECT_NE(failed.second.find("registered constructor inner cause"), std::string::npos);
    EXPECT_NE(failed.second.find("operator=unstable"), std::string::npos);
    EXPECT_EQ(graph.view().identity(), before.identity());
    EXPECT_TRUE(graph.view().vertices().empty());

    // Key assertion: Retrying the same name succeeds without consuming an ID on failure.
    (void)row.add_to_graph(graph, "unstable", initial_params);
    const auto after = graph.view();
    ASSERT_EQ(after.vertices().size(), 1U);
    EXPECT_EQ(after.vertices().front().id.value, 1U);
  }
}

// REQ: AUTHOR-001
TEST(AuthoringThunkParameters, FailedInitialParametersRollBackInsertion) {
  // Key setup: Initial parameters would change the registered Operator's default port shape.
  const auto& row = operator_row(kShapeChangingOperatorName);
  const std::array assignments{holoscan::authoring::ParameterAssignment{
      "extra_output", holoscan::authoring::ParameterValue::from(true)}};
  auto params = materialize(row, assignments);
  holoscan::Graph graph{"atomic-initial-params"};
  const auto before = graph.view();

  const auto failed = capture_error([&] { (void)row.add_to_graph(graph, "unstable", &params); });

  // Key assertion: The failed compound mutation leaves no Operator, name, generation, or used ID.
  EXPECT_EQ(failed.first, holoscan::ErrorCode::kIncompatible);
  EXPECT_NE(failed.second.find("GRAPH_PARAMETER_PORT_SHAPE_CHANGED"), std::string::npos);
  EXPECT_EQ(graph.view().identity(), before.identity());
  EXPECT_TRUE(graph.view().vertices().empty());

  (void)row.add_to_graph(graph, "unstable", nullptr);
  const auto after = graph.view();
  ASSERT_EQ(after.vertices().size(), 1U);
  EXPECT_EQ(after.vertices().front().id.value, 1U);
}

// REQ: AUTHOR-001
TEST(AuthoringThunkLifetime, DoesNotAllocateOpaqueStorageAfterOperatorInsertion) {
  // Graph:
  //
  // candidate
  //
  // Key setup: Count allocations in the typed insertion that the registered thunk delegates to.
  const auto& row = operator_row(kRebuildOperatorName);
  std::size_t typed_allocations{};
  {
    holoscan::Graph graph{"measure-opaque-operator"};
    holoscan::test::ScopedAllocationCounter counter;
    (void)graph.op<RebuildOperator>("candidate");
    typed_allocations = counter.count();
  }

  // Key assertion: The row adds no fallible opaque-holder allocation after typed insertion.
  holoscan::Graph graph{"verify-opaque-operator"};
  bool completed = false;
  {
    holoscan::test::ScopedAllocationFailure failure{typed_allocations};
    try {
      (void)row.add_to_graph(graph, "candidate", nullptr);
      completed = true;
    } catch (...) {
      // The assertions below report both a remaining allocation and any committed mutation.
    }
  }
  EXPECT_TRUE(completed);
  const auto committed = graph.view();
  ASSERT_EQ(committed.vertices().size(), 1U);
  EXPECT_EQ(committed.vertices().front().name, "candidate");

  holoscan::Graph retry_graph{"retry-opaque-operator"};
  {
    holoscan::test::ScopedAllocationFailure failure{0U};
    EXPECT_THROW((void)row.add_to_graph(retry_graph, "candidate", nullptr), std::bad_alloc);
  }
  EXPECT_TRUE(retry_graph.view().vertices().empty());
  EXPECT_NO_THROW((void)row.add_to_graph(retry_graph, "candidate", nullptr));
  EXPECT_EQ(retry_graph.view().vertices().size(), 1U);
}

// REQ: AUTHOR-002
TEST(AuthoringThunkLifetime, ResolvesFreshDeclarationAfterParameterReconstruction) {
  // Graph:
  //
  // rebuild --> sink
  //
  // Key setup: Retain one logical port reference across parameter reconstruction.
  const auto found = holoscan::authoring::authoring_registry().operator_type(kRebuildOperatorName);
  ASSERT_TRUE(found) << found.error().message;
  const auto& row = found->get();

  holoscan::Graph graph{"staleness"};
  const auto op = row.add_to_graph(graph, "rebuild", nullptr);
  const auto output = row.port(op, "output");
  ASSERT_TRUE(output);

  const std::array assignments{holoscan::authoring::ParameterAssignment{
      "generation", holoscan::authoring::ParameterValue::from(std::uint32_t{2U})}};
  auto updated = materialize(row, assignments);
  row.set_params(op, updated);

  const auto& sink_row = operator_row(kPipelineSinkName);
  const auto sink = sink_row.add_to_graph(graph, "sink", nullptr);
  const auto payload = holoscan::authoring::authoring_registry().payload_type("holoscan.uint32");
  ASSERT_TRUE(payload);
  payload->get().add_flow(graph, output, sink_row.port(sink, "input"), {});

  // Key assertion: The retained reference resolves the reconstructed declaration at thunk entry.
  const holoscan::GraphView view = graph.view();
  ASSERT_EQ(view.edges().size(), 1U);
  ASSERT_EQ(view.ports().size(), 2U);
  EXPECT_EQ(view.ports().front().name, "output");
}

// REQ: AUTHOR-002
TEST(AuthoringThunkLifetime, DoesNotExtendTheOwningGraphLifetime) {
  // Graph:
  //
  // rebuild
  //
  // Key setup: Retain only an opaque port reference after destroying its owning Graph.
  const auto& row = operator_row(kRebuildOperatorName);
  std::optional<holoscan::authoring::OpaquePortRef> output;
  {
    holoscan::Graph graph{"lifetime"};
    const auto op = row.add_to_graph(graph, "rebuild", nullptr);
    output = row.port(op, "output");
    ASSERT_TRUE(*output);
  }

  // Key assertion: The weak Graph identity expires and later port lookup fails deterministically.
  EXPECT_FALSE(static_cast<bool>(*output));
  holoscan::Graph graph{"consumer"};
  const auto& sink_row = operator_row(kPipelineSinkName);
  const auto sink = sink_row.add_to_graph(graph, "sink", nullptr);
  const auto payload = holoscan::authoring::authoring_registry().payload_type("holoscan.uint32");
  ASSERT_TRUE(payload);
  const auto error = capture_error(
      [&] { payload->get().add_flow(graph, *output, sink_row.port(sink, "input"), {}); });
  EXPECT_EQ(error.first, holoscan::ErrorCode::kInvalidArgument);
  EXPECT_NE(error.second.find("GRAPH_HANDLE_INVALID"), std::string::npos);
}

// REQ: AUTHOR-001
TEST(AuthoringThunkParity, ProducesByteIdenticalPlanThroughRowsOnly) {
  // Graph:
  //
  // source --> transform --> sink
  //
  // Key setup: Compose equivalent graphs through typed templates and registry-only thunks.
  holoscan::Graph typed_graph{"first-pipeline"};
  author_template_pipeline(typed_graph, true);
  holoscan::Graph row_graph{"first-pipeline"};
  author_registered_pipeline(row_graph, true);

  const holoscan::ExecutionPlan typed_plan = holoscan::compile(typed_graph);
  const holoscan::ExecutionPlan row_plan = holoscan::compile(row_graph);

  // Key assertion: Both authoring routes produce the same admitted plan bytes.
  ASSERT_TRUE(typed_plan.ok()) << typed_plan.json();
  ASSERT_TRUE(row_plan.ok()) << row_plan.json();
  EXPECT_EQ(row_plan.json(), typed_plan.json());
}

// REQ: AUTHOR-001
TEST(AuthoringThunkParity, PreservesDuplicateCycleAndFaninErrors) {
  // Graph scenarios:
  //
  // duplicate: same (declared twice)
  // fan-in:    a --> b <-- c
  // cycle:     a --> b --> c --> a
  //
  // Key setup: Trigger each structural error once through templates and once through rows.
  const auto typed_duplicate = capture_error([] {
    holoscan::Graph graph{"duplicate"};
    (void)graph.op<PipelinePass>("same");
    (void)graph.op<PipelinePass>("same");
  });
  const auto row_duplicate = capture_error([] {
    holoscan::Graph graph{"duplicate"};
    const auto& row = operator_row(kPipelinePassName);
    (void)row.add_to_graph(graph, "same", nullptr);
    (void)row.add_to_graph(graph, "same", nullptr);
  });

  // Key assertion: Registry-only authoring preserves every typed-path error code and token.
  EXPECT_EQ(row_duplicate.first, typed_duplicate.first);
  EXPECT_NE(typed_duplicate.second.find("GRAPH_OPERATOR_NAME_DUPLICATE"), std::string::npos);
  EXPECT_NE(row_duplicate.second.find("GRAPH_OPERATOR_NAME_DUPLICATE"), std::string::npos);

  const auto typed_fanin = capture_error([] {
    holoscan::Graph graph{"fanin"};
    const auto a = graph.op<PipelinePass>("a");
    const auto b = graph.op<PipelinePass>("b");
    const auto c = graph.op<PipelinePass>("c");
    (void)graph.add_flow(a->output, b->input);
    (void)graph.add_flow(c->output, b->input);
  });
  const auto row_fanin = capture_error([] {
    holoscan::Graph graph{"fanin"};
    const auto& row = operator_row(kPipelinePassName);
    const auto a = row.add_to_graph(graph, "a", nullptr);
    const auto b = row.add_to_graph(graph, "b", nullptr);
    const auto c = row.add_to_graph(graph, "c", nullptr);
    add_registered_flow(graph, row, a, "output", row, b, "input");
    add_registered_flow(graph, row, c, "output", row, b, "input");
  });
  EXPECT_EQ(row_fanin.first, typed_fanin.first);
  EXPECT_NE(typed_fanin.second.find("GRAPH_INPUT_FANIN_UNSUPPORTED"), std::string::npos);
  EXPECT_NE(row_fanin.second.find("GRAPH_INPUT_FANIN_UNSUPPORTED"), std::string::npos);

  const auto typed_cycle = capture_error([] {
    holoscan::Graph graph{"cycle"};
    const auto a = graph.op<PipelinePass>("a");
    const auto b = graph.op<PipelinePass>("b");
    const auto c = graph.op<PipelinePass>("c");
    (void)graph.add_flow(a->output, b->input);
    (void)graph.add_flow(b->output, c->input);
    (void)graph.add_flow(c->output, a->input);
  });
  const auto row_cycle = capture_error([] {
    holoscan::Graph graph{"cycle"};
    const auto& row = operator_row(kPipelinePassName);
    const auto a = row.add_to_graph(graph, "a", nullptr);
    const auto b = row.add_to_graph(graph, "b", nullptr);
    const auto c = row.add_to_graph(graph, "c", nullptr);
    add_registered_flow(graph, row, a, "output", row, b, "input");
    add_registered_flow(graph, row, b, "output", row, c, "input");
    add_registered_flow(graph, row, c, "output", row, a, "input");
  });
  EXPECT_EQ(row_cycle.first, typed_cycle.first);
  EXPECT_NE(typed_cycle.second.find("GRAPH_DIRECT_CYCLE"), std::string::npos);
  EXPECT_NE(row_cycle.second.find("GRAPH_DIRECT_CYCLE"), std::string::npos);
}

// REQ: AUTHOR-001
TEST(AuthoringThunkParity, ReportsPayloadMismatchBeforeTypedThunkEntry) {
  // Attempted Graph:
  //
  // source(uint32) --X--> sink(uint32) through the int32 payload row
  //
  // Key setup: Pair valid uint32 declarations with the wrong registered payload thunk.
  holoscan::Graph graph{"mismatch"};
  const auto& source_row = operator_row(kPipelineSourceName);
  const auto& sink_row = operator_row(kPipelineSinkName);
  const auto source = source_row.add_to_graph(graph, "source", nullptr);
  const auto sink = sink_row.add_to_graph(graph, "sink", nullptr);
  const auto payload = holoscan::authoring::authoring_registry().payload_type("holoscan.int32");
  ASSERT_TRUE(payload);
  const auto mismatch = capture_error([&] {
    payload->get().add_flow(
        graph, source_row.port(source, "output"), sink_row.port(sink, "input"), {});
  });

  // Key assertion: The identity mismatch fails before the thunk downcasts either declaration.
  EXPECT_EQ(mismatch.first, holoscan::ErrorCode::kIncompatible);
  EXPECT_NE(mismatch.second.find("GRAPH_PORT_TYPE_MISMATCH"), std::string::npos);
}

// REQ: AUTHOR-001
TEST(AuthoringThunkParity, PreservesMissingDefaultClockDiagnostic) {
  // Graph:
  //
  // source --> transform --> sink    (default clock omitted)
  //
  // Key setup: Compile equivalent clock-triggered graphs without declaring their default clock.
  holoscan::Graph typed_graph{"missing-clock"};
  author_template_pipeline(typed_graph, false);
  holoscan::Graph row_graph{"missing-clock"};
  author_registered_pipeline(row_graph, false);
  const holoscan::ExecutionPlan typed_plan = holoscan::compile(typed_graph);
  const holoscan::ExecutionPlan row_plan = holoscan::compile(row_graph);
  ASSERT_FALSE(typed_plan.ok());
  ASSERT_FALSE(row_plan.ok());
  const auto typed_tokens = diagnostic_tokens(typed_plan);
  const auto row_tokens = diagnostic_tokens(row_plan);

  // Key assertion: Registry authoring preserves the typed compiler's complete diagnostic set.
  EXPECT_EQ(row_tokens, typed_tokens);
  EXPECT_NE(std::find(typed_tokens.begin(), typed_tokens.end(), "TEMPORAL_CLOCK_UNDECLARED"),
            typed_tokens.end());
}
