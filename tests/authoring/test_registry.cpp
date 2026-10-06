/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/authoring/register_graph.hpp>
#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/authoring/register_payload.hpp>
#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/time/manual_clock.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <optional>
#include <semaphore>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "../../src/core/graph_state.hpp"
#include "../common/allocation_counter.hpp"

namespace {

// Supplies bounded parameter metadata for RegistryOperator.
// Representative test: AuthoringRegistry.HarvestMatchesGraphViewGroundTruth.
struct RegistryParams {
  std::uint32_t count{3U};
};

HOLOSCAN_PARAM_SCHEMA(RegistryParams,
                      holoscan::param_schema("com.nvidia.holoscan.test.RegistryParams", 1),
                      holoscan::field<&RegistryParams::count>("count").range(1U, 8U));

// Exposes one typed input, output, and schema for registry metadata and thunk tests.
// Representative test: AuthoringRegistry.EnumeratesAndLooksUpRegisteredOperators.
class RegistryOperator final : public holoscan::Operator<RegistryParams> {
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

// Supplies one required field for automatic registration metadata coverage.
// Representative test: AuthoringRegistry.AutomaticRowsExposeRequiredOnLoadParameters.
struct RequiredRegistryParams {
  std::uint32_t required_count{1U};
};

HOLOSCAN_PARAM_SCHEMA(RequiredRegistryParams,
                      holoscan::param_schema("holoscan.tests.RequiredRegistryParams", 1),
                      holoscan::field<&RequiredRegistryParams::required_count>("required_count")
                          .required_on_load()
                          .range(1U, 8U));

// Provides a default-constructible Operator whose schema has one required language-facing field.
// Representative test: AuthoringRegistry.AutomaticRowsExposeRequiredOnLoadParameters.
class RequiredRegistryOperator final : public holoscan::Operator<RequiredRegistryParams> {
 public:
  void setup(holoscan::OperatorSpec&) override {}

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }
};

struct RegistryGraphParams {
  std::uint32_t count{3U};
};

HOLOSCAN_PARAM_SCHEMA(RegistryGraphParams,
                      holoscan::param_schema("holoscan.tests.RegistryGraphParams", 1),
                      holoscan::field<&RegistryGraphParams::count>("count").range(1U, 8U));

// Re-export one RegistryOperator for graph-row discovery and typed-handle tests.
// Representative test: AuthoringGraphRegistry.RowPreservesTypedHandleOperationsAndWeakOwnership.
class RegistryGraph final : public holoscan::Graph {
 public:
  using Params = RegistryGraphParams;

  explicit RegistryGraph(const Params& params) noexcept : params_(params) {}

  holoscan::Input<std::uint32_t> input{"input"};
  holoscan::Output<std::uint32_t> output{"output"};

 protected:
  void compose() override {
    auto member = op<RegistryOperator>("operator");
    member.set_params(RegistryParams{.count = params_.count});
    export_input(input, member->input);
    export_output(output, member->output);
  }

 private:
  Params params_;
};

std::atomic<std::size_t> lazy_graph_defaults_constructed{};

std::uint32_t tracked_lazy_graph_default() noexcept {
  lazy_graph_defaults_constructed.fetch_add(1U, std::memory_order_relaxed);
  return 3U;
}

struct LazyDefaultGraphParams {
  std::uint32_t count{tracked_lazy_graph_default()};
};

HOLOSCAN_PARAM_SCHEMA(LazyDefaultGraphParams,
                      holoscan::param_schema("holoscan.tests.LazyDefaultGraphParams", 1),
                      holoscan::field<&LazyDefaultGraphParams::count>("count").range(1U, 8U));

class LazyDefaultRegistryGraph final : public holoscan::Graph {
 public:
  using Params = LazyDefaultGraphParams;

  explicit LazyDefaultRegistryGraph(const Params& params) noexcept : params_(params) {}

 protected:
  void compose() override {}

 private:
  Params params_;
};

class RegistryTensorOperator final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { (void)spec.output(output); }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Output<holoscan::Tensor> output{"output"};
};

// Attempt a self-embedding graph so registry thunks can prove recursion remains rejected.
// Representative test: AuthoringGraphRegistry.RejectsDuplicateUnknownAndRecursiveGraphRows.
class RegistryRecursiveGraph final : public holoscan::Graph {
 protected:
  void compose() override { (void)embed<RegistryRecursiveGraph>("recursive"); }
};

// Models a reusable Graph whose positional constructor cannot use automatic registration.
// Representative test: AuthoringGraphRegistry.SupportsHandAuthoredPassThroughRows.
class ManualRegistryGraph final : public holoscan::Graph {
 public:
  explicit ManualRegistryGraph(std::uint32_t count) noexcept : count_(count) {}

  holoscan::Input<std::uint32_t> input{"input"};
  holoscan::Output<std::uint32_t> output{"output"};

 protected:
  void compose() override {
    auto member = op<RegistryOperator>("operator");
    member.set_params(RegistryParams{.count = count_});
    export_input(input, member->input);
    export_output(output, member->output);
  }

 private:
  std::uint32_t count_{};
};

holoscan::authoring::OpaqueGraphRef embed_manual_registry_graph(
    holoscan::Graph& graph, std::string_view name,
    const holoscan::authoring::MaterializedParameters* params) {
  const RegistryGraphParams defaults{};
  if (params != nullptr &&
      (params->type() == nullptr ||
       params->type()->local_type_token !=
           holoscan::param_type_descriptor<RegistryGraphParams>().local_type_token)) {
    throw holoscan::RuntimeError{holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                                                 "AUTHORING_PARAMETER_TYPE_MISMATCH"}};
  }
  const auto* values =
      params == nullptr ? &defaults : static_cast<const RegistryGraphParams*>(params->data());
  return holoscan::authoring::OpaqueGraphRef::from_handle(
      graph.embed<ManualRegistryGraph>(name, values->count));
}

holoscan::expected<holoscan::authoring::OpaqueGraphRef, holoscan::Error>
embed_manual_registry_graph_from_assignments(
    holoscan::Graph& graph, std::string_view name,
    std::span<const holoscan::authoring::ParameterAssignment> assignments) {
  std::uint32_t count{RegistryGraphParams{}.count};
  if (!assignments.empty() &&
      (assignments.size() != 1U || assignments.front().name != "count" ||
       assignments.front().value.kind != holoscan::authoring::ParameterValueKind::kScalar ||
       !assignments.front().value.scalar.try_as(count) || count == 0U)) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                                                     "AUTHORING_MANUAL_GRAPH_ASSIGNMENT_INVALID"});
  }
  try {
    return holoscan::authoring::OpaqueGraphRef::from_handle(
        graph.embed<ManualRegistryGraph>(name, count));
  } catch (const holoscan::RuntimeError& error) {
    return holoscan::make_unexpected(holoscan::make_error(error));
  }
}

std::vector<holoscan::PortRecord> manual_registry_graph_ports(
    const holoscan::authoring::OpaqueGraphRef& graph) {
  return graph.interface_port_records();
}

bool fail_manual_registry_graph_ports{};
holoscan::RegionHandle* manual_registry_graph_region{};
const holoscan::Input<std::uint32_t>* manual_registry_parent_input{};
bool mutate_manual_registry_parent_clock{};
const holoscan::Output<holoscan::Tensor>* manual_registry_parent_tensor_output{};
holoscan::authoring::OpaqueGraphRef retained_failed_graph;
holoscan::authoring::OpaquePortRef retained_failed_graph_port;
holoscan::authoring::OpaquePortConfigurator retained_failed_input_configurator;
holoscan::authoring::OpaquePortConfigurator retained_failed_output_configurator;
std::optional<holoscan::InputPortConfigurator<std::uint32_t>> retained_typed_input_configurator;
std::optional<holoscan::OutputPortConfigurator<std::uint32_t>> retained_typed_output_configurator;
std::optional<holoscan::OperatorHandle<RegistryOperator>> retained_operator_handle;
std::optional<holoscan::FlowHandle<std::uint32_t>> retained_flow_handle;
std::optional<holoscan::FlowConfigurator<std::uint32_t>> retained_flow_configurator;
holoscan::TensorOutputPortConfigurator* retained_parent_tensor_configurator{};
holoscan::TensorOutputPortConfigurator* retained_failed_tensor_configurator{};
std::optional<holoscan::RegionHandle> retained_region_handle;
std::optional<holoscan::PartitionHandle> retained_partition_handle;
std::optional<holoscan::ImportedTopic<std::uint32_t>> retained_imported_topic;
std::optional<holoscan::ExportedTopic<std::uint32_t>> retained_exported_topic;

holoscan::expected<holoscan::authoring::OpaqueGraphRef, holoscan::Error>
embed_manual_registry_graph_in_existing_region(
    holoscan::Graph& graph, std::string_view name,
    std::span<const holoscan::authoring::ParameterAssignment> assignments) {
  auto embedded = embed_manual_registry_graph_from_assignments(graph, name, assignments);
  if (embedded && manual_registry_graph_region != nullptr) {
    embedded->region_add(*manual_registry_graph_region);
  }
  return embedded;
}

holoscan::expected<holoscan::authoring::OpaqueGraphRef, holoscan::Error>
embed_manual_registry_graph_after_mutating_parent(
    holoscan::Graph& graph, std::string_view name,
    std::span<const holoscan::authoring::ParameterAssignment> assignments) {
  auto embedded = embed_manual_registry_graph_from_assignments(graph, name, assignments);
  if (embedded && manual_registry_parent_input != nullptr) {
    graph.configure(*manual_registry_parent_input).queue_depth(17U);
  }
  if (embedded && mutate_manual_registry_parent_clock) {
    const auto temporary = graph.add_clock<holoscan::ManualClock>("temporary", std::int64_t{0});
    graph.set_default_clock(temporary);
  }
  return embedded;
}

holoscan::expected<holoscan::authoring::OpaqueGraphRef, holoscan::Error>
embed_manual_registry_graph_after_replacing_parent_params(
    holoscan::Graph& graph, std::string_view name,
    std::span<const holoscan::authoring::ParameterAssignment> assignments) {
  auto embedded = embed_manual_registry_graph_from_assignments(graph, name, assignments);
  graph.handle<RegistryOperator>("parent").set_params(RegistryParams{.count = 7U});
  return embedded;
}

holoscan::expected<holoscan::authoring::OpaqueGraphRef, holoscan::Error>
embed_manual_registry_graph_after_configuring_parent_tensor(
    holoscan::Graph& graph, std::string_view name,
    std::span<const holoscan::authoring::ParameterAssignment> assignments) {
  auto embedded = embed_manual_registry_graph_from_assignments(graph, name, assignments);
  if (embedded && manual_registry_parent_tensor_output != nullptr) {
    retained_parent_tensor_configurator = &graph.configure(*manual_registry_parent_tensor_output);
    retained_parent_tensor_configurator->diagnostic_alias("temporary-tensor");
  }
  return embedded;
}

holoscan::expected<holoscan::authoring::OpaqueGraphRef, holoscan::Error>
embed_manual_registry_graph_retaining_typed_capabilities(
    holoscan::Graph& graph, std::string_view name,
    std::span<const holoscan::authoring::ParameterAssignment>) {
  try {
    auto embedded = graph.embed<ManualRegistryGraph>(name, 3U);
    retained_typed_input_configurator.emplace(graph.configure(embedded->input));
    retained_typed_output_configurator.emplace(graph.configure(embedded->output));

    auto source = graph.op<RegistryOperator>("retired-source");
    auto target = graph.op<RegistryOperator>("retired-target");
    retained_operator_handle.emplace(source);
    auto flow = graph.add_flow(source->output, target->input);
    retained_flow_handle.emplace(flow);
    retained_flow_configurator.emplace(graph.configure(flow));

    auto tensor = graph.op<RegistryTensorOperator>("retired-tensor");
    retained_failed_tensor_configurator = &graph.configure(tensor->output);
    return holoscan::authoring::OpaqueGraphRef::from_handle(embedded);
  } catch (const holoscan::RuntimeError& error) {
    return holoscan::make_unexpected(holoscan::make_error(error));
  }
}

holoscan::expected<holoscan::authoring::OpaqueGraphRef, holoscan::Error>
embed_manual_registry_graph_retaining_placement_handles(
    holoscan::Graph& graph, std::string_view name,
    std::span<const holoscan::authoring::ParameterAssignment> assignments) {
  auto embedded = embed_manual_registry_graph_from_assignments(graph, name, assignments);
  retained_region_handle.emplace(graph.region("discarded-region"));
  retained_partition_handle.emplace(graph.partition("discarded-partition"));
  return embedded;
}

holoscan::expected<holoscan::authoring::OpaqueGraphRef, holoscan::Error>
embed_manual_registry_graph_retaining_topic_handles(
    holoscan::Graph& graph, std::string_view name,
    std::span<const holoscan::authoring::ParameterAssignment> assignments) {
  retained_imported_topic.emplace(graph.import_topic<std::uint32_t>("holomq://discarded-import"));
  retained_exported_topic.emplace(graph.export_topic<std::uint32_t>("holomq://discarded-export"));
  return embed_manual_registry_graph_from_assignments(graph, name, assignments);
}

std::binary_semaphore registered_embed_paused{0};
std::binary_semaphore release_registered_embed{0};

holoscan::expected<holoscan::authoring::OpaqueGraphRef, holoscan::Error>
embed_manual_registry_graph_with_pause(
    holoscan::Graph& graph, std::string_view name,
    std::span<const holoscan::authoring::ParameterAssignment> assignments) {
  auto embedded = embed_manual_registry_graph_from_assignments(graph, name, assignments);
  registered_embed_paused.release();
  release_registered_embed.acquire();
  return embedded;
}

holoscan::expected<holoscan::authoring::OpaqueGraphRef, holoscan::Error>
embed_manual_registry_graph_retaining_configurators(
    holoscan::Graph& graph, std::string_view name,
    std::span<const holoscan::authoring::ParameterAssignment> assignments) {
  auto embedded = embed_manual_registry_graph_from_assignments(graph, name, assignments);
  if (embedded) {
    const auto payload =
        holoscan::authoring::detail::make_registered_payload<std::uint32_t>("holoscan.uint32");
    retained_failed_input_configurator =
        payload.configure_input(graph, embedded->interface_port("input"));
    retained_failed_output_configurator =
        payload.configure_output(graph, embedded->interface_port("output"));
  }
  return embedded;
}

template <typename Operation>
void expect_graph_handle_invalid(Operation&& operation) {
  try {
    std::forward<Operation>(operation)();
    FAIL() << "retired configurator accepted a mutation";
  } catch (const holoscan::RuntimeError& error) {
    EXPECT_EQ(holoscan::make_error(error).message, "GRAPH_HANDLE_INVALID");
  }
}

template <typename Operation>
void expect_retired_capability(Operation&& operation) {
  try {
    std::forward<Operation>(operation)();
    FAIL() << "retired authoring capability accepted an operation";
  } catch (const holoscan::RuntimeError& error) {
    EXPECT_EQ(error.code(), holoscan::ErrorCode::kInvalidArgument);
  }
}

void expect_retained_configurators_invalid() {
  expect_graph_handle_invalid([] { retained_failed_input_configurator.queue_depth(17U); });
  expect_graph_handle_invalid(
      [] { retained_failed_input_configurator.diagnostic_alias("stale-input"); });
  expect_graph_handle_invalid(
      [] { retained_failed_output_configurator.max_emits_per_compute(17U); });
  expect_graph_handle_invalid(
      [] { retained_failed_output_configurator.late_joiner_history_depth(17U); });
  expect_graph_handle_invalid(
      [] { retained_failed_output_configurator.diagnostic_alias("stale-output"); });
}

std::vector<holoscan::PortRecord> fallible_manual_registry_graph_ports(
    const holoscan::authoring::OpaqueGraphRef& graph) {
  if (fail_manual_registry_graph_ports) {
    throw holoscan::RuntimeError{
        holoscan::Error{holoscan::ErrorCode::kFailure, "AUTHORING_TEST_GRAPH_PORTS_FAILED"}};
  }
  return manual_registry_graph_ports(graph);
}

std::vector<holoscan::PortRecord> retaining_failing_manual_registry_graph_ports(
    const holoscan::authoring::OpaqueGraphRef& graph) {
  retained_failed_graph = graph;
  retained_failed_graph_port = graph.interface_port("input");
  throw holoscan::RuntimeError{
      holoscan::Error{holoscan::ErrorCode::kFailure, "AUTHORING_TEST_GRAPH_PORTS_FAILED"}};
}

holoscan::authoring::RegisteredGraphType make_manual_registry_graph_row(std::string_view name) {
  return {
      .name = name,
      .param_type = &holoscan::param_type_descriptor<RegistryGraphParams>(),
      .param_schema = nullptr,
      .parameter_introspection = holoscan::authoring::GraphParameterIntrospection::kPassThrough,
      .embed = &embed_manual_registry_graph,
      .embed_from_assignments = &embed_manual_registry_graph_from_assignments,
      .ports = &manual_registry_graph_ports,
  };
}

constexpr std::string_view kRegistryOperatorName{"holoscan.test.RegistryOperator"};
constexpr std::string_view kRegistryGraphName{"holoscan.test.RegistryGraph"};
constexpr std::string_view kRegistryRecursiveGraphName{"holoscan.test.RegistryRecursiveGraph"};

HOLOSCAN_REGISTER_OPERATOR(RegistryOperator, kRegistryOperatorName);
HOLOSCAN_REGISTER_GRAPH(RegistryGraph, kRegistryGraphName);
HOLOSCAN_REGISTER_GRAPH(RegistryRecursiveGraph, kRegistryRecursiveGraphName);

}  // namespace

// REQ: AUTHOR-003
TEST(AuthoringRegistry, EnumeratesAndLooksUpRegisteredOperators) {
  auto& registry = holoscan::authoring::authoring_registry();
  const auto found = registry.operator_type(kRegistryOperatorName);
  ASSERT_TRUE(found) << found.error().message;
  EXPECT_EQ(found->get().name, kRegistryOperatorName);
  EXPECT_EQ(found->get().param_schema, holoscan::param_schema_descriptor<RegistryParams>());

  const auto rows = registry.operators();
  EXPECT_NE(std::find_if(rows.begin(),
                         rows.end(),
                         [](const auto& row) { return row.get().name == kRegistryOperatorName; }),
            rows.end());
  EXPECT_TRUE(std::is_sorted(rows.begin(), rows.end(), [](const auto& left, const auto& right) {
    return left.get().name < right.get().name;
  }));
}

// REQ: AUTHOR-022
TEST(AuthoringRegistry, AutomaticRowsExposeRequiredOnLoadParameters) {
  // Key setup: Register an automatic row from a schema containing one required-on-load field.
  const auto row = holoscan::authoring::detail::make_registered_operator<RequiredRegistryOperator>(
      "holoscan.tests.RequiredRegistryOperator");
  holoscan::authoring::AuthoringRegistry registry;
  const auto registered = registry.register_operator(row);

  // Key assertion: The declared-schema row is admitted and exposes its canonical required field.
  ASSERT_TRUE(registered) << registered.error().message;
  ASSERT_EQ(row.required_parameters.size(), 1U);
  EXPECT_EQ(row.required_parameters.front(), "required_count");
}

// REQ: AUTHOR-022
TEST(AuthoringRegistry, PassThroughRowsExposeRequiredConstructorParameters) {
  // Key setup: Adapt an automatic row into a schema-less hand-authored pass-through row.
  auto row = holoscan::authoring::detail::make_registered_operator<RequiredRegistryOperator>(
      "holoscan.tests.RequiredPassThroughOperator");
  row.param_schema = nullptr;
  row.parameter_introspection = holoscan::authoring::OperatorParameterIntrospection::kPassThrough;
  holoscan::authoring::AuthoringRegistry registry;

  // Key assertion: Required-key metadata remains valid without claiming schema introspection.
  const auto registered = registry.register_operator(row);
  ASSERT_TRUE(registered) << registered.error().message;
  const auto description = registry.describe_operator(row.name);
  ASSERT_TRUE(description) << description.error().message;
  ASSERT_EQ(description->required_parameters.size(), 1U);
  EXPECT_EQ(description->required_parameters.front(), "required_count");
}

// REQ: AUTHOR-003
// REQ: AUTHOR-015
TEST(AuthoringRegistry, RejectsDuplicateRegistrationWithoutReplacingTheRow) {
  // Key setup: Attempt to replace the statically registered row under the same stable name.
  auto& registry = holoscan::authoring::authoring_registry();
  auto duplicate = registry.register_operator(
      holoscan::authoring::detail::make_registered_operator<RegistryOperator>(
          kRegistryOperatorName));

  // Key assertion: Duplicate registration fails and leaves the original metadata intact.
  ASSERT_FALSE(duplicate);
  EXPECT_EQ(duplicate.error().code, holoscan::ErrorCode::kDuplicateName);
  EXPECT_NE(duplicate.error().message.find("AUTHORING_REGISTRY_DUPLICATE"), std::string::npos);

  const auto found = registry.operator_type(kRegistryOperatorName);
  ASSERT_TRUE(found);
  EXPECT_EQ(found->get().param_schema, holoscan::param_schema_descriptor<RegistryParams>());
}

// REQ: AUTHOR-003
// REQ: AUTHOR-015
TEST(AuthoringRegistry, RejectsMalformedRegistrationRowsWithoutInsertion) {
  // Key setup: Submit an empty Operator row to an isolated registry.
  holoscan::authoring::AuthoringRegistry registry;
  const auto registered = registry.register_operator({});

  // Key assertion: Validation rejects the row and leaves the catalog empty.
  ASSERT_FALSE(registered);
  EXPECT_EQ(registered.error().code, holoscan::ErrorCode::kInvalidArgument);
  EXPECT_NE(registered.error().message.find("AUTHORING_REGISTRY_ROW_INVALID"), std::string::npos);
  EXPECT_TRUE(registry.operators().empty());
}

// REQ: AUTHOR-003
// REQ: AUTHOR-015
TEST(AuthoringRegistry, DefersAutomaticRegistrationFailureWithoutThrowing) {
  // Key setup: Repeat an automatic registration against an isolated local registry.
  holoscan::authoring::AuthoringRegistry registry;
  ASSERT_TRUE(registry.register_operator(
      holoscan::authoring::detail::make_registered_operator<RegistryOperator>(
          "holoscan.test.LocalRegistryOperator")));

  const bool registered = registry.register_operator_automatically(
      holoscan::authoring::detail::make_registered_operator<RegistryOperator>(
          "holoscan.test.LocalRegistryOperator"));

  // Key assertion: Static-initialization callers get no exception and can inspect deferred status.
  EXPECT_FALSE(registered);
  const auto status = registry.registration_status();
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().code, holoscan::ErrorCode::kDuplicateName);
  EXPECT_NE(status.error().message.find("AUTHORING_REGISTRY_DUPLICATE"), std::string::npos);
}

// REQ: AUTHOR-003
// REQ: AUTHOR-015
TEST(AuthoringRegistry, RetainsAutomaticFailureWhenDiagnosticsCannotAllocate) {
  // Key setup: Exhaust allocation throughout invalid-row handling and diagnostic retention.
  holoscan::authoring::AuthoringRegistry registry;
  bool registered = true;
  {
    const holoscan::test::ScopedAllocationExhaustion exhaustion{0U};
    registered = registry.register_operator_automatically({});
  }

  // Key assertion: The allocation-free latch keeps the failure inspectable without a rich error.
  EXPECT_FALSE(registered);
  const auto status = registry.registration_status();
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().code, holoscan::ErrorCode::kFailure);
  EXPECT_TRUE(status.error().message.empty());
}

// REQ: AUTHOR-003
// REQ: AUTHOR-015
TEST(AuthoringRegistry, RetainsAutomaticGraphFailureWhenDiagnosticsCannotAllocate) {
  // Key setup: Exhaust allocation throughout invalid Graph-row handling and diagnostic retention.
  holoscan::authoring::AuthoringRegistry registry;
  bool registered = true;
  {
    const holoscan::test::ScopedAllocationExhaustion exhaustion{0U};
    registered = registry.register_graph_automatically({});
  }

  // Key assertion: The allocation-free latch keeps the Graph failure inspectable too.
  EXPECT_FALSE(registered);
  const auto status = registry.registration_status();
  ASSERT_FALSE(status);
  EXPECT_EQ(status.error().code, holoscan::ErrorCode::kFailure);
  EXPECT_TRUE(status.error().message.empty());
}

// REQ: AUTHOR-003
TEST(AuthoringRegistry, IdentifiesMissingMandatoryOperatorRowCallback) {
  // Key setup: Clear one callback added to the complete registered-Operator row contract.
  holoscan::authoring::AuthoringRegistry registry;
  auto row = holoscan::authoring::detail::make_registered_operator<RegistryOperator>(
      "holoscan.test.IncompleteRegistryOperator");
  row.lifecycle_allow_self_restart = nullptr;

  const auto registered = registry.register_operator(std::move(row));

  // Key assertion: Registration names the missing field without publishing an incomplete row.
  ASSERT_FALSE(registered);
  EXPECT_EQ(registered.error().code, holoscan::ErrorCode::kInvalidArgument);
  EXPECT_EQ(registered.error().message,
            "AUTHORING_REGISTRY_ROW_INVALID kind=operator field=lifecycle_allow_self_restart");
  EXPECT_FALSE(registry.operator_type("holoscan.test.IncompleteRegistryOperator"));
}

// REQ: AUTHOR-003
// REQ: AUTHOR-015
TEST(AuthoringRegistry, ReturnsRichUnknownNameError) {
  const auto found =
      holoscan::authoring::authoring_registry().operator_type("holoscan.test.DoesNotExist");
  ASSERT_FALSE(found);
  EXPECT_EQ(found.error().code, holoscan::ErrorCode::kNotFound);
  EXPECT_NE(found.error().message.find("AUTHORING_REGISTRY_UNKNOWN"), std::string::npos);
  EXPECT_NE(found.error().message.find("did you load the module"), std::string::npos);
}

// REQ: AUTHOR-013
TEST(AuthoringRegistry, CopiesInstancePortsThroughTheRegisteredThunk) {
  // Graph:
  //
  // operator
  //
  // Key setup: Add one registered operator and query only that handle's setup records.
  const auto row = holoscan::authoring::authoring_registry().operator_type(kRegistryOperatorName);
  ASSERT_TRUE(row) << row.error().message;
  holoscan::Graph graph{"registered-ports"};
  auto added = row->get().add_to_graph_from_assignments(graph, "operator", {});
  ASSERT_TRUE(added) << added.error().message;

  const auto ports = added->ports();

  // Key assertion: Port records are copied in setup order without requiring a Graph snapshot.
  ASSERT_EQ(ports.size(), 2U);
  EXPECT_EQ(ports[0].name, "input");
  EXPECT_EQ(ports[0].direction, holoscan::PortDirection::kInput);
  EXPECT_EQ(ports[1].name, "output");
  EXPECT_EQ(ports[1].direction, holoscan::PortDirection::kOutput);
}

// REQ: AUTHOR-002
// REQ: AUTHOR-014
TEST(AuthoringRegistry, HarvestMatchesGraphViewGroundTruth) {
  // Graph:
  //
  // operator
  //
  // Key setup: Compare registry-harvested metadata with an independently composed Graph view.
  auto& registry = holoscan::authoring::authoring_registry();
  const auto harvested = registry.describe_operator(kRegistryOperatorName);
  ASSERT_TRUE(harvested) << harvested.error().message;
  ASSERT_EQ(harvested->ports.size(), 2U);
  EXPECT_EQ(harvested->param_schema, holoscan::param_schema_descriptor<RegistryParams>());

  holoscan::Graph graph{"ground_truth"};
  (void)graph.op<RegistryOperator>("operator");
  const holoscan::GraphView view = graph.view();

  // Key assertion: Harvested ports preserve the typed Graph's names, directions, and identities.
  ASSERT_EQ(view.ports().size(), harvested->ports.size());
  for (std::size_t index = 0U; index < view.ports().size(); ++index) {
    EXPECT_EQ(harvested->ports[index].name, view.ports()[index].name);
    EXPECT_EQ(harvested->ports[index].direction, view.ports()[index].direction);
    EXPECT_TRUE(holoscan::authoring::payload_identities_equal(
        harvested->ports[index].payload_identity, view.ports()[index].payload_identity));
  }
}

// REQ: AUTHOR-020
//   RegistryGraph
//   input --> RegistryOperator("operator") --> output
TEST(AuthoringGraphRegistry, EnumeratesLooksUpAndHarvestsReusableGraphs) {
  // Key setup: Look up the registered graph and enumerate all graph rows before lazy discovery.
  auto& registry = holoscan::authoring::authoring_registry();
  const auto found = registry.graph_type(kRegistryGraphName);
  ASSERT_TRUE(found) << found.error().message;
  EXPECT_EQ(found->get().name, kRegistryGraphName);
  EXPECT_EQ(found->get().param_schema, holoscan::param_schema_descriptor<RegistryGraphParams>());

  const auto rows = registry.graphs();
  EXPECT_NE(std::find_if(rows.begin(),
                         rows.end(),
                         [](const auto& row) { return row.get().name == kRegistryGraphName; }),
            rows.end());
  EXPECT_TRUE(std::is_sorted(rows.begin(), rows.end(), [](const auto& left, const auto& right) {
    return left.get().name < right.get().name;
  }));

  const auto description = registry.describe_graph(kRegistryGraphName);

  // Key assertion: Discovery exposes the graph's schema and ordered input/output interface.
  ASSERT_TRUE(description) << description.error().message;
  ASSERT_EQ(description->ports.size(), 2U);
  EXPECT_EQ(description->ports[0].name, "input");
  EXPECT_EQ(description->ports[0].direction, holoscan::PortDirection::kInput);
  EXPECT_EQ(description->ports[1].name, "output");
  EXPECT_EQ(description->ports[1].direction, holoscan::PortDirection::kOutput);
  EXPECT_EQ(description->param_schema, holoscan::param_schema_descriptor<RegistryGraphParams>());
}

// REQ: AUTHOR-020
//   graph["graph_row"]
//   `-- embed<RegistryGraph>("child")
//       input --> RegistryOperator("operator") --> output
TEST(AuthoringGraphRegistry, RowPreservesTypedHandleOperationsAndWeakOwnership) {
  // Key setup: Retain an erased child-graph reference after its owning graph leaves scope.
  auto& registry = holoscan::authoring::authoring_registry();
  const auto found = registry.graph_type(kRegistryGraphName);
  ASSERT_TRUE(found);

  holoscan::authoring::OpaqueGraphRef stale;
  {
    holoscan::Graph graph{"graph_row"};
    const std::array<holoscan::authoring::ParameterAssignment, 1U> assignments{{
        {.name = "count", .value = holoscan::authoring::ParameterValue::from(5U)},
    }};
    auto embedded = found->get().embed_from_assignments(
        graph, "child", std::span<const holoscan::authoring::ParameterAssignment>{assignments});
    ASSERT_TRUE(embedded) << embedded.error().message;
    stale = std::move(*embedded);
    ASSERT_TRUE(stale);
    const auto path = stale.path();
    ASSERT_TRUE(path.has_value());
    if (path.has_value()) {
      EXPECT_EQ(path->segments().front(), "child");
    }
    const auto input = stale.interface_port_record("input");
    const auto output = stale.interface_port_record("output");
    EXPECT_EQ(input.direction, holoscan::PortDirection::kInput);
    EXPECT_EQ(output.direction, holoscan::PortDirection::kOutput);
    EXPECT_TRUE(stale.interface_port("input"));
    EXPECT_TRUE(stale.interface_port("output"));
    auto region = graph.region("children");
    region.constrain(holoscan::RegionConstraint::bounded_memory());
    stale.region_add(region);
    EXPECT_EQ(graph.view().regions().front().graph_members.size(), 1U);
  }

  // Key assertion: Graph destruction invalidates both identity and typed interface access.
  EXPECT_FALSE(stale);
  EXPECT_FALSE(stale.path());
  try {
    (void)stale.interface_port("input");
    FAIL() << "stale reusable-Graph reference did not fail closed";
  } catch (const holoscan::RuntimeError& error) {
    EXPECT_EQ(holoscan::make_error(error).message, "GRAPH_HANDLE_INVALID");
  }
}

// REQ: AUTHOR-020
TEST(AuthoringGraphRegistry, AddsMixedErasedRegionBatchAtomically) {
  // Key setup: Combine an Operator and duplicate embedded Graph references in one erased batch.
  auto& registry = holoscan::authoring::authoring_registry();
  const auto graph_row = registry.graph_type(kRegistryGraphName);
  const auto operator_row = registry.operator_type(kRegistryOperatorName);
  ASSERT_TRUE(graph_row);
  ASSERT_TRUE(operator_row);

  holoscan::Graph graph{"mixed-region-batch"};
  const auto member = operator_row->get().add_to_graph(graph, "member", nullptr);
  const std::array<holoscan::authoring::ParameterAssignment, 1U> assignments{{
      {.name = "count", .value = holoscan::authoring::ParameterValue::from(5U)},
  }};
  auto embedded = graph_row->get().embed_from_assignments(
      graph, "child", std::span<const holoscan::authoring::ParameterAssignment>{assignments});
  ASSERT_TRUE(embedded) << embedded.error().message;
  auto region = graph.region("stage");
  const std::array operators{member};
  const std::array duplicate_graphs{*embedded, *embedded};

  try {
    holoscan::authoring::OpaqueOperatorRef::region_add_members(region, operators, duplicate_graphs);
    FAIL() << "duplicate mixed region batch was accepted";
  } catch (const holoscan::RuntimeError& exception) {
    const holoscan::Error error = holoscan::make_error(exception);
    EXPECT_EQ(error.code, holoscan::ErrorCode::kDuplicateName);
    EXPECT_NE(error.message.find("GRAPH_REGION_MEMBER_DUPLICATE"), std::string::npos);
  }
  ASSERT_EQ(graph.view().regions().size(), 1U);
  EXPECT_TRUE(graph.view().regions().front().members.empty());
  EXPECT_TRUE(graph.view().regions().front().graph_members.empty());

  const std::array corrected_graphs{*embedded};
  holoscan::authoring::OpaqueOperatorRef::region_add_members(region, operators, corrected_graphs);

  // Key assertion: The rejected batch changed neither member set and the mixed retry commits both.
  EXPECT_EQ(graph.view().regions().front().members.size(), 1U);
  EXPECT_EQ(graph.view().regions().front().graph_members.size(), 1U);
}

// REQ: AUTHOR-020
//   graph["recursive_row"]
//   `-- embed<RegistryRecursiveGraph>("recursive") --> self-embed rejected
TEST(AuthoringGraphRegistry, RejectsDuplicateUnknownAndRecursiveGraphRows) {
  // Key setup: Exercise duplicate, unknown-name, and recursive-embedding failure paths.
  auto& registry = holoscan::authoring::authoring_registry();
  auto duplicate = registry.register_graph(
      holoscan::authoring::detail::make_registered_graph<RegistryGraph>(kRegistryGraphName));
  ASSERT_FALSE(duplicate);
  EXPECT_EQ(duplicate.error().code, holoscan::ErrorCode::kDuplicateName);
  EXPECT_NE(duplicate.error().message.find("AUTHORING_REGISTRY_DUPLICATE"), std::string::npos);

  const auto unknown = registry.graph_type("holoscan.test.DoesNotExistGraph");
  ASSERT_FALSE(unknown);
  EXPECT_EQ(unknown.error().code, holoscan::ErrorCode::kNotFound);
  EXPECT_NE(unknown.error().message.find("AUTHORING_REGISTRY_UNKNOWN"), std::string::npos);

  const auto recursive = registry.graph_type(kRegistryRecursiveGraphName);
  ASSERT_TRUE(recursive);
  holoscan::Graph graph{"recursive_row"};

  // Key assertion: The erased registry thunk preserves the concrete graph recursion guard.
  try {
    (void)recursive->get().embed(graph, "recursive", nullptr);
    FAIL() << "registered concrete graph identity bypassed the recursion guard";
  } catch (const holoscan::RuntimeError& error) {
    EXPECT_EQ(holoscan::make_error(error).message, "GRAPH_EMBED_CYCLE");
  }
  EXPECT_EQ(graph.view().graphs().size(), 1U);
}

// REQ: AUTHOR-020
//   parent
//   `-- child[ManualRegistryGraph]
//       input --> RegistryOperator("operator") --> output
TEST(AuthoringGraphRegistry, SupportsHandAuthoredPassThroughRows) {
  // Key setup: Register a positional-constructor graph with a hand-authored assignment thunk.
  holoscan::authoring::AuthoringRegistry registry;
  constexpr std::string_view kName{"holoscan.test.ManualRegistryGraph"};
  ASSERT_TRUE(registry.register_graph(make_manual_registry_graph_row(kName)));
  const auto row = registry.graph_type(kName);
  ASSERT_TRUE(row) << row.error().message;
  holoscan::Graph graph{"parent"};
  const holoscan::authoring::ParameterAssignment assignment{
      .name = "count", .value = holoscan::authoring::ParameterValue::from(6U)};
  const auto embedded = row->get().embed_from_assignments(graph, "child", {&assignment, 1U});
  ASSERT_TRUE(embedded) << embedded.error().message;
  const auto described = registry.describe_graph(kName);

  // Key assertion: Pass-through rows remain discoverable and embed through their custom mapper.
  ASSERT_TRUE(described) << described.error().message;
  EXPECT_EQ(described->parameter_introspection,
            holoscan::authoring::GraphParameterIntrospection::kPassThrough);
  EXPECT_EQ(described->parameter_repr,
            "unknown constructor parameters, passed through by hand-authored row");
  ASSERT_EQ(row->get().ports(*embedded).size(), 2U);
}

// REQ: AUTHOR-020
TEST(AuthoringGraphRegistry, ExplicitParamsDoNotConstructUnusedGraphDefaults) {
  // Key setup: Materialize an explicit aggregate, then reset the observable default initializer.
  const auto row = holoscan::authoring::detail::make_registered_graph<LazyDefaultRegistryGraph>(
      "holoscan.test.LazyDefaultRegistryGraph");
  const holoscan::authoring::ParameterAssignment assignment{
      .name = "count", .value = holoscan::authoring::ParameterValue::from(5U)};
  auto materialized = holoscan::authoring::materialize_parameters(
      *row.param_type, row.param_schema, {&assignment, 1U});
  ASSERT_TRUE(materialized) << materialized.error().message;
  lazy_graph_defaults_constructed.store(0U, std::memory_order_relaxed);

  holoscan::Graph explicit_graph{"explicit-params"};
  const auto explicit_child = row.embed(explicit_graph, "child", &*materialized);

  // Key assertion: Explicit values bypass defaults, while the null path constructs them once.
  EXPECT_TRUE(explicit_child);
  EXPECT_EQ(lazy_graph_defaults_constructed.load(std::memory_order_relaxed), 0U);
  holoscan::Graph default_graph{"default-params"};
  const auto default_child = row.embed(default_graph, "child", nullptr);
  EXPECT_TRUE(default_child);
  EXPECT_EQ(lazy_graph_defaults_constructed.load(std::memory_order_relaxed), 1U);
}

// REQ: GRAPH-005
TEST(AuthoringGraphRegistry, ParameterReplacementAllocatesBeforePublishingChanges) {
  // Key setup: Count every allocation made by one successful parameter replacement.
  std::size_t allocation_count{};
  {
    holoscan::Graph graph{"parameter-count"};
    auto op = graph.op<RegistryOperator>("existing");
    const holoscan::test::ScopedAllocationCounter counter;
    op.set_params(RegistryParams{.count = 5U});
    allocation_count = counter.count();
  }
  ASSERT_GT(allocation_count, 0U);

  std::size_t observed_failures{};
  for (std::size_t checkpoint = 0U; checkpoint < allocation_count; ++checkpoint) {
    SCOPED_TRACE(checkpoint);
    holoscan::Graph graph{"parameter-failure"};
    auto op = graph.op<RegistryOperator>("existing");
    const auto* original_input = &op->input;
    const holoscan::GraphView before = graph.view();
    bool allocation_failed{};
    {
      const holoscan::test::ScopedAllocationFailure failure{checkpoint};
      try {
        op.set_params(RegistryParams{.count = 5U});
      } catch (const std::bad_alloc&) {
        allocation_failed = true;
      } catch (const holoscan::RuntimeError&) {
        allocation_failed = true;
      }
    }

    // Key assertion: Every observed allocation failure precedes publication, leaving the original
    // prototype and declaration usable before a successful replacement retry.
    if (allocation_failed) {
      ++observed_failures;
      EXPECT_EQ(&op->input, original_input);
      EXPECT_EQ(graph.view().identity(), before.identity());
      EXPECT_EQ(graph.view().generation(), before.generation());
      EXPECT_EQ(op->params().count, RegistryParams{}.count);
      EXPECT_NO_THROW(graph.configure(op->input).queue_depth(7U));
    }
    EXPECT_NO_THROW(op.set_params(RegistryParams{.count = 5U}));
    EXPECT_EQ(op->params().count, 5U);
    EXPECT_NO_THROW(graph.configure(op->input).queue_depth(9U));
  }
  EXPECT_GT(observed_failures, 0U);
}

// REQ: AUTHOR-020
TEST(AuthoringGraphRegistry, RollsBackWhenInstancePortHarvestFails) {
  // Key setup: Use a valid hand-authored embed thunk followed by a deliberately failing port thunk.
  auto row = make_manual_registry_graph_row("holoscan.test.FallibleManualRegistryGraph");
  row.ports = &fallible_manual_registry_graph_ports;
  holoscan::Graph graph{"atomic-registered-embed"};
  const auto before = graph.view();
  fail_manual_registry_graph_ports = true;

  try {
    (void)holoscan::authoring::embed_registered_graph(graph, row, "child", {});
    FAIL() << "failing instance port harvest unexpectedly succeeded";
  } catch (const holoscan::RuntimeError& error) {
    EXPECT_EQ(holoscan::make_error(error).message, "AUTHORING_TEST_GRAPH_PORTS_FAILED");
  }
  fail_manual_registry_graph_ports = false;

  // Key assertion: Failure restores identity and name ownership so the same child can be retried.
  EXPECT_EQ(graph.view().identity(), before.identity());
  EXPECT_EQ(graph.view().graphs().size(), 1U);
  const auto retried = holoscan::authoring::embed_registered_graph(graph, row, "child", {});
  ASSERT_TRUE(retried) << retried.error().message;
  EXPECT_EQ(retried->ports.size(), 2U);
}

// REQ: AUTHOR-020
TEST(AuthoringGraphRegistry, RollbackRestoresExistingRegionMembership) {
  // Key setup: Let the erased embed thunk add its new child to a region that predates the compound
  // operation, then fail during port harvesting after that nested mutation committed.
  auto row = make_manual_registry_graph_row("holoscan.test.RegionMutatingRegistryGraph");
  row.embed_from_assignments = &embed_manual_registry_graph_in_existing_region;
  row.ports = &fallible_manual_registry_graph_ports;
  holoscan::Graph graph{"atomic-existing-region"};
  auto region = graph.region("existing");
  const auto before = graph.view();
  manual_registry_graph_region = &region;
  fail_manual_registry_graph_ports = true;

  try {
    (void)holoscan::authoring::embed_registered_graph(graph, row, "child", {});
    FAIL() << "failing instance port harvest unexpectedly succeeded";
  } catch (const holoscan::RuntimeError& error) {
    EXPECT_EQ(holoscan::make_error(error).message, "AUTHORING_TEST_GRAPH_PORTS_FAILED");
  }
  fail_manual_registry_graph_ports = false;

  // Key assertion: Rollback restores both the child records and the pre-existing region record,
  // so observation and a same-name, same-membership retry remain valid.
  const auto restored = graph.view();
  EXPECT_EQ(restored.identity(), before.identity());
  ASSERT_EQ(restored.regions().size(), 1U);
  EXPECT_TRUE(restored.regions().front().graph_members.empty());
  const auto retried = holoscan::authoring::embed_registered_graph(graph, row, "child", {});
  manual_registry_graph_region = nullptr;
  ASSERT_TRUE(retried) << retried.error().message;
  ASSERT_EQ(graph.view().regions().size(), 1U);
  EXPECT_EQ(graph.view().regions().front().graph_members.size(), 1U);
}

// REQ: AUTHOR-020
TEST(AuthoringGraphRegistry, RollbackRestoresEveryExistingParentRecord) {
  // Key setup: Mutate a pre-existing port and replace the default clock after embedding the child,
  // then fail while harvesting the child interface.
  auto row = make_manual_registry_graph_row("holoscan.test.ParentMutatingRegistryGraph");
  row.embed_from_assignments = &embed_manual_registry_graph_after_mutating_parent;
  row.ports = &fallible_manual_registry_graph_ports;
  holoscan::Graph graph{"atomic-existing-parent-records"};
  auto parent = graph.op<RegistryOperator>("parent");
  graph.configure(parent->input).queue_depth(7U);
  const auto retained_clock = graph.add_clock<holoscan::ManualClock>("retained", std::int64_t{0});
  graph.set_default_clock(retained_clock);
  const auto before = graph.view();
  manual_registry_parent_input = &parent->input;
  mutate_manual_registry_parent_clock = true;
  fail_manual_registry_graph_ports = true;

  try {
    (void)holoscan::authoring::embed_registered_graph(graph, row, "child", {});
    FAIL() << "parent-mutating port harvest unexpectedly succeeded";
  } catch (const holoscan::RuntimeError& error) {
    EXPECT_EQ(holoscan::make_error(error).message, "AUTHORING_TEST_GRAPH_PORTS_FAILED");
  }
  fail_manual_registry_graph_ports = false;
  mutate_manual_registry_parent_clock = false;
  manual_registry_parent_input = nullptr;

  // Key assertion: Rollback restores the complete parent snapshot, including records mutated in
  // place, and removes the temporary clock name together with its dangling default selection.
  const auto restored = graph.view();
  EXPECT_EQ(restored.identity(), before.identity());
  const auto input = std::ranges::find_if(restored.ports(), [](const holoscan::PortRecord& port) {
    return port.owner_kind == holoscan::PortOwnerKind::kOperator &&
           port.direction == holoscan::PortDirection::kInput && port.name == "input";
  });
  ASSERT_NE(input, restored.ports().end());
  EXPECT_EQ(input->graph_override.queue_depth, 7U);
  ASSERT_EQ(restored.clocks().size(), 1U);
  EXPECT_EQ(graph.default_clock(), retained_clock);
  EXPECT_TRUE(restored.clocks().front().is_default);
  EXPECT_NO_THROW((void)graph.add_clock<holoscan::ManualClock>("temporary", std::int64_t{0}));
}

// REQ: AUTHOR-020
TEST(AuthoringGraphRegistry, ParentTensorConfiguratorSurvivesUnrelatedRollback) {
  // Key setup: Create a graph-owned Tensor configurator for an existing parent port inside an
  // embed callback, then fail while harvesting the child interface ports.
  auto row = make_manual_registry_graph_row("holoscan.test.TensorConfiguringRegistryGraph");
  row.embed_from_assignments = &embed_manual_registry_graph_after_configuring_parent_tensor;
  row.ports = &fallible_manual_registry_graph_ports;
  holoscan::Graph graph{"atomic-tensor-configurator"};
  auto parent = graph.op<RegistryTensorOperator>("parent");
  const auto state = holoscan::detail::GraphPayloadRegistrationAccess::state(graph).lock();
  ASSERT_TRUE(state);
  ASSERT_TRUE(state->tensor_configurators_.empty());
  manual_registry_parent_tensor_output = &parent->output;
  retained_parent_tensor_configurator = nullptr;
  fail_manual_registry_graph_ports = true;

  EXPECT_THROW((void)holoscan::authoring::embed_registered_graph(graph, row, "child", {}),
               holoscan::RuntimeError);
  fail_manual_registry_graph_ports = false;
  manual_registry_parent_tensor_output = nullptr;

  // Key assertion: Rollback preserves the promised reference target and the existing port's exact
  // declaration identity while restoring the failed callback's temporary mutation.
  ASSERT_NE(retained_parent_tensor_configurator, nullptr);
  ASSERT_EQ(state->tensor_configurators_.size(), 1U);
  EXPECT_FALSE(graph.view().ports().front().graph_override.diagnostic_alias);
  EXPECT_NO_THROW(retained_parent_tensor_configurator->diagnostic_alias("live-parent-tensor"));
  EXPECT_EQ(graph.view().ports().front().graph_override.diagnostic_alias, "live-parent-tensor");
  retained_parent_tensor_configurator = nullptr;
}

// REQ: AUTHOR-020
TEST(AuthoringGraphRegistry, RegisteredEmbedBlocksObservationUntilCommit) {
  // Key setup: Pause a registered embed after its child has mutated the parent, then attempt to
  // observe that same parent from another thread.
  auto row = make_manual_registry_graph_row("holoscan.test.PausingRegistryGraph");
  row.embed_from_assignments = &embed_manual_registry_graph_with_pause;
  holoscan::Graph graph{"serialized-registered-embed"};
  auto embedding = std::async(std::launch::async, [&] {
    return holoscan::authoring::embed_registered_graph(graph, row, "child", {});
  });
  const bool paused = registered_embed_paused.try_acquire_for(std::chrono::seconds{2});
  if (!paused) {
    release_registered_embed.release();
  }
  ASSERT_TRUE(paused);
  std::binary_semaphore observation_started{0};
  auto observation = std::async(std::launch::async, [&] {
    observation_started.release();
    return graph.view().graphs().size();
  });
  const bool observing = observation_started.try_acquire_for(std::chrono::seconds{2});
  if (!observing) {
    release_registered_embed.release();
  }
  ASSERT_TRUE(observing);
  const auto observation_status = observation.wait_for(std::chrono::milliseconds{50});
  release_registered_embed.release();

  // Key assertion: No snapshot can observe the partially attached child before the transaction
  // commits; after commit, both the result and complete two-graph view are visible.
  EXPECT_EQ(observation_status, std::future_status::timeout);
  const auto embedded = embedding.get();
  ASSERT_TRUE(embedded) << embedded.error().message;
  EXPECT_EQ(observation.get(), 2U);
}

// REQ: AUTHOR-020
TEST(AuthoringGraphRegistry, RollbackRestoresExistingOperatorPrototype) {
  // Key setup: Replace a pre-existing Operator prototype after embedding a child, then fail while
  // harvesting the child interface.
  auto row = make_manual_registry_graph_row("holoscan.test.PrototypeMutatingRegistryGraph");
  row.embed_from_assignments = &embed_manual_registry_graph_after_replacing_parent_params;
  row.ports = &retaining_failing_manual_registry_graph_ports;
  holoscan::Graph graph{"atomic-existing-prototype"};
  auto parent = graph.op<RegistryOperator>("parent");
  // The pointer identity is the subject of the post-rollback assertion; the analyzer does not
  // model its use through the GoogleTest comparison macro.
  // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
  const auto* original_input = &parent->input;
  graph.configure(parent->input).queue_depth(7U);
  const auto before = graph.view();

  try {
    (void)holoscan::authoring::embed_registered_graph(graph, row, "child", {});
    FAIL() << "prototype-mutating port harvest unexpectedly succeeded";
  } catch (const holoscan::RuntimeError& error) {
    EXPECT_EQ(holoscan::make_error(error).message, "AUTHORING_TEST_GRAPH_PORTS_FAILED");
  }

  // Key assertion: Rollback restores the original prototype and declaration addresses, including
  // their prior configuration, rather than retaining the replacement built inside the callback.
  EXPECT_EQ(graph.view().identity(), before.identity());
  EXPECT_EQ(graph.view().generation(), before.generation());
  EXPECT_EQ(&parent->input, original_input);
  EXPECT_EQ(parent->params().count, RegistryParams{}.count);
  EXPECT_NO_THROW(graph.configure(*original_input).queue_depth(9U));
  retained_failed_graph = {};
  retained_failed_graph_port = {};
}

// REQ: AUTHOR-020
TEST(AuthoringGraphRegistry, RolledBackGraphReferencesRemainPermanentlyInvalid) {
  // Key setup: Retain the erased handle published to the port thunk, then fail and reuse its
  // storage index for a different concrete child Graph.
  auto row = make_manual_registry_graph_row("holoscan.test.RetainedFailedRegistryGraph");
  row.ports = &retaining_failing_manual_registry_graph_ports;
  holoscan::Graph graph{"atomic-stale-handle"};
  retained_failed_graph = {};
  retained_failed_graph_port = {};
  try {
    (void)holoscan::authoring::embed_registered_graph(graph, row, "failed-child", {});
    FAIL() << "retaining port harvest unexpectedly succeeded";
  } catch (const holoscan::RuntimeError& error) {
    EXPECT_EQ(holoscan::make_error(error).message, "AUTHORING_TEST_GRAPH_PORTS_FAILED");
  }
  EXPECT_FALSE(retained_failed_graph);
  EXPECT_FALSE(retained_failed_graph.path());
  EXPECT_FALSE(retained_failed_graph_port);

  const auto replacement = graph.embed<RegistryGraph>("replacement", RegistryGraphParams{});
  ASSERT_TRUE(replacement);

  // Key assertion: Index reuse cannot resurrect the rolled-back identity or expose the replacement
  // child's path and ports through the retained reference.
  EXPECT_FALSE(retained_failed_graph);
  EXPECT_FALSE(retained_failed_graph.path());
  try {
    (void)retained_failed_graph.interface_port_records();
    FAIL() << "rolled-back graph reference resolved to its index replacement";
  } catch (const holoscan::RuntimeError& error) {
    EXPECT_EQ(holoscan::make_error(error).message, "GRAPH_HANDLE_INVALID");
  }
  const auto payload = holoscan::authoring::authoring_registry().payload_type("holoscan.uint32");
  ASSERT_TRUE(payload) << payload.error().message;
  try {
    (void)payload->get().configure_input(graph, retained_failed_graph_port);
    FAIL() << "rolled-back interface port configured its index replacement";
  } catch (const holoscan::RuntimeError& error) {
    EXPECT_EQ(holoscan::make_error(error).message, "GRAPH_HANDLE_INVALID");
  }
  retained_failed_graph = {};
  retained_failed_graph_port = {};
}

// REQ: AUTHOR-020
TEST(AuthoringGraphRegistry, RolledBackGraphConfiguratorsRemainPermanentlyInvalid) {
  // Key setup: Retain both configurator directions from an embedded interface, then fail port
  // harvest and reuse the removed child's storage index.
  auto row = make_manual_registry_graph_row("holoscan.test.RetainedFailedConfiguratorGraph");
  row.embed_from_assignments = &embed_manual_registry_graph_retaining_configurators;
  row.ports = &retaining_failing_manual_registry_graph_ports;
  holoscan::Graph graph{"atomic-stale-configurator"};
  retained_failed_input_configurator = {};
  retained_failed_output_configurator = {};
  try {
    (void)holoscan::authoring::embed_registered_graph(graph, row, "failed-child", {});
    FAIL() << "retaining port harvest unexpectedly succeeded";
  } catch (const holoscan::RuntimeError& error) {
    EXPECT_EQ(holoscan::make_error(error).message, "AUTHORING_TEST_GRAPH_PORTS_FAILED");
  }
  expect_retained_configurators_invalid();

  const auto replacement = graph.embed<RegistryGraph>("replacement", RegistryGraphParams{});
  ASSERT_TRUE(replacement);
  const auto live = holoscan::authoring::OpaqueGraphRef::from_handle(replacement);
  const auto before = graph.view();

  // Key assertion: Every mutator, alias, copy and moved configurator retains the failed
  // registration identity before and after index reuse without changing the replacement Graph.
  expect_retained_configurators_invalid();
  auto copied_input = retained_failed_input_configurator;
  auto moved_output = std::move(retained_failed_output_configurator);
  expect_graph_handle_invalid([&] { copied_input.queue_depth(17U); });
  expect_graph_handle_invalid([&] { moved_output.max_emits_per_compute(17U); });
  EXPECT_EQ(graph.view().identity(), before.identity());
  EXPECT_EQ(graph.view().generation(), before.generation());
  EXPECT_FALSE(live.interface_port_record("input").graph_override.queue_depth);
  EXPECT_FALSE(live.interface_port_record("output").graph_override.max_emits_per_compute);
  retained_failed_input_configurator = {};
  retained_failed_output_configurator = {};
  retained_failed_graph = {};
  retained_failed_graph_port = {};
}

// REQ: AUTHOR-020
TEST(AuthoringGraphRegistry, RolledBackTypedCapabilitiesRemainPermanentlyInvalid) {
  // Key setup: Retain typed port configurators, Operator/Flow handles, a flow configurator, and a
  // stable Tensor configurator reference from a callback whose later port harvest fails.
  auto row = make_manual_registry_graph_row("holoscan.test.RetainedTypedCapabilitiesGraph");
  row.embed_from_assignments = &embed_manual_registry_graph_retaining_typed_capabilities;
  row.ports = &retaining_failing_manual_registry_graph_ports;
  holoscan::Graph graph{"atomic-stale-typed-capabilities"};
  retained_typed_input_configurator.reset();
  retained_typed_output_configurator.reset();
  retained_operator_handle.reset();
  retained_flow_handle.reset();
  retained_flow_configurator.reset();
  retained_failed_tensor_configurator = nullptr;

  EXPECT_THROW((void)holoscan::authoring::embed_registered_graph(graph, row, "failed-child", {}),
               holoscan::RuntimeError);
  ASSERT_TRUE(retained_typed_input_configurator);
  ASSERT_TRUE(retained_typed_output_configurator);
  ASSERT_TRUE(retained_operator_handle);
  ASSERT_TRUE(retained_flow_handle);
  ASSERT_TRUE(retained_flow_configurator);
  ASSERT_NE(retained_failed_tensor_configurator, nullptr);
  auto retained_opaque_operator =
      holoscan::authoring::OpaqueOperatorRef::from_handle(*retained_operator_handle);

  expect_retired_capability([] { retained_typed_input_configurator->queue_depth(17U); });
  expect_retired_capability(
      [] { retained_typed_output_configurator->diagnostic_alias("retired-output"); });
  EXPECT_FALSE(retained_operator_handle->graph_valid());
  expect_retired_capability(
      [] { retained_operator_handle->set_params(RegistryParams{.count = 5U}); });
  expect_retired_capability([&] { (void)retained_opaque_operator.ports(); });
  EXPECT_FALSE(retained_flow_handle->graph_valid());
  expect_retired_capability([] { retained_flow_configurator->queue_depth(17U); });
  expect_retired_capability(
      [] { retained_failed_tensor_configurator->diagnostic_alias("retired-tensor"); });

  // Reuse every rolled-back storage position with declarations of the same concrete types.
  const auto replacement = graph.embed<RegistryGraph>("replacement", RegistryGraphParams{});
  auto replacement_source = graph.op<RegistryOperator>("replacement-source");
  auto replacement_target = graph.op<RegistryOperator>("replacement-target");
  auto replacement_flow = graph.add_flow(replacement_source->output, replacement_target->input);
  auto replacement_tensor = graph.op<RegistryTensorOperator>("replacement-tensor");
  auto& live_tensor = graph.configure(replacement_tensor->output);
  const auto before = graph.view();

  // Key assertion: Copies and moves retain the retired declaration tokens, and same-type/same-ID
  // replacements cannot revive any stale capability or receive one of its mutations.
  auto copied_input = *retained_typed_input_configurator;
  auto moved_output = std::move(*retained_typed_output_configurator);
  auto copied_operator = *retained_operator_handle;
  auto moved_flow = std::move(*retained_flow_handle);
  auto copied_flow_configurator = *retained_flow_configurator;
  expect_retired_capability([&] { copied_input.queue_depth(19U); });
  expect_retired_capability([&] { moved_output.max_emits_per_compute(19U); });
  EXPECT_FALSE(copied_operator.graph_valid());
  expect_retired_capability([&] { (void)copied_operator->input; });
  expect_retired_capability([&] { (void)retained_opaque_operator.ports(); });
  EXPECT_FALSE(moved_flow.graph_valid());
  expect_retired_capability([&] { (void)moved_flow.source(); });
  expect_retired_capability([&] { copied_flow_configurator.topic_name("retired-flow"); });
  expect_retired_capability(
      [] { retained_failed_tensor_configurator->diagnostic_alias("retired-tensor-again"); });
  EXPECT_EQ(graph.view().identity(), before.identity());
  EXPECT_EQ(graph.view().generation(), before.generation());

  EXPECT_NO_THROW(graph.configure(replacement->input).queue_depth(7U));
  EXPECT_NO_THROW(graph.configure(replacement_flow).queue_depth(7U));
  EXPECT_NO_THROW(live_tensor.diagnostic_alias("live-replacement-tensor"));

  retained_typed_input_configurator.reset();
  retained_typed_output_configurator.reset();
  retained_operator_handle.reset();
  retained_flow_handle.reset();
  retained_flow_configurator.reset();
  retained_failed_tensor_configurator = nullptr;
  retained_failed_graph = {};
  retained_failed_graph_port = {};
}

// REQ: AUTHOR-020
TEST(AuthoringGraphRegistry, RolledBackRegionHandleCannotMutateSameIdReplacement) {
  // Key setup: Retain a region handle from a callback whose later port harvest fails.
  auto row = make_manual_registry_graph_row("holoscan.test.RetainedRegionHandleGraph");
  row.embed_from_assignments = &embed_manual_registry_graph_retaining_placement_handles;
  row.ports = &fallible_manual_registry_graph_ports;
  holoscan::Graph graph{"atomic-stale-region"};
  auto live = graph.op<RegistryOperator>("live");
  retained_region_handle.reset();
  retained_partition_handle.reset();
  fail_manual_registry_graph_ports = true;
  EXPECT_THROW((void)holoscan::authoring::embed_registered_graph(graph, row, "failed-child", {}),
               holoscan::RuntimeError);
  fail_manual_registry_graph_ports = false;
  ASSERT_TRUE(retained_region_handle);
  EXPECT_TRUE(graph.view().regions().empty());
  expect_retired_capability([&] { retained_region_handle->add(live); });

  auto replacement = graph.region("replacement-region");
  ASSERT_EQ(replacement.id(), retained_region_handle->id());
  auto copied = *retained_region_handle;
  const auto before = graph.view();

  // Key assertion: Reusing the numeric region ID does not revive the retired capability or alter
  // the replacement declaration.
  expect_retired_capability([&] { copied.add(live); });
  EXPECT_EQ(graph.view().identity(), before.identity());
  EXPECT_EQ(graph.view().generation(), before.generation());
  ASSERT_EQ(graph.view().regions().size(), 1U);
  EXPECT_TRUE(graph.view().regions().front().members.empty());
  retained_region_handle.reset();
  retained_partition_handle.reset();
}

// REQ: AUTHOR-020
TEST(AuthoringGraphRegistry, RolledBackPartitionHandleCannotMutateSameIdReplacement) {
  // Key setup: Retain a partition handle from a callback whose later port harvest fails.
  auto row = make_manual_registry_graph_row("holoscan.test.RetainedPartitionHandleGraph");
  row.embed_from_assignments = &embed_manual_registry_graph_retaining_placement_handles;
  row.ports = &fallible_manual_registry_graph_ports;
  holoscan::Graph graph{"atomic-stale-partition"};
  auto live = graph.op<RegistryOperator>("live");
  retained_region_handle.reset();
  retained_partition_handle.reset();
  fail_manual_registry_graph_ports = true;
  EXPECT_THROW((void)holoscan::authoring::embed_registered_graph(graph, row, "failed-child", {}),
               holoscan::RuntimeError);
  fail_manual_registry_graph_ports = false;
  ASSERT_TRUE(retained_partition_handle);
  EXPECT_TRUE(graph.view().partitions().empty());
  expect_retired_capability([&] { retained_partition_handle->add(live); });

  auto replacement = graph.partition("replacement-partition");
  auto moved = std::move(*retained_partition_handle);
  const auto before = graph.view();

  // Key assertion: Reusing the numeric partition ID does not revive the retired capability or
  // alter the replacement declaration.
  expect_retired_capability([&] { moved.add(live); });
  EXPECT_EQ(graph.view().identity(), before.identity());
  EXPECT_EQ(graph.view().generation(), before.generation());
  ASSERT_EQ(graph.view().partitions().size(), 1U);
  EXPECT_TRUE(graph.view().partitions().front().members.empty());
  EXPECT_NO_THROW(replacement.add(live));
  retained_region_handle.reset();
  retained_partition_handle.reset();
}

// REQ: AUTHOR-020
TEST(AuthoringGraphRegistry, RolledBackImportedTopicCannotResolveSameIdReplacement) {
  // Key setup: Retain imported and exported topics from a callback whose port harvest fails.
  auto row = make_manual_registry_graph_row("holoscan.test.RetainedImportedTopicGraph");
  row.embed_from_assignments = &embed_manual_registry_graph_retaining_topic_handles;
  row.ports = &fallible_manual_registry_graph_ports;
  holoscan::Graph graph{"atomic-stale-import"};
  retained_imported_topic.reset();
  retained_exported_topic.reset();
  fail_manual_registry_graph_ports = true;
  EXPECT_THROW((void)holoscan::authoring::embed_registered_graph(graph, row, "failed-child", {}),
               holoscan::RuntimeError);
  fail_manual_registry_graph_ports = false;
  ASSERT_TRUE(retained_imported_topic);
  EXPECT_FALSE(retained_imported_topic->graph_valid());

  auto replacement = graph.import_topic<std::uint32_t>("holomq://replacement-import");
  ASSERT_TRUE(replacement.graph_valid());
  auto copied = *retained_imported_topic;
  const auto before = graph.view();

  // Key assertion: A same-ID replacement port cannot be resolved or configured through the
  // retired imported-topic capability.
  EXPECT_FALSE(copied.graph_valid());
  expect_retired_capability(
      [&] { graph.configure(copied.output()).diagnostic_alias("stale-import"); });
  EXPECT_EQ(graph.view().identity(), before.identity());
  EXPECT_EQ(graph.view().generation(), before.generation());
  EXPECT_FALSE(graph.view().ports().front().graph_override.diagnostic_alias);
  retained_imported_topic.reset();
  retained_exported_topic.reset();
}

// REQ: AUTHOR-020
TEST(AuthoringGraphRegistry, RolledBackExportedTopicCannotResolveSameIdReplacement) {
  // Key setup: Reuse both rolled-back external port IDs so the exported topic's numeric ID names
  // a live replacement input.
  auto row = make_manual_registry_graph_row("holoscan.test.RetainedExportedTopicGraph");
  row.embed_from_assignments = &embed_manual_registry_graph_retaining_topic_handles;
  row.ports = &fallible_manual_registry_graph_ports;
  holoscan::Graph graph{"atomic-stale-export"};
  retained_imported_topic.reset();
  retained_exported_topic.reset();
  fail_manual_registry_graph_ports = true;
  EXPECT_THROW((void)holoscan::authoring::embed_registered_graph(graph, row, "failed-child", {}),
               holoscan::RuntimeError);
  fail_manual_registry_graph_ports = false;
  ASSERT_TRUE(retained_exported_topic);
  EXPECT_FALSE(retained_exported_topic->graph_valid());

  auto replacement_import = graph.import_topic<std::uint32_t>("holomq://replacement-import");
  auto replacement_export = graph.export_topic<std::uint32_t>("holomq://replacement-export");
  ASSERT_TRUE(replacement_import.graph_valid());
  ASSERT_TRUE(replacement_export.graph_valid());
  auto moved = std::move(*retained_exported_topic);
  const auto before = graph.view();

  // Key assertion: A same-ID replacement port cannot be resolved or configured through the
  // retired exported-topic capability.
  EXPECT_FALSE(moved.graph_valid());
  expect_retired_capability([&] { graph.configure(moved.input()).queue_depth(17U); });
  EXPECT_EQ(graph.view().identity(), before.identity());
  EXPECT_EQ(graph.view().generation(), before.generation());
  EXPECT_FALSE(graph.view().ports().back().graph_override.queue_depth);
  retained_imported_topic.reset();
  retained_exported_topic.reset();
}

// REQ: AUTHOR-020
TEST(AuthoringGraphRegistry, RetiredPlacementRejectsAlternateAndErasedMutationRoutes) {
  // Key setup: Retain placement handles from a failed callback, then reuse both declaration IDs.
  holoscan::Graph graph{"atomic-stale-placement-routes"};
  auto first = graph.op<RegistryOperator>("first");
  auto second = graph.op<RegistryOperator>("second");
  auto child = graph.embed<RegistryGraph>("live-child", RegistryGraphParams{});
  const auto operator_row =
      holoscan::authoring::authoring_registry().operator_type(kRegistryOperatorName);
  ASSERT_TRUE(operator_row);
  const auto erased = operator_row->get().add_to_graph(graph, "erased", nullptr);
  const std::array erased_operators{erased};
  const std::array erased_graphs{holoscan::authoring::OpaqueGraphRef::from_handle(child)};

  auto row = make_manual_registry_graph_row("holoscan.test.RetainedPlacementRoutesGraph");
  row.embed_from_assignments = &embed_manual_registry_graph_retaining_placement_handles;
  row.ports = &fallible_manual_registry_graph_ports;
  retained_region_handle.reset();
  retained_partition_handle.reset();
  fail_manual_registry_graph_ports = true;
  EXPECT_THROW((void)holoscan::authoring::embed_registered_graph(graph, row, "failed-child", {}),
               holoscan::RuntimeError);
  fail_manual_registry_graph_ports = false;
  ASSERT_TRUE(retained_region_handle);
  ASSERT_TRUE(retained_partition_handle);

  auto replacement_region = graph.region("replacement-region");
  auto replacement_partition = graph.partition("replacement-partition");
  auto stale_region = std::move(*retained_region_handle);
  auto stale_partition = *retained_partition_handle;
  const auto before = graph.view();

  // Key assertion: Every placement mutator and erased bridge rejects the retired declaration
  // identity without changing the same-ID replacements.
  expect_retired_capability(
      [&] { stale_region.constrain(holoscan::RegionConstraint::bounded_memory()); });
  expect_retired_capability([&] { stale_region.profile(holoscan::Profile::kStandard); });
  expect_retired_capability([&] { stale_region.add(child); });
  expect_retired_capability([&] {
    holoscan::authoring::OpaqueOperatorRef::region_add_members(
        stale_region, erased_operators, erased_graphs);
  });
  expect_retired_capability([&] { stale_partition.add(first, second); });
  expect_retired_capability([&] { stale_partition.add(child); });
  expect_retired_capability([&] { stale_partition.include_all(); });
  expect_retired_capability([&] {
    holoscan::authoring::OpaqueOperatorRef::partition_add(stale_partition, erased_operators);
  });
  EXPECT_EQ(graph.view().identity(), before.identity());
  EXPECT_EQ(graph.view().generation(), before.generation());
  EXPECT_NO_THROW(
      replacement_region.add(first).constrain(holoscan::RegionConstraint::bounded_memory()));
  EXPECT_NO_THROW(replacement_partition.add(first, second));
  retained_region_handle.reset();
  retained_partition_handle.reset();
}

// REQ: AUTHOR-020
TEST(AuthoringGraphRegistry, LivePlacementAndTopicsSurviveUnrelatedRollback) {
  // Key setup: Keep live placement and topic handles across unrelated failed registrations.
  holoscan::Graph graph{"live-capabilities-across-rollback"};
  auto first = graph.op<RegistryOperator>("first");
  auto second = graph.op<RegistryOperator>("second");
  auto region = graph.region("live-region");
  auto partition = graph.partition("live-partition");
  auto imported = graph.import_topic<std::uint32_t>("holomq://live-import");
  auto exported = graph.export_topic<std::uint32_t>("holomq://live-export");
  const auto before = graph.view();
  retained_region_handle.reset();
  retained_partition_handle.reset();
  retained_imported_topic.reset();
  retained_exported_topic.reset();

  const auto fail_embed = [&](std::string_view type_name, auto embed_from_assignments) {
    auto row = make_manual_registry_graph_row(type_name);
    row.embed_from_assignments = embed_from_assignments;
    row.ports = &fallible_manual_registry_graph_ports;
    fail_manual_registry_graph_ports = true;
    EXPECT_THROW((void)holoscan::authoring::embed_registered_graph(graph, row, "failed-child", {}),
                 holoscan::RuntimeError);
    fail_manual_registry_graph_ports = false;
  };
  fail_embed("holoscan.test.UnrelatedPlacementFailure",
             &embed_manual_registry_graph_retaining_placement_handles);
  fail_embed("holoscan.test.UnrelatedTopicFailure",
             &embed_manual_registry_graph_retaining_topic_handles);

  // Key assertion: Complete rollback preserves the identity of declarations that predate it, so
  // copied and moved capabilities remain usable through their normal mutation paths.
  EXPECT_EQ(graph.view().identity(), before.identity());
  EXPECT_EQ(graph.view().generation(), before.generation());
  auto copied_region = region;
  auto moved_partition = std::move(partition);
  auto moved_import = std::move(imported);
  auto copied_export = exported;
  EXPECT_NO_THROW(copied_region.add(first).profile(holoscan::Profile::kStandard));
  EXPECT_NO_THROW(moved_partition.add(first, second));
  EXPECT_TRUE(moved_import.graph_valid());
  EXPECT_TRUE(copied_export.graph_valid());
  EXPECT_NO_THROW(graph.configure(moved_import.output()).diagnostic_alias("live-import"));
  EXPECT_NO_THROW(graph.configure(copied_export.input()).queue_depth(7U));
  retained_region_handle.reset();
  retained_partition_handle.reset();
  retained_imported_topic.reset();
  retained_exported_topic.reset();
}

// REQ: AUTHOR-020
TEST(AuthoringGraphRegistry, LiveGraphConfiguratorsSurviveRollbackAndExpireWithParent) {
  // Key setup: Retain live configurators, then roll back an unrelated failed child registration.
  {
    holoscan::Graph graph{"live-configurator-lifetime"};
    auto row = make_manual_registry_graph_row("holoscan.test.LiveConfiguratorGraph");
    row.embed_from_assignments = &embed_manual_registry_graph_retaining_configurators;
    const auto live = holoscan::authoring::embed_registered_graph(graph, row, "live", {});
    ASSERT_TRUE(live) << live.error().message;
    auto failing_row = make_manual_registry_graph_row("holoscan.test.UnrelatedFailedGraph");
    failing_row.ports = &retaining_failing_manual_registry_graph_ports;
    EXPECT_THROW(
        (void)holoscan::authoring::embed_registered_graph(graph, failing_row, "failed", {}),
        holoscan::RuntimeError);

    // Key assertion: Unrelated rollback preserves the live identity and all direction-appropriate
    // mutations while direction mismatches continue to reject normally.
    EXPECT_NO_THROW(
        retained_failed_input_configurator.queue_depth(7U).diagnostic_alias("live-input"));
    EXPECT_NO_THROW(retained_failed_output_configurator.max_emits_per_compute(2U)
                        .late_joiner_history_depth(4U)
                        .diagnostic_alias("live-output"));
    const auto input = live->graph.interface_port_record("input");
    const auto output = live->graph.interface_port_record("output");
    EXPECT_EQ(input.graph_override.queue_depth, 7U);
    EXPECT_EQ(input.graph_override.diagnostic_alias, "live-input");
    EXPECT_EQ(output.graph_override.max_emits_per_compute, 2U);
    EXPECT_EQ(output.graph_override.late_joiner_history_depth, 4U);
    EXPECT_EQ(output.graph_override.diagnostic_alias, "live-output");
    EXPECT_THROW(retained_failed_input_configurator.max_emits_per_compute(2U),
                 holoscan::RuntimeError);
    EXPECT_THROW(retained_failed_output_configurator.queue_depth(7U), holoscan::RuntimeError);
  }

  // Key assertion: Destruction of the parent permanently invalidates both retained configurators.
  expect_retained_configurators_invalid();
  retained_failed_input_configurator = {};
  retained_failed_output_configurator = {};
  retained_failed_graph = {};
  retained_failed_graph_port = {};
}

// REQ: AUTHOR-020
TEST(AuthoringGraphRegistry, RollsBackEveryAllocationFailureInCompleteRegisteredEmbed) {
  // Key setup: Count typed embedding and the complete row operation independently so fault
  // injection starts only after the typed child has committed.
  const auto row = holoscan::authoring::detail::make_registered_graph<RegistryGraph>(
      "holoscan.test.AtomicRegistryGraph");
  std::size_t typed_allocation_count{};
  {
    holoscan::Graph graph{"measure-typed-embed"};
    const holoscan::test::ScopedAllocationCounter counter;
    (void)graph.embed<RegistryGraph>("child", RegistryGraphParams{});
    typed_allocation_count = counter.count();
  }
  std::size_t allocation_count{};
  {
    holoscan::Graph graph{"measure-registered-embed"};
    const holoscan::test::ScopedAllocationCounter counter;
    const auto embedded = holoscan::authoring::embed_registered_graph(graph, row, "child", {});
    ASSERT_TRUE(embedded) << embedded.error().message;
    allocation_count = counter.count();
  }
  ASSERT_GT(allocation_count, typed_allocation_count + 1U);

  // Key assertion: Erased-holder and port-result allocation failures after typed insertion leave
  // the snapshot unchanged and permit same-name retry. Earlier typed allocation coverage belongs
  // to Graph::embed; the extra leading allocation owns this compound rollback scope.
  for (std::size_t checkpoint = typed_allocation_count + 1U; checkpoint < allocation_count;
       ++checkpoint) {
    holoscan::Graph graph{"fail-registered-embed"};
    const auto before = graph.view();
    bool succeeded = false;
    try {
      const holoscan::test::ScopedAllocationFailure failure{checkpoint};
      succeeded =
          static_cast<bool>(holoscan::authoring::embed_registered_graph(graph, row, "child", {}));
    } catch (const std::bad_alloc&) {
      succeeded = false;
    }
    EXPECT_FALSE(succeeded) << "checkpoint " << checkpoint;
    EXPECT_EQ(graph.view().identity(), before.identity()) << "checkpoint " << checkpoint;
    EXPECT_EQ(graph.view().graphs().size(), 1U) << "checkpoint " << checkpoint;
    const auto retried = holoscan::authoring::embed_registered_graph(graph, row, "child", {});
    ASSERT_TRUE(retried) << "checkpoint " << checkpoint << ": " << retried.error().message;
  }
}
