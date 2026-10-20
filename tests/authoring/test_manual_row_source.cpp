/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "manual_row_source.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <gtest/gtest.h>

#include <holoscan/authoring/register_clock.hpp>
#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/authoring/register_payload.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "../common/allocation_counter.hpp"

namespace holoscan::authoring::examples {

static_assert(!std::default_initializable<ManualRowSource>);

}  // namespace holoscan::authoring::examples

namespace {

using holoscan::authoring::OpaqueOperatorRef;
using holoscan::authoring::OperatorParameterIntrospection;
using holoscan::authoring::ParameterAssignment;
using holoscan::authoring::RegisteredOperatorType;
using holoscan::authoring::examples::ManualRowSource;
using holoscan::authoring::examples::ManualRowSourceParameters;

constexpr std::uint32_t kLimit = 4U;
constexpr std::string_view kSinkName{"holoscan.test.manual_row.CountingSink"};
constexpr std::string_view kPassThroughLimitName{"count"};

std::atomic<std::uint32_t> g_received{};

// Supplies a non-allocating std::exception for the invoke_add fallback regression.
// Representative test: AuthoringManualRow.TranslatesUnexpectedExceptionsWithoutAllocating.
class StaticAddException final : public std::exception {
 public:
  [[nodiscard]] const char* what() const noexcept override { return "unexpected add failure"; }
};

// Supplies the public manual registration seam exercised by
// AuthoringManualRow.MatchesTypedAuthoringAndExecutionForBothRows.
struct ManualRowSourceThunks final
    : holoscan::authoring::ManualOperatorRegistrationThunks<ManualRowSource> {
  using ConstructorParameters = ManualRowSourceParameters;

  static OpaqueOperatorRef add_with_limit(holoscan::Graph& graph, std::string_view name,
                                          std::uint32_t limit) {
    return make_operator_ref(graph.op<ManualRowSource>(name, limit));
  }

  static OpaqueOperatorRef add_to_graph(holoscan::Graph& graph, std::string_view name,
                                        const holoscan::authoring::MaterializedParameters* params) {
    const ConstructorParameters defaults{};
    if (params != nullptr &&
        (params->type() == nullptr ||
         params->type()->local_type_token !=
             holoscan::param_type_descriptor<ConstructorParameters>().local_type_token)) {
      throw holoscan::RuntimeError{holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                                                   "AUTHORING_PARAMETER_TYPE_MISMATCH"}};
    }
    const auto* values =
        params == nullptr ? &defaults : static_cast<const ConstructorParameters*>(params->data());
    return add_with_limit(graph, name, values->limit);
  }

  static void set_params(const OpaqueOperatorRef&,
                         const holoscan::authoring::MaterializedParameters&) {
    throw holoscan::RuntimeError{holoscan::Error{holoscan::ErrorCode::kNotSupported,
                                                 "AUTHORING_MANUAL_ROW_PARAMETERS_IMMUTABLE"}};
  }
};

// Counts values from ManualRowSource and completes the representative
// AuthoringManualRow.MatchesTypedAuthoringAndExecutionForBothRows graph.
class CountingSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input"); }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    auto value = input.receive_data();
    if (!value) {
      return holoscan::make_unexpected(std::move(value).error());
    }
    g_received.fetch_add(1U, std::memory_order_relaxed);
    if (*value == kLimit) {
      context.request_completion();
    }
    return {};
  }

  holoscan::Input<std::uint32_t> input;
};

HOLOSCAN_REGISTER_OPERATOR(CountingSink, "holoscan.test.manual_row.CountingSink");

holoscan::expected<OpaqueOperatorRef, holoscan::Error> add_declared_row_from_assignments(
    holoscan::Graph& graph, std::string_view name,
    std::span<const ParameterAssignment> assignments) {
  if (assignments.empty()) {
    return ManualRowSourceThunks::invoke_add(
        [&] { return ManualRowSourceThunks::add_to_graph(graph, name, nullptr); });
  }
  auto materialized = holoscan::authoring::materialize_parameters(
      holoscan::param_type_descriptor<ManualRowSourceParameters>(),
      holoscan::param_schema_descriptor<ManualRowSourceParameters>(),
      assignments);
  if (!materialized) {
    return holoscan::make_unexpected(std::move(materialized).error());
  }
  return ManualRowSourceThunks::invoke_add(
      [&] { return ManualRowSourceThunks::add_to_graph(graph, name, &*materialized); });
}

holoscan::expected<OpaqueOperatorRef, holoscan::Error> add_pass_through_row_from_assignments(
    holoscan::Graph& graph, std::string_view name,
    std::span<const ParameterAssignment> assignments) {
  if (assignments.empty()) {
    return ManualRowSourceThunks::invoke_add(
        [&] { return ManualRowSourceThunks::add_to_graph(graph, name, nullptr); });
  }
  if (assignments.size() != 1U || assignments.front().name != kPassThroughLimitName) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                                                     "AUTHORING_MANUAL_ROW_ASSIGNMENT_UNKNOWN"});
  }
  std::uint32_t limit{};
  if (assignments.front().value.kind != holoscan::authoring::ParameterValueKind::kScalar ||
      !assignments.front().value.scalar.try_as(limit) || limit == 0U) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                                                     "AUTHORING_MANUAL_ROW_ASSIGNMENT_INVALID"});
  }
  return ManualRowSourceThunks::invoke_add(
      [&] { return ManualRowSourceThunks::add_with_limit(graph, name, limit); });
}

RegisteredOperatorType make_manual_row(std::string_view name, bool expose_descriptor) {
  using Thunks = ManualRowSourceThunks;
  return RegisteredOperatorType{
      .name = name,
      .param_type = &holoscan::param_type_descriptor<ManualRowSourceParameters>(),
      .param_schema = expose_descriptor
                          ? holoscan::param_schema_descriptor<ManualRowSourceParameters>()
                          : nullptr,
      .parameter_introspection = expose_descriptor
                                     ? OperatorParameterIntrospection::kConstructorReplay
                                     : OperatorParameterIntrospection::kPassThrough,
      .add_to_graph = &Thunks::add_to_graph,
      .add_to_graph_from_assignments = expose_descriptor ? &add_declared_row_from_assignments
                                                         : &add_pass_through_row_from_assignments,
      .set_params = &Thunks::set_params,
      .port = &Thunks::port,
      .ports = &Thunks::ports,
      .region_add = &Thunks::region_add,
      .partition_add = &Thunks::partition_add,
      .lifecycle_before = &Thunks::lifecycle_before,
      .lifecycle_allow_self_restart = &Thunks::lifecycle_allow_self_restart,
      .lifecycle_allow_external_restart = &Thunks::lifecycle_allow_external_restart,
      .lifecycle_allow_restart_request = &Thunks::lifecycle_allow_restart_request,
  };
}

[[maybe_unused]] const bool kManualRowsRegistered = [] {
  auto& registry = holoscan::authoring::authoring_registry();
  const bool with_descriptor = registry.register_operator_automatically(
      make_manual_row(holoscan::authoring::examples::kManualRowWithDescriptorName, true));
  const bool without_descriptor = registry.register_operator_automatically(
      make_manual_row(holoscan::authoring::examples::kManualRowWithoutDescriptorName, false));
  return with_descriptor && without_descriptor;
}();

[[nodiscard]] holoscan::ExecutionPlan build_typed_plan() {
  holoscan::Graph graph{"manual-row-parity"};
  const auto source = graph.op<ManualRowSource>("source", kLimit);
  const auto sink = graph.op<CountingSink>("sink");
  graph.add_flow(source->output, sink->input, holoscan::ConnectionOptions{.queue_depth = kLimit});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
  return holoscan::compile(graph);
}

[[nodiscard]] holoscan::ExecutionPlan build_row_plan(std::string_view row_name) {
  auto& registry = holoscan::authoring::authoring_registry();
  const auto source_row = registry.operator_type(row_name);
  const auto sink_row = registry.operator_type(kSinkName);
  const auto payload_row = registry.payload_type("holoscan.uint32");
  const auto clock_row = registry.clock_type("holoscan.RealtimeClock");
  if (!source_row || !sink_row || !payload_row || !clock_row) {
    throw std::runtime_error("manual-row test registry setup failed");
  }

  holoscan::Graph graph{"manual-row-parity"};
  const std::array assignments{
      ParameterAssignment{row_name == holoscan::authoring::examples::kManualRowWithDescriptorName
                              ? std::string_view{"limit"}
                              : kPassThroughLimitName,
                          holoscan::authoring::ParameterValue::from(kLimit)}};
  auto source = source_row->get().add_to_graph_from_assignments(graph, "source", assignments);
  auto sink = sink_row->get().add_to_graph_from_assignments(graph, "sink", {});
  if (!source || !sink) {
    throw std::runtime_error("manual-row test assignment mapping failed");
  }
  payload_row->get().add_flow(graph,
                              source_row->get().port(*source, "output"),
                              sink_row->get().port(*sink, "input"),
                              holoscan::ConnectionOptions{.queue_depth = kLimit});
  graph.set_default_clock(clock_row->get().add_to_graph(graph, "clock"));
  return holoscan::compile(graph);
}

void expect_clean_run(const holoscan::ExecutionPlan& plan) {
  ASSERT_TRUE(plan.ok()) << plan.json();
  g_received.store(0U, std::memory_order_relaxed);
  holoscan::RunSession session = holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  const bool timed_out = session.is_running();
  if (timed_out) {
    session.request_stop();
  }
  session.wait();

  EXPECT_FALSE(timed_out);
  EXPECT_EQ(g_received.load(std::memory_order_relaxed), kLimit);
}

// REQ: AUTHOR-016
// REQ: AUTHOR-025
TEST(AuthoringManualRow, ExposesDeclaredOrDegradedParameterIntrospection) {
  // Key setup: Resolve the two manual rows that differ only in schema visibility.
  auto& registry = holoscan::authoring::authoring_registry();
  const auto described_row =
      registry.operator_type(holoscan::authoring::examples::kManualRowWithDescriptorName);
  const auto described =
      registry.describe_operator(holoscan::authoring::examples::kManualRowWithDescriptorName);
  ASSERT_TRUE(described_row);
  ASSERT_TRUE(described);
  EXPECT_NE(described->param_schema, nullptr);
  EXPECT_EQ(described->parameter_introspection, OperatorParameterIntrospection::kConstructorReplay);
  EXPECT_NE(described->parameter_repr.find("ManualRowSourceParameters"), std::string::npos);
  EXPECT_EQ(described_row->get().bind_config, nullptr);

  const auto degraded_row =
      registry.operator_type(holoscan::authoring::examples::kManualRowWithoutDescriptorName);
  const auto degraded =
      registry.describe_operator(holoscan::authoring::examples::kManualRowWithoutDescriptorName);
  ASSERT_TRUE(degraded_row);
  ASSERT_TRUE(degraded);

  // Key assertion: The rows advertise declared and pass-through introspection respectively.
  EXPECT_EQ(degraded->param_schema, nullptr);
  EXPECT_EQ(degraded->parameter_introspection, OperatorParameterIntrospection::kPassThrough);
  EXPECT_EQ(degraded->parameter_repr, "unknown parameters, passed through by hand-authored row");
  EXPECT_EQ(degraded_row->get().bind_config, nullptr);
}

// REQ: AUTHOR-016
TEST(AuthoringManualRow, ValidatesNamedConstructorAssignmentsWhenDeclared) {
  // Key setup: Materialize an out-of-range constructor assignment through the declared row.
  auto row = holoscan::authoring::authoring_registry().operator_type(
      holoscan::authoring::examples::kManualRowWithDescriptorName);
  ASSERT_TRUE(row);
  holoscan::Graph graph{"manual-row-validation"};
  const std::array assignments{
      ParameterAssignment{"limit", holoscan::authoring::ParameterValue::from(0U)}};
  auto added = row->get().add_to_graph_from_assignments(graph, "source", assignments);

  // Key assertion: Schema validation rejects the assignment before construction.
  ASSERT_FALSE(added);
  EXPECT_NE(added.error().message.find("PARAM_VALIDATION_FAILED"), std::string::npos);
}

// REQ: AUTHOR-016
TEST(AuthoringManualRow, PassThroughUsesPrivateConstructorVocabulary) {
  // Key setup: Address the metadata-free row with a private name absent from its null schema.
  auto row = holoscan::authoring::authoring_registry().operator_type(
      holoscan::authoring::examples::kManualRowWithoutDescriptorName);
  ASSERT_TRUE(row);
  holoscan::Graph graph{"manual-row-pass-through"};
  const std::array accepted{ParameterAssignment{kPassThroughLimitName,
                                                holoscan::authoring::ParameterValue::from(kLimit)}};
  const std::array rejected{
      ParameterAssignment{"limit", holoscan::authoring::ParameterValue::from(kLimit)}};

  auto added = row->get().add_to_graph_from_assignments(graph, "source", accepted);
  auto unknown = row->get().add_to_graph_from_assignments(graph, "other", rejected);

  // Key assertion: The row owns both acceptance and rejection without discoverable metadata.
  EXPECT_TRUE(added);
  ASSERT_FALSE(unknown);
  EXPECT_NE(unknown.error().message.find("AUTHORING_MANUAL_ROW_ASSIGNMENT_UNKNOWN"),
            std::string::npos);
}

// REQ: AUTHOR-001
TEST(AuthoringManualRow, ReturnsGraphAddFailuresAsData) {
  // Key setup: Add the same registered automatic Operator name twice through the expected path.
  auto row = holoscan::authoring::authoring_registry().operator_type(kSinkName);
  ASSERT_TRUE(row);
  holoscan::Graph graph{"duplicate-registered-add"};
  auto first = row->get().add_to_graph_from_assignments(graph, "sink", {});
  ASSERT_TRUE(first);
  auto duplicate = row->get().add_to_graph_from_assignments(graph, "sink", {});

  // Key assertion: The ordinary duplicate-name mistake remains failure data at the erased seam.
  ASSERT_FALSE(duplicate);
  EXPECT_EQ(duplicate.error().code, holoscan::ErrorCode::kDuplicateName);
  EXPECT_NE(duplicate.error().message.find("GRAPH_OPERATOR_NAME_DUPLICATE"), std::string::npos);
}

// REQ: AUTHOR-025
TEST(AuthoringManualRow, ReturnsManualRowGraphAddFailuresAsData) {
  // Key setup: Add each hand-authored row twice under one name through the value-only route.
  for (const std::string_view name :
       {holoscan::authoring::examples::kManualRowWithDescriptorName,
        holoscan::authoring::examples::kManualRowWithoutDescriptorName}) {
    SCOPED_TRACE(name);
    auto row = holoscan::authoring::authoring_registry().operator_type(name);
    ASSERT_TRUE(row);
    holoscan::Graph graph{"duplicate-manual-row-add"};
    auto first = row->get().add_to_graph_from_assignments(graph, "source", {});
    ASSERT_TRUE(first) << first.error().message;
    auto duplicate = row->get().add_to_graph_from_assignments(graph, "source", {});

    // Key assertion: The manual thunk returns the ordinary Graph failure as data.
    ASSERT_FALSE(duplicate);
    EXPECT_EQ(duplicate.error().code, holoscan::ErrorCode::kDuplicateName);
    EXPECT_NE(duplicate.error().message.find("GRAPH_OPERATOR_NAME_DUPLICATE"), std::string::npos);
  }
}

// REQ: AUTHOR-013
// REQ: AUTHOR-025
TEST(AuthoringManualRow, TranslatesUnexpectedExceptionsWithoutAllocating) {
  // Key setup: Invoke every fallback catch path while all dynamic allocations fail.
  const holoscan::RuntimeError runtime_error{
      holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "detailed graph failure"}};
  holoscan::expected<OpaqueOperatorRef, holoscan::Error> runtime_result =
      holoscan::make_unexpected(holoscan::Error{});
  {
    const holoscan::test::ScopedAllocationExhaustion exhaustion{0U};
    runtime_result =
        ManualRowSourceThunks::invoke_add([&]() -> OpaqueOperatorRef { throw runtime_error; });
  }

  holoscan::expected<OpaqueOperatorRef, holoscan::Error> standard_result =
      holoscan::make_unexpected(holoscan::Error{});
  {
    const holoscan::test::ScopedAllocationExhaustion exhaustion{0U};
    standard_result = ManualRowSourceThunks::invoke_add(
        []() -> OpaqueOperatorRef { throw StaticAddException{}; });
  }

  holoscan::expected<OpaqueOperatorRef, holoscan::Error> unknown_result =
      holoscan::make_unexpected(holoscan::Error{});
  {
    const holoscan::test::ScopedAllocationExhaustion exhaustion{0U};
    unknown_result = ManualRowSourceThunks::invoke_add([]() -> OpaqueOperatorRef { throw 17; });
  }

  // Key assertion: Every fallback remains Error data and uses no diagnostic allocation.
  ASSERT_FALSE(runtime_result);
  EXPECT_EQ(runtime_result.error().code, holoscan::ErrorCode::kInvalidArgument);
  EXPECT_TRUE(runtime_result.error().message.empty());
  ASSERT_FALSE(standard_result);
  EXPECT_EQ(standard_result.error().code, holoscan::ErrorCode::kFailure);
  EXPECT_TRUE(standard_result.error().message.empty());
  ASSERT_FALSE(unknown_result);
  EXPECT_EQ(unknown_result.error().code, holoscan::ErrorCode::kFailure);
  EXPECT_TRUE(unknown_result.error().message.empty());
}

// REQ: AUTHOR-017
TEST(AuthoringManualRow, KeepsAutomaticRegistrationAsTheSdkDefault) {
  // Key setup: Compare an ordinary macro row with the explicit manual row.
  static_assert(std::default_initializable<CountingSink>);
  auto& registry = holoscan::authoring::authoring_registry();
  const auto automatic = registry.describe_operator(kSinkName);
  const auto manual =
      registry.describe_operator(holoscan::authoring::examples::kManualRowWithDescriptorName);
  ASSERT_TRUE(automatic);
  ASSERT_TRUE(manual);

  // Key assertion: Automatic registration needs no manual parameter-introspection contract.
  EXPECT_EQ(automatic->parameter_introspection, OperatorParameterIntrospection::kNone);
  EXPECT_EQ(manual->parameter_introspection, OperatorParameterIntrospection::kConstructorReplay);
}

// REQ: AUTHOR-013
// REQ: AUTHOR-025
TEST(AuthoringManualRow, MatchesTypedAuthoringAndExecutionForBothRows) {
  // Key setup: Compile the same graph through typed, described-row, and pass-through-row paths.
  //
  //   source --uint32--> sink
  const holoscan::ExecutionPlan typed = build_typed_plan();
  const holoscan::ExecutionPlan described =
      build_row_plan(holoscan::authoring::examples::kManualRowWithDescriptorName);
  const holoscan::ExecutionPlan degraded =
      build_row_plan(holoscan::authoring::examples::kManualRowWithoutDescriptorName);
  ASSERT_TRUE(typed.ok()) << typed.json();
  ASSERT_TRUE(described.ok()) << described.json();
  ASSERT_TRUE(degraded.ok()) << degraded.json();

  // Key assertion: Both manual rows preserve plan bytes and runtime behavior.
  EXPECT_EQ(described.json(), typed.json());
  EXPECT_EQ(degraded.json(), typed.json());
  expect_clean_run(typed);
  expect_clean_run(described);
  expect_clean_run(degraded);
}

}  // namespace
