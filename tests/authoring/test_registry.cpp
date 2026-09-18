/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

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

constexpr std::string_view kRegistryOperatorName{"holoscan.test.RegistryOperator"};

HOLOSCAN_REGISTER_OPERATOR(RegistryOperator, kRegistryOperatorName);

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

// REQ: AUTHOR-003
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
TEST(AuthoringRegistry, RetainsAutomaticFailureWhenDiagnosticsCannotAllocate) {
  // Key setup: Exhaust allocation throughout invalid-row handling and diagnostic retention.
  holoscan::authoring::AuthoringRegistry registry;
  bool registered = true;
  {
    holoscan::test::ScopedAllocationExhaustion exhaustion{0U};
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
TEST(AuthoringRegistry, ReturnsRichUnknownNameError) {
  const auto found =
      holoscan::authoring::authoring_registry().operator_type("holoscan.test.DoesNotExist");
  ASSERT_FALSE(found);
  EXPECT_EQ(found.error().code, holoscan::ErrorCode::kNotFound);
  EXPECT_NE(found.error().message.find("AUTHORING_REGISTRY_UNKNOWN"), std::string::npos);
  EXPECT_NE(found.error().message.find("did you load the module"), std::string::npos);
}

// REQ: AUTHOR-002
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
