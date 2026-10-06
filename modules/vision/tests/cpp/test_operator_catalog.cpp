// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

#include <gtest/gtest.h>

#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/vision/operator_catalog.hpp>
#include <holoscan/vision/resize_op.hpp>

namespace holoscan::vision {
namespace {

using authoring::ParameterAssignment;
using authoring::ParameterValue;

// Describes one expected catalog row.
// Representative test: VisionOperatorCatalog.RegistersEveryOperatorIdempotently.
struct OperatorExpectation {
  std::string_view name;
  std::size_t port_count;
  std::size_t required_parameter_count;
};

constexpr std::array<OperatorExpectation, 9U> kOperatorExpectations{
    OperatorExpectation{"holoscan.vision.BayerDemosaicOp", 2U, 0U},
    OperatorExpectation{"holoscan.vision.ConvertToOp", 2U, 0U},
    OperatorExpectation{"holoscan.vision.CopyMakeBorderOp", 2U, 3U},
    OperatorExpectation{"holoscan.vision.CustomCropOp", 2U, 1U},
    OperatorExpectation{"holoscan.vision.NonMaximumSuppressionOp", 3U, 2U},
    OperatorExpectation{"holoscan.vision.NormalizeOp", 2U, 0U},
    OperatorExpectation{"holoscan.vision.ReformatOp", 2U, 0U},
    OperatorExpectation{"holoscan.vision.RemapOp", 3U, 1U},
    OperatorExpectation{"holoscan.vision.ResizeOp", 2U, 1U},
};

// REQ: AUTHOR-017
// REQ: AUTHOR-022
TEST(VisionOperatorCatalog, RegistersEveryOperatorIdempotently) {
  // Key setup: Initialize the complete Vision catalog twice.
  ASSERT_TRUE(register_vision_operator_catalog());
  ASSERT_TRUE(register_vision_operator_catalog());

  // Key assertion: Every expected row remains unique, declared, and fully described.
  auto& registry = authoring::authoring_registry();
  for (const OperatorExpectation& expectation : kOperatorExpectations) {
    auto row = registry.operator_type(expectation.name);
    ASSERT_TRUE(row) << expectation.name << ": " << row.error().message;
    EXPECT_EQ(row->get().parameter_introspection,
              authoring::OperatorParameterIntrospection::kDeclared);
    EXPECT_EQ(row->get().bind_config, nullptr) << expectation.name;
    auto description = registry.describe_operator(expectation.name);
    ASSERT_TRUE(description) << expectation.name << ": " << description.error().message;
    EXPECT_EQ(description->ports.size(), expectation.port_count) << expectation.name;
    EXPECT_EQ(description->required_parameters.size(), expectation.required_parameter_count)
        << expectation.name;
    EXPECT_NE(description->param_schema, nullptr) << expectation.name;
  }
}

// REQ: AUTHOR-021
// REQ: AUTHOR-018
TEST(VisionOperatorCatalog, AuthorsTheSameResizePlanAsTheTypedApi) {
  // Key setup: Author equivalent single-operator graphs through typed, named, and numeric forms.
  //
  //   typed:      resize
  //   registered: resize
  //   numeric:    resize
  ASSERT_TRUE(register_vision_operator_catalog());
  constexpr ResizeOpParams kParams{
      .width = 3, .height = 2, .max_batch_size = 2U, .interpolation = InterpolationMode::kCubic};

  Graph typed_graph{"vision-registry-parity"};
  auto typed_resize = typed_graph.op<ResizeOp>("resize");
  typed_resize.set_params(kParams);
  typed_graph.set_default_clock(
      typed_graph.add_clock<ManualClock>("compile-clock", std::int64_t{0}));

  Graph registered_graph{"vision-registry-parity"};
  Graph numeric_graph{"vision-registry-parity"};
  const auto row = authoring::authoring_registry().operator_type("holoscan.vision.ResizeOp");
  ASSERT_TRUE(row) << row.error().message;
  const std::array assignments{
      ParameterAssignment{"max_batch_size", ParameterValue::from(std::uint64_t{2})},
      ParameterAssignment{"width", ParameterValue::from(std::int64_t{3})},
      ParameterAssignment{"height", ParameterValue::from(std::int64_t{2})},
      ParameterAssignment{"interpolation", ParameterValue::from(std::string_view{"cubic"})},
  };
  auto registered_resize =
      row->get().add_to_graph_from_assignments(registered_graph, "resize", assignments);
  ASSERT_TRUE(registered_resize) << registered_resize.error().message;
  registered_graph.set_default_clock(
      registered_graph.add_clock<ManualClock>("compile-clock", std::int64_t{0}));

  const std::array numeric_assignments{
      ParameterAssignment{"max_batch_size", ParameterValue::from(std::uint64_t{2})},
      ParameterAssignment{"width", ParameterValue::from(std::int64_t{3})},
      ParameterAssignment{"height", ParameterValue::from(std::int64_t{2})},
      ParameterAssignment{"interpolation", ParameterValue::from(std::uint64_t{2})},
  };
  auto numeric_resize =
      row->get().add_to_graph_from_assignments(numeric_graph, "resize", numeric_assignments);
  ASSERT_TRUE(numeric_resize) << numeric_resize.error().message;
  numeric_graph.set_default_clock(
      numeric_graph.add_clock<ManualClock>("compile-clock", std::int64_t{0}));

  CompileOptions typed_options;
  typed_options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
      .operator_path = "resize", .output_port = "output", .device = DeviceId{0}});
  CompileOptions registered_options;
  registered_options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
      .operator_path = "resize", .output_port = "output", .device = DeviceId{0}});
  CompileOptions numeric_options;
  numeric_options.deployment.bind_tensor_output_device(TensorOutputDevicePlacement{
      .operator_path = "resize", .output_port = "output", .device = DeviceId{0}});
  const ExecutionPlan typed_plan = compile(typed_graph, std::move(typed_options));
  const ExecutionPlan registered_plan = compile(registered_graph, std::move(registered_options));
  const ExecutionPlan numeric_plan = compile(numeric_graph, std::move(numeric_options));

  // Key assertion: Every authoring form produces the same admitted plan.
  ASSERT_TRUE(typed_plan.ok()) << typed_plan.json();
  ASSERT_TRUE(registered_plan.ok()) << registered_plan.json();
  ASSERT_TRUE(numeric_plan.ok()) << numeric_plan.json();
  EXPECT_EQ(registered_plan.json(), typed_plan.json());
  EXPECT_EQ(numeric_plan.json(), typed_plan.json());
}

// REQ: AUTHOR-015
// REQ: AUTHOR-018
TEST(VisionOperatorCatalog, RejectsInvalidCapacityAndEnumAssignments) {
  // Key setup: Author single-Resize graphs with invalid capacity and enum assignments.
  //
  //   invalid assignment -X-> resize
  ASSERT_TRUE(register_vision_operator_catalog());
  const auto row = authoring::authoring_registry().operator_type("holoscan.vision.ResizeOp");
  ASSERT_TRUE(row) << row.error().message;

  Graph invalid_capacity_graph{"invalid-capacity"};
  const std::array invalid_capacity{
      ParameterAssignment{"max_batch_size", ParameterValue::from(std::uint64_t{0})}};
  auto capacity_result =
      row->get().add_to_graph_from_assignments(invalid_capacity_graph, "resize", invalid_capacity);

  // Key assertion: Invalid capacity and enum values retain their stable error families.
  ASSERT_FALSE(capacity_result);
  EXPECT_NE(capacity_result.error().message.find("PARAM_VALIDATION_FAILED"), std::string::npos);

  Graph invalid_enum_graph{"invalid-enum"};
  const std::array invalid_enum{
      ParameterAssignment{"interpolation", ParameterValue::from(std::string_view{"lanczos"})}};
  auto enum_result =
      row->get().add_to_graph_from_assignments(invalid_enum_graph, "resize", invalid_enum);
  ASSERT_FALSE(enum_result);
  EXPECT_NE(enum_result.error().message.find("AUTHORING_PARAMETER_TYPE_MISMATCH"),
            std::string::npos);
}

// REQ: AUTHOR-021
TEST(VisionOperatorCatalog, ReturnsTypedAddFailuresAsData) {
  // Key setup: Add one valid Resize operator, then repeat its Graph-local name.
  //
  //   resize --duplicate name--X
  ASSERT_TRUE(register_vision_operator_catalog());
  const auto row = authoring::authoring_registry().operator_type("holoscan.vision.ResizeOp");
  ASSERT_TRUE(row) << row.error().message;
  Graph graph{"typed-add-failure"};
  const std::array assignments{
      ParameterAssignment{"max_batch_size", ParameterValue::from(std::uint64_t{2})},
  };
  ASSERT_TRUE(row->get().add_to_graph_from_assignments(graph, "resize", assignments));

  const auto duplicate = row->get().add_to_graph_from_assignments(graph, "resize", assignments);

  // Key assertion: The erased authoring seam translates the typed exception into Error data.
  ASSERT_FALSE(duplicate);
  EXPECT_NE(duplicate.error().message.find("GRAPH_OPERATOR_NAME_DUPLICATE"), std::string::npos);
}

}  // namespace
}  // namespace holoscan::vision
