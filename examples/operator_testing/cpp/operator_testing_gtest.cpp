/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>
#include <holoscan/testing/operator_test_harness.hpp>
#include "operators.hpp"

using namespace operator_testing_example;

TEST(ScaleOperator, ConfiguredFactor) {
  auto h = holoscan::testing::create_operator_test<ScaleOp>();
  h.set_params(ScaleParams{.factor = 2});
  h.push(&ScaleOp::in, std::int32_t{21});
  h.capture<std::int32_t>("out");
  const auto result = h.run();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.value<std::int32_t>("out"), 42);
}
