/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <vector>

#include <holoscan/core/operator.hpp>
#include <holoscan/operators/ping_tx/ping_tx.hpp>
#include <holoscan/test/test_harness.hpp>
#include <holoscan/test/validation_functions.hpp>

namespace holoscan::test {

TEST_F(OperatorTestBase, PingTx) {
  // Create test values operator will emit
  // PingRxOp emits values starting from 1 and incrementing by 1 each time
  const int num_pings = 5;
  std::vector<int> expected_outputs;
  for (int i = 1; i <= num_pings; ++i) {
    expected_outputs.push_back(i);
  }

  // Create validator for test harness to check outputs
  auto validator = validators<int>(create_exact_equality_validator(expected_outputs));

  // Create test harness and run
  auto test_harness = create_operator_test<ops::PingTxOp>()
                          ->add_condition<holoscan::CountCondition>("count_condition", num_pings)
                          ->add_output_port("out", validator);

  test_harness->run_test();
}

}  // namespace holoscan::test
