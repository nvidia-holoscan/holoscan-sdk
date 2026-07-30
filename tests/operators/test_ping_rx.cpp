/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <vector>

#include <holoscan/core/operator.hpp>
#include <holoscan/operators/ping_rx/ping_rx.hpp>
#include <holoscan/test/test_harness.hpp>

namespace holoscan::test {

TEST_F(OperatorTestBase, PingRx) {
  auto test_harness =
      create_operator_test<ops::PingRxOp>()->add_input_port("in", std::vector<int>{1, 2, 3});

  test_harness->run_test();
}

}  // namespace holoscan::test
