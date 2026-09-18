/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>

// Requires a constructor argument so automatic registration cannot instantiate its typed thunks.
// Representative test: AUTHORING_COMPILE_FAIL.non_default_operator.
class NonDefaultOperator final : public holoscan::Operator<> {
 public:
  explicit NonDefaultOperator(int value) : value_(value) {}

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

 private:
  int value_{};
};

// REQ: AUTHOR-003
HOLOSCAN_REGISTER_OPERATOR(NonDefaultOperator, "holoscan.test.NonDefaultOperator");

int main() {
  return 0;
}
