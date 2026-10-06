/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>

// Omits a declared schema so AUTHORING_COMPILE_FAIL.opaque_params can pin macro rejection.
struct OpaqueParams {
  int value{};
};

// Carries the unsupported opaque aggregate into the automatic registration path.
// Representative test: AUTHORING_COMPILE_FAIL.opaque_params.
class OpaqueParamsOperator final : public holoscan::Operator<OpaqueParams> {
 public:
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }
};

// REQ: AUTHOR-003
// REQ: AUTHOR-015
HOLOSCAN_REGISTER_OPERATOR(OpaqueParamsOperator, "holoscan.test.OpaqueParamsOperator");

int main() {
  return 0;
}
