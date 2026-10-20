/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Smoke: an operator with a typed parameter struct receives the bound values
// inside compute() via ExecutionContext::params<T>(), and the param-generation
// counter is delivered alongside the struct.

// Topology: [synthetic fixture] → ParamSmokeOperator (no I/O ports; reads params only)

#include <cstdint>

#include <holoscan/core/parameter_schema.hpp>

#include "operator_test_fixture.hpp"

namespace {

// A minimal typed parameter struct with a single range-validated field.
struct SmokeParams {
  std::uint32_t multiplier{1};
};

// REQ: PARAM-010
HOLOSCAN_PARAM_SCHEMA(SmokeParams, holoscan::param_schema("holoscan.test.SmokeParams", 1),
                      holoscan::field<&SmokeParams::multiplier>("multiplier").range(1U, 8U));

// Records the multiplier and param generation seen during compute() for
// post-invocation verification.
class ParamSmokeOperator final : public holoscan::Operator<SmokeParams> {
 public:
  std::uint32_t observed{};
  holoscan::ParamGeneration generation{99};

  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    observed = context.params<SmokeParams>().multiplier;
    generation = context.params_generation();
    return {};
  }
};

}  // namespace

int main() {
  ParamSmokeOperator op;
  holoscan::test::SyntheticOperatorFixture fixture{op};
  if (!fixture.declare_ports() || !fixture.bind_initial_params(SmokeParams{.multiplier = 4})) {
    return 1;  // exit 1: port declaration or initial param binding failed
  }
  const holoscan::detail::SyntheticActivationOptions initial{
      .params_generation = holoscan::ParamGeneration{0},
  };
  if (!fixture.invoke_once(initial)) {
    return 2;  // exit 2: synthetic operator invocation failed
  }
  return op.observed == 4 && op.generation == 0 ? 0 : 3;
  // Exit 3: observed multiplier != 4 or param generation != 0
}
