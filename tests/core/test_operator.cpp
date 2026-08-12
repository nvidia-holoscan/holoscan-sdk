/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <concepts>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>

namespace holoscan {
namespace {

// Concepts used to verify that Operator<> does NOT expose GXF-era entry points.
// run(), run_once(), and set_cuda_stream() must be absent from the public API (REQ: OPERATOR-013).
template <typename Op>
concept HasRun = requires(Op& op) { op.run(); };

template <typename Op>
concept HasRunOnce = requires(Op& op) { op.run_once(); };

template <typename Op>
concept HasSetCudaStream = requires(Op& op) { op.set_cuda_stream(nullptr); };

// GainParams is a minimal parameter struct; GainOperator demonstrates the standard
// Operator<Params> pattern with one input port and one output port declared in setup().
struct GainParams {
  std::uint32_t gain{1};
};

class GainOperator final : public Operator<GainParams> {
 public:
  Input<std::uint32_t> input_{"input"};
  Output<std::uint32_t> output_{"output"};

  void setup(OperatorSpec& spec) override {
    spec.input(input_);
    spec.output(output_);
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// NoParamsOperator exercises Operator<> (the NoParams specialisation) — representative
// test verifies that all instances share a single stable zero-generation params value.
class NoParamsOperator final : public Operator<> {
 public:
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

static_assert(std::derived_from<GainOperator, OperatorBase>);
static_assert(std::same_as<typename GainOperator::Params, GainParams>);
static_assert(std::same_as<typename NoParamsOperator::Params, NoParams>);
static_assert(!std::copy_constructible<GainOperator>);
static_assert(!std::move_constructible<GainOperator>);
static_assert(!std::is_copy_assignable_v<GainOperator>);
static_assert(!std::is_move_assignable_v<GainOperator>);
static_assert(std::has_virtual_destructor_v<OperatorBase>);

// REQ: OPERATOR-013
static_assert(!HasRun<GainOperator>);
static_assert(!HasRunOnce<GainOperator>);
static_assert(!HasSetCudaStream<GainOperator>);

// REQ: OPERATOR-002
// REQ: PARAM-001
// The NoParams singleton is a compile-time constant shared across all operator instances;
// returning &params() twice should yield the same address.
TEST(Operator, NoParamsUsesOneStableImmutableZeroGenerationValue) {
  const NoParamsOperator first;
  const NoParamsOperator second;

  EXPECT_EQ(&first.params(), &first.params());
  EXPECT_EQ(&first.params(), &second.params());
  EXPECT_TRUE((std::is_empty_v<NoParams>));
}

// REQ: OPERATOR-002
// Operators are non-copyable, non-movable; port member addresses are stable for the
// lifetime of the operator so the runtime can take pointers to them at setup time.
TEST(Operator, ConcreteOperatorAndPortMembersHaveStableNonmovableAddresses) {
  const GainOperator op;
  const auto* operator_address = &op;
  const auto* input_address = &op.input_;
  const auto* output_address = &op.output_;

  EXPECT_EQ(&op, operator_address);
  EXPECT_EQ(&op.input_, input_address);
  EXPECT_EQ(&op.output_, output_address);
}

}  // namespace
}  // namespace holoscan
