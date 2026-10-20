/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <stdexcept>

// CCCL's no-exception CUDA driver header names this host exception through the enclosing
// cuda::std namespace even though the throw path is disabled.  Make that dependent header
// parse without changing the production target or its exception policy.
namespace cuda::std {
using ::std::invalid_argument;
}  // namespace cuda::std

#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/rate.hpp>

#include "core/operator_spec_compiler_access.hpp"

namespace holoscan {
namespace {

using ScalarInput = Input<std::uint32_t>;
using ScalarOutput = Output<std::uint32_t>;

// REQ: OPERATOR-048
bool valid_and_invalid_rate_authoring_is_observable() {
  const Rate unreduced_rate{120U, 2U};
  const Rate canonical_rate{60U, 1U};
  const RateRatio unreduced_ratio{6U, 8U};
  const RateRatio canonical_ratio{3U, 4U};
  const auto normalized_rate = unreduced_rate.normalized();
  const auto normalized_ratio = unreduced_ratio.normalized();
  if (normalized_rate != canonical_rate || normalized_ratio != canonical_ratio ||
      !unreduced_rate.equivalent_to(canonical_rate) ||
      !unreduced_ratio.equivalent_to(canonical_ratio) || Rate{0U, 1U}.normalized() ||
      RateRatio{1U, 0U}.normalized()) {
    return false;
  }

  OperatorSpec valid_spec;
  ScalarInput input{"input"};
  ScalarOutput output{"output"};
  ScalarOutput followed{"followed"};
  ScalarOutput lifecycle{"lifecycle"};
  valid_spec.input(input).expected_rate(Rate::hz(60U));
  valid_spec.output(output).nominal_rate(Rate::hz(120U));
  valid_spec.output(followed).follows(input, RateRatio{1U, 2U});
  valid_spec.output(lifecycle)
      .max_nominal_rate(Rate::hz(120U))
      .lifecycle_nominal_rate(LifecycleStage::kDiscover);

  const auto valid_input = detail::OperatorSpecCompilerAccess::port(valid_spec, 0U);
  const auto valid_output = detail::OperatorSpecCompilerAccess::port(valid_spec, 1U);
  const auto valid_followed = detail::OperatorSpecCompilerAccess::port(valid_spec, 2U);
  const auto valid_lifecycle = detail::OperatorSpecCompilerAccess::port(valid_spec, 3U);
  constexpr RateRatio kExpectedFollowedRatio{1U, 2U};
  if (valid_input.rate_error.code != ErrorCode::kSuccess ||
      valid_input.expected_rate != Rate::hz(60U) ||
      valid_output.rate_error.code != ErrorCode::kSuccess ||
      valid_output.nominal_rate != Rate::hz(120U) ||
      valid_followed.rate_error.code != ErrorCode::kSuccess ||
      valid_followed.rate_kind != PortRateKind::kFollowsInput ||
      valid_followed.follows_input_declaration != &input ||
      valid_followed.follows_ratio != kExpectedFollowedRatio ||
      valid_lifecycle.rate_error.code != ErrorCode::kSuccess ||
      valid_lifecycle.rate_kind != PortRateKind::kLifecycleNominal ||
      valid_lifecycle.max_nominal_rate != Rate::hz(120U) ||
      valid_lifecycle.lifecycle_binding_stage != LifecycleStage::kDiscover) {
    return false;
  }

  OperatorSpec zero_spec;
  ScalarOutput zero_output{"zero_output"};
  zero_spec.output(zero_output).nominal_rate(Rate::hz(0U)).nominal_rate(Rate::hz(60U));
  const auto zero = detail::OperatorSpecCompilerAccess::port(zero_spec, 0U);
  if (zero.rate_error.code != ErrorCode::kInvalidArgument ||
      zero.rate_error.message != "PORT_RATE_INVALID" ||
      zero.rate_kind != PortRateKind::kUnspecified) {
    return false;
  }

  OperatorSpec conflict_spec;
  ScalarOutput conflict_output{"conflict_output"};
  conflict_spec.output(conflict_output).nominal_rate(Rate::hz(60U)).follows(input);
  const auto conflict = detail::OperatorSpecCompilerAccess::port(conflict_spec, 0U);
  return conflict.rate_error.code == ErrorCode::kInvalidArgument &&
         conflict.rate_error.message == "PORT_RATE_CONFLICT" &&
         conflict.rate_kind == PortRateKind::kFixedNominal;
}

}  // namespace
}  // namespace holoscan

int main() {
  return holoscan::valid_and_invalid_rate_authoring_is_observable() ? 0 : 1;
}
