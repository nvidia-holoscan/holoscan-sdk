/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

#include <holoscan/core/parameter_schema.hpp>

namespace holoscan {
namespace {

// WideParams exercises range validation at the extremes of uint64 and double.
// The sequence field's range [max-1, max] probes near-overflow boundary behaviour;
// the ratio field uses double to catch NaN/infinity rejection (REQ: PARAM-005).
struct WideParams {
  std::uint64_t sequence{std::numeric_limits<std::uint64_t>::max()};
  double ratio{1.0};
};

HOLOSCAN_PARAM_SCHEMA(WideParams, holoscan::param_schema("nvidia.holoscan.test.WideParams", 1),
                      holoscan::field<&WideParams::sequence>("sequence")
                          .range(std::numeric_limits<std::uint64_t>::max() - 1U,
                                 std::numeric_limits<std::uint64_t>::max()),
                      holoscan::field<&WideParams::ratio>("ratio").range(0.0, 2.0));

// REQ: PARAM-005
// Verifies that range constraints work correctly at the top of the uint64 domain and
// that NaN/infinity are always rejected for floating-point range fields.
TEST(ParameterSchemaWide, WideUnsignedAndFloatingRangesRetainExactDomains) {
  // Key setup: sequence starts at max (in-range), then stepped to max-1 (still in range),
  // then max-2 (out of range). Separately, NaN and infinity are injected into ratio.
  WideParams exact_wide;
  EXPECT_TRUE(validate_params(exact_wide).ok());
  exact_wide.sequence = std::numeric_limits<std::uint64_t>::max() - 1U;
  EXPECT_TRUE(validate_params(exact_wide).ok());
  exact_wide.sequence = std::numeric_limits<std::uint64_t>::max() - 2U;
  EXPECT_EQ(validate_params(exact_wide).code, ParamValidationCode::kOutOfRange);

  exact_wide.sequence = std::numeric_limits<std::uint64_t>::max();
  exact_wide.ratio = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(validate_params(exact_wide).code, ParamValidationCode::kOutOfRange);
  exact_wide.ratio = std::numeric_limits<double>::infinity();
  EXPECT_EQ(validate_params(exact_wide).code, ParamValidationCode::kOutOfRange);
}

}  // namespace
}  // namespace holoscan
