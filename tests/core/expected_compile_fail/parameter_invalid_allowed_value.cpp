/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <limits>

#include <holoscan/core/parameter_schema.hpp>

struct Params {
  double value{};
};

// REQ: PARAM-003
HOLOSCAN_PARAM_SCHEMA(Params, holoscan::param_schema("holoscan.test.InvalidAllowedValue", 1),
                      holoscan::field<&Params::value>("value").allowed_values(
                          {0.0, std::numeric_limits<double>::infinity()}));

constexpr auto kSchema = holoscan_param_schema(holoscan::detail::type_tag<Params>{});

int main() {}
