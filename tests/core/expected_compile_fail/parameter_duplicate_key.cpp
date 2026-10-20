/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/parameter_schema.hpp>

struct Params {
  int first{};
  int second{};
};

// REQ: PARAM-003
HOLOSCAN_PARAM_SCHEMA(Params, holoscan::param_schema("holoscan.test.DuplicateKey", 1),
                      holoscan::field<&Params::first>("value"),
                      holoscan::field<&Params::second>("value"));

constexpr auto kSchema = holoscan_param_schema(holoscan::detail::type_tag<Params>{});

int main() {}
