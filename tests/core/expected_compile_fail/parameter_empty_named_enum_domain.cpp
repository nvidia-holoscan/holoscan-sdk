/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>

#include <holoscan/core/parameter_schema.hpp>

enum class Mode : std::uint8_t {
  kAutomatic,
};

struct Params {
  Mode mode{Mode::kAutomatic};
};

// REQ: PARAM-055
HOLOSCAN_PARAM_SCHEMA(Params,
                      holoscan::param_schema("nvidia.holoscan.test.EmptyNamedEnumDomain", 1),
                      holoscan::field<&Params::mode>("mode").enum_values<Mode>({}));

constexpr auto kSchema = holoscan_param_schema(holoscan::detail::type_tag<Params>{});

int main() {}
