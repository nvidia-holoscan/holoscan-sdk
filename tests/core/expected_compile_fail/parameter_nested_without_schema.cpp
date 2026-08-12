/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/parameter_schema.hpp>

struct UndeclaredNested {
  int value{};
};

struct Params {
  UndeclaredNested nested{};
};

// REQ: PARAM-003
HOLOSCAN_PARAM_SCHEMA(Params, holoscan::param_schema("nvidia.holoscan.test.MissingNested", 1),
                      holoscan::field<&Params::nested>("nested").nested<UndeclaredNested>());

int main() {}
