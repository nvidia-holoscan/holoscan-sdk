/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/parameter_schema.hpp>

struct OpaqueParams {
  int value{};
};

// REQ: PARAM-002
constexpr auto kRef = holoscan::param_ref<&OpaqueParams::value>();

int main() {}
