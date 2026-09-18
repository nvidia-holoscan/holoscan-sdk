/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/params/config.hpp>

// REQ: PARAM-038
struct ConstParams {
  const int value{};
};

HOLOSCAN_PARAM_SCHEMA(ConstParams, holoscan::param_schema("nvidia.holoscan.test.ConstCodec", 1),
                      holoscan::field<&ConstParams::value>("value"));

template <>
struct holoscan::config::ConfigCodec<ConstParams>
    : holoscan::params::DefaultedParamSchemaCodec<ConstParams> {};

int main() {}
