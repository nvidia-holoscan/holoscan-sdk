/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// REQ: GRAPH-001

#include <holoscan/holoscan.hpp>

using ForbiddenPipeline = holoscan::Pipeline;

int main() {
  ForbiddenPipeline* pipeline = nullptr;
  return pipeline == nullptr ? 0 : 1;
}
