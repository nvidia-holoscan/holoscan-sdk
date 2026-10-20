/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// REQ: GRAPH-001

#include <holoscan/holoscan.hpp>

using ForbiddenApplication = holoscan::Application;

int main() {
  ForbiddenApplication* application = nullptr;
  return application == nullptr ? 0 : 1;
}
