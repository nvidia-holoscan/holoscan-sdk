/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/channel/runtime.hpp>

// REQ: CHAN-070
int main() {
  auto runtime = holoscan::ChannelRuntime::create();
  return runtime->transport_capabilities().typed_payloads ? 0 : 1;
}
