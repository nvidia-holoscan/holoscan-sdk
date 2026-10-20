/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/channel/runtime.hpp>

// REQ: CHAN-072
int main() {
  holoscan::ChannelRuntimeOptions options{};
  return static_cast<int>(options.tensor_sidecar_record_capacity);
}
