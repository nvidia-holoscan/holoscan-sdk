/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>

#include <holoscan/authoring/register_payload.hpp>

// Supplies a process-local payload for AUTHORING_COMPILE_FAIL.local_enum_payload_macro.
enum class LocalEnum : std::uint32_t { kValue = 1 };

// Key setup: Instantiate HOLOSCAN_REGISTER_PAYLOAD with a process-local enum.
// Key assertion: Compilation fails at the ExternalTopicPayload constraint.
// REQ: AUTHOR-012
HOLOSCAN_REGISTER_PAYLOAD(LocalEnum, "local_enum");

int main() {
  return 0;
}
