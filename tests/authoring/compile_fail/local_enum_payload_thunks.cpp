/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>

#include <holoscan/authoring/register_payload.hpp>

// Supplies a process-local payload for AUTHORING_COMPILE_FAIL.local_enum_payload_thunks.
enum class LocalEnum : std::uint32_t { kValue = 1 };

// Key setup: Instantiate the payload thunk table with a process-local enum.
// Key assertion: Compilation fails at the ExternalTopicPayload constraint.
// REQ: AUTHOR-012
int main() {
  (void)sizeof(holoscan::authoring::detail::PayloadRegistrationThunks<LocalEnum>);
}
