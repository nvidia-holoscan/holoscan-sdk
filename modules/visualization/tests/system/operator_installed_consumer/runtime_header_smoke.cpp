/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <type_traits>

#include <holoscan/core/run.hpp>

static_assert(!std::is_default_constructible_v<holoscan::RunSession>);
static_assert(std::is_nothrow_move_constructible_v<holoscan::RunSession>);
static_assert(!std::is_copy_constructible_v<holoscan::RunSession>);
static_assert(sizeof(holoscan::RunSession) == sizeof(void*));

int runtime_header_smoke() {
  const auto run_sync = static_cast<int (*)(const holoscan::ExecutionPlan&)>(&holoscan::run);
  const auto run_async =
      static_cast<holoscan::RunSession (*)(const holoscan::ExecutionPlan&)>(&holoscan::run_async);
  const holoscan::CompileError documented_error{"installed runtime header closure"};
  return run_sync != nullptr && run_async != nullptr && documented_error.what() != nullptr ? 0 : 1;
}
