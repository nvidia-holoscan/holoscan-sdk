/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/expected.hpp>
#include <holomq/wait_set.hpp>
#include <holomq/wait_strategy.hpp>

namespace holomq::detail {

/// Private access for a Runtime loop that exhaustively drains every subscriber after each wait.
class WaitSetRuntimeAccess {
 public:
  [[nodiscard]] static auto wait_assuming_no_ready(WaitSet& wait_set,
                                                   const WaitStrategy& strategy) noexcept
      -> expected<WaitResult, Error> {
    return wait_set.wait_impl_assuming_no_ready(strategy);
  }
};

}  // namespace holomq::detail
