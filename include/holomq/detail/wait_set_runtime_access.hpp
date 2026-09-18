/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/expected.hpp>
#include <holomq/wait_set.hpp>
#include <holomq/wait_strategy.hpp>

namespace holomq::detail {

/// Private Runtime access to \ref WaitSet::wait_impl_assuming_no_ready.
///
/// The Runtime channel pump used this to skip the pre-block readiness scan after what it took to be
/// an exhaustive drain. That invariant does not hold (readiness can appear between the drain and
/// the block), so the entry point now performs the checked wait and has no callers. It is retained
/// for the exported ABI baseline until a change can absorb the export removal.
class WaitSetRuntimeAccess {
 public:
  [[nodiscard]] static auto wait_assuming_no_ready(WaitSet& wait_set,
                                                   const WaitStrategy& strategy) noexcept
      -> expected<WaitResult, Error> {
    return wait_set.wait_impl_assuming_no_ready(strategy);
  }
};

}  // namespace holomq::detail
