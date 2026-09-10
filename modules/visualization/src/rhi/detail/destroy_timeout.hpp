/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>

namespace holoviz::rhi::detail {

/// Timeout every destructor passes to destroy(): long enough to drain a healthy GPU,
/// short enough not to hang teardown on a wedged one. Destructors log kBackpressured
/// and leak rather than block forever. Explicit destroy() calls keep their documented
/// zero-timeout default (fail fast with kBackpressured when in use).
inline constexpr std::chrono::nanoseconds kDestructorDestroyTimeout = std::chrono::seconds{1};

}  // namespace holoviz::rhi::detail
