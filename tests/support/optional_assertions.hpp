/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <gtest/gtest.h>

/// Fail the current test unless an optional holds a value, naming the optional expression.
///
/// Unlike `ASSERT_TRUE(optional.has_value())`, the presence check stays visible to clang-tidy's
/// optional-access analysis, so later dereferences in the same scope are known to be checked.
/// Additional context can be streamed: `HOLOSCAN_ASSERT_HAS_VALUE(reserved) << "context";`.
/// Use it as a complete statement; like other unbraced `if` bodies, it must not precede an `else`.
#define HOLOSCAN_ASSERT_HAS_VALUE(optional) \
  if (!(optional).has_value())              \
  FAIL() << "Expected " #optional " to have a value. "
