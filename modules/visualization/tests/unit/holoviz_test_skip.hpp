// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Environment-skip sentinel shared by the GPU tests. HOLOVIZ_ENV_SKIP is for the
// "this machine cannot run these tests at all" cases (no adapter, no window system,
// no CUDA context). CTest classification is exit-code based (SKIP_RETURN_CODE 77 via
// holoviz_test_main.cpp, tests/unit/CMakeLists.txt): a binary whose every selected test
// skipped reports Skipped, a failure anywhere reports Failed, and a partially-skipped
// binary reports Passed. The [holoviz-skip] prefix stays in the log output so
// environment skips remain grep-able and distinguishable from feature-specific
// GTEST_SKIPs (a missing extension or test font).

#pragma once

#include <gtest/gtest.h>

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage) — must expand GTEST_SKIP in the caller.
#define HOLOVIZ_ENV_SKIP() GTEST_SKIP() << "[holoviz-skip] "
