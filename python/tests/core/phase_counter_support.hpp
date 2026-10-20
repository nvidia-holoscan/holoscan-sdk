/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

namespace holoscan::python_test_support {

/// Register `_with_phase_counters(plan)` and `_PhaseCounters` on the test-support module.
void register_phase_counter_support(nanobind::module_& module);

}  // namespace holoscan::python_test_support
