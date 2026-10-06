/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

namespace holoscan {

namespace nb = nanobind;

void init_errors(nb::module_& module);

[[noreturn]] void throw_consumed_loan_error();

}  // namespace holoscan
