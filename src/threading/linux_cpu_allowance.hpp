/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/threading/thread_placement.hpp>

namespace holoscan::detail {

// A read-only seam for procfs, cgroupfs and sysfs observations. The resolver does
// not cache permission: Runtime must still apply and verify the requested mask.
using CPUAllowanceFileReader = std::function<expected<std::string, Error>(const std::string&)>;

[[nodiscard]] expected<CPUAffinity, Error> read_linux_cpu_allowance(
    const CPUAllowanceFileReader& reader, std::uint64_t thread_id);
[[nodiscard]] expected<CPUAffinity, Error> read_linux_cpu_allowance();

// Kernel-isolated CPUs (isolcpus) from sysfs. The scheduler never balances an unpinned thread
// onto or off them. A missing or empty list means none are isolated.
[[nodiscard]] expected<CPUAffinity, Error> read_linux_isolated_cpus(
    const CPUAllowanceFileReader& reader);
[[nodiscard]] expected<CPUAffinity, Error> read_linux_isolated_cpus();

}  // namespace holoscan::detail
