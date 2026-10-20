/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

#include <holoscan/core/execution_plan.hpp>
#include <holoscan/threading/thread_placement.hpp>

#include "../compiler/execution_plan_state.hpp"

namespace holoscan::detail {

/// The CUDA devices this process group's plan materializes Tensor resources or compute streams on,
/// in ascending order without duplicates. A zero (unselected) group takes every group's devices,
/// as materialization does.
[[nodiscard]] std::vector<DeviceId> runtime_cuda_devices(const ExecutionPlanState& plan,
                                                         ProcessGroupId process_group);

/// Create the CUDA primary context of each device, and CUDA's host-function thread for it, before
/// any Runtime thread is pinned.
///
/// CUDA starts its event-handler threads when a context is created, and the thread that runs host
/// functions inside the context's first host-function launch; each takes the CPU set of the thread
/// that caused it. One started from a lane or support thread pinned to a kernel-isolated CPU stays
/// there, and CUDA then delivers host functions late unless work is launched from that same CPU.
/// HoloMQ releases each imported buffer through a host function, so a saturated cross-process
/// Tensor channel stalls on them. This starts both from a short-lived thread on the process's
/// cgroup CPU allowance (or the caller's CPUs when that cannot be read) minus the isolated ones.
/// The launch is not waited for, so a host function that is already running, even one waiting
/// for this session, cannot delay startup. CUDA threads that already exist are left where they
/// are. When only isolated CPUs are allowed, the helper runs on all of them, before any thread is
/// pinned to one. This does nothing on platforms other than Linux; CUDA failures are left for the
/// first real use to report.
void create_cuda_contexts_off_isolated_cpus(const std::vector<DeviceId>& devices) noexcept;

#ifdef __linux__
/// \p allowed without the \p isolated CPUs, or \p allowed unchanged when that would leave no CPU.
[[nodiscard]] cpu_set_t without_isolated_cpus_unless_all(const cpu_set_t& allowed,
                                                         const CPUAffinity& isolated);
#endif

}  // namespace holoscan::detail
