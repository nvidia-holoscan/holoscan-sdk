/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_cuda_context_placement.hpp"

#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

#include <cuda_runtime_api.h>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

#include "../threading/linux_cpu_allowance.hpp"

namespace holoscan::detail {

std::vector<DeviceId> runtime_cuda_devices(const ExecutionPlanState& plan,
                                           const ProcessGroupId process_group) {
  std::vector<DeviceId> devices;
  for (const auto& resource : plan.tensor_device_resources) {
    if ((!process_group || resource.process_group == process_group) && resource.device.valid()) {
      devices.push_back(resource.device);
    }
  }
  for (const auto& stream : plan.cuda_compute_streams) {
    if ((!process_group || stream.process_group == process_group) && stream.device.valid()) {
      devices.push_back(stream.device);
    }
  }
  std::ranges::sort(devices, {}, &DeviceId::value);
  const auto [first, last] = std::ranges::unique(devices, {}, &DeviceId::value);
  devices.erase(first, last);
  return devices;
}

#ifdef __linux__
cpu_set_t without_isolated_cpus_unless_all(const cpu_set_t& allowed, const CPUAffinity& isolated) {
  cpu_set_t reduced = allowed;
  for (const std::uint32_t cpu : isolated.cores) {
    if (cpu < CPU_SETSIZE) {
      CPU_CLR(cpu, &reduced);
    }
  }
  return CPU_COUNT(&reduced) != 0 ? reduced : allowed;
}

namespace {

// The CPUs the helper runs on: the process's cgroup allowance, or the caller's own CPUs when the
// allowance cannot be proven, without the kernel-isolated ones unless only those remain. Empty
// only when no CPU set can be read.
[[nodiscard]] cpu_set_t helper_cpus() {
  cpu_set_t allowed{};
  if (const auto allowance = read_linux_cpu_allowance(); allowance && !allowance->cores.empty()) {
    for (const std::uint32_t cpu : allowance->cores) {
      if (cpu < CPU_SETSIZE) {
        CPU_SET(cpu, &allowed);
      }
    }
  } else if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
    CPU_ZERO(&allowed);
    return allowed;
  }
  if (const auto isolated = read_linux_isolated_cpus(); isolated) {
    return without_isolated_cpus_unless_all(allowed, *isolated);
  }
  return allowed;
}

}  // namespace
#endif

void create_cuda_contexts_off_isolated_cpus(const std::vector<DeviceId>& devices) noexcept {
#ifdef __linux__
  if (devices.empty()) {
    return;
  }
  try {
    const cpu_set_t allowed = helper_cpus();
    if (CPU_COUNT(&allowed) == 0) {
      return;
    }
    std::thread creator([&devices, &allowed]() noexcept {
      if (pthread_setaffinity_np(pthread_self(), sizeof(allowed), &allowed) != 0) {
        return;
      }
      for (const DeviceId device : devices) {
        if (cudaSetDevice(device.value) != cudaSuccess || cudaFree(nullptr) != cudaSuccess) {
          continue;
        }
        // CUDA starts the thread that runs host functions inside a context's first host-function
        // launch, on the launching thread's CPUs, so launch one here too. Never wait for it: CUDA
        // may run host functions one at a time, and one already running can wait for this
        // session. Destroying the stream with the launch pending is allowed.
        cudaStream_t stream{};
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
          continue;
        }
        static_cast<void>(cudaLaunchHostFunc(stream, [](void*) {}, nullptr));
        static_cast<void>(cudaStreamDestroy(stream));
      }
      static_cast<void>(cudaGetLastError());
    });
    creator.join();
  } catch (...) {
    // Without a helper thread the first real CUDA use creates the context, as before.
  }
#else
  static_cast<void>(devices);
#endif
}

}  // namespace holoscan::detail
