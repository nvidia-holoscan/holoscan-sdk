/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <thread>
#include <vector>

#include <cuda_runtime_api.h>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

#include "compiler/execution_plan_state.hpp"
#include "runtime/runtime_cuda_context_placement.hpp"
#include "threading/linux_cpu_allowance.hpp"

namespace holoscan::detail {
namespace {

[[nodiscard]] std::vector<std::int32_t> ordinals(const std::vector<DeviceId>& devices) {
  std::vector<std::int32_t> result;
  for (const DeviceId device : devices) {
    result.push_back(device.value);
  }
  return result;
}

// Runtime creates a context only for devices this process group materializes Tensor resources or
// compute streams on, once per device; an unselected group takes every group's devices.
// REQ: THREAD-LINUX-012
TEST(RuntimeCudaContextPlacement, SelectsEachDeviceOfTheProcessGroupOnce) {
  ExecutionPlanState plan;
  plan.tensor_device_resources = {
      TensorDeviceResourcePlanRecord{.process_group = ProcessGroupId{1U}, .device = DeviceId{1}},
      TensorDeviceResourcePlanRecord{.process_group = ProcessGroupId{2U}, .device = DeviceId{3}},
      TensorDeviceResourcePlanRecord{.process_group = ProcessGroupId{1U}, .device = DeviceId{0}},
      TensorDeviceResourcePlanRecord{.process_group = ProcessGroupId{1U}, .device = DeviceId{1}},
  };
  plan.cuda_compute_streams = {
      CudaComputeStreamPlanRecord{.process_group = ProcessGroupId{1U}, .device = DeviceId{2}},
      CudaComputeStreamPlanRecord{.process_group = ProcessGroupId{1U}, .device = DeviceId{}},
  };

  EXPECT_EQ(ordinals(runtime_cuda_devices(plan, ProcessGroupId{1U})),
            (std::vector<std::int32_t>{0, 1, 2}));
  EXPECT_EQ(ordinals(runtime_cuda_devices(plan, ProcessGroupId{2U})),
            (std::vector<std::int32_t>{3}));
  EXPECT_EQ(ordinals(runtime_cuda_devices(plan, ProcessGroupId{})),
            (std::vector<std::int32_t>{0, 1, 2, 3}));
  EXPECT_TRUE(runtime_cuda_devices(ExecutionPlanState{}, ProcessGroupId{1U}).empty());
}

#ifdef __linux__
[[nodiscard]] cpu_set_t without_isolated_cpus(cpu_set_t cpus) {
  if (const auto isolated = read_linux_isolated_cpus(); isolated) {
    for (const std::uint32_t cpu : isolated->cores) {
      if (cpu < CPU_SETSIZE) {
        CPU_CLR(cpu, &cpus);
      }
    }
  }
  return cpus;
}

// The helper avoids isolated CPUs but must still run, on the unreduced set, when every allowed CPU
// is isolated; skipping it would leave context creation to a pinned Runtime thread.
// REQ: THREAD-LINUX-012
TEST(RuntimeCudaContextPlacement, HelperFallsBackToTheUnreducedSetWhenEveryCPUIsIsolated) {
  cpu_set_t allowed{};
  CPU_SET(2, &allowed);
  CPU_SET(3, &allowed);
  CPU_SET(10, &allowed);

  const cpu_set_t reduced = without_isolated_cpus_unless_all(allowed, CPUAffinity{2U, 3U});
  EXPECT_EQ(CPU_COUNT(&reduced), 1);
  EXPECT_TRUE(CPU_ISSET(10, &reduced));

  const cpu_set_t all_isolated =
      without_isolated_cpus_unless_all(allowed, CPUAffinity{2U, 3U, 10U});
  EXPECT_TRUE(CPU_EQUAL(&all_isolated, &allowed));

  const cpu_set_t none_isolated = without_isolated_cpus_unless_all(allowed, CPUAffinity{});
  EXPECT_TRUE(CPU_EQUAL(&none_isolated, &allowed));
}

// Runs in a fresh process: pins the caller to one CPU, creates the context, then reports where
// CUDA runs a host function launched from the pinned caller. 0 means off the pin and off every
// isolated CPU.
[[nodiscard]] int host_function_cpus_probe(const int pinned_cpu) {
  cpu_set_t pin{};
  CPU_SET(pinned_cpu, &pin);
  if (pthread_setaffinity_np(pthread_self(), sizeof(pin), &pin) != 0) {
    return 2;
  }
  create_cuda_contexts_off_isolated_cpus({DeviceId{0}});
  cpu_set_t ran_on{};
  cudaStream_t stream{};
  if (cudaSetDevice(0) != cudaSuccess ||
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
      cudaLaunchHostFunc(
          stream,
          [](void* data) {
            static_cast<void>(
                sched_getaffinity(0, sizeof(cpu_set_t), static_cast<cpu_set_t*>(data)));
          },
          &ran_on) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    return 3;
  }
  static_cast<void>(cudaStreamDestroy(stream));
  if (CPU_EQUAL(&ran_on, &pin)) {
    return 4;
  }
  const cpu_set_t off_isolated = without_isolated_cpus(ran_on);
  return CPU_EQUAL(&off_isolated, &ran_on) ? 0 : 5;
}
#endif

// CUDA starts its host-function thread on the CPUs of the thread whose launch first needs it. The
// Runtime starts it before pinning anything, so a later launch from a pinned thread does not drag
// it onto that thread's CPU.
// REQ: THREAD-LINUX-012
TEST(RuntimeCudaContextPlacementDeathTest, HostFunctionsRunOffAPinnedCallersCPU) {
#ifndef __linux__
  GTEST_SKIP() << "CPU placement of CUDA threads is Linux-only";
#else
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  cpu_set_t allowed{};
  ASSERT_EQ(sched_getaffinity(0, sizeof(allowed), &allowed), 0);
  const cpu_set_t candidates = without_isolated_cpus(allowed);
  if (CPU_COUNT(&candidates) < 2) {
    GTEST_SKIP() << "Two non-isolated CPUs are needed to tell a pinned caller from the helper";
  }
  int pinned_cpu = 0;
  while (!CPU_ISSET(pinned_cpu, &candidates)) {
    ++pinned_cpu;
  }
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(std::_Exit(host_function_cpus_probe(pinned_cpu)), testing::ExitedWithCode(0), "");
#endif
}

// CUDA may run host functions from independent streams one at a time. A session starting while an
// earlier host function is still running, possibly waiting for that session, must not wait for it.
// REQ: THREAD-LINUX-012
TEST(RuntimeCudaContextPlacement, ReturnsWhileAnotherHostFunctionIsRunning) {
  int device_count{};
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "No CUDA device is available";
  }
  struct Gate {
    std::atomic_bool entered{};
    std::atomic_bool release{};
  } gate;
  cudaStream_t stream{};
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaLaunchHostFunc(
                stream,
                [](void* data) {
                  auto& held = *static_cast<Gate*>(data);
                  held.entered.store(true);
                  while (!held.release.load()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                  }
                },
                &gate),
            cudaSuccess);
  const auto entered_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!gate.entered.load() && std::chrono::steady_clock::now() < entered_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  const bool entered = gate.entered.load();

  auto placement =
      std::async(std::launch::async, [] { create_cuda_contexts_off_isolated_cpus({DeviceId{0}}); });
  const bool returned =
      entered && placement.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
  gate.release.store(true);
  placement.wait();
  EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
  ASSERT_TRUE(entered) << "The blocking host function never started";
  EXPECT_TRUE(returned) << "CUDA placement waited for an unrelated running host function";
}

}  // namespace
}  // namespace holoscan::detail
