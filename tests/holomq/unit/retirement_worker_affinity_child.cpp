/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "buffer_pool/sync/stream_release.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include <sched.h>
#include <unistd.h>

// The worker's expected set: the process's CPUs except kernel-isolated ones, or all of them when
// every allowed CPU is isolated.
[[nodiscard]] cpu_set_t expected_worker_cpus(const cpu_set_t& allowed) {
  cpu_set_t expected = allowed;
  std::string list;
  std::getline(std::ifstream{"/sys/devices/system/cpu/isolated"}, list);
  std::stringstream ranges{list};
  for (std::string range; std::getline(ranges, range, ',');) {
    if (range.empty()) {
      continue;
    }
    const auto dash = range.find('-');
    const int first = std::stoi(range.substr(0U, dash));
    const int last = dash == std::string::npos ? first : std::stoi(range.substr(dash + 1U));
    for (int cpu = first; cpu <= last && cpu < CPU_SETSIZE; ++cpu) {
      CPU_CLR(cpu, &expected);
    }
  }
  return CPU_COUNT(&expected) != 0 ? expected : allowed;
}

// REQ: THREAD-LINUX-012
int main() {
  cpu_set_t allowed{};
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
    return 1;
  }
  const cpu_set_t expected = expected_worker_cpus(allowed);
  if (CPU_COUNT(&allowed) < 2) {
    return 77;
  }
  cpu_set_t pinned{};
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(cpu, &allowed)) {
      CPU_SET(cpu, &pinned);
      break;
    }
  }
  if (sched_setaffinity(0, sizeof(pinned), &pinned) != 0) {
    return 1;
  }
  // Start the real worker from the pinned importer, with no CUDA work or device required.
  holomq::EndpointStream::drain_retirements();
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (std::chrono::steady_clock::now() < deadline) {
    for (const auto& task : std::filesystem::directory_iterator{"/proc/self/task"}) {
      std::string name;
      std::ifstream{task.path() / "comm"} >> name;
      if (name != "holomq-retire") {
        continue;
      }
      cpu_set_t worker{};
      const auto tid = static_cast<pid_t>(std::stoi(task.path().filename().string()));
      if (sched_getaffinity(tid, sizeof(worker), &worker) != 0 || !CPU_EQUAL(&expected, &worker)) {
        std::cerr << "release worker CPU set is not the process set without isolated CPUs\n";
        return 1;
      }
      return 0;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  std::cerr << "release worker did not start\n";
  return 1;
}
