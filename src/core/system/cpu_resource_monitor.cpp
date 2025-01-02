/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "holoscan/core/system/cpu_resource_monitor.hpp"

#include <hwloc.h>
#include <sys/statvfs.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>

#include "holoscan/logger/logger.hpp"

namespace holoscan {

// Static methods
static void get_proc_stats(uint64_t* stats) {
  // Get the total CPU time
  FILE* file = fopen("/proc/stat", "r");
  if (file == nullptr) {
    HOLOSCAN_LOG_ERROR("CPUResourceMonitor::get_proc_stats() - Failed to open /proc/stat");
    return;
  }

  // Read the total CPU time
  int matched = fscanf(file, "cpu %lu %lu %lu %lu", &stats[0], &stats[1], &stats[2], &stats[3]);
  if (matched != 4) {
    HOLOSCAN_LOG_ERROR("CPUResourceMonitor::get_proc_stats() - Failed to read /proc/stat");
  }
  fclose(file);
}

static void get_proc_meminfo(uint64_t* stats) {
  // Get the total CPU time
  std::unique_ptr<FILE, decltype(&std::fclose)> file(fopen("/proc/meminfo", "r"), &std::fclose);
  if (file == nullptr) {
    HOLOSCAN_LOG_ERROR("CPUResourceMonitor::get_proc_meminfo() - Failed to open /proc/meminfo");
    return;
  }

  // Read MemTotal, MemFree, MemAvailable from the file, using fgets() to read the whole line and
  // fscanf() to parse the values.
  //
  // Example of /proc/meminfo:
  //   MemTotal:       131829248 kB
  //   MemFree:        87604880 kB
  //   MemAvailable:   104760392 kB

  char line[256];
  for (int line_count = 1; line_count <= 3; ++line_count) {
    if (fgets(line, sizeof(line), file.get()) == nullptr) {
      HOLOSCAN_LOG_ERROR(
          "CPUResourceMonitor::get_proc_meminfo() - Failed to read /proc/meminfo line {}",
          line_count);
      return;
    }

    int matched = 0;
    switch (line_count) {
      case 1:
        matched = sscanf(line, "MemTotal: %lu kB", &stats[0]);
        break;
      case 2:
        matched = sscanf(line, "MemFree: %lu kB", &stats[1]);
        break;
      case 3:
        matched = sscanf(line, "MemAvailable: %lu kB", &stats[2]);
        break;
    }
    if (matched != 1) {
      HOLOSCAN_LOG_ERROR(
          "CPUResourceMonitor::get_proc_meminfo() - Failed to parse /proc/meminfo line {}: '{}'",
          line_count,
          line);
      return;
    }
  }
}

static bool get_folder_space_info(const char* folder_path, uint64_t* stats) {
  if (stats == nullptr || folder_path == nullptr) {
    HOLOSCAN_LOG_ERROR("CPUResourceMonitor::get_folder_space_info() - Invalid arguments");
    return false;
  }

  struct statvfs stat;
  if (statvfs(folder_path, &stat) != 0) {
    HOLOSCAN_LOG_ERROR(
        "CPUResourceMonitor::get_folder_space_info() - Failed to get space info for folder '{}'",
        folder_path);
    return false;
  }
  stats[0] = stat.f_blocks * stat.f_frsize;
  stats[1] = stat.f_bfree * stat.f_frsize;
  stats[2] = stat.f_bavail * stat.f_frsize;
  return true;
}

CPUResourceMonitor::CPUResourceMonitor(void* context, uint64_t metric_flags)
    : context_(context), metric_flags_(metric_flags) {}

uint64_t CPUResourceMonitor::metric_flags() const {
  return metric_flags_;
}

void CPUResourceMonitor::metric_flags(uint64_t metric_flags) {
  metric_flags_ = metric_flags;
}

CPUInfo CPUResourceMonitor::update(uint64_t metric_flags) {
  update(cpu_info_, metric_flags);
  is_cached_ = true;
  return cpu_info_;
}

CPUInfo& CPUResourceMonitor::update(CPUInfo& cpu_info, uint64_t metric_flags) {
  if (metric_flags == CPUMetricFlag::DEFAULT) { metric_flags = metric_flags_; }

  if (metric_flags & CPUMetricFlag::CORE_COUNT) {
    cpu_info.num_cores =
        hwloc_get_nbobjs_by_type(static_cast<hwloc_topology_t>(context_), HWLOC_OBJ_CORE);
  }

  if (metric_flags & CPUMetricFlag::CPU_COUNT) {
    cpu_info.num_cpus =
        hwloc_get_nbobjs_by_type(static_cast<hwloc_topology_t>(context_), HWLOC_OBJ_PU);
  }

  if (metric_flags & CPUMetricFlag::AVAILABLE_PROCESSOR_COUNT) {
    // https://linux.die.net/man/2/sched_getaffinity
    if (sched_getaffinity(0, sizeof(cpu_set_), &cpu_set_) == 0) {
      unsigned long count = CPU_COUNT(&cpu_set_);
      cpu_info.num_processors = count;
    }
  }

  if (metric_flags & CPUMetricFlag::CPU_USAGE) {
    // If the last total stats are not valid, then just we update the last total stats
    if (!is_last_total_stats_valid_) {
      get_proc_stats(last_total_stats_);
      is_last_total_stats_valid_ = true;
    } else {
      uint64_t current_total_stats[4] = {};
      get_proc_stats(current_total_stats);

      // Calculate the CPU usage
      uint64_t total_diff = 0;
      for (int i = 0; i < 3; i++) { total_diff += (current_total_stats[i] - last_total_stats_[i]); }

      uint64_t idle_diff = current_total_stats[3] - last_total_stats_[3];
      total_diff += idle_diff;
      if (idle_diff > 0 && total_diff > 0) {
        cpu_info.cpu_usage = static_cast<float>(1.0 - (static_cast<double>(idle_diff) /
                                                       static_cast<double>(total_diff))) *
                             100.0F;
      }

      // Update the last total stats
      memcpy(last_total_stats_, current_total_stats, sizeof(*last_total_stats_) * 4);
    }
  }

  if (metric_flags & CPUMetricFlag::MEMORY_USAGE) {
    // Get the memory information from /proc/meminfo
    uint64_t mem_info[3]{};
    get_proc_meminfo(mem_info);
    cpu_info.memory_total = mem_info[0] * 1024;
    cpu_info.memory_free = mem_info[1] * 1024;
    cpu_info.memory_available = mem_info[2] * 1024;
    double memory_total = static_cast<double>(mem_info[0]);
    if (memory_total <= 0) {
      throw std::runtime_error(
          fmt::format("Invalid cpu_info.memory_total value: {}", memory_total));
    }
    cpu_info.memory_usage =
        static_cast<float>(1.0 - (static_cast<double>(mem_info[2]) / memory_total)) * 100.0F;
  }

  if (metric_flags & CPUMetricFlag::SHARED_MEMORY_USAGE) {
    // Get the shared memory information from /dev/shm
    uint64_t shm_info[3]{};
    get_folder_space_info("/dev/shm", shm_info);
    cpu_info.shared_memory_total = shm_info[0];
    cpu_info.shared_memory_free = shm_info[1];
    cpu_info.shared_memory_available = shm_info[2];
    cpu_info.shared_memory_usage = static_cast<float>(1.0 - (static_cast<double>(shm_info[2]) /
                                                             static_cast<double>(shm_info[0]))) *
                                   100.0F;
  }

  return cpu_info;
}

CPUInfo CPUResourceMonitor::cpu_info(uint64_t metric_flags) {
  if (metric_flags == CPUMetricFlag::DEFAULT) {
    if (!is_cached_) { update(); }
    return cpu_info_;
  }

  // Create the CPU information
  CPUInfo cpu_info;
  update(cpu_info, metric_flags);
  return cpu_info;
}

cpu_set_t CPUResourceMonitor::cpu_set() const {
  return cpu_set_;
}

}  // namespace holoscan
