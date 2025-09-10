/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>

#include <chrono>
#include <string>
#include <thread>

#include <holoscan/core/system/system_resource_manager.hpp>
#include <holoscan/holoscan.hpp>

namespace holoscan {

static void wait_time(int ms) {
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  do {
    end = std::chrono::system_clock::now();
  } while (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() < ms);
}

TEST(SystemResourceManager, TestReportCPUResourceInfo) {
  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  holoscan::SystemResourceManager system_resource_manager;

  //////////////////////////////////////////////////////////////
  // Updating CPU information through the SystemResourceManager
  //////////////////////////////////////////////////////////////

  // For the first time, the CPU usage is 0
  auto cpuinfo = system_resource_manager.cpu_monitor()->update(holoscan::CPUMetricFlag::CPU_USAGE);
  EXPECT_EQ(cpuinfo.cpu_usage, 0);

  // Wait for 10ms
  wait_time(10);

  // For the second time, the CPU usage is not 0
  cpuinfo = system_resource_manager.cpu_monitor()->update(holoscan::CPUMetricFlag::CPU_USAGE);
  EXPECT_GE(cpuinfo.cpu_usage, 0);  // CPU usage can be 0 (for 0.9% CPU usage)

  // Check if memory usage and cpu count information is not available
  EXPECT_EQ(cpuinfo.num_cores, 0);
  EXPECT_EQ(cpuinfo.num_cpus, 0);
  EXPECT_EQ(cpuinfo.num_processors, 0);
  EXPECT_EQ(cpuinfo.memory_free, 0);
  EXPECT_EQ(cpuinfo.memory_total, 0);
  EXPECT_EQ(cpuinfo.memory_available, 0);
  EXPECT_EQ(cpuinfo.memory_usage, 0);
  EXPECT_EQ(cpuinfo.shared_memory_free, 0);
  EXPECT_EQ(cpuinfo.shared_memory_total, 0);
  EXPECT_EQ(cpuinfo.shared_memory_available, 0);

  // Update CPU count info
  cpuinfo = system_resource_manager.cpu_monitor()->update(holoscan::CPUMetricFlag::COUNT);

  // Check if the information is valid
  EXPECT_GT(cpuinfo.num_cores, 0);
  EXPECT_GT(cpuinfo.num_cpus, 0);
  EXPECT_GT(cpuinfo.num_processors, 0);

  // Update memory usage info
  cpuinfo = system_resource_manager.cpu_monitor()->update(holoscan::CPUMetricFlag::MEMORY_USAGE);

  // Check if the information is valid
  EXPECT_GT(cpuinfo.memory_free, 0);
  EXPECT_GT(cpuinfo.memory_total, 0);
  EXPECT_GT(cpuinfo.memory_available, 0);
  EXPECT_TRUE(cpuinfo.memory_usage > 0 && cpuinfo.memory_usage <= 100);

  // Update shared memory usage info
  cpuinfo =
      system_resource_manager.cpu_monitor()->update(holoscan::CPUMetricFlag::SHARED_MEMORY_USAGE);

  // Check if the information is valid
  EXPECT_GE(cpuinfo.shared_memory_free, 0);
  EXPECT_GE(cpuinfo.shared_memory_total, 64 * 1024 * 1024);  // >= 64MB
  EXPECT_GE(cpuinfo.shared_memory_available, 0);

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("[error]") == std::string::npos) << "=== LOG ===\n"
                                                               << log_output << "\n===========\n";
}

TEST(SystemResourceManager, TestReportGPUResourceInfo) {
  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  holoscan::SystemResourceManager system_resource_manager;

  //////////////////////////////////////////////////////////////
  // Updating GPU information through the SystemResourceManager
  //////////////////////////////////////////////////////////////

  // Check if the information is valid
  auto num_gpus = system_resource_manager.gpu_monitor()->num_gpus();
  EXPECT_GT(num_gpus, 0);

  auto gpuinfo_list = system_resource_manager.gpu_monitor()->update(holoscan::GPUMetricFlag::ALL);

  for (size_t i = 0; i < num_gpus; i++) {
    auto& gpuinfo = gpuinfo_list[i];
    // EXPECT_GE(gpuinfo.index, 0);
    EXPECT_NE(*gpuinfo.name, 0);
    EXPECT_NE(*gpuinfo.uuid, 0);
    EXPECT_NE(*gpuinfo.pci.busId, 0);
    EXPECT_NE(*gpuinfo.pci.busIdLegacy, 0);

    if (gpuinfo.is_integrated) {
      // For integrated GPUs (like Jetson Thor), memory values may be 0 or use system memory
      // NVML returns "Not Supported" for memory APIs on integrated GPUs
      EXPECT_GE(gpuinfo.memory_total, 0);  // Can be 0 for unsupported or use system memory
      EXPECT_GE(gpuinfo.memory_free, 0);
      EXPECT_GE(gpuinfo.memory_used, 0);
      EXPECT_GE(gpuinfo.memory_usage, 0.0F);
      // Note: Some integrated GPUs (like Jetson Thor) may still have PCI info available
      // The key indicator of integrated GPU is that memory/power APIs return "Not Supported"
      // We don't enforce PCI/serial being 0 for integrated GPUs with valid PCI info
      EXPECT_EQ(gpuinfo.gpu_utilization, 0);
      EXPECT_EQ(gpuinfo.memory_utilization, 0);
      EXPECT_EQ(gpuinfo.power_limit, 0);
      EXPECT_EQ(gpuinfo.power_usage, 0);
      EXPECT_EQ(gpuinfo.temperature, 0);
    } else {
      // For non-integrated GPUs, memory values should be available
      EXPECT_GT(gpuinfo.memory_total, 0);
      EXPECT_GT(gpuinfo.memory_free, 0);
      EXPECT_GT(gpuinfo.memory_used, 0);
      EXPECT_GT(gpuinfo.memory_usage, 0.0F);
      // PCI, serial, gpu_utilization, memory_utilization, power_limit, power_usage, temperature
      // are available
      // EXPECT_GE(gpuinfo.pci.domain, 0);
      // EXPECT_GE(gpuinfo.pci.bus, 0);
      // EXPECT_GE(gpuinfo.pci.device, 0);
      EXPECT_NE(gpuinfo.pci.pciDeviceId, 0);
      EXPECT_NE(gpuinfo.pci.pciSubSystemId, 0);
      EXPECT_NE(*gpuinfo.serial, 0);
      // EXPECT_GE(gpuinfo.gpu_utilization, 0);
      // EXPECT_GE(gpuinfo.memory_utilization, 0);
      EXPECT_NE(gpuinfo.power_limit, 0);
      EXPECT_NE(gpuinfo.power_usage, 0);
      EXPECT_NE(gpuinfo.temperature, 0);
    }
  }

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("[error]") == std::string::npos) << "=== LOG ===\n"
                                                               << log_output << "\n===========\n";
}

TEST(SystemResourceManager, TestGetCPUInfo) {
  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  //////////////////////////////////////////////////////////////
  // Updating CPU information standalone
  //////////////////////////////////////////////////////////////

  holoscan::Topology topology;
  topology.load();
  holoscan::CPUResourceMonitor cpu_resource_monitor(topology.context());
  holoscan::CPUInfo cpu_info = cpu_resource_monitor.cpu_info(holoscan::CPUMetricFlag::ALL);

  // Wait for 10ms
  wait_time(10);

  // Update CPU usage info
  cpu_resource_monitor.update(cpu_info, holoscan::CPUMetricFlag::CPU_USAGE);

  // Check if the information is valid
  EXPECT_GE(cpu_info.cpu_usage, 0);  // CPU usage can be 0 (for 0.9% CPU usage)
  EXPECT_GT(cpu_info.num_cores, 0);
  EXPECT_GT(cpu_info.num_cpus, 0);
  EXPECT_GT(cpu_info.num_processors, 0);
  EXPECT_GT(cpu_info.memory_free, 0);
  EXPECT_GT(cpu_info.memory_total, 0);
  EXPECT_GT(cpu_info.memory_available, 0);
  EXPECT_TRUE(cpu_info.memory_usage > 0 && cpu_info.memory_usage <= 100);

  // Check if cpu_set is valid
  cpu_set_t cpu_set = cpu_resource_monitor.cpu_set();
  int num_of_processors = 0;
  for (int i = 0; i < cpu_info.num_cpus; i++) {
    if (CPU_ISSET(i, &cpu_set)) {
      num_of_processors++;
    }
  }
  EXPECT_EQ(num_of_processors, cpu_info.num_processors);

  HOLOSCAN_LOG_INFO("CPU cores: {}", cpu_info.num_cores);
  HOLOSCAN_LOG_INFO("CPU cpus: {}", cpu_info.num_cpus);
  HOLOSCAN_LOG_INFO("CPU processors: {}", cpu_info.num_processors);
  HOLOSCAN_LOG_INFO("CPU usage: {}", cpu_info.cpu_usage);
  HOLOSCAN_LOG_INFO("CPU memory total: {}", cpu_info.memory_total);
  HOLOSCAN_LOG_INFO("CPU memory free: {}", cpu_info.memory_free);
  HOLOSCAN_LOG_INFO("CPU memory available: {}", cpu_info.memory_available);
  HOLOSCAN_LOG_INFO("CPU memory usage: {}", cpu_info.memory_usage);

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("[error]") == std::string::npos) << "=== LOG ===\n"
                                                               << log_output << "\n===========\n";
}

TEST(SystemResourceManager, TestGetGPUInfo) {
  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  //////////////////////////////////////////////////////////////
  // Updating GPU information standalone
  //////////////////////////////////////////////////////////////

  holoscan::GPUResourceMonitor gpu_resource_monitor;
  gpu_resource_monitor.update(holoscan::GPUMetricFlag::ALL);
  auto gpu_info = gpu_resource_monitor.gpu_info();
  auto gpu_count = gpu_resource_monitor.num_gpus();

  for (int i = 0; i < gpu_count; i++) {
    auto& gpuinfo = gpu_info[i];
    // EXPECT_GE(gpuinfo.index, 0);
    EXPECT_NE(*gpuinfo.name, 0);
    EXPECT_NE(*gpuinfo.uuid, 0);
    EXPECT_NE(*gpuinfo.pci.busId, 0);
    EXPECT_NE(*gpuinfo.pci.busIdLegacy, 0);

    if (gpuinfo.is_integrated) {
      // For integrated GPUs (like Jetson Thor), memory values may be 0 or use system memory
      // NVML returns "Not Supported" for memory APIs on integrated GPUs
      EXPECT_GE(gpuinfo.memory_total, 0);  // Can be 0 for unsupported or use system memory
      EXPECT_GE(gpuinfo.memory_free, 0);
      EXPECT_GE(gpuinfo.memory_used, 0);
      EXPECT_GE(gpuinfo.memory_usage, 0.0F);
      // Note: Some integrated GPUs (like Jetson Thor) may still have PCI info available
      // The key indicator of integrated GPU is that memory/power APIs return "Not Supported"
      // We don't enforce PCI/serial being 0 for integrated GPUs with valid PCI info
      EXPECT_EQ(gpuinfo.gpu_utilization, 0);
      EXPECT_EQ(gpuinfo.memory_utilization, 0);
      EXPECT_EQ(gpuinfo.power_limit, 0);
      EXPECT_EQ(gpuinfo.power_usage, 0);
      EXPECT_EQ(gpuinfo.temperature, 0);
    } else {
      // For non-integrated GPUs, memory values should be available
      EXPECT_GT(gpuinfo.memory_total, 0);
      EXPECT_GT(gpuinfo.memory_free, 0);
      EXPECT_GT(gpuinfo.memory_used, 0);
      EXPECT_GT(gpuinfo.memory_usage, 0.0F);
      // PCI, serial, gpu_utilization, memory_utilization, power_limit, power_usage, temperature
      // are available
      // EXPECT_GE(gpuinfo.pci.domain, 0);
      // EXPECT_GE(gpuinfo.pci.bus, 0);
      // EXPECT_GE(gpuinfo.pci.device, 0);
      EXPECT_NE(gpuinfo.pci.pciDeviceId, 0);
      EXPECT_NE(gpuinfo.pci.pciSubSystemId, 0);
      EXPECT_NE(*gpuinfo.serial, 0);
      // EXPECT_GE(gpuinfo.gpu_utilization, 0);
      // EXPECT_GE(gpuinfo.memory_utilization, 0);
      EXPECT_NE(gpuinfo.power_limit, 0);
      EXPECT_NE(gpuinfo.power_usage, 0);
      EXPECT_NE(gpuinfo.temperature, 0);
    }

    HOLOSCAN_LOG_INFO("GPU {} is available", gpu_info[i].index);
    HOLOSCAN_LOG_INFO("GPU {} name: {}", i, gpu_info[i].name);
    HOLOSCAN_LOG_INFO("GPU {} is iGPU: {}", i, gpu_info[i].is_integrated);
    HOLOSCAN_LOG_INFO("GPU {} pci.busId: {}", i, gpu_info[i].pci.busId);
    HOLOSCAN_LOG_INFO("GPU {} pci.busIdLegacy: {}", i, gpu_info[i].pci.busIdLegacy);
    HOLOSCAN_LOG_INFO("GPU {} pci.domain: {}", i, gpu_info[i].pci.domain);
    HOLOSCAN_LOG_INFO("GPU {} pci.bus: {}", i, gpu_info[i].pci.bus);
    HOLOSCAN_LOG_INFO("GPU {} pci.device: {}", i, gpu_info[i].pci.device);
    HOLOSCAN_LOG_INFO("GPU {} pci.pciDeviceId: {:x}:{:x}",
                      i,
                      gpu_info[i].pci.pciDeviceId & 0xffff,
                      gpu_info[i].pci.pciDeviceId >> 16);
    HOLOSCAN_LOG_INFO("GPU {} pci.pciSubSystemId: {:x}:{:x}",
                      i,
                      gpu_info[i].pci.pciSubSystemId & 0xffff,
                      gpu_info[i].pci.pciSubSystemId >> 16);
    HOLOSCAN_LOG_INFO("GPU {} serial: {}", i, gpu_info[i].serial);
    HOLOSCAN_LOG_INFO("GPU {} uuid: {}", i, gpu_info[i].uuid);
    HOLOSCAN_LOG_INFO("GPU {} gpu_utilization: {}", i, gpu_info[i].gpu_utilization);
    HOLOSCAN_LOG_INFO("GPU {} memory_utilization: {}", i, gpu_info[i].memory_utilization);
    HOLOSCAN_LOG_INFO("GPU {} memory_total: {}", i, gpu_info[i].memory_total);
    HOLOSCAN_LOG_INFO("GPU {} memory_free: {}", i, gpu_info[i].memory_free);
    HOLOSCAN_LOG_INFO("GPU {} memory_used: {}", i, gpu_info[i].memory_used);
    HOLOSCAN_LOG_INFO("GPU {} memory_usage: {}", i, gpu_info[i].memory_usage);
    HOLOSCAN_LOG_INFO("GPU {} power_limit: {}", i, gpu_info[i].power_limit);
    HOLOSCAN_LOG_INFO("GPU {} power_usage: {}", i, gpu_info[i].power_usage);
    HOLOSCAN_LOG_INFO("GPU {} temperature: {}", i, gpu_info[i].temperature);
  }

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("[error]") == std::string::npos) << "=== LOG ===\n"
                                                               << log_output << "\n===========\n";
}

}  // namespace holoscan
