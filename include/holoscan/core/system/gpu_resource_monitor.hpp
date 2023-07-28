/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef HOLOSCAN_CORE_SYSTEM_GPU_RESOURCE_MONITOR_HPP
#define HOLOSCAN_CORE_SYSTEM_GPU_RESOURCE_MONITOR_HPP

#include <memory>
#include <vector>

#include "cuda_runtime_wrapper.h"
#include "gpu_info.hpp"
#include "nvml_wrapper.h"

namespace holoscan {

constexpr uint64_t kDefaultGpuMetrics = GPUMetricFlag::GPU_DEVICE_ID;

/**
 * @brief GPUResourceMonitor class
 *
 * This class is responsible for monitoring the GPU resources.
 * It provides the information about the GPU resources (through `holoscan::GPUInfo`) to the
 * SystemResourceManager class.
 *
 * The following `holoscan::GPUMetricFlag` flags are supported:
 * - `DEFAULT`: Default GPU metrics (`GPU_DEVICE_ID`)
 * - `GPU_DEVICE_ID`: GPU device ID (name, pci, serial, uuid)
 * - `GPU_UTILIZATION`: GPU utilization (gpu_utilization, memory_utilization)
 * - `MEMORY_USAGE`: GPU memory usage (memory_total, memory_free, memory_used, memory_usage)
 * - `POWER_LIMIT`: GPU power limit (power_limit)
 * - `POWER_USAGE`: GPU power usage (power_usage)
 * - `TEMPERATURE`: GPU temperature (temperature)
 * - `ALL`: All GPU metrics above
 *
 * `index` information is always available.
 *
 * This uses the NVML library to get the GPU information.
 * If NVML library is not available (in case of iGPU), this class uses the CUDA Runtime API to get
 * the GPU information.
 *
 * The following information is not available when using the CUDA Runtime API:
 * - `GPU_DEVICE_ID`: `pci.pciDeviceId` and `pci.pciSubSystemId` are not available
 * - `GPU_UTILIZATION`: `gpu_utilization` and `memory_utilization` are not available
 * - `POWER_LIMIT`: `power_limit` is not available
 * - `POWER_USAGE`: `power_usage` is not available
 * - `TEMPERATURE`: `temperature` is not available
 *
 * Example:
 *
 * ```cpp
 * #include <holoscan/core/system/system_resource_manager.hpp>
 * #include <holoscan/logger/logger.hpp>
 *
 * ...
 *
 * holoscan::GPUResourceMonitor gpu_resource_monitor;
 * gpu_resource_monitor.update(holoscan::GPUMetricFlag::ALL);
 * auto gpu_info = gpu_resource_monitor.gpu_info();
 * auto gpu_count = gpu_resource_monitor.num_gpus();
 *
 * for (int i = 0; i < gpu_count; i++) {
 *   // Print GPU information (GPUInfo)
 *   HOLOSCAN_LOG_INFO("GPU {} is available", gpu_info[i].index);
 *   HOLOSCAN_LOG_INFO("GPU {} name: {}", i, gpu_info[i].name);
 *   HOLOSCAN_LOG_INFO("GPU {} is iGPU: {}", i, gpu_info[i].is_integrated);
 *   HOLOSCAN_LOG_INFO("GPU {} pci.busId: {}", i, gpu_info[i].pci.busId);
 *   HOLOSCAN_LOG_INFO("GPU {} pci.busIdLegacy: {}", i, gpu_info[i].pci.busIdLegacy);
 *   HOLOSCAN_LOG_INFO("GPU {} pci.domain: {}", i, gpu_info[i].pci.domain);
 *   HOLOSCAN_LOG_INFO("GPU {} pci.bus: {}", i, gpu_info[i].pci.bus);
 *   HOLOSCAN_LOG_INFO("GPU {} pci.device: {}", i, gpu_info[i].pci.device);
 *   HOLOSCAN_LOG_INFO("GPU {} pci.pciDeviceId: {:x}:{:x}",
 *                     i,
 *                     gpu_info[i].pci.pciDeviceId & 0xffff,
 *                     gpu_info[i].pci.pciDeviceId >> 16);
 *   HOLOSCAN_LOG_INFO("GPU {} pci.pciSubSystemId: {:x}:{:x}",
 *                     i,
 *                     gpu_info[i].pci.pciSubSystemId & 0xffff,
 *                     gpu_info[i].pci.pciSubSystemId >> 16);
 *   HOLOSCAN_LOG_INFO("GPU {} serial: {}", i, gpu_info[i].serial);
 *   HOLOSCAN_LOG_INFO("GPU {} uuid: {}", i, gpu_info[i].uuid);
 *   HOLOSCAN_LOG_INFO("GPU {} gpu_utilization: {}", i, gpu_info[i].gpu_utilization);
 *   HOLOSCAN_LOG_INFO("GPU {} memory_utilization: {}", i, gpu_info[i].memory_utilization);
 *   HOLOSCAN_LOG_INFO("GPU {} memory_total: {}", i, gpu_info[i].memory_total);
 *   HOLOSCAN_LOG_INFO("GPU {} memory_free: {}", i, gpu_info[i].memory_free);
 *   HOLOSCAN_LOG_INFO("GPU {} memory_used: {}", i, gpu_info[i].memory_used);
 *   HOLOSCAN_LOG_INFO("GPU {} memory_usage: {}", i, gpu_info[i].memory_usage);
 *   HOLOSCAN_LOG_INFO("GPU {} power_limit: {}", i, gpu_info[i].power_limit);
 *   HOLOSCAN_LOG_INFO("GPU {} power_usage: {}", i, gpu_info[i].power_usage);
 *   HOLOSCAN_LOG_INFO("GPU {} temperature: {}", i, gpu_info[i].temperature);
 * }
 */
class GPUResourceMonitor {
 public:
  /**
   * @brief Construct a new GPUResourceMonitor object.
   *
   * This constructor creates a new GPUResourceMonitor object.
   *
   * @param metric_flags The metric flags (default: `GPU_DEVICE_ID`)
   */
  explicit GPUResourceMonitor(uint64_t metric_flags = kDefaultGpuMetrics);
  virtual ~GPUResourceMonitor();

  /**
   * @brief Initialize the GPU resource monitor.
   */
  void init();

  /**
   * @brief Close handle of the GPU resource monitor.
   *
   * This function closes the handle of the opened NVML and CUDA Runtime libraries if they are open.
   */
  void close();

  /**
   * @brief Get metric flags.
   *
   * This function returns the metric flags.
   *
   * @return The metric flags.
   */
  uint64_t metric_flags() const;

  /**
   * @brief Set metric flags.
   *
   * This function sets the metric flags.
   *
   * @param metric_flags The metric flags
   */
  void metric_flags(uint64_t metric_flags);

  /**
   * @brief Update the GPU information and cache it.
   *
   * This function updates information for the GPU with the given index based on the given metric
   * flags and returns the GPU information. If the metric flags are not provided, the existing
   * metric flags are used. It also caches the GPU information.
   *
   * @param index The GPU index.
   * @param metric_flags The metric flags.
   * @return The GPU information.
   */
  GPUInfo update(uint32_t index, uint64_t metric_flags = GPUMetricFlag::DEFAULT);

  /**
   * @brief Update all GPU information and cache it.
   *
   * This function updates the information for all GPUs based on the given metric flags and returns
   * a vector of GPU information. If the metric flags are not provided, the existing metric flags
   * are used. It also caches the GPU information.
   *
   * @param metric_flags The metric flags.
   * @return The vector of GPU information.
   */
  std::vector<GPUInfo> update(uint64_t metric_flags = GPUMetricFlag::DEFAULT);

  /**
   * @brief Update the GPU information.
   *
   * This function fills the GPU information given as the argument based on the given metric flags
   * and returns the GPU information. If the metric flags are not provided, the existing metric
   * flags are used.
   *
   * @param index The GPU index.
   * @param gpu_info The GPU information.
   * @param metric_flags The metric flags.
   * @return The GPU information filled with the updated values (same as the argument).
   */
  GPUInfo& update(uint32_t index, GPUInfo& gpu_info,
                  uint64_t metric_flags = GPUMetricFlag::DEFAULT);

  /**
   * @brief Get the GPU information.
   *
   * This method returns the GPU information based on the given index.
   *
   * If the metric flags are provided, it returns the vector of GPU information based on the given
   * metric flags. If the metric flags are not provided, it returns the cached GPU information.
   *
   * @param index The GPU index.
   * @param metric_flags The metric flags.
   * @return The GPU information.
   */
  GPUInfo gpu_info(uint32_t index, uint64_t metric_flags = GPUMetricFlag::DEFAULT);

  /**
   * @brief Get all GPU information.
   *
   * This method returns the vector of GPU information.
   * If the metric flags are provided, it returns the GPU information based on the given
   * metric flags. If the metric flags are not provided, it returns the cached GPU information.
   *
   * @param metric_flags The metric flags.
   * @return All GPU information.
   */
  std::vector<GPUInfo> gpu_info(uint64_t metric_flags = GPUMetricFlag::DEFAULT);

  /**
   * @brief Get the number of GPUs.
   *
   * @return The number of GPUs.
   */
  uint32_t num_gpus() const;

  /**
   * @brief Check whether the GPU is integrated (iGPU)
   *
   * @return True if the GPU is integrated (iGPU), false otherwise.
   */
  bool is_integrated_gpu(uint32_t index);

 protected:
  bool bind_nvml_methods();
  bool bind_cuda_runtime_methods();

  void* handle_ = nullptr;       ///< The handle of the GPU resource monitor
  void* cuda_handle_ = nullptr;  ///< The handle of the CUDA Runtime library

  // NVML function pointers

  /// The function pointer to the nvmlErrorString function
  nvml::nvmlErrorString_t nvmlErrorString = nullptr;
  /// The function pointer to the nvmlInit function
  nvml::nvmlInit_t nvmlInit = nullptr;
  /// The function pointer to the nvmlDeviceGetCount function
  nvml::nvmlDeviceGetCount_t nvmlDeviceGetCount = nullptr;
  /// The function pointer to the nvmlDeviceGetHandleByIndex function
  nvml::nvmlDeviceGetHandleByIndex_t nvmlDeviceGetHandleByIndex = nullptr;
  /// The function pointer to the nvmlDeviceGetHandleByPciBusId function
  nvml::nvmlDeviceGetHandleByPciBusId_t nvmlDeviceGetHandleByPciBusId = nullptr;
  /// The function pointer to the nvmlDeviceGetHandleBySerial function
  nvml::nvmlDeviceGetHandleBySerial_t nvmlDeviceGetHandleBySerial = nullptr;
  /// The function pointer to the nvmlDeviceGetHandleByUUID function
  nvml::nvmlDeviceGetHandleByUUID_t nvmlDeviceGetHandleByUUID = nullptr;
  /// The function pointer to the nvmlDeviceGetName function
  nvml::nvmlDeviceGetName_t nvmlDeviceGetName = nullptr;
  /// The function pointer to the nvmlDeviceGetIndex function
  nvml::nvmlDeviceGetIndex_t nvmlDeviceGetIndex = nullptr;
  /// The function pointer to the nvmlDeviceGetPciInfo function
  nvml::nvmlDeviceGetPciInfo_t nvmlDeviceGetPciInfo = nullptr;
  /// The function pointer to the nvmlDeviceGetSerial function
  nvml::nvmlDeviceGetSerial_t nvmlDeviceGetSerial = nullptr;
  /// The function pointer to the nvmlDeviceGetUUID function
  nvml::nvmlDeviceGetUUID_t nvmlDeviceGetUUID = nullptr;
  /// The function pointer to the nvmlDeviceGetMemoryInfo function
  nvml::nvmlDeviceGetMemoryInfo_t nvmlDeviceGetMemoryInfo = nullptr;
  /// The function pointer to the nvmlDeviceGetUtilizationRates function
  nvml::nvmlDeviceGetUtilizationRates_t nvmlDeviceGetUtilizationRates = nullptr;
  /// The function pointer to the nvmlDeviceGetPowerManagementLimit function
  nvml::nvmlDeviceGetPowerManagementLimit_t nvmlDeviceGetPowerManagementLimit = nullptr;
  /// The function pointer to the nvmlDeviceGetPowerUsage function
  nvml::nvmlDeviceGetPowerUsage_t nvmlDeviceGetPowerUsage = nullptr;
  /// The function pointer to the nvmlDeviceGetTemperature function
  nvml::nvmlDeviceGetTemperature_t nvmlDeviceGetTemperature = nullptr;
  /// The function pointer to the nvmlShutdown function
  nvml::nvmlShutdown_t nvmlShutdown = nullptr;

  // CUDA Runtime function pointers

  /// The function pointer to the cudaGetErrorString function
  cuda::cudaGetErrorString_t cudaGetErrorString = nullptr;
  /// The function pointer to the cudaGetDeviceCount function
  cuda::cudaGetDeviceCount_t cudaGetDeviceCount = nullptr;
  /// The function pointer to the cudaGetDeviceProperties function
  cuda::cudaGetDeviceProperties_t cudaGetDeviceProperties = nullptr;
  /// The function pointer to the cudaDeviceGetPCIBusId function
  cuda::cudaDeviceGetPCIBusId_t cudaDeviceGetPCIBusId = nullptr;
  /// The function pointer to the cudaMemGetInfo function
  cuda::cudaMemGetInfo_t cudaMemGetInfo = nullptr;

  uint64_t metric_flags_ = kDefaultGpuMetrics;  ///< The metric flags
  bool is_cached_ = false;         ///< The flag to indicate whether the GPU information is cached
  uint32_t gpu_count_ = 0;         ///< The cached number of GPUs
  std::vector<GPUInfo> gpu_info_;  ///< The cached GPU information
  std::vector<nvml::nvmlDevice_t> nvml_devices_;  ///< The cached NVML devices
};
}  // namespace holoscan

#endif /* HOLOSCAN_CORE_SYSTEM_GPU_RESOURCE_MONITOR_HPP */
