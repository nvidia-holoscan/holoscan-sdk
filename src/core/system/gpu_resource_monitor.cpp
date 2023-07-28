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
#include "holoscan/core/system/gpu_resource_monitor.hpp"

#include <dlfcn.h>

#include <iostream>
#include <vector>

#include "holoscan/logger/logger.hpp"

#define HOLOSCAN_NVML_CALL(stmt)                                                           \
  ({                                                                                       \
    nvml::nvmlReturn_t _holoscan_nvml_err = -1;                                            \
    if (handle_ == nullptr) {                                                              \
      HOLOSCAN_LOG_ERROR(                                                                  \
          "NVML library not loaded but NVML method '{}' in line {} of file {} was called", \
          #stmt,                                                                           \
          __LINE__,                                                                        \
          __FILE__);                                                                       \
    } else {                                                                               \
      _holoscan_nvml_err = stmt;                                                           \
      if (_holoscan_nvml_err != 0) {                                                       \
        HOLOSCAN_LOG_ERROR("NVML call '{}' in line {} of file {} failed with '{}' ({})",   \
                           #stmt,                                                          \
                           __LINE__,                                                       \
                           __FILE__,                                                       \
                           nvmlErrorString(_holoscan_nvml_err),                            \
                           _holoscan_nvml_err);                                            \
      }                                                                                    \
    }                                                                                      \
    _holoscan_nvml_err;                                                                    \
  })

#define HOLOSCAN_NVML_CALL_RETURN(stmt)                               \
  {                                                                   \
    nvml::nvmlReturn_t _holoscan_nvml_err = HOLOSCAN_NVML_CALL(stmt); \
    if (_holoscan_nvml_err != 0) {                                    \
      close();                                                        \
      return;                                                         \
    }                                                                 \
  }

#define HOLOSCAN_NVML_CALL_RETURN_VALUE_MSG(stmt, return_value, ...)  \
  {                                                                   \
    nvml::nvmlReturn_t _holoscan_nvml_err = HOLOSCAN_NVML_CALL(stmt); \
    if (_holoscan_nvml_err != 0) {                                    \
      HOLOSCAN_LOG_ERROR(__VA_ARGS__);                                \
      close();                                                        \
      return return_value;                                            \
    }                                                                 \
  }

#define HOLOSCAN_CUDA_CALL(stmt)                                                                \
  ({                                                                                            \
    holoscan::cuda::cudaError_t _holoscan_cuda_err = -1;                                        \
    if (cuda_handle_ == nullptr) {                                                              \
      HOLOSCAN_LOG_ERROR(                                                                       \
          "CUDA Runtime API library not loaded but CUDA Runtime API method '{}' in line {} of " \
          "file {} was called",                                                                 \
          #stmt,                                                                                \
          __LINE__,                                                                             \
          __FILE__);                                                                            \
    } else {                                                                                    \
      _holoscan_cuda_err = stmt;                                                                \
      if (_holoscan_cuda_err != 0) {                                                            \
        HOLOSCAN_LOG_ERROR(                                                                     \
            "CUDA Runtime API call '{}' in line {} of file {} failed with '{}' ({})",           \
            #stmt,                                                                              \
            __LINE__,                                                                           \
            __FILE__,                                                                           \
            cudaGetErrorString(_holoscan_cuda_err),                                             \
            _holoscan_cuda_err);                                                                \
      }                                                                                         \
    }                                                                                           \
    _holoscan_cuda_err;                                                                         \
  })

#define HOLOSCAN_CUDA_CALL_RETURN(stmt)                                        \
  {                                                                            \
    holoscan::cuda::cudaError_t _holoscan_cuda_err = HOLOSCAN_CUDA_CALL(stmt); \
    if (_holoscan_cuda_err != 0) {                                             \
      close();                                                                 \
      return;                                                                  \
    }                                                                          \
  }

#define HOLOSCAN_CUDA_CALL_RETURN_VALUE_MSG(stmt, return_value, ...)           \
  {                                                                            \
    holoscan::cuda::cudaError_t _holoscan_cuda_err = HOLOSCAN_CUDA_CALL(stmt); \
    if (_holoscan_cuda_err != 0) {                                             \
      HOLOSCAN_LOG_ERROR(__VA_ARGS__);                                         \
      close();                                                                 \
      return return_value;                                                     \
    }                                                                          \
  }

namespace holoscan {

namespace {
constexpr char kDefaultNvmlLibraryPath[] = "libnvidia-ml.so.1";  // /usr/lib/x86_64-linux-gnu/
constexpr const char* kDefaultCudaRuntimeLibraryPaths[] = {
    "libcudart.so.11.0", "libcudart.so"};  // /usr/local/cuda/lib64/
}  // namespace

GPUResourceMonitor::GPUResourceMonitor(uint64_t metric_flags) : metric_flags_(metric_flags) {
  init();
}

GPUResourceMonitor::~GPUResourceMonitor() {
  close();
}

void GPUResourceMonitor::init() {
  // Close the previous session if any
  close();

  handle_ = dlopen(kDefaultNvmlLibraryPath, RTLD_NOW);
  const char* libcudart_path = nullptr;

  if (handle_ == nullptr) {
    HOLOSCAN_LOG_DEBUG(
        "Unable to load NVML shared library from '{}'. Trying to use CUDA Runtime API instead.",
        kDefaultNvmlLibraryPath);
    for (uint32_t i = 0; i < sizeof(kDefaultCudaRuntimeLibraryPaths) / sizeof(char*); ++i) {
      libcudart_path = kDefaultCudaRuntimeLibraryPaths[i];
      cuda_handle_ = dlopen(libcudart_path, RTLD_NOW);
      if (cuda_handle_ != nullptr) { break; }
    }
    if (cuda_handle_ == nullptr) {
      HOLOSCAN_LOG_WARN(
          "Unable to load CUDA Runtime API shared library from '{}'. GPU Information will not be "
          "available.",
          libcudart_path);
      return;
    }
    HOLOSCAN_LOG_DEBUG("CUDA Runtime API library loaded from '{}'", libcudart_path);
    bind_cuda_runtime_methods();
    int gpu_count = 0;
    HOLOSCAN_CUDA_CALL_RETURN(cudaGetDeviceCount(&gpu_count));
    gpu_count_ = gpu_count;
  } else {
    HOLOSCAN_LOG_DEBUG("NVML library loaded from '{}'", kDefaultNvmlLibraryPath);

    bind_nvml_methods();

    HOLOSCAN_NVML_CALL_RETURN(nvmlInit());

    // Get the GPU count and initialize the GPU info vector
    HOLOSCAN_NVML_CALL(nvmlDeviceGetCount(&gpu_count_));

    // Initialize nvml devices vector
    nvml_devices_.resize(gpu_count_, nullptr);
  }

  // Initialize the GPU info vector
  gpu_info_.resize(gpu_count_);
}

void GPUResourceMonitor::close() {
  if (handle_) {
    nvml::nvmlReturn_t result = nvmlShutdown();
    if (result != 0) { HOLOSCAN_LOG_ERROR("Could not shutdown NVML"); }
    dlclose(handle_);
    handle_ = nullptr;
    is_cached_ = false;
  }
  if (cuda_handle_) {
    dlclose(cuda_handle_);
    cuda_handle_ = nullptr;
    is_cached_ = false;
  }
}

uint64_t GPUResourceMonitor::metric_flags() const {
  return metric_flags_;
}

void GPUResourceMonitor::metric_flags(uint64_t metric_flags) {
  metric_flags_ = metric_flags;
}

GPUInfo GPUResourceMonitor::update(uint32_t index, uint64_t metric_flags) {
  update(index, gpu_info_[index], metric_flags);
  return gpu_info_[index];
}

std::vector<GPUInfo> GPUResourceMonitor::update(uint64_t metric_flags) {
  // Create the GPU information
  for (uint32_t index = 0; index < gpu_count_; ++index) {
    update(index, gpu_info_[index], metric_flags);
  }
  is_cached_ = true;
  return gpu_info_;
}

GPUInfo& GPUResourceMonitor::update(uint32_t index, GPUInfo& gpu_info, uint64_t metric_flags) {
  if (metric_flags == GPUMetricFlag::DEFAULT) { metric_flags = metric_flags_; }

  gpu_info.index = index;

  if (handle_) {
    // NVML API requires the device handle
    nvml::nvmlDevice_t& device = nvml_devices_[index];
    if (device == nullptr) {  // if not cached
      HOLOSCAN_NVML_CALL_RETURN_VALUE_MSG(nvmlDeviceGetHandleByIndex(index, &device),
                                          gpu_info,
                                          "Could not get the device handle for GPU {}",
                                          index);
    }

    if (metric_flags & GPUMetricFlag::GPU_DEVICE_ID) {
      // // No need to get index again
      // HOLOSCAN_NVML_CALL_RETURN_VALUE_MSG(nvmlDeviceGetIndex(device, &gpu_info.index),
      //                            gpu_info,
      //                            "Could not get the device index for GPU {}",
      //                            index);
      HOLOSCAN_NVML_CALL_RETURN_VALUE_MSG(
          nvmlDeviceGetName(device, gpu_info.name, NVML_DEVICE_NAME_BUFFER_SIZE),
          gpu_info,
          "Could not get the device name for GPU {}",
          index);

      HOLOSCAN_NVML_CALL_RETURN_VALUE_MSG(nvmlDeviceGetPciInfo(device, &gpu_info.pci),
                                          gpu_info,
                                          "Could not get the PCI info for GPU {}",
                                          index);

      HOLOSCAN_NVML_CALL_RETURN_VALUE_MSG(
          nvmlDeviceGetSerial(device, gpu_info.serial, NVML_DEVICE_SERIAL_BUFFER_SIZE),
          gpu_info,
          "Could not get the serial number for GPU {}",
          index);
      HOLOSCAN_NVML_CALL_RETURN_VALUE_MSG(
          nvmlDeviceGetUUID(device, gpu_info.uuid, NVML_DEVICE_UUID_BUFFER_SIZE),
          gpu_info,
          "Could not get the UUID for GPU {}",
          index);
    }

    if (metric_flags & GPUMetricFlag::GPU_UTILIZATION) {
      nvml::nvmlUtilization_t utilization;
      HOLOSCAN_NVML_CALL_RETURN_VALUE_MSG(nvmlDeviceGetUtilizationRates(device, &utilization),
                                          gpu_info,
                                          "Could not get the utilization rates for GPU {}",
                                          index);
      gpu_info.gpu_utilization = utilization.gpu;
      gpu_info.memory_utilization = utilization.memory;
    }

    if (metric_flags & GPUMetricFlag::MEMORY_USAGE) {
      nvml::nvmlMemory_t memory;
      HOLOSCAN_NVML_CALL_RETURN_VALUE_MSG(nvmlDeviceGetMemoryInfo(device, &memory),
                                          gpu_info,
                                          "Could not get the memory info for GPU {}",
                                          index);
      gpu_info.memory_total = memory.total;
      gpu_info.memory_free = memory.free;
      gpu_info.memory_used = memory.used;
      gpu_info.memory_usage = 100.0 * memory.used / memory.total;
    }

    if (metric_flags & GPUMetricFlag::POWER_LIMIT) {
      HOLOSCAN_NVML_CALL_RETURN_VALUE_MSG(
          nvmlDeviceGetPowerManagementLimit(device, &gpu_info.power_limit),
          gpu_info,
          "Could not get the power limit for GPU {}",
          index);
    }

    if (metric_flags & GPUMetricFlag::POWER_USAGE) {
      HOLOSCAN_NVML_CALL_RETURN_VALUE_MSG(nvmlDeviceGetPowerUsage(device, &gpu_info.power_usage),
                                          gpu_info,
                                          "Could not get the power usage for GPU {}",
                                          index);
    }

    if (metric_flags & GPUMetricFlag::TEMPERATURE) {
      HOLOSCAN_NVML_CALL_RETURN_VALUE_MSG(
          nvmlDeviceGetTemperature(device, nvml::NVML_TEMPERATURE_GPU, &gpu_info.temperature),
          gpu_info,
          "Could not get the temperature for GPU {}",
          index);
    }
  } else if (cuda_handle_) {
    if (metric_flags & GPUMetricFlag::GPU_DEVICE_ID) {
      cuda::cudaDeviceProp prop;

      HOLOSCAN_CUDA_CALL_RETURN_VALUE_MSG(cudaGetDeviceProperties(&prop, index),
                                          gpu_info,
                                          "Could not get the device "
                                          "properties for GPU {}",
                                          index);
      // Check if the device is integrated GPU sharing Host Memory
      if (prop.integrated) { gpu_info.is_integrated = true; }

      // prop.name has a length of 256 but copy only the first 63 characters
      // to gpu_info.name which has a length of 64
      std::strncpy(gpu_info.name, prop.name, NVML_DEVICE_NAME_BUFFER_SIZE - 1);
      gpu_info.name[NVML_DEVICE_NAME_BUFFER_SIZE - 1] = '\0';

      HOLOSCAN_CUDA_CALL_RETURN_VALUE_MSG(
          cudaDeviceGetPCIBusId(
              gpu_info.pci.busIdLegacy, NVML_DEVICE_PCI_BUS_ID_BUFFER_V2_SIZE, index),
          gpu_info,
          "Could not get the PCI bus ID for GPU {}",
          index);

      // Note:: Looks like gpu_info.busId is '0000'<busIdLegacy> but need to confirm
      auto busid = fmt::format("0000{}", gpu_info.pci.busIdLegacy);
      std::strncpy(gpu_info.pci.busId, busid.c_str(), NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE - 1);
      gpu_info.pci.busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE - 1] = '\0';

      auto& uuid_bytes = prop.uuid.bytes;
      // e.g., 'GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-c30b948a-0eae-a09e-00e4-b7094efa64b0)'
      auto uuid_str = fmt::format(
          "GPU-{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:"
          "02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
          static_cast<uint8_t>(uuid_bytes[0]),
          static_cast<uint8_t>(uuid_bytes[1]),
          static_cast<uint8_t>(uuid_bytes[2]),
          static_cast<uint8_t>(uuid_bytes[3]),
          static_cast<uint8_t>(uuid_bytes[4]),
          static_cast<uint8_t>(uuid_bytes[5]),
          static_cast<uint8_t>(uuid_bytes[6]),
          static_cast<uint8_t>(uuid_bytes[7]),
          static_cast<uint8_t>(uuid_bytes[8]),
          static_cast<uint8_t>(uuid_bytes[9]),
          static_cast<uint8_t>(uuid_bytes[10]),
          static_cast<uint8_t>(uuid_bytes[11]),
          static_cast<uint8_t>(uuid_bytes[12]),
          static_cast<uint8_t>(uuid_bytes[13]),
          static_cast<uint8_t>(uuid_bytes[14]),
          static_cast<uint8_t>(uuid_bytes[15]));
      strncpy(gpu_info.uuid, uuid_str.c_str(), NVML_DEVICE_UUID_BUFFER_SIZE - 1);
      gpu_info.uuid[NVML_DEVICE_UUID_BUFFER_SIZE - 1] = '\0';
      gpu_info.pci.domain = prop.pciDomainID;
      gpu_info.pci.bus = prop.pciBusID;
      gpu_info.pci.device = prop.pciDeviceID;

      // 'pciDeviceId' information is not exposed in cudaDeviceProp struct but can be obtained from
      // hwloc (using gpu_info.pci.busIdLegacy with lower case) e.g,
      //     HOLOSCAN_LOG_INFO("pci.busIdLegacy: {}", gpu_info.pci.busIdLegacy);
      //     HOLOSCAN_LOG_INFO("pci.pciDeviceId: {:x}:{:x}", gpu_info.pci.pciDeviceId & 0xffff,
      //     gpu_info.pci.pciDeviceId >> 16);
      //     // shows:
      //     //   pci.busIdLegacy: 0000:2D:00.0
      //     //   pci.pciDeviceId: 10de:2204
      // 'hwloc-ls --whole-io' shows:
      //    ...
      //    'PCI L#14 (busid=0000:2d:00.0 id=10de:2204 class=0300(VGA) link=31.51GB/s)'
      //    ...
      // Cannot find a way to get the pciSubSystemId yet.

      // gpu_info.pci.pciDeviceId = ;
      // gpu_info.pci.pciSubSystemId = ;
    }

    if (metric_flags & GPUMetricFlag::GPU_UTILIZATION) {
      // iGPU load: /sys/devices/platform/gpu.0/load

      // gpu_info.gpu_utilization = ;
      // gpu_info.memory_utilization = ;
    }

    if (metric_flags & GPUMetricFlag::MEMORY_USAGE) {
      // Use cudaMemGetInfo for getting memory usage
      size_t free, total;
      HOLOSCAN_CUDA_CALL_RETURN_VALUE_MSG(cudaMemGetInfo(&free, &total),
                                          gpu_info,
                                          "Could not get the memory info for GPU {}",
                                          index);
      gpu_info.memory_total = total;
      gpu_info.memory_free = free;
      gpu_info.memory_used = total - free;
      gpu_info.memory_usage = 100.0 * (total - free) / total;
    }

    if (metric_flags & GPUMetricFlag::POWER_LIMIT) {
      // gpu_info.power_limit = ;
    }

    if (metric_flags & GPUMetricFlag::POWER_USAGE) {
      // gpu_info.power_usage = ;
    }

    if (metric_flags & GPUMetricFlag::TEMPERATURE) {
      // On McCoy, the temperature can be obtained from the thermal zone

      // nvidia@tegra-ubuntu:~$ cat /sys/devices/virtual/thermal/thermal_zone1/type
      // GPU-therm
      // nvidia@tegra-ubuntu:~$ cat /sys/devices/virtual/thermal/thermal_zone1/temp
      // 39000  ==> 39.0 C
    }
  }

  return gpu_info;
}

GPUInfo GPUResourceMonitor::gpu_info(uint32_t index, uint64_t metric_flags) {
  if (metric_flags == GPUMetricFlag::DEFAULT) {
    if (!is_cached_) { update(metric_flags); }
    return gpu_info_[index];
  }

  // Create the GPU information
  GPUInfo gpu_info;
  update(index, gpu_info, metric_flags);
  return gpu_info;
}

std::vector<GPUInfo> GPUResourceMonitor::gpu_info(uint64_t metric_flags) {
  if (metric_flags == GPUMetricFlag::DEFAULT) {
    if (!is_cached_) { update(); }
    return gpu_info_;
  }

  // Create the GPU information
  std::vector<GPUInfo> gpu_info;
  for (uint32_t index = 0; index < gpu_count_; ++index) {
    GPUInfo gpu_item;
    update(index, gpu_item, metric_flags);
    gpu_info.push_back(gpu_item);
  }
  return gpu_info;
}

uint32_t GPUResourceMonitor::num_gpus() const {
  return gpu_count_;
}

bool GPUResourceMonitor::is_integrated_gpu(uint32_t index) {
  if (index >= gpu_count_) { return false; }

  return gpu_info_[index].is_integrated;
}

bool GPUResourceMonitor::bind_nvml_methods() {
  // for nvmlErrorString method
  nvmlErrorString = reinterpret_cast<nvml::nvmlErrorString_t>(dlsym(handle_, "nvmlErrorString"));
  // for nvmlInit_v2 method
  nvmlInit = reinterpret_cast<nvml::nvmlInit_t>(dlsym(handle_, "nvmlInit_v2"));
  // for nvmlDeviceGetCount_v2 method
  nvmlDeviceGetCount =
      reinterpret_cast<nvml::nvmlDeviceGetCount_t>(dlsym(handle_, "nvmlDeviceGetCount_v2"));
  // for nvmlDeviceGetHandleByIndex
  nvmlDeviceGetHandleByIndex = reinterpret_cast<nvml::nvmlDeviceGetHandleByIndex_t>(
      dlsym(handle_, "nvmlDeviceGetHandleByIndex"));
  // for nvmlDeviceGetHandleByPciBusId_v2
  nvmlDeviceGetHandleByPciBusId = reinterpret_cast<nvml::nvmlDeviceGetHandleByPciBusId_t>(
      dlsym(handle_, "nvmlDeviceGetHandleByPciBusId_v2"));
  // for nvmlDeviceGetHandleBySerial
  nvmlDeviceGetHandleBySerial = reinterpret_cast<nvml::nvmlDeviceGetHandleBySerial_t>(
      dlsym(handle_, "nvmlDeviceGetHandleBySerial"));
  // for nvmlDeviceGetHandleByUUID
  nvmlDeviceGetHandleByUUID = reinterpret_cast<nvml::nvmlDeviceGetHandleByUUID_t>(
      dlsym(handle_, "nvmlDeviceGetHandleByUUID"));
  // for nvmlDeviceGetName
  nvmlDeviceGetName =
      reinterpret_cast<nvml::nvmlDeviceGetName_t>(dlsym(handle_, "nvmlDeviceGetName"));
  // for nvmlDeviceGetIndex
  nvmlDeviceGetIndex =
      reinterpret_cast<nvml::nvmlDeviceGetIndex_t>(dlsym(handle_, "nvmlDeviceGetIndex"));
  // for nvmlDeviceGetPciInfo_v3
  nvmlDeviceGetPciInfo =
      reinterpret_cast<nvml::nvmlDeviceGetPciInfo_t>(dlsym(handle_, "nvmlDeviceGetPciInfo_v3"));
  // for nvmlDeviceGetSerial
  nvmlDeviceGetSerial =
      reinterpret_cast<nvml::nvmlDeviceGetSerial_t>(dlsym(handle_, "nvmlDeviceGetSerial"));
  // for nvmlDeviceGetUUID
  nvmlDeviceGetUUID =
      reinterpret_cast<nvml::nvmlDeviceGetUUID_t>(dlsym(handle_, "nvmlDeviceGetUUID"));
  // for nvmlDeviceGetMemoryInfo
  nvmlDeviceGetMemoryInfo =
      reinterpret_cast<nvml::nvmlDeviceGetMemoryInfo_t>(dlsym(handle_, "nvmlDeviceGetMemoryInfo"));
  // for nvmlDeviceGetUtilizationRates
  nvmlDeviceGetUtilizationRates = reinterpret_cast<nvml::nvmlDeviceGetUtilizationRates_t>(
      dlsym(handle_, "nvmlDeviceGetUtilizationRates"));
  // for nvmlDeviceGetPowerManagementLimit
  nvmlDeviceGetPowerManagementLimit = reinterpret_cast<nvml::nvmlDeviceGetPowerManagementLimit_t>(
      dlsym(handle_, "nvmlDeviceGetPowerManagementLimit"));
  // for nvmlDeviceGetPowerUsage
  nvmlDeviceGetPowerUsage =
      reinterpret_cast<nvml::nvmlDeviceGetPowerUsage_t>(dlsym(handle_, "nvmlDeviceGetPowerUsage"));
  // for nvmlDeviceGetTemperature>
  nvmlDeviceGetTemperature = reinterpret_cast<nvml::nvmlDeviceGetTemperature_t>(
      dlsym(handle_, "nvmlDeviceGetTemperature"));
  // for nvmlShutdown
  nvmlShutdown = reinterpret_cast<nvml::nvmlShutdown_t>(dlsym(handle_, "nvmlShutdown"));

  if (nvmlErrorString == nullptr || nvmlInit == nullptr || nvmlDeviceGetCount == nullptr ||
      nvmlDeviceGetHandleByIndex == nullptr || nvmlDeviceGetHandleByPciBusId == nullptr ||
      nvmlDeviceGetHandleBySerial == nullptr || nvmlDeviceGetHandleByUUID == nullptr ||
      nvmlDeviceGetName == nullptr || nvmlDeviceGetIndex == nullptr ||
      nvmlDeviceGetPciInfo == nullptr || nvmlDeviceGetSerial == nullptr ||
      nvmlDeviceGetUUID == nullptr || nvmlDeviceGetMemoryInfo == nullptr ||
      nvmlDeviceGetUtilizationRates == nullptr || nvmlDeviceGetPowerManagementLimit == nullptr ||
      nvmlDeviceGetPowerUsage == nullptr || nvmlDeviceGetTemperature == nullptr ||
      nvmlShutdown == nullptr) {
    HOLOSCAN_LOG_ERROR("Could not find NVML function(s)");
    close();
    return false;
  }
  return true;
}

bool GPUResourceMonitor::bind_cuda_runtime_methods() {
  // for cudaGetErrorString method
  cudaGetErrorString =
      reinterpret_cast<cuda::cudaGetErrorString_t>(dlsym(handle_, "cudaGetErrorString"));
  // for cudaGetDeviceCount method
  cudaGetDeviceCount =
      reinterpret_cast<cuda::cudaGetDeviceCount_t>(dlsym(handle_, "cudaGetDeviceCount"));
  // for cudaGetDeviceProperties method
  cudaGetDeviceProperties =
      reinterpret_cast<cuda::cudaGetDeviceProperties_t>(dlsym(handle_, "cudaGetDeviceProperties"));
  // for cudaDeviceGetPCIBusId method
  cudaDeviceGetPCIBusId =
      reinterpret_cast<cuda::cudaDeviceGetPCIBusId_t>(dlsym(handle_, "cudaDeviceGetPCIBusId"));
  // for cudaMemGetInfo method
  cudaMemGetInfo = reinterpret_cast<cuda::cudaMemGetInfo_t>(dlsym(handle_, "cudaMemGetInfo"));

  if (cudaGetErrorString == nullptr || cudaGetDeviceCount == nullptr ||
      cudaGetDeviceProperties == nullptr || cudaDeviceGetPCIBusId == nullptr ||
      cudaMemGetInfo == nullptr) {
    HOLOSCAN_LOG_ERROR("Could not find CUDA Runtime function(s)");
    close();
    return false;
  }

  return true;
}

}  // namespace holoscan
