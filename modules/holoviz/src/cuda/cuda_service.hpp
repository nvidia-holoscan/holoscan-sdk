/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOVIZ_SRC_CUDA_CUDA_SERVICE_HPP
#define HOLOVIZ_SRC_CUDA_CUDA_SERVICE_HPP

#include <cuda.h>
#include <cuda_runtime.h>

#include <memory>
#include <sstream>
#include <utility>

#include "../util/unique_value.hpp"

namespace holoscan::viz {

/**
 * CUDA runtime API error check helper
 */
#define CudaRTCheck(FUNC)                                                                   \
  {                                                                                         \
    const cudaError_t result = FUNC;                                                        \
    if (result != cudaSuccess) {                                                            \
      std::stringstream buf;                                                                \
      buf << "[" << __FILE__ << ":" << __LINE__ << "] CUDA driver error " << result << " (" \
          << cudaGetErrorName(result) << "): " << cudaGetErrorString(result);               \
      throw std::runtime_error(buf.str().c_str());                                          \
    }                                                                                       \
  }

/**
 * CUDA driver API error check helper
 */
#define CudaCheck(FUNC)                                                                     \
  {                                                                                         \
    const CUresult result = FUNC;                                                           \
    if (result != CUDA_SUCCESS) {                                                           \
      const char* error_name = "";                                                          \
      cuGetErrorName(result, &error_name);                                                  \
      const char* error_string = "";                                                        \
      cuGetErrorString(result, &error_string);                                              \
      std::stringstream buf;                                                                \
      buf << "[" << __FILE__ << ":" << __LINE__ << "] CUDA driver error " << result << " (" \
          << error_name << "): " << error_string;                                           \
      throw std::runtime_error(buf.str().c_str());                                          \
    }                                                                                       \
  }

/**
 * Helper function for CUDA memory allocated with cuMemAllocAsync. cuMemFreeAsync has two arguments
 * but UniqueValue support a single argument only. Therefore we pack the arguments in a std::pair.
 *
 * @param args passed to cuMemFreeAsync
 */
extern void cuMemFreeAsyncHelper(const std::pair<CUdeviceptr, CUstream>& args);

/**
 * UniqueValue's for a CUDA objects
 */
/**@{*/
using UniqueCUdeviceptr = UniqueValue<CUdeviceptr, decltype(&cuMemFree), &cuMemFree>;
using UniqueCUsurfObject =
    UniqueValue<CUsurfObject, decltype(&cuSurfObjectDestroy), &cuSurfObjectDestroy>;
using UniqueCUtexObject =
    UniqueValue<CUtexObject, decltype(&cuTexObjectDestroy), &cuTexObjectDestroy>;
using UniqueCUexternalMemory =
    UniqueValue<CUexternalMemory, decltype(&cuDestroyExternalMemory), &cuDestroyExternalMemory>;
using UniqueCUmipmappedArray =
    UniqueValue<CUmipmappedArray, decltype(&cuMipmappedArrayDestroy), &cuMipmappedArrayDestroy>;
using UniqueCUexternalSemaphore =
    UniqueValue<CUexternalSemaphore, decltype(&cuDestroyExternalSemaphore),
                &cuDestroyExternalSemaphore>;
using UniqueAsyncCUdeviceptr = UniqueValue<std::pair<CUdeviceptr, CUstream>,
                                           decltype(&cuMemFreeAsyncHelper), &cuMemFreeAsyncHelper>;
using UniqueCUevent = UniqueValue<CUevent, decltype(&cuEventDestroy), &cuEventDestroy>;
using UniqueCUstream = UniqueValue<CUstream, decltype(&cuStreamDestroy), &cuStreamDestroy>;
/**@}*/

/// Global CUDA service class
class CudaService {
 private:
  class ScopedPushImpl;  ///< Internal class handling the lifetime of the pushed context.

 public:
  /**
   * Construct a new CUDA service object using a device with the given UUID
   *
   * @param device_uuid device UUID
   */
  explicit CudaService(const CUuuid& device_uuid);

  /**
   * Construct a new CUDA service object using a device ordinal
   *
   * @param device_ordinal device ordinal
   */
  explicit CudaService(uint32_t device_ordinal);

  CudaService() = delete;

  /**
   * Destroy the CUDA service object.
   */
  ~CudaService();

  /**
   * @returns true if running on a multi GPU system
   */
  bool IsMultiGPU() const;

  /**
   * Check if the memory is allocated on the same device the CudaService had been created with
   *
   * @param device_ptr CUDA device memory to check
   *
   * @returns true if the memory is on the same device as the CudaService
   */
  bool IsMemOnDevice(CUdeviceptr device_ptr) const;

  /// RAII type object to pop a CUDA context on destruction
  typedef std::shared_ptr<ScopedPushImpl> ScopedPush;

  /**
   * Push the primary CUDA context.
   *
   * @return ScopedPush   RAII type object to pop the CUDA primiary context on destruction.
   */
  ScopedPush PushContext();

  /**
   * Push a CUDA context.
   *
   * @param context the context to push
   *
   * @return ScopedPush   RAII type object to pop the CUDA context on destruction.
   */
  ScopedPush PushContext(CUcontext context);

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace holoscan::viz

#endif /* HOLOVIZ_SRC_CUDA_CUDA_SERVICE_HPP */
