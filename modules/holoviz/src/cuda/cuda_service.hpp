/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>
#include <sstream>

#include "../util/unique_value.hpp"

namespace holoscan::viz {

/**
 * Cuda driver API error check helper
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
      buf << "Cuda driver error " << result << " (" << error_name << "): " << error_string; \
      throw std::runtime_error(buf.str().c_str());                                          \
    }                                                                                       \
  }

/**
 * UniqueValue's for a Cuda objects
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
/**@}*/

/// Global Cuda service class
class CudaService {
 private:
  class ScopedPushImpl;  ///< Internal class handling the lifetime of the pushed context.

 public:
  /**
   * Destroy the Cuda service object.
   */
  ~CudaService();

  /**
   * Get the global Cuda service, if there is no global Cuda service yet it will be created.
   *
   * @return global Cuda service
   */
  static CudaService& get();

  /**
   * Shutdown and cleanup the global Cuda service.
   */
  static void shutdown();

  /// RAII type object to pop the Cuda primiary context on destruction
  typedef std::shared_ptr<ScopedPushImpl> ScopedPush;

  /**
   * Push the primary Cuda context.
   *
   * @return ScopedPush   RAII type object to pop the Cuda primiary context on destruction.
   */
  ScopedPush PushContext();

 private:
  CudaService();

  struct Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace holoscan::viz

#endif /* HOLOVIZ_SRC_CUDA_CUDA_SERVICE_HPP */
