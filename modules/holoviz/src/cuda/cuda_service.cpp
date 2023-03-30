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

#include "cuda_service.hpp"

#include <holoscan/logger/logger.hpp>

namespace holoscan::viz {

static std::unique_ptr<CudaService> g_cuda_service;

struct CudaService::Impl {
  CUdevice device_ = 0;
  CUcontext cuda_context_ = nullptr;
};

CudaService::CudaService() : impl_(new Impl) {
  /// @todo make configurable
  const uint32_t device_ordinal = 0;

  CudaCheck(cuInit(0));
  CudaCheck(cuDeviceGet(&impl_->device_, device_ordinal));
  CudaCheck(cuDevicePrimaryCtxRetain(&impl_->cuda_context_, impl_->device_));
}

CudaService::~CudaService() {
  if (impl_->cuda_context_) {
    // avoid CudaCheck() here, the driver might already be uninitialized
    // when global variables are destroyed
    cuDevicePrimaryCtxRelease(impl_->device_);
  }
}

CudaService& CudaService::get() {
  if (!g_cuda_service) { g_cuda_service.reset(new CudaService()); }
  return *g_cuda_service;
}

class CudaService::ScopedPushImpl {
 public:
  ScopedPushImpl() {
    // might be called from a different thread than the thread
    // which constructed CudaPrimaryContext, therefore call cuInit()
    CudaCheck(cuInit(0));
    CudaCheck(cuCtxPushCurrent(CudaService::get().impl_->cuda_context_));
  }

  ~ScopedPushImpl() {
    try {
      CUcontext popped_context;
      CudaCheck(cuCtxPopCurrent(&popped_context));
      if (popped_context != CudaService::get().impl_->cuda_context_) {
        HOLOSCAN_LOG_ERROR("Cuda: Unexpected context popped");
      }
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("ScopedPush destructor failed with {}", e.what());
    }
  }
};

CudaService::ScopedPush CudaService::PushContext() {
  return std::make_shared<ScopedPushImpl>();
}

}  // namespace holoscan::viz
