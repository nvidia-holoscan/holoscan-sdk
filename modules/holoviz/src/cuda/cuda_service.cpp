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

#include <memory>
#include <utility>

#include <holoscan/logger/logger.hpp>

namespace holoscan::viz {

static std::unique_ptr<CudaService> g_cuda_service;

void cuMemFreeAsyncHelper(const std::pair<CUdeviceptr, CUstream>& args) {
  cuMemFreeAsync(args.first, args.second);
}

struct CudaService::Impl {
  bool is_mgpu_ = false;
  CUdevice device_ = 0;
  uint32_t device_ordinal_ = 0;
  CUcontext cuda_context_ = nullptr;
};

CudaService::CudaService(const CUuuid& device_uuid) : impl_(new Impl) {
  CudaCheck(cuInit(0));
  int device_count = 0;
  CudaCheck(cuDeviceGetCount(&device_count));
  impl_->is_mgpu_ = device_count > 1;
  for (int i = 0; i < device_count; ++i) {
    CUdevice device;
    CudaCheck(cuDeviceGet(&device, i));
    CUuuid uuid;
    CudaCheck(cuDeviceGetUuid(&uuid, device));
    if (std::memcmp(uuid.bytes, device_uuid.bytes, sizeof(CUuuid)) == 0) {
      impl_->device_ = device;
      impl_->device_ordinal_ = i;
      CudaCheck(cuDevicePrimaryCtxRetain(&impl_->cuda_context_, impl_->device_));
      return;
    }
  }
  throw std::runtime_error("CUDA service can't find device UUID");
}

CudaService::CudaService(uint32_t device_ordinal) : impl_(new Impl) {
  CudaCheck(cuInit(0));
  CudaCheck(cuDeviceGet(&impl_->device_, device_ordinal));
  impl_->device_ordinal_ = device_ordinal;
  CudaCheck(cuDevicePrimaryCtxRetain(&impl_->cuda_context_, impl_->device_));
}

CudaService::~CudaService() {
  if (impl_->cuda_context_) {
    // avoid CudaCheck() here, the driver might already be uninitialized
    // when global variables are destroyed
    cuDevicePrimaryCtxRelease(impl_->device_);
  }
}

bool CudaService::IsMultiGPU() const {
  return impl_->is_mgpu_;
}

bool CudaService::IsMemOnDevice(CUdeviceptr device_ptr) const {
  // if not MGPU, memory is always on the same device
  if (!impl_->is_mgpu_) { return true; }
  // else check the memory location
  int mem_device_ordinal;
  CudaCheck(
      cuPointerGetAttribute(&mem_device_ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, device_ptr));
  return (mem_device_ordinal == impl_->device_ordinal_);
}

class CudaService::ScopedPushImpl {
 public:
  /**
   * @brief Construct a new scoped cuda context object
   *
   * @param cuda_context context to push
   */
  explicit ScopedPushImpl(CUcontext cuda_context) : cuda_context_(cuda_context) {
    // might be called from a different thread than the thread
    // which constructed CudaPrimaryContext, therefore call cuInit()
    CudaCheck(cuInit(0));
    CudaCheck(cuCtxPushCurrent(cuda_context_));
  }
  ScopedPushImpl() = delete;

  ~ScopedPushImpl() {
    try {
      CUcontext popped_context;
      CudaCheck(cuCtxPopCurrent(&popped_context));
      if (popped_context != cuda_context_) {
        HOLOSCAN_LOG_ERROR("Cuda: Unexpected context popped");
      }
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("ScopedPush destructor failed with {}", e.what());
    }
  }

 private:
  const CUcontext cuda_context_;
};

CudaService::ScopedPush CudaService::PushContext() {
  return std::make_shared<ScopedPushImpl>(impl_->cuda_context_);
}

CudaService::ScopedPush CudaService::PushContext(CUcontext context) {
  return std::make_shared<ScopedPushImpl>(context);
}

}  // namespace holoscan::viz
