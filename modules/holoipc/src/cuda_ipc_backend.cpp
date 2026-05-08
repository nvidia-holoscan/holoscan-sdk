/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cuda_ipc_backend.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <memory>

#include <holoscan/ipc/log.hpp>

namespace {

constexpr std::size_t HANDLE_SIZE = sizeof(cudaIpcMemHandle_t);

}  // namespace

namespace holoscan {
namespace ipc {

HandleType CudaIpcBackend::handle_type() const {
  return HandleType::CUDA_IPC;
}

bool CudaIpcBackend::is_valid_handle(const IpcBackendInterface::HandleBytes& handle) const {
  return handle.size() == HANDLE_SIZE;
}

IpcBackendInterface::HandleBytes CudaIpcBackend::export_handle(const void* ptr) const {
  if (ptr == nullptr) {
    return {};
  }

  cudaIpcMemHandle_t ipc_handle{};
  // CUDA API takes non-const void* but does not modify the pointee.
  cudaError_t err = cudaIpcGetMemHandle(&ipc_handle, const_cast<void*>(ptr));
  if (err != cudaSuccess) {
    HOLOSCAN_IPC_LOG_ERROR(
        "{}: cudaIpcGetMemHandle failed: {}", __FUNCTION__, cudaGetErrorString(err));
    return {};
  }
  const auto* bytes = reinterpret_cast<const uint8_t*>(&ipc_handle);
  return HandleBytes(bytes, bytes + HANDLE_SIZE);
}

void CudaIpcBackend::release_export([[maybe_unused]] const HandleBytes& handle_bytes) const {
  // CUDA IPC does not allocate per-export resources on the publisher side; no-op.
}

std::shared_ptr<void> CudaIpcBackend::open_handle(
    const IpcBackendInterface::HandleBytes& handle) const {
  if (handle.size() != HANDLE_SIZE) {
    return {};
  }

  // Copy the handle bytes to a cudaIpcMemHandle_t.
  cudaIpcMemHandle_t ipc_handle{};
  std::copy_n(handle.data(), HANDLE_SIZE, reinterpret_cast<uint8_t*>(&ipc_handle));
  void* device_ptr = nullptr;
  cudaError_t err = cudaIpcOpenMemHandle(&device_ptr, ipc_handle, cudaIpcMemLazyEnablePeerAccess);
  if (err != cudaSuccess || device_ptr == nullptr) {
    const char* err_msg = (err != cudaSuccess) ? cudaGetErrorString(err) : "returned null";
    HOLOSCAN_IPC_LOG_ERROR("{}: cudaIpcOpenMemHandle failed: {}", __FUNCTION__, err_msg);
    return {};
  }
  return std::shared_ptr<void>(device_ptr, [](void* p) {
    cudaError_t err = cudaIpcCloseMemHandle(p);
    if (err != cudaSuccess) {
      HOLOSCAN_IPC_LOG_ERROR("CudaIpcBackend: cudaIpcCloseMemHandle failed: {}",
                             cudaGetErrorString(err));
    }
  });
}

}  // namespace ipc
}  // namespace holoscan
