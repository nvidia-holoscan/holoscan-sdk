/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use it except in compliance with the License.
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

/**
 * @file cuda_ipc_backend.hpp
 *
 * Internal: IpcBackendInterface implementation using CUDA IPC memory handles (cudaIpcMemHandle_t).
 * Not part of the public API; only IpcCore and cuda_ipc_backend.cpp include this header.
 */

#ifndef HOLOSCAN_IPC_CUDAIPCBACKEND_HPP
#define HOLOSCAN_IPC_CUDAIPCBACKEND_HPP

#include <memory>

#include "ipc_backend_interface.hpp"

namespace holoscan {
namespace ipc {

/** Backend that exports and opens CUDA IPC memory handles. */
class CudaIpcBackend : public IpcBackendInterface {
 public:
  HandleType handle_type() const override;
  bool is_valid_handle(const HandleBytes& handle) const override;
  HandleBytes export_handle(const void* ptr) const override;
  void release_export(const HandleBytes& handle_bytes) const override;
  std::shared_ptr<void> open_handle(const HandleBytes& handle) const override;
};

}  // namespace ipc
}  // namespace holoscan

#endif  // HOLOSCAN_IPC_CUDAIPCBACKEND_HPP
