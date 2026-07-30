/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
