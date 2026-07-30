/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file ipc_backend_interface.hpp
 *
 * Internal: Interface for pointer-sharing backends (e.g. CUDA IPC). Implementations
 * are responsible for the handle format and lifecycle (export from raw pointer,
 * open from descriptor). IpcCore uses a registry of backends keyed by
 * holoscan::ipc::HandleType.
 */

#ifndef HOLOSCAN_IPC_IPCBACKENDINTERFACE_HPP
#define HOLOSCAN_IPC_IPCBACKENDINTERFACE_HPP

#include <memory>
#include <vector>

#include <holoscan/ipc/detail/control_message.hpp>
#include <holoscan/ipc/detail/pointer_descriptor.hpp>

namespace holoscan {
namespace ipc {

/**
 * @brief Abstract backend for exporting and opening shared-memory handles.
 *
 * Enables multiple pointer-sharing mechanisms (e.g. CUDA IPC) behind a single
 * API. Each implementation corresponds to one HandleType and defines the
 * opaque handle bytes stored in ipc::PointerDescriptor.handle.
 */
class IpcBackendInterface {
 public:
  /** Opaque handle bytes (e.g. for PointerDescriptor.handle). */
  using HandleBytes = std::vector<uint8_t>;

  virtual ~IpcBackendInterface() = default;

  /** HandleType this backend implements. */
  virtual HandleType handle_type() const = 0;

  /** Returns true if \a handle is valid for this backend (e.g. correct size/format). */
  virtual bool is_valid_handle(const HandleBytes& handle) const = 0;

  /**
   * Export a raw pointer to opaque handle bytes.
   * @param ptr Non-null device (or shared) pointer to export.
   * @return Handle bytes on success, or empty vector on failure (e.g. driver error).
   */
  virtual HandleBytes export_handle(const void* ptr) const = 0;

  /**
   * Release any resources associated with a previously exported handle.
   * Called if descriptor creation or registration fails after export_handle, or when
   * the last reference to the descriptor is released. Default no-op (e.g. CUDA IPC
   * does not allocate per-export resources on the publisher side).
   */
  virtual void release_export(const HandleBytes& handle_bytes) const = 0;

  /**
   * Open a handle from opaque bytes and return a shared_ptr whose deleter only closes the handle.
   * Called when a subscriber wants to map a descriptor locally. In strict acquire this runs after
   * ACK; in eager acquire it may run before ACK.
   * @param handle Opaque handle bytes (e.g. from PointerDescriptor.handle).
   * @return shared_ptr to the opened pointer, or empty on failure.
   */
  virtual std::shared_ptr<void> open_handle(const HandleBytes& handle) const = 0;
};

}  // namespace ipc
}  // namespace holoscan

#endif  // HOLOSCAN_IPC_IPCBACKENDINTERFACE_HPP
