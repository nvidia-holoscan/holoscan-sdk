/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>

#include <cstddef>
#include <cstdint>

namespace holomq {
namespace cuda {

/// Owns a \c cuMemHostRegister mapping; calls \c cuMemHostUnregister on destruction when
/// registration succeeded. If registration fails, the object is invalid (\c operator bool() is \c
/// false) and the destructor is a no-op.
///
/// The caller must keep the appropriate \c CUcontext current for \c cuMemHostRegister / \c
/// cuMemHostUnregister whenever this object may run those APIs.
class RegisteredHostMemory {
 public:
  RegisteredHostMemory() noexcept = default;

  RegisteredHostMemory(void* base, std::size_t size_bytes, unsigned int flags) noexcept;

  RegisteredHostMemory(const RegisteredHostMemory&) = delete;
  RegisteredHostMemory& operator=(const RegisteredHostMemory&) = delete;

  RegisteredHostMemory(RegisteredHostMemory&& other) noexcept;
  RegisteredHostMemory& operator=(RegisteredHostMemory&& other) noexcept;

  ~RegisteredHostMemory() noexcept;

  [[nodiscard]] explicit operator bool() const noexcept { return registered_; }

  [[nodiscard]] void* base() const noexcept { return base_; }

  [[nodiscard]] CUresult register_result() const noexcept { return register_result_; }

 private:
  void unregister_if_registered() noexcept;

  void* base_{nullptr};
  CUresult register_result_{CUDA_SUCCESS};
  bool registered_{false};
};

}  // namespace cuda
}  // namespace holomq
